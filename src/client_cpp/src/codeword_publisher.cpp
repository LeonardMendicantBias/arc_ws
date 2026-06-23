#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <arc_interfaces/msg/code.hpp>
#include <arc_interfaces/msg/decoded_image.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int BITS_PER_CODEWORD = 13;   // 2^13 == 8192 vocab size
static constexpr int VOCAB_SIZE = 8192;
static constexpr float LOGIT_LAPLACE_EPS = 0.1f;

// Inverse of dall_e.utils.map_pixels: clamp((x - eps) / (1 - 2 eps), 0, 1).
static torch::Tensor unmap_pixels(torch::Tensor x)
{
  return torch::clamp(
    (x - LOGIT_LAPLACE_EPS) / (1.0f - 2.0f * LOGIT_LAPLACE_EPS), 0.0f, 1.0f);
}

class TRTLogger : public nvinfer1::ILogger
{
  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[TRT] %s\n", msg);
    }
  }
};

class CodeSubscriber : public rclcpp::Node
{
public:
  CodeSubscriber()
  : Node("code_subscriber"),
    img_w_(320), img_h_(240),
    h_prime_(img_h_ / 8), w_prime_(img_w_ / 8),
    n_codewords_(h_prime_ * w_prime_),
    device_id_(declare_parameter<int>("device", 0)),
    device_(torch::kCUDA, static_cast<torch::DeviceIndex>(device_id_))
  {
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    if (device_id_ < 0 || device_id_ >= n_devices) {
      throw std::runtime_error(
              "requested CUDA device " + std::to_string(device_id_) +
              " but only " + std::to_string(n_devices) + " visible");
    }
    // Bind this thread (constructor + ROS callback thread are the same under
    // rclcpp::spin) to the chosen device. The CUDAStreamGuard in the
    // callback re-pins inside the scope, but cudaStreamCreate / TRT
    // deserialize must see the right current device too.
    if (cudaSetDevice(device_id_) != cudaSuccess) {
      throw std::runtime_error("cudaSetDevice failed for device " +
              std::to_string(device_id_));
    }
    RCLCPP_INFO(get_logger(), "using CUDA device %d", device_id_);

    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      throw std::runtime_error("cudaStreamCreate failed");
    }

    std::string share_dir = ament_index_cpp::get_package_share_directory("server_cpp");
    std::string engine_name = declare_parameter<std::string>("engine", "dec_fp16.trt");
    load_engine(share_dir + "/checkpoints/" + engine_name);
    load_mask_token(share_dir + "/checkpoints/mask_token.bin");

    // Pre-allocated CUDA buffers whose pointers are handed directly to TRT.
    // Sized for the TRT engine's fixed input: 1 x V x H' x W'.
    input_buf_ = torch::zeros(
      {1, VOCAB_SIZE, h_prime_, w_prime_},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    // Output dtype must match what the engine writes. build_decoder.py
    // currently emits FP16 outputs (DataType::kHALF); allocating FP32 here
    // silently reinterprets pairs of FP16s as single FP32s and produces a
    // tiled / garbled reconstruction.
    auto out_dtype = engine_->getTensorDataType(output_name_.c_str());
    torch::Dtype out_torch_dtype;
    if (out_dtype == nvinfer1::DataType::kHALF) {
      out_torch_dtype = torch::kHalf;
    } else if (out_dtype == nvinfer1::DataType::kFLOAT) {
      out_torch_dtype = torch::kFloat32;
    } else {
      throw std::runtime_error(
              "decoder engine output dtype must be FP16 or FP32");
    }
    output_buf_ = torch::empty(
      {1, 6, img_h_, img_w_},
      torch::TensorOptions().dtype(out_torch_dtype).device(device_));

    if (input_name_.empty() || output_name_.empty()) {
      throw std::runtime_error(
              "TRT engine must have exactly one input and one output tensor");
    }
    auto in_dims  = engine_->getTensorShape(input_name_.c_str());
    auto out_dims = engine_->getTensorShape(output_name_.c_str());
    if (in_dims.d[1] != VOCAB_SIZE ||
      in_dims.d[2] != h_prime_ || in_dims.d[3] != w_prime_)
    {
      throw std::runtime_error(
              "decoder engine input dims [V,H',W'] don't match expected " +
              std::to_string(VOCAB_SIZE) + "x" +
              std::to_string(h_prime_) + "x" + std::to_string(w_prime_));
    }
    if (out_dims.d[2] != img_h_ || out_dims.d[3] != img_w_) {
      throw std::runtime_error("decoder engine output spatial dims don't match "
              + std::to_string(img_h_) + "x" + std::to_string(img_w_));
    }

    // TRT 10 tensor-address API: set once on the context, then enqueueV3.
    if (!context_->setTensorAddress(input_name_.c_str(),
            input_buf_.data_ptr<float>()))
    {
      throw std::runtime_error("setTensorAddress failed for input");
    }
    if (!context_->setTensorAddress(output_name_.c_str(),
            output_buf_.data_ptr()))
    {
      throw std::runtime_error("setTensorAddress failed for output");
    }

    // Pinned staging buffers for the per-message decomposition of mask →
    // (transmitted codeword, transmitted position) pairs. Both sized to the
    // worst case n_codewords_ (= every position transmitted).
    pinned_tx_codes_ = torch::empty(
      {n_codewords_},
      torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
    pinned_tx_pos_ = torch::empty(
      {n_codewords_},
      torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));

    // Pre-allocated GPU ones buffer used as the source for index_put_ when
    // setting the one-hot 1.0 at (codeword, position). Slice down to the
    // actual n_selected per message.
    ones_full_ = torch::ones(
      {n_codewords_},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

    code_sub_ = create_subscription<arc_interfaces::msg::Code>(
      "/camera/camera/color/code", 1,
      std::bind(&CodeSubscriber::code_callback, this, std::placeholders::_1));
    rec_pub_ = create_publisher<sensor_msgs::msg::Image>(
      "/camera/camera/color/reconstructed", 1);
    my_rec_pub_ = create_publisher<arc_interfaces::msg::DecodedImage>(
      "/camera/camera/color/recon", 1);
  }

  ~CodeSubscriber()
  {
    if (context_) {delete context_;}
    if (engine_) {delete engine_;}
    if (runtime_) {delete runtime_;}
    if (stream_) {cudaStreamDestroy(stream_);}
  }

private:
  void load_engine(const std::string & path)
  {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
      throw std::runtime_error("Cannot open TRT engine: " + path);
    }
    std::streamsize size = file.tellg();
    file.seekg(0);
    std::vector<char> data(size);
    file.read(data.data(), size);

    runtime_ = nvinfer1::createInferRuntime(logger_);
    engine_ = runtime_->deserializeCudaEngine(data.data(), size);
    if (!engine_) {
      throw std::runtime_error("Failed to deserialize TRT engine: " + path);
    }
    context_ = engine_->createExecutionContext();
    if (!context_) {
      throw std::runtime_error("Failed to create TRT execution context");
    }

    int nb = engine_->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
      const char * name = engine_->getIOTensorName(i);
      auto mode  = engine_->getTensorIOMode(name);
      auto dtype = engine_->getTensorDataType(name);
      auto dims  = engine_->getTensorShape(name);
      const bool is_input = (mode == nvinfer1::TensorIOMode::kINPUT);
      if (is_input) {input_name_ = name;}
      else          {output_name_ = name;}
      const char * dt_str =
        (dtype == nvinfer1::DataType::kFLOAT) ? "FP32" :
        (dtype == nvinfer1::DataType::kHALF)  ? "FP16" :
        (dtype == nvinfer1::DataType::kINT8)  ? "INT8" : "other";
      std::fprintf(stderr, "[TRT] tensor[%d] %-22s  %s  %s  dims=[%d,%d,%d,%d]\n",
        i, name,
        is_input ? "INPUT " : "OUTPUT",
        dt_str,
        dims.nbDims > 0 ? static_cast<int>(dims.d[0]) : -1,
        dims.nbDims > 1 ? static_cast<int>(dims.d[1]) : -1,
        dims.nbDims > 2 ? static_cast<int>(dims.d[2]) : -1,
        dims.nbDims > 3 ? static_cast<int>(dims.d[3]) : -1);
    }
  }

  // Loads softmax(mask_token) as a flat [V] float32 binary written by
  // build_decoder.py. This is the learned distribution the fine-tuned
  // decoder expects at non-transmitted spatial positions — replaces the
  // old encoder-on-uniform-gray per-position one-hot mask_codes.bin.
  void load_mask_token(const std::string & path)
  {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
      throw std::runtime_error("Cannot open mask token: " + path);
    }
    std::streamsize sz = f.tellg();
    const std::streamsize expected =
      static_cast<std::streamsize>(VOCAB_SIZE * sizeof(float));
    if (sz != expected) {
      throw std::runtime_error(
              "mask_token.bin size mismatch: got " + std::to_string(sz) +
              " bytes, expected " + std::to_string(expected));
    }
    f.seekg(0);
    std::vector<float> host(VOCAB_SIZE);
    f.read(reinterpret_cast<char *>(host.data()), sz);

    // Sanity check: distribution should sum to ~1.
    double sum = 0.0;
    for (float v : host) {sum += v;}
    if (std::abs(sum - 1.0) > 1e-2) {
      RCLCPP_WARN(get_logger(),
        "mask_token distribution sum=%.6f (expected ~1.0)", sum);
    }

    mask_token_dist_ = torch::from_blob(
      host.data(), {VOCAB_SIZE},
      torch::TensorOptions().dtype(torch::kFloat32)
    ).clone().to(device_);

    RCLCPP_INFO(get_logger(),
      "loaded mask_token distribution from %s (V=%d, sum=%.4f)",
      path.c_str(), VOCAB_SIZE, sum);
  }

  // Unpacks M * 13 bits MSB-first from packed bytes into M int64 codewords.
  // Mirrors the pack done in client_cpp/codeword_publisher.cpp; int64 is the
  // dtype required by LibTorch advanced indexing / one-hot.
  void unpack_codewords(const std::vector<uint8_t> & data, int n_selected,
                        int64_t * out)
  {
    int64_t bit_pos = 0;
    for (int i = 0; i < n_selected; ++i) {
      uint16_t cw = 0;
      for (int b = BITS_PER_CODEWORD - 1; b >= 0; --b) {
        uint8_t bit = (data[bit_pos / 8] >> (7 - (bit_pos % 8))) & 1u;
        cw |= static_cast<uint16_t>(bit) << b;
        ++bit_pos;
      }
      out[i] = static_cast<int64_t>(cw);
    }
  }

  void code_callback(const arc_interfaces::msg::Code::SharedPtr msg)
  {
    rclcpp::Time t0 = get_clock()->now();

    const int n_selected = static_cast<int>(msg->length);
    if (static_cast<int>(msg->mask.size()) != n_codewords_) {
      RCLCPP_ERROR(get_logger(),
        "mask length %zu != expected %d, dropping message",
        msg->mask.size(), n_codewords_);
      return;
    }

    // 1) On the CPU side, decompose the bitmask into the list of transmitted
    //    spatial positions, and unpack the M codeword indices that the
    //    publisher sent for those positions. Stage both into pinned int64
    //    host buffers so the H2D copies can overlap with TRT enqueue.
    int64_t * pos_dst  = pinned_tx_pos_.data_ptr<int64_t>();
    int64_t * code_dst = pinned_tx_codes_.data_ptr<int64_t>();

    int sel_idx = 0;
    for (int i = 0; i < n_codewords_; ++i) {
      if (msg->mask[i]) {
        if (sel_idx >= n_selected) {
          RCLCPP_ERROR(get_logger(),
            "mask has more set bits than msg.length=%d", n_selected);
          return;
        }
        pos_dst[sel_idx++] = static_cast<int64_t>(i);
      }
    }
    if (sel_idx != n_selected) {
      RCLCPP_WARN(get_logger(),
        "mask set bits (%d) != msg.length (%d)", sel_idx, n_selected);
      return;
    }
    unpack_codewords(msg->data, n_selected, code_dst);

    // 2) All LibTorch CUDA ops below go on stream_, ordered with TRT enqueue.
    at::cuda::CUDAStreamGuard guard(
      at::cuda::getStreamFromExternal(stream_, device_.index()));

    auto tx_codes_gpu = pinned_tx_codes_.slice(0, 0, n_selected)
                          .to(device_, /*non_blocking=*/true);
    auto tx_pos_gpu   = pinned_tx_pos_.slice(0, 0, n_selected)
                          .to(device_, /*non_blocking=*/true);

    // 3) Build the (V, N) decoder input.
    //    Non-transmitted columns hold the learned MASK_TOKEN distribution;
    //    transmitted columns hold a hard one-hot at the received codeword.
    auto flat = input_buf_.view({VOCAB_SIZE, n_codewords_});

    // (a) Broadcast mask_token distribution into every column. This sets
    //     non-transmitted columns to their final value and gives transmitted
    //     columns a placeholder that we'll overwrite next.
    flat.copy_(mask_token_dist_.view({VOCAB_SIZE, 1})
                 .expand({VOCAB_SIZE, n_codewords_}));

    // (b) Zero transmitted columns, then place 1.0 at (codeword, position).
    flat.index_fill_(/*dim=*/1, tx_pos_gpu, 0.0);
    flat.index_put_({tx_codes_gpu, tx_pos_gpu},
                    ones_full_.slice(0, 0, n_selected));

    // 4) TRT inference (TRT 10: tensor addresses already set in the ctor).
    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("TRT enqueueV3 failed");
    }

    // 5) Post-process (still on stream_): sigmoid + unmap_pixels + to uint8.
    //    Cast to float32 because output_buf_ may be FP16 (matches engine).
    using torch::indexing::Slice;
    auto x_stats = output_buf_.index({Slice(), Slice(0, 3)})
                     .to(torch::kFloat32);                     // (1, 3, H, W)
    auto x_rec = unmap_pixels(torch::sigmoid(x_stats));        // (1, 3, H, W)
    auto img_uint8 = (x_rec * 255.0f).clamp(0.0f, 255.0f)
                       .to(torch::kUInt8)
                       .squeeze(0)
                       .permute({1, 2, 0})       // (H, W, 3)
                       .contiguous()
                       .to(torch::kCPU);

    cudaStreamSynchronize(stream_);

    // 6) Build and publish messages.
    const int h = img_h_, w = img_w_;
    const uint8_t * img_ptr = img_uint8.data_ptr<uint8_t>();
    std::vector<uint8_t> img_bytes(img_ptr, img_ptr + h * w * 3);

    sensor_msgs::msg::Image rec_msg;
    rec_msg.header.stamp = get_clock()->now();
    rec_msg.header.frame_id = msg->header.frame_id;
    rec_msg.height = static_cast<uint32_t>(h);
    rec_msg.width = static_cast<uint32_t>(w);
    rec_msg.encoding = "rgb8";
    rec_msg.is_bigendian = 0;
    rec_msg.step = static_cast<uint32_t>(w * 3);
    rec_msg.data = img_bytes;
    rec_pub_->publish(rec_msg);

    arc_interfaces::msg::DecodedImage my_msg;
    my_msg.header = rec_msg.header;
    my_msg.length = msg->length;
    my_msg.mask = msg->mask;
    my_msg.height = rec_msg.height;
    my_msg.width = rec_msg.width;
    my_msg.encoding = rec_msg.encoding;
    my_msg.is_bigendian = rec_msg.is_bigendian;
    my_msg.step = rec_msg.step;
    my_msg.data = std::move(img_bytes);
    my_rec_pub_->publish(my_msg);

    double latency_ms = (get_clock()->now() - t0).nanoseconds() / 1e6;
    RCLCPP_INFO(get_logger(),
      "reconstructed %dx%d  latency: %.4f ms", w, h, latency_ms);
  }

  const int img_w_, img_h_;
  const int h_prime_, w_prime_, n_codewords_;
  const int device_id_;
  torch::Device device_;

  TRTLogger logger_;
  nvinfer1::IRuntime * runtime_{nullptr};
  nvinfer1::ICudaEngine * engine_{nullptr};
  nvinfer1::IExecutionContext * context_{nullptr};
  cudaStream_t stream_{nullptr};
  std::string input_name_, output_name_;

  torch::Tensor input_buf_, output_buf_;
  torch::Tensor pinned_tx_codes_, pinned_tx_pos_;
  torch::Tensor ones_full_;
  torch::Tensor mask_token_dist_;

  rclcpp::Subscription<arc_interfaces::msg::Code>::SharedPtr code_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rec_pub_;
  rclcpp::Publisher<arc_interfaces::msg::DecodedImage>::SharedPtr my_rec_pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CodeSubscriber>());
  rclcpp::shutdown();
  return 0;
}
