#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <arc_interfaces/msg/code.hpp>
#include <arc_interfaces/msg/mask.hpp>
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

static constexpr int BITS_PER_CODEWORD = 13;  // 2^13 == 8192 vocab size

static torch::Tensor map_pixels(torch::Tensor x, float eps = 0.1f)
{
  return (1.0f - 2.0f * eps) * x + eps;
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

class CodePublisher : public rclcpp::Node
{
public:
  CodePublisher()
  : Node("code_publisher"),
    img_width_(640), img_height_(480),
    h_prime_(img_height_ / 8), w_prime_(img_width_ / 8),
    n_codewords_(h_prime_ * w_prime_),
    device_(torch::kCUDA)
  {
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      throw std::runtime_error("cudaStreamCreate failed");
    }

    std::string share_dir = ament_index_cpp::get_package_share_directory("client_cpp");
    load_engine(share_dir + "/checkpoints/enc.trt");

    // Pinned (page-locked) host buffer for async H2D of raw uint8 pixels.
    // 3 channels hardcoded — this is a color camera topic.
    pinned_input_ = torch::empty(
      {img_height_, img_width_, 3},
      torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true));

    // Pre-allocated CUDA buffers whose pointers are handed directly to TRT.
    input_buf_ = torch::empty(
      {1, 3, img_height_, img_width_},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    output_buf_ = torch::empty(
      {1, 8192, h_prime_, w_prime_},
      torch::TensorOptions().dtype(torch::kFloat32).device(device_));

    int input_idx = -1, output_idx = -1;
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
      if (engine_->bindingIsInput(i)) input_idx  = i;
      else                            output_idx = i;
    }
    if (input_idx < 0 || output_idx < 0) {
      throw std::runtime_error("TRT engine must have exactly one input and one output binding");
    }
    auto in_dims  = engine_->getBindingDimensions(input_idx);
    auto out_dims = engine_->getBindingDimensions(output_idx);
    if (in_dims.d[2] != img_height_ || in_dims.d[3] != img_width_) {
      throw std::runtime_error(
        "enc.trt was built for " +
        std::to_string(in_dims.d[3]) + "x" + std::to_string(in_dims.d[2]) +
        " but node expects " +
        std::to_string(img_width_) + "x" + std::to_string(img_height_));
    }
    // Sanity-check that h_prime/w_prime match the engine's output spatial dims.
    if (out_dims.d[2] != h_prime_ || out_dims.d[3] != w_prime_) {
      throw std::runtime_error("enc.trt output spatial dims don't match h_prime/w_prime");
    }

    bindings_[input_idx] = input_buf_.data_ptr<float>();
    bindings_[output_idx] = output_buf_.data_ptr<float>();

    img_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/camera/camera/color/image_raw", 1,
      std::bind(&CodePublisher::img_callback, this, std::placeholders::_1));
    mask_sub_ = create_subscription<arc_interfaces::msg::Mask>(
      "/camera/camera/color/mask", 1,
      std::bind(&CodePublisher::mask_callback, this, std::placeholders::_1));
    code_pub_ = create_publisher<arc_interfaces::msg::Code>(
      "/camera/camera/color/code", 1);

    // Default mask: transmit all codewords until a Mask message arrives.
    mask_indices_ = torch::arange(
      n_codewords_,
      torch::TensorOptions().dtype(torch::kInt64).device(device_));
  }

  ~CodePublisher()
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
    int nb = engine_->getNbBindings();
    bindings_.resize(static_cast<size_t>(nb));

    for (int i = 0; i < nb; ++i) {
      auto dtype = engine_->getBindingDataType(i);
      auto dims  = engine_->getBindingDimensions(i);
      const char * dt_str =
        (dtype == nvinfer1::DataType::kFLOAT) ? "FP32" :
        (dtype == nvinfer1::DataType::kHALF)  ? "FP16" : "other";
      std::fprintf(stderr, "[TRT] binding[%d] %-22s  %s  %s  dims=[%d,%d,%d,%d]\n",
        i, engine_->getBindingName(i),
        engine_->bindingIsInput(i) ? "INPUT " : "OUTPUT",
        dt_str,
        dims.nbDims > 0 ? dims.d[0] : -1,
        dims.nbDims > 1 ? dims.d[1] : -1,
        dims.nbDims > 2 ? dims.d[2] : -1,
        dims.nbDims > 3 ? dims.d[3] : -1);
    }
  }

  // Copies the ROS image into pinned memory, then asynchronously transfers
  // it to the GPU and preprocesses (HWC uint8 -> CHW float32 in [eps,1-eps])
  // entirely on stream_, writing directly into input_buf_.
  void load_image(const sensor_msgs::msg::Image::SharedPtr & msg)
  {
    if (static_cast<int>(msg->height) != img_height_ ||
        static_cast<int>(msg->width) != img_width_)
    {
      throw std::runtime_error(
        "unexpected image resolution: " +
        std::to_string(msg->width) + "x" + std::to_string(msg->height));
    }
    const int n_ch = static_cast<int>(msg->data.size()) /
      (static_cast<int>(msg->height) * static_cast<int>(msg->width));
    uint8_t * dst = pinned_input_.data_ptr<uint8_t>();
    const uint8_t * src = msg->data.data();

    if (n_ch == 3) {
      std::memcpy(dst, src, static_cast<size_t>(img_height_ * img_width_ * 3));
    } else {
      // Strip extra channels (e.g. RGBA -> RGB) on CPU before transfer.
      for (int i = 0; i < img_height_ * img_width_; ++i, dst += 3, src += n_ch) {
        dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
      }
    }

    // All LibTorch CUDA ops below go on stream_, ordered with TRT enqueue.
    at::cuda::CUDAStreamGuard guard(
      at::cuda::getStreamFromExternal(stream_, device_.index()));

    // Async H2D: 0.9 MB uint8 (vs 3.7 MB float without this optimization).
    auto gpu_uint8 = pinned_input_.to(device_, /*non_blocking=*/true);

    // GPU: (H,W,3) uint8 -> (1,3,H,W) float32 in [eps, 1-eps].
    input_buf_.copy_(
      map_pixels(
        gpu_uint8.unsqueeze(0).permute({0, 3, 1, 2})
        .to(torch::kFloat32).mul_(1.0f / 255.0f)));
  }

  // Submits input_buf_ to TRT on stream_, synchronizes, returns output_buf_.
  torch::Tensor trt_forward()
  {
    if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
      throw std::runtime_error("TRT enqueueV2 failed");
    }
    cudaStreamSynchronize(stream_);
    return output_buf_;
  }

  // Packs N codewords of 13 bits each (MSB-first) into a byte vector.
  std::vector<uint8_t> pack_codewords(const torch::Tensor & z_flat)
  {
    auto z_cpu = z_flat.to(torch::kCPU).to(torch::kInt32).contiguous();
    int64_t n = z_cpu.size(0);
    int64_t total_bytes = (n * BITS_PER_CODEWORD + 7) / 8;
    std::vector<uint8_t> packed(static_cast<size_t>(total_bytes), 0u);

    const int32_t * data = z_cpu.data_ptr<int32_t>();
    int64_t bit_pos = 0;
    for (int64_t i = 0; i < n; ++i) {
      uint16_t cw = static_cast<uint16_t>(data[i]);
      for (int b = BITS_PER_CODEWORD - 1; b >= 0; --b) {
        uint8_t bit = (cw >> b) & 1u;
        packed[bit_pos / 8] |= static_cast<uint8_t>(bit << (7 - (bit_pos % 8)));
        ++bit_pos;
      }
    }
    return packed;
  }

  void mask_callback(const arc_interfaces::msg::Mask::SharedPtr msg)
  {
    std::vector<int64_t> indices;
    indices.reserve(msg->mask.size());
    for (size_t i = 0; i < msg->mask.size(); ++i) {
      if (msg->mask[i]) {
        indices.push_back(static_cast<int64_t>(i));
      }
    }
    if (indices.empty()) {
      mask_indices_ = torch::empty(
        {0}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
    } else {
      mask_indices_ = torch::tensor(
        indices, torch::TensorOptions().dtype(torch::kInt64)).to(device_);
    }
  }

  void img_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    rclcpp::Time t0 = get_clock()->now();

    load_image(msg);
    auto z = torch::argmax(trt_forward(), 1);          // (1, H', W')
    auto z_flat = z.squeeze(0).flatten();              // (H'*W',)
    auto z_selected = z_flat.index_select(0, mask_indices_);

    auto packed = pack_codewords(z_selected);
    uint16_t n_selected = static_cast<uint16_t>(mask_indices_.size(0));

    // Reconstruct a bool mask from mask_indices_ for the receiver.
    auto indices_cpu = mask_indices_.to(torch::kCPU).contiguous();
    std::vector<bool> mask_out(static_cast<size_t>(n_codewords_), false);
    const int64_t * idx_data = indices_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < indices_cpu.size(0); ++i) {
      mask_out[static_cast<size_t>(idx_data[i])] = true;
    }

    arc_interfaces::msg::Code code_msg;
    code_msg.header.stamp = msg->header.stamp;
    code_msg.header.frame_id = msg->header.frame_id;
    code_msg.length = n_selected;
    code_msg.data = packed;
    code_msg.mask = mask_out;
    code_pub_->publish(code_msg);

    double latency_ms = (get_clock()->now() - t0).nanoseconds() / 1e6;
    RCLCPP_INFO(get_logger(),
      "codewords: %d  selected: %d  packed: %zu bytes  latency: %.4f ms",
      n_codewords_, n_selected, packed.size(), latency_ms);
  }

  const int img_width_, img_height_;
  const int h_prime_, w_prime_, n_codewords_;
  torch::Device device_;

  TRTLogger logger_;
  nvinfer1::IRuntime * runtime_{nullptr};
  nvinfer1::ICudaEngine * engine_{nullptr};
  nvinfer1::IExecutionContext * context_{nullptr};
  cudaStream_t stream_{nullptr};
  std::vector<void *> bindings_;

  torch::Tensor pinned_input_;
  torch::Tensor input_buf_, output_buf_;

  torch::Tensor mask_indices_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Subscription<arc_interfaces::msg::Mask>::SharedPtr mask_sub_;
  rclcpp::Publisher<arc_interfaces::msg::Code>::SharedPtr code_pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CodePublisher>());
  rclcpp::shutdown();
  return 0;
}
