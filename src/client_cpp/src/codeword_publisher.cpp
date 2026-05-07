#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <arc_interfaces/msg/code.hpp>
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

    int input_idx = engine_->getBindingIndex("pixels");
    int output_idx = engine_->getBindingIndex("conv2d_36");
    context_->setBindingDimensions(input_idx,
      nvinfer1::Dims4{1, 3, img_height_, img_width_});
    bindings_[input_idx] = input_buf_.data_ptr<float>();
    bindings_[output_idx] = output_buf_.data_ptr<float>();

    img_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/camera/camera/color/image_raw", 1,
      std::bind(&CodePublisher::img_callback, this, std::placeholders::_1));
    mask_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/camera/camera/color/mask", 1,
      std::bind(&CodePublisher::mask_callback, this, std::placeholders::_1));
    code_pub_ = create_publisher<arc_interfaces::msg::Code>(
      "/camera/camera/color/code", 1);

    // Warm-up: zeros through the full pipeline.
    {
      c10::cuda::CUDAStreamGuard guard(
        c10::cuda::getStreamFromExternal(stream_, device_.index()));
      input_buf_.copy_(map_pixels(
        torch::zeros({1, 3, img_height_, img_width_}, device_)));
    }
    mask_z_ = torch::argmax(trt_forward(), 1);
    RCLCPP_INFO(get_logger(), "mask_z shape: [%ld, %ld, %ld]",
      mask_z_.size(0), mask_z_.size(1), mask_z_.size(2));
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
  }

  // Copies the ROS image into pinned memory, then asynchronously transfers
  // it to the GPU and preprocesses (HWC uint8 -> CHW float32 in [eps,1-eps])
  // entirely on stream_, writing directly into input_buf_.
  void load_image(const sensor_msgs::msg::Image::SharedPtr & msg)
  {
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
    context_->enqueueV2(bindings_, stream_, nullptr);
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

  void mask_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    load_image(msg);
    mask_z_ = torch::argmax(trt_forward(), 1);
  }

  void img_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    int64_t now_ns = get_clock()->now().nanoseconds();

    load_image(msg);
    auto z = torch::argmax(trt_forward(), 1);  // (1, H', W')
    auto packed = pack_codewords(z.squeeze(0).flatten());

    arc_interfaces::msg::Code code_msg;
    code_msg.header.stamp = get_clock()->now();
    code_msg.header.frame_id = msg->header.frame_id;
    code_msg.length = static_cast<uint16_t>(z.size(1) * z.size(2));
    code_msg.data = packed;
    code_pub_->publish(code_msg);

    double latency_ms = (get_clock()->now().nanoseconds() - now_ns) / 1e6;
    RCLCPP_INFO(get_logger(),
      "codewords: %d  packed: %zu bytes  latency: %.4f ms",
      n_codewords_, packed.size(), latency_ms);
  }

  const int img_width_, img_height_;
  const int h_prime_, w_prime_, n_codewords_;
  torch::Device device_;

  TRTLogger logger_;
  nvinfer1::IRuntime * runtime_{nullptr};
  nvinfer1::ICudaEngine * engine_{nullptr};
  nvinfer1::IExecutionContext * context_{nullptr};
  cudaStream_t stream_{nullptr};
  void * bindings_[2]{};

  torch::Tensor pinned_input_;
  torch::Tensor input_buf_, output_buf_;
  torch::Tensor mask_z_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr mask_sub_;
  rclcpp::Publisher<arc_interfaces::msg::Code>::SharedPtr code_pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CodePublisher>());
  rclcpp::shutdown();
  return 0;
}
