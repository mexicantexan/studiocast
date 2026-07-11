#include "core/open_video/vulkan_matting_session.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <string>
#include <utility>

namespace studiocast::open_vulkan {

namespace {

constexpr const char *kDeviceResidentInferenceUnavailable =
    "OpenVulkanMattingSession: device-resident Vulkan matting inference "
    "runtime is unavailable in this build. The current ncnn Vulkan spike uses "
    "CPU Mat input/output and is not a production Vulkan virtual background "
    "runtime.";

bool FileExists(const std::filesystem::path &p) {
  std::error_code ec;
  return std::filesystem::exists(p, ec) && !ec;
}

bool IsFinite3(const std::array<double, 3> &v) {
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

const studiocast::open_video::ModelFile *
FindMainOnnxFile(const studiocast::open_video::ModelPack &p) {
  for (const auto &f : p.files) {
    if (f.kind == "onnx" && (f.role.empty() || f.role == "main"))
      return &f;
  }
  for (const auto &f : p.files) {
    if (f.kind == "onnx")
      return &f;
  }
  return nullptr;
}

} // namespace

struct OpenVulkanMattingSession::Impl {
  studiocast::vulkan::VulkanDevice *device = nullptr;
  studiocast::vulkan::kernels::UtilityKernels *kernels = nullptr;
  studiocast::open_video::ModelPack pack;
  std::filesystem::path onnx_path;
  Options opts;

  bool validated_pack = false;
  bool buffers_ready = false;
  int last_frame_w = 0;
  int last_frame_h = 0;

  studiocast::vulkan::kernels::ModelPreprocessSpec preprocess{};
  studiocast::vulkan::VulkanTensor input_tensor;

  bool ValidatePack(std::string *error_out) {
    if (validated_pack)
      return true;

    if (pack.task != "matting") {
      if (error_out)
        *error_out =
            "OpenVulkanMattingSession: model pack task must be 'matting'.";
      return false;
    }
    if (!pack.matting.has_value()) {
      if (error_out)
        *error_out = "OpenVulkanMattingSession: matting model pack is missing "
                     "required metadata.";
      return false;
    }

    const auto &spec = *pack.matting;
    if (spec.input.layout != "nchw") {
      if (error_out)
        *error_out =
            "OpenVulkanMattingSession: only NCHW models are supported.";
      return false;
    }
    if (spec.input.dtype != "float32") {
      if (error_out)
        *error_out =
            "OpenVulkanMattingSession: only float32 input models are "
            "supported.";
      return false;
    }
    if (spec.output.dtype != "float32") {
      if (error_out)
        *error_out =
            "OpenVulkanMattingSession: only float32 output models are "
            "supported.";
      return false;
    }
    if (spec.input.channels != 3) {
      if (error_out)
        *error_out =
            "OpenVulkanMattingSession: model input must have 3 channels.";
      return false;
    }
    if (spec.input.width <= 0 || spec.input.height <= 0) {
      if (error_out)
        *error_out = "OpenVulkanMattingSession: invalid model input size.";
      return false;
    }
    if (!IsFinite3(spec.preprocess.mean) || !IsFinite3(spec.preprocess.std)) {
      if (error_out)
        *error_out =
            "OpenVulkanMattingSession: preprocess mean/std must be finite.";
      return false;
    }
    if (spec.preprocess.std[0] == 0.0 || spec.preprocess.std[1] == 0.0 ||
        spec.preprocess.std[2] == 0.0) {
      if (error_out)
        *error_out =
            "OpenVulkanMattingSession: preprocess std must be non-zero.";
      return false;
    }
    if (spec.preprocess.color != "rgb" || spec.preprocess.range != "0..1") {
      if (error_out)
        *error_out = "OpenVulkanMattingSession: unsupported preprocess spec "
                     "(expected rgb + 0..1).";
      return false;
    }

    const auto *onnx = FindMainOnnxFile(pack);
    if (!onnx) {
      if (error_out)
        *error_out =
            "OpenVulkanMattingSession: missing ONNX file (kind=onnx).";
      return false;
    }
    onnx_path = onnx->path;
    if (!FileExists(onnx_path)) {
      if (error_out)
        *error_out = "OpenVulkanMattingSession: missing ONNX file at " +
                     onnx_path.string();
      return false;
    }

    preprocess.dst_w = spec.input.width;
    preprocess.dst_h = spec.input.height;
    for (std::size_t i = 0; i < 3; ++i) {
      preprocess.mean[i] = static_cast<float>(spec.preprocess.mean[i]);
      preprocess.std[i] = static_cast<float>(spec.preprocess.std[i]);
    }
    preprocess.dst_order = studiocast::vulkan::kernels::ChannelOrder::rgb;

    validated_pack = true;
    return true;
  }
};

OpenVulkanMattingSession::OpenVulkanMattingSession(
    studiocast::vulkan::VulkanDevice *device,
    studiocast::vulkan::kernels::UtilityKernels *kernels,
    studiocast::open_video::ModelPack pack, Options opts)
    : impl_(std::make_unique<Impl>()) {
  impl_->device = device;
  impl_->kernels = kernels;
  impl_->pack = std::move(pack);
  impl_->opts = opts;
}

OpenVulkanMattingSession::OpenVulkanMattingSession(
    studiocast::vulkan::VulkanDevice *device,
    studiocast::vulkan::kernels::UtilityKernels *kernels,
    studiocast::open_video::ModelPack pack)
    : OpenVulkanMattingSession(device, kernels, std::move(pack), Options{}) {}

OpenVulkanMattingSession::~OpenVulkanMattingSession() = default;

const studiocast::open_video::ModelPack &
OpenVulkanMattingSession::pack() const {
  return impl_->pack;
}

const OpenVulkanMattingSession::Options &
OpenVulkanMattingSession::options() const {
  return impl_->opts;
}

bool OpenVulkanMattingSession::DeviceResidentInferenceAvailable() const {
  return false;
}

bool OpenVulkanMattingSession::EnsureInitialized(int frame_w, int frame_h,
                                                 std::string *error_out) {
  if (error_out)
    error_out->clear();

  impl_->last_frame_w = frame_w;
  impl_->last_frame_h = frame_h;

  (void)impl_->opts.device_id;
  (void)impl_->opts.allow_cpu_layers;

  if (!impl_->device || !impl_->device->Initialized()) {
    if (error_out)
      *error_out =
          "OpenVulkanMattingSession: Vulkan device is not initialized.";
    return false;
  }
  if (!impl_->kernels || !impl_->kernels->Initialized()) {
    if (error_out)
      *error_out =
          "OpenVulkanMattingSession: Vulkan utility kernels are not "
          "initialized.";
    return false;
  }
  if (frame_w <= 0 || frame_h <= 0) {
    if (error_out)
      *error_out = "OpenVulkanMattingSession: invalid frame size.";
    return false;
  }
  if (!impl_->ValidatePack(error_out))
    return false;

  std::string alloc_err;
  if (!impl_->input_tensor.Valid() || impl_->input_tensor.n() != 1 ||
      impl_->input_tensor.c() != 3 ||
      impl_->input_tensor.h() != impl_->pack.matting->input.height ||
      impl_->input_tensor.w() != impl_->pack.matting->input.width) {
    if (!impl_->input_tensor.AllocateNchwF32(
            impl_->device, /*n_in=*/1, /*c_in=*/3,
            /*h_in=*/impl_->pack.matting->input.height,
            /*w_in=*/impl_->pack.matting->input.width,
            /*map_memory=*/false, &alloc_err)) {
      if (error_out)
        *error_out =
            "OpenVulkanMattingSession: failed to allocate input tensor: " +
            alloc_err;
      return false;
    }
  }

  impl_->buffers_ready = true;
  if (impl_->opts.require_device_residency &&
      !DeviceResidentInferenceAvailable()) {
    if (error_out)
      *error_out = kDeviceResidentInferenceUnavailable;
    return false;
  }
  return true;
}

bool OpenVulkanMattingSession::Warmup(std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!impl_->buffers_ready) {
    if (error_out)
      *error_out = "OpenVulkanMattingSession: warmup requires initialized "
                   "buffers.";
    return false;
  }
  if (!DeviceResidentInferenceAvailable()) {
    if (error_out)
      *error_out = kDeviceResidentInferenceUnavailable;
    return false;
  }
  (void)impl_->opts.warmup_runs;
  return true;
}

bool OpenVulkanMattingSession::Run(
    const studiocast::vulkan::VulkanImage &input_rgb_gpu,
    studiocast::vulkan::VulkanImage *output_alpha_gpu,
    std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!output_alpha_gpu) {
    if (error_out)
      *error_out = "OpenVulkanMattingSession::Run: output_alpha_gpu is null.";
    return false;
  }
  if (!EnsureInitialized(input_rgb_gpu.width(), input_rgb_gpu.height(),
                         error_out)) {
    return false;
  }
  if (!input_rgb_gpu.Valid()) {
    if (error_out)
      *error_out = "OpenVulkanMattingSession::Run: invalid input image.";
    return false;
  }
  if (!output_alpha_gpu->Valid() ||
      output_alpha_gpu->format() !=
          studiocast::vulkan::VulkanPixelFormat::f32_1 ||
      output_alpha_gpu->width() != impl_->pack.matting->input.width ||
      output_alpha_gpu->height() != impl_->pack.matting->input.height) {
    if (error_out) {
      *error_out =
          "OpenVulkanMattingSession::Run: output alpha image shape mismatch "
          "(expected f32_1 " +
          std::to_string(impl_->pack.matting->input.width) + "x" +
          std::to_string(impl_->pack.matting->input.height) + ")";
    }
    return false;
  }

  std::string pp_err;
  if (!impl_->kernels->PreprocessToTensor(input_rgb_gpu, impl_->input_tensor,
                                          impl_->preprocess, &pp_err)) {
    if (error_out)
      *error_out =
          "OpenVulkanMattingSession::Run: preprocess failed: " + pp_err;
    return false;
  }

  if (error_out)
    *error_out = kDeviceResidentInferenceUnavailable;
  return false;
}

} // namespace studiocast::open_vulkan
