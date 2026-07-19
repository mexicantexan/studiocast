#include "core/open_video/vulkan_matting_session.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace studiocast::open_vulkan {

namespace {

bool IsFinite3(const std::array<double, 3> &v) {
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

template <typename T> std::uintptr_t OpaqueHandle(T handle) {
  if constexpr (std::is_pointer_v<T>) {
    return reinterpret_cast<std::uintptr_t>(handle);
  } else {
    return static_cast<std::uintptr_t>(handle);
  }
}

} // namespace

struct OpenVulkanMattingSession::Impl {
  studiocast::vulkan::VulkanDevice *device = nullptr;
  studiocast::vulkan::kernels::UtilityKernels *kernels = nullptr;
  studiocast::open_video::ModelPack pack;
  Options opts;

  bool pack_validation_attempted = false;
  bool validated_pack = false;
  std::string pack_validation_error;
  bool buffers_ready = false;
  int last_frame_w = 0;
  int last_frame_h = 0;
  studiocast::vulkan::VulkanContextIdentity bound_context{};

  studiocast::vulkan::kernels::ModelPreprocessSpec preprocess{};
  studiocast::vulkan::VulkanTensor input_tensor;
  VulkanMattingGraphDescriptor graph;
  VulkanMattingPersistentResources resources;
  std::unique_ptr<VulkanMattingRuntimeLifecycle> runtime_lifecycle;

  bool PackValidationFailure(std::string message, std::string *error_out) {
    pack_validation_error = std::move(message);
    if (error_out)
      *error_out = pack_validation_error;
    return false;
  }

  VulkanMattingReadiness Readiness() const {
    VulkanMattingReadinessInput input;
    input.production_build_enabled = ProductionVulkanMattingBuildEnabled();
    input.production_adapter_available =
        ProductionVulkanMattingAdapterAvailable();
    input.model_pack_selected = !pack.id.empty();
    input.model_contract_validated = validated_pack;
    if (device) {
      const auto diagnostics = device->DiagnosticsSnapshot();
      input.non_cpu_device_selected = diagnostics.non_cpu_device_selected;
      input.compute_queue_available = diagnostics.compute_queue_available;
      input.context_healthy = diagnostics.context_healthy;
    }
    input.session_initialized =
        buffers_ready && runtime_lifecycle && runtime_lifecycle->prepared();
    input.session_warmed = runtime_lifecycle && runtime_lifecycle->warmed();
    if (runtime_lifecycle) {
      input.failure_latched = runtime_lifecycle->failure_latched();
      input.failure = runtime_lifecycle->latched_failure();
      input.failure_detail = runtime_lifecycle->latched_error();
      input.runtime = runtime_lifecycle->Evidence();
    }
    return EvaluateVulkanMattingReadiness(input);
  }

  bool Latch(VulkanMattingRuntimeFailure failure, std::string detail,
             std::string *error_out) {
    if (!runtime_lifecycle) {
      if (error_out)
        *error_out = FormatVulkanMattingReadiness(Readiness());
      return false;
    }
    std::string ignored;
    (void)runtime_lifecycle->LatchExternalFailure(failure, std::move(detail),
                                                  &ignored);
    if (error_out)
      *error_out = FormatVulkanMattingReadiness(Readiness());
    return false;
  }

  bool ReportReadiness(std::string *error_out) const {
    if (error_out)
      *error_out = FormatVulkanMattingReadiness(Readiness());
    return false;
  }

  bool ValidatePack(std::string *error_out) {
    if (validated_pack)
      return true;
    if (pack_validation_attempted) {
      if (error_out)
        *error_out = pack_validation_error;
      return false;
    }
    pack_validation_attempted = true;

    if (pack.task != "matting") {
      return PackValidationFailure(
          "OpenVulkanMattingSession: model pack task must be 'matting'.",
          error_out);
    }
    if (!pack.matting.has_value()) {
      return PackValidationFailure(
          "OpenVulkanMattingSession: matting model pack is missing required "
          "metadata.",
          error_out);
    }

    const auto &spec = *pack.matting;
    if (spec.input.layout != "nchw") {
      return PackValidationFailure(
          "OpenVulkanMattingSession: only NCHW models are supported.",
          error_out);
    }
    if (spec.input.dtype != "float32") {
      return PackValidationFailure(
          "OpenVulkanMattingSession: only float32 input models are supported.",
          error_out);
    }
    if (spec.output.dtype != "float32") {
      return PackValidationFailure(
          "OpenVulkanMattingSession: only float32 output models are supported.",
          error_out);
    }
    if (spec.input.channels != 3) {
      return PackValidationFailure(
          "OpenVulkanMattingSession: model input must have 3 channels.",
          error_out);
    }
    if (spec.input.width <= 0 || spec.input.height <= 0) {
      return PackValidationFailure(
          "OpenVulkanMattingSession: invalid model input size.", error_out);
    }
    if (!IsFinite3(spec.preprocess.mean) || !IsFinite3(spec.preprocess.std)) {
      return PackValidationFailure(
          "OpenVulkanMattingSession: preprocess mean/std must be finite.",
          error_out);
    }
    if (spec.preprocess.std[0] == 0.0 || spec.preprocess.std[1] == 0.0 ||
        spec.preprocess.std[2] == 0.0) {
      return PackValidationFailure(
          "OpenVulkanMattingSession: preprocess std must be non-zero.",
          error_out);
    }
    if (spec.preprocess.color != "rgb" || spec.preprocess.range != "0..1") {
      return PackValidationFailure(
          "OpenVulkanMattingSession: unsupported preprocess spec (expected "
          "rgb + 0..1).",
          error_out);
    }
    if (opts.allow_cpu_layers) {
      return PackValidationFailure(
          "OpenVulkanMattingSession: production Vulkan matting does not permit "
          "CPU fallback layers.",
          error_out);
    }
    if (opts.warmup_runs != 1) {
      return PackValidationFailure(
          "OpenVulkanMattingSession: production warmup_runs must be exactly 1.",
          error_out);
    }

    std::string contract_error;
    if (!studiocast::open_video::ValidateProductionNcnnVulkanMattingPack(
            pack, &contract_error)) {
      return PackValidationFailure(
          "OpenVulkanMattingSession: production ncnn Vulkan model contract "
          "failed: " +
              contract_error,
          error_out);
    }

    const auto &ncnn = *pack.ncnn_vulkan;
    graph.param_path = ncnn.param_path.string();
    graph.bin_path = ncnn.bin_path.string();
    graph.param_sha256 = ncnn.param_sha256;
    graph.bin_sha256 = ncnn.bin_sha256;
    graph.input_blob = ncnn.input_blob;
    graph.output_blob = ncnn.output_blob;
    graph.converter_name = ncnn.converter_name;
    graph.converter_version = ncnn.converter_version;
    graph.precision = ncnn.precision;
    graph.input_n = 1;
    graph.input_c = 3;
    graph.input_h = spec.input.height;
    graph.input_w = spec.input.width;
    graph.output_n = 1;
    graph.output_c = 1;
    graph.output_h = spec.output.height;
    graph.output_w = spec.output.width;

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
    studiocast::open_video::ModelPack pack, Options opts,
    std::unique_ptr<VulkanMattingRuntime> runtime)
    : impl_(std::make_unique<Impl>()) {
  impl_->device = device;
  impl_->kernels = kernels;
  impl_->pack = std::move(pack);
  impl_->opts = opts;
  impl_->runtime_lifecycle =
      std::make_unique<VulkanMattingRuntimeLifecycle>(std::move(runtime));
}

OpenVulkanMattingSession::OpenVulkanMattingSession(
    studiocast::vulkan::VulkanDevice *device,
    studiocast::vulkan::kernels::UtilityKernels *kernels,
    studiocast::open_video::ModelPack pack, Options opts)
    : OpenVulkanMattingSession(device, kernels, std::move(pack), opts,
                               CreateProductionVulkanMattingRuntime()) {}

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
  return Readiness().production_ready;
}

VulkanMattingReadiness OpenVulkanMattingSession::Readiness() const {
  return impl_->Readiness();
}

VulkanMattingRuntimeEvidence OpenVulkanMattingSession::RuntimeEvidence() const {
  return impl_->runtime_lifecycle ? impl_->runtime_lifecycle->Evidence()
                                  : VulkanMattingRuntimeEvidence{};
}

VulkanMattingRuntimeFailure
OpenVulkanMattingSession::LatchedRuntimeFailure() const {
  return impl_->runtime_lifecycle ? impl_->runtime_lifecycle->latched_failure()
                                  : VulkanMattingRuntimeFailure::unavailable;
}

const std::string &OpenVulkanMattingSession::LatchedRuntimeError() const {
  static const std::string empty;
  return impl_->runtime_lifecycle ? impl_->runtime_lifecycle->latched_error()
                                  : empty;
}

bool OpenVulkanMattingSession::LatchExternalFailure(
    VulkanMattingRuntimeFailure failure, std::string detail,
    std::string *error_out) {
  return impl_->Latch(failure, std::move(detail), error_out);
}

bool OpenVulkanMattingSession::EnsureInitialized(int frame_w, int frame_h,
                                                 std::string *error_out) {
  if (error_out)
    error_out->clear();

  impl_->last_frame_w = frame_w;
  impl_->last_frame_h = frame_h;

  if (!impl_->device || !impl_->device->Initialized()) {
    return impl_->Latch(
        VulkanMattingRuntimeFailure::device_identity_mismatch,
        "OpenVulkanMattingSession: Vulkan device is not initialized.",
        error_out);
  }
  if (!impl_->kernels || !impl_->kernels->Initialized()) {
    return impl_->Latch(
        VulkanMattingRuntimeFailure::synchronization_failed,
        "OpenVulkanMattingSession: Vulkan utility kernels are not "
        "initialized.",
        error_out);
  }
  if (frame_w <= 0 || frame_h <= 0) {
    return impl_->Latch(VulkanMattingRuntimeFailure::invalid_shape,
                        "OpenVulkanMattingSession: invalid frame size.",
                        error_out);
  }
  std::string validation_error;
  if (!impl_->ValidatePack(&validation_error)) {
    return impl_->Latch(VulkanMattingRuntimeFailure::invalid_graph,
                        std::move(validation_error), error_out);
  }

  // Contract/hash validation is one-time and precedes readiness. An absent
  // shared-device adapter is then rejected before any Vulkan tensor/resource
  // allocation.
  if (!ProductionVulkanMattingBuildEnabled() ||
      !ProductionVulkanMattingAdapterAvailable()) {
    return impl_->Latch(
        VulkanMattingRuntimeFailure::unavailable,
        "No production shared-device ncnn Vulkan adapter is available.",
        error_out);
  }

  const auto diagnostics = impl_->device->DiagnosticsSnapshot();
  const auto current_context = impl_->device->context_identity();
  if (!diagnostics.production_hardware_ready || !current_context.Valid()) {
    return impl_->Latch(
        diagnostics.context_health == "device_lost"
            ? VulkanMattingRuntimeFailure::device_lost
            : VulkanMattingRuntimeFailure::device_identity_mismatch,
        "OpenVulkanMattingSession: a healthy non-CPU Vulkan context and "
        "compute queue are required.",
        error_out);
  }
  if (impl_->bound_context.Valid() &&
      !(impl_->bound_context == current_context)) {
    return impl_->Latch(
        VulkanMattingRuntimeFailure::context_generation_mismatch,
        "OpenVulkanMattingSession: Vulkan context identity/generation "
        "changed; recreate the matting session.",
        error_out);
  }

  std::string alloc_err;
  if (!impl_->input_tensor.Valid() || impl_->input_tensor.n() != 1 ||
      impl_->input_tensor.c() != 3 ||
      impl_->input_tensor.h() != impl_->pack.matting->input.height ||
      impl_->input_tensor.w() != impl_->pack.matting->input.width ||
      !impl_->input_tensor.device_local() ||
      impl_->input_tensor.mapped() != nullptr) {
    if (!impl_->input_tensor.AllocateNchwF32(
            impl_->device, /*n_in=*/1, /*c_in=*/3,
            /*h_in=*/impl_->pack.matting->input.height,
            /*w_in=*/impl_->pack.matting->input.width,
            /*map_memory=*/false, &alloc_err)) {
      return impl_->Latch(
          VulkanMattingRuntimeFailure::allocation_contract_failed,
          "OpenVulkanMattingSession: failed to allocate input tensor: " +
              alloc_err,
          error_out);
    }
  }

  studiocast::vulkan::VulkanTensorSize alpha_size;
  std::string size_error;
  if (!studiocast::vulkan::CheckedNchwF32Size(
          1, 1, impl_->pack.matting->output.height,
          impl_->pack.matting->output.width, &alpha_size, &size_error)) {
    return impl_->Latch(
        VulkanMattingRuntimeFailure::invalid_shape,
        "OpenVulkanMattingSession: invalid alpha allocation size: " +
            size_error,
        error_out);
  }
  impl_->resources.input_bytes = impl_->input_tensor.bytes();
  impl_->resources.alpha_bytes = alpha_size.bytes;
  impl_->resources.allow_cpu_layers = false;
  impl_->resources.require_device_residency =
      impl_->opts.require_device_residency;

  VulkanMattingDeviceContext context;
  context.ownership_domain = impl_->device;
  context.physical_device = OpaqueHandle(impl_->device->physical_device());
  context.logical_device = OpaqueHandle(impl_->device->device());
  context.queue = OpaqueHandle(impl_->device->queue());
  context.queue_family_index = impl_->device->queue_family_index();
  context.vendor_id = impl_->device->identity().vendor_id;
  context.device_id = impl_->device->identity().device_id;
  context.selected_device_index =
      impl_->device->diagnostics().selected_device_index;
  context.stable_device_id = impl_->device->identity().stable_id;
  context.context_id = current_context.context_id;
  context.context_generation = current_context.generation;
  context.non_cpu_device_selected = diagnostics.non_cpu_device_selected;
  context.compute_queue_available = diagnostics.compute_queue_available;
  context.context_healthy = diagnostics.context_healthy;
  if (!impl_->runtime_lifecycle ||
      !impl_->runtime_lifecycle->Prepare(context, impl_->graph,
                                         impl_->resources, error_out)) {
    return impl_->ReportReadiness(error_out);
  }
  impl_->bound_context = current_context;
  impl_->buffers_ready = true;
  return true;
}

bool OpenVulkanMattingSession::Warmup(std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!impl_->buffers_ready)
    return impl_->ReportReadiness(error_out);
  if (!impl_->runtime_lifecycle ||
      !impl_->runtime_lifecycle->Warmup(error_out)) {
    return impl_->ReportReadiness(error_out);
  }
  return true;
}

bool OpenVulkanMattingSession::Run(
    const studiocast::vulkan::VulkanImage &input_rgb_gpu,
    studiocast::vulkan::VulkanImage *output_alpha_gpu, std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!output_alpha_gpu) {
    return impl_->Latch(
        VulkanMattingRuntimeFailure::invalid_shape,
        "OpenVulkanMattingSession::Run: output_alpha_gpu is null.", error_out);
  }
  if (!impl_->buffers_ready) {
    if (!EnsureInitialized(input_rgb_gpu.width(), input_rgb_gpu.height(),
                           error_out)) {
      return false;
    }
  }
  if (!input_rgb_gpu.Valid()) {
    return impl_->Latch(VulkanMattingRuntimeFailure::residency_check_failed,
                        "OpenVulkanMattingSession::Run: invalid input image.",
                        error_out);
  }
  const auto current_context = impl_->device->context_identity();
  if (!impl_->bound_context.Valid() || !current_context.Valid() ||
      !(impl_->bound_context == current_context)) {
    return impl_->Latch(
        VulkanMattingRuntimeFailure::context_generation_mismatch,
        "OpenVulkanMattingSession::Run: Vulkan context identity/generation "
        "changed; recreate the matting session.",
        error_out);
  }
  if (!input_rgb_gpu.BelongsTo(*impl_->device)) {
    return impl_->Latch(
        VulkanMattingRuntimeFailure::device_identity_mismatch,
        "OpenVulkanMattingSession::Run: input image belongs to a different "
        "Vulkan context identity/generation.",
        error_out);
  }
  if (!impl_->runtime_lifecycle || !impl_->runtime_lifecycle->available()) {
    if (error_out)
      *error_out = FormatVulkanMattingReadiness(Readiness());
    return false;
  }
  if (!output_alpha_gpu->Valid() ||
      output_alpha_gpu->format() !=
          studiocast::vulkan::VulkanPixelFormat::f32_1 ||
      output_alpha_gpu->width() != impl_->pack.matting->output.width ||
      output_alpha_gpu->height() != impl_->pack.matting->output.height) {
    if (error_out) {
      error_out->clear();
    }
    return impl_->Latch(
        VulkanMattingRuntimeFailure::invalid_shape,
        "OpenVulkanMattingSession::Run: output alpha image shape mismatch "
        "(expected f32_1 " +
            std::to_string(impl_->pack.matting->output.width) + "x" +
            std::to_string(impl_->pack.matting->output.height) + ")",
        error_out);
  }
  if (!output_alpha_gpu->BelongsTo(*impl_->device)) {
    return impl_->Latch(
        VulkanMattingRuntimeFailure::device_identity_mismatch,
        "OpenVulkanMattingSession::Run: output alpha image belongs to a "
        "different Vulkan context identity/generation.",
        error_out);
  }

  std::string pp_err;
  if (!impl_->kernels->PreprocessToTensor(input_rgb_gpu, impl_->input_tensor,
                                          impl_->preprocess, &pp_err)) {
    const auto health = impl_->device->health();
    return impl_->Latch(
        health.health == studiocast::vulkan::VulkanContextHealth::device_lost
            ? VulkanMattingRuntimeFailure::device_lost
            : VulkanMattingRuntimeFailure::execution_failed,
        "OpenVulkanMattingSession::Run: preprocess failed: " + pp_err,
        error_out);
  }

  VulkanMattingBufferBinding input_binding;
  input_binding.ownership_domain = impl_->device;
  input_binding.logical_device = OpaqueHandle(impl_->device->device());
  input_binding.buffer = OpaqueHandle(impl_->input_tensor.buffer());
  input_binding.byte_size = impl_->input_tensor.bytes();
  input_binding.n = impl_->input_tensor.n();
  input_binding.c = impl_->input_tensor.c();
  input_binding.h = impl_->input_tensor.h();
  input_binding.w = impl_->input_tensor.w();
  input_binding.device_resident = impl_->input_tensor.Valid();
  input_binding.device_local_memory = impl_->input_tensor.device_local();
  input_binding.host_mapped = impl_->input_tensor.mapped() != nullptr;
  input_binding.context_id = impl_->bound_context.context_id;
  input_binding.context_generation = impl_->bound_context.generation;
  // PreprocessToTensor records the producer barrier and completes through the
  // device's serialized SubmitAndWait queue contract. A production adapter
  // must run synchronously on that same queue before returning.
  input_binding.access_synchronized = true;
  input_binding.runtime_queue_ownership = true;

  VulkanMattingBufferBinding alpha_binding;
  alpha_binding.ownership_domain = impl_->device;
  alpha_binding.logical_device = OpaqueHandle(impl_->device->device());
  alpha_binding.buffer = OpaqueHandle(output_alpha_gpu->buffer());
  alpha_binding.byte_size =
      static_cast<std::size_t>(output_alpha_gpu->byte_size());
  alpha_binding.n = 1;
  alpha_binding.c = 1;
  alpha_binding.h = output_alpha_gpu->height();
  alpha_binding.w = output_alpha_gpu->width();
  alpha_binding.device_resident = output_alpha_gpu->Valid();
  alpha_binding.device_local_memory = output_alpha_gpu->device_local();
  alpha_binding.host_mapped = output_alpha_gpu->mapped() != nullptr;
  alpha_binding.context_id = impl_->bound_context.context_id;
  alpha_binding.context_generation = impl_->bound_context.generation;
  alpha_binding.access_synchronized = true;
  alpha_binding.runtime_queue_ownership = true;

  if (!impl_->runtime_lifecycle->Run(input_binding, alpha_binding, error_out))
    return impl_->ReportReadiness(error_out);
  return true;
}

} // namespace studiocast::open_vulkan
