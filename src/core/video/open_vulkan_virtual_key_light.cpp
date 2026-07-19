#include "core/video/open_vulkan_virtual_key_light.h"

#include <algorithm>

namespace studiocast::video {
namespace detail {

bool IsOpenVulkanVirtualKeyLightSameFrameArtifactCompatible(
    std::uint64_t capture_sequence, std::uint64_t cached_matte_sequence,
    std::uint64_t cached_frame_alpha_sequence,
    const studiocast::open_video::FrameMatteArtifactKey &key,
    const studiocast::open_video::FrameMatteArtifact *artifact,
    std::string_view active_model_id, int frame_width, int frame_height,
    int matte_width, int matte_height, std::uintptr_t device_context,
    std::uintptr_t queue, std::uintptr_t expected_object_handle,
    std::uintptr_t expected_buffer_handle) {
  return capture_sequence != 0 && cached_matte_sequence == capture_sequence &&
         cached_frame_alpha_sequence == capture_sequence && artifact &&
         artifact->key == key && key.provider_id == "open_vulkan" &&
         key.model_id == active_model_id &&
         key.storage ==
             studiocast::open_video::FrameMatteStorage::device_f32_alpha &&
         key.frame_width == frame_width && key.frame_height == frame_height &&
         key.matte_width == matte_width && key.matte_height == matte_height &&
         key.config_fingerprint == 0 && key.device_context == device_context &&
         key.stream == queue && artifact->handle == expected_object_handle &&
         artifact->aux_handle == expected_buffer_handle;
}

} // namespace detail
namespace {

std::string StableFailure(std::string_view code, std::string_view summary,
                          std::string_view detail) {
  std::string out = "[";
  out += code;
  out += "] ";
  out += summary;
  if (!detail.empty()) {
    out += ": ";
    out += detail;
  }
  return out;
}

OpenVulkanVirtualKeyLightReadiness NotReady(std::string reason_code,
                                            std::string detail) {
  OpenVulkanVirtualKeyLightReadiness result;
  result.reason_code = std::move(reason_code);
  result.detail = std::move(detail);
  return result;
}

bool Aliases(const studiocast::vulkan::VulkanImage *left,
             const studiocast::vulkan::VulkanImage *right) {
  return left == right || left->buffer() == right->buffer();
}

} // namespace

std::string
OpenVulkanVirtualKeyLightInitializationFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanVirtualKeyLightInitializationReason,
                       "Open Vulkan virtual key light initialization failed",
                       detail);
}

std::string OpenVulkanVirtualKeyLightRuntimeFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanVirtualKeyLightRuntimeReason,
                       "Open Vulkan virtual key light runtime failed", detail);
}

OpenVulkanVirtualKeyLightParameters ResolveOpenVulkanVirtualKeyLightParameters(
    int intensity_percent, int temperature_preset, int direction_pan_degrees) {
  OpenVulkanVirtualKeyLightParameters parameters;
  parameters.intensity =
      static_cast<float>(std::clamp(intensity_percent, 0, 100)) / 100.0f;
  if (temperature_preset == 1) {
    parameters.target_r = 255.0f;
    parameters.target_g = 242.0f;
    parameters.target_b = 228.0f;
  } else if (temperature_preset == 2) {
    parameters.target_r = 228.0f;
    parameters.target_g = 242.0f;
    parameters.target_b = 255.0f;
  }
  const int clamped_pan = std::clamp(direction_pan_degrees, -90, 90);
  parameters.direction = static_cast<float>(clamped_pan) / 90.0f;
  return parameters;
}

OpenVulkanVirtualKeyLightReadiness EvaluateOpenVulkanVirtualKeyLightReadiness(
    const studiocast::vulkan::OpenVulkanDiagnostics &diagnostics,
    const studiocast::open_vulkan::VulkanMattingReadiness &matting_readiness) {
  if (!diagnostics.compiled_enabled) {
    return NotReady("vulkan_backend_disabled_in_build",
                    "Open Vulkan support is disabled in this build");
  }
  if (!diagnostics.runtime_library_found) {
    return NotReady("vulkan_runtime_not_found",
                    "the Vulkan runtime library is unavailable");
  }
  if (!diagnostics.physical_device_found) {
    return NotReady("vulkan_no_physical_device",
                    "no Vulkan physical device was found");
  }
  if (diagnostics.cpu_device_selected || !diagnostics.non_cpu_device_selected) {
    return NotReady(
        "vulkan_only_cpu_devices_available",
        "the selected Vulkan device is a CPU/software implementation");
  }
  if (!diagnostics.compute_queue_available) {
    return NotReady("vulkan_no_compute_queue",
                    "the selected Vulkan device has no compute queue");
  }
  if (!diagnostics.logical_device_created) {
    return NotReady("vulkan_device_create_failed",
                    "the Vulkan logical device was not created");
  }
  if (!diagnostics.context_created || !diagnostics.context_healthy) {
    return NotReady(diagnostics.context_failure_reason.empty()
                        ? "vulkan_context_uninitialized"
                        : diagnostics.context_failure_reason,
                    diagnostics.error.empty()
                        ? "the shared Vulkan context is not healthy"
                        : diagnostics.error);
  }
  if (!diagnostics.production_hardware_ready) {
    return NotReady("vulkan_production_hardware_not_ready",
                    "shared Vulkan diagnostics did not prove production "
                    "hardware readiness");
  }
  if (!diagnostics.shader_pipeline_created || !diagnostics.ok) {
    return NotReady("open_vulkan_utility_kernels_unavailable",
                    "the shared Vulkan utility shader pipeline is unavailable");
  }
  if (!matting_readiness.production_ready) {
    return NotReady(
        matting_readiness.reason_code.empty()
            ? studiocast::open_vulkan::kOpenVulkanMattingUnavailableReason
            : matting_readiness.reason_code,
        studiocast::open_vulkan::FormatVulkanMattingReadiness(
            matting_readiness));
  }

  const auto &runtime = matting_readiness.runtime;
  if (!runtime.runtime_created || !runtime.graph_loaded ||
      !runtime.warmup_complete || runtime.cpu_layers_used ||
      !runtime.device_identity_matches || !runtime.input_device_resident ||
      !runtime.alpha_device_resident || !runtime.output_device_resident ||
      !runtime.shared_device_imported || !runtime.queue_ownership_explicit ||
      !runtime.synchronous_completion ||
      !runtime.bounded_reusable_allocations ||
      runtime.dynamic_allocation_count != 0 ||
      runtime.cpu_readback_count != 0) {
    return NotReady(
        studiocast::open_vulkan::kOpenVulkanMattingUnavailableReason,
        "shared matting readiness omitted required residency, synchronization, "
        "or bounded-allocation evidence");
  }

  OpenVulkanVirtualKeyLightReadiness result;
  result.production_ready = true;
  return result;
}

void OpenVulkanVirtualKeyLight::Shutdown() noexcept {
  kernels_ = nullptr;
  context_identity_ = {};
  parameters_ = {};
  width_ = 0;
  height_ = 0;
  initialized_ = false;
}

bool OpenVulkanVirtualKeyLight::EnsureInitialized(
    studiocast::vulkan::kernels::UtilityKernels *kernels, int width, int height,
    int intensity_percent, int temperature_preset, int direction_pan_degrees,
    const studiocast::open_vulkan::VulkanMattingReadiness &matting_readiness,
    std::string *error_out) {
  if (error_out)
    error_out->clear();
  auto fail = [&](std::string detail) {
    Shutdown();
    if (error_out)
      *error_out = OpenVulkanVirtualKeyLightInitializationFailure(detail);
    return false;
  };
  if (!kernels || width <= 0 || height <= 0)
    return fail("invalid shared context or frame dimensions");

  std::string detail;
  if (!kernels->Initialized() && !kernels->Initialize(&detail))
    return fail(detail);
  const auto readiness = EvaluateOpenVulkanVirtualKeyLightReadiness(
      kernels->Diagnostics(), matting_readiness);
  if (!readiness.production_ready)
    return fail(StableFailure(readiness.reason_code, readiness.detail, {}));
  if (!kernels->device() || !kernels->device()->context_identity().Valid())
    return fail("shared Vulkan context identity is invalid");

  const auto identity = kernels->device()->context_identity();
  const auto &runtime_device = matting_readiness.runtime.active_device;
  if (runtime_device.ownership_domain != kernels->device() ||
      runtime_device.logical_device !=
          reinterpret_cast<std::uintptr_t>(kernels->device()->device()) ||
      runtime_device.queue !=
          reinterpret_cast<std::uintptr_t>(kernels->device()->queue()) ||
      runtime_device.context_id != identity.context_id ||
      runtime_device.context_generation != identity.generation) {
    return fail(StableFailure(
        studiocast::open_vulkan::kOpenVulkanMattingDeviceMismatchReason,
        "matting readiness does not belong to the shared utility context", {}));
  }

  kernels_ = kernels;
  context_identity_ = identity;
  parameters_ = ResolveOpenVulkanVirtualKeyLightParameters(
      intensity_percent, temperature_preset, direction_pan_degrees);
  width_ = width;
  height_ = height;
  initialized_ = true;
  return true;
}

bool OpenVulkanVirtualKeyLight::Apply(
    const OpenVulkanVirtualKeyLightInput &input,
    OpenVulkanVirtualKeyLightCounters *counters, std::string *error_out) {
  if (error_out)
    error_out->clear();
  auto fail = [&](std::string detail) {
    if (counters) {
      ++counters->runtime_failure_frames;
      if (kernels_ && kernels_->device() &&
          kernels_->device()->health().health ==
              studiocast::vulkan::VulkanContextHealth::device_lost) {
        ++counters->device_loss_frames;
      }
    }
    if (error_out)
      *error_out = OpenVulkanVirtualKeyLightRuntimeFailure(detail);
    return false;
  };
  if (!counters)
    return fail("counter output is null");
  if (!initialized_ || !kernels_ || !kernels_->device())
    return fail("wrapper is not initialized");
  if (kernels_->device()->context_identity() != context_identity_) {
    return fail("[vulkan_context_generation_changed] shared Vulkan context "
                "generation changed");
  }
  if (!input.foreground || !input.output || !input.matting_readiness)
    return fail("one or more resident effect resources are null");

  const auto &matting = *input.matting_readiness;
  const auto &runtime = matting.runtime;
  const auto current_readiness = EvaluateOpenVulkanVirtualKeyLightReadiness(
      kernels_->Diagnostics(), matting);
  if (!current_readiness.production_ready) {
    return fail(StableFailure(current_readiness.reason_code,
                              current_readiness.detail, {}));
  }
  if (runtime.active_device.ownership_domain != kernels_->device() ||
      runtime.active_device.logical_device !=
          reinterpret_cast<std::uintptr_t>(kernels_->device()->device()) ||
      runtime.active_device.queue !=
          reinterpret_cast<std::uintptr_t>(kernels_->device()->queue()) ||
      runtime.active_device.context_id != context_identity_.context_id ||
      runtime.active_device.context_generation !=
          context_identity_.generation) {
    return fail("[open_vulkan_matting_device_mismatch] current matting "
                "readiness does not belong to the initialized shared context");
  }

  const auto validate_rgb = [&](const studiocast::vulkan::VulkanImage *image) {
    return image->Valid() && image->BelongsTo(*kernels_->device()) &&
           image->context_identity() == context_identity_ &&
           image->width() == width_ && image->height() == height_ &&
           image->format() == studiocast::vulkan::VulkanPixelFormat::rgb_u8;
  };
  if (!validate_rgb(input.foreground) || !validate_rgb(input.output)) {
    return fail("[vulkan_foreign_context] key-light RGB resources must be "
                "same-context rgb_u8 images with the configured geometry");
  }
  if (Aliases(input.foreground, input.output))
    return fail("foreground and output resources must be distinct");
  // The mapped output is the pipeline's shared final RGB transport. This
  // effect writes it only through Vulkan and never maps or host-reads it.
  if (!input.output->mapped())
    return fail("output must be the mapped final-frame transport resource");

  // A zero-intensity frame is a true no-op and must not force inference. The
  // wrapper still required production matting readiness at setup and above,
  // so pass-through is never availability evidence.
  if (parameters_.passthrough()) {
    ++counters->passthrough_frames;
    return true;
  }

  if (!input.alpha)
    return fail("resident alpha resource is null");
  if (!input.alpha->Valid() || !input.alpha->BelongsTo(*kernels_->device()) ||
      input.alpha->context_identity() != context_identity_ ||
      input.alpha->width() != width_ || input.alpha->height() != height_ ||
      input.alpha->format() != studiocast::vulkan::VulkanPixelFormat::f32_1) {
    return fail("[vulkan_foreign_context] key-light alpha must be a "
                "same-context f32_1 image with the configured geometry");
  }
  if (Aliases(input.foreground, input.alpha) ||
      Aliases(input.alpha, input.output)) {
    return fail("foreground, alpha, and output resources must be distinct");
  }
  if (input.alpha->mapped() || !input.alpha->device_local()) {
    return fail("alpha must be a non-mapped DEVICE_LOCAL resource");
  }
  if (runtime.cpu_layers_used || runtime.cpu_readback_count != 0 ||
      !runtime.input_device_resident || !runtime.alpha_device_resident ||
      !runtime.output_device_resident || !runtime.queue_ownership_explicit ||
      !runtime.synchronous_completion || runtime.inference_count == 0 ||
      runtime.completion_count <
          runtime.warmup_inference_count + runtime.inference_count ||
      input.capture_sequence == 0 ||
      input.resident_alpha_sequence != input.capture_sequence ||
      input.alpha_resize_completion_count_after == 0) {
    return fail("[open_vulkan_matting_unavailable] current-frame resident "
                "alpha inference/residency/synchronization evidence is "
                "incomplete");
  }

  if (input.matte_source ==
      OpenVulkanVirtualKeyLightMatteSource::reused_same_frame) {
    if (runtime.inference_count != input.matting_inference_count_before ||
        input.alpha_resize_completion_count_after !=
            input.alpha_resize_completion_count_before) {
      return fail("same-frame matte reuse unexpectedly performed inference "
                  "or alpha resize work");
    }
  } else if (runtime.inference_count !=
                 input.matting_inference_count_before + 1 ||
             input.alpha_resize_completion_count_after !=
                 input.alpha_resize_completion_count_before + 1) {
    return fail("independent key-light matte production must perform exactly "
                "one inference and one alpha resize");
  }

  std::string detail;
  if (!kernels_->ApplyKeyLightU8x3(
          *input.foreground, *input.alpha, parameters_.target_r,
          parameters_.target_g, parameters_.target_b, parameters_.intensity,
          parameters_.direction, *input.output, &detail)) {
    return fail(detail);
  }
  ++counters->dispatch_calls;
  if (input.matte_source ==
      OpenVulkanVirtualKeyLightMatteSource::reused_same_frame) {
    ++counters->shared_matte_reuse_calls;
  } else {
    ++counters->independent_matte_inference_calls;
  }
  return true;
}

} // namespace studiocast::video
