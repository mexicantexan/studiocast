#include "core/video/open_vulkan_virtual_background_blur.h"

#include <algorithm>
#include <array>

#include "core/video/effects/broadcast_effect_contract.h"

namespace studiocast::video {
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

OpenVulkanVirtualBackgroundBlurReadiness NotReady(std::string reason_code,
                                                  std::string detail) {
  OpenVulkanVirtualBackgroundBlurReadiness result;
  result.reason_code = std::move(reason_code);
  result.detail = std::move(detail);
  return result;
}

} // namespace

std::string
OpenVulkanVirtualBackgroundBlurInitializationFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanVirtualBackgroundBlurInitializationReason,
                       "Open Vulkan virtual background blur initialization "
                       "failed",
                       detail);
}

std::string
OpenVulkanVirtualBackgroundBlurRuntimeFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanVirtualBackgroundBlurRuntimeReason,
                       "Open Vulkan virtual background blur runtime failed",
                       detail);
}

OpenVulkanVirtualBackgroundBlurParameters
ResolveOpenVulkanVirtualBackgroundBlurParameters(int strength) {
  OpenVulkanVirtualBackgroundBlurParameters parameters;
  parameters.background_radius =
      std::clamp(strength, effects::contract::kVbStrengthMin,
                 effects::contract::kVbStrengthMax);
  parameters.alpha_feather_radius =
      std::min(4, parameters.background_radius / 16);
  return parameters;
}

OpenVulkanVirtualBackgroundBlurReadiness
EvaluateOpenVulkanVirtualBackgroundBlurReadiness(
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

  OpenVulkanVirtualBackgroundBlurReadiness result;
  result.production_ready = true;
  return result;
}

void OpenVulkanVirtualBackgroundBlur::Shutdown() noexcept {
  kernels_ = nullptr;
  context_identity_ = {};
  width_ = 0;
  height_ = 0;
  initialized_ = false;
}

bool OpenVulkanVirtualBackgroundBlur::EnsureInitialized(
    studiocast::vulkan::kernels::UtilityKernels *kernels, int width, int height,
    const studiocast::open_vulkan::VulkanMattingReadiness &matting_readiness,
    std::string *error_out) {
  if (error_out)
    error_out->clear();
  auto fail = [&](std::string detail) {
    Shutdown();
    if (error_out) {
      *error_out = OpenVulkanVirtualBackgroundBlurInitializationFailure(detail);
    }
    return false;
  };
  if (!kernels || width <= 0 || height <= 0)
    return fail("invalid shared context or frame dimensions");

  std::string detail;
  if (!kernels->Initialized() && !kernels->Initialize(&detail))
    return fail(detail);
  const auto readiness = EvaluateOpenVulkanVirtualBackgroundBlurReadiness(
      kernels->Diagnostics(), matting_readiness);
  if (!readiness.production_ready) {
    return fail(StableFailure(readiness.reason_code, readiness.detail, {}));
  }
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

  if (initialized_ && kernels_ == kernels && width_ == width &&
      height_ == height && context_identity_ == identity) {
    return true;
  }
  kernels_ = kernels;
  context_identity_ = identity;
  width_ = width;
  height_ = height;
  initialized_ = true;
  return true;
}

bool OpenVulkanVirtualBackgroundBlur::Apply(
    const OpenVulkanVirtualBackgroundBlurInput &input,
    OpenVulkanVirtualBackgroundBlurCounters *counters, std::string *error_out) {
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
      *error_out = OpenVulkanVirtualBackgroundBlurRuntimeFailure(detail);
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
  if (!input.foreground || !input.alpha || !input.alpha_tmp ||
      !input.alpha_feathered || !input.blur_tmp || !input.blurred ||
      !input.output || !input.matting_readiness) {
    return fail("one or more resident effect resources are null");
  }
  const auto &matting = *input.matting_readiness;
  const auto &runtime = matting.runtime;
  const auto current_readiness =
      EvaluateOpenVulkanVirtualBackgroundBlurReadiness(kernels_->Diagnostics(),
                                                       matting);
  if (!current_readiness.production_ready) {
    return fail(StableFailure(current_readiness.reason_code,
                              current_readiness.detail, {}));
  }
  if (runtime.cpu_layers_used || runtime.cpu_readback_count != 0 ||
      !runtime.input_device_resident || !runtime.alpha_device_resident ||
      !runtime.output_device_resident || !runtime.queue_ownership_explicit ||
      !runtime.synchronous_completion || runtime.inference_count == 0 ||
      runtime.completion_count <
          runtime.warmup_inference_count + runtime.inference_count ||
      input.capture_sequence == 0 ||
      input.resident_alpha_sequence != input.capture_sequence ||
      input.alpha_resize_completion_count == 0) {
    return fail("[open_vulkan_matting_unavailable] current-frame resident "
                "alpha inference/residency/synchronization evidence is "
                "incomplete");
  }
  if (runtime.active_device.ownership_domain != kernels_->device() ||
      runtime.active_device.logical_device !=
          reinterpret_cast<std::uintptr_t>(kernels_->device()->device()) ||
      runtime.active_device.queue !=
          reinterpret_cast<std::uintptr_t>(kernels_->device()->queue()) ||
      runtime.active_device.context_id != context_identity_.context_id ||
      runtime.active_device.context_generation !=
          context_identity_.generation) {
    return fail("[open_vulkan_matting_device_mismatch] current-frame matte "
                "does not belong to the initialized shared context");
  }

  const std::array<const studiocast::vulkan::VulkanImage *, 7> images = {
      input.foreground, input.alpha,   input.alpha_tmp, input.alpha_feathered,
      input.blur_tmp,   input.blurred, input.output};
  for (const auto *image : images) {
    if (!image->Valid())
      return fail("one or more resident effect resources are invalid");
    if (!image->BelongsTo(*kernels_->device()) ||
        image->context_identity() != context_identity_) {
      return fail("[vulkan_foreign_context] blur resources do not belong to "
                  "the initialized shared context");
    }
    if (image->width() != width_ || image->height() != height_)
      return fail("frame dimensions changed");
  }
  const auto aliases = [](const auto *left, const auto *right) {
    return left == right || left->buffer() == right->buffer();
  };
  if (aliases(input.foreground, input.output) ||
      aliases(input.foreground, input.blur_tmp) ||
      aliases(input.foreground, input.blurred) ||
      aliases(input.blur_tmp, input.blurred) ||
      aliases(input.blur_tmp, input.output) ||
      aliases(input.blurred, input.output) ||
      aliases(input.alpha, input.alpha_tmp) ||
      aliases(input.alpha, input.alpha_feathered) ||
      aliases(input.alpha_tmp, input.alpha_feathered)) {
    return fail("blur, alpha scratch, and output resources must be distinct");
  }
  if (input.foreground->format() !=
          studiocast::vulkan::VulkanPixelFormat::rgb_u8 ||
      input.blur_tmp->format() !=
          studiocast::vulkan::VulkanPixelFormat::rgb_u8 ||
      input.blurred->format() !=
          studiocast::vulkan::VulkanPixelFormat::rgb_u8 ||
      input.output->format() != studiocast::vulkan::VulkanPixelFormat::rgb_u8) {
    return fail("canonical live foreground, blur scratch, and output formats "
                "must be rgb_u8");
  }
  if (input.alpha->format() != studiocast::vulkan::VulkanPixelFormat::f32_1 ||
      input.alpha_tmp->format() !=
          studiocast::vulkan::VulkanPixelFormat::f32_1 ||
      input.alpha_feathered->format() !=
          studiocast::vulkan::VulkanPixelFormat::f32_1) {
    return fail("alpha and alpha scratch formats must be f32_1");
  }
  if (input.alpha->mapped() || input.alpha_tmp->mapped() ||
      input.alpha_feathered->mapped() || !input.alpha->device_local() ||
      !input.alpha_tmp->device_local() ||
      !input.alpha_feathered->device_local() || input.blur_tmp->mapped() ||
      input.blurred->mapped() || !input.blur_tmp->device_local() ||
      !input.blurred->device_local()) {
    return fail("alpha and blur scratch must be non-mapped DEVICE_LOCAL "
                "resources");
  }
  // The camera's output transport is intentionally host-visible for its one
  // final RGB readback. This effect writes it only through Vulkan and never
  // invalidates, maps, or reads it on the host.
  if (!input.output->mapped()) {
    return fail("output must be the mapped final-frame transport resource");
  }

  const auto parameters =
      ResolveOpenVulkanVirtualBackgroundBlurParameters(input.strength);
  const studiocast::vulkan::VulkanImage *alpha_use = input.alpha;
  std::string detail;
  if (parameters.alpha_feather_radius > 0) {
    if (!kernels_->BoxBlurSeparableF32_1(
            *input.alpha, *input.alpha_tmp, *input.alpha_feathered,
            parameters.alpha_feather_radius, &detail)) {
      return fail(detail);
    }
    ++counters->alpha_feather_dispatch_calls;
    alpha_use = input.alpha_feathered;
  }

  if (!kernels_->BoxBlurCompositeAlphaU8x3(
          *input.foreground, *input.blur_tmp, *input.blurred, *alpha_use,
          *input.output, parameters.background_radius, &detail)) {
    return fail(detail);
  }
  ++counters->blur_composite_dispatch_calls;
  return true;
}

} // namespace studiocast::video
