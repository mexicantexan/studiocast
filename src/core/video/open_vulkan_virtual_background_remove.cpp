#include "core/video/open_vulkan_virtual_background_remove.h"

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

OpenVulkanVirtualBackgroundRemoveReadiness NotReady(std::string reason_code,
                                                    std::string detail) {
  OpenVulkanVirtualBackgroundRemoveReadiness result;
  result.reason_code = std::move(reason_code);
  result.detail = std::move(detail);
  return result;
}

bool ParseCompatibleRgbHex(std::string_view value, std::uint32_t *rgb) {
  if (!rgb)
    return false;
  *rgb = 0;
  if (value.starts_with("0x") || value.starts_with("0X"))
    value.remove_prefix(2);
  if (!value.empty() && value.front() == '#')
    value.remove_prefix(1);
  if (value.size() != 6)
    return false;

  std::uint32_t parsed = 0;
  for (const char ch : value) {
    parsed <<= 4u;
    if (ch >= '0' && ch <= '9')
      parsed |= static_cast<std::uint32_t>(ch - '0');
    else if (ch >= 'a' && ch <= 'f')
      parsed |= static_cast<std::uint32_t>(ch - 'a' + 10);
    else if (ch >= 'A' && ch <= 'F')
      parsed |= static_cast<std::uint32_t>(ch - 'A' + 10);
    else
      return false;
  }
  *rgb = parsed & 0xFFFFFFu;
  return true;
}

} // namespace

std::string OpenVulkanVirtualBackgroundRemoveInitializationFailure(
    std::string_view detail) {
  return StableFailure(kOpenVulkanVirtualBackgroundRemoveInitializationReason,
                       "Open Vulkan virtual background remove initialization "
                       "failed",
                       detail);
}

std::string
OpenVulkanVirtualBackgroundRemoveRuntimeFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanVirtualBackgroundRemoveRuntimeReason,
                       "Open Vulkan virtual background remove runtime failed",
                       detail);
}

OpenVulkanVirtualBackgroundRemoveParameters
ResolveOpenVulkanVirtualBackgroundRemoveParameters(
    int strength, std::string_view remove_color) {
  OpenVulkanVirtualBackgroundRemoveParameters parameters;
  const int clamped_strength =
      std::clamp(strength, effects::contract::kVbStrengthMin,
                 effects::contract::kVbStrengthMax);
  parameters.alpha_feather_radius = std::min(4, clamped_strength / 16);

  std::uint32_t rgb = 0;
  if (!ParseCompatibleRgbHex(remove_color, &rgb))
    rgb = 0;
  parameters.background_r = static_cast<std::uint8_t>((rgb >> 16u) & 0xFFu);
  parameters.background_g = static_cast<std::uint8_t>((rgb >> 8u) & 0xFFu);
  parameters.background_b = static_cast<std::uint8_t>(rgb & 0xFFu);
  return parameters;
}

OpenVulkanVirtualBackgroundRemoveReadiness
EvaluateOpenVulkanVirtualBackgroundRemoveReadiness(
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

  OpenVulkanVirtualBackgroundRemoveReadiness result;
  result.production_ready = true;
  return result;
}

void OpenVulkanVirtualBackgroundRemove::Shutdown() noexcept {
  kernels_ = nullptr;
  context_identity_ = {};
  parameters_ = {};
  width_ = 0;
  height_ = 0;
  initialized_ = false;
}

bool OpenVulkanVirtualBackgroundRemove::EnsureInitialized(
    studiocast::vulkan::kernels::UtilityKernels *kernels, int width, int height,
    int strength, std::string_view remove_color,
    const studiocast::open_vulkan::VulkanMattingReadiness &matting_readiness,
    std::string *error_out) {
  if (error_out)
    error_out->clear();
  auto fail = [&](std::string detail) {
    Shutdown();
    if (error_out) {
      *error_out =
          OpenVulkanVirtualBackgroundRemoveInitializationFailure(detail);
    }
    return false;
  };
  if (!kernels || width <= 0 || height <= 0)
    return fail("invalid shared context or frame dimensions");

  std::string detail;
  if (!kernels->Initialized() && !kernels->Initialize(&detail))
    return fail(detail);
  const auto readiness = EvaluateOpenVulkanVirtualBackgroundRemoveReadiness(
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
  parameters_ = ResolveOpenVulkanVirtualBackgroundRemoveParameters(
      strength, remove_color);
  width_ = width;
  height_ = height;
  initialized_ = true;
  return true;
}

bool OpenVulkanVirtualBackgroundRemove::Apply(
    const OpenVulkanVirtualBackgroundRemoveInput &input,
    OpenVulkanVirtualBackgroundRemoveCounters *counters,
    std::string *error_out) {
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
      *error_out = OpenVulkanVirtualBackgroundRemoveRuntimeFailure(detail);
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
      !input.alpha_feathered || !input.output || !input.matting_readiness) {
    return fail("one or more resident effect resources are null");
  }

  const auto &matting = *input.matting_readiness;
  const auto &runtime = matting.runtime;
  const auto current_readiness =
      EvaluateOpenVulkanVirtualBackgroundRemoveReadiness(
          kernels_->Diagnostics(), matting);
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

  const std::array<const studiocast::vulkan::VulkanImage *, 5> images = {
      input.foreground, input.alpha, input.alpha_tmp, input.alpha_feathered,
      input.output};
  for (const auto *image : images) {
    if (!image->Valid())
      return fail("one or more resident effect resources are invalid");
    if (!image->BelongsTo(*kernels_->device()) ||
        image->context_identity() != context_identity_) {
      return fail("[vulkan_foreign_context] remove resources do not belong to "
                  "the initialized shared context");
    }
    if (image->width() != width_ || image->height() != height_)
      return fail("frame dimensions changed");
  }
  for (std::size_t left = 0; left < images.size(); ++left) {
    for (std::size_t right = left + 1; right < images.size(); ++right) {
      if (images[left] == images[right] ||
          images[left]->buffer() == images[right]->buffer()) {
        return fail("foreground, alpha scratch, and output resources must be "
                    "distinct");
      }
    }
  }
  if (input.foreground->format() !=
          studiocast::vulkan::VulkanPixelFormat::rgb_u8 ||
      input.output->format() != studiocast::vulkan::VulkanPixelFormat::rgb_u8) {
    return fail("canonical live foreground and output formats must be rgb_u8");
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
      !input.alpha_feathered->device_local()) {
    return fail("alpha and alpha scratch must be non-mapped DEVICE_LOCAL "
                "resources");
  }
  // The mapped output is the pipeline's one final RGB transport. This effect
  // only writes it through Vulkan and never maps, invalidates, or host-reads it.
  if (!input.output->mapped())
    return fail("output must be the mapped final-frame transport resource");

  const studiocast::vulkan::VulkanImage *alpha_use = input.alpha;
  std::string detail;
  if (parameters_.alpha_feather_radius > 0) {
    if (!kernels_->BoxBlurSeparableF32_1(
            *input.alpha, *input.alpha_tmp, *input.alpha_feathered,
            parameters_.alpha_feather_radius, &detail)) {
      return fail(detail);
    }
    ++counters->alpha_feather_dispatch_calls;
    alpha_use = input.alpha_feathered;
  }
  if (!kernels_->CompositeAlphaSolidU8x3(
          *input.foreground, *alpha_use, parameters_.background_r,
          parameters_.background_g, parameters_.background_b, *input.output,
          &detail)) {
    return fail(detail);
  }
  ++counters->solid_composite_dispatch_calls;
  return true;
}

} // namespace studiocast::video
