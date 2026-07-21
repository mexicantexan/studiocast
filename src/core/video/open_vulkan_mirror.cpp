#include "core/video/open_vulkan_mirror.h"

#include <utility>

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

OpenVulkanMirrorReadiness NotProductionReady(std::string reason_code,
                                             std::string detail) {
  OpenVulkanMirrorReadiness result;
  result.shared_reason_code = std::move(reason_code);
  result.detail = std::move(detail);
  return result;
}

std::string NestedSharedFailure(const OpenVulkanMirrorReadiness &readiness) {
  return StableFailure(readiness.shared_reason_code, readiness.detail, {});
}

bool IsDeviceLocalResident(const studiocast::vulkan::VulkanImage &image) {
  return image.Valid() && image.device_local() && !image.mapped();
}

std::string ResidencyFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanMirrorResidencyFailureReason,
                       "Open Vulkan mirror residency contract failed", detail);
}

} // namespace

std::string OpenVulkanMirrorInitializationFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanMirrorInitializationFailureReason,
                       "Open Vulkan mirror initialization failed", detail);
}

std::string OpenVulkanMirrorRuntimeFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanMirrorRuntimeFailureReason,
                       "Open Vulkan mirror runtime failed", detail);
}

OpenVulkanMirrorReadiness EvaluateOpenVulkanMirrorReadiness(
    const studiocast::vulkan::OpenVulkanDiagnostics &diagnostics) {
  if (!diagnostics.compiled_enabled) {
    return NotProductionReady("vulkan_backend_disabled_in_build",
                              "Open Vulkan support is disabled in this build");
  }
  if (!diagnostics.runtime_library_found) {
    return NotProductionReady("vulkan_runtime_not_found",
                              "the Vulkan runtime library is unavailable");
  }
  if (!diagnostics.physical_device_found) {
    return NotProductionReady("vulkan_no_physical_device",
                              "no Vulkan physical device was found");
  }
  if (diagnostics.cpu_device_selected || !diagnostics.non_cpu_device_selected) {
    return NotProductionReady(
        "vulkan_only_cpu_devices_available",
        "the selected Vulkan device is a CPU/software implementation");
  }
  if (!diagnostics.compute_queue_available) {
    return NotProductionReady(
        "vulkan_no_compute_queue",
        "the selected Vulkan device has no compute queue");
  }
  if (!diagnostics.logical_device_created) {
    return NotProductionReady("vulkan_device_create_failed",
                              "the Vulkan logical device was not created");
  }
  if (!diagnostics.context_created || !diagnostics.context_healthy) {
    const std::string reason = diagnostics.context_failure_reason.empty()
                                   ? "vulkan_context_uninitialized"
                                   : diagnostics.context_failure_reason;
    const std::string detail = diagnostics.error.empty()
                                   ? "the shared Vulkan context is not healthy"
                                   : diagnostics.error;
    return NotProductionReady(reason, detail);
  }
  if (!diagnostics.production_hardware_ready) {
    return NotProductionReady("vulkan_production_hardware_not_ready",
                              "shared Vulkan diagnostics did not prove "
                              "production hardware readiness");
  }
  if (!diagnostics.shader_pipeline_created || !diagnostics.ok) {
    return NotProductionReady(
        "open_vulkan_utility_kernels_unavailable",
        "the shared Vulkan utility shader pipeline is unavailable");
  }

  OpenVulkanMirrorReadiness result;
  result.production_ready = true;
  return result;
}

bool OpenVulkanMirror::EnsureInitialized(
    studiocast::vulkan::kernels::UtilityKernels *kernels, int width, int height,
    std::string *error_out) {
  if (error_out)
    error_out->clear();
  kernels_ = nullptr;
  width_ = 0;
  height_ = 0;
  initialized_ = false;
  if (!kernels || width <= 0 || height <= 0) {
    if (error_out) {
      *error_out = OpenVulkanMirrorInitializationFailure(
          "invalid shared context or frame dimensions");
    }
    return false;
  }
  std::string detail;
  if (!kernels->Initialized() && !kernels->Initialize(&detail)) {
    if (error_out)
      *error_out = OpenVulkanMirrorInitializationFailure(detail);
    return false;
  }
  const OpenVulkanMirrorReadiness readiness =
      EvaluateOpenVulkanMirrorReadiness(kernels->Diagnostics());
  if (!readiness.production_ready) {
    if (error_out) {
      *error_out =
          OpenVulkanMirrorInitializationFailure(NestedSharedFailure(readiness));
    }
    return false;
  }

  kernels_ = kernels;
  width_ = width;
  height_ = height;
  initialized_ = true;
  return true;
}

bool OpenVulkanMirror::ApplyFinal(const OpenVulkanMirrorFinalStageInput &input,
                                  OpenVulkanMirrorCounters *counters,
                                  std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!counters) {
    if (error_out)
      *error_out = OpenVulkanMirrorRuntimeFailure("counter output is null");
    return false;
  }
  if (!input.unmirrored_analysis_complete || !input.output_geometry_ready) {
    if (error_out) {
      *error_out = OpenVulkanMirrorRuntimeFailure(
          "final-stage ordering boundary is not satisfied");
    }
    return false;
  }
  if (!initialized_ || !kernels_) {
    if (error_out)
      *error_out = OpenVulkanMirrorRuntimeFailure("wrapper is not initialized");
    ++counters->runtime_failure_frames;
    return false;
  }
  if (!input.src || !input.dst) {
    if (error_out)
      *error_out = OpenVulkanMirrorRuntimeFailure("input or output is null");
    ++counters->runtime_failure_frames;
    return false;
  }
  const auto &src = *input.src;
  const auto &dst = *input.dst;
  if (src.width() != width_ || src.height() != height_ ||
      dst.width() != width_ || dst.height() != height_) {
    if (error_out)
      *error_out = OpenVulkanMirrorRuntimeFailure("frame dimensions changed");
    ++counters->runtime_failure_frames;
    return false;
  }
  if (!IsDeviceLocalResident(src) || !IsDeviceLocalResident(dst)) {
    if (error_out) {
      *error_out = OpenVulkanMirrorRuntimeFailure(ResidencyFailure(
          "source and output must be non-mapped DEVICE_LOCAL resources"));
    }
    ++counters->runtime_failure_frames;
    return false;
  }

  std::string detail;
  const std::uint64_t submissions_before =
      kernels_->synchronous_submission_count();
  if (!kernels_->MirrorHorizontalU8x3(src, dst, &detail)) {
    if (kernels_->synchronous_submission_count() > submissions_before)
      ++counters->dispatch_calls;
    // Preserve shared Vulkan reason codes (notably device loss and foreign
    // context) in the detail while keeping the effect-level stable reason.
    if (error_out)
      *error_out = OpenVulkanMirrorRuntimeFailure(detail);
    ++counters->runtime_failure_frames;
    return false;
  }
  ++counters->dispatch_calls;
  return true;
}

bool OpenVulkanMirror::ApplyResizeFinal(
    const OpenVulkanMirrorResizeFinalStageInput &input,
    OpenVulkanMirrorCounters *counters, std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!counters) {
    if (error_out)
      *error_out = OpenVulkanMirrorRuntimeFailure("counter output is null");
    return false;
  }
  if (!input.unmirrored_analysis_complete) {
    if (error_out) {
      *error_out = OpenVulkanMirrorRuntimeFailure(
          "final-stage ordering boundary is not satisfied");
    }
    return false;
  }
  if (!initialized_ || !kernels_) {
    if (error_out)
      *error_out = OpenVulkanMirrorRuntimeFailure("wrapper is not initialized");
    ++counters->runtime_failure_frames;
    return false;
  }
  if (!input.src || !input.resized || !input.dst) {
    if (error_out) {
      *error_out = OpenVulkanMirrorRuntimeFailure(
          "input, resize scratch, or output is null");
    }
    ++counters->runtime_failure_frames;
    return false;
  }
  if (input.resized->width() != width_ || input.resized->height() != height_ ||
      input.dst->width() != width_ || input.dst->height() != height_) {
    if (error_out)
      *error_out = OpenVulkanMirrorRuntimeFailure("frame dimensions changed");
    ++counters->runtime_failure_frames;
    return false;
  }
  if (!IsDeviceLocalResident(*input.src) ||
      !IsDeviceLocalResident(*input.resized) ||
      !IsDeviceLocalResident(*input.dst)) {
    if (error_out) {
      *error_out = OpenVulkanMirrorRuntimeFailure(ResidencyFailure(
          "source, resize scratch, and output must be non-mapped "
          "DEVICE_LOCAL resources"));
    }
    ++counters->runtime_failure_frames;
    return false;
  }

  std::string detail;
  const std::uint64_t submissions_before =
      kernels_->synchronous_submission_count();
  if (!kernels_->ResizeMirrorHorizontalU8x3(*input.src, *input.resized,
                                            *input.dst, &detail)) {
    if (kernels_->synchronous_submission_count() > submissions_before)
      ++counters->dispatch_calls;
    if (error_out)
      *error_out = OpenVulkanMirrorRuntimeFailure(detail);
    ++counters->runtime_failure_frames;
    return false;
  }
  ++counters->dispatch_calls;
  return true;
}

} // namespace studiocast::video
