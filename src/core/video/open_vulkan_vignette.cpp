#include "core/video/open_vulkan_vignette.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

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

OpenVulkanVignetteReadiness NotProductionReady(std::string reason_code,
                                               std::string detail) {
  OpenVulkanVignetteReadiness result;
  result.shared_reason_code = std::move(reason_code);
  result.detail = std::move(detail);
  return result;
}

std::string NestedSharedFailure(const OpenVulkanVignetteReadiness &readiness) {
  return StableFailure(readiness.shared_reason_code, readiness.detail, {});
}

bool FinalImageMatches(const studiocast::vulkan::VulkanImage *image, int width,
                       int height) {
  return image && image->width() == width && image->height() == height;
}

bool IsDeviceLocalResident(const studiocast::vulkan::VulkanImage &image) {
  return image.Valid() && image.device_local() && !image.mapped();
}

std::string ResidencyFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanVignetteResidencyFailureReason,
                       "Open Vulkan vignette residency contract failed",
                       detail);
}

} // namespace

std::string OpenVulkanVignetteInitializationFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanVignetteInitializationFailureReason,
                       "Open Vulkan vignette initialization failed", detail);
}

std::string OpenVulkanVignetteRuntimeFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanVignetteRuntimeFailureReason,
                       "Open Vulkan vignette runtime failed", detail);
}

OpenVulkanVignetteReadiness EvaluateOpenVulkanVignetteReadiness(
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
  OpenVulkanVignetteReadiness result;
  result.production_ready = true;
  return result;
}

namespace detail {

float OpenVulkanVignetteRadialSquaredAt(int x, int y, int width, int height,
                                        float center_x_px, float center_y_px) {
  if (width <= 0 || height <= 0 || x < 0 || y < 0 || x >= width ||
      y >= height || !std::isfinite(center_x_px) ||
      !std::isfinite(center_y_px)) {
    return 0.0f;
  }
  const float inv_half_w = 2.0f / static_cast<float>(width);
  const float inv_half_h = 2.0f / static_cast<float>(height);
  const float fx = (static_cast<float>(x) + 0.5f - center_x_px) * inv_half_w;
  const float fy = (static_cast<float>(y) + 0.5f - center_y_px) * inv_half_h;
  // Keep the exact CUDA-reference operation order and constant. Squaring the
  // clamped radius (instead of simplifying to 0.5*(fx^2+fy^2)) preserves its
  // floating-point and off-center saturation behavior.
  float radius = std::sqrt(fx * fx + fy * fy) * 0.70710677f;
  radius = std::clamp(radius, 0.0f, 1.0f);
  return radius * radius;
}

} // namespace detail

void OpenVulkanVignette::Shutdown() noexcept {
  attenuation_factor_.Free();
  kernels_ = nullptr;
  width_ = 0;
  height_ = 0;
  intensity_percent_ = 0;
  initialized_ = false;
}

bool OpenVulkanVignette::EnsureInitialized(
    studiocast::vulkan::kernels::UtilityKernels *kernels, int width, int height,
    int intensity_percent, OpenVulkanVignetteCounters *counters,
    std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!counters) {
    if (error_out) {
      *error_out =
          OpenVulkanVignetteInitializationFailure("counter output is null");
    }
    return false;
  }
  if (!kernels || width <= 0 || height <= 0) {
    Shutdown();
    if (error_out) {
      *error_out = OpenVulkanVignetteInitializationFailure(
          "invalid shared context or frame dimensions");
    }
    return false;
  }
  intensity_percent = std::clamp(intensity_percent, 0, 100);
  std::string detail;
  if (!kernels->Initialized() && !kernels->Initialize(&detail)) {
    if (error_out)
      *error_out = OpenVulkanVignetteInitializationFailure(detail);
    return false;
  }
  const OpenVulkanVignetteReadiness readiness =
      EvaluateOpenVulkanVignetteReadiness(kernels->Diagnostics());
  if (!readiness.production_ready) {
    if (error_out) {
      *error_out = OpenVulkanVignetteInitializationFailure(
          NestedSharedFailure(readiness));
    }
    return false;
  }
  const bool reusable_factor =
      initialized_ && kernels_ == kernels && width_ == width &&
      height_ == height && attenuation_factor_.Valid() &&
      attenuation_factor_.BelongsTo(*kernels->device()) &&
      IsDeviceLocalResident(attenuation_factor_);
  if (reusable_factor && intensity_percent_ == intensity_percent) {
    return true;
  }
  if (!reusable_factor) {
    Shutdown();
    if (!attenuation_factor_.Allocate(
            kernels->device(), width, height,
            studiocast::vulkan::VulkanPixelFormat::f32_1,
            /*map_memory=*/false, &detail)) {
      if (error_out)
        *error_out = OpenVulkanVignetteInitializationFailure(detail);
      return false;
    }
    ++counters->factor_allocation_calls;
  }

  if (!IsDeviceLocalResident(attenuation_factor_)) {
    Shutdown();
    if (error_out) {
      *error_out = OpenVulkanVignetteInitializationFailure(
          ResidencyFailure(
              "attenuation-factor mask is not non-mapped DEVICE_LOCAL"));
    }
    return false;
  }
  const std::size_t factor_count = static_cast<std::size_t>(width) *
                                   static_cast<std::size_t>(height);
  std::vector<float> factors(factor_count);
  const float center_x = static_cast<float>(width) * 0.5f;
  const float center_y = static_cast<float>(height) * 0.5f;
  const float intensity = static_cast<float>(intensity_percent) / 100.0f;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float radial_squared = detail::OpenVulkanVignetteRadialSquaredAt(
          x, y, width, height, center_x, center_y);
      factors[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
              static_cast<std::size_t>(x)] =
          std::max(0.0f, 1.0f - intensity * radial_squared);
    }
  }
  ++counters->factor_generation_calls;
  studiocast::vulkan::VulkanImage upload_staging;
  if (!upload_staging.Allocate(
          kernels->device(), width, height,
          studiocast::vulkan::VulkanPixelFormat::f32_1,
          /*map_memory=*/true, &detail)) {
    Shutdown();
    if (error_out)
      *error_out = OpenVulkanVignetteInitializationFailure(detail);
    return false;
  }
  ++counters->factor_staging_allocation_calls;
  if (!kernels->UploadF32_1ToDeviceLocal(
          factors.data(), factors.size(), upload_staging, attenuation_factor_,
          &detail)) {
    Shutdown();
    if (error_out)
      *error_out = OpenVulkanVignetteInitializationFailure(detail);
    return false;
  }
  ++counters->factor_upload_calls;
  ++counters->factor_upload_completion_calls;

  if (!reusable_factor) {
    // The old mask was destroyed before this allocation. Vulkan is allowed to
    // reuse its opaque buffer handle, which is insufficient for the utility
    // kernel's raw-handle descriptor cache to recognize the new resource.
    // Invalidate all three descriptor tuples once at setup/reconfiguration;
    // stable frames then retain their descriptor bindings without updates.
    kernels->InvalidateDescriptorBindingCacheForSetup();
  }

  kernels_ = kernels;
  width_ = width;
  height_ = height;
  intensity_percent_ = intensity_percent;
  initialized_ = true;
  return true;
}

bool OpenVulkanVignette::ApplyFinal(
    const OpenVulkanVignetteFinalStageInput &input,
    OpenVulkanVignetteCounters *counters, std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!counters) {
    if (error_out)
      *error_out = OpenVulkanVignetteRuntimeFailure("counter output is null");
    return false;
  }
  if (!input.preceding_effects_complete) {
    if (error_out) {
      *error_out = OpenVulkanVignetteRuntimeFailure(
          "final-stage ordering boundary is not satisfied");
    }
    return false;
  }
  if (!initialized_ || !kernels_ || !attenuation_factor_.Valid()) {
    if (error_out) {
      *error_out =
          OpenVulkanVignetteRuntimeFailure("wrapper is not initialized");
    }
    ++counters->runtime_failure_frames;
    return false;
  }
  if (!IsDeviceLocalResident(attenuation_factor_)) {
    if (error_out) {
      *error_out = OpenVulkanVignetteRuntimeFailure(ResidencyFailure(
          "persistent attenuation-factor mask is not non-mapped "
          "DEVICE_LOCAL"));
    }
    ++counters->runtime_failure_frames;
    return false;
  }
  if (!input.src || !input.dst) {
    if (error_out)
      *error_out = OpenVulkanVignetteRuntimeFailure("input or output is null");
    ++counters->runtime_failure_frames;
    return false;
  }
  const auto &final_source =
      input.resize_scratch ? *input.resize_scratch : *input.src;
  if (!FinalImageMatches(&final_source, width_, height_) ||
      !FinalImageMatches(input.dst, width_, height_) ||
      (input.mirrored_dst &&
       !FinalImageMatches(input.mirrored_dst, width_, height_))) {
    if (error_out)
      *error_out = OpenVulkanVignetteRuntimeFailure("frame dimensions changed");
    ++counters->runtime_failure_frames;
    return false;
  }
  if (!IsDeviceLocalResident(*input.src) ||
      (input.resize_scratch &&
       !IsDeviceLocalResident(*input.resize_scratch)) ||
      !IsDeviceLocalResident(*input.dst) ||
      (input.mirrored_dst && !IsDeviceLocalResident(*input.mirrored_dst))) {
    if (error_out) {
      *error_out = OpenVulkanVignetteRuntimeFailure(ResidencyFailure(
          "source, resize scratch, vignette output, and optional mirror "
          "output must all be non-mapped DEVICE_LOCAL resources"));
    }
    ++counters->runtime_failure_frames;
    return false;
  }
  if (std::clamp(input.intensity_percent, 0, 100) != intensity_percent_) {
    if (error_out) {
      *error_out = OpenVulkanVignetteRuntimeFailure("effect intensity changed");
    }
    ++counters->runtime_failure_frames;
    return false;
  }

  std::string detail;
  const std::uint64_t submissions_before =
      kernels_->synchronous_submission_count();
  if (!kernels_->ApplyFinalVignetteU8x3(*input.src, input.resize_scratch,
                                        attenuation_factor_, *input.dst,
                                        input.mirrored_dst, &detail)) {
    if (kernels_->synchronous_submission_count() > submissions_before)
      ++counters->dispatch_calls;
    if (error_out)
      *error_out = OpenVulkanVignetteRuntimeFailure(detail);
    ++counters->runtime_failure_frames;
    return false;
  }
  ++counters->dispatch_calls;
  return true;
}

} // namespace studiocast::video
