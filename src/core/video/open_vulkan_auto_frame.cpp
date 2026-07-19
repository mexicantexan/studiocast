#include "core/video/open_vulkan_auto_frame.h"

#include <cmath>

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

} // namespace

std::string OpenVulkanAutoFrameInitializationFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanAutoFrameInitializationReason,
                       "Open Vulkan Auto Frame initialization failed", detail);
}

std::string OpenVulkanAutoFrameRuntimeFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanAutoFrameRuntimeReason,
                       "Open Vulkan Auto Frame runtime failed", detail);
}

std::string OpenVulkanAutoFrameFaceProviderFailure(std::string_view detail) {
  return StableFailure(kOpenVulkanAutoFrameFaceProviderReason,
                       "CPU-only YuNet tracking provider is unavailable",
                       detail);
}

const char *OpenVulkanAutoFrameResetReasonName(
    OpenVulkanAutoFrameResetReason reason) {
  switch (reason) {
  case OpenVulkanAutoFrameResetReason::enablement_or_effect_generation:
    return "enablement_or_effect_generation";
  case OpenVulkanAutoFrameResetReason::capture_sequence_discontinuity:
    return "capture_sequence_discontinuity";
  case OpenVulkanAutoFrameResetReason::geometry_model_or_provider_change:
    return "geometry_model_or_provider_change";
  case OpenVulkanAutoFrameResetReason::vulkan_context_generation_change:
    return "vulkan_context_generation_change";
  case OpenVulkanAutoFrameResetReason::runtime_or_device_failure:
    return "runtime_or_device_failure";
  case OpenVulkanAutoFrameResetReason::pipeline_restart:
    return "pipeline_restart";
  case OpenVulkanAutoFrameResetReason::none:
  default:
    return "none";
  }
}

bool OpenVulkanAutoFrameSequenceNeedsReset(std::uint64_t previous_sequence,
                                           std::uint64_t next_sequence) {
  if (previous_sequence == 0 || next_sequence == previous_sequence)
    return false;
  return next_sequence != previous_sequence + 1;
}

OpenVulkanAutoFrameResetReason OpenVulkanAutoFrameReuseKeyResetReason(
    const OpenVulkanAutoFrameReuseKey &previous,
    const OpenVulkanAutoFrameReuseKey &next) {
  if (!previous.observed || !next.observed)
    return OpenVulkanAutoFrameResetReason::none;
  if (previous.enabled != next.enabled ||
      previous.effects_generation != next.effects_generation) {
    return OpenVulkanAutoFrameResetReason::
        enablement_or_effect_generation;
  }
  if (previous.frame_width != next.frame_width ||
      previous.frame_height != next.frame_height ||
      previous.model_id != next.model_id ||
      previous.provider_id != next.provider_id) {
    return OpenVulkanAutoFrameResetReason::
        geometry_model_or_provider_change;
  }
  if (previous.context_identity != next.context_identity) {
    return OpenVulkanAutoFrameResetReason::
        vulkan_context_generation_change;
  }
  return OpenVulkanAutoFrameResetReason::none;
}

void OpenVulkanAutoFrame::Shutdown() noexcept {
  kernels_ = nullptr;
  context_identity_ = {};
  width_ = 0;
  height_ = 0;
  initialized_ = false;
}

void OpenVulkanAutoFrame::ResetTemporal(
    OpenVulkanAutoFrameResetReason reason,
    OpenVulkanAutoFrameCounters *counters) noexcept {
  if (reason != OpenVulkanAutoFrameResetReason::none && counters)
    ++counters->temporal_reset_calls;
}

bool OpenVulkanAutoFrame::EnsureInitialized(
    studiocast::vulkan::kernels::UtilityKernels *kernels, int width, int height,
    OpenVulkanAutoFrameCounters *counters, std::string *error_out) {
  if (error_out)
    error_out->clear();
  auto fail = [&](std::string detail) {
    Shutdown();
    if (counters)
      ++counters->initialization_failures;
    if (error_out)
      *error_out = OpenVulkanAutoFrameInitializationFailure(detail);
    return false;
  };
  if (!counters)
    return fail("counter output is null");
  if (!kernels || width <= 0 || height <= 0)
    return fail("invalid shared context or frame dimensions");

  std::string detail;
  if (!kernels->Initialized() && !kernels->Initialize(&detail))
    return fail(detail);
  const auto diagnostics = kernels->Diagnostics();
  if (!diagnostics.production_hardware_ready || !diagnostics.context_healthy ||
      !diagnostics.shader_pipeline_created || !diagnostics.ok) {
    const std::string reason = diagnostics.context_failure_reason.empty()
                                   ? "open_vulkan_utility_kernels_unavailable"
                                   : diagnostics.context_failure_reason;
    return fail(StableFailure(reason,
                              diagnostics.error.empty()
                                  ? "shared Vulkan production context is not ready"
                                  : diagnostics.error,
                              {}));
  }
  if (!kernels->device() || !kernels->device()->context_identity().Valid())
    return fail("shared Vulkan context identity is invalid");

  const auto identity = kernels->device()->context_identity();
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

bool OpenVulkanAutoFrame::ApplyCrop(
    const OpenVulkanAutoFrameCropInput &input,
    OpenVulkanAutoFrameCounters *counters, std::string *error_out) {
  if (error_out)
    error_out->clear();
  auto fail = [&](std::string detail, bool device_lost) {
    if (counters) {
      ++counters->runtime_failure_frames;
      if (device_lost)
        ++counters->device_loss_frames;
    }
    if (error_out)
      *error_out = OpenVulkanAutoFrameRuntimeFailure(detail);
    return false;
  };
  if (!counters)
    return fail("counter output is null", false);
  if (!input.host_analysis_complete || !input.cpu_crop_plan_complete) {
    return fail("host-analysis/crop-plan ordering boundary is not satisfied",
                false);
  }
  if (!initialized_ || !kernels_)
    return fail("wrapper is not initialized", false);
  if (!input.src || !input.dst)
    return fail("input or output is null", false);
  if (input.src == input.dst)
    return fail("crop/resize must be out of place", false);
  if (input.src->width() != width_ || input.src->height() != height_ ||
      input.dst->width() != width_ || input.dst->height() != height_)
    return fail("frame dimensions changed", false);
  if (!input.src->BelongsTo(*kernels_->device()) ||
      !input.dst->BelongsTo(*kernels_->device()) ||
      input.src->context_identity() != context_identity_ ||
      input.dst->context_identity() != context_identity_) {
    return fail("[vulkan_resource_foreign_context] crop resources do not "
                "belong to the initialized shared context",
                false);
  }
  if (!std::isfinite(input.crop_x) || !std::isfinite(input.crop_y) ||
      !std::isfinite(input.crop_w) || !std::isfinite(input.crop_h) ||
      input.crop_w <= 0.0f || input.crop_h <= 0.0f) {
    return fail("invalid CPU crop plan", false);
  }

  std::string detail;
  const std::uint64_t submissions_before =
      kernels_->synchronous_submission_count();
  if (!kernels_->CropResizeBilinear(
          *input.src, *input.dst, input.crop_x, input.crop_y, input.crop_w,
          input.crop_h, &detail)) {
    if (kernels_->synchronous_submission_count() > submissions_before)
      ++counters->crop_resize_dispatch_calls;
    const bool device_lost =
        detail.find("[vulkan_device_lost]") != std::string::npos;
    return fail(detail, device_lost);
  }
  ++counters->crop_resize_dispatch_calls;
  return true;
}

} // namespace studiocast::video
