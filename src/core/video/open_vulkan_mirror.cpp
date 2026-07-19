#include "core/video/open_vulkan_mirror.h"

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

std::string OpenVulkanMirrorInitializationFailure(std::string_view detail) {
  return StableFailure(kVulkanEffectInitializationFailed,
                       "Open Vulkan mirror initialization failed", detail);
}

std::string OpenVulkanMirrorRuntimeFailure(std::string_view detail) {
  return StableFailure(kVulkanEffectRuntimeFailed,
                       "Open Vulkan mirror runtime failed", detail);
}

bool OpenVulkanMirror::EnsureInitialized(
    studiocast::vulkan::kernels::UtilityKernels *kernels, int width, int height,
    std::string *error_out) {
  if (error_out)
    error_out->clear();
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

} // namespace studiocast::video
