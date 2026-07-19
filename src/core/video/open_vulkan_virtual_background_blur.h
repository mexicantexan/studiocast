#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "core/open_video/vulkan_matting_runtime.h"
#include "core/vulkan/kernels/utility_kernels.h"

namespace studiocast::video {

inline constexpr std::string_view
    kOpenVulkanVirtualBackgroundBlurInitializationReason =
        "vulkan_virtual_background_blur_initialization_failed";
inline constexpr std::string_view
    kOpenVulkanVirtualBackgroundBlurRuntimeReason =
        "vulkan_virtual_background_blur_runtime_failed";

std::string
OpenVulkanVirtualBackgroundBlurInitializationFailure(std::string_view detail);
std::string
OpenVulkanVirtualBackgroundBlurRuntimeFailure(std::string_view detail);

struct OpenVulkanVirtualBackgroundBlurParameters {
  int background_radius = 0;
  int alpha_feather_radius = 0;
};

// Preserves the canonical public 1..64 strength semantics shared with the
// Open CUDA implementation. This is pure so endpoint/default parity is
// testable without treating a utility kernel as production availability.
OpenVulkanVirtualBackgroundBlurParameters
ResolveOpenVulkanVirtualBackgroundBlurParameters(int strength);

struct OpenVulkanVirtualBackgroundBlurReadiness {
  bool production_ready = false;
  std::string reason_code;
  std::string detail;
};

// Blur readiness requires both production hardware/utility evidence and the
// actual shared matting-session verdict. A successful synthetic-alpha kernel
// dispatch is intentionally not an availability input.
OpenVulkanVirtualBackgroundBlurReadiness
EvaluateOpenVulkanVirtualBackgroundBlurReadiness(
    const studiocast::vulkan::OpenVulkanDiagnostics &diagnostics,
    const studiocast::open_vulkan::VulkanMattingReadiness &matting_readiness);

struct OpenVulkanVirtualBackgroundBlurInput {
  const studiocast::vulkan::VulkanImage *foreground = nullptr;
  const studiocast::vulkan::VulkanImage *alpha = nullptr;
  const studiocast::vulkan::VulkanImage *alpha_tmp = nullptr;
  const studiocast::vulkan::VulkanImage *alpha_feathered = nullptr;
  const studiocast::vulkan::VulkanImage *blur_tmp = nullptr;
  const studiocast::vulkan::VulkanImage *blurred = nullptr;
  const studiocast::vulkan::VulkanImage *output = nullptr;
  int strength = 0;
  std::uint64_t capture_sequence = 0;
  std::uint64_t resident_alpha_sequence = 0;
  std::uint64_t alpha_resize_completion_count = 0;
  // This is the current post-inference session evidence, not the cold setup
  // verdict used to initialize the wrapper.
  const studiocast::open_vulkan::VulkanMattingReadiness *matting_readiness =
      nullptr;
};

struct OpenVulkanVirtualBackgroundBlurCounters {
  std::uint64_t alpha_feather_dispatch_calls = 0;
  std::uint64_t blur_composite_dispatch_calls = 0;
  std::uint64_t runtime_failure_frames = 0;
  std::uint64_t device_loss_frames = 0;
  // These remain explicit evidence: this effect owns no CPU implementation or
  // alpha-readback path. Final RGB transport is accounted by the pipeline.
  std::uint64_t alpha_readback_calls = 0;
  std::uint64_t cpu_fallback_calls = 0;
};

// Canonical live-boundary wrapper for virtual_background.blur. It owns no
// Vulkan device and allocates nothing per frame. All scratch and output images
// are persistent allocations from the shared camera Vulkan context.
class OpenVulkanVirtualBackgroundBlur {
public:
  bool EnsureInitialized(
      studiocast::vulkan::kernels::UtilityKernels *kernels, int width,
      int height,
      const studiocast::open_vulkan::VulkanMattingReadiness &matting_readiness,
      std::string *error_out);
  bool Apply(const OpenVulkanVirtualBackgroundBlurInput &input,
             OpenVulkanVirtualBackgroundBlurCounters *counters,
             std::string *error_out);
  void Shutdown() noexcept;

  bool initialized() const { return initialized_; }

private:
  studiocast::vulkan::kernels::UtilityKernels *kernels_ = nullptr;
  studiocast::vulkan::VulkanContextIdentity context_identity_{};
  int width_ = 0;
  int height_ = 0;
  bool initialized_ = false;
};

} // namespace studiocast::video
