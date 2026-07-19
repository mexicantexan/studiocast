#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "core/vulkan/kernels/utility_kernels.h"

namespace studiocast::video {

inline constexpr std::string_view kVulkanEffectInitializationFailed =
    "vulkan_effect_initialization_failed";
inline constexpr std::string_view kVulkanEffectRuntimeFailed =
    "vulkan_effect_runtime_failed";

std::string OpenVulkanMirrorInitializationFailure(std::string_view detail);
std::string OpenVulkanMirrorRuntimeFailure(std::string_view detail);

struct OpenVulkanMirrorFinalStageInput {
  const studiocast::vulkan::VulkanImage *src = nullptr;
  const studiocast::vulkan::VulkanImage *dst = nullptr;
  // The camera pipeline sets this only after all subject/face analysis and
  // preceding effects have consumed the unmirrored frame.
  bool unmirrored_analysis_complete = false;
  // The camera pipeline sets this after the resident output resize, or
  // immediately when the source already has final output geometry.
  bool output_geometry_ready = false;
};

struct OpenVulkanMirrorCounters {
  std::uint64_t dispatch_calls = 0;
  std::uint64_t runtime_failure_frames = 0;
};

// Mirror-specific production wrapper over the shared Vulkan context. It owns
// no device or per-frame allocation and accepts only same-context,
// same-size/out-of-place RGB/BGR images.
class OpenVulkanMirror {
public:
  bool EnsureInitialized(studiocast::vulkan::kernels::UtilityKernels *kernels,
                         int width, int height, std::string *error_out);
  bool ApplyFinal(const OpenVulkanMirrorFinalStageInput &input,
                  OpenVulkanMirrorCounters *counters, std::string *error_out);

  bool initialized() const { return initialized_; }
  int width() const { return width_; }
  int height() const { return height_; }

private:
  studiocast::vulkan::kernels::UtilityKernels *kernels_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  bool initialized_ = false;
};

} // namespace studiocast::video
