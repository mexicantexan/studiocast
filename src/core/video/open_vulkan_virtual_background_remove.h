#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "core/open_video/vulkan_matting_runtime.h"
#include "core/vulkan/kernels/utility_kernels.h"

namespace studiocast::video {

inline constexpr std::string_view
    kOpenVulkanVirtualBackgroundRemoveInitializationReason =
        "vulkan_virtual_background_remove_initialization_failed";
inline constexpr std::string_view
    kOpenVulkanVirtualBackgroundRemoveRuntimeReason =
        "vulkan_virtual_background_remove_runtime_failed";

std::string
OpenVulkanVirtualBackgroundRemoveInitializationFailure(std::string_view detail);
std::string
OpenVulkanVirtualBackgroundRemoveRuntimeFailure(std::string_view detail);

struct OpenVulkanVirtualBackgroundRemoveParameters {
  int alpha_feather_radius = 0;
  std::uint8_t background_r = 0;
  std::uint8_t background_g = 0;
  std::uint8_t background_b = 0;
};

// Preserves Open CUDA compatibility: strength affects remove mode only through
// alpha feathering, and malformed legacy color strings fall back to black.
OpenVulkanVirtualBackgroundRemoveParameters
ResolveOpenVulkanVirtualBackgroundRemoveParameters(
    int strength, std::string_view remove_color);

struct OpenVulkanVirtualBackgroundRemoveReadiness {
  bool production_ready = false;
  std::string reason_code;
  std::string detail;
};

OpenVulkanVirtualBackgroundRemoveReadiness
EvaluateOpenVulkanVirtualBackgroundRemoveReadiness(
    const studiocast::vulkan::OpenVulkanDiagnostics &diagnostics,
    const studiocast::open_vulkan::VulkanMattingReadiness &matting_readiness);

struct OpenVulkanVirtualBackgroundRemoveInput {
  const studiocast::vulkan::VulkanImage *foreground = nullptr;
  const studiocast::vulkan::VulkanImage *alpha = nullptr;
  const studiocast::vulkan::VulkanImage *alpha_tmp = nullptr;
  const studiocast::vulkan::VulkanImage *alpha_feathered = nullptr;
  const studiocast::vulkan::VulkanImage *output = nullptr;
  std::uint64_t capture_sequence = 0;
  std::uint64_t resident_alpha_sequence = 0;
  std::uint64_t alpha_resize_completion_count = 0;
  // Current post-inference evidence, not the cold setup verdict.
  const studiocast::open_vulkan::VulkanMattingReadiness *matting_readiness =
      nullptr;
};

struct OpenVulkanVirtualBackgroundRemoveCounters {
  std::uint64_t alpha_feather_dispatch_calls = 0;
  std::uint64_t solid_composite_dispatch_calls = 0;
  std::uint64_t runtime_failure_frames = 0;
  std::uint64_t device_loss_frames = 0;
  // Explicit proof that remove owns neither a CPU implementation nor an alpha
  // readback. Final RGB transport remains pipeline accounting.
  std::uint64_t alpha_readback_calls = 0;
  std::uint64_t cpu_fallback_calls = 0;
};

// Canonical live-boundary wrapper for virtual_background.remove. Parameters
// are resolved during setup/reconfiguration and the frame path allocates
// nothing. All images are persistent allocations from the shared context.
class OpenVulkanVirtualBackgroundRemove {
public:
  bool EnsureInitialized(
      studiocast::vulkan::kernels::UtilityKernels *kernels, int width,
      int height, int strength, std::string_view remove_color,
      const studiocast::open_vulkan::VulkanMattingReadiness &matting_readiness,
      std::string *error_out);
  bool Apply(const OpenVulkanVirtualBackgroundRemoveInput &input,
             OpenVulkanVirtualBackgroundRemoveCounters *counters,
             std::string *error_out);
  void Shutdown() noexcept;

  bool initialized() const { return initialized_; }
  const OpenVulkanVirtualBackgroundRemoveParameters &parameters() const {
    return parameters_;
  }

private:
  studiocast::vulkan::kernels::UtilityKernels *kernels_ = nullptr;
  studiocast::vulkan::VulkanContextIdentity context_identity_{};
  OpenVulkanVirtualBackgroundRemoveParameters parameters_{};
  int width_ = 0;
  int height_ = 0;
  bool initialized_ = false;
};

} // namespace studiocast::video
