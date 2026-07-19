#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "core/vulkan/kernels/utility_kernels.h"

namespace studiocast::video {

inline constexpr std::string_view kOpenVulkanAutoFrameCpuTailReason =
    "vulkan_effect_cpu_tail";
inline constexpr std::string_view kOpenVulkanAutoFrameFaceTrackingTailReason =
    "vulkan_auto_frame_cpu_face_tracking_tail";
inline constexpr std::string_view kOpenVulkanAutoFrameMatteReadbackTailReason =
    "vulkan_auto_frame_matte_alpha_readback_cpu_box_tail";
inline constexpr std::string_view kOpenVulkanAutoFrameCropPlanTailReason =
    "vulkan_auto_frame_cpu_crop_plan_smoothing_tail";
inline constexpr std::string_view kOpenVulkanAutoFrameFaceProviderReason =
    "vulkan_auto_frame_yunet_unavailable";
inline constexpr std::string_view kOpenVulkanAutoFrameCpuResizeFallbackReason =
    "vulkan_auto_frame_cpu_resize_fallback";
inline constexpr std::string_view kOpenVulkanAutoFrameInitializationReason =
    "vulkan_effect_initialization_failed";
inline constexpr std::string_view kOpenVulkanAutoFrameRuntimeReason =
    "vulkan_effect_runtime_failed";

std::string OpenVulkanAutoFrameInitializationFailure(std::string_view detail);
std::string OpenVulkanAutoFrameRuntimeFailure(std::string_view detail);
std::string OpenVulkanAutoFrameFaceProviderFailure(std::string_view detail);

enum class OpenVulkanAutoFrameResetReason {
  none,
  enablement_or_effect_generation,
  capture_sequence_discontinuity,
  geometry_model_or_provider_change,
  vulkan_context_generation_change,
  runtime_or_device_failure,
  pipeline_restart,
};

const char *OpenVulkanAutoFrameResetReasonName(
    OpenVulkanAutoFrameResetReason reason);

// A duplicate sequence is not a new frame; all other non-consecutive sequence
// transitions invalidate temporal tracking/smoothing state.
bool OpenVulkanAutoFrameSequenceNeedsReset(std::uint64_t previous_sequence,
                                           std::uint64_t next_sequence);

// Pure temporal-state reuse contract. This is intentionally independent of a
// live Vulkan device so provider/model/configuration changes can be tested
// deterministically without a camera or GPU.
struct OpenVulkanAutoFrameReuseKey {
  bool observed = false;
  bool enabled = false;
  std::uint64_t effects_generation = 0;
  int frame_width = 0;
  int frame_height = 0;
  std::string model_id;
  std::string provider_id;
  studiocast::vulkan::VulkanContextIdentity context_identity{};
};

OpenVulkanAutoFrameResetReason OpenVulkanAutoFrameReuseKeyResetReason(
    const OpenVulkanAutoFrameReuseKey &previous,
    const OpenVulkanAutoFrameReuseKey &next);

struct OpenVulkanAutoFrameCounters {
  std::uint64_t crop_resize_dispatch_calls = 0;
  std::uint64_t cpu_face_tracking_calls = 0;
  std::uint64_t matte_alpha_readback_calls = 0;
  std::uint64_t matte_cpu_box_calls = 0;
  std::uint64_t cpu_crop_plan_smoothing_calls = 0;
  // Explicit Vulkan never invokes a CPU crop/resize substitute. This counter
  // remains available to prove that invariant and characterize only a future
  // explicitly documented legacy path.
  std::uint64_t cpu_resize_fallback_calls = 0;
  std::uint64_t initialization_failures = 0;
  std::uint64_t runtime_failure_frames = 0;
  std::uint64_t device_loss_frames = 0;
  std::uint64_t temporal_reset_calls = 0;
};

struct OpenVulkanAutoFrameCropInput {
  const studiocast::vulkan::VulkanImage *src = nullptr;
  const studiocast::vulkan::VulkanImage *dst = nullptr;
  float crop_x = 0.0f;
  float crop_y = 0.0f;
  float crop_w = 0.0f;
  float crop_h = 0.0f;
  bool host_analysis_complete = false;
  bool cpu_crop_plan_complete = false;
};

// Canonical live-boundary wrapper for the degraded Auto Frame Vulkan crop.
// Tracking and crop planning deliberately remain explicit CPU tails; the
// frame crop itself must execute on the same production-ready shared Vulkan
// context and never substitutes a CPU resize when it fails.
class OpenVulkanAutoFrame {
public:
  bool EnsureInitialized(studiocast::vulkan::kernels::UtilityKernels *kernels,
                         int width, int height,
                         OpenVulkanAutoFrameCounters *counters,
                         std::string *error_out);
  bool ApplyCrop(const OpenVulkanAutoFrameCropInput &input,
                 OpenVulkanAutoFrameCounters *counters,
                 std::string *error_out);
  void ResetTemporal(OpenVulkanAutoFrameResetReason reason,
                     OpenVulkanAutoFrameCounters *counters) noexcept;
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
