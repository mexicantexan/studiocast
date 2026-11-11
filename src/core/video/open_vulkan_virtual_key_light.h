#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "core/open_video/frame_analysis_cache.h"
#include "core/open_video/vulkan_matting_runtime.h"
#include "core/vulkan/kernels/utility_kernels.h"

namespace studiocast::video {

namespace detail {

// Pure cache-identity gate used by the live pipeline before it reuses a matte
// for key light. A lookup hit is insufficient: the exact object/buffer and all
// frame/model/geometry/device/queue fields must still match.
bool IsOpenVulkanVirtualKeyLightSameFrameArtifactCompatible(
    std::uint64_t capture_sequence, std::uint64_t cached_matte_sequence,
    std::uint64_t cached_frame_alpha_sequence,
    const studiocast::open_video::FrameMatteArtifactKey &key,
    const studiocast::open_video::FrameMatteArtifact *artifact,
    std::string_view active_model_id, int frame_width, int frame_height,
    int matte_width, int matte_height, std::uintptr_t device_context,
    std::uintptr_t queue, std::uintptr_t expected_object_handle,
    std::uintptr_t expected_buffer_handle);

} // namespace detail

inline constexpr std::string_view
    kOpenVulkanVirtualKeyLightInitializationReason =
        "vulkan_virtual_key_light_initialization_failed";
inline constexpr std::string_view kOpenVulkanVirtualKeyLightRuntimeReason =
    "vulkan_virtual_key_light_runtime_failed";

std::string
OpenVulkanVirtualKeyLightInitializationFailure(std::string_view detail);
std::string OpenVulkanVirtualKeyLightRuntimeFailure(std::string_view detail);

struct OpenVulkanVirtualKeyLightParameters {
  float target_r = 255.0f;
  float target_g = 255.0f;
  float target_b = 255.0f;
  float intensity = 0.0f;
  float direction = 0.0f;

  bool passthrough() const { return intensity <= 0.0001f; }
};

// Preserves the established Open CUDA masked-lift semantics. The public pan
// range remains -180..180, while the current backend behavior saturates at
// +/-90 degrees. Legacy Kelvin and HDRI fields are intentionally not inputs to
// this Open CUDA-compatible kernel.
OpenVulkanVirtualKeyLightParameters ResolveOpenVulkanVirtualKeyLightParameters(
    int intensity_percent, int temperature_preset, int direction_pan_degrees);

struct OpenVulkanVirtualKeyLightReadiness {
  bool production_ready = false;
  std::string reason_code;
  std::string detail;
};

OpenVulkanVirtualKeyLightReadiness EvaluateOpenVulkanVirtualKeyLightReadiness(
    const studiocast::vulkan::OpenVulkanDiagnostics &diagnostics,
    const studiocast::open_vulkan::VulkanMattingReadiness &matting_readiness);

enum class OpenVulkanVirtualKeyLightMatteSource {
  independently_inferred,
  reused_same_frame,
};

struct OpenVulkanVirtualKeyLightInput {
  const studiocast::vulkan::VulkanImage *foreground = nullptr;
  const studiocast::vulkan::VulkanImage *alpha = nullptr;
  const studiocast::vulkan::VulkanImage *output = nullptr;
  std::uint64_t capture_sequence = 0;
  std::uint64_t resident_alpha_sequence = 0;
  std::uint64_t alpha_resize_completion_count_before = 0;
  std::uint64_t alpha_resize_completion_count_after = 0;
  std::uint64_t matting_inference_count_before = 0;
  OpenVulkanVirtualKeyLightMatteSource matte_source =
      OpenVulkanVirtualKeyLightMatteSource::independently_inferred;
  // Current post-inference evidence, not the cold setup verdict.
  const studiocast::open_vulkan::VulkanMattingReadiness *matting_readiness =
      nullptr;
};

struct OpenVulkanVirtualKeyLightCounters {
  std::uint64_t dispatch_calls = 0;
  std::uint64_t shared_matte_reuse_calls = 0;
  std::uint64_t independent_matte_inference_calls = 0;
  std::uint64_t passthrough_frames = 0;
  std::uint64_t runtime_failure_frames = 0;
  std::uint64_t device_loss_frames = 0;
  // The effect owns neither a readback nor a selectable CPU implementation.
  std::uint64_t alpha_readback_calls = 0;
  std::uint64_t cpu_fallback_calls = 0;
};

// Canonical live-boundary wrapper for virtual_key_light. Setup freezes the
// Open CUDA-compatible parameter mapping and the exact shared context. Apply
// allocates nothing and accepts only same-frame resident matting evidence.
class OpenVulkanVirtualKeyLight {
public:
  bool EnsureInitialized(
      studiocast::vulkan::kernels::UtilityKernels *kernels, int width,
      int height, int intensity_percent, int temperature_preset,
      int direction_pan_degrees,
      const studiocast::open_vulkan::VulkanMattingReadiness &matting_readiness,
      std::string *error_out);
  bool Apply(const OpenVulkanVirtualKeyLightInput &input,
             OpenVulkanVirtualKeyLightCounters *counters,
             std::string *error_out);
  void Shutdown() noexcept;

  bool initialized() const { return initialized_; }
  const OpenVulkanVirtualKeyLightParameters &parameters() const {
    return parameters_;
  }

private:
  studiocast::vulkan::kernels::UtilityKernels *kernels_ = nullptr;
  studiocast::vulkan::VulkanContextIdentity context_identity_{};
  OpenVulkanVirtualKeyLightParameters parameters_{};
  int width_ = 0;
  int height_ = 0;
  bool initialized_ = false;
};

} // namespace studiocast::video
