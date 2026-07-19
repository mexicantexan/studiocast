#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "core/vulkan/kernels/utility_kernels.h"

namespace studiocast::video {

inline constexpr std::string_view
    kOpenVulkanVignetteInitializationFailureReason =
        "vulkan_effect_initialization_failed";
inline constexpr std::string_view kOpenVulkanVignetteRuntimeFailureReason =
    "vulkan_effect_runtime_failed";

std::string OpenVulkanVignetteInitializationFailure(std::string_view detail);
std::string OpenVulkanVignetteRuntimeFailure(std::string_view detail);

struct OpenVulkanVignetteReadiness {
  bool production_ready = false;
  std::string shared_reason_code;
  std::string detail;
};

OpenVulkanVignetteReadiness EvaluateOpenVulkanVignetteReadiness(
    const studiocast::vulkan::OpenVulkanDiagnostics &diagnostics);

namespace detail {

// CUDA-reference coordinate contract used to build the device-resident
// attenuation-factor lookup. Production uses center=(width/2,height/2);
// explicit center arguments keep the radial^2 component and clamping behavior
// independently testable.
float OpenVulkanVignetteRadialSquaredAt(int x, int y, int width, int height,
                                        float center_x_px, float center_y_px);

} // namespace detail

struct OpenVulkanVignetteFinalStageInput {
  const studiocast::vulkan::VulkanImage *src = nullptr;
  // Optional final-output resize target. Null means src already has final
  // output geometry.
  const studiocast::vulkan::VulkanImage *resize_scratch = nullptr;
  const studiocast::vulkan::VulkanImage *dst = nullptr;
  // Optional final mirror output. When present the one resident batch is
  // resize (if needed) -> vignette -> mirror.
  const studiocast::vulkan::VulkanImage *mirrored_dst = nullptr;
  bool preceding_effects_complete = false;
  int intensity_percent = 0;
};

struct OpenVulkanVignetteCounters {
  // Setup/configuration counters. They must not change on repeated frames at
  // the same geometry and shared-context generation.
  std::uint64_t factor_allocation_calls = 0;
  std::uint64_t factor_generation_calls = 0;
  std::uint64_t factor_upload_calls = 0;
  std::uint64_t dispatch_calls = 0;
  std::uint64_t runtime_failure_frames = 0;
};

// Production fixed-center Vulkan vignette. The attenuation-factor lookup is
// generated and uploaded once per geometry/intensity/context configuration,
// then all frame work stays device resident. This backend does not consume
// tracked-center data: planning rejects that semantic when a retained Auto
// Frame stage would make it observable, while standalone vignette remains
// fixed-center and introduces no analysis tail.
class OpenVulkanVignette {
public:
  bool EnsureInitialized(studiocast::vulkan::kernels::UtilityKernels *kernels,
                         int width, int height, int intensity_percent,
                         OpenVulkanVignetteCounters *counters,
                         std::string *error_out);
  bool ApplyFinal(const OpenVulkanVignetteFinalStageInput &input,
                  OpenVulkanVignetteCounters *counters, std::string *error_out);
  void Shutdown() noexcept;

  bool initialized() const { return initialized_; }
  int width() const { return width_; }
  int height() const { return height_; }
  const studiocast::vulkan::VulkanImage &attenuation_factor_mask() const {
    return attenuation_factor_;
  }

private:
  studiocast::vulkan::kernels::UtilityKernels *kernels_ = nullptr;
  studiocast::vulkan::VulkanImage attenuation_factor_;
  int width_ = 0;
  int height_ = 0;
  int intensity_percent_ = 0;
  bool initialized_ = false;
};

} // namespace studiocast::video
