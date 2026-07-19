#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "core/video/open_vulkan_mirror.h"
#include "core/video/open_vulkan_vignette.h"

namespace studiocast::video {

// Dependencies and caller-owned reusable images for the canonical final
// resident output block. CameraPipeline and the executable integration test
// both invoke ExecuteOpenVulkanFinalResidentStage; this is not a test seam.
struct OpenVulkanFinalResidentStageResources {
  studiocast::vulkan::kernels::UtilityKernels *kernels = nullptr;
  studiocast::vulkan::VulkanImage *resize_scratch = nullptr;
  studiocast::vulkan::VulkanImage *vignette_output = nullptr;
  studiocast::vulkan::VulkanImage *mirror_output = nullptr;
  OpenVulkanVignette *vignette = nullptr;
  OpenVulkanVignetteCounters *vignette_counters = nullptr;
  OpenVulkanMirror *mirror = nullptr;
  OpenVulkanMirrorCounters *mirror_counters = nullptr;
};

struct OpenVulkanFinalResidentStageInput {
  const studiocast::vulkan::VulkanImage *source = nullptr;
  int output_width = 0;
  int output_height = 0;
  bool request_fixed_center_vignette = false;
  bool request_mirror = false;
  int vignette_intensity_percent = 0;

  // These flags represent the exact ordering boundaries established by the
  // live CameraPipeline. Keeping them explicit lets the production helper
  // reject an invalid call without dispatching while preserving isolation of
  // an unrelated final stage whose own boundary is satisfied.
  bool preceding_effects_complete = false;
  bool unmirrored_analysis_complete = false;
};

struct OpenVulkanFinalResidentStageCounters {
  std::uint64_t execution_calls = 0;
  std::uint64_t resource_allocation_calls = 0;
  std::uint64_t successful_output_frames = 0;
  // The resident helper has no host continuation. These stay zero and are
  // asserted by the executable production-path test.
  std::uint64_t intermediate_readback_calls = 0;
  std::uint64_t cpu_fallback_calls = 0;
};

struct OpenVulkanFinalResidentStageResult {
  const studiocast::vulkan::VulkanImage *output = nullptr;
  bool output_valid = false;
  bool resize_applied = false;
  bool vignette_applied = false;
  bool mirror_applied = false;
  bool vignette_failed = false;
  bool mirror_failed = false;
  bool vignette_initialization_failed = false;
  bool mirror_initialization_failed = false;
  std::uint64_t synchronous_completion_count = 0;
  std::size_t resident_stage_count = 0;
  std::string_view vignette_backend;
  std::string_view mirror_backend;
  std::string vignette_error;
  std::string mirror_error;
  std::string fatal_error;
};

// Runs the canonical resident final-output decision in this fixed order:
// output resize (when needed) -> fixed-center vignette -> mirror. It performs
// no CPU fallback and no readback; the caller owns the single final output
// readback required by the camera writer boundary.
bool ExecuteOpenVulkanFinalResidentStage(
    const OpenVulkanFinalResidentStageInput &input,
    const OpenVulkanFinalResidentStageResources &resources,
    OpenVulkanFinalResidentStageCounters *counters,
    OpenVulkanFinalResidentStageResult *result);

} // namespace studiocast::video
