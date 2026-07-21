#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "core/video/open_vulkan_auto_frame.h"
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
  std::uint64_t device_local_allocation_calls = 0;
  // The helper never allocates mapped effect/intermediate storage. Initial
  // upload and final readback staging remain caller-owned transport resources.
  std::uint64_t host_visible_intermediate_allocation_calls = 0;
  std::uint64_t residency_rejection_calls = 0;
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

// Per-frame state for the production Open Vulkan resident orchestration seam.
// CameraPipeline and the executable integration test both use these exact
// branch decisions; the seam performs no transfer or CPU fallback.
struct OpenVulkanResidentFrameState {
  const studiocast::vulkan::VulkanImage *current = nullptr;
  bool auto_frame_applied = false;
  bool auto_frame_crop_applied = false;
};

struct OpenVulkanResidentAutoFrameInput {
  const studiocast::vulkan::VulkanImage *source = nullptr;
  studiocast::vulkan::VulkanImage *crop_output = nullptr;
  bool request_crop = false;
  float crop_x = 0.0f;
  float crop_y = 0.0f;
  float crop_width = 0.0f;
  float crop_height = 0.0f;
  bool host_analysis_complete = false;
  bool cpu_crop_plan_complete = false;
};

bool ExecuteOpenVulkanResidentAutoFrameStage(
    const OpenVulkanResidentAutoFrameInput &input,
    OpenVulkanAutoFrame *auto_frame, OpenVulkanAutoFrameCounters *counters,
    OpenVulkanResidentFrameState *state, std::string *error_out);

// Completes resize -> fixed-center vignette -> mirror from the current
// resident frame. This is the only CameraPipeline call site for the exact
// final resident helper, so tests cannot drift into a second orchestration.
bool ExecuteOpenVulkanResidentFrameFinalStage(
    const OpenVulkanResidentFrameState &state,
    const OpenVulkanFinalResidentStageInput &input_without_source,
    const OpenVulkanFinalResidentStageResources &resources,
    OpenVulkanFinalResidentStageCounters *counters,
    OpenVulkanFinalResidentStageResult *result);

// Runs the canonical resident final-output decision in this fixed order:
// output resize (when needed) -> fixed-center vignette -> mirror. It performs
// no CPU fallback and no readback. Source, scratch, and every effect output
// must be non-mapped DEVICE_LOCAL resources; the caller owns the bounded
// upload staging and the single final readback staging required by the camera
// writer boundary.
bool ExecuteOpenVulkanFinalResidentStage(
    const OpenVulkanFinalResidentStageInput &input,
    const OpenVulkanFinalResidentStageResources &resources,
    OpenVulkanFinalResidentStageCounters *counters,
    OpenVulkanFinalResidentStageResult *result);

} // namespace studiocast::video
