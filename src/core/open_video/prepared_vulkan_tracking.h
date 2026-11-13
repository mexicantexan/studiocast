#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "core/hal/prepared_program.h"
#include "core/open_video/dlib_shape_predictor_data.h"
#include "core/vulkan/vulkan_image.h"
#include "core/vulkan/vulkan_tensor.h"

namespace studiocast::vulkan {
class ResidentCommandBatch;
}

namespace studiocast::open_vulkan {

struct VulkanEyeInputSlots {
  const vulkan::VulkanTensor *rgb = nullptr;
  const vulkan::VulkanTensor *anchors = nullptr;
  const vulkan::VulkanTensor *angles = nullptr;
};
struct VulkanEyePrefixOutputSlots {
  const vulkan::VulkanTensor *flow = nullptr;
  const vulkan::VulkanTensor *lcm = nullptr;
};

struct PreparedVulkanTrackingSetup {
  vulkan::VulkanDevice *device = nullptr;
  hal::ExecutionDomainId domain{};
  std::uint64_t owner_generation = 0;
  int frame_width = 0, frame_height = 0;
  int final_width = 0, final_height = 0;
  bool face_analysis = false, eye_contact = false;
  bool auto_frame = false, vignette = false;
  int eye_strength = 0;
  bool look_away_enabled = false;
  int auto_frame_strength = 50, auto_frame_smoothing = 50;
  float auto_frame_headroom = .15F;
  int vignette_intensity = 0;
  bool vignette_tracked_center = false;
  const open_video::DlibPredictorData *predictor = nullptr;
  open_video::ValidatedModelArtifactIdentity face_artifact{};
  open_video::ValidatedModelArtifactIdentity landmark_artifact{};
  open_video::ValidatedModelArtifactIdentity eye_left_artifact{};
  open_video::ValidatedModelArtifactIdentity eye_right_artifact{};
  hal::ResourceBudget reservation{};
};

struct PreparedVulkanTrackingCounters {
  std::uint64_t committed_command_groups = 0;
  std::uint64_t aborted_command_groups = 0;
  std::uint64_t unsafe_command_groups = 0;
  std::uint64_t retired_command_groups = 0;
};

// One concrete synchronous resident leaf owner. All analysis/temporal state is
// DEVICE_LOCAL. The generic inference runtime binds these canonical contiguous
// FP32 slots once; this owner never invokes inference or submits a queue.
class PreparedVulkanTracking {
public:
  PreparedVulkanTracking();
  ~PreparedVulkanTracking();
  PreparedVulkanTracking(const PreparedVulkanTracking &) = delete;
  PreparedVulkanTracking &operator=(const PreparedVulkanTracking &) = delete;

  bool Prepare(const PreparedVulkanTrackingSetup &, std::string *error);
  const vulkan::VulkanTensor &FaceInput() const noexcept;
  // cls8/16/32, obj8/16/32, bbox8/16/32, kps8/16/32. Each
  // canonical output is contiguous [1,1,cells,channels], channels last.
  std::array<const vulkan::VulkanTensor *, 12> FaceOutputs() const noexcept;
  VulkanEyeInputSlots EyeInputs(unsigned eye) const noexcept;
  VulkanEyePrefixOutputSlots EyePrefixOutputs(unsigned eye) const noexcept;

  bool AppendFacePreprocess(vulkan::ResidentCommandBatch &,
                            const vulkan::VulkanImage &, std::string *error);
  bool AppendFaceDecode(vulkan::ResidentCommandBatch &, std::string *error);
  bool AppendLandmarks(vulkan::ResidentCommandBatch &,
                       const vulkan::VulkanImage &, std::string *error);
  bool AppendEyeInputs(vulkan::ResidentCommandBatch &,
                       const vulkan::VulkanImage &, std::string *error);
  bool AppendEyeTailAndComposite(vulkan::ResidentCommandBatch &,
                                 const vulkan::VulkanImage &,
                                 const vulkan::VulkanImage &output,
                                 std::string *error);
  bool AppendAutoFrame(vulkan::ResidentCommandBatch &,
                       const vulkan::VulkanImage &,
                       const vulkan::VulkanImage *alpha,
                       const vulkan::VulkanImage &output, std::string *error);
  bool AppendVignette(vulkan::ResidentCommandBatch &,
                      const vulkan::VulkanImage &,
                      const vulkan::VulkanImage &output, std::string *error);
  bool AppendReset(vulkan::ResidentCommandBatch &, std::string *error);
  // Candidate setup/warmup establishes every enabled phase's fixed image
  // binding. Once sealed, live calls may only replay those exact bindings.
  bool SealBindingsForExecution(std::string *error);

  bool Shutdown(std::string *error) noexcept;
  bool ready() const noexcept;
  PreparedVulkanTrackingCounters counters() const noexcept;
  hal::ResourceBudget resources() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::open_vulkan
