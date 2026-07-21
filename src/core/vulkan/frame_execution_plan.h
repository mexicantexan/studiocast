#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace studiocast::vulkan {

// Hardware-independent description of the ordered GPU work for one frame.
// This is deliberately a planning contract: command recording remains
// synchronous until command-buffer, descriptor, and mapped-parameter reuse is
// protected by an implemented completion mechanism.
enum class FrameExecutionStage {
  preprocess,
  inference_output_copy,
  alpha_resize,
  alpha_feather,
  background_blur,
  composite,
  key_light,
  crop_resize,
};

const char *FrameExecutionStageName(FrameExecutionStage stage);

struct FrameDispatchRequest {
  FrameExecutionStage stage = FrameExecutionStage::preprocess;
  std::string label;
  std::vector<std::string> reads;
  std::vector<std::string> writes;
};

enum class FrameInitialResourceState {
  compute_read,
  host_write,
  transfer_write,
};

struct FrameInitialResource {
  std::string resource;
  FrameInitialResourceState state = FrameInitialResourceState::compute_read;
};

struct FrameExecutionRequest {
  // Resources already valid before command recording, such as the captured
  // frame. They are read-only within this plan.
  std::vector<std::string> external_resources;
  // Explicit initial state for uploaded/staged resources. Names must not also
  // appear in external_resources.
  std::vector<FrameInitialResource> initial_resources;
  std::vector<FrameDispatchRequest> dispatches;
  // Resources made host-visible at the frame completion/readback boundary.
  std::vector<std::string> host_readbacks;
  std::size_t parameter_slot_capacity = 0;
  std::size_t descriptor_slot_capacity = 0;
  // Reuse is opt-in. When enabled, a later dispatch may overwrite a resource
  // and receives an explicit compute-write to compute-write barrier.
  bool allow_resource_reuse = false;
};

struct PlannedFrameDispatch {
  FrameExecutionStage stage = FrameExecutionStage::preprocess;
  std::string label;
  std::vector<std::string> reads;
  std::vector<std::string> writes;
  std::size_t parameter_slot = 0;
  std::size_t descriptor_slot = 0;
};

enum class FrameBarrierKind {
  host_write_to_compute_read,
  transfer_write_to_compute_read,
  compute_write_to_compute_read,
  compute_write_to_compute_write,
  compute_write_to_host_read,
};

struct FrameResourceBarrier {
  FrameBarrierKind kind = FrameBarrierKind::compute_write_to_compute_read;
  std::string resource;
  std::size_t producer_dispatch = 0;
  // For a host-read barrier this equals dispatches.size(), the completion
  // boundary immediately following the final dispatch.
  std::size_t consumer_boundary = 0;
};

struct FrameExecutionPlan {
  std::vector<PlannedFrameDispatch> dispatches;
  std::vector<FrameResourceBarrier> barriers;
  std::vector<std::string> host_readbacks;
  std::size_t completion_boundary = 0;
  // The initial implementation intentionally has no fence ring or timeline
  // semaphore. Slots must not be updated until the frame submission completes.
  bool requires_synchronous_completion_before_slot_reuse = true;
};

bool BuildFrameExecutionPlan(const FrameExecutionRequest &request,
                             FrameExecutionPlan *plan, std::string *error_out);

// Revalidates a built (or test-mutated) plan before it is used for command
// recording. In particular, parameter and descriptor slots must be unique.
bool ValidateFrameExecutionPlan(const FrameExecutionPlan &plan,
                                std::string *error_out);

} // namespace studiocast::vulkan
