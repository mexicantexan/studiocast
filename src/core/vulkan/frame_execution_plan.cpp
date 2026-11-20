#include "core/vulkan/frame_execution_plan.h"

#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace studiocast::vulkan {
namespace {

struct ResourceState {
  bool external = false;
  bool written = false;
  FrameInitialResourceState initial_state =
      FrameInitialResourceState::compute_read;
  bool initial_state_consumed = true;
  bool compute_write_visible = false;
  std::size_t writer = 0;
};

bool Fail(std::string message, std::string *error_out) {
  if (error_out)
    *error_out = std::move(message);
  return false;
}

bool InsertUniqueResource(const std::string &resource, const std::string &what,
                          std::unordered_set<std::string> *seen,
                          std::string *error_out) {
  if (resource.empty())
    return Fail(std::string(what) + " contains an empty resource name.",
                error_out);
  if (!seen->insert(resource).second)
    return Fail(std::string(what) + " contains duplicate resource '" +
                    resource + "'.",
                error_out);
  return true;
}

} // namespace

const char *FrameExecutionStageName(FrameExecutionStage stage) {
  switch (stage) {
  case FrameExecutionStage::preprocess:
    return "preprocess";
  case FrameExecutionStage::inference_output_copy:
    return "inference_output_copy";
  case FrameExecutionStage::alpha_resize:
    return "alpha_resize";
  case FrameExecutionStage::alpha_feather:
    return "alpha_feather";
  case FrameExecutionStage::background_blur:
    return "background_blur";
  case FrameExecutionStage::composite:
    return "composite";
  case FrameExecutionStage::key_light:
    return "key_light";
  case FrameExecutionStage::crop_resize:
    return "crop_resize";
  }
  return "unknown";
}

bool BuildFrameExecutionPlan(const FrameExecutionRequest &request,
                             FrameExecutionPlan *plan, std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!plan)
    return Fail("Frame execution plan output is null.", error_out);
  *plan = {};

  if (request.dispatches.empty())
    return Fail("Frame execution request has no dispatches.", error_out);
  if (request.parameter_slot_capacity < request.dispatches.size()) {
    return Fail("Frame execution parameter slot capacity is smaller than the "
                "dispatch count.",
                error_out);
  }
  if (request.descriptor_slot_capacity < request.dispatches.size()) {
    return Fail("Frame execution descriptor slot capacity is smaller than the "
                "dispatch count.",
                error_out);
  }

  std::unordered_map<std::string, ResourceState> resources;
  std::unordered_set<std::string> external_seen;
  for (const std::string &resource : request.external_resources) {
    if (!InsertUniqueResource(resource, "Frame external resources",
                              &external_seen, error_out)) {
      return false;
    }
    ResourceState state;
    state.external = true;
    resources.emplace(resource, state);
  }

  std::unordered_set<std::string> initial_seen = external_seen;
  for (const FrameInitialResource &initial : request.initial_resources) {
    if (!InsertUniqueResource(initial.resource, "Frame initial resources",
                              &initial_seen, error_out)) {
      return false;
    }
    ResourceState state;
    state.external = true;
    state.initial_state = initial.state;
    state.initial_state_consumed =
        initial.state == FrameInitialResourceState::compute_read;
    resources.emplace(initial.resource, state);
  }

  std::unordered_set<std::string> labels;
  plan->dispatches.reserve(request.dispatches.size());
  for (std::size_t index = 0; index < request.dispatches.size(); ++index) {
    const FrameDispatchRequest &dispatch = request.dispatches[index];
    const std::string label = dispatch.label.empty()
                                  ? FrameExecutionStageName(dispatch.stage)
                                  : dispatch.label;
    if (!labels.insert(label).second) {
      return Fail(
          "Frame execution request contains duplicate dispatch label '" +
              label + "'.",
          error_out);
    }
    if (dispatch.writes.empty()) {
      return Fail("Frame dispatch '" + label + "' has no output resource.",
                  error_out);
    }

    std::unordered_set<std::string> reads;
    for (const std::string &resource : dispatch.reads) {
      if (!InsertUniqueResource(resource,
                                "Frame dispatch '" + label + "' reads", &reads,
                                error_out)) {
        return false;
      }
      const auto found = resources.find(resource);
      if (found == resources.end() ||
          (!found->second.external && !found->second.written)) {
        return Fail("Frame dispatch '" + label +
                        "' reads resource before it is initialized: '" +
                        resource + "'.",
                    error_out);
      }
      ResourceState &state = found->second;
      if (!state.initial_state_consumed) {
        const FrameBarrierKind kind =
            state.initial_state == FrameInitialResourceState::host_write
                ? FrameBarrierKind::host_write_to_compute_read
                : FrameBarrierKind::transfer_write_to_compute_read;
        plan->barriers.push_back({kind, resource, 0, index});
        state.initial_state_consumed = true;
      } else if (state.written && !state.compute_write_visible) {
        plan->barriers.push_back(
            {FrameBarrierKind::compute_write_to_compute_read, resource,
             state.writer, index});
        state.compute_write_visible = true;
      }
    }

    std::unordered_set<std::string> writes;
    for (const std::string &resource : dispatch.writes) {
      if (!InsertUniqueResource(resource,
                                "Frame dispatch '" + label + "' writes",
                                &writes, error_out)) {
        return false;
      }
      if (reads.count(resource) != 0) {
        return Fail("Frame dispatch '" + label +
                        "' aliases resource for read and write: '" + resource +
                        "'.",
                    error_out);
      }
      auto found = resources.find(resource);
      if (found != resources.end() && found->second.external) {
        return Fail("Frame dispatch '" + label +
                        "' writes read-only external resource '" + resource +
                        "'.",
                    error_out);
      }
      if (found != resources.end() && found->second.written) {
        if (!request.allow_resource_reuse) {
          return Fail(
              "Frame dispatch '" + label +
                  "' overwrites resource already written by this frame: '" +
                  resource + "'.",
              error_out);
        }
        plan->barriers.push_back(
            {FrameBarrierKind::compute_write_to_compute_write, resource,
             found->second.writer, index});
      }
    }

    PlannedFrameDispatch planned;
    planned.stage = dispatch.stage;
    planned.label = label;
    planned.reads = dispatch.reads;
    planned.writes = dispatch.writes;
    planned.parameter_slot = index;
    planned.descriptor_slot = index;
    plan->dispatches.push_back(std::move(planned));

    for (const std::string &resource : dispatch.writes) {
      ResourceState state;
      state.written = true;
      state.writer = index;
      state.compute_write_visible = false;
      resources[resource] = state;
    }
  }

  std::unordered_set<std::string> readbacks;
  for (const std::string &resource : request.host_readbacks) {
    if (!InsertUniqueResource(resource, "Frame host readbacks", &readbacks,
                              error_out)) {
      return false;
    }
    const auto found = resources.find(resource);
    if (found == resources.end() || !found->second.written) {
      return Fail("Frame completion reads resource without a frame writer: '" +
                      resource + "'.",
                  error_out);
    }
    plan->barriers.push_back({FrameBarrierKind::compute_write_to_host_read,
                              resource, found->second.writer,
                              request.dispatches.size()});
  }

  plan->host_readbacks = request.host_readbacks;
  plan->completion_boundary = request.dispatches.size();
  plan->requires_synchronous_completion_before_slot_reuse = true;
  return ValidateFrameExecutionPlan(*plan, error_out);
}

bool ValidateFrameExecutionPlan(const FrameExecutionPlan &plan,
                                std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (plan.dispatches.empty())
    return Fail("Frame execution plan has no dispatches.", error_out);
  if (plan.completion_boundary != plan.dispatches.size()) {
    return Fail("Frame execution completion boundary is not after the final "
                "dispatch.",
                error_out);
  }
  if (!plan.requires_synchronous_completion_before_slot_reuse) {
    return Fail("Frame execution plan does not protect slot reuse with a "
                "completion requirement.",
                error_out);
  }

  std::unordered_set<std::size_t> parameter_slots;
  std::unordered_set<std::size_t> descriptor_slots;
  for (const PlannedFrameDispatch &dispatch : plan.dispatches) {
    if (!parameter_slots.insert(dispatch.parameter_slot).second) {
      return Fail("Frame execution plan reuses parameter slot " +
                      std::to_string(dispatch.parameter_slot) + ".",
                  error_out);
    }
    if (!descriptor_slots.insert(dispatch.descriptor_slot).second) {
      return Fail("Frame execution plan reuses descriptor slot " +
                      std::to_string(dispatch.descriptor_slot) + ".",
                  error_out);
    }
  }

  std::unordered_set<std::string> host_barrier_resources;
  for (const FrameResourceBarrier &barrier : plan.barriers) {
    if (barrier.resource.empty())
      return Fail("Frame execution plan contains a barrier without a resource.",
                  error_out);
    if (barrier.producer_dispatch >= plan.dispatches.size()) {
      return Fail("Frame execution barrier has an invalid producer.",
                  error_out);
    }
    if (barrier.kind != FrameBarrierKind::compute_write_to_host_read) {
      if (barrier.consumer_boundary >= plan.dispatches.size() ||
          (barrier.kind != FrameBarrierKind::host_write_to_compute_read &&
           barrier.kind != FrameBarrierKind::transfer_write_to_compute_read &&
           barrier.consumer_boundary <= barrier.producer_dispatch)) {
        return Fail("Frame compute barrier has an invalid consumer boundary.",
                    error_out);
      }
    } else {
      if (barrier.consumer_boundary != plan.completion_boundary) {
        return Fail("Frame host barrier is not at the completion boundary.",
                    error_out);
      }
      host_barrier_resources.insert(barrier.resource);
    }
  }
  for (const std::string &resource : plan.host_readbacks) {
    if (host_barrier_resources.count(resource) == 0) {
      return Fail("Frame host readback is missing its completion barrier: '" +
                      resource + "'.",
                  error_out);
    }
  }
  return true;
}

} // namespace studiocast::vulkan
