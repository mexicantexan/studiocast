#include <iostream>
#include <string>
#include <unordered_set>

#include "core/vulkan/frame_execution_plan.h"

namespace {

using studiocast::vulkan::BuildFrameExecutionPlan;
using studiocast::vulkan::FrameBarrierKind;
using studiocast::vulkan::FrameDispatchRequest;
using studiocast::vulkan::FrameExecutionPlan;
using studiocast::vulkan::FrameExecutionRequest;
using studiocast::vulkan::FrameExecutionStage;
using studiocast::vulkan::FrameInitialResource;
using studiocast::vulkan::FrameInitialResourceState;
using studiocast::vulkan::ValidateFrameExecutionPlan;

bool Require(bool condition, const std::string &message) {
  if (!condition)
    std::cerr << message << '\n';
  return condition;
}

FrameExecutionRequest FullFrameRequest() {
  FrameExecutionRequest request;
  request.external_resources = {"captured_rgb"};
  request.dispatches = {
      {FrameExecutionStage::preprocess,
       "preprocess",
       {"captured_rgb"},
       {"model_input"}},
      {FrameExecutionStage::inference_output_copy,
       "inference_output_copy",
       {"model_input"},
       {"alpha_model"}},
      {FrameExecutionStage::alpha_resize,
       "alpha_resize",
       {"alpha_model"},
       {"alpha_resized"}},
      {FrameExecutionStage::alpha_feather,
       "alpha_feather",
       {"alpha_resized"},
       {"alpha_feathered"}},
      {FrameExecutionStage::background_blur,
       "background_blur",
       {"captured_rgb"},
       {"blurred_background"}},
      {FrameExecutionStage::composite,
       "composite",
       {"captured_rgb", "blurred_background", "alpha_feathered"},
       {"composited_rgb"}},
      {FrameExecutionStage::key_light,
       "key_light",
       {"composited_rgb", "alpha_feathered"},
       {"lit_rgb"}},
      {FrameExecutionStage::crop_resize,
       "crop_resize",
       {"lit_rgb"},
       {"output_rgb"}},
  };
  request.host_readbacks = {"output_rgb"};
  request.parameter_slot_capacity = request.dispatches.size();
  request.descriptor_slot_capacity = request.dispatches.size();
  return request;
}

bool TestFullFramePlan() {
  FrameExecutionPlan plan;
  std::string error;
  bool ok = Require(BuildFrameExecutionPlan(FullFrameRequest(), &plan, &error),
                    "full frame plan should build: " + error);
  ok &= Require(plan.dispatches.size() == 8,
                "full frame plan should represent all candidate stages");
  ok &= Require(plan.completion_boundary == plan.dispatches.size(),
                "completion boundary should follow the final dispatch");
  ok &= Require(plan.requires_synchronous_completion_before_slot_reuse,
                "planner must retain synchronous slot lifetime protection");

  std::unordered_set<std::size_t> parameters;
  std::unordered_set<std::size_t> descriptors;
  for (const auto &dispatch : plan.dispatches) {
    parameters.insert(dispatch.parameter_slot);
    descriptors.insert(dispatch.descriptor_slot);
  }
  ok &= Require(parameters.size() == plan.dispatches.size(),
                "each dispatch should have a unique parameter slot");
  ok &= Require(descriptors.size() == plan.dispatches.size(),
                "each dispatch should have a unique descriptor slot");

  bool tensor_ready = false;
  bool alpha_ready = false;
  bool blur_ready = false;
  bool composite_ready = false;
  bool host_ready = false;
  for (const auto &barrier : plan.barriers) {
    if (barrier.kind == FrameBarrierKind::compute_write_to_compute_read &&
        barrier.resource == "model_input" && barrier.producer_dispatch == 0 &&
        barrier.consumer_boundary == 1) {
      tensor_ready = true;
    }
    if (barrier.kind == FrameBarrierKind::compute_write_to_compute_read &&
        barrier.resource == "alpha_model" && barrier.producer_dispatch == 1 &&
        barrier.consumer_boundary == 2) {
      alpha_ready = true;
    }
    if (barrier.kind == FrameBarrierKind::compute_write_to_compute_read &&
        barrier.resource == "blurred_background" &&
        barrier.producer_dispatch == 4 && barrier.consumer_boundary == 5) {
      blur_ready = true;
    }
    if (barrier.kind == FrameBarrierKind::compute_write_to_compute_read &&
        barrier.resource == "composited_rgb" &&
        barrier.producer_dispatch == 5 && barrier.consumer_boundary == 6) {
      composite_ready = true;
    }
    if (barrier.kind == FrameBarrierKind::compute_write_to_host_read &&
        barrier.resource == "output_rgb" && barrier.producer_dispatch == 7 &&
        barrier.consumer_boundary == plan.completion_boundary) {
      host_ready = true;
    }
  }
  ok &=
      Require(tensor_ready && alpha_ready && blur_ready && composite_ready,
              "planner should insert explicit compute write-to-read barriers");
  ok &= Require(host_ready,
                "planner should insert the final host readback barrier");
  return ok;
}

bool TestInvalidRequests() {
  bool ok = true;
  std::string error;
  FrameExecutionPlan plan;

  auto request = FullFrameRequest();
  request.dispatches[0].reads = {"missing_input"};
  ok &= Require(!BuildFrameExecutionPlan(request, &plan, &error) &&
                    error.find("before it is initialized") != std::string::npos,
                "planner should reject read-before-write");

  request = FullFrameRequest();
  request.dispatches[2].writes = {"alpha_model"};
  ok &= Require(!BuildFrameExecutionPlan(request, &plan, &error) &&
                    error.find("aliases resource") != std::string::npos,
                "planner should reject dispatch-local read/write aliasing");

  request = FullFrameRequest();
  request.dispatches[3].writes = {"model_input_2"};
  request.dispatches[3].reads = {"alpha_resized"};
  request.dispatches[4].writes = {"model_input_2"};
  ok &= Require(!BuildFrameExecutionPlan(request, &plan, &error) &&
                    error.find("overwrites resource") != std::string::npos,
                "planner should reject unsafe frame-local resource reuse");

  request = FullFrameRequest();
  request.parameter_slot_capacity = request.dispatches.size() - 1;
  ok &= Require(!BuildFrameExecutionPlan(request, &plan, &error) &&
                    error.find("parameter slot capacity") != std::string::npos,
                "planner should reject insufficient parameter slots");

  request = FullFrameRequest();
  request.host_readbacks = {"not_written"};
  ok &= Require(!BuildFrameExecutionPlan(request, &plan, &error) &&
                    error.find("without a frame writer") != std::string::npos,
                "planner should reject invalid completion readback");
  return ok;
}

bool TestMutatedPlanValidation() {
  FrameExecutionPlan plan;
  std::string error;
  bool ok = Require(BuildFrameExecutionPlan(FullFrameRequest(), &plan, &error),
                    "baseline plan should build: " + error);
  if (!ok)
    return false;

  plan.dispatches[1].parameter_slot = plan.dispatches[0].parameter_slot;
  ok &= Require(!ValidateFrameExecutionPlan(plan, &error) &&
                    error.find("reuses parameter slot") != std::string::npos,
                "validation should detect parameter slot reuse");

  BuildFrameExecutionPlan(FullFrameRequest(), &plan, &error);
  plan.dispatches[1].descriptor_slot = plan.dispatches[0].descriptor_slot;
  ok &= Require(!ValidateFrameExecutionPlan(plan, &error) &&
                    error.find("reuses descriptor slot") != std::string::npos,
                "validation should detect descriptor slot reuse");

  BuildFrameExecutionPlan(FullFrameRequest(), &plan, &error);
  plan.requires_synchronous_completion_before_slot_reuse = false;
  ok &= Require(!ValidateFrameExecutionPlan(plan, &error) &&
                    error.find("does not protect slot reuse") !=
                        std::string::npos,
                "validation should reject unimplemented asynchronous reuse");
  return ok;
}

bool TestExplicitInitialStatesAndWriteReuse() {
  FrameExecutionRequest request;
  request.initial_resources = {
      FrameInitialResource{"host_upload",
                           FrameInitialResourceState::host_write},
      FrameInitialResource{"transfer_upload",
                           FrameInitialResourceState::transfer_write},
  };
  request.dispatches = {
      {FrameExecutionStage::preprocess,
       "consume_uploads",
       {"host_upload", "transfer_upload"},
       {"seed"}},
      {FrameExecutionStage::alpha_resize, "first_write", {"seed"}, {"reused"}},
      {FrameExecutionStage::alpha_feather, "overwrite", {"seed"}, {"reused"}},
      {FrameExecutionStage::composite,
       "consume_reused",
       {"reused"},
       {"output"}},
  };
  request.host_readbacks = {"output"};
  request.parameter_slot_capacity = request.dispatches.size();
  request.descriptor_slot_capacity = request.dispatches.size();
  request.allow_resource_reuse = true;

  FrameExecutionPlan plan;
  std::string error;
  bool ok = Require(BuildFrameExecutionPlan(request, &plan, &error),
                    "explicit-state frame plan should build: " + error);
  bool host_to_compute = false;
  bool transfer_to_compute = false;
  bool write_to_write = false;
  bool reused_to_read = false;
  bool final_host = false;
  for (const auto &barrier : plan.barriers) {
    host_to_compute |=
        barrier.kind == FrameBarrierKind::host_write_to_compute_read &&
        barrier.resource == "host_upload" && barrier.consumer_boundary == 0;
    transfer_to_compute |=
        barrier.kind == FrameBarrierKind::transfer_write_to_compute_read &&
        barrier.resource == "transfer_upload" && barrier.consumer_boundary == 0;
    write_to_write |=
        barrier.kind == FrameBarrierKind::compute_write_to_compute_write &&
        barrier.resource == "reused" && barrier.producer_dispatch == 1 &&
        barrier.consumer_boundary == 2;
    reused_to_read |=
        barrier.kind == FrameBarrierKind::compute_write_to_compute_read &&
        barrier.resource == "reused" && barrier.producer_dispatch == 2 &&
        barrier.consumer_boundary == 3;
    final_host |=
        barrier.kind == FrameBarrierKind::compute_write_to_host_read &&
        barrier.resource == "output" &&
        barrier.consumer_boundary == plan.completion_boundary;
  }
  ok &= Require(host_to_compute && transfer_to_compute && write_to_write &&
                    reused_to_read && final_host,
                "planner should model host/transfer writes, compute "
                "write-to-write/read, and final host visibility");
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok &= TestFullFramePlan();
  ok &= TestInvalidRequests();
  ok &= TestMutatedPlanValidation();
  ok &= TestExplicitInitialStatesAndWriteReuse();
  return ok ? 0 : 1;
}
