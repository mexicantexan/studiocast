#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "core/open_video/vulkan_matting_runtime.h"

namespace {

using studiocast::open_vulkan::VulkanMattingBufferBinding;
using studiocast::open_vulkan::VulkanMattingDeviceContext;
using studiocast::open_vulkan::VulkanMattingGraphDescriptor;
using studiocast::open_vulkan::VulkanMattingPersistentResources;
using studiocast::open_vulkan::VulkanMattingRuntime;
using studiocast::open_vulkan::VulkanMattingRuntimeEvidence;
using studiocast::open_vulkan::VulkanMattingRuntimeFailure;
using studiocast::open_vulkan::VulkanMattingRuntimeLifecycle;
using studiocast::open_vulkan::VulkanMattingRuntimeResult;

bool Expect(bool condition, const std::string &message) {
  if (!condition)
    std::cerr << message << "\n";
  return condition;
}

VulkanMattingDeviceContext Device() {
  VulkanMattingDeviceContext device;
  device.ownership_domain = reinterpret_cast<void *>(0x1000);
  device.physical_device = 0x2000;
  device.logical_device = 0x3000;
  device.queue = 0x4000;
  device.queue_family_index = 7;
  device.vendor_id = 0x8086;
  device.device_id = 0x1234;
  device.selected_device_index = 1;
  device.stable_device_id = "v1:8086:1234:integrated:test";
  return device;
}

VulkanMattingGraphDescriptor Graph() {
  VulkanMattingGraphDescriptor graph;
  graph.param_path = "/models/model.ncnn.param";
  graph.bin_path = "/models/model.ncnn.bin";
  graph.param_sha256 = std::string(64, 'a');
  graph.bin_sha256 = std::string(64, 'b');
  graph.input_blob = "input";
  graph.output_blob = "alpha";
  graph.converter_name = "pnnx";
  graph.converter_version = "20260712";
  graph.precision = "fp32";
  graph.input_h = 2;
  graph.input_w = 3;
  graph.output_h = 2;
  graph.output_w = 3;
  return graph;
}

VulkanMattingPersistentResources Resources() {
  VulkanMattingPersistentResources resources;
  resources.input_bytes = 1u * 3u * 2u * 3u * sizeof(float);
  resources.alpha_bytes = 1u * 1u * 2u * 3u * sizeof(float);
  resources.allow_cpu_layers = false;
  resources.require_device_residency = true;
  return resources;
}

VulkanMattingBufferBinding InputBinding() {
  VulkanMattingBufferBinding binding;
  binding.ownership_domain = Device().ownership_domain;
  binding.logical_device = Device().logical_device;
  binding.buffer = 0x5000;
  binding.byte_size = Resources().input_bytes;
  binding.n = 1;
  binding.c = 3;
  binding.h = 2;
  binding.w = 3;
  binding.device_resident = true;
  return binding;
}

VulkanMattingBufferBinding AlphaBinding() {
  VulkanMattingBufferBinding binding;
  binding.ownership_domain = Device().ownership_domain;
  binding.logical_device = Device().logical_device;
  binding.buffer = 0x6000;
  binding.byte_size = Resources().alpha_bytes;
  binding.n = 1;
  binding.c = 1;
  binding.h = 2;
  binding.w = 3;
  binding.device_resident = true;
  return binding;
}

struct FakeState {
  int initialize_calls = 0;
  int graph_load_calls = 0;
  int allocation_calls = 0;
  int warmup_calls = 0;
  int run_calls = 0;
  VulkanMattingRuntimeFailure initialize_failure =
      VulkanMattingRuntimeFailure::none;
  VulkanMattingRuntimeFailure graph_failure = VulkanMattingRuntimeFailure::none;
  VulkanMattingRuntimeFailure allocation_failure =
      VulkanMattingRuntimeFailure::none;
  VulkanMattingRuntimeFailure warmup_failure =
      VulkanMattingRuntimeFailure::none;
  VulkanMattingRuntimeFailure run_failure = VulkanMattingRuntimeFailure::none;
  bool cpu_layers_used = false;
  bool identity_matches = true;
  bool residency_proven = true;
};

class FakeRuntime final : public VulkanMattingRuntime {
public:
  explicit FakeRuntime(FakeState *state) : state_(state) {
    evidence_.runtime_name = "fake-ncnn-vulkan";
  }

  VulkanMattingRuntimeResult
  Initialize(const VulkanMattingDeviceContext &device) override {
    ++state_->initialize_calls;
    if (state_->initialize_failure != VulkanMattingRuntimeFailure::none)
      return Failure(state_->initialize_failure);
    evidence_.runtime_created = true;
    evidence_.device_identity_matches = state_->identity_matches;
    evidence_.active_device = device;
    if (!state_->identity_matches)
      ++evidence_.active_device.queue_family_index;
    return VulkanMattingRuntimeResult::Success();
  }

  VulkanMattingRuntimeResult
  LoadGraph(const VulkanMattingGraphDescriptor &) override {
    ++state_->graph_load_calls;
    if (state_->graph_failure != VulkanMattingRuntimeFailure::none)
      return Failure(state_->graph_failure);
    evidence_.graph_loaded = true;
    evidence_.cpu_layers_used = state_->cpu_layers_used;
    return VulkanMattingRuntimeResult::Success();
  }

  VulkanMattingRuntimeResult PreparePersistentResources(
      const VulkanMattingPersistentResources &) override {
    ++state_->allocation_calls;
    if (state_->allocation_failure != VulkanMattingRuntimeFailure::none)
      return Failure(state_->allocation_failure);
    evidence_.persistent_resources_allocated = true;
    evidence_.input_device_resident = state_->residency_proven;
    evidence_.alpha_device_resident = state_->residency_proven;
    evidence_.output_device_resident = state_->residency_proven;
    return VulkanMattingRuntimeResult::Success();
  }

  VulkanMattingRuntimeResult Warmup() override {
    ++state_->warmup_calls;
    if (state_->warmup_failure != VulkanMattingRuntimeFailure::none)
      return Failure(state_->warmup_failure);
    evidence_.warmup_complete = true;
    return VulkanMattingRuntimeResult::Success();
  }

  VulkanMattingRuntimeResult
  Run(const VulkanMattingBufferBinding &,
      const VulkanMattingBufferBinding &) override {
    ++state_->run_calls;
    if (state_->run_failure != VulkanMattingRuntimeFailure::none)
      return Failure(state_->run_failure);
    return VulkanMattingRuntimeResult::Success();
  }

  VulkanMattingRuntimeEvidence Evidence() const override { return evidence_; }

private:
  VulkanMattingRuntimeResult Failure(VulkanMattingRuntimeFailure failure) {
    return VulkanMattingRuntimeResult::Failure(failure, "injected failure");
  }

  FakeState *state_;
  VulkanMattingRuntimeEvidence evidence_;
};

bool PrepareAndWarm(VulkanMattingRuntimeLifecycle *lifecycle,
                    std::string *error) {
  return lifecycle->Prepare(Device(), Graph(), Resources(), error) &&
         lifecycle->Warmup(error);
}

bool TestLifecycleDoesSetupAndWarmupExactlyOnce() {
  FakeState state;
  VulkanMattingRuntimeLifecycle lifecycle(
      std::make_unique<FakeRuntime>(&state));
  std::string error;
  if (!Expect(PrepareAndWarm(&lifecycle, &error), error) ||
      !Expect(lifecycle.Prepare(Device(), Graph(), Resources(), &error), error) ||
      !Expect(lifecycle.Warmup(&error), error)) {
    return false;
  }
  for (int i = 0; i < 5; ++i) {
    if (!Expect(lifecycle.Run(InputBinding(), AlphaBinding(), &error), error))
      return false;
  }
  return Expect(state.initialize_calls == 1, "initialize must run once") &&
         Expect(state.graph_load_calls == 1, "graph load must run once") &&
         Expect(state.allocation_calls == 1,
                "persistent allocation must run once") &&
         Expect(state.warmup_calls == 1, "warmup must run once") &&
         Expect(state.run_calls == 5, "run count mismatch") &&
         Expect(lifecycle.available(), "successful lifecycle must be available");
}

bool TestCpuLayersAreRejected() {
  FakeState state;
  state.cpu_layers_used = true;
  VulkanMattingRuntimeLifecycle lifecycle(
      std::make_unique<FakeRuntime>(&state));
  std::string error;
  return Expect(!lifecycle.Prepare(Device(), Graph(), Resources(), &error),
                "CPU layers must fail setup") &&
         Expect(lifecycle.latched_failure() ==
                    VulkanMattingRuntimeFailure::cpu_layer_rejected,
                "CPU-layer rejection must be classified") &&
         Expect(state.graph_load_calls == 1 && state.allocation_calls == 1,
                "CPU-layer evidence must be checked during setup");
}

bool TestDeviceIdentityMismatchIsRejected() {
  FakeState state;
  state.identity_matches = false;
  VulkanMattingRuntimeLifecycle lifecycle(
      std::make_unique<FakeRuntime>(&state));
  std::string error;
  return Expect(!lifecycle.Prepare(Device(), Graph(), Resources(), &error),
                "device mismatch must fail setup") &&
         Expect(lifecycle.latched_failure() ==
                    VulkanMattingRuntimeFailure::device_identity_mismatch,
                "device mismatch classification missing") &&
         Expect(state.graph_load_calls == 0,
                "graph must not load after device mismatch");
}

bool TestResidencyMustBeProven() {
  FakeState state;
  state.residency_proven = false;
  VulkanMattingRuntimeLifecycle lifecycle(
      std::make_unique<FakeRuntime>(&state));
  std::string error;
  return Expect(!lifecycle.Prepare(Device(), Graph(), Resources(), &error),
                "missing residency evidence must fail setup") &&
         Expect(lifecycle.latched_failure() ==
                    VulkanMattingRuntimeFailure::residency_check_failed,
                "residency classification missing");
}

bool TestForeignAndInvalidBindingsFailBeforeRuntime() {
  FakeState state;
  VulkanMattingRuntimeLifecycle lifecycle(
      std::make_unique<FakeRuntime>(&state));
  std::string error;
  if (!Expect(PrepareAndWarm(&lifecycle, &error), error))
    return false;
  auto alpha = AlphaBinding();
  alpha.ownership_domain = reinterpret_cast<void *>(0x9999);
  return Expect(!lifecycle.Run(InputBinding(), alpha, &error),
                "foreign output must fail") &&
         Expect(state.run_calls == 0,
                "runtime must not see invalid/foreign bindings") &&
         Expect(lifecycle.latched_failure() ==
                    VulkanMattingRuntimeFailure::device_identity_mismatch,
                "foreign binding failure classification missing");
}

bool TestNamedFatalFailuresLatchWithoutRetry() {
  const VulkanMattingRuntimeFailure failures[] = {
      VulkanMattingRuntimeFailure::out_of_memory,
      VulkanMattingRuntimeFailure::device_lost,
      VulkanMattingRuntimeFailure::invalid_graph,
      VulkanMattingRuntimeFailure::invalid_shape,
  };
  for (const auto failure : failures) {
    FakeState state;
    state.run_failure = failure;
    VulkanMattingRuntimeLifecycle lifecycle(
        std::make_unique<FakeRuntime>(&state));
    std::string first_error;
    if (!Expect(PrepareAndWarm(&lifecycle, &first_error), first_error) ||
        !Expect(!lifecycle.Run(InputBinding(), AlphaBinding(), &first_error),
                "injected fatal run failure must fail") ||
        !Expect(lifecycle.latched_failure() == failure,
                "fatal failure classification was not latched")) {
      return false;
    }
    std::string second_error;
    if (!Expect(!lifecycle.Run(InputBinding(), AlphaBinding(), &second_error),
                "latched failure must reject later frames") ||
        !Expect(state.run_calls == 1,
                "latched failure must not retry runtime execution") ||
        !Expect(second_error == first_error,
                "latched error must remain stable across frames")) {
      return false;
    }
  }
  return true;
}

bool TestInvalidPersistentAllocationShapeLatchesBeforeRuntime() {
  FakeState state;
  VulkanMattingRuntimeLifecycle lifecycle(
      std::make_unique<FakeRuntime>(&state));
  auto resources = Resources();
  ++resources.alpha_bytes;
  std::string error;
  return Expect(!lifecycle.Prepare(Device(), Graph(), resources, &error),
                "allocation/shape mismatch must fail") &&
         Expect(lifecycle.latched_failure() ==
                    VulkanMattingRuntimeFailure::invalid_shape,
                "allocation/shape mismatch must latch invalid_shape") &&
         Expect(state.initialize_calls == 0 && state.graph_load_calls == 0 &&
                    state.allocation_calls == 0,
                "invalid shapes must fail before runtime work");
}

bool TestIncompleteGraphMetadataLatchesBeforeRuntime() {
  FakeState state;
  VulkanMattingRuntimeLifecycle lifecycle(
      std::make_unique<FakeRuntime>(&state));
  auto graph = Graph();
  graph.output_blob.clear();
  std::string error;
  return Expect(!lifecycle.Prepare(Device(), graph, Resources(), &error),
                "incomplete graph metadata must fail") &&
         Expect(lifecycle.latched_failure() ==
                    VulkanMattingRuntimeFailure::invalid_graph,
                "incomplete graph metadata must latch invalid_graph") &&
         Expect(state.initialize_calls == 0 && state.graph_load_calls == 0,
                "incomplete graph metadata must fail before runtime work");
}

bool TestUnavailableAdapterFailsClosed() {
  VulkanMattingRuntimeLifecycle lifecycle(
      studiocast::open_vulkan::CreateUnavailableVulkanMattingRuntime(
          "production ncnn Vulkan matting is not built"));
  std::string error;
  return Expect(!lifecycle.Prepare(Device(), Graph(), Resources(), &error),
                "unavailable adapter must fail closed") &&
         Expect(error.find("not built") != std::string::npos,
                "unavailable reason must remain visible") &&
         Expect(!lifecycle.available(),
                "unavailable adapter must never report availability");
}

} // namespace

int main() {
  bool ok = true;
  ok = TestLifecycleDoesSetupAndWarmupExactlyOnce() && ok;
  ok = TestCpuLayersAreRejected() && ok;
  ok = TestDeviceIdentityMismatchIsRejected() && ok;
  ok = TestResidencyMustBeProven() && ok;
  ok = TestForeignAndInvalidBindingsFailBeforeRuntime() && ok;
  ok = TestNamedFatalFailuresLatchWithoutRetry() && ok;
  ok = TestInvalidPersistentAllocationShapeLatchesBeforeRuntime() && ok;
  ok = TestIncompleteGraphMetadataLatchesBeforeRuntime() && ok;
  ok = TestUnavailableAdapterFailsClosed() && ok;
  if (!ok)
    return 1;
  std::cout << "Vulkan matting runtime lifecycle tests passed.\n";
  return 0;
}
