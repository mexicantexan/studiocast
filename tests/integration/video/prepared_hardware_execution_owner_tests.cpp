#include "core/video/prepared_hardware_execution_owner.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>

namespace {

using namespace studiocast;

std::atomic<bool> g_count_allocations{false};
std::atomic<std::uint64_t> g_allocations{0};

bool Require(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

hal::ExecutionDomainId Domain(std::uint64_t adapter = 77) {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::vulkan;
  domain.api_family = hal::ApiFamily::vulkan;
  domain.physical_device.vendor_id = 0x1002;
  domain.physical_device.device_id = 0x73bf;
  domain.physical_device.stable_id[0] = 0xa5;
  domain.logical_context_generation = 19;
  domain.allocator_family = hal::AllocatorFamily::vulkan;
  domain.memory_handle_family = hal::MemoryHandleFamily::vulkan_buffer;
  domain.queue_or_stream_identity = 0x1234;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::vulkan_queue_fence;
  domain.adapter_instance_id = adapter;
  return domain;
}

hal::ResourceHandle Resource(const hal::PreparedProgram &program,
                             std::uint64_t id, hal::MemoryDomain memory_domain,
                             std::uint64_t size_bytes) {
  hal::ResourceHandle resource;
  resource.resource_id = id;
  resource.allocator_owner_id = program.owner_adapter_instance_id;
  resource.lifetime_generation =
      program.execution_domain.logical_context_generation;
  resource.memory_domain = memory_domain;
  resource.lifetime = hal::ResourceLifetime::frame_borrowed;
  resource.synchronization_state = hal::SynchronizationState::ready;
  resource.execution_domain = program.execution_domain;
  resource.size_bytes = size_bytes;
  resource.alignment_bytes = 1;
  resource.mutable_resource = true;
  return resource;
}

enum class ExecuteMode {
  success,
  optional_fail_open,
  device_loss,
  foreign_output,
};

class FakeVulkanAdapter final : public hal::IResidentAdapter {
public:
  std::uint64_t adapter_instance_id() const noexcept override { return 77; }
  hal::BackendId backend_id() const noexcept override {
    return hal::BackendId::vulkan;
  }

  hal::StatusId Prepare(const hal::PreparedProgram &program,
                        hal::PreparedProgramHandle *candidate,
                        hal::ReadinessSnapshot *readiness,
                        hal::ReasonId *reason) noexcept override {
    ++prepare_calls;
    if (fail_prepare) {
      if (reason)
        *reason = hal::ReasonId::resource_exhausted;
      return hal::StatusId::blocked;
    }
    program_ = program;
    candidate->handle_id = 100 + prepare_calls;
    candidate->program_id = program.program_id;
    candidate->adapter_instance_id = adapter_instance_id();
    candidate->invalidation_key = program.invalidation_key;
    candidate->execution_domain = program.execution_domain;
    candidate->reserved_budget = program.budget;
    readiness->status = hal::StatusId::ready;
    readiness->reason = hal::ReasonId::none;
    readiness->requested_backend = hal::BackendId::vulkan;
    readiness->selected_backend = hal::BackendId::vulkan;
    readiness->provider =
        hal::SemanticEffectProviderId::open_vulkan_production_kernels;
    readiness->execution_domain = program.execution_domain;
    readiness->ready_stage_mask = (std::uint32_t{1} << program.stage_count) - 1;
    readiness->available = true;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::StatusId BindHostFrame(const hal::PreparedProgramHandle &,
                              const hal::HostFrameBinding &binding,
                              hal::HostFrameTokens *tokens,
                              hal::ReasonId *reason) noexcept override {
    ++bind_calls;
    binding_ = binding;
    tokens->input = Resource(program_, 1000 + binding.capture_sequence,
                             hal::MemoryDomain::upload_staging,
                             binding.input_planes[0].span_bytes);
    tokens->output = Resource(program_, 2000 + binding.capture_sequence,
                              hal::MemoryDomain::readback_staging,
                              binding.output_planes[0].span_bytes);
    tokens_ = *tokens;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::ExecuteResult
  Execute(const hal::PreparedProgramHandle &,
          const hal::ExecuteRequest &request) noexcept override {
    ++execute_calls;
    hal::ExecuteResult result;
    result.status = hal::StatusId::executed;
    result.completion.token_id = next_token_++;
    result.completion.program_id = program_.program_id;
    result.completion.adapter_instance_id = adapter_instance_id();
    result.completion.capture_sequence = request.capture_sequence;
    result.completion.complete = true;
    result.output = tokens_.output;
    result.output_valid = true;
    result.stage_count = program_.stage_count;
    result.transfers.uploads = 1;
    result.transfers.final_readbacks = 1;
    result.transfers.synchronizations = 1;
    const std::size_t bytes =
        static_cast<std::size_t>(binding_.output_planes[0].span_bytes);
    std::memcpy(binding_.output_planes[0].write_data,
                binding_.input_planes[0].read_data, bytes);
    for (std::size_t index = 0; index < result.stage_count; ++index) {
      auto &stage = result.stages[index];
      stage.stage_id = program_.stages[index].stage_id;
      stage.semantic = program_.stages[index].semantic;
      stage.status = hal::StageExecutionStatus::succeeded;
      stage.valid_output_preserved = true;
      stage.elapsed_ns = 100 + index;
      if ((request.disabled_stage_mask & (std::uint32_t{1} << index)) != 0) {
        stage.status = hal::StageExecutionStatus::disabled_preserved;
        stage.reason = hal::ReasonId::circuit_breaker_open;
        stage.elapsed_ns = 0;
      }
    }
    switch (execute_mode) {
    case ExecuteMode::success:
      break;
    case ExecuteMode::optional_fail_open:
      result.status = hal::StatusId::degraded;
      result.reason = hal::ReasonId::optional_stage_failed_open;
      result.stages[0].status = hal::StageExecutionStatus::optional_failed_open;
      result.stages[0].reason = hal::ReasonId::optional_stage_failed_open;
      break;
    case ExecuteMode::device_loss:
      result.status = hal::StatusId::failed;
      result.reason = hal::ReasonId::device_lost;
      result.output_valid = false;
      break;
    case ExecuteMode::foreign_output:
      ++result.output.execution_domain.adapter_instance_id;
      break;
    }
    return result;
  }

  hal::StatusId ResolveHostFrameOutput(
      const hal::PreparedProgramHandle &, const hal::CompletionToken &,
      const hal::ResourceHandle &, hal::HostFrameOutputResolution *resolution,
      hal::ReasonId *reason) noexcept override {
    ++resolve_calls;
    resolution->descriptor = binding_.output_descriptor;
    resolution->planes = binding_.output_planes;
    resolution->plane_count = binding_.output_plane_count;
    resolution->capture_sequence = binding_.capture_sequence;
    resolution->output_ready = true;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::StatusId ResetTemporalState(const hal::PreparedProgramHandle &,
                                   hal::ReasonId) noexcept override {
    return hal::StatusId::ready;
  }
  void DestroyPrepared(hal::PreparedProgramHandle *handle) noexcept override {
    ++destroy_calls;
    *handle = {};
  }
  void Shutdown() noexcept override { ++shutdown_calls; }

  bool fail_prepare = false;
  ExecuteMode execute_mode = ExecuteMode::success;
  std::uint64_t prepare_calls = 0;
  std::uint64_t bind_calls = 0;
  std::uint64_t execute_calls = 0;
  std::uint64_t resolve_calls = 0;
  std::uint64_t destroy_calls = 0;
  std::uint64_t shutdown_calls = 0;

private:
  std::uint64_t next_token_ = 1;
  hal::PreparedProgram program_{};
  hal::HostFrameBinding binding_{};
  hal::HostFrameTokens tokens_{};
};

hal::SemanticCapabilityEvidence Capability(hal::SemanticEffectId semantic) {
  hal::SemanticCapabilityEvidence capability;
  capability.semantic = semantic;
  capability.provider =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  capability.status = hal::StatusId::ready;
  capability.reason = hal::ReasonId::none;
  capability.live_stage = true;
  capability.production_eligible = true;
  capability.exact_evidence = true;
  return capability;
}

hal::AdapterRegistration Registration(FakeVulkanAdapter *adapter) {
  hal::AdapterRegistration registration;
  registration.adapter_instance_id = adapter->adapter_instance_id();
  registration.backend = hal::BackendId::vulkan;
  registration.execution_domain = Domain();
  registration.selection_priority = 10;
  registration.resident_adapter = adapter;
  registration.semantic_capabilities[0] =
      Capability(hal::SemanticEffectId::vignette);
  registration.semantic_capabilities[1] =
      Capability(hal::SemanticEffectId::mirror);
  registration.semantic_capability_count = 2;
  return registration;
}

video::PreparedFinalHardwareRequest Request(bool vignette = true,
                                            bool mirror = true) {
  video::PreparedFinalHardwareRequest request;
  request.requested_backend = hal::BackendId::vulkan;
  request.strict_backend_selection = true;
  request.request_fixed_center_vignette = vignette;
  request.request_mirror = mirror;
  request.host_input_is_tight_rgb24 = true;
  request.host_output_is_tight_rgb24 = true;
  request.input_width = 4;
  request.input_height = 2;
  request.output_width = 4;
  request.output_height = 2;
  request.vignette_intensity_percent = 45;
  request.generations.configuration = 1;
  request.generations.geometry = 1;
  return request;
}

bool TestEligibility() {
  bool ok = true;
  auto request = Request();
  ok &= Require(video::EvaluatePreparedFinalHardwareEligibility(request) ==
                    video::PreparedHardwareEligibility::eligible,
                "qualified strict Vulkan request was not eligible");
  request.strict_backend_selection = false;
  ok &= Require(
      video::EvaluatePreparedFinalHardwareEligibility(request) ==
          video::PreparedHardwareEligibility::not_strict_explicit_vulkan,
      "automatic selection entered live route");
  request = Request(false, false);
  ok &=
      Require(video::EvaluatePreparedFinalHardwareEligibility(request) ==
                  video::PreparedHardwareEligibility::no_qualified_final_stage,
              "empty final section entered live route");
  request = Request();
  request.contains_unqualified_stage = true;
  ok &= Require(
      video::EvaluatePreparedFinalHardwareEligibility(request) ==
          video::PreparedHardwareEligibility::contains_unqualified_stage,
      "unqualified stage entered live route");
  request = Request();
  request.tracked_vignette_center = true;
  ok &= Require(video::EvaluatePreparedFinalHardwareEligibility(request) ==
                    video::PreparedHardwareEligibility::tracked_vignette_center,
                "tracked vignette entered fixed-center route");
  request = Request();
  request.host_output_is_tight_rgb24 = false;
  ok &= Require(video::EvaluatePreparedFinalHardwareEligibility(request) ==
                    video::PreparedHardwareEligibility::unsupported_host_layout,
                "padded/non-RGB output entered live route");
  request = Request();
  request.generations.geometry = 0;
  ok &= Require(video::EvaluatePreparedFinalHardwareEligibility(request) ==
                    video::PreparedHardwareEligibility::invalid_generation,
                "zero geometry generation entered live route");
  ok &= Require(video::PreparedHardwareReasonCode(hal::ReasonId::device_lost) ==
                        "device_lost" &&
                    video::PreparedHardwareReasonCode(
                        hal::ReasonId::optional_stage_failed_open) ==
                        "optional_stage_failed_open",
                "stable compatibility reason codes changed");
  return ok;
}

bool TestExecutionCommitAndTelemetry() {
  FakeVulkanAdapter adapter;
  const auto registration = Registration(&adapter);
  video::PreparedHardwareExecutionOwner owner(registration);
  const auto setup = owner.ConfigureFinal(Request());
  bool ok = Require(setup.route == video::PreparedHardwareRoute::hal_live,
                    "eligible setup did not enter live route");
  ok &= Require(owner.active_request().host_input_is_tight_rgb24 &&
                    owner.active_request().host_output_is_tight_rgb24 &&
                    owner.active_request().generations.geometry == 1,
                "active request lost measured transport/generation facts");
  ok &= Require(!setup.setup.shadow_compared,
                "owner mislabeled HAL self-comparison as legacy parity");

  std::array<std::uint8_t, 24> input{};
  std::array<std::uint8_t, 24> output{};
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<std::uint8_t>(index);
  auto frame = owner.ExecuteHostFrame(input.data(), input.size(), 12,
                                      output.data(), output.size(), 12, 1);
  ok &= Require(frame.output_ready && frame.completion_pending,
                "executed frame was not completion-shaped");
  hal::ReasonId reason = hal::ReasonId::none;
  ok &= Require(owner.CommitOutput(frame.completion, true, &reason) ==
                    hal::StatusId::ready,
                "successful writer commit was rejected");
  auto snapshot = owner.Snapshot();
  ok &= Require(snapshot.telemetry.committed_output_frames == 1 &&
                    snapshot.telemetry.active_backend_frames == 1 &&
                    snapshot.telemetry.transfers.uploads == 1 &&
                    snapshot.telemetry.transfers.final_readbacks == 1 &&
                    snapshot.telemetry.transfers.intermediate_readbacks == 0 &&
                    snapshot.telemetry.transfers.intermediate_uploads == 0 &&
                    snapshot.telemetry.transfers.cpu_tail_boundaries == 0 &&
                    snapshot.latest_committed_frame.valid &&
                    snapshot.latest_committed_frame.stage_count == 2,
                "committed telemetry or exact transfer counts were wrong");
  const auto compatibility =
      video::ProjectPreparedHardwareCompatibilityTelemetry(snapshot);
  ok &= Require(compatibility.active_frames == 1 &&
                    compatibility.upload_calls == 1 &&
                    compatibility.download_calls == 1 &&
                    compatibility.final_download_calls == 1 &&
                    compatibility.forced_sync_calls == 1 &&
                    compatibility.fallback_frames == 0 &&
                    compatibility.runtime_failure_frames == 0,
                "legacy compatibility telemetry projection was not exact");
  ok &= Require(input == output, "resolved output did not reach writer buffer");

  frame = owner.ExecuteHostFrame(input.data(), input.size(), 12, output.data(),
                                 output.size(), 12, 2);
  ok &= Require(owner.CommitOutput(frame.completion, false, &reason) ==
                        hal::StatusId::failed &&
                    reason == hal::ReasonId::output_write_failed,
                "failed writer was not recorded as uncommitted");
  snapshot = owner.Snapshot();
  ok &= Require(snapshot.telemetry.committed_output_frames == 1 &&
                    snapshot.latest_committed_frame.capture_sequence == 1,
                "unwritten execution changed committed attribution");
  return ok;
}

bool TestRollbackAndDomainFailure() {
  FakeVulkanAdapter adapter;
  video::PreparedHardwareExecutionOwner owner(Registration(&adapter));
  bool ok = Require(owner.ConfigureFinal(Request()).route ==
                        video::PreparedHardwareRoute::hal_live,
                    "initial rollback setup failed");
  adapter.fail_prepare = true;
  auto same = Request();
  same.generations.configuration = 2;
  const auto retained = owner.ConfigureFinal(same);
  ok &= Require(retained.route == video::PreparedHardwareRoute::hal_live &&
                    retained.retained_previous_live_program,
                "same-semantics setup failure did not retain old program");
  auto changed = same;
  changed.output_width = 8;
  const auto rejected = owner.ConfigureFinal(changed);
  ok &= Require(rejected.route == video::PreparedHardwareRoute::legacy_shadow &&
                    !owner.Snapshot().prepared,
                "geometry-changing setup failure retained stale resources");

  auto tracked = Request();
  tracked.tracked_vignette_center = true;
  const auto tracked_setup = owner.ConfigureFinal(tracked);
  ok &= Require(
      tracked_setup.route == video::PreparedHardwareRoute::legacy_shadow &&
          tracked_setup.eligibility ==
              video::PreparedHardwareEligibility::tracked_vignette_center,
      "tracked-center pipeline request entered fixed-center HAL");

  adapter.fail_prepare = false;
  ok &= Require(owner.ConfigureFinal(Request()).route ==
                    video::PreparedHardwareRoute::hal_live,
                "reconfigure after invalidation failed");
  adapter.execute_mode = ExecuteMode::foreign_output;
  std::array<std::uint8_t, 24> input{};
  std::array<std::uint8_t, 24> output{};
  const auto frame = owner.ExecuteHostFrame(
      input.data(), input.size(), 12, output.data(), output.size(), 12, 3);
  ok &= Require(!frame.output_ready &&
                    frame.reason == hal::ReasonId::invalid_resource,
                "cross-adapter output residency was not rejected");
  return ok;
}

bool TestOptionalFailureAndDeviceLoss() {
  FakeVulkanAdapter adapter;
  video::PreparedHardwareExecutionOwner owner(Registration(&adapter));
  bool ok = Require(owner.ConfigureFinal(Request()).route ==
                        video::PreparedHardwareRoute::hal_live,
                    "failure-mode setup failed");
  std::array<std::uint8_t, 24> input{};
  std::array<std::uint8_t, 24> output{};
  adapter.execute_mode = ExecuteMode::optional_fail_open;
  auto frame = owner.ExecuteHostFrame(input.data(), input.size(), 12,
                                      output.data(), output.size(), 12, 1);
  ok &= Require(frame.output_ready && frame.status == hal::StatusId::degraded,
                "optional failure did not preserve valid output");
  hal::ReasonId reason = hal::ReasonId::none;
  ok &= Require(owner.CommitOutput(frame.completion, true, &reason) ==
                    hal::StatusId::ready,
                "preserved optional output was not committable");
  ok &= Require(
      owner.Snapshot().latest_committed_frame.stages[0].status ==
              hal::StageExecutionStatus::optional_failed_open &&
          owner.Snapshot().latest_committed_frame.stages[0].reason ==
              hal::ReasonId::optional_stage_failed_open,
      "optional degradation was not available for legacy status projection");
  adapter.execute_mode = ExecuteMode::device_loss;
  frame = owner.ExecuteHostFrame(input.data(), input.size(), 12, output.data(),
                                 output.size(), 12, 2);
  const auto failed_compat =
      video::ProjectPreparedHardwareCompatibilityTelemetry(owner.Snapshot());
  ok &= Require(!frame.output_ready &&
                    frame.reason == hal::ReasonId::device_lost &&
                    owner.Snapshot().telemetry.device_losses == 1 &&
                    failed_compat.runtime_failure_frames == 1,
                "device loss was not quarantined and reported");
  const auto quarantined = owner.ExecuteHostFrame(
      input.data(), input.size(), 12, output.data(), output.size(), 12, 3);
  ok &= Require(!quarantined.output_ready &&
                    quarantined.status == hal::StatusId::quarantined,
                "quarantined program executed again");
  return ok;
}

bool TestNoFrameAllocations() {
  FakeVulkanAdapter adapter;
  video::PreparedHardwareExecutionOwner owner(Registration(&adapter));
  bool ok = Require(owner.ConfigureFinal(Request()).route ==
                        video::PreparedHardwareRoute::hal_live,
                    "allocation test setup failed");
  std::array<std::uint8_t, 24> input{};
  std::array<std::uint8_t, 24> output{};
  hal::ReasonId reason = hal::ReasonId::none;
  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations.store(true, std::memory_order_release);
  for (std::uint64_t sequence = 1; sequence <= 128; ++sequence) {
    const auto frame =
        owner.ExecuteHostFrame(input.data(), input.size(), 12, output.data(),
                               output.size(), 12, sequence);
    if (!frame.output_ready ||
        owner.CommitOutput(frame.completion, true, &reason) !=
            hal::StatusId::ready) {
      ok = false;
      break;
    }
  }
  g_count_allocations.store(false, std::memory_order_release);
  ok &= Require(g_allocations.load(std::memory_order_relaxed) == 0,
                "prepared frame loop allocated");
  ok &= Require(owner.Snapshot().telemetry.committed_output_frames == 128,
                "allocation loop did not commit every frame");
  return ok;
}

} // namespace

void *operator new(std::size_t size) {
  if (g_count_allocations.load(std::memory_order_acquire))
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *memory = std::malloc(size))
    return memory;
  throw std::bad_alloc();
}

void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }

int main() {
  bool ok = true;
  ok &= TestEligibility();
  ok &= TestExecutionCommitAndTelemetry();
  ok &= TestRollbackAndDomainFailure();
  ok &= TestOptionalFailureAndDeviceLoss();
  ok &= TestNoFrameAllocations();
  if (ok)
    std::cout << "prepared hardware execution owner tests passed\n";
  return ok ? 0 : 1;
}
