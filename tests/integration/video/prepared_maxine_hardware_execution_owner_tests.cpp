#include "core/video/prepared_maxine_hardware_execution_owner.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>

namespace {

using namespace studiocast;

std::atomic<std::uint64_t> g_allocations{0};
thread_local bool g_count_allocations = false;

void *Allocate(std::size_t size) {
  if (void *memory = std::malloc(size == 0 ? 1 : size)) {
    if (g_count_allocations)
      g_allocations.fetch_add(1, std::memory_order_relaxed);
    return memory;
  }
  throw std::bad_alloc();
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << __func__ << ':' << __LINE__                                 \
                << ": CHECK failed: " << #condition << '\n';                   \
      return false;                                                            \
    }                                                                          \
  } while (false)

hal::ExecutionDomainId Domain(std::uint64_t adapter = 91) {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::maxine;
  domain.api_family = hal::ApiFamily::maxine_nvcv;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = 0x1cbb;
  domain.physical_device.stable_id[0] = 0x42;
  domain.logical_context_generation = 17;
  domain.allocator_family = hal::AllocatorFamily::maxine_nvcv;
  domain.memory_handle_family = hal::MemoryHandleFamily::nvcv_image;
  domain.queue_or_stream_identity = 0x3000;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::maxine_bound_cuda_stream;
  domain.adapter_instance_id = adapter;
  return domain;
}

hal::ResourceHandle Resource(const hal::PreparedProgram &program,
                             std::uint64_t id,
                             hal::MemoryDomain memory_domain,
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
  all_optional_fail_open,
  device_loss,
  foreign_output,
  wrong_generation,
};

class FakeMaxineAdapter final : public hal::IResidentAdapter {
public:
  std::uint64_t adapter_instance_id() const noexcept override { return 91; }
  hal::BackendId backend_id() const noexcept override {
    return hal::BackendId::maxine;
  }

  hal::StatusId Prepare(const hal::PreparedProgram &program,
                        hal::PreparedProgramHandle *candidate,
                        hal::ReadinessSnapshot *readiness,
                        hal::ReasonId *reason) noexcept override {
    ++prepare_calls;
    if (fail_prepare) {
      if (reason)
        *reason = hal::ReasonId::resource_exhausted;
      return hal::StatusId::resource_exhausted;
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
    readiness->requested_backend = hal::BackendId::maxine;
    readiness->selected_backend = hal::BackendId::maxine;
    readiness->execution_domain = program.execution_domain;
    readiness->ready_stage_mask =
        (std::uint32_t{1} << program.stage_count) - 1u;
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
    tokens->input =
        Resource(program_, 1000 + binding.capture_sequence,
                 hal::MemoryDomain::upload_staging,
                 binding.input_planes[0].span_bytes);
    tokens->input.mutable_resource = false;
    tokens->output =
        Resource(program_, 2000 + binding.capture_sequence,
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
    result.transfers.cpu_tail_boundaries = program_.cpu_tail_count;
    std::memcpy(binding_.output_planes[0].write_data,
                binding_.input_planes[0].read_data,
                static_cast<std::size_t>(
                    binding_.output_planes[0].span_bytes));
    for (std::size_t index = 0; index < result.stage_count; ++index) {
      auto &stage = result.stages[index];
      stage.stage_id = program_.stages[index].stage_id;
      stage.semantic = program_.stages[index].semantic;
      stage.status = hal::StageExecutionStatus::succeeded;
      stage.valid_output_preserved = true;
      if ((request.disabled_stage_mask & (std::uint32_t{1} << index)) != 0) {
        stage.status = hal::StageExecutionStatus::disabled_preserved;
        stage.reason = hal::ReasonId::circuit_breaker_open;
      }
    }
    switch (mode) {
    case ExecuteMode::success:
      break;
    case ExecuteMode::optional_fail_open:
      result.status = hal::StatusId::degraded;
      result.reason = hal::ReasonId::optional_stage_failed_open;
      result.stages[0].status =
          hal::StageExecutionStatus::optional_failed_open;
      result.stages[0].reason =
          hal::ReasonId::optional_stage_failed_open;
      break;
    case ExecuteMode::all_optional_fail_open:
      result.status = hal::StatusId::degraded;
      result.reason = hal::ReasonId::optional_stage_failed_open;
      for (std::size_t index = 0; index < result.stage_count; ++index) {
        result.stages[index].status =
            hal::StageExecutionStatus::optional_failed_open;
        result.stages[index].reason =
            hal::ReasonId::optional_stage_failed_open;
      }
      break;
    case ExecuteMode::device_loss:
      result.status = hal::StatusId::failed;
      result.reason = hal::ReasonId::device_lost;
      result.output_valid = false;
      break;
    case ExecuteMode::foreign_output:
      ++result.output.execution_domain.adapter_instance_id;
      break;
    case ExecuteMode::wrong_generation:
      ++result.output.lifetime_generation;
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
    return hal::StatusId::failed;
  }
  void DestroyPrepared(hal::PreparedProgramHandle *handle) noexcept override {
    ++destroy_calls;
    *handle = {};
  }
  void Shutdown() noexcept override { ++shutdown_calls; }

  bool fail_prepare = false;
  ExecuteMode mode = ExecuteMode::success;
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

hal::AdapterRegistration Registration(FakeMaxineAdapter *adapter) {
  hal::AdapterRegistration registration;
  registration.adapter_instance_id = adapter->adapter_instance_id();
  registration.backend = hal::BackendId::maxine;
  registration.execution_domain = Domain();
  registration.selection_priority = 300;
  registration.resident_adapter = adapter;
  return registration;
}

video::PreparedMaxineHardwareRequest Request() {
  video::PreparedMaxineHardwareRequest request;
  request.requested_backend = hal::BackendId::maxine;
  request.strict_backend_selection = true;
  request.single_resident_island = true;
  request.host_input_is_tight_rgb24 = true;
  request.host_output_is_tight_rgb24 = true;
  request.adapter_instance_id = 91;
  request.width = 8;
  request.height = 6;
  request.generations = {3, 4, 5, 6, 7, 0};
  request.vfx_model_directory = "/prepared/maxine/models";
  request.stages[0] = {
      3, hal::SemanticEffectId::virtual_background_blur, false};
  request.stages[1] = {
      6, hal::SemanticEffectId::virtual_key_light, false};
  request.stages[2] = {8, hal::SemanticEffectId::vignette, false};
  request.stage_count = 3;
  for (std::size_t index = 0; index < 2; ++index) {
    request.stages[index].model.model_id_hash[0] =
        static_cast<std::uint8_t>(0x31 + index);
    request.stages[index].model.artifact_set_hash[0] =
        static_cast<std::uint8_t>(0x51 + index);
    request.stages[index].model.artifact_generation = 5;
  }
  request.effects.virtual_background.mode =
      video::effects::VirtualBackgroundMode::blur;
  request.effects.virtual_key_light.enabled = true;
  request.effects.vignette.enabled = true;
  return request;
}

struct Frames {
  std::array<std::uint8_t, 8 * 6 * 3> input{};
  std::array<std::uint8_t, 8 * 6 * 3> output{};
};

bool EligibilityPreservesTemporalAndCanonicalCompatibilityRoutes() {
  auto request = Request();
  CHECK(video::EvaluatePreparedMaxineHardwareEligibility(request) ==
        video::PreparedMaxineHardwareEligibility::eligible);
  request.temporal_reset_required = true;
  CHECK(video::EvaluatePreparedMaxineHardwareEligibility(request) ==
        video::PreparedMaxineHardwareEligibility::eligible);
  request.temporal_reset_required = false;
  request.single_resident_island = false;
  CHECK(video::EvaluatePreparedMaxineHardwareEligibility(request) ==
        video::PreparedMaxineHardwareEligibility::
            multiple_resident_islands);
  request.single_resident_island = true;
  request.contains_other_execution_domain = true;
  CHECK(video::EvaluatePreparedMaxineHardwareEligibility(request) ==
        video::PreparedMaxineHardwareEligibility::cross_domain_chain);
  request.contains_other_execution_domain = false;
  request.stages[0].model = {};
  CHECK(video::EvaluatePreparedMaxineHardwareEligibility(request) ==
        video::PreparedMaxineHardwareEligibility::
            model_artifact_evidence_incomplete);
  request = Request();
  std::swap(request.stages[0], request.stages[1]);
  CHECK(video::EvaluatePreparedMaxineHardwareEligibility(request) ==
        video::PreparedMaxineHardwareEligibility::unsupported_stage);
  request = Request();
  request.stages[1] = {
      4, hal::SemanticEffectId::virtual_background_remove, false};
  CHECK(video::EvaluatePreparedMaxineHardwareEligibility(request) ==
        video::PreparedMaxineHardwareEligibility::unsupported_stage);
  request = Request();
  request.stages[0] = {
      8, hal::SemanticEffectId::vignette, false};
  request.stage_count = 1;
  CHECK(video::EvaluatePreparedMaxineHardwareEligibility(request) ==
        video::PreparedMaxineHardwareEligibility::eligible);
  return true;
}

bool ExactProgramIdentityRollbackAndResourceExhaustion() {
  FakeMaxineAdapter adapter;
  auto registration = Registration(&adapter);
  video::PreparedMaxineHardwareExecutionOwner owner(registration);
  auto request = Request();
  const auto setup = owner.Configure(request);
  if (setup.route != video::PreparedMaxineHardwareRoute::hal_live) {
    std::cerr << "setup failed status=" << static_cast<unsigned>(setup.status)
              << " reason=" << static_cast<unsigned>(setup.reason)
              << " eligibility="
              << static_cast<unsigned>(setup.eligibility)
              << " prepare_calls=" << adapter.prepare_calls << '\n';
  }
  CHECK(setup.route == video::PreparedMaxineHardwareRoute::hal_live);
  CHECK(setup.execution_domain == registration.execution_domain);
  auto snapshot = owner.Snapshot();
  CHECK(snapshot.prepared);
  CHECK(snapshot.active_program.execution_domain ==
        registration.execution_domain);
  CHECK(snapshot.active_program.invalidation_key.generations.configuration ==
        request.generations.configuration);
  CHECK(snapshot.active_program.invalidation_key.generations.geometry != 0);
  CHECK(snapshot.active_program.invalidation_key.generations.model != 0);
  CHECK(snapshot.active_program.invalidation_key.generations.provider != 0);
  CHECK(snapshot.active_program.invalidation_key.generations.device != 0);
  CHECK(snapshot.active_program.stage_count == request.stage_count);
  CHECK(snapshot.active_program.cpu_tail_count == 1);

  const auto retained_program = snapshot.active_program.program_id;
  adapter.fail_prepare = true;
  ++request.generations.configuration;
  const auto exhausted = owner.Configure(request);
  CHECK(exhausted.status == hal::StatusId::resource_exhausted);
  CHECK(exhausted.reason == hal::ReasonId::resource_exhausted);
  CHECK(exhausted.retained_previous_live_program);
  snapshot = owner.Snapshot();
  CHECK(snapshot.prepared);
  CHECK(snapshot.active_program.program_id == retained_program);

  auto foreign_registration = registration;
  ++foreign_registration.execution_domain.adapter_instance_id;
  video::PreparedMaxineHardwareExecutionOwner foreign(foreign_registration);
  CHECK(foreign.Configure(Request()).reason ==
        hal::ReasonId::wrong_execution_domain);
  return true;
}

bool WriterCommitOptionalFailureAndForeignResourcesFailClosed() {
  FakeMaxineAdapter adapter;
  auto registration = Registration(&adapter);
  video::PreparedMaxineHardwareExecutionOwner owner(registration);
  CHECK(owner.Configure(Request()).status == hal::StatusId::ready);
  Frames frames;
  frames.input.fill(0x4a);

  auto frame = owner.ExecuteHostFrame(
      frames.input.data(), frames.input.size(), 8 * 3,
      frames.output.data(), frames.output.size(), 8 * 3, 1);
  if (!frame.output_ready) {
    std::cerr << "frame failed status=" << static_cast<unsigned>(frame.status)
              << " reason=" << static_cast<unsigned>(frame.reason)
              << " pending=" << frame.completion_pending << '\n';
  }
  CHECK(frame.output_ready && frame.completion_pending);
  hal::ReasonId reason = hal::ReasonId::none;
  CHECK(frame.transfers.uploads == 1 &&
        frame.transfers.final_readbacks == 1 &&
        frame.transfers.cpu_tail_boundaries == 1);
  CHECK(frames.output == frames.input);
  CHECK(owner.CommitOutput(frame.completion, true, &reason) ==
        hal::StatusId::ready);
  const auto baseline_committed =
      owner.Snapshot().latest_committed_frame;
  CHECK(baseline_committed.valid &&
        baseline_committed.capture_sequence == 1);

  frames.input.fill(0x5b);
  frames.output.fill(0);
  frame = owner.ExecuteHostFrame(
      frames.input.data(), frames.input.size(), 8 * 3,
      frames.output.data(), frames.output.size(), 8 * 3, 2);
  CHECK(frame.output_ready && frames.output == frames.input);
  CHECK(owner.CommitOutput(frame.completion, false, &reason) ==
        hal::StatusId::failed);
  CHECK(reason == hal::ReasonId::output_write_failed);
  CHECK(owner.Snapshot().latest_committed_frame.program_id ==
        baseline_committed.program_id);
  CHECK(owner.Snapshot().latest_committed_frame.capture_sequence ==
        baseline_committed.capture_sequence);

  adapter.mode = ExecuteMode::optional_fail_open;
  frames.input.fill(0x6c);
  frames.output.fill(0);
  frame = owner.ExecuteHostFrame(
      frames.input.data(), frames.input.size(), 8 * 3,
      frames.output.data(), frames.output.size(), 8 * 3, 3);
  CHECK(frame.status == hal::StatusId::degraded);
  CHECK(frame.output_ready);
  CHECK(frames.output == frames.input);
  CHECK(frame.stages[0].status ==
        hal::StageExecutionStatus::optional_failed_open);
  CHECK(owner.CommitOutput(frame.completion, true, &reason) ==
        hal::StatusId::ready);
  const auto committed = owner.Snapshot().latest_committed_frame;
  CHECK(committed.valid && committed.capture_sequence == 3);
  CHECK(committed.execution_domain == registration.execution_domain);

  adapter.mode = ExecuteMode::foreign_output;
  frame = owner.ExecuteHostFrame(
      frames.input.data(), frames.input.size(), 8 * 3,
      frames.output.data(), frames.output.size(), 8 * 3, 4);
  CHECK(!frame.output_ready && !frame.completion_pending);
  CHECK(frame.reason == hal::ReasonId::wrong_execution_domain ||
        frame.reason == hal::ReasonId::invalid_resource);

  adapter.mode = ExecuteMode::wrong_generation;
  frame = owner.ExecuteHostFrame(
      frames.input.data(), frames.input.size(), 8 * 3,
      frames.output.data(), frames.output.size(), 8 * 3, 5);
  CHECK(!frame.output_ready && !frame.completion_pending);
  CHECK(frame.reason == hal::ReasonId::wrong_generation ||
        frame.reason == hal::ReasonId::invalid_resource);
  return true;
}

bool AllocationFreeFramesReconfigurationDeviceLossAndShutdown() {
  FakeMaxineAdapter adapter;
  auto registration = Registration(&adapter);
  video::PreparedMaxineHardwareExecutionOwner owner(registration);
  auto request = Request();
  CHECK(owner.Configure(request).status == hal::StatusId::ready);
  Frames frames;
  frames.input.fill(0x7c);
  hal::ReasonId reason = hal::ReasonId::none;

  auto warm = owner.ExecuteHostFrame(
      frames.input.data(), frames.input.size(), 8 * 3,
      frames.output.data(), frames.output.size(), 8 * 3, 1);
  CHECK(warm.output_ready);
  CHECK(owner.CommitOutput(warm.completion, true, &reason) ==
        hal::StatusId::ready);

  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations = true;
  bool success = true;
  for (std::uint64_t sequence = 2; sequence != 130; ++sequence) {
    const auto frame = owner.ExecuteHostFrame(
        frames.input.data(), frames.input.size(), 8 * 3,
        frames.output.data(), frames.output.size(), 8 * 3, sequence);
    if (!frame.output_ready || frame.transfers.uploads != 1 ||
        frame.transfers.final_readbacks != 1 ||
        frame.transfers.intermediate_uploads != 0 ||
        frame.transfers.intermediate_readbacks != 0 ||
        owner.CommitOutput(frame.completion, true, &reason) !=
            hal::StatusId::ready) {
      success = false;
      break;
    }
  }
  g_count_allocations = false;
  CHECK(success);
  CHECK(g_allocations.load(std::memory_order_relaxed) == 0);
  CHECK(adapter.execute_calls == 129);
  CHECK(owner.Snapshot().telemetry.committed_output_frames == 129);

  const auto before_reconfiguration = owner.Snapshot().active_program;
  ++request.generations.configuration;
  CHECK(owner.Configure(request).status == hal::StatusId::ready);
  const auto after_reconfiguration = owner.Snapshot().active_program;
  CHECK(after_reconfiguration.invalidation_key.generations.configuration ==
        request.generations.configuration);
  CHECK(after_reconfiguration.owner_adapter_instance_id ==
        before_reconfiguration.owner_adapter_instance_id);
  CHECK(after_reconfiguration.execution_domain ==
        before_reconfiguration.execution_domain);
  CHECK(after_reconfiguration.invalidation_key.generations.geometry ==
        before_reconfiguration.invalidation_key.generations.geometry);
  CHECK(after_reconfiguration.invalidation_key.generations.model ==
        before_reconfiguration.invalidation_key.generations.model);
  CHECK(after_reconfiguration.invalidation_key.generations.provider ==
        before_reconfiguration.invalidation_key.generations.provider);
  CHECK(after_reconfiguration.invalidation_key.generations.device ==
        before_reconfiguration.invalidation_key.generations.device);

  adapter.mode = ExecuteMode::all_optional_fail_open;
  const auto failed_open = owner.ExecuteHostFrame(
      frames.input.data(), frames.input.size(), 8 * 3,
      frames.output.data(), frames.output.size(), 8 * 3, 130);
  CHECK(failed_open.output_ready && failed_open.completion_pending);
  CHECK(failed_open.status == hal::StatusId::degraded);
  CHECK(owner.CommitOutput(failed_open.completion, true, &reason) ==
        hal::StatusId::ready);

  adapter.mode = ExecuteMode::success;
  const auto disabled = owner.ExecuteHostFrame(
      frames.input.data(), frames.input.size(), 8 * 3,
      frames.output.data(), frames.output.size(), 8 * 3, 131);
  if (!disabled.output_ready) {
    std::cerr << "disabled frame failed status="
              << static_cast<unsigned>(disabled.status)
              << " reason=" << static_cast<unsigned>(disabled.reason)
              << " pending=" << disabled.completion_pending << '\n';
  }
  CHECK(disabled.output_ready && disabled.completion_pending);
  CHECK(disabled.transfers.uploads == 1 &&
        disabled.transfers.final_readbacks == 1);
  for (std::size_t index = 0; index < disabled.stage_count; ++index) {
    CHECK(disabled.stages[index].status ==
          hal::StageExecutionStatus::disabled_preserved);
    CHECK(disabled.stages[index].reason ==
          hal::ReasonId::circuit_breaker_open);
  }
  CHECK(owner.CommitOutput(disabled.completion, true, &reason) ==
        hal::StatusId::ready);
  CHECK(owner.Snapshot().latest_committed_frame.valid);
  CHECK(owner.Snapshot().latest_committed_frame.capture_sequence == 131);
  CHECK(owner.Snapshot().telemetry.active_backend_frames == 129);

  adapter.mode = ExecuteMode::device_loss;
  const auto lost = owner.ExecuteHostFrame(
      frames.input.data(), frames.input.size(), 8 * 3,
      frames.output.data(), frames.output.size(), 8 * 3, 132);
  CHECK(lost.reason == hal::ReasonId::device_lost);
  owner.NotifyDeviceLost();
  CHECK(owner.Snapshot().quarantined);
  owner.Shutdown();
  CHECK(owner.Snapshot().stopped);
  return true;
}

} // namespace

void *operator new(std::size_t size) { return Allocate(size); }
void *operator new[](std::size_t size) { return Allocate(size); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}

int main() {
  if (!EligibilityPreservesTemporalAndCanonicalCompatibilityRoutes() ||
      !ExactProgramIdentityRollbackAndResourceExhaustion() ||
      !WriterCommitOptionalFailureAndForeignResourcesFailClosed() ||
      !AllocationFreeFramesReconfigurationDeviceLossAndShutdown()) {
    return EXIT_FAILURE;
  }
  std::cout << "Prepared Maxine hardware execution owner tests passed\n";
  return EXIT_SUCCESS;
}
