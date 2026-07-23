#include "core/maxine/hal_resident_adapter.h"
#include "maxine/fake_resident_frame_executor.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>

namespace {

using namespace studiocast;
using maxine::CpuTailKind;
using maxine::HostRgbFrameView;
using maxine::MaxineFrameOutputBinding;
using maxine::MaxineHalAdapterConfiguration;
using maxine::MaxineHalAdapterHooks;
using maxine::MaxineHalResidentAdapter;
using maxine::MaxineOutputBoundary;
using maxine::ResidentBoundaryResult;
using maxine::ResidentReadbackBoundary;
using maxine::ResidentStageKind;
using maxine::testing::FakeResidentFrameExecutor;

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

constexpr std::uint64_t kAdapterId = 41;
constexpr std::uint64_t kContextGeneration = 5;

hal::ExecutionDomainId Domain(std::uint64_t adapter_id = kAdapterId) {
  hal::ExecutionDomainId domain{};
  domain.backend = hal::BackendId::maxine;
  domain.api_family = hal::ApiFamily::maxine_nvcv;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = 0x1cbb;
  domain.physical_device.stable_id[0] = 0xA5;
  domain.logical_context_generation = kContextGeneration;
  domain.allocator_family = hal::AllocatorFamily::maxine_nvcv;
  domain.memory_handle_family = hal::MemoryHandleFamily::nvcv_image;
  domain.queue_or_stream_identity = 12;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::maxine_bound_cuda_stream;
  domain.adapter_instance_id = adapter_id;
  return domain;
}

hal::ImageDescriptor ResidentImageDescriptor() {
  hal::ImageDescriptor image{};
  image.width = 2;
  image.height = 2;
  image.plane_count = 1;
  image.format = hal::PixelFormat::bgr_u8;
  image.channel_order = hal::ChannelOrder::bgr;
  image.color_space = hal::ColorSpace::srgb;
  image.color_range = hal::ColorRange::full;
  image.orientation = hal::Orientation::top_left;
  image.host_coherence = hal::HostCoherence::not_host_visible;
  image.mutability = hal::Mutability::read_write;
  image.aliasing = hal::AliasingPermission::forbidden;
  image.base_alignment_bytes = 1;
  image.planes[0] = hal::PlaneDescriptor{0, 6, 12, 2, 2, 1};
  return image;
}

hal::PreparedStage Stage(std::uint32_t id, hal::SemanticEffectId semantic,
                         hal::SemanticEffectProviderId provider,
                         bool required) {
  hal::PreparedStage stage{};
  stage.stage_id = id;
  stage.semantic = semantic;
  stage.provider = provider;
  stage.backend = hal::BackendId::maxine;
  stage.execution_domain = Domain();
  stage.input.kind = hal::StageIoKind::image;
  stage.input.image = ResidentImageDescriptor();
  stage.output = stage.input;
  stage.required = required;
  stage.failure_action =
      required ? hal::FailureAction::fail_closed
               : hal::FailureAction::fail_open_preserve_last_valid;
  stage.circuit_breaker.consecutive_failure_limit = 1;
  stage.circuit_breaker.action =
      required ? hal::CircuitBreakerAction::quarantine_program
               : hal::CircuitBreakerAction::disable_stage;
  stage.model.model_id_hash[0] = 0x31;
  stage.model.artifact_set_hash[0] = 0x57;
  stage.model.artifact_generation = 1;
  return stage;
}

hal::PreparedProgram Program() {
  hal::PreparedProgram program{};
  program.program_id = 71;
  program.owner_adapter_instance_id = kAdapterId;
  program.execution_domain = Domain();
  program.requested_backend = hal::BackendId::maxine;
  program.strict_backend_selection = true;
  program.invalidation_key.generations =
      hal::ProgramGenerations{1, 1, 1, 1, 1, kContextGeneration};
  program.invalidation_key.canonical_plan_hash = 0xAA;
  program.invalidation_key.model_artifact_hash = 0xBB;
  program.invalidation_key.execution_domain_hash = 0xCC;
  program.budget.persistent_bytes = 4096;
  program.budget.scratch_bytes = 4096;
  program.budget.upload_staging_bytes = 4096;
  program.budget.readback_staging_bytes = 4096;
  program.budget.resource_count = 16;
  program.budget.in_flight_slots = 1;
  program.stages[0] = Stage(1, hal::SemanticEffectId::virtual_background_blur,
                            hal::SemanticEffectProviderId::maxine_vfx, true);
  program.stages[1] = Stage(2, hal::SemanticEffectId::virtual_key_light,
                            hal::SemanticEffectProviderId::maxine_vfx, true);
  program.stage_count = 2;
  return program;
}

MaxineHalAdapterConfiguration
Configuration(std::uint64_t adapter_id = kAdapterId) {
  MaxineHalAdapterConfiguration config{};
  config.adapter_instance_id = adapter_id;
  config.execution_domain = Domain(adapter_id);
  config.resident_key = maxine::ResidentFrameKey{1, 2, 2, 11, 12};
  config.capacity.persistent_bytes = 8192;
  config.capacity.scratch_bytes = 8192;
  config.capacity.upload_staging_bytes = 8192;
  config.capacity.readback_staging_bytes = 8192;
  config.capacity.resource_count = 32;
  config.capacity.in_flight_slots = 1;
  return config;
}

struct HooksFixture {
  FakeResidentFrameExecutor *executor = nullptr;
  std::array<std::uint8_t, 64> output{};
  std::uint64_t capture_sequence = 0;
  hal::ReasonId failure_reason = hal::ReasonId::none;
  std::uint64_t reset_calls = 0;
  std::uint64_t shutdown_calls = 0;
  ResidentReadbackBoundary boundary = ResidentReadbackBoundary::final_output;
  bool suppress_download_success = false;
  bool hide_upload = false;
  bool hidden_intermediate_readback = false;

  static hal::ReasonId Classify(void *owner, ResidentBoundaryResult) noexcept {
    return static_cast<HooksFixture *>(owner)->failure_reason;
  }
  static bool Reset(void *owner, hal::ReasonId) noexcept {
    ++static_cast<HooksFixture *>(owner)->reset_calls;
    return true;
  }
  static bool Query(void *owner, ResidentReadbackBoundary boundary,
                    std::uint64_t capture_sequence,
                    MaxineFrameOutputBinding *output) noexcept {
    auto &self = *static_cast<HooksFixture *>(owner);
    if (!output)
      return false;
    self.boundary = boundary;
    output->version = maxine::kMaxineHalBindingVersion;
    output->capture_sequence = capture_sequence;
    output->host_bgr = self.output.data();
    output->width = 2;
    output->height = 2;
    output->stride_bytes = 6;
    return true;
  }
  static bool QueryTelemetry(
      void *owner,
      maxine::MaxineAuthoritativeTelemetrySnapshot *snapshot) noexcept {
    auto &self = *static_cast<HooksFixture *>(owner);
    if (!snapshot || !self.executor)
      return false;
    const auto &calls = self.executor->calls();
    snapshot->version = maxine::kMaxineHalBindingVersion;
    snapshot->logical_context_generation = kContextGeneration;
    snapshot->stream_identity = 12;
    snapshot->host_uploads = self.hide_upload ? 0 : calls.upload;
    const std::uint64_t successful_downloads =
        calls.download -
        (self.suppress_download_success && calls.download != 0 ? 1 : 0);
    if (calls.last_download_boundary ==
        ResidentReadbackBoundary::cpu_continuation) {
      snapshot->cpu_tail_readbacks = successful_downloads;
    } else {
      snapshot->final_readbacks = successful_downloads;
    }
    snapshot->synchronizations = calls.synchronize;
    snapshot->intermediate_readbacks =
        self.hidden_intermediate_readback && calls.upload != 0 ? 1 : 0;
    snapshot->device_lost = self.failure_reason == hal::ReasonId::device_lost &&
                            calls.download != 0;
    return true;
  }
  static void Shutdown(void *owner) noexcept {
    ++static_cast<HooksFixture *>(owner)->shutdown_calls;
  }

  MaxineHalAdapterHooks Hooks(FakeResidentFrameExecutor &fake,
                              bool temporal = true) {
    executor = &fake;
    MaxineHalAdapterHooks hooks{};
    hooks.owner = this;
    hooks.classify_failure = Classify;
    hooks.reset_temporal = temporal ? Reset : nullptr;
    hooks.query_host_output = Query;
    hooks.query_authoritative_telemetry = QueryTelemetry;
    hooks.shutdown = Shutdown;
    return hooks;
  }
};

struct HostFixture {
  std::array<std::uint8_t, 64> input{};
  HostRgbFrameView View() const {
    return HostRgbFrameView{input.data(), 2, 2, 6};
  }
};

bool Configure(hal::PreparedProgramSession &session,
               MaxineHalResidentAdapter &adapter,
               const hal::PreparedProgram &program) {
  hal::ReadinessSnapshot readiness{};
  hal::ReasonId reason = hal::ReasonId::none;
  CHECK(session.Configure(program, adapter, &readiness, &reason) ==
        hal::StatusId::ready);
  CHECK(readiness.available);
  CHECK(readiness.selected_backend == hal::BackendId::maxine);
  CHECK(reason == hal::ReasonId::none);
  return true;
}

hal::ExecuteRequest Request(MaxineHalResidentAdapter &adapter,
                            const hal::PreparedProgram &program,
                            const HostRgbFrameView &frame,
                            std::uint64_t sequence,
                            std::uint64_t resource_id = 101) {
  hal::ResourceHandle input{};
  hal::ReasonId reason = hal::ReasonId::none;
  if (adapter.BindHostFrame(frame, sequence, resource_id, &input, &reason) !=
      hal::StatusId::ready) {
    std::abort();
  }
  hal::ExecuteRequest request{};
  request.invalidation_key = program.invalidation_key;
  request.execution_domain = program.execution_domain;
  request.capture_sequence = sequence;
  request.input = input;
  return request;
}

bool ValidCombinedProgramUsesOneBoundaryAndCommitsAttribution() {
  FakeResidentFrameExecutor executor{};
  HooksFixture hooks{};
  MaxineHalResidentAdapter adapter(Configuration(), executor,
                                   hooks.Hooks(executor));
  hal::PreparedProgramSession session{};
  auto program = Program();
  CHECK(Configure(session, adapter, program));
  CHECK(executor.calls().prepare == 1);

  HostFixture host{};
  hooks.capture_sequence = 1;
  const auto result =
      session.Execute(Request(adapter, program, host.View(), 1));
  CHECK(result.status == hal::StatusId::executed);
  CHECK(result.reason == hal::ReasonId::none);
  CHECK(result.output_valid && result.completion.complete);
  CHECK(result.stage_count == 2);
  CHECK(result.stages[0].status == hal::StageExecutionStatus::succeeded);
  CHECK(result.stages[1].status == hal::StageExecutionStatus::succeeded);
  CHECK(result.transfers.uploads == 1);
  CHECK(result.transfers.final_readbacks == 1);
  CHECK(result.transfers.intermediate_uploads == 0);
  CHECK(result.transfers.intermediate_readbacks == 0);
  CHECK(result.transfers.synchronizations == 1);
  CHECK(executor.calls().prepare == 1);
  CHECK(executor.calls().upload == 1 && executor.calls().download == 1);
  CHECK(executor.calls().matte == 1);
  CHECK(executor.calls().stage_count == 2);
  CHECK(executor.calls().stage_kinds[0] == ResidentStageKind::background_blur);
  CHECK(executor.calls().stage_kinds[1] == ResidentStageKind::relighting);
  CHECK(executor.calls().stage_input_identities[1] ==
        executor.calls().stage_output_identities[0]);
  CHECK(session.telemetry().active_backend == hal::BackendId::invalid);

  MaxineFrameOutputBinding output{};
  CHECK(adapter.ResolveOutput(result.output, &output));
  CHECK(output.ValidHost());
  CHECK(output.boundary == MaxineOutputBoundary::final_readback);
  for (std::size_t mutation = 0; mutation < 5; ++mutation) {
    auto forged_output = result.output;
    switch (mutation) {
    case 0:
      ++forged_output.size_bytes;
      break;
    case 1:
      forged_output.memory_domain = hal::MemoryDomain::device_local;
      break;
    case 2:
      forged_output.lifetime = hal::ResourceLifetime::frame_borrowed;
      break;
    case 3:
      forged_output.synchronization_state =
          hal::SynchronizationState::completion_pending;
      break;
    case 4:
      ++forged_output.execution_domain.logical_context_generation;
      break;
    }
    CHECK(!adapter.ResolveOutput(forged_output, &output));
  }

  hal::ReasonId reason = hal::ReasonId::none;
  auto reconfigured = program;
  ++reconfigured.program_id;
  hal::ReadinessSnapshot blocked_readiness{};
  CHECK(session.Configure(reconfigured, adapter, &blocked_readiness, &reason) ==
        hal::StatusId::blocked);
  CHECK(reason == hal::ReasonId::output_not_committed);
  CHECK(session.program().program_id == program.program_id);
  CHECK(session.CommitOutput(hal::OutputCommit{result.completion, true},
                             &reason) == hal::StatusId::ready);
  CHECK(session.telemetry().active_backend == hal::BackendId::maxine);
  CHECK(session.telemetry().active_provider ==
        hal::SemanticEffectProviderId::maxine_vfx);
  CHECK(session.telemetry().committed_output_frames == 1);
  CHECK(session.telemetry().active_backend_frames == 1);
  return true;
}

bool NonStrictFallbackIsDegradedUntilCommitted() {
  FakeResidentFrameExecutor executor{};
  HooksFixture hooks{};
  MaxineHalResidentAdapter adapter(Configuration(), executor,
                                   hooks.Hooks(executor));
  hal::PreparedProgramSession session{};
  auto program = Program();
  program.requested_backend = hal::BackendId::cuda;
  program.strict_backend_selection = false;
  hal::ReadinessSnapshot readiness{};
  hal::ReasonId reason = hal::ReasonId::none;
  CHECK(session.Configure(program, adapter, &readiness, &reason) ==
        hal::StatusId::degraded);
  CHECK(readiness.status == hal::StatusId::degraded);
  CHECK(readiness.degraded);
  CHECK(readiness.requested_backend == hal::BackendId::cuda);
  CHECK(readiness.selected_backend == hal::BackendId::maxine);
  CHECK(readiness.reason == hal::ReasonId::semantic_provider_unavailable);
  CHECK(reason == hal::ReasonId::none);
  CHECK(session.telemetry().fallback_frames == 0);

  HostFixture host{};
  hooks.capture_sequence = 1;
  const auto result =
      session.Execute(Request(adapter, program, host.View(), 1));
  CHECK(result.status == hal::StatusId::executed);
  CHECK(session.telemetry().fallback_frames == 0);
  CHECK(session.CommitOutput({result.completion, true}, &reason) ==
        hal::StatusId::ready);
  CHECK(session.telemetry().fallback_frames == 1);
  CHECK(session.telemetry().requested_backend == hal::BackendId::cuda);
  CHECK(session.telemetry().selected_backend == hal::BackendId::maxine);
  CHECK(session.telemetry().active_backend == hal::BackendId::maxine);
  return true;
}

bool CpuTailIsExplicitAndMeasured() {
  FakeResidentFrameExecutor executor{};
  HooksFixture hooks{};
  MaxineHalResidentAdapter adapter(Configuration(), executor,
                                   hooks.Hooks(executor));
  hal::PreparedProgramSession session{};
  auto program = Program();
  program.stage_count = 1;
  program.cpu_tails[0] = hal::DeclaredCpuTail{
      hal::CpuTailId::output_format_conversion, 1, 0, true};
  program.cpu_tail_count = 1;
  CHECK(Configure(session, adapter, program));

  HostFixture host{};
  hooks.capture_sequence = 2;
  const auto result =
      session.Execute(Request(adapter, program, host.View(), 2));
  CHECK(result.status == hal::StatusId::executed);
  CHECK(result.reason == hal::ReasonId::cpu_tail_required);
  CHECK(result.transfers.uploads == 1);
  CHECK(result.transfers.final_readbacks == 1);
  CHECK(result.transfers.cpu_tail_boundaries == 1);
  CHECK(hooks.boundary == ResidentReadbackBoundary::cpu_continuation);
  MaxineFrameOutputBinding output{};
  CHECK(adapter.ResolveOutput(result.output, &output));
  CHECK(output.boundary == MaxineOutputBoundary::cpu_tail_readback);
  return true;
}

bool OptionalFailurePreservesCurrentAndRequiredFailureStops() {
  HostFixture host{};
  {
    FakeResidentFrameExecutor executor{};
    executor.FailStage(ResidentStageKind::denoise,
                       ResidentBoundaryResult::runtime_failure);
    HooksFixture hooks{};
    MaxineHalResidentAdapter adapter(Configuration(), executor,
                                     hooks.Hooks(executor));
    hal::PreparedProgramSession session{};
    auto program = Program();
    program.stages[0] = Stage(1, hal::SemanticEffectId::video_noise_removal,
                              hal::SemanticEffectProviderId::maxine_vfx, false);
    program.stages[1] =
        Stage(2, hal::SemanticEffectId::vignette,
              hal::SemanticEffectProviderId::maxine_cuda_utility, true);
    CHECK(Configure(session, adapter, program));
    hooks.capture_sequence = 1;
    const auto result =
        session.Execute(Request(adapter, program, host.View(), 1));
    CHECK(result.status == hal::StatusId::degraded);
    CHECK(result.reason == hal::ReasonId::optional_stage_failed_open);
    CHECK(result.stages[0].status ==
          hal::StageExecutionStatus::optional_failed_open);
    CHECK(result.stages[0].valid_output_preserved);
    CHECK(result.stages[1].status == hal::StageExecutionStatus::succeeded);
    CHECK(executor.calls().stage_input_identities[1] ==
          executor.calls().uploaded_image_identity);
  }
  {
    FakeResidentFrameExecutor executor{};
    executor.FailStage(ResidentStageKind::denoise,
                       ResidentBoundaryResult::runtime_failure);
    HooksFixture hooks{};
    MaxineHalResidentAdapter adapter(Configuration(), executor,
                                     hooks.Hooks(executor));
    hal::PreparedProgramSession session{};
    auto program = Program();
    program.stage_count = 1;
    program.stages[0] = Stage(1, hal::SemanticEffectId::video_noise_removal,
                              hal::SemanticEffectProviderId::maxine_vfx, true);
    CHECK(Configure(session, adapter, program));
    hooks.capture_sequence = 1;
    const auto result =
        session.Execute(Request(adapter, program, host.View(), 1));
    CHECK(result.status == hal::StatusId::failed);
    CHECK(result.reason == hal::ReasonId::required_stage_failed);
    CHECK(!result.output_valid);
    CHECK(result.stages[0].status ==
          hal::StageExecutionStatus::required_failed);
  }
  return true;
}

bool StrictBackendProviderDomainAndGenerationReject() {
  FakeResidentFrameExecutor executor{};
  HooksFixture hooks{};
  MaxineHalResidentAdapter adapter(Configuration(), executor,
                                   hooks.Hooks(executor));
  hal::PreparedProgramHandle handle{};
  hal::ReadinessSnapshot readiness{};
  hal::ReasonId reason = hal::ReasonId::none;

  auto foreign = Program();
  foreign.requested_backend = hal::BackendId::cuda;
  CHECK(adapter.Prepare(foreign, &handle, &readiness, &reason) ==
        hal::StatusId::blocked);
  CHECK(reason == hal::ReasonId::strict_backend_unavailable);

  auto provider = Program();
  provider.stages[0].provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  CHECK(adapter.Prepare(provider, &handle, &readiness, &reason) ==
        hal::StatusId::blocked);
  CHECK(reason == hal::ReasonId::semantic_provider_unavailable);

  auto vignette_provider = Program();
  vignette_provider.stage_count = 1;
  vignette_provider.stages[0] =
      Stage(1, hal::SemanticEffectId::vignette,
            hal::SemanticEffectProviderId::maxine_vfx, true);
  vignette_provider.stages[0].model = {};
  CHECK(adapter.Prepare(vignette_provider, &handle, &readiness, &reason) ==
        hal::StatusId::blocked);
  CHECK(reason == hal::ReasonId::semantic_provider_unavailable);
  vignette_provider.stages[0].provider =
      hal::SemanticEffectProviderId::maxine_cuda_utility;
  CHECK(adapter.Prepare(vignette_provider, &handle, &readiness, &reason) ==
        hal::StatusId::ready);
  adapter.DestroyPrepared(&handle);

  auto missing_model = Program();
  missing_model.stages[0].model = {};
  CHECK(adapter.Prepare(missing_model, &handle, &readiness, &reason) ==
        hal::StatusId::blocked);
  CHECK(reason == hal::ReasonId::capability_evidence_incomplete);

  auto cross_adapter = Program();
  cross_adapter.stages[1].execution_domain.adapter_instance_id = 99;
  CHECK(adapter.Prepare(cross_adapter, &handle, &readiness, &reason) ==
        hal::StatusId::blocked);
  CHECK(reason == hal::ReasonId::cross_adapter_residency_forbidden);

  hal::PreparedProgramSession session{};
  auto valid = Program();
  CHECK(Configure(session, adapter, valid));
  HostFixture host{};
  hooks.capture_sequence = 1;
  auto wrong_domain = Request(adapter, valid, host.View(), 1);
  wrong_domain.execution_domain.queue_or_stream_identity = 99;
  CHECK(session.Execute(wrong_domain).reason ==
        hal::ReasonId::wrong_execution_domain);

  hooks.capture_sequence = 2;
  auto wrong_generation = Request(adapter, valid, host.View(), 2, 102);
  --wrong_generation.input.lifetime_generation;
  CHECK(session.Execute(wrong_generation).reason ==
        hal::ReasonId::wrong_generation);

  hooks.capture_sequence = 3;
  auto forged = Request(adapter, valid, host.View(), 3, 103);
  forged.input.adapter_opaque[0] ^= 1;
  CHECK(session.Execute(forged).reason == hal::ReasonId::invalid_resource);

  hooks.capture_sequence = 4;
  auto mutated_metadata = Request(adapter, valid, host.View(), 4, 104);
  mutated_metadata.input.memory_domain = hal::MemoryDomain::device_local;
  CHECK(session.Execute(mutated_metadata).reason ==
        hal::ReasonId::invalid_resource);
  return true;
}

bool ResourceExhaustionMalformedBindingAndOldPlanRetention() {
  FakeResidentFrameExecutor executor{};
  HooksFixture hooks{};
  auto small = Configuration();
  small.capacity.persistent_bytes = 32;
  MaxineHalResidentAdapter small_adapter(small, executor,
                                         hooks.Hooks(executor));
  auto program = Program();
  hal::PreparedProgramHandle handle{};
  hal::ReadinessSnapshot readiness{};
  hal::ReasonId reason = hal::ReasonId::none;
  CHECK(small_adapter.Prepare(program, &handle, &readiness, &reason) ==
        hal::StatusId::resource_exhausted);
  CHECK(reason == hal::ReasonId::resource_exhausted);

  {
    MaxineHalResidentAdapter slot_adapter(Configuration(), executor,
                                          hooks.Hooks(executor));
    std::array<hal::PreparedProgramHandle, maxine::kMaxineHalPreparedSlots>
        handles{};
    for (std::size_t index = 0; index < handles.size(); ++index) {
      auto candidate_program = program;
      candidate_program.program_id += index;
      CHECK(slot_adapter.Prepare(candidate_program, &handles[index], &readiness,
                                 &reason) == hal::StatusId::ready);
    }
    auto exhausted_program = program;
    exhausted_program.program_id += handles.size();
    CHECK(slot_adapter.Prepare(exhausted_program, &handle, &readiness,
                               &reason) == hal::StatusId::resource_exhausted);
    CHECK(reason == hal::ReasonId::resource_exhausted);
    for (auto &prepared : handles)
      slot_adapter.DestroyPrepared(&prepared);
  }

  MaxineHalResidentAdapter adapter(Configuration(), executor,
                                   hooks.Hooks(executor));
  HostFixture host{};
  hal::ResourceHandle input{};
  auto malformed = host.View();
  malformed.stride_bytes = 1;
  CHECK(adapter.BindHostFrame(malformed, 1, 1, &input, &reason) ==
        hal::StatusId::blocked);
  CHECK(reason == hal::ReasonId::malformed_descriptor);
  malformed = host.View();
  malformed.data = reinterpret_cast<const std::uint8_t *>(
      std::numeric_limits<std::uintptr_t>::max() - 4);
  CHECK(adapter.BindHostFrame(malformed, 1, 1, &input, &reason) ==
        hal::StatusId::blocked);
  CHECK(reason == hal::ReasonId::malformed_descriptor);

  hal::PreparedProgramSession session{};
  CHECK(Configure(session, adapter, program));
  auto invalid_candidate = program;
  ++invalid_candidate.program_id;
  invalid_candidate.stages[0].provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  CHECK(session.Configure(invalid_candidate, adapter, &readiness, &reason) ==
        hal::StatusId::blocked);
  CHECK(session.prepared());
  CHECK(session.program().program_id == program.program_id);

  hooks.capture_sequence = 4;
  const auto result =
      session.Execute(Request(adapter, program, host.View(), 4));
  CHECK(result.status == hal::StatusId::executed);
  return true;
}

bool TemporalResetDeviceLossAndShutdownAreTyped() {
  HostFixture host{};
  {
    FakeResidentFrameExecutor executor{};
    HooksFixture hooks{};
    MaxineHalResidentAdapter adapter(Configuration(), executor,
                                     hooks.Hooks(executor));
    hal::PreparedProgramSession session{};
    auto program = Program();
    program.temporal.temporal = true;
    program.temporal.requires_contiguous_capture_sequence = true;
    program.temporal.discontinuity_action =
        hal::TemporalDiscontinuityAction::reset_and_continue;
    program.temporal.frame_drop_behavior =
        hal::FrameDropBehavior::reset_temporal_state;
    program.temporal.history_frames = 2;
    CHECK(Configure(session, adapter, program));
    hooks.capture_sequence = 1;
    auto first = session.Execute(Request(adapter, program, host.View(), 1));
    CHECK(first.status == hal::StatusId::executed);
    hal::ReasonId reason = hal::ReasonId::none;
    CHECK(session.CommitOutput({first.completion, true}, &reason) ==
          hal::StatusId::ready);
    hooks.capture_sequence = 3;
    const auto discontinuous =
        session.Execute(Request(adapter, program, host.View(), 3, 103));
    CHECK(discontinuous.status == hal::StatusId::executed);
    CHECK(hooks.reset_calls == 1);
    CHECK(session.telemetry().temporal_resets == 1);
  }
  {
    FakeResidentFrameExecutor executor{};
    executor.FailDownload(ResidentBoundaryResult::runtime_failure);
    HooksFixture hooks{};
    hooks.failure_reason = hal::ReasonId::device_lost;
    hooks.suppress_download_success = true;
    MaxineHalResidentAdapter adapter(Configuration(), executor,
                                     hooks.Hooks(executor));
    hal::PreparedProgramSession session{};
    auto program = Program();
    CHECK(Configure(session, adapter, program));
    hooks.capture_sequence = 1;
    const auto lost =
        session.Execute(Request(adapter, program, host.View(), 1));
    CHECK(lost.status == hal::StatusId::quarantined);
    CHECK(lost.reason == hal::ReasonId::device_lost);
    CHECK(session.quarantined());
    CHECK(session.telemetry().device_losses == 1);
    session.Shutdown();
    CHECK(hooks.shutdown_calls == 1);
    CHECK(session.Execute({}).status == hal::StatusId::stopped);
  }
  return true;
}

bool AutoFrameRequiresDeclaredCpuTailAndTemporalNeedsResetHook() {
  FakeResidentFrameExecutor executor{};
  HooksFixture hooks{};
  MaxineHalResidentAdapter adapter(Configuration(), executor,
                                   hooks.Hooks(executor));
  auto program = Program();
  program.stage_count = 1;
  program.stages[0] = Stage(1, hal::SemanticEffectId::auto_frame,
                            hal::SemanticEffectProviderId::maxine_ar, true);
  hal::PreparedProgramHandle handle{};
  hal::ReadinessSnapshot readiness{};
  hal::ReasonId reason = hal::ReasonId::none;
  CHECK(adapter.Prepare(program, &handle, &readiness, &reason) ==
        hal::StatusId::blocked);
  CHECK(reason == hal::ReasonId::cpu_tail_required);

  program.cpu_tails[0] =
      hal::DeclaredCpuTail{hal::CpuTailId::auto_frame_tracking, 1, 0, true};
  program.cpu_tail_count = 1;
  CHECK(adapter.Prepare(program, &handle, &readiness, &reason) ==
        hal::StatusId::ready);
  adapter.DestroyPrepared(&handle);

  MaxineHalResidentAdapter no_reset_adapter(Configuration(), executor,
                                            hooks.Hooks(executor, false));
  auto temporal = Program();
  temporal.temporal.temporal = true;
  CHECK(no_reset_adapter.Prepare(temporal, &handle, &readiness, &reason) ==
        hal::StatusId::blocked);
  CHECK(reason == hal::ReasonId::capability_evidence_incomplete);
  return true;
}

bool AuthoritativeTelemetryIsMandatoryAndRejectsHiddenBoundaries() {
  auto program = Program();
  hal::PreparedProgramHandle handle{};
  hal::ReadinessSnapshot readiness{};
  hal::ReasonId reason = hal::ReasonId::none;
  {
    FakeResidentFrameExecutor executor{};
    HooksFixture hooks{};
    auto incomplete_hooks = hooks.Hooks(executor);
    incomplete_hooks.query_authoritative_telemetry = nullptr;
    MaxineHalResidentAdapter adapter(Configuration(), executor,
                                     incomplete_hooks);
    CHECK(adapter.Prepare(program, &handle, &readiness, &reason) ==
          hal::StatusId::blocked);
    CHECK(reason == hal::ReasonId::capability_evidence_incomplete);
  }

  HostFixture host{};
  {
    FakeResidentFrameExecutor executor{};
    HooksFixture hooks{};
    hooks.hide_upload = true;
    MaxineHalResidentAdapter adapter(Configuration(), executor,
                                     hooks.Hooks(executor));
    hal::PreparedProgramSession session{};
    CHECK(Configure(session, adapter, program));
    hooks.capture_sequence = 1;
    const auto result =
        session.Execute(Request(adapter, program, host.View(), 1));
    CHECK(result.status == hal::StatusId::failed);
    CHECK(result.reason == hal::ReasonId::synchronization_failed);
    CHECK(!result.output_valid);
  }
  {
    FakeResidentFrameExecutor executor{};
    HooksFixture hooks{};
    hooks.hidden_intermediate_readback = true;
    MaxineHalResidentAdapter adapter(Configuration(), executor,
                                     hooks.Hooks(executor));
    hal::PreparedProgramSession session{};
    CHECK(Configure(session, adapter, program));
    hooks.capture_sequence = 1;
    const auto result =
        session.Execute(Request(adapter, program, host.View(), 1));
    CHECK(result.status == hal::StatusId::failed);
    CHECK(result.reason == hal::ReasonId::synchronization_failed);
    CHECK(result.transfers.intermediate_readbacks == 1);
    CHECK(!result.output_valid);
  }
  return true;
}

bool MatteFingerprintTracksPreparedInvalidationIdentity() {
  FakeResidentFrameExecutor executor{};
  HooksFixture hooks{};
  MaxineHalResidentAdapter adapter(Configuration(), executor,
                                   hooks.Hooks(executor));
  hal::PreparedProgramSession session{};
  auto first_program = Program();
  CHECK(Configure(session, adapter, first_program));
  HostFixture host{};
  hooks.capture_sequence = 1;
  const auto first =
      session.Execute(Request(adapter, first_program, host.View(), 1));
  CHECK(first.status == hal::StatusId::executed);
  const auto first_fingerprint = executor.calls().last_matte_fingerprint;
  CHECK(first_fingerprint != 0);
  hal::ReasonId reason = hal::ReasonId::none;
  CHECK(session.CommitOutput({first.completion, true}, &reason) ==
        hal::StatusId::ready);

  auto second_program = first_program;
  ++second_program.program_id;
  ++second_program.invalidation_key.canonical_plan_hash;
  ++second_program.invalidation_key.generations.provider;
  ++second_program.stages[0].model.artifact_set_hash[0];
  second_program.stages[1].model = second_program.stages[0].model;
  CHECK(Configure(session, adapter, second_program));
  hooks.capture_sequence = 2;
  const auto second =
      session.Execute(Request(adapter, second_program, host.View(), 2, 102));
  CHECK(second.status == hal::StatusId::executed);
  CHECK(executor.calls().last_matte_fingerprint != first_fingerprint);
  CHECK(executor.calls().matte == 2);
  return true;
}

bool ExecuteIsAllocationFreeAfterPrepare() {
  FakeResidentFrameExecutor executor{};
  HooksFixture hooks{};
  MaxineHalResidentAdapter adapter(Configuration(), executor,
                                   hooks.Hooks(executor));
  hal::PreparedProgramSession session{};
  auto program = Program();
  CHECK(Configure(session, adapter, program));
  HostFixture host{};
  hal::ReasonId reason = hal::ReasonId::none;

  hooks.capture_sequence = 1;
  auto warm = session.Execute(Request(adapter, program, host.View(), 1));
  CHECK(warm.status == hal::StatusId::executed);
  CHECK(session.CommitOutput({warm.completion, true}, &reason) ==
        hal::StatusId::ready);

  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations = true;
  bool succeeded = true;
  for (std::uint64_t sequence = 2; sequence != 130; ++sequence) {
    hooks.capture_sequence = sequence;
    const auto result = session.Execute(
        Request(adapter, program, host.View(), sequence, 1000 + sequence));
    if (result.status != hal::StatusId::executed ||
        session.CommitOutput({result.completion, true}, &reason) !=
            hal::StatusId::ready) {
      succeeded = false;
      break;
    }
  }
  g_count_allocations = false;
  CHECK(succeeded);
  CHECK(g_allocations.load(std::memory_order_relaxed) == 0);
  CHECK(executor.calls().prepare == 1);
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
  const std::array tests{
      ValidCombinedProgramUsesOneBoundaryAndCommitsAttribution,
      NonStrictFallbackIsDegradedUntilCommitted,
      CpuTailIsExplicitAndMeasured,
      OptionalFailurePreservesCurrentAndRequiredFailureStops,
      StrictBackendProviderDomainAndGenerationReject,
      ResourceExhaustionMalformedBindingAndOldPlanRetention,
      TemporalResetDeviceLossAndShutdownAreTyped,
      AutoFrameRequiresDeclaredCpuTailAndTemporalNeedsResetHook,
      AuthoritativeTelemetryIsMandatoryAndRejectsHiddenBoundaries,
      MatteFingerprintTracksPreparedInvalidationIdentity,
      ExecuteIsAllocationFreeAfterPrepare,
  };
  for (const auto test : tests) {
    if (!test())
      return EXIT_FAILURE;
  }
  std::cout << "Maxine HAL resident adapter tests passed\n";
  return EXIT_SUCCESS;
}
