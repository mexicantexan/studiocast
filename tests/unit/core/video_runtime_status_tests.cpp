#include "core/video/video_runtime_status.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <string_view>
#include <type_traits>

namespace {

std::atomic<std::uint64_t> g_allocations{0};

[[nodiscard]] bool Expect(bool condition, std::string_view message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

using studiocast::hal::BackendId;
using studiocast::hal::ReasonId;
using studiocast::hal::SemanticEffectId;
using studiocast::hal::SemanticEffectProviderId;
using studiocast::hal::StageExecutionStatus;
using studiocast::hal::StatusId;
using studiocast::video::CapabilityRefreshSnapshot;
using studiocast::video::CapabilityRefreshStatus;
using studiocast::video::CapabilityValidationCode;
using studiocast::video::MultiIslandStatusValidationCode;
using studiocast::video::MultiIslandVideoRuntimeStatusBuildInput;
using studiocast::video::MultiIslandVideoRuntimeStatusSnapshot;
using studiocast::video::PreparedExecutionIslandStatus;
using studiocast::video::PreparedExecutionRouteId;
using studiocast::video::ProductCapabilityDisposition;
using studiocast::video::ProductCapabilityGenerations;
using studiocast::video::ProductCapabilityLane;
using studiocast::video::ProductCapabilityLaneId;
using studiocast::video::ProductCapabilitySnapshot;
using studiocast::video::StableEffectRuntimeState;
using studiocast::video::VideoRuntimeLifecycle;
using studiocast::video::VideoRuntimeStatusSnapshot;
using Code = studiocast::video::effects::EffectCapabilityCode;

static_assert(std::is_trivially_copyable_v<ProductCapabilitySnapshot>);
static_assert(std::is_trivially_copyable_v<VideoRuntimeStatusSnapshot>);
static_assert(std::is_trivially_destructible_v<ProductCapabilitySnapshot>);
static_assert(std::is_trivially_destructible_v<VideoRuntimeStatusSnapshot>);
static_assert(
    std::is_trivially_copyable_v<MultiIslandVideoRuntimeStatusSnapshot>);
static_assert(
    std::is_trivially_destructible_v<MultiIslandVideoRuntimeStatusSnapshot>);

ProductCapabilityGenerations Generations(std::uint64_t generation = 7) {
  ProductCapabilityGenerations out;
  out.capability = generation;
  out.configuration = 11;
  out.provider = 12;
  out.device = 13;
  out.context = 14;
  return out;
}

CapabilityRefreshSnapshot CurrentRefresh(std::uint64_t generation = 7) {
  CapabilityRefreshSnapshot out;
  out.status = CapabilityRefreshStatus::succeeded;
  out.code = Code::capability_snapshot_current;
  out.requested_generation = generation;
  out.completed_generation = generation;
  return out;
}

ProductCapabilitySnapshot CurrentCapabilities() {
  return studiocast::video::MakeFailClosedProductCapabilitySnapshot(
      Generations(), CurrentRefresh());
}

ProductCapabilityLane *Lane(ProductCapabilitySnapshot *snapshot,
                            SemanticEffectId effect,
                            ProductCapabilityLaneId lane) {
  return studiocast::video::FindProductCapabilityLane(snapshot, effect, lane);
}

void Promote(ProductCapabilitySnapshot *snapshot, SemanticEffectId effect,
             ProductCapabilityLaneId lane, BackendId backend,
             SemanticEffectProviderId provider, Code code,
             StatusId status = StatusId::ready) {
  ProductCapabilityLane *record = Lane(snapshot, effect, lane);
  if (!record)
    std::abort();
  record->disposition =
      status == StatusId::degraded
          ? ProductCapabilityDisposition::usable_with_degraded_behavior
          : ProductCapabilityDisposition::production_usable;
  record->code = code;
  record->nested_code = Code::none;
  record->semantic.semantic = effect;
  record->semantic.provider = provider;
  record->semantic.status = status;
  record->semantic.reason = ReasonId::none;
  record->semantic.live_stage = true;
  record->semantic.production_eligible = true;
  record->semantic.exact_evidence = true;
  if (lane == ProductCapabilityLaneId::open_vulkan) {
    constexpr std::array common{
        studiocast::hal::CapabilityEvidenceCode::vulkan_build_enabled,
        studiocast::hal::CapabilityEvidenceCode::vulkan_loader_available,
        studiocast::hal::CapabilityEvidenceCode::
            vulkan_physical_device_available,
        studiocast::hal::CapabilityEvidenceCode::vulkan_non_cpu_device,
        studiocast::hal::CapabilityEvidenceCode::vulkan_compute_queue_available,
        studiocast::hal::CapabilityEvidenceCode::vulkan_context_healthy,
        studiocast::hal::CapabilityEvidenceCode::vulkan_utility_kernels_ready,
        studiocast::hal::CapabilityEvidenceCode::vulkan_exact_device_shared,
        studiocast::hal::CapabilityEvidenceCode::vulkan_device_local_residency,
        studiocast::hal::CapabilityEvidenceCode::vulkan_bounded_allocations,
        studiocast::hal::CapabilityEvidenceCode::vulkan_no_cpu_layers,
        studiocast::hal::CapabilityEvidenceCode::
            vulkan_no_intermediate_readbacks,
    };
    for (const auto evidence_code : common) {
      record->semantic.evidence[record->semantic.evidence_count++] = {
          evidence_code, StatusId::ready,   ReasonId::none,
          effect,        BackendId::vulkan, true};
    }
    const auto evidence_code =
        effect == SemanticEffectId::mirror
            ? studiocast::hal::CapabilityEvidenceCode::
                  open_vulkan_mirror_production_ready
            : studiocast::hal::CapabilityEvidenceCode::
                  open_vulkan_vignette_fixed_center_production_ready;
    record->semantic.evidence[record->semantic.evidence_count++] = {
        evidence_code, StatusId::ready,   ReasonId::none,
        effect,        BackendId::vulkan, true};
  }
  (void)backend;
}

studiocast::hal::ProgramInvalidationKey
Invalidation(const ProductCapabilityGenerations &generations) {
  studiocast::hal::ProgramInvalidationKey key;
  key.generations.configuration = generations.configuration;
  key.generations.geometry = 21;
  key.generations.model = 22;
  key.generations.provider = generations.provider;
  key.generations.device = generations.device;
  key.generations.context = generations.context;
  key.canonical_plan_hash = 31;
  key.model_artifact_hash = 32;
  key.execution_domain_hash = 33;
  return key;
}

studiocast::hal::ProgramGenerations
ProgramGenerations(const ProductCapabilityGenerations &generations) {
  studiocast::hal::ProgramGenerations out;
  out.configuration = generations.configuration;
  out.geometry = 21;
  out.model = 22;
  out.provider = generations.provider;
  out.device = generations.device;
  out.context = generations.context;
  return out;
}

studiocast::hal::ExecutionDomainId Domain(BackendId backend,
                                          std::uint64_t adapter_id) {
  studiocast::hal::ExecutionDomainId domain;
  domain.backend = backend;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = 0x1c30;
  domain.physical_device.stable_id[0] = static_cast<std::uint8_t>(adapter_id);
  domain.logical_context_generation = 14;
  domain.queue_or_stream_identity = 1000 + adapter_id;
  domain.ownership_model = studiocast::hal::OwnershipModel::adapter_exclusive;
  domain.adapter_instance_id = adapter_id;
  if (backend == BackendId::vulkan) {
    domain.api_family = studiocast::hal::ApiFamily::vulkan;
    domain.allocator_family = studiocast::hal::AllocatorFamily::vulkan;
    domain.memory_handle_family =
        studiocast::hal::MemoryHandleFamily::vulkan_buffer;
    domain.synchronization_protocol =
        studiocast::hal::SynchronizationProtocol::vulkan_queue_fence;
  } else if (backend == BackendId::maxine) {
    domain.api_family = studiocast::hal::ApiFamily::maxine_nvcv;
    domain.allocator_family = studiocast::hal::AllocatorFamily::maxine_nvcv;
    domain.memory_handle_family =
        studiocast::hal::MemoryHandleFamily::nvcv_image;
    domain.synchronization_protocol =
        studiocast::hal::SynchronizationProtocol::maxine_bound_cuda_stream;
  } else {
    domain.api_family = studiocast::hal::ApiFamily::cuda_driver;
    domain.allocator_family = studiocast::hal::AllocatorFamily::cuda_driver;
    domain.memory_handle_family =
        studiocast::hal::MemoryHandleFamily::cuda_device_pointer;
    domain.synchronization_protocol =
        studiocast::hal::SynchronizationProtocol::cuda_stream_ordered;
  }
  return domain;
}

MultiIslandVideoRuntimeStatusBuildInput MultiIslandPassthroughInput() {
  MultiIslandVideoRuntimeStatusBuildInput input;
  input.status_generation = 100;
  input.generations = Generations();
  input.program_generations = ProgramGenerations(input.generations);
  input.lifecycle = VideoRuntimeLifecycle::running;
  input.capabilities = CurrentCapabilities();
  input.effect_count = studiocast::video::kCanonicalVideoEffectCount;
  for (std::size_t i = 0; i < input.effect_count; ++i) {
    input.effects[i].effect = studiocast::video::kCanonicalVideoEffects[i];
    input.effects[i].status = StatusId::passthrough;
  }
  return input;
}

void ConfigureIsland(MultiIslandVideoRuntimeStatusBuildInput *input,
                     PreparedExecutionRouteId route, std::uint64_t island_id,
                     std::uint64_t adapter_id, std::uint32_t stage_id,
                     SemanticEffectId effect, BackendId backend,
                     SemanticEffectProviderId provider,
                     ProductCapabilityLaneId lane, Code capability_code,
                     std::uint64_t uploads, std::uint64_t readbacks,
                     std::uint64_t cpu_tails) {
  if (!input || input->island_count >= input->islands.size())
    std::abort();
  Promote(&input->capabilities, effect, lane, backend, provider,
          capability_code);
  const ProductCapabilityLane *capability =
      studiocast::video::FindProductCapabilityLane(input->capabilities, effect,
                                                   lane);
  if (!capability)
    std::abort();

  PreparedExecutionIslandStatus &island = input->islands[input->island_count++];
  island.route = route;
  island.island_id = island_id;
  island.adapter_instance_id = adapter_id;
  island.snapshot_generation = input->status_generation;
  island.capability_generation = input->generations.capability;
  island.backend = backend;
  island.execution_domain = Domain(backend, adapter_id);
  island.valid = true;
  island.current = true;
  island.resolution.status = StatusId::ready;
  island.resolution.available = true;
  island.resolution.selected_backend = backend;
  island.resolution.selected_provider = provider;
  island.resolution.stage_count = 1;
  auto &resolved = island.resolution.stages[0];
  resolved.intent.stage_id = stage_id;
  resolved.intent.semantic = effect;
  resolved.intent.failure_action =
      studiocast::hal::FailureAction::fail_open_preserve_input;
  resolved.status = StatusId::ready;
  resolved.selected_backend = backend;
  resolved.provider = provider;
  resolved.adapter_instance_id = adapter_id;
  resolved.execution_domain = island.execution_domain;

  island.readiness.configured = true;
  island.readiness.program_id = 1000 + island_id;
  island.readiness.invalidation_key = Invalidation(input->generations);
  island.readiness.value.status = StatusId::ready;
  island.readiness.value.available = true;
  island.readiness.value.selected_backend = backend;
  island.readiness.value.provider = provider;
  island.readiness.value.execution_domain = island.execution_domain;
  island.telemetry.runtime_status = StatusId::executed;
  island.telemetry.transfers.uploads = uploads;
  island.telemetry.transfers.final_readbacks = readbacks;
  island.telemetry.transfers.cpu_tail_boundaries = cpu_tails;
  island.telemetry.transfers.synchronizations = 1;

  island.latest_committed_frame.valid = true;
  island.latest_committed_frame.program_id = island.readiness.program_id;
  island.latest_committed_frame.invalidation_key =
      island.readiness.invalidation_key;
  island.latest_committed_frame.execution_domain = island.execution_domain;
  island.latest_committed_frame.capture_sequence =
      input->writer_commit.capture_sequence;
  island.latest_committed_frame.output_commit_sequence =
      input->writer_commit.output_commit_sequence;
  island.latest_committed_frame.stage_count = 1;
  island.latest_committed_frame.stages[0] = {
      stage_id,       effect,   studiocast::hal::PrimitiveOperationId::invalid,
      backend,        provider, StageExecutionStatus::succeeded,
      ReasonId::none,
  };

  island.stage_count = 1;
  island.stages[0].stage_id = stage_id;
  island.stages[0].effect = effect;
  island.stages[0].backend = backend;
  island.stages[0].provider = provider;
  island.stages[0].status = StatusId::ready;
  island.stages[0].failure_action =
      studiocast::hal::FailureAction::fail_open_preserve_input;
  island.stages[0].committed_status = StageExecutionStatus::succeeded;
  island.stages[0].resolved = true;
  island.stages[0].committed = true;

  StableEffectRuntimeState &state =
      input->effects[static_cast<std::size_t>(effect) - 1];
  state.capability_lane = lane;
  state.capability_code = capability->code;
  state.nested_capability_code = capability->nested_code;
  state.capability = capability->semantic;
  state.selected_capability_lane = lane;
  state.selected_capability_code = capability->code;
  state.selected_nested_capability_code = capability->nested_code;
  state.selected_capability = capability->semantic;
  state.requested_backend = backend;
  state.requested_provider = provider;
  state.selected_backend = backend;
  state.selected_provider = provider;
  state.active_backend = backend;
  state.active_provider = provider;
  state.status = StatusId::executed;
  state.commit = input->writer_commit;
  state.selected_island_id = island_id;
  state.committed_stage_id = stage_id;
  state.committed_stage_status = StageExecutionStatus::succeeded;
  state.requested = true;
  state.available = true;
  state.selected = true;
  state.active = true;
}

MultiIslandVideoRuntimeStatusBuildInput ThreeIslandInput() {
  MultiIslandVideoRuntimeStatusBuildInput input = MultiIslandPassthroughInput();
  input.writer_commit.valid = true;
  input.writer_commit.capture_sequence = 501;
  input.writer_commit.output_commit_sequence = 9;
  ConfigureIsland(&input, PreparedExecutionRouteId::final_visual, 1, 101, 11,
                  SemanticEffectId::mirror, BackendId::vulkan,
                  SemanticEffectProviderId::open_vulkan_production_kernels,
                  ProductCapabilityLaneId::open_vulkan,
                  Code::open_vulkan_mirror_production_ready, 1, 1, 0);
  ConfigureIsland(&input, PreparedExecutionRouteId::maxine_resident, 2, 102, 12,
                  SemanticEffectId::eye_contact, BackendId::maxine,
                  SemanticEffectProviderId::maxine_ar,
                  ProductCapabilityLaneId::maxine, Code::maxine_effect_usable,
                  1, 1, 1);
  ConfigureIsland(
      &input, PreparedExecutionRouteId::open_cuda_virtual_background, 3, 103,
      13, SemanticEffectId::virtual_background_blur, BackendId::cuda,
      SemanticEffectProviderId::open_video_onnxruntime_cuda,
      ProductCapabilityLaneId::open_cuda, Code::open_cuda_effect_available, 1,
      1, 1);
  return input;
}

VideoRuntimeStatusSnapshot MixedCommittedSnapshot() {
  VideoRuntimeStatusSnapshot snapshot;
  snapshot.status_generation = 99;
  snapshot.generations = Generations();
  snapshot.lifecycle = VideoRuntimeLifecycle::running;
  snapshot.capabilities = CurrentCapabilities();
  Promote(&snapshot.capabilities, SemanticEffectId::mirror,
          ProductCapabilityLaneId::open_vulkan, BackendId::vulkan,
          SemanticEffectProviderId::open_vulkan_production_kernels,
          Code::open_vulkan_mirror_production_ready);
  Promote(&snapshot.capabilities, SemanticEffectId::eye_contact,
          ProductCapabilityLaneId::maxine, BackendId::maxine,
          SemanticEffectProviderId::maxine_ar, Code::maxine_effect_usable);

  snapshot.resolution.status = StatusId::ready;
  snapshot.resolution.available = true;
  snapshot.resolution.stage_count = 2;
  snapshot.resolution.stages[0].intent.stage_id = 1;
  snapshot.resolution.stages[0].intent.semantic = SemanticEffectId::mirror;
  snapshot.resolution.stages[0].status = StatusId::ready;
  snapshot.resolution.stages[0].selected_backend = BackendId::vulkan;
  snapshot.resolution.stages[0].provider =
      SemanticEffectProviderId::open_vulkan_production_kernels;
  snapshot.resolution.stages[1].intent.stage_id = 2;
  snapshot.resolution.stages[1].intent.semantic = SemanticEffectId::eye_contact;
  snapshot.resolution.stages[1].status = StatusId::ready;
  snapshot.resolution.stages[1].selected_backend = BackendId::maxine;
  snapshot.resolution.stages[1].provider = SemanticEffectProviderId::maxine_ar;

  snapshot.readiness.configured = true;
  snapshot.readiness.program_id = 44;
  snapshot.readiness.invalidation_key = Invalidation(snapshot.generations);
  snapshot.readiness.value.status = StatusId::ready;
  snapshot.readiness.value.available = true;

  snapshot.latest_committed_frame.valid = true;
  snapshot.latest_committed_frame.program_id = 44;
  snapshot.latest_committed_frame.invalidation_key =
      snapshot.readiness.invalidation_key;
  snapshot.latest_committed_frame.execution_domain =
      snapshot.readiness.value.execution_domain;
  snapshot.latest_committed_frame.capture_sequence = 101;
  snapshot.latest_committed_frame.output_commit_sequence = 8;
  snapshot.latest_committed_frame.stage_count = 2;
  snapshot.latest_committed_frame.stages[0] = {
      1,
      SemanticEffectId::mirror,
      studiocast::hal::PrimitiveOperationId::invalid,
      BackendId::vulkan,
      SemanticEffectProviderId::open_vulkan_production_kernels,
      StageExecutionStatus::succeeded,
      ReasonId::none,
  };
  snapshot.latest_committed_frame.stages[1] = {
      2,
      SemanticEffectId::eye_contact,
      studiocast::hal::PrimitiveOperationId::invalid,
      BackendId::maxine,
      SemanticEffectProviderId::maxine_ar,
      StageExecutionStatus::succeeded,
      ReasonId::none,
  };
  return snapshot;
}

bool TestStableCodes() {
  using studiocast::video::effects::EffectCapabilityCodeToken;
  bool ok = true;
  ok &= Expect(studiocast::video::effects::kEffectCapabilityCodeVersion == 1,
               "effect capability code schema changed");
  ok &= Expect(static_cast<std::uint32_t>(
                   Code::open_vulkan_mirror_production_ready) == 100,
               "mirror readiness code changed");
  ok &= Expect(static_cast<std::uint32_t>(Code::capability_snapshot_current) ==
                   200,
               "refresh code range changed");
  ok &= Expect(
      studiocast::video::kMultiIslandVideoRuntimeStatusSchemaVersion == 1 &&
          static_cast<std::uint32_t>(PreparedExecutionRouteId::final_visual) ==
              1 &&
          static_cast<std::uint32_t>(
              PreparedExecutionRouteId::maxine_resident) == 2 &&
          static_cast<std::uint32_t>(
              PreparedExecutionRouteId::open_cuda_virtual_background) == 3 &&
          static_cast<std::uint16_t>(
              MultiIslandStatusValidationCode::mixed_writer_commit) == 18 &&
          static_cast<std::uint16_t>(
              MultiIslandStatusValidationCode::aggregate_mismatch) == 31,
      "multi-island schema and stable numeric IDs changed");
  ok &= Expect(EffectCapabilityCodeToken(
                   Code::cpu_virtual_background_replace_unavailable) ==
                   "effect.virtual_background.replace.cpu."
                   "no_selectable_production_path",
               "CPU product blocker token changed");
  ok &= Expect(EffectCapabilityCodeToken(
                   Code::open_vulkan_vignette_fixed_center_production_ready) ==
                   "open_vulkan_vignette_fixed_center_production_ready",
               "Vulkan vignette readiness token changed");
  ok &=
      Expect(EffectCapabilityCodeToken(static_cast<Code>(0xffffffffu)).empty(),
             "unknown capability code must not invent a token");
  return ok;
}

bool TestCompleteFailClosedMatrix() {
  ProductCapabilitySnapshot snapshot = CurrentCapabilities();
  const auto validation =
      studiocast::video::ValidateProductCapabilitySnapshot(snapshot);
  bool ok = Expect(validation.code == CapabilityValidationCode::valid,
                   "canonical fail-closed matrix should validate");
  ok &= Expect(snapshot.lane_count ==
                   studiocast::video::kProductCapabilityRecordCount,
               "matrix must contain exactly 9 effects x 4 lanes");

  for (const SemanticEffectId effect :
       studiocast::video::kCanonicalVideoEffects) {
    const ProductCapabilityLane *cpu =
        studiocast::video::FindProductCapabilityLane(
            snapshot, effect, ProductCapabilityLaneId::cpu);
    ok &= Expect(
        cpu && cpu->disposition == ProductCapabilityDisposition::unavailable &&
            !studiocast::video::IsProductCapabilityLaneEligible(snapshot, *cpu),
        "each CPU lane must be explicitly unavailable");
  }

  const ProductCapabilityLane *mirror =
      studiocast::video::FindProductCapabilityLane(
          snapshot, SemanticEffectId::mirror,
          ProductCapabilityLaneId::open_vulkan);
  const ProductCapabilityLane *vignette =
      studiocast::video::FindProductCapabilityLane(
          snapshot, SemanticEffectId::vignette,
          ProductCapabilityLaneId::open_vulkan);
  const ProductCapabilityLane *matting =
      studiocast::video::FindProductCapabilityLane(
          snapshot, SemanticEffectId::virtual_background_blur,
          ProductCapabilityLaneId::open_vulkan);
  const ProductCapabilityLane *eye =
      studiocast::video::FindProductCapabilityLane(
          snapshot, SemanticEffectId::eye_contact,
          ProductCapabilityLaneId::open_vulkan);
  ok &= Expect(
      mirror && vignette &&
          mirror->disposition == ProductCapabilityDisposition::unavailable &&
          vignette->disposition == ProductCapabilityDisposition::unavailable,
      "production Vulkan pixel effects must still fail closed "
      "without exact evidence");
  ok &= Expect(matting &&
                   matting->disposition ==
                       ProductCapabilityDisposition::experimental &&
                   matting->code == Code::open_vulkan_matting_unavailable,
               "Vulkan matting consumers must remain experimental");
  ok &= Expect(eye &&
                   eye->disposition ==
                       ProductCapabilityDisposition::diagnostics_only &&
                   eye->code == Code::open_vulkan_eye_contact_unavailable,
               "Vulkan eye contact must remain diagnostics-only");
  return ok;
}

bool TestStructuralCorruption() {
  bool ok = true;
  ProductCapabilitySnapshot snapshot = CurrentCapabilities();
  snapshot.lane_count--;
  ok &= Expect(
      studiocast::video::ValidateProductCapabilitySnapshot(snapshot).code ==
          CapabilityValidationCode::missing_lane,
      "missing lane must fail structural validation");

  snapshot = CurrentCapabilities();
  snapshot.lanes.back() = snapshot.lanes.front();
  ok &= Expect(
      studiocast::video::ValidateProductCapabilitySnapshot(snapshot).code ==
          CapabilityValidationCode::duplicate_lane,
      "duplicate lane must fail structural validation");

  snapshot = CurrentCapabilities();
  snapshot.lanes[0].lane = ProductCapabilityLaneId::invalid;
  ok &= Expect(
      studiocast::video::ValidateProductCapabilitySnapshot(snapshot).code ==
          CapabilityValidationCode::unknown_lane,
      "unknown lane must fail structural validation");

  snapshot = CurrentCapabilities();
  snapshot.lanes[0].effect = SemanticEffectId::invalid;
  ok &= Expect(
      studiocast::video::ValidateProductCapabilitySnapshot(snapshot).code ==
          CapabilityValidationCode::unknown_effect,
      "unknown effect must fail structural validation");

  snapshot = CurrentCapabilities();
  ProductCapabilityLane *cuda = Lane(&snapshot, SemanticEffectId::mirror,
                                     ProductCapabilityLaneId::open_cuda);
  cuda->disposition = ProductCapabilityDisposition::production_usable;
  ok &= Expect(
      studiocast::video::ValidateProductCapabilitySnapshot(snapshot).code ==
          CapabilityValidationCode::readiness_inconsistent,
      "success-looking disposition cannot replace exact HAL evidence");

  snapshot = CurrentCapabilities();
  ProductCapabilityLane *eye = Lane(&snapshot, SemanticEffectId::eye_contact,
                                    ProductCapabilityLaneId::open_vulkan);
  eye->disposition = ProductCapabilityDisposition::production_usable;
  eye->code = Code::open_vulkan_mirror_production_ready;
  eye->semantic.provider =
      SemanticEffectProviderId::open_vulkan_production_kernels;
  eye->semantic.status = StatusId::ready;
  eye->semantic.live_stage = true;
  eye->semantic.production_eligible = true;
  eye->semantic.exact_evidence = true;
  ok &= Expect(
      studiocast::video::ValidateProductCapabilitySnapshot(snapshot).code ==
          CapabilityValidationCode::vulkan_capability_promoted,
      "non-production Vulkan effect must not be promoted");

  snapshot = CurrentCapabilities();
  Promote(&snapshot, SemanticEffectId::mirror,
          ProductCapabilityLaneId::open_vulkan, BackendId::vulkan,
          SemanticEffectProviderId::open_vulkan_production_kernels,
          Code::open_vulkan_mirror_production_ready);
  ProductCapabilityLane *mirror = Lane(&snapshot, SemanticEffectId::mirror,
                                       ProductCapabilityLaneId::open_vulkan);
  mirror->semantic.evidence_count--;
  ok &= Expect(
      studiocast::video::ValidateProductCapabilitySnapshot(snapshot).code ==
          CapabilityValidationCode::vulkan_matrix_inconsistent,
      "Vulkan production success must include all common and per-effect "
      "typed evidence");
  return ok;
}

bool TestRefreshState() {
  ProductCapabilitySnapshot snapshot = CurrentCapabilities();
  Promote(&snapshot, SemanticEffectId::mirror,
          ProductCapabilityLaneId::open_vulkan, BackendId::vulkan,
          SemanticEffectProviderId::open_vulkan_production_kernels,
          Code::open_vulkan_mirror_production_ready);
  const ProductCapabilityLane *mirror =
      studiocast::video::FindProductCapabilityLane(
          snapshot, SemanticEffectId::mirror,
          ProductCapabilityLaneId::open_vulkan);
  bool ok = Expect(mirror && studiocast::video::IsProductCapabilityLaneEligible(
                                 snapshot, *mirror),
                   "current exact readiness should be eligible");

  ProductCapabilitySnapshot stale = snapshot;
  stale.refresh.status = CapabilityRefreshStatus::stale;
  stale.refresh.code = Code::capability_snapshot_stale;
  stale.refresh.requested_generation = 6;
  stale.refresh.completed_generation = 6;
  ok &=
      Expect(studiocast::video::ValidateProductCapabilitySnapshot(stale).code ==
                 CapabilityValidationCode::valid,
             "a structurally valid stale snapshot should remain reportable");
  ok &= Expect(!studiocast::video::IsProductCapabilityLaneEligible(
                   stale, *Lane(&stale, SemanticEffectId::mirror,
                                ProductCapabilityLaneId::open_vulkan)),
               "stale exact evidence must fail closed");

  ProductCapabilitySnapshot failed = snapshot;
  failed.refresh.status = CapabilityRefreshStatus::failed;
  failed.refresh.code = Code::capability_refresh_failed;
  failed.refresh.completed_generation = 6;
  ok &= Expect(
      studiocast::video::ValidateProductCapabilitySnapshot(failed).code ==
          CapabilityValidationCode::valid,
      "failed refresh state should be structurally reportable");
  ok &= Expect(!studiocast::video::IsProductCapabilityLaneEligible(
                   failed, *Lane(&failed, SemanticEffectId::mirror,
                                 ProductCapabilityLaneId::open_vulkan)),
               "failed refresh must revoke eligibility");

  ProductCapabilitySnapshot malformed = snapshot;
  malformed.refresh.completed_generation = 6;
  ok &= Expect(
      studiocast::video::ValidateProductCapabilitySnapshot(malformed).code ==
          CapabilityValidationCode::refresh_inconsistent,
      "success with a stale completed generation must fail validation");
  return ok;
}

bool TestActiveAttribution() {
  VideoRuntimeStatusSnapshot snapshot = MixedCommittedSnapshot();
  bool ok = Expect(
      studiocast::video::ValidateVideoRuntimeStatusSnapshot(snapshot).code ==
          studiocast::video::RuntimeStatusValidationCode::valid,
      "mixed committed runtime fixture should validate");
  const auto active =
      studiocast::video::DeriveActiveRuntimeAttribution(snapshot);
  ok &= Expect(active.effect_count == 2 && active.mixed_backends &&
                   active.active_backend == BackendId::invalid,
               "mixed successful committed stages require mixed attribution");
  const auto legacy =
      studiocast::video::BuildLegacyVideoRuntimeCompatibility(snapshot);
  ok &= Expect(legacy.selected_backend ==
                       studiocast::video::LegacyComputeBackendId::mixed &&
                   legacy.active_backend ==
                       studiocast::video::LegacyComputeBackendId::mixed &&
                   legacy.selected_effect_count == 2 &&
                   legacy.active_effect_count == 2 &&
                   legacy.active_effects[0].backend == "open_vulkan" &&
                   legacy.active_effects[1].backend == "maxine",
               "legacy projection should preserve mixed committed effects");

  VideoRuntimeStatusSnapshot startup = snapshot;
  startup.lifecycle = VideoRuntimeLifecycle::starting;
  startup.telemetry.active_backend = BackendId::vulkan;
  startup.telemetry.active_backend_frames = 999;
  ok &= Expect(
      studiocast::video::DeriveActiveRuntimeAttribution(startup).effect_count ==
          0,
      "startup and cumulative telemetry must not imply activity");

  VideoRuntimeStatusSnapshot absent_commit = snapshot;
  absent_commit.latest_committed_frame = {};
  absent_commit.telemetry.committed_output_frames = 999;
  absent_commit.telemetry.committed_stage_successes[0] = 999;
  ok &= Expect(studiocast::video::DeriveActiveRuntimeAttribution(absent_commit)
                       .effect_count == 0,
               "telemetry cannot replace a committed frame");

  VideoRuntimeStatusSnapshot failed_commit = snapshot;
  failed_commit.latest_committed_frame.stages[0].status =
      StageExecutionStatus::optional_failed_open;
  failed_commit.latest_committed_frame.stages[1].status =
      StageExecutionStatus::required_failed;
  ok &= Expect(studiocast::video::DeriveActiveRuntimeAttribution(failed_commit)
                       .effect_count == 0,
               "failed committed stages must not be attributed active");

  VideoRuntimeStatusSnapshot wrong_generation = snapshot;
  wrong_generation.latest_committed_frame.invalidation_key.generations
      .configuration++;
  wrong_generation.telemetry.active_backend_frames = 999;
  ok &=
      Expect(studiocast::video::DeriveActiveRuntimeAttribution(wrong_generation)
                     .effect_count == 0,
             "wrong-generation commit must not be active");

  VideoRuntimeStatusSnapshot failed_refresh = snapshot;
  failed_refresh.capabilities.refresh.status = CapabilityRefreshStatus::failed;
  failed_refresh.capabilities.refresh.code = Code::capability_refresh_failed;
  failed_refresh.capabilities.refresh.completed_generation = 6;
  ok &= Expect(studiocast::video::DeriveActiveRuntimeAttribution(failed_refresh)
                       .effect_count == 0,
               "failed capability refresh must fail active attribution closed");
  return ok;
}

bool TestReadinessConsistency() {
  VideoRuntimeStatusSnapshot snapshot = MixedCommittedSnapshot();
  snapshot.readiness.value.available = false;
  bool ok = Expect(
      studiocast::video::ValidateVideoRuntimeStatusSnapshot(snapshot).code ==
          studiocast::video::RuntimeStatusValidationCode::
              readiness_inconsistent,
      "configured runtime readiness cannot be unavailable");

  snapshot = MixedCommittedSnapshot();
  snapshot.capabilities.generations.context++;
  ok &= Expect(
      studiocast::video::ValidateVideoRuntimeStatusSnapshot(snapshot).code ==
          studiocast::video::RuntimeStatusValidationCode::
              capability_generation_mismatch,
      "capability and runtime generations must agree");
  return ok;
}

bool TestMultiIslandPassthroughAndProductionFixtures() {
  MultiIslandVideoRuntimeStatusSnapshot passthrough;
  const auto passthrough_result =
      studiocast::video::BuildMultiIslandVideoRuntimeStatusSnapshot(
          MultiIslandPassthroughInput(), &passthrough);
  bool ok = Expect(
      passthrough_result.code == MultiIslandStatusValidationCode::valid &&
          passthrough.island_count == 0 &&
          passthrough.effect_count ==
              studiocast::video::kCanonicalVideoEffectCount &&
          passthrough.aggregate.current_island_count == 0 &&
          passthrough.aggregate.transfers.uploads == 0,
      "zero-island CPU passthrough must remain explicit and transfer-free");

  MultiIslandVideoRuntimeStatusSnapshot snapshot;
  const auto result =
      studiocast::video::BuildMultiIslandVideoRuntimeStatusSnapshot(
          ThreeIslandInput(), &snapshot);
  ok &= Expect(
      result.code == MultiIslandStatusValidationCode::valid &&
          snapshot.island_count == 3 &&
          snapshot.aggregate.current_island_count == 3 &&
          snapshot.aggregate.transfers.uploads == 3 &&
          snapshot.aggregate.transfers.final_readbacks == 3 &&
          snapshot.aggregate.transfers.synchronizations == 3 &&
          snapshot.aggregate.transfers.cpu_tail_boundaries == 2,
      "Vulkan, Maxine, and Open CUDA islands must aggregate exactly once");
  ok &= Expect(
      snapshot.effects[static_cast<std::size_t>(SemanticEffectId::mirror) - 1]
              .active &&
          snapshot
              .effects[static_cast<std::size_t>(SemanticEffectId::eye_contact) -
                       1]
              .active &&
          snapshot
              .effects[static_cast<std::size_t>(
                           SemanticEffectId::virtual_background_blur) -
                       1]
              .active,
      "only successfully committed current effects should be active");
  ok &= Expect(
      studiocast::video::IsWriterOutputCommitMonotonic(
          {true, 500, 8}, snapshot.writer_commit) &&
          studiocast::video::IsWriterOutputCommitMonotonic(
              snapshot.writer_commit, snapshot.writer_commit) &&
          !studiocast::video::IsWriterOutputCommitMonotonic(
              snapshot.writer_commit, {true, 499, 8}) &&
          !studiocast::video::IsWriterOutputCommitMonotonic(
              snapshot.writer_commit, {true, 502, 7}) &&
          !studiocast::video::IsWriterOutputCommitMonotonic(
              snapshot.writer_commit, {}),
      "writer commit identity must remain nonzero, repeat exactly, or advance "
      "monotonically");

  MultiIslandVideoRuntimeStatusSnapshot vulkan_only;
  MultiIslandVideoRuntimeStatusBuildInput vulkan_input =
      MultiIslandPassthroughInput();
  vulkan_input.writer_commit = {true, 601, 10};
  ConfigureIsland(&vulkan_input, PreparedExecutionRouteId::final_visual, 7, 107,
                  17, SemanticEffectId::vignette, BackendId::vulkan,
                  SemanticEffectProviderId::open_vulkan_production_kernels,
                  ProductCapabilityLaneId::open_vulkan,
                  Code::open_vulkan_vignette_fixed_center_production_ready, 1,
                  1, 0);
  ok &= Expect(
      studiocast::video::BuildMultiIslandVideoRuntimeStatusSnapshot(
          vulkan_input, &vulkan_only)
                  .code == MultiIslandStatusValidationCode::valid &&
          vulkan_only.island_count == 1 &&
          vulkan_only
              .effects[static_cast<std::size_t>(SemanticEffectId::vignette) - 1]
              .active,
      "singular Vulkan compatibility route must project into one island");
  return ok;
}

bool TestMultiIslandStructuralRejection() {
  MultiIslandVideoRuntimeStatusSnapshot snapshot;
  bool ok =
      Expect(studiocast::video::BuildMultiIslandVideoRuntimeStatusSnapshot(
                 ThreeIslandInput(), &snapshot)
                     .code == MultiIslandStatusValidationCode::valid,
             "multi-island corruption fixture must start valid");

  MultiIslandVideoRuntimeStatusSnapshot malformed = snapshot;
  malformed.schema_version++;
  ok &= Expect(studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(
                   malformed)
                       .code ==
                   MultiIslandStatusValidationCode::wrong_schema_version,
               "wrong multi-island schema must fail closed");

  malformed = snapshot;
  malformed.islands[1].island_id = malformed.islands[0].island_id;
  ok &=
      Expect(studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(
                 malformed)
                     .code == MultiIslandStatusValidationCode::duplicate_island,
             "duplicate island IDs must fail closed");

  malformed = snapshot;
  malformed.effects.back() = malformed.effects.front();
  ok &=
      Expect(studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(
                 malformed)
                     .code == MultiIslandStatusValidationCode::duplicate_effect,
             "duplicate effect IDs must fail closed");

  malformed = snapshot;
  malformed.effect_count--;
  ok &= Expect(studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(
                   malformed)
                       .code == MultiIslandStatusValidationCode::missing_effect,
               "missing stable effect records must fail closed");

  malformed = snapshot;
  malformed.islands[0].current = false;
  ok &= Expect(studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(
                   malformed)
                       .code == MultiIslandStatusValidationCode::stale_island,
               "stale islands must fail closed");

  malformed = snapshot;
  malformed.islands[0].adapter_instance_id++;
  ok &=
      Expect(studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(
                 malformed)
                     .code == MultiIslandStatusValidationCode::foreign_adapter,
             "foreign adapter identity must fail closed");

  malformed = snapshot;
  malformed.islands[0].readiness.value.execution_domain =
      malformed.islands[1].execution_domain;
  ok &= Expect(studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(
                   malformed)
                       .code ==
                   MultiIslandStatusValidationCode::wrong_execution_domain,
               "wrong execution domain must fail closed");

  malformed = snapshot;
  malformed.islands[0].readiness.invalidation_key.generations.model++;
  ok &=
      Expect(studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(
                 malformed)
                     .code == MultiIslandStatusValidationCode::wrong_generation,
             "wrong full program generation must fail closed");

  malformed = snapshot;
  malformed.islands[0].latest_committed_frame.program_id++;
  ok &= Expect(studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(
                   malformed)
                       .code == MultiIslandStatusValidationCode::wrong_program,
               "foreign committed program identity must fail closed");

  malformed = snapshot;
  malformed.islands[1].latest_committed_frame.output_commit_sequence++;
  ok &= Expect(studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(
                   malformed)
                       .code ==
                   MultiIslandStatusValidationCode::mixed_writer_commit,
               "mixed writer commits must fail closed");

  malformed = snapshot;
  malformed.effects[static_cast<std::size_t>(SemanticEffectId::mirror) - 1]
      .capability.exact_evidence = false;
  ok &= Expect(studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(
                   malformed)
                       .code ==
                   MultiIslandStatusValidationCode::invalid_capability_evidence,
               "partial or foreign capability evidence must fail closed");

  malformed = snapshot;
  malformed.lifecycle = VideoRuntimeLifecycle::idle;
  ok &= Expect(studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(
                   malformed)
                       .code ==
                   MultiIslandStatusValidationCode::committed_activity_mismatch,
               "an idle runtime must not retain current active attribution");

  malformed = snapshot;
  malformed.aggregate.transfers.uploads++;
  ok &= Expect(studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(
                   malformed)
                       .code ==
                   MultiIslandStatusValidationCode::aggregate_mismatch,
               "aggregate transfers must equal the once-only island sum");

  MultiIslandVideoRuntimeStatusBuildInput overflow = ThreeIslandInput();
  overflow.islands[0].telemetry.transfers.uploads =
      std::numeric_limits<std::uint64_t>::max();
  MultiIslandVideoRuntimeStatusSnapshot failed_output = snapshot;
  ok &=
      Expect(studiocast::video::BuildMultiIslandVideoRuntimeStatusSnapshot(
                 overflow, &failed_output)
                         .code ==
                     MultiIslandStatusValidationCode::aggregate_overflow &&
                 failed_output.status_generation == 0 &&
                 failed_output.island_count == 0,
             "aggregate overflow must fail closed without publishing a partial "
             "snapshot");
  return ok;
}

bool TestMultiIslandStrictAndFailureSemantics() {
  MultiIslandVideoRuntimeStatusSnapshot snapshot;
  bool ok =
      Expect(studiocast::video::BuildMultiIslandVideoRuntimeStatusSnapshot(
                 ThreeIslandInput(), &snapshot)
                     .code == MultiIslandStatusValidationCode::valid,
             "strict/failure fixture must start valid");

  auto &cuda_state =
      snapshot.effects[static_cast<std::size_t>(
                           SemanticEffectId::virtual_background_blur) -
                       1];
  cuda_state.strict_backend_selection = true;
  cuda_state.requested_backend = BackendId::open_video_onnxruntime;
  ok &= Expect(
      studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(snapshot)
              .code ==
          MultiIslandStatusValidationCode::strict_backend_substitution,
      "strict CUDA-family selection must not substitute a different backend "
      "identity");

  ok &= Expect(studiocast::video::BuildMultiIslandVideoRuntimeStatusSnapshot(
                   ThreeIslandInput(), &snapshot)
                       .code == MultiIslandStatusValidationCode::valid,
               "provider substitution fixture must rebuild");
  auto &maxine_state =
      snapshot
          .effects[static_cast<std::size_t>(SemanticEffectId::eye_contact) - 1];
  maxine_state.strict_provider_selection = true;
  maxine_state.requested_provider = SemanticEffectProviderId::maxine_vfx;
  ok &= Expect(
      studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(snapshot)
              .code ==
          MultiIslandStatusValidationCode::strict_provider_substitution,
      "strict provider selection must not substitute another Maxine provider");

  ok &= Expect(studiocast::video::BuildMultiIslandVideoRuntimeStatusSnapshot(
                   ThreeIslandInput(), &snapshot)
                       .code == MultiIslandStatusValidationCode::valid,
               "optional failure fixture must rebuild");
  auto &open_cuda =
      snapshot.effects[static_cast<std::size_t>(
                           SemanticEffectId::virtual_background_blur) -
                       1];
  open_cuda.active = false;
  open_cuda.active_backend = BackendId::invalid;
  open_cuda.active_provider = SemanticEffectProviderId::invalid;
  open_cuda.degraded = true;
  open_cuda.fallback = true;
  open_cuda.status = StatusId::degraded;
  open_cuda.reason = ReasonId::optional_stage_failed_open;
  open_cuda.committed_stage_status = StageExecutionStatus::optional_failed_open;
  snapshot.islands[2].latest_committed_frame.stages[0].status =
      StageExecutionStatus::optional_failed_open;
  snapshot.islands[2].latest_committed_frame.stages[0].reason =
      ReasonId::optional_stage_failed_open;
  snapshot.islands[2].stages[0].committed_status =
      StageExecutionStatus::optional_failed_open;
  ok &= Expect(
      studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(snapshot)
              .code == MultiIslandStatusValidationCode::valid,
      "optional fail-open must preserve the writer commit while degrading the "
      "effect without active backend attribution");

  ok &= Expect(studiocast::video::BuildMultiIslandVideoRuntimeStatusSnapshot(
                   ThreeIslandInput(), &snapshot)
                       .code == MultiIslandStatusValidationCode::valid,
               "non-strict fallback fixture must rebuild");
  auto &fallback_state =
      snapshot.effects[static_cast<std::size_t>(
                           SemanticEffectId::virtual_background_blur) -
                       1];
  fallback_state.requested_backend = BackendId::open_video_onnxruntime;
  fallback_state.fallback = true;
  ok &= Expect(
      studiocast::video::ValidateMultiIslandVideoRuntimeStatusSnapshot(snapshot)
                  .code == MultiIslandStatusValidationCode::valid &&
          fallback_state.active,
      "non-strict backend substitution may remain actively attributed only "
      "after the exact selected lane commits the writer output");

  MultiIslandVideoRuntimeStatusBuildInput blocked =
      MultiIslandPassthroughInput();
  StableEffectRuntimeState &blocked_maxine =
      blocked
          .effects[static_cast<std::size_t>(SemanticEffectId::eye_contact) - 1];
  const ProductCapabilityLane *lane =
      studiocast::video::FindProductCapabilityLane(
          blocked.capabilities, SemanticEffectId::eye_contact,
          ProductCapabilityLaneId::maxine);
  if (!lane)
    std::abort();
  blocked_maxine.capability_lane = ProductCapabilityLaneId::maxine;
  blocked_maxine.capability_code = lane->code;
  blocked_maxine.nested_capability_code = lane->nested_code;
  blocked_maxine.capability = lane->semantic;
  blocked_maxine.requested_backend = BackendId::maxine;
  blocked_maxine.requested_provider = SemanticEffectProviderId::maxine_ar;
  blocked_maxine.status = StatusId::blocked;
  blocked_maxine.reason = ReasonId::capability_evidence_incomplete;
  blocked_maxine.strict_backend_selection = true;
  blocked_maxine.strict_provider_selection = true;
  blocked_maxine.requested = true;
  blocked_maxine.blocked = true;
  MultiIslandVideoRuntimeStatusSnapshot blocked_snapshot;
  ok &= Expect(
      studiocast::video::BuildMultiIslandVideoRuntimeStatusSnapshot(
          blocked, &blocked_snapshot)
                  .code == MultiIslandStatusValidationCode::valid &&
          !blocked_snapshot
               .effects
                   [static_cast<std::size_t>(SemanticEffectId::eye_contact) - 1]
               .available,
      "missing Maxine artifact/provider evidence must remain exactly blocked "
      "without broad availability promotion");
  return ok;
}

bool TestNoAllocation() {
  const std::uint64_t before = g_allocations.load(std::memory_order_relaxed);
  ProductCapabilitySnapshot capabilities =
      studiocast::video::MakeFailClosedProductCapabilitySnapshot(
          Generations(), CurrentRefresh());
  const auto capability_validation =
      studiocast::video::ValidateProductCapabilitySnapshot(capabilities);
  VideoRuntimeStatusSnapshot snapshot = MixedCommittedSnapshot();
  const auto runtime_validation =
      studiocast::video::ValidateVideoRuntimeStatusSnapshot(snapshot);
  const auto active =
      studiocast::video::DeriveActiveRuntimeAttribution(snapshot);
  const auto legacy =
      studiocast::video::BuildLegacyVideoRuntimeCompatibility(snapshot);
  MultiIslandVideoRuntimeStatusSnapshot multi;
  MultiIslandStatusValidationCode multi_code =
      MultiIslandStatusValidationCode::valid;
  for (int i = 0; i < 32; ++i) {
    multi_code = studiocast::video::BuildMultiIslandVideoRuntimeStatusSnapshot(
                     ThreeIslandInput(), &multi)
                     .code;
  }
  const std::uint64_t after = g_allocations.load(std::memory_order_relaxed);
  return Expect(capability_validation.code == CapabilityValidationCode::valid &&
                    runtime_validation.code ==
                        studiocast::video::RuntimeStatusValidationCode::valid &&
                    active.effect_count == 2 &&
                    legacy.selected_effect_count == 2 &&
                    legacy.active_effect_count == 2 &&
                    multi_code == MultiIslandStatusValidationCode::valid &&
                    multi.island_count == 3 && after == before,
                "snapshot construction/validation/projection must not "
                "allocate");
}

} // namespace

void *operator new(std::size_t size) {
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *memory = std::malloc(size))
    return memory;
  throw std::bad_alloc();
}

void *operator new[](std::size_t size) {
  g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *memory = std::malloc(size))
    return memory;
  throw std::bad_alloc();
}

void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}

int main() {
  bool ok = true;
  ok = TestStableCodes() && ok;
  ok = TestCompleteFailClosedMatrix() && ok;
  ok = TestStructuralCorruption() && ok;
  ok = TestRefreshState() && ok;
  ok = TestActiveAttribution() && ok;
  ok = TestReadinessConsistency() && ok;
  ok = TestMultiIslandPassthroughAndProductionFixtures() && ok;
  ok = TestMultiIslandStructuralRejection() && ok;
  ok = TestMultiIslandStrictAndFailureSemantics() && ok;
  ok = TestNoAllocation() && ok;
  if (ok)
    std::cout << "video runtime status tests passed\n";
  return ok ? 0 : 1;
}
