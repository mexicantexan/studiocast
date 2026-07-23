#include "core/video/video_runtime_status.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
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
using studiocast::video::ProductCapabilityDisposition;
using studiocast::video::ProductCapabilityGenerations;
using studiocast::video::ProductCapabilityLane;
using studiocast::video::ProductCapabilityLaneId;
using studiocast::video::ProductCapabilitySnapshot;
using studiocast::video::VideoRuntimeLifecycle;
using studiocast::video::VideoRuntimeStatusSnapshot;
using Code = studiocast::video::effects::EffectCapabilityCode;

static_assert(std::is_trivially_copyable_v<ProductCapabilitySnapshot>);
static_assert(std::is_trivially_copyable_v<VideoRuntimeStatusSnapshot>);
static_assert(std::is_trivially_destructible_v<ProductCapabilitySnapshot>);
static_assert(std::is_trivially_destructible_v<VideoRuntimeStatusSnapshot>);

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
  const std::uint64_t after = g_allocations.load(std::memory_order_relaxed);
  return Expect(capability_validation.code == CapabilityValidationCode::valid &&
                    runtime_validation.code ==
                        studiocast::video::RuntimeStatusValidationCode::valid &&
                    active.effect_count == 2 &&
                    legacy.selected_effect_count == 2 &&
                    legacy.active_effect_count == 2 && after == before,
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
  ok = TestNoAllocation() && ok;
  if (ok)
    std::cout << "video runtime status tests passed\n";
  return ok ? 0 : 1;
}
