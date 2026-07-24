#pragma once

#include "core/hal/resolver.h"
#include "core/video/effects/capability_reason_codes.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace studiocast::video {

inline constexpr std::uint32_t kVideoRuntimeStatusSchemaVersion = 1;
inline constexpr std::size_t kCanonicalVideoEffectCount = 9;
inline constexpr std::size_t kProductCapabilityLaneCount = 4;
inline constexpr std::size_t kProductCapabilityRecordCount =
    kCanonicalVideoEffectCount * kProductCapabilityLaneCount;

inline constexpr std::array<hal::SemanticEffectId, kCanonicalVideoEffectCount>
    kCanonicalVideoEffects{
        hal::SemanticEffectId::video_noise_removal,
        hal::SemanticEffectId::eye_contact,
        hal::SemanticEffectId::virtual_background_blur,
        hal::SemanticEffectId::virtual_background_remove,
        hal::SemanticEffectId::virtual_background_replace,
        hal::SemanticEffectId::virtual_key_light,
        hal::SemanticEffectId::auto_frame,
        hal::SemanticEffectId::vignette,
        hal::SemanticEffectId::mirror,
    };

enum class ProductCapabilityLaneId : std::uint8_t {
  invalid = 0,
  maxine = 1,
  open_cuda = 2,
  open_vulkan = 3,
  cpu = 4,
};

inline constexpr std::array<ProductCapabilityLaneId,
                            kProductCapabilityLaneCount>
    kProductCapabilityLanes{
        ProductCapabilityLaneId::maxine,
        ProductCapabilityLaneId::open_cuda,
        ProductCapabilityLaneId::open_vulkan,
        ProductCapabilityLaneId::cpu,
    };

enum class ProductCapabilityDisposition : std::uint8_t {
  invalid = 0,
  production_usable,
  usable_with_degraded_behavior,
  experimental,
  diagnostics_only,
  unavailable,
};

enum class CapabilityRefreshStatus : std::uint8_t {
  never = 0,
  in_progress,
  succeeded,
  failed,
  stale,
};

struct ProductCapabilityGenerations {
  std::uint64_t capability = 0;
  std::uint64_t configuration = 0;
  std::uint64_t provider = 0;
  std::uint64_t device = 0;
  std::uint64_t context = 0;

  friend bool operator==(const ProductCapabilityGenerations &,
                         const ProductCapabilityGenerations &) = default;
};

struct CapabilityRefreshSnapshot {
  CapabilityRefreshStatus status = CapabilityRefreshStatus::never;
  effects::EffectCapabilityCode code =
      effects::EffectCapabilityCode::capability_refresh_not_run;
  std::uint64_t requested_generation = 0;
  std::uint64_t completed_generation = 0;
};

struct ProductCapabilityLane {
  hal::SemanticEffectId effect = hal::SemanticEffectId::invalid;
  ProductCapabilityLaneId lane = ProductCapabilityLaneId::invalid;
  ProductCapabilityDisposition disposition =
      ProductCapabilityDisposition::invalid;
  effects::EffectCapabilityCode code = effects::EffectCapabilityCode::none;
  effects::EffectCapabilityCode nested_code =
      effects::EffectCapabilityCode::none;
  hal::SemanticCapabilityEvidence semantic{};
};

struct ProductCapabilitySnapshot {
  std::uint32_t schema_version = kVideoRuntimeStatusSchemaVersion;
  std::uint32_t hal_contract_version = hal::kContractVersion;
  std::uint32_t reason_code_version = effects::kEffectCapabilityCodeVersion;
  ProductCapabilityGenerations generations{};
  CapabilityRefreshSnapshot refresh{};
  std::array<ProductCapabilityLane, kProductCapabilityRecordCount> lanes{};
  std::size_t lane_count = 0;
};

enum class CapabilityValidationCode : std::uint8_t {
  valid = 0,
  wrong_schema_version,
  wrong_hal_contract_version,
  wrong_reason_code_version,
  invalid_generation,
  refresh_inconsistent,
  wrong_record_count,
  unknown_effect,
  unknown_lane,
  duplicate_lane,
  missing_lane,
  semantic_mismatch,
  readiness_inconsistent,
  provider_lane_mismatch,
  cpu_effect_provider_forbidden,
  vulkan_capability_promoted,
  vulkan_matrix_inconsistent,
};

struct CapabilityValidationResult {
  CapabilityValidationCode code = CapabilityValidationCode::valid;
  hal::SemanticEffectId effect = hal::SemanticEffectId::invalid;
  ProductCapabilityLaneId lane = ProductCapabilityLaneId::invalid;
  std::size_t record_index = 0;
};

[[nodiscard]] bool
IsCapabilitySnapshotCurrent(const ProductCapabilitySnapshot &snapshot) noexcept;

[[nodiscard]] CapabilityValidationResult ValidateProductCapabilitySnapshot(
    const ProductCapabilitySnapshot &snapshot) noexcept;

// Creates a complete, explicit, fail-closed 9x4 product matrix.  Maxine and
// Open CUDA begin unavailable; CPU is explicitly unavailable for every
// product effect; and Vulkan preserves the current production matrix (only
// mirror/fixed-center vignette may later be promoted with exact evidence).
[[nodiscard]] ProductCapabilitySnapshot MakeFailClosedProductCapabilitySnapshot(
    const ProductCapabilityGenerations &generations,
    const CapabilityRefreshSnapshot &refresh) noexcept;

[[nodiscard]] const ProductCapabilityLane *
FindProductCapabilityLane(const ProductCapabilitySnapshot &snapshot,
                          hal::SemanticEffectId effect,
                          ProductCapabilityLaneId lane) noexcept;

[[nodiscard]] ProductCapabilityLane *
FindProductCapabilityLane(ProductCapabilitySnapshot *snapshot,
                          hal::SemanticEffectId effect,
                          ProductCapabilityLaneId lane) noexcept;

// Eligibility is intentionally derived from the shared HAL predicate.  Product
// disposition or a success-looking reason token cannot independently promote
// a lane.
[[nodiscard]] bool
IsProductCapabilityLaneEligible(const ProductCapabilitySnapshot &snapshot,
                                const ProductCapabilityLane &lane) noexcept;

enum class VideoRuntimeLifecycle : std::uint8_t {
  disabled = 0,
  idle,
  starting,
  running,
  stopping,
  failed,
};

struct TypedRuntimeReadiness {
  bool configured = false;
  std::uint64_t program_id = 0;
  hal::ProgramInvalidationKey invalidation_key{};
  hal::ReadinessSnapshot value{};
};

struct VideoRuntimeStatusSnapshot {
  std::uint32_t schema_version = kVideoRuntimeStatusSchemaVersion;
  std::uint32_t hal_contract_version = hal::kContractVersion;
  std::uint64_t status_generation = 0;
  ProductCapabilityGenerations generations{};
  VideoRuntimeLifecycle lifecycle = VideoRuntimeLifecycle::disabled;
  ProductCapabilitySnapshot capabilities{};
  hal::ResolutionResult resolution{};
  TypedRuntimeReadiness readiness{};
  hal::CommittedFrameSnapshot latest_committed_frame{};
  hal::ExecutionTelemetry telemetry{};
};

enum class RuntimeStatusValidationCode : std::uint8_t {
  valid = 0,
  wrong_schema_version,
  wrong_hal_contract_version,
  invalid_generation,
  capability_generation_mismatch,
  capability_snapshot_invalid,
  resolution_malformed,
  readiness_inconsistent,
};

struct RuntimeStatusValidationResult {
  RuntimeStatusValidationCode code = RuntimeStatusValidationCode::valid;
  CapabilityValidationResult capability{};
};

[[nodiscard]] RuntimeStatusValidationResult ValidateVideoRuntimeStatusSnapshot(
    const VideoRuntimeStatusSnapshot &snapshot) noexcept;

// Additive multi-island status schema.  The singular
// VideoRuntimeStatusSnapshot remains the compatibility projection while this
// fixed-capacity contract represents every current prepared execution island
// and one daemon-authoritative state record for every stable product effect.
inline constexpr std::uint32_t kMultiIslandVideoRuntimeStatusSchemaVersion = 1;
inline constexpr std::size_t kMaxPreparedExecutionIslands = 8;

enum class PreparedExecutionRouteId : std::uint32_t {
  invalid = 0,
  final_visual = 1,
  maxine_resident = 2,
  open_cuda_virtual_background = 3,
};

struct WriterOutputCommitIdentity {
  bool valid = false;
  std::uint64_t capture_sequence = 0;
  std::uint64_t output_commit_sequence = 0;

  friend bool operator==(const WriterOutputCommitIdentity &,
                         const WriterOutputCommitIdentity &) = default;
};

// Status polling may observe the same commit repeatedly.  A changed identity
// must advance the nonzero writer sequence and may never erase a previously
// committed identity.
[[nodiscard]] bool IsWriterOutputCommitMonotonic(
    const WriterOutputCommitIdentity &previous,
    const WriterOutputCommitIdentity &current) noexcept;

struct PreparedIslandStageRecord {
  std::uint32_t stage_id = 0;
  hal::SemanticEffectId effect = hal::SemanticEffectId::invalid;
  hal::BackendId backend = hal::BackendId::invalid;
  hal::SemanticEffectProviderId provider =
      hal::SemanticEffectProviderId::invalid;
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::FailureAction failure_action = hal::FailureAction::fail_closed;
  hal::StageExecutionStatus committed_status =
      hal::StageExecutionStatus::not_run;
  bool required = false;
  bool resolved = false;
  bool committed = false;
};

struct PreparedExecutionIslandStatus {
  PreparedExecutionRouteId route = PreparedExecutionRouteId::invalid;
  std::uint64_t island_id = 0;
  std::uint64_t adapter_instance_id = 0;
  std::uint64_t snapshot_generation = 0;
  std::uint64_t capability_generation = 0;
  hal::BackendId backend = hal::BackendId::invalid;
  hal::ExecutionDomainId execution_domain{};
  hal::ResolutionResult resolution{};
  TypedRuntimeReadiness readiness{};
  hal::ExecutionTelemetry telemetry{};
  hal::CommittedFrameSnapshot latest_committed_frame{};
  std::array<PreparedIslandStageRecord, hal::kMaxPreparedStages> stages{};
  std::size_t stage_count = 0;
  bool valid = false;
  bool current = false;
};

struct StableEffectRuntimeState {
  hal::SemanticEffectId effect = hal::SemanticEffectId::invalid;
  ProductCapabilityLaneId capability_lane = ProductCapabilityLaneId::invalid;
  effects::EffectCapabilityCode capability_code =
      effects::EffectCapabilityCode::none;
  effects::EffectCapabilityCode nested_capability_code =
      effects::EffectCapabilityCode::none;
  hal::SemanticCapabilityEvidence capability{};
  ProductCapabilityLaneId selected_capability_lane =
      ProductCapabilityLaneId::invalid;
  effects::EffectCapabilityCode selected_capability_code =
      effects::EffectCapabilityCode::none;
  effects::EffectCapabilityCode selected_nested_capability_code =
      effects::EffectCapabilityCode::none;
  hal::SemanticCapabilityEvidence selected_capability{};
  hal::BackendId requested_backend = hal::BackendId::invalid;
  hal::SemanticEffectProviderId requested_provider =
      hal::SemanticEffectProviderId::invalid;
  hal::BackendId selected_backend = hal::BackendId::invalid;
  hal::SemanticEffectProviderId selected_provider =
      hal::SemanticEffectProviderId::invalid;
  hal::BackendId active_backend = hal::BackendId::invalid;
  hal::SemanticEffectProviderId active_provider =
      hal::SemanticEffectProviderId::invalid;
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::FailureAction failure_action =
      hal::FailureAction::fail_open_preserve_input;
  WriterOutputCommitIdentity commit{};
  std::uint64_t selected_island_id = 0;
  std::uint32_t committed_stage_id = 0;
  hal::StageExecutionStatus committed_stage_status =
      hal::StageExecutionStatus::not_run;
  bool strict_backend_selection = false;
  bool strict_provider_selection = false;
  bool required = false;
  bool requested = false;
  bool available = false;
  bool selected = false;
  bool active = false;
  bool degraded = false;
  bool blocked = false;
  bool fallback = false;
};

struct MultiIslandAggregateTelemetry {
  hal::TransferCounters transfers{};
  std::uint64_t current_island_count = 0;
};

// The v1 command is authoritative only for effect stages owned by a current
// prepared HAL island. A requested effect with no selected prepared island may
// still be running through the retained compatibility pipeline, so it is
// explicitly marked uncovered and the standalone serializer fails closed.
struct MultiIslandStatusAuthorityCoverage {
  std::uint16_t prepared_route_effect_mask = 0;
  std::uint16_t compatibility_uncovered_effect_mask = 0;
  bool complete = true;
};

struct MultiIslandVideoRuntimeStatusSnapshot {
  std::uint32_t schema_version = kMultiIslandVideoRuntimeStatusSchemaVersion;
  std::uint32_t hal_contract_version = hal::kContractVersion;
  std::uint32_t reason_code_version = effects::kEffectCapabilityCodeVersion;
  std::uint64_t status_generation = 0;
  ProductCapabilityGenerations generations{};
  hal::ProgramGenerations program_generations{};
  VideoRuntimeLifecycle lifecycle = VideoRuntimeLifecycle::disabled;
  ProductCapabilitySnapshot capabilities{};
  WriterOutputCommitIdentity writer_commit{};
  std::array<PreparedExecutionIslandStatus, kMaxPreparedExecutionIslands>
      islands{};
  std::size_t island_count = 0;
  std::array<StableEffectRuntimeState, kCanonicalVideoEffectCount> effects{};
  std::size_t effect_count = 0;
  MultiIslandStatusAuthorityCoverage authority{};
  MultiIslandAggregateTelemetry aggregate{};
};

struct MultiIslandVideoRuntimeStatusBuildInput {
  std::uint64_t status_generation = 0;
  ProductCapabilityGenerations generations{};
  hal::ProgramGenerations program_generations{};
  VideoRuntimeLifecycle lifecycle = VideoRuntimeLifecycle::disabled;
  ProductCapabilitySnapshot capabilities{};
  WriterOutputCommitIdentity writer_commit{};
  std::array<PreparedExecutionIslandStatus, kMaxPreparedExecutionIslands>
      islands{};
  std::size_t island_count = 0;
  std::array<StableEffectRuntimeState, kCanonicalVideoEffectCount> effects{};
  std::size_t effect_count = 0;
};

enum class MultiIslandStatusValidationCode : std::uint16_t {
  valid = 0,
  wrong_schema_version = 1,
  wrong_hal_contract_version = 2,
  wrong_reason_code_version = 3,
  invalid_generation = 4,
  capability_generation_mismatch = 5,
  capability_snapshot_invalid = 6,
  malformed_writer_commit = 7,
  too_many_islands = 8,
  invalid_island = 9,
  duplicate_island = 10,
  stale_island = 11,
  foreign_adapter = 12,
  wrong_execution_domain = 13,
  wrong_program = 14,
  wrong_generation = 15,
  malformed_island_resolution = 16,
  malformed_island_stages = 17,
  mixed_writer_commit = 18,
  wrong_effect_count = 19,
  unknown_effect = 20,
  duplicate_effect = 21,
  missing_effect = 22,
  invalid_capability_lane = 23,
  invalid_capability_evidence = 24,
  effect_state_inconsistent = 25,
  strict_backend_substitution = 26,
  strict_provider_substitution = 27,
  committed_activity_mismatch = 28,
  optional_failure_inconsistent = 29,
  aggregate_overflow = 30,
  aggregate_mismatch = 31,
  malformed_authority_coverage = 32,
  incomplete_effect_authority = 33,
};

struct MultiIslandStatusValidationResult {
  MultiIslandStatusValidationCode code = MultiIslandStatusValidationCode::valid;
  std::size_t island_index = 0;
  std::size_t effect_index = 0;
  std::uint64_t island_id = 0;
  hal::SemanticEffectId effect = hal::SemanticEffectId::invalid;
};

[[nodiscard]] MultiIslandStatusValidationResult
ValidateMultiIslandVideoRuntimeStatusSnapshot(
    const MultiIslandVideoRuntimeStatusSnapshot &snapshot) noexcept;

// Copies setup/live facts into the versioned schema, computes aggregate
// transfers exactly once per current island, and returns a default
// fail-closed output when any fact is malformed or inconsistent.
[[nodiscard]] MultiIslandStatusValidationResult
BuildMultiIslandVideoRuntimeStatusSnapshot(
    const MultiIslandVideoRuntimeStatusBuildInput &input,
    MultiIslandVideoRuntimeStatusSnapshot *output) noexcept;

struct ActiveEffectAttribution {
  std::uint32_t stage_id = 0;
  hal::SemanticEffectId effect = hal::SemanticEffectId::invalid;
  hal::BackendId backend = hal::BackendId::invalid;
  hal::SemanticEffectProviderId provider =
      hal::SemanticEffectProviderId::invalid;
};

struct ActiveRuntimeAttribution {
  std::array<ActiveEffectAttribution, hal::kMaxPreparedStages> effects{};
  std::size_t effect_count = 0;
  hal::BackendId active_backend = hal::BackendId::invalid;
  bool mixed_backends = false;
};

// Active attribution is derived only from a generation-valid committed output
// while the runtime is running.  Resolution, selected backend, startup state,
// and cumulative telemetry never imply activity.
[[nodiscard]] ActiveRuntimeAttribution DeriveActiveRuntimeAttribution(
    const VideoRuntimeStatusSnapshot &snapshot) noexcept;

enum class LegacyComputeBackendId : std::uint8_t {
  none = 0,
  cpu,
  cuda,
  maxine,
  vulkan,
  mixed,
};

struct LegacyEffectBackendEntry {
  std::string_view effect_id{};
  std::string_view backend{};
  hal::SemanticEffectId effect = hal::SemanticEffectId::invalid;
  hal::SemanticEffectProviderId provider =
      hal::SemanticEffectProviderId::invalid;
};

struct LegacyVideoRuntimeCompatibility {
  LegacyComputeBackendId selected_backend = LegacyComputeBackendId::none;
  LegacyComputeBackendId active_backend = LegacyComputeBackendId::none;
  std::string_view selected_backend_token{};
  std::string_view active_backend_token{};
  std::array<LegacyEffectBackendEntry, hal::kMaxPreparedStages>
      selected_effects{};
  std::size_t selected_effect_count = 0;
  std::array<LegacyEffectBackendEntry, hal::kMaxPreparedStages>
      active_effects{};
  std::size_t active_effect_count = 0;
};

[[nodiscard]] std::string_view
LegacyComputeBackendToken(LegacyComputeBackendId backend) noexcept;

[[nodiscard]] LegacyVideoRuntimeCompatibility
BuildLegacyVideoRuntimeCompatibility(
    const VideoRuntimeStatusSnapshot &snapshot) noexcept;

} // namespace studiocast::video
