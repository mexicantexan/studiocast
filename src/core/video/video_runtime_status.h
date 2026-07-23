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
