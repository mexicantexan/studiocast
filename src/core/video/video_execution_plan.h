#pragma once

#include "core/hal/domain_planner.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace studiocast::video {

inline constexpr std::size_t kMaxCanonicalStageParameters = 16;
inline constexpr std::uint64_t kMaxPerFrameSynchronizationsV1 = 16;

// Fixed setup-time contract for one synchronous v1 frame. Legacy transfer
// counts remain exact. A factory may explicitly declare bounded variability
// for setup-proven conditional work; camera upload/final readback stay exact.
struct PerFrameTransferContract {
  std::uint32_t contract_version = hal::kContractVersion;
  std::uint64_t uploads = 0;
  std::uint64_t final_readbacks = 0;
  std::uint64_t intermediate_readbacks = 0;
  std::uint64_t intermediate_uploads = 0;
  std::uint64_t cpu_tail_boundaries = 0;
  std::uint64_t minimum_synchronizations = 0;
  std::uint64_t maximum_synchronizations = 0;
  bool variable_boundaries_declared = false;
  std::uint64_t minimum_intermediate_readbacks = 0;
  std::uint64_t maximum_intermediate_readbacks = 0;
  std::uint64_t minimum_intermediate_uploads = 0;
  std::uint64_t maximum_intermediate_uploads = 0;
  std::uint64_t minimum_cpu_tail_boundaries = 0;
  std::uint64_t maximum_cpu_tail_boundaries = 0;
};

struct VariablePerFrameTransferBounds {
  std::uint64_t minimum_intermediate_readbacks = 0;
  std::uint64_t maximum_intermediate_readbacks = 0;
  std::uint64_t minimum_intermediate_uploads = 0;
  std::uint64_t maximum_intermediate_uploads = 0;
  std::uint64_t minimum_cpu_tail_boundaries = 0;
  std::uint64_t maximum_cpu_tail_boundaries = 0;
};

// Numeric counterparts of the stable Broadcast parameter IDs. These values
// are an in-process prepared-plan contract; IPC continues to use the existing
// string IDs in broadcast_effect_contract.h.
enum class CanonicalVideoParameterId : std::uint16_t {
  invalid = 0,
  enabled = 1,
  strength = 2,
  smoothing = 3,
  headroom = 4,
  remove_color_rgb = 5,
  greenscreen_mode = 6,
  greenscreen_temporal = 7,
  look_away_enabled = 8,
  intensity = 9,
  temperature_preset = 10,
  direction_pan_degrees = 11,
  center_on_tracked_face = 12,
};

enum class CanonicalVideoParameterKind : std::uint8_t {
  invalid = 0,
  boolean,
  signed_integer,
  unsigned_integer,
  real,
};

struct CanonicalVideoParameter {
  CanonicalVideoParameterId id = CanonicalVideoParameterId::invalid;
  CanonicalVideoParameterKind kind = CanonicalVideoParameterKind::invalid;
  std::int64_t signed_value = 0;
  std::uint64_t unsigned_value = 0;
  double real_value = 0.0;
  bool boolean_value = false;
};

struct CanonicalVideoStage {
  hal::CanonicalStageIntent intent{};
  std::array<CanonicalVideoParameter, kMaxCanonicalStageParameters>
      parameters{};
  std::size_t parameter_count = 0;
  hal::ModelArtifactIdentity model{};
};

// This is the normalized daemon-owned intent. It deliberately does not contain
// an execution route or a backend object. The coordinator owns route selection.
struct NormalizedVideoExecutionIntent {
  hal::BackendRequest backend{};
  hal::PreferenceConstraints preferences{};
  hal::SemanticEffectProviderId requested_provider =
      hal::SemanticEffectProviderId::invalid;
  bool strict_provider_selection = false;
  std::array<CanonicalVideoStage, hal::kMaxPreparedStages> stages{};
  std::size_t stage_count = 0;
};

enum class VideoExecutionMigrationPolicy : std::uint8_t {
  prepared_only = 1,
  prepared_then_same_resolution_compatibility,
};

enum class VideoExecutionRoute : std::uint8_t {
  passthrough = 1,
  prepared_resident,
  legacy_compatibility,
};

struct VideoExecutionPlanRequest {
  std::uint32_t contract_version = hal::kContractVersion;
  std::uint64_t plan_id = 0;
  NormalizedVideoExecutionIntent intent{};
  hal::ImageDescriptor host_input{};
  hal::ImageDescriptor host_output{};
  hal::ProgramInvalidationKey invalidation_key{};
  hal::TemporalContract temporal{};
  hal::ResourceBudget budget{};
  std::array<hal::DeclaredCpuTail, hal::kMaxCpuTails> cpu_tails{};
  std::size_t cpu_tail_count = 0;
  hal::SetupFailurePolicy setup_failure_policy =
      hal::SetupFailurePolicy::retain_old_program;
  VideoExecutionMigrationPolicy migration_policy =
      VideoExecutionMigrationPolicy::prepared_only;
};

struct FrozenVideoExecutionPlan {
  std::uint32_t contract_version = hal::kContractVersion;
  std::uint64_t plan_id = 0;
  VideoExecutionRoute route = VideoExecutionRoute::passthrough;
  NormalizedVideoExecutionIntent intent{};
  hal::ImageDescriptor host_input{};
  hal::ImageDescriptor host_output{};
  hal::ProgramInvalidationKey invalidation_key{};
  hal::TemporalContract temporal{};
  hal::ResourceBudget budget{};
  std::array<hal::DeclaredCpuTail, hal::kMaxCpuTails> cpu_tails{};
  std::size_t cpu_tail_count = 0;
  hal::SetupFailurePolicy setup_failure_policy =
      hal::SetupFailurePolicy::retain_old_program;
  VideoExecutionMigrationPolicy migration_policy =
      VideoExecutionMigrationPolicy::prepared_only;
  hal::ResolutionResult resolution{};
  hal::DomainPlan domain_plan{};
  std::uint64_t factory_id = 0;
  PerFrameTransferContract per_frame_transfers{};
  hal::PreparedProgram prepared_program{};
};

[[nodiscard]] bool
ValidateVideoExecutionPlanRequest(const VideoExecutionPlanRequest &request,
                                  hal::ReasonId *reason) noexcept;

[[nodiscard]] hal::ResolutionRequest BuildVideoResolutionRequest(
    const NormalizedVideoExecutionIntent &intent) noexcept;

[[nodiscard]] PerFrameTransferContract
BuildPerFrameTransferContract(const hal::DomainPlan &domain_plan,
                              std::uint64_t minimum_synchronizations,
                              std::uint64_t maximum_synchronizations) noexcept;

[[nodiscard]] PerFrameTransferContract BuildVariablePerFrameTransferContract(
    const hal::DomainPlan &domain_plan,
    std::uint64_t minimum_synchronizations,
    std::uint64_t maximum_synchronizations,
    const VariablePerFrameTransferBounds &bounds) noexcept;

[[nodiscard]] bool
ValidatePerFrameTransferContract(const PerFrameTransferContract &contract,
                                 const hal::DomainPlan &domain_plan,
                                 hal::ReasonId *reason) noexcept;

} // namespace studiocast::video
