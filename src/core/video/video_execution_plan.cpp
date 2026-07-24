#include "core/video/video_execution_plan.h"

#include <cmath>
#include <type_traits>

namespace studiocast::video {
namespace {

void SetReason(hal::ReasonId *reason, hal::ReasonId value) noexcept {
  if (reason != nullptr)
    *reason = value;
}

bool ValidParameter(const CanonicalVideoParameter &parameter) noexcept {
  if (parameter.id == CanonicalVideoParameterId::invalid ||
      parameter.kind == CanonicalVideoParameterKind::invalid) {
    return false;
  }
  switch (parameter.kind) {
  case CanonicalVideoParameterKind::boolean:
    return parameter.signed_value == 0 && parameter.unsigned_value == 0 &&
           parameter.real_value == 0.0;
  case CanonicalVideoParameterKind::signed_integer:
    return parameter.unsigned_value == 0 && parameter.real_value == 0.0 &&
           !parameter.boolean_value;
  case CanonicalVideoParameterKind::unsigned_integer:
    return parameter.signed_value == 0 && parameter.real_value == 0.0 &&
           !parameter.boolean_value;
  case CanonicalVideoParameterKind::real:
    return parameter.signed_value == 0 && parameter.unsigned_value == 0 &&
           !parameter.boolean_value && std::isfinite(parameter.real_value);
  case CanonicalVideoParameterKind::invalid:
    break;
  }
  return false;
}

bool ValidGenerations(const hal::ProgramGenerations &generations) noexcept {
  return generations.configuration != 0 && generations.geometry != 0 &&
         generations.model != 0 && generations.provider != 0 &&
         generations.device != 0 && generations.context != 0;
}

} // namespace

static_assert(std::is_standard_layout_v<PerFrameTransferContract>);
static_assert(std::is_trivially_copyable_v<PerFrameTransferContract>);

bool ValidateVideoExecutionPlanRequest(const VideoExecutionPlanRequest &request,
                                       hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  if (request.contract_version != hal::kContractVersion) {
    SetReason(reason, hal::ReasonId::unsupported_contract_version);
    return false;
  }
  if (request.plan_id == 0 ||
      request.intent.stage_count > hal::kMaxPreparedStages ||
      request.cpu_tail_count > hal::kMaxCpuTails ||
      !hal::ValidateImageDescriptor(request.host_input) ||
      !hal::ValidateImageDescriptor(request.host_output) ||
      !ValidGenerations(request.invalidation_key.generations) ||
      request.invalidation_key.canonical_plan_hash == 0 ||
      request.invalidation_key.execution_domain_hash == 0 ||
      request.budget.in_flight_slots != 1 ||
      (request.migration_policy !=
           VideoExecutionMigrationPolicy::prepared_only &&
       request.migration_policy !=
           VideoExecutionMigrationPolicy::
               prepared_then_same_resolution_compatibility)) {
    SetReason(reason, hal::ReasonId::malformed_contract);
    return false;
  }
  if (request.intent.backend.mode != hal::BackendRequestMode::automatic &&
      request.intent.backend.mode !=
          hal::BackendRequestMode::explicit_backend) {
    SetReason(reason, hal::ReasonId::malformed_contract);
    return false;
  }
  if (request.intent.backend.mode ==
          hal::BackendRequestMode::explicit_backend &&
      request.intent.backend.backend == hal::BackendId::invalid) {
    SetReason(reason, hal::ReasonId::malformed_contract);
    return false;
  }
  if (request.intent.strict_provider_selection &&
      request.intent.requested_provider ==
          hal::SemanticEffectProviderId::invalid) {
    SetReason(reason, hal::ReasonId::malformed_contract);
    return false;
  }

  std::uint32_t stage_ids = 0;
  std::uint32_t previous_stage_id = 0;
  for (std::size_t index = 0; index < request.intent.stage_count; ++index) {
    const auto &stage = request.intent.stages[index];
    if (stage.intent.stage_id == 0 ||
        stage.intent.semantic == hal::SemanticEffectId::invalid ||
        stage.parameter_count > kMaxCanonicalStageParameters ||
        stage.intent.stage_id > hal::kMaxPreparedStages ||
        stage.intent.stage_id <= previous_stage_id) {
      SetReason(reason, hal::ReasonId::malformed_contract);
      return false;
    }
    const auto bit = std::uint32_t{1} << (stage.intent.stage_id - 1);
    if ((stage_ids & bit) != 0) {
      SetReason(reason, hal::ReasonId::malformed_contract);
      return false;
    }
    stage_ids |= bit;
    previous_stage_id = stage.intent.stage_id;
    std::uint32_t parameter_ids = 0;
    for (std::size_t parameter_index = 0;
         parameter_index < stage.parameter_count; ++parameter_index) {
      const auto &parameter = stage.parameters[parameter_index];
      const auto numeric = static_cast<std::uint16_t>(parameter.id);
      if (!ValidParameter(parameter) || numeric > 31) {
        SetReason(reason, hal::ReasonId::malformed_contract);
        return false;
      }
      const auto parameter_bit = std::uint32_t{1} << (numeric - 1);
      if ((parameter_ids & parameter_bit) != 0) {
        SetReason(reason, hal::ReasonId::malformed_contract);
        return false;
      }
      parameter_ids |= parameter_bit;
    }
  }

  for (std::size_t index = 0; index < request.cpu_tail_count; ++index) {
    const auto &tail = request.cpu_tails[index];
    if (tail.id == hal::CpuTailId::invalid ||
        tail.id == hal::CpuTailId::backend_resolved_output ||
        (tail.placement == hal::DeclaredCpuTail::Placement::before_stage &&
         (tail.before_stage_id == 0 || tail.after_stage_id != 0)) ||
        (tail.placement == hal::DeclaredCpuTail::Placement::after_stage &&
         (tail.after_stage_id == 0 || tail.before_stage_id != 0)) ||
        tail.boundary_index > request.intent.stage_count) {
      SetReason(reason, hal::ReasonId::malformed_contract);
      return false;
    }
  }
  return true;
}

hal::ResolutionRequest BuildVideoResolutionRequest(
    const NormalizedVideoExecutionIntent &intent) noexcept {
  hal::ResolutionRequest request;
  request.backend = intent.backend;
  request.preferences = intent.preferences;
  request.stage_count = intent.stage_count;
  for (std::size_t index = 0; index < intent.stage_count; ++index)
    request.stages[index] = intent.stages[index].intent;
  return request;
}

PerFrameTransferContract
BuildPerFrameTransferContract(const hal::DomainPlan &domain_plan,
                              std::uint64_t minimum_synchronizations,
                              std::uint64_t maximum_synchronizations) noexcept {
  PerFrameTransferContract contract;
  contract.uploads = domain_plan.expected_transfers.uploads;
  contract.final_readbacks = domain_plan.expected_transfers.final_readbacks;
  contract.intermediate_readbacks =
      domain_plan.expected_transfers.intermediate_readbacks;
  contract.intermediate_uploads =
      domain_plan.expected_transfers.intermediate_uploads;
  contract.cpu_tail_boundaries =
      domain_plan.expected_transfers.cpu_tail_boundaries;
  contract.minimum_synchronizations = minimum_synchronizations;
  contract.maximum_synchronizations = maximum_synchronizations;
  return contract;
}

bool ValidatePerFrameTransferContract(const PerFrameTransferContract &contract,
                                      const hal::DomainPlan &domain_plan,
                                      hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  if (contract.contract_version != hal::kContractVersion ||
      domain_plan.passthrough || domain_plan.island_count != 1 ||
      contract.uploads != domain_plan.expected_transfers.uploads ||
      contract.final_readbacks !=
          domain_plan.expected_transfers.final_readbacks ||
      contract.intermediate_readbacks !=
          domain_plan.expected_transfers.intermediate_readbacks ||
      contract.intermediate_uploads !=
          domain_plan.expected_transfers.intermediate_uploads ||
      contract.cpu_tail_boundaries !=
          domain_plan.expected_transfers.cpu_tail_boundaries ||
      contract.minimum_synchronizations > contract.maximum_synchronizations ||
      contract.maximum_synchronizations > kMaxPerFrameSynchronizationsV1) {
    SetReason(reason, hal::ReasonId::malformed_contract);
    return false;
  }

  bool has_device_island = false;
  for (std::size_t index = 0; index < domain_plan.island_count; ++index) {
    has_device_island =
        has_device_island || hal::ClassifyExecutionResidency(
                                 domain_plan.islands[index].execution_domain) ==
                                 hal::ExecutionResidencyClass::device;
  }
  if (has_device_island && contract.minimum_synchronizations == 0) {
    SetReason(reason, hal::ReasonId::malformed_contract);
    return false;
  }
  return true;
}

} // namespace studiocast::video
