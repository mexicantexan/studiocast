#include "core/video/production_video_setup_lease.h"

#include <limits>
#include <utility>

namespace studiocast::video {
namespace {

void SetReason(hal::ReasonId *reason, hal::ReasonId value) noexcept {
  if (reason)
    *reason = value;
}

bool IsZeroHash(const std::array<std::uint8_t, 32> &value) noexcept {
  for (const auto byte : value) {
    if (byte != 0)
      return false;
  }
  return true;
}

bool ValidGenerations(const hal::ProgramGenerations &generations) noexcept {
  return generations.configuration != 0 && generations.geometry != 0 &&
         generations.model != 0 && generations.provider != 0 &&
         generations.device != 0 && generations.context != 0;
}

bool AddWithoutOverflow(std::uint64_t left, std::uint64_t right,
                        std::uint64_t *sum) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    return false;
  *sum = left + right;
  return true;
}

bool ValidBudget(const hal::ResourceBudget &budget) noexcept {
  if (budget.resource_count == 0 || budget.in_flight_slots != 1)
    return false;
  std::uint64_t total = 0;
  return AddWithoutOverflow(total, budget.persistent_bytes, &total) &&
         AddWithoutOverflow(total, budget.scratch_bytes, &total) &&
         AddWithoutOverflow(total, budget.upload_staging_bytes, &total) &&
         AddWithoutOverflow(total, budget.readback_staging_bytes, &total) &&
         total != 0;
}

bool ContainsProvider(const ProductionVideoSetupLeaseSnapshot &snapshot,
                      hal::SemanticEffectProviderId provider) noexcept {
  for (std::size_t index = 0; index < snapshot.provider_count; ++index) {
    if (snapshot.providers[index] == provider)
      return true;
  }
  return false;
}

bool HasSelectableCapability(
    const ProductionVideoSetupLeaseSnapshot &snapshot,
    hal::SemanticEffectProviderId provider,
    hal::SemanticEffectId semantic = hal::SemanticEffectId::invalid) noexcept {
  for (std::size_t index = 0;
       index < snapshot.adapter.semantic_capability_count; ++index) {
    const auto &capability = snapshot.adapter.semantic_capabilities[index];
    if (capability.provider == provider &&
        (semantic == hal::SemanticEffectId::invalid ||
         capability.semantic == semantic) &&
        hal::IsSelectableSemanticCapability(capability)) {
      return true;
    }
  }
  return false;
}

bool CapabilityRequiresArtifact(
    const hal::SemanticCapabilityEvidence &capability) noexcept {
  for (std::size_t index = 0; index < capability.evidence_count; ++index) {
    const auto &evidence = capability.evidence[index];
    if (evidence.code ==
            hal::CapabilityEvidenceCode::model_artifact_identity_valid &&
        evidence.satisfied) {
      return true;
    }
  }
  return false;
}

bool HasArtifact(const ProductionVideoSetupLeaseSnapshot &snapshot,
                 hal::SemanticEffectId semantic,
                 hal::SemanticEffectProviderId provider) noexcept {
  for (std::size_t index = 0; index < snapshot.artifact_count; ++index) {
    const auto &binding = snapshot.artifacts[index];
    if (binding.semantic == semantic && binding.provider == provider)
      return true;
  }
  return false;
}

hal::StatusId Fail(hal::ReasonId value, hal::ReasonId *reason) noexcept {
  SetReason(reason, value);
  if (value == hal::ReasonId::resource_exhausted)
    return hal::StatusId::resource_exhausted;
  if (value == hal::ReasonId::wrong_generation ||
      value == hal::ReasonId::stale_program) {
    return hal::StatusId::stale;
  }
  return hal::StatusId::blocked;
}

ProductionVideoSetupLeaseShutdownResult
CompleteShutdown(std::uint64_t attempts) noexcept {
  ProductionVideoSetupLeaseShutdownResult result;
  result.status = hal::StatusId::stopped;
  result.reason = hal::ReasonId::shutdown_requested;
  result.shutdown_attempts = attempts;
  result.shutdown_requested = true;
  result.quiescent = true;
  result.driver_context_release_permitted = true;
  return result;
}

ProductionVideoSetupLeaseShutdownResult PendingShutdown(
    std::uint64_t attempts, hal::StatusId status = hal::StatusId::degraded,
    hal::ReasonId reason = hal::ReasonId::synchronization_failed) noexcept {
  ProductionVideoSetupLeaseShutdownResult result;
  result.status =
      status == hal::StatusId::invalid || status == hal::StatusId::stopped
          ? hal::StatusId::degraded
          : status;
  result.reason = reason == hal::ReasonId::none
                      ? hal::ReasonId::synchronization_failed
                      : reason;
  result.shutdown_attempts = attempts;
  result.shutdown_requested = true;
  result.pending = true;
  return result;
}

ProductionVideoSetupLeaseShutdownResult
NormalizeShutdown(const ProductionVideoSetupLeaseShutdownResult &owner_result,
                  std::uint64_t attempts) noexcept {
  const bool proven_complete =
      owner_result.contract_version ==
          kProductionVideoSetupLeaseContractVersion &&
      owner_result.shutdown_requested && !owner_result.pending &&
      owner_result.status == hal::StatusId::stopped &&
      owner_result.reason == hal::ReasonId::shutdown_requested &&
      owner_result.quiescent && owner_result.driver_context_release_permitted;
  if (proven_complete)
    return CompleteShutdown(attempts);

  const bool truthful_pending =
      owner_result.contract_version ==
          kProductionVideoSetupLeaseContractVersion &&
      owner_result.shutdown_requested && owner_result.pending &&
      !owner_result.quiescent && !owner_result.driver_context_release_permitted;
  if (truthful_pending) {
    return PendingShutdown(attempts, owner_result.status, owner_result.reason);
  }

  // Malformed or contradictory shutdown evidence fails closed. In particular,
  // an unproven release-permitted claim can never make the lease stopped.
  return PendingShutdown(attempts);
}

} // namespace

ProductionVideoSetupLeaseShutdownResult
IProductionVideoSetupLeaseOwner::RequestShutdown() noexcept {
  Shutdown();
  return CompleteShutdown(1);
}

hal::StatusId ValidateProductionVideoSetupLease(
    const ProductionVideoSetupLeaseSnapshot &snapshot,
    const IProductionVideoSetupLeaseOwner &owner,
    hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  if (snapshot.contract_version != kProductionVideoSetupLeaseContractVersion) {
    return Fail(hal::ReasonId::unsupported_contract_version, reason);
  }
  if (snapshot.state != ProductionVideoSetupLeaseState::live ||
      (snapshot.status != hal::StatusId::ready &&
       snapshot.status != hal::StatusId::degraded) ||
      (snapshot.status == hal::StatusId::ready &&
       snapshot.reason != hal::ReasonId::none) ||
      (snapshot.status == hal::StatusId::degraded &&
       snapshot.reason == hal::ReasonId::none)) {
    return Fail(snapshot.status == hal::StatusId::resource_exhausted
                    ? hal::ReasonId::resource_exhausted
                    : hal::ReasonId::malformed_contract,
                reason);
  }
  if (snapshot.backend == hal::BackendId::invalid ||
      !hal::ValidateExecutionDomain(snapshot.execution_domain)) {
    return Fail(hal::ReasonId::malformed_execution_domain, reason);
  }
  if (!ValidGenerations(snapshot.generations) ||
      snapshot.generations.context !=
          snapshot.execution_domain.logical_context_generation) {
    return Fail(hal::ReasonId::wrong_generation, reason);
  }
  if (!ValidBudget(snapshot.reserved_budget))
    return Fail(hal::ReasonId::resource_exhausted, reason);

  if (snapshot.provider_count == 0 ||
      snapshot.provider_count > snapshot.providers.size() ||
      snapshot.artifact_count > snapshot.artifacts.size()) {
    return Fail(hal::ReasonId::capability_evidence_incomplete, reason);
  }
  for (std::size_t index = 0; index < snapshot.provider_count; ++index) {
    const auto provider = snapshot.providers[index];
    if (provider == hal::SemanticEffectProviderId::invalid ||
        !HasSelectableCapability(snapshot, provider)) {
      return Fail(hal::ReasonId::capability_evidence_incomplete, reason);
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (snapshot.providers[previous] == provider)
        return Fail(hal::ReasonId::malformed_contract, reason);
    }
  }

  const auto *resident_adapter = owner.resident_adapter();
  const auto *factory = owner.video_adapter_factory();
  if (resident_adapter == nullptr || factory == nullptr ||
      snapshot.adapter.resident_adapter != resident_adapter ||
      snapshot.factory.factory != factory) {
    return Fail(hal::ReasonId::foreign_adapter, reason);
  }
  if (snapshot.adapter.adapter_instance_id == 0 ||
      snapshot.factory.factory_id == 0 ||
      snapshot.adapter.adapter_instance_id !=
          snapshot.factory.adapter_instance_id ||
      snapshot.adapter.adapter_instance_id !=
          snapshot.execution_domain.adapter_instance_id ||
      snapshot.adapter.backend != snapshot.backend ||
      snapshot.factory.backend != snapshot.backend ||
      snapshot.execution_domain.backend != snapshot.backend ||
      resident_adapter->adapter_instance_id() !=
          snapshot.adapter.adapter_instance_id ||
      resident_adapter->backend_id() != snapshot.backend) {
    return Fail(hal::ReasonId::foreign_adapter, reason);
  }
  if (snapshot.adapter.execution_domain != snapshot.execution_domain ||
      snapshot.factory.execution_domain != snapshot.execution_domain)
    return Fail(hal::ReasonId::wrong_execution_domain, reason);
  if (snapshot.adapter.selection_priority == 0 ||
      snapshot.factory.selection_priority == 0 ||
      snapshot.adapter.selection_priority !=
          snapshot.factory.selection_priority ||
      snapshot.factory.route != VideoExecutionRoute::prepared_resident) {
    return Fail(hal::ReasonId::malformed_contract, reason);
  }

  if (snapshot.factory.provider != hal::SemanticEffectProviderId::invalid &&
      (snapshot.provider_count != 1 ||
       snapshot.providers[0] != snapshot.factory.provider)) {
    return Fail(hal::ReasonId::semantic_provider_unavailable, reason);
  }
  if (snapshot.readiness.status != snapshot.status ||
      snapshot.readiness.reason != snapshot.reason ||
      snapshot.readiness.requested_backend != snapshot.backend ||
      snapshot.readiness.selected_backend != snapshot.backend ||
      snapshot.readiness.provider != snapshot.factory.provider ||
      snapshot.readiness.execution_domain != snapshot.execution_domain ||
      !snapshot.readiness.available ||
      snapshot.readiness.degraded !=
          (snapshot.status == hal::StatusId::degraded) ||
      snapshot.readiness.rebuild_requested) {
    return Fail(hal::ReasonId::capability_evidence_incomplete, reason);
  }

  hal::StaticAdapterRegistry registry;
  hal::ReasonId registration_reason = hal::ReasonId::none;
  const auto registration_status =
      registry.Register(snapshot.adapter, &registration_reason);
  if (registration_status != hal::StatusId::ready)
    return Fail(registration_reason, reason);

  for (std::size_t index = 0;
       index < snapshot.adapter.semantic_capability_count; ++index) {
    const auto &capability = snapshot.adapter.semantic_capabilities[index];
    if (!hal::IsSelectableSemanticCapability(capability))
      continue;
    if (!ContainsProvider(snapshot, capability.provider))
      return Fail(hal::ReasonId::capability_evidence_incomplete, reason);
    if (CapabilityRequiresArtifact(capability) &&
        !HasArtifact(snapshot, capability.semantic, capability.provider)) {
      return Fail(hal::ReasonId::capability_evidence_incomplete, reason);
    }
  }

  for (std::size_t index = 0; index < snapshot.artifact_count; ++index) {
    const auto &binding = snapshot.artifacts[index];
    if (binding.semantic == hal::SemanticEffectId::invalid ||
        binding.provider == hal::SemanticEffectProviderId::invalid ||
        !ContainsProvider(snapshot, binding.provider) ||
        !HasSelectableCapability(snapshot, binding.provider,
                                 binding.semantic) ||
        IsZeroHash(binding.artifact.model_id_hash) ||
        IsZeroHash(binding.artifact.artifact_set_hash) ||
        binding.artifact.artifact_generation == 0) {
      return Fail(hal::ReasonId::capability_evidence_incomplete, reason);
    }
    if (binding.artifact.artifact_generation != snapshot.generations.model) {
      return Fail(hal::ReasonId::wrong_generation, reason);
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (snapshot.artifacts[previous].semantic == binding.semantic &&
          snapshot.artifacts[previous].provider == binding.provider) {
        return Fail(hal::ReasonId::malformed_contract, reason);
      }
    }
  }
  return hal::StatusId::ready;
}

ProductionVideoSetupLease::ProductionVideoSetupLease(
    ProductionVideoSetupLeaseSnapshot snapshot,
    std::unique_ptr<IProductionVideoSetupLeaseOwner> owner) noexcept
    : snapshot_(std::move(snapshot)), owner_(std::move(owner)) {}

ProductionVideoSetupLease::~ProductionVideoSetupLease() {
  // Exactly one final bounded attempt is made. If it remains pending, the
  // backend owner's destructor is responsible for retaining unsafe resources.
  (void)Shutdown();
}

ProductionVideoSetupLease::ProductionVideoSetupLease(
    ProductionVideoSetupLease &&other) noexcept
    : snapshot_(std::move(other.snapshot_)), owner_(std::move(other.owner_)),
      shutdown_result_(other.shutdown_result_) {
  other.snapshot_ = {};
  other.shutdown_result_ = {};
}

ProductionVideoSetupLease &ProductionVideoSetupLease::operator=(
    ProductionVideoSetupLease &&other) noexcept {
  if (this == &other)
    return *this;
  (void)Shutdown();
  if (shutdown_pending())
    (void)Shutdown();
  owner_.reset();
  snapshot_ = std::move(other.snapshot_);
  owner_ = std::move(other.owner_);
  shutdown_result_ = other.shutdown_result_;
  other.snapshot_ = {};
  other.shutdown_result_ = {};
  return *this;
}

hal::StatusId ProductionVideoSetupLease::Acquire(
    ProductionVideoSetupLeaseSnapshot snapshot,
    std::unique_ptr<IProductionVideoSetupLeaseOwner> candidate_owner,
    ProductionVideoSetupLease *destination,
    ProductionVideoSetupLeaseAcquireEvidence *evidence,
    hal::ReasonId *reason) noexcept {
  if (evidence)
    *evidence = {};
  SetReason(reason, hal::ReasonId::none);
  const bool had_previous = destination != nullptr && destination->live();
  if (!candidate_owner || destination == nullptr) {
    ProductionVideoSetupLeaseShutdownResult candidate_shutdown;
    if (candidate_owner) {
      candidate_shutdown =
          NormalizeShutdown(candidate_owner->RequestShutdown(), 1);
    }
    const auto failure = hal::ReasonId::malformed_contract;
    SetReason(reason, failure);
    if (evidence) {
      evidence->status = hal::StatusId::blocked;
      evidence->reason = failure;
      evidence->readiness = snapshot.readiness;
      evidence->candidate_shutdown = candidate_shutdown;
      evidence->candidate_owner_rolled_back =
          candidate_owner != nullptr && !candidate_shutdown.pending;
      evidence->candidate_cleanup_pending =
          candidate_owner != nullptr && candidate_shutdown.pending;
      evidence->previous_lease_retained = had_previous;
    }
    return hal::StatusId::blocked;
  }

  hal::ReasonId validation_reason = hal::ReasonId::none;
  const auto status = ValidateProductionVideoSetupLease(
      snapshot, *candidate_owner, &validation_reason);
  if (status != hal::StatusId::ready) {
    const auto candidate_shutdown =
        NormalizeShutdown(candidate_owner->RequestShutdown(), 1);
    SetReason(reason, validation_reason);
    if (evidence) {
      evidence->status = status;
      evidence->reason = validation_reason;
      evidence->readiness = snapshot.readiness;
      evidence->candidate_shutdown = candidate_shutdown;
      evidence->candidate_owner_rolled_back = !candidate_shutdown.pending;
      evidence->candidate_cleanup_pending = candidate_shutdown.pending;
      evidence->previous_lease_retained = had_previous;
    }
    return status;
  }

  ProductionVideoSetupLease candidate(std::move(snapshot),
                                      std::move(candidate_owner));
  *destination = std::move(candidate);
  if (evidence) {
    evidence->status = destination->snapshot_.status;
    evidence->reason = destination->snapshot_.reason;
    evidence->readiness = destination->snapshot_.readiness;
  }
  return destination->snapshot_.status;
}

bool ProductionVideoSetupLease::live() const noexcept {
  return owner_ != nullptr && !shutdown_result_.shutdown_requested &&
         snapshot_.state == ProductionVideoSetupLeaseState::live;
}

bool ProductionVideoSetupLease::shutdown_pending() const noexcept {
  return owner_ != nullptr && shutdown_result_.shutdown_requested &&
         shutdown_result_.pending &&
         snapshot_.state == ProductionVideoSetupLeaseState::shutdown_pending;
}

ProductionVideoSetupLeaseShutdownResult
ProductionVideoSetupLease::Shutdown() noexcept {
  if (!owner_)
    return shutdown_result_.shutdown_requested ? shutdown_result_
                                               : CompleteShutdown(0);
  if (shutdown_result_.shutdown_requested && !shutdown_result_.pending)
    return shutdown_result_;

  const auto attempts = shutdown_result_.shutdown_attempts + 1;
  shutdown_result_ = NormalizeShutdown(owner_->RequestShutdown(), attempts);
  snapshot_.shutdown = shutdown_result_;
  snapshot_.state = shutdown_result_.pending
                        ? ProductionVideoSetupLeaseState::shutdown_pending
                        : ProductionVideoSetupLeaseState::stopped;
  snapshot_.status = shutdown_result_.status;
  snapshot_.reason = shutdown_result_.reason;
  snapshot_.readiness.status = shutdown_result_.status;
  snapshot_.readiness.reason = shutdown_result_.reason;
  snapshot_.readiness.available = false;
  snapshot_.readiness.degraded = shutdown_result_.pending;
  return shutdown_result_;
}

} // namespace studiocast::video
