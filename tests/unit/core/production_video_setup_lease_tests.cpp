#include "core/video/production_video_setup_lease.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {

namespace hal = studiocast::hal;
namespace video = studiocast::video;

constexpr std::uint64_t kAdapterId = 0x12340001;
constexpr std::uint64_t kFactoryId = 0x12340002;

bool Require(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

hal::ExecutionDomainId Domain() {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::maxine;
  domain.api_family = hal::ApiFamily::maxine_nvcv;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = 0x1cb3;
  domain.physical_device.stable_id[0] = 0x5a;
  domain.logical_context_generation = 6;
  domain.allocator_family = hal::AllocatorFamily::maxine_nvcv;
  domain.memory_handle_family = hal::MemoryHandleFamily::nvcv_image;
  domain.queue_or_stream_identity = 0x7788;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::maxine_bound_cuda_stream;
  domain.adapter_instance_id = kAdapterId;
  return domain;
}

class FakeResidentAdapter final : public hal::IResidentAdapter {
public:
  std::uint64_t adapter_instance_id() const noexcept override {
    return kAdapterId;
  }
  hal::BackendId backend_id() const noexcept override {
    return hal::BackendId::maxine;
  }
  hal::StatusId Prepare(const hal::PreparedProgram &,
                        hal::PreparedProgramHandle *, hal::ReadinessSnapshot *,
                        hal::ReasonId *) noexcept override {
    return hal::StatusId::blocked;
  }
  hal::ExecuteResult Execute(const hal::PreparedProgramHandle &,
                             const hal::ExecuteRequest &) noexcept override {
    return {};
  }
  hal::StatusId ResetTemporalState(const hal::PreparedProgramHandle &,
                                   hal::ReasonId) noexcept override {
    return hal::StatusId::blocked;
  }
  void DestroyPrepared(hal::PreparedProgramHandle *) noexcept override {}
  void Shutdown() noexcept override {}
};

class FakeFactory final : public video::IVideoAdapterFactory {
public:
  video::VideoAdapterFactoryPreflight
  Preflight(const video::VideoExecutionPlanRequest &,
            const hal::ResolutionResult &, const hal::DomainPlan &,
            video::VideoExecutionRoute) noexcept override {
    return {};
  }
  std::unique_ptr<video::IVideoExecutionExecutor>
  Prepare(const video::FrozenVideoExecutionPlan &,
          video::VideoAdapterPrepareEvidence *) noexcept override {
    return nullptr;
  }
};

struct OwnerCounters {
  std::uint64_t shutdown_calls = 0;
  std::uint64_t destructor_calls = 0;
  std::uint64_t pending_attempts_before_complete = 0;
  bool permanently_pending = false;
  bool destructor_saw_pending = false;
  std::vector<int> *events = nullptr;
};

class FakeOwner final : public video::IProductionVideoSetupLeaseOwner {
public:
  explicit FakeOwner(OwnerCounters *counters) : counters_(counters) {}
  ~FakeOwner() override {
    ++counters_->destructor_calls;
    counters_->destructor_saw_pending = last_shutdown_pending_;
    if (counters_->events)
      counters_->events->push_back(3);
  }

  hal::IResidentAdapter *resident_adapter() const noexcept override {
    return const_cast<FakeResidentAdapter *>(&adapter);
  }
  video::IVideoAdapterFactory *video_adapter_factory() const noexcept override {
    return const_cast<FakeFactory *>(&factory);
  }
  void Shutdown() noexcept override {
    ++counters_->shutdown_calls;
    if (counters_->events)
      counters_->events->push_back(2);
  }
  video::ProductionVideoSetupLeaseShutdownResult
  RequestShutdown() noexcept override {
    Shutdown();
    video::ProductionVideoSetupLeaseShutdownResult result;
    result.shutdown_attempts = counters_->shutdown_calls;
    result.shutdown_requested = true;
    const bool pending = counters_->permanently_pending ||
                         counters_->shutdown_calls <=
                             counters_->pending_attempts_before_complete;
    if (pending) {
      result.status = hal::StatusId::degraded;
      result.reason = hal::ReasonId::synchronization_failed;
      result.pending = true;
      last_shutdown_pending_ = true;
      return result;
    }
    result.status = hal::StatusId::stopped;
    result.reason = hal::ReasonId::shutdown_requested;
    result.quiescent = true;
    result.driver_context_release_permitted = true;
    last_shutdown_pending_ = false;
    return result;
  }

  FakeResidentAdapter adapter;
  FakeFactory factory;

private:
  OwnerCounters *counters_ = nullptr;
  bool last_shutdown_pending_ = false;
};

hal::CapabilityEvidenceRecord Evidence(hal::CapabilityEvidenceCode code,
                                       hal::SemanticEffectId semantic) {
  return {code,     hal::StatusId::ready,   hal::ReasonId::none,
          semantic, hal::BackendId::maxine, true};
}

video::ProductionVideoSetupLeaseSnapshot Snapshot(FakeOwner *owner) {
  video::ProductionVideoSetupLeaseSnapshot snapshot;
  snapshot.state = video::ProductionVideoSetupLeaseState::live;
  snapshot.status = hal::StatusId::ready;
  snapshot.backend = hal::BackendId::maxine;
  snapshot.execution_domain = Domain();
  snapshot.generations = {1, 2, 3, 4, 5, 6};
  snapshot.reserved_budget.persistent_bytes = 1024;
  snapshot.reserved_budget.scratch_bytes = 256;
  snapshot.reserved_budget.upload_staging_bytes = 128;
  snapshot.reserved_budget.readback_staging_bytes = 128;
  snapshot.reserved_budget.resource_count = 8;
  snapshot.reserved_budget.in_flight_slots = 1;
  snapshot.providers[0] = hal::SemanticEffectProviderId::maxine_vfx;
  snapshot.providers[1] = hal::SemanticEffectProviderId::maxine_cuda_utility;
  snapshot.provider_count = 2;

  snapshot.adapter.adapter_instance_id = kAdapterId;
  snapshot.adapter.backend = hal::BackendId::maxine;
  snapshot.adapter.execution_domain = Domain();
  snapshot.adapter.selection_priority = 10;
  snapshot.adapter.resident_adapter = owner ? &owner->adapter : nullptr;
  auto &matting = snapshot.adapter.semantic_capabilities[0];
  matting.semantic = hal::SemanticEffectId::virtual_background_blur;
  matting.provider = hal::SemanticEffectProviderId::maxine_vfx;
  matting.status = hal::StatusId::ready;
  matting.live_stage = true;
  matting.production_eligible = true;
  matting.exact_evidence = true;
  matting.evidence[0] =
      Evidence(hal::CapabilityEvidenceCode::provider_runtime_available,
               matting.semantic);
  matting.evidence[1] =
      Evidence(hal::CapabilityEvidenceCode::model_artifact_identity_valid,
               matting.semantic);
  matting.evidence_count = 2;
  auto &vignette = snapshot.adapter.semantic_capabilities[1];
  vignette.semantic = hal::SemanticEffectId::vignette;
  vignette.provider = hal::SemanticEffectProviderId::maxine_cuda_utility;
  vignette.status = hal::StatusId::ready;
  vignette.live_stage = true;
  vignette.production_eligible = true;
  vignette.exact_evidence = true;
  vignette.evidence[0] =
      Evidence(hal::CapabilityEvidenceCode::provider_runtime_available,
               vignette.semantic);
  vignette.evidence_count = 1;
  snapshot.adapter.semantic_capability_count = 2;

  auto &artifact = snapshot.artifacts[0];
  artifact.semantic = matting.semantic;
  artifact.provider = matting.provider;
  artifact.artifact.model_id_hash[0] = 1;
  artifact.artifact.artifact_set_hash[0] = 2;
  artifact.artifact.artifact_generation = snapshot.generations.model;
  snapshot.artifact_count = 1;

  snapshot.factory.factory_id = kFactoryId;
  snapshot.factory.route = video::VideoExecutionRoute::prepared_resident;
  snapshot.factory.backend = hal::BackendId::maxine;
  // Invalid is the exact declaration for a factory that consumes the frozen
  // mixed-provider stage set without performing provider selection.
  snapshot.factory.provider = hal::SemanticEffectProviderId::invalid;
  snapshot.factory.adapter_instance_id = kAdapterId;
  snapshot.factory.execution_domain = Domain();
  snapshot.factory.selection_priority = 10;
  snapshot.factory.factory = owner ? &owner->factory : nullptr;

  snapshot.readiness.status = hal::StatusId::ready;
  snapshot.readiness.requested_backend = hal::BackendId::maxine;
  snapshot.readiness.selected_backend = hal::BackendId::maxine;
  snapshot.readiness.provider = hal::SemanticEffectProviderId::invalid;
  snapshot.readiness.execution_domain = Domain();
  snapshot.readiness.ready_stage_mask = 3;
  snapshot.readiness.available = true;
  return snapshot;
}

bool Acquire(FakeOwner *owner,
             std::unique_ptr<video::IProductionVideoSetupLeaseOwner> ownership,
             video::ProductionVideoSetupLease *lease,
             video::ProductionVideoSetupLeaseAcquireEvidence *evidence,
             hal::ReasonId *reason) {
  return video::ProductionVideoSetupLease::Acquire(
             Snapshot(owner), std::move(ownership), lease, evidence, reason) ==
         hal::StatusId::ready;
}

bool TestValidMixedProviderLeaseAndMove() {
  OwnerCounters counters;
  auto owner = std::make_unique<FakeOwner>(&counters);
  auto *raw = owner.get();
  video::ProductionVideoSetupLease lease;
  video::ProductionVideoSetupLeaseAcquireEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  bool ok = Require(Acquire(raw, std::move(owner), &lease, &evidence, &reason),
                    "exact mixed-provider lease must acquire") &&
            Require(lease.live() && evidence.status == hal::StatusId::ready &&
                        reason == hal::ReasonId::none,
                    "acquired lease must expose typed ready evidence") &&
            Require(lease.snapshot().factory.provider ==
                            hal::SemanticEffectProviderId::invalid &&
                        lease.snapshot().provider_count == 2,
                    "mixed factory must retain exact per-stage provider set");

  video::ProductionVideoSetupLease moved(std::move(lease));
  ok &= Require(!lease.live() && moved.live(),
                "lease ownership must be move-only");
  moved.Shutdown();
  moved.Shutdown();
  ok &=
      Require(counters.shutdown_calls == 1 &&
                  moved.snapshot().state ==
                      video::ProductionVideoSetupLeaseState::stopped &&
                  moved.snapshot().status == hal::StatusId::stopped &&
                  moved.snapshot().adapter.resident_adapter == &raw->adapter &&
                  moved.snapshot().factory.factory == &raw->factory,
              "shutdown must be idempotent and visible");
  return ok;
}

bool TestMalformedAndForeignContracts() {
  auto validate = [](auto mutate, hal::StatusId expected_status,
                     hal::ReasonId expected_reason, const char *message) {
    OwnerCounters counters;
    FakeOwner owner(&counters);
    auto snapshot = Snapshot(&owner);
    mutate(snapshot, owner);
    hal::ReasonId reason = hal::ReasonId::none;
    return Require(video::ValidateProductionVideoSetupLease(
                       snapshot, owner, &reason) == expected_status &&
                       reason == expected_reason,
                   message);
  };

  bool ok = validate(
      [](auto &snapshot, auto &) { ++snapshot.contract_version; },
      hal::StatusId::blocked, hal::ReasonId::unsupported_contract_version,
      "foreign lease version must fail closed");
  ok &= validate(
      [](auto &snapshot, auto &) { snapshot.factory.factory = nullptr; },
      hal::StatusId::blocked, hal::ReasonId::foreign_adapter,
      "null factory pointer must be foreign");
  ok &= validate(
      [](auto &snapshot, auto &) { ++snapshot.adapter.adapter_instance_id; },
      hal::StatusId::blocked, hal::ReasonId::foreign_adapter,
      "mismatched adapter identity must be foreign");
  ok &=
      validate([](auto &snapshot,
                  auto &) { snapshot.factory.backend = hal::BackendId::cuda; },
               hal::StatusId::blocked, hal::ReasonId::foreign_adapter,
               "mismatched factory backend must be foreign");
  ok &= validate(
      [](auto &snapshot, auto &) {
        ++snapshot.factory.execution_domain.queue_or_stream_identity;
      },
      hal::StatusId::blocked, hal::ReasonId::wrong_execution_domain,
      "factory domain mismatch must fail with exact reason");
  ok &= validate([](auto &snapshot, auto &) { ++snapshot.generations.context; },
                 hal::StatusId::stale, hal::ReasonId::wrong_generation,
                 "context generation mismatch must be stale");
  ok &= validate(
      [](auto &snapshot, auto &) {
        ++snapshot.artifacts[0].artifact.artifact_generation;
      },
      hal::StatusId::stale, hal::ReasonId::wrong_generation,
      "artifact generation mismatch must be stale");
  ok &= validate(
      [](auto &snapshot, auto &) {
        snapshot.artifacts[0].semantic = hal::SemanticEffectId::eye_contact;
      },
      hal::StatusId::blocked, hal::ReasonId::capability_evidence_incomplete,
      "foreign artifact evidence must fail closed");
  ok &= validate(
      [](auto &snapshot, auto &) { snapshot.artifact_count = 0; },
      hal::StatusId::blocked, hal::ReasonId::capability_evidence_incomplete,
      "model-backed capability must require exact artifact evidence");
  ok &= validate(
      [](auto &snapshot, auto &) {
        snapshot.factory.provider = hal::SemanticEffectProviderId::maxine_vfx;
        snapshot.readiness.provider = snapshot.factory.provider;
      },
      hal::StatusId::blocked, hal::ReasonId::semantic_provider_unavailable,
      "single-provider factory declaration must not accept a mixed set");
  ok &= validate(
      [](auto &snapshot, auto &) {
        snapshot.reserved_budget.persistent_bytes =
            std::numeric_limits<std::uint64_t>::max();
      },
      hal::StatusId::resource_exhausted, hal::ReasonId::resource_exhausted,
      "overflowing resource budget must fail as exhausted");
  return ok;
}

bool TestFailedCandidateRetainsOldLease() {
  OwnerCounters old_counters;
  auto old_owner = std::make_unique<FakeOwner>(&old_counters);
  auto *old_raw = old_owner.get();
  video::ProductionVideoSetupLease lease;
  video::ProductionVideoSetupLeaseAcquireEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  if (!Require(
          Acquire(old_raw, std::move(old_owner), &lease, &evidence, &reason),
          "old lease must acquire")) {
    return false;
  }

  OwnerCounters candidate_counters;
  auto candidate = std::make_unique<FakeOwner>(&candidate_counters);
  auto *candidate_raw = candidate.get();
  auto malformed = Snapshot(candidate_raw);
  ++malformed.generations.context;
  const auto status = video::ProductionVideoSetupLease::Acquire(
      malformed, std::move(candidate), &lease, &evidence, &reason);
  return Require(status == hal::StatusId::stale &&
                     reason == hal::ReasonId::wrong_generation,
                 "invalid replacement must preserve exact failure") &&
         Require(evidence.candidate_owner_rolled_back &&
                     evidence.previous_lease_retained && lease.live(),
                 "failed candidate must retain the old live lease") &&
         Require(candidate_counters.shutdown_calls == 1 &&
                     candidate_counters.destructor_calls == 1,
                 "failed candidate must roll back and destroy once") &&
         Require(old_counters.shutdown_calls == 0,
                 "failed replacement must not shutdown old lease");
}

bool TestCoordinatorBeforeLeaseDestructionOrder() {
  struct FakeCoordinator {
    explicit FakeCoordinator(std::vector<int> *events_out)
        : event_log(events_out) {}
    ~FakeCoordinator() { event_log->push_back(1); }
    std::vector<int> *event_log;
  };

  std::vector<int> events;
  OwnerCounters counters;
  counters.events = &events;
  {
    auto owner = std::make_unique<FakeOwner>(&counters);
    auto *raw = owner.get();
    video::ProductionVideoSetupLease lease;
    video::ProductionVideoSetupLeaseAcquireEvidence evidence;
    hal::ReasonId reason = hal::ReasonId::none;
    if (!Require(Acquire(raw, std::move(owner), &lease, &evidence, &reason),
                 "ordered lifetime lease must acquire")) {
      return false;
    }
    // Composition must declare/create the lease before the coordinator so
    // reverse destruction tears down coordinator first.
    FakeCoordinator coordinator(&events);
  }
  const std::array<int, 3> expected{1, 2, 3};
  return Require(events.size() == expected.size(),
                 "ordered teardown must record three events") &&
         Require(std::equal(events.begin(), events.end(), expected.begin()),
                 "coordinator must die before lease shutdown and owner delete");
}

bool TestPendingShutdownRetriesOnlyUntilProvenComplete() {
  OwnerCounters counters;
  counters.pending_attempts_before_complete = 1;
  auto owner = std::make_unique<FakeOwner>(&counters);
  auto *raw = owner.get();
  video::ProductionVideoSetupLease lease;
  video::ProductionVideoSetupLeaseAcquireEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  if (!Require(Acquire(raw, std::move(owner), &lease, &evidence, &reason),
               "retryable shutdown lease must acquire")) {
    return false;
  }

  const auto pending = lease.Shutdown();
  bool ok = Require(
      pending.pending && pending.shutdown_requested && !pending.quiescent &&
          !pending.driver_context_release_permitted &&
          pending.status == hal::StatusId::degraded &&
          pending.reason == hal::ReasonId::synchronization_failed &&
          pending.shutdown_attempts == 1 && lease.shutdown_pending() &&
          !lease.live() &&
          lease.snapshot().state ==
              video::ProductionVideoSetupLeaseState::shutdown_pending,
      "first unproven shutdown must remain visibly pending");

  const auto complete = lease.Shutdown();
  ok &= Require(!complete.pending && complete.quiescent &&
                    complete.driver_context_release_permitted &&
                    complete.status == hal::StatusId::stopped &&
                    complete.reason == hal::ReasonId::shutdown_requested &&
                    complete.shutdown_attempts == 2 &&
                    !lease.shutdown_pending() &&
                    lease.snapshot().state ==
                        video::ProductionVideoSetupLeaseState::stopped,
                "second proven shutdown must become stopped");
  const auto idempotent = lease.Shutdown();
  ok &=
      Require(idempotent.shutdown_attempts == 2 && counters.shutdown_calls == 2,
              "completed shutdown must be idempotent");
  return ok;
}

bool TestPermanentPendingHasBoundedCallsAndFinalAttempt() {
  OwnerCounters counters;
  counters.permanently_pending = true;
  {
    auto owner = std::make_unique<FakeOwner>(&counters);
    auto *raw = owner.get();
    video::ProductionVideoSetupLease lease;
    video::ProductionVideoSetupLeaseAcquireEvidence evidence;
    hal::ReasonId reason = hal::ReasonId::none;
    if (!Require(Acquire(raw, std::move(owner), &lease, &evidence, &reason),
                 "permanent-pending lease must acquire")) {
      return false;
    }
    const auto first = lease.Shutdown();
    const auto second = lease.Shutdown();
    if (!Require(first.pending && first.shutdown_attempts == 1 &&
                     second.pending && second.shutdown_attempts == 2 &&
                     counters.shutdown_calls == 2,
                 "explicit pending retries must perform one bounded attempt "
                 "per call")) {
      return false;
    }
  }
  return Require(
             counters.shutdown_calls == 3,
             "destruction must perform exactly one final bounded attempt") &&
         Require(counters.destructor_calls == 1 &&
                     counters.destructor_saw_pending,
                 "permanent pending destruction must preserve fail-safe owner "
                 "responsibility");
}

} // namespace

int main() {
  bool ok = true;
  ok &= TestValidMixedProviderLeaseAndMove();
  ok &= TestMalformedAndForeignContracts();
  ok &= TestFailedCandidateRetainsOldLease();
  ok &= TestCoordinatorBeforeLeaseDestructionOrder();
  ok &= TestPendingShutdownRetriesOnlyUntilProvenComplete();
  ok &= TestPermanentPendingHasBoundedCallsAndFinalAttempt();
  if (!ok)
    return 1;
  std::cout << "production video setup lease tests passed\n";
  return 0;
}
