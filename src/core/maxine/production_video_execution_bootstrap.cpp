#include "core/maxine/production_video_execution_bootstrap.h"

#include "core/cuda/cuda_driver_api.h"

#include <atomic>
#include <new>
#include <utility>

namespace studiocast::maxine {
namespace {

std::uint64_t NextBootstrapAdapterInstanceId() noexcept {
  static std::atomic<std::uint64_t> next{UINT64_C(0x4d58010000000000)};
  for (;;) {
    const std::uint64_t candidate =
        next.fetch_add(1, std::memory_order_relaxed);
    if (candidate != 0)
      return candidate;
  }
}

void ResetResult(MaxineProductionVideoExecutionBootstrapResult *result) {
  if (result)
    *result = {};
}

void SetFailure(MaxineProductionVideoExecutionBootstrapResult *result,
                hal::StatusId status, hal::ReasonId reason,
                video::PreparedMaxineHardwareEligibility eligibility) noexcept {
  if (!result)
    return;
  result->status = status;
  result->reason = reason;
  result->eligibility = eligibility;
  result->diagnostic.clear();
}

bool ZeroBudget(const hal::ResourceBudget &budget) noexcept {
  return budget.persistent_bytes == 0 && budget.scratch_bytes == 0 &&
         budget.upload_staging_bytes == 0 &&
         budget.readback_staging_bytes == 0 && budget.resource_count == 0;
}

void SetReason(hal::ReasonId *reason, hal::ReasonId value) noexcept {
  if (reason)
    *reason = value;
}

class StableResidentAdapter final : public hal::IResidentAdapter {
public:
  explicit StableResidentAdapter(hal::IResidentAdapter &delegate) noexcept
      : delegate_(&delegate),
        adapter_instance_id_(delegate.adapter_instance_id()),
        backend_(delegate.backend_id()) {}

  [[nodiscard]] std::uint64_t adapter_instance_id() const noexcept override {
    return adapter_instance_id_;
  }
  [[nodiscard]] hal::BackendId backend_id() const noexcept override {
    return backend_;
  }

  hal::StatusId Prepare(const hal::PreparedProgram &program,
                        hal::PreparedProgramHandle *candidate,
                        hal::ReadinessSnapshot *readiness,
                        hal::ReasonId *reason) noexcept override {
    if (!delegate_)
      return Stopped(readiness, reason);
    return delegate_->Prepare(program, candidate, readiness, reason);
  }

  hal::ExecuteResult
  Execute(const hal::PreparedProgramHandle &program,
          const hal::ExecuteRequest &request) noexcept override {
    if (delegate_)
      return delegate_->Execute(program, request);
    hal::ExecuteResult result;
    result.status = hal::StatusId::stopped;
    result.reason = hal::ReasonId::shutdown_requested;
    return result;
  }

  hal::StatusId ResetTemporalState(const hal::PreparedProgramHandle &program,
                                   hal::ReasonId reason) noexcept override {
    if (!delegate_)
      return hal::StatusId::stopped;
    return delegate_->ResetTemporalState(program, reason);
  }

  hal::StatusId BindHostFrame(const hal::PreparedProgramHandle &program,
                              const hal::HostFrameBinding &binding,
                              hal::HostFrameTokens *tokens,
                              hal::ReasonId *reason) noexcept override {
    if (!delegate_)
      return Stopped(nullptr, reason);
    return delegate_->BindHostFrame(program, binding, tokens, reason);
  }

  hal::StatusId
  ResolveHostFrameOutput(const hal::PreparedProgramHandle &program,
                         const hal::CompletionToken &completion,
                         const hal::ResourceHandle &resource,
                         hal::HostFrameOutputResolution *resolution,
                         hal::ReasonId *reason) noexcept override {
    if (!delegate_)
      return Stopped(nullptr, reason);
    return delegate_->ResolveHostFrameOutput(program, completion, resource,
                                             resolution, reason);
  }

  void DestroyPrepared(hal::PreparedProgramHandle *program) noexcept override {
    if (delegate_) {
      delegate_->DestroyPrepared(program);
    } else if (program) {
      *program = {};
    }
  }

  // Per-program sessions borrow this view. Whole-island shutdown remains
  // owned by the bootstrap State below.
  void Shutdown() noexcept override {}

  void StopForwarding() noexcept { delegate_ = nullptr; }

private:
  static hal::StatusId Stopped(hal::ReadinessSnapshot *readiness,
                               hal::ReasonId *reason) noexcept {
    SetReason(reason, hal::ReasonId::shutdown_requested);
    if (readiness) {
      *readiness = {};
      readiness->status = hal::StatusId::stopped;
      readiness->reason = hal::ReasonId::shutdown_requested;
    }
    return hal::StatusId::stopped;
  }

  hal::IResidentAdapter *delegate_ = nullptr;
  std::uint64_t adapter_instance_id_ = 0;
  hal::BackendId backend_ = hal::BackendId::invalid;
};

bool StageKind(hal::SemanticEffectId semantic,
               ResidentStageKind *kind) noexcept {
  if (!kind)
    return false;
  switch (semantic) {
  case hal::SemanticEffectId::virtual_background_blur:
    *kind = ResidentStageKind::background_blur;
    return true;
  case hal::SemanticEffectId::virtual_background_remove:
    *kind = ResidentStageKind::background_remove;
    return true;
  case hal::SemanticEffectId::virtual_background_replace:
    *kind = ResidentStageKind::background_replace;
    return true;
  case hal::SemanticEffectId::virtual_key_light:
    *kind = ResidentStageKind::relighting;
    return true;
  case hal::SemanticEffectId::vignette:
    *kind = ResidentStageKind::vignette;
    return true;
  case hal::SemanticEffectId::video_noise_removal:
  case hal::SemanticEffectId::eye_contact:
  case hal::SemanticEffectId::auto_frame:
  case hal::SemanticEffectId::mirror:
  case hal::SemanticEffectId::invalid:
    return false;
  }
  return false;
}

hal::SemanticEffectProviderId
ProviderFor(hal::SemanticEffectId semantic) noexcept {
  switch (semantic) {
  case hal::SemanticEffectId::virtual_background_blur:
  case hal::SemanticEffectId::virtual_background_remove:
  case hal::SemanticEffectId::virtual_background_replace:
  case hal::SemanticEffectId::virtual_key_light:
    return hal::SemanticEffectProviderId::maxine_vfx;
  case hal::SemanticEffectId::vignette:
    return hal::SemanticEffectProviderId::maxine_cuda_utility;
  case hal::SemanticEffectId::video_noise_removal:
  case hal::SemanticEffectId::eye_contact:
  case hal::SemanticEffectId::auto_frame:
  case hal::SemanticEffectId::mirror:
  case hal::SemanticEffectId::invalid:
    return hal::SemanticEffectProviderId::invalid;
  }
  return hal::SemanticEffectProviderId::invalid;
}

ProductionResidentSetup
ProductionSetup(const video::PreparedMaxineHardwareRequest &request,
                std::uint32_t stage_mask) {
  ProductionResidentSetup setup;
  setup.configuration_generation = request.generations.configuration;
  setup.width = request.width;
  setup.height = request.height;
  setup.output_width = request.width;
  setup.output_height = request.height;
  setup.enabled_maxine_stage_mask = stage_mask;
  setup.vfx_model_directory = request.vfx_model_directory;
  setup.effects = request.effects;
  setup.vignette_center_x_px = request.vignette_center_x_px;
  setup.vignette_center_y_px = request.vignette_center_y_px;
  setup.replacement_background = {
      request.replacement.rgb,        request.replacement.width,
      request.replacement.height,     request.replacement.stride_bytes,
      request.replacement.generation,
  };
  return setup;
}

hal::CapabilityEvidenceRecord
Evidence(hal::CapabilityEvidenceCode code,
         hal::SemanticEffectId semantic) noexcept {
  return {
      code,     hal::StatusId::ready,   hal::ReasonId::none,
      semantic, hal::BackendId::maxine, true,
  };
}

hal::AdapterRegistration
Registration(const video::PreparedMaxineHardwareRequest &request,
             MaxineProductionHalOwner &owner,
             std::uint32_t selection_priority) noexcept {
  hal::AdapterRegistration registration;
  registration.adapter_instance_id =
      owner.execution_domain().adapter_instance_id;
  registration.backend = hal::BackendId::maxine;
  registration.execution_domain = owner.execution_domain();
  registration.selection_priority = selection_priority;
  registration.resident_adapter = owner.adapter();
  for (std::size_t index = 0; index < request.stage_count; ++index) {
    const auto semantic = request.stages[index].semantic;
    auto &capability =
        registration
            .semantic_capabilities[registration.semantic_capability_count++];
    capability.semantic = semantic;
    capability.provider = ProviderFor(semantic);
    capability.status = hal::StatusId::ready;
    capability.live_stage = true;
    capability.production_eligible = true;
    capability.exact_evidence = true;
    capability.evidence[capability.evidence_count++] = Evidence(
        hal::CapabilityEvidenceCode::provider_runtime_available, semantic);
    capability.evidence[capability.evidence_count++] = Evidence(
        hal::CapabilityEvidenceCode::synchronization_evidence_valid, semantic);
    if (semantic != hal::SemanticEffectId::vignette) {
      capability.evidence[capability.evidence_count++] = Evidence(
          hal::CapabilityEvidenceCode::model_artifact_identity_valid, semantic);
    }
    if (index + 1 == request.stage_count &&
        request.declares_output_format_cpu_tail) {
      capability.cpu_tail = hal::CpuTailId::output_format_conversion;
      capability.cpu_tail_placement =
          hal::DeclaredCpuTail::Placement::after_stage;
    }
  }
  return registration;
}

hal::ReasonId EligibilityReason(
    video::PreparedMaxineHardwareEligibility eligibility) noexcept {
  switch (eligibility) {
  case video::PreparedMaxineHardwareEligibility::
      model_artifact_evidence_incomplete:
  case video::PreparedMaxineHardwareEligibility::temporal_reset_unsupported:
  case video::PreparedMaxineHardwareEligibility::runtime_unavailable:
    return hal::ReasonId::capability_evidence_incomplete;
  case video::PreparedMaxineHardwareEligibility::cross_domain_chain:
  case video::PreparedMaxineHardwareEligibility::multiple_resident_islands:
    return hal::ReasonId::cross_adapter_residency_forbidden;
  case video::PreparedMaxineHardwareEligibility::invalid_generation:
    return hal::ReasonId::wrong_generation;
  case video::PreparedMaxineHardwareEligibility::unsupported_host_layout:
    return hal::ReasonId::malformed_descriptor;
  case video::PreparedMaxineHardwareEligibility::not_exact_maxine:
    return hal::ReasonId::strict_backend_unavailable;
  case video::PreparedMaxineHardwareEligibility::no_qualified_stage:
  case video::PreparedMaxineHardwareEligibility::unsupported_stage:
    return hal::ReasonId::semantic_effect_unsupported;
  case video::PreparedMaxineHardwareEligibility::eligible:
    return hal::ReasonId::none;
  }
  return hal::ReasonId::capability_evidence_incomplete;
}

bool SameOrContainedBudget(const hal::ResourceBudget &reserved,
                           const hal::ResourceBudget &known) noexcept {
  return reserved.persistent_bytes >= known.persistent_bytes &&
         reserved.scratch_bytes >= known.scratch_bytes &&
         reserved.upload_staging_bytes >= known.upload_staging_bytes &&
         reserved.readback_staging_bytes >= known.readback_staging_bytes &&
         reserved.resource_count >= known.resource_count &&
         reserved.in_flight_slots == known.in_flight_slots;
}

bool ContainsProvider(const video::ProductionVideoSetupLeaseSnapshot &snapshot,
                      hal::SemanticEffectProviderId provider) noexcept {
  for (std::size_t index = 0; index < snapshot.provider_count; ++index) {
    if (snapshot.providers[index] == provider)
      return true;
  }
  return false;
}

const video::PreparedMaxineHardwareStageRequest *
FindStage(const video::PreparedMaxineHardwareRequest &request,
          hal::SemanticEffectId semantic) noexcept {
  for (std::size_t index = 0; index < request.stage_count; ++index) {
    if (request.stages[index].semantic == semantic)
      return &request.stages[index];
  }
  return nullptr;
}

bool BuildLeaseSnapshot(
    const MaxineProductionVideoExecutionBootstrapRequest &request,
    MaxineProductionVideoExecutionBootstrap &bootstrap,
    const MaxineProductionVideoExecutionBootstrapResult &result,
    video::ProductionVideoSetupLeaseSnapshot *snapshot,
    hal::ReasonId *reason) noexcept {
  if (!snapshot || result.status != hal::StatusId::ready ||
      result.reason != hal::ReasonId::none ||
      result.execution_domain !=
          bootstrap.adapter_registration().execution_domain ||
      bootstrap.production_owner() == nullptr) {
    SetReason(reason, hal::ReasonId::capability_evidence_incomplete);
    return false;
  }

  *snapshot = {};
  snapshot->state = video::ProductionVideoSetupLeaseState::live;
  snapshot->status = result.status;
  snapshot->reason = result.reason;
  snapshot->backend = hal::BackendId::maxine;
  snapshot->execution_domain = result.execution_domain;
  snapshot->generations = request.prepared_request.generations;
  if (snapshot->generations.context == 0) {
    snapshot->generations.context =
        result.execution_domain.logical_context_generation;
  }
  snapshot->reserved_budget = bootstrap.production_owner()->capacity();
  if (!SameOrContainedBudget(snapshot->reserved_budget,
                             result.known_minimum_budget)) {
    SetReason(reason, hal::ReasonId::resource_exhausted);
    return false;
  }

  snapshot->adapter = bootstrap.adapter_registration();
  snapshot->factory = bootstrap.factory_registration();
  for (std::size_t index = 0;
       index < snapshot->adapter.semantic_capability_count; ++index) {
    const auto &capability = snapshot->adapter.semantic_capabilities[index];
    if (!hal::IsSelectableSemanticCapability(capability))
      continue;
    if (!ContainsProvider(*snapshot, capability.provider)) {
      if (snapshot->provider_count >= snapshot->providers.size()) {
        SetReason(reason, hal::ReasonId::resource_exhausted);
        return false;
      }
      snapshot->providers[snapshot->provider_count++] = capability.provider;
    }
    for (std::size_t evidence_index = 0;
         evidence_index < capability.evidence_count; ++evidence_index) {
      if (snapshot->readiness.evidence_count >=
          snapshot->readiness.evidence.size()) {
        SetReason(reason, hal::ReasonId::capability_evidence_incomplete);
        return false;
      }
      snapshot->readiness.evidence[snapshot->readiness.evidence_count++] =
          capability.evidence[evidence_index];
    }
    bool requires_artifact = false;
    for (std::size_t evidence_index = 0;
         evidence_index < capability.evidence_count; ++evidence_index) {
      requires_artifact |=
          capability.evidence[evidence_index].code ==
              hal::CapabilityEvidenceCode::model_artifact_identity_valid &&
          capability.evidence[evidence_index].satisfied;
    }
    if (requires_artifact) {
      const auto *stage =
          FindStage(request.prepared_request, capability.semantic);
      if (!stage || snapshot->artifact_count >= snapshot->artifacts.size()) {
        SetReason(reason, hal::ReasonId::capability_evidence_incomplete);
        return false;
      }
      auto &binding = snapshot->artifacts[snapshot->artifact_count++];
      binding.semantic = capability.semantic;
      binding.provider = capability.provider;
      binding.artifact = stage->model;
    }
  }

  snapshot->readiness.status = snapshot->status;
  snapshot->readiness.reason = snapshot->reason;
  snapshot->readiness.requested_backend = hal::BackendId::maxine;
  snapshot->readiness.selected_backend = hal::BackendId::maxine;
  snapshot->readiness.provider = snapshot->factory.provider;
  snapshot->readiness.execution_domain = snapshot->execution_domain;
  snapshot->readiness.ready_stage_mask =
      request.prepared_request.stage_count >= 32
          ? UINT32_MAX
          : (std::uint32_t{1} << request.prepared_request.stage_count) - 1u;
  snapshot->readiness.available = true;
  snapshot->readiness.degraded = false;
  SetReason(reason, hal::ReasonId::none);
  return true;
}

class MaxineProductionVideoSetupLeaseOwner final
    : public video::IProductionVideoSetupLeaseOwner {
public:
  explicit MaxineProductionVideoSetupLeaseOwner(
      std::unique_ptr<MaxineProductionVideoExecutionBootstrap>
          bootstrap) noexcept
      : bootstrap_(std::move(bootstrap)) {}

  [[nodiscard]] hal::IResidentAdapter *
  resident_adapter() const noexcept override {
    return bootstrap_ ? bootstrap_->adapter_registration().resident_adapter
                      : nullptr;
  }

  [[nodiscard]] video::IVideoAdapterFactory *
  video_adapter_factory() const noexcept override {
    return bootstrap_ ? bootstrap_->factory_registration().factory : nullptr;
  }

  void Shutdown() noexcept override {
    if (bootstrap_)
      bootstrap_->Shutdown();
  }

private:
  std::unique_ptr<MaxineProductionVideoExecutionBootstrap> bootstrap_;
};

} // namespace

struct MaxineProductionVideoExecutionBootstrap::State {
  std::unique_ptr<MaxineProductionHalOwner> owner;
  std::unique_ptr<StableResidentAdapter> adapter_view;
  hal::AdapterRegistration registration{};
  bool stopped = false;

  void Shutdown() noexcept {
    if (stopped)
      return;
    stopped = true;
    if (adapter_view)
      adapter_view->StopForwarding();
    if (owner)
      owner->Shutdown();
  }

  ~State() { Shutdown(); }
};

std::unique_ptr<MaxineProductionVideoExecutionBootstrap>
MaxineProductionVideoExecutionBootstrap::Create(
    const MaxineProductionVideoExecutionBootstrapRequest &request,
    MaxineProductionVideoExecutionBootstrapResult *result) noexcept {
  ResetResult(result);
  const auto eligibility = video::EvaluatePreparedMaxineHardwareEligibility(
      request.prepared_request);
  if (eligibility != video::PreparedMaxineHardwareEligibility::eligible) {
    SetFailure(result, hal::StatusId::blocked, EligibilityReason(eligibility),
               eligibility);
    return nullptr;
  }
  if (request.factory_id == 0 || request.selection_priority == 0 ||
      request.minimum_synchronizations_per_frame == 0 ||
      request.minimum_synchronizations_per_frame >
          request.maximum_synchronizations_per_frame ||
      request.maximum_synchronizations_per_frame >
          video::kMaxPerFrameSynchronizationsV1) {
    SetFailure(result, hal::StatusId::blocked,
               hal::ReasonId::malformed_contract, eligibility);
    return nullptr;
  }

  try {
    auto normalized_request = request.prepared_request;
    if (normalized_request.adapter_instance_id == 0) {
      normalized_request.adapter_instance_id = NextBootstrapAdapterInstanceId();
    }
    std::uint32_t stage_mask = 0;
    for (std::size_t index = 0; index < normalized_request.stage_count;
         ++index) {
      ResidentStageKind kind{};
      if (!StageKind(normalized_request.stages[index].semantic, &kind)) {
        SetFailure(result, hal::StatusId::blocked,
                   hal::ReasonId::semantic_effect_unsupported, eligibility);
        return nullptr;
      }
      stage_mask |= ProductionResidentStageBit(kind);
    }
    const auto setup = ProductionSetup(normalized_request, stage_mask);
    hal::ResourceBudget capacity = normalized_request.capacity;
    if (ZeroBudget(capacity) && !KnownMinimumMaxineBudget(setup, &capacity)) {
      SetFailure(result, hal::StatusId::blocked,
                 hal::ReasonId::malformed_descriptor, eligibility);
      return nullptr;
    }

    MaxineProductionBackendRequest backend_request;
    backend_request.adapter_instance_id =
        normalized_request.adapter_instance_id;
    backend_request.runtime.vfx =
        reinterpret_cast<vfx::VfxApi *>(normalized_request.runtime.vfx);
    backend_request.runtime.nvcv =
        reinterpret_cast<NvcvApi *>(normalized_request.runtime.nvcv);
    backend_request.runtime.ar =
        reinterpret_cast<ar::ArApi *>(normalized_request.runtime.ar);
    backend_request.runtime.cuda = reinterpret_cast<cuda::CudaDriverApi *>(
        normalized_request.runtime.cuda);
    backend_request.setup = setup;
    backend_request.capacity = capacity;
    backend_request.device_ops = reinterpret_cast<IResidentDeviceOps *>(
        normalized_request.runtime.device_ops);

    MaxineProductionPreparationResult preparation;
    auto owner =
        MaxineProductionHalOwner::CreateBackend(backend_request, &preparation);
    if (!owner) {
      SetFailure(result, preparation.status, preparation.reason, eligibility);
      if (result) {
        result->execution_domain = preparation.execution_domain;
        result->known_minimum_budget = preparation.known_minimum_budget;
        result->diagnostic = std::move(preparation.diagnostic);
      }
      return nullptr;
    }

    auto state = std::make_shared<State>();
    state->owner = std::move(owner);
    state->adapter_view =
        std::make_unique<StableResidentAdapter>(*state->owner->adapter());
    state->registration = Registration(normalized_request, *state->owner,
                                       request.selection_priority);
    state->registration.resident_adapter = state->adapter_view.get();
    hal::StaticAdapterRegistry registry_validation;
    hal::ReasonId registration_reason = hal::ReasonId::none;
    if (registry_validation.Register(state->registration,
                                     &registration_reason) !=
        hal::StatusId::ready) {
      SetFailure(result, hal::StatusId::blocked, registration_reason,
                 eligibility);
      return nullptr;
    }

    MaxineVideoExecutionFactoryConfiguration factory_configuration;
    factory_configuration.adapter_registration = &state->registration;
    factory_configuration.adapter_lifetime = state;
    factory_configuration.device_loss_owner = state.get();
    factory_configuration.notify_device_lost = [](void *opaque) noexcept {
      auto *lifetime = static_cast<State *>(opaque);
      if (lifetime && lifetime->owner)
        lifetime->owner->NotifyDeviceLost();
    };
    factory_configuration.setup_template = std::move(normalized_request);
    factory_configuration.setup_template.capacity = capacity;
    factory_configuration.factory_id = request.factory_id;
    factory_configuration.selection_priority = request.selection_priority;
    factory_configuration.minimum_synchronizations_per_frame =
        request.minimum_synchronizations_per_frame;
    factory_configuration.maximum_synchronizations_per_frame =
        request.maximum_synchronizations_per_frame;

    auto bootstrap = std::unique_ptr<MaxineProductionVideoExecutionBootstrap>(
        new MaxineProductionVideoExecutionBootstrap());
    bootstrap->state_ = std::move(state);
    bootstrap->factory_ = std::make_unique<MaxineVideoExecutionFactory>(
        std::move(factory_configuration));
    if (result) {
      result->status = hal::StatusId::ready;
      result->reason = hal::ReasonId::none;
      result->eligibility = eligibility;
      result->execution_domain = bootstrap->state_->owner->execution_domain();
      result->known_minimum_budget =
          bootstrap->state_->owner->known_minimum_budget();
      result->diagnostic.clear();
    }
    return bootstrap;
  } catch (const std::bad_alloc &) {
    SetFailure(result, hal::StatusId::resource_exhausted,
               hal::ReasonId::resource_exhausted, eligibility);
  } catch (...) {
    SetFailure(result, hal::StatusId::blocked,
               hal::ReasonId::capability_evidence_incomplete, eligibility);
  }
  return nullptr;
}

hal::StatusId MaxineProductionVideoExecutionBootstrap::AcquireSetupLease(
    const MaxineProductionVideoExecutionBootstrapRequest &request,
    video::ProductionVideoSetupLease *destination,
    video::ProductionVideoSetupLeaseAcquireEvidence *evidence,
    MaxineProductionVideoExecutionBootstrapResult *result,
    hal::ReasonId *reason) noexcept {
  if (evidence)
    *evidence = {};
  SetReason(reason, hal::ReasonId::none);

  MaxineProductionVideoExecutionBootstrapResult candidate_result;
  auto bootstrap = Create(request, &candidate_result);
  if (!bootstrap) {
    if (result)
      *result = candidate_result;
    if (evidence) {
      evidence->status = candidate_result.status;
      evidence->reason = candidate_result.reason;
      evidence->previous_lease_retained =
          destination != nullptr && destination->live();
    }
    SetReason(reason, candidate_result.reason);
    return candidate_result.status;
  }

  try {
    video::ProductionVideoSetupLeaseSnapshot snapshot;
    hal::ReasonId snapshot_reason = hal::ReasonId::none;
    if (!BuildLeaseSnapshot(request, *bootstrap, candidate_result, &snapshot,
                            &snapshot_reason)) {
      bootstrap->Shutdown();
      candidate_result.status =
          snapshot_reason == hal::ReasonId::resource_exhausted
              ? hal::StatusId::resource_exhausted
              : hal::StatusId::blocked;
      candidate_result.reason = snapshot_reason;
      if (result)
        *result = candidate_result;
      if (evidence) {
        evidence->status = candidate_result.status;
        evidence->reason = candidate_result.reason;
        evidence->candidate_owner_rolled_back = true;
        evidence->previous_lease_retained =
            destination != nullptr && destination->live();
      }
      SetReason(reason, snapshot_reason);
      return candidate_result.status;
    }

    auto owner = std::make_unique<MaxineProductionVideoSetupLeaseOwner>(
        std::move(bootstrap));
    const auto status = video::ProductionVideoSetupLease::Acquire(
        std::move(snapshot), std::move(owner), destination, evidence, reason);
    if (status != hal::StatusId::ready && status != hal::StatusId::degraded) {
      candidate_result.status = status;
      candidate_result.reason =
          reason ? *reason : hal::ReasonId::capability_evidence_incomplete;
    }
    if (result)
      *result = candidate_result;
    return status;
  } catch (const std::bad_alloc &) {
    bootstrap->Shutdown();
    SetReason(reason, hal::ReasonId::resource_exhausted);
    if (result) {
      *result = candidate_result;
      result->status = hal::StatusId::resource_exhausted;
      result->reason = hal::ReasonId::resource_exhausted;
    }
    if (evidence) {
      evidence->status = hal::StatusId::resource_exhausted;
      evidence->reason = hal::ReasonId::resource_exhausted;
      evidence->candidate_owner_rolled_back = true;
      evidence->previous_lease_retained =
          destination != nullptr && destination->live();
    }
    return hal::StatusId::resource_exhausted;
  } catch (...) {
    bootstrap->Shutdown();
    SetReason(reason, hal::ReasonId::capability_evidence_incomplete);
    if (result) {
      *result = candidate_result;
      result->status = hal::StatusId::blocked;
      result->reason = hal::ReasonId::capability_evidence_incomplete;
    }
    if (evidence) {
      evidence->status = hal::StatusId::blocked;
      evidence->reason = hal::ReasonId::capability_evidence_incomplete;
      evidence->candidate_owner_rolled_back = true;
      evidence->previous_lease_retained =
          destination != nullptr && destination->live();
    }
    return hal::StatusId::blocked;
  }
}

MaxineProductionVideoExecutionBootstrap::
    ~MaxineProductionVideoExecutionBootstrap() = default;

const hal::AdapterRegistration &
MaxineProductionVideoExecutionBootstrap::adapter_registration() const noexcept {
  return state_->registration;
}

video::VideoAdapterFactoryRegistration
MaxineProductionVideoExecutionBootstrap::factory_registration() noexcept {
  return factory_->registration();
}

MaxineVideoExecutionFactory &
MaxineProductionVideoExecutionBootstrap::factory() noexcept {
  return *factory_;
}

const MaxineProductionHalOwner *
MaxineProductionVideoExecutionBootstrap::production_owner() const noexcept {
  return state_ ? state_->owner.get() : nullptr;
}

bool MaxineProductionVideoExecutionBootstrap::stopped() const noexcept {
  return !state_ || state_->stopped;
}

void MaxineProductionVideoExecutionBootstrap::Shutdown() noexcept {
  if (state_)
    state_->Shutdown();
}

} // namespace studiocast::maxine
