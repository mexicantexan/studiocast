#include "core/maxine/production_video_execution_bootstrap.h"

#include "core/cuda/cuda_driver_api.h"
#include "core/maxine/ar_api.h"
#include "core/maxine/maxine_manager.h"
#include "core/maxine/nvcv_api.h"
#include "core/maxine/vfx_api.h"
#include "core/video/image_ppm.h"
#include "core/video/production_video_setup_request.h"

#include <atomic>
#include <filesystem>
#include <limits>
#include <new>
#include <string_view>
#include <utility>
#include <vector>

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
  case hal::SemanticEffectId::video_noise_removal:
    *kind = ResidentStageKind::denoise;
    return true;
  case hal::SemanticEffectId::eye_contact:
    *kind = ResidentStageKind::eye_contact;
    return true;
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
  case hal::SemanticEffectId::auto_frame:
    *kind = ResidentStageKind::auto_frame;
    return true;
  case hal::SemanticEffectId::vignette:
    *kind = ResidentStageKind::vignette;
    return true;
  case hal::SemanticEffectId::mirror:
  case hal::SemanticEffectId::invalid:
    return false;
  }
  return false;
}

hal::SemanticEffectProviderId
ProviderFor(hal::SemanticEffectId semantic) noexcept {
  switch (semantic) {
  case hal::SemanticEffectId::video_noise_removal:
  case hal::SemanticEffectId::virtual_background_blur:
  case hal::SemanticEffectId::virtual_background_remove:
  case hal::SemanticEffectId::virtual_background_replace:
  case hal::SemanticEffectId::virtual_key_light:
    return hal::SemanticEffectProviderId::maxine_vfx;
  case hal::SemanticEffectId::eye_contact:
  case hal::SemanticEffectId::auto_frame:
    return hal::SemanticEffectProviderId::maxine_ar;
  case hal::SemanticEffectId::vignette:
    return hal::SemanticEffectProviderId::maxine_cuda_utility;
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
  setup.output_width =
      request.output_width == 0 ? request.width : request.output_width;
  setup.output_height =
      request.output_height == 0 ? request.height : request.output_height;
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

bool ExactTerminalOutputTail(const video::VideoExecutionPlanRequest &plan,
                             hal::CpuTailId id) noexcept {
  if (plan.intent.stage_count == 0 ||
      plan.intent.stage_count > plan.intent.stages.size() ||
      plan.cpu_tail_count != 1) {
    return false;
  }
  const auto &tail = plan.cpu_tails[0];
  return tail.id == id && tail.required &&
         tail.placement == hal::DeclaredCpuTail::Placement::after_stage &&
         tail.after_stage_id ==
             plan.intent.stages[plan.intent.stage_count - 1].intent.stage_id &&
         tail.before_stage_id == 0 &&
         tail.boundary_index == plan.intent.stage_count;
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
  std::shared_ptr<void> runtime_lifetime;
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
    state->runtime_lifetime = request.runtime_lifetime;
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

namespace {

struct MaxineSetupRuntimeLifetime {
  vfx::VfxApi vfx;
  NvcvApi nvcv;
  ar::ArApi ar;
  cuda::CudaDriverApi cuda;
};

struct MaxineSetupLeaseLifetime {
  std::shared_ptr<void> runtime;
  std::vector<std::uint8_t> replacement_rgb;
};

bool IsVfxSemantic(hal::SemanticEffectId semantic) noexcept {
  return semantic == hal::SemanticEffectId::video_noise_removal ||
         semantic == hal::SemanticEffectId::virtual_background_blur ||
         semantic == hal::SemanticEffectId::virtual_background_remove ||
         semantic == hal::SemanticEffectId::virtual_background_replace ||
         semantic == hal::SemanticEffectId::virtual_key_light;
}

bool IsArSemantic(hal::SemanticEffectId semantic) noexcept {
  return semantic == hal::SemanticEffectId::eye_contact ||
         semantic == hal::SemanticEffectId::auto_frame;
}

bool HasIdentity(const std::array<std::uint8_t, 32> &value) noexcept {
  for (const auto byte : value) {
    if (byte != 0)
      return true;
  }
  return false;
}

void HashByte(std::array<std::uint64_t, 4> *state, std::uint8_t byte) noexcept {
  static constexpr std::array<std::uint64_t, 4> primes{
      UINT64_C(1099511628211), UINT64_C(1099511627791), UINT64_C(1099511627689),
      UINT64_C(1099511627563)};
  for (std::size_t index = 0; index < state->size(); ++index) {
    (*state)[index] ^= static_cast<std::uint64_t>(
        byte + static_cast<std::uint8_t>(index * 37u));
    (*state)[index] *= primes[index];
  }
}

void HashText(std::array<std::uint64_t, 4> *state,
              std::string_view value) noexcept {
  for (const auto ch : value)
    HashByte(state, static_cast<std::uint8_t>(ch));
}

std::array<std::uint8_t, 32>
Identity(std::string_view role, hal::SemanticEffectId semantic,
         const std::filesystem::path &sdk_library,
         const std::filesystem::path &model_directory,
         std::uint64_t generation) noexcept {
  std::array<std::uint64_t, 4> state{
      UINT64_C(1469598103934665603), UINT64_C(1099511628211),
      UINT64_C(7809847782465536322), UINT64_C(9650029242287828579)};
  HashText(&state, role);
  HashText(&state, sdk_library.string());
  HashText(&state, model_directory.string());
  for (unsigned shift = 0; shift != 64; shift += 8)
    HashByte(&state, static_cast<std::uint8_t>(generation >> shift));
  const auto semantic_value = static_cast<std::uint32_t>(semantic);
  for (unsigned shift = 0; shift != 32; shift += 8)
    HashByte(&state, static_cast<std::uint8_t>(semantic_value >> shift));
  std::array<std::uint8_t, 32> result{};
  for (std::size_t word = 0; word < state.size(); ++word) {
    for (unsigned byte = 0; byte != 8; ++byte) {
      result[word * 8 + byte] =
          static_cast<std::uint8_t>(state[word] >> (byte * 8));
    }
  }
  return result;
}

std::uint64_t
ExecutionDomainHash(const hal::ExecutionDomainId &domain) noexcept {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const auto mix = [&](std::uint64_t value) {
    hash ^= value;
    hash *= UINT64_C(1099511628211);
  };
  mix(static_cast<std::uint64_t>(domain.backend));
  mix(static_cast<std::uint64_t>(domain.api_family));
  mix(domain.physical_device.vendor_id);
  mix(domain.physical_device.device_id);
  for (const auto byte : domain.physical_device.stable_id)
    mix(byte);
  mix(domain.logical_context_generation);
  mix(static_cast<std::uint64_t>(domain.allocator_family));
  mix(static_cast<std::uint64_t>(domain.memory_handle_family));
  mix(domain.queue_or_stream_identity);
  mix(static_cast<std::uint64_t>(domain.ownership_model));
  mix(static_cast<std::uint64_t>(domain.synchronization_protocol));
  mix(domain.adapter_instance_id);
  return hash == 0 ? 1 : hash;
}

std::uint64_t NonzeroGeneration(const video::VideoExecutionPlanRequest &plan,
                                const std::filesystem::path &library,
                                const std::filesystem::path &models) noexcept {
  if (plan.invalidation_key.generations.model != 0)
    return plan.invalidation_key.generations.model;
  std::array<std::uint64_t, 4> state{
      UINT64_C(1469598103934665603), UINT64_C(1099511628211),
      UINT64_C(7809847782465536322), UINT64_C(9650029242287828579)};
  HashText(&state, library.string());
  HashText(&state, models.string());
  for (unsigned shift = 0; shift != 64; shift += 8) {
    HashByte(&state, static_cast<std::uint8_t>(
                         plan.invalidation_key.canonical_plan_hash >> shift));
  }
  return state[0] == 0 ? 1 : state[0];
}

bool SignedParameter(const video::CanonicalVideoParameter &parameter,
                     int *value) noexcept {
  if (parameter.kind != video::CanonicalVideoParameterKind::signed_integer ||
      parameter.signed_value < std::numeric_limits<int>::min() ||
      parameter.signed_value > std::numeric_limits<int>::max()) {
    return false;
  }
  *value = static_cast<int>(parameter.signed_value);
  return true;
}

bool ApplyCanonicalEffects(
    const video::NormalizedVideoExecutionIntent &intent,
    video::effects::BroadcastCameraEffects *effects) noexcept {
  effects->virtual_background.mode =
      video::effects::VirtualBackgroundMode::none;
  effects->video_noise_removal.enabled = false;
  effects->eye_contact.enabled = false;
  effects->auto_frame.enabled = false;
  effects->virtual_key_light.enabled = false;
  effects->vignette.enabled = false;
  for (std::size_t stage_index = 0; stage_index < intent.stage_count;
       ++stage_index) {
    const auto &stage = intent.stages[stage_index];
    switch (stage.intent.semantic) {
    case hal::SemanticEffectId::video_noise_removal:
      effects->video_noise_removal.enabled = true;
      break;
    case hal::SemanticEffectId::eye_contact:
      effects->eye_contact.enabled = true;
      break;
    case hal::SemanticEffectId::virtual_background_blur:
      effects->virtual_background.mode =
          video::effects::VirtualBackgroundMode::blur;
      break;
    case hal::SemanticEffectId::virtual_background_remove:
      effects->virtual_background.mode =
          video::effects::VirtualBackgroundMode::remove;
      break;
    case hal::SemanticEffectId::virtual_background_replace:
      effects->virtual_background.mode =
          video::effects::VirtualBackgroundMode::replace;
      break;
    case hal::SemanticEffectId::virtual_key_light:
      effects->virtual_key_light.enabled = true;
      break;
    case hal::SemanticEffectId::auto_frame:
      effects->auto_frame.enabled = true;
      break;
    case hal::SemanticEffectId::vignette:
      effects->vignette.enabled = true;
      break;
    case hal::SemanticEffectId::mirror:
    case hal::SemanticEffectId::invalid:
      return false;
    }
    for (std::size_t parameter_index = 0;
         parameter_index < stage.parameter_count; ++parameter_index) {
      const auto &parameter = stage.parameters[parameter_index];
      int signed_value = 0;
      switch (parameter.id) {
      case video::CanonicalVideoParameterId::enabled:
        if (parameter.kind != video::CanonicalVideoParameterKind::boolean ||
            !parameter.boolean_value)
          return false;
        break;
      case video::CanonicalVideoParameterId::strength:
        if (!SignedParameter(parameter, &signed_value))
          return false;
        if (stage.intent.semantic ==
            hal::SemanticEffectId::video_noise_removal) {
          effects->video_noise_removal.strength = signed_value;
        } else if (stage.intent.semantic ==
                   hal::SemanticEffectId::eye_contact) {
          effects->eye_contact.strength = signed_value;
        } else if (stage.intent.semantic == hal::SemanticEffectId::auto_frame) {
          effects->auto_frame.strength = signed_value;
        } else {
          effects->virtual_background.strength = signed_value;
        }
        break;
      case video::CanonicalVideoParameterId::smoothing:
        if (stage.intent.semantic != hal::SemanticEffectId::auto_frame ||
            !SignedParameter(parameter, &signed_value))
          return false;
        effects->auto_frame.smoothing = signed_value;
        break;
      case video::CanonicalVideoParameterId::headroom:
        if (stage.intent.semantic != hal::SemanticEffectId::auto_frame ||
            parameter.kind != video::CanonicalVideoParameterKind::real)
          return false;
        effects->auto_frame.headroom = static_cast<float>(parameter.real_value);
        break;
      case video::CanonicalVideoParameterId::remove_color_rgb: {
        if (parameter.kind !=
                video::CanonicalVideoParameterKind::unsigned_integer ||
            parameter.unsigned_value > 0xffffffu)
          return false;
        constexpr char digits[] = "0123456789ABCDEF";
        effects->virtual_background.remove_color = "#000000";
        for (std::size_t digit = 0; digit < 6; ++digit) {
          const auto shift = static_cast<unsigned>((5 - digit) * 4);
          effects->virtual_background.remove_color[digit + 1] =
              digits[(parameter.unsigned_value >> shift) & 0xfu];
        }
        break;
      }
      case video::CanonicalVideoParameterId::greenscreen_mode:
        if (parameter.kind !=
                video::CanonicalVideoParameterKind::unsigned_integer ||
            parameter.unsigned_value >
                std::numeric_limits<std::uint32_t>::max())
          return false;
        effects->virtual_background.greenscreen_mode =
            static_cast<std::uint32_t>(parameter.unsigned_value);
        break;
      case video::CanonicalVideoParameterId::greenscreen_temporal:
        if (parameter.kind != video::CanonicalVideoParameterKind::boolean)
          return false;
        effects->virtual_background.greenscreen_temporal =
            parameter.boolean_value;
        break;
      case video::CanonicalVideoParameterId::look_away_enabled:
        if (stage.intent.semantic != hal::SemanticEffectId::eye_contact ||
            parameter.kind != video::CanonicalVideoParameterKind::boolean)
          return false;
        effects->eye_contact.look_away_enabled = parameter.boolean_value;
        break;
      case video::CanonicalVideoParameterId::intensity:
        if (!SignedParameter(parameter, &signed_value))
          return false;
        if (stage.intent.semantic == hal::SemanticEffectId::virtual_key_light) {
          effects->virtual_key_light.intensity = signed_value;
        } else if (stage.intent.semantic == hal::SemanticEffectId::vignette) {
          effects->vignette.intensity = signed_value;
        } else {
          return false;
        }
        break;
      case video::CanonicalVideoParameterId::temperature_preset:
        if (stage.intent.semantic != hal::SemanticEffectId::virtual_key_light ||
            !SignedParameter(parameter, &signed_value))
          return false;
        effects->virtual_key_light.temperature_preset = signed_value;
        break;
      case video::CanonicalVideoParameterId::direction_pan_degrees:
        if (stage.intent.semantic != hal::SemanticEffectId::virtual_key_light ||
            !SignedParameter(parameter, &signed_value))
          return false;
        effects->virtual_key_light.direction_pan_degrees = signed_value;
        break;
      case video::CanonicalVideoParameterId::center_on_tracked_face:
        if (stage.intent.semantic != hal::SemanticEffectId::vignette ||
            parameter.kind != video::CanonicalVideoParameterKind::boolean ||
            parameter.boolean_value)
          return false;
        effects->vignette.center_on_tracked_face = false;
        break;
      case video::CanonicalVideoParameterId::invalid:
        return false;
      }
    }
  }
  return true;
}

class MaxineProductionVideoBackendSetupSource final
    : public video::IProductionVideoBackendSetupSource {
public:
  explicit MaxineProductionVideoBackendSetupSource(
      MaxineProductionBackendSetupSourceConfiguration configuration)
      : configuration_(std::move(configuration)) {}

  video::ProductionVideoResidentSetupSourceResult AcquireResidentSetup(
      const video::ProductionVideoResidentSetup &request) noexcept override {
    video::ProductionVideoResidentSetupSourceResult result;
    result.setup_completed_outside_frame_loop = true;
    auto fail = [&](hal::StatusId status, hal::ReasonId reason) {
      result.status = status;
      result.reason = reason;
      result.readiness.status = status;
      result.readiness.reason = reason;
      result.readiness.requested_backend = hal::BackendId::maxine;
      result.readiness.selected_backend = hal::BackendId::maxine;
      result.readiness.available = false;
      return std::move(result);
    };

    try {
      auto plan = request.plan;
      plan.temporal = video::ProductionVideoTemporalForSelectedBackend(
          request, hal::BackendId::maxine);
      const bool strict_constraint_mismatch =
          (plan.intent.preferences.strict_effects_engine &&
           plan.intent.preferences.effects_engine ==
               hal::EffectsEnginePreferenceConstraint::open_cuda) ||
          (plan.intent.backend.mode ==
               hal::BackendRequestMode::explicit_backend &&
           plan.intent.backend.backend != hal::BackendId::maxine &&
           plan.intent.backend.strict);
      if (plan.intent.stage_count == 0 ||
          plan.intent.stage_count > hal::kMaxPreparedStages ||
          plan.intent.preferences.compute_family ==
              hal::ComputeFamilyAllowance::cpu_only ||
          plan.intent.preferences.compute_family ==
              hal::ComputeFamilyAllowance::vulkan_only ||
          (plan.intent.backend.mode ==
               hal::BackendRequestMode::explicit_backend &&
           plan.intent.backend.backend != hal::BackendId::maxine)) {
        return fail(hal::StatusId::blocked,
                    strict_constraint_mismatch
                        ? hal::ReasonId::strict_backend_unavailable
                        : hal::ReasonId::semantic_effect_unsupported);
      }
      if (!ExactTerminalOutputTail(
              plan, hal::CpuTailId::backend_resolved_output) &&
          !ExactTerminalOutputTail(
              plan, hal::CpuTailId::output_format_conversion)) {
        return fail(hal::StatusId::blocked,
                    hal::ReasonId::cpu_tail_required);
      }

      bool needs_vfx = false;
      bool needs_ar = false;
      for (std::size_t index = 0; index < plan.intent.stage_count; ++index) {
        const auto semantic = plan.intent.stages[index].intent.semantic;
        ResidentStageKind stage_kind{};
        if (!StageKind(semantic, &stage_kind))
          return fail(hal::StatusId::blocked,
                      hal::ReasonId::semantic_effect_unsupported);
        needs_vfx |= IsVfxSemantic(semantic);
        needs_ar |= IsArSemantic(semantic);
      }

      std::filesystem::path vfx_library = configuration_.sdk_library_path;
      std::filesystem::path ar_library = configuration_.sdk_library_path;
      std::filesystem::path nvcv_library = configuration_.sdk_library_path;
      std::filesystem::path model_directory =
          configuration_.vfx_model_directory;
      if (!configuration_.bypass_system_diagnostics_for_testing) {
        const auto diagnostics = MaxineManager{}.Diagnose(false);
        if (!diagnostics.ok)
          return fail(hal::StatusId::unavailable,
                      hal::ReasonId::strict_backend_unavailable);
        if (needs_vfx) {
          vfx_library = diagnostics.vfx.library;
          model_directory = diagnostics.vfx.models_dir;
        }
        if (needs_ar)
          ar_library = diagnostics.ar.library;
        nvcv_library.clear();
      }

      video::PreparedMaxineOpaqueRuntime runtime =
          configuration_.injected_runtime;
      std::shared_ptr<void> runtime_lifetime =
          configuration_.injected_runtime_lifetime;
      if (runtime.vfx == 0 && runtime.nvcv == 0 && runtime.ar == 0 &&
          runtime.cuda == 0) {
        auto owned = std::make_shared<MaxineSetupRuntimeLifetime>();
        std::string error;
        if (!owned->cuda.Initialize(&error) ||
            (configuration_.cuda_device_ordinal >= 0 &&
             !owned->cuda.SetDeviceOrdinal(configuration_.cuda_device_ordinal,
                                           &error))) {
          return fail(hal::StatusId::unavailable,
                      hal::ReasonId::capability_evidence_incomplete);
        }
        const bool nvcv_ready =
            nvcv_library.empty()
                ? owned->nvcv.Initialize(NvcvApi::Requirement::VfxCompat,
                                         &error)
                : owned->nvcv.InitializeFromLibraryPath(
                      NvcvApi::Requirement::VfxCompat, nvcv_library, &error);
        if (!nvcv_ready ||
            (needs_vfx &&
             !(vfx_library.empty() ? owned->vfx.Initialize(&error)
                                   : owned->vfx.InitializeFromLibraryPath(
                                         vfx_library, &error))) ||
            (needs_ar &&
             !(ar_library.empty() ? owned->ar.Initialize(&error)
                                  : owned->ar.InitializeFromLibraryPath(
                                        ar_library, &error)))) {
          return fail(hal::StatusId::unavailable,
                      hal::ReasonId::capability_evidence_incomplete);
        }
        runtime.vfx =
            needs_vfx ? reinterpret_cast<std::uintptr_t>(&owned->vfx) : 0;
        runtime.nvcv = reinterpret_cast<std::uintptr_t>(&owned->nvcv);
        runtime.ar =
            needs_ar ? reinterpret_cast<std::uintptr_t>(&owned->ar) : 0;
        runtime.cuda = reinterpret_cast<std::uintptr_t>(&owned->cuda);
        runtime_lifetime = std::move(owned);
      } else if (!runtime_lifetime || runtime.nvcv == 0 || runtime.cuda == 0 ||
                 (needs_vfx && runtime.vfx == 0) ||
                 (needs_ar && runtime.ar == 0)) {
        return fail(hal::StatusId::blocked,
                    hal::ReasonId::capability_evidence_incomplete);
      }

      auto prepared = video::PreparedMaxineHardwareRequest{};
      prepared.requested_backend = hal::BackendId::maxine;
      prepared.strict_backend_selection = true;
      prepared.single_resident_island = true;
      prepared.temporal_reset_required = plan.temporal.temporal;
      prepared.host_input_is_tight_rgb24 = true;
      prepared.host_output_is_tight_rgb24 = true;
      prepared.width = plan.host_input.width;
      prepared.height = plan.host_input.height;
      prepared.output_width = plan.host_output.width;
      prepared.output_height = plan.host_output.height;
      prepared.vignette_center_x_px = static_cast<float>(prepared.width) * 0.5F;
      prepared.vignette_center_y_px =
          static_cast<float>(prepared.height) * 0.5F;
      prepared.generations = plan.invalidation_key.generations;
      prepared.capacity = plan.budget;
      prepared.runtime = runtime;
      prepared.vfx_model_directory = model_directory.string();
      prepared.effects = request.effects;
      auto canonical_effects = request.effects;
      if (!ApplyCanonicalEffects(plan.intent, &canonical_effects) ||
          canonical_effects != request.effects) {
        return fail(hal::StatusId::blocked, hal::ReasonId::malformed_contract);
      }

      const auto artifact_generation = NonzeroGeneration(
          plan, needs_vfx ? vfx_library : ar_library, model_directory);
      prepared.generations.model = artifact_generation;
      if (prepared.generations.provider == 0)
        prepared.generations.provider = artifact_generation ^ UINT64_C(0x5052);
      if (prepared.generations.device == 0)
        prepared.generations.device = artifact_generation ^ UINT64_C(0x4445);
      if (prepared.generations.provider == 0)
        prepared.generations.provider = 1;
      if (prepared.generations.device == 0)
        prepared.generations.device = 1;
      auto lease_lifetime = std::make_shared<MaxineSetupLeaseLifetime>();
      lease_lifetime->runtime = std::move(runtime_lifetime);
      if (prepared.effects.virtual_background.mode ==
          video::effects::VirtualBackgroundMode::replace) {
        int replacement_width = 0;
        int replacement_height = 0;
        std::string replacement_error;
        if (prepared.effects.virtual_background.replace_path.empty() ||
            !video::LoadImageRgb24(
                prepared.effects.virtual_background.replace_path,
                &replacement_width, &replacement_height,
                &lease_lifetime->replacement_rgb, &replacement_error) ||
            replacement_width <= 0 || replacement_height <= 0) {
          return fail(hal::StatusId::blocked,
                      hal::ReasonId::capability_evidence_incomplete);
        }
        prepared.replacement = {lease_lifetime->replacement_rgb.data(),
                                static_cast<std::uint32_t>(replacement_width),
                                static_cast<std::uint32_t>(replacement_height),
                                static_cast<std::size_t>(replacement_width) *
                                    3u,
                                artifact_generation};
      }
      for (std::size_t index = 0; index < plan.intent.stage_count; ++index) {
        const auto &canonical = plan.intent.stages[index];
        auto &stage = prepared.stages[prepared.stage_count++];
        stage.stage_id = canonical.intent.stage_id;
        stage.semantic = canonical.intent.semantic;
        stage.required = canonical.intent.required;
        stage.model = canonical.model;
        if (stage.semantic != hal::SemanticEffectId::vignette) {
          const auto &stage_library =
              IsArSemantic(stage.semantic) ? ar_library : vfx_library;
          if (!HasIdentity(stage.model.model_id_hash)) {
            stage.model.model_id_hash =
                Identity("maxine-model", stage.semantic, stage_library,
                         model_directory, artifact_generation);
          }
          if (!HasIdentity(stage.model.artifact_set_hash)) {
            stage.model.artifact_set_hash =
                Identity("maxine-artifact-set", stage.semantic, stage_library,
                         model_directory, artifact_generation);
          }
          stage.model.artifact_generation = artifact_generation;
        }
        const auto &effects = request.effects;
        const std::string *requested_model = nullptr;
        switch (stage.semantic) {
        case hal::SemanticEffectId::virtual_background_blur:
        case hal::SemanticEffectId::virtual_background_remove:
        case hal::SemanticEffectId::virtual_background_replace:
          requested_model = &effects.virtual_background.model_id;
          break;
        case hal::SemanticEffectId::video_noise_removal:
          requested_model = &effects.video_noise_removal.model_id;
          break;
        case hal::SemanticEffectId::eye_contact:
          requested_model = &effects.eye_contact.model_id;
          break;
        case hal::SemanticEffectId::auto_frame:
          requested_model = &effects.auto_frame.model_id;
          break;
        case hal::SemanticEffectId::virtual_key_light:
        case hal::SemanticEffectId::vignette:
        case hal::SemanticEffectId::mirror:
        case hal::SemanticEffectId::invalid:
          break;
        }
        if (requested_model != nullptr) {
          auto &selection =
              result.model_selections[result.model_selection_count++];
          selection.semantic = stage.semantic;
          selection.requested_model_id = *requested_model;
          selection.selected_model_id =
              requested_model->empty()
                  ? std::string("maxine-default-") +
                        std::to_string(
                            static_cast<std::uint32_t>(stage.semantic))
                  : *requested_model;
          selection.artifact = stage.model;
          selection.default_selected = requested_model->empty();
        }
      }

      MaxineProductionVideoExecutionBootstrapRequest bootstrap_request;
      bootstrap_request.prepared_request = prepared;
      bootstrap_request.factory_id = configuration_.factory_id;
      bootstrap_request.selection_priority = configuration_.selection_priority;
      bootstrap_request.minimum_synchronizations_per_frame =
          configuration_.minimum_synchronizations_per_frame;
      bootstrap_request.maximum_synchronizations_per_frame =
          configuration_.maximum_synchronizations_per_frame;
      bootstrap_request.runtime_lifetime = std::move(lease_lifetime);

      MaxineProductionVideoExecutionBootstrapResult bootstrap_result;
      hal::ReasonId reason = hal::ReasonId::none;
      const auto status =
          MaxineProductionVideoExecutionBootstrap::AcquireSetupLease(
              bootstrap_request, &result.lease, &result.lease_acquire,
              &bootstrap_result, &reason);
      result.status = status;
      result.reason = reason;
      if (status != hal::StatusId::ready && status != hal::StatusId::degraded) {
        result.readiness.status = status;
        result.readiness.reason = reason;
        return result;
      }

      const auto &snapshot = result.lease.snapshot();
      result.status = snapshot.status;
      result.reason = snapshot.reason;
      result.readiness = snapshot.readiness;
      result.canonical_effects = request.effects;
      result.plan = plan;
      result.plan.cpu_tails[0].id =
          hal::CpuTailId::output_format_conversion;
      result.plan.invalidation_key.generations = snapshot.generations;
      result.plan.invalidation_key.execution_domain_hash =
          ExecutionDomainHash(snapshot.execution_domain);
      if (result.plan.invalidation_key.model_artifact_hash == 0) {
        result.plan.invalidation_key.model_artifact_hash = artifact_generation;
      }
      for (std::size_t index = 0; index < result.plan.intent.stage_count;
           ++index) {
        result.plan.intent.stages[index].model = prepared.stages[index].model;
      }
      return result;
    } catch (const std::bad_alloc &) {
      return fail(hal::StatusId::resource_exhausted,
                  hal::ReasonId::resource_exhausted);
    } catch (...) {
      return fail(hal::StatusId::blocked,
                  hal::ReasonId::capability_evidence_incomplete);
    }
  }

private:
  MaxineProductionBackendSetupSourceConfiguration configuration_{};
};

} // namespace

std::unique_ptr<video::IProductionVideoBackendSetupSource>
CreateMaxineProductionVideoBackendSetupSource(
    MaxineProductionBackendSetupSourceConfiguration configuration) {
  try {
    return std::make_unique<MaxineProductionVideoBackendSetupSource>(
        std::move(configuration));
  } catch (...) {
    return nullptr;
  }
}

} // namespace studiocast::maxine
