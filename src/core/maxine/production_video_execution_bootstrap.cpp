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
      request.replacement.rgb, request.replacement.width,
      request.replacement.height, request.replacement.stride_bytes,
      request.replacement.generation,
  };
  return setup;
}

hal::CapabilityEvidenceRecord
Evidence(hal::CapabilityEvidenceCode code,
         hal::SemanticEffectId semantic) noexcept {
  return {
      code,
      hal::StatusId::ready,
      hal::ReasonId::none,
      semantic,
      hal::BackendId::maxine,
      true,
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
        registration.semantic_capabilities[registration
                                               .semantic_capability_count++];
    capability.semantic = semantic;
    capability.provider = ProviderFor(semantic);
    capability.status = hal::StatusId::ready;
    capability.live_stage = true;
    capability.production_eligible = true;
    capability.exact_evidence = true;
    capability.evidence[capability.evidence_count++] =
        Evidence(hal::CapabilityEvidenceCode::provider_runtime_available,
                 semantic);
    capability.evidence[capability.evidence_count++] =
        Evidence(hal::CapabilityEvidenceCode::synchronization_evidence_valid,
                 semantic);
    if (semantic != hal::SemanticEffectId::vignette) {
      capability.evidence[capability.evidence_count++] =
          Evidence(hal::CapabilityEvidenceCode::model_artifact_identity_valid,
                   semantic);
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

hal::ReasonId
EligibilityReason(
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

} // namespace

struct MaxineProductionVideoExecutionBootstrap::State {
  std::unique_ptr<MaxineProductionHalOwner> owner;
  hal::AdapterRegistration registration{};

  ~State() {
    if (owner)
      owner->Shutdown();
  }
};

std::unique_ptr<MaxineProductionVideoExecutionBootstrap>
MaxineProductionVideoExecutionBootstrap::Create(
    const MaxineProductionVideoExecutionBootstrapRequest &request,
    MaxineProductionVideoExecutionBootstrapResult *result) noexcept {
  ResetResult(result);
  const auto eligibility =
      video::EvaluatePreparedMaxineHardwareEligibility(
          request.prepared_request);
  if (eligibility != video::PreparedMaxineHardwareEligibility::eligible) {
    SetFailure(result, hal::StatusId::blocked,
               EligibilityReason(eligibility), eligibility);
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
      normalized_request.adapter_instance_id =
          NextBootstrapAdapterInstanceId();
    }
    std::uint32_t stage_mask = 0;
    for (std::size_t index = 0;
         index < normalized_request.stage_count; ++index) {
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
    if (ZeroBudget(capacity) &&
        !KnownMinimumMaxineBudget(setup, &capacity)) {
      SetFailure(result, hal::StatusId::blocked,
                 hal::ReasonId::malformed_descriptor, eligibility);
      return nullptr;
    }

    MaxineProductionBackendRequest backend_request;
    backend_request.adapter_instance_id =
        normalized_request.adapter_instance_id;
    backend_request.runtime.vfx = reinterpret_cast<vfx::VfxApi *>(
        normalized_request.runtime.vfx);
    backend_request.runtime.nvcv = reinterpret_cast<NvcvApi *>(
        normalized_request.runtime.nvcv);
    backend_request.runtime.ar = reinterpret_cast<ar::ArApi *>(
        normalized_request.runtime.ar);
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
    state->registration =
        Registration(normalized_request, *state->owner,
                     request.selection_priority);
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

    auto bootstrap =
        std::unique_ptr<MaxineProductionVideoExecutionBootstrap>(
            new MaxineProductionVideoExecutionBootstrap());
    bootstrap->state_ = std::move(state);
    bootstrap->factory_ = std::make_unique<MaxineVideoExecutionFactory>(
        std::move(factory_configuration));
    if (result) {
      result->status = hal::StatusId::ready;
      result->reason = hal::ReasonId::none;
      result->eligibility = eligibility;
      result->execution_domain =
          bootstrap->state_->owner->execution_domain();
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

} // namespace studiocast::maxine
