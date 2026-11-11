#include "core/video/production_video_backend_manager.h"
#include "core/video/production_video_setup_request.h"

#include <cstddef>
#include <utility>

namespace studiocast::video {
namespace {

void SetReason(hal::ReasonId *reason, hal::ReasonId value) noexcept {
  if (reason != nullptr)
    *reason = value;
}

bool IsSetupSuccess(hal::StatusId status) noexcept {
  return status == hal::StatusId::ready || status == hal::StatusId::degraded;
}

bool SameGenerations(const hal::ProgramGenerations &left,
                     const hal::ProgramGenerations &right) noexcept {
  return left.configuration == right.configuration &&
         left.geometry == right.geometry && left.model == right.model &&
         left.provider == right.provider && left.device == right.device &&
         left.context == right.context;
}

bool SameBudget(const hal::ResourceBudget &left,
                const hal::ResourceBudget &right) noexcept {
  return left.persistent_bytes == right.persistent_bytes &&
         left.scratch_bytes == right.scratch_bytes &&
         left.upload_staging_bytes == right.upload_staging_bytes &&
         left.readback_staging_bytes == right.readback_staging_bytes &&
         left.resource_count == right.resource_count &&
         left.in_flight_slots == right.in_flight_slots;
}

bool BudgetWithin(const hal::ResourceBudget &maximum,
                  const hal::ResourceBudget &reserved) noexcept {
  return reserved.persistent_bytes <= maximum.persistent_bytes &&
         reserved.scratch_bytes <= maximum.scratch_bytes &&
         reserved.upload_staging_bytes <= maximum.upload_staging_bytes &&
         reserved.readback_staging_bytes <= maximum.readback_staging_bytes &&
         reserved.resource_count <= maximum.resource_count &&
         maximum.in_flight_slots == 1 && reserved.in_flight_slots == 1;
}

bool SameModel(const hal::ModelArtifactIdentity &left,
               const hal::ModelArtifactIdentity &right) noexcept {
  return left.model_id_hash == right.model_id_hash &&
         left.artifact_set_hash == right.artifact_set_hash &&
         left.artifact_generation == right.artifact_generation;
}

bool SameImage(const hal::ImageDescriptor &left,
               const hal::ImageDescriptor &right) noexcept {
  if (left.width != right.width || left.height != right.height ||
      left.plane_count != right.plane_count || left.format != right.format ||
      left.channel_order != right.channel_order ||
      left.color_space != right.color_space ||
      left.color_range != right.color_range ||
      left.alpha_meaning != right.alpha_meaning ||
      left.orientation != right.orientation ||
      left.host_coherence != right.host_coherence ||
      left.mutability != right.mutability || left.aliasing != right.aliasing ||
      left.in_place_permitted != right.in_place_permitted ||
      left.base_alignment_bytes != right.base_alignment_bytes) {
    return false;
  }
  for (std::size_t index = 0; index < left.plane_count; ++index) {
    const auto &a = left.planes[index];
    const auto &b = right.planes[index];
    if (a.offset_bytes != b.offset_bytes ||
        a.row_stride_bytes != b.row_stride_bytes ||
        a.size_bytes != b.size_bytes || a.width != b.width ||
        a.height != b.height || a.alignment_bytes != b.alignment_bytes) {
      return false;
    }
  }
  return true;
}

bool SameTemporal(const hal::TemporalContract &left,
                  const hal::TemporalContract &right) noexcept {
  return left.temporal == right.temporal &&
         left.requires_contiguous_capture_sequence ==
             right.requires_contiguous_capture_sequence &&
         left.frame_drop_behavior == right.frame_drop_behavior &&
         left.discontinuity_action == right.discontinuity_action &&
         left.history_frames == right.history_frames;
}

bool SameParameter(const CanonicalVideoParameter &left,
                   const CanonicalVideoParameter &right) noexcept {
  return left.id == right.id && left.kind == right.kind &&
         left.signed_value == right.signed_value &&
         left.unsigned_value == right.unsigned_value &&
         left.real_value == right.real_value &&
         left.boolean_value == right.boolean_value;
}

bool IsZeroHash(const std::array<std::uint8_t, 32> &hash) noexcept {
  for (const auto byte : hash) {
    if (byte != 0)
      return false;
  }
  return true;
}

bool CompatibleModel(const hal::ModelArtifactIdentity &caller,
                     const hal::ModelArtifactIdentity &bound) noexcept {
  const bool caller_model_unbound = IsZeroHash(caller.model_id_hash);
  const bool caller_artifact_unbound = IsZeroHash(caller.artifact_set_hash);
  if (caller_model_unbound && caller_artifact_unbound &&
      caller.artifact_generation == 0) {
    return true;
  }
  if (!caller_model_unbound && caller.model_id_hash != bound.model_id_hash)
    return false;
  if (!caller_artifact_unbound &&
      caller.artifact_set_hash != bound.artifact_set_hash) {
    return false;
  }
  if (caller.artifact_generation != 0 &&
      caller.artifact_generation != bound.artifact_generation) {
    return false;
  }
  // An explicit artifact identity is all-or-nothing. A model-id-only selector
  // is allowed and is bound to the source's exact artifact/generation.
  if ((!caller_artifact_unbound || caller.artifact_generation != 0) &&
      (caller_model_unbound || caller_artifact_unbound ||
       caller.artifact_generation == 0)) {
    return false;
  }
  return true;
}

bool SameCanonicalIntent(const NormalizedVideoExecutionIntent &caller,
                         const NormalizedVideoExecutionIntent &bound) noexcept {
  if (caller.backend.mode != bound.backend.mode ||
      caller.backend.backend != bound.backend.backend ||
      caller.backend.strict != bound.backend.strict ||
      caller.preferences.compute_family != bound.preferences.compute_family ||
      caller.preferences.effects_engine != bound.preferences.effects_engine ||
      caller.preferences.strict_effects_engine !=
          bound.preferences.strict_effects_engine ||
      caller.requested_provider != bound.requested_provider ||
      caller.strict_provider_selection != bound.strict_provider_selection ||
      (caller.stage_count != 0 && caller.stage_count != bound.stage_count)) {
    return false;
  }
  if (caller.stage_count == 0)
    return true;
  for (std::size_t index = 0; index < caller.stage_count; ++index) {
    const auto &a = caller.stages[index];
    const auto &b = bound.stages[index];
    if (a.intent.stage_id != b.intent.stage_id ||
        a.intent.semantic != b.intent.semantic ||
        a.intent.required != b.intent.required ||
        a.intent.failure_action != b.intent.failure_action ||
        a.intent.requires_tracked_vignette_center !=
            b.intent.requires_tracked_vignette_center ||
        a.parameter_count != b.parameter_count ||
        !CompatibleModel(a.model, b.model)) {
      return false;
    }
    for (std::size_t parameter = 0; parameter < a.parameter_count;
         ++parameter) {
      if (!SameParameter(a.parameters[parameter], b.parameters[parameter]))
        return false;
    }
  }
  return true;
}

bool SameTail(const hal::DeclaredCpuTail &left,
              const hal::DeclaredCpuTail &right) noexcept {
  return left.id == right.id && left.after_stage_id == right.after_stage_id &&
         left.scratch_bytes == right.scratch_bytes &&
         left.required == right.required && left.placement == right.placement &&
         left.before_stage_id == right.before_stage_id &&
         left.boundary_index == right.boundary_index;
}

bool IsTerminalBackendResolvedOutputTail(
    const VideoExecutionPlanRequest &plan) noexcept {
  if (plan.cpu_tail_count != 1 || plan.intent.stage_count == 0 ||
      plan.intent.stage_count > plan.intent.stages.size())
    return false;
  const auto &tail = plan.cpu_tails[0];
  const auto final_stage_id =
      plan.intent.stages[plan.intent.stage_count - 1].intent.stage_id;
  if (final_stage_id == 0)
    return false;
  return tail.id == hal::CpuTailId::backend_resolved_output &&
         tail.placement == hal::DeclaredCpuTail::Placement::after_stage &&
         tail.after_stage_id == final_stage_id &&
         tail.before_stage_id == 0 &&
         tail.boundary_index == plan.intent.stage_count;
}

bool CompatibleCpuTails(const VideoExecutionPlanRequest &caller,
                        const VideoExecutionPlanRequest &bound) noexcept {
  if (!IsTerminalBackendResolvedOutputTail(caller)) {
    if (caller.cpu_tail_count != bound.cpu_tail_count)
      return false;
    for (std::size_t index = 0; index < caller.cpu_tail_count; ++index) {
      if (!SameTail(caller.cpu_tails[index], bound.cpu_tails[index]))
        return false;
    }
    return true;
  }
  if (bound.cpu_tail_count == 0)
    return true;
  if (bound.cpu_tail_count != 1)
    return false;
  const auto &bound_tail = bound.cpu_tails[0];
  if (bound_tail.id != hal::CpuTailId::writer_copy &&
      bound_tail.id != hal::CpuTailId::output_format_conversion &&
      bound_tail.id != hal::CpuTailId::provider_postprocess) {
    return false;
  }
  if (!bound_tail.required ||
      bound_tail.placement != hal::DeclaredCpuTail::Placement::after_stage ||
      bound_tail.before_stage_id != 0 || bound_tail.boundary_index == 0 ||
      bound_tail.boundary_index > caller.intent.stage_count)
    return false;
  const auto selected_index = bound_tail.boundary_index - 1;
  return bound_tail.after_stage_id ==
             caller.intent.stages[selected_index].intent.stage_id &&
         bound_tail.scratch_bytes == caller.cpu_tails[0].scratch_bytes;
}

bool ValidWaitPolicy(const ProductionVideoBoundedWaitPolicy &wait) noexcept {
  return wait.contract_version ==
             kProductionVideoBackendManagerContractVersion &&
         wait.timeout_nanoseconds != 0 &&
         wait.poll_interval_nanoseconds != 0 &&
         wait.poll_interval_nanoseconds <= wait.timeout_nanoseconds &&
         wait.maximum_poll_count != 0 && wait.stop_requested != nullptr &&
         wait.now_nanoseconds != nullptr && wait.poll_wait != nullptr &&
         wait.callback_context != nullptr &&
         wait.callback_context_generation != 0 &&
         wait.callback_context_lifetime ==
             hal::ResourceLifetime::adapter_context;
}

const std::string *ConfiguredModelId(
    const effects::BroadcastCameraEffects &effects,
    hal::SemanticEffectId semantic) noexcept {
  switch (semantic) {
  case hal::SemanticEffectId::virtual_background_blur:
  case hal::SemanticEffectId::virtual_background_remove:
  case hal::SemanticEffectId::virtual_background_replace:
    return &effects.virtual_background.model_id;
  case hal::SemanticEffectId::auto_frame:
    return &effects.auto_frame.model_id;
  case hal::SemanticEffectId::eye_contact:
    return &effects.eye_contact.model_id;
  case hal::SemanticEffectId::video_noise_removal:
    return &effects.video_noise_removal.model_id;
  case hal::SemanticEffectId::virtual_key_light:
  case hal::SemanticEffectId::vignette:
  case hal::SemanticEffectId::mirror:
  case hal::SemanticEffectId::invalid:
    return nullptr;
  }
  return nullptr;
}

bool StageEnabledByEffects(const effects::BroadcastCameraEffects &effects,
                           hal::SemanticEffectId semantic) noexcept {
  switch (semantic) {
  case hal::SemanticEffectId::video_noise_removal:
    return effects.video_noise_removal.enabled;
  case hal::SemanticEffectId::eye_contact:
    return effects.eye_contact.enabled;
  case hal::SemanticEffectId::virtual_background_blur:
    return effects.virtual_background.mode ==
           effects::VirtualBackgroundMode::blur;
  case hal::SemanticEffectId::virtual_background_remove:
    return effects.virtual_background.mode ==
           effects::VirtualBackgroundMode::remove;
  case hal::SemanticEffectId::virtual_background_replace:
    return effects.virtual_background.mode ==
           effects::VirtualBackgroundMode::replace;
  case hal::SemanticEffectId::virtual_key_light:
    return effects.virtual_key_light.enabled;
  case hal::SemanticEffectId::auto_frame:
    return effects.auto_frame.enabled;
  case hal::SemanticEffectId::vignette:
    return effects.vignette.enabled;
  case hal::SemanticEffectId::mirror:
    return effects.mirror;
  case hal::SemanticEffectId::invalid:
    return false;
  }
  return false;
}

const CanonicalVideoParameter *
FindParameter(const CanonicalVideoStage &stage,
              CanonicalVideoParameterId id) noexcept {
  for (std::size_t index = 0; index < stage.parameter_count; ++index) {
    if (stage.parameters[index].id == id)
      return &stage.parameters[index];
  }
  return nullptr;
}

bool SignedParameter(const CanonicalVideoStage &stage,
                     CanonicalVideoParameterId id, int value) noexcept {
  const auto *parameter = FindParameter(stage, id);
  return parameter != nullptr &&
         parameter->kind == CanonicalVideoParameterKind::signed_integer &&
         parameter->signed_value == value;
}

bool BooleanParameter(const CanonicalVideoStage &stage,
                      CanonicalVideoParameterId id, bool value) noexcept {
  const auto *parameter = FindParameter(stage, id);
  return parameter != nullptr &&
         parameter->kind == CanonicalVideoParameterKind::boolean &&
         parameter->boolean_value == value;
}

bool UnsignedParameter(const CanonicalVideoStage &stage,
                       CanonicalVideoParameterId id,
                       std::uint64_t value) noexcept {
  const auto *parameter = FindParameter(stage, id);
  return parameter != nullptr &&
         parameter->kind == CanonicalVideoParameterKind::unsigned_integer &&
         parameter->unsigned_value == value;
}

bool RealParameter(const CanonicalVideoStage &stage,
                   CanonicalVideoParameterId id, double value) noexcept {
  const auto *parameter = FindParameter(stage, id);
  return parameter != nullptr &&
         parameter->kind == CanonicalVideoParameterKind::real &&
         parameter->real_value == value;
}

bool ParseRgb(const std::string &value, std::uint64_t *rgb) noexcept {
  if (rgb == nullptr || value.size() != 7 || value[0] != '#')
    return false;
  std::uint64_t result = 0;
  for (std::size_t index = 1; index < value.size(); ++index) {
    const char character = value[index];
    std::uint64_t nibble = 0;
    if (character >= '0' && character <= '9')
      nibble = static_cast<std::uint64_t>(character - '0');
    else if (character >= 'a' && character <= 'f')
      nibble = static_cast<std::uint64_t>(character - 'a' + 10);
    else if (character >= 'A' && character <= 'F')
      nibble = static_cast<std::uint64_t>(character - 'A' + 10);
    else
      return false;
    result = (result << 4u) | nibble;
  }
  *rgb = result;
  return true;
}

bool StageParametersMatchEffects(
    const CanonicalVideoStage &stage,
    const effects::BroadcastCameraEffects &effects) noexcept {
  if (!StageEnabledByEffects(effects, stage.intent.semantic))
    return false;
  switch (stage.intent.semantic) {
  case hal::SemanticEffectId::virtual_background_blur:
  case hal::SemanticEffectId::virtual_background_remove:
  case hal::SemanticEffectId::virtual_background_replace: {
    std::uint64_t rgb = 0;
    return ParseRgb(effects.virtual_background.remove_color, &rgb) &&
           SignedParameter(stage, CanonicalVideoParameterId::strength,
                           effects.virtual_background.strength) &&
           UnsignedParameter(stage,
                             CanonicalVideoParameterId::remove_color_rgb,
                             rgb) &&
           UnsignedParameter(stage,
                             CanonicalVideoParameterId::greenscreen_mode,
                             effects.virtual_background.greenscreen_mode) &&
           BooleanParameter(stage,
                            CanonicalVideoParameterId::greenscreen_temporal,
                            effects.virtual_background.greenscreen_temporal);
  }
  case hal::SemanticEffectId::auto_frame:
    return SignedParameter(stage, CanonicalVideoParameterId::strength,
                           effects.auto_frame.strength) &&
           SignedParameter(stage, CanonicalVideoParameterId::smoothing,
                           effects.auto_frame.smoothing) &&
           RealParameter(stage, CanonicalVideoParameterId::headroom,
                         effects.auto_frame.headroom);
  case hal::SemanticEffectId::eye_contact:
    return SignedParameter(stage, CanonicalVideoParameterId::strength,
                           effects.eye_contact.strength) &&
           BooleanParameter(stage,
                            CanonicalVideoParameterId::look_away_enabled,
                            effects.eye_contact.look_away_enabled);
  case hal::SemanticEffectId::video_noise_removal:
    return SignedParameter(stage, CanonicalVideoParameterId::strength,
                           effects.video_noise_removal.strength);
  case hal::SemanticEffectId::virtual_key_light:
    return SignedParameter(stage, CanonicalVideoParameterId::intensity,
                           effects.virtual_key_light.intensity) &&
           SignedParameter(
               stage, CanonicalVideoParameterId::temperature_preset,
               effects.virtual_key_light.temperature_preset) &&
           SignedParameter(
               stage, CanonicalVideoParameterId::direction_pan_degrees,
               effects.virtual_key_light.direction_pan_degrees);
  case hal::SemanticEffectId::vignette:
    return SignedParameter(stage, CanonicalVideoParameterId::intensity,
                           effects.vignette.intensity) &&
           BooleanParameter(
               stage, CanonicalVideoParameterId::center_on_tracked_face,
               effects.vignette.center_on_tracked_face);
  case hal::SemanticEffectId::mirror:
    return stage.parameter_count == 0;
  case hal::SemanticEffectId::invalid:
    return false;
  }
  return false;
}

bool IsZeroModel(const hal::ModelArtifactIdentity &model) noexcept;

bool StageSelectable(const hal::AdapterRegistration &registration,
                     hal::SemanticEffectId semantic) noexcept {
  for (std::size_t index = 0;
       index < registration.semantic_capability_count; ++index) {
    if (registration.semantic_capabilities[index].semantic == semantic &&
        hal::IsSelectableSemanticCapability(
            registration.semantic_capabilities[index]))
      return true;
  }
  return false;
}

bool SameResolutionRegistration(const hal::AdapterRegistration &left,
                                const hal::AdapterRegistration &right) noexcept {
  if (left.adapter_instance_id != right.adapter_instance_id ||
      left.backend != right.backend ||
      left.execution_domain != right.execution_domain ||
      left.semantic_capability_count != right.semantic_capability_count)
    return false;
  for (std::size_t index = 0; index < left.semantic_capability_count; ++index) {
    const auto &a = left.semantic_capabilities[index];
    const auto &b = right.semantic_capabilities[index];
    if (a.semantic != b.semantic || a.provider != b.provider ||
        a.status != b.status || a.reason != b.reason ||
        a.live_stage != b.live_stage ||
        a.production_eligible != b.production_eligible ||
        a.exact_evidence != b.exact_evidence || a.cpu_tail != b.cpu_tail ||
        a.cpu_tail_placement != b.cpu_tail_placement)
      return false;
  }
  return true;
}

bool ValidModelSelections(
    const ProductionVideoResidentSetupSourceResult &candidate,
    const effects::BroadcastCameraEffects &effects) noexcept {
  if (candidate.model_selection_count > candidate.model_selections.size())
    return false;
  for (std::size_t index = 0; index < candidate.model_selection_count;
       ++index) {
    const auto &selection = candidate.model_selections[index];
    const auto *configured = ConfiguredModelId(effects, selection.semantic);
    if (configured == nullptr || selection.selected_model_id.empty() ||
        IsZeroModel(selection.artifact))
      return false;
    if ((!configured->empty() &&
         (selection.requested_model_id != *configured ||
          selection.selected_model_id != *configured ||
          selection.default_selected)) ||
        (configured->empty() &&
         (!selection.requested_model_id.empty() ||
          !selection.default_selected))) {
      return false;
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (candidate.model_selections[previous].semantic == selection.semantic)
        return false;
    }
  }
  for (std::size_t stage_index = 0;
       stage_index < candidate.plan.intent.stage_count; ++stage_index) {
    const auto &stage = candidate.plan.intent.stages[stage_index];
    const auto *configured = ConfiguredModelId(effects, stage.intent.semantic);
    if (configured == nullptr ||
        !StageSelectable(candidate.resolution_registration,
                         stage.intent.semantic))
      continue;
    bool found = false;
    for (std::size_t selection_index = 0;
         selection_index < candidate.model_selection_count;
         ++selection_index) {
      const auto &selection =
          candidate.model_selections[selection_index];
      found = found ||
              (selection.semantic == stage.intent.semantic &&
               SameModel(selection.artifact, stage.model));
    }
    if (!found)
      return false;
  }
  return true;
}

bool ProviderMatchesCompatibilityDomain(
    const ProductionVideoResidentSetupSourceResult::CompatibilityDomain
        &entry) noexcept {
  if (entry.backend != entry.domain.backend)
    return false;
  const auto residency = hal::ClassifyExecutionResidency(entry.domain);
  switch (entry.provider) {
  case hal::SemanticEffectProviderId::open_video_onnxruntime_cuda:
  case hal::SemanticEffectProviderId::open_cuda_production_kernels:
    return entry.backend == hal::BackendId::cuda &&
           residency == hal::ExecutionResidencyClass::device;
  case hal::SemanticEffectProviderId::open_video_onnxruntime_cpu:
    return entry.backend == hal::BackendId::open_video_onnxruntime &&
           residency == hal::ExecutionResidencyClass::host;
  case hal::SemanticEffectProviderId::maxine_vfx:
  case hal::SemanticEffectProviderId::maxine_ar:
  case hal::SemanticEffectProviderId::maxine_cuda_utility:
    return entry.backend == hal::BackendId::maxine &&
           residency == hal::ExecutionResidencyClass::device;
  case hal::SemanticEffectProviderId::open_vulkan_production_kernels:
    return entry.backend == hal::BackendId::vulkan &&
           residency == hal::ExecutionResidencyClass::device;
  case hal::SemanticEffectProviderId::test_fake:
    return entry.backend == hal::BackendId::test_fake;
  case hal::SemanticEffectProviderId::invalid:
    return false;
  }
  return false;
}

bool ValidCompatibilityTopology(
    const ProductionVideoResidentSetupSourceResult::
        CompatibilityExecutionTopology &topology) noexcept {
  if (topology.contract_version !=
          kProductionVideoBackendManagerContractVersion ||
      topology.domain_count == 0 ||
      topology.domain_count > topology.domains.size() ||
      topology.boundary_count > topology.boundaries.size() ||
      !topology.exact_ordered_boundaries ||
      !topology.cross_adapter_residency_forbidden) {
    return false;
  }
  if (topology.domain_count == 1) {
    const auto &domain = topology.domains[0];
    return topology.boundary_count == 0 &&
           hal::ValidateExecutionDomain(domain.domain) &&
           ProviderMatchesCompatibilityDomain(domain) &&
           domain.first_stage_id != 0 &&
           domain.last_stage_id >= domain.first_stage_id &&
           topology.expected_transfers.intermediate_readbacks == 0 &&
           topology.expected_transfers.intermediate_uploads == 0 &&
           topology.expected_transfers.cpu_tail_boundaries == 0;
  }
  if (topology.boundary_count != topology.domain_count - 1)
    return false;
  std::uint64_t readbacks = 0;
  std::uint64_t uploads = 0;
  for (std::size_t index = 0; index < topology.domain_count; ++index) {
    const auto &domain = topology.domains[index];
    if (!hal::ValidateExecutionDomain(domain.domain) ||
        !ProviderMatchesCompatibilityDomain(domain) ||
        domain.first_stage_id == 0 ||
        domain.last_stage_id < domain.first_stage_id ||
        (index != 0 &&
         topology.domains[index - 1].last_stage_id >=
             domain.first_stage_id)) {
      return false;
    }
    if (index == 0)
      continue;
    const auto &previous = topology.domains[index - 1];
    const auto &boundary = topology.boundaries[index - 1];
    if (boundary.kind != hal::ExecutionBoundaryKind::adapter_transfer ||
        boundary.after_stage_id != previous.last_stage_id ||
        boundary.before_stage_id != domain.first_stage_id ||
        boundary.boundary_index != domain.first_stage_id - 1 ||
        boundary.source_domain != previous.domain ||
        boundary.destination_domain != domain.domain ||
        boundary.source_domain == boundary.destination_domain ||
        !boundary.readback_required ||
        !boundary.following_upload_required) {
      return false;
    }
    ++readbacks;
    ++uploads;
  }
  return topology.expected_transfers.intermediate_readbacks == readbacks &&
         topology.expected_transfers.intermediate_uploads == uploads &&
         topology.expected_transfers.cpu_tail_boundaries == 0;
}

bool CompatibleGenerations(const hal::ProgramGenerations &caller,
                           const hal::ProgramGenerations &bound) noexcept {
  const auto compatible = [](std::uint64_t requested,
                             std::uint64_t selected) {
    return selected != 0 && (requested == 0 || requested == selected);
  };
  return caller.configuration != 0 &&
         caller.configuration == bound.configuration &&
         caller.geometry != 0 && caller.geometry == bound.geometry &&
         compatible(caller.model, bound.model) &&
         compatible(caller.provider, bound.provider) &&
         compatible(caller.device, bound.device) &&
         compatible(caller.context, bound.context);
}

bool SameCanonicalPlanRequest(const ProductionVideoResidentSetup &request,
                              const VideoExecutionPlanRequest &bound,
                              hal::BackendId selected_backend) noexcept {
  const auto &caller = request.plan;
  if (caller.contract_version != bound.contract_version ||
      caller.plan_id != bound.plan_id ||
      !SameCanonicalIntent(caller.intent, bound.intent) ||
      !SameImage(caller.host_input, bound.host_input) ||
      !SameImage(caller.host_output, bound.host_output) ||
      !CompatibleGenerations(caller.invalidation_key.generations,
                             bound.invalidation_key.generations) ||
      caller.invalidation_key.canonical_plan_hash !=
          bound.invalidation_key.canonical_plan_hash ||
      (caller.invalidation_key.model_artifact_hash != 0 &&
       caller.invalidation_key.model_artifact_hash !=
           bound.invalidation_key.model_artifact_hash) ||
      caller.invalidation_key.execution_domain_hash != 0 ||
      bound.invalidation_key.execution_domain_hash == 0 ||
      !SameTemporal(ProductionVideoTemporalForSelectedBackend(
                        request, selected_backend),
                    bound.temporal) ||
      !BudgetWithin(caller.budget, bound.budget) ||
      !CompatibleCpuTails(caller, bound) ||
      caller.setup_failure_policy != bound.setup_failure_policy ||
      caller.migration_policy != bound.migration_policy) {
    return false;
  }
  return true;
}

bool SamePassthroughPlanRequest(
    const VideoExecutionPlanRequest &caller,
    const VideoExecutionPlanRequest &bound) noexcept {
  if (caller.contract_version != bound.contract_version ||
      caller.plan_id != bound.plan_id ||
      !SameCanonicalIntent(caller.intent, bound.intent) ||
      !SameImage(caller.host_input, bound.host_input) ||
      !SameImage(caller.host_output, bound.host_output) ||
      caller.invalidation_key.generations !=
          bound.invalidation_key.generations ||
      caller.invalidation_key.canonical_plan_hash !=
          bound.invalidation_key.canonical_plan_hash ||
      caller.invalidation_key.model_artifact_hash !=
          bound.invalidation_key.model_artifact_hash ||
      caller.invalidation_key.execution_domain_hash != 0 ||
      bound.invalidation_key.execution_domain_hash !=
          kProductionVideoNeutralPassthroughExecutionHash ||
      !SameTemporal(caller.temporal, bound.temporal) ||
      !SameBudget(caller.budget, bound.budget) ||
      caller.cpu_tail_count != bound.cpu_tail_count ||
      caller.setup_failure_policy != bound.setup_failure_policy ||
      caller.migration_policy != bound.migration_policy) {
    return false;
  }
  for (std::size_t index = 0; index < caller.cpu_tail_count; ++index) {
    if (!SameTail(caller.cpu_tails[index], bound.cpu_tails[index]))
      return false;
  }
  return true;
}

bool IsZeroModel(const hal::ModelArtifactIdentity &model) noexcept {
  if (model.artifact_generation != 0)
    return false;
  for (const auto byte : model.model_id_hash) {
    if (byte != 0)
      return false;
  }
  for (const auto byte : model.artifact_set_hash) {
    if (byte != 0)
      return false;
  }
  return true;
}

bool ContainsProvider(const ProductionVideoSetupLeaseSnapshot &snapshot,
                      hal::SemanticEffectProviderId provider) noexcept {
  for (std::size_t index = 0; index < snapshot.provider_count; ++index) {
    if (snapshot.providers[index] == provider)
      return true;
  }
  return false;
}

hal::SemanticEffectProviderId ProviderForStage(
    const ProductionVideoSetupLeaseSnapshot &snapshot,
    hal::SemanticEffectId semantic) noexcept {
  hal::SemanticEffectProviderId provider =
      hal::SemanticEffectProviderId::invalid;
  for (std::size_t index = 0;
       index < snapshot.adapter.semantic_capability_count; ++index) {
    const auto &capability = snapshot.adapter.semantic_capabilities[index];
    if (capability.semantic != semantic ||
        !hal::IsSelectableSemanticCapability(capability)) {
      continue;
    }
    if (provider != hal::SemanticEffectProviderId::invalid &&
        provider != capability.provider) {
      return hal::SemanticEffectProviderId::invalid;
    }
    provider = capability.provider;
  }
  return provider;
}

bool HasExactArtifact(const ProductionVideoSetupLeaseSnapshot &snapshot,
                      hal::SemanticEffectId semantic,
                      hal::SemanticEffectProviderId provider,
                      const hal::ModelArtifactIdentity &model) noexcept {
  for (std::size_t index = 0; index < snapshot.artifact_count; ++index) {
    const auto &artifact = snapshot.artifacts[index];
    if (artifact.semantic == semantic && artifact.provider == provider &&
        SameModel(artifact.artifact, model)) {
      return true;
    }
  }
  return false;
}

hal::StatusId Fail(hal::StatusId status, hal::ReasonId failure,
                   bool retained, ProductionVideoBackendSetupEvidence *evidence,
                   hal::ReasonId *reason) noexcept {
  SetReason(reason, failure);
  if (evidence != nullptr) {
    evidence->status = status;
    evidence->reason = failure;
    evidence->candidate_rejected = true;
    evidence->old_plan_retained = retained;
  }
  return status;
}

bool ValidateCpuPrimitiveTable(const ProductionVideoCpuPrimitiveSetup &setup,
                               hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  const auto *table = setup.table;
  if (setup.contract_version !=
          kProductionVideoBackendManagerContractVersion ||
      table == nullptr || table->table_version != kCpuPrimitiveTableVersion ||
      table->struct_size != sizeof(CpuPrimitiveTable) ||
      table->backend != hal::BackendId::cpu ||
      table->semantic_effect_provider !=
          hal::SemanticEffectProviderId::invalid ||
      (setup.required_primitive_bits & ~table->capability_bits) != 0) {
    SetReason(reason, hal::ReasonId::malformed_contract);
    return false;
  }

  const auto conversion =
      CpuPrimitiveCapabilityBit(hal::PrimitiveOperationId::convert_pixel_format);
  const auto resize =
      CpuPrimitiveCapabilityBit(hal::PrimitiveOperationId::resize);
  const auto mirror =
      CpuPrimitiveCapabilityBit(hal::PrimitiveOperationId::mirror);
  if (((table->capability_bits & conversion) != 0 &&
       (table->yuyv_to_rgb24 == nullptr ||
        table->rgb24_to_yuyv_scratch_bytes == nullptr ||
        table->rgb24_to_yuyv == nullptr ||
        table->rgb24_bgr24 == nullptr)) ||
      ((table->capability_bits & resize) != 0 &&
       (table->resize_rgb24_bilinear_plan_scratch_bytes == nullptr ||
        table->prepare_resize_rgb24_bilinear == nullptr ||
        table->apply_resize_rgb24_bilinear == nullptr)) ||
      ((table->capability_bits & mirror) != 0 &&
       table->mirror_rgb24_in_place == nullptr)) {
    SetReason(reason, hal::ReasonId::capability_evidence_incomplete);
    return false;
  }
  return true;
}

hal::StatusId ValidateResidentCandidate(
    const VideoExecutionPlanRequest &plan,
    const ProductionVideoSetupLeaseSnapshot &candidate,
    hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  hal::ReasonId plan_reason = hal::ReasonId::none;
  if (!ValidateVideoExecutionPlanRequest(plan, &plan_reason)) {
    SetReason(reason, plan_reason);
    return plan_reason == hal::ReasonId::wrong_generation
               ? hal::StatusId::stale
               : hal::StatusId::blocked;
  }
  if (plan.invalidation_key.execution_domain_hash !=
      ProductionVideoExecutionDomainHash(candidate.execution_domain)) {
    SetReason(reason, hal::ReasonId::wrong_execution_domain);
    return hal::StatusId::blocked;
  }
  if (!SameGenerations(plan.invalidation_key.generations,
                       candidate.generations)) {
    SetReason(reason, hal::ReasonId::wrong_generation);
    return hal::StatusId::stale;
  }
  if (!SameBudget(plan.budget, candidate.reserved_budget)) {
    SetReason(reason, hal::ReasonId::resource_exhausted);
    return hal::StatusId::resource_exhausted;
  }

  const auto &backend = plan.intent.backend;
  if (backend.mode == hal::BackendRequestMode::explicit_backend &&
      backend.backend != candidate.backend) {
    SetReason(reason, backend.strict ? hal::ReasonId::strict_backend_unavailable
                                    : hal::ReasonId::foreign_adapter);
    return hal::StatusId::blocked;
  }
  const auto requested_provider = plan.intent.requested_provider;
  if (plan.intent.strict_provider_selection &&
      !ContainsProvider(candidate, requested_provider)) {
    SetReason(reason, hal::ReasonId::semantic_provider_unavailable);
    return hal::StatusId::blocked;
  }

  hal::StaticAdapterRegistry registry;
  hal::ReasonId registration_reason = hal::ReasonId::none;
  if (registry.Register(candidate.adapter, &registration_reason) !=
      hal::StatusId::ready) {
    SetReason(reason, registration_reason);
    return hal::StatusId::blocked;
  }
  const auto resolution = hal::ResolveSemanticStages(
      registry, BuildVideoResolutionRequest(plan.intent));
  if (!resolution.available || resolution.passthrough ||
      resolution.stage_count != plan.intent.stage_count) {
    SetReason(reason, resolution.reason == hal::ReasonId::none
                          ? hal::ReasonId::capability_evidence_incomplete
                          : resolution.reason);
    return hal::StatusId::blocked;
  }

  for (std::size_t index = 0; index < plan.intent.stage_count;
       ++index) {
    const auto &stage = plan.intent.stages[index];
    if (resolution.stages[index].skipped_optional) {
      if (stage.intent.required) {
        SetReason(reason, hal::ReasonId::required_stage_failed);
        return hal::StatusId::blocked;
      }
      continue;
    }
    const auto provider = ProviderForStage(candidate, stage.intent.semantic);
    if (provider == hal::SemanticEffectProviderId::invalid ||
        (plan.intent.strict_provider_selection &&
         provider != requested_provider)) {
      SetReason(reason, hal::ReasonId::semantic_provider_unavailable);
      return hal::StatusId::blocked;
    }
    if (!IsZeroModel(stage.model) &&
        !HasExactArtifact(candidate, stage.intent.semantic, provider,
                          stage.model)) {
      SetReason(reason,
                stage.model.artifact_generation != candidate.generations.model
                    ? hal::ReasonId::wrong_generation
                    : hal::ReasonId::capability_evidence_incomplete);
      return stage.model.artifact_generation != candidate.generations.model
                 ? hal::StatusId::stale
                 : hal::StatusId::blocked;
    }
  }
  return hal::StatusId::ready;
}

ProductionVideoSetupLeaseShutdownResult ImmediateShutdown() noexcept {
  ProductionVideoSetupLeaseShutdownResult result;
  result.status = hal::StatusId::stopped;
  result.reason = hal::ReasonId::shutdown_requested;
  result.shutdown_requested = true;
  result.quiescent = true;
  result.driver_context_release_permitted = true;
  return result;
}

} // namespace

ProductionVideoBackendManager::ProductionVideoBackendManager(
    std::unique_ptr<IProductionVideoBackendSetupSource> setup_source)
    : setup_source_(std::move(setup_source)) {}

ProductionVideoBackendManager::~ProductionVideoBackendManager() = default;

bool ProductionVideoBackendManager::HasRetainableSetup() const noexcept {
  return snapshot_.setup_complete &&
         (snapshot_.state ==
              ProductionVideoBackendManagerState::cpu_primitives_only ||
          snapshot_.resident_prepared ||
          snapshot_.state ==
              ProductionVideoBackendManagerState::compatibility_ready);
}

void ProductionVideoBackendManager::RefreshResidentSnapshot() noexcept {
  snapshot_.resident = composition_.Snapshot();
  snapshot_.resident_prepared = snapshot_.resident.prepared;
  snapshot_.frame_pending = snapshot_.resident.frame_pending;
  if (resident_selected_) {
    snapshot_.readiness = snapshot_.resident.readiness;
    snapshot_.status = snapshot_.resident.readiness.status;
    snapshot_.reason = snapshot_.resident.readiness.reason;
  }
}

hal::StatusId ProductionVideoBackendManager::Setup(
    const ProductionVideoBackendSetupRequest &request,
    ProductionVideoBackendSetupEvidence *evidence,
    hal::ReasonId *reason) noexcept {
  if (evidence != nullptr)
    *evidence = {};
  SetReason(reason, hal::ReasonId::none);
  ++snapshot_.setup_attempts;
  const bool retained = HasRetainableSetup();
  if (snapshot_.state == ProductionVideoBackendManagerState::stopped ||
      snapshot_.state ==
          ProductionVideoBackendManagerState::shutdown_pending) {
    return Fail(snapshot_.state ==
                        ProductionVideoBackendManagerState::shutdown_pending
                    ? snapshot_.shutdown.status
                    : hal::StatusId::stopped,
                snapshot_.state ==
                        ProductionVideoBackendManagerState::shutdown_pending
                    ? snapshot_.shutdown.reason
                    : hal::ReasonId::shutdown_requested,
                retained, evidence, reason);
  }
  if (request.contract_version !=
      kProductionVideoBackendManagerContractVersion) {
    return Fail(hal::StatusId::blocked,
                hal::ReasonId::unsupported_contract_version, retained,
                evidence, reason);
  }

  if (request.kind == ProductionVideoBackendSetupKind::cpu_primitives_only) {
    hal::ReasonId validation_reason = hal::ReasonId::none;
    if (!ValidateCpuPrimitiveTable(request.cpu, &validation_reason)) {
      return Fail(hal::StatusId::blocked, validation_reason, retained, evidence,
                  reason);
    }
    if (request.resident.plan.plan_id != 0 ||
        request.resident.plan.invalidation_key.execution_domain_hash != 0) {
      return Fail(hal::StatusId::blocked, hal::ReasonId::malformed_contract,
                  retained, evidence, reason);
    }
    if (snapshot_.resident_prepared)
      composition_.Invalidate(hal::ReasonId::reconfiguration_required);
    if (compatibility_executor_) {
      compatibility_executor_->Shutdown();
      compatibility_executor_.reset();
    }
    resident_selected_ = false;
    snapshot_.state =
        ProductionVideoBackendManagerState::cpu_primitives_only;
    snapshot_.status = hal::StatusId::ready;
    snapshot_.reason = hal::ReasonId::none;
    snapshot_.readiness = {};
    snapshot_.readiness.status = hal::StatusId::ready;
    snapshot_.readiness.requested_backend = hal::BackendId::cpu;
    snapshot_.readiness.selected_backend = hal::BackendId::cpu;
    snapshot_.readiness.available = true;
    snapshot_.cpu_primitive_bits = request.cpu.table->capability_bits;
    snapshot_.setup_complete = true;
    snapshot_.resident_prepared = false;
    snapshot_.frame_pending = false;
    ++snapshot_.setup_successes;
    if (evidence != nullptr) {
      evidence->status = hal::StatusId::ready;
      evidence->cpu_primitive_only = true;
    }
    return hal::StatusId::ready;
  }

  if (request.kind != ProductionVideoBackendSetupKind::resident_island) {
    return Fail(hal::StatusId::blocked, hal::ReasonId::malformed_contract,
                retained, evidence, reason);
  }
  hal::ReasonId plan_reason = hal::ReasonId::none;
  auto validation_plan = request.resident.plan;
  if (IsTerminalBackendResolvedOutputTail(validation_plan))
    validation_plan.cpu_tails[0].id = hal::CpuTailId::writer_copy;
  validation_plan.invalidation_key.execution_domain_hash = 1;
  auto &validation_generations = validation_plan.invalidation_key.generations;
  if (validation_generations.model == 0)
    validation_generations.model = 1;
  if (validation_generations.provider == 0)
    validation_generations.provider = 1;
  if (validation_generations.device == 0)
    validation_generations.device = 1;
  if (validation_generations.context == 0)
    validation_generations.context = 1;
  if (request.resident.contract_version !=
          kProductionVideoBackendManagerContractVersion ||
      request.resident.effects.schema_version !=
          effects::kBroadcastEffectsSchemaVersion ||
      !ValidWaitPolicy(request.resident.wait) ||
      request.resident.plan.invalidation_key.execution_domain_hash != 0 ||
      !ValidateVideoExecutionPlanRequest(validation_plan, &plan_reason)) {
    const auto failure =
        request.resident.contract_version !=
                kProductionVideoBackendManagerContractVersion
            ? hal::ReasonId::unsupported_contract_version
            : hal::ReasonId::malformed_contract;
    return Fail(hal::StatusId::blocked, failure, retained, evidence, reason);
  }
  if (setup_source_ == nullptr) {
    return Fail(hal::StatusId::unavailable,
                hal::ReasonId::strict_backend_unavailable, retained, evidence,
                reason);
  }

  if (evidence != nullptr)
    evidence->setup_source_called = true;
  auto candidate =
      setup_source_->AcquireResidentSetup(request.resident);
  if (evidence != nullptr) {
    evidence->candidate_readiness = candidate.readiness;
    evidence->lease_acquire = candidate.lease_acquire;
    evidence->candidate_setup_only =
        candidate.setup_completed_outside_frame_loop;
  }
  if (!candidate.setup_completed_outside_frame_loop) {
    return Fail(hal::StatusId::blocked, hal::ReasonId::malformed_contract,
                retained, evidence, reason);
  }
  if (candidate.status == hal::StatusId::degraded && !candidate.lease.live() &&
      !candidate.compatibility_executor) {
    hal::StaticAdapterRegistry registry;
    hal::ReasonId registration_reason = hal::ReasonId::none;
    if (!IsSetupSuccess(registry.Register(candidate.resolution_registration,
                                          &registration_reason))) {
      return Fail(hal::StatusId::blocked, registration_reason, retained,
                  evidence, reason);
    }
    const auto resolution = hal::ResolveSemanticStages(
        registry, BuildVideoResolutionRequest(candidate.plan.intent));
    if (!resolution.passthrough || !resolution.available ||
        !resolution.degraded ||
        resolution.status != hal::StatusId::passthrough ||
        resolution.selected_backend != hal::BackendId::invalid ||
        resolution.selected_provider !=
            hal::SemanticEffectProviderId::invalid ||
        candidate.canonical_effects != request.resident.effects ||
        !SamePassthroughPlanRequest(request.resident.plan, candidate.plan)) {
      return Fail(hal::StatusId::blocked,
                  hal::ReasonId::capability_evidence_incomplete, retained,
                  evidence, reason);
    }
    for (std::size_t index = 0; index < resolution.stage_count; ++index) {
      if (!resolution.stages[index].skipped_optional ||
          !StageParametersMatchEffects(candidate.plan.intent.stages[index],
                                       request.resident.effects)) {
        return Fail(hal::StatusId::blocked,
                    hal::ReasonId::capability_evidence_incomplete, retained,
                    evidence, reason);
      }
    }
    VideoExecutionPrepareEvidence prepare;
    const auto passthrough_status = composition_.PreparePassthrough(
        candidate.resolution_registration, candidate.plan, &prepare, reason);
    if (evidence)
      evidence->composition_prepare = prepare;
    if (passthrough_status != hal::StatusId::passthrough) {
      if (evidence) {
        evidence->status = passthrough_status;
        evidence->reason = reason ? *reason : prepare.reason;
        evidence->candidate_rejected = true;
        evidence->old_plan_retained = prepare.old_plan_retained || retained;
      }
      return passthrough_status;
    }
    resident_selected_ = true;
    if (compatibility_executor_) {
      compatibility_executor_->Shutdown();
      compatibility_executor_.reset();
    }
    snapshot_.state = ProductionVideoBackendManagerState::resident_ready;
    snapshot_.cpu_primitive_bits = 0;
    snapshot_.setup_complete = true;
    ++snapshot_.setup_successes;
    RefreshResidentSnapshot();
    if (evidence) {
      evidence->status = hal::StatusId::degraded;
      evidence->reason = resolution.reason;
    }
    SetReason(reason, resolution.reason);
    return hal::StatusId::degraded;
  }
  if (!IsSetupSuccess(candidate.status)) {
    const auto status = candidate.status == hal::StatusId::invalid
                            ? hal::StatusId::blocked
                            : candidate.status;
    const auto failure =
        candidate.reason == hal::ReasonId::none
            ? hal::ReasonId::capability_evidence_incomplete
            : candidate.reason;
    return Fail(status, failure, retained, evidence, reason);
  }
  if (candidate.canonical_effects != request.resident.effects ||
      !ValidModelSelections(candidate, request.resident.effects)) {
    return Fail(hal::StatusId::blocked, hal::ReasonId::malformed_contract,
                retained, evidence, reason);
  }
  for (std::size_t index = 0; index < candidate.plan.intent.stage_count;
       ++index) {
    if (!StageParametersMatchEffects(candidate.plan.intent.stages[index],
                                     request.resident.effects)) {
      return Fail(hal::StatusId::blocked, hal::ReasonId::malformed_contract,
                  retained, evidence, reason);
    }
  }
  if (!BudgetWithin(request.resident.plan.budget, candidate.plan.budget)) {
    return Fail(hal::StatusId::resource_exhausted,
                hal::ReasonId::resource_exhausted, retained, evidence, reason);
  }
  if (!SameCanonicalPlanRequest(request.resident, candidate.plan,
                                candidate.readiness.execution_domain.backend)) {
    return Fail(hal::StatusId::blocked, hal::ReasonId::malformed_contract,
                retained, evidence, reason);
  }
  hal::ReasonId bound_plan_reason = hal::ReasonId::none;
  if (!ValidateVideoExecutionPlanRequest(candidate.plan, &bound_plan_reason)) {
    return Fail(hal::StatusId::blocked, bound_plan_reason, retained, evidence,
                reason);
  }

  if (candidate.execution_kind ==
      ProductionVideoResidentSetupSourceResult::ExecutionKind::
          typed_compatibility) {
    if (candidate.lease.live() || !candidate.compatibility_executor ||
        !ValidCompatibilityTopology(candidate.compatibility_topology)) {
      return Fail(hal::StatusId::blocked, hal::ReasonId::malformed_contract,
                  retained, evidence, reason);
    }
    const auto executor_snapshot = candidate.compatibility_executor->Snapshot();
    if (!executor_snapshot.prepared || executor_snapshot.stopped ||
        executor_snapshot.plan_id != candidate.plan.plan_id ||
        !executor_snapshot.readiness.available) {
      return Fail(hal::StatusId::blocked,
                  hal::ReasonId::capability_evidence_incomplete, retained,
                  evidence, reason);
    }
    if (resident_selected_)
      composition_.Invalidate(hal::ReasonId::reconfiguration_required);
    if (compatibility_executor_)
      compatibility_executor_->Shutdown();
    compatibility_executor_ = std::move(candidate.compatibility_executor);
    resident_selected_ = false;
    snapshot_.state =
        ProductionVideoBackendManagerState::compatibility_ready;
    snapshot_.status = candidate.status;
    snapshot_.reason = candidate.reason;
    snapshot_.readiness = candidate.readiness;
    snapshot_.compatibility_topology = candidate.compatibility_topology;
    snapshot_.compatibility = compatibility_executor_->Snapshot();
    snapshot_.cpu_primitive_bits = 0;
    snapshot_.setup_complete = true;
    snapshot_.resident_prepared = false;
    snapshot_.frame_pending = false;
    ++snapshot_.setup_successes;
    if (evidence != nullptr) {
      evidence->status = candidate.status;
      evidence->reason = candidate.reason;
    }
    return candidate.status;
  }
  if (candidate.execution_kind !=
          ProductionVideoResidentSetupSourceResult::ExecutionKind::
              resident_island ||
      !candidate.lease.live() || candidate.compatibility_executor) {
    return Fail(hal::StatusId::blocked, hal::ReasonId::malformed_contract,
                retained, evidence, reason);
  }
  const auto &lease_snapshot = candidate.lease.snapshot();
  if (candidate.status != lease_snapshot.status ||
      candidate.reason != lease_snapshot.reason ||
      candidate.readiness.status != lease_snapshot.readiness.status ||
      candidate.readiness.reason != lease_snapshot.readiness.reason ||
      candidate.readiness.execution_domain !=
          lease_snapshot.readiness.execution_domain) {
    return Fail(hal::StatusId::blocked,
                hal::ReasonId::capability_evidence_incomplete, retained,
                evidence, reason);
  }
  if (lease_snapshot.backend == hal::BackendId::cuda &&
      !SameResolutionRegistration(candidate.resolution_registration,
                                  lease_snapshot.adapter)) {
    return Fail(hal::StatusId::blocked,
                hal::ReasonId::capability_evidence_incomplete, retained,
                evidence, reason);
  }

  hal::ReasonId validation_reason = hal::ReasonId::none;
  const auto validation_status = ValidateResidentCandidate(
      candidate.plan, lease_snapshot, &validation_reason);
  if (validation_status != hal::StatusId::ready) {
    return Fail(validation_status, validation_reason, retained, evidence,
                reason);
  }

  VideoExecutionPrepareEvidence prepare;
  const auto status = composition_.PrepareWithSetupLease(
      std::move(candidate.lease), candidate.plan, &prepare, reason);
  if (evidence != nullptr)
    evidence->composition_prepare = prepare;
  if (!IsSetupSuccess(status)) {
    if (evidence != nullptr) {
      evidence->status = status;
      evidence->reason = reason != nullptr ? *reason : prepare.reason;
      evidence->candidate_rejected = true;
      evidence->old_plan_retained = prepare.old_plan_retained || retained;
    }
    return status;
  }

  resident_selected_ = true;
  if (compatibility_executor_) {
    compatibility_executor_->Shutdown();
    compatibility_executor_.reset();
  }
  snapshot_.state = ProductionVideoBackendManagerState::resident_ready;
  snapshot_.cpu_primitive_bits = 0;
  snapshot_.setup_complete = true;
  ++snapshot_.setup_successes;
  RefreshResidentSnapshot();
  const auto final_status =
      candidate.status == hal::StatusId::degraded ||
              status == hal::StatusId::degraded
          ? hal::StatusId::degraded
          : hal::StatusId::ready;
  const auto final_reason = candidate.status == hal::StatusId::degraded
                                ? candidate.reason
                                : (reason != nullptr ? *reason : prepare.reason);
  if (final_status == hal::StatusId::degraded) {
    snapshot_.status = final_status;
    snapshot_.reason = final_reason;
    snapshot_.readiness.status = final_status;
    snapshot_.readiness.reason = final_reason;
    snapshot_.readiness.degraded = true;
  }
  if (evidence != nullptr) {
    evidence->status = final_status;
    evidence->reason = final_reason;
  }
  SetReason(reason, final_reason);
  return final_status;
}

VideoExecutionFrameResult ProductionVideoBackendManager::Execute(
    const hal::HostFrameBinding &binding) noexcept {
  if (compatibility_executor_) {
    auto result = compatibility_executor_->Execute(binding);
    snapshot_.compatibility = compatibility_executor_->Snapshot();
    snapshot_.frame_pending =
        result.status == hal::StatusId::executed ||
        result.status == hal::StatusId::degraded;
    return result;
  }
  if (!resident_selected_) {
    VideoExecutionFrameResult result;
    if (snapshot_.state ==
        ProductionVideoBackendManagerState::cpu_primitives_only) {
      // The CPU table remains a separate synchronous primitive HAL. Returning
      // passthrough with output_ready=false tells the caller there is no
      // resident result to publish and adds no hidden transfer or fallback.
      result.status = hal::StatusId::passthrough;
      result.passthrough = true;
      return result;
    }
    result.status =
        snapshot_.state == ProductionVideoBackendManagerState::stopped
            ? hal::StatusId::stopped
            : hal::StatusId::stale;
    result.reason =
        snapshot_.state == ProductionVideoBackendManagerState::stopped
            ? hal::ReasonId::shutdown_requested
            : hal::ReasonId::reconfiguration_required;
    return result;
  }
  auto result = composition_.Execute(binding);
  RefreshResidentSnapshot();
  return result;
}

hal::StatusId ProductionVideoBackendManager::CommitOutput(
    const hal::CompletionToken &completion, bool output_write_succeeded,
    hal::ReasonId *reason) noexcept {
  if (compatibility_executor_) {
    const auto status = compatibility_executor_->CommitOutput(
        completion, output_write_succeeded, reason);
    snapshot_.compatibility = compatibility_executor_->Snapshot();
    snapshot_.frame_pending = false;
    return status;
  }
  if (!resident_selected_) {
    SetReason(reason, snapshot_.state ==
                          ProductionVideoBackendManagerState::stopped
                          ? hal::ReasonId::shutdown_requested
                          : hal::ReasonId::output_not_committed);
    return snapshot_.state == ProductionVideoBackendManagerState::stopped
               ? hal::StatusId::stopped
               : hal::StatusId::blocked;
  }
  const auto status =
      composition_.CommitOutput(completion, output_write_succeeded, reason);
  RefreshResidentSnapshot();
  return status;
}

void ProductionVideoBackendManager::Invalidate(hal::ReasonId reason) noexcept {
  if (compatibility_executor_) {
    compatibility_executor_->Invalidate(reason);
    snapshot_.compatibility = compatibility_executor_->Snapshot();
  }
  if (resident_selected_) {
    composition_.Invalidate(reason);
    snapshot_.resident = composition_.Snapshot();
  }
  resident_selected_ = false;
  snapshot_.state = ProductionVideoBackendManagerState::invalidated;
  snapshot_.status = hal::StatusId::stale;
  snapshot_.reason = reason == hal::ReasonId::none
                         ? hal::ReasonId::reconfiguration_required
                         : reason;
  snapshot_.readiness.status = hal::StatusId::stale;
  snapshot_.readiness.reason = snapshot_.reason;
  snapshot_.readiness.available = false;
  snapshot_.readiness.rebuild_requested = true;
  snapshot_.setup_complete = false;
  snapshot_.resident_prepared = false;
  snapshot_.frame_pending = false;
}

hal::StatusId
ProductionVideoBackendManager::ResetTemporal(hal::ReasonId reason) noexcept {
  if (compatibility_executor_) {
    const auto status = compatibility_executor_->ResetTemporal(reason);
    snapshot_.compatibility = compatibility_executor_->Snapshot();
    if (status == hal::StatusId::ready ||
        status == hal::StatusId::passthrough) {
      return status;
    }
    const auto invalidate_reason =
        status == hal::StatusId::quarantined ||
                snapshot_.compatibility.readiness.reason ==
                    hal::ReasonId::device_lost ||
                reason == hal::ReasonId::device_lost
            ? hal::ReasonId::device_lost
            : (reason == hal::ReasonId::temporal_discontinuity
                   ? hal::ReasonId::temporal_discontinuity
                   : hal::ReasonId::reconfiguration_required);
    Invalidate(invalidate_reason);
    return status;
  }
  if (!resident_selected_)
    return snapshot_.state ==
                   ProductionVideoBackendManagerState::cpu_primitives_only
               ? hal::StatusId::passthrough
               : hal::StatusId::stale;
  const auto status = composition_.ResetTemporal(reason);
  RefreshResidentSnapshot();
  if (status == hal::StatusId::ready || status == hal::StatusId::passthrough)
    return status;
  const auto invalidate_reason =
      status == hal::StatusId::quarantined ||
              snapshot_.resident.readiness.reason ==
                  hal::ReasonId::device_lost ||
              reason == hal::ReasonId::device_lost
          ? hal::ReasonId::device_lost
          : (reason == hal::ReasonId::temporal_discontinuity
                 ? hal::ReasonId::temporal_discontinuity
                 : hal::ReasonId::reconfiguration_required);
  Invalidate(invalidate_reason);
  return status;
}

ProductionVideoSetupLeaseShutdownResult
ProductionVideoBackendManager::Shutdown() noexcept {
  if (compatibility_executor_) {
    compatibility_executor_->Shutdown();
    snapshot_.compatibility = compatibility_executor_->Snapshot();
    compatibility_executor_.reset();
  }
  auto result = composition_.Shutdown();
  if (!result.shutdown_requested)
    result = ImmediateShutdown();
  snapshot_.shutdown = result;
  snapshot_.state =
      result.pending ? ProductionVideoBackendManagerState::shutdown_pending
                     : ProductionVideoBackendManagerState::stopped;
  snapshot_.status = result.status;
  snapshot_.reason = result.reason;
  snapshot_.readiness.status = result.status;
  snapshot_.readiness.reason = result.reason;
  snapshot_.readiness.available = false;
  snapshot_.readiness.degraded = result.pending;
  snapshot_.setup_complete = false;
  snapshot_.resident_prepared = false;
  snapshot_.frame_pending = false;
  resident_selected_ = false;
  return result;
}

ProductionVideoBackendManagerSnapshot
ProductionVideoBackendManager::Snapshot() const noexcept {
  return snapshot_;
}

} // namespace studiocast::video
