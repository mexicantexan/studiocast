#include "core/video/video_execution_coordinator.h"

#include <limits>
#include <utility>

namespace studiocast::video {
namespace {

void SetReason(hal::ReasonId *reason, hal::ReasonId value) noexcept {
  if (reason != nullptr)
    *reason = value;
}

bool IsSetupSuccess(hal::StatusId status) noexcept {
  return status == hal::StatusId::ready || status == hal::StatusId::degraded ||
         status == hal::StatusId::passthrough;
}

bool IsFrameSuccess(hal::StatusId status) noexcept {
  return status == hal::StatusId::executed ||
         status == hal::StatusId::degraded ||
         status == hal::StatusId::passthrough;
}

bool IsCompatibilityEligibleSetupFailure(
    hal::StatusId status, hal::ReasonId reason,
    const VideoAdapterPrepareEvidence &prepare_evidence) noexcept {
  const bool no_candidate_attempted =
      prepare_evidence.status == hal::StatusId::invalid;
  if (!no_candidate_attempted &&
      !prepare_evidence.candidate_resources_rolled_back) {
    return false;
  }
  if (status == hal::StatusId::resource_exhausted)
    return reason == hal::ReasonId::resource_exhausted;
  if (status != hal::StatusId::unavailable)
    return false;
  return reason == hal::ReasonId::semantic_provider_unavailable ||
         reason == hal::ReasonId::capability_evidence_incomplete ||
         reason == hal::ReasonId::resource_exhausted;
}

hal::BackendId
ExpectedRequestedBackend(const VideoExecutionPlanRequest &request,
                         const hal::ResolutionResult &resolution) noexcept {
  return request.intent.backend.mode ==
                 hal::BackendRequestMode::explicit_backend
             ? request.intent.backend.backend
             : resolution.selected_backend;
}

hal::SemanticEffectProviderId
ExpectedRequestedProvider(const VideoExecutionPlanRequest &request,
                          const hal::ResolutionResult &resolution) noexcept {
  return request.intent.requested_provider !=
                 hal::SemanticEffectProviderId::invalid
             ? request.intent.requested_provider
             : resolution.selected_provider;
}

bool SameCompletion(const hal::CompletionToken &left,
                    const hal::CompletionToken &right) noexcept {
  return left.token_id == right.token_id &&
         left.program_id == right.program_id &&
         left.adapter_instance_id == right.adapter_instance_id &&
         left.capture_sequence == right.capture_sequence &&
         left.complete == right.complete;
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

bool SameModel(const hal::ModelArtifactIdentity &left,
               const hal::ModelArtifactIdentity &right) noexcept {
  return left.model_id_hash == right.model_id_hash &&
         left.artifact_set_hash == right.artifact_set_hash &&
         left.artifact_generation == right.artifact_generation;
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

bool SameTemporal(const hal::TemporalContract &left,
                  const hal::TemporalContract &right) noexcept {
  return left.temporal == right.temporal &&
         left.requires_contiguous_capture_sequence ==
             right.requires_contiguous_capture_sequence &&
         left.frame_drop_behavior == right.frame_drop_behavior &&
         left.discontinuity_action == right.discontinuity_action &&
         left.history_frames == right.history_frames;
}

bool SameTail(const hal::DeclaredCpuTail &left,
              const hal::DeclaredCpuTail &right) noexcept {
  return left.id == right.id && left.after_stage_id == right.after_stage_id &&
         left.scratch_bytes == right.scratch_bytes &&
         left.required == right.required && left.placement == right.placement &&
         left.before_stage_id == right.before_stage_id &&
         left.boundary_index == right.boundary_index;
}

bool SameParameter(const CanonicalVideoParameter &left,
                   const CanonicalVideoParameter &right) noexcept {
  return left.id == right.id && left.kind == right.kind &&
         left.signed_value == right.signed_value &&
         left.unsigned_value == right.unsigned_value &&
         left.real_value == right.real_value &&
         left.boolean_value == right.boolean_value;
}

bool SameIntent(const NormalizedVideoExecutionIntent &left,
                const NormalizedVideoExecutionIntent &right) noexcept {
  if (left.backend.mode != right.backend.mode ||
      left.backend.backend != right.backend.backend ||
      left.backend.strict != right.backend.strict ||
      left.preferences.compute_family != right.preferences.compute_family ||
      left.preferences.effects_engine != right.preferences.effects_engine ||
      left.preferences.strict_effects_engine !=
          right.preferences.strict_effects_engine ||
      left.requested_provider != right.requested_provider ||
      left.strict_provider_selection != right.strict_provider_selection ||
      left.stage_count != right.stage_count) {
    return false;
  }
  for (std::size_t index = 0; index < left.stage_count; ++index) {
    const auto &a = left.stages[index];
    const auto &b = right.stages[index];
    if (a.intent.stage_id != b.intent.stage_id ||
        a.intent.semantic != b.intent.semantic ||
        a.intent.required != b.intent.required ||
        a.intent.failure_action != b.intent.failure_action ||
        a.intent.requires_tracked_vignette_center !=
            b.intent.requires_tracked_vignette_center ||
        a.parameter_count != b.parameter_count ||
        !SameModel(a.model, b.model)) {
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

bool SamePassthroughBinding(const hal::HostFrameBinding &binding,
                            const FrozenVideoExecutionPlan &plan) noexcept {
  if (binding.capture_sequence == 0 ||
      binding.ownership != hal::OwnershipModel::borrowed_until_completion ||
      binding.lifetime != hal::ResourceLifetime::frame_borrowed ||
      !SameImage(binding.input_descriptor, plan.host_input) ||
      !SameImage(binding.output_descriptor, plan.host_output) ||
      !SameImage(binding.input_descriptor, binding.output_descriptor) ||
      binding.input_plane_count != binding.input_descriptor.plane_count ||
      binding.output_plane_count != binding.output_descriptor.plane_count ||
      binding.input_plane_count != binding.output_plane_count) {
    return false;
  }
  for (std::size_t index = 0; index < binding.input_plane_count; ++index) {
    if (binding.input_planes[index].read_data == nullptr ||
        binding.output_planes[index].write_data == nullptr ||
        binding.input_planes[index].read_data !=
            binding.output_planes[index].write_data ||
        binding.input_planes[index].span_bytes !=
            binding.output_planes[index].span_bytes ||
        binding.input_planes[index].row_stride_bytes !=
            binding.output_planes[index].row_stride_bytes) {
      return false;
    }
  }
  return true;
}

bool ExactPreparedProgram(const hal::PreparedProgram &program,
                          const VideoExecutionPlanRequest &request,
                          const hal::ResolutionResult &resolution,
                          const hal::DomainPlan &domain_plan,
                          hal::ReasonId *reason) noexcept {
  hal::ReasonId validation_reason = hal::ReasonId::none;
  if (!hal::ValidatePreparedProgram(program, &validation_reason)) {
    SetReason(reason, validation_reason);
    return false;
  }
  if (domain_plan.island_count != 1) {
    SetReason(reason, hal::ReasonId::cross_adapter_residency_forbidden);
    return false;
  }
  const auto &island = domain_plan.islands[0];
  if (program.program_id != request.plan_id ||
      program.owner_adapter_instance_id != island.adapter_instance_id ||
      program.execution_domain != island.execution_domain ||
      program.stage_count != island.stage_count ||
      !SameImage(program.host_input, request.host_input) ||
      !SameImage(program.host_output, request.host_output) ||
      program.invalidation_key != request.invalidation_key ||
      !SameTemporal(program.temporal, request.temporal) ||
      !SameBudget(program.budget, request.budget) ||
      program.cpu_tail_count != request.cpu_tail_count ||
      program.setup_failure_policy != request.setup_failure_policy ||
      program.requested_backend !=
          ExpectedRequestedBackend(request, resolution) ||
      program.strict_backend_selection != request.intent.backend.strict ||
      program.requested_provider !=
          ExpectedRequestedProvider(request, resolution) ||
      program.strict_provider_selection !=
          request.intent.strict_provider_selection ||
      program.requested_execution_domain != island.execution_domain ||
      !program.strict_execution_domain ||
      program.requires_upload != island.requires_upload ||
      program.requires_final_readback != island.requires_final_readback) {
    SetReason(reason, program.execution_domain != island.execution_domain
                          ? hal::ReasonId::wrong_execution_domain
                          : hal::ReasonId::malformed_contract);
    return false;
  }
  std::size_t prepared_index = 0;
  std::size_t expected_tail_count = 0;
  for (std::size_t index = 0; index < resolution.stage_count; ++index) {
    const auto &resolved = resolution.stages[index];
    if (resolved.skipped_optional)
      continue;
    if (prepared_index >= program.stage_count ||
        prepared_index >= island.stage_count ||
        island.resolved_stage_indices[prepared_index] != index) {
      SetReason(reason, hal::ReasonId::malformed_contract);
      return false;
    }
    const auto &stage = program.stages[prepared_index];
    if (stage.stage_id != resolved.intent.stage_id ||
        stage.semantic != resolved.intent.semantic ||
        stage.primitive != hal::PrimitiveOperationId::invalid ||
        stage.backend != resolved.selected_backend ||
        stage.provider != resolved.provider ||
        stage.execution_domain != resolved.execution_domain ||
        stage.required != resolved.intent.required ||
        stage.failure_action != resolved.intent.failure_action ||
        !SameModel(stage.model, request.intent.stages[index].model)) {
      SetReason(reason, stage.execution_domain != resolved.execution_domain
                            ? hal::ReasonId::wrong_execution_domain
                            : hal::ReasonId::malformed_contract);
      return false;
    }
    if (resolved.cpu_tail != hal::CpuTailId::invalid) {
      if (expected_tail_count >= program.cpu_tail_count) {
        SetReason(reason, hal::ReasonId::malformed_contract);
        return false;
      }
      const auto &tail = program.cpu_tails[expected_tail_count++];
      const bool before = resolved.cpu_tail_placement ==
                          hal::DeclaredCpuTail::Placement::before_stage;
      const auto expected_boundary =
          static_cast<std::uint32_t>(before ? index : index + 1);
      if (tail.id != resolved.cpu_tail ||
          tail.scratch_bytes != resolved.cpu_tail_scratch_bytes ||
          tail.placement != resolved.cpu_tail_placement ||
          tail.boundary_index != expected_boundary ||
          (before && (tail.before_stage_id != resolved.intent.stage_id ||
                      tail.after_stage_id != 0)) ||
          (!before && (tail.after_stage_id != resolved.intent.stage_id ||
                       tail.before_stage_id != 0))) {
        SetReason(reason, hal::ReasonId::malformed_contract);
        return false;
      }
    }
    ++prepared_index;
  }
  for (std::size_t index = 0; index < request.cpu_tail_count; ++index) {
    if (!SameTail(program.cpu_tails[index], request.cpu_tails[index])) {
      SetReason(reason, hal::ReasonId::malformed_contract);
      return false;
    }
  }
  if (prepared_index != program.stage_count ||
      prepared_index != island.stage_count ||
      expected_tail_count != program.cpu_tail_count ||
      domain_plan.expected_transfers.cpu_tail_boundaries !=
          program.cpu_tail_count) {
    SetReason(reason, hal::ReasonId::malformed_contract);
    return false;
  }
  return true;
}

bool ExactPreflight(const VideoAdapterFactoryRegistration &registration,
                    const VideoAdapterFactoryPreflight &preflight,
                    const hal::ResolutionResult &resolution,
                    const hal::DomainPlan &domain_plan,
                    VideoExecutionRoute route, hal::ReasonId *reason) noexcept {
  hal::ReasonId transfer_reason = hal::ReasonId::none;
  if (!preflight.cached_evidence_only || !preflight.accepts_frozen_resolution ||
      preflight.performs_selection ||
      preflight.backend != resolution.selected_backend ||
      preflight.provider != resolution.selected_provider ||
      preflight.backend != registration.backend ||
      (registration.provider != hal::SemanticEffectProviderId::invalid &&
       preflight.provider != registration.provider) ||
      domain_plan.island_count != 1 ||
      preflight.execution_domain != domain_plan.islands[0].execution_domain ||
      preflight.execution_domain != registration.execution_domain ||
      preflight.execution_domain.adapter_instance_id !=
          registration.adapter_instance_id ||
      registration.route != route ||
      !ValidatePerFrameTransferContract(preflight.per_frame_transfers,
                                        domain_plan, &transfer_reason)) {
    SetReason(reason, preflight.execution_domain !=
                              domain_plan.islands[0].execution_domain
                          ? hal::ReasonId::wrong_execution_domain
                          : hal::ReasonId::malformed_contract);
    return false;
  }
  return true;
}

bool ValidateExecutedStages(const VideoExecutionFrameResult &frame,
                            const FrozenVideoExecutionPlan &plan,
                            hal::ReasonId *reason) noexcept {
  if (plan.route == VideoExecutionRoute::passthrough) {
    if (frame.stage_count != 0) {
      SetReason(reason, hal::ReasonId::malformed_contract);
      return false;
    }
    return true;
  }
  std::size_t result_index = 0;
  for (std::size_t index = 0; index < plan.resolution.stage_count; ++index) {
    const auto &resolved = plan.resolution.stages[index];
    if (resolved.skipped_optional)
      continue;
    if (result_index >= frame.stage_count) {
      SetReason(reason, hal::ReasonId::incomplete_execution);
      return false;
    }
    const auto &stage = frame.stages[result_index++];
    if (stage.stage_id != resolved.intent.stage_id ||
        stage.semantic != resolved.intent.semantic ||
        stage.primitive != hal::PrimitiveOperationId::invalid ||
        stage.status == hal::StageExecutionStatus::not_run ||
        (resolved.intent.required &&
         stage.status != hal::StageExecutionStatus::succeeded) ||
        (!resolved.intent.required &&
         stage.status == hal::StageExecutionStatus::required_failed) ||
        ((stage.status == hal::StageExecutionStatus::optional_failed_open ||
          stage.status == hal::StageExecutionStatus::disabled_preserved) &&
         !stage.valid_output_preserved)) {
      SetReason(reason,
                resolved.intent.required &&
                        stage.status != hal::StageExecutionStatus::succeeded
                    ? hal::ReasonId::required_stage_failed
                    : hal::ReasonId::malformed_contract);
      return false;
    }
  }
  if (result_index != frame.stage_count) {
    SetReason(reason, hal::ReasonId::malformed_contract);
    return false;
  }
  return true;
}

const hal::ResolvedStage *
FindResolvedStage(const hal::ResolutionResult &resolution,
                  std::uint32_t stage_id) noexcept {
  for (std::size_t index = 0; index < resolution.stage_count; ++index) {
    if (!resolution.stages[index].skipped_optional &&
        resolution.stages[index].intent.stage_id == stage_id) {
      return &resolution.stages[index];
    }
  }
  return nullptr;
}

} // namespace

VideoExecutionCoordinator::VideoExecutionCoordinator(
    const hal::StaticAdapterRegistry &capabilities) noexcept
    : capabilities_(&capabilities) {}

VideoExecutionCoordinator::~VideoExecutionCoordinator() { Shutdown(); }

hal::StatusId VideoExecutionCoordinator::RegisterFactory(
    const VideoAdapterFactoryRegistration &registration,
    hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  if (stopped_) {
    SetReason(reason, hal::ReasonId::shutdown_requested);
    return hal::StatusId::stopped;
  }
  if (factory_count_ >= factories_.size()) {
    SetReason(reason, hal::ReasonId::resource_exhausted);
    return hal::StatusId::resource_exhausted;
  }
  if (registration.factory_id == 0 || registration.factory == nullptr ||
      registration.backend == hal::BackendId::invalid ||
      registration.adapter_instance_id == 0 ||
      registration.execution_domain.adapter_instance_id !=
          registration.adapter_instance_id ||
      !hal::ValidateExecutionDomain(registration.execution_domain) ||
      (registration.route != VideoExecutionRoute::prepared_resident &&
       registration.route != VideoExecutionRoute::legacy_compatibility)) {
    SetReason(reason, hal::ReasonId::malformed_contract);
    return hal::StatusId::blocked;
  }
  for (std::size_t index = 0; index < factory_count_; ++index) {
    if (factories_[index].factory_id == registration.factory_id) {
      SetReason(reason, hal::ReasonId::foreign_adapter);
      return hal::StatusId::blocked;
    }
  }
  factories_[factory_count_++] = registration;
  return hal::StatusId::ready;
}

const VideoAdapterFactoryRegistration *VideoExecutionCoordinator::FindFactory(
    VideoExecutionRoute route, const hal::ResolutionResult &resolution,
    const hal::DomainPlan &domain_plan) const noexcept {
  if (domain_plan.island_count != 1)
    return nullptr;
  const auto &island = domain_plan.islands[0];
  const VideoAdapterFactoryRegistration *best = nullptr;
  for (std::size_t index = 0; index < factory_count_; ++index) {
    const auto &candidate = factories_[index];
    if (candidate.route != route ||
        candidate.backend != resolution.selected_backend ||
        (candidate.provider != hal::SemanticEffectProviderId::invalid &&
         candidate.provider != resolution.selected_provider) ||
        candidate.adapter_instance_id != island.adapter_instance_id ||
        candidate.execution_domain != island.execution_domain) {
      continue;
    }
    if (best == nullptr ||
        candidate.selection_priority < best->selection_priority ||
        (candidate.selection_priority == best->selection_priority &&
         candidate.factory_id < best->factory_id)) {
      best = &candidate;
    }
  }
  return best;
}

hal::StatusId VideoExecutionCoordinator::TryPrepareRoute(
    const VideoExecutionPlanRequest &request, VideoExecutionRoute route,
    const hal::ResolutionResult &resolution, const hal::DomainPlan &domain_plan,
    std::unique_ptr<IVideoExecutionExecutor> *candidate,
    FrozenVideoExecutionPlan *candidate_plan,
    VideoAdapterFactoryPreflight *preflight,
    VideoAdapterPrepareEvidence *prepare_evidence,
    hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  const auto *registration = FindFactory(route, resolution, domain_plan);
  if (registration == nullptr) {
    SetReason(reason, hal::ReasonId::semantic_provider_unavailable);
    return hal::StatusId::unavailable;
  }
  *preflight =
      registration->factory->Preflight(request, resolution, domain_plan, route);
  if (!IsSetupSuccess(preflight->status)) {
    SetReason(reason, preflight->reason);
    return preflight->status;
  }
  if (!ExactPreflight(*registration, *preflight, resolution, domain_plan, route,
                      reason)) {
    return hal::StatusId::blocked;
  }
  if (request.intent.strict_provider_selection &&
      preflight->provider != request.intent.requested_provider) {
    SetReason(reason, hal::ReasonId::semantic_provider_unavailable);
    return hal::StatusId::blocked;
  }
  if (request.intent.backend.strict &&
      preflight->backend != request.intent.backend.backend) {
    SetReason(reason, hal::ReasonId::strict_backend_unavailable);
    return hal::StatusId::blocked;
  }
  if (route == VideoExecutionRoute::prepared_resident &&
      !ExactPreparedProgram(preflight->prepared_program, request, resolution,
                            domain_plan, reason)) {
    return hal::StatusId::blocked;
  }

  *candidate_plan = {};
  candidate_plan->plan_id = request.plan_id;
  candidate_plan->route = route;
  candidate_plan->intent = request.intent;
  candidate_plan->host_input = request.host_input;
  candidate_plan->host_output = request.host_output;
  candidate_plan->invalidation_key = request.invalidation_key;
  candidate_plan->temporal = request.temporal;
  candidate_plan->budget = request.budget;
  candidate_plan->cpu_tails = request.cpu_tails;
  candidate_plan->cpu_tail_count = request.cpu_tail_count;
  candidate_plan->setup_failure_policy = request.setup_failure_policy;
  candidate_plan->migration_policy = request.migration_policy;
  candidate_plan->resolution = resolution;
  candidate_plan->domain_plan = domain_plan;
  candidate_plan->factory_id = registration->factory_id;
  candidate_plan->per_frame_transfers = preflight->per_frame_transfers;
  if (route == VideoExecutionRoute::prepared_resident)
    candidate_plan->prepared_program = preflight->prepared_program;

  *candidate =
      registration->factory->Prepare(*candidate_plan, prepare_evidence);
  if (!*candidate || !IsSetupSuccess(prepare_evidence->status)) {
    if (*candidate)
      (*candidate)->Shutdown();
    const auto failure = prepare_evidence->reason == hal::ReasonId::none
                             ? hal::ReasonId::incomplete_execution
                             : prepare_evidence->reason;
    candidate->reset();
    if (!prepare_evidence->candidate_resources_rolled_back) {
      SetReason(reason, hal::ReasonId::incomplete_execution);
      return hal::StatusId::blocked;
    }
    SetReason(reason, failure);
    return prepare_evidence->status == hal::StatusId::invalid
               ? hal::StatusId::failed
               : prepare_evidence->status;
  }
  const auto snapshot = (*candidate)->Snapshot();
  if (!snapshot.prepared || snapshot.stopped ||
      snapshot.plan_id != request.plan_id ||
      snapshot.execution_domain != domain_plan.islands[0].execution_domain ||
      snapshot.readiness.selected_backend != resolution.selected_backend ||
      snapshot.readiness.provider != resolution.selected_provider) {
    (*candidate)->Shutdown();
    candidate->reset();
    SetReason(reason, snapshot.execution_domain !=
                              domain_plan.islands[0].execution_domain
                          ? hal::ReasonId::wrong_execution_domain
                          : hal::ReasonId::malformed_contract);
    return hal::StatusId::blocked;
  }
  const auto status =
      preflight->status == hal::StatusId::degraded ||
              prepare_evidence->status == hal::StatusId::degraded
          ? hal::StatusId::degraded
          : hal::StatusId::ready;
  SetReason(reason, prepare_evidence->reason != hal::ReasonId::none
                        ? prepare_evidence->reason
                        : preflight->reason);
  return status;
}

hal::StatusId
VideoExecutionCoordinator::Prepare(const VideoExecutionPlanRequest &request,
                                   VideoExecutionPrepareEvidence *evidence,
                                   hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  prepare_ = {};
  if (stopped_) {
    prepare_.status = hal::StatusId::stopped;
    prepare_.reason = hal::ReasonId::shutdown_requested;
    SetReason(reason, prepare_.reason);
    if (evidence)
      *evidence = prepare_;
    return prepare_.status;
  }
  if (frame_pending_) {
    prepare_.status = hal::StatusId::blocked;
    prepare_.reason = hal::ReasonId::output_not_committed;
    SetReason(reason, prepare_.reason);
    if (evidence)
      *evidence = prepare_;
    return prepare_.status;
  }
  hal::ReasonId validation_reason = hal::ReasonId::none;
  if (!ValidateVideoExecutionPlanRequest(request, &validation_reason)) {
    prepare_.status = hal::StatusId::blocked;
    prepare_.reason = validation_reason;
    SetReason(reason, validation_reason);
    if (evidence)
      *evidence = prepare_;
    return prepare_.status;
  }

  const auto resolution = hal::ResolveSemanticStages(
      *capabilities_, BuildVideoResolutionRequest(request.intent));
  const auto domain_plan = hal::PlanExecutionDomains(
      *capabilities_, resolution, hal::CrossAdapterPolicy::reject);
  prepare_.resolution = resolution;
  prepare_.domain_plan = domain_plan;
  if (!IsSetupSuccess(resolution.status)) {
    prepare_.status = resolution.status;
    prepare_.reason = resolution.reason;
  } else if (!IsSetupSuccess(domain_plan.status)) {
    prepare_.status = domain_plan.status;
    prepare_.reason = domain_plan.reason;
  } else if (request.intent.backend.strict &&
             resolution.selected_backend != request.intent.backend.backend) {
    prepare_.status = hal::StatusId::blocked;
    prepare_.reason = hal::ReasonId::strict_backend_unavailable;
  } else if (request.intent.strict_provider_selection &&
             resolution.selected_provider !=
                 request.intent.requested_provider) {
    prepare_.status = hal::StatusId::blocked;
    prepare_.reason = hal::ReasonId::semantic_provider_unavailable;
  } else if (domain_plan.passthrough) {
    ActivatePassthrough(request, resolution, domain_plan);
    prepare_.status = hal::StatusId::passthrough;
    prepare_.reason = hal::ReasonId::none;
    prepare_.selected_route = VideoExecutionRoute::passthrough;
    prepare_.readiness = readiness_;
    if (evidence)
      *evidence = prepare_;
    return prepare_.status;
  } else {
    std::unique_ptr<IVideoExecutionExecutor> candidate;
    FrozenVideoExecutionPlan candidate_plan;
    VideoAdapterFactoryPreflight preflight;
    VideoAdapterPrepareEvidence adapter_prepare;
    hal::ReasonId route_reason = hal::ReasonId::none;
    prepare_.prepared_factory_attempted = true;
    auto status =
        TryPrepareRoute(request, VideoExecutionRoute::prepared_resident,
                        resolution, domain_plan, &candidate, &candidate_plan,
                        &preflight, &adapter_prepare, &route_reason);
    prepare_.candidate_resources_rolled_back =
        adapter_prepare.candidate_resources_rolled_back;
    if (IsCompatibilityEligibleSetupFailure(status, route_reason,
                                            adapter_prepare) &&
        request.migration_policy ==
            VideoExecutionMigrationPolicy::
                prepared_then_same_resolution_compatibility) {
      prepare_.compatibility_factory_attempted = true;
      preflight = {};
      adapter_prepare = {};
      route_reason = hal::ReasonId::none;
      status =
          TryPrepareRoute(request, VideoExecutionRoute::legacy_compatibility,
                          resolution, domain_plan, &candidate, &candidate_plan,
                          &preflight, &adapter_prepare, &route_reason);
      prepare_.compatibility_consumed_frozen_resolution =
          IsSetupSuccess(status);
      prepare_.candidate_resources_rolled_back =
          prepare_.candidate_resources_rolled_back ||
          adapter_prepare.candidate_resources_rolled_back;
    }
    if (IsSetupSuccess(status)) {
      if (executor_) {
        executor_->Invalidate(hal::ReasonId::reconfiguration_required);
        executor_->Shutdown();
      }
      executor_ = std::move(candidate);
      plan_ = candidate_plan;
      const auto snapshot = executor_->Snapshot();
      readiness_ = snapshot.readiness;
      prepared_ = true;
      passthrough_ = false;
      quarantined_ = false;
      committed_ = {};
      prepare_.status = status;
      prepare_.reason = route_reason;
      prepare_.selected_route = plan_.route;
      prepare_.selected_factory_id = plan_.factory_id;
      prepare_.readiness = readiness_;
      if (evidence)
        *evidence = prepare_;
      return status;
    }
    prepare_.status = status;
    prepare_.reason = route_reason;
  }

  if (MayRetainOld(request)) {
    prepare_.old_plan_retained = true;
  } else {
    Invalidate(prepare_.reason == hal::ReasonId::none
                   ? hal::ReasonId::reconfiguration_required
                   : prepare_.reason);
  }
  SetReason(reason, prepare_.reason);
  if (evidence)
    *evidence = prepare_;
  return prepare_.status;
}

bool VideoExecutionCoordinator::MayRetainOld(
    const VideoExecutionPlanRequest &request) const noexcept {
  return prepared_ &&
         request.setup_failure_policy ==
             hal::SetupFailurePolicy::retain_old_program &&
         plan_.invalidation_key == request.invalidation_key &&
         SameIntent(plan_.intent, request.intent) &&
         SameImage(plan_.host_input, request.host_input) &&
         SameImage(plan_.host_output, request.host_output) &&
         SameTemporal(plan_.temporal, request.temporal) &&
         SameBudget(plan_.budget, request.budget) &&
         plan_.setup_failure_policy == request.setup_failure_policy &&
         plan_.migration_policy == request.migration_policy &&
         plan_.cpu_tail_count == request.cpu_tail_count && [&]() noexcept {
           for (std::size_t index = 0; index < request.cpu_tail_count;
                ++index) {
             if (!SameTail(plan_.cpu_tails[index], request.cpu_tails[index]))
               return false;
           }
           return true;
         }();
}

void VideoExecutionCoordinator::ActivatePassthrough(
    const VideoExecutionPlanRequest &request,
    const hal::ResolutionResult &resolution,
    const hal::DomainPlan &domain_plan) noexcept {
  if (executor_) {
    executor_->Invalidate(hal::ReasonId::reconfiguration_required);
    executor_->Shutdown();
    executor_.reset();
  }
  plan_ = {};
  plan_.plan_id = request.plan_id;
  plan_.route = VideoExecutionRoute::passthrough;
  plan_.intent = request.intent;
  plan_.host_input = request.host_input;
  plan_.host_output = request.host_output;
  plan_.invalidation_key = request.invalidation_key;
  plan_.temporal = request.temporal;
  plan_.budget = request.budget;
  plan_.cpu_tails = request.cpu_tails;
  plan_.cpu_tail_count = request.cpu_tail_count;
  plan_.setup_failure_policy = request.setup_failure_policy;
  plan_.migration_policy = request.migration_policy;
  plan_.resolution = resolution;
  plan_.domain_plan = domain_plan;
  readiness_ = {};
  readiness_.status = hal::StatusId::passthrough;
  readiness_.reason = hal::ReasonId::none;
  readiness_.requested_backend = resolution.requested_backend;
  readiness_.available = true;
  prepared_ = true;
  passthrough_ = true;
  quarantined_ = false;
  committed_ = {};
}

VideoExecutionFrameResult VideoExecutionCoordinator::Execute(
    const hal::HostFrameBinding &binding) noexcept {
  VideoExecutionFrameResult frame;
  if (stopped_) {
    frame.status = hal::StatusId::stopped;
    frame.reason = hal::ReasonId::shutdown_requested;
    return frame;
  }
  if (frame_pending_) {
    frame.reason = hal::ReasonId::output_not_committed;
    return frame;
  }
  if (!prepared_ || quarantined_) {
    frame.status =
        quarantined_ ? hal::StatusId::quarantined : hal::StatusId::blocked;
    frame.reason = quarantined_ ? hal::ReasonId::device_lost
                                : hal::ReasonId::stale_program;
    return frame;
  }
  if (passthrough_) {
    if (!SamePassthroughBinding(binding, plan_)) {
      frame.status = hal::StatusId::blocked;
      frame.reason = hal::ReasonId::host_frame_binding_invalid;
      return frame;
    }
    if (passthrough_token_ == std::numeric_limits<std::uint64_t>::max()) {
      frame.status = hal::StatusId::resource_exhausted;
      frame.reason = hal::ReasonId::resource_exhausted;
      return frame;
    }
    ++passthrough_token_;
    frame.status = hal::StatusId::passthrough;
    frame.completion = {passthrough_token_, plan_.plan_id, 0,
                        binding.capture_sequence, true};
    frame.output.descriptor = binding.output_descriptor;
    frame.output.planes = binding.output_planes;
    frame.output.plane_count = binding.output_plane_count;
    frame.output.capture_sequence = binding.capture_sequence;
    frame.output.output_ready = true;
    frame.output_ready = true;
    frame.passthrough = true;
  } else {
    frame = executor_->Execute(binding);
  }

  ++telemetry_.execution_attempts;
  telemetry_.transfers.uploads += frame.transfers.uploads;
  telemetry_.transfers.final_readbacks += frame.transfers.final_readbacks;
  telemetry_.transfers.intermediate_readbacks +=
      frame.transfers.intermediate_readbacks;
  telemetry_.transfers.intermediate_uploads +=
      frame.transfers.intermediate_uploads;
  telemetry_.transfers.synchronizations += frame.transfers.synchronizations;
  telemetry_.transfers.cpu_tail_boundaries +=
      frame.transfers.cpu_tail_boundaries;
  telemetry_.runtime_status = frame.status;
  telemetry_.runtime_reason = frame.reason;

  if (frame.reason == hal::ReasonId::device_lost) {
    ++telemetry_.device_losses;
    quarantined_ = true;
    readiness_.status = hal::StatusId::quarantined;
    readiness_.reason = hal::ReasonId::device_lost;
    readiness_.available = false;
    if (executor_)
      executor_->Invalidate(hal::ReasonId::device_lost);
    return frame;
  }
  if (!IsFrameSuccess(frame.status))
    return frame;
  const auto expected_adapter =
      passthrough_ ? 0 : plan_.domain_plan.islands[0].adapter_instance_id;
  if (!frame.completion.complete || frame.completion.token_id == 0 ||
      frame.completion.program_id != plan_.plan_id ||
      frame.completion.adapter_instance_id != expected_adapter ||
      frame.completion.capture_sequence != binding.capture_sequence ||
      !frame.output_ready || !frame.output.output_ready ||
      frame.output.capture_sequence != binding.capture_sequence ||
      !SameImage(frame.output.descriptor, plan_.host_output) ||
      (!passthrough_ && frame.execution_domain !=
                            plan_.domain_plan.islands[0].execution_domain) ||
      frame.stage_count > hal::kMaxPreparedStages) {
    frame.status = hal::StatusId::failed;
    frame.reason =
        !passthrough_ && frame.execution_domain !=
                             plan_.domain_plan.islands[0].execution_domain
            ? hal::ReasonId::wrong_execution_domain
            : hal::ReasonId::incomplete_execution;
    telemetry_.runtime_status = frame.status;
    telemetry_.runtime_reason = frame.reason;
    return frame;
  }
  const auto within = [](std::uint64_t value, std::uint64_t minimum,
                         std::uint64_t maximum) noexcept {
    return value >= minimum && value <= maximum;
  };
  const auto &transfers = plan_.per_frame_transfers;
  const bool variable_boundary_mismatch =
      transfers.variable_boundaries_declared &&
      (!within(frame.transfers.intermediate_readbacks,
               transfers.minimum_intermediate_readbacks,
               transfers.maximum_intermediate_readbacks) ||
       !within(frame.transfers.intermediate_uploads,
               transfers.minimum_intermediate_uploads,
               transfers.maximum_intermediate_uploads) ||
       !within(frame.transfers.cpu_tail_boundaries,
               transfers.minimum_cpu_tail_boundaries,
               transfers.maximum_cpu_tail_boundaries));
  const bool exact_boundary_mismatch =
      !transfers.variable_boundaries_declared &&
      (frame.transfers.intermediate_readbacks !=
           transfers.intermediate_readbacks ||
       frame.transfers.intermediate_uploads !=
           transfers.intermediate_uploads ||
       frame.transfers.cpu_tail_boundaries != transfers.cpu_tail_boundaries);
  if (!passthrough_ &&
      (frame.transfers.uploads != plan_.per_frame_transfers.uploads ||
       frame.transfers.final_readbacks !=
           plan_.per_frame_transfers.final_readbacks ||
       variable_boundary_mismatch || exact_boundary_mismatch ||
       frame.transfers.synchronizations <
           plan_.per_frame_transfers.minimum_synchronizations ||
       frame.transfers.synchronizations >
           plan_.per_frame_transfers.maximum_synchronizations)) {
    frame.status = hal::StatusId::failed;
    frame.reason = hal::ReasonId::transfer_contract_violated;
    telemetry_.runtime_status = frame.status;
    telemetry_.runtime_reason = frame.reason;
    return frame;
  }
  hal::ReasonId stage_reason = hal::ReasonId::none;
  if (!ValidateExecutedStages(frame, plan_, &stage_reason)) {
    frame.status = hal::StatusId::failed;
    frame.reason = stage_reason;
    telemetry_.runtime_status = frame.status;
    telemetry_.runtime_reason = frame.reason;
    return frame;
  }

  ++telemetry_.execution_successes;
  pending_completion_ = frame.completion;
  pending_stages_ = frame.stages;
  pending_stage_count_ = frame.stage_count;
  frame_pending_ = true;
  return frame;
}

hal::StatusId
VideoExecutionCoordinator::CommitOutput(const hal::CompletionToken &completion,
                                        bool output_write_succeeded,
                                        hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  if (!frame_pending_ || !SameCompletion(completion, pending_completion_)) {
    SetReason(reason, hal::ReasonId::output_not_committed);
    return hal::StatusId::failed;
  }
  hal::ReasonId commit_reason = hal::ReasonId::none;
  hal::StatusId status = hal::StatusId::ready;
  if (!passthrough_) {
    status = executor_->CommitOutput(completion, output_write_succeeded,
                                     &commit_reason);
  } else if (!output_write_succeeded) {
    status = hal::StatusId::failed;
    commit_reason = hal::ReasonId::output_write_failed;
  }
  if (!output_write_succeeded && commit_reason == hal::ReasonId::none) {
    status = hal::StatusId::failed;
    commit_reason = hal::ReasonId::output_write_failed;
  }
  if (status == hal::StatusId::ready && output_write_succeeded) {
    ++telemetry_.committed_output_frames;
    committed_ = {};
    committed_.valid = true;
    committed_.program_id = plan_.plan_id;
    committed_.invalidation_key = plan_.invalidation_key;
    if (!passthrough_)
      committed_.execution_domain =
          plan_.domain_plan.islands[0].execution_domain;
    committed_.capture_sequence = completion.capture_sequence;
    committed_.output_commit_sequence = telemetry_.committed_output_frames;
    bool active_frame = false;
    for (std::size_t index = 0; index < pending_stage_count_; ++index) {
      const auto &stage = pending_stages_[index];
      auto &activity = committed_.stages[committed_.stage_count++];
      activity.stage_id = stage.stage_id;
      activity.semantic = stage.semantic;
      activity.primitive = stage.primitive;
      activity.status = stage.status;
      activity.reason = stage.reason;
      if (stage.status == hal::StageExecutionStatus::succeeded) {
        const auto *resolved =
            FindResolvedStage(plan_.resolution, stage.stage_id);
        if (resolved != nullptr) {
          activity.backend = resolved->selected_backend;
          activity.provider = resolved->provider;
        }
        active_frame = true;
        if (telemetry_.active_backend == hal::BackendId::invalid)
          telemetry_.active_backend = activity.backend;
        else if (telemetry_.active_backend != activity.backend)
          telemetry_.active_backend = hal::BackendId::invalid;
        if (telemetry_.active_provider ==
            hal::SemanticEffectProviderId::invalid)
          telemetry_.active_provider = activity.provider;
        else if (telemetry_.active_provider != activity.provider)
          telemetry_.active_provider = hal::SemanticEffectProviderId::invalid;
        const std::size_t stage_slot =
            committed_.stage_count <= hal::kMaxPreparedStages
                ? committed_.stage_count - 1
                : 0;
        ++telemetry_.committed_stage_successes[stage_slot];
      } else if (stage.status ==
                 hal::StageExecutionStatus::optional_failed_open) {
        ++telemetry_.optional_fail_open_stages;
      }
    }
    if (active_frame)
      ++telemetry_.active_backend_frames;
  }
  ClearPending();
  telemetry_.runtime_status = status;
  telemetry_.runtime_reason = commit_reason;
  SetReason(reason, commit_reason);
  return status;
}

void VideoExecutionCoordinator::Invalidate(hal::ReasonId reason) noexcept {
  if (executor_)
    executor_->Invalidate(reason);
  ClearPending();
  prepared_ = false;
  passthrough_ = false;
  quarantined_ = reason == hal::ReasonId::device_lost;
  readiness_.status =
      quarantined_ ? hal::StatusId::quarantined : hal::StatusId::stale;
  readiness_.reason = reason;
  readiness_.available = false;
  committed_ = {};
  plan_ = {};
}

hal::StatusId
VideoExecutionCoordinator::ResetTemporal(hal::ReasonId reason) noexcept {
  if (stopped_)
    return hal::StatusId::stopped;
  if (frame_pending_)
    return hal::StatusId::blocked;
  if (!prepared_)
    return hal::StatusId::stale;
  const auto status =
      passthrough_ ? hal::StatusId::ready : executor_->ResetTemporal(reason);
  if (status == hal::StatusId::ready)
    ++telemetry_.temporal_resets;
  return status;
}

void VideoExecutionCoordinator::Shutdown() noexcept {
  if (stopped_)
    return;
  if (executor_)
    executor_->Shutdown();
  executor_.reset();
  ClearPending();
  ClearActive();
  stopped_ = true;
  readiness_.status = hal::StatusId::stopped;
  readiness_.reason = hal::ReasonId::shutdown_requested;
  readiness_.available = false;
  telemetry_.runtime_status = hal::StatusId::stopped;
  telemetry_.runtime_reason = hal::ReasonId::shutdown_requested;
}

void VideoExecutionCoordinator::ClearPending() noexcept {
  frame_pending_ = false;
  pending_completion_ = {};
  pending_stages_ = {};
  pending_stage_count_ = 0;
}

void VideoExecutionCoordinator::ClearActive() noexcept {
  prepared_ = false;
  passthrough_ = false;
  quarantined_ = false;
  plan_ = {};
  committed_ = {};
}

VideoExecutionCoordinatorSnapshot
VideoExecutionCoordinator::Snapshot() const noexcept {
  VideoExecutionCoordinatorSnapshot snapshot;
  snapshot.plan = plan_;
  snapshot.prepare = prepare_;
  snapshot.readiness = readiness_;
  snapshot.telemetry = telemetry_;
  snapshot.latest_committed_frame = committed_;
  snapshot.prepared = prepared_;
  snapshot.frame_pending = frame_pending_;
  snapshot.quarantined = quarantined_;
  snapshot.stopped = stopped_;
  return snapshot;
}

} // namespace studiocast::video
