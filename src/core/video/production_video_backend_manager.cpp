#include "core/video/production_video_backend_manager.h"

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
      caller.stage_count != bound.stage_count) {
    return false;
  }
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

bool SameCanonicalPlanRequest(const VideoExecutionPlanRequest &caller,
                              const VideoExecutionPlanRequest &bound) noexcept {
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
      !SameTemporal(caller.temporal, bound.temporal) ||
      !BudgetWithin(caller.budget, bound.budget) ||
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

  for (std::size_t index = 0; index < plan.intent.stage_count;
       ++index) {
    const auto &stage = plan.intent.stages[index];
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

std::uint64_t ProductionVideoExecutionDomainHash(
    const hal::ExecutionDomainId &domain) noexcept {
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

ProductionVideoBackendManager::ProductionVideoBackendManager(
    std::unique_ptr<IProductionVideoBackendSetupSource> setup_source)
    : setup_source_(std::move(setup_source)) {}

ProductionVideoBackendManager::~ProductionVideoBackendManager() = default;

bool ProductionVideoBackendManager::HasRetainableSetup() const noexcept {
  return snapshot_.setup_complete &&
         (snapshot_.state ==
              ProductionVideoBackendManagerState::cpu_primitives_only ||
          snapshot_.resident_prepared);
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
      request.resident.plan.invalidation_key.execution_domain_hash != 0 ||
      !ValidateVideoExecutionPlanRequest(validation_plan, &plan_reason)) {
    const auto failure =
        request.resident.contract_version !=
                kProductionVideoBackendManagerContractVersion
            ? hal::ReasonId::unsupported_contract_version
            : plan_reason;
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
  if (!IsSetupSuccess(candidate.status) || !candidate.lease.live()) {
    const auto status = candidate.status == hal::StatusId::invalid
                            ? hal::StatusId::blocked
                            : candidate.status;
    const auto failure =
        candidate.reason == hal::ReasonId::none
            ? hal::ReasonId::capability_evidence_incomplete
            : candidate.reason;
    return Fail(status, failure, retained, evidence, reason);
  }
  if (!BudgetWithin(request.resident.plan.budget, candidate.plan.budget)) {
    return Fail(hal::StatusId::resource_exhausted,
                hal::ReasonId::resource_exhausted, retained, evidence, reason);
  }
  if (!SameCanonicalPlanRequest(request.resident.plan, candidate.plan)) {
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
  snapshot_.state = ProductionVideoBackendManagerState::resident_ready;
  snapshot_.cpu_primitive_bits = 0;
  snapshot_.setup_complete = true;
  ++snapshot_.setup_successes;
  RefreshResidentSnapshot();
  if (evidence != nullptr) {
    evidence->status = status;
    evidence->reason = reason != nullptr ? *reason : prepare.reason;
  }
  return status;
}

VideoExecutionFrameResult ProductionVideoBackendManager::Execute(
    const hal::HostFrameBinding &binding) noexcept {
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
  if (resident_selected_)
    composition_.Invalidate(reason);
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
  if (!resident_selected_)
    return snapshot_.state ==
                   ProductionVideoBackendManagerState::cpu_primitives_only
               ? hal::StatusId::passthrough
               : hal::StatusId::stale;
  const auto status = composition_.ResetTemporal(reason);
  RefreshResidentSnapshot();
  return status;
}

ProductionVideoSetupLeaseShutdownResult
ProductionVideoBackendManager::Shutdown() noexcept {
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
