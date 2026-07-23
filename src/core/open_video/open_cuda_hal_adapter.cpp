#include "core/open_video/open_cuda_hal_adapter.h"

#include <algorithm>
#include <limits>

namespace studiocast::open_cuda {
namespace {

using namespace studiocast::hal;

void SetReason(ReasonId *out, ReasonId reason) noexcept {
  if (out)
    *out = reason;
}

ExecuteResult Failed(StatusId status, ReasonId reason) noexcept {
  ExecuteResult result;
  result.status = status;
  result.reason = reason;
  return result;
}

bool IsVirtualBackground(SemanticEffectId semantic) noexcept {
  return semantic == SemanticEffectId::virtual_background_blur ||
         semantic == SemanticEffectId::virtual_background_remove ||
         semantic == SemanticEffectId::virtual_background_replace;
}

bool IsSupportedResidentSemantic(SemanticEffectId semantic) noexcept {
  return IsVirtualBackground(semantic) ||
         semantic == SemanticEffectId::virtual_key_light;
}

bool HasIdentity(const std::array<std::uint8_t, 32> &identity) noexcept {
  return std::any_of(identity.begin(), identity.end(),
                     [](std::uint8_t value) { return value != 0; });
}

bool SameModel(const ModelArtifactIdentity &lhs,
               const ModelArtifactIdentity &rhs) noexcept {
  return lhs.model_id_hash == rhs.model_id_hash &&
         lhs.artifact_set_hash == rhs.artifact_set_hash &&
         lhs.artifact_generation == rhs.artifact_generation;
}

bool SamePlane(const PlaneDescriptor &lhs,
               const PlaneDescriptor &rhs) noexcept {
  return lhs.offset_bytes == rhs.offset_bytes &&
         lhs.row_stride_bytes == rhs.row_stride_bytes &&
         lhs.size_bytes == rhs.size_bytes && lhs.width == rhs.width &&
         lhs.height == rhs.height && lhs.alignment_bytes == rhs.alignment_bytes;
}

bool SameImage(const ImageDescriptor &lhs,
               const ImageDescriptor &rhs) noexcept {
  if (lhs.width != rhs.width || lhs.height != rhs.height ||
      lhs.plane_count != rhs.plane_count || lhs.format != rhs.format ||
      lhs.channel_order != rhs.channel_order ||
      lhs.color_space != rhs.color_space ||
      lhs.color_range != rhs.color_range ||
      lhs.alpha_meaning != rhs.alpha_meaning ||
      lhs.orientation != rhs.orientation ||
      lhs.host_coherence != rhs.host_coherence ||
      lhs.mutability != rhs.mutability || lhs.aliasing != rhs.aliasing ||
      lhs.in_place_permitted != rhs.in_place_permitted ||
      lhs.base_alignment_bytes != rhs.base_alignment_bytes) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.planes.size(); ++index) {
    if (!SamePlane(lhs.planes[index], rhs.planes[index]))
      return false;
  }
  return true;
}

bool SameTensor(const TensorDescriptor &lhs,
                const TensorDescriptor &rhs) noexcept {
  return lhs.element_type == rhs.element_type && lhs.layout == rhs.layout &&
         lhs.channel_order == rhs.channel_order &&
         lhs.color_space == rhs.color_space &&
         lhs.color_range == rhs.color_range &&
         lhs.alpha_meaning == rhs.alpha_meaning &&
         lhs.orientation == rhs.orientation &&
         lhs.host_coherence == rhs.host_coherence &&
         lhs.mutability == rhs.mutability && lhs.aliasing == rhs.aliasing &&
         lhs.in_place_permitted == rhs.in_place_permitted &&
         lhs.rank == rhs.rank &&
         lhs.base_alignment_bytes == rhs.base_alignment_bytes &&
         lhs.dimensions == rhs.dimensions &&
         lhs.strides_bytes == rhs.strides_bytes;
}

bool SameIo(const StageIoDescriptor &lhs,
            const StageIoDescriptor &rhs) noexcept {
  return lhs.kind == rhs.kind &&
         (lhs.kind == StageIoKind::image ? SameImage(lhs.image, rhs.image)
                                         : SameTensor(lhs.tensor, rhs.tensor));
}

bool ExactCudaDomain(const ExecutionDomainId &domain,
                     std::uint64_t adapter_id) noexcept {
  return ValidateExecutionDomain(domain) && domain.backend == BackendId::cuda &&
         domain.api_family == ApiFamily::cuda_driver &&
         domain.allocator_family == AllocatorFamily::cuda_driver &&
         domain.memory_handle_family ==
             MemoryHandleFamily::cuda_device_pointer &&
         domain.synchronization_protocol ==
             SynchronizationProtocol::cuda_stream_ordered &&
         domain.ownership_model == OwnershipModel::adapter_exclusive &&
         domain.adapter_instance_id == adapter_id;
}

bool CompleteCapability(const ResidentCapabilityFacts &facts) noexcept {
  return facts.cuda_driver_ready && facts.stable_device_identity &&
         facts.exact_context_generation && facts.adapter_owned_allocator &&
         facts.adapter_owned_stream && facts.ort_cuda_provider_ready &&
         facts.ort_cuda_iobinding_ready &&
         (facts.ort_user_stream_bound ||
          facts.explicit_ort_boundary_synchronization) &&
         facts.sessions_prepared && facts.reusable_buffers_prepared &&
         facts.bounded_resources && facts.steady_state_allocation_free &&
         facts.steady_state_probe_free && facts.no_cpu_fallback &&
         facts.shared_frame_analysis &&
         facts.completion_synchronizations_per_frame != 0 &&
         facts.completion_synchronizations_per_frame <= 4 &&
         facts.device_preprocess && facts.device_postprocess;
}

const PreparedStageEvidence *
FindEvidence(const PreparedExecutorEvidence &evidence,
             std::uint32_t stage_id) noexcept {
  for (std::size_t index = 0; index < evidence.stage_count; ++index) {
    if (evidence.stages[index].stage_id == stage_id)
      return &evidence.stages[index];
  }
  return nullptr;
}

bool BudgetContains(const ResourceBudget &available,
                    const ResourceBudget &required) noexcept {
  return available.persistent_bytes >= required.persistent_bytes &&
         available.scratch_bytes >= required.scratch_bytes &&
         available.upload_staging_bytes >= required.upload_staging_bytes &&
         available.readback_staging_bytes >= required.readback_staging_bytes &&
         available.resource_count >= required.resource_count &&
         available.in_flight_slots >= required.in_flight_slots;
}

bool SameBudget(const ResourceBudget &lhs, const ResourceBudget &rhs) noexcept {
  return lhs.persistent_bytes == rhs.persistent_bytes &&
         lhs.scratch_bytes == rhs.scratch_bytes &&
         lhs.upload_staging_bytes == rhs.upload_staging_bytes &&
         lhs.readback_staging_bytes == rhs.readback_staging_bytes &&
         lhs.resource_count == rhs.resource_count &&
         lhs.in_flight_slots == rhs.in_flight_slots;
}

bool ResourceFitsBoundaryBudget(const ResourceHandle &resource,
                                const PreparedProgram &program,
                                const ResourceBudget &budget,
                                bool input) noexcept {
  if (input) {
    if (program.requires_upload) {
      return resource.memory_domain == MemoryDomain::upload_staging &&
             resource.size_bytes <= budget.upload_staging_bytes;
    }
    return resource.memory_domain == MemoryDomain::device_local &&
           resource.size_bytes <= budget.persistent_bytes;
  }
  if (program.requires_final_readback) {
    return resource.memory_domain == MemoryDomain::readback_staging &&
           resource.size_bytes <= budget.readback_staging_bytes;
  }
  return resource.memory_domain == MemoryDomain::device_local &&
         resource.size_bytes <= budget.persistent_bytes;
}

bool SameResource(const ResourceHandle &lhs,
                  const ResourceHandle &rhs) noexcept {
  return lhs.resource_id == rhs.resource_id &&
         lhs.allocator_owner_id == rhs.allocator_owner_id &&
         lhs.lifetime_generation == rhs.lifetime_generation &&
         lhs.memory_domain == rhs.memory_domain &&
         lhs.lifetime == rhs.lifetime &&
         lhs.synchronization_state == rhs.synchronization_state &&
         lhs.execution_domain == rhs.execution_domain &&
         lhs.size_bytes == rhs.size_bytes &&
         lhs.alignment_bytes == rhs.alignment_bytes &&
         lhs.mutable_resource == rhs.mutable_resource &&
         lhs.aliasable == rhs.aliasable &&
         lhs.adapter_opaque == rhs.adapter_opaque &&
         lhs.destruction_owner == rhs.destruction_owner &&
         lhs.destroy == rhs.destroy;
}

bool MonotonicDelta(std::uint64_t before, std::uint64_t after,
                    std::uint64_t *delta) noexcept {
  if (after < before)
    return false;
  if (delta)
    *delta = after - before;
  return true;
}

bool ValidatePreparedEvidence(const PreparedProgram &program,
                              const ExecutorPrepareResult &prepared,
                              ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (prepared.status != StatusId::ready) {
    SetReason(reason, prepared.reason);
    return false;
  }
  if (!CompleteCapability(prepared.evidence.capability) ||
      prepared.evidence.stage_count != program.stage_count ||
      program.invalidation_key.generations.model == 0 ||
      program.invalidation_key.model_artifact_hash == 0) {
    SetReason(reason, ReasonId::capability_evidence_incomplete);
    return false;
  }
  if (prepared.handle.handle_id == 0 ||
      prepared.handle.program_id != program.program_id ||
      prepared.handle.execution_domain != program.execution_domain ||
      !BudgetContains(program.budget, prepared.handle.reserved_budget) ||
      prepared.handle.reserved_budget.in_flight_slots != 1 ||
      !ValidateResourceHandle(
          prepared.handle.prepared_output, program.execution_domain,
          program.execution_domain.logical_context_generation,
          prepared.handle.reserved_budget) ||
      !ResourceFitsBoundaryBudget(prepared.handle.prepared_output, program,
                                  prepared.handle.reserved_budget, false)) {
    SetReason(reason, ReasonId::malformed_contract);
    return false;
  }

  const ModelArtifactIdentity &island_model = program.stages[0].model;
  for (std::size_t index = 0; index < program.stage_count; ++index) {
    const auto &stage = program.stages[index];
    const auto *evidence = FindEvidence(prepared.evidence, stage.stage_id);
    if (!evidence || evidence->semantic != stage.semantic ||
        evidence->provider != stage.provider ||
        !SameIo(evidence->input, stage.input) ||
        !SameIo(evidence->output, stage.output) ||
        !SameModel(evidence->model, stage.model) ||
        !SameModel(stage.model, island_model) ||
        !evidence->production_qualified || !evidence->cuda_device_input ||
        !evidence->cuda_device_output || !evidence->cuda_iobinding ||
        !evidence->device_preprocess || !evidence->device_postprocess ||
        !evidence->shared_analysis || !evidence->cpu_fallback_disabled) {
      SetReason(reason, ReasonId::capability_evidence_incomplete);
      return false;
    }
    if (program.temporal.temporal &&
        !evidence->temporal_history_device_resident) {
      SetReason(reason, ReasonId::capability_evidence_incomplete);
      return false;
    }
    if (!IsSupportedResidentSemantic(stage.semantic) ||
        stage.provider !=
            SemanticEffectProviderId::open_video_onnxruntime_cuda ||
        !HasIdentity(stage.model.model_id_hash) ||
        !HasIdentity(stage.model.artifact_set_hash) ||
        stage.model.artifact_generation == 0) {
      SetReason(reason, !IsSupportedResidentSemantic(stage.semantic)
                            ? ReasonId::semantic_effect_unsupported
                            : ReasonId::capability_evidence_incomplete);
      return false;
    }
  }

  for (std::size_t index = 0; index < program.cpu_tail_count; ++index) {
    const auto tail = program.cpu_tails[index].id;
    if (tail != CpuTailId::output_format_conversion &&
        tail != CpuTailId::writer_copy) {
      SetReason(reason, ReasonId::cpu_tail_required);
      return false;
    }
  }
  return true;
}

bool ValidateTelemetry(const ExecutorTelemetrySnapshot &before,
                       const ExecutorTelemetrySnapshot &after,
                       const ExecuteResult &result,
                       std::size_t declared_cpu_tails,
                       std::uint32_t expected_synchronizations,
                       bool completed) noexcept {
  std::uint64_t allocation_delta = 0;
  std::uint64_t probe_delta = 0;
  std::uint64_t scan_delta = 0;
  std::uint64_t parse_delta = 0;
  std::uint64_t session_delta = 0;
  std::uint64_t json_delta = 0;
  std::uint64_t uploads = 0;
  std::uint64_t readbacks = 0;
  std::uint64_t intermediate_readbacks = 0;
  std::uint64_t intermediate_uploads = 0;
  std::uint64_t synchronizations = 0;
  std::uint64_t cpu_tails = 0;
  std::uint64_t analyses = 0;
  const bool monotonic =
      MonotonicDelta(before.allocation_requests, after.allocation_requests,
                     &allocation_delta) &&
      MonotonicDelta(before.provider_probes, after.provider_probes,
                     &probe_delta) &&
      MonotonicDelta(before.model_scans, after.model_scans, &scan_delta) &&
      MonotonicDelta(before.manifest_parses, after.manifest_parses,
                     &parse_delta) &&
      MonotonicDelta(before.session_constructions, after.session_constructions,
                     &session_delta) &&
      MonotonicDelta(before.json_serializations, after.json_serializations,
                     &json_delta) &&
      MonotonicDelta(before.uploads, after.uploads, &uploads) &&
      MonotonicDelta(before.final_readbacks, after.final_readbacks,
                     &readbacks) &&
      MonotonicDelta(before.intermediate_readbacks,
                     after.intermediate_readbacks, &intermediate_readbacks) &&
      MonotonicDelta(before.intermediate_uploads, after.intermediate_uploads,
                     &intermediate_uploads) &&
      MonotonicDelta(before.synchronizations, after.synchronizations,
                     &synchronizations) &&
      MonotonicDelta(before.cpu_tail_boundaries, after.cpu_tail_boundaries,
                     &cpu_tails) &&
      MonotonicDelta(before.shared_analysis_runs, after.shared_analysis_runs,
                     &analyses);
  if (!monotonic || allocation_delta != 0 || probe_delta != 0 ||
      scan_delta != 0 || parse_delta != 0 || session_delta != 0 ||
      json_delta != 0 || analyses != (completed ? 1u : 0u)) {
    return false;
  }
  return uploads == result.transfers.uploads &&
         readbacks == result.transfers.final_readbacks &&
         intermediate_readbacks == result.transfers.intermediate_readbacks &&
         intermediate_uploads == result.transfers.intermediate_uploads &&
         synchronizations == result.transfers.synchronizations &&
         (!completed || synchronizations == expected_synchronizations) &&
         cpu_tails == result.transfers.cpu_tail_boundaries &&
         (!completed || cpu_tails == declared_cpu_tails);
}

} // namespace

ResidentSemanticCapability
EvaluateResidentCapability(SemanticEffectId semantic,
                           const ResidentCapabilityFacts &facts) noexcept {
  ResidentSemanticCapability capability;
  capability.semantic = semantic;
  capability.live_stage = IsSupportedResidentSemantic(semantic);
  if (!capability.live_stage) {
    capability.reason = ReasonId::semantic_effect_unsupported;
    return capability;
  }

  const bool semantic_ready = IsVirtualBackground(semantic)
                                  ? facts.virtual_background_composite
                                  : facts.virtual_key_light;
  if (!facts.cuda_driver_ready || !facts.ort_cuda_provider_ready ||
      !facts.ort_cuda_iobinding_ready) {
    capability.reason = ReasonId::semantic_provider_unavailable;
    return capability;
  }
  if (!semantic_ready || !CompleteCapability(facts)) {
    capability.reason = ReasonId::capability_evidence_incomplete;
    return capability;
  }
  capability.status = StatusId::ready;
  capability.reason = ReasonId::none;
  capability.production_eligible = true;
  capability.exact_evidence = true;
  return capability;
}

ResidentAdapter::ResidentAdapter(std::uint64_t adapter_instance_id,
                                 IPreparedResidentExecutor &executor) noexcept
    : adapter_instance_id_(adapter_instance_id), executor_(&executor) {}

ResidentAdapter::~ResidentAdapter() { Shutdown(); }

StatusId ResidentAdapter::Prepare(const PreparedProgram &program,
                                  PreparedProgramHandle *candidate,
                                  ReadinessSnapshot *readiness,
                                  ReasonId *reason) noexcept {
  if (candidate)
    *candidate = {};
  if (readiness)
    *readiness = {};
  SetReason(reason, ReasonId::none);
  if (stopped_ || !executor_) {
    SetReason(reason, ReasonId::shutdown_requested);
    return StatusId::stopped;
  }
  if (!candidate || !readiness || adapter_instance_id_ == 0) {
    SetReason(reason, ReasonId::malformed_contract);
    return StatusId::blocked;
  }

  ReasonId validation_reason = ReasonId::none;
  if (!ValidatePreparedProgram(program, &validation_reason)) {
    SetReason(reason, validation_reason);
    return validation_reason == ReasonId::resource_exhausted
               ? StatusId::resource_exhausted
               : StatusId::blocked;
  }
  if (!ExactCudaDomain(program.execution_domain, adapter_instance_id_) ||
      program.owner_adapter_instance_id != adapter_instance_id_) {
    SetReason(reason, program.execution_domain.adapter_instance_id !=
                              adapter_instance_id_
                          ? ReasonId::foreign_adapter
                          : ReasonId::wrong_execution_domain);
    return StatusId::blocked;
  }

  auto slot = std::find_if(slots_.begin(), slots_.end(),
                           [](const Slot &entry) { return !entry.active; });
  if (slot == slots_.end()) {
    SetReason(reason, ReasonId::resource_exhausted);
    return StatusId::resource_exhausted;
  }

  ExecutorPrepareResult prepared = executor_->Prepare(program);
  ReasonId evidence_reason = ReasonId::none;
  if (!ValidatePreparedEvidence(program, prepared, &evidence_reason)) {
    if (prepared.handle.handle_id != 0)
      executor_->DestroyPrepared(&prepared.handle);
    SetReason(reason, evidence_reason);
    return evidence_reason == ReasonId::resource_exhausted
               ? StatusId::resource_exhausted
               : StatusId::blocked;
  }

  slot->active = true;
  slot->public_handle_id = next_handle_id_++;
  if (slot->public_handle_id == 0)
    slot->public_handle_id = next_handle_id_++;
  slot->program = program;
  slot->executor_handle = prepared.handle;
  slot->requires_explicit_provider_sync =
      !prepared.evidence.capability.ort_user_stream_bound &&
      prepared.evidence.capability.explicit_ort_boundary_synchronization;
  slot->completion_synchronizations_per_frame =
      prepared.evidence.capability.completion_synchronizations_per_frame;

  candidate->handle_id = slot->public_handle_id;
  candidate->program_id = program.program_id;
  candidate->adapter_instance_id = adapter_instance_id_;
  candidate->invalidation_key = program.invalidation_key;
  candidate->execution_domain = program.execution_domain;
  candidate->reserved_budget = prepared.handle.reserved_budget;
  candidate->adapter_opaque[0] = prepared.handle.handle_id;
  candidate->adapter_opaque[1] =
      program.execution_domain.logical_context_generation;

  const bool fallback = program.requested_backend != BackendId::cuda;
  const bool cpu_tail = program.cpu_tail_count != 0;
  readiness->status =
      fallback || cpu_tail ? StatusId::degraded : StatusId::ready;
  readiness->reason =
      fallback ? ReasonId::semantic_provider_unavailable
               : (cpu_tail ? ReasonId::cpu_tail_required : ReasonId::none);
  readiness->requested_backend = program.requested_backend;
  readiness->selected_backend = BackendId::cuda;
  readiness->provider = SemanticEffectProviderId::open_video_onnxruntime_cuda;
  readiness->execution_domain = program.execution_domain;
  readiness->available = true;
  readiness->degraded = readiness->status == StatusId::degraded;
  readiness->ready_stage_mask =
      program.stage_count >= 32
          ? std::numeric_limits<std::uint32_t>::max()
          : ((std::uint32_t{1} << program.stage_count) - 1u);
  SetReason(reason, readiness->reason);
  return readiness->status;
}

ExecuteResult ResidentAdapter::Execute(const PreparedProgramHandle &program,
                                       const ExecuteRequest &request) noexcept {
  if (stopped_ || !executor_)
    return Failed(StatusId::stopped, ReasonId::shutdown_requested);
  Slot *slot = Find(program);
  if (!slot)
    return Failed(StatusId::stale, ReasonId::stale_program);
  if (slot->quarantined)
    return Failed(StatusId::quarantined, ReasonId::device_lost);
  if (request.invalidation_key != slot->program.invalidation_key)
    return Failed(StatusId::stale, ReasonId::reconfiguration_required);
  if (request.execution_domain != slot->program.execution_domain)
    return Failed(StatusId::blocked, ReasonId::wrong_execution_domain);
  if (!ValidateResourceHandle(
          request.input, slot->program.execution_domain,
          slot->program.execution_domain.logical_context_generation,
          slot->program.budget) ||
      !ResourceFitsBoundaryBudget(request.input, slot->program,
                                  slot->executor_handle.reserved_budget,
                                  true)) {
    const ReasonId resource_reason =
        request.input.execution_domain != slot->program.execution_domain
            ? ReasonId::wrong_execution_domain
            : (request.input.lifetime_generation !=
                       slot->program.execution_domain.logical_context_generation
                   ? ReasonId::wrong_generation
                   : ReasonId::invalid_resource);
    return Failed(StatusId::blocked, resource_reason);
  }

  const ExecutorTelemetrySnapshot before = executor_->Telemetry();
  ExecuteResult result = executor_->Execute(slot->executor_handle, request);
  const ExecutorTelemetrySnapshot after = executor_->Telemetry();
  const bool completed = result.status == StatusId::executed ||
                         result.status == StatusId::degraded;

  if (!ValidateTelemetry(before, after, result, slot->program.cpu_tail_count,
                         slot->completion_synchronizations_per_frame,
                         completed)) {
    return Failed(StatusId::failed, ReasonId::synchronization_failed);
  }
  if (result.status != StatusId::executed &&
      result.status != StatusId::degraded) {
    if (result.reason == ReasonId::device_lost) {
      slot->quarantined = true;
      result.status = StatusId::quarantined;
    }
    return result;
  }
  if (result.stage_count != slot->program.stage_count || !result.output_valid ||
      !SameResource(result.output, slot->executor_handle.prepared_output)) {
    return Failed(StatusId::failed, ReasonId::invalid_resource);
  }
  if ((slot->program.requires_upload && result.transfers.uploads != 1) ||
      (!slot->program.requires_upload && result.transfers.uploads != 0) ||
      (slot->program.requires_final_readback &&
       result.transfers.final_readbacks != 1) ||
      (!slot->program.requires_final_readback &&
       result.transfers.final_readbacks != 0) ||
      result.transfers.intermediate_readbacks != 0 ||
      result.transfers.intermediate_uploads != 0 ||
      (slot->requires_explicit_provider_sync &&
       result.transfers.synchronizations == 0)) {
    return Failed(StatusId::failed, ReasonId::synchronization_failed);
  }

  for (std::size_t index = 0; index < result.stage_count; ++index) {
    const auto &expected = slot->program.stages[index];
    const auto &actual = result.stages[index];
    if (actual.stage_id != expected.stage_id ||
        actual.semantic != expected.semantic ||
        actual.status == StageExecutionStatus::not_run ||
        (actual.status == StageExecutionStatus::optional_failed_open &&
         (expected.required || !actual.valid_output_preserved)) ||
        actual.status == StageExecutionStatus::required_failed) {
      return Failed(StatusId::failed, ReasonId::required_stage_failed);
    }
  }

  result.completion.token_id = next_completion_id_++;
  if (result.completion.token_id == 0)
    result.completion.token_id = next_completion_id_++;
  result.completion.program_id = slot->program.program_id;
  result.completion.adapter_instance_id = adapter_instance_id_;
  result.completion.capture_sequence = request.capture_sequence;
  result.completion.complete = true;
  return result;
}

StatusId
ResidentAdapter::ResetTemporalState(const PreparedProgramHandle &program,
                                    ReasonId reason) noexcept {
  if (stopped_ || !executor_)
    return StatusId::stopped;
  Slot *slot = Find(program);
  if (!slot)
    return StatusId::stale;
  if (slot->quarantined)
    return StatusId::quarantined;
  const StatusId status =
      executor_->ResetTemporalState(slot->executor_handle, reason);
  if (status == StatusId::quarantined)
    slot->quarantined = true;
  return status;
}

void ResidentAdapter::DestroyPrepared(PreparedProgramHandle *program) noexcept {
  if (!program)
    return;
  Slot *slot = Find(*program);
  if (slot && executor_) {
    executor_->DestroyPrepared(&slot->executor_handle);
    *slot = {};
  }
  *program = {};
}

void ResidentAdapter::Shutdown() noexcept {
  if (stopped_)
    return;
  if (executor_) {
    for (auto &slot : slots_) {
      if (slot.active)
        executor_->DestroyPrepared(&slot.executor_handle);
      slot = {};
    }
    executor_->Shutdown();
  }
  stopped_ = true;
}

ResidentAdapter::Slot *
ResidentAdapter::Find(const PreparedProgramHandle &program) noexcept {
  for (auto &slot : slots_) {
    if (slot.active && slot.public_handle_id == program.handle_id &&
        slot.program.program_id == program.program_id &&
        program.adapter_instance_id == adapter_instance_id_ &&
        program.adapter_opaque[0] == slot.executor_handle.handle_id &&
        program.adapter_opaque[1] ==
            slot.program.execution_domain.logical_context_generation &&
        program.invalidation_key == slot.program.invalidation_key &&
        program.execution_domain == slot.program.execution_domain &&
        SameBudget(program.reserved_budget,
                   slot.executor_handle.reserved_budget)) {
      return &slot;
    }
  }
  return nullptr;
}

const ResidentAdapter::Slot *
ResidentAdapter::Find(const PreparedProgramHandle &program) const noexcept {
  return const_cast<ResidentAdapter *>(this)->Find(program);
}

} // namespace studiocast::open_cuda
