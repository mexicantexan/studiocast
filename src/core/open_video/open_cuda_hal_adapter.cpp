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
         semantic == SemanticEffectId::virtual_key_light ||
         semantic == SemanticEffectId::vignette;
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

bool ExactRgbImage(const ImageDescriptor &descriptor, HostCoherence coherence,
                   Mutability mutability) noexcept {
  if (!ValidateImageDescriptor(descriptor) || descriptor.plane_count != 1 ||
      descriptor.format != PixelFormat::rgb_u8 ||
      descriptor.channel_order != ChannelOrder::rgb ||
      descriptor.color_space != ColorSpace::srgb ||
      descriptor.color_range != ColorRange::full ||
      descriptor.alpha_meaning != AlphaMeaning::none ||
      descriptor.orientation != Orientation::top_left ||
      descriptor.host_coherence != coherence ||
      descriptor.mutability != mutability ||
      descriptor.aliasing != AliasingPermission::forbidden ||
      descriptor.in_place_permitted) {
    return false;
  }
  const auto &plane = descriptor.planes[0];
  return plane.offset_bytes == 0 && plane.width == descriptor.width &&
         plane.height == descriptor.height &&
         plane.row_stride_bytes >=
             static_cast<std::uint64_t>(descriptor.width) * 3u &&
         plane.row_stride_bytes <=
             std::numeric_limits<std::uint64_t>::max() / plane.height &&
         plane.size_bytes ==
             plane.row_stride_bytes * static_cast<std::uint64_t>(plane.height);
}

bool ExactCudaProgramImages(const PreparedProgram &program) noexcept {
  if (!ExactRgbImage(program.host_input, HostCoherence::coherent,
                     Mutability::read_only) ||
      !ExactRgbImage(program.host_output, HostCoherence::coherent,
                     Mutability::write_only)) {
    return false;
  }
  for (std::size_t index = 0; index < program.stage_count; ++index) {
    const auto &stage = program.stages[index];
    if (stage.input.kind != StageIoKind::image ||
        stage.output.kind != StageIoKind::image ||
        !ExactRgbImage(stage.input.image, HostCoherence::not_host_visible,
                       Mutability::read_only) ||
        !ExactRgbImage(stage.output.image, HostCoherence::not_host_visible,
                       Mutability::read_write)) {
      return false;
    }
  }
  return true;
}

bool CompleteCapability(const ResidentCapabilityFacts &facts) noexcept {
  return facts.cuda_driver_ready && facts.stable_device_identity &&
         facts.exact_context_generation && facts.adapter_owned_allocator &&
         facts.adapter_owned_stream && facts.reusable_buffers_prepared &&
         facts.bounded_resources && facts.steady_state_allocation_free &&
         facts.steady_state_probe_free && facts.no_cpu_fallback &&
         facts.completion_synchronizations_per_frame != 0 &&
         facts.completion_synchronizations_per_frame <= 4 &&
         facts.device_postprocess;
}

bool CompleteOrtCapability(const ResidentCapabilityFacts &facts) noexcept {
  return CompleteCapability(facts) && facts.ort_cuda_provider_ready &&
         facts.ort_cuda_iobinding_ready &&
         (facts.ort_user_stream_bound ||
          facts.explicit_ort_boundary_synchronization) &&
         facts.sessions_prepared && facts.shared_frame_analysis &&
         facts.device_preprocess;
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

bool ResourceCoversImage(const ResourceHandle &resource,
                         const ImageDescriptor &descriptor) noexcept {
  std::uint64_t required_bytes = 0;
  for (std::size_t index = 0; index < descriptor.plane_count; ++index) {
    const auto &plane = descriptor.planes[index];
    if (plane.offset_bytes >
        std::numeric_limits<std::uint64_t>::max() - plane.size_bytes) {
      return false;
    }
    required_bytes =
        std::max(required_bytes, plane.offset_bytes + plane.size_bytes);
  }
  return required_bytes != 0 && resource.size_bytes >= required_bytes &&
         resource.alignment_bytes >= descriptor.base_alignment_bytes;
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

bool SameTelemetry(const ExecutorTelemetrySnapshot &lhs,
                   const ExecutorTelemetrySnapshot &rhs) noexcept {
  return lhs.allocation_requests == rhs.allocation_requests &&
         lhs.provider_probes == rhs.provider_probes &&
         lhs.model_scans == rhs.model_scans &&
         lhs.manifest_parses == rhs.manifest_parses &&
         lhs.session_constructions == rhs.session_constructions &&
         lhs.json_serializations == rhs.json_serializations &&
         lhs.uploads == rhs.uploads &&
         lhs.final_readbacks == rhs.final_readbacks &&
         lhs.intermediate_readbacks == rhs.intermediate_readbacks &&
         lhs.intermediate_uploads == rhs.intermediate_uploads &&
         lhs.synchronizations == rhs.synchronizations &&
         lhs.cpu_tail_boundaries == rhs.cpu_tail_boundaries &&
         lhs.shared_analysis_runs == rhs.shared_analysis_runs;
}

std::uint32_t StageMask(std::size_t stage_count) noexcept {
  return stage_count >= 32 ? std::numeric_limits<std::uint32_t>::max()
                           : ((std::uint32_t{1} << stage_count) - 1u);
}

bool AnyModelStageRan(const ExecuteResult &result) noexcept {
  for (std::size_t index = 0;
       index < result.stage_count && index < result.stages.size(); ++index) {
    const auto status = result.stages[index].status;
    if (result.stages[index].semantic != SemanticEffectId::vignette &&
        status != StageExecutionStatus::not_run &&
        status != StageExecutionStatus::disabled_preserved) {
      return true;
    }
  }
  return false;
}

ReasonId ResourceFailureReason(const ResourceHandle &resource,
                               const PreparedProgram &program) noexcept {
  if (resource.execution_domain != program.execution_domain)
    return ReasonId::wrong_execution_domain;
  if (resource.lifetime_generation !=
      program.execution_domain.logical_context_generation) {
    return ReasonId::wrong_generation;
  }
  return ReasonId::invalid_resource;
}

bool ExactCompletion(const CompletionToken &actual,
                     const CompletionToken &expected) noexcept {
  return actual.token_id == expected.token_id &&
         actual.program_id == expected.program_id &&
         actual.adapter_instance_id == expected.adapter_instance_id &&
         actual.capture_sequence == expected.capture_sequence &&
         actual.complete == expected.complete;
}

bool SameHostPlane(const HostPlaneBinding &lhs,
                   const HostPlaneBinding &rhs) noexcept {
  return lhs.read_data == rhs.read_data && lhs.write_data == rhs.write_data &&
         lhs.span_bytes == rhs.span_bytes &&
         lhs.row_stride_bytes == rhs.row_stride_bytes;
}

bool OneHostResourceAllowed(const HostFrameBinding &binding) noexcept {
  if (!binding.input_descriptor.in_place_permitted ||
      !binding.output_descriptor.in_place_permitted ||
      binding.input_descriptor.aliasing == AliasingPermission::forbidden ||
      binding.output_descriptor.aliasing == AliasingPermission::forbidden ||
      binding.input_plane_count != binding.output_plane_count) {
    return false;
  }
  for (std::size_t index = 0; index < binding.input_plane_count; ++index) {
    const auto &input = binding.input_planes[index];
    const auto &output = binding.output_planes[index];
    if (input.read_data != output.write_data ||
        input.span_bytes != output.span_bytes ||
        input.row_stride_bytes != output.row_stride_bytes) {
      return false;
    }
  }
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
  bool has_model_consumer = false;
  for (std::size_t index = 0; index < program.stage_count; ++index)
    has_model_consumer |= program.stages[index].semantic != SemanticEffectId::vignette;
  if (!CompleteCapability(prepared.evidence.capability) ||
      (has_model_consumer && !CompleteOrtCapability(prepared.evidence.capability)) ||
      prepared.evidence.stage_count != program.stage_count ||
      (has_model_consumer && (program.invalidation_key.generations.model == 0 ||
                              program.invalidation_key.model_artifact_hash == 0))) {
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
        (stage.semantic != SemanticEffectId::vignette &&
         !SameModel(stage.model, island_model)) ||
        !evidence->production_qualified || !evidence->cuda_device_input ||
        !evidence->cuda_device_output ||
        (stage.semantic != SemanticEffectId::vignette &&
         (!evidence->cuda_iobinding || !evidence->device_preprocess ||
          !evidence->shared_analysis)) ||
        !evidence->device_postprocess || !evidence->cpu_fallback_disabled) {
      SetReason(reason, ReasonId::capability_evidence_incomplete);
      return false;
    }
    if (program.temporal.temporal &&
        !evidence->temporal_history_device_resident) {
      SetReason(reason, ReasonId::capability_evidence_incomplete);
      return false;
    }
    if (!IsSupportedResidentSemantic(stage.semantic) ||
        (stage.semantic == SemanticEffectId::vignette
             ? stage.provider != SemanticEffectProviderId::open_cuda_production_kernels ||
                   HasIdentity(stage.model.model_id_hash) ||
                   HasIdentity(stage.model.artifact_set_hash) ||
                   stage.model.artifact_generation != 0
             : stage.provider != SemanticEffectProviderId::open_video_onnxruntime_cuda ||
                   !HasIdentity(stage.model.model_id_hash) ||
                   !HasIdentity(stage.model.artifact_set_hash) ||
                   stage.model.artifact_generation == 0)) {
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
                       std::uint32_t expected_synchronizations, bool completed,
                       bool analysis_ran) noexcept {
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
      json_delta != 0 || analyses != (analysis_ran ? 1u : 0u)) {
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

  bool semantic_ready = false;
  switch (semantic) {
  case SemanticEffectId::virtual_background_blur:
    semantic_ready = facts.virtual_background_blur;
    break;
  case SemanticEffectId::virtual_background_remove:
    semantic_ready = facts.virtual_background_remove;
    break;
  case SemanticEffectId::virtual_background_replace:
    semantic_ready = facts.virtual_background_replace;
    break;
  case SemanticEffectId::virtual_key_light:
    semantic_ready = facts.virtual_key_light;
    break;
  case SemanticEffectId::vignette:
    semantic_ready = facts.vignette && facts.native_vignette_kernel_prepared;
    capability.provider = SemanticEffectProviderId::open_cuda_production_kernels;
    break;
  case SemanticEffectId::video_noise_removal:
  case SemanticEffectId::eye_contact:
  case SemanticEffectId::auto_frame:
  case SemanticEffectId::mirror:
  case SemanticEffectId::invalid:
    break;
  }
  const bool native = semantic == SemanticEffectId::vignette;
  if (!facts.cuda_driver_ready ||
      (!native && (!facts.ort_cuda_provider_ready ||
                   !facts.ort_cuda_iobinding_ready))) {
    capability.reason = ReasonId::semantic_provider_unavailable;
    return capability;
  }
  if (!semantic_ready || !CompleteCapability(facts) ||
      (!native && !CompleteOrtCapability(facts))) {
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
  if (!ExactCudaProgramImages(program)) {
    SetReason(reason, ReasonId::malformed_descriptor);
    return StatusId::blocked;
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
  const bool vignette_last =
      program.stages[program.stage_count - 1].semantic ==
      SemanticEffectId::vignette;
  const bool has_model_consumer =
      program.stage_count > (vignette_last ? 1u : 0u);
  readiness->provider =
      has_model_consumer && vignette_last
          ? SemanticEffectProviderId::invalid
          : has_model_consumer ? SemanticEffectProviderId::open_video_onnxruntime_cuda
                               : SemanticEffectProviderId::open_cuda_production_kernels;
  readiness->execution_domain = program.execution_domain;
  readiness->available = true;
  readiness->degraded = readiness->status == StatusId::degraded;
  readiness->ready_stage_mask = StageMask(program.stage_count);
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
  const std::uint32_t valid_stage_mask = StageMask(slot->program.stage_count);
  if ((request.disabled_stage_mask & ~valid_stage_mask) != 0)
    return Failed(StatusId::blocked, ReasonId::malformed_contract);
  for (std::size_t index = 0; index < slot->program.stage_count; ++index) {
    if ((request.disabled_stage_mask & (std::uint32_t{1} << index)) != 0 &&
        slot->program.stages[index].required) {
      return Failed(StatusId::failed, ReasonId::required_stage_failed);
    }
  }
  if (!ValidateResourceHandle(
          request.input, slot->program.execution_domain,
          slot->program.execution_domain.logical_context_generation,
          slot->program.budget) ||
      !ResourceFitsBoundaryBudget(request.input, slot->program,
                                  slot->executor_handle.reserved_budget,
                                  true)) {
    return Failed(StatusId::blocked,
                  ResourceFailureReason(request.input, slot->program));
  }
  if (slot->host_binding_active &&
      (slot->host_execution_completed ||
       request.capture_sequence != slot->host_binding.capture_sequence ||
       !SameResource(request.input, slot->host_tokens.input))) {
    slot->host_binding_active = false;
    slot->host_execution_completed = false;
    return Failed(StatusId::blocked, ReasonId::host_frame_binding_invalid);
  }

  const ExecutorTelemetrySnapshot before = executor_->Telemetry();
  ExecuteResult result = executor_->Execute(slot->executor_handle, request);
  const ExecutorTelemetrySnapshot after = executor_->Telemetry();
  const bool completed = result.status == StatusId::executed ||
                         result.status == StatusId::degraded;

  if (!ValidateTelemetry(before, after, result, slot->program.cpu_tail_count,
                         slot->completion_synchronizations_per_frame, completed,
                         AnyModelStageRan(result))) {
    slot->host_binding_active = false;
    slot->host_execution_completed = false;
    return Failed(StatusId::failed, ReasonId::synchronization_failed);
  }
  if (result.transfers.intermediate_readbacks != 0 ||
      result.transfers.intermediate_uploads != 0 ||
      result.transfers.uploads > (slot->program.requires_upload ? 1u : 0u) ||
      result.transfers.final_readbacks >
          (slot->program.requires_final_readback ? 1u : 0u) ||
      result.transfers.synchronizations >
          slot->completion_synchronizations_per_frame ||
      result.transfers.cpu_tail_boundaries > slot->program.cpu_tail_count) {
    slot->host_binding_active = false;
    slot->host_execution_completed = false;
    return Failed(StatusId::failed, ReasonId::synchronization_failed);
  }

  if (result.stage_count > slot->program.stage_count) {
    slot->host_binding_active = false;
    slot->host_execution_completed = false;
    return Failed(StatusId::failed, ReasonId::malformed_contract);
  }
  for (std::size_t index = 0; index < result.stage_count; ++index) {
    const auto &expected = slot->program.stages[index];
    const auto &actual = result.stages[index];
    const bool disabled =
        (request.disabled_stage_mask & (std::uint32_t{1} << index)) != 0;
    if (actual.stage_id != expected.stage_id ||
        actual.semantic != expected.semantic ||
        actual.primitive != expected.primitive ||
        (completed && actual.status == StageExecutionStatus::not_run) ||
        (disabled &&
         (actual.status != StageExecutionStatus::disabled_preserved ||
          actual.reason != ReasonId::circuit_breaker_open ||
          !actual.valid_output_preserved || actual.elapsed_ns != 0)) ||
        (!disabled &&
         actual.status == StageExecutionStatus::disabled_preserved) ||
        (actual.status == StageExecutionStatus::optional_failed_open &&
         (expected.required || !actual.valid_output_preserved)) ||
        (actual.status == StageExecutionStatus::required_failed &&
         !expected.required)) {
      slot->host_binding_active = false;
      slot->host_execution_completed = false;
      return Failed(StatusId::failed, ReasonId::required_stage_failed);
    }
  }

  if (!completed) {
    slot->host_binding_active = false;
    slot->host_execution_completed = false;
    if (result.reason == ReasonId::device_lost) {
      slot->quarantined = true;
      result.status = StatusId::quarantined;
    }
    return result;
  }
  if (result.stage_count != slot->program.stage_count || !result.output_valid) {
    slot->host_binding_active = false;
    slot->host_execution_completed = false;
    return Failed(StatusId::failed, ReasonId::invalid_resource);
  }
  const ResourceHandle &expected_output =
      slot->host_binding_active ? slot->host_tokens.output
                                : slot->executor_handle.prepared_output;
  if (!SameResource(result.output, expected_output)) {
    slot->host_binding_active = false;
    slot->host_execution_completed = false;
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
    slot->host_binding_active = false;
    slot->host_execution_completed = false;
    return Failed(StatusId::failed, ReasonId::synchronization_failed);
  }

  result.completion.token_id = next_completion_id_++;
  if (result.completion.token_id == 0)
    result.completion.token_id = next_completion_id_++;
  result.completion.program_id = slot->program.program_id;
  result.completion.adapter_instance_id = adapter_instance_id_;
  result.completion.capture_sequence = request.capture_sequence;
  result.completion.complete = true;
  if (slot->host_binding_active) {
    slot->host_completion = result.completion;
    slot->host_execution_completed = true;
  }
  return result;
}

StatusId ResidentAdapter::BindHostFrame(const PreparedProgramHandle &program,
                                        const HostFrameBinding &binding,
                                        HostFrameTokens *tokens,
                                        ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (tokens)
    *tokens = {};
  if (stopped_ || !executor_) {
    SetReason(reason, ReasonId::shutdown_requested);
    return StatusId::stopped;
  }
  Slot *slot = Find(program);
  if (!slot) {
    SetReason(reason, ReasonId::stale_program);
    return StatusId::stale;
  }
  if (slot->quarantined) {
    SetReason(reason, ReasonId::device_lost);
    return StatusId::quarantined;
  }
  if (!tokens) {
    SetReason(reason, ReasonId::malformed_contract);
    return StatusId::blocked;
  }
  ReasonId validation_reason = ReasonId::none;
  if (!ValidateHostFrameBinding(binding, slot->program, &validation_reason)) {
    SetReason(reason, validation_reason);
    return StatusId::blocked;
  }
  if (slot->host_binding_active && !slot->host_execution_completed) {
    SetReason(reason, ReasonId::output_not_committed);
    return StatusId::blocked;
  }

  HostFrameTokens candidate;
  const ExecutorTelemetrySnapshot before = executor_->Telemetry();
  const StatusId status = executor_->BindHostFrame(slot->executor_handle,
                                                   binding, &candidate, reason);
  const ExecutorTelemetrySnapshot after = executor_->Telemetry();
  if (!SameTelemetry(before, after)) {
    SetReason(reason, ReasonId::synchronization_failed);
    return StatusId::failed;
  }
  if (status != StatusId::ready)
    return status;

  const bool input_valid =
      ValidateResourceHandle(
          candidate.input, slot->program.execution_domain,
          slot->program.execution_domain.logical_context_generation,
          slot->executor_handle.reserved_budget) &&
      candidate.input.lifetime == ResourceLifetime::frame_borrowed &&
      ResourceFitsBoundaryBudget(candidate.input, slot->program,
                                 slot->executor_handle.reserved_budget, true) &&
      ResourceCoversImage(candidate.input, binding.input_descriptor);
  const bool output_valid =
      ValidateResourceHandle(
          candidate.output, slot->program.execution_domain,
          slot->program.execution_domain.logical_context_generation,
          slot->executor_handle.reserved_budget) &&
      candidate.output.lifetime == ResourceLifetime::frame_borrowed &&
      ResourceFitsBoundaryBudget(candidate.output, slot->program,
                                 slot->executor_handle.reserved_budget,
                                 false) &&
      ResourceCoversImage(candidate.output, binding.output_descriptor);
  if (!input_valid || !output_valid ||
      (SameResource(candidate.input, candidate.output) &&
       !OneHostResourceAllowed(binding))) {
    const ReasonId token_reason =
        !input_valid
            ? ResourceFailureReason(candidate.input, slot->program)
            : (!output_valid
                   ? ResourceFailureReason(candidate.output, slot->program)
                   : ReasonId::host_frame_binding_invalid);
    SetReason(reason, token_reason);
    return StatusId::blocked;
  }

  slot->host_binding = binding;
  slot->host_tokens = candidate;
  slot->host_completion = {};
  slot->host_binding_active = true;
  slot->host_execution_completed = false;
  *tokens = candidate;
  SetReason(reason, ReasonId::none);
  return StatusId::ready;
}

StatusId ResidentAdapter::ResolveHostFrameOutput(
    const PreparedProgramHandle &program, const CompletionToken &completion,
    const ResourceHandle &output, HostFrameOutputResolution *resolution,
    ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (resolution)
    *resolution = {};
  if (stopped_ || !executor_) {
    SetReason(reason, ReasonId::shutdown_requested);
    return StatusId::stopped;
  }
  Slot *slot = Find(program);
  if (!slot) {
    SetReason(reason, ReasonId::stale_program);
    return StatusId::stale;
  }
  if (slot->quarantined) {
    SetReason(reason, ReasonId::device_lost);
    return StatusId::quarantined;
  }
  if (!resolution || !slot->host_binding_active ||
      !slot->host_execution_completed ||
      !ExactCompletion(completion, slot->host_completion)) {
    SetReason(reason, ReasonId::output_not_committed);
    return StatusId::blocked;
  }
  if (!ValidateResourceHandle(
          output, slot->program.execution_domain,
          slot->program.execution_domain.logical_context_generation,
          slot->executor_handle.reserved_budget) ||
      output.lifetime != ResourceLifetime::frame_borrowed ||
      !ResourceFitsBoundaryBudget(output, slot->program,
                                  slot->executor_handle.reserved_budget,
                                  false)) {
    SetReason(reason, ResourceFailureReason(output, slot->program));
    return StatusId::blocked;
  }
  if (!SameResource(output, slot->host_tokens.output)) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return StatusId::blocked;
  }

  HostFrameOutputResolution candidate;
  const ExecutorTelemetrySnapshot before = executor_->Telemetry();
  const StatusId status = executor_->ResolveHostFrameOutput(
      slot->executor_handle, completion, output, &candidate, reason);
  const ExecutorTelemetrySnapshot after = executor_->Telemetry();
  if (!SameTelemetry(before, after)) {
    SetReason(reason, ReasonId::synchronization_failed);
    return StatusId::failed;
  }
  if (status != StatusId::ready)
    return status;
  if (!candidate.output_ready ||
      candidate.capture_sequence != slot->host_binding.capture_sequence ||
      candidate.plane_count != slot->host_binding.output_plane_count ||
      !SameImage(candidate.descriptor, slot->host_binding.output_descriptor)) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return StatusId::blocked;
  }
  for (std::size_t index = 0; index < candidate.plane_count; ++index) {
    if (!SameHostPlane(candidate.planes[index],
                       slot->host_binding.output_planes[index])) {
      SetReason(reason, ReasonId::host_frame_binding_invalid);
      return StatusId::blocked;
    }
  }
  *resolution = candidate;
  SetReason(reason, ReasonId::none);
  return StatusId::ready;
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
