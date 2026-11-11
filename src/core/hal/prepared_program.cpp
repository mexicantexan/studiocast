#include "core/hal/prepared_program.h"

#include <algorithm>
#include <limits>

namespace studiocast::hal {
namespace {

bool IsPowerOfTwo(std::uint64_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

bool HasStableDeviceId(const PhysicalDeviceIdentity &device) noexcept {
  return std::any_of(device.stable_id.begin(), device.stable_id.end(),
                     [](std::uint8_t value) { return value != 0; });
}

bool HasConsistentExecutionFamilies(const ExecutionDomainId &domain) noexcept {
  switch (domain.api_family) {
  case ApiFamily::host:
    return domain.allocator_family == AllocatorFamily::host &&
           domain.memory_handle_family == MemoryHandleFamily::host_pointer &&
           domain.synchronization_protocol ==
               SynchronizationProtocol::host_synchronous;
  case ApiFamily::cuda_driver:
    return domain.allocator_family == AllocatorFamily::cuda_driver &&
           domain.memory_handle_family ==
               MemoryHandleFamily::cuda_device_pointer &&
           domain.synchronization_protocol ==
               SynchronizationProtocol::cuda_stream_ordered;
  case ApiFamily::maxine_nvcv:
    return domain.allocator_family == AllocatorFamily::maxine_nvcv &&
           domain.memory_handle_family == MemoryHandleFamily::nvcv_image &&
           domain.synchronization_protocol ==
               SynchronizationProtocol::maxine_bound_cuda_stream;
  case ApiFamily::onnxruntime:
    return domain.allocator_family == AllocatorFamily::onnxruntime &&
           domain.memory_handle_family == MemoryHandleFamily::ort_value &&
           domain.synchronization_protocol ==
               SynchronizationProtocol::ort_provider_synchronous;
  case ApiFamily::vulkan:
    return domain.allocator_family == AllocatorFamily::vulkan &&
           (domain.memory_handle_family == MemoryHandleFamily::vulkan_buffer ||
            domain.memory_handle_family == MemoryHandleFamily::vulkan_image) &&
           domain.synchronization_protocol ==
               SynchronizationProtocol::vulkan_queue_fence;
  case ApiFamily::invalid:
    return false;
  }
  return false;
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

std::uint64_t BoundaryBudget(const ResourceBudget &budget,
                             MemoryDomain domain) noexcept {
  switch (domain) {
  case MemoryDomain::upload_staging:
    return budget.upload_staging_bytes;
  case MemoryDomain::readback_staging:
    return budget.readback_staging_bytes;
  case MemoryDomain::scratch:
    return budget.scratch_bytes;
  case MemoryDomain::device_local:
  case MemoryDomain::provider_owned:
    return budget.persistent_bytes;
  case MemoryDomain::host:
    return budget.persistent_bytes;
  case MemoryDomain::invalid:
    return 0;
  }
  return 0;
}

bool SameImageLayout(const ImageDescriptor &left,
                     const ImageDescriptor &right) noexcept;

bool SameStageIoDescriptor(const StageIoDescriptor &left,
                           const StageIoDescriptor &right) noexcept {
  if (left.kind != right.kind)
    return false;
  if (left.kind == StageIoKind::image)
    return SameImageLayout(left.image, right.image);
  return left.tensor.element_type == right.tensor.element_type &&
         left.tensor.layout == right.tensor.layout &&
         left.tensor.channel_order == right.tensor.channel_order &&
         left.tensor.color_space == right.tensor.color_space &&
         left.tensor.color_range == right.tensor.color_range &&
         left.tensor.alpha_meaning == right.tensor.alpha_meaning &&
         left.tensor.orientation == right.tensor.orientation &&
         left.tensor.host_coherence == right.tensor.host_coherence &&
         left.tensor.rank == right.tensor.rank &&
         left.tensor.base_alignment_bytes ==
             right.tensor.base_alignment_bytes &&
         left.tensor.dimensions == right.tensor.dimensions &&
         left.tensor.strides_bytes == right.tensor.strides_bytes;
}

bool SameImageLayout(const ImageDescriptor &left,
                     const ImageDescriptor &right) noexcept {
  if (left.width != right.width || left.height != right.height ||
      left.plane_count != right.plane_count || left.format != right.format ||
      left.channel_order != right.channel_order ||
      left.color_space != right.color_space ||
      left.color_range != right.color_range ||
      left.alpha_meaning != right.alpha_meaning ||
      left.orientation != right.orientation ||
      left.host_coherence != right.host_coherence ||
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

bool SameImageDescriptor(const ImageDescriptor &left,
                         const ImageDescriptor &right) noexcept {
  return SameImageLayout(left, right) && left.mutability == right.mutability &&
         left.aliasing == right.aliasing &&
         left.in_place_permitted == right.in_place_permitted;
}

bool ValidHostBoundaryDescriptor(const ImageDescriptor &descriptor,
                                 Mutability mutability) noexcept {
  return ValidateImageDescriptor(descriptor) &&
         descriptor.host_coherence == HostCoherence::coherent &&
         descriptor.mutability == mutability &&
         descriptor.aliasing == AliasingPermission::forbidden &&
         !descriptor.in_place_permitted;
}

std::uint64_t RequiredImageSpan(const ImageDescriptor &descriptor) noexcept {
  std::uint64_t required = 0;
  for (std::size_t index = 0; index < descriptor.plane_count; ++index) {
    const auto &plane = descriptor.planes[index];
    if (plane.offset_bytes >
        std::numeric_limits<std::uint64_t>::max() - plane.size_bytes) {
      return 0;
    }
    required = std::max(required, plane.offset_bytes + plane.size_bytes);
  }
  return required;
}

bool SameResourceToken(const ResourceHandle &left,
                       const ResourceHandle &right) noexcept {
  return left.resource_id == right.resource_id &&
         left.allocator_owner_id == right.allocator_owner_id &&
         left.lifetime_generation == right.lifetime_generation &&
         left.memory_domain == right.memory_domain &&
         left.lifetime == right.lifetime &&
         left.synchronization_state == right.synchronization_state &&
         left.execution_domain == right.execution_domain &&
         left.size_bytes == right.size_bytes &&
         left.alignment_bytes == right.alignment_bytes &&
         left.mutable_resource == right.mutable_resource &&
         left.aliasable == right.aliasable &&
         left.adapter_opaque == right.adapter_opaque;
}

bool ValidCapabilityEvidence(
    const CapabilityEvidenceRecord &evidence) noexcept {
  if (evidence.code == CapabilityEvidenceCode::invalid ||
      evidence.status == StatusId::invalid ||
      evidence.backend == BackendId::invalid) {
    return false;
  }
  if (evidence.satisfied)
    return (evidence.status == StatusId::ready &&
            evidence.reason == ReasonId::none) ||
           (evidence.status == StatusId::degraded &&
            evidence.reason != ReasonId::none);
  return (evidence.status == StatusId::blocked ||
          evidence.status == StatusId::unavailable) &&
         evidence.reason != ReasonId::none;
}

bool ValidReadinessEvidence(const ReadinessSnapshot &readiness) noexcept {
  if (readiness.evidence_count > readiness.evidence.size())
    return false;
  for (std::size_t index = 0; index < readiness.evidence_count; ++index) {
    if (!ValidCapabilityEvidence(readiness.evidence[index]))
      return false;
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (readiness.evidence[previous].code == readiness.evidence[index].code &&
          readiness.evidence[previous].semantic ==
              readiness.evidence[index].semantic) {
        return false;
      }
    }
  }
  return true;
}

void DestroyResource(ResourceHandle *resource) noexcept {
  if (resource != nullptr && resource->destroy != nullptr)
    resource->destroy(resource->destruction_owner, resource);
  if (resource != nullptr)
    *resource = {};
}

ExecuteResult FailedExecute(StatusId status, ReasonId reason) noexcept {
  ExecuteResult result;
  result.status = status;
  result.reason = reason;
  return result;
}

bool IsSuccessfulExecution(StatusId status) noexcept {
  return status == StatusId::executed || status == StatusId::degraded ||
         status == StatusId::passthrough;
}

const PreparedStage *FindStage(const PreparedProgram &program,
                               std::uint32_t stage_id) noexcept {
  for (std::size_t index = 0; index < program.stage_count; ++index) {
    if (program.stages[index].stage_id == stage_id)
      return &program.stages[index];
  }
  return nullptr;
}

void SetReason(ReasonId *reason, ReasonId value) noexcept {
  if (reason)
    *reason = value;
}

} // namespace

bool ValidateImageDescriptor(const ImageDescriptor &descriptor) noexcept {
  if (descriptor.width == 0 || descriptor.height == 0 ||
      descriptor.plane_count == 0 ||
      descriptor.plane_count > descriptor.planes.size() ||
      descriptor.format == PixelFormat::invalid ||
      !IsPowerOfTwo(descriptor.base_alignment_bytes)) {
    return false;
  }

  for (std::size_t index = 0; index < descriptor.plane_count; ++index) {
    const auto &plane = descriptor.planes[index];
    if (plane.width == 0 || plane.height == 0 || plane.size_bytes == 0 ||
        plane.row_stride_bytes == 0 || !IsPowerOfTwo(plane.alignment_bytes) ||
        plane.alignment_bytes > descriptor.base_alignment_bytes ||
        plane.offset_bytes >
            std::numeric_limits<std::uint64_t>::max() - plane.size_bytes) {
      return false;
    }
  }
  return true;
}

bool ValidateTensorDescriptor(const TensorDescriptor &descriptor) noexcept {
  if (descriptor.element_type == TensorElementType::invalid ||
      descriptor.layout == TensorLayout::invalid || descriptor.rank == 0 ||
      descriptor.rank > descriptor.dimensions.size() ||
      !IsPowerOfTwo(descriptor.base_alignment_bytes)) {
    return false;
  }

  for (std::size_t index = 0; index < descriptor.rank; ++index) {
    if (descriptor.dimensions[index] == 0 ||
        descriptor.strides_bytes[index] == 0) {
      return false;
    }
  }
  return true;
}

bool ValidateExecutionDomain(const ExecutionDomainId &domain) noexcept {
  return domain.backend != BackendId::invalid &&
         domain.api_family != ApiFamily::invalid &&
         HasStableDeviceId(domain.physical_device) &&
         domain.logical_context_generation != 0 &&
         domain.allocator_family != AllocatorFamily::invalid &&
         domain.memory_handle_family != MemoryHandleFamily::invalid &&
         domain.queue_or_stream_identity != 0 &&
         domain.ownership_model != OwnershipModel::invalid &&
         domain.synchronization_protocol != SynchronizationProtocol::invalid &&
         domain.adapter_instance_id != 0 &&
         HasConsistentExecutionFamilies(domain);
}

bool ValidateResourceHandle(const ResourceHandle &resource,
                            const ExecutionDomainId &expected_domain,
                            std::uint64_t expected_lifetime_generation,
                            const ResourceBudget &budget) noexcept {
  const std::uint64_t boundary_budget =
      BoundaryBudget(budget, resource.memory_domain);

  return resource.resource_id != 0 && resource.allocator_owner_id != 0 &&
         resource.allocator_owner_id == expected_domain.adapter_instance_id &&
         resource.lifetime_generation == expected_lifetime_generation &&
         resource.memory_domain != MemoryDomain::invalid &&
         resource.lifetime != ResourceLifetime::invalid &&
         resource.synchronization_state == SynchronizationState::ready &&
         resource.execution_domain == expected_domain &&
         resource.size_bytes != 0 && resource.size_bytes <= boundary_budget &&
         IsPowerOfTwo(resource.alignment_bytes);
}

bool ValidatePreparedProgram(const PreparedProgram &program,
                             ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (program.contract_version != kContractVersion) {
    SetReason(reason, ReasonId::unsupported_contract_version);
    return false;
  }
  if (program.program_id == 0 || program.owner_adapter_instance_id == 0 ||
      program.stage_count == 0 || program.stage_count > program.stages.size() ||
      program.cpu_tail_count > program.cpu_tails.size() ||
      (program.setup_failure_policy != SetupFailurePolicy::retain_old_program &&
       program.setup_failure_policy != SetupFailurePolicy::stop_pipeline) ||
      (program.temporal.frame_drop_behavior !=
           FrameDropBehavior::latest_frame_wins &&
       program.temporal.frame_drop_behavior !=
           FrameDropBehavior::preserve_order &&
       program.temporal.frame_drop_behavior !=
           FrameDropBehavior::reset_temporal_state) ||
      (program.temporal.discontinuity_action !=
           TemporalDiscontinuityAction::reset_and_continue &&
       program.temporal.discontinuity_action !=
           TemporalDiscontinuityAction::fail_frame) ||
      !ValidateExecutionDomain(program.execution_domain)) {
    SetReason(reason, ReasonId::malformed_contract);
    return false;
  }
  if (program.owner_adapter_instance_id !=
      program.execution_domain.adapter_instance_id) {
    SetReason(reason, ReasonId::cross_adapter_residency_forbidden);
    return false;
  }
  if (program.requested_backend == BackendId::invalid) {
    SetReason(reason, ReasonId::malformed_contract);
    return false;
  }
  if (program.strict_backend_selection &&
      program.requested_backend != program.execution_domain.backend) {
    SetReason(reason, ReasonId::strict_backend_unavailable);
    return false;
  }
  if (program.strict_provider_selection &&
      program.requested_provider == SemanticEffectProviderId::invalid) {
    SetReason(reason, ReasonId::malformed_contract);
    return false;
  }
  if (program.strict_execution_domain &&
      (!ValidateExecutionDomain(program.requested_execution_domain) ||
       program.requested_execution_domain != program.execution_domain)) {
    SetReason(reason, ReasonId::wrong_execution_domain);
    return false;
  }
  if (program.invalidation_key.generations.configuration == 0 ||
      program.invalidation_key.generations.geometry == 0 ||
      program.invalidation_key.generations.provider == 0 ||
      program.invalidation_key.generations.device == 0 ||
      program.invalidation_key.generations.context !=
          program.execution_domain.logical_context_generation ||
      program.invalidation_key.canonical_plan_hash == 0 ||
      program.invalidation_key.execution_domain_hash == 0 ||
      program.budget.resource_count == 0 ||
      program.budget.in_flight_slots != 1) {
    SetReason(reason, ReasonId::malformed_contract);
    return false;
  }

  const std::uint64_t host_input_bytes = RequiredImageSpan(program.host_input);
  const std::uint64_t host_output_bytes =
      RequiredImageSpan(program.host_output);
  const std::uint64_t input_boundary_budget =
      program.requires_upload ? program.budget.upload_staging_bytes
                              : program.budget.persistent_bytes;
  const std::uint64_t output_boundary_budget =
      program.requires_final_readback ? program.budget.readback_staging_bytes
                                      : program.budget.persistent_bytes;
  if (!ValidHostBoundaryDescriptor(program.host_input, Mutability::read_only) ||
      !ValidHostBoundaryDescriptor(program.host_output,
                                   Mutability::write_only) ||
      host_input_bytes == 0 || host_output_bytes == 0 ||
      host_input_bytes > input_boundary_budget ||
      host_output_bytes > output_boundary_budget) {
    SetReason(reason, host_input_bytes > input_boundary_budget ||
                              host_output_bytes > output_boundary_budget
                          ? ReasonId::resource_exhausted
                          : ReasonId::malformed_descriptor);
    return false;
  }

  std::uint32_t previous_stage_id = 0;
  std::uint64_t stage_scratch = 0;
  for (std::size_t index = 0; index < program.stage_count; ++index) {
    const auto &stage = program.stages[index];
    const bool semantic_stage =
        stage.semantic != SemanticEffectId::invalid &&
        stage.primitive == PrimitiveOperationId::invalid;
    const bool primitive_stage =
        stage.semantic == SemanticEffectId::invalid &&
        stage.primitive != PrimitiveOperationId::invalid;
    if (stage.stage_id == 0 || stage.stage_id <= previous_stage_id ||
        (!semantic_stage && !primitive_stage) ||
        (semantic_stage &&
         stage.provider == SemanticEffectProviderId::invalid) ||
        (primitive_stage &&
         stage.provider != SemanticEffectProviderId::invalid) ||
        stage.backend == BackendId::invalid ||
        stage.execution_domain != program.execution_domain ||
        stage.backend != program.execution_domain.backend) {
      SetReason(reason, stage.execution_domain.adapter_instance_id !=
                                program.owner_adapter_instance_id
                            ? ReasonId::cross_adapter_residency_forbidden
                            : ReasonId::malformed_contract);
      return false;
    }
    if (program.strict_provider_selection && semantic_stage &&
        stage.provider != program.requested_provider) {
      SetReason(reason, ReasonId::semantic_provider_unavailable);
      return false;
    }
    if ((stage.input.kind == StageIoKind::image &&
         !ValidateImageDescriptor(stage.input.image)) ||
        (stage.input.kind == StageIoKind::tensor &&
         !ValidateTensorDescriptor(stage.input.tensor)) ||
        (stage.output.kind == StageIoKind::image &&
         !ValidateImageDescriptor(stage.output.image)) ||
        (stage.output.kind == StageIoKind::tensor &&
         !ValidateTensorDescriptor(stage.output.tensor)) ||
        (!stage.required &&
         stage.failure_action == FailureAction::fail_closed) ||
        (stage.required &&
         stage.failure_action != FailureAction::fail_closed) ||
        (!stage.required &&
         stage.failure_action != FailureAction::fail_open_preserve_input &&
         stage.failure_action !=
             FailureAction::fail_open_preserve_last_valid) ||
        stage.circuit_breaker.consecutive_failure_limit == 0 ||
        (stage.circuit_breaker.action != CircuitBreakerAction::disable_stage &&
         stage.circuit_breaker.action !=
             CircuitBreakerAction::quarantine_program &&
         stage.circuit_breaker.action !=
             CircuitBreakerAction::request_rebuild) ||
        (stage.required &&
         stage.circuit_breaker.action == CircuitBreakerAction::disable_stage) ||
        (index != 0 && !SameStageIoDescriptor(program.stages[index - 1].output,
                                              stage.input))) {
      SetReason(reason, ReasonId::malformed_descriptor);
      return false;
    }
    if (stage_scratch >
        std::numeric_limits<std::uint64_t>::max() - stage.scratch_bytes) {
      SetReason(reason, ReasonId::resource_exhausted);
      return false;
    }
    stage_scratch += stage.scratch_bytes;
    previous_stage_id = stage.stage_id;
  }
  if (stage_scratch > program.budget.scratch_bytes) {
    SetReason(reason, ReasonId::resource_exhausted);
    return false;
  }
  const auto &first_input = program.stages[0].input;
  const auto &last_output = program.stages[program.stage_count - 1].output;
  if ((first_input.kind == StageIoKind::image &&
       (program.host_input.width != first_input.image.width ||
        program.host_input.height != first_input.image.height ||
        program.host_input.orientation != first_input.image.orientation)) ||
      (last_output.kind == StageIoKind::image &&
       (program.host_output.width != last_output.image.width ||
        program.host_output.height != last_output.image.height ||
        program.host_output.orientation != last_output.image.orientation))) {
    SetReason(reason, ReasonId::malformed_descriptor);
    return false;
  }
  std::uint32_t previous_boundary_index = 0;
  bool have_boundary = false;
  std::uint64_t cpu_tail_scratch = 0;
  for (std::size_t index = 0; index < program.cpu_tail_count; ++index) {
    const auto &tail = program.cpu_tails[index];
    std::uint32_t derived_boundary = 0;
    if (tail.placement == DeclaredCpuTail::Placement::after_stage) {
      const auto *stage = FindStage(program, tail.after_stage_id);
      if (stage == nullptr || tail.before_stage_id != 0) {
        SetReason(reason, ReasonId::malformed_contract);
        return false;
      }
      derived_boundary =
          static_cast<std::uint32_t>(stage - program.stages.data() + 1);
    } else if (tail.placement == DeclaredCpuTail::Placement::before_stage) {
      const auto *stage = FindStage(program, tail.before_stage_id);
      if (stage == nullptr || tail.after_stage_id != 0) {
        SetReason(reason, ReasonId::malformed_contract);
        return false;
      }
      derived_boundary =
          static_cast<std::uint32_t>(stage - program.stages.data());
    } else {
      SetReason(reason, ReasonId::malformed_contract);
      return false;
    }
    if (tail.id == CpuTailId::invalid ||
        tail.id == CpuTailId::backend_resolved_output ||
        (tail.boundary_index != 0 && tail.boundary_index != derived_boundary) ||
        (have_boundary && derived_boundary < previous_boundary_index) ||
        cpu_tail_scratch >
            std::numeric_limits<std::uint64_t>::max() - tail.scratch_bytes) {
      SetReason(reason, ReasonId::malformed_contract);
      return false;
    }
    cpu_tail_scratch += tail.scratch_bytes;
    previous_boundary_index = derived_boundary;
    have_boundary = true;
  }
  if (cpu_tail_scratch > program.budget.scratch_bytes - stage_scratch) {
    SetReason(reason, ReasonId::resource_exhausted);
    return false;
  }
  return true;
}

bool ValidateHostFrameBinding(const HostFrameBinding &binding,
                              const PreparedProgram &program,
                              ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (program.stage_count == 0 || binding.capture_sequence == 0 ||
      binding.ownership != OwnershipModel::borrowed_until_completion ||
      binding.lifetime != ResourceLifetime::frame_borrowed ||
      binding.input_plane_count != binding.input_descriptor.plane_count ||
      binding.output_plane_count != binding.output_descriptor.plane_count ||
      binding.input_plane_count > binding.input_planes.size() ||
      binding.output_plane_count > binding.output_planes.size() ||
      !SameImageDescriptor(binding.input_descriptor, program.host_input) ||
      !SameImageDescriptor(binding.output_descriptor, program.host_output)) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return false;
  }
  const auto validate_planes = [](const auto &bindings, std::size_t count,
                                  const ImageDescriptor &descriptor,
                                  bool input) noexcept {
    for (std::size_t index = 0; index < count; ++index) {
      const auto &plane = bindings[index];
      const auto &expected = descriptor.planes[index];
      const auto *pointer =
          input ? plane.read_data
                : static_cast<const std::byte *>(plane.write_data);
      const auto address = reinterpret_cast<std::uintptr_t>(pointer);
      if (pointer == nullptr || plane.span_bytes < expected.size_bytes ||
          plane.row_stride_bytes != expected.row_stride_bytes ||
          address % expected.alignment_bytes != 0 ||
          address >
              std::numeric_limits<std::uintptr_t>::max() - plane.span_bytes) {
        return false;
      }
    }
    return true;
  };
  if (!validate_planes(binding.input_planes, binding.input_plane_count,
                       binding.input_descriptor, true) ||
      !validate_planes(binding.output_planes, binding.output_plane_count,
                       binding.output_descriptor, false)) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return false;
  }
  for (std::size_t input = 0; input < binding.input_plane_count; ++input) {
    const auto input_begin =
        reinterpret_cast<std::uintptr_t>(binding.input_planes[input].read_data);
    const auto input_end = input_begin + binding.input_planes[input].span_bytes;
    for (std::size_t output = 0; output < binding.output_plane_count;
         ++output) {
      const auto output_begin = reinterpret_cast<std::uintptr_t>(
          binding.output_planes[output].write_data);
      const auto output_end =
          output_begin + binding.output_planes[output].span_bytes;
      if (input_begin >= output_end || output_begin >= input_end)
        continue;
      const bool exact = input_begin == output_begin && input_end == output_end;
      const auto input_alias = binding.input_descriptor.aliasing;
      const auto output_alias = binding.output_descriptor.aliasing;
      const bool alias_allowed =
          binding.input_descriptor.in_place_permitted &&
          binding.output_descriptor.in_place_permitted &&
          input_alias != AliasingPermission::forbidden &&
          output_alias != AliasingPermission::forbidden &&
          ((input_alias == AliasingPermission::overlapping_allowed &&
            output_alias == AliasingPermission::overlapping_allowed) ||
           exact);
      if (!alias_allowed) {
        SetReason(reason, ReasonId::host_frame_binding_invalid);
        return false;
      }
    }
  }
  return true;
}

PreparedProgramSession::~PreparedProgramSession() { Shutdown(); }

StatusId PreparedProgramSession::Configure(const PreparedProgram &program,
                                           IResidentAdapter &adapter,
                                           ReadinessSnapshot *readiness,
                                           ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  // A setup attempt establishes a new authority boundary even when the old
  // program is retained. The old committed frame must not be presented as
  // evidence for the candidate configuration.
  latest_committed_frame_ = {};
  if (stopped_) {
    SetReason(reason, ReasonId::shutdown_requested);
    return StatusId::stopped;
  }
  if (completion_pending_) {
    SetReason(reason, ReasonId::output_not_committed);
    return StatusId::blocked;
  }
  ++telemetry_.setup_attempts;

  ReasonId validation_reason = ReasonId::none;
  if (!ValidatePreparedProgram(program, &validation_reason)) {
    SetReason(reason, validation_reason);
    if (prepared_) {
      if (MayRetainOldProgram(program))
        ++telemetry_.setup_rollbacks;
      else
        RejectOldProgramRetention(ReasonId::old_program_policy_incompatible);
    }
    return validation_reason == ReasonId::resource_exhausted
               ? StatusId::resource_exhausted
               : StatusId::blocked;
  }
  if (adapter.adapter_instance_id() != program.owner_adapter_instance_id ||
      adapter.backend_id() != program.execution_domain.backend) {
    SetReason(reason, ReasonId::foreign_adapter);
    if (prepared_) {
      if (MayRetainOldProgram(program))
        ++telemetry_.setup_rollbacks;
      else
        RejectOldProgramRetention(ReasonId::old_program_policy_incompatible);
    }
    return StatusId::blocked;
  }

  PreparedProgramHandle candidate;
  ReadinessSnapshot candidate_readiness;
  ReasonId prepare_reason = ReasonId::none;
  const StatusId prepare_status = adapter.Prepare(
      program, &candidate, &candidate_readiness, &prepare_reason);
  if (prepare_status != StatusId::ready &&
      prepare_status != StatusId::degraded) {
    if (candidate.handle_id != 0)
      adapter.DestroyPrepared(&candidate);
    if (prepared_) {
      if (MayRetainOldProgram(program))
        ++telemetry_.setup_rollbacks;
      else
        RejectOldProgramRetention(ReasonId::old_program_policy_incompatible);
    }
    if (readiness)
      *readiness = candidate_readiness;
    SetReason(reason, prepare_reason);
    return prepare_status;
  }

  const bool budget_valid =
      BudgetContains(program.budget, candidate.reserved_budget) &&
      candidate.reserved_budget.in_flight_slots == 1;
  const bool handle_valid =
      candidate.handle_id != 0 && candidate.program_id == program.program_id &&
      candidate.adapter_instance_id == program.owner_adapter_instance_id &&
      candidate.invalidation_key == program.invalidation_key &&
      candidate.execution_domain == program.execution_domain && budget_valid;
  const bool readiness_valid =
      candidate_readiness.status == prepare_status &&
      candidate_readiness.available &&
      candidate_readiness.requested_backend == program.requested_backend &&
      candidate_readiness.selected_backend ==
          program.execution_domain.backend &&
      candidate_readiness.execution_domain == program.execution_domain &&
      ValidReadinessEvidence(candidate_readiness);
  if (!handle_valid || !readiness_valid) {
    if (candidate.handle_id != 0)
      adapter.DestroyPrepared(&candidate);
    if (prepared_) {
      if (MayRetainOldProgram(program))
        ++telemetry_.setup_rollbacks;
      else
        RejectOldProgramRetention(ReasonId::old_program_policy_incompatible);
    }
    SetReason(reason, !budget_valid ? ReasonId::resource_exhausted
                                    : ReasonId::malformed_contract);
    return StatusId::blocked;
  }

  DestroyActive();
  adapter_ = &adapter;
  program_ = program;
  handle_ = candidate;
  prepared_ = true;
  quarantined_ = false;
  stopped_ = false;
  completion_pending_ = false;
  have_capture_sequence_ = false;
  rebuild_requested_ = false;
  disabled_stage_mask_ = 0;
  permanent_disabled_stage_mask_ = 0;
  consecutive_failures_.fill(0);
  retry_after_sequence_.fill(0);
  ++telemetry_.setup_successes;
  telemetry_.requested_backend = program.requested_backend;
  telemetry_.selected_backend = program.execution_domain.backend;
  telemetry_.runtime_status = prepare_status;
  telemetry_.runtime_reason = candidate_readiness.reason;
  telemetry_.disabled_stage_mask = 0;
  readiness_ = candidate_readiness;
  configured_readiness_ = candidate_readiness;
  if (readiness)
    *readiness = candidate_readiness;
  return prepare_status;
}

ExecuteResult
PreparedProgramSession::Execute(const ExecuteRequest &request) noexcept {
  if (stopped_)
    return FailedExecute(StatusId::stopped, ReasonId::shutdown_requested);
  if (quarantined_)
    return FailedExecute(StatusId::quarantined,
                         readiness_.reason == ReasonId::none
                             ? ReasonId::circuit_breaker_open
                             : readiness_.reason);
  if (rebuild_requested_)
    return FailedExecute(StatusId::stale, ReasonId::reconfiguration_required);
  if (!prepared_ || adapter_ == nullptr)
    return FailedExecute(StatusId::blocked, ReasonId::stale_program);
  if (completion_pending_)
    return FailedExecute(StatusId::failed, ReasonId::output_not_committed);
  if (request.invalidation_key != program_.invalidation_key)
    return FailedExecute(StatusId::stale, ReasonId::reconfiguration_required);
  if (request.execution_domain != program_.execution_domain)
    return FailedExecute(StatusId::blocked, ReasonId::wrong_execution_domain);
  if (request.disabled_stage_mask != 0)
    return FailedExecute(StatusId::blocked, ReasonId::malformed_contract);
  if (!ValidateResourceHandle(
          request.input, program_.execution_domain,
          program_.execution_domain.logical_context_generation,
          program_.budget)) {
    const ReasonId resource_reason =
        request.input.execution_domain != program_.execution_domain
            ? ReasonId::wrong_execution_domain
            : (request.input.lifetime_generation !=
                       program_.execution_domain.logical_context_generation
                   ? ReasonId::wrong_generation
                   : ReasonId::invalid_resource);
    return FailedExecute(StatusId::blocked, resource_reason);
  }

  if (request.capture_sequence == 0)
    return FailedExecute(StatusId::blocked, ReasonId::malformed_contract);

  bool temporal_state_reset = false;
  std::uint64_t dropped_capture_frames = 0;
  if (have_capture_sequence_) {
    if (request.capture_sequence <= last_capture_sequence_) {
      ++telemetry_.frame_drop_events;
      ++telemetry_.preserve_order_failures;
      telemetry_.runtime_status = StatusId::failed;
      telemetry_.runtime_reason = ReasonId::frame_order_violated;
      return FailedExecute(StatusId::failed, ReasonId::frame_order_violated);
    }
    if (request.capture_sequence != last_capture_sequence_ + 1) {
      dropped_capture_frames =
          request.capture_sequence - last_capture_sequence_ - 1;
      ++telemetry_.frame_drop_events;
      telemetry_.dropped_capture_frames += dropped_capture_frames;
      switch (program_.temporal.frame_drop_behavior) {
      case FrameDropBehavior::preserve_order:
        ++telemetry_.preserve_order_failures;
        telemetry_.runtime_status = StatusId::failed;
        telemetry_.runtime_reason = ReasonId::frame_order_violated;
        return FailedExecute(StatusId::failed, ReasonId::frame_order_violated);
      case FrameDropBehavior::reset_temporal_state: {
        const StatusId reset = adapter_->ResetTemporalState(
            handle_, ReasonId::temporal_discontinuity);
        if (reset != StatusId::ready)
          return FailedExecute(reset, ReasonId::temporal_discontinuity);
        ++telemetry_.temporal_resets;
        temporal_state_reset = true;
        break;
      }
      case FrameDropBehavior::latest_frame_wins:
        ++telemetry_.latest_frame_wins_events;
        if (program_.temporal.temporal &&
            program_.temporal.requires_contiguous_capture_sequence) {
          if (program_.temporal.discontinuity_action ==
              TemporalDiscontinuityAction::fail_frame) {
            return FailedExecute(StatusId::failed,
                                 ReasonId::temporal_discontinuity);
          }
          const StatusId reset = adapter_->ResetTemporalState(
              handle_, ReasonId::temporal_discontinuity);
          if (reset != StatusId::ready)
            return FailedExecute(reset, ReasonId::temporal_discontinuity);
          ++telemetry_.temporal_resets;
          temporal_state_reset = true;
        }
        break;
      }
    }
  }

  for (std::size_t index = 0; index < program_.stage_count; ++index) {
    const std::uint32_t bit = std::uint32_t{1} << index;
    if ((disabled_stage_mask_ & bit) != 0 &&
        (permanent_disabled_stage_mask_ & bit) == 0 &&
        request.capture_sequence >= retry_after_sequence_[index]) {
      disabled_stage_mask_ &= ~bit;
      retry_after_sequence_[index] = 0;
      readiness_.blocked_stage_mask &= ~bit;
    }
  }
  if (disabled_stage_mask_ == 0 && !quarantined_ && !rebuild_requested_)
    readiness_ = configured_readiness_;
  telemetry_.disabled_stage_mask = disabled_stage_mask_;
  readiness_.disabled_stage_mask = disabled_stage_mask_;
  ExecuteRequest effective_request = request;
  effective_request.disabled_stage_mask = disabled_stage_mask_;
  ++telemetry_.execution_attempts;
  ExecuteResult result = adapter_->Execute(handle_, effective_request);
  const auto fail_after_adapter = [this](ExecuteResult failure) noexcept {
    host_binding_pending_ = false;
    pending_host_tokens_ = {};
    return failure;
  };
  result.dropped_capture_frames = dropped_capture_frames;
  result.temporal_state_reset = temporal_state_reset;
  if (result.stage_count > result.stages.size())
    return fail_after_adapter(
        FailedExecute(StatusId::failed, ReasonId::malformed_contract));
  AccumulateExecuteTelemetry(result);
  if (result.reason == ReasonId::device_lost) {
    quarantined_ = true;
    ++telemetry_.device_losses;
    result.status = StatusId::quarantined;
    readiness_.status = StatusId::quarantined;
    readiness_.reason = ReasonId::device_lost;
    readiness_.available = false;
    telemetry_.runtime_status = StatusId::quarantined;
    telemetry_.runtime_reason = ReasonId::device_lost;
    return fail_after_adapter(result);
  }
  // Adapter-side token/binding validation can fail before any stage runs.
  // Preserve that exact typed failure; stage circuit breakers have no result
  // to consume in this case.
  if (!IsSuccessfulExecution(result.status) && result.stage_count == 0)
    return fail_after_adapter(result);
  if (result.stage_count != program_.stage_count)
    return fail_after_adapter(
        FailedExecute(StatusId::failed, ReasonId::malformed_contract));

  bool optional_failure = false;
  bool required_failure = false;
  for (std::size_t index = 0; index < result.stage_count; ++index) {
    const auto &stage_result = result.stages[index];
    const auto &stage = program_.stages[index];
    const std::uint32_t bit = std::uint32_t{1} << index;
    if (stage.stage_id != stage_result.stage_id ||
        stage.semantic != stage_result.semantic ||
        stage.primitive != stage_result.primitive) {
      return fail_after_adapter(
          FailedExecute(StatusId::failed, ReasonId::malformed_contract));
    }
    if ((effective_request.disabled_stage_mask & bit) != 0) {
      if (stage.required ||
          stage_result.status != StageExecutionStatus::disabled_preserved ||
          stage_result.reason != ReasonId::circuit_breaker_open ||
          !stage_result.valid_output_preserved ||
          stage_result.elapsed_ns != 0) {
        return fail_after_adapter(
            FailedExecute(StatusId::failed, ReasonId::circuit_breaker_open));
      }
      ++telemetry_.disabled_stage_frames;
      continue;
    }
    if (stage_result.status == StageExecutionStatus::disabled_preserved)
      return fail_after_adapter(
          FailedExecute(StatusId::failed, ReasonId::malformed_contract));
    if (stage_result.status == StageExecutionStatus::optional_failed_open) {
      if (stage.required || !stage_result.valid_output_preserved)
        return fail_after_adapter(
            FailedExecute(StatusId::failed, ReasonId::required_stage_failed));
      optional_failure = true;
    }
    if (stage_result.status == StageExecutionStatus::required_failed ||
        stage_result.status == StageExecutionStatus::not_run) {
      required_failure = true;
    }
  }
  UpdateCircuitBreakers(result, request.capture_sequence);
  if (!IsSuccessfulExecution(result.status) || required_failure)
    return fail_after_adapter(result);
  if (!result.completion.complete || result.completion.token_id == 0 ||
      result.completion.program_id != program_.program_id ||
      result.completion.adapter_instance_id !=
          program_.owner_adapter_instance_id ||
      result.completion.capture_sequence != request.capture_sequence) {
    return fail_after_adapter(
        FailedExecute(StatusId::failed, ReasonId::incomplete_execution));
  }
  if (!result.output_valid ||
      !ValidateResourceHandle(
          result.output, program_.execution_domain,
          program_.execution_domain.logical_context_generation,
          program_.budget)) {
    return fail_after_adapter(
        FailedExecute(StatusId::failed, ReasonId::invalid_resource));
  }
  if (host_binding_pending_ &&
      (!SameResourceToken(request.input, pending_host_tokens_.input) ||
       !SameResourceToken(result.output, pending_host_tokens_.output))) {
    return fail_after_adapter(
        FailedExecute(StatusId::failed, ReasonId::host_frame_binding_invalid));
  }

  if (result.transfers.intermediate_readbacks != 0 ||
      result.transfers.intermediate_uploads != 0 ||
      result.transfers.uploads != (program_.requires_upload ? 1u : 0u) ||
      result.transfers.final_readbacks !=
          (program_.requires_final_readback ? 1u : 0u) ||
      result.transfers.cpu_tail_boundaries != program_.cpu_tail_count) {
    return fail_after_adapter(
        FailedExecute(StatusId::failed, ReasonId::transfer_contract_violated));
  }
  if (optional_failure) {
    result.status = StatusId::degraded;
    result.reason = ReasonId::optional_stage_failed_open;
  }

  ++telemetry_.execution_successes;
  pending_completion_ = result.completion;
  pending_stage_count_ = result.stage_count;
  std::copy_n(result.stages.begin(), result.stage_count,
              pending_stages_.begin());
  completion_pending_ = true;
  last_capture_sequence_ = request.capture_sequence;
  have_capture_sequence_ = true;
  if (!quarantined_ && !rebuild_requested_ && disabled_stage_mask_ == 0) {
    telemetry_.runtime_status = result.status;
    telemetry_.runtime_reason = result.reason;
  }
  return result;
}

StatusId PreparedProgramSession::BindHostFrame(const HostFrameBinding &binding,
                                               HostFrameTokens *tokens,
                                               ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (stopped_) {
    SetReason(reason, ReasonId::shutdown_requested);
    return StatusId::stopped;
  }
  if (!prepared_ || adapter_ == nullptr || tokens == nullptr ||
      completion_pending_ || host_binding_pending_) {
    SetReason(reason, completion_pending_ || host_binding_pending_
                          ? ReasonId::output_not_committed
                          : ReasonId::stale_program);
    return StatusId::blocked;
  }
  ReasonId validation_reason = ReasonId::none;
  if (!ValidateHostFrameBinding(binding, program_, &validation_reason)) {
    SetReason(reason, validation_reason);
    return StatusId::blocked;
  }
  HostFrameTokens candidate;
  const StatusId status =
      adapter_->BindHostFrame(handle_, binding, &candidate, reason);
  if (status != StatusId::ready)
    return status;
  if (!ValidateResourceHandle(
          candidate.input, program_.execution_domain,
          program_.execution_domain.logical_context_generation,
          program_.budget) ||
      candidate.input.lifetime != ResourceLifetime::frame_borrowed ||
      !ValidateResourceHandle(
          candidate.output, program_.execution_domain,
          program_.execution_domain.logical_context_generation,
          program_.budget) ||
      candidate.output.lifetime != ResourceLifetime::frame_borrowed) {
    const bool one_resource =
        candidate.input.resource_id != 0 &&
        candidate.input.resource_id == candidate.output.resource_id &&
        candidate.input.allocator_owner_id ==
            candidate.output.allocator_owner_id;
    DestroyResource(&candidate.input);
    if (!one_resource)
      DestroyResource(&candidate.output);
    else
      candidate.output = {};
    SetReason(reason, ReasonId::invalid_resource);
    return StatusId::blocked;
  }
  *tokens = candidate;
  pending_host_binding_ = binding;
  pending_host_tokens_ = candidate;
  host_binding_pending_ = true;
  return StatusId::ready;
}

StatusId PreparedProgramSession::ResolveHostFrameOutput(
    const CompletionToken &completion, const ResourceHandle &output,
    HostFrameOutputResolution *resolution, ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (!prepared_ || adapter_ == nullptr || resolution == nullptr ||
      !host_binding_pending_ || !completion_pending_ ||
      completion.token_id != pending_completion_.token_id ||
      completion.program_id != pending_completion_.program_id ||
      completion.adapter_instance_id !=
          pending_completion_.adapter_instance_id ||
      completion.capture_sequence != pending_completion_.capture_sequence) {
    SetReason(reason, ReasonId::output_not_committed);
    return StatusId::blocked;
  }
  if (!SameResourceToken(output, pending_host_tokens_.output)) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return StatusId::blocked;
  }
  HostFrameOutputResolution candidate;
  const StatusId status = adapter_->ResolveHostFrameOutput(
      handle_, completion, output, &candidate, reason);
  if (status != StatusId::ready)
    return status;
  if (!candidate.output_ready ||
      candidate.capture_sequence != pending_host_binding_.capture_sequence ||
      candidate.plane_count != pending_host_binding_.output_plane_count ||
      !SameImageDescriptor(candidate.descriptor,
                           pending_host_binding_.output_descriptor)) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return StatusId::blocked;
  }
  for (std::size_t index = 0; index < candidate.plane_count; ++index) {
    const auto &actual = candidate.planes[index];
    const auto &expected = pending_host_binding_.output_planes[index];
    if (actual.write_data != expected.write_data ||
        actual.span_bytes != expected.span_bytes ||
        actual.row_stride_bytes != expected.row_stride_bytes) {
      SetReason(reason, ReasonId::host_frame_binding_invalid);
      return StatusId::blocked;
    }
  }
  *resolution = candidate;
  return StatusId::ready;
}

StatusId PreparedProgramSession::CommitOutput(const OutputCommit &commit,
                                              ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (!completion_pending_ ||
      !(commit.completion.token_id == pending_completion_.token_id &&
        commit.completion.program_id == pending_completion_.program_id &&
        commit.completion.adapter_instance_id ==
            pending_completion_.adapter_instance_id &&
        commit.completion.capture_sequence ==
            pending_completion_.capture_sequence)) {
    SetReason(reason, ReasonId::output_not_committed);
    return StatusId::failed;
  }

  completion_pending_ = false;
  host_binding_pending_ = false;
  if (!commit.output_write_succeeded) {
    SetReason(reason, ReasonId::output_write_failed);
    pending_stage_count_ = 0;
    return StatusId::failed;
  }

  ++telemetry_.committed_output_frames;
  CommittedFrameSnapshot committed;
  committed.valid = true;
  committed.program_id = program_.program_id;
  committed.invalidation_key = program_.invalidation_key;
  committed.execution_domain = program_.execution_domain;
  committed.capture_sequence = commit.completion.capture_sequence;
  committed.output_commit_sequence = telemetry_.committed_output_frames;
  committed.stage_count = pending_stage_count_;
  for (std::size_t index = 0; index < pending_stage_count_; ++index) {
    const auto &execution = pending_stages_[index];
    const auto *stage = FindStage(program_, execution.stage_id);
    if (stage == nullptr) {
      // Execute validated this mapping before arming the completion. Keep the
      // guard fail-closed in case future contract code changes that ordering.
      latest_committed_frame_ = {};
      pending_stage_count_ = 0;
      SetReason(reason, ReasonId::malformed_contract);
      return StatusId::failed;
    }
    committed.stages[index] = {
        execution.stage_id, execution.semantic, execution.primitive,
        stage->backend,      stage->provider,    execution.status,
        execution.reason};
  }
  latest_committed_frame_ = committed;
  const bool effect_succeeded = std::any_of(
      pending_stages_.begin(), pending_stages_.begin() + pending_stage_count_,
      [](const StageExecutionResult &stage) {
        return stage.status == StageExecutionStatus::succeeded;
      });
  if (effect_succeeded) {
    ++telemetry_.active_backend_frames;
    telemetry_.active_backend = program_.execution_domain.backend;
    if (!program_.strict_backend_selection &&
        telemetry_.requested_backend != telemetry_.selected_backend) {
      ++telemetry_.fallback_frames;
    }
    if (program_.stage_count != 0) {
      const auto provider = program_.stages[0].provider;
      const bool one_provider =
          std::all_of(program_.stages.begin(),
                      program_.stages.begin() + program_.stage_count,
                      [provider](const PreparedStage &stage) {
                        return stage.provider == provider;
                      });
      telemetry_.active_provider =
          one_provider ? provider : SemanticEffectProviderId::invalid;
    }
  }
  for (std::size_t index = 0; index < pending_stage_count_; ++index) {
    if (pending_stages_[index].status == StageExecutionStatus::succeeded) {
      const auto *stage = FindStage(program_, pending_stages_[index].stage_id);
      if (stage) {
        const std::size_t stage_index =
            static_cast<std::size_t>(stage - program_.stages.data());
        ++telemetry_.committed_stage_successes[stage_index];
      }
    }
  }
  pending_stage_count_ = 0;
  return StatusId::ready;
}

bool PreparedProgramSession::MayRetainOldProgram(
    const PreparedProgram &candidate) const noexcept {
  if (!prepared_ || candidate.setup_failure_policy !=
                        SetupFailurePolicy::retain_old_program) {
    return false;
  }
  if (candidate.strict_backend_selection &&
      (candidate.requested_backend == BackendId::invalid ||
       program_.execution_domain.backend != candidate.requested_backend)) {
    return false;
  }
  if (candidate.strict_provider_selection) {
    if (candidate.requested_provider == SemanticEffectProviderId::invalid)
      return false;
    for (std::size_t index = 0; index < program_.stage_count; ++index) {
      if (program_.stages[index].semantic != SemanticEffectId::invalid &&
          program_.stages[index].provider != candidate.requested_provider) {
        return false;
      }
    }
  }
  if (candidate.strict_execution_domain &&
      (!ValidateExecutionDomain(candidate.requested_execution_domain) ||
       program_.execution_domain != candidate.requested_execution_domain)) {
    return false;
  }
  return true;
}

void PreparedProgramSession::RejectOldProgramRetention(
    ReasonId reason) noexcept {
  ++telemetry_.setup_retention_rejections;
  telemetry_.runtime_status = StatusId::blocked;
  telemetry_.runtime_reason = reason;
  readiness_.status = StatusId::blocked;
  readiness_.reason = reason;
  readiness_.available = false;
  DestroyActive();
}

void PreparedProgramSession::UpdateCircuitBreakers(
    const ExecuteResult &result, std::uint64_t capture_sequence) noexcept {
  for (std::size_t index = 0;
       index < result.stage_count && index < program_.stage_count; ++index) {
    const auto status = result.stages[index].status;
    if (status == StageExecutionStatus::succeeded) {
      consecutive_failures_[index] = 0;
      continue;
    }
    if (status == StageExecutionStatus::disabled_preserved)
      continue;
    if (status != StageExecutionStatus::optional_failed_open &&
        status != StageExecutionStatus::required_failed &&
        status != StageExecutionStatus::not_run) {
      continue;
    }
    if (consecutive_failures_[index] !=
        std::numeric_limits<std::uint32_t>::max()) {
      ++consecutive_failures_[index];
    }
    const auto &stage = program_.stages[index];
    if (consecutive_failures_[index] <
        stage.circuit_breaker.consecutive_failure_limit) {
      continue;
    }
    consecutive_failures_[index] = 0;
    ++telemetry_.circuit_breaker_opens;
    const std::uint32_t bit = std::uint32_t{1} << index;
    switch (stage.circuit_breaker.action) {
    case CircuitBreakerAction::disable_stage:
      if (stage.required) {
        quarantined_ = true;
        readiness_.status = StatusId::quarantined;
        readiness_.reason = ReasonId::required_stage_failed;
        readiness_.available = false;
        telemetry_.runtime_status = StatusId::quarantined;
        telemetry_.runtime_reason = ReasonId::required_stage_failed;
        break;
      }
      disabled_stage_mask_ |= bit;
      readiness_.disabled_stage_mask = disabled_stage_mask_;
      readiness_.blocked_stage_mask |= bit;
      readiness_.status = StatusId::degraded;
      readiness_.available = true;
      readiness_.degraded = true;
      readiness_.reason = stage.circuit_breaker.cooldown_frames == 0
                              ? ReasonId::circuit_breaker_permanently_disabled
                              : ReasonId::circuit_breaker_retry_pending;
      telemetry_.disabled_stage_mask = disabled_stage_mask_;
      telemetry_.runtime_status = StatusId::degraded;
      telemetry_.runtime_reason = readiness_.reason;
      if (stage.circuit_breaker.cooldown_frames == 0) {
        permanent_disabled_stage_mask_ |= bit;
        retry_after_sequence_[index] =
            std::numeric_limits<std::uint64_t>::max();
      } else if (capture_sequence <= std::numeric_limits<std::uint64_t>::max() -
                                         stage.circuit_breaker.cooldown_frames -
                                         1) {
        retry_after_sequence_[index] =
            capture_sequence + stage.circuit_breaker.cooldown_frames + 1;
      } else {
        permanent_disabled_stage_mask_ |= bit;
        retry_after_sequence_[index] =
            std::numeric_limits<std::uint64_t>::max();
      }
      break;
    case CircuitBreakerAction::quarantine_program:
      quarantined_ = true;
      readiness_.status = StatusId::quarantined;
      readiness_.reason = ReasonId::circuit_breaker_open;
      readiness_.available = false;
      telemetry_.runtime_status = StatusId::quarantined;
      telemetry_.runtime_reason = ReasonId::circuit_breaker_open;
      break;
    case CircuitBreakerAction::request_rebuild:
      rebuild_requested_ = true;
      ++telemetry_.rebuild_requests;
      readiness_.status = StatusId::stale;
      readiness_.reason = ReasonId::reconfiguration_required;
      readiness_.available = false;
      readiness_.rebuild_requested = true;
      telemetry_.runtime_status = StatusId::stale;
      telemetry_.runtime_reason = ReasonId::reconfiguration_required;
      break;
    }
  }
}

StatusId PreparedProgramSession::ResetTemporalState(ReasonId reason) noexcept {
  if (stopped_)
    return StatusId::stopped;
  if (quarantined_)
    return StatusId::quarantined;
  if (rebuild_requested_ || !prepared_ || adapter_ == nullptr)
    return StatusId::stale;
  if (completion_pending_ || host_binding_pending_)
    return StatusId::failed;
  const auto status = adapter_->ResetTemporalState(handle_, reason);
  if (status == StatusId::ready)
    ++telemetry_.temporal_resets;
  return status;
}

void PreparedProgramSession::Invalidate(ReasonId reason) noexcept {
  completion_pending_ = false;
  host_binding_pending_ = false;
  pending_stage_count_ = 0;
  latest_committed_frame_ = {};
  have_capture_sequence_ = false;
  if (reason == ReasonId::device_lost) {
    quarantined_ = true;
    ++telemetry_.device_losses;
  }
  DestroyActive();
}

void PreparedProgramSession::Shutdown() noexcept {
  if (stopped_)
    return;
  completion_pending_ = false;
  host_binding_pending_ = false;
  pending_stage_count_ = 0;
  latest_committed_frame_ = {};
  IResidentAdapter *adapter = adapter_;
  DestroyActive();
  if (adapter)
    adapter->Shutdown();
  stopped_ = true;
}

void PreparedProgramSession::DestroyActive() noexcept {
  if (prepared_ && adapter_)
    adapter_->DestroyPrepared(&handle_);
  handle_ = {};
  prepared_ = false;
}

void PreparedProgramSession::AccumulateExecuteTelemetry(
    const ExecuteResult &result) noexcept {
  telemetry_.transfers.uploads += result.transfers.uploads;
  telemetry_.transfers.final_readbacks += result.transfers.final_readbacks;
  telemetry_.transfers.intermediate_readbacks +=
      result.transfers.intermediate_readbacks;
  telemetry_.transfers.intermediate_uploads +=
      result.transfers.intermediate_uploads;
  telemetry_.transfers.synchronizations += result.transfers.synchronizations;
  telemetry_.transfers.cpu_tail_boundaries +=
      result.transfers.cpu_tail_boundaries;
  for (std::size_t index = 0; index < result.stage_count; ++index) {
    if (result.stages[index].status ==
        StageExecutionStatus::optional_failed_open) {
      ++telemetry_.optional_fail_open_stages;
    } else if (result.stages[index].status ==
               StageExecutionStatus::required_failed) {
      ++telemetry_.required_stage_failures;
    }
  }

  static constexpr std::array<std::uint64_t, kMaxLatencyHistogramBuckets>
      kUpperBoundsNs = {250'000,     500'000,
                        1'000'000,   2'000'000,
                        4'000'000,   8'000'000,
                        12'000'000,  16'000'000,
                        24'000'000,  33'000'000,
                        50'000'000,  66'000'000,
                        100'000'000, 150'000'000,
                        250'000'000, std::numeric_limits<std::uint64_t>::max()};
  auto &latency = telemetry_.frame_latency;
  latency.bucket_upper_ns = kUpperBoundsNs;
  ++latency.samples;
  latency.total_ns += result.frame_elapsed_ns;
  latency.maximum_ns = std::max(latency.maximum_ns, result.frame_elapsed_ns);
  for (std::size_t index = 0; index < kUpperBoundsNs.size(); ++index) {
    if (result.frame_elapsed_ns <= kUpperBoundsNs[index]) {
      ++latency.bucket_counts[index];
      break;
    }
  }
  const auto percentile = [&latency](std::uint64_t numerator) {
    const std::uint64_t target = (latency.samples * numerator + 99u) / 100u;
    std::uint64_t cumulative = 0;
    for (std::size_t index = 0; index < latency.bucket_counts.size(); ++index) {
      cumulative += latency.bucket_counts[index];
      if (cumulative >= target)
        return latency.bucket_upper_ns[index];
    }
    return latency.maximum_ns;
  };
  latency.p95_ns = percentile(95);
  latency.p99_ns = percentile(99);

  for (std::size_t stage_index = 0; stage_index < result.stage_count;
       ++stage_index) {
    auto &stage_latency = telemetry_.stage_latency[stage_index];
    stage_latency.bucket_upper_ns = kUpperBoundsNs;
    ++stage_latency.samples;
    stage_latency.total_ns += result.stages[stage_index].elapsed_ns;
    stage_latency.maximum_ns = std::max(stage_latency.maximum_ns,
                                        result.stages[stage_index].elapsed_ns);
    for (std::size_t bucket = 0; bucket < kUpperBoundsNs.size(); ++bucket) {
      if (result.stages[stage_index].elapsed_ns <= kUpperBoundsNs[bucket]) {
        ++stage_latency.bucket_counts[bucket];
        break;
      }
    }
    const auto stage_percentile = [&stage_latency](std::uint64_t numerator) {
      const std::uint64_t target =
          (stage_latency.samples * numerator + 99u) / 100u;
      std::uint64_t cumulative = 0;
      for (std::size_t bucket = 0; bucket < stage_latency.bucket_counts.size();
           ++bucket) {
        cumulative += stage_latency.bucket_counts[bucket];
        if (cumulative >= target)
          return stage_latency.bucket_upper_ns[bucket];
      }
      return stage_latency.maximum_ns;
    };
    stage_latency.p95_ns = stage_percentile(95);
    stage_latency.p99_ns = stage_percentile(99);
  }
}

} // namespace studiocast::hal
