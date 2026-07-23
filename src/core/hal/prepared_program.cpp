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
  std::uint64_t total_budget = budget.persistent_bytes;
  if (total_budget >
      std::numeric_limits<std::uint64_t>::max() - budget.scratch_bytes) {
    return false;
  }
  total_budget += budget.scratch_bytes;
  if (total_budget >
      std::numeric_limits<std::uint64_t>::max() - budget.upload_staging_bytes) {
    return false;
  }
  total_budget += budget.upload_staging_bytes;
  if (total_budget > std::numeric_limits<std::uint64_t>::max() -
                         budget.readback_staging_bytes) {
    return false;
  }
  total_budget += budget.readback_staging_bytes;

  return resource.resource_id != 0 && resource.allocator_owner_id != 0 &&
         resource.allocator_owner_id == expected_domain.adapter_instance_id &&
         resource.lifetime_generation == expected_lifetime_generation &&
         resource.memory_domain != MemoryDomain::invalid &&
         resource.lifetime != ResourceLifetime::invalid &&
         resource.synchronization_state == SynchronizationState::ready &&
         resource.execution_domain == expected_domain &&
         resource.size_bytes != 0 && resource.size_bytes <= total_budget &&
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

  std::uint32_t previous_stage_id = 0;
  std::uint64_t stage_scratch = 0;
  for (std::size_t index = 0; index < program.stage_count; ++index) {
    const auto &stage = program.stages[index];
    if (stage.stage_id == 0 || stage.stage_id <= previous_stage_id ||
        stage.semantic == SemanticEffectId::invalid ||
        stage.provider == SemanticEffectProviderId::invalid ||
        stage.backend == BackendId::invalid ||
        stage.execution_domain != program.execution_domain ||
        stage.backend != program.execution_domain.backend) {
      SetReason(reason, stage.execution_domain.adapter_instance_id !=
                                program.owner_adapter_instance_id
                            ? ReasonId::cross_adapter_residency_forbidden
                            : ReasonId::malformed_contract);
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
        stage.circuit_breaker.consecutive_failure_limit == 0) {
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
  for (std::size_t index = 0; index < program.cpu_tail_count; ++index) {
    const auto &tail = program.cpu_tails[index];
    if (tail.id == CpuTailId::invalid ||
        FindStage(program, tail.after_stage_id) == nullptr) {
      SetReason(reason, ReasonId::malformed_contract);
      return false;
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
    if (prepared_)
      ++telemetry_.setup_rollbacks;
    return validation_reason == ReasonId::resource_exhausted
               ? StatusId::resource_exhausted
               : StatusId::blocked;
  }
  if (adapter.adapter_instance_id() != program.owner_adapter_instance_id ||
      adapter.backend_id() != program.execution_domain.backend) {
    SetReason(reason, ReasonId::foreign_adapter);
    if (prepared_)
      ++telemetry_.setup_rollbacks;
    return StatusId::blocked;
  }

  PreparedProgramHandle candidate;
  ReadinessSnapshot candidate_readiness;
  ReasonId prepare_reason = ReasonId::none;
  const StatusId prepare_status = adapter.Prepare(
      program, &candidate, &candidate_readiness, &prepare_reason);
  if (prepare_status != StatusId::ready) {
    if (candidate.handle_id != 0)
      adapter.DestroyPrepared(&candidate);
    if (prepared_)
      ++telemetry_.setup_rollbacks;
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
      candidate_readiness.status == StatusId::ready &&
      candidate_readiness.available &&
      candidate_readiness.requested_backend == program.requested_backend &&
      candidate_readiness.selected_backend ==
          program.execution_domain.backend &&
      candidate_readiness.execution_domain == program.execution_domain;
  if (!handle_valid || !readiness_valid) {
    if (candidate.handle_id != 0)
      adapter.DestroyPrepared(&candidate);
    if (prepared_)
      ++telemetry_.setup_rollbacks;
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
  ++telemetry_.setup_successes;
  telemetry_.selected_backend = program.execution_domain.backend;
  if (readiness)
    *readiness = candidate_readiness;
  return StatusId::ready;
}

ExecuteResult
PreparedProgramSession::Execute(const ExecuteRequest &request) noexcept {
  if (stopped_)
    return FailedExecute(StatusId::stopped, ReasonId::shutdown_requested);
  if (quarantined_)
    return FailedExecute(StatusId::quarantined, ReasonId::device_lost);
  if (!prepared_ || adapter_ == nullptr)
    return FailedExecute(StatusId::blocked, ReasonId::stale_program);
  if (completion_pending_)
    return FailedExecute(StatusId::failed, ReasonId::output_not_committed);
  if (request.invalidation_key != program_.invalidation_key)
    return FailedExecute(StatusId::stale, ReasonId::reconfiguration_required);
  if (request.execution_domain != program_.execution_domain)
    return FailedExecute(StatusId::blocked, ReasonId::wrong_execution_domain);
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

  if (program_.temporal.temporal &&
      program_.temporal.requires_contiguous_capture_sequence &&
      have_capture_sequence_ &&
      request.capture_sequence != last_capture_sequence_ + 1) {
    if (program_.temporal.discontinuity_action ==
        TemporalDiscontinuityAction::fail_frame) {
      return FailedExecute(StatusId::failed, ReasonId::temporal_discontinuity);
    }
    const StatusId reset =
        adapter_->ResetTemporalState(handle_, ReasonId::temporal_discontinuity);
    if (reset != StatusId::ready)
      return FailedExecute(reset, ReasonId::temporal_discontinuity);
    ++telemetry_.temporal_resets;
  }

  ++telemetry_.execution_attempts;
  ExecuteResult result = adapter_->Execute(handle_, request);
  if (result.stage_count > result.stages.size())
    return FailedExecute(StatusId::failed, ReasonId::malformed_contract);
  AccumulateExecuteTelemetry(result);
  if (result.reason == ReasonId::device_lost) {
    quarantined_ = true;
    ++telemetry_.device_losses;
    result.status = StatusId::quarantined;
    return result;
  }
  if (!IsSuccessfulExecution(result.status))
    return result;
  if (!result.completion.complete || result.completion.token_id == 0 ||
      result.completion.program_id != program_.program_id ||
      result.completion.adapter_instance_id !=
          program_.owner_adapter_instance_id ||
      result.completion.capture_sequence != request.capture_sequence) {
    return FailedExecute(StatusId::failed, ReasonId::incomplete_execution);
  }
  if (result.stage_count != program_.stage_count)
    return FailedExecute(StatusId::failed, ReasonId::malformed_contract);
  if (!result.output_valid ||
      !ValidateResourceHandle(
          result.output, program_.execution_domain,
          program_.execution_domain.logical_context_generation,
          program_.budget)) {
    return FailedExecute(StatusId::failed, ReasonId::invalid_resource);
  }

  bool optional_failure = false;
  for (std::size_t index = 0; index < result.stage_count; ++index) {
    const auto &stage_result = result.stages[index];
    const auto &stage = program_.stages[index];
    if (stage.stage_id != stage_result.stage_id ||
        stage.semantic != stage_result.semantic) {
      return FailedExecute(StatusId::failed, ReasonId::malformed_contract);
    }
    if (stage_result.status == StageExecutionStatus::optional_failed_open) {
      if (stage.required || !stage_result.valid_output_preserved)
        return FailedExecute(StatusId::failed, ReasonId::required_stage_failed);
      optional_failure = true;
    }
    if (stage_result.status == StageExecutionStatus::required_failed ||
        stage_result.status == StageExecutionStatus::not_run) {
      return FailedExecute(StatusId::failed, ReasonId::required_stage_failed);
    }
  }
  if (result.transfers.intermediate_readbacks != 0 ||
      result.transfers.intermediate_uploads != 0 ||
      (program_.requires_upload && result.transfers.uploads != 1) ||
      (program_.requires_final_readback &&
       result.transfers.final_readbacks != 1)) {
    return FailedExecute(StatusId::failed, ReasonId::synchronization_failed);
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
  return result;
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
  if (!commit.output_write_succeeded) {
    SetReason(reason, ReasonId::output_write_failed);
    pending_stage_count_ = 0;
    return StatusId::failed;
  }

  ++telemetry_.committed_output_frames;
  const bool effect_succeeded = std::any_of(
      pending_stages_.begin(), pending_stages_.begin() + pending_stage_count_,
      [](const StageExecutionResult &stage) {
        return stage.status == StageExecutionStatus::succeeded;
      });
  if (effect_succeeded) {
    ++telemetry_.active_backend_frames;
    telemetry_.active_backend = program_.execution_domain.backend;
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

void PreparedProgramSession::Invalidate(ReasonId reason) noexcept {
  completion_pending_ = false;
  pending_stage_count_ = 0;
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
  pending_stage_count_ = 0;
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
  if (result.status == StatusId::passthrough)
    ++telemetry_.fallback_frames;
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
