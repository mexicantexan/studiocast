#include "core/video/hardware_execution_runtime.h"

#include <limits>

namespace studiocast::video {
namespace {

void SetReason(hal::ReasonId *reason, hal::ReasonId value) noexcept {
  if (reason != nullptr)
    *reason = value;
}

bool SameCompletion(const hal::CompletionToken &left,
                    const hal::CompletionToken &right) noexcept {
  return left.token_id == right.token_id &&
         left.program_id == right.program_id &&
         left.adapter_instance_id == right.adapter_instance_id &&
         left.capture_sequence == right.capture_sequence &&
         left.complete == right.complete;
}

bool SameImageDescriptor(const hal::ImageDescriptor &left,
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

bool IsSetupSuccess(hal::StatusId status) noexcept {
  return status == hal::StatusId::ready || status == hal::StatusId::degraded ||
         status == hal::StatusId::passthrough;
}

} // namespace

HardwareExecutionRuntime::HardwareExecutionRuntime(
    const hal::StaticAdapterRegistry &registry) noexcept
    : registry_(&registry) {}

HardwareExecutionRuntime::~HardwareExecutionRuntime() { Shutdown(); }

hal::StatusId HardwareExecutionRuntime::Configure(
    const HardwareExecutionSetupRequest &request,
    hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  latest_setup_ = {};
  latest_setup_.mode = request.mode;
  latest_setup_.legacy_path_retained_for_setup_rollback =
      request.legacy_path_retained_for_setup_rollback;

  if (stopped_) {
    latest_setup_.status = hal::StatusId::stopped;
    latest_setup_.reason = hal::ReasonId::shutdown_requested;
    SetReason(reason, latest_setup_.reason);
    return latest_setup_.status;
  }
  if (frame_pending_) {
    latest_setup_.status = hal::StatusId::blocked;
    latest_setup_.reason = hal::ReasonId::output_not_committed;
    SetReason(reason, latest_setup_.reason);
    return latest_setup_.status;
  }
  if (request.mode != HardwareExecutionSetupMode::hal_live &&
      request.mode != HardwareExecutionSetupMode::shadow_only) {
    latest_setup_.status = hal::StatusId::blocked;
    latest_setup_.reason = hal::ReasonId::malformed_contract;
    SetReason(reason, latest_setup_.reason);
    return latest_setup_.status;
  }

  latest_setup_.resolution =
      hal::ResolveSemanticStages(*registry_, request.resolution);
  latest_setup_.domain_plan = hal::PlanExecutionDomains(
      *registry_, latest_setup_.resolution, hal::CrossAdapterPolicy::reject);
  if (request.compare_legacy_resolution) {
    latest_setup_.shadow_compared = true;
    latest_setup_.shadow_comparison = hal::CompareShadowResolution(
        request.legacy_resolution, latest_setup_.resolution);
  }

  if (!IsSetupSuccess(latest_setup_.resolution.status)) {
    latest_setup_.status = latest_setup_.resolution.status;
    latest_setup_.reason = latest_setup_.resolution.reason;
    SetReason(reason, latest_setup_.reason);
    return latest_setup_.status;
  }
  if (!IsSetupSuccess(latest_setup_.domain_plan.status)) {
    latest_setup_.status = latest_setup_.domain_plan.status;
    latest_setup_.reason = latest_setup_.domain_plan.reason;
    SetReason(reason, latest_setup_.reason);
    return latest_setup_.status;
  }

  if (request.mode == HardwareExecutionSetupMode::shadow_only) {
    // Leaving a prior HAL-live setup is a setup-time action. No frame can
    // dynamically route back to an old implementation.
    if (session_.prepared())
      session_.Invalidate(hal::ReasonId::reconfiguration_required);
    active_resolution_ = latest_setup_.resolution;
    active_domain_plan_ = latest_setup_.domain_plan;
    active_mode_ = request.mode;
    passthrough_ = active_domain_plan_.passthrough;
    local_readiness_ = {};
    local_readiness_.status = latest_setup_.resolution.status;
    local_readiness_.reason = latest_setup_.resolution.reason;
    local_readiness_.requested_backend =
        latest_setup_.resolution.requested_backend;
    local_readiness_.selected_backend =
        latest_setup_.resolution.selected_backend;
    local_readiness_.provider = latest_setup_.resolution.selected_provider;
    local_readiness_.available = latest_setup_.resolution.available;
    local_readiness_.degraded = latest_setup_.resolution.degraded;
    latest_setup_.status = latest_setup_.domain_plan.status;
    latest_setup_.reason = latest_setup_.domain_plan.reason;
    return latest_setup_.status;
  }

  if (latest_setup_.domain_plan.passthrough) {
    ActivatePassthrough(request);
    latest_setup_.status = hal::StatusId::passthrough;
    latest_setup_.reason = hal::ReasonId::none;
    return latest_setup_.status;
  }

  hal::ReasonId validation_reason = hal::ReasonId::none;
  if (!ValidateExactProgram(request, latest_setup_.resolution,
                            latest_setup_.domain_plan, &validation_reason)) {
    latest_setup_.status =
        validation_reason == hal::ReasonId::resource_exhausted
            ? hal::StatusId::resource_exhausted
            : hal::StatusId::blocked;
    latest_setup_.reason = validation_reason;
    SetReason(reason, validation_reason);
    return latest_setup_.status;
  }

  auto *adapter = latest_setup_.domain_plan.islands[0].resident_adapter;
  hal::ReadinessSnapshot readiness;
  hal::ReasonId configure_reason = hal::ReasonId::none;
  const hal::StatusId configure_status = session_.Configure(
      request.prepared_program, *adapter, &readiness, &configure_reason);
  latest_setup_.status = configure_status;
  latest_setup_.reason = configure_reason;
  if (configure_status != hal::StatusId::ready &&
      configure_status != hal::StatusId::degraded) {
    // PreparedProgramSession owns exact old-program retention policy. If the
    // candidate invalidated the old authority, do not retain stale facade
    // metadata.
    if (!session_.prepared())
      ClearActive();
    SetReason(reason, configure_reason);
    return configure_status;
  }

  active_resolution_ = latest_setup_.resolution;
  active_domain_plan_ = latest_setup_.domain_plan;
  active_mode_ = HardwareExecutionSetupMode::hal_live;
  passthrough_ = false;
  telemetry_from_session_ = true;
  local_readiness_ = {};
  local_committed_frame_ = {};
  return configure_status;
}

bool HardwareExecutionRuntime::ValidateExactProgram(
    const HardwareExecutionSetupRequest &request,
    const hal::ResolutionResult &resolution, const hal::DomainPlan &domain_plan,
    hal::ReasonId *reason) const noexcept {
  SetReason(reason, hal::ReasonId::none);
  const auto &program = request.prepared_program;
  hal::ReasonId contract_reason = hal::ReasonId::none;
  if (!hal::ValidatePreparedProgram(program, &contract_reason)) {
    SetReason(reason, contract_reason);
    return false;
  }
  if (domain_plan.island_count != 1) {
    SetReason(reason, hal::ReasonId::cross_adapter_residency_forbidden);
    return false;
  }
  const auto &island = domain_plan.islands[0];
  if (island.resident_adapter == nullptr ||
      island.adapter_instance_id != program.owner_adapter_instance_id ||
      island.execution_domain != program.execution_domain) {
    SetReason(reason, island.execution_domain != program.execution_domain
                          ? hal::ReasonId::wrong_execution_domain
                          : hal::ReasonId::foreign_adapter);
    return false;
  }
  const hal::BackendId requested_backend =
      request.resolution.backend.mode ==
              hal::BackendRequestMode::explicit_backend
          ? request.resolution.backend.backend
          : resolution.selected_backend;
  if (program.requested_backend != requested_backend ||
      program.strict_backend_selection != request.resolution.backend.strict ||
      program.requires_upload != island.requires_upload ||
      program.requires_final_readback != island.requires_final_readback ||
      domain_plan.expected_transfers.intermediate_readbacks != 0 ||
      domain_plan.expected_transfers.intermediate_uploads != 0) {
    SetReason(reason, program.strict_backend_selection &&
                              program.requested_backend !=
                                  program.execution_domain.backend
                          ? hal::ReasonId::strict_backend_unavailable
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
    const auto &prepared = program.stages[prepared_index];
    if (prepared.stage_id != resolved.intent.stage_id ||
        prepared.semantic != resolved.intent.semantic ||
        prepared.primitive != hal::PrimitiveOperationId::invalid ||
        prepared.provider != resolved.provider ||
        prepared.backend != resolved.selected_backend ||
        prepared.execution_domain != resolved.execution_domain ||
        prepared.required != resolved.intent.required ||
        prepared.failure_action != resolved.intent.failure_action) {
      SetReason(reason, prepared.execution_domain != resolved.execution_domain
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
      if (tail.id != resolved.cpu_tail ||
          tail.scratch_bytes != resolved.cpu_tail_scratch_bytes ||
          tail.placement != resolved.cpu_tail_placement ||
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

HardwareFrameResult HardwareExecutionRuntime::ExecuteHostFrame(
    const hal::HostFrameBinding &binding) noexcept {
  HardwareFrameResult frame;
  if (stopped_) {
    frame.status = hal::StatusId::stopped;
    frame.reason = hal::ReasonId::shutdown_requested;
    return frame;
  }
  if (active_mode_ == HardwareExecutionSetupMode::shadow_only) {
    frame.status = hal::StatusId::blocked;
    frame.reason = hal::ReasonId::stale_program;
    return frame;
  }
  if (frame_pending_) {
    frame.status = hal::StatusId::failed;
    frame.reason = hal::ReasonId::output_not_committed;
    return frame;
  }
  if (passthrough_) {
    if (!ValidatePassthroughBinding(binding)) {
      frame.status = hal::StatusId::blocked;
      frame.reason = hal::ReasonId::host_frame_binding_invalid;
      return frame;
    }
    if (passthrough_token_sequence_ ==
        std::numeric_limits<std::uint64_t>::max()) {
      frame.status = hal::StatusId::resource_exhausted;
      frame.reason = hal::ReasonId::resource_exhausted;
      return frame;
    }
    ++passthrough_token_sequence_;
    pending_completion_ = {passthrough_token_sequence_, 0, 0,
                           binding.capture_sequence, true};
    frame.completion = pending_completion_;
    frame.output.descriptor = binding.output_descriptor;
    frame.output.planes = binding.output_planes;
    frame.output.plane_count = binding.output_plane_count;
    frame.output.capture_sequence = binding.capture_sequence;
    frame.output.output_ready = true;
    frame.output_ready = true;
    frame.passthrough = true;
    frame.status = hal::StatusId::passthrough;
    ++local_telemetry_.execution_attempts;
    ++local_telemetry_.execution_successes;
    local_telemetry_.runtime_status = hal::StatusId::passthrough;
    local_telemetry_.runtime_reason = hal::ReasonId::none;
    frame_pending_ = true;
    return frame;
  }
  if (!session_.prepared()) {
    frame.status = hal::StatusId::blocked;
    frame.reason = hal::ReasonId::stale_program;
    return frame;
  }

  hal::HostFrameTokens tokens;
  hal::ReasonId bind_reason = hal::ReasonId::none;
  const hal::StatusId bind =
      session_.BindHostFrame(binding, &tokens, &bind_reason);
  if (bind != hal::StatusId::ready) {
    frame.status = bind;
    frame.reason = bind_reason;
    return frame;
  }
  hal::ExecuteRequest request;
  request.invalidation_key = session_.program().invalidation_key;
  request.execution_domain = session_.program().execution_domain;
  request.capture_sequence = binding.capture_sequence;
  request.input = tokens.input;
  const hal::ExecuteResult executed = session_.Execute(request);
  frame.status = executed.status;
  frame.reason = executed.reason;
  frame.completion = executed.completion;
  const bool execution_succeeded =
      executed.status == hal::StatusId::executed ||
      executed.status == hal::StatusId::degraded ||
      executed.status == hal::StatusId::passthrough;
  if (execution_succeeded && executed.completion.complete &&
      executed.completion.token_id != 0) {
    pending_completion_ = executed.completion;
    frame_pending_ = true;
  }
  if (!frame_pending_)
    return frame;

  hal::ReasonId resolve_reason = hal::ReasonId::none;
  const hal::StatusId resolved = session_.ResolveHostFrameOutput(
      executed.completion, executed.output, &frame.output, &resolve_reason);
  if (resolved != hal::StatusId::ready) {
    frame.status = resolved;
    frame.reason = resolve_reason;
    return frame;
  }
  frame.output_ready = true;
  return frame;
}

hal::StatusId
HardwareExecutionRuntime::CommitOutput(const hal::CompletionToken &completion,
                                       bool output_write_succeeded,
                                       hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  if (!frame_pending_ || !SameCompletion(completion, pending_completion_)) {
    SetReason(reason, hal::ReasonId::output_not_committed);
    return hal::StatusId::failed;
  }
  if (passthrough_) {
    frame_pending_ = false;
    pending_completion_ = {};
    if (!output_write_succeeded) {
      SetReason(reason, hal::ReasonId::output_write_failed);
      return hal::StatusId::failed;
    }
    ++local_telemetry_.committed_output_frames;
    local_committed_frame_ = {};
    local_committed_frame_.valid = true;
    local_committed_frame_.capture_sequence = completion.capture_sequence;
    local_committed_frame_.output_commit_sequence =
        local_telemetry_.committed_output_frames;
    return hal::StatusId::ready;
  }

  const hal::StatusId status =
      session_.CommitOutput({completion, output_write_succeeded}, reason);
  // The session consumes an exact pending completion for either writer result.
  frame_pending_ = false;
  pending_completion_ = {};
  return status;
}

bool HardwareExecutionRuntime::ValidatePassthroughBinding(
    const hal::HostFrameBinding &binding) const noexcept {
  if (binding.capture_sequence == 0 ||
      binding.ownership != hal::OwnershipModel::borrowed_until_completion ||
      binding.lifetime != hal::ResourceLifetime::frame_borrowed ||
      !hal::ValidateImageDescriptor(binding.input_descriptor) ||
      !hal::ValidateImageDescriptor(binding.output_descriptor) ||
      !SameImageDescriptor(binding.input_descriptor,
                           binding.output_descriptor) ||
      binding.input_plane_count != binding.input_descriptor.plane_count ||
      binding.output_plane_count != binding.output_descriptor.plane_count ||
      binding.input_plane_count != binding.output_plane_count) {
    return false;
  }
  for (std::size_t index = 0; index < binding.input_plane_count; ++index) {
    const auto &input = binding.input_planes[index];
    const auto &output = binding.output_planes[index];
    if (input.read_data == nullptr || output.write_data == nullptr ||
        input.read_data != output.write_data ||
        input.span_bytes != output.span_bytes ||
        input.row_stride_bytes != output.row_stride_bytes ||
        input.span_bytes < binding.input_descriptor.planes[index].size_bytes ||
        input.row_stride_bytes !=
            binding.input_descriptor.planes[index].row_stride_bytes) {
      return false;
    }
  }
  return true;
}

void HardwareExecutionRuntime::ActivatePassthrough(
    const HardwareExecutionSetupRequest &) noexcept {
  if (telemetry_from_session_)
    local_telemetry_ = session_.telemetry();
  if (session_.prepared())
    session_.Invalidate(hal::ReasonId::reconfiguration_required);
  telemetry_from_session_ = false;
  active_resolution_ = latest_setup_.resolution;
  active_domain_plan_ = latest_setup_.domain_plan;
  active_mode_ = HardwareExecutionSetupMode::hal_live;
  passthrough_ = true;
  frame_pending_ = false;
  pending_completion_ = {};
  local_readiness_ = {};
  local_readiness_.status = hal::StatusId::passthrough;
  local_readiness_.reason = hal::ReasonId::none;
  local_readiness_.requested_backend = active_resolution_.requested_backend;
  local_readiness_.available = true;
  local_telemetry_.runtime_status = hal::StatusId::passthrough;
  local_telemetry_.runtime_reason = hal::ReasonId::none;
  local_telemetry_.active_backend = hal::BackendId::invalid;
  local_telemetry_.active_provider = hal::SemanticEffectProviderId::invalid;
  local_committed_frame_ = {};
}

void HardwareExecutionRuntime::Invalidate(hal::ReasonId reason) noexcept {
  if (telemetry_from_session_)
    local_telemetry_ = session_.telemetry();
  if (session_.prepared())
    session_.Invalidate(reason);
  telemetry_from_session_ = false;
  frame_pending_ = false;
  pending_completion_ = {};
  ClearActive();
  local_readiness_.status = reason == hal::ReasonId::device_lost
                                ? hal::StatusId::quarantined
                                : hal::StatusId::stale;
  local_readiness_.reason = reason;
  local_readiness_.available = false;
}

void HardwareExecutionRuntime::Shutdown() noexcept {
  if (stopped_)
    return;
  if (telemetry_from_session_)
    local_telemetry_ = session_.telemetry();
  session_.Shutdown();
  telemetry_from_session_ = false;
  frame_pending_ = false;
  pending_completion_ = {};
  ClearActive();
  stopped_ = true;
  local_readiness_.status = hal::StatusId::stopped;
  local_readiness_.reason = hal::ReasonId::shutdown_requested;
  local_readiness_.available = false;
  local_telemetry_.runtime_status = hal::StatusId::stopped;
  local_telemetry_.runtime_reason = hal::ReasonId::shutdown_requested;
}

void HardwareExecutionRuntime::ClearActive() noexcept {
  active_resolution_ = {};
  active_domain_plan_ = {};
  passthrough_ = false;
}

const hal::ReadinessSnapshot &
HardwareExecutionRuntime::readiness() const noexcept {
  return session_.prepared() && !passthrough_ ? session_.readiness()
                                              : local_readiness_;
}

const hal::ExecutionTelemetry &
HardwareExecutionRuntime::telemetry() const noexcept {
  return telemetry_from_session_ ? session_.telemetry() : local_telemetry_;
}

const hal::CommittedFrameSnapshot &
HardwareExecutionRuntime::latest_committed_frame() const noexcept {
  return session_.prepared() && !passthrough_
             ? session_.latest_committed_frame()
             : local_committed_frame_;
}

bool HardwareExecutionRuntime::prepared() const noexcept {
  return !stopped_ && active_mode_ == HardwareExecutionSetupMode::hal_live &&
         (passthrough_ || session_.prepared());
}

} // namespace studiocast::video
