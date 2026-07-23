#include "core/open_video/open_cuda_hal_adapter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

using namespace studiocast;

bool Require(bool condition, std::string_view message) {
  if (!condition)
    std::cerr << "[FAIL] " << message << '\n';
  return condition;
}

hal::ExecutionDomainId Domain(std::uint64_t adapter_id = 81,
                              std::uint64_t context_generation = 7,
                              std::uint64_t stream = 0xCAFE) {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::cuda;
  domain.api_family = hal::ApiFamily::cuda_driver;
  domain.physical_device.vendor_id = 0x10DE;
  domain.physical_device.device_id = 0x1CBB;
  domain.physical_device.stable_id[0] = 0x42;
  domain.logical_context_generation = context_generation;
  domain.allocator_family = hal::AllocatorFamily::cuda_driver;
  domain.memory_handle_family = hal::MemoryHandleFamily::cuda_device_pointer;
  domain.queue_or_stream_identity = stream;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::cuda_stream_ordered;
  domain.adapter_instance_id = adapter_id;
  return domain;
}

hal::ImageDescriptor Image() {
  hal::ImageDescriptor descriptor;
  descriptor.width = 64;
  descriptor.height = 48;
  descriptor.plane_count = 1;
  descriptor.format = hal::PixelFormat::rgb_u8;
  descriptor.channel_order = hal::ChannelOrder::rgb;
  descriptor.color_space = hal::ColorSpace::srgb;
  descriptor.color_range = hal::ColorRange::full;
  descriptor.orientation = hal::Orientation::top_left;
  descriptor.host_coherence = hal::HostCoherence::not_host_visible;
  descriptor.mutability = hal::Mutability::read_write;
  descriptor.aliasing = hal::AliasingPermission::forbidden;
  descriptor.base_alignment_bytes = 256;
  descriptor.planes[0].row_stride_bytes = 256;
  descriptor.planes[0].size_bytes = 256 * 48;
  descriptor.planes[0].width = 64;
  descriptor.planes[0].height = 48;
  descriptor.planes[0].alignment_bytes = 256;
  return descriptor;
}

hal::ImageDescriptor HostImage(hal::Mutability mutability) {
  hal::ImageDescriptor descriptor = Image();
  descriptor.host_coherence = hal::HostCoherence::coherent;
  descriptor.mutability = mutability;
  return descriptor;
}

hal::ModelArtifactIdentity Model(std::uint8_t key = 1) {
  hal::ModelArtifactIdentity model;
  model.model_id_hash[0] = key;
  model.artifact_set_hash[0] = static_cast<std::uint8_t>(key + 1);
  model.artifact_generation = key;
  return model;
}

hal::PreparedStage
Stage(std::uint32_t stage_id, hal::SemanticEffectId semantic,
      bool required = true,
      hal::FailureAction failure_action = hal::FailureAction::fail_closed) {
  hal::PreparedStage stage;
  stage.stage_id = stage_id;
  stage.semantic = semantic;
  stage.provider = hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  stage.backend = hal::BackendId::cuda;
  stage.execution_domain = Domain();
  stage.input.kind = hal::StageIoKind::image;
  stage.input.image = Image();
  stage.input.image.mutability = hal::Mutability::read_only;
  stage.output.kind = hal::StageIoKind::image;
  stage.output.image = Image();
  stage.model = Model(static_cast<std::uint8_t>(stage_id));
  stage.required = required;
  stage.failure_action = failure_action;
  stage.circuit_breaker.consecutive_failure_limit = 2;
  stage.circuit_breaker.cooldown_frames = 5;
  stage.circuit_breaker.action =
      required ? hal::CircuitBreakerAction::quarantine_program
               : hal::CircuitBreakerAction::disable_stage;
  stage.scratch_bytes = 1024;
  return stage;
}

hal::PreparedProgram Program(std::uint64_t id = 101) {
  hal::PreparedProgram program;
  program.program_id = id;
  program.owner_adapter_instance_id = 81;
  program.execution_domain = Domain();
  program.host_input = HostImage(hal::Mutability::read_only);
  program.host_output = HostImage(hal::Mutability::write_only);
  program.invalidation_key.generations = {
      .configuration = 1,
      .geometry = 2,
      .model = 3,
      .provider = 4,
      .device = 5,
      .context = 7,
  };
  program.invalidation_key.canonical_plan_hash = 0xA1;
  program.invalidation_key.model_artifact_hash = 0xA2;
  program.invalidation_key.execution_domain_hash = 0xA3;
  program.budget.persistent_bytes = 64 * 1024;
  program.budget.scratch_bytes = 8 * 1024;
  program.budget.upload_staging_bytes = 16 * 1024;
  program.budget.readback_staging_bytes = 16 * 1024;
  program.budget.resource_count = 8;
  program.budget.in_flight_slots = 1;
  program.stages[0] = Stage(1, hal::SemanticEffectId::virtual_background_blur);
  program.stage_count = 1;
  program.requested_backend = hal::BackendId::cuda;
  program.strict_backend_selection = true;
  program.requires_upload = true;
  program.requires_final_readback = true;
  return program;
}

hal::ResourceHandle Resource(const hal::PreparedProgram &program,
                             std::uint64_t resource_id, bool output = false) {
  hal::ResourceHandle resource;
  resource.resource_id = resource_id;
  resource.allocator_owner_id = program.owner_adapter_instance_id;
  resource.lifetime_generation =
      program.execution_domain.logical_context_generation;
  resource.memory_domain =
      output ? (program.requires_final_readback
                    ? hal::MemoryDomain::readback_staging
                    : hal::MemoryDomain::device_local)
             : (program.requires_upload ? hal::MemoryDomain::upload_staging
                                        : hal::MemoryDomain::device_local);
  resource.lifetime = hal::ResourceLifetime::prepared_program;
  resource.synchronization_state = hal::SynchronizationState::ready;
  resource.execution_domain = program.execution_domain;
  resource.size_bytes = program.stages[0].input.image.planes[0].size_bytes;
  resource.alignment_bytes = 256;
  resource.mutable_resource = true;
  resource.adapter_opaque[0] = resource_id + 1000;
  return resource;
}

hal::ExecuteRequest Request(const hal::PreparedProgram &program,
                            std::uint64_t sequence = 1) {
  hal::ExecuteRequest request;
  request.invalidation_key = program.invalidation_key;
  request.execution_domain = program.execution_domain;
  request.capture_sequence = sequence;
  request.input = Resource(program, 500);
  return request;
}

constexpr std::size_t kHostFrameBytes = 256 * 48;

hal::HostFrameBinding HostBinding(const hal::PreparedProgram &program,
                                  const std::byte *input, std::byte *output,
                                  std::uint64_t sequence) {
  hal::HostFrameBinding binding;
  binding.input_descriptor = program.host_input;
  binding.output_descriptor = program.host_output;
  binding.input_plane_count = 1;
  binding.output_plane_count = 1;
  binding.capture_sequence = sequence;
  binding.input_planes[0].read_data = input;
  binding.input_planes[0].span_bytes = kHostFrameBytes;
  binding.input_planes[0].row_stride_bytes = 256;
  binding.output_planes[0].write_data = output;
  binding.output_planes[0].span_bytes = kHostFrameBytes;
  binding.output_planes[0].row_stride_bytes = 256;
  return binding;
}

open_cuda::ResidentCapabilityFacts CompleteFacts() {
  open_cuda::ResidentCapabilityFacts facts;
  facts.cuda_driver_ready = true;
  facts.stable_device_identity = true;
  facts.exact_context_generation = true;
  facts.adapter_owned_allocator = true;
  facts.adapter_owned_stream = true;
  facts.ort_cuda_provider_ready = true;
  facts.ort_cuda_iobinding_ready = true;
  facts.ort_user_stream_bound = true;
  facts.sessions_prepared = true;
  facts.reusable_buffers_prepared = true;
  facts.bounded_resources = true;
  facts.steady_state_allocation_free = true;
  facts.steady_state_probe_free = true;
  facts.no_cpu_fallback = true;
  facts.shared_frame_analysis = true;
  facts.completion_synchronizations_per_frame = 1;
  facts.device_preprocess = true;
  facts.device_postprocess = true;
  facts.virtual_background_composite = true;
  facts.virtual_key_light = true;
  return facts;
}

class FakeExecutor final : public open_cuda::IPreparedResidentExecutor {
public:
  struct Slot {
    bool active = false;
    open_cuda::PreparedExecutorHandle handle{};
    hal::PreparedProgram program{};
    hal::HostFrameBinding binding{};
    hal::HostFrameTokens tokens{};
    bool host_bound = false;
  };

  bool fail_prepare = false;
  bool incomplete_evidence = false;
  bool mutate_evidence_layout = false;
  bool explicit_provider_sync = false;
  bool allocate_during_execute = false;
  bool hide_upload = false;
  bool hide_sync = false;
  bool add_intermediate_transfer = false;
  bool device_lost = false;
  bool optional_failure = false;
  bool required_failure = false;
  bool mutate_output = false;
  bool reset_fails = false;
  bool bind_wrong_domain = false;
  bool bind_wrong_generation = false;
  bool bind_foreign_allocator = false;
  bool bind_oversized = false;
  bool allocate_during_bind = false;
  bool allocate_during_resolve = false;
  bool mutate_resolution_descriptor = false;
  bool mutate_resolution_plane = false;
  std::uint64_t prepare_calls = 0;
  std::uint64_t execute_calls = 0;
  std::uint64_t bind_calls = 0;
  std::uint64_t resolve_calls = 0;
  std::uint64_t executed_stage_calls = 0;
  std::uint32_t last_disabled_stage_mask = 0;
  std::uint64_t reset_calls = 0;
  std::uint64_t destroy_calls = 0;
  std::uint64_t shutdown_calls = 0;
  open_cuda::ExecutorTelemetrySnapshot telemetry{};

  open_cuda::ExecutorPrepareResult
  Prepare(const hal::PreparedProgram &program) noexcept override {
    ++prepare_calls;
    open_cuda::ExecutorPrepareResult result;
    if (fail_prepare) {
      result.status = hal::StatusId::blocked;
      result.reason = hal::ReasonId::capability_evidence_incomplete;
      return result;
    }
    Slot *slot = nullptr;
    for (auto &candidate : slots_) {
      if (!candidate.active) {
        slot = &candidate;
        break;
      }
    }
    if (!slot) {
      result.status = hal::StatusId::resource_exhausted;
      result.reason = hal::ReasonId::resource_exhausted;
      return result;
    }

    result.status = hal::StatusId::ready;
    result.reason = hal::ReasonId::none;
    result.handle.handle_id = next_handle_++;
    result.handle.program_id = program.program_id;
    result.handle.execution_domain = program.execution_domain;
    result.handle.reserved_budget = program.budget;
    result.handle.prepared_output =
        Resource(program, 700 + result.handle.handle_id, true);
    result.handle.opaque[0] = result.handle.handle_id + 100;
    result.evidence.capability = CompleteFacts();
    if (incomplete_evidence)
      result.evidence.capability.no_cpu_fallback = false;
    if (explicit_provider_sync) {
      result.evidence.capability.ort_user_stream_bound = false;
      result.evidence.capability.explicit_ort_boundary_synchronization = true;
    }
    result.evidence.stage_count = program.stage_count;
    for (std::size_t index = 0; index < program.stage_count; ++index) {
      const auto &stage = program.stages[index];
      auto &evidence = result.evidence.stages[index];
      evidence.stage_id = stage.stage_id;
      evidence.semantic = stage.semantic;
      evidence.provider = stage.provider;
      evidence.input = stage.input;
      evidence.output = stage.output;
      if (mutate_evidence_layout && index == 0)
        evidence.output.image.channel_order = hal::ChannelOrder::bgr;
      evidence.model = stage.model;
      evidence.production_qualified = true;
      evidence.cuda_device_input = true;
      evidence.cuda_device_output = true;
      evidence.cuda_iobinding = true;
      evidence.device_preprocess = true;
      evidence.device_postprocess = true;
      evidence.shared_analysis = true;
      evidence.temporal_history_device_resident =
          stage.semantic != hal::SemanticEffectId::video_noise_removal;
      evidence.cpu_fallback_disabled = true;
    }
    slot->active = true;
    slot->handle = result.handle;
    slot->program = program;
    return result;
  }

  hal::ExecuteResult
  Execute(const open_cuda::PreparedExecutorHandle &handle,
          const hal::ExecuteRequest &request) noexcept override {
    ++execute_calls;
    const Slot *slot = Find(handle);
    if (!slot)
      return {};
    last_disabled_stage_mask = request.disabled_stage_mask;
    if (allocate_during_execute)
      ++telemetry.allocation_requests;
    if (device_lost) {
      hal::ExecuteResult result;
      result.status = hal::StatusId::failed;
      result.reason = hal::ReasonId::device_lost;
      return result;
    }

    hal::ExecuteResult result;
    const bool required_stage_failed =
        required_failure && (request.disabled_stage_mask & 1u) == 0;
    const bool optional_stage_failed =
        optional_failure && (request.disabled_stage_mask & 1u) == 0;
    result.status = required_stage_failed
                        ? hal::StatusId::failed
                        : (optional_stage_failed ? hal::StatusId::degraded
                                                 : hal::StatusId::executed);
    result.reason =
        optional_stage_failed
            ? hal::ReasonId::optional_stage_failed_open
            : (required_stage_failed ? hal::ReasonId::required_stage_failed
                                     : hal::ReasonId::none);
    result.output =
        slot->host_bound ? slot->tokens.output : handle.prepared_output;
    if (mutate_output)
      ++result.output.size_bytes;
    result.output_valid = !required_stage_failed;
    result.stage_count = slot->program.stage_count;
    bool any_stage_ran = false;
    for (std::size_t index = 0; index < result.stage_count; ++index) {
      const auto &stage = slot->program.stages[index];
      result.stages[index].stage_id = stage.stage_id;
      result.stages[index].semantic = stage.semantic;
      result.stages[index].primitive = stage.primitive;
      const bool disabled =
          (request.disabled_stage_mask & (std::uint32_t{1} << index)) != 0;
      if (disabled) {
        result.stages[index].status =
            hal::StageExecutionStatus::disabled_preserved;
        result.stages[index].reason = hal::ReasonId::circuit_breaker_open;
        result.stages[index].valid_output_preserved = true;
        result.stages[index].elapsed_ns = 0;
      } else if (required_stage_failed && index == 0) {
        result.stages[index].status =
            hal::StageExecutionStatus::required_failed;
        result.stages[index].reason = hal::ReasonId::required_stage_failed;
        result.stages[index].elapsed_ns = 100 + index;
        any_stage_ran = true;
        ++executed_stage_calls;
      } else if (required_stage_failed) {
        result.stages[index].status = hal::StageExecutionStatus::not_run;
      } else if (optional_stage_failed && index == 0) {
        result.stages[index].status =
            hal::StageExecutionStatus::optional_failed_open;
        result.stages[index].reason = hal::ReasonId::optional_stage_failed_open;
        result.stages[index].valid_output_preserved = true;
        result.stages[index].elapsed_ns = 100 + index;
        any_stage_ran = true;
        ++executed_stage_calls;
      } else {
        result.stages[index].status = hal::StageExecutionStatus::succeeded;
        result.stages[index].elapsed_ns = 100 + index;
        any_stage_ran = true;
        ++executed_stage_calls;
      }
    }
    result.frame_elapsed_ns = 1000;
    result.transfers.uploads = slot->program.requires_upload ? 1 : 0;
    result.transfers.final_readbacks =
        slot->program.requires_final_readback && !required_stage_failed ? 1 : 0;
    result.transfers.intermediate_readbacks = add_intermediate_transfer ? 1 : 0;
    result.transfers.cpu_tail_boundaries = slot->program.cpu_tail_count;
    if (!hide_upload)
      telemetry.uploads += result.transfers.uploads;
    telemetry.final_readbacks += result.transfers.final_readbacks;
    telemetry.intermediate_readbacks += result.transfers.intermediate_readbacks;
    telemetry.cpu_tail_boundaries += result.transfers.cpu_tail_boundaries;
    if (!hide_sync) {
      ++telemetry.synchronizations;
      result.transfers.synchronizations = 1;
    }
    if (any_stage_ran)
      ++telemetry.shared_analysis_runs;
    return result;
  }

  hal::StatusId BindHostFrame(const open_cuda::PreparedExecutorHandle &handle,
                              const hal::HostFrameBinding &binding,
                              hal::HostFrameTokens *tokens,
                              hal::ReasonId *reason) noexcept override {
    ++bind_calls;
    Slot *slot = FindMutable(handle);
    if (!slot || !tokens) {
      if (reason)
        *reason = hal::ReasonId::stale_program;
      return hal::StatusId::stale;
    }
    if (allocate_during_bind)
      ++telemetry.allocation_requests;
    tokens->input = Resource(slot->program, 800 + handle.handle_id);
    tokens->output = Resource(slot->program, 900 + handle.handle_id, true);
    tokens->input.lifetime = hal::ResourceLifetime::frame_borrowed;
    tokens->output.lifetime = hal::ResourceLifetime::frame_borrowed;
    if (bind_wrong_domain)
      ++tokens->input.execution_domain.queue_or_stream_identity;
    if (bind_wrong_generation)
      --tokens->output.lifetime_generation;
    if (bind_foreign_allocator)
      ++tokens->input.allocator_owner_id;
    if (bind_oversized)
      tokens->output.size_bytes =
          handle.reserved_budget.readback_staging_bytes + 1;
    slot->binding = binding;
    slot->tokens = *tokens;
    slot->host_bound = true;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::StatusId
  ResolveHostFrameOutput(const open_cuda::PreparedExecutorHandle &handle,
                         const hal::CompletionToken &,
                         const hal::ResourceHandle &,
                         hal::HostFrameOutputResolution *resolution,
                         hal::ReasonId *reason) noexcept override {
    ++resolve_calls;
    const Slot *slot = Find(handle);
    if (!slot || !slot->host_bound || !resolution) {
      if (reason)
        *reason = hal::ReasonId::output_not_committed;
      return hal::StatusId::blocked;
    }
    if (allocate_during_resolve)
      ++telemetry.allocation_requests;
    resolution->descriptor = slot->binding.output_descriptor;
    resolution->planes = slot->binding.output_planes;
    resolution->plane_count = slot->binding.output_plane_count;
    resolution->capture_sequence = slot->binding.capture_sequence;
    resolution->output_ready = true;
    if (mutate_resolution_descriptor)
      resolution->descriptor.channel_order = hal::ChannelOrder::bgr;
    if (mutate_resolution_plane)
      ++resolution->planes[0].span_bytes;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::StatusId
  ResetTemporalState(const open_cuda::PreparedExecutorHandle &handle,
                     hal::ReasonId) noexcept override {
    ++reset_calls;
    return Find(handle) && !reset_fails ? hal::StatusId::ready
                                        : hal::StatusId::failed;
  }

  void
  DestroyPrepared(open_cuda::PreparedExecutorHandle *handle) noexcept override {
    if (!handle)
      return;
    for (auto &slot : slots_) {
      if (slot.active && slot.handle.handle_id == handle->handle_id) {
        slot = {};
        ++destroy_calls;
        break;
      }
    }
    *handle = {};
  }

  void Shutdown() noexcept override { ++shutdown_calls; }

  open_cuda::ExecutorTelemetrySnapshot Telemetry() const noexcept override {
    return telemetry;
  }

private:
  Slot *FindMutable(const open_cuda::PreparedExecutorHandle &handle) noexcept {
    for (auto &slot : slots_) {
      if (slot.active && slot.handle.handle_id == handle.handle_id &&
          slot.handle.program_id == handle.program_id &&
          slot.handle.execution_domain == handle.execution_domain) {
        return &slot;
      }
    }
    return nullptr;
  }

  const Slot *
  Find(const open_cuda::PreparedExecutorHandle &handle) const noexcept {
    for (const auto &slot : slots_) {
      if (slot.active && slot.handle.handle_id == handle.handle_id &&
          slot.handle.program_id == handle.program_id &&
          slot.handle.execution_domain == handle.execution_domain) {
        return &slot;
      }
    }
    return nullptr;
  }

  std::array<Slot, 8> slots_{};
  std::uint64_t next_handle_ = 1;
};

bool Configure(hal::PreparedProgramSession *session,
               open_cuda::ResidentAdapter *adapter,
               const hal::PreparedProgram &program,
               hal::StatusId expected = hal::StatusId::ready) {
  hal::ReadinessSnapshot readiness;
  hal::ReasonId reason = hal::ReasonId::none;
  const auto status =
      session->Configure(program, *adapter, &readiness, &reason);
  return Require(status == expected, "unexpected configure status") &&
         Require(readiness.available, "configured adapter must be available") &&
         Require(readiness.selected_backend == hal::BackendId::cuda,
                 "configured backend must be CUDA");
}

bool TestCapabilityIsExactAndDoesNotPromoteCpuTensorProviders() {
  auto facts = CompleteFacts();
  bool ok = true;
  for (const auto semantic : {hal::SemanticEffectId::virtual_background_blur,
                              hal::SemanticEffectId::virtual_background_remove,
                              hal::SemanticEffectId::virtual_background_replace,
                              hal::SemanticEffectId::virtual_key_light}) {
    const auto capability =
        open_cuda::EvaluateResidentCapability(semantic, facts);
    ok &=
        Require(capability.status == hal::StatusId::ready &&
                    capability.production_eligible && capability.exact_evidence,
                "qualified resident CUDA semantic should be ready");
  }
  for (const auto semantic :
       {hal::SemanticEffectId::video_noise_removal,
        hal::SemanticEffectId::eye_contact, hal::SemanticEffectId::auto_frame,
        hal::SemanticEffectId::mirror, hal::SemanticEffectId::vignette}) {
    const auto capability =
        open_cuda::EvaluateResidentCapability(semantic, facts);
    ok &= Require(capability.status == hal::StatusId::blocked &&
                      capability.reason ==
                          hal::ReasonId::semantic_effect_unsupported &&
                      !capability.production_eligible,
                  "CPU-tensor or unsupported semantics must remain blocked");
  }
  facts.no_cpu_fallback = false;
  ok &=
      Require(open_cuda::EvaluateResidentCapability(
                  hal::SemanticEffectId::virtual_background_blur, facts)
                      .reason == hal::ReasonId::capability_evidence_incomplete,
              "possible CPU fallback must fail closed");
  facts = CompleteFacts();
  facts.ort_cuda_provider_ready = false;
  ok &= Require(open_cuda::EvaluateResidentCapability(
                    hal::SemanticEffectId::virtual_background_blur, facts)
                        .reason == hal::ReasonId::semantic_provider_unavailable,
                "missing ORT CUDA provider must use the typed provider reason");
  return ok;
}

bool TestPreparedExecutionAndCommittedAttribution() {
  FakeExecutor executor;
  open_cuda::ResidentAdapter adapter(81, executor);
  hal::PreparedProgramSession session;
  const auto program = Program();
  if (!Configure(&session, &adapter, program))
    return false;

  auto result = session.Execute(Request(program, 11));
  bool ok = true;
  ok &= Require(result.status == hal::StatusId::executed &&
                    result.output_valid && result.completion.complete,
                "prepared CUDA execution should complete synchronously");
  ok &= Require(result.transfers.uploads == 1 &&
                    result.transfers.final_readbacks == 1 &&
                    result.transfers.intermediate_readbacks == 0 &&
                    result.transfers.intermediate_uploads == 0,
                "eligible resident island must use one upload/readback");
  ok &= Require(session.telemetry().active_backend == hal::BackendId::invalid &&
                    session.telemetry().active_backend_frames == 0,
                "execution alone must not commit active attribution");

  hal::ReasonId reason = hal::ReasonId::none;
  ok &= Require(session.CommitOutput({.completion = result.completion,
                                      .output_write_succeeded = true},
                                     &reason) == hal::StatusId::ready,
                "successful writer commit should succeed");
  ok &= Require(session.telemetry().active_backend == hal::BackendId::cuda &&
                    session.telemetry().active_backend_frames == 1 &&
                    session.telemetry().committed_output_frames == 1,
                "active CUDA attribution must commit after output write");
  ok &= Require(executor.prepare_calls == 1 && executor.execute_calls == 1 &&
                    executor.telemetry.allocation_requests == 0 &&
                    executor.telemetry.provider_probes == 0 &&
                    executor.telemetry.model_scans == 0 &&
                    executor.telemetry.session_constructions == 0,
                "steady-state execution must not perform setup work");
  return ok;
}

bool TestMalformedAndForeignProgramsFailClosed() {
  FakeExecutor executor;
  open_cuda::ResidentAdapter adapter(81, executor);
  bool ok = true;
  {
    executor.mutate_evidence_layout = true;
    auto program = Program();
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          hal::StatusId::blocked &&
                      reason == hal::ReasonId::capability_evidence_incomplete,
                  "prepared I/O layout evidence must match exactly");
    executor.mutate_evidence_layout = false;
  }
  {
    auto program = Program();
    program.host_input = program.stages[0].input.image;
    program.host_input.mutability = hal::Mutability::read_only;
    const std::uint64_t prepare_calls_before = executor.prepare_calls;
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          hal::StatusId::blocked &&
                      reason == hal::ReasonId::malformed_descriptor &&
                      executor.prepare_calls == prepare_calls_before,
                  "device-local CUDA descriptor must not qualify as a host "
                  "transport boundary");
  }
  {
    auto program = Program();
    program.stages[1] = Stage(2, hal::SemanticEffectId::virtual_key_light);
    program.stages[1].model = Model(9);
    program.stage_count = 2;
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          hal::StatusId::blocked &&
                      reason == hal::ReasonId::capability_evidence_incomplete,
                  "one shared resident island must bind one matte identity");
  }
  {
    auto program = Program();
    program.stages[0].model = {};
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          hal::StatusId::blocked &&
                      reason == hal::ReasonId::capability_evidence_incomplete,
                  "missing model/artifact identity must be blocked");
  }
  {
    auto program = Program();
    program.invalidation_key.generations.model = 0;
    program.invalidation_key.model_artifact_hash = 0;
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          hal::StatusId::blocked &&
                      reason == hal::ReasonId::capability_evidence_incomplete,
                  "model generation and artifact invalidation are required");
  }
  {
    auto program = Program();
    program.stages[0].semantic = hal::SemanticEffectId::video_noise_removal;
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          hal::StatusId::blocked &&
                      reason == hal::ReasonId::semantic_effect_unsupported,
                  "non-resident FastDVDnet path must remain blocked");
  }
  {
    auto program = Program();
    program.owner_adapter_instance_id = 90;
    program.execution_domain = Domain(90);
    program.stages[0].execution_domain = program.execution_domain;
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          hal::StatusId::blocked &&
                      reason == hal::ReasonId::foreign_adapter,
                  "foreign adapter ownership must be rejected");
  }
  {
    auto program = Program();
    program.execution_domain.api_family = hal::ApiFamily::onnxruntime;
    program.execution_domain.allocator_family =
        hal::AllocatorFamily::onnxruntime;
    program.execution_domain.memory_handle_family =
        hal::MemoryHandleFamily::ort_value;
    program.execution_domain.synchronization_protocol =
        hal::SynchronizationProtocol::ort_provider_synchronous;
    program.stages[0].execution_domain = program.execution_domain;
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                      hal::StatusId::blocked,
                  "CUDA and ORT domains must not be assumed interoperable");
  }
  {
    auto program = Program();
    program.strict_provider_selection = true;
    program.requested_provider =
        hal::SemanticEffectProviderId::open_video_onnxruntime_cpu;
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          hal::StatusId::blocked &&
                      reason == hal::ReasonId::semantic_provider_unavailable,
                  "strict provider selection must never fall back");
  }
  {
    auto program = Program();
    program.strict_execution_domain = true;
    program.requested_execution_domain = program.execution_domain;
    ++program.requested_execution_domain.queue_or_stream_identity;
    hal::PreparedProgramSession session;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          hal::StatusId::blocked &&
                      reason == hal::ReasonId::wrong_execution_domain,
                  "strict execution-domain selection must never fall back");
  }
  return ok;
}

bool TestStaleDomainGenerationAndResourceChecks() {
  FakeExecutor executor;
  open_cuda::ResidentAdapter adapter(81, executor);
  hal::PreparedProgramSession session;
  const auto program = Program();
  if (!Configure(&session, &adapter, program))
    return false;
  bool ok = true;

  auto stale = Request(program);
  ++stale.invalidation_key.generations.configuration;
  auto result = session.Execute(stale);
  ok &= Require(result.status == hal::StatusId::stale &&
                    result.reason == hal::ReasonId::reconfiguration_required,
                "stale invalidation key must request reconfiguration");

  auto foreign = Request(program);
  foreign.execution_domain.queue_or_stream_identity = 0xBAD;
  result = session.Execute(foreign);
  ok &= Require(result.reason == hal::ReasonId::wrong_execution_domain,
                "wrong stream/domain must be blocked");

  auto generation = Request(program);
  --generation.input.lifetime_generation;
  result = session.Execute(generation);
  ok &= Require(result.reason == hal::ReasonId::wrong_generation,
                "wrong resource generation must be blocked");

  auto resource = Request(program);
  resource.input.allocator_owner_id = 99;
  result = session.Execute(resource);
  ok &= Require(result.reason == hal::ReasonId::invalid_resource,
                "foreign resource allocator must be blocked");

  // Exercise the adapter's independent guard rather than the outer session.
  FakeExecutor direct_executor;
  open_cuda::ResidentAdapter direct_adapter(81, direct_executor);
  hal::PreparedProgramHandle handle;
  hal::ReadinessSnapshot readiness;
  hal::ReasonId reason = hal::ReasonId::none;
  ok &= Require(direct_adapter.Prepare(program, &handle, &readiness, &reason) ==
                    hal::StatusId::ready,
                "direct adapter prepare should succeed");
  auto direct_request = Request(program);
  direct_request.input.memory_domain = hal::MemoryDomain::device_local;
  result = direct_adapter.Execute(handle, direct_request);
  ok &= Require(result.status == hal::StatusId::blocked &&
                    result.reason == hal::ReasonId::invalid_resource &&
                    direct_executor.execute_calls == 0,
                "adapter must reject wrong boundary memory before executor");
  auto forged_handle = handle;
  ++forged_handle.reserved_budget.upload_staging_bytes;
  result = direct_adapter.Execute(forged_handle, Request(program));
  ok &= Require(result.status == hal::StatusId::stale &&
                    result.reason == hal::ReasonId::stale_program &&
                    direct_executor.execute_calls == 0,
                "mutated prepared-handle budget must be rejected");
  direct_adapter.DestroyPrepared(&handle);
  return ok;
}

bool TestResourceExhaustionAndOldPlanRetention() {
  bool ok = true;
  {
    FakeExecutor executor;
    open_cuda::ResidentAdapter adapter(81, executor);
    std::array<hal::PreparedProgramSession,
               open_cuda::ResidentAdapter::kMaxPreparedPrograms>
        sessions;
    for (std::size_t index = 0; index < sessions.size(); ++index) {
      ok &= Configure(&sessions[index], &adapter, Program(200 + index));
    }
    hal::PreparedProgramSession exhausted;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(
        exhausted.Configure(Program(300), adapter, &readiness, &reason) ==
                hal::StatusId::resource_exhausted &&
            reason == hal::ReasonId::resource_exhausted,
        "bounded prepared slots must report resource exhaustion");
  }
  {
    FakeExecutor executor;
    open_cuda::ResidentAdapter adapter(81, executor);
    hal::PreparedProgramSession rollback;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    const auto old_program = Program(400);
    ok &= Configure(&rollback, &adapter, old_program);
    executor.fail_prepare = true;
    auto next = Program(401);
    ok &= Require(rollback.Configure(next, adapter, &readiness, &reason) ==
                      hal::StatusId::blocked,
                  "candidate setup failure should be reported");
    executor.fail_prepare = false;
    auto result = rollback.Execute(Request(old_program, 9));
    ok &= Require(result.status == hal::StatusId::executed,
                  "failed reconfiguration must retain the old prepared plan");
  }
  return ok;
}

bool TestCommonHostBindingAndOutputResolution() {
  alignas(256) static std::array<std::byte, kHostFrameBytes> input{};
  alignas(256) static std::array<std::byte, kHostFrameBytes> output{};
  FakeExecutor executor;
  open_cuda::ResidentAdapter adapter(81, executor);
  hal::PreparedProgramSession session;
  const auto program = Program(350);
  if (!Configure(&session, &adapter, program))
    return false;

  auto binding = HostBinding(program, input.data(), output.data(), 1);
  hal::HostFrameTokens tokens;
  hal::ReasonId reason = hal::ReasonId::none;
  bool ok = Require(
      session.BindHostFrame(binding, &tokens, &reason) == hal::StatusId::ready,
      "common host binding should use the prepared CUDA executor");
  ok &= Require(
      tokens.input.lifetime == hal::ResourceLifetime::frame_borrowed &&
          tokens.output.lifetime == hal::ResourceLifetime::frame_borrowed &&
          executor.bind_calls == 1 &&
          executor.telemetry.allocation_requests == 0,
      "host tokens must be exact borrowed preprepared resources");

  auto request = Request(program, 1);
  request.input = tokens.input;
  const auto result = session.Execute(request);
  ok &=
      Require(result.status == hal::StatusId::executed &&
                  result.output.resource_id == tokens.output.resource_id,
              "host-bound execution must return the exact bound output token");

  auto foreign_completion = result.completion;
  ++foreign_completion.adapter_instance_id;
  hal::HostFrameOutputResolution resolution;
  ok &= Require(session.ResolveHostFrameOutput(
                    foreign_completion, result.output, &resolution, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::output_not_committed &&
                    executor.resolve_calls == 0,
                "foreign completion tokens must not reach the executor");
  ok &=
      Require(session.ResolveHostFrameOutput(result.completion, result.output,
                                             &resolution,
                                             &reason) == hal::StatusId::ready &&
                  resolution.output_ready && resolution.capture_sequence == 1 &&
                  resolution.planes[0].write_data == output.data() &&
                  executor.resolve_calls == 1,
              "host output must resolve to the exact borrowed output plane");
  ok &= Require(session.CommitOutput({result.completion, true}, &reason) ==
                    hal::StatusId::ready,
                "resolved host output should commit");

  binding.capture_sequence = 2;
  ok &= Require(session.BindHostFrame(binding, &tokens, &reason) ==
                    hal::StatusId::ready,
                "prepared binding state must be reusable without allocation");
  request = Request(program, 2);
  request.input = tokens.input;
  const auto second = session.Execute(request);
  executor.mutate_resolution_plane = true;
  ok &= Require(session.ResolveHostFrameOutput(second.completion, second.output,
                                               &resolution, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::host_frame_binding_invalid,
                "mutated output-plane resolution must fail closed");
  ok &= Require(session.CommitOutput({second.completion, false}, &reason) ==
                    hal::StatusId::failed,
                "invalid host output must not be committed");
  return ok;
}

bool TestHostBindingRejectsMalformedForeignAndExhaustedTokens() {
  alignas(256) static std::array<std::byte, kHostFrameBytes> input{};
  alignas(256) static std::array<std::byte, kHostFrameBytes> output{};
  FakeExecutor executor;
  open_cuda::ResidentAdapter adapter(81, executor);
  const auto program = Program(360);
  hal::PreparedProgramHandle handle;
  hal::ReadinessSnapshot readiness;
  hal::ReasonId reason = hal::ReasonId::none;
  if (!Require(adapter.Prepare(program, &handle, &readiness, &reason) ==
                   hal::StatusId::ready,
               "direct host-binding prepare should succeed")) {
    return false;
  }

  auto binding = HostBinding(program, input.data(), output.data(), 1);
  hal::HostFrameTokens tokens;
  bool ok = true;
  auto malformed = binding;
  ++malformed.input_descriptor.planes[0].row_stride_bytes;
  ok &= Require(adapter.BindHostFrame(handle, malformed, &tokens, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::host_frame_binding_invalid &&
                    executor.bind_calls == 0,
                "malformed host descriptors must fail before executor binding");
  malformed = binding;
  malformed.input_descriptor = program.stages[0].input.image;
  malformed.input_descriptor.mutability = hal::Mutability::read_only;
  ok &= Require(adapter.BindHostFrame(handle, malformed, &tokens, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::host_frame_binding_invalid &&
                    executor.bind_calls == 0,
                "resident CUDA descriptor must not bind borrowed host storage");

  executor.bind_wrong_domain = true;
  ok &= Require(adapter.BindHostFrame(handle, binding, &tokens, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::wrong_execution_domain,
                "foreign host-token domains must be rejected");
  executor.bind_wrong_domain = false;
  executor.bind_wrong_generation = true;
  ok &= Require(adapter.BindHostFrame(handle, binding, &tokens, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::wrong_generation,
                "stale host-token generations must be rejected");
  executor.bind_wrong_generation = false;
  executor.bind_foreign_allocator = true;
  ok &= Require(adapter.BindHostFrame(handle, binding, &tokens, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::invalid_resource,
                "foreign host-token ownership must be rejected");
  executor.bind_foreign_allocator = false;
  executor.bind_oversized = true;
  ok &= Require(adapter.BindHostFrame(handle, binding, &tokens, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::invalid_resource,
                "host-token budget exhaustion must fail closed");
  executor.bind_oversized = false;
  executor.allocate_during_bind = true;
  ok &= Require(adapter.BindHostFrame(handle, binding, &tokens, &reason) ==
                        hal::StatusId::failed &&
                    reason == hal::ReasonId::synchronization_failed,
                "per-frame binding allocation must be rejected");
  executor.allocate_during_bind = false;

  ok &= Require(adapter.BindHostFrame(handle, binding, &tokens, &reason) ==
                    hal::StatusId::ready,
                "valid host binding should remain possible after rejection");
  auto request = Request(program, 1);
  request.input = tokens.input;
  const auto result = adapter.Execute(handle, request);
  auto wrong_output = result.output;
  ++wrong_output.execution_domain.queue_or_stream_identity;
  hal::HostFrameOutputResolution resolution;
  ok &= Require(adapter.ResolveHostFrameOutput(
                    handle, result.completion, wrong_output, &resolution,
                    &reason) == hal::StatusId::blocked &&
                    reason == hal::ReasonId::wrong_execution_domain &&
                    executor.resolve_calls == 0,
                "foreign output domains must fail before resolution");
  auto wrong_generation = result.output;
  --wrong_generation.lifetime_generation;
  ok &= Require(adapter.ResolveHostFrameOutput(
                    handle, result.completion, wrong_generation, &resolution,
                    &reason) == hal::StatusId::blocked &&
                    reason == hal::ReasonId::wrong_generation &&
                    executor.resolve_calls == 0,
                "stale output generations must fail before resolution");
  executor.mutate_resolution_descriptor = true;
  ok &= Require(adapter.ResolveHostFrameOutput(
                    handle, result.completion, result.output, &resolution,
                    &reason) == hal::StatusId::blocked &&
                    reason == hal::ReasonId::host_frame_binding_invalid,
                "mutated output descriptors must fail closed");
  executor.mutate_resolution_descriptor = false;
  executor.allocate_during_resolve = true;
  ok &= Require(adapter.ResolveHostFrameOutput(
                    handle, result.completion, result.output, &resolution,
                    &reason) == hal::StatusId::failed &&
                    reason == hal::ReasonId::synchronization_failed,
                "per-frame resolution allocation must be rejected");
  adapter.DestroyPrepared(&handle);
  return ok;
}

bool TestOptionalFailurePreservesValidOutput() {
  FakeExecutor executor;
  executor.optional_failure = true;
  open_cuda::ResidentAdapter adapter(81, executor);
  auto program = Program();
  program.stages[0] =
      Stage(1, hal::SemanticEffectId::virtual_background_blur, false,
            hal::FailureAction::fail_open_preserve_input);
  hal::PreparedProgramSession session;
  if (!Configure(&session, &adapter, program))
    return false;
  const auto result = session.Execute(Request(program));
  return Require(
      result.status == hal::StatusId::degraded &&
          result.reason == hal::ReasonId::optional_stage_failed_open &&
          result.output_valid && result.stages[0].valid_output_preserved,
      "optional failure must preserve a valid resident frame");
}

bool TestDisabledOptionalStageIsSkippedAndPreserved() {
  FakeExecutor executor;
  executor.optional_failure = true;
  open_cuda::ResidentAdapter adapter(81, executor);
  auto program = Program(370);
  program.stages[0] =
      Stage(1, hal::SemanticEffectId::virtual_background_blur, false,
            hal::FailureAction::fail_open_preserve_input);
  program.stages[0].circuit_breaker.consecutive_failure_limit = 2;
  program.stages[0].circuit_breaker.cooldown_frames = 0;
  hal::PreparedProgramSession session;
  if (!Configure(&session, &adapter, program))
    return false;

  hal::ReasonId reason = hal::ReasonId::none;
  bool ok = true;
  for (std::uint64_t sequence = 1; sequence <= 2; ++sequence) {
    const auto failed = session.Execute(Request(program, sequence));
    ok &= Require(failed.status == hal::StatusId::degraded &&
                      failed.stages[0].status ==
                          hal::StageExecutionStatus::optional_failed_open,
                  "optional stage should fail open before breaker activation");
    ok &= Require(session.CommitOutput({failed.completion, true}, &reason) ==
                      hal::StatusId::ready,
                  "preserved optional-failure output should commit");
  }
  ok &= Require(session.disabled_stage_mask() == 1u,
                "optional circuit breaker should disable the failed stage");
  const auto stage_calls_before = executor.executed_stage_calls;
  const auto analyses_before = executor.telemetry.shared_analysis_runs;
  executor.optional_failure = false;
  const auto disabled = session.Execute(Request(program, 3));
  ok &= Require(
      disabled.status == hal::StatusId::executed && disabled.stage_count == 1 &&
          disabled.stages[0].status ==
              hal::StageExecutionStatus::disabled_preserved &&
          disabled.stages[0].reason == hal::ReasonId::circuit_breaker_open &&
          disabled.stages[0].valid_output_preserved &&
          disabled.stages[0].elapsed_ns == 0 &&
          executor.last_disabled_stage_mask == 1u &&
          executor.executed_stage_calls == stage_calls_before &&
          executor.telemetry.shared_analysis_runs == analyses_before,
      "disabled optional stage must be zero-time preserved and skipped");
  ok &= Require(session.CommitOutput({disabled.completion, true}, &reason) ==
                    hal::StatusId::ready,
                "disabled preserved output should remain valid");
  return ok;
}

bool TestRequiredFailureFailsClosedInsideAdapter() {
  FakeExecutor executor;
  executor.required_failure = true;
  open_cuda::ResidentAdapter adapter(81, executor);
  const auto program = Program();
  hal::PreparedProgramHandle handle;
  hal::ReadinessSnapshot readiness;
  hal::ReasonId reason = hal::ReasonId::none;
  if (!Require(adapter.Prepare(program, &handle, &readiness, &reason) ==
                   hal::StatusId::ready,
               "direct adapter prepare should succeed")) {
    return false;
  }
  const auto result = adapter.Execute(handle, Request(program));
  adapter.DestroyPrepared(&handle);
  return Require(result.status == hal::StatusId::failed &&
                     result.reason == hal::ReasonId::required_stage_failed,
                 "required-stage failure must fail closed in the adapter");
}

bool TestRequiredStagesQuarantineOrRequestRebuild() {
  bool ok = true;
  {
    FakeExecutor executor;
    executor.required_failure = true;
    open_cuda::ResidentAdapter adapter(81, executor);
    auto program = Program(380);
    program.stages[0].circuit_breaker.consecutive_failure_limit = 1;
    program.stages[0].circuit_breaker.action =
        hal::CircuitBreakerAction::quarantine_program;
    hal::PreparedProgramSession session;
    if (!Configure(&session, &adapter, program))
      return false;
    const auto failed = session.Execute(Request(program, 1));
    ok &= Require(failed.status == hal::StatusId::failed &&
                      failed.reason == hal::ReasonId::required_stage_failed &&
                      session.quarantined(),
                  "required failure must quarantine under fail-closed policy");
    const auto execute_calls = executor.execute_calls;
    const auto quarantined = session.Execute(Request(program, 2));
    ok &= Require(quarantined.status == hal::StatusId::quarantined &&
                      executor.execute_calls == execute_calls,
                  "quarantined required stage must not re-enter the executor");
  }
  {
    FakeExecutor executor;
    executor.required_failure = true;
    open_cuda::ResidentAdapter adapter(81, executor);
    auto program = Program(381);
    program.stages[0].circuit_breaker.consecutive_failure_limit = 1;
    program.stages[0].circuit_breaker.action =
        hal::CircuitBreakerAction::request_rebuild;
    hal::PreparedProgramSession session;
    if (!Configure(&session, &adapter, program))
      return false;
    const auto failed = session.Execute(Request(program, 1));
    ok &= Require(failed.status == hal::StatusId::failed &&
                      session.rebuild_requested(),
                  "required failure must request rebuild under rebuild policy");
    const auto execute_calls = executor.execute_calls;
    const auto stale = session.Execute(Request(program, 2));
    ok &= Require(stale.status == hal::StatusId::stale &&
                      stale.reason == hal::ReasonId::reconfiguration_required &&
                      executor.execute_calls == execute_calls,
                  "rebuild-requested program must not re-enter the executor");
  }
  {
    FakeExecutor executor;
    open_cuda::ResidentAdapter adapter(81, executor);
    const auto program = Program(382);
    hal::PreparedProgramHandle handle;
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    if (!Require(adapter.Prepare(program, &handle, &readiness, &reason) ==
                     hal::StatusId::ready,
                 "required-mask direct prepare should succeed")) {
      return false;
    }
    auto request = Request(program, 1);
    request.disabled_stage_mask = 1;
    const auto rejected = adapter.Execute(handle, request);
    ok &= Require(rejected.status == hal::StatusId::failed &&
                      rejected.reason == hal::ReasonId::required_stage_failed &&
                      executor.execute_calls == 0,
                  "required disabled bits must fail closed before execution");
    request.disabled_stage_mask = 2;
    const auto malformed = adapter.Execute(handle, request);
    ok &= Require(malformed.status == hal::StatusId::blocked &&
                      malformed.reason == hal::ReasonId::malformed_contract &&
                      executor.execute_calls == 0,
                  "out-of-range disabled bits must fail closed");
    adapter.DestroyPrepared(&handle);
  }
  return ok;
}

bool TestHiddenWorkAndOutputMutationAreRejected() {
  bool ok = true;
  for (const int mode : {0, 1, 2, 3, 4}) {
    FakeExecutor executor;
    executor.allocate_during_execute = mode == 0;
    executor.hide_upload = mode == 1;
    executor.add_intermediate_transfer = mode == 2;
    executor.mutate_output = mode == 3;
    executor.explicit_provider_sync = mode == 4;
    executor.hide_sync = mode == 4;
    open_cuda::ResidentAdapter adapter(81, executor);
    hal::PreparedProgramSession session;
    const auto program = Program(500u + static_cast<std::uint64_t>(mode));
    if (!Configure(&session, &adapter, program))
      return false;
    const auto result = session.Execute(Request(program));
    const auto expected_reason = mode == 3
                                     ? hal::ReasonId::invalid_resource
                                     : hal::ReasonId::synchronization_failed;
    ok &= Require(result.status == hal::StatusId::failed &&
                      result.reason == expected_reason,
                  "hidden frame work/transfer or output mutation must fail");
  }
  return ok;
}

bool TestCpuTailTelemetryAndStrictFallback() {
  FakeExecutor executor;
  open_cuda::ResidentAdapter adapter(81, executor);
  auto program = Program();
  program.cpu_tails[0] = {
      .id = hal::CpuTailId::output_format_conversion,
      .after_stage_id = 1,
      .scratch_bytes = 2048,
      .required = true,
  };
  program.cpu_tail_count = 1;
  {
    hal::PreparedProgramHandle direct_handle;
    hal::ReadinessSnapshot direct_readiness;
    hal::ReasonId direct_reason = hal::ReasonId::none;
    bool direct_ok =
        adapter.Prepare(program, &direct_handle, &direct_readiness,
                        &direct_reason) == hal::StatusId::degraded &&
        direct_readiness.reason == hal::ReasonId::cpu_tail_required &&
        direct_reason == hal::ReasonId::cpu_tail_required;
    adapter.DestroyPrepared(&direct_handle);
    if (!Require(direct_ok,
                 "declared CPU tail must degrade readiness exactly")) {
      return false;
    }
  }
  hal::PreparedProgramSession session;
  if (!Configure(&session, &adapter, program, hal::StatusId::degraded))
    return false;
  auto result = session.Execute(Request(program));
  bool ok = Require(result.transfers.cpu_tail_boundaries == 1,
                    "declared CPU tail must be measured");
  hal::ReasonId reason = hal::ReasonId::none;
  ok &= Require(session.CommitOutput({.completion = result.completion,
                                      .output_write_succeeded = true},
                                     &reason) == hal::StatusId::ready,
                "CPU-tail frame should commit");

  auto fallback_program = Program(600);
  fallback_program.requested_backend = hal::BackendId::vulkan;
  fallback_program.strict_backend_selection = false;
  {
    hal::PreparedProgramHandle direct_handle;
    hal::ReadinessSnapshot direct_readiness;
    hal::ReasonId direct_reason = hal::ReasonId::none;
    bool direct_ok =
        adapter.Prepare(fallback_program, &direct_handle, &direct_readiness,
                        &direct_reason) == hal::StatusId::degraded &&
        direct_readiness.reason ==
            hal::ReasonId::semantic_provider_unavailable &&
        direct_reason == hal::ReasonId::semantic_provider_unavailable;
    adapter.DestroyPrepared(&direct_handle);
    if (!Require(direct_ok,
                 "non-strict fallback must use provider-unavailable reason")) {
      return false;
    }
  }
  hal::PreparedProgramSession fallback_session;
  ok &= Configure(&fallback_session, &adapter, fallback_program,
                  hal::StatusId::degraded);
  result = fallback_session.Execute(Request(fallback_program));
  ok &=
      Require(fallback_session.CommitOutput({.completion = result.completion,
                                             .output_write_succeeded = true},
                                            &reason) == hal::StatusId::ready &&
                  fallback_session.telemetry().fallback_frames == 1,
              "explicit non-strict fallback must be visible after commit");

  auto strict = fallback_program;
  strict.program_id = 601;
  strict.strict_backend_selection = true;
  hal::PreparedProgramSession strict_session;
  hal::ReadinessSnapshot readiness;
  ok &= Require(strict_session.Configure(strict, adapter, &readiness,
                                         &reason) == hal::StatusId::blocked &&
                    reason == hal::ReasonId::strict_backend_unavailable,
                "strict selection must not execute CUDA for another backend");
  return ok;
}

bool TestTemporalDiscontinuityDeviceLossAndShutdown() {
  bool ok = true;
  {
    FakeExecutor executor;
    open_cuda::ResidentAdapter adapter(81, executor);
    auto program = Program();
    program.temporal.temporal = true;
    program.temporal.requires_contiguous_capture_sequence = true;
    program.temporal.frame_drop_behavior =
        hal::FrameDropBehavior::reset_temporal_state;
    program.temporal.discontinuity_action =
        hal::TemporalDiscontinuityAction::reset_and_continue;
    program.temporal.history_frames = 2;
    hal::PreparedProgramSession session;
    if (!Configure(&session, &adapter, program))
      return false;
    auto first = session.Execute(Request(program, 1));
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(session.CommitOutput({.completion = first.completion,
                                        .output_write_succeeded = true},
                                       &reason) == hal::StatusId::ready,
                  "first temporal frame should commit");
    auto skipped = session.Execute(Request(program, 3));
    ok &= Require(skipped.status == hal::StatusId::executed &&
                      executor.reset_calls == 1 &&
                      session.telemetry().temporal_resets == 1,
                  "capture discontinuity must reset temporal state");
  }
  {
    FakeExecutor executor;
    executor.device_lost = true;
    open_cuda::ResidentAdapter adapter(81, executor);
    const auto program = Program();
    hal::PreparedProgramSession session;
    if (!Configure(&session, &adapter, program))
      return false;
    auto result = session.Execute(Request(program));
    ok &= Require(result.status == hal::StatusId::quarantined &&
                      result.reason == hal::ReasonId::device_lost,
                  "device loss must quarantine the prepared program");
    result = session.Execute(Request(program, 2));
    ok &= Require(result.status == hal::StatusId::quarantined,
                  "quarantined program must not re-enter executor");
    session.Shutdown();
    ok &= Require(executor.shutdown_calls == 1,
                  "shutdown must reach the injected executor");
    result = session.Execute(Request(program, 3));
    ok &= Require(result.status == hal::StatusId::stopped,
                  "execution after shutdown must stop");
  }
  return ok;
}

bool TestAdapterLatchesDeviceLossPerSlot() {
  FakeExecutor executor;
  executor.device_lost = true;
  open_cuda::ResidentAdapter adapter(81, executor);
  const auto program = Program(650);
  hal::PreparedProgramHandle handle;
  hal::ReadinessSnapshot readiness;
  hal::ReasonId reason = hal::ReasonId::none;
  if (!Require(adapter.Prepare(program, &handle, &readiness, &reason) ==
                   hal::StatusId::ready,
               "device-loss direct prepare should succeed")) {
    return false;
  }
  auto result = adapter.Execute(handle, Request(program));
  bool ok = Require(result.status == hal::StatusId::quarantined &&
                        result.reason == hal::ReasonId::device_lost,
                    "adapter must quarantine device loss");
  executor.device_lost = false;
  result = adapter.Execute(handle, Request(program, 2));
  ok &= Require(result.status == hal::StatusId::quarantined &&
                    result.reason == hal::ReasonId::device_lost &&
                    executor.execute_calls == 1,
                "quarantined slot must prevent executor re-entry");
  adapter.DestroyPrepared(&handle);
  return ok;
}

bool TestReconfigurationWaitsForOutputCommit() {
  FakeExecutor executor;
  open_cuda::ResidentAdapter adapter(81, executor);
  hal::PreparedProgramSession session;
  const auto program = Program(700);
  if (!Configure(&session, &adapter, program))
    return false;
  const auto result = session.Execute(Request(program));
  if (!Require(result.status == hal::StatusId::executed,
               "frame before reconfiguration should execute")) {
    return false;
  }
  hal::ReadinessSnapshot readiness;
  hal::ReasonId reason = hal::ReasonId::none;
  auto next = Program(701);
  bool ok = Require(session.Configure(next, adapter, &readiness, &reason) ==
                            hal::StatusId::blocked &&
                        reason == hal::ReasonId::output_not_committed,
                    "reconfiguration must not replace an in-use output");
  ok &= Require(session.CommitOutput({.completion = result.completion,
                                      .output_write_succeeded = true},
                                     &reason) == hal::StatusId::ready,
                "pending output should remain committable");
  ok &= Configure(&session, &adapter, next);
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok &= TestCapabilityIsExactAndDoesNotPromoteCpuTensorProviders();
  ok &= TestPreparedExecutionAndCommittedAttribution();
  ok &= TestMalformedAndForeignProgramsFailClosed();
  ok &= TestStaleDomainGenerationAndResourceChecks();
  ok &= TestResourceExhaustionAndOldPlanRetention();
  ok &= TestCommonHostBindingAndOutputResolution();
  ok &= TestHostBindingRejectsMalformedForeignAndExhaustedTokens();
  ok &= TestOptionalFailurePreservesValidOutput();
  ok &= TestDisabledOptionalStageIsSkippedAndPreserved();
  ok &= TestRequiredFailureFailsClosedInsideAdapter();
  ok &= TestRequiredStagesQuarantineOrRequestRebuild();
  ok &= TestHiddenWorkAndOutputMutationAreRejected();
  ok &= TestCpuTailTelemetryAndStrictFallback();
  ok &= TestTemporalDiscontinuityDeviceLossAndShutdown();
  ok &= TestAdapterLatchesDeviceLossPerSlot();
  ok &= TestReconfigurationWaitsForOutputCommit();
  if (!ok)
    return 1;
  std::cout << "OPEN CUDA HAL ADAPTER TESTS OK\n";
  return 0;
}
