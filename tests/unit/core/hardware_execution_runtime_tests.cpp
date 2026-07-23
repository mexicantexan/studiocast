#include "core/video/hardware_execution_runtime.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>

namespace {

using namespace studiocast;

std::atomic<bool> g_count_allocations{false};
std::atomic<std::uint64_t> g_allocations{0};

bool Require(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

hal::ExecutionDomainId Domain(std::uint64_t adapter, std::uint8_t device) {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::cuda;
  domain.api_family = hal::ApiFamily::cuda_driver;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = device;
  domain.physical_device.stable_id[0] = device;
  domain.logical_context_generation = 10 + adapter;
  domain.allocator_family = hal::AllocatorFamily::cuda_driver;
  domain.memory_handle_family = hal::MemoryHandleFamily::cuda_device_pointer;
  domain.queue_or_stream_identity = 100 + adapter;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::cuda_stream_ordered;
  domain.adapter_instance_id = adapter;
  return domain;
}

hal::ImageDescriptor Image(bool alias = false) {
  hal::ImageDescriptor image;
  image.width = 4;
  image.height = 2;
  image.plane_count = 1;
  image.format = hal::PixelFormat::rgb_u8;
  image.channel_order = hal::ChannelOrder::rgb;
  image.color_space = hal::ColorSpace::srgb;
  image.color_range = hal::ColorRange::full;
  image.orientation = hal::Orientation::top_left;
  image.host_coherence = hal::HostCoherence::coherent;
  image.mutability = hal::Mutability::read_write;
  image.aliasing = alias ? hal::AliasingPermission::exact_alias_only
                         : hal::AliasingPermission::forbidden;
  image.in_place_permitted = alias;
  image.base_alignment_bytes = 8;
  image.planes[0].row_stride_bytes = 12;
  image.planes[0].size_bytes = 24;
  image.planes[0].width = 4;
  image.planes[0].height = 2;
  image.planes[0].alignment_bytes = 8;
  return image;
}

hal::CanonicalStageIntent
Intent(std::uint32_t id, hal::SemanticEffectId semantic, bool required) {
  hal::CanonicalStageIntent intent;
  intent.stage_id = id;
  intent.semantic = semantic;
  intent.required = required;
  intent.failure_action = required
                              ? hal::FailureAction::fail_closed
                              : hal::FailureAction::fail_open_preserve_input;
  return intent;
}

hal::ResolutionRequest Request(bool two_stages = true) {
  hal::ResolutionRequest request;
  request.backend.mode = hal::BackendRequestMode::explicit_backend;
  request.backend.backend = hal::BackendId::cuda;
  request.backend.strict = true;
  request.stages[0] =
      Intent(1, hal::SemanticEffectId::virtual_background_blur, true);
  request.stage_count = 1;
  if (two_stages) {
    request.stages[1] = Intent(2, hal::SemanticEffectId::mirror, false);
    request.stage_count = 2;
  }
  return request;
}

hal::PreparedProgram Program(std::uint64_t adapter = 41,
                             std::uint64_t program_id = 100,
                             std::uint64_t configuration = 1,
                             bool two_stages = true) {
  hal::PreparedProgram program;
  program.program_id = program_id;
  program.owner_adapter_instance_id = adapter;
  program.execution_domain =
      Domain(adapter, static_cast<std::uint8_t>(adapter));
  program.invalidation_key.generations.configuration = configuration;
  program.invalidation_key.generations.geometry = 1;
  program.invalidation_key.generations.model = 1;
  program.invalidation_key.generations.provider = 1;
  program.invalidation_key.generations.device = 1;
  program.invalidation_key.generations.context =
      program.execution_domain.logical_context_generation;
  program.invalidation_key.canonical_plan_hash = 1000 + configuration;
  program.invalidation_key.model_artifact_hash = 2000;
  program.invalidation_key.execution_domain_hash = 3000 + adapter;
  program.temporal.temporal = true;
  program.temporal.requires_contiguous_capture_sequence = true;
  program.temporal.frame_drop_behavior =
      hal::FrameDropBehavior::reset_temporal_state;
  program.temporal.discontinuity_action =
      hal::TemporalDiscontinuityAction::reset_and_continue;
  program.temporal.history_frames = 2;
  program.budget.persistent_bytes = 256;
  program.budget.scratch_bytes = 128;
  program.budget.upload_staging_bytes = 64;
  program.budget.readback_staging_bytes = 64;
  program.budget.resource_count = 4;
  program.budget.in_flight_slots = 1;
  program.requested_backend = hal::BackendId::cuda;
  program.strict_backend_selection = true;
  program.requires_upload = true;
  program.requires_final_readback = true;
  program.stage_count = two_stages ? 2 : 1;

  for (std::size_t index = 0; index < program.stage_count; ++index) {
    auto &stage = program.stages[index];
    stage.stage_id = static_cast<std::uint32_t>(index + 1);
    stage.semantic = index == 0 ? hal::SemanticEffectId::virtual_background_blur
                                : hal::SemanticEffectId::mirror;
    stage.provider = hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
    stage.backend = hal::BackendId::cuda;
    stage.execution_domain = program.execution_domain;
    stage.input.kind = hal::StageIoKind::image;
    stage.input.image = Image();
    stage.output.kind = hal::StageIoKind::image;
    stage.output.image = Image();
    stage.required = index == 0;
    stage.failure_action = stage.required
                               ? hal::FailureAction::fail_closed
                               : hal::FailureAction::fail_open_preserve_input;
    stage.circuit_breaker.consecutive_failure_limit = 2;
    stage.circuit_breaker.action =
        stage.required ? hal::CircuitBreakerAction::quarantine_program
                       : hal::CircuitBreakerAction::disable_stage;
    stage.scratch_bytes = 16;
  }
  return program;
}

hal::ResourceHandle Resource(const hal::PreparedProgram &program,
                             std::uint64_t id,
                             hal::MemoryDomain memory_domain) {
  hal::ResourceHandle resource;
  resource.resource_id = id;
  resource.allocator_owner_id = program.owner_adapter_instance_id;
  resource.lifetime_generation =
      program.execution_domain.logical_context_generation;
  resource.memory_domain = memory_domain;
  resource.lifetime = hal::ResourceLifetime::frame_borrowed;
  resource.synchronization_state = hal::SynchronizationState::ready;
  resource.execution_domain = program.execution_domain;
  resource.size_bytes = 24;
  resource.alignment_bytes = 8;
  resource.mutable_resource = true;
  return resource;
}

enum class ExecuteMode {
  success,
  optional_fail_open,
  device_loss,
  stale_completion,
  foreign_output,
};

enum class BindingMode {
  valid,
  foreign,
  wrong_generation,
  wrong_domain,
  exhausted,
};

class FakeAdapter final : public hal::IResidentAdapter {
public:
  explicit FakeAdapter(std::uint64_t instance = 41) : instance_(instance) {}

  std::uint64_t adapter_instance_id() const noexcept override {
    return instance_;
  }
  hal::BackendId backend_id() const noexcept override {
    return hal::BackendId::cuda;
  }

  hal::StatusId Prepare(const hal::PreparedProgram &program,
                        hal::PreparedProgramHandle *candidate,
                        hal::ReadinessSnapshot *readiness,
                        hal::ReasonId *reason) noexcept override {
    ++prepare_calls;
    prepared_program = program;
    candidate->handle_id = 500 + prepare_calls;
    candidate->program_id = program.program_id;
    candidate->adapter_instance_id = instance_;
    candidate->invalidation_key = program.invalidation_key;
    candidate->execution_domain = program.execution_domain;
    candidate->reserved_budget = program.budget;
    if (exhaust_prepare)
      ++candidate->reserved_budget.resource_count;
    readiness->status = hal::StatusId::ready;
    readiness->reason = hal::ReasonId::none;
    readiness->requested_backend = program.requested_backend;
    readiness->selected_backend = program.execution_domain.backend;
    readiness->provider =
        hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
    readiness->execution_domain = program.execution_domain;
    readiness->ready_stage_mask = (std::uint32_t{1} << program.stage_count) - 1;
    readiness->available = true;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::StatusId BindHostFrame(const hal::PreparedProgramHandle &,
                              const hal::HostFrameBinding &binding,
                              hal::HostFrameTokens *tokens,
                              hal::ReasonId *reason) noexcept override {
    ++bind_calls;
    host_binding = binding;
    tokens->input = Resource(prepared_program, 7000 + binding.capture_sequence,
                             hal::MemoryDomain::upload_staging);
    tokens->output = Resource(prepared_program, 8000 + binding.capture_sequence,
                              hal::MemoryDomain::readback_staging);
    switch (binding_mode) {
    case BindingMode::valid:
      break;
    case BindingMode::foreign:
      ++tokens->input.allocator_owner_id;
      break;
    case BindingMode::wrong_generation:
      ++tokens->input.lifetime_generation;
      break;
    case BindingMode::wrong_domain:
      ++tokens->input.execution_domain.queue_or_stream_identity;
      break;
    case BindingMode::exhausted:
      tokens->input.size_bytes =
          prepared_program.budget.upload_staging_bytes + 1;
      break;
    }
    bound_tokens = *tokens;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::ExecuteResult
  Execute(const hal::PreparedProgramHandle &,
          const hal::ExecuteRequest &request) noexcept override {
    ++execute_calls;
    hal::ExecuteResult result;
    result.status = hal::StatusId::executed;
    result.completion.token_id = next_token++;
    result.completion.program_id = prepared_program.program_id;
    result.completion.adapter_instance_id = instance_;
    result.completion.capture_sequence = request.capture_sequence;
    result.completion.complete = true;
    result.output = bound_tokens.output;
    result.output_valid = true;
    result.stage_count = prepared_program.stage_count;
    result.transfers.uploads = 1;
    result.transfers.final_readbacks = 1;
    result.transfers.synchronizations = 1;
    for (std::size_t index = 0; index < result.stage_count; ++index) {
      result.stages[index].stage_id = prepared_program.stages[index].stage_id;
      result.stages[index].semantic = prepared_program.stages[index].semantic;
      result.stages[index].status = hal::StageExecutionStatus::succeeded;
      result.stages[index].valid_output_preserved = true;
      result.stages[index].elapsed_ns = 100;
      if ((request.disabled_stage_mask & (std::uint32_t{1} << index)) != 0) {
        result.stages[index].status =
            hal::StageExecutionStatus::disabled_preserved;
        result.stages[index].reason = hal::ReasonId::circuit_breaker_open;
        result.stages[index].elapsed_ns = 0;
      }
    }
    switch (execute_mode) {
    case ExecuteMode::success:
      break;
    case ExecuteMode::optional_fail_open:
      result.status = hal::StatusId::degraded;
      result.reason = hal::ReasonId::optional_stage_failed_open;
      result.stages[1].status = hal::StageExecutionStatus::optional_failed_open;
      break;
    case ExecuteMode::device_loss:
      result.status = hal::StatusId::failed;
      result.reason = hal::ReasonId::device_lost;
      result.output_valid = false;
      break;
    case ExecuteMode::stale_completion:
      ++result.completion.program_id;
      break;
    case ExecuteMode::foreign_output:
      ++result.output.execution_domain.adapter_instance_id;
      break;
    }
    return result;
  }

  hal::StatusId ResolveHostFrameOutput(
      const hal::PreparedProgramHandle &, const hal::CompletionToken &,
      const hal::ResourceHandle &, hal::HostFrameOutputResolution *resolution,
      hal::ReasonId *reason) noexcept override {
    ++resolve_calls;
    resolution->descriptor = host_binding.output_descriptor;
    resolution->planes = host_binding.output_planes;
    resolution->plane_count = host_binding.output_plane_count;
    resolution->capture_sequence = host_binding.capture_sequence;
    resolution->output_ready = true;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::StatusId ResetTemporalState(const hal::PreparedProgramHandle &,
                                   hal::ReasonId) noexcept override {
    ++reset_calls;
    return hal::StatusId::ready;
  }
  void DestroyPrepared(hal::PreparedProgramHandle *handle) noexcept override {
    ++destroy_calls;
    *handle = {};
  }
  void Shutdown() noexcept override { ++shutdown_calls; }

  BindingMode binding_mode = BindingMode::valid;
  ExecuteMode execute_mode = ExecuteMode::success;
  bool exhaust_prepare = false;
  std::uint64_t prepare_calls = 0;
  std::uint64_t bind_calls = 0;
  std::uint64_t execute_calls = 0;
  std::uint64_t resolve_calls = 0;
  std::uint64_t reset_calls = 0;
  std::uint64_t destroy_calls = 0;
  std::uint64_t shutdown_calls = 0;

private:
  std::uint64_t instance_;
  std::uint64_t next_token = 1;
  hal::PreparedProgram prepared_program{};
  hal::HostFrameBinding host_binding{};
  hal::HostFrameTokens bound_tokens{};
};

hal::SemanticCapabilityEvidence Capability(hal::SemanticEffectId semantic) {
  hal::SemanticCapabilityEvidence capability;
  capability.semantic = semantic;
  capability.provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  capability.status = hal::StatusId::ready;
  capability.live_stage = true;
  capability.production_eligible = true;
  capability.exact_evidence = true;
  return capability;
}

bool Register(hal::StaticAdapterRegistry *registry, FakeAdapter *adapter,
              bool blur, bool mirror, std::uint32_t priority = 100) {
  hal::AdapterRegistration registration;
  registration.adapter_instance_id = adapter->adapter_instance_id();
  registration.backend = hal::BackendId::cuda;
  registration.execution_domain =
      Domain(adapter->adapter_instance_id(),
             static_cast<std::uint8_t>(adapter->adapter_instance_id()));
  registration.selection_priority = priority;
  registration.resident_adapter = adapter;
  if (blur)
    registration
        .semantic_capabilities[registration.semantic_capability_count++] =
        Capability(hal::SemanticEffectId::virtual_background_blur);
  if (mirror)
    registration
        .semantic_capabilities[registration.semantic_capability_count++] =
        Capability(hal::SemanticEffectId::mirror);
  hal::ReasonId reason = hal::ReasonId::none;
  return Require(registry->Register(registration, &reason) ==
                     hal::StatusId::ready,
                 "fake adapter registration failed");
}

video::HardwareExecutionSetupRequest Setup(std::uint64_t program_id = 100,
                                           std::uint64_t configuration = 1,
                                           bool two_stages = true) {
  video::HardwareExecutionSetupRequest setup;
  setup.resolution = Request(two_stages);
  setup.prepared_program = Program(41, program_id, configuration, two_stages);
  return setup;
}

hal::HostFrameBinding Binding(std::uint64_t sequence,
                              std::array<std::byte, 24> *input,
                              std::array<std::byte, 24> *output,
                              bool passthrough = false) {
  hal::HostFrameBinding binding;
  binding.input_descriptor = Image(passthrough);
  binding.output_descriptor = Image(passthrough);
  binding.input_plane_count = 1;
  binding.output_plane_count = 1;
  binding.input_planes[0].read_data = input->data();
  binding.input_planes[0].span_bytes = input->size();
  binding.input_planes[0].row_stride_bytes = 12;
  binding.output_planes[0].write_data =
      passthrough ? input->data() : output->data();
  binding.output_planes[0].span_bytes = output->size();
  binding.output_planes[0].row_stride_bytes = 12;
  binding.capture_sequence = sequence;
  binding.ownership = hal::OwnershipModel::borrowed_until_completion;
  binding.lifetime = hal::ResourceLifetime::frame_borrowed;
  return binding;
}

bool TestPassthroughAndShadow() {
  FakeAdapter adapter;
  hal::StaticAdapterRegistry registry;
  bool ok = Register(&registry, &adapter, true, true);
  video::HardwareExecutionRuntime runtime(registry);

  video::HardwareExecutionSetupRequest passthrough;
  passthrough.resolution.backend.mode = hal::BackendRequestMode::automatic;
  passthrough.compare_legacy_resolution = true;
  passthrough.legacy_resolution.status = hal::StatusId::ready;
  hal::ReasonId reason = hal::ReasonId::none;
  ok &= Require(runtime.Configure(passthrough, &reason) ==
                    hal::StatusId::passthrough,
                "zero-stage setup was not passthrough");
  ok &= Require(adapter.prepare_calls == 0 && runtime.passthrough() &&
                    runtime.latest_setup().shadow_compared &&
                    !runtime.latest_setup().shadow_comparison.exact_match,
                "passthrough touched adapter or lost shadow discrepancy");

  alignas(8) std::array<std::byte, 24> frame{};
  alignas(8) std::array<std::byte, 24> unused{};
  const auto result =
      runtime.ExecuteHostFrame(Binding(1, &frame, &unused, true));
  ok &= Require(result.status == hal::StatusId::passthrough &&
                    result.output_ready && adapter.bind_calls == 0 &&
                    runtime.telemetry().transfers.uploads == 0,
                "passthrough touched hardware or failed output resolution");
  ok &=
      Require(!runtime.latest_committed_frame().valid &&
                  runtime.telemetry().active_backend == hal::BackendId::invalid,
              "passthrough attributed activity before writer commit");
  ok &=
      Require(runtime.CommitOutput(result.completion, true, &reason) ==
                      hal::StatusId::ready &&
                  runtime.latest_committed_frame().valid &&
                  runtime.latest_committed_frame().stage_count == 0 &&
                  runtime.telemetry().active_backend == hal::BackendId::invalid,
              "passthrough commit attribution was incorrect");

  auto shadow = Setup();
  shadow.mode = video::HardwareExecutionSetupMode::shadow_only;
  shadow.legacy_path_retained_for_setup_rollback = true;
  ok &= Require(
      runtime.Configure(shadow, &reason) == hal::StatusId::ready &&
          !runtime.prepared() && adapter.prepare_calls == 0 &&
          runtime.latest_setup().legacy_path_retained_for_setup_rollback,
      "shadow-only mode prepared or executed HAL");
  ok &= Require(runtime.ExecuteHostFrame(Binding(2, &frame, &unused)).reason ==
                    hal::ReasonId::stale_program,
                "shadow-only mode attempted frame fallback");
  return ok;
}

bool TestLiveExecutionCommitFailureAndOptionalFailOpen() {
  FakeAdapter adapter;
  hal::StaticAdapterRegistry registry;
  bool ok = Register(&registry, &adapter, true, true);
  video::HardwareExecutionRuntime runtime(registry);
  hal::ReasonId reason = hal::ReasonId::none;
  ok &= Require(runtime.Configure(Setup(), &reason) == hal::StatusId::ready &&
                    runtime.prepared(),
                "live facade failed setup");

  alignas(8) std::array<std::byte, 24> input{};
  alignas(8) std::array<std::byte, 24> output{};
  const auto first = runtime.ExecuteHostFrame(Binding(1, &input, &output));
  ok &=
      Require(first.status == hal::StatusId::executed && first.output_ready &&
                  runtime.telemetry().transfers.uploads == 1 &&
                  runtime.telemetry().transfers.final_readbacks == 1 &&
                  runtime.telemetry().transfers.intermediate_readbacks == 0 &&
                  runtime.telemetry().active_backend == hal::BackendId::invalid,
              "live frame violated boundary or early attribution");
  ok &=
      Require(runtime.CommitOutput(first.completion, false, &reason) ==
                      hal::StatusId::failed &&
                  reason == hal::ReasonId::output_write_failed &&
                  !runtime.latest_committed_frame().valid &&
                  runtime.telemetry().active_backend == hal::BackendId::invalid,
              "failed writer committed active backend");

  adapter.execute_mode = ExecuteMode::optional_fail_open;
  const auto second = runtime.ExecuteHostFrame(Binding(2, &input, &output));
  ok &=
      Require(second.status == hal::StatusId::degraded &&
                  second.reason == hal::ReasonId::optional_stage_failed_open &&
                  second.output_ready,
              "optional stage did not fail open");
  ok &= Require(runtime.CommitOutput(second.completion, true, &reason) ==
                        hal::StatusId::ready &&
                    runtime.latest_committed_frame().valid &&
                    runtime.latest_committed_frame().stage_count == 2 &&
                    runtime.latest_committed_frame().stages[1].status ==
                        hal::StageExecutionStatus::optional_failed_open &&
                    runtime.telemetry().active_backend == hal::BackendId::cuda,
                "successful writer did not commit exact activity");
  return ok;
}

bool TestStrictCrossAdapterAndExactProgramRejection() {
  bool ok = true;
  hal::ReasonId reason = hal::ReasonId::none;
  {
    FakeAdapter cuda;
    hal::StaticAdapterRegistry registry;
    ok &= Register(&registry, &cuda, true, true);
    video::HardwareExecutionRuntime runtime(registry);
    auto strict = Setup();
    strict.resolution.backend.backend = hal::BackendId::vulkan;
    strict.prepared_program.requested_backend = hal::BackendId::vulkan;
    ok &=
        Require(runtime.Configure(strict, &reason) == hal::StatusId::blocked &&
                    runtime.latest_setup().resolution.reason ==
                        hal::ReasonId::strict_backend_unavailable &&
                    cuda.prepare_calls == 0,
                "strict backend silently selected CUDA");
  }
  {
    FakeAdapter blur(41);
    FakeAdapter mirror(42);
    hal::StaticAdapterRegistry registry;
    ok &= Register(&registry, &blur, true, false, 100);
    ok &= Register(&registry, &mirror, false, true, 100);
    video::HardwareExecutionRuntime runtime(registry);
    auto mixed = Setup();
    ok &= Require(runtime.Configure(mixed, &reason) == hal::StatusId::blocked &&
                      runtime.latest_setup().domain_plan.reason ==
                          hal::ReasonId::cross_adapter_residency_forbidden &&
                      blur.prepare_calls == 0 && mirror.prepare_calls == 0,
                  "cross-adapter resident plan was accepted");
  }
  {
    FakeAdapter adapter;
    hal::StaticAdapterRegistry registry;
    ok &= Register(&registry, &adapter, true, true);
    video::HardwareExecutionRuntime runtime(registry);
    auto wrong = Setup();
    wrong.prepared_program.stages[1].semantic = hal::SemanticEffectId::vignette;
    ok &= Require(runtime.Configure(wrong, &reason) == hal::StatusId::blocked &&
                      reason == hal::ReasonId::malformed_contract &&
                      adapter.prepare_calls == 0,
                  "program not matching canonical resolution was prepared");
    wrong = Setup();
    wrong.prepared_program.execution_domain.queue_or_stream_identity += 1;
    for (std::size_t index = 0; index < wrong.prepared_program.stage_count;
         ++index) {
      wrong.prepared_program.stages[index].execution_domain =
          wrong.prepared_program.execution_domain;
    }
    wrong.prepared_program.invalidation_key.generations.context =
        wrong.prepared_program.execution_domain.logical_context_generation;
    ok &= Require(runtime.Configure(wrong, &reason) == hal::StatusId::blocked &&
                      reason == hal::ReasonId::wrong_execution_domain &&
                      adapter.prepare_calls == 0,
                  "wrong-domain prepared program was accepted");
  }
  return ok;
}

bool TestMalformedResourcesExhaustionAndStaleCompletion() {
  bool ok = true;
  const std::array<BindingMode, 4> modes = {
      BindingMode::foreign, BindingMode::wrong_generation,
      BindingMode::wrong_domain, BindingMode::exhausted};
  for (const auto mode : modes) {
    FakeAdapter adapter;
    adapter.binding_mode = mode;
    hal::StaticAdapterRegistry registry;
    ok &= Register(&registry, &adapter, true, true);
    video::HardwareExecutionRuntime runtime(registry);
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(runtime.Configure(Setup(), &reason) == hal::StatusId::ready,
                  "resource test setup failed");
    alignas(8) std::array<std::byte, 24> input{};
    alignas(8) std::array<std::byte, 24> output{};
    const auto result = runtime.ExecuteHostFrame(Binding(1, &input, &output));
    ok &= Require(result.status == hal::StatusId::blocked &&
                      result.reason == hal::ReasonId::invalid_resource &&
                      adapter.execute_calls == 0,
                  "malformed/foreign resource reached execution");
  }
  {
    FakeAdapter adapter;
    adapter.exhaust_prepare = true;
    hal::StaticAdapterRegistry registry;
    ok &= Register(&registry, &adapter, true, true);
    video::HardwareExecutionRuntime runtime(registry);
    hal::ReasonId reason = hal::ReasonId::none;
    ok &=
        Require(runtime.Configure(Setup(), &reason) == hal::StatusId::blocked &&
                    reason == hal::ReasonId::resource_exhausted,
                "prepared resource exhaustion was accepted");
  }
  {
    FakeAdapter adapter;
    adapter.execute_mode = ExecuteMode::stale_completion;
    hal::StaticAdapterRegistry registry;
    ok &= Register(&registry, &adapter, true, true);
    video::HardwareExecutionRuntime runtime(registry);
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(runtime.Configure(Setup(), &reason) == hal::StatusId::ready,
                  "stale completion setup failed");
    alignas(8) std::array<std::byte, 24> input{};
    alignas(8) std::array<std::byte, 24> output{};
    const auto result = runtime.ExecuteHostFrame(Binding(1, &input, &output));
    ok &= Require(result.status == hal::StatusId::failed &&
                      result.reason == hal::ReasonId::incomplete_execution &&
                      !runtime.frame_pending(),
                  "stale completion was accepted");
  }
  {
    FakeAdapter adapter;
    adapter.execute_mode = ExecuteMode::foreign_output;
    hal::StaticAdapterRegistry registry;
    ok &= Register(&registry, &adapter, true, true);
    video::HardwareExecutionRuntime runtime(registry);
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(runtime.Configure(Setup(), &reason) == hal::StatusId::ready,
                  "foreign output setup failed");
    alignas(8) std::array<std::byte, 24> input{};
    alignas(8) std::array<std::byte, 24> output{};
    const auto result = runtime.ExecuteHostFrame(Binding(1, &input, &output));
    ok &= Require(result.status == hal::StatusId::failed &&
                      result.reason == hal::ReasonId::invalid_resource &&
                      !runtime.frame_pending(),
                  "foreign output was accepted");
  }
  return ok;
}

bool TestReconfigureTemporalDeviceLossShutdownAndNoAllocation() {
  FakeAdapter adapter;
  hal::StaticAdapterRegistry registry;
  bool ok = Register(&registry, &adapter, true, true);
  video::HardwareExecutionRuntime runtime(registry);
  hal::ReasonId reason = hal::ReasonId::none;
  ok &= Require(runtime.Configure(Setup(), &reason) == hal::StatusId::ready,
                "lifecycle setup failed");
  alignas(8) std::array<std::byte, 24> input{};
  alignas(8) std::array<std::byte, 24> output{};

  auto frame = runtime.ExecuteHostFrame(Binding(1, &input, &output));
  const auto prepare_before = adapter.prepare_calls;
  ok &= Require(runtime.Configure(Setup(101, 2), &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::output_not_committed &&
                    adapter.prepare_calls == prepare_before,
                "reconfiguration replaced a pending frame");
  ok &= Require(runtime.CommitOutput(frame.completion, true, &reason) ==
                    hal::StatusId::ready,
                "pending frame did not commit");

  frame = runtime.ExecuteHostFrame(Binding(3, &input, &output));
  ok &= Require(frame.status == hal::StatusId::executed &&
                    adapter.reset_calls == 1 &&
                    runtime.telemetry().dropped_capture_frames == 1,
                "temporal frame drop was not reset and measured");
  ok &= Require(runtime.CommitOutput(frame.completion, true, &reason) ==
                    hal::StatusId::ready,
                "post-reset frame did not commit");

  g_allocations.store(0);
  g_count_allocations.store(true);
  for (std::uint64_t sequence = 4; sequence < 36; ++sequence) {
    const auto current =
        runtime.ExecuteHostFrame(Binding(sequence, &input, &output));
    if (current.status != hal::StatusId::executed ||
        runtime.CommitOutput(current.completion, true, &reason) !=
            hal::StatusId::ready) {
      ok = false;
      break;
    }
  }
  g_count_allocations.store(false);
  ok &= Require(g_allocations.load() == 0,
                "live facade allocated in the frame path");

  adapter.execute_mode = ExecuteMode::device_loss;
  const auto lost = runtime.ExecuteHostFrame(Binding(36, &input, &output));
  ok &= Require(
      lost.status == hal::StatusId::quarantined &&
          lost.reason == hal::ReasonId::device_lost &&
          runtime.readiness().status == hal::StatusId::quarantined &&
          runtime.ExecuteHostFrame(Binding(37, &input, &output)).status ==
              hal::StatusId::quarantined,
      "device loss did not quarantine the prepared program");
  runtime.Shutdown();
  ok &= Require(
      runtime.stopped() && adapter.shutdown_calls == 1 &&
          runtime.ExecuteHostFrame(Binding(38, &input, &output)).reason ==
              hal::ReasonId::shutdown_requested &&
          runtime.Configure(Setup(102, 3), &reason) == hal::StatusId::stopped,
      "shutdown allowed setup or execution");
  return ok;
}

} // namespace

void *operator new(std::size_t size) {
  if (g_count_allocations.load(std::memory_order_relaxed))
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *memory = std::malloc(size))
    return memory;
  throw std::bad_alloc();
}

void *operator new[](std::size_t size) {
  if (g_count_allocations.load(std::memory_order_relaxed))
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *memory = std::malloc(size))
    return memory;
  throw std::bad_alloc();
}

void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}

int main() {
  bool ok = true;
  ok &= TestPassthroughAndShadow();
  ok &= TestLiveExecutionCommitFailureAndOptionalFailOpen();
  ok &= TestStrictCrossAdapterAndExactProgramRejection();
  ok &= TestMalformedResourcesExhaustionAndStaleCompletion();
  ok &= TestReconfigureTemporalDeviceLossShutdownAndNoAllocation();
  if (!ok)
    return 1;
  std::cout << "hardware execution runtime tests passed\n";
  return 0;
}
