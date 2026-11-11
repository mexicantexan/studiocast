#include "core/hal/prepared_program.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>
#include <type_traits>

namespace {

static_assert(static_cast<std::uint32_t>(
                  studiocast::hal::ReasonId::semantic_provider_unavailable) ==
              26);
static_assert(static_cast<std::uint32_t>(
                  studiocast::hal::ReasonId::semantic_effect_unsupported) ==
              27);
static_assert(static_cast<std::uint32_t>(
                  studiocast::hal::ReasonId::capability_evidence_incomplete) ==
              28);
static_assert(static_cast<std::uint32_t>(
                  studiocast::hal::ReasonId::frame_order_violated) == 35);
static_assert(static_cast<std::uint32_t>(
                  studiocast::hal::CapabilityEvidenceCode::
                      per_effect_capability_evidence_missing) == 46);
static_assert(
    static_cast<std::uint32_t>(
        studiocast::hal::SemanticEffectProviderId::maxine_cuda_utility) == 6);
static_assert(static_cast<std::uint32_t>(
                  studiocast::hal::SemanticEffectProviderId::
                      open_cuda_production_kernels) == 7);
static_assert(std::is_standard_layout_v<studiocast::hal::ResourceHandle>);

std::atomic<std::size_t> g_allocations{0};
std::atomic<bool> g_count_allocations{false};

bool Require(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

using namespace studiocast::hal;

ExecutionDomainId Domain(std::uint64_t adapter = 77,
                         std::uint64_t context_generation = 5) {
  ExecutionDomainId domain;
  domain.backend = BackendId::cuda;
  domain.api_family = ApiFamily::cuda_driver;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = 0x2684;
  domain.physical_device.stable_id[0] = 0xa5;
  domain.logical_context_generation = context_generation;
  domain.allocator_family = AllocatorFamily::cuda_driver;
  domain.memory_handle_family = MemoryHandleFamily::cuda_device_pointer;
  domain.queue_or_stream_identity = 0x1234;
  domain.ownership_model = OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      SynchronizationProtocol::cuda_stream_ordered;
  domain.adapter_instance_id = adapter;
  return domain;
}

ImageDescriptor RgbDescriptor() {
  ImageDescriptor descriptor;
  descriptor.width = 640;
  descriptor.height = 480;
  descriptor.plane_count = 1;
  descriptor.format = PixelFormat::rgb_u8;
  descriptor.channel_order = ChannelOrder::rgb;
  descriptor.color_space = ColorSpace::srgb;
  descriptor.color_range = ColorRange::full;
  descriptor.orientation = Orientation::top_left;
  descriptor.host_coherence = HostCoherence::not_host_visible;
  descriptor.mutability = Mutability::read_write;
  descriptor.base_alignment_bytes = 256;
  descriptor.planes[0].width = 640;
  descriptor.planes[0].height = 480;
  descriptor.planes[0].row_stride_bytes = 640 * 3;
  descriptor.planes[0].size_bytes = 640 * 480 * 3;
  descriptor.planes[0].alignment_bytes = 256;
  return descriptor;
}

ImageDescriptor HostRgbDescriptor(Mutability mutability) {
  ImageDescriptor descriptor = RgbDescriptor();
  descriptor.host_coherence = HostCoherence::coherent;
  descriptor.mutability = mutability;
  descriptor.base_alignment_bytes = 256;
  descriptor.planes[0].alignment_bytes = 256;
  return descriptor;
}

PreparedProgram Program(std::uint64_t program_id = 100,
                        std::uint64_t adapter = 77,
                        std::uint64_t config_generation = 1) {
  PreparedProgram program;
  program.program_id = program_id;
  program.owner_adapter_instance_id = adapter;
  program.execution_domain = Domain(adapter);
  program.host_input = HostRgbDescriptor(Mutability::read_only);
  program.host_output = HostRgbDescriptor(Mutability::write_only);
  program.invalidation_key.generations.configuration = config_generation;
  program.invalidation_key.generations.geometry = 1;
  program.invalidation_key.generations.model = 1;
  program.invalidation_key.generations.provider = 1;
  program.invalidation_key.generations.device = 1;
  program.invalidation_key.generations.context =
      program.execution_domain.logical_context_generation;
  program.invalidation_key.canonical_plan_hash = 11 + config_generation;
  program.invalidation_key.model_artifact_hash = 22;
  program.invalidation_key.execution_domain_hash = 33;
  program.temporal.temporal = true;
  program.temporal.requires_contiguous_capture_sequence = true;
  program.temporal.frame_drop_behavior =
      FrameDropBehavior::reset_temporal_state;
  program.temporal.discontinuity_action =
      TemporalDiscontinuityAction::reset_and_continue;
  program.temporal.history_frames = 2;
  program.budget.persistent_bytes = 4 * 1024 * 1024;
  program.budget.scratch_bytes = 2 * 1024 * 1024;
  program.budget.upload_staging_bytes = 1024 * 1024;
  program.budget.readback_staging_bytes = 1024 * 1024;
  program.budget.resource_count = 8;
  program.budget.in_flight_slots = 1;
  program.requested_backend = BackendId::cuda;
  program.stage_count = 2;

  auto &required = program.stages[0];
  required.stage_id = 1;
  required.semantic = SemanticEffectId::virtual_background_blur;
  required.provider = SemanticEffectProviderId::open_video_onnxruntime_cuda;
  required.backend = BackendId::cuda;
  required.execution_domain = program.execution_domain;
  required.input.kind = StageIoKind::image;
  required.input.image = RgbDescriptor();
  required.output.kind = StageIoKind::image;
  required.output.image = RgbDescriptor();
  required.required = true;
  required.failure_action = FailureAction::fail_closed;
  required.circuit_breaker.consecutive_failure_limit = 2;
  required.circuit_breaker.action = CircuitBreakerAction::quarantine_program;
  required.scratch_bytes = 512 * 1024;

  auto &optional = program.stages[1];
  optional.stage_id = 2;
  optional.semantic = SemanticEffectId::mirror;
  optional.provider = SemanticEffectProviderId::open_video_onnxruntime_cuda;
  optional.backend = BackendId::cuda;
  optional.execution_domain = program.execution_domain;
  optional.input.kind = StageIoKind::image;
  optional.input.image = RgbDescriptor();
  optional.output.kind = StageIoKind::image;
  optional.output.image = RgbDescriptor();
  optional.required = false;
  optional.failure_action = FailureAction::fail_open_preserve_input;
  optional.circuit_breaker.consecutive_failure_limit = 1;
  optional.circuit_breaker.action = CircuitBreakerAction::disable_stage;
  optional.scratch_bytes = 256 * 1024;
  return program;
}

ResourceHandle Resource(const PreparedProgram &program,
                        std::uint64_t id = 800) {
  ResourceHandle resource;
  resource.resource_id = id;
  resource.allocator_owner_id = program.owner_adapter_instance_id;
  resource.lifetime_generation =
      program.execution_domain.logical_context_generation;
  resource.memory_domain = MemoryDomain::device_local;
  resource.lifetime = ResourceLifetime::frame_borrowed;
  resource.synchronization_state = SynchronizationState::ready;
  resource.execution_domain = program.execution_domain;
  resource.size_bytes = 640 * 480 * 3;
  resource.alignment_bytes = 256;
  resource.mutable_resource = true;
  return resource;
}

ExecuteRequest Request(const PreparedProgram &program, std::uint64_t sequence) {
  ExecuteRequest request;
  request.invalidation_key = program.invalidation_key;
  request.execution_domain = program.execution_domain;
  request.capture_sequence = sequence;
  request.input = Resource(program, 800 + sequence);
  return request;
}

enum class FakeExecuteMode {
  success,
  optional_failure,
  required_failure,
  device_loss,
  incomplete,
  hidden_crossing,
  foreign_output,
};

class FakeAdapter final : public IResidentAdapter {
public:
  std::uint64_t instance = 77;
  BackendId backend = BackendId::cuda;
  StatusId prepare_status = StatusId::ready;
  ReasonId prepare_reason = ReasonId::none;
  bool foreign_handle = false;
  bool exhaust_budget = false;
  bool wrong_readiness_requested_backend = false;
  bool corrupt_host_resolution = false;
  bool provide_evidence = false;
  bool invalid_evidence = false;
  FakeExecuteMode execute_mode = FakeExecuteMode::success;
  std::uint64_t next_token = 1;
  std::uint64_t prepare_calls = 0;
  std::uint64_t execute_calls = 0;
  std::uint64_t destroy_calls = 0;
  std::uint64_t reset_calls = 0;
  std::uint64_t shutdown_calls = 0;
  PreparedProgram prepared_program{};
  HostFrameBinding host_binding{};
  HostFrameTokens host_tokens{};

  std::uint64_t adapter_instance_id() const noexcept override {
    return instance;
  }
  BackendId backend_id() const noexcept override { return backend; }

  StatusId Prepare(const PreparedProgram &program,
                   PreparedProgramHandle *candidate,
                   ReadinessSnapshot *readiness,
                   ReasonId *reason) noexcept override {
    ++prepare_calls;
    if (reason)
      *reason = prepare_reason;
    if (prepare_status != StatusId::ready &&
        prepare_status != StatusId::degraded) {
      if (readiness) {
        readiness->status = prepare_status;
        readiness->reason = prepare_reason;
      }
      return prepare_status;
    }
    prepared_program = program;
    candidate->handle_id = 900 + prepare_calls;
    candidate->program_id = program.program_id;
    candidate->adapter_instance_id = foreign_handle ? instance + 1 : instance;
    candidate->invalidation_key = program.invalidation_key;
    candidate->execution_domain = program.execution_domain;
    candidate->reserved_budget = program.budget;
    if (exhaust_budget)
      ++candidate->reserved_budget.resource_count;
    readiness->status = prepare_status;
    readiness->reason = prepare_reason;
    readiness->requested_backend = wrong_readiness_requested_backend
                                       ? BackendId::vulkan
                                       : program.requested_backend;
    readiness->selected_backend = program.execution_domain.backend;
    readiness->provider = program.stages[0].provider;
    readiness->execution_domain = program.execution_domain;
    readiness->ready_stage_mask = 0x3;
    if (provide_evidence) {
      readiness->evidence_count = 1;
      readiness->evidence[0].code =
          CapabilityEvidenceCode::provider_runtime_available;
      readiness->evidence[0].status =
          invalid_evidence ? StatusId::blocked : StatusId::ready;
      readiness->evidence[0].reason =
          invalid_evidence ? ReasonId::none : ReasonId::none;
      readiness->evidence[0].backend = program.execution_domain.backend;
      readiness->evidence[0].satisfied = true;
    }
    readiness->available = true;
    readiness->degraded = prepare_status == StatusId::degraded;
    return prepare_status;
  }

  ExecuteResult Execute(const PreparedProgramHandle &,
                        const ExecuteRequest &request) noexcept override {
    ++execute_calls;
    ExecuteResult result;
    result.status = StatusId::executed;
    result.reason = ReasonId::none;
    result.completion.token_id = next_token++;
    result.completion.program_id = prepared_program.program_id;
    result.completion.adapter_instance_id = instance;
    result.completion.capture_sequence = request.capture_sequence;
    result.completion.complete = execute_mode != FakeExecuteMode::incomplete;
    result.output = request.input;
    result.output.resource_id += 1000;
    if (host_binding.capture_sequence == request.capture_sequence)
      result.output = host_tokens.output;
    result.output_valid = true;
    result.stage_count = prepared_program.stage_count;
    result.frame_elapsed_ns = 3'000'000;
    result.transfers.uploads = prepared_program.requires_upload ? 1 : 0;
    result.transfers.final_readbacks =
        prepared_program.requires_final_readback ? 1 : 0;
    result.transfers.cpu_tail_boundaries = prepared_program.cpu_tail_count;
    result.transfers.synchronizations = 1;
    for (std::size_t index = 0; index < result.stage_count; ++index) {
      result.stages[index].stage_id = prepared_program.stages[index].stage_id;
      result.stages[index].semantic = prepared_program.stages[index].semantic;
      result.stages[index].status = StageExecutionStatus::succeeded;
      result.stages[index].valid_output_preserved = true;
      result.stages[index].elapsed_ns = 1'000'000;
      const std::uint32_t bit = std::uint32_t{1} << index;
      if ((request.disabled_stage_mask & bit) != 0) {
        result.stages[index].status = StageExecutionStatus::disabled_preserved;
        result.stages[index].reason = ReasonId::circuit_breaker_open;
        result.stages[index].elapsed_ns = 0;
      }
    }
    switch (execute_mode) {
    case FakeExecuteMode::success:
    case FakeExecuteMode::incomplete:
      break;
    case FakeExecuteMode::optional_failure:
      if ((request.disabled_stage_mask & (std::uint32_t{1} << 1)) == 0) {
        result.status = StatusId::degraded;
        result.reason = ReasonId::optional_stage_failed_open;
        result.stages[1].status = StageExecutionStatus::optional_failed_open;
      }
      break;
    case FakeExecuteMode::required_failure:
      result.status = StatusId::failed;
      result.reason = ReasonId::required_stage_failed;
      result.stages[0].status = StageExecutionStatus::required_failed;
      result.output_valid = false;
      break;
    case FakeExecuteMode::device_loss:
      result.status = StatusId::failed;
      result.reason = ReasonId::device_lost;
      result.output_valid = false;
      break;
    case FakeExecuteMode::hidden_crossing:
      result.transfers.intermediate_readbacks = 1;
      break;
    case FakeExecuteMode::foreign_output:
      result.output.execution_domain.adapter_instance_id += 1;
      break;
    }
    return result;
  }

  StatusId ResetTemporalState(const PreparedProgramHandle &,
                              ReasonId) noexcept override {
    ++reset_calls;
    return reset_status;
  }
  StatusId reset_status = StatusId::ready;
  StatusId BindHostFrame(const PreparedProgramHandle &,
                         const HostFrameBinding &binding,
                         HostFrameTokens *tokens,
                         ReasonId *reason) noexcept override {
    host_binding = binding;
    tokens->input = Resource(prepared_program, 5001);
    tokens->output = Resource(prepared_program, 5002);
    host_tokens = *tokens;
    if (reason)
      *reason = ReasonId::none;
    return StatusId::ready;
  }
  StatusId ResolveHostFrameOutput(const PreparedProgramHandle &,
                                  const CompletionToken &,
                                  const ResourceHandle &,
                                  HostFrameOutputResolution *resolution,
                                  ReasonId *reason) noexcept override {
    resolution->descriptor = host_binding.output_descriptor;
    resolution->planes = host_binding.output_planes;
    resolution->plane_count = host_binding.output_plane_count;
    resolution->capture_sequence = host_binding.capture_sequence;
    resolution->output_ready = true;
    if (corrupt_host_resolution)
      ++resolution->planes[0].span_bytes;
    if (reason)
      *reason = ReasonId::none;
    return StatusId::ready;
  }
  void DestroyPrepared(PreparedProgramHandle *program) noexcept override {
    ++destroy_calls;
    *program = {};
  }
  void Shutdown() noexcept override { ++shutdown_calls; }
};

bool Configure(PreparedProgramSession *session, FakeAdapter *adapter,
               const PreparedProgram &program) {
  ReadinessSnapshot readiness;
  ReasonId reason = ReasonId::none;
  return Require(session->Configure(program, *adapter, &readiness, &reason) ==
                     StatusId::ready,
                 "valid fake program must configure") &&
         Require(readiness.available &&
                     readiness.execution_domain == program.execution_domain,
                 "readiness must retain exact execution domain");
}

bool TestIdsDescriptorsAndDomain() {
  bool ok = true;
  ok &= Require(kContractVersion == 1, "contract version changed");
  ok &= Require(
      static_cast<std::uint16_t>(CpuTailId::output_format_conversion) == 1 &&
          static_cast<std::uint16_t>(CpuTailId::auto_frame_tracking) == 2 &&
          static_cast<std::uint16_t>(CpuTailId::provider_postprocess) == 3 &&
          static_cast<std::uint16_t>(CpuTailId::writer_copy) == 4 &&
          static_cast<std::uint16_t>(CpuTailId::cpu_primitive_mirror) == 5 &&
          static_cast<std::uint16_t>(CpuTailId::backend_resolved_output) == 6,
      "stable CPU tail IDs changed");
  ok &= Require(StableEffectId(SemanticEffectId::virtual_background_blur) ==
                    "virtual_background.blur",
                "stable effect ID mapping changed");
  auto image = RgbDescriptor();
  ok &= Require(ValidateImageDescriptor(image),
                "valid image descriptor rejected");
  image.planes[0].alignment_bytes = 3;
  ok &= Require(!ValidateImageDescriptor(image),
                "non-power-of-two plane alignment accepted");

  TensorDescriptor tensor;
  tensor.element_type = TensorElementType::float32;
  tensor.layout = TensorLayout::nchw;
  tensor.rank = 4;
  tensor.base_alignment_bytes = 256;
  tensor.dimensions = {1, 3, 256, 256};
  tensor.strides_bytes = {3 * 256 * 256 * 4, 256 * 256 * 4, 256 * 4, 4};
  ok &= Require(ValidateTensorDescriptor(tensor),
                "valid tensor descriptor rejected");
  tensor.dimensions[2] = 0;
  ok &= Require(!ValidateTensorDescriptor(tensor),
                "zero tensor dimension accepted");

  auto domain = Domain();
  ok &= Require(ValidateExecutionDomain(domain),
                "complete execution domain rejected");
  domain.queue_or_stream_identity = 0;
  ok &= Require(!ValidateExecutionDomain(domain),
                "domain without queue/stream identity accepted");
  domain = Domain();
  domain.allocator_family = AllocatorFamily::vulkan;
  ok &= Require(!ValidateExecutionDomain(domain),
                "cross-family allocator evidence was accepted");
  domain = Domain();
  domain.memory_handle_family = MemoryHandleFamily::vulkan_buffer;
  ok &= Require(!ValidateExecutionDomain(domain),
                "cross-family memory handle evidence was accepted");
  domain = Domain();
  domain.synchronization_protocol = SynchronizationProtocol::vulkan_queue_fence;
  ok &= Require(!ValidateExecutionDomain(domain),
                "cross-family synchronization evidence was accepted");
  return ok;
}

bool TestMalformedCrossAdapterAndExhaustion() {
  bool ok = true;
  ReasonId reason = ReasonId::none;
  auto program = Program();
  ok &= Require(ValidatePreparedProgram(program, &reason),
                "valid prepared program rejected");

  auto wrong_version = program;
  ++wrong_version.contract_version;
  ok &= Require(!ValidatePreparedProgram(wrong_version, &reason) &&
                    reason == ReasonId::unsupported_contract_version,
                "unsupported contract version was accepted");

  auto cross_adapter = program;
  cross_adapter.stages[1].execution_domain.adapter_instance_id = 88;
  ok &= Require(!ValidatePreparedProgram(cross_adapter, &reason) &&
                    reason == ReasonId::cross_adapter_residency_forbidden,
                "cross-adapter resident island was not rejected");

  auto strict_mismatch = program;
  strict_mismatch.requested_backend = BackendId::vulkan;
  ok &= Require(!ValidatePreparedProgram(strict_mismatch, &reason) &&
                    reason == ReasonId::strict_backend_unavailable,
                "strict backend mismatch was not rejected");

  auto exhausted = program;
  exhausted.budget.scratch_bytes = 1;
  ok &= Require(!ValidatePreparedProgram(exhausted, &reason) &&
                    reason == ReasonId::resource_exhausted,
                "stage scratch exhaustion was not rejected");

  auto missing_host_boundary = program;
  missing_host_boundary.host_input = {};
  ok &= Require(!ValidatePreparedProgram(missing_host_boundary, &reason) &&
                    reason == ReasonId::malformed_descriptor,
                "missing host input boundary was accepted");

  auto resident_as_host = program;
  resident_as_host.host_input = program.stages[0].input.image;
  resident_as_host.host_input.mutability = Mutability::read_only;
  ok &= Require(!ValidatePreparedProgram(resident_as_host, &reason) &&
                    reason == ReasonId::malformed_descriptor,
                "device-resident descriptor was accepted as a host boundary");

  auto wrong_host_geometry = program;
  ++wrong_host_geometry.host_output.width;
  ++wrong_host_geometry.host_output.planes[0].width;
  ok &= Require(
      !ValidatePreparedProgram(wrong_host_geometry, &reason) &&
          reason == ReasonId::malformed_descriptor,
      "host boundary geometry unrelated to the resident edge was accepted");

  auto exhausted_host_boundary = program;
  exhausted_host_boundary.budget.upload_staging_bytes =
      exhausted_host_boundary.host_input.planes[0].size_bytes - 1;
  ok &= Require(!ValidatePreparedProgram(exhausted_host_boundary, &reason) &&
                    reason == ReasonId::resource_exhausted,
                "host boundary exceeded its transport budget");

  auto primitive = program;
  primitive.stages[1].semantic = SemanticEffectId::invalid;
  primitive.stages[1].primitive = PrimitiveOperationId::mirror;
  primitive.stages[1].provider = SemanticEffectProviderId::invalid;
  ok &= Require(ValidatePreparedProgram(primitive, &reason),
                "explicit primitive prepared stage was rejected");
  primitive.stages[1].semantic = SemanticEffectId::mirror;
  ok &= Require(!ValidatePreparedProgram(primitive, &reason),
                "stage carrying both semantic and primitive was accepted");
  primitive.stages[1].semantic = SemanticEffectId::invalid;
  primitive.stages[1].primitive = PrimitiveOperationId::invalid;
  ok &= Require(!ValidatePreparedProgram(primitive, &reason),
                "stage carrying neither semantic nor primitive was accepted");

  auto boundaries = program;
  boundaries.cpu_tail_count = 2;
  boundaries.cpu_tails[0].id = CpuTailId::provider_postprocess;
  boundaries.cpu_tails[0].placement = DeclaredCpuTail::Placement::after_stage;
  boundaries.cpu_tails[0].after_stage_id = 1;
  boundaries.cpu_tails[0].boundary_index = 1;
  boundaries.cpu_tails[0].scratch_bytes = 64;
  boundaries.cpu_tails[1].id = CpuTailId::output_format_conversion;
  boundaries.cpu_tails[1].placement = DeclaredCpuTail::Placement::before_stage;
  boundaries.cpu_tails[1].before_stage_id = 2;
  boundaries.cpu_tails[1].boundary_index = 1;
  boundaries.cpu_tails[1].scratch_bytes = 64;
  ok &= Require(ValidatePreparedProgram(boundaries, &reason),
                "exact before/after CPU boundaries were rejected");
  auto setup_only_tail = boundaries;
  setup_only_tail.cpu_tail_count = 1;
  setup_only_tail.cpu_tails[0].id = CpuTailId::backend_resolved_output;
  ok &= Require(!ValidatePreparedProgram(setup_only_tail, &reason) &&
                    reason == ReasonId::malformed_contract,
                "setup-only output-tail marker reached prepared execution");
  boundaries.cpu_tails[1].boundary_index = 2;
  ok &= Require(!ValidatePreparedProgram(boundaries, &reason),
                "wrong CPU boundary index was accepted");

  auto boundary_resource = Resource(program);
  boundary_resource.memory_domain = MemoryDomain::upload_staging;
  boundary_resource.size_bytes = program.budget.upload_staging_bytes + 1;
  ok &= Require(!ValidateResourceHandle(
                    boundary_resource, program.execution_domain,
                    program.execution_domain.logical_context_generation,
                    program.budget),
                "upload resource borrowed capacity from another budget class");

  FakeAdapter adapter;
  PreparedProgramSession session;
  adapter.foreign_handle = true;
  ReadinessSnapshot readiness;
  ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                        StatusId::blocked &&
                    reason == ReasonId::malformed_contract &&
                    adapter.destroy_calls == 1,
                "foreign prepared handle was not destroyed and rejected");

  adapter.foreign_handle = false;
  adapter.exhaust_budget = true;
  const auto exhausted_status =
      session.Configure(program, adapter, &readiness, &reason);
  ok &= Require(exhausted_status == StatusId::blocked &&
                    reason == ReasonId::resource_exhausted,
                "over-budget prepared handle was not rejected");

  adapter.exhaust_budget = false;
  adapter.wrong_readiness_requested_backend = true;
  ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                        StatusId::blocked &&
                    reason == ReasonId::malformed_contract,
                "readiness for a different requested backend was accepted");

  FakeAdapter evidence_adapter;
  evidence_adapter.provide_evidence = true;
  PreparedProgramSession evidence_session;
  ok &=
      Require(evidence_session.Configure(program, evidence_adapter, &readiness,
                                         &reason) == StatusId::ready &&
                  readiness.evidence_count == 1 &&
                  readiness.evidence[0].code ==
                      CapabilityEvidenceCode::provider_runtime_available,
              "typed readiness evidence was not preserved");
  evidence_session.Invalidate(ReasonId::reconfiguration_required);
  evidence_adapter.invalid_evidence = true;
  ok &=
      Require(evidence_session.Configure(program, evidence_adapter, &readiness,
                                         &reason) == StatusId::blocked &&
                  reason == ReasonId::malformed_contract,
              "internally inconsistent typed evidence was accepted");
  return ok;
}

bool TestExecutionValidationAndNoAllocation() {
  FakeAdapter adapter;
  PreparedProgramSession session;
  const auto program = Program();
  bool ok = Configure(&session, &adapter, program);

  auto stale = Request(program, 1);
  ++stale.invalidation_key.generations.configuration;
  ok &= Require(session.Execute(stale).reason ==
                    ReasonId::reconfiguration_required,
                "stale invalidation key was not rejected");

  auto foreign = Request(program, 1);
  foreign.execution_domain.adapter_instance_id = 99;
  ok &= Require(session.Execute(foreign).reason ==
                    ReasonId::wrong_execution_domain,
                "foreign request domain was not rejected");

  auto wrong_generation = Request(program, 1);
  ++wrong_generation.input.lifetime_generation;
  ok &= Require(session.Execute(wrong_generation).reason ==
                    ReasonId::wrong_generation,
                "wrong resource generation was not rejected");

  auto malformed = Request(program, 1);
  malformed.input.alignment_bytes = 3;
  ok &= Require(session.Execute(malformed).reason == ReasonId::invalid_resource,
                "malformed resource was not rejected");

  auto foreign_allocator = Request(program, 1);
  ++foreign_allocator.input.allocator_owner_id;
  ok &= Require(session.Execute(foreign_allocator).reason ==
                    ReasonId::invalid_resource,
                "foreign allocator resource was not rejected");

  g_allocations.store(0);
  g_count_allocations.store(true);
  const auto result = session.Execute(Request(program, 1));
  g_count_allocations.store(false);
  ok &= Require(g_allocations.load() == 0,
                "prepared execution allocated in the frame path");
  ok &= Require(result.status == StatusId::executed &&
                    result.transfers.uploads == 1 &&
                    result.transfers.final_readbacks == 1 &&
                    result.transfers.intermediate_readbacks == 0,
                "successful full-domain transfer contract changed");
  ok &= Require(!session.latest_committed_frame().valid,
                "execution claimed committed activity before output write");

  const auto blocked_pending = session.Execute(Request(program, 2));
  ok &= Require(blocked_pending.reason == ReasonId::output_not_committed,
                "reuse before output commit was not blocked");

  ReasonId reason = ReasonId::none;
  ReadinessSnapshot readiness;
  const auto replacement = Program(101, 77, 2);
  ok &= Require(session.Configure(replacement, adapter, &readiness, &reason) ==
                        StatusId::blocked &&
                    reason == ReasonId::output_not_committed &&
                    session.program().program_id == program.program_id,
                "reconfiguration replaced an in-use prepared program");
  ok &= Require(session.CommitOutput({result.completion, true}, &reason) ==
                    StatusId::ready,
                "successful writer commit failed");
  ok &= Require(session.telemetry().active_backend_frames == 1 &&
                    session.telemetry().committed_output_frames == 1 &&
                    session.telemetry().committed_stage_successes[0] == 1 &&
                    session.telemetry().transfers.uploads == 1 &&
                    session.telemetry().frame_latency.samples == 1 &&
                    session.telemetry().stage_latency[0].samples == 1,
                "active attribution or transfer telemetry was not committed");
  const auto &committed = session.latest_committed_frame();
  ok &= Require(committed.valid &&
                    committed.program_id == program.program_id &&
                    committed.invalidation_key == program.invalidation_key &&
                    committed.execution_domain == program.execution_domain &&
                    committed.capture_sequence == 1 &&
                    committed.output_commit_sequence == 1 &&
                    committed.stage_count == program.stage_count &&
                    committed.stages[0].stage_id ==
                        program.stages[0].stage_id &&
                    committed.stages[0].semantic ==
                        program.stages[0].semantic &&
                    committed.stages[0].backend ==
                        program.stages[0].backend &&
                    committed.stages[0].provider ==
                        program.stages[0].provider &&
                    committed.stages[0].status ==
                        StageExecutionStatus::succeeded,
                "latest committed-frame attribution is incomplete");
  return ok;
}

bool TestNeutralHostBindingAndTransportZeros() {
  alignas(256) static std::array<std::byte, 640 * 480 * 3> input_bytes{};
  alignas(256) static std::array<std::byte, 640 * 480 * 3> output_bytes{};
  FakeAdapter adapter;
  PreparedProgramSession session;
  auto program = Program();
  program.requires_upload = false;
  program.requires_final_readback = false;
  bool ok = Configure(&session, &adapter, program);

  HostFrameBinding binding;
  binding.input_descriptor = program.host_input;
  binding.output_descriptor = program.host_output;
  binding.input_plane_count = 1;
  binding.output_plane_count = 1;
  binding.capture_sequence = 1;
  binding.input_planes[0].read_data = input_bytes.data();
  binding.input_planes[0].span_bytes = input_bytes.size();
  binding.input_planes[0].row_stride_bytes = 640 * 3;
  binding.output_planes[0].write_data = output_bytes.data();
  binding.output_planes[0].span_bytes = output_bytes.size();
  binding.output_planes[0].row_stride_bytes = 640 * 3;

  HostFrameTokens tokens;
  ReasonId reason = ReasonId::none;
  auto malformed = binding;
  ++malformed.input_planes[0].row_stride_bytes;
  ok &= Require(session.BindHostFrame(malformed, &tokens, &reason) ==
                        StatusId::blocked &&
                    reason == ReasonId::host_frame_binding_invalid,
                "malformed host stride was accepted");
  auto resident_descriptor = binding;
  resident_descriptor.input_descriptor = program.stages[0].input.image;
  resident_descriptor.input_descriptor.mutability = Mutability::read_only;
  ok &=
      Require(session.BindHostFrame(resident_descriptor, &tokens, &reason) ==
                      StatusId::blocked &&
                  reason == ReasonId::host_frame_binding_invalid,
              "resident descriptor was accepted at the host binding boundary");
  ok &= Require(session.BindHostFrame(binding, &tokens, &reason) ==
                    StatusId::ready,
                "valid bounded host binding was rejected");
  auto request = Request(program, 1);
  request.input = tokens.input;
  const auto result = session.Execute(request);
  ok &= Require(result.status == StatusId::executed &&
                    result.transfers.uploads == 0 &&
                    result.transfers.final_readbacks == 0,
                "declared no-transport program did not enforce exact zeros");

  HostFrameOutputResolution resolution;
  ok &= Require(session.ResolveHostFrameOutput(result.completion, result.output,
                                               &resolution,
                                               &reason) == StatusId::ready &&
                    resolution.output_ready &&
                    resolution.planes[0].write_data == output_bytes.data(),
                "neutral host output did not resolve exact borrowed storage");
  ok &= Require(session.CommitOutput({result.completion, true}, &reason) ==
                    StatusId::ready,
                "host-bound output commit failed");

  binding.capture_sequence = 2;
  ok &= Require(session.BindHostFrame(binding, &tokens, &reason) ==
                    StatusId::ready,
                "second host binding failed");
  request = Request(program, 2);
  request.input = tokens.input;
  const auto corrupt = session.Execute(request);
  adapter.corrupt_host_resolution = true;
  ok &= Require(session.ResolveHostFrameOutput(corrupt.completion,
                                               corrupt.output, &resolution,
                                               &reason) == StatusId::blocked &&
                    reason == ReasonId::host_frame_binding_invalid,
                "mutated host output span was accepted");
  ok &= Require(session.CommitOutput({corrupt.completion, false}, &reason) ==
                    StatusId::failed,
                "failed corrupt output was not released");
  return ok;
}

bool TestOptionalFailureAndOutputCommitAttribution() {
  FakeAdapter adapter;
  PreparedProgramSession session;
  adapter.execute_mode = FakeExecuteMode::optional_failure;
  const auto program = Program();
  bool ok = Configure(&session, &adapter, program);
  const auto result = session.Execute(Request(program, 1));
  ok &= Require(result.status == StatusId::degraded &&
                    result.reason == ReasonId::optional_stage_failed_open &&
                    result.output_valid &&
                    result.stages[1].valid_output_preserved,
                "optional failure did not preserve valid output");
  ok &= Require(session.telemetry().active_backend_frames == 0,
                "backend became active before output write");

  ReasonId reason = ReasonId::none;
  ok &= Require(session.CommitOutput({result.completion, false}, &reason) ==
                        StatusId::failed &&
                    reason == ReasonId::output_write_failed &&
                    session.telemetry().active_backend_frames == 0 &&
                    !session.latest_committed_frame().valid,
                "failed output write incorrectly committed attribution");

  const auto retry = session.Execute(Request(program, 2));
  ok &= Require(session.CommitOutput({retry.completion, true}, &reason) ==
                        StatusId::ready &&
                    session.telemetry().active_backend_frames == 1 &&
                    session.telemetry().optional_fail_open_stages == 1 &&
                    session.telemetry().disabled_stage_frames == 1 &&
                    session.telemetry().committed_stage_successes[0] == 1 &&
                    session.telemetry().committed_stage_successes[1] == 0,
                "optional breaker disable or attribution changed");
  ok &= Require(session.latest_committed_frame().valid &&
                    session.latest_committed_frame().stage_count == 2 &&
                    session.latest_committed_frame().stages[0].status ==
                        StageExecutionStatus::succeeded &&
                    session.latest_committed_frame().stages[1].status ==
                        StageExecutionStatus::disabled_preserved,
                "committed snapshot hid optional-stage degradation");
  return ok;
}

bool TestExplicitFallbackReadinessAndCommittedAttribution() {
  bool ok = true;
  auto program = Program();
  program.requested_backend = BackendId::vulkan;
  program.strict_backend_selection = false;

  FakeAdapter adapter;
  adapter.prepare_status = StatusId::degraded;
  adapter.prepare_reason = ReasonId::semantic_provider_unavailable;
  PreparedProgramSession session;
  ReadinessSnapshot readiness;
  ReasonId reason = ReasonId::none;
  ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                    StatusId::degraded,
                "explicit fallback setup was not reported as degraded");
  ok &=
      Require(session.prepared() && readiness.degraded &&
                  readiness.requested_backend == BackendId::vulkan &&
                  readiness.selected_backend == BackendId::cuda &&
                  readiness.reason == ReasonId::semantic_provider_unavailable &&
                  session.telemetry().requested_backend == BackendId::vulkan &&
                  session.telemetry().selected_backend == BackendId::cuda,
              "fallback readiness lost requested/selected attribution");

  const auto result = session.Execute(Request(program, 1));
  ok &= Require(result.status == StatusId::executed,
                "degraded fallback setup did not execute");
  ok &= Require(session.telemetry().fallback_frames == 0,
                "fallback was attributed before output commit");
  ok &= Require(session.CommitOutput({result.completion, true}, &reason) ==
                        StatusId::ready &&
                    session.telemetry().fallback_frames == 1,
                "committed fallback frame was not attributed exactly once");
  return ok;
}

bool TestFailuresDeviceLossAndTransferEnforcement() {
  bool ok = true;
  {
    FakeAdapter adapter;
    PreparedProgramSession session;
    adapter.execute_mode = FakeExecuteMode::required_failure;
    const auto program = Program();
    ok &= Configure(&session, &adapter, program);
    ok &= Require(session.Execute(Request(program, 1)).reason ==
                          ReasonId::required_stage_failed &&
                      session.telemetry().active_backend_frames == 0 &&
                      session.telemetry().required_stage_failures == 1,
                  "required failure did not fail closed");
  }
  {
    FakeAdapter adapter;
    PreparedProgramSession session;
    adapter.execute_mode = FakeExecuteMode::device_loss;
    const auto program = Program();
    ok &= Configure(&session, &adapter, program);
    ok &= Require(session.Execute(Request(program, 1)).status ==
                          StatusId::quarantined &&
                      session.quarantined(),
                  "device loss did not quarantine the program");
    ok &= Require(session.Execute(Request(program, 2)).reason ==
                          ReasonId::device_lost &&
                      adapter.execute_calls == 1,
                  "quarantined program re-entered the adapter");
  }
  {
    FakeAdapter adapter;
    PreparedProgramSession session;
    adapter.execute_mode = FakeExecuteMode::hidden_crossing;
    const auto program = Program();
    ok &= Configure(&session, &adapter, program);
    ok &= Require(session.Execute(Request(program, 1)).reason ==
                      ReasonId::transfer_contract_violated,
                  "hidden intermediate crossing was accepted");
  }
  {
    FakeAdapter adapter;
    PreparedProgramSession session;
    adapter.execute_mode = FakeExecuteMode::foreign_output;
    const auto program = Program();
    ok &= Configure(&session, &adapter, program);
    ok &= Require(session.Execute(Request(program, 1)).reason ==
                      ReasonId::invalid_resource,
                  "foreign adapter output resource was accepted");
  }
  {
    FakeAdapter adapter;
    PreparedProgramSession session;
    adapter.execute_mode = FakeExecuteMode::incomplete;
    const auto program = Program();
    ok &= Configure(&session, &adapter, program);
    ok &= Require(session.Execute(Request(program, 1)).reason ==
                      ReasonId::incomplete_execution,
                  "non-complete v1 token was accepted");
  }
  return ok;
}

bool TestCircuitBreakerActionsAndRetryTransitions() {
  bool ok = true;
  ReasonId reason = ReasonId::none;
  {
    auto program = Program();
    program.stages[1].circuit_breaker.cooldown_frames = 2;
    FakeAdapter adapter;
    adapter.execute_mode = FakeExecuteMode::optional_failure;
    PreparedProgramSession session;
    ok &= Configure(&session, &adapter, program);

    const auto first = session.Execute(Request(program, 1));
    ok &= Require(first.status == StatusId::degraded &&
                      session.disabled_stage_mask() == 0x2 &&
                      session.readiness().reason ==
                          ReasonId::circuit_breaker_retry_pending,
                  "cooldown breaker did not open on threshold");
    ok &= Require(session.CommitOutput({first.completion, true}, &reason) ==
                      StatusId::ready,
                  "breaker-opening frame did not commit valid output");

    const auto cooldown_one = session.Execute(Request(program, 2));
    ok &= Require(cooldown_one.stages[1].status ==
                      StageExecutionStatus::disabled_preserved,
                  "disabled optional stage executed during cooldown");
    ok &= Require(session.CommitOutput({cooldown_one.completion, true},
                                       &reason) == StatusId::ready,
                  "first cooldown frame did not commit");
    const auto cooldown_two = session.Execute(Request(program, 3));
    ok &= Require(cooldown_two.stages[1].status ==
                      StageExecutionStatus::disabled_preserved,
                  "cooldown ended one frame early");
    ok &= Require(session.CommitOutput({cooldown_two.completion, true},
                                       &reason) == StatusId::ready,
                  "second cooldown frame did not commit");
    const auto retry = session.Execute(Request(program, 4));
    ok &= Require(retry.stages[1].status ==
                          StageExecutionStatus::optional_failed_open &&
                      session.telemetry().circuit_breaker_opens == 2 &&
                      session.telemetry().disabled_stage_frames == 2,
                  "cooldown retry transition did not execute exactly once");
  }
  {
    auto program = Program();
    program.stages[1].circuit_breaker.action =
        CircuitBreakerAction::quarantine_program;
    FakeAdapter adapter;
    adapter.execute_mode = FakeExecuteMode::optional_failure;
    PreparedProgramSession session;
    ok &= Configure(&session, &adapter, program);
    const auto result = session.Execute(Request(program, 1));
    ok &=
        Require(result.status == StatusId::degraded && session.quarantined() &&
                    session.readiness().status == StatusId::quarantined,
                "quarantine circuit action did not latch");
    ok &= Require(session.CommitOutput({result.completion, true}, &reason) ==
                      StatusId::ready,
                  "quarantine action corrupted current valid output");
    ok &= Require(session.Execute(Request(program, 2)).status ==
                      StatusId::quarantined,
                  "quarantined program re-entered adapter");
  }
  {
    auto program = Program();
    program.stages[1].circuit_breaker.action =
        CircuitBreakerAction::request_rebuild;
    FakeAdapter adapter;
    adapter.execute_mode = FakeExecuteMode::optional_failure;
    PreparedProgramSession session;
    ok &= Configure(&session, &adapter, program);
    const auto result = session.Execute(Request(program, 1));
    ok &= Require(result.status == StatusId::degraded &&
                      session.rebuild_requested() &&
                      session.readiness().rebuild_requested &&
                      session.telemetry().rebuild_requests == 1,
                  "request-rebuild circuit action did not latch");
    ok &= Require(session.CommitOutput({result.completion, true}, &reason) ==
                      StatusId::ready,
                  "rebuild-request frame did not preserve current output");
    ok &= Require(session.Execute(Request(program, 2)).reason ==
                      ReasonId::reconfiguration_required,
                  "rebuild-requested program executed another frame");
  }
  {
    auto invalid = Program();
    invalid.stages[0].circuit_breaker.action =
        CircuitBreakerAction::disable_stage;
    ok &= Require(!ValidatePreparedProgram(invalid, &reason),
                  "required stage was allowed to silently disable");
  }
  {
    auto program = Program();
    program.stages[0].circuit_breaker.consecutive_failure_limit = 1;
    FakeAdapter adapter;
    adapter.execute_mode = FakeExecuteMode::required_failure;
    PreparedProgramSession session;
    ok &= Configure(&session, &adapter, program);
    const auto result = session.Execute(Request(program, 1));
    ok &= Require(result.status == StatusId::failed &&
                      result.reason == ReasonId::required_stage_failed &&
                      session.quarantined() &&
                      session.readiness().status == StatusId::quarantined &&
                      session.telemetry().required_stage_failures == 1,
                  "required failure did not enforce quarantine action");
  }
  return ok;
}

bool TestExplicitTemporalResetGuardsAndForwarding() {
  FakeAdapter adapter;
  PreparedProgramSession session;
  const auto program = Program(199, 77, 1);
  bool ok = Require(session.ResetTemporalState(ReasonId::temporal_discontinuity) ==
                        StatusId::stale,
                    "unprepared reset must reject stale state");
  ok &= Configure(&session, &adapter, program);
  auto frame = session.Execute(Request(program, 1));
  ok &= Require(session.ResetTemporalState(ReasonId::temporal_discontinuity) ==
                        StatusId::failed && adapter.reset_calls == 0,
                    "reset cannot discard uncommitted output");
  ReasonId reason = ReasonId::none;
  ok &= Require(session.CommitOutput({frame.completion, true}, &reason) == StatusId::ready,
                "initial commit failed");
  const auto committed = session.latest_committed_frame();
  adapter.reset_status = StatusId::blocked;
  ok &= Require(session.ResetTemporalState(ReasonId::temporal_discontinuity) ==
                        StatusId::blocked && session.telemetry().temporal_resets == 0,
                    "reset must return exact adapter failure");
  adapter.reset_status = StatusId::ready;
  ok &= Require(session.ResetTemporalState(ReasonId::temporal_discontinuity) ==
                        StatusId::ready && session.telemetry().temporal_resets == 1 &&
                        session.latest_committed_frame().valid == committed.valid,
                    "successful reset must forward and preserve writer attribution");
  ok &= Require(session.Execute(Request(program, 1)).reason == ReasonId::frame_order_violated,
                "explicit reset must preserve monotonic capture order");
  session.Invalidate(ReasonId::device_lost);
  const auto resets = adapter.reset_calls;
  ok &= Require(session.ResetTemporalState(ReasonId::temporal_discontinuity) ==
                        StatusId::quarantined && adapter.reset_calls == resets,
                    "quarantined reset must never reach adapter");
  session.Shutdown();
  ok &= Require(session.ResetTemporalState(ReasonId::temporal_discontinuity) == StatusId::stopped,
                "stopped reset must remain stopped");
  return ok;
}

bool TestRollbackReconfigurationTemporalAndShutdown() {
  FakeAdapter adapter;
  PreparedProgramSession session;
  const auto first = Program(100, 77, 1);
  bool ok = Configure(&session, &adapter, first);

  const auto initial = session.Execute(Request(first, 1));
  ReasonId reason = ReasonId::none;
  ok &= Require(session.CommitOutput({initial.completion, true}, &reason) ==
                    StatusId::ready,
                "initial output commit failed");

  adapter.prepare_status = StatusId::blocked;
  adapter.prepare_reason = ReasonId::resource_exhausted;
  const auto replacement = Program(200, 77, 2);
  ReadinessSnapshot readiness;
  ok &= Require(session.Configure(replacement, adapter, &readiness, &reason) ==
                        StatusId::blocked &&
                    session.prepared() && session.program().program_id == 100 &&
                    session.telemetry().setup_rollbacks == 1 &&
                    !session.latest_committed_frame().valid,
                "failed setup did not retain the old plan");

  const auto after_rollback = session.Execute(Request(first, 3));
  ok &= Require(adapter.reset_calls == 1 &&
                    session.telemetry().temporal_resets == 1 &&
                    after_rollback.status == StatusId::executed,
                "temporal discontinuity did not reset prepared history");
  ok &= Require(session.CommitOutput({after_rollback.completion, true},
                                     &reason) == StatusId::ready,
                "post-reset output commit failed");

  adapter.prepare_status = StatusId::ready;
  adapter.prepare_reason = ReasonId::none;
  ok &= Require(session.Configure(replacement, adapter, &readiness, &reason) ==
                        StatusId::ready &&
                    session.program().program_id == 200,
                "valid reconfiguration did not replace the old plan");
  ok &= Require(session.Execute(Request(first, 4)).reason ==
                    ReasonId::reconfiguration_required,
                "old invalidation key remained executable");

  session.Shutdown();
  ok &= Require(
      session.stopped() && !session.prepared() && adapter.shutdown_calls == 1 &&
          !session.latest_committed_frame().valid &&
          session.Execute(Request(replacement, 1)).status == StatusId::stopped,
      "shutdown did not stop execution and release the plan");
  ok &= Require(session.Configure(replacement, adapter, &readiness, &reason) ==
                        StatusId::stopped &&
                    reason == ReasonId::shutdown_requested,
                "terminal session accepted configuration after shutdown");
  return ok;
}

bool TestObservableFrameDropPolicies() {
  bool ok = true;
  ReasonId reason = ReasonId::none;
  {
    auto program = Program();
    program.temporal.frame_drop_behavior = FrameDropBehavior::preserve_order;
    FakeAdapter adapter;
    PreparedProgramSession session;
    ok &= Configure(&session, &adapter, program);
    const auto first = session.Execute(Request(program, 1));
    ok &= Require(session.CommitOutput({first.completion, true}, &reason) ==
                      StatusId::ready,
                  "preserve-order first frame failed");
    const auto gap = session.Execute(Request(program, 3));
    ok &= Require(gap.reason == ReasonId::frame_order_violated &&
                      adapter.execute_calls == 1 &&
                      session.telemetry().preserve_order_failures == 1 &&
                      session.telemetry().dropped_capture_frames == 1,
                  "preserve-order gap was not observable and fail-closed");
    const auto next = session.Execute(Request(program, 2));
    ok &= Require(next.status == StatusId::executed,
                  "preserve-order failure advanced accepted sequence");
  }
  {
    auto program = Program();
    program.temporal.temporal = false;
    program.temporal.requires_contiguous_capture_sequence = false;
    program.temporal.frame_drop_behavior = FrameDropBehavior::latest_frame_wins;
    FakeAdapter adapter;
    PreparedProgramSession session;
    ok &= Configure(&session, &adapter, program);
    const auto first = session.Execute(Request(program, 10));
    ok &= Require(session.CommitOutput({first.completion, true}, &reason) ==
                      StatusId::ready,
                  "latest-frame first output failed");
    const auto latest = session.Execute(Request(program, 14));
    ok &= Require(latest.status == StatusId::executed &&
                      latest.dropped_capture_frames == 3 &&
                      !latest.temporal_state_reset &&
                      session.telemetry().latest_frame_wins_events == 1 &&
                      session.telemetry().dropped_capture_frames == 3,
                  "latest-frame-wins gap was not observable");
  }
  {
    auto program = Program();
    program.temporal.frame_drop_behavior =
        FrameDropBehavior::reset_temporal_state;
    FakeAdapter adapter;
    PreparedProgramSession session;
    ok &= Configure(&session, &adapter, program);
    const auto first = session.Execute(Request(program, 1));
    ok &= Require(session.CommitOutput({first.completion, true}, &reason) ==
                      StatusId::ready,
                  "reset policy first output failed");
    const auto reset = session.Execute(Request(program, 4));
    ok &= Require(
        reset.status == StatusId::executed && reset.temporal_state_reset &&
            reset.dropped_capture_frames == 2 && adapter.reset_calls == 1 &&
            session.telemetry().temporal_resets == 1,
        "reset-on-discontinuity was not observable");
  }
  return ok;
}

bool TestStrictSafeOldPlanRetention() {
  bool ok = true;
  ReasonId reason = ReasonId::none;
  ReadinessSnapshot readiness;
  {
    FakeAdapter adapter;
    PreparedProgramSession session;
    const auto old = Program();
    ok &= Configure(&session, &adapter, old);
    auto invalid_backend = Program(200, 77, 2);
    invalid_backend.requested_backend = BackendId::vulkan;
    ok &= Require(session.Configure(invalid_backend, adapter, &readiness,
                                    &reason) == StatusId::blocked &&
                      !session.prepared() &&
                      session.telemetry().setup_retention_rejections == 1,
                  "strict backend change retained an incompatible old plan");
  }
  {
    FakeAdapter adapter;
    PreparedProgramSession session;
    const auto old = Program();
    ok &= Configure(&session, &adapter, old);
    auto invalid_provider = Program(201, 77, 2);
    invalid_provider.strict_provider_selection = true;
    invalid_provider.requested_provider = SemanticEffectProviderId::maxine_vfx;
    ok &= Require(session.Configure(invalid_provider, adapter, &readiness,
                                    &reason) == StatusId::blocked &&
                      !session.prepared(),
                  "strict provider change retained an incompatible old plan");
  }
  {
    FakeAdapter adapter;
    PreparedProgramSession session;
    const auto old = Program();
    ok &= Configure(&session, &adapter, old);
    auto replacement = Program(202, 77, 2);
    replacement.execution_domain = Domain(77, 6);
    replacement.invalidation_key.generations.context = 6;
    replacement.invalidation_key.execution_domain_hash = 44;
    for (std::size_t index = 0; index < replacement.stage_count; ++index)
      replacement.stages[index].execution_domain = replacement.execution_domain;
    replacement.strict_execution_domain = true;
    replacement.requested_execution_domain = replacement.execution_domain;
    adapter.prepare_status = StatusId::blocked;
    adapter.prepare_reason = ReasonId::resource_exhausted;
    ok &= Require(session.Configure(replacement, adapter, &readiness,
                                    &reason) == StatusId::blocked &&
                      !session.prepared() &&
                      session.telemetry().setup_retention_rejections == 1,
                  "failed strict domain setup retained the old domain");
  }
  {
    FakeAdapter adapter;
    PreparedProgramSession session;
    const auto old = Program();
    ok &= Configure(&session, &adapter, old);
    auto replacement = Program(203, 77, 2);
    replacement.setup_failure_policy = SetupFailurePolicy::stop_pipeline;
    adapter.prepare_status = StatusId::blocked;
    adapter.prepare_reason = ReasonId::resource_exhausted;
    ok &= Require(session.Configure(replacement, adapter, &readiness,
                                    &reason) == StatusId::blocked &&
                      !session.prepared(),
                  "stop-on-setup-failure retained the old plan");
  }
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

void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }

int main() {
  bool ok = true;
  ok &= TestIdsDescriptorsAndDomain();
  ok &= TestMalformedCrossAdapterAndExhaustion();
  ok &= TestExecutionValidationAndNoAllocation();
  ok &= TestNeutralHostBindingAndTransportZeros();
  ok &= TestOptionalFailureAndOutputCommitAttribution();
  ok &= TestExplicitFallbackReadinessAndCommittedAttribution();
  ok &= TestFailuresDeviceLossAndTransferEnforcement();
  ok &= TestCircuitBreakerActionsAndRetryTransitions();
  ok &= TestExplicitTemporalResetGuardsAndForwarding();
  ok &= TestRollbackReconfigurationTemporalAndShutdown();
  ok &= TestObservableFrameDropPolicies();
  ok &= TestStrictSafeOldPlanRetention();
  if (ok)
    std::cout << "StudioCast HAL contract tests passed\n";
  return ok ? 0 : 1;
}
