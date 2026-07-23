#include "core/hal/prepared_program.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>

namespace {

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

PreparedProgram Program(std::uint64_t program_id = 100,
                        std::uint64_t adapter = 77,
                        std::uint64_t config_generation = 1) {
  PreparedProgram program;
  program.program_id = program_id;
  program.owner_adapter_instance_id = adapter;
  program.execution_domain = Domain(adapter);
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
  FakeExecuteMode execute_mode = FakeExecuteMode::success;
  std::uint64_t next_token = 1;
  std::uint64_t prepare_calls = 0;
  std::uint64_t execute_calls = 0;
  std::uint64_t destroy_calls = 0;
  std::uint64_t reset_calls = 0;
  std::uint64_t shutdown_calls = 0;
  PreparedProgram prepared_program{};

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
    if (prepare_status != StatusId::ready) {
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
    readiness->status = StatusId::ready;
    readiness->reason = ReasonId::none;
    readiness->requested_backend = wrong_readiness_requested_backend
                                       ? BackendId::vulkan
                                       : program.requested_backend;
    readiness->selected_backend = program.execution_domain.backend;
    readiness->provider = program.stages[0].provider;
    readiness->execution_domain = program.execution_domain;
    readiness->ready_stage_mask = 0x3;
    readiness->available = true;
    return StatusId::ready;
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
    result.output_valid = true;
    result.stage_count = prepared_program.stage_count;
    result.frame_elapsed_ns = 3'000'000;
    result.transfers.uploads = 1;
    result.transfers.final_readbacks = 1;
    result.transfers.synchronizations = 1;
    for (std::size_t index = 0; index < result.stage_count; ++index) {
      result.stages[index].stage_id = prepared_program.stages[index].stage_id;
      result.stages[index].semantic = prepared_program.stages[index].semantic;
      result.stages[index].status = StageExecutionStatus::succeeded;
      result.stages[index].valid_output_preserved = true;
      result.stages[index].elapsed_ns = 1'000'000;
    }
    switch (execute_mode) {
    case FakeExecuteMode::success:
    case FakeExecuteMode::incomplete:
      break;
    case FakeExecuteMode::optional_failure:
      result.status = StatusId::degraded;
      result.reason = ReasonId::optional_stage_failed_open;
      result.stages[1].status = StageExecutionStatus::optional_failed_open;
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
                    session.telemetry().active_backend_frames == 0,
                "failed output write incorrectly committed attribution");

  const auto retry = session.Execute(Request(program, 2));
  ok &= Require(session.CommitOutput({retry.completion, true}, &reason) ==
                        StatusId::ready &&
                    session.telemetry().active_backend_frames == 1 &&
                    session.telemetry().optional_fail_open_stages == 2 &&
                    session.telemetry().committed_stage_successes[0] == 1 &&
                    session.telemetry().committed_stage_successes[1] == 0,
                "optional fail-open attribution changed");
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
                      ReasonId::synchronization_failed,
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
                    session.telemetry().setup_rollbacks == 1,
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
          session.Execute(Request(replacement, 1)).status == StatusId::stopped,
      "shutdown did not stop execution and release the plan");
  ok &= Require(session.Configure(replacement, adapter, &readiness, &reason) ==
                        StatusId::stopped &&
                    reason == ReasonId::shutdown_requested,
                "terminal session accepted configuration after shutdown");
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
  ok &= TestOptionalFailureAndOutputCommitAttribution();
  ok &= TestFailuresDeviceLossAndTransferEnforcement();
  ok &= TestRollbackReconfigurationTemporalAndShutdown();
  if (ok)
    std::cout << "StudioCast HAL contract tests passed\n";
  return ok ? 0 : 1;
}
