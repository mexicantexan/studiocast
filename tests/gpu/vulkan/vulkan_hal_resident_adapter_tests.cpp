#include "core/vulkan/hal_resident_adapter.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <string>

namespace {

std::atomic<std::uint64_t> g_allocations{0};

} // namespace

void *operator new(std::size_t size) {
  ++g_allocations;
  if (void *pointer = std::malloc(size))
    return pointer;
  throw std::bad_alloc();
}

void *operator new[](std::size_t size) {
  ++g_allocations;
  if (void *pointer = std::malloc(size))
    return pointer;
  throw std::bad_alloc();
}

void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete[](void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void *pointer, std::size_t) noexcept {
  std::free(pointer);
}

namespace {

using studiocast::hal::AliasingPermission;
using studiocast::hal::BackendId;
using studiocast::hal::ChannelOrder;
using studiocast::hal::ColorRange;
using studiocast::hal::ColorSpace;
using studiocast::hal::FailureAction;
using studiocast::hal::HostCoherence;
using studiocast::hal::ImageDescriptor;
using studiocast::hal::MemoryHandleFamily;
using studiocast::hal::Mutability;
using studiocast::hal::OutputCommit;
using studiocast::hal::OwnershipModel;
using studiocast::hal::PixelFormat;
using studiocast::hal::PreparedProgram;
using studiocast::hal::PreparedProgramHandle;
using studiocast::hal::PreparedProgramSession;
using studiocast::hal::ReadinessSnapshot;
using studiocast::hal::ReasonId;
using studiocast::hal::ResourceHandle;
using studiocast::hal::SemanticEffectId;
using studiocast::hal::SemanticEffectProviderId;
using studiocast::hal::StageExecutionStatus;
using studiocast::hal::StatusId;
using studiocast::hal::SynchronizationProtocol;
using studiocast::vulkan::IQualifiedVulkanResidentExecutor;
using studiocast::vulkan::VulkanExecutorPreparedProgram;
using studiocast::vulkan::VulkanExecutorResult;
using studiocast::vulkan::VulkanHalResidentAdapter;
using studiocast::vulkan::VulkanHostFrameBinding;
using studiocast::vulkan::VulkanResidentCapabilityEvidence;
using studiocast::vulkan::VulkanResidentProgramOptions;
using studiocast::vulkan::VulkanVignetteCenterMode;

constexpr std::uint64_t kAdapterId = 42;
constexpr std::size_t kFrameBytes = 12;

bool Require(bool condition, const std::string &message) {
  if (!condition)
    std::cerr << message << '\n';
  return condition;
}

ImageDescriptor FrameDescriptor(Mutability mutability) {
  ImageDescriptor descriptor;
  descriptor.width = 2;
  descriptor.height = 2;
  descriptor.plane_count = 1;
  descriptor.format = PixelFormat::rgb_u8;
  descriptor.channel_order = ChannelOrder::rgb;
  descriptor.color_space = ColorSpace::srgb;
  descriptor.color_range = ColorRange::full;
  descriptor.host_coherence = HostCoherence::not_host_visible;
  descriptor.mutability = mutability;
  descriptor.aliasing = AliasingPermission::forbidden;
  descriptor.base_alignment_bytes = 1;
  descriptor.planes[0].row_stride_bytes = 6;
  descriptor.planes[0].size_bytes = kFrameBytes;
  descriptor.planes[0].width = 2;
  descriptor.planes[0].height = 2;
  descriptor.planes[0].alignment_bytes = 1;
  return descriptor;
}

studiocast::hal::ExecutionDomainId Domain(
    std::uint64_t adapter_id = kAdapterId,
    std::uint64_t context_generation = 7) {
  studiocast::hal::ExecutionDomainId domain;
  domain.backend = BackendId::vulkan;
  domain.api_family = studiocast::hal::ApiFamily::vulkan;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = 0x1cb1;
  domain.physical_device.stable_id[0] = 0x56;
  domain.physical_device.stable_id[15] = 0xa9;
  domain.logical_context_generation = context_generation;
  domain.allocator_family = studiocast::hal::AllocatorFamily::vulkan;
  domain.memory_handle_family = MemoryHandleFamily::vulkan_image;
  domain.queue_or_stream_identity = 99;
  domain.ownership_model = OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      SynchronizationProtocol::vulkan_queue_fence;
  domain.adapter_instance_id = adapter_id;
  return domain;
}

PreparedProgram Program(std::uint64_t program_id = 100,
                        bool vignette = true, bool mirror = true,
                        bool required = false) {
  PreparedProgram program;
  program.program_id = program_id;
  program.owner_adapter_instance_id = kAdapterId;
  program.execution_domain = Domain();
  program.requested_backend = BackendId::vulkan;
  program.strict_backend_selection = true;
  program.invalidation_key.generations.configuration = program_id;
  program.invalidation_key.generations.geometry = 1;
  program.invalidation_key.generations.provider = 1;
  program.invalidation_key.generations.device = 1;
  program.invalidation_key.generations.context =
      program.execution_domain.logical_context_generation;
  program.invalidation_key.canonical_plan_hash = 0x1000 + program_id;
  program.invalidation_key.execution_domain_hash = 0x2000 + program_id;
  program.budget.persistent_bytes = 256;
  program.budget.scratch_bytes = 256;
  program.budget.upload_staging_bytes = 256;
  program.budget.readback_staging_bytes = 256;
  program.budget.resource_count = 8;
  program.budget.in_flight_slots = 1;
  program.requires_upload = true;
  program.requires_final_readback = true;

  const ImageDescriptor input = FrameDescriptor(Mutability::read_only);
  const ImageDescriptor output = FrameDescriptor(Mutability::read_write);
  if (vignette) {
    auto &stage = program.stages[program.stage_count++];
    stage.stage_id = 1;
    stage.semantic = SemanticEffectId::vignette;
    stage.provider =
        SemanticEffectProviderId::open_vulkan_production_kernels;
    stage.backend = BackendId::vulkan;
    stage.execution_domain = program.execution_domain;
    stage.input.image = input;
    stage.output.image = output;
    stage.required = required;
    stage.failure_action =
        required ? FailureAction::fail_closed
                 : FailureAction::fail_open_preserve_input;
  }
  if (mirror) {
    auto &stage = program.stages[program.stage_count++];
    stage.stage_id = vignette ? 2 : 1;
    stage.semantic = SemanticEffectId::mirror;
    stage.provider =
        SemanticEffectProviderId::open_vulkan_production_kernels;
    stage.backend = BackendId::vulkan;
    stage.execution_domain = program.execution_domain;
    stage.input.image = input;
    stage.output.image = output;
    stage.required = required;
    stage.failure_action =
        required ? FailureAction::fail_closed
                 : FailureAction::fail_open_preserve_input;
  }
  return program;
}

VulkanResidentProgramOptions Options(const PreparedProgram &program,
                                     bool vignette = true,
                                     bool mirror = true) {
  VulkanResidentProgramOptions options;
  options.program_id = program.program_id;
  options.canonical_plan_hash =
      program.invalidation_key.canonical_plan_hash;
  options.request_vignette = vignette;
  options.request_mirror = mirror;
  options.vignette_center = VulkanVignetteCenterMode::fixed_frame_center;
  options.vignette_intensity_percent = 50;
  return options;
}

class FakeQualifiedExecutor final
    : public IQualifiedVulkanResidentExecutor {
public:
  explicit FakeQualifiedExecutor(
      studiocast::hal::ExecutionDomainId domain = Domain()) noexcept {
    evidence.execution_domain = domain;
    evidence.build_enabled = true;
    evidence.runtime_loaded = true;
    evidence.non_cpu_physical_device = true;
    evidence.compute_queue_available = true;
    evidence.owned_context_healthy = true;
    evidence.utility_kernels_ready = true;
    evidence.exact_device_and_queue_shared = true;
    evidence.persistent_device_local_resources = true;
    evidence.bounded_reusable_allocations = true;
    evidence.no_host_intermediate_storage = true;
    evidence.one_upload_transport = true;
    evidence.one_final_readback_transport = true;
    evidence.mirror_production_ready = true;
    evidence.vignette_fixed_center_production_ready = true;
  }

  VulkanResidentCapabilityEvidence
  CapabilityEvidence() const noexcept override {
    ++capability_calls;
    return evidence;
  }

  StatusId Prepare(const PreparedProgram &,
                   const VulkanResidentProgramOptions &,
                   VulkanExecutorPreparedProgram *prepared,
                   ReasonId *reason) noexcept override {
    ++prepare_calls;
    if (reason)
      *reason = prepare_reason;
    if (fail_prepare)
      return StatusId::blocked;
    prepared->state_id = next_state_id++;
    prepared->reserved_budget.persistent_bytes = 64;
    prepared->reserved_budget.scratch_bytes = 64;
    prepared->reserved_budget.upload_staging_bytes = 64;
    prepared->reserved_budget.readback_staging_bytes = 64;
    prepared->reserved_budget.resource_count = 4;
    prepared->reserved_budget.in_flight_slots = 1;
    return StatusId::ready;
  }

  VulkanExecutorResult
  Execute(const VulkanExecutorPreparedProgram &, const PreparedProgram &program,
          const VulkanResidentProgramOptions &,
          const VulkanHostFrameBinding &binding,
          std::uint64_t) noexcept override {
    ++execute_calls;
    VulkanExecutorResult result;
    result.status = StatusId::executed;
    result.stage_count = program.stage_count;
    result.transfers.uploads = 1;
    result.transfers.final_readbacks = 1;
    result.transfers.synchronizations = 3;
    result.transfers.intermediate_readbacks =
        hidden_intermediate_crossing ? 1 : 0;
    result.frame_elapsed_ns = 1'000'000;
    for (std::size_t index = 0; index < program.stage_count; ++index) {
      result.stages[index].succeeded =
          fail_stage_index != static_cast<int>(index);
      result.stages[index].reason =
          result.stages[index].succeeded ? ReasonId::none : stage_failure_reason;
      result.stages[index].elapsed_ns = 100'000 + index;
    }
    if (report_device_loss) {
      result.status = StatusId::failed;
      result.reason = ReasonId::device_lost;
      result.device_lost = true;
      return result;
    }
    if (write_output) {
      std::memcpy(binding.output, binding.input,
                  std::min(binding.input_size_bytes,
                           binding.output_size_bytes));
      result.output_written = true;
    }
    return result;
  }

  void
  DestroyPrepared(VulkanExecutorPreparedProgram *prepared) noexcept override {
    ++destroy_calls;
    *prepared = {};
  }

  void Shutdown() noexcept override {
    ++shutdown_calls;
    shutdown = true;
  }

  mutable std::uint64_t capability_calls = 0;
  VulkanResidentCapabilityEvidence evidence{};
  bool fail_prepare = false;
  ReasonId prepare_reason = ReasonId::capability_evidence_incomplete;
  int fail_stage_index = -1;
  ReasonId stage_failure_reason = ReasonId::optional_stage_failed_open;
  bool hidden_intermediate_crossing = false;
  bool report_device_loss = false;
  bool write_output = true;
  bool shutdown = false;
  std::uint64_t prepare_calls = 0;
  std::uint64_t execute_calls = 0;
  std::uint64_t destroy_calls = 0;
  std::uint64_t shutdown_calls = 0;
  std::uint64_t next_state_id = 1;
};

VulkanHostFrameBinding Binding(std::array<std::uint8_t, kFrameBytes> &input,
                               std::array<std::uint8_t, kFrameBytes> &output) {
  VulkanHostFrameBinding binding;
  binding.input = input.data();
  binding.input_size_bytes = input.size();
  binding.input_row_stride_bytes = 6;
  binding.output = output.data();
  binding.output_size_bytes = output.size();
  binding.output_row_stride_bytes = 6;
  return binding;
}

studiocast::hal::ExecuteRequest Request(const PreparedProgram &program,
                                        const ResourceHandle &input,
                                        std::uint64_t sequence) {
  studiocast::hal::ExecuteRequest request;
  request.invalidation_key = program.invalidation_key;
  request.execution_domain = program.execution_domain;
  request.capture_sequence = sequence;
  request.input = input;
  return request;
}

bool Configure(PreparedProgramSession *session,
               VulkanHalResidentAdapter *adapter,
               const PreparedProgram &program,
               const VulkanResidentProgramOptions &options,
               ReadinessSnapshot *readiness = nullptr) {
  if (!adapter->SetProgramOptions(options))
    return false;
  ReasonId reason = ReasonId::none;
  ReadinessSnapshot local;
  const StatusId status =
      session->Configure(program, *adapter, readiness ? readiness : &local,
                         &reason);
  return status == StatusId::ready && reason == ReasonId::none;
}

bool TestPreparedExecutionAndCommittedAttribution() {
  FakeQualifiedExecutor executor;
  VulkanHalResidentAdapter adapter(kAdapterId, executor);
  PreparedProgramSession session;
  const PreparedProgram program = Program();
  ReadinessSnapshot readiness;
  bool ok = Require(Configure(&session, &adapter, program, Options(program),
                              &readiness),
                    "qualified mirror/vignette program should prepare");
  ok &= Require(readiness.available && readiness.status == StatusId::ready &&
                    readiness.ready_stage_mask == 3 &&
                    readiness.provider ==
                        SemanticEffectProviderId::
                            open_vulkan_production_kernels &&
                    executor.capability_calls == 1 &&
                    executor.prepare_calls == 1,
                "prepare must consume exact cached evidence exactly once");

  std::array<std::uint8_t, kFrameBytes> input{0, 1, 2, 3, 4, 5,
                                              6, 7, 8, 9, 10, 11};
  std::array<std::uint8_t, kFrameBytes> output{};
  ResourceHandle resource;
  ReasonId reason = ReasonId::none;
  ok &= Require(adapter.BindFrame(program.program_id, 1,
                                  Binding(input, output), &resource,
                                  &reason) == StatusId::ready,
                "frame binding should be accepted");
  const auto result = session.Execute(Request(program, resource, 1));
  ok &= Require(result.status == StatusId::executed && result.output_valid &&
                    result.completion.complete && result.stage_count == 2 &&
                    result.stages[0].status ==
                        StageExecutionStatus::succeeded &&
                    result.stages[1].status ==
                        StageExecutionStatus::succeeded &&
                    result.transfers.uploads == 1 &&
                    result.transfers.final_readbacks == 1 &&
                    result.transfers.intermediate_readbacks == 0 &&
                    result.transfers.intermediate_uploads == 0 &&
                    result.transfers.synchronizations == 3 &&
                    output == input,
                "execution should expose one upload, one readback, no hidden "
                "crossing, and exact stage/sync evidence");
  ok &= Require(
      session.telemetry().active_backend == BackendId::invalid &&
          session.telemetry().active_backend_frames == 0 &&
          session.telemetry().committed_output_frames == 0,
      "active Vulkan attribution must remain uncommitted before output write");
  ok &= Require(session.CommitOutput(OutputCommit{result.completion, true},
                                     &reason) == StatusId::ready &&
                    session.telemetry().active_backend == BackendId::vulkan &&
                    session.telemetry().active_backend_frames == 1 &&
                    session.telemetry().committed_output_frames == 1,
                "successful output write should commit Vulkan attribution");

  output.fill(0);
  ok &= Require(adapter.BindFrame(program.program_id, 2,
                                  Binding(input, output), &resource,
                                  &reason) == StatusId::ready,
                "second frame binding should be accepted");
  const auto failed_commit_frame =
      session.Execute(Request(program, resource, 2));
  ok &= Require(failed_commit_frame.status == StatusId::executed &&
                    session.CommitOutput(
                        {failed_commit_frame.completion, false},
                        &reason) == StatusId::failed &&
                    reason == ReasonId::output_write_failed &&
                    session.telemetry().active_backend_frames == 1 &&
                    session.telemetry().committed_output_frames == 1,
                "failed output write must not commit additional active "
                "backend attribution");
  return ok;
}

bool TestCapabilityAndSemanticGates() {
  bool ok = true;
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    PreparedProgram program = Program(200);
    auto options = Options(program);
    options.vignette_center = VulkanVignetteCenterMode::tracked_face;
    ok &= Require(adapter.SetProgramOptions(options),
                  "tracked-center options should bind for rejection");
    ReasonId reason = ReasonId::none;
    ReadinessSnapshot readiness;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          StatusId::blocked &&
                      reason == ReasonId::semantic_effect_unsupported &&
                      executor.capability_calls == 0 &&
                      executor.prepare_calls == 0,
                  "tracked-center vignette must fail before capability or "
                  "executor setup");
  }
  {
    FakeQualifiedExecutor executor;
    executor.evidence.vignette_fixed_center_production_ready = false;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    const PreparedProgram program = Program(201);
    ok &= Require(adapter.SetProgramOptions(Options(program)),
                  "capability-gate options should bind");
    ReasonId reason = ReasonId::none;
    ReadinessSnapshot readiness;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          StatusId::blocked &&
                      reason == ReasonId::capability_evidence_incomplete &&
                      executor.prepare_calls == 0,
                  "missing exact vignette readiness must fail closed");
  }
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    PreparedProgram program = Program(202);
    program.stages[0].semantic = SemanticEffectId::virtual_background_blur;
    ok &= Require(adapter.SetProgramOptions(Options(program)),
                  "unsupported-semantic options should bind");
    ReasonId reason = ReasonId::none;
    ReadinessSnapshot readiness;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          StatusId::blocked &&
                      reason == ReasonId::semantic_effect_unsupported &&
                      executor.prepare_calls == 0,
                  "Vulkan matting semantics must remain fail-closed");
  }
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    PreparedProgram program = Program(203);
    program.requested_backend = BackendId::cuda;
    program.strict_backend_selection = false;
    ok &= Require(adapter.SetProgramOptions(Options(program)),
                  "fallback options should bind");
    ReasonId reason = ReasonId::none;
    ReadinessSnapshot readiness;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          StatusId::degraded &&
                      readiness.degraded &&
                      readiness.reason ==
                          ReasonId::semantic_provider_unavailable &&
                      readiness.requested_backend == BackendId::cuda &&
                      readiness.selected_backend == BackendId::vulkan,
                  "explicit non-strict Vulkan fallback must remain degraded "
                  "with requested/selected attribution");
  }
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    PreparedProgram program = Program(204);
    std::swap(program.stages[0], program.stages[1]);
    program.stages[0].stage_id = 1;
    program.stages[1].stage_id = 2;
    ok &= Require(adapter.SetProgramOptions(Options(program)),
                  "ordering-rejection options should bind");
    ReasonId reason = ReasonId::none;
    ReadinessSnapshot readiness;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          StatusId::blocked &&
                      reason == ReasonId::malformed_contract &&
                      executor.prepare_calls == 0,
                  "mirror-before-vignette ordering must fail before executor "
                  "setup");
  }
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    PreparedProgram program = Program(205);
    program.stages[1].execution_domain.adapter_instance_id = kAdapterId + 1;
    ok &= Require(adapter.SetProgramOptions(Options(program)),
                  "cross-adapter options should bind");
    ReasonId reason = ReasonId::none;
    ReadinessSnapshot readiness;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          StatusId::blocked &&
                      reason ==
                          ReasonId::cross_adapter_residency_forbidden &&
                      executor.prepare_calls == 0,
                  "cross-adapter Vulkan residency must fail closed");
  }
  return ok;
}

bool TestOptionalAndRequiredFailure() {
  bool ok = true;
  {
    FakeQualifiedExecutor executor;
    executor.fail_stage_index = 0;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    const PreparedProgram program = Program(300);
    ok &= Require(Configure(&session, &adapter, program, Options(program)),
                  "optional-failure program should prepare");
    std::array<std::uint8_t, kFrameBytes> input{};
    std::array<std::uint8_t, kFrameBytes> output{};
    ResourceHandle resource;
    ReasonId reason = ReasonId::none;
    ok &= Require(adapter.BindFrame(program.program_id, 1,
                                    Binding(input, output), &resource,
                                    &reason) == StatusId::ready,
                  "optional-failure frame should bind");
    const auto result = session.Execute(Request(program, resource, 1));
    ok &= Require(
        result.status == StatusId::degraded && result.output_valid &&
            result.stages[0].status ==
                StageExecutionStatus::optional_failed_open &&
            result.stages[0].valid_output_preserved &&
            result.stages[1].status == StageExecutionStatus::succeeded &&
            session.telemetry().optional_fail_open_stages == 1,
        "optional vignette failure must preserve a valid resident frame");
    ok &= Require(session.CommitOutput({result.completion, true}, &reason) ==
                          StatusId::ready &&
                      session.telemetry().active_backend_frames == 1,
                  "successful mirror must still commit exact Vulkan "
                  "attribution after optional vignette failure");
  }
  {
    FakeQualifiedExecutor executor;
    executor.fail_stage_index = 0;
    executor.stage_failure_reason = ReasonId::required_stage_failed;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    const PreparedProgram program =
        Program(301, /*vignette=*/true, /*mirror=*/false, /*required=*/true);
    ok &= Require(Configure(&session, &adapter, program,
                            Options(program, true, false)),
                  "required-failure program should prepare");
    std::array<std::uint8_t, kFrameBytes> input{};
    std::array<std::uint8_t, kFrameBytes> output{};
    ResourceHandle resource;
    ReasonId reason = ReasonId::none;
    ok &= Require(adapter.BindFrame(program.program_id, 1,
                                    Binding(input, output), &resource,
                                    &reason) == StatusId::ready,
                  "required-failure frame should bind");
    const auto result = session.Execute(Request(program, resource, 1));
    ok &= Require(result.status == StatusId::failed &&
                      result.reason == ReasonId::required_stage_failed &&
                      result.stage_count == 1 &&
                      result.stages[0].status ==
                          StageExecutionStatus::required_failed &&
                      session.telemetry().required_stage_failures == 1,
                  "required Vulkan stage failure must fail closed with typed "
                  "telemetry");
  }
  return ok;
}

bool TestForeignStaleAndHiddenCrossingRejection() {
  FakeQualifiedExecutor executor;
  VulkanHalResidentAdapter adapter(kAdapterId, executor);
  PreparedProgramSession session;
  const PreparedProgram program = Program(400);
  bool ok = Require(Configure(&session, &adapter, program, Options(program)),
                    "validation program should prepare");
  std::array<std::uint8_t, kFrameBytes> input{};
  std::array<std::uint8_t, kFrameBytes> output{};
  ResourceHandle resource;
  ReasonId reason = ReasonId::none;
  ok &= Require(adapter.BindFrame(program.program_id, 1,
                                  Binding(input, output), &resource,
                                  &reason) == StatusId::ready,
                "validation frame should bind");

  auto wrong_generation = resource;
  --wrong_generation.lifetime_generation;
  ok &= Require(
      session.Execute(Request(program, wrong_generation, 1)).reason ==
          ReasonId::wrong_generation,
      "wrong context generation must be rejected before executor entry");
  auto forged = resource;
  ++forged.adapter_opaque[1];
  ok &= Require(session.Execute(Request(program, forged, 1)).reason ==
                        ReasonId::invalid_resource &&
                    executor.execute_calls == 0,
                "mutated adapter-private binding must be rejected");
  auto wrong_domain_request = Request(program, resource, 1);
  wrong_domain_request.execution_domain.queue_or_stream_identity++;
  ok &= Require(session.Execute(wrong_domain_request).reason ==
                        ReasonId::wrong_execution_domain &&
                    executor.execute_calls == 0,
                "wrong execution domain must be rejected");

  executor.hidden_intermediate_crossing = true;
  const auto hidden = session.Execute(Request(program, resource, 1));
  ok &= Require(hidden.status == StatusId::failed &&
                    hidden.reason == ReasonId::synchronization_failed &&
                    hidden.transfers.uploads == 1 &&
                    hidden.transfers.final_readbacks == 1 &&
                    hidden.transfers.intermediate_readbacks == 1 &&
                    session.telemetry().transfers.intermediate_readbacks == 1,
                "hidden intermediate crossing must fail while retaining exact "
                "transfer telemetry");
  return ok;
}

bool TestImageSequenceAndBoundaryBudgetValidation() {
  bool ok = true;
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    PreparedProgram program = Program(450);
    program.stages[0].input.image.format = PixelFormat::bgr_u8;
    program.stages[0].input.image.channel_order = ChannelOrder::bgr;
    ok &= Require(adapter.SetProgramOptions(Options(program)),
                  "BGR rejection options should bind");
    ReadinessSnapshot readiness;
    ReasonId reason = ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          StatusId::blocked &&
                      reason == ReasonId::malformed_descriptor &&
                      executor.prepare_calls == 0,
                  "Vulkan production adapter must reject non-RGB image "
                  "contracts even when generically valid");
  }
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    PreparedProgram program = Program(451);
    auto &second_input = program.stages[1].input.image;
    second_input.width = 3;
    second_input.planes[0].width = 3;
    second_input.planes[0].row_stride_bytes = 9;
    second_input.planes[0].size_bytes = 18;
    ok &= Require(adapter.SetProgramOptions(Options(program)),
                  "stage-layout rejection options should bind");
    ReadinessSnapshot readiness;
    ReasonId reason = ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          StatusId::blocked &&
                      reason == ReasonId::malformed_descriptor &&
                      executor.prepare_calls == 0,
                  "stage-to-stage geometry and stride mismatch must fail "
                  "before executor setup");
  }
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    PreparedProgram program = Program(452);
    program.budget.upload_staging_bytes = kFrameBytes - 1;
    ok &= Require(adapter.SetProgramOptions(Options(program)),
                  "upload-budget options should bind");
    ReadinessSnapshot readiness;
    ReasonId reason = ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          StatusId::resource_exhausted &&
                      reason == ReasonId::resource_exhausted &&
                      executor.prepare_calls == 0,
                  "input span must fit the upload-staging budget "
                  "independently");
  }
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    PreparedProgram program = Program(453);
    program.budget.readback_staging_bytes = kFrameBytes - 1;
    ok &= Require(adapter.SetProgramOptions(Options(program)),
                  "readback-budget options should bind");
    ReadinessSnapshot readiness;
    ReasonId reason = ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          StatusId::resource_exhausted &&
                      reason == ReasonId::resource_exhausted &&
                      executor.prepare_calls == 0,
                  "output span must fit the readback-staging budget "
                  "independently");
  }
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    const PreparedProgram program = Program(454);
    ok &= Require(Configure(&session, &adapter, program, Options(program)),
                  "sequence validation program should prepare");
    std::array<std::uint8_t, kFrameBytes> input{};
    std::array<std::uint8_t, kFrameBytes> output{};
    ResourceHandle resource;
    ReasonId reason = ReasonId::none;
    ok &= Require(adapter.BindFrame(program.program_id, 11,
                                    Binding(input, output), &resource,
                                    &reason) == StatusId::ready &&
                      resource.adapter_opaque[1] == 11,
                  "binding must encode its nonzero capture sequence");
    ok &= Require(session.Execute(Request(program, resource, 12)).status ==
                          StatusId::stale &&
                      executor.execute_calls == 0,
                  "binding/request capture-sequence mismatch must reject a "
                  "stale token before executor entry");
    const auto result = session.Execute(Request(program, resource, 11));
    ok &= Require(result.status == StatusId::executed &&
                      executor.execute_calls == 1,
                  "matching capture sequence should execute exactly once");
    ok &= Require(session.CommitOutput({result.completion, true}, &reason) ==
                          StatusId::ready,
                  "sequence-validation output should commit");
    ok &= Require(session.Execute(Request(program, resource, 11)).reason ==
                          ReasonId::invalid_resource &&
                      executor.execute_calls == 1,
                  "consumed frame binding must reject replay");
    ResourceHandle zero_sequence_resource;
    ok &= Require(adapter.BindFrame(program.program_id, 0,
                                    Binding(input, output),
                                    &zero_sequence_resource,
                                    &reason) == StatusId::blocked &&
                      reason == ReasonId::invalid_resource,
                  "zero capture sequence must fail binding");
  }
  return ok;
}

bool TestRollbackExhaustionDeviceLossAndShutdown() {
  bool ok = true;
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramHandle handles[studiocast::vulkan::
                                      kMaxVulkanPreparedPrograms]{};
    for (std::size_t index = 0;
         index < studiocast::vulkan::kMaxVulkanPreparedPrograms; ++index) {
      const PreparedProgram program = Program(500 + index);
      ok &= Require(adapter.SetProgramOptions(Options(program)),
                    "bounded-slot options should bind");
      ReadinessSnapshot readiness;
      ReasonId reason = ReasonId::none;
      ok &= Require(adapter.Prepare(program, &handles[index], &readiness,
                                    &reason) == StatusId::ready,
                    "bounded Vulkan slot should prepare");
    }
    const PreparedProgram exhausted = Program(599);
    ok &= Require(adapter.SetProgramOptions(Options(exhausted)),
                  "exhaustion options should bind");
    PreparedProgramHandle candidate;
    ReadinessSnapshot readiness;
    ReasonId reason = ReasonId::none;
    ok &= Require(adapter.Prepare(exhausted, &candidate, &readiness, &reason) ==
                          StatusId::resource_exhausted &&
                      reason == ReasonId::resource_exhausted &&
                      candidate.handle_id == 0,
                  "fifth prepared Vulkan program must fail with bounded "
                  "resource exhaustion");
    for (auto &handle : handles)
      adapter.DestroyPrepared(&handle);
  }
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    const PreparedProgram old_program = Program(600);
    ok &= Require(
        Configure(&session, &adapter, old_program, Options(old_program)),
        "old Vulkan program should prepare");
    const PreparedProgram replacement = Program(601);
    executor.fail_prepare = true;
    ok &= Require(adapter.SetProgramOptions(Options(replacement)),
                  "replacement options should bind");
    ReadinessSnapshot readiness;
    ReasonId reason = ReasonId::none;
    ok &= Require(session.Configure(replacement, adapter, &readiness,
                                    &reason) == StatusId::blocked &&
                      session.prepared() &&
                      session.program().program_id == old_program.program_id &&
                      session.telemetry().setup_rollbacks == 1,
                  "failed replacement setup must retain the old program");
    std::array<std::uint8_t, kFrameBytes> input{};
    std::array<std::uint8_t, kFrameBytes> output{};
    ResourceHandle resource;
    ok &= Require(
        adapter.BindFrame(old_program.program_id, 1, Binding(input, output),
                          &resource, &reason) == StatusId::ready,
        "retained old program frame should bind");
    const auto result =
        session.Execute(Request(old_program, resource, 1));
    ok &= Require(result.status == StatusId::executed,
                  "retained old program must remain executable");
    ok &= Require(session.CommitOutput({result.completion, true}, &reason) ==
                          StatusId::ready,
                  "retained old program output should commit");
  }
  {
    FakeQualifiedExecutor executor;
    executor.report_device_loss = true;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    const PreparedProgram program = Program(700);
    ok &= Require(Configure(&session, &adapter, program, Options(program)),
                  "device-loss program should prepare");
    std::array<std::uint8_t, kFrameBytes> input{};
    std::array<std::uint8_t, kFrameBytes> output{};
    ResourceHandle resource;
    ReasonId reason = ReasonId::none;
    ok &= Require(adapter.BindFrame(program.program_id, 1,
                                    Binding(input, output), &resource,
                                    &reason) == StatusId::ready,
                  "device-loss frame should bind");
    const auto lost = session.Execute(Request(program, resource, 1));
    ok &= Require(lost.status == StatusId::quarantined &&
                      lost.reason == ReasonId::device_lost &&
                      session.quarantined() &&
                      session.telemetry().device_losses == 1 &&
                      session.Execute(Request(program, resource, 2)).status ==
                          StatusId::quarantined,
                  "device loss must quarantine the prepared program");
    session.Shutdown();
    ok &= Require(session.Execute(Request(program, resource, 3)).status ==
                          StatusId::stopped &&
                      executor.shutdown,
                  "shutdown must stop future execution and release executor");
  }
  return ok;
}

bool TestReconfigurationTemporalCpuTailAndAllocationFreeLoop() {
  bool ok = true;
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    PreparedProgram temporal = Program(800);
    temporal.temporal.temporal = true;
    ok &= Require(adapter.SetProgramOptions(Options(temporal)),
                  "temporal options should bind for rejection");
    ReadinessSnapshot readiness;
    ReasonId reason = ReasonId::none;
    ok &= Require(session.Configure(temporal, adapter, &readiness, &reason) ==
                          StatusId::blocked &&
                      reason == ReasonId::semantic_effect_unsupported,
                  "non-temporal Vulkan final effects must reject temporal "
                  "history contracts");
  }
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    PreparedProgram program = Program(801);
    program.cpu_tail_count = 1;
    program.cpu_tails[0].id =
        studiocast::hal::CpuTailId::output_format_conversion;
    program.cpu_tails[0].after_stage_id =
        program.stages[program.stage_count - 1].stage_id;
    ok &= Require(adapter.SetProgramOptions(Options(program)),
                  "CPU-tail options should bind");
    ReadinessSnapshot readiness;
    ReasonId reason = ReasonId::none;
    ok &= Require(session.Configure(program, adapter, &readiness, &reason) ==
                          StatusId::blocked &&
                      reason == ReasonId::cpu_tail_required &&
                      !readiness.available && executor.prepare_calls == 0,
                  "production Vulkan mirror/vignette plans must reject every "
                  "CPU tail");
  }
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    const PreparedProgram program = Program(802);
    ok &= Require(Configure(&session, &adapter, program, Options(program)),
                  "reconfiguration program should prepare");
    std::array<std::uint8_t, kFrameBytes> input{};
    std::array<std::uint8_t, kFrameBytes> output{};
    ResourceHandle resource;
    ReasonId reason = ReasonId::none;
    ok &= Require(adapter.BindFrame(program.program_id, 1,
                                    Binding(input, output), &resource,
                                    &reason) == StatusId::ready,
                  "reconfiguration frame should bind");
    const auto result = session.Execute(Request(program, resource, 1));
    const PreparedProgram replacement = Program(803);
    ok &= Require(adapter.SetProgramOptions(Options(replacement)),
                  "pending replacement options should bind");
    ReadinessSnapshot readiness;
    ok &= Require(session.Configure(replacement, adapter, &readiness,
                                    &reason) == StatusId::blocked &&
                      reason == ReasonId::output_not_committed,
                  "reconfiguration must not replace a program with an "
                  "uncommitted completion");
    ok &= Require(session.CommitOutput({result.completion, true}, &reason) ==
                          StatusId::ready,
                  "pending output should commit before reconfiguration");
  }
  {
    FakeQualifiedExecutor executor;
    VulkanHalResidentAdapter adapter(kAdapterId, executor);
    PreparedProgramSession session;
    const PreparedProgram program = Program(804);
    ok &= Require(Configure(&session, &adapter, program, Options(program)),
                  "allocation-loop program should prepare");
    std::array<std::uint8_t, kFrameBytes> input{};
    std::array<std::uint8_t, kFrameBytes> output{};
    ReasonId reason = ReasonId::none;
    const std::uint64_t allocations_before = g_allocations.load();
    for (std::uint64_t frame = 1; frame <= 128; ++frame) {
      ResourceHandle resource;
      const StatusId bound =
          adapter.BindFrame(program.program_id, frame, Binding(input, output),
                            &resource, &reason);
      if (bound != StatusId::ready) {
        ok = false;
        break;
      }
      const auto result = session.Execute(Request(program, resource, frame));
      if (result.status != StatusId::executed ||
          session.CommitOutput({result.completion, true}, &reason) !=
              StatusId::ready) {
        ok = false;
        break;
      }
    }
    const std::uint64_t allocations_after = g_allocations.load();
    ok &= Require(allocations_after == allocations_before &&
                      executor.capability_calls == 1 &&
                      executor.prepare_calls == 1 &&
                      executor.execute_calls == 128,
                  "steady-state adapter execution must allocate and probe "
                  "zero times per frame");
  }
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok &= TestPreparedExecutionAndCommittedAttribution();
  ok &= TestCapabilityAndSemanticGates();
  ok &= TestOptionalAndRequiredFailure();
  ok &= TestForeignStaleAndHiddenCrossingRejection();
  ok &= TestImageSequenceAndBoundaryBudgetValidation();
  ok &= TestRollbackExhaustionDeviceLossAndShutdown();
  ok &= TestReconfigurationTemporalCpuTailAndAllocationFreeLoop();
  return ok ? 0 : 1;
}
