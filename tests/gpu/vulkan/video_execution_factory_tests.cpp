#include "core/video/video_execution_coordinator.h"
#include "core/vulkan/video_execution_factory.h"

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
std::atomic<bool> g_count_allocations{false};
std::atomic<std::uint64_t> g_allocations{0};
} // namespace

void *operator new(std::size_t size) {
  if (g_count_allocations.load(std::memory_order_relaxed))
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *pointer = std::malloc(size))
    return pointer;
  throw std::bad_alloc();
}
void *operator new[](std::size_t size) {
  if (g_count_allocations.load(std::memory_order_relaxed))
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *pointer = std::malloc(size))
    return pointer;
  throw std::bad_alloc();
}
void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete[](void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete[](void *pointer, std::size_t) noexcept {
  std::free(pointer);
}

namespace {

using namespace studiocast;

constexpr std::uint64_t kAdapterId = 77;
constexpr std::uint64_t kFactoryId = 91;
constexpr std::uint32_t kWidth = 4;
constexpr std::uint32_t kHeight = 2;
constexpr std::uint64_t kStride = kWidth * 3u;
constexpr std::uint64_t kBytes = kStride * kHeight;

bool Require(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

hal::ExecutionDomainId Domain() {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::vulkan;
  domain.api_family = hal::ApiFamily::vulkan;
  domain.physical_device.vendor_id = 0x1002;
  domain.physical_device.device_id = 0x73bf;
  domain.physical_device.stable_id[0] = 0xa5;
  domain.logical_context_generation = 19;
  domain.allocator_family = hal::AllocatorFamily::vulkan;
  domain.memory_handle_family = hal::MemoryHandleFamily::vulkan_buffer;
  domain.queue_or_stream_identity = 0x1234;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::vulkan_queue_fence;
  domain.adapter_instance_id = kAdapterId;
  return domain;
}

hal::ImageDescriptor Image(hal::HostCoherence coherence,
                           hal::Mutability mutability) {
  hal::ImageDescriptor image;
  image.width = kWidth;
  image.height = kHeight;
  image.plane_count = 1;
  image.format = hal::PixelFormat::rgb_u8;
  image.channel_order = hal::ChannelOrder::rgb;
  image.color_space = hal::ColorSpace::srgb;
  image.color_range = hal::ColorRange::full;
  image.alpha_meaning = hal::AlphaMeaning::none;
  image.orientation = hal::Orientation::top_left;
  image.host_coherence = coherence;
  image.mutability = mutability;
  image.aliasing = hal::AliasingPermission::forbidden;
  image.base_alignment_bytes = 1;
  image.planes[0].row_stride_bytes = kStride;
  image.planes[0].size_bytes = kBytes;
  image.planes[0].width = kWidth;
  image.planes[0].height = kHeight;
  image.planes[0].alignment_bytes = 1;
  return image;
}

hal::ResourceHandle Resource(const hal::PreparedProgram &program,
                             std::uint64_t id,
                             hal::MemoryDomain memory_domain) {
  hal::ResourceHandle resource;
  resource.resource_id = id;
  resource.allocator_owner_id = kAdapterId;
  resource.lifetime_generation =
      program.execution_domain.logical_context_generation;
  resource.memory_domain = memory_domain;
  resource.lifetime = hal::ResourceLifetime::frame_borrowed;
  resource.synchronization_state = hal::SynchronizationState::ready;
  resource.execution_domain = program.execution_domain;
  resource.size_bytes = kBytes;
  resource.alignment_bytes = 1;
  resource.mutable_resource = true;
  return resource;
}

enum class ExecuteMode {
  success,
  optional_fail_open,
  transfer_mismatch,
  device_loss,
};

class FakeVulkanAdapter final : public hal::IResidentAdapter {
public:
  std::uint64_t adapter_instance_id() const noexcept override {
    return kAdapterId;
  }
  hal::BackendId backend_id() const noexcept override {
    return hal::BackendId::vulkan;
  }

  hal::StatusId Prepare(const hal::PreparedProgram &program,
                        hal::PreparedProgramHandle *candidate,
                        hal::ReadinessSnapshot *readiness,
                        hal::ReasonId *reason) noexcept override {
    ++prepare_calls;
    program_ = program;
    candidate->handle_id = 100 + prepare_calls;
    candidate->program_id = program.program_id;
    candidate->adapter_instance_id = kAdapterId;
    candidate->invalidation_key = program.invalidation_key;
    candidate->execution_domain = program.execution_domain;
    candidate->reserved_budget = program.budget;
    if (fail_prepare) {
      if (reason)
        *reason = hal::ReasonId::resource_exhausted;
      return hal::StatusId::resource_exhausted;
    }
    readiness->status = hal::StatusId::ready;
    readiness->reason = hal::ReasonId::none;
    readiness->requested_backend = hal::BackendId::vulkan;
    readiness->selected_backend = hal::BackendId::vulkan;
    readiness->provider =
        hal::SemanticEffectProviderId::open_vulkan_production_kernels;
    readiness->execution_domain = program.execution_domain;
    readiness->ready_stage_mask =
        (std::uint32_t{1} << program.stage_count) - 1u;
    readiness->available = true;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::StatusId BindHostFrame(const hal::PreparedProgramHandle &,
                              const hal::HostFrameBinding &binding,
                              hal::HostFrameTokens *tokens,
                              hal::ReasonId *reason) noexcept override {
    binding_ = binding;
    tokens->input = Resource(program_, 1000 + binding.capture_sequence,
                             hal::MemoryDomain::upload_staging);
    tokens->output = Resource(program_, 2000 + binding.capture_sequence,
                              hal::MemoryDomain::readback_staging);
    tokens_ = *tokens;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::ExecuteResult
  Execute(const hal::PreparedProgramHandle &,
          const hal::ExecuteRequest &request) noexcept override {
    ++execute_calls;
    hal::ExecuteResult result;
    if (mode == ExecuteMode::device_loss) {
      result.status = hal::StatusId::quarantined;
      result.reason = hal::ReasonId::device_lost;
      return result;
    }
    std::memcpy(binding_.output_planes[0].write_data,
                binding_.input_planes[0].read_data, kBytes);
    result.status = mode == ExecuteMode::optional_fail_open
                        ? hal::StatusId::degraded
                        : hal::StatusId::executed;
    result.reason = mode == ExecuteMode::optional_fail_open
                        ? hal::ReasonId::optional_stage_failed_open
                        : hal::ReasonId::none;
    result.completion = {next_token_++, program_.program_id, kAdapterId,
                         request.capture_sequence, true};
    result.output = tokens_.output;
    result.output_valid = true;
    result.stage_count = program_.stage_count;
    for (std::size_t index = 0; index < result.stage_count; ++index) {
      auto &stage = result.stages[index];
      stage.stage_id = program_.stages[index].stage_id;
      stage.semantic = program_.stages[index].semantic;
      stage.status = hal::StageExecutionStatus::succeeded;
      stage.valid_output_preserved = true;
      stage.elapsed_ns = 100 + index;
    }
    if (mode == ExecuteMode::optional_fail_open) {
      result.stages[0].status = hal::StageExecutionStatus::optional_failed_open;
      result.stages[0].reason = hal::ReasonId::optional_stage_failed_open;
    }
    result.transfers.uploads = 1;
    result.transfers.final_readbacks = 1;
    result.transfers.synchronizations =
        mode == ExecuteMode::transfer_mismatch ? 2 : 1;
    result.frame_elapsed_ns = 500;
    return result;
  }

  hal::StatusId
  ResolveHostFrameOutput(const hal::PreparedProgramHandle &,
                         const hal::CompletionToken &completion,
                         const hal::ResourceHandle &output,
                         hal::HostFrameOutputResolution *resolution,
                         hal::ReasonId *reason) noexcept override {
    if (completion.program_id != program_.program_id ||
        output.resource_id != tokens_.output.resource_id) {
      if (reason)
        *reason = hal::ReasonId::stale_program;
      return hal::StatusId::stale;
    }
    resolution->descriptor = binding_.output_descriptor;
    resolution->planes = binding_.output_planes;
    resolution->plane_count = binding_.output_plane_count;
    resolution->capture_sequence = binding_.capture_sequence;
    resolution->output_ready = true;
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }

  hal::StatusId ResetTemporalState(const hal::PreparedProgramHandle &,
                                   hal::ReasonId) noexcept override {
    return hal::StatusId::ready;
  }

  void DestroyPrepared(hal::PreparedProgramHandle *program) noexcept override {
    ++destroy_calls;
    *program = {};
  }

  void Shutdown() noexcept override {
    ++shutdown_calls;
    shutdown = true;
  }

  ExecuteMode mode = ExecuteMode::success;
  bool fail_prepare = false;
  bool shutdown = false;
  std::uint64_t prepare_calls = 0;
  std::uint64_t execute_calls = 0;
  std::uint64_t destroy_calls = 0;
  std::uint64_t shutdown_calls = 0;

private:
  hal::PreparedProgram program_{};
  hal::HostFrameBinding binding_{};
  hal::HostFrameTokens tokens_{};
  std::uint64_t next_token_ = 1;
};

hal::SemanticCapabilityEvidence
Capability(hal::SemanticEffectId semantic,
           hal::CapabilityEvidenceCode production_code) {
  hal::SemanticCapabilityEvidence capability;
  capability.semantic = semantic;
  capability.provider =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  capability.status = hal::StatusId::ready;
  capability.live_stage = true;
  capability.production_eligible = true;
  capability.exact_evidence = true;
  constexpr hal::CapabilityEvidenceCode common[]{
      hal::CapabilityEvidenceCode::vulkan_build_enabled,
      hal::CapabilityEvidenceCode::vulkan_loader_available,
      hal::CapabilityEvidenceCode::vulkan_non_cpu_device,
      hal::CapabilityEvidenceCode::vulkan_compute_queue_available,
      hal::CapabilityEvidenceCode::vulkan_context_healthy,
      hal::CapabilityEvidenceCode::vulkan_utility_kernels_ready,
      hal::CapabilityEvidenceCode::vulkan_exact_device_shared,
      hal::CapabilityEvidenceCode::vulkan_device_local_residency,
      hal::CapabilityEvidenceCode::vulkan_bounded_allocations,
      hal::CapabilityEvidenceCode::vulkan_no_cpu_layers,
      hal::CapabilityEvidenceCode::vulkan_no_intermediate_readbacks,
      hal::CapabilityEvidenceCode::vulkan_live_stage_present,
  };
  for (const auto code : common) {
    capability.evidence[capability.evidence_count++] = {
        code,     hal::StatusId::ready,   hal::ReasonId::none,
        semantic, hal::BackendId::vulkan, true};
  }
  capability.evidence[capability.evidence_count++] = {
      production_code, hal::StatusId::ready,   hal::ReasonId::none,
      semantic,        hal::BackendId::vulkan, true};
  return capability;
}

hal::AdapterRegistration Registration(FakeVulkanAdapter *adapter) {
  hal::AdapterRegistration registration;
  registration.adapter_instance_id = kAdapterId;
  registration.backend = hal::BackendId::vulkan;
  registration.execution_domain = Domain();
  registration.selection_priority = 300;
  registration.resident_adapter = adapter;
  registration.semantic_capabilities[0] =
      Capability(hal::SemanticEffectId::vignette,
                 hal::CapabilityEvidenceCode::
                     open_vulkan_vignette_fixed_center_production_ready);
  registration.semantic_capabilities[1] = Capability(
      hal::SemanticEffectId::mirror,
      hal::CapabilityEvidenceCode::open_vulkan_mirror_production_ready);
  registration.semantic_capability_count = 2;
  return registration;
}

video::VideoExecutionPlanRequest Request() {
  video::VideoExecutionPlanRequest request;
  request.plan_id = 500;
  request.intent.backend.mode = hal::BackendRequestMode::explicit_backend;
  request.intent.backend.backend = hal::BackendId::vulkan;
  request.intent.backend.strict = true;
  request.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::vulkan_only;
  request.intent.requested_provider =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  request.intent.strict_provider_selection = true;

  auto &vignette = request.intent.stages[0];
  vignette.intent.stage_id = 8;
  vignette.intent.semantic = hal::SemanticEffectId::vignette;
  vignette.intent.failure_action = hal::FailureAction::fail_open_preserve_input;
  vignette.parameters[0].id = video::CanonicalVideoParameterId::intensity;
  vignette.parameters[0].kind =
      video::CanonicalVideoParameterKind::signed_integer;
  vignette.parameters[0].signed_value = 61;
  vignette.parameters[1].id =
      video::CanonicalVideoParameterId::center_on_tracked_face;
  vignette.parameters[1].kind = video::CanonicalVideoParameterKind::boolean;
  vignette.parameters[1].boolean_value = false;
  vignette.parameter_count = 2;

  auto &mirror = request.intent.stages[1];
  mirror.intent.stage_id = 9;
  mirror.intent.semantic = hal::SemanticEffectId::mirror;
  mirror.intent.failure_action = hal::FailureAction::fail_open_preserve_input;
  request.intent.stage_count = 2;

  request.host_input =
      Image(hal::HostCoherence::coherent, hal::Mutability::read_only);
  request.host_output =
      Image(hal::HostCoherence::coherent, hal::Mutability::write_only);
  request.invalidation_key.generations = {1, 1, 1, 1, 1, 19};
  request.invalidation_key.canonical_plan_hash = 1001;
  request.invalidation_key.execution_domain_hash = 3001;
  request.budget.persistent_bytes = 4096;
  request.budget.scratch_bytes = 4096;
  request.budget.upload_staging_bytes = 4096;
  request.budget.readback_staging_bytes = 4096;
  request.budget.resource_count = 16;
  request.budget.in_flight_slots = 1;
  return request;
}

hal::HostFrameBinding Binding(std::uint8_t *input, std::uint8_t *output,
                              std::uint64_t capture_sequence) {
  hal::HostFrameBinding binding;
  binding.input_descriptor =
      Image(hal::HostCoherence::coherent, hal::Mutability::read_only);
  binding.output_descriptor =
      Image(hal::HostCoherence::coherent, hal::Mutability::write_only);
  binding.input_planes[0].read_data =
      reinterpret_cast<const std::byte *>(input);
  binding.input_planes[0].span_bytes = kBytes;
  binding.input_planes[0].row_stride_bytes = kStride;
  binding.output_planes[0].write_data = reinterpret_cast<std::byte *>(output);
  binding.output_planes[0].span_bytes = kBytes;
  binding.output_planes[0].row_stride_bytes = kStride;
  binding.input_plane_count = 1;
  binding.output_plane_count = 1;
  binding.capture_sequence = capture_sequence;
  return binding;
}

struct Fixture {
  FakeVulkanAdapter adapter;
  vulkan::VulkanVideoExecutionFactory factory;
  hal::StaticAdapterRegistry registry;
  video::VideoExecutionCoordinator coordinator;

  Fixture()
      : factory(Registration(&adapter), {kFactoryId, 300}),
        coordinator(registry) {}

  bool Initialize() {
    hal::ReasonId reason = hal::ReasonId::none;
    return factory.Initialize(&reason) == hal::StatusId::ready &&
           registry.Register(factory.adapter_registration(), &reason) ==
               hal::StatusId::ready &&
           coordinator.RegisterFactory(factory.factory_registration(),
                                       &reason) == hal::StatusId::ready;
  }

  bool Prepare(const video::VideoExecutionPlanRequest &request,
               video::VideoExecutionPrepareEvidence *evidence = nullptr) {
    hal::ReasonId reason = hal::ReasonId::none;
    return coordinator.Prepare(request, evidence, &reason) ==
           hal::StatusId::ready;
  }
};

bool HasEvidence(const hal::ReadinessSnapshot &readiness,
                 hal::CapabilityEvidenceCode code) {
  for (std::size_t index = 0; index < readiness.evidence_count; ++index) {
    if (readiness.evidence[index].code == code &&
        readiness.evidence[index].satisfied) {
      return true;
    }
  }
  return false;
}

bool TestExactFactoryAndCommit() {
  Fixture fixture;
  video::VideoExecutionPrepareEvidence evidence;
  if (!Require(fixture.Initialize(), "factory fixture must initialize") ||
      !Require(fixture.Prepare(Request(), &evidence),
               "qualified Vulkan plan must prepare")) {
    return false;
  }
  const auto snapshot = fixture.coordinator.Snapshot();
  bool ok =
      Require(snapshot.plan.prepared_program.stage_count == 2 &&
                  snapshot.plan.prepared_program.stages[0].semantic ==
                      hal::SemanticEffectId::vignette &&
                  snapshot.plan.prepared_program.stages[1].semantic ==
                      hal::SemanticEffectId::mirror,
              "factory must preserve canonical vignette-before-mirror order");
  ok &= Require(
      HasEvidence(evidence.readiness,
                  hal::CapabilityEvidenceCode::
                      open_vulkan_vignette_fixed_center_production_ready) &&
          HasEvidence(
              evidence.readiness,
              hal::CapabilityEvidenceCode::open_vulkan_mirror_production_ready),
      "readiness must preserve both exact per-effect production attestations");

  std::uint8_t input[kBytes]{};
  std::uint8_t output[kBytes]{};
  for (std::size_t index = 0; index < kBytes; ++index)
    input[index] = static_cast<std::uint8_t>(index + 3);
  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations.store(true, std::memory_order_relaxed);
  const auto frame = fixture.coordinator.Execute(Binding(input, output, 1));
  g_count_allocations.store(false, std::memory_order_relaxed);
  ok &= Require(frame.status == hal::StatusId::executed &&
                    frame.transfers.uploads == 1 &&
                    frame.transfers.final_readbacks == 1 &&
                    frame.transfers.intermediate_readbacks == 0 &&
                    frame.transfers.intermediate_uploads == 0 &&
                    frame.transfers.synchronizations == 1 &&
                    std::memcmp(input, output, kBytes) == 0,
                "prepared frame must keep the exact transfer contract");
  ok &= Require(g_allocations.load(std::memory_order_relaxed) == 0,
                "Execute must not allocate");
  hal::ReasonId reason = hal::ReasonId::none;
  ok &= Require(fixture.coordinator.CommitOutput(
                    frame.completion, true, &reason) == hal::StatusId::ready,
                "successful writer commit must succeed");
  const auto committed = fixture.coordinator.Snapshot();
  ok &= Require(committed.latest_committed_frame.valid &&
                    committed.telemetry.active_backend ==
                        hal::BackendId::vulkan &&
                    committed.telemetry.committed_output_frames == 1,
                "active Vulkan attribution must occur only after commit");
  return ok;
}

bool TestUnsupportedAndForeignPreflight() {
  Fixture fixture;
  if (!fixture.Initialize() || !fixture.Prepare(Request()))
    return false;
  const auto frozen = fixture.coordinator.Snapshot();
  auto request = Request();

  request.intent.stages[0].parameters[1].boolean_value = true;
  auto preflight = fixture.factory.Preflight(
      request, frozen.plan.resolution, frozen.plan.domain_plan,
      video::VideoExecutionRoute::prepared_resident);
  bool ok = Require(preflight.status == hal::StatusId::blocked &&
                        preflight.reason ==
                            hal::ReasonId::semantic_effect_unsupported,
                    "tracked-center vignette must remain fail-closed");

  request = Request();
  request.intent.stages[0].intent.semantic =
      hal::SemanticEffectId::virtual_background_blur;
  preflight = fixture.factory.Preflight(
      request, frozen.plan.resolution, frozen.plan.domain_plan,
      video::VideoExecutionRoute::prepared_resident);
  ok &= Require(preflight.status == hal::StatusId::blocked &&
                    preflight.reason ==
                        hal::ReasonId::semantic_effect_unsupported,
                "Vulkan matting semantics must remain fail-closed");

  request = Request();
  request.invalidation_key.generations.context++;
  preflight = fixture.factory.Preflight(
      request, frozen.plan.resolution, frozen.plan.domain_plan,
      video::VideoExecutionRoute::prepared_resident);
  ok &= Require(preflight.status == hal::StatusId::stale &&
                    preflight.reason == hal::ReasonId::wrong_generation,
                "wrong context generation must be rejected");

  auto foreign_domain = frozen.plan.domain_plan;
  foreign_domain.islands[0].execution_domain.adapter_instance_id++;
  preflight = fixture.factory.Preflight(
      Request(), frozen.plan.resolution, foreign_domain,
      video::VideoExecutionRoute::prepared_resident);
  ok &= Require(preflight.status == hal::StatusId::blocked &&
                    preflight.reason == hal::ReasonId::wrong_execution_domain,
                "foreign execution domain must be rejected");

  auto malformed_plan = frozen.plan;
  ++malformed_plan.prepared_program.program_id;
  video::VideoAdapterPrepareEvidence prepare_evidence;
  ok &=
      Require(!fixture.factory.Prepare(malformed_plan, &prepare_evidence) &&
                  prepare_evidence.status == hal::StatusId::blocked &&
                  prepare_evidence.reason == hal::ReasonId::malformed_contract,
              "foreign prepared program identity must be rejected");
  return ok;
}

bool TestFailureAndLifecycle() {
  bool ok = true;
  {
    Fixture fixture;
    if (!fixture.Initialize() || !fixture.Prepare(Request()))
      return false;
    std::uint8_t input[kBytes]{};
    std::uint8_t output[kBytes]{};
    const auto frame = fixture.coordinator.Execute(Binding(input, output, 1));
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(
        fixture.coordinator.CommitOutput(frame.completion, false, &reason) ==
                hal::StatusId::failed &&
            reason == hal::ReasonId::output_write_failed &&
            !fixture.coordinator.Snapshot().latest_committed_frame.valid &&
            fixture.coordinator.Snapshot().telemetry.active_backend ==
                hal::BackendId::invalid,
        "writer failure must not commit active attribution");
  }
  {
    Fixture fixture;
    fixture.adapter.mode = ExecuteMode::optional_fail_open;
    if (!fixture.Initialize() || !fixture.Prepare(Request()))
      return false;
    std::uint8_t input[kBytes]{};
    std::uint8_t output[kBytes]{};
    const auto frame = fixture.coordinator.Execute(Binding(input, output, 1));
    hal::ReasonId reason = hal::ReasonId::none;
    ok &= Require(
        frame.status == hal::StatusId::degraded &&
            frame.stages[0].status ==
                hal::StageExecutionStatus::optional_failed_open &&
            frame.stages[0].valid_output_preserved &&
            fixture.coordinator.CommitOutput(frame.completion, true, &reason) ==
                hal::StatusId::ready,
        "optional failure must preserve valid output and commit");
  }
  {
    Fixture fixture;
    fixture.adapter.mode = ExecuteMode::transfer_mismatch;
    if (!fixture.Initialize() || !fixture.Prepare(Request()))
      return false;
    std::uint8_t input[kBytes]{};
    std::uint8_t output[kBytes]{};
    const auto frame = fixture.coordinator.Execute(Binding(input, output, 1));
    ok &= Require(frame.status == hal::StatusId::failed &&
                      frame.reason == hal::ReasonId::transfer_contract_violated,
                  "unexpected transfer/synchronization counts must fail");
  }
  {
    Fixture fixture;
    fixture.adapter.mode = ExecuteMode::device_loss;
    if (!fixture.Initialize() || !fixture.Prepare(Request()))
      return false;
    std::uint8_t input[kBytes]{};
    std::uint8_t output[kBytes]{};
    const auto frame = fixture.coordinator.Execute(Binding(input, output, 1));
    ok &= Require(frame.reason == hal::ReasonId::device_lost &&
                      fixture.coordinator.Snapshot().quarantined,
                  "device loss must quarantine the coordinator route");
  }
  {
    Fixture fixture;
    if (!fixture.Initialize() || !fixture.Prepare(Request()))
      return false;
    fixture.adapter.fail_prepare = true;
    video::VideoExecutionPrepareEvidence evidence;
    hal::ReasonId reason = hal::ReasonId::none;
    const auto status =
        fixture.coordinator.Prepare(Request(), &evidence, &reason);
    ok &= Require(status == hal::StatusId::resource_exhausted &&
                      evidence.candidate_resources_rolled_back &&
                      evidence.old_plan_retained &&
                      fixture.adapter.destroy_calls >= 1,
                  "failed candidate setup must roll back and retain the exact "
                  "old plan");
  }
  return ok;
}

bool TestExecutorShutdownDoesNotOwnSharedAdapter() {
  Fixture fixture;
  if (!fixture.Initialize() || !fixture.Prepare(Request()))
    return false;
  fixture.coordinator.Shutdown();
  bool ok =
      Require(fixture.adapter.shutdown_calls == 0 && !fixture.adapter.shutdown,
              "coordinator executor shutdown must not stop the "
              "factory-owned Vulkan island");

  video::VideoExecutionCoordinator replacement(fixture.registry);
  hal::ReasonId reason = hal::ReasonId::none;
  ok &= Require(
      replacement.RegisterFactory(fixture.factory.factory_registration(),
                                  &reason) == hal::StatusId::ready &&
          replacement.Prepare(Request(), nullptr, &reason) ==
              hal::StatusId::ready,
      "factory-owned adapter must remain reusable after executor shutdown");
  replacement.Shutdown();
  return ok;
}

#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
int OptionalVulkanSkip(hal::ReasonId reason) {
  std::cout << "[SKIP] Vulkan video execution factory production runtime: "
            << static_cast<std::uint32_t>(reason) << '\n';
  const char *required = std::getenv("STUDIOCAST_REQUIRE_VULKAN_RUNTIME");
  return required && std::string(required) == "1" ? 1 : 0;
}

bool RunProductionCombination(vulkan::VulkanVideoExecutionFactory *factory,
                              const hal::StaticAdapterRegistry &registry,
                              video::VideoExecutionPlanRequest request,
                              bool vignette, bool mirror,
                              std::uint64_t sequence) {
  request.invalidation_key.generations.context =
      factory->adapter_registration()
          .execution_domain.logical_context_generation;
  if (!vignette) {
    request.intent.stages[0] = request.intent.stages[1];
    request.intent.stages[1] = {};
    request.intent.stage_count = 1;
  } else if (!mirror) {
    request.intent.stages[1] = {};
    request.intent.stage_count = 1;
  }
  request.plan_id += sequence;
  request.invalidation_key.canonical_plan_hash += sequence;
  const auto resolution = hal::ResolveSemanticStages(
      registry, video::BuildVideoResolutionRequest(request.intent));
  const auto domain_plan = hal::PlanExecutionDomains(
      registry, resolution, hal::CrossAdapterPolicy::reject);
  const auto preflight =
      factory->Preflight(request, resolution, domain_plan,
                         video::VideoExecutionRoute::prepared_resident);
  if (preflight.status != hal::StatusId::ready) {
    std::cerr << "production preflight status/reason: "
              << static_cast<std::uint32_t>(preflight.status) << '/'
              << static_cast<std::uint32_t>(preflight.reason) << '\n';
  }
  if (!Require(preflight.status == hal::StatusId::ready,
               "production combination preflight must succeed")) {
    return false;
  }
  video::FrozenVideoExecutionPlan plan;
  plan.plan_id = request.plan_id;
  plan.route = video::VideoExecutionRoute::prepared_resident;
  plan.intent = request.intent;
  plan.host_input = request.host_input;
  plan.host_output = request.host_output;
  plan.invalidation_key = request.invalidation_key;
  plan.temporal = request.temporal;
  plan.budget = request.budget;
  plan.setup_failure_policy = request.setup_failure_policy;
  plan.migration_policy = request.migration_policy;
  plan.resolution = resolution;
  plan.domain_plan = domain_plan;
  plan.factory_id = factory->factory_registration().factory_id;
  plan.prepared_program = preflight.prepared_program;
  video::VideoAdapterPrepareEvidence prepare_evidence;
  auto executor = factory->Prepare(plan, &prepare_evidence);
  if (!executor) {
    std::cerr << "production prepare status/reason: "
              << static_cast<std::uint32_t>(prepare_evidence.status) << '/'
              << static_cast<std::uint32_t>(prepare_evidence.reason) << '\n';
  }
  if (!Require(executor != nullptr &&
                   prepare_evidence.status == hal::StatusId::ready,
               "production combination must prepare")) {
    return false;
  }
  std::array<std::uint8_t, kBytes> input{};
  std::array<std::uint8_t, kBytes> output{};
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<std::uint8_t>(index * 17u + 5u);
  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations.store(true, std::memory_order_relaxed);
  const auto frame =
      executor->Execute(Binding(input.data(), output.data(), sequence));
  g_count_allocations.store(false, std::memory_order_relaxed);
  bool ok =
      Require((frame.status == hal::StatusId::executed ||
               frame.status == hal::StatusId::degraded) &&
                  frame.transfers.uploads == 1 &&
                  frame.transfers.final_readbacks == 1 &&
                  frame.transfers.intermediate_readbacks == 0 &&
                  frame.transfers.intermediate_uploads == 0 &&
                  frame.transfers.synchronizations == 3,
              "each successful qualified Vulkan stage combination must report "
              "upload + one resident batch + readback synchronization");
  ok &= Require(g_allocations.load(std::memory_order_relaxed) == 0,
                "production Execute must not allocate host storage");
  hal::ReasonId reason = hal::ReasonId::none;
  ok &= Require(executor->CommitOutput(frame.completion, true, &reason) ==
                    hal::StatusId::ready,
                "production combination output must commit");
  executor->Shutdown();
  return ok;
}

int TestProductionRuntimeIfAvailable() {
  vulkan::VulkanVideoExecutionFactory factory({kFactoryId + 1, 300});
  hal::ReasonId reason = hal::ReasonId::none;
  if (factory.Initialize(&reason) != hal::StatusId::ready)
    return OptionalVulkanSkip(reason);
  hal::StaticAdapterRegistry registry;
  if (!Require(registry.Register(factory.adapter_registration(), &reason) ==
                   hal::StatusId::ready,
               "production Vulkan registration must be valid")) {
    return 1;
  }
  bool ok = true;
  ok &=
      RunProductionCombination(&factory, registry, Request(), false, true, 101);
  ok &=
      RunProductionCombination(&factory, registry, Request(), true, false, 102);
  ok &=
      RunProductionCombination(&factory, registry, Request(), true, true, 103);
  return ok ? 0 : 1;
}
#endif

} // namespace

int main() {
  bool ok = true;
  ok &= TestExactFactoryAndCommit();
  ok &= TestUnsupportedAndForeignPreflight();
  ok &= TestFailureAndLifecycle();
  ok &= TestExecutorShutdownDoesNotOwnSharedAdapter();
#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
  ok &= TestProductionRuntimeIfAvailable() == 0;
#endif
  return ok ? 0 : 1;
}
