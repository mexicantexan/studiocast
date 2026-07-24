#include "core/maxine/video_execution_factory.h"
#include "core/video/video_execution_coordinator.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>

namespace {

using namespace studiocast;

std::atomic<std::uint64_t> g_allocations{0};
thread_local bool g_count_allocations = false;

void *Allocate(std::size_t size) {
  if (void *memory = std::malloc(size == 0 ? 1 : size)) {
    if (g_count_allocations)
      g_allocations.fetch_add(1, std::memory_order_relaxed);
    return memory;
  }
  throw std::bad_alloc();
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << __func__ << ':' << __LINE__                                 \
                << ": CHECK failed: " << #condition << '\n';                   \
      return false;                                                            \
    }                                                                          \
  } while (false)

hal::ExecutionDomainId Domain(std::uint64_t adapter = 771) {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::maxine;
  domain.api_family = hal::ApiFamily::maxine_nvcv;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = 0x1cbb;
  domain.physical_device.stable_id[0] = 0x71;
  domain.logical_context_generation = 37;
  domain.allocator_family = hal::AllocatorFamily::maxine_nvcv;
  domain.memory_handle_family = hal::MemoryHandleFamily::nvcv_image;
  domain.queue_or_stream_identity = 0x7710;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::maxine_bound_cuda_stream;
  domain.adapter_instance_id = adapter;
  return domain;
}

hal::ImageDescriptor Image(hal::PixelFormat format, hal::ChannelOrder order,
                           hal::HostCoherence coherence,
                           hal::Mutability mutability) {
  hal::ImageDescriptor image;
  image.width = 8;
  image.height = 3;
  image.plane_count = 1;
  image.format = format;
  image.channel_order = order;
  image.color_space = hal::ColorSpace::srgb;
  image.color_range = hal::ColorRange::full;
  image.orientation = hal::Orientation::top_left;
  image.host_coherence = coherence;
  image.mutability = mutability;
  image.aliasing = hal::AliasingPermission::forbidden;
  image.base_alignment_bytes = 1;
  image.planes[0].row_stride_bytes = 24;
  image.planes[0].size_bytes = 72;
  image.planes[0].width = 8;
  image.planes[0].height = 3;
  image.planes[0].alignment_bytes = 1;
  return image;
}

hal::ResourceHandle Resource(const hal::PreparedProgram &program,
                             std::uint64_t id, hal::MemoryDomain domain,
                             std::uint64_t bytes) {
  hal::ResourceHandle resource;
  resource.resource_id = id;
  resource.allocator_owner_id = program.owner_adapter_instance_id;
  resource.lifetime_generation =
      program.execution_domain.logical_context_generation;
  resource.memory_domain = domain;
  resource.lifetime = hal::ResourceLifetime::frame_borrowed;
  resource.synchronization_state = hal::SynchronizationState::ready;
  resource.execution_domain = program.execution_domain;
  resource.size_bytes = bytes;
  resource.alignment_bytes = 1;
  resource.mutable_resource = true;
  return resource;
}

enum class ExecuteMode {
  success,
  optional_fail_open,
  wrong_generation,
  device_loss,
};

class FakeMaxineResident final : public hal::IResidentAdapter {
public:
  std::uint64_t adapter_instance_id() const noexcept override { return 771; }
  hal::BackendId backend_id() const noexcept override {
    return hal::BackendId::maxine;
  }

  hal::StatusId Prepare(const hal::PreparedProgram &program,
                        hal::PreparedProgramHandle *candidate,
                        hal::ReadinessSnapshot *readiness,
                        hal::ReasonId *reason) noexcept override {
    ++prepare_calls;
    program_ = program;
    candidate->handle_id = 100 + prepare_calls;
    candidate->program_id = program.program_id;
    candidate->adapter_instance_id = adapter_instance_id();
    candidate->invalidation_key = program.invalidation_key;
    candidate->execution_domain = program.execution_domain;
    candidate->reserved_budget = program.budget;
    readiness->status = hal::StatusId::degraded;
    readiness->reason = hal::ReasonId::cpu_tail_required;
    readiness->requested_backend = hal::BackendId::maxine;
    readiness->selected_backend = hal::BackendId::maxine;
    readiness->provider = hal::SemanticEffectProviderId::maxine_vfx;
    readiness->execution_domain = program.execution_domain;
    readiness->ready_stage_mask = 1;
    readiness->available = true;
    readiness->degraded = true;
    if (reason)
      *reason = hal::ReasonId::cpu_tail_required;
    return hal::StatusId::degraded;
  }

  hal::StatusId BindHostFrame(const hal::PreparedProgramHandle &,
                              const hal::HostFrameBinding &binding,
                              hal::HostFrameTokens *tokens,
                              hal::ReasonId *reason) noexcept override {
    binding_ = binding;
    tokens->input = Resource(program_, 1000 + binding.capture_sequence,
                             hal::MemoryDomain::upload_staging,
                             binding.input_planes[0].span_bytes);
    tokens->input.mutable_resource = false;
    tokens->output = Resource(program_, 2000 + binding.capture_sequence,
                              hal::MemoryDomain::readback_staging,
                              binding.output_planes[0].span_bytes);
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
    result.status = hal::StatusId::executed;
    result.completion = {next_token_++, program_.program_id,
                         adapter_instance_id(), request.capture_sequence, true};
    result.output = tokens_.output;
    result.output_valid = true;
    result.stage_count = program_.stage_count;
    result.transfers.uploads = 1;
    result.transfers.final_readbacks = 1;
    result.transfers.synchronizations = 1;
    result.transfers.cpu_tail_boundaries = 1;
    std::memcpy(binding_.output_planes[0].write_data,
                binding_.input_planes[0].read_data, 72);
    for (std::size_t index = 0; index < result.stage_count; ++index) {
      result.stages[index].stage_id = program_.stages[index].stage_id;
      result.stages[index].semantic = program_.stages[index].semantic;
      result.stages[index].status = hal::StageExecutionStatus::succeeded;
      result.stages[index].valid_output_preserved = true;
    }
    switch (mode) {
    case ExecuteMode::success:
      break;
    case ExecuteMode::optional_fail_open:
      result.status = hal::StatusId::degraded;
      result.reason = hal::ReasonId::optional_stage_failed_open;
      result.stages[0].status = hal::StageExecutionStatus::optional_failed_open;
      result.stages[0].reason = hal::ReasonId::optional_stage_failed_open;
      break;
    case ExecuteMode::wrong_generation:
      ++result.output.lifetime_generation;
      break;
    case ExecuteMode::device_loss:
      result.status = hal::StatusId::failed;
      result.reason = hal::ReasonId::device_lost;
      result.output_valid = false;
      break;
    }
    return result;
  }

  hal::StatusId ResolveHostFrameOutput(
      const hal::PreparedProgramHandle &, const hal::CompletionToken &,
      const hal::ResourceHandle &, hal::HostFrameOutputResolution *resolution,
      hal::ReasonId *reason) noexcept override {
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
    return hal::StatusId::blocked;
  }
  void DestroyPrepared(hal::PreparedProgramHandle *handle) noexcept override {
    ++destroy_calls;
    *handle = {};
  }
  void Shutdown() noexcept override { ++shutdown_calls; }

  ExecuteMode mode = ExecuteMode::success;
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

hal::AdapterRegistration Registration(FakeMaxineResident *resident) {
  hal::AdapterRegistration registration;
  registration.adapter_instance_id = resident->adapter_instance_id();
  registration.backend = hal::BackendId::maxine;
  registration.execution_domain = Domain();
  registration.selection_priority = 10;
  registration.resident_adapter = resident;
  auto &capability = registration.semantic_capabilities[0];
  capability.semantic = hal::SemanticEffectId::virtual_background_blur;
  capability.provider = hal::SemanticEffectProviderId::maxine_vfx;
  capability.status = hal::StatusId::ready;
  capability.cpu_tail = hal::CpuTailId::output_format_conversion;
  capability.cpu_tail_placement = hal::DeclaredCpuTail::Placement::after_stage;
  capability.live_stage = true;
  capability.production_eligible = true;
  capability.exact_evidence = true;
  capability.evidence[0] = {
      hal::CapabilityEvidenceCode::provider_runtime_available,
      hal::StatusId::ready,
      hal::ReasonId::none,
      capability.semantic,
      hal::BackendId::maxine,
      true,
  };
  capability.evidence_count = 1;
  registration.semantic_capability_count = 1;
  return registration;
}

video::VideoExecutionPlanRequest Request() {
  video::VideoExecutionPlanRequest request;
  request.plan_id = 9001;
  request.intent.backend.mode = hal::BackendRequestMode::explicit_backend;
  request.intent.backend.backend = hal::BackendId::maxine;
  request.intent.backend.strict = true;
  request.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::cuda_family;
  request.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::maxine;
  request.intent.preferences.strict_effects_engine = true;
  request.intent.requested_provider = hal::SemanticEffectProviderId::maxine_vfx;
  request.intent.strict_provider_selection = true;
  auto &stage = request.intent.stages[0];
  stage.intent.stage_id = 1;
  stage.intent.semantic = hal::SemanticEffectId::virtual_background_blur;
  stage.intent.failure_action =
      hal::FailureAction::fail_open_preserve_last_valid;
  stage.parameters[0] = {
      video::CanonicalVideoParameterId::strength,
      video::CanonicalVideoParameterKind::signed_integer,
      8,
  };
  stage.parameter_count = 1;
  stage.model.model_id_hash[0] = 0x11;
  stage.model.artifact_set_hash[0] = 0x22;
  stage.model.artifact_generation = 13;
  request.intent.stage_count = 1;
  request.host_input =
      Image(hal::PixelFormat::rgb_u8, hal::ChannelOrder::rgb,
            hal::HostCoherence::coherent, hal::Mutability::read_only);
  request.host_output =
      Image(hal::PixelFormat::rgb_u8, hal::ChannelOrder::rgb,
            hal::HostCoherence::coherent, hal::Mutability::write_only);
  request.invalidation_key.generations = {2, 3, 13, 5, 7, 37};
  request.invalidation_key.canonical_plan_hash = 101;
  request.invalidation_key.model_artifact_hash = 102;
  request.invalidation_key.execution_domain_hash = 103;
  request.budget.persistent_bytes = 1024;
  request.budget.scratch_bytes = 256;
  request.budget.upload_staging_bytes = 72;
  request.budget.readback_staging_bytes = 72;
  request.budget.resource_count = 8;
  request.budget.in_flight_slots = 1;
  request.cpu_tails[0].id = hal::CpuTailId::output_format_conversion;
  request.cpu_tails[0].placement = hal::DeclaredCpuTail::Placement::after_stage;
  request.cpu_tails[0].after_stage_id = 1;
  request.cpu_tails[0].boundary_index = 1;
  request.cpu_tail_count = 1;
  return request;
}

hal::HostFrameBinding Binding(std::uint8_t *input, std::uint8_t *output,
                              std::uint64_t sequence) {
  hal::HostFrameBinding binding;
  binding.input_descriptor =
      Image(hal::PixelFormat::rgb_u8, hal::ChannelOrder::rgb,
            hal::HostCoherence::coherent, hal::Mutability::read_only);
  binding.output_descriptor =
      Image(hal::PixelFormat::rgb_u8, hal::ChannelOrder::rgb,
            hal::HostCoherence::coherent, hal::Mutability::write_only);
  binding.input_planes[0].read_data =
      reinterpret_cast<const std::byte *>(input);
  binding.input_planes[0].span_bytes = 72;
  binding.input_planes[0].row_stride_bytes = 24;
  binding.output_planes[0].write_data = reinterpret_cast<std::byte *>(output);
  binding.output_planes[0].span_bytes = 72;
  binding.output_planes[0].row_stride_bytes = 24;
  binding.input_plane_count = 1;
  binding.output_plane_count = 1;
  binding.capture_sequence = sequence;
  binding.ownership = hal::OwnershipModel::borrowed_until_completion;
  binding.lifetime = hal::ResourceLifetime::frame_borrowed;
  return binding;
}

struct Fixture {
  Fixture(std::uint64_t minimum_synchronizations = 1,
          std::uint64_t maximum_synchronizations = 1)
      : adapter_registration(Registration(&resident)), factory([&] {
          maxine::MaxineVideoExecutionFactoryConfiguration config;
          config.adapter_registration = &adapter_registration;
          config.factory_id = 8801;
          config.selection_priority = 10;
          config.minimum_synchronizations_per_frame = minimum_synchronizations;
          config.maximum_synchronizations_per_frame = maximum_synchronizations;
          config.setup_template.vfx_model_directory = "/hermetic/maxine/models";
          return config;
        }()),
        coordinator(registry) {
    hal::ReasonId reason = hal::ReasonId::none;
    registry_status = registry.Register(adapter_registration, &reason);
    factory_status =
        coordinator.RegisterFactory(factory.registration(), &reason);
  }

  bool Prepare(video::VideoExecutionPlanRequest request = Request()) {
    video::VideoExecutionPrepareEvidence evidence;
    hal::ReasonId reason = hal::ReasonId::none;
    const auto status = coordinator.Prepare(request, &evidence, &reason);
    return (status == hal::StatusId::ready ||
            status == hal::StatusId::degraded) &&
           evidence.selected_factory_id == 8801;
  }

  FakeMaxineResident resident;
  hal::AdapterRegistration adapter_registration;
  hal::StaticAdapterRegistry registry;
  maxine::MaxineVideoExecutionFactory factory;
  video::VideoExecutionCoordinator coordinator;
  hal::StatusId registry_status = hal::StatusId::invalid;
  hal::StatusId factory_status = hal::StatusId::invalid;
};

bool ExactPreflightAndMalformedForeignGenerationRejection() {
  Fixture fixture;
  CHECK(fixture.registry_status == hal::StatusId::ready);
  CHECK(fixture.factory_status == hal::StatusId::ready);
  const auto request = Request();
  const auto resolution = hal::ResolveSemanticStages(
      fixture.registry, video::BuildVideoResolutionRequest(request.intent));
  const auto domains = hal::PlanExecutionDomains(
      fixture.registry, resolution, hal::CrossAdapterPolicy::reject);
  auto preflight =
      fixture.factory.Preflight(request, resolution, domains,
                                video::VideoExecutionRoute::prepared_resident);
  CHECK(preflight.status == hal::StatusId::degraded);
  CHECK(preflight.cached_evidence_only);
  CHECK(preflight.accepts_frozen_resolution);
  CHECK(!preflight.performs_selection);
  CHECK(preflight.backend == hal::BackendId::maxine);
  CHECK(preflight.provider == hal::SemanticEffectProviderId::maxine_vfx);
  CHECK(preflight.execution_domain == Domain());
  CHECK(preflight.prepared_program.program_id == request.plan_id);
  CHECK(preflight.prepared_program.invalidation_key ==
        request.invalidation_key);
  CHECK(preflight.prepared_program.execution_domain == Domain());
  CHECK(preflight.prepared_program.stage_count == 1);
  CHECK(preflight.prepared_program.cpu_tail_count == 1);
  CHECK(preflight.per_frame_transfers.minimum_synchronizations == 1);
  CHECK(preflight.per_frame_transfers.maximum_synchronizations == 1);

  auto wrong_generation = request;
  ++wrong_generation.invalidation_key.generations.context;
  preflight =
      fixture.factory.Preflight(wrong_generation, resolution, domains,
                                video::VideoExecutionRoute::prepared_resident);
  CHECK(preflight.status == hal::StatusId::blocked);
  CHECK(preflight.reason == hal::ReasonId::wrong_generation);

  auto malformed = request;
  ++malformed.host_input.planes[0].row_stride_bytes;
  preflight =
      fixture.factory.Preflight(malformed, resolution, domains,
                                video::VideoExecutionRoute::prepared_resident);
  CHECK(preflight.status == hal::StatusId::blocked);
  CHECK(preflight.reason == hal::ReasonId::malformed_descriptor);

  auto foreign = domains;
  ++foreign.islands[0].execution_domain.adapter_instance_id;
  preflight =
      fixture.factory.Preflight(request, resolution, foreign,
                                video::VideoExecutionRoute::prepared_resident);
  CHECK(preflight.status == hal::StatusId::blocked);
  CHECK(preflight.reason == hal::ReasonId::wrong_execution_domain);

  Fixture malformed_sync(0, 0);
  const auto malformed_resolution = hal::ResolveSemanticStages(
      malformed_sync.registry,
      video::BuildVideoResolutionRequest(request.intent));
  const auto malformed_domains =
      hal::PlanExecutionDomains(malformed_sync.registry, malformed_resolution,
                                hal::CrossAdapterPolicy::reject);
  preflight = malformed_sync.factory.Preflight(
      request, malformed_resolution, malformed_domains,
      video::VideoExecutionRoute::prepared_resident);
  CHECK(preflight.status == hal::StatusId::blocked);
  CHECK(preflight.reason == hal::ReasonId::malformed_contract);
  return true;
}

bool TransfersCpuTailWriterAttributionAndAllocationFreeExecute() {
  Fixture fixture;
  CHECK(fixture.Prepare());
  std::array<std::uint8_t, 72> input{};
  std::array<std::uint8_t, 72> output{};
  input.fill(0x5a);
  hal::ReasonId reason = hal::ReasonId::none;

  auto first =
      fixture.coordinator.Execute(Binding(input.data(), output.data(), 1));
  CHECK(first.status == hal::StatusId::executed);
  CHECK(first.output_ready);
  CHECK(first.execution_domain == Domain());
  CHECK(first.transfers.uploads == 1);
  CHECK(first.transfers.final_readbacks == 1);
  CHECK(first.transfers.intermediate_uploads == 0);
  CHECK(first.transfers.intermediate_readbacks == 0);
  CHECK(first.transfers.synchronizations == 1);
  CHECK(first.transfers.cpu_tail_boundaries == 1);
  CHECK(output == input);
  CHECK(fixture.coordinator.CommitOutput(first.completion, false, &reason) ==
        hal::StatusId::failed);
  CHECK(reason == hal::ReasonId::output_write_failed);
  auto snapshot = fixture.coordinator.Snapshot();
  CHECK(!snapshot.latest_committed_frame.valid);
  CHECK(snapshot.telemetry.active_backend == hal::BackendId::invalid);

  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations = true;
  bool success = true;
  for (std::uint64_t sequence = 2; sequence != 66; ++sequence) {
    const auto frame = fixture.coordinator.Execute(
        Binding(input.data(), output.data(), sequence));
    if (frame.status != hal::StatusId::executed ||
        fixture.coordinator.CommitOutput(frame.completion, true, &reason) !=
            hal::StatusId::ready) {
      success = false;
      break;
    }
  }
  g_count_allocations = false;
  CHECK(success);
  CHECK(g_allocations.load(std::memory_order_relaxed) == 0);
  snapshot = fixture.coordinator.Snapshot();
  CHECK(snapshot.latest_committed_frame.valid);
  CHECK(snapshot.latest_committed_frame.capture_sequence == 65);
  CHECK(snapshot.telemetry.active_backend == hal::BackendId::maxine);
  CHECK(snapshot.telemetry.active_provider ==
        hal::SemanticEffectProviderId::maxine_vfx);
  return true;
}

bool OptionalFailOpenWrongGenerationDeviceLossAndShutdown() {
  {
    Fixture fixture;
    fixture.resident.mode = ExecuteMode::optional_fail_open;
    CHECK(fixture.Prepare());
    std::array<std::uint8_t, 72> input{};
    std::array<std::uint8_t, 72> output{};
    input.fill(0x3c);
    const auto frame =
        fixture.coordinator.Execute(Binding(input.data(), output.data(), 1));
    CHECK(frame.status == hal::StatusId::degraded);
    CHECK(frame.output_ready && output == input);
    CHECK(frame.stages[0].status ==
          hal::StageExecutionStatus::optional_failed_open);
    hal::ReasonId reason = hal::ReasonId::none;
    CHECK(fixture.coordinator.CommitOutput(frame.completion, true, &reason) ==
          hal::StatusId::ready);
    const auto snapshot = fixture.coordinator.Snapshot();
    CHECK(snapshot.latest_committed_frame.valid);
    CHECK(snapshot.telemetry.active_backend == hal::BackendId::invalid);
    CHECK(snapshot.telemetry.optional_fail_open_stages == 1);
  }
  {
    Fixture fixture;
    fixture.resident.mode = ExecuteMode::wrong_generation;
    CHECK(fixture.Prepare());
    std::array<std::uint8_t, 72> input{};
    std::array<std::uint8_t, 72> output{};
    const auto frame =
        fixture.coordinator.Execute(Binding(input.data(), output.data(), 1));
    CHECK(frame.status == hal::StatusId::failed ||
          frame.status == hal::StatusId::blocked);
    CHECK(frame.reason == hal::ReasonId::wrong_generation ||
          frame.reason == hal::ReasonId::invalid_resource);
    CHECK(!fixture.coordinator.Snapshot().frame_pending);
  }
  {
    Fixture fixture;
    fixture.resident.mode = ExecuteMode::device_loss;
    CHECK(fixture.Prepare());
    std::array<std::uint8_t, 72> input{};
    std::array<std::uint8_t, 72> output{};
    const auto frame =
        fixture.coordinator.Execute(Binding(input.data(), output.data(), 1));
    CHECK(frame.reason == hal::ReasonId::device_lost);
    CHECK(fixture.coordinator.Snapshot().quarantined);
    fixture.coordinator.Shutdown();
    CHECK(fixture.coordinator.Snapshot().stopped);
    CHECK(fixture.resident.shutdown_calls != 0);
  }
  return true;
}

bool TemporalDiscontinuityQuarantinesUntilSetupReprepare() {
  Fixture fixture;
  auto request = Request();
  request.temporal.temporal = true;
  request.temporal.requires_contiguous_capture_sequence = true;
  request.temporal.frame_drop_behavior =
      hal::FrameDropBehavior::reset_temporal_state;
  request.temporal.discontinuity_action =
      hal::TemporalDiscontinuityAction::reset_and_continue;
  request.temporal.history_frames = 1;
  CHECK(fixture.Prepare(request));

  std::array<std::uint8_t, 72> input{};
  std::array<std::uint8_t, 72> output{};
  hal::ReasonId reason = hal::ReasonId::none;
  const auto first =
      fixture.coordinator.Execute(Binding(input.data(), output.data(), 1));
  CHECK(first.status == hal::StatusId::executed);
  CHECK(fixture.coordinator.CommitOutput(first.completion, true, &reason) ==
        hal::StatusId::ready);

  CHECK(fixture.coordinator.ResetTemporal(
            hal::ReasonId::temporal_discontinuity) ==
        hal::StatusId::quarantined);
  const auto quarantined =
      fixture.coordinator.Execute(Binding(input.data(), output.data(), 2));
  CHECK(quarantined.status == hal::StatusId::quarantined);
  CHECK(quarantined.reason == hal::ReasonId::temporal_discontinuity);

  CHECK(fixture.Prepare(request));
  const auto rebuilt =
      fixture.coordinator.Execute(Binding(input.data(), output.data(), 3));
  CHECK(rebuilt.status == hal::StatusId::executed);
  CHECK(fixture.coordinator.CommitOutput(rebuilt.completion, true, &reason) ==
        hal::StatusId::ready);
  return true;
}

} // namespace

void *operator new(std::size_t size) { return Allocate(size); }
void *operator new[](std::size_t size) { return Allocate(size); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}

int main() {
  if (!ExactPreflightAndMalformedForeignGenerationRejection() ||
      !TransfersCpuTailWriterAttributionAndAllocationFreeExecute() ||
      !OptionalFailOpenWrongGenerationDeviceLossAndShutdown() ||
      !TemporalDiscontinuityQuarantinesUntilSetupReprepare()) {
    return EXIT_FAILURE;
  }
  std::cout << "Maxine video execution factory tests passed\n";
  return EXIT_SUCCESS;
}
