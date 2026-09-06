#include "core/video/video_execution_coordinator.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
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

hal::ExecutionDomainId Domain(hal::BackendId backend = hal::BackendId::cuda,
                              std::uint64_t adapter = 41) {
  hal::ExecutionDomainId domain;
  domain.backend = backend;
  domain.api_family = backend == hal::BackendId::cuda
                          ? hal::ApiFamily::cuda_driver
                          : hal::ApiFamily::maxine_nvcv;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = 0x1cbb;
  domain.physical_device.stable_id[0] = static_cast<std::uint8_t>(adapter);
  domain.logical_context_generation = 100 + adapter;
  domain.allocator_family = backend == hal::BackendId::cuda
                                ? hal::AllocatorFamily::cuda_driver
                                : hal::AllocatorFamily::maxine_nvcv;
  domain.memory_handle_family =
      backend == hal::BackendId::cuda
          ? hal::MemoryHandleFamily::cuda_device_pointer
          : hal::MemoryHandleFamily::nvcv_image;
  domain.queue_or_stream_identity = 200 + adapter;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      backend == hal::BackendId::cuda
          ? hal::SynchronizationProtocol::cuda_stream_ordered
          : hal::SynchronizationProtocol::maxine_bound_cuda_stream;
  domain.adapter_instance_id = adapter;
  return domain;
}

hal::ImageDescriptor Image(hal::HostCoherence coherence,
                           hal::Mutability mutability) {
  hal::ImageDescriptor image;
  image.width = 4;
  image.height = 2;
  image.plane_count = 1;
  image.format = hal::PixelFormat::rgb_u8;
  image.channel_order = hal::ChannelOrder::rgb;
  image.color_space = hal::ColorSpace::srgb;
  image.color_range = hal::ColorRange::full;
  image.orientation = hal::Orientation::top_left;
  image.host_coherence = coherence;
  image.mutability = mutability;
  image.aliasing = hal::AliasingPermission::forbidden;
  image.base_alignment_bytes = 1;
  image.planes[0].row_stride_bytes = 12;
  image.planes[0].size_bytes = 24;
  image.planes[0].width = 4;
  image.planes[0].height = 2;
  image.planes[0].alignment_bytes = 1;
  return image;
}

video::VideoExecutionPlanRequest Request(bool with_stage = true) {
  video::VideoExecutionPlanRequest request;
  request.plan_id = 500;
  request.intent.backend.mode = with_stage
                                    ? hal::BackendRequestMode::explicit_backend
                                    : hal::BackendRequestMode::automatic;
  request.intent.backend.backend =
      with_stage ? hal::BackendId::cuda : hal::BackendId::invalid;
  request.intent.backend.strict = with_stage;
  request.intent.preferences.compute_family =
      with_stage ? hal::ComputeFamilyAllowance::cuda_family
                 : hal::ComputeFamilyAllowance::any;
  request.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::open_cuda;
  request.intent.preferences.strict_effects_engine = with_stage;
  request.intent.requested_provider =
      with_stage ? hal::SemanticEffectProviderId::open_video_onnxruntime_cuda
                 : hal::SemanticEffectProviderId::invalid;
  request.intent.strict_provider_selection = with_stage;
  if (with_stage) {
    auto &stage = request.intent.stages[0];
    stage.intent.stage_id = 1;
    stage.intent.semantic = hal::SemanticEffectId::virtual_background_blur;
    stage.intent.required = false;
    stage.intent.failure_action = hal::FailureAction::fail_open_preserve_input;
    stage.parameters[0].id = video::CanonicalVideoParameterId::strength;
    stage.parameters[0].kind =
        video::CanonicalVideoParameterKind::signed_integer;
    stage.parameters[0].signed_value = 8;
    stage.parameter_count = 1;
    request.intent.stage_count = 1;
  }
  request.host_input =
      Image(hal::HostCoherence::coherent, hal::Mutability::read_only);
  request.host_output =
      Image(hal::HostCoherence::coherent, hal::Mutability::write_only);
  request.invalidation_key.generations = {1, 1, 1, 1, 1, 141};
  request.invalidation_key.canonical_plan_hash = 1001;
  request.invalidation_key.model_artifact_hash = 2001;
  request.invalidation_key.execution_domain_hash = 3001;
  request.budget.persistent_bytes = 128;
  request.budget.scratch_bytes = 64;
  request.budget.upload_staging_bytes = 24;
  request.budget.readback_staging_bytes = 24;
  request.budget.resource_count = 4;
  request.budget.in_flight_slots = 1;
  return request;
}

hal::HostFrameBinding Binding(std::uint8_t *input, std::uint8_t *output,
                              std::uint64_t sequence,
                              bool passthrough = false) {
  hal::HostFrameBinding binding;
  binding.input_descriptor =
      Image(hal::HostCoherence::coherent, hal::Mutability::read_only);
  binding.output_descriptor =
      Image(hal::HostCoherence::coherent, hal::Mutability::write_only);
  if (passthrough) {
    binding.input_descriptor.mutability = hal::Mutability::write_only;
    binding.output_descriptor = binding.input_descriptor;
  }
  binding.input_planes[0].read_data =
      reinterpret_cast<const std::byte *>(input);
  binding.input_planes[0].span_bytes = 24;
  binding.input_planes[0].row_stride_bytes = 12;
  binding.output_planes[0].write_data =
      reinterpret_cast<std::byte *>(passthrough ? input : output);
  binding.output_planes[0].span_bytes = 24;
  binding.output_planes[0].row_stride_bytes = 12;
  binding.input_plane_count = 1;
  binding.output_plane_count = 1;
  binding.capture_sequence = sequence;
  return binding;
}

class DummyResident final : public hal::IResidentAdapter {
public:
  explicit DummyResident(hal::ExecutionDomainId domain) : domain_(domain) {}
  std::uint64_t adapter_instance_id() const noexcept override {
    return domain_.adapter_instance_id;
  }
  hal::BackendId backend_id() const noexcept override {
    return domain_.backend;
  }
  hal::StatusId Prepare(const hal::PreparedProgram &,
                        hal::PreparedProgramHandle *, hal::ReadinessSnapshot *,
                        hal::ReasonId *) noexcept override {
    return hal::StatusId::blocked;
  }
  hal::ExecuteResult Execute(const hal::PreparedProgramHandle &,
                             const hal::ExecuteRequest &) noexcept override {
    return {};
  }
  hal::StatusId ResetTemporalState(const hal::PreparedProgramHandle &,
                                   hal::ReasonId) noexcept override {
    return hal::StatusId::ready;
  }
  void DestroyPrepared(hal::PreparedProgramHandle *) noexcept override {}
  void Shutdown() noexcept override {}

private:
  hal::ExecutionDomainId domain_{};
};

hal::SemanticCapabilityEvidence
Capability(hal::BackendId backend, hal::SemanticEffectProviderId provider,
           hal::CpuTailId cpu_tail = hal::CpuTailId::invalid,
           std::uint64_t cpu_tail_scratch_bytes = 0) {
  hal::SemanticCapabilityEvidence capability;
  capability.semantic = hal::SemanticEffectId::virtual_background_blur;
  capability.provider = provider;
  capability.status = hal::StatusId::ready;
  capability.live_stage = true;
  capability.production_eligible = true;
  capability.exact_evidence = true;
  capability.cpu_tail = cpu_tail;
  capability.cpu_tail_scratch_bytes = cpu_tail_scratch_bytes;
  capability.cpu_tail_placement = hal::DeclaredCpuTail::Placement::after_stage;
  capability.evidence[0] = {
      hal::CapabilityEvidenceCode::provider_runtime_available,
      hal::StatusId::ready,
      hal::ReasonId::none,
      capability.semantic,
      backend,
      true};
  capability.evidence_count = 1;
  return capability;
}

bool RegisterCapability(hal::StaticAdapterRegistry *registry,
                        DummyResident *adapter,
                        const hal::ExecutionDomainId &domain,
                        hal::SemanticEffectProviderId provider,
                        std::uint32_t priority = 10,
                        hal::CpuTailId cpu_tail = hal::CpuTailId::invalid,
                        std::uint64_t cpu_tail_scratch_bytes = 0) {
  hal::AdapterRegistration registration;
  registration.adapter_instance_id = domain.adapter_instance_id;
  registration.backend = domain.backend;
  registration.execution_domain = domain;
  registration.selection_priority = priority;
  registration.resident_adapter = adapter;
  registration.semantic_capabilities[0] =
      Capability(domain.backend, provider, cpu_tail, cpu_tail_scratch_bytes);
  registration.semantic_capability_count = 1;
  hal::ReasonId reason = hal::ReasonId::none;
  return registry->Register(registration, &reason) == hal::StatusId::ready;
}

enum class ExecuteMode {
  success,
  optional_fail_open,
  wrong_domain,
  stale_program,
  device_loss,
  transfer_mismatch,
  legacy_boundary_mismatch,
  variable_maximum,
  variable_outside,
};

class FakeExecutor final : public video::IVideoExecutionExecutor {
public:
  FakeExecutor(const video::FrozenVideoExecutionPlan &plan, ExecuteMode mode)
      : plan_(plan), mode_(mode) {
    snapshot_.status = hal::StatusId::ready;
    snapshot_.execution_domain = plan_.domain_plan.islands[0].execution_domain;
    snapshot_.plan_id = plan_.plan_id;
    snapshot_.prepared = true;
    snapshot_.readiness.status = hal::StatusId::ready;
    snapshot_.readiness.requested_backend = plan_.resolution.requested_backend;
    snapshot_.readiness.selected_backend = plan_.resolution.selected_backend;
    snapshot_.readiness.provider = plan_.resolution.selected_provider;
    snapshot_.readiness.execution_domain = snapshot_.execution_domain;
    snapshot_.readiness.ready_stage_mask = 1;
    snapshot_.readiness.available = true;
  }

  video::VideoExecutionFrameResult
  Execute(const hal::HostFrameBinding &binding) noexcept override {
    ++execute_calls;
    video::VideoExecutionFrameResult result;
    result.status = hal::StatusId::executed;
    result.completion = {next_token_++, plan_.plan_id,
                         plan_.domain_plan.islands[0].adapter_instance_id,
                         binding.capture_sequence, true};
    result.output.descriptor = binding.output_descriptor;
    result.output.planes = binding.output_planes;
    result.output.plane_count = binding.output_plane_count;
    result.output.capture_sequence = binding.capture_sequence;
    result.output.output_ready = true;
    result.output_ready = true;
    result.execution_domain = plan_.domain_plan.islands[0].execution_domain;
    result.stage_count = 1;
    result.stages[0].stage_id = 1;
    result.stages[0].semantic = hal::SemanticEffectId::virtual_background_blur;
    result.stages[0].status = hal::StageExecutionStatus::succeeded;
    result.stages[0].valid_output_preserved = true;
    result.transfers.uploads = plan_.per_frame_transfers.uploads;
    result.transfers.final_readbacks =
        plan_.per_frame_transfers.final_readbacks;
    result.transfers.intermediate_readbacks =
        plan_.per_frame_transfers.intermediate_readbacks;
    result.transfers.intermediate_uploads =
        plan_.per_frame_transfers.intermediate_uploads;
    result.transfers.synchronizations =
        plan_.per_frame_transfers.minimum_synchronizations;
    result.transfers.cpu_tail_boundaries =
        plan_.per_frame_transfers.cpu_tail_boundaries;
    std::memcpy(binding.output_planes[0].write_data,
                binding.input_planes[0].read_data, 24);
    switch (mode_) {
    case ExecuteMode::success:
      break;
    case ExecuteMode::optional_fail_open:
      result.status = hal::StatusId::degraded;
      result.reason = hal::ReasonId::optional_stage_failed_open;
      result.stages[0].status = hal::StageExecutionStatus::optional_failed_open;
      result.stages[0].reason = hal::ReasonId::optional_stage_failed_open;
      break;
    case ExecuteMode::wrong_domain:
      ++result.execution_domain.adapter_instance_id;
      break;
    case ExecuteMode::stale_program:
      ++result.completion.program_id;
      break;
    case ExecuteMode::device_loss:
      result.status = hal::StatusId::failed;
      result.reason = hal::ReasonId::device_lost;
      result.output_ready = false;
      break;
    case ExecuteMode::transfer_mismatch:
      result.transfers.synchronizations =
          plan_.per_frame_transfers.maximum_synchronizations + 1;
      break;
    case ExecuteMode::legacy_boundary_mismatch:
      ++result.transfers.intermediate_readbacks;
      break;
    case ExecuteMode::variable_maximum:
      result.transfers.intermediate_readbacks =
          plan_.per_frame_transfers.maximum_intermediate_readbacks;
      result.transfers.intermediate_uploads =
          plan_.per_frame_transfers.maximum_intermediate_uploads;
      result.transfers.cpu_tail_boundaries =
          plan_.per_frame_transfers.maximum_cpu_tail_boundaries;
      break;
    case ExecuteMode::variable_outside:
      result.transfers.intermediate_readbacks =
          plan_.per_frame_transfers.maximum_intermediate_readbacks + 1;
      break;
    }
    return result;
  }

  hal::StatusId CommitOutput(const hal::CompletionToken &,
                             bool output_write_succeeded,
                             hal::ReasonId *reason) noexcept override {
    ++commit_calls;
    if (!output_write_succeeded) {
      if (reason)
        *reason = hal::ReasonId::output_write_failed;
      return hal::StatusId::failed;
    }
    if (reason)
      *reason = hal::ReasonId::none;
    return hal::StatusId::ready;
  }
  void Invalidate(hal::ReasonId) noexcept override {
    ++invalidate_calls;
    snapshot_.prepared = false;
  }
  hal::StatusId ResetTemporal(hal::ReasonId) noexcept override {
    ++reset_calls;
    return hal::StatusId::ready;
  }
  void Shutdown() noexcept override {
    ++shutdown_calls;
    snapshot_.stopped = true;
  }
  video::VideoExecutionAdapterSnapshot Snapshot() const noexcept override {
    return snapshot_;
  }

  std::uint64_t execute_calls = 0;
  std::uint64_t commit_calls = 0;
  std::uint64_t invalidate_calls = 0;
  std::uint64_t reset_calls = 0;
  std::uint64_t shutdown_calls = 0;

private:
  video::FrozenVideoExecutionPlan plan_{};
  ExecuteMode mode_ = ExecuteMode::success;
  video::VideoExecutionAdapterSnapshot snapshot_{};
  std::uint64_t next_token_ = 1;
};

class FakeFactory final : public video::IVideoAdapterFactory {
public:
  FakeFactory(hal::ExecutionDomainId domain,
              hal::SemanticEffectProviderId provider,
              ExecuteMode mode = ExecuteMode::success)
      : domain_(domain), provider_(provider), execute_mode_(mode) {}

  video::VideoAdapterFactoryPreflight
  Preflight(const video::VideoExecutionPlanRequest &request,
            const hal::ResolutionResult &resolution,
            const hal::DomainPlan &domain_plan,
            video::VideoExecutionRoute route) noexcept override {
    ++preflight_calls;
    observed_resolution = resolution;
    observed_domain_plan = domain_plan;
    observed_route = route;
    video::VideoAdapterFactoryPreflight result;
    result.status = preflight_status;
    result.reason = preflight_reason;
    result.backend = domain_.backend;
    result.provider =
        reported_provider == hal::SemanticEffectProviderId::invalid
            ? provider_
            : reported_provider;
    result.execution_domain =
        wrong_preflight_domain
            ? Domain(domain_.backend, domain_.adapter_instance_id + 9)
            : domain_;
    result.cached_evidence_only = cached_evidence_only;
    result.accepts_frozen_resolution = accepts_frozen_resolution;
    result.performs_selection = performs_selection;
    result.per_frame_transfers =
        variable_boundaries
            ? video::BuildVariablePerFrameTransferContract(
                  domain_plan, minimum_synchronizations,
                  maximum_synchronizations, variable_bounds)
            : video::BuildPerFrameTransferContract(
                  domain_plan, minimum_synchronizations,
                  maximum_synchronizations);
    if (malformed_transfer_boundary)
      ++result.per_frame_transfers.intermediate_readbacks;
    if (malformed_transfer_version)
      ++result.per_frame_transfers.contract_version;
    result.readiness.status = preflight_status;
    result.readiness.requested_backend = resolution.requested_backend;
    result.readiness.selected_backend = result.backend;
    result.readiness.provider = result.provider;
    result.readiness.execution_domain = result.execution_domain;
    result.readiness.available = preflight_status == hal::StatusId::ready;
    if (route == video::VideoExecutionRoute::prepared_resident) {
      result.prepared_program = Program(request, resolution);
      if (malformed_prepared_program)
        result.prepared_program.stages[0].provider =
            hal::SemanticEffectProviderId::invalid;
      if (malformed_prepared_tail &&
          result.prepared_program.cpu_tail_count != 0) {
        ++result.prepared_program.cpu_tails[0].scratch_bytes;
      }
    }
    return result;
  }

  std::unique_ptr<video::IVideoExecutionExecutor>
  Prepare(const video::FrozenVideoExecutionPlan &plan,
          video::VideoAdapterPrepareEvidence *evidence) noexcept override {
    ++prepare_calls;
    observed_plan = plan;
    evidence->status = prepare_status;
    evidence->reason = prepare_reason;
    evidence->candidate_resources_rolled_back = candidate_rolled_back;
    if (prepare_status != hal::StatusId::ready)
      return {};
    auto executor = std::make_unique<FakeExecutor>(plan, execute_mode_);
    latest_executor = executor.get();
    return executor;
  }

  hal::PreparedProgram Program(const video::VideoExecutionPlanRequest &request,
                               const hal::ResolutionResult &resolution) const {
    hal::PreparedProgram program;
    program.program_id = request.plan_id;
    program.owner_adapter_instance_id = domain_.adapter_instance_id;
    program.execution_domain = domain_;
    program.host_input = request.host_input;
    program.host_output = request.host_output;
    program.invalidation_key = request.invalidation_key;
    program.temporal = request.temporal;
    program.budget = request.budget;
    program.cpu_tails = request.cpu_tails;
    program.cpu_tail_count = request.cpu_tail_count;
    program.setup_failure_policy = request.setup_failure_policy;
    program.requested_backend = resolution.requested_backend;
    program.strict_backend_selection = request.intent.backend.strict;
    program.requested_provider = request.intent.requested_provider;
    program.strict_provider_selection =
        request.intent.strict_provider_selection;
    program.requested_execution_domain = domain_;
    program.strict_execution_domain = true;
    program.requires_upload = true;
    program.requires_final_readback = true;
    program.stage_count = 1;
    auto &stage = program.stages[0];
    stage.stage_id = request.intent.stages[0].intent.stage_id;
    stage.semantic = request.intent.stages[0].intent.semantic;
    stage.provider = provider_;
    stage.backend = domain_.backend;
    stage.execution_domain = domain_;
    stage.input.kind = hal::StageIoKind::image;
    stage.input.image = Image(hal::HostCoherence::not_host_visible,
                              hal::Mutability::read_write);
    stage.output = stage.input;
    stage.required = request.intent.stages[0].intent.required;
    stage.failure_action = request.intent.stages[0].intent.failure_action;
    stage.model = request.intent.stages[0].model;
    return program;
  }

  hal::StatusId preflight_status = hal::StatusId::ready;
  hal::ReasonId preflight_reason = hal::ReasonId::none;
  hal::StatusId prepare_status = hal::StatusId::ready;
  hal::ReasonId prepare_reason = hal::ReasonId::none;
  hal::SemanticEffectProviderId reported_provider =
      hal::SemanticEffectProviderId::invalid;
  bool cached_evidence_only = true;
  bool accepts_frozen_resolution = true;
  bool performs_selection = false;
  bool wrong_preflight_domain = false;
  bool malformed_prepared_program = false;
  bool malformed_prepared_tail = false;
  bool malformed_transfer_boundary = false;
  bool malformed_transfer_version = false;
  bool variable_boundaries = false;
  video::VariablePerFrameTransferBounds variable_bounds{};
  bool candidate_rolled_back = false;
  std::uint64_t minimum_synchronizations = 1;
  std::uint64_t maximum_synchronizations = 1;
  std::uint64_t preflight_calls = 0;
  std::uint64_t prepare_calls = 0;
  FakeExecutor *latest_executor = nullptr;
  hal::ResolutionResult observed_resolution{};
  hal::DomainPlan observed_domain_plan{};
  video::VideoExecutionRoute observed_route =
      video::VideoExecutionRoute::passthrough;
  video::FrozenVideoExecutionPlan observed_plan{};

private:
  hal::ExecutionDomainId domain_{};
  hal::SemanticEffectProviderId provider_{};
  ExecuteMode execute_mode_{};
};

video::VideoAdapterFactoryRegistration
FactoryRegistration(std::uint64_t id, video::VideoExecutionRoute route,
                    const hal::ExecutionDomainId &domain,
                    hal::SemanticEffectProviderId provider,
                    FakeFactory *factory, std::uint32_t priority = 10) {
  video::VideoAdapterFactoryRegistration registration;
  registration.factory_id = id;
  registration.route = route;
  registration.backend = domain.backend;
  registration.provider = provider;
  registration.adapter_instance_id = domain.adapter_instance_id;
  registration.execution_domain = domain;
  registration.selection_priority = priority;
  registration.factory = factory;
  return registration;
}

bool Configure(video::VideoExecutionCoordinator *coordinator,
               const video::VideoExecutionPlanRequest &request) {
  video::VideoExecutionPrepareEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  return coordinator->Prepare(request, &evidence, &reason) ==
         hal::StatusId::ready;
}

bool TestZeroGpuPassthrough() {
  hal::StaticAdapterRegistry registry;
  video::VideoExecutionCoordinator coordinator(registry);
  auto request = Request(false);
  request.host_input.mutability = hal::Mutability::write_only;
  request.host_output = request.host_input;
  video::VideoExecutionPrepareEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  if (!Require(coordinator.Prepare(request, &evidence, &reason) ==
                   hal::StatusId::passthrough,
               "zero-stage plan must select passthrough") ||
      !Require(evidence.selected_route ==
                   video::VideoExecutionRoute::passthrough,
               "passthrough route must be explicit")) {
    return false;
  }
  std::uint8_t frame[24]{};
  auto result = coordinator.Execute(Binding(frame, frame, 1, true));
  if (!Require(result.status == hal::StatusId::passthrough,
               "passthrough must execute") ||
      !Require(result.transfers.uploads == 0 &&
                   result.transfers.final_readbacks == 0,
               "passthrough must have zero GPU transfers") ||
      !Require(coordinator.CommitOutput(result.completion, true, &reason) ==
                   hal::StatusId::ready,
               "passthrough output must commit")) {
    return false;
  }
  const auto snapshot = coordinator.Snapshot();
  return Require(snapshot.latest_committed_frame.valid,
                 "passthrough writer commit must be recorded") &&
         Require(snapshot.telemetry.active_backend == hal::BackendId::invalid,
                 "passthrough must not attribute a backend");
}

bool TestStrictMismatch() {
  const auto maxine_domain = Domain(hal::BackendId::maxine, 55);
  DummyResident maxine(maxine_domain);
  hal::StaticAdapterRegistry registry;
  if (!RegisterCapability(&registry, &maxine, maxine_domain,
                          hal::SemanticEffectProviderId::maxine_vfx)) {
    return false;
  }
  video::VideoExecutionCoordinator coordinator(registry);
  auto request = Request();
  video::VideoExecutionPrepareEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  const auto status = coordinator.Prepare(request, &evidence, &reason);
  return Require(status == hal::StatusId::blocked,
                 "strict CUDA request must not use Maxine") &&
         Require(reason == hal::ReasonId::strict_backend_unavailable,
                 "strict mismatch must preserve exact reason") &&
         Require(!coordinator.Snapshot().prepared,
                 "strict mismatch must leave no active plan");
}

bool TestExecuteCommitAndWriterFailure() {
  const auto domain = Domain();
  DummyResident resident(domain);
  hal::StaticAdapterRegistry registry;
  if (!RegisterCapability(
          &registry, &resident, domain,
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda)) {
    return false;
  }
  FakeFactory factory(
      domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  video::VideoExecutionCoordinator coordinator(registry);
  hal::ReasonId reason = hal::ReasonId::none;
  if (coordinator.RegisterFactory(
          FactoryRegistration(
              1, video::VideoExecutionRoute::prepared_resident, domain,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              &factory),
          &reason) != hal::StatusId::ready ||
      !Configure(&coordinator, Request())) {
    return false;
  }
  std::uint8_t input[24]{};
  std::uint8_t output[24]{};
  auto first = coordinator.Execute(Binding(input, output, 1));
  if (!Require(first.status == hal::StatusId::executed,
               "prepared executor must run once") ||
      !Require(factory.latest_executor->execute_calls == 1,
               "exactly one executor call is allowed") ||
      !Require(coordinator.CommitOutput(first.completion, false, &reason) ==
                   hal::StatusId::failed,
               "writer failure must reject commit")) {
    return false;
  }
  auto snapshot = coordinator.Snapshot();
  if (!Require(!snapshot.latest_committed_frame.valid,
               "writer failure must not create committed activity") ||
      !Require(snapshot.telemetry.active_backend == hal::BackendId::invalid,
               "writer failure must not attribute active backend")) {
    return false;
  }
  const auto second = coordinator.Execute(Binding(input, output, 2));
  if (!Require(second.status == hal::StatusId::executed,
               "next frame must remain executable") ||
      !Require(coordinator.CommitOutput(second.completion, true, &reason) ==
                   hal::StatusId::ready,
               "successful writer commit must publish activity")) {
    return false;
  }
  snapshot = coordinator.Snapshot();
  return Require(snapshot.latest_committed_frame.valid &&
                     snapshot.latest_committed_frame.capture_sequence == 2,
                 "successful commit identity must be current") &&
         Require(snapshot.telemetry.active_backend == hal::BackendId::cuda,
                 "backend attribution must follow writer commit") &&
         Require(snapshot.telemetry.transfers.uploads == 2 &&
                     snapshot.telemetry.transfers.final_readbacks == 2,
                 "transfer telemetry must include both executions");
}

bool TestInvalidationStaleGenerationAndOldPlanRetention() {
  const auto domain = Domain();
  DummyResident resident(domain);
  hal::StaticAdapterRegistry registry;
  if (!RegisterCapability(
          &registry, &resident, domain,
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda)) {
    return false;
  }
  FakeFactory factory(
      domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  video::VideoExecutionCoordinator coordinator(registry);
  hal::ReasonId reason = hal::ReasonId::none;
  if (coordinator.RegisterFactory(
          FactoryRegistration(
              1, video::VideoExecutionRoute::prepared_resident, domain,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              &factory),
          &reason) != hal::StatusId::ready ||
      !Configure(&coordinator, Request())) {
    return false;
  }
  factory.preflight_status = hal::StatusId::unavailable;
  factory.preflight_reason = hal::ReasonId::resource_exhausted;
  auto same = Request();
  video::VideoExecutionPrepareEvidence evidence;
  if (!Require(coordinator.Prepare(same, &evidence, &reason) ==
                       hal::StatusId::unavailable &&
                   evidence.old_plan_retained,
               "exact same identity may retain old plan")) {
    return false;
  }
  auto changed = same;
  changed.invalidation_key.generations.configuration = 2;
  changed.invalidation_key.canonical_plan_hash = 1002;
  if (!Require(coordinator.Prepare(changed, &evidence, &reason) ==
                       hal::StatusId::unavailable &&
                   !evidence.old_plan_retained,
               "changed generation must retire old plan") ||
      !Require(!coordinator.Snapshot().prepared,
               "stale plan must be invalidated")) {
    return false;
  }
  std::uint8_t input[24]{};
  std::uint8_t output[24]{};
  const auto stale = coordinator.Execute(Binding(input, output, 3));
  return Require(stale.status == hal::StatusId::blocked &&
                     stale.reason == hal::ReasonId::stale_program,
                 "invalidated plan must fail stale");
}

bool TestWrongDomainAndStaleCompletion() {
  const auto run = [](ExecuteMode mode, hal::ReasonId expected) {
    const auto domain = Domain();
    DummyResident resident(domain);
    hal::StaticAdapterRegistry registry;
    if (!RegisterCapability(
            &registry, &resident, domain,
            hal::SemanticEffectProviderId::open_video_onnxruntime_cuda)) {
      return false;
    }
    FakeFactory factory(
        domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
        mode);
    video::VideoExecutionCoordinator coordinator(registry);
    hal::ReasonId reason = hal::ReasonId::none;
    if (coordinator.RegisterFactory(
            FactoryRegistration(
                1, video::VideoExecutionRoute::prepared_resident, domain,
                hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
                &factory),
            &reason) != hal::StatusId::ready ||
        !Configure(&coordinator, Request())) {
      return false;
    }
    std::uint8_t input[24]{};
    std::uint8_t output[24]{};
    const auto frame = coordinator.Execute(Binding(input, output, 1));
    return Require(frame.status == hal::StatusId::failed &&
                       frame.reason == expected,
                   "foreign/stale executor result must fail closed") &&
           Require(!coordinator.Snapshot().frame_pending,
                   "rejected result must not be commit-pending");
  };
  return run(ExecuteMode::wrong_domain,
             hal::ReasonId::wrong_execution_domain) &&
         run(ExecuteMode::stale_program, hal::ReasonId::incomplete_execution);
}

bool TestOptionalFailOpenPreservesOutput() {
  const auto domain = Domain();
  DummyResident resident(domain);
  hal::StaticAdapterRegistry registry;
  if (!RegisterCapability(
          &registry, &resident, domain,
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda)) {
    return false;
  }
  FakeFactory factory(
      domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
      ExecuteMode::optional_fail_open);
  video::VideoExecutionCoordinator coordinator(registry);
  hal::ReasonId reason = hal::ReasonId::none;
  if (coordinator.RegisterFactory(
          FactoryRegistration(
              1, video::VideoExecutionRoute::prepared_resident, domain,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              &factory),
          &reason) != hal::StatusId::ready ||
      !Configure(&coordinator, Request())) {
    return false;
  }
  std::uint8_t input[24]{};
  std::uint8_t output[24]{};
  const auto frame = coordinator.Execute(Binding(input, output, 1));
  if (!Require(frame.status == hal::StatusId::degraded && frame.output_ready,
               "optional failure must preserve valid output") ||
      !Require(coordinator.CommitOutput(frame.completion, true, &reason) ==
                   hal::StatusId::ready,
               "preserved output must remain committable")) {
    return false;
  }
  const auto snapshot = coordinator.Snapshot();
  return Require(snapshot.latest_committed_frame.valid,
                 "degraded output must record its commit") &&
         Require(snapshot.telemetry.active_backend == hal::BackendId::invalid,
                 "failed-open stage must not claim active execution") &&
         Require(snapshot.telemetry.optional_fail_open_stages == 1,
                 "optional degradation must be counted");
}

bool TestDeviceLossAndShutdown() {
  const auto domain = Domain();
  DummyResident resident(domain);
  hal::StaticAdapterRegistry registry;
  if (!RegisterCapability(
          &registry, &resident, domain,
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda)) {
    return false;
  }
  FakeFactory factory(
      domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
      ExecuteMode::device_loss);
  FakeFactory compatibility(
      domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  video::VideoExecutionCoordinator coordinator(registry);
  hal::ReasonId reason = hal::ReasonId::none;
  if (coordinator.RegisterFactory(
          FactoryRegistration(
              1, video::VideoExecutionRoute::prepared_resident, domain,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              &factory),
          &reason) != hal::StatusId::ready ||
      coordinator.RegisterFactory(
          FactoryRegistration(
              2, video::VideoExecutionRoute::legacy_compatibility, domain,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              &compatibility),
          &reason) != hal::StatusId::ready) {
    return false;
  }
  auto request = Request();
  request.migration_policy = video::VideoExecutionMigrationPolicy::
      prepared_then_same_resolution_compatibility;
  if (!Configure(&coordinator, request))
    return false;
  std::uint8_t input[24]{};
  std::uint8_t output[24]{};
  const auto lost = coordinator.Execute(Binding(input, output, 1));
  if (!Require(lost.reason == hal::ReasonId::device_lost &&
                   coordinator.Snapshot().quarantined,
               "device loss must quarantine the plan") ||
      !Require(compatibility.preflight_calls == 0,
               "Execute failure must never trigger compatibility fallback")) {
    return false;
  }
  coordinator.Shutdown();
  video::VideoExecutionPrepareEvidence evidence;
  return Require(coordinator.Snapshot().stopped,
                 "shutdown must publish stopped state") &&
         Require(coordinator.Prepare(Request(), &evidence, &reason) ==
                         hal::StatusId::stopped &&
                     reason == hal::ReasonId::shutdown_requested,
                 "shutdown must be terminal");
}

bool TestCompatibilityConsumesFrozenResolutionWithoutReselection() {
  const auto cuda_domain = Domain();
  const auto maxine_domain = Domain(hal::BackendId::maxine, 55);
  DummyResident cuda(cuda_domain);
  DummyResident maxine(maxine_domain);
  hal::StaticAdapterRegistry registry;
  if (!RegisterCapability(
          &registry, &cuda, cuda_domain,
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda, 10) ||
      !RegisterCapability(&registry, &maxine, maxine_domain,
                          hal::SemanticEffectProviderId::maxine_vfx, 20)) {
    return false;
  }
  FakeFactory prepared(
      cuda_domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  prepared.preflight_status = hal::StatusId::unavailable;
  prepared.preflight_reason = hal::ReasonId::resource_exhausted;
  FakeFactory compatibility(
      cuda_domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  FakeFactory unrelated(maxine_domain,
                        hal::SemanticEffectProviderId::maxine_vfx);
  video::VideoExecutionCoordinator coordinator(registry);
  hal::ReasonId reason = hal::ReasonId::none;
  if (coordinator.RegisterFactory(
          FactoryRegistration(
              1, video::VideoExecutionRoute::prepared_resident, cuda_domain,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              &prepared),
          &reason) != hal::StatusId::ready ||
      coordinator.RegisterFactory(
          FactoryRegistration(
              2, video::VideoExecutionRoute::legacy_compatibility, cuda_domain,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              &compatibility),
          &reason) != hal::StatusId::ready ||
      coordinator.RegisterFactory(
          FactoryRegistration(
              3, video::VideoExecutionRoute::legacy_compatibility,
              maxine_domain, hal::SemanticEffectProviderId::maxine_vfx,
              &unrelated),
          &reason) != hal::StatusId::ready) {
    return false;
  }
  auto request = Request();
  request.migration_policy = video::VideoExecutionMigrationPolicy::
      prepared_then_same_resolution_compatibility;
  video::VideoExecutionPrepareEvidence evidence;
  if (!Require(coordinator.Prepare(request, &evidence, &reason) ==
                   hal::StatusId::ready,
               "explicit migration policy must permit compatibility setup") ||
      !Require(evidence.selected_route ==
                   video::VideoExecutionRoute::legacy_compatibility,
               "coordinator must freeze compatibility route") ||
      !Require(evidence.compatibility_consumed_frozen_resolution,
               "compatibility must consume the original resolution") ||
      !Require(compatibility.observed_resolution.selected_backend ==
                   hal::BackendId::cuda,
               "compatibility must observe frozen CUDA selection") ||
      !Require(unrelated.preflight_calls == 0,
               "unrelated compatibility backend must not be probed")) {
    return false;
  }
  return Require(
      compatibility.observed_plan.resolution.selected_backend ==
              hal::BackendId::cuda &&
          compatibility.observed_plan.resolution.selected_provider ==
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
      "compatibility plan must retain exact backend/provider");
}

bool TestFactorySelectionDeclarationFailsClosed() {
  const auto domain = Domain();
  DummyResident resident(domain);
  hal::StaticAdapterRegistry registry;
  if (!RegisterCapability(
          &registry, &resident, domain,
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda)) {
    return false;
  }
  FakeFactory factory(
      domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  factory.performs_selection = true;
  FakeFactory compatibility(
      domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  video::VideoExecutionCoordinator coordinator(registry);
  hal::ReasonId reason = hal::ReasonId::none;
  if (coordinator.RegisterFactory(
          FactoryRegistration(
              1, video::VideoExecutionRoute::prepared_resident, domain,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              &factory),
          &reason) != hal::StatusId::ready ||
      coordinator.RegisterFactory(
          FactoryRegistration(
              2, video::VideoExecutionRoute::legacy_compatibility, domain,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              &compatibility),
          &reason) != hal::StatusId::ready) {
    return false;
  }
  auto request = Request();
  request.migration_policy = video::VideoExecutionMigrationPolicy::
      prepared_then_same_resolution_compatibility;
  video::VideoExecutionPrepareEvidence evidence;
  return Require(coordinator.Prepare(request, &evidence, &reason) ==
                         hal::StatusId::blocked &&
                     reason == hal::ReasonId::malformed_contract,
                 "factory-side selection must fail closed") &&
         Require(compatibility.preflight_calls == 0,
                 "factory-side selection must never reach compatibility");
}

bool TestPreparedIntegrityFailureCannotReachCompatibility() {
  const auto run = [](bool wrong_domain, bool malformed_program) {
    const auto domain = Domain();
    DummyResident resident(domain);
    hal::StaticAdapterRegistry registry;
    if (!RegisterCapability(
            &registry, &resident, domain,
            hal::SemanticEffectProviderId::open_video_onnxruntime_cuda)) {
      return false;
    }
    FakeFactory prepared(
        domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
    prepared.wrong_preflight_domain = wrong_domain;
    prepared.malformed_prepared_program = malformed_program;
    FakeFactory compatibility(
        domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
    video::VideoExecutionCoordinator coordinator(registry);
    hal::ReasonId reason = hal::ReasonId::none;
    if (coordinator.RegisterFactory(
            FactoryRegistration(
                1, video::VideoExecutionRoute::prepared_resident, domain,
                hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
                &prepared),
            &reason) != hal::StatusId::ready ||
        coordinator.RegisterFactory(
            FactoryRegistration(
                2, video::VideoExecutionRoute::legacy_compatibility, domain,
                hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
                &compatibility),
            &reason) != hal::StatusId::ready) {
      return false;
    }
    auto request = Request();
    request.migration_policy = video::VideoExecutionMigrationPolicy::
        prepared_then_same_resolution_compatibility;
    video::VideoExecutionPrepareEvidence evidence;
    const auto status = coordinator.Prepare(request, &evidence, &reason);
    return Require(status == hal::StatusId::blocked,
                   "prepared integrity failure must remain blocked") &&
           Require(reason == (wrong_domain
                                  ? hal::ReasonId::wrong_execution_domain
                                  : hal::ReasonId::malformed_contract),
                   "prepared integrity reason must be preserved") &&
           Require(compatibility.preflight_calls == 0,
                   "prepared integrity failure must not reach compatibility");
  };
  return run(true, false) && run(false, true);
}

bool TestNoPreparedFactoryMayUseCompatibility() {
  const auto domain = Domain();
  DummyResident resident(domain);
  hal::StaticAdapterRegistry registry;
  if (!RegisterCapability(
          &registry, &resident, domain,
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda)) {
    return false;
  }
  FakeFactory compatibility(
      domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  video::VideoExecutionCoordinator coordinator(registry);
  hal::ReasonId reason = hal::ReasonId::none;
  if (coordinator.RegisterFactory(
          FactoryRegistration(
              2, video::VideoExecutionRoute::legacy_compatibility, domain,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              &compatibility),
          &reason) != hal::StatusId::ready) {
    return false;
  }
  auto request = Request();
  request.migration_policy = video::VideoExecutionMigrationPolicy::
      prepared_then_same_resolution_compatibility;
  video::VideoExecutionPrepareEvidence evidence;
  return Require(coordinator.Prepare(request, &evidence, &reason) ==
                     hal::StatusId::ready,
                 "missing prepared factory may use explicit compatibility") &&
         Require(evidence.selected_route ==
                         video::VideoExecutionRoute::legacy_compatibility &&
                     evidence.compatibility_consumed_frozen_resolution,
                 "missing factory fallback must consume frozen resolution");
}

bool TestCompatibilityRequiresPreparedRollbackEvidence() {
  const auto domain = Domain();
  DummyResident resident(domain);
  hal::StaticAdapterRegistry registry;
  if (!RegisterCapability(
          &registry, &resident, domain,
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda)) {
    return false;
  }
  FakeFactory prepared(
      domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  prepared.prepare_status = hal::StatusId::failed;
  prepared.prepare_reason = hal::ReasonId::resource_exhausted;
  FakeFactory compatibility(
      domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  video::VideoExecutionCoordinator coordinator(registry);
  hal::ReasonId reason = hal::ReasonId::none;
  if (coordinator.RegisterFactory(
          FactoryRegistration(
              1, video::VideoExecutionRoute::prepared_resident, domain,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              &prepared),
          &reason) != hal::StatusId::ready ||
      coordinator.RegisterFactory(
          FactoryRegistration(
              2, video::VideoExecutionRoute::legacy_compatibility, domain,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              &compatibility),
          &reason) != hal::StatusId::ready) {
    return false;
  }
  auto request = Request();
  request.migration_policy = video::VideoExecutionMigrationPolicy::
      prepared_then_same_resolution_compatibility;
  video::VideoExecutionPrepareEvidence evidence;
  return Require(coordinator.Prepare(request, &evidence, &reason) ==
                         hal::StatusId::blocked &&
                     reason == hal::ReasonId::incomplete_execution,
                 "unproven candidate rollback must fail closed") &&
         Require(compatibility.preflight_calls == 0,
                 "compatibility must wait for rollback evidence");
}

bool TestPreparedTransferContractAndFinalCpuTail() {
  const auto run = [](ExecuteMode mode, bool malformed_tail) {
    const auto domain = Domain();
    DummyResident resident(domain);
    hal::StaticAdapterRegistry registry;
    constexpr std::uint64_t kTailScratch = 16;
    if (!RegisterCapability(
            &registry, &resident, domain,
            hal::SemanticEffectProviderId::open_video_onnxruntime_cuda, 10,
            hal::CpuTailId::writer_copy, kTailScratch)) {
      return false;
    }
    FakeFactory factory(
        domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
        mode);
    factory.malformed_prepared_tail = malformed_tail;
    FakeFactory compatibility(
        domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
    video::VideoExecutionCoordinator coordinator(registry);
    hal::ReasonId reason = hal::ReasonId::none;
    if (coordinator.RegisterFactory(
            FactoryRegistration(
                1, video::VideoExecutionRoute::prepared_resident, domain,
                hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
                &factory),
            &reason) != hal::StatusId::ready ||
        coordinator.RegisterFactory(
            FactoryRegistration(
                2, video::VideoExecutionRoute::legacy_compatibility, domain,
                hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
                &compatibility),
            &reason) != hal::StatusId::ready) {
      return false;
    }
    auto request = Request();
    request.cpu_tails[0].id = hal::CpuTailId::writer_copy;
    request.cpu_tails[0].after_stage_id = 1;
    request.cpu_tails[0].scratch_bytes = kTailScratch;
    request.cpu_tails[0].required = true;
    request.cpu_tails[0].placement =
        hal::DeclaredCpuTail::Placement::after_stage;
    request.cpu_tails[0].boundary_index = 1;
    request.cpu_tail_count = 1;
    if (malformed_tail) {
      request.migration_policy = video::VideoExecutionMigrationPolicy::
          prepared_then_same_resolution_compatibility;
      video::VideoExecutionPrepareEvidence evidence;
      return Require(coordinator.Prepare(request, &evidence, &reason) ==
                             hal::StatusId::blocked &&
                         reason == hal::ReasonId::malformed_contract,
                     "malformed prepared CPU tail must fail closed") &&
             Require(
                 compatibility.preflight_calls == 0,
                 "malformed prepared CPU tail must not reach compatibility");
    }
    if (!Configure(&coordinator, request))
      return false;
    std::uint8_t input[24]{};
    std::uint8_t output[24]{};
    const auto frame = coordinator.Execute(Binding(input, output, 1));
    if (mode == ExecuteMode::transfer_mismatch ||
        mode == ExecuteMode::legacy_boundary_mismatch) {
      return Require(frame.status == hal::StatusId::failed &&
                         frame.reason ==
                             hal::ReasonId::transfer_contract_violated,
                     "any prepared transfer-counter mismatch must fail");
    }
    return Require(frame.status == hal::StatusId::executed,
                   "declared final CPU-tail frame must execute") &&
           Require(frame.transfers.uploads == 1 &&
                       frame.transfers.final_readbacks == 1 &&
                       frame.transfers.intermediate_readbacks == 0 &&
                       frame.transfers.intermediate_uploads == 0 &&
                       frame.transfers.synchronizations == 1 &&
                       frame.transfers.cpu_tail_boundaries == 1,
                   "all six prepared transfer counters must match the plan");
  };
  return run(ExecuteMode::success, false) &&
         run(ExecuteMode::transfer_mismatch, false) &&
         run(ExecuteMode::legacy_boundary_mismatch, false) &&
         run(ExecuteMode::success, true);
}

bool TestTransferContractValidation() {
  const auto run = [](bool malformed_boundary, bool malformed_version,
                      std::uint64_t minimum, std::uint64_t maximum,
                      const char *message) {
    const auto domain = Domain();
    DummyResident resident(domain);
    hal::StaticAdapterRegistry registry;
    if (!RegisterCapability(
            &registry, &resident, domain,
            hal::SemanticEffectProviderId::open_video_onnxruntime_cuda)) {
      return false;
    }
    FakeFactory factory(
        domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
    factory.malformed_transfer_boundary = malformed_boundary;
    factory.malformed_transfer_version = malformed_version;
    factory.minimum_synchronizations = minimum;
    factory.maximum_synchronizations = maximum;
    video::VideoExecutionCoordinator coordinator(registry);
    hal::ReasonId reason = hal::ReasonId::none;
    if (coordinator.RegisterFactory(
            FactoryRegistration(
                1, video::VideoExecutionRoute::prepared_resident, domain,
                hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
                &factory),
            &reason) != hal::StatusId::ready) {
      return false;
    }
    video::VideoExecutionPrepareEvidence evidence;
    return Require(coordinator.Prepare(Request(), &evidence, &reason) ==
                           hal::StatusId::blocked &&
                       reason == hal::ReasonId::malformed_contract,
                   message) &&
           Require(factory.prepare_calls == 0,
                   "invalid transfer evidence must fail before Prepare");
  };

  bool ok = run(true, false, 1, 1, "boundary mismatch must fail preflight");
  ok &= run(false, true, 1, 1,
            "foreign transfer contract version must fail preflight");
  ok &= run(false, false, 2, 1,
            "inverted synchronization range must fail preflight");
  ok &= run(false, false, 0, 0, "device synchronization range must be nonzero");
  ok &= run(false, false, 1, video::kMaxPerFrameSynchronizationsV1 + 1,
            "unbounded synchronization range must fail preflight");

  const auto domain = Domain();
  DummyResident resident(domain);
  hal::StaticAdapterRegistry registry;
  if (!RegisterCapability(
          &registry, &resident, domain,
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda)) {
    return false;
  }
  FakeFactory compatibility(
      domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  compatibility.malformed_transfer_boundary = true;
  video::VideoExecutionCoordinator coordinator(registry);
  hal::ReasonId reason = hal::ReasonId::none;
  if (coordinator.RegisterFactory(
          FactoryRegistration(
              2, video::VideoExecutionRoute::legacy_compatibility, domain,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              &compatibility),
          &reason) != hal::StatusId::ready) {
    return false;
  }
  auto request = Request();
  request.migration_policy = video::VideoExecutionMigrationPolicy::
      prepared_then_same_resolution_compatibility;
  video::VideoExecutionPrepareEvidence evidence;
  ok &= Require(coordinator.Prepare(request, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::malformed_contract,
                "compatibility transfer crossings must be bounded");
  return ok;
}

bool TestVariableTransferBoundaryContract() {
  const auto run = [](ExecuteMode mode, int malformed,
                      hal::StatusId expected_status) {
    const auto domain = Domain();
    DummyResident resident(domain);
    hal::StaticAdapterRegistry registry;
    if (!RegisterCapability(
            &registry, &resident, domain,
            hal::SemanticEffectProviderId::open_video_onnxruntime_cuda))
      return false;
    FakeFactory factory(
        domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
        mode);
    factory.variable_boundaries = true;
    factory.variable_bounds.maximum_intermediate_readbacks = 12;
    factory.variable_bounds.maximum_intermediate_uploads = 1;
    factory.variable_bounds.maximum_cpu_tail_boundaries = 1;
    if (malformed == 1)
      factory.variable_bounds.minimum_intermediate_readbacks = 13;
    if (malformed == 2)
      factory.variable_bounds.maximum_intermediate_readbacks =
          hal::kMaxExecutionBoundaries + 1;
    video::VideoExecutionCoordinator coordinator(registry);
    hal::ReasonId reason = hal::ReasonId::none;
    if (coordinator.RegisterFactory(
            FactoryRegistration(
                1, video::VideoExecutionRoute::prepared_resident, domain,
                hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
                &factory),
            &reason) != hal::StatusId::ready)
      return false;
    video::VideoExecutionPrepareEvidence evidence;
    const auto prepared = coordinator.Prepare(Request(), &evidence, &reason);
    if (malformed)
      return Require(prepared == hal::StatusId::blocked &&
                         reason == hal::ReasonId::malformed_contract &&
                         factory.prepare_calls == 0,
                     "malformed variable bounds reached Prepare");
    if (prepared != hal::StatusId::ready)
      return false;
    std::uint8_t input[24]{}, output[24]{};
    const auto frame = coordinator.Execute(Binding(input, output, 1));
    return Require(frame.status == expected_status,
                   expected_status == hal::StatusId::failed
                       ? "out-of-range variable transfer was accepted"
                       : "in-range variable transfer was rejected") &&
           Require(expected_status != hal::StatusId::failed ||
                       frame.reason ==
                           hal::ReasonId::transfer_contract_violated,
                   "out-of-range variable transfer used wrong reason");
  };
  bool ok = run(ExecuteMode::success, 0, hal::StatusId::executed);
  ok &= run(ExecuteMode::optional_fail_open, 0, hal::StatusId::degraded);
  ok &= run(ExecuteMode::variable_maximum, 0,
            hal::StatusId::executed);
  ok &= run(ExecuteMode::variable_outside, 0, hal::StatusId::failed);
  ok &= run(ExecuteMode::success, 1, hal::StatusId::blocked);
  ok &= run(ExecuteMode::success, 2, hal::StatusId::blocked);

  return ok;
}

bool TestNoAllocationDuringExecute() {
  const auto domain = Domain();
  DummyResident resident(domain);
  hal::StaticAdapterRegistry registry;
  if (!RegisterCapability(
          &registry, &resident, domain,
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda)) {
    return false;
  }
  FakeFactory factory(
      domain, hal::SemanticEffectProviderId::open_video_onnxruntime_cuda);
  video::VideoExecutionCoordinator coordinator(registry);
  hal::ReasonId reason = hal::ReasonId::none;
  if (coordinator.RegisterFactory(
          FactoryRegistration(
              1, video::VideoExecutionRoute::prepared_resident, domain,
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda,
              &factory),
          &reason) != hal::StatusId::ready ||
      !Configure(&coordinator, Request())) {
    return false;
  }
  std::uint8_t input[24]{};
  std::uint8_t output[24]{};
  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations.store(true, std::memory_order_release);
  const auto frame = coordinator.Execute(Binding(input, output, 1));
  g_count_allocations.store(false, std::memory_order_release);
  return Require(frame.status == hal::StatusId::executed,
                 "allocation measurement frame must execute") &&
         Require(g_allocations.load(std::memory_order_relaxed) == 0,
                 "Execute must not allocate");
}

} // namespace

void *operator new(std::size_t size) {
  if (g_count_allocations.load(std::memory_order_acquire))
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *pointer = std::malloc(size))
    return pointer;
  throw std::bad_alloc();
}

void *operator new[](std::size_t size) {
  if (g_count_allocations.load(std::memory_order_acquire))
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

int main() {
  bool ok = true;
  ok &= TestZeroGpuPassthrough();
  ok &= TestStrictMismatch();
  ok &= TestExecuteCommitAndWriterFailure();
  ok &= TestInvalidationStaleGenerationAndOldPlanRetention();
  ok &= TestWrongDomainAndStaleCompletion();
  ok &= TestOptionalFailOpenPreservesOutput();
  ok &= TestDeviceLossAndShutdown();
  ok &= TestCompatibilityConsumesFrozenResolutionWithoutReselection();
  ok &= TestFactorySelectionDeclarationFailsClosed();
  ok &= TestPreparedIntegrityFailureCannotReachCompatibility();
  ok &= TestNoPreparedFactoryMayUseCompatibility();
  ok &= TestCompatibilityRequiresPreparedRollbackEvidence();
  ok &= TestPreparedTransferContractAndFinalCpuTail();
  ok &= TestTransferContractValidation();
  ok &= TestVariableTransferBoundaryContract();
  ok &= TestNoAllocationDuringExecute();
  if (!ok)
    return 1;
  std::cout << "video execution coordinator tests passed\n";
  return 0;
}
