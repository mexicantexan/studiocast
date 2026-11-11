#include "core/video/production_video_backend_manager.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>

namespace {

namespace hal = studiocast::hal;
namespace video = studiocast::video;

constexpr std::uint64_t kAdapterId = 0x42420001;
constexpr std::uint64_t kFactoryId = 0x42420002;
constexpr std::size_t kFrameBytes = 24;

bool Require(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

hal::ExecutionDomainId Domain() {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::maxine;
  domain.api_family = hal::ApiFamily::maxine_nvcv;
  domain.physical_device.vendor_id = 0x1234;
  domain.physical_device.device_id = 0x5678;
  domain.physical_device.stable_id[0] = 0x42;
  domain.logical_context_generation = 7;
  domain.allocator_family = hal::AllocatorFamily::maxine_nvcv;
  domain.memory_handle_family = hal::MemoryHandleFamily::nvcv_image;
  domain.queue_or_stream_identity = 0x99;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::maxine_bound_cuda_stream;
  domain.adapter_instance_id = kAdapterId;
  return domain;
}

hal::ImageDescriptor Image(hal::Mutability mutability) {
  hal::ImageDescriptor image;
  image.width = 4;
  image.height = 2;
  image.plane_count = 1;
  image.format = hal::PixelFormat::rgb_u8;
  image.channel_order = hal::ChannelOrder::rgb;
  image.color_space = hal::ColorSpace::srgb;
  image.color_range = hal::ColorRange::full;
  image.alpha_meaning = hal::AlphaMeaning::none;
  image.orientation = hal::Orientation::top_left;
  image.host_coherence = hal::HostCoherence::coherent;
  image.mutability = mutability;
  image.aliasing = hal::AliasingPermission::forbidden;
  image.base_alignment_bytes = 1;
  image.planes[0].row_stride_bytes = 12;
  image.planes[0].size_bytes = kFrameBytes;
  image.planes[0].width = 4;
  image.planes[0].height = 2;
  image.planes[0].alignment_bytes = 1;
  return image;
}

video::VideoExecutionPlanRequest Plan() {
  video::VideoExecutionPlanRequest plan;
  plan.plan_id = 101;
  plan.intent.backend.mode = hal::BackendRequestMode::explicit_backend;
  plan.intent.backend.backend = hal::BackendId::maxine;
  plan.intent.backend.strict = true;
  plan.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::cuda_family;
  plan.intent.requested_provider =
      hal::SemanticEffectProviderId::maxine_vfx;
  plan.intent.strict_provider_selection = true;
  plan.intent.stages[0].intent.stage_id = 1;
  plan.intent.stages[0].intent.semantic =
      hal::SemanticEffectId::virtual_background_blur;
  plan.intent.stages[0].intent.failure_action =
      hal::FailureAction::fail_open_preserve_input;
  plan.intent.stages[0].parameters[0] = {
      video::CanonicalVideoParameterId::strength,
      video::CanonicalVideoParameterKind::signed_integer, 8};
  plan.intent.stages[0].parameters[1].id =
      video::CanonicalVideoParameterId::remove_color_rgb;
  plan.intent.stages[0].parameters[1].kind =
      video::CanonicalVideoParameterKind::unsigned_integer;
  plan.intent.stages[0].parameters[2].id =
      video::CanonicalVideoParameterId::greenscreen_mode;
  plan.intent.stages[0].parameters[2].kind =
      video::CanonicalVideoParameterKind::unsigned_integer;
  plan.intent.stages[0].parameters[3].id =
      video::CanonicalVideoParameterId::greenscreen_temporal;
  plan.intent.stages[0].parameters[3].kind =
      video::CanonicalVideoParameterKind::boolean;
  plan.intent.stages[0].parameters[3].boolean_value = true;
  plan.intent.stages[0].parameter_count = 4;
  plan.intent.stage_count = 1;
  plan.temporal.temporal = true;
  plan.temporal.requires_contiguous_capture_sequence = true;
  plan.temporal.history_frames = 1;
  plan.temporal.frame_drop_behavior = hal::FrameDropBehavior::reset_temporal_state;
  plan.host_input = Image(hal::Mutability::read_only);
  plan.host_output = Image(hal::Mutability::write_only);
  plan.invalidation_key.generations = {1, 2, 3, 4, 5, 7};
  plan.invalidation_key.canonical_plan_hash = 11;
  plan.invalidation_key.execution_domain_hash = 0;
  plan.budget.persistent_bytes = 128;
  plan.budget.scratch_bytes = 64;
  plan.budget.upload_staging_bytes = kFrameBytes;
  plan.budget.readback_staging_bytes = kFrameBytes;
  plan.budget.resource_count = 4;
  plan.budget.in_flight_slots = 1;
  plan.setup_failure_policy = hal::SetupFailurePolicy::retain_old_program;
  plan.migration_policy = video::VideoExecutionMigrationPolicy::prepared_only;
  return plan;
}

struct WaitContext {
  bool stop = false;
  std::uint64_t now = 1;
};

bool StopRequested(void *context) noexcept {
  return static_cast<WaitContext *>(context)->stop;
}

std::uint64_t NowNanoseconds(void *context) noexcept {
  return static_cast<WaitContext *>(context)->now;
}

void PollWait(void *context, std::uint64_t wait_nanoseconds) noexcept {
  static_cast<WaitContext *>(context)->now += wait_nanoseconds;
}

hal::ModelArtifactIdentity ModelIdentity(std::uint8_t value = 0x31) {
  hal::ModelArtifactIdentity model;
  model.model_id_hash[0] = value;
  model.artifact_set_hash[0] = static_cast<std::uint8_t>(value + 1);
  model.artifact_generation = 3;
  return model;
}

video::ProductionVideoBackendSetupRequest ResidentRequest() {
  static WaitContext wait_context;
  video::ProductionVideoBackendSetupRequest request;
  request.kind = video::ProductionVideoBackendSetupKind::resident_island;
  request.resident.effects.virtual_background.mode =
      video::effects::VirtualBackgroundMode::blur;
  request.resident.plan = Plan();
  request.resident.wait.timeout_nanoseconds = 1000;
  request.resident.wait.poll_interval_nanoseconds = 10;
  request.resident.wait.maximum_poll_count = 100;
  request.resident.wait.stop_requested = &StopRequested;
  request.resident.wait.cancel_requested = &StopRequested;
  request.resident.wait.now_nanoseconds = &NowNanoseconds;
  request.resident.wait.poll_wait = &PollWait;
  request.resident.wait.callback_context = &wait_context;
  request.resident.wait.callback_context_generation = 1;
  request.resident.wait.callback_context_lifetime =
      hal::ResourceLifetime::adapter_context;
  return request;
}

hal::HostFrameBinding Binding(std::array<std::byte, kFrameBytes> *input,
                              std::array<std::byte, kFrameBytes> *output,
                              std::uint64_t sequence) {
  hal::HostFrameBinding binding;
  binding.input_descriptor = Image(hal::Mutability::read_only);
  binding.output_descriptor = Image(hal::Mutability::write_only);
  binding.input_planes[0].read_data = input->data();
  binding.input_planes[0].span_bytes = input->size();
  binding.input_planes[0].row_stride_bytes = 12;
  binding.output_planes[0].write_data = output->data();
  binding.output_planes[0].span_bytes = output->size();
  binding.output_planes[0].row_stride_bytes = 12;
  binding.input_plane_count = 1;
  binding.output_plane_count = 1;
  binding.capture_sequence = sequence;
  binding.ownership = hal::OwnershipModel::borrowed_until_completion;
  binding.lifetime = hal::ResourceLifetime::frame_borrowed;
  return binding;
}

struct Counters {
  std::uint64_t source_calls = 0;
  std::uint64_t execute_calls = 0;
  std::uint64_t commit_calls = 0;
  std::uint64_t owner_shutdown_calls = 0;
  std::uint64_t owner_destructor_calls = 0;
  std::uint64_t pending_shutdown_attempts = 0;
  hal::StatusId reset_status = hal::StatusId::ready;
};

class FakeResidentAdapter final : public hal::IResidentAdapter {
public:
  std::uint64_t adapter_instance_id() const noexcept override {
    return kAdapterId;
  }
  hal::BackendId backend_id() const noexcept override {
    return hal::BackendId::maxine;
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
};

class FakeExecutor final : public video::IVideoExecutionExecutor {
public:
  FakeExecutor(video::FrozenVideoExecutionPlan plan, Counters *counters)
      : plan_(std::move(plan)), counters_(counters) {
    snapshot_.status = hal::StatusId::ready;
    snapshot_.readiness.status = hal::StatusId::ready;
    snapshot_.readiness.requested_backend = hal::BackendId::maxine;
    snapshot_.readiness.selected_backend = hal::BackendId::maxine;
    snapshot_.readiness.provider =
        hal::SemanticEffectProviderId::maxine_vfx;
    snapshot_.readiness.execution_domain = Domain();
    snapshot_.readiness.ready_stage_mask = 1;
    snapshot_.readiness.available = true;
    snapshot_.execution_domain = Domain();
    snapshot_.plan_id = plan_.plan_id;
    snapshot_.prepared = true;
  }

  video::VideoExecutionFrameResult
  Execute(const hal::HostFrameBinding &binding) noexcept override {
    ++counters_->execute_calls;
    video::VideoExecutionFrameResult result;
    std::memcpy(binding.output_planes[0].write_data,
                binding.input_planes[0].read_data, kFrameBytes);
    result.status = hal::StatusId::executed;
    result.completion = {next_token_++, plan_.plan_id, kAdapterId,
                         binding.capture_sequence, true};
    result.output.descriptor = binding.output_descriptor;
    result.output.planes = binding.output_planes;
    result.output.plane_count = 1;
    result.output.capture_sequence = binding.capture_sequence;
    result.output.output_ready = true;
    result.execution_domain = Domain();
    result.stages[0].stage_id = 1;
    result.stages[0].semantic =
        hal::SemanticEffectId::virtual_background_blur;
    result.stages[0].status = hal::StageExecutionStatus::succeeded;
    result.stages[0].valid_output_preserved = true;
    result.stage_count = 1;
    result.transfers.uploads = 1;
    result.transfers.final_readbacks = 1;
    result.transfers.synchronizations = 1;
    result.frame_elapsed_ns = 100;
    result.output_ready = true;
    pending_ = result.completion;
    return result;
  }

  hal::StatusId CommitOutput(const hal::CompletionToken &completion,
                             bool writer_succeeded,
                             hal::ReasonId *reason) noexcept override {
    ++counters_->commit_calls;
    if (completion.token_id == 0 ||
        completion.token_id != pending_.token_id) {
      if (reason != nullptr)
        *reason = hal::ReasonId::stale_program;
      return hal::StatusId::stale;
    }
    pending_ = {};
    if (reason != nullptr)
      *reason = writer_succeeded ? hal::ReasonId::none
                                 : hal::ReasonId::output_write_failed;
    return writer_succeeded ? hal::StatusId::ready : hal::StatusId::failed;
  }

  void Invalidate(hal::ReasonId) noexcept override {
    snapshot_.prepared = false;
  }
  hal::StatusId ResetTemporal(hal::ReasonId) noexcept override {
    return counters_->reset_status;
  }
  void Shutdown() noexcept override { snapshot_.stopped = true; }
  video::VideoExecutionAdapterSnapshot Snapshot() const noexcept override {
    return snapshot_;
  }

private:
  video::FrozenVideoExecutionPlan plan_{};
  Counters *counters_ = nullptr;
  video::VideoExecutionAdapterSnapshot snapshot_{};
  hal::CompletionToken pending_{};
  std::uint64_t next_token_ = 1;
};

class FakeFactory final : public video::IVideoAdapterFactory {
public:
  explicit FakeFactory(Counters *counters) : counters_(counters) {}

  video::VideoAdapterFactoryPreflight
  Preflight(const video::VideoExecutionPlanRequest &request,
            const hal::ResolutionResult &,
            const hal::DomainPlan &domain_plan,
            video::VideoExecutionRoute route) noexcept override {
    video::VideoAdapterFactoryPreflight result;
    result.status = hal::StatusId::ready;
    result.backend = hal::BackendId::maxine;
    result.provider =
        hal::SemanticEffectProviderId::maxine_vfx;
    result.execution_domain = Domain();
    result.readiness.status = hal::StatusId::ready;
    result.readiness.requested_backend = hal::BackendId::maxine;
    result.readiness.selected_backend = hal::BackendId::maxine;
    result.readiness.provider = result.provider;
    result.readiness.execution_domain = Domain();
    result.readiness.ready_stage_mask = 1;
    result.readiness.available = true;
    result.cached_evidence_only = true;
    result.accepts_frozen_resolution = true;
    result.performs_selection = false;
    result.per_frame_transfers =
        video::BuildPerFrameTransferContract(domain_plan, 1, 1);
    if (route != video::VideoExecutionRoute::prepared_resident)
      return result;

    auto &program = result.prepared_program;
    program.program_id = request.plan_id;
    program.owner_adapter_instance_id = kAdapterId;
    program.execution_domain = Domain();
    program.host_input = request.host_input;
    program.host_output = request.host_output;
    program.invalidation_key = request.invalidation_key;
    program.temporal = request.temporal;
    program.budget = request.budget;
    program.setup_failure_policy = request.setup_failure_policy;
    program.requested_backend = hal::BackendId::maxine;
    program.strict_backend_selection = request.intent.backend.strict;
    program.requested_provider =
        request.intent.requested_provider == hal::SemanticEffectProviderId::invalid
            ? result.provider
            : request.intent.requested_provider;
    program.strict_provider_selection = request.intent.strict_provider_selection;
    program.requested_execution_domain = Domain();
    program.strict_execution_domain = true;
    program.requires_upload = true;
    program.requires_final_readback = true;
    program.cpu_tail_count = request.cpu_tail_count;
    for (std::size_t index = 0; index < request.cpu_tail_count; ++index)
      program.cpu_tails[index] = request.cpu_tails[index];
    program.stage_count = 1;
    auto &stage = program.stages[0];
    stage.stage_id = request.intent.stages[0].intent.stage_id;
    stage.semantic = hal::SemanticEffectId::virtual_background_blur;
    stage.model = request.intent.stages[0].model;
    stage.provider = result.provider;
    stage.backend = hal::BackendId::maxine;
    stage.execution_domain = Domain();
    stage.input.kind = hal::StageIoKind::image;
    stage.input.image = request.host_input;
    stage.input.image.host_coherence = hal::HostCoherence::not_host_visible;
    stage.output.kind = hal::StageIoKind::image;
    stage.output.image = request.host_output;
    stage.output.image.host_coherence = hal::HostCoherence::not_host_visible;
    stage.output.image.mutability = hal::Mutability::read_write;
    stage.failure_action = hal::FailureAction::fail_open_preserve_input;
    stage.circuit_breaker.consecutive_failure_limit = 1;
    stage.circuit_breaker.cooldown_frames = 2;
    stage.circuit_breaker.action = hal::CircuitBreakerAction::disable_stage;
    return result;
  }

  std::unique_ptr<video::IVideoExecutionExecutor>
  Prepare(const video::FrozenVideoExecutionPlan &plan,
          video::VideoAdapterPrepareEvidence *evidence) noexcept override {
    evidence->status = hal::StatusId::ready;
    return std::make_unique<FakeExecutor>(plan, counters_);
  }

private:
  Counters *counters_ = nullptr;
};

class FakeLeaseOwner final : public video::IProductionVideoSetupLeaseOwner {
public:
  explicit FakeLeaseOwner(Counters *counters)
      : factory(counters), counters_(counters) {}
  ~FakeLeaseOwner() override { ++counters_->owner_destructor_calls; }

  hal::IResidentAdapter *resident_adapter() const noexcept override {
    return const_cast<FakeResidentAdapter *>(&adapter);
  }
  video::IVideoAdapterFactory *video_adapter_factory() const noexcept override {
    return const_cast<FakeFactory *>(&factory);
  }
  void Shutdown() noexcept override { ++counters_->owner_shutdown_calls; }
  video::ProductionVideoSetupLeaseShutdownResult
  RequestShutdown() noexcept override {
    Shutdown();
    video::ProductionVideoSetupLeaseShutdownResult result;
    result.shutdown_requested = true;
    result.shutdown_attempts = counters_->owner_shutdown_calls;
    if (counters_->owner_shutdown_calls <=
        counters_->pending_shutdown_attempts) {
      result.status = hal::StatusId::degraded;
      result.reason = hal::ReasonId::synchronization_failed;
      result.pending = true;
      return result;
    }
    result.status = hal::StatusId::stopped;
    result.reason = hal::ReasonId::shutdown_requested;
    result.quiescent = true;
    result.driver_context_release_permitted = true;
    return result;
  }

  FakeResidentAdapter adapter;
  FakeFactory factory;

private:
  Counters *counters_ = nullptr;
};

hal::AdapterRegistration AdapterRegistration(FakeLeaseOwner *owner) {
  hal::AdapterRegistration registration;
  registration.adapter_instance_id = kAdapterId;
  registration.backend = hal::BackendId::maxine;
  registration.execution_domain = Domain();
  registration.selection_priority = 100;
  registration.resident_adapter = owner != nullptr ? &owner->adapter : nullptr;
  auto &capability = registration.semantic_capabilities[0];
  capability.semantic = hal::SemanticEffectId::virtual_background_blur;
  capability.provider =
      hal::SemanticEffectProviderId::maxine_vfx;
  capability.status = hal::StatusId::ready;
  capability.live_stage = true;
  capability.production_eligible = true;
  capability.exact_evidence = true;
  capability.evidence[0].code =
      hal::CapabilityEvidenceCode::model_artifact_identity_valid;
  capability.evidence[0].status = hal::StatusId::ready;
  capability.evidence[0].semantic =
      hal::SemanticEffectId::virtual_background_blur;
  capability.evidence[0].backend = hal::BackendId::maxine;
  capability.evidence[0].satisfied = true;
  capability.evidence_count = 1;
  registration.semantic_capability_count = 1;
  return registration;
}

video::ProductionVideoSetupLeaseSnapshot LeaseSnapshot(
    FakeLeaseOwner *owner) {
  video::ProductionVideoSetupLeaseSnapshot snapshot;
  snapshot.state = video::ProductionVideoSetupLeaseState::live;
  snapshot.status = hal::StatusId::ready;
  snapshot.backend = hal::BackendId::maxine;
  snapshot.execution_domain = Domain();
  snapshot.generations = {1, 2, 3, 4, 5, 7};
  snapshot.reserved_budget = Plan().budget;
  snapshot.providers[0] =
      hal::SemanticEffectProviderId::maxine_vfx;
  snapshot.provider_count = 1;
  snapshot.artifacts[0].semantic =
      hal::SemanticEffectId::virtual_background_blur;
  snapshot.artifacts[0].provider =
      hal::SemanticEffectProviderId::maxine_vfx;
  snapshot.artifacts[0].artifact = ModelIdentity();
  snapshot.artifact_count = 1;
  snapshot.adapter = AdapterRegistration(owner);
  snapshot.factory.factory_id = kFactoryId;
  snapshot.factory.route = video::VideoExecutionRoute::prepared_resident;
  snapshot.factory.backend = hal::BackendId::maxine;
  snapshot.factory.provider =
      hal::SemanticEffectProviderId::maxine_vfx;
  snapshot.factory.adapter_instance_id = kAdapterId;
  snapshot.factory.execution_domain = Domain();
  snapshot.factory.selection_priority = 100;
  snapshot.factory.factory = owner != nullptr ? &owner->factory : nullptr;
  snapshot.readiness.status = hal::StatusId::ready;
  snapshot.readiness.requested_backend = hal::BackendId::maxine;
  snapshot.readiness.selected_backend = hal::BackendId::maxine;
  snapshot.readiness.provider =
      hal::SemanticEffectProviderId::maxine_vfx;
  snapshot.readiness.execution_domain = Domain();
  snapshot.readiness.ready_stage_mask = 1;
  snapshot.readiness.available = true;
  return snapshot;
}

enum class SourceMode {
  ready,
  maxine_temporal_refinement,
  erased_temporal,
  excessive_temporal_history,
  optional_passthrough,
  malformed_optional_passthrough,
  nondeterministic_optional_passthrough_hash,
  malformed_returned_plan,
  foreign_returned_domain,
  single_domain_compatibility,
  single_domain_fabricated_boundary,
  single_domain_fabricated_transfer,
};

enum class SourceTailMode {
  preserve,
  writer_copy,
  output_format_conversion,
  provider_postprocess,
  no_tail,
  wrong_tail,
  altered_boundary,
};

class FakeSetupSource final : public video::IProductionVideoBackendSetupSource {
public:
  explicit FakeSetupSource(Counters *counters) : counters_(counters) {}

  video::ProductionVideoResidentSetupSourceResult
  AcquireResidentSetup(
      const video::ProductionVideoResidentSetup &request) noexcept override {
    ++counters_->source_calls;
    video::ProductionVideoResidentSetupSourceResult result;
    result.plan = request.plan;
    result.canonical_effects = request.effects;
    if (mode == SourceMode::maxine_temporal_refinement ||
        mode == SourceMode::excessive_temporal_history) {
      result.plan.temporal.temporal = true;
      result.plan.temporal.requires_contiguous_capture_sequence = true;
      result.plan.temporal.history_frames =
          mode == SourceMode::excessive_temporal_history ? 5 : 1;
      result.plan.temporal.frame_drop_behavior =
          hal::FrameDropBehavior::reset_temporal_state;
      result.plan.temporal.discontinuity_action =
          hal::TemporalDiscontinuityAction::reset_and_continue;
    }
    if (mode == SourceMode::erased_temporal)
      result.plan.temporal = {};

    if (mode == SourceMode::optional_passthrough ||
        mode == SourceMode::malformed_optional_passthrough ||
        mode == SourceMode::nondeterministic_optional_passthrough_hash) {
      auto domain = Domain();
      domain.backend = hal::BackendId::cuda;
      domain.api_family = hal::ApiFamily::cuda_driver;
      domain.logical_context_generation =
          request.plan.invalidation_key.generations.context;
      domain.allocator_family = hal::AllocatorFamily::cuda_driver;
      domain.memory_handle_family =
          hal::MemoryHandleFamily::cuda_device_pointer;
      domain.synchronization_protocol =
          hal::SynchronizationProtocol::cuda_stream_ordered;
      auto &registration = result.resolution_registration;
      registration.adapter_instance_id = domain.adapter_instance_id;
      registration.backend = domain.backend;
      registration.execution_domain = domain;
      registration.selection_priority = 100;
      auto &capability = registration.semantic_capabilities[0];
      capability.semantic = request.plan.intent.stages[0].intent.semantic;
      capability.provider =
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
      capability.status = hal::StatusId::unavailable;
      capability.reason = hal::ReasonId::capability_evidence_incomplete;
      registration.semantic_capability_count = 1;
      result.plan.invalidation_key.execution_domain_hash =
          video::kProductionVideoNeutralPassthroughExecutionHash;
      if (mode == SourceMode::malformed_optional_passthrough)
        registration.adapter_instance_id = 0;
      if (mode == SourceMode::nondeterministic_optional_passthrough_hash)
        result.plan.invalidation_key.execution_domain_hash +=
            counters_->source_calls;
      result.status = hal::StatusId::degraded;
      result.reason = hal::ReasonId::capability_evidence_incomplete;
      result.readiness.status = hal::StatusId::degraded;
      result.readiness.reason = result.reason;
      result.readiness.requested_backend = hal::BackendId::cuda;
      result.readiness.available = true;
      result.readiness.degraded = true;
      result.setup_completed_outside_frame_loop = true;
      return result;
    }
    if (result.plan.invalidation_key.generations.model == 0)
      result.plan.invalidation_key.generations.model = 3;
    if (result.plan.invalidation_key.generations.provider == 0)
      result.plan.invalidation_key.generations.provider = 4;
    if (result.plan.invalidation_key.generations.device == 0)
      result.plan.invalidation_key.generations.device = 5;
    if (result.plan.invalidation_key.generations.context == 0)
      result.plan.invalidation_key.generations.context = 7;
    result.plan.invalidation_key.model_artifact_hash = 0x3132;
    result.plan.intent.stages[0].model = ModelIdentity();
    result.plan.budget = Plan().budget;
    result.plan.invalidation_key.execution_domain_hash =
        video::ProductionVideoExecutionDomainHash(Domain());
    if (tail_mode == SourceTailMode::no_tail) {
      result.plan.cpu_tails = {};
      result.plan.cpu_tail_count = 0;
    } else if (result.plan.cpu_tail_count == 1) {
      auto &tail = result.plan.cpu_tails[0];
      switch (tail_mode) {
      case SourceTailMode::preserve:
        break;
      case SourceTailMode::writer_copy:
        tail.id = hal::CpuTailId::writer_copy;
        break;
      case SourceTailMode::output_format_conversion:
        tail.id = hal::CpuTailId::output_format_conversion;
        break;
      case SourceTailMode::provider_postprocess:
        tail.id = hal::CpuTailId::provider_postprocess;
        break;
      case SourceTailMode::wrong_tail:
        tail.id = hal::CpuTailId::auto_frame_tracking;
        break;
      case SourceTailMode::altered_boundary:
        tail.id = hal::CpuTailId::writer_copy;
        --tail.boundary_index;
        break;
      case SourceTailMode::no_tail:
        break;
      }
    }
    if (mode == SourceMode::malformed_returned_plan)
      ++result.plan.plan_id;
    if (mode == SourceMode::foreign_returned_domain) {
      auto foreign = Domain();
      ++foreign.queue_or_stream_identity;
      result.plan.invalidation_key.execution_domain_hash =
          video::ProductionVideoExecutionDomainHash(foreign);
    }

    auto &selection = result.model_selections[0];
    selection.semantic = hal::SemanticEffectId::virtual_background_blur;
    selection.requested_model_id = request.effects.virtual_background.model_id;
    selection.selected_model_id =
        request.effects.virtual_background.model_id.empty()
            ? "default_matting_model"
            : "substituted_matting_model";
    selection.artifact = ModelIdentity();
    selection.default_selected =
        request.effects.virtual_background.model_id.empty();
    result.model_selection_count = 1;

    if (mode == SourceMode::single_domain_compatibility ||
        mode == SourceMode::single_domain_fabricated_boundary ||
        mode == SourceMode::single_domain_fabricated_transfer) {
      video::FrozenVideoExecutionPlan frozen;
      frozen.plan_id = result.plan.plan_id;
      result.execution_kind =
          video::ProductionVideoResidentSetupSourceResult::ExecutionKind::
              typed_compatibility;
      result.compatibility_executor =
          std::make_unique<FakeExecutor>(std::move(frozen), counters_);
      auto &topology = result.compatibility_topology;
      topology.domain_count = 1;
      topology.domains[0].domain = Domain();
      topology.domains[0].backend = hal::BackendId::maxine;
      topology.domains[0].provider =
          hal::SemanticEffectProviderId::maxine_vfx;
      topology.domains[0].first_stage_id = 1;
      topology.domains[0].last_stage_id = 1;
      topology.exact_ordered_boundaries = true;
      topology.cross_adapter_residency_forbidden = true;
      if (mode == SourceMode::single_domain_fabricated_boundary) {
        topology.boundary_count = 1;
        topology.boundaries[0].kind =
            hal::ExecutionBoundaryKind::adapter_transfer;
      }
      if (mode == SourceMode::single_domain_fabricated_transfer)
        topology.expected_transfers.intermediate_readbacks = 1;
      result.status = hal::StatusId::ready;
      result.readiness.status = hal::StatusId::ready;
      result.readiness.requested_backend = hal::BackendId::maxine;
      result.readiness.selected_backend = hal::BackendId::maxine;
      result.readiness.provider =
          hal::SemanticEffectProviderId::maxine_vfx;
      result.readiness.execution_domain = Domain();
      result.readiness.ready_stage_mask = 1;
      result.readiness.available = true;
      result.setup_completed_outside_frame_loop = true;
      return result;
    }

    auto owner = std::make_unique<FakeLeaseOwner>(counters_);
    auto snapshot = LeaseSnapshot(owner.get());
    video::ProductionVideoSetupLeaseAcquireEvidence acquire;
    hal::ReasonId reason = hal::ReasonId::none;
    const auto status = video::ProductionVideoSetupLease::Acquire(
        snapshot, std::move(owner), &result.lease, &acquire, &reason);
    result.status = status;
    result.reason = reason;
    result.readiness = snapshot.readiness;
    result.lease_acquire = acquire;
    result.setup_completed_outside_frame_loop = true;
    return result;
  }

  SourceMode mode = SourceMode::ready;
  SourceTailMode tail_mode = SourceTailMode::preserve;

private:
  Counters *counters_ = nullptr;
};

video::CpuPrimitiveResult CpuConvert(
    const std::uint8_t *, int, int, std::size_t, std::uint8_t *,
    std::size_t) noexcept {
  return video::CpuPrimitiveResult::success;
}
std::size_t CpuScratch(int, int) noexcept { return 0; }
video::CpuPrimitiveResult CpuConvertScratch(
    const std::uint8_t *, int, int, std::size_t, std::uint8_t *, std::size_t,
    std::uint8_t *, std::size_t) noexcept {
  return video::CpuPrimitiveResult::success;
}
video::CpuPrimitiveResult CpuSwap(const std::uint8_t *, std::uint8_t *, int,
                                  int, std::size_t, std::size_t) noexcept {
  return video::CpuPrimitiveResult::success;
}
video::CpuPrimitiveResult CpuMirror(std::uint8_t *, int, int,
                                    std::size_t) noexcept {
  return video::CpuPrimitiveResult::success;
}
std::size_t CpuResizeScratch(int, int, int, int) noexcept { return 0; }
video::CpuPrimitiveResult CpuPrepareResize(
    int, int, int, int, void *, std::size_t,
    video::CpuRgb24BilinearResizePlan *) noexcept {
  return video::CpuPrimitiveResult::success;
}
video::CpuPrimitiveResult
CpuResize(const video::CpuRgb24BilinearResizePlan *, const std::uint8_t *,
          std::size_t, std::uint8_t *, std::size_t) noexcept {
  return video::CpuPrimitiveResult::success;
}

video::CpuPrimitiveTable CpuTable() {
  video::CpuPrimitiveTable table;
  table.table_version = video::kCpuPrimitiveTableVersion;
  table.struct_size = sizeof(table);
  table.backend = hal::BackendId::cpu;
  table.capability_bits =
      video::CpuPrimitiveCapabilityBit(
          hal::PrimitiveOperationId::convert_pixel_format) |
      video::CpuPrimitiveCapabilityBit(hal::PrimitiveOperationId::resize) |
      video::CpuPrimitiveCapabilityBit(hal::PrimitiveOperationId::mirror);
  table.yuyv_to_rgb24 = &CpuConvert;
  table.rgb24_to_yuyv_scratch_bytes = &CpuScratch;
  table.rgb24_to_yuyv = &CpuConvertScratch;
  table.rgb24_bgr24 = &CpuSwap;
  table.mirror_rgb24_in_place = &CpuMirror;
  table.resize_rgb24_bilinear_plan_scratch_bytes = &CpuResizeScratch;
  table.prepare_resize_rgb24_bilinear = &CpuPrepareResize;
  table.apply_resize_rgb24_bilinear = &CpuResize;
  return table;
}

bool TestCpuPrimitiveOnlyIsExplicitAndNonSemantic() {
  const auto table = CpuTable();
  video::ProductionVideoBackendManager manager(nullptr);
  video::ProductionVideoBackendSetupRequest request;
  request.kind = video::ProductionVideoBackendSetupKind::cpu_primitives_only;
  request.cpu.table = &table;
  request.cpu.required_primitive_bits =
      video::CpuPrimitiveCapabilityBit(hal::PrimitiveOperationId::resize);
  video::ProductionVideoBackendSetupEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  bool ok = Require(manager.Setup(request, &evidence, &reason) ==
                        hal::StatusId::ready,
                    "valid CPU primitive-only setup must be ready");
  const auto snapshot = manager.Snapshot();
  ok &= Require(
      snapshot.state ==
              video::ProductionVideoBackendManagerState::cpu_primitives_only &&
          snapshot.readiness.selected_backend == hal::BackendId::cpu &&
          snapshot.readiness.provider ==
              hal::SemanticEffectProviderId::invalid &&
          !hal::ValidateExecutionDomain(snapshot.readiness.execution_domain) &&
          snapshot.resident.telemetry.transfers.uploads == 0 &&
          snapshot.resident.telemetry.transfers.final_readbacks == 0,
      "CPU setup must claim no semantic provider, domain, or transfer");
  std::array<std::byte, kFrameBytes> input{};
  std::array<std::byte, kFrameBytes> output{};
  const auto frame = manager.Execute(Binding(&input, &output, 1));
  ok &= Require(frame.status == hal::StatusId::passthrough &&
                    frame.passthrough && !frame.output_ready &&
                    frame.transfers.uploads == 0 &&
                    frame.transfers.final_readbacks == 0,
                "resident manager must leave CPU primitive execution separate");
  ok &= Require(manager.CommitOutput({}, true, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::output_not_committed,
                "CPU-only manager must not invent a resident commit");
  return ok;
}

bool TestPreparedExecutionAndWriterGatedCommit() {
  Counters counters;
  auto source = std::make_unique<FakeSetupSource>(&counters);
  video::ProductionVideoBackendManager manager(std::move(source));
  auto request = ResidentRequest();
  video::ProductionVideoBackendSetupEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  const auto setup_status = manager.Setup(request, &evidence, &reason);
  bool ok = Require(setup_status == hal::StatusId::ready &&
                        evidence.setup_source_called &&
                        evidence.candidate_setup_only,
                    "resident setup must acquire and compose once");
  const auto before = manager.Snapshot();
  const auto again = manager.Snapshot();
  ok &= Require(counters.source_calls == 1 && before.resident_prepared &&
                    again.readiness.execution_domain == Domain(),
                "cached snapshots must not call the setup source");

  std::array<std::byte, kFrameBytes> input{};
  std::array<std::byte, kFrameBytes> output{};
  input[0] = std::byte{0x5a};
  auto frame = manager.Execute(Binding(&input, &output, 1));
  ok &= Require(frame.status == hal::StatusId::executed &&
                    frame.output_ready && output[0] == input[0] &&
                    frame.transfers.uploads == 1 &&
                    frame.transfers.final_readbacks == 1 &&
                    counters.source_calls == 1,
                "Execute must use only the prepared resident island");
  ok &= Require(manager.CommitOutput(frame.completion, false, &reason) ==
                        hal::StatusId::failed &&
                    reason == hal::ReasonId::output_write_failed &&
                    !manager.Snapshot().resident.latest_committed_frame.valid,
                "failed writer publication must not create active attribution");

  frame = manager.Execute(Binding(&input, &output, 2));
  ok &= Require(manager.CommitOutput(frame.completion, true, &reason) ==
                        hal::StatusId::ready &&
                    manager.Snapshot()
                        .resident.latest_committed_frame.capture_sequence == 2,
                "successful writer publication must commit exact frame");
  ok &= Require(counters.source_calls == 1 && counters.execute_calls == 2 &&
                    counters.commit_calls == 2,
                "frame calls must not perform setup or discovery");
  return ok;
}

bool TestOptionalUnavailableUsesNeutralPassthrough() {
  Counters counters;
  auto source = std::make_unique<FakeSetupSource>(&counters);
  auto *raw_source = source.get();
  raw_source->mode = SourceMode::optional_passthrough;
  video::ProductionVideoBackendManager manager(std::move(source));
  auto request = ResidentRequest();
  request.resident.plan.intent.backend.backend = hal::BackendId::cuda;
  request.resident.plan.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::open_cuda;
  request.resident.plan.intent.requested_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  video::ProductionVideoBackendSetupEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  const auto setup = manager.Setup(request, &evidence, &reason);
  bool ok =
      Require(setup == hal::StatusId::degraded &&
                  reason == hal::ReasonId::capability_evidence_incomplete,
              "all optional unavailable setup must degrade to passthrough");
  const auto prepared = manager.Snapshot();
  ok &= Require(
      prepared.state ==
              video::ProductionVideoBackendManagerState::resident_ready &&
          prepared.resident.plan.route ==
              video::VideoExecutionRoute::passthrough &&
          prepared.resident.prepare.resolution.stages[0].skipped_optional &&
          prepared.readiness.selected_backend == hal::BackendId::invalid &&
          prepared.readiness.provider == hal::SemanticEffectProviderId::invalid,
      "manager must retain typed resolution without selecting CUDA");

  std::array<std::byte, kFrameBytes> input{}, output{};
  auto frame = manager.Execute(Binding(&input, &output, 1));
  ok &=
      Require(frame.status == hal::StatusId::passthrough && frame.passthrough &&
                  !frame.output_ready && frame.transfers.uploads == 0 &&
                  frame.transfers.final_readbacks == 0,
              "manager passthrough must publish with zero GPU transfers");
  ok &= Require(manager.CommitOutput(frame.completion, false, &reason) ==
                        hal::StatusId::failed &&
                    !manager.Snapshot().resident.latest_committed_frame.valid,
                "writer failure must leave passthrough uncommitted");
  frame = manager.Execute(Binding(&input, &output, 2));
  ok &= Require(
      manager.CommitOutput(frame.completion, true, &reason) ==
              hal::StatusId::ready &&
          manager.Snapshot().resident.latest_committed_frame.valid &&
          manager.Snapshot().resident.latest_committed_frame.stage_count == 0 &&
          manager.Snapshot().resident.telemetry.active_backend ==
              hal::BackendId::invalid,
      "writer success must commit without backend attribution");

  auto bad_binding = Binding(&input, &output, 3);
  bad_binding.output_descriptor.width = 8;
  ok &= Require(manager.Execute(bad_binding).status == hal::StatusId::blocked,
                "passthrough must preserve exact geometry binding");

  bad_binding = Binding(&input, &output, 4);
  --bad_binding.input_planes[0].span_bytes;
  ok &= Require(manager.Execute(bad_binding).status == hal::StatusId::blocked,
                "passthrough must reject a short borrowed input plane");
  bad_binding = Binding(&input, &output, 5);
  ++bad_binding.output_planes[0].row_stride_bytes;
  ok &= Require(manager.Execute(bad_binding).status == hal::StatusId::blocked,
                "passthrough must reject a wrong borrowed output stride");

  raw_source->mode = SourceMode::nondeterministic_optional_passthrough_hash;
  ++request.resident.plan.plan_id;
  ok &= Require(manager.Setup(request, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    evidence.old_plan_retained,
                "source-defined neutral route hashes must be rejected");

  raw_source->mode = SourceMode::malformed_optional_passthrough;
  ++request.resident.plan.plan_id;
  ok &= Require(manager.Setup(request, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    evidence.old_plan_retained &&
                    manager.Snapshot().resident.plan.route ==
                        video::VideoExecutionRoute::passthrough,
                "malformed reconfiguration must retain active passthrough");
  const auto stopped = manager.Shutdown();
  ok &= Require(stopped.quiescent && stopped.driver_context_release_permitted &&
                    manager.Snapshot().state ==
                        video::ProductionVideoBackendManagerState::stopped,
                "neutral passthrough shutdown must be complete");
  return ok;
}

video::ProductionVideoBackendSetupRequest BackendResolvedTailRequest() {
  auto request = ResidentRequest();
  auto &plan = request.resident.plan;
  plan.intent.stages[0].intent.stage_id = 2;
  plan.cpu_tail_count = 1;
  plan.cpu_tails[0].id = hal::CpuTailId::backend_resolved_output;
  plan.cpu_tails[0].after_stage_id = 2;
  plan.cpu_tails[0].required = true;
  plan.cpu_tails[0].placement =
      hal::DeclaredCpuTail::Placement::after_stage;
  plan.cpu_tails[0].boundary_index = 1;
  return request;
}

bool TestSetupOnlyOutputTailRefinementIsExact() {
  Counters counters;
  auto source = std::make_unique<FakeSetupSource>(&counters);
  auto *raw_source = source.get();
  raw_source->mode = SourceMode::single_domain_compatibility;
  video::ProductionVideoBackendManager manager(std::move(source));
  const auto request = BackendResolvedTailRequest();
  video::ProductionVideoBackendSetupEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;

  raw_source->tail_mode = SourceTailMode::writer_copy;
  const auto writer_status = manager.Setup(request, &evidence, &reason);
  bool ok = Require(
      writer_status == hal::StatusId::ready &&
          manager.Snapshot().compatibility_topology.domain_count == 1 &&
          raw_source->tail_mode == SourceTailMode::writer_copy &&
          request.resident.plan.cpu_tails[0].id ==
              hal::CpuTailId::backend_resolved_output &&
          evidence.candidate_setup_only &&
          manager.Snapshot().compatibility.prepared &&
          manager.Snapshot().compatibility.status == hal::StatusId::ready &&
          manager.Snapshot().readiness.available &&
          manager.Snapshot().state ==
              video::ProductionVideoBackendManagerState::compatibility_ready,
      "non-ordinal terminal marker must refine to writer_copy");
  raw_source->tail_mode = SourceTailMode::output_format_conversion;
  ok &= Require(
      manager.Setup(request, &evidence, &reason) == hal::StatusId::ready &&
          manager.Snapshot().state ==
              video::ProductionVideoBackendManagerState::compatibility_ready,
      "terminal marker must refine to output conversion");
  raw_source->tail_mode = SourceTailMode::provider_postprocess;
  ok &= Require(manager.Setup(request, &evidence, &reason) ==
                        hal::StatusId::ready,
                "terminal marker must refine to provider postprocess");
  raw_source->tail_mode = SourceTailMode::no_tail;
  ok &= Require(
      manager.Setup(request, &evidence, &reason) == hal::StatusId::ready &&
          manager.Snapshot().compatibility_topology.boundary_count == 0,
      "Vulkan-style zero-tail refinement must be accepted");
  raw_source->tail_mode = SourceTailMode::preserve;
  ok &= Require(manager.Setup(request, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::malformed_contract,
                "setup marker must not reach prepared execution");
  raw_source->tail_mode = SourceTailMode::wrong_tail;
  ok &= Require(manager.Setup(request, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::malformed_contract,
                "unapproved tail refinement must fail closed");
  raw_source->tail_mode = SourceTailMode::altered_boundary;
  ok &= Require(manager.Setup(request, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::malformed_contract,
                "tail refinement must preserve terminal placement");
  return ok;
}

bool TestSourceBindsDefaultModelWithinCallerBudget() {
  Counters counters;
  auto source = std::make_unique<FakeSetupSource>(&counters);
  video::ProductionVideoBackendManager manager(std::move(source));
  auto request = ResidentRequest();
  request.resident.plan.intent.stages[0].model = {};
  request.resident.plan.invalidation_key.generations.model = 0;
  request.resident.plan.invalidation_key.model_artifact_hash = 0;
  request.resident.plan.budget.persistent_bytes += 64;
  video::ProductionVideoBackendSetupEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  bool ok = Require(manager.Setup(request, &evidence, &reason) ==
                        hal::StatusId::ready,
                    "setup source must bind an unbound default model");
  const auto snapshot = manager.Snapshot();
  ok &= Require(
      snapshot.resident.plan.invalidation_key.generations.model == 3 &&
          snapshot.resident.plan.invalidation_key.model_artifact_hash != 0 &&
          snapshot.resident.plan.budget.persistent_bytes <
              request.resident.plan.budget.persistent_bytes,
      "bound plan must publish exact model generation/hash and reserved budget");

  auto exact_substitution = ResidentRequest();
  exact_substitution.resident.effects.virtual_background.model_id =
      "explicit_matting_model";
  exact_substitution.resident.plan.intent.stages[0].model =
      ModelIdentity(0x71);
  exact_substitution.resident.plan.invalidation_key.model_artifact_hash =
      0x7172;
  ok &= Require(manager.Setup(exact_substitution, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::malformed_contract &&
                    evidence.old_plan_retained,
                "source must not substitute an explicit artifact identity");

  auto selector_substitution = ResidentRequest();
  selector_substitution.resident.effects.virtual_background.model_id =
      "other_explicit_model";
  selector_substitution.resident.plan.intent.stages[0].model.model_id_hash[0] =
      0x72;
  selector_substitution.resident.plan.invalidation_key.generations.model = 0;
  ok &= Require(manager.Setup(selector_substitution, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::malformed_contract &&
                    evidence.old_plan_retained,
                "source must preserve a caller model-id selector");
  return ok;
}

bool TestMalformedStaleWrongDomainAndStrictSubstitution() {
  Counters counters;
  auto source = std::make_unique<FakeSetupSource>(&counters);
  auto *raw_source = source.get();
  video::ProductionVideoBackendManager manager(std::move(source));
  video::ProductionVideoBackendSetupEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;

  auto malformed = ResidentRequest();
  malformed.resident.plan.plan_id = 0;
  bool ok = Require(manager.Setup(malformed, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                        reason == hal::ReasonId::malformed_contract &&
                        counters.source_calls == 0,
                    "malformed request must fail before setup source");

  auto null_callback = ResidentRequest();
  null_callback.resident.wait.stop_requested = nullptr;
  ok &= Require(manager.Setup(null_callback, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::malformed_contract &&
                    counters.source_calls == 0,
                "missing stop callback must fail before setup source");

  auto null_context = ResidentRequest();
  null_context.resident.wait.callback_context = nullptr;
  ok &= Require(manager.Setup(null_context, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::malformed_contract &&
                    counters.source_calls == 0,
                "null callback context must fail before setup source");

  auto stale_lifetime = ResidentRequest();
  stale_lifetime.resident.wait.callback_context_generation = 0;
  ok &= Require(manager.Setup(stale_lifetime, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::malformed_contract &&
                    counters.source_calls == 0,
                "unbound callback lifetime generation must fail closed");

  auto short_lifetime = ResidentRequest();
  short_lifetime.resident.wait.callback_context_lifetime =
      hal::ResourceLifetime::frame_borrowed;
  ok &= Require(manager.Setup(short_lifetime, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::malformed_contract &&
                    counters.source_calls == 0,
                "frame-scoped wait context must not escape setup");

  const auto valid = ResidentRequest();
  ok &= Require(manager.Setup(valid, &evidence, &reason) ==
                        hal::StatusId::ready,
                "baseline resident plan must prepare");

  auto substituted = ResidentRequest();
  substituted.resident.plan.intent.backend.backend = hal::BackendId::cuda;
  ok &= Require(manager.Setup(substituted, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::strict_backend_unavailable &&
                    evidence.old_plan_retained &&
                    manager.Snapshot().resident_prepared,
                "strict backend substitution must retain old plan");

  auto stale = ResidentRequest();
  ++stale.resident.plan.invalidation_key.generations.configuration;
  ok &= Require(manager.Setup(stale, &evidence, &reason) ==
                        hal::StatusId::stale &&
                    reason == hal::ReasonId::wrong_generation &&
                    evidence.old_plan_retained,
                "stale candidate generation must fail exactly");

  raw_source->mode = SourceMode::foreign_returned_domain;
  ok &= Require(manager.Setup(valid, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::wrong_execution_domain &&
                    evidence.old_plan_retained,
                "foreign setup-source domain binding must fail exactly");

  raw_source->mode = SourceMode::ready;
  auto exhausted = ResidentRequest();
  --exhausted.resident.plan.budget.scratch_bytes;
  ok &= Require(manager.Setup(exhausted, &evidence, &reason) ==
                        hal::StatusId::resource_exhausted &&
                    reason == hal::ReasonId::resource_exhausted &&
                    evidence.old_plan_retained,
                "resource reservation mismatch must fail exhausted");

  raw_source->mode = SourceMode::malformed_returned_plan;
  ok &= Require(manager.Setup(valid, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::malformed_contract &&
                    evidence.old_plan_retained,
                "setup source must not rewrite the canonical plan");
  return ok;
}

bool TestInvalidationAndRetryableShutdownAreTyped() {
  Counters counters;
  counters.pending_shutdown_attempts = 1;
  auto source = std::make_unique<FakeSetupSource>(&counters);
  video::ProductionVideoBackendManager manager(std::move(source));
  auto request = ResidentRequest();
  video::ProductionVideoBackendSetupEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  if (!Require(manager.Setup(request, &evidence, &reason) ==
                   hal::StatusId::ready,
               "shutdown test setup must prepare")) {
    return false;
  }
  manager.Invalidate(hal::ReasonId::reconfiguration_required);
  std::array<std::byte, kFrameBytes> input{};
  std::array<std::byte, kFrameBytes> output{};
  bool ok = Require(
      manager.Execute(Binding(&input, &output, 1)).status ==
              hal::StatusId::stale &&
          manager.Snapshot().state ==
              video::ProductionVideoBackendManagerState::invalidated,
      "invalidated manager must reject execution without rebuilding");

  const auto pending = manager.Shutdown();
  ok &= Require(pending.pending &&
                    manager.Snapshot().state ==
                        video::ProductionVideoBackendManagerState::
                            shutdown_pending,
                "first unproven shutdown must remain pending");
  const auto stopped = manager.Shutdown();
  ok &= Require(!stopped.pending && stopped.quiescent &&
                    stopped.driver_context_release_permitted &&
                    manager.Snapshot().state ==
                        video::ProductionVideoBackendManagerState::stopped &&
                    counters.owner_shutdown_calls == 2,
                "explicit retry must stop only after quiescence proof");
  ok &= Require(manager.Setup(request, &evidence, &reason) ==
                        hal::StatusId::stopped &&
                    reason == hal::ReasonId::shutdown_requested,
                "stopped manager must reject setup");
  return ok;
}

bool TestSingleDomainCompatibilityTopologyIsExact() {
  Counters counters;
  auto source = std::make_unique<FakeSetupSource>(&counters);
  auto *raw_source = source.get();
  video::ProductionVideoBackendManager manager(std::move(source));
  const auto request = ResidentRequest();
  video::ProductionVideoBackendSetupEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;

  raw_source->mode = SourceMode::single_domain_compatibility;
  bool ok = Require(
      manager.Setup(request, &evidence, &reason) == hal::StatusId::ready &&
          manager.Snapshot().state ==
              video::ProductionVideoBackendManagerState::compatibility_ready &&
          manager.Snapshot().compatibility_topology.domain_count == 1 &&
          manager.Snapshot().compatibility_topology.boundary_count == 0,
      "one exact provider-matching compatibility domain must be valid");

  raw_source->mode = SourceMode::single_domain_fabricated_boundary;
  ok &= Require(manager.Setup(request, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::malformed_contract &&
                    evidence.old_plan_retained,
                "one-domain topology must reject a fabricated boundary");

  raw_source->mode = SourceMode::single_domain_fabricated_transfer;
  ok &= Require(manager.Setup(request, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    reason == hal::ReasonId::malformed_contract &&
                    evidence.old_plan_retained,
                "one-domain topology must reject an intermediate transfer");
  return ok;
}

bool TestFailedTemporalResetRequiresSetupReprepare() {
  Counters counters;
  auto source = std::make_unique<FakeSetupSource>(&counters);
  video::ProductionVideoBackendManager manager(std::move(source));
  auto request = ResidentRequest();
  video::ProductionVideoBackendSetupEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  if (!Require(manager.Setup(request, &evidence, &reason) ==
                   hal::StatusId::ready,
               "temporal reset test setup must prepare")) {
    return false;
  }
  counters.reset_status = hal::StatusId::failed;
  const auto reset_status =
      manager.ResetTemporal(hal::ReasonId::temporal_discontinuity);
  bool ok =
      Require(reset_status == hal::StatusId::failed,
              "backend temporal reset failure must propagate");
  const auto failed = manager.Snapshot();
  ok &= Require(
      failed.state ==
              video::ProductionVideoBackendManagerState::invalidated &&
          failed.reason == hal::ReasonId::temporal_discontinuity &&
          !failed.resident_prepared && !failed.setup_complete,
      "failed temporal reset must invalidate resident setup immediately");
  std::array<std::byte, kFrameBytes> input{};
  std::array<std::byte, kFrameBytes> output{};
  ok &= Require(manager.Execute(Binding(&input, &output, 1)).status ==
                        hal::StatusId::stale,
                "Execute must require setup-thread reprepare after reset failure");

  counters.reset_status = hal::StatusId::ready;
  ok &= Require(manager.Setup(request, &evidence, &reason) ==
                        hal::StatusId::ready,
                "setup thread must be able to reprepare invalidated manager");
  counters.reset_status = hal::StatusId::quarantined;
  ok &= Require(manager.ResetTemporal(hal::ReasonId::device_lost) ==
                        hal::StatusId::quarantined &&
                    manager.Snapshot().reason ==
                        hal::ReasonId::device_lost &&
                    manager.Snapshot().resident.quarantined,
                "device-loss reset failure must retain quarantine reason");
  return ok;
}

bool TestOnlySelectedMaxineMayAddGreenscreenHistory() {
  bool ok = true;
  for (const bool automatic : {false, true}) {
    Counters counters;
    auto source = std::make_unique<FakeSetupSource>(&counters);
    source->mode = SourceMode::maxine_temporal_refinement;
    auto *raw = source.get();
    video::ProductionVideoBackendManager manager(std::move(source));
    auto request = ResidentRequest();
    request.resident.plan.temporal = {};
    if (automatic) {
      request.resident.plan.intent.backend = {};
      request.resident.plan.intent.requested_provider =
          hal::SemanticEffectProviderId::invalid;
      request.resident.plan.intent.strict_provider_selection = false;
    }
    video::ProductionVideoBackendSetupEvidence evidence;
    hal::ReasonId reason = hal::ReasonId::none;
    const auto refinement_status = manager.Setup(request, &evidence, &reason);
    ok &= Require(refinement_status == hal::StatusId::ready,
                  "selected Maxine must add its default greenscreen history "
                  "for auto and explicit intent");
    request.resident.effects.virtual_background.greenscreen_temporal = false;
    request.resident.plan.intent.stages[0].parameters[3].boolean_value = false;
    ok &= Require(
        manager.Setup(request, &evidence, &reason) == hal::StatusId::blocked &&
            reason == hal::ReasonId::malformed_contract,
        "Maxine must not add history when its canonical knob is disabled");
    request.resident.effects.virtual_background.greenscreen_temporal = true;
    request.resident.plan.intent.stages[0].parameters[3].boolean_value = true;
    raw->mode = SourceMode::excessive_temporal_history;
    ok &= Require(
        manager.Setup(request, &evidence, &reason) == hal::StatusId::blocked &&
            reason == hal::ReasonId::malformed_contract,
        "Maxine greenscreen refinement must not add arbitrary history");
    raw->mode = SourceMode::erased_temporal;
    request.resident.plan.temporal.temporal = true;
    request.resident.plan.temporal.requires_contiguous_capture_sequence = true;
    request.resident.plan.temporal.history_frames = 5;
    request.resident.plan.temporal.frame_drop_behavior =
        hal::FrameDropBehavior::reset_temporal_state;
    request.resident.plan.temporal.discontinuity_action =
        hal::TemporalDiscontinuityAction::reset_and_continue;
    ok &= Require(manager.Setup(request, &evidence, &reason) ==
                          hal::StatusId::blocked &&
                      reason == hal::ReasonId::malformed_contract,
                  "setup source must not erase meaningful temporal history");
    raw->mode = SourceMode::maxine_temporal_refinement;
    ok &= Require(manager.Setup(request, &evidence, &reason) ==
                          hal::StatusId::blocked &&
                      reason == hal::ReasonId::malformed_contract,
                  "Maxine must not shorten meaningful temporal history");
  }
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok &= TestOnlySelectedMaxineMayAddGreenscreenHistory();
  ok &= TestCpuPrimitiveOnlyIsExplicitAndNonSemantic();
  ok &= TestPreparedExecutionAndWriterGatedCommit();
  ok &= TestOptionalUnavailableUsesNeutralPassthrough();
  ok &= TestSetupOnlyOutputTailRefinementIsExact();
  ok &= TestSourceBindsDefaultModelWithinCallerBudget();
  ok &= TestMalformedStaleWrongDomainAndStrictSubstitution();
  ok &= TestInvalidationAndRetryableShutdownAreTyped();
  ok &= TestSingleDomainCompatibilityTopologyIsExact();
  ok &= TestFailedTemporalResetRequiresSetupReprepare();
  if (!ok)
    return 1;
  std::cout << "Production video backend manager tests passed\n";
  return 0;
}
