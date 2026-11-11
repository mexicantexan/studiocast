#include "core/video/production_video_execution_composition.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace {

namespace hal = studiocast::hal;
namespace video = studiocast::video;

constexpr std::uint64_t kAdapterId = 9001;
constexpr std::uint64_t kFactoryId = 9002;
constexpr std::size_t kBytes = 24;

bool Require(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

hal::ExecutionDomainId Domain() {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::vulkan;
  domain.api_family = hal::ApiFamily::vulkan;
  domain.physical_device.vendor_id = 0x1234;
  domain.physical_device.device_id = 0x5678;
  domain.physical_device.stable_id[0] = 1;
  domain.logical_context_generation = 7;
  domain.allocator_family = hal::AllocatorFamily::vulkan;
  domain.memory_handle_family = hal::MemoryHandleFamily::vulkan_buffer;
  domain.queue_or_stream_identity = 11;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::vulkan_queue_fence;
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
  image.planes[0].size_bytes = kBytes;
  image.planes[0].width = 4;
  image.planes[0].height = 2;
  image.planes[0].alignment_bytes = 1;
  return image;
}

class UnusedAdapter final : public hal::IResidentAdapter {
public:
  std::uint64_t adapter_instance_id() const noexcept override {
    return reported_adapter_id;
  }
  hal::BackendId backend_id() const noexcept override {
    return hal::BackendId::vulkan;
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
    return hal::StatusId::blocked;
  }
  void DestroyPrepared(hal::PreparedProgramHandle *) noexcept override {}
  void Shutdown() noexcept override {}

  std::uint64_t reported_adapter_id = kAdapterId;
};

struct FakeExecutorCounters {
  std::uint64_t execute_calls = 0;
  std::uint64_t commit_calls = 0;
  std::uint64_t invalidate_calls = 0;
  std::uint64_t reset_calls = 0;
  std::uint64_t shutdown_calls = 0;
  std::vector<int> *events = nullptr;
};

class FakeExecutor final : public video::IVideoExecutionExecutor {
public:
  FakeExecutor(video::FrozenVideoExecutionPlan plan,
               FakeExecutorCounters *counters)
      : plan_(plan), counters_(counters) {
    snapshot_.status = hal::StatusId::ready;
    snapshot_.execution_domain = plan_.domain_plan.islands[0].execution_domain;
    snapshot_.plan_id = plan_.plan_id;
    snapshot_.prepared = true;
    snapshot_.readiness.status = hal::StatusId::ready;
    snapshot_.readiness.requested_backend = hal::BackendId::vulkan;
    snapshot_.readiness.selected_backend = hal::BackendId::vulkan;
    snapshot_.readiness.provider =
        hal::SemanticEffectProviderId::open_vulkan_production_kernels;
    snapshot_.readiness.execution_domain = snapshot_.execution_domain;
    snapshot_.readiness.ready_stage_mask = 1;
    snapshot_.readiness.available = true;
  }

  video::VideoExecutionFrameResult
  Execute(const hal::HostFrameBinding &binding) noexcept override {
    ++counters_->execute_calls;
    video::VideoExecutionFrameResult result;
    if (invalidated_ || stopped_) {
      result.status = stopped_ ? hal::StatusId::stopped : hal::StatusId::stale;
      result.reason = stopped_ ? hal::ReasonId::shutdown_requested
                               : hal::ReasonId::stale_program;
      return result;
    }
    std::memcpy(binding.output_planes[0].write_data,
                binding.input_planes[0].read_data, kBytes);
    result.status = hal::StatusId::executed;
    result.completion = {next_token_++, plan_.plan_id, kAdapterId,
                         binding.capture_sequence, true};
    result.output.descriptor = binding.output_descriptor;
    result.output.planes = binding.output_planes;
    result.output.plane_count = 1;
    result.output.capture_sequence = binding.capture_sequence;
    result.output.output_ready = true;
    result.execution_domain = Domain();
    result.stage_count = 1;
    result.stages[0].stage_id = 1;
    result.stages[0].semantic = hal::SemanticEffectId::mirror;
    result.stages[0].status = hal::StageExecutionStatus::succeeded;
    result.stages[0].valid_output_preserved = true;
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
    if (completion.token_id != pending_.token_id) {
      if (reason)
        *reason = hal::ReasonId::stale_program;
      return hal::StatusId::stale;
    }
    pending_ = {};
    if (reason)
      *reason = writer_succeeded ? hal::ReasonId::none
                                 : hal::ReasonId::output_write_failed;
    return writer_succeeded ? hal::StatusId::ready : hal::StatusId::failed;
  }

  void Invalidate(hal::ReasonId) noexcept override {
    ++counters_->invalidate_calls;
    invalidated_ = true;
    snapshot_.prepared = false;
  }
  hal::StatusId ResetTemporal(hal::ReasonId) noexcept override {
    ++counters_->reset_calls;
    return hal::StatusId::ready;
  }
  void Shutdown() noexcept override {
    ++counters_->shutdown_calls;
    if (counters_->events)
      counters_->events->push_back(1);
    stopped_ = true;
    snapshot_.stopped = true;
  }
  video::VideoExecutionAdapterSnapshot Snapshot() const noexcept override {
    return snapshot_;
  }

private:
  video::FrozenVideoExecutionPlan plan_{};
  FakeExecutorCounters *counters_ = nullptr;
  video::VideoExecutionAdapterSnapshot snapshot_{};
  hal::CompletionToken pending_{};
  std::uint64_t next_token_ = 1;
  bool invalidated_ = false;
  bool stopped_ = false;
};

class FakeFactory final : public video::IVideoAdapterFactory {
public:
  explicit FakeFactory(FakeExecutorCounters *external_counters = nullptr)
      : counters_(external_counters ? external_counters : &counters) {}

  video::VideoAdapterFactoryPreflight
  Preflight(const video::VideoExecutionPlanRequest &request,
            const hal::ResolutionResult &resolution,
            const hal::DomainPlan &domain_plan,
            video::VideoExecutionRoute route) noexcept override {
    video::VideoAdapterFactoryPreflight preflight;
    (void)resolution;
    preflight.status = hal::StatusId::ready;
    preflight.backend = hal::BackendId::vulkan;
    preflight.provider =
        hal::SemanticEffectProviderId::open_vulkan_production_kernels;
    preflight.execution_domain = Domain();
    preflight.cached_evidence_only = true;
    preflight.accepts_frozen_resolution = true;
    preflight.performs_selection = false;
    preflight.per_frame_transfers =
        video::BuildPerFrameTransferContract(domain_plan, 1, 1);
    preflight.readiness.status = hal::StatusId::ready;
    preflight.readiness.requested_backend = hal::BackendId::vulkan;
    preflight.readiness.selected_backend = hal::BackendId::vulkan;
    preflight.readiness.provider = preflight.provider;
    preflight.readiness.execution_domain = Domain();
    preflight.readiness.ready_stage_mask = 1;
    preflight.readiness.available = true;
    if (route != video::VideoExecutionRoute::prepared_resident)
      return preflight;

    auto &program = preflight.prepared_program;
    program.program_id = request.plan_id;
    program.owner_adapter_instance_id = kAdapterId;
    program.execution_domain = Domain();
    program.host_input = request.host_input;
    program.host_output = request.host_output;
    program.invalidation_key = request.invalidation_key;
    program.temporal = request.temporal;
    program.budget = request.budget;
    program.setup_failure_policy = request.setup_failure_policy;
    program.requested_backend = hal::BackendId::vulkan;
    program.strict_backend_selection = true;
    program.requested_provider = preflight.provider;
    program.strict_provider_selection = true;
    program.requested_execution_domain = Domain();
    program.strict_execution_domain = true;
    program.requires_upload = true;
    program.requires_final_readback = true;
    program.stage_count = 1;
    auto &stage = program.stages[0];
    stage.stage_id = 1;
    stage.semantic = hal::SemanticEffectId::mirror;
    stage.provider = preflight.provider;
    stage.backend = hal::BackendId::vulkan;
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
    stage.circuit_breaker.cooldown_frames = 30;
    stage.circuit_breaker.action = hal::CircuitBreakerAction::disable_stage;
    return preflight;
  }

  std::unique_ptr<video::IVideoExecutionExecutor>
  Prepare(const video::FrozenVideoExecutionPlan &plan,
          video::VideoAdapterPrepareEvidence *evidence) noexcept override {
    if (fail_prepare) {
      evidence->status = hal::StatusId::unavailable;
      evidence->reason = hal::ReasonId::semantic_provider_unavailable;
      evidence->candidate_resources_rolled_back = true;
      return nullptr;
    }
    evidence->status = hal::StatusId::ready;
    return std::make_unique<FakeExecutor>(plan, counters_);
  }

  bool fail_prepare = false;
  FakeExecutorCounters counters{};

private:
  FakeExecutorCounters *counters_ = nullptr;
};

struct FakeLeaseOwnerCounters {
  std::uint64_t shutdown_calls = 0;
  std::uint64_t destructor_calls = 0;
  std::uint64_t pending_attempts_before_complete = 0;
  bool permanently_pending = false;
  bool destructor_saw_pending = false;
  std::vector<int> *events = nullptr;
};

class FakeLeaseOwner final : public video::IProductionVideoSetupLeaseOwner {
public:
  FakeLeaseOwner(FakeLeaseOwnerCounters *owner_counters,
                 FakeExecutorCounters *executor_counters)
      : factory(executor_counters), owner_counters_(owner_counters) {}

  ~FakeLeaseOwner() override {
    ++owner_counters_->destructor_calls;
    owner_counters_->destructor_saw_pending = last_shutdown_pending_;
    if (owner_counters_->events)
      owner_counters_->events->push_back(3);
  }

  hal::IResidentAdapter *resident_adapter() const noexcept override {
    return const_cast<UnusedAdapter *>(&adapter);
  }

  video::IVideoAdapterFactory *video_adapter_factory() const noexcept override {
    return const_cast<FakeFactory *>(&factory);
  }

  void Shutdown() noexcept override {
    ++owner_counters_->shutdown_calls;
    if (owner_counters_->events)
      owner_counters_->events->push_back(2);
  }
  video::ProductionVideoSetupLeaseShutdownResult
  RequestShutdown() noexcept override {
    Shutdown();
    video::ProductionVideoSetupLeaseShutdownResult result;
    result.shutdown_attempts = owner_counters_->shutdown_calls;
    result.shutdown_requested = true;
    const bool pending = owner_counters_->permanently_pending ||
                         owner_counters_->shutdown_calls <=
                             owner_counters_->pending_attempts_before_complete;
    if (pending) {
      result.status = hal::StatusId::degraded;
      result.reason = hal::ReasonId::synchronization_failed;
      result.pending = true;
      last_shutdown_pending_ = true;
      return result;
    }
    result.status = hal::StatusId::stopped;
    result.reason = hal::ReasonId::shutdown_requested;
    result.quiescent = true;
    result.driver_context_release_permitted = true;
    last_shutdown_pending_ = false;
    return result;
  }

  UnusedAdapter adapter;
  FakeFactory factory;

private:
  FakeLeaseOwnerCounters *owner_counters_ = nullptr;
  bool last_shutdown_pending_ = false;
};

hal::AdapterRegistration Registration(UnusedAdapter *adapter) {
  hal::AdapterRegistration registration;
  registration.adapter_instance_id = kAdapterId;
  registration.backend = hal::BackendId::vulkan;
  registration.execution_domain = Domain();
  registration.selection_priority = 1;
  registration.resident_adapter = adapter;
  auto &capability = registration.semantic_capabilities[0];
  capability.semantic = hal::SemanticEffectId::mirror;
  capability.provider =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  capability.status = hal::StatusId::ready;
  capability.live_stage = true;
  capability.production_eligible = true;
  capability.exact_evidence = true;
  registration.semantic_capability_count = 1;
  return registration;
}

video::VideoAdapterFactoryRegistration
FactoryRegistration(FakeFactory *factory) {
  video::VideoAdapterFactoryRegistration registration;
  registration.factory_id = kFactoryId;
  registration.route = video::VideoExecutionRoute::prepared_resident;
  registration.backend = hal::BackendId::vulkan;
  registration.provider =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  registration.adapter_instance_id = kAdapterId;
  registration.execution_domain = Domain();
  registration.selection_priority = 1;
  registration.factory = factory;
  return registration;
}

video::ProductionVideoSetupLeaseSnapshot LeaseSnapshot(FakeLeaseOwner *owner) {
  video::ProductionVideoSetupLeaseSnapshot snapshot;
  snapshot.state = video::ProductionVideoSetupLeaseState::live;
  snapshot.status = hal::StatusId::ready;
  snapshot.backend = hal::BackendId::vulkan;
  snapshot.execution_domain = Domain();
  snapshot.generations = {1, 1, 1, 1, 1, 7};
  snapshot.reserved_budget.persistent_bytes = 128;
  snapshot.reserved_budget.scratch_bytes = 128;
  snapshot.reserved_budget.upload_staging_bytes = kBytes;
  snapshot.reserved_budget.readback_staging_bytes = kBytes;
  snapshot.reserved_budget.resource_count = 4;
  snapshot.reserved_budget.in_flight_slots = 1;
  snapshot.providers[0] =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  snapshot.provider_count = 1;
  snapshot.adapter = Registration(owner ? &owner->adapter : nullptr);
  snapshot.factory = FactoryRegistration(owner ? &owner->factory : nullptr);
  snapshot.readiness.status = hal::StatusId::ready;
  snapshot.readiness.requested_backend = hal::BackendId::vulkan;
  snapshot.readiness.selected_backend = hal::BackendId::vulkan;
  snapshot.readiness.provider =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  snapshot.readiness.execution_domain = Domain();
  snapshot.readiness.ready_stage_mask = 1;
  snapshot.readiness.available = true;
  return snapshot;
}

bool AcquireLease(std::unique_ptr<FakeLeaseOwner> owner,
                  video::ProductionVideoSetupLease *lease,
                  FakeLeaseOwner **raw_owner = nullptr) {
  if (raw_owner)
    *raw_owner = owner.get();
  const auto snapshot = LeaseSnapshot(owner.get());
  video::ProductionVideoSetupLeaseAcquireEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  return video::ProductionVideoSetupLease::Acquire(snapshot, std::move(owner),
                                                   lease, &evidence, &reason) ==
             hal::StatusId::ready &&
         reason == hal::ReasonId::none;
}

video::VideoExecutionPlanRequest Request() {
  video::VideoExecutionPlanRequest request;
  request.plan_id = 101;
  request.intent.backend.mode = hal::BackendRequestMode::explicit_backend;
  request.intent.backend.backend = hal::BackendId::vulkan;
  request.intent.backend.strict = true;
  request.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::vulkan_only;
  request.intent.requested_provider =
      hal::SemanticEffectProviderId::open_vulkan_production_kernels;
  request.intent.strict_provider_selection = true;
  request.intent.stages[0].intent.stage_id = 1;
  request.intent.stages[0].intent.semantic = hal::SemanticEffectId::mirror;
  request.intent.stages[0].intent.failure_action =
      hal::FailureAction::fail_open_preserve_input;
  request.intent.stage_count = 1;
  request.host_input = Image(hal::Mutability::read_only);
  request.host_output = Image(hal::Mutability::write_only);
  request.invalidation_key.generations = {1, 1, 1, 1, 1, 7};
  request.invalidation_key.canonical_plan_hash = 17;
  request.invalidation_key.execution_domain_hash = 19;
  request.budget.persistent_bytes = 128;
  request.budget.scratch_bytes = 128;
  request.budget.upload_staging_bytes = kBytes;
  request.budget.readback_staging_bytes = kBytes;
  request.budget.resource_count = 4;
  request.budget.in_flight_slots = 1;
  return request;
}

hal::AdapterRegistration UnavailableRegistration() {
  auto registration = Registration(nullptr);
  auto &capability = registration.semantic_capabilities[0];
  capability.status = hal::StatusId::unavailable;
  capability.reason = hal::ReasonId::capability_evidence_incomplete;
  capability.live_stage = false;
  capability.production_eligible = false;
  capability.exact_evidence = false;
  capability.evidence_count = 0;
  return registration;
}

hal::HostFrameBinding Binding(std::array<std::uint8_t, kBytes> *input,
                              std::array<std::uint8_t, kBytes> *output,
                              std::uint64_t sequence) {
  hal::HostFrameBinding binding;
  binding.input_descriptor = Image(hal::Mutability::read_only);
  binding.output_descriptor = Image(hal::Mutability::write_only);
  binding.input_planes[0].read_data =
      reinterpret_cast<const std::byte *>(input->data());
  binding.input_planes[0].span_bytes = input->size();
  binding.input_planes[0].row_stride_bytes = 12;
  binding.output_planes[0].write_data =
      reinterpret_cast<std::byte *>(output->data());
  binding.output_planes[0].span_bytes = output->size();
  binding.output_planes[0].row_stride_bytes = 12;
  binding.input_plane_count = 1;
  binding.output_plane_count = 1;
  binding.capture_sequence = sequence;
  binding.ownership = hal::OwnershipModel::borrowed_until_completion;
  binding.lifetime = hal::ResourceLifetime::frame_borrowed;
  return binding;
}

bool TestLifecycleAndAttribution() {
  UnusedAdapter adapter;
  FakeFactory factory;
  const auto adapter_registration = Registration(&adapter);
  const auto factory_registration = FactoryRegistration(&factory);
  video::ProductionVideoExecutionComposition composition(adapter_registration,
                                                         factory_registration);

  video::VideoExecutionPrepareEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  const auto prepare_status =
      composition.Prepare(Request(), &evidence, &reason);
  bool ok = Require(prepare_status == hal::StatusId::ready,
                    "injected exact-domain plan must prepare") &&
            Require(evidence.selected_route ==
                        video::VideoExecutionRoute::prepared_resident,
                    "prepared route must be selected");

  std::array<std::uint8_t, kBytes> input{};
  std::array<std::uint8_t, kBytes> output{};
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<std::uint8_t>(index);

  auto frame = composition.Execute(Binding(&input, &output, 1));
  ok &= Require(frame.output_ready && output == input,
                "execute must publish the copied output");
  ok &= Require(frame.execution_domain == Domain(),
                "execute must preserve the exact domain");
  ok &= Require(frame.transfers.uploads == 1 &&
                    frame.transfers.final_readbacks == 1 &&
                    frame.transfers.intermediate_uploads == 0 &&
                    frame.transfers.intermediate_readbacks == 0 &&
                    frame.transfers.synchronizations == 1,
                "execute must preserve the declared transfer boundary");
  ok &= Require(composition.CommitOutput(frame.completion, false, &reason) ==
                    hal::StatusId::failed,
                "writer failure must fail the commit");
  ok &= Require(!composition.Snapshot().latest_committed_frame.valid,
                "writer failure must not attribute an active backend");

  frame = composition.Execute(Binding(&input, &output, 2));
  ok &= Require(composition.CommitOutput(frame.completion, true, &reason) ==
                    hal::StatusId::ready,
                "writer success must commit");
  const auto committed = composition.Snapshot().latest_committed_frame;
  ok &= Require(committed.valid && committed.capture_sequence == 2 &&
                    committed.execution_domain == Domain() &&
                    committed.stage_count == 1 &&
                    committed.stages[0].semantic ==
                        hal::SemanticEffectId::mirror &&
                    committed.stages[0].backend == hal::BackendId::vulkan,
                "successful writer commit must publish exact attribution");

  composition.Invalidate(hal::ReasonId::reconfiguration_required);
  const auto invalidated = composition.Execute(Binding(&input, &output, 3));
  ok &= Require(invalidated.status != hal::StatusId::executed &&
                    invalidated.status != hal::StatusId::degraded &&
                    invalidated.status != hal::StatusId::passthrough,
                "invalidated composition must reject execution");
  composition.Shutdown();
  ok &= Require(composition.Snapshot().stopped,
                "shutdown must be visible in the neutral snapshot");
  ok &= Require(factory.counters.execute_calls == 2 &&
                    factory.counters.commit_calls == 2 &&
                    factory.counters.invalidate_calls == 1 &&
                    factory.counters.shutdown_calls == 1,
                "composition must own the complete executor lifecycle");
  return ok;
}

bool TestLeaseFreeOptionalPassthroughLifecycle() {
  video::ProductionVideoExecutionComposition composition;
  auto request = Request();
  video::VideoExecutionPrepareEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  bool ok = Require(composition.PreparePassthrough(
                        UnavailableRegistration(), request, &evidence,
                        &reason) == hal::StatusId::passthrough,
                    "lease-free optional candidate must prepare passthrough") &&
            Require(evidence.resolution.stages[0].skipped_optional &&
                        evidence.resolution.selected_backend ==
                            hal::BackendId::invalid,
                    "composition must independently resolve no backend");
  std::array<std::uint8_t, kBytes> input{}, output{};
  auto frame = composition.Execute(Binding(&input, &output, 4));
  ok &= Require(
      frame.status == hal::StatusId::passthrough &&
          frame.transfers.uploads == 0 && frame.transfers.final_readbacks == 0,
      "lease-free composition must execute zero-transfer passthrough");
  ok &= Require(composition.CommitOutput(frame.completion, true, &reason) ==
                        hal::StatusId::ready &&
                    composition.Snapshot().latest_committed_frame.valid &&
                    composition.Snapshot().latest_committed_frame.stage_count ==
                        0,
                "passthrough writer success must commit without attribution");
  const auto stopped = composition.Shutdown();
  ok &= Require(stopped.quiescent && stopped.driver_context_release_permitted,
                "lease-free passthrough shutdown must be complete");
  return ok;
}

bool TestMalformedFactoryRegistrationRollsBack() {
  std::unique_ptr<video::ProductionVideoExecutionComposition> composition;
  {
    UnusedAdapter adapter;
    FakeFactory factory;
    const auto adapter_registration = Registration(&adapter);
    auto malformed_factory = FactoryRegistration(&factory);
    ++malformed_factory.execution_domain.adapter_instance_id;
    composition = std::make_unique<video::ProductionVideoExecutionComposition>(
        adapter_registration, malformed_factory);
  }

  video::VideoExecutionPrepareEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  const auto first = composition->Prepare(Request(), &evidence, &reason);
  bool ok =
      Require(first == hal::StatusId::unavailable &&
                  reason == hal::ReasonId::capability_evidence_incomplete &&
                  !composition->Snapshot().prepared,
              "malformed factory registration must roll back completely");
  reason = hal::ReasonId::none;
  const auto retry = composition->Prepare(Request(), &evidence, &reason);
  ok &= Require(retry == hal::StatusId::unavailable &&
                    reason == hal::ReasonId::capability_evidence_incomplete &&
                    !composition->Snapshot().prepared,
                "failed injection retry must remain safe and unavailable");
  composition->Shutdown();
  return ok;
}

bool TestOwnedLeaseLifecycleAndOrderedShutdown() {
  std::vector<int> events;
  FakeLeaseOwnerCounters owner_counters;
  owner_counters.events = &events;
  FakeExecutorCounters executor_counters;
  executor_counters.events = &events;
  {
    auto owner =
        std::make_unique<FakeLeaseOwner>(&owner_counters, &executor_counters);
    video::ProductionVideoSetupLease lease;
    if (!Require(AcquireLease(std::move(owner), &lease),
                 "valid production lease must acquire")) {
      return false;
    }

    video::ProductionVideoExecutionComposition composition;
    video::VideoExecutionPrepareEvidence evidence;
    hal::ReasonId reason = hal::ReasonId::none;
    bool ok =
        Require(composition.PrepareWithSetupLease(std::move(lease), Request(),
                                                  &evidence, &reason) ==
                        hal::StatusId::ready &&
                    evidence.selected_route ==
                        video::VideoExecutionRoute::prepared_resident,
                "owned lease must prepare through its frozen registrations");

    std::array<std::uint8_t, kBytes> input{};
    std::array<std::uint8_t, kBytes> output{};
    for (std::size_t index = 0; index < input.size(); ++index)
      input[index] = static_cast<std::uint8_t>(index + 3);
    const auto frame = composition.Execute(Binding(&input, &output, 10));
    ok &= Require(frame.output_ready && output == input,
                  "owned lease must execute with stable owner addresses");
    ok &= Require(composition.CommitOutput(frame.completion, true, &reason) ==
                          hal::StatusId::ready &&
                      composition.Snapshot().latest_committed_frame.valid,
                  "owned lease must preserve writer-gated attribution");

    composition.Shutdown();
    composition.Shutdown();
    ok &= Require(composition.Snapshot().stopped &&
                      executor_counters.shutdown_calls == 1 &&
                      owner_counters.shutdown_calls == 1,
                  "composition shutdown must be idempotent across executor "
                  "and lease owner");
    if (!ok)
      return false;
  }

  const std::array<int, 3> expected{1, 2, 3};
  return Require(events.size() == expected.size(),
                 "ordered owned teardown must emit exactly three events") &&
         Require(std::equal(events.begin(), events.end(), expected.begin()),
                 "coordinator must shut down before owner shutdown and "
                 "destruction") &&
         Require(owner_counters.destructor_calls == 1,
                 "owned lease owner must be destroyed exactly once");
}

bool TestFailedLeaseCandidatesRetainOldPlan() {
  FakeLeaseOwnerCounters old_owner_counters;
  FakeExecutorCounters old_executor_counters;
  auto old_owner = std::make_unique<FakeLeaseOwner>(&old_owner_counters,
                                                    &old_executor_counters);
  video::ProductionVideoSetupLease old_lease;
  if (!Require(AcquireLease(std::move(old_owner), &old_lease),
               "old production lease must acquire")) {
    return false;
  }

  video::ProductionVideoExecutionComposition composition;
  video::VideoExecutionPrepareEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  if (!Require(composition.PrepareWithSetupLease(
                   std::move(old_lease), Request(), &evidence, &reason) ==
                   hal::StatusId::ready,
               "old lease must prepare before rollback checks")) {
    return false;
  }

  FakeLeaseOwnerCounters registration_owner_counters;
  FakeExecutorCounters registration_executor_counters;
  auto registration_owner = std::make_unique<FakeLeaseOwner>(
      &registration_owner_counters, &registration_executor_counters);
  video::ProductionVideoSetupLease registration_lease;
  FakeLeaseOwner *registration_raw = nullptr;
  if (!Require(AcquireLease(std::move(registration_owner), &registration_lease,
                            &registration_raw),
               "registration candidate must first acquire validly")) {
    return false;
  }
  ++registration_raw->adapter.reported_adapter_id;
  reason = hal::ReasonId::none;
  const auto registration_status = composition.PrepareWithSetupLease(
      std::move(registration_lease), Request(), &evidence, &reason);
  bool ok =
      Require(registration_status == hal::StatusId::blocked &&
                  reason == hal::ReasonId::foreign_adapter &&
                  evidence.old_plan_retained,
              "foreign post-acquire registration must fail exactly and "
              "retain the old plan") &&
      Require(registration_owner_counters.shutdown_calls == 1 &&
                  registration_owner_counters.destructor_calls == 1,
              "failed registration candidate must roll back its owner once") &&
      Require(old_owner_counters.shutdown_calls == 0,
              "failed registration must not stop the old lease");

  FakeLeaseOwnerCounters prepare_owner_counters;
  FakeExecutorCounters prepare_executor_counters;
  auto prepare_owner = std::make_unique<FakeLeaseOwner>(
      &prepare_owner_counters, &prepare_executor_counters);
  prepare_owner->factory.fail_prepare = true;
  video::ProductionVideoSetupLease prepare_lease;
  if (!Require(AcquireLease(std::move(prepare_owner), &prepare_lease),
               "prepare-failure candidate must acquire validly")) {
    return false;
  }
  reason = hal::ReasonId::none;
  const auto prepare_status = composition.PrepareWithSetupLease(
      std::move(prepare_lease), Request(), &evidence, &reason);
  ok &= Require(prepare_status == hal::StatusId::unavailable &&
                    reason == hal::ReasonId::semantic_provider_unavailable &&
                    evidence.old_plan_retained,
                "candidate prepare failure must retain exact reason and old "
                "plan") &&
        Require(prepare_owner_counters.shutdown_calls == 1 &&
                    prepare_owner_counters.destructor_calls == 1,
                "failed prepare candidate must roll back its owner once") &&
        Require(old_owner_counters.shutdown_calls == 0,
                "failed candidate prepare must not stop the old lease");

  std::array<std::uint8_t, kBytes> input{};
  std::array<std::uint8_t, kBytes> output{};
  input.fill(0xa5);
  const auto frame = composition.Execute(Binding(&input, &output, 20));
  ok &= Require(frame.output_ready && output == input,
                "old plan must remain executable after both candidate "
                "failures") &&
        Require(composition.CommitOutput(frame.completion, true, &reason) ==
                    hal::StatusId::ready,
                "old plan must remain committable after candidate failures");
  composition.Shutdown();
  return ok;
}

bool TestPendingLeaseShutdownIsRetriedAndReported() {
  FakeLeaseOwnerCounters owner_counters;
  owner_counters.pending_attempts_before_complete = 1;
  FakeExecutorCounters executor_counters;
  auto owner =
      std::make_unique<FakeLeaseOwner>(&owner_counters, &executor_counters);
  video::ProductionVideoSetupLease lease;
  if (!Require(AcquireLease(std::move(owner), &lease),
               "pending shutdown composition lease must acquire")) {
    return false;
  }

  video::ProductionVideoExecutionComposition composition;
  video::VideoExecutionPrepareEvidence evidence;
  hal::ReasonId reason = hal::ReasonId::none;
  if (!Require(composition.PrepareWithSetupLease(std::move(lease), Request(),
                                                 &evidence, &reason) ==
                   hal::StatusId::ready,
               "pending shutdown composition must prepare")) {
    return false;
  }

  const auto pending = composition.Shutdown();
  const auto pending_snapshot = composition.Snapshot();
  bool ok =
      Require(pending.pending && pending.shutdown_attempts == 1 &&
                  pending.status == hal::StatusId::degraded &&
                  pending.reason == hal::ReasonId::synchronization_failed &&
                  !pending.quiescent &&
                  !pending.driver_context_release_permitted,
              "composition must expose truthful pending lease shutdown") &&
      Require(!pending_snapshot.stopped &&
                  pending_snapshot.readiness.status ==
                      hal::StatusId::degraded &&
                  pending_snapshot.readiness.reason ==
                      hal::ReasonId::synchronization_failed,
              "terminal snapshot must distinguish pending from stopped") &&
      Require(executor_counters.shutdown_calls == 1 &&
                  owner_counters.shutdown_calls == 1,
              "coordinator must stop once before the first lease attempt");

  const auto complete = composition.Shutdown();
  ok &= Require(!complete.pending && complete.shutdown_attempts == 2 &&
                    complete.quiescent &&
                    complete.driver_context_release_permitted &&
                    composition.Snapshot().stopped,
                "retry must publish stopped only after proven completion");
  const auto idempotent = composition.Shutdown();
  ok &= Require(idempotent.shutdown_attempts == 2 &&
                    owner_counters.shutdown_calls == 2 &&
                    executor_counters.shutdown_calls == 1,
                "completed composition shutdown must be idempotent");
  return ok;
}

bool TestPermanentPendingCompositionUsesCallerBoundedAttempts() {
  FakeLeaseOwnerCounters owner_counters;
  owner_counters.permanently_pending = true;
  FakeExecutorCounters executor_counters;
  {
    auto owner =
        std::make_unique<FakeLeaseOwner>(&owner_counters, &executor_counters);
    video::ProductionVideoSetupLease lease;
    if (!Require(AcquireLease(std::move(owner), &lease),
                 "permanent pending composition lease must acquire")) {
      return false;
    }
    video::ProductionVideoExecutionComposition composition;
    video::VideoExecutionPrepareEvidence evidence;
    hal::ReasonId reason = hal::ReasonId::none;
    if (!Require(composition.PrepareWithSetupLease(std::move(lease), Request(),
                                                   &evidence, &reason) ==
                     hal::StatusId::ready,
                 "permanent pending composition must prepare")) {
      return false;
    }

    const auto first = composition.Shutdown();
    const auto second = composition.Shutdown();
    if (!Require(first.pending && first.shutdown_attempts == 1 &&
                     second.pending && second.shutdown_attempts == 2 &&
                     owner_counters.shutdown_calls == 2 &&
                     executor_counters.shutdown_calls == 1 &&
                     !composition.Snapshot().stopped,
                 "permanent pending retries must remain explicit and "
                 "bounded")) {
      return false;
    }
  }
  return Require(owner_counters.shutdown_calls == 3,
                 "composition destruction must make one final bounded owner "
                 "attempt") &&
         Require(owner_counters.destructor_calls == 1 &&
                     owner_counters.destructor_saw_pending,
                 "permanent pending owner must retain fail-safe destruction "
                 "responsibility") &&
         Require(executor_counters.shutdown_calls == 1,
                 "coordinator teardown must not repeat during lease retries");
}

} // namespace

int main() {
  return TestLifecycleAndAttribution() &&
                 TestLeaseFreeOptionalPassthroughLifecycle() &&
                 TestMalformedFactoryRegistrationRollsBack() &&
                 TestOwnedLeaseLifecycleAndOrderedShutdown() &&
                 TestFailedLeaseCandidatesRetainOldPlan() &&
                 TestPendingLeaseShutdownIsRetriedAndReported() &&
                 TestPermanentPendingCompositionUsesCallerBoundedAttempts()
             ? 0
             : 1;
}
