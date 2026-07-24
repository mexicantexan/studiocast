#include "core/video/production_video_execution_composition.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>

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
    return kAdapterId;
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
};

struct FakeExecutorCounters {
  std::uint64_t execute_calls = 0;
  std::uint64_t commit_calls = 0;
  std::uint64_t invalidate_calls = 0;
  std::uint64_t reset_calls = 0;
  std::uint64_t shutdown_calls = 0;
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
    evidence->status = hal::StatusId::ready;
    return std::make_unique<FakeExecutor>(plan, &counters);
  }

  FakeExecutorCounters counters{};
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

} // namespace

int main() {
  return TestLifecycleAndAttribution() &&
                 TestMalformedFactoryRegistrationRollsBack()
             ? 0
             : 1;
}
