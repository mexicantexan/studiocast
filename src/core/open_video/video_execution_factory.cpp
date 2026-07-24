#include "core/open_video/video_execution_factory.h"

#include <new>
#include <utility>

namespace studiocast::open_cuda {
namespace {

constexpr auto kProvider =
    hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;

void SetReason(hal::ReasonId *reason, hal::ReasonId value) noexcept {
  if (reason != nullptr)
    *reason = value;
}

bool IsSetupSuccess(hal::StatusId status) noexcept {
  return status == hal::StatusId::ready || status == hal::StatusId::degraded;
}

bool SameModel(const hal::ModelArtifactIdentity &left,
               const hal::ModelArtifactIdentity &right) noexcept {
  return left.model_id_hash == right.model_id_hash &&
         left.artifact_set_hash == right.artifact_set_hash &&
         left.artifact_generation == right.artifact_generation;
}

bool TightRgb24(const hal::ImageDescriptor &image,
                hal::Mutability mutability) noexcept {
  if (image.width == 0 || image.height == 0 || image.plane_count != 1 ||
      image.format != hal::PixelFormat::rgb_u8 ||
      image.channel_order != hal::ChannelOrder::rgb ||
      image.color_space != hal::ColorSpace::srgb ||
      image.color_range != hal::ColorRange::full ||
      image.alpha_meaning != hal::AlphaMeaning::none ||
      image.orientation != hal::Orientation::top_left ||
      image.host_coherence != hal::HostCoherence::coherent ||
      image.mutability != mutability ||
      image.aliasing != hal::AliasingPermission::forbidden ||
      image.in_place_permitted || image.base_alignment_bytes != 1 ||
      image.planes[0].offset_bytes != 0 ||
      image.planes[0].alignment_bytes != 1 ||
      image.planes[0].width != image.width ||
      image.planes[0].height != image.height) {
    return false;
  }
  const auto stride = static_cast<std::uint64_t>(image.width) * 3u;
  return image.planes[0].row_stride_bytes == stride &&
         image.planes[0].size_bytes ==
             stride * static_cast<std::uint64_t>(image.height);
}

hal::ImageDescriptor ResidentImage(const hal::ImageDescriptor &host,
                                   hal::Mutability mutability) noexcept {
  auto result = host;
  result.host_coherence = hal::HostCoherence::not_host_visible;
  result.mutability = mutability;
  return result;
}

bool ExactCachedEvidence(const OpenCudaVideoExecutionCachedEvidence &cached,
                         const video::VideoExecutionPlanRequest &request,
                         const hal::ResolutionResult &resolution,
                         const hal::DomainPlan &domain_plan) noexcept {
  if (!cached.model_artifact_validated || !cached.provider_runtime_validated ||
      !cached.device_context_validated || !cached.cuda_iobinding_validated ||
      !cached.user_stream_bound || !cached.reusable_resources_supported ||
      !cached.device_preprocess_supported ||
      !cached.device_postprocess_supported || !cached.one_upload_one_readback ||
      !cached.no_intermediate_crossings || !cached.stop_aware_bounded_wait ||
      !cached.allocation_free_execute ||
      !hal::ValidateExecutionDomain(cached.execution_domain) ||
      domain_plan.island_count != 1 ||
      cached.execution_domain != domain_plan.islands[0].execution_domain ||
      cached.execution_domain.backend != hal::BackendId::cuda ||
      cached.readiness.execution_domain != cached.execution_domain ||
      cached.readiness.requested_backend != hal::BackendId::cuda ||
      cached.readiness.selected_backend != hal::BackendId::cuda ||
      cached.readiness.provider != kProvider || !cached.readiness.available ||
      (cached.readiness.status != hal::StatusId::ready &&
       cached.readiness.status != hal::StatusId::degraded) ||
      // Readiness masks use prepared-stage slots, not stable stage_id values.
      // The only live Open CUDA route owns slot zero, so its exact mask is 1
      // even when the canonical stage_id is greater than 31.
      cached.readiness.ready_stage_mask != 1 ||
      cached.readiness.blocked_stage_mask != 0 ||
      cached.readiness.disabled_stage_mask != 0 ||
      resolution.selected_backend != hal::BackendId::cuda ||
      resolution.selected_provider != kProvider ||
      request.invalidation_key.generations != cached.generations ||
      request.invalidation_key.generations.context !=
          cached.execution_domain.logical_context_generation ||
      request.intent.stage_count != 1 ||
      !SameModel(request.intent.stages[0].model, cached.model)) {
    return false;
  }
  return true;
}

bool ConservativeRoute(const video::VideoExecutionPlanRequest &request,
                       const hal::ResolutionResult &resolution,
                       const hal::DomainPlan &domain_plan,
                       hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  if (request.intent.backend.mode !=
          hal::BackendRequestMode::explicit_backend ||
      request.intent.backend.backend != hal::BackendId::cuda ||
      !request.intent.backend.strict ||
      request.intent.preferences.effects_engine !=
          hal::EffectsEnginePreferenceConstraint::open_cuda ||
      !request.intent.preferences.strict_effects_engine ||
      (request.intent.preferences.compute_family !=
           hal::ComputeFamilyAllowance::any &&
       request.intent.preferences.compute_family !=
           hal::ComputeFamilyAllowance::cuda_family) ||
      request.intent.requested_provider != kProvider ||
      !request.intent.strict_provider_selection ||
      request.intent.stage_count != 1 || resolution.stage_count != 1 ||
      domain_plan.island_count != 1 ||
      domain_plan.islands[0].stage_count != 1 ||
      resolution.stages[0].skipped_optional ||
      resolution.stages[0].selected_backend != hal::BackendId::cuda ||
      resolution.stages[0].provider != kProvider) {
    SetReason(reason, hal::ReasonId::strict_backend_unavailable);
    return false;
  }
  const auto &intent = request.intent.stages[0].intent;
  if (intent.stage_id == 0 ||
      (intent.semantic != hal::SemanticEffectId::virtual_background_blur &&
       intent.semantic != hal::SemanticEffectId::virtual_background_remove) ||
      intent.requires_tracked_vignette_center ||
      (intent.required &&
       intent.failure_action != hal::FailureAction::fail_closed) ||
      (!intent.required &&
       intent.failure_action != hal::FailureAction::fail_open_preserve_input)) {
    SetReason(reason, hal::ReasonId::semantic_effect_unsupported);
    return false;
  }
  if (!TightRgb24(request.host_input, hal::Mutability::read_only) ||
      !TightRgb24(request.host_output, hal::Mutability::write_only) ||
      request.host_input.width != request.host_output.width ||
      request.host_input.height != request.host_output.height ||
      request.temporal.temporal ||
      request.temporal.requires_contiguous_capture_sequence ||
      request.cpu_tail_count != 1 ||
      request.cpu_tails[0].id != hal::CpuTailId::writer_copy ||
      request.cpu_tails[0].after_stage_id != intent.stage_id ||
      request.cpu_tails[0].required != true ||
      request.cpu_tails[0].placement !=
          hal::DeclaredCpuTail::Placement::after_stage ||
      request.cpu_tails[0].before_stage_id != 0 ||
      request.cpu_tails[0].boundary_index != 1 ||
      !domain_plan.islands[0].requires_upload ||
      !domain_plan.islands[0].requires_final_readback ||
      domain_plan.expected_transfers.uploads != 1 ||
      domain_plan.expected_transfers.final_readbacks != 1 ||
      domain_plan.expected_transfers.intermediate_uploads != 0 ||
      domain_plan.expected_transfers.intermediate_readbacks != 0 ||
      domain_plan.expected_transfers.cpu_tail_boundaries != 1) {
    SetReason(reason, hal::ReasonId::malformed_contract);
    return false;
  }
  return true;
}

hal::PreparedProgram
BuildProgram(const video::VideoExecutionPlanRequest &request,
             const hal::ResolutionResult &resolution,
             const hal::DomainPlan &domain_plan) noexcept {
  hal::PreparedProgram program;
  const auto &domain = domain_plan.islands[0].execution_domain;
  const auto &intent = request.intent.stages[0].intent;
  program.program_id = request.plan_id;
  program.owner_adapter_instance_id = domain.adapter_instance_id;
  program.execution_domain = domain;
  program.host_input = request.host_input;
  program.host_output = request.host_output;
  program.invalidation_key = request.invalidation_key;
  program.temporal = request.temporal;
  program.budget = request.budget;
  program.setup_failure_policy = request.setup_failure_policy;
  program.cpu_tails = request.cpu_tails;
  program.cpu_tail_count = request.cpu_tail_count;
  program.requested_backend = hal::BackendId::cuda;
  program.strict_backend_selection = true;
  program.requested_provider = kProvider;
  program.strict_provider_selection = true;
  program.requested_execution_domain = domain;
  program.strict_execution_domain = true;
  program.requires_upload = true;
  program.requires_final_readback = true;

  auto &stage = program.stages[0];
  stage.stage_id = intent.stage_id;
  stage.semantic = intent.semantic;
  stage.provider = resolution.stages[0].provider;
  stage.backend = resolution.stages[0].selected_backend;
  stage.execution_domain = domain;
  stage.input.kind = hal::StageIoKind::image;
  stage.input.image =
      ResidentImage(request.host_input, hal::Mutability::read_only);
  stage.output.kind = hal::StageIoKind::image;
  stage.output.image =
      ResidentImage(request.host_output, hal::Mutability::read_write);
  stage.model = request.intent.stages[0].model;
  stage.required = intent.required;
  stage.failure_action = intent.failure_action;
  stage.circuit_breaker.consecutive_failure_limit = 2;
  stage.circuit_breaker.cooldown_frames = 30;
  stage.circuit_breaker.action =
      intent.required ? hal::CircuitBreakerAction::quarantine_program
                      : hal::CircuitBreakerAction::disable_stage;
  program.stage_count = 1;
  return program;
}

bool ExactRegistration(const hal::AdapterRegistration &registration,
                       const video::FrozenVideoExecutionPlan &plan) noexcept {
  if (registration.adapter_instance_id !=
          plan.prepared_program.owner_adapter_instance_id ||
      registration.backend != hal::BackendId::cuda ||
      registration.execution_domain != plan.prepared_program.execution_domain ||
      registration.resident_adapter == nullptr ||
      registration.resident_adapter->adapter_instance_id() !=
          registration.adapter_instance_id ||
      registration.resident_adapter->backend_id() != hal::BackendId::cuda) {
    return false;
  }
  for (std::size_t index = 0; index < registration.semantic_capability_count;
       ++index) {
    const auto &capability = registration.semantic_capabilities[index];
    if (capability.semantic == plan.prepared_program.stages[0].semantic &&
        capability.provider == kProvider &&
        hal::IsSelectableSemanticCapability(capability)) {
      return true;
    }
  }
  return false;
}

bool ExactReadiness(const hal::ReadinessSnapshot &readiness,
                    const video::FrozenVideoExecutionPlan &plan) noexcept {
  return (readiness.status == hal::StatusId::ready ||
          readiness.status == hal::StatusId::degraded) &&
         readiness.available &&
         readiness.requested_backend == hal::BackendId::cuda &&
         readiness.selected_backend == hal::BackendId::cuda &&
         readiness.provider == kProvider &&
         readiness.execution_domain == plan.prepared_program.execution_domain &&
         readiness.ready_stage_mask == 1 && readiness.blocked_stage_mask == 0 &&
         readiness.disabled_stage_mask == 0;
}

class OpenCudaVideoExecutionExecutor final
    : public video::IVideoExecutionExecutor {
public:
  OpenCudaVideoExecutionExecutor(
      OpenCudaVideoExecutionCandidate candidate,
      std::unique_ptr<hal::PreparedProgramSession> session,
      std::uint64_t plan_id) noexcept
      : candidate_(std::move(candidate)), session_(std::move(session)),
        plan_id_(plan_id) {}

  ~OpenCudaVideoExecutionExecutor() override { Shutdown(); }

  video::VideoExecutionFrameResult
  Execute(const hal::HostFrameBinding &binding) noexcept override {
    video::VideoExecutionFrameResult frame;
    if (!session_ || !session_->prepared()) {
      frame.status = session_ && session_->quarantined()
                         ? hal::StatusId::quarantined
                         : hal::StatusId::stale;
      frame.reason = session_ && session_->quarantined()
                         ? hal::ReasonId::device_lost
                         : hal::ReasonId::stale_program;
      return frame;
    }
    hal::HostFrameTokens tokens;
    frame.status = session_->BindHostFrame(binding, &tokens, &frame.reason);
    if (frame.status != hal::StatusId::ready)
      return frame;

    hal::ExecuteRequest request;
    request.invalidation_key = session_->program().invalidation_key;
    request.execution_domain = session_->program().execution_domain;
    request.capture_sequence = binding.capture_sequence;
    request.input = tokens.input;
    const auto execution = session_->Execute(request);
    frame.status = execution.status;
    frame.reason = execution.reason;
    frame.completion = execution.completion;
    frame.execution_domain = session_->program().execution_domain;
    frame.stages = execution.stages;
    frame.stage_count = execution.stage_count;
    frame.transfers = execution.transfers;
    frame.frame_elapsed_ns = execution.frame_elapsed_ns;
    if (!execution.output_valid || !execution.completion.complete)
      return frame;

    hal::ReasonId resolution_reason = hal::ReasonId::none;
    frame.status =
        session_->ResolveHostFrameOutput(execution.completion, execution.output,
                                         &frame.output, &resolution_reason);
    frame.reason = resolution_reason;
    if (frame.status != hal::StatusId::ready || !frame.output.output_ready ||
        frame.output.capture_sequence != binding.capture_sequence) {
      hal::ReasonId ignored = hal::ReasonId::none;
      (void)session_->CommitOutput({execution.completion, false}, &ignored);
      frame.output = {};
      return frame;
    }
    frame.status = execution.status;
    frame.reason = execution.reason;
    frame.output_ready = true;
    return frame;
  }

  hal::StatusId CommitOutput(const hal::CompletionToken &completion,
                             bool output_write_succeeded,
                             hal::ReasonId *reason) noexcept override {
    if (!session_) {
      SetReason(reason, hal::ReasonId::stale_program);
      return hal::StatusId::stale;
    }
    return session_->CommitOutput({completion, output_write_succeeded}, reason);
  }

  void Invalidate(hal::ReasonId reason) noexcept override {
    if (session_)
      session_->Invalidate(reason);
  }

  hal::StatusId ResetTemporal(hal::ReasonId reason) noexcept override {
    (void)reason;
    if (!session_)
      return hal::StatusId::stale;
    if (session_->stopped())
      return hal::StatusId::stopped;
    if (session_->quarantined())
      return hal::StatusId::quarantined;
    // The first Open CUDA route rejects all temporal history at preflight, so
    // there is no resident history to reset.
    return session_->prepared() ? hal::StatusId::ready : hal::StatusId::stale;
  }

  void Shutdown() noexcept override {
    if (session_ && !session_->stopped())
      session_->Shutdown();
  }

  video::VideoExecutionAdapterSnapshot Snapshot() const noexcept override {
    video::VideoExecutionAdapterSnapshot snapshot;
    snapshot.plan_id = plan_id_;
    if (!session_) {
      snapshot.status = hal::StatusId::stale;
      snapshot.reason = hal::ReasonId::stale_program;
      return snapshot;
    }
    snapshot.readiness = session_->readiness();
    snapshot.telemetry = session_->telemetry();
    snapshot.latest_committed_frame = session_->latest_committed_frame();
    snapshot.prepared = session_->prepared();
    snapshot.stopped = session_->stopped();
    snapshot.execution_domain = snapshot.prepared
                                    ? session_->program().execution_domain
                                    : candidate_.registration.execution_domain;
    snapshot.status =
        snapshot.stopped ? hal::StatusId::stopped
                         : (session_->quarantined()
                                ? hal::StatusId::quarantined
                                : (snapshot.prepared ? snapshot.readiness.status
                                                     : hal::StatusId::stale));
    snapshot.reason = snapshot.stopped ? hal::ReasonId::shutdown_requested
                                       : (session_->quarantined()
                                              ? hal::ReasonId::device_lost
                                              : snapshot.readiness.reason);
    return snapshot;
  }

private:
  // Session must be destroyed before its candidate-owned adapter.
  OpenCudaVideoExecutionCandidate candidate_{};
  std::unique_ptr<hal::PreparedProgramSession> session_;
  std::uint64_t plan_id_ = 0;
};

} // namespace

OpenCudaVideoExecutionCandidate::~OpenCudaVideoExecutionCandidate() { Reset(); }

OpenCudaVideoExecutionCandidate::OpenCudaVideoExecutionCandidate(
    OpenCudaVideoExecutionCandidate &&other) noexcept
    : owner(std::exchange(other.owner, nullptr)),
      destroy_owner(std::exchange(other.destroy_owner, nullptr)),
      registration(other.registration) {
  other.registration = {};
}

OpenCudaVideoExecutionCandidate &OpenCudaVideoExecutionCandidate::operator=(
    OpenCudaVideoExecutionCandidate &&other) noexcept {
  if (this == &other)
    return *this;
  Reset();
  owner = std::exchange(other.owner, nullptr);
  destroy_owner = std::exchange(other.destroy_owner, nullptr);
  registration = other.registration;
  other.registration = {};
  return *this;
}

void OpenCudaVideoExecutionCandidate::Reset() noexcept {
  if (owner != nullptr && destroy_owner != nullptr)
    destroy_owner(owner);
  owner = nullptr;
  destroy_owner = nullptr;
  registration = {};
}

struct OpenCudaVideoExecutionFactory::Impl {
  explicit Impl(OpenCudaVideoExecutionFactoryConfig value) noexcept
      : config(std::move(value)) {}

  OpenCudaVideoExecutionFactoryConfig config{};
};

OpenCudaVideoExecutionFactory::OpenCudaVideoExecutionFactory(
    OpenCudaVideoExecutionFactoryConfig config) noexcept
    : impl_(new (std::nothrow) Impl(std::move(config))) {}

OpenCudaVideoExecutionFactory::~OpenCudaVideoExecutionFactory() = default;

video::VideoAdapterFactoryRegistration
OpenCudaVideoExecutionFactory::Registration() noexcept {
  video::VideoAdapterFactoryRegistration registration;
  if (!impl_)
    return registration;
  registration.factory_id = impl_->config.factory_id;
  registration.route = video::VideoExecutionRoute::prepared_resident;
  registration.backend = hal::BackendId::cuda;
  registration.provider = kProvider;
  registration.adapter_instance_id =
      impl_->config.cached.execution_domain.adapter_instance_id;
  registration.execution_domain = impl_->config.cached.execution_domain;
  registration.selection_priority = impl_->config.selection_priority;
  registration.factory = this;
  return registration;
}

video::VideoAdapterFactoryPreflight OpenCudaVideoExecutionFactory::Preflight(
    const video::VideoExecutionPlanRequest &request,
    const hal::ResolutionResult &frozen_resolution,
    const hal::DomainPlan &frozen_domain_plan,
    video::VideoExecutionRoute route) noexcept {
  video::VideoAdapterFactoryPreflight result;
  result.backend = hal::BackendId::cuda;
  result.provider = kProvider;
  result.cached_evidence_only = true;
  result.accepts_frozen_resolution = true;
  result.performs_selection = false;
  if (!impl_ || impl_->config.factory_id == 0 ||
      impl_->config.prepare_candidate == nullptr) {
    result.status = hal::StatusId::unavailable;
    result.reason = hal::ReasonId::capability_evidence_incomplete;
    return result;
  }
  result.execution_domain = impl_->config.cached.execution_domain;
  result.readiness = impl_->config.cached.readiness;
  if (route != video::VideoExecutionRoute::prepared_resident) {
    result.status = hal::StatusId::blocked;
    result.reason = hal::ReasonId::semantic_effect_unsupported;
    return result;
  }
  hal::ReasonId reason = hal::ReasonId::none;
  if (!ConservativeRoute(request, frozen_resolution, frozen_domain_plan,
                         &reason)) {
    result.status = hal::StatusId::blocked;
    result.reason = reason;
    return result;
  }
  if (!ExactCachedEvidence(impl_->config.cached, request, frozen_resolution,
                           frozen_domain_plan)) {
    result.status = hal::StatusId::blocked;
    result.reason = hal::ReasonId::capability_evidence_incomplete;
    return result;
  }
  result.prepared_program =
      BuildProgram(request, frozen_resolution, frozen_domain_plan);
  if (!hal::ValidatePreparedProgram(result.prepared_program, &reason)) {
    result.status = hal::StatusId::blocked;
    result.reason = reason;
    return result;
  }
  result.status = result.readiness.status;
  result.reason = result.readiness.reason;
  return result;
}

std::unique_ptr<video::IVideoExecutionExecutor>
OpenCudaVideoExecutionFactory::Prepare(
    const video::FrozenVideoExecutionPlan &plan,
    video::VideoAdapterPrepareEvidence *evidence) noexcept {
  video::VideoAdapterPrepareEvidence local;
  if (evidence == nullptr)
    evidence = &local;
  *evidence = {};
  if (!impl_ || impl_->config.prepare_candidate == nullptr ||
      plan.route != video::VideoExecutionRoute::prepared_resident ||
      plan.prepared_program.execution_domain !=
          impl_->config.cached.execution_domain ||
      plan.prepared_program.owner_adapter_instance_id !=
          impl_->config.cached.execution_domain.adapter_instance_id ||
      plan.prepared_program.requested_backend != hal::BackendId::cuda ||
      !plan.prepared_program.strict_backend_selection ||
      plan.prepared_program.requested_provider != kProvider ||
      !plan.prepared_program.strict_provider_selection ||
      plan.prepared_program.stage_count != 1 ||
      !SameModel(plan.prepared_program.stages[0].model,
                 impl_->config.cached.model) ||
      plan.prepared_program.invalidation_key.generations !=
          impl_->config.cached.generations) {
    evidence->status = hal::StatusId::blocked;
    evidence->reason = hal::ReasonId::malformed_contract;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }

  OpenCudaVideoExecutionCandidate candidate;
  auto status = impl_->config.prepare_candidate(impl_->config.setup_context,
                                                plan, &candidate, evidence);
  if (!IsSetupSuccess(status) || !IsSetupSuccess(evidence->status)) {
    if (evidence->status == hal::StatusId::invalid)
      evidence->status = status;
    if (evidence->reason == hal::ReasonId::none)
      evidence->reason = hal::ReasonId::capability_evidence_incomplete;
    candidate.Reset();
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }
  if (candidate.owner == nullptr || candidate.destroy_owner == nullptr ||
      !ExactRegistration(candidate.registration, plan)) {
    candidate.Reset();
    evidence->status = hal::StatusId::blocked;
    evidence->reason = hal::ReasonId::foreign_adapter;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }

  try {
    auto session = std::make_unique<hal::PreparedProgramSession>();
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    status = session->Configure(plan.prepared_program,
                                *candidate.registration.resident_adapter,
                                &readiness, &reason);
    if (!IsSetupSuccess(status) || !ExactReadiness(readiness, plan)) {
      session->Shutdown();
      candidate.Reset();
      evidence->status =
          IsSetupSuccess(status) ? hal::StatusId::blocked : status;
      evidence->reason = IsSetupSuccess(status)
                             ? hal::ReasonId::capability_evidence_incomplete
                             : reason;
      evidence->readiness = readiness;
      evidence->candidate_resources_rolled_back = true;
      return nullptr;
    }
    evidence->status = status;
    evidence->reason = reason;
    evidence->readiness = readiness;
    evidence->candidate_resources_rolled_back = false;
    return std::make_unique<OpenCudaVideoExecutionExecutor>(
        std::move(candidate), std::move(session), plan.plan_id);
  } catch (const std::bad_alloc &) {
    candidate.Reset();
    evidence->status = hal::StatusId::resource_exhausted;
    evidence->reason = hal::ReasonId::resource_exhausted;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  } catch (...) {
    candidate.Reset();
    evidence->status = hal::StatusId::blocked;
    evidence->reason = hal::ReasonId::capability_evidence_incomplete;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }
}

} // namespace studiocast::open_cuda
