#include "core/open_video/video_execution_factory.h"

#include <limits>
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

bool AddBudgetBytes(std::uint64_t left, std::uint64_t right,
                    std::uint64_t *result) noexcept {
  if (!result || right > std::numeric_limits<std::uint64_t>::max() - left)
    return false;
  *result = left + right;
  return true;
}

bool ValidReservedBudget(const hal::ResourceBudget &budget) noexcept {
  if (budget.resource_count == 0 || budget.in_flight_slots != 1)
    return false;
  std::uint64_t total = 0;
  return AddBudgetBytes(total, budget.persistent_bytes, &total) &&
         AddBudgetBytes(total, budget.scratch_bytes, &total) &&
         AddBudgetBytes(total, budget.upload_staging_bytes, &total) &&
         AddBudgetBytes(total, budget.readback_staging_bytes, &total) &&
         total != 0;
}

bool SameBudget(const hal::ResourceBudget &left,
                const hal::ResourceBudget &right) noexcept {
  return left.persistent_bytes == right.persistent_bytes &&
         left.scratch_bytes == right.scratch_bytes &&
         left.upload_staging_bytes == right.upload_staging_bytes &&
         left.readback_staging_bytes == right.readback_staging_bytes &&
         left.resource_count == right.resource_count &&
         left.in_flight_slots == right.in_flight_slots;
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
  const auto transfer_contract = video::BuildPerFrameTransferContract(
      domain_plan, cached.minimum_synchronizations_per_frame,
      cached.maximum_synchronizations_per_frame);
  hal::ReasonId transfer_reason = hal::ReasonId::none;
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
      (ValidReservedBudget(cached.reserved_budget) &&
       !SameBudget(request.budget, cached.reserved_budget)) ||
      request.invalidation_key.generations.context !=
          cached.execution_domain.logical_context_generation ||
      request.intent.stage_count != 1 ||
      !SameModel(request.intent.stages[0].model, cached.model) ||
      !video::ValidatePerFrameTransferContract(transfer_contract, domain_plan,
                                               &transfer_reason)) {
    return false;
  }
  return true;
}

bool ConservativeRoute(const video::VideoExecutionPlanRequest &request,
                       const hal::ResolutionResult &resolution,
                       const hal::DomainPlan &domain_plan,
                       hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  const bool automatic_request =
      request.intent.backend.mode == hal::BackendRequestMode::automatic &&
      request.intent.backend.backend == hal::BackendId::invalid &&
      !request.intent.backend.strict &&
      request.intent.preferences.compute_family ==
          hal::ComputeFamilyAllowance::any &&
      request.intent.preferences.effects_engine ==
          hal::EffectsEnginePreferenceConstraint::automatic &&
      !request.intent.preferences.strict_effects_engine &&
      request.intent.requested_provider ==
          hal::SemanticEffectProviderId::invalid &&
      !request.intent.strict_provider_selection;
  const bool strict_request =
      request.intent.backend.mode ==
          hal::BackendRequestMode::explicit_backend &&
      request.intent.backend.backend == hal::BackendId::cuda &&
      request.intent.backend.strict &&
      request.intent.preferences.effects_engine ==
          hal::EffectsEnginePreferenceConstraint::open_cuda &&
      request.intent.preferences.strict_effects_engine &&
      request.intent.requested_provider == kProvider &&
      request.intent.strict_provider_selection;
  if ((!automatic_request && !strict_request) ||
      (request.intent.preferences.compute_family !=
           hal::ComputeFamilyAllowance::any &&
       request.intent.preferences.compute_family !=
           hal::ComputeFamilyAllowance::cuda_family) ||
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
       intent.semantic != hal::SemanticEffectId::virtual_background_remove &&
       intent.semantic != hal::SemanticEffectId::virtual_background_replace &&
       intent.semantic != hal::SemanticEffectId::virtual_key_light) ||
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
  program.requested_backend =
      request.intent.backend.mode == hal::BackendRequestMode::explicit_backend
          ? request.intent.backend.backend
          : resolution.selected_backend;
  program.strict_backend_selection = request.intent.backend.strict;
  program.requested_provider =
      request.intent.requested_provider !=
              hal::SemanticEffectProviderId::invalid
          ? request.intent.requested_provider
          : resolution.selected_provider;
  program.strict_provider_selection = request.intent.strict_provider_selection;
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
    if (candidate_.owner != nullptr &&
        candidate_.observe_runtime_owner != nullptr) {
      candidate_.observe_runtime_owner(candidate_.owner, execution.status,
                                       execution.reason);
    }
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
    if (candidate_.owner != nullptr && candidate_.shutdown_owner != nullptr)
      candidate_.shutdown_owner(candidate_.owner);
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
      commit_prepared_owner(
          std::exchange(other.commit_prepared_owner, nullptr)),
      observe_runtime_owner(
          std::exchange(other.observe_runtime_owner, nullptr)),
      shutdown_owner(std::exchange(other.shutdown_owner, nullptr)),
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
  commit_prepared_owner = std::exchange(other.commit_prepared_owner, nullptr);
  observe_runtime_owner = std::exchange(other.observe_runtime_owner, nullptr);
  shutdown_owner = std::exchange(other.shutdown_owner, nullptr);
  registration = other.registration;
  other.registration = {};
  return *this;
}

void OpenCudaVideoExecutionCandidate::Reset() noexcept {
  if (owner != nullptr && shutdown_owner != nullptr)
    shutdown_owner(owner);
  if (owner != nullptr && destroy_owner != nullptr)
    destroy_owner(owner);
  owner = nullptr;
  destroy_owner = nullptr;
  commit_prepared_owner = nullptr;
  observe_runtime_owner = nullptr;
  shutdown_owner = nullptr;
  registration = {};
}

struct OpenCudaVideoExecutionFactory::Impl {
  explicit Impl(OpenCudaVideoExecutionFactoryConfig value) noexcept
      : config(std::move(value)) {}

  struct ProductionLease {
    OpenCudaProductionHalOwner *owner = nullptr;
  };

  class BorrowedResidentAdapter final : public hal::IResidentAdapter {
  public:
    explicit BorrowedResidentAdapter(
        OpenCudaProductionHalOwner *owner) noexcept
        : owner_(owner),
          adapter_instance_id_(owner && owner->adapter()
                                   ? owner->adapter()->adapter_instance_id()
                                   : 0) {}

    std::uint64_t adapter_instance_id() const noexcept override {
      return adapter_instance_id_;
    }
    hal::BackendId backend_id() const noexcept override {
      return adapter_instance_id_ != 0 ? hal::BackendId::cuda
                                       : hal::BackendId::invalid;
    }
    hal::StatusId Prepare(const hal::PreparedProgram &program,
                          hal::PreparedProgramHandle *candidate,
                          hal::ReadinessSnapshot *readiness,
                          hal::ReasonId *reason) noexcept override {
      auto *adapter = NativeAdapter();
      if (!adapter)
        return Stopped(readiness, reason);
      return adapter->Prepare(program, candidate, readiness, reason);
    }
    hal::ExecuteResult
    Execute(const hal::PreparedProgramHandle &program,
            const hal::ExecuteRequest &request) noexcept override {
      auto *adapter = NativeAdapter();
      if (adapter)
        return adapter->Execute(program, request);
      hal::ExecuteResult result;
      result.status = hal::StatusId::stopped;
      result.reason = hal::ReasonId::shutdown_requested;
      return result;
    }
    hal::StatusId BindHostFrame(const hal::PreparedProgramHandle &program,
                                const hal::HostFrameBinding &binding,
                                hal::HostFrameTokens *tokens,
                                hal::ReasonId *reason) noexcept override {
      auto *adapter = NativeAdapter();
      if (!adapter)
        return Stopped(nullptr, reason);
      return adapter->BindHostFrame(program, binding, tokens, reason);
    }
    hal::StatusId
    ResolveHostFrameOutput(const hal::PreparedProgramHandle &program,
                           const hal::CompletionToken &completion,
                           const hal::ResourceHandle &output,
                           hal::HostFrameOutputResolution *resolution,
                           hal::ReasonId *reason) noexcept override {
      auto *adapter = NativeAdapter();
      if (!adapter)
        return Stopped(nullptr, reason);
      return adapter->ResolveHostFrameOutput(program, completion, output,
                                             resolution, reason);
    }
    hal::StatusId ResetTemporalState(const hal::PreparedProgramHandle &program,
                                     hal::ReasonId reason) noexcept override {
      auto *adapter = NativeAdapter();
      return adapter ? adapter->ResetTemporalState(program, reason)
                     : hal::StatusId::stopped;
    }
    void
    DestroyPrepared(hal::PreparedProgramHandle *program) noexcept override {
      if (auto *adapter = NativeAdapter())
        adapter->DestroyPrepared(program);
    }
    void Shutdown() noexcept override {
      // The registry owns this view for the factory lifetime. Native adapter
      // shutdown is deferred until factory retirement.
    }

    void Retire() noexcept { retired_ = true; }

  private:
    hal::IResidentAdapter *NativeAdapter() const noexcept {
      return !retired_ && owner_ ? owner_->adapter() : nullptr;
    }

    static hal::StatusId
    Stopped(hal::ReadinessSnapshot *readiness,
            hal::ReasonId *reason) noexcept {
      if (readiness) {
        readiness->status = hal::StatusId::stopped;
        readiness->reason = hal::ReasonId::shutdown_requested;
      }
      SetReason(reason, hal::ReasonId::shutdown_requested);
      return hal::StatusId::stopped;
    }

    OpenCudaProductionHalOwner *owner_ = nullptr;
    std::uint64_t adapter_instance_id_ = 0;
    bool retired_ = false;
  };

  static void DestroyProductionLease(void *opaque) noexcept {
    delete static_cast<ProductionLease *>(opaque);
  }

  static hal::StatusId
  CommitProductionOwner(void *opaque, const hal::PreparedProgram &program,
                        const hal::ReadinessSnapshot &readiness,
                        hal::ReasonId *reason) noexcept {
    auto *lease = static_cast<ProductionLease *>(opaque);
    auto *owner = lease ? lease->owner : nullptr;
    if (!owner) {
      SetReason(reason, hal::ReasonId::stale_program);
      return hal::StatusId::stale;
    }
    const auto committed = owner->CommitPreparedCapability(program, readiness);
    SetReason(reason, committed.reason);
    return committed.prepared_capability_committed
               ? committed.hal_status
               : (committed.hal_status == hal::StatusId::invalid
                      ? hal::StatusId::blocked
                      : committed.hal_status);
  }

  static void ObserveProductionOwner(void *opaque, hal::StatusId status,
                                     hal::ReasonId reason) noexcept {
    auto *lease = static_cast<ProductionLease *>(opaque);
    if (lease && lease->owner)
      lease->owner->ObserveRuntimeStatus(status, reason);
  }

  static void ShutdownProductionLease(void *) noexcept {
    // PreparedProgramSession::Shutdown destroys only this executor's prepared
    // slot. The factory retains the native adapter because the registry also
    // retains its pointer. Factory retirement performs native shutdown after
    // the coordinator has retired all executors.
  }

  static hal::StatusId PrepareProductionCandidate(
      void *opaque, const video::FrozenVideoExecutionPlan &plan,
      OpenCudaVideoExecutionCandidate *candidate,
      video::VideoAdapterPrepareEvidence *evidence) noexcept {
    auto *self = static_cast<Impl *>(opaque);
    if (!self || !candidate || !evidence || !self->production_owner) {
      if (evidence) {
        evidence->status = hal::StatusId::unavailable;
        evidence->reason = hal::ReasonId::stale_program;
        evidence->candidate_resources_rolled_back = true;
      }
      return hal::StatusId::unavailable;
    }
    if (plan.prepared_program.execution_domain !=
        self->production_registration.execution_domain) {
      evidence->status = hal::StatusId::blocked;
      evidence->reason = hal::ReasonId::wrong_execution_domain;
      evidence->candidate_resources_rolled_back = true;
      return evidence->status;
    }
    if (plan.prepared_program.owner_adapter_instance_id !=
        self->production_registration.adapter_instance_id) {
      evidence->status = hal::StatusId::blocked;
      evidence->reason = hal::ReasonId::foreign_adapter;
      evidence->candidate_resources_rolled_back = true;
      return evidence->status;
    }
    if (plan.prepared_program.invalidation_key.generations !=
        self->config.cached.generations) {
      evidence->status = hal::StatusId::stale;
      evidence->reason = hal::ReasonId::wrong_generation;
      evidence->candidate_resources_rolled_back = true;
      return evidence->status;
    }
    if (!SameBudget(plan.prepared_program.budget,
                    self->config.cached.reserved_budget)) {
      evidence->status = hal::StatusId::resource_exhausted;
      evidence->reason = hal::ReasonId::resource_exhausted;
      evidence->candidate_resources_rolled_back = true;
      return evidence->status;
    }
    if (plan.prepared_program.stage_count != 1 ||
        plan.prepared_program.stages[0].semantic != self->production_semantic) {
      evidence->status = hal::StatusId::blocked;
      evidence->reason = hal::ReasonId::semantic_effect_unsupported;
      evidence->candidate_resources_rolled_back = true;
      return evidence->status;
    }
    if (!SameModel(plan.prepared_program.stages[0].model,
                   self->config.cached.model)) {
      evidence->status = hal::StatusId::blocked;
      evidence->reason = hal::ReasonId::capability_evidence_incomplete;
      evidence->candidate_resources_rolled_back = true;
      return evidence->status;
    }

    auto *lease =
        new (std::nothrow) ProductionLease{self->production_owner.get()};
    if (!lease) {
      evidence->status = hal::StatusId::resource_exhausted;
      evidence->reason = hal::ReasonId::resource_exhausted;
      evidence->candidate_resources_rolled_back = true;
      return evidence->status;
    }
    candidate->owner = lease;
    candidate->destroy_owner = DestroyProductionLease;
    candidate->commit_prepared_owner = CommitProductionOwner;
    candidate->observe_runtime_owner = ObserveProductionOwner;
    candidate->shutdown_owner = ShutdownProductionLease;
    candidate->registration = self->production_registration;
    evidence->status = hal::StatusId::ready;
    evidence->reason = hal::ReasonId::none;
    evidence->candidate_resources_rolled_back = false;
    return evidence->status;
  }

  OpenCudaVideoExecutionFactoryConfig config{};
  std::unique_ptr<OpenCudaProductionHalOwner> production_owner;
  std::unique_ptr<BorrowedResidentAdapter> production_adapter_view;
  hal::AdapterRegistration production_registration{};
  hal::SemanticEffectId production_semantic = hal::SemanticEffectId::invalid;
  bool production_shutdown_requested = false;
  bool production_owner_retired = false;
};

class OpenCudaProductionSetupLeaseOwner final
    : public video::IProductionVideoSetupLeaseOwner {
public:
  explicit OpenCudaProductionSetupLeaseOwner(
      std::unique_ptr<OpenCudaVideoExecutionFactory> factory) noexcept
      : factory_(std::move(factory)) {
    if (factory_) {
      resident_adapter_ =
          factory_->adapter_registration().resident_adapter;
      video_factory_ = factory_.get();
    }
  }

  hal::IResidentAdapter *resident_adapter() const noexcept override {
    return resident_adapter_;
  }

  video::IVideoAdapterFactory *
  video_adapter_factory() const noexcept override {
    return video_factory_;
  }

  void Shutdown() noexcept override {
    (void)RequestShutdown();
  }

  video::ProductionVideoSetupLeaseShutdownResult
  RequestShutdown() noexcept override {
    video::ProductionVideoSetupLeaseShutdownResult result;
    result.shutdown_requested = true;
    if (!factory_) {
      result.status = hal::StatusId::stopped;
      result.reason = hal::ReasonId::shutdown_requested;
      result.quiescent = true;
      result.driver_context_release_permitted = true;
      return result;
    }

    const auto native = factory_->RequestProductionOwnerShutdown();
    result.shutdown_attempts = native.shutdown_attempts;
    result.pending = !native.driver_context_release_permitted;
    result.quiescent =
        native.driver_context_release_permitted && native.quiescent;
    result.driver_context_release_permitted =
        native.driver_context_release_permitted;
    if (result.driver_context_release_permitted) {
      result.status = hal::StatusId::stopped;
      result.reason = hal::ReasonId::shutdown_requested;
    } else {
      result.status = native.hal_status;
      result.reason = native.reason;
    }
    return result;
  }

private:
  std::unique_ptr<OpenCudaVideoExecutionFactory> factory_;
  hal::IResidentAdapter *resident_adapter_ = nullptr;
  video::IVideoAdapterFactory *video_factory_ = nullptr;
};

OpenCudaVideoExecutionFactory::OpenCudaVideoExecutionFactory(
    OpenCudaVideoExecutionFactoryConfig config) noexcept
    : impl_(new (std::nothrow) Impl(std::move(config))) {}

OpenCudaVideoExecutionFactory::~OpenCudaVideoExecutionFactory() = default;

std::unique_ptr<OpenCudaVideoExecutionFactory>
OpenCudaVideoExecutionFactory::CreateProduction(
    const OpenCudaProductionVideoExecutionFactoryRequest &request,
    OpenCudaProductionVideoExecutionBootstrapResult *result) noexcept {
  OpenCudaProductionVideoExecutionBootstrapResult local;
  if (!result)
    result = &local;
  *result = {};
  const bool supported_semantic =
      request.semantic == hal::SemanticEffectId::virtual_background_blur ||
      request.semantic == hal::SemanticEffectId::virtual_background_remove ||
      request.semantic == hal::SemanticEffectId::virtual_background_replace ||
      request.semantic == hal::SemanticEffectId::virtual_key_light;
  const bool replacement_contract =
      request.semantic != hal::SemanticEffectId::virtual_background_replace ||
      (request.backend.replacement_width > 0 &&
       request.backend.replacement_height > 0 &&
       static_cast<std::uint64_t>(request.backend.replacement_width) *
               static_cast<std::uint64_t>(request.backend.replacement_height) *
               3u ==
           request.backend.replacement_rgb.size());
  const bool key_light_contract =
      request.semantic != hal::SemanticEffectId::virtual_key_light ||
      (request.backend.key_light_intensity >= 0 &&
       request.backend.key_light_intensity <= 100 &&
       request.backend.key_light_temperature_preset >= 0 &&
       request.backend.key_light_temperature_preset <= 2 &&
       request.backend.key_light_direction_pan_degrees >= -90 &&
       request.backend.key_light_direction_pan_degrees <= 90);
  const bool exact_model =
      request.backend.model_artifact.Valid() &&
      request.model.model_id_hash ==
          request.backend.model_artifact.model_id_sha256 &&
      request.model.artifact_set_hash ==
          request.backend.model_artifact.artifact_set_sha256 &&
      request.model.artifact_generation ==
          request.backend.model_artifact.registry_generation;
  if (request.factory_id == 0 || !supported_semantic ||
      !replacement_contract || !key_light_contract || !exact_model ||
      !ValidReservedBudget(request.reserved_budget) ||
      request.generations.configuration == 0 ||
      request.generations.geometry == 0 || request.generations.model == 0 ||
      request.generations.provider == 0 || request.generations.device == 0 ||
      request.backend.configuration_generation !=
          request.generations.configuration ||
      request.backend.model_preprocess_generation !=
          request.generations.model ||
      !request.model_artifact_validated ||
      !request.provider_runtime_validated ||
      !request.cuda_iobinding_validated || !request.user_stream_bound ||
      !request.reusable_resources_supported ||
      !request.device_preprocess_supported ||
      !request.device_postprocess_supported ||
      !request.stop_aware_bounded_wait || !request.allocation_free_execute) {
    result->status = hal::StatusId::blocked;
    result->reason = hal::ReasonId::capability_evidence_incomplete;
    return nullptr;
  }

  auto owner = OpenCudaProductionHalOwner::CreateBackend(request.backend,
                                                         &result->owner);
  if (!owner || !owner->bootstrap_domain_ready() || !owner->adapter()) {
    result->status = result->owner.hal_status;
    result->reason = result->owner.reason;
    return nullptr;
  }

  try {
    OpenCudaVideoExecutionFactoryConfig config;
    config.factory_id = request.factory_id;
    config.selection_priority = request.selection_priority;
    config.cached.execution_domain = owner->execution_domain();
    config.cached.readiness.status = hal::StatusId::ready;
    config.cached.readiness.requested_backend = hal::BackendId::cuda;
    config.cached.readiness.selected_backend = hal::BackendId::cuda;
    config.cached.readiness.provider = kProvider;
    config.cached.readiness.execution_domain = owner->execution_domain();
    config.cached.readiness.ready_stage_mask = 1;
    config.cached.readiness.available = true;
    config.cached.model = request.model;
    config.cached.generations = request.generations;
    config.cached.reserved_budget = request.reserved_budget;
    config.cached.generations.context =
        owner->execution_domain().logical_context_generation;
    config.cached.model_artifact_validated = true;
    config.cached.provider_runtime_validated = true;
    config.cached.device_context_validated = true;
    config.cached.cuda_iobinding_validated = true;
    config.cached.user_stream_bound = true;
    config.cached.reusable_resources_supported = true;
    config.cached.device_preprocess_supported = true;
    config.cached.device_postprocess_supported = true;
    config.cached.one_upload_one_readback = true;
    config.cached.no_intermediate_crossings = true;
    config.cached.stop_aware_bounded_wait = true;
    config.cached.allocation_free_execute = true;
    config.cached.minimum_synchronizations_per_frame = 1;
    config.cached.maximum_synchronizations_per_frame = 1;

    auto factory = std::unique_ptr<OpenCudaVideoExecutionFactory>(
        new OpenCudaVideoExecutionFactory(std::move(config)));
    if (!factory->impl_) {
      result->status = hal::StatusId::resource_exhausted;
      result->reason = hal::ReasonId::resource_exhausted;
      return nullptr;
    }
    auto &impl = *factory->impl_;
    impl.production_owner = std::move(owner);
    impl.production_adapter_view =
        std::make_unique<Impl::BorrowedResidentAdapter>(
            impl.production_owner.get());
    impl.production_semantic = request.semantic;
    impl.production_registration.adapter_instance_id =
        impl.config.cached.execution_domain.adapter_instance_id;
    impl.production_registration.backend = hal::BackendId::cuda;
    impl.production_registration.execution_domain =
        impl.config.cached.execution_domain;
    impl.production_registration.selection_priority =
        request.selection_priority;
    impl.production_registration.resident_adapter =
        impl.production_adapter_view.get();
    auto &capability = impl.production_registration.semantic_capabilities[0];
    capability.semantic = request.semantic;
    capability.provider = kProvider;
    capability.status = hal::StatusId::ready;
    capability.reason = hal::ReasonId::none;
    capability.cpu_tail = hal::CpuTailId::writer_copy;
    capability.cpu_tail_placement =
        hal::DeclaredCpuTail::Placement::after_stage;
    capability.live_stage = true;
    capability.production_eligible = true;
    capability.exact_evidence = true;
    capability.evidence[0] = {
        hal::CapabilityEvidenceCode::provider_runtime_available,
        hal::StatusId::ready,
        hal::ReasonId::none,
        request.semantic,
        hal::BackendId::cuda,
        true};
    capability.evidence[1] = {
        hal::CapabilityEvidenceCode::model_artifact_identity_valid,
        hal::StatusId::ready,
        hal::ReasonId::none,
        request.semantic,
        hal::BackendId::cuda,
        true};
    capability.evidence[2] = {
        hal::CapabilityEvidenceCode::synchronization_evidence_valid,
        hal::StatusId::ready,
        hal::ReasonId::none,
        request.semantic,
        hal::BackendId::cuda,
        true};
    capability.evidence_count = 3;
    impl.production_registration.semantic_capability_count = 1;
    impl.config.prepare_candidate = Impl::PrepareProductionCandidate;
    impl.config.setup_context = &impl;
    result->status = hal::StatusId::ready;
    result->reason = hal::ReasonId::none;
    result->exact_candidate_ready = true;
    auto &snapshot = result->setup_lease;
    snapshot.contract_version =
        video::kProductionVideoSetupLeaseContractVersion;
    snapshot.state = video::ProductionVideoSetupLeaseState::live;
    snapshot.status = impl.config.cached.readiness.status;
    snapshot.reason = impl.config.cached.readiness.reason;
    snapshot.readiness = impl.config.cached.readiness;
    snapshot.backend = hal::BackendId::cuda;
    snapshot.execution_domain = impl.config.cached.execution_domain;
    snapshot.generations = impl.config.cached.generations;
    snapshot.reserved_budget = impl.config.cached.reserved_budget;
    snapshot.providers[0] = kProvider;
    snapshot.provider_count = 1;
    snapshot.artifacts[0].semantic = request.semantic;
    snapshot.artifacts[0].provider = kProvider;
    snapshot.artifacts[0].artifact = impl.config.cached.model;
    snapshot.artifact_count = 1;
    snapshot.adapter = factory->adapter_registration();
    snapshot.factory = factory->Registration();
    return factory;
  } catch (const std::bad_alloc &) {
    result->status = hal::StatusId::resource_exhausted;
    result->reason = hal::ReasonId::resource_exhausted;
  } catch (...) {
    result->status = hal::StatusId::blocked;
    result->reason = hal::ReasonId::capability_evidence_incomplete;
  }
  return nullptr;
}

hal::StatusId OpenCudaVideoExecutionFactory::AcquireProductionSetupLease(
    const OpenCudaProductionVideoExecutionFactoryRequest &request,
    video::ProductionVideoSetupLease *destination,
    OpenCudaProductionVideoExecutionBootstrapResult *bootstrap,
    video::ProductionVideoSetupLeaseAcquireEvidence *evidence,
    hal::ReasonId *reason) noexcept {
  OpenCudaProductionVideoExecutionBootstrapResult local_bootstrap;
  if (!bootstrap)
    bootstrap = &local_bootstrap;
  if (evidence)
    *evidence = {};
  SetReason(reason, hal::ReasonId::none);

  if (!ValidReservedBudget(request.reserved_budget)) {
    *bootstrap = {};
    bootstrap->status = hal::StatusId::resource_exhausted;
    bootstrap->reason = hal::ReasonId::resource_exhausted;
    SetReason(reason, bootstrap->reason);
    if (evidence) {
      evidence->status = bootstrap->status;
      evidence->reason = bootstrap->reason;
      evidence->readiness = bootstrap->setup_lease.readiness;
      evidence->previous_lease_retained =
          destination != nullptr && destination->live();
    }
    return bootstrap->status;
  }

  auto factory = CreateProduction(request, bootstrap);
  if (!factory) {
    SetReason(reason, bootstrap->reason);
    if (evidence) {
      evidence->status = bootstrap->status;
      evidence->reason = bootstrap->reason;
      evidence->readiness = bootstrap->setup_lease.readiness;
      evidence->previous_lease_retained =
          destination != nullptr && destination->live();
    }
    return bootstrap->status;
  }

  std::unique_ptr<video::IProductionVideoSetupLeaseOwner> owner;
  try {
    owner = std::make_unique<OpenCudaProductionSetupLeaseOwner>(
        std::move(factory));
  } catch (const std::bad_alloc &) {
    bootstrap->status = hal::StatusId::resource_exhausted;
    bootstrap->reason = hal::ReasonId::resource_exhausted;
    SetReason(reason, bootstrap->reason);
    if (evidence) {
      evidence->status = bootstrap->status;
      evidence->reason = bootstrap->reason;
      evidence->readiness = bootstrap->setup_lease.readiness;
      evidence->candidate_owner_rolled_back = true;
      evidence->previous_lease_retained =
          destination != nullptr && destination->live();
    }
    return bootstrap->status;
  } catch (...) {
    bootstrap->status = hal::StatusId::blocked;
    bootstrap->reason = hal::ReasonId::capability_evidence_incomplete;
    SetReason(reason, bootstrap->reason);
    if (evidence) {
      evidence->status = bootstrap->status;
      evidence->reason = bootstrap->reason;
      evidence->readiness = bootstrap->setup_lease.readiness;
      evidence->candidate_owner_rolled_back = true;
      evidence->previous_lease_retained =
          destination != nullptr && destination->live();
    }
    return bootstrap->status;
  }

  const auto status = video::ProductionVideoSetupLease::Acquire(
      bootstrap->setup_lease, std::move(owner), destination, evidence, reason);
  if (status != hal::StatusId::ready &&
      status != hal::StatusId::degraded) {
    bootstrap->status = status;
    bootstrap->reason = reason ? *reason : hal::ReasonId::none;
  }
  return status;
}

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

hal::AdapterRegistration
OpenCudaVideoExecutionFactory::adapter_registration() const noexcept {
  return impl_ ? impl_->production_registration : hal::AdapterRegistration{};
}

OpenCudaProductionShutdownResult
OpenCudaVideoExecutionFactory::RequestProductionOwnerShutdown() noexcept {
  OpenCudaProductionShutdownResult result;
  if (!impl_ || !impl_->production_owner) {
    result.status = OpenCudaProductionOwnerStatus::stopped;
    result.hal_status = hal::StatusId::stopped;
    result.reason = hal::ReasonId::shutdown_requested;
    result.shutdown_requested = true;
    result.quiescent = true;
    result.driver_context_release_permitted = true;
    return result;
  }
  if (impl_->production_owner_retired)
    return impl_->production_owner->ShutdownSnapshot();

  impl_->production_shutdown_requested = true;
  result = impl_->production_owner->RequestShutdown();
  if (result.driver_context_release_permitted) {
    if (impl_->production_adapter_view)
      impl_->production_adapter_view->Retire();
    impl_->production_owner_retired = true;
  }
  return result;
}

OpenCudaProductionShutdownResult
OpenCudaVideoExecutionFactory::production_shutdown_snapshot() const noexcept {
  if (!impl_ || !impl_->production_owner) {
    OpenCudaProductionShutdownResult result;
    result.status = OpenCudaProductionOwnerStatus::stopped;
    result.hal_status = hal::StatusId::stopped;
    result.reason = hal::ReasonId::shutdown_requested;
    result.shutdown_requested = true;
    result.quiescent = true;
    result.driver_context_release_permitted = true;
    return result;
  }
  return impl_->production_owner->ShutdownSnapshot();
}

bool OpenCudaVideoExecutionFactory::production_owner_retired() const noexcept {
  return impl_ && impl_->production_owner_retired;
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
  result.per_frame_transfers = video::BuildPerFrameTransferContract(
      frozen_domain_plan,
      impl_->config.cached.minimum_synchronizations_per_frame,
      impl_->config.cached.maximum_synchronizations_per_frame);
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
  if (!impl_ || impl_->config.prepare_candidate == nullptr) {
    evidence->status = hal::StatusId::unavailable;
    evidence->reason = hal::ReasonId::capability_evidence_incomplete;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }
  if (plan.route != video::VideoExecutionRoute::prepared_resident ||
      plan.prepared_program.requested_backend != hal::BackendId::cuda ||
      plan.prepared_program.requested_provider != kProvider ||
      !video::ValidatePerFrameTransferContract(plan.per_frame_transfers,
                                               plan.domain_plan, nullptr) ||
      plan.prepared_program.stage_count != 1) {
    evidence->status = hal::StatusId::blocked;
    evidence->reason = hal::ReasonId::malformed_contract;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }
  if (plan.prepared_program.execution_domain !=
      impl_->config.cached.execution_domain) {
    evidence->status = hal::StatusId::blocked;
    evidence->reason = hal::ReasonId::wrong_execution_domain;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }
  if (plan.prepared_program.owner_adapter_instance_id !=
      impl_->config.cached.execution_domain.adapter_instance_id) {
    evidence->status = hal::StatusId::blocked;
    evidence->reason = hal::ReasonId::foreign_adapter;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }
  if (plan.prepared_program.invalidation_key.generations !=
      impl_->config.cached.generations) {
    evidence->status = hal::StatusId::stale;
    evidence->reason = hal::ReasonId::wrong_generation;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }
  if (!SameModel(plan.prepared_program.stages[0].model,
                 impl_->config.cached.model)) {
    evidence->status = hal::StatusId::blocked;
    evidence->reason = hal::ReasonId::capability_evidence_incomplete;
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
    if (candidate.commit_prepared_owner != nullptr) {
      status = candidate.commit_prepared_owner(
          candidate.owner, plan.prepared_program, readiness, &reason);
      if (!IsSetupSuccess(status)) {
        session->Shutdown();
        candidate.Reset();
        evidence->status = status;
        evidence->reason = reason == hal::ReasonId::none
                               ? hal::ReasonId::capability_evidence_incomplete
                               : reason;
        evidence->readiness = readiness;
        evidence->candidate_resources_rolled_back = true;
        return nullptr;
      }
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
