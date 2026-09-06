#include "core/open_video/production_video_backend_setup_source.h"

#include <cstdlib>
#include <array>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using namespace studiocast;

constexpr std::uint64_t kFakeAdapter = UINT64_C(0xc0da0001);
constexpr std::uint64_t kFakeFactory = UINT64_C(0xc0da0002);

class UniqueTempDir {
public:
  UniqueTempDir() {
    path_ = (std::filesystem::temp_directory_path() /
             "studiocast-open-cuda-XXXXXX").string();
    if (::mkdtemp(path_.data()) == nullptr)
      throw std::runtime_error("mkdtemp failed");
  }
  ~UniqueTempDir() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  const std::string &path() const noexcept { return path_; }
private:
  std::string path_;
};

hal::ExecutionDomainId FakeCudaDomain() {
  hal::ExecutionDomainId domain;
  domain.backend = hal::BackendId::cuda;
  domain.api_family = hal::ApiFamily::cuda_driver;
  domain.physical_device.vendor_id = 0x10de;
  domain.physical_device.device_id = 1;
  domain.physical_device.stable_id[0] = 0x42;
  domain.logical_context_generation = 7;
  domain.allocator_family = hal::AllocatorFamily::cuda_driver;
  domain.memory_handle_family = hal::MemoryHandleFamily::cuda_device_pointer;
  domain.queue_or_stream_identity = 0x99;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::cuda_stream_ordered;
  domain.adapter_instance_id = kFakeAdapter;
  return domain;
}

class FakeAdapter final : public hal::IResidentAdapter {
public:
  std::uint64_t adapter_instance_id() const noexcept override {
    return kFakeAdapter;
  }
  hal::BackendId backend_id() const noexcept override {
    return hal::BackendId::cuda;
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
  explicit FakeExecutor(const video::FrozenVideoExecutionPlan &plan)
      : plan_(plan) {
    snapshot_.status = hal::StatusId::ready;
    snapshot_.prepared = true;
    snapshot_.plan_id = plan.plan_id;
    snapshot_.execution_domain = FakeCudaDomain();
    snapshot_.readiness = plan.resolution.stage_count != 0
                              ? hal::ReadinessSnapshot{}
                              : hal::ReadinessSnapshot{};
    snapshot_.readiness.status = hal::StatusId::ready;
    snapshot_.readiness.requested_backend = hal::BackendId::cuda;
    snapshot_.readiness.selected_backend = hal::BackendId::cuda;
    snapshot_.readiness.provider =
        hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
    snapshot_.readiness.execution_domain = FakeCudaDomain();
    snapshot_.readiness.ready_stage_mask = 1;
    snapshot_.readiness.available = true;
  }
  video::VideoExecutionFrameResult
  Execute(const hal::HostFrameBinding &binding) noexcept override {
    video::VideoExecutionFrameResult result;
    const auto bytes = static_cast<std::size_t>(
        binding.output_descriptor.planes[0].size_bytes);
    std::memcpy(binding.output_planes[0].write_data,
                binding.input_planes[0].read_data, bytes);
    pending_ = {next_token_++, plan_.plan_id, kFakeAdapter,
                binding.capture_sequence, true};
    result.status = hal::StatusId::executed;
    result.completion = pending_;
    result.output.descriptor = binding.output_descriptor;
    result.output.planes = binding.output_planes;
    result.output.plane_count = 1;
    result.output.capture_sequence = binding.capture_sequence;
    result.output.output_ready = true;
    result.output_ready = true;
    result.execution_domain = FakeCudaDomain();
    result.stage_count = 1;
    result.stages[0].stage_id = plan_.intent.stages[0].intent.stage_id;
    result.stages[0].semantic =
        hal::SemanticEffectId::virtual_background_blur;
    result.stages[0].status = hal::StageExecutionStatus::succeeded;
    result.stages[0].valid_output_preserved = true;
    result.transfers.uploads = 1;
    result.transfers.final_readbacks = 1;
    result.transfers.cpu_tail_boundaries = 1;
    result.transfers.synchronizations = 1;
    result.frame_elapsed_ns = 100;
    return result;
  }
  hal::StatusId CommitOutput(const hal::CompletionToken &completion,
                             bool succeeded,
                             hal::ReasonId *reason) noexcept override {
    if (completion.token_id == 0 || completion.token_id != pending_.token_id) {
      if (reason)
        *reason = hal::ReasonId::stale_program;
      return hal::StatusId::stale;
    }
    if (reason)
      *reason = succeeded ? hal::ReasonId::none
                          : hal::ReasonId::output_write_failed;
    if (succeeded) {
      snapshot_.latest_committed_frame.valid = true;
      snapshot_.latest_committed_frame.capture_sequence =
          completion.capture_sequence;
      snapshot_.latest_committed_frame.output_commit_sequence = ++commits_;
      snapshot_.latest_committed_frame.stage_count = 1;
      snapshot_.latest_committed_frame.stages[0].stage_id =
          plan_.intent.stages[0].intent.stage_id;
      snapshot_.latest_committed_frame.stages[0].semantic =
          hal::SemanticEffectId::virtual_background_blur;
      snapshot_.latest_committed_frame.stages[0].status =
          hal::StageExecutionStatus::succeeded;
      snapshot_.telemetry.active_backend = hal::BackendId::cuda;
      snapshot_.telemetry.active_provider =
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
    }
    pending_ = {};
    return succeeded ? hal::StatusId::ready : hal::StatusId::failed;
  }
  void Invalidate(hal::ReasonId) noexcept override { snapshot_.prepared = false; }
  hal::StatusId ResetTemporal(hal::ReasonId) noexcept override {
    return hal::StatusId::ready;
  }
  void Shutdown() noexcept override { snapshot_.stopped = true; }
  video::VideoExecutionAdapterSnapshot Snapshot() const noexcept override {
    return snapshot_;
  }
private:
  video::FrozenVideoExecutionPlan plan_{};
  video::VideoExecutionAdapterSnapshot snapshot_{};
  hal::CompletionToken pending_{};
  std::uint64_t next_token_ = 1;
  std::uint64_t commits_ = 0;
};

class FakeFactory final : public video::IVideoAdapterFactory {
public:
  video::VideoAdapterFactoryPreflight
  Preflight(const video::VideoExecutionPlanRequest &request,
            const hal::ResolutionResult &resolution,
            const hal::DomainPlan &domain_plan,
            video::VideoExecutionRoute) noexcept override {
    video::VideoAdapterFactoryPreflight result;
    result.status = hal::StatusId::ready;
    result.backend = hal::BackendId::cuda;
    result.provider =
        hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
    result.execution_domain = FakeCudaDomain();
    result.readiness.status = hal::StatusId::ready;
    result.readiness.requested_backend = hal::BackendId::cuda;
    result.readiness.selected_backend = hal::BackendId::cuda;
    result.readiness.provider = result.provider;
    result.readiness.execution_domain = FakeCudaDomain();
    result.readiness.ready_stage_mask = 1;
    result.readiness.available = true;
    result.cached_evidence_only = true;
    result.accepts_frozen_resolution = true;
    result.performs_selection = false;
    result.per_frame_transfers =
        video::BuildPerFrameTransferContract(domain_plan, 1, 1);
    auto &program = result.prepared_program;
    program.program_id = request.plan_id;
    program.owner_adapter_instance_id = kFakeAdapter;
    program.execution_domain = FakeCudaDomain();
    program.host_input = request.host_input;
    program.host_output = request.host_output;
    program.invalidation_key = request.invalidation_key;
    program.temporal = request.temporal;
    program.budget = request.budget;
    program.setup_failure_policy = request.setup_failure_policy;
    program.cpu_tails = request.cpu_tails;
    program.cpu_tail_count = request.cpu_tail_count;
    program.requested_backend = hal::BackendId::cuda;
    program.strict_backend_selection = request.intent.backend.strict;
    program.requested_provider = result.provider;
    program.strict_provider_selection = request.intent.strict_provider_selection;
    program.requested_execution_domain = FakeCudaDomain();
    program.strict_execution_domain = true;
    program.requires_upload = domain_plan.islands[0].requires_upload;
    program.requires_final_readback =
        domain_plan.islands[0].requires_final_readback;
    const auto &resolved = resolution.stages[0];
    auto &stage = program.stages[0];
    stage.stage_id = resolved.intent.stage_id;
    stage.semantic = resolved.intent.semantic;
    stage.provider = result.provider;
    stage.backend = hal::BackendId::cuda;
    stage.execution_domain = FakeCudaDomain();
    stage.input.kind = hal::StageIoKind::image;
    stage.input.image = request.host_input;
    stage.input.image.host_coherence = hal::HostCoherence::not_host_visible;
    stage.output.kind = hal::StageIoKind::image;
    stage.output.image = request.host_output;
    stage.output.image.host_coherence = hal::HostCoherence::not_host_visible;
    stage.output.image.mutability = hal::Mutability::read_write;
    stage.model = request.intent.stages[0].model;
    stage.required = resolved.intent.required;
    stage.failure_action = request.intent.stages[0].intent.failure_action;
    stage.circuit_breaker.consecutive_failure_limit = 1;
    stage.circuit_breaker.cooldown_frames = 2;
    stage.circuit_breaker.action = hal::CircuitBreakerAction::disable_stage;
    program.stage_count = 1;
    return result;
  }
  std::unique_ptr<video::IVideoExecutionExecutor>
  Prepare(const video::FrozenVideoExecutionPlan &plan,
          video::VideoAdapterPrepareEvidence *evidence) noexcept override {
    evidence->status = hal::StatusId::ready;
    return std::make_unique<FakeExecutor>(plan);
  }
};

class FakeOwner final : public video::IProductionVideoSetupLeaseOwner {
public:
  hal::IResidentAdapter *resident_adapter() const noexcept override {
    return const_cast<FakeAdapter *>(&adapter);
  }
  video::IVideoAdapterFactory *video_adapter_factory() const noexcept override {
    return const_cast<FakeFactory *>(&factory);
  }
  void Shutdown() noexcept override {}
  FakeAdapter adapter;
  FakeFactory factory;
};

struct InjectionState {
  bool fail_acquire = false;
  bool mismatch_artifact = false;
};

bool ResolveFakeModel(
    void *, const video::CanonicalVideoStage &, const std::string &,
    open_video::ModelPack *pack,
    open_video::ValidatedModelArtifactIdentity *identity) noexcept {
  if (!pack || !identity)
    return false;
  pack->id = "prepared-matting";
  pack->task = "matting";
  pack->registry_generation = 13;
  open_video::MattingSpec matting;
  matting.input.width = 16;
  matting.input.height = 16;
  matting.input.channels = 3;
  pack->matting = matting;
  identity->registry_generation = 13;
  identity->model_id_sha256[0] = 7;
  identity->manifest_sha256[0] = 8;
  identity->artifact_sha256[0] = 9;
  identity->artifact_set_sha256[0] = 10;
  return true;
}

hal::StatusId AcquireFakeLease(
    void *context,
    const open_cuda::OpenCudaProductionVideoExecutionFactoryRequest &request,
    video::ProductionVideoSetupLease *destination,
    open_cuda::OpenCudaProductionVideoExecutionBootstrapResult *bootstrap,
    video::ProductionVideoSetupLeaseAcquireEvidence *evidence,
    hal::ReasonId *reason) noexcept {
  auto *state = static_cast<InjectionState *>(context);
  if (state->fail_acquire) {
    if (reason)
      *reason = hal::ReasonId::resource_exhausted;
    return hal::StatusId::resource_exhausted;
  }
  auto owner = std::make_unique<FakeOwner>();
  video::ProductionVideoSetupLeaseSnapshot snapshot;
  snapshot.state = video::ProductionVideoSetupLeaseState::live;
  snapshot.status = hal::StatusId::ready;
  snapshot.backend = hal::BackendId::cuda;
  snapshot.execution_domain = FakeCudaDomain();
  snapshot.generations = request.generations;
  snapshot.generations.context = FakeCudaDomain().logical_context_generation;
  snapshot.reserved_budget = request.reserved_budget;
  snapshot.providers[0] =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  snapshot.provider_count = 1;
  snapshot.artifacts[0].semantic = request.semantic;
  snapshot.artifacts[0].provider = snapshot.providers[0];
  snapshot.artifacts[0].artifact = request.model;
  if (state->mismatch_artifact)
    ++snapshot.artifacts[0].artifact.model_id_hash[0];
  snapshot.artifact_count = 1;
  snapshot.adapter.adapter_instance_id = kFakeAdapter;
  snapshot.adapter.backend = hal::BackendId::cuda;
  snapshot.adapter.execution_domain = FakeCudaDomain();
  snapshot.adapter.selection_priority = 100;
  snapshot.adapter.resident_adapter = &owner->adapter;
  auto &capability = snapshot.adapter.semantic_capabilities[0];
  capability.semantic = request.semantic;
  capability.provider = snapshot.providers[0];
  capability.status = hal::StatusId::ready;
  capability.live_stage = true;
  capability.production_eligible = true;
  capability.exact_evidence = true;
  capability.cpu_tail = hal::CpuTailId::writer_copy;
  capability.cpu_tail_placement =
      hal::DeclaredCpuTail::Placement::after_stage;
  capability.evidence[0].code =
      hal::CapabilityEvidenceCode::model_artifact_identity_valid;
  capability.evidence[0].status = hal::StatusId::ready;
  capability.evidence[0].semantic = request.semantic;
  capability.evidence[0].backend = hal::BackendId::cuda;
  capability.evidence[0].satisfied = true;
  capability.evidence_count = 1;
  snapshot.adapter.semantic_capability_count = 1;
  snapshot.factory.factory_id = kFakeFactory;
  snapshot.factory.route = video::VideoExecutionRoute::prepared_resident;
  snapshot.factory.backend = hal::BackendId::cuda;
  snapshot.factory.provider = snapshot.providers[0];
  snapshot.factory.adapter_instance_id = kFakeAdapter;
  snapshot.factory.execution_domain = FakeCudaDomain();
  snapshot.factory.selection_priority = 100;
  snapshot.factory.factory = &owner->factory;
  snapshot.readiness.status = hal::StatusId::ready;
  snapshot.readiness.requested_backend = hal::BackendId::cuda;
  snapshot.readiness.selected_backend = hal::BackendId::cuda;
  snapshot.readiness.provider = snapshot.providers[0];
  snapshot.readiness.execution_domain = FakeCudaDomain();
  snapshot.readiness.ready_stage_mask = 1;
  snapshot.readiness.available = true;
  if (bootstrap)
    bootstrap->setup_lease = snapshot;
  return video::ProductionVideoSetupLease::Acquire(
      snapshot, std::move(owner), destination, evidence, reason);
}

bool Check(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << "\n";
  return condition;
}

hal::ImageDescriptor HostRgb(std::uint32_t width, std::uint32_t height,
                             hal::Mutability mutability) {
  hal::ImageDescriptor image;
  image.width = width;
  image.height = height;
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
  image.planes[0].row_stride_bytes = static_cast<std::uint64_t>(width) * 3u;
  image.planes[0].size_bytes = image.planes[0].row_stride_bytes * height;
  image.planes[0].width = width;
  image.planes[0].height = height;
  image.planes[0].alignment_bytes = 1;
  return image;
}

video::ProductionVideoResidentSetup
StrictRequest(hal::SemanticEffectId semantic =
                  hal::SemanticEffectId::virtual_background_blur) {
  video::ProductionVideoResidentSetup request;
  auto &plan = request.plan;
  plan.plan_id = 51;
  plan.intent.backend.mode = hal::BackendRequestMode::explicit_backend;
  plan.intent.backend.backend = hal::BackendId::cuda;
  plan.intent.backend.strict = true;
  plan.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::cuda_family;
  plan.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::open_cuda;
  plan.intent.preferences.strict_effects_engine = true;
  plan.intent.requested_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  plan.intent.strict_provider_selection = true;
  plan.intent.stage_count = 1;
  auto &stage = plan.intent.stages[0];
  stage.intent.stage_id = 7;
  stage.intent.semantic = semantic;
  stage.intent.failure_action = hal::FailureAction::fail_open_preserve_input;
  stage.parameters[0] = {
      video::CanonicalVideoParameterId::strength,
      video::CanonicalVideoParameterKind::signed_integer,
      8,
  };
  stage.parameter_count = 1;
  plan.host_input = HostRgb(64, 48, hal::Mutability::read_only);
  plan.host_output = HostRgb(64, 48, hal::Mutability::write_only);
  plan.invalidation_key.generations.configuration = 1;
  plan.invalidation_key.generations.geometry = 1;
  plan.invalidation_key.canonical_plan_hash = 1;
  plan.budget.persistent_bytes = UINT64_C(512) << 20;
  plan.budget.scratch_bytes = UINT64_C(256) << 20;
  plan.budget.upload_staging_bytes = UINT64_C(64) << 20;
  plan.budget.readback_staging_bytes = UINT64_C(64) << 20;
  plan.budget.resource_count = 128;
  plan.budget.in_flight_slots = 1;
  plan.cpu_tails[0].id = hal::CpuTailId::backend_resolved_output;
  plan.cpu_tails[0].after_stage_id = stage.intent.stage_id;
  plan.cpu_tails[0].required = true;
  plan.cpu_tails[0].placement = hal::DeclaredCpuTail::Placement::after_stage;
  plan.cpu_tails[0].boundary_index = 1;
  plan.cpu_tail_count = 1;
  plan.setup_failure_policy = hal::SetupFailurePolicy::retain_old_program;
  plan.migration_policy = video::VideoExecutionMigrationPolicy::prepared_only;
  return request;
}

bool StrictPolicyIsFailClosed() {
  auto source = open_cuda::CreateProductionVideoBackendSetupSource();
  if (!Check(source != nullptr, "source construction failed"))
    return false;

  auto request = StrictRequest();
  request.plan.intent.backend.strict = false;
  auto result = source->AcquireResidentSetup(request);
  bool ok =
      Check(result.status == hal::StatusId::blocked &&
                result.reason == hal::ReasonId::strict_backend_unavailable &&
                result.setup_completed_outside_frame_loop,
            "non-strict backend request did not fail closed");

  request = StrictRequest();
  request.plan.intent.requested_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cpu;
  result = source->AcquireResidentSetup(request);
  ok &= Check(result.status == hal::StatusId::blocked &&
                  result.reason == hal::ReasonId::strict_backend_unavailable,
              "CPU provider was accepted by strict CUDA source");

  request = StrictRequest();
  request.plan.intent.stage_count = 2;
  result = source->AcquireResidentSetup(request);
  ok &= Check(result.status == hal::StatusId::blocked &&
                  result.reason ==
                      hal::ReasonId::cross_adapter_residency_forbidden,
              "mixed resident section was not rejected");
  return ok;
}

bool SemanticCapabilityClaimsRemainExact() {
  auto source = open_cuda::CreateProductionVideoBackendSetupSource();
  const UniqueTempDir temp;
  const char *old = std::getenv("XDG_DATA_HOME");
  const std::string old_value = old == nullptr ? std::string{} : old;
  const bool had_old = old != nullptr;
  (void)setenv("XDG_DATA_HOME", temp.path().c_str(), 1);
  const hal::SemanticEffectId supported[] = {
      hal::SemanticEffectId::virtual_background_replace,
      hal::SemanticEffectId::virtual_key_light,
  };
  bool ok = true;
  for (const auto semantic : supported) {
    const auto result = source->AcquireResidentSetup(StrictRequest(semantic));
    ok &= Check(
        result.status == hal::StatusId::unavailable &&
            result.reason == hal::ReasonId::capability_evidence_incomplete &&
            result.setup_completed_outside_frame_loop && !result.lease.live(),
        "supported semantic did not advance to setup-time model evidence");
  }
  if (had_old)
    (void)setenv("XDG_DATA_HOME", old_value.c_str(), 1);
  else
    (void)unsetenv("XDG_DATA_HOME");

  const hal::SemanticEffectId unsupported[] = {
      hal::SemanticEffectId::auto_frame,
      hal::SemanticEffectId::mirror,
      hal::SemanticEffectId::vignette,
  };
  for (const auto semantic : unsupported) {
    const auto result = source->AcquireResidentSetup(StrictRequest(semantic));
    ok &=
        Check(result.status == hal::StatusId::blocked &&
                  result.reason == hal::ReasonId::semantic_effect_unsupported &&
                  result.setup_completed_outside_frame_loop,
              "unsupported semantic gained false CUDA capability");
  }
  const hal::SemanticEffectId cpu_provider_compatibility[] = {
      hal::SemanticEffectId::video_noise_removal,
      hal::SemanticEffectId::eye_contact,
  };
  for (const auto semantic : cpu_provider_compatibility) {
    const auto result = source->AcquireResidentSetup(StrictRequest(semantic));
    ok &= Check(
        result.status == hal::StatusId::blocked &&
            result.reason == hal::ReasonId::semantic_provider_unavailable &&
            result.setup_completed_outside_frame_loop && !result.lease.live() &&
            !result.compatibility_executor,
        "strict CUDA accepted a CPU-provider compatibility semantic");
  }
  return ok;
}

bool HostAndModelEvidenceAreSetupOnly() {
  auto source = open_cuda::CreateProductionVideoBackendSetupSource();
  auto request = StrictRequest();
  request.plan.host_output.width += 1;
  auto result = source->AcquireResidentSetup(request);
  bool ok = Check(result.status == hal::StatusId::blocked &&
                      result.reason == hal::ReasonId::malformed_descriptor,
                  "geometry mismatch was not rejected before setup");

  const UniqueTempDir temp;
  const char *old = std::getenv("XDG_DATA_HOME");
  const std::string old_value = old == nullptr ? std::string{} : old;
  const bool had_old = old != nullptr;
  (void)setenv("XDG_DATA_HOME", temp.path().c_str(), 1);
  result = source->AcquireResidentSetup(StrictRequest());
  if (had_old)
    (void)setenv("XDG_DATA_HOME", old_value.c_str(), 1);
  else
    (void)unsetenv("XDG_DATA_HOME");

  ok &= Check(
      result.status == hal::StatusId::unavailable &&
          result.reason == hal::ReasonId::capability_evidence_incomplete &&
          result.setup_completed_outside_frame_loop && !result.lease.live(),
      "missing model evidence did not fail during setup");
  return ok;
}

bool CanonicalAutomaticAndForcedCudaShapesReachSetup() {
  auto source = open_cuda::CreateProductionVideoBackendSetupSource();
  const UniqueTempDir temp;
  const char *old = std::getenv("XDG_DATA_HOME");
  const std::string old_value = old == nullptr ? std::string{} : old;
  const bool had_old = old != nullptr;
  (void)setenv("XDG_DATA_HOME", temp.path().c_str(), 1);

  auto automatic = StrictRequest();
  automatic.plan.intent.backend.mode = hal::BackendRequestMode::automatic;
  automatic.plan.intent.backend.backend = hal::BackendId::invalid;
  automatic.plan.intent.backend.strict = false;
  automatic.plan.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::any;
  automatic.plan.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::automatic;
  automatic.plan.intent.preferences.strict_effects_engine = false;
  automatic.plan.intent.requested_provider =
      hal::SemanticEffectProviderId::invalid;
  automatic.plan.intent.strict_provider_selection = false;
  auto result = source->AcquireResidentSetup(automatic);
  bool ok = Check(result.status == hal::StatusId::unavailable &&
                      result.reason ==
                          hal::ReasonId::capability_evidence_incomplete,
                  "automatic intent did not reach model setup");

  auto forced_cuda = automatic;
  forced_cuda.plan.intent.backend.mode =
      hal::BackendRequestMode::explicit_backend;
  forced_cuda.plan.intent.backend.backend = hal::BackendId::cuda;
  forced_cuda.plan.intent.backend.strict = true;
  forced_cuda.plan.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::cuda_family;
  result = source->AcquireResidentSetup(forced_cuda);
  ok &= Check(result.status == hal::StatusId::unavailable &&
                  result.reason ==
                      hal::ReasonId::capability_evidence_incomplete,
              "compute=cuda engine=auto intent did not reach model setup");

  if (had_old)
    (void)setenv("XDG_DATA_HOME", old_value.c_str(), 1);
  else
    (void)unsetenv("XDG_DATA_HOME");
  return ok;
}

bool CompatibilityTopologyNeverMasqueradesAsStrictCuda() {
  auto source = open_cuda::CreateProductionVideoBackendSetupSource();
  auto request = StrictRequest(hal::SemanticEffectId::video_noise_removal);
  request.plan.intent.stage_count = 2;
  request.plan.intent.stages[1].intent.stage_id = 8;
  request.plan.intent.stages[1].intent.semantic =
      hal::SemanticEffectId::eye_contact;
  request.plan.intent.stages[1].intent.failure_action =
      hal::FailureAction::fail_open_preserve_input;
  request.plan.cpu_tails[0].after_stage_id = 8;
  request.plan.cpu_tails[0].boundary_index = 2;
  auto result = source->AcquireResidentSetup(request);
  bool ok =
      Check(result.status == hal::StatusId::blocked &&
                result.reason == hal::ReasonId::semantic_provider_unavailable &&
                !result.lease.live() && !result.compatibility_executor,
            "strict CUDA accepted CPU-tensor compatibility providers");

  request.plan.intent.backend.mode = hal::BackendRequestMode::automatic;
  request.plan.intent.backend.backend = hal::BackendId::invalid;
  request.plan.intent.backend.strict = false;
  request.plan.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::any;
  request.plan.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::automatic;
  request.plan.intent.preferences.strict_effects_engine = false;
  request.plan.intent.requested_provider =
      hal::SemanticEffectProviderId::invalid;
  request.plan.intent.strict_provider_selection = false;
  request.effects.video_noise_removal.enabled = true;
  request.effects.eye_contact.enabled = true;

  const UniqueTempDir temp;
  const char *old = std::getenv("XDG_DATA_HOME");
  const std::string old_value = old == nullptr ? std::string{} : old;
  const bool had_old = old != nullptr;
  (void)setenv("XDG_DATA_HOME", temp.path().c_str(), 1);
  result = source->AcquireResidentSetup(request);
  if (had_old)
    (void)setenv("XDG_DATA_HOME", old_value.c_str(), 1);
  else
    (void)unsetenv("XDG_DATA_HOME");

  ok &= Check(
      result.status == hal::StatusId::unavailable &&
          result.reason == hal::ReasonId::capability_evidence_incomplete &&
          result.setup_completed_outside_frame_loop && !result.lease.live(),
      "automatic compatibility did not fail during model setup");
  return ok;
}

bool StopNever(void *) noexcept { return false; }
std::uint64_t Now(void *) noexcept { return 1; }
void Poll(void *, std::uint64_t) noexcept {}

class ScriptedSetupSource final
    : public video::IProductionVideoBackendSetupSource {
public:
  ScriptedSetupSource(int *calls, hal::StatusId status)
      : calls_(calls), status_(status) {}
  video::ProductionVideoResidentSetupSourceResult AcquireResidentSetup(
      const video::ProductionVideoResidentSetup &request) noexcept override {
    ++*calls_;
    video::ProductionVideoResidentSetupSourceResult result;
    result.status = status_;
    result.reason = status_ == hal::StatusId::ready
                        ? hal::ReasonId::none
                        : hal::ReasonId::strict_backend_unavailable;
    result.readiness.status = result.status;
    result.setup_completed_outside_frame_loop = true;
    result.canonical_effects = request.effects;
    return result;
  }
private:
  int *calls_;
  hal::StatusId status_;
};

bool StaticSelectionOrderIsBehavioral() {
  const auto exercise = [](video::ProductionVideoResidentSetup request,
                           hal::StatusId maxine_status, int expected_maxine,
                           int expected_cuda) {
    int maxine_calls = 0;
    int cuda_calls = 0;
    video::StaticProductionVideoBackendSetupSourcesForTesting sources;
    sources.maxine =
        std::make_unique<ScriptedSetupSource>(&maxine_calls, maxine_status);
    sources.open_cuda = std::make_unique<ScriptedSetupSource>(
        &cuda_calls, hal::StatusId::ready);
    auto source = video::CreateStaticProductionVideoBackendSetupSourceForTesting(
        std::move(sources));
    const auto result = source->AcquireResidentSetup(request);
    return Check(result.status == hal::StatusId::ready &&
                     maxine_calls == expected_maxine &&
                     cuda_calls == expected_cuda,
                 "static backend selection order changed");
  };
  auto automatic = StrictRequest();
  automatic.plan.intent.backend = {};
  automatic.plan.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::any;
  automatic.plan.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::automatic;
  automatic.plan.intent.preferences.strict_effects_engine = false;
  automatic.plan.intent.requested_provider =
      hal::SemanticEffectProviderId::invalid;
  automatic.plan.intent.strict_provider_selection = false;
  bool ok = exercise(automatic, hal::StatusId::unavailable, 1, 1);
  ok &= exercise(automatic, hal::StatusId::ready, 1, 0);
  auto forced = automatic;
  forced.plan.intent.backend.mode = hal::BackendRequestMode::explicit_backend;
  forced.plan.intent.backend.backend = hal::BackendId::cuda;
  forced.plan.intent.backend.strict = true;
  forced.plan.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::cuda_family;
  ok &= exercise(forced, hal::StatusId::ready, 0, 1);
  auto strict_provider = automatic;
  strict_provider.plan.intent.requested_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  strict_provider.plan.intent.strict_provider_selection = true;
  ok &= exercise(strict_provider, hal::StatusId::ready, 0, 1);

  {
    int maxine_calls = 0, cuda_calls = 0;
    video::StaticProductionVideoBackendSetupSourcesForTesting sources;
    sources.maxine = std::make_unique<ScriptedSetupSource>(
        &maxine_calls, hal::StatusId::unavailable);
    sources.open_cuda = std::make_unique<ScriptedSetupSource>(
        &cuda_calls, hal::StatusId::ready);
    auto source = video::CreateStaticProductionVideoBackendSetupSourceForTesting(
        std::move(sources));
    auto strict_maxine = automatic;
    strict_maxine.plan.intent.backend.mode =
        hal::BackendRequestMode::explicit_backend;
    strict_maxine.plan.intent.backend.backend = hal::BackendId::maxine;
    strict_maxine.plan.intent.backend.strict = true;
    const auto result = source->AcquireResidentSetup(strict_maxine);
    ok &= Check(result.status == hal::StatusId::unavailable &&
                    maxine_calls == 1 && cuda_calls == 0,
                "strict Maxine failure substituted CUDA");
  }
  {
    int maxine_calls = 0, cuda_calls = 0, vulkan_calls = 0;
    video::StaticProductionVideoBackendSetupSourcesForTesting sources;
    sources.maxine = std::make_unique<ScriptedSetupSource>(
        &maxine_calls, hal::StatusId::ready);
    sources.open_cuda = std::make_unique<ScriptedSetupSource>(
        &cuda_calls, hal::StatusId::ready);
    sources.vulkan = std::make_unique<ScriptedSetupSource>(
        &vulkan_calls, hal::StatusId::unavailable);
    auto source = video::CreateStaticProductionVideoBackendSetupSourceForTesting(
        std::move(sources));
    auto strict_vulkan = automatic;
    strict_vulkan.plan.intent.backend.mode =
        hal::BackendRequestMode::explicit_backend;
    strict_vulkan.plan.intent.backend.backend = hal::BackendId::vulkan;
    strict_vulkan.plan.intent.backend.strict = true;
    strict_vulkan.plan.intent.preferences.compute_family =
        hal::ComputeFamilyAllowance::vulkan_only;
    const auto result = source->AcquireResidentSetup(strict_vulkan);
    ok &= Check(result.status == hal::StatusId::unavailable &&
                    vulkan_calls == 1 && maxine_calls == 0 && cuda_calls == 0,
                "strict Vulkan failure substituted another backend");
  }
  {
    int maxine_calls = 0;
    open_cuda::ProductionVideoBackendSetupDependencies dependencies;
    InjectionState state;
    dependencies.resolve_model = &ResolveFakeModel;
    dependencies.acquire_lease = &AcquireFakeLease;
    dependencies.context = &state;
    video::StaticProductionVideoBackendSetupSourcesForTesting sources;
    sources.maxine = std::make_unique<ScriptedSetupSource>(
        &maxine_calls, hal::StatusId::ready);
    sources.open_cuda =
        open_cuda::CreateProductionVideoBackendSetupSourceForTesting(
            dependencies);
    auto source = video::CreateStaticProductionVideoBackendSetupSourceForTesting(
        std::move(sources));
    auto conflict = StrictRequest();
    conflict.plan.intent.backend.backend = hal::BackendId::maxine;
    const auto result = source->AcquireResidentSetup(conflict);
    ok &= Check(result.status == hal::StatusId::blocked && maxine_calls == 0,
                "conflicting strict backend/provider was substituted");
  }
  return ok;
}

bool ProductionSourceManagerBoundaryIsBehavioral() {
  const auto run = [](video::ProductionVideoResidentSetup resident) {
    InjectionState state;
    open_cuda::ProductionVideoBackendSetupDependencies dependencies;
    dependencies.resolve_model = &ResolveFakeModel;
    dependencies.acquire_lease = &AcquireFakeLease;
    dependencies.context = &state;
    video::ProductionVideoBackendManager manager(
        open_cuda::CreateProductionVideoBackendSetupSourceForTesting(
            dependencies));
    resident.wait.timeout_nanoseconds = 1000;
    resident.wait.poll_interval_nanoseconds = 10;
    resident.wait.maximum_poll_count = 100;
    resident.wait.stop_requested = &StopNever;
    resident.wait.cancel_requested = &StopNever;
    resident.wait.now_nanoseconds = &Now;
    resident.wait.poll_wait = &Poll;
    resident.wait.callback_context = &state;
    resident.wait.callback_context_generation = 1;
    resident.wait.callback_context_lifetime =
        hal::ResourceLifetime::adapter_context;
    resident.plan.intent.stages[0].model = {};
    resident.plan.invalidation_key.generations.model = 0;
    resident.plan.invalidation_key.generations.provider = 1;
    resident.plan.invalidation_key.generations.device = 1;
    resident.plan.invalidation_key.generations.context = 7;
    resident.plan.invalidation_key.model_artifact_hash = 0;
    resident.plan.budget.persistent_bytes = UINT64_C(1) << 20;
    resident.plan.budget.scratch_bytes = UINT64_C(1) << 20;
    resident.plan.budget.upload_staging_bytes = UINT64_C(1) << 20;
    resident.plan.budget.readback_staging_bytes = UINT64_C(1) << 20;
    resident.plan.budget.resource_count = 128;
    resident.plan.intent.stages[0].intent.stage_id = 1;
    resident.plan.cpu_tails[0].after_stage_id = 1;
    resident.effects.virtual_background.mode =
        video::effects::VirtualBackgroundMode::blur;
    auto &stage = resident.plan.intent.stages[0];
    stage.parameters[1].id =
        video::CanonicalVideoParameterId::remove_color_rgb;
    stage.parameters[1].kind =
        video::CanonicalVideoParameterKind::unsigned_integer;
    stage.parameters[2].id =
        video::CanonicalVideoParameterId::greenscreen_mode;
    stage.parameters[2].kind =
        video::CanonicalVideoParameterKind::unsigned_integer;
    stage.parameters[3].id =
        video::CanonicalVideoParameterId::greenscreen_temporal;
    stage.parameters[3].kind =
        video::CanonicalVideoParameterKind::boolean;
    stage.parameters[3].boolean_value = true;
    stage.parameter_count = 4;
    const auto original = resident.plan.intent;
    video::ProductionVideoBackendSetupRequest request;
    request.kind = video::ProductionVideoBackendSetupKind::resident_island;
    request.resident = resident;
    video::ProductionVideoBackendSetupEvidence evidence;
    hal::ReasonId reason = hal::ReasonId::none;
    const auto setup_status = manager.Setup(request, &evidence, &reason);
    if (!Check(setup_status == hal::StatusId::ready,
               "real source did not prepare through manager"))
      return false;
    const auto setup = manager.Snapshot();
    bool ok = Check(setup.resident.plan.intent.backend.mode ==
                            original.backend.mode &&
                        setup.resident.plan.intent.backend.backend ==
                            original.backend.backend &&
                        setup.resident.plan.intent.requested_provider ==
                            original.requested_provider &&
                        setup.readiness.selected_backend ==
                            hal::BackendId::cuda &&
                        setup.readiness.provider ==
                            hal::SemanticEffectProviderId::
                                open_video_onnxruntime_cuda &&
                        setup.resident.plan.intent.stages[0]
                                .model.artifact_generation == 13,
                    "source changed canonical intent or lost bound evidence");
    std::array<std::byte, 64 * 48 * 3> input{};
    std::array<std::byte, 64 * 48 * 3> output{};
    input[0] = std::byte{0x5a};
    hal::HostFrameBinding binding;
    binding.input_descriptor = resident.plan.host_input;
    binding.output_descriptor = resident.plan.host_output;
    binding.input_planes[0].read_data = input.data();
    binding.input_planes[0].span_bytes = input.size();
    binding.input_planes[0].row_stride_bytes = 64 * 3;
    binding.output_planes[0].write_data = output.data();
    binding.output_planes[0].span_bytes = output.size();
    binding.output_planes[0].row_stride_bytes = 64 * 3;
    binding.input_plane_count = binding.output_plane_count = 1;
    binding.ownership = hal::OwnershipModel::borrowed_until_completion;
    binding.lifetime = hal::ResourceLifetime::frame_borrowed;
    binding.capture_sequence = 41;
    auto frame = manager.Execute(binding);
    ok &= Check(frame.status == hal::StatusId::executed &&
                    frame.output_ready && output[0] == input[0],
                "prepared source result did not execute");
    ok &= Check(manager.CommitOutput(frame.completion, false, &reason) ==
                        hal::StatusId::failed &&
                    !manager.Snapshot()
                         .resident.latest_committed_frame.valid,
                "failed writer commit created active evidence");
    binding.capture_sequence = 42;
    frame = manager.Execute(binding);
    ok &= Check(manager.CommitOutput(frame.completion, true, &reason) ==
                        hal::StatusId::ready &&
                    manager.Snapshot()
                            .resident.latest_committed_frame.capture_sequence ==
                        42 &&
                    manager.Snapshot().resident.telemetry.active_backend ==
                        hal::BackendId::cuda,
                "successful writer commit did not attribute CUDA");
    state.fail_acquire = true;
    evidence = {};
    ok &= Check(manager.Setup(request, &evidence, &reason) ==
                        hal::StatusId::resource_exhausted &&
                    reason == hal::ReasonId::resource_exhausted &&
                    evidence.old_plan_retained &&
                    manager.Snapshot().resident_prepared,
                "injected lease failure did not retain active CUDA plan");
    state.fail_acquire = false;
    state.mismatch_artifact = true;
    evidence = {};
    ok &= Check(manager.Setup(request, &evidence, &reason) ==
                        hal::StatusId::blocked &&
                    evidence.candidate_rejected && evidence.old_plan_retained &&
                    manager.Snapshot().resident_prepared,
                "mismatched lease artifact evidence was accepted");
    return ok;
  };

  auto automatic = StrictRequest();
  automatic.plan.intent.backend.mode = hal::BackendRequestMode::automatic;
  automatic.plan.intent.backend.backend = hal::BackendId::invalid;
  automatic.plan.intent.backend.strict = false;
  automatic.plan.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::any;
  automatic.plan.intent.preferences.effects_engine =
      hal::EffectsEnginePreferenceConstraint::automatic;
  automatic.plan.intent.preferences.strict_effects_engine = false;
  automatic.plan.intent.requested_provider =
      hal::SemanticEffectProviderId::invalid;
  automatic.plan.intent.strict_provider_selection = false;
  bool ok = run(automatic);
  auto forced = automatic;
  forced.plan.intent.backend.mode = hal::BackendRequestMode::explicit_backend;
  forced.plan.intent.backend.backend = hal::BackendId::cuda;
  forced.plan.intent.backend.strict = true;
  forced.plan.intent.preferences.compute_family =
      hal::ComputeFamilyAllowance::cuda_family;
  ok &= run(forced);
  ok &= run(StrictRequest());
  return ok;
}

} // namespace

int main() {
  if (!StrictPolicyIsFailClosed() || !SemanticCapabilityClaimsRemainExact() ||
      !HostAndModelEvidenceAreSetupOnly() ||
      !CanonicalAutomaticAndForcedCudaShapesReachSetup() ||
      !StaticSelectionOrderIsBehavioral() ||
      !ProductionSourceManagerBoundaryIsBehavioral() ||
      !CompatibilityTopologyNeverMasqueradesAsStrictCuda()) {
    return 1;
  }
  std::cout << "Open CUDA production backend setup-source tests passed.\n";
  return 0;
}
