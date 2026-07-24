#include "core/vulkan/video_execution_factory.h"

#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
#include "core/vulkan/hal_resident_adapter.h"
#include "core/vulkan/kernels/utility_kernels.h"
#include "core/vulkan/qualified_final_resident_executor.h"
#endif

#include <algorithm>
#include <atomic>
#include <limits>
#include <new>
#include <string>
#include <utility>

namespace studiocast::vulkan {
namespace {

using hal::BackendId;
using hal::CapabilityEvidenceCode;
using hal::ReasonId;
using hal::SemanticEffectId;
using hal::SemanticEffectProviderId;
using hal::StatusId;

constexpr auto kProvider =
    SemanticEffectProviderId::open_vulkan_production_kernels;

void SetReason(ReasonId *reason, ReasonId value) noexcept {
  if (reason)
    *reason = value;
}

bool IsSetupSuccess(StatusId status) noexcept {
  return status == StatusId::ready || status == StatusId::degraded;
}

bool IsExecutionSuccess(StatusId status) noexcept {
  return status == StatusId::executed || status == StatusId::degraded;
}

bool IsZeroModel(const hal::ModelArtifactIdentity &model) noexcept {
  return model.artifact_generation == 0 &&
         std::all_of(model.model_id_hash.begin(), model.model_id_hash.end(),
                     [](std::uint8_t value) { return value == 0; }) &&
         std::all_of(model.artifact_set_hash.begin(),
                     model.artifact_set_hash.end(),
                     [](std::uint8_t value) { return value == 0; });
}

bool IsTightHostRgb(const hal::ImageDescriptor &image,
                    hal::Mutability mutability) noexcept {
  if (!hal::ValidateImageDescriptor(image) || image.plane_count != 1 ||
      image.format != hal::PixelFormat::rgb_u8 ||
      image.channel_order != hal::ChannelOrder::rgb ||
      image.color_space != hal::ColorSpace::srgb ||
      image.color_range != hal::ColorRange::full ||
      image.alpha_meaning != hal::AlphaMeaning::none ||
      image.orientation != hal::Orientation::top_left ||
      image.host_coherence != hal::HostCoherence::coherent ||
      image.mutability != mutability ||
      image.aliasing != hal::AliasingPermission::forbidden ||
      image.in_place_permitted || image.width == 0 || image.height == 0 ||
      image.width > std::numeric_limits<std::uint64_t>::max() / 3u) {
    return false;
  }
  const auto stride = static_cast<std::uint64_t>(image.width) * 3u;
  const auto &plane = image.planes[0];
  return plane.offset_bytes == 0 && plane.width == image.width &&
         plane.height == image.height && plane.row_stride_bytes == stride &&
         image.height <= std::numeric_limits<std::uint64_t>::max() / stride &&
         plane.size_bytes == stride * image.height;
}

bool SameImage(const hal::ImageDescriptor &left,
               const hal::ImageDescriptor &right) noexcept {
  if (left.width != right.width || left.height != right.height ||
      left.plane_count != right.plane_count || left.format != right.format ||
      left.channel_order != right.channel_order ||
      left.color_space != right.color_space ||
      left.color_range != right.color_range ||
      left.alpha_meaning != right.alpha_meaning ||
      left.orientation != right.orientation ||
      left.host_coherence != right.host_coherence ||
      left.mutability != right.mutability || left.aliasing != right.aliasing ||
      left.in_place_permitted != right.in_place_permitted ||
      left.base_alignment_bytes != right.base_alignment_bytes) {
    return false;
  }
  for (std::size_t index = 0; index < left.plane_count; ++index) {
    const auto &a = left.planes[index];
    const auto &b = right.planes[index];
    if (a.offset_bytes != b.offset_bytes ||
        a.row_stride_bytes != b.row_stride_bytes ||
        a.size_bytes != b.size_bytes || a.width != b.width ||
        a.height != b.height || a.alignment_bytes != b.alignment_bytes) {
      return false;
    }
  }
  return true;
}

hal::ImageDescriptor ResidentImage(const hal::ImageDescriptor &host,
                                   hal::Mutability mutability) noexcept {
  auto image = host;
  image.host_coherence = hal::HostCoherence::not_host_visible;
  image.mutability = mutability;
  return image;
}

const hal::SemanticCapabilityEvidence *
FindCapability(const hal::AdapterRegistration &registration,
               SemanticEffectId semantic) noexcept {
  for (std::size_t index = 0; index < registration.semantic_capability_count;
       ++index) {
    const auto &candidate = registration.semantic_capabilities[index];
    if (candidate.semantic == semantic && candidate.provider == kProvider)
      return &candidate;
  }
  return nullptr;
}

bool HasSatisfiedEvidence(const hal::SemanticCapabilityEvidence &capability,
                          CapabilityEvidenceCode code) noexcept {
  for (std::size_t index = 0; index < capability.evidence_count; ++index) {
    const auto &record = capability.evidence[index];
    if (record.code == code && record.status == StatusId::ready &&
        record.reason == ReasonId::none &&
        record.backend == BackendId::vulkan && record.satisfied) {
      return true;
    }
  }
  return false;
}

bool ExactProductionCapability(
    const hal::SemanticCapabilityEvidence &capability,
    CapabilityEvidenceCode production_code) noexcept {
  constexpr CapabilityEvidenceCode kCommon[]{
      CapabilityEvidenceCode::vulkan_build_enabled,
      CapabilityEvidenceCode::vulkan_loader_available,
      CapabilityEvidenceCode::vulkan_non_cpu_device,
      CapabilityEvidenceCode::vulkan_compute_queue_available,
      CapabilityEvidenceCode::vulkan_context_healthy,
      CapabilityEvidenceCode::vulkan_utility_kernels_ready,
      CapabilityEvidenceCode::vulkan_exact_device_shared,
      CapabilityEvidenceCode::vulkan_device_local_residency,
      CapabilityEvidenceCode::vulkan_bounded_allocations,
      CapabilityEvidenceCode::vulkan_no_cpu_layers,
      CapabilityEvidenceCode::vulkan_no_intermediate_readbacks,
      CapabilityEvidenceCode::vulkan_live_stage_present,
  };
  if (!hal::IsSelectableSemanticCapability(capability) ||
      capability.provider != kProvider ||
      capability.supports_tracked_vignette_center ||
      capability.cpu_tail != hal::CpuTailId::invalid) {
    return false;
  }
  for (const auto code : kCommon) {
    if (!HasSatisfiedEvidence(capability, code))
      return false;
  }
  return HasSatisfiedEvidence(capability, production_code);
}

bool ExactRegistration(const hal::AdapterRegistration &registration,
                       ReasonId *reason) noexcept {
  if (registration.adapter_instance_id == 0 ||
      registration.backend != BackendId::vulkan ||
      registration.resident_adapter == nullptr ||
      registration.execution_domain.backend != BackendId::vulkan ||
      registration.execution_domain.api_family != hal::ApiFamily::vulkan ||
      registration.execution_domain.adapter_instance_id !=
          registration.adapter_instance_id ||
      registration.resident_adapter->adapter_instance_id() !=
          registration.adapter_instance_id ||
      registration.resident_adapter->backend_id() != BackendId::vulkan ||
      !hal::ValidateExecutionDomain(registration.execution_domain)) {
    SetReason(reason, ReasonId::malformed_contract);
    return false;
  }
  const auto *mirror = FindCapability(registration, SemanticEffectId::mirror);
  const auto *vignette =
      FindCapability(registration, SemanticEffectId::vignette);
  if (!mirror || !vignette ||
      !ExactProductionCapability(
          *mirror,
          CapabilityEvidenceCode::open_vulkan_mirror_production_ready) ||
      !ExactProductionCapability(
          *vignette, CapabilityEvidenceCode::
                         open_vulkan_vignette_fixed_center_production_ready)) {
    SetReason(reason, ReasonId::capability_evidence_incomplete);
    return false;
  }
  SetReason(reason, ReasonId::none);
  return true;
}

struct ParsedProgramOptions {
  bool mirror = false;
  bool vignette = false;
  int vignette_intensity_percent = 0;
};

bool ParseEnabled(const video::CanonicalVideoStage &stage,
                  bool *enabled) noexcept {
  *enabled = true;
  for (std::size_t index = 0; index < stage.parameter_count; ++index) {
    const auto &parameter = stage.parameters[index];
    if (parameter.id == video::CanonicalVideoParameterId::enabled) {
      if (parameter.kind != video::CanonicalVideoParameterKind::boolean) {
        return false;
      }
      *enabled = parameter.boolean_value;
    }
  }
  return true;
}

bool ParseOptions(const video::NormalizedVideoExecutionIntent &intent,
                  ParsedProgramOptions *options, ReasonId *reason) noexcept {
  if (!options || intent.stage_count == 0) {
    SetReason(reason, ReasonId::malformed_contract);
    return false;
  }
  *options = {};
  bool saw_mirror = false;
  bool saw_vignette = false;
  for (std::size_t stage_index = 0; stage_index < intent.stage_count;
       ++stage_index) {
    const auto &stage = intent.stages[stage_index];
    bool enabled = true;
    if (!ParseEnabled(stage, &enabled) || !enabled ||
        !IsZeroModel(stage.model)) {
      SetReason(reason, ReasonId::malformed_contract);
      return false;
    }
    if (stage.intent.semantic == SemanticEffectId::vignette) {
      if (saw_vignette || saw_mirror) {
        SetReason(reason, ReasonId::malformed_contract);
        return false;
      }
      saw_vignette = true;
      bool have_intensity = false;
      bool have_center = false;
      for (std::size_t index = 0; index < stage.parameter_count; ++index) {
        const auto &parameter = stage.parameters[index];
        switch (parameter.id) {
        case video::CanonicalVideoParameterId::enabled:
          break;
        case video::CanonicalVideoParameterId::intensity:
          if (have_intensity ||
              parameter.kind !=
                  video::CanonicalVideoParameterKind::signed_integer ||
              parameter.signed_value < 0 || parameter.signed_value > 100) {
            SetReason(reason, ReasonId::malformed_contract);
            return false;
          }
          have_intensity = true;
          options->vignette_intensity_percent =
              static_cast<int>(parameter.signed_value);
          break;
        case video::CanonicalVideoParameterId::center_on_tracked_face:
          if (have_center ||
              parameter.kind != video::CanonicalVideoParameterKind::boolean) {
            SetReason(reason, ReasonId::malformed_contract);
            return false;
          }
          have_center = true;
          if (parameter.boolean_value) {
            SetReason(reason, ReasonId::semantic_effect_unsupported);
            return false;
          }
          break;
        default:
          SetReason(reason, ReasonId::malformed_contract);
          return false;
        }
      }
      if (!have_intensity || !have_center) {
        SetReason(reason, ReasonId::capability_evidence_incomplete);
        return false;
      }
      options->vignette = true;
    } else if (stage.intent.semantic == SemanticEffectId::mirror) {
      if (saw_mirror) {
        SetReason(reason, ReasonId::malformed_contract);
        return false;
      }
      saw_mirror = true;
      for (std::size_t index = 0; index < stage.parameter_count; ++index) {
        if (stage.parameters[index].id !=
            video::CanonicalVideoParameterId::enabled) {
          SetReason(reason, ReasonId::malformed_contract);
          return false;
        }
      }
      options->mirror = true;
    } else {
      SetReason(reason, ReasonId::semantic_effect_unsupported);
      return false;
    }
  }
  SetReason(reason, ReasonId::none);
  return options->mirror || options->vignette;
}

hal::PreparedProgram
BuildProgram(const video::VideoExecutionPlanRequest &request,
             const hal::ResolutionResult &resolution,
             const hal::DomainPlan &domain_plan) noexcept {
  hal::PreparedProgram program;
  if (domain_plan.island_count != 1)
    return program;
  const auto &island = domain_plan.islands[0];
  program.program_id = request.plan_id;
  program.owner_adapter_instance_id = island.adapter_instance_id;
  program.execution_domain = island.execution_domain;
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
      request.intent.requested_provider != SemanticEffectProviderId::invalid
          ? request.intent.requested_provider
          : resolution.selected_provider;
  program.strict_provider_selection = request.intent.strict_provider_selection;
  program.requested_execution_domain = island.execution_domain;
  program.strict_execution_domain = true;
  program.requires_upload = island.requires_upload;
  program.requires_final_readback = island.requires_final_readback;

  const auto first_input =
      ResidentImage(request.host_input, hal::Mutability::read_only);
  const auto chained_input =
      ResidentImage(request.host_output, hal::Mutability::read_only);
  const auto resident_output =
      ResidentImage(request.host_output, hal::Mutability::read_write);
  std::size_t prepared_index = 0;
  for (std::size_t resolved_index = 0; resolved_index < resolution.stage_count;
       ++resolved_index) {
    const auto &resolved = resolution.stages[resolved_index];
    if (resolved.skipped_optional)
      continue;
    auto &stage = program.stages[prepared_index];
    stage.stage_id = resolved.intent.stage_id;
    stage.semantic = resolved.intent.semantic;
    stage.provider = resolved.provider;
    stage.backend = resolved.selected_backend;
    stage.execution_domain = resolved.execution_domain;
    stage.input.kind = hal::StageIoKind::image;
    stage.input.image = prepared_index == 0 ? first_input : chained_input;
    stage.output.kind = hal::StageIoKind::image;
    stage.output.image = resident_output;
    stage.model = request.intent.stages[resolved_index].model;
    stage.required = resolved.intent.required;
    stage.failure_action = resolved.intent.failure_action;
    stage.circuit_breaker.consecutive_failure_limit = 1;
    stage.circuit_breaker.cooldown_frames = 30;
    stage.circuit_breaker.action = hal::CircuitBreakerAction::disable_stage;
    ++prepared_index;
  }
  program.stage_count = prepared_index;
  return program;
}

void AppendExactEvidence(const hal::AdapterRegistration &registration,
                         const video::NormalizedVideoExecutionIntent &intent,
                         hal::ReadinessSnapshot *readiness) noexcept {
  if (!readiness)
    return;
  for (std::size_t stage = 0; stage < intent.stage_count; ++stage) {
    const auto *capability =
        FindCapability(registration, intent.stages[stage].intent.semantic);
    if (!capability)
      continue;
    for (std::size_t evidence = 0; evidence < capability->evidence_count;
         ++evidence) {
      const auto &candidate = capability->evidence[evidence];
      bool duplicate = false;
      for (std::size_t existing = 0; existing < readiness->evidence_count;
           ++existing) {
        if (readiness->evidence[existing].code == candidate.code) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate &&
          readiness->evidence_count < readiness->evidence.size()) {
        readiness->evidence[readiness->evidence_count++] = candidate;
      }
    }
  }
}

#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
hal::SemanticCapabilityEvidence
ProductionCapability(SemanticEffectId semantic,
                     CapabilityEvidenceCode production_code) noexcept {
  hal::SemanticCapabilityEvidence capability;
  capability.semantic = semantic;
  capability.provider = kProvider;
  capability.status = StatusId::ready;
  capability.reason = ReasonId::none;
  capability.live_stage = true;
  capability.production_eligible = true;
  capability.exact_evidence = true;
  constexpr CapabilityEvidenceCode kCommon[]{
      CapabilityEvidenceCode::vulkan_build_enabled,
      CapabilityEvidenceCode::vulkan_loader_available,
      CapabilityEvidenceCode::vulkan_non_cpu_device,
      CapabilityEvidenceCode::vulkan_compute_queue_available,
      CapabilityEvidenceCode::vulkan_context_healthy,
      CapabilityEvidenceCode::vulkan_utility_kernels_ready,
      CapabilityEvidenceCode::vulkan_exact_device_shared,
      CapabilityEvidenceCode::vulkan_device_local_residency,
      CapabilityEvidenceCode::vulkan_bounded_allocations,
      CapabilityEvidenceCode::vulkan_no_cpu_layers,
      CapabilityEvidenceCode::vulkan_no_intermediate_readbacks,
      CapabilityEvidenceCode::vulkan_live_stage_present,
  };
  for (const auto code : kCommon) {
    capability.evidence[capability.evidence_count++] = {
        code,     StatusId::ready,   ReasonId::none,
        semantic, BackendId::vulkan, true};
  }
  capability.evidence[capability.evidence_count++] = {
      production_code, StatusId::ready,   ReasonId::none,
      semantic,        BackendId::vulkan, true};
  return capability;
}
#endif

class NonOwningResidentAdapter final : public hal::IResidentAdapter {
public:
  explicit NonOwningResidentAdapter(hal::IResidentAdapter *adapter) noexcept
      : adapter_(adapter) {}

  std::uint64_t adapter_instance_id() const noexcept override {
    return adapter_ ? adapter_->adapter_instance_id() : 0;
  }
  BackendId backend_id() const noexcept override {
    return adapter_ ? adapter_->backend_id() : BackendId::invalid;
  }
  StatusId Prepare(const hal::PreparedProgram &program,
                   hal::PreparedProgramHandle *candidate,
                   hal::ReadinessSnapshot *readiness,
                   ReasonId *reason) noexcept override {
    return adapter_->Prepare(program, candidate, readiness, reason);
  }
  hal::ExecuteResult
  Execute(const hal::PreparedProgramHandle &program,
          const hal::ExecuteRequest &request) noexcept override {
    return adapter_->Execute(program, request);
  }
  StatusId ResetTemporalState(const hal::PreparedProgramHandle &program,
                              ReasonId reason) noexcept override {
    return adapter_->ResetTemporalState(program, reason);
  }
  StatusId BindHostFrame(const hal::PreparedProgramHandle &program,
                         const hal::HostFrameBinding &binding,
                         hal::HostFrameTokens *tokens,
                         ReasonId *reason) noexcept override {
    return adapter_->BindHostFrame(program, binding, tokens, reason);
  }
  StatusId ResolveHostFrameOutput(const hal::PreparedProgramHandle &program,
                                  const hal::CompletionToken &completion,
                                  const hal::ResourceHandle &output,
                                  hal::HostFrameOutputResolution *resolution,
                                  ReasonId *reason) noexcept override {
    return adapter_->ResolveHostFrameOutput(program, completion, output,
                                            resolution, reason);
  }
  void DestroyPrepared(hal::PreparedProgramHandle *program) noexcept override {
    adapter_->DestroyPrepared(program);
  }
  // Factory lifetime owns the shared adapter. Replacing one coordinator plan
  // must release only that program's slot, never the device/queue island.
  void Shutdown() noexcept override {}

private:
  hal::IResidentAdapter *adapter_ = nullptr;
};

class VulkanVideoExecutionExecutor final
    : public video::IVideoExecutionExecutor {
public:
  VulkanVideoExecutionExecutor(
      const video::FrozenVideoExecutionPlan &plan,
      const hal::AdapterRegistration &registration) noexcept
      : plan_(plan), registration_(registration),
        adapter_(registration.resident_adapter) {}

  ~VulkanVideoExecutionExecutor() override { Shutdown(); }

  StatusId Configure(video::VideoAdapterPrepareEvidence *evidence) noexcept {
    ReasonId reason = ReasonId::none;
    hal::ReadinessSnapshot readiness;
    const auto status = session_.Configure(plan_.prepared_program, adapter_,
                                           &readiness, &reason);
    if (evidence) {
      evidence->status = status;
      evidence->reason = reason;
      evidence->readiness = readiness;
      evidence->candidate_resources_rolled_back = !IsSetupSuccess(status);
    }
    readiness_ = readiness;
    AppendExactEvidence(registration_, plan_.intent, &readiness_);
    prepared_ = IsSetupSuccess(status);
    return status;
  }

  video::VideoExecutionFrameResult
  Execute(const hal::HostFrameBinding &binding) noexcept override {
    video::VideoExecutionFrameResult frame;
    frame.execution_domain = plan_.domain_plan.island_count == 1
                                 ? plan_.domain_plan.islands[0].execution_domain
                                 : hal::ExecutionDomainId{};
    if (stopped_) {
      frame.status = StatusId::stopped;
      frame.reason = ReasonId::shutdown_requested;
      return frame;
    }
    if (!prepared_) {
      frame.status = quarantined_ ? StatusId::quarantined : StatusId::stale;
      frame.reason =
          quarantined_ ? ReasonId::device_lost : ReasonId::stale_program;
      return frame;
    }
    hal::HostFrameTokens tokens;
    ReasonId reason = ReasonId::none;
    const auto bind = session_.BindHostFrame(binding, &tokens, &reason);
    if (bind != StatusId::ready) {
      frame.status = bind;
      frame.reason = reason;
      return frame;
    }
    hal::ExecuteRequest request;
    request.invalidation_key = plan_.invalidation_key;
    request.execution_domain = frame.execution_domain;
    request.capture_sequence = binding.capture_sequence;
    request.input = tokens.input;
    const auto executed = session_.Execute(request);
    frame.status = executed.status;
    frame.reason = executed.reason;
    frame.completion = executed.completion;
    frame.stages = executed.stages;
    frame.stage_count = executed.stage_count;
    frame.transfers = executed.transfers;
    frame.frame_elapsed_ns = executed.frame_elapsed_ns;
    if (executed.reason == ReasonId::device_lost ||
        executed.status == StatusId::quarantined) {
      quarantined_ = true;
      prepared_ = false;
      return frame;
    }
    if (!IsExecutionSuccess(executed.status) || !executed.output_valid) {
      return frame;
    }
    reason = ReasonId::none;
    const auto resolved = session_.ResolveHostFrameOutput(
        executed.completion, executed.output, &frame.output, &reason);
    if (resolved != StatusId::ready) {
      ReasonId ignored = ReasonId::none;
      (void)session_.CommitOutput({executed.completion, false}, &ignored);
      frame.status = resolved;
      frame.reason = reason;
      return frame;
    }
    frame.output_ready = true;
    pending_ = true;
    return frame;
  }

  StatusId CommitOutput(const hal::CompletionToken &completion,
                        bool output_write_succeeded,
                        ReasonId *reason) noexcept override {
    if (stopped_) {
      SetReason(reason, ReasonId::shutdown_requested);
      return StatusId::stopped;
    }
    if (!pending_) {
      SetReason(reason, ReasonId::output_not_committed);
      return StatusId::failed;
    }
    const auto status =
        session_.CommitOutput({completion, output_write_succeeded}, reason);
    pending_ = false;
    return status;
  }

  void Invalidate(ReasonId reason) noexcept override {
    session_.Invalidate(reason);
    prepared_ = false;
    pending_ = false;
    quarantined_ = reason == ReasonId::device_lost;
  }

  StatusId ResetTemporal(ReasonId) noexcept override {
    if (stopped_)
      return StatusId::stopped;
    if (!prepared_)
      return quarantined_ ? StatusId::quarantined : StatusId::stale;
    // Production Vulkan final stages are explicitly non-temporal.
    return StatusId::ready;
  }

  void Shutdown() noexcept override {
    if (stopped_)
      return;
    session_.Invalidate(ReasonId::shutdown_requested);
    prepared_ = false;
    pending_ = false;
    stopped_ = true;
  }

  video::VideoExecutionAdapterSnapshot Snapshot() const noexcept override {
    video::VideoExecutionAdapterSnapshot snapshot;
    snapshot.status =
        stopped_ ? StatusId::stopped : session_.readiness().status;
    snapshot.reason =
        stopped_ ? ReasonId::shutdown_requested : session_.readiness().reason;
    snapshot.readiness = readiness_;
    if (!prepared_) {
      snapshot.readiness.status =
          stopped_ ? StatusId::stopped
                   : (quarantined_ ? StatusId::quarantined : StatusId::stale);
      snapshot.readiness.reason =
          stopped_ ? ReasonId::shutdown_requested
                   : (quarantined_ ? ReasonId::device_lost
                                   : ReasonId::stale_program);
      snapshot.readiness.available = false;
    }
    snapshot.telemetry = session_.telemetry();
    snapshot.latest_committed_frame = session_.latest_committed_frame();
    snapshot.execution_domain =
        plan_.domain_plan.island_count == 1
            ? plan_.domain_plan.islands[0].execution_domain
            : hal::ExecutionDomainId{};
    snapshot.plan_id = plan_.plan_id;
    snapshot.prepared = prepared_;
    snapshot.stopped = stopped_;
    return snapshot;
  }

private:
  video::FrozenVideoExecutionPlan plan_{};
  hal::AdapterRegistration registration_{};
  NonOwningResidentAdapter adapter_;
  hal::PreparedProgramSession session_;
  hal::ReadinessSnapshot readiness_{};
  bool prepared_ = false;
  bool pending_ = false;
  bool quarantined_ = false;
  bool stopped_ = false;
};

} // namespace

struct VulkanVideoExecutionFactory::Impl {
  explicit Impl(VulkanVideoExecutionFactoryConfiguration value)
      : configuration(value) {}

  VulkanVideoExecutionFactoryConfiguration configuration{};
  hal::AdapterRegistration registration{};
  bool injected = false;
  bool initialized = false;
  StatusId status = StatusId::invalid;
  ReasonId reason = ReasonId::none;

#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
  struct Production {
    kernels::UtilityKernels kernels;
    std::unique_ptr<QualifiedFinalResidentExecutor> executor;
    std::unique_ptr<VulkanHalResidentAdapter> adapter;
  };
  std::unique_ptr<Production> production;
#endif

  bool SetProductionOptions(const hal::PreparedProgram &program,
                            const ParsedProgramOptions &parsed) noexcept {
#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
    if (!injected) {
      if (!production || !production->adapter)
        return false;
      VulkanResidentProgramOptions options;
      options.program_id = program.program_id;
      options.canonical_plan_hash =
          program.invalidation_key.canonical_plan_hash;
      options.request_mirror = parsed.mirror;
      options.request_vignette = parsed.vignette;
      options.vignette_center = VulkanVignetteCenterMode::fixed_frame_center;
      options.vignette_intensity_percent = parsed.vignette_intensity_percent;
      return production->adapter->SetProgramOptions(options);
    }
#else
    (void)program;
    (void)parsed;
#endif
    return injected;
  }

  ~Impl() {
#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
    if (production) {
      if (production->adapter)
        production->adapter->Shutdown();
      production->adapter.reset();
      production->executor.reset();
      production->kernels.Shutdown();
    }
#endif
  }
};

VulkanVideoExecutionFactory::VulkanVideoExecutionFactory(
    VulkanVideoExecutionFactoryConfiguration configuration)
    : impl_(std::make_unique<Impl>(configuration)) {}

VulkanVideoExecutionFactory::VulkanVideoExecutionFactory(
    const hal::AdapterRegistration &injected_registration,
    VulkanVideoExecutionFactoryConfiguration configuration)
    : impl_(std::make_unique<Impl>(configuration)) {
  impl_->registration = injected_registration;
  impl_->injected = true;
}

VulkanVideoExecutionFactory::~VulkanVideoExecutionFactory() = default;

StatusId VulkanVideoExecutionFactory::Initialize(ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (impl_->initialized) {
    SetReason(reason, impl_->reason);
    return impl_->status;
  }
  if (impl_->configuration.factory_id == 0) {
    impl_->status = StatusId::blocked;
    impl_->reason = ReasonId::malformed_contract;
  } else if (impl_->injected) {
    impl_->initialized = ExactRegistration(impl_->registration, &impl_->reason);
    impl_->status = impl_->initialized ? StatusId::ready : StatusId::blocked;
  } else {
#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
    impl_->production = std::make_unique<Impl::Production>();
    std::string error;
    if (!impl_->production->kernels.Initialize(&error)) {
      impl_->status = StatusId::unavailable;
      impl_->reason = ReasonId::capability_evidence_incomplete;
      impl_->production.reset();
    } else {
      static std::atomic<std::uint64_t> next_adapter_id{
          UINT64_C(0x56554c4b00000000)};
      auto adapter_id = next_adapter_id.fetch_add(1, std::memory_order_relaxed);
      if (adapter_id == 0)
        adapter_id = next_adapter_id.fetch_add(1, std::memory_order_relaxed);
      impl_->production->executor =
          std::make_unique<QualifiedFinalResidentExecutor>(
              adapter_id, impl_->production->kernels);
      const auto evidence = impl_->production->executor->CapabilityEvidence();
      if (evidence.blocker != ReasonId::none ||
          !evidence.mirror_production_ready ||
          !evidence.vignette_fixed_center_production_ready) {
        impl_->status = StatusId::unavailable;
        impl_->reason = evidence.blocker == ReasonId::none
                            ? ReasonId::capability_evidence_incomplete
                            : evidence.blocker;
        impl_->production.reset();
      } else {
        impl_->production->adapter = std::make_unique<VulkanHalResidentAdapter>(
            adapter_id, *impl_->production->executor);
        impl_->registration.adapter_instance_id = adapter_id;
        impl_->registration.backend = BackendId::vulkan;
        impl_->registration.execution_domain = evidence.execution_domain;
        impl_->registration.selection_priority =
            impl_->configuration.selection_priority;
        impl_->registration.resident_adapter = impl_->production->adapter.get();
        impl_->registration.semantic_capabilities[0] = ProductionCapability(
            SemanticEffectId::vignette,
            CapabilityEvidenceCode::
                open_vulkan_vignette_fixed_center_production_ready);
        impl_->registration.semantic_capabilities[1] = ProductionCapability(
            SemanticEffectId::mirror,
            CapabilityEvidenceCode::open_vulkan_mirror_production_ready);
        impl_->registration.semantic_capability_count = 2;
        impl_->initialized =
            ExactRegistration(impl_->registration, &impl_->reason);
        impl_->status =
            impl_->initialized ? StatusId::ready : StatusId::blocked;
      }
    }
#else
    impl_->status = StatusId::unavailable;
    impl_->reason = ReasonId::capability_evidence_incomplete;
#endif
  }
  SetReason(reason, impl_->reason);
  return impl_->status;
}

bool VulkanVideoExecutionFactory::initialized() const noexcept {
  return impl_->initialized;
}

StatusId VulkanVideoExecutionFactory::initialization_status() const noexcept {
  return impl_->status;
}

ReasonId VulkanVideoExecutionFactory::initialization_reason() const noexcept {
  return impl_->reason;
}

hal::AdapterRegistration
VulkanVideoExecutionFactory::adapter_registration() const noexcept {
  return impl_->initialized ? impl_->registration : hal::AdapterRegistration{};
}

video::VideoAdapterFactoryRegistration
VulkanVideoExecutionFactory::factory_registration() noexcept {
  video::VideoAdapterFactoryRegistration factory;
  if (!impl_->initialized)
    return factory;
  factory.factory_id = impl_->configuration.factory_id;
  factory.route = video::VideoExecutionRoute::prepared_resident;
  factory.backend = BackendId::vulkan;
  factory.provider = kProvider;
  factory.adapter_instance_id = impl_->registration.adapter_instance_id;
  factory.execution_domain = impl_->registration.execution_domain;
  factory.selection_priority = impl_->configuration.selection_priority;
  factory.factory = this;
  return factory;
}

video::VideoAdapterFactoryPreflight VulkanVideoExecutionFactory::Preflight(
    const video::VideoExecutionPlanRequest &request,
    const hal::ResolutionResult &frozen_resolution,
    const hal::DomainPlan &frozen_domain_plan,
    video::VideoExecutionRoute route) noexcept {
  video::VideoAdapterFactoryPreflight preflight;
  preflight.status = impl_->status;
  preflight.reason = impl_->reason;
  preflight.backend = BackendId::vulkan;
  preflight.provider = kProvider;
  preflight.execution_domain = impl_->registration.execution_domain;
  preflight.cached_evidence_only = true;
  preflight.accepts_frozen_resolution = true;
  preflight.performs_selection = false;
  preflight.readiness.status = impl_->status;
  preflight.readiness.reason = impl_->reason;
  preflight.readiness.requested_backend = frozen_resolution.requested_backend;
  preflight.readiness.selected_backend = BackendId::vulkan;
  preflight.readiness.provider = kProvider;
  preflight.readiness.execution_domain = impl_->registration.execution_domain;

  const auto fail = [&](StatusId status, ReasonId reason) {
    preflight.status = status;
    preflight.reason = reason;
    preflight.readiness.status = status;
    preflight.readiness.reason = reason;
    preflight.readiness.available = false;
    return preflight;
  };
  if (!impl_->initialized)
    return fail(impl_->status, impl_->reason);
  if (route != video::VideoExecutionRoute::prepared_resident)
    return fail(StatusId::blocked, ReasonId::malformed_contract);
  if (request.intent.backend.mode !=
          hal::BackendRequestMode::explicit_backend ||
      request.intent.backend.backend != BackendId::vulkan ||
      !request.intent.backend.strict ||
      request.intent.requested_provider != kProvider ||
      !request.intent.strict_provider_selection) {
    return fail(StatusId::blocked, ReasonId::strict_backend_unavailable);
  }
  if (frozen_resolution.selected_backend != BackendId::vulkan ||
      frozen_resolution.selected_provider != kProvider ||
      frozen_domain_plan.island_count != 1 ||
      frozen_domain_plan.islands[0].adapter_instance_id !=
          impl_->registration.adapter_instance_id ||
      frozen_domain_plan.islands[0].resident_adapter !=
          impl_->registration.resident_adapter ||
      frozen_domain_plan.islands[0].execution_domain !=
          impl_->registration.execution_domain) {
    return fail(StatusId::blocked, ReasonId::wrong_execution_domain);
  }
  if (request.invalidation_key.generations.configuration == 0 ||
      request.invalidation_key.generations.geometry == 0 ||
      request.invalidation_key.generations.provider == 0 ||
      request.invalidation_key.generations.device == 0 ||
      request.invalidation_key.generations.context !=
          impl_->registration.execution_domain.logical_context_generation ||
      request.invalidation_key.canonical_plan_hash == 0 ||
      request.invalidation_key.execution_domain_hash == 0) {
    return fail(StatusId::stale, ReasonId::wrong_generation);
  }
  if (!IsTightHostRgb(request.host_input, hal::Mutability::read_only) ||
      !IsTightHostRgb(request.host_output, hal::Mutability::write_only) ||
      request.temporal.temporal || request.temporal.history_frames != 0 ||
      request.cpu_tail_count != 0 ||
      frozen_domain_plan.expected_transfers.uploads != 1 ||
      frozen_domain_plan.expected_transfers.final_readbacks != 1 ||
      frozen_domain_plan.expected_transfers.intermediate_readbacks != 0 ||
      frozen_domain_plan.expected_transfers.intermediate_uploads != 0 ||
      frozen_domain_plan.expected_transfers.cpu_tail_boundaries != 0) {
    return fail(StatusId::blocked, ReasonId::malformed_contract);
  }

  ParsedProgramOptions parsed;
  ReasonId parsed_reason = ReasonId::none;
  if (!ParseOptions(request.intent, &parsed, &parsed_reason))
    return fail(StatusId::blocked, parsed_reason);
  if (frozen_resolution.stage_count != request.intent.stage_count ||
      frozen_domain_plan.islands[0].stage_count != request.intent.stage_count) {
    return fail(StatusId::blocked, ReasonId::semantic_effect_unsupported);
  }
  for (std::size_t index = 0; index < frozen_resolution.stage_count; ++index) {
    const auto &stage = frozen_resolution.stages[index];
    if (stage.skipped_optional ||
        stage.intent.stage_id != request.intent.stages[index].intent.stage_id ||
        stage.intent.semantic != request.intent.stages[index].intent.semantic ||
        stage.selected_backend != BackendId::vulkan ||
        stage.provider != kProvider ||
        stage.execution_domain != impl_->registration.execution_domain ||
        (stage.intent.semantic != SemanticEffectId::mirror &&
         stage.intent.semantic != SemanticEffectId::vignette)) {
      return fail(StatusId::blocked, ReasonId::semantic_effect_unsupported);
    }
  }

  preflight.prepared_program =
      BuildProgram(request, frozen_resolution, frozen_domain_plan);
  ReasonId validation_reason = ReasonId::none;
  if (!hal::ValidatePreparedProgram(preflight.prepared_program,
                                    &validation_reason)) {
    return fail(validation_reason == ReasonId::resource_exhausted
                    ? StatusId::resource_exhausted
                    : StatusId::blocked,
                validation_reason);
  }
  preflight.status = StatusId::ready;
  preflight.reason = ReasonId::none;
  preflight.readiness.status = StatusId::ready;
  preflight.readiness.reason = ReasonId::none;
  preflight.readiness.available = true;
  preflight.readiness.ready_stage_mask =
      request.intent.stage_count == 32
          ? std::numeric_limits<std::uint32_t>::max()
          : ((std::uint32_t{1} << request.intent.stage_count) - 1u);
  AppendExactEvidence(impl_->registration, request.intent,
                      &preflight.readiness);
  return preflight;
}

std::unique_ptr<video::IVideoExecutionExecutor>
VulkanVideoExecutionFactory::Prepare(
    const video::FrozenVideoExecutionPlan &plan,
    video::VideoAdapterPrepareEvidence *evidence) noexcept {
  if (evidence)
    *evidence = {};
  const auto fail = [&](StatusId status, ReasonId reason) {
    if (evidence) {
      evidence->status = status;
      evidence->reason = reason;
      evidence->candidate_resources_rolled_back = true;
    }
    return std::unique_ptr<video::IVideoExecutionExecutor>{};
  };
  if (!impl_->initialized)
    return fail(impl_->status, impl_->reason);
  if (plan.route != video::VideoExecutionRoute::prepared_resident ||
      plan.factory_id != impl_->configuration.factory_id || plan.plan_id == 0 ||
      plan.domain_plan.island_count != 1 ||
      plan.domain_plan.islands[0].execution_domain !=
          impl_->registration.execution_domain ||
      plan.prepared_program.execution_domain !=
          impl_->registration.execution_domain ||
      plan.resolution.selected_backend != BackendId::vulkan ||
      plan.resolution.selected_provider != kProvider) {
    return fail(StatusId::blocked, ReasonId::wrong_execution_domain);
  }
  ReasonId reason = ReasonId::none;
  if (!hal::ValidatePreparedProgram(plan.prepared_program, &reason) ||
      plan.prepared_program.program_id != plan.plan_id ||
      plan.prepared_program.owner_adapter_instance_id !=
          impl_->registration.adapter_instance_id ||
      plan.prepared_program.invalidation_key != plan.invalidation_key ||
      !SameImage(plan.prepared_program.host_input, plan.host_input) ||
      !SameImage(plan.prepared_program.host_output, plan.host_output) ||
      plan.prepared_program.stage_count !=
          plan.domain_plan.islands[0].stage_count ||
      plan.prepared_program.requested_backend != BackendId::vulkan ||
      plan.prepared_program.requested_provider != kProvider ||
      !plan.prepared_program.strict_backend_selection ||
      !plan.prepared_program.strict_provider_selection ||
      !plan.prepared_program.strict_execution_domain) {
    if (reason == ReasonId::none)
      reason = ReasonId::malformed_contract;
    return fail(StatusId::blocked, reason);
  }
  ParsedProgramOptions parsed;
  if (!ParseOptions(plan.intent, &parsed, &reason))
    return fail(StatusId::blocked, reason);
  if (!impl_->SetProductionOptions(plan.prepared_program, parsed))
    return fail(StatusId::blocked, ReasonId::reconfiguration_required);

  auto candidate = std::unique_ptr<VulkanVideoExecutionExecutor>(new (
      std::nothrow) VulkanVideoExecutionExecutor(plan, impl_->registration));
  if (!candidate)
    return fail(StatusId::resource_exhausted, ReasonId::resource_exhausted);
  const auto status = candidate->Configure(evidence);
  if (!IsSetupSuccess(status)) {
    candidate->Shutdown();
    return {};
  }
  return candidate;
}

} // namespace studiocast::vulkan
