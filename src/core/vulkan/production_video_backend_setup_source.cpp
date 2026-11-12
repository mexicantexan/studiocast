#include "core/vulkan/production_video_backend_setup_source.h"

#include "core/vulkan/video_execution_factory.h"

#include <algorithm>
#include <memory>
#include <new>
#include <utility>

namespace studiocast::vulkan {
namespace {

constexpr auto kProvider =
    hal::SemanticEffectProviderId::open_vulkan_production_kernels;

bool IsSetupSuccess(hal::StatusId status) noexcept {
  return status == hal::StatusId::ready || status == hal::StatusId::degraded;
}

bool ExactTerminalBackendResolvedOutputTail(
    const video::VideoExecutionPlanRequest &plan) noexcept {
  if (plan.intent.stage_count == 0 ||
      plan.intent.stage_count > plan.intent.stages.size() ||
      plan.cpu_tail_count != 1) {
    return false;
  }
  const auto &tail = plan.cpu_tails[0];
  return tail.id == hal::CpuTailId::backend_resolved_output && tail.required &&
         tail.placement == hal::DeclaredCpuTail::Placement::after_stage &&
         tail.after_stage_id ==
             plan.intent.stages[plan.intent.stage_count - 1].intent.stage_id &&
         tail.before_stage_id == 0 &&
         tail.boundary_index == plan.intent.stage_count;
}

video::ProductionVideoResidentSetupSourceResult
Failure(hal::StatusId status, hal::ReasonId reason) noexcept {
  video::ProductionVideoResidentSetupSourceResult result;
  result.status = status;
  result.reason = reason;
  result.readiness.status = status;
  result.readiness.reason = reason;
  result.readiness.requested_backend = hal::BackendId::vulkan;
  result.readiness.selected_backend = hal::BackendId::invalid;
  result.readiness.provider = kProvider;
  result.setup_completed_outside_frame_loop = true;
  return result;
}

class VulkanSetupLeaseOwner final
    : public video::IProductionVideoSetupLeaseOwner {
public:
  explicit VulkanSetupLeaseOwner(
      std::unique_ptr<VulkanVideoExecutionFactory> factory) noexcept
      : factory_(std::move(factory)) {}

  ~VulkanSetupLeaseOwner() override {
    if (shutdown_.pending) {
      // The setup lease already made its final bounded attempt. Retain the
      // complete factory, adapter, graph, context, queue, and buffers.
      (void)factory_.release();
    }
  }

  hal::IResidentAdapter *resident_adapter() const noexcept override {
    return factory_ ? factory_->adapter_registration().resident_adapter
                    : nullptr;
  }

  video::IVideoAdapterFactory *video_adapter_factory() const noexcept override {
    return factory_.get();
  }

  void Shutdown() noexcept override { (void)RequestShutdown(); }

  video::ProductionVideoSetupLeaseShutdownResult
  RequestShutdown() noexcept override {
    shutdown_ = factory_->RequestShutdown();
    return shutdown_;
  }

private:
  std::unique_ptr<VulkanVideoExecutionFactory> factory_;
  video::ProductionVideoSetupLeaseShutdownResult shutdown_{};
};

class ProductionSetupSource final
    : public video::IProductionVideoBackendSetupSource {
public:
  video::ProductionVideoResidentSetupSourceResult AcquireResidentSetup(
      const video::ProductionVideoResidentSetup &request) noexcept override {
    const auto &caller = request.plan;
    if (request.contract_version !=
            video::kProductionVideoBackendManagerContractVersion ||
        caller.intent.backend.mode !=
            hal::BackendRequestMode::explicit_backend ||
        caller.intent.backend.backend != hal::BackendId::vulkan ||
        !caller.intent.backend.strict ||
        caller.intent.requested_provider != kProvider ||
        !caller.intent.strict_provider_selection ||
        caller.intent.stage_count == 0 ||
        caller.intent.stage_count > caller.intent.stages.size()) {
      return Failure(hal::StatusId::blocked,
                     hal::ReasonId::strict_backend_unavailable);
    }
    bool needs_matting = false;
    for (std::size_t index = 0; index < caller.intent.stage_count; ++index) {
      const auto semantic = caller.intent.stages[index].intent.semantic;
      const bool consumer =
          semantic == hal::SemanticEffectId::virtual_background_blur ||
          semantic == hal::SemanticEffectId::virtual_background_remove ||
          semantic == hal::SemanticEffectId::virtual_background_replace ||
          semantic == hal::SemanticEffectId::virtual_key_light;
      needs_matting = needs_matting || consumer;
      if (semantic != hal::SemanticEffectId::mirror &&
          semantic != hal::SemanticEffectId::vignette && !consumer) {
        return Failure(hal::StatusId::blocked,
                       hal::ReasonId::semantic_effect_unsupported);
      }
    }
    if (request.wait.timeout_nanoseconds == 0 ||
        request.wait.timeout_nanoseconds == UINT64_MAX)
      return Failure(hal::StatusId::blocked, hal::ReasonId::malformed_contract);
    if (caller.cpu_tail_count != 0 &&
        !ExactTerminalBackendResolvedOutputTail(caller)) {
      return Failure(hal::StatusId::blocked, hal::ReasonId::cpu_tail_required);
    }

    try {
      VulkanVideoExecutionFactoryConfiguration configuration;
      configuration.reservation = caller.budget;
      configuration.execution_fence_timeout_ns =
          request.wait.timeout_nanoseconds;
      configuration.input_width = static_cast<int>(caller.host_input.width);
      configuration.input_height = static_cast<int>(caller.host_input.height);
      configuration.effects =
          std::make_shared<const video::effects::BroadcastCameraEffects>(
              request.effects);
      if (needs_matting) {
        // Native graph resources are reserved before context/session
        // acquisition.
        constexpr std::uint64_t native_bytes = UINT64_C(256) * 1024 * 1024;
        const auto pixels =
            static_cast<std::uint64_t>(caller.host_input.width) *
            caller.host_input.height;
        const auto output_pixels =
            static_cast<std::uint64_t>(caller.host_output.width) *
            caller.host_output.height;
        if (pixels > (UINT64_MAX - native_bytes) / 64 ||
            output_pixels > (UINT64_MAX - native_bytes - pixels * 64) / 16 ||
            caller.budget.persistent_bytes <
                native_bytes + pixels * 64 + output_pixels * 16 ||
            caller.budget.scratch_bytes < pixels * 32 + output_pixels * 4 ||
            caller.budget.upload_staging_bytes < pixels * 4 ||
            caller.budget.readback_staging_bytes < output_pixels * 4 ||
            caller.budget.resource_count < 544 ||
            caller.budget.in_flight_slots != 1)
          return Failure(hal::StatusId::resource_exhausted,
                         hal::ReasonId::resource_exhausted);
        const auto registry = open_video::ModelPackRegistry::ScanDefault();
        const auto &id = request.effects.virtual_background.model_id;
        std::optional<open_video::ModelPack> selected;
        if (!id.empty())
          selected = registry.Find("matting", id);
        else {
          for (const auto preferred :
               {"modnet-webnn-256-fp32", "birefnet_lite"}) {
            auto candidate = registry.Find("matting", preferred);
            if (candidate && candidate->ncnn_vulkan) {
              selected = std::move(candidate);
              break;
            }
          }
          if (!selected)
            for (const auto &candidate : registry.ListModels())
              if (candidate.task == "matting" && candidate.ncnn_vulkan) {
                selected = candidate;
                break;
              }
        }
        std::string error;
        if (!selected || !selected->ncnn_vulkan || !selected->matting ||
            !open_video::ValidateProductionNcnnVulkanMattingPack(*selected,
                                                                 &error))
          return Failure(hal::StatusId::unavailable,
                         hal::ReasonId::capability_evidence_incomplete);
        const auto artifact = std::find_if(
            selected->files.begin(), selected->files.end(),
            [&](const auto &file) {
              return file.name == selected->ncnn_vulkan->param_file;
            });
        if (artifact == selected->files.end())
          return Failure(hal::StatusId::blocked,
                         hal::ReasonId::capability_evidence_incomplete);
        const auto identity =
            open_video::ValidateModelArtifactIdentity(*selected, *artifact);
        if (!identity.ok())
          return Failure(hal::StatusId::blocked,
                         hal::ReasonId::capability_evidence_incomplete);
        const auto &spec = *selected->matting;
        if (spec.input.width <= 0 || spec.input.height <= 0 ||
            spec.output.width <= 0 || spec.output.height <= 0 ||
            spec.input.width > 4096 || spec.input.height > 4096 ||
            spec.output.width > 4096 || spec.output.height > 4096)
          return Failure(hal::StatusId::blocked,
                         hal::ReasonId::malformed_descriptor);
        const auto model_bytes =
            static_cast<std::uint64_t>(spec.input.width) *
                static_cast<std::uint64_t>(spec.input.height) * 12u +
            static_cast<std::uint64_t>(spec.output.width) *
                static_cast<std::uint64_t>(spec.output.height) * 4u;
        if (caller.budget.persistent_bytes <
            native_bytes + pixels * 64 + output_pixels * 16 + model_bytes)
          return Failure(hal::StatusId::resource_exhausted,
                         hal::ReasonId::resource_exhausted);
        configuration.matting_identity.model_id_hash =
            identity.identity.model_id_sha256;
        configuration.matting_identity.artifact_set_hash =
            identity.identity.artifact_set_sha256;
        configuration.matting_identity.artifact_generation =
            identity.identity.registry_generation;
        configuration.matting_pack =
            std::make_shared<const open_video::ModelPack>(std::move(*selected));
        configuration.maximum_synchronizations_per_frame =
            video::kMaxPerFrameSynchronizationsV1;
        configuration.minimum_synchronizations_per_frame = 2;
      }
      auto factory =
          std::make_unique<VulkanVideoExecutionFactory>(configuration);
      hal::ReasonId reason = hal::ReasonId::none;
      const auto initialized = factory->Initialize(&reason);
      if (!IsSetupSuccess(initialized))
        return Failure(initialized, reason);

      const auto adapter = factory->adapter_registration();
      const auto factory_registration = factory->factory_registration();
      if (adapter.resident_adapter == nullptr ||
          factory_registration.factory == nullptr) {
        return Failure(hal::StatusId::blocked,
                       hal::ReasonId::capability_evidence_incomplete);
      }

      video::ProductionVideoResidentSetupSourceResult result;
      result.execution_kind = video::ProductionVideoResidentSetupSourceResult::
          ExecutionKind::resident_island;
      result.canonical_effects = request.effects;
      result.setup_completed_outside_frame_loop = true;
      result.plan = caller;
      result.resolution_registration = adapter;
      if (needs_matting) {
        for (std::size_t i = 0; i < result.plan.intent.stage_count; ++i) {
          auto &stage = result.plan.intent.stages[i];
          if (stage.intent.semantic == hal::SemanticEffectId::mirror ||
              stage.intent.semantic == hal::SemanticEffectId::vignette)
            continue;
          const auto &identity = configuration.matting_identity;
          if (stage.model.artifact_generation != 0 &&
              (stage.model.artifact_generation !=
                   identity.artifact_generation ||
               stage.model.model_id_hash != identity.model_id_hash ||
               stage.model.artifact_set_hash != identity.artifact_set_hash))
            return Failure(hal::StatusId::blocked,
                           hal::ReasonId::reconfiguration_required);
          stage.model = identity;
          if (stage.intent.semantic !=
              hal::SemanticEffectId::virtual_key_light) {
            auto &selection =
                result.model_selections[result.model_selection_count++];
            selection.semantic = stage.intent.semantic;
            selection.requested_model_id =
                request.effects.virtual_background.model_id;
            selection.selected_model_id = configuration.matting_pack->id;
            selection.default_selected = selection.requested_model_id.empty();
            selection.artifact = identity;
          }
        }
        result.plan.invalidation_key.generations.model =
            configuration.matting_identity.artifact_generation;
        std::uint64_t hash = UINT64_C(1469598103934665603);
        for (const auto value :
             configuration.matting_identity.artifact_set_hash) {
          hash ^= value;
          hash *= UINT64_C(1099511628211);
        }
        result.plan.invalidation_key.model_artifact_hash = hash ? hash : 1;
      }
      if (ExactTerminalBackendResolvedOutputTail(result.plan)) {
        result.plan.cpu_tails[0] = {};
        result.plan.cpu_tail_count = 0;
      }
      auto &generations = result.plan.invalidation_key.generations;
      if (generations.model == 0)
        generations.model = 1;
      if (generations.provider == 0)
        generations.provider = 1;
      if (generations.device == 0)
        generations.device = 1;
      generations.context = adapter.execution_domain.logical_context_generation;
      result.plan.invalidation_key.execution_domain_hash =
          video::ProductionVideoExecutionDomainHash(adapter.execution_domain);

      video::ProductionVideoSetupLeaseSnapshot snapshot;
      snapshot.state = video::ProductionVideoSetupLeaseState::live;
      snapshot.status = initialized;
      snapshot.reason = reason;
      snapshot.backend = hal::BackendId::vulkan;
      snapshot.execution_domain = adapter.execution_domain;
      snapshot.generations = generations;
      snapshot.reserved_budget = result.plan.budget;
      snapshot.providers[snapshot.provider_count++] = kProvider;
      for (std::size_t index = 0; index < result.plan.intent.stage_count;
           ++index) {
        const auto &stage = result.plan.intent.stages[index];
        if (stage.model.artifact_generation == 0)
          continue;
        if (snapshot.artifact_count == snapshot.artifacts.size())
          return Failure(hal::StatusId::resource_exhausted,
                         hal::ReasonId::resource_exhausted);
        snapshot.artifacts[snapshot.artifact_count++] = {
            stage.intent.semantic, kProvider, stage.model};
      }
      snapshot.adapter = adapter;
      snapshot.factory = factory_registration;
      snapshot.readiness.status = initialized;
      snapshot.readiness.reason = reason;
      snapshot.readiness.requested_backend = hal::BackendId::vulkan;
      snapshot.readiness.selected_backend = hal::BackendId::vulkan;
      snapshot.readiness.provider = kProvider;
      snapshot.readiness.execution_domain = adapter.execution_domain;
      snapshot.readiness.available = true;
      snapshot.readiness.degraded = initialized == hal::StatusId::degraded;

      auto owner = std::make_unique<VulkanSetupLeaseOwner>(std::move(factory));
      result.status = video::ProductionVideoSetupLease::Acquire(
          snapshot, std::move(owner), &result.lease, &result.lease_acquire,
          &result.reason);
      result.readiness = result.lease_acquire.readiness;
      if (!IsSetupSuccess(result.status) || !result.lease.live())
        return result;
      result.reason = result.lease.snapshot().reason;
      result.readiness = result.lease.snapshot().readiness;
      return result;
    } catch (const std::bad_alloc &) {
      return Failure(hal::StatusId::resource_exhausted,
                     hal::ReasonId::resource_exhausted);
    } catch (...) {
      return Failure(hal::StatusId::blocked,
                     hal::ReasonId::capability_evidence_incomplete);
    }
  }
};

} // namespace

std::unique_ptr<video::IProductionVideoSetupLeaseOwner>
CreateProductionVulkanSetupLeaseOwner(
    std::unique_ptr<VulkanVideoExecutionFactory> factory) {
  if (!factory)
    return nullptr;
  return std::make_unique<VulkanSetupLeaseOwner>(std::move(factory));
}

std::unique_ptr<video::IProductionVideoBackendSetupSource>
CreateProductionVideoBackendSetupSource() noexcept {
  try {
    return std::make_unique<ProductionSetupSource>();
  } catch (...) {
    return nullptr;
  }
}

} // namespace studiocast::vulkan
