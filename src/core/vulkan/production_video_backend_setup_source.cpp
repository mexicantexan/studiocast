#include "core/vulkan/production_video_backend_setup_source.h"

#include "core/video/replace_background_cache_policy.h"
#include "core/vulkan/video_execution_factory.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <new>
#include <utility>

namespace studiocast::vulkan {
namespace {

constexpr auto kProvider =
    hal::SemanticEffectProviderId::open_vulkan_production_kernels;

std::uint32_t SemanticBit(hal::SemanticEffectId semantic) noexcept {
  const auto value = static_cast<std::uint32_t>(semantic);
  return value < 32 ? (std::uint32_t{1} << value) : 0;
}

video::ProductionVideoResidentSetupSourceResult
NeutralUnavailable(const video::ProductionVideoResidentSetup &request) {
  static std::atomic<std::uint64_t> next_adapter{UINT64_C(0x56554e5500000000)};
  video::ProductionVideoResidentSetupSourceResult result;
  result.plan = request.plan;
  result.canonical_effects = request.effects;
  result.plan.cpu_tails = {};
  result.plan.cpu_tail_count = 0;
  result.plan.invalidation_key.execution_domain_hash =
      video::kProductionVideoNeutralPassthroughExecutionHash;
  auto &generations = result.plan.invalidation_key.generations;
  if (!generations.model)
    generations.model = 1;
  if (!generations.provider)
    generations.provider = 1;
  if (!generations.device)
    generations.device = 1;
  if (!generations.context)
    generations.context = 1;
  result.status = hal::StatusId::degraded;
  result.reason = hal::ReasonId::capability_evidence_incomplete;
  result.readiness.status = result.status;
  result.readiness.reason = result.reason;
  result.readiness.requested_backend = hal::BackendId::vulkan;
  result.readiness.available = true;
  result.readiness.degraded = true;
  result.setup_completed_outside_frame_loop = true;
  auto &registration = result.resolution_registration;
  registration.adapter_instance_id =
      next_adapter.fetch_add(1, std::memory_order_relaxed);
  registration.backend = hal::BackendId::vulkan;
  auto &domain = registration.execution_domain;
  domain.backend = hal::BackendId::vulkan;
  domain.api_family = hal::ApiFamily::vulkan;
  domain.allocator_family = hal::AllocatorFamily::vulkan;
  domain.memory_handle_family = hal::MemoryHandleFamily::vulkan_buffer;
  domain.synchronization_protocol =
      hal::SynchronizationProtocol::vulkan_queue_fence;
  domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
  domain.adapter_instance_id = registration.adapter_instance_id;
  domain.logical_context_generation = generations.context;
  domain.queue_or_stream_identity = registration.adapter_instance_id;
  domain.physical_device.vendor_id = UINT32_C(0x535443);
  domain.physical_device.device_id = 1;
  domain.physical_device.stable_id[0] = 0x56;
  domain.physical_device.stable_id[1] = 0x55;
  registration.selection_priority = 300;
  registration.semantic_capability_count = request.plan.intent.stage_count;
  for (std::size_t i = 0; i < registration.semantic_capability_count; ++i) {
    auto &capability = registration.semantic_capabilities[i];
    capability.semantic = request.plan.intent.stages[i].intent.semantic;
    capability.provider = kProvider;
    capability.status = hal::StatusId::unavailable;
    capability.reason = hal::ReasonId::capability_evidence_incomplete;
  }
  return result;
}

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
    bool needs_denoise = false, needs_eye = false, needs_face = false;
    for (std::size_t index = 0; index < caller.intent.stage_count; ++index) {
      const auto semantic = caller.intent.stages[index].intent.semantic;
      const bool consumer =
          semantic == hal::SemanticEffectId::virtual_background_blur ||
          semantic == hal::SemanticEffectId::virtual_background_remove ||
          semantic == hal::SemanticEffectId::virtual_background_replace ||
          semantic == hal::SemanticEffectId::virtual_key_light;
      needs_matting = needs_matting || consumer;
      needs_denoise |= semantic == hal::SemanticEffectId::video_noise_removal;
      needs_eye |= semantic == hal::SemanticEffectId::eye_contact;
      needs_face |= semantic == hal::SemanticEffectId::auto_frame ||
                    semantic == hal::SemanticEffectId::eye_contact;
      needs_matting |= semantic == hal::SemanticEffectId::auto_frame;
      if (semantic != hal::SemanticEffectId::mirror &&
          semantic != hal::SemanticEffectId::vignette &&
          semantic != hal::SemanticEffectId::video_noise_removal &&
          semantic != hal::SemanticEffectId::eye_contact &&
          semantic != hal::SemanticEffectId::auto_frame && !consumer) {
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
      configuration.output_width = static_cast<int>(caller.host_output.width);
      configuration.output_height = static_cast<int>(caller.host_output.height);
      configuration.effects =
          std::make_shared<const video::effects::BroadcastCameraEffects>(
              request.effects);
      std::optional<open_video::ModelPackRegistry> selected_registry;
      if (needs_matting || needs_denoise || needs_face)
        selected_registry = open_video::ModelPackRegistry::ScanDefault();
      auto runtime_effects = request.effects;
      const auto unavailable = [&](hal::SemanticEffectId semantic) {
        for (std::size_t i = 0; i < caller.intent.stage_count; ++i)
          if (caller.intent.stages[i].intent.semantic == semantic)
            configuration.unavailable_semantic_mask |= SemanticBit(semantic);
        switch (semantic) {
        case hal::SemanticEffectId::video_noise_removal:
          runtime_effects.video_noise_removal.enabled = false;
          break;
        case hal::SemanticEffectId::eye_contact:
          runtime_effects.eye_contact.enabled = false;
          break;
        case hal::SemanticEffectId::auto_frame:
          runtime_effects.auto_frame.enabled = false;
          break;
        case hal::SemanticEffectId::virtual_background_blur:
        case hal::SemanticEffectId::virtual_background_remove:
        case hal::SemanticEffectId::virtual_background_replace:
          runtime_effects.virtual_background.mode =
              video::effects::VirtualBackgroundMode::none;
          break;
        case hal::SemanticEffectId::virtual_key_light:
          runtime_effects.virtual_key_light.enabled = false;
          break;
        case hal::SemanticEffectId::vignette:
          runtime_effects.vignette.enabled = false;
          break;
        default:
          break;
        }
      };
      const auto unavailable_matting = [&]() {
        for (const auto semantic :
             {hal::SemanticEffectId::virtual_background_blur,
              hal::SemanticEffectId::virtual_background_remove,
              hal::SemanticEffectId::virtual_background_replace,
              hal::SemanticEffectId::virtual_key_light})
          unavailable(semantic);
      };
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
        const auto &registry = *selected_registry;
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
        const bool valid_pack =
            selected && selected->ncnn_vulkan && selected->matting &&
            open_video::ValidateProductionNcnnVulkanMattingPack(*selected,
                                                                &error);
        if (!valid_pack) {
          unavailable_matting();
        } else {
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
              std::make_shared<const open_video::ModelPack>(
                  std::move(*selected));
          configuration.maximum_synchronizations_per_frame =
              video::kMaxPerFrameSynchronizationsV1;
          configuration.minimum_synchronizations_per_frame = 2;
        }
      }
      if (needs_denoise || needs_face) {
        const auto &registry = *selected_registry;
        const auto resolve =
            [&](const char *task, const std::string &requested,
                const char *preferred,
                std::shared_ptr<const open_video::ModelPack> *pack,
                hal::ModelArtifactIdentity *identity, bool predictor = false) {
              auto selected = registry.Find(
                  task, requested.empty() ? preferred : requested);
              if (!selected)
                return false;
              std::string error;
              if (!predictor) {
                if (selected->ncnn_vulkan_networks.empty())
                  return false;
                for (const auto &network : selected->ncnn_vulkan_networks)
                  if (!open_video::ValidateProductionNcnnVulkanNetworkPack(
                          *selected, network, &error))
                    return false;
              }
              const auto file = std::find_if(
                  selected->files.begin(), selected->files.end(),
                  [&](const auto &artifact) {
                    return artifact.kind ==
                           (predictor ? "dlib_shape_predictor" : "ncnn_param");
                  });
              if (file == selected->files.end())
                return false;
              const auto validated =
                  open_video::ValidateModelArtifactIdentity(*selected, *file);
              if (!validated.ok())
                return false;
              identity->model_id_hash = validated.identity.model_id_sha256;
              identity->artifact_set_hash =
                  validated.identity.artifact_set_sha256;
              identity->artifact_generation =
                  validated.identity.registry_generation;
              *pack = std::make_shared<const open_video::ModelPack>(
                  std::move(*selected));
              return true;
            };
        if (needs_denoise &&
            !resolve("video_denoise",
                     request.effects.video_noise_removal.model_id,
                     "fastdvdnet_sigma15", &configuration.denoise_pack,
                     &configuration.denoise_identity))
          unavailable(hal::SemanticEffectId::video_noise_removal);
        if (needs_face &&
            !resolve("face_detection", request.effects.auto_frame.model_id,
                     "yunet_opencv_zoo_2023mar_fp32", &configuration.face_pack,
                     &configuration.face_identity)) {
          unavailable(hal::SemanticEffectId::eye_contact);
          unavailable(hal::SemanticEffectId::auto_frame);
        }
        if (needs_eye &&
            !resolve("eye_contact", request.effects.eye_contact.model_id,
                     "gaze_correction_cam_flx_v0_1_1", &configuration.eye_pack,
                     &configuration.eye_identity))
          unavailable(hal::SemanticEffectId::eye_contact);
        if (needs_eye && configuration.eye_pack) {
          std::string landmark_id;
          for (const auto &dependency : configuration.eye_pack->depends_on)
            if (dependency.rfind("face_landmarks:", 0) == 0)
              landmark_id = dependency.substr(15);
          if (!resolve("face_landmarks", landmark_id, "dlib_68_ibug_300w",
                       &configuration.landmarks_pack,
                       &configuration.landmarks_identity, true))
            unavailable(hal::SemanticEffectId::eye_contact);
        }
        configuration.maximum_synchronizations_per_frame =
            video::kMaxPerFrameSynchronizationsV1;
        configuration.minimum_synchronizations_per_frame = 1;
      }
      if (!runtime_effects.eye_contact.enabled) {
        configuration.eye_pack.reset();
        configuration.eye_identity = {};
        configuration.landmarks_pack.reset();
        configuration.landmarks_identity = {};
      }
      if (!runtime_effects.eye_contact.enabled &&
          !runtime_effects.auto_frame.enabled) {
        configuration.face_pack.reset();
        configuration.face_identity = {};
      }
      if (!runtime_effects.auto_frame.enabled &&
          request.effects.auto_frame.enabled &&
          request.effects.vignette.enabled &&
          request.effects.vignette.center_on_tracked_face)
        unavailable(hal::SemanticEffectId::vignette);
      if (runtime_effects.virtual_background.mode ==
          video::effects::VirtualBackgroundMode::replace) {
        video::detail::PreparedReplaceBackgroundSource asset;
        if (!video::detail::PrepareReplaceBackgroundSourceForConfig(
                runtime_effects.virtual_background.replace_path, &asset,
                nullptr))
          unavailable(hal::SemanticEffectId::virtual_background_replace);
      }
      bool any_selected = false;
      for (std::size_t i = 0; i < caller.intent.stage_count; ++i) {
        const auto &stage = caller.intent.stages[i].intent;
        const bool missing = (configuration.unavailable_semantic_mask &
                              SemanticBit(stage.semantic)) != 0;
        if (missing && stage.required)
          return Failure(hal::StatusId::unavailable,
                         hal::ReasonId::required_stage_failed);
        any_selected |= !missing;
      }
      if (!any_selected)
        return NeutralUnavailable(request);
      std::unique_ptr<VulkanVideoExecutionFactory> factory;
      hal::ReasonId reason = hal::ReasonId::none;
      auto initialized = hal::StatusId::unavailable;
      for (std::size_t attempt = 0; attempt < hal::kMaxPreparedStages;
           ++attempt) {
        configuration.effects =
            std::make_shared<const video::effects::BroadcastCameraEffects>(
                runtime_effects);
        factory = std::make_unique<VulkanVideoExecutionFactory>(configuration);
        initialized = factory->Initialize(&reason);
        if (IsSetupSuccess(initialized))
          break;
        const auto component = factory->initialization_failed_semantic();
        const auto proof = factory->RequestShutdown();
        if (!proof.quiescent || !proof.driver_context_release_permitted)
          return Failure(hal::StatusId::quarantined, proof.reason);
        if (component == hal::SemanticEffectId::invalid)
          return Failure(initialized, reason);
        if (component == hal::SemanticEffectId::video_noise_removal) {
          unavailable(component);
          configuration.denoise_pack.reset();
          configuration.denoise_identity = {};
        } else if (component == hal::SemanticEffectId::eye_contact) {
          unavailable(component);
          configuration.eye_pack.reset();
          configuration.eye_identity = {};
          configuration.landmarks_pack.reset();
          configuration.landmarks_identity = {};
          if (!runtime_effects.auto_frame.enabled) {
            configuration.face_pack.reset();
            configuration.face_identity = {};
          }
        } else if (component == hal::SemanticEffectId::auto_frame) {
          unavailable(hal::SemanticEffectId::auto_frame);
          unavailable(hal::SemanticEffectId::eye_contact);
          configuration.face_pack.reset();
          configuration.face_identity = {};
          configuration.eye_pack.reset();
          configuration.eye_identity = {};
          configuration.landmarks_pack.reset();
          configuration.landmarks_identity = {};
          if (request.effects.vignette.center_on_tracked_face)
            unavailable(hal::SemanticEffectId::vignette);
        } else if (component == hal::SemanticEffectId::vignette) {
          unavailable(component);
        } else {
          unavailable_matting();
          configuration.matting_pack.reset();
          configuration.matting_identity = {};
        }
        bool remaining = false;
        for (std::size_t i = 0; i < caller.intent.stage_count; ++i) {
          const auto &stage = caller.intent.stages[i].intent;
          const bool missing = (configuration.unavailable_semantic_mask &
                                SemanticBit(stage.semantic)) != 0;
          if (missing && stage.required)
            return Failure(hal::StatusId::unavailable,
                           hal::ReasonId::required_stage_failed);
          remaining |= !missing;
        }
        if (!remaining)
          return NeutralUnavailable(request);
        factory.reset();
      }
      if (!factory || !IsSetupSuccess(initialized))
        return Failure(initialized, reason);

      const auto adapter = factory->adapter_registration();
      const auto factory_registration = factory->factory_registration();
      if (adapter.resident_adapter == nullptr ||
          factory_registration.factory == nullptr) {
        return Failure(hal::StatusId::blocked,
                       hal::ReasonId::capability_evidence_incomplete);
      }

      const auto selectable = [&](hal::SemanticEffectId semantic) {
        for (std::size_t i = 0; i < adapter.semantic_capability_count; ++i)
          if (adapter.semantic_capabilities[i].semantic == semantic)
            return hal::IsSelectableSemanticCapability(
                adapter.semantic_capabilities[i]);
        return false;
      };
      bool selected_after_features = false, required_after_features = false;
      for (std::size_t i = 0; i < caller.intent.stage_count; ++i) {
        const auto &stage = caller.intent.stages[i].intent;
        const bool selected = selectable(stage.semantic);
        selected_after_features |= selected;
        required_after_features |= stage.required && !selected;
      }
      if (!selected_after_features || required_after_features) {
        const auto proof = factory->RequestShutdown();
        if (!proof.quiescent || !proof.driver_context_release_permitted)
          return Failure(hal::StatusId::quarantined, proof.reason);
        if (required_after_features)
          return Failure(hal::StatusId::unavailable,
                         hal::ReasonId::required_stage_failed);
        return NeutralUnavailable(request);
      }

      video::ProductionVideoResidentSetupSourceResult result;
      result.execution_kind = video::ProductionVideoResidentSetupSourceResult::
          ExecutionKind::resident_island;
      result.canonical_effects = request.effects;
      result.setup_completed_outside_frame_loop = true;
      result.plan = caller;
      result.resolution_registration = adapter;
      if (configuration.matting_pack) {
        for (std::size_t i = 0; i < result.plan.intent.stage_count; ++i) {
          auto &stage = result.plan.intent.stages[i];
          if (!selectable(stage.intent.semantic))
            continue;
          if (stage.intent.semantic !=
                  hal::SemanticEffectId::virtual_background_blur &&
              stage.intent.semantic !=
                  hal::SemanticEffectId::virtual_background_remove &&
              stage.intent.semantic !=
                  hal::SemanticEffectId::virtual_background_replace &&
              stage.intent.semantic != hal::SemanticEffectId::virtual_key_light)
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
      for (std::size_t i = 0; i < result.plan.intent.stage_count; ++i) {
        auto &stage = result.plan.intent.stages[i];
        if (!selectable(stage.intent.semantic))
          continue;
        const hal::ModelArtifactIdentity *identity = nullptr;
        const std::shared_ptr<const open_video::ModelPack> *pack = nullptr;
        const std::string *requested = nullptr;
        if (stage.intent.semantic ==
            hal::SemanticEffectId::video_noise_removal) {
          identity = &configuration.denoise_identity;
          pack = &configuration.denoise_pack;
          requested = &request.effects.video_noise_removal.model_id;
        } else if (stage.intent.semantic ==
                   hal::SemanticEffectId::eye_contact) {
          identity = &configuration.eye_identity;
          pack = &configuration.eye_pack;
          requested = &request.effects.eye_contact.model_id;
        } else if (stage.intent.semantic == hal::SemanticEffectId::auto_frame) {
          identity = &configuration.face_identity;
          pack = &configuration.face_pack;
          requested = &request.effects.auto_frame.model_id;
        }
        if (!identity)
          continue;
        if (!*pack || !identity->artifact_generation ||
            (stage.model.artifact_generation &&
             (stage.model.artifact_generation !=
                  identity->artifact_generation ||
              stage.model.model_id_hash != identity->model_id_hash ||
              stage.model.artifact_set_hash != identity->artifact_set_hash)))
          return Failure(hal::StatusId::blocked,
                         hal::ReasonId::reconfiguration_required);
        stage.model = *identity;
        auto &selection =
            result.model_selections[result.model_selection_count++];
        selection.semantic = stage.intent.semantic;
        selection.requested_model_id = *requested;
        selection.selected_model_id = (*pack)->id;
        selection.default_selected = requested->empty();
        selection.artifact = *identity;
      }
      if (needs_denoise || needs_face) {
        std::uint64_t hash = UINT64_C(1469598103934665603);
        for (const auto *identity :
             {&configuration.matting_identity, &configuration.denoise_identity,
              &configuration.face_identity, &configuration.landmarks_identity,
              &configuration.eye_identity}) {
          if (!identity->artifact_generation)
            continue;
          result.plan.invalidation_key.generations.model =
              std::max(result.plan.invalidation_key.generations.model,
                       identity->artifact_generation);
          for (const auto byte : identity->artifact_set_hash) {
            hash ^= byte;
            hash *= UINT64_C(1099511628211);
          }
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
        if (!selectable(stage.intent.semantic))
          continue;
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
