#include "core/vulkan/video_execution_factory.h"
#include "core/vulkan/hal_resident_adapter.h"
#include "core/vulkan/video_execution_factory_internal.h"

#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
#include "core/open_video/ncnn_vulkan_context.h"
#include "core/vulkan/hal_resident_adapter.h"
#include "core/vulkan/kernels/utility_kernels.h"
#include "core/vulkan/qualified_final_resident_executor.h"
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
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
        (record.semantic == capability.semantic ||
         record.semantic == SemanticEffectId::invalid) &&
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
      CapabilityEvidenceCode::vulkan_physical_device_available,
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
      (capability.supports_tracked_vignette_center &&
       capability.semantic != SemanticEffectId::vignette) ||
      capability.cpu_tail != hal::CpuTailId::invalid) {
    return false;
  }
  for (const auto code : kCommon) {
    if (!HasSatisfiedEvidence(capability, code))
      return false;
  }
  return HasSatisfiedEvidence(capability, production_code);
}

bool IsMattingSemantic(SemanticEffectId semantic) noexcept;

bool UnavailableCapability(
    const hal::SemanticCapabilityEvidence &capability) noexcept {
  return (capability.status == StatusId::blocked ||
          capability.status == StatusId::unavailable) &&
         capability.reason != ReasonId::none &&
         !capability.production_eligible && capability.provider == kProvider;
}
std::uint32_t SemanticBit(SemanticEffectId semantic) noexcept {
  const auto value = static_cast<std::uint32_t>(semantic);
  return value < 32 ? (std::uint32_t{1} << value) : 0;
}

CapabilityEvidenceCode ModelRuntimeCode(SemanticEffectId semantic) noexcept {
  if (semantic == SemanticEffectId::video_noise_removal)
    return CapabilityEvidenceCode::vulkan_denoise_runtime_ready;
  if (semantic == SemanticEffectId::eye_contact)
    return CapabilityEvidenceCode::vulkan_eye_runtime_ready;
  if (semantic == SemanticEffectId::auto_frame)
    return CapabilityEvidenceCode::vulkan_tracking_gpu_ready;
  return CapabilityEvidenceCode::vulkan_matting_runtime_ready;
}

bool ExactRegistration(const hal::AdapterRegistration &registration,
                       ReasonId *reason) noexcept {
  if (registration.semantic_capability_count >
          registration.semantic_capabilities.size() ||
      registration.adapter_instance_id == 0 ||
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
      (!UnavailableCapability(*vignette) &&
       !ExactProductionCapability(
           *vignette,
           CapabilityEvidenceCode::
               open_vulkan_vignette_fixed_center_production_ready))) {
    SetReason(reason, ReasonId::capability_evidence_incomplete);
    return false;
  }
  for (std::size_t i = 0; i < registration.semantic_capability_count; ++i) {
    const auto &capability = registration.semantic_capabilities[i];
    if (UnavailableCapability(capability))
      continue;
    if (!hal::HasVulkanProductionEvidence(capability)) {
      SetReason(reason, ReasonId::capability_evidence_incomplete);
      return false;
    }
    if (capability.semantic == SemanticEffectId::mirror ||
        capability.semantic == SemanticEffectId::vignette)
      continue;
    if ((!IsMattingSemantic(capability.semantic) &&
         capability.semantic != SemanticEffectId::video_noise_removal &&
         capability.semantic != SemanticEffectId::eye_contact &&
         capability.semantic != SemanticEffectId::auto_frame) ||
        !ExactProductionCapability(capability,
                                   ModelRuntimeCode(capability.semantic)) ||
        !HasSatisfiedEvidence(
            capability, CapabilityEvidenceCode::vulkan_model_artifacts_valid)) {
      SetReason(reason, ReasonId::capability_evidence_incomplete);
      return false;
    }
  }
  // The shared registry also rejects malformed status/evidence shapes,
  // duplicate proof records and duplicate semantic/provider entries.
  hal::StaticAdapterRegistry validator;
  if (validator.Register(registration, reason) != StatusId::ready)
    return false;
  SetReason(reason, ReasonId::none);
  return true;
}

bool IsMattingSemantic(SemanticEffectId semantic) noexcept {
  return semantic == SemanticEffectId::virtual_background_blur ||
         semantic == SemanticEffectId::virtual_background_remove ||
         semantic == SemanticEffectId::virtual_background_replace ||
         semantic == SemanticEffectId::virtual_key_light;
}

bool IsModelSemantic(SemanticEffectId semantic) noexcept {
  return IsMattingSemantic(semantic) ||
         semantic == SemanticEffectId::video_noise_removal ||
         semantic == SemanticEffectId::eye_contact ||
         semantic == SemanticEffectId::auto_frame;
}

const hal::ModelArtifactIdentity *
ModelIdentity(const VulkanVideoExecutionFactoryConfiguration &configuration,
              SemanticEffectId semantic) noexcept {
  if (IsMattingSemantic(semantic))
    return &configuration.matting_identity;
  if (semantic == SemanticEffectId::video_noise_removal)
    return &configuration.denoise_identity;
  if (semantic == SemanticEffectId::eye_contact)
    return &configuration.eye_identity;
  if (semantic == SemanticEffectId::auto_frame)
    return &configuration.face_identity;
  return nullptr;
}
bool TemporalIntent(
    const video::NormalizedVideoExecutionIntent &intent) noexcept {
  for (std::size_t i = 0; i < intent.stage_count; ++i) {
    const auto semantic = intent.stages[i].intent.semantic;
    if (semantic == SemanticEffectId::video_noise_removal ||
        semantic == SemanticEffectId::eye_contact ||
        semantic == SemanticEffectId::auto_frame)
      return true;
  }
  return false;
}

video::NormalizedVideoExecutionIntent
ActiveIntent(const video::NormalizedVideoExecutionIntent &intent,
             const hal::ResolutionResult &resolution) noexcept {
  auto active = intent;
  active.stages = {};
  active.stage_count = 0;
  if (intent.stage_count != resolution.stage_count)
    return active;
  for (std::size_t i = 0; i < intent.stage_count; ++i)
    if (!resolution.stages[i].skipped_optional)
      active.stages[active.stage_count++] = intent.stages[i];
  return active;
}

struct ParsedProgramOptions {
  VulkanResidentProgramOptions matting{};
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
        (!IsModelSemantic(stage.intent.semantic) &&
         !IsZeroModel(stage.model))) {
      SetReason(reason, ReasonId::malformed_contract);
      return false;
    }
    if (stage.intent.semantic == SemanticEffectId::video_noise_removal ||
        stage.intent.semantic == SemanticEffectId::eye_contact ||
        stage.intent.semantic == SemanticEffectId::auto_frame) {
      auto &parsed = options->matting;
      const auto semantic = stage.intent.semantic;
      const bool denoise = semantic == SemanticEffectId::video_noise_removal;
      const bool eye = semantic == SemanticEffectId::eye_contact;
      if (saw_mirror || saw_vignette ||
          (denoise && (parsed.request_denoise || parsed.request_eye ||
                       parsed.background != SemanticEffectId::invalid ||
                       parsed.request_keylight || parsed.request_auto_frame)) ||
          (eye && (parsed.request_eye ||
                   parsed.background != SemanticEffectId::invalid ||
                   parsed.request_keylight || parsed.request_auto_frame)) ||
          (!denoise && !eye && parsed.request_auto_frame)) {
        SetReason(reason, ReasonId::malformed_contract);
        return false;
      }
      unsigned seen = 0;
      for (std::size_t i = 0; i < stage.parameter_count; ++i) {
        const auto &p = stage.parameters[i];
        if (p.id == video::CanonicalVideoParameterId::enabled)
          continue;
        unsigned bit = 0;
        if (p.id == video::CanonicalVideoParameterId::strength &&
            p.kind == video::CanonicalVideoParameterKind::signed_integer &&
            p.signed_value >= 0 && p.signed_value <= 100) {
          if (denoise)
            parsed.denoise_strength = static_cast<int>(p.signed_value);
          else if (eye)
            parsed.eye_strength = static_cast<int>(p.signed_value);
          else
            parsed.auto_frame_strength = static_cast<int>(p.signed_value);
          bit = 1;
        } else if (eye &&
                   p.id ==
                       video::CanonicalVideoParameterId::look_away_enabled &&
                   p.kind == video::CanonicalVideoParameterKind::boolean) {
          parsed.eye_look_away = p.boolean_value;
          bit = 2;
        } else if (!denoise && !eye &&
                   p.id == video::CanonicalVideoParameterId::smoothing &&
                   p.kind ==
                       video::CanonicalVideoParameterKind::signed_integer &&
                   p.signed_value >= 0 && p.signed_value <= 100) {
          parsed.auto_frame_smoothing = static_cast<int>(p.signed_value);
          bit = 2;
        } else if (!denoise && !eye &&
                   p.id == video::CanonicalVideoParameterId::headroom &&
                   p.kind == video::CanonicalVideoParameterKind::real &&
                   std::isfinite(p.real_value) && p.real_value >= 0 &&
                   p.real_value <= 1) {
          parsed.auto_frame_headroom = static_cast<float>(p.real_value);
          bit = 4;
        } else {
          SetReason(reason, ReasonId::malformed_contract);
          return false;
        }
        if (seen & bit) {
          SetReason(reason, ReasonId::malformed_contract);
          return false;
        }
        seen |= bit;
      }
      if (seen != (denoise ? 1u : (eye ? 3u : 7u))) {
        SetReason(reason, ReasonId::malformed_contract);
        return false;
      }
      if (denoise)
        parsed.request_denoise = true;
      else if (eye)
        parsed.request_eye = true;
      else
        parsed.request_auto_frame = true;
    } else if (IsMattingSemantic(stage.intent.semantic)) {
      if (saw_vignette || saw_mirror || options->matting.request_auto_frame ||
          IsZeroModel(stage.model)) {
        SetReason(reason, ReasonId::malformed_contract);
        return false;
      }
      const bool key =
          stage.intent.semantic == SemanticEffectId::virtual_key_light;
      if ((key && options->matting.request_keylight) ||
          (!key && (options->matting.background != SemanticEffectId::invalid ||
                    options->matting.request_keylight))) {
        SetReason(reason, ReasonId::malformed_contract);
        return false;
      }
      if (key)
        options->matting.request_keylight = true;
      else
        options->matting.background = stage.intent.semantic;
      unsigned seen = 0;
      for (std::size_t i = 0; i < stage.parameter_count; ++i) {
        const auto &p = stage.parameters[i];
        const auto signed_range = [&](std::int64_t lo, std::int64_t hi) {
          return p.kind == video::CanonicalVideoParameterKind::signed_integer &&
                 p.signed_value >= lo && p.signed_value <= hi;
        };
        unsigned bit = 0;
        if (p.id == video::CanonicalVideoParameterId::enabled)
          continue;
        if (!key && p.id == video::CanonicalVideoParameterId::strength &&
            signed_range(1, 64)) {
          options->matting.background_strength =
              static_cast<int>(p.signed_value);
          bit = 1;
        } else if (!key &&
                   p.id == video::CanonicalVideoParameterId::remove_color_rgb &&
                   p.kind ==
                       video::CanonicalVideoParameterKind::unsigned_integer &&
                   p.unsigned_value <= 0xffffff) {
          options->matting.remove_color_rgb =
              static_cast<std::uint32_t>(p.unsigned_value);
          bit = 2;
        } else if (!key &&
                   p.id == video::CanonicalVideoParameterId::greenscreen_mode &&
                   p.kind ==
                       video::CanonicalVideoParameterKind::unsigned_integer &&
                   p.unsigned_value <=
                       std::numeric_limits<std::uint32_t>::max()) {
          bit = 4;
        } else if (!key &&
                   p.id ==
                       video::CanonicalVideoParameterId::greenscreen_temporal &&
                   p.kind == video::CanonicalVideoParameterKind::boolean) {
          bit = 8;
        } else if (key && p.id == video::CanonicalVideoParameterId::intensity &&
                   signed_range(0, 100)) {
          options->matting.keylight_intensity =
              static_cast<int>(p.signed_value);
          bit = 1;
        } else if (key &&
                   p.id ==
                       video::CanonicalVideoParameterId::temperature_preset &&
                   signed_range(0, 2)) {
          options->matting.keylight_temperature =
              static_cast<int>(p.signed_value);
          bit = 2;
        } else if (key &&
                   p.id == video::CanonicalVideoParameterId::
                               direction_pan_degrees &&
                   signed_range(-180, 180)) {
          options->matting.keylight_pan = static_cast<int>(p.signed_value);
          bit = 4;
        } else {
          SetReason(reason, ReasonId::malformed_contract);
          return false;
        }
        if (seen & bit) {
          SetReason(reason, ReasonId::malformed_contract);
          return false;
        }
        seen |= bit;
      }
      if (seen != (key ? 7u : 15u)) {
        SetReason(reason, ReasonId::malformed_contract);
        return false;
      }
    } else if (stage.intent.semantic == SemanticEffectId::vignette) {
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
          if (parameter.boolean_value &&
              stage.intent.requires_tracked_vignette_center) {
            options->matting.vignette_center =
                VulkanVignetteCenterMode::tracked_face;
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
  if (options->matting.vignette_center ==
          VulkanVignetteCenterMode::tracked_face &&
      !options->matting.request_auto_frame) {
    SetReason(reason, ReasonId::semantic_effect_unsupported);
    return false;
  }
  SetReason(reason, ReasonId::none);
  return options->mirror || options->vignette ||
         options->matting.request_denoise || options->matting.request_eye ||
         options->matting.request_auto_frame ||
         options->matting.request_keylight ||
         options->matting.background != SemanticEffectId::invalid;
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
    const bool native_size = IsModelSemantic(stage.semantic) &&
                             prepared_index + 1 < island.stage_count;
    const bool previous_native =
        prepared_index > 0 &&
        IsModelSemantic(program.stages[prepared_index - 1].semantic);
    stage.input.image =
        (prepared_index == 0 || previous_native) ? first_input : chained_input;
    stage.output.kind = hal::StageIoKind::image;
    stage.output.image =
        native_size
            ? ResidentImage(request.host_input, hal::Mutability::read_write)
            : resident_output;
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
      CapabilityEvidenceCode::vulkan_physical_device_available,
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
  if (IsModelSemantic(semantic)) {
    capability.evidence[capability.evidence_count++] = {
        CapabilityEvidenceCode::vulkan_model_artifacts_valid,
        StatusId::ready,
        ReasonId::none,
        semantic,
        BackendId::vulkan,
        true};
    capability.evidence[capability.evidence_count++] = {
        CapabilityEvidenceCode::vulkan_warmup_complete,
        StatusId::ready,
        ReasonId::none,
        semantic,
        BackendId::vulkan,
        true};
  }
  capability.evidence[capability.evidence_count++] = {
      production_code, StatusId::ready,   ReasonId::none,
      semantic,        BackendId::vulkan, true};
  const auto add = [&](CapabilityEvidenceCode code) {
    capability.evidence[capability.evidence_count++] = {
        code,     StatusId::ready,   ReasonId::none,
        semantic, BackendId::vulkan, true};
  };
  if (semantic == SemanticEffectId::auto_frame)
    add(CapabilityEvidenceCode::vulkan_face_runtime_ready);
  if (semantic == SemanticEffectId::eye_contact) {
    add(CapabilityEvidenceCode::vulkan_face_runtime_ready);
    add(CapabilityEvidenceCode::vulkan_landmarks_runtime_ready);
    add(CapabilityEvidenceCode::vulkan_shader_float64_enabled);
  }
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
  StatusId
  DiscardUnsubmittedHostFrame(const hal::PreparedProgramHandle &program,
                              const hal::HostFrameTokens &tokens,
                              std::uint64_t sequence,
                              ReasonId *reason) noexcept override {
    return adapter_->DiscardUnsubmittedHostFrame(program, tokens, sequence,
                                                 reason);
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
    auto status = session_.Configure(plan_.prepared_program, adapter_,
                                           &readiness, &reason);
    if (evidence) {
      evidence->status = status;
      evidence->reason = reason;
      evidence->readiness = readiness;
      evidence->candidate_resources_rolled_back = !IsSetupSuccess(status);
    }
    readiness_ = readiness;
    if (IsSetupSuccess(status)) {
      readiness_.ready_stage_mask = 0;
      readiness_.disabled_stage_mask = 0;
      for (std::size_t i = 0; i < plan_.resolution.stage_count; ++i) {
        const auto bit = std::uint32_t{1} << i;
        if (plan_.resolution.stages[i].skipped_optional)
          readiness_.disabled_stage_mask |= bit;
        else
          readiness_.ready_stage_mask |= bit;
      }
      if (evidence)
        evidence->readiness = readiness_;
    }
    if (IsSetupSuccess(status) && plan_.resolution.degraded) {
      status = StatusId::degraded;
      readiness_.status = status;
      readiness_.reason = plan_.resolution.reason;
      readiness_.degraded = true;
      if (evidence) {
        evidence->status = status;
        evidence->reason = readiness_.reason;
        evidence->readiness = readiness_;
      }
    }
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

  StatusId ResetTemporal(ReasonId reason) noexcept override {
    if (stopped_)
      return StatusId::stopped;
    if (!prepared_)
      return quarantined_ ? StatusId::quarantined : StatusId::stale;
    const auto status = session_.ResetTemporalState(reason);
    if (status != StatusId::ready && status != StatusId::degraded) {
      prepared_ = false;
      pending_ = false;
      quarantined_ = status == StatusId::quarantined;
    }
    return status;
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
  SemanticEffectId failed_semantic = SemanticEffectId::invalid;

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
      VulkanResidentProgramOptions options = parsed.matting;
      options.program_id = program.program_id;
      options.canonical_plan_hash =
          program.invalidation_key.canonical_plan_hash;
      options.request_mirror = parsed.mirror;
      options.request_vignette = parsed.vignette;
      options.vignette_intensity_percent = parsed.vignette_intensity_percent;
      return production->adapter->SetProgramOptions(options);
    }
#else
    (void)program;
    (void)parsed;
#endif
    return injected;
  }

  void DiscardFailedProduction() noexcept {
#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
    if (!NativeQuiescenceProven()) {
      status = StatusId::quarantined;
      reason = ReasonId::synchronization_failed;
      // Destructor retains this complete failed candidate; no caller may
      // mistake missing session readiness for permission to release it.
      return;
    }
    production.reset();
#endif
  }

  std::uint64_t shutdown_attempts = 0;
  bool shutdown_complete = false;

  bool NativeQuiescenceProven() const noexcept {
#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
    if (production && production->kernels.device()->Initialized()) {
      const auto health = production->kernels.device()->health_state();
      return health.healthy() &&
             health.completed_serial >= health.submitted_serial &&
             production->kernels.device()->SafeToDestroyResources();
    }
#endif
    return true;
  }

  ~Impl() {
#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
    if (production && !NativeQuiescenceProven()) {
      // Completion is unproven. Retain the entire native island for process
      // lifetime; destroying an uncertain queue's resources is unsafe.
      (void)production.release();
      return;
    }
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

bool CollectProductionVulkanModelDiagnostics(
    const VulkanVideoExecutionFactory &factory,
    VulkanProductionModelDiagnostics *out, std::string *error) {
  if (error)
    error->clear();
  if (!out)
    return false;
  *out = {};
#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
  const auto &p = *factory.impl_;
  if (p.initialized && p.production && p.production->executor) {
    out->device = p.production->kernels.Diagnostics();
    out->allocations = p.production->kernels.device()->allocation_stats();
    p.production->executor->CollectModelDiagnostics(out);
    out->matting_pack = p.configuration.matting_pack;
    out->denoise_pack = p.configuration.denoise_pack;
    out->face_pack = p.configuration.face_pack;
    out->landmarks_pack = p.configuration.landmarks_pack;
    out->eye_pack = p.configuration.eye_pack;
    return true;
  }
#else
  (void)factory;
#endif
  if (error)
    *error = "No initialized production Vulkan diagnostics owner.";
  return false;
}

StatusId VulkanVideoExecutionFactory::Initialize(ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (impl_->shutdown_attempts != 0) {
    SetReason(reason, ReasonId::shutdown_requested);
    return StatusId::stopped;
  }
  if (impl_->initialized) {
    SetReason(reason, impl_->reason);
    return impl_->status;
  }
  if (impl_->configuration.factory_id == 0 ||
      impl_->configuration.execution_fence_timeout_ns == 0 ||
      impl_->configuration.execution_fence_timeout_ns == UINT64_MAX) {
    impl_->status = StatusId::blocked;
    impl_->reason = ReasonId::malformed_contract;
  } else if (impl_->injected) {
    impl_->initialized = ExactRegistration(impl_->registration, &impl_->reason);
    impl_->status = impl_->initialized ? StatusId::ready : StatusId::blocked;
  } else {
#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
    const bool expanded_models = impl_->configuration.denoise_pack ||
                                 impl_->configuration.face_pack ||
                                 impl_->configuration.eye_pack;
    const auto block_eye = [&]() {
      impl_->configuration.unavailable_semantic_mask |=
          SemanticBit(SemanticEffectId::eye_contact);
      if (impl_->configuration.effects) {
        auto effects = *impl_->configuration.effects;
        effects.eye_contact.enabled = false;
        impl_->configuration.effects =
            std::make_shared<const video::effects::BroadcastCameraEffects>(
                effects);
        if (!effects.auto_frame.enabled) {
          impl_->configuration.face_pack.reset();
          impl_->configuration.face_identity = {};
        }
      }
      impl_->configuration.eye_pack.reset();
      impl_->configuration.eye_identity = {};
      impl_->configuration.landmarks_pack.reset();
      impl_->configuration.landmarks_identity = {};
    };
#if !defined(STUDIOCAST_HAVE_DLIB) || !STUDIOCAST_HAVE_DLIB
    if (impl_->configuration.eye_pack)
      block_eye();
#endif
    if (impl_->configuration.matting_pack &&
        (impl_->configuration.input_width <= 0 ||
         impl_->configuration.input_height <= 0 ||
         impl_->configuration.reservation.persistent_bytes <
             UINT64_C(256) * 1024 * 1024 ||
         impl_->configuration.reservation.resource_count < 544 ||
         impl_->configuration.reservation.in_flight_slots != 1)) {
      impl_->status = StatusId::resource_exhausted;
      impl_->reason = ReasonId::resource_exhausted;
      SetReason(reason, impl_->reason);
      return impl_->status;
    }
    impl_->production = std::make_unique<Impl::Production>();
    std::string error;
    VulkanDeviceConfig device_config;
    device_config.fence_timeout_ns =
        impl_->configuration.execution_fence_timeout_ns;
    if (expanded_models) {
      const auto &reservation = impl_->configuration.reservation;
      device_config.allocation_budget_bytes = reservation.persistent_bytes;
      for (const auto bytes :
           {reservation.scratch_bytes, reservation.upload_staging_bytes,
            reservation.readback_staging_bytes}) {
        if (bytes > UINT64_MAX - device_config.allocation_budget_bytes) {
          impl_->status = StatusId::resource_exhausted;
          impl_->reason = ReasonId::resource_exhausted;
          SetReason(reason, impl_->reason);
          return impl_->status;
        }
        device_config.allocation_budget_bytes += bytes;
      }
      device_config.allocation_resource_limit = reservation.resource_count;
    }
    VulkanDeviceSelection selection;
    const bool selected = VulkanDevice::ResolveSelection(&selection, &error);
    bool device_ready =
        selected && ((impl_->configuration.matting_pack || expanded_models)
                         ? open_vulkan::InitializeProductionVulkanMattingDevice(
                               impl_->production->kernels.device(), selection,
                               device_config, &error,
                               impl_->configuration.eye_pack != nullptr)
                         : impl_->production->kernels.device()->Initialize(
                               selection, device_config, &error));
    if (!device_ready && impl_->configuration.eye_pack &&
        error.rfind("[vulkan_shader_float64_unavailable]", 0) == 0) {
      // The adopted-context helper checks the selected physical feature before
      // constructing its logical device. Isolate only the affected effect.
      block_eye();
      device_ready = open_vulkan::InitializeProductionVulkanMattingDevice(
          impl_->production->kernels.device(), selection, device_config, &error,
          false);
    }
    if (!device_ready || !impl_->production->kernels.Initialize(&error)) {
      impl_->status = StatusId::unavailable;
      impl_->reason = ReasonId::capability_evidence_incomplete;
      impl_->DiscardFailedProduction();
    } else {
      static std::atomic<std::uint64_t> next_adapter_id{
          UINT64_C(0x56554c4b00000000)};
      auto adapter_id = next_adapter_id.fetch_add(1, std::memory_order_relaxed);
      if (adapter_id == 0)
        adapter_id = next_adapter_id.fetch_add(1, std::memory_order_relaxed);
      impl_->production->executor =
          std::make_unique<QualifiedFinalResidentExecutor>(
              adapter_id, impl_->production->kernels);
      if (impl_->configuration.matting_pack &&
          (!impl_->configuration.effects ||
           !impl_->production->executor->InitializeMatting(
               *impl_->configuration.matting_pack,
               *impl_->configuration.effects, impl_->configuration.input_width,
               impl_->configuration.input_height, &error))) {
        impl_->reason = ReasonId::capability_evidence_incomplete;
        impl_->failed_semantic = SemanticEffectId::virtual_background_blur;
        impl_->status = StatusId::unavailable;
        impl_->DiscardFailedProduction();
        SetReason(reason, impl_->reason);
        return impl_->status;
      }
      if (expanded_models && !impl_->production->executor->InitializeModels(
                                 impl_->configuration, &error)) {
        impl_->reason = ReasonId::capability_evidence_incomplete;
        impl_->failed_semantic =
            impl_->production->executor->InitializationFailedSemantic();
        impl_->status = StatusId::unavailable;
        impl_->DiscardFailedProduction();
        SetReason(reason, impl_->reason);
        return impl_->status;
      }
      const auto evidence = impl_->production->executor->CapabilityEvidence();
      if (evidence.blocker != ReasonId::none ||
          !evidence.mirror_production_ready ||
          !evidence.vignette_fixed_center_production_ready) {
        impl_->status = StatusId::unavailable;
        impl_->reason = evidence.blocker == ReasonId::none
                            ? ReasonId::capability_evidence_incomplete
                            : evidence.blocker;
        impl_->DiscardFailedProduction();
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
        if (evidence.vignette_tracked_center_ready) {
          auto &vignette = impl_->registration.semantic_capabilities[0];
          vignette.supports_tracked_vignette_center = true;
          for (const auto code :
               {CapabilityEvidenceCode::
                    vulkan_tracked_vignette_center_supported,
                CapabilityEvidenceCode::vulkan_model_artifacts_valid,
                CapabilityEvidenceCode::vulkan_face_runtime_ready,
                CapabilityEvidenceCode::vulkan_tracking_gpu_ready})
            vignette.evidence[vignette.evidence_count++] = {
                code,
                StatusId::ready,
                ReasonId::none,
                SemanticEffectId::vignette,
                BackendId::vulkan,
                true};
        }
        for (const auto semantic :
             {SemanticEffectId::video_noise_removal,
              SemanticEffectId::eye_contact, SemanticEffectId::auto_frame}) {
          const bool ready = semantic == SemanticEffectId::video_noise_removal
                                 ? evidence.denoise_production_ready
                                 : (semantic == SemanticEffectId::eye_contact
                                        ? evidence.eye_production_ready
                                        : evidence.face_production_ready &&
                                              evidence.tracking_gpu_ready);
          if (ready)
            impl_->registration.semantic_capabilities
                [impl_->registration.semantic_capability_count++] =
                ProductionCapability(semantic, ModelRuntimeCode(semantic));
        }
        if (evidence.matting_production_ready) {
          for (const auto semantic :
               {SemanticEffectId::virtual_background_blur,
                SemanticEffectId::virtual_background_remove,
                SemanticEffectId::virtual_background_replace,
                SemanticEffectId::virtual_key_light}) {
            impl_->registration.semantic_capabilities
                [impl_->registration.semantic_capability_count++] =
                ProductionCapability(
                    semantic,
                    CapabilityEvidenceCode::vulkan_matting_runtime_ready);
          }
        }
        for (const auto semantic :
             {SemanticEffectId::mirror, SemanticEffectId::vignette,
              SemanticEffectId::video_noise_removal,
              SemanticEffectId::eye_contact, SemanticEffectId::auto_frame,
              SemanticEffectId::virtual_background_blur,
              SemanticEffectId::virtual_background_remove,
              SemanticEffectId::virtual_background_replace,
              SemanticEffectId::virtual_key_light}) {
          if (!(impl_->configuration.unavailable_semantic_mask &
                SemanticBit(semantic)))
            continue;
          std::size_t index = 0;
          while (index < impl_->registration.semantic_capability_count &&
                 impl_->registration.semantic_capabilities[index].semantic !=
                     semantic)
            ++index;
          if (index == impl_->registration.semantic_capability_count)
            ++impl_->registration.semantic_capability_count;
          auto &capability = impl_->registration.semantic_capabilities[index];
          capability = {};
          capability.semantic = semantic;
          capability.provider = kProvider;
          capability.status = StatusId::unavailable;
          capability.reason = ReasonId::capability_evidence_incomplete;
        }
        impl_->initialized =
            ExactRegistration(impl_->registration, &impl_->reason);
        impl_->status =
            impl_->initialized ? StatusId::ready : StatusId::blocked;
        if (impl_->initialized &&
            impl_->configuration.unavailable_semantic_mask) {
          impl_->status = StatusId::degraded;
          impl_->reason = ReasonId::capability_evidence_incomplete;
        }
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

video::ProductionVideoSetupLeaseShutdownResult
VulkanVideoExecutionFactory::RequestShutdown() noexcept {
  video::ProductionVideoSetupLeaseShutdownResult result;
  result.shutdown_requested = true;
  result.shutdown_attempts = ++impl_->shutdown_attempts;
  auto proof = StatusId::ready;
  if (!impl_->shutdown_complete) {
    if (impl_->injected && impl_->configuration.injected_shutdown_probe)
      proof = impl_->configuration.injected_shutdown_probe(
          impl_->configuration.injected_shutdown_context);
    else if (!impl_->NativeQuiescenceProven()) {
      proof = StatusId::quarantined;
    }
  }
  if (proof != StatusId::ready) {
    result.status = StatusId::quarantined;
    result.reason = ReasonId::synchronization_failed;
#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
    if (impl_->production &&
        impl_->production->kernels.device()->health_state().health ==
            VulkanContextHealth::device_lost)
      result.reason = ReasonId::device_lost;
#endif
    result.pending = true;
    return result;
  }
#if defined(STUDIOCAST_ENABLE_OPEN_VULKAN) && STUDIOCAST_ENABLE_OPEN_VULKAN
  if (impl_->production) {
    if (impl_->production->adapter)
      impl_->production->adapter->Shutdown();
    if (impl_->production->executor)
      impl_->production->executor->Shutdown();
  }
#endif
  impl_->shutdown_complete = true;
  result.status = StatusId::stopped;
  result.reason = ReasonId::shutdown_requested;
  result.quiescent = true;
  result.driver_context_release_permitted = true;
  return result;
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
SemanticEffectId
VulkanVideoExecutionFactory::initialization_failed_semantic() const noexcept {
  return impl_->failed_semantic;
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
  preflight.per_frame_transfers = video::BuildPerFrameTransferContract(
      frozen_domain_plan,
      impl_->configuration.minimum_synchronizations_per_frame,
      impl_->configuration.maximum_synchronizations_per_frame);

  const auto fail = [&](StatusId status, ReasonId reason) {
    preflight.status = status;
    preflight.reason = reason;
    preflight.readiness.status = status;
    preflight.readiness.reason = reason;
    preflight.readiness.available = false;
    return preflight;
  };
  if (impl_->shutdown_attempts != 0)
    return fail(StatusId::stopped, ReasonId::shutdown_requested);
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
      request.temporal.temporal != TemporalIntent(request.intent) ||
      (!request.temporal.temporal && request.temporal.history_frames != 0) ||
      request.cpu_tail_count != 0 ||
      frozen_domain_plan.expected_transfers.uploads != 1 ||
      frozen_domain_plan.expected_transfers.final_readbacks != 1 ||
      frozen_domain_plan.expected_transfers.intermediate_readbacks != 0 ||
      frozen_domain_plan.expected_transfers.intermediate_uploads != 0 ||
      frozen_domain_plan.expected_transfers.cpu_tail_boundaries != 0) {
    return fail(StatusId::blocked, ReasonId::malformed_contract);
  }
  ReasonId transfer_reason = ReasonId::none;
  if (!video::ValidatePerFrameTransferContract(preflight.per_frame_transfers,
                                               frozen_domain_plan,
                                               &transfer_reason)) {
    return fail(StatusId::blocked, transfer_reason);
  }

  if (frozen_resolution.stage_count != request.intent.stage_count)
    return fail(StatusId::blocked, ReasonId::semantic_effect_unsupported);
  std::size_t selected_count = 0;
  for (std::size_t i = 0; i < frozen_resolution.stage_count; ++i) {
    const auto &stage = frozen_resolution.stages[i];
    if (stage.intent.stage_id != request.intent.stages[i].intent.stage_id ||
        stage.intent.semantic != request.intent.stages[i].intent.semantic)
      return fail(StatusId::blocked, ReasonId::malformed_contract);
    if (stage.skipped_optional) {
      const auto *capability =
          FindCapability(impl_->registration, stage.intent.semantic);
      if (stage.intent.required || !capability ||
          !UnavailableCapability(*capability))
        return fail(StatusId::blocked,
                    ReasonId::capability_evidence_incomplete);
      continue;
    }
    ++selected_count;
    if (stage.selected_backend != BackendId::vulkan ||
        stage.provider != kProvider ||
        stage.execution_domain != impl_->registration.execution_domain ||
        (stage.intent.semantic != SemanticEffectId::mirror &&
         stage.intent.semantic != SemanticEffectId::vignette &&
         !IsModelSemantic(stage.intent.semantic)))
      return fail(StatusId::blocked, ReasonId::semantic_effect_unsupported);
    const auto *identity =
        ModelIdentity(impl_->configuration, stage.intent.semantic);
    const auto &model = request.intent.stages[i].model;
    if (identity &&
        (!identity->artifact_generation ||
         model.model_id_hash != identity->model_id_hash ||
         model.artifact_set_hash != identity->artifact_set_hash ||
         model.artifact_generation != identity->artifact_generation))
      return fail(StatusId::blocked, ReasonId::reconfiguration_required);
  }
  if (!selected_count ||
      frozen_domain_plan.islands[0].stage_count != selected_count)
    return fail(StatusId::blocked, ReasonId::semantic_effect_unsupported);
  ParsedProgramOptions parsed;
  ReasonId parsed_reason = ReasonId::none;
  const auto active_intent = ActiveIntent(request.intent, frozen_resolution);
  if (!ParseOptions(active_intent, &parsed, &parsed_reason))
    return fail(StatusId::blocked, parsed_reason);

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
  preflight.status = frozen_resolution.degraded ? StatusId::degraded : StatusId::ready;
  preflight.reason = frozen_resolution.degraded ? frozen_resolution.reason : ReasonId::none;
  preflight.readiness.status = preflight.status;
  preflight.readiness.reason = preflight.reason;
  preflight.readiness.degraded = frozen_resolution.degraded;
  preflight.readiness.available = true;
  for (std::size_t i = 0; i < frozen_resolution.stage_count; ++i) {
    const auto bit = std::uint32_t{1} << i;
    if (frozen_resolution.stages[i].skipped_optional)
      preflight.readiness.disabled_stage_mask |= bit;
    else
      preflight.readiness.ready_stage_mask |= bit;
  }
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
  if (impl_->shutdown_attempts != 0)
    return fail(StatusId::stopped, ReasonId::shutdown_requested);
  if (!impl_->initialized)
    return fail(impl_->status, impl_->reason);
  if (plan.route != video::VideoExecutionRoute::prepared_resident ||
      plan.factory_id != impl_->configuration.factory_id || plan.plan_id == 0 ||
      plan.domain_plan.island_count != 1 ||
      plan.domain_plan.islands[0].execution_domain !=
          impl_->registration.execution_domain ||
      !video::ValidatePerFrameTransferContract(plan.per_frame_transfers,
                                               plan.domain_plan, nullptr) ||
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
  if (plan.prepared_program.maximum_metadata_readbacks_per_frame != 0 ||
      plan.prepared_program.maximum_metadata_cpu_tails_per_frame != 0) {
    return fail(StatusId::blocked, ReasonId::transfer_contract_violated);
  }
  for (std::size_t i = 0; i < plan.prepared_program.stage_count; ++i) {
    const auto &stage = plan.prepared_program.stages[i];
    const auto *identity = ModelIdentity(impl_->configuration, stage.semantic);
    if (identity &&
        (!identity->artifact_generation ||
         stage.model.model_id_hash != identity->model_id_hash ||
         stage.model.artifact_set_hash != identity->artifact_set_hash ||
         stage.model.artifact_generation != identity->artifact_generation))
      return fail(StatusId::blocked, ReasonId::reconfiguration_required);
  }
  ParsedProgramOptions parsed;
  const auto active_intent = ActiveIntent(plan.intent, plan.resolution);
  if (!ParseOptions(active_intent, &parsed, &reason))
    return fail(StatusId::blocked, reason);
  parsed.matting.conservative_temporal_contract =
      plan.prepared_program.temporal.temporal && !TemporalIntent(active_intent);
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
