#include "core/video/production_video_setup_request.h"

#include "core/video/effects/broadcast_effect_rules.h"

#include <cstring>
#include <string_view>

namespace studiocast::video {

void ApplyCameraBackendSelectionIntent(
    effects::EffectsEnginePreference engine,
    ComputeBackendPreference compute_backend,
    NormalizedVideoExecutionIntent *intent) noexcept {
  if (intent == nullptr)
    return;
  switch (engine) {
  case effects::EffectsEnginePreference::maxine:
    intent->backend.mode = hal::BackendRequestMode::explicit_backend;
    intent->backend.backend = hal::BackendId::maxine;
    intent->backend.strict = true;
    intent->preferences.compute_family =
        hal::ComputeFamilyAllowance::cuda_family;
    intent->preferences.effects_engine =
        hal::EffectsEnginePreferenceConstraint::maxine;
    intent->preferences.strict_effects_engine = true;
    break;
  case effects::EffectsEnginePreference::open_cuda:
    intent->backend.mode = hal::BackendRequestMode::explicit_backend;
    intent->backend.backend = hal::BackendId::cuda;
    intent->backend.strict = true;
    intent->preferences.compute_family =
        hal::ComputeFamilyAllowance::cuda_family;
    intent->preferences.effects_engine =
        hal::EffectsEnginePreferenceConstraint::open_cuda;
    intent->preferences.strict_effects_engine = true;
    break;
  case effects::EffectsEnginePreference::auto_select:
    intent->backend.mode = hal::BackendRequestMode::automatic;
    intent->preferences.compute_family = hal::ComputeFamilyAllowance::any;
    intent->preferences.effects_engine =
        hal::EffectsEnginePreferenceConstraint::automatic;
    intent->preferences.strict_effects_engine = false;
    break;
  }
  if (compute_backend == ComputeBackendPreference::vulkan) {
    intent->backend.mode = hal::BackendRequestMode::explicit_backend;
    intent->backend.backend = hal::BackendId::vulkan;
    intent->backend.strict = true;
    intent->preferences.compute_family =
        hal::ComputeFamilyAllowance::vulkan_only;
    if (intent->requested_provider == hal::SemanticEffectProviderId::invalid &&
        !intent->strict_provider_selection) {
      intent->requested_provider =
          hal::SemanticEffectProviderId::open_vulkan_production_kernels;
      intent->strict_provider_selection = true;
    }
  } else if (compute_backend == ComputeBackendPreference::cuda &&
             intent->backend.mode == hal::BackendRequestMode::automatic) {
    intent->backend.mode = hal::BackendRequestMode::explicit_backend;
    intent->backend.backend = hal::BackendId::cuda;
    intent->backend.strict = true;
    intent->preferences.compute_family =
        hal::ComputeFamilyAllowance::cuda_family;
  } else if (compute_backend == ComputeBackendPreference::cpu) {
    intent->backend.mode = hal::BackendRequestMode::explicit_backend;
    intent->backend.backend = hal::BackendId::cpu;
    intent->backend.strict = true;
    intent->preferences.compute_family = hal::ComputeFamilyAllowance::cpu_only;
  }
}

namespace {

constexpr std::uint64_t kHashOffset = UINT64_C(1469598103934665603);
constexpr std::uint64_t kHashPrime = UINT64_C(1099511628211);

std::uint64_t MixHash(std::uint64_t hash, std::uint64_t value) noexcept {
  hash ^= value;
  hash *= kHashPrime;
  return hash;
}

std::uint64_t MixText(std::uint64_t hash, std::string_view value) noexcept {
  for (const char character : value) {
    hash = MixHash(hash, static_cast<unsigned char>(character));
  }
  return hash;
}

bool ResidentMirrorRequested(const effects::BroadcastCameraEffects &effects,
                             ComputeBackendPreference compute_backend) {
  return compute_backend == ComputeBackendPreference::vulkan ||
         (compute_backend != ComputeBackendPreference::cpu &&
          effects.engine != effects::EffectsEnginePreference::maxine &&
          (compute_backend == ComputeBackendPreference::cuda ||
           effects.engine == effects::EffectsEnginePreference::open_cuda));
}

bool HasResidentSemanticEffects(const effects::BroadcastCameraEffects &effects,
                                ComputeBackendPreference compute_backend) {
  return effects.video_noise_removal.enabled || effects.eye_contact.enabled ||
         effects.virtual_background.mode !=
             effects::VirtualBackgroundMode::none ||
         effects.virtual_key_light.enabled || effects.auto_frame.enabled ||
         effects.vignette.enabled ||
         (effects.mirror && ResidentMirrorRequested(effects, compute_backend));
}

hal::SemanticEffectId SemanticForEffectId(std::string_view id) {
  using namespace effects::contract;
  if (id == kEffectIdVideoNoiseRemoval)
    return hal::SemanticEffectId::video_noise_removal;
  if (id == kEffectIdEyeContact)
    return hal::SemanticEffectId::eye_contact;
  if (id == kEffectIdVirtualBackgroundBlur)
    return hal::SemanticEffectId::virtual_background_blur;
  if (id == kEffectIdVirtualBackgroundRemove)
    return hal::SemanticEffectId::virtual_background_remove;
  if (id == kEffectIdVirtualBackgroundReplace)
    return hal::SemanticEffectId::virtual_background_replace;
  if (id == kEffectIdVirtualKeyLight)
    return hal::SemanticEffectId::virtual_key_light;
  if (id == kEffectIdAutoFrame)
    return hal::SemanticEffectId::auto_frame;
  if (id == kEffectIdVignette)
    return hal::SemanticEffectId::vignette;
  if (id == kEffectIdMirror)
    return hal::SemanticEffectId::mirror;
  return hal::SemanticEffectId::invalid;
}

bool ParseRgb(std::string_view value, std::uint64_t *output) {
  if (output == nullptr || value.size() != 7 || value.front() != '#')
    return false;
  std::uint64_t result = 0;
  for (std::size_t index = 1; index < value.size(); ++index) {
    const char character = value[index];
    std::uint64_t nibble = 0;
    if (character >= '0' && character <= '9')
      nibble = static_cast<std::uint64_t>(character - '0');
    else if (character >= 'a' && character <= 'f')
      nibble = static_cast<std::uint64_t>(character - 'a' + 10);
    else if (character >= 'A' && character <= 'F')
      nibble = static_cast<std::uint64_t>(character - 'A' + 10);
    else
      return false;
    result = (result << 4u) | nibble;
  }
  *output = result;
  return true;
}

void AddSigned(CanonicalVideoStage *stage, CanonicalVideoParameterId id,
               std::int64_t value) {
  auto &parameter = stage->parameters[stage->parameter_count++];
  parameter.id = id;
  parameter.kind = CanonicalVideoParameterKind::signed_integer;
  parameter.signed_value = value;
}

void AddUnsigned(CanonicalVideoStage *stage, CanonicalVideoParameterId id,
                 std::uint64_t value) {
  auto &parameter = stage->parameters[stage->parameter_count++];
  parameter.id = id;
  parameter.kind = CanonicalVideoParameterKind::unsigned_integer;
  parameter.unsigned_value = value;
}

void AddBoolean(CanonicalVideoStage *stage, CanonicalVideoParameterId id,
                bool value) {
  auto &parameter = stage->parameters[stage->parameter_count++];
  parameter.id = id;
  parameter.kind = CanonicalVideoParameterKind::boolean;
  parameter.boolean_value = value;
}

void AddReal(CanonicalVideoStage *stage, CanonicalVideoParameterId id,
             double value) {
  auto &parameter = stage->parameters[stage->parameter_count++];
  parameter.id = id;
  parameter.kind = CanonicalVideoParameterKind::real;
  parameter.real_value = value;
}

bool PopulateStageParameters(const effects::BroadcastCameraEffects &effects,
                             CanonicalVideoStage *stage) {
  std::uint64_t color = 0;
  switch (stage->intent.semantic) {
  case hal::SemanticEffectId::virtual_background_blur:
  case hal::SemanticEffectId::virtual_background_remove:
  case hal::SemanticEffectId::virtual_background_replace:
    if (!ParseRgb(effects.virtual_background.remove_color, &color))
      return false;
    AddSigned(stage, CanonicalVideoParameterId::strength,
              effects.virtual_background.strength);
    AddUnsigned(stage, CanonicalVideoParameterId::remove_color_rgb, color);
    AddUnsigned(stage, CanonicalVideoParameterId::greenscreen_mode,
                effects.virtual_background.greenscreen_mode);
    AddBoolean(stage, CanonicalVideoParameterId::greenscreen_temporal,
               effects.virtual_background.greenscreen_temporal);
    return true;
  case hal::SemanticEffectId::auto_frame:
    AddSigned(stage, CanonicalVideoParameterId::strength,
              effects.auto_frame.strength);
    AddSigned(stage, CanonicalVideoParameterId::smoothing,
              effects.auto_frame.smoothing);
    AddReal(stage, CanonicalVideoParameterId::headroom,
            effects.auto_frame.headroom);
    return true;
  case hal::SemanticEffectId::eye_contact:
    AddSigned(stage, CanonicalVideoParameterId::strength,
              effects.eye_contact.strength);
    AddBoolean(stage, CanonicalVideoParameterId::look_away_enabled,
               effects.eye_contact.look_away_enabled);
    return true;
  case hal::SemanticEffectId::video_noise_removal:
    AddSigned(stage, CanonicalVideoParameterId::strength,
              effects.video_noise_removal.strength);
    return true;
  case hal::SemanticEffectId::virtual_key_light:
    AddSigned(stage, CanonicalVideoParameterId::intensity,
              effects.virtual_key_light.intensity);
    AddSigned(stage, CanonicalVideoParameterId::temperature_preset,
              effects.virtual_key_light.temperature_preset);
    AddSigned(stage, CanonicalVideoParameterId::direction_pan_degrees,
              effects.virtual_key_light.direction_pan_degrees);
    return true;
  case hal::SemanticEffectId::vignette:
    AddSigned(stage, CanonicalVideoParameterId::intensity,
              effects.vignette.intensity);
    AddBoolean(stage, CanonicalVideoParameterId::center_on_tracked_face,
               effects.vignette.center_on_tracked_face);
    return true;
  case hal::SemanticEffectId::mirror:
    return true;
  case hal::SemanticEffectId::invalid:
    return false;
  }
  return false;
}

hal::ImageDescriptor HostRgbImage(int width, int height,
                                  hal::Mutability mutability) {
  hal::ImageDescriptor descriptor;
  if (width <= 0 || height <= 0)
    return descriptor;
  descriptor.width = static_cast<std::uint32_t>(width);
  descriptor.height = static_cast<std::uint32_t>(height);
  descriptor.plane_count = 1;
  descriptor.format = hal::PixelFormat::rgb_u8;
  descriptor.channel_order = hal::ChannelOrder::rgb;
  descriptor.color_space = hal::ColorSpace::srgb;
  descriptor.color_range = hal::ColorRange::full;
  descriptor.alpha_meaning = hal::AlphaMeaning::none;
  descriptor.orientation = hal::Orientation::top_left;
  descriptor.host_coherence = hal::HostCoherence::coherent;
  descriptor.mutability = mutability;
  descriptor.aliasing = hal::AliasingPermission::forbidden;
  descriptor.in_place_permitted = false;
  descriptor.base_alignment_bytes = 1;
  const auto stride = static_cast<std::uint64_t>(width) * 3u;
  descriptor.planes[0].row_stride_bytes = stride;
  descriptor.planes[0].size_bytes = stride * static_cast<std::uint64_t>(height);
  descriptor.planes[0].width = static_cast<std::uint32_t>(width);
  descriptor.planes[0].height = static_cast<std::uint32_t>(height);
  descriptor.planes[0].alignment_bytes = 1;
  return descriptor;
}

} // namespace

ProductionVideoBackendSetupRequest BuildProductionVideoBackendSetupRequest(
    const effects::BroadcastCameraEffects &effects,
    ComputeBackendPreference compute_backend, int input_width, int input_height,
    int output_width, int output_height, PixelFormat output_format,
    std::uint64_t configuration_generation, std::uint64_t geometry_generation,
    std::uint64_t plan_id, const ProductionVideoBoundedWaitPolicy &wait,
    const CpuPrimitiveTable &cpu_table) {
  ProductionVideoBackendSetupRequest request;
  const bool cpu_mirror =
      effects.mirror && !ResidentMirrorRequested(effects, compute_backend);
  if (!HasResidentSemanticEffects(effects, compute_backend)) {
    request.kind = ProductionVideoBackendSetupKind::cpu_primitives_only;
    request.cpu.table = &cpu_table;
    request.cpu.required_primitive_bits = CpuPrimitiveCapabilityBit(
        hal::PrimitiveOperationId::convert_pixel_format);
    if (input_width != output_width || input_height != output_height) {
      request.cpu.required_primitive_bits |=
          CpuPrimitiveCapabilityBit(hal::PrimitiveOperationId::resize);
    }
    if (cpu_mirror) {
      request.cpu.required_primitive_bits |=
          CpuPrimitiveCapabilityBit(hal::PrimitiveOperationId::mirror);
    }
    return request;
  }

  request.kind = ProductionVideoBackendSetupKind::resident_island;
  request.resident.effects = effects;
  // Exact CUDA/Vulkan intent keeps mirror in the resident island. Automatic
  // selection and Maxine retain the established measured host finalizer.
  // This primitive tail is never a CPU semantic-provider capability.
  if (cpu_mirror)
    request.resident.effects.mirror = false;
  auto &plan = request.resident.plan;
  plan.plan_id = plan_id == 0 ? 1 : plan_id;

  ApplyCameraBackendSelectionIntent(effects.engine, compute_backend,
                                    &plan.intent);

  const auto canonical =
      effects::BuildBroadcastEffectsPlan(request.resident.effects);
  for (const auto &id : canonical.ordered_effect_ids) {
    if (plan.intent.stage_count >= plan.intent.stages.size())
      break;
    const auto semantic = SemanticForEffectId(id);
    if (semantic == hal::SemanticEffectId::invalid)
      continue;
    auto &stage = plan.intent.stages[plan.intent.stage_count];
    stage.intent.stage_id =
        static_cast<std::uint32_t>(plan.intent.stage_count + 1);
    stage.intent.semantic = semantic;
    stage.intent.requires_tracked_vignette_center =
        semantic == hal::SemanticEffectId::vignette &&
        effects.vignette.center_on_tracked_face && effects.auto_frame.enabled;
    stage.intent.required = false;
    stage.intent.failure_action = hal::FailureAction::fail_open_preserve_input;
    if (!PopulateStageParameters(effects, &stage))
      continue;
    ++plan.intent.stage_count;
  }

  plan.host_input =
      HostRgbImage(input_width, input_height, hal::Mutability::read_only);
  plan.host_output =
      HostRgbImage(output_width, output_height, hal::Mutability::write_only);
  plan.invalidation_key.generations.configuration =
      configuration_generation == 0 ? 1 : configuration_generation;
  plan.invalidation_key.generations.geometry =
      geometry_generation == 0 ? 1 : geometry_generation;

  std::uint64_t hash = kHashOffset;
  hash = MixHash(hash, static_cast<std::uint64_t>(effects.engine));
  hash = MixHash(hash, static_cast<std::uint64_t>(compute_backend));
  hash = MixHash(hash, static_cast<std::uint64_t>(input_width));
  hash = MixHash(hash, static_cast<std::uint64_t>(input_height));
  hash = MixHash(hash, static_cast<std::uint64_t>(output_width));
  hash = MixHash(hash, static_cast<std::uint64_t>(output_height));
  hash = MixHash(hash, static_cast<std::uint64_t>(output_format));
  for (std::size_t index = 0; index < plan.intent.stage_count; ++index) {
    const auto &stage = plan.intent.stages[index];
    hash = MixHash(hash, stage.intent.stage_id);
    hash = MixHash(hash, static_cast<std::uint64_t>(stage.intent.semantic));
    for (std::size_t parameter = 0; parameter < stage.parameter_count;
         ++parameter) {
      const auto &value = stage.parameters[parameter];
      hash = MixHash(hash, static_cast<std::uint64_t>(value.id));
      hash = MixHash(hash, static_cast<std::uint64_t>(value.kind));
      hash = MixHash(hash, static_cast<std::uint64_t>(value.signed_value));
      hash = MixHash(hash, value.unsigned_value);
      std::uint64_t real_bits = 0;
      static_assert(sizeof(real_bits) == sizeof(value.real_value));
      std::memcpy(&real_bits, &value.real_value, sizeof(real_bits));
      hash = MixHash(hash, real_bits);
      hash = MixHash(hash, value.boolean_value ? 1 : 0);
    }
  }
  hash = MixText(hash, effects.virtual_background.model_id);
  hash = MixText(hash, effects.auto_frame.model_id);
  hash = MixText(hash, effects.eye_contact.model_id);
  hash = MixText(hash, effects.video_noise_removal.model_id);
  hash = MixText(hash, effects.virtual_background.replace_path);
  hash = MixText(hash, effects.virtual_key_light.hdri_path);
  plan.invalidation_key.canonical_plan_hash = hash == 0 ? 1 : hash;

  constexpr std::uint64_t kGiB = UINT64_C(1024) * 1024u * 1024u;
  plan.budget.persistent_bytes = 8u * kGiB;
  plan.budget.scratch_bytes = 4u * kGiB;
  plan.budget.upload_staging_bytes = kGiB;
  plan.budget.readback_staging_bytes = kGiB;
  plan.budget.resource_count = 1024;
  plan.budget.in_flight_slots = 1;
  // The greenscreen temporal knob belongs to Maxine. Its selected source
  // adds that requirement after resolution; CUDA never inherits it.
  const bool temporal = effects.video_noise_removal.enabled ||
                        effects.eye_contact.enabled ||
                        effects.auto_frame.enabled;
  plan.temporal.temporal = temporal;
  plan.temporal.requires_contiguous_capture_sequence = temporal;
  plan.temporal.history_frames =
      effects.video_noise_removal.enabled ? 5u : (temporal ? 1u : 0u);
  plan.temporal.frame_drop_behavior =
      temporal ? hal::FrameDropBehavior::reset_temporal_state
               : hal::FrameDropBehavior::latest_frame_wins;
  plan.temporal.discontinuity_action =
      hal::TemporalDiscontinuityAction::reset_and_continue;
  if (plan.intent.stage_count == 0)
    return request;
  plan.cpu_tails[0].id = hal::CpuTailId::backend_resolved_output;
  plan.cpu_tails[0].after_stage_id =
      plan.intent.stages[plan.intent.stage_count - 1].intent.stage_id;
  plan.cpu_tails[0].required = true;
  plan.cpu_tails[0].placement = hal::DeclaredCpuTail::Placement::after_stage;
  plan.cpu_tails[0].boundary_index =
      static_cast<std::uint32_t>(plan.intent.stage_count);
  plan.cpu_tail_count = 1;
  plan.setup_failure_policy = hal::SetupFailurePolicy::retain_old_program;
  plan.migration_policy = VideoExecutionMigrationPolicy::
      prepared_then_same_resolution_compatibility;

  request.resident.wait = wait;
  return request;
}

} // namespace studiocast::video
