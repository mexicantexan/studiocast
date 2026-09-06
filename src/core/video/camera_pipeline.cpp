#include "camera_pipeline.h"

#include "core/video/capture_error_policy.h"
#include "core/video/effects/broadcast_effect_contract.h"
#include "core/video/mjpeg_decode.h"
#include "core/video/open_vulkan_eye_contact.h"
#include "core/video/open_vulkan_video_noise_removal.h"
#include "core/video/scaling_policy.h"
#include "core/video/v4l2loopback.h"
#include "core/video/yuyv_passthrough.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

namespace studiocast::video {

// Implemented by the static backend-composition target. CameraPipeline only
// consumes this neutral source interface; backend construction remains behind
// the factory.
std::unique_ptr<IProductionVideoBackendSetupSource>
CreateStaticProductionVideoBackendSetupSource() noexcept;

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
    intent->preferences.compute_family = hal::ComputeFamilyAllowance::cuda_family;
    intent->preferences.effects_engine = hal::EffectsEnginePreferenceConstraint::maxine;
    intent->preferences.strict_effects_engine = true;
    break;
  case effects::EffectsEnginePreference::open_cuda:
    intent->backend.mode = hal::BackendRequestMode::explicit_backend;
    intent->backend.backend = hal::BackendId::cuda;
    intent->backend.strict = true;
    intent->preferences.compute_family = hal::ComputeFamilyAllowance::cuda_family;
    intent->preferences.effects_engine = hal::EffectsEnginePreferenceConstraint::open_cuda;
    intent->preferences.strict_effects_engine = true;
    break;
  case effects::EffectsEnginePreference::auto_select:
    intent->backend.mode = hal::BackendRequestMode::automatic;
    intent->preferences.compute_family = hal::ComputeFamilyAllowance::any;
    intent->preferences.effects_engine = hal::EffectsEnginePreferenceConstraint::automatic;
    intent->preferences.strict_effects_engine = false;
    break;
  }
  if (compute_backend == ComputeBackendPreference::vulkan) {
    intent->backend.mode = hal::BackendRequestMode::explicit_backend;
    intent->backend.backend = hal::BackendId::vulkan;
    intent->backend.strict = true;
    intent->preferences.compute_family = hal::ComputeFamilyAllowance::vulkan_only;
    if (intent->requested_provider ==
            hal::SemanticEffectProviderId::invalid &&
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
    intent->preferences.compute_family = hal::ComputeFamilyAllowance::cuda_family;
  } else if (compute_backend == ComputeBackendPreference::cpu) {
    intent->backend.mode = hal::BackendRequestMode::explicit_backend;
    intent->backend.backend = hal::BackendId::cpu;
    intent->backend.strict = true;
    intent->preferences.compute_family = hal::ComputeFamilyAllowance::cpu_only;
  }
}

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kHashOffset = UINT64_C(1469598103934665603);
constexpr std::uint64_t kHashPrime = UINT64_C(1099511628211);

std::uint64_t MixHash(std::uint64_t hash, std::uint64_t value) noexcept {
  hash ^= value;
  hash *= kHashPrime;
  return hash;
}

std::uint64_t MixText(std::uint64_t hash, std::string_view value) noexcept {
  for (const char character : value) {
    hash = MixHash(
        hash, static_cast<unsigned char>(character));
  }
  return hash;
}

double ToMs(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

struct ExponentialAverage {
  void Add(double value) {
    value_ = initialized_ ? value_ * 0.9 + value * 0.1 : value;
    initialized_ = true;
  }
  [[nodiscard]] double Value() const noexcept {
    return initialized_ ? value_ : 0.0;
  }

private:
  double value_ = 0.0;
  bool initialized_ = false;
};

int ParseVideoIndex(const std::string &name) {
  if (name.rfind("video", 0) != 0 || name.size() <= 5)
    return -1;
  int result = 0;
  for (std::size_t index = 5; index < name.size(); ++index) {
    const char value = name[index];
    if (value < '0' || value > '9')
      return -1;
    result = result * 10 + (value - '0');
    if (result > 4096)
      return -1;
  }
  return result;
}

std::string LowerAscii(std::string value) {
  for (char &character : value) {
    character = static_cast<char>(
        std::tolower(static_cast<unsigned char>(character)));
  }
  return value;
}

int CameraScore(const VideoDevice &device) {
  int score = device.driver == "uvcvideo" ? 10 : 0;
  const auto name = LowerAscii(device.name);
  if (name.find("ir") != std::string::npos ||
      name.find("depth") != std::string::npos ||
      name.find("metadata") != std::string::npos) {
    score -= 50;
  }
  if (name.find("hd") != std::string::npos)
    score += 5;
  if (name.find("camera") != std::string::npos)
    score += 2;
  return score;
}

std::vector<VideoDevice> CandidateCameras() {
  auto report = ProbeLoopback();
  std::vector<VideoDevice> cameras;
  for (const auto &device : report.devices) {
    if (!device.is_loopback && device.can_read)
      cameras.push_back(device);
  }
  std::sort(cameras.begin(), cameras.end(),
            [](const VideoDevice &left, const VideoDevice &right) {
              const int left_score = CameraScore(left);
              const int right_score = CameraScore(right);
              if (left_score != right_score)
                return left_score > right_score;
              const int left_index = ParseVideoIndex(left.sys_name);
              const int right_index = ParseVideoIndex(right.sys_name);
              if (left_index != right_index) {
                if (left_index < 0)
                  return false;
                if (right_index < 0)
                  return true;
                return left_index < right_index;
              }
              return left.dev_node < right.dev_node;
            });
  return cameras;
}

std::string DefaultLoopback(std::string *error) {
  const auto report = ProbeLoopback();
  for (const auto &device : report.devices) {
    if (device.is_loopback && device.can_write)
      return device.dev_node;
  }
  if (error != nullptr)
    *error = "No writable v4l2loopback device found.";
  return {};
}

bool ActualMatches(const ActualFormat &actual, int width, int height, int fps,
                   PixelFormat format, bool strict_fps) {
  return actual.width == width && actual.height == height &&
         actual.format == format && (!strict_fps || actual.fps == fps);
}

std::string ActualMismatch(const ActualFormat &actual, int width, int height,
                           int fps, PixelFormat format, bool strict_fps) {
  std::ostringstream stream;
  stream << "Requested " << PixelFormatName(format) << " " << width << "x"
         << height;
  if (strict_fps)
    stream << "@" << fps;
  stream << ", negotiated " << PixelFormatName(actual.format) << " "
         << actual.width << "x" << actual.height;
  if (actual.fps > 0)
    stream << "@" << actual.fps;
  return stream.str();
}

bool HasResidentSemanticEffects(
    const effects::BroadcastCameraEffects &effects,
    ComputeBackendPreference compute_backend) {
  return effects.video_noise_removal.enabled || effects.eye_contact.enabled ||
         effects.virtual_background.mode !=
             effects::VirtualBackgroundMode::none ||
         effects.virtual_key_light.enabled || effects.auto_frame.enabled ||
         effects.vignette.enabled ||
         (effects.mirror &&
          compute_backend == ComputeBackendPreference::vulkan);
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
  descriptor.planes[0].size_bytes =
      stride * static_cast<std::uint64_t>(height);
  descriptor.planes[0].width = static_cast<std::uint32_t>(width);
  descriptor.planes[0].height = static_cast<std::uint32_t>(height);
  descriptor.planes[0].alignment_bytes = 1;
  return descriptor;
}

struct ManagerWaitContext {
  std::atomic_bool *stop = nullptr;
};

bool ManagerStopRequested(void *context) noexcept {
  const auto *wait = static_cast<const ManagerWaitContext *>(context);
  return wait == nullptr || wait->stop == nullptr || wait->stop->load();
}

std::uint64_t ManagerNowNanoseconds(void *) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now().time_since_epoch())
          .count());
}

void ManagerPollWait(void *context, std::uint64_t nanoseconds) noexcept {
  if (ManagerStopRequested(context))
    return;
  const auto bounded =
      std::min<std::uint64_t>(nanoseconds, UINT64_C(1000000));
  std::this_thread::sleep_for(std::chrono::nanoseconds(bounded));
}

ProductionVideoBackendSetupRequest BuildSetupRequest(
    const effects::BroadcastCameraEffects &effects,
    ComputeBackendPreference compute_backend, int input_width, int input_height,
    int output_width, int output_height, PixelFormat output_format,
    std::uint64_t configuration_generation,
    std::uint64_t geometry_generation, std::uint64_t plan_id,
    ManagerWaitContext *wait_context, const CpuPrimitiveTable &cpu_table) {
  ProductionVideoBackendSetupRequest request;
  const bool cpu_mirror =
      effects.mirror && compute_backend != ComputeBackendPreference::vulkan;
  if (!HasResidentSemanticEffects(effects, compute_backend)) {
    request.kind = ProductionVideoBackendSetupKind::cpu_primitives_only;
    request.cpu.table = &cpu_table;
    request.cpu.required_primitive_bits =
        CpuPrimitiveCapabilityBit(
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
  // Mirror is a product-visible operation but the non-Vulkan implementation
  // is a declared host primitive finalizer, not a CPU semantic provider.
  // Strict Vulkan retains mirror in the resident canonical stage list.
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
    stage.intent.required = false;
    stage.intent.failure_action =
        hal::FailureAction::fail_open_preserve_input;
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
  const bool temporal =
      effects.video_noise_removal.enabled || effects.eye_contact.enabled ||
      effects.auto_frame.enabled ||
      (effects.virtual_background.mode !=
           effects::VirtualBackgroundMode::none &&
       effects.virtual_background.greenscreen_temporal);
  plan.temporal.temporal = temporal;
  plan.temporal.requires_contiguous_capture_sequence = temporal;
  plan.temporal.history_frames =
      effects.video_noise_removal.enabled ? 5u : (temporal ? 1u : 0u);
  plan.temporal.frame_drop_behavior =
      temporal ? hal::FrameDropBehavior::reset_temporal_state
               : hal::FrameDropBehavior::latest_frame_wins;
  plan.temporal.discontinuity_action =
      hal::TemporalDiscontinuityAction::reset_and_continue;
  plan.cpu_tails[0].id = hal::CpuTailId::backend_resolved_output;
  plan.cpu_tails[0].after_stage_id =
      plan.intent.stages[plan.intent.stage_count - 1].intent.stage_id;
  plan.cpu_tails[0].required = true;
  plan.cpu_tails[0].placement = hal::DeclaredCpuTail::Placement::after_stage;
  plan.cpu_tails[0].boundary_index =
      static_cast<std::uint32_t>(plan.intent.stage_count);
  plan.cpu_tail_count = 1;
  plan.setup_failure_policy = hal::SetupFailurePolicy::retain_old_program;
  plan.migration_policy =
      VideoExecutionMigrationPolicy::prepared_then_same_resolution_compatibility;

  request.resident.wait.timeout_nanoseconds = UINT64_C(100000000);
  request.resident.wait.poll_interval_nanoseconds = UINT64_C(1000000);
  request.resident.wait.maximum_poll_count = 100;
  request.resident.wait.stop_requested = &ManagerStopRequested;
  request.resident.wait.cancel_requested = &ManagerStopRequested;
  request.resident.wait.now_nanoseconds = &ManagerNowNanoseconds;
  request.resident.wait.poll_wait = &ManagerPollWait;
  request.resident.wait.callback_context = wait_context;
  request.resident.wait.callback_context_generation =
      configuration_generation == 0 ? 1 : configuration_generation;
  request.resident.wait.callback_context_lifetime =
      hal::ResourceLifetime::adapter_context;
  return request;
}

std::string_view BackendName(hal::BackendId backend) {
  switch (backend) {
  case hal::BackendId::cpu:
    return "cpu";
  case hal::BackendId::cuda:
  case hal::BackendId::open_video_onnxruntime:
    return "cuda";
  case hal::BackendId::maxine:
    return "maxine";
  case hal::BackendId::vulkan:
    return "vulkan";
  case hal::BackendId::invalid:
  case hal::BackendId::test_fake:
    return {};
  }
  return {};
}

std::string_view EffectId(hal::SemanticEffectId semantic) {
  using namespace effects::contract;
  switch (semantic) {
  case hal::SemanticEffectId::video_noise_removal:
    return kEffectIdVideoNoiseRemoval;
  case hal::SemanticEffectId::eye_contact:
    return kEffectIdEyeContact;
  case hal::SemanticEffectId::virtual_background_blur:
    return kEffectIdVirtualBackgroundBlur;
  case hal::SemanticEffectId::virtual_background_remove:
    return kEffectIdVirtualBackgroundRemove;
  case hal::SemanticEffectId::virtual_background_replace:
    return kEffectIdVirtualBackgroundReplace;
  case hal::SemanticEffectId::virtual_key_light:
    return kEffectIdVirtualKeyLight;
  case hal::SemanticEffectId::auto_frame:
    return kEffectIdAutoFrame;
  case hal::SemanticEffectId::vignette:
    return kEffectIdVignette;
  case hal::SemanticEffectId::mirror:
    return kEffectIdMirror;
  case hal::SemanticEffectId::invalid:
    return {};
  }
  return {};
}

void ProjectLegacyTransfers(const ProductionVideoBackendManagerSnapshot &source,
                            CameraPipelineStatus::OpenCudaTransfers *cuda,
                            CameraPipelineStatus::OpenVulkanTransfers *vulkan,
                            CameraPipelineStatus::MaxineTransfers *maxine) {
  *cuda = {};
  *vulkan = {};
  *maxine = {};
  const hal::ExecutionTelemetry *telemetry = nullptr;
  hal::BackendId backend = source.readiness.selected_backend;
  if (source.state == ProductionVideoBackendManagerState::resident_ready) {
    telemetry = &source.resident.telemetry;
  } else if (source.state ==
             ProductionVideoBackendManagerState::compatibility_ready) {
    telemetry = &source.compatibility.telemetry;
  }
  if (telemetry == nullptr)
    return;
  const auto &transfers = telemetry->transfers;
  if (backend == hal::BackendId::cuda ||
      backend == hal::BackendId::open_video_onnxruntime) {
    cuda->active_frames = telemetry->committed_output_frames;
    cuda->upload_calls = transfers.uploads;
    cuda->download_calls =
        transfers.final_readbacks + transfers.intermediate_readbacks;
    cuda->final_download_calls = transfers.final_readbacks;
    cuda->forced_sync_calls = transfers.synchronizations;
    cuda->cpu_tail_stage_calls = transfers.cpu_tail_boundaries;
  } else if (backend == hal::BackendId::vulkan) {
    vulkan->active_frames = telemetry->committed_output_frames;
    vulkan->upload_calls = transfers.uploads;
    vulkan->download_calls =
        transfers.final_readbacks + transfers.intermediate_readbacks;
    vulkan->final_download_calls = transfers.final_readbacks;
    vulkan->forced_sync_calls = transfers.synchronizations;
    vulkan->cpu_tail_stage_calls = transfers.cpu_tail_boundaries;
    vulkan->fallback_frames = telemetry->fallback_frames;
  } else if (backend == hal::BackendId::maxine) {
    maxine->active_frames = telemetry->committed_output_frames;
    maxine->upload_attempts = transfers.uploads;
    maxine->upload_calls = transfers.uploads;
    maxine->download_calls =
        transfers.final_readbacks + transfers.intermediate_readbacks;
    maxine->final_download_attempts = transfers.final_readbacks;
    maxine->final_download_calls = transfers.final_readbacks;
    maxine->forced_sync_attempts = transfers.synchronizations;
    maxine->forced_sync_calls = transfers.synchronizations;
    maxine->cpu_tail_stage_calls = transfers.cpu_tail_boundaries;
    maxine->setup_attempts = source.setup_attempts;
    maxine->setup_successes = source.setup_successes;
  }
}

} // namespace

std::string ComputeBackendPreferenceToString(ComputeBackendPreference value) {
  switch (value) {
  case ComputeBackendPreference::cpu:
    return "cpu";
  case ComputeBackendPreference::cuda:
    return "cuda";
  case ComputeBackendPreference::vulkan:
    return "vulkan";
  case ComputeBackendPreference::auto_select:
    return "auto";
  }
  return "auto";
}

std::string ComputeBackendKindToString(ComputeBackendKind value) {
  switch (value) {
  case ComputeBackendKind::cpu:
    return "cpu";
  case ComputeBackendKind::cuda:
    return "cuda";
  case ComputeBackendKind::vulkan:
    return "vulkan";
  }
  return "cpu";
}

bool ParseComputeBackendPreference(std::string_view value,
                                   ComputeBackendPreference *output) {
  std::string normalized(value);
  const auto first = normalized.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return false;
  const auto last = normalized.find_last_not_of(" \t\r\n");
  normalized = LowerAscii(normalized.substr(first, last - first + 1));
  ComputeBackendPreference result;
  if (normalized == "auto" || normalized == "auto_select" ||
      normalized == "autoselect") {
    result = ComputeBackendPreference::auto_select;
  } else if (normalized == "cpu") {
    result = ComputeBackendPreference::cpu;
  } else if (normalized == "cuda") {
    result = ComputeBackendPreference::cuda;
  } else if (normalized == "vulkan") {
    result = ComputeBackendPreference::vulkan;
  } else {
    return false;
  }
  if (output != nullptr)
    *output = result;
  return true;
}

ComputeBackendPreference
ParseComputeBackendPreferenceOr(std::string_view value,
                                ComputeBackendPreference fallback) {
  ComputeBackendPreference result = fallback;
  return ParseComputeBackendPreference(value, &result) ? result : fallback;
}

ComputeBackendSelection
ResolveComputeBackendSelection(ComputeBackendPreference preference,
                               const ComputeBackendAvailability &available,
                               bool requested) {
  ComputeBackendSelection selection;
  selection.preference = preference;
  if (!requested)
    return selection;
  const auto unavailable = [&](std::string reason) {
    selection.resolved = ComputeBackendKind::cpu;
    selection.degraded = true;
    selection.fallback_reason = reason;
    selection.degraded_reason = std::move(reason);
  };
  switch (preference) {
  case ComputeBackendPreference::cpu:
    unavailable("CPU selected; GPU-backed effects are unavailable.");
    break;
  case ComputeBackendPreference::cuda:
    if (available.cuda_available)
      selection.resolved = ComputeBackendKind::cuda;
    else
      unavailable(available.cuda_unavailable_reason.empty()
                      ? "CUDA requested but unavailable."
                      : available.cuda_unavailable_reason);
    break;
  case ComputeBackendPreference::vulkan:
    if (available.vulkan_available)
      selection.resolved = ComputeBackendKind::vulkan;
    else
      unavailable(available.vulkan_unavailable_reason.empty()
                      ? "Vulkan requested but unavailable."
                      : available.vulkan_unavailable_reason);
    break;
  case ComputeBackendPreference::auto_select:
    if (available.cuda_available)
      selection.resolved = ComputeBackendKind::cuda;
    else if (available.vulkan_available)
      selection.resolved = ComputeBackendKind::vulkan;
    else
      unavailable("No GPU compute backend is available.");
    break;
  }
  return selection;
}

bool ComputeBackendAllowsCudaVideoCompute(ComputeBackendPreference preference) {
  return preference == ComputeBackendPreference::auto_select ||
         preference == ComputeBackendPreference::cuda;
}

std::string ResolveActiveComputeBackendName(
    const ComputeBackendSelection &selection, bool requested,
    bool maxine_available, bool cuda_available, bool vulkan_available) {
  if (!requested)
    return "cpu";
  if (selection.resolved == ComputeBackendKind::vulkan && vulkan_available)
    return "vulkan";
  if (selection.resolved == ComputeBackendKind::cuda && maxine_available)
    return "maxine";
  if (selection.resolved == ComputeBackendKind::cuda && cuda_available)
    return "cuda";
  return "cpu";
}

bool MarkGpuBackendActiveFrame(bool &active, std::uint64_t &frames) {
  if (active)
    return false;
  active = true;
  ++frames;
  return true;
}

MultiIslandStatusValidationResult BuildCameraPipelineHalStatusSnapshot(
    const CameraPipelineHalStatusProjectionInput &input,
    MultiIslandVideoRuntimeStatusSnapshot *output) noexcept {
  using CapabilityCode = effects::EffectCapabilityCode;
  using Lane = ProductCapabilityLaneId;
  if (output == nullptr || input.effects == nullptr) {
    if (output != nullptr)
      *output = {};
    MultiIslandStatusValidationResult invalid;
    invalid.code = MultiIslandStatusValidationCode::invalid_generation;
    return invalid;
  }

  const auto requested = [&](hal::SemanticEffectId effect) {
    switch (effect) {
    case hal::SemanticEffectId::video_noise_removal:
      return input.effects->video_noise_removal.enabled;
    case hal::SemanticEffectId::eye_contact:
      return input.effects->eye_contact.enabled;
    case hal::SemanticEffectId::virtual_background_blur:
      return input.effects->virtual_background.mode ==
             effects::VirtualBackgroundMode::blur;
    case hal::SemanticEffectId::virtual_background_remove:
      return input.effects->virtual_background.mode ==
             effects::VirtualBackgroundMode::remove;
    case hal::SemanticEffectId::virtual_background_replace:
      return input.effects->virtual_background.mode ==
             effects::VirtualBackgroundMode::replace;
    case hal::SemanticEffectId::virtual_key_light:
      return input.effects->virtual_key_light.enabled;
    case hal::SemanticEffectId::auto_frame:
      return input.effects->auto_frame.enabled;
    case hal::SemanticEffectId::vignette:
      return input.effects->vignette.enabled;
    case hal::SemanticEffectId::mirror:
      return input.effects->mirror;
    case hal::SemanticEffectId::invalid:
      return false;
    }
    return false;
  };
  const auto lane_for_backend = [](hal::BackendId backend) {
    switch (backend) {
    case hal::BackendId::maxine:
      return Lane::maxine;
    case hal::BackendId::cuda:
    case hal::BackendId::open_video_onnxruntime:
      return Lane::open_cuda;
    case hal::BackendId::vulkan:
      return Lane::open_vulkan;
    case hal::BackendId::cpu:
      return Lane::cpu;
    case hal::BackendId::invalid:
    case hal::BackendId::test_fake:
      return Lane::invalid;
    }
    return Lane::invalid;
  };
  const auto capability_code = [](Lane lane, hal::SemanticEffectId effect) {
    switch (lane) {
    case Lane::maxine:
      return CapabilityCode::maxine_effect_usable;
    case Lane::open_cuda:
      return CapabilityCode::open_cuda_effect_available;
    case Lane::open_vulkan:
      if (effect == hal::SemanticEffectId::mirror) {
        return CapabilityCode::open_vulkan_mirror_production_ready;
      }
      if (effect == hal::SemanticEffectId::vignette) {
        return CapabilityCode::
            open_vulkan_vignette_fixed_center_production_ready;
      }
      return CapabilityCode::none;
    case Lane::cpu:
    case Lane::invalid:
      return CapabilityCode::none;
    }
    return CapabilityCode::none;
  };
  const auto provider_for = [](hal::BackendId backend,
                               hal::SemanticEffectId effect) {
    if (backend == hal::BackendId::vulkan) {
      return hal::SemanticEffectProviderId::open_vulkan_production_kernels;
    }
    if (backend == hal::BackendId::cuda ||
        backend == hal::BackendId::open_video_onnxruntime) {
      return hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
    }
    if (backend == hal::BackendId::maxine) {
      if (effect == hal::SemanticEffectId::eye_contact ||
          effect == hal::SemanticEffectId::auto_frame) {
        return hal::SemanticEffectProviderId::maxine_ar;
      }
      return effect == hal::SemanticEffectId::vignette
                 ? hal::SemanticEffectProviderId::maxine_cuda_utility
                 : hal::SemanticEffectProviderId::maxine_vfx;
    }
    return hal::SemanticEffectProviderId::invalid;
  };
  const auto desired_backend = [&](hal::SemanticEffectId effect) {
    if ((effect == hal::SemanticEffectId::mirror ||
         effect == hal::SemanticEffectId::vignette) &&
        input.compute_backend == ComputeBackendPreference::vulkan) {
      return hal::BackendId::vulkan;
    }
    if (input.effects->engine == effects::EffectsEnginePreference::open_cuda)
      return hal::BackendId::cuda;
    if (input.effects->engine == effects::EffectsEnginePreference::maxine)
      return hal::BackendId::maxine;
    if (input.compute_backend == ComputeBackendPreference::cuda)
      return hal::BackendId::cuda;
    return hal::BackendId::maxine;
  };
  const auto route_for_backend = [](hal::BackendId backend) {
    switch (backend) {
    case hal::BackendId::vulkan:
      return PreparedExecutionRouteId::final_visual;
    case hal::BackendId::maxine:
      return PreparedExecutionRouteId::maxine_resident;
    case hal::BackendId::cuda:
    case hal::BackendId::open_video_onnxruntime:
      return PreparedExecutionRouteId::open_cuda_virtual_background;
    case hal::BackendId::cpu:
    case hal::BackendId::invalid:
    case hal::BackendId::test_fake:
      return PreparedExecutionRouteId::invalid;
    }
    return PreparedExecutionRouteId::invalid;
  };

  MultiIslandVideoRuntimeStatusBuildInput build;
  build.status_generation = input.status_generation;
  build.lifecycle = input.lifecycle;
  build.effect_count = kCanonicalVideoEffectCount;
  for (std::size_t index = 0; index < build.effect_count; ++index) {
    build.effects[index].effect = kCanonicalVideoEffects[index];
    build.effects[index].status = hal::StatusId::passthrough;
  }

  const hal::PreparedProgram *program = nullptr;
  const hal::ReadinessSnapshot *readiness = nullptr;
  const hal::ExecutionTelemetry *telemetry = nullptr;
  const hal::CommittedFrameSnapshot *committed = nullptr;
  hal::PreparedProgram compatibility_program;
  hal::ProgramInvalidationKey compatibility_key;

  if (input.backend.state ==
      ProductionVideoBackendManagerState::resident_ready) {
    const auto &resident = input.backend.resident;
    if (!input.backend.setup_complete || !input.backend.resident_prepared ||
        !resident.prepared || resident.quarantined || resident.stopped) {
      *output = {};
      MultiIslandStatusValidationResult invalid;
      invalid.code = MultiIslandStatusValidationCode::invalid_island;
      return invalid;
    }
    program = &resident.plan.prepared_program;
    readiness = &resident.readiness;
    telemetry = &resident.telemetry;
    committed = &resident.latest_committed_frame;
  } else if (input.backend.state ==
             ProductionVideoBackendManagerState::compatibility_ready) {
    const auto &adapter = input.backend.compatibility;
    const auto &topology = input.backend.compatibility_topology;
    if (!input.backend.setup_complete || input.backend.resident_prepared ||
        !adapter.prepared || adapter.stopped || adapter.plan_id == 0 ||
        topology.domain_count != 1 || !topology.exact_ordered_boundaries ||
        !topology.cross_adapter_residency_forbidden ||
        !hal::ValidateExecutionDomain(adapter.execution_domain) ||
        topology.domains[0].domain != adapter.execution_domain) {
      *output = {};
      MultiIslandStatusValidationResult invalid;
      invalid.code = MultiIslandStatusValidationCode::invalid_island;
      return invalid;
    }

    compatibility_program.program_id = adapter.plan_id;
    compatibility_program.owner_adapter_instance_id =
        adapter.execution_domain.adapter_instance_id;
    compatibility_program.execution_domain = adapter.execution_domain;
    compatibility_program.requested_backend =
        input.backend.readiness.requested_backend;
    compatibility_program.strict_backend_selection =
        input.effects->engine !=
            effects::EffectsEnginePreference::auto_select ||
        input.compute_backend != ComputeBackendPreference::auto_select;
    compatibility_program.requested_provider =
        hal::SemanticEffectProviderId::invalid;
    compatibility_program.strict_provider_selection = false;
    compatibility_program.requires_upload =
        topology.expected_transfers.uploads != 0;
    compatibility_program.requires_final_readback =
        topology.expected_transfers.final_readbacks != 0;

    if (adapter.latest_committed_frame.valid) {
      compatibility_key = adapter.latest_committed_frame.invalidation_key;
    } else {
      compatibility_key.generations.configuration =
          input.configuration_generation == 0
              ? 1
              : input.configuration_generation;
      compatibility_key.generations.geometry =
          input.geometry_generation == 0 ? 1 : input.geometry_generation;
      compatibility_key.generations.model = 1;
      compatibility_key.generations.provider = 1;
      compatibility_key.generations.device = 1;
      compatibility_key.generations.context = 1;
      compatibility_key.canonical_plan_hash = adapter.plan_id;
      compatibility_key.execution_domain_hash =
          ProductionVideoExecutionDomainHash(adapter.execution_domain);
    }
    compatibility_program.invalidation_key = compatibility_key;

    auto resident_effects = *input.effects;
    if (input.compute_backend != ComputeBackendPreference::vulkan)
      resident_effects.mirror = false;
    const auto canonical =
        effects::BuildBroadcastEffectsPlan(resident_effects);
    for (const auto &id : canonical.ordered_effect_ids) {
      if (compatibility_program.stage_count >=
          compatibility_program.stages.size()) {
        break;
      }
      const auto semantic = SemanticForEffectId(id);
      if (semantic == hal::SemanticEffectId::invalid)
        continue;
      auto &stage =
          compatibility_program.stages[compatibility_program.stage_count];
      stage.stage_id = static_cast<std::uint32_t>(
          compatibility_program.stage_count + 1);
      stage.semantic = semantic;
      stage.backend = adapter.execution_domain.backend;
      stage.provider = provider_for(stage.backend, semantic);
      stage.execution_domain = adapter.execution_domain;
      stage.failure_action =
          hal::FailureAction::fail_open_preserve_input;
      stage.required = false;
      ++compatibility_program.stage_count;
    }
    program = &compatibility_program;
    readiness = &adapter.readiness;
    telemetry = &adapter.telemetry;
    committed = &adapter.latest_committed_frame;
  }

  if (program != nullptr) {
    build.program_generations = program->invalidation_key.generations;
  } else {
    build.program_generations.configuration =
        input.configuration_generation == 0 ? 1
                                            : input.configuration_generation;
    build.program_generations.geometry =
        input.geometry_generation == 0 ? 1 : input.geometry_generation;
    build.program_generations.model = 1;
    build.program_generations.provider = 1;
    build.program_generations.device = 1;
    build.program_generations.context = 1;
  }
  build.generations.capability = input.status_generation;
  build.generations.configuration =
      build.program_generations.configuration;
  build.generations.provider = build.program_generations.provider;
  build.generations.device = build.program_generations.device;
  build.generations.context = build.program_generations.context;
  CapabilityRefreshSnapshot refresh;
  refresh.status = CapabilityRefreshStatus::succeeded;
  refresh.code = CapabilityCode::capability_snapshot_current;
  refresh.requested_generation = input.status_generation;
  refresh.completed_generation = input.status_generation;
  build.capabilities =
      MakeFailClosedProductCapabilitySnapshot(build.generations, refresh);

  if (program != nullptr && readiness != nullptr && telemetry != nullptr &&
      committed != nullptr) {
    const auto route = route_for_backend(readiness->selected_backend);
    if (program->program_id == 0 || program->stage_count == 0 ||
        program->stage_count > program->stages.size() ||
        committed->stage_count > committed->stages.size() ||
        !hal::ValidateExecutionDomain(program->execution_domain) ||
        route == PreparedExecutionRouteId::invalid) {
      *output = {};
      MultiIslandStatusValidationResult invalid;
      invalid.code = MultiIslandStatusValidationCode::invalid_island;
      return invalid;
    }

    auto &island = build.islands[build.island_count++];
    island.route = route;
    island.island_id =
        (static_cast<std::uint64_t>(route) << 56u) ^ program->program_id;
    if (island.island_id == 0)
      island.island_id = program->program_id;
    island.adapter_instance_id = program->owner_adapter_instance_id;
    island.snapshot_generation = input.status_generation;
    island.capability_generation = input.status_generation;
    island.backend = program->execution_domain.backend;
    island.execution_domain = program->execution_domain;
    island.readiness.configured = true;
    island.readiness.program_id = program->program_id;
    island.readiness.invalidation_key = program->invalidation_key;
    island.readiness.value = *readiness;
    island.telemetry = *telemetry;
    island.latest_committed_frame = *committed;
    island.valid = true;
    island.current = true;

    auto &resolution = island.resolution;
    resolution.request_mode =
        program->strict_backend_selection
            ? hal::BackendRequestMode::explicit_backend
            : hal::BackendRequestMode::automatic;
    resolution.strict_backend_selection =
        program->strict_backend_selection;
    resolution.status = readiness->status;
    resolution.reason = readiness->reason;
    resolution.requested_backend = program->requested_backend;
    resolution.selected_backend = readiness->selected_backend;
    resolution.selected_provider = readiness->provider;
    resolution.stage_count = program->stage_count;
    resolution.available = readiness->available;
    resolution.degraded = readiness->degraded;
    resolution.fallback =
        program->requested_backend != hal::BackendId::invalid &&
        program->requested_backend != readiness->selected_backend;

    for (std::size_t index = 0; index < program->stage_count; ++index) {
      const auto &stage = program->stages[index];
      auto &resolved = resolution.stages[index];
      resolved.intent.stage_id = stage.stage_id;
      resolved.intent.semantic = stage.semantic;
      resolved.intent.required = stage.required;
      resolved.intent.failure_action = stage.failure_action;
      resolved.status = readiness->degraded ? hal::StatusId::degraded
                                            : hal::StatusId::ready;
      resolved.reason = readiness->reason;
      resolved.selected_backend = stage.backend;
      resolved.provider = stage.provider;
      resolved.adapter_instance_id = program->owner_adapter_instance_id;
      resolved.execution_domain = stage.execution_domain;
      resolved.fallback_selected = resolution.fallback;
      for (std::size_t evidence_index = 0;
           evidence_index < readiness->evidence_count &&
           evidence_index < resolved.evidence.size();
           ++evidence_index) {
        resolved.evidence[resolved.evidence_count++] =
            readiness->evidence[evidence_index];
      }

      auto &record = island.stages[island.stage_count++];
      record.stage_id = stage.stage_id;
      record.effect = stage.semantic;
      record.backend = stage.backend;
      record.provider = stage.provider;
      record.status = resolved.status;
      record.reason = resolved.reason;
      record.failure_action = stage.failure_action;
      record.required = stage.required;
      record.resolved = true;
      for (std::size_t committed_index = 0;
           committed_index < committed->stage_count; ++committed_index) {
        const auto &activity = committed->stages[committed_index];
        if (activity.stage_id == stage.stage_id &&
            activity.semantic == stage.semantic) {
          record.committed = true;
          record.committed_status = activity.status;
          break;
        }
      }

      const Lane lane = lane_for_backend(stage.backend);
      auto *capability =
          FindProductCapabilityLane(&build.capabilities, stage.semantic, lane);
      const auto code = capability_code(lane, stage.semantic);
      if (capability == nullptr || lane == Lane::invalid ||
          lane == Lane::cpu || code == CapabilityCode::none) {
        continue;
      }
      capability->disposition =
          resolved.status == hal::StatusId::degraded
              ? ProductCapabilityDisposition::usable_with_degraded_behavior
              : ProductCapabilityDisposition::production_usable;
      capability->code = code;
      capability->nested_code = CapabilityCode::none;
      capability->semantic.semantic = stage.semantic;
      capability->semantic.provider = stage.provider;
      capability->semantic.status = resolved.status;
      capability->semantic.reason = resolved.reason;
      capability->semantic.live_stage = true;
      capability->semantic.production_eligible = true;
      capability->semantic.exact_evidence = true;
      capability->semantic.supports_tracked_vignette_center = false;
      capability->semantic.evidence_count = resolved.evidence_count;
      for (std::size_t evidence_index = 0;
           evidence_index < resolved.evidence_count; ++evidence_index) {
        capability->semantic.evidence[evidence_index] =
            resolved.evidence[evidence_index];
      }
    }

    if (committed->valid) {
      build.writer_commit = {
          true, committed->capture_sequence,
          committed->output_commit_sequence};
    }
  }

  for (std::size_t effect_index = 0; effect_index < build.effect_count;
       ++effect_index) {
    auto &state = build.effects[effect_index];
    if (!requested(state.effect))
      continue;
    const auto desired = desired_backend(state.effect);
    state.requested = true;
    state.requested_backend = desired;
    state.requested_provider =
        input.effects->engine == effects::EffectsEnginePreference::open_cuda
            ? hal::SemanticEffectProviderId::invalid
            : provider_for(desired, state.effect);
    state.capability_lane = lane_for_backend(desired);
    if (const auto *desired_capability = FindProductCapabilityLane(
            build.capabilities, state.effect, state.capability_lane)) {
      state.capability_code = desired_capability->code;
      state.nested_capability_code = desired_capability->nested_code;
      state.capability = desired_capability->semantic;
      state.available =
          IsProductCapabilityLaneEligible(build.capabilities,
                                          *desired_capability);
    }
    state.strict_backend_selection =
        input.effects->engine !=
            effects::EffectsEnginePreference::auto_select ||
        input.compute_backend != ComputeBackendPreference::auto_select;
    state.strict_provider_selection = false;
    state.status = hal::StatusId::unavailable;
    state.reason = hal::ReasonId::capability_evidence_incomplete;
    state.blocked = true;

    if (build.island_count == 0)
      continue;
    const auto &island = build.islands[0];
    for (std::size_t stage_index = 0;
         stage_index < island.stage_count; ++stage_index) {
      const auto &stage = island.stages[stage_index];
      if (stage.effect != state.effect)
        continue;
      const auto selected_lane = lane_for_backend(stage.backend);
      const auto *selected_capability = FindProductCapabilityLane(
          build.capabilities, state.effect, selected_lane);
      if (selected_capability == nullptr)
        continue;
      state.selected_capability_lane = selected_lane;
      state.selected_capability_code = selected_capability->code;
      state.selected_nested_capability_code =
          selected_capability->nested_code;
      state.selected_capability = selected_capability->semantic;
      state.selected_backend = stage.backend;
      state.selected_provider = stage.provider;
      state.status = stage.status;
      state.reason = stage.reason;
      state.failure_action = stage.failure_action;
      state.required = stage.required;
      state.available = true;
      state.selected = true;
      state.blocked = false;
      state.selected_island_id = island.island_id;
      state.fallback =
          state.selected_backend != state.requested_backend ||
          (state.requested_provider !=
               hal::SemanticEffectProviderId::invalid &&
           state.selected_provider != state.requested_provider);
      if (!stage.committed)
        break;
      state.commit = build.writer_commit;
      state.committed_stage_id = stage.stage_id;
      state.committed_stage_status = stage.committed_status;
      if (stage.committed_status == hal::StageExecutionStatus::succeeded) {
        const bool healthy =
            island.telemetry.runtime_status == hal::StatusId::ready ||
            island.telemetry.runtime_status == hal::StatusId::degraded ||
            island.telemetry.runtime_status == hal::StatusId::executed;
        state.active =
            input.lifecycle == VideoRuntimeLifecycle::running &&
            build.writer_commit.valid && healthy;
        if (state.active) {
          state.active_backend = stage.backend;
          state.active_provider = stage.provider;
          state.status = hal::StatusId::executed;
          state.reason = hal::ReasonId::none;
        }
      } else if (stage.committed_status ==
                 hal::StageExecutionStatus::optional_failed_open) {
        state.degraded = true;
        state.fallback = true;
        state.status = hal::StatusId::degraded;
        state.reason = hal::ReasonId::optional_stage_failed_open;
      }
      break;
    }
  }

  return BuildMultiIslandVideoRuntimeStatusSnapshot(build, output);
}

OpenVulkanVignettePlanCompatibility ApplyOpenVulkanVignettePlanCompatibility(
    const effects::BroadcastCameraEffects &fx,
    effects::BroadcastEffectsPlan *retained_plan) {
  OpenVulkanVignettePlanCompatibility result;
  if (retained_plan == nullptr || !fx.vignette.enabled ||
      !fx.vignette.center_on_tracked_face) {
    return result;
  }
  const auto contains = [&](std::string_view effect_id) {
    return std::find(retained_plan->ordered_effect_ids.begin(),
                     retained_plan->ordered_effect_ids.end(),
                     effect_id) != retained_plan->ordered_effect_ids.end();
  };
  if (!contains(effects::contract::kEffectIdVignette) ||
      !contains(effects::contract::kEffectIdAutoFrame)) {
    return result;
  }
  const std::string vignette_id(effects::contract::kEffectIdVignette);
  retained_plan->ordered_effect_ids.erase(
      std::remove(retained_plan->ordered_effect_ids.begin(),
                  retained_plan->ordered_effect_ids.end(), vignette_id),
      retained_plan->ordered_effect_ids.end());
  retained_plan->vignette_attach_to_effect_id.clear();
  result.blocked = true;
  result.reason_code = kOpenVulkanVignetteTrackedCenterNotSupportedReason;
  result.detail = kOpenVulkanVignetteTrackedCenterNotSupportedDetail;
  retained_plan->disabled.push_back(effects::DisabledEffectByRule{
      .id = vignette_id,
      .reason = "[" + std::string(result.reason_code) + "] " +
                std::string(result.detail)});
  return result;
}

OpenVulkanEyeContactPlanCompatibility
ApplyOpenVulkanEyeContactPlanCompatibility(
    effects::BroadcastEffectsPlan *retained_plan) {
  OpenVulkanEyeContactPlanCompatibility result;
  if (retained_plan == nullptr)
    return result;
  const std::string effect_id(effects::contract::kEffectIdEyeContact);
  const auto stage = std::find(retained_plan->ordered_effect_ids.begin(),
                               retained_plan->ordered_effect_ids.end(),
                               effect_id);
  if (stage == retained_plan->ordered_effect_ids.end())
    return result;
  retained_plan->ordered_effect_ids.erase(stage);
  if (retained_plan->vignette_attach_to_effect_id == effect_id)
    retained_plan->vignette_attach_to_effect_id.clear();
  result.blocked = true;
  result.reason_code = kOpenVulkanEyeContactUnavailableReason;
  result.blocker_code = kOpenVulkanEyeContactRuntimeUnavailableReason;
  result.detail = kOpenVulkanEyeContactUnavailableDetail;
  retained_plan->disabled.push_back(effects::DisabledEffectByRule{
      .id = effect_id,
      .reason = "[" + std::string(result.reason_code) + "] [" +
                std::string(result.blocker_code) + "] " +
                std::string(result.detail)});
  return result;
}

OpenVulkanVideoNoiseRemovalPlanCompatibility
ApplyOpenVulkanVideoNoiseRemovalPlanCompatibility(
    effects::BroadcastEffectsPlan *retained_plan) {
  OpenVulkanVideoNoiseRemovalPlanCompatibility result;
  if (retained_plan == nullptr)
    return result;
  const std::string effect_id(
      effects::contract::kEffectIdVideoNoiseRemoval);
  const auto stage = std::find(retained_plan->ordered_effect_ids.begin(),
                               retained_plan->ordered_effect_ids.end(),
                               effect_id);
  if (stage == retained_plan->ordered_effect_ids.end())
    return result;
  retained_plan->ordered_effect_ids.erase(stage);
  if (retained_plan->vignette_attach_to_effect_id == effect_id)
    retained_plan->vignette_attach_to_effect_id.clear();
  result.blocked = true;
  result.reason_code = kOpenVulkanVideoNoiseRemovalUnavailableReason;
  result.blocker_code =
      kOpenVulkanVideoNoiseRemovalRuntimeUnavailableReason;
  result.detail = kOpenVulkanVideoNoiseRemovalUnavailableDetail;
  retained_plan->disabled.push_back(effects::DisabledEffectByRule{
      .id = effect_id,
      .reason = "[" + std::string(result.reason_code) + "] [" +
                std::string(result.blocker_code) + "] " +
                std::string(result.detail)});
  return result;
}

bool EffectsPlanRequiresRebuild(
    const effects::BroadcastCameraEffects &applied_effects,
    std::uint64_t applied_generation,
    const effects::BroadcastCameraEffects &current_effects,
    std::uint64_t current_generation) {
  return applied_generation != current_generation ||
         applied_effects != current_effects;
}

bool EffectsFramePreparationCounters::ObservePublishedGeneration(
    std::uint64_t published_generation,
    std::uint64_t applied_generation) noexcept {
  ++generation_loads;
  if (published_generation == 0) {
    ++transition_sentinel_loads;
    return false;
  }
  if (published_generation == applied_generation)
    return false;
  ++snapshot_requests;
  return true;
}

void EffectsFramePreparationCounters::RecordConfigSnapshot(bool rebuilt) noexcept {
  ++config_lock_acquisitions;
  ++config_copies;
  if (rebuilt)
    ++rebuilds;
}

LiveEffectBackendAttribution::LiveEffectBackendAttribution() {
  // Nine stable "effect:backend" records plus separators fit well below this
  // setup-time reservation. SerializeActive therefore never grows the string
  // in the ordinary frame path.
  active_backends_.reserve(384);
}

void LiveEffectBackendAttribution::BeginFrame() { pending_.fill(0); }

void LiveEffectBackendAttribution::MarkEffectSucceeded(
    std::string_view effect_id, std::string_view backend) {
  const auto slot = EffectSlot(effect_id);
  const auto code = BackendCode(backend);
  if (slot < kEffectCount && code != 0)
    pending_[slot] = code;
}

bool LiveEffectBackendAttribution::RemoveEffect(std::string_view effect_id) {
  const auto slot = EffectSlot(effect_id);
  if (slot >= kEffectCount)
    return false;
  pending_[slot] = 0;
  if (active_[slot] == 0)
    return false;
  active_[slot] = 0;
  SerializeActive();
  return true;
}

void LiveEffectBackendAttribution::DiscardFrame() { pending_.fill(0); }

bool LiveEffectBackendAttribution::CommitSuccessfulFrame(
    const std::vector<std::string> &canonical_order) {
  if (active_ == pending_) {
    pending_.fill(0);
    return false;
  }
  active_ = pending_;
  pending_.fill(0);
  RememberOrder(canonical_order);
  SerializeActive();
  return true;
}

void LiveEffectBackendAttribution::Clear() {
  pending_.fill(0);
  active_.fill(0);
  active_order_.fill(0);
  active_order_size_ = 0;
  active_backends_.clear();
}

std::size_t LiveEffectBackendAttribution::EffectSlot(
    std::string_view effect_id) {
  static constexpr std::array<std::string_view, kEffectCount> ids = {
      "mirror", "virtual_background.blur", "virtual_background.remove",
      "virtual_background.replace", "auto_frame", "eye_contact",
      "video_noise_removal", "virtual_key_light", "vignette"};
  const auto found = std::find(ids.begin(), ids.end(), effect_id);
  return found == ids.end()
             ? kEffectCount
             : static_cast<std::size_t>(std::distance(ids.begin(), found));
}

std::uint8_t LiveEffectBackendAttribution::BackendCode(
    std::string_view backend) {
  if (backend == "vulkan" || backend == "open_vulkan")
    return 1;
  if (backend == "cuda" || backend == "open_cuda")
    return 2;
  if (backend == "maxine")
    return 3;
  if (backend == "cpu")
    return 4;
  return 0;
}

std::string_view LiveEffectBackendAttribution::EffectId(std::size_t slot) {
  static constexpr std::array<std::string_view, kEffectCount> ids = {
      "mirror", "virtual_background.blur", "virtual_background.remove",
      "virtual_background.replace", "auto_frame", "eye_contact",
      "video_noise_removal", "virtual_key_light", "vignette"};
  return slot < ids.size() ? ids[slot] : std::string_view{};
}

std::string_view LiveEffectBackendAttribution::BackendId(std::uint8_t code) {
  static constexpr std::array<std::string_view, 4> ids = {
      "open_vulkan", "open_cuda", "maxine", "cpu"};
  return code > 0 && code <= ids.size() ? ids[code - 1]
                                        : std::string_view{};
}

void LiveEffectBackendAttribution::RememberOrder(
    const std::vector<std::string> &canonical_order) {
  active_order_.fill(0);
  active_order_size_ = 0;
  std::array<bool, kEffectCount> present{};
  const auto append = [&](std::size_t slot) {
    if (slot >= kEffectCount || present[slot])
      return;
    present[slot] = true;
    active_order_[active_order_size_++] = static_cast<std::uint8_t>(slot);
  };
  for (const auto &effect : canonical_order)
    append(EffectSlot(effect));
  for (std::size_t slot = 0; slot < kEffectCount; ++slot)
    append(slot);
}

void LiveEffectBackendAttribution::SerializeActive() {
  active_backends_.clear();
  for (std::size_t index = 0; index < active_order_size_; ++index) {
    const auto slot = active_order_[index];
    if (slot >= kEffectCount || active_[slot] == 0)
      continue;
    if (!active_backends_.empty())
      active_backends_.push_back(',');
    active_backends_.append(EffectId(slot));
    active_backends_.push_back(':');
    active_backends_.append(BackendId(active_[slot]));
  }
}

void CpuPrimitiveTailActivityTracker::BeginFrame(
    std::uint64_t capture_sequence) noexcept {
  pending_capture_sequence_ = capture_sequence;
  pending_elapsed_ns_ = 0;
  pending_attempted_ = false;
  pending_succeeded_ = false;
}

void CpuPrimitiveTailActivityTracker::RecordMirrorResult(
    bool succeeded, std::uint64_t elapsed_ns) noexcept {
  if (pending_capture_sequence_ == 0 || pending_attempted_)
    return;
  pending_attempted_ = true;
  pending_succeeded_ = succeeded;
  pending_elapsed_ns_ = elapsed_ns;
  ++attempts_;
  if (succeeded)
    ++successes_;
  else
    ++fail_open_frames_;
}

void CpuPrimitiveTailActivityTracker::DiscardFrame() noexcept {
  pending_capture_sequence_ = 0;
  pending_elapsed_ns_ = 0;
  pending_attempted_ = false;
  pending_succeeded_ = false;
}

bool CpuPrimitiveTailActivityTracker::CommitSuccessfulWriter(
    std::uint64_t output_commit_sequence) noexcept {
  if (pending_capture_sequence_ == 0 || output_commit_sequence == 0 ||
      (committed_.writer_commit.valid &&
       output_commit_sequence <=
           committed_.writer_commit.output_commit_sequence)) {
    DiscardFrame();
    return false;
  }
  CpuPrimitiveTailActivityRecord next;
  next.valid = true;
  next.active = pending_attempted_ && pending_succeeded_;
  next.status = !pending_attempted_
                    ? hal::StatusId::passthrough
                    : (pending_succeeded_ ? hal::StatusId::executed
                                          : hal::StatusId::degraded);
  next.reason =
      pending_attempted_ && !pending_succeeded_
          ? hal::ReasonId::optional_stage_failed_open
          : hal::ReasonId::none;
  next.writer_commit = {
      true, pending_capture_sequence_, output_commit_sequence};
  next.elapsed_ns = pending_elapsed_ns_;
  next.attempts = attempts_;
  next.successes = successes_;
  next.fail_open_frames = fail_open_frames_;
  committed_ = next;
  DiscardFrame();
  return true;
}

int OptionalEffectBreaker::CooldownFramesForFailureCount(int failure_count) {
  if (failure_count <= 0)
    return 0;
  int cooldown = kInitialCooldownFrames;
  for (int index = 1;
       index < failure_count && cooldown < kMaxCooldownFrames; ++index) {
    cooldown = std::min(kMaxCooldownFrames, cooldown * 2);
  }
  return cooldown;
}

bool OptionalEffectBreaker::AllowsAttempt(std::uint64_t frame_index) const {
  return !active_ || frame_index >= retry_frame_index_;
}

void OptionalEffectBreaker::Reset() { *this = OptionalEffectBreaker{}; }

void OptionalEffectBreaker::OnFailure(std::string_view effect,
                                      std::string_view backend,
                                      std::string failure_reason,
                                      std::uint64_t frame_index, int order) {
  active_ = true;
  effect_id_ = effect;
  backend_ = backend;
  reason_ = std::move(failure_reason);
  ++failure_count_;
  cooldown_frames_ = CooldownFramesForFailureCount(failure_count_);
  retry_frame_index_ =
      frame_index + static_cast<std::uint64_t>(cooldown_frames_);
  trip_order_ = order;
  retry_ready_published_ = false;
}

bool OptionalEffectBreaker::OnSuccess() {
  const bool changed = active_;
  Reset();
  return changed;
}

bool OptionalEffectBreaker::MarkRetryReadyIfDue(std::uint64_t frame_index) {
  if (!active_ || retry_ready_published_ || !AllowsAttempt(frame_index))
    return false;
  retry_ready_published_ = true;
  return true;
}

CameraPipelineStatus::DegradedEffect
OptionalEffectBreaker::ToStatus(std::uint64_t frame_index) const {
  CameraPipelineStatus::DegradedEffect status;
  if (!active_)
    return status;
  status.active = true;
  status.effect_id = effect_id_;
  status.backend = backend_;
  status.reason = reason_;
  status.failure_count = failure_count_;
  status.state = AllowsAttempt(frame_index) ? "retry_ready" : "cooling_down";
  if (!AllowsAttempt(frame_index)) {
    const auto remaining = retry_frame_index_ - frame_index;
    status.cooldown_frames =
        remaining > static_cast<std::uint64_t>(
                        std::numeric_limits<int>::max())
            ? std::numeric_limits<int>::max()
            : static_cast<int>(remaining);
  }
  return status;
}

CameraPipeline::CameraPipeline() {
  CameraPipelineHalStatusProjectionInput projection;
  projection.effects = &effects_;
  projection.lifecycle = VideoRuntimeLifecycle::disabled;
  projection.status_generation = ++video_hal_status_generation_;
  projection.configuration_generation = 1;
  projection.geometry_generation = 1;
  (void)BuildCameraPipelineHalStatusSnapshot(projection, &video_hal_status_);
}

CameraPipeline::~CameraPipeline() { Stop(); }

void CameraPipeline::SetMirrorEnabled(bool enabled) {
  std::lock_guard<std::mutex> update_lock(effects_update_mu_);
  std::uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> lock(effects_mu_);
    if (effects_.mirror == enabled)
      return;
    effects_generation_published_.store(0, std::memory_order_release);
    effects_.mirror = enabled;
    generation = ++effects_generation_;
  }
  effects_generation_published_.store(generation, std::memory_order_release);
}

void CameraPipeline::SetEffects(
    const effects::BroadcastCameraEffects &effects) {
  std::lock_guard<std::mutex> update_lock(effects_update_mu_);
  std::uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> lock(effects_mu_);
    ++effects_set_requests_;
    if (effects_ == effects) {
      ++effects_ignored_updates_;
      return;
    }
    effects_generation_published_.store(0, std::memory_order_release);
    effects_ = effects;
    replace_background_source_ = {};
    generation = ++effects_generation_;
    ++effects_applied_updates_;
  }
  effects_generation_published_.store(generation, std::memory_order_release);
}

void CameraPipeline::RefreshEffectsResources() {
  std::lock_guard<std::mutex> update_lock(effects_update_mu_);
  std::uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> lock(effects_mu_);
    effects_generation_published_.store(0, std::memory_order_release);
    replace_background_source_ = {};
    generation = ++effects_generation_;
    ++effects_resource_refreshes_;
  }
  effects_generation_published_.store(generation, std::memory_order_release);
}

CameraPipelineStatus CameraPipeline::Status() const {
  std::lock_guard<std::mutex> lock(mu_);
  CameraPipelineStatus status;
  status.running = running_;
  status.starting = starting_;
  status.input_device = input_device_;
  status.output_device = output_device_;
  status.capture = running_ || starting_ ? capture_ : CaptureFormat{};
  status.output =
      running_ || starting_ ? output_
                            : (writer_.IsOpen() ? writer_.Actual()
                                                : ActualFormat{});
  status.capture_fallback_state = capture_fallback_state_;
  status.capture_fallback_reason = capture_fallback_reason_;
  status.scaling_backend_active =
      running_ ? scaling_backend_active_ : std::string{};
  status.scaling_from = running_ ? scaling_from_ : CaptureFormat{};
  status.scaling_to = running_ ? scaling_to_ : ActualFormat{};
  status.compute_backend_preference = compute_backend_preference_;
  status.compute_backend_resolved = compute_backend_resolved_;
  status.compute_backend_active =
      running_ ? compute_backend_active_ : "cpu";
  status.compute_backend_fallback_reason =
      compute_backend_fallback_reason_;
  status.compute_backend_degraded_reason =
      compute_backend_degraded_reason_;
  status.frame_index = frame_index_;
  status.ms_per_frame = running_ ? ms_per_frame_
                                 : CameraPipelineStatus::MsPerFrame{};
  status.fps_actual = running_ ? fps_actual_ : 0.0;
  status.perf_sample_frames = running_ ? perf_sample_frames_ : 0;
  status.debug = running_ ? debug_ : CameraPipelineStatus::Debug{};
  status.open_cuda_transfers =
      running_ ? open_cuda_transfers_
               : CameraPipelineStatus::OpenCudaTransfers{};
  status.open_vulkan_transfers =
      running_ ? open_vulkan_transfers_
               : CameraPipelineStatus::OpenVulkanTransfers{};
  status.maxine_transfers =
      running_ ? maxine_transfers_
               : CameraPipelineStatus::MaxineTransfers{};
  status.cpu_primitive_tails =
      running_ ? cpu_primitive_tails_
               : CameraPipelineStatus::CpuPrimitiveTails{};
  status.cpu_primitive_tail_activity =
      running_ ? cpu_primitive_tail_activity_
               : CpuPrimitiveTailActivityRecord{};
  status.production_backend = production_backend_;
  status.video_hal_status = video_hal_status_;
  status.effects_backends =
      running_ && !starting_ ? effects_backends_ : std::string{};
  status.effects_note = effects_note_;
  status.degraded_effect = degraded_effect_;
  status.last_error = last_error_;
  return status;
}

bool CameraPipeline::Start(const CameraPipelineConfig &config,
                           std::string *error) {
  const bool width_set = config.width > 0;
  const bool height_set = config.height > 0;
  if ((config.capture_mode == CaptureMode::requested &&
       (!width_set || !height_set)) ||
      (config.capture_mode == CaptureMode::auto_best &&
       width_set != height_set)) {
    if (error != nullptr)
      *error = "Invalid width/height.";
    return false;
  }
  if (config.fps <= 0 || config.fps > 240) {
    if (error != nullptr)
      *error = "Invalid fps (1..240).";
    return false;
  }

  std::unique_lock<std::mutex> update_lock(effects_update_mu_);
  std::unique_lock<std::mutex> lock(mu_);
  if (running_ || starting_) {
    if (error != nullptr)
      *error = "Camera pipeline already running.";
    return false;
  }
  if (th_.joinable()) {
    lock.unlock();
    Stop();
    lock.lock();
  }
  stop_.store(false);
  {
    std::lock_guard<std::mutex> effects_lock(effects_mu_);
    effects_ = config.effects;
    replace_background_source_ = {};
    ++effects_generation_;
    effects_generation_published_.store(effects_generation_,
                                        std::memory_order_release);
  }
  last_error_.clear();
  input_device_.clear();
  output_device_.clear();
  capture_ = {};
  output_ = {};
  capture_fallback_state_ = "none";
  capture_fallback_reason_.clear();
  scaling_backend_active_.clear();
  scaling_from_ = {};
  scaling_to_ = {};
  compute_backend_preference_ =
      ComputeBackendPreferenceToString(config.compute_backend);
  compute_backend_resolved_ = "cpu";
  compute_backend_active_ = "cpu";
  compute_backend_fallback_reason_.clear();
  compute_backend_degraded_reason_.clear();
  frame_index_ = 0;
  ms_per_frame_ = {};
  fps_actual_ = 0.0;
  perf_sample_frames_ = 0;
  debug_ = {};
  open_cuda_transfers_ = {};
  open_vulkan_transfers_ = {};
  maxine_transfers_ = {};
  cpu_primitive_tails_ = {};
  cpu_primitive_tail_activity_ = {};
  production_backend_ = {};
  effects_backends_.clear();
  effects_note_.clear();
  degraded_effect_ = {};
  starting_ = true;
  running_ = false;
  start_notified_ = false;
  th_ = std::thread(&CameraPipeline::ThreadMain, this, config);
  cv_.wait(lock, [&] { return start_notified_; });
  starting_ = false;
  if (!running_) {
    const auto failure = last_error_.empty()
                             ? "Failed to start camera pipeline."
                             : last_error_;
    lock.unlock();
    if (th_.joinable())
      th_.join();
    if (error != nullptr)
      *error = failure;
    return false;
  }
  return true;
}

bool CameraPipeline::OpenOutputLocked(
    const std::string &device, int width, int height, int fps,
    PixelFormat format, bool strict_fps, bool *opened_or_renegotiated,
    std::string *error) {
  if (opened_or_renegotiated != nullptr)
    *opened_or_renegotiated = false;
  if (writer_.IsOpen() && writer_device_ == device) {
    std::string ignored;
    (void)writer_.RefreshActual(&ignored);
    if (ActualMatches(writer_.Actual(), width, height, fps, format,
                      strict_fps)) {
      output_ = writer_.Actual();
      output_device_ = device;
      return true;
    }
    if (writer_.Renegotiate(device, width, height, fps, format, &ignored) &&
        ActualMatches(writer_.Actual(), width, height, fps, format,
                      strict_fps)) {
      if (opened_or_renegotiated != nullptr)
        *opened_or_renegotiated = true;
      output_ = writer_.Actual();
      output_device_ = device;
      return true;
    }
    writer_.Close();
    writer_device_.clear();
  } else if (writer_.IsOpen()) {
    writer_.Close();
    writer_device_.clear();
  }
  if (!writer_.Open(device, width, height, fps, format, error))
    return false;
  if (!ActualMatches(writer_.Actual(), width, height, fps, format,
                     strict_fps)) {
    if (error != nullptr)
      *error = ActualMismatch(writer_.Actual(), width, height, fps, format,
                              strict_fps);
    writer_.Close();
    return false;
  }
  writer_device_ = device;
  output_ = writer_.Actual();
  output_device_ = device;
  if (opened_or_renegotiated != nullptr)
    *opened_or_renegotiated = true;
  return true;
}

bool CameraPipeline::EnsureOutputOpen(const CameraPipelineConfig &config,
                                      std::string *error) {
  if (config.fps <= 0 || config.width <= 0 || config.height <= 0) {
    if (error != nullptr)
      *error = "Invalid output format.";
    return false;
  }
  std::lock_guard<std::mutex> lock(mu_);
  std::string device = config.output_device;
  if (device.empty())
    device = DefaultLoopback(error);
  if (device.empty())
    return false;
  bool changed = false;
  if (!OpenOutputLocked(device, config.width, config.height, config.fps,
                        config.output_format, false, &changed, error)) {
    return false;
  }
  const auto actual = writer_.Actual();
  const std::size_t bytes =
      actual.size_image != 0
          ? actual.size_image
          : actual.bytes_per_line * static_cast<std::size_t>(actual.height);
  if (bytes != 0 &&
      (keepalive_frame_.size() != bytes ||
       keepalive_format_.width != actual.width ||
       keepalive_format_.height != actual.height ||
       keepalive_format_.format != actual.format)) {
    keepalive_frame_.assign(bytes, 0);
    keepalive_format_ = actual;
  }
  const auto now = Clock::now();
  if (!keepalive_frame_.empty() &&
      (last_keepalive_frame_at_.time_since_epoch().count() == 0 ||
       now - last_keepalive_frame_at_ >= std::chrono::seconds(1))) {
    const auto result = writer_.WriteFrameDetailed(
        keepalive_frame_.data(), keepalive_frame_.size());
    if (result.FrameCommitted())
      last_keepalive_frame_at_ = now;
  }
  return true;
}

void CameraPipeline::CloseOutput() {
  std::lock_guard<std::mutex> lock(mu_);
  if (running_ || starting_)
    return;
  writer_.Close();
  writer_device_.clear();
  output_device_.clear();
  output_ = {};
}

void CameraPipeline::Stop() {
  stop_.store(true);
  cv_.notify_all();
  std::thread joining;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (th_.joinable())
      joining = std::move(th_);
    else {
      running_ = false;
      starting_ = false;
      effects_backends_.clear();
      return;
    }
  }
  joining.join();
  std::lock_guard<std::mutex> lock(mu_);
  running_ = false;
  starting_ = false;
  effects_backends_.clear();
  capture_ = {};
  output_ = {};
  scaling_backend_active_.clear();
  scaling_from_ = {};
  scaling_to_ = {};
}

void CameraPipeline::ThreadMain(CameraPipelineConfig config) {
  V4l2Capture capture;
  std::string input_device = config.input_device;
  const bool auto_best = config.capture_mode == CaptureMode::auto_best;
  if (!input_device.empty()) {
    std::string capture_error;
    const bool opened =
        auto_best
            ? capture.OpenBest(input_device, config.fps, config.prefer_mjpeg,
                               &capture_error)
            : capture.Open(input_device, config.width, config.height,
                           config.fps, CapturePixelFormat::yuyv,
                           config.prefer_mjpeg, &capture_error);
    if (!opened) {
      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = "Failed to open capture device " + input_device + ": " +
                    capture_error;
      start_notified_ = true;
      cv_.notify_all();
      return;
    }
  } else {
    std::ostringstream attempts;
    for (const auto &candidate : CandidateCameras()) {
      std::string capture_error;
      const bool opened =
          auto_best
              ? capture.OpenBest(candidate.dev_node, config.fps,
                                 config.prefer_mjpeg, &capture_error)
              : capture.Open(candidate.dev_node, config.width, config.height,
                             config.fps, CapturePixelFormat::yuyv,
                             config.prefer_mjpeg, &capture_error);
      if (opened) {
        input_device = candidate.dev_node;
        break;
      }
      attempts << candidate.dev_node << ": " << capture_error << "\n";
    }
    if (!capture.IsOpen()) {
      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = "Failed to auto-select a camera.\n" + attempts.str();
      start_notified_ = true;
      cv_.notify_all();
      return;
    }
  }

  std::string output_device = config.output_device;
  if (output_device.empty())
    output_device = DefaultLoopback(nullptr);
  if (output_device.empty()) {
    std::lock_guard<std::mutex> lock(mu_);
    last_error_ = "No writable output loopback found.";
    start_notified_ = true;
    cv_.notify_all();
    return;
  }

  const auto capture_format = capture.Actual();
  int output_width = config.width;
  int output_height = config.height;
  if (auto_best && (output_width <= 0 || output_height <= 0)) {
    output_width = capture_format.width;
    output_height = capture_format.height;
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    bool changed = false;
    std::string output_error;
    if (!OpenOutputLocked(output_device, output_width, output_height,
                          config.fps, config.output_format, true, &changed,
                          &output_error)) {
      last_error_ = output_error;
      start_notified_ = true;
      cv_.notify_all();
      return;
    }
  }
  const auto output_format = writer_.Actual();

  auto setup_source =
      config.backend_setup_source_factory != nullptr
          ? config.backend_setup_source_factory()
          : CreateStaticProductionVideoBackendSetupSource();
  ProductionVideoBackendManager backend_manager(std::move(setup_source));
  ManagerWaitContext wait_context{&stop_};
  const auto &cpu = GetCpuPrimitiveTable();

  effects::BroadcastCameraEffects applied_effects;
  std::uint64_t applied_generation = 0;
  {
    std::lock_guard<std::mutex> lock(effects_mu_);
    applied_effects = effects_;
    applied_generation = effects_generation_;
  }
  std::uint64_t attempted_generation = applied_generation;
  auto applied_plan = effects::BuildBroadcastEffectsPlan(applied_effects);
  std::uint64_t geometry_generation = 1;
  std::uint64_t plan_id = 1;
  auto setup_request = BuildSetupRequest(
      applied_effects, config.compute_backend, capture_format.width,
      capture_format.height, output_format.width, output_format.height,
      output_format.format, applied_generation, geometry_generation, plan_id,
      &wait_context, cpu);
  ProductionVideoBackendSetupEvidence setup_evidence;
  hal::ReasonId setup_reason = hal::ReasonId::none;
  const auto setup_status =
      backend_manager.Setup(setup_request, &setup_evidence, &setup_reason);
  auto backend_snapshot = backend_manager.Snapshot();
  std::string active_backend_name(
      BackendName(backend_snapshot.readiness.selected_backend));
  bool semantic_effects =
      HasResidentSemanticEffects(applied_effects, config.compute_backend);
  bool manager_ready =
      setup_status == hal::StatusId::ready ||
      setup_status == hal::StatusId::degraded;
  if (semantic_effects && !manager_ready) {
    // Product effects are optional and fail open, but strict requests never
    // execute another backend. The exact blocker remains in the manager
    // snapshot and live attribution remains empty.
    manager_ready = false;
  }
  if (!CheckOutputResizeAllowed(
          capture_format.width, capture_format.height, output_format.width,
          output_format.height, manager_ready, config.allow_cpu_resize,
          nullptr)) {
    std::lock_guard<std::mutex> lock(mu_);
    last_error_ = OutputResizeDisallowedErrorMessage(
        capture_format.width, capture_format.height, output_format.width,
        output_format.height);
    start_notified_ = true;
    cv_.notify_all();
    return;
  }

  Rgb24Frame input_rgb;
  input_rgb.ResizeTight(capture_format.width, capture_format.height);
  Rgb24Frame manager_output;
  manager_output.ResizeTight(output_format.width, output_format.height);
  Rgb24Frame cpu_scaled;
  cpu_scaled.ResizeTight(output_format.width, output_format.height);
  Rgb24Frame mirror_output;
  mirror_output.ResizeTight(output_format.width, output_format.height);

  CpuRgb24BilinearResizePlan resize_plan;
  std::vector<std::uint8_t> resize_scratch;
  const bool scaling_needed =
      capture_format.width != output_format.width ||
      capture_format.height != output_format.height;
  bool cpu_resize_prepared = false;
  if (scaling_needed && config.allow_cpu_resize) {
    const auto bytes = cpu.resize_rgb24_bilinear_plan_scratch_bytes(
        capture_format.width, capture_format.height, output_format.width,
        output_format.height);
    resize_scratch.resize(bytes);
    cpu_resize_prepared =
        cpu.prepare_resize_rgb24_bilinear(
            capture_format.width, capture_format.height, output_format.width,
            output_format.height, resize_scratch.data(), resize_scratch.size(),
            &resize_plan) == CpuPrimitiveResult::success;
  }

  const std::size_t output_bytes =
      output_format.size_image != 0
          ? output_format.size_image
          : output_format.bytes_per_line *
                static_cast<std::size_t>(output_format.height);
  std::vector<std::uint8_t> output_buffer(output_bytes);
  std::vector<std::uint8_t> yuyv_scratch(
      cpu.rgb24_to_yuyv_scratch_bytes(output_format.width,
                                      output_format.height));
  auto raw_yuyv_plan = PrepareRawYuyvPassthroughPlan(
      capture_format, output_format, applied_effects);
  YuyvFramePathCounters yuyv_counters;

  {
    std::lock_guard<std::mutex> lock(mu_);
    running_ = true;
    input_device_ = input_device;
    output_device_ = output_device;
    capture_ = capture_format;
    output_ = output_format;
    scaling_from_ = capture_format;
    scaling_to_ = output_format;
    scaling_backend_active_ =
        scaling_needed ? (semantic_effects && manager_ready ? "gpu" : "cpu")
                       : "";
    production_backend_ = backend_snapshot;
    compute_backend_resolved_ = BackendName(
        backend_snapshot.readiness.selected_backend);
    if (compute_backend_resolved_.empty())
      compute_backend_resolved_ = "cpu";
    compute_backend_active_ = "cpu";
    if (!manager_ready && semantic_effects) {
      compute_backend_degraded_reason_ =
          "Requested effects unavailable [" +
          std::to_string(static_cast<std::uint32_t>(setup_reason)) + "].";
      effects_note_ = compute_backend_degraded_reason_;
    }
    start_notified_ = true;
    cv_.notify_all();
  }

  ExponentialAverage capture_average;
  ExponentialAverage effects_average;
  ExponentialAverage scale_average;
  ExponentialAverage write_average;
  ExponentialAverage period_average;
  EffectsFramePreparationCounters preparation_counters;
  LiveEffectBackendAttribution attribution;
  CameraPipelineStatus::CpuPrimitiveTails cpu_primitive_tails;
  CpuPrimitiveTailActivityTracker cpu_tail_activity;
  std::uint64_t last_sequence = 0;
  int dropped_frames = 0;
  int frame_index = 0;
  Clock::time_point previous_frame_end{};
  bool have_previous_frame = false;

  const auto publish_snapshot = [&](VideoRuntimeLifecycle lifecycle) {
    backend_snapshot = backend_manager.Snapshot();
    CameraPipelineHalStatusProjectionInput projection;
    projection.effects = &applied_effects;
    projection.compute_backend = config.compute_backend;
    projection.lifecycle = lifecycle;
    projection.status_generation = ++video_hal_status_generation_;
    if (projection.status_generation == 0)
      projection.status_generation = ++video_hal_status_generation_;
    projection.configuration_generation =
        applied_generation == 0 ? 1 : applied_generation;
    projection.geometry_generation = geometry_generation;
    projection.backend = backend_snapshot;
    MultiIslandVideoRuntimeStatusSnapshot typed_status;
    const auto validation =
        BuildCameraPipelineHalStatusSnapshot(projection, &typed_status);
    CameraPipelineStatus::OpenCudaTransfers cuda;
    CameraPipelineStatus::OpenVulkanTransfers vulkan;
    CameraPipelineStatus::MaxineTransfers maxine;
    ProjectLegacyTransfers(backend_snapshot, &cuda, &vulkan, &maxine);
    std::lock_guard<std::mutex> lock(mu_);
    production_backend_ = backend_snapshot;
    open_cuda_transfers_ = cuda;
    open_vulkan_transfers_ = vulkan;
    maxine_transfers_ = maxine;
    cpu_primitive_tails_ = cpu_primitive_tails;
    cpu_primitive_tail_activity_ = cpu_tail_activity.snapshot();
    video_hal_status_ =
        validation.code == MultiIslandStatusValidationCode::valid
            ? typed_status
            : MultiIslandVideoRuntimeStatusSnapshot{};
  };
  publish_snapshot(VideoRuntimeLifecycle::running);

  while (!stop_.load()) {
    const auto published_generation =
        effects_generation_published_.load(std::memory_order_acquire);
    if (preparation_counters.ObservePublishedGeneration(
            published_generation, attempted_generation)) {
      effects::BroadcastCameraEffects next_effects;
      std::uint64_t next_generation = 0;
      {
        std::lock_guard<std::mutex> lock(effects_mu_);
        next_effects = effects_;
        next_generation = effects_generation_;
      }
      const bool rebuild = EffectsPlanRequiresRebuild(
          applied_effects, applied_generation, next_effects, next_generation);
      preparation_counters.RecordConfigSnapshot(rebuild);
      if (rebuild) {
        attempted_generation = next_generation;
        ++plan_id;
        if (plan_id == 0)
          ++plan_id;
        auto next_request = BuildSetupRequest(
            next_effects, config.compute_backend, capture_format.width,
            capture_format.height, output_format.width, output_format.height,
            output_format.format, next_generation, geometry_generation,
            plan_id, &wait_context, cpu);
        ProductionVideoBackendSetupEvidence next_evidence;
        hal::ReasonId next_reason = hal::ReasonId::none;
        const auto next_status =
            backend_manager.Setup(next_request, &next_evidence, &next_reason);
        if (next_status == hal::StatusId::ready ||
            next_status == hal::StatusId::degraded) {
          setup_request = std::move(next_request);
          applied_effects = std::move(next_effects);
          applied_generation = next_generation;
          applied_plan =
              effects::BuildBroadcastEffectsPlan(applied_effects);
          semantic_effects =
              HasResidentSemanticEffects(applied_effects,
                                         config.compute_backend);
          raw_yuyv_plan = PrepareRawYuyvPassthroughPlan(
              capture_format, output_format, applied_effects);
          manager_ready =
              next_status == hal::StatusId::ready ||
              next_status == hal::StatusId::degraded;
          attribution.Clear();
          publish_snapshot(VideoRuntimeLifecycle::running);
          active_backend_name = std::string(
              BackendName(backend_snapshot.readiness.selected_backend));
        }
      }
    }

    CapturedFrameView frame;
    std::string capture_error;
    if (!capture.AcquireFrame(&frame, 1000, &capture_error)) {
      if (stop_.load())
        break;
      if (IsRecoverableCaptureAcquireFailure(capture_error))
        continue;
      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = "Capture failed: " + capture_error;
      break;
    }
    int dropped = 0;
    for (;;) {
      CapturedFrameView newer;
      const auto result =
          capture.AcquireFrameDetailed(&newer, 0, &capture_error);
      if (result == CaptureAcquireResult::no_frame)
        break;
      if (result == CaptureAcquireResult::failure) {
        std::string ignored;
        (void)capture.ReleaseFrame(frame, &ignored);
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = "Capture drain failed: " + capture_error;
        stop_.store(true);
        break;
      }
      std::string ignored;
      (void)capture.ReleaseFrame(frame, &ignored);
      frame = newer;
      ++dropped;
    }
    if (stop_.load())
      break;
    dropped_frames += dropped;
    last_sequence = frame.sequence;
    cpu_tail_activity.BeginFrame(frame.sequence);
    const auto capture_start = Clock::now();
    if (raw_yuyv_plan.eligible &&
        effects_generation_published_.load(std::memory_order_acquire) ==
            applied_generation) {
      const auto copied =
          CopyRawYuyvFrame(raw_yuyv_plan, frame.data, frame.bytes,
                           output_buffer.data(), output_buffer.size(),
                           &yuyv_counters);
      std::string ignored;
      (void)capture.ReleaseFrame(frame, &ignored);
      if (copied != RawYuyvCopyResult::ok) {
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = "Raw YUYV passthrough rejected the frame layout.";
        break;
      }
      capture_average.Add(ToMs(Clock::now() - capture_start));
      const auto write_start = Clock::now();
      const auto write_result = writer_.WriteFrameDetailed(
          output_buffer.data(), output_buffer.size(), &stop_);
      if (!write_result.FrameCommitted()) {
        cpu_tail_activity.DiscardFrame();
        if (write_result.status == FrameWriteStatus::would_block_dropped)
          continue;
        if (write_result.status == FrameWriteStatus::stopped)
          break;
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = "Raw passthrough output write failed.";
        break;
      }
      (void)cpu_tail_activity.CommitSuccessfulWriter(
          writer_.WriteCounters().frames_written);
      write_average.Add(ToMs(Clock::now() - write_start));
      const auto frame_end = Clock::now();
      if (have_previous_frame) {
        const auto elapsed = ToMs(frame_end - previous_frame_end);
        if (elapsed > 0.0)
          period_average.Add(elapsed);
      }
      previous_frame_end = frame_end;
      have_previous_frame = true;
      ++frame_index;
      if (frame_index == 1 || frame_index % 10 == 0) {
        publish_snapshot(VideoRuntimeLifecycle::running);
        std::lock_guard<std::mutex> lock(mu_);
        frame_index_ = frame_index;
        ms_per_frame_.capture = capture_average.Value();
        ms_per_frame_.effects = 0.0;
        ms_per_frame_.scale = 0.0;
        ms_per_frame_.write = write_average.Value();
        fps_actual_ = period_average.Value() > 0.0
                          ? 1000.0 / period_average.Value()
                          : 0.0;
        perf_sample_frames_ = frame_index;
        debug_.capture_sequence = last_sequence;
        debug_.dropped_capture_frames = dropped_frames;
        debug_.effects_preparation = preparation_counters;
        debug_.raw_yuyv_passthrough_frames =
            yuyv_counters.raw_passthrough_frames;
        debug_.raw_yuyv_passthrough_bytes =
            yuyv_counters.raw_passthrough_bytes;
      }
      continue;
    }
    bool converted = false;
    if (capture_format.format == CapturePixelFormat::yuyv) {
      converted =
          cpu.yuyv_to_rgb24(
              frame.data, capture_format.width, capture_format.height,
              capture_format.bytes_per_line, input_rgb.data(),
              input_rgb.stride_bytes) == CpuPrimitiveResult::success;
      if (converted)
        ++yuyv_counters.capture_to_rgb_calls;
    } else if (capture_format.format == CapturePixelFormat::mjpeg) {
      int decoded_width = 0;
      int decoded_height = 0;
      converted = DecodeMjpegToRgb24(
                      frame.data, frame.bytes, input_rgb, decoded_width,
                      decoded_height, &capture_error) &&
                  decoded_width == capture_format.width &&
                  decoded_height == capture_format.height;
    }
    std::string ignored;
    (void)capture.ReleaseFrame(frame, &ignored);
    if (!converted) {
      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = "Capture conversion failed.";
      break;
    }
    capture_average.Add(ToMs(Clock::now() - capture_start));

    const auto effects_start = Clock::now();
    const std::uint8_t *rgb_output = input_rgb.data();
    std::size_t rgb_stride = input_rgb.stride_bytes;
    std::size_t rgb_bytes = input_rgb.size();
    hal::CompletionToken completion;
    bool completion_pending = false;
    VideoExecutionFrameResult execution;
    attribution.BeginFrame();
    if (semantic_effects && manager_ready) {
      hal::HostFrameBinding binding;
      binding.input_descriptor = setup_request.resident.plan.host_input;
      binding.output_descriptor = setup_request.resident.plan.host_output;
      binding.input_planes[0].read_data =
          reinterpret_cast<const std::byte *>(input_rgb.data());
      binding.input_planes[0].span_bytes = input_rgb.size();
      binding.input_planes[0].row_stride_bytes = input_rgb.stride_bytes;
      binding.output_planes[0].write_data =
          reinterpret_cast<std::byte *>(manager_output.data());
      binding.output_planes[0].span_bytes = manager_output.size();
      binding.output_planes[0].row_stride_bytes =
          manager_output.stride_bytes;
      binding.input_plane_count = 1;
      binding.output_plane_count = 1;
      binding.capture_sequence = frame.sequence;
      binding.ownership = hal::OwnershipModel::borrowed_until_completion;
      binding.lifetime = hal::ResourceLifetime::frame_borrowed;
      execution = backend_manager.Execute(binding);
      completion = execution.completion;
      completion_pending = completion.token_id != 0;
      if ((execution.status == hal::StatusId::executed ||
           execution.status == hal::StatusId::degraded) &&
          execution.output_ready && execution.output.output_ready &&
          execution.output.plane_count == 1 &&
          execution.output.planes[0].write_data != nullptr) {
        // Completion makes the caller-owned borrowed output buffer readable.
        // Host output resolution carries that buffer in write_data; requiring
        // a redundant read_data alias rejects otherwise valid adapters.
        rgb_output = reinterpret_cast<const std::uint8_t *>(
            execution.output.planes[0].write_data);
        rgb_stride = execution.output.planes[0].row_stride_bytes;
        rgb_bytes = execution.output.planes[0].span_bytes;
      } else {
        if (completion_pending) {
          hal::ReasonId reason = hal::ReasonId::none;
          (void)backend_manager.CommitOutput(completion, false, &reason);
        }
        completion_pending = false;
        attribution.DiscardFrame();
      }
    }
    const double resident_effect_ms = ToMs(Clock::now() - effects_start);

    const auto scale_start = Clock::now();
    if (scaling_needed &&
        rgb_output == input_rgb.data()) {
      if (!cpu_resize_prepared ||
          cpu.apply_resize_rgb24_bilinear(
              &resize_plan, input_rgb.data(), input_rgb.stride_bytes,
              cpu_scaled.data(), cpu_scaled.stride_bytes) !=
              CpuPrimitiveResult::success) {
        if (completion_pending) {
          hal::ReasonId reason = hal::ReasonId::none;
          (void)backend_manager.CommitOutput(completion, false, &reason);
        }
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = "Prepared CPU resize failed.";
        break;
      }
      rgb_output = cpu_scaled.data();
      rgb_stride = cpu_scaled.stride_bytes;
      rgb_bytes = cpu_scaled.size();
    }
    scale_average.Add(ToMs(Clock::now() - scale_start));

    double primitive_tail_ms = 0.0;
    const bool cpu_mirror =
        applied_effects.mirror &&
        config.compute_backend != ComputeBackendPreference::vulkan;
    if (cpu_mirror) {
      const auto mirror_start = Clock::now();
      ++cpu_primitive_tails.mirror_attempts;
      const std::size_t tight_stride =
          static_cast<std::size_t>(output_format.width) * 3u;
      const std::size_t minimum_span =
          output_format.height <= 0
              ? 0
              : rgb_stride *
                        static_cast<std::size_t>(output_format.height - 1) +
                    tight_stride;
      bool copied = rgb_output != nullptr && rgb_stride >= tight_stride &&
                    rgb_bytes >= minimum_span;
      if (copied) {
        for (int row = 0; row < output_format.height; ++row) {
          std::memcpy(
              mirror_output.data() +
                  static_cast<std::size_t>(row) *
                      mirror_output.stride_bytes,
              rgb_output + static_cast<std::size_t>(row) * rgb_stride,
              tight_stride);
        }
        copied = cpu.mirror_rgb24_in_place(
                     mirror_output.data(), output_format.width,
                     output_format.height, mirror_output.stride_bytes) ==
                 CpuPrimitiveResult::success;
      }
      const auto mirror_elapsed = Clock::now() - mirror_start;
      const auto mirror_elapsed_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              mirror_elapsed)
              .count());
      primitive_tail_ms = ToMs(mirror_elapsed);
      cpu_primitive_tails.mirror_elapsed_ns += mirror_elapsed_ns;
      if (copied) {
        ++cpu_primitive_tails.mirror_successes;
        rgb_output = mirror_output.data();
        rgb_stride = mirror_output.stride_bytes;
        rgb_bytes = mirror_output.size();
        attribution.MarkEffectSucceeded(effects::contract::kEffectIdMirror,
                                        "cpu");
      } else {
        // Mirror is optional. Preserve the last valid resident/host frame and
        // make the explicit finalizer failure observable without corrupting
        // or suppressing unrelated stages.
        ++cpu_primitive_tails.mirror_fail_open_frames;
      }
      cpu_tail_activity.RecordMirrorResult(copied, mirror_elapsed_ns);
    }
    effects_average.Add(resident_effect_ms + primitive_tail_ms);

    const auto write_start = Clock::now();
    const std::uint8_t *write_data = nullptr;
    std::size_t write_bytes = 0;
    const std::size_t tight_stride =
        static_cast<std::size_t>(output_format.width) * 3u;
    if (output_format.format == PixelFormat::rgb24 &&
        output_format.bytes_per_line == tight_stride &&
        output_format.size_image == rgb_bytes &&
        rgb_stride == tight_stride) {
      write_data = rgb_output;
      write_bytes = rgb_bytes;
    } else if (output_format.format == PixelFormat::rgb24) {
      for (int row = 0; row < output_format.height; ++row) {
        std::memcpy(output_buffer.data() +
                        static_cast<std::size_t>(row) *
                            output_format.bytes_per_line,
                    rgb_output + static_cast<std::size_t>(row) * rgb_stride,
                    tight_stride);
      }
      write_data = output_buffer.data();
      write_bytes = output_buffer.size();
    } else {
      if (cpu.rgb24_to_yuyv(
              rgb_output, output_format.width, output_format.height,
              rgb_stride, output_buffer.data(), output_format.bytes_per_line,
              yuyv_scratch.data(), yuyv_scratch.size()) !=
          CpuPrimitiveResult::success) {
        if (completion_pending) {
          hal::ReasonId reason = hal::ReasonId::none;
          (void)backend_manager.CommitOutput(completion, false, &reason);
        }
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = "Output conversion failed.";
        break;
      }
      ++yuyv_counters.output_from_rgb_calls;
      write_data = output_buffer.data();
      write_bytes = output_buffer.size();
    }

    const auto write_result =
        writer_.WriteFrameDetailed(write_data, write_bytes, &stop_);
    const bool written = write_result.FrameCommitted();
    bool manager_output_committed = false;
    if (completion_pending) {
      hal::ReasonId commit_reason = hal::ReasonId::none;
      const auto commit_status =
          backend_manager.CommitOutput(completion, written, &commit_reason);
      manager_output_committed =
          written && commit_status == hal::StatusId::ready;
      if (written && commit_status != hal::StatusId::ready) {
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = "Neutral backend output attribution commit failed.";
        break;
      }
    }
    if (!written) {
      attribution.DiscardFrame();
      cpu_tail_activity.DiscardFrame();
      if (write_result.status == FrameWriteStatus::would_block_dropped)
        continue;
      if (write_result.status == FrameWriteStatus::stopped)
        break;
      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = "Output write failed.";
      break;
    }
    (void)cpu_tail_activity.CommitSuccessfulWriter(
        writer_.WriteCounters().frames_written);

    backend_snapshot = backend_manager.Snapshot();
    const auto *committed =
        backend_snapshot.state ==
                ProductionVideoBackendManagerState::resident_ready
            ? &backend_snapshot.resident.latest_committed_frame
            : &backend_snapshot.compatibility.latest_committed_frame;
    const std::string_view active_backend = active_backend_name;
    if (manager_output_committed && committed->valid &&
        committed->capture_sequence == frame.sequence) {
      for (std::size_t index = 0; index < committed->stage_count; ++index) {
        const auto &stage = committed->stages[index];
        if (stage.status == hal::StageExecutionStatus::succeeded) {
          attribution.MarkEffectSucceeded(EffectId(stage.semantic),
                                          active_backend);
        }
      }
    }
    (void)attribution.CommitSuccessfulFrame(
        applied_plan.ordered_effect_ids);
    write_average.Add(ToMs(Clock::now() - write_start));

    const auto frame_end = Clock::now();
    if (have_previous_frame) {
      const auto elapsed = ToMs(frame_end - previous_frame_end);
      if (elapsed > 0.0)
        period_average.Add(elapsed);
    }
    previous_frame_end = frame_end;
    have_previous_frame = true;
    ++frame_index;
    if (frame_index == 1 || frame_index % 10 == 0) {
      publish_snapshot(VideoRuntimeLifecycle::running);
      std::lock_guard<std::mutex> lock(mu_);
      frame_index_ = frame_index;
      ms_per_frame_.capture = capture_average.Value();
      ms_per_frame_.effects = effects_average.Value();
      ms_per_frame_.scale = scale_average.Value();
      ms_per_frame_.write = write_average.Value();
      fps_actual_ =
          period_average.Value() > 0.0 ? 1000.0 / period_average.Value() : 0.0;
      perf_sample_frames_ = frame_index;
      debug_.capture_sequence = last_sequence;
      debug_.dropped_capture_frames = dropped_frames;
      debug_.effects_preparation = preparation_counters;
      debug_.yuyv_capture_to_rgb_calls =
          yuyv_counters.capture_to_rgb_calls;
      debug_.yuyv_output_from_rgb_calls =
          yuyv_counters.output_from_rgb_calls;
      debug_.raw_yuyv_passthrough_frames =
          yuyv_counters.raw_passthrough_frames;
      debug_.raw_yuyv_passthrough_bytes =
          yuyv_counters.raw_passthrough_bytes;
      debug_.output_frames_committed =
          writer_.WriteCounters().frames_written;
      effects_backends_ = attribution.active_backends();
      compute_backend_active_ =
          effects_backends_.empty() ? "cpu" : active_backend_name;
    }
  }

  backend_manager.Invalidate(hal::ReasonId::shutdown_requested);
  auto shutdown = backend_manager.Shutdown();
  for (int attempt = 0; attempt < 8 && shutdown.pending; ++attempt)
    shutdown = backend_manager.Shutdown();
  backend_snapshot = backend_manager.Snapshot();
  publish_snapshot(VideoRuntimeLifecycle::idle);
  std::lock_guard<std::mutex> lock(mu_);
  production_backend_ = backend_snapshot;
  frame_index_ = frame_index;
  running_ = false;
  effects_backends_.clear();
  if (!start_notified_) {
    start_notified_ = true;
    cv_.notify_all();
  }
}

} // namespace studiocast::video
