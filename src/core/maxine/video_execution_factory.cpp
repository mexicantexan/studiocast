#include "core/maxine/video_execution_factory.h"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

namespace studiocast::maxine {
namespace {

void SetReason(hal::ReasonId *reason, hal::ReasonId value) noexcept {
  if (reason)
    *reason = value;
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

bool IsTightHostImage(const hal::ImageDescriptor &image,
                      hal::PixelFormat format, hal::ChannelOrder order,
                      hal::Mutability mutability) noexcept {
  const std::uint64_t stride = static_cast<std::uint64_t>(image.width) * 3u;
  return hal::ValidateImageDescriptor(image) && image.width != 0 &&
         image.height != 0 && image.plane_count == 1 &&
         image.format == format && image.channel_order == order &&
         image.color_space == hal::ColorSpace::srgb &&
         image.color_range == hal::ColorRange::full &&
         image.alpha_meaning == hal::AlphaMeaning::none &&
         image.orientation == hal::Orientation::top_left &&
         image.host_coherence == hal::HostCoherence::coherent &&
         image.mutability == mutability &&
         image.aliasing == hal::AliasingPermission::forbidden &&
         !image.in_place_permitted && image.planes[0].offset_bytes == 0 &&
         image.planes[0].row_stride_bytes == stride &&
         image.planes[0].size_bytes == stride * image.height &&
         image.planes[0].width == image.width &&
         image.planes[0].height == image.height;
}

hal::SemanticEffectProviderId
ProviderFor(hal::SemanticEffectId semantic) noexcept {
  switch (semantic) {
  case hal::SemanticEffectId::virtual_background_blur:
  case hal::SemanticEffectId::virtual_background_remove:
  case hal::SemanticEffectId::virtual_background_replace:
  case hal::SemanticEffectId::virtual_key_light:
  case hal::SemanticEffectId::video_noise_removal:
    return hal::SemanticEffectProviderId::maxine_vfx;
  case hal::SemanticEffectId::eye_contact:
  case hal::SemanticEffectId::auto_frame:
    return hal::SemanticEffectProviderId::maxine_ar;
  case hal::SemanticEffectId::vignette:
    return hal::SemanticEffectProviderId::maxine_cuda_utility;
  case hal::SemanticEffectId::mirror:
  case hal::SemanticEffectId::invalid:
    return hal::SemanticEffectProviderId::invalid;
  }
  return hal::SemanticEffectProviderId::invalid;
}

bool IsQualifiedStage(hal::SemanticEffectId semantic) noexcept {
  return semantic == hal::SemanticEffectId::virtual_background_blur ||
         semantic == hal::SemanticEffectId::virtual_background_remove ||
         semantic == hal::SemanticEffectId::virtual_background_replace ||
         semantic == hal::SemanticEffectId::virtual_key_light ||
         semantic == hal::SemanticEffectId::vignette;
}

hal::ImageDescriptor ResidentImage(std::uint32_t width, std::uint32_t height,
                                   hal::Mutability mutability) noexcept {
  hal::ImageDescriptor image;
  image.width = width;
  image.height = height;
  image.plane_count = 1;
  image.format = hal::PixelFormat::bgr_u8;
  image.channel_order = hal::ChannelOrder::bgr;
  image.color_space = hal::ColorSpace::srgb;
  image.color_range = hal::ColorRange::full;
  image.alpha_meaning = hal::AlphaMeaning::none;
  image.orientation = hal::Orientation::top_left;
  image.host_coherence = hal::HostCoherence::not_host_visible;
  image.mutability = mutability;
  image.aliasing = hal::AliasingPermission::forbidden;
  image.base_alignment_bytes = 1;
  const std::uint64_t stride = static_cast<std::uint64_t>(width) * 3u;
  image.planes[0].row_stride_bytes = stride;
  image.planes[0].size_bytes = stride * height;
  image.planes[0].width = width;
  image.planes[0].height = height;
  image.planes[0].alignment_bytes = 1;
  return image;
}

bool HasExactOutputTail(const video::VideoExecutionPlanRequest &request,
                        const hal::ResolutionResult &resolution) noexcept {
  if (request.cpu_tail_count != 1 || resolution.stage_count == 0)
    return false;
  std::size_t last_resolved = resolution.stage_count;
  while (last_resolved != 0 &&
         resolution.stages[last_resolved - 1].skipped_optional) {
    --last_resolved;
  }
  if (last_resolved == 0)
    return false;
  const auto &last = resolution.stages[last_resolved - 1];
  const auto &tail = request.cpu_tails[0];
  return tail.id == hal::CpuTailId::output_format_conversion &&
         tail.placement == hal::DeclaredCpuTail::Placement::after_stage &&
         tail.after_stage_id == last.intent.stage_id &&
         tail.before_stage_id == 0 && tail.boundary_index == last_resolved;
}

bool ExactFrozenInputs(const video::VideoExecutionPlanRequest &request,
                       const hal::ResolutionResult &resolution,
                       const hal::DomainPlan &domain_plan,
                       const hal::AdapterRegistration &registration,
                       hal::ReasonId *reason) noexcept {
  if (domain_plan.island_count != 1 ||
      domain_plan.islands[0].stage_count == 0 ||
      resolution.selected_backend != hal::BackendId::maxine ||
      domain_plan.islands[0].adapter_instance_id !=
          registration.adapter_instance_id ||
      domain_plan.islands[0].resident_adapter !=
          registration.resident_adapter ||
      domain_plan.islands[0].execution_domain !=
          registration.execution_domain) {
    SetReason(reason, domain_plan.island_count == 1 &&
                              domain_plan.islands[0].execution_domain !=
                                  registration.execution_domain
                          ? hal::ReasonId::wrong_execution_domain
                          : hal::ReasonId::foreign_adapter);
    return false;
  }
  if (request.invalidation_key.generations.context !=
          registration.execution_domain.logical_context_generation ||
      request.invalidation_key.generations.configuration == 0 ||
      request.invalidation_key.generations.geometry == 0 ||
      request.invalidation_key.generations.model == 0 ||
      request.invalidation_key.generations.provider == 0 ||
      request.invalidation_key.generations.device == 0) {
    SetReason(reason, hal::ReasonId::wrong_generation);
    return false;
  }
  if (!IsTightHostImage(request.host_input, hal::PixelFormat::rgb_u8,
                        hal::ChannelOrder::rgb, hal::Mutability::read_only) ||
      !IsTightHostImage(request.host_output, hal::PixelFormat::bgr_u8,
                        hal::ChannelOrder::bgr, hal::Mutability::write_only)) {
    SetReason(reason, hal::ReasonId::malformed_descriptor);
    return false;
  }
  if (request.host_input.width != request.host_output.width ||
      request.host_input.height != request.host_output.height ||
      request.temporal.temporal || request.temporal.history_frames != 0 ||
      !HasExactOutputTail(request, resolution) ||
      domain_plan.expected_transfers.uploads != 1 ||
      domain_plan.expected_transfers.final_readbacks != 1 ||
      domain_plan.expected_transfers.intermediate_uploads != 0 ||
      domain_plan.expected_transfers.intermediate_readbacks != 0 ||
      domain_plan.expected_transfers.cpu_tail_boundaries != 1) {
    SetReason(reason,
              request.temporal.temporal || request.temporal.history_frames != 0
                  ? hal::ReasonId::capability_evidence_incomplete
                  : hal::ReasonId::malformed_contract);
    return false;
  }
  std::size_t prepared_index = 0;
  for (std::size_t index = 0; index < resolution.stage_count; ++index) {
    const auto &stage = resolution.stages[index];
    if (stage.skipped_optional)
      continue;
    if (prepared_index >= domain_plan.islands[0].stage_count ||
        domain_plan.islands[0].resolved_stage_indices[prepared_index] !=
            index ||
        stage.selected_backend != hal::BackendId::maxine ||
        stage.adapter_instance_id != registration.adapter_instance_id ||
        stage.execution_domain != registration.execution_domain ||
        !IsQualifiedStage(stage.intent.semantic) ||
        stage.provider != ProviderFor(stage.intent.semantic)) {
      SetReason(reason, stage.execution_domain != registration.execution_domain
                            ? hal::ReasonId::wrong_execution_domain
                            : hal::ReasonId::semantic_effect_unsupported);
      return false;
    }
    ++prepared_index;
  }
  if (prepared_index != domain_plan.islands[0].stage_count) {
    SetReason(reason, hal::ReasonId::malformed_contract);
    return false;
  }
  return true;
}

hal::PreparedProgram
BuildProgram(const video::VideoExecutionPlanRequest &request,
             const hal::ResolutionResult &resolution,
             const hal::DomainPlan &domain_plan) noexcept {
  const auto &island = domain_plan.islands[0];
  hal::PreparedProgram program;
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
  program.requested_provider = request.intent.requested_provider !=
                                       hal::SemanticEffectProviderId::invalid
                                   ? request.intent.requested_provider
                                   : resolution.selected_provider;
  program.strict_provider_selection = request.intent.strict_provider_selection;
  program.requested_execution_domain = island.execution_domain;
  program.strict_execution_domain = true;
  program.requires_upload = island.requires_upload;
  program.requires_final_readback = island.requires_final_readback;

  const auto resident_input =
      ResidentImage(request.host_input.width, request.host_input.height,
                    hal::Mutability::read_only);
  const auto resident_output =
      ResidentImage(request.host_output.width, request.host_output.height,
                    hal::Mutability::read_write);
  for (std::size_t index = 0; index < resolution.stage_count; ++index) {
    const auto &resolved = resolution.stages[index];
    if (resolved.skipped_optional)
      continue;
    const auto &intent = request.intent.stages[index];
    auto &stage = program.stages[program.stage_count++];
    stage.stage_id = resolved.intent.stage_id;
    stage.semantic = resolved.intent.semantic;
    stage.provider = resolved.provider;
    stage.backend = resolved.selected_backend;
    stage.execution_domain = resolved.execution_domain;
    stage.input.kind = hal::StageIoKind::image;
    stage.input.image = resident_input;
    stage.output.kind = hal::StageIoKind::image;
    stage.output.image = resident_output;
    stage.required = resolved.intent.required;
    stage.failure_action = resolved.intent.failure_action;
    stage.circuit_breaker.consecutive_failure_limit = 1;
    stage.circuit_breaker.cooldown_frames = 30;
    stage.circuit_breaker.action =
        stage.required ? hal::CircuitBreakerAction::quarantine_program
                       : hal::CircuitBreakerAction::disable_stage;
    stage.model = intent.model;
  }
  return program;
}

bool ParameterAsSigned(const video::CanonicalVideoParameter &parameter,
                       std::int64_t *value) noexcept {
  if (parameter.kind != video::CanonicalVideoParameterKind::signed_integer) {
    return false;
  }
  *value = parameter.signed_value;
  return true;
}

bool ParameterAsUnsigned(const video::CanonicalVideoParameter &parameter,
                         std::uint64_t *value) noexcept {
  if (parameter.kind != video::CanonicalVideoParameterKind::unsigned_integer) {
    return false;
  }
  *value = parameter.unsigned_value;
  return true;
}

bool ParameterAsBool(const video::CanonicalVideoParameter &parameter,
                     bool *value) noexcept {
  if (parameter.kind != video::CanonicalVideoParameterKind::boolean)
    return false;
  *value = parameter.boolean_value;
  return true;
}

bool FitsInt(std::int64_t value) noexcept {
  return value >= std::numeric_limits<int>::min() &&
         value <= std::numeric_limits<int>::max();
}

bool ApplyStageParameters(const video::CanonicalVideoStage &stage,
                          video::effects::BroadcastCameraEffects *effects,
                          hal::ReasonId *reason) noexcept {
  auto &background = effects->virtual_background;
  auto &key_light = effects->virtual_key_light;
  auto &vignette = effects->vignette;
  switch (stage.intent.semantic) {
  case hal::SemanticEffectId::virtual_background_blur:
    background.mode = video::effects::VirtualBackgroundMode::blur;
    break;
  case hal::SemanticEffectId::virtual_background_remove:
    background.mode = video::effects::VirtualBackgroundMode::remove;
    break;
  case hal::SemanticEffectId::virtual_background_replace:
    background.mode = video::effects::VirtualBackgroundMode::replace;
    break;
  case hal::SemanticEffectId::virtual_key_light:
    key_light.enabled = true;
    break;
  case hal::SemanticEffectId::vignette:
    vignette.enabled = true;
    break;
  default:
    SetReason(reason, hal::ReasonId::semantic_effect_unsupported);
    return false;
  }

  for (std::size_t index = 0; index < stage.parameter_count; ++index) {
    const auto &parameter = stage.parameters[index];
    std::int64_t signed_value = 0;
    std::uint64_t unsigned_value = 0;
    bool bool_value = false;
    switch (parameter.id) {
    case video::CanonicalVideoParameterId::enabled:
      if (!ParameterAsBool(parameter, &bool_value) || !bool_value) {
        SetReason(reason, hal::ReasonId::malformed_contract);
        return false;
      }
      break;
    case video::CanonicalVideoParameterId::strength:
      if ((stage.intent.semantic !=
               hal::SemanticEffectId::virtual_background_blur &&
           stage.intent.semantic !=
               hal::SemanticEffectId::virtual_background_remove &&
           stage.intent.semantic !=
               hal::SemanticEffectId::virtual_background_replace) ||
          !ParameterAsSigned(parameter, &signed_value) ||
          !FitsInt(signed_value)) {
        SetReason(reason, hal::ReasonId::malformed_contract);
        return false;
      }
      background.strength = static_cast<int>(signed_value);
      break;
    case video::CanonicalVideoParameterId::remove_color_rgb:
      if ((stage.intent.semantic !=
               hal::SemanticEffectId::virtual_background_remove &&
           stage.intent.semantic !=
               hal::SemanticEffectId::virtual_background_replace) ||
          !ParameterAsUnsigned(parameter, &unsigned_value) ||
          unsigned_value > 0xffffffu) {
        SetReason(reason, hal::ReasonId::malformed_contract);
        return false;
      }
      {
        constexpr char digits[] = "0123456789ABCDEF";
        background.remove_color = "#000000";
        for (std::size_t digit = 0; digit < 6; ++digit) {
          const auto shift = static_cast<unsigned>((5 - digit) * 4);
          background.remove_color[digit + 1] =
              digits[(unsigned_value >> shift) & 0xfu];
        }
      }
      break;
    case video::CanonicalVideoParameterId::greenscreen_mode:
      if ((stage.intent.semantic !=
               hal::SemanticEffectId::virtual_background_blur &&
           stage.intent.semantic !=
               hal::SemanticEffectId::virtual_background_remove &&
           stage.intent.semantic !=
               hal::SemanticEffectId::virtual_background_replace) ||
          !ParameterAsUnsigned(parameter, &unsigned_value) ||
          unsigned_value > std::numeric_limits<std::uint32_t>::max()) {
        SetReason(reason, hal::ReasonId::malformed_contract);
        return false;
      }
      background.greenscreen_mode = static_cast<std::uint32_t>(unsigned_value);
      break;
    case video::CanonicalVideoParameterId::greenscreen_temporal:
      if ((stage.intent.semantic !=
               hal::SemanticEffectId::virtual_background_blur &&
           stage.intent.semantic !=
               hal::SemanticEffectId::virtual_background_remove &&
           stage.intent.semantic !=
               hal::SemanticEffectId::virtual_background_replace) ||
          !ParameterAsBool(parameter, &bool_value)) {
        SetReason(reason, hal::ReasonId::malformed_contract);
        return false;
      }
      background.greenscreen_temporal = bool_value;
      break;
    case video::CanonicalVideoParameterId::intensity:
      if ((stage.intent.semantic != hal::SemanticEffectId::virtual_key_light &&
           stage.intent.semantic != hal::SemanticEffectId::vignette) ||
          !ParameterAsSigned(parameter, &signed_value) ||
          !FitsInt(signed_value)) {
        SetReason(reason, hal::ReasonId::malformed_contract);
        return false;
      }
      if (stage.intent.semantic == hal::SemanticEffectId::virtual_key_light) {
        key_light.intensity = static_cast<int>(signed_value);
      } else {
        vignette.intensity = static_cast<int>(signed_value);
      }
      break;
    case video::CanonicalVideoParameterId::temperature_preset:
      if (stage.intent.semantic != hal::SemanticEffectId::virtual_key_light ||
          !ParameterAsSigned(parameter, &signed_value) ||
          !FitsInt(signed_value)) {
        SetReason(reason, hal::ReasonId::malformed_contract);
        return false;
      }
      key_light.temperature_preset = static_cast<int>(signed_value);
      break;
    case video::CanonicalVideoParameterId::direction_pan_degrees:
      if (stage.intent.semantic != hal::SemanticEffectId::virtual_key_light ||
          !ParameterAsSigned(parameter, &signed_value) ||
          !FitsInt(signed_value)) {
        SetReason(reason, hal::ReasonId::malformed_contract);
        return false;
      }
      key_light.direction_pan_degrees = static_cast<int>(signed_value);
      break;
    case video::CanonicalVideoParameterId::center_on_tracked_face:
      if (stage.intent.semantic != hal::SemanticEffectId::vignette ||
          !ParameterAsBool(parameter, &bool_value) || bool_value) {
        SetReason(reason, hal::ReasonId::capability_evidence_incomplete);
        return false;
      }
      vignette.center_on_tracked_face = false;
      break;
    case video::CanonicalVideoParameterId::smoothing:
    case video::CanonicalVideoParameterId::headroom:
    case video::CanonicalVideoParameterId::look_away_enabled:
    case video::CanonicalVideoParameterId::invalid:
      SetReason(reason, hal::ReasonId::malformed_contract);
      return false;
    }
  }
  return true;
}

bool BuildOwnerRequest(
    const video::FrozenVideoExecutionPlan &plan,
    const video::PreparedMaxineHardwareRequest &setup_template,
    video::PreparedMaxineHardwareRequest *request, hal::ReasonId *reason) {
  *request = setup_template;
  request->requested_backend = hal::BackendId::maxine;
  request->strict_backend_selection = true;
  request->single_resident_island = true;
  request->contains_other_execution_domain = false;
  request->temporal_reset_required = plan.temporal.temporal;
  request->host_input_is_tight_rgb24 = true;
  request->host_output_is_tight_bgr24 = true;
  request->declares_output_format_cpu_tail = true;
  request->adapter_instance_id =
      plan.domain_plan.islands[0].adapter_instance_id;
  request->width = plan.host_input.width;
  request->height = plan.host_input.height;
  request->stage_count = 0;
  request->generations = plan.invalidation_key.generations;
  request->capacity = plan.budget;
  request->effects.engine = video::effects::EffectsEnginePreference::maxine;
  request->effects.virtual_background.mode =
      video::effects::VirtualBackgroundMode::none;
  request->effects.virtual_key_light.enabled = false;
  request->effects.vignette.enabled = false;

  for (std::size_t index = 0; index < plan.resolution.stage_count; ++index) {
    const auto &resolved = plan.resolution.stages[index];
    if (resolved.skipped_optional)
      continue;
    const auto &canonical = plan.intent.stages[index];
    auto &stage = request->stages[request->stage_count++];
    stage.stage_id = resolved.intent.stage_id;
    stage.semantic = resolved.intent.semantic;
    stage.required = resolved.intent.required;
    stage.model = canonical.model;
    if (!ApplyStageParameters(canonical, &request->effects, reason))
      return false;
  }
  return true;
}

class MaxineVideoExecutionExecutor final
    : public video::IVideoExecutionExecutor {
public:
  MaxineVideoExecutionExecutor(
      std::unique_ptr<video::PreparedMaxineHardwareExecutionOwner> owner,
      video::FrozenVideoExecutionPlan plan,
      std::shared_ptr<void> adapter_lifetime, void *device_loss_owner,
      void (*notify_device_lost)(void *) noexcept) noexcept
      : adapter_lifetime_(std::move(adapter_lifetime)),
        owner_(std::move(owner)), plan_(std::move(plan)),
        device_loss_owner_(device_loss_owner),
        notify_device_lost_(notify_device_lost) {}

  video::VideoExecutionFrameResult
  Execute(const hal::HostFrameBinding &binding) noexcept override {
    video::VideoExecutionFrameResult result;
    if (!owner_ || binding.input_plane_count != 1 ||
        binding.output_plane_count != 1 ||
        !SameImage(binding.input_descriptor, plan_.host_input) ||
        !SameImage(binding.output_descriptor, plan_.host_output) ||
        binding.ownership != hal::OwnershipModel::borrowed_until_completion ||
        binding.lifetime != hal::ResourceLifetime::frame_borrowed ||
        binding.input_planes[0].read_data == nullptr ||
        binding.output_planes[0].write_data == nullptr) {
      result.status = hal::StatusId::blocked;
      result.reason = hal::ReasonId::host_frame_binding_invalid;
      return result;
    }
    const auto frame = owner_->ExecuteHostFrame(
        reinterpret_cast<const std::uint8_t *>(
            binding.input_planes[0].read_data),
        static_cast<std::size_t>(binding.input_planes[0].span_bytes),
        static_cast<std::size_t>(binding.input_planes[0].row_stride_bytes),
        reinterpret_cast<std::uint8_t *>(binding.output_planes[0].write_data),
        static_cast<std::size_t>(binding.output_planes[0].span_bytes),
        static_cast<std::size_t>(binding.output_planes[0].row_stride_bytes),
        binding.capture_sequence);
    result.status = frame.status;
    result.reason = frame.reason;
    result.completion = frame.completion;
    result.execution_domain = plan_.domain_plan.islands[0].execution_domain;
    result.stages = frame.stages;
    result.stage_count = frame.stage_count;
    result.transfers = frame.transfers;
    result.output.descriptor = binding.output_descriptor;
    result.output.planes = binding.output_planes;
    result.output.plane_count = binding.output_plane_count;
    result.output.capture_sequence = binding.capture_sequence;
    result.output.output_ready = frame.output_ready;
    result.output_ready = frame.output_ready;
    if (frame.reason == hal::ReasonId::device_lost) {
      owner_->NotifyDeviceLost();
      if (notify_device_lost_)
        notify_device_lost_(device_loss_owner_);
    }
    return result;
  }

  hal::StatusId CommitOutput(const hal::CompletionToken &completion,
                             bool output_write_succeeded,
                             hal::ReasonId *reason) noexcept override {
    if (!owner_) {
      SetReason(reason, hal::ReasonId::stale_program);
      return hal::StatusId::stale;
    }
    return owner_->CommitOutput(completion, output_write_succeeded, reason);
  }

  void Invalidate(hal::ReasonId reason) noexcept override {
    if (owner_)
      owner_->Invalidate(reason);
  }

  hal::StatusId ResetTemporal(hal::ReasonId) noexcept override {
    return plan_.temporal.temporal ? hal::StatusId::blocked
                                   : hal::StatusId::ready;
  }

  void Shutdown() noexcept override {
    if (owner_)
      owner_->Shutdown();
  }

  video::VideoExecutionAdapterSnapshot Snapshot() const noexcept override {
    video::VideoExecutionAdapterSnapshot result;
    if (!owner_) {
      result.status = hal::StatusId::stale;
      result.reason = hal::ReasonId::stale_program;
      return result;
    }
    const auto snapshot = owner_->Snapshot();
    result.status = snapshot.readiness.status;
    result.reason = snapshot.readiness.reason;
    result.readiness = snapshot.readiness;
    result.telemetry = snapshot.telemetry;
    result.latest_committed_frame = snapshot.latest_committed_frame;
    result.execution_domain = snapshot.execution_domain;
    result.plan_id = snapshot.active_program.program_id;
    result.prepared = snapshot.prepared;
    result.stopped = snapshot.stopped;
    return result;
  }

private:
  // Declared before owner_ so reverse member destruction tears down the
  // borrowed per-plan session while its production adapter is still alive.
  std::shared_ptr<void> adapter_lifetime_;
  std::unique_ptr<video::PreparedMaxineHardwareExecutionOwner> owner_;
  video::FrozenVideoExecutionPlan plan_{};
  void *device_loss_owner_ = nullptr;
  void (*notify_device_lost_)(void *) noexcept = nullptr;
};

} // namespace

MaxineVideoExecutionFactory::MaxineVideoExecutionFactory(
    MaxineVideoExecutionFactoryConfiguration configuration)
    : configuration_(std::move(configuration)) {}

video::VideoAdapterFactoryRegistration
MaxineVideoExecutionFactory::registration() noexcept {
  video::VideoAdapterFactoryRegistration result;
  result.factory_id = configuration_.factory_id;
  result.route = video::VideoExecutionRoute::prepared_resident;
  if (configuration_.adapter_registration) {
    const auto &adapter = *configuration_.adapter_registration;
    result.backend = adapter.backend;
    result.adapter_instance_id = adapter.adapter_instance_id;
    result.execution_domain = adapter.execution_domain;
  }
  result.provider = hal::SemanticEffectProviderId::invalid;
  result.selection_priority = configuration_.selection_priority;
  result.factory = this;
  return result;
}

video::VideoAdapterFactoryPreflight MaxineVideoExecutionFactory::Preflight(
    const video::VideoExecutionPlanRequest &request,
    const hal::ResolutionResult &frozen_resolution,
    const hal::DomainPlan &frozen_domain_plan,
    video::VideoExecutionRoute route) noexcept {
  video::VideoAdapterFactoryPreflight result;
  result.backend = hal::BackendId::maxine;
  result.provider = frozen_resolution.selected_provider;
  result.cached_evidence_only = true;
  result.accepts_frozen_resolution = true;
  result.performs_selection = false;
  if (configuration_.adapter_registration)
    result.execution_domain =
        configuration_.adapter_registration->execution_domain;
  result.per_frame_transfers = video::BuildPerFrameTransferContract(
      frozen_domain_plan, configuration_.minimum_synchronizations_per_frame,
      configuration_.maximum_synchronizations_per_frame);
  result.readiness.requested_backend = frozen_resolution.requested_backend;
  result.readiness.selected_backend = hal::BackendId::maxine;
  result.readiness.provider = frozen_resolution.selected_provider;
  result.readiness.execution_domain = result.execution_domain;

  hal::ReasonId reason = hal::ReasonId::none;
  if (route != video::VideoExecutionRoute::prepared_resident ||
      configuration_.factory_id == 0 ||
      configuration_.adapter_registration == nullptr ||
      configuration_.adapter_registration->resident_adapter == nullptr ||
      configuration_.adapter_registration->backend != hal::BackendId::maxine ||
      configuration_.adapter_registration->adapter_instance_id == 0 ||
      configuration_.adapter_registration->execution_domain.backend !=
          hal::BackendId::maxine ||
      configuration_.adapter_registration->execution_domain
              .adapter_instance_id !=
          configuration_.adapter_registration->adapter_instance_id ||
      !hal::ValidateExecutionDomain(
          configuration_.adapter_registration->execution_domain)) {
    result.status = hal::StatusId::blocked;
    result.reason = route != video::VideoExecutionRoute::prepared_resident
                        ? hal::ReasonId::semantic_provider_unavailable
                        : hal::ReasonId::malformed_contract;
    result.readiness.status = result.status;
    result.readiness.reason = result.reason;
    return result;
  }
  if (!ExactFrozenInputs(request, frozen_resolution, frozen_domain_plan,
                         *configuration_.adapter_registration, &reason)) {
    result.status = hal::StatusId::blocked;
    result.reason = reason;
    result.readiness.status = result.status;
    result.readiness.reason = result.reason;
    return result;
  }
  if (!video::ValidatePerFrameTransferContract(result.per_frame_transfers,
                                               frozen_domain_plan, &reason)) {
    result.status = hal::StatusId::blocked;
    result.reason = reason;
    result.readiness.status = result.status;
    result.readiness.reason = result.reason;
    return result;
  }
  result.prepared_program =
      BuildProgram(request, frozen_resolution, frozen_domain_plan);
  if (!hal::ValidatePreparedProgram(result.prepared_program, &reason)) {
    result.status = reason == hal::ReasonId::resource_exhausted
                        ? hal::StatusId::resource_exhausted
                        : hal::StatusId::blocked;
    result.reason = reason;
    result.readiness.status = result.status;
    result.readiness.reason = result.reason;
    return result;
  }
  result.status = frozen_resolution.status == hal::StatusId::degraded
                      ? hal::StatusId::degraded
                      : hal::StatusId::ready;
  result.reason = frozen_resolution.reason;
  result.readiness.status = result.status;
  result.readiness.reason = result.reason;
  result.readiness.ready_stage_mask =
      result.prepared_program.stage_count == 32
          ? std::numeric_limits<std::uint32_t>::max()
          : (std::uint32_t{1} << result.prepared_program.stage_count) - 1u;
  result.readiness.available = true;
  result.readiness.degraded = result.status == hal::StatusId::degraded;
  return result;
}

std::unique_ptr<video::IVideoExecutionExecutor>
MaxineVideoExecutionFactory::Prepare(
    const video::FrozenVideoExecutionPlan &plan,
    video::VideoAdapterPrepareEvidence *evidence) noexcept {
  if (evidence)
    *evidence = {};
  const auto Fail =
      [&](hal::StatusId status, hal::ReasonId reason,
          bool rolled_back) -> std::unique_ptr<video::IVideoExecutionExecutor> {
    if (evidence) {
      evidence->status = status;
      evidence->reason = reason;
      evidence->candidate_resources_rolled_back = rolled_back;
    }
    return {};
  };
  if (configuration_.adapter_registration == nullptr) {
    return Fail(hal::StatusId::blocked, hal::ReasonId::malformed_contract,
                true);
  }
  if (plan.route != video::VideoExecutionRoute::prepared_resident ||
      plan.factory_id != configuration_.factory_id ||
      plan.domain_plan.island_count != 1 ||
      plan.domain_plan.islands[0].execution_domain !=
          configuration_.adapter_registration->execution_domain ||
      !video::ValidatePerFrameTransferContract(plan.per_frame_transfers,
                                               plan.domain_plan, nullptr) ||
      plan.prepared_program.program_id != plan.plan_id ||
      plan.prepared_program.execution_domain !=
          configuration_.adapter_registration->execution_domain) {
    return Fail(
        hal::StatusId::blocked,
        plan.domain_plan.island_count == 1 &&
                plan.domain_plan.islands[0].execution_domain !=
                    configuration_.adapter_registration->execution_domain
            ? hal::ReasonId::wrong_execution_domain
            : hal::ReasonId::malformed_contract,
        true);
  }

  try {
    video::PreparedMaxineHardwareRequest owner_request;
    hal::ReasonId reason = hal::ReasonId::none;
    if (!BuildOwnerRequest(plan, configuration_.setup_template, &owner_request,
                           &reason)) {
      return Fail(hal::StatusId::blocked, reason, true);
    }
    auto owner = std::make_unique<video::PreparedMaxineHardwareExecutionOwner>(
        *configuration_.adapter_registration,
        static_cast<bool>(configuration_.adapter_lifetime));
    const auto setup =
        owner->ConfigureExact(owner_request, plan.prepared_program);
    if (setup.status != hal::StatusId::ready &&
        setup.status != hal::StatusId::degraded) {
      owner->Shutdown();
      return Fail(setup.status, setup.reason, true);
    }
    const auto snapshot = owner->Snapshot();
    if (!snapshot.prepared || snapshot.stopped ||
        snapshot.active_program.program_id != plan.plan_id ||
        snapshot.active_program.invalidation_key != plan.invalidation_key ||
        snapshot.execution_domain !=
            plan.domain_plan.islands[0].execution_domain) {
      owner->Shutdown();
      return Fail(hal::StatusId::blocked,
                  snapshot.execution_domain !=
                          plan.domain_plan.islands[0].execution_domain
                      ? hal::ReasonId::wrong_execution_domain
                      : hal::ReasonId::malformed_contract,
                  true);
    }
    if (evidence) {
      evidence->status = setup.status;
      evidence->reason = setup.reason;
      evidence->readiness = snapshot.readiness;
      evidence->candidate_resources_rolled_back = false;
    }
    return std::make_unique<MaxineVideoExecutionExecutor>(
        std::move(owner), plan, configuration_.adapter_lifetime,
        configuration_.device_loss_owner,
        configuration_.notify_device_lost);
  } catch (const std::bad_alloc &) {
    return Fail(hal::StatusId::resource_exhausted,
                hal::ReasonId::resource_exhausted, true);
  } catch (...) {
    return Fail(hal::StatusId::blocked,
                hal::ReasonId::capability_evidence_incomplete, true);
  }
}

} // namespace studiocast::maxine
