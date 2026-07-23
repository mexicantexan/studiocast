#include "core/video/effects/capability_reason_codes.h"

namespace studiocast::video::effects {

std::string_view EffectCapabilityCodeToken(EffectCapabilityCode code) noexcept {
  switch (code) {
  case EffectCapabilityCode::none:
    return {};
  case EffectCapabilityCode::maxine_effect_usable:
    return "maxine.effect.usable";
  case EffectCapabilityCode::maxine_effect_unavailable:
    return "maxine.effect.unavailable";
  case EffectCapabilityCode::open_cuda_effect_available:
    return "open_cuda_effect_available";
  case EffectCapabilityCode::open_cuda_effect_unavailable:
    return "open_cuda_effect_unavailable";
  case EffectCapabilityCode::cpu_video_noise_removal_unavailable:
    return "effect.video_noise_removal.cpu.no_selectable_production_path";
  case EffectCapabilityCode::cpu_eye_contact_unavailable:
    return "effect.eye_contact.cpu.no_selectable_production_path";
  case EffectCapabilityCode::cpu_virtual_background_blur_unavailable:
    return "effect.virtual_background.blur.cpu.no_selectable_production_path";
  case EffectCapabilityCode::cpu_virtual_background_remove_unavailable:
    return "effect.virtual_background.remove.cpu.no_selectable_production_path";
  case EffectCapabilityCode::cpu_virtual_background_replace_unavailable:
    return "effect.virtual_background.replace.cpu.no_selectable_production_"
           "path";
  case EffectCapabilityCode::cpu_virtual_key_light_unavailable:
    return "effect.virtual_key_light.cpu.no_selectable_production_path";
  case EffectCapabilityCode::cpu_auto_frame_unavailable:
    return "effect.auto_frame.cpu.no_selectable_production_path";
  case EffectCapabilityCode::cpu_vignette_unavailable:
    return "effect.vignette.cpu.no_selectable_production_path";
  case EffectCapabilityCode::cpu_mirror_unavailable:
    return "effect.mirror.cpu.no_selectable_production_path";
  case EffectCapabilityCode::open_vulkan_mirror_production_ready:
    return "open_vulkan_mirror_production_ready";
  case EffectCapabilityCode::open_vulkan_vignette_fixed_center_production_ready:
    return "open_vulkan_vignette_fixed_center_production_ready";
  case EffectCapabilityCode::vulkan_backend_disabled_in_build:
    return "vulkan_backend_disabled_in_build";
  case EffectCapabilityCode::vulkan_runtime_not_found:
    return "vulkan_runtime_not_found";
  case EffectCapabilityCode::vulkan_loader_unavailable:
    return "vulkan_loader_unavailable";
  case EffectCapabilityCode::vulkan_no_physical_device:
    return "vulkan_no_physical_device";
  case EffectCapabilityCode::vulkan_no_compute_queue:
    return "vulkan_no_compute_queue";
  case EffectCapabilityCode::vulkan_only_cpu_devices_available:
    return "vulkan_only_cpu_devices_available";
  case EffectCapabilityCode::vulkan_requested_device_not_found:
    return "vulkan_requested_device_not_found";
  case EffectCapabilityCode::vulkan_requested_device_no_compute_queue:
    return "vulkan_requested_device_no_compute_queue";
  case EffectCapabilityCode::vulkan_requested_device_ambiguous:
    return "vulkan_requested_device_ambiguous";
  case EffectCapabilityCode::vulkan_device_create_failed:
    return "vulkan_device_create_failed";
  case EffectCapabilityCode::vulkan_context_uninitialized:
    return "vulkan_context_uninitialized";
  case EffectCapabilityCode::vulkan_device_lost:
    return "vulkan_device_lost";
  case EffectCapabilityCode::vulkan_production_hardware_not_ready:
    return "vulkan_production_hardware_not_ready";
  case EffectCapabilityCode::open_vulkan_utility_kernels_unavailable:
    return "open_vulkan_utility_kernels_unavailable";
  case EffectCapabilityCode::open_vulkan_runtime_diagnostics_unavailable:
    return "open_vulkan_runtime_diagnostics_unavailable";
  case EffectCapabilityCode::open_vulkan_matting_unavailable:
    return "open_vulkan_matting_unavailable";
  case EffectCapabilityCode::open_vulkan_matting_adapter_unavailable:
    return "open_vulkan_matting_adapter_unavailable";
  case EffectCapabilityCode::open_vulkan_eye_contact_unavailable:
    return "open_vulkan_eye_contact_unavailable";
  case EffectCapabilityCode::open_vulkan_eye_contact_runtime_unavailable:
    return "open_vulkan_eye_contact_runtime_unavailable";
  case EffectCapabilityCode::open_vulkan_video_noise_removal_unavailable:
    return "open_vulkan_video_noise_removal_unavailable";
  case EffectCapabilityCode::
      open_vulkan_video_noise_removal_runtime_unavailable:
    return "open_vulkan_video_noise_removal_runtime_unavailable";
  case EffectCapabilityCode::vulkan_auto_frame_yunet_unavailable:
    return "vulkan_auto_frame_yunet_unavailable";
  case EffectCapabilityCode::vulkan_vignette_tracked_center_not_supported:
    return "vulkan_vignette_tracked_center_not_supported";
  case EffectCapabilityCode::vulkan_production_evidence_inconsistent:
    return "vulkan_production_evidence_inconsistent";
  case EffectCapabilityCode::per_effect_capability_evidence_missing:
    return "per_effect_capability_evidence_missing";
  case EffectCapabilityCode::vulkan_effect_cpu_tail:
    return "vulkan_effect_cpu_tail";
  case EffectCapabilityCode::open_vulkan_mirror_unavailable:
    return "open_vulkan_mirror_unavailable";
  case EffectCapabilityCode::open_vulkan_vignette_unavailable:
    return "open_vulkan_vignette_unavailable";
  case EffectCapabilityCode::vulkan_instance_create_failed:
    return "vulkan_instance_create_failed";
  case EffectCapabilityCode::open_vulkan_effect_evidence_not_defined:
    return "open_vulkan_effect_evidence_not_defined";
  case EffectCapabilityCode::open_vulkan_runtime_evidence_inconsistent:
    return "open_vulkan_runtime_evidence_inconsistent";
  case EffectCapabilityCode::open_vulkan_mirror_production_evidence_missing:
    return "open_vulkan_mirror_production_evidence_missing";
  case EffectCapabilityCode::
      open_vulkan_mirror_production_evidence_inconsistent:
    return "open_vulkan_mirror_production_evidence_inconsistent";
  case EffectCapabilityCode::open_vulkan_mirror_live_stage_unavailable:
    return "open_vulkan_mirror_live_stage_unavailable";
  case EffectCapabilityCode::open_vulkan_mirror_live_stage_not_active:
    return "open_vulkan_mirror_live_stage_not_active";
  case EffectCapabilityCode::open_vulkan_vignette_production_evidence_missing:
    return "open_vulkan_vignette_production_evidence_missing";
  case EffectCapabilityCode::
      open_vulkan_vignette_production_evidence_inconsistent:
    return "open_vulkan_vignette_production_evidence_inconsistent";
  case EffectCapabilityCode::open_vulkan_vignette_live_stage_unavailable:
    return "open_vulkan_vignette_live_stage_unavailable";
  case EffectCapabilityCode::open_vulkan_vignette_live_stage_not_active:
    return "open_vulkan_vignette_live_stage_not_active";
  case EffectCapabilityCode::open_vulkan_matting_production_ready:
    return "open_vulkan_matting_production_ready";
  case EffectCapabilityCode::open_vulkan_eye_contact_production_ready:
    return "open_vulkan_eye_contact_production_ready";
  case EffectCapabilityCode::open_vulkan_video_noise_removal_production_ready:
    return "open_vulkan_video_noise_removal_production_ready";
  case EffectCapabilityCode::vulkan_probe_failed:
    return "vulkan_probe_failed";
  case EffectCapabilityCode::vulkan_probe_tool_unavailable:
    return "vulkan_probe_tool_unavailable";
  case EffectCapabilityCode::capability_snapshot_current:
    return "capability_snapshot_current";
  case EffectCapabilityCode::capability_refresh_not_run:
    return "capability_refresh_not_run";
  case EffectCapabilityCode::capability_refresh_in_progress:
    return "capability_refresh_in_progress";
  case EffectCapabilityCode::capability_refresh_failed:
    return "capability_refresh_failed";
  case EffectCapabilityCode::capability_snapshot_stale:
    return "capability_snapshot_stale";
  case EffectCapabilityCode::capability_structure_invalid:
    return "capability_structure_invalid";
  }
  return {};
}

bool IsEffectCapabilitySuccessCode(EffectCapabilityCode code) noexcept {
  switch (code) {
  case EffectCapabilityCode::maxine_effect_usable:
  case EffectCapabilityCode::open_cuda_effect_available:
  case EffectCapabilityCode::open_vulkan_mirror_production_ready:
  case EffectCapabilityCode::open_vulkan_vignette_fixed_center_production_ready:
  case EffectCapabilityCode::open_vulkan_matting_production_ready:
  case EffectCapabilityCode::open_vulkan_eye_contact_production_ready:
  case EffectCapabilityCode::open_vulkan_video_noise_removal_production_ready:
  case EffectCapabilityCode::capability_snapshot_current:
    return true;
  default:
    return false;
  }
}

} // namespace studiocast::video::effects
