#pragma once

#include <cstdint>
#include <string_view>

namespace studiocast::video::effects {

inline constexpr std::uint32_t kEffectCapabilityCodeVersion = 1;

// Product capability outcomes are intentionally separate from the neutral HAL
// ReasonId vocabulary.  These values are persisted at the daemon/installer
// evidence boundary, so existing numeric values and tokens must never be
// repurposed.
enum class EffectCapabilityCode : std::uint32_t {
  none = 0,

  maxine_effect_usable = 1,
  maxine_effect_unavailable = 2,
  open_cuda_effect_available = 3,
  open_cuda_effect_unavailable = 4,

  cpu_video_noise_removal_unavailable = 10,
  cpu_eye_contact_unavailable = 11,
  cpu_virtual_background_blur_unavailable = 12,
  cpu_virtual_background_remove_unavailable = 13,
  cpu_virtual_background_replace_unavailable = 14,
  cpu_virtual_key_light_unavailable = 15,
  cpu_auto_frame_unavailable = 16,
  cpu_vignette_unavailable = 17,
  cpu_mirror_unavailable = 18,

  open_vulkan_mirror_production_ready = 100,
  open_vulkan_vignette_fixed_center_production_ready = 101,
  vulkan_backend_disabled_in_build = 102,
  vulkan_runtime_not_found = 103,
  vulkan_loader_unavailable = 104,
  vulkan_no_physical_device = 105,
  vulkan_no_compute_queue = 106,
  vulkan_only_cpu_devices_available = 107,
  vulkan_requested_device_not_found = 108,
  vulkan_requested_device_no_compute_queue = 109,
  vulkan_requested_device_ambiguous = 110,
  vulkan_device_create_failed = 111,
  vulkan_context_uninitialized = 112,
  vulkan_device_lost = 113,
  vulkan_production_hardware_not_ready = 114,
  open_vulkan_utility_kernels_unavailable = 115,
  open_vulkan_runtime_diagnostics_unavailable = 116,
  open_vulkan_matting_unavailable = 117,
  open_vulkan_matting_adapter_unavailable = 118,
  open_vulkan_eye_contact_unavailable = 119,
  open_vulkan_eye_contact_runtime_unavailable = 120,
  open_vulkan_video_noise_removal_unavailable = 121,
  open_vulkan_video_noise_removal_runtime_unavailable = 122,
  vulkan_auto_frame_yunet_unavailable = 123,
  vulkan_vignette_tracked_center_not_supported = 124,
  vulkan_production_evidence_inconsistent = 125,
  per_effect_capability_evidence_missing = 126,
  vulkan_effect_cpu_tail = 127,
  open_vulkan_mirror_unavailable = 128,
  open_vulkan_vignette_unavailable = 129,
  vulkan_instance_create_failed = 130,
  open_vulkan_effect_evidence_not_defined = 131,
  open_vulkan_runtime_evidence_inconsistent = 132,
  open_vulkan_mirror_production_evidence_missing = 133,
  open_vulkan_mirror_production_evidence_inconsistent = 134,
  open_vulkan_mirror_live_stage_unavailable = 135,
  open_vulkan_mirror_live_stage_not_active = 136,
  open_vulkan_vignette_production_evidence_missing = 137,
  open_vulkan_vignette_production_evidence_inconsistent = 138,
  open_vulkan_vignette_live_stage_unavailable = 139,
  open_vulkan_vignette_live_stage_not_active = 140,
  open_vulkan_matting_production_ready = 141,
  open_vulkan_eye_contact_production_ready = 142,
  open_vulkan_video_noise_removal_production_ready = 143,
  vulkan_probe_failed = 144,
  vulkan_probe_tool_unavailable = 145,

  capability_snapshot_current = 200,
  capability_refresh_not_run = 201,
  capability_refresh_in_progress = 202,
  capability_refresh_failed = 203,
  capability_snapshot_stale = 204,
  capability_structure_invalid = 205,
};

[[nodiscard]] std::string_view
EffectCapabilityCodeToken(EffectCapabilityCode code) noexcept;

[[nodiscard]] bool
IsEffectCapabilitySuccessCode(EffectCapabilityCode code) noexcept;

} // namespace studiocast::video::effects
