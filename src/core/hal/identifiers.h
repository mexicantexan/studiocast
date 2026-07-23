#pragma once

#include <cstdint>
#include <string_view>

namespace studiocast::hal {

inline constexpr std::uint32_t kContractVersion = 1;

enum class BackendId : std::uint32_t {
  invalid = 0,
  cpu = 1,
  cuda = 2,
  maxine = 3,
  open_video_onnxruntime = 4,
  vulkan = 5,
  test_fake = 0x7fff0001u,
};

enum class PrimitiveOperationId : std::uint32_t {
  invalid = 0,
  convert_pixel_format = 1,
  resize = 2,
  mirror = 3,
  box_blur = 4,
  composite = 5,
  alpha_feather = 6,
};

// These values map to the existing stable Broadcast effect IDs. They describe
// product semantics, not the presence of a utility primitive.
enum class SemanticEffectId : std::uint32_t {
  invalid = 0,
  video_noise_removal = 1,
  eye_contact = 2,
  virtual_background_blur = 3,
  virtual_background_remove = 4,
  virtual_background_replace = 5,
  virtual_key_light = 6,
  auto_frame = 7,
  vignette = 8,
  mirror = 9,
};

enum class SemanticEffectProviderId : std::uint32_t {
  invalid = 0,
  maxine_vfx = 1,
  maxine_ar = 2,
  open_video_onnxruntime_cuda = 3,
  open_video_onnxruntime_cpu = 4,
  open_vulkan_production_kernels = 5,
  maxine_cuda_utility = 6,
  test_fake = 0x7fff0001u,
};

enum class StatusId : std::uint32_t {
  invalid = 0,
  ready = 1,
  blocked = 2,
  degraded = 3,
  unavailable = 4,
  passthrough = 5,
  executed = 6,
  failed = 7,
  quarantined = 8,
  stopped = 9,
  resource_exhausted = 10,
  stale = 11,
};

enum class ReasonId : std::uint32_t {
  none = 0,
  malformed_contract = 1,
  unsupported_contract_version = 2,
  malformed_descriptor = 3,
  malformed_execution_domain = 4,
  foreign_adapter = 5,
  stale_program = 6,
  wrong_generation = 7,
  wrong_execution_domain = 8,
  resource_exhausted = 9,
  cross_adapter_residency_forbidden = 10,
  required_stage_failed = 11,
  optional_stage_failed_open = 12,
  device_lost = 13,
  temporal_discontinuity = 14,
  reconfiguration_required = 15,
  shutdown_requested = 16,
  output_write_failed = 17,
  output_not_committed = 18,
  incomplete_execution = 19,
  invalid_resource = 20,
  synchronization_failed = 21,
  strict_backend_unavailable = 22,
  cpu_tail_required = 23,
  circuit_breaker_open = 24,
  cancellation_unsupported_v1 = 25,
  semantic_provider_unavailable = 26,
  semantic_effect_unsupported = 27,
  capability_evidence_incomplete = 28,
  host_frame_binding_unsupported = 29,
  host_frame_binding_invalid = 30,
  circuit_breaker_retry_pending = 31,
  circuit_breaker_permanently_disabled = 32,
  old_program_policy_incompatible = 33,
  transfer_contract_violated = 34,
  frame_order_violated = 35,
};

// These are additive, machine-readable capability facts. ReasonId remains the
// concise outcome; evidence records preserve exact installer/backend proof
// without strings or JSON in the neutral contract.
enum class CapabilityEvidenceCode : std::uint32_t {
  invalid = 0,
  vulkan_build_enabled = 1,
  vulkan_loader_available = 2,
  vulkan_physical_device_available = 3,
  vulkan_non_cpu_device = 4,
  vulkan_compute_queue_available = 5,
  vulkan_context_healthy = 6,
  vulkan_utility_kernels_ready = 7,
  vulkan_exact_device_shared = 8,
  vulkan_device_local_residency = 9,
  vulkan_bounded_allocations = 10,
  vulkan_no_cpu_layers = 11,
  vulkan_no_intermediate_readbacks = 12,
  open_vulkan_mirror_production_ready = 13,
  open_vulkan_vignette_fixed_center_production_ready = 14,
  vulkan_matting_runtime_ready = 15,
  vulkan_model_artifacts_valid = 16,
  vulkan_warmup_complete = 17,
  vulkan_live_stage_present = 18,
  vulkan_tracked_vignette_center_supported = 19,
  provider_runtime_available = 20,
  model_artifact_identity_valid = 21,
  synchronization_evidence_valid = 22,
  import_export_interop_valid = 23,
  vulkan_backend_disabled_in_build = 24,
  vulkan_runtime_not_found = 25,
  vulkan_loader_unavailable = 26,
  vulkan_no_physical_device = 27,
  vulkan_no_compute_queue = 28,
  vulkan_only_cpu_devices_available = 29,
  vulkan_requested_device_not_found = 30,
  vulkan_requested_device_no_compute_queue = 31,
  vulkan_requested_device_ambiguous = 32,
  vulkan_device_create_failed = 33,
  vulkan_context_uninitialized = 34,
  vulkan_device_lost = 35,
  vulkan_production_hardware_not_ready = 36,
  open_vulkan_utility_kernels_unavailable = 37,
  open_vulkan_runtime_diagnostics_unavailable = 38,
  open_vulkan_matting_unavailable = 39,
  open_vulkan_matting_adapter_unavailable = 40,
  open_vulkan_eye_contact_runtime_unavailable = 41,
  open_vulkan_video_noise_removal_runtime_unavailable = 42,
  vulkan_auto_frame_yunet_unavailable = 43,
  vulkan_vignette_tracked_center_not_supported = 44,
  vulkan_production_evidence_inconsistent = 45,
  per_effect_capability_evidence_missing = 46,
  open_vulkan_eye_contact_unavailable = 47,
  open_vulkan_video_noise_removal_unavailable = 48,
  vulkan_effect_cpu_tail = 49,
};

[[nodiscard]] constexpr std::string_view
StableEffectId(SemanticEffectId id) noexcept {
  switch (id) {
  case SemanticEffectId::video_noise_removal:
    return "video_noise_removal";
  case SemanticEffectId::eye_contact:
    return "eye_contact";
  case SemanticEffectId::virtual_background_blur:
    return "virtual_background.blur";
  case SemanticEffectId::virtual_background_remove:
    return "virtual_background.remove";
  case SemanticEffectId::virtual_background_replace:
    return "virtual_background.replace";
  case SemanticEffectId::virtual_key_light:
    return "virtual_key_light";
  case SemanticEffectId::auto_frame:
    return "auto_frame";
  case SemanticEffectId::vignette:
    return "vignette";
  case SemanticEffectId::mirror:
    return "mirror";
  case SemanticEffectId::invalid:
    break;
  }
  return {};
}

} // namespace studiocast::hal
