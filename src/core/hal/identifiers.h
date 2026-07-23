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
