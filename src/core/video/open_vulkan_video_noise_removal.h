#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace studiocast::video {

inline constexpr std::string_view
    kOpenVulkanVideoNoiseRemovalUnavailableReason =
        "open_vulkan_video_noise_removal_unavailable";
inline constexpr std::string_view
    kOpenVulkanVideoNoiseRemovalRuntimeUnavailableReason =
        "open_vulkan_video_noise_removal_runtime_unavailable";
inline constexpr std::string_view
    kOpenVulkanVideoNoiseRemovalUnavailableDetail =
        "no production video-denoise runtime can import StudioCast's exact "
        "Vulkan device, compute queue, and resident buffers; the current "
        "FastDVDnet ONNX path supports only CUDA/CPU providers with host "
        "temporal history and CPU preprocessing/postprocessing, while the "
        "available model packs declare ONNX-only artifacts";

// Product evidence is deliberately split into independent facts. A compiled
// Vulkan backend or healthy utility context must never imply a callable live
// denoiser, compatible inference provider, reviewed artifact, or resident
// temporal implementation.
//
// The future production contract consumes and produces the canonical resident
// RGB frame at unchanged geometry. FastDVDnet parity requires a causal
// five-slot NCHW float32 window: three distinct stored frames (t-2, t-1, t)
// followed by two repeated current-frame slots. Input/output tensors,
// RGB/NCHW preprocessing and postprocessing, and all history must stay on the
// exact StudioCast Vulkan context. Allocations are built once and reused;
// queue ownership and completion are explicit before the next resident stage.
// History is discarded when disabled, strength is zero, configuration/model or
// geometry changes, or capture_sequence is non-consecutive. Failure removes
// only this optional stage and leaves the resident frame available to unrelated
// effects.
struct OpenVulkanVideoNoiseRemovalCapabilityFacts {
  bool backend_compiled = false;
  bool live_stage_implemented = false;
  bool production_adapter_available = false;
  bool vulkan_inference_provider_available = false;
  bool non_cpu_device_selected = false;
  bool compute_queue_available = false;
  bool context_healthy = false;
  bool shared_device_imported = false;
  bool queue_ownership_explicit = false;
  bool model_pack_selected = false;
  bool artifact_contract_validated = false;
  bool fully_device_resident_tensor_io = false;
  bool device_resident_preprocess = false;
  bool device_resident_postprocess = false;
  bool warmup_complete = false;
  bool synchronization_contract_validated = false;
  bool bounded_reusable_allocations = false;
  bool temporal_history_device_resident = false;
  bool temporal_history_bounded = false;
  bool history_reset_on_disable = false;
  bool history_reset_on_reconfigure = false;
  bool capture_sequence_discontinuity_reset = false;
  bool parity_validated = false;
  bool selectable_cpu_fallback = false;

  // These stay zero while this effect is diagnostics-only. They prove that
  // fail-closed planning did not dispatch, mutate history, read back an
  // intermediate, or silently execute a CPU replacement.
  std::uint64_t dispatch_count = 0;
  std::uint64_t temporal_history_reset_count = 0;
  std::uint64_t cpu_readback_count = 0;
  std::uint64_t cpu_fallback_count = 0;
};

struct OpenVulkanVideoNoiseRemovalReadiness {
  bool production_ready = false;
  std::string reason_code;
  std::string blocker_code;
  std::string detail;
  OpenVulkanVideoNoiseRemovalCapabilityFacts facts;
};

// Returns the facts for the shipped implementation. This is diagnostics-only:
// there is intentionally no callable Apply method until the resident runtime,
// model/artifact schema, temporal reset behavior, and parity are proven.
OpenVulkanVideoNoiseRemovalCapabilityFacts
CurrentOpenVulkanVideoNoiseRemovalFacts();

OpenVulkanVideoNoiseRemovalReadiness
EvaluateOpenVulkanVideoNoiseRemovalReadiness(
    const OpenVulkanVideoNoiseRemovalCapabilityFacts &facts);

std::string FormatOpenVulkanVideoNoiseRemovalReadiness(
    const OpenVulkanVideoNoiseRemovalReadiness &readiness);

} // namespace studiocast::video
