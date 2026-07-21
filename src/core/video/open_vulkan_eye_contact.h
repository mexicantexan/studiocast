#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace studiocast::video {

inline constexpr std::string_view kOpenVulkanEyeContactUnavailableReason =
    "open_vulkan_eye_contact_unavailable";
inline constexpr std::string_view
    kOpenVulkanEyeContactRuntimeUnavailableReason =
        "open_vulkan_eye_contact_runtime_unavailable";
inline constexpr std::string_view kOpenVulkanEyeContactUnavailableDetail =
    "no production eye-contact runtime can import StudioCast's exact Vulkan "
    "device, compute queue, and resident buffers; the current ONNX/dlib path "
    "uses CPU analysis, CPU tensors, and CPU postprocess, and its manifests "
    "do not declare a complete Vulkan artifact, gaze, or look-away contract";

// Product evidence is deliberately split into independent facts. A compiled
// Vulkan backend or a healthy utility context must never imply that eye
// contact has a callable live implementation, compatible inference provider,
// reviewed model contract, or selectable CPU fallback.
struct OpenVulkanEyeContactCapabilityFacts {
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
  bool device_resident_analysis = false;
  bool device_resident_tensor_io = false;
  bool warmup_complete = false;
  bool bounded_reusable_allocations = false;
  bool synchronization_contract_validated = false;
  bool parity_validated = false;
  bool selectable_cpu_fallback = false;

  // These remain zero while the effect is diagnostics-only. They are explicit
  // so status can prove that the unavailable stage did not dispatch, read
  // back an intermediate, or silently execute a CPU replacement.
  std::uint64_t dispatch_count = 0;
  std::uint64_t cpu_readback_count = 0;
  std::uint64_t cpu_fallback_count = 0;
};

struct OpenVulkanEyeContactReadiness {
  bool production_ready = false;
  std::string reason_code;
  std::string blocker_code;
  std::string detail;
  OpenVulkanEyeContactCapabilityFacts facts;
};

// Returns the facts for the shipped implementation. This is diagnostics-only:
// there is intentionally no callable Apply method until a reviewed resident
// runtime, model schema, and parity contract exist.
OpenVulkanEyeContactCapabilityFacts CurrentOpenVulkanEyeContactFacts();

OpenVulkanEyeContactReadiness EvaluateOpenVulkanEyeContactReadiness(
    const OpenVulkanEyeContactCapabilityFacts &facts);

std::string FormatOpenVulkanEyeContactReadiness(
    const OpenVulkanEyeContactReadiness &readiness);

} // namespace studiocast::video
