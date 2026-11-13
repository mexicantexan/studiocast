#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace studiocast::video {

inline constexpr std::string_view kOpenVulkanEyeContactUnavailableReason =
    "open_vulkan_eye_contact_unavailable";
inline constexpr std::string_view kOpenVulkanEyeContactProductionReadyReason =
    "open_vulkan_eye_contact_production_ready";
inline constexpr std::string_view
    kOpenVulkanEyeContactRuntimeUnavailableReason =
        "open_vulkan_eye_contact_runtime_unavailable";
inline constexpr std::string_view kOpenVulkanEyeContactUnavailableDetail =
    "Open Vulkan eye contact needs the ncnn/dlib build and validated YuNet, "
    "68-point landmark and FP32 FLX packs prepared and warmed on the selected "
    "Vulkan device/queue. "
    "Check converted artifacts, enabled FP64, resident analysis/tensors and "
    "bounded synchronization; model, context or parity evidence is missing "
    "or invalid.";

// Product evidence is deliberately split into independent facts. A compiled
// Vulkan backend or a healthy utility context must never imply that eye
// contact has a prepared compatible inference provider, reviewed model
// contract, or selectable CPU fallback. Compile-time implementation facts do
// not supply any of the selected-device/model/warmup/parity evidence below.
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

  // Compile-time facts leave these zero. Only completed prepared execution
  // supplies dispatch evidence; intermediate readback or CPU replacement
  // always prevents production readiness.
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

// Compile-time implementation facts only. The prepared owner supplies exact
// device/model/residency/completion and parity evidence after real warmup.
OpenVulkanEyeContactCapabilityFacts CurrentOpenVulkanEyeContactFacts();

OpenVulkanEyeContactReadiness EvaluateOpenVulkanEyeContactReadiness(
    const OpenVulkanEyeContactCapabilityFacts &facts);

std::string FormatOpenVulkanEyeContactReadiness(
    const OpenVulkanEyeContactReadiness &readiness);

} // namespace studiocast::video
