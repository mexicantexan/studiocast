#include "core/video/open_vulkan_eye_contact.h"

namespace studiocast::video {

OpenVulkanEyeContactCapabilityFacts CurrentOpenVulkanEyeContactFacts() {
  OpenVulkanEyeContactCapabilityFacts facts;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
  facts.backend_compiled = true;
#endif
  return facts;
}

OpenVulkanEyeContactReadiness EvaluateOpenVulkanEyeContactReadiness(
    const OpenVulkanEyeContactCapabilityFacts &facts) {
  OpenVulkanEyeContactReadiness readiness;
  readiness.facts = facts;
  readiness.production_ready =
      facts.backend_compiled && facts.live_stage_implemented &&
      facts.production_adapter_available &&
      facts.vulkan_inference_provider_available &&
      facts.non_cpu_device_selected && facts.compute_queue_available &&
      facts.context_healthy && facts.shared_device_imported &&
      facts.queue_ownership_explicit && facts.model_pack_selected &&
      facts.artifact_contract_validated && facts.device_resident_analysis &&
      facts.device_resident_tensor_io && facts.warmup_complete &&
      facts.bounded_reusable_allocations &&
      facts.synchronization_contract_validated && facts.parity_validated &&
      facts.dispatch_count > 0 && facts.cpu_readback_count == 0 &&
      facts.cpu_fallback_count == 0;

  if (!readiness.production_ready) {
    readiness.reason_code = kOpenVulkanEyeContactUnavailableReason;
    readiness.blocker_code = kOpenVulkanEyeContactRuntimeUnavailableReason;
    readiness.detail = kOpenVulkanEyeContactUnavailableDetail;
  }
  return readiness;
}

std::string FormatOpenVulkanEyeContactReadiness(
    const OpenVulkanEyeContactReadiness &readiness) {
  if (readiness.production_ready)
    return "Open Vulkan eye contact is production ready.";

  std::string out;
  if (!readiness.reason_code.empty())
    out += "[" + readiness.reason_code + "]";
  if (!readiness.blocker_code.empty()) {
    if (!out.empty())
      out += " ";
    out += "[" + readiness.blocker_code + "]";
  }
  if (!readiness.detail.empty()) {
    if (!out.empty())
      out += " ";
    out += readiness.detail;
  }
  return out;
}

} // namespace studiocast::video
