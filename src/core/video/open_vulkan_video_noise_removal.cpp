#include "core/video/open_vulkan_video_noise_removal.h"

namespace studiocast::video {

OpenVulkanVideoNoiseRemovalCapabilityFacts
CurrentOpenVulkanVideoNoiseRemovalFacts() {
  OpenVulkanVideoNoiseRemovalCapabilityFacts facts;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
  facts.backend_compiled = true;
#endif
  return facts;
}

OpenVulkanVideoNoiseRemovalReadiness
EvaluateOpenVulkanVideoNoiseRemovalReadiness(
    const OpenVulkanVideoNoiseRemovalCapabilityFacts &facts) {
  OpenVulkanVideoNoiseRemovalReadiness readiness;
  readiness.facts = facts;
  readiness.production_ready =
      facts.backend_compiled && facts.live_stage_implemented &&
      facts.production_adapter_available &&
      facts.vulkan_inference_provider_available &&
      facts.non_cpu_device_selected && facts.compute_queue_available &&
      facts.context_healthy && facts.shared_device_imported &&
      facts.queue_ownership_explicit && facts.model_pack_selected &&
      facts.artifact_contract_validated &&
      facts.fully_device_resident_tensor_io &&
      facts.device_resident_preprocess && facts.device_resident_postprocess &&
      facts.warmup_complete && facts.synchronization_contract_validated &&
      facts.bounded_reusable_allocations &&
      facts.temporal_history_device_resident &&
      facts.temporal_history_bounded && facts.history_reset_on_disable &&
      facts.history_reset_on_reconfigure &&
      facts.capture_sequence_discontinuity_reset && facts.parity_validated &&
      facts.dispatch_count > 0 && facts.cpu_readback_count == 0 &&
      facts.cpu_fallback_count == 0;

  if (!readiness.production_ready) {
    readiness.reason_code = kOpenVulkanVideoNoiseRemovalUnavailableReason;
    readiness.blocker_code =
        kOpenVulkanVideoNoiseRemovalRuntimeUnavailableReason;
    readiness.detail = kOpenVulkanVideoNoiseRemovalUnavailableDetail;
  }
  return readiness;
}

std::string FormatOpenVulkanVideoNoiseRemovalReadiness(
    const OpenVulkanVideoNoiseRemovalReadiness &readiness) {
  if (readiness.production_ready)
    return "Open Vulkan video noise removal is production ready.";

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
