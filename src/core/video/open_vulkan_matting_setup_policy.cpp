#include "core/video/open_vulkan_matting_setup_policy.h"

namespace studiocast::video::detail {

OpenVulkanMattingSetupDecision
DecideOpenVulkanMattingSetup(const OpenVulkanMattingSetupSnapshot &state,
                             int frame_w, int frame_h,
                             const std::string &requested_model_id) {
  (void)frame_w;
  (void)frame_h;
  OpenVulkanMattingSetupDecision decision;

  const bool request_changed =
      requested_model_id != state.active_requested_model_id;
  const bool context_changed =
      state.session_context_id != 0 && state.session_context_generation != 0 &&
      (state.session_context_id != state.current_context_id ||
       state.session_context_generation != state.current_context_generation);
  if (state.failure_latched && !request_changed && !context_changed) {
    decision.reuse_latched_failure = true;
    return decision;
  }
  const bool missing_resolved_model =
      !state.has_model_pack || state.active_model_id.empty();
  decision.scan_model_registry = request_changed || missing_resolved_model;
  decision.recreate_session = decision.scan_model_registry ||
                              !state.has_matting_session || context_changed;

  decision.initialize_session =
      decision.recreate_session || !state.session_initialized;
  decision.warmup_session =
      decision.initialize_session || !state.session_warmed;

  return decision;
}

} // namespace studiocast::video::detail
