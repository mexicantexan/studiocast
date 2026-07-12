#include "core/video/open_vulkan_matting_setup_policy.h"

namespace studiocast::video::detail {

OpenVulkanMattingSetupDecision DecideOpenVulkanMattingSetup(
    const OpenVulkanMattingSetupSnapshot &state, int frame_w, int frame_h,
    const std::string &requested_model_id) {
  OpenVulkanMattingSetupDecision decision;

  const bool request_changed =
      requested_model_id != state.active_requested_model_id;
  const bool missing_resolved_model =
      !state.has_model_pack || !state.has_matting_session ||
      state.active_model_id.empty();
  decision.scan_model_registry = request_changed || missing_resolved_model;
  decision.recreate_session = decision.scan_model_registry;

  const bool frame_size_changed =
      state.session_frame_w != frame_w || state.session_frame_h != frame_h;
  decision.initialize_session =
      decision.recreate_session || !state.session_initialized ||
      frame_size_changed;
  decision.warmup_session =
      decision.initialize_session || !state.session_warmed;

  return decision;
}

} // namespace studiocast::video::detail
