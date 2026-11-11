#pragma once

#include <cstdint>
#include <string>

namespace studiocast::video::detail {

struct OpenVulkanMattingSetupSnapshot {
  bool has_model_pack = false;
  bool has_matting_session = false;
  bool session_initialized = false;
  bool session_warmed = false;
  int session_frame_w = 0;
  int session_frame_h = 0;
  bool failure_latched = false;
  std::uint64_t session_context_id = 0;
  std::uint64_t session_context_generation = 0;
  std::uint64_t current_context_id = 0;
  std::uint64_t current_context_generation = 0;
  std::string active_model_id;
  std::string active_requested_model_id;
};

struct OpenVulkanMattingSetupDecision {
  bool scan_model_registry = false;
  bool recreate_session = false;
  bool initialize_session = false;
  bool warmup_session = false;
  bool reuse_latched_failure = false;
};

OpenVulkanMattingSetupDecision
DecideOpenVulkanMattingSetup(const OpenVulkanMattingSetupSnapshot &state,
                             int frame_w, int frame_h,
                             const std::string &requested_model_id);

} // namespace studiocast::video::detail
