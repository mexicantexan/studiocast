#include <iostream>
#include <string>

#include "core/video/open_vulkan_matting_setup_policy.h"

namespace {

using studiocast::video::detail::DecideOpenVulkanMattingSetup;
using studiocast::video::detail::OpenVulkanMattingSetupSnapshot;

bool Expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

bool TestFirstUseDoesSetupWork() {
  const auto d = DecideOpenVulkanMattingSetup(OpenVulkanMattingSetupSnapshot{},
                                              1280, 720, "");
  return Expect(d.scan_model_registry, "first use should scan registry") &&
         Expect(d.recreate_session, "first use should create session") &&
         Expect(d.initialize_session, "first use should initialize session") &&
         Expect(d.warmup_session, "first use should warm session");
}

bool TestRepeatedFramesDoNotRepeatSetupWork() {
  OpenVulkanMattingSetupSnapshot state;
  state.has_model_pack = true;
  state.has_matting_session = true;
  state.session_initialized = true;
  state.session_warmed = true;
  state.session_frame_w = 1280;
  state.session_frame_h = 720;
  state.active_model_id = "modnet-webnn-256-fp32";
  state.active_requested_model_id = "";

  const auto d = DecideOpenVulkanMattingSetup(state, 1280, 720, "");
  return Expect(!d.scan_model_registry,
                "steady-state frame should not scan model registry") &&
         Expect(!d.recreate_session,
                "steady-state frame should not recreate session") &&
         Expect(!d.initialize_session,
                "steady-state frame should not initialize session") &&
         Expect(!d.warmup_session,
                "steady-state frame should not warm session");
}

bool TestModelIdChangeTriggersSetupWork() {
  OpenVulkanMattingSetupSnapshot state;
  state.has_model_pack = true;
  state.has_matting_session = true;
  state.session_initialized = true;
  state.session_warmed = true;
  state.session_frame_w = 1280;
  state.session_frame_h = 720;
  state.active_model_id = "modnet-webnn-256-fp32";
  state.active_requested_model_id = "";

  const auto d =
      DecideOpenVulkanMattingSetup(state, 1280, 720, "birefnet_lite");
  return Expect(d.scan_model_registry,
                "model id change should scan model registry") &&
         Expect(d.recreate_session,
                "model id change should recreate session") &&
         Expect(d.initialize_session,
                "model id change should initialize session") &&
         Expect(d.warmup_session, "model id change should warm session");
}

bool TestFrameSizeChangeDoesNotRepeatRuntimeSetup() {
  OpenVulkanMattingSetupSnapshot state;
  state.has_model_pack = true;
  state.has_matting_session = true;
  state.session_initialized = true;
  state.session_warmed = true;
  state.session_frame_w = 1280;
  state.session_frame_h = 720;
  state.active_model_id = "modnet-webnn-256-fp32";
  state.active_requested_model_id = "";

  const auto d = DecideOpenVulkanMattingSetup(state, 1920, 1080, "");
  return Expect(!d.scan_model_registry,
                "frame size change should not scan model registry") &&
         Expect(!d.recreate_session,
                "frame size change should not recreate session") &&
         Expect(!d.initialize_session,
                "frame size change should reuse fixed model tensors") &&
         Expect(!d.warmup_session,
                "frame size change should not repeat warmup");
}

bool TestContextGenerationChangeRecreatesWithoutRegistryScan() {
  OpenVulkanMattingSetupSnapshot state;
  state.has_model_pack = true;
  state.has_matting_session = true;
  state.session_initialized = true;
  state.session_warmed = true;
  state.active_model_id = "ncnn-matting";
  state.active_requested_model_id = "";
  state.session_context_id = 17;
  state.session_context_generation = 2;
  state.current_context_id = 17;
  state.current_context_generation = 3;

  const auto d = DecideOpenVulkanMattingSetup(state, 1280, 720, "");
  return Expect(!d.scan_model_registry,
                "context change should reuse validated model pack") &&
         Expect(d.recreate_session,
                "context generation change must recreate session") &&
         Expect(d.initialize_session && d.warmup_session,
                "new context must initialize and warm once");
}

bool TestLatchedFailureIsReusedUntilContractChanges() {
  OpenVulkanMattingSetupSnapshot state;
  state.failure_latched = true;
  state.active_requested_model_id = "ncnn-matting";
  state.session_context_id = 17;
  state.session_context_generation = 2;
  state.current_context_id = 17;
  state.current_context_generation = 2;

  const auto stable =
      DecideOpenVulkanMattingSetup(state, 1280, 720, "ncnn-matting");
  const auto changed =
      DecideOpenVulkanMattingSetup(state, 1280, 720, "other-matting");
  return Expect(stable.reuse_latched_failure,
                "stable failure must be reused without frame-loop retry") &&
         Expect(!stable.scan_model_registry && !stable.recreate_session &&
                    !stable.initialize_session && !stable.warmup_session,
                "latched failure reuse must do no setup work") &&
         Expect(!changed.reuse_latched_failure && changed.scan_model_registry,
                "model contract change may retry setup");
}

bool TestRepeatedFrameProcessingKeepsSetupCountersStable() {
  OpenVulkanMattingSetupSnapshot state;
  int scan_calls = 0;
  int recreate_calls = 0;
  int initialize_calls = 0;
  int warmup_calls = 0;

  const auto process_frame = [&](int width, int height,
                                 const std::string &requested_model_id) {
    const auto d =
        DecideOpenVulkanMattingSetup(state, width, height, requested_model_id);
    if (d.scan_model_registry) {
      ++scan_calls;
    }
    if (d.recreate_session) {
      ++recreate_calls;
      state.has_model_pack = true;
      state.has_matting_session = true;
      state.active_model_id = requested_model_id.empty()
                                  ? "modnet-webnn-256-fp32"
                                  : requested_model_id;
      state.active_requested_model_id = requested_model_id;
      state.session_initialized = false;
      state.session_warmed = false;
    }
    if (d.initialize_session) {
      ++initialize_calls;
      state.session_initialized = true;
      state.session_warmed = false;
      state.session_frame_w = width;
      state.session_frame_h = height;
    }
    if (d.warmup_session) {
      ++warmup_calls;
      state.session_warmed = true;
    }
  };

  process_frame(1280, 720, "");
  for (int i = 0; i < 5; ++i) {
    process_frame(1280, 720, "");
  }

  return Expect(scan_calls == 1,
                "repeated frames should keep registry scans at first-use "
                "count") &&
         Expect(recreate_calls == 1,
                "repeated frames should keep session creates at first-use "
                "count") &&
         Expect(initialize_calls == 1,
                "repeated frames should keep session init at first-use "
                "count") &&
         Expect(warmup_calls == 1,
                "repeated frames should keep session warmup at first-use "
                "count");
}

} // namespace

namespace studiocast::tests {

bool TestOpenVulkanMattingFirstUseDoesSetupWork() {
  return TestFirstUseDoesSetupWork();
}

bool TestOpenVulkanMattingRepeatedFramesDoNotRepeatSetupWork() {
  return TestRepeatedFramesDoNotRepeatSetupWork();
}

bool TestOpenVulkanMattingModelIdChangeTriggersSetupWork() {
  return TestModelIdChangeTriggersSetupWork();
}

bool TestOpenVulkanMattingFrameSizeChangeOnlyRefreshesSession() {
  return TestFrameSizeChangeDoesNotRepeatRuntimeSetup();
}

bool TestOpenVulkanMattingRepeatedFrameProcessingKeepsSetupCountersStable() {
  return TestRepeatedFrameProcessingKeepsSetupCountersStable();
}

bool TestOpenVulkanMattingContextGenerationChangeRecreatesSession() {
  return TestContextGenerationChangeRecreatesWithoutRegistryScan();
}

bool TestOpenVulkanMattingLatchedFailureIsReused() {
  return TestLatchedFailureIsReusedUntilContractChanges();
}

} // namespace studiocast::tests
