#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include "core/video/effects/broadcast_effect_contract.h"
#include "core/video/effects/broadcast_effect_rules.h"
#include "core/video/scaling_policy.h"
#include "core/video/virtual_camera_service.h"

namespace studiocast::tests {
bool TestCameraPipelineIdenticalEffectsPreserveRuntimeGeneration();
bool TestCameraPipelineExplicitEffectsRefreshIsSeparateInvalidation();
bool TestLatestFrameWinsOverwritesPendingWithBlockedProcessor();
bool TestLatestFrameWinsStopWakesAndJoins();
bool TestLatestFrameWinsGenerationRejectsStaleResults();
bool TestLatestFrameWinsStatsCountersAndLastError();
bool TestFastDvdnetDenoiseTensorContractIsDeclared();
bool TestYunetFaceDetectionCpuTensorTailContractIsDeclared();
bool TestYunetExplicitVulkanProviderAndTensorPolicyIsFailClosed();
bool TestOpenVideoEyeContactCpuTensorTailContractIsDeclared();
bool TestOpenVulkanEyeContactCapabilityFactsAreFailClosed();
bool TestOpenVulkanVideoNoiseRemovalCapabilityFactsAreFailClosed();
bool TestOpenVulkanVideoNoiseRemovalTemporalReadinessGatesAreExact();
bool TestFrameArtifactCacheReusesCompatibleMatteWithinFrame();
bool TestFrameArtifactCacheReusesCompatibleMaxineMatteWithinFrame();
bool TestFrameArtifactCacheSeparatesIncompatibleMatteKeys();
bool TestFrameArtifactCacheInvalidatesMatteOnNewFrame();
bool TestFrameAnalysisCacheRetainsFaceAnalysisWithinFrame();
bool TestFrameArtifactCachePrecomputedMatteKeysPreserveCompatibility();
bool TestFrameArtifactCacheDeviceStorageAliasesPreserveCompatibility();
bool TestFrameArtifactCacheReusesDeviceMatteAcrossCombinedEffects();
bool TestCudaContextClassifierPrefersCurrentContext();
bool TestSignedInt32PtxPitchContractIsNoGpuSafe();
bool TestOpenCudaBoxBlurRadiusBoundsAreNoGpuSafe();
bool TestOpenCudaF32ResizeBorderContractIsReplicateNoGpuSafe();
bool TestOpenCudaAlphaClampAndSolidBgrContractNoGpuSafe();
bool TestCudaU8ResizeRoundingContractIsNoGpuSafe();
bool TestCudaResizeAvailabilityProbeIsThreadSafe();
bool TestOpenVulkanMattingFirstUseDoesSetupWork();
bool TestOpenVulkanMattingRepeatedFramesDoNotRepeatSetupWork();
bool TestOpenVulkanMattingModelIdChangeTriggersSetupWork();
bool TestOpenVulkanMattingFrameSizeChangeOnlyRefreshesSession();
bool TestOpenVulkanMattingRepeatedFrameProcessingKeepsSetupCountersStable();
bool TestOpenVulkanMattingContextGenerationChangeRecreatesSession();
bool TestOpenVulkanMattingLatchedFailureIsReused();
bool TestReplaceBackgroundRepeatedFramePathDoesNotStatImage();
bool TestReplaceBackgroundSamePathExplicitRefreshUsesPreparedMtime();
bool TestReplaceBackgroundFrameSizeChangeOnlyRefreshesResizedBackground();
bool TestV4l2CapturePreferenceTreats720pAsMjpegWorthy();
bool TestV4l2YuyvRequestTriesMjpegFirstAtHdWhenPreferred();
bool TestV4l2YuyvRequestFallsBackToMjpegAfterYuyvAtLowResolution();
bool TestV4l2YuyvRequestDoesNotAddMjpegFallbackWhenPreferenceDisabled();
bool TestV4l2UnsupportedFormatsAreSkippedWithoutDuplicates();
bool TestV4l2ExplicitMjpegRequestDoesNotFallBackToYuyvInsideOpenOrder();
bool TestV4l2FakeNegotiationUsesOrderedFallback();
bool TestV4l2MjpegDecodeFailureFallsBackToRawOnce();
bool TestYuyvToRgb24MatchesBt601AndPreservesPadding();
bool TestYuyvToRgb24BackendsMatchScalarReference();
bool TestRgb24ToYuyvMatchesBt601WithinChromaRounding();
bool TestRgb24ToYuyvBackendsMatchScalarReference();
bool TestRgb24ToYuyvPublicPathMatchesScalarWithScratchVariants();
bool TestRawYuyvPassthroughPredicateIsStrict();
bool TestRawYuyvPassthroughPreservesExactLayouts();
bool TestRawYuyvPassthroughCopiesActiveRowsAndZerosPadding();
bool TestRawYuyvPassthroughRejectsUnsafeFrames();
bool TestRawYuyvPassthroughCountersTrackActualCallSites();
bool TestRgb24Bgr24BackendsMatchScalarAndPreservePadding();
bool TestRgb24Bgr24PublicPathMatchesScalarInPlace();
bool TestResizeRgb24BilinearPreservesActivePixelsAndZerosPadding();
bool TestResizeRgb24BilinearHandlesDegenerateAxesAndPlanReuse();
bool TestBackgroundRemoveCpuMatchesReferenceAndPreservesPadding();
bool TestBackgroundBlurCpuMatchesReferenceAndPreservesPadding();
bool TestMirrorCpuReferenceCoversPaddingOddEvenDegenerateAndDoubleMirror();
} // namespace studiocast::tests

namespace {

using studiocast::video::CameraPipelineConfig;
using studiocast::video::CameraPipelineRunner;
using studiocast::video::CameraPipelineStatus;
using studiocast::video::LiveEffectBackendAttribution;
using studiocast::video::OptionalEffectBreaker;
using studiocast::video::VideoConsumerSnapshot;
using studiocast::video::VirtualCameraService;
using studiocast::video::VirtualCameraServiceConfig;
using studiocast::video::VirtualCameraServiceHooks;

using namespace std::chrono_literals;

bool WaitUntil(const std::function<bool()> &pred,
               std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred())
      return true;
    std::this_thread::sleep_for(1ms);
  }
  return pred();
}

class FakeCameraPipeline final : public CameraPipelineRunner {
public:
  bool Start(const CameraPipelineConfig &cfg, std::string *error) override {
    start_calls.fetch_add(1, std::memory_order_relaxed);
    if (fail_start.load(std::memory_order_relaxed)) {
      if (error)
        *error = "synthetic camera start failure";
      std::lock_guard<std::mutex> lock(mu_);
      status_.running = false;
      status_.starting = false;
      status_.last_error = "synthetic camera start failure";
      return false;
    }

    if (error)
      error->clear();
    std::lock_guard<std::mutex> lock(mu_);
    status_.running = true;
    status_.starting = false;
    status_.effects_backends.clear();
    status_.input_device = cfg.input_device.empty()
                               ? std::string("/dev/video-real")
                               : cfg.input_device;
    status_.output_device = cfg.output_device;
    status_.output.width = cfg.width;
    status_.output.height = cfg.height;
    status_.output.fps = cfg.fps;
    status_.output.fps_num = 1;
    status_.output.fps_den = cfg.fps;
    status_.output.format = cfg.output_format;
    status_.last_error.clear();
    applied_effects_ = cfg.effects;
    return true;
  }

  void Stop() override {
    stop_calls.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mu_);
    status_.running = false;
    status_.starting = false;
  }

  bool EnsureOutputOpen(const CameraPipelineConfig &cfg,
                        std::string *error) override {
    ensure_output_calls.fetch_add(1, std::memory_order_relaxed);
    if (fail_ensure_output.load(std::memory_order_relaxed)) {
      if (error)
        *error = "synthetic output open failure";
      return false;
    }

    if (error)
      error->clear();
    std::lock_guard<std::mutex> lock(mu_);
    status_.output_device = cfg.output_device;
    status_.output.width = cfg.width;
    status_.output.height = cfg.height;
    status_.output.fps = cfg.fps;
    status_.output.fps_num = 1;
    status_.output.fps_den = cfg.fps;
    status_.output.format = cfg.output_format;
    return true;
  }

  void CloseOutput() override {
    close_output_calls.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mu_);
    status_.output_device.clear();
    status_.output = {};
  }

  CameraPipelineStatus Status() const override {
    std::lock_guard<std::mutex> lock(mu_);
    return status_;
  }

  void SetEffects(const studiocast::video::effects::BroadcastCameraEffects
                      &effects) override {
    set_effects_calls.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mu_);
    applied_effects_ = effects;
  }

  void SetMirrorEnabled(bool) override {}

  void SetDegradedEffect(CameraPipelineStatus::DegradedEffect degraded,
                         std::string effects_note) {
    std::lock_guard<std::mutex> lock(mu_);
    status_.degraded_effect = std::move(degraded);
    status_.effects_note = std::move(effects_note);
  }

  void SetEffectsBackends(std::string effects_backends) {
    std::lock_guard<std::mutex> lock(mu_);
    status_.effects_backends = std::move(effects_backends);
  }

  studiocast::video::effects::BroadcastCameraEffects AppliedEffects() const {
    std::lock_guard<std::mutex> lock(mu_);
    return applied_effects_;
  }

  std::atomic<int> start_calls{0};
  std::atomic<int> stop_calls{0};
  std::atomic<int> ensure_output_calls{0};
  std::atomic<int> close_output_calls{0};
  std::atomic<int> set_effects_calls{0};
  std::atomic<bool> fail_start{false};
  std::atomic<bool> fail_ensure_output{false};

private:
  mutable std::mutex mu_;
  CameraPipelineStatus status_{};
  studiocast::video::effects::BroadcastCameraEffects applied_effects_{};
};

struct ServiceHarness {
  FakeCameraPipeline *pipeline = nullptr;
  std::atomic<bool> consumer_present{false};
  std::atomic<bool> device_exists{true};
  std::atomic<bool> detection_error{false};
  std::atomic<int> consumer_probe_calls{0};

  VirtualCameraServiceHooks Hooks(std::chrono::milliseconds sleep = 1ms) {
    VirtualCameraServiceHooks hooks;
    hooks.sleep_for = [sleep](std::chrono::milliseconds) {
      std::this_thread::sleep_for(sleep);
    };
    hooks.create_pipeline = [&] {
      auto fake = std::make_unique<FakeCameraPipeline>();
      pipeline = fake.get();
      return fake;
    };
    hooks.choose_output_device = [&](std::string *error) {
      if (!device_exists.load(std::memory_order_relaxed)) {
        if (error)
          *error = "synthetic loopback missing";
        return std::string();
      }
      if (error)
        error->clear();
      return std::string("/tmp/studiocast-test-video10");
    };
    hooks.output_device_exists = [&](const std::string &, std::string *error) {
      if (!device_exists.load(std::memory_order_relaxed)) {
        if (error)
          *error = "synthetic loopback missing";
        return false;
      }
      if (error)
        error->clear();
      return true;
    };
    hooks.detect_consumers = [&](const std::string &, int) {
      consumer_probe_calls.fetch_add(1, std::memory_order_relaxed);
      VideoConsumerSnapshot out;
      if (detection_error.load(std::memory_order_relaxed)) {
        out.error = "synthetic consumer scan failure";
        return out;
      }
      out.present = consumer_present.load(std::memory_order_relaxed);
      out.count = out.present ? 1 : 0;
      return out;
    };
    return hooks;
  }
};

VirtualCameraServiceConfig TestConfig() {
  VirtualCameraServiceConfig cfg;
  cfg.enabled = true;
  cfg.consumer_poll_ms = 1;
  cfg.start_grace_ms = 0;
  cfg.stop_grace_ms = 0;
  cfg.min_run_ms = 0;
  cfg.pipeline.output_device = "/tmp/studiocast-test-video10";
  cfg.pipeline.width = 640;
  cfg.pipeline.height = 480;
  cfg.pipeline.fps = 30;
  return cfg;
}

bool TestLiveEffectBackendAttributionRequiresSuccessfulFrameCommit() {
  LiveEffectBackendAttribution attribution;
  const std::vector<std::string> order = {"virtual_background.blur", "vignette",
                                          "mirror"};

  attribution.BeginFrame();
  attribution.MarkEffectSucceeded("mirror", "open_vulkan");
  if (!attribution.active_backends().empty()) {
    std::cerr << "attempted effect leaked before the first successful frame\n";
    return false;
  }

  if (!attribution.CommitSuccessfulFrame(order) ||
      attribution.active_backends() != "mirror:open_vulkan") {
    std::cerr << "first successful frame did not publish mirror attribution\n";
    return false;
  }

  // A circuit-broken frame succeeds as pass-through. Removing the effect must
  // be visible immediately and the successful frame commits the empty map.
  attribution.BeginFrame();
  if (!attribution.RemoveEffect("mirror") ||
      !attribution.active_backends().empty() ||
      attribution.CommitSuccessfulFrame(order)) {
    std::cerr << "circuit-broken effect remained active\n";
    return false;
  }

  // Retry attempts are not evidence until the output writer accepts the frame.
  attribution.BeginFrame();
  attribution.MarkEffectSucceeded("mirror", "open_vulkan");
  attribution.DiscardFrame();
  if (!attribution.active_backends().empty()) {
    std::cerr << "discarded retry republished effect attribution\n";
    return false;
  }

  attribution.BeginFrame();
  attribution.MarkEffectSucceeded("mirror", "open_vulkan");
  attribution.MarkEffectSucceeded("vignette", "open_vulkan");
  if (!attribution.CommitSuccessfulFrame(order) ||
      attribution.active_backends() !=
          "vignette:open_vulkan,mirror:open_vulkan") {
    std::cerr << "recovered effects were not published in canonical order\n";
    return false;
  }

  attribution.BeginFrame();
  attribution.MarkEffectSucceeded("mirror", "open_vulkan");
  attribution.MarkEffectSucceeded("vignette", "open_vulkan");
  if (attribution.CommitSuccessfulFrame(order)) {
    std::cerr << "unchanged attribution was treated as a status transition\n";
    return false;
  }

  attribution.BeginFrame();
  attribution.MarkEffectSucceeded("mirror", "configured_only");
  if (!attribution.CommitSuccessfulFrame(order) ||
      !attribution.active_backends().empty()) {
    std::cerr << "unknown backend attribution did not fail closed\n";
    return false;
  }

  attribution.Clear();
  if (!attribution.active_backends().empty()) {
    std::cerr << "stop/restart clear did not remove active attribution\n";
    return false;
  }
  return true;
}

bool TestVideoServiceSanitizesEffectBackendsAcrossLifecycle() {
  ServiceHarness h;
  h.consumer_present.store(true, std::memory_order_relaxed);
  VirtualCameraService service(h.Hooks());
  auto cfg = TestConfig();

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }
  if (!WaitUntil([&] { return service.Status().pipeline_state == "running"; },
                 250ms)) {
    std::cerr << "pipeline did not reach running for attribution lifecycle\n";
    service.Stop();
    return false;
  }

  // Start is setup only. The fake mirrors CameraPipeline by publishing no map
  // until a first successful frame is injected.
  if (!service.Status().pipeline.effects_backends.empty()) {
    std::cerr << "start leaked effect attribution before a successful frame\n";
    service.Stop();
    return false;
  }
  h.pipeline->SetEffectsBackends("mirror:open_vulkan");
  if (!WaitUntil(
          [&] {
            return service.Status().pipeline.effects_backends ==
                   "mirror:open_vulkan";
          },
          100ms)) {
    std::cerr << "authoritative running attribution was suppressed\n";
    service.Stop();
    return false;
  }

  // A failed restart deliberately leaves the fake runner's old map in place;
  // the service must still publish an empty map while stopped/backing off.
  h.pipeline->fail_start.store(true, std::memory_order_relaxed);
  auto restarted = cfg;
  restarted.pipeline.width = 800;
  service.UpdateConfig(restarted);
  const bool backedOffWithoutStaleMap = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.pipeline_state == "backing_off" &&
               status.pipeline.effects_backends.empty();
      },
      750ms);
  service.Stop();
  if (!backedOffWithoutStaleMap) {
    std::cerr << "failed/backoff lifecycle leaked stale attribution\n";
    return false;
  }
  if (!service.Status().pipeline.effects_backends.empty()) {
    std::cerr << "stopped service leaked stale attribution\n";
    return false;
  }
  return true;
}

bool TestVideoServiceDeliversOnlyGenuineLiveEffectChanges() {
  ServiceHarness h;
  h.consumer_present.store(true, std::memory_order_relaxed);
  VirtualCameraService service(h.Hooks());
  auto cfg = TestConfig();

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }
  if (!WaitUntil([&] { return service.Status().pipeline_state == "running"; },
                 500ms)) {
    std::cerr << "pipeline did not reach running for stable effects test\n";
    service.Stop();
    return false;
  }

  const int stable_probe_baseline =
      h.consumer_probe_calls.load(std::memory_order_relaxed);
  if (!WaitUntil(
          [&] {
            return h.consumer_probe_calls.load(std::memory_order_relaxed) >=
                   stable_probe_baseline + 4;
          },
          500ms)) {
    std::cerr << "supervisor did not complete stable polls\n";
    service.Stop();
    return false;
  }
  if (h.pipeline->set_effects_calls.load(std::memory_order_relaxed) != 0) {
    std::cerr << "stable polls redundantly delivered effects after Start\n";
    service.Stop();
    return false;
  }

  cfg.pipeline.effects.mirror = true;
  service.UpdateConfig(cfg);
  if (!WaitUntil(
          [&] {
            return h.pipeline->set_effects_calls.load(
                       std::memory_order_relaxed) == 1 &&
                   h.pipeline->AppliedEffects().mirror;
          },
          500ms)) {
    std::cerr << "genuine live effect change was not delivered exactly once\n";
    service.Stop();
    return false;
  }

  const int changed_probe_baseline =
      h.consumer_probe_calls.load(std::memory_order_relaxed);
  service.UpdateConfig(cfg);
  const bool completed_stable_polls = WaitUntil(
      [&] {
        return h.consumer_probe_calls.load(std::memory_order_relaxed) >=
               changed_probe_baseline + 4;
      },
      500ms);
  const int final_set_effects_calls =
      h.pipeline->set_effects_calls.load(std::memory_order_relaxed);
  service.Stop();

  if (!completed_stable_polls) {
    std::cerr << "supervisor did not complete post-change stable polls\n";
    return false;
  }
  if (final_set_effects_calls != 1) {
    std::cerr << "identical post-change polls redelivered effects: calls="
              << final_set_effects_calls << "\n";
    return false;
  }
  return true;
}

bool TestVideoPipelineDoesNotStartWithoutConsumer() {
  ServiceHarness h;
  VirtualCameraService service(h.Hooks());
  auto cfg = TestConfig();

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool idle = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.pipeline_state == "idle_no_consumer" &&
               status.pipeline_idle_reason == "no_consumer" &&
               !status.pipeline_active_needed && !status.pipeline.running &&
               status.virtual_device_available &&
               h.pipeline->ensure_output_calls.load() > 0;
      },
      250ms);
  const auto status = service.Status();
  const int starts = h.pipeline->start_calls.load();
  service.Stop();

  if (!idle || starts != 0) {
    std::cerr << "video pipeline started or failed to idle without consumer; "
              << "starts=" << starts << " state='" << status.pipeline_state
              << "' idle='" << status.pipeline_idle_reason
              << "' virtual_available=" << status.virtual_device_available
              << "\n";
    return false;
  }
  return true;
}

bool TestVideoPipelineStartsWhenConsumerAppears() {
  ServiceHarness h;
  VirtualCameraService service(h.Hooks());
  auto cfg = TestConfig();

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] { return service.Status().pipeline_state == "idle_no_consumer"; },
          250ms)) {
    std::cerr << "video pipeline did not reach no-consumer idle state\n";
    service.Stop();
    return false;
  }

  h.consumer_present.store(true, std::memory_order_relaxed);
  const bool started = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.consumer_present && status.pipeline.running &&
               status.pipeline_active_needed &&
               status.pipeline_state == "running" &&
               h.pipeline->start_calls.load() == 1;
      },
      250ms);
  const auto status = service.Status();
  service.Stop();

  if (!started) {
    std::cerr << "video pipeline did not start after consumer appeared; "
              << "starts=" << h.pipeline->start_calls.load() << " state='"
              << status.pipeline_state
              << "' running=" << status.pipeline.running << "\n";
    return false;
  }
  return true;
}

bool TestVideoPipelineStopsWhenConsumerDisappears() {
  ServiceHarness h;
  h.consumer_present.store(true, std::memory_order_relaxed);
  VirtualCameraService service(h.Hooks());
  auto cfg = TestConfig();
  cfg.stop_grace_ms = 20;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil([&] { return service.Status().pipeline.running; }, 250ms)) {
    std::cerr << "video pipeline did not start for initial consumer\n";
    service.Stop();
    return false;
  }

  h.consumer_present.store(false, std::memory_order_relaxed);
  const bool stopped = WaitUntil(
      [&] {
        const auto status = service.Status();
        return h.pipeline->stop_calls.load() >= 1 && !status.pipeline.running &&
               status.pipeline_state == "idle_no_consumer" &&
               !status.pipeline_active_needed;
      },
      500ms);
  const auto status = service.Status();
  service.Stop();

  if (!stopped) {
    std::cerr << "video pipeline did not stop after consumer disappeared; "
              << "stops=" << h.pipeline->stop_calls.load() << " state='"
              << status.pipeline_state
              << "' running=" << status.pipeline.running << "\n";
    return false;
  }
  return true;
}

bool TestVideoGraceWindowAbsorbsConsumerFlapping() {
  ServiceHarness h;
  h.consumer_present.store(true, std::memory_order_relaxed);
  VirtualCameraService service(h.Hooks());
  auto cfg = TestConfig();
  cfg.stop_grace_ms = 80;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil([&] { return service.Status().pipeline.running; }, 250ms)) {
    std::cerr << "video pipeline did not start before flap test\n";
    service.Stop();
    return false;
  }

  for (int i = 0; i < 3; ++i) {
    h.consumer_present.store(false, std::memory_order_relaxed);
    std::this_thread::sleep_for(20ms);
    const auto absent = service.Status();
    if (!absent.pipeline.running ||
        h.pipeline->stop_calls.load(std::memory_order_relaxed) != 0) {
      std::cerr << "video pipeline stopped inside grace window; i=" << i
                << " stops=" << h.pipeline->stop_calls.load() << " state='"
                << absent.pipeline_state << "'\n";
      service.Stop();
      return false;
    }

    h.consumer_present.store(true, std::memory_order_relaxed);
    if (!WaitUntil([&] { return service.Status().consumer_present; }, 100ms)) {
      std::cerr << "video consumer did not recover during flap test\n";
      service.Stop();
      return false;
    }
  }

  h.consumer_present.store(false, std::memory_order_relaxed);
  const bool stopped = WaitUntil(
      [&] {
        return h.pipeline->stop_calls.load(std::memory_order_relaxed) == 1 &&
               service.Status().pipeline_state == "idle_no_consumer";
      },
      500ms);
  service.Stop();

  if (!stopped || h.pipeline->start_calls.load() != 1) {
    std::cerr << "video pipeline churned during consumer flapping; starts="
              << h.pipeline->start_calls.load()
              << " stops=" << h.pipeline->stop_calls.load() << "\n";
    return false;
  }
  return true;
}

bool TestVideoConsumerDetectionErrorSurfacesWithoutStarting() {
  ServiceHarness h;
  h.detection_error.store(true, std::memory_order_relaxed);
  VirtualCameraService service(h.Hooks());
  auto cfg = TestConfig();

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool surfaced = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.pipeline_state == "consumer_detection_error" &&
               status.consumer_error.find("synthetic consumer scan failure") !=
                   std::string::npos &&
               h.pipeline->start_calls.load() == 0;
      },
      250ms);
  const auto status = service.Status();
  service.Stop();

  if (!surfaced) {
    std::cerr << "consumer detection error was not surfaced cleanly; state='"
              << status.pipeline_state << "' consumer_error='"
              << status.consumer_error
              << "' starts=" << h.pipeline->start_calls.load() << "\n";
    return false;
  }
  return true;
}

bool TestVideoOutputOpenFailureDoesNotStartPipeline() {
  ServiceHarness h;
  h.consumer_present.store(true, std::memory_order_relaxed);
  VirtualCameraService service(h.Hooks());
  auto cfg = TestConfig();
  h.pipeline->fail_ensure_output.store(true, std::memory_order_relaxed);

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool unavailable = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.pipeline_state == "device_unavailable" &&
               !status.virtual_device_available &&
               status.virtual_device_error.find("Output open failed") !=
                   std::string::npos &&
               !status.pipeline_active_needed &&
               h.pipeline->ensure_output_calls.load() > 0 &&
               h.pipeline->start_calls.load() == 0;
      },
      250ms);
  const auto status = service.Status();
  const int starts = h.pipeline->start_calls.load();
  service.Stop();

  if (!unavailable || starts != 0) {
    std::cerr << "output open failure still allowed pipeline start; starts="
              << starts << " state='" << status.pipeline_state
              << "' active_needed=" << status.pipeline_active_needed
              << " virtual_available=" << status.virtual_device_available
              << " virtual_error='" << status.virtual_device_error << "'\n";
    return false;
  }
  return true;
}

bool TestVideoStartFailureBacksOff() {
  ServiceHarness h;
  h.consumer_present.store(true, std::memory_order_relaxed);
  VirtualCameraService service(h.Hooks());
  auto cfg = TestConfig();
  h.pipeline->fail_start.store(true, std::memory_order_relaxed);

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool backedOff = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.pipeline_state == "backing_off" &&
               status.next_start_retry_ms > 0 &&
               status.last_error.find("Pipeline start failed") !=
                   std::string::npos &&
               h.pipeline->start_calls.load() == 1;
      },
      250ms);
  const int startsAfterFirstFailure = h.pipeline->start_calls.load();
  std::this_thread::sleep_for(40ms);
  const int startsAfterWait = h.pipeline->start_calls.load();
  service.Stop();

  if (!backedOff || startsAfterFirstFailure != startsAfterWait) {
    std::cerr << "video start failure did not back off; backedOff=" << backedOff
              << " starts_first=" << startsAfterFirstFailure
              << " starts_later=" << startsAfterWait << "\n";
    return false;
  }
  return true;
}

bool TestVideoStartFailureClearsAfterRecovery() {
  ServiceHarness h;
  h.consumer_present.store(true, std::memory_order_relaxed);
  VirtualCameraService service(h.Hooks());
  auto cfg = TestConfig();
  h.pipeline->fail_start.store(true, std::memory_order_relaxed);

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] { return service.Status().pipeline_state == "backing_off"; },
          250ms)) {
    std::cerr << "video pipeline did not enter backoff before recovery test\n";
    service.Stop();
    return false;
  }

  h.pipeline->fail_start.store(false, std::memory_order_relaxed);
  auto cfg2 = cfg;
  cfg2.pipeline.width = 800;
  service.UpdateConfig(cfg2);

  // The built-in retry backoff is two seconds; wait only until the retry opens.
  const bool recovered = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.pipeline_state == "running" &&
               status.last_error.empty() && h.pipeline->start_calls.load() >= 2;
      },
      2500ms);
  const auto status = service.Status();
  service.Stop();

  if (!recovered) {
    std::cerr << "video pipeline recovery left stale error; state='"
              << status.pipeline_state << "' last_error='" << status.last_error
              << "' starts=" << h.pipeline->start_calls.load() << "\n";
    return false;
  }
  return true;
}

bool TestVideoConfigRestartTransitionNameIsStable() {
  ServiceHarness h;
  h.consumer_present.store(true, std::memory_order_relaxed);
  VirtualCameraService service(h.Hooks(5ms));
  auto cfg = TestConfig();

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil([&] { return service.Status().pipeline.running; }, 250ms)) {
    std::cerr << "video pipeline did not start before config restart test\n";
    service.Stop();
    return false;
  }

  auto updated = cfg;
  updated.pipeline.width = 800;
  service.UpdateConfig(updated);

  const bool sawStop = WaitUntil(
      [&] { return service.Status().last_transition == "stop_config_restart"; },
      500ms);
  service.Stop();

  if (!sawStop) {
    std::cerr << "config restart transition was not exposed without quotes\n";
    return false;
  }
  return true;
}

bool TestVideoOutputFormatChangeRestartsPipeline() {
  ServiceHarness h;
  h.consumer_present.store(true, std::memory_order_relaxed);
  VirtualCameraService service(h.Hooks(5ms));
  auto cfg = TestConfig();
  cfg.pipeline.output_format = studiocast::video::PixelFormat::rgb24;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil([&] { return service.Status().pipeline.running; }, 250ms)) {
    std::cerr << "video pipeline did not start before format restart test\n";
    service.Stop();
    return false;
  }

  auto updated = cfg;
  updated.pipeline.output_format = studiocast::video::PixelFormat::yuyv;
  service.UpdateConfig(updated);

  const bool restarted = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.pipeline_config_restarts >= 1 &&
               h.pipeline->start_calls.load() >= 2 &&
               status.pipeline.output.format ==
                   studiocast::video::PixelFormat::yuyv;
      },
      750ms);
  const auto status = service.Status();
  service.Stop();

  if (!restarted) {
    std::cerr << "output format change did not restart pipeline; restarts="
              << status.pipeline_config_restarts
              << " starts=" << h.pipeline->start_calls.load() << "\n";
    return false;
  }
  return true;
}

bool TestOptionalVideoEffectsFailOpenCooldownAndRetry() {
  namespace contract = studiocast::video::effects::contract;

  struct Scenario {
    const char *name;
    std::string_view effect_id;
    const char *backend;
    const char *reason;
  };

  const std::array<Scenario, 4> scenarios{{
      {"denoise", contract::kEffectIdVideoNoiseRemoval, "maxine",
       "Maxine video noise removal failed: synthetic denoise failure"},
      {"relight", contract::kEffectIdVirtualKeyLight, "maxine",
       "Maxine relighting failed: synthetic relight failure"},
      {"virtual background", contract::kEffectIdVirtualBackgroundBlur, "maxine",
       "Maxine virtual background failed: synthetic VB failure"},
      {"auto frame", contract::kEffectIdAutoFrame, "maxine_ar_cuda",
       "Maxine auto frame failed: synthetic auto-frame failure"},
  }};

  for (const auto &scenario : scenarios) {
    OptionalEffectBreaker breaker;
    int order = 0;
    int attempts = 0;
    int skipped_cooldown_frames = 0;
    int output_frames = 0;
    bool saw_first_cooldown = false;
    bool saw_retry_ready = false;
    bool saw_second_cooldown = false;
    bool saw_success_reset = false;

    for (std::uint64_t frame = 10; frame <= 100; ++frame) {
      ++output_frames;

      if (breaker.MarkRetryReadyIfDue(frame)) {
        const auto retry = breaker.ToStatus(frame);
        if (!retry.active || retry.state != "retry_ready" ||
            retry.cooldown_frames != 0) {
          std::cerr << scenario.name
                    << " retry-ready status was malformed: active="
                    << retry.active << " state='" << retry.state
                    << "' cooldown=" << retry.cooldown_frames << "\n";
          return false;
        }
        saw_retry_ready = true;
      }

      if (!breaker.AllowsAttempt(frame)) {
        ++skipped_cooldown_frames;
        continue;
      }

      ++attempts;
      if (attempts <= 2) {
        breaker.OnFailure(scenario.effect_id, scenario.backend, scenario.reason,
                          frame, ++order);
        const auto degraded = breaker.ToStatus(frame);
        if (!degraded.active || degraded.effect_id != scenario.effect_id ||
            degraded.backend != scenario.backend ||
            degraded.reason != scenario.reason ||
            degraded.state != "cooling_down") {
          std::cerr << scenario.name
                    << " degraded status was malformed after failure: active="
                    << degraded.active << " id='" << degraded.effect_id
                    << "' backend='" << degraded.backend << "' reason='"
                    << degraded.reason << "' state='" << degraded.state
                    << "'\n";
          return false;
        }

        if (attempts == 1) {
          saw_first_cooldown =
              degraded.failure_count == 1 &&
              degraded.cooldown_frames ==
                  OptionalEffectBreaker::kInitialCooldownFrames;
        } else {
          saw_second_cooldown =
              degraded.failure_count == 2 &&
              degraded.cooldown_frames ==
                  (OptionalEffectBreaker::kInitialCooldownFrames * 2);
        }
      } else {
        saw_success_reset = breaker.OnSuccess();
      }
    }

    if (!saw_first_cooldown || !saw_retry_ready || !saw_second_cooldown ||
        !saw_success_reset || breaker.active() || attempts != 3 ||
        output_frames != 91 || skipped_cooldown_frames < 80) {
      std::cerr << scenario.name
                << " breaker did not fail open/retry as expected:"
                << " first_cooldown=" << saw_first_cooldown
                << " retry_ready=" << saw_retry_ready
                << " second_cooldown=" << saw_second_cooldown
                << " success_reset=" << saw_success_reset
                << " active=" << breaker.active() << " attempts=" << attempts
                << " output_frames=" << output_frames
                << " skipped=" << skipped_cooldown_frames << "\n";
      return false;
    }
  }

  return true;
}

bool TestVideoPipelineStatusExposesDegradedEffect() {
  ServiceHarness h;
  h.consumer_present.store(true, std::memory_order_relaxed);
  VirtualCameraService service(h.Hooks());
  auto cfg = TestConfig();

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil([&] { return service.Status().pipeline.running; }, 250ms)) {
    std::cerr << "video pipeline did not start before degraded status test\n";
    service.Stop();
    return false;
  }

  CameraPipelineStatus::DegradedEffect degraded;
  degraded.active = true;
  degraded.effect_id = "virtual_background.blur";
  degraded.backend = "maxine";
  degraded.reason = "synthetic Maxine virtual background failure";
  degraded.state = "cooling_down";
  degraded.failure_count = 1;
  degraded.cooldown_frames = OptionalEffectBreaker::kInitialCooldownFrames;
  h.pipeline->SetDegradedEffect(
      degraded,
      "Runtime degraded effects:\n - virtual_background.blur [maxine]: "
      "synthetic Maxine virtual background failure "
      "(state=cooling_down, failures=1, cooldown_frames=30)");

  const bool exposed = WaitUntil(
      [&] {
        const auto status = service.Status();
        const auto &fx = status.pipeline.degraded_effect;
        return fx.active && fx.effect_id == "virtual_background.blur" &&
               fx.backend == "maxine" &&
               fx.reason == "synthetic Maxine virtual background failure" &&
               fx.state == "cooling_down" && fx.failure_count == 1 &&
               fx.cooldown_frames ==
                   OptionalEffectBreaker::kInitialCooldownFrames &&
               status.pipeline.effects_note.find("Runtime degraded effects") !=
                   std::string::npos;
      },
      250ms);
  const auto status = service.Status();
  service.Stop();

  if (!exposed) {
    const auto &fx = status.pipeline.degraded_effect;
    std::cerr << "degraded effect status was not exposed; active=" << fx.active
              << " id='" << fx.effect_id << "' backend='" << fx.backend
              << "' reason='" << fx.reason << "' state='" << fx.state
              << "' note='" << status.pipeline.effects_note << "'\n";
    return false;
  }
  return true;
}

bool TestVideoOutputDisappearanceStopsPipelineAndMarksUnavailable() {
  ServiceHarness h;
  h.consumer_present.store(true, std::memory_order_relaxed);
  VirtualCameraService service(h.Hooks());
  auto cfg = TestConfig();

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil([&] { return service.Status().pipeline.running; }, 250ms)) {
    std::cerr << "video pipeline did not start before disappearance test\n";
    service.Stop();
    return false;
  }

  h.device_exists.store(false, std::memory_order_relaxed);
  const bool unavailable = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.pipeline_state == "device_unavailable" &&
               !status.virtual_device_present &&
               !status.virtual_device_available &&
               status.virtual_device_error.find("synthetic loopback missing") !=
                   std::string::npos &&
               status.pipeline_stops >= 1 &&
               status.last_transition == "stop_device_unavailable" &&
               h.pipeline->stop_calls.load() >= 1 &&
               h.pipeline->close_output_calls.load() >= 1;
      },
      500ms);
  const auto status = service.Status();
  service.Stop();

  if (!unavailable) {
    std::cerr << "output disappearance did not stop/mark unavailable; state='"
              << status.pipeline_state
              << "' present=" << status.virtual_device_present
              << " available=" << status.virtual_device_available << " error='"
              << status.virtual_device_error
              << "' pipeline_stops=" << status.pipeline_stops << " transition='"
              << status.last_transition
              << "' stops=" << h.pipeline->stop_calls.load()
              << " close_outputs=" << h.pipeline->close_output_calls.load()
              << "\n";
    return false;
  }
  return true;
}

bool TestVideoOutputRecoveryClearsUnavailableError() {
  ServiceHarness h;
  h.device_exists.store(false, std::memory_order_relaxed);
  VirtualCameraService service(h.Hooks());
  auto cfg = TestConfig();
  cfg.pipeline.output_device.clear();

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] {
            const auto status = service.Status();
            return status.pipeline_state == "device_unavailable" &&
                   !status.virtual_device_available &&
                   status.last_error.find("synthetic loopback missing") !=
                       std::string::npos;
          },
          250ms)) {
    const auto status = service.Status();
    std::cerr << "missing output was not surfaced before recovery; state='"
              << status.pipeline_state << "' last_error='" << status.last_error
              << "'\n";
    service.Stop();
    return false;
  }

  h.device_exists.store(true, std::memory_order_relaxed);
  const bool recovered = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.pipeline_state == "idle_no_consumer" &&
               status.virtual_device_present &&
               status.virtual_device_available &&
               status.virtual_device_error.empty() &&
               status.last_error.empty() &&
               h.pipeline->ensure_output_calls.load() > 0;
      },
      500ms);
  const auto status = service.Status();
  service.Stop();

  if (!recovered) {
    std::cerr << "output recovery left stale unavailable error; state='"
              << status.pipeline_state << "' virtual_error='"
              << status.virtual_device_error << "' last_error='"
              << status.last_error << "'\n";
    return false;
  }
  return true;
}

bool TestStandaloneGpuScalerPolicySkipsInactiveBackendTransfers() {
  using studiocast::video::ShouldRunStandaloneGpuScaler;

  if (ShouldRunStandaloneGpuScaler(
          /*scaling_needed=*/true,
          /*gpu_backend_active=*/true,
          /*have_deferred_gpu_out=*/false,
          /*allow_cpu_resize=*/true,
          /*same_backend_effects_ran=*/false)) {
    std::cerr << "standalone GPU scaler should skip when CPU resize is "
                 "allowed and no same-backend effect ran, including a "
                 "configured Vulkan scaler with blocked effects\n";
    return false;
  }

  if (!ShouldRunStandaloneGpuScaler(
          /*scaling_needed=*/true,
          /*gpu_backend_active=*/true,
          /*have_deferred_gpu_out=*/false,
          /*allow_cpu_resize=*/true,
          /*same_backend_effects_ran=*/true)) {
    std::cerr << "standalone GPU scaler should run when a same-backend effect "
                 "already ran\n";
    return false;
  }

  if (!ShouldRunStandaloneGpuScaler(
          /*scaling_needed=*/true,
          /*gpu_backend_active=*/true,
          /*have_deferred_gpu_out=*/false,
          /*allow_cpu_resize=*/false,
          /*same_backend_effects_ran=*/false)) {
    std::cerr << "standalone GPU scaler should run when CPU resize is "
                 "disabled\n";
    return false;
  }

  if (!ShouldRunStandaloneGpuScaler(
          /*scaling_needed=*/true,
          /*gpu_backend_active=*/true,
          /*have_deferred_gpu_out=*/true,
          /*allow_cpu_resize=*/true,
          /*same_backend_effects_ran=*/false)) {
    std::cerr << "standalone GPU scaler should run when a deferred GPU output "
                 "is available\n";
    return false;
  }

  if (ShouldRunStandaloneGpuScaler(
          /*scaling_needed=*/false,
          /*gpu_backend_active=*/true,
          /*have_deferred_gpu_out=*/true,
          /*allow_cpu_resize=*/false,
          /*same_backend_effects_ran=*/true)) {
    std::cerr
        << "standalone GPU scaler should skip when no scaling is needed\n";
    return false;
  }

  return true;
}

bool TestCudaVignetteHonorsComputeBackendPreference() {
  using studiocast::video::ComputeBackendAllowsCudaVideoCompute;
  using studiocast::video::ComputeBackendAvailability;
  using studiocast::video::ComputeBackendKind;
  using studiocast::video::ComputeBackendPreference;
  using studiocast::video::ResolveActiveComputeBackendName;
  using studiocast::video::ResolveComputeBackendSelection;

  if (ComputeBackendAllowsCudaVideoCompute(ComputeBackendPreference::cpu)) {
    std::cerr << "explicit CPU must not allow CUDA-backed vignette\n";
    return false;
  }
  if (ComputeBackendAllowsCudaVideoCompute(ComputeBackendPreference::vulkan)) {
    std::cerr << "explicit Vulkan must not allow CUDA-backed vignette\n";
    return false;
  }
  if (!ComputeBackendAllowsCudaVideoCompute(ComputeBackendPreference::cuda)) {
    std::cerr << "explicit CUDA should allow CUDA-backed vignette\n";
    return false;
  }
  if (!ComputeBackendAllowsCudaVideoCompute(
          ComputeBackendPreference::auto_select)) {
    std::cerr << "auto compute backend should allow CUDA-backed vignette\n";
    return false;
  }

  ComputeBackendAvailability available;
  available.cuda_available = true;
  available.vulkan_available = false;
  available.vulkan_unavailable_reason = "vulkan unavailable";

  auto selected =
      ResolveComputeBackendSelection(ComputeBackendPreference::cpu, available,
                                     /*compute_work_requested=*/true);
  if (selected.resolved != ComputeBackendKind::cpu ||
      ResolveActiveComputeBackendName(selected,
                                      /*compute_work_requested=*/true,
                                      /*maxine_compute_available=*/false,
                                      /*cuda_compute_available=*/true,
                                      /*vulkan_compute_available=*/false) !=
          "cpu") {
    std::cerr << "explicit CPU should report CPU even if CUDA vignette is "
                 "otherwise available\n";
    return false;
  }

  selected = ResolveComputeBackendSelection(ComputeBackendPreference::vulkan,
                                            available,
                                            /*compute_work_requested=*/true);
  if (selected.resolved != ComputeBackendKind::cpu ||
      ResolveActiveComputeBackendName(selected,
                                      /*compute_work_requested=*/true,
                                      /*maxine_compute_available=*/false,
                                      /*cuda_compute_available=*/true,
                                      /*vulkan_compute_available=*/false) !=
          "cpu") {
    std::cerr << "explicit Vulkan should not report CUDA when only CUDA "
                 "vignette is available\n";
    return false;
  }

  selected =
      ResolveComputeBackendSelection(ComputeBackendPreference::cuda, available,
                                     /*compute_work_requested=*/true);
  if (selected.resolved != ComputeBackendKind::cuda ||
      ResolveActiveComputeBackendName(selected,
                                      /*compute_work_requested=*/true,
                                      /*maxine_compute_available=*/false,
                                      /*cuda_compute_available=*/true,
                                      /*vulkan_compute_available=*/false) !=
          "cuda") {
    std::cerr << "explicit CUDA should report active CUDA when CUDA vignette "
                 "is available\n";
    return false;
  }

  selected = ResolveComputeBackendSelection(
      ComputeBackendPreference::auto_select, available,
      /*compute_work_requested=*/true);
  if (selected.resolved != ComputeBackendKind::cuda ||
      ResolveActiveComputeBackendName(selected,
                                      /*compute_work_requested=*/true,
                                      /*maxine_compute_available=*/false,
                                      /*cuda_compute_available=*/true,
                                      /*vulkan_compute_available=*/false) !=
          "cuda") {
    std::cerr << "auto compute backend should still report active CUDA when "
                 "CUDA vignette is available\n";
    return false;
  }

  return true;
}

bool TestGpuBackendActiveFrameMarkerCountsOncePerFrame() {
  using studiocast::video::MarkGpuBackendActiveFrame;

  bool active_this_frame = false;
  std::uint64_t active_frames = 0;
  if (!MarkGpuBackendActiveFrame(active_this_frame, active_frames) ||
      active_frames != 1) {
    std::cerr << "first backend touch should increment active frame count\n";
    return false;
  }
  if (MarkGpuBackendActiveFrame(active_this_frame, active_frames) ||
      active_frames != 1) {
    std::cerr << "second backend touch in the same frame should not "
                 "double-count active frames\n";
    return false;
  }

  active_this_frame = false;
  if (!MarkGpuBackendActiveFrame(active_this_frame, active_frames) ||
      active_frames != 2) {
    std::cerr << "new frame should increment active frame count once\n";
    return false;
  }

  return true;
}

bool TestComputeBackendSelectionPolicyIsNoGpuSafe() {
  using studiocast::video::ComputeBackendAvailability;
  using studiocast::video::ComputeBackendKind;
  using studiocast::video::ComputeBackendPreference;
  using studiocast::video::ResolveComputeBackendSelection;

  ComputeBackendAvailability available;
  available.cuda_available = true;
  available.vulkan_available = true;
  auto selected = ResolveComputeBackendSelection(
      ComputeBackendPreference::auto_select, available,
      /*compute_work_requested=*/true);
  if (selected.resolved != ComputeBackendKind::cuda || selected.degraded) {
    std::cerr << "auto compute backend should prefer CUDA when CUDA and Vulkan "
                 "are available\n";
    return false;
  }

  available.cuda_available = false;
  available.vulkan_available = true;
  selected = ResolveComputeBackendSelection(
      ComputeBackendPreference::auto_select, available,
      /*compute_work_requested=*/true);
  if (selected.resolved != ComputeBackendKind::vulkan || selected.degraded) {
    std::cerr << "auto compute backend should choose Vulkan when CUDA is "
                 "unavailable and Vulkan is available\n";
    return false;
  }

  available.cuda_available = false;
  available.vulkan_available = true;
  available.cuda_unavailable_reason = "cuda unavailable";
  selected =
      ResolveComputeBackendSelection(ComputeBackendPreference::cuda, available,
                                     /*compute_work_requested=*/true);
  if (selected.resolved != ComputeBackendKind::cpu || !selected.degraded ||
      selected.fallback_reason.empty()) {
    std::cerr << "explicit CUDA should degrade visibly to CPU instead of "
                 "running Vulkan\n";
    return false;
  }

  available.cuda_available = true;
  available.vulkan_available = false;
  available.vulkan_unavailable_reason = "vulkan unavailable";
  selected = ResolveComputeBackendSelection(ComputeBackendPreference::vulkan,
                                            available,
                                            /*compute_work_requested=*/true);
  if (selected.resolved != ComputeBackendKind::cpu || !selected.degraded ||
      selected.fallback_reason.empty()) {
    std::cerr << "explicit Vulkan should degrade visibly to CPU instead of "
                 "running CUDA\n";
    return false;
  }

  available.cuda_available = true;
  available.vulkan_available = true;
  selected = ResolveComputeBackendSelection(
      ComputeBackendPreference::auto_select, available,
      /*compute_work_requested=*/false);
  if (selected.resolved != ComputeBackendKind::cpu || selected.degraded) {
    std::cerr << "no-effects path should resolve compute backend to CPU\n";
    return false;
  }

  return true;
}

bool TestMirrorCanonicalPlanAndBackendContractIsNoGpuSafe() {
  namespace effects = studiocast::video::effects;
  using studiocast::video::ComputeBackendAvailability;
  using studiocast::video::ComputeBackendKind;
  using studiocast::video::ComputeBackendPreference;
  using studiocast::video::ResolveComputeBackendSelection;

  effects::BroadcastCameraEffects mirror_only;
  mirror_only.mirror = true;
  auto plan = effects::BuildBroadcastEffectsPlan(mirror_only);
  if (plan.ordered_effect_ids.size() != 1 ||
      plan.ordered_effect_ids.front() != effects::contract::kEffectIdMirror ||
      !plan.disabled.empty() ||
      !effects::BroadcastEffectsPlanRequestsCompute(plan)) {
    std::cerr << "mirror-only plan must schedule final compute work\n";
    return false;
  }

  effects::BroadcastCameraEffects combined;
  combined.mirror = true;
  combined.video_noise_removal.enabled = true;
  combined.eye_contact.enabled = true;
  combined.auto_frame.enabled = true;
  combined.virtual_key_light.enabled = true;
  combined.vignette.enabled = true;
  combined.virtual_background.mode = effects::VirtualBackgroundMode::blur;
  plan = effects::BuildBroadcastEffectsPlan(combined);
  if (plan.ordered_effect_ids.empty() ||
      plan.ordered_effect_ids.back() != effects::contract::kEffectIdMirror ||
      plan.vignette_attach_to_effect_id == effects::contract::kEffectIdMirror) {
    std::cerr << "mirror must be last without becoming a vignette attachment "
                 "target\n";
    return false;
  }

  ComputeBackendAvailability available;
  available.vulkan_available = true;
  const auto selected = ResolveComputeBackendSelection(
      ComputeBackendPreference::vulkan, available,
      effects::BroadcastEffectsPlanRequestsCompute(
          effects::BuildBroadcastEffectsPlan(mirror_only)));
  if (selected.resolved != ComputeBackendKind::vulkan || selected.degraded) {
    std::cerr << "mirror-only explicit Vulkan must resolve Vulkan compute\n";
    return false;
  }

  CameraPipelineStatus::OpenVulkanTransfers counters;
  ++counters.active_frames;
  ++counters.upload_calls;
  ++counters.mirror_dispatch_calls;
  ++counters.download_calls;
  ++counters.final_download_calls;
  if (counters.active_frames != 1 || counters.upload_calls != 1 ||
      counters.mirror_dispatch_calls != 1 || counters.download_calls != 1 ||
      counters.final_download_calls != 1 ||
      counters.cpu_continuation_download_calls != 0 ||
      counters.cpu_tail_stage_calls != 0) {
    std::cerr << "mirror transfer counter contract must be one resident "
                 "section without a CPU tail\n";
    return false;
  }
  return true;
}

bool TestVignetteCanonicalPlanDefaultAndBackendContractIsNoGpuSafe() {
  namespace effects = studiocast::video::effects;
  using studiocast::video::ComputeBackendAvailability;
  using studiocast::video::ComputeBackendKind;
  using studiocast::video::ComputeBackendPreference;
  using studiocast::video::ResolveComputeBackendSelection;

  effects::BroadcastCameraEffects vignette_only;
  if (vignette_only.vignette.intensity !=
      effects::contract::kVignetteIntensityDefault) {
    std::cerr << "canonical vignette default construction must reference the "
                 "stable 35 percent contract\n";
    return false;
  }
  vignette_only.vignette.enabled = true;
  auto plan = effects::BuildBroadcastEffectsPlan(vignette_only);
  if (plan.ordered_effect_ids.size() != 1 ||
      plan.ordered_effect_ids.front() != effects::contract::kEffectIdVignette ||
      !plan.vignette_attach_to_effect_id.empty() ||
      !effects::BroadcastEffectsPlanRequestsCompute(plan)) {
    std::cerr << "vignette-only plan must schedule standalone final compute "
                 "work\n";
    return false;
  }

  effects::BroadcastCameraEffects combined;
  combined.auto_frame.enabled = true;
  combined.vignette.enabled = true;
  combined.mirror = true;
  plan = effects::BuildBroadcastEffectsPlan(combined);
  const auto auto_frame =
      std::find(plan.ordered_effect_ids.begin(), plan.ordered_effect_ids.end(),
                effects::contract::kEffectIdAutoFrame);
  const auto vignette =
      std::find(plan.ordered_effect_ids.begin(), plan.ordered_effect_ids.end(),
                effects::contract::kEffectIdVignette);
  const auto mirror =
      std::find(plan.ordered_effect_ids.begin(), plan.ordered_effect_ids.end(),
                effects::contract::kEffectIdMirror);
  if (auto_frame == plan.ordered_effect_ids.end() ||
      vignette == plan.ordered_effect_ids.end() ||
      mirror == plan.ordered_effect_ids.end() || !(auto_frame < vignette) ||
      !(vignette < mirror) ||
      plan.vignette_attach_to_effect_id == effects::contract::kEffectIdMirror) {
    std::cerr << "canonical ordering must be framing -> vignette -> mirror\n";
    return false;
  }

  const auto tracked_center_compatibility =
      studiocast::video::ApplyOpenVulkanVignettePlanCompatibility(combined,
                                                                  &plan);
  const auto retained_auto_frame =
      std::find(plan.ordered_effect_ids.begin(), plan.ordered_effect_ids.end(),
                effects::contract::kEffectIdAutoFrame);
  const auto blocked_vignette =
      std::find(plan.ordered_effect_ids.begin(), plan.ordered_effect_ids.end(),
                effects::contract::kEffectIdVignette);
  const auto retained_mirror =
      std::find(plan.ordered_effect_ids.begin(), plan.ordered_effect_ids.end(),
                effects::contract::kEffectIdMirror);
  if (!tracked_center_compatibility.blocked ||
      tracked_center_compatibility.reason_code !=
          studiocast::video::
              kOpenVulkanVignetteTrackedCenterNotSupportedReason ||
      tracked_center_compatibility.detail !=
          studiocast::video::
              kOpenVulkanVignetteTrackedCenterNotSupportedDetail ||
      retained_auto_frame == plan.ordered_effect_ids.end() ||
      blocked_vignette != plan.ordered_effect_ids.end() ||
      retained_mirror == plan.ordered_effect_ids.end() ||
      !plan.vignette_attach_to_effect_id.empty() || plan.disabled.size() != 1 ||
      plan.disabled.front().id != effects::contract::kEffectIdVignette ||
      plan.disabled.front().reason.find(
          "[vulkan_vignette_tracked_center_not_supported]") ==
          std::string::npos) {
    std::cerr << "explicit Vulkan tracked-center compatibility must visibly "
                 "block only vignette while retaining Auto Frame and mirror\n";
    return false;
  }

  combined.vignette.center_on_tracked_face = false;
  plan = effects::BuildBroadcastEffectsPlan(combined);
  if (studiocast::video::ApplyOpenVulkanVignettePlanCompatibility(combined,
                                                                  &plan)
          .blocked ||
      std::find(plan.ordered_effect_ids.begin(), plan.ordered_effect_ids.end(),
                effects::contract::kEffectIdVignette) ==
          plan.ordered_effect_ids.end()) {
    std::cerr << "explicit fixed-center Vulkan config must retain vignette\n";
    return false;
  }

  combined.vignette.center_on_tracked_face = true;
  plan = effects::BuildBroadcastEffectsPlan(combined);
  plan.ordered_effect_ids.erase(
      std::remove(plan.ordered_effect_ids.begin(),
                  plan.ordered_effect_ids.end(),
                  std::string(effects::contract::kEffectIdAutoFrame)),
      plan.ordered_effect_ids.end());
  if (studiocast::video::ApplyOpenVulkanVignettePlanCompatibility(combined,
                                                                  &plan)
          .blocked ||
      std::find(plan.ordered_effect_ids.begin(), plan.ordered_effect_ids.end(),
                effects::contract::kEffectIdVignette) ==
          plan.ordered_effect_ids.end()) {
    std::cerr << "vignette must remain when unavailable Auto Frame was removed "
                 "before Vulkan compatibility evaluation\n";
    return false;
  }

  ComputeBackendAvailability available;
  available.vulkan_available = true;
  const auto selected = ResolveComputeBackendSelection(
      ComputeBackendPreference::vulkan, available,
      effects::BroadcastEffectsPlanRequestsCompute(
          effects::BuildBroadcastEffectsPlan(vignette_only)));
  if (selected.resolved != ComputeBackendKind::vulkan || selected.degraded) {
    std::cerr << "vignette-only explicit Vulkan must resolve Vulkan compute\n";
    return false;
  }

  CameraPipelineStatus::OpenVulkanTransfers counters;
  ++counters.active_frames;
  ++counters.upload_calls;
  ++counters.vignette_dispatch_calls;
  ++counters.vignette_factor_allocation_calls;
  ++counters.vignette_factor_generation_calls;
  ++counters.vignette_factor_upload_calls;
  ++counters.mirror_dispatch_calls;
  ++counters.download_calls;
  ++counters.final_download_calls;
  ++counters.forced_sync_calls;
  if (counters.active_frames != 1 || counters.upload_calls != 1 ||
      counters.vignette_dispatch_calls != 1 ||
      counters.vignette_factor_allocation_calls != 1 ||
      counters.vignette_factor_generation_calls != 1 ||
      counters.vignette_factor_upload_calls != 1 ||
      counters.mirror_dispatch_calls != 1 || counters.download_calls != 1 ||
      counters.final_download_calls != 1 || counters.forced_sync_calls != 1 ||
      counters.cpu_continuation_download_calls != 0 ||
      counters.cpu_tail_stage_calls != 0) {
    std::cerr << "combined Vulkan vignette/mirror contract must be one "
                 "resident section/completion without a CPU tail\n";
    return false;
  }
  return true;
}

bool TestOpenVulkanEyeContactPlanCompatibilityIsIsolatedAndFailClosed() {
  namespace effects = studiocast::video::effects;

  effects::BroadcastCameraEffects combined;
  combined.video_noise_removal.enabled = true;
  combined.eye_contact.enabled = true;
  combined.virtual_background.mode = effects::VirtualBackgroundMode::blur;
  combined.virtual_key_light.enabled = true;
  combined.auto_frame.enabled = true;
  combined.vignette.enabled = true;
  combined.mirror = true;
  auto plan = effects::BuildBroadcastEffectsPlan(combined);

  const auto compatibility =
      studiocast::video::ApplyOpenVulkanEyeContactPlanCompatibility(&plan);
  const auto contains = [&](std::string_view effect_id) {
    return std::find(plan.ordered_effect_ids.begin(),
                     plan.ordered_effect_ids.end(), effect_id) !=
           plan.ordered_effect_ids.end();
  };

  if (!compatibility.blocked ||
      compatibility.reason_code !=
          studiocast::video::kOpenVulkanEyeContactUnavailableReason ||
      compatibility.blocker_code !=
          studiocast::video::kOpenVulkanEyeContactRuntimeUnavailableReason ||
      contains(effects::contract::kEffectIdEyeContact) ||
      !contains(effects::contract::kEffectIdVideoNoiseRemoval) ||
      !contains(effects::contract::kEffectIdVirtualBackgroundBlur) ||
      !contains(effects::contract::kEffectIdVirtualKeyLight) ||
      !contains(effects::contract::kEffectIdAutoFrame) ||
      !contains(effects::contract::kEffectIdVignette) ||
      !contains(effects::contract::kEffectIdMirror) ||
      plan.vignette_attach_to_effect_id !=
          effects::contract::kEffectIdAutoFrame ||
      plan.disabled.size() != 1 ||
      plan.disabled.front().id != effects::contract::kEffectIdEyeContact ||
      plan.disabled.front().reason.find(
          "[open_vulkan_eye_contact_unavailable] "
          "[open_vulkan_eye_contact_runtime_unavailable]") ==
          std::string::npos) {
    std::cerr << "explicit Vulkan eye-contact compatibility must remove and "
                 "diagnose only eye contact\n";
    return false;
  }

  const std::size_t disabled_count = plan.disabled.size();
  if (studiocast::video::ApplyOpenVulkanEyeContactPlanCompatibility(&plan)
          .blocked ||
      plan.disabled.size() != disabled_count) {
    std::cerr << "eye-contact compatibility must be idempotent after removal\n";
    return false;
  }

  effects::BroadcastCameraEffects mirror_only;
  mirror_only.mirror = true;
  auto mirror_plan = effects::BuildBroadcastEffectsPlan(mirror_only);
  if (studiocast::video::ApplyOpenVulkanEyeContactPlanCompatibility(
          &mirror_plan)
          .blocked ||
      mirror_plan.ordered_effect_ids.size() != 1 ||
      mirror_plan.ordered_effect_ids.front() !=
          effects::contract::kEffectIdMirror) {
    std::cerr << "eye-contact compatibility must not alter unrelated plans\n";
    return false;
  }
  return true;
}

bool TestOpenVulkanVideoNoiseRemovalPlanCompatibilityIsIsolatedAndFailClosed() {
  namespace effects = studiocast::video::effects;

  effects::BroadcastCameraEffects combined;
  combined.video_noise_removal.enabled = true;
  combined.eye_contact.enabled = true;
  combined.virtual_background.mode = effects::VirtualBackgroundMode::blur;
  combined.virtual_key_light.enabled = true;
  combined.auto_frame.enabled = true;
  combined.vignette.enabled = true;
  combined.mirror = true;
  auto plan = effects::BuildBroadcastEffectsPlan(combined);

  const auto compatibility =
      studiocast::video::ApplyOpenVulkanVideoNoiseRemovalPlanCompatibility(
          &plan);
  const auto contains = [&](std::string_view effect_id) {
    return std::find(plan.ordered_effect_ids.begin(),
                     plan.ordered_effect_ids.end(),
                     effect_id) != plan.ordered_effect_ids.end();
  };

  if (!compatibility.blocked ||
      compatibility.reason_code !=
          studiocast::video::kOpenVulkanVideoNoiseRemovalUnavailableReason ||
      compatibility.blocker_code !=
          studiocast::video::
              kOpenVulkanVideoNoiseRemovalRuntimeUnavailableReason ||
      contains(effects::contract::kEffectIdVideoNoiseRemoval) ||
      !contains(effects::contract::kEffectIdEyeContact) ||
      !contains(effects::contract::kEffectIdVirtualBackgroundBlur) ||
      !contains(effects::contract::kEffectIdVirtualKeyLight) ||
      !contains(effects::contract::kEffectIdAutoFrame) ||
      !contains(effects::contract::kEffectIdVignette) ||
      !contains(effects::contract::kEffectIdMirror) ||
      plan.vignette_attach_to_effect_id !=
          effects::contract::kEffectIdAutoFrame ||
      plan.disabled.size() != 1 ||
      plan.disabled.front().id !=
          effects::contract::kEffectIdVideoNoiseRemoval ||
      plan.disabled.front().reason.find(
          "[open_vulkan_video_noise_removal_unavailable] "
          "[open_vulkan_video_noise_removal_runtime_unavailable]") ==
          std::string::npos) {
    std::cerr << "explicit Vulkan video-denoise compatibility must remove and "
                 "diagnose only video noise removal\n";
    return false;
  }

  const std::size_t disabled_count = plan.disabled.size();
  if (studiocast::video::ApplyOpenVulkanVideoNoiseRemovalPlanCompatibility(
          &plan)
          .blocked ||
      plan.disabled.size() != disabled_count) {
    std::cerr << "video-denoise compatibility must be idempotent after "
                 "removal\n";
    return false;
  }

  effects::BroadcastCameraEffects mirror_only;
  mirror_only.mirror = true;
  auto mirror_plan = effects::BuildBroadcastEffectsPlan(mirror_only);
  if (studiocast::video::ApplyOpenVulkanVideoNoiseRemovalPlanCompatibility(
          &mirror_plan)
          .blocked ||
      mirror_plan.ordered_effect_ids.size() != 1 ||
      mirror_plan.ordered_effect_ids.front() !=
          effects::contract::kEffectIdMirror) {
    std::cerr << "video-denoise compatibility must not alter unrelated plans\n";
    return false;
  }

  effects::BroadcastCameraEffects denoise_and_vignette;
  denoise_and_vignette.video_noise_removal.enabled = true;
  denoise_and_vignette.vignette.enabled = true;
  auto attached_plan = effects::BuildBroadcastEffectsPlan(denoise_and_vignette);
  const bool vignette_was_standalone =
      attached_plan.vignette_attach_to_effect_id.empty();
  const auto attached_compatibility =
      studiocast::video::ApplyOpenVulkanVideoNoiseRemovalPlanCompatibility(
          &attached_plan);
  if (!vignette_was_standalone || !attached_compatibility.blocked ||
      !attached_plan.vignette_attach_to_effect_id.empty() ||
      attached_plan.ordered_effect_ids.size() != 1 ||
      attached_plan.ordered_effect_ids.front() !=
          effects::contract::kEffectIdVignette) {
    std::cerr << "removing video denoise must preserve standalone vignette\n";
    return false;
  }
  return true;
}

} // namespace

int main() {
  const struct {
    const char *name;
    bool (*fn)();
  } tests[] = {
      {"live effect attribution requires successful frame commit",
       &TestLiveEffectBackendAttributionRequiresSuccessfulFrameCommit},
      {"video service sanitizes effect attribution across lifecycle",
       &TestVideoServiceSanitizesEffectBackendsAcrossLifecycle},
      {"video service delivers only genuine live effect changes",
       &TestVideoServiceDeliversOnlyGenuineLiveEffectChanges},
      {"identical pipeline effects preserve runtime generation",
       &studiocast::tests::
           TestCameraPipelineIdenticalEffectsPreserveRuntimeGeneration},
      {"explicit effects resource refresh is a separate invalidation",
       &studiocast::tests::
           TestCameraPipelineExplicitEffectsRefreshIsSeparateInvalidation},
      {"video pipeline does not start without consumer",
       &TestVideoPipelineDoesNotStartWithoutConsumer},
      {"video pipeline starts when consumer appears",
       &TestVideoPipelineStartsWhenConsumerAppears},
      {"video pipeline stops when consumer disappears",
       &TestVideoPipelineStopsWhenConsumerDisappears},
      {"video grace window absorbs consumer flapping",
       &TestVideoGraceWindowAbsorbsConsumerFlapping},
      {"video consumer detection error surfaces without starting",
       &TestVideoConsumerDetectionErrorSurfacesWithoutStarting},
      {"video output open failure does not start pipeline",
       &TestVideoOutputOpenFailureDoesNotStartPipeline},
      {"video start failure backs off", &TestVideoStartFailureBacksOff},
      {"video start failure clears after recovery",
       &TestVideoStartFailureClearsAfterRecovery},
      {"video config restart transition name is stable",
       &TestVideoConfigRestartTransitionNameIsStable},
      {"video output format change restarts pipeline",
       &TestVideoOutputFormatChangeRestartsPipeline},
      {"optional video effects fail open with cooldown retry",
       &TestOptionalVideoEffectsFailOpenCooldownAndRetry},
      {"video pipeline exposes degraded effect status",
       &TestVideoPipelineStatusExposesDegradedEffect},
      {"video output disappearance marks unavailable",
       &TestVideoOutputDisappearanceStopsPipelineAndMarksUnavailable},
      {"video output recovery clears unavailable error",
       &TestVideoOutputRecoveryClearsUnavailableError},
      {"standalone GPU scaler skips inactive backend transfers",
       &TestStandaloneGpuScalerPolicySkipsInactiveBackendTransfers},
      {"CUDA vignette honors compute backend preference",
       &TestCudaVignetteHonorsComputeBackendPreference},
      {"GPU backend active frame marker counts once per frame",
       &TestGpuBackendActiveFrameMarkerCountsOncePerFrame},
      {"compute backend selection policy is no-GPU safe",
       &TestComputeBackendSelectionPolicyIsNoGpuSafe},
      {"mirror canonical plan and backend contract is no-GPU safe",
       &TestMirrorCanonicalPlanAndBackendContractIsNoGpuSafe},
      {"vignette canonical plan/default/backend contract is no-GPU safe",
       &TestVignetteCanonicalPlanDefaultAndBackendContractIsNoGpuSafe},
      {"Open Vulkan eye contact plan compatibility is isolated/fail closed",
       &TestOpenVulkanEyeContactPlanCompatibilityIsIsolatedAndFailClosed},
      {"Open Vulkan video denoise plan compatibility is isolated/fail closed",
       &TestOpenVulkanVideoNoiseRemovalPlanCompatibilityIsIsolatedAndFailClosed},
      {"latest-frame worker overwrites pending blocked work",
       &studiocast::tests::
           TestLatestFrameWinsOverwritesPendingWithBlockedProcessor},
      {"latest-frame worker Stop wakes and joins",
       &studiocast::tests::TestLatestFrameWinsStopWakesAndJoins},
      {"latest-frame worker generation rejects stale results",
       &studiocast::tests::TestLatestFrameWinsGenerationRejectsStaleResults},
      {"latest-frame worker stats counters and last error",
       &studiocast::tests::TestLatestFrameWinsStatsCountersAndLastError},
      {"FastDVDnet denoise tensor contract is declared",
       &studiocast::tests::TestFastDvdnetDenoiseTensorContractIsDeclared},
      {"YuNet face detection CPU tensor tail contract is declared",
       &studiocast::tests::
           TestYunetFaceDetectionCpuTensorTailContractIsDeclared},
      {"YuNet explicit Vulkan provider and tensor policy is fail closed",
       &studiocast::tests::
           TestYunetExplicitVulkanProviderAndTensorPolicyIsFailClosed},
      {"Open Video eye contact CPU tensor tail contract is declared",
       &studiocast::tests::
           TestOpenVideoEyeContactCpuTensorTailContractIsDeclared},
      {"Open Vulkan eye contact capability facts are fail closed",
       &studiocast::tests::
           TestOpenVulkanEyeContactCapabilityFactsAreFailClosed},
      {"Open Vulkan video denoise capability facts are fail closed",
       &studiocast::tests::
           TestOpenVulkanVideoNoiseRemovalCapabilityFactsAreFailClosed},
      {"Open Vulkan video denoise temporal readiness gates are exact",
       &studiocast::tests::
           TestOpenVulkanVideoNoiseRemovalTemporalReadinessGatesAreExact},
      {"frame artifact cache reuses compatible matte within frame",
       &studiocast::tests::
           TestFrameArtifactCacheReusesCompatibleMatteWithinFrame},
      {"frame artifact cache reuses compatible Maxine matte within frame",
       &studiocast::tests::
           TestFrameArtifactCacheReusesCompatibleMaxineMatteWithinFrame},
      {"frame artifact cache separates incompatible matte keys",
       &studiocast::tests::
           TestFrameArtifactCacheSeparatesIncompatibleMatteKeys},
      {"frame artifact cache invalidates matte on new frame",
       &studiocast::tests::TestFrameArtifactCacheInvalidatesMatteOnNewFrame},
      {"frame analysis cache retains face analysis within frame",
       &studiocast::tests::
           TestFrameAnalysisCacheRetainsFaceAnalysisWithinFrame},
      {"frame artifact cache precomputed matte keys preserve compatibility",
       &studiocast::tests::
           TestFrameArtifactCachePrecomputedMatteKeysPreserveCompatibility},
      {"frame artifact cache device storage aliases preserve compatibility",
       &studiocast::tests::
           TestFrameArtifactCacheDeviceStorageAliasesPreserveCompatibility},
      {"frame artifact cache reuses device matte across combined effects",
       &studiocast::tests::
           TestFrameArtifactCacheReusesDeviceMatteAcrossCombinedEffects},
      {"CUDA context classifier prefers current context",
       &studiocast::tests::TestCudaContextClassifierPrefersCurrentContext},
      {"signed int32 PTX pitch contract is no-GPU safe",
       &studiocast::tests::TestSignedInt32PtxPitchContractIsNoGpuSafe},
      {"Open CUDA box blur radius bounds are no-GPU safe",
       &studiocast::tests::TestOpenCudaBoxBlurRadiusBoundsAreNoGpuSafe},
      {"Open CUDA f32 resize border contract is replicate",
       &studiocast::tests::
           TestOpenCudaF32ResizeBorderContractIsReplicateNoGpuSafe},
      {"Open CUDA alpha clamp and solid BGR contract are no-GPU safe",
       &studiocast::tests::TestOpenCudaAlphaClampAndSolidBgrContractNoGpuSafe},
      {"CUDA u8 resize rounding contract is no-GPU safe",
       &studiocast::tests::TestCudaU8ResizeRoundingContractIsNoGpuSafe},
      {"CUDA resize availability probe is thread-safe",
       &studiocast::tests::TestCudaResizeAvailabilityProbeIsThreadSafe},
      {"Open Vulkan matting first use does setup work",
       &studiocast::tests::TestOpenVulkanMattingFirstUseDoesSetupWork},
      {"Open Vulkan matting repeated frames skip setup work",
       &studiocast::tests::
           TestOpenVulkanMattingRepeatedFramesDoNotRepeatSetupWork},
      {"Open Vulkan matting model id change refreshes setup",
       &studiocast::tests::TestOpenVulkanMattingModelIdChangeTriggersSetupWork},
      {"Open Vulkan matting frame size change only refreshes session",
       &studiocast::tests::
           TestOpenVulkanMattingFrameSizeChangeOnlyRefreshesSession},
      {"Open Vulkan matting repeated frame processing keeps setup stable",
       &studiocast::tests::
           TestOpenVulkanMattingRepeatedFrameProcessingKeepsSetupCountersStable},
      {"Open Vulkan matting context generation recreates session",
       &studiocast::tests::
           TestOpenVulkanMattingContextGenerationChangeRecreatesSession},
      {"Open Vulkan matting latched failure is reused",
       &studiocast::tests::TestOpenVulkanMattingLatchedFailureIsReused},
      {"replace background repeated frames do not stat image",
       &studiocast::tests::
           TestReplaceBackgroundRepeatedFramePathDoesNotStatImage},
      {"replace background same-path refresh uses prepared mtime",
       &studiocast::tests::
           TestReplaceBackgroundSamePathExplicitRefreshUsesPreparedMtime},
      {"replace background frame size change refreshes resize only",
       &studiocast::tests::
           TestReplaceBackgroundFrameSizeChangeOnlyRefreshesResizedBackground},
      {"V4L2 capture treats 720p as MJPEG-worthy",
       &studiocast::tests::TestV4l2CapturePreferenceTreats720pAsMjpegWorthy},
      {"V4L2 YUYV request tries MJPEG first at HD when preferred",
       &studiocast::tests::TestV4l2YuyvRequestTriesMjpegFirstAtHdWhenPreferred},
      {"V4L2 YUYV request falls back to MJPEG after YUYV at low resolution",
       &studiocast::tests::
           TestV4l2YuyvRequestFallsBackToMjpegAfterYuyvAtLowResolution},
      {"V4L2 YUYV request does not add MJPEG fallback when disabled",
       &studiocast::tests::
           TestV4l2YuyvRequestDoesNotAddMjpegFallbackWhenPreferenceDisabled},
      {"V4L2 unsupported formats are skipped without duplicates",
       &studiocast::tests::
           TestV4l2UnsupportedFormatsAreSkippedWithoutDuplicates},
      {"V4L2 explicit MJPEG request does not fall back inside Open order",
       &studiocast::tests::
           TestV4l2ExplicitMjpegRequestDoesNotFallBackToYuyvInsideOpenOrder},
      {"V4L2 fake negotiation uses ordered fallback",
       &studiocast::tests::TestV4l2FakeNegotiationUsesOrderedFallback},
      {"V4L2 MJPEG decode failure falls back to raw once",
       &studiocast::tests::TestV4l2MjpegDecodeFailureFallsBackToRawOnce},
      {"YUYV to RGB24 matches BT.601 and preserves padding",
       &studiocast::tests::TestYuyvToRgb24MatchesBt601AndPreservesPadding},
      {"YUYV to RGB24 backends match scalar reference",
       &studiocast::tests::TestYuyvToRgb24BackendsMatchScalarReference},
      {"RGB24 to YUYV matches BT.601 within chroma rounding",
       &studiocast::tests::TestRgb24ToYuyvMatchesBt601WithinChromaRounding},
      {"RGB24 to YUYV backends match scalar reference",
       &studiocast::tests::TestRgb24ToYuyvBackendsMatchScalarReference},
      {"RGB24 to YUYV public path matches scalar with scratch variants",
       &studiocast::tests::
           TestRgb24ToYuyvPublicPathMatchesScalarWithScratchVariants},
      {"raw YUYV passthrough predicate rejects every RGB operation",
       &studiocast::tests::TestRawYuyvPassthroughPredicateIsStrict},
      {"raw YUYV passthrough preserves exact tight and padded layouts",
       &studiocast::tests::TestRawYuyvPassthroughPreservesExactLayouts},
      {"raw YUYV passthrough copies active rows and zeros destination padding",
       &studiocast::tests::
           TestRawYuyvPassthroughCopiesActiveRowsAndZerosPadding},
      {"raw YUYV passthrough rejects unsafe frame buffers",
       &studiocast::tests::TestRawYuyvPassthroughRejectsUnsafeFrames},
      {"raw YUYV passthrough counters track actual conversion call sites",
       &studiocast::tests::TestRawYuyvPassthroughCountersTrackActualCallSites},
      {"RGB24/BGR24 backends match scalar and preserve padding",
       &studiocast::tests::TestRgb24Bgr24BackendsMatchScalarAndPreservePadding},
      {"RGB24/BGR24 public path matches scalar in-place",
       &studiocast::tests::TestRgb24Bgr24PublicPathMatchesScalarInPlace},
      {"RGB24 bilinear resize preserves active pixels and zeros padding",
       &studiocast::tests::
           TestResizeRgb24BilinearPreservesActivePixelsAndZerosPadding},
      {"RGB24 bilinear resize handles degenerate axes and plan reuse",
       &studiocast::tests::
           TestResizeRgb24BilinearHandlesDegenerateAxesAndPlanReuse},
      {"Background remove CPU matches reference and preserves padding",
       &studiocast::tests::
           TestBackgroundRemoveCpuMatchesReferenceAndPreservesPadding},
      {"Background blur CPU matches reference and preserves padding",
       &studiocast::tests::
           TestBackgroundBlurCpuMatchesReferenceAndPreservesPadding},
      {"Mirror CPU reference covers padded odd/even/1x1 and double mirror",
       &studiocast::tests::
           TestMirrorCpuReferenceCoversPaddingOddEvenDegenerateAndDoubleMirror},
  };

  int failed = 0;
  for (const auto &test : tests) {
    const bool ok = test.fn();
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << test.name << "\n";
    if (!ok)
      ++failed;
  }

  return failed == 0 ? 0 : 1;
}
