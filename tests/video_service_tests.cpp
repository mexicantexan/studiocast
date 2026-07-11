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
#include "core/video/scaling_policy.h"
#include "core/video/virtual_camera_service.h"

namespace studiocast::tests {
bool TestLatestFrameWinsOverwritesPendingWithBlockedProcessor();
bool TestLatestFrameWinsStopWakesAndJoins();
bool TestLatestFrameWinsGenerationRejectsStaleResults();
bool TestLatestFrameWinsStatsCountersAndLastError();
bool TestFastDvdnetDenoiseTensorContractIsDeclared();
bool TestYunetFaceDetectionCpuTensorTailContractIsDeclared();
bool TestOpenVideoEyeContactCpuTensorTailContractIsDeclared();
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
bool TestRgb24Bgr24BackendsMatchScalarAndPreservePadding();
bool TestRgb24Bgr24PublicPathMatchesScalarInPlace();
bool TestResizeRgb24BilinearPreservesActivePixelsAndZerosPadding();
bool TestResizeRgb24BilinearHandlesDegenerateAxesAndPlanReuse();
bool TestBackgroundRemoveCpuMatchesReferenceAndPreservesPadding();
bool TestBackgroundBlurCpuMatchesReferenceAndPreservesPadding();
} // namespace studiocast::tests

namespace {

using studiocast::video::CameraPipelineConfig;
using studiocast::video::CameraPipelineRunner;
using studiocast::video::CameraPipelineStatus;
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

  void SetEffects(
      const studiocast::video::effects::BroadcastCameraEffects &) override {
    set_effects_calls.fetch_add(1, std::memory_order_relaxed);
  }

  void SetMirrorEnabled(bool) override {}

  void SetDegradedEffect(CameraPipelineStatus::DegradedEffect degraded,
                         std::string effects_note) {
    std::lock_guard<std::mutex> lock(mu_);
    status_.degraded_effect = std::move(degraded);
    status_.effects_note = std::move(effects_note);
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
};

struct ServiceHarness {
  FakeCameraPipeline *pipeline = nullptr;
  std::atomic<bool> consumer_present{false};
  std::atomic<bool> device_exists{true};
  std::atomic<bool> detection_error{false};

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
                 "allowed and no same-backend effect ran\n";
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
    std::cerr << "standalone GPU scaler should skip when no scaling is needed\n";
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
  selected = ResolveComputeBackendSelection(
      ComputeBackendPreference::cuda, available,
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
  selected = ResolveComputeBackendSelection(
      ComputeBackendPreference::vulkan, available,
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

} // namespace

int main() {
  const struct {
    const char *name;
    bool (*fn)();
  } tests[] = {
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
      {"compute backend selection policy is no-GPU safe",
       &TestComputeBackendSelectionPolicyIsNoGpuSafe},
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
      {"Open Video eye contact CPU tensor tail contract is declared",
       &studiocast::tests::
           TestOpenVideoEyeContactCpuTensorTailContractIsDeclared},
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
       &studiocast::tests::
           TestOpenCudaAlphaClampAndSolidBgrContractNoGpuSafe},
      {"CUDA u8 resize rounding contract is no-GPU safe",
       &studiocast::tests::TestCudaU8ResizeRoundingContractIsNoGpuSafe},
      {"CUDA resize availability probe is thread-safe",
       &studiocast::tests::TestCudaResizeAvailabilityProbeIsThreadSafe},
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
