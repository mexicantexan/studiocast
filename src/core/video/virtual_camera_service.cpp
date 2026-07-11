#include "virtual_camera_service.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>

#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>

#include "../open_video/diagnose.h"
#include "core/maxine/maxine_manager.h"
#include "core/util/proc.h"
#include "core/video/effects/broadcast_effect_maxine_gate.h"
#include "core/video/effects/broadcast_effect_open_cuda_gate.h"
#include "core/video/v4l2loopback.h"

namespace studiocast::video {
namespace {

std::string ChooseDefaultOutputLoopback(std::string *error) {
  const auto rep = ProbeLoopback();
  for (const auto &d : rep.devices) {
    if (d.is_loopback && d.can_write)
      return d.dev_node;
  }
  if (error) {
    std::ostringstream oss;
    oss << "No writable v4l2loopback device found.\n"
        << "Run the suggested command from studiocast-video status, e.g.:\n"
        << "  " << rep.suggested_modprobe_cmd << "\n";
    *error = oss.str();
  }
  return {};
}

// Debug helper for diagnosing consumer-driven start/stop flapping.
//
// Enable with: STUDIOCAST_DEBUG_VCAM_SUPERVISOR=1
bool DebugVcamSupervisor() {
  static const bool enabled =
      (std::getenv("STUDIOCAST_DEBUG_VCAM_SUPERVISOR") != nullptr);
  return enabled;
}

void VcamDbg(const std::string &msg) {
  if (!DebugVcamSupervisor())
    return;
  static const auto t0 = std::chrono::steady_clock::now();
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  std::cerr << "[vcam_dbg +" << ms << "ms] " << msg << "\n";
}

std::string FormatPidsWithNames(const std::vector<int> &pids) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < pids.size(); ++i) {
    if (i)
      oss << ", ";
    const int pid = pids[i];
    oss << pid;
    const auto name = util::ProcessNameFromPid(pid);
    if (!name.empty())
      oss << "(" << name << ")";
  }
  oss << "]";
  return oss.str();
}

std::string DescribeConsumersHoldingDevice(const std::string &dev,
                                           int excludePid) {
  if (dev.empty())
    return {};

  util::OpenFileScanOptions opt;
  opt.exclude_pid = excludePid;
  opt.stop_at_first = false;

  std::string scanErr;
  const auto pids = util::PidsWithOpenFile(dev, opt, &scanErr);
  if (pids.empty() && scanErr.empty())
    return {};

  std::ostringstream oss;
  if (!pids.empty()) {
    oss << "\nConsumers holding " << dev << ": " << FormatPidsWithNames(pids)
        << "\nHint: close the consumer app(s) and retry.";
  }
  if (!scanErr.empty()) {
    oss << "\n(consumer scan warning: " << scanErr << ")";
  }
  return oss.str();
}

std::string DiffPipelineCfg(const CameraPipelineConfig &a,
                            const CameraPipelineConfig &b) {
  std::ostringstream oss;
  bool first = true;
  const auto add_s = [&](const char *k, const std::string &av,
                         const std::string &bv) {
    if (av == bv)
      return;
    if (!first)
      oss << ", ";
    first = false;
    oss << k << ":'" << av << "'->'" << bv << "'";
  };
  const auto add_i = [&](const char *k, int av, int bv) {
    if (av == bv)
      return;
    if (!first)
      oss << ", ";
    first = false;
    oss << k << ":" << av << "->" << bv;
  };
  const auto add_b = [&](const char *k, bool av, bool bv) {
    if (av == bv)
      return;
    if (!first)
      oss << ", ";
    first = false;
    oss << k << ":" << (av ? 1 : 0) << "->" << (bv ? 1 : 0);
  };

  add_s("in", a.input_device, b.input_device);
  add_s("out", a.output_device, b.output_device);
  add_i("capture_mode", static_cast<int>(a.capture_mode),
        static_cast<int>(b.capture_mode));
  add_i("w", a.width, b.width);
  add_i("h", a.height, b.height);
  add_i("fps", a.fps, b.fps);
  add_i("output_format", static_cast<int>(a.output_format),
        static_cast<int>(b.output_format));
  add_b("prefer_mjpeg", a.prefer_mjpeg, b.prefer_mjpeg);
  add_i("scaling_backend", static_cast<int>(a.scaling_backend),
        static_cast<int>(b.scaling_backend));
  add_i("compute_backend", static_cast<int>(a.compute_backend),
        static_cast<int>(b.compute_backend));
  add_b("allow_cpu_resize", a.allow_cpu_resize, b.allow_cpu_resize);

  return first ? std::string("(no changes)") : oss.str();
}

} // namespace

VirtualCameraService::VirtualCameraService()
    : VirtualCameraService(VirtualCameraServiceHooks{}) {}

VirtualCameraService::VirtualCameraService(VirtualCameraServiceHooks hooks)
    : hooks_(std::move(hooks)) {
  if (hooks_.create_pipeline) {
    pipeline_ = hooks_.create_pipeline();
  } else {
    pipeline_ = std::make_unique<CameraPipeline>();
  }
}

VirtualCameraService::~VirtualCameraService() { Stop(); }

void VirtualCameraService::SleepFor(std::chrono::milliseconds d) const {
  if (hooks_.sleep_for) {
    hooks_.sleep_for(d);
    return;
  }
  std::this_thread::sleep_for(d);
}

std::string VirtualCameraService::ChooseOutputDevice(std::string *error) const {
  if (hooks_.choose_output_device)
    return hooks_.choose_output_device(error);
  return ChooseDefaultOutputLoopback(error);
}

bool VirtualCameraService::OutputDeviceExists(const std::string &path,
                                              std::string *error) const {
  if (hooks_.output_device_exists)
    return hooks_.output_device_exists(path, error);

  if (path.empty()) {
    if (error)
      *error = "Output device path is empty.";
    return false;
  }

  struct stat st {};
  if (::stat(path.c_str(), &st) != 0) {
    if (error) {
      *error = "Output device " + path +
               " not available: " + std::string(std::strerror(errno));
    }
    return false;
  }

  if (error)
    error->clear();
  return true;
}

VideoConsumerSnapshot
VirtualCameraService::DetectConsumers(const std::string &dev,
                                      int excludePid) const {
  if (hooks_.detect_consumers)
    return hooks_.detect_consumers(dev, excludePid);

  VideoConsumerSnapshot out;
  util::OpenFileScanOptions opt;
  opt.exclude_pid = excludePid;
  opt.stop_at_first = false;

  const auto pids = util::PidsWithOpenFile(dev, opt, &out.error);
  out.count = static_cast<int>(pids.size());
  out.present = out.count > 0;
  return out;
}

bool VirtualCameraService::NeedsPipelineRestart(const CameraPipelineConfig &a,
                                                const CameraPipelineConfig &b) {
  // Mirror (and other future live-toggles) can be applied without restart.
  return a.input_device != b.input_device ||
         a.output_device != b.output_device ||
         a.capture_mode != b.capture_mode || a.width != b.width ||
         a.height != b.height || a.fps != b.fps ||
         a.output_format != b.output_format ||
         a.prefer_mjpeg != b.prefer_mjpeg ||
         a.scaling_backend != b.scaling_backend ||
         a.compute_backend != b.compute_backend ||
         a.allow_cpu_resize != b.allow_cpu_resize;
}

bool VirtualCameraService::Start(const VirtualCameraServiceConfig &cfg,
                                 std::string *error) {
  std::thread toJoin;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (running_) {
      if (error)
        *error = "VirtualCameraService already running.";
      return false;
    }
    if (!pipeline_) {
      if (error)
        *error = "Camera pipeline factory returned null.";
      return false;
    }

    if (th_.joinable()) {
      // A previous thread finished but wasn't joined (should be rare). Join it
      // now.
      toJoin = std::move(th_);
    }

    cfg_ = cfg;
    virtual_device_present_ = false;
    virtual_device_available_ = false;
    virtual_device_error_.clear();
    last_error_.clear();
    consumer_present_ = false;
    consumer_count_ = 0;
    consumer_error_.clear();
    pipeline_active_needed_ = false;
    pipeline_state_.clear();
    pipeline_idle_reason_.clear();
    last_consumer_seen_ = std::chrono::steady_clock::time_point{};
    next_start_retry_ = std::chrono::steady_clock::time_point{};

    // Reset supervisor counters/diagnostics for this service session.
    pipeline_start_attempts_ = 0;
    pipeline_starts_ = 0;
    pipeline_start_failures_ = 0;
    pipeline_stops_ = 0;
    pipeline_config_restarts_ = 0;
    stabilizing_ = false;
    thrash_events_10s_ = 0;
    last_transition_.clear();
    last_transition_at_ = std::chrono::steady_clock::time_point{};

    stop_.store(false);
    running_ = true;
  }

  if (toJoin.joinable()) {
    toJoin.join();
  }

  // Best-effort: open the v4l2loopback output immediately so apps like OBS can
  // discover it. This also helps avoid a small race where the supervisor thread
  // hasn't yet run its first poll cycle when the user opens an app.
  {
    VirtualCameraServiceConfig localCfg;
    {
      std::lock_guard<std::mutex> lock(mu_);
      localCfg = cfg_;
    }

    if (localCfg.pipeline.output_device.empty()) {
      std::string e;
      const auto out = ChooseOutputDevice(&e);
      if (!out.empty()) {
        localCfg.pipeline.output_device = out;
        std::lock_guard<std::mutex> lock(mu_);
        cfg_.pipeline.output_device = out;
      } else if (!e.empty()) {
        std::lock_guard<std::mutex> lock(mu_);
        virtual_device_present_ = false;
        virtual_device_available_ = false;
        virtual_device_error_ = e;
        pipeline_state_ = "device_unavailable";
        last_error_ = e;
      }
    }

    if (!localCfg.pipeline.output_device.empty()) {
      std::string oerr;
      if (!pipeline_->EnsureOutputOpen(localCfg.pipeline, &oerr)) {
        const int selfPid = static_cast<int>(::getpid());
        const std::string hint = DescribeConsumersHoldingDevice(
            localCfg.pipeline.output_device, selfPid);
        std::lock_guard<std::mutex> lock(mu_);
        virtual_device_present_ = true;
        virtual_device_available_ = false;
        virtual_device_error_ = "Output open failed: " + oerr;
        pipeline_state_ = "device_unavailable";
        last_error_ = "Output open failed: " + oerr + hint;
      } else {
        std::lock_guard<std::mutex> lock(mu_);
        virtual_device_present_ = true;
        virtual_device_available_ = true;
        virtual_device_error_.clear();
        if (last_error_.rfind("Output open failed:", 0) == 0)
          last_error_.clear();
      }
    }
  }

  // Launch supervisor thread outside the lock to avoid deadlocks.
  std::thread t(&VirtualCameraService::ThreadMain, this);
  {
    std::lock_guard<std::mutex> lock(mu_);
    th_ = std::move(t);
  }

  return true;
}

void VirtualCameraService::Stop() {
  stop_.store(true);

  std::thread toJoin;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (th_.joinable())
      toJoin = std::move(th_);
    running_ = false;
  }

  if (toJoin.joinable())
    toJoin.join();

  // Ensure the pipeline is stopped even if the supervisor thread exits early.
  if (pipeline_)
    pipeline_->Stop();

  std::lock_guard<std::mutex> lock(mu_);
  consumer_present_ = false;
  consumer_count_ = 0;
  pipeline_active_needed_ = false;
  if (pipeline_state_ != "device_unavailable" && pipeline_state_ != "disabled")
    pipeline_state_ = cfg_.enabled ? "idle_no_consumer" : "disabled";
}

void VirtualCameraService::UpdateConfig(const VirtualCameraServiceConfig &cfg) {
  std::lock_guard<std::mutex> lock(mu_);
  cfg_ = cfg;
}

VirtualCameraServiceStatus VirtualCameraService::Status() const {
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mu_);
  VirtualCameraServiceStatus s;
  s.service_running = running_;
  s.virtual_device_present = virtual_device_present_;
  s.virtual_device_available = virtual_device_available_;
  s.virtual_device_error = virtual_device_error_;
  s.consumer_present = consumer_present_;
  s.consumer_count = consumer_count_;
  s.consumer_error = consumer_error_;
  s.pipeline_active_needed = pipeline_active_needed_;
  s.pipeline_state = pipeline_state_;
  s.pipeline_idle_reason = pipeline_idle_reason_;

  s.pipeline_start_attempts = pipeline_start_attempts_;
  s.pipeline_starts = pipeline_starts_;
  s.pipeline_start_failures = pipeline_start_failures_;
  s.pipeline_stops = pipeline_stops_;
  s.pipeline_config_restarts = pipeline_config_restarts_;
  s.stabilizing = stabilizing_;
  s.thrash_events_10s = thrash_events_10s_;
  s.last_transition = last_transition_;

  if (last_transition_at_ != std::chrono::steady_clock::time_point{}) {
    s.last_transition_ms_ago =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_transition_at_)
            .count();
  }

  if (next_start_retry_ != std::chrono::steady_clock::time_point{}) {
    if (now >= next_start_retry_) {
      s.next_start_retry_ms = 0;
    } else {
      s.next_start_retry_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              next_start_retry_ - now)
              .count();
    }
  }

  s.pipeline = pipeline_ ? pipeline_->Status() : CameraPipelineStatus{};
  s.last_error = last_error_;
  return s;
}

VirtualCameraServiceConfig VirtualCameraService::Config() const {
  std::lock_guard<std::mutex> lock(mu_);
  return cfg_;
}

void VirtualCameraService::ThreadMain() {
  const int selfPid = static_cast<int>(::getpid());

  const bool dbg = DebugVcamSupervisor();
  int prevConsumerCount = -1;
  bool prevConsumerPresent = false;
  std::string prevPidList;
  std::string prevScanErr;

  bool prevEnabled = false;
  bool prevWantRunRequested = false;
  bool prevWantRun = false;
  bool prevBlocked = false;
  bool prevEffectsSuppressed = false;

  CameraPipelineStatus prevPipeline{};
  bool havePrevPipeline = false;

  if (dbg) {
    VcamDbg("supervisor start pid=" + std::to_string(selfPid));
  }

  // This runs independently of the camera pipeline thread.
  auto lastConsumerSeen = std::chrono::steady_clock::now();
  auto nextStartRetry = std::chrono::steady_clock::time_point{};

  auto consumerStableSince = std::chrono::steady_clock::time_point{};
  auto pipelineBecameRunningAt = std::chrono::steady_clock::time_point{};

  auto stabilizeUntil = std::chrono::steady_clock::time_point{};
  std::deque<std::chrono::steady_clock::time_point> thrashEvents;

  std::optional<studiocast::maxine::MaxineDiagnostics> maxineDiag;
  auto lastMaxineDiagAt = std::chrono::steady_clock::time_point{};

  std::optional<studiocast::open_cuda::OpenCudaDiagnostics> openCudaDiag;
  auto lastOpenCudaDiagAt = std::chrono::steady_clock::time_point{};

  bool haveAppliedCfg = false;
  CameraPipelineConfig appliedPipelineCfg{};

  while (!stop_.load()) {
    VirtualCameraServiceConfig cfg;
    {
      std::lock_guard<std::mutex> lock(mu_);
      cfg = cfg_;
    }
    bool outputAvailableForState = false;

    if (dbg && cfg.enabled != prevEnabled) {
      VcamDbg(std::string("cfg.enabled=") + (cfg.enabled ? "true" : "false"));
      prevEnabled = cfg.enabled;
    }

    // If the configured output device disappeared (e.g., v4l2loopback was
    // reloaded), clear it so we can re-probe.
    if (!cfg.pipeline.output_device.empty()) {
      std::string devErr;
      if (!OutputDeviceExists(cfg.pipeline.output_device, &devErr)) {
        {
          std::lock_guard<std::mutex> lock(mu_);
          virtual_device_present_ = false;
          virtual_device_available_ = false;
          virtual_device_error_ = devErr;
          last_error_ = devErr;
          cfg_.pipeline.output_device.clear();
          consumer_present_ = false;
          consumer_count_ = 0;
          consumer_error_.clear();
          pipeline_active_needed_ = false;
          pipeline_state_ = "device_unavailable";
          pipeline_idle_reason_.clear();
        }

        const auto beforeStop = pipeline_->Status();
        pipeline_->Stop();
        pipeline_->CloseOutput();
        if (beforeStop.running || beforeStop.starting) {
          std::lock_guard<std::mutex> lock(mu_);
          ++pipeline_stops_;
          last_transition_ = "stop_device_unavailable";
          last_transition_at_ = std::chrono::steady_clock::now();
        }
        haveAppliedCfg = false;

        SleepFor(std::chrono::milliseconds(std::max(50, cfg.consumer_poll_ms)));
        continue;
      }
    }

    // Ensure we have an output device to monitor (and to start the pipeline).
    if (cfg.pipeline.output_device.empty()) {
      std::string e;
      const auto out = ChooseOutputDevice(&e);
      if (!out.empty()) {
        cfg.pipeline.output_device = out;
        // Persist best-effort into shared cfg_ so we don't re-probe every loop.
        {
          std::lock_guard<std::mutex> lock(mu_);
          cfg_.pipeline.output_device = out;
        }
      } else {
        // No loopback yet: record error and wait.
        {
          std::lock_guard<std::mutex> lock(mu_);
          virtual_device_present_ = false;
          virtual_device_available_ = false;
          virtual_device_error_ = e;
          last_error_ = e;
          consumer_present_ = false;
          consumer_count_ = 0;
          consumer_error_.clear();
          pipeline_active_needed_ = false;
          pipeline_state_ = "device_unavailable";
          pipeline_idle_reason_.clear();
        }
        const auto beforeStop = pipeline_->Status();
        pipeline_->Stop();
        pipeline_->CloseOutput();
        if (beforeStop.running || beforeStop.starting) {
          std::lock_guard<std::mutex> lock(mu_);
          ++pipeline_stops_;
          last_transition_ = "stop_device_unavailable";
          last_transition_at_ = std::chrono::steady_clock::now();
        }
        haveAppliedCfg = false;
        SleepFor(std::chrono::milliseconds(std::max(50, cfg.consumer_poll_ms)));
        continue;
      }
    }

    // IMPORTANT: When v4l2loopback is loaded with exclusive_caps=1, the
    // device announces OUTPUT-only until a producer opens it. Many apps
    // will not list / open the virtual camera until it reports CAPTURE.
    //
    // Keep the output open even when we're idle so consumers can discover
    // the device. Heavy processing (camera capture + effects) is still
    // gated by consumerPresent below.
    {
      std::string oerr;
      if (!pipeline_->EnsureOutputOpen(cfg.pipeline, &oerr)) {
        const std::string hint =
            DescribeConsumersHoldingDevice(cfg.pipeline.output_device, selfPid);
        std::lock_guard<std::mutex> lock(mu_);
        virtual_device_present_ = true;
        virtual_device_available_ = false;
        virtual_device_error_ = "Output open failed: " + oerr;
        pipeline_active_needed_ = false;
        pipeline_state_ = "device_unavailable";
        pipeline_idle_reason_.clear();
        last_error_ = "Output open failed: " + oerr + hint;
      } else {
        outputAvailableForState = true;
        std::lock_guard<std::mutex> lock(mu_);
        virtual_device_present_ = true;
        virtual_device_available_ = true;
        if (last_error_ == virtual_device_error_ ||
            last_error_.rfind("Output open failed:", 0) == 0) {
          last_error_.clear();
        }
        virtual_device_error_.clear();
      }
    }

    // Detect consumers of the loopback device.
    bool consumerPresent = false;
    int consumerCount = 0;
    std::string consumerError;
    {
      const auto snapshot =
          DetectConsumers(cfg.pipeline.output_device, selfPid);
      consumerCount = snapshot.count;
      consumerPresent = snapshot.present;
      consumerError = snapshot.error;

      if (dbg) {
        const std::string pidList =
            consumerPresent ? ("count=" + std::to_string(consumerCount)) : "[]";
        if (consumerCount != prevConsumerCount ||
            consumerPresent != prevConsumerPresent || pidList != prevPidList ||
            consumerError != prevScanErr) {
          std::ostringstream oss;
          oss << "consumer_scan dev='" << cfg.pipeline.output_device
              << "' present=" << (consumerPresent ? 1 : 0)
              << " count=" << consumerCount << " pids=" << pidList;
          if (!consumerError.empty())
            oss << " scanErr=\"" << consumerError << "\"";
          VcamDbg(oss.str());
          prevConsumerCount = consumerCount;
          prevConsumerPresent = consumerPresent;
          prevPidList = pidList;
          prevScanErr = consumerError;
        }
      }

      // Non-fatal: keep running even if scan can't see all processes.
      if (!consumerError.empty()) {
        std::lock_guard<std::mutex> lock(mu_);
        consumer_error_ = consumerError;
        last_error_ = consumerError;
      } else {
        std::lock_guard<std::mutex> lock(mu_);
        if (!consumer_error_.empty() && last_error_ == consumer_error_)
          last_error_.clear();
        consumer_error_.clear();
        if (last_error_.rfind("stat(", 0) == 0 ||
            last_error_.rfind("Failed to iterate /proc:", 0) == 0) {
          last_error_.clear();
        }
      }
    }

    const auto now = std::chrono::steady_clock::now();

    // Supervisor diagnostics helpers (Phase 2).

    const auto CountStartAttempt = [&]() {
      std::lock_guard<std::mutex> lock(mu_);
      ++pipeline_start_attempts_;
    };

    const auto CountStartSuccess = [&](const char *reason) {
      std::lock_guard<std::mutex> lock(mu_);
      ++pipeline_starts_;
      last_transition_ = reason ? std::string(reason) : std::string();
      last_transition_at_ = now;
    };

    const auto CountStartFailure = [&]() {
      std::lock_guard<std::mutex> lock(mu_);
      ++pipeline_start_failures_;
      last_transition_ = "start_failed";
      last_transition_at_ = now;
    };

    const auto CountStop = [&](const char *reason, bool configRestart) {
      std::lock_guard<std::mutex> lock(mu_);
      ++pipeline_stops_;
      if (configRestart)
        ++pipeline_config_restarts_;
      last_transition_ = reason ? std::string(reason) : std::string();
      last_transition_at_ = now;
    };

    if (consumerPresent)
      lastConsumerSeen = now;

    if (consumerPresent) {
      if (consumerStableSince == std::chrono::steady_clock::time_point{})
        consumerStableSince = now;
    } else {
      consumerStableSince = std::chrono::steady_clock::time_point{};
    }

    auto effects_for_pipeline = cfg.pipeline.effects;
    bool effectsSuppressed = false;
    std::string suppressMsg;

    const bool wantRunRequested =
        cfg.enabled && outputAvailableForState &&
        (cfg.always_on || consumerPresent);

    const int startGraceMs = std::max(0, cfg.start_grace_ms);
    const bool consumerStable =
        cfg.always_on ||
        (consumerPresent &&
         (startGraceMs == 0 ||
          (consumerStableSince != std::chrono::steady_clock::time_point{} &&
           (now - consumerStableSince) >=
               std::chrono::milliseconds(startGraceMs))));

    // Self-stabilization mode: if we observe repeated start/stop flapping,
    // temporarily ignore consumer disconnects while enabled.
    const bool stabilizing =
        cfg.enabled &&
        (stabilizeUntil != std::chrono::steady_clock::time_point{} &&
         now < stabilizeUntil);

    bool wantRun = cfg.enabled && outputAvailableForState &&
                   (cfg.always_on || stabilizing || consumerStable);

    // Gate expensive capture/processing threads based on engine availability.
    // Keep loopback output alive (see EnsureOutputOpen above) so consumers
    // still see a stable device.
    bool blocked = false;
    std::string blockedMsg;

    if (wantRun) {
      const auto plan =
          effects::BuildBroadcastEffectsPlan(cfg.pipeline.effects);
      const std::set<std::string> planned(plan.ordered_effect_ids.begin(),
                                          plan.ordered_effect_ids.end());
      const auto has = [&](std::string_view id) {
        return planned.count(std::string(id)) != 0;
      };

      const bool wants_vb =
          has(effects::contract::kEffectIdVirtualBackgroundBlur) ||
          has(effects::contract::kEffectIdVirtualBackgroundRemove) ||
          has(effects::contract::kEffectIdVirtualBackgroundReplace);

      const bool wants_denoise =
          has(effects::contract::kEffectIdVideoNoiseRemoval);
      const bool wants_auto_frame = has(effects::contract::kEffectIdAutoFrame);
      const bool wants_key_light =
          has(effects::contract::kEffectIdVirtualKeyLight);
      const bool wants_eye_contact = has(effects::contract::kEffectIdEyeContact);
      const bool wants_vignette = has(effects::contract::kEffectIdVignette);
      const bool wants_open_cuda_fx =
          wants_vb || wants_denoise || wants_auto_frame || wants_key_light;
      const bool wants_compute_fx = wants_open_cuda_fx || wants_eye_contact ||
                                    wants_vignette;
      const bool compute_backend_forces_vulkan =
          cfg.pipeline.compute_backend == ComputeBackendPreference::vulkan;

      if (wants_compute_fx &&
          cfg.pipeline.compute_backend == ComputeBackendPreference::cpu) {
        effectsSuppressed = true;
        suppressMsg = "CPU compute backend selected; GPU-backed video "
                      "effects are disabled.";
        effects_for_pipeline.video_noise_removal.enabled = false;
        effects_for_pipeline.auto_frame.enabled = false;
        effects_for_pipeline.eye_contact.enabled = false;
        effects_for_pipeline.virtual_key_light.enabled = false;
        effects_for_pipeline.virtual_background.mode =
            effects::VirtualBackgroundMode::none;
        effects_for_pipeline.vignette.enabled = false;
      }

      // Effects that currently require Maxine (no Open CUDA/Open Video
      // fallback). NOTE: Eye Contact has an Open Video fallback in AUTO engine
      // mode, so it should not block the pipeline when Maxine is unavailable.
      const bool wants_maxine_only = false;

      const auto ttl = std::chrono::seconds(2);

      // --- Maxine gate ---
      bool needMaxineGate = false;
      bool needMaxineDiag = false;
      if (compute_backend_forces_vulkan) {
        needMaxineGate = false;
        needMaxineDiag = false;
      } else if (cfg.pipeline.effects.engine ==
                 effects::EffectsEnginePreference::maxine) {
        // Forced Maxine: any Maxine-backed effect blocks the pipeline when
        // unavailable.
        needMaxineGate =
            effects::WantsMaxineForPlannedEffects(cfg.pipeline.effects);
        needMaxineDiag = needMaxineGate;
      } else if (cfg.pipeline.effects.engine ==
                 effects::EffectsEnginePreference::auto_select) {
        // Auto-select: only gate Maxine for effects that have no Open CUDA
        // fallback.
        needMaxineGate = wants_maxine_only;
        // Still cache Maxine diagnostics if we need to decide whether Open CUDA
        // effects can run on Maxine.
        needMaxineDiag = needMaxineGate || wants_open_cuda_fx;
      }

      if (!effectsSuppressed && needMaxineDiag) {
        if (!maxineDiag.has_value() ||
            lastMaxineDiagAt == std::chrono::steady_clock::time_point{} ||
            (now - lastMaxineDiagAt) >= ttl) {
          studiocast::maxine::MaxineManager mgr;
          maxineDiag = mgr.Diagnose(false);
          lastMaxineDiagAt = now;
        }
      }

      if (!effectsSuppressed && needMaxineGate && maxineDiag.has_value()) {
        const auto gate =
            effects::EvaluateMaxineGate(cfg.pipeline.effects, *maxineDiag);
        if (!gate.ok) {
          blocked = true;
          blockedMsg = gate.message;
          wantRun = false;
        }
      }

      // --- Open CUDA gate ---
      if (!effectsSuppressed && !blocked && !compute_backend_forces_vulkan &&
          wants_open_cuda_fx) {
        bool needOpenCudaGate = false;
        if (cfg.pipeline.effects.engine ==
            effects::EffectsEnginePreference::open_cuda) {
          needOpenCudaGate = true;
        } else if (cfg.pipeline.effects.engine ==
                   effects::EffectsEnginePreference::auto_select) {
          // Auto-select: prefer Maxine when available; otherwise require Open
          // CUDA for effects that have an Open CUDA implementation.
          bool vb_available_in_maxine = false;
          bool denoise_available_in_maxine = false;
          bool auto_frame_available_in_maxine = false;
          bool key_light_available_in_maxine = false;
          if (maxineDiag.has_value()) {
            const std::set<std::string> mx_avail(
                maxineDiag->available_effects.begin(),
                maxineDiag->available_effects.end());
            if (wants_denoise) {
              denoise_available_in_maxine =
                  mx_avail.count(std::string(
                      effects::contract::kEffectIdVideoNoiseRemoval)) != 0;
            }
            if (wants_vb) {
              vb_available_in_maxine = true;
              if (has(effects::contract::kEffectIdVirtualBackgroundBlur) &&
                  !mx_avail.count(std::string(
                      effects::contract::kEffectIdVirtualBackgroundBlur))) {
                vb_available_in_maxine = false;
              }
              if (has(effects::contract::kEffectIdVirtualBackgroundRemove) &&
                  !mx_avail.count(std::string(
                      effects::contract::kEffectIdVirtualBackgroundRemove))) {
                vb_available_in_maxine = false;
              }
              if (has(effects::contract::kEffectIdVirtualBackgroundReplace) &&
                  !mx_avail.count(std::string(
                      effects::contract::kEffectIdVirtualBackgroundReplace))) {
                vb_available_in_maxine = false;
              }
            }
            if (wants_auto_frame) {
              auto_frame_available_in_maxine =
                  mx_avail.count(
                      std::string(effects::contract::kEffectIdAutoFrame)) != 0;
            }
            if (wants_key_light) {
              key_light_available_in_maxine =
                  mx_avail.count(std::string(
                      effects::contract::kEffectIdVirtualKeyLight)) != 0;
            }
          }

          // If any requested Open CUDA-capable effect is not available in
          // Maxine, require Open CUDA.
          needOpenCudaGate =
              (wants_denoise && !denoise_available_in_maxine) ||
              (wants_vb && !vb_available_in_maxine) ||
              (wants_auto_frame && !auto_frame_available_in_maxine) ||
              (wants_key_light && !key_light_available_in_maxine);
        }

        if (needOpenCudaGate &&
            effects::WantsOpenCudaForPlannedEffects(cfg.pipeline.effects)) {
          if (!openCudaDiag.has_value() ||
              lastOpenCudaDiagAt == std::chrono::steady_clock::time_point{} ||
              (now - lastOpenCudaDiagAt) >= ttl) {
            openCudaDiag = studiocast::open_cuda::DiagnoseOpenCudaDefault();
            lastOpenCudaDiagAt = now;
          }

          if (openCudaDiag.has_value()) {
            const auto gate = effects::EvaluateOpenCudaGate(
                cfg.pipeline.effects, *openCudaDiag);
            if (!gate.ok) {
              effectsSuppressed = true;
              suppressMsg = gate.message;

              // Selectively suppress only the effects that Open CUDA reports as
              // unavailable.
              const auto oc_has = [&](std::string_view id) {
                const std::string sid(id);
                return std::find(openCudaDiag->available_effects.begin(),
                                 openCudaDiag->available_effects.end(),
                                 sid) != openCudaDiag->available_effects.end();
              };

              if (effects_for_pipeline.video_noise_removal.enabled &&
                  !oc_has(effects::contract::kEffectIdVideoNoiseRemoval)) {
                effects_for_pipeline.video_noise_removal.enabled = false;
              }

              if (effects_for_pipeline.auto_frame.enabled &&
                  !oc_has(effects::contract::kEffectIdAutoFrame)) {
                effects_for_pipeline.auto_frame.enabled = false;
              }

              if (effects_for_pipeline.virtual_key_light.enabled &&
                  !oc_has(effects::contract::kEffectIdVirtualKeyLight)) {
                effects_for_pipeline.virtual_key_light.enabled = false;
              }

              if (effects_for_pipeline.virtual_background.mode !=
                  effects::VirtualBackgroundMode::none) {
                std::string_view vb_id;
                switch (effects_for_pipeline.virtual_background.mode) {
                case effects::VirtualBackgroundMode::blur:
                  vb_id = effects::contract::kEffectIdVirtualBackgroundBlur;
                  break;
                case effects::VirtualBackgroundMode::remove:
                  vb_id = effects::contract::kEffectIdVirtualBackgroundRemove;
                  break;
                case effects::VirtualBackgroundMode::replace:
                  vb_id = effects::contract::kEffectIdVirtualBackgroundReplace;
                  break;
                default:
                  break;
                }
                if (!vb_id.empty() && !oc_has(vb_id)) {
                  effects_for_pipeline.virtual_background.mode =
                      effects::VirtualBackgroundMode::none;
                }
              }
            }
          }
        }
      }
    }

    if (dbg) {
      if (wantRunRequested != prevWantRunRequested || wantRun != prevWantRun ||
          blocked != prevBlocked ||
          effectsSuppressed != prevEffectsSuppressed) {
        std::ostringstream oss;
        oss << "decision enabled=" << (cfg.enabled ? 1 : 0)
            << " consumerPresent=" << (consumerPresent ? 1 : 0)
            << " consumerStable=" << (consumerStable ? 1 : 0)
            << " stabilizing=" << (stabilizing ? 1 : 0)
            << " wantRunRequested=" << (wantRunRequested ? 1 : 0)
            << " wantRun=" << (wantRun ? 1 : 0);
        if (blocked)
          oss << " blocked=\"" << blockedMsg << "\"";
        if (effectsSuppressed)
          oss << " suppressed=\"" << suppressMsg << "\"";
        VcamDbg(oss.str());
        prevWantRunRequested = wantRunRequested;
        prevWantRun = wantRun;
        prevBlocked = blocked;
        prevEffectsSuppressed = effectsSuppressed;
      }
    }

    // Apply effects live regardless of consumer state.
    pipeline_->SetEffects(effects_for_pipeline);

    auto pipelineCfgForPipeline = cfg.pipeline;
    pipelineCfgForPipeline.effects = effects_for_pipeline;

    auto pst = pipeline_->Status();
    if (blocked) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (last_error_ != blockedMsg)
          last_error_ = blockedMsg;
      }

      if (pst.running || pst.starting) {
        if (dbg)
          VcamDbg("pipeline Stop: blocked");
        pipeline_->Stop();
        CountStop("stop_blocked", false);
        haveAppliedCfg = false;
        nextStartRetry = std::chrono::steady_clock::time_point{};
        pst = pipeline_->Status();
      }
    } else if (effectsSuppressed) {
      std::lock_guard<std::mutex> lock(mu_);
      if (last_error_ != suppressMsg)
        last_error_ = suppressMsg;
    }

    // Restart if config changed and we are running.
    if (pst.running && haveAppliedCfg &&
        NeedsPipelineRestart(appliedPipelineCfg, pipelineCfgForPipeline)) {
      if (dbg)
        VcamDbg("pipeline Stop: config restart " +
                DiffPipelineCfg(appliedPipelineCfg, pipelineCfgForPipeline));
      pipeline_->Stop();
      CountStop("stop_config_restart", true);
      haveAppliedCfg = false;
      nextStartRetry = std::chrono::steady_clock::time_point{};
    }

    if (wantRun) {
      if (!pst.running && !pst.starting) {
        if (nextStartRetry != std::chrono::steady_clock::time_point{} &&
            now < nextStartRetry) {
          // still in backoff window
        } else {
          const auto recordThrashEvent = [&](const char *what) {
            (void)what;
            thrashEvents.push_back(now);
            constexpr auto kWindow = std::chrono::seconds(10);
            while (!thrashEvents.empty() &&
                   (now - thrashEvents.front()) > kWindow) {
              thrashEvents.pop_front();
            }
            constexpr std::size_t kThreshold = 6;
            if (thrashEvents.size() >= kThreshold &&
                (stabilizeUntil == std::chrono::steady_clock::time_point{} ||
                 now >= stabilizeUntil)) {
              stabilizeUntil = now + std::chrono::seconds(5);
              if (dbg) {
                std::ostringstream oss;
                oss << "thrash_detected events=" << thrashEvents.size()
                    << " -> stabilize 5s";
                VcamDbg(oss.str());
              }
            }
          };

          if (dbg) {
            std::ostringstream oss;
            oss << "pipeline Start attempt enabled=" << (cfg.enabled ? 1 : 0)
                << " consumerPresent=" << (consumerPresent ? 1 : 0) << " in='"
                << (pipelineCfgForPipeline.input_device.empty()
                        ? std::string("(auto)")
                        : pipelineCfgForPipeline.input_device)
                << "' out='"
                << (pipelineCfgForPipeline.output_device.empty()
                        ? std::string("(auto)")
                        : pipelineCfgForPipeline.output_device)
                << "' mode="
                << static_cast<int>(pipelineCfgForPipeline.capture_mode)
                << " w=" << pipelineCfgForPipeline.width
                << " h=" << pipelineCfgForPipeline.height
                << " fps=" << pipelineCfgForPipeline.fps;
            VcamDbg(oss.str());
          }

          recordThrashEvent("start_attempt");
          CountStartAttempt();
          std::string perr;
          if (!pipeline_->Start(pipelineCfgForPipeline, &perr)) {
            {
              std::lock_guard<std::mutex> lock(mu_);
              last_error_ = "Pipeline start failed: " + perr;
            }
            CountStartFailure();
            if (dbg)
              VcamDbg(std::string("pipeline Start FAILED: ") + perr);
            // Back off a bit; many apps probe cameras aggressively.
            nextStartRetry = now + std::chrono::seconds(2);
          } else {
            {
              std::lock_guard<std::mutex> lock(mu_);
              if (!effectsSuppressed &&
                  last_error_.rfind("Pipeline start failed:", 0) == 0) {
                last_error_.clear();
              }
              virtual_device_present_ = true;
              virtual_device_available_ = true;
              virtual_device_error_.clear();
            }
            outputAvailableForState = true;
            haveAppliedCfg = true;
            appliedPipelineCfg = pipelineCfgForPipeline;
            nextStartRetry = std::chrono::steady_clock::time_point{};
            pipelineBecameRunningAt = now;
            if (cfg.always_on) {
              CountStartSuccess("start_always_on");
            } else if (stabilizing) {
              CountStartSuccess("start_stabilizing");
            } else {
              CountStartSuccess("start_consumer");
            }
            if (dbg)
              VcamDbg("pipeline Start OK");
          }
        }
      }
    } else {
      if (pst.running || pst.starting) {
        const int graceMs = cfg.enabled ? std::max(0, cfg.stop_grace_ms) : 0;
        const auto grace = std::chrono::milliseconds(graceMs);

        const int minRunMs = std::max(0, cfg.min_run_ms);
        const bool withinMinRun =
            cfg.enabled && minRunMs > 0 &&
            (pipelineBecameRunningAt !=
                 std::chrono::steady_clock::time_point{} &&
             (now - pipelineBecameRunningAt) <
                 std::chrono::milliseconds(minRunMs));

        if (!withinMinRun &&
            (graceMs == 0 || (now - lastConsumerSeen) >= grace)) {
          if (dbg) {
            const auto idleMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - lastConsumerSeen)
                    .count();
            std::ostringstream oss;
            oss << "pipeline Stop: wantRun=0 enabled=" << (cfg.enabled ? 1 : 0)
                << " consumerPresent=" << (consumerPresent ? 1 : 0)
                << " idleMs=" << idleMs << " graceMs=" << graceMs;
            VcamDbg(oss.str());
          }

          // Record stop as a thrash event.
          thrashEvents.push_back(now);
          constexpr auto kWindow = std::chrono::seconds(10);
          while (!thrashEvents.empty() &&
                 (now - thrashEvents.front()) > kWindow) {
            thrashEvents.pop_front();
          }
          constexpr std::size_t kThreshold = 6;
          if (thrashEvents.size() >= kThreshold &&
              (stabilizeUntil == std::chrono::steady_clock::time_point{} ||
               now >= stabilizeUntil)) {
            stabilizeUntil = now + std::chrono::seconds(5);
            if (dbg) {
              std::ostringstream oss;
              oss << "thrash_detected events=" << thrashEvents.size()
                  << " -> stabilize 5s";
              VcamDbg(oss.str());
            }
          }

          pipeline_->Stop();
          if (cfg.enabled) {
            CountStop("stop_no_consumers", false);
          } else {
            CountStop("stop_disabled", false);
          }
          haveAppliedCfg = false;
        }
      }
    }

    if (dbg) {
      const auto st = pipeline_->Status();
      if (!havePrevPipeline || st.running != prevPipeline.running ||
          st.starting != prevPipeline.starting ||
          st.input_device != prevPipeline.input_device ||
          st.output_device != prevPipeline.output_device ||
          st.last_error != prevPipeline.last_error ||
          st.effects_note != prevPipeline.effects_note) {
        std::ostringstream oss;
        oss << "pipeline_status running=" << (st.running ? 1 : 0)
            << " starting=" << (st.starting ? 1 : 0)
            << " in=" << (st.input_device.empty() ? "(auto)" : st.input_device)
            << " out="
            << (st.output_device.empty() ? "(auto)" : st.output_device);
        if (!st.last_error.empty())
          oss << " last_error=\"" << st.last_error << "\"";
        if (!st.effects_note.empty())
          oss << " note=\"" << st.effects_note << "\"";
        VcamDbg(oss.str());
        prevPipeline = st;
        havePrevPipeline = true;
      }
    }

    const auto stateStatus = pipeline_->Status();
    std::string pipelineState;
    std::string idleReason;
    bool activeNeeded = wantRunRequested;

    const bool backingOff =
        nextStartRetry != std::chrono::steady_clock::time_point{} &&
        now < nextStartRetry && (cfg.always_on || consumerPresent);

    if (!cfg.enabled) {
      pipelineState = "disabled";
      idleReason = "disabled";
      activeNeeded = false;
    } else if (stateStatus.running) {
      pipelineState = "running";
    } else if (stateStatus.starting) {
      pipelineState = "starting";
    } else if (!consumerError.empty() && !consumerPresent && !cfg.always_on) {
      pipelineState = "consumer_detection_error";
      activeNeeded = false;
    } else if (!outputAvailableForState) {
      pipelineState = "device_unavailable";
      activeNeeded = false;
    } else if (blocked) {
      pipelineState = "blocked";
    } else if (backingOff) {
      pipelineState = "backing_off";
      idleReason = "start_retry_backoff";
    } else if (consumerPresent && !consumerStable && !cfg.always_on) {
      pipelineState = "waiting_for_stable_consumer";
      idleReason = "start_grace";
    } else if (cfg.always_on) {
      pipelineState = "idle_always_on";
      idleReason = "waiting_to_start";
    } else {
      pipelineState = "idle_no_consumer";
      idleReason = "no_consumer";
      activeNeeded = false;
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      consumer_present_ = consumerPresent;
      consumer_count_ = consumerCount;
      consumer_error_ = consumerError;
      pipeline_active_needed_ = activeNeeded;
      pipeline_state_ = pipelineState;
      pipeline_idle_reason_ = idleReason;
      last_consumer_seen_ = lastConsumerSeen;
      next_start_retry_ = nextStartRetry;
      stabilizing_ = stabilizing;
      thrash_events_10s_ = static_cast<int>(thrashEvents.size());
    }

    SleepFor(std::chrono::milliseconds(std::max(50, cfg.consumer_poll_ms)));
  }

  // Ensure pipeline stops when the supervisor exits.
  pipeline_->Stop();

  std::lock_guard<std::mutex> lock(mu_);
  running_ = false;
}

} // namespace studiocast::video
