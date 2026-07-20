#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/audio/audio_backend_resolver.h"
#include "core/audio/audio_device_safety.h"
#include "core/audio/audio_pipeline.h"
#include "core/audio/audio_processor.h"
#include "core/audio/pulse/pactl.h"
#include "core/audio/virtual_audio_service.h"
#include "core/audio/virtual_mic.h"
#include "core/audio/virtual_mic_state.h"
#include "core/audio/virtual_speaker.h"
#include "core/audio/virtual_speaker_state.h"
#include "core/util/exec.h"

namespace {

using studiocast::audio::AudioBackendAvailability;
using studiocast::audio::AudioConsumerSnapshot;
using studiocast::audio::AudioPipeline;
using studiocast::audio::AudioPipelineConfig;
using studiocast::audio::AudioPipelineHooks;
using studiocast::audio::AudioPipelineIo;
using studiocast::audio::AudioPipelineRunner;
using studiocast::audio::AudioPipelineStats;
using studiocast::audio::AudioProcessor;
using studiocast::audio::VirtualAudioService;
using studiocast::audio::VirtualAudioServiceConfig;
using studiocast::audio::VirtualAudioServiceHooks;
using studiocast::audio::VirtualMicState;
using studiocast::audio::VirtualSpeakerState;

using namespace std::chrono_literals;

class ScopedPactlExecHook final {
public:
  explicit ScopedPactlExecHook(
      studiocast::audio::pulse::PactlExecCaptureHook hook) {
    studiocast::audio::pulse::SetPactlExecCaptureHookForTesting(
        std::move(hook));
  }

  ~ScopedPactlExecHook() {
    studiocast::audio::pulse::SetPactlExecCaptureHookForTesting(nullptr);
  }

  ScopedPactlExecHook(const ScopedPactlExecHook &) = delete;
  ScopedPactlExecHook &operator=(const ScopedPactlExecHook &) = delete;
};

class EnvGuard final {
public:
  EnvGuard(const char *name, const std::string &value) : name_(name) {
    if (const char *old = std::getenv(name)) {
      had_old_ = true;
      old_ = old;
    }
    ::setenv(name, value.c_str(), 1);
  }

  ~EnvGuard() {
    if (had_old_) {
      ::setenv(name_, old_.c_str(), 1);
    } else {
      ::unsetenv(name_);
    }
  }

  EnvGuard(const EnvGuard &) = delete;
  EnvGuard &operator=(const EnvGuard &) = delete;

private:
  const char *name_;
  bool had_old_ = false;
  std::string old_;
};

class ScopedXdgStateHome final {
public:
  ScopedXdgStateHome() {
    if (const char *old = std::getenv("XDG_STATE_HOME")) {
      had_old_ = true;
      old_ = old;
    }

    std::error_code ec;
    auto base = std::filesystem::temp_directory_path(ec);
    if (ec)
      base = "/tmp";

    static std::atomic<int> counter{0};
    path_ =
        base /
        ("studiocast-audio-tests-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()) +
         "-" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));

    std::filesystem::create_directories(path_, ec);
    ::setenv("XDG_STATE_HOME", path_.string().c_str(), 1);
  }

  ~ScopedXdgStateHome() {
    if (had_old_) {
      ::setenv("XDG_STATE_HOME", old_.c_str(), 1);
    } else {
      ::unsetenv("XDG_STATE_HOME");
    }

    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  ScopedXdgStateHome(const ScopedXdgStateHome &) = delete;
  ScopedXdgStateHome &operator=(const ScopedXdgStateHome &) = delete;

private:
  bool had_old_ = false;
  std::string old_;
  std::filesystem::path path_;
};

studiocast::util::ExecResult ExecResult(int exit_code,
                                        std::string stdout_str = {}) {
  studiocast::util::ExecResult result;
  result.exit_code = exit_code;
  result.stdout_str = std::move(stdout_str);
  return result;
}

bool CommandWasRun(const std::vector<std::string> &commands,
                   const std::string &needle) {
  for (const auto &command : commands) {
    if (command.find(needle) != std::string::npos)
      return true;
  }
  return false;
}

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

AudioConsumerSnapshot ConsumerSnapshot(bool present, int count = 1) {
  AudioConsumerSnapshot out;
  out.present = present;
  out.count = present ? count : 0;
  return out;
}

studiocast::audio::pulse::PactlExecCaptureHook SafeMicrophoneSourcePactlHook() {
  return [](const std::string &command) {
    if (command == "pactl get-default-source 2>&1")
      return ExecResult(0, "physical_test_mic\n");
    if (command == "pactl list short sources 2>&1") {
      return ExecResult(0, "1\tstudiocast_sink.monitor\tmodule-null-sink.c\t"
                           "s16le 2ch 48000Hz\n"
                           "2\tphysical_test_mic\tmodule-alsa-card.c\t"
                           "s16le 2ch 48000Hz\n");
    }
    return ExecResult(99, "unexpected command: " + command);
  };
}

void HookMicrophoneConsumerFlag(VirtualAudioServiceHooks *hooks,
                                std::atomic<bool> *present) {
  hooks->detect_microphone_consumers = [present] {
    return ConsumerSnapshot(present->load(std::memory_order_relaxed));
  };
}

void HookSpeakerConsumerFlag(VirtualAudioServiceHooks *hooks,
                             std::atomic<bool> *present) {
  hooks->detect_speaker_consumers = [present] {
    return ConsumerSnapshot(present->load(std::memory_order_relaxed));
  };
}

class DeadPipeline final : public AudioPipelineRunner {
public:
  explicit DeadPipeline(std::atomic<int> *stop_calls)
      : stop_calls_(stop_calls) {}

  bool Start(const AudioPipelineConfig &, std::string *error) override {
    if (error)
      error->clear();
    return true;
  }

  void Stop() override {
    if (stop_calls_)
      stop_calls_->fetch_add(1, std::memory_order_relaxed);
  }

  AudioPipelineStats GetStats() const override {
    AudioPipelineStats stats;
    stats.running = false;
    return stats;
  }

private:
  std::atomic<int> *stop_calls_ = nullptr;
};

class StartFailPipeline final : public AudioPipelineRunner {
public:
  bool Start(const AudioPipelineConfig &, std::string *error) override {
    if (error)
      *error = "synthetic start failure";
    return false;
  }

  void Stop() override {}

  AudioPipelineStats GetStats() const override { return {}; }
};

class FixedStatsPipeline final : public AudioPipelineRunner {
public:
  FixedStatsPipeline(bool running, std::string last_error,
                     std::atomic<int> *stop_calls = nullptr)
      : stop_calls_(stop_calls) {
    stats_.running = running;
    stats_.last_error = std::move(last_error);
  }

  FixedStatsPipeline(AudioPipelineStats stats,
                     std::atomic<int> *stop_calls = nullptr)
      : stats_(std::move(stats)), stop_calls_(stop_calls) {}

  bool Start(const AudioPipelineConfig &, std::string *error) override {
    if (error)
      error->clear();
    return true;
  }

  void Stop() override {
    if (stop_calls_)
      stop_calls_->fetch_add(1, std::memory_order_relaxed);
  }

  AudioPipelineStats GetStats() const override { return stats_; }

private:
  AudioPipelineStats stats_{};
  std::atomic<int> *stop_calls_ = nullptr;
};

class CopyProcessor final : public AudioProcessor {
public:
  bool Process(const float *in, float *out, std::uint32_t frames,
               std::uint32_t channels, std::string *error) override {
    if (error)
      error->clear();
    std::memcpy(out, in,
                static_cast<std::size_t>(frames) * channels * sizeof(float));
    return true;
  }
};

class CountingCopyProcessor final : public AudioProcessor {
public:
  explicit CountingCopyProcessor(std::atomic<int> *reset_calls)
      : reset_calls_(reset_calls) {}

  bool Process(const float *in, float *out, std::uint32_t frames,
               std::uint32_t channels, std::string *error) override {
    if (error)
      error->clear();
    std::memcpy(out, in,
                static_cast<std::size_t>(frames) * channels * sizeof(float));
    return true;
  }

  void Reset() override {
    if (reset_calls_)
      reset_calls_->fetch_add(1, std::memory_order_relaxed);
  }

private:
  std::atomic<int> *reset_calls_ = nullptr;
};

enum class BlockMode {
  kRead,
  kWrite,
};

struct BlockingIoState {
  std::mutex mu;
  std::condition_variable cv;
  bool block_entered = false;
  bool stop_requested = false;
  int request_stop_calls = 0;
};

class BlockingIo final : public AudioPipelineIo {
public:
  BlockingIo(std::shared_ptr<BlockingIoState> state, BlockMode mode,
             std::chrono::milliseconds block_timeout)
      : state_(std::move(state)), mode_(mode), block_timeout_(block_timeout) {}

  bool Open(const AudioPipelineConfig &, std::string *error) override {
    if (error)
      error->clear();
    return true;
  }

  bool Read(void *dst, std::size_t bytes, std::string *error) override {
    if (mode_ != BlockMode::kRead) {
      std::memset(dst, 0, bytes);
      if (error)
        error->clear();
      return true;
    }
    return Block("capture read", error);
  }

  bool Write(const void *, std::size_t, std::string *error) override {
    if (mode_ != BlockMode::kWrite) {
      if (error)
        error->clear();
      return true;
    }
    return Block("playback write", error);
  }

  bool GetCaptureLatencyUs(std::uint64_t *) override { return false; }
  bool GetPlaybackLatencyUs(std::uint64_t *) override { return false; }
  void Flush() override {}

  void RequestStop() override {
    std::lock_guard<std::mutex> lock(state_->mu);
    state_->stop_requested = true;
    ++state_->request_stop_calls;
    state_->cv.notify_all();
  }

private:
  bool Block(const char *label, std::string *error) {
    std::unique_lock<std::mutex> lock(state_->mu);
    state_->block_entered = true;
    state_->cv.notify_all();

    const bool stopped = state_->cv.wait_for(
        lock, block_timeout_, [&] { return state_->stop_requested; });
    if (stopped) {
      if (error)
        error->clear();
      return false;
    }

    if (error)
      *error = std::string(label) + " remained blocked";
    return false;
  }

  std::shared_ptr<BlockingIoState> state_;
  BlockMode mode_;
  std::chrono::milliseconds block_timeout_;
};

struct ResettingOpenIoState {
  std::mutex mu;
  std::condition_variable cv;
  bool open_started = false;
  bool stop_requested = false;
  int request_stop_calls = 0;
};

struct BlockingFlushIoState {
  std::mutex mu;
  std::condition_variable cv;
  bool flush_entered = false;
  bool stop_requested = false;
  int request_stop_calls = 0;
};

class BlockingFlushIo final : public AudioPipelineIo {
public:
  explicit BlockingFlushIo(std::shared_ptr<BlockingFlushIoState> state,
                           std::chrono::milliseconds block_timeout)
      : state_(std::move(state)), block_timeout_(block_timeout) {}

  void SetStopRequestedFlag(const std::atomic<bool> *stop_requested) override {
    external_stop_requested_ = stop_requested;
  }

  bool Open(const AudioPipelineConfig &, std::string *error) override {
    if (error)
      error->clear();
    return true;
  }

  bool Read(void *dst, std::size_t bytes, std::string *error) override {
    std::memset(dst, 0, bytes);
    if (error)
      error->clear();
    return true;
  }

  bool Write(const void *, std::size_t, std::string *error) override {
    if (error)
      error->clear();
    return true;
  }

  bool GetCaptureLatencyUs(std::uint64_t *) override { return false; }
  bool GetPlaybackLatencyUs(std::uint64_t *) override { return false; }

  void Flush() override {
    std::unique_lock<std::mutex> lock(state_->mu);
    state_->flush_entered = true;
    state_->cv.notify_all();
    state_->cv.wait_for(lock, block_timeout_, [&] {
      return state_->stop_requested ||
             (external_stop_requested_ &&
              external_stop_requested_->load(std::memory_order_acquire));
    });
  }

  void RequestStop() override {
    std::lock_guard<std::mutex> lock(state_->mu);
    state_->stop_requested = true;
    ++state_->request_stop_calls;
    state_->cv.notify_all();
  }

private:
  std::shared_ptr<BlockingFlushIoState> state_;
  const std::atomic<bool> *external_stop_requested_ = nullptr;
  std::chrono::milliseconds block_timeout_;
};

class ResettingOpenIo final : public AudioPipelineIo {
public:
  ResettingOpenIo(std::shared_ptr<ResettingOpenIoState> state,
                  std::chrono::milliseconds reset_delay,
                  std::chrono::milliseconds block_timeout)
      : state_(std::move(state)), reset_delay_(reset_delay),
        block_timeout_(block_timeout) {}

  void SetStopRequestedFlag(const std::atomic<bool> *stop_requested) override {
    external_stop_requested_ = stop_requested;
  }

  bool Open(const AudioPipelineConfig &, std::string *error) override {
    {
      std::lock_guard<std::mutex> lock(state_->mu);
      state_->open_started = true;
      state_->cv.notify_all();
    }

    std::this_thread::sleep_for(reset_delay_);

    std::unique_lock<std::mutex> lock(state_->mu);
    state_->stop_requested = false;
    const bool stopped = state_->cv.wait_for(lock, block_timeout_, [&] {
      return state_->stop_requested ||
             (external_stop_requested_ &&
              external_stop_requested_->load(std::memory_order_acquire));
    });
    if (stopped) {
      if (error)
        error->clear();
      return false;
    }

    if (error)
      *error = "open remained blocked after stop";
    return false;
  }

  bool Read(void *, std::size_t, std::string *) override { return false; }
  bool Write(const void *, std::size_t, std::string *) override {
    return false;
  }
  bool GetCaptureLatencyUs(std::uint64_t *) override { return false; }
  bool GetPlaybackLatencyUs(std::uint64_t *) override { return false; }
  void Flush() override {}

  void RequestStop() override {
    std::lock_guard<std::mutex> lock(state_->mu);
    state_->stop_requested = true;
    ++state_->request_stop_calls;
    state_->cv.notify_all();
  }

private:
  std::shared_ptr<ResettingOpenIoState> state_;
  const std::atomic<bool> *external_stop_requested_ = nullptr;
  std::chrono::milliseconds reset_delay_;
  std::chrono::milliseconds block_timeout_;
};

class OpenFailIo final : public AudioPipelineIo {
public:
  explicit OpenFailIo(std::atomic<int> *open_calls) : open_calls_(open_calls) {}

  bool Open(const AudioPipelineConfig &, std::string *error) override {
    if (open_calls_)
      open_calls_->fetch_add(1, std::memory_order_relaxed);
    if (error)
      *error = "synthetic open failure";
    return false;
  }

  bool Read(void *, std::size_t, std::string *) override { return false; }
  bool Write(const void *, std::size_t, std::string *) override {
    return false;
  }
  bool GetCaptureLatencyUs(std::uint64_t *) override { return false; }
  bool GetPlaybackLatencyUs(std::uint64_t *) override { return false; }
  void Flush() override {}
  void RequestStop() override {}

private:
  std::atomic<int> *open_calls_ = nullptr;
};

class ReadFailIo final : public AudioPipelineIo {
public:
  explicit ReadFailIo(std::atomic<int> *read_calls) : read_calls_(read_calls) {}

  bool Open(const AudioPipelineConfig &, std::string *error) override {
    if (error)
      error->clear();
    return true;
  }

  bool Read(void *, std::size_t, std::string *error) override {
    if (read_calls_)
      read_calls_->fetch_add(1, std::memory_order_relaxed);
    if (error)
      *error = "synthetic Pulse capture stream disconnected";
    return false;
  }

  bool Write(const void *, std::size_t, std::string *) override {
    return false;
  }
  bool GetCaptureLatencyUs(std::uint64_t *) override { return false; }
  bool GetPlaybackLatencyUs(std::uint64_t *) override { return false; }
  void Flush() override {}
  void RequestStop() override {}

private:
  std::atomic<int> *read_calls_ = nullptr;
};

struct LatencyIoState {
  std::atomic<bool> stop_requested{false};
  std::atomic<int> flush_calls{0};
  std::atomic<int> capture_latency_queries{0};
  std::atomic<int> playback_latency_queries{0};
};

class HighLatencyIo final : public AudioPipelineIo {
public:
  explicit HighLatencyIo(std::shared_ptr<LatencyIoState> state)
      : state_(std::move(state)) {}

  void SetStopRequestedFlag(const std::atomic<bool> *stop_requested) override {
    external_stop_requested_ = stop_requested;
  }

  bool Open(const AudioPipelineConfig &, std::string *error) override {
    if (error)
      error->clear();
    return true;
  }

  bool Read(void *dst, std::size_t bytes, std::string *error) override {
    if (ShouldStop()) {
      if (error)
        error->clear();
      return false;
    }
    std::memset(dst, 0, bytes);
    std::this_thread::sleep_for(1ms);
    if (error)
      error->clear();
    return true;
  }

  bool Write(const void *, std::size_t, std::string *error) override {
    if (ShouldStop()) {
      if (error)
        error->clear();
      return false;
    }
    if (error)
      error->clear();
    return true;
  }

  bool GetCaptureLatencyUs(std::uint64_t *latency_us) override {
    state_->capture_latency_queries.fetch_add(1, std::memory_order_relaxed);
    if (latency_us)
      *latency_us = 80000;
    return true;
  }

  bool GetPlaybackLatencyUs(std::uint64_t *latency_us) override {
    state_->playback_latency_queries.fetch_add(1, std::memory_order_relaxed);
    if (latency_us)
      *latency_us = 80000;
    return true;
  }

  void Flush() override {
    state_->flush_calls.fetch_add(1, std::memory_order_relaxed);
  }

  void RequestStop() override {
    state_->stop_requested.store(true, std::memory_order_release);
  }

private:
  bool ShouldStop() const {
    return state_->stop_requested.load(std::memory_order_acquire) ||
           (external_stop_requested_ &&
            external_stop_requested_->load(std::memory_order_acquire));
  }

  std::shared_ptr<LatencyIoState> state_;
  const std::atomic<bool> *external_stop_requested_ = nullptr;
};

struct FlakyLatencyIoState {
  std::atomic<bool> stop_requested{false};
  std::atomic<int> capture_latency_queries{0};
  std::atomic<int> playback_latency_queries{0};
};

class FlakyLatencyIo final : public AudioPipelineIo {
public:
  explicit FlakyLatencyIo(std::shared_ptr<FlakyLatencyIoState> state)
      : state_(std::move(state)) {}

  void SetStopRequestedFlag(const std::atomic<bool> *stop_requested) override {
    external_stop_requested_ = stop_requested;
  }

  bool Open(const AudioPipelineConfig &, std::string *error) override {
    if (error)
      error->clear();
    return true;
  }

  bool Read(void *dst, std::size_t bytes, std::string *error) override {
    if (ShouldStop()) {
      if (error)
        error->clear();
      return false;
    }
    std::memset(dst, 0, bytes);
    std::this_thread::sleep_for(1ms);
    if (error)
      error->clear();
    return true;
  }

  bool Write(const void *, std::size_t, std::string *error) override {
    if (ShouldStop()) {
      if (error)
        error->clear();
      return false;
    }
    if (error)
      error->clear();
    return true;
  }

  bool GetCaptureLatencyUs(std::uint64_t *latency_us) override {
    const int q = state_->capture_latency_queries.fetch_add(
                      1, std::memory_order_relaxed) +
                  1;
    if (q == 1) {
      if (latency_us)
        *latency_us = 12000;
      return true;
    }
    return false;
  }

  bool GetPlaybackLatencyUs(std::uint64_t *latency_us) override {
    const int q = state_->playback_latency_queries.fetch_add(
                      1, std::memory_order_relaxed) +
                  1;
    if (latency_us)
      *latency_us = (q == 1) ? 34000 : 5000;
    return true;
  }

  void Flush() override {}

  void RequestStop() override {
    state_->stop_requested.store(true, std::memory_order_release);
  }

private:
  bool ShouldStop() const {
    return state_->stop_requested.load(std::memory_order_acquire) ||
           (external_stop_requested_ &&
            external_stop_requested_->load(std::memory_order_acquire));
  }

  std::shared_ptr<FlakyLatencyIoState> state_;
  const std::atomic<bool> *external_stop_requested_ = nullptr;
};

struct ScriptedQualityIoState {
  std::mutex mu;
  std::vector<float> input;
  std::vector<float> output;
  std::size_t read_offset_samples = 0;
  std::size_t bytes_per_frame = 0;
  std::uint32_t open_sample_rate = 0;
  std::uint32_t open_channels = 0;
  int read_calls = 0;
  int write_calls = 0;
  int flush_calls = 0;
  bool stop_requested = false;
};

class ScriptedQualityIo final : public AudioPipelineIo {
public:
  explicit ScriptedQualityIo(std::shared_ptr<ScriptedQualityIoState> state)
      : state_(std::move(state)) {}

  void SetStopRequestedFlag(const std::atomic<bool> *stop_requested) override {
    external_stop_requested_ = stop_requested;
  }

  bool Open(const AudioPipelineConfig &cfg, std::string *error) override {
    std::lock_guard<std::mutex> lock(state_->mu);
    state_->open_sample_rate = static_cast<std::uint32_t>(cfg.sample_rate);
    state_->open_channels = cfg.channels;
    state_->bytes_per_frame = static_cast<std::size_t>(cfg.frame_samples) *
                              cfg.channels * sizeof(float);
    state_->read_offset_samples = 0;
    state_->output.clear();
    state_->read_calls = 0;
    state_->write_calls = 0;
    state_->flush_calls = 0;
    state_->stop_requested = false;

    const std::size_t samples_per_frame =
        static_cast<std::size_t>(cfg.frame_samples) * cfg.channels;
    if (samples_per_frame == 0 ||
        (state_->input.size() % samples_per_frame) != 0) {
      if (error)
        *error = "scripted audio input is not frame aligned";
      return false;
    }
    if (error)
      error->clear();
    return true;
  }

  bool Read(void *dst, std::size_t bytes, std::string *error) override {
    std::lock_guard<std::mutex> lock(state_->mu);
    if (ShouldStopLocked()) {
      if (error)
        error->clear();
      return false;
    }
    if (bytes != state_->bytes_per_frame) {
      if (error)
        *error = "unexpected scripted read size";
      return false;
    }
    if (state_->read_offset_samples >= state_->input.size()) {
      if (error)
        error->clear();
      return false;
    }

    const std::size_t samples = bytes / sizeof(float);
    if (state_->read_offset_samples + samples > state_->input.size()) {
      if (error)
        *error = "scripted read exceeded input";
      return false;
    }
    std::memcpy(dst, state_->input.data() + state_->read_offset_samples, bytes);
    state_->read_offset_samples += samples;
    ++state_->read_calls;
    if (error)
      error->clear();
    return true;
  }

  bool Write(const void *src, std::size_t bytes, std::string *error) override {
    std::lock_guard<std::mutex> lock(state_->mu);
    if (ShouldStopLocked()) {
      if (error)
        error->clear();
      return false;
    }
    if (bytes != state_->bytes_per_frame) {
      if (error)
        *error = "unexpected scripted write size";
      return false;
    }
    const auto *samples = static_cast<const float *>(src);
    state_->output.insert(state_->output.end(), samples,
                          samples + (bytes / sizeof(float)));
    ++state_->write_calls;
    if (error)
      error->clear();
    return true;
  }

  bool GetCaptureLatencyUs(std::uint64_t *) override { return false; }
  bool GetPlaybackLatencyUs(std::uint64_t *) override { return false; }

  void Flush() override {
    std::lock_guard<std::mutex> lock(state_->mu);
    ++state_->flush_calls;
  }

  void RequestStop() override {
    std::lock_guard<std::mutex> lock(state_->mu);
    state_->stop_requested = true;
  }

private:
  bool ShouldStopLocked() const {
    return state_->stop_requested ||
           (external_stop_requested_ &&
            external_stop_requested_->load(std::memory_order_acquire));
  }

  std::shared_ptr<ScriptedQualityIoState> state_;
  const std::atomic<bool> *external_stop_requested_ = nullptr;
};

std::vector<float> MakeSyntheticStereoAudio(std::uint32_t frame_samples,
                                            int frame_count) {
  constexpr double kPi = 3.141592653589793238462643383279502884;
  constexpr double kSampleRate = 48000.0;
  const std::size_t total_frames = static_cast<std::size_t>(frame_samples) *
                                   static_cast<std::size_t>(frame_count);
  std::vector<float> audio(total_frames * 2, 0.0f);

  for (std::size_t n = 0; n < total_frames; ++n) {
    const std::size_t packet = n / frame_samples;
    float left = 0.0f;
    float right = 0.0f;
    const double t = static_cast<double>(n) / kSampleRate;

    if (packet == 1 || packet == 2) {
      left = static_cast<float>(0.35 * std::sin(2.0 * kPi * 1000.0 * t));
      right = static_cast<float>(0.25 * std::sin(2.0 * kPi * 440.0 * t));
    } else if (packet == 3) {
      if ((n % frame_samples) == 0) {
        left = 0.75f;
        right = -0.75f;
      }
    } else if (packet > 3) {
      const double sweep01 =
          static_cast<double>(n) /
          static_cast<double>(std::max<std::size_t>(total_frames - 1, 1));
      const double hz = 220.0 + 1780.0 * sweep01;
      left = static_cast<float>(0.2 * std::sin(2.0 * kPi * hz * t));
      right = static_cast<float>(0.2 * std::sin(2.0 * kPi * hz * t + 0.3));
    }

    audio[n * 2 + 0] = left;
    audio[n * 2 + 1] = right;
  }
  return audio;
}

double Rms(const std::vector<float> &samples) {
  if (samples.empty())
    return 0.0;
  double sum = 0.0;
  for (float sample : samples)
    sum += static_cast<double>(sample) * static_cast<double>(sample);
  return std::sqrt(sum / static_cast<double>(samples.size()));
}

bool TestPactlLoadModuleQuotesVectorArguments() {
  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    studiocast::util::ExecResult result;
    result.exit_code = 0;
    result.stdout_str = "123\n";
    return result;
  });

  std::string err;
  const auto id = studiocast::audio::pulse::LoadModule(
      "module-null-sink",
      {
          "sink_name=studio cast",
          "sink_properties=device.description=\"StudioCast Speakers\"",
          "weird=a'b",
      },
      &err);

  const std::string expected =
      "pactl load-module 'module-null-sink' 'sink_name=studio cast' "
      "'sink_properties=device.description=\"StudioCast Speakers\"' "
      "'weird=a'\"'\"'b' 2>&1";

  if (!id || *id != 123) {
    std::cerr << "expected pactl module id 123; error='" << err << "'\n";
    return false;
  }

  if (commands.size() != 1 || commands[0] != expected) {
    std::cerr << "unexpected pactl load-module command\nexpected: " << expected
              << "\nactual:   "
              << (commands.empty() ? std::string("<none>") : commands[0])
              << "\n";
    return false;
  }

  return true;
}

bool TestPactlLoadModuleStringCompatibilitySplitter() {
  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    studiocast::util::ExecResult result;
    result.exit_code = 0;
    result.stdout_str = "124\n";
    return result;
  });

  std::string err;
  const auto id = studiocast::audio::pulse::LoadModule(
      "module-loopback",
      "source='studio cast.monitor' sink=\"physical sink\" latency_msec=10 "
      "escaped=hello\\ world",
      &err);

  const std::string expected =
      "pactl load-module 'module-loopback' 'source=studio cast.monitor' "
      "'sink=physical sink' 'latency_msec=10' 'escaped=hello world' 2>&1";

  if (!id || *id != 124) {
    std::cerr << "expected pactl module id 124; error='" << err << "'\n";
    return false;
  }

  if (commands.size() != 1 || commands[0] != expected) {
    std::cerr << "unexpected pactl compatibility command\nexpected: "
              << expected << "\nactual:   "
              << (commands.empty() ? std::string("<none>") : commands[0])
              << "\n";
    return false;
  }

  return true;
}

bool TestPactlDefaultSourceAndSinkFallbackToInfo() {
  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl get-default-source 2>&1")
      return ExecResult(1, "unknown command\n");
    if (command == "pactl get-default-sink 2>&1")
      return ExecResult(1, "unknown command\n");
    if (command == "pactl info 2>&1") {
      return ExecResult(0, "Server String: /run/user/1000/pulse/native\n"
                           "Default Source: alsa_input.usb_test_mic\n"
                           "Default Sink: alsa_output.pci_test_speakers\n");
    }
    return ExecResult(99, "unexpected command: " + command);
  });

  std::string err;
  const auto source = studiocast::audio::pulse::GetDefaultSourceName(&err);
  const auto sink = studiocast::audio::pulse::GetDefaultSinkName(&err);

  const std::vector<std::string> expected = {
      "pactl get-default-source 2>&1",
      "pactl info 2>&1",
      "pactl get-default-sink 2>&1",
      "pactl info 2>&1",
  };

  if (!source || *source != "alsa_input.usb_test_mic" || !sink ||
      *sink != "alsa_output.pci_test_speakers") {
    std::cerr << "default source/sink fallback returned source='"
              << (source ? *source : std::string("<none>")) << "' sink='"
              << (sink ? *sink : std::string("<none>")) << "' error='" << err
              << "'\n";
    return false;
  }

  if (commands != expected) {
    std::cerr << "unexpected default source/sink command sequence\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  return true;
}

bool TestPactlProplistCommandsQuoteArgumentsAndDetectFailures() {
  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (commands.size() == 1)
      return ExecResult(0);
    return ExecResult(0, "Failure: No such entity\n");
  });

  std::string err;
  const bool sink_ok = studiocast::audio::pulse::UpdateSinkProplist(
      "sink's name",
      {
          "device.description=StudioCast Speakers",
          "node.description=StudioCast Speakers",
      },
      &err);
  if (!sink_ok) {
    std::cerr << "expected sink proplist update to succeed; error='" << err
              << "'\n";
    return false;
  }

  err.clear();
  const bool source_ok = studiocast::audio::pulse::UpdateSourceProplist(
      "source name",
      {
          "device.description=StudioCast Microphone",
          "node.description=StudioCast Microphone",
      },
      &err);
  if (source_ok || err.find("Failure: No such entity") == std::string::npos) {
    std::cerr << "expected source proplist failure to be detected; ok="
              << source_ok << " error='" << err << "'\n";
    return false;
  }

  const std::vector<std::string> expected = {
      "pactl update-sink-proplist 'sink'\"'\"'s name' "
      "'device.description=StudioCast Speakers' "
      "'node.description=StudioCast Speakers' 2>&1",
      "pactl update-source-proplist 'source name' "
      "'device.description=StudioCast Microphone' "
      "'node.description=StudioCast Microphone' 2>&1",
  };

  if (commands != expected) {
    std::cerr << "unexpected proplist command sequence\nexpected:\n";
    for (const auto &command : expected)
      std::cerr << "  " << command << "\n";
    std::cerr << "actual:\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  return true;
}

bool TestPactlConsumerListsParseSourceOutputsAndSinkInputs() {
  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl list short source-outputs 2>&1") {
      return ExecResult(0,
                        "7\t3\t51\tprotocol-native.c\tfloat32le 1ch 48000Hz\n"
                        "8\tstudiocast_mic\t52\tprotocol-native.c\t"
                        "float32le 1ch 48000Hz\n");
    }
    if (command == "pactl list short sink-inputs 2>&1") {
      return ExecResult(0,
                        "9\t4\t61\tprotocol-native.c\tfloat32le 2ch 48000Hz\n"
                        "10\tstudiocast_speakers\t62\tprotocol-native.c\t"
                        "float32le 2ch 48000Hz\n");
    }
    return ExecResult(99, "unexpected command: " + command);
  });

  std::string err;
  const auto outputs = studiocast::audio::pulse::ListSourceOutputs(&err);
  if (!err.empty() || outputs.size() != 2 || outputs[0].id != 7 ||
      outputs[0].source != "3" || outputs[1].source != "studiocast_mic") {
    std::cerr << "source-output parsing failed; count=" << outputs.size()
              << " error='" << err << "'\n";
    return false;
  }

  err.clear();
  const auto inputs = studiocast::audio::pulse::ListSinkInputs(&err);
  if (!err.empty() || inputs.size() != 2 || inputs[0].id != 9 ||
      inputs[0].sink != "4" || inputs[1].sink != "studiocast_speakers") {
    std::cerr << "sink-input parsing failed; count=" << inputs.size()
              << " error='" << err << "'\n";
    return false;
  }

  const std::vector<std::string> expected = {
      "pactl list short source-outputs 2>&1",
      "pactl list short sink-inputs 2>&1",
  };
  if (commands != expected) {
    std::cerr << "unexpected consumer-list command sequence\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  return true;
}

bool TestAudioSourceSafetyRejectsVirtualAndMonitorSources() {
  std::string reason;
  if (!studiocast::audio::IsUnsafeInputSourceName("studiocast_mic", &reason) ||
      reason.find("StudioCast virtual source") == std::string::npos) {
    std::cerr << "StudioCast virtual mic source was not rejected; reason='"
              << reason << "'\n";
    return false;
  }

  reason.clear();
  if (!studiocast::audio::IsUnsafeInputSourceName(
          "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor", &reason) ||
      reason.find("monitor source") == std::string::npos) {
    std::cerr << "monitor source was not rejected; reason='" << reason << "'\n";
    return false;
  }

  const auto explicit_virtual =
      studiocast::audio::ResolveSafeInputSourceName("studiocast_mic");
  if (explicit_virtual.ok ||
      explicit_virtual.error.find("StudioCast virtual source") ==
          std::string::npos) {
    std::cerr << "explicit virtual source was not rejected; ok="
              << explicit_virtual.ok << " error='" << explicit_virtual.error
              << "'\n";
    return false;
  }

  const auto explicit_physical =
      studiocast::audio::ResolveSafeInputSourceName("alsa_input.usb_test_mic");
  if (!explicit_physical.ok ||
      explicit_physical.source_name != "alsa_input.usb_test_mic") {
    std::cerr << "explicit physical source was not accepted; ok="
              << explicit_physical.ok << " source='"
              << explicit_physical.source_name << "' error='"
              << explicit_physical.error << "'\n";
    return false;
  }

  return true;
}

bool TestAudioSourceAutoFallsBackFromUnsafeDefaultSource() {
  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl get-default-source 2>&1")
      return ExecResult(0, "studiocast_speakers.monitor\n");
    if (command == "pactl list short sources 2>&1") {
      return ExecResult(0,
                        "0\tstudiocast_speakers.monitor\tmodule-null-sink.c\t"
                        "s16le 2ch 48000Hz\tIDLE\n"
                        "1\talsa_output.pci_test.monitor\tmodule-alsa-card.c\t"
                        "s16le 2ch 48000Hz\tIDLE\n"
                        "2\talsa_input.usb_test_mic\tmodule-alsa-card.c\t"
                        "s16le 1ch 48000Hz\tRUNNING\n");
    }
    return ExecResult(99, "unexpected command: " + command);
  });

  const auto resolved = studiocast::audio::ResolveSafeInputSourceName("");
  if (!resolved.ok || resolved.source_name != "alsa_input.usb_test_mic" ||
      resolved.warnings.empty()) {
    std::cerr << "auto source did not fall back to physical mic; ok="
              << resolved.ok << " source='" << resolved.source_name
              << "' error='" << resolved.error
              << "' warnings=" << resolved.warnings.size() << "\n";
    return false;
  }

  const std::vector<std::string> expected = {
      "pactl get-default-source 2>&1",
      "pactl list short sources 2>&1",
  };
  if (commands != expected) {
    std::cerr << "unexpected source auto fallback command sequence\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  return true;
}

bool TestAudioSourceAutoFailsWhenNoSafeSourceExists() {
  ScopedPactlExecHook hook([](const std::string &command) {
    if (command == "pactl get-default-source 2>&1")
      return ExecResult(0, "studiocast_mic\n");
    if (command == "pactl list short sources 2>&1") {
      return ExecResult(0,
                        "0\tstudiocast_mic\tmodule-remap-source.c\t"
                        "s16le 1ch 48000Hz\tIDLE\n"
                        "1\talsa_output.pci_test.monitor\tmodule-alsa-card.c\t"
                        "s16le 2ch 48000Hz\tIDLE\n");
    }
    return ExecResult(99, "unexpected command: " + command);
  });

  const auto resolved = studiocast::audio::ResolveSafeInputSourceName("auto");
  if (resolved.ok ||
      resolved.error.find("No safe Pulse microphone source") ==
          std::string::npos ||
      resolved.error.find("Select a physical microphone") ==
          std::string::npos) {
    std::cerr << "auto source without safe fallback did not fail clearly; ok="
              << resolved.ok << " source='" << resolved.source_name
              << "' error='" << resolved.error << "'\n";
    return false;
  }

  return true;
}

bool TestVirtualAudioServiceReportsResolvedAutoSourceAndWarnings() {
  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl get-default-source 2>&1")
      return ExecResult(0, "studiocast_speakers.monitor\n");
    if (command == "pactl list short sources 2>&1") {
      return ExecResult(0,
                        "0\tstudiocast_speakers.monitor\tmodule-null-sink.c\t"
                        "s16le 2ch 48000Hz\tIDLE\n"
                        "1\talsa_input.usb_service_mic\tmodule-alsa-card.c\t"
                        "s16le 1ch 48000Hz\tRUNNING\n");
    }
    return ExecResult(99, "unexpected command: " + command);
  });

  std::atomic<bool> mic_consumer_present{true};
  std::atomic<int> pipeline_creates{0};
  std::string pipeline_source;

  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    ++pipeline_creates;
    class CapturingPipeline final : public AudioPipelineRunner {
    public:
      explicit CapturingPipeline(std::string *source) : source_(source) {}

      bool Start(const AudioPipelineConfig &cfg, std::string *error) override {
        if (source_)
          *source_ = cfg.source_name;
        if (error)
          error->clear();
        return true;
      }

      void Stop() override {}

      AudioPipelineStats GetStats() const override {
        AudioPipelineStats stats;
        stats.running = true;
        return stats;
      }

    private:
      std::string *source_ = nullptr;
    };
    return std::make_unique<CapturingPipeline>(&pipeline_source);
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.source_name.clear(); // auto
  cfg.poll_ms = 1;
  cfg.consumer_grace_ms = 0;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool resolved = WaitUntil(
      [&] {
        const auto status = service.Status();
        return pipeline_creates.load(std::memory_order_relaxed) == 1 &&
               pipeline_source == "alsa_input.usb_service_mic" &&
               status.selected_source == "alsa_input.usb_service_mic" &&
               status.source_error.empty() && !status.source_warnings.empty();
      },
      250ms);
  const auto status = service.Status();
  service.Stop();

  if (!resolved) {
    std::cerr << "service did not report/pass resolved auto source; "
              << "creates=" << pipeline_creates.load() << " pipeline_source='"
              << pipeline_source << "' selected_source='"
              << status.selected_source << "' source_error='"
              << status.source_error
              << "' warnings=" << status.source_warnings.size() << "\n";
    return false;
  }

  const std::vector<std::string> expected = {
      "pactl get-default-source 2>&1",
      "pactl list short sources 2>&1",
  };
  if (commands.size() < expected.size() ||
      !std::equal(expected.begin(), expected.end(), commands.begin())) {
    std::cerr << "unexpected source resolution command prefix\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  return true;
}

bool TestVirtualAudioServicePreservesUnavailableConfiguredSource() {
  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl list short sources 2>&1") {
      return ExecResult(0, "0\talsa_input.other_mic\tmodule-alsa-card.c\t"
                           "s16le 1ch 48000Hz\tRUNNING\n");
    }
    return ExecResult(99, "unexpected command: " + command);
  });

  VirtualAudioServiceHooks hooks;
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = false;
  cfg.create_virtual_mic = false;
  cfg.source_name = "alsa_input.disconnected_mic";
  cfg.poll_ms = 1;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool reported = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.selected_source == "alsa_input.disconnected_mic" &&
               status.source_availability == "unavailable" &&
               status.source_error.find("not currently available") !=
                   std::string::npos;
      },
      250ms);
  const auto status = service.Status();
  const auto preservedConfig = service.Config();
  service.Stop();

  if (!reported) {
    std::cerr << "unavailable configured source was not reported; selected='"
              << status.selected_source << "' availability='"
              << status.source_availability << "' error='"
              << status.source_error << "'\n";
    return false;
  }
  if (preservedConfig.source_name != "alsa_input.disconnected_mic") {
    std::cerr << "configured source was mutated; got '"
              << preservedConfig.source_name << "'\n";
    return false;
  }

  const std::vector<std::string> expected = {
      "pactl list short sources 2>&1",
  };
  if (commands.size() < expected.size() ||
      !std::equal(expected.begin(), expected.end(), commands.begin())) {
    std::cerr << "unexpected unavailable-source command prefix\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  return true;
}

bool TestSpeakerTargetSafetyRejectsVirtualAndMonitorEndpoints() {
  std::string reason;
  if (!studiocast::audio::IsUnsafeSpeakerTargetSinkName("studiocast_sink",
                                                        &reason) ||
      reason.find("feedback loop") == std::string::npos) {
    std::cerr << "virtual speaker target was not rejected; reason='" << reason
              << "'\n";
    return false;
  }

  reason.clear();
  if (!studiocast::audio::IsUnsafeSpeakerTargetSinkName(
          "alsa_output.pci_test.monitor", &reason) ||
      reason.find("monitor source") == std::string::npos) {
    std::cerr << "monitor endpoint target was not rejected; reason='" << reason
              << "'\n";
    return false;
  }

  std::string err;
  const auto chosen = studiocast::audio::ChooseSafeSpeakerTargetSinkName(
      "studiocast_sink", &err);
  if (chosen || err.find("feedback loop") == std::string::npos) {
    std::cerr << "virtual target chooser did not reject sink; chosen='"
              << (chosen ? *chosen : std::string("<none>")) << "' error='"
              << err << "'\n";
    return false;
  }

  return true;
}

bool TestSpeakerTargetAutoFallsBackFromUnsafeDefaultSink() {
  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl get-default-sink 2>&1")
      return ExecResult(0, "studiocast_speakers\n");
    if (command == "pactl list short sinks 2>&1") {
      return ExecResult(0, "0\tstudiocast_speakers\tmodule-null-sink.c\t"
                           "s16le 2ch 48000Hz\tIDLE\n"
                           "1\tstudiocast_sink\tmodule-null-sink.c\t"
                           "s16le 2ch 48000Hz\tIDLE\n"
                           "2\tphysical_test_sink\tmodule-alsa-card.c\t"
                           "s16le 2ch 48000Hz\tRUNNING\n");
    }
    return ExecResult(99, "unexpected command: " + command);
  });

  std::string err;
  const auto chosen = studiocast::audio::ChooseSafeSpeakerTargetSinkName(
      /*configured_target=*/"", &err);
  if (!chosen || *chosen != "physical_test_sink") {
    std::cerr << "speaker auto target did not fall back to physical sink; "
              << "chosen='" << (chosen ? *chosen : std::string("<none>"))
              << "' error='" << err << "'\n";
    return false;
  }

  const std::vector<std::string> expected = {
      "pactl get-default-sink 2>&1",
      "pactl list short sinks 2>&1",
  };
  if (commands != expected) {
    std::cerr << "unexpected speaker target command sequence\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  return true;
}

bool TestStatusTextSurfacesModuleListFailure() {
  ScopedXdgStateHome state_home;
  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl --version 2>&1")
      return ExecResult(0, "pactl 16.1\n");
    if (command == "pactl list short modules 2>&1")
      return ExecResult(1, "Failure: synthetic module list failure\n");
    if (command == "pactl list short sources 2>&1")
      return ExecResult(0);
    if (command == "pactl get-default-sink 2>&1")
      return ExecResult(0, "physical_test_sink\n");
    if (command == "pactl list short sinks 2>&1")
      return ExecResult(0, "2\tphysical_test_sink\tmodule-alsa-card.c\t"
                           "s16le 2ch 48000Hz\tRUNNING\n");
    return ExecResult(99, "unexpected command: " + command);
  });

  const std::string text = studiocast::audio::StatusText();

  const std::string unavailable = "loaded ids: unavailable";
  std::size_t unavailable_count = 0;
  std::size_t pos = 0;
  while ((pos = text.find(unavailable, pos)) != std::string::npos) {
    ++unavailable_count;
    pos += unavailable.size();
  }

  if (unavailable_count < 2 ||
      text.find("synthetic module list failure") == std::string::npos) {
    std::cerr << "status text did not surface module list failures:\n"
              << text << "\n";
    return false;
  }

  if (text.find("loaded ids: sink=none, remap=none, loopback=none") !=
          std::string::npos ||
      text.find("loaded ids: sink=none, loopback=none") != std::string::npos) {
    std::cerr << "status text rendered unknown loaded ids as none:\n"
              << text << "\n";
    return false;
  }

  return true;
}

bool TestCreateVirtualMicPropagatesListFailureWithoutLoading() {
  ScopedXdgStateHome state_home;
  VirtualMicState initial;
  initial.null_sink_module_id = 10;
  initial.remap_source_module_id = 11;
  std::string err;
  if (!studiocast::audio::SaveVirtualMicState(initial, &err)) {
    std::cerr << "failed to seed virtual mic state: " << err << "\n";
    return false;
  }

  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl --version 2>&1")
      return ExecResult(0, "pactl 16.1\n");
    if (command == "pactl list short modules 2>&1")
      return ExecResult(1, "Failure: synthetic mic list failure\n");
    return ExecResult(99, "unexpected command: " + command);
  });

  err.clear();
  const bool ok = studiocast::audio::CreateVirtualMic(&err);
  if (ok || err.find("synthetic mic list failure") == std::string::npos) {
    std::cerr << "expected virtual mic create to fail on list failure; ok="
              << ok << " error='" << err << "'\n";
    return false;
  }

  if (CommandWasRun(commands, "pactl load-module")) {
    std::cerr << "virtual mic create loaded modules after list failure\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  const auto state = studiocast::audio::LoadVirtualMicState();
  if (!state.null_sink_module_id || *state.null_sink_module_id != 10 ||
      !state.remap_source_module_id || *state.remap_source_module_id != 11) {
    std::cerr << "virtual mic create changed state after list failure; sink="
              << (state.null_sink_module_id
                      ? std::to_string(*state.null_sink_module_id)
                      : std::string("<none>"))
              << " remap="
              << (state.remap_source_module_id
                      ? std::to_string(*state.remap_source_module_id)
                      : std::string("<none>"))
              << "\n";
    return false;
  }

  return true;
}

bool TestCreateVirtualSpeakerPropagatesListFailureWithoutLoading() {
  ScopedXdgStateHome state_home;
  VirtualSpeakerState initial;
  initial.null_sink_module_id = 10;
  initial.loopback_module_id = 42;
  initial.loopback_target_sink_name = "physical_test_sink";
  std::string err;
  if (!studiocast::audio::SaveVirtualSpeakerState(initial, &err)) {
    std::cerr << "failed to seed virtual speaker state: " << err << "\n";
    return false;
  }

  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl --version 2>&1")
      return ExecResult(0, "pactl 16.1\n");
    if (command == "pactl list short modules 2>&1")
      return ExecResult(1, "Failure: synthetic speaker list failure\n");
    return ExecResult(99, "unexpected command: " + command);
  });

  err.clear();
  const bool ok = studiocast::audio::CreateVirtualSpeaker(&err);
  if (ok || err.find("synthetic speaker list failure") == std::string::npos) {
    std::cerr << "expected virtual speaker create to fail on list failure; ok="
              << ok << " error='" << err << "'\n";
    return false;
  }

  if (CommandWasRun(commands, "pactl load-module")) {
    std::cerr << "virtual speaker create loaded modules after list failure\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  const auto state = studiocast::audio::LoadVirtualSpeakerState();
  if (!state.null_sink_module_id || *state.null_sink_module_id != 10 ||
      !state.loopback_module_id || *state.loopback_module_id != 42 ||
      !state.loopback_target_sink_name ||
      *state.loopback_target_sink_name != "physical_test_sink") {
    std::cerr
        << "virtual speaker create changed state after list failure; sink="
        << (state.null_sink_module_id
                ? std::to_string(*state.null_sink_module_id)
                : std::string("<none>"))
        << " loopback="
        << (state.loopback_module_id ? std::to_string(*state.loopback_module_id)
                                     : std::string("<none>"))
        << " target='"
        << (state.loopback_target_sink_name ? *state.loopback_target_sink_name
                                            : std::string("<none>"))
        << "'\n";
    return false;
  }

  return true;
}

bool TestVirtualSpeakerLoopbackFallsBackFromVirtualDefaultSink() {
  ScopedXdgStateHome state_home;
  std::vector<std::string> commands;
  const std::string expected_loopback =
      "pactl load-module 'module-loopback' "
      "'source=studiocast_speakers.monitor' 'sink=physical_test_sink' "
      "'latency_msec=12' 2>&1";

  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl --version 2>&1")
      return ExecResult(0, "pactl 16.1\n");
    if (command == "pactl list short modules 2>&1")
      return ExecResult(0);
    if (command.find("pactl load-module 'module-null-sink'") == 0)
      return ExecResult(0, "10\n");
    if (command.find("pactl update-sink-proplist 'studiocast_speakers'") == 0)
      return ExecResult(0);
    if (command == "pactl get-default-sink 2>&1")
      return ExecResult(0, "studiocast_speakers\n");
    if (command == "pactl list short sinks 2>&1") {
      return ExecResult(0, "0\tstudiocast_speakers\tmodule-null-sink.c\t"
                           "s16le 2ch 48000Hz\tRUNNING\n"
                           "1\tstudiocast_sink\tmodule-null-sink.c\t"
                           "s16le 2ch 48000Hz\tRUNNING\n"
                           "2\tphysical_test_sink\tmodule-alsa-card.c\t"
                           "s16le 2ch 48000Hz\tRUNNING\n");
    }
    if (command == expected_loopback)
      return ExecResult(0, "20\n");
    return ExecResult(99, "unexpected command: " + command);
  });

  std::string err;
  const bool ok = studiocast::audio::StartSpeakerLoopback("", 12, &err);
  if (!ok) {
    std::cerr << "expected speaker loopback fallback to succeed; error='" << err
              << "'\n";
    return false;
  }

  const auto state = studiocast::audio::LoadVirtualSpeakerState();
  if (!state.loopback_module_id || *state.loopback_module_id != 20 ||
      !state.loopback_target_sink_name ||
      *state.loopback_target_sink_name != "physical_test_sink") {
    std::cerr << "speaker loopback state did not record fallback sink; id="
              << (state.loopback_module_id
                      ? std::to_string(*state.loopback_module_id)
                      : std::string("<none>"))
              << " target='"
              << (state.loopback_target_sink_name
                      ? *state.loopback_target_sink_name
                      : std::string("<none>"))
              << "'\n";
    return false;
  }

  if (!CommandWasRun(commands, expected_loopback) ||
      CommandWasRun(commands, "'sink=studiocast_speakers'")) {
    std::cerr << "speaker loopback did not use the physical fallback sink\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  return true;
}

bool TestVirtualSpeakerLoopbackRejectsVirtualTarget() {
  ScopedXdgStateHome state_home;
  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl --version 2>&1")
      return ExecResult(0, "pactl 16.1\n");
    if (command == "pactl list short modules 2>&1")
      return ExecResult(0);
    if (command.find("pactl load-module 'module-null-sink'") == 0)
      return ExecResult(0, "10\n");
    if (command.find("pactl update-sink-proplist 'studiocast_speakers'") == 0)
      return ExecResult(0);
    return ExecResult(99, "unexpected command: " + command);
  });

  std::string err;
  const bool ok =
      studiocast::audio::StartSpeakerLoopback("studiocast_sink", 10, &err);
  if (ok || err.find("feedback loop") == std::string::npos) {
    std::cerr << "expected virtual speaker target to be rejected; ok=" << ok
              << " error='" << err << "'\n";
    return false;
  }

  if (CommandWasRun(commands, "pactl load-module 'module-loopback'")) {
    std::cerr << "module-loopback was loaded for an invalid virtual target\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  return true;
}

bool TestVirtualSpeakerLoopbackRejectsVirtualTargetBeforeStop() {
  ScopedXdgStateHome state_home;
  VirtualSpeakerState initial;
  initial.null_sink_module_id = 10;
  initial.loopback_module_id = 42;
  initial.loopback_target_sink_name = "physical_test_sink";
  std::string err;
  if (!studiocast::audio::SaveVirtualSpeakerState(initial, &err)) {
    std::cerr << "failed to seed virtual speaker state: " << err << "\n";
    return false;
  }

  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl --version 2>&1")
      return ExecResult(0, "pactl 16.1\n");
    return ExecResult(99, "unexpected command: " + command);
  });

  err.clear();
  const bool ok =
      studiocast::audio::StartSpeakerLoopback("studiocast_sink", 10, &err);
  if (ok || err.find("feedback loop") == std::string::npos) {
    std::cerr
        << "expected virtual speaker target to be rejected before stop; ok="
        << ok << " error='" << err << "'\n";
    return false;
  }

  const auto state = studiocast::audio::LoadVirtualSpeakerState();
  if (!state.loopback_module_id || *state.loopback_module_id != 42 ||
      !state.loopback_target_sink_name ||
      *state.loopback_target_sink_name != "physical_test_sink") {
    std::cerr << "invalid target changed active speaker route state; id="
              << (state.loopback_module_id
                      ? std::to_string(*state.loopback_module_id)
                      : std::string("<none>"))
              << " target='"
              << (state.loopback_target_sink_name
                      ? *state.loopback_target_sink_name
                      : std::string("<none>"))
              << "'\n";
    return false;
  }

  if (CommandWasRun(commands, "pactl unload-module") ||
      CommandWasRun(commands, "pactl load-module")) {
    std::cerr << "invalid target performed destructive pactl operation\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  return true;
}

bool TestVirtualSpeakerLoopbackRestartPropagatesStopFailure() {
  ScopedXdgStateHome state_home;
  VirtualSpeakerState initial;
  initial.null_sink_module_id = 10;
  initial.loopback_module_id = 42;
  initial.loopback_target_sink_name = "old_physical_sink";
  std::string err;
  if (!studiocast::audio::SaveVirtualSpeakerState(initial, &err)) {
    std::cerr << "failed to seed virtual speaker state: " << err << "\n";
    return false;
  }

  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl --version 2>&1")
      return ExecResult(0, "pactl 16.1\n");
    if (command == "pactl list short modules 2>&1") {
      return ExecResult(
          0, "10\tmodule-null-sink\tsink_name=studiocast_speakers\n"
             "42\tmodule-loopback\tsource=studiocast_speakers.monitor "
             "sink=old_physical_sink\n");
    }
    if (command.find("pactl update-sink-proplist 'studiocast_speakers'") == 0)
      return ExecResult(0);
    if (command == "pactl unload-module 42 2>&1")
      return ExecResult(1, "Failure: synthetic unload failure\n");
    return ExecResult(99, "unexpected command: " + command);
  });

  err.clear();
  const bool ok =
      studiocast::audio::StartSpeakerLoopback("new_physical_sink", 10, &err);
  if (ok || err.find("synthetic unload failure") == std::string::npos) {
    std::cerr << "expected loopback restart to fail on unload failure; ok="
              << ok << " error='" << err << "'\n";
    return false;
  }

  const auto state = studiocast::audio::LoadVirtualSpeakerState();
  if (!state.loopback_module_id || *state.loopback_module_id != 42 ||
      !state.loopback_target_sink_name ||
      *state.loopback_target_sink_name != "old_physical_sink") {
    std::cerr << "failed loopback stop did not preserve active state; id="
              << (state.loopback_module_id
                      ? std::to_string(*state.loopback_module_id)
                      : std::string("<none>"))
              << " target='"
              << (state.loopback_target_sink_name
                      ? *state.loopback_target_sink_name
                      : std::string("<none>"))
              << "'\n";
    return false;
  }

  if (CommandWasRun(commands, "pactl load-module 'module-loopback'")) {
    std::cerr << "new module-loopback was loaded after stop failure\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  return true;
}

bool TestDestroyVirtualSpeakerPropagatesNullSinkUnloadFailure() {
  ScopedXdgStateHome state_home;
  VirtualSpeakerState initial;
  initial.null_sink_module_id = 10;
  initial.loopback_module_id = 42;
  initial.loopback_target_sink_name = "old_physical_sink";
  std::string err;
  if (!studiocast::audio::SaveVirtualSpeakerState(initial, &err)) {
    std::cerr << "failed to seed virtual speaker state: " << err << "\n";
    return false;
  }

  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl --version 2>&1")
      return ExecResult(0, "pactl 16.1\n");
    if (command == "pactl list short modules 2>&1") {
      return ExecResult(
          0, "10\tmodule-null-sink\tsink_name=studiocast_speakers\n");
    }
    if (command == "pactl unload-module 10 2>&1")
      return ExecResult(1, "Failure: synthetic null unload failure\n");
    return ExecResult(99, "unexpected command: " + command);
  });

  err.clear();
  const bool ok = studiocast::audio::DestroyVirtualSpeaker(&err);
  if (ok || err.find("synthetic null unload failure") == std::string::npos) {
    std::cerr << "expected virtual speaker destroy to fail on unload failure; "
              << "ok=" << ok << " error='" << err << "'\n";
    return false;
  }

  const auto state = studiocast::audio::LoadVirtualSpeakerState();
  if (!state.null_sink_module_id || *state.null_sink_module_id != 10) {
    std::cerr << "failed virtual speaker destroy cleared null sink state\n";
    return false;
  }

  return true;
}

bool TestVirtualMicStopLoopbackPropagatesUnloadFailure() {
  ScopedXdgStateHome state_home;
  VirtualMicState initial;
  initial.null_sink_module_id = 10;
  initial.remap_source_module_id = 11;
  initial.loopback_module_id = 42;
  std::string err;
  if (!studiocast::audio::SaveVirtualMicState(initial, &err)) {
    std::cerr << "failed to seed virtual mic state: " << err << "\n";
    return false;
  }

  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl --version 2>&1")
      return ExecResult(0, "pactl 16.1\n");
    if (command == "pactl list short modules 2>&1") {
      return ExecResult(0,
                        "10\tmodule-null-sink\tsink_name=studiocast_sink\n"
                        "11\tmodule-remap-source\tsource_name=studiocast_mic\n"
                        "42\tmodule-loopback\tsource=physical_test_mic "
                        "sink=studiocast_sink\n");
    }
    if (command == "pactl unload-module 42 2>&1")
      return ExecResult(1, "Failure: synthetic mic loopback unload failure\n");
    return ExecResult(99, "unexpected command: " + command);
  });

  err.clear();
  const bool ok = studiocast::audio::StopLoopback(&err);
  if (ok ||
      err.find("synthetic mic loopback unload failure") == std::string::npos) {
    std::cerr << "expected mic loopback stop to fail on unload failure; ok="
              << ok << " error='" << err << "'\n";
    return false;
  }

  const auto state = studiocast::audio::LoadVirtualMicState();
  if (!state.loopback_module_id || *state.loopback_module_id != 42) {
    std::cerr << "failed mic loopback stop cleared active loopback state\n";
    return false;
  }

  if (CommandWasRun(commands, "pactl load-module")) {
    std::cerr << "mic loopback stop unexpectedly loaded a module\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  return true;
}

bool TestDestroyVirtualMicPreservesRemainingStateOnNullUnloadFailure() {
  ScopedXdgStateHome state_home;
  VirtualMicState initial;
  initial.null_sink_module_id = 10;
  initial.remap_source_module_id = 11;
  std::string err;
  if (!studiocast::audio::SaveVirtualMicState(initial, &err)) {
    std::cerr << "failed to seed virtual mic state: " << err << "\n";
    return false;
  }

  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl --version 2>&1")
      return ExecResult(0, "pactl 16.1\n");
    if (command == "pactl list short modules 2>&1") {
      return ExecResult(
          0, "10\tmodule-null-sink\tsink_name=studiocast_sink\n"
             "11\tmodule-remap-source\tsource_name=studiocast_mic\n");
    }
    if (command == "pactl unload-module 11 2>&1")
      return ExecResult(0);
    if (command == "pactl unload-module 10 2>&1")
      return ExecResult(1, "Failure: synthetic mic null unload failure\n");
    return ExecResult(99, "unexpected command: " + command);
  });

  err.clear();
  const bool ok = studiocast::audio::DestroyVirtualMic(&err);
  if (ok ||
      err.find("synthetic mic null unload failure") == std::string::npos) {
    std::cerr << "expected virtual mic destroy to fail on null sink unload; "
              << "ok=" << ok << " error='" << err << "'\n";
    return false;
  }

  const auto state = studiocast::audio::LoadVirtualMicState();
  if (!state.null_sink_module_id || *state.null_sink_module_id != 10 ||
      state.remap_source_module_id) {
    std::cerr << "failed virtual mic destroy did not preserve remaining state; "
              << "sink="
              << (state.null_sink_module_id
                      ? std::to_string(*state.null_sink_module_id)
                      : std::string("<none>"))
              << " remap="
              << (state.remap_source_module_id
                      ? std::to_string(*state.remap_source_module_id)
                      : std::string("<none>"))
              << "\n";
    return false;
  }

  return true;
}

bool TestMicrophonePipelineDoesNotStartWithoutConsumer() {
  std::atomic<int> pipeline_creates{0};
  std::atomic<bool> mic_consumer_present{false};

  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<FixedStatsPipeline>(true, "");
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.poll_ms = 1;
  cfg.consumer_grace_ms = 0;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool saw_idle = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.pipeline_state == "idle_no_consumer" &&
               !status.pipeline_active_needed && !status.pipeline_running &&
               !status.mic_consumer_present;
      },
      250ms);
  std::this_thread::sleep_for(30ms);
  const int creates = pipeline_creates.load(std::memory_order_relaxed);
  const auto status = service.Status();
  service.Stop();

  if (!saw_idle || creates != 0) {
    std::cerr << "microphone pipeline did not stay idle without consumer; "
              << "creates=" << creates << " state='" << status.pipeline_state
              << "' idle='" << status.pipeline_idle_reason
              << "' running=" << status.pipeline_running << "\n";
    return false;
  }

  return true;
}

bool TestMicrophonePipelineStartsWhenConsumerAppears() {
  std::atomic<int> pipeline_creates{0};
  std::atomic<bool> mic_consumer_present{false};

  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<FixedStatsPipeline>(true, "");
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.poll_ms = 1;
  cfg.consumer_grace_ms = 0;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] { return service.Status().pipeline_state == "idle_no_consumer"; },
          250ms)) {
    std::cerr << "microphone pipeline did not reach no-consumer idle state\n";
    service.Stop();
    return false;
  }

  mic_consumer_present.store(true, std::memory_order_relaxed);
  const bool started = WaitUntil(
      [&] {
        const auto status = service.Status();
        return pipeline_creates.load(std::memory_order_relaxed) >= 1 &&
               status.pipeline_running && status.pipeline_active_needed &&
               status.pipeline_state == "running" &&
               status.mic_consumer_present;
      },
      250ms);
  const auto status = service.Status();
  service.Stop();

  if (!started) {
    std::cerr << "microphone pipeline did not start after consumer appeared; "
              << "creates=" << pipeline_creates.load() << " state='"
              << status.pipeline_state
              << "' running=" << status.pipeline_running << "\n";
    return false;
  }

  return true;
}

bool TestMicrophonePipelineStopsWhenConsumerDisappears() {
  std::atomic<int> pipeline_creates{0};
  std::atomic<int> pipeline_stops{0};
  std::atomic<bool> mic_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<FixedStatsPipeline>(true, "", &pipeline_stops);
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.poll_ms = 1;
  cfg.consumer_grace_ms = 0;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] {
            return pipeline_creates.load(std::memory_order_relaxed) >= 1 &&
                   service.Status().pipeline_running;
          },
          250ms)) {
    std::cerr << "microphone pipeline did not start before consumer vanished\n";
    service.Stop();
    return false;
  }

  mic_consumer_present.store(false, std::memory_order_relaxed);
  const bool stopped = WaitUntil(
      [&] {
        const auto status = service.Status();
        return pipeline_stops.load(std::memory_order_relaxed) >= 1 &&
               !status.pipeline_running &&
               status.pipeline_state == "idle_no_consumer" &&
               !status.pipeline_active_needed;
      },
      250ms);
  const auto status = service.Status();
  service.Stop();

  if (!stopped) {
    std::cerr << "microphone pipeline did not stop after consumer disappeared; "
              << "stops=" << pipeline_stops.load() << " state='"
              << status.pipeline_state
              << "' running=" << status.pipeline_running << "\n";
    return false;
  }

  return true;
}

bool TestMicrophoneGraceWindowAbsorbsConsumerFlapping() {
  std::atomic<int> pipeline_creates{0};
  std::atomic<int> pipeline_stops{0};
  std::atomic<bool> mic_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<FixedStatsPipeline>(true, "", &pipeline_stops);
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.poll_ms = 1;
  cfg.consumer_grace_ms = 200;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] {
            const auto status = service.Status();
            return pipeline_creates.load(std::memory_order_relaxed) == 1 &&
                   status.pipeline_running &&
                   status.pipeline_state == "running";
          },
          250ms)) {
    std::cerr << "microphone pipeline did not start before flapping test\n";
    service.Stop();
    return false;
  }

  for (int i = 0; i < 5; ++i) {
    mic_consumer_present.store(false, std::memory_order_relaxed);
    std::this_thread::sleep_for(10ms);
    const auto absent_status = service.Status();
    if (pipeline_stops.load(std::memory_order_relaxed) != 0 ||
        !absent_status.pipeline_running ||
        !absent_status.pipeline_active_needed) {
      std::cerr << "microphone pipeline churned during grace window; i=" << i
                << " creates=" << pipeline_creates.load()
                << " stops=" << pipeline_stops.load()
                << " running=" << absent_status.pipeline_running
                << " needed=" << absent_status.pipeline_active_needed
                << " state='" << absent_status.pipeline_state << "'\n";
      service.Stop();
      return false;
    }

    mic_consumer_present.store(true, std::memory_order_relaxed);
    if (!WaitUntil(
            [&] {
              const auto status = service.Status();
              return status.pipeline_running && status.mic_consumer_present;
            },
            100ms)) {
      std::cerr << "microphone consumer did not recover during flap cycle\n";
      service.Stop();
      return false;
    }
  }

  if (pipeline_creates.load(std::memory_order_relaxed) != 1 ||
      pipeline_stops.load(std::memory_order_relaxed) != 0) {
    std::cerr << "microphone pipeline restarted during flapping; creates="
              << pipeline_creates.load() << " stops=" << pipeline_stops.load()
              << "\n";
    service.Stop();
    return false;
  }

  mic_consumer_present.store(false, std::memory_order_relaxed);
  const bool stopped = WaitUntil(
      [&] {
        const auto status = service.Status();
        return pipeline_stops.load(std::memory_order_relaxed) == 1 &&
               !status.pipeline_running && !status.pipeline_active_needed &&
               status.pipeline_state == "idle_no_consumer";
      },
      700ms);
  const auto status = service.Status();
  service.Stop();

  if (!stopped) {
    std::cerr << "microphone pipeline did not stop after sustained absence; "
              << "creates=" << pipeline_creates.load()
              << " stops=" << pipeline_stops.load()
              << " running=" << status.pipeline_running
              << " needed=" << status.pipeline_active_needed << " state='"
              << status.pipeline_state << "'\n";
    return false;
  }

  return true;
}

bool TestMicrophoneConsumerDetectionRecoversAfterErrors() {
  std::atomic<int> pipeline_creates{0};
  std::atomic<int> detection_stage{0};

  VirtualAudioServiceHooks hooks;
  hooks.detect_microphone_consumers = [&] {
    AudioConsumerSnapshot out;
    const int stage = detection_stage.load(std::memory_order_relaxed);
    if (stage == 0) {
      out.error = "synthetic Pulse server restart";
    } else if (stage == 1) {
      out.error = "Pulse source 'studiocast_mic' is not present.";
    } else {
      out = ConsumerSnapshot(true, 1);
    }
    return out;
  };
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<FixedStatsPipeline>(true, "");
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.poll_ms = 1;
  cfg.consumer_grace_ms = 0;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] {
            const auto status = service.Status();
            return status.pipeline_state == "idle_no_consumer" &&
                   status.mic_consumer_error.find("server restart") !=
                       std::string::npos &&
                   pipeline_creates.load(std::memory_order_relaxed) == 0;
          },
          250ms)) {
    const auto status = service.Status();
    std::cerr << "microphone detection error was not surfaced; state='"
              << status.pipeline_state << "' error='"
              << status.mic_consumer_error
              << "' creates=" << pipeline_creates.load() << "\n";
    service.Stop();
    return false;
  }

  detection_stage.store(1, std::memory_order_relaxed);
  if (!WaitUntil(
          [&] {
            const auto status = service.Status();
            return status.pipeline_state == "idle_no_consumer" &&
                   status.mic_consumer_error.find("not present") !=
                       std::string::npos &&
                   pipeline_creates.load(std::memory_order_relaxed) == 0;
          },
          250ms)) {
    const auto status = service.Status();
    std::cerr << "missing virtual mic source was not surfaced; state='"
              << status.pipeline_state << "' error='"
              << status.mic_consumer_error
              << "' creates=" << pipeline_creates.load() << "\n";
    service.Stop();
    return false;
  }

  detection_stage.store(2, std::memory_order_relaxed);
  const bool recovered = WaitUntil(
      [&] {
        const auto status = service.Status();
        return pipeline_creates.load(std::memory_order_relaxed) == 1 &&
               status.pipeline_running && status.pipeline_state == "running" &&
               status.mic_consumer_present && status.mic_consumer_error.empty();
      },
      250ms);
  const auto status = service.Status();
  service.Stop();

  if (!recovered) {
    std::cerr << "microphone pipeline did not recover after detection errors; "
              << "creates=" << pipeline_creates.load() << " state='"
              << status.pipeline_state
              << "' running=" << status.pipeline_running
              << " consumer=" << status.mic_consumer_present << " error='"
              << status.mic_consumer_error << "'\n";
    return false;
  }

  return true;
}

bool TestSpeakerPipelineFollowsConsumerGate() {
  std::atomic<int> pipeline_creates{0};
  std::atomic<int> pipeline_stops{0};
  std::atomic<bool> speaker_consumer_present{false};

  VirtualAudioServiceHooks hooks;
  HookSpeakerConsumerFlag(&hooks, &speaker_consumer_present);
  hooks.create_virtual_speaker = [](std::string *error) {
    if (error)
      error->clear();
    return true;
  };
  hooks.probe_speaker_backend_availability =
      [](const VirtualAudioServiceConfig &) {
        AudioBackendAvailability avail;
        avail.maxine_reason = "synthetic maxine unavailable";
        avail.open_source_reason = "synthetic open audio unavailable";
        return avail;
      };
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<FixedStatsPipeline>(true, "", &pipeline_stops);
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = false;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = true;
  cfg.speakers_enabled = true;
  cfg.speaker_target_sink = "physical_test_sink";
  cfg.effects.speaker.noise_removal_enabled = true;
  cfg.poll_ms = 1;
  cfg.consumer_grace_ms = 0;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool saw_idle = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.speakers_pipeline_state == "idle_no_consumer" &&
               !status.speakers_pipeline_active_needed &&
               !status.speakers_pipeline_running &&
               !status.speakers_consumer_present;
      },
      250ms);
  std::this_thread::sleep_for(30ms);
  if (!saw_idle || pipeline_creates.load(std::memory_order_relaxed) != 0) {
    const auto status = service.Status();
    std::cerr << "speaker pipeline did not stay idle without consumer; creates="
              << pipeline_creates.load() << " state='"
              << status.speakers_pipeline_state
              << "' running=" << status.speakers_pipeline_running << "\n";
    service.Stop();
    return false;
  }

  speaker_consumer_present.store(true, std::memory_order_relaxed);
  const bool started = WaitUntil(
      [&] {
        const auto status = service.Status();
        return pipeline_creates.load(std::memory_order_relaxed) >= 1 &&
               status.speakers_pipeline_running &&
               status.speakers_pipeline_active_needed &&
               status.speakers_pipeline_state == "running";
      },
      250ms);
  if (!started) {
    const auto status = service.Status();
    std::cerr << "speaker pipeline did not start after consumer appeared; "
              << "creates=" << pipeline_creates.load() << " state='"
              << status.speakers_pipeline_state
              << "' running=" << status.speakers_pipeline_running << "\n";
    service.Stop();
    return false;
  }

  speaker_consumer_present.store(false, std::memory_order_relaxed);
  const bool stopped = WaitUntil(
      [&] {
        const auto status = service.Status();
        return pipeline_stops.load(std::memory_order_relaxed) >= 1 &&
               !status.speakers_pipeline_running &&
               status.speakers_pipeline_state == "idle_no_consumer" &&
               !status.speakers_pipeline_active_needed;
      },
      250ms);
  const auto status = service.Status();
  service.Stop();

  if (!stopped) {
    std::cerr << "speaker pipeline did not stop after consumer disappeared; "
              << "stops=" << pipeline_stops.load() << " state='"
              << status.speakers_pipeline_state
              << "' running=" << status.speakers_pipeline_running << "\n";
    return false;
  }

  return true;
}

bool TestSpeakerGraceWindowAbsorbsConsumerFlapping() {
  std::atomic<int> pipeline_creates{0};
  std::atomic<int> pipeline_stops{0};
  std::atomic<bool> speaker_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookSpeakerConsumerFlag(&hooks, &speaker_consumer_present);
  hooks.create_virtual_speaker = [](std::string *error) {
    if (error)
      error->clear();
    return true;
  };
  hooks.probe_speaker_backend_availability =
      [](const VirtualAudioServiceConfig &) {
        AudioBackendAvailability avail;
        avail.maxine_reason = "synthetic maxine unavailable";
        avail.open_source_reason = "synthetic open audio unavailable";
        return avail;
      };
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<FixedStatsPipeline>(true, "", &pipeline_stops);
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = false;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = true;
  cfg.speakers_enabled = true;
  cfg.speaker_target_sink = "physical_test_sink";
  cfg.effects.speaker.noise_removal_enabled = true;
  cfg.poll_ms = 1;
  cfg.consumer_grace_ms = 200;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] {
            const auto status = service.Status();
            return pipeline_creates.load(std::memory_order_relaxed) == 1 &&
                   status.speakers_pipeline_running &&
                   status.speakers_pipeline_state == "running";
          },
          250ms)) {
    std::cerr << "speaker pipeline did not start before flapping test\n";
    service.Stop();
    return false;
  }

  for (int i = 0; i < 5; ++i) {
    speaker_consumer_present.store(false, std::memory_order_relaxed);
    std::this_thread::sleep_for(10ms);
    const auto absent_status = service.Status();
    if (pipeline_stops.load(std::memory_order_relaxed) != 0 ||
        !absent_status.speakers_pipeline_running ||
        !absent_status.speakers_pipeline_active_needed) {
      std::cerr << "speaker pipeline churned during grace window; i=" << i
                << " creates=" << pipeline_creates.load()
                << " stops=" << pipeline_stops.load()
                << " running=" << absent_status.speakers_pipeline_running
                << " needed=" << absent_status.speakers_pipeline_active_needed
                << " state='" << absent_status.speakers_pipeline_state << "'\n";
      service.Stop();
      return false;
    }

    speaker_consumer_present.store(true, std::memory_order_relaxed);
    if (!WaitUntil(
            [&] {
              const auto status = service.Status();
              return status.speakers_pipeline_running &&
                     status.speakers_consumer_present;
            },
            100ms)) {
      std::cerr << "speaker consumer did not recover during flap cycle\n";
      service.Stop();
      return false;
    }
  }

  if (pipeline_creates.load(std::memory_order_relaxed) != 1 ||
      pipeline_stops.load(std::memory_order_relaxed) != 0) {
    std::cerr << "speaker pipeline restarted during flapping; creates="
              << pipeline_creates.load() << " stops=" << pipeline_stops.load()
              << "\n";
    service.Stop();
    return false;
  }

  speaker_consumer_present.store(false, std::memory_order_relaxed);
  const bool stopped = WaitUntil(
      [&] {
        const auto status = service.Status();
        return pipeline_stops.load(std::memory_order_relaxed) == 1 &&
               !status.speakers_pipeline_running &&
               !status.speakers_pipeline_active_needed &&
               status.speakers_pipeline_state == "idle_no_consumer";
      },
      700ms);
  const auto status = service.Status();
  service.Stop();

  if (!stopped) {
    std::cerr << "speaker pipeline did not stop after sustained absence; "
              << "creates=" << pipeline_creates.load()
              << " stops=" << pipeline_stops.load()
              << " running=" << status.speakers_pipeline_running
              << " needed=" << status.speakers_pipeline_active_needed
              << " state='" << status.speakers_pipeline_state << "'\n";
    return false;
  }

  return true;
}

bool TestSpeakerLoopbackPassThroughStatusIsNotConsumerGated() {
  std::atomic<int> loopback_start_calls{0};
  std::atomic<int> speaker_detection_calls{0};

  VirtualAudioServiceHooks hooks;
  hooks.create_virtual_speaker = [](std::string *error) {
    if (error)
      error->clear();
    return true;
  };
  hooks.start_speaker_loopback = [&](const std::string &, int,
                                     std::string *error) {
    loopback_start_calls.fetch_add(1, std::memory_order_relaxed);
    if (error)
      error->clear();
    return true;
  };
  hooks.detect_speaker_consumers = [&] {
    speaker_detection_calls.fetch_add(1, std::memory_order_relaxed);
    return ConsumerSnapshot(false, 0);
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = false;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = true;
  cfg.speakers_enabled = true;
  cfg.speaker_target_sink = "physical_test_sink";
  cfg.poll_ms = 1;
  cfg.consumer_grace_ms = 0;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool active = WaitUntil(
      [&] {
        const auto status = service.Status();
        return loopback_start_calls.load(std::memory_order_relaxed) >= 1 &&
               status.speakers_route_mode == "loopback" &&
               status.speakers_routing_active &&
               status.speakers_pipeline_state == "disabled" &&
               status.speakers_pipeline_idle_reason ==
                   "Speaker processing is not requested.";
      },
      250ms);
  std::this_thread::sleep_for(30ms);
  const auto status = service.Status();
  const int detections =
      speaker_detection_calls.load(std::memory_order_relaxed);
  service.Stop();

  if (!active) {
    std::cerr << "speaker pass-through loopback status was not explicit; "
              << "starts=" << loopback_start_calls.load() << " route='"
              << status.speakers_route_mode
              << "' active=" << status.speakers_routing_active
              << " pipeline_state='" << status.speakers_pipeline_state
              << "' idle='" << status.speakers_pipeline_idle_reason << "'\n";
    return false;
  }

  if (detections != 0) {
    std::cerr << "speaker pass-through loopback was consumer-gated; "
              << "consumer detections=" << detections << "\n";
    return false;
  }

  return true;
}

bool TestMicrophonePipelineRestartsWhenWorkerDies() {
  std::atomic<int> pipeline_creates{0};
  std::atomic<int> pipeline_stops{0};
  std::atomic<int> sleep_calls{0};
  std::atomic<bool> mic_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<DeadPipeline>(&pipeline_stops);
  };
  hooks.sleep_for = [&](std::chrono::milliseconds) {
    sleep_calls.fetch_add(1, std::memory_order_relaxed);
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.poll_ms = 1;
  cfg.start_retry_ms = 1;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool restarted =
      WaitUntil([&] { return pipeline_creates.load() >= 2; }, 250ms);
  service.Stop();

  if (!restarted) {
    std::cerr << "expected microphone pipeline to restart after worker death;"
              << " creates=" << pipeline_creates.load()
              << " sleeps=" << sleep_calls.load() << "\n";
    return false;
  }

  if (pipeline_stops.load(std::memory_order_relaxed) == 0) {
    std::cerr
        << "expected dead microphone pipeline to be stopped before restart\n";
    return false;
  }

  return true;
}

bool TestMicrophonePipelinePreservesWorkerDeathError() {
  std::atomic<int> pipeline_creates{0};
  std::atomic<int> pipeline_stops{0};
  std::atomic<bool> mic_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    const int create_index =
        pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    if (create_index == 0) {
      return std::make_unique<FixedStatsPipeline>(
          false, "synthetic terminal pipeline failure", &pipeline_stops);
    }
    return std::make_unique<FixedStatsPipeline>(true, "", &pipeline_stops);
  };
  hooks.sleep_for = [&](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.poll_ms = 1;
  cfg.start_retry_ms = 1;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool restarted =
      WaitUntil([&] { return pipeline_creates.load() >= 2; }, 250ms);
  const auto status = service.Status();
  service.Stop();

  if (!restarted) {
    std::cerr << "expected microphone pipeline to restart after terminal error;"
              << " creates=" << pipeline_creates.load() << "\n";
    return false;
  }

  if (status.last_error.find("synthetic terminal pipeline failure") ==
      std::string::npos) {
    std::cerr << "expected terminal pipeline error to remain visible; got '"
              << status.last_error << "'\n";
    return false;
  }

  return true;
}

bool TestStatusDoesNotBlockDuringRetrySleep() {
  std::mutex mu;
  std::condition_variable cv;
  bool sleep_entered = false;
  bool release_sleep = false;
  std::atomic<int> sleep_calls{0};
  std::atomic<bool> mic_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    return std::make_unique<StartFailPipeline>();
  };
  hooks.sleep_for = [&](std::chrono::milliseconds) {
    const int call_index = sleep_calls.fetch_add(1, std::memory_order_relaxed);
    if (call_index != 0) {
      std::this_thread::sleep_for(1ms);
      return;
    }

    std::unique_lock<std::mutex> lock(mu);
    sleep_entered = true;
    cv.notify_all();
    cv.wait(lock, [&] { return release_sleep; });
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.poll_ms = 1;
  cfg.start_retry_ms = 1;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  {
    std::unique_lock<std::mutex> lock(mu);
    if (!cv.wait_for(lock, 250ms, [&] { return sleep_entered; })) {
      std::cerr << "service did not enter retry sleep\n";
      service.Stop();
      return false;
    }
  }

  auto status_future =
      std::async(std::launch::async, [&] { return service.Status(); });
  const bool ready_before_release =
      (status_future.wait_for(50ms) == std::future_status::ready);

  {
    std::lock_guard<std::mutex> lock(mu);
    release_sleep = true;
  }
  cv.notify_all();

  service.Stop();
  (void)status_future.get();

  if (!ready_before_release) {
    std::cerr << "Status() blocked while the service was backing off\n";
    return false;
  }

  return true;
}

bool TestMicrophoneNullPipelineFactoryFailsWithoutCrash() {
  std::atomic<bool> mic_consumer_present{true};
  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.create_pipeline =
      [](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    return nullptr;
  };
  hooks.sleep_for = [&](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.poll_ms = 1;
  cfg.start_retry_ms = 250;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool saw_error = WaitUntil(
      [&] {
        const auto status = service.Status();
        return !status.pipeline_running &&
               status.last_error.find("pipeline factory returned null") !=
                   std::string::npos;
      },
      250ms);
  const auto status = service.Status();
  service.Stop();

  if (!saw_error) {
    std::cerr << "expected null microphone pipeline factory to surface an "
                 "error; got '"
              << status.last_error << "' running=" << status.pipeline_running
              << "\n";
    return false;
  }

  return true;
}

bool TestSpeakerPipelineStartFailureClearsRouteState() {
  std::atomic<int> pipeline_creates{0};
  std::atomic<bool> speaker_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookSpeakerConsumerFlag(&hooks, &speaker_consumer_present);
  hooks.create_virtual_speaker = [](std::string *error) {
    if (error)
      error->clear();
    return true;
  };
  hooks.probe_speaker_backend_availability =
      [](const VirtualAudioServiceConfig &) {
        AudioBackendAvailability avail;
        avail.maxine_reason = "synthetic maxine unavailable";
        avail.open_source_reason = "synthetic open audio unavailable";
        return avail;
      };
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<StartFailPipeline>();
  };
  hooks.sleep_for = [&](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = false;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.speakers_enabled = true;
  cfg.speaker_target_sink = "physical_test_sink";
  cfg.effects.speaker.noise_removal_enabled = true;
  cfg.poll_ms = 1;
  cfg.start_retry_ms = 250;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool saw_failure = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.speakers_pipeline_last_error.find(
                   "synthetic start failure") != std::string::npos;
      },
      250ms);
  const auto status = service.Status();
  service.Stop();

  if (!saw_failure) {
    std::cerr << "expected speaker pipeline start failure; creates="
              << pipeline_creates.load() << " error='"
              << status.speakers_pipeline_last_error << "'\n";
    return false;
  }

  if (status.speakers_routing_active ||
      !status.speaker_target_sink_active.empty()) {
    std::cerr << "speaker route looked active after pipeline start failure; "
              << "routing=" << status.speakers_routing_active << " target='"
              << status.speaker_target_sink_active << "'\n";
    return false;
  }

  return true;
}

bool TestOpenAudioFailureCooldownAvoidsRestartChurn() {
  std::atomic<int> pipeline_creates{0};
  std::atomic<bool> mic_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.probe_microphone_backend_availability =
      [](const VirtualAudioServiceConfig &) {
        AudioBackendAvailability avail;
        avail.open_source_ok = true;
        return avail;
      };
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<FixedStatsPipeline>(true, "");
  };
  hooks.sleep_for = [&](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.effects.engine =
      studiocast::audio::effects::AudioEffectsEnginePreference::kOpenSource;
  cfg.effects.microphone.noise_removal_enabled = true;
  cfg.effects.microphone.model_path =
      "/tmp/studiocast-definitely-missing-open-audio-model.onnx";
  cfg.poll_ms = 1;
  cfg.start_retry_ms = 1000;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool started_fallback =
      WaitUntil([&] { return pipeline_creates.load() >= 1; }, 250ms);
  std::this_thread::sleep_for(75ms);
  const int creates_after_cooldown_window = pipeline_creates.load();
  const auto status = service.Status();
  service.Stop();

  if (!started_fallback) {
    std::cerr
        << "expected fallback pipeline to start after Open Audio failure\n";
    return false;
  }

  if (creates_after_cooldown_window != 1) {
    std::cerr << "Open Audio cooldown did not prevent restart churn; creates="
              << creates_after_cooldown_window << "\n";
    return false;
  }

  if (status.effects_backend_active != "passthrough") {
    std::cerr
        << "expected passthrough backend during Open Audio cooldown; got '"
        << status.effects_backend_active << "'\n";
    return false;
  }

  return true;
}

bool TestForcedMaxineMicrophoneFailureFallsBackToPassthrough() {
  EnvGuard afx_env("STUDIOCAST_AFX_SDK_ROOT",
                   "/tmp/studiocast-definitely-missing-afx-sdk");

  std::atomic<int> pipeline_creates{0};
  std::atomic<bool> mic_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.probe_microphone_backend_availability =
      [](const VirtualAudioServiceConfig &) {
        AudioBackendAvailability avail;
        avail.maxine_ok = true;
        avail.open_source_ok = false;
        avail.open_source_reason = "synthetic open audio unavailable";
        return avail;
      };
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<FixedStatsPipeline>(true, "");
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.effects.engine =
      studiocast::audio::effects::AudioEffectsEnginePreference::kMaxine;
  cfg.effects.microphone.noise_removal_enabled = true;
  cfg.poll_ms = 1;
  cfg.start_retry_ms = 500;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool started_fallback = WaitUntil(
      [&] {
        const auto status = service.Status();
        return pipeline_creates.load(std::memory_order_relaxed) >= 1 &&
               status.pipeline_running && status.pipeline_state == "running" &&
               status.effects_backend_active == "passthrough" &&
               status.effects_note.find("using pass-through") !=
                   std::string::npos;
      },
      300ms);
  std::this_thread::sleep_for(75ms);
  const int creates_after_settle =
      pipeline_creates.load(std::memory_order_relaxed);
  const auto status = service.Status();
  service.Stop();

  if (!started_fallback) {
    std::cerr << "forced Maxine microphone failure did not fall back to "
                 "pass-through; creates="
              << pipeline_creates.load() << " state='" << status.pipeline_state
              << "' backend='" << status.effects_backend_active << "' note='"
              << status.effects_note << "' error='" << status.last_error
              << "'\n";
    return false;
  }

  if (creates_after_settle != 1) {
    std::cerr << "forced Maxine microphone fallback churned restarts; creates="
              << creates_after_settle << "\n";
    return false;
  }

  return true;
}

bool TestMicrophoneAvailabilityCacheIgnoresSpeakerOnlyChanges() {
  std::atomic<int> mic_probes{0};
  std::atomic<bool> mic_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.probe_microphone_backend_availability =
      [&](const VirtualAudioServiceConfig &) {
        mic_probes.fetch_add(1, std::memory_order_relaxed);
        AudioBackendAvailability avail;
        avail.maxine_reason = "synthetic maxine unavailable";
        avail.open_source_reason = "synthetic open audio unavailable";
        return avail;
      };
  hooks.create_pipeline =
      [](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    return std::make_unique<FixedStatsPipeline>(true, "");
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.effects.microphone.noise_removal_enabled = true;
  cfg.poll_ms = 1;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil([&] { return mic_probes.load() >= 1; }, 250ms)) {
    std::cerr << "microphone availability was not probed\n";
    service.Stop();
    return false;
  }

  const int probes_before = mic_probes.load();
  cfg.effects.speaker.noise_removal_enabled = true;
  cfg.effects.speaker.strength = 73;
  service.UpdateConfig(cfg);
  std::this_thread::sleep_for(90ms);
  const int probes_after = mic_probes.load();
  service.Stop();

  if (probes_after != probes_before) {
    std::cerr << "speaker-only effects change invalidated microphone "
                 "availability cache; before="
              << probes_before << " after=" << probes_after << "\n";
    return false;
  }

  return true;
}

bool TestSpeakerAvailabilityCacheIgnoresMicrophoneOnlyChanges() {
  std::atomic<int> speaker_probes{0};
  std::atomic<bool> speaker_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookSpeakerConsumerFlag(&hooks, &speaker_consumer_present);
  hooks.create_virtual_speaker = [](std::string *error) {
    if (error)
      error->clear();
    return true;
  };
  hooks.probe_speaker_backend_availability =
      [&](const VirtualAudioServiceConfig &) {
        speaker_probes.fetch_add(1, std::memory_order_relaxed);
        AudioBackendAvailability avail;
        avail.maxine_reason = "synthetic maxine unavailable";
        avail.open_source_reason = "synthetic open audio unavailable";
        return avail;
      };
  hooks.create_pipeline =
      [](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    return std::make_unique<FixedStatsPipeline>(true, "");
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = false;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = true;
  cfg.speakers_enabled = true;
  cfg.speaker_target_sink = "physical_test_sink";
  cfg.effects.speaker.noise_removal_enabled = true;
  cfg.poll_ms = 1;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil([&] { return speaker_probes.load() >= 1; }, 250ms)) {
    std::cerr << "speaker availability was not probed\n";
    service.Stop();
    return false;
  }

  const int probes_before = speaker_probes.load();
  cfg.effects.microphone.noise_removal_enabled = true;
  cfg.effects.microphone.strength = 82;
  service.UpdateConfig(cfg);
  std::this_thread::sleep_for(90ms);
  const int probes_after = speaker_probes.load();
  service.Stop();

  if (probes_after != probes_before) {
    std::cerr << "microphone-only effects change invalidated speaker "
                 "availability cache; before="
              << probes_before << " after=" << probes_after << "\n";
    return false;
  }

  return true;
}

bool TestStableAudioPreparationUsesExplicitInvalidation() {
  std::atomic<int> backend_probes{0};
  std::atomic<int> source_probes{0};
  std::atomic<int> pipeline_creates{0};
  std::atomic<bool> mic_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.probe_microphone_backend_availability =
      [&](const VirtualAudioServiceConfig &) {
        backend_probes.fetch_add(1, std::memory_order_relaxed);
        AudioBackendAvailability avail;
        avail.maxine_reason = "synthetic maxine unavailable";
        avail.open_source_reason = "synthetic open audio unavailable";
        return avail;
      };
  hooks.before_preparation_probe = [&](std::string_view name,
                                       const std::atomic_bool &) {
    if (name == "source")
      source_probes.fetch_add(1, std::memory_order_relaxed);
  };
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<FixedStatsPipeline>(true, "");
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.effects.microphone.noise_removal_enabled = true;
  cfg.poll_ms = 1;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] {
            return backend_probes.load(std::memory_order_relaxed) == 1 &&
                   source_probes.load(std::memory_order_relaxed) == 1 &&
                   pipeline_creates.load(std::memory_order_relaxed) == 1;
          },
          250ms)) {
    std::cerr << "initial audio preparation did not complete exactly once\n";
    service.Stop();
    return false;
  }

  // Cross the previous two-second TTL. Stable active service must remain on
  // the cached preparation and authoritative cached source status.
  std::this_thread::sleep_for(2100ms);
  if (backend_probes.load() != 1 || source_probes.load() != 1 ||
      pipeline_creates.load() != 1) {
    std::cerr << "stable audio repeated preparation after old TTL; backend="
              << backend_probes.load() << " source=" << source_probes.load()
              << " pipelines=" << pipeline_creates.load() << "\n";
    service.Stop();
    return false;
  }

  service.RefreshPreparation();
  const bool refreshed = WaitUntil(
      [&] {
        return backend_probes.load(std::memory_order_relaxed) == 2 &&
               source_probes.load(std::memory_order_relaxed) == 2 &&
               pipeline_creates.load(std::memory_order_relaxed) == 2;
      },
      250ms);
  cfg.effects.microphone.model_id = "synthetic-new-model";
  service.UpdateConfig(cfg);
  const bool config_rebuilt = WaitUntil(
      [&] {
        return backend_probes.load(std::memory_order_relaxed) == 3 &&
               source_probes.load(std::memory_order_relaxed) == 2 &&
               pipeline_creates.load(std::memory_order_relaxed) == 2;
      },
      250ms);
  std::this_thread::sleep_for(50ms);
  const int final_backend = backend_probes.load();
  const int final_source = source_probes.load();
  const int final_pipelines = pipeline_creates.load();
  service.Stop();

  if (!refreshed || !config_rebuilt || final_backend != 3 ||
      final_source != 2 || final_pipelines != 2) {
    std::cerr << "explicit refresh did not produce exactly one preparation; "
              << "backend=" << final_backend << " source=" << final_source
              << " pipelines=" << final_pipelines << "\n";
    return false;
  }
  return true;
}

bool TestForcedOpenAudioSkipsMaxinePreparation() {
  std::atomic<int> maxine_probes{0};
  std::atomic<bool> mic_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.before_preparation_probe = [&](std::string_view name,
                                       const std::atomic_bool &) {
    if (name == "settings" || name == "gpu_selection" || name == "sdk_paths") {
      maxine_probes.fetch_add(1, std::memory_order_relaxed);
    }
  };
  hooks.create_pipeline =
      [](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    return std::make_unique<FixedStatsPipeline>(true, "");
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.effects.engine =
      studiocast::audio::effects::AudioEffectsEnginePreference::kOpenSource;
  cfg.effects.microphone.noise_removal_enabled = true;
  cfg.effects.microphone.model_path =
      "/tmp/studiocast-definitely-missing-open-audio-model.onnx";
  cfg.poll_ms = 1;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }
  std::this_thread::sleep_for(100ms);
  service.Stop();
  if (maxine_probes.load() != 0) {
    std::cerr << "forced Open Audio performed Maxine preparation; probes="
              << maxine_probes.load() << "\n";
    return false;
  }
  return true;
}

bool TestStopInterruptsBlockedPreparationHook() {
  std::mutex mu;
  std::condition_variable cv;
  bool entered = false;

  VirtualAudioServiceHooks hooks;
  hooks.before_preparation_probe = [&](std::string_view name,
                                       const std::atomic_bool &stop_requested) {
    if (name != "source")
      return;
    {
      std::lock_guard<std::mutex> lock(mu);
      entered = true;
    }
    cv.notify_all();
    while (!stop_requested.load(std::memory_order_acquire))
      std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }
  {
    std::unique_lock<std::mutex> lock(mu);
    if (!cv.wait_for(lock, 250ms, [&] { return entered; })) {
      std::cerr << "service did not enter blocked preparation hook\n";
      service.Stop();
      return false;
    }
  }
  const auto started = std::chrono::steady_clock::now();
  service.Stop();
  const auto elapsed = std::chrono::steady_clock::now() - started;
  if (elapsed > 100ms) {
    std::cerr << "Stop exceeded bound during blocked preparation: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                     .count()
              << "ms\n";
    return false;
  }
  return true;
}

bool TestExecCaptureCancellationBoundsProviderHelper() {
  std::atomic_bool stop_requested{false};
  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = 5000;
  options.stop_requested = &stop_requested;

  const auto started = std::chrono::steady_clock::now();
  auto future = std::async(std::launch::async, [&] {
    return studiocast::util::ExecCapture("exec sleep 5", options);
  });
  std::this_thread::sleep_for(50ms);
  stop_requested.store(true, std::memory_order_release);
  const auto result = future.get();
  const auto elapsed = std::chrono::steady_clock::now() - started;
  if (!result.cancelled || elapsed > 500ms) {
    std::cerr << "provider helper cancellation was not bounded; cancelled="
              << result.cancelled << " elapsed_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                     .count()
              << "\n";
    return false;
  }
  return true;
}

bool TestMicrophoneDeadWorkerBacksOffBeforeRestart() {
  std::atomic<int> pipeline_creates{0};
  std::atomic<int> pipeline_stops{0};
  std::atomic<bool> mic_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookMicrophoneConsumerFlag(&hooks, &mic_consumer_present);
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<DeadPipeline>(&pipeline_stops);
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = true;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = false;
  cfg.poll_ms = 1;
  cfg.start_retry_ms = 80;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil([&] { return pipeline_creates.load() >= 1; }, 250ms)) {
    std::cerr << "microphone pipeline was not started\n";
    service.Stop();
    return false;
  }

  std::this_thread::sleep_for(40ms);
  const int creates_during_backoff = pipeline_creates.load();
  const bool restarted =
      WaitUntil([&] { return pipeline_creates.load() >= 2; }, 300ms);
  service.Stop();

  if (creates_during_backoff != 1) {
    std::cerr << "dead microphone worker restarted before retry backoff; "
              << "creates=" << creates_during_backoff << "\n";
    return false;
  }

  if (!restarted) {
    std::cerr << "dead microphone worker did not restart after retry backoff; "
              << "creates=" << pipeline_creates.load() << "\n";
    return false;
  }

  return true;
}

bool TestSpeakerDeadWorkerBacksOffAndClearsRoute() {
  std::atomic<int> pipeline_creates{0};
  std::atomic<int> pipeline_stops{0};
  std::atomic<bool> speaker_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookSpeakerConsumerFlag(&hooks, &speaker_consumer_present);
  hooks.create_virtual_speaker = [](std::string *error) {
    if (error)
      error->clear();
    return true;
  };
  hooks.probe_speaker_backend_availability =
      [](const VirtualAudioServiceConfig &) {
        AudioBackendAvailability avail;
        avail.maxine_reason = "synthetic maxine unavailable";
        avail.open_source_reason = "synthetic open audio unavailable";
        return avail;
      };
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<DeadPipeline>(&pipeline_stops);
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = false;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = true;
  cfg.speakers_enabled = true;
  cfg.speaker_target_sink = "physical_test_sink";
  cfg.effects.speaker.noise_removal_enabled = true;
  cfg.poll_ms = 1;
  cfg.start_retry_ms = 80;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil([&] { return pipeline_creates.load() >= 1; }, 250ms)) {
    std::cerr << "speaker pipeline was not started\n";
    service.Stop();
    return false;
  }

  const bool saw_inactive = WaitUntil(
      [&] {
        const auto status = service.Status();
        return !status.speakers_routing_active &&
               status.speaker_target_sink_active.empty() &&
               status.speakers_pipeline_last_error.find(
                   "Speaker audio pipeline stopped") != std::string::npos;
      },
      250ms);
  std::this_thread::sleep_for(40ms);
  const int creates_during_backoff = pipeline_creates.load();
  const bool restarted =
      WaitUntil([&] { return pipeline_creates.load() >= 2; }, 300ms);
  service.Stop();

  if (!saw_inactive) {
    std::cerr << "speaker worker death did not clear active route status\n";
    return false;
  }

  if (creates_during_backoff != 1) {
    std::cerr << "dead speaker worker restarted before retry backoff; creates="
              << creates_during_backoff << "\n";
    return false;
  }

  if (!restarted) {
    std::cerr << "dead speaker worker did not restart after retry backoff; "
              << "creates=" << pipeline_creates.load() << "\n";
    return false;
  }

  return true;
}

bool TestSpeakerPipelineStatsClearWhenProcessingDisabled() {
  std::atomic<bool> speaker_consumer_present{true};
  AudioPipelineStats synthetic_stats;
  synthetic_stats.running = true;
  synthetic_stats.frames_processed = 123;
  synthetic_stats.process_time_us_sum = 456;
  synthetic_stats.process_time_us_max = 78;
  synthetic_stats.process_time_us_last = 9;
  synthetic_stats.process_overruns = 2;
  synthetic_stats.pulse_capture_latency_us_last = 1000;
  synthetic_stats.pulse_playback_latency_us_last = 2000;
  synthetic_stats.pulse_latency_us_max = 3000;
  synthetic_stats.resync_events = 4;

  VirtualAudioServiceHooks hooks;
  HookSpeakerConsumerFlag(&hooks, &speaker_consumer_present);
  hooks.create_virtual_speaker = [](std::string *error) {
    if (error)
      error->clear();
    return true;
  };
  hooks.probe_speaker_backend_availability =
      [](const VirtualAudioServiceConfig &) {
        AudioBackendAvailability avail;
        avail.maxine_reason = "synthetic maxine unavailable";
        avail.open_source_reason = "synthetic open audio unavailable";
        return avail;
      };
  hooks.create_pipeline =
      [synthetic_stats](
          AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    return std::make_unique<FixedStatsPipeline>(synthetic_stats);
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = false;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = true;
  cfg.speakers_enabled = true;
  cfg.speaker_target_sink = "physical_test_sink";
  cfg.effects.speaker.noise_removal_enabled = true;
  cfg.poll_ms = 1;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool saw_stats = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.speakers_pipeline_frames_processed == 123 &&
               status.speakers_pipeline_pulse_latency_us_max == 3000;
      },
      250ms);
  if (!saw_stats) {
    const auto status = service.Status();
    std::cerr << "speaker pipeline stats were not published; frames="
              << status.speakers_pipeline_frames_processed << " max_latency="
              << status.speakers_pipeline_pulse_latency_us_max << "\n";
    service.Stop();
    return false;
  }

  cfg.speakers_enabled = false;
  service.UpdateConfig(cfg);
  const bool cleared = WaitUntil(
      [&] {
        const auto status = service.Status();
        return status.speakers_route_mode == "off" &&
               !status.speakers_pipeline_running &&
               status.speakers_pipeline_frames_processed == 0 &&
               status.speakers_pipeline_process_time_us_sum == 0 &&
               status.speakers_pipeline_pulse_capture_latency_us_last == 0 &&
               status.speakers_pipeline_pulse_playback_latency_us_last == 0 &&
               status.speakers_pipeline_pulse_latency_us_max == 0 &&
               status.speakers_pipeline_resync_events == 0;
      },
      250ms);
  const auto status = service.Status();
  service.Stop();

  if (!cleared) {
    std::cerr << "speaker pipeline stats stayed stale after disabling "
                 "processing; route='"
              << status.speakers_route_mode
              << "' frames=" << status.speakers_pipeline_frames_processed
              << " max_latency="
              << status.speakers_pipeline_pulse_latency_us_max << "\n";
    return false;
  }

  return true;
}

bool TestSpeakerLoopbackRestartFailureClearsRouteState() {
  std::atomic<int> loopback_start_calls{0};

  VirtualAudioServiceHooks hooks;
  hooks.create_virtual_speaker = [](std::string *error) {
    if (error)
      error->clear();
    return true;
  };
  hooks.start_speaker_loopback = [&](const std::string &, int,
                                     std::string *error) {
    const int call =
        loopback_start_calls.fetch_add(1, std::memory_order_relaxed);
    if (call == 0) {
      if (error)
        error->clear();
      return true;
    }
    if (error)
      *error = "synthetic loopback load failure";
    return false;
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = false;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = true;
  cfg.speakers_enabled = true;
  cfg.speaker_target_sink = "physical_test_sink";
  cfg.poll_ms = 1;
  cfg.start_retry_ms = 200;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool first_route_active = WaitUntil(
      [&] {
        const auto status = service.Status();
        return loopback_start_calls.load(std::memory_order_relaxed) >= 1 &&
               status.speakers_routing_active &&
               status.speakers_route_mode == "loopback";
      },
      250ms);
  if (!first_route_active) {
    const auto status = service.Status();
    std::cerr << "speaker loopback route did not become active; starts="
              << loopback_start_calls.load()
              << " active=" << status.speakers_routing_active << " route='"
              << status.speakers_route_mode << "' error='"
              << status.speakers_last_error << "'\n";
    service.Stop();
    return false;
  }

  cfg.speaker_target_sink = "other_physical_test_sink";
  service.UpdateConfig(cfg);

  const bool cleared_after_failure = WaitUntil(
      [&] {
        const auto status = service.Status();
        return loopback_start_calls.load(std::memory_order_relaxed) >= 2 &&
               !status.speakers_routing_active &&
               status.speaker_target_sink_active.empty() &&
               status.speakers_last_error.find(
                   "synthetic loopback load failure") != std::string::npos;
      },
      250ms);
  const int starts_after_failure =
      loopback_start_calls.load(std::memory_order_relaxed);
  std::this_thread::sleep_for(50ms);
  const int starts_during_backoff =
      loopback_start_calls.load(std::memory_order_relaxed);
  const auto status = service.Status();
  service.Stop();

  if (!cleared_after_failure) {
    std::cerr << "speaker loopback failure left route active; starts="
              << loopback_start_calls.load()
              << " active=" << status.speakers_routing_active << " target='"
              << status.speaker_target_sink_active << "' error='"
              << status.speakers_last_error << "'\n";
    return false;
  }

  if (starts_during_backoff != starts_after_failure) {
    std::cerr << "speaker loopback failure retried before backoff elapsed; "
              << "after_failure=" << starts_after_failure
              << " during_backoff=" << starts_during_backoff << "\n";
    return false;
  }

  return true;
}

bool TestSpeakerLoopbackRealHelperStopFailureKeepsOldRouteActive() {
  ScopedXdgStateHome state_home;
  VirtualSpeakerState initial;
  initial.null_sink_module_id = 10;
  initial.loopback_module_id = 42;
  initial.loopback_target_sink_name = "old_physical_sink";
  std::string err;
  if (!studiocast::audio::SaveVirtualSpeakerState(initial, &err)) {
    std::cerr << "failed to seed virtual speaker state: " << err << "\n";
    return false;
  }

  std::atomic<int> unload_calls{0};
  std::vector<std::string> commands;
  ScopedPactlExecHook hook([&](const std::string &command) {
    commands.push_back(command);
    if (command == "pactl --version 2>&1")
      return ExecResult(0, "pactl 16.1\n");
    if (command == "pactl list short modules 2>&1") {
      return ExecResult(
          0, "10\tmodule-null-sink\tsink_name=studiocast_speakers\n"
             "42\tmodule-loopback\tsource=studiocast_speakers.monitor "
             "sink=old_physical_sink\n");
    }
    if (command.find("pactl update-sink-proplist 'studiocast_speakers'") == 0)
      return ExecResult(0);
    if (command == "pactl unload-module 42 2>&1") {
      unload_calls.fetch_add(1, std::memory_order_relaxed);
      return ExecResult(1, "Failure: synthetic old loopback unload failure\n");
    }
    return ExecResult(99, "unexpected command: " + command);
  });

  VirtualAudioServiceHooks hooks;
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = false;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = true;
  cfg.speakers_enabled = true;
  cfg.speaker_target_sink = "new_physical_sink";
  cfg.poll_ms = 1;
  cfg.start_retry_ms = 200;

  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  const bool old_route_preserved = WaitUntil(
      [&] {
        const auto status = service.Status();
        return unload_calls.load(std::memory_order_relaxed) >= 1 &&
               status.speakers_route_mode == "loopback" &&
               status.speakers_routing_active &&
               status.speaker_target_sink_active == "old_physical_sink" &&
               status.speakers_last_error.find(
                   "synthetic old loopback unload failure") !=
                   std::string::npos;
      },
      250ms);
  const int unloads_after_failure =
      unload_calls.load(std::memory_order_relaxed);
  std::this_thread::sleep_for(50ms);
  const int unloads_during_backoff =
      unload_calls.load(std::memory_order_relaxed);
  const auto status = service.Status();
  service.Stop();

  if (!old_route_preserved) {
    std::cerr << "real helper stop failure did not preserve old loopback; "
              << "unloads=" << unload_calls.load() << " route='"
              << status.speakers_route_mode
              << "' active=" << status.speakers_routing_active << " target='"
              << status.speaker_target_sink_active << "' error='"
              << status.speakers_last_error << "'\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  if (unloads_during_backoff != unloads_after_failure) {
    std::cerr << "real helper stop failure retried before backoff elapsed; "
              << "after_failure=" << unloads_after_failure
              << " during_backoff=" << unloads_during_backoff << "\n";
    return false;
  }

  if (CommandWasRun(commands, "pactl load-module 'module-loopback'")) {
    std::cerr << "new speaker loopback loaded after old route stop failed\n";
    for (const auto &command : commands)
      std::cerr << "  " << command << "\n";
    return false;
  }

  return true;
}

bool TestVirtualSpeakerDestroyFailureBacksOffAndKeepsPresent() {
  std::atomic<int> create_calls{0};
  std::atomic<int> destroy_calls{0};

  VirtualAudioServiceHooks hooks;
  hooks.create_virtual_speaker = [&](std::string *error) {
    create_calls.fetch_add(1, std::memory_order_relaxed);
    if (error)
      error->clear();
    return true;
  };
  hooks.destroy_virtual_speaker = [&](std::string *error) {
    destroy_calls.fetch_add(1, std::memory_order_relaxed);
    if (error)
      *error = "synthetic virtual speaker destroy failure";
    return false;
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = false;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = true;
  cfg.speakers_enabled = false;
  cfg.poll_ms = 1;
  cfg.start_retry_ms = 200;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] {
            const auto status = service.Status();
            return create_calls.load(std::memory_order_relaxed) >= 1 &&
                   status.speakers_present;
          },
          250ms)) {
    std::cerr << "virtual speakers did not become present before destroy\n";
    service.Stop();
    return false;
  }

  cfg.create_virtual_speakers = false;
  service.UpdateConfig(cfg);

  const bool destroy_failed = WaitUntil(
      [&] {
        const auto status = service.Status();
        return destroy_calls.load(std::memory_order_relaxed) >= 1 &&
               status.speakers_present &&
               status.speakers_last_error.find(
                   "synthetic virtual speaker destroy failure") !=
                   std::string::npos;
      },
      250ms);
  const int destroys_after_failure =
      destroy_calls.load(std::memory_order_relaxed);
  std::this_thread::sleep_for(50ms);
  const int destroys_during_backoff =
      destroy_calls.load(std::memory_order_relaxed);
  const auto status = service.Status();
  service.Stop();

  if (!destroy_failed) {
    std::cerr << "virtual speaker destroy failure did not keep status present; "
              << "destroys=" << destroy_calls.load()
              << " present=" << status.speakers_present << " error='"
              << status.speakers_last_error << "'\n";
    return false;
  }

  if (destroys_during_backoff != destroys_after_failure) {
    std::cerr << "virtual speaker destroy retried before backoff elapsed; "
              << "after_failure=" << destroys_after_failure
              << " during_backoff=" << destroys_during_backoff << "\n";
    return false;
  }

  return true;
}

bool TestSpeakerLoopbackStopFailureBlocksPipelineStart() {
  std::atomic<int> loopback_start_calls{0};
  std::atomic<int> loopback_stop_calls{0};
  std::atomic<int> pipeline_creates{0};
  std::atomic<bool> speaker_consumer_present{true};

  VirtualAudioServiceHooks hooks;
  HookSpeakerConsumerFlag(&hooks, &speaker_consumer_present);
  hooks.create_virtual_speaker = [](std::string *error) {
    if (error)
      error->clear();
    return true;
  };
  hooks.start_speaker_loopback = [&](const std::string &, int,
                                     std::string *error) {
    loopback_start_calls.fetch_add(1, std::memory_order_relaxed);
    if (error)
      error->clear();
    return true;
  };
  hooks.stop_speaker_loopback = [&](std::string *error) {
    loopback_stop_calls.fetch_add(1, std::memory_order_relaxed);
    if (error)
      *error = "synthetic loopback stop failure";
    return false;
  };
  hooks.probe_speaker_backend_availability =
      [](const VirtualAudioServiceConfig &) {
        AudioBackendAvailability avail;
        avail.maxine_reason = "synthetic maxine unavailable";
        avail.open_source_reason = "synthetic open audio unavailable";
        return avail;
      };
  hooks.create_pipeline =
      [&](AudioProcessor *) -> std::unique_ptr<AudioPipelineRunner> {
    pipeline_creates.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<FixedStatsPipeline>(true, "");
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = false;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = true;
  cfg.speakers_enabled = true;
  cfg.speaker_target_sink = "physical_test_sink";
  cfg.poll_ms = 1;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] {
            const auto status = service.Status();
            return loopback_start_calls.load(std::memory_order_relaxed) >= 1 &&
                   status.speakers_route_mode == "loopback" &&
                   status.speakers_routing_active;
          },
          250ms)) {
    const auto status = service.Status();
    std::cerr << "speaker loopback did not become active before processing "
                 "transition; starts="
              << loopback_start_calls.load() << " route='"
              << status.speakers_route_mode
              << "' active=" << status.speakers_routing_active << "\n";
    service.Stop();
    return false;
  }

  cfg.effects.speaker.noise_removal_enabled = true;
  service.UpdateConfig(cfg);

  const bool stop_failed = WaitUntil(
      [&] {
        const auto status = service.Status();
        return loopback_stop_calls.load(std::memory_order_relaxed) >= 1 &&
               status.speakers_route_mode == "loopback" &&
               status.speakers_routing_active &&
               !status.speakers_pipeline_running &&
               !status.speakers_pipeline_starting &&
               status.speakers_last_error.find("synthetic loopback stop "
                                               "failure") != std::string::npos;
      },
      250ms);
  const int stops_after_failure =
      loopback_stop_calls.load(std::memory_order_relaxed);
  std::this_thread::sleep_for(30ms);
  const int stops_during_backoff =
      loopback_stop_calls.load(std::memory_order_relaxed);
  const int creates_after_stop_failure =
      pipeline_creates.load(std::memory_order_relaxed);
  const auto status = service.Status();
  service.Stop();

  if (!stop_failed) {
    std::cerr << "speaker stop failure did not keep loopback route active; "
              << "stops=" << loopback_stop_calls.load() << " route='"
              << status.speakers_route_mode
              << "' active=" << status.speakers_routing_active
              << " pipeline_running=" << status.speakers_pipeline_running
              << " pipeline_starting=" << status.speakers_pipeline_starting
              << " error='" << status.speakers_last_error << "'\n";
    return false;
  }

  if (stops_during_backoff != stops_after_failure) {
    std::cerr << "speaker loopback stop retried before backoff elapsed; "
              << "after_failure=" << stops_after_failure
              << " during_backoff=" << stops_during_backoff << "\n";
    return false;
  }

  if (creates_after_stop_failure != 0) {
    std::cerr << "speaker pipeline started while loopback stop was failing; "
              << "creates=" << creates_after_stop_failure << "\n";
    return false;
  }

  return true;
}

bool TestSpeakerLoopbackStopFailurePreventsDestroyAndKeepsRoute() {
  std::atomic<int> loopback_start_calls{0};
  std::atomic<int> loopback_stop_calls{0};
  std::atomic<int> destroy_calls{0};

  VirtualAudioServiceHooks hooks;
  hooks.create_virtual_speaker = [](std::string *error) {
    if (error)
      error->clear();
    return true;
  };
  hooks.start_speaker_loopback = [&](const std::string &, int,
                                     std::string *error) {
    loopback_start_calls.fetch_add(1, std::memory_order_relaxed);
    if (error)
      error->clear();
    return true;
  };
  hooks.stop_speaker_loopback = [&](std::string *error) {
    loopback_stop_calls.fetch_add(1, std::memory_order_relaxed);
    if (error)
      *error = "synthetic loopback stop failure";
    return false;
  };
  hooks.destroy_virtual_speaker = [&](std::string *error) {
    destroy_calls.fetch_add(1, std::memory_order_relaxed);
    if (error)
      error->clear();
    return true;
  };
  hooks.sleep_for = [](std::chrono::milliseconds) {
    std::this_thread::sleep_for(1ms);
  };

  VirtualAudioService service(std::move(hooks));
  VirtualAudioServiceConfig cfg;
  cfg.enabled = false;
  cfg.create_virtual_mic = false;
  cfg.create_virtual_speakers = true;
  cfg.speakers_enabled = true;
  cfg.speaker_target_sink = "physical_test_sink";
  cfg.poll_ms = 1;

  std::string err;
  if (!service.Start(cfg, &err)) {
    std::cerr << "service.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] {
            const auto status = service.Status();
            return loopback_start_calls.load(std::memory_order_relaxed) >= 1 &&
                   status.speakers_route_mode == "loopback" &&
                   status.speakers_routing_active;
          },
          250ms)) {
    std::cerr << "speaker loopback did not become active before disable\n";
    service.Stop();
    return false;
  }

  cfg.speakers_enabled = false;
  cfg.create_virtual_speakers = false;
  service.UpdateConfig(cfg);

  const bool stop_failed = WaitUntil(
      [&] {
        const auto status = service.Status();
        return loopback_stop_calls.load(std::memory_order_relaxed) >= 1 &&
               status.speakers_route_mode == "loopback" &&
               status.speakers_routing_active && status.speakers_present &&
               status.speakers_last_error.find("synthetic loopback stop "
                                               "failure") != std::string::npos;
      },
      250ms);
  const int stops_after_failure =
      loopback_stop_calls.load(std::memory_order_relaxed);
  std::this_thread::sleep_for(30ms);
  const int stops_during_backoff =
      loopback_stop_calls.load(std::memory_order_relaxed);
  const int destroys_after_stop_failure =
      destroy_calls.load(std::memory_order_relaxed);
  const auto status = service.Status();
  service.Stop();

  if (!stop_failed) {
    std::cerr
        << "speaker disable did not preserve failed loopback route; stops="
        << loopback_stop_calls.load() << " route='"
        << status.speakers_route_mode
        << "' active=" << status.speakers_routing_active
        << " present=" << status.speakers_present << " error='"
        << status.speakers_last_error << "'\n";
    return false;
  }

  if (stops_during_backoff != stops_after_failure) {
    std::cerr << "speaker loopback stop during disable retried before backoff "
                 "elapsed; after_failure="
              << stops_after_failure
              << " during_backoff=" << stops_during_backoff << "\n";
    return false;
  }

  if (destroys_after_stop_failure != 0) {
    std::cerr << "virtual speakers were destroyed while loopback stop failed; "
              << "destroys=" << destroys_after_stop_failure << "\n";
    return false;
  }

  return true;
}

bool TestStopInterruptsOpenAfterEarlyStopReset() {
  auto state = std::make_shared<ResettingOpenIoState>();
  CopyProcessor processor;

  AudioPipelineHooks hooks;
  hooks.create_io = [state] {
    return std::make_unique<ResettingOpenIo>(state, 25ms, 200ms);
  };

  AudioPipeline pipeline(&processor, std::move(hooks));
  AudioPipelineConfig cfg;

  std::string err;
  auto start_future =
      std::async(std::launch::async, [&] { return pipeline.Start(cfg, &err); });

  if (!WaitUntil(
          [&] {
            std::lock_guard<std::mutex> lock(state->mu);
            return state->open_started;
          },
          250ms)) {
    std::cerr << "pipeline.Start did not enter Open()\n";
    pipeline.Stop();
    (void)start_future.get();
    return false;
  }

  auto stop_future = std::async(std::launch::async, [&] {
    pipeline.Stop();
    return true;
  });
  const bool stop_ready =
      (stop_future.wait_for(100ms) == std::future_status::ready);
  (void)stop_future.get();
  const bool start_ready =
      (start_future.wait_for(100ms) == std::future_status::ready);
  const bool start_ok = start_ready ? start_future.get() : true;

  int request_stop_calls = 0;
  {
    std::lock_guard<std::mutex> lock(state->mu);
    request_stop_calls = state->request_stop_calls;
  }

  if (!stop_ready) {
    std::cerr << "Stop() stayed blocked while Open() reset an early stop\n";
    return false;
  }

  if (!start_ready || start_ok) {
    std::cerr
        << "Start() did not return false after Stop() interrupted Open(); "
        << "ready=" << start_ready << " ok=" << start_ok << " err='" << err
        << "'\n";
    return false;
  }

  if (request_stop_calls == 0) {
    std::cerr << "Stop() never signaled Open() to stop\n";
    return false;
  }

  return true;
}

bool TestStopInterruptsBlockedFlush() {
  auto state = std::make_shared<BlockingFlushIoState>();
  CopyProcessor processor;

  AudioPipelineHooks hooks;
  hooks.create_io = [state] {
    return std::make_unique<BlockingFlushIo>(state, 200ms);
  };

  AudioPipeline pipeline(&processor, std::move(hooks));
  AudioPipelineConfig cfg;

  std::string err;
  if (!pipeline.Start(cfg, &err)) {
    std::cerr << "pipeline.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] {
            std::lock_guard<std::mutex> lock(state->mu);
            return state->flush_entered;
          },
          250ms)) {
    std::cerr << "pipeline did not enter blocked flush\n";
    pipeline.Stop();
    return false;
  }

  auto stop_future = std::async(std::launch::async, [&] {
    pipeline.Stop();
    return true;
  });
  const bool stop_ready =
      (stop_future.wait_for(50ms) == std::future_status::ready);
  (void)stop_future.get();

  int request_stop_calls = 0;
  {
    std::lock_guard<std::mutex> lock(state->mu);
    request_stop_calls = state->request_stop_calls;
  }

  if (!stop_ready) {
    std::cerr << "Stop() stayed blocked while Flush() was stuck\n";
    return false;
  }

  if (request_stop_calls == 0) {
    std::cerr << "Stop() never signaled the flush transport to stop\n";
    return false;
  }

  return true;
}

bool TestStartReturnsOpenFailureAndCanRetry() {
  std::atomic<int> open_calls{0};
  CopyProcessor processor;

  AudioPipelineHooks hooks;
  hooks.create_io = [&] { return std::make_unique<OpenFailIo>(&open_calls); };

  AudioPipeline pipeline(&processor, std::move(hooks));
  AudioPipelineConfig cfg;

  std::string err;
  if (pipeline.Start(cfg, &err)) {
    std::cerr << "first pipeline.Start succeeded despite open failure\n";
    return false;
  }
  if (err.find("synthetic open failure") == std::string::npos) {
    std::cerr << "first Start() did not return open error; err='" << err
              << "'\n";
    return false;
  }

  if (open_calls.load(std::memory_order_relaxed) != 1 ||
      pipeline.GetStats().running) {
    std::cerr << "first failed start left pipeline running; opens="
              << open_calls.load() << " running=" << pipeline.GetStats().running
              << "\n";
    return false;
  }

  err.clear();
  if (pipeline.Start(cfg, &err)) {
    std::cerr << "second pipeline.Start succeeded despite open failure\n";
    return false;
  }
  if (err.find("synthetic open failure") == std::string::npos) {
    std::cerr << "second Start() did not return open error; err='" << err
              << "'\n";
    return false;
  }

  if (open_calls.load(std::memory_order_relaxed) != 2 ||
      pipeline.GetStats().running) {
    std::cerr << "second failed start left pipeline running; opens="
              << open_calls.load() << " running=" << pipeline.GetStats().running
              << "\n";
    return false;
  }

  pipeline.Stop();
  return true;
}

bool TestPipelineSurfacesCaptureDisconnectError() {
  std::atomic<int> read_calls{0};
  CopyProcessor processor;

  AudioPipelineHooks hooks;
  hooks.create_io = [&] { return std::make_unique<ReadFailIo>(&read_calls); };

  AudioPipeline pipeline(&processor, std::move(hooks));
  AudioPipelineConfig cfg;

  std::string err;
  if (!pipeline.Start(cfg, &err)) {
    std::cerr << "pipeline.Start failed: " << err << "\n";
    return false;
  }

  const bool saw_disconnect = WaitUntil(
      [&] {
        const auto stats = pipeline.GetStats();
        return read_calls.load(std::memory_order_relaxed) > 0 &&
               !stats.running &&
               stats.last_error.find("capture stream disconnected") !=
                   std::string::npos;
      },
      250ms);
  const auto stats = pipeline.GetStats();
  pipeline.Stop();

  if (!saw_disconnect) {
    std::cerr << "expected capture disconnect error; reads="
              << read_calls.load() << " running=" << stats.running << " error='"
              << stats.last_error << "'\n";
    return false;
  }

  return true;
}

bool TestLatencyGuardSumsCaptureAndPlaybackBeforeResync() {
  auto state = std::make_shared<LatencyIoState>();
  std::atomic<int> reset_calls{0};
  CountingCopyProcessor processor(&reset_calls);

  AudioPipelineHooks hooks;
  hooks.create_io = [state] { return std::make_unique<HighLatencyIo>(state); };

  AudioPipeline pipeline(&processor, std::move(hooks));
  AudioPipelineConfig cfg;

  std::string err;
  if (!pipeline.Start(cfg, &err)) {
    std::cerr << "pipeline.Start failed: " << err << "\n";
    return false;
  }

  const bool resynced =
      WaitUntil([&] { return pipeline.GetStats().resync_events >= 1; }, 1700ms);
  const auto stats = pipeline.GetStats();
  pipeline.Stop();

  if (!resynced) {
    std::cerr << "expected latency guard to resync when capture+playback "
                 "latency exceeded threshold; max_us="
              << stats.pulse_latency_us_max
              << " capture_queries=" << state->capture_latency_queries.load()
              << " playback_queries=" << state->playback_latency_queries.load()
              << "\n";
    return false;
  }

  if (stats.pulse_capture_latency_us_last != 80000 ||
      stats.pulse_playback_latency_us_last != 80000 ||
      stats.pulse_latency_us_max < 160000) {
    std::cerr << "latency stats did not account for capture+playback sum; "
              << "capture=" << stats.pulse_capture_latency_us_last
              << " playback=" << stats.pulse_playback_latency_us_last
              << " max=" << stats.pulse_latency_us_max << "\n";
    return false;
  }

  if (reset_calls.load(std::memory_order_relaxed) < 2 ||
      state->flush_calls.load(std::memory_order_relaxed) < 2) {
    std::cerr << "resync did not flush/reset pipeline state; resets="
              << reset_calls.load() << " flushes=" << state->flush_calls.load()
              << "\n";
    return false;
  }

  return true;
}

bool TestLatencyQueryFailureClearsStaleLastValue() {
  auto state = std::make_shared<FlakyLatencyIoState>();
  CopyProcessor processor;

  AudioPipelineHooks hooks;
  hooks.create_io = [state] { return std::make_unique<FlakyLatencyIo>(state); };

  AudioPipeline pipeline(&processor, std::move(hooks));
  AudioPipelineConfig cfg;

  std::string err;
  if (!pipeline.Start(cfg, &err)) {
    std::cerr << "pipeline.Start failed: " << err << "\n";
    return false;
  }

  const bool saw_initial_latency = WaitUntil(
      [&] {
        const auto stats = pipeline.GetStats();
        return stats.pulse_capture_latency_us_last == 12000 &&
               stats.pulse_playback_latency_us_last == 34000 &&
               stats.pulse_latency_us_max >= 46000;
      },
      1300ms);

  const bool cleared_failed_side = WaitUntil(
      [&] {
        const auto stats = pipeline.GetStats();
        return state->capture_latency_queries.load(std::memory_order_relaxed) >=
                   2 &&
               state->playback_latency_queries.load(
                   std::memory_order_relaxed) >= 2 &&
               stats.pulse_capture_latency_us_last == 0 &&
               stats.pulse_playback_latency_us_last == 5000 &&
               stats.pulse_latency_us_max >= 46000;
      },
      1300ms);
  const auto stats = pipeline.GetStats();
  pipeline.Stop();

  if (!saw_initial_latency || !cleared_failed_side) {
    std::cerr << "latency query failure left stale status; initial="
              << saw_initial_latency << " cleared=" << cleared_failed_side
              << " capture_last=" << stats.pulse_capture_latency_us_last
              << " playback_last=" << stats.pulse_playback_latency_us_last
              << " max=" << stats.pulse_latency_us_max
              << " capture_queries=" << state->capture_latency_queries.load()
              << " playback_queries=" << state->playback_latency_queries.load()
              << "\n";
    return false;
  }

  return true;
}

bool TestOfflinePassthroughPipelineAudioQuality() {
  constexpr std::uint32_t kFrameSamples = 480;
  constexpr int kFrameCount = 8;
  constexpr std::uint32_t kChannels = 2;

  auto state = std::make_shared<ScriptedQualityIoState>();
  const std::vector<float> input =
      MakeSyntheticStereoAudio(kFrameSamples, kFrameCount);
  {
    std::lock_guard<std::mutex> lock(state->mu);
    state->input = input;
  }

  studiocast::audio::PassthroughAudioProcessor processor;
  AudioPipelineHooks hooks;
  hooks.create_io = [state] {
    return std::make_unique<ScriptedQualityIo>(state);
  };

  AudioPipeline pipeline(&processor, std::move(hooks));
  AudioPipelineConfig cfg;
  cfg.sample_rate = 48000;
  cfg.frame_samples = kFrameSamples;
  cfg.channels = kChannels;

  std::string err;
  if (!pipeline.Start(cfg, &err)) {
    std::cerr << "pipeline.Start failed: " << err << "\n";
    return false;
  }

  const bool finished = WaitUntil(
      [&] {
        const auto stats = pipeline.GetStats();
        return stats.frames_processed ==
                   static_cast<std::uint64_t>(kFrameCount) &&
               !stats.running;
      },
      500ms);
  const auto stats = pipeline.GetStats();
  pipeline.Stop();

  std::vector<float> output;
  std::uint32_t open_sample_rate = 0;
  std::uint32_t open_channels = 0;
  int read_calls = 0;
  int write_calls = 0;
  int flush_calls = 0;
  {
    std::lock_guard<std::mutex> lock(state->mu);
    output = state->output;
    open_sample_rate = state->open_sample_rate;
    open_channels = state->open_channels;
    read_calls = state->read_calls;
    write_calls = state->write_calls;
    flush_calls = state->flush_calls;
  }

  if (!finished) {
    std::cerr << "offline passthrough pipeline did not finish expected "
                 "buffers; frames="
              << stats.frames_processed << " running=" << stats.running
              << " error='" << stats.last_error << "'\n";
    return false;
  }

  if (open_sample_rate != 48000 || open_channels != kChannels ||
      read_calls != kFrameCount || write_calls != kFrameCount ||
      flush_calls < 2) {
    std::cerr << "offline pipeline format/buffer behavior changed; rate="
              << open_sample_rate << " channels=" << open_channels
              << " reads=" << read_calls << " writes=" << write_calls
              << " flushes=" << flush_calls << "\n";
    return false;
  }

  if (output.size() != input.size()) {
    std::cerr << "offline pipeline changed sample count; input=" << input.size()
              << " output=" << output.size() << "\n";
    return false;
  }

  for (std::size_t i = 0; i < output.size(); ++i) {
    if (!std::isfinite(output[i]) || std::fabs(output[i]) > 1.000001f) {
      std::cerr << "offline pipeline produced invalid sample at " << i << ": "
                << output[i] << "\n";
      return false;
    }
  }

  const std::size_t silence_samples =
      static_cast<std::size_t>(kFrameSamples) * kChannels;
  for (std::size_t i = 0; i < silence_samples; ++i) {
    if (output[i] != 0.0f) {
      std::cerr << "offline pipeline did not preserve leading silence at " << i
                << ": " << output[i] << "\n";
      return false;
    }
  }

  if (output != input) {
    double max_abs_diff = 0.0;
    for (std::size_t i = 0; i < output.size(); ++i) {
      max_abs_diff =
          std::max(max_abs_diff, std::fabs(static_cast<double>(output[i]) -
                                           static_cast<double>(input[i])));
    }
    std::cerr << "passthrough pipeline changed samples; max_abs_diff="
              << max_abs_diff << "\n";
    return false;
  }

  const double input_rms = Rms(input);
  const double output_rms = Rms(output);
  if (std::fabs(input_rms - output_rms) > 1e-9) {
    std::cerr << "passthrough RMS changed; input=" << input_rms
              << " output=" << output_rms << "\n";
    return false;
  }

  return true;
}

bool TestStopInterruptsBlockedCaptureRead() {
  auto state = std::make_shared<BlockingIoState>();
  CopyProcessor processor;

  AudioPipelineHooks hooks;
  hooks.create_io = [state] {
    return std::make_unique<BlockingIo>(state, BlockMode::kRead, 200ms);
  };

  AudioPipeline pipeline(&processor, std::move(hooks));
  AudioPipelineConfig cfg;

  std::string err;
  if (!pipeline.Start(cfg, &err)) {
    std::cerr << "pipeline.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] {
            std::lock_guard<std::mutex> lock(state->mu);
            return state->block_entered;
          },
          250ms)) {
    std::cerr << "pipeline did not enter blocked capture read\n";
    pipeline.Stop();
    return false;
  }

  auto stop_future = std::async(std::launch::async, [&] {
    pipeline.Stop();
    return true;
  });
  const bool stop_ready =
      (stop_future.wait_for(50ms) == std::future_status::ready);
  (void)stop_future.get();

  int request_stop_calls = 0;
  {
    std::lock_guard<std::mutex> lock(state->mu);
    request_stop_calls = state->request_stop_calls;
  }

  if (!stop_ready) {
    std::cerr << "Stop() stayed blocked while capture read was stuck\n";
    return false;
  }

  if (request_stop_calls == 0) {
    std::cerr << "Stop() never signaled the capture transport to stop\n";
    return false;
  }

  return true;
}

bool TestStopInterruptsBlockedPlaybackWrite() {
  auto state = std::make_shared<BlockingIoState>();
  CopyProcessor processor;

  AudioPipelineHooks hooks;
  hooks.create_io = [state] {
    return std::make_unique<BlockingIo>(state, BlockMode::kWrite, 200ms);
  };

  AudioPipeline pipeline(&processor, std::move(hooks));
  AudioPipelineConfig cfg;

  std::string err;
  if (!pipeline.Start(cfg, &err)) {
    std::cerr << "pipeline.Start failed: " << err << "\n";
    return false;
  }

  if (!WaitUntil(
          [&] {
            std::lock_guard<std::mutex> lock(state->mu);
            return state->block_entered;
          },
          250ms)) {
    std::cerr << "pipeline did not enter blocked playback write\n";
    pipeline.Stop();
    return false;
  }

  auto stop_future = std::async(std::launch::async, [&] {
    pipeline.Stop();
    return true;
  });
  const bool stop_ready =
      (stop_future.wait_for(50ms) == std::future_status::ready);
  (void)stop_future.get();

  int request_stop_calls = 0;
  {
    std::lock_guard<std::mutex> lock(state->mu);
    request_stop_calls = state->request_stop_calls;
  }

  if (!stop_ready) {
    std::cerr << "Stop() stayed blocked while playback write was stuck\n";
    return false;
  }

  if (request_stop_calls == 0) {
    std::cerr << "Stop() never signaled the playback transport to stop\n";
    return false;
  }

  return true;
}

} // namespace

int main() {
  const struct {
    const char *name;
    bool (*fn)();
    bool mock_safe_mic_source = false;
  } tests[] = {
      {"pactl load-module quotes vector arguments",
       &TestPactlLoadModuleQuotesVectorArguments},
      {"pactl load-module string compatibility splitter",
       &TestPactlLoadModuleStringCompatibilitySplitter},
      {"pactl default source/sink fallback to info",
       &TestPactlDefaultSourceAndSinkFallbackToInfo},
      {"pactl proplist commands quote args and detect failures",
       &TestPactlProplistCommandsQuoteArgumentsAndDetectFailures},
      {"pactl consumer lists parse source outputs and sink inputs",
       &TestPactlConsumerListsParseSourceOutputsAndSinkInputs},
      {"audio source safety rejects virtual and monitor sources",
       &TestAudioSourceSafetyRejectsVirtualAndMonitorSources},
      {"audio source auto falls back from unsafe default source",
       &TestAudioSourceAutoFallsBackFromUnsafeDefaultSource},
      {"audio source auto fails when no safe source exists",
       &TestAudioSourceAutoFailsWhenNoSafeSourceExists},
      {"virtual audio service reports resolved auto source and warnings",
       &TestVirtualAudioServiceReportsResolvedAutoSourceAndWarnings},
      {"virtual audio service preserves unavailable configured source",
       &TestVirtualAudioServicePreservesUnavailableConfiguredSource},
      {"speaker target safety rejects virtual and monitor endpoints",
       &TestSpeakerTargetSafetyRejectsVirtualAndMonitorEndpoints},
      {"speaker target auto falls back from unsafe default sink",
       &TestSpeakerTargetAutoFallsBackFromUnsafeDefaultSink},
      {"status text surfaces module list failures",
       &TestStatusTextSurfacesModuleListFailure},
      {"virtual mic create propagates list failure without loading",
       &TestCreateVirtualMicPropagatesListFailureWithoutLoading},
      {"virtual speaker create propagates list failure without loading",
       &TestCreateVirtualSpeakerPropagatesListFailureWithoutLoading},
      {"virtual speaker loopback falls back from virtual default sink",
       &TestVirtualSpeakerLoopbackFallsBackFromVirtualDefaultSink},
      {"virtual speaker loopback rejects virtual target",
       &TestVirtualSpeakerLoopbackRejectsVirtualTarget},
      {"virtual speaker loopback rejects virtual target before stop",
       &TestVirtualSpeakerLoopbackRejectsVirtualTargetBeforeStop},
      {"virtual speaker loopback restart propagates stop failure",
       &TestVirtualSpeakerLoopbackRestartPropagatesStopFailure},
      {"destroy virtual speaker propagates null sink unload failure",
       &TestDestroyVirtualSpeakerPropagatesNullSinkUnloadFailure},
      {"virtual mic loopback stop propagates unload failure",
       &TestVirtualMicStopLoopbackPropagatesUnloadFailure},
      {"destroy virtual mic preserves remaining state on null unload failure",
       &TestDestroyVirtualMicPreservesRemainingStateOnNullUnloadFailure},
      {"mic pipeline does not start without consumer",
       &TestMicrophonePipelineDoesNotStartWithoutConsumer, true},
      {"mic pipeline starts when consumer appears",
       &TestMicrophonePipelineStartsWhenConsumerAppears, true},
      {"mic pipeline stops when consumer disappears",
       &TestMicrophonePipelineStopsWhenConsumerDisappears, true},
      {"mic grace window absorbs consumer flapping",
       &TestMicrophoneGraceWindowAbsorbsConsumerFlapping, true},
      {"mic consumer detection recovers after errors",
       &TestMicrophoneConsumerDetectionRecoversAfterErrors, true},
      {"speaker pipeline follows consumer gate",
       &TestSpeakerPipelineFollowsConsumerGate},
      {"speaker grace window absorbs consumer flapping",
       &TestSpeakerGraceWindowAbsorbsConsumerFlapping},
      {"speaker loopback pass-through status is not consumer-gated",
       &TestSpeakerLoopbackPassThroughStatusIsNotConsumerGated},
      {"mic pipeline restarts after worker death",
       &TestMicrophonePipelineRestartsWhenWorkerDies, true},
      {"mic pipeline preserves worker death error",
       &TestMicrophonePipelinePreservesWorkerDeathError, true},
      {"status remains responsive during retry sleep",
       &TestStatusDoesNotBlockDuringRetrySleep, true},
      {"mic null pipeline factory fails without crash",
       &TestMicrophoneNullPipelineFactoryFailsWithoutCrash, true},
      {"speaker pipeline start failure clears route state",
       &TestSpeakerPipelineStartFailureClearsRouteState},
      {"open audio failure cooldown avoids restart churn",
       &TestOpenAudioFailureCooldownAvoidsRestartChurn, true},
      {"forced Maxine mic failure falls back to pass-through",
       &TestForcedMaxineMicrophoneFailureFallsBackToPassthrough, true},
      {"mic availability cache ignores speaker-only changes",
       &TestMicrophoneAvailabilityCacheIgnoresSpeakerOnlyChanges, true},
      {"speaker availability cache ignores microphone-only changes",
       &TestSpeakerAvailabilityCacheIgnoresMicrophoneOnlyChanges},
      {"stable audio preparation uses explicit invalidation",
       &TestStableAudioPreparationUsesExplicitInvalidation, true},
      {"forced Open Audio skips Maxine preparation",
       &TestForcedOpenAudioSkipsMaxinePreparation, true},
      {"stop interrupts blocked preparation hook",
       &TestStopInterruptsBlockedPreparationHook},
      {"exec capture cancellation bounds provider helper",
       &TestExecCaptureCancellationBoundsProviderHelper},
      {"mic dead worker backs off before restart",
       &TestMicrophoneDeadWorkerBacksOffBeforeRestart, true},
      {"speaker dead worker backs off and clears route",
       &TestSpeakerDeadWorkerBacksOffAndClearsRoute},
      {"speaker pipeline stats clear when processing disabled",
       &TestSpeakerPipelineStatsClearWhenProcessingDisabled},
      {"speaker loopback restart failure clears route state",
       &TestSpeakerLoopbackRestartFailureClearsRouteState},
      {"speaker real helper stop failure keeps old route active",
       &TestSpeakerLoopbackRealHelperStopFailureKeepsOldRouteActive},
      {"virtual speaker destroy failure backs off and keeps present",
       &TestVirtualSpeakerDestroyFailureBacksOffAndKeepsPresent},
      {"speaker loopback stop failure blocks pipeline start",
       &TestSpeakerLoopbackStopFailureBlocksPipelineStart},
      {"speaker loopback stop failure prevents destroy and keeps route",
       &TestSpeakerLoopbackStopFailurePreventsDestroyAndKeepsRoute},
      {"stop interrupts open after early stop reset",
       &TestStopInterruptsOpenAfterEarlyStopReset},
      {"stop interrupts blocked flush", &TestStopInterruptsBlockedFlush},
      {"start returns open failure and can retry",
       &TestStartReturnsOpenFailureAndCanRetry},
      {"pipeline surfaces capture disconnect",
       &TestPipelineSurfacesCaptureDisconnectError},
      {"latency guard sums capture and playback before resync",
       &TestLatencyGuardSumsCaptureAndPlaybackBeforeResync},
      {"latency query failure clears stale last value",
       &TestLatencyQueryFailureClearsStaleLastValue},
      {"offline passthrough pipeline audio quality",
       &TestOfflinePassthroughPipelineAudioQuality},
      {"stop interrupts blocked capture read",
       &TestStopInterruptsBlockedCaptureRead},
      {"stop interrupts blocked playback write",
       &TestStopInterruptsBlockedPlaybackWrite},
  };

  int failed = 0;
  for (const auto &test : tests) {
    std::optional<ScopedPactlExecHook> safe_mic_source;
    if (test.mock_safe_mic_source)
      safe_mic_source.emplace(SafeMicrophoneSourcePactlHook());
    const bool ok = test.fn();
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << test.name << "\n";
    if (!ok)
      ++failed;
  }

  return failed == 0 ? 0 : 1;
}
