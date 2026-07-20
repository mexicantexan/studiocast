#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "core/audio/audio_backend_resolver.h"
#include "core/audio/effects/broadcast_audio_effects.h"

namespace studiocast::audio {

struct VirtualAudioServiceConfig {
  // If false, the service will not run the real-time processing pipeline.
  // The virtual devices may still be created (see `create_virtual_mic`).
  bool enabled = false;

  // Keep the virtual microphone device available even when processing is
  // disabled. This is the preferred daemon-owned behavior for MVP.
  bool create_virtual_mic = true;

  // Keep the virtual speakers device available even when microphone processing
  // is disabled. When enabled, StudioCast creates a virtual sink named
  // "studiocast_speakers".
  bool create_virtual_speakers = false;

  // When true, StudioCast routes the virtual speakers monitor stream into a
  // physical sink. Phase 9 uses a Pulse module-loopback pass-through route (no
  // ML processing yet).
  bool speakers_enabled = false;

  // Target sink name for speakers routing. Empty = Pulse default sink.
  // The routing helper will refuse to loop back into StudioCast virtual sinks
  // to avoid feedback.
  std::string speaker_target_sink;

  // Latency (ms) for module-loopback when speakers_enabled is true.
  int speaker_latency_ms = 10;

  // Selected input source (Pulse source name). Empty = Pulse default source.
  std::string source_name;

  // Canonical Broadcast-style audio effect settings.
  studiocast::audio::effects::BroadcastAudioEffects effects{};

  // Supervisor cadence.
  int poll_ms = 250;

  // When a start attempt fails, wait this long before retrying.
  int start_retry_ms = 2000;

  // After the last consumer disappears, keep an already-running pipeline alive
  // briefly to avoid churn from apps that probe or reconnect audio streams.
  int consumer_grace_ms = 1000;
};

struct AudioConsumerSnapshot {
  bool present = false;
  int count = 0;
  std::string error;
};

struct OpenAudioRuntimeStatus {
  bool active = false;
  bool using_cpu_fallback = false;
  bool disabled = false;
  std::string active_provider;
  std::string selected_model_id;
  std::string selected_model_path;
  std::string last_runtime_warning;
};

struct VirtualAudioServiceStatus {
  bool service_running = false;

  bool mic_present = false;
  bool mic_consumer_present = false;
  int mic_consumer_count = 0;
  std::string mic_consumer_error;

  bool speakers_present = false;
  bool speakers_consumer_present = false;
  int speakers_consumer_count = 0;
  std::string speakers_consumer_error;
  bool speakers_routing_active = false;
  // "off" (no routing), "loopback" (Pulse module-loopback pass-through),
  // or "pipeline" (daemon processed pipeline).
  std::string speakers_route_mode;
  std::string speaker_target_sink_active;
  std::string speakers_last_error;

  // When speakers_route_mode == "pipeline", these describe the processed
  // pipeline state.
  bool speakers_pipeline_running = false;
  bool speakers_pipeline_starting = false;
  bool speakers_pipeline_active_needed = false;
  std::string speakers_pipeline_state;
  std::string speakers_pipeline_idle_reason;
  std::string speakers_backend_active;
  std::string speakers_effects_note;
  float speakers_intensity = 0.0f;
  std::string speakers_pipeline_last_error;
  OpenAudioRuntimeStatus speakers_open_audio_runtime;

  // Speaker pipeline performance (best-effort realtime stats).
  std::uint64_t speakers_pipeline_frames_processed = 0;
  std::uint64_t speakers_pipeline_process_time_us_sum = 0;
  std::uint64_t speakers_pipeline_process_time_us_max = 0;
  std::uint64_t speakers_pipeline_process_time_us_last = 0;
  std::uint64_t speakers_pipeline_process_overruns = 0;

  // Best-effort Pulse latency (microseconds) for speaker pipeline.
  std::uint64_t speakers_pipeline_pulse_capture_latency_us_last = 0;
  std::uint64_t speakers_pipeline_pulse_playback_latency_us_last = 0;
  std::uint64_t speakers_pipeline_pulse_latency_us_max = 0;
  std::uint64_t speakers_pipeline_resync_events = 0;

  bool pipeline_running = false;
  bool pipeline_starting = false;
  bool pipeline_active_needed = false;
  std::string pipeline_state;
  std::string pipeline_idle_reason;

  // Microphone pipeline performance (best-effort realtime stats).
  std::uint64_t pipeline_frames_processed = 0;
  std::uint64_t pipeline_process_time_us_sum = 0;
  std::uint64_t pipeline_process_time_us_max = 0;
  std::uint64_t pipeline_process_time_us_last = 0;
  std::uint64_t pipeline_process_overruns = 0;

  // Best-effort Pulse latency (microseconds) for microphone pipeline.
  std::uint64_t pipeline_pulse_capture_latency_us_last = 0;
  std::uint64_t pipeline_pulse_playback_latency_us_last = 0;
  std::uint64_t pipeline_pulse_latency_us_max = 0;
  std::uint64_t pipeline_resync_events = 0;

  // Actual microphone source selected for capture after resolving config
  // intent such as "auto". Empty means unresolved/auto.
  std::string selected_source;
  // "unknown", "available", or "unavailable".
  std::string source_availability = "unknown";
  std::string source_error;
  std::vector<std::string> source_warnings;
  std::string pipeline_sink = "studiocast_sink";

  // Active effects backend (selected by the runtime resolver).
  // Common values: "passthrough", "maxine", "open_source".
  std::string effects_backend_active;
  OpenAudioRuntimeStatus open_audio_runtime;

  // Human-friendly backend selection / fallback note.
  // Intended for GUI banners and daemon status.
  std::string effects_note;

  // What effect is currently active (derived from the Broadcast effect model).
  std::string effect_selector;
  std::string feature_id;
  float intensity = 0.0f;

  // Best-effort selected GPU summary (for actual pipeline configuration).
  int gpu_index = -1;
  std::string gpu_name;
  std::string gpu_compute_cap;

  std::string last_error;
};

class AudioPipelineRunner;
class AudioProcessor;
class AudioConsumerDetector;

struct VirtualAudioServiceHooks {
  std::function<void(std::chrono::milliseconds)> sleep_for;
  std::function<std::unique_ptr<AudioPipelineRunner>(AudioProcessor *)>
      create_pipeline;
  std::function<bool(std::string *)> create_virtual_mic;
  std::function<bool(std::string *)> create_virtual_speaker;
  std::function<bool(const std::string &, int, std::string *)>
      start_speaker_loopback;
  std::function<bool(std::string *)> stop_speaker_loopback;
  std::function<bool(std::string *)> destroy_virtual_speaker;
  std::function<AudioBackendAvailability(const VirtualAudioServiceConfig &)>
      probe_microphone_backend_availability;
  std::function<AudioBackendAvailability(const VirtualAudioServiceConfig &)>
      probe_speaker_backend_availability;
  std::function<AudioConsumerSnapshot()> detect_microphone_consumers;
  std::function<AudioConsumerSnapshot()> detect_speaker_consumers;
  // Test/measurement seam for setup-only discovery. Called immediately before
  // the named operation. A hook may wait while observing stop_requested to
  // exercise bounded shutdown without launching a real provider helper.
  std::function<void(std::string_view, const std::atomic_bool &)>
      before_preparation_probe;
};

// Minimal daemon-friendly owner of StudioCast virtual audio devices and
// processing pipelines.
//
// MVP scope:
//  - Virtual microphone is created via `pactl` modules.
//  - Real-time processing is Maxine AFX-backed (no CPU fallbacks).
//  - Virtual speakers are created via `pactl` modules.
//  - Phase 9 provides a pass-through speakers route via module-loopback (no ML
//  yet).
class VirtualAudioService final {
public:
  VirtualAudioService();
  explicit VirtualAudioService(VirtualAudioServiceHooks hooks);
  ~VirtualAudioService();

  VirtualAudioService(const VirtualAudioService &) = delete;
  VirtualAudioService &operator=(const VirtualAudioService &) = delete;

  // Starts the supervisor thread.
  bool Start(const VirtualAudioServiceConfig &cfg, std::string *error);
  void Stop();

  void UpdateConfig(const VirtualAudioServiceConfig &cfg);

  // Explicitly invalidates cached provider/model/device discovery. Normal
  // status polling never performs rediscovery; callers use this after model
  // installation/removal, settings refresh, or an external device transition.
  void RefreshPreparation();

  VirtualAudioServiceConfig Config() const;
  VirtualAudioServiceStatus Status() const;

private:
  void ThreadMain();

  void SleepFor(std::chrono::milliseconds d) const;
  std::unique_ptr<AudioPipelineRunner>
  CreatePipeline(AudioProcessor *processor) const;
  bool CreateVirtualMicDevice(std::string *error) const;
  bool CreateVirtualSpeakerDevice(std::string *error) const;
  bool StartSpeakerLoopbackRoute(const std::string &target_sink_name,
                                 int latency_ms, std::string *error) const;
  bool StopSpeakerLoopbackRoute(std::string *error) const;
  bool DestroyVirtualSpeakerDevice(std::string *error) const;
  AudioBackendAvailability ProbeMicrophoneBackendAvailability(
      const VirtualAudioServiceConfig &cfg) const;
  AudioBackendAvailability
  ProbeSpeakerBackendAvailability(const VirtualAudioServiceConfig &cfg) const;
  AudioConsumerSnapshot DetectMicrophoneConsumers() const;
  AudioConsumerSnapshot DetectSpeakerConsumers() const;
  void SetLastError(std::string msg);

  mutable std::mutex mu_;
  std::thread th_;
  std::atomic_bool stop_{false};
  std::atomic<std::uint64_t> mic_backend_generation_{1};
  std::atomic<std::uint64_t> speaker_backend_generation_{1};
  std::atomic<std::uint64_t> device_generation_{1};

  bool running_ = false;
  bool mic_created_ = false;

  bool speakers_created_ = false;
  bool speakers_loopback_running_ = false;
  std::string speakers_loopback_target_;
  int speakers_loopback_latency_ms_ = 0;

  VirtualAudioServiceHooks hooks_{};
  VirtualAudioServiceConfig cfg_{};
  VirtualAudioServiceStatus st_{};
  mutable std::unique_ptr<AudioConsumerDetector> consumer_detector_;
};

} // namespace studiocast::audio
