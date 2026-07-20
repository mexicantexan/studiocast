#include "core/audio/virtual_audio_service.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <thread>
#include <utility>

#include "core/audio/audio_backend_resolver.h"
#include "core/audio/audio_consumer_detector.h"
#include "core/audio/audio_device_safety.h"
#include "core/audio/audio_pipeline.h"
#include "core/audio/audio_processor.h"
#include "core/audio/pulse/pactl.h"
#include "core/audio/virtual_mic.h"
#include "core/audio/virtual_speaker.h"
#include "core/config/settings.h"
#include "core/maxine/availability.h"
#include "core/maxine/gpu_selection.h"
#include "core/maxine/paths.h"
#include "core/open_audio/open_audio_audio_processor.h"

// Effect planning is build-time independent from the Pulse audio pipeline.
#include "core/maxine/afx/afx_effect.h"

#if STUDIOCAST_HAVE_PULSE_SIMPLE
#include "core/maxine/afx/afx_audio_processor.h"
#include "core/maxine/afx/afx_stereo_audio_processor.h"
#include "core/maxine/afx_api.h"
#endif

namespace studiocast::audio {

namespace {

constexpr const char *kVirtualMicSourceName = "studiocast_mic";
constexpr const char *kVirtualSpeakersSinkName = "studiocast_speakers";

using AudioEffectsEnginePreference =
    studiocast::audio::effects::AudioEffectsEnginePreference;
using BroadcastAudioEffects = studiocast::audio::effects::BroadcastAudioEffects;
using BroadcastMicrophoneEffects =
    studiocast::audio::effects::BroadcastMicrophoneEffects;
using BroadcastSpeakerEffects =
    studiocast::audio::effects::BroadcastSpeakerEffects;

struct MicrophoneAvailabilityCacheKey {
  int schema_version = 0;
  AudioEffectsEnginePreference engine = AudioEffectsEnginePreference::kAuto;
  BroadcastMicrophoneEffects microphone{};
};

struct SpeakerAvailabilityCacheKey {
  int schema_version = 0;
  AudioEffectsEnginePreference engine = AudioEffectsEnginePreference::kAuto;
  BroadcastSpeakerEffects speaker{};
};

struct MicrophoneSourceStatus {
  std::string selected_source;
  std::string availability = "unknown";
  std::string error;
  std::vector<std::string> warnings;
};

struct CachedMicrophoneSourceStatus {
  std::string configured_source;
  bool resolve_auto = false;
  MicrophoneSourceStatus status;
};

bool operator==(const MicrophoneAvailabilityCacheKey &a,
                const MicrophoneAvailabilityCacheKey &b) {
  return a.schema_version == b.schema_version && a.engine == b.engine &&
         a.microphone == b.microphone;
}

bool operator!=(const MicrophoneAvailabilityCacheKey &a,
                const MicrophoneAvailabilityCacheKey &b) {
  return !(a == b);
}

bool operator==(const SpeakerAvailabilityCacheKey &a,
                const SpeakerAvailabilityCacheKey &b) {
  return a.schema_version == b.schema_version && a.engine == b.engine &&
         a.speaker == b.speaker;
}

bool operator!=(const SpeakerAvailabilityCacheKey &a,
                const SpeakerAvailabilityCacheKey &b) {
  return !(a == b);
}

MicrophoneAvailabilityCacheKey
MakeMicrophoneAvailabilityCacheKey(const BroadcastAudioEffects &fx) {
  auto mic = fx.microphone;
  mic.strength = 0;
  return MicrophoneAvailabilityCacheKey{fx.schema_version, fx.engine,
                                        std::move(mic)};
}

SpeakerAvailabilityCacheKey
MakeSpeakerAvailabilityCacheKey(const BroadcastAudioEffects &fx) {
  auto speaker = fx.speaker;
  speaker.strength = 0;
  return SpeakerAvailabilityCacheKey{fx.schema_version, fx.engine,
                                     std::move(speaker)};
}

BroadcastMicrophoneEffects
MicrophoneRestartRelevantEffects(BroadcastMicrophoneEffects mic) {
  // Strength maps to runtime intensity/aux control; it should not by itself
  // force Pulse stream teardown.
  mic.strength = 0;
  return mic;
}

BroadcastSpeakerEffects
SpeakerRestartRelevantEffects(BroadcastSpeakerEffects speaker) {
  // Open Audio speaker strength maps to an atomic runtime control. Maxine still
  // restarts on strength changes because AFX intensity updates are not proven
  // safe while the audio thread is processing.
  speaker.strength = 0;
  return speaker;
}

bool MicrophonePipelineEffectsRequireRestart(
    const BroadcastAudioEffects &oldFx, const BroadcastAudioEffects &newFx) {
  if (oldFx.schema_version != newFx.schema_version ||
      oldFx.engine != newFx.engine) {
    return true;
  }
  return MicrophoneRestartRelevantEffects(oldFx.microphone) !=
         MicrophoneRestartRelevantEffects(newFx.microphone);
}

bool SpeakerPipelineEffectsRequireRestart(
    const BroadcastSpeakerEffects &oldSpeaker,
    const BroadcastSpeakerEffects &newSpeaker) {
  return SpeakerRestartRelevantEffects(oldSpeaker) !=
         SpeakerRestartRelevantEffects(newSpeaker);
}

std::chrono::milliseconds
StartFailureRetryDelay(const VirtualAudioServiceConfig &cfg) {
  return std::chrono::milliseconds(std::max(250, cfg.start_retry_ms));
}

std::chrono::milliseconds
WorkerDeathRetryDelay(const VirtualAudioServiceConfig &cfg) {
  return std::chrono::milliseconds(std::max(25, cfg.start_retry_ms));
}

void SetOpenAudioRuntimeModel(
    OpenAudioRuntimeStatus *status,
    const studiocast::open_audio::ResolvedOpenAudioModel &model) {
  if (!status)
    return;
  status->selected_model_id = model.model_id;
  status->selected_model_path = model.onnx_path.string();
}

OpenAudioRuntimeStatus MakeActiveOpenAudioRuntimeStatus(
    const studiocast::open_audio::ResolvedOpenAudioModel &model,
    const studiocast::open_audio::OpenAudioAudioProcessor &processor) {
  OpenAudioRuntimeStatus status;
  status.active = true;
  status.active_provider = processor.ActiveProviderForStatus();
  status.using_cpu_fallback = processor.UsingCpuFallbackForStatus();
  status.last_runtime_warning = processor.LastStartupWarningForStatus();
  SetOpenAudioRuntimeModel(&status, model);
  return status;
}

void MarkOpenAudioRuntimeDisabled(OpenAudioRuntimeStatus *status,
                                  const std::string &warning) {
  if (!status)
    return;
  status->active = false;
  status->disabled = true;
  status->active_provider = "disabled";
  status->last_runtime_warning = warning;
}

std::string StripAudioProcessorWarningPrefix(std::string warning) {
  const std::string prefix = "AudioProcessor warning: ";
  if (warning.compare(0, prefix.size(), prefix) == 0)
    warning.erase(0, prefix.size());
  return warning;
}

void ApplyOpenAudioRuntimeWarning(OpenAudioRuntimeStatus *status,
                                  const std::string &pipeline_error) {
  if (!status || pipeline_error.find("Open Audio") == std::string::npos)
    return;

  const std::string warning = StripAudioProcessorWarningPrefix(pipeline_error);
  status->last_runtime_warning = warning;

  if (warning.find("switched to CPU fallback") != std::string::npos) {
    status->active = true;
    status->disabled = false;
    status->using_cpu_fallback = true;
    status->active_provider = "cpu";
  } else if (warning.find("disabled after repeated runtime failures") !=
             std::string::npos) {
    MarkOpenAudioRuntimeDisabled(status, warning);
  } else if (warning.find("initialization failed") != std::string::npos ||
             warning.find("init failed") != std::string::npos ||
             warning.find("temporarily disabled") != std::string::npos) {
    MarkOpenAudioRuntimeDisabled(status, warning);
  }
}

std::optional<std::string>
ChooseSpeakerTargetSinkName(const std::string &configured_target,
                            std::string *error,
                            const std::atomic_bool *stop_requested) {
  return ChooseSafeSpeakerTargetSinkName(configured_target, error,
                                         stop_requested);
}

bool PulseSourceListContains(const std::vector<pulse::PactlSource> &sources,
                             const std::string &name) {
  return std::any_of(sources.begin(), sources.end(),
                     [&](const auto &source) { return source.name == name; });
}

MicrophoneSourceStatus
ResolveMicrophoneSourceStatus(const VirtualAudioServiceConfig &cfg,
                              bool resolve_auto,
                              const std::atomic_bool *stop_requested) {
  MicrophoneSourceStatus out;

  std::string configured = cfg.source_name;
  if (configured == "auto")
    configured.clear();

  if (!configured.empty()) {
    out.selected_source = configured;

    std::string unsafeReason;
    if (IsUnsafeInputSourceName(configured, &unsafeReason)) {
      out.availability = "unavailable";
      out.error = unsafeReason;
      return out;
    }

    std::string listErr;
    const auto sources = pulse::ListSources(&listErr, stop_requested);
    if (!listErr.empty()) {
      out.warnings.push_back(
          "Pulse source list could not confirm configured source "
          "availability: " +
          listErr);
      out.availability = "unknown";
      return out;
    }

    if (!PulseSourceListContains(sources, configured)) {
      out.availability = "unavailable";
      out.error = "Configured Pulse source '" + configured +
                  "' is not currently available. Reconnect it or choose "
                  "another physical microphone/input source.";
      return out;
    }

    out.availability = "available";
    return out;
  }

  if (!resolve_auto)
    return out;

  const auto resolved =
      ResolveSafeInputSourceName(cfg.source_name, stop_requested);
  out.warnings = resolved.warnings;
  if (!resolved.ok) {
    out.availability = "unavailable";
    out.error = resolved.error.empty()
                    ? "Failed to resolve a safe Pulse microphone source."
                    : resolved.error;
    return out;
  }

  out.selected_source = resolved.source_name;
  out.availability = "available";
  return out;
}

void BeforePreparationProbe(const VirtualAudioServiceHooks *hooks,
                            std::string_view name,
                            const std::atomic_bool *stop_requested) {
  if (hooks && hooks->before_preparation_probe && stop_requested)
    hooks->before_preparation_probe(name, *stop_requested);
}

void FillMaxineAvailability(AudioBackendAvailability *out,
                            const VirtualAudioServiceHooks *hooks,
                            const std::atomic_bool *stop_requested) {
  if (!out)
    return;

  // Maxine availability probe (audio needs AFX).
  if (!studiocast::maxine::BackendBuilt()) {
    out->maxine_ok = false;
    out->maxine_reason = "Maxine support not enabled in this build.";
    return;
  }

  // Detailed GPU and SDK discovery is performed once by the preparation that
  // will consume it. Repeating it here would duplicate settings reads,
  // nvidia-smi execution, and SDK scans before every restart.
  (void)hooks;
  (void)stop_requested;
  out->maxine_ok = true;
  out->maxine_reason.clear();
}

AudioBackendAvailability ProbeAudioBackendAvailabilityForMicrophone(
    const VirtualAudioServiceConfig &cfg, const VirtualAudioServiceHooks *hooks,
    const std::atomic_bool *stop_requested) {
  AudioBackendAvailability out;

#if !STUDIOCAST_HAVE_ONNXRUNTIME
  (void)cfg;
#endif

  using Pref = AudioEffectsEnginePreference;
  const bool probeMaxine =
      cfg.effects.engine == Pref::kAuto || cfg.effects.engine == Pref::kMaxine;
  const bool probeOpenAudio = cfg.effects.engine == Pref::kAuto ||
                              cfg.effects.engine == Pref::kOpenSource;
  if (probeMaxine)
    FillMaxineAvailability(&out, hooks, stop_requested);
  else
    out.maxine_reason = "Maxine was not selected for this preparation.";

#if !STUDIOCAST_ENABLE_OPEN_AUDIO
  (void)probeOpenAudio;
  out.open_source_ok = false;
  out.open_source_reason = "Open Audio backend is disabled in this build.";
#elif STUDIOCAST_HAVE_ONNXRUNTIME
  if (probeOpenAudio) {
    // Model resolution is part of the single processor preparation pass.
    out.open_source_ok = true;
    out.open_source_reason.clear();
  } else {
    out.open_source_ok = false;
    out.open_source_reason =
        "Open Audio was not selected for this preparation.";
  }
#else
  out.open_source_ok = false;
  out.open_source_reason =
      "Open Audio backend unavailable: ONNX Runtime not found at build time.";
#endif

  return out;
}

AudioBackendAvailability ProbeAudioBackendAvailabilityForSpeaker(
    const VirtualAudioServiceConfig &cfg, const VirtualAudioServiceHooks *hooks,
    const std::atomic_bool *stop_requested) {
  AudioBackendAvailability out;

#if !STUDIOCAST_HAVE_ONNXRUNTIME
  (void)cfg;
#endif

  using Pref = AudioEffectsEnginePreference;
  const bool probeMaxine =
      cfg.effects.engine == Pref::kAuto || cfg.effects.engine == Pref::kMaxine;
  const bool probeOpenAudio = cfg.effects.engine == Pref::kAuto ||
                              cfg.effects.engine == Pref::kOpenSource;
  if (probeMaxine)
    FillMaxineAvailability(&out, hooks, stop_requested);
  else
    out.maxine_reason = "Maxine was not selected for this preparation.";

#if !STUDIOCAST_ENABLE_OPEN_AUDIO
  (void)probeOpenAudio;
  out.open_source_ok = false;
  out.open_source_reason = "Open Audio backend is disabled in this build.";
#elif STUDIOCAST_HAVE_ONNXRUNTIME
  if (probeOpenAudio) {
    out.open_source_ok = true;
    out.open_source_reason.clear();
  } else {
    out.open_source_ok = false;
    out.open_source_reason =
        "Open Audio was not selected for this preparation.";
  }
#else
  out.open_source_ok = false;
  out.open_source_reason =
      "Open Audio backend unavailable: ONNX Runtime not found at build time.";
#endif

  return out;
}

} // namespace

VirtualAudioService::VirtualAudioService() = default;

VirtualAudioService::VirtualAudioService(VirtualAudioServiceHooks hooks)
    : hooks_(std::move(hooks)) {}

VirtualAudioService::~VirtualAudioService() { Stop(); }

bool VirtualAudioService::Start(const VirtualAudioServiceConfig &cfg,
                                std::string *error) {
  Stop();
  {
    std::lock_guard<std::mutex> lock(mu_);
    cfg_ = cfg;
    st_ = VirtualAudioServiceStatus{};
    st_.service_running = false;
    st_.pipeline_running = false;
    st_.pipeline_starting = false;
    st_.pipeline_active_needed = false;
    st_.pipeline_state.clear();
    st_.pipeline_idle_reason.clear();
    st_.mic_consumer_present = false;
    st_.mic_consumer_count = 0;
    st_.mic_consumer_error.clear();
    st_.speakers_routing_active = false;
    st_.speakers_route_mode.clear();
    st_.speakers_pipeline_running = false;
    st_.speakers_pipeline_starting = false;
    st_.speakers_pipeline_active_needed = false;
    st_.speakers_pipeline_state.clear();
    st_.speakers_pipeline_idle_reason.clear();
    st_.speakers_consumer_present = false;
    st_.speakers_consumer_count = 0;
    st_.speakers_consumer_error.clear();
    st_.open_audio_runtime = {};
    st_.speakers_open_audio_runtime = {};
    st_.pipeline_frames_processed = 0;
    st_.pipeline_process_time_us_sum = 0;
    st_.pipeline_process_time_us_max = 0;
    st_.pipeline_process_time_us_last = 0;
    st_.pipeline_process_overruns = 0;
    st_.speakers_pipeline_frames_processed = 0;
    st_.speakers_pipeline_process_time_us_sum = 0;
    st_.speakers_pipeline_process_time_us_max = 0;
    st_.speakers_pipeline_process_time_us_last = 0;
    st_.speakers_pipeline_process_overruns = 0;
    mic_created_ = false;
    speakers_created_ = false;
    speakers_loopback_running_ = false;
    speakers_loopback_target_.clear();
    speakers_loopback_latency_ms_ = 0;
    consumer_detector_.reset();
  }

  stop_.store(false, std::memory_order_release);
  try {
    th_ = std::thread([this]() { ThreadMain(); });
  } catch (const std::exception &e) {
    if (error)
      *error = std::string("Failed to start VirtualAudioService thread: ") +
               e.what();
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mu_);
    running_ = true;
    st_.service_running = true;
  }

  return true;
}

void VirtualAudioService::Stop() {
  stop_.store(true, std::memory_order_release);
  if (th_.joinable()) {
    th_.join();
  }
  std::unique_ptr<AudioConsumerDetector> consumerDetector;
  {
    std::lock_guard<std::mutex> lock(mu_);
    running_ = false;
    st_.service_running = false;
    st_.pipeline_running = false;
    st_.pipeline_starting = false;
    st_.pipeline_active_needed = false;
    st_.pipeline_state.clear();
    st_.pipeline_idle_reason.clear();
    st_.mic_consumer_present = false;
    st_.mic_consumer_count = 0;
    st_.mic_consumer_error.clear();
    st_.speakers_routing_active = false;
    st_.speakers_route_mode.clear();
    st_.speakers_pipeline_running = false;
    st_.speakers_pipeline_starting = false;
    st_.speakers_pipeline_active_needed = false;
    st_.speakers_pipeline_state.clear();
    st_.speakers_pipeline_idle_reason.clear();
    st_.speakers_consumer_present = false;
    st_.speakers_consumer_count = 0;
    st_.speakers_consumer_error.clear();
    st_.open_audio_runtime = {};
    st_.speakers_open_audio_runtime = {};
    st_.pipeline_frames_processed = 0;
    st_.pipeline_process_time_us_sum = 0;
    st_.pipeline_process_time_us_max = 0;
    st_.pipeline_process_time_us_last = 0;
    st_.pipeline_process_overruns = 0;
    st_.pipeline_pulse_capture_latency_us_last = 0;
    st_.pipeline_pulse_playback_latency_us_last = 0;
    st_.pipeline_pulse_latency_us_max = 0;
    st_.pipeline_resync_events = 0;
    st_.speakers_pipeline_frames_processed = 0;
    st_.speakers_pipeline_process_time_us_sum = 0;
    st_.speakers_pipeline_process_time_us_max = 0;
    st_.speakers_pipeline_process_time_us_last = 0;
    st_.speakers_pipeline_process_overruns = 0;
    st_.speakers_pipeline_pulse_capture_latency_us_last = 0;
    st_.speakers_pipeline_pulse_playback_latency_us_last = 0;
    st_.speakers_pipeline_pulse_latency_us_max = 0;
    st_.speakers_pipeline_resync_events = 0;
    consumerDetector = std::move(consumer_detector_);
  }
  // The fallback monitor may need to terminate a blocked `pactl subscribe`
  // child. Keep that bounded wait outside the status/configuration mutex.
  consumerDetector.reset();
}

void VirtualAudioService::UpdateConfig(const VirtualAudioServiceConfig &cfg) {
  std::lock_guard<std::mutex> lock(mu_);
  cfg_ = cfg;
}

void VirtualAudioService::RefreshPreparation() {
  mic_backend_generation_.fetch_add(1, std::memory_order_acq_rel);
  speaker_backend_generation_.fetch_add(1, std::memory_order_acq_rel);
  device_generation_.fetch_add(1, std::memory_order_acq_rel);
}

VirtualAudioServiceConfig VirtualAudioService::Config() const {
  std::lock_guard<std::mutex> lock(mu_);
  return cfg_;
}

VirtualAudioServiceStatus VirtualAudioService::Status() const {
  std::lock_guard<std::mutex> lock(mu_);
  return st_;
}

void VirtualAudioService::SleepFor(std::chrono::milliseconds d) const {
  if (hooks_.sleep_for) {
    hooks_.sleep_for(d);
    return;
  }
  std::this_thread::sleep_for(d);
}

std::unique_ptr<AudioPipelineRunner>
VirtualAudioService::CreatePipeline(AudioProcessor *processor) const {
  if (hooks_.create_pipeline) {
    return hooks_.create_pipeline(processor);
  }
#if STUDIOCAST_HAVE_PULSE_SIMPLE
  return std::make_unique<studiocast::audio::AudioPipeline>(processor);
#else
  (void)processor;
  return nullptr;
#endif
}

bool VirtualAudioService::CreateVirtualMicDevice(std::string *error) const {
  if (hooks_.create_virtual_mic) {
    return hooks_.create_virtual_mic(error);
  }
  return studiocast::audio::CreateVirtualMic(error);
}

bool VirtualAudioService::CreateVirtualSpeakerDevice(std::string *error) const {
  if (hooks_.create_virtual_speaker) {
    return hooks_.create_virtual_speaker(error);
  }
  return studiocast::audio::CreateVirtualSpeaker(error);
}

bool VirtualAudioService::StartSpeakerLoopbackRoute(
    const std::string &target_sink_name, int latency_ms,
    std::string *error) const {
  if (hooks_.start_speaker_loopback) {
    return hooks_.start_speaker_loopback(target_sink_name, latency_ms, error);
  }
  return studiocast::audio::StartSpeakerLoopback(target_sink_name, latency_ms,
                                                 error);
}

bool VirtualAudioService::StopSpeakerLoopbackRoute(std::string *error) const {
  if (hooks_.stop_speaker_loopback) {
    return hooks_.stop_speaker_loopback(error);
  }
  return studiocast::audio::StopSpeakerLoopback(error);
}

bool VirtualAudioService::DestroyVirtualSpeakerDevice(
    std::string *error) const {
  if (hooks_.destroy_virtual_speaker) {
    return hooks_.destroy_virtual_speaker(error);
  }
  return studiocast::audio::DestroyVirtualSpeaker(error);
}

AudioBackendAvailability
VirtualAudioService::ProbeMicrophoneBackendAvailability(
    const VirtualAudioServiceConfig &cfg) const {
  if (hooks_.probe_microphone_backend_availability) {
    return hooks_.probe_microphone_backend_availability(cfg);
  }
  return ProbeAudioBackendAvailabilityForMicrophone(cfg, &hooks_, &stop_);
}

AudioBackendAvailability VirtualAudioService::ProbeSpeakerBackendAvailability(
    const VirtualAudioServiceConfig &cfg) const {
  if (hooks_.probe_speaker_backend_availability) {
    return hooks_.probe_speaker_backend_availability(cfg);
  }
  return ProbeAudioBackendAvailabilityForSpeaker(cfg, &hooks_, &stop_);
}

AudioConsumerSnapshot VirtualAudioService::DetectMicrophoneConsumers() const {
  if (hooks_.detect_microphone_consumers) {
    return hooks_.detect_microphone_consumers();
  }
  if (!consumer_detector_) {
    consumer_detector_ = CreateDefaultAudioConsumerDetector(&stop_);
  }
  return consumer_detector_->DetectSourceConsumersByName(kVirtualMicSourceName);
}

AudioConsumerSnapshot VirtualAudioService::DetectSpeakerConsumers() const {
  if (hooks_.detect_speaker_consumers) {
    return hooks_.detect_speaker_consumers();
  }
  if (!consumer_detector_) {
    consumer_detector_ = CreateDefaultAudioConsumerDetector(&stop_);
  }
  return consumer_detector_->DetectSinkConsumersByName(
      kVirtualSpeakersSinkName);
}

void VirtualAudioService::SetLastError(std::string msg) {
  std::lock_guard<std::mutex> lock(mu_);
  st_.last_error = std::move(msg);
}

void VirtualAudioService::ThreadMain() {
  using namespace std::chrono;

  steady_clock::time_point nextStartRetry{};
  steady_clock::time_point nextSpeakerStartRetry{};
  steady_clock::time_point nextSpeakerLoopbackStartRetry{};
  steady_clock::time_point nextSpeakerLoopbackStopRetry{};
  steady_clock::time_point nextSpeakerDestroyRetry{};
  std::string speakerLoopbackStopRetryError;
  std::string speakerDestroyRetryError;

  // Provider preparation failures remain sticky until a relevant config key
  // changes or RefreshPreparation() advances the generation. Wall time alone
  // must never relaunch provider discovery helpers.
  steady_clock::time_point openAudioCooldownUntil{};
  std::string openAudioCooldownReason;

  // If microphone Maxine setup fails after availability selection, keep the
  // mic route alive in pass-through without periodic rediscovery.
  steady_clock::time_point maxineCooldownUntil{};
  std::string maxineCooldownReason;

  // Separate cooldown for speaker processing, so a failure in one direction
  // doesn't permanently disable the other.
  steady_clock::time_point speakerOpenAudioCooldownUntil{};
  std::string speakerOpenAudioCooldownReason;
  steady_clock::time_point speakerMaxineCooldownUntil{};
  std::string speakerMaxineCooldownReason;

  steady_clock::time_point lastMicConsumerSeen{};
  steady_clock::time_point lastSpeakerConsumerSeen{};

  std::optional<CachedMicrophoneSourceStatus> cachedMicSourceStatus;
  std::uint64_t cachedMicSourceGeneration = 0;

  std::optional<AudioBackendAvailability> cachedMicAvailability;
  std::optional<MicrophoneAvailabilityCacheKey> cachedMicAvailabilityKey;
  std::uint64_t cachedMicBackendGeneration = 0;

  std::optional<AudioBackendAvailability> cachedSpeakerAvailability;
  std::optional<SpeakerAvailabilityCacheKey> cachedSpeakerAvailabilityKey;
  std::uint64_t cachedSpeakerBackendGeneration = 0;

  std::string cachedSpeakerSinkConfigured;
  std::optional<std::string> cachedSpeakerSink;
  std::string cachedSpeakerSinkError;
  std::uint64_t cachedSpeakerSinkGeneration = 0;

#if STUDIOCAST_HAVE_PULSE_SIMPLE
  std::unique_ptr<studiocast::maxine::afx::AfxApi> api;
  std::unique_ptr<studiocast::maxine::afx::AfxEffect> fx;
  std::unique_ptr<AudioProcessor> processor;
  std::unique_ptr<AudioPipelineRunner> pipeline;

  // Independent speaker processing pipeline (virtual speakers -> physical
  // sink).
  std::unique_ptr<studiocast::maxine::afx::AfxApi> spk_api;
  std::unique_ptr<studiocast::maxine::afx::AfxEffect> spk_fx;
  std::unique_ptr<AudioProcessor> spk_processor;
  std::unique_ptr<AudioPipelineRunner> spk_pipeline;

  std::string lastBackend;
  std::optional<studiocast::audio::effects::BroadcastAudioEffects> lastFx;
  std::string lastSource;
  std::filesystem::path lastAfxLib;
  std::uint64_t lastMicPreparedGeneration = 0;

  std::string lastSpeakerBackend;
  std::optional<studiocast::audio::effects::BroadcastSpeakerEffects>
      lastSpeakerFx;
  std::string lastSpeakerTargetSink;
  std::filesystem::path lastSpeakerAfxLib;
  std::uint64_t lastSpeakerPreparedGeneration = 0;

  bool micRestartingAfterTerminalFailure = false;
  bool speakerRestartingAfterTerminalFailure = false;
#endif

#if STUDIOCAST_HAVE_PULSE_SIMPLE
  auto clearMicPipelineStatsLocked = [this]() {
    st_.pipeline_frames_processed = 0;
    st_.pipeline_process_time_us_sum = 0;
    st_.pipeline_process_time_us_max = 0;
    st_.pipeline_process_time_us_last = 0;
    st_.pipeline_process_overruns = 0;
    st_.pipeline_pulse_capture_latency_us_last = 0;
    st_.pipeline_pulse_playback_latency_us_last = 0;
    st_.pipeline_pulse_latency_us_max = 0;
    st_.pipeline_resync_events = 0;
  };

  auto clearSpeakerPipelineStatsLocked = [this]() {
    st_.speakers_pipeline_frames_processed = 0;
    st_.speakers_pipeline_process_time_us_sum = 0;
    st_.speakers_pipeline_process_time_us_max = 0;
    st_.speakers_pipeline_process_time_us_last = 0;
    st_.speakers_pipeline_process_overruns = 0;
    st_.speakers_pipeline_pulse_capture_latency_us_last = 0;
    st_.speakers_pipeline_pulse_playback_latency_us_last = 0;
    st_.speakers_pipeline_pulse_latency_us_max = 0;
    st_.speakers_pipeline_resync_events = 0;
  };
#endif

  while (!stop_.load(std::memory_order_acquire)) {
    VirtualAudioServiceConfig cfg;
    {
      std::lock_guard<std::mutex> lock(mu_);
      cfg = cfg_;
    }

    const int pollMs = std::max(25, cfg.poll_ms);
    const auto deviceGeneration =
        device_generation_.load(std::memory_order_acquire);
    const auto micBackendGeneration =
        mic_backend_generation_.load(std::memory_order_acquire);
    const auto speakerBackendGeneration =
        speaker_backend_generation_.load(std::memory_order_acquire);
    if (cachedMicBackendGeneration != 0 &&
        cachedMicBackendGeneration != micBackendGeneration) {
      openAudioCooldownUntil = {};
      openAudioCooldownReason.clear();
      maxineCooldownUntil = {};
      maxineCooldownReason.clear();
    }
    if (cachedSpeakerBackendGeneration != 0 &&
        cachedSpeakerBackendGeneration != speakerBackendGeneration) {
      speakerOpenAudioCooldownUntil = {};
      speakerOpenAudioCooldownReason.clear();
      speakerMaxineCooldownUntil = {};
      speakerMaxineCooldownReason.clear();
    }

    const bool resolveAutoMicSource = cfg.enabled;
    const bool shouldProbeMicSource = cfg.enabled || !cfg.source_name.empty();
    MicrophoneSourceStatus micSourceStatus;
    if (shouldProbeMicSource) {
      const bool sourceStatusExpired =
          !cachedMicSourceStatus ||
          cachedMicSourceStatus->configured_source != cfg.source_name ||
          cachedMicSourceStatus->resolve_auto != resolveAutoMicSource ||
          cachedMicSourceGeneration != deviceGeneration;
      if (sourceStatusExpired) {
        BeforePreparationProbe(&hooks_, "source", &stop_);
        if (stop_.load(std::memory_order_acquire))
          break;
        cachedMicSourceStatus = CachedMicrophoneSourceStatus{
            cfg.source_name, resolveAutoMicSource,
            ResolveMicrophoneSourceStatus(cfg, resolveAutoMicSource, &stop_)};
        cachedMicSourceGeneration = deviceGeneration;
      }
      micSourceStatus = cachedMicSourceStatus->status;
    } else {
      cachedMicSourceStatus.reset();
      cachedMicSourceGeneration = 0;
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      st_.selected_source = micSourceStatus.selected_source;
      st_.source_availability = micSourceStatus.availability;
      st_.source_error = micSourceStatus.error;
      st_.source_warnings = micSourceStatus.warnings;
    }

    using Pref = studiocast::audio::effects::AudioEffectsEnginePreference;
    const bool speakerEffectsRequested = AnySpeakerEffectRequested(cfg.effects);
    const bool wantSpeakerProcessing = cfg.speakers_enabled &&
                                       speakerEffectsRequested &&
                                       (cfg.effects.engine != Pref::kOff);

#if STUDIOCAST_HAVE_PULSE_SIMPLE
    const bool wantSpeakerProcessingEffective = wantSpeakerProcessing;
#else
    // Speaker processing requires the daemon audio pipeline (libpulse-simple).
    // If unavailable, always fall back to loopback pass-through.
    const bool wantSpeakerProcessingEffective = false;
#endif

    AudioConsumerSnapshot micConsumers;
    AudioConsumerSnapshot speakerConsumers;
    bool micPipelineNeeded = false;
    bool wantSpeakerProcessingActive = false;

    // Ensure virtual devices are present (best-effort).
    //
    // Note: `enabled` controls the microphone processing pipeline; virtual
    // devices may be created and routed independently.
    bool speakerLoopbackStopBlockedProcessing = false;
    std::string speakerLoopbackStopError;
    {
      // Mic device.
      if (cfg.create_virtual_mic && !mic_created_) {
        std::string err;
        if (CreateVirtualMicDevice(&err)) {
          std::lock_guard<std::mutex> lock(mu_);
          mic_created_ = true;
          st_.mic_present = true;
        } else {
          SetLastError("Failed to create virtual mic: " + err);
        }
      }

      // Speakers device and pass-through routing.
      auto setSpeakersError = [&](std::string msg) {
        std::lock_guard<std::mutex> lock(mu_);
        st_.speakers_last_error = std::move(msg);
      };
      auto clearSpeakersError = [&]() {
        std::lock_guard<std::mutex> lock(mu_);
        st_.speakers_last_error.clear();
      };
      auto preserveActiveLoopbackFromState = [&]() -> bool {
        const auto state = studiocast::audio::LoadVirtualSpeakerState();
        if (!state.loopback_module_id)
          return false;

        speakers_loopback_running_ = true;
        if (state.loopback_target_sink_name) {
          speakers_loopback_target_ = *state.loopback_target_sink_name;
        }
        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.speakers_route_mode = "loopback";
          st_.speakers_routing_active = true;
          st_.speaker_target_sink_active =
              state.loopback_target_sink_name.value_or(
                  speakers_loopback_target_);
        }
        return true;
      };
      auto markActiveLoopbackStopFailure = [&](const std::string &err) {
        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.speakers_route_mode = "loopback";
          st_.speakers_routing_active = true;
          if (st_.speaker_target_sink_active.empty()) {
            st_.speaker_target_sink_active = speakers_loopback_target_;
          }
        }
        setSpeakersError("Failed to stop speakers routing: " + err);
      };
      auto stopSpeakerLoopbackForTransition = [&](std::string *error) -> bool {
        const auto stopNow = steady_clock::now();
        if (stopNow < nextSpeakerLoopbackStopRetry) {
          const std::string err =
              speakerLoopbackStopRetryError.empty()
                  ? "previous stop failure is still in retry backoff"
                  : speakerLoopbackStopRetryError;
          if (error)
            *error = err;
          markActiveLoopbackStopFailure(err);
          return false;
        }

        std::string err;
        if (StopSpeakerLoopbackRoute(&err)) {
          nextSpeakerLoopbackStopRetry = steady_clock::time_point{};
          speakerLoopbackStopRetryError.clear();
          speakers_loopback_running_ = false;
          speakers_loopback_target_.clear();
          speakers_loopback_latency_ms_ = 0;
          {
            std::lock_guard<std::mutex> lock(mu_);
            st_.speakers_routing_active = false;
            st_.speaker_target_sink_active.clear();
          }
          clearSpeakersError();
          if (error)
            error->clear();
          return true;
        }

        if (err.empty())
          err = "speaker loopback stop failed";
        nextSpeakerLoopbackStopRetry = stopNow + StartFailureRetryDelay(cfg);
        speakerLoopbackStopRetryError = err;
        if (error)
          *error = err;
        markActiveLoopbackStopFailure(err);
        return false;
      };

      const bool wantSpeakersDevice =
          cfg.create_virtual_speakers || cfg.speakers_enabled;

      if (wantSpeakersDevice && !speakers_created_) {
        std::string err;
        if (CreateVirtualSpeakerDevice(&err)) {
          speakers_created_ = true;
          {
            std::lock_guard<std::mutex> lock(mu_);
            st_.speakers_present = true;
          }
          clearSpeakersError();
        } else {
          setSpeakersError("Failed to create virtual speakers: " + err);
        }
      }

      const auto consumerNow = steady_clock::now();
      const auto consumerGrace =
          milliseconds(std::max(0, cfg.consumer_grace_ms));

      if (cfg.enabled) {
        micConsumers = DetectMicrophoneConsumers();
        if (micConsumers.present)
          lastMicConsumerSeen = consumerNow;
      }

      if (wantSpeakerProcessingEffective) {
        speakerConsumers = DetectSpeakerConsumers();
        if (speakerConsumers.present)
          lastSpeakerConsumerSeen = consumerNow;
      }

#if STUDIOCAST_HAVE_PULSE_SIMPLE
      const bool keepMicDuringGrace =
          !micConsumers.present && pipeline &&
          lastMicConsumerSeen != steady_clock::time_point{} &&
          consumerGrace.count() > 0 &&
          (consumerNow - lastMicConsumerSeen) < consumerGrace;
      const bool keepSpeakerDuringGrace =
          !speakerConsumers.present && spk_pipeline &&
          lastSpeakerConsumerSeen != steady_clock::time_point{} &&
          consumerGrace.count() > 0 &&
          (consumerNow - lastSpeakerConsumerSeen) < consumerGrace;
#else
      const bool keepMicDuringGrace = false;
      const bool keepSpeakerDuringGrace = false;
#endif

      micPipelineNeeded =
          cfg.enabled && (micConsumers.present || keepMicDuringGrace);
      wantSpeakerProcessingActive =
          wantSpeakerProcessingEffective &&
          (speakerConsumers.present || keepSpeakerDuringGrace);

      {
        std::lock_guard<std::mutex> lock(mu_);
        st_.mic_consumer_present = micConsumers.present;
        st_.mic_consumer_count = micConsumers.count;
        st_.mic_consumer_error = micConsumers.error;
        st_.speakers_consumer_present = speakerConsumers.present;
        st_.speakers_consumer_count = speakerConsumers.count;
        st_.speakers_consumer_error = speakerConsumers.error;
        st_.pipeline_active_needed = micPipelineNeeded;
        st_.speakers_pipeline_active_needed = wantSpeakerProcessingActive;
      }

      // Keep speakers routing state consistent with config.
      //
      // Pass-through mode uses Pulse module-loopback.
      // When speaker effects are enabled, we disable the loopback and run a
      // processed pipeline (see speaker pipeline supervisor below).
      if (cfg.speakers_enabled) {
        if (wantSpeakerProcessingEffective) {
          // Ensure loopback is stopped (avoid double-routing).
          if (speakers_loopback_running_) {
            std::string err;
            if (!stopSpeakerLoopbackForTransition(&err)) {
              speakerLoopbackStopBlockedProcessing = true;
              speakerLoopbackStopError = err;
            }
          }

          if (!speakerLoopbackStopBlockedProcessing) {
            std::lock_guard<std::mutex> lock(mu_);
            st_.speakers_route_mode = "pipeline";
          }
        } else {
#if STUDIOCAST_HAVE_PULSE_SIMPLE
          // If we are switching back to pass-through, stop the processed
          // pipeline before enabling loopback.
          if (spk_pipeline) {
            spk_pipeline->Stop();
            spk_pipeline.reset();
          }
          spk_processor.reset();
          if (spk_fx) {
            spk_fx->Destroy();
            spk_fx.reset();
          }
          spk_api.reset();
          lastSpeakerFx.reset();
          lastSpeakerBackend.clear();
          lastSpeakerTargetSink.clear();
          lastSpeakerAfxLib.clear();
          {
            std::lock_guard<std::mutex> lock(mu_);
            st_.speakers_pipeline_running = false;
            st_.speakers_pipeline_starting = false;
            st_.speakers_backend_active.clear();
            st_.speakers_effects_note.clear();
            st_.speakers_intensity = 0.0f;
            st_.speakers_pipeline_last_error.clear();
            st_.speakers_open_audio_runtime = {};
            st_.speakers_pipeline_state = "disabled";
            st_.speakers_pipeline_idle_reason =
                "Speaker processing is not requested.";
            clearSpeakerPipelineStatsLocked();
          }
#endif

          const bool needLoopbackRestart =
              (!speakers_loopback_running_) ||
              (speakers_loopback_target_ != cfg.speaker_target_sink) ||
              (speakers_loopback_latency_ms_ != cfg.speaker_latency_ms);
          const auto loopbackNow = steady_clock::now();
          if (needLoopbackRestart &&
              loopbackNow >= nextSpeakerLoopbackStartRetry) {
            std::string err;
            if (StartSpeakerLoopbackRoute(cfg.speaker_target_sink,
                                          std::max(1, cfg.speaker_latency_ms),
                                          &err)) {
              speakers_created_ = true;
              speakers_loopback_running_ = true;
              speakers_loopback_target_ = cfg.speaker_target_sink;
              speakers_loopback_latency_ms_ = cfg.speaker_latency_ms;
              nextSpeakerLoopbackStartRetry = steady_clock::time_point{};
              nextSpeakerLoopbackStopRetry = steady_clock::time_point{};
              speakerLoopbackStopRetryError.clear();

              const auto state = studiocast::audio::LoadVirtualSpeakerState();
              {
                std::lock_guard<std::mutex> lock(mu_);
                st_.speakers_present = true;
                st_.speakers_routing_active = true;
                st_.speakers_route_mode = "loopback";
                st_.speaker_target_sink_active =
                    state.loopback_target_sink_name.value_or(std::string());
              }
              clearSpeakersError();
            } else {
              nextSpeakerLoopbackStartRetry =
                  loopbackNow + StartFailureRetryDelay(cfg);
              const bool failedStoppingExisting =
                  err.find("Failed to stop existing speaker loopback") !=
                  std::string::npos;
              bool preservedActiveLoopback = false;
              if (failedStoppingExisting) {
                preservedActiveLoopback = preserveActiveLoopbackFromState();
              }
              if (!preservedActiveLoopback) {
                speakers_loopback_running_ = false;
                speakers_loopback_target_.clear();
                speakers_loopback_latency_ms_ = 0;
                {
                  std::lock_guard<std::mutex> lock(mu_);
                  st_.speakers_route_mode = "loopback";
                  st_.speakers_routing_active = false;
                  st_.speaker_target_sink_active.clear();
                }
              }
              setSpeakersError("Failed to start speakers routing: " + err);
            }
          } else {
            std::lock_guard<std::mutex> lock(mu_);
            st_.speakers_route_mode = "loopback";
          }
        }
      } else {
#if STUDIOCAST_HAVE_PULSE_SIMPLE
        // Stop processed speaker pipeline.
        if (spk_pipeline) {
          spk_pipeline->Stop();
          spk_pipeline.reset();
        }
        spk_processor.reset();
        if (spk_fx) {
          spk_fx->Destroy();
          spk_fx.reset();
        }
        spk_api.reset();
        lastSpeakerFx.reset();
        lastSpeakerBackend.clear();
        lastSpeakerTargetSink.clear();
        lastSpeakerAfxLib.clear();
        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.speakers_pipeline_running = false;
          st_.speakers_pipeline_starting = false;
          st_.speakers_backend_active.clear();
          st_.speakers_effects_note.clear();
          st_.speakers_intensity = 0.0f;
          st_.speakers_pipeline_last_error.clear();
          st_.speakers_open_audio_runtime = {};
          st_.speakers_pipeline_state = "disabled";
          st_.speakers_pipeline_idle_reason =
              "Speaker processing is not requested.";
          clearSpeakerPipelineStatsLocked();
        }
#endif

        if (speakers_loopback_running_) {
          std::string err;
          (void)stopSpeakerLoopbackForTransition(&err);
        }

        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.speakers_route_mode =
              speakers_loopback_running_ ? "loopback" : "off";
        }

        // Optional cleanup: if the user disables the device, and we previously
        // created it, destroy it. This keeps daemon-managed speaker state
        // predictable.
        if (!cfg.create_virtual_speakers && speakers_created_ &&
            !speakers_loopback_running_) {
          const auto destroyNow = steady_clock::now();
          if (destroyNow >= nextSpeakerDestroyRetry) {
            std::string err;
            if (DestroyVirtualSpeakerDevice(&err)) {
              nextSpeakerDestroyRetry = steady_clock::time_point{};
              speakerDestroyRetryError.clear();
              speakers_created_ = false;
              speakers_loopback_running_ = false;
              speakers_loopback_target_.clear();
              speakers_loopback_latency_ms_ = 0;
              {
                std::lock_guard<std::mutex> lock(mu_);
                st_.speakers_present = false;
                st_.speakers_routing_active = false;
                st_.speaker_target_sink_active.clear();
              }
              clearSpeakersError();
            } else {
              if (err.empty())
                err = "virtual speaker destroy failed";
              nextSpeakerDestroyRetry =
                  destroyNow + StartFailureRetryDelay(cfg);
              speakerDestroyRetryError = err;
              setSpeakersError("Failed to destroy virtual speakers: " + err);
            }
          } else {
            const std::string err =
                speakerDestroyRetryError.empty()
                    ? "previous destroy failure is still in retry backoff"
                    : speakerDestroyRetryError;
            setSpeakersError("Failed to destroy virtual speakers: " + err);
          }
        }
      }
    }

    (void)speakerLoopbackStopError;

#if STUDIOCAST_HAVE_PULSE_SIMPLE
    // Speaker processed pipeline supervisor.
    //
    // This runs independently from the microphone pipeline, allowing speaker
    // noise removal even when microphone effects are disabled.
    if (wantSpeakerProcessingEffective && !wantSpeakerProcessingActive) {
      if (spk_pipeline) {
        spk_pipeline->Stop();
        spk_pipeline.reset();
      }
      spk_processor.reset();
      if (spk_fx) {
        spk_fx->Destroy();
        spk_fx.reset();
      }
      spk_api.reset();
      lastSpeakerFx.reset();
      lastSpeakerBackend.clear();
      lastSpeakerTargetSink.clear();
      lastSpeakerAfxLib.clear();
      speakerRestartingAfterTerminalFailure = false;

      {
        std::lock_guard<std::mutex> lock(mu_);
        if (!speakerLoopbackStopBlockedProcessing) {
          st_.speakers_route_mode = "pipeline";
          st_.speakers_routing_active = false;
          st_.speaker_target_sink_active.clear();
        }
        st_.speakers_pipeline_running = false;
        st_.speakers_pipeline_starting = false;
        st_.speakers_backend_active.clear();
        st_.speakers_effects_note.clear();
        st_.speakers_intensity = 0.0f;
        st_.speakers_pipeline_last_error.clear();
        st_.speakers_open_audio_runtime = {};
        st_.speakers_pipeline_state = "idle_no_consumer";
        st_.speakers_pipeline_idle_reason =
            speakerConsumers.error.empty()
                ? "No active virtual speakers consumer."
                : "Virtual speakers consumer detection unavailable: " +
                      speakerConsumers.error;
        clearSpeakerPipelineStatsLocked();
      }
    } else if (wantSpeakerProcessingEffective) {
      if (speakerLoopbackStopBlockedProcessing) {
        if (spk_pipeline) {
          spk_pipeline->Stop();
          spk_pipeline.reset();
        }
        spk_processor.reset();
        if (spk_fx) {
          spk_fx->Destroy();
          spk_fx.reset();
        }
        spk_api.reset();
        lastSpeakerFx.reset();
        lastSpeakerBackend.clear();
        lastSpeakerTargetSink.clear();
        lastSpeakerAfxLib.clear();

        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.speakers_route_mode = "loopback";
          st_.speakers_pipeline_running = false;
          st_.speakers_pipeline_starting = false;
          st_.speakers_routing_active = true;
          if (st_.speaker_target_sink_active.empty()) {
            st_.speaker_target_sink_active = speakers_loopback_target_;
          }
          st_.speakers_pipeline_last_error =
              "Speaker processing blocked while stopping existing loopback: " +
              speakerLoopbackStopError;
          st_.speakers_open_audio_runtime = {};
          st_.speakers_pipeline_state = "blocked";
          st_.speakers_pipeline_idle_reason.clear();
          clearSpeakerPipelineStatsLocked();
        }
      } else if (!speakers_created_) {
        // We can't route/process speaker audio until the virtual speakers
        // device exists.
        if (spk_pipeline) {
          spk_pipeline->Stop();
          spk_pipeline.reset();
        }
        spk_processor.reset();
        if (spk_fx) {
          spk_fx->Destroy();
          spk_fx.reset();
        }

        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.speakers_routing_active = false;
          st_.speakers_route_mode = "pipeline";
          st_.speaker_target_sink_active.clear();
          st_.speakers_pipeline_running = false;
          st_.speakers_pipeline_starting = false;
          clearSpeakerPipelineStatsLocked();
          if (st_.speakers_pipeline_last_error.empty()) {
            st_.speakers_pipeline_last_error =
                "Virtual speakers device not created.";
          }
          st_.speakers_open_audio_runtime = {};
          st_.speakers_pipeline_state = "failed";
          st_.speakers_pipeline_idle_reason.clear();
        }
      } else {
        const auto now = steady_clock::now();

        // Compute the "plan" for speaker noise removal using the same AFX
        // planner.
        auto speakerPlan =
            studiocast::maxine::afx::PlanBroadcastMicrophoneEffect(
                /*studio_voice_enabled=*/false,
                /*noise_removal_enabled=*/
                cfg.effects.speaker.noise_removal_enabled,
                /*room_echo_removal_enabled=*/
                cfg.effects.speaker.room_echo_removal_enabled,
                /*strength=*/cfg.effects.speaker.strength);
        if (!speakerPlan.enabled)
          speakerPlan.intensity = 0.0f;

        // Availability + backend selection.
        const auto speakerAvailabilityKey =
            MakeSpeakerAvailabilityCacheKey(cfg.effects);
        if (cachedSpeakerAvailabilityKey.has_value() &&
            *cachedSpeakerAvailabilityKey != speakerAvailabilityKey) {
          speakerOpenAudioCooldownUntil = {};
          speakerOpenAudioCooldownReason.clear();
          speakerMaxineCooldownUntil = {};
          speakerMaxineCooldownReason.clear();
        }
        const bool speakerAvailabilityExpired =
            !cachedSpeakerAvailability ||
            !cachedSpeakerAvailabilityKey.has_value() ||
            *cachedSpeakerAvailabilityKey != speakerAvailabilityKey ||
            cachedSpeakerBackendGeneration != speakerBackendGeneration;
        if (speakerAvailabilityExpired) {
          cachedSpeakerAvailability = ProbeSpeakerBackendAvailability(cfg);
          cachedSpeakerAvailabilityKey = speakerAvailabilityKey;
          cachedSpeakerBackendGeneration = speakerBackendGeneration;
        }
        AudioBackendAvailability speakerAvail = *cachedSpeakerAvailability;
        bool speakerOpenAudioCooldownActive = false;
        if (now < speakerOpenAudioCooldownUntil) {
          speakerOpenAudioCooldownActive = true;
          speakerAvail.open_source_ok = false;
          speakerAvail.open_source_reason =
              speakerOpenAudioCooldownReason.empty()
                  ? "Open Audio backend is disabled after a preparation "
                    "failure. Refresh diagnostics to retry."
                  : speakerOpenAudioCooldownReason;
        }
        if (now < speakerMaxineCooldownUntil) {
          speakerAvail.maxine_ok = false;
          speakerAvail.maxine_reason =
              speakerMaxineCooldownReason.empty()
                  ? "Maxine backend is disabled after a preparation failure."
                  : speakerMaxineCooldownReason;
        }
        const auto speakerDecision =
            ResolveAudioBackend(cfg.effects, speakerAvail);

        const bool wantSpkMaxine =
            (speakerDecision.backend == AudioBackendKind::kMaxine) &&
            speakerPlan.enabled;
        const bool wantSpkOpenAudio =
            (speakerDecision.backend == AudioBackendKind::kOpenSource) &&
            speakerPlan.enabled;

        std::string desiredSpkBackend = "passthrough";
        if (wantSpkMaxine) {
          desiredSpkBackend = "maxine";
        } else if (wantSpkOpenAudio) {
          desiredSpkBackend = "open_source";
        }
        {
          std::lock_guard<std::mutex> lock(mu_);
          if (speakerOpenAudioCooldownActive) {
            MarkOpenAudioRuntimeDisabled(&st_.speakers_open_audio_runtime,
                                         speakerAvail.open_source_reason);
          } else if (!wantSpkOpenAudio) {
            st_.speakers_open_audio_runtime = {};
          }
        }

        // Choose target sink. If misconfigured (virtual sink), fall back to a
        // safe physical sink.
        const bool sinkPreparationChanged =
            cachedSpeakerSinkGeneration != deviceGeneration ||
            cachedSpeakerSinkConfigured != cfg.speaker_target_sink;
        if (sinkPreparationChanged) {
          BeforePreparationProbe(&hooks_, "sink", &stop_);
          if (stop_.load(std::memory_order_acquire))
            break;
          cachedSpeakerSinkConfigured = cfg.speaker_target_sink;
          cachedSpeakerSinkError.clear();
          cachedSpeakerSink = ChooseSpeakerTargetSinkName(
              cfg.speaker_target_sink, &cachedSpeakerSinkError, &stop_);
          if (!cachedSpeakerSink) {
            std::string fallbackError;
            cachedSpeakerSink = ChooseSpeakerTargetSinkName(
                /*configured_target=*/"", &fallbackError, &stop_);
            if (!cachedSpeakerSink && cachedSpeakerSinkError.empty())
              cachedSpeakerSinkError = std::move(fallbackError);
          }
          cachedSpeakerSinkGeneration = deviceGeneration;
        }
        const auto sinkOpt = cachedSpeakerSink;
        const std::string &sinkErr = cachedSpeakerSinkError;

        if (!sinkOpt) {
          // Can't route speakers anywhere.
          if (spk_pipeline) {
            spk_pipeline->Stop();
            spk_pipeline.reset();
          }
          spk_processor.reset();
          if (spk_fx) {
            spk_fx->Destroy();
            spk_fx.reset();
          }
          spk_api.reset();
          lastSpeakerFx.reset();
          lastSpeakerBackend.clear();
          lastSpeakerTargetSink.clear();
          lastSpeakerAfxLib.clear();

          {
            std::lock_guard<std::mutex> lock(mu_);
            st_.speakers_route_mode = "pipeline";
            st_.speakers_routing_active = false;
            st_.speaker_target_sink_active.clear();
            st_.speakers_pipeline_running = false;
            st_.speakers_pipeline_starting = false;
            st_.speakers_backend_active = "passthrough";
            st_.speakers_effects_note = "Speakers processing is enabled, but "
                                        "no valid output sink was found.";
            st_.speakers_intensity = speakerPlan.intensity;
            st_.speakers_pipeline_last_error = sinkErr;
            st_.speakers_open_audio_runtime = {};
            st_.speakers_pipeline_state = "failed";
            st_.speakers_pipeline_idle_reason.clear();
            clearSpeakerPipelineStatsLocked();
          }
        } else {
          const std::string sinkName = *sinkOpt;

          // Start/restart the pipeline if needed.
          const bool speakerRestartKeyChanged =
              (!lastSpeakerFx.has_value() ||
               SpeakerPipelineEffectsRequireRestart(*lastSpeakerFx,
                                                    cfg.effects.speaker));
          const bool speakerStrengthChanged =
              lastSpeakerFx.has_value() &&
              lastSpeakerFx->strength != cfg.effects.speaker.strength;
          const bool speakerEffectsChanged =
              speakerRestartKeyChanged ||
              (wantSpkMaxine && speakerStrengthChanged);
          std::optional<AudioPipelineStats> spkStatsBeforeRestart;
          if (spk_pipeline) {
            spkStatsBeforeRestart = spk_pipeline->GetStats();
          }
          const bool spkPipelineDead =
              (spkStatsBeforeRestart && !spkStatsBeforeRestart->running);
          std::string spkTerminalError;
          if (spkPipelineDead) {
            spkTerminalError = spkStatsBeforeRestart->last_error.empty()
                                   ? "Speaker audio pipeline stopped."
                                   : spkStatsBeforeRestart->last_error;
          }
          const bool needSpkRestart =
              (!spk_pipeline) || spkPipelineDead ||
              (lastSpeakerPreparedGeneration != speakerBackendGeneration) ||
              (lastSpeakerBackend != desiredSpkBackend) ||
              (lastSpeakerTargetSink != sinkName) ||
              ((wantSpkMaxine || wantSpkOpenAudio) && speakerEffectsChanged);

          if (spkPipelineDead) {
            if (spk_pipeline) {
              spk_pipeline->Stop();
              spk_pipeline.reset();
            }
            spk_processor.reset();
            if (spk_fx) {
              spk_fx->Destroy();
              spk_fx.reset();
            }
            if (!wantSpkMaxine) {
              spk_api.reset();
              lastSpeakerAfxLib.clear();
            }
            lastSpeakerFx.reset();
            lastSpeakerBackend.clear();
            lastSpeakerTargetSink.clear();
            speakerRestartingAfterTerminalFailure = true;
            nextSpeakerStartRetry = now + WorkerDeathRetryDelay(cfg);

            {
              std::lock_guard<std::mutex> lock(mu_);
              st_.speakers_route_mode = "pipeline";
              st_.speakers_pipeline_starting = false;
              st_.speakers_pipeline_running = false;
              st_.speakers_routing_active = false;
              st_.speaker_target_sink_active.clear();
              st_.speakers_pipeline_last_error = spkTerminalError;
              ApplyOpenAudioRuntimeWarning(&st_.speakers_open_audio_runtime,
                                           spkTerminalError);
              if (st_.speakers_open_audio_runtime.active)
                st_.speakers_open_audio_runtime.active = false;
              st_.speakers_pipeline_state = "failed";
              st_.speakers_pipeline_idle_reason.clear();
              clearSpeakerPipelineStatsLocked();
            }
          } else if (now >= nextSpeakerStartRetry && needSpkRestart) {
            if (spk_pipeline) {
              spk_pipeline->Stop();
              spk_pipeline.reset();
            }
            spk_processor.reset();
            if (spk_fx) {
              spk_fx->Destroy();
              spk_fx.reset();
            }
            if (!wantSpkMaxine) {
              // Pass-through + open-source don't need the AFX runtime.
              spk_api.reset();
              lastSpeakerAfxLib.clear();
              lastSpeakerFx.reset();
            }

            {
              std::lock_guard<std::mutex> lock(mu_);
              st_.speakers_route_mode = "pipeline";
              st_.speakers_pipeline_starting = true;
              st_.speakers_pipeline_running = false;
              st_.speakers_routing_active = false;
              st_.speaker_target_sink_active.clear();
              st_.speakers_backend_active =
                  std::string(ToString(speakerDecision.backend));
              st_.speakers_effects_note = speakerDecision.note;
              st_.speakers_intensity = speakerPlan.intensity;
              st_.speakers_pipeline_last_error = spkTerminalError;
              st_.speakers_pipeline_state = "starting";
              st_.speakers_pipeline_idle_reason.clear();
            }

            // Build the processor (Maxine/Open Audio/Passthrough), with
            // graceful fallback.
            if (wantSpkOpenAudio) {
              BeforePreparationProbe(&hooks_, "model_registry", &stop_);
              if (stop_.load(std::memory_order_acquire))
                break;
              studiocast::open_audio::ResolvedOpenAudioModel selected;
              std::string oerr;
              auto oa = studiocast::open_audio::OpenAudioAudioProcessor::
                  CreateForSpeaker(cfg.effects, &selected, &oerr);
              if (!oa) {
                speakerOpenAudioCooldownUntil =
                    steady_clock::time_point::max();
                speakerOpenAudioCooldownReason = oerr;

                desiredSpkBackend = "passthrough";
                spk_processor = std::make_unique<PassthroughAudioProcessor>();
                {
                  std::lock_guard<std::mutex> lock(mu_);
                  st_.speakers_open_audio_runtime = {};
                  SetOpenAudioRuntimeModel(&st_.speakers_open_audio_runtime,
                                           selected);
                  MarkOpenAudioRuntimeDisabled(&st_.speakers_open_audio_runtime,
                                               "Open Audio init failed: " +
                                                   oerr);
                  st_.speakers_backend_active = "passthrough";
                  st_.speakers_effects_note =
                      "Open-source speaker backend failed to initialize; using "
                      "pass-through.\n" +
                      oerr;
                  st_.speakers_pipeline_last_error =
                      "Open Audio init failed: " + oerr;
                }
              } else {
                {
                  std::lock_guard<std::mutex> lock(mu_);
                  st_.speakers_open_audio_runtime =
                      MakeActiveOpenAudioRuntimeStatus(selected, *oa);
                }
                spk_processor = std::move(oa);
              }
            } else if (wantSpkMaxine) {
              // Resolve GPU selection (settings.conf) and AFX SDK paths.
              BeforePreparationProbe(&hooks_, "settings", &stop_);
              if (stop_.load(std::memory_order_acquire))
                break;
              const auto settings = hooks_.load_settings
                                        ? hooks_.load_settings()
                                        : studiocast::config::LoadSettings();
              BeforePreparationProbe(&hooks_, "gpu_selection", &stop_);
              if (stop_.load(std::memory_order_acquire))
                break;
              const auto sel = hooks_.select_gpu
                                   ? hooks_.select_gpu(settings.gpu, &stop_)
                                   : studiocast::maxine::SelectGpu(settings.gpu,
                                                                  &stop_);
              if (!sel.selected || !sel.selected->compute_capability) {
                speakerMaxineCooldownUntil =
                    steady_clock::time_point::max();
                speakerMaxineCooldownReason =
                    "GPU selection failed: " + sel.error;
                desiredSpkBackend = "passthrough";
                spk_processor = std::make_unique<PassthroughAudioProcessor>();
                {
                  std::lock_guard<std::mutex> lock(mu_);
                  st_.speakers_backend_active = "passthrough";
                  st_.speakers_effects_note =
                      "Failed to select a supported NVIDIA GPU for speaker "
                      "effects; using pass-through.\n" +
                      sel.error;
                  st_.speakers_pipeline_last_error =
                      "GPU selection failed: " + sel.error;
                }
              } else {
                BeforePreparationProbe(&hooks_, "sdk_paths", &stop_);
                if (stop_.load(std::memory_order_acquire))
                  break;
                const auto paths =
                    hooks_.resolve_maxine_paths
                        ? hooks_.resolve_maxine_paths()
                        : studiocast::maxine::ResolveMaxinePaths();
                if (!paths.afx.ok) {
                  std::string msg = "AFX SDK not available";
                  if (!paths.afx.problems.empty()) {
                    msg += ": ";
                    msg += paths.afx.problems.front();
                  }
                  desiredSpkBackend = "passthrough";
                  speakerMaxineCooldownUntil =
                      steady_clock::time_point::max();
                  speakerMaxineCooldownReason = msg;
                  spk_processor = std::make_unique<PassthroughAudioProcessor>();
                  {
                    std::lock_guard<std::mutex> lock(mu_);
                    st_.speakers_backend_active = "passthrough";
                    st_.speakers_effects_note =
                        "AFX SDK not available for speaker effects; using "
                        "pass-through.\n" +
                        msg;
                    st_.speakers_pipeline_last_error = msg;
                  }
                } else {
                  if (!spk_api || paths.afx.library != lastSpeakerAfxLib) {
                    spk_api =
                        std::make_unique<studiocast::maxine::afx::AfxApi>();
                    std::string aerr;
                    if (!spk_api->InitializeFromLibraryPath(paths.afx.library,
                                                            &aerr)) {
                      speakerMaxineCooldownUntil =
                          steady_clock::time_point::max();
                      speakerMaxineCooldownReason =
                          "AFX init failed: " + aerr;
                      desiredSpkBackend = "passthrough";
                      spk_api.reset();
                      spk_processor =
                          std::make_unique<PassthroughAudioProcessor>();
                      {
                        std::lock_guard<std::mutex> lock(mu_);
                        st_.speakers_backend_active = "passthrough";
                        st_.speakers_effects_note =
                            "Failed to initialize AFX runtime for speaker "
                            "effects; using pass-through.\n" +
                            aerr;
                        st_.speakers_pipeline_last_error =
                            "AFX init failed: " + aerr;
                      }
                    }
                    lastSpeakerAfxLib = paths.afx.library;
                  }

                  if (!spk_processor) {
                    if (!spk_fx) {
                      spk_fx =
                          std::make_unique<studiocast::maxine::afx::AfxEffect>(
                              spk_api.get());
                    } else {
                      spk_fx->SetApi(spk_api.get());
                    }

                    studiocast::maxine::afx::AfxEffectConfig e;
                    e.effect_selector = speakerPlan.effect_selector;
                    e.feature_id = speakerPlan.feature_id;
                    e.features_dir = paths.afx.features_dir;
                    e.compute_capability = sel.selected->compute_capability;
                    e.sample_rate = 48000;
                    e.frame_samples = 480;
                    // AFX speaker effects are treated as mono voice processors.
                    // We keep the effect configured for mono and preserve
                    // stereo in the AudioProcessor wrapper via Mid/Side
                    // processing.
                    e.channels = 1;
                    e.intensity = speakerPlan.intensity;
                    e.use_denoiser_v2_model = speakerPlan.use_denoiser_v2_model;

                    std::string ferr;
                    if (!spk_fx->Configure(e, &ferr) || !spk_fx->Load(&ferr)) {
                      speakerMaxineCooldownUntil =
                          steady_clock::time_point::max();
                      speakerMaxineCooldownReason =
                          "AFX load failed: " + ferr;
                      desiredSpkBackend = "passthrough";
                      spk_fx->Destroy();
                      spk_fx.reset();
                      spk_processor =
                          std::make_unique<PassthroughAudioProcessor>();
                      {
                        std::lock_guard<std::mutex> lock(mu_);
                        st_.speakers_backend_active = "passthrough";
                        st_.speakers_effects_note =
                            "Failed to configure/load AFX speaker effect; "
                            "using pass-through.\n" +
                            ferr;
                        st_.speakers_pipeline_last_error =
                            "AFX load failed: " + ferr;
                      }
                    } else {
                      spk_processor = std::make_unique<
                          studiocast::maxine::afx::AfxStereoAudioProcessor>(
                          spk_fx.get());
                    }
                  }
                }
              }
            } else {
              spk_processor = std::make_unique<PassthroughAudioProcessor>();
            }

            // Start pipeline (even in pass-through mode; this replaces
            // module-loopback when speaker effects are enabled).
            spk_pipeline = CreatePipeline(spk_processor.get());
            std::string perr;
            bool spkStartOk = false;
            if (!spk_pipeline) {
              perr = "speaker audio pipeline factory returned null";
            } else {
              studiocast::audio::AudioPipelineConfig pcfg;
              pcfg.source_name =
                  studiocast::audio::VirtualSpeakerMonitorSourceName();
              pcfg.allow_monitor_source = true;
              pcfg.sink_name = sinkName;
              // Speaker processing should preserve stereo.
              pcfg.channels = 2;
              spkStartOk = spk_pipeline->Start(pcfg, &perr);
              if (spkStartOk) {
                lastSpeakerBackend = desiredSpkBackend;
                lastSpeakerTargetSink = sinkName;
                lastSpeakerFx = cfg.effects.speaker;
                lastSpeakerPreparedGeneration = speakerBackendGeneration;

                {
                  std::lock_guard<std::mutex> lock(mu_);
                  st_.speakers_pipeline_starting = false;
                  st_.speakers_pipeline_running = true;
                  st_.speakers_routing_active = true;
                  st_.speakers_backend_active = desiredSpkBackend;
                  st_.speaker_target_sink_active = sinkName;
                  st_.speakers_pipeline_state = "running";
                  st_.speakers_pipeline_idle_reason.clear();
                  // Preserve st_.speakers_effects_note (decision/fallback
                  // message).
                  if (!speakerRestartingAfterTerminalFailure) {
                    st_.speakers_pipeline_last_error.clear();
                  }
                }
                speakerRestartingAfterTerminalFailure = false;
              }
            }

            if (!spkStartOk) {
              if (perr.empty()) {
                perr = "speaker audio pipeline failed to start";
              }
              {
                std::lock_guard<std::mutex> lock(mu_);
                st_.speakers_pipeline_starting = false;
                st_.speakers_pipeline_running = false;
                st_.speakers_routing_active = false;
                st_.speaker_target_sink_active.clear();
                st_.speakers_pipeline_frames_processed = 0;
                st_.speakers_pipeline_process_time_us_sum = 0;
                st_.speakers_pipeline_process_time_us_max = 0;
                st_.speakers_pipeline_process_time_us_last = 0;
                st_.speakers_pipeline_process_overruns = 0;
                st_.speakers_pipeline_pulse_capture_latency_us_last = 0;
                st_.speakers_pipeline_pulse_playback_latency_us_last = 0;
                st_.speakers_pipeline_pulse_latency_us_max = 0;
                st_.speakers_pipeline_resync_events = 0;
                st_.speakers_pipeline_last_error =
                    "Failed to start speaker pipeline: " + perr;
                if (st_.speakers_open_audio_runtime.active)
                  st_.speakers_open_audio_runtime.active = false;
                st_.speakers_pipeline_state = "failed";
                st_.speakers_pipeline_idle_reason.clear();
              }

              spk_pipeline.reset();
              spk_processor.reset();
              nextSpeakerStartRetry = now + StartFailureRetryDelay(cfg);
            }
          } else if (wantSpkOpenAudio) {
            if (auto *oa = dynamic_cast<
                    studiocast::open_audio::OpenAudioAudioProcessor *>(
                    spk_processor.get())) {
              oa->UpdateFromSpeakerConfig(cfg.effects.speaker);
              lastSpeakerFx = cfg.effects.speaker;
              std::lock_guard<std::mutex> lock(mu_);
              st_.speakers_intensity = speakerPlan.intensity;
            }
          }

          if (spk_pipeline) {
            const auto stats = spk_pipeline->GetStats();
            {
              std::lock_guard<std::mutex> lock(mu_);
              st_.speakers_pipeline_running = stats.running;
              st_.speakers_routing_active = stats.running;
              if (stats.running) {
                st_.speaker_target_sink_active = sinkName;
              } else {
                st_.speaker_target_sink_active.clear();
              }
              st_.speakers_pipeline_frames_processed = stats.frames_processed;
              st_.speakers_pipeline_process_time_us_sum =
                  stats.process_time_us_sum;
              st_.speakers_pipeline_process_time_us_max =
                  stats.process_time_us_max;
              st_.speakers_pipeline_process_time_us_last =
                  stats.process_time_us_last;
              st_.speakers_pipeline_process_overruns = stats.process_overruns;
              st_.speakers_pipeline_pulse_capture_latency_us_last =
                  stats.pulse_capture_latency_us_last;
              st_.speakers_pipeline_pulse_playback_latency_us_last =
                  stats.pulse_playback_latency_us_last;
              st_.speakers_pipeline_pulse_latency_us_max =
                  stats.pulse_latency_us_max;
              st_.speakers_pipeline_resync_events = stats.resync_events;
              if (!stats.last_error.empty()) {
                st_.speakers_pipeline_last_error = stats.last_error;
                ApplyOpenAudioRuntimeWarning(&st_.speakers_open_audio_runtime,
                                             stats.last_error);
              }
              st_.speakers_pipeline_state =
                  stats.running ? "running" : "failed";
              if (stats.running)
                st_.speakers_pipeline_idle_reason.clear();
            }
          }
        }
      }
    }
#endif

    if (!micPipelineNeeded) {
#if STUDIOCAST_HAVE_PULSE_SIMPLE
      if (pipeline) {
        pipeline->Stop();
        pipeline.reset();
      }
      processor.reset();
      if (fx) {
        fx->Destroy();
        fx.reset();
      }
      api.reset();
      lastFx.reset();
      lastSource.clear();
      lastBackend.clear();
      lastAfxLib.clear();
      micRestartingAfterTerminalFailure = false;
#endif
      {
        std::lock_guard<std::mutex> lock(mu_);
        st_.pipeline_running = false;
        st_.pipeline_starting = false;
        st_.pipeline_frames_processed = 0;
        st_.pipeline_process_time_us_sum = 0;
        st_.pipeline_process_time_us_max = 0;
        st_.pipeline_process_time_us_last = 0;
        st_.pipeline_process_overruns = 0;
        st_.pipeline_pulse_capture_latency_us_last = 0;
        st_.pipeline_pulse_playback_latency_us_last = 0;
        st_.pipeline_pulse_latency_us_max = 0;
        st_.pipeline_resync_events = 0;
        st_.effect_selector.clear();
        st_.feature_id.clear();
        st_.intensity = 0.0f;
        st_.effects_backend_active.clear();
        st_.effects_note.clear();
        st_.open_audio_runtime = {};
        st_.pipeline_active_needed = false;
        st_.pipeline_state = cfg.enabled ? "idle_no_consumer" : "disabled";
        st_.pipeline_idle_reason =
            cfg.enabled
                ? (micConsumers.error.empty()
                       ? "No active virtual microphone consumer."
                       : "Virtual microphone consumer detection unavailable: " +
                             micConsumers.error)
                : "Audio processing disabled.";
      }
      SleepFor(milliseconds(pollMs));
      continue;
    }

    if (!micSourceStatus.error.empty()) {
#if STUDIOCAST_HAVE_PULSE_SIMPLE
      if (pipeline) {
        pipeline->Stop();
        pipeline.reset();
      }
      processor.reset();
      if (fx) {
        fx->Destroy();
        fx.reset();
      }
      api.reset();
      lastFx.reset();
      lastSource.clear();
      lastBackend.clear();
      lastAfxLib.clear();
      micRestartingAfterTerminalFailure = false;
#endif
      SetLastError(micSourceStatus.error);
      {
        std::lock_guard<std::mutex> lock(mu_);
        st_.pipeline_running = false;
        st_.pipeline_starting = false;
        st_.pipeline_state = "failed";
        st_.pipeline_idle_reason.clear();
        st_.pipeline_frames_processed = 0;
        st_.pipeline_process_time_us_sum = 0;
        st_.pipeline_process_time_us_max = 0;
        st_.pipeline_process_time_us_last = 0;
        st_.pipeline_process_overruns = 0;
        st_.pipeline_pulse_capture_latency_us_last = 0;
        st_.pipeline_pulse_playback_latency_us_last = 0;
        st_.pipeline_pulse_latency_us_max = 0;
        st_.pipeline_resync_events = 0;
        st_.effect_selector.clear();
        st_.feature_id.clear();
        st_.intensity = 0.0f;
        st_.effects_backend_active.clear();
        st_.effects_note.clear();
        st_.open_audio_runtime = {};
      }
      SleepFor(milliseconds(pollMs));
      continue;
    }

    auto plan = studiocast::maxine::afx::PlanBroadcastMicrophoneEffect(
        cfg.effects.microphone.studio_voice_enabled,
        cfg.effects.microphone.noise_removal_enabled,
        cfg.effects.microphone.room_echo_removal_enabled,
        cfg.effects.microphone.strength);

    if (!plan.enabled) {
      // Planner maps strength -> intensity even when disabled; normalize for
      // pass-through status.
      plan.intensity = 0.0f;
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      st_.effect_selector = plan.effect_selector;
      st_.feature_id = plan.feature_id;
      st_.intensity = plan.intensity;
    }

    // Backend selection.
    AudioBackendAvailability avail;
    bool micOpenAudioCooldownActive = false;
    if (AnyMicrophoneEffectRequested(cfg.effects)) {
      const auto now2 = steady_clock::now();
      const auto micAvailabilityKey =
          MakeMicrophoneAvailabilityCacheKey(cfg.effects);
      if (cachedMicAvailabilityKey.has_value() &&
          *cachedMicAvailabilityKey != micAvailabilityKey) {
        openAudioCooldownUntil = {};
        openAudioCooldownReason.clear();
        maxineCooldownUntil = {};
        maxineCooldownReason.clear();
      }
      const bool micAvailabilityExpired =
          !cachedMicAvailability || !cachedMicAvailabilityKey.has_value() ||
          *cachedMicAvailabilityKey != micAvailabilityKey ||
          cachedMicBackendGeneration != micBackendGeneration;
      if (micAvailabilityExpired) {
        cachedMicAvailability = ProbeMicrophoneBackendAvailability(cfg);
        cachedMicAvailabilityKey = micAvailabilityKey;
        cachedMicBackendGeneration = micBackendGeneration;
      }
      avail = *cachedMicAvailability;
      if (now2 < openAudioCooldownUntil) {
        micOpenAudioCooldownActive = true;
        avail.open_source_ok = false;
        avail.open_source_reason = openAudioCooldownReason.empty()
                                       ? "Open Audio backend is disabled after "
                                         "a preparation failure. Refresh "
                                         "diagnostics to retry."
                                       : openAudioCooldownReason;
      }
      if (now2 < maxineCooldownUntil) {
        avail.maxine_ok = false;
        avail.maxine_reason = maxineCooldownReason.empty()
                                  ? "Maxine backend is disabled after a "
                                    "preparation failure. Refresh diagnostics "
                                    "to retry."
                                  : maxineCooldownReason;
      }
    }
    const auto decision = ResolveAudioBackend(cfg.effects, avail);
    {
      std::lock_guard<std::mutex> lock(mu_);
      st_.effects_backend_active = std::string(ToString(decision.backend));
      st_.effects_note = decision.note;
      if (micOpenAudioCooldownActive) {
        MarkOpenAudioRuntimeDisabled(&st_.open_audio_runtime,
                                     avail.open_source_reason);
      }
    }

#if !STUDIOCAST_HAVE_PULSE_SIMPLE
    SetLastError(
        "Audio pipeline disabled at build time (libpulse-simple not found)");
    {
      std::lock_guard<std::mutex> lock(mu_);
      st_.pipeline_running = false;
      st_.pipeline_starting = false;
      st_.pipeline_state = "failed";
      st_.pipeline_idle_reason.clear();
    }
    SleepFor(milliseconds(pollMs));
    continue;
#else

    const auto now = steady_clock::now();
    if (now < nextStartRetry) {
      if (pipeline) {
        const auto stats = pipeline->GetStats();
        if (!stats.last_error.empty()) {
          SetLastError(stats.last_error);
        }
        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.pipeline_running = stats.running;
          st_.pipeline_frames_processed = stats.frames_processed;
          st_.pipeline_process_time_us_sum = stats.process_time_us_sum;
          st_.pipeline_process_time_us_max = stats.process_time_us_max;
          st_.pipeline_process_time_us_last = stats.process_time_us_last;
          st_.pipeline_process_overruns = stats.process_overruns;
          st_.pipeline_pulse_capture_latency_us_last =
              stats.pulse_capture_latency_us_last;
          st_.pipeline_pulse_playback_latency_us_last =
              stats.pulse_playback_latency_us_last;
          st_.pipeline_pulse_latency_us_max = stats.pulse_latency_us_max;
          st_.pipeline_resync_events = stats.resync_events;
          ApplyOpenAudioRuntimeWarning(&st_.open_audio_runtime,
                                       stats.last_error);
          st_.pipeline_state = stats.running ? "running" : "failed";
          if (!stats.running) {
            st_.pipeline_starting = false;
          }
          if (stats.running)
            st_.pipeline_idle_reason.clear();
        }
      } else {
        std::lock_guard<std::mutex> lock(mu_);
        st_.pipeline_running = false;
        st_.pipeline_starting = false;
        st_.pipeline_state = "failed";
        st_.pipeline_idle_reason.clear();
      }
      SleepFor(milliseconds(pollMs));
      continue;
    }

    const bool wantMaxine =
        (decision.backend == AudioBackendKind::kMaxine) && plan.enabled;
    const bool wantOpenAudio =
        (decision.backend == AudioBackendKind::kOpenSource) && plan.enabled;

    std::string desiredBackend = "passthrough";
    if (wantMaxine) {
      desiredBackend = "maxine";
    } else if (wantOpenAudio) {
      desiredBackend = "open_source";
    }
    if (!wantOpenAudio && !micOpenAudioCooldownActive) {
      std::lock_guard<std::mutex> lock(mu_);
      st_.open_audio_runtime = {};
    }

    const bool micRestartKeyChanged =
        (!lastFx.has_value() ||
         MicrophonePipelineEffectsRequireRestart(*lastFx, cfg.effects));
    const bool micStrengthChanged =
        lastFx.has_value() &&
        lastFx->microphone.strength != cfg.effects.microphone.strength;
    const bool effectsChanged =
        micRestartKeyChanged || (wantMaxine && micStrengthChanged);
    std::optional<AudioPipelineStats> statsBeforeRestart;
    if (pipeline) {
      statsBeforeRestart = pipeline->GetStats();
    }
    const bool pipelineDead =
        (statsBeforeRestart && !statsBeforeRestart->running);
    std::string terminalError;
    if (pipelineDead) {
      terminalError = statsBeforeRestart->last_error.empty()
                          ? "Audio pipeline stopped."
                          : statsBeforeRestart->last_error;
      SetLastError(terminalError);
    }
    if (pipelineDead) {
      if (pipeline) {
        pipeline->Stop();
        pipeline.reset();
      }
      processor.reset();
      if (fx) {
        fx->Destroy();
        fx.reset();
      }
      if (!wantMaxine) {
        api.reset();
        lastAfxLib.clear();
      }
      lastFx.reset();
      lastSource.clear();
      lastBackend.clear();
      micRestartingAfterTerminalFailure = true;
      nextStartRetry = now + WorkerDeathRetryDelay(cfg);
      {
        std::lock_guard<std::mutex> lock(mu_);
        st_.pipeline_running = false;
        st_.pipeline_starting = false;
        st_.pipeline_state = "failed";
        st_.pipeline_idle_reason.clear();
        ApplyOpenAudioRuntimeWarning(&st_.open_audio_runtime, terminalError);
        if (st_.open_audio_runtime.active)
          st_.open_audio_runtime.active = false;
        clearMicPipelineStatsLocked();
      }
      SleepFor(milliseconds(pollMs));
      continue;
    }

    const std::string micPipelineSource = micSourceStatus.selected_source;
    const bool needRestart =
        (!pipeline) || (lastBackend != desiredBackend) ||
        (lastSource != micPipelineSource) ||
        (lastMicPreparedGeneration != micBackendGeneration) ||
        ((wantMaxine || wantOpenAudio) && effectsChanged);

    if (needRestart) {
      if (pipeline) {
        pipeline->Stop();
        pipeline.reset();
      }
      processor.reset();
      if (fx) {
        fx->Destroy();
        fx.reset();
      }
      if (!wantMaxine) {
        // Pass-through mode does not need the AFX runtime.
        api.reset();
        lastAfxLib.clear();
        lastFx.reset();
      }
    }

    auto startMicPassthroughFallback =
        [&](const std::string &note,
            const std::string &failure_reason) -> bool {
      maxineCooldownUntil = steady_clock::time_point::max();
      maxineCooldownReason = failure_reason;
      SetLastError(failure_reason);

      if (pipeline) {
        pipeline->Stop();
        pipeline.reset();
      }
      processor.reset();
      if (fx) {
        fx->Destroy();
        fx.reset();
      }
      api.reset();
      lastAfxLib.clear();
      lastFx.reset();
      lastSource.clear();
      lastBackend.clear();

      {
        std::lock_guard<std::mutex> lock(mu_);
        st_.pipeline_starting = true;
        st_.pipeline_running = false;
        st_.pipeline_state = "starting";
        st_.pipeline_idle_reason.clear();
        st_.effects_backend_active = "passthrough";
        st_.effects_note = note;
        st_.open_audio_runtime = {};
        clearMicPipelineStatsLocked();
      }

      processor = std::make_unique<PassthroughAudioProcessor>();
      pipeline = CreatePipeline(processor.get());

      studiocast::audio::AudioPipelineConfig pcfg;
      pcfg.source_name = micPipelineSource;
      pcfg.sink_name = "studiocast_sink";

      std::string perr;
      bool pipelineStartOk = false;
      if (!pipeline) {
        perr = "audio pipeline factory returned null";
      } else {
        pipelineStartOk = pipeline->Start(pcfg, &perr);
      }
      if (!pipelineStartOk) {
        if (perr.empty()) {
          perr = "audio pipeline failed to start";
        }
        SetLastError("Failed to start audio pipeline: " + perr);
        pipeline.reset();
        processor.reset();
        nextStartRetry = steady_clock::now() + StartFailureRetryDelay(cfg);
        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.pipeline_starting = false;
          st_.pipeline_running = false;
          st_.pipeline_state = "failed";
          st_.pipeline_idle_reason.clear();
          clearMicPipelineStatsLocked();
        }
        return false;
      }

      lastBackend = "passthrough";
      lastSource = micPipelineSource;
      lastFx = cfg.effects;
      lastMicPreparedGeneration = micBackendGeneration;
      {
        std::lock_guard<std::mutex> lock(mu_);
        st_.pipeline_starting = false;
        st_.pipeline_running = true;
        st_.pipeline_state = "running";
        st_.pipeline_idle_reason.clear();
        st_.last_error.clear();
      }
      micRestartingAfterTerminalFailure = false;
      return true;
    };

    if (wantOpenAudio) {
      // Open-source backend (Phase 4 stub): validate model selection and keep
      // the pipeline alive.
      if (needRestart) {
        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.pipeline_starting = true;
          st_.pipeline_running = false;
          st_.pipeline_state = "starting";
          st_.pipeline_idle_reason.clear();

          // Open-source backend does not require Maxine GPU selection.
          st_.gpu_index = -1;
          st_.gpu_name.clear();
          st_.gpu_compute_cap.clear();
        }

        BeforePreparationProbe(&hooks_, "model_registry", &stop_);
        if (stop_.load(std::memory_order_acquire))
          break;
        studiocast::open_audio::ResolvedOpenAudioModel selected;
        std::string oerr;
        auto oa = studiocast::open_audio::OpenAudioAudioProcessor::
            CreateForMicrophone(cfg.effects, &selected, &oerr);
        if (!oa) {
          // Fall back to pass-through with a cooldown to avoid restart loops.
          SetLastError("Open Audio initialization failed: " + oerr);
          openAudioCooldownUntil = steady_clock::time_point::max();
          openAudioCooldownReason = oerr;

          desiredBackend = "passthrough";
          processor = std::make_unique<PassthroughAudioProcessor>();

          // Update status to reflect actual backend (decision may still say
          // open_source this tick).
          {
            std::lock_guard<std::mutex> lock(mu_);
            st_.open_audio_runtime = {};
            SetOpenAudioRuntimeModel(&st_.open_audio_runtime, selected);
            MarkOpenAudioRuntimeDisabled(&st_.open_audio_runtime,
                                         "Open Audio init failed: " + oerr);
            st_.effects_backend_active = "passthrough";
            st_.effects_note = "Open-source audio backend failed to "
                               "initialize; using pass-through.\n" +
                               oerr;
          }
        } else {
          {
            std::lock_guard<std::mutex> lock(mu_);
            st_.open_audio_runtime =
                MakeActiveOpenAudioRuntimeStatus(selected, *oa);
          }
          processor = std::move(oa);
        }

        pipeline = CreatePipeline(processor.get());

        studiocast::audio::AudioPipelineConfig pcfg;
        pcfg.source_name = micPipelineSource;
        pcfg.sink_name = "studiocast_sink";

        std::string perr;
        bool pipelineStartOk = false;
        if (!pipeline) {
          perr = "audio pipeline factory returned null";
        } else {
          pipelineStartOk = pipeline->Start(pcfg, &perr);
        }
        if (!pipelineStartOk) {
          if (perr.empty()) {
            perr = "audio pipeline failed to start";
          }
          SetLastError("Failed to start audio pipeline: " + perr);
          pipeline.reset();
          processor.reset();
          nextStartRetry = now + StartFailureRetryDelay(cfg);
          {
            std::lock_guard<std::mutex> lock(mu_);
            st_.pipeline_starting = false;
            st_.pipeline_running = false;
            st_.pipeline_state = "failed";
            st_.pipeline_idle_reason.clear();
            if (st_.open_audio_runtime.active)
              st_.open_audio_runtime.active = false;
            clearMicPipelineStatsLocked();
          }
          SleepFor(milliseconds(pollMs));
          continue;
        }

        lastBackend = desiredBackend;
        lastSource = micPipelineSource;
        lastFx = cfg.effects;
        lastMicPreparedGeneration = micBackendGeneration;

        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.pipeline_starting = false;
          st_.pipeline_running = true;
          st_.pipeline_state = "running";
          st_.pipeline_idle_reason.clear();
          if (!micRestartingAfterTerminalFailure) {
            st_.last_error.clear();
          }
        }
        micRestartingAfterTerminalFailure = false;
      } else if (auto *oa = dynamic_cast<
                     studiocast::open_audio::OpenAudioAudioProcessor *>(
                     processor.get())) {
        oa->UpdateFromMicrophoneConfig(cfg.effects.microphone);
        lastFx = cfg.effects;
      }

      if (pipeline) {
        const auto stats = pipeline->GetStats();
        if (!stats.last_error.empty()) {
          SetLastError(stats.last_error);
        }
        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.pipeline_running = stats.running;
          st_.pipeline_frames_processed = stats.frames_processed;
          st_.pipeline_process_time_us_sum = stats.process_time_us_sum;
          st_.pipeline_process_time_us_max = stats.process_time_us_max;
          st_.pipeline_process_time_us_last = stats.process_time_us_last;
          st_.pipeline_process_overruns = stats.process_overruns;
          st_.pipeline_pulse_capture_latency_us_last =
              stats.pulse_capture_latency_us_last;
          st_.pipeline_pulse_playback_latency_us_last =
              stats.pulse_playback_latency_us_last;
          st_.pipeline_pulse_latency_us_max = stats.pulse_latency_us_max;
          st_.pipeline_resync_events = stats.resync_events;
          ApplyOpenAudioRuntimeWarning(&st_.open_audio_runtime,
                                       stats.last_error);
          st_.pipeline_state = stats.running ? "running" : "failed";
          if (stats.running)
            st_.pipeline_idle_reason.clear();
        }
      }

      SleepFor(milliseconds(pollMs));
      continue;
    }

    if (!wantMaxine) {
      // Effects are disabled: keep the pipeline alive with pass-through audio.
      if (needRestart) {
        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.pipeline_starting = true;
          st_.pipeline_running = false;
          st_.pipeline_state = "starting";
          st_.pipeline_idle_reason.clear();

          // No GPU requirement in pass-through mode.
          st_.gpu_index = -1;
          st_.gpu_name.clear();
          st_.gpu_compute_cap.clear();
        }

        processor = std::make_unique<PassthroughAudioProcessor>();
        pipeline = CreatePipeline(processor.get());

        studiocast::audio::AudioPipelineConfig pcfg;
        pcfg.source_name = micPipelineSource;
        pcfg.sink_name = "studiocast_sink";

        std::string perr;
        bool pipelineStartOk = false;
        if (!pipeline) {
          perr = "audio pipeline factory returned null";
        } else {
          pipelineStartOk = pipeline->Start(pcfg, &perr);
        }
        if (!pipelineStartOk) {
          if (perr.empty()) {
            perr = "audio pipeline failed to start";
          }
          SetLastError("Failed to start audio pipeline: " + perr);
          pipeline.reset();
          processor.reset();
          nextStartRetry = now + StartFailureRetryDelay(cfg);
          {
            std::lock_guard<std::mutex> lock(mu_);
            st_.pipeline_starting = false;
            st_.pipeline_running = false;
            st_.pipeline_state = "failed";
            st_.pipeline_idle_reason.clear();
            clearMicPipelineStatsLocked();
          }
          SleepFor(milliseconds(pollMs));
          continue;
        }

        lastBackend = desiredBackend;
        lastSource = micPipelineSource;
        lastMicPreparedGeneration = micBackendGeneration;

        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.pipeline_starting = false;
          st_.pipeline_running = true;
          st_.pipeline_state = "running";
          st_.pipeline_idle_reason.clear();
          if (!micRestartingAfterTerminalFailure) {
            st_.last_error.clear();
          }
        }
        micRestartingAfterTerminalFailure = false;
      }

      if (pipeline) {
        const auto stats = pipeline->GetStats();
        if (!stats.last_error.empty()) {
          SetLastError(stats.last_error);
        }
        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.pipeline_running = stats.running;
          st_.pipeline_frames_processed = stats.frames_processed;
          st_.pipeline_process_time_us_sum = stats.process_time_us_sum;
          st_.pipeline_process_time_us_max = stats.process_time_us_max;
          st_.pipeline_process_time_us_last = stats.process_time_us_last;
          st_.pipeline_process_overruns = stats.process_overruns;
          st_.pipeline_pulse_capture_latency_us_last =
              stats.pulse_capture_latency_us_last;
          st_.pipeline_pulse_playback_latency_us_last =
              stats.pulse_playback_latency_us_last;
          st_.pipeline_pulse_latency_us_max = stats.pulse_latency_us_max;
          st_.pipeline_resync_events = stats.resync_events;
          st_.pipeline_state = stats.running ? "running" : "failed";
          if (stats.running)
            st_.pipeline_idle_reason.clear();
        }
      }

      SleepFor(milliseconds(pollMs));
      continue;
    }

    if (needRestart) {
      // Resolve GPU selection (settings.conf) and AFX SDK paths only while
      // preparing a new pipeline generation. Stable active audio must not
      // rediscover provider configuration on supervisor polls.
      BeforePreparationProbe(&hooks_, "settings", &stop_);
      if (stop_.load(std::memory_order_acquire))
        break;
      const auto settings = hooks_.load_settings
                                ? hooks_.load_settings()
                                : studiocast::config::LoadSettings();
      BeforePreparationProbe(&hooks_, "gpu_selection", &stop_);
      if (stop_.load(std::memory_order_acquire))
        break;
      const auto sel = hooks_.select_gpu
                           ? hooks_.select_gpu(settings.gpu, &stop_)
                           : studiocast::maxine::SelectGpu(settings.gpu,
                                                          &stop_);
      {
        std::lock_guard<std::mutex> lock(mu_);
        st_.gpu_index = sel.selected ? sel.selected->index : -1;
        st_.gpu_name = sel.selected ? sel.selected->name : std::string();
        st_.gpu_compute_cap = (sel.selected && sel.selected->compute_capability)
                                  ? sel.selected->ComputeCapString()
                                  : std::string();
      }
      if (!sel.selected || !sel.selected->compute_capability) {
        const std::string reason =
            "Failed to select a supported NVIDIA GPU: " + sel.error;
        (void)startMicPassthroughFallback(
            "Failed to select a supported NVIDIA GPU for microphone effects; "
            "using pass-through.\n" +
                sel.error,
            reason);
        SleepFor(milliseconds(pollMs));
        continue;
      }

      BeforePreparationProbe(&hooks_, "sdk_paths", &stop_);
      if (stop_.load(std::memory_order_acquire))
        break;
      const auto paths = hooks_.resolve_maxine_paths
                             ? hooks_.resolve_maxine_paths()
                             : studiocast::maxine::ResolveMaxinePaths();
      if (!paths.afx.ok) {
        std::string msg = "AFX SDK not available";
        if (!paths.afx.problems.empty()) {
          msg += ": ";
          msg += paths.afx.problems.front();
        }
        (void)startMicPassthroughFallback(
            "AFX SDK not available for microphone effects; using "
            "pass-through.\n" +
                msg,
            msg);
        SleepFor(milliseconds(pollMs));
        continue;
      }

      if (!api || paths.afx.library != lastAfxLib) {
        api = std::make_unique<studiocast::maxine::afx::AfxApi>();
        std::string aerr;
        if (!api->InitializeFromLibraryPath(paths.afx.library, &aerr)) {
          (void)startMicPassthroughFallback(
              "Failed to initialize AFX runtime for microphone effects; using "
              "pass-through.\n" +
                  aerr,
              "Failed to initialize AFX runtime: " + aerr);
          SleepFor(milliseconds(pollMs));
          continue;
        }
        lastAfxLib = paths.afx.library;
      }

      if (!fx) {
        fx = std::make_unique<studiocast::maxine::afx::AfxEffect>(api.get());
      } else {
        fx->SetApi(api.get());
      }

      studiocast::maxine::afx::AfxEffectConfig e;
      e.effect_selector = plan.effect_selector;
      e.feature_id = plan.feature_id;
      e.features_dir = paths.afx.features_dir;
      e.compute_capability = sel.selected->compute_capability;
      e.sample_rate = 48000;
      e.frame_samples = 480;
      e.channels = 1;
      e.intensity = plan.intensity;
      e.use_denoiser_v2_model = plan.use_denoiser_v2_model;

      {
        std::lock_guard<std::mutex> lock(mu_);
        st_.pipeline_starting = true;
        st_.pipeline_running = false;
        st_.pipeline_state = "starting";
        st_.pipeline_idle_reason.clear();
      }

      std::string ferr;
      if (!fx->Configure(e, &ferr)) {
        (void)startMicPassthroughFallback(
            "Failed to configure AFX microphone effect; using "
            "pass-through.\n" +
                ferr,
            "Failed to configure AFX effect: " + ferr);
        SleepFor(milliseconds(pollMs));
        continue;
      }
      if (!fx->Load(&ferr)) {
        (void)startMicPassthroughFallback(
            "Failed to load AFX microphone effect; using pass-through.\n" +
                ferr,
            "Failed to load AFX effect: " + ferr);
        SleepFor(milliseconds(pollMs));
        continue;
      }

      processor = std::make_unique<studiocast::maxine::afx::AfxAudioProcessor>(
          fx.get());
      pipeline = CreatePipeline(processor.get());
      studiocast::audio::AudioPipelineConfig pcfg;
      pcfg.source_name = micPipelineSource;
      pcfg.sink_name = "studiocast_sink";

      std::string perr;
      bool pipelineStartOk = false;
      if (!pipeline) {
        perr = "audio pipeline factory returned null";
      } else {
        pipelineStartOk = pipeline->Start(pcfg, &perr);
      }
      if (!pipelineStartOk) {
        if (perr.empty()) {
          perr = "audio pipeline failed to start";
        }
        SetLastError("Failed to start audio pipeline: " + perr);
        pipeline.reset();
        processor.reset();
        nextStartRetry = now + StartFailureRetryDelay(cfg);
        {
          std::lock_guard<std::mutex> lock(mu_);
          st_.pipeline_starting = false;
          st_.pipeline_running = false;
          st_.pipeline_state = "failed";
          st_.pipeline_idle_reason.clear();
          clearMicPipelineStatsLocked();
        }
        SleepFor(milliseconds(pollMs));
        continue;
      }

      lastFx = cfg.effects;
      lastSource = micPipelineSource;
      lastBackend = desiredBackend;
      lastMicPreparedGeneration = micBackendGeneration;
      {
        std::lock_guard<std::mutex> lock(mu_);
        st_.pipeline_starting = false;
        st_.pipeline_running = true;
        st_.pipeline_state = "running";
        st_.pipeline_idle_reason.clear();
        if (!micRestartingAfterTerminalFailure) {
          st_.last_error.clear();
        }
      }
      micRestartingAfterTerminalFailure = false;
    }

    if (pipeline) {
      const auto stats = pipeline->GetStats();
      if (!stats.last_error.empty()) {
        SetLastError(stats.last_error);
      }
      {
        std::lock_guard<std::mutex> lock(mu_);
        st_.pipeline_running = stats.running;
        st_.pipeline_frames_processed = stats.frames_processed;
        st_.pipeline_process_time_us_sum = stats.process_time_us_sum;
        st_.pipeline_process_time_us_max = stats.process_time_us_max;
        st_.pipeline_process_time_us_last = stats.process_time_us_last;
        st_.pipeline_process_overruns = stats.process_overruns;
        st_.pipeline_pulse_capture_latency_us_last =
            stats.pulse_capture_latency_us_last;
        st_.pipeline_pulse_playback_latency_us_last =
            stats.pulse_playback_latency_us_last;
        st_.pipeline_pulse_latency_us_max = stats.pulse_latency_us_max;
        st_.pipeline_resync_events = stats.resync_events;
        st_.pipeline_state = stats.running ? "running" : "failed";
        if (stats.running)
          st_.pipeline_idle_reason.clear();
      }
    }

    SleepFor(milliseconds(pollMs));
#endif
  }

#if STUDIOCAST_HAVE_PULSE_SIMPLE
  if (pipeline)
    pipeline->Stop();
  processor.reset();
  if (fx)
    fx->Destroy();

  if (spk_pipeline)
    spk_pipeline->Stop();
  spk_processor.reset();
  if (spk_fx)
    spk_fx->Destroy();
#endif
}

} // namespace studiocast::audio
