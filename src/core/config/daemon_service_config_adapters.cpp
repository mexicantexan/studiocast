#include "core/config/daemon_service_config_adapters.h"

#include <algorithm>
#include <cctype>

#include "core/util/strings.h"

namespace studiocast::config {
namespace {

studiocast::video::ScalingBackendPreference ParseScalingBackendPreference(
    const std::string &raw,
    studiocast::video::ScalingBackendPreference fallback) {
  auto value = studiocast::util::TrimCopy(raw);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  if (value == "cpu")
    return studiocast::video::ScalingBackendPreference::cpu;
  if (value == "gpu")
    return studiocast::video::ScalingBackendPreference::gpu;
  if (value == "auto" || value == "auto_select" || value == "autoselect")
    return studiocast::video::ScalingBackendPreference::auto_select;
  return fallback;
}

std::string ScalingBackendPreferenceToString(
    studiocast::video::ScalingBackendPreference value) {
  switch (value) {
  case studiocast::video::ScalingBackendPreference::cpu:
    return "cpu";
  case studiocast::video::ScalingBackendPreference::gpu:
    return "gpu";
  default:
    return "auto";
  }
}

} // namespace

studiocast::video::VirtualCameraServiceConfig
ToVideoServiceConfig(const DaemonConfig &s) {
  studiocast::video::VirtualCameraServiceConfig cfg;
  cfg.enabled = s.video_enabled;
  cfg.pipeline.input_device = s.video_input_device;
  cfg.pipeline.output_device = s.video_output_device;
  cfg.pipeline.capture_mode = s.video_capture_mode;
  cfg.pipeline.width = s.video_width;
  cfg.pipeline.height = s.video_height;
  cfg.pipeline.fps = s.video_fps;
  cfg.pipeline.output_format = s.video_output_format;
  cfg.pipeline.prefer_mjpeg = s.video_prefer_mjpeg;
  cfg.pipeline.scaling_backend = ParseScalingBackendPreference(
      s.video_scaling_backend,
      studiocast::video::ScalingBackendPreference::auto_select);
  cfg.pipeline.compute_backend =
      studiocast::video::ParseComputeBackendPreferenceOr(
          s.video_compute_backend,
          studiocast::video::ComputeBackendPreference::auto_select);
  cfg.pipeline.allow_cpu_resize = s.video_allow_cpu_resize;
  cfg.pipeline.effects = s.video_effects;

  cfg.consumer_poll_ms = s.consumer_poll_ms;
  cfg.start_grace_ms = s.start_grace_ms;
  cfg.stop_grace_ms = s.stop_grace_ms;
  cfg.min_run_ms = s.min_run_ms;
  cfg.always_on = s.always_on;
  return cfg;
}

void ApplyVideoServiceConfigToDaemonConfig(
    const studiocast::video::VirtualCameraServiceConfig &cfg,
    DaemonConfig *out) {
  if (!out)
    return;
  out->video_enabled = cfg.enabled;
  out->video_input_device = cfg.pipeline.input_device;
  out->video_output_device = cfg.pipeline.output_device;
  out->video_capture_mode = cfg.pipeline.capture_mode;
  out->video_width = cfg.pipeline.width;
  out->video_height = cfg.pipeline.height;
  out->video_fps = cfg.pipeline.fps;
  out->video_output_format = cfg.pipeline.output_format;
  out->video_prefer_mjpeg = cfg.pipeline.prefer_mjpeg;
  out->video_scaling_backend =
      ScalingBackendPreferenceToString(cfg.pipeline.scaling_backend);
  out->video_compute_backend =
      studiocast::video::ComputeBackendPreferenceToString(
          cfg.pipeline.compute_backend);
  out->video_allow_cpu_resize = cfg.pipeline.allow_cpu_resize;
  out->video_effects = cfg.pipeline.effects;

  out->consumer_poll_ms = cfg.consumer_poll_ms;
  out->start_grace_ms = cfg.start_grace_ms;
  out->stop_grace_ms = cfg.stop_grace_ms;
  out->min_run_ms = cfg.min_run_ms;
  out->always_on = cfg.always_on;
}

studiocast::audio::VirtualAudioServiceConfig
ToAudioServiceConfig(const DaemonConfig &s) {
  studiocast::audio::VirtualAudioServiceConfig cfg;
  cfg.enabled = s.audio_enabled;
  cfg.create_virtual_mic = s.audio_create_virtual_mic;
  cfg.create_virtual_speakers = s.audio_create_virtual_speakers;
  cfg.speakers_enabled = s.audio_speakers_enabled;
  cfg.speaker_target_sink = s.audio_speaker_target_sink;
  cfg.speaker_latency_ms = s.audio_speaker_latency_ms;
  cfg.source_name = s.audio_source;
  cfg.effects = s.audio_effects;
  return cfg;
}

void ApplyAudioServiceConfigToDaemonConfig(
    const studiocast::audio::VirtualAudioServiceConfig &cfg,
    DaemonConfig *out) {
  if (!out)
    return;
  out->audio_enabled = cfg.enabled;
  out->audio_create_virtual_mic = cfg.create_virtual_mic;
  out->audio_create_virtual_speakers = cfg.create_virtual_speakers;
  out->audio_speakers_enabled = cfg.speakers_enabled;
  out->audio_speaker_target_sink = cfg.speaker_target_sink;
  out->audio_speaker_latency_ms = cfg.speaker_latency_ms;
  out->audio_source = cfg.source_name;
  out->audio_effects = cfg.effects;
}

} // namespace studiocast::config
