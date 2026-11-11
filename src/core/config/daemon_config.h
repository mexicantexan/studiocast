#pragma once

#include <filesystem>
#include <string>

#include "core/audio/effects/broadcast_audio_effects.h"
#include "core/video/effects/broadcast_effects.h"
#include "core/video/video_config_types.h"

namespace studiocast::config {

// Persistent configuration for studiocastd.
// Stored as a simple key=value file under XDG_CONFIG_HOME.
struct DaemonConfig {
  // Video
  bool video_enabled = true;
  std::string video_input_device;  // empty = auto
  std::string video_output_device; // empty = auto

  // Capture mode:
  // - requested: use `video_width`/`video_height`
  // - auto_best: choose a good capture mode automatically; width/height may be
  // <= 0 (sentinel)
  studiocast::video::CaptureMode video_capture_mode =
      studiocast::video::CaptureMode::requested;

  int video_width = 1280;
  int video_height = 720;
  int video_fps = 30;
  studiocast::video::PixelFormat video_output_format =
      studiocast::video::PixelFormat::rgb24;
  bool video_prefer_mjpeg = true;

  // Output scaling backend preference: "cpu" | "gpu" | "auto".
  // "auto" = use GPU scaling when available; otherwise CPU.
  std::string video_scaling_backend = "auto";

  // Video compute backend preference: "auto" | "cpu" | "cuda" | "vulkan".
  std::string video_compute_backend = "auto";

  // Persistent Open Vulkan adapter identity. "auto" keeps hardware-first
  // selection and permits the legacy environment selector as a compatibility
  // override. A stable v1:... identity fails closed when absent or ambiguous.
  std::string video_vulkan_device = "auto";
  bool video_vulkan_allow_cpu = false;

  // Allow CPU resize/scale fallback when output size mismatches cannot be
  // resolved on GPU. Users can opt out from Advanced settings.
  bool video_allow_cpu_resize = true;

  // Audio
  bool audio_enabled = false;
  bool audio_create_virtual_mic = true;

  // Keep a virtual sink named "studiocast_speakers" available (daemon-owned).
  // When enabled, other apps can select this as their output device.
  bool audio_create_virtual_speakers = false;

  // Pass-through speakers routing toggle (Phase 9): routes
  // studiocast_speakers.monitor -> physical sink.
  bool audio_speakers_enabled = false;

  // Optional target sink name for speakers routing. Empty = Pulse default sink.
  std::string audio_speaker_target_sink;

  // Latency (ms) for speakers module-loopback.
  int audio_speaker_latency_ms = 10;
  std::string audio_source; // empty = Pulse default

  // Canonical Broadcast-style audio effects.
  // Persisted as a single JSON blob under `audio.effects.json`.
  studiocast::audio::effects::BroadcastAudioEffects audio_effects{};

  // Canonical Broadcast-style camera effects (video).
  //
  // Persisted as a single JSON blob under `video.effects.json`.
  studiocast::video::effects::BroadcastCameraEffects video_effects{};

  // Service behavior
  int consumer_poll_ms = 250;
  int start_grace_ms = 300;
  int stop_grace_ms = 1000;
  int min_run_ms = 1500;
  bool always_on = false;
};

// ~/.config/studiocast/daemon.conf (respecting XDG_CONFIG_HOME)
std::filesystem::path DaemonConfigPath();

DaemonConfig LoadDaemonConfig();
bool SaveDaemonConfig(const DaemonConfig &s, std::string *error);

} // namespace studiocast::config
