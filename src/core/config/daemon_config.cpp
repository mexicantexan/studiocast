#include "daemon_config.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/util/fs.h"
#include "core/util/strings.h"
#include "core/util/xdg.h"

#include "core/audio/effects/broadcast_audio_effects_json.h"
#include "core/video/broadcast_camera_effects_json.h"
#include "core/video/effects/broadcast_effects_json.h"
#include "core/video/effects/effect_types.h"

namespace fs = std::filesystem;

namespace studiocast::config {
namespace {

std::map<std::string, std::string>
ParseKeyValueFile(const std::string &content) {
  std::map<std::string, std::string> kv;
  for (const auto &lineRaw : studiocast::util::SplitLines(content)) {
    auto line = studiocast::util::TrimCopy(lineRaw);
    if (line.empty())
      continue;
    if (line[0] == '#')
      continue;

    const auto pos = line.find('=');
    if (pos == std::string::npos)
      continue;

    auto key = studiocast::util::TrimCopy(line.substr(0, pos));
    auto val = studiocast::util::TrimCopy(line.substr(pos + 1));
    if (key.empty())
      continue;
    kv[key] = val;
  }
  return kv;
}

bool ParseBool(const std::string &raw, bool fallback) {
  const auto v = studiocast::util::TrimCopy(raw);
  if (v.empty())
    return fallback;
  if (v == "1" || v == "true" || v == "yes" || v == "on")
    return true;
  if (v == "0" || v == "false" || v == "no" || v == "off")
    return false;
  return fallback;
}

int ParseInt(const std::string &raw, int fallback) {
  const auto v = studiocast::util::TrimCopy(raw);
  if (v.empty())
    return fallback;
  return std::atoi(v.c_str());
}

float ParseFloat(const std::string &raw, float fallback) {
  const auto v = studiocast::util::TrimCopy(raw);
  if (v.empty())
    return fallback;
  return static_cast<float>(std::atof(v.c_str()));
}

int ParseKeyLightTemperaturePreset(const std::string &raw, int fallback) {
  auto v = studiocast::util::TrimCopy(raw);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (v.empty())
    return fallback;
  if (v == "0" || v == "neutral")
    return 0;
  if (v == "1" || v == "warm")
    return 1;
  if (v == "2" || v == "cool")
    return 2;
  return fallback;
}

std::string NormalizeScalingBackendPreference(const std::string &raw,
                                              std::string fallback) {
  auto v = studiocast::util::TrimCopy(raw);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (v == "cpu")
    return "cpu";
  if (v == "gpu")
    return "gpu";
  if (v == "auto" || v == "auto_select" || v == "autoselect") {
    return "auto";
  }
  return fallback;
}

std::string NormalizeComputeBackendPreference(const std::string &raw,
                                              std::string fallback) {
  auto v = studiocast::util::TrimCopy(raw);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (v == "cpu")
    return "cpu";
  if (v == "cuda")
    return "cuda";
  if (v == "vulkan")
    return "vulkan";
  if (v == "auto" || v == "auto_select" || v == "autoselect")
    return "auto";
  return fallback;
}

studiocast::video::CaptureMode
ParseCaptureMode(const std::string &raw,
                 studiocast::video::CaptureMode fallback) {
  auto v = studiocast::util::TrimCopy(raw);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (v == "requested")
    return studiocast::video::CaptureMode::requested;
  if (v == "auto" || v == "auto_best" || v == "autobest")
    return studiocast::video::CaptureMode::auto_best;
  return fallback;
}

std::string CaptureModeToString(studiocast::video::CaptureMode v) {
  switch (v) {
  case studiocast::video::CaptureMode::auto_best:
    return "auto_best";
  default:
    return "requested";
  }
}

} // namespace

std::filesystem::path DaemonConfigPath() {
  const auto dir = studiocast::util::StudioCastConfigDir();
  if (dir.empty())
    return {};
  return dir / "daemon.conf";
}

DaemonConfig LoadDaemonConfig() {
  DaemonConfig s;

  const auto path = DaemonConfigPath();
  if (!path.empty()) {
    if (auto content = studiocast::util::ReadTextFile(path.string())) {
      const auto kv = ParseKeyValueFile(*content);

      const bool capture_mode_key_present =
          (kv.find("video.capture_mode") != kv.end()) ||
          (kv.find("video.capture.mode") != kv.end());

      if (auto it = kv.find("video.capture_mode"); it != kv.end()) {
        s.video_capture_mode =
            ParseCaptureMode(it->second, s.video_capture_mode);
      }
      if (auto it = kv.find("video.capture.mode"); it != kv.end()) {
        s.video_capture_mode =
            ParseCaptureMode(it->second, s.video_capture_mode);
      }

      if (auto it = kv.find("video.enabled"); it != kv.end()) {
        s.video_enabled = ParseBool(it->second, s.video_enabled);
      }
      if (auto it = kv.find("video.input"); it != kv.end()) {
        s.video_input_device = it->second;
      }
      if (auto it = kv.find("video.output"); it != kv.end()) {
        s.video_output_device = it->second;
      }
      if (auto it = kv.find("video.width"); it != kv.end()) {
        s.video_width = ParseInt(it->second, s.video_width);
      }
      if (auto it = kv.find("video.height"); it != kv.end()) {
        s.video_height = ParseInt(it->second, s.video_height);
      }
      if (auto it = kv.find("video.fps"); it != kv.end()) {
        s.video_fps = ParseInt(it->second, s.video_fps);
      }
      if (auto it = kv.find("video.output_format"); it != kv.end()) {
        if (const auto parsed = studiocast::video::ParsePixelFormat(it->second))
          s.video_output_format = *parsed;
      }
      if (auto it = kv.find("video.prefer_mjpeg"); it != kv.end()) {
        s.video_prefer_mjpeg = ParseBool(it->second, s.video_prefer_mjpeg);
      }

      if (auto it = kv.find("video.scaling.backend"); it != kv.end()) {
        s.video_scaling_backend = NormalizeScalingBackendPreference(
            it->second,
            NormalizeScalingBackendPreference(s.video_scaling_backend,
                                              "auto"));
      }

      if (auto it = kv.find("video.compute.backend"); it != kv.end()) {
        s.video_compute_backend = NormalizeComputeBackendPreference(
            it->second,
            NormalizeComputeBackendPreference(s.video_compute_backend,
                                              "auto"));
      }
      if (auto it = kv.find("video.compute_backend"); it != kv.end()) {
        s.video_compute_backend = NormalizeComputeBackendPreference(
            it->second,
            NormalizeComputeBackendPreference(s.video_compute_backend,
                                              "auto"));
      }

      if (auto it = kv.find("video.vulkan.device"); it != kv.end()) {
        const auto value = studiocast::util::TrimCopy(it->second);
        s.video_vulkan_device = value.empty() ? "auto" : value;
      }
      if (auto it = kv.find("video.vulkan.allow_cpu"); it != kv.end()) {
        s.video_vulkan_allow_cpu =
            ParseBool(it->second, s.video_vulkan_allow_cpu);
      }

      if (auto it = kv.find("video.scaling.allow_cpu_resize"); it != kv.end()) {
        s.video_allow_cpu_resize =
            ParseBool(it->second, s.video_allow_cpu_resize);
      }

      // Backward-compatible inference: if width/height are set to a
      // non-positive sentinel and no capture mode was specified, treat that as
      // capture auto.
      if (!capture_mode_key_present &&
          (s.video_width <= 0 || s.video_height <= 0)) {
        s.video_capture_mode = studiocast::video::CaptureMode::auto_best;
      }

      // Audio
      if (auto it = kv.find("audio.enabled"); it != kv.end()) {
        s.audio_enabled = ParseBool(it->second, s.audio_enabled);
      }
      if (auto it = kv.find("audio.create_virtual_mic"); it != kv.end()) {
        s.audio_create_virtual_mic =
            ParseBool(it->second, s.audio_create_virtual_mic);
      }
      if (auto it = kv.find("audio.create_virtual_speakers"); it != kv.end()) {
        s.audio_create_virtual_speakers =
            ParseBool(it->second, s.audio_create_virtual_speakers);
      }
      if (auto it = kv.find("audio.speakers.enabled"); it != kv.end()) {
        s.audio_speakers_enabled =
            ParseBool(it->second, s.audio_speakers_enabled);
      }
      if (auto it = kv.find("audio.speakers.target_sink"); it != kv.end()) {
        s.audio_speaker_target_sink = it->second;
      }
      if (auto it = kv.find("audio.speakers.latency_ms"); it != kv.end()) {
        s.audio_speaker_latency_ms =
            ParseInt(it->second, s.audio_speaker_latency_ms);
      }
      if (auto it = kv.find("audio.source"); it != kv.end()) {
        s.audio_source = it->second;
      }

      // Canonical audio effects persistence.
      if (auto it = kv.find("audio.effects.json");
          it != kv.end() && !it->second.empty()) {
        studiocast::audio::effects::BroadcastAudioEffects parsed;
        studiocast::audio::effects::BroadcastAudioEffectsJsonParseOptions
            options;
        options.allow_unknown_keys = true; // tolerate forward/backward drift
        std::vector<std::string> warnings;
        std::string parse_error;
        if (studiocast::audio::effects::ParseBroadcastAudioEffectsJsonText(
                it->second, &parsed, options, &warnings, &parse_error)) {
          s.audio_effects = parsed;
        }
      }

      // Canonical effects persistence: prefer `video.effects.json`.
      bool effects_loaded = false;
      if (auto it = kv.find("video.effects.json");
          it != kv.end() && !it->second.empty()) {
        studiocast::video::effects::BroadcastCameraEffects parsed;
        studiocast::video::effects::BroadcastEffectsJsonParseOptions options;
        options.allow_unknown_keys = true; // tolerate forward/backward drift
        options.allow_compat_keys = true;
        std::vector<std::string> warnings;
        std::string parse_error;
        if (studiocast::video::effects::ParseBroadcastCameraEffectsJsonText(
                it->second, &parsed, options, &warnings, &parse_error)) {
          s.video_effects = parsed;
          effects_loaded = true;
        }
      }

      // Migration: when the JSON blob is absent (or failed to parse), translate
      // legacy keys.
      if (!effects_loaded) {
        auto fx = s.video_effects;

        // Mirror (legacy top-level key).
        if (auto it = kv.find("video.mirror"); it != kv.end()) {
          fx.mirror = ParseBool(it->second, fx.mirror);
        }

        // Engine preference.
        if (auto it = kv.find("video.effects.engine"); it != kv.end()) {
          std::string raw = studiocast::util::TrimCopy(it->second);
          std::transform(raw.begin(), raw.end(), raw.begin(),
                         [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                         });
          fx.engine =
              (raw == "maxine")
                  ? studiocast::video::effects::EffectsEnginePreference::maxine
                  : studiocast::video::effects::EffectsEnginePreference::
                        auto_select;
        } else if (auto it2 = kv.find("video.background_backend");
                   it2 != kv.end()) {
          // Legacy backend selection: only `maxine` is honored; all other
          // legacy values map to `auto_select`.
          studiocast::video::effects::EffectBackend legacy =
              studiocast::video::effects::EffectBackend::auto_select;
          if (studiocast::video::effects::ParseEffectBackend(it2->second,
                                                             &legacy) &&
              legacy == studiocast::video::effects::EffectBackend::maxine) {
            fx.engine =
                studiocast::video::effects::EffectsEnginePreference::maxine;
          }
        }

        // Background + auto-frame migration.
        const bool vb_has_any_new =
            (kv.find("video.effects.virtual_background.mode") != kv.end()) ||
            (kv.find("video.effects.virtual_background.blur_strength") !=
             kv.end()) ||
            (kv.find("video.effects.virtual_background.remove_color") !=
             kv.end()) ||
            (kv.find("video.effects.virtual_background.replace_path") !=
             kv.end());
        const bool af_has_any_new =
            (kv.find("video.effects.auto_frame.enabled") != kv.end()) ||
            (kv.find("video.effects.auto_frame.zoom") != kv.end()) ||
            (kv.find("video.effects.auto_frame.smoothing") != kv.end()) ||
            (kv.find("video.effects.auto_frame.headroom") != kv.end());

        const bool legacy_bg_has_any =
            (kv.find("video.background") != kv.end()) ||
            (kv.find("video.background_strength") != kv.end()) ||
            (kv.find("video.background_remove_color") != kv.end()) ||
            (kv.find("video.background_replace_image") != kv.end()) ||
            (kv.find("video.auto_frame_strength") != kv.end()) ||
            (kv.find("video.auto_frame_smoothing") != kv.end()) ||
            (kv.find("video.auto_frame_headroom") != kv.end());

        std::string vb_mode_raw;
        int vb_strength = fx.virtual_background.strength;
        std::string vb_remove_color = fx.virtual_background.remove_color;
        std::string vb_replace_path = fx.virtual_background.replace_path;
        bool af_enabled = fx.auto_frame.enabled;
        int af_strength = fx.auto_frame.strength;
        int af_smoothing = fx.auto_frame.smoothing;
        float af_headroom = fx.auto_frame.headroom;

        if (auto it = kv.find("video.effects.virtual_background.mode");
            it != kv.end()) {
          vb_mode_raw = it->second;
        }
        if (auto it = kv.find("video.effects.virtual_background.blur_strength");
            it != kv.end()) {
          vb_strength = ParseInt(it->second, vb_strength);
        }
        if (auto it = kv.find("video.effects.virtual_background.remove_color");
            it != kv.end()) {
          vb_remove_color = it->second;
        }
        if (auto it = kv.find("video.effects.virtual_background.replace_path");
            it != kv.end()) {
          vb_replace_path = it->second;
        }

        if (auto it = kv.find("video.effects.auto_frame.enabled");
            it != kv.end()) {
          af_enabled = ParseBool(it->second, af_enabled);
        }
        if (auto it = kv.find("video.effects.auto_frame.zoom");
            it != kv.end()) {
          af_strength = ParseInt(it->second, af_strength);
        }
        if (auto it = kv.find("video.effects.auto_frame.smoothing");
            it != kv.end()) {
          af_smoothing = ParseInt(it->second, af_smoothing);
        }
        if (auto it = kv.find("video.effects.auto_frame.headroom");
            it != kv.end()) {
          af_headroom = ParseFloat(it->second, af_headroom);
        }

        // Legacy background keys only migrate forward if the new
        // virtual_background/auto_frame keys are absent.
        if (legacy_bg_has_any && !vb_has_any_new && !af_has_any_new) {
          std::string legacy_bg_raw;
          if (auto it = kv.find("video.background"); it != kv.end())
            legacy_bg_raw = it->second;
          if (legacy_bg_raw.empty())
            legacy_bg_raw = "none";

          studiocast::video::effects::BackgroundEffect legacy_bg =
              studiocast::video::effects::BackgroundEffect::none;
          (void)studiocast::video::effects::ParseBackgroundEffect(legacy_bg_raw,
                                                                  &legacy_bg);

          if (legacy_bg ==
              studiocast::video::effects::BackgroundEffect::auto_frame) {
            af_enabled = true;
            if (auto it = kv.find("video.auto_frame_strength");
                it != kv.end()) {
              af_strength = ParseInt(it->second, af_strength);
            }
            if (auto it = kv.find("video.auto_frame_smoothing");
                it != kv.end()) {
              af_smoothing = ParseInt(it->second, af_smoothing);
            }
            if (auto it = kv.find("video.auto_frame_headroom");
                it != kv.end()) {
              af_headroom = ParseFloat(it->second, af_headroom);
            }
            vb_mode_raw = "none";
          } else {
            af_enabled = false;
            vb_mode_raw = studiocast::video::effects::ToString(legacy_bg);
            if (auto it = kv.find("video.background_strength");
                it != kv.end()) {
              vb_strength = ParseInt(it->second, vb_strength);
            }
            if (auto it = kv.find("video.background_remove_color");
                it != kv.end()) {
              vb_remove_color = it->second;
            }
            if (auto it = kv.find("video.background_replace_image");
                it != kv.end()) {
              vb_replace_path = it->second;
            }
          }
        }

        fx.auto_frame.enabled = af_enabled;
        fx.auto_frame.strength = std::max(0, std::min(100, af_strength));
        fx.auto_frame.smoothing = std::max(0, std::min(100, af_smoothing));
        fx.auto_frame.headroom = std::max(0.0f, std::min(1.0f, af_headroom));

        // Background mutex: auto-frame wins.
        if (fx.auto_frame.enabled) {
          fx.virtual_background.mode =
              studiocast::video::effects::VirtualBackgroundMode::none;
        } else {
          studiocast::video::effects::VirtualBackgroundMode m =
              studiocast::video::effects::VirtualBackgroundMode::none;
          std::string raw = studiocast::util::TrimCopy(vb_mode_raw);
          std::transform(raw.begin(), raw.end(), raw.begin(),
                         [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                         });
          if (raw == "auto_frame") {
            m = studiocast::video::effects::VirtualBackgroundMode::none;
          } else {
            (void)studiocast::video::effects::ParseVirtualBackgroundMode(raw,
                                                                         &m);
          }
          fx.virtual_background.mode = m;
        }
        fx.virtual_background.strength = std::max(1, std::min(64, vb_strength));
        fx.virtual_background.remove_color = vb_remove_color;
        fx.virtual_background.replace_path = vb_replace_path;

        // Video noise removal.
        if (auto it = kv.find("video.effects.video_noise_removal.enabled");
            it != kv.end()) {
          fx.video_noise_removal.enabled =
              ParseBool(it->second, fx.video_noise_removal.enabled);
        }
        if (auto it = kv.find("video.effects.video_noise_removal.strength");
            it != kv.end()) {
          fx.video_noise_removal.strength = std::max(
              0, std::min(100, ParseInt(it->second,
                                        fx.video_noise_removal.strength)));
        }

        // Virtual key light (new keys preferred, legacy keys as fallback).
        if (auto it = kv.find("video.effects.virtual_key_light.enabled");
            it != kv.end()) {
          fx.virtual_key_light.enabled =
              ParseBool(it->second, fx.virtual_key_light.enabled);
        } else if (auto itLegacy = kv.find("video.virtual_key_light");
                   itLegacy != kv.end()) {
          fx.virtual_key_light.enabled =
              ParseBool(itLegacy->second, fx.virtual_key_light.enabled);
        }

        if (auto it = kv.find("video.effects.virtual_key_light.intensity");
            it != kv.end()) {
          fx.virtual_key_light.intensity = std::max(
              0, std::min(100, ParseInt(it->second,
                                        fx.virtual_key_light.intensity)));
        } else if (auto itLegacy = kv.find("video.virtual_key_light_intensity");
                   itLegacy != kv.end()) {
          fx.virtual_key_light.intensity = std::max(
              0, std::min(100, ParseInt(itLegacy->second,
                                        fx.virtual_key_light.intensity)));
        }

        std::string temp_preset_raw;
        if (auto it =
                kv.find("video.effects.virtual_key_light.temperature_preset");
            it != kv.end()) {
          temp_preset_raw = it->second;
        } else if (auto itLegacy =
                       kv.find("video.virtual_key_light_temperature");
                   itLegacy != kv.end()) {
          temp_preset_raw = itLegacy->second;
        }
        if (!temp_preset_raw.empty()) {
          fx.virtual_key_light.temperature_preset =
              ParseKeyLightTemperaturePreset(
                  temp_preset_raw, fx.virtual_key_light.temperature_preset);
          // Match KelvinFromPreset() in broadcast_effects_json.cpp.
          switch (fx.virtual_key_light.temperature_preset) {
          case 1:
            fx.virtual_key_light.temperature = 3200;
            break;
          case 2:
            fx.virtual_key_light.temperature = 6500;
            break;
          default:
            fx.virtual_key_light.temperature = 4500;
            break;
          }
        }

        if (auto it = kv.find("video.effects.virtual_key_light.pan");
            it != kv.end()) {
          fx.virtual_key_light.direction_pan_degrees = std::max(
              -180,
              std::min(180,
                       ParseInt(it->second,
                                fx.virtual_key_light.direction_pan_degrees)));
        } else if (auto itLegacy = kv.find("video.virtual_key_light_pan");
                   itLegacy != kv.end()) {
          fx.virtual_key_light.direction_pan_degrees = std::max(
              -180,
              std::min(180,
                       ParseInt(itLegacy->second,
                                fx.virtual_key_light.direction_pan_degrees)));
        }

        if (auto it = kv.find("video.effects.virtual_key_light.hdri_path");
            it != kv.end()) {
          fx.virtual_key_light.hdri_path = it->second;
        } else if (auto itLegacy = kv.find("video.virtual_key_light_hdri");
                   itLegacy != kv.end()) {
          fx.virtual_key_light.hdri_path = itLegacy->second;
        }

        // Eye contact (new keys preferred, legacy keys as fallback).
        if (auto it = kv.find("video.effects.eye_contact.enabled");
            it != kv.end()) {
          fx.eye_contact.enabled =
              ParseBool(it->second, fx.eye_contact.enabled);
        } else if (auto itLegacy = kv.find("video.eye_contact");
                   itLegacy != kv.end()) {
          fx.eye_contact.enabled =
              ParseBool(itLegacy->second, fx.eye_contact.enabled);
        }

        if (auto it = kv.find("video.effects.eye_contact.strength");
            it != kv.end()) {
          fx.eye_contact.strength = std::max(
              0, std::min(100, ParseInt(it->second, fx.eye_contact.strength)));
        } else if (auto itLegacy = kv.find("video.eye_contact_strength");
                   itLegacy != kv.end()) {
          fx.eye_contact.strength =
              std::max(0, std::min(100, ParseInt(itLegacy->second,
                                                 fx.eye_contact.strength)));
        }

        if (auto it = kv.find("video.effects.eye_contact.look_away");
            it != kv.end()) {
          fx.eye_contact.look_away_enabled =
              ParseBool(it->second, fx.eye_contact.look_away_enabled);
        } else if (auto itLegacy = kv.find("video.eye_contact_look_away");
                   itLegacy != kv.end()) {
          fx.eye_contact.look_away_enabled =
              ParseBool(itLegacy->second, fx.eye_contact.look_away_enabled);
        }

        // Vignette (new keys preferred, legacy keys as fallback).
        if (auto it = kv.find("video.effects.vignette.enabled");
            it != kv.end()) {
          fx.vignette.enabled = ParseBool(it->second, fx.vignette.enabled);
        } else if (auto itLegacy = kv.find("video.vignette");
                   itLegacy != kv.end()) {
          fx.vignette.enabled =
              ParseBool(itLegacy->second, fx.vignette.enabled);
        }

        if (auto it = kv.find("video.effects.vignette.intensity");
            it != kv.end()) {
          fx.vignette.intensity = std::max(
              0, std::min(100, ParseInt(it->second, fx.vignette.intensity)));
        } else if (auto itLegacy = kv.find("video.vignette_intensity");
                   itLegacy != kv.end()) {
          fx.vignette.intensity = std::max(
              0,
              std::min(100, ParseInt(itLegacy->second, fx.vignette.intensity)));
        }

        if (auto it = kv.find("video.effects.vignette.center_on_face");
            it != kv.end()) {
          fx.vignette.center_on_tracked_face =
              ParseBool(it->second, fx.vignette.center_on_tracked_face);
        } else if (auto itLegacy = kv.find("video.vignette_center_on_face");
                   itLegacy != kv.end()) {
          fx.vignette.center_on_tracked_face =
              ParseBool(itLegacy->second, fx.vignette.center_on_tracked_face);
        }

        s.video_effects = fx;
      }

      if (auto it = kv.find("service.consumer_poll_ms"); it != kv.end()) {
        s.consumer_poll_ms = ParseInt(it->second, s.consumer_poll_ms);
      }
      if (auto it = kv.find("service.start_grace_ms"); it != kv.end()) {
        s.start_grace_ms = ParseInt(it->second, s.start_grace_ms);
      }
      if (auto it = kv.find("service.stop_grace_ms"); it != kv.end()) {
        s.stop_grace_ms = ParseInt(it->second, s.stop_grace_ms);
      }
      if (auto it = kv.find("service.min_run_ms"); it != kv.end()) {
        s.min_run_ms = ParseInt(it->second, s.min_run_ms);
      }
      if (auto it = kv.find("service.always_on"); it != kv.end()) {
        s.always_on = ParseBool(it->second, s.always_on);
      }
    }
  }

  return s;
}

bool SaveDaemonConfig(const DaemonConfig &s, std::string *error) {
  const auto path = DaemonConfigPath();
  if (path.empty()) {
    if (error)
      *error =
          "DaemonConfigPath() is empty (HOME/XDG_CONFIG_HOME not available).";
    return false;
  }

  std::error_code ec;
  fs::create_directories(path.parent_path(), ec);
  if (ec) {
    if (error)
      *error = "Failed to create config dir: " + ec.message();
    return false;
  }

  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    if (error)
      *error = "Failed to open daemon config for writing: " + path.string();
    return false;
  }

  out << "# StudioCast daemon (studiocastd) configuration\n";
  out << "# This file is managed by the StudioCast GUI / studiocastctl.\n\n";

  out << "video.enabled = " << (s.video_enabled ? "true" : "false") << "\n";
  if (!s.video_input_device.empty())
    out << "video.input = " << s.video_input_device << "\n";
  if (!s.video_output_device.empty())
    out << "video.output = " << s.video_output_device << "\n";
  out << "video.capture_mode = " << CaptureModeToString(s.video_capture_mode)
      << "\n";
  out << "video.width = " << s.video_width << "\n";
  out << "video.height = " << s.video_height << "\n";
  out << "video.fps = " << s.video_fps << "\n";
  out << "video.output_format = "
      << studiocast::video::PixelFormatName(s.video_output_format) << "\n";
  out << "video.prefer_mjpeg = " << (s.video_prefer_mjpeg ? "true" : "false")
      << "\n";
  out << "video.scaling.backend = " << s.video_scaling_backend << "\n";
  out << "video.compute.backend = " << s.video_compute_backend << "\n";
  out << "video.vulkan.device = " << s.video_vulkan_device << "\n";
  out << "video.vulkan.allow_cpu = "
      << (s.video_vulkan_allow_cpu ? "true" : "false") << "\n";
  out << "video.scaling.allow_cpu_resize = "
      << (s.video_allow_cpu_resize ? "true" : "false") << "\n";
  out << "\n";

  out << "# Audio\n";
  out << "audio.enabled = " << (s.audio_enabled ? "true" : "false") << "\n";
  out << "audio.create_virtual_mic = "
      << (s.audio_create_virtual_mic ? "true" : "false") << "\n";
  out << "audio.create_virtual_speakers = "
      << (s.audio_create_virtual_speakers ? "true" : "false") << "\n";
  out << "audio.speakers.enabled = "
      << (s.audio_speakers_enabled ? "true" : "false") << "\n";
  if (!s.audio_speaker_target_sink.empty()) {
    out << "audio.speakers.target_sink = " << s.audio_speaker_target_sink
        << "\n";
  }
  out << "audio.speakers.latency_ms = " << s.audio_speaker_latency_ms << "\n";
  if (!s.audio_source.empty())
    out << "audio.source = " << s.audio_source << "\n";
  out << "\n";

  out << "# Canonical audio effects (Broadcast schema)\n";
  out << "# Single-line JSON blob, managed by the StudioCast GUI / "
         "studiocastctl.\n";
  out << "audio.effects.json = "
      << studiocast::audio::effects::BroadcastAudioEffectsToJson(
             s.audio_effects)
      << "\n\n";

  out << "# Canonical video effects (Broadcast schema)\n";
  out << "# Single-line JSON blob, managed by the StudioCast GUI / "
         "studiocastctl.\n";
  out << "video.effects.json = "
      << studiocast::video::effects::BroadcastCameraEffectsToJson(
             s.video_effects)
      << "\n\n";

  out << "service.consumer_poll_ms = " << s.consumer_poll_ms << "\n";
  out << "service.start_grace_ms = " << s.start_grace_ms << "\n";
  out << "service.stop_grace_ms = " << s.stop_grace_ms << "\n";
  out << "service.min_run_ms = " << s.min_run_ms << "\n";
  out << "service.always_on = " << (s.always_on ? "true" : "false") << "\n";

  return true;
}

} // namespace studiocast::config
