#pragma once

#include <cstdint>
#include <string>

#include "core/video/effects/broadcast_effect_contract.h"

namespace studiocast::video::effects {

inline constexpr int kBroadcastEffectsSchemaVersion = 1;

// User-facing preference for effect engine selection.
// Note: CPU is intentionally not represented here.
enum class EffectsEnginePreference {
  auto_select = 0,
  maxine = 1,
  open_cuda = 2,
};

std::string ToString(EffectsEnginePreference v);
bool ParseEffectsEnginePreference(const std::string &s,
                                  EffectsEnginePreference *out);

enum class VirtualBackgroundMode {
  none = 0,
  blur = 1,
  remove = 2,
  replace = 3,
};

std::string ToString(VirtualBackgroundMode v);
bool ParseVirtualBackgroundMode(const std::string &s,
                                VirtualBackgroundMode *out);

struct VirtualBackgroundSettings {
  VirtualBackgroundMode mode = VirtualBackgroundMode::none;

  // Open CUDA model pack selection for matting.
  // Empty = deterministic default (ModelPackRegistry::DefaultModelId()).
  std::string model_id;

  // Used by blur (and future AI effects) as an intensity knob.
  // Interpreted as a blur radius for the current CPU placeholder.
  int strength = 8;

  // Used when mode==replace.
  std::string replace_path;

  // Used when mode==remove or mode==replace.
  // Stored as a string to keep this type Qt-free and easy to serialize.
  // Canonical form: "#RRGGBB".
  std::string remove_color = "#000000";

  // Green screen / matte parameters used by Maxine VFX.
  // Stored as raw values to avoid build-time dependency on NVIDIA headers.
  std::uint32_t greenscreen_mode = 0;
  bool greenscreen_temporal = true;
};

inline bool operator==(const VirtualBackgroundSettings &a,
                       const VirtualBackgroundSettings &b) {
  return a.mode == b.mode && a.model_id == b.model_id &&
         a.strength == b.strength && a.replace_path == b.replace_path &&
         a.remove_color == b.remove_color &&
         a.greenscreen_mode == b.greenscreen_mode &&
         a.greenscreen_temporal == b.greenscreen_temporal;
}

inline bool operator!=(const VirtualBackgroundSettings &a,
                       const VirtualBackgroundSettings &b) {
  return !(a == b);
}

struct AutoFrameSettings {
  bool enabled = false;

  // 0..100-ish user knob (implementation-defined).
  int strength = 50;

  // 0..100-ish smoothing (implementation-defined).
  int smoothing = 50;

  // Fractional extra headroom above the detected subject (0..1).
  float headroom = 0.15f;

  // Optional Open Video model pack override for face detection (YuNet).
  // Empty means "auto" (registry default).
  std::string model_id;
};

inline bool operator==(const AutoFrameSettings &a, const AutoFrameSettings &b) {
  return a.enabled == b.enabled && a.strength == b.strength &&
         a.smoothing == b.smoothing && a.headroom == b.headroom &&
         a.model_id == b.model_id;
}

inline bool operator!=(const AutoFrameSettings &a, const AutoFrameSettings &b) {
  return !(a == b);
}

struct EyeContactSettings {
  bool enabled = false;
  int strength = 50;
  bool look_away_enabled = true;

  // Optional Open Video model pack override for eye contact
  // (gaze_correction_cam). Empty means "auto" (registry default).
  std::string model_id;
};

inline bool operator==(const EyeContactSettings &a,
                       const EyeContactSettings &b) {
  return a.enabled == b.enabled && a.strength == b.strength &&
         a.look_away_enabled == b.look_away_enabled && a.model_id == b.model_id;
}

inline bool operator!=(const EyeContactSettings &a,
                       const EyeContactSettings &b) {
  return !(a == b);
}

struct VideoNoiseRemovalSettings {
  bool enabled = false;
  int strength = 50;

  // Optional Open Video model pack override for video denoise (FastDVDnet).
  // Empty means "auto" (registry default).
  std::string model_id;
};

inline bool operator==(const VideoNoiseRemovalSettings &a,
                       const VideoNoiseRemovalSettings &b) {
  return a.enabled == b.enabled && a.strength == b.strength &&
         a.model_id == b.model_id;
}

inline bool operator!=(const VideoNoiseRemovalSettings &a,
                       const VideoNoiseRemovalSettings &b) {
  return !(a == b);
}

struct VirtualKeyLightSettings {
  bool enabled = false;
  int intensity = 50;

  // In Kelvin (roughly). UI may map presets.
  int temperature = 4500;

  // Preset for contract/IPC (neutral/warm/cool). Stored as numeric code to
  // keep comparisons simple and match the legacy pipeline settings.
  // 0 = neutral, 1 = warm, 2 = cool.
  int temperature_preset = 0;

  // Optional direction control (pan angle, degrees).
  int direction_pan_degrees = 0;

  // Optional HDRI override. Empty = auto/default.
  std::string hdri_path;
};

inline bool operator==(const VirtualKeyLightSettings &a,
                       const VirtualKeyLightSettings &b) {
  return a.enabled == b.enabled && a.intensity == b.intensity &&
         a.temperature == b.temperature &&
         a.temperature_preset == b.temperature_preset &&
         a.direction_pan_degrees == b.direction_pan_degrees &&
         a.hdri_path == b.hdri_path;
}

inline bool operator!=(const VirtualKeyLightSettings &a,
                       const VirtualKeyLightSettings &b) {
  return !(a == b);
}

struct VignetteSettings {
  bool enabled = false;
  // The stable contract/descriptors/GUI and legacy representation all use 35.
  // Reference that canonical source directly so default construction cannot
  // drift from the public schema again.
  int intensity = contract::kVignetteIntensityDefault;

  bool center_on_tracked_face = true;
};

inline bool operator==(const VignetteSettings &a, const VignetteSettings &b) {
  return a.enabled == b.enabled && a.intensity == b.intensity &&
         a.center_on_tracked_face == b.center_on_tracked_face;
}

inline bool operator!=(const VignetteSettings &a, const VignetteSettings &b) {
  return !(a == b);
}

// Canonical, versioned representation of Broadcast-style camera effect
// settings. This type is intended to be used across config persistence, IPC,
// pipeline config, and GUI state.
struct BroadcastCameraEffects {
  int schema_version = kBroadcastEffectsSchemaVersion;

  bool mirror = false;

  // Global engine preference for effects that can run on Maxine.
  EffectsEnginePreference engine = EffectsEnginePreference::auto_select;

  VirtualBackgroundSettings virtual_background{};
  AutoFrameSettings auto_frame{};
  EyeContactSettings eye_contact{};
  VideoNoiseRemovalSettings video_noise_removal{};
  VirtualKeyLightSettings virtual_key_light{};
  VignetteSettings vignette{};
};

inline bool operator==(const BroadcastCameraEffects &a,
                       const BroadcastCameraEffects &b) {
  return a.schema_version == b.schema_version && a.mirror == b.mirror &&
         a.engine == b.engine && a.virtual_background == b.virtual_background &&
         a.auto_frame == b.auto_frame && a.eye_contact == b.eye_contact &&
         a.video_noise_removal == b.video_noise_removal &&
         a.virtual_key_light == b.virtual_key_light && a.vignette == b.vignette;
}

inline bool operator!=(const BroadcastCameraEffects &a,
                       const BroadcastCameraEffects &b) {
  return !(a == b);
}

} // namespace studiocast::video::effects
