#include <iostream>
#include <string>
#include <vector>

#include "core/audio/effects/broadcast_audio_effects.h"
#include "core/audio/effects/broadcast_audio_effects_json.h"
#include "core/video/effects/broadcast_effects.h"
#include "core/video/effects/broadcast_effects_json.h"

namespace {

bool Expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

studiocast::audio::effects::BroadcastAudioEffects SeedAudioEffects() {
  using studiocast::audio::effects::AudioEffectsEnginePreference;
  using studiocast::audio::effects::SuperresMode;

  studiocast::audio::effects::BroadcastAudioEffects effects;
  effects.engine = AudioEffectsEnginePreference::kOpenSource;
  effects.microphone.model_id = "mic-model-id";
  effects.microphone.model_path = "/models/mic.onnx";
  effects.microphone.noise_removal_enabled = true;
  effects.microphone.room_echo_removal_enabled = false;
  effects.microphone.strength = 41;
  effects.microphone.studio_voice_enabled = false;
  effects.microphone.aec.enabled = true;
  effects.microphone.aec.reference_source = "alsa_output.monitor";
  effects.microphone.superres.enabled = true;
  effects.microphone.superres.mode = SuperresMode::k8kTo16k;

  effects.speaker.model_id = "speaker-model-id";
  effects.speaker.model_path = "/models/speaker.onnx";
  effects.speaker.noise_removal_enabled = true;
  effects.speaker.room_echo_removal_enabled = true;
  effects.speaker.strength = 64;
  effects.speaker.superres.enabled = true;
  effects.speaker.superres.mode = SuperresMode::k16kTo48k;

  return effects;
}

bool TestAudioPatchPreservesHiddenFields() {
  using studiocast::audio::effects::ApplyBroadcastAudioEffectsPatchJsonText;

  const auto original = SeedAudioEffects();
  auto patched = original;

  std::vector<std::string> warnings;
  std::string error;
  const bool ok = ApplyBroadcastAudioEffectsPatchJsonText(
      R"({"microphone":{"strength":77}})", &patched, {}, &warnings, &error);

  return Expect(ok, error.c_str()) &&
         Expect(patched.microphone.strength == 77,
                "audio patch should update visible microphone strength") &&
         Expect(patched.engine == original.engine,
                "audio patch should preserve hidden engine") &&
         Expect(patched.microphone.model_id == original.microphone.model_id,
                "audio patch should preserve microphone model_id") &&
         Expect(patched.microphone.model_path == original.microphone.model_path,
                "audio patch should preserve microphone model_path") &&
         Expect(patched.microphone.aec.enabled ==
                    original.microphone.aec.enabled,
                "audio patch should preserve microphone AEC enabled") &&
         Expect(patched.microphone.aec.reference_source ==
                    original.microphone.aec.reference_source,
                "audio patch should preserve microphone AEC reference source") &&
         Expect(patched.microphone.superres.enabled ==
                    original.microphone.superres.enabled,
                "audio patch should preserve microphone superres enabled") &&
         Expect(patched.microphone.superres.mode ==
                    original.microphone.superres.mode,
                "audio patch should preserve microphone superres mode") &&
         Expect(patched.speaker == original.speaker,
                "audio patch should preserve all omitted speaker fields");
}

bool TestAudioPatchFailureIsTransactional() {
  using studiocast::audio::effects::ApplyBroadcastAudioEffectsPatchJsonText;

  const auto original = SeedAudioEffects();
  auto patched = original;

  std::vector<std::string> warnings;
  std::string error;
  const bool ok = ApplyBroadcastAudioEffectsPatchJsonText(
      R"({"microphone":{"strength":120},"speaker":{"strength":1}})",
      &patched, {}, &warnings, &error);

  return Expect(!ok, "invalid audio patch should fail") &&
         Expect(patched == original,
                "failed audio patch should leave effects unchanged");
}

bool TestVideoPersistenceParserAcceptsAutoFrameWithVirtualBackground() {
  using studiocast::video::effects::BroadcastCameraEffects;
  using studiocast::video::effects::BroadcastEffectsJsonParseOptions;
  using studiocast::video::effects::ParseBroadcastCameraEffectsJsonText;
  using studiocast::video::effects::VirtualBackgroundMode;

  const std::string json =
      R"({"schema_version":1,"auto_frame":{"enabled":true},)"
      R"("virtual_background":{"mode":"blur","strength":12}})";

  BroadcastCameraEffects parsed;
  BroadcastEffectsJsonParseOptions options;
  std::vector<std::string> warnings;
  std::string error;
  const bool ok =
      ParseBroadcastCameraEffectsJsonText(json, &parsed, options, &warnings,
                                          &error);

  return Expect(ok, error.c_str()) &&
         Expect(parsed.auto_frame.enabled,
                "video parser should keep auto_frame enabled") &&
         Expect(parsed.virtual_background.mode == VirtualBackgroundMode::blur,
                "video parser should keep virtual_background blur enabled");
}

} // namespace

int main() {
  int failures = 0;

  if (!TestAudioPatchPreservesHiddenFields())
    ++failures;
  if (!TestAudioPatchFailureIsTransactional())
    ++failures;
  if (!TestVideoPersistenceParserAcceptsAutoFrameWithVirtualBackground())
    ++failures;

  if (failures != 0) {
    std::cerr << failures << " effect schema test(s) failed\n";
    return 1;
  }

  std::cout << "effect schema tests passed\n";
  return 0;
}
