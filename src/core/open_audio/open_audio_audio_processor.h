#pragma once

#include <atomic>

#include <cstdint>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "core/audio/audio_processor.h"
#include "core/audio/dsp/post_dsp_chain.h"
#include "core/audio/effects/broadcast_audio_effects.h"
#include "core/open_audio/open_audio_onnx_session.h"

namespace studiocast::open_audio {

// Resolved Open Audio model selection for a given effects config.
struct ResolvedOpenAudioModel {
  // Stable model pack ID when resolved from the registry.
  // When resolved from a user-provided path, this may be empty.
  std::string model_id;

  // Human-friendly name (best-effort).
  std::string display_name;

  // Fully resolved ONNX file path.
  std::filesystem::path onnx_path;

  // Optional metadata propagated from model pack (if available).
  // These are best-effort hints for future extensions (e.g., resampling).
  int sample_rate = 0; // 0 = unknown
  int channels = 1;    // expected model channels (usually 1)

  struct AuxInput {
    // Name of the tensor input.
    std::string name;

    // Optional range mapping for user-facing strength (0..1 normalized) to
    // model domain.
    float min_value = 0.0f;
    float max_value = 1.0f;

    // Optional explicit tensor shape for scalar inputs. Default is [1].
    // The engine currently supports only scalar-shaped aux inputs
    // (product(shape)==1).
    std::vector<int64_t> shape;
  };

  struct OnnxIo {
    // Expected frame size (in samples) per inference call at model sample_rate.
    // For the default 10ms framing this is sample_rate / 100 (e.g., 160 @
    // 16kHz).
    int frame_samples = 0;

    // Optional explicit tensor names for the primary waveform I/O.
    std::string audio_input;
    std::string audio_output;

    // Optional explicit state tensor names for streaming models.
    std::vector<std::string> state_inputs;
    std::vector<std::string> state_outputs;

    bool has_strength_input = false;
    AuxInput strength_input{};
  };

  bool has_onnx_io = false;
  OnnxIo onnx_io{};

  // Indicates whether the ONNX file came from a user-specified path.
  bool is_user_path = false;
};

// Resolve the Open Audio model selection for the microphone effects.
//
// Resolution order:
//  1) microphone.model_path (file .onnx or directory containing model.json)
//  2) microphone.model_id (installed pack id)
//  3) default installed pack id
//
// Returns false with an actionable error string if no model can be resolved.
bool ResolveOpenAudioModelForMicrophone(
    const studiocast::audio::effects::BroadcastAudioEffects &fx,
    ResolvedOpenAudioModel *out, std::string *error);

// Resolve the Open Audio model selection for the speaker effects.
//
// Resolution order:
//  1) speaker.model_path (file .onnx or directory containing model.json)
//  2) speaker.model_id (installed pack id)
//  3) default installed pack id
//
// Returns false with an actionable error string if no model can be resolved.
bool ResolveOpenAudioModelForSpeaker(
    const studiocast::audio::effects::BroadcastAudioEffects &fx,
    ResolvedOpenAudioModel *out, std::string *error);

// Open Audio processor (Phase 4 stub).
//
// For Phase 4 this processor is pass-through, but it validates model selection
// and exposes the resolved model path for status/UI.
class OpenAudioAudioProcessor final : public studiocast::audio::AudioProcessor {
public:
  // Creates a processor for microphone effects.
  // Returns nullptr and fills error if model selection cannot be resolved.
  static std::unique_ptr<OpenAudioAudioProcessor> CreateForMicrophone(
      const studiocast::audio::effects::BroadcastAudioEffects &fx,
      ResolvedOpenAudioModel *resolved_out, std::string *error);

  // Same as CreateForMicrophone but allows overriding ORT session options
  // (e.g., CPU-only self-test).
  static std::unique_ptr<OpenAudioAudioProcessor>
  CreateForMicrophoneWithOrtOptions(
      const studiocast::audio::effects::BroadcastAudioEffects &fx,
      const studiocast::open_audio::OrtSessionOptions &ort_opts,
      ResolvedOpenAudioModel *resolved_out, std::string *error);

  // Creates a processor for speaker effects.
  // Returns nullptr and fills error if model selection cannot be resolved.
  static std::unique_ptr<OpenAudioAudioProcessor>
  CreateForSpeaker(const studiocast::audio::effects::BroadcastAudioEffects &fx,
                   ResolvedOpenAudioModel *resolved_out, std::string *error);

  // Same as CreateForSpeaker but allows overriding ORT session options.
  static std::unique_ptr<OpenAudioAudioProcessor>
  CreateForSpeakerWithOrtOptions(
      const studiocast::audio::effects::BroadcastAudioEffects &fx,
      const studiocast::open_audio::OrtSessionOptions &ort_opts,
      ResolvedOpenAudioModel *resolved_out, std::string *error);

  explicit OpenAudioAudioProcessor(ResolvedOpenAudioModel model);
  ~OpenAudioAudioProcessor() override;

  // Update strength / mode without requiring a pipeline restart.
  // Thread-safe for calls from the supervisor thread while Process() runs on
  // the audio thread.
  void UpdateFromMicrophoneConfig(
      const studiocast::audio::effects::BroadcastMicrophoneEffects &mic);

  // Update speaker strength without requiring a pipeline restart.
  void UpdateFromSpeakerConfig(
      const studiocast::audio::effects::BroadcastSpeakerEffects &spk);

  void Reset() override;

  const ResolvedOpenAudioModel &model() const { return model_; }

  // Supervisor/status helpers. These are intended to be read immediately after
  // processor creation, before the processor is handed to AudioPipeline.
  std::string ActiveProviderForStatus() const;
  bool UsingCpuFallbackForStatus() const { return using_cpu_fallback_; }
  std::string LastStartupWarningForStatus() const;

  // StudioCast-side prepared binding counters. These deliberately exclude
  // opaque allocations performed inside ONNX Runtime.
  OpenAudioOrtSession::PreparedRunStats PreparedRunStatsForTesting() const;

  bool Process(const float *in, float *out, std::uint32_t frames,
               std::uint32_t channels, std::string *error) override;

private:
  bool InitializeBindings(std::string *error);

  ResolvedOpenAudioModel model_;

  // Active ORT session (CUDA if available, CPU otherwise). If a CUDA session is
  // active and a CPU fallback was created, we can switch to CPU on the first
  // runtime failure.
  std::unique_ptr<OpenAudioOrtSession> ort_session_cuda_;
  std::unique_ptr<OpenAudioOrtSession> ort_session_cpu_;
  OpenAudioOrtSession *ort_session_active_ = nullptr;
  bool using_cpu_fallback_ = false;

  // Model I/O binding (names, shapes, and optional streaming state buffers).
  std::string audio_input_name_;
  std::string audio_output_name_;
  bool has_strength_input_ = false;
  std::string strength_input_name_;
  std::vector<int64_t> strength_input_shape_;
  std::vector<float> strength_input_buf_;
  float strength_input_min_ = 0.0f;
  float strength_input_max_ = 1.0f;
  std::vector<std::string> state_input_names_;
  std::vector<std::string> state_output_names_;
  std::vector<int64_t> audio_input_shape_;
  std::vector<int64_t> audio_output_shape_;
  std::vector<std::vector<int64_t>> state_shapes_;
  std::vector<std::size_t> state_sizes_;
  std::vector<OpenAudioOrtSession::OrtRunInput> ort_inputs_;
  std::vector<OpenAudioOrtSession::OrtRunOutput> ort_outputs_;

  int state_toggle_ = 0;
  std::vector<std::vector<float>> state_buf_[2];

  // Runtime settings updated from the supervisor thread.
  std::atomic<int> strength_{50};
  std::atomic<bool> studio_voice_enabled_{false};

  // Effect context for strength curve mapping (set from config).
  // 0=noise_removal, 1=room_echo_removal, 2=studio_voice,
  // 3=speaker_noise_removal, 4=speaker_room_echo_removal
  std::atomic<int> strength_mode_{0};

  // Engine configuration derived from model pack metadata.
  int model_sample_rate_ = 48000;
  std::uint32_t model_frame_samples_ = 480;

  // Scratch buffers for mono processing. We convert interleaved input to mono,
  // run the model on mono, and then fan out the processed signal to all
  // channels with wet/dry mixing.
  std::vector<float> mono_in_;
  std::vector<float> mono_out_;

  // Scratch buffer used when preserving stereo via Mid/Side processing in the
  // speaker pipeline.
  std::vector<float> side_;

  std::vector<float> model_in_;
  std::vector<float> model_out_;

  // Resampler state (48k <-> model_sample_rate). Only used when sample rates
  // differ.
  struct Decimator3;
  struct Interpolator3;
  std::unique_ptr<Decimator3> decim3_;
  std::unique_ptr<Interpolator3> interp3_;

  // Sticky warning after a runtime failure; surfaced via AudioPipeline
  // last_error.
  std::string sticky_warning_;
  bool model_disabled_ = false;

  // Post-processing "polish" DSP chain applied to the processed output.
  // This is primarily a safety net to prevent clipping and stabilize output
  // levels (especially for Studio Voice).
  studiocast::audio::dsp::PostDspChain post_dsp_;
};

} // namespace studiocast::open_audio
