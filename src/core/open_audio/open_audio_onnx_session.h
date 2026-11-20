#pragma once

#include <cstdint>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "core/onnx/ort_session.h"

namespace studiocast::open_audio {

// Best-effort ONNX Runtime runtime information.
struct OrtRuntimeInfo {
  std::string version;
  std::vector<std::string> providers;
  bool cuda_provider_present = false;
  bool tensorrt_provider_present = false;
  bool cpu_provider_present = false;
  bool cuda_ep_v2_build = false;
  std::string library_path;
  std::vector<std::string> warnings;
};

// Options for creating an ONNX Runtime session.
struct OrtSessionOptions {
  // If true, attempt to use CUDA EP and fall back to CPU EP if CUDA EP is not
  // available.
  bool prefer_cuda = true;

  // CUDA device id to use when CUDA EP is available.
  int cuda_device_id = 0;
};

// Best-effort model I/O description extracted from the session.
struct OrtSessionInfo {
  bool using_cuda = false;

  std::vector<std::string> advertised_providers;
  bool cuda_provider_advertised = false;
  bool tensorrt_provider_advertised = false;
  bool cpu_provider_advertised = false;
  bool cuda_provider_appended = false;
  bool cuda_provider_usable = false;
  bool cpu_provider_usable = false;
  bool cuda_session_create_failed_fell_back_to_cpu = false;
  std::string active_provider;
  std::string appended_provider;
  std::vector<std::string> appended_providers;

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // Human-friendly strings like: "tensor(float32) shape=[1, -1]".
  std::vector<std::string> input_descriptions;
  std::vector<std::string> output_descriptions;

  // Structured tensor metadata for tool/debugging and engine bindings.
  // Shapes are best-effort (may contain -1 for dynamic dimensions).
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<int> input_elem_types;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int> output_elem_types;

  // Non-fatal warnings collected during session creation (e.g., CUDA EP
  // unavailable).
  std::vector<std::string> warnings;
};

// Thin wrapper around an ONNX Runtime session used by the Open Audio backend.
//
// This is primarily used to validate that the selected model can be
// loaded by ORT and to expose basic introspection for diagnostics/tools.
class OpenAudioOrtSession {
public:
  using PreparedRunStats = studiocast::onnx::OrtSession::PreparedRunStats;
  static OrtRuntimeInfo QueryRuntimeInfo();

  // Create an ORT session for the given model.
  //
  // Returns nullptr on failure and fills `error`.
  static std::unique_ptr<OpenAudioOrtSession>
  Create(const std::filesystem::path &model_path, const OrtSessionOptions &opts,
         OrtSessionInfo *info_out, std::string *error);

  ~OpenAudioOrtSession();

  OpenAudioOrtSession(const OpenAudioOrtSession &) = delete;
  OpenAudioOrtSession &operator=(const OpenAudioOrtSession &) = delete;

  const OrtSessionInfo &info() const;

  // Pre-size reusable run scratch buffers during setup.
  void ReserveRunScratch(std::size_t input_count, std::size_t output_count);

  using OrtRunInput = studiocast::onnx::OrtSession::RunInput;
  using OrtRunOutput = studiocast::onnx::OrtSession::RunOutput;

  // Run an ORT session with pre-allocated input/output tensors.
  //
  // All tensors are assumed to be float32 CPU buffers.
  // Returns false on failure and fills `error`.
  bool Run(const OrtRunInput *inputs, std::size_t input_count,
           const OrtRunOutput *outputs, std::size_t output_count,
           std::string *error);

  // Prepared streaming run. Slot 0/1 correspond to the two bounded recurrent
  // state ping-pong buffer sets. Caller-owned names, shapes, and buffers must
  // remain valid until InvalidatePreparedBindings() or session destruction.
  bool RunPrepared(std::size_t binding_slot, const OrtRunInput *inputs,
                   std::size_t input_count, const OrtRunOutput *outputs,
                   std::size_t output_count, std::string *error);

  void InvalidatePreparedBindings();
  PreparedRunStats prepared_run_stats() const;

  // Convenience helper for waveform-style models with a single float tensor
  // input/output. The first input and first output of the underlying ORT
  // session are used.
  //
  // - input: mono samples
  // - samples: number of mono samples
  // - output: caller-provided buffer
  // - output_capacity: maximum number of samples that can be written to output
  // - output_samples: filled with number of valid samples on success
  //
  // Returns false on failure and fills `error`.
  bool Run1D(const float *input, std::size_t samples, float *output,
             std::size_t output_capacity, std::size_t *output_samples,
             std::string *error);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit OpenAudioOrtSession(std::unique_ptr<Impl> impl);
};

} // namespace studiocast::open_audio
