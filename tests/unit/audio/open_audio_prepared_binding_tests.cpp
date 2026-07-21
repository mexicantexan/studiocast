#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include "core/audio/effects/broadcast_audio_effects.h"
#include "core/onnx/ort_session.h"
#include "core/open_audio/open_audio_audio_processor.h"

namespace {

[[maybe_unused]] bool Expect(bool condition, const std::string &message) {
  if (condition)
    return true;
  std::cerr << message << "\n";
  return false;
}

class TempModelPack {
public:
  TempModelPack() {
    dir_ = std::filesystem::temp_directory_path() /
           ("studiocast-open-audio-binding-" +
            std::to_string(static_cast<long long>(::getpid())));
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
    std::filesystem::create_directories(dir_, ec);
    if (ec) {
      error_ = ec.message();
      return;
    }

    static constexpr char kModelHex[] =
        "0808120f73747564696f636173742d746573743adb010a1f0a08617564696f5f696e120961756469"
        "6f5f6f757422084964656e746974790a1f0a0873746174655f696e120973746174655f6f75742208"
        "4964656e74697479121773747564696f636173745f62696e64696e675f746573745a210a08617564"
        "696f5f696e12150a130801120f0a0208010a09120773616d706c65735a1a0a0873746174655f696e"
        "120e0a0c080112080a0208010a02080462220a09617564696f5f6f757412150a130801120f0a0208"
        "010a09120773616d706c6573621b0a0973746174655f6f7574120e0a0c080112080a0208010a0208"
        "0442040a00100d";
    const auto hex_digit = [](char c) -> unsigned char {
      return static_cast<unsigned char>(c <= '9' ? c - '0' : c - 'a' + 10);
    };
    std::vector<unsigned char> model_bytes;
    model_bytes.reserve((sizeof(kModelHex) - 1) / 2);
    for (std::size_t i = 0; i + 1 < sizeof(kModelHex) - 1; i += 2) {
      model_bytes.push_back(static_cast<unsigned char>(
          (hex_digit(kModelHex[i]) << 4u) | hex_digit(kModelHex[i + 1])));
    }

    const auto model_path = dir_ / "model.onnx";
    std::ofstream model(model_path, std::ios::binary);
    model.write(reinterpret_cast<const char *>(model_bytes.data()),
                static_cast<std::streamsize>(model_bytes.size()));
    model.close();

    std::ofstream manifest(dir_ / "model.json");
    manifest << R"({
  "id": "binding-test",
  "display_name": "Prepared binding test",
  "onnx_filename": "model.onnx",
  "sample_rate": 48000,
  "channels": 1,
  "onnx_io": {
    "frame_samples": 480,
    "audio_input": "audio_in",
    "audio_output": "audio_out",
    "state_inputs": ["state_in"],
    "state_outputs": ["state_out"]
  }
})";
    manifest.close();
    if (!model || !manifest)
      error_ = "failed to write model fixture";
  }

  ~TempModelPack() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }

  const std::filesystem::path &dir() const { return dir_; }
  std::filesystem::path model_path() const { return dir_ / "model.onnx"; }
  const std::string &error() const { return error_; }

private:
  std::filesystem::path dir_;
  std::string error_;
};

#if STUDIOCAST_HAVE_ONNXRUNTIME
bool RunPrepared(studiocast::onnx::OrtSession *session, std::size_t slot,
                 std::vector<float> *audio_in,
                 std::vector<float> *audio_out,
                 std::vector<float> *state_in,
                 std::vector<float> *state_out,
                 const std::array<int64_t, 2> &audio_shape,
                 std::string *error) {
  const std::array<int64_t, 2> state_shape{1, 4};
  const std::array<studiocast::onnx::OrtSession::RunInput, 2> inputs{{
      {"audio_in", audio_in->data(), audio_in->size(), audio_shape.data(),
       audio_shape.size()},
      {"state_in", state_in->data(), state_in->size(), state_shape.data(),
       state_shape.size()},
  }};
  const std::array<studiocast::onnx::OrtSession::RunOutput, 2> outputs{{
      {"audio_out", audio_out->data(), audio_out->size(), audio_shape.data(),
       audio_shape.size()},
      {"state_out", state_out->data(), state_out->size(), state_shape.data(),
       state_shape.size()},
  }};
  return session->RunCpuPrepared(slot, inputs.data(), inputs.size(),
                                 outputs.data(), outputs.size(), error);
}

bool TestPreparedCpuBindings() {
  TempModelPack pack;
  if (!Expect(pack.error().empty(), pack.error()))
    return false;

  studiocast::onnx::OrtSessionOptions opts;
  opts.prefer_cuda = false;
  studiocast::onnx::OrtSessionInfo info;
  std::string error;
  auto session = studiocast::onnx::OrtSession::Create(pack.model_path(), opts,
                                                       &info, &error);
  if (!Expect(session != nullptr, "CPU session creation failed: " + error))
    return false;
  session->ReserveRunScratch(2, 2);

  std::array<std::vector<float>, 2> audio_in{
      std::vector<float>(480, 0.25f), std::vector<float>(480, -0.5f)};
  std::array<std::vector<float>, 2> audio_out{
      std::vector<float>(480), std::vector<float>(480)};
  std::array<std::vector<float>, 2> state{
      std::vector<float>(4, 1.0f), std::vector<float>(4)};
  const std::array<int64_t, 2> shape480{1, 480};

  if (!RunPrepared(session.get(), 0, &audio_in[0], &audio_out[0], &state[0],
                   &state[1], shape480, &error) ||
      !RunPrepared(session.get(), 1, &audio_in[1], &audio_out[1], &state[1],
                   &state[0], shape480, &error)) {
    return Expect(false, "initial prepared runs failed: " + error);
  }
  auto stats = session->prepared_run_stats();
  bool ok = true;
  ok &= Expect(stats.binding_rebuilds == 2,
               "two ping-pong slots must build exactly twice");
  ok &= Expect(stats.tensor_wrapper_constructions == 8,
               "two slots must construct exactly eight tensor wrappers");
  const auto warm_allocation_requests =
      stats.application_binding_allocation_requests;

  for (std::size_t i = 0; i < 128; ++i) {
    const std::size_t slot = i & 1u;
    if (!RunPrepared(session.get(), slot, &audio_in[slot], &audio_out[slot],
                     &state[slot], &state[1 - slot], shape480, &error)) {
      return Expect(false, "stable prepared run failed: " + error);
    }
  }
  stats = session->prepared_run_stats();
  ok &= Expect(stats.binding_rebuilds == 2,
               "stable pointer/shape contracts rebuilt bindings");
  ok &= Expect(stats.tensor_wrapper_constructions == 8,
               "stable runs reconstructed tensor wrappers");
  ok &= Expect(stats.cache_hits == 128,
               "stable runs did not hit prepared bindings 128 times");
  ok &= Expect(stats.application_binding_allocation_requests ==
                   warm_allocation_requests,
               "stable runs requested StudioCast binding storage");

  std::vector<float> changed_in(480, 0.75f);
  std::vector<float> changed_out(480);
  if (!RunPrepared(session.get(), 0, &changed_in, &changed_out, &state[0],
                   &state[1], shape480, &error)) {
    return Expect(false, "pointer-change run failed: " + error);
  }
  stats = session->prepared_run_stats();
  ok &= Expect(stats.binding_rebuilds == 3,
               "pointer change did not cause exactly one rebuild");

  std::vector<float> shape_in(240, 0.125f);
  std::vector<float> shape_out(240);
  const std::array<int64_t, 2> shape240{1, 240};
  if (!RunPrepared(session.get(), 0, &shape_in, &shape_out, &state[0],
                   &state[1], shape240, &error)) {
    return Expect(false, "shape-change run failed: " + error);
  }
  stats = session->prepared_run_stats();
  ok &= Expect(stats.binding_rebuilds == 4,
               "shape change did not cause exactly one rebuild");

  session->InvalidatePreparedBindings();
  if (!RunPrepared(session.get(), 0, &shape_in, &shape_out, &state[0],
                   &state[1], shape240, &error)) {
    return Expect(false, "explicit invalidation run failed: " + error);
  }
  stats = session->prepared_run_stats();
  ok &= Expect(stats.binding_rebuilds == 5,
               "explicit provider/config invalidation did not rebuild once");

  // A model/provider/session replacement owns a new bounded cache. Its first
  // run performs one preparation; no binding survives across session identity.
  auto replacement = studiocast::onnx::OrtSession::Create(
      pack.model_path(), opts, &info, &error);
  if (!Expect(replacement != nullptr,
              "replacement session creation failed: " + error))
    return false;
  replacement->ReserveRunScratch(2, 2);
  if (!RunPrepared(replacement.get(), 0, &shape_in, &shape_out, &state[0],
                   &state[1], shape240, &error)) {
    return Expect(false, "replacement session run failed: " + error);
  }
  ok &= Expect(replacement->prepared_run_stats().binding_rebuilds == 1,
               "replacement model/provider/session must prepare once");
  return ok;
}
#endif

bool TestOpenAudioProcessorSteadyBindings() {
#if !STUDIOCAST_ENABLE_OPEN_AUDIO || !STUDIOCAST_HAVE_ONNXRUNTIME
  std::cout << "SKIP: Open Audio/ONNX Runtime unavailable in this build\n";
  return true;
#else
  TempModelPack pack;
  if (!Expect(pack.error().empty(), pack.error()))
    return false;

  studiocast::audio::effects::BroadcastAudioEffects effects;
  effects.microphone.model_path = pack.dir().string();
  effects.microphone.noise_removal_enabled = true;
  studiocast::open_audio::OrtSessionOptions opts;
  opts.prefer_cuda = false;
  studiocast::open_audio::ResolvedOpenAudioModel resolved;
  std::string error;
  auto processor = studiocast::open_audio::OpenAudioAudioProcessor::
      CreateForMicrophoneWithOrtOptions(effects, opts, &resolved, &error);
  if (!Expect(processor != nullptr, "processor creation failed: " + error))
    return false;

  std::vector<float> input(480, 0.1f);
  std::vector<float> output(480);
  for (int i = 0; i < 4; ++i) {
    if (!processor->Process(input.data(), output.data(), 480, 1, &error))
      return Expect(false, "processor warmup failed: " + error);
  }
  const auto warm_stats = processor->PreparedRunStatsForTesting();
  bool ok = true;
  ok &= Expect(warm_stats.binding_rebuilds == 2,
               "recurrent processor must prepare exactly two slots");
  ok &= Expect(warm_stats.tensor_wrapper_constructions == 8,
               "recurrent processor constructed unexpected wrappers");

  bool process_ok = true;
  for (int i = 0; i < 128; ++i) {
    if (!processor->Process(input.data(), output.data(), 480, 1, &error)) {
      process_ok = false;
      break;
    }
  }
  ok &= Expect(process_ok, "steady processor run failed: " + error);
  const auto steady_stats = processor->PreparedRunStatsForTesting();
  ok &= Expect(steady_stats.binding_rebuilds == warm_stats.binding_rebuilds,
               "steady processor rebuilt a prepared binding");
  ok &= Expect(steady_stats.tensor_wrapper_constructions ==
                   warm_stats.tensor_wrapper_constructions,
               "steady processor reconstructed a tensor wrapper");
  ok &= Expect(steady_stats.cache_hits == warm_stats.cache_hits + 128,
               "steady processor did not reuse prepared bindings");
  ok &= Expect(
      steady_stats.application_binding_allocation_requests ==
          warm_stats.application_binding_allocation_requests,
      "128 post-warm processor blocks requested StudioCast binding storage");
  ok &= Expect(std::all_of(output.begin(), output.end(),
                           [](float v) { return std::isfinite(v); }),
               "processor produced non-finite output");
  return ok;
#endif
}

} // namespace

int main() {
  bool ok = true;
#if STUDIOCAST_HAVE_ONNXRUNTIME
  ok &= TestPreparedCpuBindings();
#else
  std::cout << "SKIP: shared ONNX Runtime prepared-binding test unavailable\n";
#endif
  ok &= TestOpenAudioProcessorSteadyBindings();
  if (!ok)
    return 1;
  std::cout << "Open Audio prepared binding tests passed\n";
  return 0;
}
