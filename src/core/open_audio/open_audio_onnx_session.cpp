#include "core/open_audio/open_audio_onnx_session.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "core/onnx/ort_session.h"

namespace studiocast::open_audio {

namespace {

OrtRuntimeInfo ToOpenAudioRuntimeInfo(
    const studiocast::onnx::OrtRuntimeInfo &shared) {
  OrtRuntimeInfo out;
  out.version = shared.version;
  out.providers = shared.providers;
  out.cuda_provider_present = shared.cuda_provider_present;
  out.tensorrt_provider_present = shared.tensorrt_provider_present;
  out.cpu_provider_present = shared.cpu_provider_present;
  out.cuda_ep_v2_build = shared.cuda_ep_v2_build;
  out.library_path = shared.library_path;
  out.warnings = shared.warnings;
  return out;
}

OrtSessionInfo ToOpenAudioSessionInfo(
    const studiocast::onnx::OrtSessionInfo &shared) {
  OrtSessionInfo out;
  out.using_cuda = shared.using_cuda;
  out.advertised_providers = shared.advertised_providers;
  out.cuda_provider_advertised = shared.cuda_provider_advertised;
  out.tensorrt_provider_advertised = shared.tensorrt_provider_advertised;
  out.cpu_provider_advertised = shared.cpu_provider_advertised;
  out.cuda_provider_appended = shared.cuda_provider_appended;
  out.cuda_provider_usable = shared.cuda_provider_usable;
  out.cpu_provider_usable = shared.cpu_provider_usable;
  out.cuda_session_create_failed_fell_back_to_cpu =
      shared.cuda_session_create_failed_fell_back_to_cpu;
  out.active_provider = shared.active_provider;
  out.appended_provider = shared.appended_provider;
  out.appended_providers = shared.appended_providers;
  out.input_names = shared.input_names;
  out.output_names = shared.output_names;
  out.input_descriptions = shared.input_descriptions;
  out.output_descriptions = shared.output_descriptions;
  out.input_shapes = shared.input_shapes;
  out.input_elem_types = shared.input_elem_types;
  out.output_shapes = shared.output_shapes;
  out.output_elem_types = shared.output_elem_types;
  out.warnings = shared.warnings;
  return out;
}

studiocast::onnx::OrtSessionOptions
ToSharedOptions(const OrtSessionOptions &opts) {
  studiocast::onnx::OrtSessionOptions shared;
  shared.prefer_cuda = opts.prefer_cuda;
  shared.cuda_device_id = opts.cuda_device_id;
  return shared;
}

} // namespace

struct OpenAudioOrtSession::Impl {
  OrtSessionInfo info;
  std::unique_ptr<studiocast::onnx::OrtSession> session;

  // Run1D is a diagnostics/self-test convenience path, not the live audio path.
  std::vector<int64_t> run1d_input_shape;
  std::vector<int64_t> run1d_output_shape;
  std::vector<float> run1d_temp_output;
};

OpenAudioOrtSession::OpenAudioOrtSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
OpenAudioOrtSession::~OpenAudioOrtSession() = default;

const OrtSessionInfo &OpenAudioOrtSession::info() const { return impl_->info; }

void OpenAudioOrtSession::ReserveRunScratch(std::size_t input_count,
                                            std::size_t output_count) {
  if (!impl_)
    return;
  if (impl_->session) {
    impl_->session->ReserveRunScratch(input_count, output_count);
  }
}

OrtRuntimeInfo OpenAudioOrtSession::QueryRuntimeInfo() {
  return ToOpenAudioRuntimeInfo(
      studiocast::onnx::OrtSession::QueryRuntimeInfo());
}

std::unique_ptr<OpenAudioOrtSession>
OpenAudioOrtSession::Create(const std::filesystem::path &model_path,
                            const OrtSessionOptions &opts,
                            OrtSessionInfo *info_out, std::string *error) {
  if (error)
    error->clear();
  if (info_out)
    *info_out = OrtSessionInfo{};

  studiocast::onnx::OrtSessionInfo shared_info;
  auto shared_session = studiocast::onnx::OrtSession::Create(
      model_path, ToSharedOptions(opts), &shared_info, error);
  if (!shared_session) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->session = std::move(shared_session);
  impl->info = ToOpenAudioSessionInfo(shared_info);

  if (info_out) {
    *info_out = impl->info;
  }

  return std::unique_ptr<OpenAudioOrtSession>(
      new OpenAudioOrtSession(std::move(impl)));
}

bool OpenAudioOrtSession::Run(const OrtRunInput *inputs,
                              std::size_t input_count,
                              const OrtRunOutput *outputs,
                              std::size_t output_count, std::string *error) {
  return RunPrepared(0, inputs, input_count, outputs, output_count, error);
}

bool OpenAudioOrtSession::RunPrepared(
    std::size_t binding_slot, const OrtRunInput *inputs,
    std::size_t input_count, const OrtRunOutput *outputs,
    std::size_t output_count, std::string *error) {
  if (error)
    error->clear();

  if (!impl_ || !impl_->session) {
    if (error)
      *error = "ORT session is not initialized.";
    return false;
  }
  if (!inputs || !outputs) {
    if (error)
      *error = "null inputs/outputs passed to ORT Run().";
    return false;
  }

  return impl_->session->RunCpuPrepared(binding_slot, inputs, input_count,
                                        outputs, output_count, error);
}

void OpenAudioOrtSession::InvalidatePreparedBindings() {
  if (impl_ && impl_->session)
    impl_->session->InvalidatePreparedBindings();
}

OpenAudioOrtSession::PreparedRunStats
OpenAudioOrtSession::prepared_run_stats() const {
  return (impl_ && impl_->session) ? impl_->session->prepared_run_stats()
                                  : PreparedRunStats{};
}

bool OpenAudioOrtSession::Run1D(const float *input, std::size_t samples,
                                float *output, std::size_t output_capacity,
                                std::size_t *output_samples,
                                std::string *error) {
  if (error)
    error->clear();
  if (output_samples)
    *output_samples = 0;

  if (!impl_ || !impl_->session) {
    if (error)
      *error = "ORT session is not initialized.";
    return false;
  }
  if (!input || !output) {
    if (error)
      *error = "null buffer passed to Run1D";
    return false;
  }
  if (samples == 0)
    return true;
  if (impl_->info.input_names.empty() || impl_->info.output_names.empty() ||
      impl_->info.input_names[0].empty() ||
      impl_->info.output_names[0].empty()) {
    if (error)
      *error = "ORT session does not expose input/output names for Run1D.";
    return false;
  }

  impl_->run1d_input_shape = {1, static_cast<int64_t>(samples)};
  impl_->run1d_output_shape = {1, static_cast<int64_t>(samples)};

  float *output_buffer = output;
  if (output_capacity < samples) {
    impl_->run1d_temp_output.assign(samples, 0.0f);
    output_buffer = impl_->run1d_temp_output.data();
  }

  OrtRunInput in;
  in.name = impl_->info.input_names[0].c_str();
  in.data = input;
  in.num_floats = samples;
  in.shape = impl_->run1d_input_shape.data();
  in.shape_rank = impl_->run1d_input_shape.size();

  OrtRunOutput out;
  out.name = impl_->info.output_names[0].c_str();
  out.data = output_buffer;
  out.num_floats = samples;
  out.shape = impl_->run1d_output_shape.data();
  out.shape_rank = impl_->run1d_output_shape.size();

  if (!Run(&in, 1, &out, 1, error)) {
    return false;
  }

  const std::size_t copied = std::min(samples, output_capacity);
  if (output_buffer != output && copied > 0) {
    std::copy_n(output_buffer, copied, output);
  }
  if (output_samples)
    *output_samples = copied;
  return true;
}

} // namespace studiocast::open_audio
