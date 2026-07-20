#include "core/open_video/fastdvdnet_denoiser.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <sstream>
#include <utility>

#include "core/util/fs.h"
#include "core/util/json.h"

namespace studiocast::open_video {
namespace {

constexpr int kFastDvdnetSpatialAlignment = 4;
constexpr int kFastDvdnetWindowFrames = 5;
constexpr int kFastDvdnetHistoryFrames = 3; // t-2, t-1, t
constexpr int kFastDvdnetRepeatedFutureFrames = 2;
constexpr int kFastDvdnetRgbChannels = 3;
constexpr int kFastDvdnetNoisyChannels =
    kFastDvdnetWindowFrames * kFastDvdnetRgbChannels;
constexpr int kFastDvdnetNoiseMapChannels = 1;
constexpr float kFastDvdnetMaxSigma = 55.0f;

std::vector<int64_t> NchwShape(int channels, int height, int width) {
  if (height > 0 && width > 0) {
    return {1, channels, height, width};
  }
  return {1, channels, -1, -1};
}

DenoiseTensorAdapterContract BuildFastDvdnetTensorContract(
    int proc_w, int proc_h, const std::string &noisy_name,
    const std::string &noise_map_name, const std::string &denoised_name) {
  DenoiseTensorAdapterContract c;
  c.adapter_id = "fastdvdnet";
  c.model_family = "FastDVDnet";

  DenoiseTensorSpec noisy;
  noisy.role = "temporal_rgb_window";
  noisy.name = noisy_name.empty() ? "noisy" : noisy_name;
  noisy.element_type = "float32";
  noisy.layout = "NCHW";
  noisy.shape = NchwShape(kFastDvdnetNoisyChannels, proc_h, proc_w);

  DenoiseTensorSpec noise;
  noise.role = "noise_map";
  noise.name = noise_map_name.empty() ? "noise_map" : noise_map_name;
  noise.element_type = "float32";
  noise.layout = "NCHW";
  noise.shape = NchwShape(kFastDvdnetNoiseMapChannels, proc_h, proc_w);

  c.inputs = {std::move(noisy), std::move(noise)};

  c.output.role = "denoised_rgb";
  c.output.name = denoised_name.empty() ? "denoised" : denoised_name;
  c.output.element_type = "float32";
  c.output.layout = "NCHW";
  c.output.shape = NchwShape(kFastDvdnetRgbChannels, proc_h, proc_w);

  c.temporal.window_frames = kFastDvdnetWindowFrames;
  c.temporal.history_frames = kFastDvdnetHistoryFrames;
  c.temporal.repeated_future_frames = kFastDvdnetRepeatedFutureFrames;
  c.temporal.causal = true;

  c.normalization.frame_input = "RGB uint8 -> RGB float32 0..1";
  c.normalization.strength_input = "strength 0..100 -> sigma 0..55 -> /255";
  c.normalization.max_sigma = kFastDvdnetMaxSigma;

  c.supports_cpu_tensor_io = true;
  c.supports_cuda_device_tensor_io = true;

  c.requires_cpu_preprocess = true;
  c.requires_cpu_postprocess = true;
  c.requires_output_device_to_cpu_for_postprocess = true;
  return c;
}

int ParseSigmaFromIdFallback(const std::string &id) {
  // Best-effort extraction for curated ids like: fastdvdnet_sigma25.
  const std::string key = "sigma";
  const auto pos = id.find(key);
  if (pos == std::string::npos)
    return 25;
  int v = 0;
  bool have = false;
  for (std::size_t i = pos + key.size(); i < id.size(); ++i) {
    const char c = id[i];
    if (c < '0' || c > '9')
      break;
    have = true;
    v = v * 10 + (c - '0');
    if (v > 255)
      break;
  }
  if (!have)
    return 25;
  return std::clamp(v, 0, 255);
}

} // namespace

FastDvdnetDenoiser::FastDvdnetDenoiser() { RefreshTensorContract(); }
FastDvdnetDenoiser::~FastDvdnetDenoiser() { ResetCudaTensorIo(); }

int FastDvdnetDenoiser::AlignUp(int v, int align) {
  if (align <= 1)
    return v;
  const int r = v % align;
  if (r == 0)
    return v;
  return v + (align - r);
}

float FastDvdnetDenoiser::Clamp01(float x) {
  if (x < 0.0f)
    return 0.0f;
  if (x > 1.0f)
    return 1.0f;
  return x;
}

bool FastDvdnetDenoiser::active_session_uses_cuda_tensor_io() const {
  return active_session_uses_cuda_ep() &&
         tensor_contract_.supports_cuda_device_tensor_io &&
         cuda_tensor_io_ready_;
}

bool FastDvdnetDenoiser::active_session_uses_cpu_tensor_io() const {
  if (!ort_session_active_)
    return false;
  if (using_cpu_fallback_)
    return true;
  return !active_session_uses_cuda_tensor_io();
}

DenoiseTensorIoStatus FastDvdnetDenoiser::tensor_io_status() const {
  DenoiseTensorIoStatus s;
  s.cuda_ep_active = active_session_uses_cuda_ep();
  s.cuda_device_tensor_io_supported =
      tensor_contract_.supports_cuda_device_tensor_io;
  s.cuda_device_tensor_io_active = active_session_uses_cuda_tensor_io();
  s.cuda_ep_cpu_tensor_io_active = s.cuda_ep_active &&
                                   !s.cuda_device_tensor_io_active &&
                                   tensor_contract_.supports_cpu_tensor_io;
  s.cpu_only_fallback_active =
      ort_session_active_ != nullptr &&
      (using_cpu_fallback_ || !session_info_.using_cuda);
  s.cpu_tensor_tail_active =
      s.cuda_ep_active && (tensor_contract_.requires_cpu_preprocess ||
                           tensor_contract_.requires_cpu_postprocess ||
                           s.cuda_ep_cpu_tensor_io_active);
  s.output_readback_required_for_postprocess =
      s.cuda_device_tensor_io_active &&
      tensor_contract_.requires_output_device_to_cpu_for_postprocess;

  if (s.cuda_device_tensor_io_active) {
    s.summary =
        "Open Video denoise: CUDA EP active with adapter-declared CUDA "
        "device tensor I/O for ORT tensors; CPU preprocessing, denoised "
        "tensor readback, and RGB postprocess remain an explicit CPU tensor "
        "tail.";
  } else if (s.cuda_ep_cpu_tensor_io_active) {
    s.summary =
        "Open Video denoise: CUDA EP active, but this adapter is using CPU "
        "ORT tensors; preprocessing and RGB postprocess remain an explicit "
        "CPU tensor tail.";
  } else if (s.cpu_only_fallback_active) {
    s.summary = "Open Video denoise: CPU ORT fallback active; no CUDA tensor "
                "transfers are used.";
  }

  return s;
}

void FastDvdnetDenoiser::RefreshTensorContract() {
  tensor_contract_ = BuildFastDvdnetTensorContract(
      proc_w_, proc_h_, noisy_name_, noise_map_name_, denoised_name_);
}

int FastDvdnetDenoiser::HistoryFrameCount() const {
  return std::max(1, tensor_contract_.temporal.history_frames);
}

std::string
FastDvdnetDenoiser::ChoosePreferredModelId(const ModelPackRegistry &reg) {
  // Prefer the "medium" curated pack if present; otherwise fall back to any
  // installed model for the task.
  const auto &tasks = reg.Tasks();
  const auto it = tasks.find("video_denoise");
  if (it == tasks.end() || it->second.empty())
    return {};

  auto has = [&](const char *id) {
    return std::find(it->second.begin(), it->second.end(), std::string(id)) !=
           it->second.end();
  };

  if (has("fastdvdnet_sigma25"))
    return "fastdvdnet_sigma25";
  if (has("fastdvdnet_sigma15"))
    return "fastdvdnet_sigma15";
  if (has("fastdvdnet_sigma50"))
    return "fastdvdnet_sigma50";

  // Deterministic fallback: first model id for task.
  return it->second.front();
}

bool FastDvdnetDenoiser::LoadDefaultSigmaFromManifest(
    const std::filesystem::path &manifest_path, int *out_sigma) {
  if (out_sigma)
    *out_sigma = 25;
  if (manifest_path.empty())
    return false;

  const auto textOpt = util::ReadTextFile(manifest_path.string());
  if (!textOpt.has_value())
    return false;

  util::json::Value root;
  std::string err;
  if (!util::json::Parse(*textOpt, &root, &err)) {
    return false;
  }
  const auto *obj = root.AsObject();
  if (!obj)
    return false;

  auto it = obj->find("runtime_hints");
  if (it == obj->end())
    return false;
  const auto *hints = it->second.AsObject();
  if (!hints)
    return false;

  auto itS = hints->find("default_sigma");
  if (itS == hints->end())
    return false;
  const double *n = itS->second.AsNumber();
  if (!n)
    return false;
  const int v = static_cast<int>(std::lround(*n));
  if (out_sigma)
    *out_sigma = std::clamp(v, 0, 255);
  return true;
}

bool FastDvdnetDenoiser::ResolveModelFromRegistry(
    const ModelPackRegistry &reg, const std::string &requested_model_id,
    LoadedModel *out, std::string *error) {
  if (error)
    error->clear();
  if (!out) {
    if (error)
      *error = "internal error: out is null";
    return false;
  }

  const std::string chosen = requested_model_id.empty()
                                 ? ChoosePreferredModelId(reg)
                                 : requested_model_id;
  if (chosen.empty()) {
    if (error)
      *error = "Open Video FastDVDnet: no installed model packs for task "
               "'video_denoise'.";
    return false;
  }

  const auto pack = reg.Find("video_denoise", chosen);
  if (!pack.has_value()) {
    if (error) {
      *error = requested_model_id.empty()
                   ? ("Open Video FastDVDnet: selected model id not found: " +
                      chosen)
                   : ("Open Video FastDVDnet: requested model id not found: " +
                      chosen);
    }
    return false;
  }

  // Pick the ONNX file marked role=main, else first ONNX.
  std::filesystem::path onnx;
  for (const auto &f : pack->files) {
    if (f.kind == "onnx" && f.role == "main") {
      onnx = pack->root_dir / f.name;
      break;
    }
  }
  if (onnx.empty()) {
    for (const auto &f : pack->files) {
      if (f.kind == "onnx") {
        onnx = pack->root_dir / f.name;
        break;
      }
    }
  }

  if (onnx.empty()) {
    if (error)
      *error = "Open Video FastDVDnet: model pack declares no ONNX file.";
    return false;
  }

  out->id = pack->id;
  out->onnx = onnx;
  out->default_sigma = ParseSigmaFromIdFallback(pack->id);
  int hinted = out->default_sigma;
  if (LoadDefaultSigmaFromManifest(pack->manifest_path, &hinted)) {
    out->default_sigma = hinted;
  }
  return true;
}

bool FastDvdnetDenoiser::EnsureSessionForModel(const LoadedModel &model,
                                               std::string *error) {
  if (error)
    error->clear();

  if (model.onnx.empty()) {
    if (error)
      *error = "Open Video FastDVDnet: model ONNX path is empty.";
    return false;
  }

  // Avoid reloading if already active.
  if (initialized_ && !active_model_id_.empty() &&
      model.id == active_model_id_ && model.onnx == active_model_path_ &&
      ort_session_active_) {
    model_default_sigma_ = model.default_sigma;
    return true;
  }

  ort_session_cuda_.reset();
  ort_session_cpu_.reset();
  ort_session_active_ = nullptr;
  using_cpu_fallback_ = false;
  session_info_ = studiocast::onnx::OrtSessionInfo{};
  ResetCudaTensorIo();

  studiocast::onnx::OrtSessionOptions cuda_opts;
  cuda_opts.prefer_cuda = true;

  std::string cuda_setup_err;
  if (cuda_.Initialize(&cuda_setup_err) &&
      cuda_.EnsureContext(&cuda_setup_err) &&
      cuda_.CreateStream(&cuda_stream_, &cuda_setup_err)) {
    cuda_stream_owned_ = true;
    cuda_opts.user_compute_stream = reinterpret_cast<void *>(cuda_stream_);
  }

  std::string err;
  studiocast::onnx::OrtSessionInfo info_cuda;
  auto cuda = studiocast::onnx::OrtSession::Create(model.onnx, cuda_opts,
                                                   &info_cuda, &err);
  if (!cuda) {
    ResetCudaTensorIo();
    if (error) {
      *error = "Open Video FastDVDnet: failed to create ORT session: " +
               (err.empty() ? std::string("unknown") : err);
    }
    return false;
  }

  std::unique_ptr<studiocast::onnx::OrtSession> cpu;
  studiocast::onnx::OrtSessionInfo info_cpu;
  if (info_cuda.using_cuda) {
    studiocast::onnx::OrtSessionOptions cpu_opts;
    cpu_opts.prefer_cuda = false;
    std::string err_cpu;
    cpu = studiocast::onnx::OrtSession::Create(model.onnx, cpu_opts, &info_cpu,
                                               &err_cpu);
    // CPU fallback is best-effort; if it fails, we can still run CUDA-only.
    if (!cpu && !err_cpu.empty()) {
      if (!info_cuda.warnings.empty()) {
        // Keep existing warnings.
      }
    }
  }

  ort_session_cuda_ = std::move(cuda);
  ort_session_cpu_ = std::move(cpu);
  ort_session_active_ = ort_session_cuda_.get();
  session_info_ = info_cuda;
  using_cpu_fallback_ = false;
  if (!session_info_.using_cuda) {
    ResetCudaTensorIo();
  }

  active_model_id_ = model.id;
  active_model_path_ = model.onnx;
  model_default_sigma_ = model.default_sigma;

  // Force IO re-detection.
  noisy_name_.clear();
  noise_map_name_.clear();
  denoised_name_.clear();

  ResetTemporalState();
  return true;
}

bool FastDvdnetDenoiser::RefreshGeometry(int src_w, int src_h,
                                         std::string *error) {
  if (error)
    error->clear();

  if (src_w <= 0 || src_h <= 0) {
    if (error)
      *error = "Open Video FastDVDnet: invalid frame size.";
    return false;
  }

  const int new_proc_w = AlignUp(src_w, kFastDvdnetSpatialAlignment);
  const int new_proc_h = AlignUp(src_h, kFastDvdnetSpatialAlignment);

  if (new_proc_w == proc_w_ && new_proc_h == proc_h_ && src_w == src_w_ &&
      src_h == src_h_) {
    return true;
  }

  proc_w_ = new_proc_w;
  proc_h_ = new_proc_h;
  src_w_ = src_w;
  src_h_ = src_h;

  const std::size_t plane =
      static_cast<std::size_t>(proc_w_) * static_cast<std::size_t>(proc_h_);

  noisy_shape_ = NchwShape(kFastDvdnetNoisyChannels, proc_h_, proc_w_);
  noise_map_shape_ = NchwShape(kFastDvdnetNoiseMapChannels, proc_h_, proc_w_);
  denoised_shape_ = NchwShape(kFastDvdnetRgbChannels, proc_h_, proc_w_);
  RefreshTensorContract();

  noisy_tensor_.assign(kFastDvdnetNoisyChannels * plane, 0.0f);
  noise_map_tensor_.assign(plane, 0.0f);
  denoised_tensor_.assign(kFastDvdnetRgbChannels * plane, 0.0f);
  last_noise_map_value_ = -1.0f;

  history_.clear();
  history_.resize(static_cast<std::size_t>(HistoryFrameCount()));
  for (auto &f : history_) {
    f.assign(kFastDvdnetRgbChannels * plane, 0.0f);
  }
  history_filled_ = 0;
  history_write_idx_ = 0;
  have_last_sequence_ = false;
  last_capture_sequence_ = 0;

  // Pre-build ORT bindings.
  ort_inputs_.clear();
  ort_outputs_.clear();

  if (!DetectIoNames(error)) {
    return false;
  }

  studiocast::onnx::OrtSession::RunInput in_noisy;
  in_noisy.name = noisy_name_.c_str();
  in_noisy.data = noisy_tensor_.data();
  in_noisy.num_floats = noisy_tensor_.size();
  in_noisy.shape = noisy_shape_.data();
  in_noisy.shape_rank = noisy_shape_.size();

  studiocast::onnx::OrtSession::RunInput in_noise;
  in_noise.name = noise_map_name_.c_str();
  in_noise.data = noise_map_tensor_.data();
  in_noise.num_floats = noise_map_tensor_.size();
  in_noise.shape = noise_map_shape_.data();
  in_noise.shape_rank = noise_map_shape_.size();

  ort_inputs_.push_back(in_noisy);
  ort_inputs_.push_back(in_noise);

  studiocast::onnx::OrtSession::RunOutput out_den;
  out_den.name = denoised_name_.c_str();
  out_den.data = denoised_tensor_.data();
  out_den.num_floats = denoised_tensor_.size();
  out_den.shape = denoised_shape_.data();
  out_den.shape_rank = denoised_shape_.size();
  ort_outputs_.push_back(out_den);

  if (ort_session_cuda_) {
    ort_session_cuda_->ReserveRunScratch(ort_inputs_.size(),
                                         ort_outputs_.size());
  }
  if (ort_session_cpu_) {
    ort_session_cpu_->ReserveRunScratch(ort_inputs_.size(),
                                        ort_outputs_.size());
  }

  if (session_info_.using_cuda && !using_cpu_fallback_) {
    std::string cuda_err;
    if (!EnsureCudaTensorIo(&cuda_err)) {
      cuda_tensor_io_ready_ = false;
      if (!cuda_err.empty()) {
        sticky_warning_ =
            "Open Video FastDVDnet: CUDA tensor IoBinding unavailable; "
            "using CPU tensor I/O with CUDA EP. " +
            cuda_err;
      }
    }
  } else {
    ResetCudaTensorIo();
  }

  return true;
}

bool FastDvdnetDenoiser::EnsureCudaTensorIo(std::string *error) {
  if (error)
    error->clear();
  cuda_tensor_io_ready_ = false;

  if (!session_info_.using_cuda || using_cpu_fallback_) {
    if (error)
      *error = "CUDA EP is not active.";
    return false;
  }
  if (!tensor_contract_.supports_cuda_device_tensor_io) {
    if (error)
      *error = "denoise tensor adapter does not support CUDA device tensors.";
    return false;
  }

  std::string err;
  if (!cuda_.IsInitialized() && !cuda_.Initialize(&err)) {
    if (error)
      *error = "CUDA driver initialization failed: " + err;
    return false;
  }
  if (!cuda_.EnsureContext(&err)) {
    if (error)
      *error = "CUDA context setup failed: " + err;
    return false;
  }
  if (!cuda_stream_) {
    if (!cuda_.CreateStream(&cuda_stream_, &err)) {
      if (error)
        *error = "CUDA stream creation failed: " + err;
      return false;
    }
    cuda_stream_owned_ = true;
  }

  if (!cuda_noisy_tensor_.ReallocIfNeededNchwF32(
          &cuda_, 1, kFastDvdnetNoisyChannels, proc_h_, proc_w_, &err)) {
    if (error)
      *error = "failed to allocate noisy tensor: " + err;
    return false;
  }
  if (!cuda_noise_map_tensor_.ReallocIfNeededNchwF32(
          &cuda_, 1, kFastDvdnetNoiseMapChannels, proc_h_, proc_w_, &err)) {
    if (error)
      *error = "failed to allocate noise map tensor: " + err;
    return false;
  }
  if (!cuda_denoised_tensor_.ReallocIfNeededNchwF32(
          &cuda_, 1, kFastDvdnetRgbChannels, proc_h_, proc_w_, &err)) {
    if (error)
      *error = "failed to allocate denoised tensor: " + err;
    return false;
  }

  cuda_ort_inputs_.clear();
  cuda_ort_outputs_.clear();

  studiocast::onnx::OrtSession::CudaBindingInput in_noisy;
  in_noisy.name = noisy_name_.c_str();
  in_noisy.device_ptr = reinterpret_cast<const float *>(
      static_cast<std::uintptr_t>(cuda_noisy_tensor_.ptr));
  in_noisy.num_floats = cuda_noisy_tensor_.ElementCount();
  in_noisy.shape = noisy_shape_.data();
  in_noisy.shape_rank = noisy_shape_.size();

  studiocast::onnx::OrtSession::CudaBindingInput in_noise;
  in_noise.name = noise_map_name_.c_str();
  in_noise.device_ptr = reinterpret_cast<const float *>(
      static_cast<std::uintptr_t>(cuda_noise_map_tensor_.ptr));
  in_noise.num_floats = cuda_noise_map_tensor_.ElementCount();
  in_noise.shape = noise_map_shape_.data();
  in_noise.shape_rank = noise_map_shape_.size();

  studiocast::onnx::OrtSession::CudaBindingOutput out_den;
  out_den.name = denoised_name_.c_str();
  out_den.device_ptr = reinterpret_cast<float *>(
      static_cast<std::uintptr_t>(cuda_denoised_tensor_.ptr));
  out_den.num_floats = cuda_denoised_tensor_.ElementCount();
  out_den.shape = denoised_shape_.data();
  out_den.shape_rank = denoised_shape_.size();

  cuda_ort_inputs_.push_back(in_noisy);
  cuda_ort_inputs_.push_back(in_noise);
  cuda_ort_outputs_.push_back(out_den);
  cuda_tensor_io_ready_ = true;
  constexpr const char *kCudaTensorIoUnavailablePrefix =
      "Open Video FastDVDnet: CUDA tensor IoBinding unavailable;";
  if (sticky_warning_.rfind(kCudaTensorIoUnavailablePrefix, 0) == 0) {
    sticky_warning_.clear();
  }
  return true;
}

void FastDvdnetDenoiser::ResetCudaTensorIo() {
  std::string ignored;
  (void)cuda_noisy_tensor_.Free(&cuda_, &ignored);
  (void)cuda_noise_map_tensor_.Free(&cuda_, &ignored);
  (void)cuda_denoised_tensor_.Free(&cuda_, &ignored);
  if (cuda_stream_owned_ && cuda_stream_) {
    (void)cuda_.DestroyStream(cuda_stream_, &ignored);
  }
  cuda_stream_ = nullptr;
  cuda_stream_owned_ = false;
  cuda_tensor_io_ready_ = false;
  cuda_ort_inputs_.clear();
  cuda_ort_outputs_.clear();
}

bool FastDvdnetDenoiser::DetectIoNames(std::string *error) {
  if (error)
    error->clear();
  if (!ort_session_active_) {
    if (error)
      *error = "Open Video FastDVDnet: ORT session not initialized.";
    return false;
  }

  const auto &info = session_info_;
  if (info.input_names.size() < 2 || info.output_names.empty()) {
    if (error)
      *error = "Open Video FastDVDnet: unexpected ONNX IO count (need >=2 "
               "inputs, >=1 output).";
    return false;
  }

  // Heuristic: find FastDVDnet adapter inputs by declared channel count.
  int noisy_idx = -1;
  int noise_idx = -1;
  for (std::size_t i = 0; i < info.input_shapes.size(); ++i) {
    const auto &s = info.input_shapes[i];
    if (s.size() == 4) {
      if (s[1] == kFastDvdnetNoisyChannels)
        noisy_idx = static_cast<int>(i);
      if (s[1] == kFastDvdnetNoiseMapChannels)
        noise_idx = static_cast<int>(i);
    }
  }
  if (noisy_idx < 0 || noise_idx < 0 || noisy_idx == noise_idx) {
    // Fallback: match by name.
    for (std::size_t i = 0; i < info.input_names.size(); ++i) {
      const auto &n = info.input_names[i];
      if (noisy_idx < 0 &&
          (n == "noisy" || n.find("noisy") != std::string::npos))
        noisy_idx = static_cast<int>(i);
      if (noise_idx < 0 &&
          (n == "noise_map" || n.find("noise") != std::string::npos))
        noise_idx = static_cast<int>(i);
    }
  }
  if (noisy_idx < 0)
    noisy_idx = 0;
  if (noise_idx < 0)
    noise_idx = (noisy_idx == 0 ? 1 : 0);

  int out_idx = -1;
  for (std::size_t i = 0; i < info.output_shapes.size(); ++i) {
    const auto &s = info.output_shapes[i];
    if (s.size() == 4 && s[1] == kFastDvdnetRgbChannels) {
      out_idx = static_cast<int>(i);
      break;
    }
  }
  if (out_idx < 0)
    out_idx = 0;

  noisy_name_ = info.input_names[static_cast<std::size_t>(noisy_idx)];
  noise_map_name_ = info.input_names[static_cast<std::size_t>(noise_idx)];
  denoised_name_ = info.output_names[static_cast<std::size_t>(out_idx)];

  if (noisy_name_.empty() || noise_map_name_.empty() ||
      denoised_name_.empty()) {
    if (error)
      *error = "Open Video FastDVDnet: failed to resolve IO tensor names.";
    return false;
  }
  RefreshTensorContract();
  return true;
}

bool FastDvdnetDenoiser::EnsureInitialized(
    int src_w, int src_h, const std::string &requested_model_id,
    std::string *error) {
  if (error)
    error->clear();

  if (disabled_) {
    if (error && !sticky_warning_.empty())
      *error = sticky_warning_;
    return false;
  }

  if (initialized_ && ort_session_active_ &&
      requested_model_id == active_requested_model_id_) {
    return RefreshGeometry(src_w, src_h, error);
  }

  // Scan installed packs only when the requested model configuration changes.
  registry_ = ModelPackRegistry::ScanDefault();
  LoadedModel model;
  std::string resolve_err;
  if (!ResolveModelFromRegistry(registry_, requested_model_id, &model,
                                &resolve_err)) {
    if (error)
      *error = resolve_err;
    return false;
  }

  std::string sess_err;
  if (!EnsureSessionForModel(model, &sess_err)) {
    if (error)
      *error = sess_err;
    return false;
  }

  std::string geo_err;
  if (!RefreshGeometry(src_w, src_h, &geo_err)) {
    if (error)
      *error = geo_err;
    return false;
  }

  initialized_ = true;
  active_requested_model_id_ = requested_model_id;
  return true;
}

void FastDvdnetDenoiser::ResetTemporalState() {
  history_filled_ = 0;
  history_write_idx_ = 0;
  have_last_sequence_ = false;
  last_capture_sequence_ = 0;
}

void FastDvdnetDenoiser::PreprocessRgbToChwPadded(
    const std::uint8_t *rgb, int width, int height, std::size_t stride,
    std::vector<float> *out_chw) const {
  if (!out_chw)
    return;
  const std::size_t plane =
      static_cast<std::size_t>(proc_w_) * static_cast<std::size_t>(proc_h_);
  if (out_chw->size() != kFastDvdnetRgbChannels * plane) {
    out_chw->assign(kFastDvdnetRgbChannels * plane, 0.0f);
  }

  float *out_r = out_chw->data() + 0 * plane;
  float *out_g = out_chw->data() + 1 * plane;
  float *out_b = out_chw->data() + 2 * plane;

  // Convert + pad by edge replication.
  for (int y = 0; y < proc_h_; ++y) {
    const int sy = std::clamp(y, 0, height - 1);
    const std::uint8_t *row = rgb + static_cast<std::size_t>(sy) * stride;
    for (int x = 0; x < proc_w_; ++x) {
      const int sx = std::clamp(x, 0, width - 1);
      const std::uint8_t *px = row + sx * 3;
      const std::size_t idx =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(proc_w_) +
          static_cast<std::size_t>(x);
      out_r[idx] = static_cast<float>(px[0]) * (1.0f / 255.0f);
      out_g[idx] = static_cast<float>(px[1]) * (1.0f / 255.0f);
      out_b[idx] = static_cast<float>(px[2]) * (1.0f / 255.0f);
    }
  }
}

void FastDvdnetDenoiser::BuildNoisyTensorFromHistory(
    std::vector<float> *out_noisy) const {
  if (!out_noisy)
    return;
  const std::size_t plane =
      static_cast<std::size_t>(proc_w_) * static_cast<std::size_t>(proc_h_);
  if (out_noisy->size() != kFastDvdnetNoisyChannels * plane) {
    out_noisy->assign(kFastDvdnetNoisyChannels * plane, 0.0f);
  }
  const int history_frames = HistoryFrameCount();

  // Determine indices for t-2, t-1, t.
  auto idx_t = [&](int back) -> int {
    // back=0 -> newest, back=1 -> t-1, back=2 -> t-2.
    if (history_filled_ <= 0)
      return 0;
    const int newest =
        (history_write_idx_ - 1 + history_frames) % history_frames;
    if (back == 0)
      return newest;
    if (history_filled_ < back + 1) {
      // Not enough history; replicate oldest available.
      const int oldest =
          (history_write_idx_ - history_filled_ + history_frames * 8) %
          history_frames;
      return oldest;
    }
    return (newest - back + history_frames) % history_frames;
  };

  const int i_t2 = idx_t(2);
  const int i_t1 = idx_t(1);
  const int i_t0 = idx_t(0);

  const std::vector<float> *f_t2 = &history_[static_cast<std::size_t>(i_t2)];
  const std::vector<float> *f_t1 = &history_[static_cast<std::size_t>(i_t1)];
  const std::vector<float> *f_t0 = &history_[static_cast<std::size_t>(i_t0)];

  // Future frames replicated as current.
  const std::vector<float> *f_tp1 = f_t0;
  const std::vector<float> *f_tp2 = f_t0;

  const std::array<const std::vector<float> *, kFastDvdnetWindowFrames> frames =
      {f_t2, f_t1, f_t0, f_tp1, f_tp2};

  for (std::size_t fi = 0; fi < frames.size(); ++fi) {
    const auto *f = frames[fi];
    if (!f || f->size() < kFastDvdnetRgbChannels * plane)
      continue;

    const float *in_r = f->data() + 0 * plane;
    const float *in_g = f->data() + 1 * plane;
    const float *in_b = f->data() + 2 * plane;

    float *out_r = out_noisy->data() + (fi * 3 + 0) * plane;
    float *out_g = out_noisy->data() + (fi * 3 + 1) * plane;
    float *out_b = out_noisy->data() + (fi * 3 + 2) * plane;
    std::memcpy(out_r, in_r, plane * sizeof(float));
    std::memcpy(out_g, in_g, plane * sizeof(float));
    std::memcpy(out_b, in_b, plane * sizeof(float));
  }
}

void FastDvdnetDenoiser::EnsureNoiseMap(float sigma_over_255) {
  const float v = std::clamp(sigma_over_255, 0.0f, 1.0f);
  if (std::abs(v - last_noise_map_value_) < 1e-6f && !noise_map_tensor_.empty())
    return;
  last_noise_map_value_ = v;
  std::fill(noise_map_tensor_.begin(), noise_map_tensor_.end(), v);
}

void FastDvdnetDenoiser::PostprocessToRgbInPlace(std::uint8_t *rgb, int width,
                                                 int height,
                                                 std::size_t stride) const {
  if (!rgb)
    return;
  if (denoised_tensor_.empty())
    return;

  const std::size_t plane =
      static_cast<std::size_t>(proc_w_) * static_cast<std::size_t>(proc_h_);
  if (denoised_tensor_.size() < kFastDvdnetRgbChannels * plane)
    return;

  const float *in_r = denoised_tensor_.data() + 0 * plane;
  const float *in_g = denoised_tensor_.data() + 1 * plane;
  const float *in_b = denoised_tensor_.data() + 2 * plane;

  for (int y = 0; y < height; ++y) {
    std::uint8_t *row = rgb + static_cast<std::size_t>(y) * stride;
    const std::size_t base =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(proc_w_);
    for (int x = 0; x < width; ++x) {
      const std::size_t idx = base + static_cast<std::size_t>(x);
      const float r = std::clamp(in_r[idx], 0.0f, 1.0f);
      const float g = std::clamp(in_g[idx], 0.0f, 1.0f);
      const float b = std::clamp(in_b[idx], 0.0f, 1.0f);

      const int ir = static_cast<int>(r * 255.0f + 0.5f);
      const int ig = static_cast<int>(g * 255.0f + 0.5f);
      const int ib = static_cast<int>(b * 255.0f + 0.5f);

      std::uint8_t *px = row + x * 3;
      px[0] = static_cast<std::uint8_t>(std::clamp(ir, 0, 255));
      px[1] = static_cast<std::uint8_t>(std::clamp(ig, 0, 255));
      px[2] = static_cast<std::uint8_t>(std::clamp(ib, 0, 255));
    }
  }
}

void FastDvdnetDenoiser::DisableAfterFailure(const std::string &why) {
  disabled_ = true;
  sticky_warning_ = why;
}

bool FastDvdnetDenoiser::ApplyRgbInPlace(std::uint64_t capture_sequence,
                                         std::uint8_t *rgb, int width,
                                         int height, std::size_t stride,
                                         int strength,
                                         const std::string &requested_model_id,
                                         std::string *error) {
  last_tensor_run_stats_ = DenoiseTensorRunStats{};
  if (error)
    error->clear();

  if (disabled_) {
    if (error && !sticky_warning_.empty())
      *error = sticky_warning_;
    return false;
  }
  if (!rgb) {
    if (error)
      *error = "Open Video FastDVDnet: null RGB buffer.";
    return false;
  }
  const int s = std::clamp(strength, 0, 100);
  if (s <= 0) {
    ResetTemporalState();
    return true;
  }

  std::string init_err;
  if (!EnsureInitialized(width, height, requested_model_id, &init_err)) {
    if (error)
      *error = init_err;
    return false;
  }
  if (!ort_session_active_) {
    if (error)
      *error = "Open Video FastDVDnet: ORT session missing.";
    return false;
  }
  last_tensor_run_stats_.cuda_ep_active = active_session_uses_cuda_ep();
  last_tensor_run_stats_.cpu_only_fallback_active =
      ort_session_active_ != nullptr &&
      (using_cpu_fallback_ || !session_info_.using_cuda);
  last_tensor_run_stats_.used_cpu_session =
      last_tensor_run_stats_.cpu_only_fallback_active;
  last_tensor_run_stats_.cpu_tensor_tail_active =
      tensor_io_status().cpu_tensor_tail_active;
  if (last_tensor_run_stats_.cpu_tensor_tail_active) {
    last_tensor_run_stats_.cpu_tail_stage_calls = 1;
  }

  // If capture sequence jumps (drop/restart), reset temporal history to avoid
  // blending stale frames.
  if (have_last_sequence_) {
    if (capture_sequence != last_capture_sequence_ + 1) {
      ResetTemporalState();
    }
  }
  have_last_sequence_ = true;
  last_capture_sequence_ = capture_sequence;

  // Preprocess current frame into history buffer (CHW float + padded).
  if (history_.empty()) {
    // Shouldn't happen (RefreshGeometry allocates), but be defensive.
    const std::size_t plane =
        static_cast<std::size_t>(proc_w_) * static_cast<std::size_t>(proc_h_);
    history_.resize(static_cast<std::size_t>(HistoryFrameCount()));
    for (auto &f : history_)
      f.assign(kFastDvdnetRgbChannels * plane, 0.0f);
  }

  PreprocessRgbToChwPadded(
      rgb, width, height, stride,
      &history_[static_cast<std::size_t>(history_write_idx_)]);
  const int history_frames = HistoryFrameCount();
  history_write_idx_ = (history_write_idx_ + 1) % history_frames;
  history_filled_ = std::min(history_filled_ + 1, history_frames);

  // Assemble ORT inputs.
  BuildNoisyTensorFromHistory(&noisy_tensor_);

  // Strength -> sigma in [0..55], then normalize to [0..1] via /255.
  const float t = Clamp01(static_cast<float>(s) / 100.0f);
  const float sigma =
      std::clamp(kFastDvdnetMaxSigma * t, 0.0f, kFastDvdnetMaxSigma);
  const float sigma_over_255 = sigma * (1.0f / 255.0f);
  EnsureNoiseMap(sigma_over_255);

  // Refresh binding pointers (in case vectors reallocated).
  if (ort_inputs_.size() != 2 || ort_outputs_.size() != 1) {
    // Rebuild.
    std::string geo_err;
    if (!RefreshGeometry(width, height, &geo_err)) {
      if (error)
        *error = geo_err;
      return false;
    }
  }

  ort_inputs_[0].data = noisy_tensor_.data();
  ort_inputs_[0].num_floats = noisy_tensor_.size();
  ort_inputs_[1].data = noise_map_tensor_.data();
  ort_inputs_[1].num_floats = noise_map_tensor_.size();
  ort_outputs_[0].data = denoised_tensor_.data();
  ort_outputs_[0].num_floats = denoised_tensor_.size();

  std::string ort_err;
  bool ort_ok = false;
  if (active_session_uses_cuda_tensor_io()) {
    last_tensor_run_stats_.used_cuda_device_tensor_io = true;
    if (cuda_ort_inputs_.size() != 2 || cuda_ort_outputs_.size() != 1) {
      std::string cuda_err;
      if (!EnsureCudaTensorIo(&cuda_err)) {
        ort_err = "FastDVDnet CUDA tensor path is unavailable: " + cuda_err;
      }
    }

    if (ort_err.empty()) {
      std::string cuda_err;
      bool upload_ok = true;
      ++last_tensor_run_stats_.cuda_tensor_upload_calls;
      if (!cuda_noisy_tensor_.UploadFromCpuF32(&cuda_, noisy_tensor_.data(),
                                               noisy_tensor_.size(),
                                               cuda_stream_, &cuda_err)) {
        upload_ok = false;
      }
      if (upload_ok) {
        ++last_tensor_run_stats_.cuda_tensor_upload_calls;
        if (!cuda_noise_map_tensor_.UploadFromCpuF32(
                &cuda_, noise_map_tensor_.data(), noise_map_tensor_.size(),
                cuda_stream_, &cuda_err)) {
          upload_ok = false;
        }
      }
      if (!upload_ok) {
        ort_err = "FastDVDnet CUDA tensor upload failed: " + cuda_err;
      }
    }

    if (ort_err.empty() && session_info_.cuda_needs_stream_sync) {
      std::string sync_err;
      ++last_tensor_run_stats_.forced_sync_calls;
      if (!cuda_.StreamSynchronize(cuda_stream_, &sync_err)) {
        ort_err = "FastDVDnet CUDA tensor upload sync failed: " + sync_err;
      }
    }

    if (ort_err.empty()) {
      cuda_ort_inputs_[0].device_ptr = reinterpret_cast<const float *>(
          static_cast<std::uintptr_t>(cuda_noisy_tensor_.ptr));
      cuda_ort_inputs_[0].num_floats = cuda_noisy_tensor_.ElementCount();
      cuda_ort_inputs_[1].device_ptr = reinterpret_cast<const float *>(
          static_cast<std::uintptr_t>(cuda_noise_map_tensor_.ptr));
      cuda_ort_inputs_[1].num_floats = cuda_noise_map_tensor_.ElementCount();
      cuda_ort_outputs_[0].device_ptr = reinterpret_cast<float *>(
          static_cast<std::uintptr_t>(cuda_denoised_tensor_.ptr));
      cuda_ort_outputs_[0].num_floats = cuda_denoised_tensor_.ElementCount();

      ort_ok = ort_session_active_->RunCudaIoBinding(
          cuda_ort_inputs_.data(), cuda_ort_inputs_.size(),
          cuda_ort_outputs_.data(), cuda_ort_outputs_.size(), &ort_err);
      if (ort_ok && session_info_.cuda_needs_stream_sync) {
        ++last_tensor_run_stats_.forced_sync_calls;
      }
    }

    if (ort_ok) {
      std::string cuda_err;
      ++last_tensor_run_stats_.cuda_tensor_download_calls;
      if (!cuda_denoised_tensor_.DownloadToCpuF32(&cuda_, &denoised_tensor_,
                                                  cuda_stream_, &cuda_err)) {
        ort_ok = false;
        ort_err = "FastDVDnet CUDA tensor download failed: " + cuda_err;
      } else {
        ++last_tensor_run_stats_.forced_sync_calls;
        if (!cuda_.StreamSynchronize(cuda_stream_, &cuda_err)) {
          ort_ok = false;
          ort_err = "FastDVDnet CUDA tensor download sync failed: " + cuda_err;
        }
      }
    }
  } else {
    if (active_session_uses_cuda_ep()) {
      last_tensor_run_stats_.used_cuda_ep_cpu_tensor_io = true;
    } else {
      last_tensor_run_stats_.used_cpu_session = true;
      last_tensor_run_stats_.cpu_only_fallback_active = true;
    }
    ort_ok = ort_session_active_->RunCpu(ort_inputs_.data(), ort_inputs_.size(),
                                         ort_outputs_.data(),
                                         ort_outputs_.size(), &ort_err);
  }

  if (!ort_ok) {
    std::ostringstream oss;
    oss << "Open Video FastDVDnet ORT run failed: "
        << (ort_err.empty() ? "unknown" : ort_err);

    // If CUDA is active and CPU fallback exists, switch once.
    if (!using_cpu_fallback_ && ort_session_cuda_ && ort_session_cpu_ &&
        session_info_.using_cuda) {
      ort_session_active_ = ort_session_cpu_.get();
      using_cpu_fallback_ = true;
      ResetCudaTensorIo();
      last_tensor_run_stats_.used_cpu_session = true;
      last_tensor_run_stats_.cpu_only_fallback_active = true;
      sticky_warning_ = "Open Video FastDVDnet: switched to CPU fallback after "
                        "a CUDA runtime failure.";
      runtime_failures_ = 0;
      // Try once on CPU immediately.
      std::string cpu_err;
      if (!ort_session_active_->RunCpu(ort_inputs_.data(), ort_inputs_.size(),
                                       ort_outputs_.data(), ort_outputs_.size(),
                                       &cpu_err)) {
        oss << " (CPU fallback also failed: "
            << (cpu_err.empty() ? "unknown" : cpu_err) << ")";
      } else {
        // CPU fallback succeeded.
        PostprocessToRgbInPlace(rgb, width, height, stride);
        return true;
      }
    }

    runtime_failures_++;
    if (runtime_failures_ >= 3) {
      DisableAfterFailure(
          "Open Video FastDVDnet: disabled after repeated runtime failures.");
    }

    if (error)
      *error = oss.str();
    return false;
  }

  runtime_failures_ = 0;

  // Convert output back into the RGB buffer.
  PostprocessToRgbInPlace(rgb, width, height, stride);
  return true;
}

} // namespace studiocast::open_video
