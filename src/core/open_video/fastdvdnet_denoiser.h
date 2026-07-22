#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "core/cuda/cuda_tensor.h"
#include "core/cuda/cuda_driver_api.h"
#include "core/onnx/ort_session.h"
#include "core/open_video/model_pack_registry.h"

namespace studiocast::open_video {

struct DenoiseTensorSpec {
  std::string role;
  std::string name;
  std::string element_type;
  std::string layout;
  std::vector<int64_t> shape;
};

struct DenoiseTemporalWindowSpec {
  int window_frames = 0;
  int history_frames = 0;
  int repeated_future_frames = 0;
  bool causal = true;
};

struct DenoiseNormalizationSpec {
  std::string frame_input;
  std::string strength_input;
  float max_sigma = 0.0f;
};

struct DenoiseTensorAdapterContract {
  std::string adapter_id;
  std::string model_family;
  std::vector<DenoiseTensorSpec> inputs;
  DenoiseTensorSpec output;
  DenoiseTemporalWindowSpec temporal;
  DenoiseNormalizationSpec normalization;

  bool supports_cpu_tensor_io = true;
  bool supports_cuda_device_tensor_io = false;

  bool requires_cpu_preprocess = true;
  bool requires_cpu_postprocess = true;
  bool requires_output_device_to_cpu_for_postprocess = false;
};

struct DenoiseTensorIoStatus {
  bool cuda_ep_active = false;
  bool cuda_device_tensor_io_supported = false;
  bool cuda_device_tensor_io_active = false;
  bool cuda_ep_cpu_tensor_io_active = false;
  bool cpu_only_fallback_active = false;
  bool cpu_tensor_tail_active = false;
  bool output_readback_required_for_postprocess = false;
  std::string summary;
};

struct DenoiseTensorRunStats {
  bool cuda_ep_active = false;
  bool used_cuda_device_tensor_io = false;
  bool used_cuda_ep_cpu_tensor_io = false;
  bool used_cpu_session = false;
  bool cpu_only_fallback_active = false;
  bool cpu_tensor_tail_active = false;

  std::uint64_t cuda_tensor_upload_calls = 0;
  std::uint64_t cuda_tensor_download_calls = 0;
  std::uint64_t forced_sync_calls = 0;
  std::uint64_t cpu_tail_stage_calls = 0;
};

// Streaming FastDVDnet video denoiser (ONNX Runtime).
//
// This is used as the preferred open-source implementation for the
// "Video Noise Removal" effect when a compatible Open Video model pack is
// installed.
class FastDvdnetDenoiser {
public:
  FastDvdnetDenoiser();
  ~FastDvdnetDenoiser();

  FastDvdnetDenoiser(const FastDvdnetDenoiser &) = delete;
  FastDvdnetDenoiser &operator=(const FastDvdnetDenoiser &) = delete;

  // Attempt to initialize the denoiser using the default Open Video model root.
  //
  // The chosen model pack is selected from installed packs under
  // task="video_denoise". If multiple packs are installed, a deterministic
  // preference order is used.
  bool EnsureInitialized(int src_w, int src_h,
                         const std::string &requested_model_id,
                         std::string *error);

  // Reset temporal windowing state (but keep the loaded model/session).
  void ResetTemporalState();

  // Apply denoising in-place on an RGB24 frame.
  //
  // Returns true on success, false on fatal errors (in which case the caller
  // should bypass the effect).
  bool ApplyRgbInPlace(std::uint64_t capture_sequence, std::uint8_t *rgb,
                       int width, int height, std::size_t stride, int strength,
                       const std::string &requested_model_id,
                       std::string *error);

  // Diagnostics.
  bool initialized() const { return initialized_; }
  bool disabled() const { return disabled_; }
  bool using_cpu_fallback() const { return using_cpu_fallback_; }
  bool active_session_uses_cuda_ep() const {
    return ort_session_active_ != nullptr && session_info_.using_cuda &&
           !using_cpu_fallback_;
  }
  bool active_session_uses_cuda_tensor_io() const;
  bool active_session_uses_cpu_tensor_io() const;
  const DenoiseTensorAdapterContract &tensor_io_contract() const {
    return tensor_contract_;
  }
  DenoiseTensorIoStatus tensor_io_status() const;
  const DenoiseTensorRunStats &last_tensor_run_stats() const {
    return last_tensor_run_stats_;
  }
  const std::string &active_model_id() const { return active_model_id_; }
  const std::filesystem::path &active_model_path() const {
    return active_model_path_;
  }
  const std::string &sticky_warning() const { return sticky_warning_; }

private:
  struct LoadedModel {
    std::string id;
    std::filesystem::path onnx;
    int default_sigma = 25; // best-effort hint, not required
  };

  static int AlignUp(int v, int align);
  static float Clamp01(float x);

  static std::string ChoosePreferredModelId(const ModelPackRegistry &reg);
  static bool
  LoadDefaultSigmaFromManifest(const std::filesystem::path &manifest_path,
                               int *out_sigma);
  static bool ResolveModelFromRegistry(const ModelPackRegistry &reg,
                                       const std::string &requested_model_id,
                                       LoadedModel *out, std::string *error);

  bool EnsureSessionForModel(const LoadedModel &model, std::string *error);
  bool RefreshGeometry(int src_w, int src_h, std::string *error);
  bool DetectIoNames(std::string *error);
  bool EnsureCudaTensorIo(std::string *error);
  void ResetCudaTensorIo();
  void RefreshTensorContract();
  int HistoryFrameCount() const;

  void PreprocessRgbToChwPadded(const std::uint8_t *rgb, int width, int height,
                                std::size_t stride,
                                std::vector<float> *out_chw) const;

  void BuildNoisyTensorFromHistory(std::vector<float> *out_noisy) const;
  void EnsureNoiseMap(float sigma_over_255);
  void PostprocessToRgbInPlace(std::uint8_t *rgb, int width, int height,
                               std::size_t stride) const;

  void DisableAfterFailure(const std::string &why);

  bool initialized_ = false;
  bool disabled_ = false;

  // Model/session state.
  ModelPackRegistry registry_;
  std::string active_requested_model_id_;
  std::string active_model_id_;
  std::filesystem::path active_model_path_;
  int model_default_sigma_ = 25;

  studiocast::onnx::OrtSessionInfo session_info_;
  std::unique_ptr<studiocast::onnx::OrtSession> ort_session_cuda_;
  std::unique_ptr<studiocast::onnx::OrtSession> ort_session_cpu_;
  studiocast::onnx::OrtSession *ort_session_active_ = nullptr;
  bool using_cpu_fallback_ = false;

  std::string noisy_name_;
  std::string noise_map_name_;
  std::string denoised_name_;

  DenoiseTensorAdapterContract tensor_contract_;
  DenoiseTensorRunStats last_tensor_run_stats_;

  // Geometry for the current session run.
  int proc_w_ = 0;
  int proc_h_ = 0;
  int src_w_ = 0;
  int src_h_ = 0;

  std::vector<int64_t> noisy_shape_;
  std::vector<int64_t> noise_map_shape_;
  std::vector<int64_t> denoised_shape_;

  // Temporal history (preprocessed CHW frames, proc_w_*proc_h_).
  std::vector<std::vector<float>> history_;
  int history_filled_ = 0;
  int history_write_idx_ = 0;
  std::uint64_t last_capture_sequence_ = 0;
  bool have_last_sequence_ = false;

  // ORT I/O buffers.
  std::vector<float> noisy_tensor_;
  std::vector<float> noise_map_tensor_;
  std::vector<float> denoised_tensor_;
  float last_noise_map_value_ = -1.0f;

  // Scratch for ORT bindings.
  std::vector<studiocast::onnx::OrtSession::RunInput> ort_inputs_;
  std::vector<studiocast::onnx::OrtSession::RunOutput> ort_outputs_;

  // Optional CUDA ORT IoBinding buffers. These only cover ONNX tensor I/O;
  // frame preprocessing and output postprocess remain CPU-side in this class.
  studiocast::cuda::CudaDriverApi cuda_;
  studiocast::cuda::CUstream cuda_stream_ = nullptr;
  bool cuda_stream_owned_ = false;
  bool cuda_tensor_io_ready_ = false;
  studiocast::cuda::CudaTensor cuda_noisy_tensor_;
  studiocast::cuda::CudaTensor cuda_noise_map_tensor_;
  studiocast::cuda::CudaTensor cuda_denoised_tensor_;
  std::vector<studiocast::onnx::OrtSession::CudaBindingInput> cuda_ort_inputs_;
  std::vector<studiocast::onnx::OrtSession::CudaBindingOutput>
      cuda_ort_outputs_;

  // Sticky warning surfaced through the daemon status (best-effort).
  std::string sticky_warning_;
  int runtime_failures_ = 0;
};

} // namespace studiocast::open_video
