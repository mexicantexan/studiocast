#include "core/maxine/effects/vfx_denoise_effect.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace studiocast::maxine::effects {

VfxDenoiseEffect::VfxDenoiseEffect(maxine::vfx::VfxApi *vfx,
                                   maxine::NvcvApi *nvcv,
                                   std::filesystem::path model_dir)
    : vfx_(vfx), nvcv_(nvcv), model_dir_(std::move(model_dir)) {}

VfxDenoiseEffect::~VfxDenoiseEffect() { Destroy(); }

bool VfxDenoiseEffect::SetExternalCudaStream(maxine::CUstream stream,
                                             std::string *error) {
  if (!stream) {
    if (error)
      *error = "External CUDA stream must be non-null.";
    return false;
  }
  if (stream_ == stream && !own_stream_)
    return !handle_ || EnsureStreamBound(error);
  if (handle_) {
    const auto status = vfx_->f().NvVFX_SetCudaStream(
        handle_, maxine::vfx::NVVFX_CUDA_STREAM, stream);
    if (status != maxine::NVCV_SUCCESS) {
      if (error)
        *error = "NvVFX_SetCudaStream failed: " + vfx_->StatusToString(status);
      return false;
    }
  }
  if (own_stream_ && stream_ && vfx_ && vfx_->IsInitialized() &&
      vfx_->f().NvVFX_CudaStreamDestroy)
    vfx_->f().NvVFX_CudaStreamDestroy(stream_);
  stream_ = stream;
  external_stream_ = stream;
  external_stream_selected_ = true;
  own_stream_ = false;
  stream_bound_ = handle_ != nullptr;
  return true;
}

void VfxDenoiseEffect::InvalidateBindings() noexcept {
  stream_bound_ = false;
  bound_input_ = nullptr;
  bound_output_ = nullptr;
  bound_width_ = 0;
  bound_height_ = 0;
  output_ready_ = false;
}

void VfxDenoiseEffect::Destroy() {
  output_ready_ = false;

  if (out_allocated_ && nvcv_ && nvcv_->IsInitialized() &&
      nvcv_->f().NvCVImage_Dealloc) {
    (void)nvcv_->f().NvCVImage_Dealloc(&out_gpu_);
  }
  out_gpu_ = maxine::NvCVImage{};
  out_allocated_ = false;

  if (state_device_ && vfx_ && vfx_->HasCudaRuntime() &&
      vfx_->cuda().cudaFree) {
    (void)vfx_->cuda().cudaFree(state_device_);
  }
  state_device_ = nullptr;
  state_bytes_ = 0;
  state_bound_ = false;

  if (handle_ && vfx_ && vfx_->IsInitialized() &&
      vfx_->f().NvVFX_DestroyEffect) {
    vfx_->f().NvVFX_DestroyEffect(handle_);
  }
  handle_ = nullptr;

  if (own_stream_ && stream_ && vfx_ && vfx_->IsInitialized() &&
      vfx_->f().NvVFX_CudaStreamDestroy) {
    vfx_->f().NvVFX_CudaStreamDestroy(stream_);
  }
  stream_ = nullptr;
  own_stream_ = false;
  model_bound_ = false;
  cfg_dirty_ = true;
  InvalidateBindings();
}

std::string VfxDenoiseEffect::CudaErrorToString(
    maxine::vfx::VfxApi::cudaError_t err) const {
  if (vfx_ && vfx_->HasCudaRuntime() && vfx_->cuda().cudaGetErrorString) {
    const char *s = vfx_->cuda().cudaGetErrorString(err);
    if (s && *s)
      return std::string(s);
  }
  std::ostringstream oss;
  oss << "cudaError(" << err << ")";
  return oss.str();
}

bool VfxDenoiseEffect::EnsureEffectCreated(std::string *error) {
  if (handle_)
    return true;
  if (!vfx_ || !vfx_->IsInitialized()) {
    if (error)
      *error = "VFX runtime not initialized.";
    return false;
  }

  const auto &f = vfx_->f();
  const maxine::NvCV_Status st =
      f.NvVFX_CreateEffect(maxine::vfx::NVVFX_FX_DENOISING, &handle_);
  if (st != maxine::NVCV_SUCCESS || !handle_) {
    if (error) {
      *error =
          "NvVFX_CreateEffect(Denoising) failed: " + vfx_->StatusToString(st);
    }
    handle_ = nullptr;
    return false;
  }

  if (external_stream_selected_) {
    stream_ = external_stream_;
    own_stream_ = false;
  } else if (!stream_ && f.NvVFX_CudaStreamCreate) {
    maxine::CUstream s = nullptr;
    const auto st2 = f.NvVFX_CudaStreamCreate(&s);
    if (st2 == maxine::NVCV_SUCCESS && s) {
      stream_ = s;
      own_stream_ = true;
    }
  }

  if (!stream_) {
    if (error)
      *error = "Denoising CUDA stream unavailable.";
    Destroy();
    return false;
  }
  if (!EnsureStreamBound(error)) {
    Destroy();
    return false;
  }
  if (f.NvVFX_SetString && !model_dir_.empty()) {
    const auto model = model_dir_.string();
    const auto model_status = f.NvVFX_SetString(
        handle_, maxine::vfx::NVVFX_MODEL_DIRECTORY, model.c_str());
    if (model_status != maxine::NVCV_SUCCESS) {
      if (error)
        *error = "NvVFX_SetString(modelDir) failed: " +
                 vfx_->StatusToString(model_status);
      Destroy();
      return false;
    }
  }
  model_bound_ = true;

  cfg_dirty_ = true;
  state_bound_ = false;
  return true;
}

bool VfxDenoiseEffect::QueryStateBytesLocked(std::size_t *out_bytes,
                                             std::string *error) {
  if (!out_bytes)
    return false;
  *out_bytes = 0;
  if (!handle_) {
    if (error)
      *error = "Denoising effect not created.";
    return false;
  }

  // Denoising state size selector varies across SDK versions; probe.
  const char *candidates[] = {
      maxine::vfx::NVVFX_STATE_SIZE,
      maxine::vfx::NVVFX_STATE_SIZE_BYTES,
      "stateSizeInBytes",
      "StateSize",
  };

  for (const char *name : candidates) {
    if (!name)
      continue;
    std::uint32_t v = 0;
    const auto st = vfx_->f().NvVFX_GetU32(handle_, name, &v);
    if (st == maxine::NVCV_SUCCESS && v > 0) {
      *out_bytes = static_cast<std::size_t>(v);
      return true;
    }
  }

  if (error) {
    *error =
        "Could not query denoise state size (no working stateSize selector).";
  }
  return false;
}

bool VfxDenoiseEffect::EnsureStateBufferLocked(std::string *error) {
  if (!vfx_ || !vfx_->HasCudaRuntime()) {
    if (error)
      *error = "CUDA runtime (libcudart) not available; denoise state "
               "allocation requires cudaMalloc.";
    return false;
  }
  const auto &c = vfx_->cuda();
  if (!c.cudaMalloc || !c.cudaFree || !c.cudaMemset) {
    if (error)
      *error = "CUDA runtime missing required symbols "
               "(cudaMalloc/cudaFree/cudaMemset).";
    return false;
  }

  std::size_t want = 0;
  std::string qerr;
  if (!QueryStateBytesLocked(&want, &qerr)) {
    if (error)
      *error = qerr;
    return false;
  }

  if (want == 0) {
    if (error)
      *error = "Denoise state size reported as 0.";
    return false;
  }

  if (!state_device_ || state_bytes_ != want) {
    if (state_device_) {
      (void)c.cudaFree(state_device_);
      state_device_ = nullptr;
      state_bytes_ = 0;
      state_bound_ = false;
    }

    void *p = nullptr;
    const auto rc = c.cudaMalloc(&p, want);
    if (rc != 0 || !p) {
      if (error) {
        *error = "cudaMalloc(" + std::to_string(want) +
                 ") failed: " + CudaErrorToString(rc);
      }
      return false;
    }
    state_device_ = p;
    state_bytes_ = want;

    const auto rc2 = c.cudaMemset(state_device_, 0, state_bytes_);
    if (rc2 != 0) {
      if (error) {
        *error = "cudaMemset(state,0) failed: " + CudaErrorToString(rc2);
      }
      return false;
    }
  }

  if (!state_bound_) {
    const auto st = vfx_->f().NvVFX_SetObject(handle_, maxine::vfx::NVVFX_STATE,
                                              state_device_);
    if (st != maxine::NVCV_SUCCESS) {
      if (error) {
        *error = "NvVFX_SetObject(state) failed: " + vfx_->StatusToString(st);
      }
      return false;
    }
    state_bound_ = true;
  }

  return true;
}

bool VfxDenoiseEffect::ApplyConfigLocked(std::string *error) {
  if (!cfg_dirty_)
    return true;
  if (!handle_) {
    if (error)
      *error = "Denoising effect not created.";
    return false;
  }

  const auto &f = vfx_->f();

  // Strength: quantized to discrete levels (filter supports limited steps).
  const float q = QuantizeStrength01(strength_);
  const auto strength_status =
      f.NvVFX_SetF32(handle_, maxine::vfx::NVVFX_STRENGTH, q);
  if (strength_status != maxine::NVCV_SUCCESS) {
    if (error)
      *error = "NvVFX_SetF32(strength) failed: " +
               vfx_->StatusToString(strength_status);
    return false;
  }

  std::string st_err;
  if (!EnsureStateBufferLocked(&st_err)) {
    if (error)
      *error = st_err;
    return false;
  }

  const auto st = f.NvVFX_Load(handle_);
  if (st != maxine::NVCV_SUCCESS) {
    if (error) {
      *error = "NvVFX_Load(Denoising) failed: " + vfx_->StatusToString(st);
    }
    return false;
  }

  cfg_dirty_ = false;
  return true;
}

bool VfxDenoiseEffect::EnsureStreamBound(std::string *error) {
  if (stream_bound_)
    return true;
  if (!handle_ || !stream_ || !vfx_->f().NvVFX_SetCudaStream) {
    if (error)
      *error = "Denoising CUDA stream binding unavailable.";
    return false;
  }
  const auto status = vfx_->f().NvVFX_SetCudaStream(
      handle_, maxine::vfx::NVVFX_CUDA_STREAM, stream_);
  if (status != maxine::NVCV_SUCCESS) {
    if (error)
      *error = "NvVFX_SetCudaStream failed: " + vfx_->StatusToString(status);
    return false;
  }
  stream_bound_ = true;
  return true;
}

bool VfxDenoiseEffect::EnsureOutputImage(unsigned width, unsigned height,
                                         std::string *error) {
  if (!nvcv_ || !nvcv_->IsInitialized() || !nvcv_->f().NvCVImage_Alloc) {
    if (error)
      *error = "NvCVImage runtime not initialized.";
    return false;
  }

  const auto want_pf = maxine::NVCV_BGR;
  const auto want_ct = maxine::NVCV_F32;
  const unsigned want_layout = maxine::NVCV_PLANAR;
  const unsigned want_mem = maxine::NVCV_GPU;

  if (out_allocated_) {
    if (out_gpu_.width == width && out_gpu_.height == height &&
        out_gpu_.pixelFormat == want_pf && out_gpu_.componentType == want_ct &&
        out_gpu_.planar == want_layout && out_gpu_.gpuMem == want_mem) {
      return true;
    }

    if (nvcv_->f().NvCVImage_Realloc) {
      const auto st = nvcv_->f().NvCVImage_Realloc(
          &out_gpu_, width, height, want_pf, want_ct, want_layout, want_mem,
          /*alignment=*/0);
      if (st == maxine::NVCV_SUCCESS) {
        bound_input_ = nullptr;
        bound_output_ = nullptr;
        bound_width_ = width;
        bound_height_ = height;
        return true;
      }
    }

    if (nvcv_->f().NvCVImage_Dealloc) {
      (void)nvcv_->f().NvCVImage_Dealloc(&out_gpu_);
    }
    out_gpu_ = maxine::NvCVImage{};
    out_allocated_ = false;
  }

  const auto st =
      nvcv_->f().NvCVImage_Alloc(&out_gpu_, width, height, want_pf, want_ct,
                                 want_layout, want_mem, /*alignment=*/0);
  if (st != maxine::NVCV_SUCCESS) {
    if (error)
      *error = "NvCVImage_Alloc(denoise output) failed: " + std::to_string(st);
    return false;
  }
  out_allocated_ = true;
  bound_input_ = nullptr;
  bound_output_ = nullptr;
  bound_width_ = width;
  bound_height_ = height;
  return true;
}

float VfxDenoiseEffect::QuantizeStrength01(float strength01) {
  const float s = std::clamp(strength01, 0.0f, 1.0f);
  // Denoising strength supports limited discrete levels; quantize to quarters.
  const int steps = static_cast<int>(std::lround(s * 4.0f));
  return static_cast<float>(steps) / 4.0f;
}

bool VfxDenoiseEffect::Initialize(std::string *error) {
  if (!EnsureEffectCreated(error))
    return false;
  if (!ApplyConfigLocked(error))
    return false;
  return true;
}

bool VfxDenoiseEffect::Configure(
    const studiocast::video::effects::BroadcastCameraEffects &settings,
    std::string *) {
  // Strength is carried as an integer percentage in the canonical model.
  const float s = static_cast<float>(std::clamp(
                      settings.video_noise_removal.strength, 0, 100)) /
                  100.0f;
  if (std::fabs(s - strength_) > 1e-6f) {
    strength_ = s;
    cfg_dirty_ = true;
  }
  return true;
}

maxine::NvCV_Status
VfxDenoiseEffect::Process(studiocast::video::GpuFrame &frame,
                          std::string *error) {
  output_ready_ = false;
  if (!frame.ValidDimensions() || !frame.nvcv_gpu) {
    if (error)
      *error = "Denoising requires frame.nvcv_gpu and valid dimensions.";
    return -1;
  }

  std::string err;
  if (!Initialize(&err)) {
    if (error)
      *error = err;
    return -1;
  }
  if (!EnsureStreamBound(error))
    return -1;

  if (!EnsureOutputImage(static_cast<unsigned>(frame.width),
                         static_cast<unsigned>(frame.height), &err)) {
    if (error)
      *error = err;
    return -1;
  }

  const auto &f = vfx_->f();
  const auto width = static_cast<unsigned>(frame.width);
  const auto height = static_cast<unsigned>(frame.height);
  if (bound_width_ != width || bound_height_ != height) {
    bound_input_ = nullptr;
    bound_output_ = nullptr;
    bound_width_ = width;
    bound_height_ = height;
  }
  if (bound_input_ != frame.nvcv_gpu) {
    const auto bind = f.NvVFX_SetImage(handle_, maxine::vfx::NVVFX_INPUT_IMAGE,
                                       frame.nvcv_gpu);
    if (bind != maxine::NVCV_SUCCESS) {
      if (error)
        *error = "NvVFX_SetImage(denoise input) failed: " +
                 vfx_->StatusToString(bind);
      return bind;
    }
    bound_input_ = frame.nvcv_gpu;
  }
  if (bound_output_ != &out_gpu_) {
    const auto bind =
        f.NvVFX_SetImage(handle_, maxine::vfx::NVVFX_OUTPUT_IMAGE, &out_gpu_);
    if (bind != maxine::NVCV_SUCCESS) {
      if (error)
        *error = "NvVFX_SetImage(denoise output) failed: " +
                 vfx_->StatusToString(bind);
      return bind;
    }
    bound_output_ = &out_gpu_;
  }

  ++execution_telemetry_.asynchronous_run_attempts;
  const auto st = f.NvVFX_Run(handle_, /*async=*/1);
  if (st != maxine::NVCV_SUCCESS) {
    InvalidateBindings();
    if (error)
      *error = "NvVFX_Run(Denoising) failed: " + vfx_->StatusToString(st);
    return st;
  }
  ++execution_telemetry_.asynchronous_run_successes;

  output_ready_ = true;
  return maxine::NVCV_SUCCESS;
}

} // namespace studiocast::maxine::effects
