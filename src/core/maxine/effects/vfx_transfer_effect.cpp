#include "core/maxine/effects/vfx_transfer_effect.h"

#include <sstream>

namespace studiocast::maxine::effects {

VfxTransferEffect::VfxTransferEffect(maxine::vfx::VfxApi *vfx,
                                     maxine::NvcvApi *nvcv,
                                     std::filesystem::path model_dir,
                                     OutputFormat out_fmt)
    : vfx_(vfx), nvcv_(nvcv), model_dir_(std::move(model_dir)),
      out_fmt_(out_fmt) {}

VfxTransferEffect::~VfxTransferEffect() { Destroy(); }

bool VfxTransferEffect::SetExternalCudaStream(maxine::CUstream stream,
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

void VfxTransferEffect::InvalidateBindings() noexcept {
  stream_bound_ = false;
  bound_input_ = nullptr;
  bound_output_ = nullptr;
  bound_width_ = 0;
  bound_height_ = 0;
  output_ready_ = false;
}

void VfxTransferEffect::Destroy() {
  output_ready_ = false;

  if (out_allocated_ && nvcv_ && nvcv_->IsInitialized() &&
      nvcv_->f().NvCVImage_Dealloc) {
    (void)nvcv_->f().NvCVImage_Dealloc(&out_gpu_);
  }
  out_gpu_ = maxine::NvCVImage{};
  out_allocated_ = false;

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

bool VfxTransferEffect::EnsureEffectCreated(std::string *error) {
  if (handle_)
    return true;
  if (!vfx_ || !vfx_->IsInitialized()) {
    if (error)
      *error = "VFX runtime not initialized.";
    return false;
  }

  const auto &f = vfx_->f();
  if (!f.NvVFX_CreateEffect) {
    if (error)
      *error = "NvVFX_CreateEffect missing.";
    return false;
  }

  const maxine::NvCV_Status st =
      f.NvVFX_CreateEffect(maxine::vfx::NVVFX_FX_TRANSFER, &handle_);
  if (st != maxine::NVCV_SUCCESS || !handle_) {
    if (error) {
      std::ostringstream oss;
      oss << "NvVFX_CreateEffect(Transfer) failed: "
          << vfx_->StatusToString(st);
      *error = oss.str();
    }
    handle_ = nullptr;
    return false;
  }

  // Create a dedicated CUDA stream for this effect (to match other wrappers).
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
      *error = "Transfer CUDA stream unavailable.";
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
  return true;
}

bool VfxTransferEffect::ApplyConfigLocked(std::string *error) {
  if (!cfg_dirty_)
    return true;
  if (!handle_) {
    if (error)
      *error = "Transfer effect not created.";
    return false;
  }
  const auto &f = vfx_->f();

  const auto st = f.NvVFX_Load(handle_);
  if (st != maxine::NVCV_SUCCESS) {
    if (error) {
      std::ostringstream oss;
      oss << "NvVFX_Load(Transfer) failed: " << vfx_->StatusToString(st);
      *error = oss.str();
    }
    return false;
  }

  cfg_dirty_ = false;
  return true;
}

bool VfxTransferEffect::EnsureStreamBound(std::string *error) {
  if (stream_bound_)
    return true;
  if (!handle_ || !stream_ || !vfx_->f().NvVFX_SetCudaStream) {
    if (error)
      *error = "Transfer CUDA stream binding unavailable.";
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

bool VfxTransferEffect::EnsureOutputImage(unsigned width, unsigned height,
                                          std::string *error) {
  if (!nvcv_ || !nvcv_->IsInitialized() || !nvcv_->f().NvCVImage_Alloc) {
    if (error)
      *error = "NvCVImage runtime not initialized.";
    return false;
  }

  // (Re)allocate output image for the requested dimensions/format.
  if (out_allocated_) {
    if (out_gpu_.width == width && out_gpu_.height == height &&
        out_gpu_.pixelFormat == out_fmt_.pixel_format &&
        out_gpu_.componentType == out_fmt_.component_type &&
        out_gpu_.planar == out_fmt_.layout &&
        out_gpu_.gpuMem == out_fmt_.mem_space) {
      return true;
    }

    if (nvcv_->f().NvCVImage_Realloc) {
      const auto st = nvcv_->f().NvCVImage_Realloc(
          &out_gpu_, width, height, out_fmt_.pixel_format,
          out_fmt_.component_type, out_fmt_.layout, out_fmt_.mem_space,
          /*alignment=*/0);
      if (st == maxine::NVCV_SUCCESS) {
        bound_input_ = nullptr;
        bound_output_ = nullptr;
        bound_width_ = width;
        bound_height_ = height;
        return true;
      }
    }

    // Fall back to dealloc + alloc.
    if (nvcv_->f().NvCVImage_Dealloc) {
      (void)nvcv_->f().NvCVImage_Dealloc(&out_gpu_);
    }
    out_gpu_ = maxine::NvCVImage{};
    out_allocated_ = false;
  }

  const auto st = nvcv_->f().NvCVImage_Alloc(
      &out_gpu_, width, height, out_fmt_.pixel_format, out_fmt_.component_type,
      out_fmt_.layout, out_fmt_.mem_space,
      /*alignment=*/0);
  if (st != maxine::NVCV_SUCCESS) {
    if (error) {
      *error = "NvCVImage_Alloc(Transfer output) failed: " + std::to_string(st);
    }
    return false;
  }
  out_allocated_ = true;
  bound_input_ = nullptr;
  bound_output_ = nullptr;
  bound_width_ = width;
  bound_height_ = height;
  return true;
}

bool VfxTransferEffect::Initialize(std::string *error) {
  if (!EnsureEffectCreated(error))
    return false;
  if (!ApplyConfigLocked(error))
    return false;
  return true;
}

bool VfxTransferEffect::Configure(
    const studiocast::video::effects::BroadcastCameraEffects &, std::string *) {
  // Transfer has no user-facing configuration in StudioCast.
  return true;
}

maxine::NvCV_Status
VfxTransferEffect::Process(studiocast::video::GpuFrame &frame,
                           std::string *error) {
  output_ready_ = false;
  if (!frame.ValidDimensions() || !frame.nvcv_gpu) {
    if (error)
      *error = "Transfer requires frame.nvcv_gpu and valid dimensions.";
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
        *error = "NvVFX_SetImage(transfer input) failed: " +
                 vfx_->StatusToString(bind);
      return bind;
    }
    bound_input_ = frame.nvcv_gpu;
  }
  if (bound_output_ != &out_gpu_) {
    const auto bind = f.NvVFX_SetImage(
        handle_, maxine::vfx::NVVFX_OUTPUT_IMAGE, &out_gpu_);
    if (bind != maxine::NVCV_SUCCESS) {
      if (error)
        *error = "NvVFX_SetImage(transfer output) failed: " +
                 vfx_->StatusToString(bind);
      return bind;
    }
    bound_output_ = &out_gpu_;
  }

  const auto st = f.NvVFX_Run(handle_, /*async=*/1);
  if (st != maxine::NVCV_SUCCESS) {
    InvalidateBindings();
    if (error) {
      *error = "NvVFX_Run(Transfer) failed: " + vfx_->StatusToString(st);
    }
    return st;
  }

  output_ready_ = true;
  return maxine::NVCV_SUCCESS;
}

} // namespace studiocast::maxine::effects
