#include "core/maxine/effects/vfx_background_blur_effect.h"

#include <algorithm>
#include <sstream>

namespace studiocast::maxine::effects {

namespace {

std::string StatusToString(const maxine::vfx::VfxApi *vfx,
                           const maxine::NvcvApi *nvcv, maxine::NvCV_Status s) {
  if (vfx && vfx->IsInitialized()) {
    return vfx->StatusToString(s);
  }
  if (nvcv && nvcv->IsInitialized() && nvcv->f().NvCV_GetErrorStringFromCode) {
    const char *msg = nvcv->f().NvCV_GetErrorStringFromCode(s);
    if (msg)
      return msg;
  }
  std::ostringstream oss;
  oss << "NvCV_Status(" << s << ")";
  return oss.str();
}

float Clamp01(float v) { return std::max(0.0f, std::min(1.0f, v)); }

// Canonical UI model uses a legacy-style integer strength knob for background
// blur. Map [1..64] -> [0..1].
float MapStrengthToUnitInterval(int strength) {
  // Be permissive in case callers supply 0..1 or other ranges.
  if (strength <= 1)
    return 0.0f;
  if (strength >= 64)
    return 1.0f;
  return static_cast<float>(strength - 1) / 63.0f;
}

} // namespace

VfxBackgroundBlurEffect::VfxBackgroundBlurEffect(
    maxine::vfx::VfxApi *vfx, maxine::NvcvApi *nvcv,
    std::filesystem::path model_dir)
    : vfx_(vfx), nvcv_(nvcv), model_dir_(std::move(model_dir)) {
  output_gpu_ = maxine::NvCVImage{};
}

VfxBackgroundBlurEffect::~VfxBackgroundBlurEffect() { Destroy(); }

void VfxBackgroundBlurEffect::SetConfig(const Config &cfg) {
  Config next = cfg;
  next.strength = Clamp01(next.strength);
  if (next.strength == cfg_.strength)
    return;
  cfg_ = next;
  cfg_dirty_ = true;
}

bool VfxBackgroundBlurEffect::SetExternalCudaStream(maxine::CUstream stream,
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
        *error = "NvVFX_SetCudaStream failed: " +
                 StatusToString(vfx_, nvcv_, status);
      return false;
    }
  }
  if (own_stream_ && stream_ && vfx_ && vfx_->IsInitialized())
    vfx_->f().NvVFX_CudaStreamDestroy(stream_);
  stream_ = stream;
  external_stream_ = stream;
  external_stream_selected_ = true;
  own_stream_ = false;
  stream_bound_ = handle_ != nullptr;
  return true;
}

void VfxBackgroundBlurEffect::InvalidateBindings() noexcept {
  stream_bound_ = false;
  bound_input_ = nullptr;
  bound_matte_ = nullptr;
  bound_output_ = nullptr;
  bound_width_ = 0;
  bound_height_ = 0;
  output_ready_ = false;
}

bool VfxBackgroundBlurEffect::Initialize(std::string *error) {
  return EnsureEffectCreated(error);
}

bool VfxBackgroundBlurEffect::Configure(
    const studiocast::video::effects::BroadcastCameraEffects &settings,
    std::string *error) {
  Config next = cfg_;
  next.strength =
      MapStrengthToUnitInterval(settings.virtual_background.strength);
  next.strength = Clamp01(next.strength);

  if (next.strength != cfg_.strength) {
    cfg_ = next;
    cfg_dirty_ = true;
  }

  if (handle_) {
    return ApplyConfigLocked(error);
  }
  return true;
}

NvCV_Status VfxBackgroundBlurEffect::Process(studiocast::video::GpuFrame &frame,
                                             std::string *error) {
  output_ready_ = false;

  if (!frame.ValidDimensions()) {
    if (error)
      *error = "Invalid frame dimensions.";
    return -1;
  }
  if (!frame.nvcv_gpu) {
    if (error)
      *error = "Background Blur requires frame.nvcv_gpu (NvCVImage on GPU).";
    return -1;
  }
  if (!frame.matte_gpu) {
    if (error)
      *error = "Background Blur requires frame.matte_gpu (Au8 matte on GPU).";
    return -1;
  }

  std::string init_err;
  if (!EnsureEffectCreated(&init_err)) {
    if (error)
      *error = init_err;
    return -1;
  }
  if (!EnsureStreamBound(error))
    return -1;

  if (cfg_dirty_) {
    std::string cfg_err;
    if (!ApplyConfigLocked(&cfg_err)) {
      if (error)
        *error = cfg_err;
      return -1;
    }
  }

  std::string out_err;
  if (!EnsureOutputImage(static_cast<unsigned>(frame.width),
                         static_cast<unsigned>(frame.height), &out_err)) {
    if (error)
      *error = out_err;
    return -1;
  }

  auto &f = vfx_->f();

  const auto width = static_cast<unsigned>(frame.width);
  const auto height = static_cast<unsigned>(frame.height);
  if (bound_width_ != width || bound_height_ != height) {
    bound_input_ = nullptr;
    bound_matte_ = nullptr;
    bound_output_ = nullptr;
    bound_width_ = width;
    bound_height_ = height;
  }
  NvCV_Status s = maxine::NVCV_SUCCESS;
  if (bound_input_ != frame.nvcv_gpu) {
    s = f.NvVFX_SetImage(handle_, maxine::vfx::NVVFX_INPUT_IMAGE,
                         frame.nvcv_gpu);
    if (s != maxine::NVCV_SUCCESS) {
      if (error)
        *error = "NvVFX_SetImage(srcImage) failed: " +
                 StatusToString(vfx_, nvcv_, s);
      bound_input_ = nullptr;
      return s;
    }
    bound_input_ = frame.nvcv_gpu;
  }
  if (!BindMatte(frame.matte_gpu, error)) {
    return -1;
  }
  if (bound_output_ != &output_gpu_) {
    s = f.NvVFX_SetImage(handle_, maxine::vfx::NVVFX_OUTPUT_IMAGE,
                         &output_gpu_);
    if (s != maxine::NVCV_SUCCESS) {
      if (error)
        *error = "NvVFX_SetImage(dstImage) failed: " +
                 StatusToString(vfx_, nvcv_, s);
      bound_output_ = nullptr;
      return s;
    }
    bound_output_ = &output_gpu_;
  }

  s = f.NvVFX_Run(handle_, /*async=*/0);
  if (s != maxine::NVCV_SUCCESS) {
    InvalidateBindings();
    if (error)
      *error = "NvVFX_Run failed: " + StatusToString(vfx_, nvcv_, s);
    return s;
  }

  output_ready_ = true;
  return s;
}

bool VfxBackgroundBlurEffect::EnsureEffectCreated(std::string *error) {
  if (!vfx_ || !vfx_->IsInitialized()) {
    if (error)
      *error = "VFX runtime not initialized (VfxApi).";
    return false;
  }
  if (!nvcv_ || !nvcv_->IsInitialized()) {
    if (error)
      *error = "NvCVImage runtime not initialized (NvcvApi).";
    return false;
  }
  if (handle_)
    return true;

  auto &f = vfx_->f();

  NvCV_Status s = f.NvVFX_CreateEffect(maxine::vfx::NVVFX_FX_BGBLUR, &handle_);
  if (s != maxine::NVCV_SUCCESS || !handle_) {
    if (error)
      *error = "NvVFX_CreateEffect(Background Blur) failed: " +
               StatusToString(vfx_, nvcv_, s);
    handle_ = nullptr;
    return false;
  }

  if (external_stream_selected_) {
    stream_ = external_stream_;
    own_stream_ = false;
  } else if (!stream_) {
    s = f.NvVFX_CudaStreamCreate(&stream_);
    if (s != maxine::NVCV_SUCCESS || !stream_) {
      if (error)
        *error = "NvVFX_CudaStreamCreate failed: " +
                 StatusToString(vfx_, nvcv_, s);
      Destroy();
      return false;
    }
    own_stream_ = true;
  }
  if (!EnsureStreamBound(error)) {
    Destroy();
    return false;
  }

  // Model directory.
  if (!model_dir_.empty()) {
    const auto model_str = model_dir_.string();
    s = f.NvVFX_SetString(handle_, maxine::vfx::NVVFX_MODEL_DIRECTORY,
                          model_str.c_str());
    if (s != maxine::NVCV_SUCCESS) {
      if (error)
        *error = "NvVFX_SetString(modelDir) failed: " +
                 StatusToString(vfx_, nvcv_, s);
      Destroy();
      return false;
    }
  }

  // Apply initial strength before Load().
  cfg_dirty_ = true;
  if (!ApplyConfigLocked(error)) {
    Destroy();
    return false;
  }

  s = f.NvVFX_Load(handle_);
  if (s != maxine::NVCV_SUCCESS) {
    if (error)
      *error = "NvVFX_Load failed: " + StatusToString(vfx_, nvcv_, s);
    Destroy();
    return false;
  }

  return true;
}

bool VfxBackgroundBlurEffect::ApplyConfigLocked(std::string *error) {
  if (!cfg_dirty_)
    return true;
  if (!handle_) {
    if (error)
      *error = "Background Blur effect not created.";
    return false;
  }

  auto &f = vfx_->f();
  const float s01 = Clamp01(cfg_.strength);
  const NvCV_Status s =
      f.NvVFX_SetF32(handle_, maxine::vfx::NVVFX_STRENGTH, s01);
  if (s != maxine::NVCV_SUCCESS) {
    if (error)
      *error =
          "NvVFX_SetF32(strength) failed: " + StatusToString(vfx_, nvcv_, s);
    return false;
  }

  cfg_dirty_ = false;
  return true;
}

bool VfxBackgroundBlurEffect::EnsureStreamBound(std::string *error) {
  if (stream_bound_)
    return true;
  if (!handle_ || !stream_) {
    if (error)
      *error = "Background Blur stream is unavailable.";
    return false;
  }
  const auto status = vfx_->f().NvVFX_SetCudaStream(
      handle_, maxine::vfx::NVVFX_CUDA_STREAM, stream_);
  if (status != maxine::NVCV_SUCCESS) {
    if (error)
      *error = "NvVFX_SetCudaStream failed: " +
               StatusToString(vfx_, nvcv_, status);
    return false;
  }
  stream_bound_ = true;
  return true;
}

bool VfxBackgroundBlurEffect::EnsureOutputImage(unsigned width, unsigned height,
                                                std::string *error) {
  if (!nvcv_ || !nvcv_->IsInitialized()) {
    if (error)
      *error = "NvCVImage runtime not initialized.";
    return false;
  }

  if (output_allocated_ && output_gpu_.width == width &&
      output_gpu_.height == height && output_gpu_.gpuMem == maxine::NVCV_GPU) {
    return true;
  }

  auto &nf = nvcv_->f();
  if (!nf.NvCVImage_Alloc || !nf.NvCVImage_Dealloc) {
    if (error)
      *error = "NvCVImage_Alloc/Dealloc unavailable.";
    return false;
  }

  if (output_allocated_) {
    (void)nf.NvCVImage_Dealloc(&output_gpu_);
    output_gpu_ = maxine::NvCVImage{};
    output_allocated_ = false;
  }

  const maxine::NvCV_Status s =
      nf.NvCVImage_Alloc(&output_gpu_, width, height, maxine::NVCV_BGR,
                         maxine::NVCV_U8, maxine::NVCV_CHUNKY, maxine::NVCV_GPU,
                         /*alignment=*/0);
  if (s != maxine::NVCV_SUCCESS) {
    if (error)
      *error = "NvCVImage_Alloc(output BGRu8 GPU) failed: " +
               StatusToString(vfx_, nvcv_, s);
    return false;
  }

  output_allocated_ = true;
  bound_input_ = nullptr;
  bound_matte_ = nullptr;
  bound_output_ = nullptr;
  bound_width_ = width;
  bound_height_ = height;
  return true;
}

bool VfxBackgroundBlurEffect::BindMatte(const maxine::NvCVImage *matte,
                                        std::string *error) {
  if (!handle_) {
    if (error)
      *error = "Background Blur effect not created.";
    return false;
  }
  if (!matte) {
    if (error)
      *error = "Null matte image.";
    return false;
  }
  if (bound_matte_ == matte)
    return true;

  auto &f = vfx_->f();

  // Primary selector per VFX docs.
  NvCV_Status s = static_cast<NvCV_Status>(-1);
  if (matte_selector_) {
    s = f.NvVFX_SetImage(handle_, matte_selector_,
                         const_cast<maxine::NvCVImage *>(matte));
    if (s == maxine::NVCV_SUCCESS) {
      bound_matte_ = matte;
      return true;
    }
    matte_selector_ = nullptr;
  }
  s = f.NvVFX_SetImage(handle_, maxine::vfx::NVVFX_INPUT_MATTE,
                       const_cast<maxine::NvCVImage *>(matte));
  if (s == maxine::NVCV_SUCCESS) {
    matte_selector_ = maxine::vfx::NVVFX_INPUT_MATTE;
    bound_matte_ = matte;
    return true;
  }

  // Some distributions/builds use different selector strings; try a small set
  // of known alternates before failing.
  static constexpr const char *kAlternates[] = {"matte", "srcMask", "mask"};
  for (const char *alt : kAlternates) {
    s = f.NvVFX_SetImage(handle_, alt, const_cast<maxine::NvCVImage *>(matte));
    if (s == maxine::NVCV_SUCCESS) {
      matte_selector_ = alt;
      bound_matte_ = matte;
      return true;
    }
  }

  if (error) {
    *error = "NvVFX_SetImage(matte) failed for selectors '";
    *error += maxine::vfx::NVVFX_INPUT_MATTE;
    *error += "', 'matte', 'srcMask', 'mask': ";
    *error += StatusToString(vfx_, nvcv_, s);
  }
  return false;
}

void VfxBackgroundBlurEffect::Destroy() {
  if (nvcv_ && nvcv_->IsInitialized() && output_allocated_) {
    if (nvcv_->f().NvCVImage_Dealloc) {
      (void)nvcv_->f().NvCVImage_Dealloc(&output_gpu_);
    }
  }
  output_gpu_ = maxine::NvCVImage{};
  output_allocated_ = false;
  output_ready_ = false;

  if (vfx_ && vfx_->IsInitialized() && handle_) {
    vfx_->f().NvVFX_DestroyEffect(handle_);
  }
  handle_ = nullptr;

  if (vfx_ && vfx_->IsInitialized() && stream_ && own_stream_) {
    vfx_->f().NvVFX_CudaStreamDestroy(stream_);
  }
  stream_ = nullptr;
  own_stream_ = false;
  InvalidateBindings();

  cfg_dirty_ = true;
}

} // namespace studiocast::maxine::effects
