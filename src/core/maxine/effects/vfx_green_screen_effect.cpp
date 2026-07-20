#include "core/maxine/effects/vfx_green_screen_effect.h"

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

} // namespace

VfxGreenScreenEffect::VfxGreenScreenEffect(maxine::vfx::VfxApi *vfx,
                                           maxine::NvcvApi *nvcv,
                                           std::filesystem::path model_dir)
    : vfx_(vfx), nvcv_(nvcv), model_dir_(std::move(model_dir)) {
  // Clear POD image.
  matte_gpu_ = maxine::NvCVImage{};
}

VfxGreenScreenEffect::~VfxGreenScreenEffect() { Destroy(); }

void VfxGreenScreenEffect::SetConfig(const Config &cfg) {
  Config next = cfg;
  if (next.state_count == 0)
    next.state_count = 1;
  if (next.mode == cfg_.mode && next.temporal == cfg_.temporal &&
      next.state_count == cfg_.state_count)
    return;
  cfg_ = next;
  cfg_dirty_ = true;
}

bool VfxGreenScreenEffect::SetExternalCudaStream(maxine::CUstream stream,
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

void VfxGreenScreenEffect::InvalidateBindings() noexcept {
  stream_bound_ = false;
  bound_input_ = nullptr;
  bound_output_ = nullptr;
  bound_width_ = 0;
  bound_height_ = 0;
  matte_ready_ = false;
}

bool VfxGreenScreenEffect::Initialize(std::string *error) {
  return EnsureEffectCreated(error);
}

bool VfxGreenScreenEffect::Configure(
    const studiocast::video::effects::BroadcastCameraEffects &settings,
    std::string *error) {
  // Map canonical settings onto our config.
  Config next = cfg_;
  next.mode = settings.virtual_background.greenscreen_mode;
  next.temporal = settings.virtual_background.greenscreen_temporal;

  // Keep state_count stable for now; it can be wired later via effect params.
  if (next.temporal && next.state_count == 0)
    next.state_count = 1;

  if (next.mode != cfg_.mode || next.temporal != cfg_.temporal ||
      next.state_count != cfg_.state_count) {
    cfg_ = next;
    cfg_dirty_ = true;
  }

  // If initialized already, apply immediately.
  if (handle_) {
    return ApplyConfigLocked(error);
  }
  return true;
}

NvCV_Status VfxGreenScreenEffect::Process(studiocast::video::GpuFrame &frame,
                                          std::string *error) {
  matte_ready_ = false;

  if (!frame.ValidDimensions()) {
    if (error)
      *error = "Invalid frame dimensions.";
    return -1;
  }
  if (!frame.nvcv_gpu) {
    if (error)
      *error = "Green Screen requires frame.nvcv_gpu (NvCVImage on GPU).";
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

  std::string matte_err;
  if (!EnsureMatteImage(static_cast<unsigned>(frame.width),
                        static_cast<unsigned>(frame.height), &matte_err)) {
    if (error)
      *error = matte_err;
    return -1;
  }

  // Bind I/O images.
  auto &f = vfx_->f();
  const auto width = static_cast<unsigned>(frame.width);
  const auto height = static_cast<unsigned>(frame.height);
  if (bound_width_ != width || bound_height_ != height) {
    bound_input_ = nullptr;
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
  if (bound_output_ != &matte_gpu_) {
    s = f.NvVFX_SetImage(handle_, maxine::vfx::NVVFX_OUTPUT_IMAGE,
                         &matte_gpu_);
    if (s != maxine::NVCV_SUCCESS) {
      if (error)
        *error = "NvVFX_SetImage(dstImage) failed: " +
                 StatusToString(vfx_, nvcv_, s);
      bound_output_ = nullptr;
      return s;
    }
    bound_output_ = &matte_gpu_;
  }

  // Run synchronously for simplicity.
  s = f.NvVFX_Run(handle_, /*async=*/0);
  if (s != maxine::NVCV_SUCCESS) {
    InvalidateBindings();
    if (error)
      *error = "NvVFX_Run failed: " + StatusToString(vfx_, nvcv_, s);
    return s;
  }

  matte_ready_ = true;
  return s;
}

bool VfxGreenScreenEffect::EnsureEffectCreated(std::string *error) {
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

  NvCV_Status s =
      f.NvVFX_CreateEffect(maxine::vfx::NVVFX_FX_GREEN_SCREEN, &handle_);
  if (s != maxine::NVCV_SUCCESS || !handle_) {
    if (error)
      *error = "NvVFX_CreateEffect(Green Screen) failed: " +
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

  // Apply initial config (mode/temporal/state binding) before Load().
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

bool VfxGreenScreenEffect::ApplyConfigLocked(std::string *error) {
  if (!cfg_dirty_)
    return true;
  if (!handle_) {
    if (error)
      *error = "Green Screen effect not created.";
    return false;
  }

  if (cfg_.state_count == 0)
    cfg_.state_count = 1;

  auto &f = vfx_->f();

  // Mode.
  NvCV_Status s = f.NvVFX_SetU32(handle_, maxine::vfx::NVVFX_MODE, cfg_.mode);
  if (s != maxine::NVCV_SUCCESS) {
    if (error)
      *error = "NvVFX_SetU32(mode) failed: " + StatusToString(vfx_, nvcv_, s);
    return false;
  }

  // Temporal flag.
  s = f.NvVFX_SetU32(handle_, maxine::vfx::NVVFX_TEMPORAL,
                     cfg_.temporal ? 1u : 0u);
  if (s != maxine::NVCV_SUCCESS) {
    if (error)
      *error =
          "NvVFX_SetU32(temporal) failed: " + StatusToString(vfx_, nvcv_, s);
    return false;
  }

  if (!EnsureTemporalStateLocked(error)) {
    return false;
  }

  cfg_dirty_ = false;
  return true;
}

bool VfxGreenScreenEffect::EnsureStreamBound(std::string *error) {
  if (stream_bound_)
    return true;
  if (!handle_ || !stream_) {
    if (error)
      *error = "Green Screen stream is unavailable.";
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

bool VfxGreenScreenEffect::EnsureTemporalStateLocked(std::string *error) {
  if (!handle_) {
    if (error)
      *error = "Green Screen effect not created.";
    return false;
  }

  auto &f = vfx_->f();

  if (!cfg_.temporal) {
    // Disable and free any previous state handles.
    for (auto &h : states_) {
      if (!h)
        continue;
      (void)f.NvVFX_DeallocateState(handle_, h);
      h = nullptr;
    }
    states_.clear();
    states_bound_ = false;
    return true;
  }

  // Allocate/bind state handle array.
  if (states_.size() != cfg_.state_count) {
    for (auto &h : states_) {
      if (!h)
        continue;
      (void)f.NvVFX_DeallocateState(handle_, h);
      h = nullptr;
    }
    states_.assign(cfg_.state_count, nullptr);
    states_bound_ = false;
  }

  for (auto &h : states_) {
    if (h)
      continue;
    NvCV_Status s = f.NvVFX_AllocateState(handle_, &h);
    if (s != maxine::NVCV_SUCCESS || !h) {
      if (error)
        *error =
            "NvVFX_AllocateState failed: " + StatusToString(vfx_, nvcv_, s);
      return false;
    }
  }

  if (!states_bound_) {
    NvCV_Status s = f.NvVFX_SetStateObjectHandleArray(
        handle_, maxine::vfx::NVVFX_STATE, states_.data(),
        static_cast<std::uint32_t>(states_.size()));
    if (s != maxine::NVCV_SUCCESS) {
      if (error)
        *error = "NvVFX_SetStateObjectHandleArray(state) failed: " +
                 StatusToString(vfx_, nvcv_, s);
      return false;
    }
    states_bound_ = true;
  }

  return true;
}

bool VfxGreenScreenEffect::EnsureMatteImage(unsigned width, unsigned height,
                                            std::string *error) {
  if (!nvcv_ || !nvcv_->IsInitialized()) {
    if (error)
      *error = "NvCVImage runtime not initialized.";
    return false;
  }

  if (matte_allocated_ && matte_gpu_.width == width &&
      matte_gpu_.height == height && matte_gpu_.gpuMem == maxine::NVCV_GPU) {
    return true;
  }

  auto &nf = nvcv_->f();
  if (!nf.NvCVImage_Alloc || !nf.NvCVImage_Dealloc) {
    if (error)
      *error = "NvCVImage_Alloc/Dealloc unavailable.";
    return false;
  }

  if (matte_allocated_) {
    (void)nf.NvCVImage_Dealloc(&matte_gpu_);
    matte_gpu_ = maxine::NvCVImage{};
    matte_allocated_ = false;
  }

  const maxine::NvCV_Status s =
      nf.NvCVImage_Alloc(&matte_gpu_, width, height, maxine::NVCV_A,
                         maxine::NVCV_U8, maxine::NVCV_CHUNKY, maxine::NVCV_GPU,
                         /*alignment=*/0);
  if (s != maxine::NVCV_SUCCESS) {
    if (error)
      *error = "NvCVImage_Alloc(matte Au8 GPU) failed: " +
               StatusToString(vfx_, nvcv_, s);
    return false;
  }

  matte_allocated_ = true;
  bound_input_ = nullptr;
  bound_output_ = nullptr;
  bound_width_ = width;
  bound_height_ = height;
  return true;
}

void VfxGreenScreenEffect::Destroy() {
  // Free NvCVImages we own.
  if (nvcv_ && nvcv_->IsInitialized() && matte_allocated_) {
    if (nvcv_->f().NvCVImage_Dealloc) {
      (void)nvcv_->f().NvCVImage_Dealloc(&matte_gpu_);
    }
  }
  matte_gpu_ = maxine::NvCVImage{};
  matte_allocated_ = false;
  matte_ready_ = false;

  // Free VFX state handles.
  if (vfx_ && vfx_->IsInitialized() && handle_) {
    auto &f = vfx_->f();
    for (auto &h : states_) {
      if (!h)
        continue;
      (void)f.NvVFX_DeallocateState(handle_, h);
      h = nullptr;
    }
  }
  states_.clear();
  states_bound_ = false;

  // Destroy effect.
  if (vfx_ && vfx_->IsInitialized() && handle_) {
    vfx_->f().NvVFX_DestroyEffect(handle_);
  }
  handle_ = nullptr;

  // Destroy stream.
  if (vfx_ && vfx_->IsInitialized() && stream_ && own_stream_) {
    vfx_->f().NvVFX_CudaStreamDestroy(stream_);
  }
  stream_ = nullptr;
  own_stream_ = false;
  InvalidateBindings();

  cfg_dirty_ = true;
}

} // namespace studiocast::maxine::effects
