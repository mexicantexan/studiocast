#include "core/maxine/effects/ar_eye_contact_effect.h"

#include <algorithm>
#include <sstream>

namespace studiocast::maxine::effects {
namespace {

std::string MakeStatusError(const studiocast::maxine::ar::ArApi &ar,
                            const char *what,
                            studiocast::maxine::NvCV_Status st) {
  std::ostringstream oss;
  oss << what << " failed: " << static_cast<int>(st) << " ("
      << ar.StatusToString(st) << ")";
  return oss.str();
}

} // namespace

ArEyeContactEffect::ArEyeContactEffect(studiocast::maxine::ar::ArApi *ar)
    : ar_(ar) {}

ArEyeContactEffect::~ArEyeContactEffect() {
  if (handle_ && ar_ && ar_->IsInitialized() && ar_->f().NvAR_Destroy) {
    (void)ar_->f().NvAR_Destroy(handle_);
  }
  handle_ = nullptr;

  if (stream_owned_ && stream_ && ar_ && ar_->IsInitialized() &&
      ar_->f().NvAR_CudaStreamDestroy) {
    (void)ar_->f().NvAR_CudaStreamDestroy(stream_);
  }
  stream_ = nullptr;
  stream_owned_ = false;
}

bool ArEyeContactEffect::SetExternalCudaStream(maxine::CUstream stream,
                                               std::string *error) {
  if (!stream) {
    if (error)
      *error = "External CUDA stream must be non-null.";
    return false;
  }
  if (stream_ == stream && !stream_owned_ && (!handle_ || stream_bound_))
    return true;
  if (handle_) {
    if (!ar_ || !ar_->IsInitialized() || !ar_->f().NvAR_SetCudaStream) {
      if (error)
        *error = "NvAR_SetCudaStream unavailable for external stream.";
      return false;
    }
    const auto status = ar_->f().NvAR_SetCudaStream(
        handle_, NvAR_Parameter_Config(CUDAStream), stream);
    if (status != studiocast::maxine::NVCV_SUCCESS) {
      if (error)
        *error = MakeStatusError(*ar_, "NvAR_SetCudaStream", status);
      return false;
    }
  }
  if (stream_owned_ && stream_ && ar_ && ar_->IsInitialized() &&
      ar_->f().NvAR_CudaStreamDestroy)
    (void)ar_->f().NvAR_CudaStreamDestroy(stream_);
  stream_ = stream;
  stream_owned_ = false;
  external_stream_selected_ = true;
  stream_bound_ = handle_ != nullptr;
  return true;
}

void ArEyeContactEffect::InvalidateBindings() noexcept {
  stream_bound_ = false;
  bound_input_ = nullptr;
  bound_output_ = nullptr;
  bound_width_ = 0;
  bound_height_ = 0;
}

bool ArEyeContactEffect::Configure(
    const studiocast::video::effects::BroadcastCameraEffects &settings,
    std::string * /*error*/) {
  // Configuration is applied lazily (after feature creation) since selector
  // support may vary by SDK version.
  const bool look_away = settings.eye_contact.look_away_enabled;
  const int strength = std::clamp(settings.eye_contact.strength, 0, 100);
  if (look_away != look_away_enabled_ || strength != strength_) {
    look_away_enabled_ = look_away;
    strength_ = strength;
    config_dirty_ = true;
  }
  return true;
}

bool ArEyeContactEffect::EnsureCreated(std::string *error) {
  if (handle_)
    return true;
  if (!ar_ || !ar_->IsInitialized() || !ar_->f().NvAR_Create) {
    if (error)
      *error = "NvAR API not initialized.";
    return false;
  }
  const auto st = ar_->f().NvAR_Create(
      studiocast::maxine::ar::NVAR_FEATURE_GAZE_REDIRECTION, &handle_);
  if (st != studiocast::maxine::NVCV_SUCCESS) {
    if (error)
      *error = MakeStatusError(*ar_, "NvAR_Create(GazeRedirection)", st);
    return false;
  }
  loaded_ = false;
  config_dirty_ = true;
  InvalidateBindings();
  return true;
}

bool ArEyeContactEffect::EnsureLoaded(std::string *error) {
  if (loaded_)
    return true;
  if (!EnsureCreated(error))
    return false;
  if (!ar_->f().NvAR_Load) {
    if (error)
      *error = "NvAR_Load symbol unavailable.";
    return false;
  }
  const auto st = ar_->f().NvAR_Load(handle_);
  if (st != studiocast::maxine::NVCV_SUCCESS) {
    if (error)
      *error = MakeStatusError(*ar_, "NvAR_Load", st);
    return false;
  }
  loaded_ = true;
  return true;
}

bool ArEyeContactEffect::EnsureStreamBound(studiocast::video::GpuFrame &frame,
                                           std::string *error) {
  // If a stream is provided by the pipeline, prefer it.
  if (frame.cuda_stream && !SetExternalCudaStream(frame.cuda_stream, error))
    return false;

  // Otherwise, create a stream if possible.
  if (!stream_ && ar_ && ar_->IsInitialized() &&
      ar_->f().NvAR_CudaStreamCreate) {
    const auto st = ar_->f().NvAR_CudaStreamCreate(&stream_);
    if (st != studiocast::maxine::NVCV_SUCCESS) {
      if (error)
        *error = MakeStatusError(*ar_, "NvAR_CudaStreamCreate", st);
      return false;
    }
    stream_owned_ = true;
    external_stream_selected_ = false;
  }

  if (!stream_)
    return true; // fall back to default stream

  if (stream_bound_)
    return true;

  // Bind to feature if API is present.
  if (ar_->f().NvAR_SetCudaStream) {
    const auto st = ar_->f().NvAR_SetCudaStream(
        handle_, NvAR_Parameter_Config(CUDAStream), stream_);
    if (st != studiocast::maxine::NVCV_SUCCESS) {
      if (!external_stream_selected_) {
        // Older SDKs may reject this selector while continuing on their
        // default stream. Preserve that standalone compatibility behavior.
        stream_bound_ = true;
        return true;
      }
      if (error)
        *error = MakeStatusError(*ar_, "NvAR_SetCudaStream", st);
      return false;
    }
    stream_bound_ = true;
  } else if (stream_ && external_stream_selected_) {
    if (error)
      *error = "NvAR_SetCudaStream unavailable.";
    return false;
  } else {
    stream_bound_ = true;
  }
  return true;
}

bool ArEyeContactEffect::ApplyConfig(std::string *error) {
  if (!config_dirty_)
    return true;
  if (!SetU32Required(NvAR_Parameter_Config(GazeRedirect), 1u, error))
    return false;
  SetU32Optional(NvAR_Parameter_Config(EnableLookAway),
                 look_away_enabled_ ? 1u : 0u);
  const float strength = static_cast<float>(strength_) / 100.0f;
  SetF32Optional(NvAR_Parameter_Config(Strength), strength);
  SetF32Optional(NvAR_Parameter_Config(GazeRedirectStrength), strength);
  SetF32Optional(NvAR_Parameter_Config(RedirectionStrength), strength);
  config_dirty_ = false;
  return true;
}

bool ArEyeContactEffect::SetU32Required(
    studiocast::maxine::ar::NvAR_ParameterSelector sel, std::uint32_t val,
    std::string *error) {
  const auto st = ar_->f().NvAR_SetU32(handle_, sel, val);
  if (st != studiocast::maxine::NVCV_SUCCESS) {
    if (error)
      *error = MakeStatusError(*ar_, "NvAR_SetU32", st) + " for selector '" +
               sel + "'";
    return false;
  }
  return true;
}

void ArEyeContactEffect::SetU32Optional(
    studiocast::maxine::ar::NvAR_ParameterSelector sel, std::uint32_t val) {
  if (!ar_->f().NvAR_SetU32)
    return;
  (void)ar_->f().NvAR_SetU32(handle_, sel, val);
}

void ArEyeContactEffect::SetF32Optional(
    studiocast::maxine::ar::NvAR_ParameterSelector sel, float val) {
  if (!ar_->f().NvAR_SetF32)
    return;
  (void)ar_->f().NvAR_SetF32(handle_, sel, val);
}

NvCV_Status ArEyeContactEffect::Process(studiocast::video::GpuFrame &frame,
                                        std::string *error) {
  if (!error) {
    static std::string ignored;
    error = &ignored;
  }

  if (!ar_ || !ar_->IsInitialized()) {
    *error = "NvAR not initialized.";
    return -1;
  }
  if (!frame.nvcv_gpu) {
    *error = "ArEyeContactEffect requires frame.nvcv_gpu.";
    return -2;
  }
  if (!ar_->f().NvAR_Run || !ar_->f().NvAR_SetObject || !ar_->f().NvAR_SetU32) {
    *error = "NvAR required symbols missing (Run/SetObject/SetU32).";
    return -1;
  }

  if (!EnsureLoaded(error))
    return -1;
  if (!EnsureStreamBound(frame, error))
    return -1;
  if (!ApplyConfig(error))
    return -2;

  const auto width = static_cast<unsigned>(frame.width);
  const auto height = static_cast<unsigned>(frame.height);
  if (bound_width_ != width || bound_height_ != height) {
    bound_input_ = nullptr;
    bound_output_ = nullptr;
    bound_width_ = width;
    bound_height_ = height;
  }
  studiocast::maxine::NvCVImage *out =
      frame.nvcv_tmp ? frame.nvcv_tmp : frame.nvcv_gpu;
  auto st = studiocast::maxine::NVCV_SUCCESS;
  if (bound_input_ != frame.nvcv_gpu) {
    st = ar_->f().NvAR_SetObject(handle_, NvAR_Parameter_Input(Image),
                                 frame.nvcv_gpu, sizeof(*frame.nvcv_gpu));
    if (st != studiocast::maxine::NVCV_SUCCESS) {
      *error = MakeStatusError(*ar_, "NvAR_SetObject(Input Image)", st);
      bound_input_ = nullptr;
      return st;
    }
    bound_input_ = frame.nvcv_gpu;
  }
  if (bound_output_ != out) {
    st = ar_->f().NvAR_SetObject(handle_, NvAR_Parameter_Output(Image), out,
                                 sizeof(*out));
    if (st != studiocast::maxine::NVCV_SUCCESS) {
      *error = MakeStatusError(*ar_, "NvAR_SetObject(Output Image)", st);
      bound_output_ = nullptr;
      return st;
    }
    bound_output_ = out;
  }

  ++execution_telemetry_.synchronous_run_attempts;
  st = ar_->f().NvAR_Run(handle_);
  if (st != studiocast::maxine::NVCV_SUCCESS) {
    InvalidateBindings();
    *error = MakeStatusError(*ar_, "NvAR_Run", st);
  } else {
    ++execution_telemetry_.synchronous_run_successes;
  }
  return st;
}

} // namespace studiocast::maxine::effects
