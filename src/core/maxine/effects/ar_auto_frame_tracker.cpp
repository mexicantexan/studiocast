#include "core/maxine/effects/ar_auto_frame_tracker.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace studiocast::maxine::effects {
namespace {

RectF MakeRect(float x, float y, float w, float h) {
  RectF r;
  r.x = x;
  r.y = y;
  r.w = w;
  r.h = h;
  return r;
}

bool LooksNormalized01(const RectF &r) {
  const float maxv = std::max({r.x, r.y, r.w, r.h});
  return maxv <= 2.0f;
}

} // namespace

ArAutoFrameTracker::ArAutoFrameTracker(studiocast::maxine::ar::ArApi *ar)
    : ar_(ar) {}

ArAutoFrameTracker::~ArAutoFrameTracker() {
  if (ar_ && ar_->IsInitialized() && ar_->f().NvAR_Destroy) {
    if (face_handle_)
      (void)ar_->f().NvAR_Destroy(face_handle_);
    if (body_handle_)
      (void)ar_->f().NvAR_Destroy(body_handle_);
  }
}

bool ArAutoFrameTracker::SetExternalCudaStream(
    studiocast::maxine::CUstream stream, std::string *error_out) {
  if (!stream) {
    if (error_out)
      *error_out = "External CUDA stream must be non-null.";
    return false;
  }
  if (external_stream_ == stream && (!face_handle_ || face_stream_bound_) &&
      (!body_handle_ || body_stream_bound_))
    return true;
  external_stream_ = stream;
  external_stream_selected_ = true;
  face_stream_bound_ = false;
  body_stream_bound_ = false;
  if (!BindStream(face_handle_, &face_stream_bound_, error_out))
    return false;
  if (!BindStream(body_handle_, &body_stream_bound_, error_out))
    return false;
  return true;
}

void ArAutoFrameTracker::InvalidateBindings() noexcept {
  face_input_bound_ = false;
  body_input_configured_ = false;
  face_stream_bound_ = false;
  body_stream_bound_ = false;
}

bool ArAutoFrameTracker::BindStream(NvAR_FeatureHandle handle, bool *bound,
                                    std::string *error_out) {
  if (!handle || !external_stream_selected_)
    return true;
  if (bound && *bound)
    return true;
  if (!ar_ || !ar_->IsInitialized() || !ar_->f().NvAR_SetCudaStream) {
    if (error_out)
      *error_out = "NvAR_SetCudaStream unavailable for Auto Frame.";
    return false;
  }
  const auto status = ar_->f().NvAR_SetCudaStream(
      handle, NvAR_Parameter_Config(CUDAStream), external_stream_);
  if (status != studiocast::maxine::NVCV_SUCCESS) {
    if (error_out)
      *error_out = "NvAR_SetCudaStream failed: " + ar_->StatusToString(status);
    return false;
  }
  if (bound)
    *bound = true;
  return true;
}

void ArAutoFrameTracker::Reset() {
  have_smoothed_ = false;
  crop_smoothed_px_ = {};
  last_had_detection_ = false;
}

bool ArAutoFrameTracker::EnsureFeatureInitialized(
    const char *feature_id, NvAR_FeatureHandle *out_handle,
    std::string *error_out) {
  if (!out_handle)
    return false;
  if (*out_handle)
    return true;
  if (!ar_ || !ar_->IsInitialized() || !ar_->f().NvAR_Create) {
    if (error_out)
      *error_out = "NvAR API not initialized.";
    return false;
  }

  NvAR_FeatureHandle h = nullptr;
  auto st = ar_->f().NvAR_Create(feature_id, &h);
  if (st != studiocast::maxine::NVCV_SUCCESS || !h) {
    if (error_out)
      *error_out = "NvAR_Create(" + std::string(feature_id) +
                   ") failed: " + ar_->StatusToString(st);
    return false;
  }

  if (!ar_->f().NvAR_Load) {
    if (error_out)
      *error_out = "NvAR_Load symbol unavailable.";
    return false;
  }
  st = ar_->f().NvAR_Load(h);
  if (st != studiocast::maxine::NVCV_SUCCESS) {
    if (error_out)
      *error_out = "NvAR_Load(" + std::string(feature_id) +
                   ") failed: " + ar_->StatusToString(st);
    (void)ar_->f().NvAR_Destroy(h);
    return false;
  }

  *out_handle = h;
  return true;
}

bool ArAutoFrameTracker::ConfigureBoxOutputs(NvAR_FeatureHandle handle,
                                             std::string *error_out) {
  if (!handle)
    return false;
  if (!ar_ || !ar_->IsInitialized())
    return false;
  if (!ar_->f().NvAR_SetF32Array) {
    // Some builds may still allow GetF32Array; we treat this as non-fatal.
    return true;
  }

  // Allocate scratch for up to 16 boxes (x,y,w,h).
  boxes_scratch_.assign(16 * 4, 0.0f);

  // Try a conservative set of likely output names.
  // NOTE: Maxine AR parameter naming is proprietary; we must not vendor
  // headers. These are best-effort guesses aligned with the SDK's macro naming
  // scheme.
  const char *candidates[] = {
      NvAR_Parameter_Output(BoundingBoxes),
      NvAR_Parameter_Output(BBoxes),
      NvAR_Parameter_Output(FaceBoxes),
      NvAR_Parameter_Output(FaceBoundingBoxes),
  };

  for (const char *name : candidates) {
    const auto st =
        ar_->f().NvAR_SetF32Array(handle, name, boxes_scratch_.data(),
                                  static_cast<int>(boxes_scratch_.size()));
    if (st == studiocast::maxine::NVCV_SUCCESS) {
      boxes_param_name_ = name;
      return true;
    }
  }

  // Not fatal: we can still try GetF32Array after Run.
  if (error_out) {
    *error_out =
        "Unable to bind AR bounding box output array (unknown selector).";
  }
  return true;
}

bool ArAutoFrameTracker::EnsureInitialized(NvCVImage *input_bgr_gpu,
                                           std::string *error_out) {
  if (!input_bgr_gpu) {
    if (error_out)
      *error_out = "AutoFrame tracker input image is null.";
    return false;
  }

  std::string err;
  if (!EnsureFeatureInitialized(
          studiocast::maxine::ar::NVAR_FEATURE_FACE_BOX_DETECTION,
          &face_handle_, &err)) {
    if (error_out)
      *error_out = err;
    return false;
  }

  // Optional body fallback.
  (void)EnsureFeatureInitialized(
      studiocast::maxine::ar::NVAR_FEATURE_BODY_BOX_DETECTION, &body_handle_,
      nullptr);

  if (!BindStream(face_handle_, &face_stream_bound_, error_out) ||
      !BindStream(body_handle_, &body_stream_bound_, error_out))
    return false;

  const bool input_changed = input_bgr_gpu_ != input_bgr_gpu ||
                             input_width_ != input_bgr_gpu->width ||
                             input_height_ != input_bgr_gpu->height;
  if (input_changed) {
    face_input_bound_ = false;
    body_input_configured_ = false;
  }
  input_bgr_gpu_ = input_bgr_gpu;
  input_width_ = input_bgr_gpu->width;
  input_height_ = input_bgr_gpu->height;

  // Bind input image.
  if (!face_input_bound_ && ar_->f().NvAR_SetObject) {
    const auto st =
        ar_->f().NvAR_SetObject(face_handle_, NvAR_Parameter_Input(Image),
                                input_bgr_gpu_, sizeof(*input_bgr_gpu_));
    if (st != studiocast::maxine::NVCV_SUCCESS) {
      if (error_out)
        *error_out =
            "NvAR_SetObject(Input.Image) failed: " + ar_->StatusToString(st);
      return false;
    }
    face_input_bound_ = true;
  }
  if (body_handle_ && !body_input_configured_ && ar_->f().NvAR_SetObject) {
    // Body detection is optional. Cache an unsupported input selector until
    // the input identity changes or a runtime failure invalidates bindings.
    // This preserves best-effort fallback without retrying a stable failure
    // on every frame.
    (void)ar_->f().NvAR_SetObject(body_handle_, NvAR_Parameter_Input(Image),
                                  input_bgr_gpu_, sizeof(*input_bgr_gpu_));
    body_input_configured_ = true;
  }

  // Best-effort output binding.
  if (!face_outputs_configured_) {
    (void)ConfigureBoxOutputs(face_handle_, nullptr);

    // Best-effort: discover a count selector.
    if (ar_->f().NvAR_GetU32) {
      const char *countCandidates[] = {
          NvAR_Parameter_Output(NumBoxes),
          NvAR_Parameter_Output(NumFaces),
          NvAR_Parameter_Output(FaceCount),
      };
      uint32_t tmp = 0;
      for (const char *sel : countCandidates) {
        const auto st = ar_->f().NvAR_GetU32(face_handle_, sel, &tmp);
        if (st == studiocast::maxine::NVCV_SUCCESS) {
          num_param_selector_ = sel;
          break;
        }
      }
    }
    face_outputs_configured_ = true;
  }
  if (body_handle_ && !body_outputs_configured_) {
    (void)ConfigureBoxOutputs(body_handle_, nullptr);
    body_outputs_configured_ = true;
  }

  return true;
}

float ArAutoFrameTracker::SmoothingAlpha(int smoothing_percent) {
  const float s =
      static_cast<float>(std::clamp(smoothing_percent, 0, 100)) / 100.0f;
  // Map [0..1] to [1.0 .. ~0.1].
  return 1.0f / (1.0f + 9.0f * s);
}

RectF ArAutoFrameTracker::Lerp(const RectF &a, const RectF &b, float alpha) {
  return MakeRect(a.x + (b.x - a.x) * alpha, a.y + (b.y - a.y) * alpha,
                  a.w + (b.w - a.w) * alpha, a.h + (b.h - a.h) * alpha);
}

RectF ArAutoFrameTracker::ClampToFrame(const RectF &r, int frame_w,
                                       int frame_h) {
  RectF out = r;
  out.w = std::max(1.0f, std::min(out.w, static_cast<float>(frame_w)));
  out.h = std::max(1.0f, std::min(out.h, static_cast<float>(frame_h)));
  out.x = std::max(0.0f, std::min(out.x, static_cast<float>(frame_w) - out.w));
  out.y = std::max(0.0f, std::min(out.y, static_cast<float>(frame_h) - out.h));
  return out;
}

RectF ArAutoFrameTracker::CenterCrop(int frame_w, int frame_h,
                                     float output_aspect, float zoom) {
  zoom = std::max(1.0f, zoom);
  const float fw = static_cast<float>(frame_w);
  const float fh = static_cast<float>(frame_h);

  float cropW = fw;
  float cropH = fw / output_aspect;
  if (cropH > fh) {
    cropH = fh;
    cropW = fh * output_aspect;
  }

  cropW = std::min(fw, cropW / zoom);
  cropH = std::min(fh, cropH / zoom);

  const float x = (fw - cropW) * 0.5f;
  const float y = (fh - cropH) * 0.5f;
  return ClampToFrame(MakeRect(x, y, cropW, cropH), frame_w, frame_h);
}

RectF ArAutoFrameTracker::ComputeTargetCropFromBoxPx(
    const RectF &box_px, int frame_w, int frame_h, float output_aspect,
    const AutoFrameKnobs &knobs) {
  const float strength =
      static_cast<float>(std::clamp(knobs.strength, 0, 100)) / 100.0f;
  const float headroom = std::max(0.0f, std::min(1.0f, knobs.headroom));

  const float fw = static_cast<float>(frame_w);
  const float fh = static_cast<float>(frame_h);

  // Tightness controls how much padding we keep around the detected subject.
  const float pad =
      2.4f - strength * 1.2f; // strength 0 -> 2.4x, strength 1 -> 1.2x

  const float cx = box_px.x + box_px.w * 0.5f;
  const float cy = box_px.y + box_px.h * 0.5f - headroom * box_px.h;

  float cropH = box_px.h * pad;
  float cropW = cropH * output_aspect;
  const float minW = box_px.w * pad;
  if (cropW < minW) {
    cropW = minW;
    cropH = cropW / output_aspect;
  }

  // Avoid extreme zoom.
  cropW = std::max(cropW, fw * 0.35f);
  cropH = std::max(cropH, fh * 0.35f);
  cropW = std::min(cropW, fw);
  cropH = std::min(cropH, fh);

  const float x = cx - cropW * 0.5f;
  const float y = cy - cropH * 0.5f;
  return ClampToFrame(MakeRect(x, y, cropW, cropH), frame_w, frame_h);
}

bool ArAutoFrameTracker::RunAndExtractBestBox(NvAR_FeatureHandle handle,
                                              int frame_w, int frame_h,
                                              RectF *out_best_box_px,
                                              bool *out_found,
                                              std::string *error_out) {
  if (out_found)
    *out_found = false;
  if (!handle)
    return true;
  if (!ar_ || !ar_->IsInitialized() || !ar_->f().NvAR_Run) {
    if (error_out)
      *error_out = "NvAR_Run symbol unavailable.";
    return false;
  }

  ++execution_telemetry_.synchronous_run_attempts;
  auto st = ar_->f().NvAR_Run(handle);
  if (st != studiocast::maxine::NVCV_SUCCESS) {
    InvalidateBindings();
    if (error_out)
      *error_out = "NvAR_Run failed: " + ar_->StatusToString(st);
    return false;
  }
  ++execution_telemetry_.synchronous_run_successes;

  uint32_t num = 0;
  if (!num_param_selector_.empty() && ar_->f().NvAR_GetU32) {
    (void)ar_->f().NvAR_GetU32(handle, num_param_selector_.c_str(), &num);
  }

  const float *vals = nullptr;
  int count = 0;
  if (!boxes_param_name_.empty() && ar_->f().NvAR_GetF32Array) {
    st = ar_->f().NvAR_GetF32Array(handle, boxes_param_name_.c_str(), &vals,
                                   &count);
    if (st != studiocast::maxine::NVCV_SUCCESS) {
      vals = nullptr;
      count = 0;
    }
  }

  // If GetF32Array isn't available/doesn't work, fall back to our scratch.
  if ((!vals || count <= 0) && !boxes_scratch_.empty()) {
    vals = boxes_scratch_.data();
    count = static_cast<int>(boxes_scratch_.size());
  }

  if (!vals || count < 4) {
    return true;
  }

  int n = count / 4;
  if (num > 0)
    n = std::min(n, static_cast<int>(num));

  RectF best{};
  float bestArea = 0.0f;
  for (int i = 0; i < n; ++i) {
    const float x = vals[i * 4 + 0];
    const float y = vals[i * 4 + 1];
    const float w = vals[i * 4 + 2];
    const float h = vals[i * 4 + 3];
    if (!(w > 1e-3f && h > 1e-3f))
      continue;

    RectF r = MakeRect(x, y, w, h);
    if (LooksNormalized01(r)) {
      r.x *= static_cast<float>(frame_w);
      r.y *= static_cast<float>(frame_h);
      r.w *= static_cast<float>(frame_w);
      r.h *= static_cast<float>(frame_h);
    }

    const float area = r.w * r.h;
    if (area > bestArea) {
      bestArea = area;
      best = r;
    }
  }

  if (bestArea > 0.0f) {
    if (out_best_box_px)
      *out_best_box_px = best;
    if (out_found)
      *out_found = true;
  }
  return true;
}

bool ArAutoFrameTracker::Update(int frame_w, int frame_h,
                                std::string *error_out) {
  if (!input_bgr_gpu_) {
    if (error_out)
      *error_out = "AutoFrame tracker not initialized.";
    return false;
  }

  RectF best{};
  bool found = false;
  if (!RunAndExtractBestBox(face_handle_, frame_w, frame_h, &best, &found,
                            error_out)) {
    return false;
  }
  if (!found && body_handle_) {
    if (!RunAndExtractBestBox(body_handle_, frame_w, frame_h, &best, &found,
                              error_out)) {
      return false;
    }
  }

  last_had_detection_ = found;
  RectF target{};
  if (found) {
    target = ComputeTargetCropFromBoxPx(best, frame_w, frame_h, output_aspect_,
                                        knobs_);
  } else {
    const float strength =
        static_cast<float>(std::clamp(knobs_.strength, 0, 100)) / 100.0f;
    const float zoom = 1.0f + strength * 0.5f;
    target = CenterCrop(frame_w, frame_h, output_aspect_, zoom);
  }

  if (!have_smoothed_) {
    crop_smoothed_px_ = target;
    have_smoothed_ = true;
  } else {
    const float a = SmoothingAlpha(knobs_.smoothing);
    crop_smoothed_px_ =
        ClampToFrame(Lerp(crop_smoothed_px_, target, a), frame_w, frame_h);
  }

  return true;
}

} // namespace studiocast::maxine::effects
