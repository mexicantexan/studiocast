#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/maxine/ar_api.h"
#include "core/maxine/nvcv_types.h"

namespace studiocast::maxine::effects {

using NvAR_FeatureHandle = studiocast::maxine::ar::NvAR_FeatureHandle;

struct RectF {
  float x = 0.0f;
  float y = 0.0f;
  float w = 0.0f;
  float h = 0.0f;
};

struct AutoFrameKnobs {
  int strength = 50;      // 0..100
  int smoothing = 70;     // 0..100
  float headroom = 0.15f; // 0..1
};

// AR-backed Auto Frame tracker.
//
// Responsibilities:
//  - Run Maxine AR face box detection (optionally body fallback)
//  - Convert detections to a target crop rectangle (with configurable headroom)
//  - Smooth the crop rectangle over time (EMA) to avoid jitter
//  - Provide a reasonable center-crop fallback when no detections are available
class ArAutoFrameTracker {
public:
  explicit ArAutoFrameTracker(studiocast::maxine::ar::ArApi *ar);
  ~ArAutoFrameTracker();

  ArAutoFrameTracker(const ArAutoFrameTracker &) = delete;
  ArAutoFrameTracker &operator=(const ArAutoFrameTracker &) = delete;

  // `input_bgr_gpu` must remain valid for the lifetime of the initialized
  // tracker.
  bool EnsureInitialized(NvCVImage *input_bgr_gpu, std::string *error_out);
  bool SetExternalCudaStream(studiocast::maxine::CUstream stream,
                             std::string *error_out);
  void InvalidateBindings() noexcept;
  void Reset();

  void SetKnobs(const AutoFrameKnobs &knobs) { knobs_ = knobs; }
  void SetOutputAspect(float aspect) { output_aspect_ = aspect; }

  // Run detection + update the smoothed crop rectangle.
  // Returns false only on hard AR API failures.
  bool Update(int frame_w, int frame_h, std::string *error_out);

  RectF SmoothedCropPx() const { return crop_smoothed_px_; }
  bool last_had_detection() const { return last_had_detection_; }

  // Math helpers (deterministic; used by self-test).
  static float SmoothingAlpha(int smoothing_percent);
  static RectF CenterCrop(int frame_w, int frame_h, float output_aspect,
                          float zoom);
  static RectF ClampToFrame(const RectF &r, int frame_w, int frame_h);
  static RectF ComputeTargetCropFromBoxPx(const RectF &box_px, int frame_w,
                                          int frame_h, float output_aspect,
                                          const AutoFrameKnobs &knobs);
  static RectF Lerp(const RectF &a, const RectF &b, float alpha);

private:
  bool EnsureFeatureInitialized(const char *feature_id,
                                NvAR_FeatureHandle *out_handle,
                                std::string *error_out);
  bool ConfigureBoxOutputs(NvAR_FeatureHandle handle, std::string *error_out);
  bool RunAndExtractBestBox(NvAR_FeatureHandle handle, int frame_w, int frame_h,
                            RectF *out_best_box_px, bool *out_found,
                            std::string *error_out);
  bool BindStream(NvAR_FeatureHandle handle, bool *bound,
                  std::string *error_out);

  studiocast::maxine::ar::ArApi *ar_ = nullptr; // non-owning
  NvCVImage *input_bgr_gpu_ = nullptr;          // non-owning
  unsigned input_width_ = 0;
  unsigned input_height_ = 0;
  bool face_input_bound_ = false;
  bool body_input_configured_ = false;

  NvAR_FeatureHandle face_handle_ = nullptr;
  NvAR_FeatureHandle body_handle_ = nullptr; // optional
  studiocast::maxine::CUstream external_stream_ = nullptr;
  bool external_stream_selected_ = false;
  bool face_stream_bound_ = false;
  bool body_stream_bound_ = false;
  bool face_outputs_configured_ = false;
  bool body_outputs_configured_ = false;

  // Output binding (best-effort). Some AR builds may require SetF32Array.
  std::string boxes_param_name_;
  std::string num_param_selector_;
  std::vector<float> boxes_scratch_;

  AutoFrameKnobs knobs_{};
  float output_aspect_ = 16.0f / 9.0f;

  RectF crop_smoothed_px_{};
  bool have_smoothed_ = false;
  bool last_had_detection_ = false;
};

} // namespace studiocast::maxine::effects
