#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "core/onnx/ort_session.h"
#include "core/open_video/dlib_face_landmarks.h"
#include "core/open_video/frame_analysis_cache.h"
#include "core/open_video/model_pack_registry.h"
#include "core/open_video/yunet_face_detector.h"
#include "core/video/image_ppm.h"

namespace studiocast::open_video {

struct EyeContactRuntimeStatus {
  bool uses_cpu_face_detection = true;
  bool uses_cpu_landmarks = true;
  bool uses_cpu_preprocess = true;
  bool uses_cpu_tensor_io = true;
  bool uses_cpu_postprocess = true;
  bool device_resident_gpu_path = false;

  bool left_cuda_ep_active = false;
  bool right_cuda_ep_active = false;
  bool cuda_ep_cpu_tensor_io_active = false;
  bool cpu_only_session_active = false;

  std::string summary;
};

struct EyeContactScratchStatus {
  std::uint64_t geometry_rebuilds = 0;
  std::uint64_t resize_plan_rebuilds = 0;
  int frame_width = 0;
  int frame_height = 0;
  std::size_t upscale_capacity = 0;
};

// Open Video Eye Contact effect using gaze-correction-cam style models.
//
// Expected pipeline (mirrors the upstream project):
//   1) YuNet detects a face bbox (cached)
//   2) dlib predicts 68 face landmarks (cached)
//   3) Eye crops + anchor maps are extracted for L/R eyes
//   4) L/R eye models run inference (ONNX Runtime)
//   5) The resulting flow field is used to warp the eye crops
//   6) Corrected eyes are composited back into the frame
//
// NOTE: This runtime intentionally starts with a conservative, best-effort
// implementation. Model conversions may evolve; the code tries to be robust
// to common ONNX layouts (NCHW/NHWC) and output variants (flow vs direct RGB).
class GazeCorrectionEyeContact {
public:
  GazeCorrectionEyeContact();
  ~GazeCorrectionEyeContact();

  GazeCorrectionEyeContact(const GazeCorrectionEyeContact &) = delete;
  GazeCorrectionEyeContact &
  operator=(const GazeCorrectionEyeContact &) = delete;

  void Reset();

  // Loads the Open Video eye_contact pack and its dependencies.
  bool EnsureInitialized(const std::string &requested_model_id,
                         std::string *error);

  // Apply eye contact correction in-place on an RGB24 frame.
  //
  // Returns true on success (or clean bypass), false on fatal errors.
  bool ApplyRgbInPlace(std::uint64_t capture_sequence, std::uint8_t *rgb,
                       int width, int height, std::size_t stride, int strength,
                       bool look_away_enabled,
                       const std::string &face_detection_model_id,
                       const std::string &requested_model_id,
                       YunetFaceDetector *yunet, FrameAnalysisCache *cache,
                       std::string *error);

  bool initialized() const { return initialized_; }
  bool disabled() const { return disabled_; }
  bool using_cpu_fallback() const { return using_cpu_fallback_; }
  const std::string &active_model_id() const { return active_model_id_; }
  const std::string &sticky_warning() const { return sticky_warning_; }
  EyeContactRuntimeStatus runtime_status() const;
  EyeContactScratchStatus scratch_status() const;

  // Setup/reconfiguration entry point for bounded per-eye scratch. Repeated
  // calls with the same frame and model geometry are allocation-free.
  bool PrepareScratch(int frame_width, int frame_height, int left_input_width,
                      int left_input_height, int right_input_width,
                      int right_input_height, std::string *error);

private:
  struct EyeRuntime {
    studiocast::onnx::OrtSessionInfo session_info;
    std::unique_ptr<studiocast::onnx::OrtSession> session_cuda;
    std::unique_ptr<studiocast::onnx::OrtSession> session_cpu;
    studiocast::onnx::OrtSession *session_active = nullptr;

    bool eye_is_nhwc = false;
    bool anchors_is_nhwc = false;
    bool output_is_nhwc = false;
    int input_w = 64;
    int input_h = 48;
    int anchor_channels = 12;
    int out_channels = 0;

    std::string eye_name;
    std::string anchors_name;
    std::string angles_name;
    std::string output_name;

    std::vector<int64_t> eye_shape;
    std::vector<int64_t> anchors_shape;
    std::vector<int64_t> angles_shape;
    std::vector<int64_t> output_shape;

    // ORT I/O buffers.
    std::vector<float> eye_tensor;
    std::vector<float> anchors_tensor;
    std::vector<float> angles_tensor;
    std::vector<float> output_tensor;

    // Scratch for ORT bindings.
    std::vector<studiocast::onnx::OrtSession::RunInput> ort_inputs;
    std::vector<studiocast::onnx::OrtSession::RunOutput> ort_outputs;
  };

  struct EyeData {
    bool valid = false;
    bool is_left = false;
    int crop_top = 0;
    int crop_left = 0;
    int crop_w = 0;
    int crop_h = 0;
    int input_w = 64;
    int input_h = 48;
    std::vector<std::uint8_t> eye_rgb_u8; // resized (input_w*input_h*3)
    std::vector<float> eye_nchw_f32;      // (3*input_h*input_w)
    std::vector<float> anchors_nchw_f32;  // (12*input_h*input_w)
  };

  static float Clamp01(float x);
  static float Lerp(float a, float b, float t);

  static std::string ChoosePreferredModelId(const ModelPackRegistry &reg);

  static bool DetectEyeInputsFromSession(
      const studiocast::onnx::OrtSessionInfo &info, std::string *out_eye_name,
      std::vector<int64_t> *out_eye_shape, std::string *out_anchors_name,
      std::vector<int64_t> *out_anchors_shape, std::string *out_angles_name,
      std::vector<int64_t> *out_angles_shape, std::string *error);

  static bool
  DetectOutputFromSession(const studiocast::onnx::OrtSessionInfo &info,
                          std::string *out_name,
                          std::vector<int64_t> *out_shape, std::string *error);

  bool LoadModelPack(const std::string &model_id, std::string *error);
  bool InitRuntimeForEye(const std::filesystem::path &onnx_path, EyeRuntime *rt,
                         std::string *error);
  bool ConfigureRuntimeIo(EyeRuntime *rt, std::string *error);

  bool EnsureDlibDependency(std::string *error);

  static const FaceDetection *
  ChooseBestFace(const std::vector<FaceDetection> &faces);
  bool ExtractEyeData(const std::uint8_t *rgb, int frame_w, int frame_h,
                      std::size_t frame_stride, const FaceLandmarks &lms,
                      bool left_eye, int input_w, int input_h, EyeData *out,
                      std::string *error);

  bool RunModelForEye(EyeRuntime *rt, const EyeData &eye, float yaw,
                      float pitch, std::string *error);

  bool WarpOrDecodeOutputToRgbU8(const EyeRuntime &rt, const EyeData &eye,
                                 std::vector<std::uint8_t> *out_rgb_u8,
                                 std::string *error) const;

  void CompositeEyeIntoFrame(std::uint8_t *frame_rgb, int frame_w, int frame_h,
                             std::size_t frame_stride, const EyeData &eye,
                             const std::vector<std::uint8_t> &corrected_rgb_u8,
                             float strength01, bool left_eye);

  bool EnsureResizePlan(bool left_eye, bool upscale, int src_w, int src_h,
                        int dst_w, int dst_h, std::string *error);

  void DisableAfterFailure(const std::string &why);

  bool initialized_ = false;
  bool disabled_ = false;
  bool using_cpu_fallback_ = false;

  ModelPackRegistry registry_;
  std::string active_model_id_;
  std::string active_requested_model_id_;
  std::string required_landmarks_id_;
  std::filesystem::path left_model_path_;
  std::filesystem::path right_model_path_;

  DlibFaceLandmarks dlib_landmarks_;
  EyeRuntime left_;
  EyeRuntime right_;

  EyeData left_eye_scratch_;
  EyeData right_eye_scratch_;
  std::vector<std::uint8_t> left_corrected_scratch_;
  std::vector<std::uint8_t> right_corrected_scratch_;
  std::vector<std::uint8_t> upscale_scratch_;
  studiocast::video::Rgb24BilinearResizePlan left_extract_plan_;
  studiocast::video::Rgb24BilinearResizePlan right_extract_plan_;
  studiocast::video::Rgb24BilinearResizePlan left_upscale_plan_;
  studiocast::video::Rgb24BilinearResizePlan right_upscale_plan_;
  std::array<int, 4> left_extract_geometry_{};
  std::array<int, 4> right_extract_geometry_{};
  std::array<int, 4> left_upscale_geometry_{};
  std::array<int, 4> right_upscale_geometry_{};
  int scratch_frame_width_ = 0;
  int scratch_frame_height_ = 0;
  int scratch_left_input_width_ = 0;
  int scratch_left_input_height_ = 0;
  int scratch_right_input_width_ = 0;
  int scratch_right_input_height_ = 0;
  std::uint64_t scratch_geometry_rebuilds_ = 0;
  std::uint64_t scratch_resize_plan_rebuilds_ = 0;

  std::string sticky_warning_;
  int runtime_failures_ = 0;
};

} // namespace studiocast::open_video
