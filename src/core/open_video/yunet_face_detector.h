#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "core/onnx/ort_session.h"
#include "core/open_video/frame_analysis_cache.h"
#include "core/open_video/model_pack_registry.h"

namespace studiocast::open_video {

struct FaceDetectionRuntimeStatus {
  bool uses_cpu_preprocess = true;
  bool uses_cpu_tensor_io = true;
  bool uses_cpu_postprocess = true;
  bool device_resident_gpu_path = false;

  bool cuda_ep_active = false;
  bool cuda_ep_cpu_tensor_io_active = false;
  bool cpu_only_session_active = false;

  std::string summary;
};

// YuNet face detector wrapper.
//
// This is used by the open-source video pipeline for Auto Frame tracking.
// It loads a "face_detection" Open Video model pack (YuNet) and produces
// bounding boxes in the current frame's pixel space.
//
// Design goals:
//  - Real-time streaming friendly (runs on a downscaled letterboxed input).
//  - Best-effort GPU acceleration via ONNX Runtime CUDA EP when available.
//  - No duplicated inference within a single capture frame via
//  FrameAnalysisCache.
class YunetFaceDetector {
public:
  YunetFaceDetector() = default;
  ~YunetFaceDetector() = default;

  YunetFaceDetector(const YunetFaceDetector &) = delete;
  YunetFaceDetector &operator=(const YunetFaceDetector &) = delete;

  void Reset();

  // Loads a suitable YuNet model pack and creates an ORT session.
  // Safe to call multiple times. If requested_model_id is empty, the registry
  // default is used.
  bool EnsureInitialized(const std::string &requested_model_id,
                         std::string *error);

  // Ensures cache->face_detections is populated for the given capture_sequence.
  //
  // Returns true if the model ran (even if zero faces were detected).
  // Returns false if the model could not run; cache->face_detections is left
  // unset.
  bool EnsureDetectionsForFrame(const std::uint8_t *rgb, int width, int height,
                                std::size_t stride,
                                const std::string &requested_model_id,
                                std::uint64_t capture_sequence,
                                FrameAnalysisCache *cache, std::string *error);

  bool available() const { return initialized_; }
  const std::string &active_model_id() const { return active_model_id_; }
  FaceDetectionRuntimeStatus runtime_status() const;

private:
  struct Settings {
    int input_w = 320;
    int input_h = 320;
    float score_threshold = 0.9f;
    float nms_threshold = 0.3f;
    int top_k = 5000;
  } settings_;

  struct Letterbox {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int out_w = 0;
    int out_h = 0;
  };

  bool initialized_ = false;
  bool input_is_nhwc_ = false;

  std::string active_model_id_;
  std::string active_requested_model_id_;
  std::filesystem::path active_model_path_;

  ModelPackRegistry registry_;
  std::unique_ptr<studiocast::onnx::OrtSession> session_;
  studiocast::onnx::OrtSessionInfo session_info_;

  // Pre-allocated buffers.
  std::vector<float> input_tensor_;
  std::vector<int64_t> input_shape_;

  struct OutputBinding {
    std::string name;
    std::vector<float> data;
    std::vector<int64_t> shape;
  };

  // Output bindings in the order passed to ORT.
  std::vector<OutputBinding> outputs_;

  // Indices in outputs_ for each stride.
  std::array<int, 3> cls_idx_{{-1, -1, -1}};
  std::array<int, 3> obj_idx_{{-1, -1, -1}};
  std::array<int, 3> bbox_idx_{{-1, -1, -1}};
  std::array<int, 3> kps_idx_{{-1, -1, -1}};

  bool LoadSettingsFromManifest(const std::filesystem::path &manifest_path,
                                std::string *error);
  bool BuildBindings(std::string *error);

  static Letterbox ComputeLetterbox(int src_w, int src_h, int dst_w, int dst_h);
  void FillInputTensorBgr(const std::uint8_t *rgb, int src_w, int src_h,
                          std::size_t src_stride, const Letterbox &lb);

  static float IoU(const FaceDetection &a, const FaceDetection &b);
};

} // namespace studiocast::open_video
