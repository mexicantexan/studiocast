#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "core/onnx/ort_session.h"
#include "core/open_video/frame_analysis_cache.h"
#include "core/open_video/model_pack_registry.h"

namespace studiocast::open_video {

inline constexpr std::string_view kYunetTensorGeometryMismatchReason =
    "vulkan_auto_frame_yunet_tensor_geometry_mismatch";
inline constexpr std::string_view kYunetProviderPolicyViolationReason =
    "vulkan_auto_frame_yunet_provider_policy_violation";
inline constexpr std::string_view kYunetWarmupFailedReason =
    "vulkan_auto_frame_yunet_warmup_failed";

enum class YunetProviderPolicy {
  // Existing Open Video/Open CUDA behavior: prefer CUDA EP and allow the
  // canonical ORT wrapper to fall back to CPU.
  prefer_cuda,
  // Explicit Open Vulkan behavior: CPUExecutionProvider only. A Vulkan effect
  // must never silently run analysis on CUDA/TensorRT or another GPU/device.
  cpu_only,
};

const char *YunetProviderPolicyName(YunetProviderPolicy policy);
studiocast::onnx::OrtSessionOptions
YunetOrtSessionOptions(YunetProviderPolicy policy);

// Validates the graph's first input against the manifest/runtime contract.
// Fixed graph dimensions must match exactly; dynamic dimensions are accepted.
bool ValidateYunetInputTensorContract(const std::vector<int64_t> &graph_shape,
                                      bool manifest_nhwc, int manifest_width,
                                      int manifest_height, std::string *error);

// Bounded, reusable CPU postprocess workspace for YuNet. Configure is a setup
// operation; BeginFrame/Consider/Finalize perform no heap allocation while the
// configured top-k remains unchanged.
class YunetDetectionScratch {
public:
  static constexpr std::size_t kMaximumTopK = 20000;

  struct Stats {
    std::uint64_t rebuilds = 0;
    std::uint64_t invalidations = 0;
    std::size_t top_k = 0;
    std::size_t candidate_capacity = 0;
    std::size_t mapped_capacity = 0;
    std::size_t kept_capacity = 0;
  };

  bool Configure(std::size_t top_k);
  void Invalidate();
  void BeginFrame();
  void Consider(FaceDetection detection);
  const std::vector<FaceDetection> &Finalize(float scale, int pad_x, int pad_y,
                                             int frame_width, int frame_height,
                                             float nms_threshold);
  Stats stats() const;

private:
  struct Candidate {
    FaceDetection detection;
    std::uint64_t insertion_order = 0;
  };

  std::size_t top_k_ = 0;
  std::uint64_t next_insertion_order_ = 0;
  std::uint64_t rebuilds_ = 0;
  std::uint64_t invalidations_ = 0;
  std::vector<Candidate> candidates_;
  std::vector<FaceDetection> mapped_;
  std::vector<FaceDetection> kept_;
};

struct FaceDetectionRuntimeStatus {
  bool uses_cpu_preprocess = true;
  bool uses_cpu_tensor_io = true;
  bool uses_cpu_postprocess = true;
  bool device_resident_gpu_path = false;

  bool cuda_ep_active = false;
  bool cuda_ep_cpu_tensor_io_active = false;
  bool cpu_only_session_active = false;
  bool warmup_complete = false;

  std::string provider_policy;
  std::string reason_code;

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
  bool EnsureInitialized(
      const std::string &requested_model_id, std::string *error,
      YunetProviderPolicy provider_policy = YunetProviderPolicy::prefer_cuda);

  // Ensures cache->face_detections is populated for the given capture_sequence.
  //
  // Returns true if the model ran (even if zero faces were detected).
  // Returns false if the model could not run; cache->face_detections is left
  // unset.
  bool EnsureDetectionsForFrame(
      const std::uint8_t *rgb, int width, int height, std::size_t stride,
      const std::string &requested_model_id, std::uint64_t capture_sequence,
      FrameAnalysisCache *cache, std::string *error,
      YunetProviderPolicy provider_policy = YunetProviderPolicy::prefer_cuda);

  bool available() const { return initialized_; }
  const std::string &active_model_id() const { return active_model_id_; }
  FaceDetectionRuntimeStatus runtime_status() const;

private:
  struct Settings {
    int input_w = 320;
    int input_h = 320;
    bool input_nhwc = false;
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
  bool warmed_ = false;
  bool input_is_nhwc_ = false;
  YunetProviderPolicy active_provider_policy_ =
      YunetProviderPolicy::prefer_cuda;
  std::string last_reason_code_;

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
  studiocast::onnx::OrtSession::RunInput run_input_{};
  std::vector<studiocast::onnx::OrtSession::RunOutput> run_outputs_;
  YunetDetectionScratch detection_scratch_;

  // Indices in outputs_ for each stride.
  std::array<int, 3> cls_idx_{{-1, -1, -1}};
  std::array<int, 3> obj_idx_{{-1, -1, -1}};
  std::array<int, 3> bbox_idx_{{-1, -1, -1}};
  std::array<int, 3> kps_idx_{{-1, -1, -1}};

  bool LoadSettingsFromManifest(const std::filesystem::path &manifest_path,
                                std::string *error);
  bool BuildBindings(std::string *error);
  bool Warmup(std::string *error);

  static Letterbox ComputeLetterbox(int src_w, int src_h, int dst_w, int dst_h);
  void FillInputTensorBgr(const std::uint8_t *rgb, int src_w, int src_h,
                          std::size_t src_stride, const Letterbox &lb);

  static float IoU(const FaceDetection &a, const FaceDetection &b);
};

} // namespace studiocast::open_video
