#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "core/open_video/frame_analysis_cache.h"
#include "core/open_video/model_pack_registry.h"

namespace studiocast::open_video {

// dlib-based face landmark extractor.
//
// This is used by the Open Video Eye Contact implementation to obtain
// dense face landmarks (typically the 68-point iBUG 300-W predictor).
//
// The predictor model is discovered via the Open Video model registry:
//   ~/.local/share/studiocast/models/open_video/face_landmarks/<id>/
//
// The results are cached in FrameAnalysisCache so other stages can reuse
// the same landmarks without re-running inference.
class DlibFaceLandmarks {
public:
  DlibFaceLandmarks();
  ~DlibFaceLandmarks();

  DlibFaceLandmarks(const DlibFaceLandmarks &) = delete;
  DlibFaceLandmarks &operator=(const DlibFaceLandmarks &) = delete;

  void Reset();

  // Initialize the shape predictor.
  //
  // If model_id_override is non-empty, the registry will try to load that
  // exact face_landmarks pack id.
  bool EnsureInitialized(const std::string &model_id_override,
                         std::string *error);
  bool EnsureInitialized(std::string *error) {
    return EnsureInitialized(std::string(), error);
  }

  // Ensure cache->face_landmarks is populated for this frame.
  //
  // `face` is the selected face detection bounding box (pixels, in the full
  // frame).
  bool EnsureLandmarksForFrame(const std::uint8_t *rgb, int width, int height,
                               std::size_t stride,
                               std::uint64_t capture_sequence,
                               const FaceDetection &face,
                               FrameAnalysisCache *cache, std::string *error);
  bool EnsurePreparedLandmarksForFrame(const std::uint8_t *rgb, int width,
                                       int height, std::size_t stride,
                                       std::uint64_t capture_sequence,
                                       const FaceDetection &face,
                                       FrameAnalysisCache *cache,
                                       std::string *error);

  bool initialized() const { return initialized_; }
  bool disabled() const { return disabled_; }
  const std::string &active_model_id() const { return active_model_id_; }
  const std::filesystem::path &active_model_path() const {
    return active_model_path_;
  }
  const std::string &sticky_warning() const { return sticky_warning_; }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  void DisableAfterFailure(const std::string &why);
  bool EnsureLandmarksForFrameImpl(const std::uint8_t *rgb, int width,
                                   int height, std::size_t stride,
                                   std::uint64_t capture_sequence,
                                   const FaceDetection &face,
                                   FrameAnalysisCache *cache, bool allow_setup,
                                   std::string *error);

  bool initialized_ = false;
  bool disabled_ = false;

  ModelPackRegistry registry_;
  std::string active_model_id_;
  std::filesystem::path active_model_path_;
  std::filesystem::path predictor_path_;

  std::string sticky_warning_;
};

} // namespace studiocast::open_video
