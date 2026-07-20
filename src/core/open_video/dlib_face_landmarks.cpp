#include "core/open_video/dlib_face_landmarks.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#if STUDIOCAST_HAVE_DLIB
#include <dlib/image_processing.h>
#include <dlib/image_processing/shape_predictor.h>
#endif

namespace studiocast::open_video {

struct DlibFaceLandmarks::Impl {
#if STUDIOCAST_HAVE_DLIB
  dlib::shape_predictor predictor;
#endif
};

DlibFaceLandmarks::DlibFaceLandmarks() : impl_(std::make_unique<Impl>()) {}
DlibFaceLandmarks::~DlibFaceLandmarks() = default;

void DlibFaceLandmarks::Reset() {
  initialized_ = false;
  disabled_ = false;
  active_model_id_.clear();
  active_model_path_.clear();
  predictor_path_.clear();
  sticky_warning_.clear();
  registry_ = ModelPackRegistry();
}

void DlibFaceLandmarks::DisableAfterFailure(const std::string &why) {
  disabled_ = true;
  sticky_warning_ = why;
}

bool DlibFaceLandmarks::EnsureInitialized(const std::string &model_id_override,
                                          std::string *error) {
  if (error)
    error->clear();

  if (disabled_) {
    if (error && !sticky_warning_.empty())
      *error = sticky_warning_;
    return false;
  }

  if (initialized_)
    return true;

#if !STUDIOCAST_HAVE_DLIB
  (void)model_id_override;
  if (error) {
    *error = "dlib is not available in this build (STUDIOCAST_HAVE_DLIB=0).";
  }
  return false;
#else
  registry_ = ModelPackRegistry::ScanDefault();

  std::string model_id = model_id_override;
  if (model_id.empty()) {
    model_id = registry_.DefaultModelIdForTask("face_landmarks");
  }
  if (model_id.empty()) {
    if (error) {
      *error = "No Open Video face_landmarks model packs are installed.";
    }
    return false;
  }

  const auto pack = registry_.Find("face_landmarks", model_id);
  if (!pack.has_value()) {
    if (error) {
      *error = "Requested face_landmarks model pack not found: id='" +
               model_id + "'";
    }
    return false;
  }

  const ModelFile *predictor = nullptr;
  for (const auto &f : pack->files) {
    if (f.kind == "dlib_shape_predictor") {
      predictor = &f;
      break;
    }
  }
  if (!predictor) {
    // Fallback: accept any file kind that looks like a dlib predictor.
    for (const auto &f : pack->files) {
      if (f.kind.find("dlib") != std::string::npos) {
        predictor = &f;
        break;
      }
    }
  }
  if (!predictor || predictor->path.empty()) {
    if (error) {
      *error = "face_landmarks pack '" + model_id +
               "' does not contain a dlib_shape_predictor file.";
    }
    return false;
  }

  active_model_id_ = model_id;
  active_model_path_ = pack->root_dir;
  predictor_path_ = predictor->path;

  try {
    dlib::deserialize(predictor_path_.string()) >> impl_->predictor;
  } catch (const std::exception &e) {
    if (error) {
      std::ostringstream oss;
      oss << "Failed to load dlib shape predictor from '"
          << predictor_path_.string() << "': " << e.what();
      *error = oss.str();
    }
    return false;
  }

  initialized_ = true;
  return true;
#endif
}

bool DlibFaceLandmarks::EnsureLandmarksForFrame(
    const std::uint8_t *rgb, int width, int height, std::size_t stride,
    std::uint64_t capture_sequence, const FaceDetection &face,
    FrameAnalysisCache *cache, std::string *error) {
  if (error)
    error->clear();

  if (!cache) {
    if (error)
      *error = "EnsureLandmarksForFrame: cache is null";
    return false;
  }
  if (!rgb || width <= 0 || height <= 0 ||
      stride < static_cast<std::size_t>(width * 3)) {
    if (error)
      *error = "EnsureLandmarksForFrame: invalid RGB frame";
    return false;
  }

  cache->BeginFrame(capture_sequence);
  if (cache->face_landmarks.has_value())
    return true;

  if (!EnsureInitialized(error))
    return false;

#if !STUDIOCAST_HAVE_DLIB
  (void)face;
  if (error)
    *error = "dlib is not available in this build";
  return false;
#else
  // Build a small ROI around the detected face to avoid copying the full frame.
  const int fx0 = std::max(0, static_cast<int>(std::floor(face.x)));
  const int fy0 = std::max(0, static_cast<int>(std::floor(face.y)));
  const int fx1 = std::min(width, static_cast<int>(std::ceil(face.x + face.w)));
  const int fy1 =
      std::min(height, static_cast<int>(std::ceil(face.y + face.h)));

  const int fw = std::max(1, fx1 - fx0);
  const int fh = std::max(1, fy1 - fy0);
  const int mx = static_cast<int>(std::round(fw * 0.20f));
  const int my = static_cast<int>(std::round(fh * 0.20f));

  const int rx0 = std::max(0, fx0 - mx);
  const int ry0 = std::max(0, fy0 - my);
  const int rx1 = std::min(width, fx1 + mx);
  const int ry1 = std::min(height, fy1 + my);

  const int rw = std::max(1, rx1 - rx0);
  const int rh = std::max(1, ry1 - ry0);

  if (rw <= 1 || rh <= 1) {
    if (error)
      *error = "EnsureLandmarksForFrame: ROI invalid";
    return false;
  }

  dlib::array2d<dlib::rgb_pixel> roi;
  roi.set_size(rh, rw);
  for (int y = 0; y < rh; ++y) {
    const std::uint8_t *src = rgb + static_cast<std::size_t>(ry0 + y) * stride +
                              static_cast<std::size_t>(rx0) * 3;
    for (int x = 0; x < rw; ++x) {
      const std::uint8_t *p = src + x * 3;
      dlib::rgb_pixel px;
      px.red = p[0];
      px.green = p[1];
      px.blue = p[2];
      roi[y][x] = px;
    }
  }

  const int face_left = std::max(0, fx0 - rx0);
  const int face_top = std::max(0, fy0 - ry0);
  const int face_right = std::min(rw - 1, (fx1 - rx0) - 1);
  const int face_bottom = std::min(rh - 1, (fy1 - ry0) - 1);
  if (face_right <= face_left || face_bottom <= face_top) {
    if (error)
      *error =
          "EnsureLandmarksForFrame: face rectangle invalid after ROI transform";
    return false;
  }

  dlib::rectangle rect(face_left, face_top, face_right, face_bottom);

  dlib::full_object_detection det;
  try {
    det = impl_->predictor(roi, rect);
  } catch (const std::exception &e) {
    if (error) {
      *error = std::string("dlib shape predictor failed: ") + e.what();
    }
    DisableAfterFailure("dlib shape predictor failed (disabled): " +
                        std::string(e.what()));
    return false;
  }

  const unsigned long n = det.num_parts();
  auto &out = cache->PrepareFaceLandmarks(n);
  for (unsigned long i = 0; i < n; ++i) {
    const auto pt = det.part(i);
    out.points.emplace_back(static_cast<float>(pt.x() + rx0),
                            static_cast<float>(pt.y() + ry0));
  }
  return true;
#endif
}

} // namespace studiocast::open_video
