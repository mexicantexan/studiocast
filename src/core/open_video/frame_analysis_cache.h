#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace studiocast::open_video {

// Shared analysis/artifact outputs computed once per capture frame.
//
// Many video effects need the same underlying signal (foreground matte, face
// boxes, landmarks). StudioCast avoids re-running expensive inference multiple
// times per frame by caching analysis outputs keyed by capture_sequence.
struct FaceDetection {
  float x = 0.0f;
  float y = 0.0f;
  float w = 0.0f;
  float h = 0.0f;
  float score = 0.0f;
};

struct FaceLandmarks {
  // Normalized pixel coordinates in the input frame.
  //  - x,y are in pixel space (0..width-1 / 0..height-1).
  std::vector<std::pair<float, float>> points;
};

enum class FrameMatteStorage {
  cpu_f32_alpha,
  device_f32_alpha,
  cuda_f32_alpha = device_f32_alpha,
  vulkan_f32_alpha = device_f32_alpha,
  maxine_gpu_alpha,
};

struct FrameMatteArtifactKey {
  std::string provider_id;
  std::string model_id;
  FrameMatteStorage storage = FrameMatteStorage::cpu_f32_alpha;

  int frame_width = 0;
  int frame_height = 0;
  int matte_width = 0;
  int matte_height = 0;

  // Provider-owned semantic key for parameters that affect matte output, such
  // as green-screen mode, temporal state policy, or model preprocess options.
  std::uint64_t config_fingerprint = 0;

  // Optional opaque compatibility handles for device-local artifacts. They are
  // intentionally integers to keep this header independent of CUDA/Vulkan/
  // Maxine headers and to make the cache testable without GPU hardware. For
  // Vulkan, stream is the selected queue identity.
  std::uintptr_t device_context = 0;
  std::uintptr_t stream = 0;

  bool CompatibleWith(const FrameMatteArtifactKey &other) const {
    return provider_id == other.provider_id && model_id == other.model_id &&
           storage == other.storage && frame_width == other.frame_width &&
           frame_height == other.frame_height &&
           matte_width == other.matte_width &&
           matte_height == other.matte_height &&
           config_fingerprint == other.config_fingerprint &&
           device_context == other.device_context && stream == other.stream;
  }
};

inline bool operator==(const FrameMatteArtifactKey &a,
                       const FrameMatteArtifactKey &b) {
  return a.CompatibleWith(b);
}

inline bool operator!=(const FrameMatteArtifactKey &a,
                       const FrameMatteArtifactKey &b) {
  return !(a == b);
}

struct FrameMatteArtifact {
  FrameMatteArtifactKey key;

  // CPU-visible alpha payload when storage == cpu_f32_alpha.
  std::vector<float> cpu_alpha;

  // Opaque provider handles for GPU/device-local artifacts. For Open CUDA this
  // can identify a CudaImage/CudaTensor allocation; for Open Vulkan this can
  // identify a VulkanImage/VulkanTensor allocation; for Maxine it can identify
  // an NvCVImage. Ownership stays with the provider that published the
  // artifact.
  std::uintptr_t handle = 0;
  std::uintptr_t aux_handle = 0;
};

class FrameArtifactCache {
public:
  // The prepared pipeline currently has fewer than eight matte consumers. Keep
  // the per-frame cache explicitly bounded so a malformed effect graph cannot
  // turn it into an unbounded hot-path cache.
  static constexpr std::size_t kMaxMatteArtifactsPerFrame = 8;

  std::uint64_t capture_sequence() const { return capture_sequence_; }

  void BeginFrame(std::uint64_t seq) {
    if (seq == capture_sequence_)
      return;
    capture_sequence_ = seq;
    ClearArtifacts();
  }

  void ClearArtifacts() { active_mattes_ = 0; }

  std::size_t active_matte_count() const { return active_mattes_; }

  const FrameMatteArtifact *FindMatte(std::uint64_t seq,
                                      const FrameMatteArtifactKey &key) const {
    if (seq != capture_sequence_)
      return nullptr;
    for (std::size_t i = 0; i < active_mattes_; ++i) {
      const auto &matte = mattes_[i];
      if (matte.key == key)
        return &matte;
    }
    return nullptr;
  }

  const FrameMatteArtifact *StoreMatte(std::uint64_t seq,
                                       FrameMatteArtifact artifact) {
    BeginFrame(seq);
    for (std::size_t i = 0; i < active_mattes_; ++i) {
      auto &matte = mattes_[i];
      if (matte.key == artifact.key) {
        AssignPreservingStorage(artifact, &matte);
        return &matte;
      }
    }
    if (active_mattes_ >= mattes_.size())
      return nullptr;
    auto &slot = mattes_[active_mattes_++];
    AssignPreservingStorage(artifact, &slot);
    return &slot;
  }

  template <typename Producer>
  bool GetOrComputeMatte(std::uint64_t seq, const FrameMatteArtifactKey &key,
                         Producer &&producer, const FrameMatteArtifact **out,
                         std::string *error) {
    if (error)
      error->clear();
    if (out)
      *out = nullptr;

    BeginFrame(seq);
    if (const auto *hit = FindMatte(seq, key)) {
      if (out)
        *out = hit;
      return true;
    }

    if (active_mattes_ >= mattes_.size()) {
      if (error)
        *error = "per-frame matte artifact capacity exceeded";
      return false;
    }

    // Produce directly into an inactive cache slot. Clearing the payload keeps
    // its allocation available for the next capture sequence.
    auto &produced = mattes_[active_mattes_];
    produced.key = key;
    produced.cpu_alpha.clear();
    produced.handle = 0;
    produced.aux_handle = 0;
    std::string producer_error;
    if (!producer(&produced, &producer_error)) {
      if (error)
        *error = producer_error;
      return false;
    }
    if (produced.key != key) {
      if (error)
        *error = "matte artifact producer returned an incompatible key";
      return false;
    }

    const auto *stored = &produced;
    ++active_mattes_;
    if (out)
      *out = stored;
    return true;
  }

private:
  static void AssignPreservingStorage(const FrameMatteArtifact &src,
                                      FrameMatteArtifact *dst) {
    dst->key = src.key;
    dst->cpu_alpha.assign(src.cpu_alpha.begin(), src.cpu_alpha.end());
    dst->handle = src.handle;
    dst->aux_handle = src.aux_handle;
  }

  std::uint64_t capture_sequence_ = 0;
  std::array<FrameMatteArtifact, kMaxMatteArtifactsPerFrame> mattes_{};
  std::size_t active_mattes_ = 0;
};

struct FrameAnalysisCache {
  std::uint64_t capture_sequence = 0;
  FrameArtifactCache artifacts;

  // Best-effort "begin frame" helper; clears cached results when the sequence
  // changes.
  void BeginFrame(std::uint64_t seq) {
    if (seq == capture_sequence)
      return;
    capture_sequence = seq;
    RecycleFaceDetections();
    RecycleFaceLandmarks();
    artifacts.BeginFrame(seq);
  }

  void Clear() {
    RecycleFaceDetections();
    RecycleFaceLandmarks();
    artifacts.ClearArtifacts();
  }

  std::vector<FaceDetection> &PrepareFaceDetections(std::size_t capacity) {
    if (!face_detections) {
      face_detections.emplace();
      face_detections->swap(recycled_face_detections_);
    }
    face_detections->clear();
    if (face_detections->capacity() < capacity)
      face_detections->reserve(capacity);
    return *face_detections;
  }

  FaceLandmarks &PrepareFaceLandmarks(std::size_t capacity) {
    if (!face_landmarks) {
      face_landmarks.emplace();
      face_landmarks->points.swap(recycled_face_landmarks_.points);
    }
    face_landmarks->points.clear();
    if (face_landmarks->points.capacity() < capacity)
      face_landmarks->points.reserve(capacity);
    return *face_landmarks;
  }

  std::optional<std::vector<FaceDetection>> face_detections;
  std::optional<FaceLandmarks> face_landmarks;

private:
  void RecycleFaceDetections() {
    if (!face_detections)
      return;
    recycled_face_detections_.clear();
    face_detections->swap(recycled_face_detections_);
    face_detections.reset();
  }

  void RecycleFaceLandmarks() {
    if (!face_landmarks)
      return;
    recycled_face_landmarks_.points.clear();
    face_landmarks->points.swap(recycled_face_landmarks_.points);
    face_landmarks.reset();
  }

  std::vector<FaceDetection> recycled_face_detections_;
  FaceLandmarks recycled_face_landmarks_;
};

} // namespace studiocast::open_video
