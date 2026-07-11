#pragma once

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
  // an NvCVImage. Ownership stays with the provider that published the artifact.
  std::uintptr_t handle = 0;
  std::uintptr_t aux_handle = 0;
};

class FrameArtifactCache {
public:
  std::uint64_t capture_sequence() const { return capture_sequence_; }

  void BeginFrame(std::uint64_t seq) {
    if (seq == capture_sequence_)
      return;
    capture_sequence_ = seq;
    ClearArtifacts();
  }

  void ClearArtifacts() { mattes_.clear(); }

  const FrameMatteArtifact *
  FindMatte(std::uint64_t seq, const FrameMatteArtifactKey &key) const {
    if (seq != capture_sequence_)
      return nullptr;
    for (const auto &matte : mattes_) {
      if (matte.key == key)
        return &matte;
    }
    return nullptr;
  }

  const FrameMatteArtifact *
  StoreMatte(std::uint64_t seq, FrameMatteArtifact artifact) {
    BeginFrame(seq);
    for (auto &matte : mattes_) {
      if (matte.key == artifact.key) {
        matte = std::move(artifact);
        return &matte;
      }
    }
    mattes_.push_back(std::move(artifact));
    return &mattes_.back();
  }

  template <typename Producer>
  bool GetOrComputeMatte(std::uint64_t seq,
                         const FrameMatteArtifactKey &key,
                         Producer &&producer,
                         const FrameMatteArtifact **out,
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

    FrameMatteArtifact produced;
    produced.key = key;
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

    const auto *stored = StoreMatte(seq, std::move(produced));
    if (out)
      *out = stored;
    return true;
  }

private:
  std::uint64_t capture_sequence_ = 0;
  std::vector<FrameMatteArtifact> mattes_;
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
    face_detections.reset();
    face_landmarks.reset();
    artifacts.BeginFrame(seq);
  }

  void Clear() {
    face_detections.reset();
    face_landmarks.reset();
    artifacts.ClearArtifacts();
  }

  std::optional<std::vector<FaceDetection>> face_detections;
  std::optional<FaceLandmarks> face_landmarks;
};

} // namespace studiocast::open_video
