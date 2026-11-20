#include "core/open_video/frame_analysis_cache.h"
#include "core/open_video/gaze_correction_eye_contact.h"
#include "core/open_video/yunet_face_detector.h"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>
#include <vector>

namespace {

std::atomic<std::size_t> g_allocations{0};
std::atomic<bool> g_count_allocations{false};

bool Require(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

studiocast::open_video::FrameMatteArtifactKey
MatteKey(const std::string &model_id = "steady-model") {
  return studiocast::open_video::FrameMatteArtifactKey{
      .provider_id = "open-video-test",
      .model_id = model_id,
      .storage = studiocast::open_video::FrameMatteStorage::cpu_f32_alpha,
      .frame_width = 16,
      .frame_height = 16,
      .matte_width = 8,
      .matte_height = 8,
      .config_fingerprint = 7,
  };
}

bool TestFrameAnalysisStorageIsRecycled() {
  using namespace studiocast::open_video;
  FrameAnalysisCache cache;
  cache.BeginFrame(1);
  auto &detections = cache.PrepareFaceDetections(32);
  detections.resize(32);
  auto &landmarks = cache.PrepareFaceLandmarks(68);
  landmarks.points.resize(68);
  const auto *detection_data = detections.data();
  const auto *landmark_data = landmarks.points.data();

  cache.BeginFrame(2);
  bool ok = true;
  ok &= Require(!cache.face_detections.has_value(),
                "new sequence must invalidate face detections");
  ok &= Require(!cache.face_landmarks.has_value(),
                "new sequence must invalidate landmarks");
  auto &reused_detections = cache.PrepareFaceDetections(32);
  auto &reused_landmarks = cache.PrepareFaceLandmarks(68);
  ok &= Require(reused_detections.data() == detection_data,
                "face detection capacity was not recycled");
  ok &= Require(reused_landmarks.points.data() == landmark_data,
                "landmark capacity was not recycled");
  return ok;
}

bool TestMatteSlotsAreBoundedAndReusable() {
  using namespace studiocast::open_video;
  FrameArtifactCache cache;
  const FrameMatteArtifact *matte = nullptr;
  std::string error;
  auto producer = [](FrameMatteArtifact *out, std::string *) {
    out->cpu_alpha.resize(64, 0.5f);
    return true;
  };
  bool ok = cache.GetOrComputeMatte(1, MatteKey(), producer, &matte, &error);
  const auto *payload = matte ? matte->cpu_alpha.data() : nullptr;
  ok &= cache.GetOrComputeMatte(2, MatteKey(), producer, &matte, &error);
  ok &= Require(matte && matte->cpu_alpha.data() == payload,
                "matte payload capacity was not reused across frames");

  for (std::size_t i = 0; i < FrameArtifactCache::kMaxMatteArtifactsPerFrame;
       ++i) {
    ok &= cache.GetOrComputeMatte(3, MatteKey("model-" + std::to_string(i)),
                                  producer, &matte, &error);
  }
  ok &= Require(cache.active_matte_count() ==
                    FrameArtifactCache::kMaxMatteArtifactsPerFrame,
                "matte cache did not reach its declared bound");
  ok &= Require(!cache.GetOrComputeMatte(3, MatteKey("overflow"), producer,
                                         &matte, &error),
                "matte cache must fail closed at its declared bound");
  ok &= Require(matte == nullptr,
                "bounded matte failure must not return a stale artifact");
  return ok;
}

bool TestYunetBoundedTopKAndInvalidation() {
  using namespace studiocast::open_video;
  YunetDetectionScratch scratch;
  bool ok = Require(scratch.Configure(4), "YuNet scratch configure failed");
  scratch.BeginFrame();
  scratch.Consider({0, 0, 10, 10, 0.5f});
  scratch.Consider({20, 0, 10, 10, 0.9f});
  scratch.Consider({40, 0, 10, 10, 0.7f});
  scratch.Consider({60, 0, 10, 10, 0.9f});
  scratch.Consider({80, 0, 10, 10, 0.8f});
  const auto &kept = scratch.Finalize(1.0f, 0, 0, 200, 100, 0.3f);
  ok &= Require(kept.size() == 4, "YuNet scratch did not enforce top-k");
  ok &= Require(kept.size() == 4 && kept[0].x == 20.0f && kept[1].x == 60.0f &&
                    kept[2].x == 80.0f && kept[3].x == 40.0f,
                "YuNet top-k score/tie ordering changed");
  const auto configured = scratch.stats();
  ok &= Require(scratch.Configure(4), "stable YuNet configure failed");
  ok &= Require(scratch.stats().rebuilds == configured.rebuilds,
                "stable YuNet configure rebuilt scratch");
  ok &= Require(scratch.Configure(8), "changed YuNet configure failed");
  ok &= Require(scratch.stats().rebuilds == configured.rebuilds + 1,
                "YuNet top-k change must rebuild exactly once");
  scratch.Invalidate();
  ok &=
      Require(scratch.stats().invalidations == 1 && scratch.stats().top_k == 0,
              "YuNet invalidation did not clear the active contract");
  return ok;
}

bool TestSteadyScratchRegionsAllocateZero() {
  using namespace studiocast::open_video;
  FrameAnalysisCache analysis;
  FrameArtifactCache artifacts;
  YunetDetectionScratch yunet;
  GazeCorrectionEyeContact eye_contact;
  std::string error;
  const auto key = MatteKey();
  const FrameMatteArtifact *matte = nullptr;
  auto producer = [](FrameMatteArtifact *out, std::string *) {
    out->cpu_alpha.resize(64, 0.25f);
    return true;
  };

  (void)yunet.Configure(32);
  (void)eye_contact.PrepareScratch(1280, 720, 64, 48, 64, 48, &error);
  analysis.BeginFrame(1);
  analysis.PrepareFaceDetections(32).resize(32);
  analysis.PrepareFaceLandmarks(68).points.resize(68);
  (void)artifacts.GetOrComputeMatte(1, key, producer, &matte, &error);

  // Warm each transition once before measuring.
  analysis.BeginFrame(2);
  analysis.PrepareFaceDetections(32).resize(32);
  analysis.PrepareFaceLandmarks(68).points.resize(68);
  (void)artifacts.GetOrComputeMatte(2, key, producer, &matte, &error);

  g_allocations.store(0);
  g_count_allocations.store(true);
  for (std::uint64_t frame = 3; frame < 131; ++frame) {
    analysis.BeginFrame(frame);
    analysis.PrepareFaceDetections(32).resize(32);
    analysis.PrepareFaceLandmarks(68).points.resize(68);
    (void)artifacts.GetOrComputeMatte(frame, key, producer, &matte, &error);

    yunet.BeginFrame();
    for (int i = 0; i < 64; ++i) {
      yunet.Consider(FaceDetection{static_cast<float>(i * 12), 0.0f, 10.0f,
                                   10.0f, static_cast<float>(i) / 64.0f});
    }
    (void)yunet.Finalize(1.0f, 0, 0, 1280, 720, 0.3f);
    (void)eye_contact.PrepareScratch(1280, 720, 64, 48, 64, 48, &error);
  }
  g_count_allocations.store(false);

  const auto measured_allocations = g_allocations.load();
  bool ok = Require(measured_allocations == 0,
                    "post-warm Open Video scratch regions allocated");
  const auto before = eye_contact.scratch_status();
  ok &= Require(eye_contact.PrepareScratch(1920, 1080, 64, 48, 64, 48, &error),
                "eye scratch geometry rebuild failed");
  const auto after = eye_contact.scratch_status();
  ok &= Require(after.geometry_rebuilds == before.geometry_rebuilds + 1,
                "eye geometry change must rebuild exactly once");
  ok &=
      Require(eye_contact.PrepareScratch(1920, 1080, 64, 48, 64, 48, &error) &&
                  eye_contact.scratch_status().geometry_rebuilds ==
                      after.geometry_rebuilds,
              "stable eye geometry rebuilt scratch");
  if (ok) {
    std::cout << "scratch_measurement iterations=128 allocations="
              << measured_allocations
              << " yunet_top_k=32 matte_slots=1 eye_geometry_rebuilds="
              << eye_contact.scratch_status().geometry_rebuilds << '\n';
  }
  return ok;
}

} // namespace

void *operator new(std::size_t size) {
  if (g_count_allocations.load(std::memory_order_relaxed))
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *pointer = std::malloc(size))
    return pointer;
  throw std::bad_alloc();
}

void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete[](void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete[](void *pointer, std::size_t) noexcept {
  std::free(pointer);
}

int main() {
  bool ok = true;
  ok &= TestFrameAnalysisStorageIsRecycled();
  ok &= TestMatteSlotsAreBoundedAndReusable();
  ok &= TestYunetBoundedTopKAndInvalidation();
  ok &= TestSteadyScratchRegionsAllocateZero();
  if (ok)
    std::cout << "Open Video scratch reuse tests passed\n";
  return ok ? 0 : 1;
}
