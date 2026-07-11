#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "core/open_video/frame_analysis_cache.h"

namespace {

bool Require(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

studiocast::open_video::FrameMatteArtifactKey MatteKey(
    std::string model_id,
    studiocast::open_video::FrameMatteStorage storage =
        studiocast::open_video::FrameMatteStorage::cpu_f32_alpha,
    std::string provider_id = "open_cuda") {
  studiocast::open_video::FrameMatteArtifactKey key;
  key.provider_id = std::move(provider_id);
  key.model_id = std::move(model_id);
  key.storage = storage;
  key.frame_width = 1280;
  key.frame_height = 720;
  key.matte_width = 256;
  key.matte_height = 144;
  key.config_fingerprint = 42;
  return key;
}

} // namespace

namespace studiocast::tests {

bool TestFrameArtifactCacheReusesCompatibleMatteWithinFrame() {
  studiocast::open_video::FrameArtifactCache cache;
  int matte_runs = 0;
  std::string error;
  const studiocast::open_video::FrameMatteArtifact *first = nullptr;
  const studiocast::open_video::FrameMatteArtifact *second = nullptr;

  const auto producer =
      [&](studiocast::open_video::FrameMatteArtifact *artifact,
          std::string *producer_error) {
        if (producer_error)
          producer_error->clear();
        ++matte_runs;
        artifact->key = MatteKey("rvm-test");
        artifact->cpu_alpha = {0.0f, 0.5f, 1.0f};
        return true;
      };

  bool ok = true;
  ok &= Require(cache.GetOrComputeMatte(7, MatteKey("rvm-test"), producer,
                                        &first, &error),
                "first compatible matte request should succeed");
  ok &= Require(cache.GetOrComputeMatte(7, MatteKey("rvm-test"), producer,
                                        &second, &error),
                "second compatible matte request should succeed");
  ok &= Require(matte_runs == 1,
                "compatible matte consumers should share one producer run");
  ok &= Require(first != nullptr && second != nullptr,
                "matte cache should return artifacts");
  ok &= Require(first == second,
                "compatible requests should return the cached artifact");
  if (first) {
    ok &= Require(first->cpu_alpha.size() == 3,
                  "cached matte should preserve CPU alpha payload");
  }
  return ok;
}

bool TestFrameArtifactCacheReusesCompatibleMaxineMatteWithinFrame() {
  studiocast::open_video::FrameArtifactCache cache;
  int matte_runs = 0;
  std::string error;
  const studiocast::open_video::FrameMatteArtifact *virtual_background_matte =
      nullptr;
  const studiocast::open_video::FrameMatteArtifact *key_light_matte = nullptr;
  const auto key =
      MatteKey("green-screen", studiocast::open_video::FrameMatteStorage::
                                   maxine_gpu_alpha,
               "maxine");

  const auto producer =
      [&](studiocast::open_video::FrameMatteArtifact *artifact,
          std::string *producer_error) {
        if (producer_error)
          producer_error->clear();
        ++matte_runs;
        artifact->key = key;
        artifact->handle = 0xBEEFu;
        return true;
      };

  bool ok = true;
  ok &= Require(cache.GetOrComputeMatte(9, key, producer,
                                        &virtual_background_matte, &error),
                "Maxine virtual background matte request should succeed");
  ok &= Require(cache.GetOrComputeMatte(9, key, producer, &key_light_matte,
                                        &error),
                "Maxine key-light matte request should reuse compatible matte");
  ok &= Require(matte_runs == 1,
                "compatible Maxine matte consumers should share producer run");
  ok &= Require(virtual_background_matte != nullptr &&
                    key_light_matte != nullptr &&
                    virtual_background_matte == key_light_matte,
                "compatible Maxine requests should return the cached artifact");
  return ok;
}

bool TestFrameArtifactCacheSeparatesIncompatibleMatteKeys() {
  studiocast::open_video::FrameArtifactCache cache;
  int matte_runs = 0;
  std::string error;
  const studiocast::open_video::FrameMatteArtifact *artifact = nullptr;

  const auto producer =
      [&](studiocast::open_video::FrameMatteArtifact *out,
          std::string *producer_error) {
        if (producer_error)
          producer_error->clear();
        ++matte_runs;
        out->key = (matte_runs == 1) ? MatteKey("model-a") : MatteKey("model-b");
        out->cpu_alpha = {static_cast<float>(matte_runs)};
        return true;
      };

  bool ok = true;
  ok &= Require(cache.GetOrComputeMatte(11, MatteKey("model-a"), producer,
                                        &artifact, &error),
                "first matte request should succeed");
  ok &= Require(cache.GetOrComputeMatte(11, MatteKey("model-b"), producer,
                                        &artifact, &error),
                "incompatible matte request should recompute");
  ok &= Require(matte_runs == 2,
                "different matte keys should not reuse stale artifacts");
  if (artifact) {
    ok &= Require(artifact->key.model_id == "model-b",
                  "cache should retain the latest incompatible matte");
  }
  return ok;
}

bool TestFrameArtifactCacheInvalidatesMatteOnNewFrame() {
  studiocast::open_video::FrameAnalysisCache cache;
  int matte_runs = 0;
  std::string error;
  const studiocast::open_video::FrameMatteArtifact *artifact = nullptr;

  const auto producer =
      [&](studiocast::open_video::FrameMatteArtifact *out,
          std::string *producer_error) {
        if (producer_error)
          producer_error->clear();
        ++matte_runs;
        out->key = MatteKey("rvm-test");
        out->cpu_alpha = {static_cast<float>(matte_runs)};
        return true;
      };

  bool ok = true;
  cache.BeginFrame(21);
  ok &= Require(cache.artifacts.GetOrComputeMatte(21, MatteKey("rvm-test"),
                                                  producer, &artifact, &error),
                "first frame matte request should succeed");
  cache.BeginFrame(22);
  ok &= Require(cache.artifacts.GetOrComputeMatte(22, MatteKey("rvm-test"),
                                                  producer, &artifact, &error),
                "new frame matte request should succeed");
  ok &= Require(matte_runs == 2,
                "matte artifacts must be invalidated across capture frames");
  if (artifact) {
    ok &= Require(artifact->cpu_alpha.size() == 1 &&
                      artifact->cpu_alpha.front() == 2.0f,
                  "new frame should expose newly produced matte payload");
  }
  return ok;
}

bool TestFrameArtifactCachePrecomputedMatteKeysPreserveCompatibility() {
  studiocast::open_video::FrameArtifactCache cache;
  auto cuda_key = MatteKey("rvm-test",
                           studiocast::open_video::FrameMatteStorage::
                               cuda_f32_alpha,
                           "open_cuda");
  cuda_key.stream = 0x1234u;

  auto cpu_key = cuda_key;
  cpu_key.storage =
      studiocast::open_video::FrameMatteStorage::cpu_f32_alpha;
  auto vulkan_key = cuda_key;
  vulkan_key.provider_id = "open_vulkan";
  vulkan_key.storage =
      studiocast::open_video::FrameMatteStorage::vulkan_f32_alpha;
  vulkan_key.stream = 0;

  studiocast::open_video::FrameMatteArtifact artifact;
  artifact.key = cuda_key;
  artifact.handle = 0xCAFEu;
  (void)cache.StoreMatte(31, std::move(artifact));

  auto other_provider = cuda_key;
  other_provider.provider_id = "maxine";
  auto other_model = cuda_key;
  other_model.model_id = "rvm-next";
  auto other_frame_size = cuda_key;
  other_frame_size.frame_width = 1920;
  auto other_stream = cuda_key;
  other_stream.stream = 0x5678u;

  bool ok = true;
  ok &= Require(cache.FindMatte(31, cuda_key) != nullptr,
                "precomputed CUDA matte key should find stored artifact");
  ok &= Require(cache.FindMatte(31, cpu_key) == nullptr,
                "precomputed CPU/CUDA storage variants must stay distinct");
  ok &= Require(cache.FindMatte(31, vulkan_key) == nullptr,
                "precomputed CUDA/Vulkan storage variants must stay distinct");

  studiocast::open_video::FrameMatteArtifact vk_artifact;
  vk_artifact.key = vulkan_key;
  vk_artifact.handle = 0xF00Du;
  (void)cache.StoreMatte(31, std::move(vk_artifact));
  ok &= Require(cache.FindMatte(31, vulkan_key) != nullptr,
                "precomputed Vulkan matte key should find stored artifact");
  ok &= Require(cache.FindMatte(31, cuda_key) != nullptr,
                "storing a Vulkan matte should not replace CUDA storage");
  ok &= Require(cache.FindMatte(31, other_provider) == nullptr,
                "provider changes must not reuse matte artifacts");
  ok &= Require(cache.FindMatte(31, other_model) == nullptr,
                "model changes must not reuse matte artifacts");
  ok &= Require(cache.FindMatte(31, other_frame_size) == nullptr,
                "frame size changes must not reuse matte artifacts");
  ok &= Require(cache.FindMatte(31, other_stream) == nullptr,
                "stream changes must not reuse device-local matte artifacts");
  return ok;
}

} // namespace studiocast::tests
