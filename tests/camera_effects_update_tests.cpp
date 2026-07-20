#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "core/video/camera_pipeline.h"

namespace studiocast::video::detail {

struct CameraPipelineIssue1TestAccess {
  struct Snapshot {
    effects::BroadcastCameraEffects effects;
    std::uint64_t generation = 0;
    std::uint64_t set_requests = 0;
    std::uint64_t ignored_updates = 0;
    std::uint64_t applied_updates = 0;
    std::uint64_t resource_refreshes = 0;
    std::uint64_t replacement_preparations = 0;
  };

  static Snapshot Get(const CameraPipeline &pipeline) {
    std::lock_guard<std::mutex> lock(pipeline.effects_mu_);
    return Snapshot{
        .effects = pipeline.effects_,
        .generation = pipeline.effects_generation_,
        .set_requests = pipeline.effects_set_requests_,
        .ignored_updates = pipeline.effects_ignored_updates_,
        .applied_updates = pipeline.effects_applied_updates_,
        .resource_refreshes = pipeline.effects_resource_refreshes_,
        .replacement_preparations =
            pipeline.replacement_background_preparations_,
    };
  }
};

} // namespace studiocast::video::detail

namespace {

using studiocast::video::CameraPipeline;
using studiocast::video::EffectsPlanRequiresRebuild;
using studiocast::video::OptionalEffectBreaker;
using studiocast::video::detail::CameraPipelineIssue1TestAccess;
using studiocast::video::effects::BroadcastCameraEffects;
using studiocast::video::effects::VirtualBackgroundMode;

class ScopedReplaceImage final {
public:
  ScopedReplaceImage() {
    const auto suffix =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("studiocast-effects-update-" + std::to_string(suffix) + ".ppm");
    std::ofstream out(path_, std::ios::binary);
    out << "P6\n1 1\n255\n";
    const char pixel[3] = {'\0', '\0', '\0'};
    out.write(pixel, sizeof(pixel));
    valid_ = out.good();
  }

  ~ScopedReplaceImage() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  bool valid() const { return valid_; }
  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
  bool valid_ = false;
};

bool Expect(bool condition, const std::string &message) {
  if (!condition)
    std::cerr << message << "\n";
  return condition;
}

bool TestIdenticalPipelineEffectsPreserveRuntimeGeneration() {
  ScopedReplaceImage replace_image;
  if (!Expect(replace_image.valid(), "failed to create replacement fixture"))
    return false;

  CameraPipeline pipeline;
  BroadcastCameraEffects effects;
  effects.virtual_background.mode = VirtualBackgroundMode::replace;
  effects.virtual_background.replace_path = replace_image.path().string();
  effects.auto_frame.enabled = true;

  pipeline.SetEffects(effects);
  auto applied = CameraPipelineIssue1TestAccess::Get(pipeline);
  if (!Expect(applied.generation == 1 && applied.applied_updates == 1,
              "first real effect change did not publish one generation") ||
      !Expect(applied.replacement_preparations == 1,
              "replacement source was not prepared exactly once")) {
    return false;
  }

  OptionalEffectBreaker breaker;
  breaker.OnFailure("auto_frame", "open_video", "synthetic failure", 10, 1);
  int yunet_session_identity = 17;
  int temporal_state_identity = 23;
  int rebuilds = 0;
  int yunet_resets = 0;
  int temporal_resets = 0;
  int breaker_resets = 0;

  const auto observe_generation = [&]() {
    const auto current = CameraPipelineIssue1TestAccess::Get(pipeline);
    if (!EffectsPlanRequiresRebuild(applied.effects, applied.generation,
                                    current.effects, current.generation)) {
      return;
    }
    ++rebuilds;
    ++yunet_resets;
    ++temporal_resets;
    ++breaker_resets;
    yunet_session_identity = 0;
    temporal_state_identity = 0;
    breaker.Reset();
    applied = current;
  };

  for (int poll = 0; poll < 8; ++poll) {
    pipeline.SetEffects(effects);
    observe_generation();
  }

  const auto stable = CameraPipelineIssue1TestAccess::Get(pipeline);
  if (!Expect(stable.generation == 1,
              "identical SetEffects advanced the generation") ||
      !Expect(stable.set_requests == 9 && stable.ignored_updates == 8 &&
                  stable.applied_updates == 1,
              "stable SetEffects counters did not prove idempotence") ||
      !Expect(stable.replacement_preparations == 1,
              "identical replacement configuration repeated preparation") ||
      !Expect(rebuilds == 0 && yunet_resets == 0 && temporal_resets == 0 &&
                  breaker_resets == 0,
              "stable generation crossed a runtime invalidation boundary") ||
      !Expect(yunet_session_identity == 17 && temporal_state_identity == 23,
              "stable generation discarded YuNet or temporal identity") ||
      !Expect(breaker.active() && breaker.failure_count() == 1,
              "stable generation reset circuit-breaker cooldown")) {
    return false;
  }

  // An unrelated genuine effect change must publish one new generation while
  // reusing the unchanged replacement source preparation.
  effects.vignette.enabled = true;
  pipeline.SetEffects(effects);
  observe_generation();
  pipeline.SetEffects(effects);
  observe_generation();

  const auto changed = CameraPipelineIssue1TestAccess::Get(pipeline);
  return Expect(changed.generation == 2 && changed.applied_updates == 2,
                "real effect change did not publish exactly one generation") &&
         Expect(changed.replacement_preparations == 1,
                "unrelated effect change re-statted replacement source") &&
         Expect(rebuilds == 1 && yunet_resets == 1 && temporal_resets == 1 &&
                    breaker_resets == 1,
                "real generation did not produce exactly one invalidation") &&
         Expect(!breaker.active(),
                "real generation did not reset the simulated breaker boundary");
}

bool TestExplicitEffectsResourceRefreshIsSeparateInvalidation() {
  ScopedReplaceImage replace_image;
  if (!Expect(replace_image.valid(), "failed to create replacement fixture"))
    return false;

  CameraPipeline pipeline;
  BroadcastCameraEffects effects;
  effects.virtual_background.mode = VirtualBackgroundMode::replace;
  effects.virtual_background.replace_path = replace_image.path().string();
  pipeline.SetEffects(effects);
  const auto before = CameraPipelineIssue1TestAccess::Get(pipeline);

  pipeline.RefreshEffectsResources();
  const auto refreshed = CameraPipelineIssue1TestAccess::Get(pipeline);
  if (!Expect(refreshed.effects == before.effects,
              "explicit refresh changed canonical effects") ||
      !Expect(refreshed.generation == before.generation + 1,
              "explicit refresh did not create one generation boundary") ||
      !Expect(refreshed.resource_refreshes == 1,
              "explicit refresh counter was not recorded") ||
      !Expect(refreshed.replacement_preparations ==
                  before.replacement_preparations + 1,
              "explicit refresh did not revalidate replacement source")) {
    return false;
  }

  pipeline.SetEffects(effects);
  const auto stable = CameraPipelineIssue1TestAccess::Get(pipeline);
  return Expect(stable.generation == refreshed.generation,
                "identical SetEffects repeated an explicit refresh") &&
         Expect(stable.replacement_preparations ==
                    refreshed.replacement_preparations,
                "post-refresh stable update repeated replacement stat");
}

} // namespace

namespace studiocast::tests {

bool TestCameraPipelineIdenticalEffectsPreserveRuntimeGeneration() {
  return TestIdenticalPipelineEffectsPreserveRuntimeGeneration();
}

bool TestCameraPipelineExplicitEffectsRefreshIsSeparateInvalidation() {
  return TestExplicitEffectsResourceRefreshIsSeparateInvalidation();
}

} // namespace studiocast::tests
