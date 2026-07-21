#include "core/video/camera_pipeline.h"
#include "core/video/effects/broadcast_effect_contract.h"
#include "core/video/effects/broadcast_effect_rules.h"
#include "core/video/v4l2_capture.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>
#include <tuple>
#include <utility>

namespace {

std::atomic<std::size_t> g_allocations{0};
std::atomic<bool> g_count_allocations{false};

bool Require(bool condition, const char *message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

studiocast::video::effects::BroadcastCameraEffects AllEffects() {
  studiocast::video::effects::BroadcastCameraEffects effects;
  effects.video_noise_removal.enabled = true;
  effects.eye_contact.enabled = true;
  effects.virtual_background.mode =
      studiocast::video::effects::VirtualBackgroundMode::blur;
  effects.virtual_key_light.enabled = true;
  effects.auto_frame.enabled = true;
  effects.vignette.enabled = true;
  effects.mirror = true;
  return effects;
}

bool TestCanonicalStageMappingAndOrder() {
  namespace effects = studiocast::video::effects;
  using Stage = effects::BroadcastEffectStage;

  constexpr std::array<Stage, effects::kBroadcastEffectStageCount> stages = {
      Stage::video_noise_removal,
      Stage::eye_contact,
      Stage::virtual_background_blur,
      Stage::virtual_background_remove,
      Stage::virtual_background_replace,
      Stage::virtual_key_light,
      Stage::auto_frame,
      Stage::vignette,
      Stage::mirror,
  };
  bool ok = true;
  for (const Stage stage : stages) {
    const auto id = effects::BroadcastEffectStageId(stage);
    ok &= Require(!id.empty(), "canonical stage must have an ID");
    ok &= Require(effects::BroadcastEffectStageFromId(id) == stage,
                  "stage ID round trip changed");
  }
  ok &= Require(effects::BroadcastEffectStageFromId("unknown") == Stage::none,
                "unknown stage ID must fail closed");

  const auto prepared = effects::CompileBroadcastEffectsFramePlan(
      effects::BuildBroadcastEffectsPlan(AllEffects()));
  constexpr std::array<Stage, 7> expected = {
      Stage::video_noise_removal, Stage::eye_contact,
      Stage::virtual_background_blur, Stage::virtual_key_light,
      Stage::auto_frame, Stage::vignette, Stage::mirror,
  };
  ok &= Require(prepared.valid, "canonical all-effects plan did not compile");
  ok &= Require(prepared.size == expected.size(),
                "compiled all-effects stage count changed");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    ok &= Require(prepared.StageAt(index) == expected[index],
                  "compiled canonical stage order changed");
    ok &= Require(prepared.Contains(expected[index]),
                  "compiled plan lost an enabled stage");
    if (index + 1 < expected.size()) {
      ok &= Require(prepared.AppearsAfter(expected[index], expected[index + 1]),
                    "compiled position lookup changed");
    }
  }
  ok &= Require(prepared.vignette_attachment == Stage::auto_frame,
                "vignette attachment was not compiled");
  ok &= Require(prepared.last_deferred_stage == Stage::auto_frame,
                "last deferred stage changed");
  ok &= Require(prepared.StageAt(prepared.size) == Stage::none,
                "out-of-range stage lookup must fail closed");

  constexpr std::array<std::pair<effects::VirtualBackgroundMode, Stage>, 3>
      background_variants = {{
          {effects::VirtualBackgroundMode::blur,
           Stage::virtual_background_blur},
          {effects::VirtualBackgroundMode::remove,
           Stage::virtual_background_remove},
          {effects::VirtualBackgroundMode::replace,
           Stage::virtual_background_replace},
      }};
  for (const auto &[mode, expected_stage] : background_variants) {
    effects::BroadcastCameraEffects background_only;
    background_only.virtual_background.mode = mode;
    if (mode == effects::VirtualBackgroundMode::replace)
      background_only.virtual_background.replace_path = "/prepared/source.ppm";
    const auto variant = effects::CompileBroadcastEffectsFramePlan(
        effects::BuildBroadcastEffectsPlan(background_only));
    ok &= Require(variant.valid && variant.size == 1 &&
                      variant.StageAt(0) == expected_stage &&
                      variant.last_deferred_stage == expected_stage,
                  "background variant membership/defer metadata changed");
  }
  return ok;
}

bool TestPreparedMaxineResidentIslandsFollowResolvedBackends() {
  namespace effects = studiocast::video::effects;
  using BroadcastStage = effects::BroadcastEffectStage;
  using ResidentStage = studiocast::maxine::ResidentStageKind;

  auto all_maxine_effects = AllEffects();
  all_maxine_effects.mirror = false;
  const auto all_plan = effects::CompileBroadcastEffectsFramePlan(
      effects::BuildBroadcastEffectsPlan(all_maxine_effects));
  std::uint32_t all_mask = 0;
  for (std::size_t i = 0; i < all_plan.size; ++i)
    all_mask |= studiocast::video::MaxineResolvedStageBit(all_plan.StageAt(i));
  const auto all = studiocast::video::CompileMaxineResidentIslands(
      all_plan, all_mask, 0xCAFE);
  bool ok = Require(all.valid && all.count == 1,
                    "all-Maxine stages must compile into one island");
  ok &= Require(all.islands[0].first_frame_stage == 0 &&
                    all.islands[0].end_frame_stage == all_plan.size &&
                    !all.islands[0].plan.has_cpu_tail,
                "terminal all-Maxine island must use final readback");
  ok &= Require(all.islands[0].plan.stage_count == all_plan.size,
                "all-Maxine island lost a stage");
  ok &= Require(all.islands[0].plan.stages[2].kind ==
                        ResidentStage::background_blur &&
                    all.islands[0].plan.stages[2].requires_shared_matte &&
                    all.islands[0].plan.stages[3].kind ==
                        ResidentStage::relighting &&
                    all.islands[0].plan.stages[3].matte_fingerprint == 0xCAFE,
                "shared-matte stages must retain the setup fingerprint");

  const std::uint32_t mixed_mask =
      studiocast::video::MaxineResolvedStageBit(
          BroadcastStage::video_noise_removal) |
      studiocast::video::MaxineResolvedStageBit(
          BroadcastStage::virtual_background_blur) |
      studiocast::video::MaxineResolvedStageBit(
          BroadcastStage::virtual_key_light) |
      studiocast::video::MaxineResolvedStageBit(BroadcastStage::vignette);
  const auto mixed = studiocast::video::CompileMaxineResidentIslands(
      all_plan, mixed_mask, 7);
  ok &= Require(mixed.valid && mixed.count == 3,
                "mixed backends must compile maximal bounded islands");
  ok &= Require(mixed.islands[0].first_frame_stage == 0 &&
                    mixed.islands[0].end_frame_stage == 1 &&
                    mixed.islands[0].plan.has_cpu_tail,
                "first mixed island must expose its continuation boundary");
  ok &= Require(mixed.islands[1].first_frame_stage == 2 &&
                    mixed.islands[1].end_frame_stage == 4 &&
                    mixed.islands[1].plan.has_cpu_tail,
                "adjacent background/key-light stages must share an island");
  ok &= Require(mixed.islands[2].first_frame_stage == 5 &&
                    mixed.islands[2].end_frame_stage == 6 &&
                    !mixed.islands[2].plan.has_cpu_tail,
                "terminal vignette island must use final readback");

  const auto none = studiocast::video::CompileMaxineResidentIslands(
      all_plan, 0, 1);
  ok &= Require(none.valid && none.count == 0 &&
                    none.enabled_resident_stage_mask == 0,
                "no resolved Maxine stage must compile no executor work");

  for (const auto &[mode, broadcast, resident] :
       std::array{
           std::tuple{effects::VirtualBackgroundMode::remove,
                      BroadcastStage::virtual_background_remove,
                      ResidentStage::background_remove},
           std::tuple{effects::VirtualBackgroundMode::replace,
                      BroadcastStage::virtual_background_replace,
                      ResidentStage::background_replace}}) {
    effects::BroadcastCameraEffects background;
    background.virtual_background.mode = mode;
    if (mode == effects::VirtualBackgroundMode::replace)
      background.virtual_background.replace_path = "/prepared/source.ppm";
    const auto frame = effects::CompileBroadcastEffectsFramePlan(
        effects::BuildBroadcastEffectsPlan(background));
    const auto compiled = studiocast::video::CompileMaxineResidentIslands(
        frame, studiocast::video::MaxineResolvedStageBit(broadcast), 9);
    ok &= Require(compiled.valid && compiled.count == 1 &&
                      compiled.islands[0].plan.stage_count == 1 &&
                      compiled.islands[0].plan.stages[0].kind == resident &&
                      compiled.islands[0].plan.stages[0].requires_shared_matte,
                  "background remove/replace resident mapping changed");
  }
  return ok;
}

bool TestProductionMaxineTelemetryFeedsCameraRollup() {
  using Cpu = studiocast::maxine::ProductionCpuStageKind;
  using Stage = studiocast::maxine::ResidentStageKind;
  using Transfer = studiocast::maxine::ProductionTransferKind;
  studiocast::maxine::ProductionResidentTelemetry previous{};
  studiocast::maxine::ProductionResidentTelemetry current{};
  const auto transfer_index = [](Transfer kind) {
    return static_cast<std::size_t>(kind);
  };
  const auto upload = transfer_index(Transfer::host_upload);
  const auto final_download = transfer_index(Transfer::final_download);
  const auto continuation =
      transfer_index(Transfer::cpu_continuation_download);
  previous.transfers[upload] = {4, 3};
  current.transfers[upload] = {6, 5};
  previous.transfers[final_download] = {8, 7};
  current.transfers[final_download] = {9, 8};
  previous.transfers[continuation] = {2, 2};
  current.transfers[continuation] = {3, 3};
  const auto bridge = transfer_index(Transfer::device_format_bridge);
  previous.transfers[bridge] = {6, 5};
  current.transfers[bridge] = {9, 7};
  previous.matte_inferences = {10, 9};
  current.matte_inferences = {12, 10};
  previous.shared_matte_reuses = 5;
  current.shared_matte_reuses = 7;
  previous.explicit_synchronizations = {20, 19};
  current.explicit_synchronizations = {22, 21};
  previous.composites = {3, 2};
  current.composites = {5, 4};
  previous.synchronous_sdk_runs = {50, 49};
  current.synchronous_sdk_runs = {53, 51};
  previous.asynchronous_sdk_runs = {60, 59};
  current.asynchronous_sdk_runs = {62, 61};
  const auto rgb = static_cast<std::size_t>(Cpu::rgb_to_bgr_staging);
  previous.cpu_stages[rgb] = {30, 29};
  current.cpu_stages[rgb] = {32, 31};
  const auto blur = static_cast<std::size_t>(Stage::background_blur);
  previous.stages[blur] = {40, 39};
  current.stages[blur] = {43, 41};

  const auto delta = studiocast::video::ComputeMaxineResidentTelemetryDelta(
      current, previous);
  bool ok = Require(delta.upload_attempts == 2 && delta.upload_calls == 2,
                    "camera rollup lost actual upload attempt/success deltas");
  ok &= Require(delta.final_download_attempts == 1 &&
                    delta.final_download_calls == 1 &&
                    delta.cpu_continuation_download_attempts == 1 &&
                    delta.cpu_continuation_download_calls == 1,
                "camera rollup lost named readback boundaries");
  ok &= Require(delta.matte_inference_attempts == 2 &&
                    delta.matte_inference_calls == 1 &&
                    delta.shared_matte_reuses == 2,
                "camera rollup lost matte attempts, successes, or reuse");
  ok &= Require(delta.sync_attempts == 2 && delta.sync_calls == 2 &&
                    delta.rgb_to_bgr_calls == 2,
                "camera rollup lost synchronization or staging callsites");
  ok &= Require(delta.device_bridge_attempts == 3 &&
                    delta.device_bridge_calls == 2 &&
                    delta.composite_attempts == 2 &&
                    delta.composite_calls == 2,
                "camera rollup lost extra device transfer/composite calls");
  ok &= Require(delta.synchronous_sdk_run_attempts == 3 &&
                    delta.synchronous_sdk_run_calls == 2 &&
                    delta.asynchronous_sdk_run_attempts == 2 &&
                    delta.asynchronous_sdk_run_calls == 2,
                "camera rollup lost synchronous/asynchronous SDK calls");
  ok &= Require(delta.stage_attempts[blur] == 3 &&
                    delta.stage_successes[blur] == 2,
                "camera rollup lost named resident stage attribution");

  const auto reset_delta =
      studiocast::video::ComputeMaxineResidentTelemetryDelta(previous, current);
  ok &= Require(reset_delta.upload_attempts == 0 &&
                    reset_delta.stage_attempts[blur] == 0,
                "counter reset must fail closed instead of underflowing");
  return ok;
}

bool TestCompiledPlanTracksSetupMutationsAndRejectsStaleInput() {
  namespace effects = studiocast::video::effects;
  using Stage = effects::BroadcastEffectStage;

  auto plan = effects::BuildBroadcastEffectsPlan(AllEffects());
  const auto eye_compatibility =
      studiocast::video::ApplyOpenVulkanEyeContactPlanCompatibility(&plan);
  const auto denoise_compatibility =
      studiocast::video::ApplyOpenVulkanVideoNoiseRemovalPlanCompatibility(
          &plan);
  const auto prepared = effects::CompileBroadcastEffectsFramePlan(plan);
  bool ok = Require(eye_compatibility.blocked && denoise_compatibility.blocked,
                    "availability mutation precondition failed");
  ok &= Require(prepared.valid,
                "availability-mutated setup plan did not compile");
  ok &= Require(!prepared.Contains(Stage::eye_contact) &&
                    !prepared.Contains(Stage::video_noise_removal),
                "compiled plan retained unavailable stages");
  ok &= Require(prepared.Contains(Stage::virtual_background_blur) &&
                    prepared.Contains(Stage::virtual_key_light) &&
                    prepared.Contains(Stage::auto_frame) &&
                    prepared.Contains(Stage::vignette) &&
                    prepared.Contains(Stage::mirror),
                "compiled plan lost an unrelated available stage");
  ok &= Require(prepared.vignette_attachment == Stage::auto_frame &&
                    prepared.last_deferred_stage == Stage::auto_frame,
                "compiled attachment/defer metadata ignored setup mutation");

  effects::BroadcastEffectsPlan invalid;
  invalid.ordered_effect_ids.emplace_back(effects::contract::kEffectIdMirror);
  invalid.ordered_effect_ids.emplace_back(effects::contract::kEffectIdMirror);
  invalid.ordered_effect_ids.emplace_back("unknown-effect");
  invalid.vignette_attach_to_effect_id =
      effects::contract::kEffectIdAutoFrame;
  const auto rejected = effects::CompileBroadcastEffectsFramePlan(invalid);
  ok &= Require(!rejected.valid,
                "duplicate/unknown/stale attachment plan must fail closed");
  return ok;
}

bool TestSteadyFramePlanDecisionAllocatesAndSnapshotsZero() {
  namespace effects = studiocast::video::effects;
  using Stage = effects::BroadcastEffectStage;
  using studiocast::video::EffectsFramePreparationCounters;

  const auto no_effects = effects::CompileBroadcastEffectsFramePlan(
      effects::BuildBroadcastEffectsPlan({}));
  const auto active = effects::CompileBroadcastEffectsFramePlan(
      effects::BuildBroadcastEffectsPlan(AllEffects()));
  if (!Require(no_effects.valid && active.valid,
               "frame-plan warmup failed")) {
    return false;
  }

  EffectsFramePreparationCounters no_effects_counters;
  EffectsFramePreparationCounters active_counters;
  std::uint64_t decision_sink = 0;

  // Warm the same production-connected query/counter operations once.
  (void)no_effects_counters.ObservePublishedGeneration(7, 7);
  (void)active_counters.ObservePublishedGeneration(11, 11);
  (void)active.Contains(Stage::auto_frame);
  (void)active.AppearsAfter(Stage::virtual_background_blur,
                            Stage::virtual_key_light);
  (void)active.StageAt(0);
  no_effects_counters = {};
  active_counters = {};

  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations.store(true, std::memory_order_release);
  for (std::size_t frame = 0; frame < 128; ++frame) {
    decision_sink += no_effects_counters.ObservePublishedGeneration(7, 7);
    decision_sink += no_effects.Contains(Stage::mirror);
    decision_sink += static_cast<std::uint64_t>(no_effects.StageAt(0));
  }
  for (std::size_t frame = 0; frame < 128; ++frame) {
    decision_sink += active_counters.ObservePublishedGeneration(11, 11);
    const Stage stage = active.StageAt(frame % active.size);
    decision_sink += active.Contains(stage);
    decision_sink += active.AppearsAfter(
        Stage::virtual_background_blur, Stage::virtual_key_light);
    decision_sink += effects::BroadcastEffectStageId(stage).size();
  }
  g_count_allocations.store(false, std::memory_order_release);

  const std::size_t measured_allocations =
      g_allocations.load(std::memory_order_relaxed);
  bool ok = Require(measured_allocations == 0,
                    "steady frame-plan dispatch allocated");
  ok &= Require(no_effects_counters.generation_loads == 128 &&
                    active_counters.generation_loads == 128,
                "steady generation decision count changed");
  ok &= Require(no_effects_counters.snapshot_requests == 0 &&
                    no_effects_counters.config_lock_acquisitions == 0 &&
                    no_effects_counters.config_copies == 0 &&
                    no_effects_counters.rebuilds == 0 &&
                    active_counters.snapshot_requests == 0 &&
                    active_counters.config_lock_acquisitions == 0 &&
                    active_counters.config_copies == 0 &&
                    active_counters.rebuilds == 0,
                "steady frame requested/copied/rebuilt configuration");

  EffectsFramePreparationCounters changed;
  ok &= Require(!changed.ObservePublishedGeneration(0, 11),
                "transition sentinel must retain the prepared generation");
  ok &= Require(changed.ObservePublishedGeneration(12, 11),
                "new publication must request one snapshot");
  changed.RecordConfigSnapshot(/*rebuilt=*/true);
  for (std::size_t frame = 0; frame < 128; ++frame)
    decision_sink += changed.ObservePublishedGeneration(12, 12);
  ok &= Require(changed.transition_sentinel_loads == 1 &&
                    changed.snapshot_requests == 1 &&
                    changed.config_lock_acquisitions == 1 &&
                    changed.config_copies == 1 && changed.rebuilds == 1,
                "one generation change must snapshot and rebuild exactly once");

  if (ok) {
    std::cout << "frame_plan_measurement steady_frames_no_effects=128 "
                 "steady_frames_active=128 allocations="
              << measured_allocations
              << " steady_lock_acquisitions=0 steady_config_copies=0 "
                 "generation_change_rebuilds="
              << changed.rebuilds << " decision_sink=" << decision_sink
              << '\n';
  }
  return ok;
}

bool TestCapturePollNoFrameIsDistinctFromFailure() {
  using studiocast::video::CaptureAcquireResult;
  using studiocast::video::ClassifyCapturePollResult;

  bool ok = Require(ClassifyCapturePollResult(0) ==
                        CaptureAcquireResult::no_frame,
                    "zero-timeout poll must classify as expected no-frame");
  ok &= Require(ClassifyCapturePollResult(-1) == CaptureAcquireResult::failure,
                "poll failure must remain distinguishable from no-frame");
  ok &= Require(ClassifyCapturePollResult(1) == CaptureAcquireResult::frame,
                "ready poll must classify as a frame");

  studiocast::video::V4l2Capture closed_capture;
  studiocast::video::CapturedFrameView frame;
  std::string error;
  ok &= Require(closed_capture.AcquireFrameDetailed(&frame, 0, &error) ==
                    CaptureAcquireResult::failure,
                "closed capture must report failure, not no-frame");
  ok &= Require(error == "Capture not open.",
                "typed capture failure lost its detailed diagnostic");
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
  ok &= TestCanonicalStageMappingAndOrder();
  ok &= TestPreparedMaxineResidentIslandsFollowResolvedBackends();
  ok &= TestProductionMaxineTelemetryFeedsCameraRollup();
  ok &= TestCompiledPlanTracksSetupMutationsAndRejectsStaleInput();
  ok &= TestSteadyFramePlanDecisionAllocatesAndSnapshotsZero();
  ok &= TestCapturePollNoFrameIsDistinctFromFailure();
  if (ok)
    std::cout << "Video frame-plan tests passed\n";
  return ok ? 0 : 1;
}
