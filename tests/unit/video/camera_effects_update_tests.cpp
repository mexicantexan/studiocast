#include <cstdint>
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
using studiocast::video::ComputeBackendPreference;
using studiocast::video::EffectsPlanRequiresRebuild;
using studiocast::video::OptionalEffectBreaker;
using studiocast::video::detail::CameraPipelineIssue1TestAccess;
using studiocast::video::effects::BroadcastCameraEffects;
using studiocast::video::effects::VirtualBackgroundMode;

bool Expect(bool condition, const std::string &message) {
  if (!condition)
    std::cerr << message << "\n";
  return condition;
}

bool TestIdenticalPipelineEffectsPreserveRuntimeGeneration() {
  CameraPipeline pipeline;
  BroadcastCameraEffects effects;
  effects.virtual_background.mode = VirtualBackgroundMode::replace;
  effects.virtual_background.replace_path =
      "/must/not/be/scanned/by/camera-pipeline/replacement.png";
  effects.auto_frame.enabled = true;

  pipeline.SetEffects(effects);
  auto applied = CameraPipelineIssue1TestAccess::Get(pipeline);
  if (!Expect(applied.generation == 1 && applied.applied_updates == 1,
              "first real effect change did not publish one generation") ||
      !Expect(applied.replacement_preparations == 0,
              "camera pipeline performed backend-owned replacement scanning")) {
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
      !Expect(stable.replacement_preparations == 0,
              "stable update performed backend-owned replacement scanning") ||
      !Expect(rebuilds == 0 && yunet_resets == 0 && temporal_resets == 0 &&
                  breaker_resets == 0,
              "stable generation crossed a runtime invalidation boundary") ||
      !Expect(yunet_session_identity == 17 && temporal_state_identity == 23,
              "stable generation discarded YuNet or temporal identity") ||
      !Expect(breaker.active() && breaker.failure_count() == 1,
              "stable generation reset circuit-breaker cooldown")) {
    return false;
  }

  // An unrelated genuine effect change publishes one generation. Artifact
  // preparation remains exclusively in the backend setup source.
  effects.vignette.enabled = true;
  pipeline.SetEffects(effects);
  observe_generation();
  pipeline.SetEffects(effects);
  observe_generation();

  const auto changed = CameraPipelineIssue1TestAccess::Get(pipeline);
  return Expect(changed.generation == 2 && changed.applied_updates == 2,
                "real effect change did not publish exactly one generation") &&
         Expect(changed.replacement_preparations == 0,
                "effect change performed pipeline-side replacement scanning") &&
         Expect(rebuilds == 1 && yunet_resets == 1 && temporal_resets == 1 &&
                    breaker_resets == 1,
                "real generation did not produce exactly one invalidation") &&
         Expect(!breaker.active(),
                "real generation did not reset the simulated breaker boundary");
}

bool TestExplicitEffectsResourceRefreshIsSeparateInvalidation() {
  CameraPipeline pipeline;
  BroadcastCameraEffects effects;
  effects.virtual_background.mode = VirtualBackgroundMode::replace;
  effects.virtual_background.replace_path =
      "/must/not/be/scanned/by/camera-pipeline/replacement.png";
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
      !Expect(
          refreshed.replacement_preparations == 0 &&
              before.replacement_preparations == 0,
          "explicit refresh performed pipeline-side replacement scanning")) {
    return false;
  }

  pipeline.SetEffects(effects);
  const auto stable = CameraPipelineIssue1TestAccess::Get(pipeline);
  return Expect(stable.generation == refreshed.generation,
                "identical SetEffects repeated an explicit refresh") &&
         Expect(stable.replacement_preparations == 0,
                "post-refresh update performed replacement scanning");
}

bool TestCameraBackendSelectionPreservesExplicitProviderIntent() {
  namespace hal = studiocast::hal;
  using Engine = studiocast::video::effects::EffectsEnginePreference;

  studiocast::video::NormalizedVideoExecutionIntent intent;
  studiocast::video::ApplyCameraBackendSelectionIntent(
      Engine::open_cuda, ComputeBackendPreference::auto_select, &intent);
  if (!Expect(intent.backend.mode ==
                      hal::BackendRequestMode::explicit_backend &&
                  intent.backend.backend == hal::BackendId::cuda &&
                  intent.backend.strict &&
                  intent.preferences.effects_engine ==
                      hal::EffectsEnginePreferenceConstraint::open_cuda &&
                  intent.preferences.strict_effects_engine &&
                  intent.requested_provider ==
                      hal::SemanticEffectProviderId::invalid &&
                  !intent.strict_provider_selection,
              "OpenCUDA engine intent invented an exact provider request")) {
    return false;
  }

  intent.requested_provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  intent.strict_provider_selection = true;
  studiocast::video::ApplyCameraBackendSelectionIntent(
      Engine::open_cuda, ComputeBackendPreference::vulkan, &intent);
  return Expect(
      intent.backend.backend == hal::BackendId::vulkan &&
          intent.requested_provider ==
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda &&
          intent.strict_provider_selection,
      "backend mapping erased a caller-owned exact provider constraint");
}

bool TestCanonicalProductionRequestPreservesBackendSemantics() {
  namespace video = studiocast::video;
  namespace hal = studiocast::hal;
  BroadcastCameraEffects effects;
  effects.engine = video::effects::EffectsEnginePreference::open_cuda;
  effects.virtual_background.mode = VirtualBackgroundMode::blur;
  const video::CpuPrimitiveTable cpu_table;
  const auto build = [&](ComputeBackendPreference compute) {
    return video::BuildProductionVideoBackendSetupRequest(
        effects, compute, 64, 48, 80, 60, video::PixelFormat::rgb24, 7, 3, 11,
        {}, cpu_table);
  };
  auto request = build(ComputeBackendPreference::cuda);
  bool ok =
      Expect(request.resident.effects.virtual_background.greenscreen_temporal &&
                 request.resident.plan.intent.stages[0]
                     .parameters[3]
                     .boolean_value,
             "default Maxine greenscreen flag was erased") &&
      Expect(!request.resident.plan.temporal.temporal &&
                 request.resident.plan.temporal.history_frames == 0,
             "Maxine-only default greenscreen history blocked CUDA background");
  const auto maxine_temporal = video::ProductionVideoTemporalForSelectedBackend(
      request.resident, hal::BackendId::maxine);
  ok &=
      Expect(maxine_temporal.temporal && maxine_temporal.history_frames == 1 &&
                 maxine_temporal.requires_contiguous_capture_sequence &&
                 maxine_temporal.frame_drop_behavior ==
                     hal::FrameDropBehavior::reset_temporal_state,
             "selected Maxine lost canonical greenscreen temporal history");
  const auto cuda_temporal = video::ProductionVideoTemporalForSelectedBackend(
      request.resident, hal::BackendId::cuda);
  ok &= Expect(!cuda_temporal.temporal && cuda_temporal.history_frames == 0,
               "CUDA source inherited Maxine-only greenscreen history");
  effects.vignette.enabled = true;
  request = build(ComputeBackendPreference::cuda);
  ok &= Expect(!request.resident.plan.intent.stages[1]
                    .intent.requires_tracked_vignette_center &&
                   request.resident.effects.vignette.center_on_tracked_face,
               "vignette without Auto Frame must preserve its fixed-center behavior");
  effects.auto_frame.enabled = true;
  request = build(ComputeBackendPreference::cuda);
  ok &= Expect(request.resident.plan.intent.stages[2]
                   .intent.requires_tracked_vignette_center,
               "Auto Frame tracked vignette intent was silently lowered to fixed center");
  effects.auto_frame.enabled = false;
  effects.mirror = true;
  request = build(ComputeBackendPreference::cuda);
  ok &= Expect(request.resident.effects.mirror &&
                   request.resident.plan.intent.stage_count == 3 &&
                   request.resident.plan.intent.stages[2].intent.semantic ==
                       hal::SemanticEffectId::mirror,
               "CUDA resident combination lost final mirror");
  effects = {};
  effects.vignette.enabled = true;
  request = build(ComputeBackendPreference::vulkan);
  ok &= Expect(request.resident.plan.intent.stage_count == 1 &&
                   !request.resident.plan.intent.stages[0]
                        .intent.requires_tracked_vignette_center &&
                   request.resident.effects.vignette.center_on_tracked_face,
               "default Vulkan vignette without Auto Frame was incorrectly blocked");
  effects = {};
  effects.mirror = true;
  request = build(ComputeBackendPreference::auto_select);
  ok &= Expect(request.kind ==
                   video::ProductionVideoBackendSetupKind::cpu_primitives_only,
               "automatic mirror-only path created a heavy execution island");
  request = build(ComputeBackendPreference::cuda);
  ok &= Expect(
      request.kind == video::ProductionVideoBackendSetupKind::resident_island &&
          request.resident.plan.intent.stage_count == 1 &&
          request.resident.plan.intent.stages[0].intent.semantic ==
              hal::SemanticEffectId::mirror,
      "strict CUDA mirror-only request lost resident ownership");
  effects.engine = video::effects::EffectsEnginePreference::open_cuda;
  request = build(ComputeBackendPreference::cpu);
  ok &=
      Expect(request.kind ==
                 video::ProductionVideoBackendSetupKind::cpu_primitives_only,
             "explicit CPU mirror-only path created a heavy execution island");
  effects.engine = video::effects::EffectsEnginePreference::maxine;
  effects.virtual_background.mode = VirtualBackgroundMode::blur;
  request = build(ComputeBackendPreference::auto_select);
  ok &= Expect(
      !request.resident.effects.mirror &&
          request.resident.plan.intent.stage_count == 1,
      "Maxine resident request lost its established host mirror finalizer");
  effects.engine = video::effects::EffectsEnginePreference::auto_select;
  request = build(ComputeBackendPreference::auto_select);
  ok &= Expect(
      !request.resident.effects.mirror &&
          request.resident.plan.intent.stage_count == 1,
      "automatic selection lost Maxine-compatible host mirror ownership");
  for (int effect = 0; effect < 3; ++effect) {
    effects = {};
    effects.auto_frame.enabled = effect == 0;
    effects.eye_contact.enabled = effect == 1;
    effects.video_noise_removal.enabled = effect == 2;
    request = build(ComputeBackendPreference::cuda);
    ok &= Expect(request.resident.plan.temporal.temporal &&
                     request.resident.plan.temporal
                         .requires_contiguous_capture_sequence &&
                     request.resident.plan.temporal.history_frames ==
                         (effect == 2 ? 5u : 1u),
                 "meaningful CUDA effect history was lost");
  }

  return ok;
}

bool TestCudaNativeMirrorDoesNotInventProviderRequest() {
  namespace video = studiocast::video;
  namespace hal = studiocast::hal;
  BroadcastCameraEffects effects;
  effects.mirror = true;
  video::CameraPipelineHalStatusProjectionInput projection;
  projection.effects = &effects;
  projection.compute_backend = ComputeBackendPreference::cuda;
  projection.lifecycle = video::VideoRuntimeLifecycle::starting;
  projection.status_generation = 1;
  projection.configuration_generation = 1;
  projection.geometry_generation = 1;
  video::MultiIslandVideoRuntimeStatusSnapshot snapshot;
  const auto projected =
      video::BuildCameraPipelineHalStatusSnapshot(projection, &snapshot);
  return Expect(projected.code == video::MultiIslandStatusValidationCode::valid &&
                   snapshot.effects[8].requested_provider ==
                       hal::SemanticEffectProviderId::invalid,
               "strict CUDA native mirror fabricated an ONNX provider request");
}

} // namespace

namespace studiocast::tests {

bool TestCameraPipelineIdenticalEffectsPreserveRuntimeGeneration() {
  return TestIdenticalPipelineEffectsPreserveRuntimeGeneration();
}

bool TestCameraPipelineExplicitEffectsRefreshIsSeparateInvalidation() {
  return TestExplicitEffectsResourceRefreshIsSeparateInvalidation() &&
         TestCameraBackendSelectionPreservesExplicitProviderIntent() &&
         TestCanonicalProductionRequestPreservesBackendSemantics() &&
         TestCudaNativeMirrorDoesNotInventProviderRequest();
}

} // namespace studiocast::tests
