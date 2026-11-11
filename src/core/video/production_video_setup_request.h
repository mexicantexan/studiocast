#pragma once

#include "core/video/production_video_backend_manager.h"
#include "core/video/video_config_types.h"

namespace studiocast::video {

// Applies only backend/engine intent. An independently supplied exact provider
// constraint remains caller-owned and is never rewritten here.
void ApplyCameraBackendSelectionIntent(
    effects::EffectsEnginePreference engine,
    ComputeBackendPreference compute_backend,
    NormalizedVideoExecutionIntent *intent) noexcept;

// The sole allowed backend temporal refinement: selected Maxine matting may
// add its canonical greenscreen history. Existing effect history stays exact.
[[nodiscard]] inline hal::TemporalContract
ProductionVideoTemporalForSelectedBackend(
    const ProductionVideoResidentSetup &request,
    hal::BackendId backend) noexcept {
  auto temporal = request.plan.temporal;
  if (backend != hal::BackendId::maxine || temporal.temporal ||
      !request.effects.virtual_background.greenscreen_temporal ||
      request.plan.intent.stage_count > request.plan.intent.stages.size())
    return temporal;
  for (std::size_t index = 0; index < request.plan.intent.stage_count;
       ++index) {
    const auto semantic = request.plan.intent.stages[index].intent.semantic;
    if (semantic != hal::SemanticEffectId::virtual_background_blur &&
        semantic != hal::SemanticEffectId::virtual_background_remove &&
        semantic != hal::SemanticEffectId::virtual_background_replace &&
        semantic != hal::SemanticEffectId::virtual_key_light)
      continue;
    temporal.temporal = true;
    temporal.requires_contiguous_capture_sequence = true;
    temporal.history_frames = 1;
    temporal.frame_drop_behavior = hal::FrameDropBehavior::reset_temporal_state;
    temporal.discontinuity_action =
        hal::TemporalDiscontinuityAction::reset_and_continue;
    break;
  }
  return temporal;
}

// Shared setup-only camera contract. Hardware smoke tests use this same builder
// so defaults, canonical stage order, and temporal requirements cannot drift.
[[nodiscard]] ProductionVideoBackendSetupRequest
BuildProductionVideoBackendSetupRequest(
    const effects::BroadcastCameraEffects &effects,
    ComputeBackendPreference compute_backend, int input_width, int input_height,
    int output_width, int output_height, PixelFormat output_format,
    std::uint64_t configuration_generation, std::uint64_t geometry_generation,
    std::uint64_t plan_id, const ProductionVideoBoundedWaitPolicy &wait,
    const CpuPrimitiveTable &cpu_table);

} // namespace studiocast::video
