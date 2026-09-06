#pragma once

#include "core/video/video_execution_plan.h"

namespace studiocast::open_cuda {

inline bool AcceptsOpenCudaSelectionIntent(
    const video::NormalizedVideoExecutionIntent &intent) noexcept {
  const bool provider_unbound =
      intent.requested_provider == hal::SemanticEffectProviderId::invalid &&
      !intent.strict_provider_selection;
  const bool automatic =
      intent.backend.mode == hal::BackendRequestMode::automatic &&
      intent.backend.backend == hal::BackendId::invalid &&
      !intent.backend.strict &&
      intent.preferences.compute_family == hal::ComputeFamilyAllowance::any &&
      intent.preferences.effects_engine ==
          hal::EffectsEnginePreferenceConstraint::automatic &&
      !intent.preferences.strict_effects_engine && provider_unbound;
  const bool strict_open_cuda =
      intent.backend.mode == hal::BackendRequestMode::explicit_backend &&
      intent.backend.backend == hal::BackendId::cuda &&
      intent.backend.strict &&
      intent.preferences.compute_family ==
          hal::ComputeFamilyAllowance::cuda_family &&
      intent.preferences.effects_engine ==
          hal::EffectsEnginePreferenceConstraint::open_cuda &&
      intent.preferences.strict_effects_engine &&
      intent.requested_provider ==
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda &&
      intent.strict_provider_selection;
  const bool forced_cuda_with_automatic_engine =
      intent.backend.mode == hal::BackendRequestMode::explicit_backend &&
      intent.backend.backend == hal::BackendId::cuda &&
      intent.backend.strict &&
      intent.preferences.compute_family ==
          hal::ComputeFamilyAllowance::cuda_family &&
      intent.preferences.effects_engine ==
          hal::EffectsEnginePreferenceConstraint::automatic &&
      !intent.preferences.strict_effects_engine && provider_unbound;
  return automatic || strict_open_cuda || forced_cuda_with_automatic_engine;
}

} // namespace studiocast::open_cuda
