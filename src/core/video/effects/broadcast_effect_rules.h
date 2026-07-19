#pragma once

#include <string>
#include <vector>

#include "core/video/effects/broadcast_effects.h"

namespace studiocast::video::effects {

// Centralized ordering + compatibility/dependency rules for
// `BroadcastCameraEffects`.
//
// This is the single source of truth for:
//  - pipeline stage ordering (enabled stages in correct order)
//  - rule-based disablement (dependency missing / incompatible)
//  - deterministic reporting of rule decisions in daemon status
//
// Availability (GPU / Maxine presence) is NOT handled here; it is owned by
// MaxineManager diagnostics.

struct DisabledEffectByRule {
  std::string id;     // stable effect ID (see `broadcast_effect_contract.h`)
  std::string reason; // human-friendly reason
};

struct BroadcastEffectsPlan {
  // Enabled effect IDs in pipeline order. IDs are stable contract IDs.
  std::vector<std::string> ordered_effect_ids;

  // Effects that were effectively disabled by local rules.
  std::vector<DisabledEffectByRule> disabled;

  // When vignette is enabled and at least one GPU stage is enabled, the
  // pipeline applies vignette by attaching it to the *last* enabled GPU stage.
  // This string is the effect ID of that stage, or empty to mean "standalone
  // vignette stage".
  std::string vignette_attach_to_effect_id;
};

// Resolves effect ordering and compatibility/dependency rules.
BroadcastEffectsPlan
BuildBroadcastEffectsPlan(const BroadcastCameraEffects &fx);

// Returns true when the canonical plan requests work from the selected video
// compute backend. Mirror is compute work even when it is the only requested
// effect; the live pipeline uses this seam when resolving an explicit Vulkan
// backend and when deciding whether to initialize the shared Vulkan runtime.
bool BroadcastEffectsPlanRequestsCompute(const BroadcastEffectsPlan &plan);

} // namespace studiocast::video::effects
