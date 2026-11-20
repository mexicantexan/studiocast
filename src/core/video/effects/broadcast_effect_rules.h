#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
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

// Allocation-free frame-loop representation compiled from the setup-time
// string plan after backend availability and compatibility mutation is done.
// Keep this enum aligned with the stable canonical effect IDs.
enum class BroadcastEffectStage : std::uint8_t {
  video_noise_removal,
  eye_contact,
  virtual_background_blur,
  virtual_background_remove,
  virtual_background_replace,
  virtual_key_light,
  auto_frame,
  vignette,
  mirror,
  none,
};

inline constexpr std::size_t kBroadcastEffectStageCount = 9;

std::string_view BroadcastEffectStageId(BroadcastEffectStage stage) noexcept;
BroadcastEffectStage
BroadcastEffectStageFromId(std::string_view effect_id) noexcept;

struct PreparedBroadcastEffectsFramePlan {
  PreparedBroadcastEffectsFramePlan() noexcept;

  bool valid = true;
  std::array<BroadcastEffectStage, kBroadcastEffectStageCount> ordered{};
  std::array<std::int8_t, kBroadcastEffectStageCount> positions{};
  std::uint8_t size = 0;
  BroadcastEffectStage vignette_attachment = BroadcastEffectStage::none;
  BroadcastEffectStage last_deferred_stage = BroadcastEffectStage::none;

  bool Contains(BroadcastEffectStage stage) const noexcept;
  bool AppearsAfter(BroadcastEffectStage stage,
                    BroadcastEffectStage later) const noexcept;
  BroadcastEffectStage StageAt(std::size_t index) const noexcept;
};

PreparedBroadcastEffectsFramePlan
CompileBroadcastEffectsFramePlan(const BroadcastEffectsPlan &plan) noexcept;

// Resolves effect ordering and compatibility/dependency rules.
BroadcastEffectsPlan
BuildBroadcastEffectsPlan(const BroadcastCameraEffects &fx);

// Returns true when the canonical plan requests work from the selected video
// compute backend. Mirror is compute work even when it is the only requested
// effect; the live pipeline uses this seam when resolving an explicit Vulkan
// backend and when deciding whether to initialize the shared Vulkan runtime.
bool BroadcastEffectsPlanRequestsCompute(const BroadcastEffectsPlan &plan);

} // namespace studiocast::video::effects
