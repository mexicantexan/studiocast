#include "core/video/effects/broadcast_effect_rules.h"

#include <string_view>

#include "core/video/effects/broadcast_effect_contract.h"

namespace studiocast::video::effects {
namespace {

inline bool VignetteEffective(const BroadcastCameraEffects &fx) {
  return fx.vignette.enabled && fx.vignette.intensity > 0;
}

inline std::string VirtualBackgroundEffectId(const BroadcastCameraEffects &fx) {
  using contract::kEffectIdVirtualBackgroundBlur;
  using contract::kEffectIdVirtualBackgroundRemove;
  using contract::kEffectIdVirtualBackgroundReplace;

  switch (fx.virtual_background.mode) {
  case VirtualBackgroundMode::blur:
    return std::string(kEffectIdVirtualBackgroundBlur);
  case VirtualBackgroundMode::remove:
    return std::string(kEffectIdVirtualBackgroundRemove);
  case VirtualBackgroundMode::replace:
    return std::string(kEffectIdVirtualBackgroundReplace);
  case VirtualBackgroundMode::none:
  default:
    return {};
  }
}

inline bool IsGpuStageEffectId(std::string_view id) {
  // "GPU stage" here means a stage that may host vignette attachment.
  using namespace contract;

  if (id == kEffectIdEyeContact)
    return true;
  if (id == kEffectIdVirtualKeyLight)
    return true;
  if (id == kEffectIdAutoFrame)
    return true;
  if (id == kEffectIdVirtualBackgroundBlur)
    return true;
  if (id == kEffectIdVirtualBackgroundRemove)
    return true;
  if (id == kEffectIdVirtualBackgroundReplace)
    return true;
  return false;
}

} // namespace

BroadcastEffectsPlan
BuildBroadcastEffectsPlan(const BroadcastCameraEffects &fx) {
  BroadcastEffectsPlan plan;

  // ---- Requested flags ----
  bool enable_noise_removal = fx.video_noise_removal.enabled;
  bool enable_eye_contact = fx.eye_contact.enabled;
  bool enable_key_light = fx.virtual_key_light.enabled;
  bool enable_auto_frame = fx.auto_frame.enabled;
  bool enable_vignette = VignetteEffective(fx);
  const bool enable_mirror = fx.mirror;

  std::string vb_id = VirtualBackgroundEffectId(fx);
  bool enable_virtual_background = !vb_id.empty();

  // ---- Dependency / compatibility rules ----
  // 1) VB replace requires a replace path.
  if (enable_virtual_background &&
      fx.virtual_background.mode == VirtualBackgroundMode::replace &&
      fx.virtual_background.replace_path.empty()) {
    plan.disabled.push_back(DisabledEffectByRule{
        .id = std::string(contract::kEffectIdVirtualBackgroundReplace),
        .reason =
            "Disabled: virtual background replace requires `replace_path`."});
    enable_virtual_background = false;
    vb_id.clear();
  }

  // ---- Ordering rules ----
  // Rationale (high-level):
  //  - Noise removal early improves subsequent stages.
  //  - Eye Contact operates on facial features; run before background/key-light
  //  composites.
  //  - Virtual Background and Key Light both depend on segmentation; run before
  //    framing. Prefer Virtual Background before Key Light so the matting
  //    output can be reused (avoids redundant matting work and extra transfers
  //    on the Open CUDA path).
  //  - Auto Frame last so it frames the final composite.
  //  - Vignette after framing.

  if (enable_noise_removal) {
    plan.ordered_effect_ids.push_back(
        std::string(contract::kEffectIdVideoNoiseRemoval));
  }

  if (enable_eye_contact) {
    plan.ordered_effect_ids.push_back(
        std::string(contract::kEffectIdEyeContact));
  }

  if (enable_virtual_background) {
    plan.ordered_effect_ids.push_back(vb_id);
  }

  if (enable_key_light) {
    plan.ordered_effect_ids.push_back(
        std::string(contract::kEffectIdVirtualKeyLight));
  }

  if (enable_auto_frame) {
    plan.ordered_effect_ids.push_back(
        std::string(contract::kEffectIdAutoFrame));
  }

  if (enable_vignette) {
    plan.ordered_effect_ids.push_back(std::string(contract::kEffectIdVignette));
  }

  // Mirror is the final visual transform. Face/subject analysis and every
  // other effect must consume the unmirrored frame, and the live Vulkan path
  // applies this stage after any output resize.
  if (enable_mirror) {
    plan.ordered_effect_ids.push_back(std::string(contract::kEffectIdMirror));
  }

  // Decide vignette attachment.
  if (enable_vignette) {
    // Attach to the last enabled GPU stage (excluding vignette itself).
    for (auto it = plan.ordered_effect_ids.rbegin();
         it != plan.ordered_effect_ids.rend(); ++it) {
      const std::string &id = *it;
      if (id == contract::kEffectIdVignette)
        continue;
      if (IsGpuStageEffectId(id)) {
        plan.vignette_attach_to_effect_id = id;
        break;
      }
    }
  }

  return plan;
}

bool BroadcastEffectsPlanRequestsCompute(const BroadcastEffectsPlan &plan) {
  using namespace contract;
  for (const std::string &id : plan.ordered_effect_ids) {
    if (id == kEffectIdMirror || id == kEffectIdVideoNoiseRemoval ||
        id == kEffectIdEyeContact || id == kEffectIdVirtualBackgroundBlur ||
        id == kEffectIdVirtualBackgroundRemove ||
        id == kEffectIdVirtualBackgroundReplace || id == kEffectIdAutoFrame ||
        id == kEffectIdVirtualKeyLight || id == kEffectIdVignette) {
      return true;
    }
  }
  return false;
}

} // namespace studiocast::video::effects
