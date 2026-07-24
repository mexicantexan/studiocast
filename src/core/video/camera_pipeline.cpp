#include "camera_pipeline.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../open_video/matting_session.h"
#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_tensor.h"
#include "core/cuda/kernels/open_cuda_vb_kernels.h"
#include "core/cuda/kernels/resize_bilinear.h"
#include "core/maxine/availability.h"
#include "core/maxine/cuda_crop_scale.h"
#include "core/maxine/cuda_driver_api.h"
#include "core/maxine/cuda_vignette.h"
#include "core/maxine/effects/ar_auto_frame_tracker.h"
#include "core/maxine/effects/ar_eye_contact_effect.h"
#include "core/maxine/effects/vfx_background_blur_effect.h"
#include "core/maxine/effects/vfx_denoise_effect.h"
#include "core/maxine/effects/vfx_green_screen_effect.h"
#include "core/maxine/effects/vfx_relighting_effect.h"
#include "core/maxine/maxine_manager.h"
#include "core/maxine/nvcv_api.h"
#include "core/maxine/production_resident_frame_executor.h"
#include "core/maxine/vfx_api.h"
#include "core/open_video/fastdvdnet_denoiser.h"
#include "core/open_video/frame_analysis_cache.h"
#include "core/open_video/gaze_correction_eye_contact.h"
#include "core/open_video/model_pack_registry.h"
#include "core/open_video/yunet_face_detector.h"
#include "core/video/capture_error_policy.h"
#include "core/video/convert.h"
#include "core/video/effects/broadcast_effect_contract.h"
#include "core/video/effects/broadcast_effect_open_cuda_gate.h"
#include "core/video/effects/broadcast_effect_rules.h"
#include "core/video/effects/effect_chain.h"
#include "core/video/image_ppm.h"
#include "core/video/mjpeg_decode.h"
#include "core/video/open_vulkan_matting_setup_policy.h"
#include "core/video/production_video_execution_composition.h"
#include "core/video/replace_background_cache_policy.h"
#include "core/video/scaling_policy.h"
#include "core/video/v4l2loopback.h"
#include "core/video/yuyv_passthrough.h"

#if STUDIOCAST_ENABLE_OPEN_VULKAN
#include "core/open_video/vulkan_matting_session.h"
#include "core/video/open_vulkan_auto_frame.h"
#include "core/video/open_vulkan_final_resident_stage.h"
#include "core/video/open_vulkan_mirror.h"
#include "core/video/open_vulkan_vignette.h"
#include "core/video/open_vulkan_virtual_background_blur.h"
#include "core/video/open_vulkan_virtual_background_remove.h"
#include "core/video/open_vulkan_virtual_background_replace.h"
#include "core/video/open_vulkan_virtual_key_light.h"
#include "core/vulkan/kernels/resize_bilinear.h"
#include "core/vulkan/kernels/utility_kernels.h"
#endif

namespace studiocast::video {

std::string ComputeBackendPreferenceToString(ComputeBackendPreference pref) {
  switch (pref) {
  case ComputeBackendPreference::cpu:
    return "cpu";
  case ComputeBackendPreference::cuda:
    return "cuda";
  case ComputeBackendPreference::vulkan:
    return "vulkan";
  default:
    return "auto";
  }
}

std::string ComputeBackendKindToString(ComputeBackendKind kind) {
  switch (kind) {
  case ComputeBackendKind::cuda:
    return "cuda";
  case ComputeBackendKind::vulkan:
    return "vulkan";
  default:
    return "cpu";
  }
}

bool ParseComputeBackendPreference(std::string_view value,
                                   ComputeBackendPreference *out) {
  std::string v(value);
  const auto begin = v.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos)
    return false;
  const auto end = v.find_last_not_of(" \t\r\n");
  v = v.substr(begin, end - begin + 1);
  for (char &c : v) {
    const unsigned char uc = static_cast<unsigned char>(c);
    c = static_cast<char>(std::tolower(uc));
  }

  if (v == "auto" || v == "auto_select" || v == "autoselect") {
    if (out)
      *out = ComputeBackendPreference::auto_select;
    return true;
  }
  if (v == "cpu") {
    if (out)
      *out = ComputeBackendPreference::cpu;
    return true;
  }
  if (v == "cuda") {
    if (out)
      *out = ComputeBackendPreference::cuda;
    return true;
  }
  if (v == "vulkan") {
    if (out)
      *out = ComputeBackendPreference::vulkan;
    return true;
  }
  return false;
}

ComputeBackendPreference
ParseComputeBackendPreferenceOr(std::string_view value,
                                ComputeBackendPreference fallback) {
  ComputeBackendPreference parsed = fallback;
  if (!ParseComputeBackendPreference(value, &parsed))
    return fallback;
  return parsed;
}

ComputeBackendSelection
ResolveComputeBackendSelection(ComputeBackendPreference pref,
                               const ComputeBackendAvailability &available,
                               bool compute_work_requested) {
  ComputeBackendSelection out;
  out.preference = pref;

  if (!compute_work_requested) {
    out.resolved = ComputeBackendKind::cpu;
    return out;
  }

  const auto mark_degraded = [&](std::string reason) {
    out.degraded = true;
    out.fallback_reason = reason;
    out.degraded_reason = reason;
  };

  switch (pref) {
  case ComputeBackendPreference::cpu:
    out.resolved = ComputeBackendKind::cpu;
    out.degraded = true;
    out.degraded_reason =
        "CPU compute backend selected; GPU-backed video effects are disabled.";
    return out;
  case ComputeBackendPreference::cuda:
    if (available.cuda_available) {
      out.resolved = ComputeBackendKind::cuda;
      return out;
    }
    out.resolved = ComputeBackendKind::cpu;
    mark_degraded(available.cuda_unavailable_reason.empty()
                      ? std::string("CUDA compute backend requested, but CUDA "
                                    "is not available.")
                      : available.cuda_unavailable_reason);
    return out;
  case ComputeBackendPreference::vulkan:
    if (available.vulkan_available) {
      out.resolved = ComputeBackendKind::vulkan;
      return out;
    }
    out.resolved = ComputeBackendKind::cpu;
    mark_degraded(available.vulkan_unavailable_reason.empty()
                      ? std::string("Vulkan compute backend requested, but "
                                    "Vulkan compute is not available in this "
                                    "build.")
                      : available.vulkan_unavailable_reason);
    return out;
  default:
    if (available.cuda_available) {
      out.resolved = ComputeBackendKind::cuda;
      return out;
    }
    if (available.vulkan_available) {
      out.resolved = ComputeBackendKind::vulkan;
      return out;
    }
    out.resolved = ComputeBackendKind::cpu;
    mark_degraded("No GPU compute backend is available; using CPU/pass-through "
                  "fallback.");
    return out;
  }
}

bool ComputeBackendAllowsCudaVideoCompute(ComputeBackendPreference pref) {
  return pref == ComputeBackendPreference::auto_select ||
         pref == ComputeBackendPreference::cuda;
}

std::string ResolveActiveComputeBackendName(
    const ComputeBackendSelection &selection, bool compute_work_requested,
    bool maxine_compute_available, bool cuda_compute_available,
    bool vulkan_compute_available) {
  if (!compute_work_requested)
    return "cpu";

  if (selection.resolved == ComputeBackendKind::vulkan &&
      vulkan_compute_available) {
    return "vulkan";
  }
  if (selection.resolved == ComputeBackendKind::cuda &&
      maxine_compute_available) {
    return "maxine";
  }
  if (selection.resolved == ComputeBackendKind::cuda &&
      cuda_compute_available) {
    return "cuda";
  }
  return "cpu";
}

bool MarkGpuBackendActiveFrame(bool &active_this_frame,
                               std::uint64_t &active_frames) {
  if (active_this_frame)
    return false;
  active_this_frame = true;
  ++active_frames;
  return true;
}

PreparedMaxineResidentIslands CompileMaxineResidentIslands(
    const effects::PreparedBroadcastEffectsFramePlan &frame_plan,
    std::uint32_t resolved_maxine_stage_mask,
    std::uint64_t matte_fingerprint) noexcept {
  PreparedMaxineResidentIslands result;
  if (!frame_plan.valid) {
    result.valid = false;
    return result;
  }

  const auto resident_kind = [](effects::BroadcastEffectStage stage) {
    using BroadcastStage = effects::BroadcastEffectStage;
    using ResidentStage = maxine::ResidentStageKind;
    switch (stage) {
    case BroadcastStage::video_noise_removal:
      return ResidentStage::denoise;
    case BroadcastStage::eye_contact:
      return ResidentStage::eye_contact;
    case BroadcastStage::virtual_background_blur:
      return ResidentStage::background_blur;
    case BroadcastStage::virtual_background_remove:
      return ResidentStage::background_remove;
    case BroadcastStage::virtual_background_replace:
      return ResidentStage::background_replace;
    case BroadcastStage::virtual_key_light:
      return ResidentStage::relighting;
    case BroadcastStage::auto_frame:
      return ResidentStage::auto_frame;
    case BroadcastStage::vignette:
      return ResidentStage::vignette;
    case BroadcastStage::mirror:
    case BroadcastStage::none:
      return ResidentStage::count;
    }
    return ResidentStage::count;
  };

  std::size_t frame_index = 0;
  while (frame_index < frame_plan.size) {
    const auto stage = frame_plan.StageAt(frame_index);
    const auto kind = resident_kind(stage);
    const bool is_resident = kind < maxine::ResidentStageKind::count &&
                             (resolved_maxine_stage_mask &
                              MaxineResolvedStageBit(stage)) != 0;
    if (!is_resident) {
      ++frame_index;
      continue;
    }
    if (result.count >= result.islands.size()) {
      result.valid = false;
      return result;
    }

    auto &island = result.islands[result.count++];
    island.first_frame_stage = frame_index;
    while (frame_index < frame_plan.size) {
      const auto candidate_stage = frame_plan.StageAt(frame_index);
      const auto candidate_kind = resident_kind(candidate_stage);
      if (candidate_kind >= maxine::ResidentStageKind::count ||
          (resolved_maxine_stage_mask &
           MaxineResolvedStageBit(candidate_stage)) == 0) {
        break;
      }
      const bool needs_matte =
          candidate_kind == maxine::ResidentStageKind::background_blur ||
          candidate_kind == maxine::ResidentStageKind::background_remove ||
          candidate_kind == maxine::ResidentStageKind::background_replace ||
          candidate_kind == maxine::ResidentStageKind::relighting;
      if (!island.plan.AddCompatible(candidate_kind, /*optional=*/true,
                                     needs_matte,
                                     needs_matte ? matte_fingerprint : 0)) {
        result.valid = false;
        return result;
      }
      result.enabled_resident_stage_mask |=
          std::uint32_t{1} << static_cast<std::uint32_t>(candidate_kind);
      ++frame_index;
    }
    island.end_frame_stage = frame_index;
    if (frame_index < frame_plan.size &&
        !island.plan.SetCpuTail(maxine::CpuTailKind::incompatible_stage)) {
      result.valid = false;
      return result;
    }
  }
  return result;
}

MaxineResidentTelemetryDelta ComputeMaxineResidentTelemetryDelta(
    const maxine::ProductionResidentTelemetry &current,
    const maxine::ProductionResidentTelemetry &previous) noexcept {
  MaxineResidentTelemetryDelta delta;
  const auto difference = [](std::uint64_t now, std::uint64_t before) {
    return now >= before ? now - before : 0;
  };
  const auto counter_delta = [&](const maxine::ProductionCallCounter &now,
                                 const maxine::ProductionCallCounter &before,
                                 std::uint64_t *attempts,
                                 std::uint64_t *successes) {
    *attempts = difference(now.attempts, before.attempts);
    *successes = difference(now.successes, before.successes);
  };
  const auto transfer_delta = [&](maxine::ProductionTransferKind kind,
                                  std::uint64_t *attempts,
                                  std::uint64_t *successes) {
    const auto index = static_cast<std::size_t>(kind);
    counter_delta(current.transfers[index], previous.transfers[index], attempts,
                  successes);
  };
  transfer_delta(maxine::ProductionTransferKind::host_upload,
                 &delta.upload_attempts, &delta.upload_calls);
  transfer_delta(maxine::ProductionTransferKind::final_download,
                 &delta.final_download_attempts, &delta.final_download_calls);
  transfer_delta(maxine::ProductionTransferKind::cpu_continuation_download,
                 &delta.cpu_continuation_download_attempts,
                 &delta.cpu_continuation_download_calls);
  transfer_delta(maxine::ProductionTransferKind::device_format_bridge,
                 &delta.device_bridge_attempts, &delta.device_bridge_calls);
  counter_delta(current.matte_inferences, previous.matte_inferences,
                &delta.matte_inference_attempts,
                &delta.matte_inference_calls);
  counter_delta(current.explicit_synchronizations,
                previous.explicit_synchronizations, &delta.sync_attempts,
                &delta.sync_calls);
  counter_delta(current.composites, previous.composites,
                &delta.composite_attempts, &delta.composite_calls);
  counter_delta(current.synchronous_sdk_runs, previous.synchronous_sdk_runs,
                &delta.synchronous_sdk_run_attempts,
                &delta.synchronous_sdk_run_calls);
  counter_delta(current.asynchronous_sdk_runs,
                previous.asynchronous_sdk_runs,
                &delta.asynchronous_sdk_run_attempts,
                &delta.asynchronous_sdk_run_calls);
  const auto rgb_index = static_cast<std::size_t>(
      maxine::ProductionCpuStageKind::rgb_to_bgr_staging);
  std::uint64_t ignored_attempts = 0;
  counter_delta(current.cpu_stages[rgb_index], previous.cpu_stages[rgb_index],
                &ignored_attempts, &delta.rgb_to_bgr_calls);
  delta.shared_matte_reuses = difference(
      current.shared_matte_reuses, previous.shared_matte_reuses);
  for (std::size_t i = 0; i < current.stages.size(); ++i) {
    counter_delta(current.stages[i], previous.stages[i],
                  &delta.stage_attempts[i], &delta.stage_successes[i]);
  }
  return delta;
}

bool EffectsPlanRequiresRebuild(
    const effects::BroadcastCameraEffects &applied_effects,
    std::uint64_t applied_generation,
    const effects::BroadcastCameraEffects &current_effects,
    std::uint64_t current_generation) {
  return applied_effects != current_effects ||
         applied_generation != current_generation;
}

bool EffectsFramePreparationCounters::ObservePublishedGeneration(
    std::uint64_t published_generation,
    std::uint64_t applied_generation) noexcept {
  ++generation_loads;
  if (published_generation == 0) {
    ++transition_sentinel_loads;
    return false;
  }
  if (published_generation == applied_generation)
    return false;
  ++snapshot_requests;
  return true;
}

void EffectsFramePreparationCounters::RecordConfigSnapshot(
    bool rebuilt) noexcept {
  ++config_lock_acquisitions;
  ++config_copies;
  if (rebuilt)
    ++rebuilds;
}

void LiveEffectBackendAttribution::BeginFrame() { pending_.fill(0); }

void LiveEffectBackendAttribution::MarkEffectSucceeded(
    std::string_view effect_id, std::string_view backend) {
  const std::size_t slot = EffectSlot(effect_id);
  const std::uint8_t code = BackendCode(backend);
  if (slot >= kEffectCount || code == 0)
    return;
  pending_[slot] = code;
}

bool LiveEffectBackendAttribution::RemoveEffect(std::string_view effect_id) {
  const std::size_t slot = EffectSlot(effect_id);
  if (slot >= kEffectCount)
    return false;
  pending_[slot] = 0;
  if (active_[slot] == 0)
    return false;
  active_[slot] = 0;
  SerializeActive();
  return true;
}

void LiveEffectBackendAttribution::DiscardFrame() { pending_.fill(0); }

bool LiveEffectBackendAttribution::CommitSuccessfulFrame(
    const std::vector<std::string> &canonical_order) {
  if (active_ == pending_) {
    pending_.fill(0);
    return false;
  }
  active_ = pending_;
  pending_.fill(0);
  RememberOrder(canonical_order);
  SerializeActive();
  return true;
}

void LiveEffectBackendAttribution::Clear() {
  pending_.fill(0);
  active_.fill(0);
  active_order_.fill(0);
  active_order_size_ = 0;
  active_backends_.clear();
}

std::size_t
LiveEffectBackendAttribution::EffectSlot(std::string_view effect_id) {
  static constexpr std::array<std::string_view, kEffectCount> ids = {
      "mirror",
      "virtual_background.blur",
      "virtual_background.remove",
      "virtual_background.replace",
      "auto_frame",
      "eye_contact",
      "video_noise_removal",
      "virtual_key_light",
      "vignette"};
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (effect_id == ids[i])
      return i;
  }
  return kEffectCount;
}

std::uint8_t
LiveEffectBackendAttribution::BackendCode(std::string_view backend) {
  static constexpr std::array<std::string_view, 7> ids = {
      "open_vulkan", "open_cuda",      "open_video", "maxine",
      "maxine_ar",   "maxine_ar_cuda", "cuda"};
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (backend == ids[i])
      return static_cast<std::uint8_t>(i + 1);
  }
  return 0;
}

std::string_view LiveEffectBackendAttribution::EffectId(std::size_t slot) {
  static constexpr std::array<std::string_view, kEffectCount> ids = {
      "mirror",
      "virtual_background.blur",
      "virtual_background.remove",
      "virtual_background.replace",
      "auto_frame",
      "eye_contact",
      "video_noise_removal",
      "virtual_key_light",
      "vignette"};
  return slot < ids.size() ? ids[slot] : std::string_view{};
}

std::string_view LiveEffectBackendAttribution::BackendId(std::uint8_t code) {
  static constexpr std::array<std::string_view, 7> ids = {
      "open_vulkan", "open_cuda",      "open_video", "maxine",
      "maxine_ar",   "maxine_ar_cuda", "cuda"};
  return code > 0 && code <= ids.size() ? ids[code - 1] : std::string_view{};
}

void LiveEffectBackendAttribution::RememberOrder(
    const std::vector<std::string> &canonical_order) {
  active_order_.fill(0);
  active_order_size_ = 0;
  std::array<bool, kEffectCount> present{};
  const auto append_slot = [&](std::size_t slot) {
    if (slot >= kEffectCount || present[slot])
      return;
    present[slot] = true;
    active_order_[active_order_size_++] = static_cast<std::uint8_t>(slot);
  };
  for (const auto &effect_id : canonical_order)
    append_slot(EffectSlot(effect_id));
  for (std::size_t slot = 0; slot < kEffectCount; ++slot)
    append_slot(slot);
}

void LiveEffectBackendAttribution::SerializeActive() {
  active_backends_.clear();
  for (std::size_t i = 0; i < active_order_size_; ++i) {
    const std::size_t slot = active_order_[i];
    if (slot >= kEffectCount || active_[slot] == 0)
      continue;
    if (!active_backends_.empty())
      active_backends_.push_back(',');
    active_backends_.append(EffectId(slot));
    active_backends_.push_back(':');
    active_backends_.append(BackendId(active_[slot]));
  }
}

OpenVulkanVignettePlanCompatibility ApplyOpenVulkanVignettePlanCompatibility(
    const effects::BroadcastCameraEffects &fx,
    effects::BroadcastEffectsPlan *retained_plan) {
  OpenVulkanVignettePlanCompatibility result;
  if (!retained_plan || !fx.vignette.enabled ||
      !fx.vignette.center_on_tracked_face) {
    return result;
  }

  const auto contains = [&](std::string_view effect_id) {
    return std::find(retained_plan->ordered_effect_ids.begin(),
                     retained_plan->ordered_effect_ids.end(),
                     effect_id) != retained_plan->ordered_effect_ids.end();
  };
  if (!contains(effects::contract::kEffectIdVignette) ||
      !contains(effects::contract::kEffectIdAutoFrame)) {
    return result;
  }

  const std::string vignette_id(effects::contract::kEffectIdVignette);
  retained_plan->ordered_effect_ids.erase(
      std::remove(retained_plan->ordered_effect_ids.begin(),
                  retained_plan->ordered_effect_ids.end(), vignette_id),
      retained_plan->ordered_effect_ids.end());
  retained_plan->vignette_attach_to_effect_id.clear();
  result.blocked = true;
  result.reason_code = kOpenVulkanVignetteTrackedCenterNotSupportedReason;
  result.detail = kOpenVulkanVignetteTrackedCenterNotSupportedDetail;
  retained_plan->disabled.push_back(effects::DisabledEffectByRule{
      .id = vignette_id,
      .reason = "[" + std::string(result.reason_code) + "] " +
                std::string(result.detail)});
  return result;
}

OpenVulkanEyeContactPlanCompatibility
ApplyOpenVulkanEyeContactPlanCompatibility(
    effects::BroadcastEffectsPlan *retained_plan) {
  OpenVulkanEyeContactPlanCompatibility result;
  if (!retained_plan)
    return result;

  const std::string eye_contact_id(effects::contract::kEffectIdEyeContact);
  const auto stage = std::find(retained_plan->ordered_effect_ids.begin(),
                               retained_plan->ordered_effect_ids.end(),
                               eye_contact_id);
  if (stage == retained_plan->ordered_effect_ids.end())
    return result;

  retained_plan->ordered_effect_ids.erase(stage);
  if (retained_plan->vignette_attach_to_effect_id == eye_contact_id)
    retained_plan->vignette_attach_to_effect_id.clear();

  result.blocked = true;
  result.reason_code = kOpenVulkanEyeContactUnavailableReason;
  result.blocker_code = kOpenVulkanEyeContactRuntimeUnavailableReason;
  result.detail = kOpenVulkanEyeContactUnavailableDetail;
  retained_plan->disabled.push_back(effects::DisabledEffectByRule{
      .id = eye_contact_id,
      .reason = "[" + std::string(result.reason_code) + "] [" +
                std::string(result.blocker_code) + "] " +
                std::string(result.detail)});
  return result;
}

OpenVulkanVideoNoiseRemovalPlanCompatibility
ApplyOpenVulkanVideoNoiseRemovalPlanCompatibility(
    effects::BroadcastEffectsPlan *retained_plan) {
  OpenVulkanVideoNoiseRemovalPlanCompatibility result;
  if (!retained_plan)
    return result;

  const std::string denoise_id(effects::contract::kEffectIdVideoNoiseRemoval);
  const auto stage =
      std::find(retained_plan->ordered_effect_ids.begin(),
                retained_plan->ordered_effect_ids.end(), denoise_id);
  if (stage == retained_plan->ordered_effect_ids.end())
    return result;

  retained_plan->ordered_effect_ids.erase(stage);
  if (retained_plan->vignette_attach_to_effect_id == denoise_id)
    retained_plan->vignette_attach_to_effect_id.clear();

  result.blocked = true;
  result.reason_code = kOpenVulkanVideoNoiseRemovalUnavailableReason;
  result.blocker_code = kOpenVulkanVideoNoiseRemovalRuntimeUnavailableReason;
  result.detail = kOpenVulkanVideoNoiseRemovalUnavailableDetail;
  retained_plan->disabled.push_back(effects::DisabledEffectByRule{
      .id = denoise_id,
      .reason = "[" + std::string(result.reason_code) + "] [" +
                std::string(result.blocker_code) + "] " +
                std::string(result.detail)});
  return result;
}

namespace {

detail::PreparedReplaceBackgroundSource
PrepareReplaceBackgroundSourceForEffects(
    const studiocast::video::effects::BroadcastCameraEffects &fx) {
  detail::PreparedReplaceBackgroundSource prepared;
  if (fx.virtual_background.mode !=
      studiocast::video::effects::VirtualBackgroundMode::replace) {
    return prepared;
  }

  std::string error;
  (void)detail::PrepareReplaceBackgroundSourceForConfig(
      fx.virtual_background.replace_path, &prepared, &error);
  return prepared;
}

bool CanReusePreparedReplaceBackgroundSource(
    const studiocast::video::effects::BroadcastCameraEffects &current,
    const studiocast::video::effects::BroadcastCameraEffects &next) {
  using studiocast::video::effects::VirtualBackgroundMode;
  return current.virtual_background.mode == VirtualBackgroundMode::replace &&
         next.virtual_background.mode == VirtualBackgroundMode::replace &&
         current.virtual_background.replace_path ==
             next.virtual_background.replace_path;
}

std::string ChooseDefaultOutputLoopback(std::string *error) {
  const auto rep = ProbeLoopback();
  for (const auto &d : rep.devices) {
    if (d.is_loopback && d.can_write)
      return d.dev_node;
  }
  if (error) {
    std::ostringstream oss;
    oss << "No writable v4l2loopback device found.\n"
        << "Run the suggested command from studiocast-video status, e.g.:\n"
        << "  " << rep.suggested_modprobe_cmd << "\n";
    *error = oss.str();
  }
  return {};
}

std::string ToLowerAscii(std::string s) {
  for (char &c : s) {
    const unsigned char uc = static_cast<unsigned char>(c);
    c = static_cast<char>(std::tolower(uc));
  }
  return s;
}

bool ActualMatchesOutputRequest(const ActualFormat &actual, int width,
                                int height, int fps,
                                PixelFormat requested_format, bool strict_fps) {
  if (actual.width != width || actual.height != height)
    return false;
  if (actual.format != requested_format)
    return false;
  if (strict_fps && actual.fps != fps)
    return false;
  return true;
}

std::string ActualOutputMismatchMessage(const ActualFormat &actual, int width,
                                        int height, int fps,
                                        PixelFormat requested_format,
                                        bool strict_fps) {
  std::ostringstream oss;
  oss << "requested output_format=" << PixelFormatName(requested_format) << " "
      << width << "x" << height;
  if (strict_fps)
    oss << "@" << fps;
  oss << ", negotiated output_format=" << PixelFormatName(actual.format) << " "
      << actual.width << "x" << actual.height;
  if (actual.fps > 0)
    oss << "@" << actual.fps;
  if (!actual.pixfmt.empty())
    oss << " (" << actual.pixfmt << ")";
  return oss.str();
}

bool OpenCudaTensorRtEnabledFromEnv() {
  const char *v = std::getenv("STUDIOCAST_OPEN_CUDA_TENSORRT");
  if (!v || !*v)
    return false;

  const std::string value = ToLowerAscii(v);
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

const studiocast::open_video::ModelFile *
FindPreparedOpenCudaMainOnnx(
    const studiocast::open_video::ModelPack &pack) noexcept {
  for (const auto &file : pack.files) {
    if (file.kind == "onnx" &&
        (file.role.empty() || file.role == "main")) {
      return &file;
    }
  }
  for (const auto &file : pack.files) {
    if (file.kind == "onnx")
      return &file;
  }
  return nullptr;
}

bool CheckedOpenCudaBudgetMultiply(std::uint64_t lhs, std::uint64_t rhs,
                                   std::uint64_t *result) noexcept {
  if (!result ||
      (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool CheckedOpenCudaBudgetAdd(std::uint64_t value,
                              std::uint64_t *total) noexcept {
  if (!total ||
      value > std::numeric_limits<std::uint64_t>::max() - *total) {
    return false;
  }
  *total += value;
  return true;
}

bool PreparedOpenCudaBudgetFor(
    std::uint32_t frame_width, std::uint32_t frame_height,
    std::uint32_t model_width, std::uint32_t model_height,
    hal::SemanticEffectId semantic, int strength,
    hal::ResourceBudget *budget) noexcept {
  if (!budget || frame_width == 0 || frame_height == 0 || model_width == 0 ||
      model_height == 0) {
    return false;
  }

  // CUDA pitched allocations are implementation-owned. Reserve every image
  // row at a conservative 4-KiB-aligned upper bound while retaining exact
  // geometry/model dimensions. The resident executor records the exact pitch
  // and commits only the actually reserved budget after allocation.
  const auto aligned_row = [](std::uint64_t active,
                              std::uint64_t *aligned) noexcept {
    constexpr std::uint64_t kAlignment = 4096;
    if (!aligned ||
        active > std::numeric_limits<std::uint64_t>::max() -
                     (kAlignment - 1)) {
      return false;
    }
    *aligned = ((active + kAlignment - 1) / kAlignment) * kAlignment;
    return true;
  };
  std::uint64_t rgb_row = 0;
  std::uint64_t alpha_row = 0;
  std::uint64_t frame_rgb = 0;
  std::uint64_t frame_alpha = 0;
  std::uint64_t model_alpha_row = 0;
  std::uint64_t model_tensor_row = 0;
  std::uint64_t model_alpha = 0;
  std::uint64_t model_tensor = 0;
  std::uint64_t boundary = 0;
  if (!aligned_row(static_cast<std::uint64_t>(frame_width) * 3u, &rgb_row) ||
      !aligned_row(static_cast<std::uint64_t>(frame_width) * 4u, &alpha_row) ||
      !CheckedOpenCudaBudgetMultiply(rgb_row, frame_height, &frame_rgb) ||
      !CheckedOpenCudaBudgetMultiply(alpha_row, frame_height, &frame_alpha) ||
      !CheckedOpenCudaBudgetMultiply(model_width, 4u, &model_alpha_row) ||
      !aligned_row(model_alpha_row, &model_alpha_row) ||
      !CheckedOpenCudaBudgetMultiply(model_alpha_row, model_height,
                                     &model_alpha) ||
      !CheckedOpenCudaBudgetMultiply(model_width, 12u, &model_tensor_row) ||
      !aligned_row(model_tensor_row, &model_tensor_row) ||
      !CheckedOpenCudaBudgetMultiply(model_tensor_row, model_height,
                                     &model_tensor) ||
      !CheckedOpenCudaBudgetMultiply(frame_width, frame_height, &boundary) ||
      !CheckedOpenCudaBudgetMultiply(boundary, 3u, &boundary)) {
    return false;
  }

  hal::ResourceBudget result{};
  // Input, output, model alpha, NCHW input tensor, and frame-sized alpha.
  if (!CheckedOpenCudaBudgetAdd(frame_rgb, &result.persistent_bytes) ||
      !CheckedOpenCudaBudgetAdd(frame_rgb, &result.persistent_bytes) ||
      !CheckedOpenCudaBudgetAdd(model_alpha, &result.persistent_bytes) ||
      !CheckedOpenCudaBudgetAdd(model_tensor, &result.persistent_bytes) ||
      !CheckedOpenCudaBudgetAdd(frame_alpha, &result.persistent_bytes)) {
    return false;
  }
  const int canonical_strength = std::clamp(
      strength, effects::contract::kVbStrengthMin,
      effects::contract::kVbStrengthMax);
  const bool feather = std::min(4, canonical_strength / 16) > 0;
  const bool blur =
      semantic == hal::SemanticEffectId::virtual_background_blur;
  if (feather &&
      (!CheckedOpenCudaBudgetAdd(frame_alpha, &result.scratch_bytes) ||
       !CheckedOpenCudaBudgetAdd(frame_alpha, &result.scratch_bytes))) {
    return false;
  }
  if (blur &&
      (!CheckedOpenCudaBudgetAdd(frame_rgb, &result.scratch_bytes) ||
       !CheckedOpenCudaBudgetAdd(frame_rgb, &result.scratch_bytes))) {
    return false;
  }
  // The facade rejects a zero scratch declaration; one byte explicitly means
  // no scratch allocation is required for remove-without-feather.
  result.scratch_bytes = std::max<std::uint64_t>(result.scratch_bytes, 1u);
  result.upload_staging_bytes = boundary;
  result.readback_staging_bytes = boundary;
  result.resource_count =
      (blur ? 19u : 17u) + (feather ? 2u : 0u);
  result.in_flight_slots = 1;

  constexpr std::uint64_t kMaxPersistent = UINT64_C(2) << 30;
  constexpr std::uint64_t kMaxScratch = UINT64_C(1) << 30;
  constexpr std::uint64_t kMaxBoundary = UINT64_C(1) << 30;
  if (result.persistent_bytes > kMaxPersistent ||
      result.scratch_bytes > kMaxScratch ||
      result.upload_staging_bytes > kMaxBoundary ||
      result.readback_staging_bytes > kMaxBoundary) {
    return false;
  }
  *budget = result;
  return true;
}

bool ParseRgbHex(const std::string &s, std::uint32_t *out_rgb) {
  if (!out_rgb)
    return false;
  *out_rgb = 0;

  std::string t = s;
  if (t.rfind("0x", 0) == 0 || t.rfind("0X", 0) == 0) {
    t = t.substr(2);
  }
  if (!t.empty() && t.front() == '#') {
    t = t.substr(1);
  }
  if (t.size() != 6)
    return false;

  std::uint32_t v = 0;
  for (const char ch : t) {
    v <<= 4u;
    if (ch >= '0' && ch <= '9') {
      v |= static_cast<std::uint32_t>(ch - '0');
    } else if (ch >= 'a' && ch <= 'f') {
      v |= static_cast<std::uint32_t>(ch - 'a' + 10);
    } else if (ch >= 'A' && ch <= 'F') {
      v |= static_cast<std::uint32_t>(ch - 'A' + 10);
    } else {
      return false;
    }
  }
  *out_rgb = (v & 0xFFFFFFu);
  return true;
}

float Clamp01FromPercent(int percent) {
  const int p = std::max(0, std::min(100, percent));
  return static_cast<float>(p) / 100.0f;
}

int ParseVideoIndex(const std::string &sys_name) {
  // sys_name is usually "videoN".
  if (sys_name.rfind("video", 0) != 0)
    return -1;
  const std::string rest = sys_name.substr(5);
  if (rest.empty())
    return -1;
  int v = 0;
  for (const char ch : rest) {
    if (ch < '0' || ch > '9')
      return -1;
    v = v * 10 + (ch - '0');
    if (v > 4096)
      return -1;
  }
  return v;
}

bool CheckedPositiveDims(int width, int height, unsigned *out_w,
                         unsigned *out_h, std::string *error,
                         const char *what) {
  if (!out_w || !out_h)
    return false;

  if (width <= 0 || height <= 0) {
    if (error) {
      std::ostringstream oss;
      oss << (what ? what : "") << " invalid frame size: " << width << "x"
          << height << ".";
      *error = oss.str();
    }
    return false;
  }

  *out_w = static_cast<unsigned>(width);
  *out_h = static_cast<unsigned>(height);
  return true;
}

[[maybe_unused]] static bool
SyncAfterGpuToCpuTransfer(studiocast::maxine::CudaDriverApi &cuda,
                          studiocast::maxine::CUstream stream,
                          const char *context, std::string *error) {
  std::string sync_err;
  if (!cuda.StreamSynchronize(stream, &sync_err)) {
    if (error) {
      const char *ctx = context ? context : "(unknown context)";
      *error =
          std::string(ctx) + ": cuda StreamSynchronize failed: " + sync_err;
    }
    return false;
  }
  return true;
}

int ScoreCamera(const VideoDevice &d) {
  // Heuristic score for "auto" camera selection.
  // Prefer typical UVC webcams, avoid IR/depth/metadata nodes when possible.
  int score = 0;
  if (d.driver == "uvcvideo")
    score += 10;

  const auto name = ToLowerAscii(d.name);
  if (name.find("ir") != std::string::npos)
    score -= 50;
  if (name.find("depth") != std::string::npos)
    score -= 50;
  if (name.find("metadata") != std::string::npos)
    score -= 50;

  if (name.find("hd") != std::string::npos)
    score += 5;
  if (name.find("camera") != std::string::npos)
    score += 2;

  return score;
}

std::vector<VideoDevice> ListCandidateCameras() {
  const auto rep = ProbeLoopback();
  std::vector<VideoDevice> cams;
  cams.reserve(rep.devices.size());
  for (const auto &d : rep.devices) {
    if (!d.is_loopback && d.can_read)
      cams.push_back(d);
  }

  std::sort(cams.begin(), cams.end(),
            [](const VideoDevice &a, const VideoDevice &b) {
              const int sa = ScoreCamera(a);
              const int sb = ScoreCamera(b);
              if (sa != sb)
                return sa > sb;

              const int ia = ParseVideoIndex(a.sys_name);
              const int ib = ParseVideoIndex(b.sys_name);
              if (ia != ib) {
                // Prefer lower indices (video0, video1, ...)
                if (ia < 0)
                  return false;
                if (ib < 0)
                  return true;
                return ia < ib;
              }

              return a.dev_node < b.dev_node;
            });

  return cams;
}

constexpr const char *kOptionalEffectCoolingDownState = "cooling_down";
constexpr const char *kOptionalEffectRetryReadyState = "retry_ready";

std::string OptionalEffectFailureReason(const char *prefix,
                                        const std::string &detail) {
  if (!prefix || !*prefix)
    return detail.empty() ? std::string("Effect failed.") : detail;
  if (detail.empty())
    return std::string(prefix);
  return std::string(prefix) + ": " + detail;
}

bool IsVignetteFailureReason(const std::string &reason) {
  return reason == "Vignette failed" ||
         reason.rfind("Vignette failed:", 0) == 0;
}

template <std::size_t N>
std::string
BuildDegradedEffectsNote(const std::array<OptionalEffectBreaker, N> &breakers,
                         std::uint64_t frame_index) {
  std::string note;
  for (const auto &breaker : breakers) {
    if (!breaker.active())
      continue;
    const auto status = breaker.ToStatus(frame_index);
    if (note.empty()) {
      note = "Runtime degraded effects:";
    }
    note += "\n - ";
    note += status.effect_id;
    note += " [";
    note += status.backend;
    note += "]: ";
    note += status.reason;
    note += " (state=";
    note += status.state;
    note += ", failures=";
    note += std::to_string(status.failure_count);
    note += ", cooldown_frames=";
    note += std::to_string(status.cooldown_frames);
    note += ")";
  }
  return note;
}

template <std::size_t N>
CameraPipelineStatus::DegradedEffect
MostRecentDegradedEffect(const std::array<OptionalEffectBreaker, N> &breakers,
                         std::uint64_t frame_index) {
  const OptionalEffectBreaker *latest = nullptr;
  for (const auto &breaker : breakers) {
    if (!breaker.active())
      continue;
    if (!latest || breaker.trip_order() > latest->trip_order())
      latest = &breaker;
  }
  return latest ? latest->ToStatus(frame_index)
                : CameraPipelineStatus::DegradedEffect{};
}

std::string CombineEffectsNotes(const std::string &base,
                                const std::string &runtime) {
  if (base.empty())
    return runtime;
  if (runtime.empty())
    return base;
  return base + "\n" + runtime;
}

} // namespace

bool BuildPreparedOpenCudaPipelineResourceBudget(
    std::uint32_t frame_width, std::uint32_t frame_height,
    std::uint32_t model_width, std::uint32_t model_height,
    hal::SemanticEffectId semantic, int strength,
    hal::ResourceBudget *budget) noexcept {
  return PreparedOpenCudaBudgetFor(frame_width, frame_height, model_width,
                                   model_height, semantic, strength, budget);
}

bool PreparedOpenCudaPipelineMayCommitAttribution(
    hal::StageExecutionStatus stage_status, bool writer_frame_committed,
    hal::StatusId output_commit_status) noexcept {
  return writer_frame_committed &&
         output_commit_status == hal::StatusId::ready &&
         stage_status == hal::StageExecutionStatus::succeeded;
}

bool PreparedOpenCudaPipelineShouldSuspend(
    hal::StatusId status, hal::ReasonId reason) noexcept {
  return reason == hal::ReasonId::device_lost ||
         reason == hal::ReasonId::shutdown_requested ||
         status == hal::StatusId::quarantined ||
         status == hal::StatusId::stopped;
}

MultiIslandStatusValidationResult BuildCameraPipelineHalStatusSnapshot(
    const CameraPipelineHalStatusProjectionInput &input,
    MultiIslandVideoRuntimeStatusSnapshot *output) noexcept {
  using CapabilityCode = effects::EffectCapabilityCode;
  using Lane = ProductCapabilityLaneId;
  if (!output || !input.effects) {
    if (output)
      *output = {};
    MultiIslandStatusValidationResult invalid;
    invalid.code = MultiIslandStatusValidationCode::invalid_generation;
    return invalid;
  }

  const auto requested = [&](hal::SemanticEffectId effect) {
    switch (effect) {
    case hal::SemanticEffectId::video_noise_removal:
      return input.effects->video_noise_removal.enabled;
    case hal::SemanticEffectId::eye_contact:
      return input.effects->eye_contact.enabled;
    case hal::SemanticEffectId::virtual_background_blur:
      return input.effects->virtual_background.mode ==
             effects::VirtualBackgroundMode::blur;
    case hal::SemanticEffectId::virtual_background_remove:
      return input.effects->virtual_background.mode ==
             effects::VirtualBackgroundMode::remove;
    case hal::SemanticEffectId::virtual_background_replace:
      return input.effects->virtual_background.mode ==
             effects::VirtualBackgroundMode::replace;
    case hal::SemanticEffectId::virtual_key_light:
      return input.effects->virtual_key_light.enabled;
    case hal::SemanticEffectId::auto_frame:
      return input.effects->auto_frame.enabled;
    case hal::SemanticEffectId::vignette:
      return input.effects->vignette.enabled;
    case hal::SemanticEffectId::mirror:
      return input.effects->mirror;
    case hal::SemanticEffectId::invalid:
      return false;
    }
    return false;
  };
  const auto lane_for_backend = [](hal::BackendId backend) {
    switch (backend) {
    case hal::BackendId::maxine:
      return Lane::maxine;
    case hal::BackendId::cuda:
    case hal::BackendId::open_video_onnxruntime:
      return Lane::open_cuda;
    case hal::BackendId::vulkan:
      return Lane::open_vulkan;
    case hal::BackendId::cpu:
      return Lane::cpu;
    case hal::BackendId::invalid:
    case hal::BackendId::test_fake:
      return Lane::invalid;
    }
    return Lane::invalid;
  };
  const auto capability_code = [](Lane lane, hal::SemanticEffectId effect) {
    switch (lane) {
    case Lane::maxine:
      return CapabilityCode::maxine_effect_usable;
    case Lane::open_cuda:
      return CapabilityCode::open_cuda_effect_available;
    case Lane::open_vulkan:
      if (effect == hal::SemanticEffectId::mirror)
        return CapabilityCode::open_vulkan_mirror_production_ready;
      if (effect == hal::SemanticEffectId::vignette) {
        return CapabilityCode::
            open_vulkan_vignette_fixed_center_production_ready;
      }
      return CapabilityCode::none;
    case Lane::cpu:
    case Lane::invalid:
      return CapabilityCode::none;
    }
    return CapabilityCode::none;
  };
  const auto requested_provider =
      [](hal::BackendId backend, hal::SemanticEffectId effect) {
        if (backend == hal::BackendId::vulkan) {
          return hal::SemanticEffectProviderId::
              open_vulkan_production_kernels;
        }
        if (backend == hal::BackendId::cuda ||
            backend == hal::BackendId::open_video_onnxruntime) {
          return hal::SemanticEffectProviderId::
              open_video_onnxruntime_cuda;
        }
        if (backend == hal::BackendId::maxine) {
          if (effect == hal::SemanticEffectId::eye_contact ||
              effect == hal::SemanticEffectId::auto_frame) {
            return hal::SemanticEffectProviderId::maxine_ar;
          }
          return effect == hal::SemanticEffectId::vignette
                     ? hal::SemanticEffectProviderId::maxine_cuda_utility
                     : hal::SemanticEffectProviderId::maxine_vfx;
        }
        return hal::SemanticEffectProviderId::invalid;
      };
  const auto desired_backend = [&](hal::SemanticEffectId effect) {
    if ((effect == hal::SemanticEffectId::mirror ||
         effect == hal::SemanticEffectId::vignette) &&
        input.compute_backend == ComputeBackendPreference::vulkan) {
      return hal::BackendId::vulkan;
    }
    if (input.effects->engine == effects::EffectsEnginePreference::open_cuda)
      return hal::BackendId::cuda;
    if (input.effects->engine == effects::EffectsEnginePreference::maxine)
      return hal::BackendId::maxine;
    if (input.compute_backend == ComputeBackendPreference::cuda)
      return hal::BackendId::cuda;
    return hal::BackendId::maxine;
  };

  MultiIslandVideoRuntimeStatusBuildInput build;
  build.status_generation = input.status_generation;
  build.lifecycle = input.lifecycle;
  build.effect_count = kCanonicalVideoEffectCount;
  for (std::size_t index = 0; index < build.effect_count; ++index) {
    build.effects[index].effect = kCanonicalVideoEffects[index];
    build.effects[index].status = hal::StatusId::passthrough;
  }

  struct IslandSource {
    PreparedExecutionRouteId route = PreparedExecutionRouteId::invalid;
    hal::PreparedProgram program{};
    hal::ResolutionResult resolution{};
    hal::ReadinessSnapshot readiness{};
    hal::ExecutionTelemetry telemetry{};
    hal::CommittedFrameSnapshot committed{};
    bool current = false;
  };
  std::array<IslandSource, 3> sources{};
  std::size_t source_count = 0;

  const auto reject_malformed_source =
      [&](MultiIslandStatusValidationCode code) {
        *output = {};
        MultiIslandStatusValidationResult invalid;
        invalid.code = code;
        return invalid;
      };

  if (input.final_visual_live) {
    if (!input.final_visual.prepared ||
        input.final_visual.route != PreparedHardwareRoute::hal_live ||
        !input.final_visual.active_program.valid ||
        !hal::ValidateExecutionDomain(
            input.final_visual.active_program.execution_domain)) {
      return reject_malformed_source(
          MultiIslandStatusValidationCode::invalid_island);
    }
    if (input.final_visual.setup.resolution.stage_count == 0 ||
        input.final_visual.setup.resolution.stage_count >
            input.final_visual.setup.resolution.stages.size() ||
        input.final_visual.latest_committed_frame.stage_count >
            input.final_visual.latest_committed_frame.stages.size()) {
      return reject_malformed_source(
          MultiIslandStatusValidationCode::malformed_island_stages);
    }
    IslandSource source;
    source.route = PreparedExecutionRouteId::final_visual;
    source.program.contract_version = hal::kContractVersion;
    source.program.program_id =
        input.final_visual.active_program.program_id;
    source.program.owner_adapter_instance_id =
        input.final_visual.active_program.owner_adapter_instance_id;
    source.program.execution_domain =
        input.final_visual.active_program.execution_domain;
    source.program.invalidation_key =
        input.final_visual.active_program.invalidation_key;
    source.program.requested_backend =
        input.final_visual.readiness.requested_backend;
    source.program.strict_backend_selection =
        input.final_visual.setup.resolution.strict_backend_selection;
    source.program.requested_provider =
        input.final_visual.setup.resolution.selected_provider;
    source.program.strict_provider_selection = false;
    source.program.stage_count =
        input.final_visual.setup.resolution.stage_count;
    for (std::size_t index = 0; index < source.program.stage_count; ++index) {
      const auto &resolved =
          input.final_visual.setup.resolution.stages[index];
      if (resolved.intent.semantic != hal::SemanticEffectId::mirror &&
          resolved.intent.semantic != hal::SemanticEffectId::vignette) {
        return reject_malformed_source(
            MultiIslandStatusValidationCode::invalid_capability_evidence);
      }
      auto &stage = source.program.stages[index];
      stage.stage_id = resolved.intent.stage_id;
      stage.semantic = resolved.intent.semantic;
      stage.provider = resolved.provider;
      stage.backend = resolved.selected_backend;
      stage.execution_domain = resolved.execution_domain;
      stage.required = resolved.intent.required;
      stage.failure_action = resolved.intent.failure_action;
    }
    source.resolution = input.final_visual.setup.resolution;
    source.readiness = input.final_visual.readiness;
    source.telemetry = input.final_visual.telemetry;
    source.committed = input.final_visual.latest_committed_frame;
    source.current = true;
    sources[source_count++] = source;
  }

  const auto add_program_source =
      [&](PreparedExecutionRouteId route, const hal::PreparedProgram &program,
          const hal::ReadinessSnapshot &readiness,
          const hal::ExecutionTelemetry &telemetry,
          const hal::CommittedFrameSnapshot &committed) {
        if (source_count >= sources.size() || program.program_id == 0 ||
            program.stage_count == 0 ||
            program.stage_count > program.stages.size() ||
            committed.stage_count > committed.stages.size() ||
            !hal::ValidateExecutionDomain(program.execution_domain)) {
          return false;
        }
        IslandSource source;
        source.route = route;
        source.program = program;
        source.readiness = readiness;
        source.telemetry = telemetry;
        source.committed = committed;
        source.current = true;
        source.resolution.request_mode =
            program.strict_backend_selection
                ? hal::BackendRequestMode::explicit_backend
                : hal::BackendRequestMode::automatic;
        source.resolution.strict_backend_selection =
            program.strict_backend_selection;
        source.resolution.status = readiness.status;
        source.resolution.reason = readiness.reason;
        source.resolution.requested_backend = program.requested_backend;
        source.resolution.selected_backend = readiness.selected_backend;
        source.resolution.selected_provider = readiness.provider;
        source.resolution.stage_count = program.stage_count;
        source.resolution.available = readiness.available;
        source.resolution.degraded = readiness.degraded;
        source.resolution.fallback =
            program.requested_backend != program.execution_domain.backend;
        for (std::size_t index = 0; index < program.stage_count; ++index) {
          const auto &stage = program.stages[index];
          auto &resolved = source.resolution.stages[index];
          resolved.intent.stage_id = stage.stage_id;
          resolved.intent.semantic = stage.semantic;
          resolved.intent.required = stage.required;
          resolved.intent.failure_action = stage.failure_action;
          resolved.status = readiness.degraded ? hal::StatusId::degraded
                                               : hal::StatusId::ready;
          resolved.reason = readiness.reason;
          resolved.selected_backend = stage.backend;
          resolved.provider = stage.provider;
          resolved.adapter_instance_id =
              program.owner_adapter_instance_id;
          resolved.execution_domain = stage.execution_domain;
          resolved.cpu_tail =
              index + 1 == program.stage_count && program.cpu_tail_count != 0
                  ? program.cpu_tails[0].id
                  : hal::CpuTailId::invalid;
          resolved.cpu_tail_scratch_bytes =
              resolved.cpu_tail == hal::CpuTailId::invalid
                  ? 0
                  : program.cpu_tails[0].scratch_bytes;
          resolved.cpu_tail_placement =
              resolved.cpu_tail == hal::CpuTailId::invalid
                  ? hal::DeclaredCpuTail::Placement::after_stage
                  : program.cpu_tails[0].placement;
          resolved.fallback_selected = source.resolution.fallback;
        }
        sources[source_count++] = source;
        return true;
      };
  const auto readiness_is_qualified =
      [](const hal::ReadinessSnapshot &readiness) {
        return readiness.available &&
               (readiness.status == hal::StatusId::ready ||
                readiness.status == hal::StatusId::degraded ||
                readiness.status == hal::StatusId::executed) &&
               readiness.selected_backend != hal::BackendId::invalid &&
               readiness.provider !=
                   hal::SemanticEffectProviderId::invalid &&
               hal::ValidateExecutionDomain(readiness.execution_domain);
      };

  if (input.maxine_resident_live) {
    if (!input.maxine_resident.prepared ||
        input.maxine_resident.route !=
            PreparedMaxineHardwareRoute::hal_live ||
        input.maxine_resident.eligibility !=
            PreparedMaxineHardwareEligibility::eligible ||
        !readiness_is_qualified(input.maxine_resident.readiness) ||
        input.maxine_resident.quarantined ||
        input.maxine_resident.stopped ||
        !add_program_source(
            PreparedExecutionRouteId::maxine_resident,
            input.maxine_resident.active_program,
            input.maxine_resident.readiness,
            input.maxine_resident.telemetry,
            input.maxine_resident.latest_committed_frame)) {
      return reject_malformed_source(
          MultiIslandStatusValidationCode::invalid_island);
    }
  }
  if (input.open_cuda_virtual_background_live) {
    if (!input.open_cuda_virtual_background.prepared ||
        input.open_cuda_virtual_background.route !=
            PreparedOpenCudaHardwareRoute::hal_live ||
        input.open_cuda_virtual_background.eligibility !=
            PreparedOpenCudaHardwareEligibility::eligible ||
        !input.open_cuda_virtual_background.production
             .prepared_capability_committed ||
        !readiness_is_qualified(
            input.open_cuda_virtual_background.readiness) ||
        input.open_cuda_virtual_background.quarantined ||
        input.open_cuda_virtual_background.stopped ||
        input.open_cuda_virtual_background.shutdown_pending ||
        !add_program_source(
            PreparedExecutionRouteId::open_cuda_virtual_background,
            input.open_cuda_virtual_background.active_program,
            input.open_cuda_virtual_background.readiness,
            input.open_cuda_virtual_background.telemetry,
            input.open_cuda_virtual_background.latest_committed_frame)) {
      return reject_malformed_source(
          MultiIslandStatusValidationCode::invalid_island);
    }
    const auto &program =
        input.open_cuda_virtual_background.active_program;
    for (std::size_t index = 0; index < program.stage_count; ++index) {
      const hal::SemanticEffectId semantic = program.stages[index].semantic;
      if (semantic != hal::SemanticEffectId::virtual_background_blur &&
          semantic != hal::SemanticEffectId::virtual_background_remove &&
          semantic != hal::SemanticEffectId::virtual_background_replace) {
        return reject_malformed_source(
            MultiIslandStatusValidationCode::invalid_capability_evidence);
      }
    }
  }

  hal::ProgramGenerations generations;
  if (source_count != 0) {
    generations = sources[0].program.invalidation_key.generations;
  } else {
    generations.configuration =
        input.configuration_generation == 0 ? 1
                                            : input.configuration_generation;
    generations.geometry =
        input.geometry_generation == 0 ? 1 : input.geometry_generation;
    generations.model = 1;
    generations.provider = 1;
    generations.device = 1;
    generations.context = 1;
  }
  build.program_generations = generations;
  build.generations.capability = input.status_generation;
  build.generations.configuration = generations.configuration;
  build.generations.provider = generations.provider;
  build.generations.device = generations.device;
  build.generations.context = generations.context;
  CapabilityRefreshSnapshot refresh;
  refresh.status = CapabilityRefreshStatus::succeeded;
  refresh.code = CapabilityCode::capability_snapshot_current;
  refresh.requested_generation = input.status_generation;
  refresh.completed_generation = input.status_generation;
  build.capabilities =
      MakeFailClosedProductCapabilitySnapshot(build.generations, refresh);

  for (std::size_t source_index = 0; source_index < source_count;
       ++source_index) {
    const IslandSource &source = sources[source_index];
    if (build.island_count >= build.islands.size())
      break;
    PreparedExecutionIslandStatus &island =
        build.islands[build.island_count++];
    island.route = source.route;
    island.island_id =
        (static_cast<std::uint64_t>(source.route) << 56u) ^
        source.program.program_id;
    if (island.island_id == 0)
      island.island_id = source.program.program_id;
    island.adapter_instance_id =
        source.program.owner_adapter_instance_id;
    island.snapshot_generation = input.status_generation;
    island.capability_generation = input.status_generation;
    island.backend = source.program.execution_domain.backend;
    island.execution_domain = source.program.execution_domain;
    island.resolution = source.resolution;
    island.readiness.configured = true;
    island.readiness.program_id = source.program.program_id;
    island.readiness.invalidation_key =
        source.program.invalidation_key;
    island.readiness.value = source.readiness;
    island.telemetry = source.telemetry;
    island.latest_committed_frame = source.committed;
    island.valid = true;
    island.current = true;
    island.stage_count = source.resolution.stage_count;

    for (std::size_t stage_index = 0;
         stage_index < island.stage_count; ++stage_index) {
      const auto &resolved = source.resolution.stages[stage_index];
      auto &record = island.stages[stage_index];
      record.stage_id = resolved.intent.stage_id;
      record.effect = resolved.intent.semantic;
      record.backend = resolved.selected_backend;
      record.provider = resolved.provider;
      record.status = resolved.status;
      record.reason = resolved.reason;
      record.failure_action = resolved.intent.failure_action;
      record.required = resolved.intent.required;
      record.resolved = true;
      for (std::size_t committed_index = 0;
           committed_index < source.committed.stage_count;
           ++committed_index) {
        const auto &committed = source.committed.stages[committed_index];
        if (committed.stage_id == record.stage_id &&
            committed.semantic == record.effect) {
          record.committed = true;
          record.committed_status = committed.status;
          break;
        }
      }

      const Lane lane = lane_for_backend(record.backend);
      ProductCapabilityLane *capability = FindProductCapabilityLane(
          &build.capabilities, record.effect, lane);
      const CapabilityCode selected_code =
          capability_code(lane, record.effect);
      if (!capability || lane == Lane::invalid || lane == Lane::cpu ||
          selected_code == CapabilityCode::none) {
        continue;
      }
      capability->disposition =
          resolved.status == hal::StatusId::degraded
              ? ProductCapabilityDisposition::
                    usable_with_degraded_behavior
              : ProductCapabilityDisposition::production_usable;
      capability->code = selected_code;
      capability->nested_code = CapabilityCode::none;
      capability->semantic.semantic = record.effect;
      capability->semantic.provider = record.provider;
      capability->semantic.status = resolved.status;
      capability->semantic.reason = resolved.reason;
      capability->semantic.cpu_tail = resolved.cpu_tail;
      capability->semantic.cpu_tail_scratch_bytes =
          resolved.cpu_tail_scratch_bytes;
      capability->semantic.cpu_tail_placement =
          resolved.cpu_tail_placement;
      capability->semantic.live_stage = true;
      capability->semantic.production_eligible = true;
      capability->semantic.exact_evidence = true;
      capability->semantic.supports_tracked_vignette_center = false;
      capability->semantic.evidence_count =
          std::min(resolved.evidence_count,
                   capability->semantic.evidence.size());
      for (std::size_t evidence_index = 0;
           evidence_index < capability->semantic.evidence_count;
           ++evidence_index) {
        capability->semantic.evidence[evidence_index] =
            resolved.evidence[evidence_index];
      }
    }
  }

  bool have_writer_commit = false;
  for (std::size_t island_index = 0; island_index < build.island_count;
       ++island_index) {
    const auto &committed =
        build.islands[island_index].latest_committed_frame;
    if (!committed.valid)
      continue;
    const WriterOutputCommitIdentity identity{
        true, committed.capture_sequence,
        committed.output_commit_sequence};
    if (!have_writer_commit) {
      build.writer_commit = identity;
      have_writer_commit = true;
    } else if (build.writer_commit != identity) {
      // Preserve every current typed island and let the shared validator
      // reject mixed output commits without publishing partial attribution.
      build.writer_commit = identity;
    }
  }

  for (std::size_t effect_index = 0; effect_index < build.effect_count;
       ++effect_index) {
    StableEffectRuntimeState &state = build.effects[effect_index];
    if (!requested(state.effect))
      continue;

    const hal::BackendId desired = desired_backend(state.effect);
    const Lane desired_lane = lane_for_backend(desired);
    const ProductCapabilityLane *desired_capability =
        FindProductCapabilityLane(build.capabilities, state.effect,
                                  desired_lane);
    state.requested = true;
    state.requested_backend = desired;
    state.requested_provider = requested_provider(desired, state.effect);
    state.capability_lane = desired_lane;
    if (desired_capability) {
      state.capability_code = desired_capability->code;
      state.nested_capability_code = desired_capability->nested_code;
      state.capability = desired_capability->semantic;
      state.available =
          IsProductCapabilityLaneEligible(build.capabilities,
                                          *desired_capability);
    }
    state.strict_backend_selection =
        input.effects->engine !=
            effects::EffectsEnginePreference::auto_select ||
        input.compute_backend == ComputeBackendPreference::vulkan;
    state.strict_provider_selection =
        input.effects->engine ==
        effects::EffectsEnginePreference::open_cuda;
    state.status = hal::StatusId::unavailable;
    state.reason = hal::ReasonId::capability_evidence_incomplete;
    state.blocked = true;

    for (std::size_t island_index = 0;
         island_index < build.island_count; ++island_index) {
      const auto &island = build.islands[island_index];
      for (std::size_t stage_index = 0;
           stage_index < island.stage_count; ++stage_index) {
        const auto &stage = island.stages[stage_index];
        if (stage.effect != state.effect)
          continue;
        const ProductCapabilityLaneId selected_lane =
            lane_for_backend(stage.backend);
        const ProductCapabilityLane *selected_capability =
            FindProductCapabilityLane(build.capabilities, state.effect,
                                      selected_lane);
        if (!selected_capability)
          continue;
        // StableEffectRuntimeState represents the user's canonical request,
        // not the adapter-local exact choice made after automatic resolution.
        // Automatic engine intent uses Maxine as the documented first
        // preference but remains non-strict, so a later selected lane is
        // reported as fallback rather than a strict substitution.
        state.strict_backend_selection =
            (state.requested_backend == hal::BackendId::vulkan &&
             input.compute_backend == ComputeBackendPreference::vulkan) ||
            (state.requested_backend == hal::BackendId::maxine &&
             input.effects->engine ==
                 effects::EffectsEnginePreference::maxine) ||
            ((state.requested_backend == hal::BackendId::cuda ||
              state.requested_backend ==
                  hal::BackendId::open_video_onnxruntime) &&
             input.effects->engine ==
                 effects::EffectsEnginePreference::open_cuda);
        state.strict_provider_selection =
            input.effects->engine ==
            effects::EffectsEnginePreference::open_cuda;
        state.capability_lane =
            lane_for_backend(state.requested_backend);
        const ProductCapabilityLane *requested_capability =
            FindProductCapabilityLane(build.capabilities, state.effect,
                                      state.capability_lane);
        if (requested_capability) {
          state.capability_code = requested_capability->code;
          state.nested_capability_code =
              requested_capability->nested_code;
          state.capability = requested_capability->semantic;
        }
        state.selected_capability_lane = selected_lane;
        state.selected_capability_code = selected_capability->code;
        state.selected_nested_capability_code =
            selected_capability->nested_code;
        state.selected_capability = selected_capability->semantic;
        state.selected_backend = stage.backend;
        state.selected_provider = stage.provider;
        state.status = stage.status;
        state.reason = stage.reason;
        state.failure_action = stage.failure_action;
        state.required = stage.required;
        state.available = true;
        state.selected = true;
        state.blocked = false;
        state.selected_island_id = island.island_id;
        state.fallback =
            state.selected_backend != state.requested_backend ||
            (state.requested_provider !=
                 hal::SemanticEffectProviderId::invalid &&
             state.selected_provider != state.requested_provider);
        if (stage.committed) {
          state.commit = build.writer_commit;
          state.committed_stage_id = stage.stage_id;
          state.committed_stage_status = stage.committed_status;
          if (stage.committed_status ==
              hal::StageExecutionStatus::succeeded) {
            const bool runtime_healthy =
                island.telemetry.runtime_status == hal::StatusId::ready ||
                island.telemetry.runtime_status == hal::StatusId::degraded ||
                island.telemetry.runtime_status == hal::StatusId::executed;
            state.active =
                input.lifecycle == VideoRuntimeLifecycle::running &&
                build.writer_commit.valid && runtime_healthy;
            if (state.active) {
              state.active_backend = stage.backend;
              state.active_provider = stage.provider;
              state.status = hal::StatusId::executed;
              state.reason = hal::ReasonId::none;
            }
          } else if (stage.committed_status ==
                     hal::StageExecutionStatus::optional_failed_open) {
            state.degraded = true;
            state.fallback = true;
            state.status = hal::StatusId::degraded;
            state.reason =
                hal::ReasonId::optional_stage_failed_open;
          }
        }
      }
    }
  }

  return BuildMultiIslandVideoRuntimeStatusSnapshot(build, output);
}

int OptionalEffectBreaker::CooldownFramesForFailureCount(int failure_count) {
  if (failure_count <= 0)
    return 0;

  int cooldown = kInitialCooldownFrames;
  for (int i = 1; i < failure_count && cooldown < kMaxCooldownFrames; ++i) {
    if (cooldown > (kMaxCooldownFrames / 2)) {
      cooldown = kMaxCooldownFrames;
    } else {
      cooldown *= 2;
    }
  }
  return std::min(cooldown, kMaxCooldownFrames);
}

bool OptionalEffectBreaker::AllowsAttempt(std::uint64_t frame_index) const {
  return !active_ || frame_index >= retry_frame_index_;
}

void OptionalEffectBreaker::Reset() {
  active_ = false;
  effect_id_.clear();
  backend_.clear();
  reason_.clear();
  failure_count_ = 0;
  cooldown_frames_ = 0;
  retry_frame_index_ = 0;
  trip_order_ = 0;
  retry_ready_published_ = false;
}

void OptionalEffectBreaker::OnFailure(std::string_view effect,
                                      std::string_view effect_backend,
                                      std::string failure_reason,
                                      std::uint64_t frame_index, int order) {
  active_ = true;
  effect_id_ = std::string(effect);
  backend_ = std::string(effect_backend);
  reason_ = failure_reason.empty() ? std::string("Effect failed.")
                                   : std::move(failure_reason);
  ++failure_count_;
  cooldown_frames_ = CooldownFramesForFailureCount(failure_count_);
  const auto cooldown = static_cast<std::uint64_t>(cooldown_frames_);
  retry_frame_index_ =
      frame_index > (std::numeric_limits<std::uint64_t>::max() - cooldown)
          ? std::numeric_limits<std::uint64_t>::max()
          : frame_index + cooldown;
  trip_order_ = order;
  retry_ready_published_ = false;
}

bool OptionalEffectBreaker::OnSuccess() {
  if (!active_)
    return false;
  Reset();
  return true;
}

bool OptionalEffectBreaker::MarkRetryReadyIfDue(std::uint64_t frame_index) {
  if (!active_ || retry_ready_published_ || !AllowsAttempt(frame_index))
    return false;
  retry_ready_published_ = true;
  return true;
}

CameraPipelineStatus::DegradedEffect
OptionalEffectBreaker::ToStatus(std::uint64_t frame_index) const {
  CameraPipelineStatus::DegradedEffect s;
  if (!active_)
    return s;

  s.active = true;
  s.effect_id = effect_id_;
  s.backend = backend_;
  s.reason = reason_;
  s.failure_count = failure_count_;
  if (AllowsAttempt(frame_index)) {
    s.state = kOptionalEffectRetryReadyState;
    s.cooldown_frames = 0;
  } else {
    s.state = kOptionalEffectCoolingDownState;
    const auto remaining = retry_frame_index_ - frame_index;
    s.cooldown_frames =
        remaining > static_cast<std::uint64_t>(std::numeric_limits<int>::max())
            ? std::numeric_limits<int>::max()
            : static_cast<int>(remaining);
  }
  return s;
}

CameraPipeline::CameraPipeline() {
  CameraPipelineHalStatusProjectionInput projection;
  projection.effects = &effects_;
  projection.lifecycle = VideoRuntimeLifecycle::disabled;
  projection.status_generation = ++video_hal_status_generation_;
  projection.configuration_generation = 1;
  projection.geometry_generation = 1;
  (void)BuildCameraPipelineHalStatusSnapshot(projection,
                                             &video_hal_status_);
}

CameraPipeline::~CameraPipeline() { Stop(); }

void CameraPipeline::SetMirrorEnabled(bool enabled) {
  std::lock_guard<std::mutex> update_lock(effects_update_mu_);
  std::uint64_t published_generation = 0;
  {
    std::lock_guard<std::mutex> lock(effects_mu_);
    if (effects_.mirror == enabled)
      return;
    effects_generation_published_.store(0, std::memory_order_release);
    effects_.mirror = enabled;
    published_generation = ++effects_generation_;
  }
  effects_generation_published_.store(published_generation,
                                      std::memory_order_release);
}

void CameraPipeline::SetEffects(
    const studiocast::video::effects::BroadcastCameraEffects &effects) {
  std::lock_guard<std::mutex> update_lock(effects_update_mu_);

  detail::PreparedReplaceBackgroundSource replace_source;
  bool prepare_replace_source = false;
  {
    std::lock_guard<std::mutex> lock(effects_mu_);
    ++effects_set_requests_;
    if (effects_ == effects) {
      ++effects_ignored_updates_;
      return;
    }
    effects_generation_published_.store(0, std::memory_order_release);

    if (CanReusePreparedReplaceBackgroundSource(effects_, effects)) {
      replace_source = replace_background_source_;
    } else {
      prepare_replace_source =
          effects.virtual_background.mode ==
          studiocast::video::effects::VirtualBackgroundMode::replace;
    }
  }

  // Genuine configuration preparation may touch the filesystem. Keep it
  // outside effects_mu_ so the frame thread can continue using the currently
  // published generation while the replacement source is prepared.
  if (prepare_replace_source) {
    replace_source = PrepareReplaceBackgroundSourceForEffects(effects);
    std::lock_guard<std::mutex> lock(effects_mu_);
    ++replacement_background_preparations_;
  }

  std::uint64_t published_generation = 0;
  {
    std::lock_guard<std::mutex> lock(effects_mu_);
    effects_ = effects;
    replace_background_source_ = std::move(replace_source);
    published_generation = ++effects_generation_;
    ++effects_applied_updates_;
  }
  effects_generation_published_.store(published_generation,
                                      std::memory_order_release);
}

void CameraPipeline::RefreshEffectsResources() {
  std::lock_guard<std::mutex> update_lock(effects_update_mu_);

  studiocast::video::effects::BroadcastCameraEffects effects;
  {
    std::lock_guard<std::mutex> lock(effects_mu_);
    effects = effects_;
    effects_generation_published_.store(0, std::memory_order_release);
  }

  const bool prepare_replace_source =
      effects.virtual_background.mode ==
      studiocast::video::effects::VirtualBackgroundMode::replace;
  auto replace_source = prepare_replace_source
                            ? PrepareReplaceBackgroundSourceForEffects(effects)
                            : detail::PreparedReplaceBackgroundSource{};

  std::uint64_t published_generation = 0;
  {
    std::lock_guard<std::mutex> lock(effects_mu_);
    if (prepare_replace_source)
      ++replacement_background_preparations_;
    replace_background_source_ = std::move(replace_source);
    published_generation = ++effects_generation_;
    ++effects_resource_refreshes_;
  }
  effects_generation_published_.store(published_generation,
                                      std::memory_order_release);
}

CameraPipelineStatus CameraPipeline::Status() const {
  std::lock_guard<std::mutex> lock(mu_);
  CameraPipelineStatus s;
  s.running = running_;
  s.starting = starting_;
  s.input_device = input_device_;
  s.output_device = output_device_;
  if (running_ || starting_) {
    s.capture = capture_;
    s.output = output_;
    s.capture_fallback_state = capture_fallback_state_;
    s.capture_fallback_reason = capture_fallback_reason_;
    s.scaling_backend_active = scaling_backend_active_;
    s.scaling_from = scaling_from_;
    s.scaling_to = scaling_to_;
    s.compute_backend_preference = compute_backend_preference_;
    s.compute_backend_resolved = compute_backend_resolved_;
    s.compute_backend_active = compute_backend_active_;
    s.compute_backend_fallback_reason = compute_backend_fallback_reason_;
    s.compute_backend_degraded_reason = compute_backend_degraded_reason_;
    s.ms_per_frame = ms_per_frame_;
    s.fps_actual = fps_actual_;
    s.perf_sample_frames = perf_sample_frames_;
    s.debug = debug_;
    s.open_cuda_transfers = open_cuda_transfers_;
    s.open_vulkan_transfers = open_vulkan_transfers_;
    s.maxine_transfers = maxine_transfers_;
    s.prepared_hardware_execution = prepared_hardware_execution_;
    s.prepared_open_cuda_hardware_execution =
        prepared_open_cuda_hardware_execution_;
    s.prepared_maxine_hardware_execution =
        prepared_maxine_hardware_execution_;
    s.video_hal_status = video_hal_status_;
  } else {
    // Avoid exposing stale negotiated formats when the pipeline is idle.
    //
    // Exception: if the loopback writer is held open
    // (v4l2loopback/exclusive_caps), expose the negotiated output format so
    // consumers (including the GUI preview) can open the device without
    // attempting to renegotiate caps.
    s.capture = CaptureFormat{};
    s.output = writer_.IsOpen() ? writer_.Actual() : ActualFormat{};
    s.capture_fallback_state = "none";
    s.capture_fallback_reason.clear();
    s.scaling_backend_active.clear();
    s.scaling_from = CaptureFormat{};
    s.scaling_to = ActualFormat{};
    s.compute_backend_preference = compute_backend_preference_;
    s.compute_backend_resolved = compute_backend_resolved_;
    s.compute_backend_active = "cpu";
    s.compute_backend_fallback_reason = compute_backend_fallback_reason_;
    s.compute_backend_degraded_reason = compute_backend_degraded_reason_;
    s.ms_per_frame = CameraPipelineStatus::MsPerFrame{};
    s.fps_actual = 0.0;
    s.perf_sample_frames = 0;
    s.debug = CameraPipelineStatus::Debug{};
    s.open_cuda_transfers = CameraPipelineStatus::OpenCudaTransfers{};
    s.open_vulkan_transfers = CameraPipelineStatus::OpenVulkanTransfers{};
    s.maxine_transfers = CameraPipelineStatus::MaxineTransfers{};
    s.prepared_hardware_execution = PreparedHardwareExecutionSnapshot{};
    s.prepared_open_cuda_hardware_execution =
        PreparedOpenCudaHardwareExecutionSnapshot{};
    s.prepared_maxine_hardware_execution =
        PreparedMaxineHardwareExecutionSnapshot{};
    s.video_hal_status = video_hal_status_;
  }
  s.frame_index = frame_index_;
  // Live attribution is meaningful only while a successfully-started pipeline
  // is running. Starting/stopped status must never expose setup or prior-run
  // backend choices.
  s.effects_backends =
      running_ && !starting_ ? effects_backends_ : std::string{};
  s.effects_note = effects_note_;
  s.degraded_effect = degraded_effect_;
  s.last_error = last_error_;
  return s;
}

bool CameraPipeline::Start(const CameraPipelineConfig &cfg,
                           std::string *error) {
  const bool w_set = cfg.width > 0;
  const bool h_set = cfg.height > 0;
  if (cfg.capture_mode == CaptureMode::requested) {
    if (!w_set || !h_set) {
      if (error)
        *error = "Invalid width/height.";
      return false;
    }
  } else {
    // In auto capture mode, width/height are optional (sentinel <= 0). If one
    // is set, both must be set.
    if (w_set != h_set) {
      if (error)
        *error = "Invalid width/height (must both be >0 or both be <=0 in auto "
                 "capture mode).";
      return false;
    }
  }
  if (cfg.fps <= 0 || cfg.fps > 240) {
    if (error)
      *error = "Invalid fps (1..240).";
    return false;
  }

  std::unique_lock<std::mutex> effects_update_lock(effects_update_mu_);
  const auto replace_source =
      PrepareReplaceBackgroundSourceForEffects(cfg.effects);
  const bool prepared_replace_source =
      cfg.effects.virtual_background.mode ==
      studiocast::video::effects::VirtualBackgroundMode::replace;

  std::unique_lock<std::mutex> lock(mu_);
  if (running_ || starting_) {
    if (error)
      *error = "Camera pipeline already running.";
    return false;
  }

  if (th_.joinable()) {
    lock.unlock();
    Stop();
    lock.lock();
  }

  stop_.store(false);

  {
    std::lock_guard<std::mutex> fxLock(effects_mu_);
    if (prepared_replace_source)
      ++replacement_background_preparations_;
    effects_ = cfg.effects;
    replace_background_source_ = replace_source;
    ++effects_generation_;
    effects_generation_published_.store(effects_generation_,
                                        std::memory_order_release);
  }

  last_error_.clear();
  input_device_.clear();
  output_device_.clear();
  capture_ = CaptureFormat{};
  output_ = ActualFormat{};
  capture_fallback_state_ = "none";
  capture_fallback_reason_.clear();
  scaling_backend_active_.clear();
  scaling_from_ = CaptureFormat{};
  scaling_to_ = ActualFormat{};
  compute_backend_preference_ =
      ComputeBackendPreferenceToString(cfg.compute_backend);
  compute_backend_resolved_ = "cpu";
  compute_backend_active_ = "cpu";
  compute_backend_fallback_reason_.clear();
  compute_backend_degraded_reason_.clear();
  frame_index_ = 0;
  ms_per_frame_ = CameraPipelineStatus::MsPerFrame{};
  fps_actual_ = 0.0;
  perf_sample_frames_ = 0;
  debug_ = CameraPipelineStatus::Debug{};
  open_cuda_transfers_ = CameraPipelineStatus::OpenCudaTransfers{};
  open_vulkan_transfers_ = CameraPipelineStatus::OpenVulkanTransfers{};
  maxine_transfers_ = CameraPipelineStatus::MaxineTransfers{};
  prepared_hardware_execution_ = PreparedHardwareExecutionSnapshot{};
  prepared_open_cuda_hardware_execution_ =
      PreparedOpenCudaHardwareExecutionSnapshot{};
  prepared_maxine_hardware_execution_ =
      PreparedMaxineHardwareExecutionSnapshot{};
  CameraPipelineHalStatusProjectionInput starting_projection;
  starting_projection.effects = &cfg.effects;
  starting_projection.compute_backend = cfg.compute_backend;
  starting_projection.lifecycle = VideoRuntimeLifecycle::starting;
  ++video_hal_status_generation_;
  if (video_hal_status_generation_ == 0)
    ++video_hal_status_generation_;
  starting_projection.status_generation = video_hal_status_generation_;
  starting_projection.configuration_generation = effects_generation_;
  starting_projection.geometry_generation = 1;
  (void)BuildCameraPipelineHalStatusSnapshot(starting_projection,
                                             &video_hal_status_);

  effects_backends_.clear();
  effects_note_.clear();
  degraded_effect_ = CameraPipelineStatus::DegradedEffect{};

  starting_ = true;
  running_ = false;
  start_notified_ = false;

  th_ = std::thread(&CameraPipeline::ThreadMain, this, cfg);

  cv_.wait(lock, [&]() { return start_notified_; });
  starting_ = false;

  if (!running_) {
    const std::string err =
        last_error_.empty() ? "Failed to start camera pipeline." : last_error_;
    lock.unlock();
    if (th_.joinable())
      th_.join();
    lock.lock();
    CameraPipelineHalStatusProjectionInput failed_projection;
    failed_projection.effects = &cfg.effects;
    failed_projection.compute_backend = cfg.compute_backend;
    failed_projection.lifecycle = VideoRuntimeLifecycle::failed;
    ++video_hal_status_generation_;
    if (video_hal_status_generation_ == 0)
      ++video_hal_status_generation_;
    failed_projection.status_generation = video_hal_status_generation_;
    failed_projection.configuration_generation =
        effects_generation_ == 0 ? 1 : effects_generation_;
    failed_projection.geometry_generation = 1;
    (void)BuildCameraPipelineHalStatusSnapshot(failed_projection,
                                               &video_hal_status_);
    lock.unlock();
    if (error)
      *error = err;
    return false;
  }

  return true;
}

bool CameraPipeline::OpenOutputLocked(const std::string &outDev, int width,
                                      int height, int fps,
                                      PixelFormat output_format,
                                      bool strict_fps,
                                      bool *out_opened_or_renegotiated,
                                      std::string *error) {
  if (out_opened_or_renegotiated)
    *out_opened_or_renegotiated = false;

  std::string reopen_reason;

  // Try to reuse an existing open writer when possible.
  if (writer_.IsOpen() && writer_device_ == outDev) {
    // Refresh cached negotiated format.
    //
    // v4l2loopback allows capture clients to renegotiate global caps via
    // VIDIOC_S_FMT. If that happens, size_image/bytes_per_line can change under
    // an already-open writer fd. Refreshing avoids write() failures and
    // start/stop thrashing. NOTE: RefreshActual() is best-effort. Some
    // v4l2loopback configurations can temporarily reject VIDIOC_G_FMT (e.g.
    // when consumers disconnect). Do not tear down a working writer fd just
    // because the refresh query fails.
    std::string refresh_err;
    (void)writer_.RefreshActual(&refresh_err);

    const auto &a = writer_.Actual();

    // Idle keep-alive path (strict_fps=false): keep a matching producer FD
    // alive.
    //
    // Some v4l2loopback configurations transiently report/accept different
    // formats while no capture clients are connected. If we aggressively close
    // and renegotiate here, we can hit a window where VIDIOC_S_FMT/G_FMT
    // returns EINVAL, which manifests as start/stop flapping when the GUI
    // preview is restarted after the camera fully idles.
    //
    // Still enforce the configured shared producer format while idle so browser
    // consumers see the selected FourCC before opening the device.
    if (!strict_fps &&
        ActualMatchesOutputRequest(a, width, height, fps, output_format,
                                   /*strict_fps=*/false)) {
      output_ = a;
      output_device_ = outDev;
      return true;
    }

    if (ActualMatchesOutputRequest(a, width, height, fps, output_format,
                                   strict_fps)) {
      output_ = a;
      output_device_ = outDev;
      return true;
    }

    // Attempt in-place renegotiation on the existing producer FD.
    //
    // Closing/re-opening the writer FD is a known destabilizer for some
    // v4l2loopback configurations (especially around consumer disconnect
    // windows). Prefer applying VIDIOC_S_FMT/VIDIOC_S_PARM on the same FD.
    std::string nerr;
    if (writer_.Renegotiate(outDev, width, height, fps, output_format, &nerr) &&
        ActualMatchesOutputRequest(writer_.Actual(), width, height, fps,
                                   output_format, strict_fps)) {
      if (out_opened_or_renegotiated)
        *out_opened_or_renegotiated = true;
      output_ = writer_.Actual();
      output_device_ = outDev;
      return true;
    }

    if (nerr.empty() && writer_.IsOpen()) {
      nerr = ActualOutputMismatchMessage(writer_.Actual(), width, height, fps,
                                         output_format, strict_fps);
    }

    reopen_reason = "Renegotiation failed for requested output_format=" +
                    PixelFormatName(output_format) + ": " + nerr;
    writer_.Close();
    writer_device_.clear();
    output_ = ActualFormat{};
    output_device_.clear();
  } else if (writer_.IsOpen() && writer_device_ != outDev) {
    writer_.Close();
    writer_device_.clear();
  }

  // Open writer with the daemon-owned requested shared producer format.
  if (out_opened_or_renegotiated)
    *out_opened_or_renegotiated = true;

  // v4l2loopback can transiently reject format negotiation (EINVAL) right after
  // the producer FD was closed (or during consumer disconnect windows). Retry
  // for a short budget with backoff to avoid supervisor thrash (and to make
  // fast restarts more reliable).
  constexpr auto kOpenRetryBudget = std::chrono::seconds(5);
  constexpr auto kOpenRetrySleepMin = std::chrono::milliseconds(50);
  constexpr auto kOpenRetrySleepMax = std::chrono::milliseconds(500);

  auto contains_invalid_argument = [](const std::string &s) {
    return s.find("Invalid argument") != std::string::npos;
  };

  std::string werr;
  bool opened = false;
  int attempts = 0;

  const auto t0 = std::chrono::steady_clock::now();
  auto sleep = kOpenRetrySleepMin;
  auto sleep_or_stop = [this](std::chrono::milliseconds d) {
    constexpr auto kStopCheckQuantum = std::chrono::milliseconds(25);
    auto remaining = d;
    while (!stop_.load() && remaining > std::chrono::milliseconds::zero()) {
      const auto chunk = std::min(remaining, kStopCheckQuantum);
      std::this_thread::sleep_for(chunk);
      remaining -= chunk;
    }
  };
  for (int attempt = 0;; ++attempt) {
    attempts = attempt + 1;
    werr.clear();

    if (stop_.load())
      break;

    if (writer_.Open(outDev, width, height, fps, output_format, &werr)) {
      if (ActualMatchesOutputRequest(writer_.Actual(), width, height, fps,
                                     output_format, strict_fps)) {
        opened = true;
        break;
      }
      werr = ActualOutputMismatchMessage(writer_.Actual(), width, height, fps,
                                         output_format, strict_fps);
      writer_.Close();
    }

    const bool maybe_transient = contains_invalid_argument(werr);
    if (!maybe_transient)
      break;

    if ((std::chrono::steady_clock::now() - t0) >= kOpenRetryBudget)
      break;

    sleep_or_stop(sleep);
    sleep = std::min(sleep * 2, kOpenRetrySleepMax);
  }

  if (!opened) {
    if (error) {
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - t0)
              .count();
      *error =
          "Failed to open v4l2loopback output " + outDev +
          " with requested output_format=" + PixelFormatName(output_format) +
          ":\n";
      if (!reopen_reason.empty())
        *error += reopen_reason + "\n\n";
      *error += werr + "\n\n" + "(open retries: " + std::to_string(attempts) +
                ", elapsed=" + std::to_string(elapsed) + "ms)";

      if (contains_invalid_argument(werr)) {
        *error +=
            "\nHint: this can be a transient v4l2loopback restart window; "
            "if it persists, try waiting a few seconds or reloading the "
            "module.";
      }
    }
    return false;
  }

  writer_device_ = outDev;
  output_ = writer_.Actual();
  output_device_ = outDev;
  return true;
}

bool CameraPipeline::EnsureOutputOpen(const CameraPipelineConfig &cfg,
                                      std::string *error) {
  const bool w_set = cfg.width > 0;
  const bool h_set = cfg.height > 0;
  if (cfg.capture_mode == CaptureMode::requested) {
    if (!w_set || !h_set) {
      if (error)
        *error = "Invalid width/height.";
      return false;
    }
  } else {
    // In auto capture mode, width/height are optional (sentinel <= 0). If one
    // is set, both must be set.
    if (w_set != h_set) {
      if (error)
        *error = "Invalid width/height (must both be >0 or both be <=0 in auto "
                 "capture mode).";
      return false;
    }
  }
  if (cfg.fps <= 0 || cfg.fps > 240) {
    if (error)
      *error = "Invalid fps (1..240).";
    return false;
  }

  std::unique_lock<std::mutex> lock(mu_);
  if (running_ || starting_) {
    // Output is already open (or will be shortly) as part of Start().
    return true;
  }

  const auto now = std::chrono::steady_clock::now();
  constexpr auto kIdleKeepaliveInterval = std::chrono::milliseconds(1000);
  const auto write_keepalive = [&](bool force, std::string *outErr) -> bool {
    if (!writer_.IsOpen())
      return true;
    if (!force &&
        last_keepalive_frame_at_ != std::chrono::steady_clock::time_point{} &&
        (now - last_keepalive_frame_at_) < kIdleKeepaliveInterval) {
      return true;
    }

    ActualFormat outA = writer_.Actual();
    std::size_t size = outA.size_image;
    if (size == 0 && outA.bytes_per_line > 0 && outA.height > 0) {
      size = outA.bytes_per_line * static_cast<std::size_t>(outA.height);
    }
    if (size == 0)
      return true;

    const bool need_refill =
        (keepalive_frame_.size() != size) ||
        (keepalive_format_.width != outA.width) ||
        (keepalive_format_.height != outA.height) ||
        (keepalive_format_.pixfmt_fourcc != outA.pixfmt_fourcc) ||
        (keepalive_format_.bytes_per_line != outA.bytes_per_line) ||
        (keepalive_format_.size_image != outA.size_image) ||
        (keepalive_format_.format != outA.format);

    if (need_refill) {
      keepalive_frame_.assign(size, 0);
      if (outA.format == PixelFormat::yuyv) {
        // YUYV "black": Y=16, U=128, V=128 (limited range neutral chroma)
        for (std::size_t i = 0; i + 3 < keepalive_frame_.size(); i += 4) {
          keepalive_frame_[i + 0] = 16;
          keepalive_frame_[i + 1] = 128;
          keepalive_frame_[i + 2] = 16;
          keepalive_frame_[i + 3] = 128;
        }
      }
      keepalive_format_ = outA;
    }

    const FrameWriteResult write_result = writer_.WriteFrameDetailed(
        keepalive_frame_.data(), keepalive_frame_.size());
    if (write_result.status == FrameWriteStatus::would_block_dropped) {
      // Keepalive has no queue. A later supervisor tick supplies the newest
      // frame without treating output backpressure as device failure.
      last_keepalive_frame_at_ = now;
      return true;
    }
    if (!write_result.FrameCommitted()) {
      const std::string werr = DescribeFrameWriteResult(write_result);
      const std::string msg = "Failed to write keepalive frame: " + werr;
      writer_.Close();
      writer_device_.clear();
      output_device_.clear();
      output_ = ActualFormat{};
      keepalive_format_ = ActualFormat{};
      keepalive_frame_.clear();
      last_keepalive_frame_at_ = std::chrono::steady_clock::time_point{};
      if (outErr)
        *outErr = msg;
      return false;
    }
    last_keepalive_frame_at_ = now;
    return true;
  };

  std::string outDev = cfg.output_device;
  if (outDev.empty()) {
    std::string e;
    outDev = ChooseDefaultOutputLoopback(&e);
    if (outDev.empty()) {
      if (error)
        *error = e.empty() ? "No output loopback found." : e;
      return false;
    }
  }

  // If capture is auto and there is no explicit output override, we can't know
  // what output dimensions to choose until capture is negotiated in
  // ThreadMain(). Treat this as best-effort: keep any already-open writer,
  // otherwise defer without error.
  if (cfg.capture_mode == CaptureMode::auto_best && (!w_set || !h_set)) {
    if (writer_.IsOpen() && writer_device_ == outDev) {
      std::string refresh_err;
      (void)writer_.RefreshActual(&refresh_err);
      output_ = writer_.Actual();
      output_device_ = outDev;

      std::string kerr;
      if (!write_keepalive(false, &kerr)) {
        if (error)
          *error = kerr;
        return false;
      }
    }
    last_error_.clear();
    return true;
  }

  bool opened_or_renegotiated = false;
  std::string oerr;
  if (!OpenOutputLocked(outDev, cfg.width, cfg.height, cfg.fps,
                        cfg.output_format, false, &opened_or_renegotiated,
                        &oerr)) {
    if (error)
      *error = oerr;
    return false;
  }

  last_error_.clear();

  // While idle, keep the loopback alive by periodically writing a black frame.
  //
  // This helps prevent consumer probing loops (open->timeout/no frames->close)
  // that can otherwise drive rapid start/stop cycles in the supervisor.
  std::string kerr;
  if (!write_keepalive(opened_or_renegotiated, &kerr)) {
    if (error)
      *error = kerr;
    return false;
  }

  return true;
}

void CameraPipeline::CloseOutput() {
  std::lock_guard<std::mutex> lock(mu_);
  if (running_ || starting_)
    return;
  writer_.Close();
  writer_device_.clear();
  output_device_.clear();
  output_ = ActualFormat{};
}

void CameraPipeline::Stop() {
  stop_.store(true);
  cv_.notify_all();

  std::thread toJoin;
  {
    std::lock_guard<std::mutex> lock(mu_);
    effects_backends_.clear();
    if (th_.joinable()) {
      toJoin = std::move(th_);
    } else {
      running_ = false;
      starting_ = false;
      capture_ = CaptureFormat{};
      output_ = ActualFormat{};
      capture_fallback_state_ = "none";
      capture_fallback_reason_.clear();
      scaling_backend_active_.clear();
      scaling_from_ = CaptureFormat{};
      scaling_to_ = ActualFormat{};
      return;
    }
  }

  toJoin.join();

  std::lock_guard<std::mutex> lock(mu_);
  running_ = false;
  starting_ = false;
  effects_backends_.clear();
  capture_ = CaptureFormat{};
  output_ = ActualFormat{};
  capture_fallback_state_ = "none";
  capture_fallback_reason_.clear();
  scaling_backend_active_.clear();
  scaling_from_ = CaptureFormat{};
  scaling_to_ = ActualFormat{};
}

void CameraPipeline::ThreadMain(CameraPipelineConfig cfg) {
  // Open input capture.
  V4l2Capture cap;
  std::string inDev = cfg.input_device;
  std::ostringstream inAttempts;

  const bool auto_best = (cfg.capture_mode == CaptureMode::auto_best);

  if (!inDev.empty()) {
    std::string cerr;
    const bool opened =
        auto_best ? cap.OpenBest(inDev, cfg.fps, cfg.prefer_mjpeg, &cerr)
                  : cap.Open(inDev, cfg.width, cfg.height, cfg.fps,
                             CapturePixelFormat::yuyv, cfg.prefer_mjpeg, &cerr);

    if (!opened) {
      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = "Failed to open capture device " + inDev + ":\n" + cerr;
      running_ = false;
      start_notified_ = true;
      cv_.notify_all();
      return;
    }
  } else {
    const auto candidates = ListCandidateCameras();
    if (candidates.empty()) {
      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = "No readable camera device found (no non-loopback "
                    "/dev/video* with read access).";
      running_ = false;
      start_notified_ = true;
      cv_.notify_all();
      return;
    }

    bool opened = false;
    for (const auto &d : candidates) {
      std::string cerr;
      const bool thisOpened =
          auto_best
              ? cap.OpenBest(d.dev_node, cfg.fps, cfg.prefer_mjpeg, &cerr)
              : cap.Open(d.dev_node, cfg.width, cfg.height, cfg.fps,
                         CapturePixelFormat::yuyv, cfg.prefer_mjpeg, &cerr);

      if (thisOpened) {
        inDev = d.dev_node;
        opened = true;
        break;
      }

      inAttempts << "  - " << d.dev_node;
      if (!d.name.empty())
        inAttempts << " (" << d.name << ")";
      inAttempts << ":\n";
      inAttempts << "    " << cerr << "\n";
    }

    if (!opened) {
      std::ostringstream oss;
      oss << "Failed to auto-select a usable camera.\n";
      oss << "Tried the following devices:\n";
      oss << inAttempts.str();
      oss << "\nTip: specify an input explicitly (e.g. studiocastd --input "
             "/dev/video0)\n";
      oss << "     or try a smaller resolution like 640x480 (many laptop "
             "webcams only expose YUYV at lower resolutions).\n";

      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = oss.str();
      running_ = false;
      start_notified_ = true;
      cv_.notify_all();
      return;
    }
  }

  std::string outDev = cfg.output_device;
  if (outDev.empty()) {
    std::string e;
    outDev = ChooseDefaultOutputLoopback(&e);
    if (outDev.empty()) {
      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = e.empty() ? "No output loopback found." : e;
      running_ = false;
      start_notified_ = true;
      cv_.notify_all();
      return;
    }
  }

  auto capA = cap.Actual();

  // Open (or reuse) writer to v4l2loopback output.
  {
    std::lock_guard<std::mutex> lock(mu_);
    bool opened_or_renegotiated = false;
    std::string oerr;
    int out_w = cfg.width;
    int out_h = cfg.height;
    if (cfg.capture_mode == CaptureMode::auto_best &&
        (out_w <= 0 || out_h <= 0)) {
      // In auto capture mode (without an explicit output override), align
      // output to the negotiated capture size.
      out_w = capA.width;
      out_h = capA.height;
    }

    // Keep output size aligned with an explicitly configured width/height.
    // Many webcams silently negotiate down when requesting YUYV at HD sizes; if
    // we switch the loopback format to that smaller negotiated size while a
    // consumer still requests the configured size, the frame can display in the
    // top-left quadrant.
    if (!OpenOutputLocked(outDev, out_w, out_h, cfg.fps, cfg.output_format,
                          true, &opened_or_renegotiated, &oerr)) {
      last_error_ = oerr;
      running_ = false;
      start_notified_ = true;
      cv_.notify_all();
      return;
    }
  }

  ActualFormat outA = writer_.Actual();

  // Scaling backend selection (cpu|gpu|auto).
  //
  // GPU scaling is optional and depends on runtime-loaded CUDA.
  //
  // Backends (fallback order):
  //   1) Maxine-compatible NvCV resize (NVCV)
  //   2) Pure CUDA resize kernel (OpenCUDA)
  //   3) None
  enum class GpuResizeBackend {
    none,
    maxine_nvcv,
    open_cuda,
    vulkan,
  };

  auto GpuResizeBackendToActiveString = [](GpuResizeBackend b) -> std::string {
    switch (b) {
    case GpuResizeBackend::maxine_nvcv:
      return "gpu:maxine";
    case GpuResizeBackend::open_cuda:
      return "gpu:open_cuda";
    case GpuResizeBackend::vulkan:
      return "gpu:vulkan";
    default:
      return "cpu";
    }
  };

  struct MaxineGpuScaler {
    bool initialized = false;
    std::string init_error;

    studiocast::maxine::NvcvApi nvcv;
    studiocast::maxine::CudaDriverApi cuda;
    studiocast::maxine::CudaBgrResizeBilinear resize;

    studiocast::maxine::NvCVImage gpu_in{};
    bool gpu_in_allocated = false;

    studiocast::maxine::NvCVImage gpu_scaled{};
    bool gpu_scaled_allocated = false;

    // Tight CPU-side staging buffers.
    std::vector<std::uint8_t> bgr_in;
    std::vector<std::uint8_t> bgr_out;
    studiocast::maxine::NvCVImage cpu_bgr_in{};
    studiocast::maxine::NvCVImage cpu_bgr_out{};

    ~MaxineGpuScaler() {
      if (gpu_scaled_allocated && nvcv.IsInitialized() &&
          nvcv.f().NvCVImage_Dealloc) {
        (void)nvcv.f().NvCVImage_Dealloc(&gpu_scaled);
      }
      if (gpu_in_allocated && nvcv.IsInitialized() &&
          nvcv.f().NvCVImage_Dealloc) {
        (void)nvcv.f().NvCVImage_Dealloc(&gpu_in);
      }
    }
  };

  struct OpenCudaGpuScaler {
    bool initialized = false;
    std::string init_error;

    studiocast::maxine::CudaDriverApi cuda;
    studiocast::maxine::CUstream stream = nullptr;
    bool stream_created = false;

    studiocast::cuda::CudaImage gpu_in{};
    studiocast::cuda::CudaImage gpu_scaled{};

    ~OpenCudaGpuScaler() {
      if (initialized) {
        std::string e;
        if (gpu_scaled.Valid())
          (void)gpu_scaled.Free(&cuda, &e);
        if (gpu_in.Valid())
          (void)gpu_in.Free(&cuda, &e);
        if (stream_created)
          (void)cuda.DestroyStream(stream, &e);
      }
    }

    bool EnsureInitialized(std::string *error_out) {
      if (error_out)
        error_out->clear();
      if (initialized)
        return true;
      std::string e;

      if (!cuda.Initialize(&e)) {
        init_error = e;
        if (error_out)
          *error_out = e;
        return false;
      }
      if (!cuda.EnsureContext(&e)) {
        init_error = e;
        if (error_out)
          *error_out = e;
        return false;
      }
      if (!studiocast::cuda::kernels::IsResizeBilinearAvailable(&e)) {
        init_error = e;
        if (error_out)
          *error_out = e;
        return false;
      }
      if (!cuda.CreateStream(&stream, &e)) {
        init_error = e;
        if (error_out)
          *error_out = e;
        return false;
      }
      stream_created = true;

      initialized = true;
      init_error.clear();
      return true;
    }
  };

  struct VulkanGpuScaler {
    bool initialized = false;
    std::string init_error;

#if STUDIOCAST_ENABLE_OPEN_VULKAN
    studiocast::vulkan::kernels::ResizeBilinear resize;
#endif

    bool EnsureInitialized(int src_w, int src_h, int dst_w, int dst_h,
                           std::string *error_out) {
      if (error_out)
        error_out->clear();
      if (initialized)
        return true;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
      std::string e;
      if (!resize.EnsureInitialized(src_w, src_h, dst_w, dst_h, &e)) {
        init_error = e;
        if (error_out)
          *error_out = e;
        return false;
      }
      initialized = true;
      init_error.clear();
      return true;
#else
      (void)src_w;
      (void)src_h;
      (void)dst_w;
      (void)dst_h;
      init_error =
          "Open Vulkan resize backend is disabled in this build. Rebuild with "
          "-DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON.";
      if (error_out)
        *error_out = init_error;
      return false;
#endif
    }
  };

  MaxineGpuScaler maxine_scaler;
  OpenCudaGpuScaler open_cuda_scaler;
  VulkanGpuScaler vulkan_scaler;

  const bool resize_needed =
      (capA.width != outA.width) || (capA.height != outA.height);
  const bool compute_effects_configured =
      studiocast::video::effects::BroadcastEffectsPlanRequestsCompute(
          studiocast::video::effects::BuildBroadcastEffectsPlan(cfg.effects));
  const bool compute_backend_allows_cuda =
      ComputeBackendAllowsCudaVideoCompute(cfg.compute_backend);
  const bool want_vulkan_runtime =
      compute_effects_configured &&
      cfg.compute_backend == ComputeBackendPreference::vulkan;
  const bool want_vulkan_scaling =
      want_vulkan_runtime && resize_needed &&
      (cfg.scaling_backend != ScalingBackendPreference::cpu);
  const bool want_gpu_scaling =
      resize_needed && compute_effects_configured &&
      compute_backend_allows_cuda &&
      (cfg.scaling_backend != ScalingBackendPreference::cpu);
  GpuResizeBackend gpu_backend = GpuResizeBackend::none;
  std::string scaling_backend_active = "cpu";
  std::string gpu_backend_init_note;

  if (want_vulkan_runtime) {
    std::string vk_err;
    const int vk_dst_w = want_vulkan_scaling ? outA.width : capA.width;
    const int vk_dst_h = want_vulkan_scaling ? outA.height : capA.height;
    if (vulkan_scaler.EnsureInitialized(capA.width, capA.height, vk_dst_w,
                                        vk_dst_h, &vk_err)) {
      if (want_vulkan_scaling)
        gpu_backend = GpuResizeBackend::vulkan;
    } else {
      gpu_backend_init_note = "Open Vulkan resize init failed: " + vk_err;
    }
    scaling_backend_active = GpuResizeBackendToActiveString(gpu_backend);
  } else if (want_gpu_scaling) {
    std::string maxine_err;
    bool maxine_ok = true;
    if (!maxine_scaler.cuda.Initialize(&maxine_err))
      maxine_ok = false;
    if (maxine_ok &&
        !maxine_scaler.nvcv.Initialize(
            studiocast::maxine::NvcvApi::Requirement::VfxCompat, &maxine_err)) {
      maxine_ok = false;
    }
    if (maxine_ok &&
        !maxine_scaler.resize.Initialize(&maxine_scaler.cuda, &maxine_err))
      maxine_ok = false;

    maxine_scaler.initialized = maxine_ok;
    maxine_scaler.init_error = maxine_err;

    if (maxine_ok) {
      gpu_backend = GpuResizeBackend::maxine_nvcv;
    } else {
      std::string open_err;
      if (open_cuda_scaler.EnsureInitialized(&open_err)) {
        gpu_backend = GpuResizeBackend::open_cuda;
      } else {
        // Keep some diagnostics for status/error reporting.
        std::ostringstream oss;
        if (!maxine_err.empty())
          oss << "Maxine/NVCV init failed: " << maxine_err;
        if (!open_err.empty()) {
          if (oss.tellp() > 0)
            oss << " | ";
          oss << "OpenCUDA resize init failed: " << open_err;
        }
        gpu_backend_init_note = oss.str();
      }
    }

    scaling_backend_active = GpuResizeBackendToActiveString(gpu_backend);
  }

  {
    std::string policy_err;
    const bool gpu_resize_available = (gpu_backend != GpuResizeBackend::none);
    if (!CheckOutputResizeAllowed(capA.width, capA.height, outA.width,
                                  outA.height, gpu_resize_available,
                                  cfg.allow_cpu_resize, &policy_err)) {
      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = policy_err;
      if (!gpu_backend_init_note.empty()) {
        last_error_ += "\nGPU resize diagnostics: " + gpu_backend_init_note;
      }
      running_ = false;
      start_notified_ = true;
      cv_.notify_all();
      return;
    }
  }

  studiocast::video::Rgb24Frame rgb;
  rgb.ResizeTight(capA.width, capA.height);
  std::size_t rgbStride = rgb.stride_bytes;

  std::vector<std::uint8_t> rgbScaled;
  if (outA.width > 0 && outA.height > 0) {
    // Prepared final-output adapters borrow this fixed writer buffer. Size it
    // during setup so their frame path never grows the vector.
    rgbScaled.resize(static_cast<std::size_t>(outA.width) *
                     static_cast<std::size_t>(outA.height) * 3u);
  }
  Rgb24BilinearResizePlan cpuResizePlan;

  std::vector<std::uint8_t> outBuf(outA.size_image);
  std::vector<std::uint8_t> yuyvConversionScratch;

  auto ResizeYuyvConversionScratch = [&] {
    const std::size_t need =
        (outA.format == PixelFormat::yuyv)
            ? Rgb24ToYuyvScratchBytes(outA.width, outA.height)
            : 0u;
    if (need > 0) {
      yuyvConversionScratch.resize(need);
    } else {
      yuyvConversionScratch.clear();
    }
  };
  ResizeYuyvConversionScratch();

  auto ZeroRgbOutputPadding = [&] {
    if (outA.format != PixelFormat::rgb24 || outA.width <= 0 ||
        outA.height <= 0)
      return;
    const std::size_t active_row = static_cast<std::size_t>(outA.width) * 3u;
    if (outA.bytes_per_line <= active_row)
      return;
    const std::size_t required =
        outA.bytes_per_line * static_cast<std::size_t>(outA.height);
    if (outBuf.size() < required)
      return;
    for (int y = 0; y < outA.height; ++y) {
      std::uint8_t *row =
          outBuf.data() + static_cast<std::size_t>(y) * outA.bytes_per_line;
      std::memset(row + active_row, 0, outA.bytes_per_line - active_row);
    }
  };
  ZeroRgbOutputPadding();

  {
    std::lock_guard<std::mutex> lock(mu_);
    input_device_ = inDev;
    output_device_ = outDev;
    capture_ = capA;
    output_ = outA;
    scaling_backend_active_ = scaling_backend_active;
    scaling_from_ = capA;
    scaling_to_ = outA;
    frame_index_ = 0;
    last_error_.clear();
    running_ = true;

    start_notified_ = true;
    cv_.notify_all();
  }

  int frameIndex = 0;

  // Build a modular effect chain (Maxine-ready). This can be rebuilt live when
  // the GUI/daemon updates effect settings.
  studiocast::video::effects::EffectChain chain;
  studiocast::video::effects::BroadcastCameraEffects appliedFx{};
  studiocast::video::effects::BroadcastEffectsPlan appliedPlan{};
  studiocast::video::effects::PreparedBroadcastEffectsFramePlan
      appliedFramePlan{};
  std::uint64_t appliedEffectsGeneration = 0;
  EffectsFramePreparationCounters effects_preparation_counters{};
  RawYuyvPassthroughPlan raw_yuyv_plan{};
  YuyvFramePathCounters yuyv_frame_path_counters{};

  // Optional deferred GPU output (used to avoid CPU resize when scaling is
  // needed and no CPU tail effects are active).
  enum class DeferredGpuKind {
    none,
    nvcv_bgr,
    cuda_rgb,
#if STUDIOCAST_ENABLE_OPEN_VULKAN
    vulkan_rgb,
#endif
  };

  struct DeferredGpuOut {
    DeferredGpuKind kind = DeferredGpuKind::none;

    // Maxine / NvCV (BGR)
    const studiocast::maxine::NvCVImage *nvcv_img = nullptr; // non-owning
    studiocast::maxine::NvcvApi *nvcv = nullptr;             // non-owning

    // Open CUDA (RGB)
    const studiocast::cuda::CudaImage *cuda_img = nullptr; // non-owning

#if STUDIOCAST_ENABLE_OPEN_VULKAN
    // Open Vulkan (RGB)
    const studiocast::vulkan::VulkanImage *vulkan_img = nullptr; // non-owning
    studiocast::vulkan::kernels::UtilityKernels *vulkan_kernels =
        nullptr; // non-owning
#endif

    // Common
    studiocast::maxine::CUstream stream = nullptr;
    studiocast::maxine::CudaDriverApi *cuda = nullptr; // non-owning
  };

  struct MaxineFrameLocalSharedHandles {
    enum class Lifetime {
      frame_local_producer_owned,
    };

    enum class RejectReason {
      none,
      no_producer,
      ownership,
      capture_sequence,
      dimensions,
      nvcv_runtime,
      stream,
      input_not_current_rgb,
      input_image,
      matte_image,
      green_screen_config,
    };

    bool valid = false;

    // All handles are non-owning and are only valid until the producing
    // Maxine context runs again, reallocates, or is destroyed.
    Lifetime lifetime = Lifetime::frame_local_producer_owned;
    bool owns_input = false;
    bool owns_matte = false;

    const char *producer_stage_id = nullptr;
    std::uint64_t capture_sequence = 0;
    int width = 0;
    int height = 0;

    studiocast::maxine::NvcvApi *nvcv = nullptr;       // non-owning
    studiocast::maxine::CudaDriverApi *cuda = nullptr; // non-owning
    studiocast::maxine::CUstream stream = nullptr;

    const studiocast::maxine::NvCVImage *input_bgr_gpu = nullptr;
    const studiocast::maxine::NvCVImage *matte_gpu = nullptr;

    // True only when the input handle still represents the current pipeline
    // RGB buffer. Virtual Background publishes this as false because it has
    // already mutated the CPU RGB continuation by the time a later stage runs.
    bool input_matches_current_rgb = false;

    // Green Screen uses NvVFX_Run(async=0), so a successful Process() makes the
    // matte visible without adding a synchronization point here.
    bool matte_ready_for_cross_stream_reuse = false;

    studiocast::maxine::effects::VfxGreenScreenEffect::Config
        green_screen_config{};

    void Reset() { *this = MaxineFrameLocalSharedHandles{}; }

    static bool GreenScreenConfigEqual(
        const studiocast::maxine::effects::VfxGreenScreenEffect::Config &a,
        const studiocast::maxine::effects::VfxGreenScreenEffect::Config &b) {
      return a.mode == b.mode && a.temporal == b.temporal &&
             a.state_count == b.state_count;
    }

    static bool SameNvcvRuntime(const studiocast::maxine::NvcvApi *a,
                                const studiocast::maxine::NvcvApi *b) {
      if (!a || !b || !a->IsInitialized() || !b->IsInitialized())
        return false;
      if (a == b)
        return true;
      return !a->library_path().empty() &&
             a->library_path() == b->library_path();
    }

    static bool
    ImageMatches(const studiocast::maxine::NvCVImage *img, int expected_width,
                 int expected_height,
                 studiocast::maxine::NvCVImage_PixelFormat expected_format,
                 studiocast::maxine::NvCVImage_ComponentType expected_type,
                 unsigned expected_layout, unsigned expected_mem) {
      return img && img->pixels && expected_width > 0 && expected_height > 0 &&
             img->width == static_cast<std::uint32_t>(expected_width) &&
             img->height == static_cast<std::uint32_t>(expected_height) &&
             img->pixelFormat == expected_format &&
             img->componentType == expected_type &&
             img->planar == expected_layout && img->gpuMem == expected_mem;
    }

    void PublishGreenScreen(
        const char *producer, std::uint64_t sequence, int frame_width,
        int frame_height, studiocast::maxine::NvcvApi *producer_nvcv,
        studiocast::maxine::CudaDriverApi *producer_cuda,
        studiocast::maxine::CUstream producer_stream,
        const studiocast::maxine::NvCVImage *input_gpu,
        const studiocast::maxine::NvCVImage *matte,
        const studiocast::maxine::effects::VfxGreenScreenEffect::Config &cfg,
        bool input_is_current_rgb, bool matte_ready_cross_stream) {
      valid = true;
      lifetime = Lifetime::frame_local_producer_owned;
      owns_input = false;
      owns_matte = false;
      producer_stage_id = producer;
      capture_sequence = sequence;
      width = frame_width;
      height = frame_height;
      nvcv = producer_nvcv;
      cuda = producer_cuda;
      stream = producer_stream;
      input_bgr_gpu = input_gpu;
      matte_gpu = matte;
      input_matches_current_rgb = input_is_current_rgb;
      matte_ready_for_cross_stream_reuse = matte_ready_cross_stream;
      green_screen_config = cfg;
    }

    RejectReason ValidateMatteForConsumer(
        std::uint64_t sequence, int frame_width, int frame_height,
        const studiocast::maxine::NvcvApi *consumer_nvcv,
        studiocast::maxine::CUstream consumer_stream,
        const studiocast::maxine::effects::VfxGreenScreenEffect::Config
            &consumer_cfg) const {
      if (!valid)
        return RejectReason::no_producer;
      if (owns_input || owns_matte)
        return RejectReason::ownership;
      if (capture_sequence != sequence)
        return RejectReason::capture_sequence;
      if (width != frame_width || height != frame_height)
        return RejectReason::dimensions;
      if (!SameNvcvRuntime(nvcv, consumer_nvcv))
        return RejectReason::nvcv_runtime;
      if (!matte_ready_for_cross_stream_reuse && stream != consumer_stream)
        return RejectReason::stream;
      if (!ImageMatches(matte_gpu, frame_width, frame_height,
                        studiocast::maxine::NVCV_A, studiocast::maxine::NVCV_U8,
                        studiocast::maxine::NVCV_CHUNKY,
                        studiocast::maxine::NVCV_GPU)) {
        return RejectReason::matte_image;
      }
      if (!GreenScreenConfigEqual(green_screen_config, consumer_cfg))
        return RejectReason::green_screen_config;
      return RejectReason::none;
    }

    RejectReason ValidateInputForCurrentRgbConsumer(
        std::uint64_t sequence, int frame_width, int frame_height,
        const studiocast::maxine::NvcvApi *consumer_nvcv,
        studiocast::maxine::CUstream consumer_stream) const {
      if (!valid)
        return RejectReason::no_producer;
      if (owns_input || owns_matte)
        return RejectReason::ownership;
      if (capture_sequence != sequence)
        return RejectReason::capture_sequence;
      if (width != frame_width || height != frame_height)
        return RejectReason::dimensions;
      if (!SameNvcvRuntime(nvcv, consumer_nvcv))
        return RejectReason::nvcv_runtime;
      if (!matte_ready_for_cross_stream_reuse && stream != consumer_stream)
        return RejectReason::stream;
      if (!input_matches_current_rgb)
        return RejectReason::input_not_current_rgb;
      if (!ImageMatches(
              input_bgr_gpu, frame_width, frame_height,
              studiocast::maxine::NVCV_BGR, studiocast::maxine::NVCV_U8,
              studiocast::maxine::NVCV_CHUNKY, studiocast::maxine::NVCV_GPU)) {
        return RejectReason::input_image;
      }
      return RejectReason::none;
    }
  };

  // GPU resize cache (allocated on demand when we can defer readback).
  studiocast::maxine::CudaBgrResizeBilinear gpu_resize_bilinear;
  studiocast::maxine::CudaDriverApi *gpu_resize_cuda = nullptr; // non-owning
  studiocast::maxine::NvcvApi *gpu_resize_nvcv = nullptr;       // non-owning
  studiocast::maxine::NvCVImage gpu_bgr_scaled{};
  bool gpu_bgr_scaled_allocated = false;
  std::vector<std::uint8_t> bgr_scaled_out;
  studiocast::maxine::NvCVImage cpu_bgr_scaled{};

  // Open CUDA deferred GPU output (RGB) + resize cache.
  studiocast::cuda::CudaImage gpu_rgb_scaled;
  bool gpu_rgb_scaled_allocated = false;

#if STUDIOCAST_ENABLE_OPEN_VULKAN
  // Open Vulkan deferred GPU output (RGB) + resize cache.
  studiocast::vulkan::VulkanImage vulkan_rgb_scaled;
  bool vulkan_rgb_scaled_allocated = false;
#endif

  // Open CUDA per-frame GPU buffers (ping-pong) for GPU-only Open CUDA stages.
  // These persist across frames and are reallocated only when the capture size
  // changes.
  studiocast::cuda::CudaImage open_cuda_frame_a;
  studiocast::cuda::CudaImage open_cuda_frame_b;
  const studiocast::cuda::CudaImage *open_cuda_curr = nullptr;
  studiocast::cuda::CudaImage *open_cuda_next = nullptr;
  bool open_cuda_uploaded_this_frame = false;

  struct MaxineBackgroundBlurContext {
    bool initialized = false;
    bool enabled = false;
    std::string last_error;

    studiocast::maxine::vfx::VfxApi vfx;
    studiocast::maxine::NvcvApi nvcv;
    studiocast::maxine::CudaDriverApi cuda;
    studiocast::maxine::CudaBgrVignette vignette;

    std::filesystem::path model_dir;

    std::unique_ptr<studiocast::maxine::effects::VfxGreenScreenEffect>
        greenscreen;
    std::unique_ptr<studiocast::maxine::effects::VfxBackgroundBlurEffect> blur;

    std::vector<std::uint8_t> bgr_in;
    std::vector<std::uint8_t> bgr_out;
    std::vector<std::uint8_t> bgr_bg;
    studiocast::maxine::NvCVImage cpu_bgr_in{};
    studiocast::maxine::NvCVImage cpu_bgr_out{};
    studiocast::maxine::NvCVImage cpu_bgr_bg{};

    studiocast::maxine::NvCVImage gpu_bgr{};
    bool gpu_bgr_allocated = false;

    studiocast::maxine::NvCVImage gpu_bgr_bg{};
    bool gpu_bgr_bg_allocated = false;

    studiocast::maxine::NvCVImage gpu_bgr_out_img{};
    bool gpu_bgr_out_allocated = false;

    // Background cache keys.
    std::uint32_t cached_bg_color_rgb = 0xFFFFFFFFu; // invalid
    std::filesystem::path cached_bg_path;
    bool cached_bg_is_replace = false;

    // Temporary CPU-side buffers for replace image decoding/resizing.
    std::vector<std::uint8_t> tmp_replace_rgb_src;
    std::vector<std::uint8_t> tmp_replace_rgb_resized;

    ~MaxineBackgroundBlurContext() { Destroy(); }

    void Destroy() {
      greenscreen.reset();
      blur.reset();

      if (gpu_bgr_out_allocated && nvcv.IsInitialized() &&
          nvcv.f().NvCVImage_Dealloc) {
        (void)nvcv.f().NvCVImage_Dealloc(&gpu_bgr_out_img);
      }
      gpu_bgr_out_img = studiocast::maxine::NvCVImage{};
      gpu_bgr_out_allocated = false;

      if (gpu_bgr_bg_allocated && nvcv.IsInitialized() &&
          nvcv.f().NvCVImage_Dealloc) {
        (void)nvcv.f().NvCVImage_Dealloc(&gpu_bgr_bg);
      }
      gpu_bgr_bg = studiocast::maxine::NvCVImage{};
      gpu_bgr_bg_allocated = false;

      if (gpu_bgr_allocated && nvcv.IsInitialized() &&
          nvcv.f().NvCVImage_Dealloc) {
        (void)nvcv.f().NvCVImage_Dealloc(&gpu_bgr);
      }
      gpu_bgr = studiocast::maxine::NvCVImage{};
      gpu_bgr_allocated = false;

      bgr_in.clear();
      bgr_out.clear();
      bgr_bg.clear();
      cpu_bgr_in = studiocast::maxine::NvCVImage{};
      cpu_bgr_out = studiocast::maxine::NvCVImage{};
      cpu_bgr_bg = studiocast::maxine::NvCVImage{};

      cached_bg_color_rgb = 0xFFFFFFFFu;
      cached_bg_path.clear();
      cached_bg_is_replace = false;

      initialized = false;
      enabled = false;
    }

    static std::filesystem::path
    InferModelsDirFromLibrary(const std::filesystem::path &lib) {
      std::error_code ec;
      std::filesystem::path p = lib.parent_path();
      for (int i = 0; i < 5 && !p.empty(); ++i) {
        const auto cand = p / "models";
        if (std::filesystem::exists(cand, ec) &&
            std::filesystem::is_directory(cand, ec)) {
          return cand;
        }
        p = p.parent_path();
      }
      return {};
    }

    bool EnsureInitialized(
        int width, int height,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error) {
      last_error.clear();

      if (initialized) {
        // Effects can be reconfigured live.
        std::string cfg_err;
        if (greenscreen && !greenscreen->Configure(fx, &cfg_err)) {
          if (error)
            *error = cfg_err;
          return false;
        }
        if (blur && !blur->Configure(fx, &cfg_err)) {
          if (error)
            *error = cfg_err;
          return false;
        }
        return true;
      }

      std::string err;
      if (!vfx.Initialize(&err)) {
        last_error = "Maxine VFX runtime unavailable: " + err;
        if (error)
          *error = last_error;
        return false;
      }
      if (!nvcv.Initialize(studiocast::maxine::NvcvApi::Requirement::VfxCompat,
                           &err)) {
        last_error = "NvCVImage runtime unavailable: " + err;
        if (error)
          *error = last_error;
        return false;
      }

      {
        std::string cuda_err;
        if (!cuda.Initialize(&cuda_err)) {
          last_error = "CUDA driver API unavailable: " + cuda_err;
          if (error)
            *error = last_error;
          return false;
        }
        std::string vig_err;
        if (!vignette.Initialize(&cuda, &vig_err)) {
          last_error = "CUDA vignette init failed: " + vig_err;
          if (error)
            *error = last_error;
          return false;
        }
      }
      if (model_dir.empty()) {
        model_dir = InferModelsDirFromLibrary(vfx.library_path());
      }

      // Allocate CPU-side BGR staging buffers and wrap with NvCVImage_Init.
      const std::size_t stride = static_cast<std::size_t>(width) * 3u;
      bgr_in.resize(stride * static_cast<std::size_t>(height));
      bgr_out.resize(stride * static_cast<std::size_t>(height));
      bgr_bg.resize(stride * static_cast<std::size_t>(height));

      if (!nvcv.f().NvCVImage_Init) {
        last_error = "NvCVImage_Init missing from NvCVImage runtime.";
        if (error)
          *error = last_error;
        return false;
      }

      auto st = nvcv.f().NvCVImage_Init(
          &cpu_bgr_in, static_cast<unsigned>(width),
          static_cast<unsigned>(height), static_cast<int>(stride),
          bgr_in.data(), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_CPU);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        last_error = "NvCVImage_Init(cpu BGR in) failed: " + std::to_string(st);
        if (error)
          *error = last_error;
        return false;
      }

      st = nvcv.f().NvCVImage_Init(
          &cpu_bgr_out, static_cast<unsigned>(width),
          static_cast<unsigned>(height), static_cast<int>(stride),
          bgr_out.data(), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_CPU);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        last_error =
            "NvCVImage_Init(cpu BGR out) failed: " + std::to_string(st);
        if (error)
          *error = last_error;
        return false;
      }

      st = nvcv.f().NvCVImage_Init(
          &cpu_bgr_bg, static_cast<unsigned>(width),
          static_cast<unsigned>(height), static_cast<int>(stride),
          bgr_bg.data(), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_CPU);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        last_error = "NvCVImage_Init(cpu BGR bg) failed: " + std::to_string(st);
        if (error)
          *error = last_error;
        return false;
      }

      // Allocate GPU input image.
      st = nvcv.f().NvCVImage_Alloc(
          &gpu_bgr, static_cast<unsigned>(width), static_cast<unsigned>(height),
          studiocast::maxine::NVCV_BGR, studiocast::maxine::NVCV_U8,
          studiocast::maxine::NVCV_CHUNKY, studiocast::maxine::NVCV_GPU,
          /*alignment=*/0);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        last_error = "NvCVImage_Alloc(gpu BGR) failed: " + std::to_string(st);
        if (error)
          *error = last_error;
        return false;
      }
      gpu_bgr_allocated = true;

      // Allocate GPU background image (BGRu8) and a GPU output image used by
      // compositing.
      st = nvcv.f().NvCVImage_Alloc(
          &gpu_bgr_bg, static_cast<unsigned>(width),
          static_cast<unsigned>(height), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_GPU,
          /*alignment=*/0);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        last_error =
            "NvCVImage_Alloc(gpu BGR bg) failed: " + std::to_string(st);
        if (error)
          *error = last_error;
        return false;
      }
      gpu_bgr_bg_allocated = true;

      st = nvcv.f().NvCVImage_Alloc(
          &gpu_bgr_out_img, static_cast<unsigned>(width),
          static_cast<unsigned>(height), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_GPU,
          /*alignment=*/0);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        last_error =
            "NvCVImage_Alloc(gpu BGR out) failed: " + std::to_string(st);
        if (error)
          *error = last_error;
        return false;
      }
      gpu_bgr_out_allocated = true;

      greenscreen =
          std::make_unique<studiocast::maxine::effects::VfxGreenScreenEffect>(
              &vfx, &nvcv, model_dir);
      blur = std::make_unique<
          studiocast::maxine::effects::VfxBackgroundBlurEffect>(&vfx, &nvcv,
                                                                model_dir);

      if (!greenscreen->Configure(fx, &err) || !blur->Configure(fx, &err)) {
        last_error = "Maxine effect Configure failed: " + err;
        if (error)
          *error = last_error;
        Destroy();
        return false;
      }
      if (!greenscreen->Initialize(&err)) {
        last_error = "Maxine green screen Initialize failed: " + err;
        if (error)
          *error = last_error;
        Destroy();
        return false;
      }
      if (!blur->Initialize(&err)) {
        last_error = "Maxine background blur Initialize failed: " + err;
        if (error)
          *error = last_error;
        Destroy();
        return false;
      }

      initialized = true;
      return true;
    }

    bool EnsureBackgroundGpu(
        int width, int height,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        studiocast::maxine::CUstream stream, std::string *error) {
      if (!initialized || !nvcv.IsInitialized()) {
        if (error)
          *error = "NvCVImage runtime not initialized.";
        return false;
      }
      if (!nvcv.f().NvCVImage_Transfer) {
        if (error)
          *error = "NvCVImage_Transfer unavailable.";
        return false;
      }

      const bool wantReplace =
          (fx.virtual_background.mode ==
           studiocast::video::effects::VirtualBackgroundMode::replace);

      std::uint32_t remove_rgb = 0x000000u;
      if (!wantReplace) {
        // remove mode uses a solid color
        if (!ParseRgbHex(fx.virtual_background.remove_color, &remove_rgb)) {
          remove_rgb = 0x000000u;
        }
      }

      bool needsUpload = false;
      if (wantReplace) {
        const std::filesystem::path path = fx.virtual_background.replace_path;
        if (!cached_bg_is_replace || cached_bg_path != path) {
          needsUpload = true;
        }
      } else {
        // remove mode uses a solid color
        if (cached_bg_is_replace ||
            cached_bg_color_rgb != (remove_rgb & 0xFFFFFFu)) {
          needsUpload = true;
        }
      }

      if (!needsUpload) {
        return true;
      }

      const std::size_t stride = static_cast<std::size_t>(width) * 3u;
      if (bgr_bg.size() != stride * static_cast<std::size_t>(height)) {
        // Should not happen for a running pipeline; fail safe.
        if (error)
          *error = "Background buffer size mismatch.";
        return false;
      }

      if (wantReplace) {
        if (fx.virtual_background.replace_path.empty()) {
          if (error)
            *error = "virtual_background.replace_path not set.";
          return false;
        }

        const std::filesystem::path img_path =
            fx.virtual_background.replace_path;

        int iw = 0, ih = 0;
        std::string img_err;
        if (!LoadImageRgb24(img_path, &iw, &ih, &tmp_replace_rgb_src,
                            &img_err)) {
          if (error)
            *error = "Failed to load replace image: " + img_err;
          return false;
        }

        if (!ResizeRgb24Bilinear(tmp_replace_rgb_src.data(), iw, ih,
                                 static_cast<std::size_t>(iw) * 3u, width,
                                 height, &tmp_replace_rgb_resized, stride,
                                 &img_err)) {
          if (error)
            *error = "Failed to resize replace image: " + img_err;
          return false;
        }

        // RGB -> BGR into the staging buffer.
        for (int y = 0; y < height; ++y) {
          const auto *src_row = tmp_replace_rgb_resized.data() +
                                static_cast<std::size_t>(y) * stride;
          auto *dst_row = bgr_bg.data() + static_cast<std::size_t>(y) * stride;
          for (int x = 0; x < width; ++x) {
            const std::size_t i = static_cast<std::size_t>(x) * 3u;
            dst_row[i + 0] = src_row[i + 2];
            dst_row[i + 1] = src_row[i + 1];
            dst_row[i + 2] = src_row[i + 0];
          }
        }

        cached_bg_is_replace = true;
        cached_bg_path = img_path;
      } else {
        const std::uint32_t rgb = remove_rgb & 0xFFFFFFu;
        const std::uint8_t r = static_cast<std::uint8_t>((rgb >> 16u) & 0xFFu);
        const std::uint8_t g = static_cast<std::uint8_t>((rgb >> 8u) & 0xFFu);
        const std::uint8_t b = static_cast<std::uint8_t>((rgb >> 0u) & 0xFFu);

        for (int y = 0; y < height; ++y) {
          auto *row = bgr_bg.data() + static_cast<std::size_t>(y) * stride;
          for (int x = 0; x < width; ++x) {
            const std::size_t i = static_cast<std::size_t>(x) * 3u;
            row[i + 0] = b;
            row[i + 1] = g;
            row[i + 2] = r;
          }
        }

        cached_bg_is_replace = false;
        cached_bg_color_rgb = rgb;
        cached_bg_path.clear();
      }

      const auto up = nvcv.f().NvCVImage_Transfer(&cpu_bgr_bg, &gpu_bgr_bg,
                                                  1.0f, stream, nullptr);
      if (up != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error =
              "NvCVImage_Transfer(bg cpu->gpu) failed: " + std::to_string(up);
        return false;
      }
      return true;
    }

    bool ApplyRgbInPlace(
        std::uint8_t *rgb, int width, int height, std::size_t rgb_stride,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        bool apply_vignette, float vignette_center_x_px,
        float vignette_center_y_px, std::string *error, bool defer_readback,
        DeferredGpuOut *deferred_out) {
      if (!initialized || !greenscreen || !blur) {
        if (error)
          *error = "Maxine virtual background not initialized.";
        return false;
      }
      if (!rgb || width <= 0 || height <= 0 || rgb_stride == 0) {
        if (error)
          *error = "Invalid RGB buffer.";
        return false;
      }

      // RGB -> BGR staging
      studiocast::video::Rgb24ToBgr24(rgb, bgr_in.data(), width, height,
                                      rgb_stride, rgb_stride);

      // Upload CPU->GPU.
      const auto up = nvcv.f().NvCVImage_Transfer(
          &cpu_bgr_in, &gpu_bgr, 1.0f, greenscreen->cuda_stream(), nullptr);
      if (up != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error = "NvCVImage_Transfer(cpu->gpu) failed: " + std::to_string(up);
        return false;
      }

      studiocast::video::GpuFrame frame;
      frame.width = width;
      frame.height = height;
      frame.nvcv_gpu = &gpu_bgr;
      frame.cuda_stream = greenscreen->cuda_stream();

      std::string cfg_err;
      if (!greenscreen->Configure(fx, &cfg_err)) {
        if (error)
          *error = cfg_err;
        return false;
      }

      const studiocast::maxine::NvCVImage *out_gpu = nullptr;
      const auto stream = greenscreen->cuda_stream();

      if (fx.virtual_background.mode ==
          studiocast::video::effects::VirtualBackgroundMode::blur) {
        if (!blur->Configure(fx, &cfg_err)) {
          if (error)
            *error = cfg_err;
          return false;
        }
      }

      std::string proc_err;
      const auto st = greenscreen->Process(frame, &proc_err);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error = proc_err.empty() ? std::to_string(st) : proc_err;
        return false;
      }

      const auto *matte = greenscreen->MatteGpu();
      if (!matte) {
        if (error)
          *error = "Green Screen did not produce a matte.";
        return false;
      }

      frame.matte_gpu = matte;

      if (fx.virtual_background.mode ==
          studiocast::video::effects::VirtualBackgroundMode::blur) {
        const auto st2 = blur->Process(frame, &proc_err);
        if (st2 != studiocast::maxine::NVCV_SUCCESS) {
          if (error)
            *error = proc_err.empty() ? std::to_string(st2) : proc_err;
          return false;
        }

        out_gpu = blur->OutputGpu();
        if (!out_gpu) {
          if (error)
            *error = "Background Blur did not produce an output image.";
          return false;
        }

        if (apply_vignette && fx.vignette.enabled) {
          const float vig_intensity = Clamp01FromPercent(fx.vignette.intensity);
          if (vig_intensity > 0.0001f) {
            std::string ve;
            if (!vignette.ApplyInPlace(
                    const_cast<studiocast::maxine::NvCVImage *>(out_gpu),
                    vig_intensity, vignette_center_x_px, vignette_center_y_px,
                    blur->cuda_stream(), &ve)) {
              if (error)
                *error = OptionalEffectFailureReason("Vignette failed", ve);
              return false;
            }
          }
        }

        if (defer_readback) {
          if (!deferred_out) {
            if (error)
              *error = "defer_readback requires deferred_out.";
            return false;
          }
          deferred_out->kind = DeferredGpuKind::nvcv_bgr;
          deferred_out->nvcv_img = out_gpu;
          deferred_out->nvcv = &nvcv;
          deferred_out->cuda_img = nullptr;
          deferred_out->cuda = &cuda;
          deferred_out->stream = blur->cuda_stream();
          return true;
        }

        const auto down = nvcv.f().NvCVImage_Transfer(
            out_gpu, &cpu_bgr_out, 1.0f, blur->cuda_stream(), nullptr);
        if (down != studiocast::maxine::NVCV_SUCCESS) {
          if (error)
            *error =
                "NvCVImage_Transfer(gpu->cpu) failed: " + std::to_string(down);
          return false;
        }

        if (!SyncAfterGpuToCpuTransfer(
                cuda, blur->cuda_stream(),
                "Maxine VB blur: after gpu->cpu transfer", error)) {
          return false;
        }
      } else if (
          fx.virtual_background.mode ==
              studiocast::video::effects::VirtualBackgroundMode::remove ||
          fx.virtual_background.mode ==
              studiocast::video::effects::VirtualBackgroundMode::replace) {
        if (!nvcv.f().NvCVImage_Composite) {
          if (error)
            *error = "NvCVImage_Composite unavailable.";
          return false;
        }

        std::string bg_err;
        if (!EnsureBackgroundGpu(width, height, fx, stream, &bg_err)) {
          if (error)
            *error = bg_err;
          return false;
        }

        const auto stc = nvcv.f().NvCVImage_Composite(
            &gpu_bgr, &gpu_bgr_bg, matte, &gpu_bgr_out_img, stream);
        if (stc != studiocast::maxine::NVCV_SUCCESS) {
          if (error)
            *error = "NvCVImage_Composite failed: " + std::to_string(stc);
          return false;
        }

        if (apply_vignette && fx.vignette.enabled) {
          const float vig_intensity = Clamp01FromPercent(fx.vignette.intensity);
          if (vig_intensity > 0.0001f) {
            std::string ve;
            if (!vignette.ApplyInPlace(&gpu_bgr_out_img, vig_intensity,
                                       vignette_center_x_px,
                                       vignette_center_y_px, stream, &ve)) {
              if (error)
                *error = OptionalEffectFailureReason("Vignette failed", ve);
              return false;
            }
          }
        }

        if (defer_readback) {
          if (!deferred_out) {
            if (error)
              *error = "defer_readback requires deferred_out.";
            return false;
          }
          deferred_out->kind = DeferredGpuKind::nvcv_bgr;
          deferred_out->nvcv_img = &gpu_bgr_out_img;
          deferred_out->nvcv = &nvcv;
          deferred_out->cuda_img = nullptr;
          deferred_out->cuda = &cuda;
          deferred_out->stream = stream;
          return true;
        }

        const auto down = nvcv.f().NvCVImage_Transfer(
            &gpu_bgr_out_img, &cpu_bgr_out, 1.0f, stream, nullptr);
        if (down != studiocast::maxine::NVCV_SUCCESS) {
          if (error)
            *error =
                "NvCVImage_Transfer(gpu->cpu) failed: " + std::to_string(down);
          return false;
        }

        if (!SyncAfterGpuToCpuTransfer(
                cuda, stream, "Maxine VB composite: after gpu->cpu transfer",
                error)) {
          return false;
        }
      } else {
        // Not a virtual background mode we handle.
        return true;
      }

      // BGR -> RGB back into the pipeline buffer.
      studiocast::video::Bgr24ToRgb24(bgr_out.data(), rgb, width, height,
                                      rgb_stride, rgb_stride);
      return true;
    }

    const studiocast::maxine::NvCVImage *GreenScreenMatteGpu() const {
      return greenscreen ? greenscreen->MatteGpu() : nullptr;
    }

    const studiocast::maxine::NvCVImage *GreenScreenInputGpu() const {
      return gpu_bgr_allocated ? &gpu_bgr : nullptr;
    }

    studiocast::maxine::CUstream GreenScreenStream() const {
      return greenscreen ? greenscreen->cuda_stream() : nullptr;
    }

    studiocast::maxine::effects::VfxGreenScreenEffect::Config
    GreenScreenConfig() const {
      return greenscreen
                 ? greenscreen->config()
                 : studiocast::maxine::effects::VfxGreenScreenEffect::Config{};
    }
  } maxine_bg_blur;

  struct OpenCudaVirtualBackgroundContext {
    bool initialized = false;
    bool enabled = false;
    std::string last_error;

    // The model pack currently active in the pipeline. This is used to hot-swap
    // the Open CUDA session when the user changes
    // fx.virtual_background.model_id.
    std::string active_model_id;
    std::string active_requested_model_id;

    studiocast::maxine::CudaDriverApi cuda;

    // Persistent stream used for the entire Open CUDA virtual background
    // pipeline. This avoids accidental default-stream work and allows ORT (when
    // supported) to run on the same explicit stream via user_compute_stream.
    studiocast::maxine::CUstream vb_stream = nullptr;

    std::optional<studiocast::open_video::ModelPack> pack;
    std::unique_ptr<studiocast::open_cuda::OpenCudaMattingSession> session;
    studiocast::open_video::FrameArtifactCache matte_artifacts;
    studiocast::open_video::FrameMatteArtifactKey cuda_matte_artifact_key;
    studiocast::open_video::FrameMatteArtifactKey cpu_matte_artifact_key;
    bool matte_artifact_keys_valid = false;

    // GPU buffers.
    studiocast::cuda::CudaImage frame_rgb; // rgb_u8, WxH
    studiocast::cuda::CudaImage out_rgb;   // rgb_u8, WxH

    studiocast::cuda::CudaTensor alpha_tensor; // 1x1xH_inxW_in (contiguous)
    studiocast::cuda::CudaImage
        alpha_model_view; // f32_1 view over alpha_tensor (no ownership)

    // Matting inference cache. Multiple effects (VB, Auto Frame, Key Light) can
    // consume the same foreground matte in a single frame. Cache the model
    // output keyed by capture sequence so we don't re-run the ONNX model when
    // multiple effects are enabled.
    std::uint64_t cached_matte_sequence = 0;
    bool cached_matte_valid = false;

    // Optional CPU-side cache of the alpha tensor (downloaded once per frame
    // when needed).
    std::uint64_t cached_alpha_cpu_sequence = 0;
    bool cached_alpha_cpu_valid = false;
    std::vector<float> alpha_cpu;
    studiocast::cuda::CudaImage alpha_resized; // f32_1, WxH
    studiocast::cuda::CudaImage alpha_tmp;     // f32_1, WxH
    studiocast::cuda::CudaImage alpha_feather; // f32_1, WxH

    studiocast::cuda::CudaImage blur_tmp; // rgb_u8, WxH
    studiocast::cuda::CudaImage blurred;  // rgb_u8, WxH

    studiocast::cuda::CudaImage bg_rgb; // rgb_u8, WxH (replace mode)
    studiocast::cuda::CudaImage
        bg_src_rgb; // rgb_u8, WxH_src (decoded image, cached allocation)

    // Replace-mode background cache.
    // Cache the decoded+uploaded source image at its native resolution, then
    // GPU-resize into bg_rgb for the current frame size.
    std::filesystem::path cached_bg_src_path;
    std::filesystem::file_time_type cached_bg_src_mtime{};
    int cached_bg_src_w = 0;
    int cached_bg_src_h = 0;
    bool cached_bg_src_valid = false;
    std::uint64_t cached_bg_src_gen = 0;

    int cached_bg_dst_w = 0;
    int cached_bg_dst_h = 0;
    bool cached_bg_dst_valid = false;
    std::uint64_t cached_bg_dst_src_gen = 0;

    detail::PreparedReplaceBackgroundSource prepared_bg_src;
    std::vector<std::uint8_t> tmp_replace_rgb_src;

    std::uint64_t matte_frame_upload_calls = 0;
    std::uint64_t matting_inference_calls = 0;
    std::uint64_t alpha_download_calls = 0;
    std::uint64_t forced_sync_calls = 0;

    ~OpenCudaVirtualBackgroundContext() { Destroy(); }

    void Destroy() {
      if (cuda.IsInitialized() && vb_stream != nullptr) {
        std::string serr;
        (void)cuda.StreamSynchronize(vb_stream, &serr);
      }

      if (cuda.IsInitialized()) {
        (void)frame_rgb.Free(&cuda, nullptr);
        (void)out_rgb.Free(&cuda, nullptr);
        (void)alpha_resized.Free(&cuda, nullptr);
        (void)alpha_tmp.Free(&cuda, nullptr);
        (void)alpha_feather.Free(&cuda, nullptr);
        (void)blur_tmp.Free(&cuda, nullptr);
        (void)blurred.Free(&cuda, nullptr);
        (void)bg_rgb.Free(&cuda, nullptr);
        (void)bg_src_rgb.Free(&cuda, nullptr);
        (void)alpha_tensor.Free(&cuda, nullptr);

        if (vb_stream != nullptr) {
          std::string derr;
          (void)cuda.DestroyStream(vb_stream, &derr);
          vb_stream = nullptr;
        }
      }

      alpha_model_view.ClearMetadata();
      session.reset();
      pack.reset();
      active_model_id.clear();
      active_requested_model_id.clear();

      cached_matte_sequence = 0;
      cached_matte_valid = false;
      cached_alpha_cpu_sequence = 0;
      cached_alpha_cpu_valid = false;
      matte_artifacts.ClearArtifacts();
      ClearMatteArtifactKeys();

      cached_bg_src_path.clear();
      cached_bg_src_mtime = {};
      cached_bg_src_w = 0;
      cached_bg_src_h = 0;
      cached_bg_src_valid = false;
      cached_bg_src_gen = 0;

      cached_bg_dst_w = 0;
      cached_bg_dst_h = 0;
      cached_bg_dst_valid = false;
      cached_bg_dst_src_gen = 0;
      prepared_bg_src = {};
      tmp_replace_rgb_src.clear();

      initialized = false;
      enabled = false;
      last_error.clear();
    }

    void ConfigureReplaceBackgroundSource(
        const detail::PreparedReplaceBackgroundSource &source) {
      prepared_bg_src = source;
    }

    void ClearMatteArtifactKeys() {
      cuda_matte_artifact_key = {};
      cpu_matte_artifact_key = {};
      matte_artifact_keys_valid = false;
    }

    void RefreshMatteArtifactKeys(int frame_w, int frame_h) {
      if (!pack.has_value() || !pack->matting.has_value()) {
        ClearMatteArtifactKeys();
        return;
      }

      const int matte_w = pack->matting->input.width;
      const int matte_h = pack->matting->input.height;
      const auto stream_handle = reinterpret_cast<std::uintptr_t>(vb_stream);
      if (matte_artifact_keys_valid &&
          cuda_matte_artifact_key.frame_width == frame_w &&
          cuda_matte_artifact_key.frame_height == frame_h &&
          cuda_matte_artifact_key.matte_width == matte_w &&
          cuda_matte_artifact_key.matte_height == matte_h &&
          cuda_matte_artifact_key.stream == stream_handle) {
        return;
      }

      if (matte_artifact_keys_valid) {
        InvalidateMatteCache();
      }

      studiocast::open_video::FrameMatteArtifactKey base_key;
      base_key.provider_id = "open_cuda";
      base_key.model_id = active_model_id;
      base_key.frame_width = frame_w;
      base_key.frame_height = frame_h;
      base_key.matte_width = matte_w;
      base_key.matte_height = matte_h;
      base_key.stream = stream_handle;

      cpu_matte_artifact_key = base_key;
      cpu_matte_artifact_key.storage =
          studiocast::open_video::FrameMatteStorage::cpu_f32_alpha;
      cuda_matte_artifact_key = std::move(base_key);
      cuda_matte_artifact_key.storage =
          studiocast::open_video::FrameMatteStorage::device_f32_alpha;
      matte_artifact_keys_valid = true;
    }

    const studiocast::open_video::FrameMatteArtifactKey *
    MatteArtifactKeyForStorage(
        studiocast::open_video::FrameMatteStorage storage) const {
      if (!matte_artifact_keys_valid) {
        return nullptr;
      }
      if (storage == studiocast::open_video::FrameMatteStorage::cpu_f32_alpha) {
        return &cpu_matte_artifact_key;
      }
      if (storage ==
          studiocast::open_video::FrameMatteStorage::device_f32_alpha) {
        return &cuda_matte_artifact_key;
      }
      return nullptr;
    }

    bool EnsureInitialized(
        int frame_w, int frame_h,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error, bool require_vb_buffers = true) {
      if (error)
        error->clear();
      if (frame_w <= 0 || frame_h <= 0) {
        last_error = "Open CUDA: invalid frame size.";
        if (error)
          *error = last_error;
        return false;
      }

      std::string err;
      if (!cuda.IsInitialized()) {
        if (!cuda.Initialize(&err)) {
          last_error = "Open CUDA: CUDA unavailable: " + err;
          if (error)
            *error = last_error;
          return false;
        }
      }
      if (!cuda.EnsureContext(&err)) {
        last_error = "Open CUDA: failed to ensure CUDA context: " + err;
        if (error)
          *error = last_error;
        return false;
      }

      if (vb_stream == nullptr) {
        std::string serr;
        if (!cuda.CreateStream(&vb_stream, &serr)) {
          last_error = "Open CUDA: failed to create CUDA stream: " + serr;
          if (error)
            *error = last_error;
          return false;
        }
      }

      // Resolve model pack only when the requested model key changes. Empty
      // model id means "use the current default"; new installs are picked up on
      // explicit reconfiguration/refresh rather than by rescanning per frame.
      const std::string requested_model_key = fx.virtual_background.model_id;
      std::string requested_model_id = active_model_id;
      const bool needs_model_resolve =
          !initialized || !pack.has_value() || !session ||
          requested_model_key != active_requested_model_id;

      if (needs_model_resolve) {
        const auto reg =
            studiocast::open_video::ModelPackRegistry::ScanDefault();
        requested_model_id = requested_model_key;
        if (requested_model_id.empty()) {
          requested_model_id = reg.DefaultModelIdForTask("matting");
        }

        if (requested_model_id.empty()) {
          last_error = "Open CUDA: no usable model packs found (install under "
                       "~/.local/share/studiocast/models/open_video/<subject>/"
                       "<pack_dir>/).";
          if (error)
            *error = last_error;
          return false;
        }

        auto DescribeInstalledModelIds = [&reg]() -> std::string {
          std::ostringstream oss;
          std::vector<std::string> ids;
          for (const auto &m : reg.ListModels()) {
            if (m.task != "matting")
              continue;
            ids.push_back(m.id);
          }
          if (ids.empty()) {
            oss << "<none>";
            return oss.str();
          }
          bool first = true;
          for (const auto &id : ids) {
            if (!first)
              oss << ", ";
            first = false;
            oss << id;
          }
          return oss.str();
        };

        if (requested_model_id != active_model_id) {
          // Model changed: reset model-dependent state so ORT session and
          // model-sized buffers are rebuilt cleanly.
          session.reset();
          pack.reset();
          (void)alpha_tensor.Free(&cuda, nullptr);
          alpha_model_view.ClearMetadata();

          cached_matte_sequence = 0;
          cached_matte_valid = false;
          cached_alpha_cpu_sequence = 0;
          cached_alpha_cpu_valid = false;
          alpha_cpu.clear();
          matte_artifacts.ClearArtifacts();
          ClearMatteArtifactKeys();
        }

        const auto p = reg.Find("matting", requested_model_id);
        if (!p.has_value()) {
          last_error = "Open CUDA: selected model_id '" + requested_model_id +
                       "' not found. Installed: " + DescribeInstalledModelIds();
          if (error)
            *error = last_error;
          return false;
        }
        pack = *p;
        active_model_id = requested_model_id;
        active_requested_model_id = requested_model_key;

        if (pack->task != "matting") {
          last_error = "Open CUDA: selected model_id '" + requested_model_id +
                       "' has task '" + pack->task + "' (expected 'matting').";
          if (error)
            *error = last_error;
          return false;
        }
        if (!pack->matting.has_value()) {
          last_error = "Open CUDA: selected model_id '" + requested_model_id +
                       "' is missing matting metadata.";
          if (error)
            *error = last_error;
          return false;
        }
      }

      if (!session) {
        studiocast::open_cuda::OpenCudaMattingSession::Options session_opts;
        session_opts.device_id = cuda.device_ordinal();
        session_opts.enable_tensorrt = OpenCudaTensorRtEnabledFromEnv();
        session =
            std::make_unique<studiocast::open_cuda::OpenCudaMattingSession>(
                &cuda, *pack, session_opts);
      }

      // Allocate alpha tensor at model resolution.
      {
        std::string aerr;
        if (!alpha_tensor.ReallocIfNeededNchwF32(
                &cuda, 1, 1, pack->matting->input.height,
                pack->matting->input.width, &aerr)) {
          last_error = "Open CUDA: failed to allocate alpha tensor: " + aerr;
          if (error)
            *error = last_error;
          return false;
        }

        // View alpha tensor as a 2D f32 image.
        alpha_model_view.ptr = alpha_tensor.ptr;
        alpha_model_view.pitch =
            static_cast<std::size_t>(pack->matting->input.width) * 4u;
        alpha_model_view.w = pack->matting->input.width;
        alpha_model_view.h = pack->matting->input.height;
        alpha_model_view.format = studiocast::cuda::PixelFormatGpu::f32_1;
        alpha_model_view.owns_memory = false;
      }

      // Per-frame-size buffers.
      {
        std::string berr;
        if (!frame_rgb.ReallocIfNeeded(&cuda, frame_w, frame_h,
                                       studiocast::cuda::PixelFormatGpu::rgb_u8,
                                       &berr) ||
            !out_rgb.ReallocIfNeeded(&cuda, frame_w, frame_h,
                                     studiocast::cuda::PixelFormatGpu::rgb_u8,
                                     &berr)) {
          last_error = "Open CUDA: failed to allocate GPU buffers: " + berr;
          if (error)
            *error = last_error;
          return false;
        }

        // Only allocate VB-specific buffers when the caller is actually going
        // to run the virtual background stage. Auto Frame / Key Light use the
        // matte only.
        if (require_vb_buffers) {
          if (!alpha_resized.ReallocIfNeeded(
                  &cuda, frame_w, frame_h,
                  studiocast::cuda::PixelFormatGpu::f32_1, &berr) ||
              !alpha_tmp.ReallocIfNeeded(
                  &cuda, frame_w, frame_h,
                  studiocast::cuda::PixelFormatGpu::f32_1, &berr) ||
              !alpha_feather.ReallocIfNeeded(
                  &cuda, frame_w, frame_h,
                  studiocast::cuda::PixelFormatGpu::f32_1, &berr) ||
              !blur_tmp.ReallocIfNeeded(
                  &cuda, frame_w, frame_h,
                  studiocast::cuda::PixelFormatGpu::rgb_u8, &berr) ||
              !blurred.ReallocIfNeeded(&cuda, frame_w, frame_h,
                                       studiocast::cuda::PixelFormatGpu::rgb_u8,
                                       &berr)) {
            last_error = "Open CUDA: failed to allocate GPU buffers: " + berr;
            if (error)
              *error = last_error;
            return false;
          }
        }
      }

      // If the frame size changed, keep the uploaded source background but
      // invalidate the resized destination so we'll re-run the GPU resize.
      if (cached_bg_dst_w != frame_w || cached_bg_dst_h != frame_h) {
        cached_bg_dst_valid = false;
        cached_bg_dst_w = frame_w;
        cached_bg_dst_h = frame_h;
      }

      // Strength is the canonical knob; clamp once for deterministic behavior.
      (void)fx;

      RefreshMatteArtifactKeys(frame_w, frame_h);

      initialized = true;
      enabled = true;
      return true;
    }

    bool EnsureReplaceBackgroundGpu(int width, int height,
                                    const std::filesystem::path &path,
                                    std::string *error) {
      if (error)
        error->clear();
      if (!initialized) {
        if (error)
          *error = "Open CUDA: not initialized.";
        return false;
      }
      if (path.empty()) {
        if (error)
          *error = "Open CUDA: virtual_background.replace_path not set.";
        return false;
      }
      if (!prepared_bg_src.valid || prepared_bg_src.path != path) {
        if (error)
          *error = prepared_bg_src.error.empty()
                       ? "Open CUDA: replace image was not prepared."
                       : "Open CUDA: " + prepared_bg_src.error;
        return false;
      }

      const detail::ReplaceBackgroundSourceCacheSnapshot source_cache{
          cached_bg_src_path, cached_bg_src_mtime,
          cached_bg_src_valid && bg_src_rgb.Valid(), cached_bg_src_gen};
      const detail::ReplaceBackgroundResizeCacheSnapshot resize_cache{
          cached_bg_dst_w, cached_bg_dst_h,
          cached_bg_dst_valid && bg_rgb.Valid(), cached_bg_dst_src_gen};
      const auto cache_decision = detail::DecideReplaceBackgroundFrameCache(
          prepared_bg_src, source_cache, resize_cache, width, height);

      if (cache_decision.refresh_source) {
        int iw = 0, ih = 0;
        std::string img_err;
        if (!studiocast::video::LoadImageRgb24(prepared_bg_src.path, &iw, &ih,
                                               &tmp_replace_rgb_src,
                                               &img_err)) {
          if (error)
            *error = "Open CUDA: failed to load replace image: " + img_err;
          return false;
        }

        std::string berr;
        if (!bg_src_rgb.ReallocIfNeeded(
                &cuda, iw, ih, studiocast::cuda::PixelFormatGpu::rgb_u8,
                &berr)) {
          if (error)
            *error = "Open CUDA: failed to allocate bg_src_rgb: " + berr;
          return false;
        }
        if (!bg_src_rgb.UploadFromCpuRgb24(&cuda, tmp_replace_rgb_src.data(),
                                           static_cast<std::size_t>(iw) * 3u,
                                           vb_stream, &berr)) {
          if (error)
            *error = "Open CUDA: failed to upload bg_src_rgb: " + berr;
          return false;
        }

        cached_bg_src_path = prepared_bg_src.path;
        cached_bg_src_mtime = prepared_bg_src.mtime;
        cached_bg_src_w = iw;
        cached_bg_src_h = ih;
        cached_bg_src_valid = true;
        cached_bg_src_gen = cache_decision.source_generation_after_refresh;

        // Source changed -> destination must be re-generated.
        cached_bg_dst_valid = false;
      }

      if (!cache_decision.refresh_resized_destination) {
        return true;
      }

      std::string berr;
      if (!bg_rgb.ReallocIfNeeded(&cuda, width, height,
                                  studiocast::cuda::PixelFormatGpu::rgb_u8,
                                  &berr)) {
        if (error)
          *error = "Open CUDA: failed to allocate bg_rgb: " + berr;
        return false;
      }

      if (!studiocast::cuda::kernels::ResizeBilinear(bg_src_rgb, bg_rgb,
                                                     vb_stream, &berr)) {
        if (error)
          *error = "Open CUDA: failed to resize replace image on GPU: " + berr;
        return false;
      }

      cached_bg_dst_w = width;
      cached_bg_dst_h = height;
      cached_bg_dst_src_gen = cached_bg_src_gen;
      cached_bg_dst_valid = true;
      return true;
    }

    void PublishMatteArtifact(
        std::uint64_t capture_sequence,
        const studiocast::open_video::FrameMatteArtifactKey &key,
        std::uintptr_t handle, std::uintptr_t aux_handle = 0) {
      studiocast::open_video::FrameMatteArtifact artifact;
      artifact.key = key;
      artifact.handle = handle;
      artifact.aux_handle = aux_handle;
      (void)matte_artifacts.StoreMatte(capture_sequence, std::move(artifact));
    }

    void InvalidateMatteCache() {
      cached_matte_sequence = 0;
      cached_matte_valid = false;
      cached_alpha_cpu_sequence = 0;
      cached_alpha_cpu_valid = false;
      alpha_cpu.clear();
      matte_artifacts.ClearArtifacts();
    }

    // Ensure matting inference has been run for this frame, using a GPU input
    // frame. If matting was already computed for the same capture sequence,
    // this is a no-op.
    bool EnsureMatteForFrameGpu(
        const studiocast::cuda::CudaImage &in_rgb,
        std::uint64_t capture_sequence, int width, int height,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error) {
      if (error)
        error->clear();

      std::string init_err;
      if (!EnsureInitialized(width, height, fx, &init_err,
                             /*require_vb_buffers=*/true)) {
        if (error)
          *error = init_err;
        return false;
      }

      const auto *matte_key = MatteArtifactKeyForStorage(
          studiocast::open_video::FrameMatteStorage::device_f32_alpha);
      if (!matte_key) {
        if (error)
          *error = "Open CUDA: matte artifact key not initialized.";
        return false;
      }
      if (cached_matte_valid && cached_matte_sequence == capture_sequence &&
          matte_artifacts.FindMatte(capture_sequence, *matte_key)) {
        return true;
      }

      std::string matte_err;
      if (!session->Run(vb_stream, in_rgb, &alpha_tensor, &matte_err)) {
        if (error)
          *error = matte_err;
        return false;
      }
      ++matting_inference_calls;

      cached_matte_valid = true;
      cached_matte_sequence = capture_sequence;
      cached_alpha_cpu_valid = false;
      PublishMatteArtifact(capture_sequence, *matte_key,
                           static_cast<std::uintptr_t>(alpha_tensor.ptr),
                           static_cast<std::uintptr_t>(alpha_model_view.ptr));
      return true;
    }

    // Ensure matting inference has been run for this frame, using a CPU RGB
    // input frame. If matting was already computed for the same capture
    // sequence, this is a no-op.
    bool EnsureMatteForFrameCpu(
        const std::uint8_t *rgb, std::size_t rgb_stride,
        std::uint64_t capture_sequence, int width, int height,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error) {
      if (error)
        error->clear();
      if (!rgb || rgb_stride == 0) {
        if (error)
          *error = "Open CUDA: invalid RGB buffer.";
        return false;
      }

      std::string init_err;
      if (!EnsureInitialized(width, height, fx, &init_err,
                             /*require_vb_buffers=*/false)) {
        if (error)
          *error = init_err;
        return false;
      }

      const auto *matte_key = MatteArtifactKeyForStorage(
          studiocast::open_video::FrameMatteStorage::device_f32_alpha);
      if (!matte_key) {
        if (error)
          *error = "Open CUDA: matte artifact key not initialized.";
        return false;
      }
      if (cached_matte_valid && cached_matte_sequence == capture_sequence &&
          matte_artifacts.FindMatte(capture_sequence, *matte_key)) {
        return true;
      }

      std::string up_err;
      if (!frame_rgb.UploadFromCpuRgb24(&cuda, rgb, rgb_stride, vb_stream,
                                        &up_err)) {
        if (error)
          *error = "Open CUDA: frame upload failed: " + up_err;
        return false;
      }
      ++matte_frame_upload_calls;

      std::string matte_err;
      if (!session->Run(vb_stream, frame_rgb, &alpha_tensor, &matte_err)) {
        if (error)
          *error = matte_err;
        return false;
      }
      ++matting_inference_calls;

      cached_matte_valid = true;
      cached_matte_sequence = capture_sequence;
      cached_alpha_cpu_valid = false;
      PublishMatteArtifact(capture_sequence, *matte_key,
                           static_cast<std::uintptr_t>(alpha_tensor.ptr),
                           static_cast<std::uintptr_t>(alpha_model_view.ptr));
      return true;
    }

    // Download the alpha tensor (once per frame) for CPU-side consumers (Auto
    // Frame / Key Light).
    bool GetAlphaCpuForFrame(
        const std::uint8_t *rgb, std::size_t rgb_stride,
        std::uint64_t capture_sequence, int width, int height,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        const std::vector<float> **out_alpha, int *out_alpha_w,
        int *out_alpha_h, std::string *error) {
      if (error)
        error->clear();
      if (out_alpha)
        *out_alpha = nullptr;
      if (out_alpha_w)
        *out_alpha_w = 0;
      if (out_alpha_h)
        *out_alpha_h = 0;

      if (width > 0 && height > 0) {
        RefreshMatteArtifactKeys(width, height);
      }
      const auto *cached_cpu_matte_key = MatteArtifactKeyForStorage(
          studiocast::open_video::FrameMatteStorage::cpu_f32_alpha);
      if (width > 0 && height > 0 && cached_cpu_matte_key &&
          cached_alpha_cpu_valid &&
          cached_alpha_cpu_sequence == capture_sequence &&
          matte_artifacts.FindMatte(capture_sequence, *cached_cpu_matte_key)) {
        if (out_alpha)
          *out_alpha = &alpha_cpu;
        if (pack.has_value()) {
          if (out_alpha_w)
            *out_alpha_w = pack->matting->input.width;
          if (out_alpha_h)
            *out_alpha_h = pack->matting->input.height;
        }
        return true;
      }

      // If we don't have a matte yet for this frame, compute it from the CPU
      // input.
      if (!cached_matte_valid || cached_matte_sequence != capture_sequence) {
        std::string matte_err;
        if (!EnsureMatteForFrameCpu(rgb, rgb_stride, capture_sequence, width,
                                    height, fx, &matte_err)) {
          if (error)
            *error = matte_err;
          return false;
        }
      }

      if (!pack.has_value()) {
        if (error)
          *error = "Open CUDA: matting model pack not initialized.";
        return false;
      }

      std::string derr;
      if (!alpha_tensor.DownloadToCpuF32(&cuda, &alpha_cpu, vb_stream, &derr)) {
        if (error)
          *error = "Open CUDA: failed to download alpha tensor: " + derr;
        return false;
      }
      ++alpha_download_calls;
      if (!cuda.StreamSynchronize(vb_stream, &derr)) {
        if (error)
          *error = "Open CUDA: failed to download alpha tensor: " + derr;
        return false;
      }
      ++forced_sync_calls;

      cached_alpha_cpu_valid = true;
      cached_alpha_cpu_sequence = capture_sequence;
      const auto *cpu_matte_key = MatteArtifactKeyForStorage(
          studiocast::open_video::FrameMatteStorage::cpu_f32_alpha);
      if (!cpu_matte_key) {
        if (error)
          *error = "Open CUDA: matte artifact key not initialized.";
        return false;
      }
      PublishMatteArtifact(capture_sequence, *cpu_matte_key,
                           reinterpret_cast<std::uintptr_t>(alpha_cpu.data()),
                           static_cast<std::uintptr_t>(alpha_cpu.size()));
      if (out_alpha)
        *out_alpha = &alpha_cpu;
      if (out_alpha_w)
        *out_alpha_w = pack->matting->input.width;
      if (out_alpha_h)
        *out_alpha_h = pack->matting->input.height;
      return true;
    }

    // GPU-only Open CUDA stage.
    //
    // - Expects input/output as GPU RGB images.
    // - Does not perform CPU<->GPU transfers.
    bool
    ApplyGpuStage(const studiocast::cuda::CudaImage &in_rgb,
                  studiocast::cuda::CudaImage *out_rgb_img, int width,
                  int height, std::uint64_t capture_sequence,
                  const studiocast::video::effects::BroadcastCameraEffects &fx,
                  std::string *error) {
      if (error)
        error->clear();

      using studiocast::video::effects::VirtualBackgroundMode;
      if (fx.virtual_background.mode == VirtualBackgroundMode::none)
        return true;

      if (!out_rgb_img) {
        if (error)
          *error = "Open CUDA: out_rgb is null.";
        return false;
      }
      if (!in_rgb.Valid() || in_rgb.w != width || in_rgb.h != height ||
          in_rgb.format != studiocast::cuda::PixelFormatGpu::rgb_u8) {
        if (error)
          *error = "Open CUDA: invalid input GPU RGB image.";
        return false;
      }

      std::string matte_err;
      if (!EnsureMatteForFrameGpu(in_rgb, capture_sequence, width, height, fx,
                                  &matte_err)) {
        if (error)
          *error = matte_err;
        return false;
      }

      std::string berr;
      if (!out_rgb_img->ReallocIfNeeded(
              &cuda, width, height, studiocast::cuda::PixelFormatGpu::rgb_u8,
              &berr)) {
        if (error)
          *error = "Open CUDA: failed to allocate output RGB image: " + berr;
        return false;
      }

      const int strength = std::max(
          studiocast::video::effects::contract::kVbStrengthMin,
          std::min(studiocast::video::effects::contract::kVbStrengthMax,
                   fx.virtual_background.strength));

      // Resize alpha to frame size.
      std::string kerr;
      if (!studiocast::cuda::kernels::ResizeBilinearF32_1(
              alpha_model_view, alpha_resized, vb_stream, &kerr)) {
        if (error)
          *error = "Open CUDA: alpha resize failed: " + kerr;
        return false;
      }

      // Optional feathering: blur alpha with a small radius (scaled by
      // strength).
      const int feather_radius = std::min(4, strength / 16);
      const studiocast::cuda::CudaImage *alpha_use = &alpha_resized;
      if (feather_radius > 0) {
        if (!studiocast::cuda::kernels::BoxBlurSeparableF32_1(
                alpha_resized, alpha_tmp, alpha_feather, feather_radius,
                vb_stream, &kerr)) {
          if (error)
            *error = "Open CUDA: alpha feather blur failed: " + kerr;
          return false;
        }
        alpha_use = &alpha_feather;
      }

      // Build background and composite.
      if (fx.virtual_background.mode == VirtualBackgroundMode::blur) {
        if (!studiocast::cuda::kernels::BoxBlurSeparableU8x3(
                in_rgb, blur_tmp, blurred, /*radius=*/strength, vb_stream,
                &kerr)) {
          if (error)
            *error = "Open CUDA: background blur failed: " + kerr;
          return false;
        }
        if (!studiocast::cuda::kernels::CompositeAlphaU8x3(
                in_rgb, blurred, *alpha_use, *out_rgb_img, vb_stream, &kerr)) {
          if (error)
            *error = "Open CUDA: composite failed: " + kerr;
          return false;
        }
      } else if (fx.virtual_background.mode == VirtualBackgroundMode::remove) {
        std::uint32_t rgb_hex = 0x000000u;
        if (!ParseRgbHex(fx.virtual_background.remove_color, &rgb_hex)) {
          rgb_hex = 0x000000u;
        }
        const std::uint8_t r =
            static_cast<std::uint8_t>((rgb_hex >> 16) & 0xFFu);
        const std::uint8_t g =
            static_cast<std::uint8_t>((rgb_hex >> 8) & 0xFFu);
        const std::uint8_t b = static_cast<std::uint8_t>((rgb_hex)&0xFFu);
        if (!studiocast::cuda::kernels::CompositeAlphaSolidU8x3(
                in_rgb, *alpha_use, r, g, b, *out_rgb_img, vb_stream, &kerr)) {
          if (error)
            *error = "Open CUDA: composite(solid) failed: " + kerr;
          return false;
        }
      } else if (fx.virtual_background.mode == VirtualBackgroundMode::replace) {
        std::string bg_err;
        if (!EnsureReplaceBackgroundGpu(
                width, height, fx.virtual_background.replace_path, &bg_err)) {
          if (error)
            *error = bg_err;
          return false;
        }
        if (!studiocast::cuda::kernels::CompositeAlphaU8x3(
                in_rgb, bg_rgb, *alpha_use, *out_rgb_img, vb_stream, &kerr)) {
          if (error)
            *error = "Open CUDA: composite(replace) failed: " + kerr;
          return false;
        }
      }

      return true;
    }

    // GPU-in / GPU-out entrypoint for Open CUDA VB.
    //
    // - Expects input/output as GPU RGB images.
    // - Does not perform CPU<->GPU transfers.
    // - Always populates DeferredGpuOut to describe the produced GPU frame.
    bool
    ApplyCudaRgb(const studiocast::cuda::CudaImage &in_rgb,
                 studiocast::cuda::CudaImage *out_rgb_img,
                 const studiocast::video::effects::BroadcastCameraEffects &fx,
                 std::uint64_t capture_sequence, std::string *error,
                 DeferredGpuOut *deferred_out) {
      if (error)
        error->clear();

      using studiocast::video::effects::VirtualBackgroundMode;
      if (fx.virtual_background.mode == VirtualBackgroundMode::none) {
        if (deferred_out) {
          deferred_out->kind = DeferredGpuKind::none;
          deferred_out->nvcv_img = nullptr;
          deferred_out->nvcv = nullptr;
          deferred_out->cuda_img = nullptr;
          deferred_out->cuda = nullptr;
          deferred_out->stream = nullptr;
        }
        return true;
      }

      if (!deferred_out) {
        if (error)
          *error = "Open CUDA: ApplyCudaRgb requires deferred_out.";
        return false;
      }

      std::string stage_err;
      if (!ApplyGpuStage(in_rgb, out_rgb_img, in_rgb.w, in_rgb.h,
                         capture_sequence, fx, &stage_err)) {
        if (error)
          *error = stage_err;
        return false;
      }

      deferred_out->kind = DeferredGpuKind::cuda_rgb;
      deferred_out->nvcv_img = nullptr;
      deferred_out->nvcv = nullptr;
      deferred_out->cuda_img = out_rgb_img;
      deferred_out->cuda = &cuda;
      deferred_out->stream = vb_stream;
      return true;
    }

    bool ApplyRgbInPlace(
        std::uint8_t *rgb, int width, int height, std::size_t rgb_stride,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error, bool defer_readback, DeferredGpuOut *deferred_out) {
      if (error)
        error->clear();

      using studiocast::video::effects::VirtualBackgroundMode;
      if (fx.virtual_background.mode == VirtualBackgroundMode::none)
        return true;
      if (!rgb || width <= 0 || height <= 0 || rgb_stride == 0) {
        if (error)
          *error = "Open CUDA: invalid RGB buffer.";
        return false;
      }

      std::string init_err;
      if (!EnsureInitialized(width, height, fx, &init_err)) {
        if (error)
          *error = init_err;
        return false;
      }

      // Upload CPU->GPU.
      std::string up_err;
      if (!frame_rgb.UploadFromCpuRgb24(&cuda, rgb, rgb_stride, vb_stream,
                                        &up_err)) {
        if (error)
          *error = "Open CUDA: frame upload failed: " + up_err;
        return false;
      }

      DeferredGpuOut tmp_deferred_out{};
      DeferredGpuOut *use_deferred_out =
          defer_readback ? deferred_out : &tmp_deferred_out;
      if (!use_deferred_out) {
        if (error)
          *error = "Open CUDA: defer_readback requires deferred_out.";
        return false;
      }

      std::string stage_err;
      if (!ApplyCudaRgb(frame_rgb, &out_rgb, fx, /*capture_sequence=*/0,
                        &stage_err, use_deferred_out)) {
        if (error)
          *error = stage_err;
        return false;
      }

      if (defer_readback) {
        return true;
      }

      // Download GPU->CPU.
      std::string down_err;
      if (!out_rgb.DownloadToCpuRgb24(&cuda, rgb, rgb_stride, vb_stream,
                                      &down_err)) {
        if (error)
          *error = "Open CUDA: frame download failed: " + down_err;
        return false;
      }
      if (!cuda.StreamSynchronize(vb_stream, &down_err)) {
        if (error)
          *error = "Open CUDA: StreamSynchronize failed: " + down_err;
        return false;
      }
      return true;
    }
  } open_cuda_vb;

#if STUDIOCAST_ENABLE_OPEN_VULKAN
  struct OpenVulkanVirtualBackgroundContext {
    bool initialized = false;
    bool enabled = false;

    std::string last_error;
    std::string active_model_id;
    std::string active_requested_model_id;
    bool matting_session_initialized = false;
    bool matting_session_warmed = false;
    int matting_session_frame_w = 0;
    int matting_session_frame_h = 0;
    bool matting_failure_latched = false;
    std::uint64_t matting_session_context_id = 0;
    std::uint64_t matting_session_context_generation = 0;

    studiocast::vulkan::kernels::UtilityKernels kernels;
    std::optional<studiocast::open_video::ModelPack> model_pack;
    std::unique_ptr<studiocast::open_vulkan::OpenVulkanMattingSession>
        matting_session;
    OpenVulkanVirtualBackgroundBlur blur_effect;
    OpenVulkanVirtualBackgroundBlurCounters blur_effect_counters;
    OpenVulkanVirtualBackgroundRemove remove_effect;
    OpenVulkanVirtualBackgroundRemoveCounters remove_effect_counters;
    OpenVulkanVirtualBackgroundReplace replace_effect;
    OpenVulkanVirtualBackgroundReplaceCounters replace_effect_counters;
    OpenVulkanVirtualKeyLight key_light_effect;
    OpenVulkanVirtualKeyLightCounters key_light_effect_counters;

    studiocast::open_video::FrameArtifactCache matte_artifacts;
    studiocast::open_video::FrameMatteArtifactKey vulkan_matte_artifact_key;
    bool vulkan_matte_artifact_key_valid = false;

    studiocast::vulkan::VulkanImage frame_rgb;      // rgb_u8, WxH
    studiocast::vulkan::VulkanImage out_rgb;        // rgb_u8, WxH
    studiocast::vulkan::VulkanImage alpha_model;    // f32_1, model WxH
    studiocast::vulkan::VulkanImage alpha_readback; // mapped staging, model WxH
    studiocast::vulkan::VulkanImage alpha_resized;  // f32_1, frame WxH
    studiocast::vulkan::VulkanImage alpha_tmp;      // f32_1, frame WxH
    studiocast::vulkan::VulkanImage alpha_feather;  // f32_1, frame WxH
    studiocast::vulkan::VulkanImage blur_tmp;       // rgb_u8, WxH
    studiocast::vulkan::VulkanImage blurred;        // rgb_u8, WxH
    studiocast::vulkan::VulkanImage auto_frame_rgb; // rgb_u8, WxH
    // Mirror/fixed-center-vignette production transport and resident block.
    // The two mapped resources are boundary-only staging; source, resize, mask,
    // and effect outputs are separate non-mapped DEVICE_LOCAL allocations.
    studiocast::vulkan::VulkanImage final_upload_staging;   // mapped capture RGB
    studiocast::vulkan::VulkanImage final_source_rgb;      // device-local RGB
    studiocast::vulkan::VulkanImage final_readback_staging; // mapped final RGB
    studiocast::vulkan::VulkanImage vignette_rgb;   // rgb_u8, final output WxH
    studiocast::vulkan::VulkanImage mirror_rgb;     // rgb_u8, final output WxH

    detail::PreparedReplaceBackgroundSource prepared_bg_src;

    std::uint64_t cached_matte_sequence = 0;
    bool cached_matte_valid = false;
    std::uint64_t cached_frame_alpha_sequence = 0;
    bool cached_frame_alpha_valid = false;
    std::uint64_t cached_alpha_cpu_sequence = 0;
    bool cached_alpha_cpu_valid = false;
    std::vector<float> alpha_cpu;

    std::uint64_t alpha_download_calls = 0;
    std::uint64_t preprocess_dispatch_calls = 0;
    std::uint64_t matting_inference_calls = 0;
    std::uint64_t alpha_resize_dispatch_calls = 0;
    std::uint64_t alpha_resize_completion_count = 0;
    std::uint64_t blur_dispatch_calls = 0;
    std::uint64_t composite_dispatch_calls = 0;
    std::uint64_t crop_resize_dispatch_calls = 0;
    std::uint64_t forced_sync_calls = 0;

    ~OpenVulkanVirtualBackgroundContext() { Destroy(); }

    void Destroy() {
      FreeImages();
      matting_session.reset();
      blur_effect.Shutdown();
      remove_effect.Shutdown();
      replace_effect.Shutdown();
      key_light_effect.Shutdown();
      model_pack.reset();
      ResetMattingSessionState();
      kernels.Shutdown();
      initialized = false;
      enabled = false;
      last_error.clear();
      active_model_id.clear();
      active_requested_model_id.clear();
      ClearMatteArtifactKey();
      InvalidateMatteCache();

      prepared_bg_src = {};
    }

    void ConfigureReplaceBackgroundSource(
        const detail::PreparedReplaceBackgroundSource &source) {
      prepared_bg_src = source;
    }

    void ResetMattingSessionState() {
      matting_session_initialized = false;
      matting_session_warmed = false;
      matting_session_frame_w = 0;
      matting_session_frame_h = 0;
      matting_failure_latched = false;
      matting_session_context_id = 0;
      matting_session_context_generation = 0;
    }

    void FreeImages() {
      frame_rgb.Free();
      out_rgb.Free();
      alpha_model.Free();
      alpha_readback.Free();
      alpha_resized.Free();
      alpha_tmp.Free();
      alpha_feather.Free();
      blur_tmp.Free();
      blurred.Free();
      auto_frame_rgb.Free();
      final_upload_staging.Free();
      final_source_rgb.Free();
      final_readback_staging.Free();
      vignette_rgb.Free();
      mirror_rgb.Free();
    }

    void ClearMatteArtifactKey() {
      vulkan_matte_artifact_key =
          studiocast::open_video::FrameMatteArtifactKey{};
      vulkan_matte_artifact_key_valid = false;
      matte_artifacts.ClearArtifacts();
    }

    void InvalidateMatteCache() {
      cached_matte_sequence = 0;
      cached_matte_valid = false;
      cached_frame_alpha_sequence = 0;
      cached_frame_alpha_valid = false;
      cached_alpha_cpu_sequence = 0;
      cached_alpha_cpu_valid = false;
      alpha_cpu.clear();
      matte_artifacts.ClearArtifacts();
    }

    bool EnsureImage(studiocast::vulkan::VulkanImage *image, int width,
                     int height, studiocast::vulkan::VulkanPixelFormat format,
                     bool map_memory, const char *label,
                     std::string *error_out) {
      if (!image) {
        if (error_out)
          *error_out =
              std::string("Open Vulkan: null image for ") + label + ".";
        return false;
      }
      if (image->Valid() && image->width() == width &&
          image->height() == height && image->format() == format &&
          (map_memory ? image->mapped() != nullptr
                      : image->mapped() == nullptr) &&
          (map_memory || image->device_local())) {
        return true;
      }
      std::string err;
      kernels.InvalidateDescriptorBindingCacheForSetup();
      if (!image->Allocate(kernels.device(), width, height, format, map_memory,
                           &err)) {
        if (error_out)
          *error_out = std::string("Open Vulkan: failed to allocate ") + label +
                       ": " + err;
        return false;
      }
      return true;
    }

    bool UploadRgbToImage(const std::uint8_t *rgb, std::size_t stride,
                          int width, int height,
                          studiocast::vulkan::VulkanImage *image,
                          std::string *error_out) {
      if (!rgb || stride == 0 || width <= 0 || height <= 0) {
        if (error_out)
          *error_out = "Open Vulkan: invalid RGB upload input.";
        return false;
      }
      if (!EnsureImage(image, width, height,
                       studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                       /*map_memory=*/true, "RGB upload image", error_out)) {
        return false;
      }
      if (!image->mapped()) {
        if (error_out)
          *error_out = "Open Vulkan: RGB upload image is not mapped.";
        return false;
      }
      studiocast::vulkan::PackRgb24ToRgba32(
          rgb, stride, width, height,
          static_cast<std::uint32_t *>(image->mapped()), image->pitch_pixels());
      std::string ferr;
      if (!image->Flush(&ferr)) {
        if (error_out)
          *error_out = "Open Vulkan: failed to flush RGB upload image: " + ferr;
        return false;
      }
      return true;
    }

    bool UploadRgbToDeviceLocal(const std::uint8_t *rgb, std::size_t stride,
                                int width, int height,
                                std::string *error_out) {
      if (!EnsureImage(&final_upload_staging, width, height,
                       studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                       /*map_memory=*/true, "final upload staging", error_out) ||
          !EnsureImage(&final_source_rgb, width, height,
                       studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                       /*map_memory=*/false, "final resident source",
                       error_out)) {
        return false;
      }
      if (!final_source_rgb.device_local() || final_source_rgb.mapped()) {
        if (error_out) {
          *error_out =
              "[vulkan_effect_residency_contract_failed] Open Vulkan final "
              "source is not non-mapped DEVICE_LOCAL.";
        }
        return false;
      }
      return kernels.UploadRgb24ToDeviceLocal(
          rgb, stride, final_upload_staging, final_source_rgb, error_out);
    }

    bool DownloadImageToRgb(const studiocast::vulkan::VulkanImage &image,
                            std::uint8_t *rgb, std::size_t stride,
                            std::string *error_out) {
      if (!image.Valid() ||
          image.format() != studiocast::vulkan::VulkanPixelFormat::rgb_u8) {
        if (error_out)
          *error_out = "Open Vulkan: invalid RGB image for download.";
        return false;
      }
      if (!rgb || stride == 0) {
        if (error_out)
          *error_out = "Open Vulkan: invalid RGB download output.";
        return false;
      }
      if (!image.mapped()) {
        if (!image.device_local()) {
          if (error_out) {
            *error_out = "[vulkan_effect_residency_contract_failed] Open "
                         "Vulkan RGB download source is neither mapped "
                         "transport nor DEVICE_LOCAL.";
          }
          return false;
        }
        if (!EnsureImage(&final_readback_staging, image.width(), image.height(),
                         image.format(), /*map_memory=*/true,
                         "final readback staging", error_out)) {
          return false;
        }
        return kernels.ReadbackU8x3(image, final_readback_staging, rgb, stride,
                                    error_out);
      }
      std::string ierr;
      if (!image.Invalidate(&ierr)) {
        if (error_out)
          *error_out =
              "Open Vulkan: failed to invalidate RGB download image: " + ierr;
        return false;
      }
      studiocast::vulkan::UnpackRgba32ToRgb24(
          static_cast<const std::uint32_t *>(image.mapped()),
          image.pitch_pixels(), image.width(), image.height(), rgb, stride);
      return true;
    }

    void PublishMatteArtifact(
        std::uint64_t capture_sequence,
        const studiocast::open_video::FrameMatteArtifactKey &key,
        std::uintptr_t handle, std::uintptr_t aux_handle = 0) {
      studiocast::open_video::FrameMatteArtifact artifact;
      artifact.key = key;
      artifact.handle = handle;
      artifact.aux_handle = aux_handle;
      (void)matte_artifacts.StoreMatte(capture_sequence, std::move(artifact));
    }

    std::string ResolveMattingModelId(
        const studiocast::open_video::ModelPackRegistry &registry,
        const std::string &requested_model_id) const {
      if (!requested_model_id.empty())
        return requested_model_id;

      const auto is_production_candidate = [&](const std::string &id) {
        const auto candidate = registry.Find("matting", id);
        return candidate && candidate->ncnn_vulkan.has_value();
      };
      if (is_production_candidate("modnet-webnn-256-fp32"))
        return "modnet-webnn-256-fp32";
      if (is_production_candidate("birefnet_lite"))
        return "birefnet_lite";
      for (const auto &m : registry.ListModels()) {
        if (m.task == "matting" && is_production_candidate(m.id))
          return m.id;
      }
      return {};
    }

    std::string MattingReadinessError(bool model_selected,
                                      bool model_validated) const {
      studiocast::open_vulkan::VulkanMattingReadinessInput input;
      input.production_build_enabled =
          studiocast::open_vulkan::ProductionVulkanMattingBuildEnabled();
      input.production_adapter_available =
          studiocast::open_vulkan::ProductionVulkanMattingAdapterAvailable();
      input.model_pack_selected = model_selected;
      input.model_contract_validated = model_validated;
      if (kernels.device()) {
        const auto diagnostics = kernels.device()->DiagnosticsSnapshot();
        input.non_cpu_device_selected = diagnostics.non_cpu_device_selected;
        input.compute_queue_available = diagnostics.compute_queue_available;
        input.context_healthy = diagnostics.context_healthy;
      }
      return "Open Vulkan: " +
             studiocast::open_vulkan::FormatVulkanMattingReadiness(
                 studiocast::open_vulkan::EvaluateVulkanMattingReadiness(
                     input));
    }

    void RefreshMatteArtifactKey(int frame_w, int frame_h) {
      if (!model_pack || !model_pack->matting) {
        ClearMatteArtifactKey();
        return;
      }
      const int matte_w = model_pack->matting->output.width;
      const int matte_h = model_pack->matting->output.height;
      const auto device_handle =
          reinterpret_cast<std::uintptr_t>(kernels.device());
      const auto queue_handle =
          reinterpret_cast<std::uintptr_t>(kernels.device()->queue());
      if (vulkan_matte_artifact_key_valid &&
          vulkan_matte_artifact_key.frame_width == frame_w &&
          vulkan_matte_artifact_key.frame_height == frame_h &&
          vulkan_matte_artifact_key.matte_width == matte_w &&
          vulkan_matte_artifact_key.matte_height == matte_h &&
          vulkan_matte_artifact_key.device_context == device_handle &&
          vulkan_matte_artifact_key.stream == queue_handle) {
        return;
      }
      if (vulkan_matte_artifact_key_valid) {
        InvalidateMatteCache();
      }
      studiocast::open_video::FrameMatteArtifactKey key;
      key.provider_id = "open_vulkan";
      key.model_id = active_model_id;
      key.storage = studiocast::open_video::FrameMatteStorage::device_f32_alpha;
      key.frame_width = frame_w;
      key.frame_height = frame_h;
      key.matte_width = matte_w;
      key.matte_height = matte_h;
      key.config_fingerprint = 0;
      key.device_context = device_handle;
      key.stream = queue_handle;
      vulkan_matte_artifact_key = key;
      vulkan_matte_artifact_key_valid = true;
    }

    bool HasCompatibleFrameAlpha(std::uint64_t capture_sequence, int frame_w,
                                 int frame_h) const {
      const auto *artifact =
          capture_sequence != 0 && vulkan_matte_artifact_key_valid
              ? matte_artifacts.FindMatte(capture_sequence,
                                          vulkan_matte_artifact_key)
              : nullptr;
      if (!model_pack || !model_pack->matting || !kernels.device() ||
          !vulkan_matte_artifact_key_valid || !cached_matte_valid ||
          !cached_frame_alpha_valid ||
          !detail::IsOpenVulkanVirtualKeyLightSameFrameArtifactCompatible(
              capture_sequence, cached_matte_sequence,
              cached_frame_alpha_sequence, vulkan_matte_artifact_key, artifact,
              active_model_id, frame_w, frame_h,
              model_pack->matting->output.width,
              model_pack->matting->output.height,
              reinterpret_cast<std::uintptr_t>(kernels.device()),
              reinterpret_cast<std::uintptr_t>(kernels.device()->queue()),
              reinterpret_cast<std::uintptr_t>(&alpha_model),
              reinterpret_cast<std::uintptr_t>(alpha_model.buffer())) ||
          !alpha_resized.Valid() ||
          !alpha_resized.BelongsTo(*kernels.device()) ||
          alpha_resized.context_identity() !=
              kernels.device()->context_identity() ||
          alpha_resized.width() != frame_w ||
          alpha_resized.height() != frame_h || alpha_resized.mapped() ||
          !alpha_resized.device_local()) {
        return false;
      }
      return true;
    }

    bool EnsureInitialized(
        int frame_w, int frame_h,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error_out, bool require_vb_buffers = true,
        bool require_cpu_alpha_readback = false) {
      if (error_out)
        error_out->clear();
      if (frame_w <= 0 || frame_h <= 0) {
        last_error = "Open Vulkan: invalid frame size.";
        if (error_out)
          *error_out = last_error;
        return false;
      }

      std::string e;
      if (!kernels.Initialized() && !kernels.Initialize(&e)) {
        last_error = "Open Vulkan: utility kernels unavailable: " + e;
        if (error_out)
          *error_out = last_error;
        return false;
      }

      const std::string requested_model_id = fx.virtual_background.model_id;

      detail::OpenVulkanMattingSetupSnapshot setup_state;
      setup_state.has_model_pack = model_pack.has_value();
      setup_state.has_matting_session = static_cast<bool>(matting_session);
      setup_state.session_initialized = matting_session_initialized;
      setup_state.session_warmed = matting_session_warmed;
      setup_state.session_frame_w = matting_session_frame_w;
      setup_state.session_frame_h = matting_session_frame_h;
      setup_state.failure_latched = matting_failure_latched;
      const auto current_context = kernels.device()->context_identity();
      setup_state.session_context_id = matting_session_context_id;
      setup_state.session_context_generation =
          matting_session_context_generation;
      setup_state.current_context_id = current_context.context_id;
      setup_state.current_context_generation = current_context.generation;
      setup_state.active_model_id = active_model_id;
      setup_state.active_requested_model_id = active_requested_model_id;
      const auto setup_decision = detail::DecideOpenVulkanMattingSetup(
          setup_state, frame_w, frame_h, requested_model_id);

      if (setup_decision.reuse_latched_failure) {
        if (error_out)
          *error_out = last_error;
        return false;
      }

      if (setup_decision.scan_model_registry) {
        const auto registry =
            studiocast::open_video::ModelPackRegistry::ScanDefault();
        const std::string model_id =
            ResolveMattingModelId(registry, requested_model_id);
        if (model_id.empty()) {
          last_error = MattingReadinessError(/*model_selected=*/false,
                                             /*model_validated=*/false);
          matting_failure_latched = true;
          active_requested_model_id = requested_model_id;
          if (error_out)
            *error_out = last_error;
          return false;
        }

        auto resolved = registry.Find("matting", model_id);
        if (!resolved) {
          const bool malformed_requested_pack =
              registry.Problems().find(model_id) != registry.Problems().end();
          last_error = MattingReadinessError(
              /*model_selected=*/malformed_requested_pack,
              /*model_validated=*/false);
          matting_failure_latched = true;
          active_requested_model_id = requested_model_id;
          if (error_out)
            *error_out = last_error;
          return false;
        }

        model_pack = std::move(resolved);
        active_model_id = model_id;
        active_requested_model_id = requested_model_id;
        ClearMatteArtifactKey();
        InvalidateMatteCache();
      }

      if (setup_decision.recreate_session) {
        matting_session.reset();
        ResetMattingSessionState();
        if (!model_pack || !model_pack->matting) {
          last_error = MattingReadinessError(/*model_selected=*/false,
                                             /*model_validated=*/false);
          matting_failure_latched = true;
          active_requested_model_id = requested_model_id;
          if (error_out)
            *error_out = last_error;
          return false;
        }
        studiocast::open_vulkan::OpenVulkanMattingSession::Options opts;
        opts.require_device_residency = true;
        matting_session =
            std::make_unique<studiocast::open_vulkan::OpenVulkanMattingSession>(
                kernels.device(), &kernels, *model_pack, opts);
        matting_session_context_id = current_context.context_id;
        matting_session_context_generation = current_context.generation;
      }

      if (!matting_session) {
        last_error = "Open Vulkan: matting session is not available.";
        matting_failure_latched = true;
        if (error_out)
          *error_out = last_error;
        return false;
      }

      if (setup_decision.initialize_session) {
        if (!matting_session->EnsureInitialized(frame_w, frame_h, &e)) {
          matting_session_initialized = false;
          matting_session_warmed = false;
          last_error = "Open Vulkan: " + e;
          matting_failure_latched = true;
          if (error_out)
            *error_out = last_error;
          return false;
        }
        matting_session_initialized = true;
        matting_session_warmed = false;
        matting_session_frame_w = frame_w;
        matting_session_frame_h = frame_h;
      }
      if (setup_decision.warmup_session || !matting_session_warmed) {
        if (!matting_session->Warmup(&e)) {
          matting_session_warmed = false;
          last_error = "Open Vulkan: " + e;
          matting_failure_latched = true;
          if (error_out)
            *error_out = last_error;
          return false;
        }
        matting_session_warmed = true;
      }

      const auto readiness = matting_session->Readiness();
      if (!readiness.production_ready) {
        matting_failure_latched = true;
        last_error =
            "Open Vulkan: " +
            studiocast::open_vulkan::FormatVulkanMattingReadiness(readiness);
        if (error_out)
          *error_out = last_error;
        return false;
      }

      RefreshMatteArtifactKey(frame_w, frame_h);

      const int matte_w = model_pack->matting->output.width;
      const int matte_h = model_pack->matting->output.height;
      if (!EnsureImage(&frame_rgb, frame_w, frame_h,
                       studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                       /*map_memory=*/false, "resident frame_rgb", error_out) ||
          !EnsureImage(&out_rgb, frame_w, frame_h,
                       studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                       /*map_memory=*/false, "resident out_rgb", error_out) ||
          !EnsureImage(&alpha_model, matte_w, matte_h,
                       studiocast::vulkan::VulkanPixelFormat::f32_1,
                       /*map_memory=*/false, "alpha_model", error_out) ||
          !EnsureImage(&alpha_resized, frame_w, frame_h,
                       studiocast::vulkan::VulkanPixelFormat::f32_1,
                       /*map_memory=*/false, "alpha_resized", error_out)) {
        last_error = error_out ? *error_out : "Open Vulkan allocation failed.";
        if (error_out)
          *error_out = last_error;
        return false;
      }
      if (require_cpu_alpha_readback) {
        if (!EnsureImage(&alpha_readback, matte_w, matte_h,
                         studiocast::vulkan::VulkanPixelFormat::f32_1,
                         /*map_memory=*/true, "alpha_readback", error_out)) {
          last_error =
              "[vulkan_auto_frame_alpha_readback_setup_failed] Explicit "
              "degraded Auto Frame alpha readback staging allocation failed: " +
              (error_out ? *error_out : "Open Vulkan allocation failed.");
          if (error_out)
            *error_out = last_error;
          return false;
        }
        const std::size_t alpha_count =
            static_cast<std::size_t>(matte_w) *
            static_cast<std::size_t>(matte_h);
        if (alpha_cpu.size() != alpha_count)
          alpha_cpu.resize(alpha_count);
      }

      if (require_vb_buffers) {
        if (!EnsureImage(&alpha_tmp, frame_w, frame_h,
                         studiocast::vulkan::VulkanPixelFormat::f32_1,
                         /*map_memory=*/false, "alpha_tmp", error_out) ||
            !EnsureImage(&alpha_feather, frame_w, frame_h,
                         studiocast::vulkan::VulkanPixelFormat::f32_1,
                         /*map_memory=*/false, "alpha_feather", error_out)) {
          last_error =
              error_out ? *error_out : "Open Vulkan allocation failed.";
          if (error_out)
            *error_out = last_error;
          return false;
        }
        if (fx.virtual_background.mode ==
            studiocast::video::effects::VirtualBackgroundMode::blur) {
          if (!EnsureImage(&blur_tmp, frame_w, frame_h,
                           studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                           /*map_memory=*/false, "blur_tmp", error_out) ||
              !EnsureImage(&blurred, frame_w, frame_h,
                           studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                           /*map_memory=*/false, "blurred", error_out)) {
            last_error =
                error_out ? *error_out : "Open Vulkan allocation failed.";
            if (error_out)
              *error_out = last_error;
            return false;
          }
          const auto matting_readiness = matting_session->Readiness();
          std::string blur_error;
          if (!blur_effect.EnsureInitialized(&kernels, frame_w, frame_h,
                                             matting_readiness,
                                             &blur_error)) {
            last_error = blur_error;
            if (error_out)
              *error_out = last_error;
            return false;
          }
        } else if (fx.virtual_background.mode ==
                   studiocast::video::effects::VirtualBackgroundMode::remove) {
          const auto matting_readiness = matting_session->Readiness();
          std::string remove_error;
          if (!remove_effect.EnsureInitialized(
                  &kernels, frame_w, frame_h, fx.virtual_background.strength,
                  fx.virtual_background.remove_color, matting_readiness,
                  &remove_error)) {
            last_error = remove_error;
            if (error_out)
              *error_out = last_error;
            return false;
          }
        } else if (fx.virtual_background.mode ==
                   studiocast::video::effects::VirtualBackgroundMode::replace) {
          const auto matting_readiness = matting_session->Readiness();
          std::string replace_error;
          if (prepared_bg_src.path != fx.virtual_background.replace_path) {
            replace_error = OpenVulkanVirtualBackgroundReplaceAssetInvalidFailure(
                "prepared replacement path does not match the active "
                "configuration");
          } else if (!replace_effect.EnsureInitialized(
                         &kernels, frame_w, frame_h,
                         fx.virtual_background.strength, prepared_bg_src,
                         matting_readiness, &replace_effect_counters,
                         &replace_error)) {
            // The wrapper already supplied the effect-specific outer reason.
          }
          if (!replace_error.empty()) {
            if (replace_error.find(
                    "[vulkan_virtual_background_replace_initialization_failed]") ==
                std::string::npos) {
              replace_error =
                  OpenVulkanVirtualBackgroundReplaceInitializationFailure(
                      replace_error);
            }
            last_error = replace_error;
            if (error_out)
              *error_out = last_error;
            return false;
          }
        }
      }

      initialized = true;
      enabled = true;
      last_error.clear();
      return true;
    }

    bool EnsureMatteForFrameGpu(
        const studiocast::vulkan::VulkanImage &in_rgb,
        std::uint64_t capture_sequence, int width, int height,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error_out) {
      if (error_out)
        error_out->clear();

      std::string init_err;
      if (!EnsureInitialized(width, height, fx, &init_err,
                             /*require_vb_buffers=*/false)) {
        if (error_out)
          *error_out = init_err;
        return false;
      }
      if (!vulkan_matte_artifact_key_valid) {
        if (error_out)
          *error_out = "Open Vulkan: matte artifact key not initialized.";
        return false;
      }
      if (cached_matte_valid && cached_matte_sequence == capture_sequence &&
          matte_artifacts.FindMatte(capture_sequence,
                                    vulkan_matte_artifact_key)) {
        return true;
      }

      // A production Run() has two independently synchronous boundaries: the
      // utility preprocess submission and the resident runtime inference. Count
      // the observed completion deltas, including work that completed before a
      // later failure, instead of treating the whole call as one hardcoded
      // sync.
      const std::uint64_t utility_submissions_before =
          kernels.synchronous_submission_count();
      const std::uint64_t runtime_completions_before =
          matting_session->RuntimeEvidence().completion_count;
      std::string matte_err;
      const bool matting_run_ok =
          matting_session->Run(in_rgb, &alpha_model, &matte_err);
      const std::uint64_t utility_submissions_after =
          kernels.synchronous_submission_count();
      const std::uint64_t runtime_completions_after =
          matting_session->RuntimeEvidence().completion_count;
      if (utility_submissions_after >= utility_submissions_before) {
        forced_sync_calls +=
            utility_submissions_after - utility_submissions_before;
      }
      if (runtime_completions_after >= runtime_completions_before) {
        forced_sync_calls +=
            runtime_completions_after - runtime_completions_before;
      }
      if (!matting_run_ok) {
        matting_failure_latched = true;
        last_error = matte_err;
        if (error_out)
          *error_out = matte_err;
        return false;
      }
      ++preprocess_dispatch_calls;
      ++matting_inference_calls;

      cached_matte_valid = true;
      cached_matte_sequence = capture_sequence;
      cached_frame_alpha_valid = false;
      cached_alpha_cpu_valid = false;
      PublishMatteArtifact(
          capture_sequence, vulkan_matte_artifact_key,
          reinterpret_cast<std::uintptr_t>(&alpha_model),
          reinterpret_cast<std::uintptr_t>(alpha_model.buffer()));
      return true;
    }

    bool EnsureFrameAlphaForFrameGpu(
        const studiocast::vulkan::VulkanImage &in_rgb,
        std::uint64_t capture_sequence, int width, int height,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error_out) {
      if (!EnsureMatteForFrameGpu(in_rgb, capture_sequence, width, height, fx,
                                  error_out)) {
        return false;
      }
      if (cached_frame_alpha_valid &&
          cached_frame_alpha_sequence == capture_sequence &&
          alpha_resized.Valid() && alpha_resized.width() == width &&
          alpha_resized.height() == height) {
        return true;
      }
      std::string kerr;
      if (!kernels.ResizeBilinearF32_1(alpha_model, alpha_resized, &kerr)) {
        if (error_out)
          *error_out = "Open Vulkan: alpha resize failed: " + kerr;
        return false;
      }
      ++alpha_resize_dispatch_calls;
      ++forced_sync_calls;
      ++alpha_resize_completion_count;
      cached_frame_alpha_valid = true;
      cached_frame_alpha_sequence = capture_sequence;
      return true;
    }

    bool GetAlphaCpuForFrame(
        const studiocast::vulkan::VulkanImage &in_rgb,
        std::uint64_t capture_sequence, int width, int height,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        const std::vector<float> **out_alpha, int *out_alpha_w,
        int *out_alpha_h, std::string *error_out) {
      if (error_out)
        error_out->clear();
      if (out_alpha)
        *out_alpha = nullptr;
      if (out_alpha_w)
        *out_alpha_w = 0;
      if (out_alpha_h)
        *out_alpha_h = 0;

      if (!model_pack || !model_pack->matting) {
        if (error_out)
          *error_out = "Open Vulkan: matting model pack not initialized.";
        return false;
      }
      const int matte_w = model_pack->matting->output.width;
      const int matte_h = model_pack->matting->output.height;
      if (cached_alpha_cpu_valid &&
          cached_alpha_cpu_sequence == capture_sequence) {
        if (out_alpha)
          *out_alpha = &alpha_cpu;
        if (out_alpha_w)
          *out_alpha_w = matte_w;
        if (out_alpha_h)
          *out_alpha_h = matte_h;
        return true;
      }

      if (!alpha_readback.Valid() || !alpha_readback.mapped() ||
          alpha_cpu.size() !=
              static_cast<std::size_t>(matte_w) *
                  static_cast<std::size_t>(matte_h)) {
        if (error_out) {
          *error_out =
              "Open Vulkan: CPU alpha readback was not explicitly prepared "
              "for degraded Auto Frame matte tracking.";
        }
        return false;
      }

      if (!EnsureMatteForFrameGpu(in_rgb, capture_sequence, width, height, fx,
                                  error_out)) {
        return false;
      }
      std::string readback_error;
      if (!kernels.ReadbackF32_1(alpha_model, alpha_readback, alpha_cpu.data(),
                                 alpha_cpu.size(), &readback_error)) {
        if (error_out)
          *error_out = "Open Vulkan: explicit degraded alpha readback failed: " +
                       readback_error;
        return false;
      }
      ++alpha_download_calls;
      ++forced_sync_calls;
      cached_alpha_cpu_valid = true;
      cached_alpha_cpu_sequence = capture_sequence;
      if (out_alpha)
        *out_alpha = &alpha_cpu;
      if (out_alpha_w)
        *out_alpha_w = matte_w;
      if (out_alpha_h)
        *out_alpha_h = matte_h;
      return true;
    }

    bool
    ApplyGpuStage(const studiocast::vulkan::VulkanImage &in_rgb,
                  studiocast::vulkan::VulkanImage *out_rgb_img, int width,
                  int height, std::uint64_t capture_sequence,
                  const studiocast::video::effects::BroadcastCameraEffects &fx,
                  std::string *error_out) {
      if (error_out)
        error_out->clear();

      using studiocast::video::effects::VirtualBackgroundMode;
      if (fx.virtual_background.mode == VirtualBackgroundMode::none)
        return true;
      if (!out_rgb_img) {
        if (error_out)
          *error_out = "Open Vulkan: out_rgb is null.";
        return false;
      }
      if (!in_rgb.Valid()) {
        if (error_out)
          *error_out = "Open Vulkan: invalid input GPU RGB image.";
        return false;
      }
      if (in_rgb.width() != width || in_rgb.height() != height ||
          in_rgb.format() != studiocast::vulkan::VulkanPixelFormat::rgb_u8) {
        if (error_out)
          *error_out = "Open Vulkan: invalid input GPU RGB image.";
        return false;
      }
      if (!EnsureFrameAlphaForFrameGpu(in_rgb, capture_sequence, width, height,
                                       fx, error_out)) {
        return false;
      }
      if (!EnsureImage(out_rgb_img, width, height,
                       studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                       /*map_memory=*/false, "resident out_rgb", error_out)) {
        return false;
      }

      if (fx.virtual_background.mode == VirtualBackgroundMode::blur) {
        const auto parameters = ResolveOpenVulkanVirtualBackgroundBlurParameters(
            fx.virtual_background.strength);
        OpenVulkanVirtualBackgroundBlurInput blur_input;
        blur_input.foreground = &in_rgb;
        blur_input.alpha = &alpha_resized;
        blur_input.alpha_tmp = &alpha_tmp;
        blur_input.alpha_feathered = &alpha_feather;
        blur_input.blur_tmp = &blur_tmp;
        blur_input.blurred = &blurred;
        blur_input.output = out_rgb_img;
        blur_input.strength = fx.virtual_background.strength;
        blur_input.capture_sequence = capture_sequence;
        blur_input.resident_alpha_sequence = cached_frame_alpha_sequence;
        blur_input.alpha_resize_completion_count =
            alpha_resize_completion_count;
        const auto matting_readiness = matting_session->Readiness();
        blur_input.matting_readiness = &matting_readiness;
        if (!blur_effect.Apply(blur_input, &blur_effect_counters, error_out)) {
          return false;
        }
        blur_dispatch_calls +=
            1u + (parameters.alpha_feather_radius > 0 ? 1u : 0u);
        ++composite_dispatch_calls;
        forced_sync_calls +=
            1u + (parameters.alpha_feather_radius > 0 ? 1u : 0u);
      } else if (fx.virtual_background.mode == VirtualBackgroundMode::remove) {
        OpenVulkanVirtualBackgroundRemoveInput remove_input;
        remove_input.foreground = &in_rgb;
        remove_input.alpha = &alpha_resized;
        remove_input.alpha_tmp = &alpha_tmp;
        remove_input.alpha_feathered = &alpha_feather;
        remove_input.output = out_rgb_img;
        remove_input.capture_sequence = capture_sequence;
        remove_input.resident_alpha_sequence = cached_frame_alpha_sequence;
        remove_input.alpha_resize_completion_count =
            alpha_resize_completion_count;
        const auto matting_readiness = matting_session->Readiness();
        remove_input.matting_readiness = &matting_readiness;
        if (!remove_effect.Apply(remove_input, &remove_effect_counters,
                                 error_out)) {
          return false;
        }
        const auto &parameters = remove_effect.parameters();
        blur_dispatch_calls +=
            parameters.alpha_feather_radius > 0 ? 1u : 0u;
        ++composite_dispatch_calls;
        forced_sync_calls +=
            1u + (parameters.alpha_feather_radius > 0 ? 1u : 0u);
      } else if (fx.virtual_background.mode == VirtualBackgroundMode::replace) {
        OpenVulkanVirtualBackgroundReplaceInput replace_input;
        replace_input.foreground = &in_rgb;
        replace_input.alpha = &alpha_resized;
        replace_input.alpha_tmp = &alpha_tmp;
        replace_input.alpha_feathered = &alpha_feather;
        replace_input.output = out_rgb_img;
        replace_input.capture_sequence = capture_sequence;
        replace_input.resident_alpha_sequence = cached_frame_alpha_sequence;
        replace_input.alpha_resize_completion_count =
            alpha_resize_completion_count;
        const auto matting_readiness = matting_session->Readiness();
        replace_input.matting_readiness = &matting_readiness;
        if (!replace_effect.Apply(replace_input, &replace_effect_counters,
                                  error_out)) {
          return false;
        }
        const auto &parameters = replace_effect.parameters();
        blur_dispatch_calls +=
            parameters.alpha_feather_radius > 0 ? 1u : 0u;
        ++composite_dispatch_calls;
        forced_sync_calls +=
            1u + (parameters.alpha_feather_radius > 0 ? 1u : 0u);
      }
      return true;
    }

    bool
    ApplyVulkanRgb(const studiocast::vulkan::VulkanImage &in_rgb,
                   studiocast::vulkan::VulkanImage *out_rgb_img,
                   const studiocast::video::effects::BroadcastCameraEffects &fx,
                   std::uint64_t capture_sequence, std::string *error_out,
                   DeferredGpuOut *deferred_out) {
      if (error_out)
        error_out->clear();
      using studiocast::video::effects::VirtualBackgroundMode;
      if (fx.virtual_background.mode == VirtualBackgroundMode::none) {
        if (deferred_out)
          *deferred_out = DeferredGpuOut{};
        return true;
      }
      if (!deferred_out) {
        if (error_out)
          *error_out = "Open Vulkan: ApplyVulkanRgb requires deferred_out.";
        return false;
      }
      std::string stage_err;
      if (!ApplyGpuStage(in_rgb, out_rgb_img, in_rgb.width(), in_rgb.height(),
                         capture_sequence, fx, &stage_err)) {
        if (error_out)
          *error_out = stage_err;
        return false;
      }
      deferred_out->kind = DeferredGpuKind::vulkan_rgb;
      deferred_out->nvcv_img = nullptr;
      deferred_out->nvcv = nullptr;
      deferred_out->cuda_img = nullptr;
      deferred_out->cuda = nullptr;
      deferred_out->stream = nullptr;
      deferred_out->vulkan_img = out_rgb_img;
      deferred_out->vulkan_kernels = &kernels;
      return true;
    }
  } open_vulkan_vb;

  OpenVulkanVignette open_vulkan_vignette;
  OpenVulkanVignetteCounters open_vulkan_vignette_counters;
  OpenVulkanMirror open_vulkan_mirror;
#endif

  struct OpenCudaAutoFrameContext {
    bool initialized = false;
    bool enabled = false;

    std::string last_error;
    std::string active_model_id;

    // Shared foreground matte provider (Open CUDA matting). We reuse the same
    // matting model pack and inference output as virtual background/key light
    // when possible.
    OpenCudaVirtualBackgroundContext *matte = nullptr;
    int last_frame_w = 0;
    int last_frame_h = 0;

    // Matte fallback throttling.
    // When Auto Frame falls back to foreground matting (no face detections),
    // avoid running the matting model every frame unless another stage (e.g.
    // virtual background) already computed the matte for this capture sequence.
    std::uint64_t last_matte_query_sequence = 0;
    bool have_last_matte_box = false;
    studiocast::maxine::effects::RectF last_matte_box_px{};

    std::vector<std::uint8_t> tmp_rgb;

    bool have_smoothed_crop = false;
    studiocast::maxine::effects::RectF crop_smoothed_px;
    bool last_had_detection = false;
    enum class TrackingSource {
      kNone = 0,
      kFace = 1,
      kBody = 2,
      kLandmarks = 3,
      kMatte = 4,
    };
    TrackingSource last_tracking_source = TrackingSource::kNone;

    struct TrackingBox {
      TrackingSource source = TrackingSource::kNone;
      studiocast::maxine::effects::RectF box_px{};
      float confidence = 0.0f;
    };

    struct CropPlan {
      bool should_crop = false;
      bool found_tracking_box = false;
      TrackingSource tracking_source = TrackingSource::kNone;
      studiocast::maxine::effects::RectF crop_px{};
    };

    ~OpenCudaAutoFrameContext() { Destroy(); }

    void Destroy() {
      initialized = false;
      enabled = false;
      last_error.clear();
      active_model_id.clear();
      have_smoothed_crop = false;
      last_had_detection = false;
      last_tracking_source = TrackingSource::kNone;
      last_frame_w = 0;
      last_frame_h = 0;
      last_matte_query_sequence = 0;
      have_last_matte_box = false;
      last_matte_box_px = {};
    }

    static bool
    ComputeMatteBoxPxFromAlpha(const std::vector<float> &alpha, int alpha_w,
                               int alpha_h, int frame_w, int frame_h,
                               studiocast::maxine::effects::RectF *out_box_px) {
      if (!out_box_px) {
        return false;
      }
      if (alpha.empty() || alpha_w <= 0 || alpha_h <= 0) {
        return false;
      }

      // Threshold the matte and compute a tight bounding box around the
      // foreground.
      constexpr float kAlphaThreshold = 0.35f;
      int min_x = alpha_w;
      int min_y = alpha_h;
      int max_x = -1;
      int max_y = -1;
      int count = 0;
      for (int y = 0; y < alpha_h; ++y) {
        const float *row = alpha.data() + (static_cast<size_t>(y) *
                                           static_cast<size_t>(alpha_w));
        for (int x = 0; x < alpha_w; ++x) {
          if (row[x] > kAlphaThreshold) {
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
            ++count;
          }
        }
      }

      // Reject tiny detections to avoid jitter when the matte is effectively
      // empty.
      const int min_pixels = std::max(32, (alpha_w * alpha_h) / 500); // ~0.2%
      if (count < min_pixels || max_x < min_x || max_y < min_y) {
        return false;
      }

      const float sx =
          static_cast<float>(frame_w) / static_cast<float>(alpha_w);
      const float sy =
          static_cast<float>(frame_h) / static_cast<float>(alpha_h);

      const float x0 = static_cast<float>(min_x) * sx;
      const float y0 = static_cast<float>(min_y) * sy;
      const float x1 = static_cast<float>(max_x + 1) * sx;
      const float y1 = static_cast<float>(max_y + 1) * sy;

      out_box_px->x = x0;
      out_box_px->y = y0;
      out_box_px->w = x1 - x0;
      out_box_px->h = y1 - y0;
      return true;
    }

    static bool IsUsableTrackingBox(const TrackingBox &box, int frame_w,
                                    int frame_h) {
      if (box.source == TrackingSource::kNone)
        return false;
      if (frame_w <= 0 || frame_h <= 0)
        return false;
      const auto &r = box.box_px;
      return std::isfinite(r.x) && std::isfinite(r.y) && std::isfinite(r.w) &&
             std::isfinite(r.h) && r.w > 0.5f && r.h > 0.5f &&
             r.x < static_cast<float>(frame_w) &&
             r.y < static_cast<float>(frame_h) && (r.x + r.w) > 0.0f &&
             (r.y + r.h) > 0.0f;
    }

    static bool BuildBestFaceTrackingBox(
        const std::vector<studiocast::open_video::FaceDetection> &detections,
        TrackingBox *out_box) {
      if (out_box)
        *out_box = TrackingBox{};
      if (!out_box || detections.empty())
        return false;

      // Pick the largest detected face (most stable for framing), then expand
      // to a subject box. Future detectors can skip this adapter and publish a
      // TrackingBox directly.
      const studiocast::open_video::FaceDetection *best = &detections.front();
      float best_area = best->w * best->h;
      for (const auto &f : detections) {
        const float a = f.w * f.h;
        if (a > best_area) {
          best = &f;
          best_area = a;
        }
      }

      const float fx0 = best->x;
      const float fy0 = best->y;
      const float fw = best->w;
      const float fh = best->h;
      const float cx = fx0 + fw * 0.5f;

      const float expand_w = fw * 1.8f;
      const float expand_h = fh * 2.6f;
      const float new_x = cx - expand_w * 0.5f;
      const float new_y = fy0 - fh * 0.4f;

      out_box->source = TrackingSource::kFace;
      out_box->box_px =
          studiocast::maxine::effects::RectF{new_x, new_y, expand_w, expand_h};
      out_box->confidence = best->score;
      return true;
    }

    static void CropResizeRgb24Bilinear(const std::uint8_t *src, int src_w,
                                        int src_h, size_t src_stride,
                                        std::uint8_t *dst, int dst_w, int dst_h,
                                        size_t dst_stride, float crop_x,
                                        float crop_y, float crop_w,
                                        float crop_h) {
      if (!src || !dst) {
        return;
      }
      if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return;
      }
      crop_w = std::max(1.0f, crop_w);
      crop_h = std::max(1.0f, crop_h);

      const float scale_x = crop_w / static_cast<float>(dst_w);
      const float scale_y = crop_h / static_cast<float>(dst_h);

      for (int y = 0; y < dst_h; ++y) {
        const float src_y =
            crop_y + (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const float sy = std::clamp(src_y, 0.0f, static_cast<float>(src_h - 1));
        const int y0 = static_cast<int>(sy);
        const int y1 = std::min(y0 + 1, src_h - 1);
        const float ty = sy - static_cast<float>(y0);

        const std::uint8_t *row0 = src + static_cast<size_t>(y0) * src_stride;
        const std::uint8_t *row1 = src + static_cast<size_t>(y1) * src_stride;
        std::uint8_t *out = dst + static_cast<size_t>(y) * dst_stride;

        for (int x = 0; x < dst_w; ++x) {
          const float src_x =
              crop_x + (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
          const float sx =
              std::clamp(src_x, 0.0f, static_cast<float>(src_w - 1));
          const int x0 = static_cast<int>(sx);
          const int x1 = std::min(x0 + 1, src_w - 1);
          const float tx = sx - static_cast<float>(x0);

          const std::uint8_t *p00 = row0 + x0 * 3;
          const std::uint8_t *p01 = row0 + x1 * 3;
          const std::uint8_t *p10 = row1 + x0 * 3;
          const std::uint8_t *p11 = row1 + x1 * 3;

          for (int c = 0; c < 3; ++c) {
            const float v00 = static_cast<float>(p00[c]);
            const float v01 = static_cast<float>(p01[c]);
            const float v10 = static_cast<float>(p10[c]);
            const float v11 = static_cast<float>(p11[c]);

            const float v0 = v00 + (v01 - v00) * tx;
            const float v1 = v10 + (v11 - v10) * tx;
            const float v = v0 + (v1 - v0) * ty;

            out[x * 3 + c] =
                static_cast<std::uint8_t>(std::clamp(v, 0.0f, 255.0f));
          }
        }
      }
    }

    bool EnsureInitialized(
        int frame_w, int frame_h,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        bool require_matte_tracking, std::string *error) {
      if (error)
        error->clear();

      if (frame_w <= 0 || frame_h <= 0) {
        last_error = "Open CUDA Auto Frame: invalid frame size.";
        if (error)
          *error = last_error;
        return false;
      }
      std::string current_model_id;
      if (require_matte_tracking) {
        if (!matte) {
          last_error = "Open CUDA Auto Frame: matte provider not set.";
          if (error)
            *error = last_error;
          return false;
        }

        // Ensure the shared matting model/session exists, but don't allocate
        // the heavy virtual-background buffers if the VB stage isn't scheduled.
        std::string matte_init_err;
        if (!matte->EnsureInitialized(frame_w, frame_h, fx, &matte_init_err,
                                      /*require_vb_buffers=*/false)) {
          last_error = "Open CUDA Auto Frame: " + matte_init_err;
          if (error)
            *error = last_error;
          return false;
        }
        current_model_id = matte->active_model_id;
      }

      // Track model/frame changes to reset smoothing state.
      if (initialized && enabled && current_model_id == active_model_id &&
          last_frame_w == frame_w && last_frame_h == frame_h) {
        return true;
      }

      have_smoothed_crop = false;
      last_had_detection = false;
      last_tracking_source = TrackingSource::kNone;
      last_matte_query_sequence = 0;
      have_last_matte_box = false;
      last_matte_box_px = {};
      active_model_id = current_model_id;
      last_frame_w = frame_w;
      last_frame_h = frame_h;

      initialized = true;
      enabled = true;
      return true;
    }

    bool ResolveMatteTrackingBox(
        std::uint64_t capture_sequence, const std::uint8_t *rgb, int width,
        int height, size_t stride,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        TrackingBox *out_box, std::string *error) {
      if (out_box)
        *out_box = TrackingBox{};
      if (!out_box || !matte)
        return true;

      // If another stage already computed matting for this frame (e.g.
      // Virtual Background or Virtual Key Light), reuse the cached result
      // immediately.
      const bool matte_cached_this_frame =
          matte->cached_matte_valid &&
          matte->cached_matte_sequence == capture_sequence;

      // Otherwise, rate-limit matting inference when it is only used for
      // tracking. Matting can be significantly more expensive than box
      // detectors.
      constexpr std::uint64_t kMatteFallbackIntervalFrames = 10;
      const bool should_query_matte =
          matte_cached_this_frame || (last_matte_query_sequence == 0) ||
          (capture_sequence >=
           (last_matte_query_sequence + kMatteFallbackIntervalFrames));

      if (!should_query_matte) {
        if (have_last_matte_box) {
          out_box->source = TrackingSource::kMatte;
          out_box->box_px = last_matte_box_px;
          out_box->confidence = 1.0f;
        }
        return true;
      }

      // Ensure the shared matting session exists, but avoid allocating VB-only
      // buffers when matting is only used for tracking.
      std::string matte_init_err;
      if (!matte->EnsureInitialized(width, height, fx, &matte_init_err,
                                    /*require_vb_buffers=*/false)) {
        if (error && !matte_init_err.empty()) {
          *error = "Open CUDA Auto Frame: " + matte_init_err;
        }
        return true;
      }

      const std::vector<float> *alpha_cpu = nullptr;
      int alpha_w = 0;
      int alpha_h = 0;
      std::string alpha_err;
      studiocast::maxine::effects::RectF box_px;
      if (matte->GetAlphaCpuForFrame(rgb, stride, capture_sequence, width,
                                     height, fx, &alpha_cpu, &alpha_w, &alpha_h,
                                     &alpha_err) &&
          alpha_cpu) {
        const bool found = ComputeMatteBoxPxFromAlpha(
            *alpha_cpu, alpha_w, alpha_h, width, height, &box_px);

        last_matte_query_sequence = capture_sequence;
        if (found) {
          have_last_matte_box = true;
          last_matte_box_px = box_px;
          out_box->source = TrackingSource::kMatte;
          out_box->box_px = box_px;
          out_box->confidence = 1.0f;
        } else {
          have_last_matte_box = false;
          last_matte_box_px = {};
        }
      } else {
        if (error && !alpha_err.empty()) {
          *error = "Open CUDA Auto Frame: " + alpha_err;
        }
        return true;
      }

      return true;
    }

    bool ResolveCropPlan(
        std::uint64_t capture_sequence, const std::uint8_t *rgb, int width,
        int height, size_t stride,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        const TrackingBox *tracking_box, bool tracking_provider_ran,
        CropPlan *plan, std::string *error) {
      if (plan) {
        *plan = CropPlan{};
      }
      if (!fx.auto_frame.enabled) {
        return true;
      }
      if (!plan) {
        if (error) {
          *error = "Open CUDA Auto Frame: null crop plan.";
        }
        return true;
      }

      std::string err;
      if (!EnsureInitialized(width, height, fx,
                             /*require_matte_tracking=*/!tracking_provider_ran,
                             &err)) {
        // Best-effort: if we can't initialize tracking, just bypass auto frame.
        if (error && !err.empty()) {
          *error = err;
        }
        return true;
      }

      TrackingBox resolved_box;
      bool found = false;

      if (tracking_box && IsUsableTrackingBox(*tracking_box, width, height)) {
        resolved_box = *tracking_box;
        found = true;
      } else if (!tracking_provider_ran) {
        if (!ResolveMatteTrackingBox(capture_sequence, rgb, width, height,
                                     stride, fx, &resolved_box, error)) {
          return false;
        }
        found = IsUsableTrackingBox(resolved_box, width, height);
      }

      const TrackingSource source =
          found ? resolved_box.source : TrackingSource::kNone;
      if (source != last_tracking_source) {
        have_smoothed_crop = false;
        last_tracking_source = source;
      }

      // Use the same crop math as the Maxine auto-frame tracker for
      // consistency.
      studiocast::maxine::effects::AutoFrameKnobs knobs;
      knobs.strength = fx.auto_frame.strength;
      knobs.smoothing = fx.auto_frame.smoothing;
      knobs.headroom = fx.auto_frame.headroom;

      const float out_aspect =
          static_cast<float>(width) / static_cast<float>(height);
      studiocast::maxine::effects::RectF target_crop = {};
      if (found) {
        target_crop = studiocast::maxine::effects::ArAutoFrameTracker::
            ComputeTargetCropFromBoxPx(resolved_box.box_px, width, height,
                                       out_aspect, knobs);
      } else {
        const float strength01 =
            std::clamp(static_cast<float>(knobs.strength) / 100.0f, 0.0f, 1.0f);
        const float zoom = 1.0f + strength01 * 0.5f; // up to ~1.5x
        target_crop =
            studiocast::maxine::effects::ArAutoFrameTracker::CenterCrop(
                width, height, out_aspect, zoom);
      }

      auto clamp_crop = [&](studiocast::maxine::effects::RectF *r) {
        if (!r) {
          return;
        }
        r->w = std::clamp(r->w, 1.0f, static_cast<float>(width));
        r->h = std::clamp(r->h, 1.0f, static_cast<float>(height));
        r->x = std::clamp(r->x, 0.0f, static_cast<float>(width) - r->w);
        r->y = std::clamp(r->y, 0.0f, static_cast<float>(height) - r->h);
      };

      if (!have_smoothed_crop) {
        crop_smoothed_px = target_crop;
        have_smoothed_crop = true;
      } else {
        const float a =
            studiocast::maxine::effects::ArAutoFrameTracker::SmoothingAlpha(
                knobs.smoothing);
        crop_smoothed_px =
            studiocast::maxine::effects::ArAutoFrameTracker::Lerp(
                crop_smoothed_px, target_crop, a);
      }
      clamp_crop(&crop_smoothed_px);
      last_had_detection = found;

      // Early-out if crop is effectively full-frame.
      constexpr float kEpsPx = 0.5f;
      if (std::abs(crop_smoothed_px.x) <= kEpsPx &&
          std::abs(crop_smoothed_px.y) <= kEpsPx &&
          std::abs(crop_smoothed_px.w - static_cast<float>(width)) <= kEpsPx &&
          std::abs(crop_smoothed_px.h - static_cast<float>(height)) <= kEpsPx) {
        return true;
      }

      plan->should_crop = true;
      plan->found_tracking_box = found;
      plan->tracking_source = source;
      plan->crop_px = crop_smoothed_px;
      return true;
    }

    bool
    ApplyCudaRgb(std::uint64_t capture_sequence,
                 const studiocast::cuda::CudaImage &in_rgb,
                 studiocast::cuda::CudaImage *out_rgb,
                 const std::uint8_t *tracking_rgb, int width, int height,
                 size_t tracking_stride,
                 const studiocast::video::effects::BroadcastCameraEffects &fx,
                 const TrackingBox *tracking_box, bool tracking_provider_ran,
                 DeferredGpuOut *deferred_out, std::string *error) {
      if (!fx.auto_frame.enabled) {
        return true;
      }
      if (!out_rgb || !deferred_out) {
        if (error) {
          *error = "Open CUDA Auto Frame: null GPU output.";
        }
        return false;
      }
      if (!in_rgb.Valid() || in_rgb.w != width || in_rgb.h != height ||
          in_rgb.format != studiocast::cuda::PixelFormatGpu::rgb_u8) {
        if (error) {
          *error = "Open CUDA Auto Frame: invalid input GPU RGB image.";
        }
        return false;
      }

      CropPlan plan;
      std::string plan_err;
      if (!ResolveCropPlan(capture_sequence, tracking_rgb, width, height,
                           tracking_stride, fx, tracking_box,
                           tracking_provider_ran, &plan, &plan_err)) {
        if (error) {
          *error = plan_err;
        }
        return false;
      }

      deferred_out->kind = DeferredGpuKind::cuda_rgb;
      deferred_out->nvcv_img = nullptr;
      deferred_out->nvcv = nullptr;
      deferred_out->cuda = matte ? &matte->cuda : nullptr;
      deferred_out->stream = matte ? matte->vb_stream : nullptr;

      if (!plan.should_crop) {
        deferred_out->cuda_img = &in_rgb;
        return true;
      }

      if (!matte) {
        if (error) {
          *error = "Open CUDA Auto Frame: matte provider not set.";
        }
        return false;
      }

      std::string berr;
      if (!out_rgb->ReallocIfNeeded(&matte->cuda, width, height,
                                    studiocast::cuda::PixelFormatGpu::rgb_u8,
                                    &berr)) {
        if (error) {
          *error =
              "Open CUDA Auto Frame: failed to allocate GPU output: " + berr;
        }
        return false;
      }

      std::string kerr;
      if (!studiocast::cuda::kernels::CropResizeBilinear(
              in_rgb, *out_rgb, plan.crop_px.x, plan.crop_px.y, plan.crop_px.w,
              plan.crop_px.h, matte->vb_stream, &kerr)) {
        if (error) {
          *error = "Open CUDA Auto Frame: GPU crop/resize failed: " + kerr;
        }
        return false;
      }

      deferred_out->cuda_img = out_rgb;
      return true;
    }

    bool ApplyRgbInPlace(
        std::uint64_t capture_sequence, std::uint8_t *rgb_data, int width,
        int height, size_t stride,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        const TrackingBox *tracking_box, bool tracking_provider_ran,
        std::string *error, bool *out_cpu_crop_resize_applied = nullptr) {
      if (out_cpu_crop_resize_applied)
        *out_cpu_crop_resize_applied = false;
      if (!fx.auto_frame.enabled) {
        return true;
      }
      if (!rgb_data) {
        if (error) {
          *error = "Open CUDA Auto Frame: null RGB buffer.";
        }
        return false;
      }

      CropPlan plan;
      if (!ResolveCropPlan(capture_sequence, rgb_data, width, height, stride,
                           fx, tracking_box, tracking_provider_ran, &plan,
                           error)) {
        return false;
      }
      if (!plan.should_crop) {
        return true;
      }
      if (out_cpu_crop_resize_applied)
        *out_cpu_crop_resize_applied = true;

      const size_t min_row_bytes = static_cast<size_t>(width) * 3;
      if (stride < min_row_bytes) {
        if (error) {
          *error = "Open CUDA Auto Frame: unexpected RGB stride.";
        }
        return false;
      }

      tmp_rgb.resize(static_cast<size_t>(height) * stride);
      CropResizeRgb24Bilinear(rgb_data, width, height, stride, tmp_rgb.data(),
                              width, height, stride, plan.crop_px.x,
                              plan.crop_px.y, plan.crop_px.w, plan.crop_px.h);

      for (int y = 0; y < height; ++y) {
        std::memcpy(rgb_data + (static_cast<size_t>(y) * stride),
                    tmp_rgb.data() + (static_cast<size_t>(y) * stride),
                    min_row_bytes);
      }

      return true;
    }
  } open_cuda_auto_frame;

  // Open-source (Open CUDA) Virtual Key Light.
  //
  // Implementation approach:
  //  - Use the same Open CUDA matting model packs (foreground matte) used by
  //    virtual background.
  //  - Prefer a GPU-in/GPU-out blend when the frame and matte are already
  //    resident in the Open CUDA section.
  //  - Keep the CPU implementation as fallback and for standalone CPU-tail
  //    continuation cases.
  struct OpenCudaKeyLightContext {
    bool initialized = false;
    bool enabled = false;
    std::string last_error;
    std::string active_model_id;
    OpenCudaVirtualBackgroundContext *matte =
        nullptr; // shared Open CUDA matting + cache
    int last_frame_w = 0;
    int last_frame_h = 0;

    void Destroy() {
      initialized = false;
      enabled = false;
      last_error.clear();
      active_model_id.clear();
      last_frame_w = 0;
      last_frame_h = 0;
    }

    static void ResolveTargetColorFromTemperaturePreset(int preset,
                                                        float *out_r,
                                                        float *out_g,
                                                        float *out_b) {
      // Simple perceptual targets (sRGB-ish). These are not physically
      // accurate; they are tuned to "feel" like Broadcast's warm/cool key light
      // in a subtle way. preset: 0=neutral, 1=warm, 2=cool
      float r = 255.0f;
      float g = 255.0f;
      float b = 255.0f;
      if (preset == 1) {
        // Warm.
        r = 255.0f;
        g = 242.0f;
        b = 228.0f;
      } else if (preset == 2) {
        // Cool.
        r = 228.0f;
        g = 242.0f;
        b = 255.0f;
      }
      if (out_r)
        *out_r = r;
      if (out_g)
        *out_g = g;
      if (out_b)
        *out_b = b;
    }

    bool EnsureInitialized(
        int frame_w, int frame_h,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error) {
      if (error)
        error->clear();

      if (frame_w <= 0 || frame_h <= 0) {
        last_error = "Open CUDA Virtual Key Light: invalid frame size.";
        if (error)
          *error = last_error;
        return false;
      }
      if (!matte) {
        last_error = "Open CUDA Virtual Key Light: matte provider not set.";
        if (error)
          *error = last_error;
        return false;
      }

      // Ensure the shared matting model/session exists, but don't allocate the
      // heavy virtual-background buffers if the VB stage isn't scheduled.
      std::string matte_init_err;
      if (!matte->EnsureInitialized(frame_w, frame_h, fx, &matte_init_err,
                                    /*require_vb_buffers=*/false)) {
        last_error = "Open CUDA Virtual Key Light: " + matte_init_err;
        if (error)
          *error = last_error;
        return false;
      }

      const std::string current_model_id = matte->active_model_id;
      if (initialized && enabled && current_model_id == active_model_id &&
          last_frame_w == frame_w && last_frame_h == frame_h) {
        return true;
      }

      active_model_id = current_model_id;
      last_frame_w = frame_w;
      last_frame_h = frame_h;
      initialized = true;
      enabled = true;
      return true;
    }

    bool ApplyRgbInPlace(
        std::uint64_t capture_sequence, std::uint8_t *rgb, int width,
        int height, size_t stride,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error) {
      if (!fx.virtual_key_light.enabled) {
        return true;
      }
      if (!rgb) {
        if (error) {
          *error = "Open CUDA Virtual Key Light: null RGB buffer.";
        }
        return false;
      }

      const float intensity01 =
          Clamp01FromPercent(fx.virtual_key_light.intensity);
      if (intensity01 <= 0.0001f) {
        return true;
      }

      std::string err;
      if (!EnsureInitialized(width, height, fx, &err)) {
        if (error)
          *error = err;
        return false;
      }
      if (!matte) {
        if (error)
          *error = "Open CUDA Virtual Key Light: matte provider not set.";
        return false;
      }

      const std::vector<float> *alpha_cpu = nullptr;
      int alpha_w = 0;
      int alpha_h = 0;
      std::string alpha_err;
      if (!matte->GetAlphaCpuForFrame(rgb, stride, capture_sequence, width,
                                      height, fx, &alpha_cpu, &alpha_w,
                                      &alpha_h, &alpha_err)) {
        if (error) {
          *error = "Open CUDA Virtual Key Light: " + alpha_err;
        }
        return false;
      }
      if (!alpha_cpu) {
        if (error) {
          *error = "Open CUDA Virtual Key Light: alpha CPU buffer is null.";
        }
        return false;
      }

      // Compute a matte bounding box so we only process pixels near the
      // subject.
      studiocast::maxine::effects::RectF box_px;
      const bool found = OpenCudaAutoFrameContext::ComputeMatteBoxPxFromAlpha(
          *alpha_cpu, alpha_w, alpha_h, width, height, &box_px);
      if (!found) {
        return true; // no subject detected
      }

      // Expand ROI slightly.
      const float margin_x = std::max(16.0f, box_px.w * 0.15f);
      const float margin_y = std::max(16.0f, box_px.h * 0.15f);
      float x0f =
          std::clamp(box_px.x - margin_x, 0.0f, static_cast<float>(width));
      float y0f =
          std::clamp(box_px.y - margin_y, 0.0f, static_cast<float>(height));
      float x1f = std::clamp(box_px.x + box_px.w + margin_x, 0.0f,
                             static_cast<float>(width));
      float y1f = std::clamp(box_px.y + box_px.h + margin_y, 0.0f,
                             static_cast<float>(height));
      const int x0 = static_cast<int>(std::floor(x0f));
      const int y0 = static_cast<int>(std::floor(y0f));
      const int x1 = static_cast<int>(std::ceil(x1f));
      const int y1 = static_cast<int>(std::ceil(y1f));

      // Directional weighting (left/right).
      const float pan = std::clamp(
          static_cast<float>(fx.virtual_key_light.direction_pan_degrees),
          -90.0f, 90.0f);
      const float dir = std::clamp(pan / 90.0f, -1.0f, 1.0f);
      const float cx = static_cast<float>(width) * 0.5f;
      const float inv_cx = (cx > 1.0f) ? (1.0f / cx) : 0.0f;

      float target_r = 255.0f, target_g = 255.0f, target_b = 255.0f;
      ResolveTargetColorFromTemperaturePreset(
          fx.virtual_key_light.temperature_preset, &target_r, &target_g,
          &target_b);

      // Nearest-neighbor matte sampling using fixed-point mapping.
      if (alpha_w <= 0 || alpha_h <= 0 || alpha_cpu->empty()) {
        return true;
      }

      const int step_x_fp = (alpha_w << 16) / std::max(1, width);
      const int step_y_fp = (alpha_h << 16) / std::max(1, height);
      int ay_fp = y0 * step_y_fp;
      for (int y = y0; y < y1; ++y) {
        const int ay = std::clamp(ay_fp >> 16, 0, alpha_h - 1);
        const float *alpha_row =
            alpha_cpu->data() +
            static_cast<size_t>(ay) * static_cast<size_t>(alpha_w);
        std::uint8_t *row = rgb + static_cast<size_t>(y) * stride;
        int ax_fp = x0 * step_x_fp;
        for (int x = x0; x < x1; ++x) {
          const int ax = std::clamp(ax_fp >> 16, 0, alpha_w - 1);
          const float a = std::clamp(alpha_row[ax], 0.0f, 1.0f);
          if (a > 0.02f) {
            // Compute per-pixel blend factor.
            const float x_norm =
                (static_cast<float>(x) - cx) * inv_cx; // [-1..1]
            float field = 1.0f + dir * x_norm * 0.35f;
            field = std::clamp(field, 0.65f, 1.35f);

            float t = intensity01 * a * field;
            t = std::clamp(t, 0.0f, 1.0f);

            std::uint8_t *px = row + x * 3;
            const float in_r = static_cast<float>(px[0]);
            const float in_g = static_cast<float>(px[1]);
            const float in_b = static_cast<float>(px[2]);

            const float out_r = in_r + (target_r - in_r) * t;
            const float out_g = in_g + (target_g - in_g) * t;
            const float out_b = in_b + (target_b - in_b) * t;

            px[0] = static_cast<std::uint8_t>(std::clamp(out_r, 0.0f, 255.0f));
            px[1] = static_cast<std::uint8_t>(std::clamp(out_g, 0.0f, 255.0f));
            px[2] = static_cast<std::uint8_t>(std::clamp(out_b, 0.0f, 255.0f));
          }
          ax_fp += step_x_fp;
        }
        ay_fp += step_y_fp;
      }

      return true;
    }

    bool
    ApplyCudaRgb(const studiocast::cuda::CudaImage &in_rgb,
                 studiocast::cuda::CudaImage *out_rgb_img,
                 const studiocast::video::effects::BroadcastCameraEffects &fx,
                 std::uint64_t capture_sequence, std::string *error,
                 DeferredGpuOut *deferred_out) {
      if (error)
        error->clear();
      if (!fx.virtual_key_light.enabled) {
        return true;
      }

      const float intensity01 =
          Clamp01FromPercent(fx.virtual_key_light.intensity);
      if (intensity01 <= 0.0001f) {
        return true;
      }
      if (!matte) {
        if (error)
          *error = "Open CUDA Virtual Key Light: matte provider not set.";
        return false;
      }
      if (!out_rgb_img || !deferred_out) {
        if (error)
          *error = "Open CUDA Virtual Key Light: invalid GPU output.";
        return false;
      }
      if (!in_rgb.Valid() ||
          in_rgb.format != studiocast::cuda::PixelFormatGpu::rgb_u8 ||
          in_rgb.w <= 0 || in_rgb.h <= 0) {
        if (error)
          *error = "Open CUDA Virtual Key Light: invalid input GPU RGB image.";
        return false;
      }

      std::string init_err;
      if (!EnsureInitialized(in_rgb.w, in_rgb.h, fx, &init_err)) {
        if (error)
          *error = init_err;
        return false;
      }

      std::string matte_err;
      if (!matte->EnsureMatteForFrameGpu(in_rgb, capture_sequence, in_rgb.w,
                                         in_rgb.h, fx, &matte_err)) {
        if (error)
          *error = "Open CUDA Virtual Key Light: " + matte_err;
        return false;
      }

      std::string berr;
      if (!out_rgb_img->ReallocIfNeeded(
              &matte->cuda, in_rgb.w, in_rgb.h,
              studiocast::cuda::PixelFormatGpu::rgb_u8, &berr)) {
        if (error)
          *error = "Open CUDA Virtual Key Light: failed to allocate output RGB "
                   "image: " +
                   berr;
        return false;
      }

      std::string kerr;
      if (!studiocast::cuda::kernels::ResizeBilinearF32_1(
              matte->alpha_model_view, matte->alpha_resized, matte->vb_stream,
              &kerr)) {
        if (error)
          *error = "Open CUDA Virtual Key Light: alpha resize failed: " + kerr;
        return false;
      }

      const float pan = std::clamp(
          static_cast<float>(fx.virtual_key_light.direction_pan_degrees),
          -90.0f, 90.0f);
      const float dir = std::clamp(pan / 90.0f, -1.0f, 1.0f);

      float target_r = 255.0f, target_g = 255.0f, target_b = 255.0f;
      ResolveTargetColorFromTemperaturePreset(
          fx.virtual_key_light.temperature_preset, &target_r, &target_g,
          &target_b);

      if (!studiocast::cuda::kernels::ApplyKeyLightU8x3(
              in_rgb, matte->alpha_resized, target_r, target_g, target_b,
              intensity01, dir, *out_rgb_img, matte->vb_stream, &kerr)) {
        if (error)
          *error = "Open CUDA Virtual Key Light: GPU blend failed: " + kerr;
        return false;
      }

      deferred_out->kind = DeferredGpuKind::cuda_rgb;
      deferred_out->nvcv_img = nullptr;
      deferred_out->nvcv = nullptr;
      deferred_out->cuda_img = out_rgb_img;
      deferred_out->cuda = &matte->cuda;
      deferred_out->stream = matte->vb_stream;
      return true;
    }
  } open_cuda_key_light;

#if STUDIOCAST_ENABLE_OPEN_VULKAN
  struct OpenVulkanKeyLightContext {
    std::string last_error;
    OpenVulkanVirtualBackgroundContext *matte = nullptr;

    void Destroy() {
      last_error.clear();
      if (matte)
        matte->key_light_effect.Shutdown();
    }

    bool EnsureInitialized(
        int frame_w, int frame_h,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error) {
      if (error)
        error->clear();
      if (frame_w <= 0 || frame_h <= 0) {
        last_error = OpenVulkanVirtualKeyLightInitializationFailure(
            "invalid frame size");
        if (error)
          *error = last_error;
        return false;
      }
      if (!matte) {
        last_error = OpenVulkanVirtualKeyLightInitializationFailure(
            "matte provider not set");
        if (error)
          *error = last_error;
        return false;
      }
      std::string matte_err;
      if (!matte->EnsureInitialized(frame_w, frame_h, fx, &matte_err,
                                    /*require_vb_buffers=*/false)) {
        last_error = OpenVulkanVirtualKeyLightInitializationFailure(matte_err);
        if (error)
          *error = last_error;
        return false;
      }
      const auto matting_readiness = matte->matting_session->Readiness();
      if (!matte->key_light_effect.EnsureInitialized(
              &matte->kernels, frame_w, frame_h, fx.virtual_key_light.intensity,
              fx.virtual_key_light.temperature_preset,
              fx.virtual_key_light.direction_pan_degrees, matting_readiness,
              &last_error)) {
        if (error)
          *error = last_error;
        return false;
      }
      last_error.clear();
      return true;
    }

    bool
    ApplyVulkanRgb(const studiocast::vulkan::VulkanImage &in_rgb,
                   studiocast::vulkan::VulkanImage *out_rgb_img,
                   const studiocast::video::effects::BroadcastCameraEffects &fx,
                   std::uint64_t capture_sequence, std::string *error,
                   DeferredGpuOut *deferred_out) {
      if (error)
        error->clear();
      if (!deferred_out) {
        if (error)
          *error =
              OpenVulkanVirtualKeyLightRuntimeFailure("invalid GPU output");
        return false;
      }
      const auto publish_passthrough = [&]() {
        deferred_out->kind = DeferredGpuKind::vulkan_rgb;
        deferred_out->nvcv_img = nullptr;
        deferred_out->nvcv = nullptr;
        deferred_out->cuda_img = nullptr;
        deferred_out->cuda = nullptr;
        deferred_out->stream = nullptr;
        deferred_out->vulkan_img = &in_rgb;
        deferred_out->vulkan_kernels = matte ? &matte->kernels : nullptr;
      };
      if (!fx.virtual_key_light.enabled) {
        publish_passthrough();
        return true;
      }
      if (!matte || !out_rgb_img) {
        if (error)
          *error = OpenVulkanVirtualKeyLightRuntimeFailure(
              "invalid GPU stage state");
        return false;
      }
      if (!in_rgb.Valid() ||
          in_rgb.format() != studiocast::vulkan::VulkanPixelFormat::rgb_u8) {
        if (error)
          *error = OpenVulkanVirtualKeyLightRuntimeFailure(
              "invalid input GPU RGB image");
        return false;
      }
      std::string init_err;
      if (!EnsureInitialized(in_rgb.width(), in_rgb.height(), fx, &init_err)) {
        if (error)
          *error = init_err;
        return false;
      }
      const bool passthrough =
          matte->key_light_effect.parameters().passthrough();
      const bool reuse_same_frame =
          !passthrough &&
          matte->HasCompatibleFrameAlpha(capture_sequence, in_rgb.width(),
                                         in_rgb.height());
      const auto readiness_before = matte->matting_session->Readiness();
      const std::uint64_t inference_count_before =
          readiness_before.runtime.inference_count;
      const std::uint64_t resize_count_before =
          matte->alpha_resize_completion_count;

      if (!passthrough && !matte->EnsureFrameAlphaForFrameGpu(
                              in_rgb, capture_sequence, in_rgb.width(),
                              in_rgb.height(), fx, error)) {
        if (error && !error->empty())
          *error = OpenVulkanVirtualKeyLightRuntimeFailure(*error);
        return false;
      }

      const auto current_readiness = matte->matting_session->Readiness();
      OpenVulkanVirtualKeyLightInput input;
      input.foreground = &in_rgb;
      input.alpha = &matte->alpha_resized;
      input.output = out_rgb_img;
      input.capture_sequence = capture_sequence;
      input.resident_alpha_sequence = matte->cached_frame_alpha_sequence;
      input.alpha_resize_completion_count_before = resize_count_before;
      input.alpha_resize_completion_count_after =
          matte->alpha_resize_completion_count;
      input.matting_inference_count_before = inference_count_before;
      input.matte_source =
          reuse_same_frame
              ? OpenVulkanVirtualKeyLightMatteSource::reused_same_frame
              : OpenVulkanVirtualKeyLightMatteSource::independently_inferred;
      input.matting_readiness = &current_readiness;
      if (!matte->key_light_effect.Apply(
              input, &matte->key_light_effect_counters, error)) {
        return false;
      }

      if (passthrough) {
        publish_passthrough();
        return true;
      }
      ++matte->forced_sync_calls;

      deferred_out->kind = DeferredGpuKind::vulkan_rgb;
      deferred_out->nvcv_img = nullptr;
      deferred_out->nvcv = nullptr;
      deferred_out->cuda_img = nullptr;
      deferred_out->cuda = nullptr;
      deferred_out->stream = nullptr;
      deferred_out->vulkan_img = out_rgb_img;
      deferred_out->vulkan_kernels = &matte->kernels;
      return true;
    }
  } open_vulkan_key_light;

  struct OpenVulkanAutoFrameContext {
    using TrackingSource = OpenCudaAutoFrameContext::TrackingSource;
    using TrackingBox = OpenCudaAutoFrameContext::TrackingBox;
    using CropPlan = OpenCudaAutoFrameContext::CropPlan;

    bool initialized = false;
    bool enabled = false;
    std::string last_error;
    std::string active_model_id;
    OpenVulkanVirtualBackgroundContext *matte = nullptr;
    int last_frame_w = 0;
    int last_frame_h = 0;
    std::uint64_t last_capture_sequence = 0;
    studiocast::vulkan::VulkanContextIdentity last_context_identity{};
    OpenVulkanAutoFrameReuseKey reuse_key{};

    OpenVulkanAutoFrame production_crop;
    OpenVulkanAutoFrameCounters counters;

    std::uint64_t last_matte_query_sequence = 0;
    bool have_last_matte_box = false;
    studiocast::maxine::effects::RectF last_matte_box_px{};

    bool have_smoothed_crop = false;
    studiocast::maxine::effects::RectF crop_smoothed_px;
    bool last_had_detection = false;
    TrackingSource last_tracking_source = TrackingSource::kNone;

    void ResetTrackingState(OpenVulkanAutoFrameResetReason reason) {
      production_crop.ResetTemporal(reason, &counters);
      last_capture_sequence = 0;
      last_context_identity = {};
      last_matte_query_sequence = 0;
      have_last_matte_box = false;
      last_matte_box_px = {};
      have_smoothed_crop = false;
      crop_smoothed_px = {};
      last_had_detection = false;
      last_tracking_source = TrackingSource::kNone;
      if (matte)
        matte->InvalidateMatteCache();
    }

    void Destroy() {
      ResetTrackingState(OpenVulkanAutoFrameResetReason::pipeline_restart);
      production_crop.Shutdown();
      initialized = false;
      enabled = false;
      last_error.clear();
      active_model_id.clear();
      last_frame_w = 0;
      last_frame_h = 0;
      reuse_key = {};
    }

    void ObserveEnablementAndGeneration(bool current_enabled,
                                        std::uint64_t effects_generation) {
      OpenVulkanAutoFrameReuseKey next = reuse_key;
      next.enabled = current_enabled;
      next.effects_generation = effects_generation;
      const auto reason =
          OpenVulkanAutoFrameReuseKeyResetReason(reuse_key, next);
      if (reason != OpenVulkanAutoFrameResetReason::none)
        ResetTrackingState(reason);
      reuse_key = std::move(next);
    }

    void PrepareFrame(std::uint64_t capture_sequence) {
      if (!matte || !matte->kernels.device())
        return;
      const auto context_identity =
          matte->kernels.device()->context_identity();
      OpenVulkanAutoFrameResetReason reason =
          OpenVulkanAutoFrameResetReason::none;
      if (last_context_identity.Valid() &&
          last_context_identity != context_identity) {
        reason = OpenVulkanAutoFrameResetReason::
            vulkan_context_generation_change;
      } else if (OpenVulkanAutoFrameSequenceNeedsReset(last_capture_sequence,
                                                       capture_sequence)) {
        reason =
            OpenVulkanAutoFrameResetReason::capture_sequence_discontinuity;
      }
      if (reason != OpenVulkanAutoFrameResetReason::none)
        ResetTrackingState(reason);
      last_context_identity = context_identity;
      last_capture_sequence = capture_sequence;
      if (reuse_key.observed)
        reuse_key.context_identity = context_identity;
    }

    bool EnsureInitialized(
        int frame_w, int frame_h,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        bool require_matte_tracking, std::uint64_t effects_generation,
        std::string_view face_model_id, std::string *error) {
      if (error)
        error->clear();
      auto fail = [&](std::string detail) {
        last_error = OpenVulkanAutoFrameInitializationFailure(detail);
        if (error)
          *error = last_error;
        return false;
      };
      if (frame_w <= 0 || frame_h <= 0) {
        return fail("invalid frame size");
      }

      std::string current_model_id;
      std::string current_provider_id;
      if (!matte) {
        return fail("matte/shared utility provider is not set");
      }
      if (require_matte_tracking) {
        std::string matte_err;
        if (!matte->EnsureInitialized(frame_w, frame_h, fx, &matte_err,
                                      /*require_vb_buffers=*/false,
                                      /*require_cpu_alpha_readback=*/true)) {
          return fail(matte_err);
        }
        current_model_id = matte->active_model_id;
        current_provider_id = "ncnn_vulkan";
      } else if (!matte->kernels.Initialized()) {
        std::string vk_err;
        if (!matte->kernels.Initialize(&vk_err)) {
          return fail("utility kernels unavailable: " + vk_err);
        }
        current_model_id = std::string(face_model_id);
        current_provider_id = "onnxruntime_cpu";
      } else {
        current_model_id = std::string(face_model_id);
        current_provider_id = "onnxruntime_cpu";
      }

      std::string crop_init_error;
      if (!production_crop.EnsureInitialized(&matte->kernels, frame_w, frame_h,
                                             &counters,
                                             &crop_init_error)) {
        last_error = crop_init_error;
        if (error)
          *error = last_error;
        return false;
      }

      OpenVulkanAutoFrameReuseKey next_key;
      next_key.observed = true;
      next_key.enabled = fx.auto_frame.enabled;
      next_key.effects_generation = effects_generation;
      next_key.frame_width = frame_w;
      next_key.frame_height = frame_h;
      next_key.model_id = current_model_id;
      next_key.provider_id = current_provider_id;
      next_key.context_identity = matte->kernels.device()->context_identity();
      const auto reset_reason =
          OpenVulkanAutoFrameReuseKeyResetReason(reuse_key, next_key);
      if (initialized && enabled &&
          reset_reason == OpenVulkanAutoFrameResetReason::none) {
        reuse_key = std::move(next_key);
        return true;
      }
      if (reset_reason != OpenVulkanAutoFrameResetReason::none)
        ResetTrackingState(reset_reason);
      reuse_key = std::move(next_key);
      active_model_id = current_model_id;
      last_frame_w = frame_w;
      last_frame_h = frame_h;
      initialized = true;
      enabled = true;
      last_error.clear();
      return true;
    }

    bool ResolveMatteTrackingBox(
        std::uint64_t capture_sequence,
        const studiocast::vulkan::VulkanImage &in_rgb, int width, int height,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        TrackingBox *out_box, std::string *error) {
      if (out_box)
        *out_box = TrackingBox{};
      if (!out_box || !matte)
        return true;

      const bool matte_cached_this_frame =
          matte->cached_matte_valid &&
          matte->cached_matte_sequence == capture_sequence;
      constexpr std::uint64_t kMatteFallbackIntervalFrames = 10;
      const bool should_query_matte =
          matte_cached_this_frame || (last_matte_query_sequence == 0) ||
          (capture_sequence >=
           (last_matte_query_sequence + kMatteFallbackIntervalFrames));
      if (!should_query_matte) {
        if (have_last_matte_box) {
          out_box->source = TrackingSource::kMatte;
          out_box->box_px = last_matte_box_px;
          out_box->confidence = 1.0f;
        }
        return true;
      }

      std::string matte_err;
      if (!matte->EnsureInitialized(width, height, fx, &matte_err,
                                    /*require_vb_buffers=*/false,
                                    /*require_cpu_alpha_readback=*/true)) {
        if (error && !matte_err.empty())
          *error = "Open Vulkan Auto Frame: " + matte_err;
        return false;
      }

      const std::vector<float> *alpha_cpu = nullptr;
      int alpha_w = 0;
      int alpha_h = 0;
      std::string alpha_err;
      studiocast::maxine::effects::RectF box_px;
      const std::uint64_t alpha_downloads_before =
          matte->alpha_download_calls;
      if (matte->GetAlphaCpuForFrame(in_rgb, capture_sequence, width, height,
                                     fx, &alpha_cpu, &alpha_w, &alpha_h,
                                     &alpha_err) &&
          alpha_cpu) {
        counters.matte_alpha_readback_calls +=
            matte->alpha_download_calls - alpha_downloads_before;
        ++counters.matte_cpu_box_calls;
        const bool found = OpenCudaAutoFrameContext::ComputeMatteBoxPxFromAlpha(
            *alpha_cpu, alpha_w, alpha_h, width, height, &box_px);

        last_matte_query_sequence = capture_sequence;
        if (found) {
          have_last_matte_box = true;
          last_matte_box_px = box_px;
          out_box->source = TrackingSource::kMatte;
          out_box->box_px = box_px;
          out_box->confidence = 1.0f;
        } else {
          have_last_matte_box = false;
          last_matte_box_px = {};
        }
      } else {
        if (error) {
          *error = alpha_err.empty()
                       ? "Open Vulkan Auto Frame: matte alpha readback failed."
                       : "Open Vulkan Auto Frame: " + alpha_err;
        }
        return false;
      }
      return true;
    }

    bool ResolveCropPlan(
        std::uint64_t capture_sequence,
        const studiocast::vulkan::VulkanImage &in_rgb, int width, int height,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        const TrackingBox *tracking_box, bool tracking_provider_ran,
        CropPlan *plan, std::string *error) {
      if (plan)
        *plan = CropPlan{};
      if (!fx.auto_frame.enabled)
        return true;
      if (!plan) {
        if (error)
          *error = "Open Vulkan Auto Frame: null crop plan.";
        return true;
      }

      std::string init_err;
      if (!EnsureInitialized(width, height, fx,
                             /*require_matte_tracking=*/!tracking_provider_ran,
                             reuse_key.effects_generation,
                             reuse_key.model_id,
                             &init_err)) {
        if (error && !init_err.empty())
          *error = init_err;
        return false;
      }

      PrepareFrame(capture_sequence);
      ++counters.cpu_crop_plan_smoothing_calls;

      TrackingBox resolved_box;
      bool found = false;
      if (tracking_box && OpenCudaAutoFrameContext::IsUsableTrackingBox(
                              *tracking_box, width, height)) {
        resolved_box = *tracking_box;
        found = true;
      } else if (!tracking_provider_ran) {
        if (!ResolveMatteTrackingBox(capture_sequence, in_rgb, width, height,
                                     fx, &resolved_box, error)) {
          return false;
        }
        found = OpenCudaAutoFrameContext::IsUsableTrackingBox(resolved_box,
                                                              width, height);
      }

      const TrackingSource source =
          found ? resolved_box.source : TrackingSource::kNone;
      if (source != last_tracking_source) {
        production_crop.ResetTemporal(
            OpenVulkanAutoFrameResetReason::
                geometry_model_or_provider_change,
            &counters);
        have_smoothed_crop = false;
        last_tracking_source = source;
      }

      studiocast::maxine::effects::AutoFrameKnobs knobs;
      knobs.strength = fx.auto_frame.strength;
      knobs.smoothing = fx.auto_frame.smoothing;
      knobs.headroom = fx.auto_frame.headroom;

      const float out_aspect =
          static_cast<float>(width) / static_cast<float>(height);
      studiocast::maxine::effects::RectF target_crop = {};
      if (found) {
        target_crop = studiocast::maxine::effects::ArAutoFrameTracker::
            ComputeTargetCropFromBoxPx(resolved_box.box_px, width, height,
                                       out_aspect, knobs);
      } else {
        const float strength01 =
            std::clamp(static_cast<float>(knobs.strength) / 100.0f, 0.0f, 1.0f);
        const float zoom = 1.0f + strength01 * 0.5f;
        target_crop =
            studiocast::maxine::effects::ArAutoFrameTracker::CenterCrop(
                width, height, out_aspect, zoom);
      }

      auto clamp_crop = [&](studiocast::maxine::effects::RectF *r) {
        if (!r)
          return;
        r->w = std::clamp(r->w, 1.0f, static_cast<float>(width));
        r->h = std::clamp(r->h, 1.0f, static_cast<float>(height));
        r->x = std::clamp(r->x, 0.0f, static_cast<float>(width) - r->w);
        r->y = std::clamp(r->y, 0.0f, static_cast<float>(height) - r->h);
      };

      if (!have_smoothed_crop) {
        crop_smoothed_px = target_crop;
        have_smoothed_crop = true;
      } else {
        const float a =
            studiocast::maxine::effects::ArAutoFrameTracker::SmoothingAlpha(
                knobs.smoothing);
        crop_smoothed_px =
            studiocast::maxine::effects::ArAutoFrameTracker::Lerp(
                crop_smoothed_px, target_crop, a);
      }
      clamp_crop(&crop_smoothed_px);
      last_had_detection = found;

      constexpr float kEpsPx = 0.5f;
      if (std::abs(crop_smoothed_px.x) <= kEpsPx &&
          std::abs(crop_smoothed_px.y) <= kEpsPx &&
          std::abs(crop_smoothed_px.w - static_cast<float>(width)) <= kEpsPx &&
          std::abs(crop_smoothed_px.h - static_cast<float>(height)) <= kEpsPx) {
        return true;
      }

      plan->should_crop = true;
      plan->found_tracking_box = found;
      plan->tracking_source = source;
      plan->crop_px = crop_smoothed_px;
      return true;
    }

    bool
    ApplyVulkanRgb(std::uint64_t capture_sequence,
                   const studiocast::vulkan::VulkanImage &in_rgb,
                   studiocast::vulkan::VulkanImage *out_rgb,
                   const studiocast::video::effects::BroadcastCameraEffects &fx,
                   const TrackingBox *tracking_box, bool tracking_provider_ran,
                   DeferredGpuOut *deferred_out, std::string *error) {
      if (!fx.auto_frame.enabled)
        return true;
      if (!matte || !out_rgb || !deferred_out) {
        if (error)
          *error = "Open Vulkan Auto Frame: invalid GPU stage state.";
        return false;
      }
      if (!in_rgb.Valid() ||
          in_rgb.format() != studiocast::vulkan::VulkanPixelFormat::rgb_u8) {
        if (error)
          *error = "Open Vulkan Auto Frame: invalid input GPU RGB image.";
        return false;
      }

      CropPlan plan;
      std::string plan_err;
      if (!ResolveCropPlan(capture_sequence, in_rgb, in_rgb.width(),
                           in_rgb.height(), fx, tracking_box,
                           tracking_provider_ran, &plan, &plan_err)) {
        if (error)
          *error = plan_err;
        return false;
      }

      deferred_out->kind = DeferredGpuKind::vulkan_rgb;
      deferred_out->nvcv_img = nullptr;
      deferred_out->nvcv = nullptr;
      deferred_out->cuda_img = nullptr;
      deferred_out->cuda = nullptr;
      deferred_out->stream = nullptr;
      deferred_out->vulkan_kernels = &matte->kernels;

      if (plan.should_crop &&
          !matte->EnsureImage(out_rgb, in_rgb.width(), in_rgb.height(),
                              studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                              /*map_memory=*/false,
                              "resident auto_frame_out", error)) {
        return false;
      }
      OpenVulkanResidentAutoFrameInput resident_input;
      resident_input.source = &in_rgb;
      resident_input.crop_output = plan.should_crop ? out_rgb : nullptr;
      resident_input.request_crop = plan.should_crop;
      resident_input.crop_x = plan.crop_px.x;
      resident_input.crop_y = plan.crop_px.y;
      resident_input.crop_width = plan.crop_px.w;
      resident_input.crop_height = plan.crop_px.h;
      resident_input.host_analysis_complete = true;
      resident_input.cpu_crop_plan_complete = true;
      OpenVulkanResidentFrameState resident_state;
      std::string kerr;
      if (!ExecuteOpenVulkanResidentAutoFrameStage(
              resident_input, &production_crop, &counters, &resident_state,
              &kerr)) {
        ResetTrackingState(
            OpenVulkanAutoFrameResetReason::runtime_or_device_failure);
        if (error)
          *error = kerr;
        return false;
      }
      if (resident_state.auto_frame_crop_applied) {
        ++matte->crop_resize_dispatch_calls;
        ++matte->forced_sync_calls;
      }
      deferred_out->vulkan_img = resident_state.current;
      return true;
    }
  } open_vulkan_auto_frame;
#endif

  // Share the same Open CUDA matting model/session (and per-frame matte cache)
  // across all open-source video effects that need a foreground matte. This
  // avoids re-running the matting network multiple times per frame.
  //
  // NOTE: The provider lives in open_cuda_vb even if the virtual background
  // stage is not scheduled; other effects request matte-only initialization.
  open_cuda_auto_frame.matte = &open_cuda_vb;
  open_cuda_key_light.matte = &open_cuda_vb;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
  open_vulkan_auto_frame.matte = &open_vulkan_vb;
  open_vulkan_key_light.matte = &open_vulkan_vb;
#endif

  // Open Video analysis cache (per-frame) and face detection.
  //
  // The cache avoids duplicated ML inference when multiple effects need the
  // same frame-level analysis.
  studiocast::open_video::FrameAnalysisCache open_video_cache;
  studiocast::open_video::YunetFaceDetector open_video_yunet;
  studiocast::open_video::FastDvdnetDenoiser open_video_fastdvdnet;
  studiocast::open_video::GazeCorrectionEyeContact open_video_eye_contact;

  // Open-source Video Noise Removal (Open CUDA backend). This implementation is
  // a lightweight temporal denoiser intended to be real-time and
  // dependency-free. It does not require model packs.
  struct OpenCudaVideoNoiseRemovalContext {
    bool last_enabled = false;
    int prev_w = 0;
    int prev_h = 0;
    size_t prev_stride = 0;
    std::vector<uint8_t> prev_rgb;

    void Reset() {
      last_enabled = false;
      prev_w = 0;
      prev_h = 0;
      prev_stride = 0;
      prev_rgb.clear();
    }

    bool ApplyRgbInPlace(uint8_t *rgb, int width, int height, size_t stride,
                         const video::effects::BroadcastCameraEffects &fx,
                         std::string *error_out) {
      (void)error_out;

      const bool enabled = fx.video_noise_removal.enabled;
      const int strength = std::clamp(fx.video_noise_removal.strength, 0, 100);

      if (!enabled || strength <= 0) {
        // When toggled off, don't keep stale temporal state.
        last_enabled = false;
        return true;
      }

      // (Re)initialize on first frame after enable, or when the image geometry
      // changes.
      const size_t bytes = static_cast<size_t>(height) * stride;
      if (!last_enabled || prev_rgb.size() != bytes || width != prev_w ||
          height != prev_h || stride != prev_stride) {
        prev_w = width;
        prev_h = height;
        prev_stride = stride;
        prev_rgb.assign(rgb, rgb + bytes);
        last_enabled = true;
        return true;
      }

      last_enabled = true;

      // Strength -> smoothing weight and motion threshold.
      // We use a squared curve for weight so low strengths do something subtle.
      const float t = static_cast<float>(strength) / 100.0f;
      const float t2 = t * t;
      const int w_prev =
          std::clamp(static_cast<int>(t2 * 220.0f + 0.5f), 0, 255);
      const int w_curr = 256 - w_prev;

      // Motion threshold: lower threshold at higher strengths to reduce
      // ghosting.
      const int thr_chan =
          std::clamp(static_cast<int>(20.0f - t * 12.0f + 0.5f), 6, 24);
      const int thr_sum = thr_chan * 3;

      uint8_t *prev = prev_rgb.data();
      for (int y = 0; y < height; ++y) {
        uint8_t *row = rgb + static_cast<size_t>(y) * stride;
        uint8_t *prow = prev + static_cast<size_t>(y) * stride;
        for (int x = 0; x < width; ++x) {
          const int i = x * 3;

          const int cr = row[i + 0];
          const int cg = row[i + 1];
          const int cb = row[i + 2];

          const int pr = prow[i + 0];
          const int pg = prow[i + 1];
          const int pb = prow[i + 2];

          const int diff =
              std::abs(cr - pr) + std::abs(cg - pg) + std::abs(cb - pb);
          if (diff > thr_sum || w_prev == 0) {
            // Motion (or effectively disabled smoothing): take current and
            // refresh temporal state.
            prow[i + 0] = static_cast<uint8_t>(cr);
            prow[i + 1] = static_cast<uint8_t>(cg);
            prow[i + 2] = static_cast<uint8_t>(cb);
            continue;
          }

          const int or_ = (cr * w_curr + pr * w_prev + 128) >> 8;
          const int og_ = (cg * w_curr + pg * w_prev + 128) >> 8;
          const int ob_ = (cb * w_curr + pb * w_prev + 128) >> 8;

          row[i + 0] = static_cast<uint8_t>(or_);
          row[i + 1] = static_cast<uint8_t>(og_);
          row[i + 2] = static_cast<uint8_t>(ob_);

          // Store filtered output as the next frame's reference.
          prow[i + 0] = row[i + 0];
          prow[i + 1] = row[i + 1];
          prow[i + 2] = row[i + 2];
        }
      }

      return true;
    }
  } open_cuda_video_denoise;

  struct MaxineRelightingContext {
    bool initialized = false;
    std::string last_error;

    studiocast::maxine::vfx::VfxApi vfx;
    studiocast::maxine::NvcvApi nvcv;
    studiocast::maxine::CudaDriverApi cuda;
    studiocast::maxine::CudaBgrVignette vignette;

    std::filesystem::path model_dir;
    std::filesystem::path features_dir;

    std::unique_ptr<studiocast::maxine::effects::VfxGreenScreenEffect>
        greenscreen;
    std::unique_ptr<studiocast::maxine::effects::VfxRelightingEffect> relight;

    std::vector<std::uint8_t> bgr_in;
    std::vector<std::uint8_t> bgr_out;
    studiocast::maxine::NvCVImage cpu_bgr_in{};
    studiocast::maxine::NvCVImage cpu_bgr_out{};

    studiocast::maxine::NvCVImage gpu_bgr{};
    bool gpu_bgr_allocated = false;

    studiocast::maxine::NvCVImage gpu_bgr_out_img{};
    bool gpu_bgr_out_allocated = false;

    studiocast::maxine::NvCVImage gpu_matte_scaled{};
    bool gpu_matte_scaled_allocated = false;

    int cached_hdri_preset = -1;
    std::filesystem::path cached_hdri_path;

    ~MaxineRelightingContext() { Destroy(); }

    void Destroy() {
      greenscreen.reset();
      relight.reset();

      if (gpu_matte_scaled_allocated && nvcv.IsInitialized() &&
          nvcv.f().NvCVImage_Dealloc) {
        (void)nvcv.f().NvCVImage_Dealloc(&gpu_matte_scaled);
      }
      gpu_matte_scaled = studiocast::maxine::NvCVImage{};
      gpu_matte_scaled_allocated = false;

      if (gpu_bgr_out_allocated && nvcv.IsInitialized() &&
          nvcv.f().NvCVImage_Dealloc) {
        (void)nvcv.f().NvCVImage_Dealloc(&gpu_bgr_out_img);
      }
      gpu_bgr_out_img = studiocast::maxine::NvCVImage{};
      gpu_bgr_out_allocated = false;

      if (gpu_bgr_allocated && nvcv.IsInitialized() &&
          nvcv.f().NvCVImage_Dealloc) {
        (void)nvcv.f().NvCVImage_Dealloc(&gpu_bgr);
      }
      gpu_bgr = studiocast::maxine::NvCVImage{};
      gpu_bgr_allocated = false;

      bgr_in.clear();
      bgr_out.clear();
      cpu_bgr_in = studiocast::maxine::NvCVImage{};
      cpu_bgr_out = studiocast::maxine::NvCVImage{};

      cached_hdri_preset = -1;
      cached_hdri_path.clear();

      initialized = false;
    }

    static std::filesystem::path
    InferModelsDirFromLibrary(const std::filesystem::path &lib) {
      std::error_code ec;
      std::filesystem::path p = lib.parent_path();
      for (int i = 0; i < 5 && !p.empty(); ++i) {
        const auto cand = p / "models";
        if (std::filesystem::exists(cand, ec) &&
            std::filesystem::is_directory(cand, ec)) {
          return cand;
        }
        p = p.parent_path();
      }
      return {};
    }

    static std::filesystem::path
    InferFeaturesDirFromLibrary(const std::filesystem::path &lib) {
      std::error_code ec;
      std::filesystem::path p = lib.parent_path();
      for (int i = 0; i < 5 && !p.empty(); ++i) {
        const auto cand = p / "features";
        if (std::filesystem::exists(cand, ec) &&
            std::filesystem::is_directory(cand, ec)) {
          return cand;
        }
        p = p.parent_path();
      }
      return {};
    }

    static int ScoreHdriName(const std::string &nameLower, int preset) {
      // preset: 0 neutral, 1 warm, 2 cool
      int score = 0;
      if (preset == 1) {
        if (nameLower.find("warm") != std::string::npos)
          score += 10;
        if (nameLower.find("sun") != std::string::npos)
          score += 3;
      } else if (preset == 2) {
        if (nameLower.find("cool") != std::string::npos)
          score += 10;
        if (nameLower.find("blue") != std::string::npos)
          score += 3;
      } else {
        if (nameLower.find("neutral") != std::string::npos)
          score += 10;
        if (nameLower.find("studio") != std::string::npos)
          score += 3;
        if (nameLower.find("white") != std::string::npos)
          score += 2;
      }
      // Prefer "hdri" named files in general.
      if (nameLower.find("hdri") != std::string::npos)
        score += 1;
      return score;
    }

    static std::filesystem::path
    FindDefaultHdriInDir(const std::filesystem::path &dir, int preset) {
      if (dir.empty())
        return {};

      std::error_code ec;
      if (!std::filesystem::exists(dir, ec) ||
          !std::filesystem::is_directory(dir, ec))
        return {};

      std::filesystem::path best;
      int bestScore = -9999;

      std::size_t visited = 0;
      for (std::filesystem::recursive_directory_iterator it(dir, ec), end;
           it != end && !ec; ++it) {
        if (it.depth() > 5) {
          it.disable_recursion_pending();
          continue;
        }
        if (++visited > 2000)
          break;
        if (!it->is_regular_file(ec))
          continue;

        const auto ext = ToLowerAscii(it->path().extension().string());
        if (ext != ".hdr" && ext != ".exr")
          continue;

        const auto nameLower = ToLowerAscii(it->path().filename().string());
        const int sc = ScoreHdriName(nameLower, preset);
        if (sc > bestScore) {
          bestScore = sc;
          best = it->path();
        }
      }

      return best;
    }

    bool ResolveHdriPath(
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::filesystem::path *out, std::string *error) {
      if (!out)
        return false;
      out->clear();

      // User override.
      if (!fx.virtual_key_light.hdri_path.empty()) {
        std::error_code ec;
        const std::filesystem::path p = fx.virtual_key_light.hdri_path;
        if (std::filesystem::exists(p, ec)) {
          *out = p;
          return true;
        }
        if (error)
          *error = "Virtual Key Light HDRI path does not exist: " + p.string();
        return false;
      }

      const int preset = fx.virtual_key_light.temperature_preset;
      if (cached_hdri_preset == preset && !cached_hdri_path.empty()) {
        *out = cached_hdri_path;
        return true;
      }

      // Prefer HDRI assets installed by the relighting feature (if present),
      // otherwise search models.
      auto hdri = FindDefaultHdriInDir(features_dir, preset);
      if (hdri.empty()) {
        hdri = FindDefaultHdriInDir(model_dir, preset);
      }

      if (hdri.empty()) {
        if (error) {
          *error = "No default HDRI found for relighting. Provide an HDRI via "
                   "video.virtual_key_light_hdri (daemon.conf) "
                   "or install the VFX relighting feature assets "
                   "(install_feature.sh for relighting).";
        }
        return false;
      }

      cached_hdri_preset = preset;
      cached_hdri_path = hdri;
      *out = hdri;
      return true;
    }

    bool EnsureInitialized(
        int width, int height,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error) {
      last_error.clear();

      if (initialized) {
        // Live reconfigure.
        std::filesystem::path hdri;
        std::string hdri_err;
        if (!ResolveHdriPath(fx, &hdri, &hdri_err)) {
          last_error = hdri_err;
          if (error)
            *error = last_error;
          return false;
        }

        auto tmp = fx;
        tmp.virtual_key_light.hdri_path = hdri.string();

        std::string cfg_err;
        if (greenscreen && !greenscreen->Configure(tmp, &cfg_err)) {
          if (error)
            *error = cfg_err;
          return false;
        }
        if (relight && !relight->Configure(tmp, &cfg_err)) {
          if (error)
            *error = cfg_err;
          return false;
        }
        return true;
      }

      std::string err;
      if (!vfx.Initialize(&err)) {
        last_error = "Maxine VFX runtime unavailable: " + err;
        if (error)
          *error = last_error;
        return false;
      }
      if (!nvcv.Initialize(studiocast::maxine::NvcvApi::Requirement::VfxCompat,
                           &err)) {
        last_error = "NvCVImage runtime unavailable: " + err;
        if (error)
          *error = last_error;
        return false;
      }

      {
        std::string cuda_err;
        if (!cuda.Initialize(&cuda_err)) {
          last_error = "CUDA driver API unavailable: " + cuda_err;
          if (error)
            *error = last_error;
          return false;
        }
        std::string vig_err;
        if (!vignette.Initialize(&cuda, &vig_err)) {
          last_error = "CUDA vignette init failed: " + vig_err;
          if (error)
            *error = last_error;
          return false;
        }
      }
      if (model_dir.empty()) {
        model_dir = InferModelsDirFromLibrary(vfx.library_path());
      }
      if (features_dir.empty()) {
        features_dir = InferFeaturesDirFromLibrary(vfx.library_path());
      }

      std::filesystem::path hdri;
      std::string hdri_err;
      if (!ResolveHdriPath(fx, &hdri, &hdri_err)) {
        last_error = hdri_err;
        if (error)
          *error = last_error;
        return false;
      }

      // Allocate CPU-side BGR staging buffers and wrap with NvCVImage_Init.
      const std::size_t stride = static_cast<std::size_t>(width) * 3u;
      bgr_in.resize(stride * static_cast<std::size_t>(height));
      bgr_out.resize(stride * static_cast<std::size_t>(height));

      if (!nvcv.f().NvCVImage_Init) {
        last_error = "NvCVImage_Init missing from NvCVImage runtime.";
        if (error)
          *error = last_error;
        return false;
      }

      auto st = nvcv.f().NvCVImage_Init(
          &cpu_bgr_in, static_cast<unsigned>(width),
          static_cast<unsigned>(height), static_cast<int>(stride),
          bgr_in.data(), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_CPU);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        last_error = "NvCVImage_Init(cpu BGR in) failed: " + std::to_string(st);
        if (error)
          *error = last_error;
        return false;
      }

      st = nvcv.f().NvCVImage_Init(
          &cpu_bgr_out, static_cast<unsigned>(width),
          static_cast<unsigned>(height), static_cast<int>(stride),
          bgr_out.data(), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_CPU);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        last_error =
            "NvCVImage_Init(cpu BGR out) failed: " + std::to_string(st);
        if (error)
          *error = last_error;
        return false;
      }

      // Allocate GPU input, output, and matte-scaled images.
      st = nvcv.f().NvCVImage_Alloc(
          &gpu_bgr, static_cast<unsigned>(width), static_cast<unsigned>(height),
          studiocast::maxine::NVCV_BGR, studiocast::maxine::NVCV_U8,
          studiocast::maxine::NVCV_CHUNKY, studiocast::maxine::NVCV_GPU,
          /*alignment=*/0);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        last_error = "NvCVImage_Alloc(gpu BGR) failed: " + std::to_string(st);
        if (error)
          *error = last_error;
        return false;
      }
      gpu_bgr_allocated = true;

      st = nvcv.f().NvCVImage_Alloc(
          &gpu_bgr_out_img, static_cast<unsigned>(width),
          static_cast<unsigned>(height), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_GPU,
          /*alignment=*/0);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        last_error =
            "NvCVImage_Alloc(gpu BGR out) failed: " + std::to_string(st);
        if (error)
          *error = last_error;
        return false;
      }
      gpu_bgr_out_allocated = true;

      st = nvcv.f().NvCVImage_Alloc(
          &gpu_matte_scaled, static_cast<unsigned>(width),
          static_cast<unsigned>(height), studiocast::maxine::NVCV_A,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_GPU,
          /*alignment=*/0);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        last_error =
            "NvCVImage_Alloc(gpu matte scaled) failed: " + std::to_string(st);
        if (error)
          *error = last_error;
        return false;
      }
      gpu_matte_scaled_allocated = true;

      greenscreen =
          std::make_unique<studiocast::maxine::effects::VfxGreenScreenEffect>(
              &vfx, &nvcv, model_dir);
      relight =
          std::make_unique<studiocast::maxine::effects::VfxRelightingEffect>(
              &vfx, &nvcv, model_dir);

      auto tmp = fx;
      tmp.virtual_key_light.hdri_path = hdri.string();

      if (!greenscreen->Configure(tmp, &err) ||
          !relight->Configure(tmp, &err)) {
        last_error = "Maxine effect Configure failed: " + err;
        if (error)
          *error = last_error;
        Destroy();
        return false;
      }
      if (!greenscreen->Initialize(&err)) {
        last_error = "Maxine green screen Initialize failed: " + err;
        if (error)
          *error = last_error;
        Destroy();
        return false;
      }
      if (!relight->Initialize(&err)) {
        last_error = "Maxine relighting Initialize failed: " + err;
        if (error)
          *error = last_error;
        Destroy();
        return false;
      }

      initialized = true;
      return true;
    }

    bool ApplyRgbInPlace(
        std::uint8_t *rgb, int width, int height, std::size_t rgb_stride,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        bool apply_vignette, float vignette_center_x_px,
        float vignette_center_y_px,
        const studiocast::maxine::NvCVImage *shared_green_screen_matte,
        bool *used_shared_green_screen_matte, std::string *error,
        bool defer_readback, DeferredGpuOut *deferred_out) {
      if (used_shared_green_screen_matte)
        *used_shared_green_screen_matte = false;
      if (!initialized || !greenscreen || !relight) {
        if (error)
          *error = "Maxine relighting not initialized.";
        return false;
      }
      if (!rgb || width <= 0 || height <= 0 || rgb_stride == 0) {
        if (error)
          *error = "Invalid RGB buffer.";
        return false;
      }

      const float intensity =
          Clamp01FromPercent(fx.virtual_key_light.intensity);
      if (intensity <= 0.0001f) {
        // No-op.
        return true;
      }

      std::filesystem::path hdri;
      std::string hdri_err;
      if (!ResolveHdriPath(fx, &hdri, &hdri_err)) {
        if (error)
          *error = hdri_err;
        return false;
      }

      auto tmp = fx;
      tmp.virtual_key_light.hdri_path = hdri.string();

      const bool use_shared_matte = (shared_green_screen_matte != nullptr);
      const auto transfer_stream = use_shared_matte
                                       ? relight->cuda_stream()
                                       : greenscreen->cuda_stream();

      // RGB -> BGR staging
      studiocast::video::Rgb24ToBgr24(rgb, bgr_in.data(), width, height,
                                      rgb_stride, rgb_stride);

      // Upload CPU->GPU.
      const auto up = nvcv.f().NvCVImage_Transfer(&cpu_bgr_in, &gpu_bgr, 1.0f,
                                                  transfer_stream, nullptr);
      if (up != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error = "NvCVImage_Transfer(cpu->gpu) failed: " + std::to_string(up);
        return false;
      }

      studiocast::video::GpuFrame frame;
      frame.width = width;
      frame.height = height;
      frame.nvcv_gpu = &gpu_bgr;
      frame.cuda_stream = greenscreen->cuda_stream();

      std::string cfg_err;
      if ((!use_shared_matte && !greenscreen->Configure(tmp, &cfg_err)) ||
          !relight->Configure(tmp, &cfg_err)) {
        if (error)
          *error = cfg_err;
        return false;
      }

      std::string proc_err;
      const studiocast::maxine::NvCVImage *matte = shared_green_screen_matte;
      if (!use_shared_matte) {
        auto st = greenscreen->Process(frame, &proc_err);
        if (st != studiocast::maxine::NVCV_SUCCESS) {
          if (error)
            *error = proc_err.empty() ? std::to_string(st) : proc_err;
          return false;
        }

        matte = greenscreen->MatteGpu();
        if (!matte) {
          if (error)
            *error = "Green Screen did not produce a matte.";
          return false;
        }
      } else if (used_shared_green_screen_matte) {
        *used_shared_green_screen_matte = true;
      }

      frame.matte_gpu = matte;

      auto st = relight->Process(frame, &proc_err);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error = proc_err.empty() ? std::to_string(st) : proc_err;
        return false;
      }

      const auto *relit_fg = relight->OutputGpu();
      if (!relit_fg) {
        if (error)
          *error = "Relighting did not produce an output image.";
        return false;
      }

      if (!nvcv.f().NvCVImage_Composite || !nvcv.f().NvCVImage_Transfer) {
        if (error)
          *error = "NvCVImage_Composite/Transfer unavailable.";
        return false;
      }

      const auto stream = greenscreen->cuda_stream();

      // Intensity: scale matte alpha before compositing.
      const studiocast::maxine::NvCVImage *mat_for_comp = matte;
      if (intensity < 0.999f) {
        const auto sc = nvcv.f().NvCVImage_Transfer(matte, &gpu_matte_scaled,
                                                    intensity, stream, nullptr);
        if (sc != studiocast::maxine::NVCV_SUCCESS) {
          if (error)
            *error =
                "NvCVImage_Transfer(matte scale) failed: " + std::to_string(sc);
          return false;
        }
        mat_for_comp = &gpu_matte_scaled;
      }

      const auto stc = nvcv.f().NvCVImage_Composite(
          relit_fg, &gpu_bgr, mat_for_comp, &gpu_bgr_out_img, stream);
      if (stc != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error = "NvCVImage_Composite failed: " + std::to_string(stc);
        return false;
      }

      if (apply_vignette && fx.vignette.enabled) {
        const float vig_intensity = Clamp01FromPercent(fx.vignette.intensity);
        if (vig_intensity > 0.0001f) {
          std::string ve;
          if (!vignette.ApplyInPlace(&gpu_bgr_out_img, vig_intensity,
                                     vignette_center_x_px, vignette_center_y_px,
                                     stream, &ve)) {
            if (error)
              *error = OptionalEffectFailureReason("Vignette failed", ve);
            return false;
          }
        }
      }

      if (defer_readback) {
        if (!deferred_out) {
          if (error)
            *error = "defer_readback requires deferred_out.";
          return false;
        }
        deferred_out->kind = DeferredGpuKind::nvcv_bgr;
        deferred_out->nvcv_img = &gpu_bgr_out_img;
        deferred_out->nvcv = &nvcv;
        deferred_out->cuda_img = nullptr;
        deferred_out->cuda = &cuda;
        deferred_out->stream = stream;
        return true;
      }

      const auto down = nvcv.f().NvCVImage_Transfer(
          &gpu_bgr_out_img, &cpu_bgr_out, 1.0f, stream, nullptr);
      if (down != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error =
              "NvCVImage_Transfer(gpu->cpu) failed: " + std::to_string(down);
        return false;
      }

      if (!SyncAfterGpuToCpuTransfer(
              cuda, stream, "Maxine relight: after gpu->cpu transfer", error)) {
        return false;
      }

      // BGR -> RGB back into the pipeline buffer.
      studiocast::video::Bgr24ToRgb24(bgr_out.data(), rgb, width, height,
                                      rgb_stride, rgb_stride);
      return true;
    }
  } maxine_relight;

  struct MaxineDenoiseContext {
    bool initialized = false;
    std::string last_error;

    studiocast::maxine::vfx::VfxApi vfx;
    studiocast::maxine::NvcvApi nvcv;
    studiocast::maxine::CudaDriverApi cuda;

    std::filesystem::path model_dir;
    std::unique_ptr<studiocast::maxine::effects::VfxDenoiseEffect> denoise;

    std::vector<std::uint8_t> bgr_in;
    std::vector<std::uint8_t> bgr_out;
    studiocast::maxine::NvCVImage cpu_bgr_in{};
    studiocast::maxine::NvCVImage cpu_bgr_out{};

    // Denoise expects BGRf32 planar normalized on GPU (see VfxDenoiseEffect
    // docs).
    studiocast::maxine::NvCVImage gpu_bgr_f32_in{};
    bool gpu_bgr_f32_in_allocated = false;

    ~MaxineDenoiseContext() { Destroy(); }

    void Destroy() {
      denoise.reset();

      if (gpu_bgr_f32_in_allocated && nvcv.IsInitialized() &&
          nvcv.f().NvCVImage_Dealloc) {
        (void)nvcv.f().NvCVImage_Dealloc(&gpu_bgr_f32_in);
      }
      gpu_bgr_f32_in = studiocast::maxine::NvCVImage{};
      gpu_bgr_f32_in_allocated = false;

      bgr_in.clear();
      bgr_out.clear();
      cpu_bgr_in = studiocast::maxine::NvCVImage{};
      cpu_bgr_out = studiocast::maxine::NvCVImage{};

      model_dir.clear();

      initialized = false;
    }

    static std::filesystem::path
    InferModelsDirFromLibrary(const std::filesystem::path &lib) {
      std::error_code ec;
      std::filesystem::path p = lib.parent_path();
      for (int i = 0; i < 5 && !p.empty(); ++i) {
        const auto cand = p / "models";
        if (std::filesystem::exists(cand, ec) &&
            std::filesystem::is_directory(cand, ec)) {
          return cand;
        }
        p = p.parent_path();
      }
      return {};
    }

    bool EnsureInitialized(
        int width, int height,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error) {
      last_error.clear();

      if (initialized) {
        // Live reconfiguration (strength).
        std::string cfg_err;
        if (denoise && !denoise->Configure(fx, &cfg_err)) {
          if (error)
            *error = cfg_err;
          return false;
        }
        return true;
      }

      std::string err;
      if (!vfx.Initialize(&err)) {
        last_error = "Maxine VFX runtime unavailable: " + err;
        if (error)
          *error = last_error;
        return false;
      }
      if (!nvcv.Initialize(studiocast::maxine::NvcvApi::Requirement::VfxCompat,
                           &err)) {
        last_error = "NvCVImage runtime unavailable: " + err;
        if (error)
          *error = last_error;
        return false;
      }

      {
        std::string cuda_err;
        if (!cuda.Initialize(&cuda_err)) {
          last_error = "CUDA driver API unavailable: " + cuda_err;
          if (error)
            *error = last_error;
          return false;
        }
      }

      // Model directory discovery.
      {
        const char *env = std::getenv("STUDIOCAST_MAXINE_MODELS_DIR");
        if (env && *env) {
          model_dir = std::filesystem::path(env);
        }
      }
      if (model_dir.empty()) {
        model_dir = InferModelsDirFromLibrary(vfx.library_path());
      }

      // Allocate CPU BGR staging buffers.
      const std::size_t stride = static_cast<std::size_t>(width) * 3u;
      bgr_in.resize(stride * static_cast<std::size_t>(height));
      bgr_out.resize(stride * static_cast<std::size_t>(height));

      if (!nvcv.f().NvCVImage_Init) {
        last_error = "NvCVImage_Init missing from NvCVImage runtime.";
        if (error)
          *error = last_error;
        return false;
      }

      auto st = nvcv.f().NvCVImage_Init(
          &cpu_bgr_in, static_cast<unsigned>(width),
          static_cast<unsigned>(height), static_cast<int>(stride),
          bgr_in.data(), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_CPU);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        last_error = "NvCVImage_Init(cpu BGR in) failed: " + std::to_string(st);
        if (error)
          *error = last_error;
        return false;
      }

      st = nvcv.f().NvCVImage_Init(
          &cpu_bgr_out, static_cast<unsigned>(width),
          static_cast<unsigned>(height), static_cast<int>(stride),
          bgr_out.data(), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_CPU);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        last_error =
            "NvCVImage_Init(cpu BGR out) failed: " + std::to_string(st);
        if (error)
          *error = last_error;
        return false;
      }

      // Allocate GPU input (BGRf32 planar).
      st = nvcv.f().NvCVImage_Alloc(
          &gpu_bgr_f32_in, static_cast<unsigned>(width),
          static_cast<unsigned>(height), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_F32, studiocast::maxine::NVCV_PLANAR,
          studiocast::maxine::NVCV_GPU,
          /*alignment=*/0);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        last_error =
            "NvCVImage_Alloc(gpu BGRf32 planar) failed: " + std::to_string(st);
        if (error)
          *error = last_error;
        return false;
      }
      gpu_bgr_f32_in_allocated = true;

      denoise = std::make_unique<studiocast::maxine::effects::VfxDenoiseEffect>(
          &vfx, &nvcv, model_dir);

      std::string cfg_err;
      if (!denoise->Configure(fx, &cfg_err)) {
        last_error =
            cfg_err.empty() ? "Maxine denoise Configure failed." : cfg_err;
        if (error)
          *error = last_error;
        return false;
      }

      std::string init_err;
      if (!denoise->Initialize(&init_err)) {
        last_error =
            init_err.empty() ? "Maxine denoise Initialize failed." : init_err;
        if (error)
          *error = last_error;
        return false;
      }

      initialized = true;
      return true;
    }

    bool ApplyRgbInPlace(
        std::uint8_t *rgb, int width, int height, std::size_t rgb_stride,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error) {
      if (!rgb) {
        if (error)
          *error = "rgb buffer is null.";
        return false;
      }

      if (!EnsureInitialized(width, height, fx, error)) {
        return false;
      }
      if (!denoise) {
        if (error)
          *error = "Maxine denoise not initialized.";
        return false;
      }

      // RGB -> BGR (CPU staging).
      const std::size_t bgr_stride = static_cast<std::size_t>(width) * 3u;
      studiocast::video::Rgb24ToBgr24(rgb, bgr_in.data(), width, height,
                                      rgb_stride, bgr_stride);

      // CPU(BGRu8) -> GPU(BGRf32 planar)
      const auto stream = denoise->cuda_stream();
      auto st = nvcv.f().NvCVImage_Transfer(&cpu_bgr_in, &gpu_bgr_f32_in, 1.0f,
                                            stream, nullptr);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error = "NvCVImage_Transfer(cpu->gpu BGRf32) failed: " +
                   std::to_string(st);
        return false;
      }

      studiocast::video::GpuFrame frame;
      frame.width = width;
      frame.height = height;
      frame.nvcv_gpu = &gpu_bgr_f32_in;
      frame.cuda_stream = stream;

      std::string mx_err;
      st = denoise->Process(frame, &mx_err);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error = mx_err.empty() ? "Maxine denoise Process failed." : mx_err;
        return false;
      }

      const auto *out_gpu = denoise->OutputGpu();
      if (!out_gpu) {
        if (error)
          *error = "Maxine denoise did not produce an output image.";
        return false;
      }

      // GPU(BGRf32 planar) -> CPU(BGRu8)
      st = nvcv.f().NvCVImage_Transfer(out_gpu, &cpu_bgr_out, 1.0f, stream,
                                       nullptr);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error = "NvCVImage_Transfer(gpu->cpu) failed: " + std::to_string(st);
        return false;
      }

      if (!SyncAfterGpuToCpuTransfer(
              cuda, stream, "Maxine denoise: after gpu->cpu transfer", error)) {
        return false;
      }

      // BGR -> RGB
      studiocast::video::Bgr24ToRgb24(bgr_out.data(), rgb, width, height,
                                      bgr_stride, rgb_stride);
      return true;
    }
  } maxine_denoise;

  struct MaxineEyeContactContext {
    bool initialized = false;
    std::string last_error;

    studiocast::maxine::ar::ArApi ar;
    studiocast::maxine::NvcvApi nvcv;
    studiocast::maxine::CudaDriverApi cuda;
    studiocast::maxine::CudaBgrVignette vignette;

    std::unique_ptr<studiocast::maxine::effects::ArEyeContactEffect>
        eye_contact;

    std::vector<std::uint8_t> bgr_in;
    std::vector<std::uint8_t> bgr_out;
    studiocast::maxine::NvCVImage cpu_bgr_in{};
    studiocast::maxine::NvCVImage cpu_bgr_out{};

    studiocast::maxine::NvCVImage gpu_bgr_in{};
    bool gpu_bgr_in_allocated = false;

    studiocast::maxine::NvCVImage gpu_bgr_out{};
    bool gpu_bgr_out_allocated = false;

    studiocast::maxine::CUstream stream = nullptr;
    bool stream_owned = false;

    ~MaxineEyeContactContext() { Destroy(); }

    void Destroy() {
      eye_contact.reset();

      if (stream_owned && stream && ar.IsInitialized() &&
          ar.f().NvAR_CudaStreamDestroy) {
        (void)ar.f().NvAR_CudaStreamDestroy(stream);
      }
      stream = nullptr;
      stream_owned = false;

      if (gpu_bgr_out_allocated && nvcv.IsInitialized() &&
          nvcv.f().NvCVImage_Dealloc) {
        (void)nvcv.f().NvCVImage_Dealloc(&gpu_bgr_out);
      }
      gpu_bgr_out = studiocast::maxine::NvCVImage{};
      gpu_bgr_out_allocated = false;

      if (gpu_bgr_in_allocated && nvcv.IsInitialized() &&
          nvcv.f().NvCVImage_Dealloc) {
        (void)nvcv.f().NvCVImage_Dealloc(&gpu_bgr_in);
      }
      gpu_bgr_in = studiocast::maxine::NvCVImage{};
      gpu_bgr_in_allocated = false;

      bgr_in.clear();
      bgr_out.clear();
      cpu_bgr_in = studiocast::maxine::NvCVImage{};
      cpu_bgr_out = studiocast::maxine::NvCVImage{};

      initialized = false;
      last_error.clear();
    }

    bool EnsureInitialized(
        int width, int height,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error) {
      if (initialized) {
        std::string cfg_err;
        if (eye_contact && !eye_contact->Configure(fx, &cfg_err)) {
          if (error)
            *error = cfg_err;
          return false;
        }
        return true;
      }

      unsigned w = 0;
      unsigned h = 0;
      {
        std::string dim_err;
        if (!CheckedPositiveDims(width, height, &w, &h, &dim_err,
                                 "Maxine eye contact:")) {
          if (error)
            *error = dim_err;
          last_error = dim_err;
          return false;
        }
      }

      std::string ar_err;
      if (!ar.Initialize(&ar_err)) {
        if (error)
          *error = ar_err;
        last_error = ar_err;
        return false;
      }

      std::string nvcv_err;
      if (!nvcv.Initialize(studiocast::maxine::NvcvApi::Requirement::VfxCompat,
                           &nvcv_err)) {
        if (error)
          *error = nvcv_err;
        last_error = nvcv_err;
        return false;
      }

      {
        std::string cuda_err;
        if (!cuda.Initialize(&cuda_err)) {
          const std::string e = "CUDA driver API unavailable: " + cuda_err;
          if (error)
            *error = e;
          last_error = e;
          return false;
        }
        std::string vig_err;
        if (!vignette.Initialize(&cuda, &vig_err)) {
          const std::string e = "CUDA vignette init failed: " + vig_err;
          if (error)
            *error = e;
          last_error = e;
          return false;
        }
      }

      if (!nvcv.f().NvCVImage_Init || !nvcv.f().NvCVImage_Alloc ||
          !nvcv.f().NvCVImage_Transfer) {
        const std::string e = "NvCVImage_Init/Alloc/Transfer unavailable.";
        if (error)
          *error = e;
        last_error = e;
        return false;
      }

      // Stream: create one so transfers and AR run on a consistent stream.
      if (ar.f().NvAR_CudaStreamCreate) {
        const auto st = ar.f().NvAR_CudaStreamCreate(&stream);
        if (st != studiocast::maxine::NVCV_SUCCESS) {
          const std::string e =
              "NvAR_CudaStreamCreate failed: " + std::to_string(st);
          if (error)
            *error = e;
          last_error = e;
          return false;
        }
        stream_owned = true;
      }

      bgr_in.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) *
                    3u);
      bgr_out.resize(bgr_in.size());

      const auto init_in = nvcv.f().NvCVImage_Init(
          &cpu_bgr_in, w, h, static_cast<int>(static_cast<std::size_t>(w) * 3u),
          bgr_in.data(), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_CPU);
      if (init_in != studiocast::maxine::NVCV_SUCCESS) {
        const std::string e =
            "NvCVImage_Init(cpu input) failed: " + std::to_string(init_in);
        if (error)
          *error = e;
        last_error = e;
        return false;
      }

      const auto init_out = nvcv.f().NvCVImage_Init(
          &cpu_bgr_out, w, h,
          static_cast<int>(static_cast<std::size_t>(w) * 3u), bgr_out.data(),
          studiocast::maxine::NVCV_BGR, studiocast::maxine::NVCV_U8,
          studiocast::maxine::NVCV_CHUNKY, studiocast::maxine::NVCV_CPU);
      if (init_out != studiocast::maxine::NVCV_SUCCESS) {
        const std::string e =
            "NvCVImage_Init(cpu output) failed: " + std::to_string(init_out);
        if (error)
          *error = e;
        last_error = e;
        return false;
      }

      const auto alloc_in = nvcv.f().NvCVImage_Alloc(
          &gpu_bgr_in, w, h, studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_GPU, 1);
      if (alloc_in != studiocast::maxine::NVCV_SUCCESS) {
        const std::string e =
            "NvCVImage_Alloc(gpu input) failed: " + std::to_string(alloc_in);
        if (error)
          *error = e;
        last_error = e;
        return false;
      }
      gpu_bgr_in_allocated = true;

      const auto alloc_out = nvcv.f().NvCVImage_Alloc(
          &gpu_bgr_out, w, h, studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_GPU, 1);
      if (alloc_out != studiocast::maxine::NVCV_SUCCESS) {
        const std::string e =
            "NvCVImage_Alloc(gpu output) failed: " + std::to_string(alloc_out);
        if (error)
          *error = e;
        last_error = e;
        return false;
      }
      gpu_bgr_out_allocated = true;

      eye_contact =
          std::make_unique<studiocast::maxine::effects::ArEyeContactEffect>(
              &ar);

      std::string cfg_err;
      if (!eye_contact->Configure(fx, &cfg_err)) {
        if (error)
          *error = cfg_err;
        last_error = cfg_err;
        return false;
      }

      initialized = true;
      return true;
    }

    bool ApplyRgbInPlace(
        std::uint8_t *rgb, int width, int height, std::size_t rgb_stride,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        bool apply_vignette, float vignette_center_x_px,
        float vignette_center_y_px, std::string *error, bool defer_readback,
        DeferredGpuOut *deferred_out) {
      if (!rgb || width <= 0 || height <= 0)
        return true;
      std::string init_err;
      if (!EnsureInitialized(width, height, fx, &init_err)) {
        if (error)
          *error = init_err;
        return false;
      }
      if (!eye_contact)
        return true;

      // RGB -> BGR
      studiocast::video::Rgb24ToBgr24(rgb, bgr_in.data(), width, height,
                                      rgb_stride,
                                      static_cast<std::size_t>(width) * 3u);

      // CPU -> GPU
      const auto up = nvcv.f().NvCVImage_Transfer(&cpu_bgr_in, &gpu_bgr_in,
                                                  1.0f, stream, nullptr);
      if (up != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error = "NvCVImage_Transfer(cpu->gpu) failed: " + std::to_string(up);
        return false;
      }

      studiocast::video::GpuFrame frame;
      frame.width = width;
      frame.height = height;
      frame.nvcv_gpu = &gpu_bgr_in;
      frame.nvcv_tmp = &gpu_bgr_out;
      frame.cuda_stream = stream;

      std::string proc_err;
      const auto st = eye_contact->Process(frame, &proc_err);
      if (st != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error = proc_err.empty() ? std::to_string(st) : proc_err;
        return false;
      }

      if (apply_vignette && fx.vignette.enabled) {
        const float vig_intensity = Clamp01FromPercent(fx.vignette.intensity);
        if (vig_intensity > 0.0001f) {
          std::string ve;
          if (!vignette.ApplyInPlace(&gpu_bgr_out, vig_intensity,
                                     vignette_center_x_px, vignette_center_y_px,
                                     stream, &ve)) {
            if (error)
              *error = OptionalEffectFailureReason("Vignette failed", ve);
            return false;
          }
        }
      }

      if (defer_readback) {
        if (!deferred_out) {
          if (error)
            *error = "defer_readback requires deferred_out.";
          return false;
        }
        deferred_out->kind = DeferredGpuKind::nvcv_bgr;
        deferred_out->nvcv_img = &gpu_bgr_out;
        deferred_out->nvcv = &nvcv;
        deferred_out->cuda_img = nullptr;
        deferred_out->cuda = &cuda;
        deferred_out->stream = stream;
        return true;
      }

      // GPU -> CPU
      const auto down = nvcv.f().NvCVImage_Transfer(&gpu_bgr_out, &cpu_bgr_out,
                                                    1.0f, stream, nullptr);
      if (down != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error =
              "NvCVImage_Transfer(gpu->cpu) failed: " + std::to_string(down);
        return false;
      }

      if (!SyncAfterGpuToCpuTransfer(
              cuda, stream, "Maxine eye contact: after gpu->cpu transfer",
              error)) {
        return false;
      }

      // BGR -> RGB
      studiocast::video::Bgr24ToRgb24(bgr_out.data(), rgb, width, height,
                                      static_cast<std::size_t>(width) * 3u,
                                      rgb_stride);
      return true;
    }
  } maxine_eye_contact;

  struct MaxineAutoFrameContext {
    bool initialized = false;
    std::string last_error;

    studiocast::maxine::ar::ArApi ar;
    studiocast::maxine::NvcvApi nvcv;
    studiocast::maxine::CudaDriverApi cuda;
    studiocast::maxine::CudaBgrCropScale crop_scale;
    studiocast::maxine::CudaBgrVignette vignette;

    std::unique_ptr<studiocast::maxine::effects::ArAutoFrameTracker> tracker;

    std::vector<std::uint8_t> bgr_in;
    std::vector<std::uint8_t> bgr_out;
    studiocast::maxine::NvCVImage cpu_bgr_in{};
    studiocast::maxine::NvCVImage cpu_bgr_out{};

    studiocast::maxine::NvCVImage gpu_bgr_in{};
    bool gpu_bgr_in_allocated = false;
    studiocast::maxine::NvCVImage gpu_bgr_out{};
    bool gpu_bgr_out_allocated = false;

    studiocast::maxine::CUstream stream = nullptr;
    bool stream_owned = false;

    ~MaxineAutoFrameContext() { Destroy(); }

    void Destroy() {
      tracker.reset();

      if (stream_owned && stream && ar.IsInitialized() &&
          ar.f().NvAR_CudaStreamDestroy) {
        (void)ar.f().NvAR_CudaStreamDestroy(stream);
      }
      stream = nullptr;
      stream_owned = false;

      if (gpu_bgr_out_allocated && nvcv.IsInitialized() &&
          nvcv.f().NvCVImage_Dealloc) {
        (void)nvcv.f().NvCVImage_Dealloc(&gpu_bgr_out);
      }
      gpu_bgr_out = studiocast::maxine::NvCVImage{};
      gpu_bgr_out_allocated = false;

      if (gpu_bgr_in_allocated && nvcv.IsInitialized() &&
          nvcv.f().NvCVImage_Dealloc) {
        (void)nvcv.f().NvCVImage_Dealloc(&gpu_bgr_in);
      }
      gpu_bgr_in = studiocast::maxine::NvCVImage{};
      gpu_bgr_in_allocated = false;

      bgr_in.clear();
      bgr_out.clear();
      cpu_bgr_in = studiocast::maxine::NvCVImage{};
      cpu_bgr_out = studiocast::maxine::NvCVImage{};

      initialized = false;
      last_error.clear();
    }

    bool EnsureInitialized(
        int width, int height,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        std::string *error) {
      if (initialized)
        return true;

      unsigned w = 0;
      unsigned h = 0;
      {
        std::string dim_err;
        if (!CheckedPositiveDims(width, height, &w, &h, &dim_err,
                                 "Maxine auto frame:")) {
          if (error)
            *error = dim_err;
          last_error = dim_err;
          return false;
        }
      }

      std::string ar_err;
      if (!ar.Initialize(&ar_err)) {
        if (error)
          *error = ar_err;
        last_error = ar_err;
        return false;
      }

      std::string nvcv_err;
      if (!nvcv.Initialize(studiocast::maxine::NvcvApi::Requirement::VfxCompat,
                           &nvcv_err)) {
        if (error)
          *error = nvcv_err;
        last_error = nvcv_err;
        return false;
      }

      std::string cuda_err;
      if (!cuda.Initialize(&cuda_err)) {
        if (error)
          *error = cuda_err;
        last_error = cuda_err;
        return false;
      }

      if (!nvcv.f().NvCVImage_Init || !nvcv.f().NvCVImage_Alloc ||
          !nvcv.f().NvCVImage_Transfer) {
        const std::string e = "NvCVImage_Init/Alloc/Transfer unavailable.";
        if (error)
          *error = e;
        last_error = e;
        return false;
      }

      // Stream: create one so transfers, AR, and kernel launch share a stream.
      if (ar.f().NvAR_CudaStreamCreate) {
        const auto st = ar.f().NvAR_CudaStreamCreate(&stream);
        if (st != studiocast::maxine::NVCV_SUCCESS) {
          const std::string e =
              "NvAR_CudaStreamCreate failed: " + std::to_string(st);
          if (error)
            *error = e;
          last_error = e;
          return false;
        }
        stream_owned = true;
      }

      bgr_in.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) *
                    3u);
      bgr_out.resize(bgr_in.size());

      const auto init_in = nvcv.f().NvCVImage_Init(
          &cpu_bgr_in, w, h, static_cast<int>(static_cast<std::size_t>(w) * 3u),
          bgr_in.data(), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_CPU);
      if (init_in != studiocast::maxine::NVCV_SUCCESS) {
        const std::string e =
            "NvCVImage_Init(cpu input) failed: " + std::to_string(init_in);
        if (error)
          *error = e;
        last_error = e;
        return false;
      }

      const auto init_out = nvcv.f().NvCVImage_Init(
          &cpu_bgr_out, w, h,
          static_cast<int>(static_cast<std::size_t>(w) * 3u), bgr_out.data(),
          studiocast::maxine::NVCV_BGR, studiocast::maxine::NVCV_U8,
          studiocast::maxine::NVCV_CHUNKY, studiocast::maxine::NVCV_CPU);
      if (init_out != studiocast::maxine::NVCV_SUCCESS) {
        const std::string e =
            "NvCVImage_Init(cpu output) failed: " + std::to_string(init_out);
        if (error)
          *error = e;
        last_error = e;
        return false;
      }

      const auto alloc_in = nvcv.f().NvCVImage_Alloc(
          &gpu_bgr_in, w, h, studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_GPU, 1);
      if (alloc_in != studiocast::maxine::NVCV_SUCCESS) {
        const std::string e =
            "NvCVImage_Alloc(gpu input) failed: " + std::to_string(alloc_in);
        if (error)
          *error = e;
        last_error = e;
        return false;
      }
      gpu_bgr_in_allocated = true;

      const auto alloc_out = nvcv.f().NvCVImage_Alloc(
          &gpu_bgr_out, w, h, studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_GPU, 1);
      if (alloc_out != studiocast::maxine::NVCV_SUCCESS) {
        const std::string e =
            "NvCVImage_Alloc(gpu output) failed: " + std::to_string(alloc_out);
        if (error)
          *error = e;
        last_error = e;
        return false;
      }
      gpu_bgr_out_allocated = true;

      std::string crop_err;
      if (!crop_scale.Initialize(&cuda, &crop_err)) {
        if (error)
          *error = crop_err;
        last_error = crop_err;
        return false;
      }

      {
        std::string vig_err;
        if (!vignette.Initialize(&cuda, &vig_err)) {
          if (error)
            *error = vig_err;
          last_error = vig_err;
          return false;
        }
      }

      tracker =
          std::make_unique<studiocast::maxine::effects::ArAutoFrameTracker>(
              &ar);
      tracker->SetOutputAspect(static_cast<float>(width) /
                               static_cast<float>(height));
      studiocast::maxine::effects::AutoFrameKnobs k;
      k.strength = fx.auto_frame.strength;
      k.smoothing = fx.auto_frame.smoothing;
      k.headroom = fx.auto_frame.headroom;
      tracker->SetKnobs(k);

      std::string tr_err;
      if (!tracker->EnsureInitialized(&gpu_bgr_in, &tr_err)) {
        if (error)
          *error = tr_err;
        last_error = tr_err;
        return false;
      }

      initialized = true;
      return true;
    }

    bool ApplyRgbInPlace(
        std::uint8_t *rgb, int width, int height, std::size_t rgb_stride,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        bool apply_vignette, float vignette_center_x_px,
        float vignette_center_y_px, std::string *error, bool defer_readback,
        DeferredGpuOut *deferred_out) {
      if (!rgb || width <= 0 || height <= 0)
        return true;

      std::string init_err;
      if (!EnsureInitialized(width, height, fx, &init_err)) {
        if (error)
          *error = init_err;
        return false;
      }
      if (!tracker)
        return true;

      // Update knobs without requiring re-init.
      tracker->SetOutputAspect(static_cast<float>(width) /
                               static_cast<float>(height));
      studiocast::maxine::effects::AutoFrameKnobs k;
      k.strength = fx.auto_frame.strength;
      k.smoothing = fx.auto_frame.smoothing;
      k.headroom = fx.auto_frame.headroom;
      tracker->SetKnobs(k);

      // RGB -> BGR
      studiocast::video::Rgb24ToBgr24(rgb, bgr_in.data(), width, height,
                                      rgb_stride,
                                      static_cast<std::size_t>(width) * 3u);

      // CPU -> GPU
      const auto up = nvcv.f().NvCVImage_Transfer(&cpu_bgr_in, &gpu_bgr_in,
                                                  1.0f, stream, nullptr);
      if (up != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error = "NvCVImage_Transfer(cpu->gpu) failed: " + std::to_string(up);
        return false;
      }

      std::string tr_err;
      if (!tracker->Update(width, height, &tr_err)) {
        if (error)
          *error = tr_err;
        return false;
      }

      const auto crop = tracker->SmoothedCropPx();

      std::string cs_err;
      if (!crop_scale.CropScale(gpu_bgr_in, &gpu_bgr_out, crop.x, crop.y,
                                crop.w, crop.h, stream, &cs_err)) {
        if (error)
          *error = cs_err;
        return false;
      }

      if (apply_vignette && fx.vignette.enabled) {
        const float vig_intensity = Clamp01FromPercent(fx.vignette.intensity);
        if (vig_intensity > 0.0001f) {
          float cx = vignette_center_x_px;
          float cy = vignette_center_y_px;
          if (fx.vignette.center_on_tracked_face &&
              tracker->last_had_detection()) {
            cx = static_cast<float>(crop.x) + static_cast<float>(crop.w) * 0.5f;
            cy = static_cast<float>(crop.y) + static_cast<float>(crop.h) * 0.5f;
          }

          std::string ve;
          if (!vignette.ApplyInPlace(&gpu_bgr_out, vig_intensity, cx, cy,
                                     stream, &ve)) {
            if (error)
              *error = OptionalEffectFailureReason("Vignette failed", ve);
            return false;
          }
        }
      }

      if (defer_readback) {
        if (!deferred_out) {
          if (error)
            *error = "defer_readback requires deferred_out.";
          return false;
        }
        deferred_out->kind = DeferredGpuKind::nvcv_bgr;
        deferred_out->nvcv_img = &gpu_bgr_out;
        deferred_out->nvcv = &nvcv;
        deferred_out->cuda_img = nullptr;
        deferred_out->cuda = &cuda;
        deferred_out->stream = stream;
        return true;
      }

      // GPU -> CPU
      const auto down = nvcv.f().NvCVImage_Transfer(&gpu_bgr_out, &cpu_bgr_out,
                                                    1.0f, stream, nullptr);
      if (down != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error =
              "NvCVImage_Transfer(gpu->cpu) failed: " + std::to_string(down);
        return false;
      }

      if (!SyncAfterGpuToCpuTransfer(
              cuda, stream, "Maxine auto frame: after gpu->cpu transfer",
              error)) {
        return false;
      }

      // BGR -> RGB
      studiocast::video::Bgr24ToRgb24(bgr_out.data(), rgb, width, height,
                                      static_cast<std::size_t>(width) * 3u,
                                      rgb_stride);
      return true;
    }
  } maxine_auto_frame;

  // Standalone GPU vignette stage.
  //
  // Used when vignette is enabled but no other Maxine GPU stage is active. This
  // keeps vignette GPU-only and avoids a silent CPU fallback.
  struct VignetteOnlyContext {
    bool initialized = false;
    std::string last_error;

    studiocast::maxine::NvcvApi nvcv;
    studiocast::maxine::CudaDriverApi cuda;
    studiocast::maxine::CudaBgrVignette vignette;

    std::vector<std::uint8_t> bgr_in;
    std::vector<std::uint8_t> bgr_out;
    studiocast::maxine::NvCVImage cpu_bgr_in{};
    studiocast::maxine::NvCVImage cpu_bgr_out{};

    studiocast::maxine::NvCVImage gpu_bgr{};
    bool gpu_bgr_allocated = false;

    ~VignetteOnlyContext() { Destroy(); }

    void Destroy() {
      if (gpu_bgr_allocated && nvcv.IsInitialized() &&
          nvcv.f().NvCVImage_Dealloc) {
        (void)nvcv.f().NvCVImage_Dealloc(&gpu_bgr);
      }
      gpu_bgr = studiocast::maxine::NvCVImage{};
      gpu_bgr_allocated = false;

      bgr_in.clear();
      bgr_out.clear();
      cpu_bgr_in = studiocast::maxine::NvCVImage{};
      cpu_bgr_out = studiocast::maxine::NvCVImage{};

      initialized = false;
      last_error.clear();
    }

    bool EnsureInitialized(int width, int height, std::string *error) {
      last_error.clear();
      if (initialized)
        return true;

      unsigned w = 0;
      unsigned h = 0;
      {
        std::string dim_err;
        if (!CheckedPositiveDims(width, height, &w, &h, &dim_err,
                                 "Vignette:")) {
          last_error = dim_err;
          if (error)
            *error = last_error;
          return false;
        }
      }

      std::string nvcv_err;
      if (!nvcv.Initialize(studiocast::maxine::NvcvApi::Requirement::VfxCompat,
                           &nvcv_err)) {
        last_error = "NvCVImage runtime unavailable: " + nvcv_err;
        if (error)
          *error = last_error;
        return false;
      }

      std::string cuda_err;
      if (!cuda.Initialize(&cuda_err)) {
        last_error = "CUDA driver API unavailable: " + cuda_err;
        if (error)
          *error = last_error;
        return false;
      }

      std::string vig_err;
      if (!vignette.Initialize(&cuda, &vig_err)) {
        last_error = "CUDA vignette init failed: " + vig_err;
        if (error)
          *error = last_error;
        return false;
      }

      if (!nvcv.f().NvCVImage_Init || !nvcv.f().NvCVImage_Alloc ||
          !nvcv.f().NvCVImage_Transfer) {
        last_error = "NvCVImage_Init/Alloc/Transfer unavailable.";
        if (error)
          *error = last_error;
        return false;
      }

      bgr_in.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) *
                    3u);
      bgr_out.resize(bgr_in.size());

      const auto init_in = nvcv.f().NvCVImage_Init(
          &cpu_bgr_in, w, h, static_cast<int>(static_cast<std::size_t>(w) * 3u),
          bgr_in.data(), studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_CPU);
      if (init_in != studiocast::maxine::NVCV_SUCCESS) {
        last_error =
            "NvCVImage_Init(cpu input) failed: " + std::to_string(init_in);
        if (error)
          *error = last_error;
        return false;
      }

      const auto init_out = nvcv.f().NvCVImage_Init(
          &cpu_bgr_out, w, h,
          static_cast<int>(static_cast<std::size_t>(w) * 3u), bgr_out.data(),
          studiocast::maxine::NVCV_BGR, studiocast::maxine::NVCV_U8,
          studiocast::maxine::NVCV_CHUNKY, studiocast::maxine::NVCV_CPU);
      if (init_out != studiocast::maxine::NVCV_SUCCESS) {
        last_error =
            "NvCVImage_Init(cpu output) failed: " + std::to_string(init_out);
        if (error)
          *error = last_error;
        return false;
      }

      const auto alloc = nvcv.f().NvCVImage_Alloc(
          &gpu_bgr, w, h, studiocast::maxine::NVCV_BGR,
          studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
          studiocast::maxine::NVCV_GPU, 1);
      if (alloc != studiocast::maxine::NVCV_SUCCESS) {
        last_error =
            "NvCVImage_Alloc(gpu bgr) failed: " + std::to_string(alloc);
        if (error)
          *error = last_error;
        return false;
      }
      gpu_bgr_allocated = true;

      initialized = true;
      return true;
    }

    bool ApplyRgbInPlace(
        std::uint8_t *rgb, int width, int height, std::size_t rgb_stride,
        const studiocast::video::effects::BroadcastCameraEffects &fx,
        float vignette_center_x_px, float vignette_center_y_px,
        std::string *error, bool defer_readback, DeferredGpuOut *deferred_out) {
      if (!rgb || width <= 0 || height <= 0)
        return true;
      if (!fx.vignette.enabled)
        return true;

      const float vig_intensity = Clamp01FromPercent(fx.vignette.intensity);
      if (vig_intensity <= 0.0001f)
        return true;

      std::string init_err;
      if (!EnsureInitialized(width, height, &init_err)) {
        if (error)
          *error = init_err;
        return false;
      }

      // RGB -> BGR
      studiocast::video::Rgb24ToBgr24(rgb, bgr_in.data(), width, height,
                                      rgb_stride,
                                      static_cast<std::size_t>(width) * 3u);

      // CPU -> GPU
      const auto up = nvcv.f().NvCVImage_Transfer(&cpu_bgr_in, &gpu_bgr, 1.0f,
                                                  /*stream=*/nullptr, nullptr);
      if (up != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error = "NvCVImage_Transfer(cpu->gpu) failed: " + std::to_string(up);
        return false;
      }

      std::string ve;
      if (!vignette.ApplyInPlace(&gpu_bgr, vig_intensity, vignette_center_x_px,
                                 vignette_center_y_px,
                                 /*stream=*/nullptr, &ve)) {
        if (error)
          *error = OptionalEffectFailureReason("Vignette failed", ve);
        return false;
      }

      if (defer_readback) {
        if (!deferred_out) {
          if (error)
            *error = "defer_readback requires deferred_out.";
          return false;
        }
        deferred_out->kind = DeferredGpuKind::nvcv_bgr;
        deferred_out->nvcv_img = &gpu_bgr;
        deferred_out->nvcv = &nvcv;
        deferred_out->cuda_img = nullptr;
        deferred_out->cuda = &cuda;
        deferred_out->stream = nullptr;
        return true;
      }

      // GPU -> CPU
      const auto down = nvcv.f().NvCVImage_Transfer(
          &gpu_bgr, &cpu_bgr_out, 1.0f, /*stream=*/nullptr, nullptr);
      if (down != studiocast::maxine::NVCV_SUCCESS) {
        if (error)
          *error =
              "NvCVImage_Transfer(gpu->cpu) failed: " + std::to_string(down);
        return false;
      }

      if (!SyncAfterGpuToCpuTransfer(cuda,
                                     /*stream=*/nullptr,
                                     "Vignette-only: after gpu->cpu transfer",
                                     error)) {
        return false;
      }

      // BGR -> RGB
      studiocast::video::Bgr24ToRgb24(bgr_out.data(), rgb, width, height,
                                      static_cast<std::size_t>(width) * 3u,
                                      rgb_stride);
      return true;
    }
  } vignette_only;

  bool want_maxine_auto_frame = false;
  bool have_maxine_auto_frame = false;

  bool want_maxine_eye_contact = false;
  bool have_maxine_eye_contact = false;

  bool want_open_video_eye_contact = false;
  bool have_open_video_eye_contact = false;

  bool want_maxine_relight = false;
  bool have_maxine_relight = false;

  bool want_maxine_denoise = false;
  bool have_maxine_denoise = false;

  bool want_open_video_video_denoise = false;
  bool have_open_video_video_denoise = false;

  bool want_open_cuda_video_denoise = false;
  bool have_open_cuda_video_denoise = false;

  bool want_maxine_bg_blur = false;
  bool have_maxine_bg_blur = false;

  bool want_open_cuda_vb = false;
  bool have_open_cuda_vb = false;

  bool want_open_vulkan_vb = false;
  bool have_open_vulkan_vb = false;

  bool want_open_vulkan_auto_frame = false;
  bool have_open_vulkan_auto_frame = false;

  bool want_open_vulkan_key_light = false;
  bool have_open_vulkan_key_light = false;

  bool want_open_vulkan_vignette = false;
  bool have_open_vulkan_vignette = false;

  bool want_open_vulkan_mirror = false;
  bool have_open_vulkan_mirror = false;

  bool want_open_cuda_auto_frame = false;
  bool have_open_cuda_auto_frame = false;

  bool want_open_cuda_key_light = false;
  bool have_open_cuda_key_light = false;

  bool want_maxine_vignette_only = false;
  bool have_maxine_vignette_only = false;

  using EffectStage =
      studiocast::video::effects::BroadcastEffectStage;
  PreparedMaxineResidentIslands prepared_maxine_resident_islands{};
  std::unique_ptr<studiocast::maxine::ProductionResidentFrameExecutor>
      maxine_resident_executor;
  studiocast::maxine::ResidentFrameSection maxine_resident_section;
  PreparedMaxineHardwareExecutionOwner prepared_maxine_hardware_owner;
  bool prepared_maxine_hardware_live = false;
  bool prepared_maxine_hardware_suspended = false;
  std::vector<std::uint8_t> prepared_maxine_bgr_output;
  std::vector<std::uint8_t> maxine_resident_background_rgb;
  studiocast::maxine::ProductionResidentTelemetry
      maxine_resident_telemetry_snapshot{};
  std::uint64_t maxine_background_setup_upload_attempts = 0;
  std::uint64_t maxine_background_setup_upload_calls = 0;
  std::uint64_t maxine_resident_setup_attempts = 0;
  std::uint64_t maxine_resident_setup_successes = 0;

  LiveEffectBackendAttribution live_effect_attribution;
  ProductionVideoExecutionComposition prepared_hardware_composition;
  bool prepared_hardware_final_live = false;
  bool prepared_hardware_runtime_suspended = false;
  std::uint64_t prepared_hardware_geometry_generation = 1;
  std::uint64_t prepared_hardware_next_plan_id = 1;
  auto prepared_hardware_snapshot = [&] {
    const auto source = prepared_hardware_composition.Snapshot();
    PreparedHardwareExecutionSnapshot snapshot;
    snapshot.route = source.prepared ? PreparedHardwareRoute::hal_live
                                     : PreparedHardwareRoute::legacy_shadow;
    snapshot.eligibility = source.prepared
                               ? PreparedHardwareEligibility::eligible
                               : PreparedHardwareEligibility::unavailable;
    snapshot.setup.mode = HardwareExecutionSetupMode::hal_live;
    snapshot.setup.status = source.prepare.status;
    snapshot.setup.reason = source.prepare.reason;
    snapshot.setup.resolution = source.plan.resolution;
    snapshot.setup.domain_plan = source.plan.domain_plan;
    snapshot.setup.legacy_path_retained_for_setup_rollback =
        source.prepare.old_plan_retained;
    snapshot.readiness = source.readiness;
    snapshot.telemetry = source.telemetry;
    snapshot.latest_committed_frame = source.latest_committed_frame;
    snapshot.prepared = source.prepared;
    if (source.prepared) {
      snapshot.active_program.valid = true;
      snapshot.active_program.program_id =
          source.plan.prepared_program.program_id;
      snapshot.active_program.owner_adapter_instance_id =
          source.plan.prepared_program.owner_adapter_instance_id;
      snapshot.active_program.invalidation_key =
          source.plan.prepared_program.invalidation_key;
      snapshot.active_program.execution_domain =
          source.plan.prepared_program.execution_domain;
      snapshot.active_program.stage_count =
          source.plan.prepared_program.stage_count;
    }
    return snapshot;
  };
  auto publish_prepared_hardware_status = [&] {
    const auto snapshot = prepared_hardware_snapshot();
    std::lock_guard<std::mutex> lock(mu_);
    prepared_hardware_execution_ = snapshot;
  };
  PreparedOpenCudaHardwareExecutionOwner prepared_open_cuda_hardware_owner;
  bool prepared_open_cuda_hardware_live = false;
  bool prepared_open_cuda_hardware_suspended = false;
  bool prepared_open_cuda_setup_attempted = false;
  std::vector<std::uint8_t> prepared_open_cuda_rgb_output;
  auto publish_prepared_open_cuda_status = [&] {
    const auto snapshot = prepared_open_cuda_hardware_owner.Snapshot();
    std::lock_guard<std::mutex> lock(mu_);
    prepared_open_cuda_hardware_execution_ = snapshot;
  };
  auto publish_prepared_maxine_status = [&] {
    const auto snapshot = prepared_maxine_hardware_owner.Snapshot();
    std::lock_guard<std::mutex> lock(mu_);
    prepared_maxine_hardware_execution_ = snapshot;
  };
  std::uint64_t video_hal_status_generation = 0;
  {
    std::lock_guard<std::mutex> lock(mu_);
    video_hal_status_generation = video_hal_status_generation_;
  }
  auto publish_video_hal_status =
      [&](const effects::BroadcastCameraEffects &status_effects,
          std::uint64_t configuration_generation,
          VideoRuntimeLifecycle lifecycle) {
        ++video_hal_status_generation;
        if (video_hal_status_generation == 0)
          ++video_hal_status_generation;
        CameraPipelineHalStatusProjectionInput projection;
        projection.effects = &status_effects;
        projection.compute_backend = cfg.compute_backend;
        projection.lifecycle = lifecycle;
        projection.status_generation = video_hal_status_generation;
        projection.configuration_generation =
            configuration_generation == 0 ? 1 : configuration_generation;
        projection.geometry_generation =
            prepared_hardware_geometry_generation;
        projection.final_visual = prepared_hardware_snapshot();
        projection.maxine_resident =
            prepared_maxine_hardware_owner.Snapshot();
        projection.open_cuda_virtual_background =
            prepared_open_cuda_hardware_owner.Snapshot();
        projection.final_visual_live =
            prepared_hardware_final_live &&
            !prepared_hardware_runtime_suspended;
        projection.maxine_resident_live =
            prepared_maxine_hardware_live &&
            !prepared_maxine_hardware_suspended &&
            !projection.maxine_resident.quarantined &&
            !projection.maxine_resident.stopped;
        projection.open_cuda_virtual_background_live =
            prepared_open_cuda_hardware_live &&
            !prepared_open_cuda_hardware_suspended &&
            !projection.open_cuda_virtual_background.quarantined &&
            !projection.open_cuda_virtual_background.stopped &&
            !projection.open_cuda_virtual_background.shutdown_pending;
        MultiIslandVideoRuntimeStatusSnapshot snapshot;
        const auto validation =
            BuildCameraPipelineHalStatusSnapshot(projection, &snapshot);
        std::lock_guard<std::mutex> lock(mu_);
        video_hal_status_generation_ = video_hal_status_generation;
        prepared_hardware_execution_ = projection.final_visual;
        prepared_maxine_hardware_execution_ =
            projection.maxine_resident;
        prepared_open_cuda_hardware_execution_ =
            projection.open_cuda_virtual_background;
        video_hal_status_ =
            validation.code == MultiIslandStatusValidationCode::valid
                ? snapshot
                : MultiIslandVideoRuntimeStatusSnapshot{};
      };

  enum class OptionalEffectSlot : std::size_t {
    denoise = 0,
    eye_contact,
    relight,
    virtual_background,
    auto_frame,
    vignette,
    mirror,
    count,
  };

  std::array<OptionalEffectBreaker,
             static_cast<std::size_t>(OptionalEffectSlot::count)>
      optional_effect_breakers{};
  int optional_effect_trip_order = 0;
  std::string appliedEffectsBaseNote;

  auto optional_breaker =
      [&](OptionalEffectSlot slot) -> OptionalEffectBreaker & {
    return optional_effect_breakers[static_cast<std::size_t>(slot)];
  };

  auto publish_optional_effect_status = [&](std::uint64_t frame_index) {
    const std::string runtime_note =
        BuildDegradedEffectsNote(optional_effect_breakers, frame_index);
    const std::string combined_note =
        CombineEffectsNotes(appliedEffectsBaseNote, runtime_note);
    const auto degraded =
        MostRecentDegradedEffect(optional_effect_breakers, frame_index);

    std::lock_guard<std::mutex> lock(mu_);
    const bool last_error_was_previous_effects_note =
        !effects_note_.empty() && last_error_ == effects_note_;

    effects_note_ = combined_note;
    degraded_effect_ = degraded;
    if (!combined_note.empty()) {
      last_error_ = combined_note;
    } else if (last_error_was_previous_effects_note) {
      last_error_.clear();
    }
  };

  auto block_optional_effect = [&](OptionalEffectSlot slot,
                                   std::string_view effect_id,
                                   std::string_view backend, std::string reason,
                                   std::uint64_t frame_index) {
    auto &breaker = optional_breaker(slot);
    breaker.OnFailure(effect_id, backend, std::move(reason), frame_index,
                      ++optional_effect_trip_order);
    if (live_effect_attribution.RemoveEffect(effect_id)) {
      std::lock_guard<std::mutex> lock(mu_);
      effects_backends_ = live_effect_attribution.active_backends();
    }
    publish_optional_effect_status(frame_index);
  };

  auto publish_optional_retry_ready_status = [&](std::uint64_t frame_index) {
    bool changed = false;
    for (auto &breaker : optional_effect_breakers) {
      changed = breaker.MarkRetryReadyIfDue(frame_index) || changed;
    }
    if (changed)
      publish_optional_effect_status(frame_index);
  };

  auto clear_optional_effect_on_success = [&](OptionalEffectBreaker &breaker,
                                              std::uint64_t frame_index) {
    if (breaker.OnSuccess())
      publish_optional_effect_status(frame_index);
  };

  auto rebuildChain = [&](const studiocast::video::effects::
                              BroadcastCameraEffects &fx,
                          const detail::PreparedReplaceBackgroundSource
                              &replace_source,
                          std::uint64_t effects_generation) {
    // A rebuilt plan is not active until one frame has executed its stages and
    // crossed the writer boundary successfully.
    live_effect_attribution.Clear();
    {
      std::lock_guard<std::mutex> lock(mu_);
      effects_backends_.clear();
    }
    chain.Clear();
    // An effects generation is a hard temporal boundary. Recreate the YuNet
    // provider session so provider/model policy cannot leak across explicit
    // backend changes, and discard all tracking/smoothing/matte state before
    // the rebuilt plan becomes live.
    open_video_yunet.Reset();
    open_video_cache.Clear();
#if STUDIOCAST_ENABLE_OPEN_VULKAN
    open_vulkan_auto_frame.ObserveEnablementAndGeneration(
        fx.auto_frame.enabled, effects_generation);
    open_vulkan_vb.InvalidateMatteCache();
#endif
    open_cuda_vb.ConfigureReplaceBackgroundSource(replace_source);
#if STUDIOCAST_ENABLE_OPEN_VULKAN
    open_vulkan_vb.ConfigureReplaceBackgroundSource(replace_source);
#endif

    auto plan = studiocast::video::effects::BuildBroadcastEffectsPlan(fx);

    std::string note;

    std::optional<studiocast::maxine::MaxineDiagnostics> maxine_diag;
    auto get_maxine_diag =
        [&]() -> const studiocast::maxine::MaxineDiagnostics & {
      if (!maxine_diag.has_value()) {
        studiocast::maxine::MaxineManager mgr;
        maxine_diag = mgr.Diagnose(false);
      }
      return *maxine_diag;
    };
    auto append_canonical_maxine_blocked =
        [&](studiocast::maxine::MaxineNeed need) {
          const auto &d = get_maxine_diag();
          const auto c =
              studiocast::maxine::BuildCanonicalMaxineBlockedCopy(d, need);
          std::string s =
              studiocast::maxine::FormatCanonicalMaxineBlockedCopy(c);
          if (s.empty())
            s = c.summary;
          if (!s.empty()) {
            if (!note.empty())
              note += "\n";
            note += s;
          }
        };

    auto append_rule_notes = [&] {
      if (plan.disabled.empty())
        return;
      if (!note.empty())
        note += "\n";
      note += "Effect rules:";
      for (const auto &d : plan.disabled) {
        note += "\n - " + d.id + ": " + d.reason;
      }
    };

    auto append_open_video_denoise_tensor_note = [&] {
      const auto status = open_video_fastdvdnet.tensor_io_status();
      if (status.summary.empty())
        return;
      if (!note.empty())
        note += "\n";
      note += status.summary;
    };
    auto append_open_video_face_detection_note = [&] {
      const auto status = open_video_yunet.runtime_status();
      if (status.summary.empty())
        return;
      if (!note.empty())
        note += "\n";
      note += status.summary;
    };
    auto append_open_video_eye_contact_note = [&] {
      const auto status = open_video_eye_contact.runtime_status();
      if (status.summary.empty())
        return;
      if (!note.empty())
        note += "\n";
      note += status.summary;
    };

    want_maxine_bg_blur = false;
    have_maxine_bg_blur = false;

    want_open_cuda_vb = false;
    have_open_cuda_vb = false;
    prepared_open_cuda_hardware_live =
        prepared_open_cuda_hardware_owner.Snapshot().prepared;
    prepared_open_cuda_hardware_suspended = false;
    prepared_open_cuda_setup_attempted = false;

    want_open_vulkan_vb = false;
    have_open_vulkan_vb = false;

    want_open_vulkan_auto_frame = false;
    have_open_vulkan_auto_frame = false;

    want_open_vulkan_key_light = false;
    have_open_vulkan_key_light = false;

    want_open_vulkan_vignette = false;
    have_open_vulkan_vignette = false;

    want_open_vulkan_mirror = false;
    have_open_vulkan_mirror = false;

    want_open_cuda_auto_frame = false;
    have_open_cuda_auto_frame = false;

    want_open_cuda_key_light = false;
    have_open_cuda_key_light = false;

    want_maxine_auto_frame = false;
    have_maxine_auto_frame = false;

    want_maxine_eye_contact = false;
    have_maxine_eye_contact = false;

    want_open_video_eye_contact = false;
    have_open_video_eye_contact = false;

    want_maxine_relight = false;
    have_maxine_relight = false;

    want_maxine_denoise = false;
    have_maxine_denoise = false;

    want_open_video_video_denoise = false;
    have_open_video_video_denoise = false;

    want_open_cuda_video_denoise = false;
    have_open_cuda_video_denoise = false;

    want_maxine_vignette_only = false;
    have_maxine_vignette_only = false;

    auto planned = [&] {
      std::set<std::string> s;
      for (const auto &id : plan.ordered_effect_ids)
        s.insert(id);
      return s;
    }();

    const auto has = [&](std::string_view id) {
      return planned.count(std::string(id)) != 0;
    };

#if STUDIOCAST_ENABLE_OPEN_VULKAN
    if (!has(studiocast::video::effects::contract::kEffectIdVignette))
      open_vulkan_vignette.Shutdown();
    if (!has(studiocast::video::effects::contract::kEffectIdVirtualKeyLight))
      open_vulkan_key_light.Destroy();
#endif

    const bool engine_maxine =
        (fx.engine ==
         studiocast::video::effects::EffectsEnginePreference::maxine);
    const bool engine_open_cuda =
        (fx.engine ==
         studiocast::video::effects::EffectsEnginePreference::open_cuda);

    bool maxine_strict_blocked = false;

    std::unordered_map<std::string, std::string> backend_for_effect;
    auto set_backend = [&](std::string_view effect_id,
                           std::string_view backend) {
      backend_for_effect[std::string(effect_id)] = std::string(backend);
    };

    auto remove_stage_from_plan = [&](std::string_view effect_id) {
      const std::string id(effect_id);
      plan.ordered_effect_ids.erase(std::remove(plan.ordered_effect_ids.begin(),
                                                plan.ordered_effect_ids.end(),
                                                id),
                                    plan.ordered_effect_ids.end());
      if (plan.vignette_attach_to_effect_id == id) {
        plan.vignette_attach_to_effect_id.clear();
      }
      planned.erase(id);
    };

    const auto append_note = [&](std::string_view s) {
      if (s.empty())
        return;
      if (!note.empty())
        note += "\n";
      note += std::string(s);
    };

    const bool compute_work_requested =
        studiocast::video::effects::BroadcastEffectsPlanRequestsCompute(plan);
    const bool cuda_video_compute_allowed =
        ComputeBackendAllowsCudaVideoCompute(cfg.compute_backend);
    ComputeBackendAvailability compute_available;
    compute_available.vulkan_available = false;
    compute_available.vulkan_unavailable_reason =
        "Vulkan compute backend requested, but no Open Vulkan video effect is "
        "available.";

    const auto remove_compute_stages_from_plan = [&] {
      remove_stage_from_plan(
          studiocast::video::effects::contract::kEffectIdVideoNoiseRemoval);
      remove_stage_from_plan(
          studiocast::video::effects::contract::kEffectIdEyeContact);
      remove_stage_from_plan(
          studiocast::video::effects::contract::kEffectIdVirtualBackgroundBlur);
      remove_stage_from_plan(studiocast::video::effects::contract::
                                 kEffectIdVirtualBackgroundRemove);
      remove_stage_from_plan(studiocast::video::effects::contract::
                                 kEffectIdVirtualBackgroundReplace);
      remove_stage_from_plan(
          studiocast::video::effects::contract::kEffectIdAutoFrame);
      remove_stage_from_plan(
          studiocast::video::effects::contract::kEffectIdVirtualKeyLight);
      remove_stage_from_plan(
          studiocast::video::effects::contract::kEffectIdVignette);
      remove_stage_from_plan(
          studiocast::video::effects::contract::kEffectIdMirror);
    };

    auto compute_selection = ResolveComputeBackendSelection(
        cfg.compute_backend, compute_available, compute_work_requested);

    if (compute_work_requested &&
        cfg.compute_backend == ComputeBackendPreference::vulkan) {
      const auto denoise_compatibility =
          ApplyOpenVulkanVideoNoiseRemovalPlanCompatibility(&plan);
      if (denoise_compatibility.blocked) {
        planned.erase(std::string(
            studiocast::video::effects::contract::kEffectIdVideoNoiseRemoval));
        compute_available.vulkan_unavailable_reason =
            "[" + std::string(denoise_compatibility.reason_code) + "] [" +
            std::string(denoise_compatibility.blocker_code) + "] " +
            std::string(denoise_compatibility.detail);
      }
      const auto eye_contact_compatibility =
          ApplyOpenVulkanEyeContactPlanCompatibility(&plan);
      if (eye_contact_compatibility.blocked) {
        planned.erase(std::string(
            studiocast::video::effects::contract::kEffectIdEyeContact));
        compute_available.vulkan_unavailable_reason =
            "[" + std::string(eye_contact_compatibility.reason_code) +
            "] [" + std::string(eye_contact_compatibility.blocker_code) +
            "] " + std::string(eye_contact_compatibility.detail);
      }
      // Vulkan vignette is a final-output stage rather than an attached model
      // stage. Tracked-center compatibility is evaluated after Auto Frame
      // readiness is known, so no hidden analysis tail is introduced.
      plan.vignette_attach_to_effect_id.clear();
      append_note("Open Vulkan: virtual background, virtual key light, and "
                  "auto frame are eligible for the Vulkan backend in this "
                  "milestone; vignette and mirror are production final-output "
                  "Vulkan stages.");
    } else if (compute_work_requested &&
               compute_selection.resolved == ComputeBackendKind::cpu &&
               cfg.compute_backend == ComputeBackendPreference::cpu) {
      remove_compute_stages_from_plan();
      append_note(compute_selection.degraded_reason);
    }

    const bool vb_requested = has(studiocast::video::effects::contract::
                                      kEffectIdVirtualBackgroundBlur) ||
                              has(studiocast::video::effects::contract::
                                      kEffectIdVirtualBackgroundRemove) ||
                              has(studiocast::video::effects::contract::
                                      kEffectIdVirtualBackgroundReplace);

    // Video Noise Removal (Maxine VFX).
    if (has(studiocast::video::effects::contract::kEffectIdVideoNoiseRemoval)) {
      const std::string stage_id = std::string(
          studiocast::video::effects::contract::kEffectIdVideoNoiseRemoval);

      if (engine_open_cuda) {
        // Open-source: prefer Open Video FastDVDnet when installed, otherwise
        // fall back to the lightweight Open CUDA temporal denoiser.
        want_open_video_video_denoise = true;

        std::string ov_err;
        if (open_video_fastdvdnet.EnsureInitialized(
                capA.width, capA.height, fx.video_noise_removal.model_id,
                &ov_err)) {
          have_open_video_video_denoise = true;
          set_backend(stage_id, "open_video");
          if (!note.empty())
            note += "\n";
          note += "Open Video: Video Noise Removal (FastDVDnet).";
          append_open_video_denoise_tensor_note();
        } else {
          want_open_cuda_video_denoise = true;
          have_open_cuda_video_denoise = true;
          set_backend(stage_id, "open_cuda");
          if (!note.empty())
            note += "\n";
          note += "Open CUDA: Video Noise Removal (temporal denoise).";
          if (!ov_err.empty()) {
            note += "\n";
            note += ov_err;
          }
        }
      } else if (!maxine_strict_blocked) {
        // AUTO / MAXINE: prefer Maxine VFX when available.
        want_maxine_denoise = true;

        std::string mx_err;
        if (maxine_denoise.EnsureInitialized(capA.width, capA.height, fx,
                                             &mx_err)) {
          have_maxine_denoise = true;
          set_backend(stage_id, "maxine");
          if (!note.empty())
            note += "\n";
          note += "Maxine VFX: Video Noise Removal.";
        } else {
          if (engine_maxine) {
            append_canonical_maxine_blocked(
                studiocast::maxine::MaxineNeed::vfx);
            maxine_strict_blocked = true;
          } else {
            // AUTO: prefer Open Video FastDVDnet when installed; fall back to
            // the lightweight Open CUDA temporal denoiser.
            want_open_video_video_denoise = true;
            std::string ov_err;
            if (open_video_fastdvdnet.EnsureInitialized(
                    capA.width, capA.height, fx.video_noise_removal.model_id,
                    &ov_err)) {
              have_open_video_video_denoise = true;
              set_backend(stage_id, "open_video");
              if (!note.empty())
                note += "\n";
              note += "Open Video: Video Noise Removal (FastDVDnet) — Maxine "
                      "VFX unavailable.";
              append_open_video_denoise_tensor_note();
              if (!mx_err.empty()) {
                note += "\n";
                note += mx_err;
              }
            } else {
              want_open_cuda_video_denoise = true;
              have_open_cuda_video_denoise = true;
              set_backend(stage_id, "open_cuda");
              if (!note.empty())
                note += "\n";
              note += "Open CUDA: Video Noise Removal (temporal denoise) — "
                      "Maxine VFX unavailable.";
              if (!mx_err.empty()) {
                note += "\n";
                note += mx_err;
              }
              if (!ov_err.empty()) {
                note += "\n";
                note += ov_err;
              }
            }
          }
        }
      }
    }

    // If Open CUDA denoise isn't active, reset the temporal state so a future
    // enable/fallback doesn't blend against stale frames.
    if (!have_open_cuda_video_denoise) {
      open_cuda_video_denoise.last_enabled = false;
    }

    // Likewise, if Open Video FastDVDnet isn't active, reset its temporal
    // history.
    if (!have_open_video_video_denoise) {
      open_video_fastdvdnet.ResetTemporalState();
    }

    // Eye Contact (AR)
    if (has(studiocast::video::effects::contract::kEffectIdEyeContact)) {
      const std::string stage_id = std::string(
          studiocast::video::effects::contract::kEffectIdEyeContact);
      const bool engine_auto = !engine_maxine && !engine_open_cuda;

      auto detach_vignette_from_eye_contact = [&] {
        if (plan.vignette_attach_to_effect_id == stage_id) {
          plan.vignette_attach_to_effect_id.clear();
        }
      };

      auto try_open_video = [&](std::string *out_err) -> bool {
        want_open_video_eye_contact = true;
        std::string ov_err;
        if (open_video_eye_contact.EnsureInitialized(fx.eye_contact.model_id,
                                                     &ov_err)) {
          have_open_video_eye_contact = true;
          set_backend(stage_id, "open_video");
          detach_vignette_from_eye_contact();
          return true;
        }
        if (out_err)
          *out_err = ov_err;
        return false;
      };

      if (engine_open_cuda) {
        std::string ov_err;
        if (try_open_video(&ov_err)) {
          if (!note.empty())
            note += "\n";
          note += "Open Video: Eye Contact (gaze correction).";
          append_open_video_eye_contact_note();
        } else {
          if (!note.empty())
            note += "\n";
          note += "Open Video: Eye Contact unavailable: " + ov_err;
          remove_stage_from_plan(stage_id);
        }
      } else {
        // Maxine preferred when allowed; Open Video fallback in AUTO.
        std::string mx_err;
        bool maxine_ok = false;

        if (!maxine_strict_blocked) {
          want_maxine_eye_contact = true;
          maxine_ok = maxine_eye_contact.EnsureInitialized(
              capA.width, capA.height, fx, &mx_err);
        }

        if (maxine_ok) {
          have_maxine_eye_contact = true;
          set_backend(stage_id, "maxine_ar");
          if (!note.empty())
            note += "\n";
          note += "Maxine AR: Eye Contact.";
        } else if (engine_auto) {
          std::string ov_err;
          if (try_open_video(&ov_err)) {
            if (!note.empty())
              note += "\n";
            note += "Open Video: Eye Contact (gaze correction) — Maxine AR "
                    "unavailable.";
            append_open_video_eye_contact_note();
            if (!mx_err.empty()) {
              note += "\n";
              note += mx_err;
            }
          } else {
            if (!note.empty())
              note += "\n";
            if (!mx_err.empty()) {
              note += mx_err;
              note += "\n";
            }
            note += "Open Video: Eye Contact unavailable: " + ov_err;
            remove_stage_from_plan(stage_id);
          }
        } else {
          // MAXINE engine: no open-source fallback.
          if (!note.empty())
            note += "\n";
          note += mx_err;
          append_canonical_maxine_blocked(studiocast::maxine::MaxineNeed::ar);
          maxine_strict_blocked = true;
          remove_stage_from_plan(stage_id);
        }
      }
    }

    // Prepare the first production Open CUDA HAL route before the frame loop.
    // This deliberately resolves and validates the model without constructing
    // the legacy OpenCudaVirtualBackgroundContext. The facade owns CUDA,
    // stream, ORT session, device buffers, execution and shutdown.
    auto try_prepare_open_cuda_virtual_background =
        [&](hal::SemanticEffectId semantic, std::string *setup_error) {
          prepared_open_cuda_setup_attempted = true;
          if (setup_error)
            setup_error->clear();
          const auto retire_unmatched_prepared = [&] {
            if (prepared_open_cuda_hardware_owner.Snapshot().prepared) {
              prepared_open_cuda_hardware_owner.Invalidate(
                  hal::ReasonId::reconfiguration_required);
              publish_prepared_open_cuda_status();
            }
            prepared_open_cuda_hardware_live = false;
          };

          PreparedOpenCudaHardwareRequest request;
          request.effects_engine =
              fx.engine == effects::EffectsEnginePreference::open_cuda
                  ? PreparedOpenCudaEffectsEngineIntent::open_cuda
                  : (fx.engine == effects::EffectsEnginePreference::maxine
                         ? PreparedOpenCudaEffectsEngineIntent::maxine
                         : PreparedOpenCudaEffectsEngineIntent::automatic);
          switch (cfg.compute_backend) {
          case ComputeBackendPreference::cpu:
            request.compute = PreparedOpenCudaComputeIntent::cpu;
            break;
          case ComputeBackendPreference::cuda:
            request.compute = PreparedOpenCudaComputeIntent::cuda;
            break;
          case ComputeBackendPreference::vulkan:
            request.compute = PreparedOpenCudaComputeIntent::vulkan;
            break;
          case ComputeBackendPreference::auto_select:
            request.compute = PreparedOpenCudaComputeIntent::automatic;
            break;
          }
          request.requested_backend = hal::BackendId::cuda;
          request.strict_backend_selection = true;
          request.requested_provider =
              hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
          request.strict_provider_selection = true;
          request.tensorrt_enabled = OpenCudaTensorRtEnabledFromEnv();
          request.stage_id = static_cast<std::uint32_t>(semantic);
          request.semantic = semantic;
          request.semantic_stage_count = plan.ordered_effect_ids.size();
          request.required = false;
          request.single_resident_island = true;
          request.contains_other_execution_domain = false;
          request.contains_other_resident_stage =
              plan.ordered_effect_ids.size() != 1;
          request.request_virtual_background_replace =
              fx.virtual_background.mode ==
              effects::VirtualBackgroundMode::replace;
          request.request_vignette =
              has(effects::contract::kEffectIdVignette);
          request.request_video_noise_removal =
              has(effects::contract::kEffectIdVideoNoiseRemoval);
          request.request_eye_contact =
              has(effects::contract::kEffectIdEyeContact);
          request.request_virtual_key_light =
              has(effects::contract::kEffectIdVirtualKeyLight);
          request.request_auto_frame =
              has(effects::contract::kEffectIdAutoFrame);
          request.request_mirror =
              has(effects::contract::kEffectIdMirror);
          request.shared_matte_required =
              request.request_virtual_key_light ||
              request.request_auto_frame;
          request.temporal_history_required =
              fx.virtual_background.greenscreen_temporal;
          request.capture_width = static_cast<std::uint32_t>(capA.width);
          request.capture_height = static_cast<std::uint32_t>(capA.height);
          request.output_width = static_cast<std::uint32_t>(outA.width);
          request.output_height = static_cast<std::uint32_t>(outA.height);
          const std::size_t input_stride =
              static_cast<std::size_t>(capA.width) * 3u;
          const std::size_t input_bytes =
              input_stride * static_cast<std::size_t>(capA.height);
          const std::size_t output_stride =
              static_cast<std::size_t>(outA.width) * 3u;
          const std::size_t output_bytes =
              output_stride * static_cast<std::size_t>(outA.height);
          request.host_input_is_tight_rgb24 =
              rgbStride == input_stride && rgb.size() == input_bytes;
          request.host_output_is_tight_rgb24 =
              outA.format == PixelFormat::rgb24 &&
              outA.bytes_per_line == output_stride &&
              outA.size_image == output_bytes;
          request.scaling_required =
              capA.width != outA.width || capA.height != outA.height;
          request.output_format_conversion_required =
              outA.format != PixelFormat::rgb24;
          request.generations.configuration = effects_generation;
          request.generations.geometry =
              prepared_hardware_geometry_generation;
          request.generations.provider = 1;
          request.generations.device = 1;
          request.cuda_device_id = 0;
          request.strength = fx.virtual_background.strength;
          std::uint32_t remove_rgb = 0;
          (void)ParseRgbHex(fx.virtual_background.remove_color, &remove_rgb);
          request.remove_background_rgb = {
              static_cast<std::uint8_t>((remove_rgb >> 16u) & 0xffu),
              static_cast<std::uint8_t>((remove_rgb >> 8u) & 0xffu),
              static_cast<std::uint8_t>(remove_rgb & 0xffu)};
          request.wait.timeout_nanoseconds = UINT64_C(500000000);
          request.wait.poll_interval_nanoseconds = UINT64_C(1000000);
          request.wait.maximum_poll_count = 500;
          request.wait.stop_requested =
              [](void *context) noexcept {
                return static_cast<std::atomic_bool *>(context)->load(
                    std::memory_order_relaxed);
              };
          request.wait.cancel_requested = request.wait.stop_requested;
          request.wait.now_nanoseconds =
              [](void *) noexcept -> std::uint64_t {
            return static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
          };
          request.wait.poll_wait =
              [](void *, std::uint64_t nanoseconds) noexcept {
            std::this_thread::sleep_for(
                std::chrono::nanoseconds(nanoseconds));
          };
          request.wait.callback_context = &stop_;
          request.declares_writer_copy_cpu_tail = true;

          // Cheap structural rejection precedes model discovery and hashing.
          PreparedOpenCudaHardwareRequest structural = request;
          structural.generations.model = 1;
          structural.model_pack.registry_generation = 1;
          structural.model_pack.task = "matting";
          structural.model_pack.matting =
              studiocast::open_video::MattingSpec{};
          structural.model_pack.matting->input = {
              "input", "nchw", "float32", 1, 1, 3};
          structural.model_pack.matting->output = {
              "alpha", "alpha", "float32", "", 0, 0, 0};
          structural.model_pack.matting->preprocess.color = "rgb";
          structural.model_pack.matting->preprocess.range = "0..1";
          structural.model_pack.files.push_back({});
          structural.model_artifact.registry_generation = 1;
          structural.model_artifact.model_id_sha256[0] = 1;
          structural.model_artifact.manifest_sha256[0] = 1;
          structural.model_artifact.artifact_sha256[0] = 1;
          structural.model_artifact.artifact_set_sha256[0] = 1;
          structural.model_pack.registry_manifest_sha256 =
              structural.model_artifact.manifest_sha256;
          structural.capacity = {1, 1, input_bytes, output_bytes, 1, 1};
          const auto structural_eligibility =
              EvaluatePreparedOpenCudaHardwareEligibility(structural);
          if (structural_eligibility !=
              PreparedOpenCudaHardwareEligibility::eligible) {
            retire_unmatched_prepared();
            if (setup_error)
              *setup_error =
                  "Prepared Open CUDA HAL route is ineligible for this "
                  "configuration.";
            return false;
          }

          const auto registry =
              studiocast::open_video::ModelPackRegistry::ScanDefault();
          std::string model_id = fx.virtual_background.model_id;
          if (model_id.empty())
            model_id = registry.DefaultModelIdForTask("matting");
          const auto pack = registry.Find("matting", model_id);
          if (!pack.has_value()) {
            retire_unmatched_prepared();
            if (setup_error)
              *setup_error =
                  "Prepared Open CUDA HAL model pack is unavailable.";
            return false;
          }
          const auto *artifact = FindPreparedOpenCudaMainOnnx(*pack);
          if (!artifact) {
            retire_unmatched_prepared();
            if (setup_error)
              *setup_error =
                  "Prepared Open CUDA HAL model has no ONNX artifact.";
            return false;
          }
          const auto validation =
              studiocast::open_video::ValidateModelArtifactIdentity(
                  *pack, *artifact);
          if (!validation.ok()) {
            retire_unmatched_prepared();
            if (setup_error) {
              *setup_error =
                  "Prepared Open CUDA HAL model artifact validation failed";
              if (!validation.detail.empty())
                *setup_error += ": " + validation.detail;
            }
            return false;
          }
          request.model_pack = *pack;
          request.model_artifact = validation.identity;
          request.generations.model = pack->registry_generation;
          const auto &matting = *pack->matting;
          if (!BuildPreparedOpenCudaPipelineResourceBudget(
                  request.capture_width, request.capture_height,
                  static_cast<std::uint32_t>(matting.input.width),
                  static_cast<std::uint32_t>(matting.input.height), semantic,
                  request.strength, &request.capacity)) {
            retire_unmatched_prepared();
            if (setup_error)
              *setup_error =
                  "Prepared Open CUDA HAL resource budget is invalid.";
            return false;
          }

          prepared_open_cuda_rgb_output.resize(output_bytes);
          const auto setup =
              prepared_open_cuda_hardware_owner.Configure(request);
          publish_prepared_open_cuda_status();
          if (setup.route != PreparedOpenCudaHardwareRoute::hal_live) {
            if (setup_error) {
              *setup_error =
                  "Prepared Open CUDA HAL setup failed [" +
                  std::string(PreparedHardwareReasonCode(setup.reason)) +
                  "].";
            }
            return false;
          }
          prepared_open_cuda_hardware_live = true;
          prepared_open_cuda_hardware_suspended = false;
          // A prior compatibility generation may have owned a legacy
          // CUDA/ORT session. It cannot coexist with the prepared island.
          open_cuda_vb.Destroy();
          return true;
        };

    // Background effects.
    if (vb_requested) {
      const auto vb_mode = fx.virtual_background.mode;

      std::optional<std::string_view> vb_effect_id;
      if (has(studiocast::video::effects::contract::
                  kEffectIdVirtualBackgroundBlur)) {
        vb_effect_id = studiocast::video::effects::contract::
            kEffectIdVirtualBackgroundBlur;
      } else if (has(studiocast::video::effects::contract::
                         kEffectIdVirtualBackgroundRemove)) {
        vb_effect_id = studiocast::video::effects::contract::
            kEffectIdVirtualBackgroundRemove;
      } else if (has(studiocast::video::effects::contract::
                         kEffectIdVirtualBackgroundReplace)) {
        vb_effect_id = studiocast::video::effects::contract::
            kEffectIdVirtualBackgroundReplace;
      }

      auto append_backend_note = [&](const char *backend) {
        if (!note.empty())
          note += "\n";
        if (vb_mode ==
            studiocast::video::effects::VirtualBackgroundMode::blur) {
          note += std::string(backend) + ": Virtual Background (blur).";
        } else if (vb_mode ==
                   studiocast::video::effects::VirtualBackgroundMode::remove) {
          note += std::string(backend) + ": Virtual Background (remove).";
        } else {
          note += std::string(backend) + ": Virtual Background (replace).";
        }
      };

      if (cfg.compute_backend == ComputeBackendPreference::vulkan) {
        want_open_vulkan_vb = true;
        std::string vk_err;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
        if (open_vulkan_vb.EnsureInitialized(capA.width, capA.height, fx,
                                             &vk_err)) {
          have_open_vulkan_vb = true;
          compute_available.vulkan_available = true;
          if (vb_effect_id.has_value())
            set_backend(*vb_effect_id, "open_vulkan");
          append_backend_note("Open Vulkan");
        } else
#else
        vk_err = "Open Vulkan: backend is disabled in this build. Rebuild "
                 "with -DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON.";
#endif
        {
#if STUDIOCAST_ENABLE_OPEN_VULKAN
          if (vb_mode ==
                  studiocast::video::effects::VirtualBackgroundMode::blur &&
              vk_err.find("[vulkan_virtual_background_blur_") ==
                  std::string::npos) {
            vk_err = OpenVulkanVirtualBackgroundBlurInitializationFailure(
                vk_err);
          } else if (
              vb_mode ==
                  studiocast::video::effects::VirtualBackgroundMode::remove &&
              vk_err.find("[vulkan_virtual_background_remove_") ==
                  std::string::npos) {
            vk_err = OpenVulkanVirtualBackgroundRemoveInitializationFailure(
                vk_err);
          } else if (
              vb_mode ==
                  studiocast::video::effects::VirtualBackgroundMode::replace &&
              vk_err.find("[vulkan_virtual_background_replace_") ==
                  std::string::npos) {
            vk_err = OpenVulkanVirtualBackgroundReplaceInitializationFailure(
                vk_err);
          }
#endif
          if (!vk_err.empty()) {
            compute_available.vulkan_unavailable_reason = vk_err;
            append_note(vk_err);
          }
          if (vb_effect_id.has_value())
            remove_stage_from_plan(*vb_effect_id);
        }
      } else if (fx.engine == studiocast::video::effects::
                                  EffectsEnginePreference::open_cuda) {
        want_open_cuda_vb = true;
        std::string oc_err;
        hal::SemanticEffectId prepared_semantic =
            hal::SemanticEffectId::invalid;
        if (vb_mode ==
            studiocast::video::effects::VirtualBackgroundMode::blur) {
          prepared_semantic =
              hal::SemanticEffectId::virtual_background_blur;
        } else if (
            vb_mode ==
            studiocast::video::effects::VirtualBackgroundMode::remove) {
          prepared_semantic =
              hal::SemanticEffectId::virtual_background_remove;
        }
        const bool prepared_live =
            prepared_semantic != hal::SemanticEffectId::invalid &&
            try_prepare_open_cuda_virtual_background(prepared_semantic,
                                                     &oc_err);
        if (prepared_live) {
          have_open_cuda_vb = true;
          if (vb_effect_id.has_value())
            set_backend(*vb_effect_id, "open_cuda");
          append_backend_note("Open CUDA HAL");
        } else if (open_cuda_vb.EnsureInitialized(
                       capA.width, capA.height, fx, &oc_err)) {
          have_open_cuda_vb = true;
          if (vb_effect_id.has_value())
            set_backend(*vb_effect_id, "open_cuda");
          append_backend_note("Open CUDA compatibility");
        } else {
          if (!note.empty())
            note += "\n";
          note += oc_err;
          if (vb_effect_id.has_value())
            remove_stage_from_plan(*vb_effect_id);
        }
      } else if (fx.engine ==
                 studiocast::video::effects::EffectsEnginePreference::maxine) {
        if (!maxine_strict_blocked) {
          want_maxine_bg_blur = true;
          std::string mx_err;
          if (maxine_bg_blur.EnsureInitialized(capA.width, capA.height, fx,
                                               &mx_err)) {
            have_maxine_bg_blur = true;
            if (vb_effect_id.has_value())
              set_backend(*vb_effect_id, "maxine");
            append_backend_note("Maxine VFX");
          } else {
            append_canonical_maxine_blocked(
                studiocast::maxine::MaxineNeed::vfx);
            maxine_strict_blocked = true;
            if (vb_effect_id.has_value())
              remove_stage_from_plan(*vb_effect_id);
          }
        } else {
          if (vb_effect_id.has_value())
            remove_stage_from_plan(*vb_effect_id);
        }
      } else {
        // auto_select: prefer Maxine when available; otherwise try Open CUDA.
        want_maxine_bg_blur = true;
        std::string mx_err;
        if (maxine_bg_blur.EnsureInitialized(capA.width, capA.height, fx,
                                             &mx_err)) {
          have_maxine_bg_blur = true;
          if (vb_effect_id.has_value())
            set_backend(*vb_effect_id, "maxine");
          append_backend_note("Maxine VFX");
        } else {
          want_open_cuda_vb = true;
          std::string oc_err;
          if (open_cuda_vb.EnsureInitialized(capA.width, capA.height, fx,
                                             &oc_err)) {
            have_open_cuda_vb = true;
            if (vb_effect_id.has_value())
              set_backend(*vb_effect_id, "open_cuda");
            append_backend_note("Open CUDA");
          } else {
            if (!note.empty())
              note += "\n";
            note += mx_err;
            if (!note.empty())
              note += "\n";
            note += oc_err;
            if (vb_effect_id.has_value())
              remove_stage_from_plan(*vb_effect_id);
          }
        }
      }

      // Prevent silent vignette skip: vignette attachment is only implemented
      // on Maxine/CUDA(BGR) paths today.
      if (have_open_cuda_vb &&
          has(studiocast::video::effects::contract::kEffectIdVignette)) {
        const std::string &attach = plan.vignette_attach_to_effect_id;
        if (attach == std::string(studiocast::video::effects::contract::
                                      kEffectIdVirtualBackgroundBlur) ||
            attach == std::string(studiocast::video::effects::contract::
                                      kEffectIdVirtualBackgroundRemove) ||
            attach == std::string(studiocast::video::effects::contract::
                                      kEffectIdVirtualBackgroundReplace)) {
          if (!note.empty())
            note += "\n";
          note += "Vignette unavailable: Open CUDA virtual background does not "
                  "support vignette yet.";
          remove_stage_from_plan(
              studiocast::video::effects::contract::kEffectIdVignette);
          plan.vignette_attach_to_effect_id.clear();
        }
      }
    }

    // Auto Frame.
    if (has(studiocast::video::effects::contract::kEffectIdAutoFrame)) {
      const std::string stage_id =
          std::string(studiocast::video::effects::contract::kEffectIdAutoFrame);

      // Vignette attachment is only supported on the Maxine GPU paths. If we
      // end up using the Open CUDA auto-frame path (or auto-frame is
      // unavailable), we run vignette as a standalone stage by clearing the
      // attachment.
      const auto detach_vignette_from_auto_frame = [&]() {
        if (has(studiocast::video::effects::contract::kEffectIdVignette) &&
            plan.vignette_attach_to_effect_id == stage_id) {
          plan.vignette_attach_to_effect_id.clear();
        }
      };

      bool auto_frame_handled_by_vulkan_request = false;
      if (cfg.compute_backend == ComputeBackendPreference::vulkan) {
        auto_frame_handled_by_vulkan_request = true;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
        want_open_vulkan_auto_frame = true;

        bool have_open_video_face_detection = false;
        std::string fd_err;
        if (open_video_yunet.EnsureInitialized(
                fx.auto_frame.model_id, &fd_err,
                studiocast::open_video::YunetProviderPolicy::cpu_only)) {
          have_open_video_face_detection = true;
        }

        std::string vk_err;
        const bool require_matte_tracking = !have_open_video_face_detection;
        const std::uint64_t auto_frame_init_failures_before =
            open_vulkan_auto_frame.counters.initialization_failures;
        if (open_vulkan_auto_frame.EnsureInitialized(
                capA.width, capA.height, fx,
                /*require_matte_tracking=*/require_matte_tracking,
                effects_generation, open_video_yunet.active_model_id(),
                &vk_err)) {
          have_open_vulkan_auto_frame = true;
          compute_available.vulkan_available = true;
          set_backend(stage_id, "open_vulkan");
          if (!note.empty()) {
            note += "\n";
          }
          if (have_open_video_face_detection) {
            note += "[vulkan_effect_cpu_tail] Open Vulkan Auto Frame is "
                    "usable with degraded behavior: "
                    "[vulkan_auto_frame_cpu_face_tracking_tail] CPU-only "
                    "YuNet tracking on the original host capture; "
                    "[vulkan_auto_frame_cpu_crop_plan_smoothing_tail] CPU "
                    "crop planning/smoothing; Vulkan crop/scale remains on "
                    "the shared resident frame.";
            append_open_video_face_detection_note();
          } else {
            note += "[vulkan_effect_cpu_tail] Open Vulkan Auto Frame is "
                    "usable with degraded behavior: "
                    "[vulkan_auto_frame_matte_alpha_readback_cpu_box_tail] "
                    "resident matte with explicit periodic alpha readback and "
                    "CPU box extraction; "
                    "[vulkan_auto_frame_cpu_crop_plan_smoothing_tail] CPU "
                    "crop planning/smoothing; Vulkan crop/scale.";
            if (!fd_err.empty()) {
              note += "\n";
              note += OpenVulkanAutoFrameFaceProviderFailure(fd_err);
            }
          }
          detach_vignette_from_auto_frame();
        } else {
          if (open_vulkan_auto_frame.counters.initialization_failures ==
              auto_frame_init_failures_before) {
            ++open_vulkan_auto_frame.counters.initialization_failures;
          }
          if (!note.empty()) {
            note += "\n";
          }
          if (!fd_err.empty() && !have_open_video_face_detection) {
            note += OpenVulkanAutoFrameInitializationFailure(
                OpenVulkanAutoFrameFaceProviderFailure(fd_err));
            note += "\n";
          }
          note += vk_err;
          compute_available.vulkan_unavailable_reason = vk_err;
          remove_stage_from_plan(stage_id);
          detach_vignette_from_auto_frame();
        }
#else
        const std::string vk_err =
            "[vulkan_effect_initialization_failed] Open Vulkan Auto Frame "
            "initialization failed: [vulkan_backend_disabled] backend is "
            "disabled in this build; rebuild with "
            "-DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON.";
        append_note(vk_err);
        compute_available.vulkan_unavailable_reason = vk_err;
        remove_stage_from_plan(stage_id);
        detach_vignette_from_auto_frame();
#endif
      }

      if (!auto_frame_handled_by_vulkan_request && engine_open_cuda) {
        // Open source: prefer Open Video face tracking when available,
        // otherwise fall back to the foreground matte path.
        want_open_cuda_auto_frame = true;

        bool have_open_video_face_detection = false;
        std::string fd_err;
        if (open_video_yunet.EnsureInitialized(fx.auto_frame.model_id,
                                               &fd_err)) {
          have_open_video_face_detection = true;
        }

        std::string oc_err;
        const bool require_matte_tracking = !have_open_video_face_detection;
        if (open_cuda_auto_frame.EnsureInitialized(
                capA.width, capA.height, fx,
                /*require_matte_tracking=*/require_matte_tracking, &oc_err)) {
          have_open_cuda_auto_frame = true;
          set_backend(stage_id, "open_cuda");
          if (!note.empty()) {
            note += "\n";
          }
          if (have_open_video_face_detection) {
            note += "Open Video: Auto Frame (CPU face tracking; GPU "
                    "crop/scale when reusing an active Open CUDA frame, CPU "
                    "crop/scale fallback).";
            append_open_video_face_detection_note();
          } else {
            note += "Open CUDA: Auto Frame (foreground matte tracking; GPU "
                    "crop/scale when reusing an active Open CUDA frame, CPU "
                    "crop/scale fallback).";
          }
          detach_vignette_from_auto_frame();
        } else {
          if (!note.empty()) {
            note += "\n";
          }
          if (!fd_err.empty() && !have_open_video_face_detection) {
            note += fd_err;
            if (!note.empty())
              note += "\n";
          }
          note += oc_err;
          remove_stage_from_plan(stage_id);
          detach_vignette_from_auto_frame();
        }
      } else if (!auto_frame_handled_by_vulkan_request &&
                 !maxine_strict_blocked) {
        // Maxine AR preferred; Open CUDA fallback when engine_preference=AUTO.
        want_maxine_auto_frame = true;
        std::string mx_err;
        if (maxine_auto_frame.EnsureInitialized(capA.width, capA.height, fx,
                                                &mx_err)) {
          have_maxine_auto_frame = true;
          set_backend(stage_id, "maxine_ar_cuda");
        } else if (engine_maxine) {
          if (!note.empty()) {
            note += "\n";
          }
          note += mx_err;
          note +=
              "\nMaxine backend required due to engine_preference=maxine.\n";
          append_canonical_maxine_blocked(studiocast::maxine::MaxineNeed::ar);
          maxine_strict_blocked = true;

          // Auto-frame isn't available; don't leave vignette attached to a
          // non-existent GPU stage.
          detach_vignette_from_auto_frame();
        } else {
          // AUTO: fall back to Open CUDA.
          want_open_cuda_auto_frame = true;
          bool have_open_video_face_detection = false;
          std::string fd_err;
          if (open_video_yunet.EnsureInitialized(fx.auto_frame.model_id,
                                                 &fd_err)) {
            have_open_video_face_detection = true;
          }

          std::string oc_err;
          const bool require_matte_tracking = !have_open_video_face_detection;
          if (open_cuda_auto_frame.EnsureInitialized(
                  capA.width, capA.height, fx,
                  /*require_matte_tracking=*/require_matte_tracking, &oc_err)) {
            have_open_cuda_auto_frame = true;
            set_backend(stage_id, "open_cuda");
            if (!note.empty()) {
              note += "\n";
            }
            note += "Maxine AR unavailable; ";
            if (have_open_video_face_detection) {
              note += "Open Video: Auto Frame (CPU face tracking; GPU "
                      "crop/scale when reusing an active Open CUDA frame, CPU "
                      "crop/scale fallback).";
              append_open_video_face_detection_note();
            } else {
              note += "Open CUDA: Auto Frame (foreground matte tracking; GPU "
                      "crop/scale when reusing an active Open CUDA frame, CPU "
                      "crop/scale fallback).";
            }
            detach_vignette_from_auto_frame();
          } else {
            if (!note.empty()) {
              note += "\n";
            }
            note += mx_err;
            if (!fd_err.empty() && !have_open_video_face_detection) {
              if (!note.empty())
                note += "\n";
              note += fd_err;
            }
            if (!note.empty()) {
              note += "\n";
            }
            note += oc_err;

            remove_stage_from_plan(stage_id);
            detach_vignette_from_auto_frame();
          }
        }
      }

      // If Auto Frame couldn't be initialized on any backend, make sure
      // vignette isn't left attached.
      if (!have_maxine_auto_frame && !have_open_cuda_auto_frame &&
          !have_open_vulkan_auto_frame) {
        detach_vignette_from_auto_frame();
      }
    }

    // Virtual Key Light.
    if (has(studiocast::video::effects::contract::kEffectIdVirtualKeyLight)) {
      const std::string stage_id = std::string(
          studiocast::video::effects::contract::kEffectIdVirtualKeyLight);

      // Vignette attachment is only supported on the Maxine GPU paths. If we
      // end up using the Open CUDA key light path (or key light is
      // unavailable), run vignette as a standalone stage by clearing the
      // attachment.
      const auto detach_vignette_from_key_light = [&]() {
        if (has(studiocast::video::effects::contract::kEffectIdVignette) &&
            plan.vignette_attach_to_effect_id == stage_id) {
          plan.vignette_attach_to_effect_id.clear();
        }
      };

      bool key_light_handled_by_vulkan_request = false;
      if (cfg.compute_backend == ComputeBackendPreference::vulkan) {
        key_light_handled_by_vulkan_request = true;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
        want_open_vulkan_key_light = true;
        std::string vk_err;
        if (open_vulkan_key_light.EnsureInitialized(capA.width, capA.height, fx,
                                                    &vk_err)) {
          have_open_vulkan_key_light = true;
          compute_available.vulkan_available = true;
          set_backend(stage_id, "open_vulkan");
          if (!note.empty()) {
            note += "\n";
          }
          note += "Open Vulkan: Virtual Key Light (shared Vulkan matte; GPU "
                  "foreground relight while the frame is Vulkan-resident).";
          detach_vignette_from_key_light();
        } else {
          if (vk_err.find("[vulkan_virtual_key_light_initialization_failed]") ==
              std::string::npos) {
            vk_err = OpenVulkanVirtualKeyLightInitializationFailure(vk_err);
          }
          if (!note.empty()) {
            note += "\n";
          }
          note += vk_err;
          compute_available.vulkan_unavailable_reason = vk_err;
          remove_stage_from_plan(stage_id);
          detach_vignette_from_key_light();
        }
#else
        const std::string vk_err =
            "[vulkan_virtual_key_light_initialization_failed] Open Vulkan "
            "virtual key light initialization failed: "
            "[vulkan_backend_disabled_in_build] Open Vulkan support is "
            "disabled in this build.";
        append_note(vk_err);
        compute_available.vulkan_unavailable_reason = vk_err;
        remove_stage_from_plan(stage_id);
        detach_vignette_from_key_light();
#endif
      }

      if (!key_light_handled_by_vulkan_request && engine_open_cuda) {
        // Open CUDA: key light via shared matting + masked lift.
        want_open_cuda_key_light = true;
        std::string oc_err;
        if (open_cuda_key_light.EnsureInitialized(capA.width, capA.height, fx,
                                                  &oc_err)) {
          have_open_cuda_key_light = true;
          set_backend(stage_id, "open_cuda");
          if (!note.empty()) {
            note += "\n";
          }
          note += "Open CUDA: Virtual Key Light (GPU blend when reusing an "
                  "active Open CUDA frame; foreground matte + CPU relight "
                  "fallback).";
          detach_vignette_from_key_light();
        } else {
          if (!note.empty()) {
            note += "\n";
          }
          note += oc_err;
          remove_stage_from_plan(stage_id);
          detach_vignette_from_key_light();
        }
      } else if (!key_light_handled_by_vulkan_request &&
                 !maxine_strict_blocked) {
        // Maxine VFX preferred; Open CUDA fallback when engine_preference=AUTO.
        want_maxine_relight = true;
        std::string mx_err;
        if (maxine_relight.EnsureInitialized(capA.width, capA.height, fx,
                                             &mx_err)) {
          have_maxine_relight = true;
          set_backend(stage_id, "maxine");
          if (!note.empty())
            note += " ";
          note += "Maxine VFX: Video Relighting (Virtual Key Light).";
          if (have_maxine_bg_blur) {
            note +=
                "\nMaxine VFX: sharing Green Screen matte between Virtual "
                "Background and Virtual Key Light when frame/config "
                "compatibility is clear; incompatible frames fall back to a "
                "separate Green Screen pass and are counted. RGB "
                "staging/readback remains per stage because the current "
                "Maxine VFX stages own separate output images and CPU "
                "continuation points; tracked input handles that cannot "
                "represent the current RGB continuation are counted.";
          }
        } else if (engine_maxine) {
          append_canonical_maxine_blocked(studiocast::maxine::MaxineNeed::vfx);
          maxine_strict_blocked = true;
          detach_vignette_from_key_light();
        } else {
          // AUTO: fall back to Open CUDA.
          want_open_cuda_key_light = true;
          std::string oc_err;
          if (open_cuda_key_light.EnsureInitialized(capA.width, capA.height, fx,
                                                    &oc_err)) {
            have_open_cuda_key_light = true;
            set_backend(stage_id, "open_cuda");
            if (!note.empty()) {
              note += "\n";
            }
            note += "Open CUDA: Virtual Key Light (GPU blend when reusing an "
                    "active Open CUDA frame; foreground matte + CPU relight "
                    "fallback).";
            detach_vignette_from_key_light();
          } else {
            if (!note.empty())
              note += " ";
            note += mx_err;
            if (!note.empty()) {
              note += "\n";
            }
            note += oc_err;

            remove_stage_from_plan(stage_id);
            detach_vignette_from_key_light();
          }
        }
      }

      // If key light couldn't be initialized on any backend, make sure vignette
      // isn't left attached.
      if (!have_maxine_relight && !have_open_cuda_key_light &&
          !have_open_vulkan_key_light) {
        detach_vignette_from_key_light();
      }
    }

    // The default-true tracked-center field has observable semantics on the
    // Auto Frame-attached CUDA path. Apply this backend compatibility rule only
    // after unavailable Auto Frame stages have been removed. Standalone
    // vignette therefore remains fixed-center production behavior, while the
    // combined tracked request fails closed without CPU tracking/mask work.
#if STUDIOCAST_ENABLE_OPEN_VULKAN
    if (cfg.compute_backend == ComputeBackendPreference::vulkan) {
      const auto compatibility =
          ApplyOpenVulkanVignettePlanCompatibility(fx, &plan);
      if (compatibility.blocked) {
        planned.erase(std::string(
            studiocast::video::effects::contract::kEffectIdVignette));
        want_open_vulkan_vignette = true;
        open_vulkan_vignette.Shutdown();
      }
    }
#endif

    // First bounded live HAL migration: one adapter owns the complete final
    // resident island, and only the already-qualified fixed-center vignette
    // and mirror semantics may enter it. Any other stage, tracked-center
    // semantics, or non-tight RGB writer layout stays on the retained path.
    {
      const std::string vignette_id(
          studiocast::video::effects::contract::kEffectIdVignette);
      const std::string mirror_id(
          studiocast::video::effects::contract::kEffectIdMirror);
      const bool contains_unqualified_stage = std::any_of(
          plan.ordered_effect_ids.begin(), plan.ordered_effect_ids.end(),
          [&](const std::string &id) {
            return id != vignette_id && id != mirror_id;
          });
      const std::size_t tight_output_bytes =
          static_cast<std::size_t>(outA.width) *
          static_cast<std::size_t>(outA.height) * 3u;
      const std::size_t tight_input_stride =
          static_cast<std::size_t>(capA.width) * 3u;
      const std::size_t tight_input_bytes =
          tight_input_stride * static_cast<std::size_t>(capA.height);
      const bool exact_vulkan_request =
          cfg.compute_backend == ComputeBackendPreference::vulkan &&
          !contains_unqualified_stage && (has(vignette_id) || has(mirror_id)) &&
          !(has(vignette_id) && fx.vignette.center_on_tracked_face) &&
          rgbStride == tight_input_stride && rgb.size() == tight_input_bytes &&
          outA.format == PixelFormat::rgb24 &&
          outA.bytes_per_line ==
              static_cast<std::size_t>(outA.width) * 3u &&
          outA.size_image == tight_output_bytes;
      VideoExecutionPlanRequest request;
      request.plan_id = prepared_hardware_next_plan_id++;
      if (prepared_hardware_next_plan_id == 0)
        prepared_hardware_next_plan_id = 1;
      request.intent.backend.mode =
          hal::BackendRequestMode::explicit_backend;
      request.intent.backend.backend = hal::BackendId::vulkan;
      request.intent.backend.strict = true;
      request.intent.preferences.compute_family =
          hal::ComputeFamilyAllowance::vulkan_only;
      request.intent.requested_provider =
          hal::SemanticEffectProviderId::open_vulkan_production_kernels;
      request.intent.strict_provider_selection = true;
      const auto add_final_stage =
          [&](hal::SemanticEffectId semantic) -> CanonicalVideoStage & {
        auto &stage = request.intent.stages[request.intent.stage_count++];
        stage.intent.stage_id = static_cast<std::uint32_t>(semantic);
        stage.intent.semantic = semantic;
        stage.intent.required = false;
        stage.intent.failure_action =
            hal::FailureAction::fail_open_preserve_input;
        return stage;
      };
      if (has(vignette_id)) {
        auto &stage = add_final_stage(hal::SemanticEffectId::vignette);
        stage.parameters[0] = {CanonicalVideoParameterId::intensity,
                               CanonicalVideoParameterKind::signed_integer,
                               fx.vignette.intensity};
        stage.parameters[1].id =
            CanonicalVideoParameterId::center_on_tracked_face;
        stage.parameters[1].kind = CanonicalVideoParameterKind::boolean;
        stage.parameters[1].boolean_value =
            fx.vignette.center_on_tracked_face;
        stage.parameter_count = 2;
      }
      if (has(mirror_id))
        (void)add_final_stage(hal::SemanticEffectId::mirror);
      const auto host_image = [](std::uint32_t width, std::uint32_t height,
                                 hal::Mutability mutability) {
        hal::ImageDescriptor image;
        image.width = width;
        image.height = height;
        image.plane_count = 1;
        image.format = hal::PixelFormat::rgb_u8;
        image.channel_order = hal::ChannelOrder::rgb;
        image.color_space = hal::ColorSpace::srgb;
        image.color_range = hal::ColorRange::full;
        image.alpha_meaning = hal::AlphaMeaning::none;
        image.orientation = hal::Orientation::top_left;
        image.host_coherence = hal::HostCoherence::coherent;
        image.mutability = mutability;
        image.aliasing = hal::AliasingPermission::forbidden;
        image.in_place_permitted = false;
        image.base_alignment_bytes = 1;
        const auto stride = static_cast<std::uint64_t>(width) * 3u;
        image.planes[0].row_stride_bytes = stride;
        image.planes[0].size_bytes = stride * height;
        image.planes[0].width = width;
        image.planes[0].height = height;
        image.planes[0].alignment_bytes = 1;
        return image;
      };
      request.host_input =
          host_image(static_cast<std::uint32_t>(capA.width),
                     static_cast<std::uint32_t>(capA.height),
                     hal::Mutability::read_only);
      request.host_output =
          host_image(static_cast<std::uint32_t>(outA.width),
                     static_cast<std::uint32_t>(outA.height),
                     hal::Mutability::write_only);
      request.invalidation_key.generations.configuration = effects_generation;
      request.invalidation_key.generations.geometry =
          prepared_hardware_geometry_generation;
      request.invalidation_key.generations.model = 1;
      request.invalidation_key.generations.provider = 1;
      request.invalidation_key.generations.device = 1;
      std::uint64_t canonical_plan_hash = UINT64_C(1469598103934665603);
      const auto mix_plan_hash = [&](std::uint64_t value) {
        canonical_plan_hash ^= value;
        canonical_plan_hash *= UINT64_C(1099511628211);
      };
      mix_plan_hash(has(vignette_id) ? 1u : 0u);
      mix_plan_hash(has(mirror_id) ? 1u : 0u);
      mix_plan_hash(static_cast<std::uint64_t>(
          static_cast<std::int64_t>(fx.vignette.intensity)));
      mix_plan_hash(fx.vignette.center_on_tracked_face ? 1u : 0u);
      mix_plan_hash(static_cast<std::uint32_t>(capA.width));
      mix_plan_hash(static_cast<std::uint32_t>(capA.height));
      mix_plan_hash(static_cast<std::uint32_t>(outA.width));
      mix_plan_hash(static_cast<std::uint32_t>(outA.height));
      request.invalidation_key.canonical_plan_hash =
          canonical_plan_hash == 0 ? 1 : canonical_plan_hash;
      const auto input_bytes = request.host_input.planes[0].size_bytes;
      const auto output_bytes = request.host_output.planes[0].size_bytes;
      std::uint64_t mask_bytes = 0;
      bool budget_valid = outA.width > 0 && outA.height > 0;
      const auto checked_multiply =
          [](std::uint64_t left, std::uint64_t right, std::uint64_t *result) {
        if (!result ||
            (right != 0 &&
             left > std::numeric_limits<std::uint64_t>::max() / right)) {
          return false;
        }
        *result = left * right;
        return true;
      };
      const auto checked_add = [](std::uint64_t *value,
                                  std::uint64_t increment) {
        if (!value ||
            *value > std::numeric_limits<std::uint64_t>::max() - increment) {
          return false;
        }
        *value += increment;
        return true;
      };
      budget_valid &=
          checked_multiply(static_cast<std::uint64_t>(outA.width),
                           static_cast<std::uint64_t>(outA.height),
                           &mask_bytes) &&
          checked_multiply(mask_bytes, sizeof(float), &mask_bytes);
      request.budget.upload_staging_bytes = input_bytes;
      request.budget.readback_staging_bytes = output_bytes;
      request.budget.scratch_bytes = output_bytes;
      request.budget.persistent_bytes = input_bytes;
      request.budget.resource_count = 4;
      request.budget.in_flight_slots = 1;
      if (has(vignette_id)) {
        budget_valid &=
            checked_add(&request.budget.persistent_bytes, output_bytes);
        budget_valid &=
            checked_add(&request.budget.persistent_bytes, mask_bytes);
        request.budget.resource_count += 2;
      }
      if (has(mirror_id)) {
        budget_valid &=
            checked_add(&request.budget.persistent_bytes, output_bytes);
        ++request.budget.resource_count;
      }
      request.migration_policy = VideoExecutionMigrationPolicy::prepared_only;
      request.setup_failure_policy =
          hal::SetupFailurePolicy::retain_old_program;

      VideoExecutionPrepareEvidence setup;
      hal::ReasonId setup_reason = hal::ReasonId::none;
      const auto setup_status =
          exact_vulkan_request && budget_valid
              ? prepared_hardware_composition.Prepare(request, &setup,
                                                      &setup_reason)
              : hal::StatusId::blocked;
      if (!exact_vulkan_request || !budget_valid)
        prepared_hardware_composition.Invalidate(
            hal::ReasonId::reconfiguration_required);
      const auto composition_snapshot =
          prepared_hardware_composition.Snapshot();
      prepared_hardware_final_live =
          setup_status == hal::StatusId::ready ||
          setup_status == hal::StatusId::degraded ||
          (setup.old_plan_retained && composition_snapshot.prepared);
      prepared_hardware_runtime_suspended = false;
      if (prepared_hardware_final_live) {
        have_open_vulkan_vignette = has(vignette_id);
        want_open_vulkan_vignette = has(vignette_id);
        have_open_vulkan_mirror = has(mirror_id);
        want_open_vulkan_mirror = has(mirror_id);
        compute_available.vulkan_available = true;
        if (has(vignette_id))
          set_backend(vignette_id, "open_vulkan");
        if (has(mirror_id))
          set_backend(mirror_id, "open_vulkan");
        append_note(
            "Open Vulkan HAL: prepared final resident section owns one "
            "upload, fixed-center vignette/mirror execution, and one final "
            "readback.");
      }
      publish_prepared_hardware_status();
    }

    // Vignette final-output stage. Compatible explicit Vulkan requests run the
    // fixed-center device-resident implementation after output geometry and
    // before mirror. Other backends retain their existing attachment behavior.
    if (has(studiocast::video::effects::contract::kEffectIdVignette) &&
        cfg.compute_backend == ComputeBackendPreference::vulkan &&
        !prepared_hardware_final_live) {
      const std::string stage_id =
          std::string(studiocast::video::effects::contract::kEffectIdVignette);
      want_open_vulkan_vignette = true;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
      std::string vk_err;
      bool vignette_ready = open_vulkan_vignette.EnsureInitialized(
          &open_vulkan_vb.kernels, outA.width, outA.height,
          fx.vignette.intensity, &open_vulkan_vignette_counters, &vk_err);
      if (vignette_ready) {
        vignette_ready =
            open_vulkan_vb.EnsureImage(
                &open_vulkan_vb.final_upload_staging, capA.width, capA.height,
                studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                /*map_memory=*/true, "final upload staging", &vk_err) &&
            open_vulkan_vb.EnsureImage(
                &open_vulkan_vb.final_source_rgb, capA.width, capA.height,
                studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                /*map_memory=*/false, "final resident source", &vk_err) &&
            open_vulkan_vb.EnsureImage(
                &vulkan_rgb_scaled, outA.width, outA.height,
                studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                /*map_memory=*/false, "final resize scratch", &vk_err) &&
            open_vulkan_vb.EnsureImage(
                &open_vulkan_vb.vignette_rgb, outA.width, outA.height,
                studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                /*map_memory=*/false, "vignette output", &vk_err) &&
            open_vulkan_vb.EnsureImage(
                &open_vulkan_vb.final_readback_staging, outA.width, outA.height,
                studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                /*map_memory=*/true, "final readback staging", &vk_err) &&
            !open_vulkan_vignette.attenuation_factor_mask().mapped() &&
            open_vulkan_vignette.attenuation_factor_mask().device_local();
        vulkan_rgb_scaled_allocated = vulkan_rgb_scaled.Valid();
      }
      if (vignette_ready) {
        have_open_vulkan_vignette = true;
        compute_available.vulkan_available = true;
        set_backend(stage_id, "open_vulkan");
        append_note("Open Vulkan: Vignette (fixed center after framing; "
                    "non-mapped DEVICE_LOCAL source/scratch/output and "
                    "attenuation mask, bounded upload/final-readback staging, "
                    "no tracking CPU tail).");
      } else {
        if (vk_err.find("[vulkan_effect_initialization_failed]") ==
            std::string::npos) {
          vk_err = OpenVulkanVignetteInitializationFailure(vk_err);
        }
        append_note(vk_err);
        compute_available.vulkan_unavailable_reason = vk_err;
        remove_stage_from_plan(stage_id);
      }
#else
      const std::string vk_err =
          "[vulkan_effect_initialization_failed] Open Vulkan vignette "
          "initialization failed: backend is disabled in this build.";
      append_note(vk_err);
      compute_available.vulkan_unavailable_reason = vk_err;
      remove_stage_from_plan(stage_id);
#endif
    } else if (!prepared_hardware_final_live &&
               has(studiocast::video::effects::contract::kEffectIdVignette) &&
               plan.vignette_attach_to_effect_id.empty()) {
      const bool have_any_maxine_gpu_stage =
          have_maxine_eye_contact || have_maxine_bg_blur ||
          have_maxine_relight || have_maxine_auto_frame;

      if (!have_any_maxine_gpu_stage && cuda_video_compute_allowed) {
        want_maxine_vignette_only = true;
        std::string mx_err;
        if (vignette_only.EnsureInitialized(capA.width, capA.height, &mx_err)) {
          have_maxine_vignette_only = true;
          set_backend(studiocast::video::effects::contract::kEffectIdVignette,
                      "cuda");
          if (!note.empty())
            note += "\n";
          note += "CUDA: Vignette.";
        } else {
          if (!note.empty())
            note += "\n";
          note += "Vignette unavailable: " + mx_err;
        }
      } else if (!have_any_maxine_gpu_stage) {
        remove_stage_from_plan(
            studiocast::video::effects::contract::kEffectIdVignette);
      }
    }

    // Mirror is an explicit-Vulkan-only final output stage. It deliberately
    // does not enter the host EffectChain or any CUDA/Maxine path.
    if (has(studiocast::video::effects::contract::kEffectIdMirror)) {
      const std::string stage_id =
          std::string(studiocast::video::effects::contract::kEffectIdMirror);
      if (cfg.compute_backend == ComputeBackendPreference::vulkan &&
          !prepared_hardware_final_live) {
        want_open_vulkan_mirror = true;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
        std::string vk_err;
        bool mirror_ready = open_vulkan_mirror.EnsureInitialized(
            &open_vulkan_vb.kernels, outA.width, outA.height, &vk_err);
        if (mirror_ready) {
          mirror_ready =
              open_vulkan_vb.EnsureImage(
                  &open_vulkan_vb.final_upload_staging, capA.width, capA.height,
                  studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                  /*map_memory=*/true, "final upload staging", &vk_err) &&
              open_vulkan_vb.EnsureImage(
                  &open_vulkan_vb.final_source_rgb, capA.width, capA.height,
                  studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                  /*map_memory=*/false, "final resident source", &vk_err) &&
              open_vulkan_vb.EnsureImage(
                  &vulkan_rgb_scaled, outA.width, outA.height,
                  studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                  /*map_memory=*/false, "final resize scratch", &vk_err) &&
              open_vulkan_vb.EnsureImage(
                  &open_vulkan_vb.mirror_rgb, outA.width, outA.height,
                  studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                  /*map_memory=*/false, "mirror output", &vk_err) &&
              open_vulkan_vb.EnsureImage(
                  &open_vulkan_vb.final_readback_staging, outA.width,
                  outA.height, studiocast::vulkan::VulkanPixelFormat::rgb_u8,
                  /*map_memory=*/true, "final readback staging", &vk_err);
          vulkan_rgb_scaled_allocated = vulkan_rgb_scaled.Valid();
        }
        if (mirror_ready) {
          have_open_vulkan_mirror = true;
          compute_available.vulkan_available = true;
          set_backend(stage_id, "open_vulkan");
          append_note("Open Vulkan: Mirror (non-mapped DEVICE_LOCAL final "
                      "source/scratch/output with bounded upload and one "
                      "final-readback staging boundary).");
        } else {
          if (vk_err.find("[vulkan_effect_initialization_failed]") ==
              std::string::npos) {
            vk_err = OpenVulkanMirrorInitializationFailure(vk_err);
          }
          append_note(vk_err);
          compute_available.vulkan_unavailable_reason = vk_err;
          remove_stage_from_plan(stage_id);
        }
#else
        const std::string vk_err =
            "[vulkan_effect_initialization_failed] Open Vulkan mirror "
            "initialization failed: backend is disabled in this build.";
        append_note(vk_err);
        compute_available.vulkan_unavailable_reason = vk_err;
        remove_stage_from_plan(stage_id);
#endif
      } else if (!prepared_hardware_final_live) {
        remove_stage_from_plan(stage_id);
      }
    }

    if (!prepared_open_cuda_setup_attempted &&
        prepared_open_cuda_hardware_owner.Snapshot().prepared) {
      prepared_open_cuda_hardware_owner.Invalidate(
          hal::ReasonId::reconfiguration_required);
      prepared_open_cuda_hardware_live = false;
      publish_prepared_open_cuda_status();
    }

    auto prepared_frame_plan =
        studiocast::video::effects::CompileBroadcastEffectsFramePlan(plan);
    if (!prepared_frame_plan.valid) {
      if (!note.empty())
        note += "\n";
      note += "Internal effect-plan compilation failed; optional effects "
              "were disabled for safety.";
      plan.ordered_effect_ids.clear();
      plan.vignette_attach_to_effect_id.clear();
      prepared_frame_plan =
          studiocast::video::effects::CompileBroadcastEffectsFramePlan(plan);
    }

    // Freeze maximal Maxine-compatible islands only after every canonical
    // stage has resolved to its actual backend. Runtime objects below borrow
    // the already discovered SDK libraries; no provider/path discovery occurs
    // in frame execution.
    std::uint32_t resolved_maxine_stage_mask = 0;
    const auto resolve_maxine_stage = [&](EffectStage stage, bool resolved) {
      if (resolved && prepared_frame_plan.Contains(stage))
        resolved_maxine_stage_mask |= MaxineResolvedStageBit(stage);
    };
    resolve_maxine_stage(EffectStage::video_noise_removal,
                         have_maxine_denoise);
    resolve_maxine_stage(EffectStage::eye_contact,
                         have_maxine_eye_contact);
    resolve_maxine_stage(EffectStage::virtual_background_blur,
                         have_maxine_bg_blur);
    resolve_maxine_stage(EffectStage::virtual_background_remove,
                         have_maxine_bg_blur);
    resolve_maxine_stage(EffectStage::virtual_background_replace,
                         have_maxine_bg_blur);
    resolve_maxine_stage(EffectStage::virtual_key_light,
                         have_maxine_relight);
    resolve_maxine_stage(EffectStage::auto_frame,
                         have_maxine_auto_frame);
    const EffectStage attached_vignette_stage =
        prepared_frame_plan.vignette_attachment;
    const bool vignette_attachment_is_maxine =
        (attached_vignette_stage == EffectStage::eye_contact &&
         have_maxine_eye_contact) ||
        ((attached_vignette_stage ==
              EffectStage::virtual_background_blur ||
          attached_vignette_stage ==
              EffectStage::virtual_background_remove ||
          attached_vignette_stage ==
              EffectStage::virtual_background_replace) &&
         have_maxine_bg_blur) ||
        (attached_vignette_stage == EffectStage::virtual_key_light &&
         have_maxine_relight) ||
        (attached_vignette_stage == EffectStage::auto_frame &&
         have_maxine_auto_frame);
    resolve_maxine_stage(EffectStage::vignette,
                         have_maxine_vignette_only ||
                             vignette_attachment_is_maxine);

    std::uint64_t matte_fingerprint =
        effects_generation * UINT64_C(0x9E3779B185EBCA87);
    matte_fingerprint ^=
        static_cast<std::uint64_t>(fx.virtual_background.greenscreen_mode)
        << 1u;
    matte_fingerprint ^=
        static_cast<std::uint64_t>(
            fx.virtual_background.greenscreen_temporal)
        << 17u;
    if (matte_fingerprint == 0)
      matte_fingerprint = 1;
    prepared_maxine_resident_islands = CompileMaxineResidentIslands(
        prepared_frame_plan, resolved_maxine_stage_mask, matte_fingerprint);
    maxine_resident_executor.reset();
    maxine_resident_section.ResetSessionStateAndCounters();
    if (prepared_maxine_hardware_live) {
      prepared_maxine_hardware_owner.Invalidate(
          hal::ReasonId::reconfiguration_required);
      publish_prepared_maxine_status();
    }
    prepared_maxine_hardware_live = false;
    prepared_maxine_hardware_suspended = false;
    prepared_maxine_bgr_output.clear();
    maxine_resident_background_rgb.clear();
    maxine_resident_telemetry_snapshot = {};

    if (prepared_maxine_resident_islands.valid &&
        prepared_maxine_resident_islands.count > 0) {
      studiocast::maxine::ProductionResidentRuntime resident_runtime{};
      std::filesystem::path resident_model_dir;
      if (have_maxine_bg_blur) {
        resident_runtime.vfx = &maxine_bg_blur.vfx;
        resident_runtime.nvcv = &maxine_bg_blur.nvcv;
        resident_runtime.cuda = &maxine_bg_blur.cuda;
        resident_model_dir = maxine_bg_blur.model_dir;
      } else if (have_maxine_relight) {
        resident_runtime.vfx = &maxine_relight.vfx;
        resident_runtime.nvcv = &maxine_relight.nvcv;
        resident_runtime.cuda = &maxine_relight.cuda;
        resident_model_dir = maxine_relight.model_dir;
      } else if (have_maxine_denoise) {
        resident_runtime.vfx = &maxine_denoise.vfx;
        resident_runtime.nvcv = &maxine_denoise.nvcv;
        resident_runtime.cuda = &maxine_denoise.cuda;
        resident_model_dir = maxine_denoise.model_dir;
      }
      if (have_maxine_eye_contact) {
        resident_runtime.ar = &maxine_eye_contact.ar;
        if (!resident_runtime.nvcv)
          resident_runtime.nvcv = &maxine_eye_contact.nvcv;
        if (!resident_runtime.cuda)
          resident_runtime.cuda = &maxine_eye_contact.cuda;
      } else if (have_maxine_auto_frame) {
        resident_runtime.ar = &maxine_auto_frame.ar;
        if (!resident_runtime.nvcv)
          resident_runtime.nvcv = &maxine_auto_frame.nvcv;
        if (!resident_runtime.cuda)
          resident_runtime.cuda = &maxine_auto_frame.cuda;
      }
      if (!resident_runtime.nvcv && have_maxine_vignette_only) {
        resident_runtime.nvcv = &vignette_only.nvcv;
        resident_runtime.cuda = &vignette_only.cuda;
      }

      studiocast::maxine::ProductionResidentSetup resident_setup{};
      resident_setup.configuration_generation = effects_generation;
      resident_setup.width = static_cast<std::uint32_t>(capA.width);
      resident_setup.height = static_cast<std::uint32_t>(capA.height);
      // Every island returns to the existing capture-sized RGB continuation.
      // Output resize remains a later prepared pipeline stage.
      resident_setup.output_width = resident_setup.width;
      resident_setup.output_height = resident_setup.height;
      resident_setup.runtime_identity =
          reinterpret_cast<std::uintptr_t>(resident_runtime.cuda);
      resident_setup.effects = fx;
      if (have_maxine_relight && !maxine_relight.cached_hdri_path.empty()) {
        resident_setup.effects.virtual_key_light.hdri_path =
            maxine_relight.cached_hdri_path.string();
      }
      resident_setup.enabled_maxine_stage_mask =
          prepared_maxine_resident_islands.enabled_resident_stage_mask;
      resident_setup.vfx_model_directory = resident_model_dir;
      resident_setup.vignette_center_x_px =
          static_cast<float>(capA.width) * 0.5f;
      resident_setup.vignette_center_y_px =
          static_cast<float>(capA.height) * 0.5f;

      int background_width = 0;
      int background_height = 0;
      std::string resident_error;
      if ((resident_setup.enabled_maxine_stage_mask &
           studiocast::maxine::ProductionResidentStageBit(
               studiocast::maxine::ResidentStageKind::background_replace)) !=
          0) {
        if (!replace_source.valid ||
            !LoadImageRgb24(replace_source.path, &background_width,
                            &background_height,
                            &maxine_resident_background_rgb,
                            &resident_error)) {
          resident_error = resident_error.empty()
                               ? "Prepared replacement source is unavailable."
                               : resident_error;
        } else {
          resident_setup.replacement_background = {
              maxine_resident_background_rgb.data(),
              static_cast<std::uint32_t>(background_width),
              static_cast<std::uint32_t>(background_height),
              static_cast<std::size_t>(background_width) * 3u,
              effects_generation};
        }
      }

      prepared_maxine_hardware_live = false;
      prepared_maxine_hardware_suspended = false;
      if (resident_error.empty() && resident_runtime.nvcv &&
          resident_runtime.cuda &&
          prepared_maxine_resident_islands.count == 1) {
        PreparedMaxineHardwareRequest request;
        request.requested_backend = hal::BackendId::maxine;
        request.strict_backend_selection = true;
        request.single_resident_island = true;
        // V1 never chains two prepared adapters, even across an explicit CPU
        // readback/upload boundary. The retained Maxine executor remains the
        // compatibility route when the qualified Vulkan final owner is live.
        request.contains_other_execution_domain =
            prepared_hardware_final_live;
        request.width = resident_setup.width;
        request.height = resident_setup.height;
        request.vignette_center_x_px =
            resident_setup.vignette_center_x_px;
        request.vignette_center_y_px =
            resident_setup.vignette_center_y_px;
        request.effects = resident_setup.effects;
        request.generations.configuration = effects_generation;
        request.host_input_is_tight_rgb24 =
            rgbStride == static_cast<std::size_t>(capA.width) * 3u &&
            rgb.size() ==
                rgbStride * static_cast<std::size_t>(capA.height);
        const std::size_t bgr_stride =
            static_cast<std::size_t>(capA.width) * 3u;
        const std::size_t bgr_bytes =
            bgr_stride * static_cast<std::size_t>(capA.height);
        prepared_maxine_bgr_output.resize(bgr_bytes);
        request.host_output_is_tight_bgr24 =
            prepared_maxine_bgr_output.size() == bgr_bytes;
        request.runtime.vfx =
            reinterpret_cast<std::uintptr_t>(resident_runtime.vfx);
        request.runtime.nvcv =
            reinterpret_cast<std::uintptr_t>(resident_runtime.nvcv);
        request.runtime.ar =
            reinterpret_cast<std::uintptr_t>(resident_runtime.ar);
        request.runtime.cuda =
            reinterpret_cast<std::uintptr_t>(resident_runtime.cuda);
        request.vfx_model_directory = resident_model_dir.string();
        request.replacement = {
            resident_setup.replacement_background.data,
            resident_setup.replacement_background.width,
            resident_setup.replacement_background.height,
            resident_setup.replacement_background.stride_bytes,
            resident_setup.replacement_background.asset_generation,
        };

        constexpr std::uint32_t temporal_stage_mask =
            studiocast::maxine::ProductionResidentStageBit(
                studiocast::maxine::ResidentStageKind::denoise) |
            studiocast::maxine::ProductionResidentStageBit(
                studiocast::maxine::ResidentStageKind::eye_contact) |
            studiocast::maxine::ProductionResidentStageBit(
                studiocast::maxine::ResidentStageKind::auto_frame);
        constexpr std::uint32_t matte_stage_mask =
            studiocast::maxine::ProductionResidentStageBit(
                studiocast::maxine::ResidentStageKind::background_blur) |
            studiocast::maxine::ProductionResidentStageBit(
                studiocast::maxine::ResidentStageKind::background_remove) |
            studiocast::maxine::ProductionResidentStageBit(
                studiocast::maxine::ResidentStageKind::background_replace) |
            studiocast::maxine::ProductionResidentStageBit(
                studiocast::maxine::ResidentStageKind::relighting);
        request.temporal_reset_required =
            (resident_setup.enabled_maxine_stage_mask &
             temporal_stage_mask) != 0 ||
            (fx.virtual_background.greenscreen_temporal &&
             (resident_setup.enabled_maxine_stage_mask &
              matte_stage_mask) != 0);

        const auto semantic_for_kind =
            [](studiocast::maxine::ResidentStageKind kind) {
              using Kind = studiocast::maxine::ResidentStageKind;
              switch (kind) {
              case Kind::background_blur:
                return hal::SemanticEffectId::virtual_background_blur;
              case Kind::background_remove:
                return hal::SemanticEffectId::virtual_background_remove;
              case Kind::background_replace:
                return hal::SemanticEffectId::virtual_background_replace;
              case Kind::relighting:
                return hal::SemanticEffectId::virtual_key_light;
              case Kind::vignette:
                return hal::SemanticEffectId::vignette;
              case Kind::denoise:
                return hal::SemanticEffectId::video_noise_removal;
              case Kind::eye_contact:
                return hal::SemanticEffectId::eye_contact;
              case Kind::auto_frame:
                return hal::SemanticEffectId::auto_frame;
              case Kind::transfer:
              case Kind::count:
                return hal::SemanticEffectId::invalid;
              }
              return hal::SemanticEffectId::invalid;
            };
        const auto &only_island =
            prepared_maxine_resident_islands.islands[0];
        for (std::size_t index = 0;
             index < only_island.plan.stage_count; ++index) {
          const auto semantic =
              semantic_for_kind(only_island.plan.stages[index].kind);
          request.stages[request.stage_count++] = {
              static_cast<std::uint32_t>(semantic),
              semantic,
              !only_island.plan.stages[index].optional,
          };
        }
        const auto hal_setup =
            prepared_maxine_hardware_owner.Configure(request);
        publish_prepared_maxine_status();
        prepared_maxine_hardware_live =
            hal_setup.route == PreparedMaxineHardwareRoute::hal_live;
        if (prepared_maxine_hardware_live) {
          if (!note.empty())
            note += "\n";
          note +=
              "Maxine HAL: one prepared adapter owns the qualified resident "
              "island, one upload, qualified resident stages, explicit RGB "
              "output tail, and one readback.";
        }
      }

      if (!prepared_maxine_hardware_live && resident_error.empty() &&
          resident_runtime.nvcv &&
          resident_runtime.cuda) {
        maxine_resident_executor = std::make_unique<
            studiocast::maxine::ProductionResidentFrameExecutor>(
            resident_runtime);
        const bool resident_configured =
            maxine_resident_executor->Configure(resident_setup,
                                                &resident_error);
        const auto &setup_telemetry = maxine_resident_executor->telemetry();
        const auto setup_transfer_index = static_cast<std::size_t>(
            studiocast::maxine::ProductionTransferKind::
                background_asset_setup_upload);
        maxine_background_setup_upload_attempts +=
            setup_telemetry.transfers[setup_transfer_index].attempts;
        maxine_background_setup_upload_calls +=
            setup_telemetry.transfers[setup_transfer_index].successes;
        maxine_resident_setup_attempts += setup_telemetry.setup_attempts;
        maxine_resident_setup_successes += setup_telemetry.setup_successes;
        if (!resident_configured) {
          maxine_resident_executor.reset();
        } else {
          // Setup transfers (for example the replacement-background upload)
          // are not frame transfers. Begin frame-time delta accounting from
          // the fully prepared executor state.
          maxine_resident_telemetry_snapshot =
              maxine_resident_executor->telemetry();
        }
      } else if (resident_error.empty()) {
        resident_error = "Resolved Maxine runtime is incomplete.";
      }
      if (!maxine_resident_executor && !prepared_maxine_hardware_live) {
        // A stage resolved to Maxine must never silently fall back to the old
        // stage-isolated upload/download path. Keep the CPU frame unchanged
        // and publish the disabled generation authoritatively instead.
        have_maxine_denoise = false;
        have_maxine_eye_contact = false;
        have_maxine_bg_blur = false;
        have_maxine_relight = false;
        have_maxine_auto_frame = false;
        have_maxine_vignette_only = false;
        prepared_maxine_resident_islands = {};
        if (!note.empty())
          note += "\n";
        note += "Maxine resident section unavailable; resolved Maxine effects "
                "are disabled for this configuration: " +
                resident_error;
      } else if (maxine_resident_executor) {
        if (!note.empty())
          note += "\n";
        note += "Maxine: compatible stages share one prepared resident frame "
                "section, stream, upload, matte, and readback per island.";
      }
    }

    {
      append_rule_notes();

      const bool have_any_maxine_compute =
          have_maxine_bg_blur || have_maxine_auto_frame ||
          have_maxine_eye_contact || have_maxine_relight || have_maxine_denoise;
      const bool have_any_vulkan_compute =
          have_open_vulkan_vb || have_open_vulkan_auto_frame ||
          have_open_vulkan_key_light || have_open_vulkan_vignette ||
          have_open_vulkan_mirror;
      const bool have_any_cuda_compute =
          have_any_maxine_compute || have_open_cuda_vb ||
          have_open_cuda_auto_frame || have_open_cuda_key_light ||
          have_open_cuda_video_denoise || have_maxine_vignette_only;

      compute_available.vulkan_available = have_any_vulkan_compute;
      if (!compute_available.vulkan_available &&
          compute_available.vulkan_unavailable_reason.empty()) {
        compute_available.vulkan_unavailable_reason =
            "Vulkan compute backend requested, but no Vulkan-backed video "
            "effect was available.";
      }
      compute_available.cuda_available =
          cuda_video_compute_allowed && have_any_cuda_compute;
      if (!compute_available.cuda_available) {
        compute_available.cuda_unavailable_reason =
            "CUDA compute backend requested, but no CUDA-backed video effect "
            "was available.";
      }
      compute_selection = ResolveComputeBackendSelection(
          cfg.compute_backend, compute_available, compute_work_requested);

      const std::string compute_active_backend =
          ResolveActiveComputeBackendName(
              compute_selection, compute_work_requested,
              have_any_maxine_compute, compute_available.cuda_available,
              compute_available.vulkan_available);

      for (auto &breaker : optional_effect_breakers)
        breaker.Reset();
      appliedEffectsBaseNote = note;

      std::lock_guard<std::mutex> lock(mu_);
      const bool last_error_was_previous_effects_note =
          !effects_note_.empty() && last_error_ == effects_note_;
      // `backend_for_effect` is setup/readiness evidence only. Public live
      // attribution is committed after successful effect execution and output
      // write, never while building this plan.
      effects_backends_.clear();
      effects_note_ = note;
      compute_backend_preference_ =
          ComputeBackendPreferenceToString(cfg.compute_backend);
      compute_backend_resolved_ =
          ComputeBackendKindToString(compute_selection.resolved);
      compute_backend_active_ = compute_active_backend;
      compute_backend_fallback_reason_ = compute_selection.fallback_reason;
      compute_backend_degraded_reason_ = compute_selection.degraded_reason;
      degraded_effect_ = CameraPipelineStatus::DegradedEffect{};
      if (!note.empty()) {
        last_error_ = note;
      } else if (last_error_was_previous_effects_note) {
        last_error_.clear();
      }
    }

    appliedPlan = plan;
    appliedFramePlan = prepared_frame_plan;
    appliedFx = fx;
    appliedEffectsGeneration = effects_generation;
    raw_yuyv_plan = PrepareRawYuyvPassthroughPlan(capA, outA, appliedFx);
    publish_video_hal_status(appliedFx, appliedEffectsGeneration,
                             VideoRuntimeLifecycle::running);
  };

  // Initial chain based on config at pipeline start.
  {
    studiocast::video::effects::BroadcastCameraEffects fx;
    detail::PreparedReplaceBackgroundSource replace_source;
    std::uint64_t effects_generation = 0;
    {
      std::lock_guard<std::mutex> fxLock(effects_mu_);
      fx = effects_;
      replace_source = replace_background_source_;
      effects_generation = effects_generation_;
    }
    rebuildChain(fx, replace_source, effects_generation);
    effects_preparation_counters.RecordConfigSnapshot(/*rebuilt=*/true);
  }

  struct Ema {
    double alpha = 0.05;
    bool initialized = false;
    double v = 0.0;

    void Add(double x) {
      if (!initialized) {
        v = x;
        initialized = true;
        return;
      }
      v = (1.0 - alpha) * v + alpha * x;
    }

    double ValueOrZero() const { return initialized ? v : 0.0; }
  };

  using Clock = std::chrono::steady_clock;
  const auto ToMs = [](Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };

  Ema ema_capture;
  Ema ema_scale;
  Ema ema_effects;
  Ema ema_write;
  Ema ema_period;

  // Output pacing to reduce burstiness (useful for browsers/WebRTC capture).
  //
  // The capture thread can sometimes "catch up" and deliver frames in a burst
  // after a stall (even when we drop old frames). Some consumers (notably
  // WebRTC stacks) are sensitive to bursty cadence. Pace writes to the
  // negotiated output FPS when we're ahead of schedule, but never add
  // extra delay when we're already behind.
  Ema ema_pace_sleep;
  Ema ema_pace_late;
  std::uint64_t pace_sleeps = 0;
  std::uint64_t pace_late_frames = 0;
  std::uint64_t pace_resyncs = 0;

  auto PeriodForFps = [](int fps) -> Clock::duration {
    if (fps <= 0)
      return Clock::duration::zero();
    return std::chrono::nanoseconds(1000000000LL / static_cast<long long>(fps));
  };

  auto PaceFpsFromOutput = [&](const ActualFormat &a) -> int {
    if (a.fps > 0)
      return a.fps;
    if (cfg.fps > 0)
      return cfg.fps;
    return 30;
  };

  int pace_fps = PaceFpsFromOutput(outA);
  Clock::duration pace_period = PeriodForFps(pace_fps);
  Clock::time_point next_pace_deadline{};
  bool have_next_pace_deadline = false;

  // Optional end-to-end latency estimate and capture backlog stats.
  Ema ema_latency;
  std::uint64_t last_capture_sequence = 0;
  int dropped_capture_frames_total = 0;

  auto NowNs = [](clockid_t clock_id) -> std::uint64_t {
    timespec ts{};
    if (::clock_gettime(clock_id, &ts) != 0)
      return 0;
    return (static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ULL) +
           static_cast<std::uint64_t>(ts.tv_nsec);
  };
  auto NowNsMonotonic = [&]() -> std::uint64_t {
    return NowNs(CLOCK_MONOTONIC);
  };
  auto NowNsRealtime = [&]() -> std::uint64_t { return NowNs(CLOCK_REALTIME); };

  Clock::time_point last_frame_end{};
  bool have_last_frame_end = false;
  int perf_frames = 0;

  const bool debug_open_cuda_transfers =
      (std::getenv("STUDIOCAST_DEBUG_OPEN_CUDA_TRANSFERS") != nullptr) ||
      (std::getenv("STUDIOCAST_DEBUG_CUDA_UPLOADS") != nullptr);
  std::uint64_t open_cuda_active_frames = 0;
  std::uint64_t open_cuda_frame_upload_calls = 0;
  std::uint64_t open_cuda_rgb_download_calls = 0;
  std::uint64_t open_cuda_final_download_calls = 0;
  std::uint64_t open_cuda_cpu_continuation_download_calls = 0;
  std::uint64_t open_cuda_denoise_tensor_upload_calls = 0;
  std::uint64_t open_cuda_denoise_tensor_download_calls = 0;
  std::uint64_t open_cuda_denoise_cpu_tail_calls = 0;
  std::uint64_t open_cuda_standalone_scaler_upload_calls = 0;
  std::uint64_t open_cuda_standalone_scaler_download_calls = 0;
  std::uint64_t open_cuda_forced_sync_calls = 0;
  std::uint64_t open_cuda_cpu_tail_stage_calls = 0;
  std::uint64_t open_cuda_cpu_tail_key_light_calls = 0;
  std::uint64_t open_cuda_cpu_tail_auto_frame_calls = 0;
  std::uint64_t open_cuda_cpu_tail_auto_frame_face_tracking_calls = 0;
  std::uint64_t open_cuda_cpu_tail_auto_frame_matte_tracking_calls = 0;
  std::uint64_t open_cuda_cpu_tail_auto_frame_cpu_crop_calls = 0;

  std::uint64_t open_vulkan_active_frames = 0;
  std::uint64_t open_vulkan_upload_calls = 0;
  std::uint64_t open_vulkan_download_calls = 0;
  std::uint64_t open_vulkan_final_download_calls = 0;
  std::uint64_t open_vulkan_cpu_continuation_download_calls = 0;
  std::uint64_t open_vulkan_cpu_tail_stage_calls = 0;
  std::uint64_t open_vulkan_cpu_tail_key_light_calls = 0;
  std::uint64_t open_vulkan_cpu_tail_auto_frame_calls = 0;
  std::uint64_t open_vulkan_cpu_tail_auto_frame_face_tracking_calls = 0;
  std::uint64_t open_vulkan_cpu_tail_auto_frame_matte_tracking_calls = 0;
  std::uint64_t open_vulkan_cpu_tail_auto_frame_cpu_crop_calls = 0;
  std::uint64_t open_vulkan_standalone_scaler_upload_calls = 0;
  std::uint64_t open_vulkan_standalone_scaler_download_calls = 0;
  std::uint64_t open_vulkan_forced_sync_calls = 0;
  std::uint64_t open_vulkan_fallback_frames = 0;
  std::uint64_t open_vulkan_runtime_failure_frames = 0;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
  OpenVulkanMirrorCounters open_vulkan_mirror_counters;
  OpenVulkanFinalResidentStageCounters open_vulkan_final_stage_counters;
#endif

  const bool debug_maxine_transfers =
      (std::getenv("STUDIOCAST_DEBUG_MAXINE_TRANSFERS") != nullptr);
  std::uint64_t maxine_active_frames = 0;
  std::uint64_t maxine_rgb_to_bgr_calls = 0;
  std::uint64_t maxine_upload_attempts = 0;
  std::uint64_t maxine_upload_calls = 0;
  std::uint64_t maxine_matte_inference_attempts = 0;
  std::uint64_t maxine_green_screen_calls = 0;
  std::uint64_t maxine_duplicate_green_screen_calls = 0;
  std::uint64_t maxine_shared_green_screen_matte_reuse_calls = 0;
  std::uint64_t maxine_shared_green_screen_matte_incompatible_calls = 0;
  std::uint64_t maxine_shared_green_screen_input_incompatible_calls = 0;
  std::uint64_t maxine_download_calls = 0;
  std::uint64_t maxine_final_download_attempts = 0;
  std::uint64_t maxine_final_download_calls = 0;
  std::uint64_t maxine_cpu_continuation_download_attempts = 0;
  std::uint64_t maxine_cpu_continuation_download_calls = 0;
  std::uint64_t maxine_device_bridge_attempts = 0;
  std::uint64_t maxine_device_bridge_calls = 0;
  std::uint64_t maxine_bgr_to_rgb_calls = 0;
  std::uint64_t maxine_deferred_readbacks = 0;
  std::uint64_t maxine_forced_sync_attempts = 0;
  std::uint64_t maxine_forced_sync_calls = 0;
  std::uint64_t maxine_composite_attempts = 0;
  std::uint64_t maxine_composite_calls = 0;
  std::uint64_t maxine_synchronous_sdk_run_attempts = 0;
  std::uint64_t maxine_synchronous_sdk_run_calls = 0;
  std::uint64_t maxine_asynchronous_sdk_run_attempts = 0;
  std::uint64_t maxine_asynchronous_sdk_run_calls = 0;
  std::uint64_t maxine_cpu_tail_stage_calls = 0;
  std::uint64_t maxine_runtime_failure_frames = 0;
  std::array<std::uint64_t,
             static_cast<std::size_t>(
                 studiocast::maxine::ResidentStageKind::count)>
      maxine_stage_attempts{};
  std::array<std::uint64_t,
             static_cast<std::size_t>(
                 studiocast::maxine::ResidentStageKind::count)>
      maxine_stage_successes{};
  std::uint64_t maxine_standalone_scaler_upload_calls = 0;
  std::uint64_t maxine_standalone_scaler_download_calls = 0;

  const bool debug_v4l2_neg =
      (std::getenv("STUDIOCAST_DEBUG_V4L2_NEGOTIATION") != nullptr);

  int output_format_changes = 0;
  int output_refresh_failures = 0;
  int output_write_recoveries = 0;
  bool raw_capture_fallback_attempted = false;

  // Caller holds mu_. The writer counters are updated only by this thread.
  const auto PublishWriterCountersLocked = [&] {
    const FrameWriteCounters &writes = writer_.WriteCounters();
    debug_.output_write_syscalls = writes.write_syscalls;
    debug_.output_frames_committed = writes.frames_written;
    debug_.output_would_block_drops = writes.would_block_drops;
    debug_.output_stopped_writes = writes.stopped_writes;
    debug_.output_eintr_retries = writes.eintr_retries;
    debug_.output_partial_frame_failures = writes.partial_frame_failures;
    debug_.output_fatal_failures = writes.fatal_failures;
  };

  auto ActualFormatToString = [](const ActualFormat &a) {
    std::ostringstream oss;
    oss << a.width << "x" << a.height << "@" << a.fps << " ";
    oss << a.pixfmt;
    oss << " bpl=" << a.bytes_per_line << " size=" << a.size_image;
    return oss.str();
  };

  auto ActualFormatEq = [](const ActualFormat &a, const ActualFormat &b) {
    return a.width == b.width && a.height == b.height && a.fps == b.fps &&
           a.pixfmt_fourcc == b.pixfmt_fourcc &&
           a.bytes_per_line == b.bytes_per_line && a.size_image == b.size_image;
  };

  auto TryRawCaptureFallbackAfterMjpegFailure =
      [&](const std::string &reason, std::string *error) -> bool {
    if (error)
      error->clear();
    if (!ShouldFallbackToRawAfterMjpegDecodeFailure(
            capA, raw_capture_fallback_attempted)) {
      if (error)
        *error = reason;
      return false;
    }

    raw_capture_fallback_attempted = true;
    const int old_w = capA.width;
    const int old_h = capA.height;
    const int fallback_fps = capA.fps > 0 ? capA.fps : cfg.fps;

    std::string open_err;
    if (!cap.Open(inDev, old_w, old_h, fallback_fps, CapturePixelFormat::yuyv,
                  /*prefer_mjpeg=*/false, &open_err)) {
      if (error)
        *error = reason + " Raw YUYV fallback failed: " + open_err;
      return false;
    }

    CaptureFormat fallback = cap.Actual();
    if (fallback.format != CapturePixelFormat::yuyv ||
        fallback.width != old_w || fallback.height != old_h) {
      if (error) {
        std::ostringstream oss;
        oss << reason << " Raw YUYV fallback negotiated " << fallback.width
            << "x" << fallback.height << " " << fallback.pixfmt << ", expected "
            << old_w << "x" << old_h << " YUYV.";
        *error = oss.str();
      }
      return false;
    }

    capA = fallback;
    rgbStride = rgb.stride_bytes;
    raw_yuyv_plan =
        PrepareRawYuyvPassthroughPlan(capA, outA, appliedFx);
    {
      std::lock_guard<std::mutex> lock(mu_);
      capture_ = capA;
      scaling_from_ = capA;
      capture_fallback_state_ = "raw_after_mjpeg_decode_failure";
      capture_fallback_reason_ = reason;
    }

    if (debug_v4l2_neg) {
      std::fprintf(stderr, "capture fallback: %s; reopened %s as %s\n",
                   reason.c_str(), inDev.c_str(), capA.pixfmt.c_str());
    }
    return true;
  };

  auto RefreshOutputActual = [&](const char *reason) -> bool {
    if (!writer_.IsOpen())
      return false;

    const ActualFormat prev = outA;
    std::string rerr;
    if (!writer_.RefreshActual(&rerr)) {
      ++output_refresh_failures;
      if (debug_v4l2_neg && (output_refresh_failures <= 5 ||
                             (output_refresh_failures % 30) == 0)) {
        std::fprintf(stderr,
                     "v4l2loopback RefreshActual failed reason=%s err=\"%s\"\n",
                     reason ? reason : "", rerr.c_str());
      }
      return false;
    }

    const ActualFormat cur = writer_.Actual();
    if (ActualFormatEq(cur, prev))
      return false;

    outA = cur;
    ++output_format_changes;

    // If the consumer renegotiated FPS, update pacing immediately so we don't
    // deliver bursty cadence after a stall (important for browser/WebRTC
    // capture).
    const int new_pace_fps = PaceFpsFromOutput(outA);
    if (new_pace_fps != pace_fps) {
      pace_fps = new_pace_fps;
      pace_period = PeriodForFps(pace_fps);
      have_next_pace_deadline = false;
      ++pace_resyncs;
    }

    // Keep output buffers sized to the negotiated kernel format.
    std::size_t need = outA.size_image;
    if (need == 0 && outA.bytes_per_line > 0 && outA.height > 0) {
      need = outA.bytes_per_line * static_cast<std::size_t>(outA.height);
    }
    if (need > 0)
      outBuf.resize(need);
    if (outA.width > 0 && outA.height > 0) {
      rgbScaled.resize(static_cast<std::size_t>(outA.width) *
                       static_cast<std::size_t>(outA.height) * 3u);
    }
    if (prepared_hardware_final_live) {
      prepared_hardware_composition.Invalidate(
          hal::ReasonId::reconfiguration_required);
      prepared_hardware_final_live = false;
      have_open_vulkan_vignette = false;
      have_open_vulkan_mirror = false;
      publish_prepared_hardware_status();
    }
    if (prepared_open_cuda_hardware_live) {
      prepared_open_cuda_hardware_owner.Invalidate(
          hal::ReasonId::reconfiguration_required);
      prepared_open_cuda_hardware_live = false;
      prepared_open_cuda_hardware_suspended = false;
      publish_prepared_open_cuda_status();
    }
    if (prepared_maxine_hardware_live) {
      prepared_maxine_hardware_owner.Invalidate(
          hal::ReasonId::reconfiguration_required);
      prepared_maxine_hardware_live = false;
      prepared_maxine_hardware_suspended = false;
      publish_prepared_maxine_status();
    }
    ++prepared_hardware_geometry_generation;
    if (prepared_hardware_geometry_generation == 0)
      prepared_hardware_geometry_generation = 1;
    publish_video_hal_status(appliedFx, appliedEffectsGeneration,
                             VideoRuntimeLifecycle::running);
    // The next frame must rebuild against the renegotiated layout. This also
    // prepares the retained legacy final stage when the new layout is not HAL
    // eligible; no silently missing vignette/mirror interval is permitted.
    appliedEffectsGeneration = 0;
    ResizeYuyvConversionScratch();
    ZeroRgbOutputPadding();
    raw_yuyv_plan =
        PrepareRawYuyvPassthroughPlan(capA, outA, appliedFx);

    {
      std::lock_guard<std::mutex> lock(mu_);
      output_ = outA;
      scaling_to_ = outA;
    }

    if (debug_v4l2_neg) {
      const std::string prevS = ActualFormatToString(prev);
      const std::string curS = ActualFormatToString(cur);
      std::fprintf(stderr,
                   "v4l2loopback output format changed (%s): %s -> %s\n",
                   reason ? reason : "", prevS.c_str(), curS.c_str());
    }
    return true;
  };

  // Reused only for exceptional zero-timeout drain failures. The expected
  // no-frame result does not touch or format this string.
  std::string capture_drain_error;
  while (!stop_.load()) {
    CapturedFrameView f{};
    std::string ferr;
    if (!cap.AcquireFrame(&f, 1000, &ferr)) {
      // Timeouts and transient interruptions can happen on some
      // devices/drivers. Treat timeouts as recoverable to avoid pipeline
      // flapping (which can manifest as periodic black frames when the loopback
      // output is kept alive).
      if (stop_.load())
        break;

      if (IsRecoverableCaptureAcquireFailure(ferr))
        continue;

      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = "Capture acquire failed: " + ferr;
      break;
    }

    // Reduce end-to-end latency by always processing the newest frame
    // available.
    //
    // If our loop ever runs behind (or the driver buffers aggressively),
    // multiple frames can be queued by the time we wake up. Drain any queued
    // frames and keep only the most recent, dropping older frames to avoid
    // "living in the past".
    int dropped_this_frame = 0;
    bool capture_drain_failed = false;
    for (;;) {
      if (stop_.load())
        break;

      CapturedFrameView newer{};
      const CaptureAcquireResult acquire_result =
          cap.AcquireFrameDetailed(&newer, 0, &capture_drain_error);
      if (acquire_result == CaptureAcquireResult::no_frame) {
        // No additional frame ready.
        break;
      }
      if (acquire_result == CaptureAcquireResult::failure) {
        capture_drain_failed = true;
        break;
      }

      // Release the older frame and keep the newer one.
      std::string rerr;
      (void)cap.ReleaseFrame(f, &rerr);
      f = newer;
      ++dropped_this_frame;
    }
    if (dropped_this_frame > 0) {
      dropped_capture_frames_total += dropped_this_frame;
    }
    if (capture_drain_failed) {
      std::string rerr;
      (void)cap.ReleaseFrame(f, &rerr);
      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = "Capture drain failed";
      if (!capture_drain_error.empty())
        last_error_ += ": " + capture_drain_error;
      break;
    }
    last_capture_sequence = f.sequence;

    if (stop_.load()) {
      std::string rerr;
      (void)cap.ReleaseFrame(f, &rerr);
      break;
    }

    const auto t_capture_start = Clock::now();
    bool raw_yuyv_passthrough_frame = false;

    // Convert capture -> internal RGB (tight stride)
    if (capA.format == CapturePixelFormat::yuyv) {
      if (ValidateYuyvCaptureFrame(capA, f.bytes) !=
          YuyvCaptureFrameValidation::ok) {
        std::string rerr;
        (void)cap.ReleaseFrame(f, &rerr);
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ =
            "Invalid or undersized negotiated YUYV capture frame layout.";
        break;
      }

      const std::uint64_t published_effects_generation =
          effects_generation_published_.load(std::memory_order_acquire);
      raw_yuyv_passthrough_frame =
          raw_yuyv_plan.eligible && published_effects_generation != 0 &&
          published_effects_generation == appliedEffectsGeneration;

      if (raw_yuyv_passthrough_frame) {
        const auto copy_result = CopyRawYuyvFrame(
            raw_yuyv_plan, f.data, f.bytes, outBuf.data(), outBuf.size(),
            &yuyv_frame_path_counters);
        if (copy_result != RawYuyvCopyResult::ok) {
          std::string rerr;
          (void)cap.ReleaseFrame(f, &rerr);
          std::lock_guard<std::mutex> lock(mu_);
          last_error_ = "Raw YUYV passthrough copy rejected the frame layout.";
          break;
        }
      } else {
        CountedYuyvToRgb24(f.data, capA.width, capA.height,
                           capA.bytes_per_line, rgb.data(), rgbStride,
                           &yuyv_frame_path_counters);
      }
    } else if (capA.format == CapturePixelFormat::mjpeg) {
      if (f.bytes == 0) {
        std::string rerr;
        (void)cap.ReleaseFrame(f, &rerr);
        std::string fallback_err;
        if (TryRawCaptureFallbackAfterMjpegFailure(
                "MJPEG frame was empty (bytesused=0).", &fallback_err)) {
          continue;
        }
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = fallback_err.empty()
                          ? "MJPEG frame was empty (bytesused=0)."
                          : fallback_err;
        break;
      }
      int decW = 0;
      int decH = 0;
      std::string decErr;
      if (!DecodeMjpegToRgb24(f.data, f.bytes, rgb, decW, decH, &decErr)) {
        std::string rerr;
        (void)cap.ReleaseFrame(f, &rerr);
        std::string reason = "MJPEG decode failed";
        if (!decErr.empty())
          reason += ": " + decErr;
        reason += ".";
        std::string fallback_err;
        if (TryRawCaptureFallbackAfterMjpegFailure(reason, &fallback_err)) {
          continue;
        }
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = fallback_err.empty() ? reason : fallback_err;
        break;
      }

      if (decW != capA.width || decH != capA.height) {
        std::string rerr;
        (void)cap.ReleaseFrame(f, &rerr);
        std::ostringstream oss;
        oss << "MJPEG decode size mismatch: got " << decW << "x" << decH
            << ", expected " << capA.width << "x" << capA.height << ".";
        const std::string reason = oss.str();
        std::string fallback_err;
        if (TryRawCaptureFallbackAfterMjpegFailure(reason, &fallback_err)) {
          continue;
        }
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = fallback_err.empty() ? reason : fallback_err;
        break;
      }

      // The decoder is allowed to resize the RGB frame; keep our cached stride
      // in sync.
      rgbStride = rgb.stride_bytes;
    } else {
      std::string rerr;
      (void)cap.ReleaseFrame(f, &rerr);
      std::lock_guard<std::mutex> lock(mu_);
      last_error_ = "Unsupported negotiated capture format: " + capA.pixfmt;
      break;
    }

    std::string rerr;
    (void)cap.ReleaseFrame(f, &rerr);

    const auto t_capture_end = Clock::now();
    const double capture_ms = ToMs(t_capture_end - t_capture_start);

    live_effect_attribution.BeginFrame();
    const auto mark_live_effect_success = [&](std::string_view effect_id,
                                              std::string_view backend) {
      live_effect_attribution.MarkEffectSucceeded(effect_id, backend);
    };
    const auto remove_live_effect = [&](std::string_view effect_id) {
      if (!live_effect_attribution.RemoveEffect(effect_id))
        return;
      std::lock_guard<std::mutex> lock(mu_);
      effects_backends_ = live_effect_attribution.active_backends();
    };

    DeferredGpuOut deferred_gpu_out{};
    bool have_deferred_gpu_out = false;
    hal::CompletionToken prepared_hardware_completion{};
    bool prepared_hardware_completion_pending = false;
    hal::CompletionToken prepared_maxine_completion{};
    bool prepared_maxine_completion_pending = false;
    hal::CompletionToken prepared_open_cuda_completion{};
    bool prepared_open_cuda_completion_pending = false;
    bool prepared_open_cuda_output_ready = false;
    bool open_cuda_gpu_frame_pending_cpu_readback = false;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
    bool open_vulkan_vignette_attempt_this_frame = false;
    bool open_vulkan_gpu_frame_pending_cpu_readback = false;
    bool open_vulkan_mirror_attempt_this_frame = false;
#endif
    bool open_cuda_active_this_frame = false;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
    bool open_vulkan_active_this_frame = false;
    bool open_vulkan_effect_ran_this_frame = false;
#endif
    bool maxine_active_this_frame = false;
    int maxine_green_screen_calls_this_frame = 0;
    MaxineFrameLocalSharedHandles maxine_shared_handles;

    // Reset per-frame Open CUDA ping-pong state.
    open_cuda_curr = nullptr;
    open_cuda_next = nullptr;
    open_cuda_uploaded_this_frame = false;

#if STUDIOCAST_ENABLE_OPEN_VULKAN
    studiocast::vulkan::VulkanImage *open_vulkan_curr = nullptr;
    studiocast::vulkan::VulkanImage *open_vulkan_next = nullptr;
    bool open_vulkan_uploaded_this_frame = false;
    OpenVulkanResidentFrameState open_vulkan_resident_frame;
#endif

    auto mark_open_cuda_active_frame = [&]() {
      (void)MarkGpuBackendActiveFrame(open_cuda_active_this_frame,
                                      open_cuda_active_frames);
    };

#if STUDIOCAST_ENABLE_OPEN_VULKAN
    auto mark_open_vulkan_active_frame = [&]() {
      (void)MarkGpuBackendActiveFrame(open_vulkan_active_this_frame,
                                      open_vulkan_active_frames);
    };

    auto mark_open_vulkan_cpu_tail = [&](std::string_view stage_id) {
      mark_open_vulkan_active_frame();
      ++open_vulkan_cpu_tail_stage_calls;
      if (stage_id ==
          studiocast::video::effects::contract::kEffectIdVirtualKeyLight) {
        ++open_vulkan_cpu_tail_key_light_calls;
      } else if (stage_id ==
                 studiocast::video::effects::contract::kEffectIdAutoFrame) {
        ++open_vulkan_cpu_tail_auto_frame_calls;
      }
    };

    auto mark_open_vulkan_auto_frame_cpu_tail =
        [&](OpenCudaAutoFrameContext::TrackingSource tracking_source,
            bool cpu_crop_resize) {
          mark_open_vulkan_cpu_tail(
              studiocast::video::effects::contract::kEffectIdAutoFrame);
          switch (tracking_source) {
          case OpenCudaAutoFrameContext::TrackingSource::kFace:
            ++open_vulkan_cpu_tail_auto_frame_face_tracking_calls;
            break;
          case OpenCudaAutoFrameContext::TrackingSource::kMatte:
            ++open_vulkan_cpu_tail_auto_frame_matte_tracking_calls;
            break;
          default:
            break;
          }
          if (cpu_crop_resize)
            ++open_vulkan_cpu_tail_auto_frame_cpu_crop_calls;
        };

    auto ensure_open_vulkan_current_from_cpu = [&](std::string *error) {
      if (error)
        error->clear();
      if (open_vulkan_uploaded_this_frame && open_vulkan_curr &&
          open_vulkan_curr->Valid()) {
        return true;
      }
      mark_open_vulkan_active_frame();
      std::string vk_err;
      if (!open_vulkan_vb.kernels.Initialized() &&
          !open_vulkan_vb.kernels.Initialize(&vk_err)) {
        if (error)
          *error = "Open Vulkan: utility kernels unavailable: " + vk_err;
        return false;
      }
      if (!open_vulkan_vb.EnsureImage(
              &open_vulkan_vb.frame_rgb, capA.width, capA.height,
              studiocast::vulkan::VulkanPixelFormat::rgb_u8,
              /*map_memory=*/false, "resident frame_rgb", error) ||
          !open_vulkan_vb.EnsureImage(
              &open_vulkan_vb.out_rgb, capA.width, capA.height,
              studiocast::vulkan::VulkanPixelFormat::rgb_u8,
              /*map_memory=*/false, "resident out_rgb", error)) {
        return false;
      }
      const std::uint64_t submissions_before =
          open_vulkan_vb.kernels.synchronous_submission_count();
      if (!open_vulkan_vb.UploadRgbToDeviceLocal(
              rgb.data(), rgbStride, capA.width, capA.height, error)) {
        return false;
      }
      open_vulkan_forced_sync_calls +=
          open_vulkan_vb.kernels.synchronous_submission_count() -
          submissions_before;
      open_vulkan_curr = &open_vulkan_vb.final_source_rgb;
      open_vulkan_next = &open_vulkan_vb.frame_rgb;
      open_vulkan_resident_frame.current = open_vulkan_curr;
      open_vulkan_uploaded_this_frame = true;
      ++open_vulkan_upload_calls;
      return true;
    };

    auto ensure_open_vulkan_final_current_from_cpu = [&](std::string *error) {
      return ensure_open_vulkan_current_from_cpu(error);
    };

    auto ensure_open_vulkan_next = [&]() {
      if (open_vulkan_next)
        return;
      open_vulkan_next = open_vulkan_curr == &open_vulkan_vb.frame_rgb
                             ? &open_vulkan_vb.out_rgb
                             : &open_vulkan_vb.frame_rgb;
    };

    auto adopt_open_vulkan_output =
        [&](const studiocast::vulkan::VulkanImage *output) {
          if (!output || output != open_vulkan_next)
            return;
          open_vulkan_curr = open_vulkan_next;
          open_vulkan_next = open_vulkan_curr == &open_vulkan_vb.frame_rgb
                                 ? &open_vulkan_vb.out_rgb
                                 : &open_vulkan_vb.frame_rgb;
          open_vulkan_resident_frame.current = open_vulkan_curr;
        };

    auto download_open_vulkan_current_to_cpu = [&](std::string *error) {
      if (error)
        error->clear();
      if (!open_vulkan_gpu_frame_pending_cpu_readback) {
        return true;
      }
      if (!open_vulkan_curr || !open_vulkan_curr->Valid()) {
        if (error)
          *error = "Open Vulkan: no current GPU frame to download.";
        return false;
      }
      if (!open_vulkan_vb.DownloadImageToRgb(*open_vulkan_curr, rgb.data(),
                                             rgbStride, error)) {
        return false;
      }
      ++open_vulkan_download_calls;
      ++open_vulkan_cpu_continuation_download_calls;
      open_vulkan_gpu_frame_pending_cpu_readback = false;
      return true;
    };
#endif

    auto mark_open_cuda_cpu_tail = [&](std::string_view stage_id) {
      mark_open_cuda_active_frame();
      ++open_cuda_cpu_tail_stage_calls;
      if (stage_id ==
          studiocast::video::effects::contract::kEffectIdVirtualKeyLight) {
        ++open_cuda_cpu_tail_key_light_calls;
      } else if (stage_id ==
                 studiocast::video::effects::contract::kEffectIdAutoFrame) {
        ++open_cuda_cpu_tail_auto_frame_calls;
      }
    };

    auto count_open_video_denoise_tensor_run =
        [&](const studiocast::open_video::DenoiseTensorRunStats &stats) {
          if (!stats.cuda_ep_active)
            return;

          mark_open_cuda_active_frame();
          open_cuda_denoise_tensor_upload_calls +=
              stats.cuda_tensor_upload_calls;
          open_cuda_denoise_tensor_download_calls +=
              stats.cuda_tensor_download_calls;
          open_cuda_cpu_continuation_download_calls +=
              stats.cuda_tensor_download_calls;
          open_cuda_forced_sync_calls += stats.forced_sync_calls;

          if (stats.cpu_tail_stage_calls > 0) {
            open_cuda_denoise_cpu_tail_calls += stats.cpu_tail_stage_calls;
            open_cuda_cpu_tail_stage_calls += stats.cpu_tail_stage_calls;
          }
        };

    auto mark_open_cuda_auto_frame_cpu_tail =
        [&](OpenCudaAutoFrameContext::TrackingSource tracking_source,
            bool cpu_crop_resize) {
          mark_open_cuda_cpu_tail(
              studiocast::video::effects::contract::kEffectIdAutoFrame);
          switch (tracking_source) {
          case OpenCudaAutoFrameContext::TrackingSource::kFace:
            ++open_cuda_cpu_tail_auto_frame_face_tracking_calls;
            break;
          case OpenCudaAutoFrameContext::TrackingSource::kMatte:
            ++open_cuda_cpu_tail_auto_frame_matte_tracking_calls;
            break;
          default:
            break;
          }
          if (cpu_crop_resize)
            ++open_cuda_cpu_tail_auto_frame_cpu_crop_calls;
        };

    auto mark_maxine_active_frame = [&]() {
      (void)MarkGpuBackendActiveFrame(maxine_active_this_frame,
                                      maxine_active_frames);
    };

    auto count_maxine_stage = [&](bool deferred_readback,
                                  bool runs_green_screen) {
      mark_maxine_active_frame();
      ++maxine_rgb_to_bgr_calls;
      ++maxine_upload_calls;
      if (runs_green_screen) {
        if (maxine_green_screen_calls_this_frame > 0) {
          ++maxine_duplicate_green_screen_calls;
        }
        ++maxine_green_screen_calls;
        ++maxine_green_screen_calls_this_frame;
      }
      if (deferred_readback) {
        return;
      }
      ++maxine_download_calls;
      ++maxine_cpu_continuation_download_calls;
      ++maxine_forced_sync_calls;
      ++maxine_bgr_to_rgb_calls;
    };

    auto accumulate_maxine_resident_telemetry = [&]() {
      if (!maxine_resident_executor)
        return;
      const auto &current = maxine_resident_executor->telemetry();
      const auto delta = ComputeMaxineResidentTelemetryDelta(
          current, maxine_resident_telemetry_snapshot);
      maxine_upload_attempts += delta.upload_attempts;
      if (delta.upload_calls > 0)
        mark_maxine_active_frame();
      maxine_upload_calls += delta.upload_calls;
      maxine_final_download_attempts += delta.final_download_attempts;
      maxine_cpu_continuation_download_attempts +=
          delta.cpu_continuation_download_attempts;
      maxine_final_download_calls += delta.final_download_calls;
      maxine_cpu_continuation_download_calls +=
          delta.cpu_continuation_download_calls;
      maxine_device_bridge_attempts += delta.device_bridge_attempts;
      maxine_device_bridge_calls += delta.device_bridge_calls;
      maxine_download_calls += delta.final_download_calls +
                               delta.cpu_continuation_download_calls;
      maxine_green_screen_calls += delta.matte_inference_calls;
      maxine_matte_inference_attempts += delta.matte_inference_attempts;
      maxine_shared_green_screen_matte_reuse_calls +=
          delta.shared_matte_reuses;
      maxine_forced_sync_calls += delta.sync_calls;
      maxine_forced_sync_attempts += delta.sync_attempts;
      maxine_composite_attempts += delta.composite_attempts;
      maxine_composite_calls += delta.composite_calls;
      maxine_synchronous_sdk_run_attempts +=
          delta.synchronous_sdk_run_attempts;
      maxine_synchronous_sdk_run_calls += delta.synchronous_sdk_run_calls;
      maxine_asynchronous_sdk_run_attempts +=
          delta.asynchronous_sdk_run_attempts;
      maxine_asynchronous_sdk_run_calls +=
          delta.asynchronous_sdk_run_calls;
      for (std::size_t i = 0; i < current.stages.size(); ++i) {
        maxine_stage_attempts[i] += delta.stage_attempts[i];
        maxine_stage_successes[i] += delta.stage_successes[i];
      }
      maxine_rgb_to_bgr_calls += delta.rgb_to_bgr_calls;
      maxine_resident_telemetry_snapshot = current;
    };

    // Open CUDA transfer invariant (pipeline contract)
    //
    // Define "Open CUDA active for this frame" as:
    //   fx.engine == EffectsEnginePreference::open_cuda
    //   AND at least one Open CUDA stage is planned+enabled and will actually
    //   run (virtual background, key light, auto frame, or standalone scaler).
    //
    // Invariant we are moving toward:
    //   - If Open CUDA is active this frame: exactly one CPU->GPU upload at the
    //     start of the Open CUDA section, then GPU-only model/effect stages,
    //     then exactly one GPU->CPU download at the end (including any resize).
    //   - If a CPU tail remains (key light/auto-frame alpha readback and CPU
    //     work), count it explicitly.
    //   - If Open CUDA is NOT active this frame: zero Open CUDA
    //     uploads/downloads.
    //
    // Current transfer points (pre-refactor):
    //   - OpenCudaVirtualBackgroundContext::ApplyRgbInPlace does
    //   UploadFromCpuRgb24(...)
    //     and, unless defer_readback is true, does DownloadToCpuRgb24(...)
    //     inside the stage.
    //   - allow_defer_readback below is currently gated by scaling_needed +
    //   “last stage”
    //     logic so that GPU resize can be combined with a single final
    //     readback.
    //   - OpenCudaGpuScaler can upload+download for resize even when no Open
    //   CUDA
    //     effect stages ran.
    //
    // Avoid mixed modes where an Open CUDA stage performs an internal readback
    // while the pipeline also expects DeferredGpuOut; when Open CUDA stages are
    // active the pipeline must have exactly one coherent upload/readback
    // strategy.

    // Apply effects (in-place on RGB buffer).
    const auto t_effects_start = Clock::now();
    bool fx_failed = false;
    {
      const std::uint64_t published_effects_generation =
          effects_generation_published_.load(std::memory_order_acquire);
      if (effects_preparation_counters.ObservePublishedGeneration(
              published_effects_generation, appliedEffectsGeneration)) {
        studiocast::video::effects::BroadcastCameraEffects next_effects;
        detail::PreparedReplaceBackgroundSource next_replace_source;
        std::uint64_t next_effects_generation = 0;
        {
          std::lock_guard<std::mutex> fxLock(effects_mu_);
          next_effects = effects_;
          next_replace_source = replace_background_source_;
          next_effects_generation = effects_generation_;
        }
        const bool rebuild =
            next_effects_generation != appliedEffectsGeneration;
        effects_preparation_counters.RecordConfigSnapshot(rebuild);
        if (rebuild) {
          rebuildChain(next_effects, next_replace_source,
                       next_effects_generation);
        }
      }

      const auto &fx = appliedFx;

      studiocast::video::effects::Rgb24FrameView view;
      view.data = rgb.data();
      view.width = capA.width;
      view.height = capA.height;
      view.stride_bytes = rgbStride;

      const float vignette_center_x_px = static_cast<float>(capA.width) * 0.5f;
      const float vignette_center_y_px = static_cast<float>(capA.height) * 0.5f;

      const auto &frame_plan = appliedFramePlan;

      const bool vignette_requested =
          frame_plan.Contains(EffectStage::vignette);
      const EffectStage vig_attach = frame_plan.vignette_attachment;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
      open_vulkan_vignette_attempt_this_frame =
          !prepared_hardware_final_live && have_open_vulkan_vignette &&
          vignette_requested &&
          optional_breaker(OptionalEffectSlot::vignette)
              .AllowsAttempt(f.sequence);
      open_vulkan_mirror_attempt_this_frame =
          !prepared_hardware_final_live && have_open_vulkan_mirror &&
          frame_plan.Contains(EffectStage::mirror) &&
          optional_breaker(OptionalEffectSlot::mirror)
              .AllowsAttempt(f.sequence);
#endif

      // Defer GPU->CPU readback for the last GPU stage only when:
      // - output scaling is needed
      // - no CPU tail effects are active
      // This allows us to resize on GPU and perform a single GPU->CPU transfer
      // for output.
      const EffectStage last_stage_for_defer =
          frame_plan.last_deferred_stage;
      const bool scaling_needed =
          (capA.width != outA.width || capA.height != outA.height);
      const bool allow_defer_readback =
          ((scaling_needed && (gpu_backend != GpuResizeBackend::none)) ||
#if STUDIOCAST_ENABLE_OPEN_VULKAN
           open_vulkan_vignette_attempt_this_frame ||
           open_vulkan_mirror_attempt_this_frame ||
#endif
           false) &&
          last_stage_for_defer != EffectStage::none;

      // Define whether any Open CUDA stage is enabled for this frame.
      // Open CUDA stages can optionally use the deferred strategy (no internal
      // readback) when the pipeline can carry a GPU output to final
      // scaling/output without CPU-side continuation.
      const bool open_cuda_any_stage_enabled =
          have_open_cuda_vb &&
          (frame_plan.Contains(EffectStage::virtual_background_blur) ||
           frame_plan.Contains(EffectStage::virtual_background_remove) ||
           frame_plan.Contains(EffectStage::virtual_background_replace));
      const bool open_cuda_auto_frame_stage_enabled =
          have_open_cuda_auto_frame &&
          frame_plan.Contains(EffectStage::auto_frame);

#if STUDIOCAST_ENABLE_OPEN_VULKAN
      const bool open_vulkan_key_light_stage_enabled =
          have_open_vulkan_key_light &&
          frame_plan.Contains(EffectStage::virtual_key_light);
      const bool open_vulkan_auto_frame_stage_enabled =
          have_open_vulkan_auto_frame &&
          frame_plan.Contains(EffectStage::auto_frame);
#endif

      const auto is_open_cuda_stage = [&](EffectStage stage) {
        if (!open_cuda_any_stage_enabled)
          return false;
        return stage == EffectStage::virtual_background_blur ||
               stage == EffectStage::virtual_background_remove ||
               stage == EffectStage::virtual_background_replace;
      };

      const std::uint64_t capture_sequence = f.sequence;
      publish_optional_retry_ready_status(capture_sequence);

      // Apply vignette exactly once, either attached to the planned last GPU
      // stage, or as a standalone GPU stage when no other GPU stage is active.
      const bool maxine_vignette_allowed =
          optional_breaker(OptionalEffectSlot::vignette)
              .AllowsAttempt(capture_sequence);
      const bool apply_vignette_on_eye_contact =
          vignette_requested && maxine_vignette_allowed &&
          vig_attach == EffectStage::eye_contact;
      const bool apply_vignette_on_relight =
          vignette_requested && maxine_vignette_allowed &&
          vig_attach == EffectStage::virtual_key_light;
      const bool apply_vignette_on_bg =
          vignette_requested && maxine_vignette_allowed &&
          (vig_attach == EffectStage::virtual_background_blur ||
           vig_attach == EffectStage::virtual_background_remove ||
           vig_attach == EffectStage::virtual_background_replace);
      const bool apply_vignette_on_auto_frame =
          vignette_requested && maxine_vignette_allowed &&
          vig_attach == EffectStage::auto_frame;

      // Reset per-frame analysis cache (Open Video). This allows effects to
      // share ML outputs without re-running inference multiple times per frame.
      open_video_cache.BeginFrame(capture_sequence);

#if STUDIOCAST_ENABLE_OPEN_VULKAN
      // Explicit Vulkan Auto Frame consumes CPU analysis from the original
      // host capture before the pipeline's one upload. This prevents a GPU
      // readback for face tracking and guarantees that the CPU-only provider
      // policy is rechecked at the actual inference boundary.
      if (open_vulkan_auto_frame_stage_enabled) {
        // Reset discontinuous temporal state before any Vulkan stage can
        // publish a current-frame matte, so reset cannot discard and rerun
        // shared inference later in this frame.
        open_vulkan_auto_frame.PrepareFrame(capture_sequence);
      }
      if (open_vulkan_auto_frame_stage_enabled &&
          open_video_yunet.available() &&
          optional_breaker(OptionalEffectSlot::auto_frame)
              .AllowsAttempt(capture_sequence)) {
        std::string face_error;
        if (open_video_yunet.EnsureDetectionsForFrame(
                rgb.data(), capA.width, capA.height, rgbStride,
                fx.auto_frame.model_id, capture_sequence, &open_video_cache,
                &face_error,
                studiocast::open_video::YunetProviderPolicy::cpu_only)) {
          ++open_vulkan_auto_frame.counters.cpu_face_tracking_calls;
        } else {
          ++open_vulkan_auto_frame.counters.runtime_failure_frames;
          open_vulkan_auto_frame.ResetTrackingState(
              OpenVulkanAutoFrameResetReason::runtime_or_device_failure);
          block_optional_effect(
              OptionalEffectSlot::auto_frame,
              studiocast::video::effects::contract::kEffectIdAutoFrame,
              "open_vulkan",
              OpenVulkanAutoFrameRuntimeFailure(face_error),
              capture_sequence);
        }
      }
#endif

      // Open CUDA optimization: keep the CUDA RGB frame live across adjacent
      // Open CUDA stages so key light can consume the VB matte without a
      // CPU-side alpha download/relight tail.
      bool pending_open_cuda_key_light = false;
      bool open_cuda_vb_ran_this_frame = false;

      auto download_open_cuda_current_to_cpu = [&](std::string *error) -> bool {
        if (error)
          error->clear();
        if (!open_cuda_gpu_frame_pending_cpu_readback) {
          return true;
        }
        if (!open_cuda_curr || !open_cuda_curr->Valid()) {
          if (error)
            *error = "Open CUDA: no current GPU frame to download.";
          return false;
        }

        std::string down_err;
        if (!open_cuda_curr->DownloadToCpuRgb24(
                &open_cuda_vb.cuda, rgb.data(), rgbStride,
                open_cuda_vb.vb_stream, &down_err) ||
            !open_cuda_vb.cuda.StreamSynchronize(open_cuda_vb.vb_stream,
                                                 &down_err)) {
          if (error)
            *error = "Open CUDA: frame download failed: " + down_err;
          return false;
        }

        ++open_cuda_rgb_download_calls;
        ++open_cuda_cpu_continuation_download_calls;
        ++open_cuda_forced_sync_calls;
        open_cuda_gpu_frame_pending_cpu_readback = false;
        return true;
      };

      auto apply_stage = [&](EffectStage stage) {
        const std::string_view stage_id =
            studiocast::video::effects::BroadcastEffectStageId(stage);
        bool defer_readback =
            allow_defer_readback && (stage == last_stage_for_defer);
        if (is_open_cuda_stage(stage) && pending_open_cuda_key_light) {
          // Force immediate readback when we know a CPU stage (key light) must
          // run after this Open CUDA stage.
          defer_readback = false;
        }

        if (stage_id ==
            studiocast::video::effects::contract::kEffectIdVideoNoiseRemoval) {
          if (have_maxine_denoise) {
            auto &breaker = optional_breaker(OptionalEffectSlot::denoise);
            if (!breaker.AllowsAttempt(capture_sequence))
              return;
            std::string mx_err;
            if (!maxine_denoise.ApplyRgbInPlace(rgb.data(), capA.width,
                                                capA.height, rgbStride, fx,
                                                &mx_err)) {
              block_optional_effect(
                  OptionalEffectSlot::denoise, stage_id, "maxine",
                  OptionalEffectFailureReason(
                      "Maxine video noise removal failed", mx_err),
                  capture_sequence);
            } else {
              count_maxine_stage(/*deferred_readback=*/false,
                                 /*runs_green_screen=*/false);
              clear_optional_effect_on_success(breaker, capture_sequence);
              mark_live_effect_success(stage_id, "maxine");
            }
            return;
          }

          if (have_open_video_video_denoise) {
            std::string ov_err;
            const bool ov_ok = open_video_fastdvdnet.ApplyRgbInPlace(
                capture_sequence, rgb.data(), capA.width, capA.height,
                rgbStride, fx.video_noise_removal.strength,
                fx.video_noise_removal.model_id, &ov_err);
            count_open_video_denoise_tensor_run(
                open_video_fastdvdnet.last_tensor_run_stats());
            if (!ov_ok) {
              if (!ov_err.empty()) {
                std::lock_guard<std::mutex> lock(mu_);
                last_error_ =
                    "Open Video video noise removal failed: " + ov_err;
              }

              // Best-effort fallback: run the lightweight Open CUDA temporal
              // denoiser.
              std::string oc_err;
              if (open_cuda_video_denoise.ApplyRgbInPlace(
                      rgb.data(), capA.width, capA.height, rgbStride, fx,
                      &oc_err)) {
                mark_live_effect_success(stage_id, "open_cuda");
              }
            } else {
              mark_live_effect_success(stage_id, "open_video");
            }
            return;
          }

          if (have_open_cuda_video_denoise) {
            std::string oc_err;
            if (!open_cuda_video_denoise.ApplyRgbInPlace(rgb.data(), capA.width,
                                                         capA.height, rgbStride,
                                                         fx, &oc_err)) {
              // Best-effort: keep the pipeline running and simply bypass this
              // stage.
              if (!oc_err.empty()) {
                std::lock_guard<std::mutex> lock(mu_);
                last_error_ = "Open CUDA video noise removal failed: " + oc_err;
              }
            } else {
              mark_live_effect_success(stage_id, "open_cuda");
            }
          }
          return;
        }

        if (stage_id ==
            studiocast::video::effects::contract::kEffectIdEyeContact) {
          if (have_maxine_eye_contact) {
            auto &breaker = optional_breaker(OptionalEffectSlot::eye_contact);
            if (!breaker.AllowsAttempt(capture_sequence))
              return;
            std::string mx_err;
            if (!maxine_eye_contact.ApplyRgbInPlace(
                    rgb.data(), capA.width, capA.height, rgbStride, fx,
                    apply_vignette_on_eye_contact, vignette_center_x_px,
                    vignette_center_y_px, &mx_err, defer_readback,
                    &deferred_gpu_out)) {
              if (IsVignetteFailureReason(mx_err)) {
                clear_optional_effect_on_success(breaker, capture_sequence);
                mark_live_effect_success(stage_id, "maxine_ar");
                block_optional_effect(
                    OptionalEffectSlot::vignette,
                    studiocast::video::effects::contract::kEffectIdVignette,
                    "cuda", mx_err, capture_sequence);
              } else {
                block_optional_effect(OptionalEffectSlot::eye_contact, stage_id,
                                      "maxine_ar",
                                      OptionalEffectFailureReason(
                                          "Maxine eye contact failed", mx_err),
                                      capture_sequence);
              }
            } else {
              count_maxine_stage(
                  /*deferred_readback=*/
                  defer_readback &&
                      deferred_gpu_out.kind != DeferredGpuKind::none,
                  /*runs_green_screen=*/false);
              clear_optional_effect_on_success(breaker, capture_sequence);
              mark_live_effect_success(stage_id, "maxine_ar");
              if (apply_vignette_on_eye_contact) {
                mark_live_effect_success(
                    studiocast::video::effects::contract::kEffectIdVignette,
                    "maxine_ar");
              }
            }
            if (defer_readback &&
                deferred_gpu_out.kind != DeferredGpuKind::none)
              have_deferred_gpu_out = true;
            return;
          }

          if (have_open_video_eye_contact) {
            std::string ov_err;
            if (!open_video_eye_contact.ApplyRgbInPlace(
                    capture_sequence, rgb.data(), capA.width, capA.height,
                    rgbStride, fx.eye_contact.strength,
                    fx.eye_contact.look_away_enabled, fx.auto_frame.model_id,
                    fx.eye_contact.model_id, &open_video_yunet,
                    &open_video_cache, &ov_err)) {
              // Best-effort: keep the pipeline running and simply bypass this
              // stage.
              if (!ov_err.empty()) {
                std::lock_guard<std::mutex> lock(mu_);
                last_error_ = "Open Video eye contact failed: " + ov_err;
              }
            } else {
              mark_live_effect_success(stage_id, "open_video");
            }
            return;
          }
          return;
        }

        if (stage_id ==
            studiocast::video::effects::contract::kEffectIdVirtualKeyLight) {
          if (have_maxine_relight) {
            auto &breaker = optional_breaker(OptionalEffectSlot::relight);
            if (!breaker.AllowsAttempt(capture_sequence))
              return;
            auto relight_green_screen_config =
                maxine_relight.greenscreen
                    ? maxine_relight.greenscreen->config()
                    : studiocast::maxine::effects::VfxGreenScreenEffect::
                          Config{};
            relight_green_screen_config.mode =
                fx.virtual_background.greenscreen_mode;
            relight_green_screen_config.temporal =
                fx.virtual_background.greenscreen_temporal;
            if (relight_green_screen_config.temporal &&
                relight_green_screen_config.state_count == 0) {
              relight_green_screen_config.state_count = 1;
            }
            const auto relight_stream =
                maxine_relight.relight ? maxine_relight.relight->cuda_stream()
                                       : nullptr;
            const auto matte_reuse_reason =
                maxine_shared_handles.ValidateMatteForConsumer(
                    capture_sequence, capA.width, capA.height,
                    &maxine_relight.nvcv, relight_stream,
                    relight_green_screen_config);
            const bool can_reuse_green_screen_matte =
                matte_reuse_reason ==
                MaxineFrameLocalSharedHandles::RejectReason::none;
            if (maxine_shared_handles.valid &&
                matte_reuse_reason !=
                    MaxineFrameLocalSharedHandles::RejectReason::none) {
              ++maxine_shared_green_screen_matte_incompatible_calls;
            }
            const auto input_reuse_reason =
                maxine_shared_handles.ValidateInputForCurrentRgbConsumer(
                    capture_sequence, capA.width, capA.height,
                    &maxine_relight.nvcv, relight_stream);
            if (maxine_shared_handles.valid &&
                input_reuse_reason !=
                    MaxineFrameLocalSharedHandles::RejectReason::none) {
              ++maxine_shared_green_screen_input_incompatible_calls;
            }
            bool used_shared_green_screen_matte = false;
            std::string mx_err;
            if (!maxine_relight.ApplyRgbInPlace(
                    rgb.data(), capA.width, capA.height, rgbStride, fx,
                    apply_vignette_on_relight, vignette_center_x_px,
                    vignette_center_y_px,
                    can_reuse_green_screen_matte
                        ? maxine_shared_handles.matte_gpu
                        : nullptr,
                    &used_shared_green_screen_matte, &mx_err, defer_readback,
                    &deferred_gpu_out)) {
              if (IsVignetteFailureReason(mx_err)) {
                clear_optional_effect_on_success(breaker, capture_sequence);
                mark_live_effect_success(stage_id, "maxine");
                block_optional_effect(
                    OptionalEffectSlot::vignette,
                    studiocast::video::effects::contract::kEffectIdVignette,
                    "cuda", mx_err, capture_sequence);
              } else {
                block_optional_effect(OptionalEffectSlot::relight, stage_id,
                                      "maxine",
                                      OptionalEffectFailureReason(
                                          "Maxine relighting failed", mx_err),
                                      capture_sequence);
              }
            } else {
              if (used_shared_green_screen_matte) {
                ++maxine_shared_green_screen_matte_reuse_calls;
              }
              count_maxine_stage(
                  /*deferred_readback=*/
                  defer_readback &&
                      deferred_gpu_out.kind != DeferredGpuKind::none,
                  /*runs_green_screen=*/!used_shared_green_screen_matte);
              clear_optional_effect_on_success(breaker, capture_sequence);
              mark_live_effect_success(stage_id, "maxine");
              if (apply_vignette_on_relight) {
                mark_live_effect_success(
                    studiocast::video::effects::contract::kEffectIdVignette,
                    "maxine");
              }
            }
            if (defer_readback &&
                deferred_gpu_out.kind != DeferredGpuKind::none)
              have_deferred_gpu_out = true;
            return;
          }

#if STUDIOCAST_ENABLE_OPEN_VULKAN
          if (have_open_vulkan_key_light) {
            auto &key_light_breaker =
                optional_breaker(OptionalEffectSlot::relight);
            if (!key_light_breaker.AllowsAttempt(capture_sequence))
              return;
            std::string vk_err;
            if (!ensure_open_vulkan_current_from_cpu(&vk_err)) {
              ++open_vulkan_runtime_failure_frames;
              ++open_vulkan_vb.key_light_effect_counters.runtime_failure_frames;
              if (open_vulkan_vb.kernels.device() &&
                  open_vulkan_vb.kernels.device()->health().health ==
                      studiocast::vulkan::VulkanContextHealth::device_lost) {
                ++open_vulkan_vb.key_light_effect_counters.device_loss_frames;
              }
              block_optional_effect(
                  OptionalEffectSlot::relight, stage_id, "open_vulkan",
                  OpenVulkanVirtualKeyLightRuntimeFailure(vk_err),
                  capture_sequence);
              return;
            }

            ensure_open_vulkan_next();

            const std::uint64_t key_light_failures_before =
                open_vulkan_vb.key_light_effect_counters.runtime_failure_frames;
            const std::uint64_t key_light_device_losses_before =
                open_vulkan_vb.key_light_effect_counters.device_loss_frames;
            if (open_vulkan_key_light.ApplyVulkanRgb(
                    *open_vulkan_curr, open_vulkan_next, fx, capture_sequence,
                    &vk_err, &deferred_gpu_out)) {
              open_vulkan_effect_ran_this_frame = true;
              adopt_open_vulkan_output(deferred_gpu_out.vulkan_img);
              open_vulkan_gpu_frame_pending_cpu_readback = true;

              const bool keep_gpu_for_auto_frame =
                  open_vulkan_auto_frame_stage_enabled &&
                  frame_plan.AppearsAfter(stage, EffectStage::auto_frame);
              if (defer_readback) {
                have_deferred_gpu_out = true;
                clear_optional_effect_on_success(key_light_breaker,
                                                 capture_sequence);
                mark_live_effect_success(stage_id, "open_vulkan");
                return;
              }
              if (keep_gpu_for_auto_frame) {
                clear_optional_effect_on_success(key_light_breaker,
                                                 capture_sequence);
                mark_live_effect_success(stage_id, "open_vulkan");
                return;
              }
              std::string down_err;
              if (!download_open_vulkan_current_to_cpu(&down_err)) {
                ++open_vulkan_runtime_failure_frames;
                ++open_vulkan_vb.key_light_effect_counters
                      .runtime_failure_frames;
                if (open_vulkan_vb.kernels.device() &&
                    open_vulkan_vb.kernels.device()->health().health ==
                        studiocast::vulkan::VulkanContextHealth::device_lost) {
                  ++open_vulkan_vb.key_light_effect_counters.device_loss_frames;
                }
                block_optional_effect(
                    OptionalEffectSlot::relight, stage_id, "open_vulkan",
                    OpenVulkanVirtualKeyLightRuntimeFailure(down_err),
                    capture_sequence);
              } else {
                clear_optional_effect_on_success(key_light_breaker,
                                                 capture_sequence);
                mark_live_effect_success(stage_id, "open_vulkan");
              }
              return;
            }

            ++open_vulkan_runtime_failure_frames;
            if (open_vulkan_vb.key_light_effect_counters
                    .runtime_failure_frames == key_light_failures_before) {
              ++open_vulkan_vb.key_light_effect_counters.runtime_failure_frames;
            }
            if (open_vulkan_vb.kernels.device() &&
                open_vulkan_vb.kernels.device()->health().health ==
                    studiocast::vulkan::VulkanContextHealth::device_lost &&
                open_vulkan_vb.key_light_effect_counters.device_loss_frames ==
                    key_light_device_losses_before) {
              ++open_vulkan_vb.key_light_effect_counters.device_loss_frames;
            }
            block_optional_effect(OptionalEffectSlot::relight, stage_id,
                                  "open_vulkan", vk_err, capture_sequence);
            return;
          }
#endif

          if (have_open_cuda_key_light) {
            // If an earlier Open CUDA stage has kept the RGB frame and matte
            // on GPU, run key light inside that GPU section. This avoids the
            // key-light alpha download and CPU relight tail.
            const bool can_apply_key_light_on_gpu =
                open_cuda_gpu_frame_pending_cpu_readback &&
                open_cuda_uploaded_this_frame && open_cuda_curr &&
                open_cuda_next && open_cuda_curr->Valid() &&
                open_cuda_curr->format ==
                    studiocast::cuda::PixelFormatGpu::rgb_u8 &&
                open_cuda_curr->w == capA.width &&
                open_cuda_curr->h == capA.height &&
                open_cuda_vb.cached_matte_valid &&
                open_cuda_vb.cached_matte_sequence == capture_sequence &&
                fx.virtual_key_light.intensity > 0;
            if (can_apply_key_light_on_gpu) {
              std::string kl_err;
              if (open_cuda_key_light.ApplyCudaRgb(
                      *open_cuda_curr, open_cuda_next, fx, capture_sequence,
                      &kl_err, &deferred_gpu_out)) {
                const bool curr_was_a = (open_cuda_curr == &open_cuda_frame_a);
                open_cuda_curr = open_cuda_next;
                open_cuda_next =
                    curr_was_a ? &open_cuda_frame_a : &open_cuda_frame_b;

                if (defer_readback) {
                  have_deferred_gpu_out = true;
                  open_cuda_gpu_frame_pending_cpu_readback = true;
                  mark_live_effect_success(stage_id, "open_cuda");
                  return;
                }

                std::string down_err;
                open_cuda_gpu_frame_pending_cpu_readback = true;
                if (!download_open_cuda_current_to_cpu(&down_err)) {
                  {
                    std::lock_guard<std::mutex> lock(mu_);
                    last_error_ =
                        "Open CUDA virtual key light failed: " + down_err;
                  }
                  if (studiocast::video::effects::
                          ShouldAbortPipelineOnOpenCudaVbApplyFailure()) {
                    fx_failed = true;
                  }
                } else {
                  mark_live_effect_success(stage_id, "open_cuda");
                }
                return;
              }

              // Fall through to the CPU implementation after first making the
              // GPU VB output visible to the CPU buffer.
              std::string down_err;
              if (open_cuda_gpu_frame_pending_cpu_readback &&
                  !download_open_cuda_current_to_cpu(&down_err)) {
                {
                  std::lock_guard<std::mutex> lock(mu_);
                  last_error_ =
                      "Open CUDA virtual key light failed: " + kl_err +
                      "; fallback download failed: " + down_err;
                }
                if (studiocast::video::effects::
                        ShouldAbortPipelineOnOpenCudaVbApplyFailure()) {
                  fx_failed = true;
                }
                return;
              }
            }

            // If Open CUDA VB is also enabled but has not run yet, defer key
            // light until after VB so the upload/matte can be reused.
            if (open_cuda_any_stage_enabled && !open_cuda_vb_ran_this_frame) {
              pending_open_cuda_key_light = true;
              return;
            }

            if (open_cuda_gpu_frame_pending_cpu_readback) {
              std::string down_err;
              if (!download_open_cuda_current_to_cpu(&down_err)) {
                {
                  std::lock_guard<std::mutex> lock(mu_);
                  last_error_ =
                      "Open CUDA virtual key light failed: " + down_err;
                }
                if (studiocast::video::effects::
                        ShouldAbortPipelineOnOpenCudaVbApplyFailure()) {
                  fx_failed = true;
                }
                return;
              }
            }

            mark_open_cuda_cpu_tail(stage_id);
            std::string oc_err;
            if (!open_cuda_key_light.ApplyRgbInPlace(
                    capture_sequence, rgb.data(), capA.width, capA.height,
                    rgbStride, fx, &oc_err)) {
              {
                std::lock_guard<std::mutex> lock(mu_);
                last_error_ = "Open CUDA virtual key light failed: " + oc_err;
              }

              // Keep the pipeline alive; bypass on failures.
              if (studiocast::video::effects::
                      ShouldAbortPipelineOnOpenCudaVbApplyFailure()) {
                fx_failed = true;
              }
            } else {
              mark_live_effect_success(stage_id, "open_cuda");
            }
          }
          return;
        }

        if (stage_id == studiocast::video::effects::contract::
                            kEffectIdVirtualBackgroundBlur ||
            stage_id == studiocast::video::effects::contract::
                            kEffectIdVirtualBackgroundRemove ||
            stage_id == studiocast::video::effects::contract::
                            kEffectIdVirtualBackgroundReplace) {
#if STUDIOCAST_ENABLE_OPEN_VULKAN
          if (have_open_vulkan_vb) {
            const bool is_vulkan_blur =
                stage_id == studiocast::video::effects::contract::
                                kEffectIdVirtualBackgroundBlur;
            const bool is_vulkan_remove =
                stage_id == studiocast::video::effects::contract::
                                kEffectIdVirtualBackgroundRemove;
            const bool is_vulkan_replace =
                stage_id == studiocast::video::effects::contract::
                                kEffectIdVirtualBackgroundReplace;
            const bool is_wrapped_vulkan_vb =
                is_vulkan_blur || is_vulkan_remove || is_vulkan_replace;
            const auto wrap_vulkan_vb_runtime_failure =
                [&](std::string_view detail) {
                  if (is_vulkan_blur) {
                    return OpenVulkanVirtualBackgroundBlurRuntimeFailure(
                        detail);
                  }
                  if (is_vulkan_remove) {
                    return OpenVulkanVirtualBackgroundRemoveRuntimeFailure(
                        detail);
                  }
                  return OpenVulkanVirtualBackgroundReplaceRuntimeFailure(
                      detail);
                };
            auto &vulkan_vb_breaker =
                optional_breaker(OptionalEffectSlot::virtual_background);
            if (is_wrapped_vulkan_vb &&
                !vulkan_vb_breaker.AllowsAttempt(capture_sequence)) {
              return;
            }
            std::string vk_err;
            if (!ensure_open_vulkan_current_from_cpu(&vk_err)) {
              ++open_vulkan_runtime_failure_frames;
              if (is_wrapped_vulkan_vb) {
                block_optional_effect(
                    OptionalEffectSlot::virtual_background, stage_id,
                    "open_vulkan", wrap_vulkan_vb_runtime_failure(vk_err),
                    capture_sequence);
              } else {
                std::lock_guard<std::mutex> lock(mu_);
                last_error_ =
                    "Open Vulkan virtual background failed: " + vk_err;
              }
              return;
            }

            ensure_open_vulkan_next();

            if (!open_vulkan_vb.ApplyVulkanRgb(
                    *open_vulkan_curr, open_vulkan_next, fx, capture_sequence,
                    &vk_err, &deferred_gpu_out)) {
              ++open_vulkan_runtime_failure_frames;
              if (is_wrapped_vulkan_vb) {
                block_optional_effect(OptionalEffectSlot::virtual_background,
                                      stage_id, "open_vulkan", vk_err,
                                      capture_sequence);
              } else {
                std::lock_guard<std::mutex> lock(mu_);
                last_error_ =
                    "Open Vulkan virtual background failed: " + vk_err;
              }
              return;
            }
            open_vulkan_effect_ran_this_frame = true;

            adopt_open_vulkan_output(deferred_gpu_out.vulkan_img);
            open_vulkan_gpu_frame_pending_cpu_readback = true;

            const bool keep_gpu_frame_for_key_light =
                open_vulkan_key_light_stage_enabled &&
                frame_plan.AppearsAfter(stage,
                                        EffectStage::virtual_key_light);
            const bool keep_gpu_frame_for_auto_frame =
                open_vulkan_auto_frame_stage_enabled &&
                frame_plan.AppearsAfter(stage, EffectStage::auto_frame);
            if (defer_readback || keep_gpu_frame_for_key_light ||
                keep_gpu_frame_for_auto_frame) {
              if (is_wrapped_vulkan_vb) {
                clear_optional_effect_on_success(vulkan_vb_breaker,
                                                 capture_sequence);
                mark_live_effect_success(stage_id, "open_vulkan");
              }
              have_deferred_gpu_out = defer_readback;
              return;
            }

            std::string down_err;
            if (!download_open_vulkan_current_to_cpu(&down_err)) {
              ++open_vulkan_runtime_failure_frames;
              if (is_wrapped_vulkan_vb) {
                block_optional_effect(
                    OptionalEffectSlot::virtual_background, stage_id,
                    "open_vulkan", wrap_vulkan_vb_runtime_failure(down_err),
                    capture_sequence);
              } else {
                std::lock_guard<std::mutex> lock(mu_);
                last_error_ =
                    "Open Vulkan virtual background failed: " + down_err;
              }
            } else if (is_wrapped_vulkan_vb) {
              clear_optional_effect_on_success(vulkan_vb_breaker,
                                               capture_sequence);
              mark_live_effect_success(stage_id, "open_vulkan");
            }
            return;
          }
#endif

          if (have_maxine_bg_blur) {
            auto &breaker =
                optional_breaker(OptionalEffectSlot::virtual_background);
            if (!breaker.AllowsAttempt(capture_sequence))
              return;
            std::string mx_err;
            if (!maxine_bg_blur.ApplyRgbInPlace(
                    rgb.data(), capA.width, capA.height, rgbStride, fx,
                    apply_vignette_on_bg, vignette_center_x_px,
                    vignette_center_y_px, &mx_err, defer_readback,
                    &deferred_gpu_out)) {
              if (IsVignetteFailureReason(mx_err)) {
                clear_optional_effect_on_success(breaker, capture_sequence);
                mark_live_effect_success(stage_id, "maxine");
                block_optional_effect(
                    OptionalEffectSlot::vignette,
                    studiocast::video::effects::contract::kEffectIdVignette,
                    "cuda", mx_err, capture_sequence);
              } else {
                block_optional_effect(
                    OptionalEffectSlot::virtual_background, stage_id, "maxine",
                    OptionalEffectFailureReason(
                        "Maxine virtual background failed", mx_err),
                    capture_sequence);
              }
            } else {
              count_maxine_stage(
                  /*deferred_readback=*/
                  defer_readback &&
                      deferred_gpu_out.kind != DeferredGpuKind::none,
                  /*runs_green_screen=*/true);
              if (const auto *matte = maxine_bg_blur.GreenScreenMatteGpu()) {
                maxine_shared_handles.PublishGreenScreen(
                    "virtual_background", capture_sequence, capA.width,
                    capA.height, &maxine_bg_blur.nvcv, &maxine_bg_blur.cuda,
                    maxine_bg_blur.GreenScreenStream(),
                    maxine_bg_blur.GreenScreenInputGpu(), matte,
                    maxine_bg_blur.GreenScreenConfig(),
                    /*input_is_current_rgb=*/false,
                    /*matte_ready_cross_stream=*/true);
              }
              clear_optional_effect_on_success(breaker, capture_sequence);
              mark_live_effect_success(stage_id, "maxine");
              if (apply_vignette_on_bg) {
                mark_live_effect_success(
                    studiocast::video::effects::contract::kEffectIdVignette,
                    "maxine");
              }
            }
            if (defer_readback &&
                deferred_gpu_out.kind != DeferredGpuKind::none)
              have_deferred_gpu_out = true;
            return;
          }

          if (have_open_cuda_vb) {
            open_cuda_vb_ran_this_frame = true;
            if (prepared_open_cuda_hardware_live) {
              // The prepared facade is the sole live executor for this
              // qualified generation. Never retry through the legacy context
              // in the same frame.
              if (prepared_open_cuda_hardware_suspended)
                return;
              auto &breaker =
                  optional_breaker(OptionalEffectSlot::virtual_background);
              if (!breaker.AllowsAttempt(capture_sequence))
                return;
              const std::size_t tight_stride =
                  static_cast<std::size_t>(capA.width) * 3u;
              const std::size_t tight_bytes =
                  tight_stride * static_cast<std::size_t>(capA.height);
              const auto prepared_frame =
                  prepared_open_cuda_hardware_owner.ExecuteHostFrame(
                      rgb.data(), rgb.size(), rgbStride,
                      prepared_open_cuda_rgb_output.data(), tight_bytes,
                      tight_stride, capture_sequence);
              publish_prepared_open_cuda_status();
              prepared_open_cuda_completion = prepared_frame.completion;
              prepared_open_cuda_completion_pending =
                  prepared_frame.completion_pending;
              const bool exact_transfer_shape =
                  prepared_frame.transfers.uploads == 1 &&
                  prepared_frame.transfers.final_readbacks == 1 &&
                  prepared_frame.transfers.intermediate_uploads == 0 &&
                  prepared_frame.transfers.intermediate_readbacks == 0 &&
                  prepared_frame.transfers.synchronizations == 1 &&
                  prepared_frame.transfers.cpu_tail_boundaries == 1;
              prepared_open_cuda_output_ready =
                  prepared_frame.output_ready && exact_transfer_shape &&
                  prepared_frame.output ==
                      prepared_open_cuda_rgb_output.data() &&
                  prepared_frame.output_size_bytes == tight_bytes &&
                  prepared_frame.output_row_stride_bytes == tight_stride;
              if (prepared_open_cuda_output_ready) {
                open_cuda_frame_upload_calls +=
                    prepared_frame.transfers.uploads;
                open_cuda_rgb_download_calls +=
                    prepared_frame.transfers.final_readbacks;
                open_cuda_final_download_calls +=
                    prepared_frame.transfers.final_readbacks;
                open_cuda_forced_sync_calls +=
                    prepared_frame.transfers.synchronizations;
                open_cuda_cpu_tail_stage_calls +=
                    prepared_frame.transfers.cpu_tail_boundaries;
                return;
              }
              if (prepared_open_cuda_completion_pending) {
                hal::ReasonId commit_reason = hal::ReasonId::none;
                (void)prepared_open_cuda_hardware_owner.CommitOutput(
                    prepared_open_cuda_completion, false, &commit_reason);
                prepared_open_cuda_completion_pending = false;
              }
              if (prepared_frame.reason == hal::ReasonId::device_lost) {
                prepared_open_cuda_hardware_owner.ObserveDeviceLoss();
              }
              prepared_open_cuda_hardware_suspended =
                  PreparedOpenCudaPipelineShouldSuspend(
                      prepared_frame.status, prepared_frame.reason);
              publish_prepared_open_cuda_status();
              publish_video_hal_status(
                  appliedFx, appliedEffectsGeneration,
                  VideoRuntimeLifecycle::running);
              block_optional_effect(
                  OptionalEffectSlot::virtual_background, stage_id,
                  "open_cuda",
                  "Prepared Open CUDA HAL runtime failure [" +
                      std::string(
                          PreparedHardwareReasonCode(prepared_frame.reason)) +
                      "].",
                  capture_sequence);
              return;
            }
            // If the Open CUDA key light stage was deferred earlier in the
            // chain (to avoid a redundant CPU->GPU upload for matting), apply
            // it here immediately after the Open CUDA VB stage finishes (or is
            // bypassed) so downstream stage ordering is preserved.
            const auto apply_deferred_open_cuda_key_light = [&]() {
              if (!pending_open_cuda_key_light)
                return;
              pending_open_cuda_key_light = false;
              if (!have_open_cuda_key_light)
                return;

              const auto fail_key_light = [&](const std::string &message) {
                {
                  std::lock_guard<std::mutex> lock(mu_);
                  last_error_ = message;
                }
                if (studiocast::video::effects::
                        ShouldAbortPipelineOnOpenCudaVbApplyFailure()) {
                  fx_failed = true;
                }
              };

              const auto apply_cpu_key_light_tail = [&]() {
                if (open_cuda_gpu_frame_pending_cpu_readback) {
                  std::string down_err;
                  if (!download_open_cuda_current_to_cpu(&down_err)) {
                    fail_key_light("Open CUDA virtual key light failed: " +
                                   down_err);
                    return;
                  }
                }

                mark_open_cuda_cpu_tail(studiocast::video::effects::contract::
                                            kEffectIdVirtualKeyLight);
                std::string kl_err;
                if (!open_cuda_key_light.ApplyRgbInPlace(
                        capture_sequence, rgb.data(), capA.width, capA.height,
                        rgbStride, fx, &kl_err)) {
                  fail_key_light("Open CUDA virtual key light failed: " +
                                 kl_err);
                } else {
                  mark_live_effect_success(
                      studiocast::video::effects::contract::
                          kEffectIdVirtualKeyLight,
                      "open_cuda");
                }
              };

              const bool can_apply_key_light_on_gpu =
                  open_cuda_gpu_frame_pending_cpu_readback &&
                  open_cuda_uploaded_this_frame && open_cuda_curr &&
                  open_cuda_next && open_cuda_curr->Valid() &&
                  open_cuda_curr->format ==
                      studiocast::cuda::PixelFormatGpu::rgb_u8 &&
                  open_cuda_curr->w == capA.width &&
                  open_cuda_curr->h == capA.height &&
                  open_cuda_vb.cached_matte_valid &&
                  open_cuda_vb.cached_matte_sequence == capture_sequence &&
                  fx.virtual_key_light.intensity > 0;
              if (can_apply_key_light_on_gpu) {
                std::string kl_err;
                if (open_cuda_key_light.ApplyCudaRgb(
                        *open_cuda_curr, open_cuda_next, fx, capture_sequence,
                        &kl_err, &deferred_gpu_out)) {
                  const bool curr_was_a =
                      (open_cuda_curr == &open_cuda_frame_a);
                  open_cuda_curr = open_cuda_next;
                  open_cuda_next =
                      curr_was_a ? &open_cuda_frame_a : &open_cuda_frame_b;
                  open_cuda_gpu_frame_pending_cpu_readback = true;

                  const bool keep_gpu_for_auto_frame =
                      open_cuda_auto_frame_stage_enabled &&
                      frame_plan.AppearsAfter(stage,
                                              EffectStage::auto_frame);
                  const bool keep_gpu_for_final =
                      allow_defer_readback &&
                      (stage == last_stage_for_defer);
                  if (keep_gpu_for_auto_frame || keep_gpu_for_final) {
                    have_deferred_gpu_out = true;
                    mark_live_effect_success(
                        studiocast::video::effects::contract::
                            kEffectIdVirtualKeyLight,
                        "open_cuda");
                    return;
                  }

                  std::string down_err;
                  if (!download_open_cuda_current_to_cpu(&down_err)) {
                    fail_key_light("Open CUDA virtual key light failed: " +
                                   down_err);
                  } else {
                    mark_live_effect_success(
                        studiocast::video::effects::contract::
                            kEffectIdVirtualKeyLight,
                        "open_cuda");
                  }
                  have_deferred_gpu_out = false;
                  return;
                }

                if (open_cuda_gpu_frame_pending_cpu_readback) {
                  std::string down_err;
                  if (!download_open_cuda_current_to_cpu(&down_err)) {
                    fail_key_light(
                        "Open CUDA virtual key light failed: " + kl_err +
                        "; fallback download failed: " + down_err);
                    return;
                  }
                  have_deferred_gpu_out = false;
                }
              }

              apply_cpu_key_light_tail();
            };

            const auto apply_deferred_open_cuda_key_light_cpu_only = [&]() {
              if (!pending_open_cuda_key_light)
                return;
              pending_open_cuda_key_light = false;
              if (!have_open_cuda_key_light)
                return;

              mark_open_cuda_cpu_tail(studiocast::video::effects::contract::
                                          kEffectIdVirtualKeyLight);
              std::string kl_err;
              if (!open_cuda_key_light.ApplyRgbInPlace(
                      capture_sequence, rgb.data(), capA.width, capA.height,
                      rgbStride, fx, &kl_err)) {
                {
                  std::lock_guard<std::mutex> lock(mu_);
                  last_error_ = "Open CUDA virtual key light failed: " + kl_err;
                }

                // Keep the pipeline alive; bypass on failures.
                if (studiocast::video::effects::
                        ShouldAbortPipelineOnOpenCudaVbApplyFailure()) {
                  fx_failed = true;
                }
              } else {
                mark_live_effect_success(studiocast::video::effects::contract::
                                             kEffectIdVirtualKeyLight,
                                         "open_cuda");
              }
            };

            // Open CUDA VB does not support vignette attachment yet.
            if (apply_vignette_on_bg) {
              {
                std::lock_guard<std::mutex> lock(mu_);
                last_error_ = "Open CUDA virtual background does not support "
                              "vignette yet.";
              }
              apply_deferred_open_cuda_key_light_cpu_only();
              return;
            }

            std::string oc_err;

            const bool defer_for_open_cuda_auto_frame =
                open_cuda_auto_frame_stage_enabled &&
                frame_plan.AppearsAfter(stage, EffectStage::auto_frame) &&
                !pending_open_cuda_key_light;
            if (defer_for_open_cuda_auto_frame) {
              defer_readback = true;
            }

            // Pipeline-level Open CUDA transfer counters.
            // This matches the transfer invariant we're moving toward: one
            // upload at the start of the Open CUDA section and one download at
            // the end (possibly deferred).
            mark_open_cuda_active_frame();

            // Upload once at the start of the Open CUDA section (per-frame),
            // then keep the frame in GPU memory across Open CUDA stages.
            if (!open_cuda_uploaded_this_frame) {
              std::string init_err;
              if (!open_cuda_vb.EnsureInitialized(capA.width, capA.height, fx,
                                                  &init_err)) {
                oc_err = init_err;
              } else {
                std::string berr;
                if (!open_cuda_frame_a.ReallocIfNeeded(
                        &open_cuda_vb.cuda, capA.width, capA.height,
                        studiocast::cuda::PixelFormatGpu::rgb_u8, &berr) ||
                    !open_cuda_frame_b.ReallocIfNeeded(
                        &open_cuda_vb.cuda, capA.width, capA.height,
                        studiocast::cuda::PixelFormatGpu::rgb_u8, &berr)) {
                  oc_err = "Open CUDA: failed to allocate ping-pong frame "
                           "buffers: " +
                           berr;
                } else if (!open_cuda_frame_a.UploadFromCpuRgb24(
                               &open_cuda_vb.cuda, rgb.data(), rgbStride,
                               open_cuda_vb.vb_stream, &berr)) {
                  oc_err = "Open CUDA: frame upload failed: " + berr;
                } else {
                  open_cuda_curr = &open_cuda_frame_a;
                  open_cuda_next = &open_cuda_frame_b;
                  open_cuda_uploaded_this_frame = true;
                  ++open_cuda_frame_upload_calls;
                }
              }

              if (!open_cuda_uploaded_this_frame) {
                {
                  std::lock_guard<std::mutex> lock(mu_);
                  last_error_ =
                      "Open CUDA virtual background failed: " + oc_err;
                }

                // Safety net: Open CUDA VB failures are treated as non-fatal so
                // we keep producing pass-through frames rather than blacking
                // out the stream.
                if (studiocast::video::effects::
                        ShouldAbortPipelineOnOpenCudaVbApplyFailure()) {
                  fx_failed = true;
                }

                apply_deferred_open_cuda_key_light();
                return;
              }
            }

            if (!open_cuda_vb.ApplyCudaRgb(*open_cuda_curr, open_cuda_next, fx,
                                           capture_sequence, &oc_err,
                                           &deferred_gpu_out)) {
              {
                std::lock_guard<std::mutex> lock(mu_);
                last_error_ = "Open CUDA virtual background failed: " + oc_err;
              }

              // Safety net: Open CUDA VB failures are treated as non-fatal so
              // we keep producing pass-through frames rather than blacking out
              // the stream.
              if (studiocast::video::effects::
                      ShouldAbortPipelineOnOpenCudaVbApplyFailure()) {
                fx_failed = true;
              }

              apply_deferred_open_cuda_key_light();
              return;
            }

            mark_live_effect_success(stage_id, "open_cuda");

            // Ping-pong swap.
            const bool curr_was_a = (open_cuda_curr == &open_cuda_frame_a);
            open_cuda_curr = open_cuda_next;
            open_cuda_next =
                curr_was_a ? &open_cuda_frame_a : &open_cuda_frame_b;

            if (pending_open_cuda_key_light) {
              open_cuda_gpu_frame_pending_cpu_readback = true;
              apply_deferred_open_cuda_key_light();
              return;
            }

            const bool keep_gpu_frame_for_key_light =
                have_open_cuda_key_light && fx.virtual_key_light.enabled &&
                fx.virtual_key_light.intensity > 0 &&
                frame_plan.AppearsAfter(stage,
                                        EffectStage::virtual_key_light);
            if (defer_readback) {
              have_deferred_gpu_out = true;
              open_cuda_gpu_frame_pending_cpu_readback = true;
              return;
            }
            if (keep_gpu_frame_for_key_light) {
              open_cuda_gpu_frame_pending_cpu_readback = true;
              return;
            }

            // Download GPU->CPU for any CPU-side continuation.
            std::string down_err;
            if (!open_cuda_curr->DownloadToCpuRgb24(
                    &open_cuda_vb.cuda, rgb.data(), rgbStride,
                    open_cuda_vb.vb_stream, &down_err) ||
                !open_cuda_vb.cuda.StreamSynchronize(open_cuda_vb.vb_stream,
                                                     &down_err)) {
              {
                std::lock_guard<std::mutex> lock(mu_);
                last_error_ = "Open CUDA virtual background failed: Open CUDA: "
                              "frame download failed: " +
                              down_err;
              }

              // Treat download failures consistently with stage failures.
              if (studiocast::video::effects::
                      ShouldAbortPipelineOnOpenCudaVbApplyFailure()) {
                fx_failed = true;
              }

              apply_deferred_open_cuda_key_light();
              remove_live_effect(stage_id);
              return;
            }

            ++open_cuda_rgb_download_calls;
            ++open_cuda_cpu_continuation_download_calls;
            ++open_cuda_forced_sync_calls;

            apply_deferred_open_cuda_key_light();
            return;
          }

          return;
        }

        if (stage_id ==
            studiocast::video::effects::contract::kEffectIdAutoFrame) {
#if STUDIOCAST_ENABLE_OPEN_VULKAN
          if (have_open_vulkan_auto_frame) {
            auto &breaker = optional_breaker(OptionalEffectSlot::auto_frame);
            if (!breaker.AllowsAttempt(capture_sequence))
              return;
            OpenCudaAutoFrameContext::TrackingBox tracking_box;
            const OpenCudaAutoFrameContext::TrackingBox *tracking_box_ptr =
                nullptr;
            bool tracking_provider_ran = false;
            auto tracking_tail_source =
                OpenCudaAutoFrameContext::TrackingSource::kMatte;

            const auto adopt_face_detections =
                [&](const std::vector<studiocast::open_video::FaceDetection>
                        &detections) {
                  tracking_provider_ran = true;
                  tracking_tail_source =
                      OpenCudaAutoFrameContext::TrackingSource::kFace;
                  if (OpenCudaAutoFrameContext::BuildBestFaceTrackingBox(
                          detections, &tracking_box)) {
                    tracking_box_ptr = &tracking_box;
                  }
                };

            if (open_video_cache.face_detections) {
              adopt_face_detections(*open_video_cache.face_detections);
            }

            std::string vk_err;
            if (!ensure_open_vulkan_current_from_cpu(&vk_err)) {
              ++open_vulkan_runtime_failure_frames;
              ++open_vulkan_auto_frame.counters.runtime_failure_frames;
              if (vk_err.find("[vulkan_device_lost]") != std::string::npos)
                ++open_vulkan_auto_frame.counters.device_loss_frames;
              open_vulkan_auto_frame.ResetTrackingState(
                  OpenVulkanAutoFrameResetReason::runtime_or_device_failure);
              block_optional_effect(
                  OptionalEffectSlot::auto_frame, stage_id, "open_vulkan",
                  OpenVulkanAutoFrameRuntimeFailure(vk_err),
                  capture_sequence);
              return;
            }
            ensure_open_vulkan_next();

            DeferredGpuOut auto_frame_gpu_out{};
            const auto *auto_frame_input = open_vulkan_curr;
            const std::uint64_t auto_frame_failures_before =
                open_vulkan_auto_frame.counters.runtime_failure_frames;
            const std::uint64_t auto_frame_device_losses_before =
                open_vulkan_auto_frame.counters.device_loss_frames;
            if (open_vulkan_auto_frame.ApplyVulkanRgb(
                    capture_sequence, *open_vulkan_curr, open_vulkan_next, fx,
                    tracking_box_ptr, tracking_provider_ran,
                    &auto_frame_gpu_out, &vk_err)) {
              open_vulkan_effect_ran_this_frame = true;
              deferred_gpu_out = auto_frame_gpu_out;
              adopt_open_vulkan_output(deferred_gpu_out.vulkan_img);
              open_vulkan_resident_frame.current =
                  deferred_gpu_out.vulkan_img;
              open_vulkan_resident_frame.auto_frame_applied = true;
              open_vulkan_resident_frame.auto_frame_crop_applied =
                  deferred_gpu_out.vulkan_img != auto_frame_input;
              open_vulkan_gpu_frame_pending_cpu_readback = true;

              const bool keep_deferred_after_auto_frame =
                  allow_defer_readback && (stage == last_stage_for_defer);
              if (keep_deferred_after_auto_frame) {
                have_deferred_gpu_out = true;
              } else {
                std::string down_err;
                if (!download_open_vulkan_current_to_cpu(&down_err)) {
                  ++open_vulkan_runtime_failure_frames;
                  ++open_vulkan_auto_frame.counters.runtime_failure_frames;
                  if (down_err.find("[vulkan_device_lost]") !=
                      std::string::npos) {
                    ++open_vulkan_auto_frame.counters.device_loss_frames;
                  }
                  open_vulkan_auto_frame.ResetTrackingState(
                      OpenVulkanAutoFrameResetReason::runtime_or_device_failure);
                  block_optional_effect(
                      OptionalEffectSlot::auto_frame, stage_id, "open_vulkan",
                      OpenVulkanAutoFrameRuntimeFailure(down_err),
                      capture_sequence);
                  return;
                }
                have_deferred_gpu_out = false;
              }
              mark_open_vulkan_auto_frame_cpu_tail(tracking_tail_source,
                                                   /*cpu_crop_resize=*/false);
              clear_optional_effect_on_success(breaker, capture_sequence);
              mark_live_effect_success(stage_id, "open_vulkan");
              return;
            }

            ++open_vulkan_runtime_failure_frames;
            if (open_vulkan_auto_frame.counters.runtime_failure_frames ==
                auto_frame_failures_before) {
              ++open_vulkan_auto_frame.counters.runtime_failure_frames;
            }
            if (vk_err.find("[vulkan_device_lost]") != std::string::npos &&
                open_vulkan_auto_frame.counters.device_loss_frames ==
                    auto_frame_device_losses_before) {
              ++open_vulkan_auto_frame.counters.device_loss_frames;
            }
            open_vulkan_auto_frame.ResetTrackingState(
                OpenVulkanAutoFrameResetReason::runtime_or_device_failure);
            block_optional_effect(OptionalEffectSlot::auto_frame, stage_id,
                                  "open_vulkan", vk_err, capture_sequence);
            return;
          }
#endif

          if (have_maxine_auto_frame) {
            auto &breaker = optional_breaker(OptionalEffectSlot::auto_frame);
            if (!breaker.AllowsAttempt(capture_sequence))
              return;
            std::string mx_err;
            if (!maxine_auto_frame.ApplyRgbInPlace(
                    rgb.data(), capA.width, capA.height, rgbStride, fx,
                    apply_vignette_on_auto_frame, vignette_center_x_px,
                    vignette_center_y_px, &mx_err, defer_readback,
                    &deferred_gpu_out)) {
              if (IsVignetteFailureReason(mx_err)) {
                clear_optional_effect_on_success(breaker, capture_sequence);
                mark_live_effect_success(stage_id, "maxine_ar_cuda");
                block_optional_effect(
                    OptionalEffectSlot::vignette,
                    studiocast::video::effects::contract::kEffectIdVignette,
                    "cuda", mx_err, capture_sequence);
              } else {
                block_optional_effect(OptionalEffectSlot::auto_frame, stage_id,
                                      "maxine_ar_cuda",
                                      OptionalEffectFailureReason(
                                          "Maxine auto frame failed", mx_err),
                                      capture_sequence);
              }
            } else {
              count_maxine_stage(
                  /*deferred_readback=*/
                  defer_readback &&
                      deferred_gpu_out.kind != DeferredGpuKind::none,
                  /*runs_green_screen=*/false);
              clear_optional_effect_on_success(breaker, capture_sequence);
              mark_live_effect_success(stage_id, "maxine_ar_cuda");
              if (apply_vignette_on_auto_frame) {
                mark_live_effect_success(
                    studiocast::video::effects::contract::kEffectIdVignette,
                    "maxine_ar_cuda");
              }
            }
            if (defer_readback &&
                deferred_gpu_out.kind != DeferredGpuKind::none)
              have_deferred_gpu_out = true;
            return;
          }

          if (have_open_cuda_auto_frame) {
            // Open Video face detection is preferred for Auto Frame (cheaper
            // than foreground matting, and helps avoid running the matting
            // model solely for tracking). Convert detector output into the
            // model-agnostic TrackingBox contract before crop planning.
            OpenCudaAutoFrameContext::TrackingBox tracking_box;
            const OpenCudaAutoFrameContext::TrackingBox *tracking_box_ptr =
                nullptr;
            bool tracking_provider_ran = false;
            auto tracking_tail_source =
                OpenCudaAutoFrameContext::TrackingSource::kMatte;

            const auto adopt_face_detections =
                [&](const std::vector<studiocast::open_video::FaceDetection>
                        &detections) {
                  tracking_provider_ran = true;
                  tracking_tail_source =
                      OpenCudaAutoFrameContext::TrackingSource::kFace;
                  if (OpenCudaAutoFrameContext::BuildBestFaceTrackingBox(
                          detections, &tracking_box)) {
                    tracking_box_ptr = &tracking_box;
                  }
                };

            if (open_video_cache.face_detections) {
              adopt_face_detections(*open_video_cache.face_detections);
            } else if (open_video_yunet.available()) {
              std::string fd_err;
              if (open_video_yunet.EnsureDetectionsForFrame(
                      rgb.data(), capA.width, capA.height, rgbStride,
                      fx.auto_frame.model_id, capture_sequence,
                      &open_video_cache, &fd_err)) {
                if (open_video_cache.face_detections) {
                  adopt_face_detections(*open_video_cache.face_detections);
                }
              } else {
                // Best-effort: if face detection can't run (runtime failure),
                // Auto Frame will fall back to matte tracking if available.
                if (!fd_err.empty()) {
                  std::lock_guard<std::mutex> lock(mu_);
                  last_error_ = "Open Video face detection failed: " + fd_err;
                }
              }
            }

            std::string oc_err;
            if (have_deferred_gpu_out &&
                deferred_gpu_out.kind == DeferredGpuKind::cuda_rgb &&
                deferred_gpu_out.cuda_img && deferred_gpu_out.cuda &&
                open_cuda_next != nullptr) {
              DeferredGpuOut auto_frame_gpu_out{};
              if (open_cuda_auto_frame.ApplyCudaRgb(
                      capture_sequence, *deferred_gpu_out.cuda_img,
                      open_cuda_next, rgb.data(), capA.width, capA.height,
                      rgbStride, fx, tracking_box_ptr, tracking_provider_ran,
                      &auto_frame_gpu_out, &oc_err)) {
                deferred_gpu_out = auto_frame_gpu_out;
                have_deferred_gpu_out =
                    deferred_gpu_out.kind != DeferredGpuKind::none;
                open_cuda_curr = deferred_gpu_out.cuda_img;
                if (open_cuda_curr == &open_cuda_frame_a) {
                  open_cuda_next = &open_cuda_frame_b;
                } else if (open_cuda_curr == &open_cuda_frame_b) {
                  open_cuda_next = &open_cuda_frame_a;
                }

                const bool keep_deferred_after_auto_frame =
                    allow_defer_readback && (stage == last_stage_for_defer);
                bool auto_frame_output_visible = keep_deferred_after_auto_frame;
                if (!keep_deferred_after_auto_frame && have_deferred_gpu_out &&
                    deferred_gpu_out.cuda_img && deferred_gpu_out.cuda) {
                  std::string derr;
                  if (deferred_gpu_out.cuda_img->DownloadToCpuRgb24(
                          deferred_gpu_out.cuda, rgb.data(), rgbStride,
                          deferred_gpu_out.stream, &derr) &&
                      deferred_gpu_out.cuda->StreamSynchronize(
                          deferred_gpu_out.stream, &derr)) {
                    ++open_cuda_rgb_download_calls;
                    ++open_cuda_cpu_continuation_download_calls;
                    ++open_cuda_forced_sync_calls;
                    have_deferred_gpu_out = false;
                    open_cuda_gpu_frame_pending_cpu_readback = false;
                    auto_frame_output_visible = true;
                  } else {
                    std::lock_guard<std::mutex> lock(mu_);
                    last_error_ =
                        "Open CUDA auto frame failed after GPU crop/resize: " +
                        derr;
                  }
                }
                mark_open_cuda_auto_frame_cpu_tail(tracking_tail_source,
                                                   /*cpu_crop_resize=*/false);
                if (auto_frame_output_visible) {
                  mark_live_effect_success(stage_id, "open_cuda");
                } else {
                  remove_live_effect(stage_id);
                }
                return;
              }

              std::string derr;
              if (deferred_gpu_out.cuda_img->DownloadToCpuRgb24(
                      deferred_gpu_out.cuda, rgb.data(), rgbStride,
                      deferred_gpu_out.stream, &derr) &&
                  deferred_gpu_out.cuda->StreamSynchronize(
                      deferred_gpu_out.stream, &derr)) {
                ++open_cuda_rgb_download_calls;
                ++open_cuda_cpu_continuation_download_calls;
                ++open_cuda_forced_sync_calls;
                have_deferred_gpu_out = false;
                open_cuda_gpu_frame_pending_cpu_readback = false;
              } else {
                std::lock_guard<std::mutex> lock(mu_);
                last_error_ = "Open CUDA auto frame failed and deferred "
                              "readback failed: " +
                              (derr.empty() ? oc_err : derr);
                return;
              }
            }

            bool cpu_crop_resize_applied = false;
            if (!open_cuda_auto_frame.ApplyRgbInPlace(
                    capture_sequence, rgb.data(), capA.width, capA.height,
                    rgbStride, fx, tracking_box_ptr, tracking_provider_ran,
                    &oc_err, &cpu_crop_resize_applied)) {
              // Best-effort: Auto Frame failures shouldn't break the entire
              // pipeline.
              if (!oc_err.empty()) {
                std::lock_guard<std::mutex> lock(mu_);
                last_error_ = "Open CUDA auto frame failed: " + oc_err;
              }
            } else {
              mark_live_effect_success(stage_id, "open_cuda");
            }
            mark_open_cuda_auto_frame_cpu_tail(tracking_tail_source,
                                               cpu_crop_resize_applied);
          }
          return;
        }

        if (stage_id ==
            studiocast::video::effects::contract::kEffectIdVignette) {
          if (!have_maxine_vignette_only)
            return;
          auto &breaker = optional_breaker(OptionalEffectSlot::vignette);
          if (!breaker.AllowsAttempt(capture_sequence))
            return;
          std::string mx_err;
          if (!vignette_only.ApplyRgbInPlace(
                  rgb.data(), capA.width, capA.height, rgbStride, fx,
                  vignette_center_x_px, vignette_center_y_px, &mx_err,
                  defer_readback, &deferred_gpu_out)) {
            block_optional_effect(OptionalEffectSlot::vignette, stage_id,
                                  "cuda",
                                  IsVignetteFailureReason(mx_err)
                                      ? mx_err
                                      : OptionalEffectFailureReason(
                                            "CUDA vignette failed", mx_err),
                                  capture_sequence);
          } else {
            count_maxine_stage(
                /*deferred_readback=*/
                defer_readback &&
                    deferred_gpu_out.kind != DeferredGpuKind::none,
                /*runs_green_screen=*/false);
            clear_optional_effect_on_success(breaker, capture_sequence);
            mark_live_effect_success(stage_id, "cuda");
          }
          if (defer_readback && deferred_gpu_out.kind != DeferredGpuKind::none)
            have_deferred_gpu_out = true;
          return;
        }
      };

      auto execute_prepared_maxine_resident_island =
          [&](const PreparedMaxineResidentIsland &) {
            const auto slot_for_semantic =
                [&](hal::SemanticEffectId semantic) {
                  switch (semantic) {
                  case hal::SemanticEffectId::video_noise_removal:
                    return OptionalEffectSlot::denoise;
                  case hal::SemanticEffectId::eye_contact:
                    return OptionalEffectSlot::eye_contact;
                  case hal::SemanticEffectId::virtual_background_blur:
                  case hal::SemanticEffectId::virtual_background_remove:
                  case hal::SemanticEffectId::virtual_background_replace:
                    return OptionalEffectSlot::virtual_background;
                  case hal::SemanticEffectId::virtual_key_light:
                    return OptionalEffectSlot::relight;
                  case hal::SemanticEffectId::auto_frame:
                    return OptionalEffectSlot::auto_frame;
                  case hal::SemanticEffectId::vignette:
                    return OptionalEffectSlot::vignette;
                  case hal::SemanticEffectId::mirror:
                  case hal::SemanticEffectId::invalid:
                    return OptionalEffectSlot::count;
                  }
                  return OptionalEffectSlot::count;
                };
            const auto effect_id_for_semantic =
                [](hal::SemanticEffectId semantic) {
                  using namespace
                      studiocast::video::effects::contract;
                  switch (semantic) {
                  case hal::SemanticEffectId::video_noise_removal:
                    return kEffectIdVideoNoiseRemoval;
                  case hal::SemanticEffectId::eye_contact:
                    return kEffectIdEyeContact;
                  case hal::SemanticEffectId::virtual_background_blur:
                    return kEffectIdVirtualBackgroundBlur;
                  case hal::SemanticEffectId::virtual_background_remove:
                    return kEffectIdVirtualBackgroundRemove;
                  case hal::SemanticEffectId::virtual_background_replace:
                    return kEffectIdVirtualBackgroundReplace;
                  case hal::SemanticEffectId::virtual_key_light:
                    return kEffectIdVirtualKeyLight;
                  case hal::SemanticEffectId::auto_frame:
                    return kEffectIdAutoFrame;
                  case hal::SemanticEffectId::vignette:
                    return kEffectIdVignette;
                  case hal::SemanticEffectId::mirror:
                    return kEffectIdMirror;
                  case hal::SemanticEffectId::invalid:
                    return std::string_view{};
                  }
                  return std::string_view{};
                };
            const auto backend_for_semantic =
                [](hal::SemanticEffectId semantic) {
                  if (semantic == hal::SemanticEffectId::vignette)
                    return std::string_view{"cuda"};
                  return std::string_view{"maxine"};
                };
            const auto resident_kind_for_semantic =
                [](hal::SemanticEffectId semantic) {
                  using Kind = studiocast::maxine::ResidentStageKind;
                  switch (semantic) {
                  case hal::SemanticEffectId::video_noise_removal:
                    return Kind::denoise;
                  case hal::SemanticEffectId::eye_contact:
                    return Kind::eye_contact;
                  case hal::SemanticEffectId::virtual_background_blur:
                    return Kind::background_blur;
                  case hal::SemanticEffectId::virtual_background_remove:
                    return Kind::background_remove;
                  case hal::SemanticEffectId::virtual_background_replace:
                    return Kind::background_replace;
                  case hal::SemanticEffectId::virtual_key_light:
                    return Kind::relighting;
                  case hal::SemanticEffectId::auto_frame:
                    return Kind::auto_frame;
                  case hal::SemanticEffectId::vignette:
                    return Kind::vignette;
                  case hal::SemanticEffectId::mirror:
                  case hal::SemanticEffectId::invalid:
                    return Kind::count;
                  }
                  return Kind::count;
                };

            const auto &active =
                prepared_maxine_hardware_owner.active_request();
            const std::size_t tight_stride =
                static_cast<std::size_t>(capA.width) * 3u;
            const std::size_t tight_bytes =
                tight_stride * static_cast<std::size_t>(capA.height);
            const auto frame =
                prepared_maxine_hardware_owner.ExecuteHostFrame(
                    rgb.data(), rgb.size(), rgbStride,
                    prepared_maxine_bgr_output.data(),
                    prepared_maxine_bgr_output.size(), tight_stride,
                    capture_sequence);

            maxine_rgb_to_bgr_calls +=
                frame.compatibility.rgb_to_bgr_calls;
            maxine_upload_attempts +=
                frame.compatibility.upload_attempts;
            maxine_upload_calls += frame.compatibility.upload_calls;
            maxine_final_download_attempts +=
                frame.compatibility.final_download_attempts;
            maxine_final_download_calls +=
                frame.compatibility.final_download_calls;
            maxine_cpu_continuation_download_attempts +=
                frame.compatibility.cpu_continuation_download_attempts;
            maxine_cpu_continuation_download_calls +=
                frame.compatibility.cpu_continuation_download_calls;
            maxine_download_calls +=
                frame.compatibility.final_download_calls +
                frame.compatibility.cpu_continuation_download_calls;
            maxine_device_bridge_attempts +=
                frame.compatibility.device_bridge_attempts;
            maxine_device_bridge_calls +=
                frame.compatibility.device_bridge_calls;
            maxine_matte_inference_attempts +=
                frame.compatibility.matte_inference_attempts;
            maxine_green_screen_calls +=
                frame.compatibility.matte_inference_calls;
            maxine_shared_green_screen_matte_reuse_calls +=
                frame.compatibility.shared_matte_reuses;
            maxine_forced_sync_attempts +=
                frame.compatibility.sync_attempts;
            maxine_forced_sync_calls += frame.compatibility.sync_calls;
            maxine_composite_attempts +=
                frame.compatibility.composite_attempts;
            maxine_composite_calls += frame.compatibility.composite_calls;
            maxine_synchronous_sdk_run_attempts +=
                frame.compatibility.synchronous_sdk_run_attempts;
            maxine_synchronous_sdk_run_calls +=
                frame.compatibility.synchronous_sdk_run_calls;
            maxine_asynchronous_sdk_run_attempts +=
                frame.compatibility.asynchronous_sdk_run_attempts;
            maxine_asynchronous_sdk_run_calls +=
                frame.compatibility.asynchronous_sdk_run_calls;
            for (std::size_t index = 0; index < active.stage_count; ++index) {
              const auto kind =
                  resident_kind_for_semantic(active.stages[index].semantic);
              if (kind >=
                  studiocast::maxine::ResidentStageKind::count) {
                continue;
              }
              const auto kind_index = static_cast<std::size_t>(kind);
              maxine_stage_attempts[kind_index] +=
                  frame.compatibility.stage_attempts[index];
              maxine_stage_successes[kind_index] +=
                  frame.compatibility.stage_successes[index];
            }

            prepared_maxine_completion = frame.completion;
            prepared_maxine_completion_pending =
                frame.completion_pending;
            if (!frame.output_ready || frame.output == nullptr ||
                frame.output_size_bytes != tight_bytes ||
                frame.output_row_stride_bytes != tight_stride) {
              ++maxine_runtime_failure_frames;
              if (prepared_maxine_completion_pending) {
                hal::ReasonId commit_reason = hal::ReasonId::none;
                (void)prepared_maxine_hardware_owner.CommitOutput(
                    prepared_maxine_completion, false, &commit_reason);
                prepared_maxine_completion_pending = false;
                publish_prepared_maxine_status();
              }
              if (frame.reason == hal::ReasonId::device_lost) {
                prepared_maxine_hardware_owner.NotifyDeviceLost();
                prepared_maxine_hardware_suspended = true;
                publish_prepared_maxine_status();
              }
              publish_video_hal_status(
                  appliedFx, appliedEffectsGeneration,
                  VideoRuntimeLifecycle::running);
              for (std::size_t index = 0; index < active.stage_count;
                   ++index) {
                const auto slot =
                    slot_for_semantic(active.stages[index].semantic);
                const auto effect_id =
                    effect_id_for_semantic(active.stages[index].semantic);
                if (slot == OptionalEffectSlot::count ||
                    effect_id.empty()) {
                  continue;
                }
                block_optional_effect(
                    slot, effect_id,
                    backend_for_semantic(active.stages[index].semantic),
                    OptionalEffectFailureReason(
                        "Prepared Maxine HAL frame boundary failed",
                        std::string(PreparedMaxineHardwareReasonCode(
                            frame.reason))),
                    capture_sequence);
              }
              return;
            }

            studiocast::video::Bgr24ToRgb24(
                frame.output, rgb.data(), capA.width, capA.height,
                frame.output_row_stride_bytes, rgbStride);
            ++maxine_bgr_to_rgb_calls;
            ++maxine_cpu_tail_stage_calls;

            for (std::size_t index = 0; index < frame.stage_count; ++index) {
              const auto &stage = frame.stages[index];
              if (stage.status !=
                  hal::StageExecutionStatus::optional_failed_open) {
                continue;
              }
              const auto slot = slot_for_semantic(stage.semantic);
              const auto effect_id =
                  effect_id_for_semantic(stage.semantic);
              if (slot == OptionalEffectSlot::count ||
                  effect_id.empty()) {
                continue;
              }
              block_optional_effect(
                  slot, effect_id, backend_for_semantic(stage.semantic),
                  OptionalEffectFailureReason(
                      "Prepared Maxine HAL optional stage failed open",
                      std::string(
                          PreparedMaxineHardwareReasonCode(stage.reason))),
                  capture_sequence);
            }
          };

      auto execute_maxine_resident_island =
          [&](const PreparedMaxineResidentIsland &island) {
            using studiocast::maxine::ResidentExecutionStatus;
            using studiocast::maxine::ResidentFramePlan;
            using studiocast::maxine::ResidentInvalidationReason;
            using studiocast::maxine::ResidentStageKind;

            const auto breaker_for_kind =
                [&](ResidentStageKind kind) -> OptionalEffectBreaker * {
              switch (kind) {
              case ResidentStageKind::denoise:
                return &optional_breaker(OptionalEffectSlot::denoise);
              case ResidentStageKind::eye_contact:
                return &optional_breaker(OptionalEffectSlot::eye_contact);
              case ResidentStageKind::background_blur:
              case ResidentStageKind::background_remove:
              case ResidentStageKind::background_replace:
                return &optional_breaker(OptionalEffectSlot::virtual_background);
              case ResidentStageKind::relighting:
                return &optional_breaker(OptionalEffectSlot::relight);
              case ResidentStageKind::auto_frame:
                return &optional_breaker(OptionalEffectSlot::auto_frame);
              case ResidentStageKind::vignette:
                return &optional_breaker(OptionalEffectSlot::vignette);
              case ResidentStageKind::transfer:
              case ResidentStageKind::count:
                return nullptr;
              }
              return nullptr;
            };
            const auto effect_id_for_kind = [](ResidentStageKind kind) {
              using namespace studiocast::video::effects::contract;
              switch (kind) {
              case ResidentStageKind::denoise:
                return kEffectIdVideoNoiseRemoval;
              case ResidentStageKind::eye_contact:
                return kEffectIdEyeContact;
              case ResidentStageKind::background_blur:
                return kEffectIdVirtualBackgroundBlur;
              case ResidentStageKind::background_remove:
                return kEffectIdVirtualBackgroundRemove;
              case ResidentStageKind::background_replace:
                return kEffectIdVirtualBackgroundReplace;
              case ResidentStageKind::relighting:
                return kEffectIdVirtualKeyLight;
              case ResidentStageKind::auto_frame:
                return kEffectIdAutoFrame;
              case ResidentStageKind::vignette:
                return kEffectIdVignette;
              case ResidentStageKind::transfer:
              case ResidentStageKind::count:
                return std::string_view{};
              }
              return std::string_view{};
            };
            const auto backend_for_kind = [](ResidentStageKind kind) {
              switch (kind) {
              case ResidentStageKind::eye_contact:
                return std::string_view{"maxine_ar"};
              case ResidentStageKind::auto_frame:
                return std::string_view{"maxine_ar_cuda"};
              case ResidentStageKind::vignette:
                return std::string_view{"cuda"};
              default:
                return std::string_view{"maxine"};
              }
            };

            ResidentFramePlan runnable{};
            for (std::size_t i = 0; i < island.plan.stage_count; ++i) {
              const auto &stage = island.plan.stages[i];
              auto *breaker = breaker_for_kind(stage.kind);
              if (breaker && !breaker->AllowsAttempt(capture_sequence))
                continue;
              (void)runnable.AddCompatible(
                  stage.kind, stage.optional, stage.requires_shared_matte,
                  stage.matte_fingerprint);
            }
            if (runnable.Empty())
              return;
            if (island.plan.has_cpu_tail)
              (void)runnable.SetCpuTail(island.plan.cpu_tail);

            const auto key = maxine_resident_executor->key();
            const studiocast::maxine::HostRgbFrameView host{
                rgb.data(), static_cast<std::uint32_t>(capA.width),
                static_cast<std::uint32_t>(capA.height), rgbStride};
            const auto result = maxine_resident_section.Execute(
                runnable, key, capture_sequence, host,
                *maxine_resident_executor);
            accumulate_maxine_resident_telemetry();

            const auto block_runnable_stages =
                [&](const char *summary, std::string_view detail) {
                  const std::string detail_string(detail);
                  for (std::size_t i = 0; i < runnable.stage_count; ++i) {
                    const auto kind = runnable.stages[i].kind;
                    auto *breaker = breaker_for_kind(kind);
                    const auto effect_id = effect_id_for_kind(kind);
                    if (!breaker || effect_id.empty())
                      continue;
                    block_optional_effect(
                        static_cast<OptionalEffectSlot>(
                            breaker - optional_effect_breakers.data()),
                        effect_id, backend_for_kind(kind),
                        OptionalEffectFailureReason(summary, detail_string),
                        capture_sequence);
                  }
                };

            const bool cpu_output_ready =
                result.status == ResidentExecutionStatus::cpu_visible_output ||
                result.status == ResidentExecutionStatus::cpu_tail_required;
            if (!cpu_output_ready) {
              ++maxine_runtime_failure_frames;
              maxine_resident_executor->InvalidateBindings();
              maxine_resident_section.Invalidate(
                  ResidentInvalidationReason::runtime_failure);
              block_runnable_stages(
                  "Maxine resident frame boundary failed",
                  maxine_resident_executor->last_error());
              return;
            }

            const auto host_output = maxine_resident_executor->host_output();
            if (!host_output.Valid() ||
                host_output.width != static_cast<std::uint32_t>(capA.width) ||
                host_output.height != static_cast<std::uint32_t>(capA.height)) {
              ++maxine_runtime_failure_frames;
              maxine_resident_executor->InvalidateBindings();
              maxine_resident_section.Invalidate(
                  ResidentInvalidationReason::runtime_failure);
              block_runnable_stages(
                  "Maxine resident host output is incompatible",
                  maxine_resident_executor->last_error());
              return;
            }
            studiocast::video::Bgr24ToRgb24(
                host_output.data, rgb.data(), capA.width, capA.height,
                host_output.stride_bytes, rgbStride);
            ++maxine_bgr_to_rgb_calls;
            if (result.status == ResidentExecutionStatus::cpu_tail_required)
              ++maxine_cpu_tail_stage_calls;

            for (std::size_t i = 0; i < runnable.stage_count; ++i) {
              const auto kind = runnable.stages[i].kind;
              auto *breaker = breaker_for_kind(kind);
              const auto effect_id = effect_id_for_kind(kind);
              if (!breaker || effect_id.empty())
                continue;
              const bool failed =
                  (result.failed_plan_stage_mask &
                   (std::uint32_t{1} << static_cast<std::uint32_t>(i))) != 0;
              if (failed) {
                block_optional_effect(
                    static_cast<OptionalEffectSlot>(breaker -
                        optional_effect_breakers.data()),
                    effect_id, backend_for_kind(kind),
                    OptionalEffectFailureReason(
                        "Maxine resident effect failed",
                        std::string(maxine_resident_executor->last_error())),
                    capture_sequence);
              } else {
                clear_optional_effect_on_success(*breaker, capture_sequence);
                mark_live_effect_success(effect_id, backend_for_kind(kind));
              }
            }
          };

      std::size_t maxine_island_index = 0;
      for (std::size_t i = 0; i < frame_plan.size; ++i) {
        if (prepared_maxine_hardware_live &&
            maxine_island_index < prepared_maxine_resident_islands.count &&
            i == prepared_maxine_resident_islands
                     .islands[maxine_island_index]
                     .first_frame_stage) {
          const auto &island =
              prepared_maxine_resident_islands.islands[maxine_island_index++];
          if (!prepared_maxine_hardware_suspended)
            execute_prepared_maxine_resident_island(island);
          i = island.end_frame_stage - 1;
          continue;
        }
        if (maxine_resident_executor &&
            maxine_island_index < prepared_maxine_resident_islands.count &&
            i == prepared_maxine_resident_islands
                     .islands[maxine_island_index]
                     .first_frame_stage) {
          const auto &island =
              prepared_maxine_resident_islands.islands[maxine_island_index++];
          execute_maxine_resident_island(island);
          i = island.end_frame_stage - 1;
          continue;
        }
        apply_stage(frame_plan.StageAt(i));
        if (fx_failed)
          break;
      }

      // Safety net: if key light was deferred but VB did not run (or bailed
      // early), apply it now so it isn't silently skipped.
      if (!fx_failed && pending_open_cuda_key_light &&
          have_open_cuda_key_light) {
        pending_open_cuda_key_light = false;
        mark_open_cuda_cpu_tail(
            studiocast::video::effects::contract::kEffectIdVirtualKeyLight);
        std::string kl_err;
        if (!open_cuda_key_light.ApplyRgbInPlace(capture_sequence, rgb.data(),
                                                 capA.width, capA.height,
                                                 rgbStride, fx, &kl_err)) {
          {
            std::lock_guard<std::mutex> lock(mu_);
            last_error_ = "Open CUDA virtual key light failed: " + kl_err;
          }
          if (studiocast::video::effects::
                  ShouldAbortPipelineOnOpenCudaVbApplyFailure()) {
            fx_failed = true;
          }
        } else {
          mark_live_effect_success(
              studiocast::video::effects::contract::kEffectIdVirtualKeyLight,
              "open_cuda");
        }
      }

      // CPU-only tail effects run last.
      if (!fx_failed) {
        if (open_cuda_gpu_frame_pending_cpu_readback &&
            !have_deferred_gpu_out) {
          std::string down_err;
          if (!download_open_cuda_current_to_cpu(&down_err)) {
            {
              std::lock_guard<std::mutex> lock(mu_);
              last_error_ = down_err;
            }
            if (studiocast::video::effects::
                    ShouldAbortPipelineOnOpenCudaVbApplyFailure()) {
              fx_failed = true;
            }
          }
        }
#if STUDIOCAST_ENABLE_OPEN_VULKAN
        if (open_vulkan_gpu_frame_pending_cpu_readback &&
            !have_deferred_gpu_out) {
          std::string down_err;
          if (!download_open_vulkan_current_to_cpu(&down_err)) {
            std::lock_guard<std::mutex> lock(mu_);
            last_error_ = down_err;
            fx_failed = true;
          }
        }
#endif
      }

      if (!fx_failed) {
        chain.Apply(view);
      }
    }

    const auto t_effects_end = Clock::now();
    const double effects_ms = ToMs(t_effects_end - t_effects_start);

    if (fx_failed) {
      break;
    }

#if STUDIOCAST_ENABLE_OPEN_VULKAN
    // Final-output vignette/mirror requests enter the same resident block as
    // earlier Vulkan stages. Otherwise upload the post-analysis CPU frame once
    // here; resize, vignette, and mirror are batched below.
    if (!prepared_hardware_final_live &&
        (open_vulkan_vignette_attempt_this_frame ||
         open_vulkan_mirror_attempt_this_frame) &&
        !have_deferred_gpu_out) {
      std::string final_err;
      if (ensure_open_vulkan_final_current_from_cpu(&final_err)) {
        deferred_gpu_out.kind = DeferredGpuKind::vulkan_rgb;
        deferred_gpu_out.nvcv_img = nullptr;
        deferred_gpu_out.nvcv = nullptr;
        deferred_gpu_out.cuda_img = nullptr;
        deferred_gpu_out.cuda = nullptr;
        deferred_gpu_out.stream = nullptr;
        deferred_gpu_out.vulkan_img = open_vulkan_curr;
        deferred_gpu_out.vulkan_kernels = &open_vulkan_vb.kernels;
        have_deferred_gpu_out = true;
        open_vulkan_gpu_frame_pending_cpu_readback = true;
      } else {
        ++open_vulkan_fallback_frames;
        ++open_vulkan_runtime_failure_frames;
        if (open_vulkan_vignette_attempt_this_frame) {
          block_optional_effect(
              OptionalEffectSlot::vignette,
              studiocast::video::effects::contract::kEffectIdVignette,
              "open_vulkan", OpenVulkanVignetteRuntimeFailure(final_err),
              f.sequence);
        }
        if (open_vulkan_mirror_attempt_this_frame) {
          block_optional_effect(
              OptionalEffectSlot::mirror,
              studiocast::video::effects::contract::kEffectIdMirror,
              "open_vulkan", OpenVulkanMirrorRuntimeFailure(final_err),
              f.sequence);
        }
        open_vulkan_vignette_attempt_this_frame = false;
        open_vulkan_mirror_attempt_this_frame = false;
      }
    }
#endif

    const auto t_scale_start = Clock::now();

    // Pack/write to output format.
    //
    // IMPORTANT: The negotiated camera capture size (capA) can differ from the
    // loopback output size (outA) due to camera format negotiation (common with
    // YUYV) and/or consumers requesting a preferred resolution. v4l2loopback
    // does not always scale between these sizes, so if we only populate the
    // top-left portion of a larger output buffer, consumers will display the
    // frame in the top-left quadrant.
    //
    // To keep output stable, scale the internal RGB frame to the writer's
    // negotiated output dimensions before packing to RGB24/YUYV.
    const int outW = outA.width;
    const int outH = outA.height;
    const bool scaling_needed = (capA.width != outW || capA.height != outH);

    int frameW = capA.width;
    int frameH = capA.height;

    const std::uint8_t *rgbOut = rgb.data();
    std::size_t rgbOutStride = rgbStride;
    std::size_t rgbOutBytes = rgb.size();
    if (prepared_open_cuda_output_ready) {
      rgbOut = prepared_open_cuda_rgb_output.data();
      rgbOutStride = static_cast<std::size_t>(capA.width) * 3u;
      rgbOutBytes = prepared_open_cuda_rgb_output.size();
    }

    if (prepared_hardware_final_live &&
        !prepared_hardware_runtime_suspended) {
      const std::size_t tight_stride = static_cast<std::size_t>(outW) * 3u;
      const std::size_t tight_bytes =
          tight_stride * static_cast<std::size_t>(outH);
      hal::HostFrameBinding binding;
      const auto plan = prepared_hardware_composition.Snapshot().plan;
      binding.input_descriptor = plan.host_input;
      binding.output_descriptor = plan.host_output;
      binding.input_planes[0].read_data =
          reinterpret_cast<const std::byte *>(rgb.data());
      binding.input_planes[0].span_bytes = rgb.size();
      binding.input_planes[0].row_stride_bytes = rgbStride;
      binding.output_planes[0].write_data =
          reinterpret_cast<std::byte *>(rgbScaled.data());
      binding.output_planes[0].span_bytes = tight_bytes;
      binding.output_planes[0].row_stride_bytes = tight_stride;
      binding.input_plane_count = 1;
      binding.output_plane_count = 1;
      binding.capture_sequence = f.sequence;
      binding.ownership = hal::OwnershipModel::borrowed_until_completion;
      binding.lifetime = hal::ResourceLifetime::frame_borrowed;
      const auto prepared_frame =
          prepared_hardware_composition.Execute(binding);
      const auto execution_snapshot =
          prepared_hardware_composition.Snapshot();
      publish_prepared_hardware_status();
      const bool prepared_frame_succeeded =
          prepared_frame.status == hal::StatusId::executed ||
          prepared_frame.status == hal::StatusId::degraded;
      const bool prepared_output_valid =
          prepared_frame_succeeded && execution_snapshot.frame_pending &&
          prepared_frame.output_ready &&
          prepared_frame.output.output_ready &&
          prepared_frame.output.planes[0].read_data ==
              reinterpret_cast<const std::byte *>(rgbScaled.data()) &&
          prepared_frame.output.planes[0].span_bytes == tight_bytes &&
          prepared_frame.output.planes[0].row_stride_bytes == tight_stride;
      prepared_hardware_completion = prepared_frame.completion;
      prepared_hardware_completion_pending = prepared_output_valid;
      if (prepared_output_valid) {
        frameW = outW;
        frameH = outH;
        rgbOut = rgbScaled.data();
        rgbOutStride = tight_stride;
        rgbOutBytes = tight_bytes;
      } else if (execution_snapshot.frame_pending) {
        hal::ReasonId reason = hal::ReasonId::none;
        (void)prepared_hardware_composition.CommitOutput(
            prepared_hardware_completion, false, &reason);
        prepared_hardware_completion_pending = false;
        publish_prepared_hardware_status();
      }
      if (!prepared_output_valid) {
        prepared_hardware_runtime_suspended = true;
        bool active_vignette = false;
        bool active_mirror = false;
        for (std::size_t index = 0; index < plan.intent.stage_count; ++index) {
          active_vignette |= plan.intent.stages[index].intent.semantic ==
                             hal::SemanticEffectId::vignette;
          active_mirror |= plan.intent.stages[index].intent.semantic ==
                           hal::SemanticEffectId::mirror;
        }
        const std::string failure =
            "Prepared Vulkan HAL runtime failure [" +
            std::string(PreparedHardwareReasonCode(prepared_frame.reason)) +
            "].";
        if (active_vignette) {
          block_optional_effect(
              OptionalEffectSlot::vignette,
              studiocast::video::effects::contract::kEffectIdVignette,
              "open_vulkan", failure, f.sequence);
        }
        if (active_mirror) {
          block_optional_effect(
              OptionalEffectSlot::mirror,
              studiocast::video::effects::contract::kEffectIdMirror,
              "open_vulkan", failure, f.sequence);
        }
        publish_video_hal_status(appliedFx, appliedEffectsGeneration,
                                 VideoRuntimeLifecycle::running);
      }
      // No legacy Vulkan retry is attempted here. Optional-stage or device
      // failure leaves the last valid CPU frame intact; typed HAL status
      // carries the exact blocker and the ordinary output transport continues.
    }

    // If we deferred GPU readback, perform GPU resize (bilinear) to output
    // dimensions and do a single GPU->CPU transfer for the final output buffer.
    if (have_deferred_gpu_out &&
        deferred_gpu_out.kind == DeferredGpuKind::cuda_rgb) {
      std::string gerr;
      bool ok = true;

      if (!deferred_gpu_out.cuda_img || !deferred_gpu_out.cuda) {
        ok = false;
        gerr = "Deferred Open CUDA output reference is incomplete.";
      }

      if (ok) {
        if (!deferred_gpu_out.cuda_img->Valid()) {
          ok = false;
          gerr = "Deferred Open CUDA output image is invalid.";
        }
      }

      if (ok) {
        if (deferred_gpu_out.cuda_img->format !=
            studiocast::cuda::PixelFormatGpu::rgb_u8) {
          ok = false;
          gerr = "Deferred Open CUDA output image format must be rgb_u8.";
        }
      }

      const studiocast::cuda::CudaImage *src_img = deferred_gpu_out.cuda_img;
      const std::size_t tightStride = static_cast<std::size_t>(outW) * 3u;
      const std::size_t tightBytes =
          tightStride * static_cast<std::size_t>(outH);

      const studiocast::cuda::CudaImage *download_img = src_img;
      if (ok && (capA.width != outW || capA.height != outH)) {
        if (!gpu_rgb_scaled.ReallocIfNeeded(
                deferred_gpu_out.cuda, outW, outH,
                studiocast::cuda::PixelFormatGpu::rgb_u8, &gerr)) {
          ok = false;
        }
        gpu_rgb_scaled_allocated = gpu_rgb_scaled.Valid();
        if (ok) {
          if (!studiocast::cuda::kernels::ResizeBilinear(
                  *src_img, gpu_rgb_scaled, deferred_gpu_out.stream, &gerr)) {
            ok = false;
          }
        }
        download_img = &gpu_rgb_scaled;
      }

      if (ok) {
        rgbScaled.resize(tightBytes);
        if (open_cuda_active_this_frame) {
          ++open_cuda_rgb_download_calls;
          ++open_cuda_final_download_calls;
        }
        if (!download_img->DownloadToCpuRgb24(deferred_gpu_out.cuda,
                                              rgbScaled.data(), tightStride,
                                              deferred_gpu_out.stream, &gerr)) {
          ok = false;
        }
      }

      if (ok) {
        if (!deferred_gpu_out.cuda->StreamSynchronize(deferred_gpu_out.stream,
                                                      &gerr)) {
          ok = false;
        } else if (open_cuda_active_this_frame) {
          ++open_cuda_forced_sync_calls;
        }
      }

      if (ok) {
        frameW = outW;
        frameH = outH;
        rgbOut = rgbScaled.data();
        rgbOutStride = tightStride;
        rgbOutBytes = rgbScaled.size();
        have_deferred_gpu_out = false;
        open_cuda_gpu_frame_pending_cpu_readback = false;
      } else {
        // Attempt a best-effort readback into the pipeline RGB buffer so CPU
        // scaling/output remains correct.
        std::string derr;
        bool readback_ok = false;
        if (deferred_gpu_out.cuda_img && deferred_gpu_out.cuda) {
          if (deferred_gpu_out.cuda_img->DownloadToCpuRgb24(
                  deferred_gpu_out.cuda, rgb.data(), rgbStride,
                  deferred_gpu_out.stream, &derr) &&
              deferred_gpu_out.cuda->StreamSynchronize(deferred_gpu_out.stream,
                                                       &derr)) {
            if (open_cuda_active_this_frame) {
              ++open_cuda_rgb_download_calls;
              ++open_cuda_cpu_continuation_download_calls;
              ++open_cuda_forced_sync_calls;
            }
            readback_ok = true;
            have_deferred_gpu_out = false;
            open_cuda_gpu_frame_pending_cpu_readback = false;
          }
        }

        if (!readback_ok)
          live_effect_attribution.DiscardFrame();

        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = "Open CUDA deferred GPU resize path failed";
        if (!gerr.empty())
          last_error_ += ": " + gerr;
        if (!readback_ok && !derr.empty())
          last_error_ += " (and readback failed: " + derr + ")";
        last_error_ += ".";
      }
#if STUDIOCAST_ENABLE_OPEN_VULKAN
    } else if (have_deferred_gpu_out &&
               deferred_gpu_out.kind == DeferredGpuKind::vulkan_rgb) {
      std::string gerr;
      bool ok = true;
      bool final_vignette_applied = false;
      bool final_mirror_applied = false;

      if (!deferred_gpu_out.vulkan_img || !deferred_gpu_out.vulkan_kernels) {
        ok = false;
        gerr = "Deferred Open Vulkan output reference is incomplete.";
      }
      if (ok && !deferred_gpu_out.vulkan_img->Valid()) {
        ok = false;
        gerr = "Deferred Open Vulkan output image is invalid.";
      }
      if (ok && deferred_gpu_out.vulkan_img->format() !=
                    studiocast::vulkan::VulkanPixelFormat::rgb_u8) {
        ok = false;
        gerr = "Deferred Open Vulkan output image format must be rgb_u8.";
      }

      const studiocast::vulkan::VulkanImage *download_img =
          deferred_gpu_out.vulkan_img;
      const std::size_t tightStride = static_cast<std::size_t>(outW) * 3u;
      const std::size_t tightBytes =
          tightStride * static_cast<std::size_t>(outH);
      if (ok) {
        OpenVulkanFinalResidentStageInput final_input;
        final_input.source = deferred_gpu_out.vulkan_img;
        final_input.output_width = outW;
        final_input.output_height = outH;
        final_input.request_fixed_center_vignette =
            open_vulkan_vignette_attempt_this_frame;
        final_input.request_mirror = open_vulkan_mirror_attempt_this_frame;
        final_input.vignette_intensity_percent = appliedFx.vignette.intensity;
        // All analysis and preceding effects have completed before this exact
        // live final-output call. The production helper keeps both boundaries
        // explicit and independently rejects a caller that violates either.
        final_input.preceding_effects_complete = true;
        final_input.unmirrored_analysis_complete = true;

        OpenVulkanFinalResidentStageResources final_resources;
        final_resources.kernels = deferred_gpu_out.vulkan_kernels;
        final_resources.resize_scratch = &vulkan_rgb_scaled;
        final_resources.vignette_output = &open_vulkan_vb.vignette_rgb;
        final_resources.mirror_output = &open_vulkan_vb.mirror_rgb;
        final_resources.vignette = &open_vulkan_vignette;
        final_resources.vignette_counters = &open_vulkan_vignette_counters;
        final_resources.mirror = &open_vulkan_mirror;
        final_resources.mirror_counters = &open_vulkan_mirror_counters;

        OpenVulkanFinalResidentStageResult final_result;
        open_vulkan_resident_frame.current = deferred_gpu_out.vulkan_img;
        ok = ExecuteOpenVulkanResidentFrameFinalStage(
            open_vulkan_resident_frame,
            final_input, final_resources, &open_vulkan_final_stage_counters,
            &final_result);
        if (vulkan_rgb_scaled.Valid())
          vulkan_rgb_scaled_allocated = true;
        open_vulkan_forced_sync_calls +=
            final_result.synchronous_completion_count;

        if (open_vulkan_vignette_attempt_this_frame) {
          if (final_result.vignette_applied) {
            final_vignette_applied = true;
            open_vulkan_effect_ran_this_frame = true;
            clear_optional_effect_on_success(
                optional_breaker(OptionalEffectSlot::vignette), f.sequence);
          } else {
            ++open_vulkan_fallback_frames;
            if (final_result.vignette_initialization_failed)
              ++open_vulkan_runtime_failure_frames;
            block_optional_effect(
                OptionalEffectSlot::vignette,
                studiocast::video::effects::contract::kEffectIdVignette,
                "open_vulkan", final_result.vignette_error, f.sequence);
          }
          open_vulkan_vignette_attempt_this_frame = false;
        }
        if (open_vulkan_mirror_attempt_this_frame) {
          if (final_result.mirror_applied) {
            final_mirror_applied = true;
            open_vulkan_effect_ran_this_frame = true;
            clear_optional_effect_on_success(
                optional_breaker(OptionalEffectSlot::mirror), f.sequence);
          } else {
            ++open_vulkan_fallback_frames;
            if (final_result.mirror_initialization_failed)
              ++open_vulkan_runtime_failure_frames;
            block_optional_effect(
                OptionalEffectSlot::mirror,
                studiocast::video::effects::contract::kEffectIdMirror,
                "open_vulkan", final_result.mirror_error, f.sequence);
          }
          open_vulkan_mirror_attempt_this_frame = false;
        }
        if (ok) {
          download_img = final_result.output;
        } else {
          gerr = final_result.fatal_error;
        }
      }

      if (ok) {
        rgbScaled.resize(tightBytes);
        const std::uint64_t readback_submissions_before =
            deferred_gpu_out.vulkan_kernels->synchronous_submission_count();
        if (!open_vulkan_vb.DownloadImageToRgb(*download_img, rgbScaled.data(),
                                               tightStride, &gerr)) {
          ok = false;
        }
        open_vulkan_forced_sync_calls +=
            deferred_gpu_out.vulkan_kernels->synchronous_submission_count() -
            readback_submissions_before;
      }

      if (ok) {
        ++open_vulkan_download_calls;
        ++open_vulkan_final_download_calls;
        if (final_vignette_applied) {
          mark_live_effect_success(
              studiocast::video::effects::contract::kEffectIdVignette,
              "open_vulkan");
        }
        if (final_mirror_applied) {
          mark_live_effect_success(
              studiocast::video::effects::contract::kEffectIdMirror,
              "open_vulkan");
        }
        frameW = outW;
        frameH = outH;
        rgbOut = rgbScaled.data();
        rgbOutStride = tightStride;
        rgbOutBytes = rgbScaled.size();
        have_deferred_gpu_out = false;
        open_vulkan_gpu_frame_pending_cpu_readback = false;
      } else {
        if (final_vignette_applied) {
          remove_live_effect(
              studiocast::video::effects::contract::kEffectIdVignette);
        }
        if (final_mirror_applied) {
          remove_live_effect(
              studiocast::video::effects::contract::kEffectIdMirror);
        }
        std::string derr;
        bool readback_ok = false;
        if (deferred_gpu_out.vulkan_img) {
          const std::uint64_t readback_submissions_before =
              deferred_gpu_out.vulkan_kernels
                  ? deferred_gpu_out.vulkan_kernels
                        ->synchronous_submission_count()
                  : 0;
          if (open_vulkan_vb.DownloadImageToRgb(*deferred_gpu_out.vulkan_img,
                                                rgb.data(), rgbStride, &derr)) {
            ++open_vulkan_download_calls;
            ++open_vulkan_cpu_continuation_download_calls;
            if (deferred_gpu_out.vulkan_kernels) {
              open_vulkan_forced_sync_calls +=
                  deferred_gpu_out.vulkan_kernels
                      ->synchronous_submission_count() -
                  readback_submissions_before;
            }
            readback_ok = true;
            have_deferred_gpu_out = false;
            open_vulkan_gpu_frame_pending_cpu_readback = false;
          }
        }

        if (!readback_ok)
          live_effect_attribution.DiscardFrame();

        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = "Open Vulkan deferred GPU resize path failed";
        if (!gerr.empty())
          last_error_ += ": " + gerr;
        if (!readback_ok && !derr.empty())
          last_error_ += " (and readback failed: " + derr + ")";
        last_error_ += ".";
      }
#endif
    } else if (have_deferred_gpu_out &&
               deferred_gpu_out.kind == DeferredGpuKind::nvcv_bgr) {
      std::string gerr;
      bool ok = true;

      if (!deferred_gpu_out.nvcv_img || !deferred_gpu_out.cuda ||
          !deferred_gpu_out.nvcv) {
        ok = false;
        gerr = "Deferred GPU output reference is incomplete.";
      }

      if (ok) {
        if (gpu_resize_cuda != deferred_gpu_out.cuda) {
          gpu_resize_cuda = deferred_gpu_out.cuda;
          if (!gpu_resize_bilinear.Initialize(gpu_resize_cuda, &gerr)) {
            ok = false;
          }
        }
      }

      auto ensure_scaled_gpu = [&](unsigned w, unsigned h) {
        if (!ok)
          return;
        auto &nvcv = *deferred_gpu_out.nvcv;
        if (!nvcv.IsInitialized() || !nvcv.f().NvCVImage_Alloc) {
          ok = false;
          gerr = "NvCVImage runtime not initialized.";
          return;
        }

        if (gpu_bgr_scaled_allocated) {
          if (gpu_bgr_scaled.width == w && gpu_bgr_scaled.height == h &&
              gpu_bgr_scaled.pixelFormat == studiocast::maxine::NVCV_BGR &&
              gpu_bgr_scaled.componentType == studiocast::maxine::NVCV_U8 &&
              gpu_bgr_scaled.planar == studiocast::maxine::NVCV_CHUNKY &&
              gpu_bgr_scaled.gpuMem == studiocast::maxine::NVCV_GPU) {
            return;
          }

          if (nvcv.f().NvCVImage_Realloc) {
            const auto st = nvcv.f().NvCVImage_Realloc(
                &gpu_bgr_scaled, w, h, studiocast::maxine::NVCV_BGR,
                studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
                studiocast::maxine::NVCV_GPU,
                /*alignment=*/0);
            if (st == studiocast::maxine::NVCV_SUCCESS) {
              return;
            }
          }

          if (nvcv.f().NvCVImage_Dealloc) {
            (void)nvcv.f().NvCVImage_Dealloc(&gpu_bgr_scaled);
          }
          gpu_bgr_scaled = studiocast::maxine::NvCVImage{};
          gpu_bgr_scaled_allocated = false;
        }

        const auto st = nvcv.f().NvCVImage_Alloc(
            &gpu_bgr_scaled, w, h, studiocast::maxine::NVCV_BGR,
            studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
            studiocast::maxine::NVCV_GPU,
            /*alignment=*/0);
        if (st != studiocast::maxine::NVCV_SUCCESS) {
          ok = false;
          gerr = "NvCVImage_Alloc(gpu scaled) failed: " + std::to_string(st);
          return;
        }
        gpu_bgr_scaled_allocated = true;
        gpu_resize_nvcv = deferred_gpu_out.nvcv;
      };

      ensure_scaled_gpu(static_cast<unsigned>(outW),
                        static_cast<unsigned>(outH));

      if (ok) {
        if (!gpu_resize_bilinear.Resize(*deferred_gpu_out.nvcv_img,
                                        &gpu_bgr_scaled,
                                        deferred_gpu_out.stream, &gerr)) {
          ok = false;
        }
      }

      if (ok) {
        if (!deferred_gpu_out.nvcv->f().NvCVImage_Init ||
            !deferred_gpu_out.nvcv->f().NvCVImage_Transfer) {
          ok = false;
          gerr = "NvCVImage_Init/Transfer unavailable.";
        }
      }

      if (ok) {
        if (outW <= 0 || outH <= 0) {
          ok = false;
          gerr = "Invalid output size for GPU resize: " + std::to_string(outW) +
                 "x" + std::to_string(outH) + ".";
        }
      }

      if (ok) {
        const unsigned out_w = static_cast<unsigned>(outW);
        const unsigned out_h = static_cast<unsigned>(outH);
        const std::size_t tightStride = static_cast<std::size_t>(outW) * 3u;
        bgr_scaled_out.resize(tightStride * static_cast<std::size_t>(outH));

        const auto init = deferred_gpu_out.nvcv->f().NvCVImage_Init(
            &cpu_bgr_scaled, out_w, out_h, static_cast<int>(tightStride),
            bgr_scaled_out.data(), studiocast::maxine::NVCV_BGR,
            studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
            studiocast::maxine::NVCV_CPU);
        if (init != studiocast::maxine::NVCV_SUCCESS) {
          ok = false;
          gerr = "NvCVImage_Init(cpu scaled) failed: " + std::to_string(init);
        }

        if (ok) {
          const auto down = deferred_gpu_out.nvcv->f().NvCVImage_Transfer(
              &gpu_bgr_scaled, &cpu_bgr_scaled, 1.0f, deferred_gpu_out.stream,
              nullptr);
          if (down != studiocast::maxine::NVCV_SUCCESS) {
            ok = false;
            gerr = "NvCVImage_Transfer(gpu->cpu scaled) failed: " +
                   std::to_string(down);
          }
        }

        if (ok) {
          if (!deferred_gpu_out.cuda) {
            ok = false;
            gerr = "Deferred GPU resize readback: missing CUDA driver API.";
          } else if (!SyncAfterGpuToCpuTransfer(*deferred_gpu_out.cuda,
                                                deferred_gpu_out.stream,
                                                "Deferred GPU resize readback: "
                                                "after gpu->cpu transfer",
                                                &gerr)) {
            ok = false;
          }
        }

        if (ok) {
          rgbScaled.resize(tightStride * static_cast<std::size_t>(outH));
          studiocast::video::Bgr24ToRgb24(bgr_scaled_out.data(),
                                          rgbScaled.data(), outW, outH,
                                          tightStride, tightStride);
          mark_maxine_active_frame();
          ++maxine_deferred_readbacks;
          ++maxine_download_calls;
          ++maxine_final_download_calls;
          ++maxine_forced_sync_calls;
          ++maxine_bgr_to_rgb_calls;
          frameW = outW;
          frameH = outH;
          rgbOut = rgbScaled.data();
          rgbOutStride = tightStride;
          rgbOutBytes = rgbScaled.size();
        }
      }

      if (!ok) {
        // Best-effort: transfer the unscaled GPU frame back to the pipeline RGB
        // buffer so we can continue with other scaling backends (or fail fast
        // if CPU resize is disallowed).
        bool readback_ok = false;
        std::string derr;
        if (deferred_gpu_out.nvcv_img && deferred_gpu_out.nvcv &&
            deferred_gpu_out.cuda) {
          const auto &nvcv_f = deferred_gpu_out.nvcv->f();
          if (deferred_gpu_out.nvcv->IsInitialized() && nvcv_f.NvCVImage_Init &&
              nvcv_f.NvCVImage_Transfer) {
            const std::size_t srcTightStride =
                static_cast<std::size_t>(frameW) * 3u;
            std::vector<std::uint8_t> bgr_readback;
            bgr_readback.resize(srcTightStride *
                                static_cast<std::size_t>(frameH));
            studiocast::maxine::NvCVImage cpu_bgr_readback{};

            const auto init = nvcv_f.NvCVImage_Init(
                &cpu_bgr_readback, static_cast<unsigned>(frameW),
                static_cast<unsigned>(frameH), static_cast<int>(srcTightStride),
                bgr_readback.data(), studiocast::maxine::NVCV_BGR,
                studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
                studiocast::maxine::NVCV_CPU);
            if (init == studiocast::maxine::NVCV_SUCCESS) {
              const auto down = nvcv_f.NvCVImage_Transfer(
                  deferred_gpu_out.nvcv_img, &cpu_bgr_readback, 1.0f,
                  deferred_gpu_out.stream, nullptr);
              if (down == studiocast::maxine::NVCV_SUCCESS) {
                std::string sync_err;
                if (SyncAfterGpuToCpuTransfer(
                        *deferred_gpu_out.cuda, deferred_gpu_out.stream,
                        "Deferred NvCV readback after resize failure",
                        &sync_err)) {
                  studiocast::video::Bgr24ToRgb24(bgr_readback.data(),
                                                  rgb.data(), frameW, frameH,
                                                  srcTightStride, rgbStride);
                  mark_maxine_active_frame();
                  ++maxine_download_calls;
                  ++maxine_cpu_continuation_download_calls;
                  ++maxine_forced_sync_calls;
                  ++maxine_bgr_to_rgb_calls;
                  readback_ok = true;
                  have_deferred_gpu_out = false;
                } else {
                  derr = sync_err;
                }
              } else {
                derr = "NvCVImage_Transfer(gpu->cpu) failed: " +
                       std::to_string(down);
              }
            } else {
              derr = "NvCVImage_Init(cpu readback) failed: " +
                     std::to_string(init);
            }
          }
        }

        if (!readback_ok)
          live_effect_attribution.DiscardFrame();

        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = "Maxine/NVCV deferred GPU resize failed";
        if (!gerr.empty())
          last_error_ += ": " + gerr;
        if (!readback_ok && !derr.empty())
          last_error_ += " (and readback failed: " + derr + ")";
        last_error_ += ".";
      }
    }

    // Standalone GPU scaling backend: upload CPU RGB->GPU BGR, resize on GPU,
    // download and convert back. Skip it when CPU resize is allowed and no
    // Maxine GPU stage ran, so resize-only/no-effect frames do not pay an
    // avoidable upload/download/sync.
    if (ShouldRunStandaloneGpuScaler(
            /*scaling_needed=*/(frameW != outW || frameH != outH),
            /*gpu_backend_active=*/
            (gpu_backend == GpuResizeBackend::maxine_nvcv),
            /*have_deferred_gpu_out=*/have_deferred_gpu_out,
            /*allow_cpu_resize=*/cfg.allow_cpu_resize,
            /*same_backend_effects_ran=*/maxine_active_this_frame)) {
      std::string gerr;
      bool ok = maxine_scaler.initialized;
      if (!ok) {
        gerr = maxine_scaler.init_error.empty()
                   ? "Maxine GPU scaler not initialized."
                   : maxine_scaler.init_error;
      }

      const auto &nvcv = maxine_scaler.nvcv;
      const auto &nvcv_f = nvcv.f();
      if (ok) {
        if (!nvcv_f.NvCVImage_Init || !nvcv_f.NvCVImage_Transfer ||
            !nvcv_f.NvCVImage_Alloc || !nvcv_f.NvCVImage_Dealloc) {
          ok = false;
          gerr = "NvCVImage API incomplete.";
        }
      }

      auto ensure_gpu_bgr = [&](studiocast::maxine::NvCVImage *img,
                                bool *allocated, unsigned w, unsigned h,
                                const char *what) {
        if (!ok)
          return;

        if (*allocated) {
          if (img->width == w && img->height == h &&
              img->pixelFormat == studiocast::maxine::NVCV_BGR &&
              img->componentType == studiocast::maxine::NVCV_U8 &&
              img->planar == studiocast::maxine::NVCV_CHUNKY &&
              img->gpuMem == studiocast::maxine::NVCV_GPU) {
            return;
          }

          if (nvcv_f.NvCVImage_Realloc) {
            const auto st = nvcv_f.NvCVImage_Realloc(
                img, w, h, studiocast::maxine::NVCV_BGR,
                studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
                studiocast::maxine::NVCV_GPU,
                /*alignment=*/0);
            if (st == studiocast::maxine::NVCV_SUCCESS) {
              return;
            }
          }

          (void)nvcv_f.NvCVImage_Dealloc(img);
          *img = studiocast::maxine::NvCVImage{};
          *allocated = false;
        }

        const auto st = nvcv_f.NvCVImage_Alloc(
            img, w, h, studiocast::maxine::NVCV_BGR,
            studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
            studiocast::maxine::NVCV_GPU,
            /*alignment=*/0);
        if (st != studiocast::maxine::NVCV_SUCCESS) {
          ok = false;
          gerr = std::string("NvCVImage_Alloc(") + what +
                 ") failed: " + std::to_string(st);
          return;
        }
        *allocated = true;
      };

      if (ok) {
        ensure_gpu_bgr(&maxine_scaler.gpu_in, &maxine_scaler.gpu_in_allocated,
                       static_cast<unsigned>(frameW),
                       static_cast<unsigned>(frameH), "gpu in");
      }

      if (ok) {
        ensure_gpu_bgr(&maxine_scaler.gpu_scaled,
                       &maxine_scaler.gpu_scaled_allocated,
                       static_cast<unsigned>(outW), static_cast<unsigned>(outH),
                       "gpu scaled");
      }

      if (ok) {
        const std::size_t srcTightStride =
            static_cast<std::size_t>(frameW) * 3u;
        maxine_scaler.bgr_in.resize(srcTightStride *
                                    static_cast<std::size_t>(frameH));
        studiocast::video::Rgb24ToBgr24(rgbOut, maxine_scaler.bgr_in.data(),
                                        frameW, frameH, rgbOutStride,
                                        srcTightStride);

        const auto init = nvcv_f.NvCVImage_Init(
            &maxine_scaler.cpu_bgr_in, static_cast<unsigned>(frameW),
            static_cast<unsigned>(frameH), static_cast<int>(srcTightStride),
            maxine_scaler.bgr_in.data(), studiocast::maxine::NVCV_BGR,
            studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
            studiocast::maxine::NVCV_CPU);
        if (init != studiocast::maxine::NVCV_SUCCESS) {
          ok = false;
          gerr = "NvCVImage_Init(cpu in) failed: " + std::to_string(init);
        }

        if (ok) {
          const auto up = nvcv_f.NvCVImage_Transfer(&maxine_scaler.cpu_bgr_in,
                                                    &maxine_scaler.gpu_in, 1.0f,
                                                    nullptr, nullptr);
          if (up != studiocast::maxine::NVCV_SUCCESS) {
            ok = false;
            gerr =
                "NvCVImage_Transfer(cpu->gpu in) failed: " + std::to_string(up);
          }
        }
      }

      if (ok) {
        if (!maxine_scaler.resize.Resize(maxine_scaler.gpu_in,
                                         &maxine_scaler.gpu_scaled, nullptr,
                                         &gerr)) {
          ok = false;
        }
      }

      if (ok) {
        const std::size_t dstTightStride = static_cast<std::size_t>(outW) * 3u;
        maxine_scaler.bgr_out.resize(dstTightStride *
                                     static_cast<std::size_t>(outH));

        const auto init = nvcv_f.NvCVImage_Init(
            &maxine_scaler.cpu_bgr_out, static_cast<unsigned>(outW),
            static_cast<unsigned>(outH), static_cast<int>(dstTightStride),
            maxine_scaler.bgr_out.data(), studiocast::maxine::NVCV_BGR,
            studiocast::maxine::NVCV_U8, studiocast::maxine::NVCV_CHUNKY,
            studiocast::maxine::NVCV_CPU);
        if (init != studiocast::maxine::NVCV_SUCCESS) {
          ok = false;
          gerr = "NvCVImage_Init(cpu out) failed: " + std::to_string(init);
        }

        if (ok) {
          const auto down = nvcv_f.NvCVImage_Transfer(
              &maxine_scaler.gpu_scaled, &maxine_scaler.cpu_bgr_out, 1.0f,
              nullptr, nullptr);
          if (down != studiocast::maxine::NVCV_SUCCESS) {
            ok = false;
            gerr = "NvCVImage_Transfer(gpu->cpu out) failed: " +
                   std::to_string(down);
          }
        }

        if (ok) {
          if (!SyncAfterGpuToCpuTransfer(
                  maxine_scaler.cuda,
                  /*stream=*/nullptr,
                  "Standalone GPU scaler readback: after gpu->cpu transfer",
                  &gerr)) {
            ok = false;
          }
        }

        if (ok) {
          rgbScaled.resize(dstTightStride * static_cast<std::size_t>(outH));
          studiocast::video::Bgr24ToRgb24(maxine_scaler.bgr_out.data(),
                                          rgbScaled.data(), outW, outH,
                                          dstTightStride, dstTightStride);
          mark_maxine_active_frame();
          ++maxine_rgb_to_bgr_calls;
          ++maxine_upload_calls;
          ++maxine_standalone_scaler_upload_calls;
          ++maxine_download_calls;
          ++maxine_final_download_calls;
          ++maxine_standalone_scaler_download_calls;
          ++maxine_forced_sync_calls;
          ++maxine_bgr_to_rgb_calls;
          frameW = outW;
          frameH = outH;
          rgbOut = rgbScaled.data();
          rgbOutStride = dstTightStride;
          rgbOutBytes = rgbScaled.size();
        }
      }

      if (!ok) {
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = "Maxine/NVCV GPU scaling failed";
        if (!gerr.empty())
          last_error_ += ": " + gerr;
        last_error_ += ".";
      }
    }

    // Standalone GPU scaling backend (OpenCUDA/pure CUDA): upload CPU RGB ->
    // GPU RGB, resize on GPU, download.
    //
    // This runs when OpenCUDA scaling is selected, and also serves as a
    // fallback when Maxine/NVCV scaling fails.
    //
    // IMPORTANT: If Open CUDA effects were not active this frame and CPU resize
    // is allowed, skip this standalone scaler so we don't do unnecessary GPU
    // uploads/downloads.
    const bool gpu_backend_supports_open_cuda_scaler =
        (gpu_backend == GpuResizeBackend::open_cuda ||
         gpu_backend == GpuResizeBackend::maxine_nvcv);
    if (ShouldRunStandaloneOpenCudaScaler(
            /*scaling_needed=*/(frameW != outW || frameH != outH),
            /*gpu_backend_is_open_cuda_or_maxine=*/
            gpu_backend_supports_open_cuda_scaler,
            /*have_deferred_gpu_out=*/have_deferred_gpu_out,
            /*allow_cpu_resize=*/cfg.allow_cpu_resize,
            /*open_cuda_effects_ran=*/open_cuda_active_this_frame)) {
      mark_open_cuda_active_frame();
      std::string gerr;
      bool ok = open_cuda_scaler.EnsureInitialized(&gerr);
      if (ok) {
        if (!open_cuda_scaler.gpu_in.ReallocIfNeeded(
                &open_cuda_scaler.cuda, frameW, frameH,
                studiocast::cuda::PixelFormatGpu::rgb_u8, &gerr)) {
          ok = false;
        }
      }
      if (ok) {
        if (!open_cuda_scaler.gpu_scaled.ReallocIfNeeded(
                &open_cuda_scaler.cuda, outW, outH,
                studiocast::cuda::PixelFormatGpu::rgb_u8, &gerr)) {
          ok = false;
        }
      }
      if (ok) {
        if (!open_cuda_scaler.gpu_in.UploadFromCpuRgb24(
                &open_cuda_scaler.cuda, rgbOut, rgbOutStride,
                open_cuda_scaler.stream, &gerr)) {
          ok = false;
        } else {
          ++open_cuda_frame_upload_calls;
          ++open_cuda_standalone_scaler_upload_calls;
        }
      }
      if (ok) {
        if (!studiocast::cuda::kernels::ResizeBilinear(
                open_cuda_scaler.gpu_in, open_cuda_scaler.gpu_scaled,
                open_cuda_scaler.stream, &gerr)) {
          ok = false;
        }
      }
      if (ok) {
        const std::size_t tightStride = static_cast<std::size_t>(outW) * 3u;
        rgbScaled.resize(tightStride * static_cast<std::size_t>(outH));
        if (!open_cuda_scaler.gpu_scaled.DownloadToCpuRgb24(
                &open_cuda_scaler.cuda, rgbScaled.data(), tightStride,
                open_cuda_scaler.stream, &gerr)) {
          ok = false;
        } else {
          ++open_cuda_rgb_download_calls;
          ++open_cuda_final_download_calls;
          ++open_cuda_standalone_scaler_download_calls;
        }
      }
      if (ok) {
        if (!open_cuda_scaler.cuda.StreamSynchronize(open_cuda_scaler.stream,
                                                     &gerr)) {
          ok = false;
        } else {
          ++open_cuda_forced_sync_calls;
        }
      }
      if (ok) {
        const std::size_t tightStride = static_cast<std::size_t>(outW) * 3u;
        frameW = outW;
        frameH = outH;
        rgbOut = rgbScaled.data();
        rgbOutStride = tightStride;
        rgbOutBytes = rgbScaled.size();
      } else {
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = "OpenCUDA GPU scaling failed";
        if (!gerr.empty())
          last_error_ += ": " + gerr;
        last_error_ += ".";
      }
    }

#if STUDIOCAST_ENABLE_OPEN_VULKAN
    // Standalone GPU scaling backend (Open Vulkan): CPU RGB -> Vulkan RGBA
    // storage buffer, bilinear resize on compute queue, final RGB readback.
    //
    // The Vulkan runtime is initialized during setup/reconfiguration. This
    // frame-path block only uses the selected backend and pre-created command
    // infrastructure.
    if (ShouldRunStandaloneGpuScaler(
            /*scaling_needed=*/(frameW != outW || frameH != outH),
            /*gpu_backend_active=*/(gpu_backend == GpuResizeBackend::vulkan),
            /*have_deferred_gpu_out=*/have_deferred_gpu_out,
            /*allow_cpu_resize=*/cfg.allow_cpu_resize,
            /*same_backend_effects_ran=*/
            open_vulkan_effect_ran_this_frame)) {
      std::string gerr;
      bool ok = vulkan_scaler.initialized;
      if (!ok) {
        gerr = vulkan_scaler.init_error.empty()
                   ? "Open Vulkan GPU scaler not initialized."
                   : vulkan_scaler.init_error;
      }

      const std::size_t tightStride = static_cast<std::size_t>(outW) * 3u;
      if (ok) {
        rgbScaled.resize(tightStride * static_cast<std::size_t>(outH));
        if (!vulkan_scaler.resize.Resize(rgbOut, rgbOutStride, rgbScaled.data(),
                                         tightStride, &gerr)) {
          ok = false;
        }
      }

      if (ok) {
        mark_open_vulkan_active_frame();
        ++open_vulkan_upload_calls;
        ++open_vulkan_download_calls;
        ++open_vulkan_final_download_calls;
        ++open_vulkan_standalone_scaler_upload_calls;
        ++open_vulkan_standalone_scaler_download_calls;
        ++open_vulkan_forced_sync_calls;
        frameW = outW;
        frameH = outH;
        rgbOut = rgbScaled.data();
        rgbOutStride = tightStride;
        rgbOutBytes = rgbScaled.size();
      } else {
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = "Open Vulkan GPU scaling failed";
        if (!gerr.empty())
          last_error_ += ": " + gerr;
        last_error_ += ".";
      }
    }
#endif

    if (frameW != outW || frameH != outH) {
      if (!cfg.allow_cpu_resize) {
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ =
            OutputResizeDisallowedErrorMessage(frameW, frameH, outW, outH);
        break;
      }
      std::string resizeErr;
      const std::size_t tightDstStride = static_cast<std::size_t>(outW) * 3u;
      if (!cpuResizePlan.Configure(frameW, frameH, outW, outH, &resizeErr) ||
          !cpuResizePlan.Apply(rgbOut, rgbOutStride, &rgbScaled, tightDstStride,
                               &resizeErr)) {
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ = "Resize failed: " + resizeErr;
        break;
      }
      rgbOut = rgbScaled.data();
      rgbOutStride = tightDstStride;
      rgbOutBytes = rgbScaled.size();
    }

    const auto t_scale_end = Clock::now();
    const double scale_ms =
        scaling_needed ? ToMs(t_scale_end - t_scale_start) : 0.0;

    const auto finish_prepared_hardware_frame =
        [&](bool output_write_succeeded) {
          const bool prepared_transition =
              prepared_hardware_completion_pending ||
              prepared_maxine_completion_pending ||
              prepared_open_cuda_completion_pending;
          bool committed = output_write_succeeded;
          if (prepared_hardware_completion_pending) {
            hal::ReasonId reason = hal::ReasonId::none;
            const hal::StatusId status =
                prepared_hardware_composition.CommitOutput(
                    prepared_hardware_completion, output_write_succeeded,
                    &reason);
            prepared_hardware_completion_pending = false;
            committed &= status == hal::StatusId::ready;
          }
          if (prepared_maxine_completion_pending) {
            hal::ReasonId reason = hal::ReasonId::none;
            const hal::StatusId status =
                prepared_maxine_hardware_owner.CommitOutput(
                    prepared_maxine_completion, output_write_succeeded,
                    &reason);
            prepared_maxine_completion_pending = false;
            committed &= status == hal::StatusId::ready;
          }
          if (prepared_open_cuda_completion_pending) {
            hal::ReasonId reason = hal::ReasonId::none;
            const hal::StatusId status =
                prepared_open_cuda_hardware_owner.CommitOutput(
                    prepared_open_cuda_completion, output_write_succeeded,
                    &reason);
            prepared_open_cuda_completion_pending = false;
            committed &= status == hal::StatusId::ready;
          }
          if (prepared_transition) {
            publish_video_hal_status(appliedFx,
                                     appliedEffectsGeneration,
                                     VideoRuntimeLifecycle::running);
          }
          return committed;
        };

    // Output pacing: enforce negotiated output FPS when we're ahead of
    // schedule to avoid bursty cadence (helps browser/WebRTC capture).
    if (pace_period != Clock::duration::zero()) {
      const auto now = Clock::now();

      if (!have_next_pace_deadline) {
        next_pace_deadline = now;
        have_next_pace_deadline = true;
      } else if (now > (next_pace_deadline + pace_period)) {
        // More than one frame late: resync to avoid "catch up" bursts.
        next_pace_deadline = now;
        ++pace_resyncs;
      }

      if (now < next_pace_deadline) {
        const auto sleep_d = next_pace_deadline - now;
        ema_pace_sleep.Add(ToMs(sleep_d));
        ++pace_sleeps;
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait_until(lock, next_pace_deadline,
                       [this] { return stop_.load(); });
      } else if (now > next_pace_deadline) {
        const auto late_d = now - next_pace_deadline;
        ema_pace_late.Add(ToMs(late_d));
        ++pace_late_frames;
      }

      next_pace_deadline += pace_period;
    }

    if (stop_.load()) {
      (void)finish_prepared_hardware_frame(false);
      break;
    }

    const auto t_write_start = Clock::now();

    if (raw_yuyv_passthrough_frame &&
        effects_generation_published_.load(std::memory_order_acquire) !=
            appliedEffectsGeneration) {
      // An effect change raced with this prepared raw frame. Drop it before
      // the write boundary; the next loop rebuilds and uses the RGB path.
      (void)finish_prepared_hardware_frame(false);
      continue;
    }

    const std::uint8_t *output_data = nullptr;
    std::size_t output_bytes = 0;
    if (outA.format == PixelFormat::rgb24) {
      const std::size_t wantRow = static_cast<std::size_t>(outW) * 3u;

      if (outA.bytes_per_line == wantRow && outA.size_image == rgbOutBytes) {
        output_data = rgbOut;
        output_bytes = rgbOutBytes;
      } else {
        // Copy into padded output buffer.
        for (int y = 0; y < outH; ++y) {
          const std::uint8_t *srcRow =
              rgbOut + static_cast<std::size_t>(y) * rgbOutStride;
          std::uint8_t *dstRow =
              outBuf.data() + static_cast<std::size_t>(y) * outA.bytes_per_line;

          for (std::size_t i = 0; i < wantRow; ++i)
            dstRow[i] = srcRow[i];
        }

        output_data = outBuf.data();
        output_bytes = outBuf.size();
      }
    } else {
      if (!raw_yuyv_passthrough_frame) {
        // Output is YUYV; convert RGB -> YUYV into outBuf with output stride.
        CountedRgb24ToYuyvWithScratch(
            rgbOut, outW, outH, rgbOutStride, outBuf.data(),
            outA.bytes_per_line,
            yuyvConversionScratch.empty() ? nullptr
                                          : yuyvConversionScratch.data(),
            yuyvConversionScratch.size(), &yuyv_frame_path_counters);
      }

      output_data = outBuf.data();
      output_bytes = outBuf.size();
    }

    const FrameWriteResult write_result =
        writer_.WriteFrameDetailed(output_data, output_bytes, &stop_);
    if (write_result.status == FrameWriteStatus::stopped) {
      (void)finish_prepared_hardware_frame(false);
      live_effect_attribution.DiscardFrame();
      std::lock_guard<std::mutex> lock(mu_);
      PublishWriterCountersLocked();
      break;
    }
    if (write_result.status == FrameWriteStatus::would_block_dropped) {
      // Zero-queue latest-frame policy: this frame is discarded and the next
      // capture replaces it. Do not probe or format an error on this path.
      (void)finish_prepared_hardware_frame(false);
      live_effect_attribution.DiscardFrame();
      {
        std::lock_guard<std::mutex> lock(mu_);
        PublishWriterCountersLocked();
      }
      continue;
    }
    if (!write_result.FrameCommitted()) {
      // A consumer-driven format change is the sole write failure that may
      // trigger an in-place refresh. Transport/disconnect and partial-frame
      // failures remain fatal and visible.
      if (write_result.ShouldRefreshFormat()) {
        (void)finish_prepared_hardware_frame(false);
        if (RefreshOutputActual("write_format_mismatch")) {
          ++output_write_recoveries;
          live_effect_attribution.DiscardFrame();
          continue;
        }
      }
      (void)finish_prepared_hardware_frame(false);
      const std::string werr = DescribeFrameWriteResult(write_result);
      std::lock_guard<std::mutex> lock(mu_);
      PublishWriterCountersLocked();
      last_error_ = "Write failed: " + werr;
      break;
    }

    const bool had_prepared_vulkan_completion =
        prepared_hardware_completion_pending;
    const bool had_prepared_maxine_completion =
        prepared_maxine_completion_pending;
    const bool had_prepared_open_cuda_completion =
        prepared_open_cuda_completion_pending;
    if (had_prepared_vulkan_completion ||
        had_prepared_maxine_completion ||
        had_prepared_open_cuda_completion) {
      if (!finish_prepared_hardware_frame(true)) {
        live_effect_attribution.DiscardFrame();
        std::lock_guard<std::mutex> lock(mu_);
        last_error_ =
            "Prepared hardware output attribution commit failed after writer "
            "success.";
        break;
      }
      if (had_prepared_vulkan_completion) {
        const auto committed =
            prepared_hardware_snapshot().latest_committed_frame;
        for (std::size_t index = 0; index < committed.stage_count; ++index) {
          const auto &stage = committed.stages[index];
          if (stage.status != hal::StageExecutionStatus::succeeded) {
            if (stage.status ==
                hal::StageExecutionStatus::optional_failed_open) {
              const std::string failure =
                  "Prepared Vulkan HAL optional stage failed open [" +
                  std::string(PreparedHardwareReasonCode(stage.reason)) +
                  "].";
              if (stage.semantic == hal::SemanticEffectId::vignette) {
                block_optional_effect(
                    OptionalEffectSlot::vignette,
                    studiocast::video::effects::contract::kEffectIdVignette,
                    "open_vulkan", failure, f.sequence);
              } else if (stage.semantic == hal::SemanticEffectId::mirror) {
                block_optional_effect(
                    OptionalEffectSlot::mirror,
                    studiocast::video::effects::contract::kEffectIdMirror,
                    "open_vulkan", failure, f.sequence);
              }
            }
            continue;
          }
          if (stage.semantic == hal::SemanticEffectId::vignette) {
            clear_optional_effect_on_success(
                optional_breaker(OptionalEffectSlot::vignette), f.sequence);
            mark_live_effect_success(
                studiocast::video::effects::contract::kEffectIdVignette,
                "open_vulkan");
          } else if (stage.semantic == hal::SemanticEffectId::mirror) {
            clear_optional_effect_on_success(
                optional_breaker(OptionalEffectSlot::mirror), f.sequence);
            mark_live_effect_success(
                studiocast::video::effects::contract::kEffectIdMirror,
                "open_vulkan");
          }
        }
      }
      if (had_prepared_maxine_completion) {
        const auto maxine_snapshot =
            prepared_maxine_hardware_owner.Snapshot();
        const auto &committed =
            maxine_snapshot.latest_committed_frame;
        const auto &active_program = maxine_snapshot.active_program;
        if (!committed.valid ||
            committed.program_id != active_program.program_id ||
            committed.invalidation_key !=
                active_program.invalidation_key ||
            committed.execution_domain !=
                active_program.execution_domain) {
          live_effect_attribution.DiscardFrame();
          std::lock_guard<std::mutex> lock(mu_);
          last_error_ =
              "Prepared Maxine HAL committed attribution is stale.";
          break;
        }
        bool maxine_effect_succeeded = false;
        for (std::size_t index = 0; index < committed.stage_count; ++index) {
          const auto &stage = committed.stages[index];
          if (stage.status != hal::StageExecutionStatus::succeeded)
            continue;
          maxine_effect_succeeded = true;
          switch (stage.semantic) {
          case hal::SemanticEffectId::virtual_background_blur:
            clear_optional_effect_on_success(
                optional_breaker(OptionalEffectSlot::virtual_background),
                f.sequence);
            mark_live_effect_success(
                studiocast::video::effects::contract::
                    kEffectIdVirtualBackgroundBlur,
                "maxine");
            break;
          case hal::SemanticEffectId::virtual_background_remove:
            clear_optional_effect_on_success(
                optional_breaker(OptionalEffectSlot::virtual_background),
                f.sequence);
            mark_live_effect_success(
                studiocast::video::effects::contract::
                    kEffectIdVirtualBackgroundRemove,
                "maxine");
            break;
          case hal::SemanticEffectId::virtual_background_replace:
            clear_optional_effect_on_success(
                optional_breaker(OptionalEffectSlot::virtual_background),
                f.sequence);
            mark_live_effect_success(
                studiocast::video::effects::contract::
                    kEffectIdVirtualBackgroundReplace,
                "maxine");
            break;
          case hal::SemanticEffectId::virtual_key_light:
            clear_optional_effect_on_success(
                optional_breaker(OptionalEffectSlot::relight), f.sequence);
            mark_live_effect_success(
                studiocast::video::effects::contract::
                    kEffectIdVirtualKeyLight,
                "maxine");
            break;
          case hal::SemanticEffectId::vignette:
            clear_optional_effect_on_success(
                optional_breaker(OptionalEffectSlot::vignette), f.sequence);
            mark_live_effect_success(
                studiocast::video::effects::contract::kEffectIdVignette,
                "cuda");
            break;
          case hal::SemanticEffectId::video_noise_removal:
          case hal::SemanticEffectId::eye_contact:
          case hal::SemanticEffectId::auto_frame:
          case hal::SemanticEffectId::mirror:
          case hal::SemanticEffectId::invalid:
            break;
          }
        }
        if (maxine_effect_succeeded) {
          (void)MarkGpuBackendActiveFrame(maxine_active_this_frame,
                                          maxine_active_frames);
        }
      }
      if (had_prepared_open_cuda_completion) {
        const auto snapshot =
            prepared_open_cuda_hardware_owner.Snapshot();
        const auto &committed = snapshot.latest_committed_frame;
        const auto &active_program = snapshot.active_program;
        if (!committed.valid ||
            committed.program_id != active_program.program_id ||
            committed.invalidation_key !=
                active_program.invalidation_key ||
            committed.execution_domain !=
                active_program.execution_domain ||
            committed.stage_count != 1) {
          live_effect_attribution.DiscardFrame();
          std::lock_guard<std::mutex> lock(mu_);
          last_error_ =
              "Prepared Open CUDA HAL committed attribution is stale.";
          break;
        }
        const auto &stage = committed.stages[0];
        std::string_view effect_id;
        if (stage.semantic ==
            hal::SemanticEffectId::virtual_background_blur) {
          effect_id = effects::contract::kEffectIdVirtualBackgroundBlur;
        } else if (stage.semantic ==
                   hal::SemanticEffectId::virtual_background_remove) {
          effect_id = effects::contract::kEffectIdVirtualBackgroundRemove;
        }
        if (PreparedOpenCudaPipelineMayCommitAttribution(
                stage.status, write_result.FrameCommitted(),
                hal::StatusId::ready) &&
            !effect_id.empty()) {
          mark_open_cuda_active_frame();
          clear_optional_effect_on_success(
              optional_breaker(OptionalEffectSlot::virtual_background),
              f.sequence);
          mark_live_effect_success(effect_id, "open_cuda");
        } else if (
            stage.status ==
                hal::StageExecutionStatus::optional_failed_open &&
            !effect_id.empty()) {
          block_optional_effect(
              OptionalEffectSlot::virtual_background, effect_id,
              "open_cuda",
              "Prepared Open CUDA HAL optional stage failed open [" +
                  std::string(PreparedHardwareReasonCode(stage.reason)) +
                  "].",
              f.sequence);
        }
      }
    }

    if (live_effect_attribution.CommitSuccessfulFrame(
            appliedPlan.ordered_effect_ids)) {
      std::lock_guard<std::mutex> lock(mu_);
      if (running_ && !stop_.load())
        effects_backends_ = live_effect_attribution.active_backends();
    }

    const auto t_write_end = Clock::now();
    const double write_ms = ToMs(t_write_end - t_write_start);

    // Approximate end-to-end latency (driver timestamp -> end of write).
    double latency_ms = 0.0;
    bool have_latency_sample = false;
    if (f.timestamp_ns != 0) {
      const std::uint64_t now_ns =
          f.timestamp_monotonic ? NowNsMonotonic() : NowNsRealtime();
      if (now_ns > f.timestamp_ns) {
        latency_ms = static_cast<double>(now_ns - f.timestamp_ns) / 1000000.0;
        have_latency_sample = true;
      }
    }

    ema_capture.Add(capture_ms);
    ema_effects.Add(effects_ms);
    ema_scale.Add(scale_ms);
    ema_write.Add(write_ms);
    if (have_latency_sample)
      ema_latency.Add(latency_ms);

    ++perf_frames;

    if (debug_open_cuda_transfers && ((perf_frames % 120) == 0)) {
      std::fprintf(
          stderr,
          "Open CUDA transfers (pipeline): active_frames=%llu "
          "upload_calls=%llu download_calls=%llu final_downloads=%llu "
          "alpha_downloads=%llu denoise_tensor_uploads=%llu "
          "denoise_tensor_downloads=%llu denoise_cpu_tails=%llu "
          "cpu_tail_stages=%llu "
          "auto_frame_face_tracking=%llu auto_frame_matte_tracking=%llu "
          "auto_frame_cpu_crop=%llu forced_syncs=%llu\n",
          static_cast<unsigned long long>(open_cuda_active_frames),
          static_cast<unsigned long long>(
              open_cuda_frame_upload_calls +
              open_cuda_vb.matte_frame_upload_calls +
              open_cuda_denoise_tensor_upload_calls),
          static_cast<unsigned long long>(
              open_cuda_rgb_download_calls +
              open_cuda_denoise_tensor_download_calls),
          static_cast<unsigned long long>(open_cuda_final_download_calls),
          static_cast<unsigned long long>(open_cuda_vb.alpha_download_calls),
          static_cast<unsigned long long>(
              open_cuda_denoise_tensor_upload_calls),
          static_cast<unsigned long long>(
              open_cuda_denoise_tensor_download_calls),
          static_cast<unsigned long long>(open_cuda_denoise_cpu_tail_calls),
          static_cast<unsigned long long>(open_cuda_cpu_tail_stage_calls),
          static_cast<unsigned long long>(
              open_cuda_cpu_tail_auto_frame_face_tracking_calls),
          static_cast<unsigned long long>(
              open_cuda_cpu_tail_auto_frame_matte_tracking_calls),
          static_cast<unsigned long long>(
              open_cuda_cpu_tail_auto_frame_cpu_crop_calls),
          static_cast<unsigned long long>(open_cuda_forced_sync_calls +
                                          open_cuda_vb.forced_sync_calls));
    }

    if (debug_maxine_transfers && ((perf_frames % 120) == 0)) {
      std::fprintf(
          stderr,
          "Maxine transfers (pipeline): active_frames=%llu upload_calls=%llu "
          "download_calls=%llu final_downloads=%llu green_screen=%llu "
          "duplicate_green_screen=%llu shared_green_screen_reuse=%llu "
          "shared_green_screen_incompatible=%llu "
          "shared_green_screen_input_incompatible=%llu forced_syncs=%llu\n",
          static_cast<unsigned long long>(maxine_active_frames),
          static_cast<unsigned long long>(maxine_upload_calls),
          static_cast<unsigned long long>(maxine_download_calls),
          static_cast<unsigned long long>(maxine_final_download_calls),
          static_cast<unsigned long long>(maxine_green_screen_calls),
          static_cast<unsigned long long>(maxine_duplicate_green_screen_calls),
          static_cast<unsigned long long>(
              maxine_shared_green_screen_matte_reuse_calls),
          static_cast<unsigned long long>(
              maxine_shared_green_screen_matte_incompatible_calls),
          static_cast<unsigned long long>(
              maxine_shared_green_screen_input_incompatible_calls),
          static_cast<unsigned long long>(maxine_forced_sync_calls));
    }

    double fps_actual = 0.0;
    if (have_last_frame_end) {
      const double period_ms = ToMs(t_write_end - last_frame_end);
      if (period_ms > 0.0) {
        ema_period.Add(period_ms);
        const double avg_period = ema_period.ValueOrZero();
        if (avg_period > 0.0)
          fps_actual = 1000.0 / avg_period;
      }
    }
    last_frame_end = t_write_end;
    have_last_frame_end = true;

    ++frameIndex;
    const bool publish_perf = (frameIndex == 1) || ((frameIndex % 10) == 0);
    if (publish_perf) {
      std::lock_guard<std::mutex> lock(mu_);
      frame_index_ = frameIndex;
      ms_per_frame_.capture = ema_capture.ValueOrZero();
      ms_per_frame_.scale = ema_scale.ValueOrZero();
      ms_per_frame_.effects = ema_effects.ValueOrZero();
      ms_per_frame_.write = ema_write.ValueOrZero();
      fps_actual_ = fps_actual;
      perf_sample_frames_ = perf_frames;

      debug_.latency_ms = ema_latency.ValueOrZero();
      debug_.capture_sequence = last_capture_sequence;
      debug_.dropped_capture_frames = dropped_capture_frames_total;

      debug_.output_format_changes = output_format_changes;
      debug_.output_refresh_failures = output_refresh_failures;
      debug_.output_write_recoveries = output_write_recoveries;
      PublishWriterCountersLocked();
      debug_.yuyv_capture_to_rgb_calls =
          yuyv_frame_path_counters.capture_to_rgb_calls;
      debug_.yuyv_output_from_rgb_calls =
          yuyv_frame_path_counters.output_from_rgb_calls;
      debug_.raw_yuyv_passthrough_frames =
          yuyv_frame_path_counters.raw_passthrough_frames;
      debug_.raw_yuyv_passthrough_bytes =
          yuyv_frame_path_counters.raw_passthrough_bytes;
      debug_.effects_preparation = effects_preparation_counters;

      debug_.pace_sleep_ms = ema_pace_sleep.ValueOrZero();
      debug_.pace_late_ms = ema_pace_late.ValueOrZero();
      debug_.pace_sleeps = pace_sleeps;
      debug_.pace_late_frames = pace_late_frames;
      debug_.pace_resyncs = pace_resyncs;

      open_cuda_transfers_.active_frames = open_cuda_active_frames;
      open_cuda_transfers_.upload_calls =
          open_cuda_frame_upload_calls + open_cuda_vb.matte_frame_upload_calls +
          open_cuda_denoise_tensor_upload_calls;
      open_cuda_transfers_.download_calls =
          open_cuda_rgb_download_calls + open_cuda_vb.alpha_download_calls +
          open_cuda_denoise_tensor_download_calls;
      open_cuda_transfers_.final_download_calls =
          open_cuda_final_download_calls;
      open_cuda_transfers_.cpu_continuation_download_calls =
          open_cuda_cpu_continuation_download_calls;
      open_cuda_transfers_.alpha_download_calls =
          open_cuda_vb.alpha_download_calls;
      open_cuda_transfers_.matte_frame_upload_calls =
          open_cuda_vb.matte_frame_upload_calls;
      open_cuda_transfers_.matting_inference_calls =
          open_cuda_vb.matting_inference_calls;
      open_cuda_transfers_.standalone_scaler_upload_calls =
          open_cuda_standalone_scaler_upload_calls;
      open_cuda_transfers_.standalone_scaler_download_calls =
          open_cuda_standalone_scaler_download_calls;
      open_cuda_transfers_.denoise_tensor_upload_calls =
          open_cuda_denoise_tensor_upload_calls;
      open_cuda_transfers_.denoise_tensor_download_calls =
          open_cuda_denoise_tensor_download_calls;
      open_cuda_transfers_.forced_sync_calls =
          open_cuda_forced_sync_calls + open_cuda_vb.forced_sync_calls;
      open_cuda_transfers_.cpu_tail_stage_calls =
          open_cuda_cpu_tail_stage_calls;
      open_cuda_transfers_.cpu_tail_key_light_calls =
          open_cuda_cpu_tail_key_light_calls;
      open_cuda_transfers_.cpu_tail_auto_frame_calls =
          open_cuda_cpu_tail_auto_frame_calls;
      open_cuda_transfers_.cpu_tail_auto_frame_face_tracking_calls =
          open_cuda_cpu_tail_auto_frame_face_tracking_calls;
      open_cuda_transfers_.cpu_tail_auto_frame_matte_tracking_calls =
          open_cuda_cpu_tail_auto_frame_matte_tracking_calls;
      open_cuda_transfers_.cpu_tail_auto_frame_cpu_crop_calls =
          open_cuda_cpu_tail_auto_frame_cpu_crop_calls;
      open_cuda_transfers_.cpu_tail_denoise_calls =
          open_cuda_denoise_cpu_tail_calls;

      const auto prepared_vulkan_compat =
          ProjectPreparedHardwareCompatibilityTelemetry(
              prepared_hardware_snapshot());
      open_vulkan_transfers_.active_frames =
          open_vulkan_active_frames + prepared_vulkan_compat.active_frames;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
      open_vulkan_transfers_.upload_calls =
          open_vulkan_upload_calls +
          open_vulkan_vb.replace_effect_counters.asset_upload_calls +
          prepared_vulkan_compat.upload_calls;
      open_vulkan_transfers_.download_calls =
          open_vulkan_download_calls + open_vulkan_vb.alpha_download_calls +
          prepared_vulkan_compat.download_calls;
#else
      open_vulkan_transfers_.upload_calls =
          open_vulkan_upload_calls + prepared_vulkan_compat.upload_calls;
      open_vulkan_transfers_.download_calls =
          open_vulkan_download_calls + prepared_vulkan_compat.download_calls;
#endif
      open_vulkan_transfers_.final_download_calls =
          open_vulkan_final_download_calls +
          prepared_vulkan_compat.final_download_calls;
      open_vulkan_transfers_.cpu_continuation_download_calls =
          open_vulkan_cpu_continuation_download_calls;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
      open_vulkan_transfers_.alpha_download_calls =
          open_vulkan_vb.alpha_download_calls;
      open_vulkan_transfers_.preprocess_dispatch_calls =
          open_vulkan_vb.preprocess_dispatch_calls;
      open_vulkan_transfers_.matting_inference_calls =
          open_vulkan_vb.matting_inference_calls;
      open_vulkan_transfers_.alpha_resize_dispatch_calls =
          open_vulkan_vb.alpha_resize_dispatch_calls;
      open_vulkan_transfers_.blur_dispatch_calls =
          open_vulkan_vb.blur_dispatch_calls;
      open_vulkan_transfers_.virtual_background_blur_dispatch_calls =
          open_vulkan_vb.blur_effect_counters.blur_composite_dispatch_calls;
      open_vulkan_transfers_.virtual_background_blur_alpha_readback_calls =
          open_vulkan_vb.blur_effect_counters.alpha_readback_calls;
      open_vulkan_transfers_.virtual_background_blur_cpu_fallback_calls =
          open_vulkan_vb.blur_effect_counters.cpu_fallback_calls;
      open_vulkan_transfers_.virtual_background_blur_runtime_failure_frames =
          open_vulkan_vb.blur_effect_counters.runtime_failure_frames;
      open_vulkan_transfers_.virtual_background_blur_device_loss_frames =
          open_vulkan_vb.blur_effect_counters.device_loss_frames;
      open_vulkan_transfers_.virtual_background_remove_dispatch_calls =
          open_vulkan_vb.remove_effect_counters.solid_composite_dispatch_calls;
      open_vulkan_transfers_.virtual_background_remove_alpha_readback_calls =
          open_vulkan_vb.remove_effect_counters.alpha_readback_calls;
      open_vulkan_transfers_.virtual_background_remove_cpu_fallback_calls =
          open_vulkan_vb.remove_effect_counters.cpu_fallback_calls;
      open_vulkan_transfers_.virtual_background_remove_runtime_failure_frames =
          open_vulkan_vb.remove_effect_counters.runtime_failure_frames;
      open_vulkan_transfers_.virtual_background_remove_device_loss_frames =
          open_vulkan_vb.remove_effect_counters.device_loss_frames;
      open_vulkan_transfers_
          .virtual_background_replace_asset_allocation_calls =
          open_vulkan_vb.replace_effect_counters.asset_allocation_calls;
      open_vulkan_transfers_.virtual_background_replace_asset_decode_calls =
          open_vulkan_vb.replace_effect_counters.asset_decode_calls;
      open_vulkan_transfers_.virtual_background_replace_asset_upload_calls =
          open_vulkan_vb.replace_effect_counters.asset_upload_calls;
      open_vulkan_transfers_
          .virtual_background_replace_asset_resize_dispatch_calls =
          open_vulkan_vb.replace_effect_counters.asset_resize_dispatch_calls;
      open_vulkan_transfers_.virtual_background_replace_dispatch_calls =
          open_vulkan_vb.replace_effect_counters
              .replacement_composite_dispatch_calls;
      open_vulkan_transfers_
          .virtual_background_replace_alpha_readback_calls =
          open_vulkan_vb.replace_effect_counters.alpha_readback_calls;
      open_vulkan_transfers_.virtual_background_replace_cpu_fallback_calls =
          open_vulkan_vb.replace_effect_counters.cpu_fallback_calls;
      open_vulkan_transfers_
          .virtual_background_replace_runtime_failure_frames =
          open_vulkan_vb.replace_effect_counters.runtime_failure_frames;
      open_vulkan_transfers_.virtual_background_replace_device_loss_frames =
          open_vulkan_vb.replace_effect_counters.device_loss_frames;
      open_vulkan_transfers_.background_upload_calls =
          open_vulkan_vb.replace_effect_counters.asset_upload_calls;
      open_vulkan_transfers_.composite_dispatch_calls =
          open_vulkan_vb.composite_dispatch_calls;
      open_vulkan_transfers_.key_light_dispatch_calls =
          open_vulkan_vb.key_light_effect_counters.dispatch_calls;
      open_vulkan_transfers_.virtual_key_light_shared_matte_reuse_calls =
          open_vulkan_vb.key_light_effect_counters.shared_matte_reuse_calls;
      open_vulkan_transfers_
          .virtual_key_light_independent_matte_inference_calls =
          open_vulkan_vb.key_light_effect_counters
              .independent_matte_inference_calls;
      open_vulkan_transfers_.virtual_key_light_passthrough_frames =
          open_vulkan_vb.key_light_effect_counters.passthrough_frames;
      open_vulkan_transfers_.virtual_key_light_alpha_readback_calls =
          open_vulkan_vb.key_light_effect_counters.alpha_readback_calls;
      open_vulkan_transfers_.virtual_key_light_cpu_fallback_calls =
          open_vulkan_vb.key_light_effect_counters.cpu_fallback_calls;
      open_vulkan_transfers_.virtual_key_light_runtime_failure_frames =
          open_vulkan_vb.key_light_effect_counters.runtime_failure_frames;
      open_vulkan_transfers_.virtual_key_light_device_loss_frames =
          open_vulkan_vb.key_light_effect_counters.device_loss_frames;
      open_vulkan_transfers_.crop_resize_dispatch_calls =
          open_vulkan_vb.crop_resize_dispatch_calls;
      open_vulkan_transfers_.vignette_dispatch_calls =
          open_vulkan_vignette_counters.dispatch_calls;
      open_vulkan_transfers_.vignette_factor_allocation_calls =
          open_vulkan_vignette_counters.factor_allocation_calls;
      open_vulkan_transfers_.vignette_factor_generation_calls =
          open_vulkan_vignette_counters.factor_generation_calls;
      open_vulkan_transfers_.vignette_factor_upload_calls =
          open_vulkan_vignette_counters.factor_upload_calls;
      open_vulkan_transfers_.mirror_dispatch_calls =
          open_vulkan_mirror_counters.dispatch_calls;
#endif
      open_vulkan_transfers_.standalone_scaler_upload_calls =
          open_vulkan_standalone_scaler_upload_calls;
      open_vulkan_transfers_.standalone_scaler_download_calls =
          open_vulkan_standalone_scaler_download_calls;
      open_vulkan_transfers_.forced_sync_calls =
#if STUDIOCAST_ENABLE_OPEN_VULKAN
          open_vulkan_forced_sync_calls + open_vulkan_vb.forced_sync_calls +
          open_vulkan_vb.replace_effect_counters.asset_upload_calls +
          open_vulkan_vb.replace_effect_counters.asset_resize_dispatch_calls +
          prepared_vulkan_compat.forced_sync_calls;
#else
          open_vulkan_forced_sync_calls +
          prepared_vulkan_compat.forced_sync_calls;
#endif
      open_vulkan_transfers_.cpu_tail_stage_calls =
          open_vulkan_cpu_tail_stage_calls;
      open_vulkan_transfers_.cpu_tail_key_light_calls =
          open_vulkan_cpu_tail_key_light_calls;
      open_vulkan_transfers_.cpu_tail_auto_frame_calls =
          open_vulkan_cpu_tail_auto_frame_calls;
      open_vulkan_transfers_.cpu_tail_auto_frame_face_tracking_calls =
          open_vulkan_cpu_tail_auto_frame_face_tracking_calls;
      open_vulkan_transfers_.cpu_tail_auto_frame_matte_tracking_calls =
          open_vulkan_cpu_tail_auto_frame_matte_tracking_calls;
      open_vulkan_transfers_.cpu_tail_auto_frame_cpu_crop_calls =
          open_vulkan_cpu_tail_auto_frame_cpu_crop_calls;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
      open_vulkan_transfers_
          .cpu_tail_auto_frame_matte_alpha_readback_calls =
          open_vulkan_auto_frame.counters.matte_alpha_readback_calls;
      open_vulkan_transfers_.cpu_tail_auto_frame_matte_cpu_box_calls =
          open_vulkan_auto_frame.counters.matte_cpu_box_calls;
      open_vulkan_transfers_
          .cpu_tail_auto_frame_cpu_crop_plan_smoothing_calls =
          open_vulkan_auto_frame.counters.cpu_crop_plan_smoothing_calls;
      open_vulkan_transfers_
          .cpu_tail_auto_frame_cpu_resize_fallback_calls =
          open_vulkan_auto_frame.counters.cpu_resize_fallback_calls;
      open_vulkan_transfers_.auto_frame_initialization_failures =
          open_vulkan_auto_frame.counters.initialization_failures;
      open_vulkan_transfers_.auto_frame_runtime_failure_frames =
          open_vulkan_auto_frame.counters.runtime_failure_frames;
      open_vulkan_transfers_.auto_frame_device_loss_frames =
          open_vulkan_auto_frame.counters.device_loss_frames;
      open_vulkan_transfers_.auto_frame_temporal_reset_calls =
          open_vulkan_auto_frame.counters.temporal_reset_calls;
#endif
      open_vulkan_transfers_.runtime_failure_frames =
          open_vulkan_runtime_failure_frames +
          prepared_vulkan_compat.runtime_failure_frames;
#if STUDIOCAST_ENABLE_OPEN_VULKAN
      open_vulkan_transfers_.runtime_failure_frames +=
          open_vulkan_vignette_counters.runtime_failure_frames;
      open_vulkan_transfers_.runtime_failure_frames +=
          open_vulkan_mirror_counters.runtime_failure_frames;
#endif
      open_vulkan_transfers_.fallback_frames =
          open_vulkan_fallback_frames + prepared_vulkan_compat.fallback_frames;

      maxine_transfers_.active_frames = maxine_active_frames;
      maxine_transfers_.rgb_to_bgr_calls = maxine_rgb_to_bgr_calls;
      maxine_transfers_.upload_attempts = maxine_upload_attempts;
      maxine_transfers_.upload_calls = maxine_upload_calls;
      maxine_transfers_.matte_inference_attempts =
          maxine_matte_inference_attempts;
      maxine_transfers_.green_screen_calls = maxine_green_screen_calls;
      maxine_transfers_.duplicate_green_screen_calls =
          maxine_duplicate_green_screen_calls;
      maxine_transfers_.shared_green_screen_matte_reuse_calls =
          maxine_shared_green_screen_matte_reuse_calls;
      maxine_transfers_.shared_green_screen_matte_incompatible_calls =
          maxine_shared_green_screen_matte_incompatible_calls;
      maxine_transfers_.shared_green_screen_input_incompatible_calls =
          maxine_shared_green_screen_input_incompatible_calls;
      maxine_transfers_.download_calls = maxine_download_calls;
      maxine_transfers_.final_download_attempts =
          maxine_final_download_attempts;
      maxine_transfers_.final_download_calls = maxine_final_download_calls;
      maxine_transfers_.cpu_continuation_download_attempts =
          maxine_cpu_continuation_download_attempts;
      maxine_transfers_.cpu_continuation_download_calls =
          maxine_cpu_continuation_download_calls;
      maxine_transfers_.device_bridge_attempts =
          maxine_device_bridge_attempts;
      maxine_transfers_.device_bridge_calls = maxine_device_bridge_calls;
      maxine_transfers_.background_setup_upload_attempts =
          maxine_background_setup_upload_attempts;
      maxine_transfers_.background_setup_upload_calls =
          maxine_background_setup_upload_calls;
      maxine_transfers_.bgr_to_rgb_calls = maxine_bgr_to_rgb_calls;
      maxine_transfers_.deferred_readbacks = maxine_deferred_readbacks;
      maxine_transfers_.forced_sync_attempts = maxine_forced_sync_attempts;
      maxine_transfers_.forced_sync_calls = maxine_forced_sync_calls;
      maxine_transfers_.composite_attempts = maxine_composite_attempts;
      maxine_transfers_.composite_calls = maxine_composite_calls;
      maxine_transfers_.synchronous_sdk_run_attempts =
          maxine_synchronous_sdk_run_attempts;
      maxine_transfers_.synchronous_sdk_run_calls =
          maxine_synchronous_sdk_run_calls;
      maxine_transfers_.asynchronous_sdk_run_attempts =
          maxine_asynchronous_sdk_run_attempts;
      maxine_transfers_.asynchronous_sdk_run_calls =
          maxine_asynchronous_sdk_run_calls;
      maxine_transfers_.setup_attempts = maxine_resident_setup_attempts;
      maxine_transfers_.setup_successes = maxine_resident_setup_successes;
      maxine_transfers_.cpu_tail_stage_calls = maxine_cpu_tail_stage_calls;
      maxine_transfers_.runtime_failure_frames =
          maxine_runtime_failure_frames;
      maxine_transfers_.stage_attempts = maxine_stage_attempts;
      maxine_transfers_.stage_successes = maxine_stage_successes;
      maxine_transfers_.standalone_scaler_upload_calls =
          maxine_standalone_scaler_upload_calls;
      maxine_transfers_.standalone_scaler_download_calls =
          maxine_standalone_scaler_download_calls;
    }
  }

  // Cleanup cached GPU resize output.
  if (gpu_bgr_scaled_allocated && gpu_resize_nvcv &&
      gpu_resize_nvcv->IsInitialized() &&
      gpu_resize_nvcv->f().NvCVImage_Dealloc) {
    (void)gpu_resize_nvcv->f().NvCVImage_Dealloc(&gpu_bgr_scaled);
  }
  gpu_bgr_scaled = studiocast::maxine::NvCVImage{};
  gpu_bgr_scaled_allocated = false;

  // Cleanup Open CUDA GPU resize cache.
  if (gpu_rgb_scaled_allocated && gpu_rgb_scaled.Valid() &&
      open_cuda_vb.cuda.IsInitialized()) {
    std::string derr;
    (void)gpu_rgb_scaled.Free(&open_cuda_vb.cuda, &derr);
  }
  gpu_rgb_scaled.ClearMetadata();

#if STUDIOCAST_ENABLE_OPEN_VULKAN
  if (vulkan_rgb_scaled_allocated && vulkan_rgb_scaled.Valid()) {
    vulkan_rgb_scaled.Free();
  }
  vulkan_rgb_scaled_allocated = false;
#endif

  // Cleanup Open CUDA ping-pong frame buffers.
  if (open_cuda_vb.cuda.IsInitialized()) {
    if (open_cuda_frame_a.Valid()) {
      std::string derr;
      (void)open_cuda_frame_a.Free(&open_cuda_vb.cuda, &derr);
    }
    if (open_cuda_frame_b.Valid()) {
      std::string derr;
      (void)open_cuda_frame_b.Free(&open_cuda_vb.cuda, &derr);
    }
  }
  open_cuda_frame_a.ClearMetadata();
  open_cuda_frame_b.ClearMetadata();
  open_cuda_curr = nullptr;
  open_cuda_next = nullptr;
  open_cuda_uploaded_this_frame = false;

  prepared_open_cuda_hardware_owner.Shutdown();
  for (int attempt = 0;
       attempt < 8 &&
       prepared_open_cuda_hardware_owner.Snapshot().shutdown_pending;
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    prepared_open_cuda_hardware_owner.Shutdown();
  }
  publish_prepared_open_cuda_status();
  prepared_maxine_hardware_owner.Shutdown();
  prepared_maxine_hardware_live = false;
  publish_prepared_maxine_status();
  prepared_hardware_composition.Shutdown();
  prepared_hardware_final_live = false;
  prepared_open_cuda_hardware_live = false;
  publish_prepared_hardware_status();
  publish_video_hal_status(appliedFx, appliedEffectsGeneration,
                           VideoRuntimeLifecycle::idle);

  {
    std::lock_guard<std::mutex> lock(mu_);
    frame_index_ = frameIndex;
    running_ = false;
    effects_backends_.clear();

    if (!start_notified_) {
      start_notified_ = true;
      cv_.notify_all();
    }
  }
}

} // namespace studiocast::video
