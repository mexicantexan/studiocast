#include "core/video/video_runtime_status.h"

namespace studiocast::video {
namespace {

[[nodiscard]] constexpr std::size_t
EffectIndex(hal::SemanticEffectId effect) noexcept {
  for (std::size_t i = 0; i < kCanonicalVideoEffects.size(); ++i) {
    if (kCanonicalVideoEffects[i] == effect)
      return i;
  }
  return kCanonicalVideoEffectCount;
}

[[nodiscard]] constexpr std::size_t
LaneIndex(ProductCapabilityLaneId lane) noexcept {
  for (std::size_t i = 0; i < kProductCapabilityLanes.size(); ++i) {
    if (kProductCapabilityLanes[i] == lane)
      return i;
  }
  return kProductCapabilityLaneCount;
}

[[nodiscard]] constexpr effects::EffectCapabilityCode
CpuUnavailableCode(hal::SemanticEffectId effect) noexcept {
  using Code = effects::EffectCapabilityCode;
  switch (effect) {
  case hal::SemanticEffectId::video_noise_removal:
    return Code::cpu_video_noise_removal_unavailable;
  case hal::SemanticEffectId::eye_contact:
    return Code::cpu_eye_contact_unavailable;
  case hal::SemanticEffectId::virtual_background_blur:
    return Code::cpu_virtual_background_blur_unavailable;
  case hal::SemanticEffectId::virtual_background_remove:
    return Code::cpu_virtual_background_remove_unavailable;
  case hal::SemanticEffectId::virtual_background_replace:
    return Code::cpu_virtual_background_replace_unavailable;
  case hal::SemanticEffectId::virtual_key_light:
    return Code::cpu_virtual_key_light_unavailable;
  case hal::SemanticEffectId::auto_frame:
    return Code::cpu_auto_frame_unavailable;
  case hal::SemanticEffectId::vignette:
    return Code::cpu_vignette_unavailable;
  case hal::SemanticEffectId::mirror:
    return Code::cpu_mirror_unavailable;
  case hal::SemanticEffectId::invalid:
    break;
  }
  return Code::none;
}

[[nodiscard]] constexpr bool
ProviderMatchesLane(ProductCapabilityLaneId lane,
                    hal::SemanticEffectProviderId provider) noexcept {
  using Provider = hal::SemanticEffectProviderId;
  if (provider == Provider::invalid)
    return true;
  switch (lane) {
  case ProductCapabilityLaneId::maxine:
    return provider == Provider::maxine_vfx ||
           provider == Provider::maxine_ar ||
           provider == Provider::maxine_cuda_utility;
  case ProductCapabilityLaneId::open_cuda:
    return provider == Provider::open_video_onnxruntime_cuda;
  case ProductCapabilityLaneId::open_vulkan:
    return provider == Provider::open_vulkan_production_kernels;
  case ProductCapabilityLaneId::cpu:
    return provider == Provider::open_video_onnxruntime_cpu;
  case ProductCapabilityLaneId::invalid:
    break;
  }
  return false;
}

[[nodiscard]] constexpr ProductCapabilityLaneId
LaneForBackend(hal::BackendId backend) noexcept {
  switch (backend) {
  case hal::BackendId::maxine:
    return ProductCapabilityLaneId::maxine;
  case hal::BackendId::cuda:
  case hal::BackendId::open_video_onnxruntime:
    return ProductCapabilityLaneId::open_cuda;
  case hal::BackendId::vulkan:
    return ProductCapabilityLaneId::open_vulkan;
  case hal::BackendId::cpu:
    return ProductCapabilityLaneId::cpu;
  case hal::BackendId::invalid:
  case hal::BackendId::test_fake:
    break;
  }
  return ProductCapabilityLaneId::invalid;
}

[[nodiscard]] constexpr bool
IsVulkanMattingConsumer(hal::SemanticEffectId effect) noexcept {
  return effect == hal::SemanticEffectId::virtual_background_blur ||
         effect == hal::SemanticEffectId::virtual_background_remove ||
         effect == hal::SemanticEffectId::virtual_background_replace ||
         effect == hal::SemanticEffectId::virtual_key_light;
}

[[nodiscard]] constexpr ProductCapabilityDisposition
ExpectedNonProductionVulkanDisposition(hal::SemanticEffectId effect) noexcept {
  if (IsVulkanMattingConsumer(effect))
    return ProductCapabilityDisposition::experimental;
  if (effect == hal::SemanticEffectId::auto_frame)
    return ProductCapabilityDisposition::usable_with_degraded_behavior;
  if (effect == hal::SemanticEffectId::eye_contact ||
      effect == hal::SemanticEffectId::video_noise_removal) {
    return ProductCapabilityDisposition::diagnostics_only;
  }
  return ProductCapabilityDisposition::unavailable;
}

[[nodiscard]] bool
HasSatisfiedEvidence(const hal::SemanticCapabilityEvidence &capability,
                     hal::CapabilityEvidenceCode code) noexcept {
  if (capability.evidence_count > capability.evidence.size())
    return false;
  for (std::size_t i = 0; i < capability.evidence_count; ++i) {
    const auto &evidence = capability.evidence[i];
    if (evidence.code == code && evidence.satisfied &&
        evidence.status == hal::StatusId::ready &&
        evidence.reason == hal::ReasonId::none &&
        (evidence.semantic == hal::SemanticEffectId::invalid ||
         evidence.semantic == capability.semantic) &&
        evidence.backend == hal::BackendId::vulkan) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool HasExactProductionVulkanEvidence(
    const hal::SemanticCapabilityEvidence &capability) noexcept {
  constexpr std::array common{
      hal::CapabilityEvidenceCode::vulkan_build_enabled,
      hal::CapabilityEvidenceCode::vulkan_loader_available,
      hal::CapabilityEvidenceCode::vulkan_physical_device_available,
      hal::CapabilityEvidenceCode::vulkan_non_cpu_device,
      hal::CapabilityEvidenceCode::vulkan_compute_queue_available,
      hal::CapabilityEvidenceCode::vulkan_context_healthy,
      hal::CapabilityEvidenceCode::vulkan_utility_kernels_ready,
      hal::CapabilityEvidenceCode::vulkan_exact_device_shared,
      hal::CapabilityEvidenceCode::vulkan_device_local_residency,
      hal::CapabilityEvidenceCode::vulkan_bounded_allocations,
      hal::CapabilityEvidenceCode::vulkan_no_cpu_layers,
      hal::CapabilityEvidenceCode::vulkan_no_intermediate_readbacks,
  };
  for (const auto code : common) {
    if (!HasSatisfiedEvidence(capability, code))
      return false;
  }
  const auto effect_code =
      capability.semantic == hal::SemanticEffectId::mirror
          ? hal::CapabilityEvidenceCode::open_vulkan_mirror_production_ready
          : hal::CapabilityEvidenceCode::
                open_vulkan_vignette_fixed_center_production_ready;
  return HasSatisfiedEvidence(capability, effect_code);
}

[[nodiscard]] constexpr bool ProgramGenerationMatches(
    const hal::ProgramInvalidationKey &key,
    const ProductCapabilityGenerations &generation) noexcept {
  return key.generations.configuration == generation.configuration &&
         key.generations.provider == generation.provider &&
         key.generations.device == generation.device &&
         key.generations.context == generation.context;
}

[[nodiscard]] const hal::ResolvedStage *
FindResolvedStage(const hal::ResolutionResult &resolution,
                  std::uint32_t stage_id,
                  hal::SemanticEffectId effect) noexcept {
  if (resolution.stage_count > resolution.stages.size())
    return nullptr;
  for (std::size_t i = 0; i < resolution.stage_count; ++i) {
    const auto &stage = resolution.stages[i];
    if (stage.intent.stage_id == stage_id && stage.intent.semantic == effect) {
      return &stage;
    }
  }
  return nullptr;
}

[[nodiscard]] constexpr LegacyComputeBackendId
LegacyBackend(hal::BackendId backend) noexcept {
  switch (backend) {
  case hal::BackendId::cpu:
    return LegacyComputeBackendId::cpu;
  case hal::BackendId::cuda:
  case hal::BackendId::open_video_onnxruntime:
    return LegacyComputeBackendId::cuda;
  case hal::BackendId::maxine:
    return LegacyComputeBackendId::maxine;
  case hal::BackendId::vulkan:
    return LegacyComputeBackendId::vulkan;
  case hal::BackendId::invalid:
  case hal::BackendId::test_fake:
    break;
  }
  return LegacyComputeBackendId::none;
}

[[nodiscard]] constexpr std::string_view
LegacyEffectBackendToken(hal::BackendId backend) noexcept {
  switch (backend) {
  case hal::BackendId::cpu:
    return "cpu";
  case hal::BackendId::cuda:
  case hal::BackendId::open_video_onnxruntime:
    return "open_cuda";
  case hal::BackendId::maxine:
    return "maxine";
  case hal::BackendId::vulkan:
    return "open_vulkan";
  case hal::BackendId::invalid:
  case hal::BackendId::test_fake:
    break;
  }
  return {};
}

} // namespace

bool IsCapabilitySnapshotCurrent(
    const ProductCapabilitySnapshot &snapshot) noexcept {
  return snapshot.refresh.status == CapabilityRefreshStatus::succeeded &&
         snapshot.refresh.code ==
             effects::EffectCapabilityCode::capability_snapshot_current &&
         snapshot.generations.capability != 0 &&
         snapshot.refresh.requested_generation ==
             snapshot.generations.capability &&
         snapshot.refresh.completed_generation ==
             snapshot.generations.capability;
}

CapabilityValidationResult ValidateProductCapabilitySnapshot(
    const ProductCapabilitySnapshot &snapshot) noexcept {
  CapabilityValidationResult result;
  if (snapshot.schema_version != kVideoRuntimeStatusSchemaVersion) {
    result.code = CapabilityValidationCode::wrong_schema_version;
    return result;
  }
  if (snapshot.hal_contract_version != hal::kContractVersion) {
    result.code = CapabilityValidationCode::wrong_hal_contract_version;
    return result;
  }
  if (snapshot.reason_code_version != effects::kEffectCapabilityCodeVersion) {
    result.code = CapabilityValidationCode::wrong_reason_code_version;
    return result;
  }
  if (snapshot.generations.capability == 0 ||
      snapshot.generations.configuration == 0 ||
      snapshot.generations.provider == 0 || snapshot.generations.device == 0 ||
      snapshot.generations.context == 0) {
    result.code = CapabilityValidationCode::invalid_generation;
    return result;
  }

  bool refresh_consistent = false;
  switch (snapshot.refresh.status) {
  case CapabilityRefreshStatus::never:
    refresh_consistent =
        snapshot.refresh.code ==
            effects::EffectCapabilityCode::capability_refresh_not_run &&
        snapshot.refresh.requested_generation == 0 &&
        snapshot.refresh.completed_generation == 0;
    break;
  case CapabilityRefreshStatus::in_progress:
    refresh_consistent =
        snapshot.refresh.code ==
            effects::EffectCapabilityCode::capability_refresh_in_progress &&
        snapshot.refresh.requested_generation ==
            snapshot.generations.capability &&
        snapshot.refresh.completed_generation <
            snapshot.refresh.requested_generation;
    break;
  case CapabilityRefreshStatus::succeeded:
    refresh_consistent = IsCapabilitySnapshotCurrent(snapshot);
    break;
  case CapabilityRefreshStatus::failed:
    refresh_consistent =
        snapshot.refresh.requested_generation ==
            snapshot.generations.capability &&
        snapshot.refresh.completed_generation <=
            snapshot.refresh.requested_generation &&
        snapshot.refresh.code != effects::EffectCapabilityCode::none &&
        !effects::IsEffectCapabilitySuccessCode(snapshot.refresh.code);
    break;
  case CapabilityRefreshStatus::stale:
    refresh_consistent =
        snapshot.refresh.code ==
            effects::EffectCapabilityCode::capability_snapshot_stale &&
        snapshot.refresh.requested_generation <=
            snapshot.generations.capability &&
        snapshot.refresh.completed_generation < snapshot.generations.capability;
    break;
  }
  if (!refresh_consistent) {
    result.code = CapabilityValidationCode::refresh_inconsistent;
    return result;
  }

  if (snapshot.lane_count > snapshot.lanes.size()) {
    result.code = CapabilityValidationCode::wrong_record_count;
    return result;
  }

  std::array<bool, kProductCapabilityRecordCount> seen{};
  for (std::size_t i = 0; i < snapshot.lane_count; ++i) {
    const ProductCapabilityLane &lane = snapshot.lanes[i];
    result.record_index = i;
    result.effect = lane.effect;
    result.lane = lane.lane;
    const std::size_t effect_index = EffectIndex(lane.effect);
    if (effect_index == kCanonicalVideoEffectCount) {
      result.code = CapabilityValidationCode::unknown_effect;
      return result;
    }
    const std::size_t lane_index = LaneIndex(lane.lane);
    if (lane_index == kProductCapabilityLaneCount) {
      result.code = CapabilityValidationCode::unknown_lane;
      return result;
    }
    const std::size_t key =
        effect_index * kProductCapabilityLaneCount + lane_index;
    if (seen[key]) {
      result.code = CapabilityValidationCode::duplicate_lane;
      return result;
    }
    seen[key] = true;

    if (lane.semantic.semantic != lane.effect) {
      result.code = CapabilityValidationCode::semantic_mismatch;
      return result;
    }
    if (!ProviderMatchesLane(lane.lane, lane.semantic.provider)) {
      result.code = CapabilityValidationCode::provider_lane_mismatch;
      return result;
    }

    const bool hal_selectable =
        hal::IsSelectableSemanticCapability(lane.semantic);
    const bool success_code = effects::IsEffectCapabilitySuccessCode(lane.code);

    if (lane.semantic.evidence_count > lane.semantic.evidence.size() ||
        lane.disposition == ProductCapabilityDisposition::invalid ||
        lane.code == effects::EffectCapabilityCode::none) {
      result.code = CapabilityValidationCode::readiness_inconsistent;
      return result;
    }
    if (lane.disposition == ProductCapabilityDisposition::production_usable &&
        (!hal_selectable || lane.semantic.status != hal::StatusId::ready ||
         !success_code)) {
      result.code = CapabilityValidationCode::readiness_inconsistent;
      return result;
    }
    if (lane.disposition ==
            ProductCapabilityDisposition::usable_with_degraded_behavior &&
        hal_selectable &&
        (lane.semantic.status != hal::StatusId::degraded || !success_code)) {
      result.code = CapabilityValidationCode::readiness_inconsistent;
      return result;
    }
    if ((lane.disposition == ProductCapabilityDisposition::experimental ||
         lane.disposition == ProductCapabilityDisposition::diagnostics_only ||
         lane.disposition == ProductCapabilityDisposition::unavailable) &&
        hal_selectable) {
      result.code = CapabilityValidationCode::readiness_inconsistent;
      return result;
    }
    if (hal_selectable !=
        (lane.disposition == ProductCapabilityDisposition::production_usable ||
         (lane.disposition ==
              ProductCapabilityDisposition::usable_with_degraded_behavior &&
          success_code))) {
      result.code = CapabilityValidationCode::readiness_inconsistent;
      return result;
    }
    if (lane.lane == ProductCapabilityLaneId::cpu) {
      if (hal_selectable ||
          lane.disposition != ProductCapabilityDisposition::unavailable ||
          lane.code != CpuUnavailableCode(lane.effect)) {
        result.code = CapabilityValidationCode::cpu_effect_provider_forbidden;
        return result;
      }
    }

    if (lane.lane == ProductCapabilityLaneId::open_vulkan) {
      const bool production_effect =
          lane.effect == hal::SemanticEffectId::mirror ||
          lane.effect == hal::SemanticEffectId::vignette;
      if (hal_selectable && !production_effect) {
        result.code = CapabilityValidationCode::vulkan_capability_promoted;
        return result;
      }
      if (hal_selectable) {
        const auto expected =
            lane.effect == hal::SemanticEffectId::mirror
                ? effects::EffectCapabilityCode::
                      open_vulkan_mirror_production_ready
                : effects::EffectCapabilityCode::
                      open_vulkan_vignette_fixed_center_production_ready;
        if (lane.code != expected ||
            lane.semantic.provider !=
                hal::SemanticEffectProviderId::open_vulkan_production_kernels ||
            !HasExactProductionVulkanEvidence(lane.semantic) ||
            (lane.effect == hal::SemanticEffectId::vignette &&
             lane.semantic.supports_tracked_vignette_center)) {
          result.code = CapabilityValidationCode::vulkan_matrix_inconsistent;
          return result;
        }
      } else if (!production_effect &&
                 lane.disposition !=
                     ExpectedNonProductionVulkanDisposition(lane.effect)) {
        result.code = CapabilityValidationCode::vulkan_matrix_inconsistent;
        return result;
      }
    }
  }

  for (std::size_t i = 0; i < seen.size(); ++i) {
    if (!seen[i]) {
      result.code = CapabilityValidationCode::missing_lane;
      result.effect = kCanonicalVideoEffects[i / kProductCapabilityLaneCount];
      result.lane = kProductCapabilityLanes[i % kProductCapabilityLaneCount];
      result.record_index = i;
      return result;
    }
  }
  result = {};
  return result;
}

ProductCapabilitySnapshot MakeFailClosedProductCapabilitySnapshot(
    const ProductCapabilityGenerations &generations,
    const CapabilityRefreshSnapshot &refresh) noexcept {
  ProductCapabilitySnapshot snapshot;
  snapshot.generations = generations;
  snapshot.refresh = refresh;

  using Code = effects::EffectCapabilityCode;
  for (const hal::SemanticEffectId effect : kCanonicalVideoEffects) {
    for (const ProductCapabilityLaneId lane_id : kProductCapabilityLanes) {
      ProductCapabilityLane lane;
      lane.effect = effect;
      lane.lane = lane_id;
      lane.disposition = ProductCapabilityDisposition::unavailable;
      lane.semantic.semantic = effect;
      lane.semantic.status = hal::StatusId::unavailable;
      lane.semantic.reason = hal::ReasonId::semantic_effect_unsupported;

      switch (lane_id) {
      case ProductCapabilityLaneId::maxine:
        lane.code = Code::maxine_effect_unavailable;
        break;
      case ProductCapabilityLaneId::open_cuda:
        lane.code = Code::open_cuda_effect_unavailable;
        break;
      case ProductCapabilityLaneId::cpu:
        lane.code = CpuUnavailableCode(effect);
        break;
      case ProductCapabilityLaneId::open_vulkan:
        lane.disposition = ExpectedNonProductionVulkanDisposition(effect);
        switch (effect) {
        case hal::SemanticEffectId::mirror:
          lane.code = Code::open_vulkan_runtime_diagnostics_unavailable;
          lane.nested_code = Code::open_vulkan_mirror_unavailable;
          break;
        case hal::SemanticEffectId::vignette:
          lane.code = Code::open_vulkan_runtime_diagnostics_unavailable;
          lane.nested_code = Code::open_vulkan_vignette_unavailable;
          break;
        case hal::SemanticEffectId::auto_frame:
          lane.code = Code::vulkan_auto_frame_yunet_unavailable;
          lane.nested_code = Code::vulkan_effect_cpu_tail;
          break;
        case hal::SemanticEffectId::eye_contact:
          lane.code = Code::open_vulkan_eye_contact_unavailable;
          lane.nested_code = Code::open_vulkan_eye_contact_runtime_unavailable;
          break;
        case hal::SemanticEffectId::video_noise_removal:
          lane.code = Code::open_vulkan_video_noise_removal_unavailable;
          lane.nested_code =
              Code::open_vulkan_video_noise_removal_runtime_unavailable;
          break;
        case hal::SemanticEffectId::virtual_background_blur:
        case hal::SemanticEffectId::virtual_background_remove:
        case hal::SemanticEffectId::virtual_background_replace:
        case hal::SemanticEffectId::virtual_key_light:
          lane.code = Code::open_vulkan_matting_unavailable;
          lane.nested_code = Code::open_vulkan_matting_adapter_unavailable;
          break;
        case hal::SemanticEffectId::invalid:
          break;
        }
        break;
      case ProductCapabilityLaneId::invalid:
        break;
      }
      snapshot.lanes[snapshot.lane_count++] = lane;
    }
  }
  return snapshot;
}

const ProductCapabilityLane *
FindProductCapabilityLane(const ProductCapabilitySnapshot &snapshot,
                          hal::SemanticEffectId effect,
                          ProductCapabilityLaneId lane) noexcept {
  if (snapshot.lane_count > snapshot.lanes.size())
    return nullptr;
  for (std::size_t i = 0; i < snapshot.lane_count; ++i) {
    if (snapshot.lanes[i].effect == effect && snapshot.lanes[i].lane == lane) {
      return &snapshot.lanes[i];
    }
  }
  return nullptr;
}

ProductCapabilityLane *
FindProductCapabilityLane(ProductCapabilitySnapshot *snapshot,
                          hal::SemanticEffectId effect,
                          ProductCapabilityLaneId lane) noexcept {
  if (!snapshot || snapshot->lane_count > snapshot->lanes.size())
    return nullptr;
  for (std::size_t i = 0; i < snapshot->lane_count; ++i) {
    if (snapshot->lanes[i].effect == effect &&
        snapshot->lanes[i].lane == lane) {
      return &snapshot->lanes[i];
    }
  }
  return nullptr;
}

bool IsProductCapabilityLaneEligible(
    const ProductCapabilitySnapshot &snapshot,
    const ProductCapabilityLane &lane) noexcept {
  return IsCapabilitySnapshotCurrent(snapshot) &&
         hal::IsSelectableSemanticCapability(lane.semantic);
}

RuntimeStatusValidationResult ValidateVideoRuntimeStatusSnapshot(
    const VideoRuntimeStatusSnapshot &snapshot) noexcept {
  RuntimeStatusValidationResult result;
  if (snapshot.schema_version != kVideoRuntimeStatusSchemaVersion) {
    result.code = RuntimeStatusValidationCode::wrong_schema_version;
    return result;
  }
  if (snapshot.hal_contract_version != hal::kContractVersion) {
    result.code = RuntimeStatusValidationCode::wrong_hal_contract_version;
    return result;
  }
  if (snapshot.status_generation == 0 || snapshot.generations.capability == 0) {
    result.code = RuntimeStatusValidationCode::invalid_generation;
    return result;
  }
  if (!(snapshot.capabilities.generations == snapshot.generations)) {
    result.code = RuntimeStatusValidationCode::capability_generation_mismatch;
    return result;
  }
  result.capability = ValidateProductCapabilitySnapshot(snapshot.capabilities);
  if (result.capability.code != CapabilityValidationCode::valid) {
    result.code = RuntimeStatusValidationCode::capability_snapshot_invalid;
    return result;
  }
  if (snapshot.resolution.stage_count > snapshot.resolution.stages.size()) {
    result.code = RuntimeStatusValidationCode::resolution_malformed;
    return result;
  }
  std::array<std::uint32_t, hal::kMaxPreparedStages> stage_ids{};
  std::size_t stage_id_count = 0;
  for (std::size_t i = 0; i < snapshot.resolution.stage_count; ++i) {
    const auto &stage = snapshot.resolution.stages[i];
    if (stage.intent.stage_id == 0 ||
        EffectIndex(stage.intent.semantic) == kCanonicalVideoEffectCount) {
      result.code = RuntimeStatusValidationCode::resolution_malformed;
      return result;
    }
    for (std::size_t j = 0; j < stage_id_count; ++j) {
      if (stage_ids[j] == stage.intent.stage_id) {
        result.code = RuntimeStatusValidationCode::resolution_malformed;
        return result;
      }
    }
    stage_ids[stage_id_count++] = stage.intent.stage_id;
  }

  if (!snapshot.readiness.configured) {
    if (snapshot.readiness.program_id != 0 ||
        snapshot.readiness.value.available) {
      result.code = RuntimeStatusValidationCode::readiness_inconsistent;
      return result;
    }
  } else {
    if (snapshot.readiness.program_id == 0 ||
        !snapshot.readiness.value.available ||
        (snapshot.readiness.value.status != hal::StatusId::ready &&
         snapshot.readiness.value.status != hal::StatusId::degraded) ||
        !ProgramGenerationMatches(snapshot.readiness.invalidation_key,
                                  snapshot.generations)) {
      result.code = RuntimeStatusValidationCode::readiness_inconsistent;
      return result;
    }
    if (snapshot.resolution.selected_backend != hal::BackendId::invalid &&
        snapshot.readiness.value.selected_backend !=
            snapshot.resolution.selected_backend) {
      result.code = RuntimeStatusValidationCode::readiness_inconsistent;
      return result;
    }
  }
  if (snapshot.lifecycle == VideoRuntimeLifecycle::running &&
      snapshot.resolution.available && !snapshot.resolution.passthrough &&
      !snapshot.readiness.configured) {
    result.code = RuntimeStatusValidationCode::readiness_inconsistent;
    return result;
  }
  result = {};
  return result;
}

ActiveRuntimeAttribution DeriveActiveRuntimeAttribution(
    const VideoRuntimeStatusSnapshot &snapshot) noexcept {
  ActiveRuntimeAttribution active;
  if (ValidateVideoRuntimeStatusSnapshot(snapshot).code !=
          RuntimeStatusValidationCode::valid ||
      snapshot.lifecycle != VideoRuntimeLifecycle::running ||
      !IsCapabilitySnapshotCurrent(snapshot.capabilities) ||
      !snapshot.readiness.configured || !snapshot.readiness.value.available ||
      (snapshot.readiness.value.status != hal::StatusId::ready &&
       snapshot.readiness.value.status != hal::StatusId::degraded)) {
    return active;
  }

  const hal::CommittedFrameSnapshot &committed =
      snapshot.latest_committed_frame;
  if (!committed.valid || committed.program_id == 0 ||
      committed.program_id != snapshot.readiness.program_id ||
      !(committed.invalidation_key == snapshot.readiness.invalidation_key) ||
      !ProgramGenerationMatches(committed.invalidation_key,
                                snapshot.generations) ||
      committed.execution_domain != snapshot.readiness.value.execution_domain ||
      committed.capture_sequence == 0 ||
      committed.output_commit_sequence == 0 ||
      committed.stage_count > committed.stages.size()) {
    return active;
  }

  hal::BackendId unique_backend = hal::BackendId::invalid;
  std::array<std::uint32_t, hal::kMaxPreparedStages> attributed_stage_ids{};
  std::size_t attributed_stage_id_count = 0;
  for (std::size_t i = 0; i < committed.stage_count; ++i) {
    const auto &stage = committed.stages[i];
    if (stage.status != hal::StageExecutionStatus::succeeded)
      continue;
    if (stage.semantic == hal::SemanticEffectId::invalid) {
      if (stage.primitive != hal::PrimitiveOperationId::invalid)
        continue;
      return {};
    }
    for (std::size_t j = 0; j < attributed_stage_id_count; ++j) {
      if (attributed_stage_ids[j] == stage.stage_id)
        return {};
    }
    attributed_stage_ids[attributed_stage_id_count++] = stage.stage_id;
    const hal::ResolvedStage *resolved =
        FindResolvedStage(snapshot.resolution, stage.stage_id, stage.semantic);
    if (!resolved ||
        (resolved->status != hal::StatusId::ready &&
         resolved->status != hal::StatusId::degraded) ||
        resolved->selected_backend != stage.backend ||
        resolved->provider != stage.provider) {
      return {};
    }
    const ProductCapabilityLaneId lane_id = LaneForBackend(stage.backend);
    const ProductCapabilityLane *capability = FindProductCapabilityLane(
        snapshot.capabilities, stage.semantic, lane_id);
    if (!capability || capability->semantic.provider != stage.provider ||
        !IsProductCapabilityLaneEligible(snapshot.capabilities, *capability)) {
      return {};
    }

    active.effects[active.effect_count++] = {stage.stage_id, stage.semantic,
                                             stage.backend, stage.provider};
    if (unique_backend == hal::BackendId::invalid) {
      unique_backend = stage.backend;
    } else if (unique_backend != stage.backend) {
      active.mixed_backends = true;
    }
  }
  active.active_backend =
      active.mixed_backends ? hal::BackendId::invalid : unique_backend;
  return active;
}

std::string_view
LegacyComputeBackendToken(LegacyComputeBackendId backend) noexcept {
  switch (backend) {
  case LegacyComputeBackendId::none:
    return {};
  case LegacyComputeBackendId::cpu:
    return "cpu";
  case LegacyComputeBackendId::cuda:
    return "cuda";
  case LegacyComputeBackendId::maxine:
    return "maxine";
  case LegacyComputeBackendId::vulkan:
    return "vulkan";
  case LegacyComputeBackendId::mixed:
    return "mixed";
  }
  return {};
}

LegacyVideoRuntimeCompatibility BuildLegacyVideoRuntimeCompatibility(
    const VideoRuntimeStatusSnapshot &snapshot) noexcept {
  LegacyVideoRuntimeCompatibility compatibility;
  if (ValidateVideoRuntimeStatusSnapshot(snapshot).code !=
      RuntimeStatusValidationCode::valid) {
    return compatibility;
  }

  LegacyComputeBackendId selected =
      LegacyBackend(snapshot.resolution.selected_backend);
  if (selected == LegacyComputeBackendId::none &&
      snapshot.resolution.stage_count <= snapshot.resolution.stages.size()) {
    for (std::size_t i = 0; i < snapshot.resolution.stage_count; ++i) {
      const auto &stage = snapshot.resolution.stages[i];
      if (stage.status != hal::StatusId::ready &&
          stage.status != hal::StatusId::degraded)
        continue;
      const LegacyComputeBackendId candidate =
          LegacyBackend(stage.selected_backend);
      if (candidate == LegacyComputeBackendId::none)
        continue;
      if (selected == LegacyComputeBackendId::none)
        selected = candidate;
      else if (selected != candidate)
        selected = LegacyComputeBackendId::mixed;
    }
  }
  for (std::size_t i = 0; i < snapshot.resolution.stage_count; ++i) {
    const auto &stage = snapshot.resolution.stages[i];
    if ((stage.status != hal::StatusId::ready &&
         stage.status != hal::StatusId::degraded) ||
        stage.selected_backend == hal::BackendId::invalid)
      continue;
    compatibility.selected_effects[compatibility.selected_effect_count++] = {
        hal::StableEffectId(stage.intent.semantic),
        LegacyEffectBackendToken(stage.selected_backend),
        stage.intent.semantic,
        stage.provider,
    };
  }
  compatibility.selected_backend = selected;
  compatibility.selected_backend_token = LegacyComputeBackendToken(selected);

  const ActiveRuntimeAttribution active =
      DeriveActiveRuntimeAttribution(snapshot);
  compatibility.active_backend = active.mixed_backends
                                     ? LegacyComputeBackendId::mixed
                                     : LegacyBackend(active.active_backend);
  compatibility.active_backend_token =
      LegacyComputeBackendToken(compatibility.active_backend);
  for (std::size_t i = 0; i < active.effect_count; ++i) {
    const auto &entry = active.effects[i];
    compatibility.active_effects[compatibility.active_effect_count++] = {
        hal::StableEffectId(entry.effect),
        LegacyEffectBackendToken(entry.backend),
        entry.effect,
        entry.provider,
    };
  }
  return compatibility;
}

} // namespace studiocast::video
