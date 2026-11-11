#include "core/video/video_runtime_status.h"

#include <limits>

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
    return provider == Provider::open_video_onnxruntime_cuda ||
           provider == Provider::open_cuda_production_kernels;
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

[[nodiscard]] constexpr bool ProgramGenerationsAreValid(
    const hal::ProgramGenerations &generations) noexcept {
  return generations.configuration != 0 && generations.geometry != 0 &&
         generations.model != 0 && generations.provider != 0 &&
         generations.device != 0 && generations.context != 0;
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

[[nodiscard]] constexpr bool
IsKnownRoute(PreparedExecutionRouteId route) noexcept {
  return route == PreparedExecutionRouteId::final_visual ||
         route == PreparedExecutionRouteId::maxine_resident ||
         route == PreparedExecutionRouteId::open_cuda_virtual_background;
}

[[nodiscard]] constexpr bool
IsHealthyRuntimeStatus(hal::StatusId status) noexcept {
  return status == hal::StatusId::ready || status == hal::StatusId::degraded ||
         status == hal::StatusId::executed;
}

[[nodiscard]] bool CapabilityEvidenceEquals(
    const hal::SemanticCapabilityEvidence &left,
    const hal::SemanticCapabilityEvidence &right) noexcept {
  if (left.semantic != right.semantic || left.provider != right.provider ||
      left.status != right.status || left.reason != right.reason ||
      left.cpu_tail != right.cpu_tail ||
      left.cpu_tail_scratch_bytes != right.cpu_tail_scratch_bytes ||
      left.cpu_tail_placement != right.cpu_tail_placement ||
      left.evidence_count != right.evidence_count ||
      left.live_stage != right.live_stage ||
      left.production_eligible != right.production_eligible ||
      left.exact_evidence != right.exact_evidence ||
      left.supports_tracked_vignette_center !=
          right.supports_tracked_vignette_center ||
      left.evidence_count > left.evidence.size()) {
    return false;
  }
  for (std::size_t i = 0; i < left.evidence_count; ++i) {
    const auto &a = left.evidence[i];
    const auto &b = right.evidence[i];
    if (a.code != b.code || a.status != b.status || a.reason != b.reason ||
        a.semantic != b.semantic || a.backend != b.backend ||
        a.satisfied != b.satisfied) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] const PreparedExecutionIslandStatus *
FindIsland(const MultiIslandVideoRuntimeStatusSnapshot &snapshot,
           std::uint64_t island_id) noexcept {
  if (snapshot.island_count > snapshot.islands.size())
    return nullptr;
  for (std::size_t i = 0; i < snapshot.island_count; ++i) {
    if (snapshot.islands[i].island_id == island_id)
      return &snapshot.islands[i];
  }
  return nullptr;
}

[[nodiscard]] const hal::CommittedStageActivity *
FindCommittedStage(const PreparedExecutionIslandStatus &island,
                   hal::SemanticEffectId effect,
                   std::uint32_t stage_id) noexcept {
  const auto &committed = island.latest_committed_frame;
  if (!committed.valid || committed.stage_count > committed.stages.size())
    return nullptr;
  for (std::size_t i = 0; i < committed.stage_count; ++i) {
    const auto &stage = committed.stages[i];
    if (stage.semantic == effect &&
        (stage_id == 0 || stage.stage_id == stage_id)) {
      return &stage;
    }
  }
  return nullptr;
}

[[nodiscard]] bool AddCounter(std::uint64_t value,
                              std::uint64_t *total) noexcept {
  if (!total || value > std::numeric_limits<std::uint64_t>::max() - *total) {
    return false;
  }
  *total += value;
  return true;
}

[[nodiscard]] bool AddTransfers(const hal::TransferCounters &value,
                                hal::TransferCounters *total) noexcept {
  return total && AddCounter(value.uploads, &total->uploads) &&
         AddCounter(value.final_readbacks, &total->final_readbacks) &&
         AddCounter(value.intermediate_readbacks,
                    &total->intermediate_readbacks) &&
         AddCounter(value.intermediate_uploads, &total->intermediate_uploads) &&
         AddCounter(value.synchronizations, &total->synchronizations) &&
         AddCounter(value.cpu_tail_boundaries, &total->cpu_tail_boundaries);
}

[[nodiscard]] bool TransfersEqual(const hal::TransferCounters &left,
                                  const hal::TransferCounters &right) noexcept {
  return left.uploads == right.uploads &&
         left.final_readbacks == right.final_readbacks &&
         left.intermediate_readbacks == right.intermediate_readbacks &&
         left.intermediate_uploads == right.intermediate_uploads &&
         left.synchronizations == right.synchronizations &&
         left.cpu_tail_boundaries == right.cpu_tail_boundaries;
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

bool IsWriterOutputCommitMonotonic(
    const WriterOutputCommitIdentity &previous,
    const WriterOutputCommitIdentity &current) noexcept {
  const auto well_formed = [](const WriterOutputCommitIdentity &commit) {
    return commit.valid ? commit.capture_sequence != 0 &&
                              commit.output_commit_sequence != 0
                        : commit.capture_sequence == 0 &&
                              commit.output_commit_sequence == 0;
  };
  if (!well_formed(previous) || !well_formed(current))
    return false;
  if (!previous.valid)
    return true;
  if (!current.valid ||
      current.output_commit_sequence < previous.output_commit_sequence) {
    return false;
  }
  if (current.output_commit_sequence == previous.output_commit_sequence)
    return current.capture_sequence == previous.capture_sequence;
  return true;
}

MultiIslandStatusValidationResult ValidateMultiIslandVideoRuntimeStatusSnapshot(
    const MultiIslandVideoRuntimeStatusSnapshot &snapshot) noexcept {
  MultiIslandStatusValidationResult result;
  if (snapshot.schema_version != kMultiIslandVideoRuntimeStatusSchemaVersion) {
    result.code = MultiIslandStatusValidationCode::wrong_schema_version;
    return result;
  }
  if (snapshot.hal_contract_version != hal::kContractVersion) {
    result.code = MultiIslandStatusValidationCode::wrong_hal_contract_version;
    return result;
  }
  if (snapshot.reason_code_version != effects::kEffectCapabilityCodeVersion) {
    result.code = MultiIslandStatusValidationCode::wrong_reason_code_version;
    return result;
  }
  if (snapshot.status_generation == 0 || snapshot.generations.capability == 0 ||
      !ProgramGenerationsAreValid(snapshot.program_generations) ||
      snapshot.program_generations.configuration !=
          snapshot.generations.configuration ||
      snapshot.program_generations.provider != snapshot.generations.provider ||
      snapshot.program_generations.device != snapshot.generations.device ||
      snapshot.program_generations.context != snapshot.generations.context) {
    result.code = MultiIslandStatusValidationCode::invalid_generation;
    return result;
  }
  if (!(snapshot.capabilities.generations == snapshot.generations)) {
    result.code =
        MultiIslandStatusValidationCode::capability_generation_mismatch;
    return result;
  }
  if (ValidateProductCapabilitySnapshot(snapshot.capabilities).code !=
          CapabilityValidationCode::valid ||
      !IsCapabilitySnapshotCurrent(snapshot.capabilities)) {
    result.code = MultiIslandStatusValidationCode::capability_snapshot_invalid;
    return result;
  }
  if (snapshot.writer_commit.valid) {
    if (snapshot.writer_commit.capture_sequence == 0 ||
        snapshot.writer_commit.output_commit_sequence == 0) {
      result.code = MultiIslandStatusValidationCode::malformed_writer_commit;
      return result;
    }
  } else if (snapshot.writer_commit.capture_sequence != 0 ||
             snapshot.writer_commit.output_commit_sequence != 0) {
    result.code = MultiIslandStatusValidationCode::malformed_writer_commit;
    return result;
  }
  if (snapshot.island_count > snapshot.islands.size()) {
    result.code = MultiIslandStatusValidationCode::too_many_islands;
    return result;
  }

  hal::TransferCounters aggregate_transfers{};
  for (std::size_t i = 0; i < snapshot.island_count; ++i) {
    const PreparedExecutionIslandStatus &island = snapshot.islands[i];
    result.island_index = i;
    result.island_id = island.island_id;
    if (!island.valid || !IsKnownRoute(island.route) || island.island_id == 0 ||
        island.adapter_instance_id == 0 ||
        island.backend == hal::BackendId::invalid ||
        island.backend == hal::BackendId::test_fake ||
        island.snapshot_generation == 0 || island.capability_generation == 0 ||
        island.readiness.value.evidence_count >
            island.readiness.value.evidence.size()) {
      result.code = MultiIslandStatusValidationCode::invalid_island;
      return result;
    }
    for (std::size_t prior = 0; prior < i; ++prior) {
      if (snapshot.islands[prior].island_id == island.island_id) {
        result.code = MultiIslandStatusValidationCode::duplicate_island;
        return result;
      }
    }
    if (!island.current ||
        island.snapshot_generation != snapshot.status_generation ||
        island.capability_generation != snapshot.generations.capability) {
      result.code = MultiIslandStatusValidationCode::stale_island;
      return result;
    }
    if (!hal::ValidateExecutionDomain(island.execution_domain) ||
        island.execution_domain.backend != island.backend ||
        island.execution_domain.adapter_instance_id !=
            island.adapter_instance_id) {
      result.code = MultiIslandStatusValidationCode::foreign_adapter;
      return result;
    }
    if (!island.readiness.configured || island.readiness.program_id == 0 ||
        island.readiness.value.execution_domain != island.execution_domain ||
        island.readiness.value.selected_backend != island.backend ||
        !island.readiness.value.available ||
        (island.readiness.value.status != hal::StatusId::ready &&
         island.readiness.value.status != hal::StatusId::degraded)) {
      result.code = MultiIslandStatusValidationCode::wrong_execution_domain;
      return result;
    }
    if (!(island.readiness.invalidation_key.generations ==
          snapshot.program_generations)) {
      result.code = MultiIslandStatusValidationCode::wrong_generation;
      return result;
    }
    if (island.resolution.stage_count > island.resolution.stages.size() ||
        !island.resolution.available || island.resolution.passthrough ||
        (island.resolution.status != hal::StatusId::ready &&
         island.resolution.status != hal::StatusId::degraded) ||
        (island.resolution.selected_backend != hal::BackendId::invalid &&
         island.resolution.selected_backend != island.backend)) {
      result.code =
          MultiIslandStatusValidationCode::malformed_island_resolution;
      return result;
    }

    std::array<std::uint32_t, hal::kMaxPreparedStages> resolution_stage_ids{};
    std::size_t resolution_stage_count = 0;
    for (std::size_t stage_index = 0;
         stage_index < island.resolution.stage_count; ++stage_index) {
      const auto &stage = island.resolution.stages[stage_index];
      if (stage.intent.stage_id == 0 ||
          EffectIndex(stage.intent.semantic) == kCanonicalVideoEffectCount ||
          stage.selected_backend != island.backend ||
          stage.adapter_instance_id != island.adapter_instance_id ||
          stage.execution_domain != island.execution_domain ||
          stage.provider == hal::SemanticEffectProviderId::invalid ||
          (stage.status != hal::StatusId::ready &&
           stage.status != hal::StatusId::degraded)) {
        result.code =
            MultiIslandStatusValidationCode::malformed_island_resolution;
        return result;
      }
      for (std::size_t prior = 0; prior < resolution_stage_count; ++prior) {
        if (resolution_stage_ids[prior] == stage.intent.stage_id) {
          result.code =
              MultiIslandStatusValidationCode::malformed_island_resolution;
          return result;
        }
      }
      resolution_stage_ids[resolution_stage_count++] = stage.intent.stage_id;
    }

    const hal::CommittedFrameSnapshot &committed =
        island.latest_committed_frame;
    if (committed.valid) {
      if (committed.program_id != island.readiness.program_id ||
          !(committed.invalidation_key == island.readiness.invalidation_key)) {
        result.code = MultiIslandStatusValidationCode::wrong_program;
        return result;
      }
      if (committed.execution_domain != island.execution_domain) {
        result.code = MultiIslandStatusValidationCode::wrong_execution_domain;
        return result;
      }
      if (!snapshot.writer_commit.valid ||
          committed.capture_sequence !=
              snapshot.writer_commit.capture_sequence ||
          committed.output_commit_sequence !=
              snapshot.writer_commit.output_commit_sequence ||
          committed.stage_count > committed.stages.size()) {
        result.code = MultiIslandStatusValidationCode::mixed_writer_commit;
        return result;
      }
      std::array<std::uint32_t, hal::kMaxPreparedStages> committed_stage_ids{};
      std::size_t committed_stage_id_count = 0;
      for (std::size_t stage_index = 0; stage_index < committed.stage_count;
           ++stage_index) {
        const auto &stage = committed.stages[stage_index];
        const hal::ResolvedStage *resolved = FindResolvedStage(
            island.resolution, stage.stage_id, stage.semantic);
        if (stage.stage_id == 0 ||
            EffectIndex(stage.semantic) == kCanonicalVideoEffectCount ||
            stage.primitive != hal::PrimitiveOperationId::invalid ||
            stage.backend != island.backend ||
            stage.provider == hal::SemanticEffectProviderId::invalid ||
            !resolved || resolved->selected_backend != stage.backend ||
            resolved->provider != stage.provider) {
          result.code =
              MultiIslandStatusValidationCode::malformed_island_stages;
          return result;
        }
        for (std::size_t prior = 0; prior < committed_stage_id_count; ++prior) {
          if (committed_stage_ids[prior] == stage.stage_id) {
            result.code =
                MultiIslandStatusValidationCode::malformed_island_stages;
            return result;
          }
        }
        committed_stage_ids[committed_stage_id_count++] = stage.stage_id;
      }
    } else if (committed.program_id != 0 || committed.capture_sequence != 0 ||
               committed.output_commit_sequence != 0 ||
               committed.stage_count != 0) {
      result.code = MultiIslandStatusValidationCode::wrong_program;
      return result;
    }

    if (island.stage_count > island.stages.size()) {
      result.code = MultiIslandStatusValidationCode::malformed_island_stages;
      return result;
    }
    std::array<std::uint32_t, hal::kMaxPreparedStages> stage_record_ids{};
    std::size_t stage_record_count = 0;
    for (std::size_t stage_index = 0; stage_index < island.stage_count;
         ++stage_index) {
      const PreparedIslandStageRecord &record = island.stages[stage_index];
      const hal::ResolvedStage *resolved =
          FindResolvedStage(island.resolution, record.stage_id, record.effect);
      if (record.stage_id == 0 ||
          EffectIndex(record.effect) == kCanonicalVideoEffectCount ||
          !record.resolved || !resolved || record.backend != island.backend ||
          record.backend != resolved->selected_backend ||
          record.provider != resolved->provider ||
          record.status != resolved->status ||
          record.reason != resolved->reason ||
          record.required != resolved->intent.required ||
          record.failure_action != resolved->intent.failure_action) {
        result.code = MultiIslandStatusValidationCode::malformed_island_stages;
        return result;
      }
      for (std::size_t prior = 0; prior < stage_record_count; ++prior) {
        if (stage_record_ids[prior] == record.stage_id) {
          result.code =
              MultiIslandStatusValidationCode::malformed_island_stages;
          return result;
        }
      }
      stage_record_ids[stage_record_count++] = record.stage_id;
      const hal::CommittedStageActivity *committed_stage =
          FindCommittedStage(island, record.effect, record.stage_id);
      if (record.committed != (committed_stage != nullptr) ||
          (record.committed &&
           record.committed_status != committed_stage->status) ||
          (!record.committed &&
           record.committed_status != hal::StageExecutionStatus::not_run)) {
        result.code = MultiIslandStatusValidationCode::malformed_island_stages;
        return result;
      }
    }

    if (!AddTransfers(island.telemetry.transfers, &aggregate_transfers)) {
      result.code = MultiIslandStatusValidationCode::aggregate_overflow;
      return result;
    }
  }

  if (snapshot.effect_count > snapshot.effects.size()) {
    result.code = MultiIslandStatusValidationCode::wrong_effect_count;
    return result;
  }
  std::array<bool, kCanonicalVideoEffectCount> seen_effects{};
  for (std::size_t i = 0; i < snapshot.effect_count; ++i) {
    const StableEffectRuntimeState &effect = snapshot.effects[i];
    result.effect_index = i;
    result.effect = effect.effect;
    const std::size_t stable_index = EffectIndex(effect.effect);
    if (stable_index == kCanonicalVideoEffectCount) {
      result.code = MultiIslandStatusValidationCode::unknown_effect;
      return result;
    }
    if (seen_effects[stable_index]) {
      result.code = MultiIslandStatusValidationCode::duplicate_effect;
      return result;
    }
    seen_effects[stable_index] = true;

    if (!effect.requested) {
      if (effect.available || effect.selected || effect.active ||
          effect.degraded || effect.blocked || effect.fallback ||
          effect.strict_backend_selection || effect.strict_provider_selection ||
          effect.required ||
          effect.requested_backend != hal::BackendId::invalid ||
          effect.requested_provider != hal::SemanticEffectProviderId::invalid ||
          effect.selected_backend != hal::BackendId::invalid ||
          effect.selected_provider != hal::SemanticEffectProviderId::invalid ||
          effect.active_backend != hal::BackendId::invalid ||
          effect.active_provider != hal::SemanticEffectProviderId::invalid ||
          effect.selected_island_id != 0 || effect.committed_stage_id != 0 ||
          effect.committed_stage_status != hal::StageExecutionStatus::not_run ||
          effect.commit.valid || effect.commit.capture_sequence != 0 ||
          effect.commit.output_commit_sequence != 0 ||
          effect.capability_lane != ProductCapabilityLaneId::invalid ||
          effect.capability_code != effects::EffectCapabilityCode::none ||
          effect.nested_capability_code !=
              effects::EffectCapabilityCode::none ||
          effect.capability.semantic != hal::SemanticEffectId::invalid ||
          effect.capability.provider !=
              hal::SemanticEffectProviderId::invalid ||
          effect.capability.evidence_count != 0 ||
          effect.selected_capability_lane != ProductCapabilityLaneId::invalid ||
          effect.selected_capability_code !=
              effects::EffectCapabilityCode::none ||
          effect.selected_nested_capability_code !=
              effects::EffectCapabilityCode::none ||
          effect.selected_capability.semantic !=
              hal::SemanticEffectId::invalid ||
          effect.selected_capability.provider !=
              hal::SemanticEffectProviderId::invalid ||
          effect.selected_capability.evidence_count != 0 ||
          effect.status != hal::StatusId::passthrough ||
          effect.reason != hal::ReasonId::none) {
        result.code =
            MultiIslandStatusValidationCode::effect_state_inconsistent;
        return result;
      }
      continue;
    }

    if (effect.requested_backend == hal::BackendId::invalid ||
        effect.capability_lane == ProductCapabilityLaneId::invalid ||
        LaneIndex(effect.capability_lane) == kProductCapabilityLaneCount) {
      result.code = MultiIslandStatusValidationCode::invalid_capability_lane;
      return result;
    }
    const ProductCapabilityLane *lane = FindProductCapabilityLane(
        snapshot.capabilities, effect.effect, effect.capability_lane);
    if (!lane || lane->code != effect.capability_code ||
        lane->nested_code != effect.nested_capability_code ||
        !CapabilityEvidenceEquals(lane->semantic, effect.capability)) {
      result.code =
          MultiIslandStatusValidationCode::invalid_capability_evidence;
      return result;
    }
    if (LaneForBackend(effect.requested_backend) != effect.capability_lane) {
      result.code = MultiIslandStatusValidationCode::invalid_capability_lane;
      return result;
    }
    if (effect.strict_backend_selection && effect.selected &&
        effect.selected_backend != effect.requested_backend) {
      result.code =
          MultiIslandStatusValidationCode::strict_backend_substitution;
      return result;
    }
    if (effect.strict_provider_selection &&
        effect.requested_provider == hal::SemanticEffectProviderId::invalid) {
      result.code = MultiIslandStatusValidationCode::effect_state_inconsistent;
      return result;
    }
    if (effect.strict_provider_selection && effect.selected &&
        effect.selected_provider != effect.requested_provider) {
      result.code =
          MultiIslandStatusValidationCode::strict_provider_substitution;
      return result;
    }
    if (effect.selected) {
      const PreparedExecutionIslandStatus *island =
          FindIsland(snapshot, effect.selected_island_id);
      if (effect.selected_capability_lane == ProductCapabilityLaneId::invalid ||
          LaneIndex(effect.selected_capability_lane) ==
              kProductCapabilityLaneCount ||
          LaneForBackend(effect.selected_backend) !=
              effect.selected_capability_lane ||
          !island || effect.selected_backend != island->backend ||
          effect.selected_provider == hal::SemanticEffectProviderId::invalid) {
        result.code =
            MultiIslandStatusValidationCode::effect_state_inconsistent;
        return result;
      }
      const ProductCapabilityLane *selected_lane =
          FindProductCapabilityLane(snapshot.capabilities, effect.effect,
                                    effect.selected_capability_lane);
      if (!selected_lane ||
          selected_lane->code != effect.selected_capability_code ||
          selected_lane->nested_code !=
              effect.selected_nested_capability_code ||
          selected_lane->semantic.provider != effect.selected_provider ||
          !CapabilityEvidenceEquals(selected_lane->semantic,
                                    effect.selected_capability) ||
          !IsProductCapabilityLaneEligible(snapshot.capabilities,
                                           *selected_lane) ||
          !effect.available) {
        result.code =
            MultiIslandStatusValidationCode::invalid_capability_evidence;
        return result;
      }
      bool resolved = false;
      for (std::size_t stage_index = 0;
           stage_index < island->resolution.stage_count; ++stage_index) {
        const auto &stage = island->resolution.stages[stage_index];
        if (stage.intent.semantic == effect.effect &&
            stage.selected_backend == effect.selected_backend &&
            stage.provider == effect.selected_provider) {
          if (stage.intent.required != effect.required ||
              stage.intent.failure_action != effect.failure_action) {
            result.code =
                MultiIslandStatusValidationCode::effect_state_inconsistent;
            return result;
          }
          resolved = true;
          break;
        }
      }
      if (!resolved) {
        result.code =
            MultiIslandStatusValidationCode::effect_state_inconsistent;
        return result;
      }
    } else if (effect.selected_backend != hal::BackendId::invalid ||
               effect.selected_provider !=
                   hal::SemanticEffectProviderId::invalid ||
               effect.selected_island_id != 0 || effect.active ||
               effect.selected_capability_lane !=
                   ProductCapabilityLaneId::invalid ||
               effect.selected_capability_code !=
                   effects::EffectCapabilityCode::none ||
               effect.selected_nested_capability_code !=
                   effects::EffectCapabilityCode::none ||
               effect.selected_capability.semantic !=
                   hal::SemanticEffectId::invalid ||
               effect.selected_capability.provider !=
                   hal::SemanticEffectProviderId::invalid ||
               effect.selected_capability.evidence_count != 0) {
      result.code = MultiIslandStatusValidationCode::effect_state_inconsistent;
      return result;
    }
    const bool requested_lane_eligible =
        IsProductCapabilityLaneEligible(snapshot.capabilities, *lane) &&
        (effect.requested_provider == hal::SemanticEffectProviderId::invalid ||
         effect.requested_provider == lane->semantic.provider);
    if (!effect.selected && effect.available != requested_lane_eligible) {
      result.code =
          MultiIslandStatusValidationCode::invalid_capability_evidence;
      return result;
    }

    if (effect.blocked &&
        (effect.selected || effect.active ||
         (effect.status != hal::StatusId::blocked &&
          effect.status != hal::StatusId::unavailable &&
          effect.status != hal::StatusId::quarantined &&
          effect.status != hal::StatusId::resource_exhausted) ||
         effect.reason == hal::ReasonId::none)) {
      result.code = MultiIslandStatusValidationCode::effect_state_inconsistent;
      return result;
    }
    if (effect.degraded != (effect.status == hal::StatusId::degraded)) {
      result.code = MultiIslandStatusValidationCode::effect_state_inconsistent;
      return result;
    }

    const PreparedExecutionIslandStatus *commit_island = nullptr;
    const hal::CommittedStageActivity *committed_stage = nullptr;
    std::size_t committed_match_count = 0;
    for (std::size_t island_index = 0; island_index < snapshot.island_count;
         ++island_index) {
      const auto &island = snapshot.islands[island_index];
      const auto &committed_frame = island.latest_committed_frame;
      for (std::size_t stage_index = 0;
           stage_index < committed_frame.stage_count; ++stage_index) {
        const auto &candidate = committed_frame.stages[stage_index];
        if (candidate.semantic == effect.effect &&
            (effect.committed_stage_id == 0 ||
             candidate.stage_id == effect.committed_stage_id)) {
          ++committed_match_count;
          commit_island = &island;
          committed_stage = &candidate;
        }
      }
    }
    if (committed_match_count > 1) {
      result.code =
          MultiIslandStatusValidationCode::committed_activity_mismatch;
      return result;
    }
    const bool committed = committed_stage != nullptr;
    if (effect.commit.valid != committed ||
        (committed &&
         (effect.commit != snapshot.writer_commit ||
          effect.committed_stage_id != committed_stage->stage_id ||
          effect.committed_stage_status != committed_stage->status)) ||
        (!committed &&
         (effect.committed_stage_id != 0 ||
          effect.committed_stage_status != hal::StageExecutionStatus::not_run ||
          effect.commit.capture_sequence != 0 ||
          effect.commit.output_commit_sequence != 0))) {
      result.code =
          MultiIslandStatusValidationCode::committed_activity_mismatch;
      return result;
    }

    const bool committed_success =
        committed &&
        committed_stage->status == hal::StageExecutionStatus::succeeded &&
        snapshot.lifecycle == VideoRuntimeLifecycle::running &&
        snapshot.writer_commit.valid && commit_island &&
        commit_island->current &&
        IsHealthyRuntimeStatus(commit_island->telemetry.runtime_status) &&
        commit_island->readiness.configured &&
        commit_island->readiness.value.available &&
        committed_stage->backend == effect.selected_backend &&
        committed_stage->provider == effect.selected_provider &&
        effect.selected && effect.available;
    if (effect.active != committed_success) {
      result.code =
          MultiIslandStatusValidationCode::committed_activity_mismatch;
      return result;
    }
    if (effect.active) {
      if (effect.active_backend != committed_stage->backend ||
          effect.active_provider != committed_stage->provider ||
          effect.status != hal::StatusId::executed ||
          effect.reason != hal::ReasonId::none || effect.blocked) {
        result.code =
            MultiIslandStatusValidationCode::committed_activity_mismatch;
        return result;
      }
    } else if (effect.active_backend != hal::BackendId::invalid ||
               effect.active_provider !=
                   hal::SemanticEffectProviderId::invalid) {
      result.code =
          MultiIslandStatusValidationCode::committed_activity_mismatch;
      return result;
    }

    if (committed && committed_stage->status ==
                         hal::StageExecutionStatus::optional_failed_open) {
      if (effect.required ||
          effect.failure_action == hal::FailureAction::fail_closed ||
          effect.active || !effect.degraded || !effect.fallback ||
          effect.status != hal::StatusId::degraded ||
          effect.reason != hal::ReasonId::optional_stage_failed_open) {
        result.code =
            MultiIslandStatusValidationCode::optional_failure_inconsistent;
        return result;
      }
    }
    const bool substitution =
        effect.selected &&
        (effect.selected_backend != effect.requested_backend ||
         (effect.requested_provider != hal::SemanticEffectProviderId::invalid &&
          effect.selected_provider != effect.requested_provider));
    const bool optional_fail_open =
        committed && committed_stage->status ==
                         hal::StageExecutionStatus::optional_failed_open;
    if (effect.fallback != (substitution || optional_fail_open)) {
      result.code = MultiIslandStatusValidationCode::effect_state_inconsistent;
      return result;
    }
  }
  for (std::size_t i = 0; i < seen_effects.size(); ++i) {
    if (!seen_effects[i]) {
      result.code = MultiIslandStatusValidationCode::missing_effect;
      result.effect = kCanonicalVideoEffects[i];
      result.effect_index = i;
      return result;
    }
  }
  constexpr std::uint16_t kCanonicalEffectMask =
      (std::uint16_t{1} << kCanonicalVideoEffectCount) - 1u;
  if ((snapshot.authority.prepared_route_effect_mask & ~kCanonicalEffectMask) !=
          0 ||
      (snapshot.authority.compatibility_uncovered_effect_mask &
       ~kCanonicalEffectMask) != 0 ||
      (snapshot.authority.prepared_route_effect_mask &
       snapshot.authority.compatibility_uncovered_effect_mask) != 0) {
    result.code = MultiIslandStatusValidationCode::malformed_authority_coverage;
    return result;
  }
  std::uint16_t expected_prepared_mask = 0;
  std::uint16_t expected_uncovered_mask = 0;
  for (std::size_t i = 0; i < snapshot.effect_count; ++i) {
    const StableEffectRuntimeState &effect = snapshot.effects[i];
    const std::size_t stable_index = EffectIndex(effect.effect);
    if (!effect.requested)
      continue;
    const std::uint16_t bit = std::uint16_t{1}
                              << static_cast<unsigned int>(stable_index);
    if (effect.selected)
      expected_prepared_mask |= bit;
    else
      expected_uncovered_mask |= bit;
  }
  if (snapshot.authority.prepared_route_effect_mask != expected_prepared_mask ||
      snapshot.authority.compatibility_uncovered_effect_mask !=
          expected_uncovered_mask ||
      snapshot.authority.complete != (expected_uncovered_mask == 0)) {
    result.code = MultiIslandStatusValidationCode::malformed_authority_coverage;
    return result;
  }
  if (snapshot.aggregate.current_island_count != snapshot.island_count ||
      !TransfersEqual(snapshot.aggregate.transfers, aggregate_transfers)) {
    result.code = MultiIslandStatusValidationCode::aggregate_mismatch;
    return result;
  }
  return {};
}

MultiIslandStatusValidationResult BuildMultiIslandVideoRuntimeStatusSnapshot(
    const MultiIslandVideoRuntimeStatusBuildInput &input,
    MultiIslandVideoRuntimeStatusSnapshot *output) noexcept {
  if (!output) {
    MultiIslandStatusValidationResult result;
    result.code = MultiIslandStatusValidationCode::invalid_island;
    return result;
  }
  MultiIslandVideoRuntimeStatusSnapshot candidate;
  candidate.status_generation = input.status_generation;
  candidate.generations = input.generations;
  candidate.program_generations = input.program_generations;
  candidate.lifecycle = input.lifecycle;
  candidate.capabilities = input.capabilities;
  candidate.writer_commit = input.writer_commit;
  candidate.island_count = input.island_count;
  candidate.effect_count = input.effect_count;
  if (input.island_count <= input.islands.size()) {
    for (std::size_t i = 0; i < input.island_count; ++i)
      candidate.islands[i] = input.islands[i];
  }
  if (input.effect_count <= input.effects.size()) {
    for (std::size_t i = 0; i < input.effect_count; ++i)
      candidate.effects[i] = input.effects[i];
  }
  if (input.effect_count <= input.effects.size()) {
    for (std::size_t i = 0; i < input.effect_count; ++i) {
      const StableEffectRuntimeState &effect = input.effects[i];
      const std::size_t stable_index = EffectIndex(effect.effect);
      if (!effect.requested || stable_index == kCanonicalVideoEffectCount)
        continue;
      const std::uint16_t bit = std::uint16_t{1}
                                << static_cast<unsigned int>(stable_index);
      if (effect.selected)
        candidate.authority.prepared_route_effect_mask |= bit;
      else
        candidate.authority.compatibility_uncovered_effect_mask |= bit;
    }
  }
  candidate.authority.complete =
      candidate.authority.compatibility_uncovered_effect_mask == 0;
  candidate.aggregate.current_island_count = input.island_count;
  if (input.island_count <= input.islands.size()) {
    for (std::size_t i = 0; i < input.island_count; ++i) {
      if (!AddTransfers(input.islands[i].telemetry.transfers,
                        &candidate.aggregate.transfers)) {
        *output = {};
        MultiIslandStatusValidationResult result;
        result.code = MultiIslandStatusValidationCode::aggregate_overflow;
        result.island_index = i;
        result.island_id = input.islands[i].island_id;
        return result;
      }
    }
  }
  const MultiIslandStatusValidationResult validation =
      ValidateMultiIslandVideoRuntimeStatusSnapshot(candidate);
  if (validation.code != MultiIslandStatusValidationCode::valid) {
    *output = {};
    return validation;
  }
  *output = candidate;
  return {};
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
