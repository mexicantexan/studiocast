#pragma once

#include "core/hal/static_registry.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace studiocast::hal {

inline constexpr std::size_t kMaxResolutionDiscrepancies =
    13 + (kMaxPreparedStages * 9);

enum class BackendRequestMode : std::uint8_t {
  automatic = 1,
  explicit_backend,
};

struct BackendRequest {
  BackendRequestMode mode = BackendRequestMode::automatic;
  BackendId backend = BackendId::invalid;
  bool strict = false;
};

struct CanonicalStageIntent {
  std::uint32_t stage_id = 0;
  SemanticEffectId semantic = SemanticEffectId::invalid;
  bool required = false;
  FailureAction failure_action = FailureAction::fail_open_preserve_input;
  bool requires_tracked_vignette_center = false;
};

struct ResolutionRequest {
  BackendRequest backend{};
  std::array<CanonicalStageIntent, kMaxPreparedStages> stages{};
  std::size_t stage_count = 0;
};

struct ResolvedStage {
  CanonicalStageIntent intent{};
  StatusId status = StatusId::invalid;
  ReasonId reason = ReasonId::none;
  BackendId selected_backend = BackendId::invalid;
  SemanticEffectProviderId provider = SemanticEffectProviderId::invalid;
  std::uint64_t adapter_instance_id = 0;
  ExecutionDomainId execution_domain{};
  CpuTailId cpu_tail = CpuTailId::invalid;
  std::uint64_t cpu_tail_scratch_bytes = 0;
  bool skipped_optional = false;
  bool fallback_selected = false;
};

struct ResolutionResult {
  BackendRequestMode request_mode = BackendRequestMode::automatic;
  bool strict_backend_selection = false;
  StatusId status = StatusId::invalid;
  ReasonId reason = ReasonId::none;
  BackendId requested_backend = BackendId::invalid;
  BackendId selected_backend = BackendId::invalid;
  // Resolution never claims live activity. The execution session commits this
  // field only after effect success and a successful output write.
  BackendId active_backend = BackendId::invalid;
  SemanticEffectProviderId selected_provider =
      SemanticEffectProviderId::invalid;
  std::array<ResolvedStage, kMaxPreparedStages> stages{};
  std::size_t stage_count = 0;
  bool available = false;
  bool degraded = false;
  bool fallback = false;
  bool passthrough = false;
};

// Setup-time only: this consumes fixed cached records and performs no runtime
// capability discovery or effect execution.
[[nodiscard]] ResolutionResult
ResolveSemanticStages(const StaticAdapterRegistry &registry,
                      const ResolutionRequest &request) noexcept;

enum class ShadowField : std::uint16_t {
  request_mode = 1,
  requested_backend,
  strict_backend_selection,
  selected_backend,
  selected_provider,
  active_backend,
  overall_status,
  overall_reason,
  available,
  degraded,
  fallback,
  passthrough,
  stage_count,
  stage_id,
  stage_semantic,
  stage_required,
  stage_backend,
  stage_provider,
  stage_status,
  stage_reason,
  stage_skipped_optional,
  stage_fallback,
};

struct ShadowDiscrepancy {
  ShadowField field = ShadowField::overall_status;
  std::uint32_t stage_index = 0;
  std::uint64_t live_value = 0;
  std::uint64_t shadow_value = 0;
};

struct ShadowComparison {
  std::array<ShadowDiscrepancy, kMaxResolutionDiscrepancies> discrepancies{};
  std::size_t discrepancy_count = 0;
  bool exact_match = true;
  bool truncated = false;
};

// This is deliberately a pure comparison. The first argument remains the live
// decision and is never mutated or replaced by the shadow result.
[[nodiscard]] ShadowComparison
CompareShadowResolution(const ResolutionResult &live,
                        const ResolutionResult &shadow) noexcept;

} // namespace studiocast::hal
