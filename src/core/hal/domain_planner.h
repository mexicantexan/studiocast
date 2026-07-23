#pragma once

#include "core/hal/resolver.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace studiocast::hal {

inline constexpr std::size_t kMaxResidentIslands = kMaxPreparedStages;
inline constexpr std::size_t kMaxExecutionBoundaries =
    (kMaxPreparedStages * 2) + kMaxCpuTails;

enum class CrossAdapterPolicy : std::uint8_t {
  reject = 1,
  split_with_measured_transfer,
};

enum class ExecutionBoundaryKind : std::uint8_t {
  adapter_transfer = 1,
  cpu_tail,
};

struct ExecutionBoundary {
  ExecutionBoundaryKind kind = ExecutionBoundaryKind::adapter_transfer;
  std::uint32_t after_stage_id = 0;
  std::uint32_t before_stage_id = 0;
  CpuTailId cpu_tail = CpuTailId::invalid;
  std::uint64_t scratch_bytes = 0;
  ExecutionDomainId source_domain{};
  ExecutionDomainId destination_domain{};
  bool readback_required = false;
  bool following_upload_required = false;
};

struct ResidentIslandPlan {
  std::uint64_t adapter_instance_id = 0;
  IResidentAdapter *resident_adapter = nullptr;
  ExecutionDomainId execution_domain{};
  std::array<std::size_t, kMaxPreparedStages> resolved_stage_indices{};
  std::size_t stage_count = 0;
  bool requires_upload = true;
  bool requires_final_readback = true;
};

struct DomainPlan {
  StatusId status = StatusId::invalid;
  ReasonId reason = ReasonId::none;
  std::array<ResidentIslandPlan, kMaxResidentIslands> islands{};
  std::size_t island_count = 0;
  std::array<ExecutionBoundary, kMaxExecutionBoundaries> boundaries{};
  std::size_t boundary_count = 0;
  TransferCounters expected_transfers{};
  bool passthrough = false;
};

[[nodiscard]] DomainPlan
PlanExecutionDomains(const StaticAdapterRegistry &registry,
                     const ResolutionResult &resolution,
                     CrossAdapterPolicy policy) noexcept;

} // namespace studiocast::hal
