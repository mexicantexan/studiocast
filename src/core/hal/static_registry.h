#pragma once

#include "core/hal/prepared_program.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace studiocast::hal {

inline constexpr std::size_t kMaxRegisteredAdapters = 16;
inline constexpr std::size_t kMaxSemanticCapabilitiesPerAdapter = 16;
inline constexpr std::size_t kMaxPrimitiveCapabilitiesPerAdapter = 16;

enum class ExecutionResidencyClass : std::uint8_t {
  invalid = 0,
  host,
  device,
};

[[nodiscard]] ExecutionResidencyClass
ClassifyExecutionResidency(const ExecutionDomainId &domain) noexcept;

struct PrimitiveCapabilityEvidence {
  PrimitiveOperationId operation = PrimitiveOperationId::invalid;
  StatusId status = StatusId::invalid;
  ReasonId reason = ReasonId::none;
  bool available = false;
};

struct SemanticCapabilityEvidence {
  SemanticEffectId semantic = SemanticEffectId::invalid;
  SemanticEffectProviderId provider = SemanticEffectProviderId::invalid;
  StatusId status = StatusId::invalid;
  ReasonId reason = ReasonId::none;
  CpuTailId cpu_tail = CpuTailId::invalid;
  std::uint64_t cpu_tail_scratch_bytes = 0;
  DeclaredCpuTail::Placement cpu_tail_placement =
      DeclaredCpuTail::Placement::after_stage;
  std::array<CapabilityEvidenceRecord, kMaxCapabilityEvidenceRecords>
      evidence{};
  std::size_t evidence_count = 0;
  bool live_stage = false;
  bool production_eligible = false;
  bool exact_evidence = false;
  bool supports_tracked_vignette_center = false;
};

// Shared by setup-time resolution and typed status projection. Keeping this
// predicate in the contract prevents daemon/installer consumers from
// reinterpreting partial capability evidence as selectable production support.
[[nodiscard]] constexpr bool IsSelectableSemanticCapability(
    const SemanticCapabilityEvidence &capability) noexcept {
  return capability.live_stage && capability.production_eligible &&
         capability.exact_evidence &&
         (capability.status == StatusId::ready ||
          capability.status == StatusId::degraded);
}

struct AdapterRegistration {
  std::uint64_t adapter_instance_id = 0;
  BackendId backend = BackendId::invalid;
  ExecutionDomainId execution_domain{};
  // Lower values win. Equal priorities are resolved by stable backend,
  // provider, and adapter identifiers rather than registration order.
  std::uint32_t selection_priority = 0;
  IResidentAdapter *resident_adapter = nullptr;
  std::array<SemanticCapabilityEvidence, kMaxSemanticCapabilitiesPerAdapter>
      semantic_capabilities{};
  std::size_t semantic_capability_count = 0;
  std::array<PrimitiveCapabilityEvidence, kMaxPrimitiveCapabilitiesPerAdapter>
      primitive_capabilities{};
  std::size_t primitive_capability_count = 0;
};

class StaticAdapterRegistry {
public:
  // Registration consumes already-cached setup-time evidence. Implementations
  // must not probe devices, providers, models, files, or manifests here.
  [[nodiscard]] StatusId Register(const AdapterRegistration &registration,
                                  ReasonId *reason) noexcept;

  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] const AdapterRegistration *at(std::size_t index) const noexcept;
  [[nodiscard]] const AdapterRegistration *
  FindAdapter(std::uint64_t adapter_instance_id) const noexcept;
  [[nodiscard]] const PrimitiveCapabilityEvidence *
  FindPrimitive(std::uint64_t adapter_instance_id,
                PrimitiveOperationId operation) const noexcept;

private:
  std::array<AdapterRegistration, kMaxRegisteredAdapters> registrations_{};
  std::size_t count_ = 0;
};

[[nodiscard]] constexpr bool
IsVulkanProductionSemantic(SemanticEffectId semantic) noexcept {
  return semantic == SemanticEffectId::mirror ||
         semantic == SemanticEffectId::vignette;
}

} // namespace studiocast::hal
