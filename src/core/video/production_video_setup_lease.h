#pragma once

#include "core/video/video_adapter_factory.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace studiocast::video {

inline constexpr std::uint32_t kProductionVideoSetupLeaseContractVersion = 2;
inline constexpr std::size_t kMaxProductionVideoSetupProviders =
    hal::kMaxSemanticCapabilitiesPerAdapter;
inline constexpr std::size_t kMaxProductionVideoSetupArtifacts =
    hal::kMaxSemanticCapabilitiesPerAdapter;

enum class ProductionVideoSetupLeaseState : std::uint8_t {
  invalid = 0,
  live,
  shutdown_pending,
  stopped,
};

// Typed, bounded result for one shutdown attempt. A pending result is not
// permission to release driver/context-owned resources. Callers may retry only
// by making another explicit Shutdown() call; no implementation may spin or
// wait unboundedly inside this contract.
struct ProductionVideoSetupLeaseShutdownResult {
  std::uint32_t contract_version = kProductionVideoSetupLeaseContractVersion;
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  std::uint64_t shutdown_attempts = 0;
  bool shutdown_requested = false;
  bool pending = false;
  bool quiescent = false;
  bool driver_context_release_permitted = false;
};

struct ProductionVideoSetupArtifactBinding {
  hal::SemanticEffectId semantic = hal::SemanticEffectId::invalid;
  hal::SemanticEffectProviderId provider =
      hal::SemanticEffectProviderId::invalid;
  hal::ModelArtifactIdentity artifact{};
};

// Immutable setup-time evidence for one exact resident execution island.
// A factory provider may be invalid only to consume the frozen per-stage
// providers listed in providers. This preserves Maxine mixed-provider islands
// without allowing the factory to select or substitute a provider.
struct ProductionVideoSetupLeaseSnapshot {
  std::uint32_t contract_version = kProductionVideoSetupLeaseContractVersion;
  ProductionVideoSetupLeaseState state =
      ProductionVideoSetupLeaseState::invalid;
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::ReadinessSnapshot readiness{};
  ProductionVideoSetupLeaseShutdownResult shutdown{};

  hal::BackendId backend = hal::BackendId::invalid;
  hal::ExecutionDomainId execution_domain{};
  hal::ProgramGenerations generations{};
  hal::ResourceBudget reserved_budget{};

  std::array<hal::SemanticEffectProviderId, kMaxProductionVideoSetupProviders>
      providers{};
  std::size_t provider_count = 0;
  std::array<ProductionVideoSetupArtifactBinding,
             kMaxProductionVideoSetupArtifacts>
      artifacts{};
  std::size_t artifact_count = 0;

  hal::AdapterRegistration adapter{};
  VideoAdapterFactoryRegistration factory{};
};

// Backend implementations own their adapter, factory, sessions, contexts,
// buffers, and scratch behind this typed interface. The returned pointers must
// remain at stable addresses until the owner is destroyed, including after
// Shutdown(). Shutdown releases active work/resources but must not invalidate
// the object addresses published in the frozen snapshot.
class IProductionVideoSetupLeaseOwner {
public:
  virtual ~IProductionVideoSetupLeaseOwner() = default;

  [[nodiscard]] virtual hal::IResidentAdapter *
  resident_adapter() const noexcept = 0;
  [[nodiscard]] virtual IVideoAdapterFactory *
  video_adapter_factory() const noexcept = 0;
  // Compatibility hook for owners whose teardown is immediately complete.
  // Retryable owners override RequestShutdown() and must retain unsafe native
  // resources in their destructor if final quiescence remains unproven.
  virtual void Shutdown() noexcept = 0;
  [[nodiscard]] virtual ProductionVideoSetupLeaseShutdownResult
  RequestShutdown() noexcept;
};

struct ProductionVideoSetupLeaseAcquireEvidence {
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::ReadinessSnapshot readiness{};
  ProductionVideoSetupLeaseShutdownResult candidate_shutdown{};
  bool candidate_owner_rolled_back = false;
  bool candidate_cleanup_pending = false;
  bool previous_lease_retained = false;
};

// Validates cached setup evidence only. It performs no probing, discovery,
// allocation, session construction, or frame-time work.
[[nodiscard]] hal::StatusId ValidateProductionVideoSetupLease(
    const ProductionVideoSetupLeaseSnapshot &snapshot,
    const IProductionVideoSetupLeaseOwner &owner,
    hal::ReasonId *reason) noexcept;

// Move-only owning lifetime for one validated production setup. Composition
// must destroy/shutdown its coordinator first and this lease second, because
// the registry/coordinator retain the snapshot's typed raw pointers.
class ProductionVideoSetupLease final {
public:
  ProductionVideoSetupLease() = default;
  ~ProductionVideoSetupLease();

  ProductionVideoSetupLease(const ProductionVideoSetupLease &) = delete;
  ProductionVideoSetupLease &
  operator=(const ProductionVideoSetupLease &) = delete;

  ProductionVideoSetupLease(ProductionVideoSetupLease &&other) noexcept;
  ProductionVideoSetupLease &
  operator=(ProductionVideoSetupLease &&other) noexcept;

  // Transactional replacement: a malformed candidate is shut down and
  // destroyed while an existing valid lease remains untouched.
  [[nodiscard]] static hal::StatusId
  Acquire(ProductionVideoSetupLeaseSnapshot snapshot,
          std::unique_ptr<IProductionVideoSetupLeaseOwner> candidate_owner,
          ProductionVideoSetupLease *destination,
          ProductionVideoSetupLeaseAcquireEvidence *evidence,
          hal::ReasonId *reason) noexcept;

  [[nodiscard]] bool live() const noexcept;
  [[nodiscard]] bool shutdown_pending() const noexcept;
  [[nodiscard]] const ProductionVideoSetupLeaseSnapshot &
  snapshot() const noexcept {
    return snapshot_;
  }

  // Idempotent after proven completion. While pending, each explicit call
  // performs exactly one additional bounded owner attempt. The owner object is
  // retained so published addresses and fail-safe backend ownership remain
  // valid until lease destruction.
  ProductionVideoSetupLeaseShutdownResult Shutdown() noexcept;
  [[nodiscard]] const ProductionVideoSetupLeaseShutdownResult &
  shutdown_result() const noexcept {
    return shutdown_result_;
  }

private:
  ProductionVideoSetupLease(
      ProductionVideoSetupLeaseSnapshot snapshot,
      std::unique_ptr<IProductionVideoSetupLeaseOwner> owner) noexcept;

  ProductionVideoSetupLeaseSnapshot snapshot_{};
  std::unique_ptr<IProductionVideoSetupLeaseOwner> owner_;
  ProductionVideoSetupLeaseShutdownResult shutdown_result_{};
};

} // namespace studiocast::video
