#pragma once

#include "core/video/video_adapter_factory.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace studiocast::video {

inline constexpr std::uint32_t kProductionVideoSetupLeaseContractVersion = 1;
inline constexpr std::size_t kMaxProductionVideoSetupProviders =
    hal::kMaxSemanticCapabilitiesPerAdapter;
inline constexpr std::size_t kMaxProductionVideoSetupArtifacts =
    hal::kMaxSemanticCapabilitiesPerAdapter;

enum class ProductionVideoSetupLeaseState : std::uint8_t {
  invalid = 0,
  live,
  stopped,
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
  virtual void Shutdown() noexcept = 0;
};

struct ProductionVideoSetupLeaseAcquireEvidence {
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::ReadinessSnapshot readiness{};
  bool candidate_owner_rolled_back = false;
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
  [[nodiscard]] const ProductionVideoSetupLeaseSnapshot &
  snapshot() const noexcept {
    return snapshot_;
  }

  // Idempotent. The owner object is intentionally retained so published typed
  // object addresses stay valid until lease destruction.
  void Shutdown() noexcept;

private:
  ProductionVideoSetupLease(
      ProductionVideoSetupLeaseSnapshot snapshot,
      std::unique_ptr<IProductionVideoSetupLeaseOwner> owner) noexcept;

  ProductionVideoSetupLeaseSnapshot snapshot_{};
  std::unique_ptr<IProductionVideoSetupLeaseOwner> owner_;
  bool shutdown_called_ = false;
};

} // namespace studiocast::video
