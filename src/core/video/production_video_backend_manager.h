#pragma once

#include "core/video/cpu_primitive_hal.h"
#include "core/video/production_video_execution_composition.h"

#include <cstdint>
#include <memory>

namespace studiocast::video {

inline constexpr std::uint32_t kProductionVideoBackendManagerContractVersion =
    1;

enum class ProductionVideoBackendSetupKind : std::uint8_t {
  invalid = 0,
  cpu_primitives_only,
  resident_island,
};

// A CPU setup publishes only startup-selected host primitive availability.
// It is never a semantic effect-provider registration, never creates a
// resident execution domain, and never authorizes an effect-level fallback.
struct ProductionVideoCpuPrimitiveSetup {
  std::uint32_t contract_version =
      kProductionVideoBackendManagerContractVersion;
  const CpuPrimitiveTable *table = nullptr;
  std::uint64_t required_primitive_bits = 0;
};

// The caller supplies canonical semantics, host descriptors, generations, and
// policy, but leaves execution_domain_hash zero. Backend discovery and exact
// adapter/context/queue domain binding belong exclusively to the setup source.
struct ProductionVideoResidentSetup {
  std::uint32_t contract_version =
      kProductionVideoBackendManagerContractVersion;
  VideoExecutionPlanRequest plan{};
};

struct ProductionVideoBackendSetupRequest {
  std::uint32_t contract_version =
      kProductionVideoBackendManagerContractVersion;
  ProductionVideoBackendSetupKind kind =
      ProductionVideoBackendSetupKind::invalid;
  ProductionVideoCpuPrimitiveSetup cpu{};
  ProductionVideoResidentSetup resident{};
};

struct ProductionVideoResidentSetupSourceResult {
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::ReadinessSnapshot readiness{};
  // The source freezes model/provider/device/context generations and resource
  // sizing into the neutral plan returned with its lease. The manager verifies
  // that this plan preserves the caller's canonical request before composing
  // it.
  VideoExecutionPlanRequest plan{};
  ProductionVideoSetupLease lease{};
  ProductionVideoSetupLeaseAcquireEvidence lease_acquire{};
  // The source is called only from Setup(). This flag attests that its
  // readiness is the immutable result of that setup attempt and that Execute()
  // needs no discovery, probing, session construction, or allocation.
  bool setup_completed_outside_frame_loop = false;
};

// Backend/runtime composition implements this interface. CameraPipeline gives
// the manager one canonical request; it cannot access a backend factory,
// adapter, model registry, provider probe, or setup lease directly.
class IProductionVideoBackendSetupSource {
public:
  virtual ~IProductionVideoBackendSetupSource() = default;

  [[nodiscard]] virtual ProductionVideoResidentSetupSourceResult
  AcquireResidentSetup(const ProductionVideoResidentSetup &request) noexcept =
      0;
};

struct ProductionVideoBackendSetupEvidence {
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::ReadinessSnapshot candidate_readiness{};
  ProductionVideoSetupLeaseAcquireEvidence lease_acquire{};
  VideoExecutionPrepareEvidence composition_prepare{};
  bool setup_source_called = false;
  bool candidate_setup_only = false;
  bool candidate_rejected = false;
  bool old_plan_retained = false;
  bool cpu_primitive_only = false;
};

enum class ProductionVideoBackendManagerState : std::uint8_t {
  invalid = 0,
  cpu_primitives_only,
  resident_ready,
  invalidated,
  shutdown_pending,
  stopped,
};

// Fixed cached state. Snapshot() performs no backend call, probing, discovery,
// synchronization, allocation, serialization, or JSON work.
struct ProductionVideoBackendManagerSnapshot {
  std::uint32_t contract_version =
      kProductionVideoBackendManagerContractVersion;
  ProductionVideoBackendManagerState state =
      ProductionVideoBackendManagerState::invalid;
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::ReadinessSnapshot readiness{};
  VideoExecutionCoordinatorSnapshot resident{};
  ProductionVideoSetupLeaseShutdownResult shutdown{};
  std::uint64_t cpu_primitive_bits = 0;
  std::uint64_t setup_attempts = 0;
  std::uint64_t setup_successes = 0;
  bool setup_complete = false;
  bool resident_prepared = false;
  bool frame_pending = false;
};

// Stable hash for the exact typed execution-domain identity. This is an
// invalidation key, not a substitute for comparing ExecutionDomainId itself.
[[nodiscard]] std::uint64_t ProductionVideoExecutionDomainHash(
    const hal::ExecutionDomainId &domain) noexcept;

class IProductionVideoBackendManager {
public:
  virtual ~IProductionVideoBackendManager() = default;

  [[nodiscard]] virtual hal::StatusId
  Setup(const ProductionVideoBackendSetupRequest &request,
        ProductionVideoBackendSetupEvidence *evidence,
        hal::ReasonId *reason) noexcept = 0;
  [[nodiscard]] virtual VideoExecutionFrameResult
  Execute(const hal::HostFrameBinding &binding) noexcept = 0;
  [[nodiscard]] virtual hal::StatusId
  CommitOutput(const hal::CompletionToken &completion,
               bool output_write_succeeded, hal::ReasonId *reason) noexcept = 0;
  virtual void Invalidate(hal::ReasonId reason) noexcept = 0;
  [[nodiscard]] virtual hal::StatusId
  ResetTemporal(hal::ReasonId reason) noexcept = 0;
  [[nodiscard]] virtual ProductionVideoSetupLeaseShutdownResult
  Shutdown() noexcept = 0;
  [[nodiscard]] virtual ProductionVideoBackendManagerSnapshot
  Snapshot() const noexcept = 0;
};

// One backend-agnostic lifecycle owner for the camera pipeline. Resident setup
// is transactional through ProductionVideoExecutionComposition; CPU setup only
// caches the separate primitive-table lane.
class ProductionVideoBackendManager final
    : public IProductionVideoBackendManager {
public:
  explicit ProductionVideoBackendManager(
      std::unique_ptr<IProductionVideoBackendSetupSource> setup_source);
  ~ProductionVideoBackendManager() override;

  ProductionVideoBackendManager(const ProductionVideoBackendManager &) =
      delete;
  ProductionVideoBackendManager &
  operator=(const ProductionVideoBackendManager &) = delete;

  [[nodiscard]] hal::StatusId
  Setup(const ProductionVideoBackendSetupRequest &request,
        ProductionVideoBackendSetupEvidence *evidence,
        hal::ReasonId *reason) noexcept override;
  [[nodiscard]] VideoExecutionFrameResult
  Execute(const hal::HostFrameBinding &binding) noexcept override;
  [[nodiscard]] hal::StatusId
  CommitOutput(const hal::CompletionToken &completion,
               bool output_write_succeeded,
               hal::ReasonId *reason) noexcept override;
  void Invalidate(hal::ReasonId reason) noexcept override;
  [[nodiscard]] hal::StatusId
  ResetTemporal(hal::ReasonId reason) noexcept override;
  [[nodiscard]] ProductionVideoSetupLeaseShutdownResult
  Shutdown() noexcept override;
  [[nodiscard]] ProductionVideoBackendManagerSnapshot
  Snapshot() const noexcept override;

private:
  void RefreshResidentSnapshot() noexcept;
  [[nodiscard]] bool HasRetainableSetup() const noexcept;

  std::unique_ptr<IProductionVideoBackendSetupSource> setup_source_;
  ProductionVideoExecutionComposition composition_;
  ProductionVideoBackendManagerSnapshot snapshot_{};
  bool resident_selected_ = false;
};

} // namespace studiocast::video
