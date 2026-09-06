#pragma once

#include "core/video/cpu_primitive_hal.h"
#include "core/video/effects/broadcast_effects.h"
#include "core/video/production_video_execution_composition.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace studiocast::video {

inline constexpr std::uint32_t kProductionVideoBackendManagerContractVersion =
    2;

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

using ProductionVideoWaitPredicate = bool (*)(void *context) noexcept;
using ProductionVideoNowNanoseconds =
    std::uint64_t (*)(void *context) noexcept;
using ProductionVideoPollWait =
    void (*)(void *context, std::uint64_t wait_nanoseconds) noexcept;

// Backend-neutral bounded completion policy. Callbacks and context are
// borrowed until Shutdown proves driver_context_release_permitted; the
// generation lets setup sources reject a stale context without dereferencing
// it. V1 execution remains synchronous and completion-shaped.
struct ProductionVideoBoundedWaitPolicy {
  std::uint32_t contract_version =
      kProductionVideoBackendManagerContractVersion;
  std::uint64_t timeout_nanoseconds = 0;
  std::uint64_t poll_interval_nanoseconds = 0;
  std::uint32_t maximum_poll_count = 0;
  ProductionVideoWaitPredicate stop_requested = nullptr;
  ProductionVideoWaitPredicate cancel_requested = nullptr;
  ProductionVideoNowNanoseconds now_nanoseconds = nullptr;
  ProductionVideoPollWait poll_wait = nullptr;
  void *callback_context = nullptr;
  std::uint64_t callback_context_generation = 0;
  hal::ResourceLifetime callback_context_lifetime =
      hal::ResourceLifetime::invalid;
};

// The caller supplies canonical semantics, host descriptors, generations, and
// policy, but leaves execution_domain_hash zero. Backend discovery and exact
// adapter/context/queue domain binding belong exclusively to the setup source.
struct ProductionVideoResidentSetup {
  std::uint32_t contract_version =
      kProductionVideoBackendManagerContractVersion;
  // Setup owns a value copy, so model IDs, replacement/HDRI paths, colors, and
  // every Maxine/Open Video parameter remain valid for the complete source
  // call without borrowing GUI/config string storage.
  effects::BroadcastCameraEffects effects{};
  VideoExecutionPlanRequest plan{};
  ProductionVideoBoundedWaitPolicy wait{};
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
  enum class ExecutionKind : std::uint8_t {
    invalid = 0,
    resident_island,
    typed_compatibility,
  };
  ExecutionKind execution_kind = ExecutionKind::resident_island;
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::ReadinessSnapshot readiness{};
  effects::BroadcastCameraEffects canonical_effects{};
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
  struct ModelSelection {
    hal::SemanticEffectId semantic = hal::SemanticEffectId::invalid;
    std::string requested_model_id;
    std::string selected_model_id;
    hal::ModelArtifactIdentity artifact{};
    bool default_selected = false;
  };
  std::array<ModelSelection, hal::kMaxPreparedStages> model_selections{};
  std::size_t model_selection_count = 0;

  static constexpr std::size_t kMaxCompatibilityDomains = 4;
  struct CompatibilityDomain {
    hal::ExecutionDomainId domain{};
    hal::BackendId backend = hal::BackendId::invalid;
    hal::SemanticEffectProviderId provider =
        hal::SemanticEffectProviderId::invalid;
    std::uint32_t first_stage_id = 0;
    std::uint32_t last_stage_id = 0;
  };
  struct CompatibilityExecutionTopology {
    std::uint32_t contract_version =
        kProductionVideoBackendManagerContractVersion;
    std::array<CompatibilityDomain, kMaxCompatibilityDomains> domains{};
    std::size_t domain_count = 0;
    std::array<hal::ExecutionBoundary, hal::kMaxExecutionBoundaries>
        boundaries{};
    std::size_t boundary_count = 0;
    hal::TransferCounters expected_transfers{};
    bool exact_ordered_boundaries = false;
    bool cross_adapter_residency_forbidden = true;
  };
  CompatibilityExecutionTopology compatibility_topology{};
  // Mutually exclusive with lease. The source prepares all sessions,
  // buffers, boundaries, and scratch before returning this neutral executor.
  std::unique_ptr<IVideoExecutionExecutor> compatibility_executor;
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

// Implemented by the backend-composition target. The returned setup source
// owns the deterministic in-tree adapter registry and performs all
// backend/provider/model discovery during Setup; callers never construct or
// select a backend implementation directly.
[[nodiscard]] std::unique_ptr<IProductionVideoBackendSetupSource>
CreateStaticProductionVideoBackendSetupSource() noexcept;

struct StaticProductionVideoBackendSetupSourcesForTesting {
  std::unique_ptr<IProductionVideoBackendSetupSource> maxine;
  std::unique_ptr<IProductionVideoBackendSetupSource> open_cuda;
  std::unique_ptr<IProductionVideoBackendSetupSource> vulkan;
};

[[nodiscard]] std::unique_ptr<IProductionVideoBackendSetupSource>
CreateStaticProductionVideoBackendSetupSourceForTesting(
    StaticProductionVideoBackendSetupSourcesForTesting sources) noexcept;

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
  compatibility_ready,
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
  ProductionVideoResidentSetupSourceResult::CompatibilityExecutionTopology
      compatibility_topology{};
  VideoExecutionAdapterSnapshot compatibility{};
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
[[nodiscard]] inline std::uint64_t ProductionVideoExecutionDomainHash(
    const hal::ExecutionDomainId &domain) noexcept {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const auto mix = [&](std::uint64_t value) {
    hash ^= value;
    hash *= UINT64_C(1099511628211);
  };
  mix(static_cast<std::uint64_t>(domain.backend));
  mix(static_cast<std::uint64_t>(domain.api_family));
  mix(domain.physical_device.vendor_id);
  mix(domain.physical_device.device_id);
  for (const auto byte : domain.physical_device.stable_id)
    mix(byte);
  mix(domain.logical_context_generation);
  mix(static_cast<std::uint64_t>(domain.allocator_family));
  mix(static_cast<std::uint64_t>(domain.memory_handle_family));
  mix(domain.queue_or_stream_identity);
  mix(static_cast<std::uint64_t>(domain.ownership_model));
  mix(static_cast<std::uint64_t>(domain.synchronization_protocol));
  mix(domain.adapter_instance_id);
  return hash == 0 ? 1 : hash;
}

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
  std::unique_ptr<IVideoExecutionExecutor> compatibility_executor_;
  ProductionVideoBackendManagerSnapshot snapshot_{};
  bool resident_selected_ = false;
};

} // namespace studiocast::video
