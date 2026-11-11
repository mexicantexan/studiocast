#pragma once

#include "core/cuda/cuda_driver_api.h"
#include "core/hal/prepared_program.h"
#include "core/open_video/model_pack_registry.h"
#include "core/open_video/open_cuda_hal_adapter.h"
#include "core/open_video/open_cuda_prepared_resident_executor.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace studiocast::open_cuda {

inline constexpr std::uint32_t kOpenCudaProductionOwnerApiVersion = 1;

enum class OpenCudaProductionOwnerStatus : std::uint16_t {
  invalid = 0,
  domain_ready = 1,
  prepared_capability_committed = 2,
  malformed_request = 3,
  driver_initialize_failed = 4,
  context_unavailable = 5,
  context_mismatch = 6,
  stream_create_failed = 7,
  stream_destroy_failed = 8,
  execution_domain_invalid = 9,
  allocation_failed = 10,
  resource_exhausted = 11,
  capability_evidence_incomplete = 12,
  wrong_execution_domain = 13,
  stale_generation = 14,
  foreign_adapter = 15,
  quarantined = 16,
  shutdown_pending = 17,
  stopped = 18,
};

// Test-only initialization is supplied through the same CudaDriverApi
// instance the owner retains. Production leaves this callback null and uses
// CudaDriverApi::Initialize().
using OpenCudaProductionDriverInitializer =
    bool (*)(void *context, cuda::CudaDriverApi *driver,
             std::string *error_out) noexcept;

struct OpenCudaProductionBackendRequest {
  std::uint32_t api_version = kOpenCudaProductionOwnerApiVersion;
  std::uint64_t adapter_instance_id = 0;
  int cuda_device_id = 0;

  // When required, this is setup-time selection evidence from the caller.
  // It is compared after the owner has created its exact stream so a rejected
  // candidate also exercises complete stream rollback.
  bool require_expected_context = false;
  cuda::CudaContextIdentity expected_context{};

  // These inputs are already-selected setup facts. This owner does not scan a
  // registry, parse a manifest, validate an artifact, or create an ORT session.
  open_video::ModelPack model_pack{};
  open_video::ValidatedModelArtifactIdentity model_artifact{};
  std::uint64_t configuration_generation = 0;
  std::uint64_t model_preprocess_generation = 0;
  std::uint64_t model_artifact_hash = 0;
  int strength = 8;
  std::array<std::uint8_t, 3> remove_background_rgb{0, 0, 0};
  std::vector<std::uint8_t> replacement_rgb{};
  int replacement_width = 0;
  int replacement_height = 0;
  int key_light_intensity = 50;
  int key_light_temperature_preset = 0;
  int key_light_direction_pan_degrees = 0;
  int vignette_intensity = 35;
  bool vignette_tracked_center = false;
  PreparedOpenCudaModelDependencies dependencies{};
  int denoise_strength = 50;
  int auto_frame_strength = 50;
  int auto_frame_smoothing = 50;
  float auto_frame_headroom = .15F;
  int eye_contact_strength = 50;
  bool eye_contact_look_away = false;
  cuda::CudaPreparedTransferCompletion::WaitPolicy wait_policy{};

  // Null selects the production CUDA/ORT runtime owned by the executor.
  // An injected runtime must outlive this owner.
  IPreparedOpenCudaRuntime *runtime = nullptr;

  OpenCudaProductionDriverInitializer initialize_driver = nullptr;
  void *initialize_driver_context = nullptr;
};

struct OpenCudaProductionPreparationResult {
  std::uint32_t api_version = kOpenCudaProductionOwnerApiVersion;
  OpenCudaProductionOwnerStatus status = OpenCudaProductionOwnerStatus::invalid;
  hal::StatusId hal_status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  cuda::CudaContextIdentity context{};
  hal::ExecutionDomainId execution_domain{};
  std::uint64_t stream_identity = 0;
  bool bootstrap_domain_ready = false;
  bool prepared_capability_committed = false;
  bool retained_cleanup_island = false;
  std::string diagnostic;
};

struct OpenCudaPreparedCapabilitySnapshot {
  bool committed = false;
  std::uint64_t program_id = 0;
  hal::ProgramInvalidationKey invalidation_key{};
  hal::ExecutionDomainId execution_domain{};
  hal::SemanticEffectProviderId provider =
      hal::SemanticEffectProviderId::invalid;
  std::uint32_t ready_stage_mask = 0;
};

struct OpenCudaProductionShutdownResult {
  std::uint32_t api_version = kOpenCudaProductionOwnerApiVersion;
  OpenCudaProductionOwnerStatus status = OpenCudaProductionOwnerStatus::invalid;
  hal::StatusId hal_status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  PreparedOpenCudaShutdownSnapshot executor{};
  std::uint64_t shutdown_attempts = 0;
  std::uint64_t owner_stream_destroy_attempts = 0;
  std::uint64_t owner_stream_destructions = 0;
  bool shutdown_requested = false;
  bool quiescent = false;
  bool driver_context_release_permitted = false;
};

struct OpenCudaProductionOwnerSnapshot {
  std::uint32_t api_version = kOpenCudaProductionOwnerApiVersion;
  OpenCudaProductionOwnerStatus status = OpenCudaProductionOwnerStatus::invalid;
  cuda::CudaContextIdentity context{};
  hal::ExecutionDomainId execution_domain{};
  OpenCudaPreparedCapabilitySnapshot prepared_capability{};
  OpenCudaProductionShutdownResult shutdown{};
  bool bootstrap_domain_ready = false;
  bool quarantined = false;
  bool device_lost = false;
  bool stopped = false;
};

// Owns one complete v1 Open CUDA resident island:
//
//   CudaDriverApi/retained primary context -> exact non-default CUstream
//   -> OpenCudaPreparedResidentExecutor -> ResidentAdapter
//
// CreateBackend always constructs a separate candidate. A caller can retain
// its old valid owner until the returned candidate is domain-ready, providing
// rollback without mutating live resources.
class OpenCudaProductionHalOwner final {
public:
  [[nodiscard]] static std::unique_ptr<OpenCudaProductionHalOwner>
  CreateBackend(const OpenCudaProductionBackendRequest &request,
                OpenCudaProductionPreparationResult *result) noexcept;

  ~OpenCudaProductionHalOwner();

  OpenCudaProductionHalOwner(const OpenCudaProductionHalOwner &) = delete;
  OpenCudaProductionHalOwner &
  operator=(const OpenCudaProductionHalOwner &) = delete;

  [[nodiscard]] const cuda::CudaContextIdentity &
  context_identity() const noexcept;
  [[nodiscard]] const hal::ExecutionDomainId &execution_domain() const noexcept;
  [[nodiscard]] cuda::CUstream prepared_stream() const noexcept;
  [[nodiscard]] bool bootstrap_domain_ready() const noexcept;
  [[nodiscard]] bool prepared_capability_committed() const noexcept;

  [[nodiscard]] cuda::CudaDriverApi *driver() noexcept;
  [[nodiscard]] const cuda::CudaDriverApi *driver() const noexcept;
  [[nodiscard]] OpenCudaPreparedResidentExecutor *executor() noexcept;
  [[nodiscard]] const OpenCudaPreparedResidentExecutor *
  executor() const noexcept;
  [[nodiscard]] ExecutorPrepareResult LastPrepareResult() const noexcept;
  [[nodiscard]] ResidentAdapter *adapter() noexcept;
  [[nodiscard]] const ResidentAdapter *adapter() const noexcept;

  // The later video facade calls this only after its PreparedProgramSession
  // successfully configures through this owner's adapter. Domain bootstrap
  // alone can never set prepared capability.
  [[nodiscard]] OpenCudaProductionPreparationResult
  CommitPreparedCapability(const hal::PreparedProgram &program,
                           const hal::ReadinessSnapshot &readiness) noexcept;

  // Consumes a cached typed runtime outcome. No device probing occurs.
  void ObserveRuntimeStatus(hal::StatusId status,
                            hal::ReasonId reason) noexcept;

  // Retry while pending. Driver/context/executor/stream ownership is retained
  // until the returned snapshot explicitly permits release.
  [[nodiscard]] OpenCudaProductionShutdownResult RequestShutdown() noexcept;
  [[nodiscard]] OpenCudaProductionShutdownResult
  ShutdownSnapshot() const noexcept;
  [[nodiscard]] OpenCudaProductionOwnerSnapshot Snapshot() const noexcept;

private:
  OpenCudaProductionHalOwner();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::open_cuda
