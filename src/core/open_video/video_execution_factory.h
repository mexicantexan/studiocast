#pragma once

#include "core/open_video/open_cuda_production_hal_owner.h"
#include "core/video/production_video_setup_lease.h"
#include "core/video/video_adapter_factory.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace studiocast::open_cuda {

struct OpenCudaVideoExecutionCandidate {
  using DestroyOwner = void (*)(void *owner) noexcept;
  using CommitPreparedOwner = hal::StatusId (*)(
      void *owner, const hal::PreparedProgram &program,
      const hal::ReadinessSnapshot &readiness, hal::ReasonId *reason) noexcept;
  using ObserveRuntimeOwner = void (*)(void *owner, hal::StatusId status,
                                       hal::ReasonId reason) noexcept;
  using ShutdownOwner = void (*)(void *owner) noexcept;

  OpenCudaVideoExecutionCandidate() = default;
  ~OpenCudaVideoExecutionCandidate();
  OpenCudaVideoExecutionCandidate(
      OpenCudaVideoExecutionCandidate &&other) noexcept;
  OpenCudaVideoExecutionCandidate &
  operator=(OpenCudaVideoExecutionCandidate &&other) noexcept;

  OpenCudaVideoExecutionCandidate(const OpenCudaVideoExecutionCandidate &) =
      delete;
  OpenCudaVideoExecutionCandidate &
  operator=(const OpenCudaVideoExecutionCandidate &) = delete;

  void Reset() noexcept;

  // The opaque owner keeps the resident adapter, CUDA context, stream, ORT
  // session, model bindings, buffers, and shutdown state alive. Backend
  // composition creates it from already-selected and already-validated setup
  // evidence. No vendor or model-pack type crosses this API.
  void *owner = nullptr;
  DestroyOwner destroy_owner = nullptr;
  CommitPreparedOwner commit_prepared_owner = nullptr;
  ObserveRuntimeOwner observe_runtime_owner = nullptr;
  ShutdownOwner shutdown_owner = nullptr;
  hal::AdapterRegistration registration{};
};

using OpenCudaPrepareVideoCandidate = hal::StatusId (*)(
    void *setup_context, const video::FrozenVideoExecutionPlan &plan,
    OpenCudaVideoExecutionCandidate *candidate,
    video::VideoAdapterPrepareEvidence *evidence) noexcept;

struct OpenCudaVideoExecutionCachedEvidence {
  hal::ReadinessSnapshot readiness{};
  hal::ExecutionDomainId execution_domain{};
  std::array<hal::ModelArtifactIdentity, hal::kMaxPreparedStages> models{};
  std::size_t stage_count = 0;
  hal::ProgramGenerations generations{};
  hal::ResourceBudget reserved_budget{};
  bool model_artifact_validated = false;
  bool provider_runtime_validated = false;
  bool device_context_validated = false;
  bool cuda_iobinding_validated = false;
  bool user_stream_bound = false;
  bool reusable_resources_supported = false;
  bool device_preprocess_supported = false;
  bool device_postprocess_supported = false;
  bool one_upload_one_readback = false;
  bool no_intermediate_crossings = false;
  bool stop_aware_bounded_wait = false;
  bool allocation_free_execute = false;
  std::uint64_t minimum_synchronizations_per_frame = 0;
  std::uint64_t maximum_synchronizations_per_frame = 0;
  std::uint32_t maximum_metadata_readbacks_per_frame = 0;
  std::uint32_t maximum_metadata_cpu_tails_per_frame = 0;
  std::uint32_t maximum_metadata_synchronizations_per_frame = 0;
};

struct OpenCudaVideoExecutionFactoryConfig {
  std::uint64_t factory_id = 0;
  std::uint32_t selection_priority = 0;
  OpenCudaVideoExecutionCachedEvidence cached{};
  OpenCudaPrepareVideoCandidate prepare_candidate = nullptr;
  // Borrowed setup-only state. It must outlive the factory and every Prepare
  // call. The prepared candidate owns all state needed by Execute.
  void *setup_context = nullptr;
};

// Setup-only production evidence. Model discovery, manifest parsing, artifact
// validation, provider selection, and sizing have already completed. The
// bootstrap creates one exact context/stream owner for the registry/factory
// lifetime. Prepared executors borrow that adapter and own only their prepared
// slot; retiring a plan therefore cannot dangle the registry registration.
// No work in this structure is consulted per frame.
struct OpenCudaProductionVideoExecutionFactoryRequest {
  std::uint64_t factory_id = 0;
  std::uint32_t selection_priority = 0;
  std::array<hal::SemanticEffectId, hal::kMaxPreparedStages> semantics{};
  std::array<hal::ModelArtifactIdentity, hal::kMaxPreparedStages> models{};
  std::size_t stage_count = 0;
  // Full canonical request.  stage_count/semantics/models above describe only
  // the compact selectable program for this preparation attempt.
  video::VideoExecutionPlanRequest plan{};
  std::array<hal::StatusId, hal::kMaxPreparedStages> canonical_status{};
  std::array<hal::ReasonId, hal::kMaxPreparedStages> canonical_reason{};
  hal::ProgramGenerations generations{};
  // Exact setup-time capacity reserved for the one prepared resident slot.
  // The lease publishes this value unchanged; frame execution cannot grow it.
  hal::ResourceBudget reserved_budget{};
  OpenCudaProductionBackendRequest backend{};
  bool model_artifact_validated = false;
  bool provider_runtime_validated = false;
  bool cuda_iobinding_validated = false;
  bool user_stream_bound = false;
  bool reusable_resources_supported = false;
  bool device_preprocess_supported = false;
  bool device_postprocess_supported = false;
  bool stop_aware_bounded_wait = false;
  bool allocation_free_execute = false;
};

struct OpenCudaProductionVideoExecutionBootstrapResult {
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  OpenCudaProductionPreparationResult owner{};
  video::ProductionVideoSetupLeaseSnapshot setup_lease{};
  bool exact_candidate_ready = false;
  OpenCudaSetupComponentId failed_component =
      OpenCudaSetupComponentId::invalid;
  video::VideoAdapterPrepareEvidence prepare{};
  video::ProductionVideoSetupLeaseShutdownResult candidate_shutdown{};
  hal::ResourceBudget actual_required_budget{};
};

// Static Open CUDA/Open Video adapter factory for the first conservative
// production route: exactly one strict ONNX Runtime CUDA virtual-background
// blur/remove stage over tight same-geometry RGB24, followed by one declared
// writer-copy CPU tail.
class OpenCudaVideoExecutionFactory final : public video::IVideoAdapterFactory {
public:
  explicit OpenCudaVideoExecutionFactory(
      OpenCudaVideoExecutionFactoryConfig config) noexcept;
  [[nodiscard]] static std::unique_ptr<OpenCudaVideoExecutionFactory>
  CreateProduction(
      const OpenCudaProductionVideoExecutionFactoryRequest &request,
      OpenCudaProductionVideoExecutionBootstrapResult *result) noexcept;
  [[nodiscard]] static hal::StatusId AcquireProductionSetupLease(
      const OpenCudaProductionVideoExecutionFactoryRequest &request,
      video::ProductionVideoSetupLease *destination,
      OpenCudaProductionVideoExecutionBootstrapResult *bootstrap,
      video::ProductionVideoSetupLeaseAcquireEvidence *evidence,
      hal::ReasonId *reason) noexcept;
  ~OpenCudaVideoExecutionFactory() override;

  OpenCudaVideoExecutionFactory(const OpenCudaVideoExecutionFactory &) = delete;
  OpenCudaVideoExecutionFactory &
  operator=(const OpenCudaVideoExecutionFactory &) = delete;

  [[nodiscard]] video::VideoAdapterFactoryRegistration Registration() noexcept;
  [[nodiscard]] hal::AdapterRegistration adapter_registration() const noexcept;

  // Performs exactly one native bounded shutdown attempt. The adapter facade
  // is retired only after the native owner proves context release is safe.
  [[nodiscard]] OpenCudaProductionShutdownResult
  RequestProductionOwnerShutdown() noexcept;
  [[nodiscard]] OpenCudaProductionShutdownResult
  production_shutdown_snapshot() const noexcept;
  [[nodiscard]] bool production_owner_retired() const noexcept;

  [[nodiscard]] video::VideoAdapterFactoryPreflight
  Preflight(const video::VideoExecutionPlanRequest &request,
            const hal::ResolutionResult &frozen_resolution,
            const hal::DomainPlan &frozen_domain_plan,
            video::VideoExecutionRoute route) noexcept override;

  [[nodiscard]] std::unique_ptr<video::IVideoExecutionExecutor>
  Prepare(const video::FrozenVideoExecutionPlan &plan,
          video::VideoAdapterPrepareEvidence *evidence) noexcept override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::open_cuda
