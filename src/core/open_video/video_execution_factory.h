#pragma once

#include "core/video/video_adapter_factory.h"

#include <cstdint>
#include <memory>

namespace studiocast::open_cuda {

struct OpenCudaVideoExecutionCandidate {
  using DestroyOwner = void (*)(void *owner) noexcept;

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
  hal::AdapterRegistration registration{};
};

using OpenCudaPrepareVideoCandidate = hal::StatusId (*)(
    void *setup_context, const video::FrozenVideoExecutionPlan &plan,
    OpenCudaVideoExecutionCandidate *candidate,
    video::VideoAdapterPrepareEvidence *evidence) noexcept;

struct OpenCudaVideoExecutionCachedEvidence {
  hal::ReadinessSnapshot readiness{};
  hal::ExecutionDomainId execution_domain{};
  hal::ModelArtifactIdentity model{};
  hal::ProgramGenerations generations{};
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

// Static Open CUDA/Open Video adapter factory for the first conservative
// production route: exactly one strict ONNX Runtime CUDA virtual-background
// blur/remove stage over tight same-geometry RGB24, followed by one declared
// writer-copy CPU tail.
class OpenCudaVideoExecutionFactory final : public video::IVideoAdapterFactory {
public:
  explicit OpenCudaVideoExecutionFactory(
      OpenCudaVideoExecutionFactoryConfig config) noexcept;
  ~OpenCudaVideoExecutionFactory() override;

  OpenCudaVideoExecutionFactory(const OpenCudaVideoExecutionFactory &) = delete;
  OpenCudaVideoExecutionFactory &
  operator=(const OpenCudaVideoExecutionFactory &) = delete;

  [[nodiscard]] video::VideoAdapterFactoryRegistration Registration() noexcept;

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
