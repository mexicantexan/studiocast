#pragma once

#include "core/open_video/model_pack_registry.h"
#include "core/open_video/video_execution_factory.h"
#include "core/video/production_video_backend_manager.h"

#include <memory>

namespace studiocast::open_cuda {

struct ProductionVideoBackendSetupDependencies {
  using ResolveModel =
      bool (*)(void *context, const video::CanonicalVideoStage &stage,
               const std::string &requested_model_id, const char *task,
               const char *role, open_video::ModelPack *pack,
               open_video::ValidatedModelArtifactIdentity *identity) noexcept;
  using AcquireLease = hal::StatusId (*)(
      void *context,
      const OpenCudaProductionVideoExecutionFactoryRequest &request,
      video::ProductionVideoSetupLease *destination,
      OpenCudaProductionVideoExecutionBootstrapResult *bootstrap,
      video::ProductionVideoSetupLeaseAcquireEvidence *evidence,
      hal::ReasonId *reason) noexcept;

  using AcquireCompatibility = video::ProductionVideoResidentSetupSourceResult (
          *)(void *context, const video::ProductionVideoResidentSetup &request,
             hal::ReasonId cuda_failure_reason) noexcept;

  ResolveModel resolve_model = nullptr;
  AcquireLease acquire_lease = nullptr;
  void *context = nullptr;
  AcquireCompatibility acquire_compatibility = nullptr;
};

// Creates the production setup-only Open CUDA/Open Video source. The returned
// object owns model discovery, artifact validation, provider selection, exact
// resource sizing, and CUDA/ORT setup behind the neutral backend-manager
// contract. AcquireResidentSetup is the only operation that may perform those
// activities; the prepared lease executes from cached resources.
[[nodiscard]] std::unique_ptr<video::IProductionVideoBackendSetupSource>
CreateProductionVideoBackendSetupSource() noexcept;

// Hermetic setup-boundary seam. Production callers use the overload above.
// Injected callbacks replace only external model discovery and native lease
// acquisition; canonical policy, evidence binding, and returned-plan
// construction remain production code.
[[nodiscard]] std::unique_ptr<video::IProductionVideoBackendSetupSource>
CreateProductionVideoBackendSetupSourceForTesting(
    ProductionVideoBackendSetupDependencies dependencies) noexcept;

} // namespace studiocast::open_cuda
