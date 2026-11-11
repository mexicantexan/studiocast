#pragma once

#include "core/maxine/production_hal_owner.h"
#include "core/maxine/video_execution_factory.h"
#include "core/video/production_video_backend_manager.h"
#include "core/video/production_video_setup_lease.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace studiocast::maxine {

struct MaxineProductionVideoExecutionBootstrapRequest {
  video::PreparedMaxineHardwareRequest prepared_request{};
  std::uint64_t factory_id = 0;
  std::uint32_t selection_priority = 0;
  std::uint64_t minimum_synchronizations_per_frame = 1;
  std::uint64_t maximum_synchronizations_per_frame = 16;
  // Retains runtime loaders and prepared replacement storage behind the lease.
  // Production setup sources populate this; direct tests may inject an exact
  // lifetime for their fake SDK/runtime objects.
  std::shared_ptr<void> runtime_lifetime;
};

struct MaxineProductionVideoExecutionBootstrapResult {
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  video::PreparedMaxineHardwareEligibility eligibility =
      video::PreparedMaxineHardwareEligibility::runtime_unavailable;
  hal::ExecutionDomainId execution_domain{};
  hal::ResourceBudget known_minimum_budget{};
  std::string diagnostic;
};

// Backend-contained setup-time lifetime for one production Maxine execution
// island. It creates exactly one native owner/executor and exposes the cached
// adapter and factory registrations needed by the neutral coordinator.
//
// The bootstrap must outlive every registry/coordinator that stores its raw
// factory pointer. Prepared executors retain the shared native state
// independently, so retiring the external bootstrap after factory preparation
// cannot leave their borrowed adapter/session dangling.
class MaxineProductionVideoExecutionBootstrap final {
public:
  [[nodiscard]] static std::unique_ptr<MaxineProductionVideoExecutionBootstrap>
  Create(const MaxineProductionVideoExecutionBootstrapRequest &request,
         MaxineProductionVideoExecutionBootstrapResult *result) noexcept;

  // Creates and transactionally acquires one exact typed setup lease. A
  // rejected candidate is shut down and destroyed while an existing live
  // destination remains untouched.
  [[nodiscard]] static hal::StatusId AcquireSetupLease(
      const MaxineProductionVideoExecutionBootstrapRequest &request,
      video::ProductionVideoSetupLease *destination,
      video::ProductionVideoSetupLeaseAcquireEvidence *evidence,
      MaxineProductionVideoExecutionBootstrapResult *result,
      hal::ReasonId *reason) noexcept;

  ~MaxineProductionVideoExecutionBootstrap();

  MaxineProductionVideoExecutionBootstrap(
      const MaxineProductionVideoExecutionBootstrap &) = delete;
  MaxineProductionVideoExecutionBootstrap &
  operator=(const MaxineProductionVideoExecutionBootstrap &) = delete;

  [[nodiscard]] const hal::AdapterRegistration &
  adapter_registration() const noexcept;
  [[nodiscard]] video::VideoAdapterFactoryRegistration
  factory_registration() noexcept;
  [[nodiscard]] MaxineVideoExecutionFactory &factory() noexcept;
  [[nodiscard]] const MaxineProductionHalOwner *
  production_owner() const noexcept;
  [[nodiscard]] bool stopped() const noexcept;

  // Idempotently releases native Maxine resources while retaining the
  // published adapter and factory objects at stable addresses.
  void Shutdown() noexcept;

private:
  struct State;

  MaxineProductionVideoExecutionBootstrap() = default;

  std::shared_ptr<State> state_;
  std::unique_ptr<MaxineVideoExecutionFactory> factory_;
};

// Backend-private construction seam. With no explicit library path the source
// performs Maxine diagnosis and runtime discovery only from
// AcquireResidentSetup (never from Execute). Tests may provide one fake SDK
// exporting VFX, AR, and NvCV together and an already-initialized CUDA runtime.
struct MaxineProductionBackendSetupSourceConfiguration {
  std::filesystem::path sdk_library_path;
  std::filesystem::path vfx_model_directory;
  video::PreparedMaxineOpaqueRuntime injected_runtime{};
  std::shared_ptr<void> injected_runtime_lifetime;
  std::uint64_t factory_id = UINT64_C(0x4d4158494e450001);
  std::uint32_t selection_priority = 100;
  std::uint64_t minimum_synchronizations_per_frame = 1;
  std::uint64_t maximum_synchronizations_per_frame = 16;
  int cuda_device_ordinal = -1;
  bool bypass_system_diagnostics_for_testing = false;
};

[[nodiscard]] std::unique_ptr<video::IProductionVideoBackendSetupSource>
CreateMaxineProductionVideoBackendSetupSource(
    MaxineProductionBackendSetupSourceConfiguration configuration = {});

} // namespace studiocast::maxine
