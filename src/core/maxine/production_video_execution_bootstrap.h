#pragma once

#include "core/maxine/production_hal_owner.h"
#include "core/maxine/video_execution_factory.h"

#include <cstdint>
#include <memory>
#include <string>

namespace studiocast::maxine {

struct MaxineProductionVideoExecutionBootstrapRequest {
  video::PreparedMaxineHardwareRequest prepared_request{};
  std::uint64_t factory_id = 0;
  std::uint32_t selection_priority = 0;
  std::uint64_t minimum_synchronizations_per_frame = 1;
  std::uint64_t maximum_synchronizations_per_frame = 16;
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
  [[nodiscard]] static std::unique_ptr<
      MaxineProductionVideoExecutionBootstrap>
  Create(const MaxineProductionVideoExecutionBootstrapRequest &request,
         MaxineProductionVideoExecutionBootstrapResult *result) noexcept;

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

private:
  struct State;

  MaxineProductionVideoExecutionBootstrap() = default;

  std::shared_ptr<State> state_;
  std::unique_ptr<MaxineVideoExecutionFactory> factory_;
};

} // namespace studiocast::maxine
