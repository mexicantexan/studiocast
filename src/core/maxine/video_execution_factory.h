#pragma once

#include "core/video/prepared_maxine_hardware_execution_owner.h"
#include "core/video/video_adapter_factory.h"

#include <cstdint>
#include <memory>

namespace studiocast::maxine {

// Setup-time configuration for the coordinator-facing Maxine factory. The
// pointed-to cached registration must outlive the factory unless
// adapter_lifetime owns it; the request template is copied. This surface
// deliberately contains no NvCV, VFX, AR, CUDA SDK, or provider types.
struct MaxineVideoExecutionFactoryConfiguration {
  const hal::AdapterRegistration *adapter_registration = nullptr;
  // Keeps a backend-owned adapter and its cached registration alive for every
  // executor prepared by this factory. Tests and legacy injected factories may
  // leave this empty when they own the registration for the full coordinator
  // lifetime.
  std::shared_ptr<void> adapter_lifetime;
  void *device_loss_owner = nullptr;
  void (*notify_device_lost)(void *owner) noexcept = nullptr;
  video::PreparedMaxineHardwareRequest setup_template{};
  std::uint64_t factory_id = 0;
  std::uint32_t selection_priority = 0;
  std::uint64_t minimum_synchronizations_per_frame = 0;
  std::uint64_t maximum_synchronizations_per_frame = 0;
};

class MaxineVideoExecutionFactory final : public video::IVideoAdapterFactory {
public:
  explicit MaxineVideoExecutionFactory(
      MaxineVideoExecutionFactoryConfiguration configuration);

  [[nodiscard]] video::VideoAdapterFactoryRegistration registration() noexcept;

  [[nodiscard]] video::VideoAdapterFactoryPreflight
  Preflight(const video::VideoExecutionPlanRequest &request,
            const hal::ResolutionResult &frozen_resolution,
            const hal::DomainPlan &frozen_domain_plan,
            video::VideoExecutionRoute route) noexcept override;

  [[nodiscard]] std::unique_ptr<video::IVideoExecutionExecutor>
  Prepare(const video::FrozenVideoExecutionPlan &plan,
          video::VideoAdapterPrepareEvidence *evidence) noexcept override;

private:
  MaxineVideoExecutionFactoryConfiguration configuration_{};
};

} // namespace studiocast::maxine
