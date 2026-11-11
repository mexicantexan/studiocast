#pragma once

#include "core/video/video_adapter_factory.h"

#include <cstdint>
#include <memory>

namespace studiocast::vulkan {

// Setup-only configuration for the production-qualified Open Vulkan factory.
// The public surface is deliberately neutral: Vulkan device, queue, and
// resource types remain private to the implementation.
struct VulkanVideoExecutionFactoryConfiguration {
  std::uint64_t factory_id = UINT64_C(0x56554c4b00000001);
  std::uint32_t selection_priority = 300;
  std::uint64_t minimum_synchronizations_per_frame = 2;
  std::uint64_t maximum_synchronizations_per_frame = 3;
};

class VulkanVideoExecutionFactory final : public video::IVideoAdapterFactory {
public:
  explicit VulkanVideoExecutionFactory(
      VulkanVideoExecutionFactoryConfiguration configuration = {});

  // Hermetic backend-off seam. The registration and its resident adapter must
  // outlive the factory and every executor produced by it. The factory still
  // validates exact Vulkan production evidence and never selects a backend or
  // provider through this constructor.
  VulkanVideoExecutionFactory(
      const hal::AdapterRegistration &injected_registration,
      VulkanVideoExecutionFactoryConfiguration configuration = {});

  ~VulkanVideoExecutionFactory() override;
  VulkanVideoExecutionFactory(const VulkanVideoExecutionFactory &) = delete;
  VulkanVideoExecutionFactory &
  operator=(const VulkanVideoExecutionFactory &) = delete;

  // Performs all production probing and resource-owner construction. No
  // probing, allocation, or capability discovery occurs in Preflight or
  // Execute.
  [[nodiscard]] hal::StatusId Initialize(hal::ReasonId *reason) noexcept;
  [[nodiscard]] bool initialized() const noexcept;
  [[nodiscard]] hal::StatusId initialization_status() const noexcept;
  [[nodiscard]] hal::ReasonId initialization_reason() const noexcept;

  [[nodiscard]] hal::AdapterRegistration adapter_registration() const noexcept;
  [[nodiscard]] video::VideoAdapterFactoryRegistration
  factory_registration() noexcept;

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

} // namespace studiocast::vulkan
