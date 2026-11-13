#include "core/open_video/ncnn_vulkan_context.h"
#include "core/vulkan/vulkan_device.h"

#include <iostream>
#include <limits>
#include <memory>
#include <thread>
#include <utility>

namespace studiocast::vulkan::detail {

struct VulkanDeviceTestAccess {
  static void AttachContext(VulkanDevice &device,
                            std::shared_ptr<VulkanContextState> context) {
    device.context_ = std::move(context);
    device.initialized_ = true;
  }
};

} // namespace studiocast::vulkan::detail

namespace {

bool CheckShutdownLifetime(bool retain_external_owner) {
  using namespace studiocast::vulkan;
  using detail::VulkanContextState;

  bool mutex_released = false;
  bool marked_shutdown = false;
  std::unique_ptr<VulkanContextState> retired;
  auto context = std::shared_ptr<VulkanContextState>(
      new VulkanContextState, [&](VulkanContextState *value) {
        // Probe from another thread: try_lock by the owning thread is
        // undefined.
        std::thread observer([&] {
          mutex_released = value->submission_mutex.try_lock();
          if (mutex_released)
            value->submission_mutex.unlock();
        });
        observer.join();
        marked_shutdown =
            value->HealthState().health == VulkanContextHealth::shutdown;
        // Keep storage alive until Shutdown returns even in the failing case.
        retired.reset(value);
      });
  context->health.health = VulkanContextHealth::healthy;

  VulkanDevice device;
  detail::VulkanDeviceTestAccess::AttachContext(device, context);
  if (!retain_external_owner)
    context.reset();

  device.Shutdown();
  if (device.Initialized() ||
      (retain_external_owner && (retired || context->Active()))) {
    std::cerr << "Shutdown must invalidate the device and retain external "
                 "ownership\n";
    return false;
  }
  context.reset();
  device.Shutdown();

  if (!retired || !marked_shutdown || !mutex_released) {
    std::cerr
        << "Context ownership must be released after its submission mutex "
           "is unlocked and shutdown is marked\n";
    return false;
  }
  return true;
}

bool CheckAggregateReservations() {
  using namespace studiocast::vulkan;
  auto context = std::make_shared<detail::VulkanContextState>();
  context->health.health = VulkanContextHealth::healthy;
  context->allocation_stats.budget_bytes = 128;
  context->allocation_stats.resource_limit = 2;
  std::string error;
  if (studiocast::open_vulkan::ValidateProductionVulkanCoreFeatures(true, false,
                                                                    &error) ||
      error.find("vulkan_shader_float64_unavailable") == std::string::npos ||
      !studiocast::open_vulkan::ValidateProductionVulkanCoreFeatures(
          false, false, &error) ||
      !studiocast::open_vulkan::ValidateProductionVulkanCoreFeatures(true, true,
                                                                     &error))
    return false;
  if (!context->ReserveAllocation(32, &error) ||
      !context->ReserveAllocation(32, &error) ||
      context->ReserveAllocation(1, &error) ||
      context->AllocationStats().current_bytes != 64 ||
      context->AllocationStats().allocation_count != 2)
    return false;
  context->ReleaseAllocation(32);
  if (context->ReserveAllocation(100, &error) ||
      !context->ReserveAllocation(64, &error) ||
      context->AllocationStats().high_water_allocation_count != 2)
    return false;
  context->ReleaseAllocation(64);
  context->ReleaseAllocation(32);
  VulkanDevice device;
  detail::VulkanDeviceTestAccess::AttachContext(device, context);
  if (device.shader_float64_enabled() ||
      !device.ConfigureAllocationLimits(64, 1, &error) ||
      device.ConfigureAllocationLimits(128, 2, &error) ||
      !context->ReserveAllocation(64, &error) ||
      context->ReserveAllocation(1, &error))
    return false;
  context->ReleaseAllocation(64);
  if (context->AllocationStats().current_bytes != 0 ||
      context->AllocationStats().allocation_count != 0 ||
      context->AllocationStats().total_allocation_count != 4)
    return false;
  context->allocation_stats.total_allocation_count =
      std::numeric_limits<std::uint64_t>::max();
  return !context->ReserveAllocation(1, &error) &&
         context->AllocationStats().current_bytes == 0 &&
         context->AllocationStats().allocation_count == 0 &&
         context->AllocationStats().total_allocation_count ==
             std::numeric_limits<std::uint64_t>::max();
}

} // namespace

int main() {
  const bool unique = CheckShutdownLifetime(false);
  const bool retained = CheckShutdownLifetime(true);
  return unique && retained && CheckAggregateReservations() ? 0 : 1;
}
