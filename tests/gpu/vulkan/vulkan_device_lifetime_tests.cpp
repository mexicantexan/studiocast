#include "core/vulkan/vulkan_device.h"

#include <iostream>
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

} // namespace

int main() {
  const bool unique = CheckShutdownLifetime(false);
  const bool retained = CheckShutdownLifetime(true);
  return unique && retained ? 0 : 1;
}
