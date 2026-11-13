#include "core/open_video/ncnn_vulkan_context.h"
#include "core/vulkan/vulkan_device.h"

#include <algorithm>
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
  static void AttachExactContext(VulkanDevice &device,
                                 std::shared_ptr<VulkanContextState> context) {
    AttachContext(device, std::move(context));
    device.loader_ = device.context_->loader;
    device.context_identity_ = device.context_->identity;
    device.device_ = device.context_->device;
    device.queue_ = device.context_->queue;
    device.fence_ = device.context_->fence;
  }
};

} // namespace studiocast::vulkan::detail

namespace {

using namespace studiocast::vulkan;
struct RetirementDriver {
  static inline RetirementDriver *active = nullptr;
  std::shared_ptr<detail::VulkanContextState> context =
      std::make_shared<detail::VulkanContextState>();
  VulkanDevice device;
  VkResult next_wait = VK_TIMEOUT;
  unsigned resets = 0, submits = 0, waits = 0;
  VkDevice waited_device = nullptr;
  VkFence waited_fence = nullptr;
  std::uint64_t timeout = 0;
  bool exact_wait = true, locked = false;
  enum class Mutation {
    none,
    identity,
    fence,
    serial
  } mutation = Mutation::none;
  RetirementDriver() {
    active = this;
    context->loader = std::make_shared<VulkanLoader>();
    context->identity = {41, 53};
    context->device = reinterpret_cast<VkDevice>(1);
    context->queue = reinterpret_cast<VkQueue>(2);
    context->fence = reinterpret_cast<VkFence>(3);
    context->health.health = VulkanContextHealth::healthy;
    context->f().vkResetFences = Reset;
    context->f().vkQueueSubmit = Submit;
    context->f().vkWaitForFences = Wait;
    detail::VulkanDeviceTestAccess::AttachExactContext(device, context);
  }
  ~RetirementDriver() {
    // Fake handles have no native work; release their storage without creating
    // intentional process-lifetime quarantine leaks in sanitizer runs.
    context->health.health = VulkanContextHealth::fatal_submission_error;
    context->health.completed_serial = context->health.submitted_serial;
    device.Shutdown();
  }
  static VkResult Reset(VkDevice, std::uint32_t, const VkFence *) {
    ++active->resets;
    return VK_SUCCESS;
  }
  static VkResult Submit(VkQueue, std::uint32_t, const VkSubmitInfo *,
                         VkFence) {
    ++active->submits;
    return VK_SUCCESS;
  }
  static VkResult Wait(VkDevice device, std::uint32_t count,
                       const VkFence *fences, VkBool32 all,
                       std::uint64_t timeout) {
    auto &s = *active;
    ++s.waits;
    s.waited_device = device;
    s.waited_fence = count == 1 && fences ? fences[0] : nullptr;
    s.timeout = timeout;
    s.exact_wait &= count == 1 && all == 1 && device == s.context->device &&
                    s.waited_fence == s.context->fence;
    std::thread observer([&] {
      const bool acquired = s.context->submission_mutex.try_lock();
      s.locked = !acquired;
      if (acquired)
        s.context->submission_mutex.unlock();
    });
    observer.join();
    if (s.mutation == Mutation::identity)
      ++s.context->identity.generation;
    else if (s.mutation == Mutation::fence)
      s.context->fence = reinterpret_cast<VkFence>(4);
    else if (s.mutation == Mutation::serial)
      ++s.context->health.submitted_serial;
    return s.next_wait;
  }
  bool Timeout() {
    std::string error;
    return !device.SubmitAndWait(reinterpret_cast<VkCommandBuffer>(5),
                                 &error) &&
           context->health.submitted_serial == 1 &&
           context->health.completed_serial == 0 &&
           context->health.health == VulkanContextHealth::unsafe_timeout &&
           context->health.poisoned && !device.SafeToDestroyResources();
  }
};

bool CheckLateSubmissionProof() {
  RetirementDriver s;
  std::string error;
  if (!s.Timeout())
    return false;
  const auto failed = s.context->HealthSnapshot();
  s.next_wait = VK_SUCCESS;
  if (!s.device.TryProvePendingSubmissionQuiescence(&error) ||
      !s.device.SafeToDestroyResources() || s.resets != 1 || s.submits != 1 ||
      s.waits != 2 || !s.exact_wait || !s.locked ||
      s.waited_fence != reinterpret_cast<VkFence>(3) ||
      s.waited_device != reinterpret_cast<VkDevice>(1) ||
      s.timeout != VK_DEFAULT_FENCE_TIMEOUT_NS)
    return false;
  const auto retired = s.context->HealthSnapshot();
  if (retired.submitted_serial != failed.submitted_serial ||
      retired.completed_serial != failed.submitted_serial ||
      retired.health != failed.health || retired.poisoned != failed.poisoned ||
      retired.last_result != failed.last_result ||
      retired.reason_code != failed.reason_code ||
      retired.operation != failed.operation ||
      retired.message != failed.message ||
      s.device.SubmitAndWait(reinterpret_cast<VkCommandBuffer>(5), &error) ||
      !s.device.TryProvePendingSubmissionQuiescence(&error) || s.resets != 1 ||
      s.submits != 1 || s.waits != 2)
    return false;
  return true;
}

bool CheckFailedSubmissionProofs() {
  RetirementDriver s;
  std::string error;
  if (!s.Timeout() || s.device.TryProvePendingSubmissionQuiescence(&error) ||
      s.waits != 2 || s.device.SafeToDestroyResources() ||
      s.context->health.completed_serial != 0)
    return false;
  s.next_wait = VK_ERROR_DEVICE_LOST;
  if (s.device.TryProvePendingSubmissionQuiescence(&error) || s.waits != 3 ||
      s.context->health.completed_serial != 0 ||
      s.context->health.health != VulkanContextHealth::device_lost)
    return false;
  s.next_wait = VK_SUCCESS;
  return !s.device.TryProvePendingSubmissionQuiescence(&error) &&
         s.waits == 3 && s.resets == 1 && s.submits == 1;
}

bool CheckExactSubmissionProofs() {
  std::string error;
  for (const auto mutation :
       {RetirementDriver::Mutation::identity, RetirementDriver::Mutation::fence,
        RetirementDriver::Mutation::serial}) {
    RetirementDriver s;
    if (!s.Timeout())
      return false;
    s.next_wait = VK_SUCCESS;
    s.mutation = mutation;
    if (s.device.TryProvePendingSubmissionQuiescence(&error) ||
        s.context->health.completed_serial != 0)
      return false;
  }
  for (const auto timeout : {UINT64_C(1000000), UINT64_C(200000000)}) {
    RetirementDriver s;
    if (!s.Timeout())
      return false;
    s.context->fence_timeout_ns = timeout;
    s.next_wait = VK_SUCCESS;
    if (!s.device.TryProvePendingSubmissionQuiescence(&error) ||
        s.timeout != std::min(timeout, VK_DEFAULT_FENCE_TIMEOUT_NS))
      return false;
  }
  for (const auto timeout : {UINT64_C(0), UINT64_MAX}) {
    RetirementDriver s;
    if (!s.Timeout())
      return false;
    s.context->fence_timeout_ns = timeout;
    s.next_wait = VK_SUCCESS;
    if (s.device.TryProvePendingSubmissionQuiescence(&error) || s.waits != 1)
      return false;
  }
  RetirementDriver s;
  if (!s.device.TryProvePendingSubmissionQuiescence(&error) || s.waits != 0)
    return false;
  s.context->health.health = VulkanContextHealth::unsafe_timeout;
  s.context->health.poisoned = true;
  if (s.device.TryProvePendingSubmissionQuiescence(&error) || s.waits != 0 ||
      s.device.SafeToDestroyResources())
    return false;
  s.context->health.health = VulkanContextHealth::device_lost;
  if (s.device.TryProvePendingSubmissionQuiescence(&error) || s.waits != 0)
    return false;
  s.context->health.health = VulkanContextHealth::healthy;
  s.context->health.poisoned = false;
  s.context->health.submitted_serial = UINT64_MAX;
  s.context->health.completed_serial = UINT64_MAX;
  return !s.device.SubmitAndWait(reinterpret_cast<VkCommandBuffer>(5),
                                 &error) &&
         s.resets == 0 && s.submits == 0;
}

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
  const bool late = CheckLateSubmissionProof();
  const bool failed = CheckFailedSubmissionProofs();
  const bool exact = CheckExactSubmissionProofs();
  if (!late || !failed || !exact)
    std::cerr << "Pending-submission retirement proof regression: late=" << late
              << " failed=" << failed << " exact=" << exact << '\n';
  return unique && retained && CheckAggregateReservations() && late && failed &&
                 exact
             ? 0
             : 1;
}
