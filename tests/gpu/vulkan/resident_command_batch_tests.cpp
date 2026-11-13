#include "core/vulkan/kernels/utility_kernels.h"
#include "core/vulkan/resident_command_batch.h"

#include <array>
#include <iostream>

namespace studiocast::vulkan::detail {
struct VulkanDeviceTestAccess {
  static std::shared_ptr<VulkanContextState> Attach(VulkanDevice &device) {
    auto state = std::make_shared<VulkanContextState>();
    state->loader = std::make_shared<VulkanLoader>();
    state->identity = {17, 23};
    state->health.health = VulkanContextHealth::healthy;
    device.context_ = state;
    device.loader_ = state->loader;
    device.context_identity_ = state->identity;
    device.initialized_ = true;
    return state;
  }
};
} // namespace studiocast::vulkan::detail

namespace studiocast::vulkan::kernels {
struct UtilityKernelsTestAccess {
  static void Prepare(UtilityKernels &kernels) {
    kernels.initialized_ = true;
    kernels.deferred_capacity_ = 2;
    for (std::size_t i = 0; i < 2; ++i) {
      auto &slot = kernels.deferred_slots_[i];
      slot.command = reinterpret_cast<VkCommandBuffer>(i + 3);
      slot.descriptors = reinterpret_cast<VkDescriptorSet>(i + 10);
    }
  }
  static bool Queue(UtilityKernels &kernels, std::string *error) {
    return kernels.BindBuffers(reinterpret_cast<VkBuffer>(7), nullptr, nullptr,
                               reinterpret_cast<VkBuffer>(8), nullptr, nullptr,
                               error) &&
           kernels.SubmitRecorded(error);
  }
  static void RejectFirstBinding(UtilityKernels &kernels) {
    kernels.deferred_slots_[0].descriptors = nullptr;
  }
};
} // namespace studiocast::vulkan::kernels

namespace {
using namespace studiocast::vulkan;
std::uint32_t submitted_commands = 0, submissions = 0, waits = 0;
std::array<VkDescriptorSet, 4> updated_sets{};
std::size_t updates = 0;
void UpdateDescriptors(VkDevice, std::uint32_t count,
                       const VkWriteDescriptorSet *writes, std::uint32_t,
                       const void *) {
  if (count && updates < updated_sets.size())
    updated_sets[updates++] = writes[0].dstSet;
}
VkResult Reset(VkDevice, std::uint32_t, const VkFence *) { return VK_SUCCESS; }
VkResult Submit(VkQueue, std::uint32_t count, const VkSubmitInfo *info,
                VkFence) {
  ++submissions;
  if (count != 1)
    return VK_ERROR_INITIALIZATION_FAILED;
  submitted_commands = info->commandBufferCount;
  return VK_SUCCESS;
}
VkResult Wait(VkDevice, std::uint32_t, const VkFence *, VkBool32,
              std::uint64_t) {
  ++waits;
  return VK_SUCCESS;
}
struct Observer {
  std::array<ResidentCommandTicket, 8> tickets{};
  std::array<ResidentCommandCompletion, 8> outcomes{};
  std::size_t count = 0;
  static void Notify(void *owner, const ResidentCommandTicket &ticket,
                     ResidentCommandCompletion outcome) noexcept {
    auto &self = *static_cast<Observer *>(owner);
    if (self.count < self.tickets.size()) {
      self.tickets[self.count] = ticket;
      self.outcomes[self.count++] = outcome;
    }
  }
};
bool Require(bool ok, const char *message) {
  if (!ok)
    std::cerr << "FAIL: " << message << '\n';
  return ok;
}
bool CheckCollector() {
  VulkanDevice device;
  const auto context = detail::VulkanDeviceTestAccess::Attach(device);
  auto &functions = context->loader->f();
  functions.vkResetFences = Reset;
  functions.vkQueueSubmit = Submit;
  functions.vkWaitForFences = Wait;
  ResidentCommandBatch batch;
  std::string error;
  Observer first, second;
  const std::array<VkCommandBuffer, 2> commands{
      reinterpret_cast<VkCommandBuffer>(1),
      reinterpret_cast<VkCommandBuffer>(2)};
  ResidentCommandGroup a{
      device.context_identity(), commands.data(), 1, &first, 9,
      Observer::Notify};
  ResidentCommandGroup b{
      device.context_identity(), commands.data() + 1, 1, &second, 10,
      Observer::Notify};
  ResidentCommandTicket ta, tb;
  bool ok =
      Require(!batch.Initialize(&device, 129, &error), "reject cap overflow") &&
      Require(batch.Initialize(&device, 2, &error), "initialize") &&
      Require(batch.BeginFrame(3, &error), "begin") &&
      Require(batch.Append(a, &ta, &error), "append first") &&
      Require(first.count == 0, "append is not execution");
  const auto mark = batch.Mark();
  auto foreign = b;
  ++foreign.context.generation;
  ok &= Require(!batch.Append(foreign, &tb, &error), "reject foreign context");
  ok &= Require(!batch.Append(a, &tb, &error), "reject pending command reuse");
  ok &= Require(batch.Append(b, &tb, &error), "append second");
  auto forged = mark;
  forged.command_count = 0;
  ok &= Require(!batch.Rollback(forged, &error), "reject partial group mark");
  ok &= Require(batch.Rollback(mark, &error), "rollback unsubmitted group");
  ok &= Require(second.count == 1 && second.tickets[0] == tb &&
                    second.outcomes[0] ==
                        ResidentCommandCompletion::aborted_unsubmitted,
                "rollback callback preserves exact ticket");
  const auto aborted_ticket = tb;
  ok &= Require(batch.Append(b, &tb, &error) && batch.Complete(&error),
                "complete frame");
  ok &= Require(tb != aborted_ticket, "rollback never reuses a stale ticket");
  ok &= Require(submissions == 1 && waits == 1 && submitted_commands == 2 &&
                    batch.completion_count() == 1,
                "two commands share one submission/wait");
  ok &= Require(first.count == 1 && first.tickets[0] == ta &&
                    first.outcomes[0] == ResidentCommandCompletion::completed,
                "successful completion exact once");
  ok &= Require(!batch.Complete(&error) && first.count == 1,
                "duplicate completion rejected");
  ok &= Require(batch.BeginFrame(7, &error), "begin new epoch");
  ok &= Require(!batch.Rollback(mark, &error), "reject old epoch rollback");
  ok &= Require(batch.Append(a, &ta, &error), "append timeout frame");
  device.InjectNextSubmissionResultForTesting(
      VulkanSubmissionPhase::wait_for_fence, VK_TIMEOUT);
  ok &= Require(!batch.Complete(&error) &&
                    batch.state() == ResidentCommandBatchState::unsafe_pending,
                "timeout retains unsafe ledger");
  ok &= Require(first.count == 2 && first.tickets[1] == ta &&
                    first.outcomes[1] ==
                        ResidentCommandCompletion::unsafe_pending,
                "unsafe completion exact ticket");
  batch.AbortUnsubmitted();
  ok &= Require(batch.command_count() == 1 && !batch.BeginFrame(8, &error) &&
                    !batch.RetireAfterQuiescence(&error),
                "unsafe frame cannot abort, reuse or retire without proof");
  // Simulate a later explicit native retirement fence proving this exact
  // context quiescent, without restoring its failed-frame success status.
  context->health.health = VulkanContextHealth::fatal_submission_error;
  context->health.completed_serial = context->health.submitted_serial;
  ok &=
      Require(batch.RetireAfterQuiescence(&error), "retire after exact proof");
  ok &= Require(first.count == 3 && first.tickets[2] == ta &&
                    first.outcomes[2] ==
                        ResidentCommandCompletion::retired_after_quiescence &&
                    batch.completion_count() == 1 && batch.command_count() == 0,
                "retirement releases references without promoting success");
  ok &= Require(!batch.RetireAfterQuiescence(&error) && first.count == 3,
                "retirement cannot repeat callbacks");
  return ok;
}

bool CheckDeferredSlots() {
  kernels::UtilityKernels utility;
  auto *device = utility.device();
  const auto context = detail::VulkanDeviceTestAccess::Attach(*device);
  auto &functions = context->loader->f();
  functions.vkResetFences = Reset;
  functions.vkQueueSubmit = Submit;
  functions.vkWaitForFences = Wait;
  functions.vkUpdateDescriptorSets = UpdateDescriptors;
  kernels::UtilityKernelsTestAccess::Prepare(utility);
  ResidentCommandBatch batch;
  std::string error;
  bool ok = Require(batch.Initialize(device, 2, &error) &&
                        batch.BeginFrame(1, &error),
                    "begin deferred utility slots");
  bool foreign_begin = true;
  std::thread foreign([&] {
    std::string own_error;
    foreign_begin = utility.BeginDeferredFrame(batch, &own_error);
  });
  foreign.join();
  ok &= Require(!foreign_begin && !utility.deferred_frame_active(),
                "foreign recording thread cannot attach or mutate slots");
  ok &= Require(utility.BeginDeferredFrame(batch, &error),
                "attach recording owner");
  const auto prior_submissions = submissions;
  ok &= Require(kernels::UtilityKernelsTestAccess::Queue(utility, &error) &&
                    kernels::UtilityKernelsTestAccess::Queue(utility, &error),
                "queue independent descriptor slots");
  ok &= Require(submissions == prior_submissions && updates == 2 &&
                    updated_sets[0] != updated_sets[1],
                "deferred descriptors are distinct and no work submitted");
  ok &= Require(!kernels::UtilityKernelsTestAccess::Queue(utility, &error),
                "reject descriptor/parameter slot reuse within epoch");
  utility.EndDeferredFrame();
  ok &= Require(utility.deferred_frame_active(), "cannot detach queued epoch");
  ok &= Require(batch.Complete(&error), "complete deferred slots once");
  utility.EndDeferredFrame();
  ok &= Require(!utility.deferred_frame_active(),
                "detach proven completed epoch");
  ok &= Require(batch.BeginFrame(2, &error) &&
                    utility.BeginDeferredFrame(batch, &error) &&
                    kernels::UtilityKernelsTestAccess::Queue(utility, &error) &&
                    kernels::UtilityKernelsTestAccess::Queue(utility, &error) &&
                    updates == 2,
                "completed stable descriptors are reused without updates");
  batch.AbortUnsubmitted();
  utility.EndDeferredFrame();
  ok &= Require(!utility.deferred_frame_active(),
                "unsubmitted abort retires slots");
  ok &= Require(batch.BeginFrame(3, &error) &&
                    utility.BeginDeferredFrame(batch, &error),
                "begin pre-submit failure fixture");
  kernels::UtilityKernelsTestAccess::RejectFirstBinding(utility);
  ok &= Require(!kernels::UtilityKernelsTestAccess::Queue(utility, &error) &&
                    batch.command_count() == 0,
                "binding failure queues no GPU work");
  ok &= Require(kernels::UtilityKernelsTestAccess::Queue(utility, &error) &&
                    batch.command_count() == 1,
                "unrelated later stage survives unsubmitted binding failure");
  device->InjectNextSubmissionResultForTesting(
      VulkanSubmissionPhase::wait_for_fence, VK_TIMEOUT);
  ok &= Require(!batch.Complete(&error), "deferred utility timeout");
  utility.EndDeferredFrame();
  utility.Shutdown();
  ok &= Require(utility.deferred_frame_active() && utility.Initialized(),
                "unsafe completion retains utility callback owners and slots");
  context->health.health = VulkanContextHealth::fatal_submission_error;
  context->health.completed_serial = context->health.submitted_serial;
  ok &= Require(batch.RetireAfterQuiescence(&error),
                "retire unsafe utility slots");
  utility.EndDeferredFrame();
  return ok;
}
} // namespace

int main() { return CheckCollector() && CheckDeferredSlots() ? 0 : 1; }
