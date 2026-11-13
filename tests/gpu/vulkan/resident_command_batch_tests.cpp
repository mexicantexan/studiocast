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

namespace {
using namespace studiocast::vulkan;
std::uint32_t submitted_commands = 0, submissions = 0, waits = 0;
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
  ResidentCommandGroup a{device.context_identity(), commands.data(), 1,
                         &first, 9, Observer::Notify};
  ResidentCommandGroup b{device.context_identity(), commands.data() + 1, 1,
                         &second, 10, Observer::Notify};
  ResidentCommandTicket ta, tb;
  bool ok = Require(!batch.Initialize(&device, 129, &error), "reject cap overflow") &&
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
                    second.outcomes[0] == ResidentCommandCompletion::aborted_unsubmitted,
                "rollback callback preserves exact ticket");
  const auto aborted_ticket = tb;
  ok &= Require(batch.Append(b, &tb, &error) && batch.Complete(&error), "complete frame");
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
                    first.outcomes[1] == ResidentCommandCompletion::unsafe_pending,
                "unsafe completion exact ticket");
  batch.AbortUnsubmitted();
  ok &= Require(batch.command_count() == 1 &&
                    !batch.BeginFrame(8, &error) &&
                    !batch.RetireAfterQuiescence(&error),
                "unsafe frame cannot abort, reuse or retire without proof");
  // Simulate a later explicit native retirement fence proving this exact
  // context quiescent, without restoring its failed-frame success status.
  context->health.health = VulkanContextHealth::fatal_submission_error;
  context->health.completed_serial = context->health.submitted_serial;
  ok &= Require(batch.RetireAfterQuiescence(&error), "retire after exact proof");
  ok &= Require(first.count == 3 && first.tickets[2] == ta &&
                    first.outcomes[2] == ResidentCommandCompletion::retired_after_quiescence &&
                    batch.completion_count() == 1 && batch.command_count() == 0,
                "retirement releases references without promoting success");
  ok &= Require(!batch.RetireAfterQuiescence(&error) && first.count == 3,
                "retirement cannot repeat callbacks");
  return ok;
}
} // namespace

int main() { return CheckCollector() ? 0 : 1; }
