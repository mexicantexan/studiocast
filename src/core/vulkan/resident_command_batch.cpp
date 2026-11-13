#include "core/vulkan/resident_command_batch.h"

#include <array>
#include <atomic>
#include <thread>

namespace studiocast::vulkan {
namespace {
bool Fail(std::string *error, const char *detail) {
  if (error)
    *error = detail;
  return false;
}
std::atomic<std::uint64_t> next_batch_id{1};
} // namespace

struct ResidentCommandBatch::Impl {
  struct Participant {
    ResidentCommandTicket ticket{};
    std::uint32_t command_end = 0;
    void *owner = nullptr;
    decltype(ResidentCommandGroup::on_completion) notify = nullptr;
  };
  VulkanDevice *device = nullptr;
  std::shared_ptr<void> context_owner;
  VulkanContextIdentity context{};
  std::array<VkCommandBuffer, kMaxResidentFrameCommands> commands{};
  std::array<Participant, kMaxResidentFrameParticipants> participants{};
  std::size_t capacity = 0;
  std::uint32_t count = 0, participant_count = 0, next_ticket_index = 0;
  std::uint64_t id = 0, epoch = 0, sequence = 0, completions = 0;
  ResidentCommandBatchState state = ResidentCommandBatchState::uninitialized;
  std::thread::id recording_owner{};
  bool notifying = false;

  bool Owned() const noexcept {
    return !notifying && recording_owner == std::this_thread::get_id();
  }
  bool ExactContext() const noexcept {
    return device && device->Initialized() && device->OwnsContext(context) &&
           device->RetainContext().get() == context_owner.get();
  }
  void Notify(std::uint32_t first, ResidentCommandCompletion completion) {
    notifying = true;
    for (std::uint32_t i = first; i < participant_count; ++i) {
      const auto &p = participants[i];
      if (p.notify)
        p.notify(p.owner, p.ticket, completion);
    }
    notifying = false;
  }
  void Clear() noexcept {
    commands.fill(nullptr);
    participants.fill({});
    count = participant_count = 0;
  }
};

ResidentCommandBatch::ResidentCommandBatch() : impl_(std::make_unique<Impl>()) {}
ResidentCommandBatch::~ResidentCommandBatch() {
  AbortUnsubmitted();
  if (impl_->state == ResidentCommandBatchState::unsafe_pending) {
    // The factory also retains participant Impls. Never destroy the exact
    // context/callback ledger merely because its outer wrapper is destroyed.
    (void)impl_.release();
  }
}

bool ResidentCommandBatch::Initialize(VulkanDevice *device,
                                     std::size_t capacity,
                                     std::string *error) {
  if (error)
    error->clear();
  auto &s = *impl_;
  if (s.state != ResidentCommandBatchState::uninitialized || !device ||
      !device->Initialized() || !device->context_identity().Valid() ||
      !device->health_state().healthy() || capacity == 0 ||
      capacity > kMaxResidentFrameCommands)
    return Fail(error, "Invalid resident command collector setup.");
  s.device = device;
  s.context_owner = device->RetainContext();
  s.context = device->context_identity();
  if (!s.context_owner)
    return Fail(error, "Resident command collector has no context owner.");
  s.capacity = capacity;
  s.id = next_batch_id.fetch_add(1, std::memory_order_relaxed);
  if (!s.id)
    s.id = next_batch_id.fetch_add(1, std::memory_order_relaxed);
  s.state = ResidentCommandBatchState::idle;
  return true;
}

bool ResidentCommandBatch::BeginFrame(std::uint64_t sequence,
                                     std::string *error) {
  if (error)
    error->clear();
  auto &s = *impl_;
  if (s.notifying || !sequence ||
      (s.state != ResidentCommandBatchState::idle &&
       s.state != ResidentCommandBatchState::completed) ||
      !s.ExactContext() || !s.device->health_state().healthy() ||
      s.epoch == UINT64_MAX)
    return Fail(error, "Resident command collector cannot begin this frame.");
  s.Clear();
  ++s.epoch;
  s.sequence = sequence;
  s.next_ticket_index = 0;
  s.recording_owner = std::this_thread::get_id();
  s.state = ResidentCommandBatchState::recording;
  return true;
}

bool ResidentCommandBatch::Append(const ResidentCommandGroup &group,
                                 ResidentCommandTicket *ticket,
                                 std::string *error) {
  if (ticket)
    *ticket = {};
  if (error)
    error->clear();
  auto &s = *impl_;
  if (!ticket || s.state != ResidentCommandBatchState::recording ||
      !s.Owned() || !s.ExactContext() || group.context != s.context ||
      !group.commands || !group.count || !group.owner_generation ||
      (group.on_completion && !group.owner) ||
      group.count > s.capacity - s.count ||
      s.participant_count == s.participants.size() ||
      s.next_ticket_index == UINT32_MAX)
    return Fail(error, "Resident command group is stale, foreign or over budget.");
  for (std::uint32_t i = 0; i < group.count; ++i) {
    if (!group.commands[i])
      return Fail(error, "Resident command group contains a null command.");
    // One primary buffer cannot be pending twice without SIMULTANEOUS_USE.
    for (std::uint32_t j = 0; j < s.count; ++j)
      if (group.commands[i] == s.commands[j])
        return Fail(error, "Resident command buffer is already queued.");
    for (std::uint32_t j = 0; j < i; ++j)
      if (group.commands[i] == group.commands[j])
        return Fail(error, "Resident command group repeats a command buffer.");
  }
  auto &p = s.participants[s.participant_count];
  p.ticket = {s.id, s.epoch, s.sequence, group.owner_generation,
              s.next_ticket_index++};
  p.owner = group.owner;
  p.notify = group.on_completion;
  for (std::uint32_t i = 0; i < group.count; ++i)
    s.commands[s.count++] = group.commands[i];
  p.command_end = s.count;
  ++s.participant_count;
  *ticket = p.ticket;
  return true;
}

ResidentCommandMark ResidentCommandBatch::Mark() const noexcept {
  const auto &s = *impl_;
  if (s.state != ResidentCommandBatchState::recording || !s.Owned())
    return {};
  return {s.id, s.epoch, s.count, s.participant_count};
}

bool ResidentCommandBatch::Rollback(const ResidentCommandMark &mark,
                                   std::string *error) {
  if (error)
    error->clear();
  auto &s = *impl_;
  if (s.state != ResidentCommandBatchState::recording || !s.Owned() ||
      mark.batch_id != s.id || mark.epoch != s.epoch ||
      mark.command_count > s.count ||
      mark.participant_count > s.participant_count ||
      (mark.participant_count == 0
           ? mark.command_count != 0
           : s.participants[mark.participant_count - 1].command_end !=
                 mark.command_count))
    return Fail(error, "Resident command rollback mark is invalid or stale.");
  s.Notify(mark.participant_count,
           ResidentCommandCompletion::aborted_unsubmitted);
  while (s.count > mark.command_count)
    s.commands[--s.count] = nullptr;
  while (s.participant_count > mark.participant_count)
    s.participants[--s.participant_count] = {};
  return true;
}

bool ResidentCommandBatch::Complete(std::string *error) {
  if (error)
    error->clear();
  auto &s = *impl_;
  if (s.state != ResidentCommandBatchState::recording || !s.Owned() ||
      !s.ExactContext() || !s.count)
    return Fail(error, "Resident command collector cannot complete this frame.");
  const bool completed = s.device->SubmitAndWait(s.commands.data(), s.count,
                                                 error);
  s.state = completed ? ResidentCommandBatchState::completed
                      : ResidentCommandBatchState::unsafe_pending;
  if (completed)
    ++s.completions;
  s.Notify(0, completed ? ResidentCommandCompletion::completed
                       : ResidentCommandCompletion::unsafe_pending);
  if (completed)
    s.Clear();
  return completed;
}

void ResidentCommandBatch::AbortUnsubmitted() noexcept {
  auto &s = *impl_;
  if (s.state != ResidentCommandBatchState::recording || !s.Owned())
    return;
  s.Notify(0, ResidentCommandCompletion::aborted_unsubmitted);
  s.Clear();
  s.state = ResidentCommandBatchState::idle;
}

bool ResidentCommandBatch::RetireAfterQuiescence(std::string *error) {
  if (error)
    error->clear();
  auto &s = *impl_;
  if (s.state != ResidentCommandBatchState::unsafe_pending || !s.Owned() ||
      !s.ExactContext() || !s.device->SafeToDestroyResources())
    return Fail(error, "Failed resident frame has no exact quiescence proof.");
  const auto health = s.device->health_state();
  if (health.completed_serial < health.submitted_serial)
    return Fail(error, "Failed resident frame still has incomplete submissions.");
  s.Notify(0, ResidentCommandCompletion::retired_after_quiescence);
  s.Clear();
  s.state = ResidentCommandBatchState::idle;
  return true;
}

ResidentCommandBatchState ResidentCommandBatch::state() const noexcept {
  return impl_->state;
}
VulkanContextIdentity ResidentCommandBatch::context_identity() const noexcept {
  return impl_->context;
}
std::uint64_t ResidentCommandBatch::capture_sequence() const noexcept {
  return impl_->sequence;
}
std::size_t ResidentCommandBatch::command_count() const noexcept {
  return impl_->count;
}
std::uint64_t ResidentCommandBatch::completion_count() const noexcept {
  return impl_->completions;
}
} // namespace studiocast::vulkan
