#pragma once

#include "core/vulkan/vulkan_device.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace studiocast::vulkan {

inline constexpr std::size_t kMaxResidentFrameCommands = 128;
inline constexpr std::size_t kMaxResidentFrameParticipants = 128;

enum class ResidentCommandCompletion : std::uint8_t {
  completed,
  aborted_unsubmitted,
  unsafe_pending,
  // Retires a failed frame after explicit quiescence proof. This is never
  // successful inference/output evidence and must not advance success counts.
  retired_after_quiescence,
};

struct ResidentCommandTicket {
  std::uint64_t batch_id = 0;
  std::uint64_t epoch = 0;
  std::uint64_t capture_sequence = 0;
  std::uint64_t owner_generation = 0;
  // Monotonic within an epoch; rollback never reuses an issued ticket index.
  std::uint32_t participant_index = 0;
  friend bool operator==(const ResidentCommandTicket &,
                         const ResidentCommandTicket &) = default;
};

struct ResidentCommandGroup {
  VulkanContextIdentity context{};
  const VkCommandBuffer *commands = nullptr;
  std::uint32_t count = 0;
  // A persistent prepared Impl, retained transitively by the execution owner.
  // Neither this pointer nor command ownership may expire while pending.
  void *owner = nullptr;
  std::uint64_t owner_generation = 0;
  void (*on_completion)(void *, const ResidentCommandTicket &,
                        ResidentCommandCompletion) noexcept = nullptr;
};

struct ResidentCommandMark {
  std::uint64_t batch_id = 0;
  std::uint64_t epoch = 0;
  std::uint32_t command_count = 0;
  std::uint32_t participant_count = 0;
};

enum class ResidentCommandBatchState : std::uint8_t {
  uninitialized,
  idle,
  recording,
  completed,
  unsafe_pending,
};

// One synchronous frame, one exact context, one finite queue submission.
// Command buffers contain their producer/consumer barriers and immutable
// per-dispatch bindings. Append only queues references: it is not execution.
// Unsafe completion retains callbacks and forbids reuse until explicit
// RetireAfterQuiescence; the owning factory must retain all participant Impls.
class ResidentCommandBatch {
public:
  ResidentCommandBatch();
  ~ResidentCommandBatch();
  ResidentCommandBatch(const ResidentCommandBatch &) = delete;
  ResidentCommandBatch &operator=(const ResidentCommandBatch &) = delete;

  bool Initialize(VulkanDevice *device, std::size_t capacity,
                  std::string *error);
  bool BeginFrame(std::uint64_t capture_sequence, std::string *error);
  bool Append(const ResidentCommandGroup &group,
              ResidentCommandTicket *ticket, std::string *error);
  ResidentCommandMark Mark() const noexcept;
  bool Rollback(const ResidentCommandMark &mark, std::string *error);
  bool Complete(std::string *error);
  void AbortUnsubmitted() noexcept;
  // Verifies the original context and native quiescence. Failed-frame tickets
  // receive retired_after_quiescence exactly once, never completed.
  bool RetireAfterQuiescence(std::string *error);

  ResidentCommandBatchState state() const noexcept;
  VulkanContextIdentity context_identity() const noexcept;
  std::uint64_t capture_sequence() const noexcept;
  std::size_t command_count() const noexcept;
  std::uint64_t completion_count() const noexcept;

private:
  friend struct ResidentCommandBatchTestAccess;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::vulkan
