#pragma once

#include "core/cuda/cuda_driver_api.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace studiocast::cuda {

// One setup-owned, single-in-flight CUDA transfer island. It intentionally
// provides completion-shaped synchronous execution without scheduling work.
class CudaPreparedTransferCompletion {
public:
  static constexpr std::uint32_t kApiVersion = 1;
  static constexpr std::size_t kMaxTimedStages = 16;
  static constexpr std::uint32_t kNoStage =
      std::numeric_limits<std::uint32_t>::max();

  enum class Status : std::uint8_t {
    ok = 0,
    unsupported = 1,
    not_prepared = 2,
    invalid_argument = 3,
    invalid_context = 4,
    stale_context = 5,
    foreign_context = 6,
    foreign_owner = 7,
    wrong_stream = 8,
    invalid_sequence = 9,
    resource_exhausted = 10,
    device_lost = 11,
    copy_failed = 12,
    event_failed = 13,
    timeout = 14,
    stop_requested = 15,
    cancelled = 16,
    shutdown = 17,
    already_prepared = 18,
    resource_release_failed = 19,
  };

  struct Result {
    Status status = Status::ok;
    CUresult cuda_result = CUDA_SUCCESS;
    std::uint32_t stage_index = kNoStage;
    std::uint32_t api_version = kApiVersion;

    [[nodiscard]] bool ok() const noexcept { return status == Status::ok; }
  };

  struct Binding {
    CudaContextIdentity context{};
    CUstream stream = nullptr;
    std::uint64_t stream_identity = 0;
    std::uint64_t owner_identity = 0;
    std::uint64_t configuration_generation = 0;
  };

  struct ResourceBudget {
    std::size_t max_pinned_host_bytes = 0;
    std::uint32_t max_events = 0;
  };

  struct PrepareOptions {
    std::uint32_t api_version = kApiVersion;
    Binding binding{};
    std::size_t upload_row_bytes = 0;
    std::size_t upload_rows = 0;
    std::size_t readback_row_bytes = 0;
    std::size_t readback_rows = 0;
    std::uint32_t timed_stage_count = 0;
    ResourceBudget budget{};
  };

  using Predicate = bool (*)(void *context) noexcept;
  using NowNanoseconds = std::uint64_t (*)(void *context) noexcept;
  using PollWait = void (*)(void *context,
                            std::uint64_t wait_nanoseconds) noexcept;

  struct WaitPolicy {
    std::uint64_t timeout_nanoseconds = 0;
    std::uint64_t poll_interval_nanoseconds = 0;
    std::uint32_t maximum_poll_count = 0;
    Predicate stop_requested = nullptr;
    Predicate cancel_requested = nullptr;
    NowNanoseconds now_nanoseconds = nullptr;
    PollWait poll_wait = nullptr;
    void *callback_context = nullptr;
  };

  struct Counters {
    std::uint64_t setup_pinned_allocations = 0;
    std::uint64_t setup_pinned_frees = 0;
    std::uint64_t setup_event_creations = 0;
    std::uint64_t setup_event_destructions = 0;
    std::uint64_t frame_uploads = 0;
    std::uint64_t frame_final_readbacks = 0;
    std::uint64_t frame_waits = 0;
    std::uint64_t event_records = 0;
    std::uint64_t event_queries = 0;
    std::uint64_t elapsed_time_queries = 0;
    std::uint64_t device_losses = 0;
    // These remain explicit zero evidence: runtime allocation and capability
    // discovery are not operations exposed by this class.
    std::uint64_t frame_allocations = 0;
    std::uint64_t frame_capability_probes = 0;
  };

  CudaPreparedTransferCompletion() = default;
  ~CudaPreparedTransferCompletion();

  CudaPreparedTransferCompletion(const CudaPreparedTransferCompletion &) =
      delete;
  CudaPreparedTransferCompletion &
  operator=(const CudaPreparedTransferCompletion &) = delete;
  CudaPreparedTransferCompletion(CudaPreparedTransferCompletion &&) = delete;
  CudaPreparedTransferCompletion &
  operator=(CudaPreparedTransferCompletion &&) = delete;

  // One object owns one prepared generation. Reconfiguration must first build
  // a fresh candidate object, then route execution to it and shut down the old
  // object. This makes old-plan retention unconditional on candidate failure.
  Result Prepare(CudaDriverApi *cuda, const PrepareOptions &options) noexcept;
  Result BeginFrame(const Binding &binding,
                    std::uint64_t capture_sequence) noexcept;
  Result EnqueueUpload(const Binding &binding, CUdeviceptr destination,
                       std::size_t destination_pitch) noexcept;
  Result RecordStageBegin(const Binding &binding,
                          std::uint32_t stage_index) noexcept;
  Result RecordStageEnd(const Binding &binding,
                        std::uint32_t stage_index) noexcept;
  Result EnqueueReadback(const Binding &binding, CUdeviceptr source,
                         std::size_t source_pitch) noexcept;
  Result RecordCompletion(const Binding &binding) noexcept;
  Result WaitForCompletion(const Binding &binding,
                           const WaitPolicy &policy) noexcept;
  // Shutdown never waits. If DMA is still outstanding it returns
  // resource_exhausted and retains all resources; the owner must keep this
  // object alive and retry after completion or CUDA-context teardown.
  Result Shutdown() noexcept;

  [[nodiscard]] bool prepared() const noexcept { return prepared_; }
  [[nodiscard]] bool quarantined() const noexcept { return quarantined_; }
  [[nodiscard]] bool frame_reuse_blocked() const noexcept {
    return frame_reuse_blocked_;
  }
  [[nodiscard]] void *upload_data() noexcept { return upload_host_; }
  [[nodiscard]] const void *upload_data() const noexcept {
    return upload_host_;
  }
  [[nodiscard]] void *readback_data() noexcept { return readback_host_; }
  [[nodiscard]] const void *readback_data() const noexcept {
    return readback_host_;
  }
  [[nodiscard]] std::size_t upload_size_bytes() const noexcept {
    return upload_size_bytes_;
  }
  [[nodiscard]] std::size_t readback_size_bytes() const noexcept {
    return readback_size_bytes_;
  }
  [[nodiscard]] std::uint32_t timed_stage_count() const noexcept {
    return timed_stage_count_;
  }
  [[nodiscard]] const ResourceBudget &resource_budget() const noexcept {
    return resource_budget_;
  }
  [[nodiscard]] std::uint64_t
  stage_elapsed_nanoseconds(std::uint32_t stage_index) const noexcept;
  [[nodiscard]] const Counters &counters() const noexcept { return counters_; }

private:
  struct Candidate {
    void *upload_host = nullptr;
    void *readback_host = nullptr;
    std::array<CUevent, kMaxTimedStages> stage_begin{};
    std::array<CUevent, kMaxTimedStages> stage_end{};
    CUevent completion = nullptr;
    std::uint32_t stage_count = 0;
  };

  [[nodiscard]] Result ValidateBinding(const Binding &binding) noexcept;
  [[nodiscard]] Result ValidateFrameBinding(const Binding &binding) noexcept;
  [[nodiscard]] Result BindContext() noexcept;
  [[nodiscard]] Result FromCudaFailure(Status fallback,
                                       CUresult result) noexcept;
  Result ReleaseCandidate(Candidate *candidate) noexcept;
  Result ReleaseCurrent(bool mark_shutdown) noexcept;
  void ResetFrameState() noexcept;

  CudaDriverApi *cuda_ = nullptr;
  Binding binding_{};
  void *upload_host_ = nullptr;
  void *readback_host_ = nullptr;
  std::size_t upload_row_bytes_ = 0;
  std::size_t upload_rows_ = 0;
  std::size_t upload_size_bytes_ = 0;
  std::size_t readback_row_bytes_ = 0;
  std::size_t readback_rows_ = 0;
  std::size_t readback_size_bytes_ = 0;
  ResourceBudget resource_budget_{};
  std::array<CUevent, kMaxTimedStages> stage_begin_events_{};
  std::array<CUevent, kMaxTimedStages> stage_end_events_{};
  CUevent completion_event_ = nullptr;
  std::array<std::uint64_t, kMaxTimedStages> stage_elapsed_ns_{};
  std::array<bool, kMaxTimedStages> stage_begin_recorded_{};
  std::array<bool, kMaxTimedStages> stage_end_recorded_{};
  std::uint32_t timed_stage_count_ = 0;
  std::uint32_t next_stage_index_ = 0;
  std::uint64_t capture_sequence_ = 0;
  bool prepared_ = false;
  bool shutdown_ = false;
  bool quarantined_ = false;
  bool frame_active_ = false;
  bool frame_reuse_blocked_ = false;
  bool upload_enqueued_ = false;
  bool readback_enqueued_ = false;
  bool completion_recorded_ = false;
  bool wait_attempted_ = false;
  Counters counters_{};
};

} // namespace studiocast::cuda
