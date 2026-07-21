#include "core/video/v4l2_writer.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using namespace studiocast::video;
using Clock = std::chrono::steady_clock;

bool Expect(bool condition, const char *message) {
  if (!condition)
    std::cerr << message << '\n';
  return condition;
}

struct ScriptStep {
  ssize_t result = -1;
  int error_number = 0;
};

struct ScriptedWrite {
  std::vector<ScriptStep> steps;
  std::size_t call = 0;
  std::atomic_bool *stop_after_call = nullptr;

  static ssize_t Invoke(void *context, int, const void *, std::size_t) {
    auto &self = *static_cast<ScriptedWrite *>(context);
    const std::size_t index = self.call++;
    if (self.stop_after_call)
      self.stop_after_call->store(true, std::memory_order_relaxed);
    if (index >= self.steps.size()) {
      errno = EIO;
      return -1;
    }
    errno = self.steps[index].error_number;
    return self.steps[index].result;
  }

  FrameWriteTransport Transport() {
    return FrameWriteTransport{this, &ScriptedWrite::Invoke};
  }
};

bool FillPipe(int fd) {
  std::array<std::uint8_t, 4096> bytes{};
  for (;;) {
    const ssize_t wrote = ::write(fd, bytes.data(), bytes.size());
    if (wrote > 0)
      continue;
    return wrote < 0 && (errno == EAGAIN || errno == EWOULDBLOCK);
  }
}

bool TestOpenFlagsDeclareNonblockingTransport() {
  const int write_only = V4l2WriterOpenFlags(false);
  const int read_write = V4l2WriterOpenFlags(true);
  return Expect((write_only & O_ACCMODE) == O_WRONLY,
                "writer must prefer write-only access") &&
         Expect((read_write & O_ACCMODE) == O_RDWR,
                "fallback writer must use read-write access") &&
         Expect((write_only & O_NONBLOCK) != 0 &&
                    (read_write & O_NONBLOCK) != 0,
                "all writer opens must set O_NONBLOCK") &&
         Expect((write_only & O_CLOEXEC) != 0 &&
                    (read_write & O_CLOEXEC) != 0,
                "all writer opens must retain O_CLOEXEC");
}

bool TestSuccessfulWriteIsOneSyscall() {
  ScriptedWrite script{{ScriptStep{64, 0}}};
  std::array<std::uint8_t, 64> frame{};
  FrameWriteCounters counters;
  const auto result = WriteFrameToTransport(
      script.Transport(), 7, frame.data(), frame.size(), frame.size(), nullptr,
      &counters);
  return Expect(result.FrameCommitted(), "full frame must commit") &&
         Expect(result.write_syscalls == 1 && script.call == 1,
                "full frame must use exactly one write syscall") &&
         Expect(counters.write_syscalls == 1 && counters.frames_written == 1,
                "success counters must reflect the actual write site") &&
         Expect(kV4l2WriterMaxQueuedFrames == 0,
                "writer contract must not introduce a userspace queue");
}

bool TestPartialAndZeroWritesAreFatalAndRequireReopen() {
  std::array<std::uint8_t, 64> frame{};
  ScriptedWrite partial{{ScriptStep{12, 0}, ScriptStep{52, 0}}};
  const auto partial_result = WriteFrameToTransport(
      partial.Transport(), 7, frame.data(), frame.size(), frame.size());
  ScriptedWrite zero{{ScriptStep{0, 0}, ScriptStep{64, 0}}};
  const auto zero_result = WriteFrameToTransport(
      zero.Transport(), 7, frame.data(), frame.size(), frame.size());
  return Expect(partial_result.status == FrameWriteStatus::fatal &&
                    partial_result.failure == FrameWriteFailure::partial_frame,
                "partial frame must be fatal") &&
         Expect(partial_result.RequiresReopen() && partial.call == 1,
                "partial frame must never write a remainder") &&
         Expect(zero_result.status == FrameWriteStatus::fatal &&
                    zero_result.failure == FrameWriteFailure::zero_write,
                "zero-byte write must be fatal") &&
         Expect(zero_result.RequiresReopen() && zero.call == 1,
                "zero-byte write must require reopening before reuse");
}

bool TestEintrIsBoundedAndStopAware() {
  std::array<std::uint8_t, 64> frame{};
  ScriptedWrite recovers{{ScriptStep{-1, EINTR}, ScriptStep{64, 0}}};
  FrameWriteCounters counters;
  const auto recovered = WriteFrameToTransport(
      recovers.Transport(), 7, frame.data(), frame.size(), frame.size(),
      nullptr, &counters);

  ScriptedWrite bounded;
  bounded.steps.assign(kV4l2WriterMaxEintrRetries + 1,
                       ScriptStep{-1, EINTR});
  const auto limited = WriteFrameToTransport(
      bounded.Transport(), 7, frame.data(), frame.size(), frame.size());

  std::atomic_bool stop{false};
  ScriptedWrite stopped{{ScriptStep{-1, EINTR}}, 0, &stop};
  const auto stop_result = WriteFrameToTransport(
      stopped.Transport(), 7, frame.data(), frame.size(), frame.size(), &stop);

  return Expect(recovered.FrameCommitted() && recovers.call == 2,
                "one EINTR must retry and then commit") &&
         Expect(counters.eintr_retries == 1 && counters.write_syscalls == 2,
                "EINTR counters must match actual attempts") &&
         Expect(limited.failure ==
                    FrameWriteFailure::interrupted_retry_limit &&
                    bounded.call == kV4l2WriterMaxEintrRetries + 1,
                "perpetual EINTR must stop at the declared attempt bound") &&
         Expect(stop_result.status == FrameWriteStatus::stopped &&
                    stopped.call == 1,
                "stop must be observed between interrupted attempts");
}

bool TestTypedFailureAndDropPolicy() {
  std::array<std::uint8_t, 64> frame{};
  ScriptedWrite would_block{{ScriptStep{-1, EAGAIN}}};
  const auto dropped = WriteFrameToTransport(
      would_block.Transport(), 7, frame.data(), frame.size(), frame.size());
  ScriptedWrite format{{ScriptStep{-1, EINVAL}}};
  const auto mismatch = WriteFrameToTransport(
      format.Transport(), 7, frame.data(), frame.size(), frame.size());
  ScriptedWrite disconnected{{ScriptStep{-1, ENODEV}}};
  const auto gone = WriteFrameToTransport(
      disconnected.Transport(), 7, frame.data(), frame.size(), frame.size());

  return Expect(dropped.status == FrameWriteStatus::would_block_dropped &&
                    dropped.failure == FrameWriteFailure::none,
                "EAGAIN must be a typed latest-frame drop") &&
         Expect(!dropped.ShouldRefreshFormat(),
                "EAGAIN must not trigger renegotiation") &&
         Expect(mismatch.status == FrameWriteStatus::fatal &&
                    mismatch.ShouldRefreshFormat(),
                "EINVAL must be the declared format-refresh trigger") &&
         Expect(gone.failure == FrameWriteFailure::disconnected &&
                    !gone.ShouldRefreshFormat(),
                "disconnect must remain fatal and visible");
}

bool TestPreflightAndPresetStopAvoidSyscalls() {
  std::array<std::uint8_t, 64> frame{};
  ScriptedWrite script{{ScriptStep{64, 0}}};
  std::atomic_bool stop{true};
  const auto stopped = WriteFrameToTransport(
      script.Transport(), 7, frame.data(), frame.size(), frame.size(), &stop);
  const auto undersized = WriteFrameToTransport(
      script.Transport(), 7, frame.data(), frame.size() - 1, frame.size());
  return Expect(stopped.status == FrameWriteStatus::stopped && script.call == 0,
                "preset stop must avoid entering write") &&
         Expect(undersized.failure == FrameWriteFailure::undersized_input &&
                    script.call == 0,
                "undersized buffers must fail before entering write");
}

bool TestFullNonblockingPipeReturnsWithinBound() {
  int fds[2] = {-1, -1};
  if (!Expect(::pipe2(fds, O_CLOEXEC | O_NONBLOCK) == 0,
              "pipe2 failed for nonblocking transport test"))
    return false;
  const bool filled = FillPipe(fds[1]);
  std::array<std::uint8_t, 1> frame{};
  const auto start = Clock::now();
  const auto result =
      WriteFrameToFd(fds[1], frame.data(), frame.size(), frame.size());
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                            start);
  ::close(fds[0]);
  ::close(fds[1]);
  std::cout << "MEASURE full_nonblocking_pipe_return_us=" << elapsed.count()
            << '\n';
  return Expect(filled, "test pipe must reach EAGAIN before measurement") &&
         Expect(result.status == FrameWriteStatus::would_block_dropped,
                "full nonblocking kernel transport must drop") &&
         Expect(result.write_syscalls == 1,
                "full nonblocking transport must make one syscall") &&
         Expect(elapsed < std::chrono::milliseconds(25),
                "full nonblocking transport must return within 25 ms");
}

bool TestBlockingBaselineIgnoresStopUntilKernelUnblocks() {
  int fds[2] = {-1, -1};
  if (!Expect(::pipe2(fds, O_CLOEXEC | O_NONBLOCK) == 0,
              "pipe2 failed for blocking baseline test"))
    return false;
  if (!Expect(FillPipe(fds[1]), "baseline pipe must be full")) {
    ::close(fds[0]);
    ::close(fds[1]);
    return false;
  }
  const int flags = ::fcntl(fds[1], F_GETFL, 0);
  if (!Expect(flags >= 0 &&
                  ::fcntl(fds[1], F_SETFL, flags & ~O_NONBLOCK) == 0,
              "failed to switch baseline pipe to blocking mode")) {
    ::close(fds[0]);
    ::close(fds[1]);
    return false;
  }

  std::atomic_bool stop{false};
  std::atomic_bool done{false};
  std::uint8_t byte = 0;
  ssize_t baseline_write = -1;
  std::thread blocked([&] {
    baseline_write = ::write(fds[1], &byte, 1);
    done.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  stop.store(true, std::memory_order_relaxed);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  const bool ignored_stop = !done.load(std::memory_order_acquire);

  std::array<std::uint8_t, 4096> drain{};
  const ssize_t drained = ::read(fds[0], drain.data(), drain.size());
  blocked.join();
  ::close(fds[0]);
  ::close(fds[1]);
  return Expect(stop.load(std::memory_order_relaxed) && ignored_stop,
                "blocking baseline write must demonstrate unbounded stop wait") &&
         Expect(drained > 0 && baseline_write == 1 &&
                    done.load(std::memory_order_acquire),
                "baseline worker must finish after kernel transport is drained");
}

bool TestStopJoinsFullNonblockingTransportWithinBound() {
  int fds[2] = {-1, -1};
  if (!Expect(::pipe2(fds, O_CLOEXEC | O_NONBLOCK) == 0,
              "pipe2 failed for stop-bound test"))
    return false;
  if (!Expect(FillPipe(fds[1]), "stop-bound pipe must be full")) {
    ::close(fds[0]);
    ::close(fds[1]);
    return false;
  }

  std::atomic_bool stop{false};
  std::atomic_uint64_t attempts{0};
  std::array<std::uint8_t, 1> frame{};
  std::thread worker([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      const auto result =
          WriteFrameToFd(fds[1], frame.data(), frame.size(), frame.size(),
                         &stop);
      if (result.status == FrameWriteStatus::stopped)
        break;
      if (result.status != FrameWriteStatus::would_block_dropped)
        break;
      attempts.fetch_add(1, std::memory_order_relaxed);
      std::this_thread::yield();
    }
  });

  const auto start_wait = Clock::now();
  while (attempts.load(std::memory_order_relaxed) == 0 &&
         Clock::now() - start_wait < std::chrono::milliseconds(25)) {
    std::this_thread::yield();
  }
  const auto stop_start = Clock::now();
  stop.store(true, std::memory_order_relaxed);
  worker.join();
  const auto stop_elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                            stop_start);
  ::close(fds[0]);
  ::close(fds[1]);
  std::cout << "MEASURE full_nonblocking_pipe_stop_join_us="
            << stop_elapsed.count() << '\n';
  return Expect(attempts.load(std::memory_order_relaxed) > 0,
                "worker must exercise full-kernel-transport EAGAIN") &&
         Expect(stop_elapsed < std::chrono::milliseconds(25),
                "stop and join must finish within 25 ms of a full transport");
}

} // namespace

int main() {
  const struct {
    const char *name;
    bool (*fn)();
  } tests[] = {
      {"open flags configure nonblocking once",
       &TestOpenFlagsDeclareNonblockingTransport},
      {"successful frame is one syscall", &TestSuccessfulWriteIsOneSyscall},
      {"partial and zero writes require reopen",
       &TestPartialAndZeroWritesAreFatalAndRequireReopen},
      {"EINTR retry is bounded and stop-aware", &TestEintrIsBoundedAndStopAware},
      {"typed failure and zero-queue drop policy", &TestTypedFailureAndDropPolicy},
      {"preflight and stop avoid syscalls", &TestPreflightAndPresetStopAvoidSyscalls},
      {"full nonblocking pipe returns within bound",
       &TestFullNonblockingPipeReturnsWithinBound},
      {"blocking baseline ignores stop until kernel unblocks",
       &TestBlockingBaselineIgnoresStopUntilKernelUnblocks},
      {"stop joins full nonblocking transport within bound",
       &TestStopJoinsFullNonblockingTransportWithinBound},
  };

  bool ok = true;
  for (const auto &test : tests) {
    const bool passed = test.fn();
    std::cout << (passed ? "PASS " : "FAIL ") << test.name << '\n';
    ok = passed && ok;
  }
  return ok ? 0 : 1;
}
