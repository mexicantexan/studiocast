#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/open_video/latest_frame_wins_worker.h"

namespace {

using Worker = studiocast::open_video::LatestFrameWinsWorker<int, int>;
using WorkerResult = Worker::Result;
using WorkerTask = Worker::Task;
using namespace std::chrono_literals;

constexpr std::size_t kOne = 1;
constexpr std::size_t kTwo = 2;

bool WaitUntil(const std::function<bool()> &predicate,
               std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(1ms);
  }
  return predicate();
}

bool Require(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

} // namespace

namespace studiocast::tests {

bool TestLatestFrameWinsOverwritesPendingWithBlockedProcessor() {
  std::mutex mu;
  std::condition_variable cv;
  bool first_started = false;
  bool release_first = false;
  std::vector<std::uint64_t> started_sequences;
  std::vector<std::uint64_t> completed_sequences;
  std::vector<int> completed_values;

  Worker worker(
      [&](const WorkerTask &task) {
        {
          std::unique_lock<std::mutex> lock(mu);
          started_sequences.push_back(task.sequence);
          if (task.sequence == 1) {
            first_started = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_first; });
          }
        }
        return WorkerResult::Success(task.frame * 10);
      },
      [&](const WorkerTask &task, int value) {
        std::lock_guard<std::mutex> lock(mu);
        completed_sequences.push_back(task.sequence);
        completed_values.push_back(value);
        cv.notify_all();
      });

  if (!worker.Submit(1, 1)) {
    std::cerr << "initial submit unexpectedly failed\n";
    worker.Stop();
    return false;
  }
  if (!WaitUntil(
          [&] {
            std::lock_guard<std::mutex> lock(mu);
            return first_started;
          },
          500ms)) {
    std::cerr << "blocked processor did not start first frame\n";
    {
      std::lock_guard<std::mutex> lock(mu);
      release_first = true;
    }
    cv.notify_all();
    worker.Stop();
    return false;
  }

  if (!worker.Submit(2, 2) || !worker.Submit(3, 3)) {
    std::cerr << "pending submits unexpectedly failed\n";
    {
      std::lock_guard<std::mutex> lock(mu);
      release_first = true;
    }
    cv.notify_all();
    worker.Stop();
    return false;
  }

  {
    const auto stats = worker.GetStats();
    if (!Require(stats.submitted == 3, "expected 3 submitted frames") ||
        !Require(stats.started == 1,
                 "expected only the blocked first frame to have started") ||
        !Require(stats.overwritten == 1,
                 "expected second pending frame to be overwritten") ||
        !Require(stats.dropped == 1,
                 "expected overwritten pending frame to count as dropped")) {
      {
        std::lock_guard<std::mutex> lock(mu);
        release_first = true;
      }
      cv.notify_all();
      worker.Stop();
      return false;
    }
  }

  {
    std::lock_guard<std::mutex> lock(mu);
    release_first = true;
  }
  cv.notify_all();

  if (!WaitUntil(
          [&] {
            std::lock_guard<std::mutex> lock(mu);
            return completed_sequences.size() == kTwo;
          },
          500ms)) {
    std::cerr << "latest pending frame did not complete after unblock\n";
    worker.Stop();
    return false;
  }

  worker.Stop();

  bool ok = true;
  {
    std::lock_guard<std::mutex> lock(mu);
    ok &= Require(started_sequences.size() == kTwo,
                  "expected exactly two frames to start");
    if (started_sequences.size() == kTwo) {
      ok &= Require(started_sequences[0] == 1 && started_sequences[1] == 3,
                    "expected worker to process sequences 1 then 3");
    }
    ok &= Require(completed_sequences.size() == kTwo,
                  "expected exactly two completed callbacks");
    if (completed_sequences.size() == kTwo) {
      ok &= Require(completed_sequences[0] == 1 && completed_sequences[1] == 3,
                    "expected completed callbacks for sequences 1 then 3");
    }
    ok &= Require(completed_values.size() == kTwo,
                  "expected exactly two completed values");
    if (completed_values.size() == kTwo) {
      ok &= Require(completed_values[0] == 10 && completed_values[1] == 30,
                    "expected completed values for frames 1 and 3");
    }
  }

  const auto stats = worker.GetStats();
  ok &= Require(stats.submitted == 3, "submitted counter mismatch");
  ok &= Require(stats.started == 2, "started counter mismatch");
  ok &= Require(stats.completed == 2, "completed counter mismatch");
  ok &= Require(stats.overwritten == 1, "overwritten counter mismatch");
  ok &= Require(stats.dropped == 1, "dropped counter mismatch");
  ok &= Require(stats.last_completed_sequence == 3,
                "last completed sequence should be latest accepted frame");
  return ok;
}

bool TestLatestFrameWinsStopWakesAndJoins() {
  std::mutex mu;
  std::condition_variable cv;
  bool processor_started = false;
  bool release_processor = false;
  std::atomic<bool> stop_returned{false};

  Worker worker(
      [&](const WorkerTask &task) {
        {
          std::unique_lock<std::mutex> lock(mu);
          processor_started = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_processor; });
        }
        return WorkerResult::Success(task.frame);
      },
      [](const WorkerTask &, int) {});

  if (!worker.Submit(10, 10)) {
    std::cerr << "submit before stop unexpectedly failed\n";
    worker.Stop();
    return false;
  }

  if (!WaitUntil(
          [&] {
            std::lock_guard<std::mutex> lock(mu);
            return processor_started;
          },
          500ms)) {
    std::cerr << "processor did not start before stop test\n";
    {
      std::lock_guard<std::mutex> lock(mu);
      release_processor = true;
    }
    cv.notify_all();
    worker.Stop();
    return false;
  }

  std::thread stopper([&] {
    worker.Stop();
    stop_returned.store(true, std::memory_order_release);
  });

  std::this_thread::sleep_for(20ms);
  if (stop_returned.load(std::memory_order_acquire)) {
    std::cerr << "Stop returned before in-flight processing finished\n";
    {
      std::lock_guard<std::mutex> lock(mu);
      release_processor = true;
    }
    cv.notify_all();
    if (stopper.joinable()) {
      stopper.join();
    }
    worker.Stop();
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mu);
    release_processor = true;
  }
  cv.notify_all();

  if (stopper.joinable()) {
    stopper.join();
  }

  if (!stop_returned.load(std::memory_order_acquire)) {
    std::cerr << "Stop did not join after processor was released\n";
    return false;
  }
  if (worker.Submit(11, 11)) {
    std::cerr << "submit unexpectedly succeeded after Stop\n";
    return false;
  }

  const auto stats = worker.GetStats();
  return Require(stats.submitted == 1,
                 "Stop test submitted counter mismatch") &&
         Require(stats.started == 1, "Stop test started counter mismatch") &&
         Require(stats.completed == 0,
                 "in-flight result should be rejected after Stop") &&
         Require(stats.stale_results == 1,
                 "stopped in-flight result should count as stale/rejected");
}

bool TestLatestFrameWinsGenerationRejectsStaleResults() {
  std::mutex mu;
  std::condition_variable cv;
  bool first_started = false;
  bool release_first = false;
  std::vector<std::uint64_t> completed_sequences;

  Worker worker(
      [&](const WorkerTask &task) {
        {
          std::unique_lock<std::mutex> lock(mu);
          if (task.sequence == 1) {
            first_started = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_first; });
          }
        }
        return WorkerResult::Success(task.frame);
      },
      [&](const WorkerTask &task, int) {
        std::lock_guard<std::mutex> lock(mu);
        completed_sequences.push_back(task.sequence);
        cv.notify_all();
      });

  if (!worker.Submit(1, 10)) {
    std::cerr << "initial generation submit failed\n";
    worker.Stop();
    return false;
  }
  if (!WaitUntil(
          [&] {
            std::lock_guard<std::mutex> lock(mu);
            return first_started;
          },
          500ms)) {
    std::cerr << "first generation frame did not start\n";
    {
      std::lock_guard<std::mutex> lock(mu);
      release_first = true;
    }
    cv.notify_all();
    worker.Stop();
    return false;
  }

  const std::uint64_t next_generation = worker.AdvanceGeneration();
  if (next_generation <= 1) {
    std::cerr << "generation did not advance\n";
    {
      std::lock_guard<std::mutex> lock(mu);
      release_first = true;
    }
    cv.notify_all();
    worker.Stop();
    return false;
  }
  if (!worker.Submit(2, 20)) {
    std::cerr << "new generation submit failed\n";
    {
      std::lock_guard<std::mutex> lock(mu);
      release_first = true;
    }
    cv.notify_all();
    worker.Stop();
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mu);
    release_first = true;
  }
  cv.notify_all();

  if (!WaitUntil(
          [&] {
            std::lock_guard<std::mutex> lock(mu);
            return completed_sequences.size() == kOne &&
                   completed_sequences[0] == 2;
          },
          500ms)) {
    std::cerr << "stale generation result was not rejected cleanly\n";
    worker.Stop();
    return false;
  }

  worker.Stop();

  const auto stats = worker.GetStats();
  return Require(stats.submitted == 2, "generation submitted mismatch") &&
         Require(stats.started == 2, "generation started mismatch") &&
         Require(stats.completed == 1, "generation completed mismatch") &&
         Require(stats.stale_results == 1,
                 "generation stale result counter mismatch") &&
         Require(stats.last_completed_sequence == 2,
                 "generation last completed sequence mismatch");
}

bool TestLatestFrameWinsStatsCountersAndLastError() {
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::uint64_t> completed_sequences;
  std::vector<std::string> errors;

  Worker worker(
      [](const WorkerTask &task) {
        if (task.sequence == 1) {
          return WorkerResult::Failure("synthetic inference failure");
        }
        return WorkerResult::Success(task.frame + 1);
      },
      [&](const WorkerTask &task, int) {
        std::lock_guard<std::mutex> lock(mu);
        completed_sequences.push_back(task.sequence);
        cv.notify_all();
      },
      [&](const WorkerTask &, const std::string &error) {
        std::lock_guard<std::mutex> lock(mu);
        errors.push_back(error);
        cv.notify_all();
      });

  if (!worker.Submit(1, 100) || !worker.WaitForIdle(500ms) ||
      !worker.Submit(2, 200) || !worker.WaitForIdle(500ms)) {
    std::cerr << "stats test submit or idle wait failed\n";
    worker.Stop();
    return false;
  }

  worker.Stop();

  bool ok = true;
  {
    std::lock_guard<std::mutex> lock(mu);
    ok &= Require(errors.size() == kOne, "expected one error callback");
    if (!errors.empty()) {
      ok &= Require(errors[0] == "synthetic inference failure",
                    "unexpected error callback message");
    }
    ok &= Require(completed_sequences.size() == kOne,
                  "expected one successful completion callback");
    if (!completed_sequences.empty()) {
      ok &= Require(completed_sequences[0] == 2,
                    "expected successful completion for sequence 2");
    }
  }

  const auto stats = worker.GetStats();
  ok &= Require(stats.submitted == 2, "stats submitted mismatch");
  ok &= Require(stats.started == 2, "stats started mismatch");
  ok &= Require(stats.completed == 1, "stats completed mismatch");
  ok &= Require(stats.failed == 1, "stats failed mismatch");
  ok &= Require(stats.overwritten == 0, "stats overwritten mismatch");
  ok &= Require(stats.dropped == 0, "stats dropped mismatch");
  ok &= Require(stats.last_completed_sequence == 2,
                "stats last completed sequence mismatch");
  ok &= Require(stats.last_error == "synthetic inference failure",
                "stats last error mismatch");
  return ok;
}

} // namespace studiocast::tests
