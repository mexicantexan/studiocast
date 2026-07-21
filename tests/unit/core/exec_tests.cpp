#include <chrono>
#include <iostream>

#include "core/util/exec.h"

namespace {

bool Expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

bool TestExecCaptureSuccess() {
  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = 1000;
  const auto result =
      studiocast::util::ExecCapture("printf studiocast", options);
  return Expect(result.exit_code == 0, "printf should exit successfully") &&
         Expect(result.stdout_str == "studiocast",
                "stdout should be captured") &&
         Expect(!result.timed_out, "successful command should not time out");
}

bool TestExecCaptureTimeoutIsBounded() {
  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = 100;
  const auto start = std::chrono::steady_clock::now();
  const auto result =
      studiocast::util::ExecCapture("sleep 2; printf late", options);
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();

  return Expect(result.exit_code == -1, "timed-out command should fail") &&
         Expect(result.timed_out, "timeout flag should be set") &&
         Expect(elapsed < 1500, "timeout should not wait for command exit");
}

bool TestExecCaptureTimeoutCoversDescendantHoldingStdoutOpen() {
  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = 100;
  const auto start = std::chrono::steady_clock::now();
  const auto result =
      studiocast::util::ExecCapture("(sleep 2) & printf done", options);
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();

  return Expect(result.exit_code == -1,
                "descendant-held pipe timeout should fail") &&
         Expect(result.timed_out,
                "descendant-held pipe should set timeout flag") &&
         Expect(elapsed < 1500,
                "timeout should not wait for background descendant exit");
}

} // namespace

int main() {
  bool ok = true;
  ok = TestExecCaptureSuccess() && ok;
  ok = TestExecCaptureTimeoutIsBounded() && ok;
  ok = TestExecCaptureTimeoutCoversDescendantHoldingStdoutOpen() && ok;
  return ok ? 0 : 1;
}
