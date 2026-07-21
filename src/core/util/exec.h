#pragma once

#include <atomic>
#include <cstddef>
#include <string>

namespace studiocast::util {

struct ExecResult {
  int exit_code = -1;
  bool timed_out = false;
  bool cancelled = false;
  std::string stdout_str;
};

struct ExecCaptureOptions {
  int timeout_ms = 5000;
  std::size_t max_output_bytes = 1024 * 1024;
  // Optional cooperative cancellation for supervisor-owned helpers. When set,
  // ExecCapture terminates the whole child process group promptly after the
  // flag becomes true. The default remains unchanged for batch callers.
  const std::atomic_bool *stop_requested = nullptr;
};

ExecResult ExecCapture(const std::string &command);
ExecResult ExecCapture(const std::string &command,
                       const ExecCaptureOptions &options);

} // namespace studiocast::util
