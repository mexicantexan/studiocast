#include "exec.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <thread>

#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

namespace studiocast::util {
namespace {

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

int PositiveOrDefault(int value, int fallback) {
  return value > 0 ? value : fallback;
}

bool SetNonBlocking(int fd) {
  const int flags = ::fcntl(fd, F_GETFL, 0);
  if (flags < 0)
    return false;
  return ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

void AppendOutput(ExecResult *result, const char *data, std::size_t size,
                  std::size_t cap) {
  if (!result || !data || size == 0 || result->stdout_str.size() >= cap)
    return;

  const std::size_t remaining = cap - result->stdout_str.size();
  result->stdout_str.append(data, std::min(size, remaining));
}

void ReadAvailable(int fd, ExecResult *result, std::size_t cap,
                   bool *pipeOpen) {
  char buffer[4096];
  for (;;) {
    const ssize_t n = ::read(fd, buffer, sizeof(buffer));
    if (n > 0) {
      AppendOutput(result, buffer, static_cast<std::size_t>(n), cap);
      continue;
    }
    if (n == 0) {
      if (pipeOpen)
        *pipeOpen = false;
      return;
    }
    if (errno == EINTR)
      continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK)
      return;
    if (pipeOpen)
      *pipeOpen = false;
    return;
  }
}

bool WaitForChild(pid_t pid, std::chrono::milliseconds timeout, int *status) {
  const auto deadline = Clock::now() + timeout;
  for (;;) {
    const pid_t r = ::waitpid(pid, status, WNOHANG);
    if (r == pid)
      return true;
    if (r < 0 && errno != EINTR)
      return false;
    if (Clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(10ms);
  }
}

void TerminateProcessGroup(pid_t pid, int *status) {
  if (pid <= 0)
    return;

  (void)::kill(-pid, SIGTERM);
  if (WaitForChild(pid, 200ms, status))
    return;

  (void)::kill(-pid, SIGKILL);
  (void)WaitForChild(pid, 1000ms, status);
}

} // namespace

ExecResult ExecCapture(const std::string &command) {
  return ExecCapture(command, ExecCaptureOptions{});
}

ExecResult ExecCapture(const std::string &command,
                       const ExecCaptureOptions &options) {
  ExecResult result;

  int pipeFds[2] = {-1, -1};
  if (::pipe2(pipeFds, O_CLOEXEC) != 0)
    return result;

  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(pipeFds[0]);
    ::close(pipeFds[1]);
    return result;
  }

  if (pid == 0) {
    (void)::setpgid(0, 0);
    ::close(pipeFds[0]);
    (void)::dup2(pipeFds[1], STDOUT_FILENO);
    ::close(pipeFds[1]);
    ::execl("/bin/sh", "sh", "-c", command.c_str(),
            static_cast<char *>(nullptr));
    _exit(127);
  }

  (void)::setpgid(pid, pid);
  ::close(pipeFds[1]);
  (void)SetNonBlocking(pipeFds[0]);

  const int timeoutMs = PositiveOrDefault(options.timeout_ms, 5000);
  const std::size_t outputCap =
      options.max_output_bytes == 0 ? 1024 * 1024 : options.max_output_bytes;
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);

  bool pipeOpen = true;
  bool childExited = false;
  int status = 0;

  while (pipeOpen || !childExited) {
    if (!childExited) {
      const pid_t r = ::waitpid(pid, &status, WNOHANG);
      if (r == pid) {
        childExited = true;
      } else if (r < 0 && errno != EINTR) {
        childExited = true;
      }
    }

    if (options.stop_requested &&
        options.stop_requested->load(std::memory_order_acquire) &&
        (!childExited || pipeOpen)) {
      result.cancelled = true;
      TerminateProcessGroup(pid, &status);
      childExited = true;
      if (pipeOpen) {
        ReadAvailable(pipeFds[0], &result, outputCap, &pipeOpen);
        pipeOpen = false;
      }
    } else if (Clock::now() >= deadline && (!childExited || pipeOpen)) {
      result.timed_out = true;
      TerminateProcessGroup(pid, &status);
      childExited = true;
      if (pipeOpen) {
        ReadAvailable(pipeFds[0], &result, outputCap, &pipeOpen);
        pipeOpen = false;
      }
    }

    if (pipeOpen) {
      pollfd pfd{};
      pfd.fd = pipeFds[0];
      pfd.events = POLLIN;

      int pollTimeoutMs = 0;
      if (!childExited) {
        const auto now = Clock::now();
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline -
                                                                  now);
        pollTimeoutMs =
            remaining.count() <= 0
                ? 0
                : static_cast<int>(std::min<long long>(
                      remaining.count(), options.stop_requested ? 25 : 100));
      }

      const int pr = ::poll(&pfd, 1, pollTimeoutMs);
      if (pr > 0 && (pfd.revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
        ReadAvailable(pipeFds[0], &result, outputCap, &pipeOpen);
      } else if (childExited) {
        ReadAvailable(pipeFds[0], &result, outputCap, &pipeOpen);
      }
    }

    if (childExited && !pipeOpen)
      break;
  }

  ::close(pipeFds[0]);

  if (result.timed_out || result.cancelled) {
    result.exit_code = -1;
  } else if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else {
    result.exit_code = -1;
  }

  return result;
}

} // namespace studiocast::util
