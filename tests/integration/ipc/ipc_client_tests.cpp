#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include "core/ipc/daemon_client.h"
#include "core/ipc/daemon_server.h"
#include "core/ipc/daemon_socket.h"

namespace {

namespace fs = std::filesystem;
using namespace std::chrono_literals;

class ScopedRuntimeDir {
public:
  explicit ScopedRuntimeDir(const std::string &name) {
    had_old_ = std::getenv("XDG_RUNTIME_DIR") != nullptr;
    if (had_old_)
      old_ = std::getenv("XDG_RUNTIME_DIR");

    std::error_code ec;
    const auto tmp = fs::temp_directory_path(ec);
    if (ec) {
      error_ = "failed to resolve temp directory: " + ec.message();
      return;
    }

    dir_ = tmp /
           (name + "-" + std::to_string(static_cast<long long>(::getpid())));
    fs::remove_all(dir_, ec);
    fs::create_directories(dir_, ec);
    if (ec) {
      error_ = "failed to create temp runtime dir: " + ec.message();
      return;
    }

    if (::setenv("XDG_RUNTIME_DIR", dir_.string().c_str(), 1) != 0)
      error_ = std::string("setenv failed: ") + std::strerror(errno);
  }

  ~ScopedRuntimeDir() {
    if (had_old_) {
      (void)::setenv("XDG_RUNTIME_DIR", old_.c_str(), 1);
    } else {
      (void)::unsetenv("XDG_RUNTIME_DIR");
    }

    std::error_code ec;
    fs::remove_all(dir_, ec);
  }

  bool ok() const { return error_.empty(); }
  const std::string &error() const { return error_; }

private:
  fs::path dir_;
  bool had_old_ = false;
  std::string old_;
  std::string error_;
};

bool BindListeningSocket(const fs::path &path, int *listen_fd,
                         std::string *error) {
  if (!listen_fd)
    return false;
  *listen_fd = -1;

  const std::string pathStr = path.string();
  if (pathStr.size() >= sizeof(sockaddr_un::sun_path)) {
    if (error)
      *error = "socket path too long";
    return false;
  }

  (void)::unlink(pathStr.c_str());

  const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    if (error)
      *error = std::string("socket failed: ") + std::strerror(errno);
    return false;
  }

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, pathStr.c_str(), sizeof(addr.sun_path) - 1);

  if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
    const int e = errno;
    ::close(fd);
    if (error)
      *error = std::string("bind failed: ") + std::strerror(e);
    return false;
  }

  if (::listen(fd, 1) != 0) {
    const int e = errno;
    ::close(fd);
    if (error)
      *error = std::string("listen failed: ") + std::strerror(e);
    return false;
  }

  *listen_fd = fd;
  return true;
}

bool ConnectUnixSocket(const fs::path &path, int *connected_fd,
                       std::string *error) {
  if (!connected_fd)
    return false;
  *connected_fd = -1;

  const std::string pathStr = path.string();
  if (pathStr.size() >= sizeof(sockaddr_un::sun_path)) {
    if (error)
      *error = "socket path too long";
    return false;
  }

  const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    if (error)
      *error = std::string("socket failed: ") + std::strerror(errno);
    return false;
  }

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, pathStr.c_str(), sizeof(addr.sun_path) - 1);

  if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
    const int e = errno;
    ::close(fd);
    if (error)
      *error = std::string("connect failed: ") + std::strerror(e);
    return false;
  }

  *connected_fd = fd;
  return true;
}

bool SendAllNoSigpipe(int fd, const void *data, std::size_t bytes,
                      std::string *error) {
  const char *p = static_cast<const char *>(data);
  std::size_t n = 0;
  while (n < bytes) {
    const ssize_t w = ::send(fd, p + n, bytes - n, MSG_NOSIGNAL);
    if (w < 0) {
      if (errno == EINTR)
        continue;
      if (error)
        *error = std::string("send failed: ") + std::strerror(errno);
      return false;
    }
    if (w == 0) {
      if (error)
        *error = "send returned 0";
      return false;
    }
    n += static_cast<std::size_t>(w);
  }
  return true;
}

bool WaitForChild(pid_t pid, std::chrono::milliseconds timeout, int *status) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    const pid_t r = ::waitpid(pid, status, WNOHANG);
    if (r == pid)
      return true;
    if (r < 0 && errno != EINTR)
      return false;
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(10ms);
  }
}

studiocast::ipc::DaemonCallOptions FastOptions() {
  studiocast::ipc::DaemonCallOptions options;
  options.connect_timeout_ms = 100;
  options.io_timeout_ms = 100;
  return options;
}

bool TestDaemonCallSuccess() {
  ScopedRuntimeDir runtime("studiocast-ipc-success");
  if (!runtime.ok()) {
    std::cerr << runtime.error() << "\n";
    return false;
  }

  std::string err;
  const auto socketPath = studiocast::ipc::DaemonSocketPath(&err);
  if (socketPath.empty()) {
    std::cerr << "DaemonSocketPath failed: " << err << "\n";
    return false;
  }

  studiocast::ipc::DaemonServer server;
  if (!server.Start(
          socketPath,
          [](const std::string &line) {
            if (line == "PING")
              return std::string("OK {\"pong\":true}");
            return std::string("ERR {\"error\":\"unexpected\"}");
          },
          &err)) {
    std::cerr << "server.Start failed: " << err << "\n";
    return false;
  }

  studiocast::ipc::DaemonCallResult res;
  err.clear();
  const bool ok = studiocast::ipc::DaemonCall("PING", &res, &err,
                                              FastOptions());
  server.Stop();

  if (!ok || !res.ok || res.json != "{\"pong\":true}") {
    std::cerr << "DaemonCall success path failed; ok=" << ok
              << " res.ok=" << res.ok << " json='" << res.json
              << "' err='" << err << "'\n";
    return false;
  }

  return true;
}

bool TestMissingSocketFailsQuickly() {
  ScopedRuntimeDir runtime("studiocast-ipc-missing");
  if (!runtime.ok()) {
    std::cerr << runtime.error() << "\n";
    return false;
  }

  studiocast::ipc::DaemonCallResult res;
  std::string err;
  const auto start = std::chrono::steady_clock::now();
  const bool ok =
      studiocast::ipc::DaemonCall("GET_STATUS", &res, &err, FastOptions());
  const auto elapsed = std::chrono::steady_clock::now() - start;

  if (ok) {
    std::cerr << "DaemonCall unexpectedly succeeded for missing socket\n";
    return false;
  }

  if (elapsed > 500ms) {
    std::cerr << "missing socket path was not bounded; elapsed="
              << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                     .count()
              << "ms err='" << err << "'\n";
    return false;
  }

  return true;
}

bool TestReadTimeoutIsBounded() {
  ScopedRuntimeDir runtime("studiocast-ipc-timeout");
  if (!runtime.ok()) {
    std::cerr << runtime.error() << "\n";
    return false;
  }

  std::string err;
  const auto socketPath = studiocast::ipc::DaemonSocketPath(&err);
  if (socketPath.empty()) {
    std::cerr << "DaemonSocketPath failed: " << err << "\n";
    return false;
  }

  int listenFd = -1;
  if (!BindListeningSocket(socketPath, &listenFd, &err)) {
    std::cerr << "BindListeningSocket failed: " << err << "\n";
    return false;
  }

  std::thread acceptThread([listenFd] {
    const int fd = ::accept4(listenFd, nullptr, nullptr, SOCK_CLOEXEC);
    if (fd >= 0) {
      std::this_thread::sleep_for(200ms);
      ::close(fd);
    }
  });

  studiocast::ipc::DaemonCallOptions options;
  options.connect_timeout_ms = 100;
  options.io_timeout_ms = 50;

  studiocast::ipc::DaemonCallResult res;
  err.clear();
  const auto start = std::chrono::steady_clock::now();
  const bool ok =
      studiocast::ipc::DaemonCall("GET_STATUS", &res, &err, options);
  const auto elapsed = std::chrono::steady_clock::now() - start;

  ::shutdown(listenFd, SHUT_RDWR);
  ::close(listenFd);
  if (acceptThread.joinable())
    acceptThread.join();
  (void)::unlink(socketPath.string().c_str());

  if (ok) {
    std::cerr << "DaemonCall unexpectedly succeeded when server never replied\n";
    return false;
  }

  if (elapsed > 500ms) {
    std::cerr << "read timeout was not bounded; elapsed="
              << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                     .count()
              << "ms err='" << err << "'\n";
    return false;
  }

  if (err.find("read timed out") == std::string::npos) {
    std::cerr << "expected read timeout error, got '" << err << "'\n";
    return false;
  }

  return true;
}

bool TestTrickleResponseUsesTotalIoDeadline() {
  ScopedRuntimeDir runtime("studiocast-ipc-trickle-timeout");
  if (!runtime.ok()) {
    std::cerr << runtime.error() << "\n";
    return false;
  }

  std::string err;
  const auto socketPath = studiocast::ipc::DaemonSocketPath(&err);
  if (socketPath.empty()) {
    std::cerr << "DaemonSocketPath failed: " << err << "\n";
    return false;
  }

  int listenFd = -1;
  if (!BindListeningSocket(socketPath, &listenFd, &err)) {
    std::cerr << "BindListeningSocket failed: " << err << "\n";
    return false;
  }

  std::thread acceptThread([listenFd] {
    const int fd = ::accept4(listenFd, nullptr, nullptr, SOCK_CLOEXEC);
    if (fd < 0)
      return;

    char ch = '\0';
    while (::read(fd, &ch, 1) == 1 && ch != '\n') {
    }

    const std::string reply = "OK {\"slow\":true}\n";
    for (char out : reply) {
      const ssize_t w = ::send(fd, &out, 1, MSG_NOSIGNAL);
      if (w <= 0)
        break;
      std::this_thread::sleep_for(25ms);
    }
    ::close(fd);
  });

  studiocast::ipc::DaemonCallOptions options;
  options.connect_timeout_ms = 100;
  options.io_timeout_ms = 100;

  studiocast::ipc::DaemonCallResult res;
  err.clear();
  const auto start = std::chrono::steady_clock::now();
  const bool ok =
      studiocast::ipc::DaemonCall("GET_STATUS", &res, &err, options);
  const auto elapsed = std::chrono::steady_clock::now() - start;

  ::shutdown(listenFd, SHUT_RDWR);
  ::close(listenFd);
  if (acceptThread.joinable())
    acceptThread.join();
  (void)::unlink(socketPath.string().c_str());

  if (ok) {
    std::cerr << "DaemonCall unexpectedly succeeded on trickled response; "
              << "elapsed="
              << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                     .count()
              << "ms raw='" << res.raw << "'\n";
    return false;
  }

  if (elapsed > 350ms) {
    std::cerr << "trickled response exceeded total I/O deadline; elapsed="
              << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                     .count()
              << "ms err='" << err << "'\n";
    return false;
  }

  if (err.find("timed out") == std::string::npos &&
      err.find("deadline") == std::string::npos) {
    std::cerr << "expected timeout/deadline error, got '" << err << "'\n";
    return false;
  }

  return true;
}

int RunEarlyDisconnectChild() {
  std::signal(SIGPIPE, SIG_DFL);

  ScopedRuntimeDir runtime("studiocast-ipc-early-disconnect");
  if (!runtime.ok()) {
    std::cerr << runtime.error() << "\n";
    return 1;
  }

  std::string err;
  const auto socketPath = studiocast::ipc::DaemonSocketPath(&err);
  if (socketPath.empty()) {
    std::cerr << "DaemonSocketPath failed: " << err << "\n";
    return 1;
  }

  studiocast::ipc::DaemonServer server;
  if (!server.Start(
          socketPath,
          [](const std::string &line) {
            if (line == "EARLY") {
              std::this_thread::sleep_for(50ms);
              return std::string("OK ") + std::string(1024 * 1024, 'x');
            }
            if (line == "PING")
              return std::string("OK {\"alive\":true}");
            return std::string("ERR {\"error\":\"unexpected\"}");
          },
          &err)) {
    std::cerr << "server.Start failed: " << err << "\n";
    return 1;
  }

  int fd = -1;
  if (!ConnectUnixSocket(socketPath, &fd, &err)) {
    std::cerr << "ConnectUnixSocket failed: " << err << "\n";
    server.Stop();
    return 1;
  }

  const std::string request = "EARLY\n";
  if (!SendAllNoSigpipe(fd, request.data(), request.size(), &err)) {
    std::cerr << "early disconnect request write failed: " << err << "\n";
    ::close(fd);
    server.Stop();
    return 1;
  }
  ::shutdown(fd, SHUT_RDWR);
  ::close(fd);

  std::this_thread::sleep_for(200ms);

  studiocast::ipc::DaemonCallResult res;
  err.clear();
  const bool ok =
      studiocast::ipc::DaemonCall("PING", &res, &err, FastOptions());
  server.Stop();

  if (!ok || !res.ok || res.json != "{\"alive\":true}") {
    std::cerr << "server did not answer after early disconnect; ok=" << ok
              << " res.ok=" << res.ok << " json='" << res.json
              << "' err='" << err << "'\n";
    return 1;
  }

  return 0;
}

bool TestDaemonEarlyDisconnectDoesNotKillOrFailServer() {
  std::cout.flush();
  std::cerr.flush();
  const pid_t pid = ::fork();
  if (pid < 0) {
    std::cerr << "fork failed: " << std::strerror(errno) << "\n";
    return false;
  }
  if (pid == 0) {
    const int rc = RunEarlyDisconnectChild();
    std::cerr.flush();
    std::cout.flush();
    _exit(rc);
  }

  int status = 0;
  if (!WaitForChild(pid, 3s, &status)) {
    (void)::kill(pid, SIGKILL);
    (void)::waitpid(pid, &status, 0);
    std::cerr << "early-disconnect child timed out\n";
    return false;
  }

  if (WIFSIGNALED(status)) {
    std::cerr << "early-disconnect child died from signal "
              << WTERMSIG(status) << " (" << strsignal(WTERMSIG(status))
              << ")\n";
    return false;
  }

  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::cerr << "early-disconnect child exited with status " << status
              << "\n";
    return false;
  }

  return true;
}

int RunManyShortClientsChild() {
  ScopedRuntimeDir runtime("studiocast-ipc-many-short-clients");
  if (!runtime.ok()) {
    std::cerr << runtime.error() << "\n";
    return 1;
  }

  std::string err;
  const auto socketPath = studiocast::ipc::DaemonSocketPath(&err);
  if (socketPath.empty()) {
    std::cerr << "DaemonSocketPath failed: " << err << "\n";
    return 1;
  }

  std::atomic<int> handled{0};
  studiocast::ipc::DaemonServer server;
  if (!server.Start(
          socketPath,
          [&handled](const std::string &line) {
            if (line == "PING") {
              handled.fetch_add(1);
              return std::string("OK {}");
            }
            return std::string("ERR {\"error\":\"unexpected\"}");
          },
          &err)) {
    std::cerr << "server.Start failed: " << err << "\n";
    return 1;
  }

  constexpr int kClientCount = 2500;
  for (int i = 0; i < kClientCount; ++i) {
    studiocast::ipc::DaemonCallResult res;
    err.clear();
    const bool ok =
        studiocast::ipc::DaemonCall("PING", &res, &err, FastOptions());
    if (!ok || !res.ok) {
      std::cerr << "short client " << i << " failed; ok=" << ok
                << " res.ok=" << res.ok << " err='" << err << "'\n";
      server.Stop();
      return 1;
    }
  }

  const auto settleDeadline = std::chrono::steady_clock::now() + 2s;
  while (handled.load() < kClientCount &&
         std::chrono::steady_clock::now() < settleDeadline) {
    std::this_thread::sleep_for(1ms);
  }
  std::this_thread::sleep_for(50ms);

  const auto stopStart = std::chrono::steady_clock::now();
  server.Stop();
  const auto stopElapsed = std::chrono::steady_clock::now() - stopStart;

  if (stopElapsed > 200ms) {
    std::cerr << "server Stop() scaled with retained completed client threads; "
              << "clients=" << kClientCount << " stop_elapsed="
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     stopElapsed)
                     .count()
              << "ms\n";
    return 1;
  }

  return 0;
}

bool TestManyShortClientsDoNotMakeStopHang() {
  std::cout.flush();
  std::cerr.flush();
  const pid_t pid = ::fork();
  if (pid < 0) {
    std::cerr << "fork failed: " << std::strerror(errno) << "\n";
    return false;
  }
  if (pid == 0) {
    const int rc = RunManyShortClientsChild();
    std::cerr.flush();
    std::cout.flush();
    _exit(rc);
  }

  int status = 0;
  if (!WaitForChild(pid, 15s, &status)) {
    (void)::kill(pid, SIGKILL);
    (void)::waitpid(pid, &status, 0);
    std::cerr << "many-short-clients child timed out\n";
    return false;
  }

  if (WIFSIGNALED(status)) {
    std::cerr << "many-short-clients child died from signal "
              << WTERMSIG(status) << " (" << strsignal(WTERMSIG(status))
              << ")\n";
    return false;
  }

  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::cerr << "many-short-clients child exited with status " << status
              << "\n";
    return false;
  }

  return true;
}

} // namespace

int main() {
  struct Test {
    const char *name;
    bool (*fn)();
  };

  const Test tests[] = {
      {"DaemonCall success", TestDaemonCallSuccess},
      {"DaemonCall missing socket bounded", TestMissingSocketFailsQuickly},
      {"DaemonCall read timeout bounded", TestReadTimeoutIsBounded},
      {"DaemonCall trickle response total I/O deadline",
       TestTrickleResponseUsesTotalIoDeadline},
      {"DaemonServer early disconnect reply survives",
       TestDaemonEarlyDisconnectDoesNotKillOrFailServer},
      {"DaemonServer Stop after many short clients is bounded",
       TestManyShortClientsDoNotMakeStopHang},
  };

  bool ok = true;
  for (const auto &test : tests) {
    if (!test.fn()) {
      std::cerr << "[FAIL] " << test.name << "\n";
      ok = false;
    } else {
      std::cout << "[PASS] " << test.name << "\n";
    }
  }

  return ok ? 0 : 1;
}
