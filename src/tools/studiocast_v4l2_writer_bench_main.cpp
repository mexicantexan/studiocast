#include "core/video/v4l2_writer.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using studiocast::video::WriteFrameToFd;

struct SampleStats {
  double average_us = 0.0;
  double p95_us = 0.0;
  double p99_us = 0.0;
  double min_us = 0.0;
  double max_us = 0.0;
};

SampleStats Summarize(std::vector<double> samples) {
  SampleStats out;
  if (samples.empty())
    return out;
  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  out.average_us = sum / static_cast<double>(samples.size());
  std::sort(samples.begin(), samples.end());
  const auto percentile = [&](double p) {
    const auto index = static_cast<std::size_t>(
        std::ceil(p * static_cast<double>(samples.size())) - 1.0);
    return samples[std::min(index, samples.size() - 1)];
  };
  out.p95_us = percentile(0.95);
  out.p99_us = percentile(0.99);
  out.min_us = samples.front();
  out.max_us = samples.back();
  return out;
}

bool BaselineBlockingLoop(int fd, const std::uint8_t *data, std::size_t bytes,
                          std::uint64_t *syscalls) {
  std::size_t offset = 0;
  while (offset < bytes) {
    const ssize_t wrote = ::write(fd, data + offset, bytes - offset);
    ++*syscalls;
    if (wrote < 0) {
      if (errno == EINTR)
        continue;
      return false;
    }
    if (wrote == 0)
      return false;
    offset += static_cast<std::size_t>(wrote);
  }
  return true;
}

template <typename Operation>
SampleStats Measure(std::size_t warmup, std::size_t iterations,
                    Operation operation, std::uint64_t *syscalls) {
  for (std::size_t i = 0; i < warmup; ++i) {
    if (!operation(syscalls))
      return {};
  }
  std::vector<double> samples;
  samples.reserve(iterations);
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto begin = Clock::now();
    const bool ok = operation(syscalls);
    const auto end = Clock::now();
    if (!ok)
      return {};
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  return Summarize(std::move(samples));
}

void Print(std::string_view implementation, int width, int height,
           std::size_t warmup, std::size_t iterations,
           std::uint64_t measured_syscalls, const SampleStats &stats) {
  std::cout << implementation << ',' << width << ',' << height << ',' << warmup
            << ',' << iterations << ',' << std::fixed << std::setprecision(6)
            << stats.average_us << ',' << stats.p95_us << ',' << stats.p99_us
            << ',' << stats.min_us << ',' << stats.max_us << ','
            << measured_syscalls << ','
            << (static_cast<double>(measured_syscalls) /
                static_cast<double>(iterations))
            << '\n';
}

bool RunGeometry(int fd, int width, int height, std::size_t warmup,
                 std::size_t iterations) {
  const std::size_t bytes = static_cast<std::size_t>(width) *
                            static_cast<std::size_t>(height) * 2u;
  std::vector<std::uint8_t> frame(bytes, 0x80);

  std::uint64_t baseline_syscalls = 0;
  const auto baseline = Measure(
      warmup, iterations,
      [&](std::uint64_t *calls) {
        return BaselineBlockingLoop(fd, frame.data(), frame.size(), calls);
      },
      &baseline_syscalls);
  baseline_syscalls -= warmup;
  Print("baseline_d2a6_blocking_loop", width, height, warmup, iterations,
        baseline_syscalls, baseline);

  std::uint64_t candidate_syscalls = 0;
  const auto candidate = Measure(
      warmup, iterations,
      [&](std::uint64_t *calls) {
        const auto result = WriteFrameToFd(fd, frame.data(), frame.size(),
                                           frame.size());
        *calls += result.write_syscalls;
        return result.FrameCommitted();
      },
      &candidate_syscalls);
  candidate_syscalls -= warmup;
  Print("candidate_nonblocking_direct_contract", width, height, warmup,
        iterations, candidate_syscalls, candidate);
  return baseline_syscalls == iterations && candidate_syscalls == iterations;
}

} // namespace

int main() {
  constexpr std::size_t kWarmup = 1000;
  constexpr std::size_t kIterations = 20000;
  const int fd = ::open("/dev/null", O_WRONLY | O_CLOEXEC | O_NONBLOCK);
  if (fd < 0) {
    std::cerr << "open(/dev/null) failed\n";
    return 1;
  }

  std::cout << "implementation,width,height,warmup,iterations,avg_us,p95_us,"
               "p99_us,min_us,max_us,write_syscalls,syscalls_per_frame\n";
  const bool ok720 = RunGeometry(fd, 1280, 720, kWarmup, kIterations);
  const bool ok1080 = RunGeometry(fd, 1920, 1080, kWarmup, kIterations);
  ::close(fd);
  return ok720 && ok1080 ? 0 : 2;
}
