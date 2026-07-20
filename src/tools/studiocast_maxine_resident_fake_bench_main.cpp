#include "core/maxine/resident_frame_section.h"
#include "core/maxine/testing/fake_resident_frame_executor.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using studiocast::maxine::HostRgbFrameView;
using studiocast::maxine::ResidentExecutionStatus;
using studiocast::maxine::ResidentFrameCounters;
using studiocast::maxine::ResidentFrameKey;
using studiocast::maxine::ResidentFramePlan;
using studiocast::maxine::ResidentFrameSection;
using studiocast::maxine::ResidentStageKind;
using studiocast::maxine::testing::FakeResidentFrameExecutor;

struct Options {
  uint64_t warmup = 128;
  uint64_t iterations = 4096;
};

bool ParseUnsigned(std::string_view text, uint64_t &value) {
  const char *begin = text.data();
  const char *end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, value);
  return parsed.ec == std::errc{} && parsed.ptr == end && value != 0;
}

bool ParseOptions(int argc, char **argv, Options &options) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if ((argument == "--warmup" || argument == "--iterations") &&
        i + 1 < argc) {
      uint64_t value = 0;
      if (!ParseUnsigned(argv[++i], value))
        return false;
      if (argument == "--warmup")
        options.warmup = value;
      else
        options.iterations = value;
      continue;
    }
    return false;
  }
  return true;
}

uint64_t CpuTailCount(const ResidentFrameCounters &counters) {
  uint64_t total = 0;
  for (const uint64_t value : counters.cpu_tail_boundaries)
    total += value;
  return total;
}

double Percentile(const std::vector<double> &sorted, double fraction) {
  const auto index = static_cast<std::size_t>(
      fraction * static_cast<double>(sorted.size() - 1));
  return sorted[index];
}

bool RunMode(std::string_view mode, const ResidentFramePlan &plan,
             const Options &options) {
  uint8_t host_token = 0;
  const ResidentFrameKey key{1, 640, 360, 0xA11, 0xB22};
  const HostRgbFrameView host{&host_token, 640, 360, 640u * 3u};
  FakeResidentFrameExecutor executor{};
  executor.SetSyntheticWork(48);
  ResidentFrameSection section{};

  for (uint64_t i = 0; i < options.warmup; ++i) {
    if (section.Execute(plan, key, i + 1, host, executor).status !=
        ResidentExecutionStatus::cpu_visible_output)
      return false;
  }
  const auto before = section.counters();

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(options.iterations));
  double sum_ms = 0.0;
  double min_ms = std::numeric_limits<double>::max();
  double max_ms = 0.0;
  for (uint64_t i = 0; i < options.iterations; ++i) {
    const auto start = Clock::now();
    const auto result =
        section.Execute(plan, key, options.warmup + i + 1, host, executor);
    const auto end = Clock::now();
    if (result.status != ResidentExecutionStatus::cpu_visible_output)
      return false;
    const double milliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
    samples.push_back(milliseconds);
    sum_ms += milliseconds;
    min_ms = std::min(min_ms, milliseconds);
    max_ms = std::max(max_ms, milliseconds);
  }
  const auto after = section.counters();
  std::sort(samples.begin(), samples.end());
  const double divisor = static_cast<double>(options.iterations);
  const double average = sum_ms / divisor;
  const double uploads =
      static_cast<double>(after.upload_successes - before.upload_successes) /
      divisor;
  const double final_downloads = static_cast<double>(
      after.final_download_successes - before.final_download_successes) /
                                 divisor;
  const double cpu_continuation_downloads = static_cast<double>(
      after.cpu_continuation_download_successes -
      before.cpu_continuation_download_successes) /
                                              divisor;
  const double matte_inferences = static_cast<double>(
      after.matte_inference_successes - before.matte_inference_successes) /
                                   divisor;
  const double forced_syncs =
      static_cast<double>(after.sync_successes - before.sync_successes) /
      divisor;
  const double cpu_tails =
      static_cast<double>(CpuTailCount(after) - CpuTailCount(before)) / divisor;

  std::cout << std::fixed << std::setprecision(6)
            << "backend=fake mode=" << mode << " geometry=640x360"
            << " warmup=" << options.warmup
            << " iterations=" << options.iterations << " avg_ms=" << average
            << " p95_ms=" << Percentile(samples, 0.95)
            << " p99_ms=" << Percentile(samples, 0.99)
            << " min_ms=" << min_ms << " max_ms=" << max_ms
            << " uploads_per_frame=" << uploads
            << " final_downloads_per_frame=" << final_downloads
            << " cpu_continuation_downloads_per_frame="
            << cpu_continuation_downloads
            << " matte_inferences_per_frame=" << matte_inferences
            << " forced_syncs_per_frame=" << forced_syncs
            << " cpu_tails_per_frame=" << cpu_tails
            << " synthetic_frame_age_avg_ms=" << (2.0 + average) << '\n';
  return true;
}

} // namespace

int main(int argc, char **argv) {
  Options options{};
  if (!ParseOptions(argc, argv, options)) {
    std::cerr << "usage: studiocast-maxine-resident-fake-bench "
                 "[--warmup N] [--iterations N]\n";
    return EXIT_FAILURE;
  }

  ResidentFramePlan single{};
  if (!single.AddCompatible(ResidentStageKind::denoise, false))
    return EXIT_FAILURE;

  ResidentFramePlan combined{};
  if (!combined.AddCompatible(ResidentStageKind::background_blur, false, true,
                              0xCAFE) ||
      !combined.AddCompatible(ResidentStageKind::relighting, false, true,
                              0xCAFE))
    return EXIT_FAILURE;

  if (!RunMode("single", single, options) ||
      !RunMode("combined", combined, options))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
