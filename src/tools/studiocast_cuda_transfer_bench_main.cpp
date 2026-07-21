#include "core/cuda/cuda_image.h"
#include "core/maxine/cuda_driver_api.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
  int width = 1280;
  int height = 720;
  int warmup = 60;
  int iterations = 600;
  bool csv = false;
};

struct Stats {
  double avg_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

bool ParseIntArg(const char *arg, int *out) {
  if (!arg || !out)
    return false;
  char *end = nullptr;
  const long value = std::strtol(arg, &end, 10);
  if (!end || *end != '\0' || value <= 0 ||
      value > static_cast<long>(std::numeric_limits<int>::max())) {
    return false;
  }
  *out = static_cast<int>(value);
  return true;
}

void PrintUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " [--width N] [--height N] [--warmup N]"
               " [--iterations N] [--csv]\n";
}

bool ParseArgs(int argc, char **argv, Options *opts) {
  if (!opts)
    return false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    auto require_value = [&](int *out) -> bool {
      if (i + 1 >= argc)
        return false;
      ++i;
      return ParseIntArg(argv[i], out);
    };

    if (arg == "--width") {
      if (!require_value(&opts->width))
        return false;
    } else if (arg == "--height") {
      if (!require_value(&opts->height))
        return false;
    } else if (arg == "--warmup") {
      if (!require_value(&opts->warmup))
        return false;
    } else if (arg == "--iterations") {
      if (!require_value(&opts->iterations))
        return false;
    } else if (arg == "--csv") {
      opts->csv = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      return false;
    }
  }
  return opts->width > 0 && opts->height > 0 && opts->warmup >= 0 &&
         opts->iterations > 0;
}

Stats ComputeStats(std::vector<double> samples) {
  Stats stats{};
  if (samples.empty())
    return stats;

  std::sort(samples.begin(), samples.end());
  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  stats.avg_ms = sum / static_cast<double>(samples.size());
  stats.min_ms = samples.front();
  stats.max_ms = samples.back();

  auto percentile = [&](double pct) {
    const double rank =
        (pct / 100.0) * static_cast<double>(samples.size() - 1u);
    const auto lo = static_cast<std::size_t>(rank);
    const auto hi = std::min(lo + 1u, samples.size() - 1u);
    const double t = rank - static_cast<double>(lo);
    return samples[lo] + (samples[hi] - samples[lo]) * t;
  };
  stats.p95_ms = percentile(95.0);
  stats.p99_ms = percentile(99.0);
  return stats;
}

std::uint64_t Checksum(const std::vector<std::uint8_t> &data) {
  std::uint64_t sum = 1469598103934665603ull;
  for (std::uint8_t v : data) {
    sum ^= static_cast<std::uint64_t>(v);
    sum *= 1099511628211ull;
  }
  return sum;
}

std::string CsvSafe(std::string s) {
  for (char &ch : s) {
    if (ch == ',')
      ch = ';';
    if (ch == '\n' || ch == '\r')
      ch = ' ';
  }
  return s;
}

void PrintCsvHeader() {
  std::cout << "backend,stage,width,height,warmup_iterations,iterations,status,"
               "skip_reason,avg_ms,p95_ms,p99_ms,min_ms,max_ms,checksum,"
               "roundtrip_ok\n";
}

void PrintCsvSkipRows(const Options &opts, const std::string &reason) {
  for (const char *stage : {"upload", "download", "roundtrip"}) {
    std::cout << "cuda," << stage << "," << opts.width << "," << opts.height
              << "," << opts.warmup << "," << opts.iterations
              << ",skipped," << CsvSafe(reason)
              << ",0,0,0,0,0,0,na\n";
  }
}

void PrintCsvResult(const Options &opts, const char *stage,
                    const Stats &stats, std::uint64_t checksum,
                    const char *roundtripOk) {
  std::cout << "cuda," << stage << "," << opts.width << "," << opts.height
            << "," << opts.warmup << "," << opts.iterations << ",ok,,"
            << stats.avg_ms << "," << stats.p95_ms << "," << stats.p99_ms
            << "," << stats.min_ms << "," << stats.max_ms << ","
            << checksum << "," << roundtripOk << "\n";
}

void PrintHumanResult(const Options &opts, const char *stage,
                      const Stats &stats, std::uint64_t checksum,
                      const char *roundtripOk) {
  std::cout << "CUDA RGB24 " << stage << " benchmark\n";
  std::cout << "  size       : " << opts.width << "x" << opts.height << "\n";
  std::cout << "  iterations : " << opts.iterations << " (warmup "
            << opts.warmup << ")\n";
  std::cout << "  avg        : " << std::fixed << std::setprecision(4)
            << stats.avg_ms << " ms\n";
  std::cout << "  p95        : " << stats.p95_ms << " ms\n";
  std::cout << "  p99        : " << stats.p99_ms << " ms\n";
  std::cout << "  min/max    : " << stats.min_ms << " / " << stats.max_ms
            << " ms\n";
  std::cout << "  checksum   : " << checksum << "\n";
  std::cout << "  roundtrip  : " << roundtripOk << "\n";
}

template <typename Fn>
bool RunTimed(const Options &opts, Fn &&fn, Stats *stats) {
  for (int i = 0; i < opts.warmup; ++i) {
    if (!fn())
      return false;
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(opts.iterations));
  using Clock = std::chrono::steady_clock;
  for (int i = 0; i < opts.iterations; ++i) {
    const auto t0 = Clock::now();
    if (!fn())
      return false;
    const auto t1 = Clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                        .count();
    samples.push_back(static_cast<double>(ns) / 1'000'000.0);
  }

  if (stats)
    *stats = ComputeStats(std::move(samples));
  return true;
}

} // namespace

int main(int argc, char **argv) {
  Options opts;
  if (!ParseArgs(argc, argv, &opts)) {
    PrintUsage(argv[0]);
    return 2;
  }

  studiocast::maxine::CudaDriverApi cuda;
  std::string err;
  if (!cuda.Initialize(&err)) {
    if (opts.csv) {
      PrintCsvHeader();
      PrintCsvSkipRows(opts, err);
    }
    std::cerr << "SKIP: CUDA unavailable: " << err << "\n";
    return 77;
  }
  if (!cuda.EnsureContext(&err)) {
    if (opts.csv) {
      PrintCsvHeader();
      PrintCsvSkipRows(opts, err);
    }
    std::cerr << "SKIP: CUDA context unavailable: " << err << "\n";
    return 77;
  }

  studiocast::maxine::CUstream stream = nullptr;
  if (!cuda.CreateStream(&stream, &err)) {
    std::cerr << "ERROR: CreateStream failed: " << err << "\n";
    return 1;
  }

  studiocast::cuda::CudaImage image;
  if (!image.Allocate(&cuda, opts.width, opts.height,
                      studiocast::cuda::PixelFormatGpu::rgb_u8, &err)) {
    std::cerr << "ERROR: CudaImage allocation failed: " << err << "\n";
    (void)cuda.DestroyStream(stream, nullptr);
    return 1;
  }

  const std::size_t stride = static_cast<std::size_t>(opts.width) * 3u;
  const std::size_t bytes = stride * static_cast<std::size_t>(opts.height);
  std::vector<std::uint8_t> src(bytes);
  std::vector<std::uint8_t> dst(bytes, 0);
  for (int y = 0; y < opts.height; ++y) {
    for (int x = 0; x < opts.width; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * stride +
                            static_cast<std::size_t>(x) * 3u;
      src[i + 0] = static_cast<std::uint8_t>((x * 3 + y * 7) & 0xFF);
      src[i + 1] = static_cast<std::uint8_t>((x * 5 + y * 11) & 0xFF);
      src[i + 2] = static_cast<std::uint8_t>((x * 13 + y * 17) & 0xFF);
    }
  }

  const auto cleanup = [&]() {
    (void)image.Free(&cuda, nullptr);
    (void)cuda.DestroyStream(stream, nullptr);
  };

  if (opts.csv)
    PrintCsvHeader();

  auto report = [&](const char *stage, const Stats &stats,
                    std::uint64_t checksum, const char *roundtripOk) {
    if (opts.csv) {
      PrintCsvResult(opts, stage, stats, checksum, roundtripOk);
    } else {
      PrintHumanResult(opts, stage, stats, checksum, roundtripOk);
      std::cout << "\n";
    }
  };

  Stats uploadStats;
  err.clear();
  if (!RunTimed(opts,
                [&]() {
                  return image.UploadFromCpuRgb24(&cuda, src.data(), stride,
                                                  stream, &err) &&
                         cuda.StreamSynchronize(stream, &err);
                },
                &uploadStats)) {
    std::cerr << "ERROR: upload benchmark failed: " << err << "\n";
    cleanup();
    return 1;
  }
  report("upload", uploadStats, Checksum(src), "na");

  err.clear();
  if (!image.UploadFromCpuRgb24(&cuda, src.data(), stride, stream, &err) ||
      !cuda.StreamSynchronize(stream, &err)) {
    std::cerr << "ERROR: preloading download benchmark failed: " << err << "\n";
    cleanup();
    return 1;
  }

  Stats downloadStats;
  err.clear();
  if (!RunTimed(opts,
                [&]() {
                  return image.DownloadToCpuRgb24(&cuda, dst.data(), stride,
                                                  stream, &err) &&
                         cuda.StreamSynchronize(stream, &err);
                },
                &downloadStats)) {
    std::cerr << "ERROR: download benchmark failed: " << err << "\n";
    cleanup();
    return 1;
  }
  const bool downloadOk = (src == dst);
  report("download", downloadStats, Checksum(dst), downloadOk ? "yes" : "no");

  Stats roundtripStats;
  err.clear();
  if (!RunTimed(opts,
                [&]() {
                  return image.UploadFromCpuRgb24(&cuda, src.data(), stride,
                                                  stream, &err) &&
                         image.DownloadToCpuRgb24(&cuda, dst.data(), stride,
                                                  stream, &err) &&
                         cuda.StreamSynchronize(stream, &err);
                },
                &roundtripStats)) {
    std::cerr << "ERROR: roundtrip benchmark failed: " << err << "\n";
    cleanup();
    return 1;
  }
  const bool roundtripOk = (src == dst);
  report("roundtrip", roundtripStats, Checksum(dst),
         roundtripOk ? "yes" : "no");

  cleanup();
  return (downloadOk && roundtripOk) ? 0 : 1;
}
