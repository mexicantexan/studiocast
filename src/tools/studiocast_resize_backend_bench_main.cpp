#include "core/cuda/cuda_image.h"
#include "core/cuda/kernels/resize_bilinear.h"
#include "core/maxine/cuda_driver_api.h"
#include "core/video/image_ppm.h"

#if STUDIOCAST_ENABLE_OPEN_VULKAN
#include "core/vulkan/kernels/resize_bilinear.h"
#endif

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
  int warmup = 20;
  int iterations = 120;
  bool csv = false;
};

struct Case {
  int src_w = 0;
  int src_h = 0;
  int dst_w = 0;
  int dst_h = 0;
  const char *label = "";
};

struct Stats {
  double avg_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};

struct CudaBenchRuntime {
  studiocast::maxine::CudaDriverApi cuda;
  studiocast::maxine::CUstream stream = nullptr;
  bool initialized = false;
  bool available = false;
  std::string error;

  ~CudaBenchRuntime() {
    if (stream)
      (void)cuda.DestroyStream(stream, nullptr);
  }

  bool Ensure() {
    if (initialized)
      return available;
    initialized = true;
    if (!cuda.Initialize(&error) || !cuda.EnsureContext(&error)) {
      available = false;
      return false;
    }
    if (!cuda.CreateStream(&stream, &error)) {
      available = false;
      return false;
    }
    available = true;
    return true;
  }
};

bool ParseIntArg(const char *arg, int *out) {
  if (!arg || !out)
    return false;
  char *end = nullptr;
  const long value = std::strtol(arg, &end, 10);
  if (!end || *end != '\0' || value < 0 ||
      value > static_cast<long>(std::numeric_limits<int>::max())) {
    return false;
  }
  *out = static_cast<int>(value);
  return true;
}

void PrintUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " [--warmup N] [--iterations N] [--csv]\n";
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
    if (arg == "--warmup") {
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
  return opts->warmup >= 0 && opts->iterations > 0;
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

std::vector<std::uint8_t> MakeRgb(int width, int height) {
  const std::size_t stride = static_cast<std::size_t>(width) * 3u;
  std::vector<std::uint8_t> rgb(stride * static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t i =
          static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x) * 3u;
      rgb[i + 0] = static_cast<std::uint8_t>((x * 17 + y * 29) & 0xff);
      rgb[i + 1] = static_cast<std::uint8_t>((x * 31 + y * 7) & 0xff);
      rgb[i + 2] = static_cast<std::uint8_t>((x * 5 + y * 43) & 0xff);
    }
  }
  return rgb;
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

void PrintResult(const Options &opts, bool csv, const char *backend,
                 const Case &c, const Stats &stats, std::uint64_t checksum) {
  if (csv) {
    std::cout << backend << "," << c.label << "," << c.src_w << "x"
              << c.src_h << "," << c.dst_w << "x" << c.dst_h << ","
              << opts.warmup << "," << opts.iterations << ",ok,,"
              << stats.avg_ms << "," << stats.p95_ms << "," << stats.p99_ms
              << "," << stats.min_ms << "," << stats.max_ms << ","
              << checksum << "\n";
    return;
  }
  std::cout << backend << " " << c.label << " resize " << c.src_w << "x"
            << c.src_h << " -> " << c.dst_w << "x" << c.dst_h << "\n";
  std::cout << "  avg/p95/p99 : " << std::fixed << std::setprecision(4)
            << stats.avg_ms << " / " << stats.p95_ms << " / "
            << stats.p99_ms << " ms\n";
  std::cout << "  min/max     : " << stats.min_ms << " / " << stats.max_ms
            << " ms\n";
  std::cout << "  checksum    : " << checksum << "\n";
}

void PrintSkip(const Options &opts, bool csv, const char *backend,
               const Case &c, const std::string &reason) {
  if (csv) {
    std::cout << backend << "," << c.label << "," << c.src_w << "x" << c.src_h
              << "," << c.dst_w << "x" << c.dst_h << "," << opts.warmup
              << "," << opts.iterations << ",skipped," << CsvSafe(reason)
              << ",0,0,0,0,0,0\n";
  } else {
    std::cout << backend << " " << c.label << " skipped: " << reason << "\n";
  }
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

bool BenchCpu(const Options &opts, const Case &c, bool csv) {
  const std::size_t src_stride = static_cast<std::size_t>(c.src_w) * 3u;
  const std::size_t dst_stride = static_cast<std::size_t>(c.dst_w) * 3u;
  const auto src = MakeRgb(c.src_w, c.src_h);
  std::vector<std::uint8_t> dst;
  dst.reserve(dst_stride * static_cast<std::size_t>(c.dst_h));
  studiocast::video::Rgb24BilinearResizePlan plan;
  std::string err;
  if (!plan.Configure(c.src_w, c.src_h, c.dst_w, c.dst_h, &err)) {
    std::cerr << "ERROR: CPU resize configure failed: " << err << "\n";
    return false;
  }
  Stats stats;
  if (!RunTimed(opts,
                [&]() {
                  return plan.Apply(src.data(), src_stride, &dst, dst_stride,
                                    &err);
                },
                &stats)) {
    std::cerr << "ERROR: CPU resize failed: " << err << "\n";
    return false;
  }
  PrintResult(opts, csv, "cpu", c, stats, Checksum(dst));
  return true;
}

bool BenchCuda(const Options &opts, const Case &c, bool csv,
               CudaBenchRuntime *runtime) {
  if (!runtime || !runtime->Ensure()) {
    PrintSkip(opts, csv, "cuda", c,
              runtime ? runtime->error : std::string("unavailable"));
    return true;
  }
  std::string err;
  studiocast::cuda::CudaImage src_img;
  studiocast::cuda::CudaImage dst_img;
  if (!src_img.Allocate(&runtime->cuda, c.src_w, c.src_h,
                        studiocast::cuda::PixelFormatGpu::rgb_u8, &err) ||
      !dst_img.Allocate(&runtime->cuda, c.dst_w, c.dst_h,
                        studiocast::cuda::PixelFormatGpu::rgb_u8, &err)) {
    std::cerr << "ERROR: CUDA image allocation failed: " << err << "\n";
    (void)src_img.Free(&runtime->cuda, nullptr);
    (void)dst_img.Free(&runtime->cuda, nullptr);
    return false;
  }

  const std::size_t src_stride = static_cast<std::size_t>(c.src_w) * 3u;
  const std::size_t dst_stride = static_cast<std::size_t>(c.dst_w) * 3u;
  const auto src = MakeRgb(c.src_w, c.src_h);
  std::vector<std::uint8_t> dst(dst_stride * static_cast<std::size_t>(c.dst_h));
  Stats stats;
  const bool ok = RunTimed(
      opts,
      [&]() {
        return src_img.UploadFromCpuRgb24(&runtime->cuda, src.data(),
                                          src_stride, runtime->stream, &err) &&
               studiocast::cuda::kernels::ResizeBilinear(src_img, dst_img,
                                                         runtime->stream,
                                                         &err) &&
               dst_img.DownloadToCpuRgb24(&runtime->cuda, dst.data(),
                                          dst_stride, runtime->stream, &err) &&
               runtime->cuda.StreamSynchronize(runtime->stream, &err);
      },
      &stats);
  if (!ok) {
    std::cerr << "ERROR: CUDA resize failed: " << err << "\n";
  } else {
    PrintResult(opts, csv, "cuda", c, stats, Checksum(dst));
  }
  (void)src_img.Free(&runtime->cuda, nullptr);
  (void)dst_img.Free(&runtime->cuda, nullptr);
  return ok;
}

bool BenchVulkan(const Options &opts, const Case &c, bool csv) {
#if STUDIOCAST_ENABLE_OPEN_VULKAN
  studiocast::vulkan::kernels::ResizeBilinear resize;
  std::string err;
  if (!resize.EnsureInitialized(c.src_w, c.src_h, c.dst_w, c.dst_h, &err)) {
    PrintSkip(opts, csv, "vulkan", c, err);
    return true;
  }
  const std::size_t src_stride = static_cast<std::size_t>(c.src_w) * 3u;
  const std::size_t dst_stride = static_cast<std::size_t>(c.dst_w) * 3u;
  const auto src = MakeRgb(c.src_w, c.src_h);
  std::vector<std::uint8_t> dst(dst_stride * static_cast<std::size_t>(c.dst_h));
  Stats stats;
  const bool ok = RunTimed(
      opts,
      [&]() {
        return resize.Resize(src.data(), src_stride, dst.data(), dst_stride,
                             &err);
      },
      &stats);
  if (!ok) {
    std::cerr << "ERROR: Vulkan resize failed: " << err << "\n";
    return false;
  }
  PrintResult(opts, csv, "vulkan", c, stats, Checksum(dst));
#else
  PrintSkip(opts, csv, "vulkan", c, "backend disabled in build");
#endif
  return true;
}

} // namespace

int main(int argc, char **argv) {
  Options opts;
  if (!ParseArgs(argc, argv, &opts)) {
    PrintUsage(argv[0]);
    return 2;
  }

  const Case cases[] = {
      {1280, 720, 640, 360, "720p"},
      {1920, 1080, 1280, 720, "1080p"},
  };
  CudaBenchRuntime cuda;

  if (opts.csv) {
    std::cout << "backend,case,src,dst,warmup_iterations,iterations,status,"
                 "skip_reason,avg_ms,p95_ms,p99_ms,min_ms,max_ms,checksum\n";
  }

  bool ok = true;
  for (const auto &c : cases) {
    ok = BenchCpu(opts, c, opts.csv) && ok;
    ok = BenchCuda(opts, c, opts.csv, &cuda) && ok;
    ok = BenchVulkan(opts, c, opts.csv) && ok;
  }
  return ok ? 0 : 1;
}
