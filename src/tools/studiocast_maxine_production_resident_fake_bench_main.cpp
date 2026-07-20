#include "core/maxine/production_resident_frame_executor.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using namespace studiocast::maxine;
using studiocast::video::effects::VirtualBackgroundMode;

struct Options {
  uint64_t warmup = 10;
  uint64_t iterations = 120;
};

bool ParseUnsigned(std::string_view text, uint64_t *value) {
  if (!value)
    return false;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), *value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size() &&
         *value != 0;
}

bool ParseOptions(int argc, char **argv, Options *options) {
  if (!options)
    return false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if ((argument == "--warmup" || argument == "--iterations") &&
        i + 1 < argc) {
      uint64_t value = 0;
      if (!ParseUnsigned(argv[++i], &value))
        return false;
      if (argument == "--warmup")
        options->warmup = value;
      else
        options->iterations = value;
      continue;
    }
    return false;
  }
  return true;
}

class FakeDeviceOps final : public IResidentDeviceOps {
public:
  bool Initialize(CudaDriverApi *, uintptr_t, uint32_t,
                  std::string *) override {
    return true;
  }
  bool CropScale(const NvCVImage &, NvCVImage *, float, float, float, float,
                 CUstream, std::string *) noexcept override {
    return true;
  }
  bool Resize(const NvCVImage &, NvCVImage *, CUstream,
              std::string *) noexcept override {
    return true;
  }
  bool Vignette(NvCVImage *, float, float, float, CUstream,
                std::string *) noexcept override {
    return true;
  }
  bool Synchronize(CUstream, std::string *) noexcept override { return true; }
};

struct RuntimeFixture {
  vfx::VfxApi vfx;
  NvcvApi nvcv;
  ar::ArApi ar;

  bool Initialize() {
    std::string error;
    const std::filesystem::path sdk(STUDIOCAST_FAKE_MAXINE_SDK_PATH);
    return vfx.InitializeFromLibraryPath(sdk, &error) &&
           nvcv.InitializeFromLibraryPath(NvcvApi::Requirement::VfxCompat, sdk,
                                          &error) &&
           ar.InitializeFromLibraryPath(sdk, &error);
  }
};

uint64_t Successes(const ProductionResidentTelemetry &telemetry,
                   ProductionTransferKind kind) {
  return telemetry.transfers[static_cast<std::size_t>(kind)].successes;
}

double Percentile(const std::vector<double> &sorted, double fraction) {
  const auto index = static_cast<std::size_t>(
      fraction * static_cast<double>(sorted.size() - 1));
  return sorted[index];
}

bool RunMode(RuntimeFixture *runtime, std::string_view mode, uint32_t width,
             uint32_t height, const Options &options) {
  if (!runtime)
    return false;
  const bool combined = mode == "combined_blur_relight";
  const uint32_t mask =
      combined
          ? ProductionResidentStageBit(ResidentStageKind::background_blur) |
                ProductionResidentStageBit(ResidentStageKind::relighting)
          : ProductionResidentStageBit(ResidentStageKind::denoise);
  ProductionResidentSetup setup{};
  setup.configuration_generation = 1;
  setup.width = width;
  setup.height = height;
  setup.output_width = width;
  setup.output_height = height;
  setup.runtime_identity = 0xA11;
  setup.enabled_maxine_stage_mask = mask;
  setup.vfx_model_directory = "/fake/models";
  setup.effects.video_noise_removal.enabled = !combined;
  setup.effects.video_noise_removal.strength = 50;
  setup.effects.virtual_background.mode =
      combined ? VirtualBackgroundMode::blur : VirtualBackgroundMode::none;
  setup.effects.virtual_background.strength = 50;
  setup.effects.virtual_key_light.enabled = combined;
  setup.effects.virtual_key_light.intensity = 100;
  setup.effects.virtual_key_light.hdri_path = "/fake/light.hdr";

  FakeDeviceOps ops{};
  ProductionResidentFrameExecutor executor(
      ProductionResidentRuntime{&runtime->vfx, &runtime->nvcv, &runtime->ar,
                                nullptr},
      &ops);
  if (!executor.Configure(setup, nullptr))
    return false;

  ResidentFramePlan plan{};
  if (combined) {
    if (!plan.AddCompatible(ResidentStageKind::background_blur, false, true,
                            0xCAFE) ||
        !plan.AddCompatible(ResidentStageKind::relighting, false, true, 0xCAFE))
      return false;
  } else if (!plan.AddCompatible(ResidentStageKind::denoise, false)) {
    return false;
  }

  const std::size_t row_bytes = static_cast<std::size_t>(width) * 3u;
  std::vector<uint8_t> rgb(row_bytes * static_cast<std::size_t>(height), 0x55);
  const HostRgbFrameView host{rgb.data(), width, height, row_bytes};
  ResidentFrameSection section{};
  for (uint64_t i = 0; i < options.warmup; ++i) {
    if (section.Execute(plan, executor.key(), i + 1, host, executor).status !=
        ResidentExecutionStatus::cpu_visible_output)
      return false;
  }
  const auto before = executor.telemetry();
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(options.iterations));
  double total_ms = 0.0;
  double min_ms = std::numeric_limits<double>::max();
  double max_ms = 0.0;
  for (uint64_t i = 0; i < options.iterations; ++i) {
    const auto start = Clock::now();
    const auto result = section.Execute(plan, executor.key(),
                                        options.warmup + i + 1, host, executor);
    const auto finish = Clock::now();
    if (result.status != ResidentExecutionStatus::cpu_visible_output)
      return false;
    const double elapsed =
        std::chrono::duration<double, std::milli>(finish - start).count();
    samples.push_back(elapsed);
    total_ms += elapsed;
    min_ms = std::min(min_ms, elapsed);
    max_ms = std::max(max_ms, elapsed);
  }
  std::sort(samples.begin(), samples.end());
  const auto after = executor.telemetry();
  const double frames = static_cast<double>(options.iterations);
  const auto per_frame = [frames](uint64_t value) {
    return static_cast<double>(value) / frames;
  };
  const double avg = total_ms / frames;
  std::cout
      << std::fixed << std::setprecision(6) << "backend=fake_exact_abi"
      << " synthetic=true"
      << " mode=" << mode << " input=rgb24 output=bgr_u8"
      << " geometry=" << width << 'x' << height << " warmup=" << options.warmup
      << " iterations=" << options.iterations << " avg_ms=" << avg
      << " p95_ms=" << Percentile(samples, 0.95)
      << " p99_ms=" << Percentile(samples, 0.99) << " min_ms=" << min_ms
      << " max_ms=" << max_ms << " uploads_per_frame="
      << per_frame(Successes(after, ProductionTransferKind::host_upload) -
                   Successes(before, ProductionTransferKind::host_upload))
      << " final_downloads_per_frame="
      << per_frame(Successes(after, ProductionTransferKind::final_download) -
                   Successes(before, ProductionTransferKind::final_download))
      << " continuation_downloads_per_frame="
      << per_frame(
             Successes(after,
                       ProductionTransferKind::cpu_continuation_download) -
             Successes(before,
                       ProductionTransferKind::cpu_continuation_download))
      << " device_bridges_per_frame="
      << per_frame(
             Successes(after, ProductionTransferKind::device_format_bridge) -
             Successes(before, ProductionTransferKind::device_format_bridge))
      << " matte_inferences_per_frame="
      << per_frame(after.matte_inferences.successes -
                   before.matte_inferences.successes)
      << " explicit_syncs_per_frame="
      << per_frame(after.explicit_synchronizations.successes -
                   before.explicit_synchronizations.successes)
      << " sdk_sync_runs_per_frame="
      << per_frame(after.synchronous_sdk_runs.successes -
                   before.synchronous_sdk_runs.successes)
      << " sdk_async_runs_per_frame="
      << per_frame(after.asynchronous_sdk_runs.successes -
                   before.asynchronous_sdk_runs.successes)
      << " cpu_control_stages_per_frame="
      << per_frame(after
                       .cpu_stages[static_cast<std::size_t>(
                           ProductionCpuStageKind::rgb_to_bgr_staging)]
                       .successes -
                   before
                       .cpu_stages[static_cast<std::size_t>(
                           ProductionCpuStageKind::rgb_to_bgr_staging)]
                       .successes)
      << " synthetic_frame_age_avg_ms=" << (2.0 + avg) << '\n';
  return true;
}

} // namespace

int main(int argc, char **argv) {
  Options options{};
  if (!ParseOptions(argc, argv, &options)) {
    std::cerr << "usage: studiocast-maxine-production-resident-fake-bench "
                 "[--warmup N] [--iterations N]\n";
    return EXIT_FAILURE;
  }
  RuntimeFixture runtime{};
  if (!runtime.Initialize())
    return EXIT_FAILURE;
  constexpr std::array geometries = {std::pair<uint32_t, uint32_t>{1280, 720},
                                     std::pair<uint32_t, uint32_t>{1920, 1080}};
  for (const auto &[width, height] : geometries) {
    if (!RunMode(&runtime, "single_denoise", width, height, options) ||
        !RunMode(&runtime, "combined_blur_relight", width, height, options))
      return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
