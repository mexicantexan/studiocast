#include "core/maxine/production_resident_frame_executor.h"
#include "maxine_effect_runtime_fake_control.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>

namespace {

using namespace studiocast::maxine;
using studiocast::video::effects::VirtualBackgroundMode;

std::atomic<uint64_t> g_allocations{0};
thread_local bool g_count_allocations = false;

void *Allocate(std::size_t size) {
  if (void *memory = std::malloc(size == 0 ? 1 : size)) {
    if (g_count_allocations)
      g_allocations.fetch_add(1, std::memory_order_relaxed);
    return memory;
  }
  throw std::bad_alloc();
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << __func__ << ':' << __LINE__                                 \
                << ": CHECK failed: " << #condition << '\n';                   \
      return false;                                                            \
    }                                                                          \
  } while (false)

uint64_t Counter(FakeMaxineCounter counter) {
  return StudioCastFakeMaxineCounter(static_cast<std::size_t>(counter));
}

uint64_t Transfers(const ProductionResidentTelemetry &telemetry,
                   ProductionTransferKind kind) {
  return telemetry.transfers[static_cast<std::size_t>(kind)].successes;
}

uint32_t Mask(std::initializer_list<ResidentStageKind> stages) {
  uint32_t mask = 0;
  for (const auto stage : stages)
    mask |= ProductionResidentStageBit(stage);
  return mask;
}

class FakeDeviceOps final : public IResidentDeviceOps {
public:
  bool Initialize(CudaDriverApi *, uintptr_t runtime_identity,
                  uint32_t required_ops_mask, std::string *) override {
    ++initialize_calls;
    if (last_runtime_identity != 0 && last_runtime_identity != runtime_identity)
      ++runtime_resets;
    last_runtime_identity = runtime_identity;
    last_required_ops = required_ops_mask;
    return !fail_initialize;
  }

  bool CropScale(const NvCVImage &, NvCVImage *, float, float, float, float,
                 CUstream, std::string *) noexcept override {
    ++crop_calls;
    return !fail_crop;
  }

  bool Resize(const NvCVImage &, NvCVImage *, CUstream,
              std::string *) noexcept override {
    ++resize_calls;
    return !fail_resize;
  }

  bool Vignette(NvCVImage *, float, float center_x, float center_y, CUstream,
                std::string *) noexcept override {
    ++vignette_calls;
    last_vignette_center_x = center_x;
    last_vignette_center_y = center_y;
    return !fail_vignette;
  }

  bool Synchronize(CUstream, std::string *) noexcept override {
    ++sync_calls;
    return !fail_sync;
  }

  uint64_t initialize_calls = 0;
  uint64_t runtime_resets = 0;
  uint64_t crop_calls = 0;
  uint64_t resize_calls = 0;
  uint64_t vignette_calls = 0;
  uint64_t sync_calls = 0;
  uintptr_t last_runtime_identity = 0;
  uint32_t last_required_ops = 0;
  float last_vignette_center_x = 0.0f;
  float last_vignette_center_y = 0.0f;
  bool fail_initialize = false;
  bool fail_crop = false;
  bool fail_resize = false;
  bool fail_vignette = false;
  bool fail_sync = false;
};

struct RuntimeFixture {
  vfx::VfxApi vfx;
  NvcvApi nvcv;
  ar::ArApi ar;

  bool Initialize() {
    std::string error;
    const std::filesystem::path sdk(STUDIOCAST_FAKE_MAXINE_SDK_PATH);
    CHECK(vfx.InitializeFromLibraryPath(sdk, &error));
    CHECK(nvcv.InitializeFromLibraryPath(NvcvApi::Requirement::VfxCompat, sdk,
                                         &error));
    CHECK(ar.InitializeFromLibraryPath(sdk, &error));
    return true;
  }

  ProductionResidentRuntime Runtime() {
    return ProductionResidentRuntime{&vfx, &nvcv, &ar, nullptr};
  }
};

ProductionResidentSetup Setup(uint32_t mask, uint64_t generation = 1,
                              uintptr_t runtime_identity = 0xA11) {
  ProductionResidentSetup setup{};
  setup.configuration_generation = generation;
  setup.width = 8;
  setup.height = 6;
  setup.output_width = 8;
  setup.output_height = 6;
  setup.runtime_identity = runtime_identity;
  setup.enabled_maxine_stage_mask = mask;
  setup.vfx_model_directory = "/fake/models";
  setup.vignette_center_x_px = 4.0f;
  setup.vignette_center_y_px = 3.0f;
  setup.effects.engine =
      studiocast::video::effects::EffectsEnginePreference::maxine;
  setup.effects.virtual_background.greenscreen_temporal = true;
  setup.effects.virtual_background.strength = 50;
  setup.effects.virtual_key_light.intensity = 100;
  setup.effects.virtual_key_light.hdri_path = "/fake/light.hdr";
  setup.effects.video_noise_removal.enabled = true;
  setup.effects.eye_contact.enabled = true;
  setup.effects.auto_frame.enabled = true;
  setup.effects.vignette.enabled = true;
  return setup;
}

struct HostFixture {
  std::array<uint8_t, 8 * 6 * 3> rgb{};
  HostRgbFrameView View() const {
    return HostRgbFrameView{rgb.data(), 8, 6, 8u * 3u};
  }
};

ResidentFramePlan CombinedPlan() {
  ResidentFramePlan plan{};
  if (!plan.AddCompatible(ResidentStageKind::background_blur, false, true,
                          0xCAFE) ||
      !plan.AddCompatible(ResidentStageKind::relighting, false, true, 0xCAFE))
    std::abort();
  return plan;
}

bool CombinedPathUsesActualOneUploadDownloadAndMatte() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  FakeDeviceOps ops{};
  ProductionResidentFrameExecutor executor(runtime.Runtime(), &ops);
  auto setup = Setup(Mask(
      {ResidentStageKind::background_blur, ResidentStageKind::relighting}));
  setup.effects.virtual_background.mode = VirtualBackgroundMode::blur;
  setup.effects.virtual_key_light.enabled = true;
  CHECK(executor.Configure(setup, nullptr));
  HostFixture host{};
  ResidentFrameSection section{};

  CHECK(
      section.Execute(CombinedPlan(), executor.key(), 1, host.View(), executor)
          .status == ResidentExecutionStatus::cpu_visible_output);
  StudioCastFakeMaxineResetCounters();
  executor.ResetTelemetry();
  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations = true;
  bool success = true;
  for (uint64_t i = 0; i < 128; ++i) {
    if (section
            .Execute(CombinedPlan(), executor.key(), i + 2, host.View(),
                     executor)
            .status != ResidentExecutionStatus::cpu_visible_output) {
      success = false;
      break;
    }
  }
  g_count_allocations = false;
  CHECK(success);
  const auto &t = executor.telemetry();
  CHECK(Transfers(t, ProductionTransferKind::host_upload) == 128);
  CHECK(Transfers(t, ProductionTransferKind::final_download) == 128);
  CHECK(Transfers(t, ProductionTransferKind::cpu_continuation_download) == 0);
  CHECK(Transfers(t, ProductionTransferKind::device_format_bridge) == 0);
  CHECK(t.matte_inferences.successes == 128);
  CHECK(t.shared_matte_reuses == 128);
  CHECK(t.composites.successes == 128);
  CHECK(t.explicit_synchronizations.successes == 128);
  CHECK(t.synchronous_sdk_runs.successes == 128 * 3);
  CHECK(t.asynchronous_sdk_runs.successes == 0);
  CHECK(Counter(FakeMaxineCounter::nvcv_transfer) == 128 * 2);
  CHECK(Counter(FakeMaxineCounter::nvcv_composite) == 128);
  CHECK(Counter(FakeMaxineCounter::vfx_run_sync) == 128 * 3);
  CHECK(Counter(FakeMaxineCounter::vfx_run_async) == 0);
  CHECK(Counter(FakeMaxineCounter::vfx_set_stream) == 0);
  CHECK(Counter(FakeMaxineCounter::vfx_set_image) == 0);
  CHECK(Counter(FakeMaxineCounter::vfx_set_f32) == 0);
  CHECK(g_allocations.load(std::memory_order_relaxed) == 0);
  CHECK(executor.host_output().Valid());
  return true;
}

bool ProductionIdentitiesAndForeignMatteFailClosed() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  FakeDeviceOps ops{};
  ProductionResidentFrameExecutor executor(runtime.Runtime(), &ops);
  auto setup = Setup(Mask(
      {ResidentStageKind::background_blur, ResidentStageKind::relighting}));
  setup.effects.virtual_background.mode = VirtualBackgroundMode::blur;
  setup.effects.virtual_key_light.enabled = true;
  CHECK(executor.Configure(setup, nullptr));
  const auto key = executor.key();
  HostFixture host{};
  CHECK(executor.Prepare(key) == ResidentBoundaryResult::success);
  CHECK(executor.StageRgbToBgr(host.View(), key) ==
        ResidentBoundaryResult::success);
  ResidentImage uploaded{};
  CHECK(executor.UploadStagedBgr(key, uploaded) ==
        ResidentBoundaryResult::success);
  ResidentMatte matte{};
  CHECK(executor.RunSharedMatte(uploaded, 9, 7, matte) ==
        ResidentBoundaryResult::success);
  CHECK(matte.source_image_identity == uploaded.image_identity);
  ResidentImage blurred{};
  CHECK(executor.RunCompatibleStage(ResidentStageKind::background_blur,
                                    uploaded, &matte, blurred) ==
        ResidentBoundaryResult::success);
  CHECK(blurred.image_identity != uploaded.image_identity);
  ResidentImage relit{};
  CHECK(executor.RunCompatibleStage(ResidentStageKind::relighting, blurred,
                                    &matte,
                                    relit) == ResidentBoundaryResult::success);
  CHECK(relit.image_identity != blurred.image_identity);

  ResidentMatte foreign = matte;
  ++foreign.matte_identity;
  ResidentImage rejected{};
  CHECK(executor.RunCompatibleStage(ResidentStageKind::relighting, blurred,
                                    &foreign, rejected) ==
        ResidentBoundaryResult::incompatible_output);
  CHECK(!rejected.image);
  CHECK(executor.RunCompatibleStage(ResidentStageKind::denoise, blurred,
                                    nullptr, rejected) ==
        ResidentBoundaryResult::incompatible_output);
  return true;
}

bool EveryResidentStageRunsOnTheRealExecutor() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  FakeDeviceOps ops{};
  const uint32_t mask =
      Mask({ResidentStageKind::denoise, ResidentStageKind::eye_contact,
            ResidentStageKind::background_blur, ResidentStageKind::relighting,
            ResidentStageKind::transfer, ResidentStageKind::auto_frame,
            ResidentStageKind::vignette});
  ProductionResidentFrameExecutor executor(runtime.Runtime(), &ops);
  auto setup = Setup(mask);
  setup.effects.virtual_background.mode = VirtualBackgroundMode::blur;
  setup.effects.virtual_key_light.enabled = true;
  CHECK(executor.Configure(setup, nullptr));
  ResidentFramePlan plan{};
  CHECK(plan.AddCompatible(ResidentStageKind::denoise, false));
  CHECK(plan.AddCompatible(ResidentStageKind::eye_contact, false));
  CHECK(plan.AddCompatible(ResidentStageKind::background_blur, false, true, 4));
  CHECK(plan.AddCompatible(ResidentStageKind::relighting, false, true, 4));
  CHECK(plan.AddCompatible(ResidentStageKind::transfer, false));
  CHECK(plan.AddCompatible(ResidentStageKind::auto_frame, false));
  CHECK(plan.AddCompatible(ResidentStageKind::vignette, false));
  HostFixture host{};
  ResidentFrameSection section{};
  const auto result =
      section.Execute(plan, executor.key(), 1, host.View(), executor);
  CHECK(result.status == ResidentExecutionStatus::cpu_visible_output);
  const auto &t = executor.telemetry();
  for (const auto kind :
       {ResidentStageKind::denoise, ResidentStageKind::eye_contact,
        ResidentStageKind::background_blur, ResidentStageKind::relighting,
        ResidentStageKind::transfer, ResidentStageKind::auto_frame,
        ResidentStageKind::vignette}) {
    CHECK(t.stages[static_cast<std::size_t>(kind)].successes == 1);
  }
  CHECK(Transfers(t, ProductionTransferKind::host_upload) == 1);
  CHECK(Transfers(t, ProductionTransferKind::final_download) == 1);
  CHECK(Transfers(t, ProductionTransferKind::device_format_bridge) == 3);
  CHECK(t.matte_inferences.successes == 1);
  CHECK(t.shared_matte_reuses == 1);
  CHECK(t.synchronous_sdk_runs.successes == 6);
  CHECK(t.asynchronous_sdk_runs.successes == 2);
  CHECK(ops.crop_calls == 1 && ops.vignette_calls == 1 && ops.sync_calls == 1);
  CHECK(ops.last_vignette_center_x == setup.vignette_center_x_px);
  CHECK(ops.last_vignette_center_y == setup.vignette_center_y_px);
  return true;
}

bool RemoveReplaceAndFinalResizeArePreparedAndResident() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  FakeDeviceOps ops{};
  HostFixture host{};

  {
    ProductionResidentFrameExecutor executor(runtime.Runtime(), &ops);
    auto setup = Setup(Mask({ResidentStageKind::background_remove}));
    setup.effects.virtual_background.mode = VirtualBackgroundMode::remove;
    setup.effects.virtual_background.remove_color = "#123456";
    CHECK(executor.Configure(setup, nullptr));
    CHECK(Transfers(executor.telemetry(),
                    ProductionTransferKind::background_asset_setup_upload) ==
          1);
    ResidentFramePlan plan{};
    CHECK(plan.AddCompatible(ResidentStageKind::background_remove, false, true,
                             1));
    ResidentFrameSection section{};
    CHECK(section.Execute(plan, executor.key(), 1, host.View(), executor)
              .status == ResidentExecutionStatus::cpu_visible_output);
    CHECK(executor.telemetry().composites.successes == 1);
  }

  std::array<uint8_t, 4 * 3 * 3> replacement{};
  {
    ProductionResidentFrameExecutor executor(runtime.Runtime(), &ops);
    auto setup = Setup(Mask({ResidentStageKind::background_replace}));
    setup.output_width = 4;
    setup.output_height = 3;
    setup.effects.virtual_background.mode = VirtualBackgroundMode::replace;
    setup.replacement_background =
        DecodedBackgroundRgbView{replacement.data(), 4, 3, 4u * 3u, 17};
    CHECK(executor.Configure(setup, nullptr));
    CHECK(Transfers(executor.telemetry(),
                    ProductionTransferKind::background_asset_setup_upload) ==
          1);
    CHECK(ops.resize_calls == 1);
    ResidentFramePlan plan{};
    CHECK(plan.AddCompatible(ResidentStageKind::background_replace, false, true,
                             1));
    ResidentFrameSection section{};
    CHECK(section.Execute(plan, executor.key(), 1, host.View(), executor)
              .status == ResidentExecutionStatus::cpu_visible_output);
    CHECK(ops.resize_calls == 2);
    CHECK(executor.host_output().width == 4 &&
          executor.host_output().height == 3);
  }
  return true;
}

bool BoundariesAndOptionalFailuresPreserveOriginalHost() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  FakeDeviceOps ops{};
  ProductionResidentFrameExecutor executor(runtime.Runtime(), &ops);
  auto setup = Setup(Mask(
      {ResidentStageKind::background_blur, ResidentStageKind::relighting}));
  setup.effects.virtual_background.mode = VirtualBackgroundMode::blur;
  setup.effects.virtual_key_light.enabled = true;
  CHECK(executor.Configure(setup, nullptr));
  const auto key = executor.key();
  HostFixture host{};
  host.rgb.fill(0x5A);
  const auto original = host.rgb;
  CHECK(executor.Prepare(key) == ResidentBoundaryResult::success);
  CHECK(executor.StageRgbToBgr(host.View(), key) ==
        ResidentBoundaryResult::success);
  StudioCastFakeMaxineFailNextNvcvTransfer();
  ResidentImage uploaded{};
  CHECK(executor.UploadStagedBgr(key, uploaded) ==
        ResidentBoundaryResult::runtime_failure);
  CHECK(host.rgb == original);

  CHECK(executor.UploadStagedBgr(key, uploaded) ==
        ResidentBoundaryResult::success);
  StudioCastFakeMaxineFailNextVfxRun();
  ResidentMatte matte{};
  CHECK(executor.RunSharedMatte(uploaded, 1, 2, matte) ==
        ResidentBoundaryResult::runtime_failure);
  CHECK(host.rgb == original);

  CHECK(executor.RunSharedMatte(uploaded, 1, 2, matte) ==
        ResidentBoundaryResult::success);
  ResidentImage blurred{};
  CHECK(executor.RunCompatibleStage(ResidentStageKind::background_blur,
                                    uploaded, &matte, blurred) ==
        ResidentBoundaryResult::success);
  StudioCastFakeMaxineFailNextNvcvComposite();
  ResidentImage prior = blurred;
  ResidentImage failed{};
  CHECK(executor.RunCompatibleStage(ResidentStageKind::relighting, prior,
                                    &matte, failed) ==
        ResidentBoundaryResult::runtime_failure);
  CHECK(!failed.image && host.rgb == original);

  CHECK(executor.DownloadToHost(ResidentReadbackBoundary::final_output,
                                prior) == ResidentBoundaryResult::success);
  ops.fail_sync = true;
  CHECK(executor.Synchronize(ResidentReadbackBoundary::final_output, key) ==
        ResidentBoundaryResult::runtime_failure);
  CHECK(!executor.host_output().Valid() && host.rgb == original);
  return true;
}

bool ReconfigurationInvalidationAndUnusedHelpersFailClosed() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  FakeDeviceOps ops{};
  ProductionResidentFrameExecutor executor(runtime.Runtime(), &ops);
  auto setup = Setup(Mask({ResidentStageKind::background_blur}));
  setup.effects.virtual_background.mode = VirtualBackgroundMode::blur;
  CHECK(executor.Configure(setup, nullptr));
  const auto stream = executor.stream_identity();
  const auto setup_successes = executor.telemetry().setup_successes;
  CHECK(executor.Configure(setup, nullptr));
  CHECK(executor.telemetry().setup_reuses == 1);

  setup.configuration_generation = 2;
  setup.effects.virtual_background.strength = 70;
  CHECK(executor.Configure(setup, nullptr));
  CHECK(executor.stream_identity() == stream);
  CHECK(executor.telemetry().setup_successes == setup_successes + 1);

  setup.runtime_identity = 0xB22;
  CHECK(executor.Configure(setup, nullptr));
  CHECK(ops.runtime_resets == 1);
  CHECK(executor.runtime_identity() == 0xB22);

  auto invalid = setup;
  invalid.enabled_maxine_stage_mask = 0;
  CHECK(!executor.Configure(invalid, nullptr));
  CHECK(!executor.key().Valid());
  invalid.enabled_maxine_stage_mask = uint32_t{1} << 31;
  CHECK(!executor.Configure(invalid, nullptr));
  CHECK(!executor.key().Valid());

  FakeDeviceOps unused{};
  ProductionResidentFrameExecutor vignette_executor(runtime.Runtime(), &unused);
  auto vignette_setup = Setup(Mask({ResidentStageKind::vignette}));
  CHECK(vignette_executor.Configure(vignette_setup, nullptr));
  CHECK((unused.last_required_ops &
         ResidentDeviceOpBit(ResidentDeviceOp::vignette)) != 0);
  CHECK((unused.last_required_ops &
         ResidentDeviceOpBit(ResidentDeviceOp::crop_scale)) == 0);
  CHECK((unused.last_required_ops &
         ResidentDeviceOpBit(ResidentDeviceOp::resize)) == 0);
  return true;
}

bool BeginFrame(ProductionResidentFrameExecutor &executor,
                const HostRgbFrameView &host, ResidentImage *uploaded) {
  CHECK(uploaded);
  const auto key = executor.key();
  CHECK(executor.Prepare(key) == ResidentBoundaryResult::success);
  CHECK(executor.StageRgbToBgr(host, key) == ResidentBoundaryResult::success);
  CHECK(executor.UploadStagedBgr(key, *uploaded) ==
        ResidentBoundaryResult::success);
  return true;
}

bool RemainingStageAndBoundaryFailuresFailOpen() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  HostFixture host{};

  for (const auto stage :
       {ResidentStageKind::denoise, ResidentStageKind::transfer}) {
    FakeDeviceOps ops{};
    ProductionResidentFrameExecutor executor(runtime.Runtime(), &ops);
    auto setup = Setup(Mask({stage}));
    CHECK(executor.Configure(setup, nullptr));
    ResidentImage current{};
    CHECK(BeginFrame(executor, host.View(), &current));
    StudioCastFakeMaxineFailNextVfxRun();
    ResidentImage output{};
    CHECK(executor.RunCompatibleStage(stage, current, nullptr, output) ==
          ResidentBoundaryResult::runtime_failure);
    CHECK(!output.image);
  }

  {
    FakeDeviceOps ops{};
    ProductionResidentFrameExecutor executor(runtime.Runtime(), &ops);
    auto setup = Setup(Mask({ResidentStageKind::eye_contact}));
    CHECK(executor.Configure(setup, nullptr));
    ResidentImage current{};
    CHECK(BeginFrame(executor, host.View(), &current));
    StudioCastFakeMaxineFailNextArRun();
    ResidentImage output{};
    CHECK(executor.RunCompatibleStage(ResidentStageKind::eye_contact, current,
                                      nullptr, output) ==
          ResidentBoundaryResult::runtime_failure);
    CHECK(!output.image);
  }

  {
    FakeDeviceOps ops{};
    ProductionResidentFrameExecutor executor(runtime.Runtime(), &ops);
    auto setup = Setup(Mask({ResidentStageKind::auto_frame}));
    CHECK(executor.Configure(setup, nullptr));
    ResidentImage current{};
    CHECK(BeginFrame(executor, host.View(), &current));
    StudioCastFakeMaxineFailNextArRun();
    ResidentImage output{};
    CHECK(executor.RunCompatibleStage(ResidentStageKind::auto_frame, current,
                                      nullptr, output) ==
          ResidentBoundaryResult::runtime_failure);
    CHECK(!output.image && ops.crop_calls == 0);

    executor.InvalidateBindings();
    CHECK(BeginFrame(executor, host.View(), &current));
    ops.fail_crop = true;
    CHECK(executor.RunCompatibleStage(ResidentStageKind::auto_frame, current,
                                      nullptr, output) ==
          ResidentBoundaryResult::runtime_failure);
    CHECK(!output.image);
  }

  {
    FakeDeviceOps ops{};
    ProductionResidentFrameExecutor executor(runtime.Runtime(), &ops);
    auto setup = Setup(Mask({ResidentStageKind::vignette}));
    CHECK(executor.Configure(setup, nullptr));
    ResidentImage current{};
    CHECK(BeginFrame(executor, host.View(), &current));
    ops.fail_vignette = true;
    ResidentImage output{};
    CHECK(executor.RunCompatibleStage(ResidentStageKind::vignette, current,
                                      nullptr, output) ==
          ResidentBoundaryResult::runtime_failure);
    CHECK(!output.image);
  }

  {
    FakeDeviceOps ops{};
    ProductionResidentFrameExecutor executor(runtime.Runtime(), &ops);
    auto setup = Setup(Mask({ResidentStageKind::background_blur}));
    setup.effects.virtual_background.mode = VirtualBackgroundMode::blur;
    CHECK(executor.Configure(setup, nullptr));
    ResidentImage current{};
    CHECK(BeginFrame(executor, host.View(), &current));
    ResidentMatte matte{};
    CHECK(executor.RunSharedMatte(current, 1, 2, matte) ==
          ResidentBoundaryResult::success);
    StudioCastFakeMaxineFailNextVfxRun();
    ResidentImage output{};
    CHECK(executor.RunCompatibleStage(ResidentStageKind::background_blur,
                                      current, &matte, output) ==
          ResidentBoundaryResult::runtime_failure);
    CHECK(!output.image);
  }

  {
    FakeDeviceOps ops{};
    ProductionResidentFrameExecutor executor(runtime.Runtime(), &ops);
    auto setup = Setup(Mask({ResidentStageKind::relighting}));
    setup.effects.virtual_key_light.enabled = true;
    CHECK(executor.Configure(setup, nullptr));
    ResidentImage current{};
    CHECK(BeginFrame(executor, host.View(), &current));
    ResidentMatte matte{};
    CHECK(executor.RunSharedMatte(current, 1, 2, matte) ==
          ResidentBoundaryResult::success);
    StudioCastFakeMaxineFailNextVfxRun();
    ResidentImage output{};
    CHECK(executor.RunCompatibleStage(ResidentStageKind::relighting, current,
                                      &matte, output) ==
          ResidentBoundaryResult::runtime_failure);
    CHECK(!output.image);
  }

  {
    FakeDeviceOps ops{};
    ProductionResidentFrameExecutor executor(runtime.Runtime(), &ops);
    auto setup = Setup(Mask({ResidentStageKind::transfer}));
    CHECK(executor.Configure(setup, nullptr));
    ResidentImage current{};
    CHECK(BeginFrame(executor, host.View(), &current));
    StudioCastFakeMaxineFailNextNvcvTransfer();
    CHECK(executor.DownloadToHost(ResidentReadbackBoundary::final_output,
                                  current) ==
          ResidentBoundaryResult::runtime_failure);
    CHECK(!executor.host_output().Valid());
    auto wrong_key = executor.key();
    ++wrong_key.configuration_generation;
    CHECK(executor.Prepare(wrong_key) ==
          ResidentBoundaryResult::incompatible_output);
    CHECK(executor.StageRgbToBgr({}, executor.key()) ==
          ResidentBoundaryResult::incompatible_output);
  }
  return true;
}

} // namespace

void *operator new(std::size_t size) { return Allocate(size); }
void *operator new[](std::size_t size) { return Allocate(size); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}

int main() {
  if (!CombinedPathUsesActualOneUploadDownloadAndMatte() ||
      !ProductionIdentitiesAndForeignMatteFailClosed() ||
      !EveryResidentStageRunsOnTheRealExecutor() ||
      !RemoveReplaceAndFinalResizeArePreparedAndResident() ||
      !BoundariesAndOptionalFailuresPreserveOriginalHost() ||
      !ReconfigurationInvalidationAndUnusedHelpersFailClosed() ||
      !RemainingStageAndBoundaryFailuresFailOpen())
    return EXIT_FAILURE;
  std::cout << "Maxine production resident executor tests passed\n";
  return EXIT_SUCCESS;
}
