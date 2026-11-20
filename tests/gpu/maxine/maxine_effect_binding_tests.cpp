#include "core/maxine/ar_api.h"
#include "core/maxine/effects/ar_auto_frame_tracker.h"
#include "core/maxine/effects/ar_eye_contact_effect.h"
#include "core/maxine/effects/vfx_background_blur_effect.h"
#include "core/maxine/effects/vfx_denoise_effect.h"
#include "core/maxine/effects/vfx_green_screen_effect.h"
#include "core/maxine/effects/vfx_relighting_effect.h"
#include "core/maxine/effects/vfx_transfer_effect.h"
#include "core/maxine/nvcv_api.h"
#include "core/maxine/vfx_api.h"
#include "maxine_effect_runtime_fake_control.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <new>

#ifndef STUDIOCAST_FAKE_MAXINE_SDK_PATH
#error "STUDIOCAST_FAKE_MAXINE_SDK_PATH is required"
#endif

namespace {

using namespace studiocast;
using namespace studiocast::maxine;
using namespace studiocast::maxine::effects;

std::atomic<std::uint64_t> g_allocations{0};
thread_local bool g_count_allocations = false;

void *Allocate(std::size_t size) {
  if (void *memory = std::malloc(size == 0 ? 1 : size)) {
    return memory;
  }
  throw std::bad_alloc();
}

#define CHECK(condition)                                                        \
  do {                                                                          \
    if (!(condition)) {                                                         \
      std::cerr << __func__ << ':' << __LINE__ << ": CHECK failed: "           \
                << #condition << '\n';                                          \
      return false;                                                             \
    }                                                                           \
  } while (false)

std::uint64_t Counter(FakeMaxineCounter counter) {
  return StudioCastFakeMaxineCounter(static_cast<std::size_t>(counter));
}

struct RuntimeFixture {
  maxine::vfx::VfxApi vfx;
  maxine::NvcvApi nvcv;
  maxine::ar::ArApi ar;

  bool Initialize() {
    std::string error;
    const std::filesystem::path path(STUDIOCAST_FAKE_MAXINE_SDK_PATH);
    return vfx.InitializeFromLibraryPath(path, &error) &&
           nvcv.InitializeFromLibraryPath(NvcvApi::Requirement::VfxCompat,
                                          path, &error) &&
           ar.InitializeFromLibraryPath(path, &error);
  }
};

NvCVImage MakeImage(unsigned width = 8, unsigned height = 6,
                    NvCVImage_PixelFormat format = NVCV_BGR) {
  NvCVImage image{};
  image.width = width;
  image.height = height;
  image.pitch = static_cast<std::int32_t>(width * (format == NVCV_A ? 1 : 3));
  image.pixelFormat = format;
  image.componentType = NVCV_U8;
  image.pixelBytes = format == NVCV_A ? 1 : 3;
  image.componentBytes = 1;
  image.numComponents = format == NVCV_A ? 1 : 3;
  image.planar = NVCV_CHUNKY;
  image.gpuMem = NVCV_GPU;
  image.pixels = reinterpret_cast<void *>(0x1000);
  image.bufferBytes = static_cast<std::uint64_t>(image.pitch) * height;
  return image;
}

video::GpuFrame MakeFrame(NvCVImage *input, NvCVImage *matte = nullptr,
                          NvCVImage *output = nullptr,
                          CUstream stream = nullptr) {
  video::GpuFrame frame{};
  frame.width = static_cast<int>(input->width);
  frame.height = static_cast<int>(input->height);
  frame.nvcv_gpu = input;
  frame.matte_gpu = matte;
  frame.nvcv_tmp = output;
  frame.cuda_stream = stream;
  return frame;
}

video::effects::BroadcastCameraEffects Settings() {
  video::effects::BroadcastCameraEffects settings{};
  settings.virtual_background.strength = 32;
  settings.virtual_background.greenscreen_mode = 1;
  settings.virtual_background.greenscreen_temporal = true;
  settings.virtual_key_light.hdri_path = "/fake/light.hdr";
  settings.virtual_key_light.direction_pan_degrees = 10;
  settings.video_noise_removal.strength = 50;
  settings.eye_contact.look_away_enabled = true;
  settings.eye_contact.strength = 50;
  return settings;
}

bool NoVfxSetterCalls() {
  return Counter(FakeMaxineCounter::vfx_set_image) == 0 &&
         Counter(FakeMaxineCounter::vfx_set_string) == 0 &&
         Counter(FakeMaxineCounter::vfx_set_f32) == 0 &&
         Counter(FakeMaxineCounter::vfx_set_u32) == 0 &&
         Counter(FakeMaxineCounter::vfx_set_object) == 0 &&
         Counter(FakeMaxineCounter::vfx_set_state_array) == 0 &&
         Counter(FakeMaxineCounter::vfx_set_stream) == 0;
}

template <typename Effect>
bool RunStableVfx(Effect &effect, video::GpuFrame &frame,
                  const video::effects::BroadcastCameraEffects &settings) {
  CHECK(effect.Configure(settings, nullptr));
  CHECK(effect.Process(frame, nullptr) == NVCV_SUCCESS);
  StudioCastFakeMaxineResetCounters();
  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations = true;
  bool success = true;
  for (int i = 0; i < 128; ++i) {
    if (!effect.Configure(settings, nullptr) ||
        effect.Process(frame, nullptr) != NVCV_SUCCESS) {
      success = false;
      break;
    }
  }
  g_count_allocations = false;
  CHECK(success);
  CHECK(NoVfxSetterCalls());
  CHECK(Counter(FakeMaxineCounter::vfx_run) == 128);
  CHECK(g_allocations.load(std::memory_order_relaxed) == 0);
  return true;
}

bool AllVfxWrappersAreStableAfterWarmup() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  const auto settings = Settings();
  auto input = MakeImage();
  auto matte = MakeImage(8, 6, NVCV_A);
  auto frame = MakeFrame(&input, &matte);
  const auto external = reinterpret_cast<CUstream>(0x9000);

  {
    VfxGreenScreenEffect effect(&runtime.vfx, &runtime.nvcv, "/fake/models");
    CHECK(effect.SetExternalCudaStream(external, nullptr));
    CHECK(RunStableVfx(effect, frame, settings));
  }
  {
    VfxBackgroundBlurEffect effect(&runtime.vfx, &runtime.nvcv,
                                   "/fake/models");
    CHECK(effect.SetExternalCudaStream(external, nullptr));
    CHECK(RunStableVfx(effect, frame, settings));
  }
  {
    VfxRelightingEffect effect(&runtime.vfx, &runtime.nvcv, "/fake/models");
    CHECK(effect.SetExternalCudaStream(external, nullptr));
    CHECK(RunStableVfx(effect, frame, settings));
  }
  {
    VfxTransferEffect effect(&runtime.vfx, &runtime.nvcv, "/fake/models", {});
    CHECK(effect.SetExternalCudaStream(external, nullptr));
    CHECK(RunStableVfx(effect, frame, settings));
  }
  {
    VfxDenoiseEffect effect(&runtime.vfx, &runtime.nvcv, "/fake/models");
    CHECK(effect.SetExternalCudaStream(external, nullptr));
    CHECK(RunStableVfx(effect, frame, settings));
  }
  return true;
}

bool VfxPointerGeometryStreamAndFailureInvalidateOnce() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  auto settings = Settings();
  auto input = MakeImage();
  auto input2 = MakeImage();
  auto matte = MakeImage(8, 6, NVCV_A);
  auto matte2 = MakeImage(8, 6, NVCV_A);
  auto frame = MakeFrame(&input, &matte);
  VfxBackgroundBlurEffect effect(&runtime.vfx, &runtime.nvcv, "/fake/models");
  CHECK(effect.SetExternalCudaStream(reinterpret_cast<CUstream>(0x9000),
                                     nullptr));
  CHECK(effect.Configure(settings, nullptr));
  CHECK(effect.Process(frame, nullptr) == NVCV_SUCCESS);

  StudioCastFakeMaxineResetCounters();
  frame.nvcv_gpu = &input2;
  CHECK(effect.Process(frame, nullptr) == NVCV_SUCCESS);
  CHECK(Counter(FakeMaxineCounter::vfx_set_image) == 1);

  StudioCastFakeMaxineResetCounters();
  frame.matte_gpu = &matte2;
  CHECK(effect.Process(frame, nullptr) == NVCV_SUCCESS);
  CHECK(Counter(FakeMaxineCounter::vfx_set_image) == 1);

  StudioCastFakeMaxineResetCounters();
  CHECK(effect.SetExternalCudaStream(reinterpret_cast<CUstream>(0xA000),
                                     nullptr));
  CHECK(Counter(FakeMaxineCounter::vfx_set_stream) == 1);
  CHECK(effect.SetExternalCudaStream(reinterpret_cast<CUstream>(0xA000),
                                     nullptr));
  CHECK(Counter(FakeMaxineCounter::vfx_set_stream) == 1);

  StudioCastFakeMaxineResetCounters();
  settings.virtual_background.strength = 48;
  CHECK(effect.Configure(settings, nullptr));
  CHECK(Counter(FakeMaxineCounter::vfx_set_f32) == 1);
  CHECK(effect.Configure(settings, nullptr));
  CHECK(Counter(FakeMaxineCounter::vfx_set_f32) == 1);

  auto larger_input = MakeImage(10, 8);
  auto larger_matte = MakeImage(10, 8, NVCV_A);
  frame = MakeFrame(&larger_input, &larger_matte);
  StudioCastFakeMaxineResetCounters();
  CHECK(effect.Process(frame, nullptr) == NVCV_SUCCESS);
  CHECK(Counter(FakeMaxineCounter::nvcv_alloc) +
            Counter(FakeMaxineCounter::nvcv_realloc) >=
        1);
  CHECK(Counter(FakeMaxineCounter::vfx_set_image) == 3);

  StudioCastFakeMaxineResetCounters();
  StudioCastFakeMaxineFailNextVfxRun();
  CHECK(effect.Process(frame, nullptr) != NVCV_SUCCESS);
  CHECK(!effect.OutputGpu());
  StudioCastFakeMaxineResetCounters();
  CHECK(effect.Process(frame, nullptr) == NVCV_SUCCESS);
  CHECK(Counter(FakeMaxineCounter::vfx_set_stream) == 1);
  CHECK(Counter(FakeMaxineCounter::vfx_set_image) == 3);
  return true;
}

bool OwnedAndExternalVfxStreamsHaveCorrectLifetime() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  {
    StudioCastFakeMaxineResetCounters();
    VfxTransferEffect owned(&runtime.vfx, &runtime.nvcv, "/fake/models", {});
    CHECK(owned.Initialize(nullptr));
    CHECK(Counter(FakeMaxineCounter::vfx_stream_create) == 1);
  }
  CHECK(Counter(FakeMaxineCounter::vfx_stream_destroy) == 1);

  StudioCastFakeMaxineResetCounters();
  {
    VfxTransferEffect external(&runtime.vfx, &runtime.nvcv, "/fake/models", {});
    CHECK(external.SetExternalCudaStream(reinterpret_cast<CUstream>(0x9000),
                                         nullptr));
    CHECK(external.Initialize(nullptr));
    CHECK(Counter(FakeMaxineCounter::vfx_stream_create) == 0);
  }
  CHECK(Counter(FakeMaxineCounter::vfx_stream_destroy) == 0);
  return true;
}

bool ExternalStreamRejectionFailsClosedAndPreservesOwnership() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());

  StudioCastFakeMaxineResetCounters();
  {
    VfxTransferEffect retry(&runtime.vfx, &runtime.nvcv, "/fake/models", {});
    CHECK(retry.SetExternalCudaStream(reinterpret_cast<CUstream>(0x9000),
                                      nullptr));
    StudioCastFakeMaxineFailNextVfxSetStream();
    CHECK(!retry.Initialize(nullptr));
    CHECK(Counter(FakeMaxineCounter::vfx_stream_create) == 0);
    CHECK(Counter(FakeMaxineCounter::vfx_stream_destroy) == 0);
    CHECK(retry.Initialize(nullptr));
    CHECK(Counter(FakeMaxineCounter::vfx_stream_create) == 0);
  }
  CHECK(Counter(FakeMaxineCounter::vfx_stream_destroy) == 0);

  StudioCastFakeMaxineResetCounters();
  {
    VfxTransferEffect owned(&runtime.vfx, &runtime.nvcv, "/fake/models", {});
    CHECK(owned.Initialize(nullptr));
    StudioCastFakeMaxineFailNextVfxSetStream();
    CHECK(!owned.SetExternalCudaStream(reinterpret_cast<CUstream>(0xA000),
                                       nullptr));
    CHECK(owned.cuda_stream() != reinterpret_cast<CUstream>(0xA000));
  }
  CHECK(Counter(FakeMaxineCounter::vfx_stream_destroy) == 1);
  return true;
}

bool ArStreamRejectionDistinguishesStandaloneAndExternal() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  auto input = MakeImage();
  auto output = MakeImage();

  StudioCastFakeMaxineResetCounters();
  {
    auto standalone_frame = MakeFrame(&input, nullptr, &output);
    ArEyeContactEffect standalone(&runtime.ar);
    StudioCastFakeMaxineFailNextArSetStream();
    CHECK(standalone.Process(standalone_frame, nullptr) == NVCV_SUCCESS);
    CHECK(Counter(FakeMaxineCounter::ar_stream_create) == 1);
    CHECK(Counter(FakeMaxineCounter::ar_set_stream) == 1);
    CHECK(standalone.Process(standalone_frame, nullptr) == NVCV_SUCCESS);
    CHECK(Counter(FakeMaxineCounter::ar_set_stream) == 1);
  }
  CHECK(Counter(FakeMaxineCounter::ar_stream_destroy) == 1);

  StudioCastFakeMaxineResetCounters();
  {
    auto external_frame = MakeFrame(&input, nullptr, &output,
                                    reinterpret_cast<CUstream>(0x9000));
    ArEyeContactEffect external(&runtime.ar);
    StudioCastFakeMaxineFailNextArSetStream();
    CHECK(external.Process(external_frame, nullptr) != NVCV_SUCCESS);
    CHECK(Counter(FakeMaxineCounter::ar_run) == 0);
    CHECK(Counter(FakeMaxineCounter::ar_stream_create) == 0);
    CHECK(external.Process(external_frame, nullptr) == NVCV_SUCCESS);
  }
  CHECK(Counter(FakeMaxineCounter::ar_stream_destroy) == 0);

  StudioCastFakeMaxineResetCounters();
  ArAutoFrameTracker tracker(&runtime.ar);
  CHECK(tracker.SetExternalCudaStream(reinterpret_cast<CUstream>(0xA000),
                                      nullptr));
  StudioCastFakeMaxineFailNextArSetStream();
  CHECK(!tracker.EnsureInitialized(&input, nullptr));
  CHECK(tracker.EnsureInitialized(&input, nullptr));
  CHECK(Counter(FakeMaxineCounter::ar_stream_create) == 0);
  CHECK(Counter(FakeMaxineCounter::ar_stream_destroy) == 0);
  return true;
}

bool EyeContactCachesStreamObjectsAndConfiguration() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  auto settings = Settings();
  auto input = MakeImage();
  auto output = MakeImage();
  auto frame = MakeFrame(&input, nullptr, &output,
                         reinterpret_cast<CUstream>(0x9000));
  ArEyeContactEffect effect(&runtime.ar);
  CHECK(effect.Configure(settings, nullptr));
  CHECK(effect.Process(frame, nullptr) == NVCV_SUCCESS);

  StudioCastFakeMaxineResetCounters();
  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations = true;
  bool success = true;
  for (int i = 0; i < 128; ++i) {
    if (!effect.Configure(settings, nullptr) ||
        effect.Process(frame, nullptr) != NVCV_SUCCESS) {
      success = false;
      break;
    }
  }
  g_count_allocations = false;
  CHECK(success);
  CHECK(Counter(FakeMaxineCounter::ar_run) == 128);
  CHECK(Counter(FakeMaxineCounter::ar_set_stream) == 0);
  CHECK(Counter(FakeMaxineCounter::ar_set_object) == 0);
  CHECK(Counter(FakeMaxineCounter::ar_set_u32) == 0);
  CHECK(Counter(FakeMaxineCounter::ar_set_f32) == 0);
  CHECK(g_allocations.load(std::memory_order_relaxed) == 0);

  StudioCastFakeMaxineResetCounters();
  settings.eye_contact.strength = 80;
  CHECK(effect.Configure(settings, nullptr));
  CHECK(effect.Process(frame, nullptr) == NVCV_SUCCESS);
  CHECK(Counter(FakeMaxineCounter::ar_set_u32) == 2);
  CHECK(Counter(FakeMaxineCounter::ar_set_f32) == 3);
  CHECK(effect.Process(frame, nullptr) == NVCV_SUCCESS);
  CHECK(Counter(FakeMaxineCounter::ar_set_u32) == 2);

  StudioCastFakeMaxineResetCounters();
  StudioCastFakeMaxineFailNextArRun();
  CHECK(effect.Process(frame, nullptr) != NVCV_SUCCESS);
  StudioCastFakeMaxineResetCounters();
  CHECK(effect.Process(frame, nullptr) == NVCV_SUCCESS);
  CHECK(Counter(FakeMaxineCounter::ar_set_stream) == 1);
  CHECK(Counter(FakeMaxineCounter::ar_set_object) == 2);
  return true;
}

bool AutoFrameCachesInputOutputsAndExternalStream() {
  RuntimeFixture runtime{};
  CHECK(runtime.Initialize());
  auto input = MakeImage();
  ArAutoFrameTracker tracker(&runtime.ar);
  CHECK(tracker.SetExternalCudaStream(reinterpret_cast<CUstream>(0x9000),
                                      nullptr));
  CHECK(tracker.EnsureInitialized(&input, nullptr));
  StudioCastFakeMaxineResetCounters();
  g_allocations.store(0, std::memory_order_relaxed);
  g_count_allocations = true;
  bool success = true;
  for (int i = 0; i < 128; ++i) {
    if (!tracker.EnsureInitialized(&input, nullptr) ||
        !tracker.Update(8, 6, nullptr)) {
      success = false;
      break;
    }
  }
  g_count_allocations = false;
  CHECK(success);
  CHECK(Counter(FakeMaxineCounter::ar_set_stream) == 0);
  CHECK(Counter(FakeMaxineCounter::ar_set_object) == 0);
  CHECK(Counter(FakeMaxineCounter::ar_set_f32_array) == 0);
  CHECK(g_allocations.load(std::memory_order_relaxed) == 0);

  StudioCastFakeMaxineResetCounters();
  auto changed = MakeImage();
  CHECK(tracker.EnsureInitialized(&changed, nullptr));
  CHECK(Counter(FakeMaxineCounter::ar_set_object) == 2);
  CHECK(tracker.SetExternalCudaStream(reinterpret_cast<CUstream>(0xA000),
                                      nullptr));
  CHECK(Counter(FakeMaxineCounter::ar_set_stream) == 2);
  CHECK(tracker.SetExternalCudaStream(reinterpret_cast<CUstream>(0xA000),
                                      nullptr));
  CHECK(Counter(FakeMaxineCounter::ar_set_stream) == 2);

  StudioCastFakeMaxineFailNextArRun();
  CHECK(!tracker.Update(8, 6, nullptr));
  StudioCastFakeMaxineResetCounters();
  CHECK(tracker.EnsureInitialized(&changed, nullptr));
  CHECK(Counter(FakeMaxineCounter::ar_set_stream) == 2);
  CHECK(Counter(FakeMaxineCounter::ar_set_object) == 2);
  return true;
}

} // namespace

void *operator new(std::size_t size) {
  if (g_count_allocations)
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  return Allocate(size);
}
void *operator new[](std::size_t size) {
  if (g_count_allocations)
    g_allocations.fetch_add(1, std::memory_order_relaxed);
  return Allocate(size);
}
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }

int main() {
  const std::array tests{
      AllVfxWrappersAreStableAfterWarmup,
      VfxPointerGeometryStreamAndFailureInvalidateOnce,
      OwnedAndExternalVfxStreamsHaveCorrectLifetime,
      ExternalStreamRejectionFailsClosedAndPreservesOwnership,
      ArStreamRejectionDistinguishesStandaloneAndExternal,
      EyeContactCachesStreamObjectsAndConfiguration,
      AutoFrameCachesInputOutputsAndExternalStream,
  };
  for (const auto test : tests) {
    if (!test())
      return EXIT_FAILURE;
  }
  std::cout << "maxine effect binding tests passed\n";
  return EXIT_SUCCESS;
}
