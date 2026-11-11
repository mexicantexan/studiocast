#include "core/cuda/kernels/prepared_mirror.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace studiocast::cuda;
using namespace studiocast::cuda::kernels;

struct State {
  CUcontext current = nullptr;
  CUresult load_result = CUDA_SUCCESS;
  CUresult resolve_result = CUDA_SUCCESS;
  CUresult unload_result = CUDA_SUCCESS;
  CUresult launch_result = CUDA_SUCCESS;
  int loads = 0, resolves = 0, unloads = 0, launches = 0;
  int allocations = 0, copies = 0, synchronizations = 0;
  CUstream stream = nullptr;
  unsigned grid_x = 0, grid_y = 0;
  unsigned long long source = 0, source_pitch = 0, destination = 0,
                     destination_pitch = 0;
  std::uint32_t width = 0, height = 0;
};
State *state = nullptr;

bool Check(bool value, std::string_view message) {
  if (!value)
    std::cerr << "[FAIL] " << message << '\n';
  return value;
}
CUresult Init(unsigned) { return CUDA_SUCCESS; }
CUresult Load(CUmodule *module, const void *) {
  ++state->loads;
  if (state->load_result == CUDA_SUCCESS)
    *module = reinterpret_cast<CUmodule>(std::uintptr_t{0x10});
  return state->load_result;
}
CUresult Resolve(CUfunction *function, CUmodule, const char *name) {
  ++state->resolves;
  if (state->resolve_result != CUDA_SUCCESS ||
      std::string_view(name) != "mirror_horizontal_u8x3")
    return 1;
  *function = reinterpret_cast<CUfunction>(std::uintptr_t{0x20});
  return CUDA_SUCCESS;
}
CUresult Unload(CUmodule) {
  ++state->unloads;
  return state->unload_result;
}
CUresult Launch(CUfunction, unsigned gx, unsigned gy, unsigned, unsigned,
                unsigned, unsigned, unsigned, CUstream stream, void **args,
                void **) {
  ++state->launches;
  state->grid_x = gx;
  state->grid_y = gy;
  state->stream = stream;
  state->source = *static_cast<unsigned long long *>(args[0]);
  state->source_pitch = *static_cast<unsigned long long *>(args[1]);
  state->destination = *static_cast<unsigned long long *>(args[2]);
  state->destination_pitch = *static_cast<unsigned long long *>(args[3]);
  state->width = *static_cast<std::uint32_t *>(args[4]);
  state->height = *static_cast<std::uint32_t *>(args[5]);
  return state->launch_result;
}
CUresult Alloc(CUdeviceptr *ptr, std::size_t *pitch, std::size_t width,
               std::size_t, unsigned) {
  ++state->allocations;
  *ptr = 0x1000;
  *pitch = width;
  return CUDA_SUCCESS;
}
CUresult Free(CUdeviceptr) { return CUDA_SUCCESS; }
CUresult Copy(const CUDA_MEMCPY2D *, CUstream) {
  ++state->copies;
  return CUDA_SUCCESS;
}
CUresult Memset(CUdeviceptr, unsigned char, std::size_t, CUstream) {
  return CUDA_SUCCESS;
}
CUresult StreamCreate(CUstream *out, unsigned) {
  *out = reinterpret_cast<CUstream>(std::uintptr_t{0x30});
  return CUDA_SUCCESS;
}
CUresult StreamDestroy(CUstream) { return CUDA_SUCCESS; }
CUresult StreamSync(CUstream) {
  ++state->synchronizations;
  return CUDA_SUCCESS;
}
CUresult DeviceCount(int *count) {
  *count = 1;
  return CUDA_SUCCESS;
}
CUresult DeviceGet(CUdevice *device, int ordinal) {
  *device = ordinal;
  return CUDA_SUCCESS;
}
CUresult DeviceUuid(CUuuid *uuid, CUdevice) {
  std::memset(uuid, 1, sizeof(*uuid));
  return CUDA_SUCCESS;
}
CUresult DevicePci(char *value, int size, CUdevice) {
  constexpr char text[] = "0000:01:00.0";
  if (size < static_cast<int>(sizeof(text)))
    return 1;
  std::memcpy(value, text, sizeof(text));
  return CUDA_SUCCESS;
}
CUresult PrimaryRetain(CUcontext *context, CUdevice) {
  *context = reinterpret_cast<CUcontext>(std::uintptr_t{0x40});
  return CUDA_SUCCESS;
}
CUresult PrimaryRelease(CUdevice) { return CUDA_SUCCESS; }
CUresult CtxSet(CUcontext context) {
  state->current = context;
  return CUDA_SUCCESS;
}
CUresult CtxGet(CUcontext *context) {
  *context = state->current;
  return CUDA_SUCCESS;
}
CUresult CtxDevice(CUdevice *device) {
  *device = 0;
  return CUDA_SUCCESS;
}

CudaDriverApi Driver(State *storage) {
  state = storage;
  CudaDriverApi::Functions f;
  f.cuInit = &Init;
  f.cuModuleLoadData = &Load;
  f.cuModuleGetFunction = &Resolve;
  f.cuModuleUnload = &Unload;
  f.cuLaunchKernel = &Launch;
  f.cuMemAllocPitch = &Alloc;
  f.cuMemFree = &Free;
  f.cuMemcpy2DAsync = &Copy;
  f.cuMemsetD8Async = &Memset;
  f.cuStreamCreate = &StreamCreate;
  f.cuStreamDestroy = &StreamDestroy;
  f.cuStreamSynchronize = &StreamSync;
  f.cuDeviceGetCount = &DeviceCount;
  f.cuDeviceGet = &DeviceGet;
  f.cuDeviceGetUuid = &DeviceUuid;
  f.cuDeviceGetPCIBusId = &DevicePci;
  f.cuDevicePrimaryCtxRetain = &PrimaryRetain;
  f.cuDevicePrimaryCtxRelease = &PrimaryRelease;
  f.cuCtxSetCurrent = &CtxSet;
  f.cuCtxGetCurrent = &CtxGet;
  f.cuCtxGetDevice = &CtxDevice;
  CudaDriverApi driver;
  std::string error;
  (void)driver.InitializeForTesting(f, &error);
  return driver;
}

CudaImage Image(CUdeviceptr ptr, int width, int height, std::size_t pitch,
                PixelFormatGpu format = PixelFormatGpu::rgb_u8) {
  CudaImage image;
  image.ptr = ptr;
  image.pitch = pitch;
  image.w = width;
  image.h = height;
  image.format = format;
  return image;
}

bool FakeContract() {
  State storage;
  auto driver = Driver(&storage);
  const auto context = reinterpret_cast<CUcontext>(std::uintptr_t{0x40});
  const auto other_context = reinterpret_cast<CUcontext>(std::uintptr_t{0x41});
  const auto stream = reinterpret_cast<CUstream>(std::uintptr_t{0x30});
  storage.current = context;
  PreparedMirrorKernel kernel;
  std::string error;
  bool ok = Check(PrepareMirrorKernel(driver, context, stream, &kernel, &error),
                  "prepare") &&
            Check(storage.loads == 1 && storage.resolves == 1,
                  "prepare loads and resolves once");
  auto source =
      Image(0x100000000ull, 31, 23, 0x80000010ull, PixelFormatGpu::bgr_u8);
  auto destination =
      Image(0x9000000000ull, 31, 23, 0x80000020ull, PixelFormatGpu::bgr_u8);
  ok &= Check(LaunchMirrorPrepared(kernel, source, destination, &error) &&
                  storage.launches == 1 && storage.stream == stream &&
                  storage.grid_x == 2 && storage.grid_y == 2 &&
                  storage.source == source.ptr &&
                  storage.source_pitch == source.pitch &&
                  storage.destination == destination.ptr &&
                  storage.destination_pitch == destination.pitch &&
                  storage.width == 31 && storage.height == 23,
              "64-bit launch ABI and owner stream");
  ok &= Check(storage.allocations == 0 && storage.copies == 0 &&
                  storage.synchronizations == 0,
              "execute performs no allocation, transfer, or synchronization");

  auto one = Image(0x2000, 1, 1, 3);
  auto one_out = Image(0x3000, 1, 1, 7);
  ok &= Check(LaunchMirrorPrepared(kernel, one, one_out, &error),
              "1x1 image accepted");
  auto tall = Image(0x4000, 1, 19, 8);
  auto tall_out = Image(0x5000, 1, 19, 9);
  ok &= Check(LaunchMirrorPrepared(kernel, tall, tall_out, &error),
              "1xN image accepted");
  auto wide = Image(0x6000, 20, 1, 64);
  auto wide_out = Image(0x7000, 20, 1, 67);
  ok &= Check(LaunchMirrorPrepared(kernel, wide, wide_out, &error),
              "Nx1 image accepted");
  const int accepted_launches = storage.launches;

  ok &= Check(!LaunchMirrorPrepared(kernel, one, one, &error),
              "exact alias rejected");
  auto partial = Image(source.ptr + source.pitch, 31, 23, source.pitch,
                       PixelFormatGpu::bgr_u8);
  ok &= Check(!LaunchMirrorPrepared(kernel, source, partial, &error),
              "partial alias rejected");
  auto wrong_format =
      Image(0xa000000000ull, 31, 23, 93, PixelFormatGpu::rgba_u8);
  ok &= Check(!LaunchMirrorPrepared(kernel, source, wrong_format, &error),
              "unsupported format rejected");
  auto wrong_geometry = Image(destination.ptr, 30, destination.h,
                              destination.pitch, destination.format);
  ok &= Check(!LaunchMirrorPrepared(kernel, source, wrong_geometry, &error),
              "geometry mismatch rejected");
  auto wrapping = Image(std::numeric_limits<CUdeviceptr>::max() - 4, 3, 2, 9);
  ok &= Check(!LaunchMirrorPrepared(kernel, wrapping, destination, &error),
              "wrapping span rejected");
  storage.current = other_context;
  ok &= Check(!LaunchMirrorPrepared(kernel, source, destination, &error),
              "foreign current context rejected");
  ok &= Check(storage.launches == accepted_launches,
              "invalid launches do not reach driver");
  storage.current = context;

  storage.unload_result = 1;
  ok &= Check(!DestroyMirrorKernelPrepared(&kernel, &error) && kernel.live,
              "failed destroy remains retryable");
  storage.unload_result = CUDA_SUCCESS;
  ok &= Check(DestroyMirrorKernelPrepared(&kernel, &error) && !kernel.live &&
                  storage.unloads == 2,
              "destroy retry succeeds");

  State partial_storage;
  auto partial_driver = Driver(&partial_storage);
  partial_storage.current = context;
  partial_storage.resolve_result = 1;
  partial_storage.unload_result = 1;
  PreparedMirrorKernel partial_kernel;
  ok &= Check(!PrepareMirrorKernel(partial_driver, context, stream,
                                   &partial_kernel, &error) &&
                  partial_kernel.live && !partial_kernel.function,
              "partial prepare retains failed-unload module");
  partial_storage.unload_result = CUDA_SUCCESS;
  ok &= Check(DestroyMirrorKernelPrepared(&partial_kernel, &error),
              "partial prepare cleanup retry");
  return ok;
}

std::vector<std::uint8_t> Expected(const std::vector<std::uint8_t> &source,
                                   int width, int height, std::size_t pitch) {
  auto expected = source;
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c)
        expected[static_cast<std::size_t>(y) * pitch + x * 3 + c] =
            source[static_cast<std::size_t>(y) * pitch + (width - 1 - x) * 3 +
                   c];
  return expected;
}

int NativeOracle() {
  if (!std::getenv("STUDIOCAST_REQUIRE_CUDA_MIRROR_TEST")) {
    std::cout << "[SKIP] native CUDA mirror oracle not requested\n";
    return 77;
  }
  CudaDriverApi driver;
  std::string error;
  if (!driver.Initialize(&error) || !driver.EnsureContextForDevice(0, &error)) {
    std::cerr << "[FAIL] required CUDA unavailable: " << error << '\n';
    return 1;
  }
  CUcontext context = nullptr;
  CUstream stream = nullptr;
  if (!driver.f().cuCtxGetCurrent ||
      driver.f().cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context ||
      !driver.CreateStream(&stream, &error)) {
    std::cerr << "[FAIL] CUDA setup: " << error << '\n';
    return 1;
  }
  PreparedMirrorKernel kernel;
  bool ok = PrepareMirrorKernel(driver, context, stream, &kernel, &error);
  const std::pair<int, int> shapes[] = {{5, 3}, {6, 4}, {1, 9}, {11, 1}};
  for (auto format : {PixelFormatGpu::rgb_u8, PixelFormatGpu::bgr_u8}) {
    for (const auto &[width, height] : shapes) {
      const std::size_t host_pitch = static_cast<std::size_t>(width) * 3 + 7;
      std::vector<std::uint8_t> source(host_pitch * height, 0xa5);
      std::vector<std::uint8_t> output(host_pitch * height, 0xcd);
      for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
          for (int c = 0; c < 3; ++c)
            source[static_cast<std::size_t>(y) * host_pitch + x * 3 + c] =
                static_cast<std::uint8_t>((y * 71 + x * 13 + c * 5) & 255);
      auto expected = Expected(source, width, height, host_pitch);
      for (int y = 0; y < height; ++y)
        std::fill(expected.begin() + static_cast<std::size_t>(y) * host_pitch +
                      width * 3,
                  expected.begin() +
                      (static_cast<std::size_t>(y) + 1) * host_pitch,
                  0xcd);
      CudaImage gpu_source, gpu_output;
      const bool pass = gpu_source.Allocate(&driver, width, height,
                                            PixelFormatGpu::rgb_u8, &error) &&
                        gpu_output.Allocate(&driver, width, height,
                                            PixelFormatGpu::rgb_u8, &error) &&
                        gpu_source.UploadFromCpuRgb24(
                            &driver, source.data(), host_pitch, stream, &error);
      gpu_source.format = format;
      gpu_output.format = format;
      const bool launched =
          pass &&
          LaunchMirrorPrepared(kernel, gpu_source, gpu_output, &error) &&
          (gpu_output.format = PixelFormatGpu::rgb_u8, true) &&
          gpu_output.DownloadToCpuRgb24(&driver, output.data(), host_pitch,
                                        stream, &error) &&
          driver.StreamSynchronize(stream, &error) && output == expected;
      ok = ok && Check(launched, "native exact-byte CPU oracle and padding");
      (void)gpu_output.Free(&driver, nullptr);
      (void)gpu_source.Free(&driver, nullptr);
    }
  }
  if (kernel.live) {
    (void)driver.StreamSynchronize(stream, nullptr);
    ok = DestroyMirrorKernelPrepared(&kernel, &error) && ok;
  }
  if (stream)
    (void)driver.DestroyStream(stream, nullptr);
  if (ok)
    std::cout << "[PASS] native CUDA mirror exact-byte oracle: odd/even, "
                 "pitched RGB/BGR, 1xN, Nx1\n";
  return ok ? 0 : 1;
}
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--native")
    return NativeOracle();
  if (!FakeContract())
    return 1;
  std::cout << "[PASS] prepared CUDA mirror fake contract: ownership, 64-bit "
               "ABI/spans, aliasing, no hot-path transfer/sync/allocation, "
               "and retryable cleanup\n";
  return 0;
}
