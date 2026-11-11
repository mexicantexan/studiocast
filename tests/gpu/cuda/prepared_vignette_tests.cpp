#include "core/cuda/kernels/prepared_vignette.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace studiocast::cuda;
using namespace studiocast::cuda::kernels;

struct State {
  CUcontext current = nullptr;
  int loads = 0;
  int resolves = 0;
  int launches = 0;
  int unloads = 0;
  CUresult unload_result = CUDA_SUCCESS;
  CUstream stream = nullptr;
  std::uint32_t grid_x = 0, grid_y = 0;
  unsigned long long src = 0, dst = 0;
  std::uint32_t src_pitch = 0, dst_pitch = 0, width = 0, height = 0;
  float intensity = 0, center_x = 0, center_y = 0;
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
  *module = reinterpret_cast<CUmodule>(std::uintptr_t{0x10});
  return CUDA_SUCCESS;
}
CUresult Resolve(CUfunction *function, CUmodule, const char *name) {
  ++state->resolves;
  if (std::string_view(name) != "vignette_u8x3")
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
  state->src = *static_cast<unsigned long long *>(args[0]);
  state->src_pitch = *static_cast<std::uint32_t *>(args[1]);
  state->dst = *static_cast<unsigned long long *>(args[2]);
  state->dst_pitch = *static_cast<std::uint32_t *>(args[3]);
  state->width = *static_cast<std::uint32_t *>(args[4]);
  state->height = *static_cast<std::uint32_t *>(args[5]);
  state->intensity = *static_cast<float *>(args[6]);
  state->center_x = *static_cast<float *>(args[7]);
  state->center_y = *static_cast<float *>(args[8]);
  return CUDA_SUCCESS;
}
CUresult Alloc(CUdeviceptr *ptr, std::size_t *pitch, std::size_t width,
               std::size_t, unsigned) {
  *ptr = 0x1000;
  *pitch = width;
  return CUDA_SUCCESS;
}
CUresult Free(CUdeviceptr) { return CUDA_SUCCESS; }
CUresult Copy(const CUDA_MEMCPY2D *, CUstream) { return CUDA_SUCCESS; }
CUresult Memset(CUdeviceptr, unsigned char, std::size_t, CUstream) {
  return CUDA_SUCCESS;
}
CUresult StreamCreate(CUstream *out, unsigned) {
  *out = reinterpret_cast<CUstream>(std::uintptr_t{0x30});
  return CUDA_SUCCESS;
}
CUresult StreamDestroy(CUstream) { return CUDA_SUCCESS; }
CUresult StreamSync(CUstream) { return CUDA_SUCCESS; }
CUresult DeviceCount(int *count) { *count = 1; return CUDA_SUCCESS; }
CUresult DeviceGet(CUdevice *device, int ordinal) { *device = ordinal; return CUDA_SUCCESS; }
CUresult DeviceUuid(CUuuid *uuid, CUdevice) { std::memset(uuid, 0, sizeof(*uuid)); return CUDA_SUCCESS; }
CUresult DevicePci(char *value, int size, CUdevice) {
  constexpr char text[] = "0000:01:00.0";
  if (size < static_cast<int>(sizeof(text))) return 1;
  std::memcpy(value, text, sizeof(text)); return CUDA_SUCCESS;
}
CUresult PrimaryRetain(CUcontext *context, CUdevice) {
  *context = reinterpret_cast<CUcontext>(std::uintptr_t{0x40}); return CUDA_SUCCESS;
}
CUresult PrimaryRelease(CUdevice) { return CUDA_SUCCESS; }
CUresult CtxSet(CUcontext context) { state->current = context; return CUDA_SUCCESS; }
CUresult CtxGet(CUcontext *context) { *context = state->current; return CUDA_SUCCESS; }
CUresult CtxDevice(CUdevice *device) { *device = 0; return CUDA_SUCCESS; }

CudaDriverApi Driver(State *storage) {
  state = storage;
  CudaDriverApi::Functions f;
  f.cuInit = &Init; f.cuModuleLoadData = &Load;
  f.cuModuleGetFunction = &Resolve; f.cuModuleUnload = &Unload;
  f.cuLaunchKernel = &Launch; f.cuMemAllocPitch = &Alloc; f.cuMemFree = &Free;
  f.cuMemcpy2DAsync = &Copy; f.cuMemsetD8Async = &Memset;
  f.cuStreamCreate = &StreamCreate; f.cuStreamDestroy = &StreamDestroy;
  f.cuStreamSynchronize = &StreamSync; f.cuDeviceGetCount = &DeviceCount;
  f.cuDeviceGet = &DeviceGet; f.cuDeviceGetUuid = &DeviceUuid;
  f.cuDeviceGetPCIBusId = &DevicePci; f.cuDevicePrimaryCtxRetain = &PrimaryRetain;
  f.cuDevicePrimaryCtxRelease = &PrimaryRelease; f.cuCtxSetCurrent = &CtxSet;
  f.cuCtxGetCurrent = &CtxGet; f.cuCtxGetDevice = &CtxDevice;
  CudaDriverApi driver;
  std::string error;
  (void)driver.InitializeForTesting(f, &error);
  return driver;
}

CudaImage Image(CUdeviceptr ptr, int width, int height, PixelFormatGpu format) {
  CudaImage image;
  image.ptr = ptr;
  image.pitch = static_cast<std::size_t>(width) * 3;
  image.w = width;
  image.h = height;
  image.format = format;
  return image;
}

bool LifecycleAndLaunchContract() {
  State storage;
  auto driver = Driver(&storage);
  const auto context = reinterpret_cast<CUcontext>(std::uintptr_t{0x40});
  const auto stream = reinterpret_cast<CUstream>(std::uintptr_t{0x30});
  storage.current = context;
  PreparedVignetteKernel kernel;
  std::string error;
  bool ok = Check(PrepareVignetteKernel(driver, context, stream, &kernel, &error),
                  "prepare failed") &&
            Check(storage.loads == 1 && storage.resolves == 1,
                  "prepare did not freeze one module/function");
  auto source = Image(0x1000, 31, 23, PixelFormatGpu::rgb_u8);
  auto destination = Image(0x2000, 31, 23, PixelFormatGpu::rgb_u8);
  ok &= Check(LaunchVignettePrepared(kernel, source, destination, 0.75F,
                                     7.5F, 11.0F, &error),
              "prepared launch failed");
  ok &= Check(storage.launches == 1 && storage.stream == stream &&
                  storage.grid_x == 2 && storage.grid_y == 2 &&
                  storage.src == 0x1000 && storage.dst == 0x2000 &&
                  storage.src_pitch == 93 && storage.dst_pitch == 93 &&
                  storage.width == 31 && storage.height == 23 &&
                  storage.intensity == 0.75F && storage.center_x == 7.5F &&
                  storage.center_y == 11.0F,
              "launch ABI or owner stream changed");
  ok &= Check(LaunchVignettePrepared(kernel, source, source, 0.0F, 0.0F,
                                     23.0F, &error) &&
                  storage.launches == 2 && storage.src == storage.dst,
              "exact in-place/no-op boundary launch failed");
  auto mismatched_pitch = Image(source.ptr, 31, 23, PixelFormatGpu::rgb_u8);
  mismatched_pitch.pitch += 3;
  ok &= Check(!LaunchVignettePrepared(kernel, source, mismatched_pitch, 0.5F,
                                      7.5F, 11.0F, &error) &&
                  storage.launches == 2,
              "same pointer with different pitch was launched");
  auto partial_overlap = Image(source.ptr + 1, 31, 23, PixelFormatGpu::rgb_u8);
  ok &= Check(!LaunchVignettePrepared(kernel, source, partial_overlap, 0.5F,
                                      7.5F, 11.0F, &error) &&
                  storage.launches == 2,
              "partially overlapping images were launched");
  auto wrapping = Image(std::numeric_limits<CUdeviceptr>::max() - 8u, 31, 23,
                        PixelFormatGpu::rgb_u8);
  ok &= Check(!LaunchVignettePrepared(kernel, wrapping, destination, 0.5F,
                                      7.5F, 11.0F, &error) &&
                  storage.launches == 2,
              "wrapping device image extent was launched");
  ok &= Check(!LaunchVignettePrepared(kernel, source, destination, 1.1F,
                                      7.5F, 11.0F, &error),
              "invalid intensity accepted");
  destination.format = PixelFormatGpu::bgr_u8;
  ok &= Check(!LaunchVignettePrepared(kernel, source, destination, 0.5F,
                                      7.5F, 11.0F, &error),
              "mixed formats accepted");
  storage.unload_result = 1;
  ok &= Check(!DestroyVignetteKernelPrepared(&kernel, &error) && kernel.live,
              "failed unload did not retain retryable owner");
  storage.unload_result = CUDA_SUCCESS;
  ok &= Check(DestroyVignetteKernelPrepared(&kernel, &error) && !kernel.live &&
                  storage.unloads == 2,
              "retry shutdown failed");
  return ok;
}

bool RealDevicePixels() {
  if (std::getenv("STUDIOCAST_REQUIRE_CUDA_VIGNETTE_TEST") == nullptr) {
    std::cout << "[SKIP] real CUDA vignette pixels not requested\n";
    return true;
  }
  CudaDriverApi driver;
  std::string error;
  if (!driver.Initialize(&error) || !driver.EnsureContextForDevice(0, &error)) {
    std::cerr << "[FAIL] required CUDA device unavailable: " << error << '\n';
    return false;
  }
  const auto *identity = driver.context_identity();
  CUcontext context = nullptr;
  CUstream stream = nullptr;
  CudaImage source, destination;
  PreparedVignetteKernel kernel;
  constexpr int width = 17, height = 13;
  constexpr float intensity = 0.8F, center_x = 6.25F, center_y = 8.0F;
  std::vector<std::uint8_t> input(static_cast<std::size_t>(width * height * 3));
  std::vector<std::uint8_t> output(input.size());
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<std::uint8_t>((index * 37u + 11u) & 0xffu);
  bool ok = identity && driver.f().cuCtxGetCurrent &&
            driver.f().cuCtxGetCurrent(&context) == CUDA_SUCCESS && context &&
            driver.CreateStream(&stream, &error) &&
            source.Allocate(&driver, width, height, PixelFormatGpu::rgb_u8, &error) &&
            destination.Allocate(&driver, width, height, PixelFormatGpu::rgb_u8,
                                 &error) &&
            source.UploadFromCpuRgb24(&driver, input.data(), width * 3u, stream,
                                      &error) &&
            PrepareVignetteKernel(driver, context, stream, &kernel, &error) &&
            LaunchVignettePrepared(kernel, source, destination, intensity,
                                   center_x, center_y, &error) &&
            destination.DownloadToCpuRgb24(&driver, output.data(), width * 3u,
                                           stream, &error) &&
            driver.StreamSynchronize(stream, &error);
  if (!ok)
    std::cerr << "[FAIL] real CUDA vignette execution: " << error << '\n';
  for (int y = 0; ok && y < height; ++y) {
    for (int x = 0; ok && x < width; ++x) {
      const float dx = (static_cast<float>(x) + 0.5F - center_x) *
                       (2.0F / static_cast<float>(width));
      const float dy = (static_cast<float>(y) + 0.5F - center_y) *
                       (2.0F / static_cast<float>(height));
      const float radius = std::clamp(std::sqrt(dx * dx + dy * dy) *
                                          0.70710677F,
                                      0.0F, 1.0F);
      const float factor = std::max(0.0F, 1.0F - intensity * radius * radius);
      for (int channel = 0; channel < 3; ++channel) {
        const auto index = static_cast<std::size_t>((y * width + x) * 3 + channel);
        const auto expected = static_cast<int>(
            std::nearbyint(static_cast<float>(input[index]) * factor));
        if (std::abs(static_cast<int>(output[index]) - expected) > 1) {
          std::cerr << "[FAIL] pixel mismatch x=" << x << " y=" << y
                    << " channel=" << channel << " expected=" << expected
                    << " actual=" << static_cast<unsigned>(output[index]) << '\n';
          ok = false;
          break;
        }
      }
    }
  }
  if (stream)
    (void)driver.StreamSynchronize(stream, nullptr);
  if (kernel.live)
    ok = DestroyVignetteKernelPrepared(&kernel, &error) && ok;
  if (destination.Valid())
    ok = destination.Free(&driver, &error) && ok;
  if (source.Valid())
    ok = source.Free(&driver, &error) && ok;
  if (stream)
    ok = driver.DestroyStream(stream, &error) && ok;
  return ok;
}
} // namespace

int main() {
  return LifecycleAndLaunchContract() && RealDevicePixels() ? 0 : 1;
}
