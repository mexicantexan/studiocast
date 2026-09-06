#include "core/cuda/kernels/prepared_matte_bounds.h"

#include <cstdlib>
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
  std::size_t pitch = sizeof(DeviceMatteBoundsRecord);
  CUresult free_result = CUDA_SUCCESS, unload_result = CUDA_SUCCESS;
  int loads = 0, resolves = 0, allocs = 0, frees = 0, launches = 0, unloads = 0;
  CUfunction init = reinterpret_cast<CUfunction>(std::uintptr_t{0x21});
  CUfunction reduce = reinterpret_cast<CUfunction>(std::uintptr_t{0x22});
  CUstream last_stream = nullptr;
  unsigned gx = 0, gy = 0, bx = 0, by = 0;
  unsigned long long source = 0, pitch_arg = 0, record = 0;
  std::uint32_t width = 0, height = 0;
};
State *g = nullptr;
bool Check(bool v, std::string_view m) {
  if (!v)
    std::cerr << "[FAIL] " << m << '\n';
  return v;
}
CUresult Init(unsigned) { return CUDA_SUCCESS; }
CUresult Load(CUmodule *m, const void *) {
  ++g->loads;
  *m = reinterpret_cast<CUmodule>(std::uintptr_t{0x10});
  return CUDA_SUCCESS;
}
CUresult Resolve(CUfunction *f, CUmodule, const char *n) {
  ++g->resolves;
  if (std::string_view(n) == "initialize_matte_bounds")
    *f = g->init;
  else if (std::string_view(n) == "reduce_matte_bounds")
    *f = g->reduce;
  else
    return 1;
  return CUDA_SUCCESS;
}
CUresult Alloc(CUdeviceptr *p, std::size_t *pitch, std::size_t, std::size_t,
               unsigned) {
  ++g->allocs;
  *p = 0x5000;
  *pitch = g->pitch;
  return CUDA_SUCCESS;
}
CUresult Free(CUdeviceptr) {
  ++g->frees;
  return g->free_result;
}
CUresult Unload(CUmodule) {
  ++g->unloads;
  return g->unload_result;
}

CUresult Copy(const CUDA_MEMCPY2D *, CUstream) { return CUDA_SUCCESS; }
CUresult Memset(CUdeviceptr, unsigned char, std::size_t, CUstream) {
  return CUDA_SUCCESS;
}
CUresult StreamCreate(CUstream *s, unsigned) {
  *s = reinterpret_cast<CUstream>(0x30);
  return CUDA_SUCCESS;
}
CUresult StreamDestroy(CUstream) { return CUDA_SUCCESS; }
CUresult StreamSync(CUstream) { return CUDA_SUCCESS; }
CUresult DeviceCount(int *n) {
  *n = 1;
  return CUDA_SUCCESS;
}
CUresult DeviceGet(CUdevice *d, int n) {
  *d = n;
  return CUDA_SUCCESS;
}
CUresult DeviceUuid(CUuuid *u, CUdevice) {
  std::memset(u, 0, sizeof(*u));
  return CUDA_SUCCESS;
}
CUresult PrimaryRetain(CUcontext *c, CUdevice) {
  *c = reinterpret_cast<CUcontext>(0x40);
  return CUDA_SUCCESS;
}
CUresult PrimaryRelease(CUdevice) { return CUDA_SUCCESS; }
CUresult CtxSet(CUcontext c) {
  g->current = c;
  return CUDA_SUCCESS;
}
CUresult CtxDevice(CUdevice *d) {
  *d = 0;
  return CUDA_SUCCESS;
}
CUresult CtxGet(CUcontext *c) {
  *c = g->current;
  return CUDA_SUCCESS;
}
CUresult Launch(CUfunction f, unsigned gx, unsigned gy, unsigned, unsigned bx,
                unsigned by, unsigned, unsigned, CUstream s, void **a,
                void **) {
  ++g->launches;
  g->last_stream = s;
  g->gx = gx;
  g->gy = gy;
  g->bx = bx;
  g->by = by;
  if (f == g->init) {
    g->record = *static_cast<unsigned long long *>(a[0]);
    g->width = *static_cast<std::uint32_t *>(a[1]);
    g->height = *static_cast<std::uint32_t *>(a[2]);
  } else {
    g->source = *static_cast<unsigned long long *>(a[0]);
    g->pitch_arg = *static_cast<unsigned long long *>(a[1]);
    g->record = *static_cast<unsigned long long *>(a[4]);
  }
  return CUDA_SUCCESS;
}
CudaDriverApi Driver(State *s) {
  g = s;
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
  f.cuDevicePrimaryCtxRetain = &PrimaryRetain;
  f.cuDevicePrimaryCtxRelease = &PrimaryRelease;
  f.cuCtxSetCurrent = &CtxSet;
  f.cuCtxGetCurrent = &CtxGet;
  f.cuCtxGetDevice = &CtxDevice;
  CudaDriverApi d;
  std::string e;
  (void)d.InitializeForTesting(f, &e);
  return d;
}
CudaImage Alpha(CUdeviceptr p, int w, int h, std::size_t pitch) {
  CudaImage i;
  i.ptr = p;
  i.w = w;
  i.h = h;
  i.pitch = pitch;
  i.format = PixelFormatGpu::f32_1;
  return i;
}

bool FakeContracts() {
  State s;
  auto d = Driver(&s);
  auto c = reinterpret_cast<CUcontext>(0x40);
  auto stream = reinterpret_cast<CUstream>(0x30);
  s.current = c;
  std::string e;
  PreparedMatteBounds p;
  const DeviceMatteBoundsRecord below{0, 0, 4, 4, 31};
  const DeviceMatteBoundsRecord exact{0, 0, 4, 4, 32};
  bool ok =
      Check(!MatteBoundsForegroundFound(below, 65, 49) &&
                MatteBoundsForegroundFound(exact, 65, 49),
            "foreground count boundary") &&
      Check(MinimumMatteBoundsBudget().device_bytes ==
                    sizeof(DeviceMatteBoundsRecord) &&
                MinimumMatteBoundsBudget().resource_count == 1,
            "budget contract") &&
      Check(!PrepareMatteBounds(d, c, stream,
                                sizeof(DeviceMatteBoundsRecord) - 1, &p, &e) &&
                s.loads == 0,
            "under-budget probed driver") &&
      Check(PrepareMatteBounds(d, c, stream, sizeof(DeviceMatteBoundsRecord),
                               &p, &e),
            "exact budget prepare") &&
      Check(s.loads == 1 && s.resolves == 2 && s.allocs == 1 &&
                p.device_record_bytes == sizeof(DeviceMatteBoundsRecord),
            "prepare lifecycle") &&
      Check(MatteBoundsDeviceRecord(p) == 0x5000 &&
                MatteBoundsDeviceRecordBytes(p) ==
                    sizeof(DeviceMatteBoundsRecord),
            "device result view");
  const auto reserved = PreparedMatteBoundsReservedBudget(p);
  ok &= Check(reserved.device_bytes == sizeof(DeviceMatteBoundsRecord) &&
                  reserved.resource_count == 1,
              "exact reserved budget evidence");
  auto a = Alpha(0x1000, 65, 49, 65u * 4u + 12u);
  ok &= Check(LaunchMatteBoundsPrepared(p, a, &e) && s.launches == 2 &&
                  s.last_stream == stream && s.source == 0x1000 &&
                  s.pitch_arg == 272 && s.record == 0x5000 && s.width == 65 &&
                  s.height == 49 && s.gx == 5 && s.gy == 4 && s.bx == 16 &&
                  s.by == 16,
              "launch ABI");
  auto wide_pitch =
      Alpha(0x1000, 4, 1, static_cast<std::size_t>(UINT32_MAX) + 1u);
  ok &= Check(LaunchMatteBoundsPrepared(p, wide_pitch, &e) && s.launches == 4 &&
                  s.pitch_arg == UINT64_C(0x100000000),
              "64-bit pitch rejected or truncated");
  auto overlap = Alpha(0x4ff8, 4, 1, 16);
  ok &= Check(!LaunchMatteBoundsPrepared(p, overlap, &e) && s.launches == 4,
              "output alias accepted");
  auto bad = Alpha(std::numeric_limits<CUdeviceptr>::max() - 4, 4, 2, 16);
  ok &= Check(!LaunchMatteBoundsPrepared(p, bad, &e) && s.launches == 4,
              "overflow accepted");
  auto short_pitch = Alpha(0x1000, 4, 2, 15);
  ok &= Check(!LaunchMatteBoundsPrepared(p, short_pitch, &e) && s.launches == 4,
              "short pitch accepted");
  auto wrong_format = Alpha(0x1000, 4, 2, 16);
  wrong_format.format = PixelFormatGpu::gray_u8;
  ok &=
      Check(!LaunchMatteBoundsPrepared(p, wrong_format, &e), "format accepted");
  s.free_result = 1;
  ok &= Check(!DestroyMatteBoundsPrepared(&p, &e) && p.live &&
                  p.device_record == 0x5000 && s.unloads == 0,
              "failed free lost ownership");
  s.free_result = CUDA_SUCCESS;
  s.unload_result = 1;
  ok &= Check(!DestroyMatteBoundsPrepared(&p, &e) && p.live &&
                  p.device_record == 0 && p.module && s.unloads == 1,
              "failed unload lost module");
  s.unload_result = CUDA_SUCCESS;
  ok &= Check(DestroyMatteBoundsPrepared(&p, &e) && !p.live && s.frees == 2 &&
                  s.unloads == 2,
              "retry destroy");
  State over;
  over.pitch = 64;
  auto od = Driver(&over);
  over.current = c;
  PreparedMatteBounds q;
  ok &= Check(!PrepareMatteBounds(od, c, stream, 63, &q, &e) &&
                  over.allocs == 1 && over.frees == 1 && over.unloads == 1 &&
                  !q.live,
              "actual pitch over-budget accepted");
  return ok;
}

DeviceMatteBoundsRecord CpuOracle(const std::vector<float> &a, int w, int h) {
  DeviceMatteBoundsRecord r{w, h, -1, -1, 0};
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (a[static_cast<std::size_t>(y * w + x)] > .35F) {
        r.x0 = std::min(r.x0, x);
        r.y0 = std::min(r.y0, y);
        r.x1 = std::max(r.x1, x);
        r.y1 = std::max(r.y1, y);
        ++r.foreground_count;
      }
  return r;
}

bool NativeOracle() {
  CudaDriverApi d;
  std::string e;
  if (!d.Initialize(&e) || !d.EnsureContextForDevice(0, &e)) {
    std::cerr << "[FAIL] required CUDA unavailable: " << e << '\n';
    return false;
  }
  CUcontext c = nullptr;
  CUstream stream = nullptr;
  CudaImage alpha;
  PreparedMatteBounds p;
  const auto run = [&](std::vector<float> data, int w, int h) -> bool {
    if (!alpha.ReallocIfNeeded(&d, w, h, PixelFormatGpu::f32_1, &e))
      return false;
    CUDA_MEMCPY2D up{};
    up.srcMemoryType = CU_MEMORYTYPE_HOST;
    up.srcHost = data.data();
    up.srcPitch = static_cast<std::size_t>(w) * 4;
    up.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    up.dstDevice = alpha.ptr;
    up.dstPitch = alpha.pitch;
    up.WidthInBytes = static_cast<std::size_t>(w) * 4;
    up.Height = h;
    DeviceMatteBoundsRecord got{};
    CUDA_MEMCPY2D down{};
    down.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    down.srcDevice = MatteBoundsDeviceRecord(p);
    down.srcPitch = MatteBoundsDeviceRecordBytes(p);
    down.dstMemoryType = CU_MEMORYTYPE_HOST;
    down.dstHost = &got;
    down.dstPitch = sizeof(got);
    down.WidthInBytes = sizeof(got);
    down.Height = 1;
    if (d.f().cuMemcpy2DAsync(&up, stream) != CUDA_SUCCESS ||
        !LaunchMatteBoundsPrepared(p, alpha, &e) ||
        d.f().cuMemcpy2DAsync(&down, stream) != CUDA_SUCCESS ||
        !d.StreamSynchronize(stream, &e))
      return false;
    auto want = CpuOracle(data, w, h);
    return std::memcmp(&got, &want, sizeof(got)) == 0 &&
           MatteBoundsForegroundFound(got, w, h) ==
               MatteBoundsForegroundFound(want, w, h);
  };
  bool ok = d.f().cuCtxGetCurrent(&c) == CUDA_SUCCESS && c &&
            d.CreateStream(&stream, &e) &&
            PrepareMatteBounds(d, c, stream, 512, &p, &e);
  std::vector<float> empty(65u * 49u, 0);
  empty[0] = .35F;
  empty[1] = std::numeric_limits<float>::quiet_NaN();
  ok = ok && run(empty, 65, 49);
  std::vector<float> below(17u * 13u, 0);
  for (int i = 0; i < 31; ++i)
    below[static_cast<std::size_t>(i)] = .8F;
  ok = ok && run(below, 17, 13);
  std::vector<float> edge = below;
  edge.back() = 1;
  ok = ok && run(edge, 17, 13);
  if (p.live) {
    (void)d.StreamSynchronize(stream, &e);
    ok = DestroyMatteBoundsPrepared(&p, &e) && ok;
  }
  if (alpha.ptr)
    ok = alpha.Free(&d, &e) && ok;
  if (stream)
    ok = d.DestroyStream(stream, &e) && ok;
  if (!ok)
    std::cerr << "[FAIL] native oracle: " << e << '\n';
  return ok;
}
} // namespace
int main() {
  if (!FakeContracts())
    return 1;
  if (std::getenv("STUDIOCAST_REQUIRE_CUDA_MATTE_BOUNDS_TEST") == nullptr) {
    std::cout << "[SKIP] real CUDA matte bounds not requested\n";
    return 77;
  }
  return NativeOracle() ? 0 : 1;
}
