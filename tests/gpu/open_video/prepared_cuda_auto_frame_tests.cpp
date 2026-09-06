#include "core/open_video/prepared_cuda_auto_frame.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>

namespace {
using namespace studiocast;
using namespace studiocast::open_video;
bool Check(bool v, std::string_view m) {
  if (!v)
    std::cerr << "[FAIL] " << m << '\n';
  return v;
}

struct DriverState {
  cuda::CUcontext context = reinterpret_cast<cuda::CUcontext>(0x41);
  int frees = 0, synchronizations = 0;
};
DriverState *driver_state = nullptr;
cuda::CUresult Init(unsigned) { return cuda::CUDA_SUCCESS; }
cuda::CUresult DeviceCount(int *count) {
  *count = 1;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult DeviceGet(cuda::CUdevice *device, int ordinal) {
  *device = ordinal;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult DeviceUuid(cuda::CUuuid *uuid, cuda::CUdevice) {
  std::memset(uuid, 0, sizeof(*uuid));
  uuid->bytes[0] = 1;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult DevicePci(char *value, int size, cuda::CUdevice) {
  constexpr char id[] = "0000:01:00.0";
  if (size < static_cast<int>(sizeof(id)))
    return 1;
  std::memcpy(value, id, sizeof(id));
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult Retain(cuda::CUcontext *context, cuda::CUdevice) {
  *context = driver_state->context;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult Release(cuda::CUdevice) { return cuda::CUDA_SUCCESS; }
cuda::CUresult SetCurrent(cuda::CUcontext context) {
  driver_state->context = context;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult GetCurrent(cuda::CUcontext *context) {
  *context = driver_state->context;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult GetDevice(cuda::CUdevice *device) {
  *device = 0;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult Alloc(cuda::CUdeviceptr *ptr, std::size_t *pitch,
                     std::size_t width, std::size_t, unsigned) {
  *ptr = 0x8000;
  *pitch = width + 512;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult Free(cuda::CUdeviceptr) {
  ++driver_state->frees;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult StreamSync(cuda::CUstream) {
  return ++driver_state->synchronizations == 1 ? cuda::CUDA_ERROR_UNKNOWN
                                               : cuda::CUDA_SUCCESS;
}
cuda::CUresult StreamCreate(cuda::CUstream *stream, unsigned) {
  *stream = reinterpret_cast<cuda::CUstream>(0x20);
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult StreamDestroy(cuda::CUstream) { return cuda::CUDA_SUCCESS; }
cuda::CUresult Load(cuda::CUmodule *module, const void *) {
  *module = reinterpret_cast<cuda::CUmodule>(0x50);
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult Resolve(cuda::CUfunction *function, cuda::CUmodule,
                       const char *) {
  *function = reinterpret_cast<cuda::CUfunction>(0x51);
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult Unload(cuda::CUmodule) { return cuda::CUDA_SUCCESS; }
cuda::CUresult Launch(cuda::CUfunction, unsigned, unsigned, unsigned, unsigned,
                      unsigned, unsigned, unsigned, cuda::CUstream, void **,
                      void **) {
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult Copy(const cuda::CUDA_MEMCPY2D *, cuda::CUstream) {
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult Memset(cuda::CUdeviceptr, unsigned char, std::size_t,
                      cuda::CUstream) {
  return cuda::CUDA_SUCCESS;
}

cuda::CudaDriverApi FakeDriver(DriverState *state) {
  driver_state = state;
  cuda::CudaDriverApi::Functions f;
  f.cuInit = &Init;
  f.cuDeviceGetCount = &DeviceCount;
  f.cuDeviceGet = &DeviceGet;
  f.cuDeviceGetUuid = &DeviceUuid;
  f.cuDeviceGetPCIBusId = &DevicePci;
  f.cuDevicePrimaryCtxRetain = &Retain;
  f.cuDevicePrimaryCtxRelease = &Release;
  f.cuCtxSetCurrent = &SetCurrent;
  f.cuCtxGetCurrent = &GetCurrent;
  f.cuCtxGetDevice = &GetDevice;
  f.cuMemAllocPitch = &Alloc;
  f.cuMemFree = &Free;
  f.cuStreamSynchronize = &StreamSync;
  f.cuStreamCreate = &StreamCreate;
  f.cuStreamDestroy = &StreamDestroy;
  f.cuModuleLoadData = &Load;
  f.cuModuleGetFunction = &Resolve;
  f.cuModuleUnload = &Unload;
  f.cuLaunchKernel = &Launch;
  f.cuMemcpy2DAsync = &Copy;
  f.cuMemsetD8Async = &Memset;
  cuda::CudaDriverApi driver;
  std::string error;
  if (!driver.InitializeForTesting(f, &error) ||
      !driver.EnsureContextForDevice(0, &error))
    std::cerr << "[FAIL] fake CUDA driver setup: " << error << '\n';
  return driver;
}

struct FakeCrop final : IPreparedCudaAutoFrameCropRuntime {
  bool prepare_ok = true, crop_ok = true, shutdown_ok = true;
  int crops = 0, shutdowns = 0;
  PreparedCudaAutoFrameRect last{};
  cuda::CudaImage output{};
  bool Prepare(const PreparedCudaAutoFrameBinding &, int w, int h, std::size_t,
               std::string *error) noexcept override {
    output.ptr = 0x9000;
    output.pitch = w * 3;
    output.w = w;
    output.h = h;
    output.format = cuda::PixelFormatGpu::rgb_u8;
    if (!prepare_ok && error)
      *error = "post-allocation kernel preparation failed";
    return prepare_ok;
  }
  bool Crop(const cuda::CudaImage &, const PreparedCudaAutoFrameRect &r,
            const cuda::CudaImage **o, std::string *) noexcept override {
    ++crops;
    last = r;
    if (!crop_ok)
      return false;
    if (o)
      *o = &output;
    return true;
  }
  bool Shutdown(std::string *error) noexcept override {
    ++shutdowns;
    if (!shutdown_ok && error)
      *error = "allocation cleanup failed";
    return shutdown_ok;
  }
};
struct Analysis final : IPreparedCudaAutoFrameAnalysis {
  std::vector<PreparedCudaAutoFrameAnalysisResult> results;
  std::size_t at = 0;
  int calls = 0;
  PreparedCudaAutoFrameAnalysisResult
  Analyze(const PreparedCudaAutoFrameBinding &, const cuda::CudaImage &,
          std::uint64_t, FrameAnalysisCache *,
          const cuda::CudaPreparedTransferCompletion::WaitPolicy &) noexcept
      override {
    ++calls;
    return results[std::min(at++, results.size() - 1)];
  }
};
PreparedCudaAutoFrameBinding Binding() {
  static cuda::CudaDriverApi driver;
  PreparedCudaAutoFrameBinding b;
  b.driver = &driver;
  b.owner_context = reinterpret_cast<cuda::CUcontext>(0x10);
  b.stream = reinterpret_cast<cuda::CUstream>(0x20);
  b.stream_identity = 3;
  b.owner_identity = 4;
  b.configuration_generation = 5;
  b.model_generation = 6;
  b.context.physical_device.uuid[0] = 1;
  b.context.physical_device.pci_bus_id[0] = 'x';
  b.context.physical_device.pci_bus_id_length = 1;
  b.context.retained_context_id = 7;
  b.context.logical_generation = 8;
  return b;
}
cuda::CudaImage Image() {
  cuda::CudaImage i;
  i.ptr = 0x1000;
  i.pitch = 1920;
  i.w = 640;
  i.h = 480;
  i.format = cuda::PixelFormatGpu::rgb_u8;
  return i;
}
PreparedCudaAutoFrameSetup Setup(Analysis *f, Analysis *m,
                                 FrameAnalysisCache *c) {
  PreparedCudaAutoFrameSetup s;
  s.binding = Binding();
  s.frame_width = 640;
  s.frame_height = 480;
  s.face = f;
  s.matte = m;
  s.analysis_cache = c;
  s.resource_budget_bytes = 640 * 480 * 3;
  return s;
}
PreparedCudaAutoFrameAnalysisResult Box(PreparedCudaAutoFrameSource source,
                                        float x, float y, float w, float h) {
  PreparedCudaAutoFrameAnalysisResult r;
  r.ok = r.found = true;
  r.box = {source, {x, y, w, h}, .9F};
  return r;
}

bool GeometryAndSequenceReset() {
  FakeCrop crop;
  Analysis face;
  FrameAnalysisCache cache;
  face.results = {Box(PreparedCudaAutoFrameSource::face, 100, 80, 100, 120),
                  Box(PreparedCudaAutoFrameSource::face, 300, 80, 100, 120),
                  Box(PreparedCudaAutoFrameSource::face, 100, 80, 100, 120)};
  PreparedCudaAutoFrame af(&crop);
  PreparedCudaAutoFrameHandle h;
  std::string e;
  auto s = Setup(&face, nullptr, &cache);
  s.knobs = {50, 50, .15F};
  bool ok = Check(af.Prepare(s, &h, &e), "prepare");
  auto image = Image();
  auto first = af.Execute(h, s.binding, 10, image);
  auto second = af.Execute(h, s.binding, 11, image);
  auto reset = af.Execute(h, s.binding, 20, image);
  ok &= Check(first.status == PreparedCudaAutoFrameStatus::executed &&
                  crop.crops == 3,
              "face frames did not crop");
  ok &= Check(second.source_crop.x > first.source_crop.x &&
                  second.source_crop.x < reset.source_crop.x + 100,
              "smoothing not applied");
  ok &= Check(std::abs(reset.source_crop.x - first.source_crop.x) < .01F,
              "discontinuity did not reset smoothing");
  ok &= Check(first.source_crop.w / first.source_crop.h > 1.332F &&
                  first.source_crop.w / first.source_crop.h < 1.334F,
              "capture crop aspect drifted");
  auto stale = s.binding;
  ++stale.model_generation;
  ok &= Check(af.Execute(h, stale, 21, image).status ==
                  PreparedCudaAutoFrameStatus::stale_binding,
              "stale model binding accepted");
  return ok;
}

bool MatteCadenceCountersAndFailures() {
  FakeCrop crop;
  Analysis face, matte;
  FrameAnalysisCache cache;
  PreparedCudaAutoFrameAnalysisResult no_face;
  no_face.ok = true;
  no_face.transfers.intermediate_readbacks = 12;
  no_face.transfers.synchronizations = 1;
  no_face.transfers.cpu_tail_boundaries = 1;
  no_face.completion_waits = 1;
  no_face.cpu_tail_boundaries = 1;
  face.results = {no_face, no_face, no_face, no_face};
  auto found = Box(PreparedCudaAutoFrameSource::matte, 120, 100, 160, 180);
  found.transfers.intermediate_readbacks = 1;
  found.transfers.synchronizations = 1;
  found.transfers.cpu_tail_boundaries = 1;
  found.completion_waits = 1;
  found.cpu_tail_boundaries = 1;
  matte.results = {found, found};
  PreparedCudaAutoFrame af(&crop);
  PreparedCudaAutoFrameHandle h;
  auto s = Setup(&face, &matte, &cache);
  std::string e;
  bool ok = af.Prepare(s, &h, &e);
  auto image = Image();
  auto a = af.Execute(h, s.binding, 1, image);
  auto b = af.Execute(h, s.binding, 2, image);
  auto c = af.Execute(h, s.binding, 11, image);
  ok &= Check(matte.calls == 2 && face.calls == 3 && a.analysis_ran &&
                  b.analysis_ran && c.analysis_ran,
              "matte cadence/cache mismatch");
  ok &= Check(a.transfers.intermediate_readbacks == 13 &&
                  a.transfers.synchronizations == 2 &&
                  a.completion_waits == 2 && a.cpu_tail_boundaries == 2,
              "face plus matte counters were not summed exactly once");
  ok &= Check(b.transfers.intermediate_readbacks == 12 &&
                  b.completion_waits == 1 && b.cpu_tail_boundaries == 1,
              "cached matte added counters");
  crop.crop_ok = false;
  auto drop = af.Execute(h, s.binding, 12, image);
  ok &= Check(drop.status == PreparedCudaAutoFrameStatus::drop_frame &&
                  !drop.output,
              "partial crop failure did not drop");
  return ok;
}

bool StrictFailureAndRetryableShutdown() {
  FakeCrop crop;
  Analysis face, matte;
  FrameAnalysisCache cache;
  PreparedCudaAutoFrameAnalysisResult failed;
  failed.pending_work = false;
  failed.transfers.intermediate_readbacks = 4;
  face.results = {failed};
  matte.results = {Box(PreparedCudaAutoFrameSource::matte, 1, 1, 20, 20)};
  PreparedCudaAutoFrame af(&crop);
  PreparedCudaAutoFrameHandle h;
  auto s = Setup(&face, &matte, &cache);
  std::string e;
  bool ok = af.Prepare(s, &h, &e);
  auto image = Image();
  auto r = af.Execute(h, s.binding, 1, image);
  ok &= Check(r.status == PreparedCudaAutoFrameStatus::preserved_input &&
                  r.valid_input_preserved && matte.calls == 0 &&
                  r.transfers.intermediate_readbacks == 4,
              "strict face failure fell back");
  crop.shutdown_ok = false;
  ok &= Check(!af.Shutdown(&e), "shutdown failure hidden");
  crop.shutdown_ok = true;
  ok &= Check(af.Shutdown(&e) && crop.shutdowns == 2, "shutdown not retryable");
  return ok;
}

bool UnsafeAnalysisFailuresAndPrepareCleanup() {
  FakeCrop crop;
  Analysis analysis;
  FrameAnalysisCache cache;
  PreparedCudaAutoFrameAnalysisResult device;
  device.device_lost = true;
  device.pending_work = true;
  PreparedCudaAutoFrameAnalysisResult synchronization;
  synchronization.synchronization_failed = true;
  synchronization.pending_work = true;
  PreparedCudaAutoFrameAnalysisResult ordinary;
  analysis.results = {device, synchronization, ordinary};
  PreparedCudaAutoFrame af(&crop);
  PreparedCudaAutoFrameHandle h;
  auto s = Setup(&analysis, nullptr, &cache);
  std::string error;
  bool ok = Check(af.Prepare(s, &h, &error), "failure setup");
  auto image = Image();
  auto lost = af.Execute(h, s.binding, 1, image);
  auto sync = af.Execute(h, s.binding, 2, image);
  auto failed = af.Execute(h, s.binding, 3, image);
  ok &= Check(lost.status == PreparedCudaAutoFrameStatus::device_lost &&
                  !lost.output && !lost.valid_input_preserved,
              "device loss was not typed unsafe");
  ok &= Check(sync.status ==
                      PreparedCudaAutoFrameStatus::synchronization_failed &&
                  !sync.output && !sync.valid_input_preserved,
              "pending synchronization failure was not typed unsafe");
  ok &= Check(failed.status == PreparedCudaAutoFrameStatus::preserved_input &&
                  failed.output == &image && failed.valid_input_preserved,
              "ordinary quiescent analysis failure did not preserve input");

  FakeCrop failed_prepare;
  failed_prepare.prepare_ok = false;
  failed_prepare.shutdown_ok = false;
  PreparedCudaAutoFrame candidate(&failed_prepare);
  PreparedCudaAutoFrameHandle rejected;
  ok &=
      Check(!candidate.Prepare(s, &rejected, &error) &&
                failed_prepare.shutdowns == 1 &&
                error.find("post-allocation kernel preparation failed") == 0 &&
                error.find("cleanup pending") != std::string::npos,
            "failed preparation did not attempt rollback");
  failed_prepare.shutdown_ok = true;
  ok &= Check(candidate.Shutdown(&error) && failed_prepare.shutdowns == 2,
              "failed preparation cleanup was not publicly retryable");
  return ok;
}

bool NativePostAllocationCleanupIsRetryable() {
  DriverState state;
  auto driver = FakeDriver(&state);
  Analysis analysis;
  analysis.results = {Box(PreparedCudaAutoFrameSource::face, 1, 1, 10, 10)};
  FrameAnalysisCache cache;
  PreparedCudaAutoFrameSetup setup;
  setup.binding.driver = &driver;
  setup.binding.context = *driver.context_identity();
  setup.binding.owner_context = state.context;
  setup.binding.stream = reinterpret_cast<cuda::CUstream>(0x20);
  setup.binding.stream_identity = 1;
  setup.binding.owner_identity = 2;
  setup.binding.configuration_generation = 3;
  setup.binding.model_generation = 4;
  setup.frame_width = 64;
  setup.frame_height = 48;
  setup.face = &analysis;
  setup.analysis_cache = &cache;
  setup.resource_budget_bytes = 64 * 48 * 3;
  PreparedCudaAutoFrame component;
  PreparedCudaAutoFrameHandle handle;
  std::string error;
  bool ok =
      Check(setup.binding.context.retained_context_id !=
                reinterpret_cast<std::uintptr_t>(setup.binding.owner_context),
            "diagnostic retained context id was treated as CUDA handle");
  ok &= Check(!component.Prepare(setup, &handle, &error) &&
                  error.find("pitched output exceeds") != std::string::npos &&
                  error.find("cleanup pending") != std::string::npos &&
                  state.frees == 0,
              "native post-allocation failure lost primary/pending state");
  ok &= Check(component.Shutdown(&error) && state.frees == 1 &&
                  state.synchronizations == 2,
              "native post-allocation cleanup did not retry");
  return ok;
}

bool MatteBoxReference() {
  std::vector<float> alpha(20 * 20);
  for (int y = 4; y < 16; ++y)
    for (int x = 5; x < 15; ++x)
      alpha[y * 20 + x] = 1;
  PreparedCudaAutoFrameTrackingBox b;
  bool ok = Check(PreparedCudaAutoFrameMatteAdapter::ComputeBox(
                      alpha.data(), 20, 20, 200, 100, &b) &&
                      b.box.x == 50 && b.box.y == 20 && b.box.w == 100 &&
                      b.box.h == 60,
                  "matte bbox changed");
  FrameMatteArtifactKey key;
  key.provider_id = "open_cuda";
  key.model_id = "matte-v1";
  key.storage = FrameMatteStorage::cpu_f32_alpha;
  key.frame_width = 200;
  key.frame_height = 100;
  key.matte_width = 20;
  key.matte_height = 20;
  key.device_context = 7;
  key.stream = 3;
  FrameArtifactCache artifacts;
  FrameMatteArtifact artifact;
  artifact.key = key;
  artifact.cpu_alpha = alpha;
  (void)artifacts.StoreMatte(42, std::move(artifact));
  FrameAnalysisCache cache;
  cache.capture_sequence = 42;
  cache.artifacts = std::move(artifacts);
  PreparedCudaAutoFrameMatteAdapter adapter(nullptr, nullptr, key, 200, 100);
  auto binding = Binding();
  auto image = Image();
  const auto shared = adapter.Analyze(binding, image, 42, &cache, {});
  ok &= Check(adapter.HasSameFrameArtifact(42, &cache) && shared.ok &&
                  shared.found && shared.box.box.x == 50 &&
                  shared.transfers.intermediate_readbacks == 0,
              "compatible same-frame matte artifact was not reused");
  key.model_id = "other";
  PreparedCudaAutoFrameMatteAdapter stale(nullptr, nullptr, key, 200, 100);
  ok &= Check(!stale.HasSameFrameArtifact(42, &cache) &&
                  !stale.Analyze(binding, image, 42, &cache, {}).ok,
              "foreign matte artifact was accepted");
  return ok;
}
} // namespace
int main() {
  bool ok = GeometryAndSequenceReset();
  ok &= MatteCadenceCountersAndFailures();
  ok &= StrictFailureAndRetryableShutdown();
  ok &= UnsafeAnalysisFailuresAndPrepareCleanup();
  ok &= NativePostAllocationCleanupIsRetryable();
  ok &= MatteBoxReference();
  return ok ? 0 : 1;
}
