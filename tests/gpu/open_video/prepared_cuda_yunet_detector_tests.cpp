#include "core/open_video/prepared_cuda_yunet_detector.h"
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
using namespace studiocast;
namespace {
struct State {
  cuda::CUcontext ctx = nullptr;
  std::vector<std::unique_ptr<std::byte[]>> memory;
  std::vector<std::size_t> sizes;
  int copies = 0, launches = 0, syncs = 0, runs = 0, invalidates = 0, frees = 0;
  int fail_copy = -1;
  bool face = false;
  cuda::CUresult query = cuda::CUDA_SUCCESS;
} * g;
cuda::CUresult OkInit(unsigned) { return 0; }
cuda::CUresult Mod(cuda::CUmodule *m, const void *) {
  *m = (cuda::CUmodule)1;
  return 0;
}
cuda::CUresult Fn(cuda::CUfunction *f, cuda::CUmodule, const char *) {
  *f = (cuda::CUfunction)2;
  return 0;
}
cuda::CUresult Unload(cuda::CUmodule) { return 0; }
cuda::CUresult Launch(cuda::CUfunction, unsigned, unsigned, unsigned, unsigned,
                      unsigned, unsigned, unsigned, cuda::CUstream, void **,
                      void **) {
  ++g->launches;
  return 0;
}
cuda::CUresult Alloc(cuda::CUdeviceptr *p, std::size_t *pi, std::size_t w,
                     std::size_t h, unsigned) {
  auto m = std::make_unique<std::byte[]>(w * h);
  *p = reinterpret_cast<cuda::CUdeviceptr>(m.get());
  *pi = w;
  g->memory.push_back(std::move(m));
  g->sizes.push_back(w * h);
  return 0;
}
cuda::CUresult Free(cuda::CUdeviceptr) {
  ++g->frees;
  return 0;
}
cuda::CUresult Copy(const cuda::CUDA_MEMCPY2D *c, cuda::CUstream) {
  int call = g->copies++;
  if (call == g->fail_copy)
    return cuda::CUDA_ERROR_OUT_OF_MEMORY;
  for (std::size_t y = 0; y < c->Height; ++y) {
    const auto *src =
        c->srcMemoryType == cuda::CU_MEMORYTYPE_HOST
            ? (const std::byte *)c->srcHost + y * c->srcPitch
            : (const std::byte *)(std::uintptr_t)c->srcDevice + y * c->srcPitch;
    auto *dst =
        c->dstMemoryType == cuda::CU_MEMORYTYPE_HOST
            ? (std::byte *)c->dstHost + y * c->dstPitch
            : (std::byte *)(std::uintptr_t)c->dstDevice + y * c->dstPitch;
    std::memcpy(dst, src, c->WidthInBytes);
  }
  return 0;
}
cuda::CUresult Memset(cuda::CUdeviceptr p, unsigned char v, std::size_t n,
                      cuda::CUstream) {
  std::memset((void *)(std::uintptr_t)p, v, n);
  return 0;
}
cuda::CUresult StreamCreate(cuda::CUstream *s, unsigned) {
  *s = (cuda::CUstream)3;
  return 0;
}
cuda::CUresult StreamDestroy(cuda::CUstream) { return 0; }
cuda::CUresult Sync(cuda::CUstream) {
  ++g->syncs;
  return 0;
}
cuda::CUresult HostAlloc(void **p, std::size_t n, unsigned) {
  auto m = std::make_unique<std::byte[]>(n);
  *p = m.get();
  g->memory.push_back(std::move(m));
  g->sizes.push_back(n);
  return 0;
}
cuda::CUresult HostFree(void *) { return 0; }
cuda::CUresult EventCreate(cuda::CUevent *e, unsigned) {
  *e = (cuda::CUevent)4;
  return 0;
}
cuda::CUresult EventDestroy(cuda::CUevent) { return 0; }
cuda::CUresult EventRecord(cuda::CUevent, cuda::CUstream) { return 0; }
cuda::CUresult EventQuery(cuda::CUevent) { return g->query; }
cuda::CUresult Elapsed(float *m, cuda::CUevent, cuda::CUevent) {
  *m = .1f;
  return 0;
}
cuda::CUresult Count(int *c) {
  *c = 1;
  return 0;
}
cuda::CUresult Device(cuda::CUdevice *d, int i) {
  *d = i;
  return 0;
}
cuda::CUresult Uuid(cuda::CUuuid *u, cuda::CUdevice) {
  std::memset(u, 0, sizeof(*u));
  u->bytes[0] = 1;
  return 0;
}
cuda::CUresult Pci(char *p, int n, cuda::CUdevice) {
  if (n < 2)
    return 1;
  std::strcpy(p, "x");
  return 0;
}
cuda::CUresult Retain(cuda::CUcontext *c, cuda::CUdevice) {
  *c = (cuda::CUcontext)5;
  return 0;
}
cuda::CUresult Release(cuda::CUdevice) { return 0; }
cuda::CUresult Set(cuda::CUcontext c) {
  g->ctx = c;
  return 0;
}
cuda::CUresult Get(cuda::CUcontext *c) {
  *c = g->ctx;
  return 0;
}
cuda::CUresult GetDev(cuda::CUdevice *d) {
  *d = 0;
  return 0;
}
cuda::CudaDriverApi::Functions Functions() {
  cuda::CudaDriverApi::Functions f;
  f.cuInit = OkInit;
  f.cuModuleLoadData = Mod;
  f.cuModuleGetFunction = Fn;
  f.cuModuleUnload = Unload;
  f.cuLaunchKernel = Launch;
  f.cuMemAllocPitch = Alloc;
  f.cuMemFree = Free;
  f.cuMemcpy2DAsync = Copy;
  f.cuMemsetD8Async = Memset;
  f.cuStreamCreate = StreamCreate;
  f.cuStreamDestroy = StreamDestroy;
  f.cuStreamSynchronize = Sync;
  f.cuMemHostAlloc = HostAlloc;
  f.cuMemFreeHost = HostFree;
  f.cuEventCreate = EventCreate;
  f.cuEventDestroy = EventDestroy;
  f.cuEventRecord = EventRecord;
  f.cuEventQuery = EventQuery;
  f.cuEventElapsedTime = Elapsed;
  f.cuDeviceGetCount = Count;
  f.cuDeviceGet = Device;
  f.cuDeviceGetUuid = Uuid;
  f.cuDeviceGetPCIBusId = Pci;
  f.cuDevicePrimaryCtxRetain = Retain;
  f.cuDevicePrimaryCtxRelease = Release;
  f.cuCtxSetCurrent = Set;
  f.cuCtxGetCurrent = Get;
  f.cuCtxGetDevice = GetDev;
  return f;
}
struct Session final : open_video::internal::IPreparedCudaYunetSession {
  onnx::OrtSessionInfo i;
  State *s;
  explicit Session(State *x, bool valid = true, bool rank4 = false) : s(x) {
    i.strict_provider_validation_passed = valid;
    i.input_names = {"input"};
    i.input_shapes = {{1, 3, 320, 320}};
    for (auto n : {"cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
                   "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32"})
      i.output_names.emplace_back(n);
    for (int k = 0; k < 12; ++k) {
      int st = k % 3 == 0 ? 8 : (k % 3 == 1 ? 16 : 32),
          ch = k < 6 ? 1 : (k < 9 ? 4 : 10);
      if (rank4)
        i.output_shapes.push_back({1, 320 / st, 320 / st, ch});
      else
        i.output_shapes.push_back({1, (320 / st) * (320 / st), ch});
    }
    i.prepared_provider_evidence = {};
    i.prepared_provider_evidence.provider = onnx::OrtExecutionProviderId::cuda;
    i.prepared_provider_evidence.cuda_device_id = 0;
    i.prepared_provider_evidence.cuda_context_generation = 1;
    i.prepared_provider_evidence.requested_stream_identity = 3;
    i.prepared_provider_evidence.user_stream_bound = true;
    i.prepared_provider_evidence.cpu_ep_fallback_disabled = true;
  }
  const onnx::OrtSessionInfo &info() const noexcept override { return i; }
  bool Prepare(const onnx::OrtSession::CudaBindingInput *, std::size_t ni,
               const onnx::OrtSession::CudaBindingOutput *, std::size_t no,
               std::string *) override {
    return ni == 1 && no == 12;
  }
  onnx::OrtPreparedCudaRunEvidence evidence() const noexcept override {
    onnx::OrtPreparedCudaRunEvidence e;
    e.provider = i.prepared_provider_evidence;
    e.session_generation = 1;
    return e;
  }
  onnx::OrtPreparedRunResult
  Execute(const onnx::OrtSession::CudaBindingInput *, std::size_t,
          const onnx::OrtSession::CudaBindingOutput *out, std::size_t n,
          const onnx::OrtPreparedCudaRunEvidence &) noexcept override {
    ++s->runs;
    for (std::size_t k = 0; k < n; ++k)
      std::memset(out[k].device_ptr, 0, out[k].num_floats * sizeof(float));
    if (s->face) {
      out[0].device_ptr[0] = 1.f;
      out[3].device_ptr[0] = 1.f;
      out[6].device_ptr[0] = out[6].device_ptr[1] = .5f;
      out[6].device_ptr[2] = out[6].device_ptr[3] = 0.f;
    }
    onnx::OrtPreparedRunResult r;
    r.status = onnx::OrtPreparedRunStatus::success;
    return r;
  }
  void Invalidate() noexcept override { ++s->invalidates; }
};
} // namespace
int main() {
  State state;
  g = &state;
  cuda::CudaDriverApi driver;
  std::string error;
  if (!driver.InitializeForTesting(Functions(), &error) ||
      !driver.EnsureContextForDevice(0, &error))
    return 1;
  open_video::PreparedCudaYunetSetup setup;
  setup.binding = {&driver, *driver.context_identity(), (cuda::CUstream)3, 3, 7,
                   9};
  setup.frame_width = 4;
  setup.frame_height = 3;
  setup.resource_budget_bytes =
      open_video::PreparedCudaYunetDetector::RequiredResourceBytes();
  open_video::ValidatedModelArtifactIdentity id;
  id.registry_generation = 1;
  id.model_id_sha256[0] = id.manifest_sha256[0] = id.artifact_sha256[0] =
      id.artifact_set_sha256[0] = 1;
  open_video::PreparedCudaYunetDetector d;
  if (!d.PrepareForTesting(setup, id, std::make_unique<Session>(&state),
                           &error)) {
    std::cerr << error;
    return 1;
  }
  open_video::FrameAnalysisCache cache;
  if (!d.PrepareCache(&cache, &error))
    return 1;
  cuda::CudaImage image;
  image.ptr = (cuda::CUdeviceptr)state.memory[0].get();
  image.pitch = 12;
  image.w = 4;
  image.h = 3;
  image.format = cuda::PixelFormatGpu::rgb_u8;
  cuda::CudaPreparedTransferCompletion::WaitPolicy wait;
  wait.timeout_nanoseconds = 100;
  wait.poll_interval_nanoseconds = 1;
  wait.maximum_poll_count = 2;
  const auto allocations = state.memory.size();
  auto r = d.Execute(setup.binding, image, 1, &cache, wait);
  bool ok = state.memory.size() == allocations &&
            r.status == open_video::PreparedCudaYunetStatus::success_no_faces &&
            r.enqueued_readbacks == 12 && r.completed_readbacks == 12 &&
            r.completion_waits == 1 && r.cpu_tail_boundaries == 1 &&
            state.runs == 2 && state.copies == 12 && !r.pending_work;
  state.fail_copy = state.copies + 3;
  auto partial = d.Execute(setup.binding, image, 2, &cache, wait);
  ok &=
      partial.status == open_video::PreparedCudaYunetStatus::readback_failed &&
      partial.enqueued_readbacks == 3 && partial.completed_readbacks == 3 &&
      !partial.pending_work && state.memory.size() == allocations;
  state.fail_copy = -1;
  state.face = true;
  auto recovered = d.Execute(setup.binding, image, 3, &cache, wait);
  ok &=
      recovered.status == open_video::PreparedCudaYunetStatus::success_faces &&
      cache.face_detections && cache.face_detections->size() == 1;
  state.face = false;
  ok &= d.Shutdown(&error) && d.Shutdown(&error) && state.invalidates == 1;
  open_video::PreparedCudaYunetDetector retry;
  auto retry_session = std::make_unique<Session>(&state);
  ok &= retry.PrepareForTesting(setup, id, std::move(retry_session), &error);
  open_video::FrameAnalysisCache retry_cache;
  ok &= retry.PrepareCache(&retry_cache, &error);
  state.query = cuda::CUDA_ERROR_NOT_READY;
  auto timed = retry.Execute(setup.binding, image, 4, &retry_cache, wait);
  bool first_shutdown = retry.Shutdown(&error);
  ok &=
      timed.status == open_video::PreparedCudaYunetStatus::completion_failed &&
      timed.pending_work && !first_shutdown;
  state.query = cuda::CUDA_SUCCESS;
  ok &= retry.Shutdown(&error);
  open_video::PreparedCudaYunetDetector rank4;
  ok &= rank4.PrepareForTesting(
            setup, id, std::make_unique<Session>(&state, true, true), &error) &&
        rank4.Shutdown(&error);
  open_video::PreparedCudaYunetDetector rejected;
  auto bad = std::make_unique<Session>(&state, false);
  ok &= !rejected.PrepareForTesting(setup, id, std::move(bad), &error) &&
        !rejected.ready();
  open_video::PreparedCudaYunetDetector missing;
  auto missing_setup = setup;
  missing_setup.models_root = "/definitely/missing/studiocast-yunet";
  missing_setup.requested_model_id = "missing";
  ok &= !missing.Prepare(missing_setup, &error) && !missing.ready();
  if (!ok) {
    std::cerr << "lifecycle failed\n";
    return 1;
  }
  std::cout << "PREPARED CUDA YUNET TESTS OK\n";
}
