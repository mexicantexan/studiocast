#include "core/open_video/prepared_cuda_yunet_detector.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <vector>
using namespace studiocast;

std::atomic<std::size_t> g_heap_allocations{0};
void *operator new(std::size_t size) {
  g_heap_allocations.fetch_add(1, std::memory_order_relaxed);
  if (void *memory = std::malloc(size))
    return memory;
  throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}
namespace {
constexpr const char *kYunetSha256 =
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4";
struct State {
  cuda::CUcontext ctx = nullptr;
  std::vector<std::unique_ptr<std::byte[]>> memory;
  std::vector<std::size_t> sizes;
  int copies = 0, launches = 0, syncs = 0, runs = 0, invalidates = 0, frees = 0;
  int allocations = 0, fail_allocation = -1;
  int fail_copy = -1;
  cuda::CUresult fail_copy_result = cuda::CUDA_ERROR_OUT_OF_MEMORY;
  int fail_run = -1;
  bool face = false;
  cuda::CUresult query = cuda::CUDA_SUCCESS;
  cuda::CUresult sync_result = cuda::CUDA_SUCCESS;
  cuda::CUresult launch_result = cuda::CUDA_SUCCESS;
  float preprocess_inverse_std = 0.0F;
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
                      unsigned, unsigned, unsigned, cuda::CUstream, void **args,
                      void **) {
  ++g->launches;
  g->preprocess_inverse_std = *static_cast<float *>(args[10]);
  return g->launch_result;
}
cuda::CUresult Alloc(cuda::CUdeviceptr *p, std::size_t *pi, std::size_t w,
                     std::size_t h, unsigned) {
  if (g->allocations++ == g->fail_allocation)
    return cuda::CUDA_ERROR_OUT_OF_MEMORY;
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
    return g->fail_copy_result;
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
  return g->sync_result;
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
bool Always(void *) noexcept { return true; }
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
  explicit Session(State *x, bool valid = true, bool rank4 = false,
                   int dim = 320)
      : s(x) {
    i.strict_provider_validation_passed = valid;
    i.input_names = {"input"};
    i.input_shapes = {{1, 3, dim, dim}};
    i.input_elem_types = {1};
    for (auto n : {"cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
                   "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32"})
      i.output_names.emplace_back(n);
    for (int k = 0; k < 12; ++k) {
      int st = k % 3 == 0 ? 8 : (k % 3 == 1 ? 16 : 32),
          ch = k < 6 ? 1 : (k < 9 ? 4 : 10);
      if (rank4)
        i.output_shapes.push_back({1, dim / st, dim / st, ch});
      else
        i.output_shapes.push_back({1, (dim / st) * (dim / st), ch});
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
    const int run = s->runs++;
    if (run == s->fail_run) {
      onnx::OrtPreparedRunResult failed;
      failed.status = onnx::OrtPreparedRunStatus::inference_failure;
      return failed;
    }
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

bool RunNativeTest(const std::filesystem::path &model) {
  if (!std::filesystem::is_regular_file(model) ||
      std::filesystem::file_size(model) != 232589) {
    std::cerr << "native YuNet artifact missing or wrong size: " << model
              << '\n';
    return false;
  }
  const auto fixture =
      std::filesystem::temp_directory_path() /
      ("studiocast-yunet-native-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto pack = fixture / "face_detection" / "yunet-native";
  std::error_code ec;
  std::filesystem::create_directories(pack, ec);
  if (ec)
    return false;
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  } cleanup{fixture};
  std::filesystem::copy_file(model, pack / "model.onnx",
                             std::filesystem::copy_options::overwrite_existing,
                             ec);
  if (ec)
    return false;
  std::ofstream manifest(pack / "model.json");
  manifest << R"({
  "schema_version": 2,
  "id": "yunet_native_acceptance",
  "display_name": "YuNet native acceptance",
  "task": "face_detection",
  "files": [{"name":"model.onnx","kind":"onnx","role":"main","sha256":")"
           << kYunetSha256 << R"("}],
  "onnx": {"main":"model.onnx","inputs":[{
    "name":"input","layout":"nchw","dtype":"float32",
    "width":640,"height":640,"channels":3
  }]}
})";
  manifest.close();
  std::ofstream(pack / "LICENSE.txt") << "Native test fixture only\n";

  cuda::CudaDriverApi native_driver;
  std::string error;
  cuda::CUstream stream = nullptr;
  cuda::CudaDriverApi::PitchAllocation image_storage;
  if (!native_driver.Initialize(&error) ||
      !native_driver.EnsureContextForDevice(0, &error) ||
      !native_driver.CreateStream(&stream, &error) ||
      !native_driver.AllocatePitch(64 * 3, 48, &image_storage, &error, 4)) {
    std::cerr << "native YuNet CUDA setup failed: " << error << '\n';
    return false;
  }
  std::vector<std::uint8_t> pixels(64 * 48 * 3);
  for (std::size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = static_cast<std::uint8_t>((i * 37u + 11u) & 0xffu);
  if (!native_driver.MemcpyHtoD2DAsync(image_storage.ptr, image_storage.pitch,
                                       pixels.data(), 64 * 3, 64 * 3, 48,
                                       stream, &error) ||
      !native_driver.StreamSynchronize(stream, &error)) {
    std::cerr << "native YuNet upload failed: " << error << '\n';
    return false;
  }
  open_video::PreparedCudaYunetSetup setup;
  setup.binding = {&native_driver,
                   *native_driver.context_identity(),
                   stream,
                   reinterpret_cast<std::uintptr_t>(stream),
                   1,
                   1};
  setup.models_root = fixture;
  setup.requested_model_id = "yunet_native_acceptance";
  setup.frame_width = 64;
  setup.frame_height = 48;
  setup.resource_budget_bytes =
      open_video::PreparedCudaYunetDetector::RequiredResourceBytes(640, 640);
  open_video::PreparedCudaYunetDetector detector;
  open_video::FrameAnalysisCache cache;
  bool ok =
      detector.Prepare(setup, &error) && detector.PrepareCache(&cache, &error);
  cuda::CudaImage image;
  image.ptr = image_storage.ptr;
  image.pitch = image_storage.pitch;
  image.w = 64;
  image.h = 48;
  image.format = cuda::PixelFormatGpu::rgb_u8;
  cuda::CudaPreparedTransferCompletion::WaitPolicy wait;
  wait.timeout_nanoseconds = 5'000'000'000ULL;
  wait.poll_interval_nanoseconds = 1'000ULL;
  wait.maximum_poll_count = 10'000'000;
  const auto result =
      ok ? detector.Execute(setup.binding, image, 1, &cache, wait)
         : open_video::PreparedCudaYunetExecutionResult{};
  ok =
      ok &&
      (result.status == open_video::PreparedCudaYunetStatus::success_no_faces ||
       result.status == open_video::PreparedCudaYunetStatus::success_faces) &&
      result.enqueued_readbacks == 12 && result.completed_readbacks == 12 &&
      result.completion_waits == 1 && result.cpu_tail_boundaries == 1 &&
      !result.pending_work;
  // Cleanup must run even when preparation or the result assertions failed.
  // Keep the input and owner stream alive until detector work is quiescent.
  bool stopped = detector.Shutdown(&error);
  if (!stopped && native_driver.StreamSynchronize(stream, &error))
    stopped = detector.Shutdown(&error);
  if (!stopped) {
    std::cerr << "native YuNet cleanup remains pending: " << error << '\n';
    return false;
  }
  ok = native_driver.Free(image_storage.ptr, &error) && ok;
  ok = native_driver.DestroyStream(stream, &error) && ok;
  if (!ok)
    std::cerr << "native YuNet acceptance failed: " << error << '\n';
  else
    std::cout << "PREPARED CUDA YUNET NATIVE TEST OK status="
              << (result.status ==
                          open_video::PreparedCudaYunetStatus::success_faces
                      ? "success_faces"
                      : "success_no_faces")
              << " readbacks=12 waits=1 cpu_tail=1\n";
  return ok;
}
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
  setup.declared_input = {"input", "nchw", "float32", 320, 320, 3};
  open_video::ValidatedModelArtifactIdentity id;
  id.registry_generation = 1;
  id.model_id_sha256[0] = id.manifest_sha256[0] = id.artifact_sha256[0] =
      id.artifact_set_sha256[0] = 1;
  for (const bool wrong_task : {true, false}) {
    open_video::ModelPack selected;
    selected.id = wrong_task ? "configured-face" : "different-face";
    selected.task = wrong_task ? "matting" : "face_detection";
    auto selected_setup = setup;
    selected_setup.requested_model_id = "configured-face";
    selected_setup.selected_pack = &selected;
    selected_setup.selected_artifact = id;
    const auto allocations = state.allocations;
    open_video::PreparedCudaYunetDetector rejected;
    open_video::PreparedCudaYunetStatus status;
    if (rejected.Prepare(selected_setup, &error, &status) ||
        status != open_video::PreparedCudaYunetStatus::model_invalid ||
        state.allocations != allocations) {
      std::cerr << "selected YuNet pack substituted requested task or model ID";
      return 1;
    }
  }
  open_video::PreparedCudaYunetDetector d;
  if (!d.PrepareForTesting(setup, id, std::make_unique<Session>(&state),
                           &error)) {
    std::cerr << error;
    return 1;
  }
  open_video::FrameAnalysisCache cache;
  if (!d.PrepareCache(&cache, &error))
    return 1;
  bool prepared_binding_ok = true;
  open_video::PreparedCudaYunetPreparedBinding prepared_binding;
  prepared_binding_ok &=
      d.ExportPreparedBinding(&cache, &prepared_binding) &&
      d.MatchesPreparedBinding(prepared_binding) &&
      prepared_binding.producer == &d && prepared_binding.cache == &cache &&
      prepared_binding.artifact == id && prepared_binding.frame_width == 4 &&
      prepared_binding.frame_height == 3 &&
      prepared_binding.producer_revision != 0;
  auto stale_prepared_binding = prepared_binding;
  ++stale_prepared_binding.binding.configuration_generation;
  prepared_binding_ok &= !d.MatchesPreparedBinding(stale_prepared_binding);
  open_video::FrameAnalysisCache foreign_cache;
  stale_prepared_binding = prepared_binding;
  stale_prepared_binding.cache = &foreign_cache;
  prepared_binding_ok &= !d.MatchesPreparedBinding(stale_prepared_binding);
  stale_prepared_binding = prepared_binding;
  ++stale_prepared_binding.artifact.registry_generation;
  prepared_binding_ok &= !d.MatchesPreparedBinding(stale_prepared_binding);
  stale_prepared_binding = prepared_binding;
  ++stale_prepared_binding.frame_width;
  prepared_binding_ok &= !d.MatchesPreparedBinding(stale_prepared_binding);
  prepared_binding_ok &=
      !d.ExportPreparedBinding(&foreign_cache, &stale_prepared_binding);
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
  const auto heap_allocations =
      g_heap_allocations.load(std::memory_order_relaxed);
  auto r = d.Execute(setup.binding, image, 1, &cache, wait);
  bool ok =
      prepared_binding_ok && state.memory.size() == allocations &&
      g_heap_allocations.load(std::memory_order_relaxed) == heap_allocations &&
      r.status == open_video::PreparedCudaYunetStatus::success_no_faces &&
      r.enqueued_readbacks == 12 && r.completed_readbacks == 12 &&
      r.completion_waits == 1 && r.cpu_tail_boundaries == 1 &&
      state.runs == 2 && state.copies == 12 && !r.pending_work &&
      std::abs(state.preprocess_inverse_std - 255.0F) < 0.001F;
  const int cached_runs = state.runs;
  const int cached_copies = state.copies;
  auto cached_no_face = d.Execute(setup.binding, image, 1, &cache, wait);
  ok &= cached_no_face.status ==
            open_video::PreparedCudaYunetStatus::success_no_faces &&
        cached_no_face.enqueued_readbacks == 0 &&
        cached_no_face.completed_readbacks == 0 &&
        cached_no_face.completion_waits == 0 &&
        cached_no_face.cpu_tail_boundaries == 0 && state.runs == cached_runs &&
        state.copies == cached_copies;
  cuda::CudaImage invalid_cached_input;
  invalid_cached_input.w = image.w;
  invalid_cached_input.h = image.h;
  invalid_cached_input.pitch = image.pitch;
  invalid_cached_input.format = cuda::PixelFormatGpu::rgb_u8;
  auto invalid_cached =
      d.Execute(setup.binding, invalid_cached_input, 1, &cache, wait);
  ok &= invalid_cached.status ==
            open_video::PreparedCudaYunetStatus::stale_binding &&
        state.runs == cached_runs && state.copies == cached_copies;
  cache.face_detections_provenance.producer = nullptr;
  auto wrong_producer = d.Execute(setup.binding, image, 1, &cache, wait);
  ok &= wrong_producer.status ==
            open_video::PreparedCudaYunetStatus::success_no_faces &&
        wrong_producer.enqueued_readbacks == 12 &&
        state.runs == cached_runs + 1;
  ++cache.face_detections_provenance.producer_revision;
  const int stale_revision_runs = state.runs;
  auto stale_revision = d.Execute(setup.binding, image, 1, &cache, wait);
  ok &= stale_revision.enqueued_readbacks == 12 &&
        state.runs == stale_revision_runs + 1;
  cache.BeginFrame(99);
  (void)cache.PrepareFaceDetections(16);
  const int empty_runs = state.runs;
  auto empty_unproven = d.Execute(setup.binding, image, 99, &cache, wait);
  ok &= empty_unproven.enqueued_readbacks == 12 && state.runs == empty_runs + 1;
  state.fail_copy = state.copies + 3;
  auto partial = d.Execute(setup.binding, image, 2, &cache, wait);
  ok &=
      partial.status == open_video::PreparedCudaYunetStatus::readback_failed &&
      partial.enqueued_readbacks == 3 && partial.completed_readbacks == 3 &&
      partial.completion_waits == 1 && partial.cpu_tail_boundaries == 0 &&
      !partial.pending_work && state.memory.size() == allocations;
  state.fail_copy = -1;
  state.face = true;
  auto recovered = d.Execute(setup.binding, image, 3, &cache, wait);
  ok &=
      recovered.status == open_video::PreparedCudaYunetStatus::success_faces &&
      cache.face_detections && cache.face_detections->size() == 1;
  const int face_runs = state.runs;
  const int face_copies = state.copies;
  auto cached_face = d.Execute(setup.binding, image, 3, &cache, wait);
  ok &= cached_face.status ==
            open_video::PreparedCudaYunetStatus::success_faces &&
        cached_face.enqueued_readbacks == 0 &&
        cached_face.completed_readbacks == 0 &&
        cached_face.completion_waits == 0 &&
        cached_face.cpu_tail_boundaries == 0 && state.runs == face_runs &&
        state.copies == face_copies;
  state.face = false;
  ok &= d.Shutdown(&error) && d.Shutdown(&error) && state.invalidates == 1;
  open_video::PreparedCudaYunetDetector allocation_failed;
  open_video::PreparedCudaYunetStatus allocation_status{};
  state.fail_allocation = state.allocations;
  ok &= !allocation_failed.PrepareForTesting(
            setup, id, std::make_unique<Session>(&state), &error,
            &allocation_status) &&
        allocation_status == open_video::PreparedCudaYunetStatus::invalid_setup;
  state.fail_allocation = -1;
  open_video::PreparedCudaYunetDetector retry;
  auto retry_session = std::make_unique<Session>(&state);
  ok &= retry.PrepareForTesting(setup, id, std::move(retry_session), &error);
  open_video::FrameAnalysisCache retry_cache;
  ok &= retry.PrepareCache(&retry_cache, &error);
  state.query = cuda::CUDA_ERROR_NOT_READY;
  auto timed = retry.Execute(setup.binding, image, 4, &retry_cache, wait);
  bool first_shutdown = retry.Shutdown(&error);
  ok &= timed.status == open_video::PreparedCudaYunetStatus::timeout &&
        timed.pending_work && !first_shutdown;
  state.query = cuda::CUDA_SUCCESS;
  ok &= retry.Shutdown(&error);
  open_video::PreparedCudaYunetDetector stopped;
  ok &= stopped.PrepareForTesting(setup, id, std::make_unique<Session>(&state),
                                  &error);
  open_video::FrameAnalysisCache stopped_cache;
  ok &= stopped.PrepareCache(&stopped_cache, &error);
  auto stopped_wait = wait;
  stopped_wait.stop_requested = Always;
  auto stopped_result =
      stopped.Execute(setup.binding, image, 41, &stopped_cache, stopped_wait);
  ok &= stopped_result.status ==
            open_video::PreparedCudaYunetStatus::stop_requested &&
        stopped_result.completion_waits == 1 && stopped_result.pending_work;
  state.query = cuda::CUDA_SUCCESS;
  ok &= stopped.Shutdown(&error);
  open_video::PreparedCudaYunetDetector inference_failed;
  ok &= inference_failed.PrepareForTesting(
      setup, id, std::make_unique<Session>(&state), &error);
  open_video::FrameAnalysisCache inference_cache;
  ok &= inference_failed.PrepareCache(&inference_cache, &error);
  state.fail_run = state.runs;
  auto inference_result = inference_failed.Execute(setup.binding, image, 42,
                                                   &inference_cache, wait);
  ok &= inference_result.status ==
            open_video::PreparedCudaYunetStatus::inference_failed &&
        inference_result.completion_waits == 1 &&
        !inference_result.pending_work;
  state.fail_run = -1;
  const int sync_loss_invalidates = state.invalidates;
  state.sync_result = cuda::CUDA_ERROR_CONTEXT_IS_DESTROYED;
  open_video::PreparedCudaYunetDetector warmup_unquiesced;
  const auto warmup_pending_allocations = state.memory.size();
  ok &= !warmup_unquiesced.PrepareForTesting(
            setup, id, std::make_unique<Session>(&state), &error) &&
        state.invalidates == sync_loss_invalidates &&
        !warmup_unquiesced.ready();
  const auto retained_allocations = state.memory.size();
  ok &= retained_allocations > warmup_pending_allocations &&
        !warmup_unquiesced.PrepareForTesting(
            setup, id, std::make_unique<Session>(&state), &error) &&
        state.memory.size() == retained_allocations &&
        !warmup_unquiesced.Prepare(setup, &error) &&
        error == "invalid prepared YuNet setup" &&
        state.memory.size() == retained_allocations &&
        !warmup_unquiesced.Shutdown(&error) &&
        state.invalidates == sync_loss_invalidates;
  state.sync_result = cuda::CUDA_SUCCESS;
  ok &= warmup_unquiesced.Shutdown(&error) &&
        state.invalidates == sync_loss_invalidates + 1;
  state.sync_result = cuda::CUDA_ERROR_CONTEXT_IS_DESTROYED;
  const int setup_destructor_invalidates = state.invalidates;
  const int setup_destructor_frees = state.frees;
  {
    open_video::PreparedCudaYunetDetector pending_setup_destructor;
    ok &= !pending_setup_destructor.PrepareForTesting(
        setup, id, std::make_unique<Session>(&state), &error);
  }
  ok &= state.invalidates == setup_destructor_invalidates &&
        state.frees == setup_destructor_frees;
  state.sync_result = cuda::CUDA_SUCCESS;
  open_video::PreparedCudaYunetDetector device_lost;
  ok &= device_lost.PrepareForTesting(
      setup, id, std::make_unique<Session>(&state), &error);
  open_video::FrameAnalysisCache device_lost_cache;
  ok &= device_lost.PrepareCache(&device_lost_cache, &error);
  state.launch_result = cuda::CUDA_ERROR_CONTEXT_IS_DESTROYED;
  auto device_lost_result =
      device_lost.Execute(setup.binding, image, 43, &device_lost_cache, wait);
  const bool device_lost_ok =
      device_lost_result.status ==
          open_video::PreparedCudaYunetStatus::device_lost &&
      device_lost_result.device_lost &&
      device_lost_result.completion_waits == 1 &&
      !device_lost_result.pending_work;
  ok &= device_lost_ok;
  state.launch_result = cuda::CUDA_SUCCESS;
  ok &= device_lost.Shutdown(&error);
  open_video::PreparedCudaYunetDetector copy_lost;
  ok &= copy_lost.PrepareForTesting(setup, id,
                                    std::make_unique<Session>(&state), &error);
  open_video::FrameAnalysisCache copy_lost_cache;
  ok &= copy_lost.PrepareCache(&copy_lost_cache, &error);
  state.fail_copy = state.copies;
  state.fail_copy_result = cuda::CUDA_ERROR_CONTEXT_IS_DESTROYED;
  auto copy_device_lost =
      copy_lost.Execute(setup.binding, image, 21, &copy_lost_cache, wait);
  ok &= copy_device_lost.status ==
            open_video::PreparedCudaYunetStatus::device_lost &&
        copy_device_lost.device_lost &&
        copy_device_lost.enqueued_readbacks == 0;
  state.fail_copy = -1;
  state.fail_copy_result = cuda::CUDA_ERROR_OUT_OF_MEMORY;
  ok &= copy_lost.Shutdown(&error);
  ok &= inference_failed.Shutdown(&error);
  const int invalidates_before_pending_destructor = state.invalidates;
  {
    open_video::PreparedCudaYunetDetector pending_destructor;
    ok &= pending_destructor.PrepareForTesting(
        setup, id, std::make_unique<Session>(&state), &error);
    open_video::FrameAnalysisCache pending_cache;
    ok &= pending_destructor.PrepareCache(&pending_cache, &error);
    state.query = cuda::CUDA_ERROR_NOT_READY;
    auto pending = pending_destructor.Execute(setup.binding, image, 40,
                                              &pending_cache, wait);
    ok &= pending.status == open_video::PreparedCudaYunetStatus::timeout &&
          pending.completion_waits == 1 && pending.pending_work;
  }
  ok &= state.invalidates == invalidates_before_pending_destructor;
  state.query = cuda::CUDA_SUCCESS;
  open_video::PreparedCudaYunetDetector size640;
  auto setup640 = setup;
  setup640.declared_input.width = 640;
  setup640.declared_input.height = 640;
  ok &= size640.PrepareForTesting(
            setup640, id, std::make_unique<Session>(&state, true, false, 640),
            &error) &&
        size640.Shutdown(&error);
  open_video::PreparedCudaYunetDetector wrong_geometry;
  ok &= !wrong_geometry.PrepareForTesting(
            setup, id, std::make_unique<Session>(&state, true, false, 640),
            &error) &&
        !wrong_geometry.ready();
  for (int mismatch = 0; mismatch < 3; ++mismatch) {
    auto bad_setup = setup;
    auto bad_session = std::make_unique<Session>(&state);
    if (mismatch == 0)
      bad_setup.declared_input.layout = "nhwc";
    else if (mismatch == 1)
      bad_setup.declared_input.dtype = "float16";
    else
      bad_session->i.input_names[0] = "wrong_input";
    open_video::PreparedCudaYunetDetector contract_rejected;
    ok &= !contract_rejected.PrepareForTesting(
              bad_setup, id, std::move(bad_session), &error) &&
          !contract_rejected.ready();
  }
  open_video::PreparedCudaYunetDetector rank4;
  ok &= rank4.PrepareForTesting(
            setup, id, std::make_unique<Session>(&state, true, true), &error) &&
        rank4.Shutdown(&error);
  open_video::PreparedCudaYunetDetector rejected;
  auto bad = std::make_unique<Session>(&state, false);
  ok &= !rejected.PrepareForTesting(setup, id, std::move(bad), &error) &&
        !rejected.ready();
  const int warmup_run = state.runs;
  const int warmup_syncs = state.syncs;
  const int warmup_invalidates = state.invalidates;
  state.fail_run = warmup_run;
  open_video::PreparedCudaYunetDetector warmup_failed;
  ok &= !warmup_failed.PrepareForTesting(
            setup, id, std::make_unique<Session>(&state), &error) &&
        state.syncs == warmup_syncs + 1 &&
        state.invalidates == warmup_invalidates + 1 && !warmup_failed.ready();
  state.fail_run = -1;
  open_video::PreparedCudaYunetDetector missing;
  auto missing_setup = setup;
  missing_setup.models_root = "/definitely/missing/studiocast-yunet";
  missing_setup.requested_model_id = "missing";
  ok &= !missing.Prepare(missing_setup, &error) && !missing.ready();
  open_video::PreparedCudaYunetDetector reprepared;
  open_video::FrameAnalysisCache reprepare_cache;
  open_video::PreparedCudaYunetPreparedBinding old_binding, new_binding;
  ok &= reprepared.PrepareForTesting(
            setup, id, std::make_unique<Session>(&state), &error) &&
        reprepared.PrepareCache(&reprepare_cache, &error) &&
        reprepared.ExportPreparedBinding(&reprepare_cache, &old_binding) &&
        reprepared.Shutdown(&error) &&
        reprepared.PrepareForTesting(
            setup, id, std::make_unique<Session>(&state), &error) &&
        reprepared.PrepareCache(&reprepare_cache, &error) &&
        reprepared.ExportPreparedBinding(&reprepare_cache, &new_binding) &&
        old_binding.producer_revision != new_binding.producer_revision &&
        !reprepared.MatchesPreparedBinding(old_binding) &&
        reprepared.MatchesPreparedBinding(new_binding) &&
        reprepared.Shutdown(&error);
  if (!ok) {
    std::cerr << "lifecycle failed runs=" << state.runs
              << " copies=" << state.copies << " syncs=" << state.syncs
              << " invalidates=" << state.invalidates
              << " inverse_std=" << state.preprocess_inverse_std << '\n';
    return 1;
  }
  if (const char *native_model =
          std::getenv("STUDIOCAST_TEST_YUNET_NATIVE_MODEL")) {
    if (!RunNativeTest(native_model))
      return 1;
  } else {
    std::cout << "PREPARED CUDA YUNET NATIVE TEST SKIPPED\n";
  }
  std::cout << "PREPARED CUDA YUNET TESTS OK\n";
}
