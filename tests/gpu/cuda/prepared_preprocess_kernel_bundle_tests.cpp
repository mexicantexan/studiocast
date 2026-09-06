#include "core/cuda/kernels/prepared_preprocess_kernel_bundle.h"
#include "core/cuda/kernels/resize_bilinear.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <string>
#include <string_view>
#include <thread>

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

using namespace studiocast::cuda;
using namespace studiocast::cuda::kernels;

struct LaunchCapture {
  CUfunction function = nullptr;
  CUstream stream = nullptr;
  unsigned int grid_x = 0;
  unsigned int grid_y = 0;
  unsigned int block_x = 0;
  unsigned int block_y = 0;
  unsigned long long source_ptr = 0;
  std::uint32_t source_pitch = 0;
  std::uint32_t source_w = 0;
  std::uint32_t source_h = 0;
  unsigned long long destination_ptr = 0;
  std::uint32_t destination_w = 0;
  std::uint32_t destination_h = 0;
  float mean[3] = {};
  float inverse_std[3] = {};
  std::uint32_t destination_is_bgr = 0;
  std::uint32_t source_is_bgr = 0;
  float crop[4] = {};
};

bool SameCapture(const LaunchCapture &left,
                 const LaunchCapture &right) noexcept {
  if (left.function != right.function || left.stream != right.stream ||
      left.grid_x != right.grid_x || left.grid_y != right.grid_y ||
      left.block_x != right.block_x || left.block_y != right.block_y ||
      left.source_ptr != right.source_ptr ||
      left.source_pitch != right.source_pitch ||
      left.source_w != right.source_w || left.source_h != right.source_h ||
      left.destination_ptr != right.destination_ptr ||
      left.destination_w != right.destination_w ||
      left.destination_h != right.destination_h ||
      left.destination_is_bgr != right.destination_is_bgr ||
      left.source_is_bgr != right.source_is_bgr) {
    return false;
  }
  for (int channel = 0; channel < 3; ++channel) {
    if (left.mean[channel] != right.mean[channel] ||
        left.inverse_std[channel] != right.inverse_std[channel]) {
      return false;
    }
  }
  return true;
}

struct FakeState {
  CUcontext current = nullptr;
  int module_loads = 0;
  int function_resolutions = 0;
  int module_unloads = 0;
  int allocations = 0;
  int launches = 0;
  int context_queries = 0;
  int synchronizations = 0;
  CUresult context_query_result = CUDA_SUCCESS;
  CUresult module_load_result = CUDA_SUCCESS;
  CUresult function_resolution_result = CUDA_SUCCESS;
  CUresult launch_result = CUDA_SUCCESS;
  LaunchCapture capture;
};

FakeState *g_state = nullptr;

bool Expect(bool condition, std::string_view message) {
  if (!condition)
    std::cerr << "[FAIL] " << message << "\n";
  return condition;
}

CUresult FakeInit(unsigned int) { return CUDA_SUCCESS; }
CUresult FakeModuleLoad(CUmodule *module, const void *) {
  ++g_state->module_loads;
  if (g_state->module_load_result != CUDA_SUCCESS)
    return g_state->module_load_result;
  *module = reinterpret_cast<CUmodule>(static_cast<std::uintptr_t>(0x200));
  return CUDA_SUCCESS;
}
CUresult FakeModuleGetFunction(CUfunction *function, CUmodule,
                               const char *name) {
  ++g_state->function_resolutions;
  if (g_state->function_resolution_result != CUDA_SUCCESS)
    return g_state->function_resolution_result;
  const auto entry = std::string_view(name);
  if (entry != "preprocess_to_nchw_f32" &&
      entry != "resize_bilinear_u8x3")
    return 3;
  *function = reinterpret_cast<CUfunction>(static_cast<std::uintptr_t>(
      entry == "resize_bilinear_u8x3" ? 0x301 : 0x300));
  return CUDA_SUCCESS;
}
CUresult FakeModuleUnload(CUmodule) {
  ++g_state->module_unloads;
  return CUDA_SUCCESS;
}
CUresult FakeLaunch(CUfunction function, unsigned int grid_x,
                    unsigned int grid_y, unsigned int, unsigned int block_x,
                    unsigned int block_y, unsigned int, unsigned int,
                    CUstream stream, void **arguments, void **) {
  ++g_state->launches;
  auto &capture = g_state->capture;
  capture.function = function;
  capture.stream = stream;
  capture.grid_x = grid_x;
  capture.grid_y = grid_y;
  capture.block_x = block_x;
  capture.block_y = block_y;
  if (function == reinterpret_cast<CUfunction>(
                      static_cast<std::uintptr_t>(0x301))) {
    capture.crop[0] = *static_cast<float *>(arguments[8]);
    capture.crop[1] = *static_cast<float *>(arguments[9]);
    capture.crop[2] = *static_cast<float *>(arguments[10]);
    capture.crop[3] = *static_cast<float *>(arguments[11]);
    return g_state->launch_result;
  }
  capture.source_ptr = *static_cast<unsigned long long *>(arguments[0]);
  capture.source_pitch = *static_cast<std::uint32_t *>(arguments[1]);
  capture.source_w = *static_cast<std::uint32_t *>(arguments[2]);
  capture.source_h = *static_cast<std::uint32_t *>(arguments[3]);
  capture.destination_ptr = *static_cast<unsigned long long *>(arguments[4]);
  capture.destination_w = *static_cast<std::uint32_t *>(arguments[5]);
  capture.destination_h = *static_cast<std::uint32_t *>(arguments[6]);
  capture.mean[0] = *static_cast<float *>(arguments[7]);
  capture.mean[1] = *static_cast<float *>(arguments[8]);
  capture.mean[2] = *static_cast<float *>(arguments[9]);
  capture.inverse_std[0] = *static_cast<float *>(arguments[10]);
  capture.inverse_std[1] = *static_cast<float *>(arguments[11]);
  capture.inverse_std[2] = *static_cast<float *>(arguments[12]);
  capture.destination_is_bgr = *static_cast<std::uint32_t *>(arguments[13]);
  capture.source_is_bgr = *static_cast<std::uint32_t *>(arguments[14]);
  return g_state->launch_result;
}
CUresult FakeAlloc(CUdeviceptr *ptr, std::size_t *pitch, std::size_t width,
                   std::size_t, unsigned int) {
  ++g_state->allocations;
  *ptr = 0x1000;
  *pitch = width;
  return CUDA_SUCCESS;
}
CUresult FakeFree(CUdeviceptr) { return CUDA_SUCCESS; }
CUresult FakeCopy(const CUDA_MEMCPY2D *, CUstream) { return CUDA_SUCCESS; }
CUresult FakeMemset(CUdeviceptr, unsigned char, std::size_t, CUstream) {
  return CUDA_SUCCESS;
}
CUresult FakeStreamCreate(CUstream *stream, unsigned int) {
  *stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(0x400));
  return CUDA_SUCCESS;
}
CUresult FakeStreamDestroy(CUstream) { return CUDA_SUCCESS; }
CUresult FakeStreamSynchronize(CUstream) {
  ++g_state->synchronizations;
  return CUDA_SUCCESS;
}
CUresult FakeDeviceGetCount(int *count) {
  *count = 1;
  return CUDA_SUCCESS;
}
CUresult FakeDeviceGet(CUdevice *device, int ordinal) {
  *device = ordinal;
  return CUDA_SUCCESS;
}
CUresult FakeDeviceUuid(CUuuid *uuid, CUdevice) {
  std::memset(uuid->bytes, 0, sizeof(uuid->bytes));
  uuid->bytes[0] = 0x63;
  uuid->bytes[15] = 0x29;
  return CUDA_SUCCESS;
}
CUresult FakeDevicePci(char *value, int size, CUdevice) {
  constexpr char kPci[] = "0000:01:00.0";
  if (size < static_cast<int>(sizeof(kPci)))
    return 1;
  std::memcpy(value, kPci, sizeof(kPci));
  return CUDA_SUCCESS;
}
CUresult FakePrimaryRetain(CUcontext *context, CUdevice) {
  *context = reinterpret_cast<CUcontext>(static_cast<std::uintptr_t>(0x500));
  return CUDA_SUCCESS;
}
CUresult FakePrimaryRelease(CUdevice) { return CUDA_SUCCESS; }
CUresult FakeCtxSet(CUcontext context) {
  g_state->current = context;
  return CUDA_SUCCESS;
}
CUresult FakeCtxGet(CUcontext *context) {
  ++g_state->context_queries;
  if (g_state->context_query_result != CUDA_SUCCESS)
    return g_state->context_query_result;
  *context = g_state->current;
  return CUDA_SUCCESS;
}
CUresult FakeCtxDevice(CUdevice *device) {
  *device = 0;
  return CUDA_SUCCESS;
}

CudaDriverApi::Functions FakeFunctions() {
  CudaDriverApi::Functions functions;
  functions.cuInit = FakeInit;
  functions.cuModuleLoadData = FakeModuleLoad;
  functions.cuModuleGetFunction = FakeModuleGetFunction;
  functions.cuModuleUnload = FakeModuleUnload;
  functions.cuLaunchKernel = FakeLaunch;
  functions.cuMemAllocPitch = FakeAlloc;
  functions.cuMemFree = FakeFree;
  functions.cuMemcpy2DAsync = FakeCopy;
  functions.cuMemsetD8Async = FakeMemset;
  functions.cuStreamCreate = FakeStreamCreate;
  functions.cuStreamDestroy = FakeStreamDestroy;
  functions.cuStreamSynchronize = FakeStreamSynchronize;
  functions.cuDeviceGetCount = FakeDeviceGetCount;
  functions.cuDeviceGet = FakeDeviceGet;
  functions.cuDeviceGetUuid = FakeDeviceUuid;
  functions.cuDeviceGetPCIBusId = FakeDevicePci;
  functions.cuDevicePrimaryCtxRetain = FakePrimaryRetain;
  functions.cuDevicePrimaryCtxRelease = FakePrimaryRelease;
  functions.cuCtxSetCurrent = FakeCtxSet;
  functions.cuCtxGetCurrent = FakeCtxGet;
  functions.cuCtxGetDevice = FakeCtxDevice;
  return functions;
}

struct Fixture {
  FakeState state;
  CudaDriverApi driver;
  PreparedPreprocessBinding binding;
  ModelPreprocessSpec spec;
  CudaImage source;
  CudaTensor destination;
  std::string error;

  bool Initialize() {
    g_state = &state;
    if (!driver.InitializeForTesting(FakeFunctions(), &error) ||
        !driver.EnsureContextForDevice(0, &error)) {
      std::cerr << "[FAIL] fake CUDA initialization: " << error << "\n";
      return false;
    }
    binding.driver = &driver;
    binding.context_identity = *driver.context_identity();
    binding.stream =
        reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(0x600));
    binding.stream_identity = 31;
    binding.configuration_generation = 41;
    binding.model_preprocess_generation = 43;

    spec.dst_w = 17;
    spec.dst_h = 18;
    spec.mean[0] = 0.1f;
    spec.mean[1] = 0.2f;
    spec.mean[2] = 0.3f;
    spec.std[0] = 0.5f;
    spec.std[1] = 0.25f;
    spec.std[2] = 2.0f;
    spec.dst_order = ChannelOrder::bgr;

    source.ptr = 0x7000;
    source.pitch = 32;
    source.w = 7;
    source.h = 9;
    source.format = PixelFormatGpu::bgr_u8;

    destination.ptr = 0x8000;
    destination.bytes = static_cast<std::size_t>(3) *
                        static_cast<std::size_t>(spec.dst_w) *
                        static_cast<std::size_t>(spec.dst_h) * sizeof(float);
    destination.pitch = destination.bytes;
    destination.n = 1;
    destination.c = 3;
    destination.h = spec.dst_h;
    destination.w = spec.dst_w;
    return true;
  }
};

bool TestSetupOnlyAndCompatibilityLaunchParity() {
  Fixture fixture;
  if (!fixture.Initialize())
    return false;

  PreparedPreprocessKernelBundle bundle;
  bool ok =
      Expect(bundle.Prepare(fixture.binding, fixture.spec,
                            bundle.RequiredResourceBytes(), &fixture.error),
             "explicit preprocess bundle should prepare");
  ok &= Expect(fixture.state.module_loads == 1 &&
                   fixture.state.function_resolutions == 1 &&
                   fixture.state.allocations == 0,
               "setup should load and resolve once without CUDA allocation");
  const auto setup_evidence = bundle.evidence();
  ok &= Expect(setup_evidence.api_version ==
                       kPreparedPreprocessKernelApiVersion &&
                   setup_evidence.launch.grid_x == 2 &&
                   setup_evidence.launch.grid_y == 2 &&
                   setup_evidence.launch.block_x == 16 &&
                   setup_evidence.launch.block_y == 16,
               "setup should freeze versioned launch geometry");

  const int loads_after_setup = fixture.state.module_loads;
  const int resolutions_after_setup = fixture.state.function_resolutions;
  const int allocations_after_setup = fixture.state.allocations;
  const std::size_t heap_allocations_after_setup =
      g_heap_allocations.load(std::memory_order_relaxed);
  PreparedPreprocessKernelStatus status{};
  ok &= Expect(bundle.Execute(fixture.binding, fixture.spec, fixture.source,
                              fixture.destination, &status) &&
                   status == PreparedPreprocessKernelStatus::launch_enqueued,
               "explicit prepared preprocess should enqueue");
  ok &= Expect(fixture.state.module_loads == loads_after_setup &&
                   fixture.state.function_resolutions ==
                       resolutions_after_setup &&
                   fixture.state.allocations == allocations_after_setup &&
                   g_heap_allocations.load(std::memory_order_relaxed) ==
                       heap_allocations_after_setup,
               "execution should not load, resolve, or allocate CUDA or host "
               "memory");

  const LaunchCapture explicit_capture = fixture.state.capture;
  ok &= Expect(
      explicit_capture.function == reinterpret_cast<CUfunction>(
                                       static_cast<std::uintptr_t>(0x300)) &&
          explicit_capture.stream == fixture.binding.stream &&
          explicit_capture.source_ptr == fixture.source.ptr &&
          explicit_capture.source_pitch == fixture.source.pitch &&
          explicit_capture.source_w ==
              static_cast<std::uint32_t>(fixture.source.w) &&
          explicit_capture.source_h ==
              static_cast<std::uint32_t>(fixture.source.h) &&
          explicit_capture.destination_ptr == fixture.destination.ptr &&
          explicit_capture.destination_w == 17 &&
          explicit_capture.destination_h == 18 &&
          explicit_capture.destination_is_bgr == 1 &&
          explicit_capture.source_is_bgr == 1 &&
          explicit_capture.mean[0] == fixture.spec.mean[0] &&
          explicit_capture.inverse_std[0] == 1.0f / fixture.spec.std[0],
      "explicit launch should preserve exact preprocess arguments");

  ok &= Expect(PreparePreprocessKernel(&fixture.driver, &fixture.error),
               "legacy prepare API should remain available");
  ok &= Expect(PreprocessToTensorPrepared(
                   &fixture.driver, fixture.binding.context_identity,
                   fixture.source, fixture.destination, fixture.spec,
                   fixture.binding.stream, &fixture.error),
               "legacy strict prepared API should remain unchanged");
  ok &= Expect(SameCapture(explicit_capture, fixture.state.capture),
               "explicit and compatibility launches should have exact "
               "argument and geometry parity");
  ok &=
      Expect(fixture.state.module_loads == loads_after_setup &&
                 fixture.state.function_resolutions == resolutions_after_setup,
             "legacy compatibility should reuse setup on its owning thread");
  return ok;
}

bool TestBudgetAndBindingFailuresFailClosed() {
  Fixture fixture;
  if (!fixture.Initialize())
    return false;

  bool ok = true;
  PreparedPreprocessKernelBundle exhausted;
  ok &= Expect(!exhausted.Prepare(fixture.binding, fixture.spec,
                                  exhausted.RequiredResourceBytes() - 1,
                                  &fixture.error) &&
                   exhausted.evidence().status ==
                       PreparedPreprocessKernelStatus::resource_exhausted,
               "insufficient resource budget should fail closed");
  ok &= Expect(fixture.state.module_loads == 0 &&
                   fixture.state.function_resolutions == 0,
               "budget rejection should precede CUDA module work");

  auto malformed_spec = fixture.spec;
  malformed_spec.std[2] = 0.0f;
  PreparedPreprocessKernelBundle malformed;
  ok &= Expect(!malformed.Prepare(fixture.binding, malformed_spec,
                                  malformed.RequiredResourceBytes(),
                                  &fixture.error) &&
                   malformed.evidence().status ==
                       PreparedPreprocessKernelStatus::invalid_argument,
               "invalid frozen preprocessing semantics should fail closed");
  ok &= Expect(fixture.state.module_loads == 0 &&
                   fixture.state.function_resolutions == 0,
               "malformed spec rejection should precede CUDA module work");

  PreparedPreprocessKernelBundle bundle;
  ok &= Expect(bundle.Prepare(fixture.binding, fixture.spec,
                              bundle.RequiredResourceBytes(), &fixture.error),
               "bundle should prepare for binding failure tests");
  const int launches = fixture.state.launches;
  PreparedPreprocessKernelStatus status{};

  auto foreign_context = fixture.binding;
  foreign_context.context_identity.retained_context_id += 1;
  ok &= Expect(!bundle.Execute(foreign_context, fixture.spec, fixture.source,
                               fixture.destination, &status) &&
                   status == PreparedPreprocessKernelStatus::foreign_context,
               "foreign context should fail closed");

  auto stale_context = fixture.binding;
  stale_context.context_identity.logical_generation += 1;
  ok &= Expect(!bundle.Execute(stale_context, fixture.spec, fixture.source,
                               fixture.destination, &status) &&
                   status ==
                       PreparedPreprocessKernelStatus::stale_context_generation,
               "stale context generation should fail closed");

  auto foreign_stream = fixture.binding;
  foreign_stream.stream =
      reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(0x601));
  ok &= Expect(!bundle.Execute(foreign_stream, fixture.spec, fixture.source,
                               fixture.destination, &status) &&
                   status == PreparedPreprocessKernelStatus::foreign_stream,
               "foreign stream handle should fail closed");
  foreign_stream = fixture.binding;
  foreign_stream.stream_identity += 1;
  ok &= Expect(!bundle.Execute(foreign_stream, fixture.spec, fixture.source,
                               fixture.destination, &status) &&
                   status == PreparedPreprocessKernelStatus::foreign_stream,
               "foreign stream generation should fail closed");

  auto reconfigured = fixture.binding;
  reconfigured.configuration_generation += 1;
  ok &= Expect(!bundle.Execute(reconfigured, fixture.spec, fixture.source,
                               fixture.destination, &status) &&
                   status == PreparedPreprocessKernelStatus::reconfigured,
               "foreign configuration generation should fail closed");
  reconfigured = fixture.binding;
  reconfigured.model_preprocess_generation += 1;
  ok &= Expect(!bundle.Execute(reconfigured, fixture.spec, fixture.source,
                               fixture.destination, &status) &&
                   status == PreparedPreprocessKernelStatus::reconfigured,
               "foreign model/preprocess generation should fail closed");

  auto foreign_spec = fixture.spec;
  foreign_spec.mean[1] += 0.01f;
  ok &= Expect(!bundle.Execute(fixture.binding, foreign_spec, fixture.source,
                               fixture.destination, &status) &&
                   status ==
                       PreparedPreprocessKernelStatus::foreign_preprocess_spec,
               "foreign preprocessing semantics should fail closed");

  const CUcontext prepared_context = fixture.state.current;
  fixture.state.current =
      reinterpret_cast<CUcontext>(static_cast<std::uintptr_t>(0x501));
  ok &= Expect(!bundle.Execute(fixture.binding, fixture.spec, fixture.source,
                               fixture.destination, &status) &&
                   status == PreparedPreprocessKernelStatus::foreign_context,
               "foreign raw current context should fail closed");
  fixture.state.current = prepared_context;

  CudaTensor invalid_destination;
  invalid_destination.ptr = fixture.destination.ptr;
  invalid_destination.pitch = fixture.destination.pitch;
  invalid_destination.bytes = fixture.destination.bytes;
  invalid_destination.n = fixture.destination.n;
  invalid_destination.c = fixture.destination.c;
  invalid_destination.h = fixture.destination.h;
  invalid_destination.w = fixture.destination.w;
  invalid_destination.bytes -= sizeof(float);
  ok &= Expect(!bundle.Execute(fixture.binding, fixture.spec, fixture.source,
                               invalid_destination, &status) &&
                   status == PreparedPreprocessKernelStatus::invalid_argument,
               "undersized destination should fail closed");
  ok &= Expect(fixture.state.launches == launches,
               "all malformed or foreign executions should reject before "
               "launch");
  return ok;
}

bool TestTypedSetupFailuresAndLifecycle() {
  Fixture fixture;
  if (!fixture.Initialize())
    return false;

  bool ok = true;
  fixture.state.module_load_result = 1;
  PreparedPreprocessKernelBundle module_failure;
  ok &= Expect(!module_failure.Prepare(fixture.binding, fixture.spec,
                                       module_failure.RequiredResourceBytes(),
                                       &fixture.error) &&
                   module_failure.evidence().status ==
                       PreparedPreprocessKernelStatus::module_load_failed,
               "module load failure should have exact typed evidence");

  fixture.state.module_load_result = CUDA_SUCCESS;
  fixture.state.function_resolution_result = 2;
  PreparedPreprocessKernelBundle function_failure;
  ok &=
      Expect(!function_failure.Prepare(fixture.binding, fixture.spec,
                                       function_failure.RequiredResourceBytes(),
                                       &fixture.error) &&
                 function_failure.evidence().status ==
                     PreparedPreprocessKernelStatus::function_resolution_failed,
             "function resolution failure should have exact typed evidence");

  fixture.state.function_resolution_result = CUDA_SUCCESS;
  PreparedPreprocessKernelStatus status{};
  PreparedPreprocessKernelBundle launch_failure;
  ok &= Expect(launch_failure.Prepare(fixture.binding, fixture.spec,
                                      launch_failure.RequiredResourceBytes(),
                                      &fixture.error),
               "bundle should prepare for ordinary launch failure");
  fixture.state.launch_result = 17;
  ok &= Expect(!launch_failure.Execute(fixture.binding, fixture.spec,
                                       fixture.source, fixture.destination,
                                       &status) &&
                   status == PreparedPreprocessKernelStatus::launch_failed &&
                   !launch_failure.evidence().quarantined,
               "ordinary launch failure should remain distinct from device "
               "loss");
  fixture.state.launch_result = CUDA_SUCCESS;
  ok &= Expect(launch_failure.Execute(fixture.binding, fixture.spec,
                                      fixture.source, fixture.destination,
                                      &status),
               "ordinary launch failure should permit owner-directed reuse");

  PreparedPreprocessKernelBundle device_loss;
  ok &= Expect(device_loss.Prepare(fixture.binding, fixture.spec,
                                   device_loss.RequiredResourceBytes(),
                                   &fixture.error),
               "bundle should prepare for device loss");
  fixture.state.launch_result = 709;
  ok &=
      Expect(!device_loss.Execute(fixture.binding, fixture.spec, fixture.source,
                                  fixture.destination, &status) &&
                 status == PreparedPreprocessKernelStatus::device_lost &&
                 device_loss.evidence().quarantined,
             "device loss should quarantine the prepared bundle");
  const int launches_after_loss = fixture.state.launches;
  fixture.state.launch_result = CUDA_SUCCESS;
  ok &=
      Expect(!device_loss.Execute(fixture.binding, fixture.spec, fixture.source,
                                  fixture.destination, &status) &&
                 status == PreparedPreprocessKernelStatus::device_lost &&
                 fixture.state.launches == launches_after_loss,
             "device-loss quarantine should reject without retry");

  PreparedPreprocessKernelBundle reconfigured;
  ok &= Expect(reconfigured.Prepare(fixture.binding, fixture.spec,
                                    reconfigured.RequiredResourceBytes(),
                                    &fixture.error),
               "fresh bundle should prepare for invalidation");
  reconfigured.InvalidateForReconfiguration(
      fixture.binding.configuration_generation + 1,
      fixture.binding.model_preprocess_generation);
  ok &= Expect(!reconfigured.Execute(fixture.binding, fixture.spec,
                                     fixture.source, fixture.destination,
                                     &status) &&
                   status == PreparedPreprocessKernelStatus::reconfigured,
               "explicit reconfiguration should invalidate evidence");

  PreparedPreprocessKernelBundle stopped;
  ok &= Expect(stopped.Prepare(fixture.binding, fixture.spec,
                               stopped.RequiredResourceBytes(), &fixture.error),
               "fresh bundle should prepare for shutdown");
  stopped.Shutdown();
  ok &= Expect(!stopped.Execute(fixture.binding, fixture.spec, fixture.source,
                                fixture.destination, &status) &&
                   status == PreparedPreprocessKernelStatus::shutdown,
               "shutdown should reject all further execution");
  return ok;
}

bool TestExplicitBundleExecutesAcrossThreadsWithoutThreadLocalEvidence() {
  Fixture fixture;
  if (!fixture.Initialize())
    return false;

  PreparedPreprocessKernelBundle bundle;
  bool ok =
      Expect(bundle.Prepare(fixture.binding, fixture.spec,
                            bundle.RequiredResourceBytes(), &fixture.error),
             "bundle should prepare on the setup thread");
  const int module_loads = fixture.state.module_loads;
  const int resolutions = fixture.state.function_resolutions;
  PreparedPreprocessKernelStatus thread_status =
      PreparedPreprocessKernelStatus::invalid;
  bool thread_result = false;
  std::thread execution_thread([&] {
    thread_result =
        bundle.Execute(fixture.binding, fixture.spec, fixture.source,
                       fixture.destination, &thread_status);
  });
  execution_thread.join();

  ok &= Expect(thread_result &&
                   thread_status ==
                       PreparedPreprocessKernelStatus::launch_enqueued,
               "explicit prepared handles should execute on a thread with no "
               "thread-local kernel cache");
  ok &= Expect(fixture.state.module_loads == module_loads &&
                   fixture.state.function_resolutions == resolutions &&
                   fixture.state.allocations == 0,
               "cross-thread execution should not load, resolve, or allocate");
  return ok;
}

bool TestPreparedResizeHasNoFrameTimeSetup() {
  Fixture fixture;
  if (!fixture.Initialize())
    return false;
  CudaImage destination;
  destination.ptr = 0x9000;
  destination.pitch = 17u * 3u;
  destination.w = 17;
  destination.h = 18;
  destination.format = PixelFormatGpu::bgr_u8;
  PreparedResizeBilinearKernel kernel;
  bool ok = Expect(PrepareResizeBilinearKernel(
                       fixture.driver, fixture.state.current,
                       fixture.binding.stream, &kernel, &fixture.error),
                   "prepared resize should load on the owner context");
  const int context_queries = fixture.state.context_queries;
  ok &= Expect(fixture.state.module_loads == 1 &&
                   fixture.state.function_resolutions == 1 && kernel.live,
               "prepared resize should load and resolve exactly once");
  ok &= Expect(LaunchCropResizeBilinearPrepared(
                   kernel, fixture.source, destination, -2.0f, -3.0f, 99.0f,
                   99.0f, &fixture.error) &&
                   LaunchResizeBilinearPrepared(
                       kernel, fixture.source, destination, &fixture.error),
               "prepared resize launches should enqueue");
  ok &= Expect(fixture.state.launches == 2 &&
                   fixture.state.module_loads == 1 &&
                   fixture.state.function_resolutions == 1 &&
                   fixture.state.context_queries == context_queries &&
                   fixture.state.allocations == 0 &&
                   fixture.state.synchronizations == 0,
               "repeated resize launches performed frame-time setup or sync");
  ok &= Expect(fixture.state.capture.stream == fixture.binding.stream &&
                   fixture.state.capture.grid_x == 2 &&
                   fixture.state.capture.grid_y == 2 &&
                   fixture.state.capture.crop[0] == 0.0f &&
                   fixture.state.capture.crop[1] == 0.0f &&
                   fixture.state.capture.crop[2] == 7.0f &&
                   fixture.state.capture.crop[3] == 9.0f,
               "prepared resize changed stream, geometry, or crop clamping");
  CudaImage invalid;
  invalid.ptr = fixture.source.ptr;
  invalid.pitch = fixture.source.pitch;
  invalid.w = fixture.source.w;
  invalid.h = fixture.source.h;
  invalid.format = PixelFormatGpu::f32_1;
  ok &= Expect(!LaunchResizeBilinearPrepared(
                    kernel, invalid, destination, &fixture.error) &&
                   fixture.state.launches == 2,
               "invalid prepared resize input reached CUDA");
  ok &= Expect(DestroyResizeBilinearKernelPrepared(&kernel, &fixture.error) &&
                   fixture.state.module_unloads == 1 && !kernel.live &&
                   DestroyResizeBilinearKernelPrepared(&kernel,
                                                       &fixture.error) &&
                   fixture.state.module_unloads == 1,
               "prepared resize cleanup should unload exactly once");
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok &= TestSetupOnlyAndCompatibilityLaunchParity();
  ok &= TestBudgetAndBindingFailuresFailClosed();
  ok &= TestTypedSetupFailuresAndLifecycle();
  ok &= TestExplicitBundleExecutesAcrossThreadsWithoutThreadLocalEvidence();
  ok &= TestPreparedResizeHasNoFrameTimeSetup();
  if (ok)
    std::cout << "Prepared preprocess kernel bundle tests passed.\n";
  return ok ? 0 : 1;
}
