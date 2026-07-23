#include "core/cuda/kernels/prepared_open_cuda_vb_kernel_bundle.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>

namespace {

using namespace studiocast::cuda;
using namespace studiocast::cuda::kernels;

struct FakeState {
  CUcontext current = nullptr;
  int module_loads = 0;
  int function_resolutions = 0;
  int allocations = 0;
  int launches = 0;
  int context_queries = 0;
  CUresult module_load_result = CUDA_SUCCESS;
  CUresult function_resolution_result = CUDA_SUCCESS;
  CUresult launch_result = CUDA_SUCCESS;
  CUfunction last_function = nullptr;
  CUstream last_stream = nullptr;
  unsigned int last_grid_x = 0;
  unsigned int last_grid_y = 0;
  unsigned int last_block_x = 0;
  unsigned int last_block_y = 0;
  std::unordered_map<std::string, CUfunction> functions;
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
  const auto value =
      static_cast<std::uintptr_t>(0x300 + g_state->function_resolutions);
  *function = reinterpret_cast<CUfunction>(value);
  g_state->functions.emplace(name, *function);
  return CUDA_SUCCESS;
}

CUresult FakeLaunch(CUfunction function, unsigned int grid_x,
                    unsigned int grid_y, unsigned int, unsigned int block_x,
                    unsigned int block_y, unsigned int, unsigned int,
                    CUstream stream, void **, void **) {
  ++g_state->launches;
  g_state->last_function = function;
  g_state->last_stream = stream;
  g_state->last_grid_x = grid_x;
  g_state->last_grid_y = grid_y;
  g_state->last_block_x = block_x;
  g_state->last_block_y = block_y;
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
CUresult FakeStreamSynchronize(CUstream) { return CUDA_SUCCESS; }
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
  uuid->bytes[0] = 0x5a;
  uuid->bytes[15] = 0x17;
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
  PreparedOpenCudaVbBinding binding;
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
    binding.stream_identity = 17;
    binding.configuration_generation = 41;
    return true;
  }
};

PreparedOpenCudaVbLaunch ValidLaunch(OpenCudaVbKernelId kernel) {
  static std::uint64_t argument = 0;
  static void *arguments[] = {&argument};
  PreparedOpenCudaVbLaunch launch;
  launch.kernel = kernel;
  launch.grid_x = 4;
  launch.grid_y = 3;
  launch.block_x = 16;
  launch.block_y = 16;
  launch.parameters = arguments;
  return launch;
}

bool TestPrepareIsSetupOnlyAndReuseIsDeterministic() {
  Fixture fixture;
  if (!fixture.Initialize())
    return false;

  PreparedOpenCudaVbKernelBundle bundle;
  bool ok =
      Expect(bundle.Prepare(fixture.binding, bundle.RequiredResourceBytes(),
                            &fixture.error),
             "prepared bundle setup should succeed");
  ok &= Expect(fixture.state.module_loads == 1,
               "setup should load exactly one embedded module");
  ok &= Expect(fixture.state.function_resolutions ==
                   static_cast<int>(kOpenCudaVbKernelCount),
               "setup should resolve every function exactly once");
  constexpr std::string_view kExpectedSymbols[] = {
      "resize_bilinear_f32_1",
      "box_blur_h_u8x3",
      "box_blur_v_u8x3",
      "box_blur_h_f32_1",
      "box_blur_v_f32_1",
      "composite_alpha_u8x3_bg",
      "composite_alpha_u8x3_solid",
      "key_light_u8x3",
  };
  for (const auto symbol : kExpectedSymbols) {
    ok &= Expect(fixture.state.functions.count(std::string(symbol)) == 1,
                 "setup should resolve the complete compatibility PTX ABI");
  }
  ok &= Expect(fixture.state.allocations == 0,
               "setup should not allocate CUDA scratch");
  ok &= Expect(bundle.evidence().api_version ==
                   kPreparedOpenCudaVbKernelBundleApiVersion,
               "typed evidence should publish the stable API version");

  const int loads_after_prepare = fixture.state.module_loads;
  const int resolutions_after_prepare = fixture.state.function_resolutions;
  const int allocations_after_prepare = fixture.state.allocations;
  const int context_queries_after_prepare = fixture.state.context_queries;

  PreparedOpenCudaVbKernelStatus status{};
  const auto resize = ValidLaunch(OpenCudaVbKernelId::resize_f32);
  ok &= Expect(bundle.Launch(fixture.binding, resize, &status),
               "first prepared launch should enqueue");
  ok &= Expect(status == PreparedOpenCudaVbKernelStatus::launch_enqueued,
               "successful launch should produce typed enqueued evidence");
  const CUfunction first_function = fixture.state.last_function;
  ok &= Expect(first_function ==
                   fixture.state.functions.at("resize_bilinear_f32_1"),
               "resize id should resolve to the compatibility PTX symbol");
  ok &= Expect(
      fixture.state.last_stream == fixture.binding.stream &&
          fixture.state.last_grid_x == 4 && fixture.state.last_grid_y == 3 &&
          fixture.state.last_block_x == 16 && fixture.state.last_block_y == 16,
      "prepared launch should preserve stream and launch metadata");

  ok &= Expect(bundle.Launch(fixture.binding, resize, &status),
               "reused prepared launch should enqueue");
  ok &= Expect(fixture.state.last_function == first_function &&
                   bundle.evidence().successful_launches == 2,
               "prepared reuse should be deterministic");
  ok &= Expect(
      fixture.state.module_loads == loads_after_prepare &&
          fixture.state.function_resolutions == resolutions_after_prepare &&
          fixture.state.allocations == allocations_after_prepare &&
          fixture.state.context_queries == context_queries_after_prepare + 2,
      "launches must not load, resolve, or allocate; each launch "
      "performs one bounded raw-context currentness check");
  return ok;
}

bool TestBudgetContextStreamAndGenerationRejections() {
  Fixture fixture;
  if (!fixture.Initialize())
    return false;

  bool ok = true;
  PreparedOpenCudaVbKernelBundle exhausted;
  ok &= Expect(!exhausted.Prepare(fixture.binding,
                                  exhausted.RequiredResourceBytes() - 1,
                                  &fixture.error) &&
                   exhausted.evidence().status ==
                       PreparedOpenCudaVbKernelStatus::resource_exhausted,
               "insufficient setup budget should fail closed");
  ok &= Expect(fixture.state.module_loads == 0 &&
                   fixture.state.function_resolutions == 0 &&
                   fixture.state.allocations == 0,
               "budget rejection should precede CUDA setup work");

  PreparedOpenCudaVbKernelBundle bundle;
  ok &= Expect(bundle.Prepare(fixture.binding, bundle.RequiredResourceBytes(),
                              &fixture.error),
               "bundle should prepare for binding rejection tests");
  const int launches = fixture.state.launches;
  auto launch = ValidLaunch(OpenCudaVbKernelId::key_light);
  PreparedOpenCudaVbKernelStatus status{};

  auto foreign_context = fixture.binding;
  foreign_context.context_identity.retained_context_id += 1;
  ok &= Expect(!bundle.Launch(foreign_context, launch, &status) &&
                   status == PreparedOpenCudaVbKernelStatus::foreign_context,
               "foreign context should fail closed");

  auto stale = fixture.binding;
  stale.context_identity.logical_generation += 1;
  ok &= Expect(!bundle.Launch(stale, launch, &status) &&
                   status ==
                       PreparedOpenCudaVbKernelStatus::stale_context_generation,
               "stale context generation should fail closed");

  const CUcontext prepared_context = fixture.state.current;
  fixture.state.current =
      reinterpret_cast<CUcontext>(static_cast<std::uintptr_t>(0x501));
  ok &= Expect(!bundle.Launch(fixture.binding, launch, &status) &&
                   status == PreparedOpenCudaVbKernelStatus::foreign_context,
               "foreign raw current context should fail closed");
  fixture.state.current = prepared_context;

  auto foreign_stream = fixture.binding;
  foreign_stream.stream =
      reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(0x601));
  ok &= Expect(!bundle.Launch(foreign_stream, launch, &status) &&
                   status == PreparedOpenCudaVbKernelStatus::foreign_stream,
               "foreign stream handle should fail closed");

  foreign_stream = fixture.binding;
  foreign_stream.stream_identity += 1;
  ok &= Expect(!bundle.Launch(foreign_stream, launch, &status) &&
                   status == PreparedOpenCudaVbKernelStatus::foreign_stream,
               "foreign stream generation should fail closed");

  auto reconfigured = fixture.binding;
  reconfigured.configuration_generation += 1;
  ok &= Expect(!bundle.Launch(reconfigured, launch, &status) &&
                   status == PreparedOpenCudaVbKernelStatus::reconfigured,
               "wrong configuration generation should fail closed");
  ok &= Expect(fixture.state.launches == launches,
               "binding rejections should not launch kernels");
  return ok;
}

bool TestSetupFailuresProduceTypedEvidence() {
  Fixture fixture;
  if (!fixture.Initialize())
    return false;

  bool ok = true;
  PreparedOpenCudaVbKernelBundle module_failure;
  fixture.state.module_load_result = 1;
  ok &= Expect(!module_failure.Prepare(fixture.binding,
                                       module_failure.RequiredResourceBytes(),
                                       &fixture.error) &&
                   module_failure.evidence().status ==
                       PreparedOpenCudaVbKernelStatus::module_load_failed,
               "module load failure should produce exact typed evidence");

  fixture.state.module_load_result = CUDA_SUCCESS;
  fixture.state.function_resolution_result = 2;
  PreparedOpenCudaVbKernelBundle function_failure;
  ok &=
      Expect(!function_failure.Prepare(fixture.binding,
                                       function_failure.RequiredResourceBytes(),
                                       &fixture.error) &&
                 function_failure.evidence().status ==
                     PreparedOpenCudaVbKernelStatus::function_resolution_failed,
             "function resolution failure should produce exact typed "
             "evidence");
  ok &= Expect(fixture.state.allocations == 0 && fixture.state.launches == 0,
               "setup failures should not allocate or launch");
  return ok;
}

bool TestDeviceLossReconfigurationAndShutdownQuarantine() {
  Fixture fixture;
  if (!fixture.Initialize())
    return false;

  PreparedOpenCudaVbKernelBundle bundle;
  bool ok =
      Expect(bundle.Prepare(fixture.binding, bundle.RequiredResourceBytes(),
                            &fixture.error),
             "bundle should prepare for lifecycle tests");
  PreparedOpenCudaVbKernelStatus status{};
  const auto launch = ValidLaunch(OpenCudaVbKernelId::composite_solid);

  fixture.state.launch_result = 709;
  ok &= Expect(!bundle.Launch(fixture.binding, launch, &status) &&
                   status == PreparedOpenCudaVbKernelStatus::device_lost &&
                   bundle.evidence().quarantined,
               "device loss should quarantine the prepared bundle");
  const int launches_after_loss = fixture.state.launches;
  fixture.state.launch_result = CUDA_SUCCESS;
  ok &= Expect(!bundle.Launch(fixture.binding, launch, &status) &&
                   status == PreparedOpenCudaVbKernelStatus::device_lost &&
                   fixture.state.launches == launches_after_loss,
               "quarantined bundle should reject without retrying CUDA");

  PreparedOpenCudaVbKernelBundle reconfigured;
  ok &= Expect(reconfigured.Prepare(fixture.binding,
                                    reconfigured.RequiredResourceBytes(),
                                    &fixture.error),
               "fresh bundle should prepare after the fake loss");
  reconfigured.InvalidateForReconfiguration(
      fixture.binding.configuration_generation + 1);
  ok &= Expect(!reconfigured.Launch(fixture.binding, launch, &status) &&
                   status == PreparedOpenCudaVbKernelStatus::reconfigured,
               "explicit reconfiguration should invalidate old launches");

  PreparedOpenCudaVbKernelBundle stopped;
  ok &= Expect(stopped.Prepare(fixture.binding, stopped.RequiredResourceBytes(),
                               &fixture.error),
               "fresh bundle should prepare for shutdown");
  stopped.Shutdown();
  ok &= Expect(!stopped.Launch(fixture.binding, launch, &status) &&
                   status == PreparedOpenCudaVbKernelStatus::shutdown,
               "shutdown should reject future launches");
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok &= TestPrepareIsSetupOnlyAndReuseIsDeterministic();
  ok &= TestBudgetContextStreamAndGenerationRejections();
  ok &= TestSetupFailuresProduceTypedEvidence();
  ok &= TestDeviceLossReconfigurationAndShutdownQuarantine();
  if (ok)
    std::cout << "Prepared Open CUDA VB kernel bundle tests passed.\n";
  return ok ? 0 : 1;
}
