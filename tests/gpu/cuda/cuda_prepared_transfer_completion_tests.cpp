#include "core/cuda/cuda_prepared_transfer_completion.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

namespace {

namespace cuda = studiocast::cuda;
using Support = cuda::CudaPreparedTransferCompletion;

static_assert(Support::kApiVersion == 1);
static_assert(static_cast<std::uint8_t>(Support::Status::ok) == 0);
static_assert(static_cast<std::uint8_t>(Support::Status::resource_exhausted) ==
              10);
static_assert(static_cast<std::uint8_t>(Support::Status::already_prepared) ==
              18);
static_assert(
    static_cast<std::uint8_t>(Support::Status::resource_release_failed) == 19);

bool Expect(bool condition, const char *message) {
  if (!condition)
    std::cerr << "[FAIL] " << message << '\n';
  return condition;
}

struct FakeDriver {
  alignas(64) std::array<std::array<std::byte, 4096>, 16> pinned{};
  std::array<const void *, 16> freed_pointers{};
  std::array<cuda::CUDA_MEMCPY2D, 16> copies{};
  std::array<cuda::CUstream, 16> copy_streams{};
  cuda::CUcontext current = nullptr;
  int pinned_allocations = 0;
  int pinned_frees = 0;
  int event_creations = 0;
  int event_destructions = 0;
  int event_records = 0;
  int event_queries = 0;
  int elapsed_queries = 0;
  int copy_calls = 0;
  int context_sets = 0;
  int context_queries = 0;
  int context_releases = 0;
  int fail_event_create_call = -1;
  int fail_copy_call = -1;
  cuda::CUresult copy_result = cuda::CUDA_SUCCESS;
  cuda::CUresult query_result = cuda::CUDA_SUCCESS;
  cuda::CUresult event_destroy_result = cuda::CUDA_SUCCESS;
  cuda::CUresult pinned_free_result = cuda::CUDA_SUCCESS;
  int not_ready_queries = 0;
  float elapsed_milliseconds = 1.25F;
  std::uint64_t now_nanoseconds = 0;
  bool stop = false;
  bool cancel = false;
};

FakeDriver *g_fake = nullptr;

cuda::CUresult FakeInit(unsigned int) { return cuda::CUDA_SUCCESS; }
cuda::CUresult FakeModuleLoad(cuda::CUmodule *module, const void *) {
  *module = reinterpret_cast<cuda::CUmodule>(0x10);
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakeModuleFunction(cuda::CUfunction *function, cuda::CUmodule,
                                  const char *) {
  *function = reinterpret_cast<cuda::CUfunction>(0x11);
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakeLaunch(cuda::CUfunction, unsigned int, unsigned int,
                          unsigned int, unsigned int, unsigned int,
                          unsigned int, unsigned int, cuda::CUstream, void **,
                          void **) {
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakeDeviceAlloc(cuda::CUdeviceptr *pointer, std::size_t *pitch,
                               std::size_t width, std::size_t, unsigned int) {
  *pointer = 0x1000;
  *pitch = width;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakeDeviceFree(cuda::CUdeviceptr) { return cuda::CUDA_SUCCESS; }
cuda::CUresult FakeCopy(const cuda::CUDA_MEMCPY2D *copy,
                        cuda::CUstream stream) {
  if (g_fake && copy &&
      g_fake->copy_calls < static_cast<int>(g_fake->copies.size())) {
    const auto index = static_cast<std::size_t>(g_fake->copy_calls);
    g_fake->copies[index] = *copy;
    g_fake->copy_streams[index] = stream;
    const int call = g_fake->copy_calls++;
    return call == g_fake->fail_copy_call ? cuda::CUDA_ERROR_OUT_OF_MEMORY
                                         : g_fake->copy_result;
  }
  return cuda::CUDA_ERROR_OUT_OF_MEMORY;
}
cuda::CUresult FakeMemset(cuda::CUdeviceptr, unsigned char, std::size_t,
                          cuda::CUstream) {
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakeStreamCreate(cuda::CUstream *stream, unsigned int) {
  *stream = reinterpret_cast<cuda::CUstream>(0x30);
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakeStreamDestroy(cuda::CUstream) { return cuda::CUDA_SUCCESS; }
cuda::CUresult FakeStreamSynchronize(cuda::CUstream) {
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakePinnedAlloc(void **pointer, std::size_t size, unsigned int) {
  if (!g_fake || size > g_fake->pinned.front().size() ||
      g_fake->pinned_allocations >= static_cast<int>(g_fake->pinned.size())) {
    return cuda::CUDA_ERROR_OUT_OF_MEMORY;
  }
  *pointer =
      g_fake->pinned[static_cast<std::size_t>(g_fake->pinned_allocations)]
          .data();
  ++g_fake->pinned_allocations;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakePinnedFree(void *pointer) {
  if (!g_fake ||
      g_fake->pinned_frees >= static_cast<int>(g_fake->freed_pointers.size())) {
    return cuda::CUDA_ERROR_UNKNOWN;
  }
  g_fake->freed_pointers[static_cast<std::size_t>(g_fake->pinned_frees)] =
      pointer;
  ++g_fake->pinned_frees;
  return g_fake->pinned_free_result;
}
cuda::CUresult FakeEventCreate(cuda::CUevent *event, unsigned int) {
  if (!g_fake)
    return cuda::CUDA_ERROR_UNKNOWN;
  const int call = g_fake->event_creations++;
  if (call == g_fake->fail_event_create_call) {
    *event = nullptr;
    return cuda::CUDA_ERROR_OUT_OF_MEMORY;
  }
  *event = reinterpret_cast<cuda::CUevent>(
      static_cast<std::uintptr_t>(0x100 + call));
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakeEventDestroy(cuda::CUevent) {
  if (g_fake) {
    ++g_fake->event_destructions;
    return g_fake->event_destroy_result;
  }
  return cuda::CUDA_ERROR_UNKNOWN;
}
cuda::CUresult FakeEventRecord(cuda::CUevent, cuda::CUstream) {
  if (g_fake)
    ++g_fake->event_records;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakeEventQuery(cuda::CUevent) {
  if (!g_fake)
    return cuda::CUDA_ERROR_UNKNOWN;
  ++g_fake->event_queries;
  if (g_fake->not_ready_queries > 0) {
    --g_fake->not_ready_queries;
    return cuda::CUDA_ERROR_NOT_READY;
  }
  return g_fake->query_result;
}
cuda::CUresult FakeEventElapsed(float *milliseconds, cuda::CUevent,
                                cuda::CUevent) {
  if (!g_fake || !milliseconds)
    return cuda::CUDA_ERROR_UNKNOWN;
  ++g_fake->elapsed_queries;
  *milliseconds = g_fake->elapsed_milliseconds;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakeDeviceCount(int *count) {
  *count = 1;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakeDeviceGet(cuda::CUdevice *device, int ordinal) {
  *device = ordinal;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakeUuid(cuda::CUuuid *uuid, cuda::CUdevice) {
  std::memset(uuid->bytes, 0, sizeof(uuid->bytes));
  uuid->bytes[0] = 0x45;
  uuid->bytes[15] = 0x7f;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakePci(char *pci, int length, cuda::CUdevice) {
  constexpr char value[] = "0000:01:00.0";
  if (length < static_cast<int>(sizeof(value)))
    return cuda::CUDA_ERROR_OUT_OF_MEMORY;
  std::memcpy(pci, value, sizeof(value));
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakePrimaryRetain(cuda::CUcontext *context, cuda::CUdevice) {
  *context = reinterpret_cast<cuda::CUcontext>(0x200);
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakePrimaryRelease(cuda::CUdevice) {
  if (g_fake)
    ++g_fake->context_releases;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakeContextSet(cuda::CUcontext context) {
  if (!g_fake)
    return cuda::CUDA_ERROR_INVALID_CONTEXT;
  g_fake->current = context;
  ++g_fake->context_sets;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakeContextGet(cuda::CUcontext *context) {
  if (!g_fake)
    return cuda::CUDA_ERROR_INVALID_CONTEXT;
  *context = g_fake->current;
  ++g_fake->context_queries;
  return cuda::CUDA_SUCCESS;
}
cuda::CUresult FakeContextDevice(cuda::CUdevice *device) {
  *device = 0;
  return cuda::CUDA_SUCCESS;
}

cuda::CudaDriverApi::Functions FakeFunctions(bool prepared_symbols = true) {
  cuda::CudaDriverApi::Functions functions;
  functions.cuInit = FakeInit;
  functions.cuModuleLoadData = FakeModuleLoad;
  functions.cuModuleGetFunction = FakeModuleFunction;
  functions.cuLaunchKernel = FakeLaunch;
  functions.cuMemAllocPitch = FakeDeviceAlloc;
  functions.cuMemFree = FakeDeviceFree;
  functions.cuMemcpy2DAsync = FakeCopy;
  functions.cuMemsetD8Async = FakeMemset;
  functions.cuStreamCreate = FakeStreamCreate;
  functions.cuStreamDestroy = FakeStreamDestroy;
  functions.cuStreamSynchronize = FakeStreamSynchronize;
  functions.cuDeviceGetCount = FakeDeviceCount;
  functions.cuDeviceGet = FakeDeviceGet;
  functions.cuDeviceGetUuid = FakeUuid;
  functions.cuDeviceGetPCIBusId = FakePci;
  functions.cuDevicePrimaryCtxRetain = FakePrimaryRetain;
  functions.cuDevicePrimaryCtxRelease = FakePrimaryRelease;
  functions.cuCtxSetCurrent = FakeContextSet;
  functions.cuCtxGetCurrent = FakeContextGet;
  functions.cuCtxGetDevice = FakeContextDevice;
  if (prepared_symbols) {
    functions.cuMemHostAlloc = FakePinnedAlloc;
    functions.cuMemFreeHost = FakePinnedFree;
    functions.cuEventCreate = FakeEventCreate;
    functions.cuEventDestroy = FakeEventDestroy;
    functions.cuEventRecord = FakeEventRecord;
    functions.cuEventQuery = FakeEventQuery;
    functions.cuEventElapsedTime = FakeEventElapsed;
  }
  return functions;
}

bool Initialize(cuda::CudaDriverApi *driver, FakeDriver *fake) {
  g_fake = fake;
  std::string error;
  return driver->InitializeForTesting(FakeFunctions(), &error) &&
         driver->EnsureContextForDevice(0, &error);
}

Support::Binding BindingFor(const cuda::CudaDriverApi &driver,
                            std::uint64_t configuration_generation = 10) {
  Support::Binding binding;
  if (driver.context_identity())
    binding.context = *driver.context_identity();
  binding.stream = reinterpret_cast<cuda::CUstream>(0x88);
  binding.stream_identity = 0x1234;
  binding.owner_identity = 0x5678;
  binding.configuration_generation = configuration_generation;
  return binding;
}

Support::PrepareOptions OptionsFor(const Support::Binding &binding,
                                   std::uint32_t stages = 2) {
  Support::PrepareOptions options;
  options.binding = binding;
  options.upload_row_bytes = 12;
  options.upload_rows = 4;
  options.readback_row_bytes = 12;
  options.readback_rows = 4;
  options.timed_stage_count = stages;
  options.budget.max_pinned_host_bytes = 96;
  options.budget.max_events = stages * 2U + 1U;
  return options;
}

std::uint64_t FakeNow(void *context) noexcept {
  return static_cast<FakeDriver *>(context)->now_nanoseconds;
}
void FakeWait(void *context, std::uint64_t nanoseconds) noexcept {
  static_cast<FakeDriver *>(context)->now_nanoseconds += nanoseconds;
}
void FakeWaitWithoutClockProgress(void *, std::uint64_t) noexcept {}
bool FakeStop(void *context) noexcept {
  return static_cast<FakeDriver *>(context)->stop;
}
bool FakeCancel(void *context) noexcept {
  return static_cast<FakeDriver *>(context)->cancel;
}

Support::WaitPolicy WaitPolicyFor(FakeDriver *fake) {
  Support::WaitPolicy policy;
  policy.timeout_nanoseconds = 100;
  policy.poll_interval_nanoseconds = 10;
  policy.maximum_poll_count = 32;
  policy.stop_requested = FakeStop;
  policy.cancel_requested = FakeCancel;
  policy.now_nanoseconds = FakeNow;
  policy.poll_wait = FakeWait;
  policy.callback_context = fake;
  return policy;
}

bool EnqueueCompleteFrame(Support *support, const Support::Binding &binding) {
  return support->BeginFrame(binding, 42).ok() &&
         support->EnqueueUpload(binding, 0x4000, 16).ok() &&
         support->RecordStageBegin(binding, 0).ok() &&
         support->RecordStageEnd(binding, 0).ok() &&
         support->RecordStageBegin(binding, 1).ok() &&
         support->RecordStageEnd(binding, 1).ok() &&
         support->EnqueueReadback(binding, 0x5000, 20).ok() &&
         support->RecordCompletion(binding).ok();
}

bool TestOptionalSymbolsDoNotBreakLegacyInitialization() {
  FakeDriver fake;
  g_fake = &fake;
  cuda::CudaDriverApi driver;
  std::string error;
  bool ok =
      Expect(driver.InitializeForTesting(FakeFunctions(false), &error),
             "legacy fake surface should initialize without prepared symbols");
  ok = Expect(driver.EnsureContextForDevice(0, &error),
              "legacy context setup should remain usable") &&
       ok;
  ok = Expect(!driver.HasPreparedTransferCompletionSupport(),
              "prepared support should report optional symbols absent") &&
       ok;
  Support support;
  auto version_mismatch = OptionsFor(BindingFor(driver));
  version_mismatch.api_version = Support::kApiVersion + 1;
  ok = Expect(support.Prepare(&driver, version_mismatch).status ==
                  Support::Status::unsupported,
              "unknown prepared API versions must fail closed") &&
       ok;
  ok =
      Expect(
          support.Prepare(&driver, OptionsFor(BindingFor(driver))).status ==
              Support::Status::unsupported,
          "prepared setup must fail closed when optional symbols are absent") &&
      ok;
  return ok;
}

bool TestPreparedHappyPathHasExactTransfersWaitAndTiming() {
  FakeDriver fake;
  cuda::CudaDriverApi driver;
  bool ok = Expect(Initialize(&driver, &fake), "fake CUDA should initialize");
  const auto binding = BindingFor(driver);
  Support support;
  ok = Expect(support.Prepare(&driver, OptionsFor(binding)).ok(),
              "prepared transfer resources should build") &&
       ok;
  ok = Expect(fake.pinned_allocations == 2 && fake.event_creations == 5,
              "setup should allocate two pinned buffers and five events") &&
       ok;
  ok = Expect(support.resource_budget().max_pinned_host_bytes == 96 &&
                  support.resource_budget().max_events == 5,
              "prepared resources should retain their exact setup budget") &&
       ok;
  const int allocations_after_setup = fake.pinned_allocations;
  const int events_after_setup = fake.event_creations;
  ok = Expect(EnqueueCompleteFrame(&support, binding),
              "one complete frame should enqueue") &&
       ok;
  fake.not_ready_queries = 2;
  const auto wait = support.WaitForCompletion(binding, WaitPolicyFor(&fake));
  ok = Expect(wait.ok(), "not-ready completion should become complete") && ok;
  ok = Expect(fake.pinned_allocations == allocations_after_setup &&
                  fake.event_creations == events_after_setup,
              "frame execution must not allocate or discover resources") &&
       ok;
  ok = Expect(fake.copy_calls == 2,
              "frame should issue exactly one upload and one readback") &&
       ok;
  ok =
      Expect(
          fake.copies[0].srcMemoryType == cuda::CU_MEMORYTYPE_HOST &&
              fake.copies[0].srcHost == support.upload_data() &&
              fake.copies[0].dstMemoryType == cuda::CU_MEMORYTYPE_DEVICE &&
              fake.copies[0].dstDevice == 0x4000 &&
              fake.copies[0].srcPitch == 12 && fake.copies[0].dstPitch == 16 &&
              fake.copies[0].WidthInBytes == 12 && fake.copies[0].Height == 4 &&
              fake.copy_streams[0] == binding.stream,
          "upload should use the exact pinned staging geometry and stream") &&
      ok;
  ok = Expect(
           fake.copies[1].srcMemoryType == cuda::CU_MEMORYTYPE_DEVICE &&
               fake.copies[1].srcDevice == 0x5000 &&
               fake.copies[1].dstMemoryType == cuda::CU_MEMORYTYPE_HOST &&
               fake.copies[1].dstHost == support.readback_data() &&
               fake.copies[1].srcPitch == 20 && fake.copies[1].dstPitch == 12 &&
               fake.copies[1].WidthInBytes == 12 && fake.copies[1].Height == 4,
           "readback should use the exact pinned staging geometry") &&
       ok;
  const auto &counters = support.counters();
  ok =
      Expect(
          counters.frame_uploads == 1 && counters.frame_final_readbacks == 1 &&
              counters.frame_waits == 1 && counters.event_records == 5 &&
              counters.event_queries == 3 &&
              counters.elapsed_time_queries == 2 &&
              counters.frame_allocations == 0 &&
              counters.frame_capability_probes == 0,
          "telemetry should prove exact transfers, wait, and no frame setup") &&
      ok;
  ok = Expect(support.stage_elapsed_nanoseconds(0) == 1250000 &&
                  support.stage_elapsed_nanoseconds(1) == 1250000,
              "fixed event pairs should report per-stage GPU time") &&
       ok;
  ok = Expect(support.Shutdown().ok() && support.Shutdown().ok(),
              "shutdown should be idempotent") &&
       ok;
  ok = Expect(support.BeginFrame(binding, 43).status ==
                  Support::Status::shutdown,
              "frame calls after shutdown should return typed shutdown") &&
       ok;
  ok = Expect(fake.pinned_frees == 2 && fake.event_destructions == 5,
              "shutdown should destroy every setup resource exactly once") &&
       ok;
  return ok;
}

bool TestBindingValidationFailsBeforeCudaCalls() {
  FakeDriver fake;
  cuda::CudaDriverApi driver;
  bool ok = Expect(Initialize(&driver, &fake), "fake CUDA should initialize");
  const auto binding = BindingFor(driver);
  Support support;
  ok = Expect(support.Prepare(&driver, OptionsFor(binding)).ok(),
              "binding test should prepare") &&
       ok;
  const int context_sets = fake.context_sets;

  auto invalid = binding;
  invalid.context = {};
  ok = Expect(support.BeginFrame(invalid, 1).status ==
                  Support::Status::invalid_context,
              "invalid context must be typed") &&
       ok;
  auto foreign = binding;
  foreign.context.retained_context_id += 1;
  ok = Expect(support.BeginFrame(foreign, 1).status ==
                  Support::Status::foreign_context,
              "foreign retained context must be typed") &&
       ok;
  auto stale = binding;
  stale.context.logical_generation += 1;
  ok = Expect(support.BeginFrame(stale, 1).status ==
                  Support::Status::stale_context,
              "stale logical context generation must be typed") &&
       ok;
  auto generation = binding;
  generation.configuration_generation += 1;
  ok = Expect(support.BeginFrame(generation, 1).status ==
                  Support::Status::stale_context,
              "stale configuration generation must be typed") &&
       ok;
  auto owner = binding;
  owner.owner_identity += 1;
  ok = Expect(support.BeginFrame(owner, 1).status ==
                  Support::Status::foreign_owner,
              "foreign resource owner must be typed") &&
       ok;
  auto stream = binding;
  stream.stream_identity += 1;
  ok = Expect(support.BeginFrame(stream, 1).status ==
                  Support::Status::wrong_stream,
              "wrong stable stream identity must be typed") &&
       ok;
  stream = binding;
  stream.stream = reinterpret_cast<cuda::CUstream>(0x99);
  ok = Expect(support.BeginFrame(stream, 1).status ==
                  Support::Status::wrong_stream,
              "wrong stream handle must be typed") &&
       ok;
  ok = Expect(fake.context_sets == context_sets && fake.copy_calls == 0 &&
                  fake.event_records == 0,
              "binding rejection must happen before CUDA frame calls") &&
       ok;
  ok = Expect(support.Shutdown().ok(), "binding test should shut down") && ok;
  return ok;
}

bool TestBudgetsRollbackAndReconfiguration() {
  FakeDriver fake;
  cuda::CudaDriverApi driver;
  bool ok = Expect(Initialize(&driver, &fake), "fake CUDA should initialize");
  const auto first_binding = BindingFor(driver, 20);
  auto exhausted = OptionsFor(first_binding);
  exhausted.budget.max_pinned_host_bytes = 95;
  Support support;
  ok = Expect(support.Prepare(&driver, exhausted).status ==
                  Support::Status::resource_exhausted,
              "pinned byte budget exhaustion must fail before allocation") &&
       ok;
  ok = Expect(fake.pinned_allocations == 0 && fake.event_creations == 0,
              "budget rejection must not allocate resources") &&
       ok;
  exhausted = OptionsFor(first_binding);
  exhausted.budget.max_events = 4;
  ok = Expect(support.Prepare(&driver, exhausted).status ==
                      Support::Status::resource_exhausted &&
                  fake.pinned_allocations == 0 && fake.event_creations == 0,
              "event budget exhaustion must fail before allocation") &&
       ok;
  ok = Expect(support.Prepare(&driver, OptionsFor(first_binding)).ok(),
              "first generation should prepare") &&
       ok;

  const auto second_binding = BindingFor(driver, 21);
  Support candidate;
  fake.fail_event_create_call = fake.event_creations + 1;
  const auto failed = candidate.Prepare(&driver, OptionsFor(second_binding));
  ok = Expect(failed.status == Support::Status::resource_exhausted,
              "candidate event exhaustion should be typed") &&
       ok;
  ok = Expect(
           support.BeginFrame(first_binding, 1).ok(),
           "failed candidate setup must retain the old prepared generation") &&
       ok;
  ok = Expect(support.Prepare(&driver, OptionsFor(second_binding)).status ==
                  Support::Status::already_prepared,
              "a prepared owner must require a separate candidate object") &&
       ok;
  ok = Expect(support.EnqueueUpload(first_binding, 0x4000, 16).ok() &&
                  support.RecordStageBegin(first_binding, 0).ok() &&
                  support.RecordStageEnd(first_binding, 0).ok() &&
                  support.RecordStageBegin(first_binding, 1).ok() &&
                  support.RecordStageEnd(first_binding, 1).ok() &&
                  support.EnqueueReadback(first_binding, 0x5000, 16).ok() &&
                  support.RecordCompletion(first_binding).ok() &&
                  support.WaitForCompletion(first_binding, WaitPolicyFor(&fake))
                      .ok(),
              "old generation should remain executable after rollback") &&
       ok;
  fake.fail_event_create_call = -1;
  ok = Expect(candidate.Prepare(&driver, OptionsFor(second_binding)).ok(),
              "fresh candidate should prepare after candidate rollback") &&
       ok;
  ok = Expect(support.BeginFrame(first_binding, 2).ok(),
              "old generation must remain valid until candidate commit") &&
       ok;
  ok = Expect(support.EnqueueUpload(first_binding, 0x4000, 16).ok() &&
                  support.RecordStageBegin(first_binding, 0).ok() &&
                  support.RecordStageEnd(first_binding, 0).ok() &&
                  support.RecordStageBegin(first_binding, 1).ok() &&
                  support.RecordStageEnd(first_binding, 1).ok() &&
                  support.EnqueueReadback(first_binding, 0x5000, 16).ok() &&
                  support.RecordCompletion(first_binding).ok() &&
                  support.WaitForCompletion(first_binding, WaitPolicyFor(&fake))
                      .ok() &&
                  support.Shutdown().ok(),
              "old generation should shut down only after candidate commit") &&
       ok;
  ok = Expect(candidate.BeginFrame(second_binding, 2).ok(),
              "new configuration binding should become active") &&
       ok;
  ok = Expect(
           candidate.EnqueueUpload(second_binding, 0x4000, 16).ok() &&
               candidate.RecordStageBegin(second_binding, 0).ok() &&
               candidate.RecordStageEnd(second_binding, 0).ok() &&
               candidate.RecordStageBegin(second_binding, 1).ok() &&
               candidate.RecordStageEnd(second_binding, 1).ok() &&
               candidate.EnqueueReadback(second_binding, 0x5000, 16).ok() &&
               candidate.RecordCompletion(second_binding).ok() &&
               candidate.WaitForCompletion(second_binding, WaitPolicyFor(&fake))
                   .ok() &&
               candidate.Shutdown().ok(),
           "candidate generation should execute and shut down") &&
       ok;
  return ok;
}

bool TestTimeoutStopCancelAndDeviceLossAreBounded() {
  bool ok = true;
  for (int mode = 0; mode < 4; ++mode) {
    FakeDriver fake;
    cuda::CudaDriverApi driver;
    ok =
        Expect(Initialize(&driver, &fake), "fake CUDA should initialize") && ok;
    const auto binding =
        BindingFor(driver, 30 + static_cast<std::uint64_t>(mode));
    Support support;
    ok = Expect(support.Prepare(&driver, OptionsFor(binding)).ok() &&
                    EnqueueCompleteFrame(&support, binding),
                "bounded-wait case should enqueue") &&
         ok;
    auto policy = WaitPolicyFor(&fake);
    Support::Status expected = Support::Status::timeout;
    if (mode == 0) {
      fake.not_ready_queries = 100;
    } else if (mode == 1) {
      fake.stop = true;
      expected = Support::Status::stop_requested;
    } else if (mode == 2) {
      fake.cancel = true;
      expected = Support::Status::cancelled;
    } else {
      fake.query_result = cuda::CUDA_ERROR_CONTEXT_IS_DESTROYED;
      expected = Support::Status::device_lost;
    }
    const auto result = support.WaitForCompletion(binding, policy);
    ok = Expect(result.status == expected,
                "bounded wait should return its exact typed terminal status") &&
         ok;
    if (mode < 3) {
      ok = Expect(support.frame_reuse_blocked(),
                  "interrupted completion must block staging reuse") &&
           ok;
      ok = Expect(support.BeginFrame(binding, 43).status ==
                      Support::Status::resource_exhausted,
                  "interrupted in-flight slot must reject reuse") &&
           ok;
      fake.stop = false;
      fake.cancel = false;
      fake.not_ready_queries = 0;
      fake.query_result = cuda::CUDA_SUCCESS;
    } else {
      ok =
          Expect(support.quarantined() && support.counters().device_losses == 1,
                 "device loss must quarantine exactly once") &&
          ok;
    }
    ok = Expect(support.Shutdown().ok() && support.Shutdown().ok(),
                "completed or lost work should permit idempotent shutdown") &&
         ok;
    ok = Expect(fake.pinned_frees == 2 && fake.event_destructions == 5,
                "bounded-wait shutdown should release setup resources") &&
         ok;
  }
  {
    FakeDriver fake;
    cuda::CudaDriverApi driver;
    ok =
        Expect(Initialize(&driver, &fake), "fake CUDA should initialize") && ok;
    const auto binding = BindingFor(driver, 39);
    Support support;
    ok = Expect(support.Prepare(&driver, OptionsFor(binding)).ok() &&
                    EnqueueCompleteFrame(&support, binding),
                "fixed poll-bound case should enqueue") &&
         ok;
    fake.not_ready_queries = 100;
    auto policy = WaitPolicyFor(&fake);
    policy.timeout_nanoseconds = 1000;
    policy.maximum_poll_count = 3;
    policy.poll_wait = FakeWaitWithoutClockProgress;
    ok = Expect(support.WaitForCompletion(binding, policy).status ==
                        Support::Status::timeout &&
                    fake.event_queries == 3,
                "maximum poll count must bound a non-progressing clock") &&
         ok;
    ok = Expect(support.Shutdown().status ==
                        Support::Status::resource_exhausted &&
                    fake.pinned_frees == 0 && fake.event_destructions == 0,
                "shutdown must retain resources while DMA is still pending") &&
         ok;
    fake.not_ready_queries = 0;
    ok = Expect(support.Shutdown().ok(),
                "fixed poll-bound case should clean up after completion") &&
         ok;
  }
  return ok;
}

bool TestCopyAndEventFailuresAreTyped() {
  bool ok = true;
  {
    FakeDriver fake;
    cuda::CudaDriverApi driver;
    ok =
        Expect(Initialize(&driver, &fake), "fake CUDA should initialize") && ok;
    const auto binding = BindingFor(driver, 40);
    Support support;
    ok = Expect(support.Prepare(&driver, OptionsFor(binding)).ok() &&
                    support.BeginFrame(binding, 1).ok(),
                "copy failure case should prepare") &&
         ok;
    fake.copy_result = 77;
    const auto result = support.EnqueueUpload(binding, 0x4000, 16);
    ok = Expect(result.status == Support::Status::copy_failed &&
                    result.cuda_result == 77,
                "ordinary copy failure should retain the CUDA status") &&
         ok;
    // No asynchronous work was accepted; the incomplete sequence correctly
    // refuses ordinary shutdown. Finish it after restoring the fake.
    fake.copy_result = cuda::CUDA_SUCCESS;
    ok =
        Expect(
            support.EnqueueUpload(binding, 0x4000, 16).ok() &&
                support.RecordStageBegin(binding, 0).ok() &&
                support.RecordStageEnd(binding, 0).ok() &&
                support.RecordStageBegin(binding, 1).ok() &&
                support.RecordStageEnd(binding, 1).ok() &&
                support.EnqueueReadback(binding, 0x5000, 16).ok() &&
                support.RecordCompletion(binding).ok() &&
                support.WaitForCompletion(binding, WaitPolicyFor(&fake)).ok() &&
                support.Shutdown().ok(),
            "copy failure should leave the frame safely recoverable") &&
        ok;
  }
  {
    FakeDriver fake;
    cuda::CudaDriverApi driver;
    ok =
        Expect(Initialize(&driver, &fake), "fake CUDA should initialize") && ok;
    const auto binding = BindingFor(driver, 41);
    Support support;
    ok = Expect(support.Prepare(&driver, OptionsFor(binding)).ok() &&
                    EnqueueCompleteFrame(&support, binding),
                "event failure case should enqueue") &&
         ok;
    fake.query_result = 78;
    const auto result =
        support.WaitForCompletion(binding, WaitPolicyFor(&fake));
    ok = Expect(result.status == Support::Status::event_failed &&
                    result.cuda_result == 78,
                "ordinary event query failure should be typed") &&
         ok;
    fake.query_result = cuda::CUDA_SUCCESS;
    ok = Expect(support.Shutdown().ok(),
                "event failure should permit cleanup after completion") &&
         ok;
  }
  return ok;
}

bool TestCleanupFailuresAreTypedAfterExactReleaseAttempts() {
  bool ok = true;
  for (int mode = 0; mode < 2; ++mode) {
    FakeDriver fake;
    cuda::CudaDriverApi driver;
    ok =
        Expect(Initialize(&driver, &fake), "fake CUDA should initialize") && ok;
    const auto binding =
        BindingFor(driver, 50 + static_cast<std::uint64_t>(mode));
    Support support;
    ok = Expect(
             support.Prepare(&driver, OptionsFor(binding)).ok() &&
                 EnqueueCompleteFrame(&support, binding) &&
                 support.WaitForCompletion(binding, WaitPolicyFor(&fake)).ok(),
             "cleanup failure case should complete its frame") &&
         ok;
    if (mode == 0)
      fake.event_destroy_result = 81;
    else
      fake.pinned_free_result = 82;
    const auto result = support.Shutdown();
    const auto expected = mode == 0 ? Support::Status::event_failed
                                    : Support::Status::resource_release_failed;
    ok = Expect(result.status == expected && fake.event_destructions == 5 &&
                    fake.pinned_frees == 2,
                "shutdown must type failure after attempting every release") &&
         ok;
    ok = Expect(
             support.Shutdown().ok(),
             "failed release reporting must still make shutdown idempotent") &&
         ok;
  }
  return ok;
}

bool TestPreparedBatchReadbacksAndPartialFailureQuiescence() {
  FakeDriver fake;
  cuda::CudaDriverApi driver;
  bool ok = Expect(Initialize(&driver, &fake), "fake CUDA should initialize");
  const auto binding = BindingFor(driver, 70);
  auto options = OptionsFor(binding, 1);
  options.readback_row_bytes = 0;
  options.readback_rows = 0;
  options.readback_count = 3;
  options.resident_work = true;
  options.readbacks[0] = {0x5000, 8, 0, 8, 1};
  options.readbacks[1] = {0x6000, 12, 8, 12, 1};
  options.readbacks[2] = {0x7000, 8, 20, 4, 2};
  options.budget.max_pinned_host_bytes = 48 + 28;
  Support support;
  const auto prepared = support.Prepare(&driver, options);
  const int copies_before_resident = fake.copy_calls;
  ok &= Expect(prepared.ok() &&
                   support.BeginFrame(binding, 71).ok() &&
                   support.EnqueueUpload(binding, 0x4000, 16).status ==
                       Support::Status::invalid_sequence &&
                   fake.copy_calls == copies_before_resident &&
                   support.SubmitResidentWork(binding).ok() &&
                   support.RecordStageBegin(binding, 0).ok() &&
                   support.RecordStageEnd(binding, 0).ok() &&
                   support.EnqueueReadbackAt(binding, 0, 0x5001, 8).status ==
                       Support::Status::invalid_sequence &&
                   support.EnqueueReadbackAt(binding, 0, 0x5000, 8).ok() &&
                   support.EnqueueReadbackAt(binding, 1, 0x6000, 12).ok() &&
                   support.EnqueueReadbackAt(binding, 2, 0x7000, 8).ok() &&
                   support.RecordCompletion(binding).ok() &&
                   support.WaitForCompletion(binding, WaitPolicyFor(&fake)).ok(),
               "three prepared readbacks should complete in order");
  ok &= Expect(support.counters().frame_readbacks == 3 &&
                   support.counters().frame_final_readbacks == 0 &&
                   fake.copies[0].dstHost == support.readback_data() &&
                   fake.copies[1].dstHost ==
                       static_cast<std::byte *>(support.readback_data()) + 8,
               "batch readbacks should report actual copies and frozen offsets");

  ok &= Expect(support.BeginFrame(binding, 72).ok() &&
                   support.SubmitResidentWork(binding).ok() &&
                   support.RecordStageBegin(binding, 0).ok() &&
                   support.RecordStageEnd(binding, 0).ok(),
               "partial failure frame should begin");
  fake.fail_copy_call = fake.copy_calls + 1;
  const auto p0 = support.EnqueueReadbackAt(binding, 0, 0x5000, 8);
  const auto p1 = support.EnqueueReadbackAt(binding, 1, 0x6000, 12);
  const auto record = support.RecordCompletion(binding);
  const auto wait = support.WaitForCompletion(binding, WaitPolicyFor(&fake));
  ok &= Expect(p0.ok() && p1.status == Support::Status::resource_exhausted &&
                   record.ok() && wait.ok(),
               "partial readback failure should record and wait for pending work");
  ok &= Expect(support.BeginFrame(binding, 73).ok() &&
                   support.SubmitResidentWork(binding).ok() &&
                   support.RecordStageBegin(binding, 0).ok() &&
                   support.RecordStageEnd(binding, 0).ok(),
               "completed partial frame should permit safe reuse");
  fake.fail_copy_call = fake.copy_calls;
  ok &= Expect(support.EnqueueReadbackAt(binding, 0, 0x5000, 8).status ==
                       Support::Status::resource_exhausted &&
                   support.RecordCompletion(binding).ok() &&
                   support.WaitForCompletion(binding, WaitPolicyFor(&fake)).ok(),
               "resident work should quiesce when its first readback fails");
  auto overlap = options;
  overlap.readbacks[1].host_offset_bytes = 4;
  Support malformed;
  ok &= Expect(malformed.Prepare(&driver, overlap).status ==
                   Support::Status::invalid_argument,
               "overlapping prepared host destinations should fail setup");
  auto overflow = options;
  overflow.readbacks[0].source =
      std::numeric_limits<cuda::CUdeviceptr>::max() - 3;
  overflow.readbacks[0].row_bytes = 8;
  overflow.readbacks[0].source_pitch = 8;
  Support single_overflow;
  ok &= Expect(single_overflow.Prepare(&driver, overflow).status ==
                   Support::Status::invalid_argument,
               "single-row source extent overflow should fail setup");
  overflow = options;
  overflow.readbacks[0].source =
      std::numeric_limits<cuda::CUdeviceptr>::max() - 15;
  overflow.readbacks[0].row_bytes = 8;
  overflow.readbacks[0].source_pitch = 12;
  overflow.readbacks[0].rows = 2;
  Support multi_overflow;
  ok &= Expect(multi_overflow.Prepare(&driver, overflow).status ==
                   Support::Status::invalid_argument,
               "multi-row source extent overflow should fail setup");
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok = TestOptionalSymbolsDoNotBreakLegacyInitialization() && ok;
  ok = TestPreparedHappyPathHasExactTransfersWaitAndTiming() && ok;
  ok = TestBindingValidationFailsBeforeCudaCalls() && ok;
  ok = TestBudgetsRollbackAndReconfiguration() && ok;
  ok = TestTimeoutStopCancelAndDeviceLossAreBounded() && ok;
  ok = TestCopyAndEventFailuresAreTyped() && ok;
  ok = TestCleanupFailuresAreTypedAfterExactReleaseAttempts() && ok;
  ok = TestPreparedBatchReadbacksAndPartialFailureQuiescence() && ok;
  if (!ok)
    return 1;
  std::cout << "CUDA PREPARED TRANSFER COMPLETION TESTS OK\n";
  return 0;
}
