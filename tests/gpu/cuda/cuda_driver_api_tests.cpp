#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_tensor.h"
#include "core/cuda/cuda_driver_api.h"
#include "core/cuda/kernels/preprocess_to_nchw.h"
#include "core/maxine/cuda_driver_api.h"

#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

static_assert(std::is_same_v<studiocast::maxine::CUstream_st,
                             studiocast::cuda::CUstream_st>);
static_assert(std::is_same_v<studiocast::maxine::CUstream,
                             studiocast::cuda::CUstream>);

bool Expect(bool cond, const std::string &msg) {
  if (!cond) {
    std::cerr << "[FAIL] " << msg << "\n";
  }
  return cond;
}

bool ExpectEq(const std::string &name, int got, int want) {
  if (got != want) {
    std::cerr << "[FAIL] " << name << ": got " << got << ", want " << want
              << "\n";
    return false;
  }
  return true;
}

bool ExpectEqSize(const std::string &name, std::size_t got, std::size_t want) {
  if (got != want) {
    std::cerr << "[FAIL] " << name << ": got " << got << ", want " << want
              << "\n";
    return false;
  }
  return true;
}

bool Contains(const std::string &s, const std::string &needle) {
  return s.find(needle) != std::string::npos;
}

struct FakeCudaDriver {
  int context_device = 3;
  int releases = 0;
  int context_queries = 0;
  int module_loads = 0;
  int function_resolutions = 0;
  int launches = 0;
  int memsets = 0;
  studiocast::cuda::CUcontext current = nullptr;
};

FakeCudaDriver *g_fake_cuda = nullptr;

studiocast::cuda::CUresult FakeInit(unsigned int) { return 0; }
studiocast::cuda::CUresult
FakeModuleLoad(studiocast::cuda::CUmodule *module, const void *) {
  if (g_fake_cuda)
    ++g_fake_cuda->module_loads;
  *module = reinterpret_cast<studiocast::cuda::CUmodule>(0x20);
  return 0;
}
studiocast::cuda::CUresult
FakeModuleGetFunction(studiocast::cuda::CUfunction *function,
                      studiocast::cuda::CUmodule, const char *) {
  if (g_fake_cuda)
    ++g_fake_cuda->function_resolutions;
  *function = reinterpret_cast<studiocast::cuda::CUfunction>(0x21);
  return 0;
}
studiocast::cuda::CUresult FakeLaunch(
    studiocast::cuda::CUfunction, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, unsigned int,
    studiocast::cuda::CUstream, void **, void **) {
  if (g_fake_cuda)
    ++g_fake_cuda->launches;
  return 0;
}
studiocast::cuda::CUresult
FakeAlloc(studiocast::cuda::CUdeviceptr *ptr, std::size_t *pitch,
          std::size_t width, std::size_t, unsigned int) {
  *ptr = 0x1000;
  *pitch = width;
  return 0;
}
studiocast::cuda::CUresult FakeFree(studiocast::cuda::CUdeviceptr) {
  return 0;
}
studiocast::cuda::CUresult
FakeCopy(const studiocast::cuda::CUDA_MEMCPY2D *,
         studiocast::cuda::CUstream) {
  return 0;
}
studiocast::cuda::CUresult
FakeMemset(studiocast::cuda::CUdeviceptr, unsigned char, std::size_t,
           studiocast::cuda::CUstream) {
  if (g_fake_cuda)
    ++g_fake_cuda->memsets;
  return 0;
}
studiocast::cuda::CUresult
FakeStreamCreate(studiocast::cuda::CUstream *stream, unsigned int) {
  *stream = reinterpret_cast<studiocast::cuda::CUstream>(0x30);
  return 0;
}
studiocast::cuda::CUresult
FakeStreamDestroy(studiocast::cuda::CUstream) {
  return 0;
}
studiocast::cuda::CUresult FakeStreamSync(studiocast::cuda::CUstream) {
  return 0;
}
studiocast::cuda::CUresult FakeDeviceCount(int *count) {
  *count = 8;
  return 0;
}
studiocast::cuda::CUresult FakeDeviceGet(studiocast::cuda::CUdevice *device,
                                         int ordinal) {
  *device = ordinal;
  return 0;
}
studiocast::cuda::CUresult
FakeDeviceUuid(studiocast::cuda::CUuuid *uuid,
               studiocast::cuda::CUdevice device) {
  std::memset(uuid->bytes, 0, sizeof(uuid->bytes));
  uuid->bytes[0] = 0x42;
  uuid->bytes[15] = static_cast<char>(device + 1);
  return 0;
}
studiocast::cuda::CUresult
FakeDevicePci(char *pci, int length, studiocast::cuda::CUdevice) {
  const char value[] = "0000:03:00.0";
  if (length < static_cast<int>(sizeof(value)))
    return 1;
  std::memcpy(pci, value, sizeof(value));
  return 0;
}
studiocast::cuda::CUresult
FakePrimaryRetain(studiocast::cuda::CUcontext *context,
                  studiocast::cuda::CUdevice device) {
  *context = reinterpret_cast<studiocast::cuda::CUcontext>(
      static_cast<std::uintptr_t>(0x100 + device));
  return 0;
}
studiocast::cuda::CUresult
FakePrimaryRelease(studiocast::cuda::CUdevice) {
  if (g_fake_cuda)
    ++g_fake_cuda->releases;
  return 0;
}
studiocast::cuda::CUresult
FakeCtxSet(studiocast::cuda::CUcontext context) {
  if (g_fake_cuda)
    g_fake_cuda->current = context;
  return 0;
}
studiocast::cuda::CUresult
FakeCtxGet(studiocast::cuda::CUcontext *context) {
  if (g_fake_cuda)
    ++g_fake_cuda->context_queries;
  *context = g_fake_cuda ? g_fake_cuda->current : nullptr;
  return 0;
}
studiocast::cuda::CUresult
FakeCtxDevice(studiocast::cuda::CUdevice *device) {
  *device = g_fake_cuda ? g_fake_cuda->context_device : -1;
  return 0;
}

studiocast::cuda::CudaDriverApi::Functions CompleteFakeFunctions() {
  studiocast::cuda::CudaDriverApi::Functions f;
  f.cuInit = FakeInit;
  f.cuModuleLoadData = FakeModuleLoad;
  f.cuModuleGetFunction = FakeModuleGetFunction;
  f.cuLaunchKernel = FakeLaunch;
  f.cuMemAllocPitch = FakeAlloc;
  f.cuMemFree = FakeFree;
  f.cuMemcpy2DAsync = FakeCopy;
  f.cuMemsetD8Async = FakeMemset;
  f.cuStreamCreate = FakeStreamCreate;
  f.cuStreamDestroy = FakeStreamDestroy;
  f.cuStreamSynchronize = FakeStreamSync;
  f.cuDeviceGetCount = FakeDeviceCount;
  f.cuDeviceGet = FakeDeviceGet;
  f.cuDeviceGetUuid = FakeDeviceUuid;
  f.cuDeviceGetPCIBusId = FakeDevicePci;
  f.cuDevicePrimaryCtxRetain = FakePrimaryRetain;
  f.cuDevicePrimaryCtxRelease = FakePrimaryRelease;
  f.cuCtxSetCurrent = FakeCtxSet;
  f.cuCtxGetCurrent = FakeCtxGet;
  f.cuCtxGetDevice = FakeCtxDevice;
  return f;
}

bool TestPreprocessKernelIsSetupOnlyAndGenerationBound() {
  FakeCudaDriver state;
  state.context_device = 2;
  g_fake_cuda = &state;
  std::string err;
  studiocast::cuda::CudaDriverApi cuda;
  bool ok = Expect(cuda.InitializeForTesting(CompleteFakeFunctions(), &err) &&
                       cuda.EnsureContextForDevice(2, &err),
                   "fake preprocess CUDA context should initialize");
  if (!ok)
    return false;
  const auto prepared_identity = *cuda.context_identity();

  studiocast::cuda::CudaImage source;
  source.ptr = 0x4000;
  source.pitch = 6;
  source.w = 2;
  source.h = 2;
  source.format = studiocast::cuda::PixelFormatGpu::rgb_u8;
  studiocast::cuda::CudaTensor tensor;
  tensor.ptr = 0x5000;
  tensor.pitch = 12 * sizeof(float);
  tensor.bytes = 12 * sizeof(float);
  tensor.n = 1;
  tensor.c = 3;
  tensor.h = 2;
  tensor.w = 2;
  studiocast::cuda::kernels::ModelPreprocessSpec spec;
  spec.dst_w = 2;
  spec.dst_h = 2;

  const int queries_before_missing = state.context_queries;
  ok = Expect(!studiocast::cuda::kernels::PreprocessToTensorPrepared(
                  &cuda, prepared_identity, source, tensor, spec, nullptr,
                  &err),
              "missing preprocess setup must fail closed") &&
       ok;
  ok = Expect(Contains(err, "missing") && state.module_loads == 0 &&
                  state.function_resolutions == 0 && state.launches == 0 &&
                  state.context_queries == queries_before_missing,
              "missing setup rejection must not probe, load, resolve, or "
              "launch") &&
       ok;

  ok = Expect(studiocast::cuda::kernels::PreparePreprocessKernel(&cuda, &err),
              "preprocess PTX should prepare during setup") &&
       ok;
  ok = Expect(cuda.MemsetD8Async(0x5000, 0, tensor.bytes, nullptr, &err) &&
                  state.memsets == 1,
              "setup should support bounded initialization of prepared CUDA "
              "buffers") &&
       ok;
  ok = Expect(state.module_loads == 1 && state.function_resolutions == 1,
              "setup should load and resolve preprocess exactly once") &&
       ok;
  ok = Expect(studiocast::cuda::kernels::PreparePreprocessKernel(&cuda, &err) &&
                  state.module_loads == 1 &&
                  state.function_resolutions == 1,
              "repeated setup for one context must reuse the prepared kernel") &&
       ok;
  const int queries_after_setup = state.context_queries;
  ok = Expect(studiocast::cuda::kernels::PreprocessToTensorPrepared(
                  &cuda, prepared_identity, source, tensor, spec, nullptr,
                  &err),
              "prepared preprocess frame should launch") &&
       ok;
  ok = Expect(state.launches == 1 && state.module_loads == 1 &&
                  state.function_resolutions == 1 &&
                  state.context_queries == queries_after_setup,
              "frame preprocess must not probe or rebuild setup state") &&
       ok;

  studiocast::cuda::CudaDriverApi empty;
  cuda = std::move(empty);
  state.context_device = 2;
  ok = Expect(cuda.InitializeForTesting(CompleteFakeFunctions(), &err) &&
                  cuda.EnsureContextForDevice(2, &err),
              "replacement context should initialize") &&
       ok;
  const int launches_before_stale = state.launches;
  ok = Expect(!studiocast::cuda::kernels::PreprocessToTensorPrepared(
                  &cuda, prepared_identity, source, tensor, spec, nullptr,
                  &err),
              "stale preprocess context generation must fail closed") &&
       ok;
  ok = Expect(Contains(err, "stale") &&
                  state.launches == launches_before_stale,
              "stale rejection must occur before kernel launch") &&
       ok;
  g_fake_cuda = nullptr;
  return ok;
}

bool TestStableContextIdentityAndFailClosedSymbols() {
  bool ok = true;
  std::string err;

  auto missing = CompleteFakeFunctions();
  missing.cuDeviceGetUuid = nullptr;
  studiocast::cuda::CudaDriverApi unavailable;
  ok = Expect(!unavailable.InitializeForTesting(missing, &err),
              "missing stable-device identity symbol must fail closed") &&
       ok;
  ok = Expect(Contains(err, "cuDeviceGetUuid"),
              "missing-symbol failure should identify UUID requirement") &&
       ok;

  FakeCudaDriver state;
  state.context_device = 3;
  g_fake_cuda = &state;
  studiocast::cuda::CudaDriverApi cuda;
  ok = Expect(cuda.InitializeForTesting(CompleteFakeFunctions(), &err),
              "complete fake CUDA ABI should initialize") &&
       ok;
  ok = Expect(cuda.EnsureContextForDevice(3, &err),
              "matching retained context should validate") &&
       ok;
  const auto *first = cuda.context_identity();
  ok = Expect(first && first->Valid(),
              "validated context must expose stable setup identity") &&
       ok;
  const auto first_copy = first ? *first : studiocast::cuda::CudaContextIdentity{};
  ok = Expect(first_copy.physical_device.uuid[0] == 0x42 &&
                  first_copy.physical_device.uuid[15] == 4,
              "identity should bind the selected device UUID") &&
       ok;
  ok = Expect(std::string(first_copy.physical_device.pci_bus_id.data()) ==
                  "0000:03:00.0",
              "identity should retain corroborating PCI bus evidence") &&
       ok;
  ok = Expect(cuda.EnsureContext(&err) &&
                  *cuda.context_identity() == first_copy,
              "rebinding the same retained context must preserve generation") &&
       ok;

  // Reinitialize the same wrapper after releasing its retained context through
  // move assignment. This exercises a distinct logical context lifetime.
  studiocast::cuda::CudaDriverApi empty;
  cuda = std::move(empty);
  state.context_device = 3;
  g_fake_cuda = &state;
  ok = Expect(cuda.InitializeForTesting(CompleteFakeFunctions(), &err) &&
                  cuda.EnsureContextForDevice(3, &err),
              "released wrapper should support a new retained context") &&
       ok;
  ok = Expect(cuda.context_identity() &&
                  cuda.context_identity()->logical_generation >
                      first_copy.logical_generation,
              "release and reinitialize must advance logical generation") &&
       ok;

  FakeCudaDriver second_state;
  second_state.context_device = 3;
  g_fake_cuda = &second_state;
  studiocast::cuda::CudaDriverApi second;
  ok = Expect(second.InitializeForTesting(CompleteFakeFunctions(), &err) &&
                  second.EnsureContextForDevice(3, &err),
              "second fake retained context should validate") &&
       ok;
  ok = Expect(second.context_identity() &&
                  second.context_identity()->logical_generation >
                      first_copy.logical_generation,
              "new context retention must advance logical generation") &&
       ok;

  FakeCudaDriver mismatch_state;
  mismatch_state.context_device = 4;
  g_fake_cuda = &mismatch_state;
  studiocast::cuda::CudaDriverApi mismatch;
  ok = Expect(mismatch.InitializeForTesting(CompleteFakeFunctions(), &err) &&
                  !mismatch.EnsureContextForDevice(3, &err),
              "context/device mismatch must fail closed") &&
       ok;
  ok = Expect(Contains(err, "identity mismatch"),
              "mismatch should produce exact identity evidence") &&
       ok;
  ok = Expect(mismatch.context_identity() == nullptr &&
                  mismatch_state.releases == 1,
              "failed setup must roll back retention and identity") &&
       ok;

  g_fake_cuda = nullptr;
  return ok;
}

bool TestDeviceOrdinalConfiguration() {
  studiocast::cuda::CudaDriverApi cuda;
  std::string err;

  bool ok = true;
  ok = ExpectEq("default CUDA device ordinal", cuda.device_ordinal(), 0) && ok;
  ok = Expect(cuda.SetDeviceOrdinal(2, &err),
              "SetDeviceOrdinal should accept non-negative ordinals") &&
       ok;
  ok = Expect(err.empty(), "SetDeviceOrdinal should clear error on success") &&
       ok;
  ok = ExpectEq("configured CUDA device ordinal", cuda.device_ordinal(), 2) &&
       ok;

  ok = Expect(!cuda.SetDeviceOrdinal(-1, &err),
              "SetDeviceOrdinal should reject negative ordinals") &&
       ok;
  ok = Expect(Contains(err, "non-negative"),
              "negative ordinal error should explain the constraint") &&
       ok;
  ok = ExpectEq("failed SetDeviceOrdinal preserves ordinal",
                cuda.device_ordinal(), 2) &&
       ok;

  studiocast::cuda::CudaDriverApi moved(std::move(cuda));
  ok = ExpectEq("move constructor preserves CUDA device ordinal",
                moved.device_ordinal(), 2) &&
       ok;

  studiocast::cuda::CudaDriverApi assigned;
  ok = Expect(assigned.SetDeviceOrdinal(1, &err),
              "pre-assignment SetDeviceOrdinal should succeed") &&
       ok;
  assigned = std::move(moved);
  ok = ExpectEq("move assignment preserves CUDA device ordinal",
                assigned.device_ordinal(), 2) &&
       ok;

  return ok;
}

bool TestCudaBufferWrappersAreMoveOnly() {
  using studiocast::cuda::CudaImage;
  using studiocast::cuda::CudaTensor;
  using studiocast::cuda::PixelFormatGpu;

  static_assert(!std::is_copy_constructible_v<CudaImage>);
  static_assert(!std::is_copy_assignable_v<CudaImage>);
  static_assert(std::is_nothrow_move_constructible_v<CudaImage>);
  static_assert(!std::is_move_assignable_v<CudaImage>);

  static_assert(!std::is_copy_constructible_v<CudaTensor>);
  static_assert(!std::is_copy_assignable_v<CudaTensor>);
  static_assert(std::is_nothrow_move_constructible_v<CudaTensor>);
  static_assert(!std::is_move_assignable_v<CudaTensor>);

  bool ok = true;

  CudaImage image;
  image.ptr = 0x1234;
  image.pitch = 64;
  image.w = 16;
  image.h = 8;
  image.format = PixelFormatGpu::rgba_u8;
  image.owns_memory = true;

  CudaImage moved_image(std::move(image));
  ok = Expect(moved_image.ptr == 0x1234 && moved_image.pitch == 64 &&
                  moved_image.w == 16 && moved_image.h == 8 &&
                  moved_image.format == PixelFormatGpu::rgba_u8 &&
                  moved_image.owns_memory,
              "CudaImage move constructor should transfer metadata") &&
       ok;
  ok = Expect(image.ptr == 0 && image.pitch == 0 && image.w == 0 &&
                  image.h == 0 && !image.owns_memory,
              "CudaImage move constructor should clear source") &&
       ok;

  CudaImage non_owning_image;
  non_owning_image.ptr = 0x5678;
  non_owning_image.pitch = 32;
  non_owning_image.w = 4;
  non_owning_image.h = 4;
  non_owning_image.owns_memory = false;
  non_owning_image.ClearMetadata();
  ok = Expect(non_owning_image.ptr == 0 && non_owning_image.pitch == 0 &&
                  non_owning_image.w == 0 && non_owning_image.h == 0 &&
                  !non_owning_image.owns_memory,
              "CudaImage ClearMetadata should reset non-owning views") &&
       ok;

  CudaTensor tensor;
  tensor.ptr = 0x9abc;
  tensor.pitch = 256;
  tensor.bytes = 128;
  tensor.n = 1;
  tensor.c = 3;
  tensor.h = 4;
  tensor.w = 5;
  tensor.owns_memory = true;

  CudaTensor moved_tensor(std::move(tensor));
  ok = Expect(moved_tensor.ptr == 0x9abc && moved_tensor.pitch == 256 &&
                  moved_tensor.bytes == 128 && moved_tensor.n == 1 &&
                  moved_tensor.c == 3 && moved_tensor.h == 4 &&
                  moved_tensor.w == 5 && moved_tensor.owns_memory,
              "CudaTensor move constructor should transfer metadata") &&
       ok;
  ok = Expect(tensor.ptr == 0 && tensor.pitch == 0 && tensor.bytes == 0 &&
                  tensor.n == 0 && tensor.c == 0 && tensor.h == 0 &&
                  tensor.w == 0 && !tensor.owns_memory,
              "CudaTensor move constructor should clear source") &&
       ok;

  return ok;
}

bool TestCudaTensorCheckedShapeMathNoGpu() {
  using studiocast::cuda::CheckedNchwF32Size;
  using studiocast::cuda::CudaTensor;
  using studiocast::cuda::CudaTensorSize;

  bool ok = true;
  std::string err;
  CudaTensorSize size;

  ok = Expect(CheckedNchwF32Size(2, 3, 4, 5, &size, &err),
              "valid NCHW F32 shape should compute") &&
       ok;
  ok = Expect(err.empty(), "valid shape should clear error") && ok;
  ok = ExpectEqSize("valid NCHW element count", size.elements, 120u) && ok;
  ok = ExpectEqSize("valid NCHW byte count", size.bytes,
                    120u * sizeof(float)) &&
       ok;
  const CudaTensorSize valid_size = size;

  ok = Expect(!CheckedNchwF32Size(1, 3, 0, 5, &size, &err),
              "zero dimension should be rejected") &&
       ok;
  ok = Expect(Contains(err, "invalid shape"),
              "zero dimension error should explain invalid shape") &&
       ok;
  ok = ExpectEqSize("failed shape clears element count", size.elements, 0u) &&
       ok;
  ok = ExpectEqSize("failed shape clears byte count", size.bytes, 0u) && ok;

  ok = Expect(!CheckedNchwF32Size(1, -3, 4, 5, &size, &err),
              "negative dimension should be rejected") &&
       ok;
  ok = Expect(Contains(err, "invalid shape"),
              "negative dimension error should explain invalid shape") &&
       ok;

  const int max_int = std::numeric_limits<int>::max();
  ok = Expect(!CheckedNchwF32Size(max_int, max_int, max_int, max_int, &size,
                                  &err),
              "overflowing NCHW shape should be rejected") &&
       ok;
  ok = Expect(Contains(err, "overflow"),
              "overflowing shape error should mention overflow") &&
       ok;

  CudaTensor overflow;
  overflow.ptr = 0x5678;
  overflow.pitch = std::numeric_limits<std::size_t>::max();
  overflow.bytes = std::numeric_limits<std::size_t>::max();
  overflow.n = max_int;
  overflow.c = max_int;
  overflow.h = max_int;
  overflow.w = max_int;
  overflow.owns_memory = true;
  ok = Expect(!overflow.Valid(),
              "CudaTensor with overflowing shape should be invalid") &&
       ok;
  ok = ExpectEqSize("overflowing CudaTensor element count",
                    overflow.ElementCount(), 0u) &&
       ok;

  CudaTensor empty;
  ok = Expect(!empty.Valid(), "empty CudaTensor should be invalid") && ok;
  ok = ExpectEqSize("empty CudaTensor element count", empty.ElementCount(),
                    0u) &&
       ok;

  CudaTensor tensor;
  tensor.ptr = 0x9abc;
  tensor.pitch = valid_size.bytes;
  tensor.bytes = valid_size.bytes;
  tensor.n = 2;
  tensor.c = 3;
  tensor.h = 4;
  tensor.w = 5;
  tensor.owns_memory = true;
  ok = Expect(tensor.Valid(),
              "manually populated tensor with checked size should be valid") &&
       ok;
  ok = ExpectEqSize("valid CudaTensor element count", tensor.ElementCount(),
                    120u) &&
       ok;

  CudaTensor moved(std::move(tensor));
  ok = Expect(tensor.ptr == 0 && tensor.pitch == 0 && tensor.bytes == 0 &&
                  tensor.n == 0 && tensor.c == 0 && tensor.h == 0 &&
                  tensor.w == 0 && !tensor.owns_memory,
              "moved-from CudaTensor should reset metadata") &&
       ok;
  ok = Expect(moved.Valid(), "moved CudaTensor should preserve valid shape") &&
       ok;

  CudaTensor mismatch;
  mismatch.ptr = 0x1234;
  mismatch.pitch = 1024;
  mismatch.bytes = 128;
  mismatch.n = 1;
  mismatch.c = 3;
  mismatch.h = 4;
  mismatch.w = 5;
  mismatch.owns_memory = true;
  ok = Expect(!mismatch.Valid(),
              "CudaTensor with mismatched byte count should be invalid") &&
       ok;

  float value = 0.0f;
  ok = Expect(!mismatch.UploadFromCpuF32(nullptr, &value, 1, 0, &err),
              "upload validation should reject bad tensor before CUDA use") &&
       ok;
  ok = Expect(Contains(err, "invalid tensor"),
              "bad upload tensor should report invalid tensor") &&
       ok;

  std::vector<float> out(1, 1.0f);
  ok = Expect(!mismatch.DownloadToCpuF32(nullptr, &out, 0, &err),
              "download validation should reject bad tensor before CUDA use") &&
       ok;
  ok = Expect(Contains(err, "invalid tensor"),
              "bad download tensor should report invalid tensor") &&
       ok;
  ok = Expect(out.empty(),
              "failed download validation should leave output vector empty") &&
       ok;

  ok = Expect(!empty.AllocateNchwF32(nullptr, 1, 0, 4, 5, &err),
              "invalid allocation shape should fail without CUDA") &&
       ok;
  ok = Expect(Contains(err, "invalid shape"),
              "invalid allocation shape should report shape error") &&
       ok;
  ok = Expect(!empty.AllocateNchwF32(nullptr, max_int, max_int, max_int,
                                     max_int, &err),
              "overflowing allocation shape should fail without CUDA") &&
       ok;
  ok = Expect(Contains(err, "overflow"),
              "overflowing allocation shape should report overflow") &&
       ok;

  const auto moved_ptr = moved.ptr;
  ok = Expect(!moved.ReallocIfNeededNchwF32(nullptr, 1, 0, 4, 5, &err),
              "invalid reallocation shape should fail without freeing") &&
       ok;
  ok = Expect(Contains(err, "invalid shape"),
              "invalid reallocation shape should report shape error") &&
       ok;
  ok = Expect(moved.ptr == moved_ptr,
              "invalid reallocation shape should preserve existing tensor") &&
       ok;

  return ok;
}

} // namespace

int main() {
  if (!TestPreprocessKernelIsSetupOnlyAndGenerationBound())
    return 1;
  if (!TestStableContextIdentityAndFailClosedSymbols())
    return 1;
  if (!TestDeviceOrdinalConfiguration())
    return 1;
  if (!TestCudaBufferWrappersAreMoveOnly())
    return 1;
  if (!TestCudaTensorCheckedShapeMathNoGpu())
    return 1;
  std::cout << "CUDA DRIVER API TESTS OK\n";
  return 0;
}
