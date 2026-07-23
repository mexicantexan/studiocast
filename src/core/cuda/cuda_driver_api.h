#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "core/util/dynlib.h"

// Minimal CUDA Driver API loader (runtime only).
//
// StudioCast must build on systems without CUDA installed, so we declare only
// the function-pointer surface we need and resolve symbols via dlopen/dlsym.

namespace studiocast::cuda {

// Forward-declare CUDA opaque handles (match CUDA driver ABI).
struct CUctx_st;
struct CUmod_st;
struct CUfunc_st;
struct CUstream_st;
struct CUarray_st;

using CUcontext = CUctx_st *;
using CUmodule = CUmod_st *;
using CUfunction = CUfunc_st *;
using CUstream = CUstream_st *;
using CUarray = CUarray_st *;
using CUdeviceptr = unsigned long long;
using CUdevice = int;

// Minimal CUDA UUID ABI surface. CUDA defines this as 16 opaque bytes.
struct CUuuid {
  char bytes[16];
};

// Minimal CUDA Driver API memcopy ABI surface.
//
// Matches the CUDA driver headers (cuda.h) layout.
// We only declare what we need and rely on runtime symbol resolution.
using CUmemorytype = unsigned int;
inline constexpr CUmemorytype CU_MEMORYTYPE_HOST = 0x01;
inline constexpr CUmemorytype CU_MEMORYTYPE_DEVICE = 0x02;
inline constexpr CUmemorytype CU_MEMORYTYPE_ARRAY = 0x03;
inline constexpr CUmemorytype CU_MEMORYTYPE_UNIFIED = 0x04;

struct CUDA_MEMCPY2D {
  std::size_t srcXInBytes;
  std::size_t srcY;
  CUmemorytype srcMemoryType;
  const void *srcHost;
  CUdeviceptr srcDevice;
  CUarray srcArray;
  std::size_t srcPitch;

  std::size_t dstXInBytes;
  std::size_t dstY;
  CUmemorytype dstMemoryType;
  void *dstHost;
  CUdeviceptr dstDevice;
  CUarray dstArray;
  std::size_t dstPitch;

  std::size_t WidthInBytes;
  std::size_t Height;
};

using CUresult = int;
inline constexpr CUresult CUDA_SUCCESS = 0;

// Minimal CUDA JIT option ABI surface (for cuModuleLoadDataEx diagnostics).
using CUjit_option = unsigned int;
inline constexpr CUjit_option CU_JIT_INFO_LOG_BUFFER = 3;
inline constexpr CUjit_option CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;
inline constexpr CUjit_option CU_JIT_ERROR_LOG_BUFFER = 5;
inline constexpr CUjit_option CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;
inline constexpr CUjit_option CU_JIT_LOG_VERBOSE = 12;

struct CudaPhysicalDeviceIdentity {
  std::array<std::uint8_t, 16> uuid{};
  std::array<char, 32> pci_bus_id{};
  std::uint8_t pci_bus_id_length = 0;

  bool Valid() const noexcept;
  bool operator==(const CudaPhysicalDeviceIdentity &) const noexcept = default;
};

// Process-local identity for one retained CUDA primary-context generation.
// No CUDA object address is exposed or used as identity.
struct CudaContextIdentity {
  CudaPhysicalDeviceIdentity physical_device;
  std::uint64_t retained_context_id = 0;
  std::uint64_t logical_generation = 0;

  bool Valid() const noexcept {
    return physical_device.Valid() && retained_context_id != 0 &&
           logical_generation != 0;
  }
  bool operator==(const CudaContextIdentity &) const noexcept = default;
};

class CudaDriverApi {
public:
  using cuInit_t = CUresult (*)(unsigned int flags);
  using cuModuleLoadData_t = CUresult (*)(CUmodule *module, const void *image);
  using cuModuleLoadDataEx_t = CUresult (*)(CUmodule *module, const void *image,
                                            unsigned int numOptions,
                                            CUjit_option *options,
                                            void **optionValues);
  using cuModuleGetFunction_t = CUresult (*)(CUfunction *hfunc, CUmodule hmod,
                                             const char *name);
  using cuLaunchKernel_t = CUresult (*)(
      CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
      unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
      unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
      void **kernelParams, void **extra);
  using cuMemAllocPitch_t = CUresult (*)(CUdeviceptr *dptr, std::size_t *pPitch,
                                         std::size_t WidthInBytes,
                                         std::size_t Height,
                                         unsigned int ElementSizeBytes);
  using cuMemFree_t = CUresult (*)(CUdeviceptr dptr);
  using cuMemcpy2DAsync_t = CUresult (*)(const CUDA_MEMCPY2D *pCopy,
                                         CUstream hStream);
  using cuMemsetD8Async_t = CUresult (*)(CUdeviceptr dstDevice,
                                         unsigned char value,
                                         std::size_t count,
                                         CUstream hStream);
  using cuStreamCreate_t = CUresult (*)(CUstream *phStream, unsigned int Flags);
  using cuStreamDestroy_t = CUresult (*)(CUstream hStream);
  using cuStreamSynchronize_t = CUresult (*)(CUstream hStream);
  using cuDeviceGetCount_t = CUresult (*)(int *count);
  using cuDeviceGet_t = CUresult (*)(CUdevice *device, int ordinal);
  using cuDeviceGetUuid_t = CUresult (*)(CUuuid *uuid, CUdevice dev);
  using cuDeviceGetPCIBusId_t = CUresult (*)(char *pciBusId, int len,
                                             CUdevice dev);
  using cuDevicePrimaryCtxRetain_t = CUresult (*)(CUcontext *pctx,
                                                  CUdevice dev);
  using cuDevicePrimaryCtxRelease_t = CUresult (*)(CUdevice dev);
  using cuCtxSetCurrent_t = CUresult (*)(CUcontext ctx);
  using cuCtxGetCurrent_t = CUresult (*)(CUcontext *pctx);
  using cuCtxGetDevice_t = CUresult (*)(CUdevice *device);
  using cuGetErrorString_t = CUresult (*)(CUresult error, const char **pStr);
  using cuDriverGetVersion_t = CUresult (*)(int *driverVersion);

  struct Functions {
    cuInit_t cuInit = nullptr;
    cuModuleLoadData_t cuModuleLoadData = nullptr;
    cuModuleLoadDataEx_t cuModuleLoadDataEx = nullptr;
    cuModuleGetFunction_t cuModuleGetFunction = nullptr;
    cuLaunchKernel_t cuLaunchKernel = nullptr;
    cuMemAllocPitch_t cuMemAllocPitch = nullptr;
    cuMemFree_t cuMemFree = nullptr;
    cuMemcpy2DAsync_t cuMemcpy2DAsync = nullptr;
    cuMemsetD8Async_t cuMemsetD8Async = nullptr;
    cuStreamCreate_t cuStreamCreate = nullptr;
    cuStreamDestroy_t cuStreamDestroy = nullptr;
    cuStreamSynchronize_t cuStreamSynchronize = nullptr;
    cuDeviceGetCount_t cuDeviceGetCount = nullptr;
    cuDeviceGet_t cuDeviceGet = nullptr;
    cuDeviceGetUuid_t cuDeviceGetUuid = nullptr;
    cuDeviceGetPCIBusId_t cuDeviceGetPCIBusId = nullptr;
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain = nullptr;
    cuDevicePrimaryCtxRelease_t cuDevicePrimaryCtxRelease = nullptr;
    cuCtxSetCurrent_t cuCtxSetCurrent = nullptr;
    cuCtxGetCurrent_t cuCtxGetCurrent = nullptr;
    cuCtxGetDevice_t cuCtxGetDevice = nullptr;
    cuGetErrorString_t cuGetErrorString = nullptr;
    cuDriverGetVersion_t cuDriverGetVersion = nullptr;
  };

  CudaDriverApi();
  ~CudaDriverApi();

  CudaDriverApi(const CudaDriverApi &) = delete;
  CudaDriverApi &operator=(const CudaDriverApi &) = delete;

  CudaDriverApi(CudaDriverApi &&) noexcept;
  CudaDriverApi &operator=(CudaDriverApi &&) noexcept;

  bool Initialize(std::string *error_out);

  // Hermetic test seam for the runtime-loaded ABI surface. Production callers
  // must use Initialize(), which resolves the same required functions from
  // libcuda.so.1.
  bool InitializeForTesting(const Functions &functions,
                            std::string *error_out);
  bool IsInitialized() const { return initialized_; }

  const Functions &f() const { return f_; }

  std::string StatusToString(CUresult code) const;
  const std::string &error() const { return error_; }
  int device_ordinal() const { return device_ordinal_; }
  const CudaContextIdentity *context_identity() const noexcept {
    return context_identity_.Valid() ? &context_identity_ : nullptr;
  }

  // Configure the CUDA device ordinal used when retaining a primary context.
  // This must be set before context-dependent work starts for this instance.
  bool SetDeviceOrdinal(int device_ordinal, std::string *error_out);

  struct PitchAllocation {
    CUdeviceptr ptr = 0;
    std::size_t pitch = 0;
  };

  bool AllocatePitch(std::size_t width_bytes, std::size_t height,
                     PitchAllocation *out, std::string *error_out,
                     unsigned int element_size_bytes = 1);

  bool Free(CUdeviceptr ptr, std::string *error_out);

  bool MemcpyHtoD2DAsync(CUdeviceptr dst, std::size_t dst_pitch,
                         const void *src, std::size_t src_pitch,
                         std::size_t width_bytes, std::size_t height,
                         CUstream stream, std::string *error_out);

  bool MemcpyDtoH2DAsync(void *dst, std::size_t dst_pitch, CUdeviceptr src,
                         std::size_t src_pitch, std::size_t width_bytes,
                         std::size_t height, CUstream stream,
                         std::string *error_out);

  bool MemsetD8Async(CUdeviceptr dst, std::uint8_t value,
                     std::size_t byte_count, CUstream stream,
                     std::string *error_out);

  bool CreateStream(CUstream *out, std::string *error_out,
                    unsigned int flags = 0);
  bool DestroyStream(CUstream stream, std::string *error_out);
  bool StreamSynchronize(CUstream stream, std::string *error_out);

  // Ensures there is a current CUDA context for subsequent Driver API calls.
  //
  // We retain and set the primary context for the configured device ordinal.
  bool EnsureContext(std::string *error_out);

  // Configure the device ordinal and ensure its primary context is current.
  // This is intended for startup/session setup where a higher-level runtime
  // such as ONNX Runtime is also configured with a CUDA device id.
  bool EnsureContextForDevice(int device_ordinal, std::string *error_out);

private:
  bool LoadSymbols(std::string *error_out);
  static bool ValidateRequiredFunctions(const Functions &functions,
                                        std::string *error_out);
  void ReleaseRetainedContext() noexcept;

  bool initialized_ = false;
  util::DynLib lib_{std::filesystem::path("libcuda.so.1")};
  Functions f_{};
  std::string error_;

  bool retained_primary_ctx_ = false;
  int device_ordinal_ = 0;
  CUdevice primary_dev_ = 0;
  CUcontext primary_ctx_ = nullptr;
  bool primary_ctx_validated_ = false;
  CudaContextIdentity context_identity_{};
};

} // namespace studiocast::cuda
