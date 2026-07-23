#include "core/cuda/cuda_driver_api.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <utility>

namespace studiocast::cuda {

namespace {

std::atomic<std::uint64_t> g_cuda_context_generation{1};
std::atomic<std::uint64_t> g_cuda_context_instance{1};

std::uint64_t NextNonzero(std::atomic<std::uint64_t> *counter) noexcept {
  std::uint64_t value = counter->fetch_add(1, std::memory_order_relaxed);
  if (value == 0) {
    value = counter->fetch_add(1, std::memory_order_relaxed);
  }
  return value;
}

} // namespace

bool CudaPhysicalDeviceIdentity::Valid() const noexcept {
  return std::any_of(uuid.begin(), uuid.end(),
                     [](std::uint8_t byte) { return byte != 0; });
}

CudaDriverApi::CudaDriverApi() = default;

CudaDriverApi::~CudaDriverApi() {
  ReleaseRetainedContext();
  lib_.Close();
}

CudaDriverApi::CudaDriverApi(CudaDriverApi &&other) noexcept
    : initialized_(other.initialized_), lib_(std::move(other.lib_)),
      f_(other.f_), error_(std::move(other.error_)),
      retained_primary_ctx_(other.retained_primary_ctx_),
      device_ordinal_(other.device_ordinal_),
      primary_dev_(other.primary_dev_), primary_ctx_(other.primary_ctx_),
      primary_ctx_validated_(other.primary_ctx_validated_),
      context_identity_(other.context_identity_) {
  other.initialized_ = false;
  other.f_ = Functions{};
  other.error_.clear();
  other.retained_primary_ctx_ = false;
  other.device_ordinal_ = 0;
  other.primary_dev_ = 0;
  other.primary_ctx_ = nullptr;
  other.primary_ctx_validated_ = false;
  other.context_identity_ = {};
}

CudaDriverApi &CudaDriverApi::operator=(CudaDriverApi &&other) noexcept {
  if (this == &other)
    return *this;

  ReleaseRetainedContext();
  lib_.Close();

  initialized_ = other.initialized_;
  lib_ = std::move(other.lib_);
  f_ = other.f_;
  error_ = std::move(other.error_);
  retained_primary_ctx_ = other.retained_primary_ctx_;
  device_ordinal_ = other.device_ordinal_;
  primary_dev_ = other.primary_dev_;
  primary_ctx_ = other.primary_ctx_;
  primary_ctx_validated_ = other.primary_ctx_validated_;
  context_identity_ = other.context_identity_;

  other.initialized_ = false;
  other.f_ = Functions{};
  other.error_.clear();
  other.retained_primary_ctx_ = false;
  other.device_ordinal_ = 0;
  other.primary_dev_ = 0;
  other.primary_ctx_ = nullptr;
  other.primary_ctx_validated_ = false;
  other.context_identity_ = {};
  return *this;
}

bool CudaDriverApi::Initialize(std::string *error_out) {
  if (initialized_)
    return true;

  std::string err;
  if (!lib_.Open(util::DynLib::Scope::Local, &err)) {
    error_ = err;
    if (error_out)
      *error_out = err;
    return false;
  }

  if (!LoadSymbols(&err)) {
    error_ = err;
    if (error_out)
      *error_out = err;
    lib_.Close();
    return false;
  }

  const CUresult st = f_.cuInit ? f_.cuInit(0) : -1;
  if (st != CUDA_SUCCESS) {
    err = "cuInit failed: " + StatusToString(st);
    error_ = err;
    if (error_out)
      *error_out = err;
    return false;
  }

  initialized_ = true;
  return true;
}

bool CudaDriverApi::InitializeForTesting(const Functions &functions,
                                         std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (initialized_)
    return true;
  if (!ValidateRequiredFunctions(functions, error_out))
    return false;
  f_ = functions;
  const CUresult st = f_.cuInit(0);
  if (st != CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuInit failed: " + StatusToString(st);
    f_ = {};
    return false;
  }
  initialized_ = true;
  return true;
}

bool CudaDriverApi::ValidateRequiredFunctions(const Functions &functions,
                                              std::string *error_out) {
  struct Required {
    const char *name;
    bool present;
  };
  const Required required[] = {
      {"cuInit", functions.cuInit != nullptr},
      {"cuModuleLoadData", functions.cuModuleLoadData != nullptr},
      {"cuModuleGetFunction", functions.cuModuleGetFunction != nullptr},
      {"cuLaunchKernel", functions.cuLaunchKernel != nullptr},
      {"cuMemAllocPitch", functions.cuMemAllocPitch != nullptr},
      {"cuMemFree", functions.cuMemFree != nullptr},
      {"cuMemcpy2DAsync", functions.cuMemcpy2DAsync != nullptr},
      {"cuStreamCreate", functions.cuStreamCreate != nullptr},
      {"cuStreamDestroy", functions.cuStreamDestroy != nullptr},
      {"cuStreamSynchronize", functions.cuStreamSynchronize != nullptr},
      {"cuDeviceGetCount", functions.cuDeviceGetCount != nullptr},
      {"cuDeviceGet", functions.cuDeviceGet != nullptr},
      {"cuDeviceGetUuid", functions.cuDeviceGetUuid != nullptr},
      {"cuDevicePrimaryCtxRetain",
       functions.cuDevicePrimaryCtxRetain != nullptr},
      {"cuDevicePrimaryCtxRelease",
       functions.cuDevicePrimaryCtxRelease != nullptr},
      {"cuCtxSetCurrent", functions.cuCtxSetCurrent != nullptr},
      {"cuCtxGetCurrent", functions.cuCtxGetCurrent != nullptr},
      {"cuCtxGetDevice", functions.cuCtxGetDevice != nullptr},
  };
  for (const auto &entry : required) {
    if (!entry.present) {
      if (error_out)
        *error_out = std::string("CUDA required symbol unavailable: ") +
                     entry.name;
      return false;
    }
  }
  return true;
}

bool CudaDriverApi::LoadSymbols(std::string *error_out) {
  std::string err;

  auto load_req = [&](const char *name, auto *fn_out) {
    if (lib_.GetSymbol(name, fn_out, &err))
      return true;
    if (error_out)
      *error_out = err;
    return false;
  };
  auto load_req_any = [&](const char *preferred, const char *fallback,
                          auto *fn_out) {
    if (lib_.GetSymbol(preferred, fn_out, nullptr))
      return true;
    if (lib_.GetSymbol(fallback, fn_out, &err))
      return true;
    if (error_out)
      *error_out = err;
    return false;
  };

  // Prefer the *_v2 entry points where the CUDA headers map macros to v2.
  if (!load_req("cuInit", &f_.cuInit) ||
      !load_req("cuModuleLoadData", &f_.cuModuleLoadData) ||
      !load_req("cuModuleGetFunction", &f_.cuModuleGetFunction) ||
      !load_req("cuLaunchKernel", &f_.cuLaunchKernel) ||
      !load_req_any("cuMemAllocPitch_v2", "cuMemAllocPitch",
                    &f_.cuMemAllocPitch) ||
      !load_req_any("cuMemFree_v2", "cuMemFree", &f_.cuMemFree) ||
      !load_req_any("cuMemcpy2DAsync_v2", "cuMemcpy2DAsync",
                    &f_.cuMemcpy2DAsync) ||
      !load_req("cuStreamCreate", &f_.cuStreamCreate) ||
      !load_req_any("cuStreamDestroy_v2", "cuStreamDestroy",
                    &f_.cuStreamDestroy) ||
      !load_req("cuStreamSynchronize", &f_.cuStreamSynchronize) ||
      !load_req("cuDeviceGetCount", &f_.cuDeviceGetCount) ||
      !load_req("cuDeviceGet", &f_.cuDeviceGet) ||
      !load_req_any("cuDeviceGetUuid_v2", "cuDeviceGetUuid",
                    &f_.cuDeviceGetUuid) ||
      !load_req("cuDevicePrimaryCtxRetain", &f_.cuDevicePrimaryCtxRetain) ||
      !load_req_any("cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease",
                    &f_.cuDevicePrimaryCtxRelease) ||
      !load_req("cuCtxSetCurrent", &f_.cuCtxSetCurrent) ||
      !load_req("cuCtxGetCurrent", &f_.cuCtxGetCurrent) ||
      !load_req("cuCtxGetDevice", &f_.cuCtxGetDevice)) {
    return false;
  }

  // Optional corroborating stable identity. UUID remains mandatory.
  (void)lib_.GetSymbol("cuDeviceGetPCIBusId", &f_.cuDeviceGetPCIBusId,
                       nullptr);

  // Optional: cuGetErrorString.
  (void)lib_.GetSymbol("cuGetErrorString", &f_.cuGetErrorString, nullptr);

  // Optional: cuDriverGetVersion (diagnostics only).
  (void)lib_.GetSymbol("cuDriverGetVersion", &f_.cuDriverGetVersion, nullptr);

  // Optional: cuModuleLoadDataEx (used for PTX JIT diagnostics when available).
  (void)lib_.GetSymbol("cuModuleLoadDataEx", &f_.cuModuleLoadDataEx, nullptr);
  return true;
}

void CudaDriverApi::ReleaseRetainedContext() noexcept {
  if (retained_primary_ctx_ && f_.cuDevicePrimaryCtxRelease)
    (void)f_.cuDevicePrimaryCtxRelease(primary_dev_);
  retained_primary_ctx_ = false;
  primary_dev_ = 0;
  primary_ctx_ = nullptr;
  primary_ctx_validated_ = false;
  context_identity_ = {};
}

std::string CudaDriverApi::StatusToString(CUresult code) const {
  if (code == CUDA_SUCCESS)
    return "CUDA_SUCCESS";
  if (f_.cuGetErrorString) {
    const char *s = nullptr;
    if (f_.cuGetErrorString(code, &s) == CUDA_SUCCESS && s) {
      return std::string(s);
    }
  }
  return "CUDA error " + std::to_string(code);
}

bool CudaDriverApi::SetDeviceOrdinal(int device_ordinal,
                                     std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (device_ordinal < 0) {
    if (error_out)
      *error_out = "CUDA device ordinal must be non-negative.";
    return false;
  }
  if (retained_primary_ctx_ && primary_ctx_ &&
      device_ordinal != device_ordinal_) {
    if (error_out) {
      *error_out = "Cannot change CUDA device ordinal after context "
                   "initialization (current=" +
                   std::to_string(device_ordinal_) +
                   ", requested=" + std::to_string(device_ordinal) + ").";
    }
    return false;
  }
  device_ordinal_ = device_ordinal;
  return true;
}

bool CudaDriverApi::EnsureContextForDevice(int device_ordinal,
                                           std::string *error_out) {
  if (!SetDeviceOrdinal(device_ordinal, error_out))
    return false;
  return EnsureContext(error_out);
}

bool CudaDriverApi::EnsureContext(std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!initialized_) {
    if (error_out)
      *error_out = "CudaDriverApi not initialized.";
    return false;
  }
  if (!f_.cuCtxGetCurrent || !f_.cuDeviceGetCount || !f_.cuDeviceGet ||
      !f_.cuDeviceGetUuid || !f_.cuDevicePrimaryCtxRetain ||
      !f_.cuCtxSetCurrent || !f_.cuCtxGetDevice) {
    if (error_out)
      *error_out = "CUDA context functions not loaded.";
    return false;
  }

  // Always bind the configured primary context.
  //
  // Rationale: other in-process components (e.g. CUDA-capable libraries) can
  // temporarily change the current context on this thread. If we accept an
  // arbitrary "current" context, we can end up with context/stream/module
  // mismatches that manifest as cuLaunchKernel errors like "invalid resource
  // handle". Binding the configured primary context keeps Driver API kernels,
  // streams, allocations, and ORT CUDA provider work on the same selected GPU.
  bool retained_during_call = false;
  CUcontext previous_ctx = nullptr;
  if (!retained_primary_ctx_ || !primary_ctx_) {
    int count = 0;
    CUresult st = f_.cuDeviceGetCount(&count);
    if (st != CUDA_SUCCESS) {
      if (error_out)
        *error_out = "cuDeviceGetCount failed: " + StatusToString(st);
      return false;
    }
    if (count <= 0) {
      if (error_out)
        *error_out = "No CUDA devices found.";
      return false;
    }
    if (device_ordinal_ >= count) {
      if (error_out) {
        *error_out = "CUDA device ordinal " +
                     std::to_string(device_ordinal_) +
                     " is not available (device count: " +
                     std::to_string(count) + ").";
      }
      return false;
    }

    CUdevice dev = 0;
    st = f_.cuDeviceGet(&dev, device_ordinal_);
    if (st != CUDA_SUCCESS) {
      if (error_out)
        *error_out = "cuDeviceGet(" + std::to_string(device_ordinal_) +
                     ") failed: " + StatusToString(st);
      return false;
    }

    CUcontext ctx = nullptr;
    st = f_.cuDevicePrimaryCtxRetain(&ctx, dev);
    if (st != CUDA_SUCCESS || !ctx) {
      if (error_out)
        *error_out = "cuDevicePrimaryCtxRetain failed: " + StatusToString(st);
      return false;
    }

    retained_primary_ctx_ = true;
    retained_during_call = true;
    primary_dev_ = dev;
    primary_ctx_ = ctx;
    primary_ctx_validated_ = false;
  }

  (void)f_.cuCtxGetCurrent(&previous_ctx);
  CUresult st = f_.cuCtxSetCurrent(primary_ctx_);
  if (st != CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuCtxSetCurrent(primary device " +
                   std::to_string(device_ordinal_) +
                   ") failed: " + StatusToString(st);
    if (retained_during_call)
      ReleaseRetainedContext();
    return false;
  }

  if (primary_ctx_validated_)
    return true;

  CUcontext cur = nullptr;
  st = f_.cuCtxGetCurrent(&cur);
  if (st != CUDA_SUCCESS || !cur || cur != primary_ctx_) {
    if (error_out)
      *error_out = "Failed to validate current CUDA context after set: " +
                   StatusToString(st);
    if (retained_during_call) {
      (void)f_.cuCtxSetCurrent(previous_ctx);
      ReleaseRetainedContext();
    }
    return false;
  }

  CUdevice context_dev = 0;
  st = f_.cuCtxGetDevice(&context_dev);
  if (st != CUDA_SUCCESS || context_dev != primary_dev_) {
    if (error_out)
      *error_out = "CUDA retained context device identity mismatch.";
    if (retained_during_call) {
      (void)f_.cuCtxSetCurrent(previous_ctx);
      ReleaseRetainedContext();
    }
    return false;
  }

  CUuuid uuid{};
  st = f_.cuDeviceGetUuid(&uuid, context_dev);
  if (st != CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuDeviceGetUuid failed: " + StatusToString(st);
    if (retained_during_call) {
      (void)f_.cuCtxSetCurrent(previous_ctx);
      ReleaseRetainedContext();
    }
    return false;
  }
  CudaPhysicalDeviceIdentity physical;
  std::memcpy(physical.uuid.data(), uuid.bytes, physical.uuid.size());
  if (!physical.Valid()) {
    if (error_out)
      *error_out = "CUDA device UUID is unavailable or invalid.";
    if (retained_during_call) {
      (void)f_.cuCtxSetCurrent(previous_ctx);
      ReleaseRetainedContext();
    }
    return false;
  }
  if (f_.cuDeviceGetPCIBusId) {
    char pci[32]{};
    st = f_.cuDeviceGetPCIBusId(pci, static_cast<int>(sizeof(pci)),
                                context_dev);
    if (st != CUDA_SUCCESS) {
      if (error_out)
        *error_out = "cuDeviceGetPCIBusId failed: " + StatusToString(st);
      if (retained_during_call) {
        (void)f_.cuCtxSetCurrent(previous_ctx);
        ReleaseRetainedContext();
      }
      return false;
    }
    const auto nul = std::find(std::begin(pci), std::end(pci), '\0');
    const std::size_t length =
        static_cast<std::size_t>(std::distance(std::begin(pci), nul));
    if (length == 0 || nul == std::end(pci)) {
      if (error_out)
        *error_out = "CUDA PCI bus identity is invalid.";
      if (retained_during_call) {
        (void)f_.cuCtxSetCurrent(previous_ctx);
        ReleaseRetainedContext();
      }
      return false;
    }
    std::copy_n(pci, length + 1, physical.pci_bus_id.begin());
    physical.pci_bus_id_length = static_cast<std::uint8_t>(length);
  }

  // Best-effort functional validation.
  if (f_.cuStreamCreate && f_.cuStreamDestroy) {
    CUstream tmp = nullptr;
    const CUresult st_stream = f_.cuStreamCreate(&tmp, 0);
    if (st_stream != CUDA_SUCCESS || !tmp) {
      if (error_out)
        *error_out = "cuStreamCreate failed (context validation): " +
                     StatusToString(st_stream);
      if (retained_during_call) {
        (void)f_.cuCtxSetCurrent(previous_ctx);
        ReleaseRetainedContext();
      }
      return false;
    }
    (void)f_.cuStreamDestroy(tmp);
  }

  const std::uint64_t generation =
      NextNonzero(&g_cuda_context_generation);
  const std::uint64_t context_id =
      NextNonzero(&g_cuda_context_instance);
  context_identity_.physical_device = physical;
  context_identity_.logical_generation = generation;
  context_identity_.retained_context_id = context_id;
  primary_ctx_validated_ = true;
  return true;
}

bool CudaDriverApi::AllocatePitch(std::size_t width_bytes, std::size_t height,
                                  PitchAllocation *out, std::string *error_out,
                                  unsigned int element_size_bytes) {
  if (error_out)
    error_out->clear();
  if (!initialized_) {
    if (error_out)
      *error_out = "CudaDriverApi not initialized.";
    return false;
  }
  if (!EnsureContext(error_out))
    return false;
  if (!out) {
    if (error_out)
      *error_out = "AllocatePitch called with null out pointer.";
    return false;
  }
  if (!f_.cuMemAllocPitch) {
    if (error_out)
      *error_out = "cuMemAllocPitch symbol not loaded.";
    return false;
  }

  out->ptr = 0;
  out->pitch = 0;
  const CUresult st = f_.cuMemAllocPitch(&out->ptr, &out->pitch, width_bytes,
                                         height, element_size_bytes);
  if (st != CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuMemAllocPitch failed: " + StatusToString(st);
    return false;
  }
  return true;
}

bool CudaDriverApi::Free(CUdeviceptr ptr, std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!initialized_) {
    if (error_out)
      *error_out = "CudaDriverApi not initialized.";
    return false;
  }
  if (!EnsureContext(error_out))
    return false;
  if (!f_.cuMemFree) {
    if (error_out)
      *error_out = "cuMemFree symbol not loaded.";
    return false;
  }

  const CUresult st = f_.cuMemFree(ptr);
  if (st != CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuMemFree failed: " + StatusToString(st);
    return false;
  }
  return true;
}

bool CudaDriverApi::MemcpyHtoD2DAsync(CUdeviceptr dst, std::size_t dst_pitch,
                                      const void *src, std::size_t src_pitch,
                                      std::size_t width_bytes,
                                      std::size_t height, CUstream stream,
                                      std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!initialized_) {
    if (error_out)
      *error_out = "CudaDriverApi not initialized.";
    return false;
  }
  if (!EnsureContext(error_out))
    return false;
  if (!f_.cuMemcpy2DAsync) {
    if (error_out)
      *error_out = "cuMemcpy2DAsync symbol not loaded.";
    return false;
  }

  CUDA_MEMCPY2D cpy{};
  cpy.srcXInBytes = 0;
  cpy.srcY = 0;
  cpy.srcMemoryType = CU_MEMORYTYPE_HOST;
  cpy.srcHost = src;
  cpy.srcDevice = 0;
  cpy.srcArray = nullptr;
  cpy.srcPitch = src_pitch;

  cpy.dstXInBytes = 0;
  cpy.dstY = 0;
  cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  cpy.dstHost = nullptr;
  cpy.dstDevice = dst;
  cpy.dstArray = nullptr;
  cpy.dstPitch = dst_pitch;

  cpy.WidthInBytes = width_bytes;
  cpy.Height = height;

  const CUresult st = f_.cuMemcpy2DAsync(&cpy, stream);
  if (st != CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuMemcpy2DAsync (HtoD) failed: " + StatusToString(st);
    return false;
  }
  return true;
}

bool CudaDriverApi::MemcpyDtoH2DAsync(void *dst, std::size_t dst_pitch,
                                      CUdeviceptr src, std::size_t src_pitch,
                                      std::size_t width_bytes,
                                      std::size_t height, CUstream stream,
                                      std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!initialized_) {
    if (error_out)
      *error_out = "CudaDriverApi not initialized.";
    return false;
  }
  if (!EnsureContext(error_out))
    return false;
  if (!f_.cuMemcpy2DAsync) {
    if (error_out)
      *error_out = "cuMemcpy2DAsync symbol not loaded.";
    return false;
  }

  CUDA_MEMCPY2D cpy{};
  cpy.srcXInBytes = 0;
  cpy.srcY = 0;
  cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  cpy.srcHost = nullptr;
  cpy.srcDevice = src;
  cpy.srcArray = nullptr;
  cpy.srcPitch = src_pitch;

  cpy.dstXInBytes = 0;
  cpy.dstY = 0;
  cpy.dstMemoryType = CU_MEMORYTYPE_HOST;
  cpy.dstHost = dst;
  cpy.dstDevice = 0;
  cpy.dstArray = nullptr;
  cpy.dstPitch = dst_pitch;

  cpy.WidthInBytes = width_bytes;
  cpy.Height = height;

  const CUresult st = f_.cuMemcpy2DAsync(&cpy, stream);
  if (st != CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuMemcpy2DAsync (DtoH) failed: " + StatusToString(st);
    return false;
  }
  return true;
}

bool CudaDriverApi::CreateStream(CUstream *out, std::string *error_out,
                                 unsigned int flags) {
  if (error_out)
    error_out->clear();
  if (!initialized_) {
    if (error_out)
      *error_out = "CudaDriverApi not initialized.";
    return false;
  }
  if (!EnsureContext(error_out))
    return false;
  if (!out) {
    if (error_out)
      *error_out = "CreateStream called with null out pointer.";
    return false;
  }
  if (!f_.cuStreamCreate) {
    if (error_out)
      *error_out = "cuStreamCreate symbol not loaded.";
    return false;
  }

  *out = nullptr;
  const CUresult st = f_.cuStreamCreate(out, flags);
  if (st != CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuStreamCreate failed: " + StatusToString(st);
    return false;
  }
  return true;
}

bool CudaDriverApi::DestroyStream(CUstream stream, std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!initialized_) {
    if (error_out)
      *error_out = "CudaDriverApi not initialized.";
    return false;
  }
  if (!EnsureContext(error_out))
    return false;
  if (!f_.cuStreamDestroy) {
    if (error_out)
      *error_out = "cuStreamDestroy symbol not loaded.";
    return false;
  }

  const CUresult st = f_.cuStreamDestroy(stream);
  if (st != CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuStreamDestroy failed: " + StatusToString(st);
    return false;
  }
  return true;
}

bool CudaDriverApi::StreamSynchronize(CUstream stream, std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!initialized_) {
    if (error_out)
      *error_out = "CudaDriverApi not initialized.";
    return false;
  }
  if (!EnsureContext(error_out))
    return false;
  if (!f_.cuStreamSynchronize) {
    if (error_out)
      *error_out = "cuStreamSynchronize symbol not loaded.";
    return false;
  }

  const CUresult st = f_.cuStreamSynchronize(stream);
  if (st != CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuStreamSynchronize failed: " + StatusToString(st);
    return false;
  }
  return true;
}

} // namespace studiocast::cuda
