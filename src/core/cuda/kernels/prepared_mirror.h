#pragma once

#include "core/cuda/cuda_driver_api.h"
#include "core/cuda/cuda_image.h"

#include <string>

namespace studiocast::cuda::kernels {

extern const char kPreparedMirrorPtx[];

struct PreparedMirrorKernel {
  CudaDriverApi *driver = nullptr;
  CUcontext owner_context = nullptr;
  CUstream owner_stream = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  bool live = false;

  PreparedMirrorKernel() = default;
  PreparedMirrorKernel(const PreparedMirrorKernel &) = delete;
  PreparedMirrorKernel &operator=(const PreparedMirrorKernel &) = delete;
  PreparedMirrorKernel(PreparedMirrorKernel &&other) noexcept;
  PreparedMirrorKernel &operator=(PreparedMirrorKernel &&) = delete;
};

bool PrepareMirrorKernel(CudaDriverApi &driver, CUcontext owner_context,
                         CUstream owner_stream, PreparedMirrorKernel *out,
                         std::string *error_out);

// Enqueues an exact horizontal reversal of a disjoint RGB/BGR U8 image on the
// frozen owner stream. Padding bytes are not touched. This performs no
// allocation, module loading, transfer, or synchronization.
bool LaunchMirrorPrepared(const PreparedMirrorKernel &kernel,
                          const CudaImage &source, const CudaImage &destination,
                          std::string *error_out);

// The caller must quiesce owner_stream and make owner_context current first.
// A failed unload leaves the handle live so shutdown can retry safely.
bool DestroyMirrorKernelPrepared(PreparedMirrorKernel *kernel,
                                 std::string *error_out);

} // namespace studiocast::cuda::kernels
