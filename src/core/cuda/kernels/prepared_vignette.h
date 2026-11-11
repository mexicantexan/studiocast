#pragma once

#include "core/cuda/cuda_driver_api.h"
#include "core/cuda/cuda_image.h"

#include <string>

namespace studiocast::cuda::kernels {

struct PreparedVignetteKernel {
  CudaDriverApi *driver = nullptr;
  CUcontext owner_context = nullptr;
  CUstream owner_stream = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  bool live = false;

  PreparedVignetteKernel() = default;
  PreparedVignetteKernel(const PreparedVignetteKernel &) = delete;
  PreparedVignetteKernel &operator=(const PreparedVignetteKernel &) = delete;
  PreparedVignetteKernel(PreparedVignetteKernel &&other) noexcept;
  PreparedVignetteKernel &operator=(PreparedVignetteKernel &&) = delete;
};

bool PrepareVignetteKernel(CudaDriverApi &driver, CUcontext owner_context,
                           CUstream owner_stream, PreparedVignetteKernel *out,
                           std::string *error_out);

// Enqueues an RGB/BGR U8 vignette on the frozen owner stream. Source and
// destination may be the same exact image. No allocation, synchronization, or
// host transfer occurs here.
bool LaunchVignettePrepared(const PreparedVignetteKernel &kernel,
                            const CudaImage &source,
                            const CudaImage &destination, float intensity,
                            float center_x_px, float center_y_px,
                            std::string *error_out);

// The caller must first quiesce owner_stream and make owner_context current.
// A failed unload leaves the handle live so shutdown can retry safely.
bool DestroyVignetteKernelPrepared(PreparedVignetteKernel *kernel,
                                   std::string *error_out);

} // namespace studiocast::cuda::kernels
