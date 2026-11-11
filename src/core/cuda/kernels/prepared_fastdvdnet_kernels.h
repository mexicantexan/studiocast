#pragma once

#include "core/cuda/cuda_driver_api.h"
#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_tensor.h"

#include <string>

namespace studiocast::cuda::kernels {

struct PreparedFastDvdnetKernels {
  CudaDriverApi *driver = nullptr;
  CUcontext context = nullptr;
  CUstream stream = nullptr;
  CUmodule module = nullptr;
  CUfunction preprocess = nullptr;
  CUfunction postprocess = nullptr;
  mutable CUresult last_result = CUDA_SUCCESS;
  bool live = false;
};

bool PrepareFastDvdnetKernels(CudaDriverApi &driver, CUcontext context,
                              CUstream stream, PreparedFastDvdnetKernels *out,
                              std::string *error);

// Produces current RGB planar history, [oldest,middle,current,current,current]
// noisy channels, and a constant sigma/255 map. Padding replicates the last
// valid source row/column. All work is enqueued on the frozen stream.
bool LaunchFastDvdnetPreprocess(const PreparedFastDvdnetKernels &kernels,
                                const CudaImage &source,
                                const CudaTensor &oldest,
                                const CudaTensor &middle,
                                const CudaTensor &current,
                                const CudaTensor &noisy,
                                const CudaTensor &noise_map, bool reset,
                                float sigma_over_255,
                                std::string *error) noexcept;

// Crops the padded denoised NCHW RGB tensor and converts [0,1] to resident
// RGB24. No synchronization or host transfer occurs.
bool LaunchFastDvdnetPostprocess(const PreparedFastDvdnetKernels &kernels,
                                 const CudaTensor &denoised,
                                 const CudaImage &destination,
                                 std::string *error) noexcept;

bool DestroyFastDvdnetKernels(PreparedFastDvdnetKernels *kernels,
                              std::string *error) noexcept;

} // namespace studiocast::cuda::kernels
