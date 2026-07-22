#pragma once

#include <string>

#include "core/cuda/cuda_driver_api.h"
#include "core/maxine/nvcv_types.h"

namespace studiocast::maxine {

// Vignette kernel for chunky BGR U8 GPU images.
//
// This is intentionally minimal and intended to be used as a GPU post-process
// step when the Maxine pipeline is active.
class CudaBgrVignette {
public:
  bool Initialize(studiocast::cuda::CudaDriverApi *cuda, std::string *error_out);

  // Applies a radial darkening centered at (center_x_px, center_y_px).
  // intensity: [0..1] where 0 is a no-op and 1 is strong darkening.
  bool ApplyInPlace(NvCVImage *bgr_gpu, float intensity, float center_x_px,
                    float center_y_px, CUstream stream, std::string *error_out);

private:
  bool EnsureKernelLoaded(std::string *error_out);

  studiocast::cuda::CudaDriverApi *cuda_ = nullptr; // non-owning
  studiocast::cuda::CUmodule module_ = nullptr;
  studiocast::cuda::CUfunction fn_ = nullptr;
  bool loaded_ = false;
};

} // namespace studiocast::maxine
