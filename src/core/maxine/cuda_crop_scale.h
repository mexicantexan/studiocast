#pragma once

#include <cstdint>
#include <string>

#include "core/cuda/cuda_driver_api.h"
#include "core/maxine/nvcv_types.h"

namespace studiocast::maxine {

namespace detail {

// BGR resize byte rounding matches the Open CUDA resize contract: clamp to byte
// range, then round non-negative interpolated values half-up.
inline std::uint8_t RoundClampBgrResizeBilinearU8(float v) {
  if (v <= 0.0f)
    return 0;
  if (v >= 255.0f)
    return 255;
  return static_cast<std::uint8_t>(static_cast<int>(v + 0.5f));
}

} // namespace detail

// Crop+scale for chunky BGR U8 GPU images using a tiny CUDA kernel.
//
// This is intentionally minimal for Milestone 1 correctness; quality can be
// improved later (e.g. bilinear, lanczos, etc.).
class CudaBgrCropScale {
public:
  bool Initialize(studiocast::cuda::CudaDriverApi *cuda, std::string *error_out);

  // Crop rectangle is expressed in source pixel coordinates.
  // Destination image must already be allocated and be BGR/U8/GPU.
  bool CropScale(const NvCVImage &src_bgr_gpu, NvCVImage *dst_bgr_gpu,
                 float crop_x, float crop_y, float crop_w, float crop_h,
                 CUstream stream, std::string *error_out);

private:
  bool EnsureKernelLoaded(std::string *error_out);

  studiocast::cuda::CudaDriverApi *cuda_ = nullptr; // non-owning
  studiocast::cuda::CUmodule module_ = nullptr;
  studiocast::cuda::CUfunction fn_ = nullptr;
  bool loaded_ = false;
};

// Bilinear resize for chunky BGR U8 GPU images using a tiny CUDA kernel.
//
// The kernel uses image pitch (bytes per row) from the NvCVImage plane.
class CudaBgrResizeBilinear {
public:
  bool Initialize(studiocast::cuda::CudaDriverApi *cuda, std::string *error_out);

  // Source/destination images must already be allocated and be BGR/U8/GPU.
  bool Resize(const NvCVImage &src_bgr_gpu, NvCVImage *dst_bgr_gpu,
              CUstream stream, std::string *error_out);

private:
  bool EnsureKernelLoaded(std::string *error_out);

  studiocast::cuda::CudaDriverApi *cuda_ = nullptr; // non-owning
  studiocast::cuda::CUmodule module_ = nullptr;
  studiocast::cuda::CUfunction fn_ = nullptr;
  bool loaded_ = false;
};

} // namespace studiocast::maxine
