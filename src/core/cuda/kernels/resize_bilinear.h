#pragma once

#include <cstdint>
#include <string>

#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_driver_api.h"

namespace studiocast::cuda::kernels {

namespace detail {

// U8 resize byte rounding matches video::ResizeRgb24Bilinear's CPU contract:
// clamp to byte range, then round non-negative finite values with half-up
// behavior. This is equivalent to std::lround for the interpolated byte values
// after the lower clamp, without changing the live kernel's transfer/sync
// behavior.
inline std::uint8_t RoundClampResizeBilinearU8(float v) {
  if (v <= 0.0f)
    return 0;
  if (v >= 255.0f)
    return 255;
  return static_cast<std::uint8_t>(static_cast<int>(v + 0.5f));
}

} // namespace detail

// Returns true when the resize kernel can be JIT-loaded and launched on this
// system.
//
// This is a lightweight probe used to decide whether GPU resize is available
// without committing to any particular pipeline path.
bool IsResizeBilinearAvailable(std::string *error_out);

// Bilinear resize for interleaved RGB/BGR U8 pitched device images.
//
// - Source and destination formats must match and be rgb_u8 or bgr_u8.
// - Uses the same sampling convention as video::ResizeRgb24Bilinear:
//     src_x = (x + 0.5) * (src_w / dst_w) - 0.5
// - Rounds interpolated U8 channels with the same half-up byte contract as
//   video::ResizeRgb24Bilinear.
//
// The operation is enqueued on the provided stream and does not synchronize.
bool ResizeBilinear(const CudaImage &src, const CudaImage &dst,
                    studiocast::cuda::CUstream stream,
                    std::string *error_out);

// Bilinear crop+resize for interleaved RGB/BGR U8 pitched device images.
//
// Crop rectangle is expressed in source pixel coordinates. The operation is
// enqueued on the provided stream and does not synchronize.
bool CropResizeBilinear(const CudaImage &src, const CudaImage &dst,
                        float crop_x, float crop_y, float crop_w, float crop_h,
                        studiocast::cuda::CUstream stream,
                        std::string *error_out);

} // namespace studiocast::cuda::kernels
