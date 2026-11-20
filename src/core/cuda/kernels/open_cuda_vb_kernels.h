#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>

#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_driver_api.h"

namespace studiocast::cuda::kernels {

namespace detail {

// Keep the Open CUDA blur helper inside the canonical virtual-background
// strength/window range used by the CPU and Maxine paths. The PTX kernels do
// O(radius) work per pixel and are not a boundary for arbitrary caller input.
inline constexpr int kBoxBlurMaxRadius = 64;

inline int NormalizeBoxBlurRadius(int radius) {
  return radius < 0 ? 0 : radius;
}

inline bool CheckBoxBlurRadiusForKernel(int radius, const char *what,
                                        std::string *error_out) {
  if (radius > kBoxBlurMaxRadius) {
    if (error_out) {
      const char *label = what ? what : "Box blur";
      *error_out = std::string(label) +
                   ": radius exceeds maximum supported radius " +
                   std::to_string(kBoxBlurMaxRadius) + ".";
    }
    return false;
  }
  return true;
}

inline float ClampAlpha01ForComposite(float alpha) {
  if (!std::isfinite(alpha))
    return 0.0f;
  return alpha < 0.0f ? 0.0f : (alpha > 1.0f ? 1.0f : alpha);
}

inline std::array<std::uint8_t, 3>
SolidBackgroundMemoryChannels(PixelFormatGpu format, std::uint8_t bg_r,
                              std::uint8_t bg_g, std::uint8_t bg_b) {
  if (format == PixelFormatGpu::bgr_u8)
    return {bg_b, bg_g, bg_r};
  return {bg_r, bg_g, bg_b};
}

} // namespace detail

// Bilinear resize for f32_1 pitched device images.
//
// Sampling convention:
//   src_x = (x + 0.5) * (src_w / dst_w) - 0.5
//   src_y = (y + 0.5) * (src_h / dst_h) - 0.5
//
// Source coordinates are clamped to the valid source extent before bilinear
// weights are computed. Samples outside the source image therefore use
// edge/replicate border semantics with no extrapolation.
bool ResizeBilinearF32_1(const CudaImage &src, const CudaImage &dst,
                         studiocast::cuda::CUstream stream,
                         std::string *error_out);

// Separable box blur for interleaved u8x3 pitched device images.
//
// - src/tmp/dst must have identical dimensions.
// - src/tmp/dst formats must match and be rgb_u8 or bgr_u8.
// - radius is clamped to >= 0 and must not exceed detail::kBoxBlurMaxRadius.
bool BoxBlurSeparableU8x3(const CudaImage &src, const CudaImage &tmp,
                          const CudaImage &dst, int radius,
                          studiocast::cuda::CUstream stream,
                          std::string *error_out);

// Separable box blur for f32_1 pitched device images.
//
// - radius is clamped to >= 0 and must not exceed detail::kBoxBlurMaxRadius.
bool BoxBlurSeparableF32_1(const CudaImage &src, const CudaImage &tmp,
                           const CudaImage &dst, int radius,
                           studiocast::cuda::CUstream stream,
                           std::string *error_out);

// out = alpha * fg + (1 - alpha) * bg
//
// - fg/bg/out must be rgb_u8 or bgr_u8 and match each other.
// - alpha must be f32_1 with same dimensions; finite values are clamped to
// [0..1], and non-finite values are treated as 0.
bool CompositeAlphaU8x3(const CudaImage &fg, const CudaImage &bg,
                        const CudaImage &alpha, const CudaImage &out,
                        studiocast::cuda::CUstream stream,
                        std::string *error_out);

// out = alpha * fg + (1 - alpha) * solid_color
//
// - bg_r/bg_g/bg_b are semantic RGB components. For bgr_u8 fg/out images the
// solid color is written in BGR memory order.
bool CompositeAlphaSolidU8x3(const CudaImage &fg, const CudaImage &alpha,
                             std::uint8_t bg_r, std::uint8_t bg_g,
                             std::uint8_t bg_b, const CudaImage &out,
                             studiocast::cuda::CUstream stream,
                             std::string *error_out);

// out = key-light blend over src, masked by a frame-sized f32 alpha matte.
//
// - src/out must be rgb_u8 with identical dimensions.
// - alpha must be f32_1 with identical dimensions; finite values are clamped to
// [0..1], and non-finite values are treated as 0.
// - The operation is enqueued on the provided stream and does not synchronize.
bool ApplyKeyLightU8x3(const CudaImage &src, const CudaImage &alpha,
                       float target_r, float target_g, float target_b,
                       float intensity01, float direction, const CudaImage &out,
                       studiocast::cuda::CUstream stream,
                       std::string *error_out);

} // namespace studiocast::cuda::kernels
