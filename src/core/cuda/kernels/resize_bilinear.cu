#include "core/cuda/kernels/resize_bilinear.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

// Reference CUDA C source for the embedded Driver API PTX module in
// resize_bilinear_ptx.cpp. The CMake freshness test compiles this file to PTX
// and checks the resize_bilinear_u8x3 entry ABI; StudioCast's normal build does
// not compile this .cu file into the application.
//
// PTX_GENERATE: nvcc -ptx -O3 -arch=compute_52 -I src src/core/cuda/kernels/resize_bilinear.cu -o resize_bilinear.ptx

namespace studiocast::cuda::kernels {
namespace {

__device__ __forceinline__ int ClampInt(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ int RoundHalfAwayFromZeroToInt(float v) {
  // Matches std::lround semantics for values we care about.
  return (v >= 0.0f) ? static_cast<int>(floorf(v + 0.5f))
                     : static_cast<int>(ceilf(v - 0.5f));
}

__global__ void CropResizeBilinearU8InterleavedKernel(
    const std::uint8_t *src, std::size_t src_pitch, int src_w, int src_h,
    std::uint8_t *dst, std::size_t dst_pitch, int dst_w, int dst_h,
    float crop_x, float crop_y, float crop_w, float crop_h, int channels) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= dst_w || y >= dst_h)
    return;

  const float scale_x = crop_w / static_cast<float>(dst_w);
  const float scale_y = crop_h / static_cast<float>(dst_h);

  const float src_x = crop_x + (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
  const float src_y = crop_y + (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;

  const int x0 = ClampInt(static_cast<int>(floorf(src_x)), 0, src_w - 1);
  const int y0 = ClampInt(static_cast<int>(floorf(src_y)), 0, src_h - 1);
  const int x1 = ClampInt(x0 + 1, 0, src_w - 1);
  const int y1 = ClampInt(y0 + 1, 0, src_h - 1);

  const float fx = src_x - static_cast<float>(x0);
  const float fy = src_y - static_cast<float>(y0);

  const auto *row0 = src + static_cast<std::size_t>(y0) * src_pitch;
  const auto *row1 = src + static_cast<std::size_t>(y1) * src_pitch;
  const auto *p00 =
      row0 + static_cast<std::size_t>(x0) * static_cast<std::size_t>(channels);
  const auto *p10 =
      row0 + static_cast<std::size_t>(x1) * static_cast<std::size_t>(channels);
  const auto *p01 =
      row1 + static_cast<std::size_t>(x0) * static_cast<std::size_t>(channels);
  const auto *p11 =
      row1 + static_cast<std::size_t>(x1) * static_cast<std::size_t>(channels);

  auto *dst_row = dst + static_cast<std::size_t>(y) * dst_pitch;
  auto *out_px = dst_row + static_cast<std::size_t>(x) *
                               static_cast<std::size_t>(channels);

  for (int c = 0; c < channels; ++c) {
    const float p00f = static_cast<float>(p00[c]);
    const float p10f = static_cast<float>(p10[c]);
    const float p01f = static_cast<float>(p01[c]);
    const float p11f = static_cast<float>(p11[c]);

    const float v0 = p00f + fx * (p10f - p00f);
    const float v1 = p01f + fx * (p11f - p01f);
    const float v = v0 + fy * (v1 - v0);

    const int iv = ClampInt(RoundHalfAwayFromZeroToInt(v), 0, 255);
    out_px[c] = static_cast<std::uint8_t>(iv);
  }
}

} // namespace

extern "C" __global__ void resize_bilinear_u8x3(
    const unsigned char *src, unsigned int src_pitch, unsigned int src_w,
    unsigned int src_h, unsigned char *dst, unsigned int dst_pitch,
    unsigned int dst_w, unsigned int dst_h, float crop_x, float crop_y,
    float crop_w, float crop_h) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= static_cast<int>(dst_w) || y >= static_cast<int>(dst_h))
    return;

  const float scale_x = crop_w / static_cast<float>(dst_w);
  const float scale_y = crop_h / static_cast<float>(dst_h);

  const float src_x = crop_x + (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
  const float src_y = crop_y + (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;

  const int src_w_max = static_cast<int>(src_w) - 1;
  const int src_h_max = static_cast<int>(src_h) - 1;
  const int x0 = ClampInt(static_cast<int>(src_x), 0, src_w_max);
  const int y0 = ClampInt(static_cast<int>(src_y), 0, src_h_max);
  const int x1 = ClampInt(x0 + 1, 0, src_w_max);
  const int y1 = ClampInt(y0 + 1, 0, src_h_max);

  const float fx = src_x - static_cast<float>(x0);
  const float fy = src_y - static_cast<float>(y0);

  const auto *row0 = src + static_cast<unsigned int>(y0) * src_pitch;
  const auto *row1 = src + static_cast<unsigned int>(y1) * src_pitch;
  const auto *p00 = row0 + static_cast<unsigned int>(x0) * 3u;
  const auto *p10 = row0 + static_cast<unsigned int>(x1) * 3u;
  const auto *p01 = row1 + static_cast<unsigned int>(x0) * 3u;
  const auto *p11 = row1 + static_cast<unsigned int>(x1) * 3u;

  auto *dst_row = dst + static_cast<unsigned int>(y) * dst_pitch;
  auto *out_px = dst_row + static_cast<unsigned int>(x) * 3u;

  for (int c = 0; c < 3; ++c) {
    const float p00f = static_cast<float>(p00[c]);
    const float p10f = static_cast<float>(p10[c]);
    const float p01f = static_cast<float>(p01[c]);
    const float p11f = static_cast<float>(p11[c]);
    const float v0 = p00f + fx * (p10f - p00f);
    const float v1 = p01f + fx * (p11f - p01f);
    const float v = v0 + fy * (v1 - v0);
    out_px[c] = static_cast<unsigned char>(
        ClampInt(static_cast<int>(v + 0.5f), 0, 255));
  }
}

bool IsResizeBilinearAvailable(std::string *error_out) {
  if (error_out)
    error_out->clear();
  int count = 0;
  const cudaError_t st = cudaGetDeviceCount(&count);
  if (st != cudaSuccess) {
    if (error_out)
      *error_out =
          std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(st);
    return false;
  }
  if (count <= 0) {
    if (error_out)
      *error_out = "No CUDA devices detected.";
    return false;
  }
  return true;
}

bool ResizeBilinear(const CudaImage &src, const CudaImage &dst,
                    studiocast::cuda::CUstream stream,
                    std::string *error_out) {
  return CropResizeBilinear(src, dst, 0.0f, 0.0f, static_cast<float>(src.w),
                            static_cast<float>(src.h), stream, error_out);
}

bool CropResizeBilinear(const CudaImage &src, const CudaImage &dst,
                        float crop_x, float crop_y, float crop_w, float crop_h,
                        studiocast::cuda::CUstream stream,
                        std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!src.Valid() || !dst.Valid()) {
    if (error_out)
      *error_out = "CropResizeBilinear: invalid src/dst image.";
    return false;
  }
  if (src.format != dst.format) {
    if (error_out)
      *error_out = "CropResizeBilinear: src/dst formats must match.";
    return false;
  }
  if (src.format != PixelFormatGpu::rgb_u8 &&
      src.format != PixelFormatGpu::bgr_u8) {
    if (error_out)
      *error_out =
          "CropResizeBilinear: unsupported format (expected rgb_u8 or bgr_u8).";
    return false;
  }
  if (!std::isfinite(crop_x) || !std::isfinite(crop_y) ||
      !std::isfinite(crop_w) || !std::isfinite(crop_h)) {
    if (error_out)
      *error_out =
          "CropResizeBilinear: crop rectangle contains non-finite values.";
    return false;
  }

  crop_w = std::clamp(crop_w, 1.0f, static_cast<float>(src.w));
  crop_h = std::clamp(crop_h, 1.0f, static_cast<float>(src.h));
  crop_x = std::clamp(crop_x, 0.0f, static_cast<float>(src.w) - crop_w);
  crop_y = std::clamp(crop_y, 0.0f, static_cast<float>(src.h) - crop_h);

  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const dim3 block(16, 16, 1);
  const dim3 grid((dst.w + block.x - 1) / block.x,
                  (dst.h + block.y - 1) / block.y, 1);
  const int channels = 3;

  const auto *src_ptr = reinterpret_cast<const std::uint8_t *>(
      static_cast<std::uintptr_t>(src.ptr));
  auto *dst_ptr =
      reinterpret_cast<std::uint8_t *>(static_cast<std::uintptr_t>(dst.ptr));

  CropResizeBilinearU8InterleavedKernel<<<grid, block, 0, s>>>(
      src_ptr, src.pitch, src.w, src.h, dst_ptr, dst.pitch, dst.w, dst.h,
      crop_x, crop_y, crop_w, crop_h, channels);
  const cudaError_t st = cudaGetLastError();
  if (st != cudaSuccess) {
    if (error_out)
      *error_out = std::string("CropResizeBilinear kernel launch failed: ") +
                   cudaGetErrorString(st);
    return false;
  }
  return true;
}

} // namespace studiocast::cuda::kernels
