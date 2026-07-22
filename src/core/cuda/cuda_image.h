#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "core/cuda/cuda_driver_api.h"

namespace studiocast::cuda {

enum class PixelFormatGpu {
  rgb_u8,
  bgr_u8,
  gray_u8,
  rgba_u8,
  f32_1,
  f32_3,
};

std::size_t BytesPerPixel(PixelFormatGpu fmt);

// Maxine-independent CUDA image buffer.
//
// Notes:
// - Uses the CUDA Driver API (loaded at runtime) via CudaDriverApi.
// - Does not require NvCVImage/NVCV.
// - Memory is pitched device allocation (cuMemAllocPitch).
struct CudaImage {
  CudaImage() = default;
  CudaImage(const CudaImage &) = delete;
  CudaImage &operator=(const CudaImage &) = delete;
  CudaImage(CudaImage &&other) noexcept;
  CudaImage &operator=(CudaImage &&other) = delete;

  studiocast::cuda::CUdeviceptr ptr = 0;
  std::size_t pitch = 0;
  int w = 0;
  int h = 0;
  PixelFormatGpu format = PixelFormatGpu::rgb_u8;
  bool owns_memory = false;

  std::size_t RowBytes() const;
  bool Valid() const;

  bool Allocate(studiocast::cuda::CudaDriverApi *cuda, int width, int height,
                PixelFormatGpu fmt, std::string *error_out);

  bool Free(studiocast::cuda::CudaDriverApi *cuda, std::string *error_out);

  bool ReallocIfNeeded(studiocast::cuda::CudaDriverApi *cuda, int width,
                       int height, PixelFormatGpu fmt, std::string *error_out);

  // Upload/download helpers for CPU-side RGB24 interop.
  //
  // These perform 2D copies and keep the operation on the provided stream.
  // The caller decides when to synchronize.
  bool UploadFromCpuRgb24(studiocast::cuda::CudaDriverApi *cuda,
                          const std::uint8_t *src, std::size_t src_stride_bytes,
                          studiocast::cuda::CUstream stream,
                          std::string *error_out) const;

  bool DownloadToCpuRgb24(studiocast::cuda::CudaDriverApi *cuda,
                          std::uint8_t *dst, std::size_t dst_stride_bytes,
                          studiocast::cuda::CUstream stream,
                          std::string *error_out) const;

  // Clears only this wrapper's metadata. It does not free device memory; call
  // Free(cuda, ...) first for owned allocations.
  void ClearMetadata() noexcept;

private:
  void MoveFrom(CudaImage &other) noexcept;
};

} // namespace studiocast::cuda
