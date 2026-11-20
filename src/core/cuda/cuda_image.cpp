#include "core/cuda/cuda_image.h"

namespace studiocast::cuda {

CudaImage::CudaImage(CudaImage &&other) noexcept { MoveFrom(other); }

void CudaImage::MoveFrom(CudaImage &other) noexcept {
  ptr = other.ptr;
  pitch = other.pitch;
  w = other.w;
  h = other.h;
  format = other.format;
  owns_memory = other.owns_memory;
  other.ClearMetadata();
}

void CudaImage::ClearMetadata() noexcept {
  ptr = 0;
  pitch = 0;
  w = 0;
  h = 0;
  format = PixelFormatGpu::rgb_u8;
  owns_memory = false;
}

std::size_t BytesPerPixel(PixelFormatGpu fmt) {
  switch (fmt) {
  case PixelFormatGpu::rgb_u8:
    return 3;
  case PixelFormatGpu::bgr_u8:
    return 3;
  case PixelFormatGpu::gray_u8:
    return 1;
  case PixelFormatGpu::rgba_u8:
    return 4;
  case PixelFormatGpu::f32_1:
    return 4;
  case PixelFormatGpu::f32_3:
    return 12;
  }
  return 0;
}

std::size_t CudaImage::RowBytes() const {
  if (w <= 0)
    return 0;
  return static_cast<std::size_t>(w) * BytesPerPixel(format);
}

bool CudaImage::Valid() const {
  return ptr != 0 && w > 0 && h > 0 && pitch >= RowBytes();
}

bool CudaImage::Allocate(studiocast::cuda::CudaDriverApi *cuda, int width,
                         int height, PixelFormatGpu fmt,
                         std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!cuda || !cuda->IsInitialized()) {
    if (error_out)
      *error_out = "CudaImage::Allocate: CUDA driver API not initialized.";
    return false;
  }
  if (width <= 0 || height <= 0) {
    if (error_out)
      *error_out = "CudaImage::Allocate: invalid dimensions.";
    return false;
  }

  const std::size_t bpp = BytesPerPixel(fmt);
  if (bpp == 0) {
    if (error_out)
      *error_out = "CudaImage::Allocate: unknown pixel format.";
    return false;
  }

  // cuMemAllocPitch aligns pitches based on ElementSizeBytes and may require
  // WidthInBytes to be a multiple of that element size. Using 4 bytes keeps the
  // allocation broadly compatible across drivers while we still copy only the
  // exact active row width (RowBytes()).
  constexpr unsigned int kElementSizeBytes = 4;

  studiocast::cuda::CudaDriverApi::PitchAllocation alloc{};
  const std::size_t width_bytes = static_cast<std::size_t>(width) * bpp;
  const std::size_t alloc_width_bytes =
      ((width_bytes + (kElementSizeBytes - 1u)) / kElementSizeBytes) *
      kElementSizeBytes;
  const bool ok =
      cuda->AllocatePitch(alloc_width_bytes, static_cast<std::size_t>(height),
                          &alloc, error_out, kElementSizeBytes);
  if (!ok)
    return false;

  ptr = alloc.ptr;
  pitch = alloc.pitch;
  w = width;
  h = height;
  format = fmt;
  owns_memory = true;
  return true;
}

bool CudaImage::Free(studiocast::cuda::CudaDriverApi *cuda,
                     std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (ptr == 0) {
    pitch = 0;
    w = 0;
    h = 0;
    owns_memory = false;
    return true;
  }
  if (!owns_memory) {
    if (error_out)
      *error_out = "CudaImage::Free called for non-owning image.";
    return false;
  }
  if (!cuda || !cuda->IsInitialized()) {
    if (error_out)
      *error_out = "CudaImage::Free: CUDA driver API not initialized.";
    return false;
  }

  const bool ok = cuda->Free(ptr, error_out);
  if (!ok)
    return false;

  ptr = 0;
  pitch = 0;
  w = 0;
  h = 0;
  owns_memory = false;
  return true;
}

bool CudaImage::ReallocIfNeeded(studiocast::cuda::CudaDriverApi *cuda,
                                int width, int height, PixelFormatGpu fmt,
                                std::string *error_out) {
  if (error_out)
    error_out->clear();

  if (ptr != 0 && w == width && h == height && format == fmt) {
    // Pitch is allowed to be larger than the minimum.
    if (pitch >= static_cast<std::size_t>(width) * BytesPerPixel(fmt))
      return true;
  }

  // Free old (if any), then allocate.
  if (ptr != 0) {
    if (!Free(cuda, error_out))
      return false;
  }
  return Allocate(cuda, width, height, fmt, error_out);
}

bool CudaImage::UploadFromCpuRgb24(studiocast::cuda::CudaDriverApi *cuda,
                                   const std::uint8_t *src,
                                   std::size_t src_stride_bytes,
                                   studiocast::cuda::CUstream stream,
                                   std::string *error_out) const {
  if (error_out)
    error_out->clear();
  if (!cuda || !cuda->IsInitialized()) {
    if (error_out)
      *error_out = "UploadFromCpuRgb24: CUDA driver API not initialized.";
    return false;
  }
  if (!Valid()) {
    if (error_out)
      *error_out = "UploadFromCpuRgb24: invalid destination image.";
    return false;
  }
  if (format != PixelFormatGpu::rgb_u8) {
    if (error_out)
      *error_out = "UploadFromCpuRgb24: destination format is not rgb_u8.";
    return false;
  }
  if (!src) {
    if (error_out)
      *error_out = "UploadFromCpuRgb24: src is null.";
    return false;
  }

  const std::size_t row_bytes = static_cast<std::size_t>(w) * 3u;
  if (src_stride_bytes < row_bytes) {
    if (error_out)
      *error_out = "UploadFromCpuRgb24: src stride too small.";
    return false;
  }

  return cuda->MemcpyHtoD2DAsync(ptr, pitch, src, src_stride_bytes, row_bytes,
                                 static_cast<std::size_t>(h), stream,
                                 error_out);
}

bool CudaImage::DownloadToCpuRgb24(studiocast::cuda::CudaDriverApi *cuda,
                                   std::uint8_t *dst,
                                   std::size_t dst_stride_bytes,
                                   studiocast::cuda::CUstream stream,
                                   std::string *error_out) const {
  if (error_out)
    error_out->clear();
  if (!cuda || !cuda->IsInitialized()) {
    if (error_out)
      *error_out = "DownloadToCpuRgb24: CUDA driver API not initialized.";
    return false;
  }
  if (!Valid()) {
    if (error_out)
      *error_out = "DownloadToCpuRgb24: invalid source image.";
    return false;
  }
  if (format != PixelFormatGpu::rgb_u8) {
    if (error_out)
      *error_out = "DownloadToCpuRgb24: source format is not rgb_u8.";
    return false;
  }
  if (!dst) {
    if (error_out)
      *error_out = "DownloadToCpuRgb24: dst is null.";
    return false;
  }

  const std::size_t row_bytes = static_cast<std::size_t>(w) * 3u;
  if (dst_stride_bytes < row_bytes) {
    if (error_out)
      *error_out = "DownloadToCpuRgb24: dst stride too small.";
    return false;
  }

  return cuda->MemcpyDtoH2DAsync(dst, dst_stride_bytes, ptr, pitch, row_bytes,
                                 static_cast<std::size_t>(h), stream,
                                 error_out);
}

} // namespace studiocast::cuda
