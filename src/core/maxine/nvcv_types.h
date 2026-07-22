#pragma once

// Minimal NVIDIA NvCVImage ABI surface.
//
// StudioCast is open-source and does not vendor proprietary NVIDIA headers.
// We declare only the types / constants we need and load the real
// implementations at runtime via dlopen(3) / dlsym(3).
//
// IMPORTANT: These declarations must match the NVIDIA SDK ABI.

#include <cstdint>
#include <limits>

#include "core/cuda/cuda_driver_api.h"

namespace studiocast::maxine {

// Preserve the Maxine SDK-facing name while sharing the neutral runtime-loaded
// CUDA ABI declaration.
using CUstream = studiocast::cuda::CUstream;

// In NVIDIA headers this is a signed integer type (often an enum).
using NvCV_Status = int32_t;
inline constexpr NvCV_Status NVCV_SUCCESS = 0;

// The format of pixels in an image.
// Values are part of the ABI.
enum NvCVImage_PixelFormat : uint32_t {
  NVCV_FORMAT_UNKNOWN = 0,
  NVCV_Y = 1,
  NVCV_A = 2,
  NVCV_YA = 3,
  NVCV_RGB = 4,
  NVCV_BGR = 5,
  NVCV_RGBA = 6,
  NVCV_BGRA = 7,
  NVCV_ARGB = 8,
  NVCV_ABGR = 9,
  NVCV_YUV420 = 10,
  NVCV_YUV422 = 11,
  NVCV_YUV444 = 12,
};

// The data type used to represent each component of an image.
// Values are part of the ABI.
enum NvCVImage_ComponentType : uint32_t {
  NVCV_TYPE_UNKNOWN = 0,
  NVCV_U8 = 1,
  NVCV_U16 = 2,
  NVCV_S16 = 3,
  NVCV_F16 = 4,
  NVCV_U32 = 5,
  NVCV_S32 = 6,
  NVCV_F32 = 7,
  NVCV_U64 = 8,
  NVCV_S64 = 9,
  NVCV_F64 = 10,
};

// Value for the planar field / layout argument.
// Note: the LSB can be used to distinguish between chunky and planar formats.
inline constexpr uint32_t NVCV_INTERLEAVED = 0;
inline constexpr uint32_t NVCV_CHUNKY = 0;
inline constexpr uint32_t NVCV_PLANAR = 1;

inline constexpr uint32_t NVCV_UYVY = 2;
inline constexpr uint32_t NVCV_VYUY = 4;
inline constexpr uint32_t NVCV_YUYV = 6;
inline constexpr uint32_t NVCV_YVYU = 8;
inline constexpr uint32_t NVCV_CYUV = 10;
inline constexpr uint32_t NVCV_CYVU = 12;

inline constexpr uint32_t NVCV_YUV = 3;
inline constexpr uint32_t NVCV_YVU = 5;
inline constexpr uint32_t NVCV_YCUV = 7;
inline constexpr uint32_t NVCV_YCVU = 9;

// FOURCC aliases for specific layouts.
inline constexpr uint32_t NVCV_I420 = NVCV_YUV;
inline constexpr uint32_t NVCV_IYUV = NVCV_YUV;
inline constexpr uint32_t NVCV_YV12 = NVCV_YVU;
inline constexpr uint32_t NVCV_NV12 = NVCV_YCUV;
inline constexpr uint32_t NVCV_NV21 = NVCV_YCVU;
inline constexpr uint32_t NVCV_YUY2 = NVCV_YUYV;

// Memory space constants (gpuMem field / memSpace argument).
inline constexpr uint32_t NVCV_CPU = 0;
inline constexpr uint32_t NVCV_GPU = 1;
inline constexpr uint32_t NVCV_CUDA = 1;
inline constexpr uint32_t NVCV_CPU_PINNED = 2;
inline constexpr uint32_t NVCV_CUDA_ARRAY = 3;

// NvCVImage POD struct used by the SDK.
// Note: field order matters for ABI compatibility.
struct NvCVImage {
  uint32_t width;
  uint32_t height;
  int32_t pitch;
  NvCVImage_PixelFormat pixelFormat;
  NvCVImage_ComponentType componentType;
  uint8_t pixelBytes;
  uint8_t componentBytes;
  uint8_t numComponents;
  uint8_t planar;     // NVCV_CHUNKY, NVCV_PLANAR, NVCV_UYVY, ...
  uint8_t gpuMem;     // NVCV_CPU, NVCV_CPU_PINNED, NVCV_CUDA, ...
  uint8_t colorspace; // OR of colorspace, range and chroma phase (YUV)
  uint8_t reserved[2];
  void *pixels;
  void *deletePtr;
  void (*deleteProc)(void *p);
  uint64_t bufferBytes;
};

enum class NvCVImageValidationStatus : uint8_t {
  ok,
  unexpected_gpu_mem,
  unexpected_pixel_format,
  unexpected_component_type,
  unexpected_layout,
  unexpected_component_bytes,
  unexpected_num_components,
  unexpected_pixel_bytes,
  zero_dimensions,
  null_pixels,
  row_bytes_exceed_pitch_range,
  invalid_pitch,
  pitch_too_small,
  buffer_too_small,
};

struct NvCVImageValidationSpec {
  NvCVImage_PixelFormat pixel_format = NVCV_BGR;
  NvCVImage_ComponentType component_type = NVCV_U8;
  uint8_t pixel_bytes = 3;
  uint8_t component_bytes = 1;
  uint8_t num_components = 3;
  uint8_t planar = static_cast<uint8_t>(NVCV_CHUNKY);
  uint8_t gpu_mem = static_cast<uint8_t>(NVCV_GPU);
  bool allow_zero_dimensions = false;
  bool require_buffer_bytes = true;
};

inline constexpr const char *
NvCVImageValidationStatusToString(NvCVImageValidationStatus status) {
  switch (status) {
  case NvCVImageValidationStatus::ok:
    return "ok";
  case NvCVImageValidationStatus::unexpected_gpu_mem:
    return "unexpected gpuMem";
  case NvCVImageValidationStatus::unexpected_pixel_format:
    return "unexpected pixel format";
  case NvCVImageValidationStatus::unexpected_component_type:
    return "unexpected component type";
  case NvCVImageValidationStatus::unexpected_layout:
    return "unexpected layout";
  case NvCVImageValidationStatus::unexpected_component_bytes:
    return "unexpected component byte count";
  case NvCVImageValidationStatus::unexpected_num_components:
    return "unexpected component count";
  case NvCVImageValidationStatus::unexpected_pixel_bytes:
    return "unexpected pixel byte count";
  case NvCVImageValidationStatus::zero_dimensions:
    return "zero dimensions";
  case NvCVImageValidationStatus::null_pixels:
    return "null pixels";
  case NvCVImageValidationStatus::row_bytes_exceed_pitch_range:
    return "row bytes exceed signed pitch range";
  case NvCVImageValidationStatus::invalid_pitch:
    return "invalid pitch";
  case NvCVImageValidationStatus::pitch_too_small:
    return "pitch is smaller than one row";
  case NvCVImageValidationStatus::buffer_too_small:
    return "bufferBytes is too small";
  }
  return "unknown NvCVImage validation status";
}

inline NvCVImageValidationStatus
ValidateNvCVImage(const NvCVImage &image,
                  const NvCVImageValidationSpec &spec) noexcept {
  if (image.gpuMem != spec.gpu_mem)
    return NvCVImageValidationStatus::unexpected_gpu_mem;
  if (image.pixelFormat != spec.pixel_format)
    return NvCVImageValidationStatus::unexpected_pixel_format;
  if (image.componentType != spec.component_type)
    return NvCVImageValidationStatus::unexpected_component_type;
  if (image.planar != spec.planar)
    return NvCVImageValidationStatus::unexpected_layout;
  if (image.componentBytes != spec.component_bytes)
    return NvCVImageValidationStatus::unexpected_component_bytes;
  if (image.numComponents != spec.num_components)
    return NvCVImageValidationStatus::unexpected_num_components;
  if (image.pixelBytes != spec.pixel_bytes)
    return NvCVImageValidationStatus::unexpected_pixel_bytes;

  if (image.width == 0 || image.height == 0) {
    return spec.allow_zero_dimensions
               ? NvCVImageValidationStatus::ok
               : NvCVImageValidationStatus::zero_dimensions;
  }

  if (!image.pixels)
    return NvCVImageValidationStatus::null_pixels;

  const uint64_t row_bytes =
      static_cast<uint64_t>(image.width) * spec.pixel_bytes;
  if (row_bytes >
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return NvCVImageValidationStatus::row_bytes_exceed_pitch_range;
  }

  if (image.pitch <= 0)
    return NvCVImageValidationStatus::invalid_pitch;

  const uint64_t pitch = static_cast<uint64_t>(image.pitch);
  if (pitch < row_bytes)
    return NvCVImageValidationStatus::pitch_too_small;

  const uint64_t required_bytes =
      pitch * (static_cast<uint64_t>(image.height) - 1u) + row_bytes;
  if ((spec.require_buffer_bytes || image.bufferBytes != 0) &&
      image.bufferBytes < required_bytes) {
    return NvCVImageValidationStatus::buffer_too_small;
  }

  return NvCVImageValidationStatus::ok;
}

inline NvCVImageValidationStatus
ValidateBgrU8CudaNvCVImage(const NvCVImage &image,
                           bool allow_zero_dimensions = false) noexcept {
  NvCVImageValidationSpec spec{};
  spec.allow_zero_dimensions = allow_zero_dimensions;
  return ValidateNvCVImage(image, spec);
}

} // namespace studiocast::maxine
