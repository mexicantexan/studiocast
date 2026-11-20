#pragma once

#include "core/hal/identifiers.h"

#include <cstddef>
#include <cstdint>

namespace studiocast::video {

// The CPU primitive table is a synchronous host-memory utility contract. Its
// capabilities must never be interpreted as selectable product-effect
// providers. In particular, a resize, mirror, blur, or composite kernel does
// not establish semantic virtual-background or other effect availability.
inline constexpr std::uint32_t kCpuPrimitiveTableVersion = 1;

enum class CpuPrimitiveResult : std::uint8_t {
  success = 0,
  invalid_argument,
  insufficient_scratch,
  unprepared_plan,
};

[[nodiscard]] constexpr std::uint64_t
CpuPrimitiveCapabilityBit(hal::PrimitiveOperationId operation) noexcept {
  const auto value = static_cast<std::uint32_t>(operation);
  return value > 0 && value < 64 ? (std::uint64_t{1} << value) : 0;
}

struct CpuRgb24BilinearResizePlan {
  std::uint32_t table_version = 0;
  std::uint32_t struct_size = 0;
  int source_width = 0;
  int source_height = 0;
  int destination_width = 0;
  int destination_height = 0;
  const void *x_samples = nullptr;
  const void *y_samples = nullptr;
  const void *scratch_base = nullptr;
  std::size_t scratch_size = 0;
  std::uint64_t validation_tag = 0;
};

using CpuYuyvToRgb24Fn = CpuPrimitiveResult (*)(
    const std::uint8_t *src, int width, int height, std::size_t src_stride,
    std::uint8_t *dst, std::size_t dst_stride) noexcept;
using CpuRgb24ToYuyvScratchBytesFn = std::size_t (*)(int width,
                                                     int height) noexcept;
using CpuRgb24ToYuyvFn = CpuPrimitiveResult (*)(
    const std::uint8_t *src, int width, int height, std::size_t src_stride,
    std::uint8_t *dst, std::size_t dst_stride, std::uint8_t *scratch,
    std::size_t scratch_size) noexcept;
using CpuRgb24Bgr24Fn = CpuPrimitiveResult (*)(const std::uint8_t *src,
                                               std::uint8_t *dst, int width,
                                               int height,
                                               std::size_t src_stride,
                                               std::size_t dst_stride) noexcept;
using CpuMirrorRgb24Fn = CpuPrimitiveResult (*)(std::uint8_t *rgb, int width,
                                                int height,
                                                std::size_t stride) noexcept;
using CpuResizePlanScratchBytesFn =
    std::size_t (*)(int source_width, int source_height, int destination_width,
                    int destination_height) noexcept;
using CpuPrepareRgb24BilinearResizeFn = CpuPrimitiveResult (*)(
    int source_width, int source_height, int destination_width,
    int destination_height, void *scratch, std::size_t scratch_size,
    CpuRgb24BilinearResizePlan *plan) noexcept;
using CpuApplyRgb24BilinearResizeFn = CpuPrimitiveResult (*)(
    const CpuRgb24BilinearResizePlan *plan, const std::uint8_t *src_rgb,
    std::size_t src_stride, std::uint8_t *dst_rgb,
    std::size_t dst_stride) noexcept;

struct CpuPrimitiveTable {
  std::uint32_t table_version = 0;
  std::uint32_t struct_size = 0;
  hal::BackendId backend = hal::BackendId::invalid;
  // This must remain invalid: primitive availability is not semantic effect
  // availability.
  hal::SemanticEffectProviderId semantic_effect_provider =
      hal::SemanticEffectProviderId::invalid;
  std::uint64_t capability_bits = 0;

  const char *yuyv_to_rgb24_implementation = nullptr;
  const char *rgb24_to_yuyv_implementation = nullptr;
  const char *rgb24_bgr24_implementation = nullptr;
  const char *mirror_rgb24_implementation = nullptr;
  const char *resize_rgb24_bilinear_implementation = nullptr;

  CpuYuyvToRgb24Fn yuyv_to_rgb24 = nullptr;
  CpuRgb24ToYuyvScratchBytesFn rgb24_to_yuyv_scratch_bytes = nullptr;
  CpuRgb24ToYuyvFn rgb24_to_yuyv = nullptr;
  CpuRgb24Bgr24Fn rgb24_bgr24 = nullptr;
  CpuMirrorRgb24Fn mirror_rgb24_in_place = nullptr;
  CpuResizePlanScratchBytesFn resize_rgb24_bilinear_plan_scratch_bytes =
      nullptr;
  CpuPrepareRgb24BilinearResizeFn prepare_resize_rgb24_bilinear = nullptr;
  CpuApplyRgb24BilinearResizeFn apply_resize_rgb24_bilinear = nullptr;
};

[[nodiscard]] constexpr bool
CpuPrimitiveTableSupports(const CpuPrimitiveTable &table,
                          hal::PrimitiveOperationId operation) noexcept {
  const auto bit = CpuPrimitiveCapabilityBit(operation);
  return bit != 0 && (table.capability_bits & bit) == bit;
}

// Returns one immutable, process-lifetime table. Backend probing and function
// selection happen once before the returned table is first used.
[[nodiscard]] const CpuPrimitiveTable &GetCpuPrimitiveTable() noexcept;

} // namespace studiocast::video
