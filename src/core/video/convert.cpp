#include "convert.h"

#include "convert_rgb_bgr_internal.h"
#include "convert_rgb_yuyv_internal.h"
#include "convert_yuyv_rgb_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

namespace studiocast::video {
namespace {

struct ResizeAxisSample {
  int i0 = 0;
  int i1 = 0;
  float factor = 0.0f;
};

constexpr std::uint64_t kResizePlanTag = 0x5354435052525031ull;

bool CheckedMultiply(std::size_t left, std::size_t right,
                     std::size_t *product) noexcept {
  if (!product ||
      (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)) {
    return false;
  }
  *product = left * right;
  return true;
}

bool RgbActiveStride(int width, std::size_t *stride) noexcept {
  return width > 0 &&
         CheckedMultiply(static_cast<std::size_t>(width), 3u, stride);
}

bool YuyvActiveStride(int width, std::size_t *stride) noexcept {
  if (width <= 0)
    return false;
  const std::size_t pairs =
      (static_cast<std::size_t>(width) + std::size_t{1}) / 2u;
  return CheckedMultiply(pairs, 4u, stride);
}

struct BufferSpan {
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
};

bool MakeBufferSpan(const void *data, int height, std::size_t stride,
                    std::size_t active_stride, BufferSpan *span) noexcept {
  if (!data || !span || height <= 0 || stride < active_stride)
    return false;

  std::size_t last_row_offset = 0;
  if (!CheckedMultiply(static_cast<std::size_t>(height - 1), stride,
                       &last_row_offset) ||
      last_row_offset >
          std::numeric_limits<std::size_t>::max() - active_stride) {
    return false;
  }
  const std::size_t byte_count = last_row_offset + active_stride;
  const auto begin = reinterpret_cast<std::uintptr_t>(data);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - byte_count)
    return false;
  *span = {begin, begin + byte_count};
  return true;
}

bool MakeContiguousSpan(const void *data, std::size_t byte_count,
                        BufferSpan *span) noexcept {
  if (!data || !span || byte_count == 0)
    return false;
  const auto begin = reinterpret_cast<std::uintptr_t>(data);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - byte_count)
    return false;
  *span = {begin, begin + byte_count};
  return true;
}

bool SpansOverlap(const BufferSpan &left, const BufferSpan &right) noexcept {
  return left.begin < right.end && right.begin < left.end;
}

std::uint64_t ResizePlanTag(const CpuRgb24BilinearResizePlan &plan) noexcept {
  auto tag = kResizePlanTag;
  tag ^= static_cast<std::uint64_t>(
      reinterpret_cast<std::uintptr_t>(plan.scratch_base));
  tag ^=
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(plan.source_width))
      << 1u;
  tag ^=
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(plan.source_height))
      << 17u;
  tag ^= static_cast<std::uint64_t>(
             static_cast<std::uint32_t>(plan.destination_width))
         << 33u;
  tag ^= static_cast<std::uint64_t>(
             static_cast<std::uint32_t>(plan.destination_height))
         << 49u;
  tag ^= static_cast<std::uint64_t>(plan.scratch_size);
  return tag;
}

bool AddressRangeContains(const void *range_base, std::size_t range_size,
                          const void *object,
                          std::size_t object_size) noexcept {
  if (!range_base || !object)
    return false;
  const auto begin = reinterpret_cast<std::uintptr_t>(range_base);
  const auto address = reinterpret_cast<std::uintptr_t>(object);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - range_size ||
      address < begin || address > begin + range_size) {
    return false;
  }
  return object_size <= begin + range_size - address;
}

bool ValidResizePlan(const CpuRgb24BilinearResizePlan &plan) noexcept {
  if (plan.table_version != kCpuPrimitiveTableVersion ||
      plan.struct_size != sizeof(CpuRgb24BilinearResizePlan) ||
      plan.source_width <= 0 || plan.source_height <= 0 ||
      plan.destination_width <= 0 || plan.destination_height <= 0 ||
      !plan.scratch_base || !plan.x_samples || !plan.y_samples ||
      plan.validation_tag != ResizePlanTag(plan)) {
    return false;
  }

  std::size_t x_bytes = 0;
  std::size_t y_bytes = 0;
  if (!CheckedMultiply(static_cast<std::size_t>(plan.destination_width),
                       sizeof(ResizeAxisSample), &x_bytes) ||
      !CheckedMultiply(static_cast<std::size_t>(plan.destination_height),
                       sizeof(ResizeAxisSample), &y_bytes) ||
      x_bytes > std::numeric_limits<std::size_t>::max() - y_bytes) {
    return false;
  }

  void *expected_x = const_cast<void *>(plan.scratch_base);
  std::size_t available = plan.scratch_size;
  if (!std::align(alignof(ResizeAxisSample), x_bytes + y_bytes, expected_x,
                  available)) {
    return false;
  }
  const auto *expected_y = static_cast<const ResizeAxisSample *>(expected_x) +
                           plan.destination_width;
  return plan.x_samples == expected_x && plan.y_samples == expected_y &&
         AddressRangeContains(plan.scratch_base, plan.scratch_size,
                              plan.x_samples, x_bytes) &&
         AddressRangeContains(plan.scratch_base, plan.scratch_size,
                              plan.y_samples, y_bytes);
}

CpuPrimitiveResult PrimitiveYuyvToRgb24(const std::uint8_t *src, int width,
                                        int height, std::size_t src_stride,
                                        std::uint8_t *dst,
                                        std::size_t dst_stride) noexcept {
  std::size_t active_src = 0;
  std::size_t active_dst = 0;
  BufferSpan src_span;
  BufferSpan dst_span;
  if (!YuyvActiveStride(width, &active_src) ||
      !RgbActiveStride(width, &active_dst) ||
      !MakeBufferSpan(src, height, src_stride, active_src, &src_span) ||
      !MakeBufferSpan(dst, height, dst_stride, active_dst, &dst_span) ||
      SpansOverlap(src_span, dst_span)) {
    return CpuPrimitiveResult::invalid_argument;
  }
  internal::YuyvToRgbDispatch(src, width, height, src_stride, dst, dst_stride);
  return CpuPrimitiveResult::success;
}

std::size_t PrimitiveRgb24ToYuyvScratchBytes(int width, int height) noexcept {
  if (width <= 0 || height <= 0)
    return 0;
  return internal::Rgb24ToYuyvDispatchScratchBytes(width, height);
}

CpuPrimitiveResult PrimitiveRgb24ToYuyv(const std::uint8_t *src, int width,
                                        int height, std::size_t src_stride,
                                        std::uint8_t *dst,
                                        std::size_t dst_stride,
                                        std::uint8_t *scratch,
                                        std::size_t scratch_size) noexcept {
  std::size_t active_src = 0;
  std::size_t active_dst = 0;
  BufferSpan src_span;
  BufferSpan dst_span;
  if (!RgbActiveStride(width, &active_src) ||
      !YuyvActiveStride(width, &active_dst) ||
      !MakeBufferSpan(src, height, src_stride, active_src, &src_span) ||
      !MakeBufferSpan(dst, height, dst_stride, active_dst, &dst_span) ||
      SpansOverlap(src_span, dst_span)) {
    return CpuPrimitiveResult::invalid_argument;
  }

  const std::size_t required = PrimitiveRgb24ToYuyvScratchBytes(width, height);
  if (required != 0 && (!scratch || scratch_size < required))
    return CpuPrimitiveResult::insufficient_scratch;
  if (scratch && scratch_size != 0) {
    BufferSpan scratch_span;
    if (!MakeContiguousSpan(scratch, scratch_size, &scratch_span) ||
        SpansOverlap(scratch_span, src_span) ||
        SpansOverlap(scratch_span, dst_span)) {
      return CpuPrimitiveResult::invalid_argument;
    }
  }

  internal::Rgb24ToYuyvDispatchWithScratch(src, width, height, src_stride, dst,
                                           dst_stride, scratch, scratch_size);
  return CpuPrimitiveResult::success;
}

CpuPrimitiveResult PrimitiveRgb24Bgr24(const std::uint8_t *src,
                                       std::uint8_t *dst, int width, int height,
                                       std::size_t src_stride,
                                       std::size_t dst_stride) noexcept {
  std::size_t active_stride = 0;
  BufferSpan src_span;
  BufferSpan dst_span;
  if (!RgbActiveStride(width, &active_stride) ||
      !MakeBufferSpan(src, height, src_stride, active_stride, &src_span) ||
      !MakeBufferSpan(dst, height, dst_stride, active_stride, &dst_span)) {
    return CpuPrimitiveResult::invalid_argument;
  }
  const bool exact_in_place = src == dst && src_stride == dst_stride;
  if (!exact_in_place && SpansOverlap(src_span, dst_span)) {
    return CpuPrimitiveResult::invalid_argument;
  }
  internal::Rgb24Bgr24Dispatch(src, dst, width, height, src_stride, dst_stride);
  return CpuPrimitiveResult::success;
}

CpuPrimitiveResult PrimitiveMirrorRgb24InPlace(std::uint8_t *rgb, int width,
                                               int height,
                                               std::size_t stride) noexcept {
  std::size_t active_stride = 0;
  BufferSpan frame_span;
  if (!RgbActiveStride(width, &active_stride) ||
      !MakeBufferSpan(rgb, height, stride, active_stride, &frame_span)) {
    return CpuPrimitiveResult::invalid_argument;
  }
  (void)frame_span;

  for (int y = 0; y < height; ++y) {
    std::uint8_t *row = rgb + static_cast<std::size_t>(y) * stride;
    for (int x = 0; x < width / 2; ++x) {
      const std::size_t left = static_cast<std::size_t>(x) * 3u;
      const std::size_t right = static_cast<std::size_t>(width - 1 - x) * 3u;
      std::swap(row[left + 0], row[right + 0]);
      std::swap(row[left + 1], row[right + 1]);
      std::swap(row[left + 2], row[right + 2]);
    }
  }
  return CpuPrimitiveResult::success;
}

std::size_t PrimitiveResizePlanScratchBytes(int source_width, int source_height,
                                            int destination_width,
                                            int destination_height) noexcept {
  if (source_width <= 0 || source_height <= 0 || destination_width <= 0 ||
      destination_height <= 0) {
    return 0;
  }

  std::size_t sample_count = 0;
  std::size_t sample_bytes = 0;
  if (static_cast<std::size_t>(destination_width) >
      std::numeric_limits<std::size_t>::max() -
          static_cast<std::size_t>(destination_height)) {
    return 0;
  }
  sample_count = static_cast<std::size_t>(destination_width) +
                 static_cast<std::size_t>(destination_height);
  if (!CheckedMultiply(sample_count, sizeof(ResizeAxisSample), &sample_bytes) ||
      sample_bytes > std::numeric_limits<std::size_t>::max() -
                         (alignof(ResizeAxisSample) - 1u)) {
    return 0;
  }
  return sample_bytes + alignof(ResizeAxisSample) - 1u;
}

CpuPrimitiveResult PrimitivePrepareResizeRgb24Bilinear(
    int source_width, int source_height, int destination_width,
    int destination_height, void *scratch, std::size_t scratch_size,
    CpuRgb24BilinearResizePlan *plan) noexcept {
  if (!plan)
    return CpuPrimitiveResult::invalid_argument;
  if (scratch && scratch_size != 0) {
    BufferSpan scratch_span;
    BufferSpan plan_span;
    if (!MakeContiguousSpan(scratch, scratch_size, &scratch_span) ||
        !MakeContiguousSpan(plan, sizeof(*plan), &plan_span) ||
        SpansOverlap(scratch_span, plan_span)) {
      return CpuPrimitiveResult::invalid_argument;
    }
  }
  *plan = {};

  const std::size_t required = PrimitiveResizePlanScratchBytes(
      source_width, source_height, destination_width, destination_height);
  if (required == 0)
    return CpuPrimitiveResult::invalid_argument;
  if (!scratch || scratch_size < required)
    return CpuPrimitiveResult::insufficient_scratch;

  std::size_t sample_count = static_cast<std::size_t>(destination_width) +
                             static_cast<std::size_t>(destination_height);
  std::size_t sample_bytes = 0;
  if (!CheckedMultiply(sample_count, sizeof(ResizeAxisSample), &sample_bytes))
    return CpuPrimitiveResult::invalid_argument;

  void *aligned = scratch;
  std::size_t space = scratch_size;
  if (!std::align(alignof(ResizeAxisSample), sample_bytes, aligned, space))
    return CpuPrimitiveResult::insufficient_scratch;

  auto *x_samples = static_cast<ResizeAxisSample *>(aligned);
  auto *y_samples = x_samples + destination_width;
  const float scale_x =
      static_cast<float>(source_width) / static_cast<float>(destination_width);
  const float scale_y = static_cast<float>(source_height) /
                        static_cast<float>(destination_height);

  for (int x = 0; x < destination_width; ++x) {
    const float source_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
    const int x0 =
        std::clamp(static_cast<int>(std::floor(source_x)), 0, source_width - 1);
    x_samples[x] = {x0, std::clamp(x0 + 1, 0, source_width - 1),
                    source_x - static_cast<float>(x0)};
  }
  for (int y = 0; y < destination_height; ++y) {
    const float source_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
    const int y0 = std::clamp(static_cast<int>(std::floor(source_y)), 0,
                              source_height - 1);
    y_samples[y] = {y0, std::clamp(y0 + 1, 0, source_height - 1),
                    source_y - static_cast<float>(y0)};
  }

  plan->table_version = kCpuPrimitiveTableVersion;
  plan->struct_size = sizeof(CpuRgb24BilinearResizePlan);
  plan->source_width = source_width;
  plan->source_height = source_height;
  plan->destination_width = destination_width;
  plan->destination_height = destination_height;
  plan->x_samples = x_samples;
  plan->y_samples = y_samples;
  plan->scratch_base = scratch;
  plan->scratch_size = scratch_size;
  plan->validation_tag = ResizePlanTag(*plan);
  return CpuPrimitiveResult::success;
}

std::uint8_t RoundClampResizeByte(float value) noexcept {
  if (value <= 0.0f)
    return 0;
  if (value >= 255.0f)
    return 255;
  return static_cast<std::uint8_t>(static_cast<int>(value + 0.5f));
}

CpuPrimitiveResult
PrimitiveApplyResizeRgb24Bilinear(const CpuRgb24BilinearResizePlan *plan,
                                  const std::uint8_t *src_rgb,
                                  std::size_t src_stride, std::uint8_t *dst_rgb,
                                  std::size_t dst_stride) noexcept {
  if (!plan || !ValidResizePlan(*plan))
    return CpuPrimitiveResult::unprepared_plan;

  std::size_t active_src_stride = 0;
  std::size_t active_dst_stride = 0;
  BufferSpan src_span;
  BufferSpan dst_span;
  BufferSpan plan_scratch_span;
  if (!RgbActiveStride(plan->source_width, &active_src_stride) ||
      !RgbActiveStride(plan->destination_width, &active_dst_stride) ||
      !MakeBufferSpan(src_rgb, plan->source_height, src_stride,
                      active_src_stride, &src_span) ||
      !MakeBufferSpan(dst_rgb, plan->destination_height, dst_stride,
                      active_dst_stride, &dst_span) ||
      !MakeContiguousSpan(plan->scratch_base, plan->scratch_size,
                          &plan_scratch_span) ||
      SpansOverlap(src_span, dst_span) ||
      SpansOverlap(src_span, plan_scratch_span) ||
      SpansOverlap(dst_span, plan_scratch_span)) {
    return CpuPrimitiveResult::invalid_argument;
  }

  const auto *x_samples =
      static_cast<const ResizeAxisSample *>(plan->x_samples);
  const auto *y_samples =
      static_cast<const ResizeAxisSample *>(plan->y_samples);

  for (int y = 0; y < plan->destination_height; ++y) {
    const ResizeAxisSample ys = y_samples[y];
    const auto *src_row0 =
        src_rgb + static_cast<std::size_t>(ys.i0) * src_stride;
    const auto *src_row1 =
        src_rgb + static_cast<std::size_t>(ys.i1) * src_stride;
    auto *dst_row = dst_rgb + static_cast<std::size_t>(y) * dst_stride;

    for (int x = 0; x < plan->destination_width; ++x) {
      const ResizeAxisSample xs = x_samples[x];
      const auto *p00 = src_row0 + static_cast<std::size_t>(xs.i0) * 3u;
      const auto *p10 = src_row0 + static_cast<std::size_t>(xs.i1) * 3u;
      const auto *p01 = src_row1 + static_cast<std::size_t>(xs.i0) * 3u;
      const auto *p11 = src_row1 + static_cast<std::size_t>(xs.i1) * 3u;
      auto *destination = dst_row + static_cast<std::size_t>(x) * 3u;

      for (std::size_t channel = 0; channel < 3u; ++channel) {
        const float row0 = static_cast<float>(p00[channel]) +
                           xs.factor * (static_cast<float>(p10[channel]) -
                                        static_cast<float>(p00[channel]));
        const float row1 = static_cast<float>(p01[channel]) +
                           xs.factor * (static_cast<float>(p11[channel]) -
                                        static_cast<float>(p01[channel]));
        destination[channel] =
            RoundClampResizeByte(row0 + ys.factor * (row1 - row0));
      }
    }

    if (dst_stride > active_dst_stride) {
      std::memset(dst_row + active_dst_stride, 0,
                  dst_stride - active_dst_stride);
    }
  }
  return CpuPrimitiveResult::success;
}

CpuPrimitiveTable BuildCpuPrimitiveTable() noexcept {
  CpuPrimitiveTable table;
  table.table_version = kCpuPrimitiveTableVersion;
  table.struct_size = sizeof(CpuPrimitiveTable);
  table.backend = hal::BackendId::cpu;
  table.semantic_effect_provider = hal::SemanticEffectProviderId::invalid;
  table.capability_bits =
      CpuPrimitiveCapabilityBit(
          hal::PrimitiveOperationId::convert_pixel_format) |
      CpuPrimitiveCapabilityBit(hal::PrimitiveOperationId::resize) |
      CpuPrimitiveCapabilityBit(hal::PrimitiveOperationId::mirror);

  table.yuyv_to_rgb24_implementation =
      internal::YuyvToRgbBackendName(internal::YuyvToRgbSelectedBackend());
  table.rgb24_to_yuyv_implementation =
      internal::Rgb24ToYuyvBackendName(internal::Rgb24ToYuyvSelectedBackend());
  table.rgb24_bgr24_implementation =
      internal::Rgb24Bgr24BackendName(internal::Rgb24Bgr24SelectedBackend());
  table.mirror_rgb24_implementation = "scalar";
  table.resize_rgb24_bilinear_implementation = "scalar_prepared_bilinear";

  table.yuyv_to_rgb24 = &PrimitiveYuyvToRgb24;
  table.rgb24_to_yuyv_scratch_bytes = &PrimitiveRgb24ToYuyvScratchBytes;
  table.rgb24_to_yuyv = &PrimitiveRgb24ToYuyv;
  table.rgb24_bgr24 = &PrimitiveRgb24Bgr24;
  table.mirror_rgb24_in_place = &PrimitiveMirrorRgb24InPlace;
  table.resize_rgb24_bilinear_plan_scratch_bytes =
      &PrimitiveResizePlanScratchBytes;
  table.prepare_resize_rgb24_bilinear = &PrimitivePrepareResizeRgb24Bilinear;
  table.apply_resize_rgb24_bilinear = &PrimitiveApplyResizeRgb24Bilinear;
  return table;
}

} // namespace

const CpuPrimitiveTable &GetCpuPrimitiveTable() noexcept {
  static const CpuPrimitiveTable table = BuildCpuPrimitiveTable();
  return table;
}

void YuyvToRgb24(const std::uint8_t *src, int width, int height,
                 std::size_t src_stride, std::uint8_t *dst,
                 std::size_t dst_stride) {
  (void)GetCpuPrimitiveTable().yuyv_to_rgb24(src, width, height, src_stride,
                                             dst, dst_stride);
}

void Rgb24ToYuyv(const std::uint8_t *src, int width, int height,
                 std::size_t src_stride, std::uint8_t *dst,
                 std::size_t dst_stride) {
  Rgb24ToYuyvWithScratch(src, width, height, src_stride, dst, dst_stride,
                         nullptr, 0);
}

std::size_t Rgb24ToYuyvScratchBytes(int width, int height) {
  return GetCpuPrimitiveTable().rgb24_to_yuyv_scratch_bytes(width, height);
}

void Rgb24ToYuyvWithScratch(const std::uint8_t *src, int width, int height,
                            std::size_t src_stride, std::uint8_t *dst,
                            std::size_t dst_stride, std::uint8_t *scratch,
                            std::size_t scratch_size) {
  const CpuPrimitiveResult result = GetCpuPrimitiveTable().rgb24_to_yuyv(
      src, width, height, src_stride, dst, dst_stride, scratch, scratch_size);
  // Preserve the historical compatibility behavior: the public wrapper uses
  // the scalar primitive when the selected optimized path needs more scratch.
  // New prepared callers should size scratch through the table and handle the
  // typed result instead of relying on this compatibility fallback.
  if (result == CpuPrimitiveResult::insufficient_scratch) {
    internal::Rgb24ToYuyvScalar(src, width, height, src_stride, dst,
                                dst_stride);
  }
}

void MirrorRgb24InPlace(std::uint8_t *rgb, int width, int height,
                        std::size_t stride) {
  (void)GetCpuPrimitiveTable().mirror_rgb24_in_place(rgb, width, height,
                                                     stride);
}

void Rgb24ToBgr24(const std::uint8_t *src, std::uint8_t *dst, int width,
                  int height, std::size_t src_stride, std::size_t dst_stride) {
  (void)GetCpuPrimitiveTable().rgb24_bgr24(src, dst, width, height, src_stride,
                                           dst_stride);
}

} // namespace studiocast::video
