#pragma once

#include <cstddef>
#include <cstdint>

#include "core/video/effects/broadcast_effects.h"
#include "core/video/v4l2_capture.h"
#include "core/video/v4l2_writer.h"

namespace studiocast::video {

enum class RawYuyvPassthroughBlocker {
  none,
  capture_format,
  output_format,
  rgb_operation,
  invalid_geometry,
  odd_width,
  geometry_mismatch,
  capture_stride,
  output_stride,
  capture_size_image,
  output_size_image,
  size_overflow,
};

struct RawYuyvPassthroughPlan {
  bool eligible = false;
  RawYuyvPassthroughBlocker blocker =
      RawYuyvPassthroughBlocker::invalid_geometry;
  int width = 0;
  int height = 0;
  std::size_t active_row_bytes = 0;
  std::size_t capture_stride = 0;
  std::size_t output_stride = 0;
  std::size_t capture_layout_bytes = 0;
  std::size_t output_size_image = 0;
  bool exact_layout = false;
};

// Counts the actual frame-path operations. The conversion helpers below are
// used at CameraPipeline's conversion call sites, while a successful raw copy
// increments only raw_passthrough_frames/bytes.
struct YuyvFramePathCounters {
  std::uint64_t capture_to_rgb_calls = 0;
  std::uint64_t output_from_rgb_calls = 0;
  std::uint64_t raw_passthrough_frames = 0;
  std::uint64_t raw_passthrough_bytes = 0;
};

enum class YuyvCaptureFrameValidation {
  ok,
  format,
  invalid_geometry,
  odd_width,
  stride,
  size_image,
  frame_bytes,
  size_overflow,
};

YuyvCaptureFrameValidation
ValidateYuyvCaptureFrame(const CaptureFormat &capture, std::size_t frame_bytes,
                         std::size_t *layout_bytes = nullptr);

bool BroadcastEffectsRequireRgbOperation(
    const effects::BroadcastCameraEffects &effects);

RawYuyvPassthroughPlan
PrepareRawYuyvPassthroughPlan(const CaptureFormat &capture,
                              const ActualFormat &output,
                              const effects::BroadcastCameraEffects &effects);

enum class RawYuyvCopyResult {
  ok,
  ineligible_plan,
  null_buffer,
  source_too_small,
  destination_too_small,
};

// Copies into caller-owned bounded storage. Exact layouts preserve every
// supplied row byte, including padding, and preserve a supplied size_image
// tail; an omitted tail is zeroed. Differing safe layouts copy active row bytes
// and initialize all destination padding/tail to zero.
RawYuyvCopyResult CopyRawYuyvFrame(const RawYuyvPassthroughPlan &plan,
                                   const std::uint8_t *source,
                                   std::size_t source_bytes,
                                   std::uint8_t *destination,
                                   std::size_t destination_bytes,
                                   YuyvFramePathCounters *counters);

void CountedYuyvToRgb24(const std::uint8_t *source, int width, int height,
                        std::size_t source_stride, std::uint8_t *destination,
                        std::size_t destination_stride,
                        YuyvFramePathCounters *counters);

void CountedRgb24ToYuyvWithScratch(const std::uint8_t *source, int width,
                                   int height, std::size_t source_stride,
                                   std::uint8_t *destination,
                                   std::size_t destination_stride,
                                   std::uint8_t *scratch,
                                   std::size_t scratch_bytes,
                                   YuyvFramePathCounters *counters);

} // namespace studiocast::video
