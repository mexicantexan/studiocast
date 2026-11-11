#include "core/video/yuyv_passthrough.h"

#include <cstring>
#include <limits>

#include "core/video/convert.h"

namespace studiocast::video {
namespace {

bool CheckedMultiply(std::size_t left, std::size_t right, std::size_t *result) {
  if (!result)
    return false;
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  *result = left * right;
  return true;
}

} // namespace

YuyvCaptureFrameValidation
ValidateYuyvCaptureFrame(const CaptureFormat &capture, std::size_t frame_bytes,
                         std::size_t *layout_bytes) {
  if (layout_bytes)
    *layout_bytes = 0;
  if (capture.format != CapturePixelFormat::yuyv)
    return YuyvCaptureFrameValidation::format;
  if (capture.width <= 0 || capture.height <= 0)
    return YuyvCaptureFrameValidation::invalid_geometry;
  if ((capture.width & 1) != 0)
    return YuyvCaptureFrameValidation::odd_width;

  std::size_t active_row_bytes = 0;
  if (!CheckedMultiply(static_cast<std::size_t>(capture.width), 2u,
                       &active_row_bytes)) {
    return YuyvCaptureFrameValidation::size_overflow;
  }
  if (capture.bytes_per_line < active_row_bytes)
    return YuyvCaptureFrameValidation::stride;

  std::size_t required = 0;
  if (!CheckedMultiply(capture.bytes_per_line,
                       static_cast<std::size_t>(capture.height), &required)) {
    return YuyvCaptureFrameValidation::size_overflow;
  }
  if (capture.size_image < required)
    return YuyvCaptureFrameValidation::size_image;
  if (frame_bytes < required)
    return YuyvCaptureFrameValidation::frame_bytes;
  if (layout_bytes)
    *layout_bytes = required;
  return YuyvCaptureFrameValidation::ok;
}

bool BroadcastEffectsRequireRgbOperation(
    const effects::BroadcastCameraEffects &effects) {
  return effects.mirror ||
         effects.virtual_background.mode !=
             effects::VirtualBackgroundMode::none ||
         effects.auto_frame.enabled || effects.eye_contact.enabled ||
         effects.video_noise_removal.enabled ||
         effects.virtual_key_light.enabled || effects.vignette.enabled;
}

RawYuyvPassthroughPlan
PrepareRawYuyvPassthroughPlan(const CaptureFormat &capture,
                              const ActualFormat &output,
                              const effects::BroadcastCameraEffects &effects) {
  RawYuyvPassthroughPlan plan;

  if (capture.format != CapturePixelFormat::yuyv) {
    plan.blocker = RawYuyvPassthroughBlocker::capture_format;
    return plan;
  }
  if (output.format != PixelFormat::yuyv) {
    plan.blocker = RawYuyvPassthroughBlocker::output_format;
    return plan;
  }
  if (BroadcastEffectsRequireRgbOperation(effects)) {
    plan.blocker = RawYuyvPassthroughBlocker::rgb_operation;
    return plan;
  }
  if (capture.width <= 0 || capture.height <= 0 || output.width <= 0 ||
      output.height <= 0) {
    plan.blocker = RawYuyvPassthroughBlocker::invalid_geometry;
    return plan;
  }
  if ((capture.width & 1) != 0 || (output.width & 1) != 0) {
    plan.blocker = RawYuyvPassthroughBlocker::odd_width;
    return plan;
  }
  if (capture.width != output.width || capture.height != output.height) {
    plan.blocker = RawYuyvPassthroughBlocker::geometry_mismatch;
    return plan;
  }

  const std::size_t width = static_cast<std::size_t>(capture.width);
  const std::size_t height = static_cast<std::size_t>(capture.height);
  if (!CheckedMultiply(width, 2u, &plan.active_row_bytes)) {
    plan.blocker = RawYuyvPassthroughBlocker::size_overflow;
    return plan;
  }
  plan.width = capture.width;
  plan.height = capture.height;
  plan.capture_stride = capture.bytes_per_line;
  plan.output_stride = output.bytes_per_line;
  plan.output_size_image = output.size_image;

  if (plan.capture_stride < plan.active_row_bytes) {
    plan.blocker = RawYuyvPassthroughBlocker::capture_stride;
    return plan;
  }
  if (plan.output_stride < plan.active_row_bytes) {
    plan.blocker = RawYuyvPassthroughBlocker::output_stride;
    return plan;
  }

  std::size_t minimum_capture_size = 0;
  std::size_t minimum_output_size = 0;
  if (!CheckedMultiply(plan.capture_stride, height, &minimum_capture_size) ||
      !CheckedMultiply(plan.output_stride, height, &minimum_output_size)) {
    plan.blocker = RawYuyvPassthroughBlocker::size_overflow;
    return plan;
  }
  if (capture.size_image < minimum_capture_size) {
    plan.blocker = RawYuyvPassthroughBlocker::capture_size_image;
    return plan;
  }
  if (output.size_image < minimum_output_size) {
    plan.blocker = RawYuyvPassthroughBlocker::output_size_image;
    return plan;
  }

  plan.capture_layout_bytes = minimum_capture_size;
  plan.exact_layout = plan.capture_stride == plan.output_stride &&
                      capture.size_image == output.size_image;
  plan.blocker = RawYuyvPassthroughBlocker::none;
  plan.eligible = true;
  return plan;
}

RawYuyvCopyResult CopyRawYuyvFrame(const RawYuyvPassthroughPlan &plan,
                                   const std::uint8_t *source,
                                   std::size_t source_bytes,
                                   std::uint8_t *destination,
                                   std::size_t destination_bytes,
                                   YuyvFramePathCounters *counters) {
  if (!plan.eligible)
    return RawYuyvCopyResult::ineligible_plan;
  if (!source || !destination)
    return RawYuyvCopyResult::null_buffer;
  if (source_bytes < plan.capture_layout_bytes)
    return RawYuyvCopyResult::source_too_small;
  if (destination_bytes < plan.output_size_image)
    return RawYuyvCopyResult::destination_too_small;

  if (plan.exact_layout) {
    // The passthrough contract requires bytesused to cover the negotiated row
    // layout, while allowing an unused size_image tail to be omitted. Preserve
    // the full tail when supplied, otherwise initialize it deterministically.
    if (source_bytes >= plan.output_size_image) {
      std::memcpy(destination, source, plan.output_size_image);
    } else {
      std::memcpy(destination, source, plan.capture_layout_bytes);
      std::memset(destination + plan.capture_layout_bytes, 0,
                  plan.output_size_image - plan.capture_layout_bytes);
    }
  } else {
    std::memset(destination, 0, plan.output_size_image);
    for (int row = 0; row < plan.height; ++row) {
      const auto row_index = static_cast<std::size_t>(row);
      std::memcpy(destination + row_index * plan.output_stride,
                  source + row_index * plan.capture_stride,
                  plan.active_row_bytes);
    }
  }

  if (counters) {
    ++counters->raw_passthrough_frames;
    counters->raw_passthrough_bytes += plan.output_size_image;
  }
  return RawYuyvCopyResult::ok;
}

void CountedYuyvToRgb24(const std::uint8_t *source, int width, int height,
                        std::size_t source_stride, std::uint8_t *destination,
                        std::size_t destination_stride,
                        YuyvFramePathCounters *counters) {
  if (counters)
    ++counters->capture_to_rgb_calls;
  YuyvToRgb24(source, width, height, source_stride, destination,
              destination_stride);
}

void CountedRgb24ToYuyvWithScratch(const std::uint8_t *source, int width,
                                   int height, std::size_t source_stride,
                                   std::uint8_t *destination,
                                   std::size_t destination_stride,
                                   std::uint8_t *scratch,
                                   std::size_t scratch_bytes,
                                   YuyvFramePathCounters *counters) {
  if (counters)
    ++counters->output_from_rgb_calls;
  Rgb24ToYuyvWithScratch(source, width, height, source_stride, destination,
                         destination_stride, scratch, scratch_bytes);
}

} // namespace studiocast::video
