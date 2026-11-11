#include "core/video/yuyv_passthrough.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace studiocast::tests {
namespace {

video::CaptureFormat Capture(int width = 4, int height = 2,
                             std::size_t stride = 8,
                             std::size_t size_image = 16) {
  video::CaptureFormat format;
  format.width = width;
  format.height = height;
  format.format = video::CapturePixelFormat::yuyv;
  format.bytes_per_line = stride;
  format.size_image = size_image;
  return format;
}

video::ActualFormat Output(int width = 4, int height = 2,
                           std::size_t stride = 8,
                           std::size_t size_image = 16) {
  video::ActualFormat format;
  format.width = width;
  format.height = height;
  format.format = video::PixelFormat::yuyv;
  format.bytes_per_line = stride;
  format.size_image = size_image;
  return format;
}

bool HasBlocker(const video::CaptureFormat &capture,
                const video::ActualFormat &output,
                const video::effects::BroadcastCameraEffects &effects,
                video::RawYuyvPassthroughBlocker blocker) {
  const auto plan =
      video::PrepareRawYuyvPassthroughPlan(capture, output, effects);
  return !plan.eligible && plan.blocker == blocker;
}

} // namespace

bool TestRawYuyvPassthroughPredicateIsStrict() {
  const auto capture = Capture();
  const auto output = Output();
  video::effects::BroadcastCameraEffects effects;

  const auto eligible =
      video::PrepareRawYuyvPassthroughPlan(capture, output, effects);
  if (!eligible.eligible || !eligible.exact_layout || eligible.width != 4 ||
      eligible.height != 2 || eligible.active_row_bytes != 8 ||
      eligible.capture_layout_bytes != 16 || eligible.output_size_image != 16)
    return false;

  auto capture_bad = capture;
  capture_bad.format = video::CapturePixelFormat::mjpeg;
  if (!HasBlocker(capture_bad, output, effects,
                  video::RawYuyvPassthroughBlocker::capture_format))
    return false;

  auto output_bad = output;
  output_bad.format = video::PixelFormat::rgb24;
  if (!HasBlocker(capture, output_bad, effects,
                  video::RawYuyvPassthroughBlocker::output_format))
    return false;

  const auto requires_rgb = [&](const auto &mutate) {
    auto candidate = effects;
    mutate(candidate);
    return HasBlocker(capture, output, candidate,
                      video::RawYuyvPassthroughBlocker::rgb_operation);
  };
  if (!requires_rgb([](auto &fx) { fx.mirror = true; }) ||
      !requires_rgb([](auto &fx) {
        fx.virtual_background.mode =
            video::effects::VirtualBackgroundMode::blur;
      }) ||
      !requires_rgb([](auto &fx) {
        fx.virtual_background.mode =
            video::effects::VirtualBackgroundMode::remove;
      }) ||
      !requires_rgb([](auto &fx) {
        fx.virtual_background.mode =
            video::effects::VirtualBackgroundMode::replace;
      }) ||
      !requires_rgb([](auto &fx) { fx.auto_frame.enabled = true; }) ||
      !requires_rgb([](auto &fx) { fx.eye_contact.enabled = true; }) ||
      !requires_rgb([](auto &fx) { fx.video_noise_removal.enabled = true; }) ||
      !requires_rgb([](auto &fx) { fx.virtual_key_light.enabled = true; }) ||
      !requires_rgb([](auto &fx) { fx.vignette.enabled = true; })) {
    return false;
  }

  // Inactive configuration details are not operations by themselves.
  auto inactive = effects;
  inactive.auto_frame.model_id = "configured-but-disabled";
  inactive.eye_contact.model_id = "configured-but-disabled";
  inactive.video_noise_removal.model_id = "configured-but-disabled";
  inactive.virtual_key_light.hdri_path = "/not-read-on-the-frame-path";
  if (!video::PrepareRawYuyvPassthroughPlan(capture, output, inactive).eligible)
    return false;

  if (!HasBlocker(Capture(0, 2, 8, 16), output, effects,
                  video::RawYuyvPassthroughBlocker::invalid_geometry) ||
      !HasBlocker(capture, Output(4, 0, 8, 16), effects,
                  video::RawYuyvPassthroughBlocker::invalid_geometry) ||
      !HasBlocker(Capture(3, 2, 8, 16), Output(3, 2, 8, 16), effects,
                  video::RawYuyvPassthroughBlocker::odd_width) ||
      !HasBlocker(capture, Output(3, 2, 8, 16), effects,
                  video::RawYuyvPassthroughBlocker::odd_width) ||
      !HasBlocker(capture, Output(6, 2, 12, 24), effects,
                  video::RawYuyvPassthroughBlocker::geometry_mismatch) ||
      !HasBlocker(Capture(4, 2, 7, 14), output, effects,
                  video::RawYuyvPassthroughBlocker::capture_stride) ||
      !HasBlocker(capture, Output(4, 2, 7, 14), effects,
                  video::RawYuyvPassthroughBlocker::output_stride) ||
      !HasBlocker(Capture(4, 2, 8, 15), output, effects,
                  video::RawYuyvPassthroughBlocker::capture_size_image) ||
      !HasBlocker(capture, Output(4, 2, 8, 15), effects,
                  video::RawYuyvPassthroughBlocker::output_size_image)) {
    return false;
  }

  // A negotiated output change gets a distinct prepared layout.
  const auto changed = video::PrepareRawYuyvPassthroughPlan(
      capture, Output(4, 2, 12, 24), effects);
  return changed.eligible && !changed.exact_layout &&
         changed.output_stride == 12 && changed.output_size_image == 24;
}

bool TestRawYuyvPassthroughPreservesExactLayouts() {
  video::YuyvFramePathCounters counters;

  for (const auto &format :
       std::array<video::CaptureFormat, 2>{Capture(), Capture(4, 2, 12, 27)}) {
    const auto output = Output(format.width, format.height,
                               format.bytes_per_line, format.size_image);
    const auto plan = video::PrepareRawYuyvPassthroughPlan(
        format, output, video::effects::BroadcastCameraEffects{});
    if (!plan.eligible || !plan.exact_layout)
      return false;

    std::vector<std::uint8_t> source(format.size_image);
    for (std::size_t i = 0; i < source.size(); ++i)
      source[i] = static_cast<std::uint8_t>((i * 37u + 11u) & 0xffu);
    std::vector<std::uint8_t> destination(source.size(), 0);
    if (video::CopyRawYuyvFrame(plan, source.data(), source.size(),
                                destination.data(), destination.size(),
                                &counters) != video::RawYuyvCopyResult::ok ||
        destination != source) {
      return false;
    }

    // The destination remains valid after the capture-owned source changes.
    const auto copied = destination;
    std::fill(source.begin(), source.end(), 0);
    if (destination != copied)
      return false;
  }

  return counters.raw_passthrough_frames == 2 &&
         counters.raw_passthrough_bytes == 43 &&
         counters.capture_to_rgb_calls == 0 &&
         counters.output_from_rgb_calls == 0;
}

bool TestRawYuyvPassthroughCopiesActiveRowsAndZerosPadding() {
  const auto capture = Capture(4, 2, 10, 23);
  const auto output = Output(4, 2, 12, 29);
  const auto plan = video::PrepareRawYuyvPassthroughPlan(
      capture, output, video::effects::BroadcastCameraEffects{});
  if (!plan.eligible || plan.exact_layout)
    return false;

  std::vector<std::uint8_t> source(capture.size_image, 0xee);
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t byte = 0; byte < 8; ++byte)
      source[row * 10 + byte] = static_cast<std::uint8_t>(row * 32 + byte);
  }
  std::vector<std::uint8_t> destination(output.size_image, 0xab);
  video::YuyvFramePathCounters counters;
  if (video::CopyRawYuyvFrame(plan, source.data(), source.size(),
                              destination.data(), destination.size(),
                              &counters) != video::RawYuyvCopyResult::ok) {
    return false;
  }

  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t byte = 0; byte < 8; ++byte) {
      if (destination[row * 12 + byte] != source[row * 10 + byte])
        return false;
    }
    for (std::size_t byte = 8; byte < 12; ++byte) {
      if (destination[row * 12 + byte] != 0)
        return false;
    }
  }
  for (std::size_t byte = 24; byte < destination.size(); ++byte) {
    if (destination[byte] != 0)
      return false;
  }

  return counters.raw_passthrough_frames == 1 &&
         counters.raw_passthrough_bytes == destination.size();
}

bool TestRawYuyvPassthroughRejectsUnsafeFrames() {
  const auto capture = Capture(4, 2, 8, 20);
  const auto output = Output(4, 2, 8, 20);
  const auto plan = video::PrepareRawYuyvPassthroughPlan(
      capture, output, video::effects::BroadcastCameraEffects{});
  std::vector<std::uint8_t> source(20, 0x41);
  std::vector<std::uint8_t> destination(20, 0x55);
  video::YuyvFramePathCounters counters;

  if (video::ValidateYuyvCaptureFrame(capture, source.size()) !=
          video::YuyvCaptureFrameValidation::ok ||
      video::ValidateYuyvCaptureFrame(capture, 15) !=
          video::YuyvCaptureFrameValidation::frame_bytes ||
      video::ValidateYuyvCaptureFrame(Capture(3, 2, 8, 16), 16) !=
          video::YuyvCaptureFrameValidation::odd_width ||
      video::ValidateYuyvCaptureFrame(Capture(4, 2, 7, 14), 14) !=
          video::YuyvCaptureFrameValidation::stride ||
      video::ValidateYuyvCaptureFrame(Capture(4, 2, 8, 15), 15) !=
          video::YuyvCaptureFrameValidation::size_image) {
    return false;
  }

  const auto overflow = Capture(2, 2, std::numeric_limits<std::size_t>::max(),
                                std::numeric_limits<std::size_t>::max());
  if (video::ValidateYuyvCaptureFrame(overflow, overflow.size_image) !=
      video::YuyvCaptureFrameValidation::size_overflow) {
    return false;
  }

  if (video::CopyRawYuyvFrame(plan, nullptr, source.size(), destination.data(),
                              destination.size(), &counters) !=
          video::RawYuyvCopyResult::null_buffer ||
      video::CopyRawYuyvFrame(plan, source.data(), 15, destination.data(),
                              destination.size(), &counters) !=
          video::RawYuyvCopyResult::source_too_small ||
      video::CopyRawYuyvFrame(plan, source.data(), source.size(),
                              destination.data(), 19, &counters) !=
          video::RawYuyvCopyResult::destination_too_small) {
    return false;
  }

  // bytesused need only cover the negotiated row layout. If an otherwise
  // exact layout omits an unused size_image tail, the tail is zeroed.
  std::vector<std::uint8_t> short_tail_destination(20, 0x77);
  if (video::CopyRawYuyvFrame(plan, source.data(), 16,
                              short_tail_destination.data(),
                              short_tail_destination.size(), nullptr) !=
          video::RawYuyvCopyResult::ok ||
      !std::equal(source.begin(), source.begin() + 16,
                  short_tail_destination.begin()) ||
      !std::all_of(short_tail_destination.begin() + 16,
                   short_tail_destination.end(),
                   [](std::uint8_t value) { return value == 0; })) {
    return false;
  }

  auto ineligible = plan;
  ineligible.eligible = false;
  if (video::CopyRawYuyvFrame(ineligible, source.data(), source.size(),
                              destination.data(), destination.size(),
                              &counters) !=
      video::RawYuyvCopyResult::ineligible_plan) {
    return false;
  }
  return counters.raw_passthrough_frames == 0 &&
         std::all_of(destination.begin(), destination.end(),
                     [](std::uint8_t value) { return value == 0x55; });
}

bool TestRawYuyvPassthroughCountersTrackActualCallSites() {
  const auto plan = video::PrepareRawYuyvPassthroughPlan(
      Capture(2, 1, 4, 4), Output(2, 1, 4, 4),
      video::effects::BroadcastCameraEffects{});
  std::array<std::uint8_t, 4> yuyv{{16, 128, 235, 128}};
  std::array<std::uint8_t, 6> rgb{};
  std::array<std::uint8_t, 4> converted{};
  std::array<std::uint8_t, 4> raw{};
  video::YuyvFramePathCounters counters;

  if (video::CopyRawYuyvFrame(plan, yuyv.data(), yuyv.size(), raw.data(),
                              raw.size(),
                              &counters) != video::RawYuyvCopyResult::ok) {
    return false;
  }
  video::CountedYuyvToRgb24(yuyv.data(), 2, 1, 4, rgb.data(), 6, &counters);
  video::CountedRgb24ToYuyvWithScratch(rgb.data(), 2, 1, 6, converted.data(), 4,
                                       nullptr, 0, &counters);

  return raw == yuyv && counters.raw_passthrough_frames == 1 &&
         counters.raw_passthrough_bytes == 4 &&
         counters.capture_to_rgb_calls == 1 &&
         counters.output_from_rgb_calls == 1;
}

} // namespace studiocast::tests
