#include "core/video/effects/background_blur_cpu.h"
#include "core/video/effects/background_remove_cpu.h"
#include "core/video/effects/box_blur.h"
#include "core/video/effects/mirror_effect.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace studiocast::tests {
namespace {

struct EffectCase {
  int width;
  int height;
  std::size_t padding;
};

void FillDeterministicRgb(std::vector<std::uint8_t> *frame, int width,
                          int height, std::size_t stride,
                          std::uint32_t seed) {
  std::uint32_t state = seed;
  for (int y = 0; y < height; ++y) {
    auto *row = frame->data() + static_cast<std::size_t>(y) * stride;
    for (std::size_t x = 0; x < static_cast<std::size_t>(width) * 3u; ++x) {
      state = state * 1664525u + 1013904223u;
      row[x] = static_cast<std::uint8_t>((state >> 24) & 0xffu);
    }
  }
}

void ApplyReferenceRemove(std::vector<std::uint8_t> *frame, int width,
                          int height, std::size_t stride) {
  const int left = static_cast<int>(width * 0.25);
  const int right = static_cast<int>(width * 0.75);
  const int top = static_cast<int>(height * 0.15);
  const int bottom = static_cast<int>(height * 0.95);
  const int feather = std::max(8, std::min(width, height) / 20);

  for (int y = 0; y < height; ++y) {
    auto *row = frame->data() + static_cast<std::size_t>(y) * stride;

    int dy = 0;
    if (y < top)
      dy = top - y;
    else if (y > bottom)
      dy = y - bottom;

    for (int x = 0; x < width; ++x) {
      int dx = 0;
      if (x < left)
        dx = left - x;
      else if (x > right)
        dx = x - right;

      const int d = std::max(dx, dy);
      if (d <= 0)
        continue;

      auto *p = row + static_cast<std::size_t>(x) * 3u;
      if (d >= feather) {
        p[0] = 0;
        p[1] = 255;
        p[2] = 0;
      } else {
        const int a = d;
        const int ia = feather - d;
        p[0] =
            static_cast<std::uint8_t>((static_cast<int>(p[0]) * ia) / feather);
        p[1] = static_cast<std::uint8_t>(
            (static_cast<int>(p[1]) * ia + 255 * a) / feather);
        p[2] =
            static_cast<std::uint8_t>((static_cast<int>(p[2]) * ia) / feather);
      }
    }
  }
}

void ApplyReferenceBlur(std::vector<std::uint8_t> *frame, int width, int height,
                        std::size_t stride, int strength) {
  const std::size_t tight_stride = static_cast<std::size_t>(width) * 3u;
  std::vector<std::uint8_t> blurred(tight_stride *
                                    static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    const auto *src_row = frame->data() + static_cast<std::size_t>(y) * stride;
    auto *dst_row = blurred.data() + static_cast<std::size_t>(y) * tight_stride;
    std::memcpy(dst_row, src_row, tight_stride);
  }

  video::effects::Rgb24FrameView blur_view;
  blur_view.data = blurred.data();
  blur_view.width = width;
  blur_view.height = height;
  blur_view.stride_bytes = tight_stride;
  std::vector<std::uint8_t> scratch;
  video::effects::BoxBlurRgb24InPlace(blur_view, strength, &scratch);

  const int left = static_cast<int>(width * 0.25);
  const int right = static_cast<int>(width * 0.75);
  const int top = static_cast<int>(height * 0.15);
  const int bottom = static_cast<int>(height * 0.95);
  const int feather = std::max(8, std::min(width, height) / 20);

  for (int y = 0; y < height; ++y) {
    auto *out_row = frame->data() + static_cast<std::size_t>(y) * stride;
    const auto *blur_row =
        blurred.data() + static_cast<std::size_t>(y) * tight_stride;

    int dy = 0;
    if (y < top)
      dy = top - y;
    else if (y > bottom)
      dy = y - bottom;

    for (int x = 0; x < width; ++x) {
      int dx = 0;
      if (x < left)
        dx = left - x;
      else if (x > right)
        dx = x - right;

      const int d = std::max(dx, dy);
      if (d <= 0)
        continue;

      const std::size_t off = static_cast<std::size_t>(x) * 3u;
      const auto *b = blur_row + off;
      auto *o = out_row + off;
      if (d >= feather) {
        o[0] = b[0];
        o[1] = b[1];
        o[2] = b[2];
      } else {
        const int a = d;
        const int ia = feather - d;
        o[0] = static_cast<std::uint8_t>(
            (static_cast<int>(o[0]) * ia + static_cast<int>(b[0]) * a) /
            feather);
        o[1] = static_cast<std::uint8_t>(
            (static_cast<int>(o[1]) * ia + static_cast<int>(b[1]) * a) /
            feather);
        o[2] = static_cast<std::uint8_t>(
            (static_cast<int>(o[2]) * ia + static_cast<int>(b[2]) * a) /
            feather);
      }
    }
  }
}

bool PaddingUnchanged(const std::vector<std::uint8_t> &before,
                      const std::vector<std::uint8_t> &after, int width,
                      int height, std::size_t stride) {
  const std::size_t active = static_cast<std::size_t>(width) * 3u;
  for (int y = 0; y < height; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * stride;
    for (std::size_t i = active; i < stride; ++i) {
      if (before[row + i] != after[row + i])
        return false;
    }
  }
  return true;
}

} // namespace

bool TestBackgroundRemoveCpuMatchesReferenceAndPreservesPadding() {
  const std::vector<EffectCase> cases{{17, 11, 5}, {32, 13, 7}, {17, 11, 5}};
  video::effects::BackgroundRemoveCpuEffect effect;
  video::effects::EffectContext ctx;

  for (const EffectCase &tc : cases) {
    const std::size_t stride = static_cast<std::size_t>(tc.width) * 3u +
                               static_cast<std::size_t>(tc.padding);
    std::vector<std::uint8_t> actual(stride *
                                         static_cast<std::size_t>(tc.height),
                                     0xa5u);
    FillDeterministicRgb(&actual, tc.width, tc.height, stride,
                         static_cast<std::uint32_t>(tc.width) * 31u +
                             static_cast<std::uint32_t>(tc.height) * 17u);
    std::vector<std::uint8_t> expected = actual;
    const std::vector<std::uint8_t> before = actual;

    ApplyReferenceRemove(&expected, tc.width, tc.height, stride);

    video::effects::Rgb24FrameView view;
    view.data = actual.data();
    view.width = tc.width;
    view.height = tc.height;
    view.stride_bytes = stride;
    effect.Apply(view, &ctx);

    if (actual != expected) {
      std::cerr << "background remove output mismatch for " << tc.width << "x"
                << tc.height << "\n";
      return false;
    }
    if (!PaddingUnchanged(before, actual, tc.width, tc.height, stride)) {
      std::cerr << "background remove changed row padding for " << tc.width
                << "x" << tc.height << "\n";
      return false;
    }
  }

  return true;
}

bool TestBackgroundBlurCpuMatchesReferenceAndPreservesPadding() {
  constexpr int kStrength = 3;
  const std::vector<EffectCase> cases{{17, 11, 5}, {32, 13, 7}, {17, 11, 5}};
  video::effects::BackgroundBlurCpuEffect effect(kStrength);
  video::effects::EffectContext ctx;

  for (const EffectCase &tc : cases) {
    const std::size_t stride = static_cast<std::size_t>(tc.width) * 3u +
                               static_cast<std::size_t>(tc.padding);
    std::vector<std::uint8_t> actual(stride *
                                         static_cast<std::size_t>(tc.height),
                                     0x5au);
    FillDeterministicRgb(&actual, tc.width, tc.height, stride,
                         0xabcdefu +
                             static_cast<std::uint32_t>(tc.width) * 13u +
                             static_cast<std::uint32_t>(tc.height) * 29u);
    std::vector<std::uint8_t> expected = actual;
    const std::vector<std::uint8_t> before = actual;

    ApplyReferenceBlur(&expected, tc.width, tc.height, stride, kStrength);

    video::effects::Rgb24FrameView view;
    view.data = actual.data();
    view.width = tc.width;
    view.height = tc.height;
    view.stride_bytes = stride;
    effect.Apply(view, &ctx);

    if (actual != expected) {
      std::cerr << "background blur output mismatch for " << tc.width << "x"
                << tc.height << "\n";
      return false;
    }
    if (!PaddingUnchanged(before, actual, tc.width, tc.height, stride)) {
      std::cerr << "background blur changed row padding for " << tc.width
                << "x" << tc.height << "\n";
      return false;
    }
  }

  return true;
}

bool TestMirrorCpuReferenceCoversPaddingOddEvenDegenerateAndDoubleMirror() {
  const std::vector<EffectCase> cases{{1, 1, 5}, {5, 3, 7}, {6, 4, 11}};
  video::effects::MirrorEffect effect;
  video::effects::EffectContext ctx;

  for (const EffectCase &tc : cases) {
    const std::size_t stride = static_cast<std::size_t>(tc.width) * 3u +
                               static_cast<std::size_t>(tc.padding);
    std::vector<std::uint8_t> actual(
        stride * static_cast<std::size_t>(tc.height), 0xd3u);
    FillDeterministicRgb(
        &actual, tc.width, tc.height, stride,
        0x4d495252u + static_cast<std::uint32_t>(tc.width * 31 + tc.height));
    const std::vector<std::uint8_t> original = actual;
    std::vector<std::uint8_t> expected = original;
    for (int y = 0; y < tc.height; ++y) {
      const auto *src = original.data() + static_cast<std::size_t>(y) * stride;
      auto *dst = expected.data() + static_cast<std::size_t>(y) * stride;
      for (int x = 0; x < tc.width; ++x) {
        const std::size_t src_offset =
            static_cast<std::size_t>(tc.width - 1 - x) * 3u;
        const std::size_t dst_offset = static_cast<std::size_t>(x) * 3u;
        std::memcpy(dst + dst_offset, src + src_offset, 3u);
      }
    }

    video::effects::Rgb24FrameView view;
    view.data = actual.data();
    view.width = tc.width;
    view.height = tc.height;
    view.stride_bytes = stride;
    effect.Apply(view, &ctx);
    if (actual != expected ||
        !PaddingUnchanged(original, actual, tc.width, tc.height, stride)) {
      std::cerr << "mirror CPU reference mismatch for " << tc.width << "x"
                << tc.height << "\n";
      return false;
    }

    effect.Apply(view, &ctx);
    if (actual != original) {
      std::cerr << "double mirror did not restore exact padded frame for "
                << tc.width << "x" << tc.height << "\n";
      return false;
    }
  }
  return true;
}

} // namespace studiocast::tests
