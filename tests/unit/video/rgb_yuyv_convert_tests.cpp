#include "core/video/convert.h"
#include "core/video/convert_rgb_bgr_internal.h"
#include "core/video/convert_rgb_yuyv_internal.h"
#include "core/video/convert_yuyv_rgb_internal.h"
#include "core/video/image_ppm.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace studiocast::tests {
namespace {

inline int RgbToY(int r, int g, int b) {
  return ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
}

inline int RgbToU(int r, int g, int b) {
  return ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
}

inline int RgbToV(int r, int g, int b) {
  return ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
}

struct ConvertCase {
  int width;
  int height;
  std::size_t src_padding;
  std::size_t dst_padding;
};

struct BackendCase {
  video::internal::Rgb24ToYuyvBackend backend;
  int chroma_tolerance;
};

struct YuyvToRgbBackendCase {
  video::internal::YuyvToRgbBackend backend;
};

struct RgbBgrBackendCase {
  video::internal::Rgb24Bgr24Backend backend;
};

constexpr std::size_t ActiveYuyvBytes(int width) {
  return static_cast<std::size_t>((width + 1) / 2) * 4u;
}

constexpr std::size_t ActiveRgbBytes(int width) {
  return static_cast<std::size_t>(width) * 3u;
}

inline std::uint8_t ClampReferenceByte(int v) {
  if (v < 0)
    return 0;
  if (v > 255)
    return 255;
  return static_cast<std::uint8_t>(v);
}

void YuvToRgbReference(int y, int u, int v, std::uint8_t *r, std::uint8_t *g,
                       std::uint8_t *b) {
  int c = y - 16;
  if (c < 0)
    c = 0;
  const int d = u - 128;
  const int e = v - 128;

  *r = ClampReferenceByte((298 * c + 409 * e + 128) >> 8);
  *g = ClampReferenceByte((298 * c - 100 * d - 208 * e + 128) >> 8);
  *b = ClampReferenceByte((298 * c + 516 * d + 128) >> 8);
}

void FillDeterministicRgb(std::vector<std::uint8_t> *src, int width, int height,
                          std::size_t stride, std::uint32_t seed) {
  std::uint32_t state = seed;
  for (int y = 0; y < height; ++y) {
    std::uint8_t *row = src->data() + static_cast<std::size_t>(y) * stride;
    for (std::size_t x = 0; x < static_cast<std::size_t>(width) * 3u; ++x) {
      state = state * 1664525u + 1013904223u;
      row[x] = static_cast<std::uint8_t>((state >> 24) & 0xffu);
    }
    for (std::size_t x = static_cast<std::size_t>(width) * 3u; x < stride;
         ++x) {
      row[x] = static_cast<std::uint8_t>(0xa5u);
    }
  }
}

bool ConvertWithBackend(video::internal::Rgb24ToYuyvBackend backend,
                        const std::uint8_t *src, int width, int height,
                        std::size_t src_stride, std::uint8_t *dst,
                        std::size_t dst_stride) {
  using video::internal::Rgb24ToYuyvBackend;

  if (!video::internal::Rgb24ToYuyvBackendAvailable(backend))
    return false;

  switch (backend) {
  case Rgb24ToYuyvBackend::scalar:
    video::internal::Rgb24ToYuyvScalar(src, width, height, src_stride, dst,
                                       dst_stride);
    return true;
  case Rgb24ToYuyvBackend::libyuv: {
    const std::size_t scratch_size =
        video::internal::Rgb24ToYuyvLibyuvScratchBytes(width, height);
    std::vector<std::uint8_t> scratch(scratch_size);
    return video::internal::Rgb24ToYuyvLibyuv(src, width, height, src_stride,
                                              dst, dst_stride, scratch.data(),
                                              scratch.size());
  }
  case Rgb24ToYuyvBackend::ssse3:
    video::internal::Rgb24ToYuyvSsse3(src, width, height, src_stride, dst,
                                      dst_stride);
    return true;
  case Rgb24ToYuyvBackend::avx2:
    video::internal::Rgb24ToYuyvAvx2(src, width, height, src_stride, dst,
                                     dst_stride);
    return true;
  }

  return false;
}

bool ConvertWithBackend(video::internal::YuyvToRgbBackend backend,
                        const std::uint8_t *src, int width, int height,
                        std::size_t src_stride, std::uint8_t *dst,
                        std::size_t dst_stride) {
  using video::internal::YuyvToRgbBackend;

  if (!video::internal::YuyvToRgbBackendAvailable(backend))
    return false;

  switch (backend) {
  case YuyvToRgbBackend::scalar:
    video::internal::YuyvToRgbScalar(src, width, height, src_stride, dst,
                                     dst_stride);
    return true;
  case YuyvToRgbBackend::sse41:
    video::internal::YuyvToRgbSse41(src, width, height, src_stride, dst,
                                    dst_stride);
    return true;
  case YuyvToRgbBackend::avx2:
    video::internal::YuyvToRgbAvx2(src, width, height, src_stride, dst,
                                   dst_stride);
    return true;
  }

  return false;
}

bool ConvertRgbBgrWithBackend(video::internal::Rgb24Bgr24Backend backend,
                              const std::uint8_t *src, int width, int height,
                              std::size_t src_stride, std::uint8_t *dst,
                              std::size_t dst_stride) {
  using video::internal::Rgb24Bgr24Backend;

  if (!video::internal::Rgb24Bgr24BackendAvailable(backend))
    return false;

  switch (backend) {
  case Rgb24Bgr24Backend::scalar:
    video::internal::Rgb24Bgr24Scalar(src, dst, width, height, src_stride,
                                      dst_stride);
    return true;
  case Rgb24Bgr24Backend::ssse3:
    video::internal::Rgb24Bgr24Ssse3(src, dst, width, height, src_stride,
                                     dst_stride);
    return true;
  case Rgb24Bgr24Backend::avx2:
    video::internal::Rgb24Bgr24Avx2(src, dst, width, height, src_stride,
                                    dst_stride);
    return true;
  }

  return false;
}

bool CompareYuyvToReference(const std::vector<std::uint8_t> &actual,
                            const std::vector<std::uint8_t> &expected,
                            int width, int height, std::size_t dst_stride,
                            int chroma_tolerance, const std::string &label) {
  const std::size_t active = ActiveYuyvBytes(width);
  for (int y = 0; y < height; ++y) {
    const std::uint8_t *a =
        actual.data() + static_cast<std::size_t>(y) * dst_stride;
    const std::uint8_t *e =
        expected.data() + static_cast<std::size_t>(y) * dst_stride;

    for (int x = 0; x < width; x += 2) {
      const std::size_t out = static_cast<std::size_t>(x) * 2u;
      if (a[out + 0] != e[out + 0] || a[out + 2] != e[out + 2]) {
        std::cerr << label << " luma mismatch at " << x << "," << y << ": "
                  << static_cast<int>(a[out + 0]) << "/"
                  << static_cast<int>(a[out + 2]) << " expected "
                  << static_cast<int>(e[out + 0]) << "/"
                  << static_cast<int>(e[out + 2]) << "\n";
        return false;
      }

      const int u_delta =
          std::abs(static_cast<int>(a[out + 1]) - static_cast<int>(e[out + 1]));
      const int v_delta =
          std::abs(static_cast<int>(a[out + 3]) - static_cast<int>(e[out + 3]));
      if (u_delta > chroma_tolerance || v_delta > chroma_tolerance) {
        std::cerr << label << " chroma mismatch at " << x << "," << y
                  << ": delta U=" << u_delta << " V=" << v_delta
                  << " tolerance=" << chroma_tolerance << "\n";
        return false;
      }
    }

    for (std::size_t i = active; i < dst_stride; ++i) {
      if (a[i] != 0xcdu) {
        std::cerr << label << " overwrote destination padding at row " << y
                  << " byte " << i << "\n";
        return false;
      }
    }
  }

  return true;
}

} // namespace

bool TestYuyvToRgb24MatchesBt601AndPreservesPadding() {
  const std::array<ConvertCase, 9> cases{{
      {1, 3, 5, 8},
      {2, 3, 4, 5},
      {7, 5, 1, 7},
      {8, 5, 8, 3},
      {17, 11, 5, 8},
      {31, 9, 7, 13},
      {640, 480, 13, 16},
      {1280, 720, 7, 32},
      {1920, 1080, 5, 64},
  }};

  for (const ConvertCase &c : cases) {
    const std::size_t src_stride = ActiveYuyvBytes(c.width) + c.src_padding;
    const std::size_t dst_stride = ActiveRgbBytes(c.width) + c.dst_padding;
    std::vector<std::uint8_t> src(src_stride *
                                  static_cast<std::size_t>(c.height));
    std::vector<std::uint8_t> dst(
        dst_stride * static_cast<std::size_t>(c.height), 0xcd);

    std::uint32_t state = static_cast<std::uint32_t>(c.width) * 131u +
                          static_cast<std::uint32_t>(c.height) * 17u + 0x9e37u;
    const std::size_t active_src = ActiveYuyvBytes(c.width);
    for (int y = 0; y < c.height; ++y) {
      std::uint8_t *row = src.data() + static_cast<std::size_t>(y) * src_stride;
      for (std::size_t x = 0; x < active_src; ++x) {
        state = state * 1664525u + 1013904223u;
        row[x] = static_cast<std::uint8_t>((state >> 24) & 0xffu);
      }
      for (std::size_t x = active_src; x < src_stride; ++x)
        row[x] = 0xa5u;
    }

    video::YuyvToRgb24(src.data(), c.width, c.height, src_stride, dst.data(),
                       dst_stride);

    for (int y = 0; y < c.height; ++y) {
      const std::uint8_t *s =
          src.data() + static_cast<std::size_t>(y) * src_stride;
      const std::uint8_t *d =
          dst.data() + static_cast<std::size_t>(y) * dst_stride;

      for (int x = 0; x < c.width; ++x) {
        const std::size_t pair = static_cast<std::size_t>(x / 2) * 4u;
        const int y_sample = s[pair + ((x & 1) ? 2u : 0u)];
        const int u = s[pair + 1u];
        const int v = s[pair + 3u];

        std::uint8_t er = 0;
        std::uint8_t eg = 0;
        std::uint8_t eb = 0;
        YuvToRgbReference(y_sample, u, v, &er, &eg, &eb);

        const std::size_t out = static_cast<std::size_t>(x) * 3u;
        if (d[out + 0] != er || d[out + 1] != eg || d[out + 2] != eb) {
          std::cerr << "YUYV->RGB mismatch at " << x << "," << y << " for "
                    << c.width << "x" << c.height << ": got "
                    << static_cast<int>(d[out + 0]) << ","
                    << static_cast<int>(d[out + 1]) << ","
                    << static_cast<int>(d[out + 2]) << " expected "
                    << static_cast<int>(er) << "," << static_cast<int>(eg)
                    << "," << static_cast<int>(eb) << "\n";
          return false;
        }
      }

      const std::size_t active_dst = ActiveRgbBytes(c.width);
      for (std::size_t i = active_dst; i < dst_stride; ++i) {
        if (d[i] != 0xcdu) {
          std::cerr << "YUYV->RGB overwrote destination padding at row " << y
                    << " byte " << i << " for " << c.width << "x" << c.height
                    << "\n";
          return false;
        }
      }
    }
  }

  return true;
}

bool TestYuyvToRgb24BackendsMatchScalarReference() {
  using video::internal::YuyvToRgbBackend;

  const std::array<ConvertCase, 11> cases{{
      {1, 3, 5, 8},
      {2, 3, 4, 5},
      {7, 5, 1, 7},
      {8, 5, 8, 3},
      {9, 5, 2, 11},
      {15, 4, 5, 6},
      {16, 4, 3, 9},
      {17, 11, 5, 8},
      {640, 480, 13, 16},
      {1280, 720, 7, 32},
      {1920, 1080, 5, 64},
  }};

  const std::array<YuyvToRgbBackendCase, 3> backends{{
      {YuyvToRgbBackend::scalar},
      {YuyvToRgbBackend::sse41},
      {YuyvToRgbBackend::avx2},
  }};

  for (const ConvertCase &c : cases) {
    const std::size_t src_stride = ActiveYuyvBytes(c.width) + c.src_padding;
    const std::size_t dst_stride = ActiveRgbBytes(c.width) + c.dst_padding;
    std::vector<std::uint8_t> src(src_stride *
                                  static_cast<std::size_t>(c.height));

    std::uint32_t state = static_cast<std::uint32_t>(c.width) * 971u +
                          static_cast<std::uint32_t>(c.height) * 37u + 0x1234u;
    const std::size_t active_src = ActiveYuyvBytes(c.width);
    for (int y = 0; y < c.height; ++y) {
      std::uint8_t *row = src.data() + static_cast<std::size_t>(y) * src_stride;
      for (std::size_t x = 0; x < active_src; ++x) {
        state = state * 1664525u + 1013904223u;
        row[x] = static_cast<std::uint8_t>((state >> 24) & 0xffu);
      }
      for (std::size_t x = active_src; x < src_stride; ++x)
        row[x] = 0xa5u;
    }

    std::vector<std::uint8_t> expected(
        dst_stride * static_cast<std::size_t>(c.height), 0xcd);
    video::internal::YuyvToRgbScalar(src.data(), c.width, c.height, src_stride,
                                     expected.data(), dst_stride);

    for (const YuyvToRgbBackendCase &backend : backends) {
      if (!video::internal::YuyvToRgbBackendAvailable(backend.backend))
        continue;

      std::vector<std::uint8_t> actual(
          dst_stride * static_cast<std::size_t>(c.height), 0xcd);
      if (!ConvertWithBackend(backend.backend, src.data(), c.width, c.height,
                              src_stride, actual.data(), dst_stride)) {
        std::cerr << "YUYV->RGB backend "
                  << video::internal::YuyvToRgbBackendName(backend.backend)
                  << " reported available but conversion failed\n";
        return false;
      }

      for (int y = 0; y < c.height; ++y) {
        const std::uint8_t *a =
            actual.data() + static_cast<std::size_t>(y) * dst_stride;
        const std::uint8_t *e =
            expected.data() + static_cast<std::size_t>(y) * dst_stride;
        for (std::size_t i = 0; i < ActiveRgbBytes(c.width); ++i) {
          if (a[i] != e[i]) {
            std::cerr << "YUYV->RGB backend "
                      << video::internal::YuyvToRgbBackendName(backend.backend)
                      << " mismatch at row " << y << " byte " << i << " for "
                      << c.width << "x" << c.height << ": got "
                      << static_cast<int>(a[i]) << " expected "
                      << static_cast<int>(e[i]) << "\n";
            return false;
          }
        }
        for (std::size_t i = ActiveRgbBytes(c.width); i < dst_stride; ++i) {
          if (a[i] != 0xcdu) {
            std::cerr << "YUYV->RGB backend "
                      << video::internal::YuyvToRgbBackendName(backend.backend)
                      << " overwrote destination padding at row " << y
                      << " byte " << i << " for " << c.width << "x" << c.height
                      << "\n";
            return false;
          }
        }
      }
    }
  }

  return true;
}

bool TestRgb24ToYuyvMatchesBt601WithinChromaRounding() {
  constexpr int width = 17;
  constexpr int height = 11;
  constexpr std::size_t src_stride = width * 3 + 5;
  constexpr std::size_t dst_stride = width * 2 + 8;

  std::vector<std::uint8_t> src(src_stride * height);
  std::vector<std::uint8_t> dst(dst_stride * height, 0xcd);

  for (std::size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<std::uint8_t>((i * 37 + (i / 5) * 17 + 91) & 0xff);

  video::Rgb24ToYuyv(src.data(), width, height, src_stride, dst.data(),
                     dst_stride);

  int max_chroma_delta = 0;
  for (int y = 0; y < height; ++y) {
    const std::uint8_t *s =
        src.data() + static_cast<std::size_t>(y) * src_stride;
    const std::uint8_t *d =
        dst.data() + static_cast<std::size_t>(y) * dst_stride;

    for (int x = 0; x < width; x += 2) {
      const int r0 = s[static_cast<std::size_t>(x) * 3u + 0];
      const int g0 = s[static_cast<std::size_t>(x) * 3u + 1];
      const int b0 = s[static_cast<std::size_t>(x) * 3u + 2];

      int r1 = r0;
      int g1 = g0;
      int b1 = b0;
      if (x + 1 < width) {
        r1 = s[static_cast<std::size_t>(x + 1) * 3u + 0];
        g1 = s[static_cast<std::size_t>(x + 1) * 3u + 1];
        b1 = s[static_cast<std::size_t>(x + 1) * 3u + 2];
      }

      const int expected_y0 = RgbToY(r0, g0, b0);
      const int expected_y1 = RgbToY(r1, g1, b1);
      const int expected_u = (RgbToU(r0, g0, b0) + RgbToU(r1, g1, b1)) / 2;
      const int expected_v = (RgbToV(r0, g0, b0) + RgbToV(r1, g1, b1)) / 2;

      const std::size_t out = static_cast<std::size_t>(x) * 2u;
      if (d[out + 0] != expected_y0 || d[out + 2] != expected_y1) {
        std::cerr << "YUYV luma mismatch at " << x << "," << y << "\n";
        return false;
      }

      max_chroma_delta =
          std::max(max_chroma_delta,
                   std::abs(static_cast<int>(d[out + 1]) - expected_u));
      max_chroma_delta =
          std::max(max_chroma_delta,
                   std::abs(static_cast<int>(d[out + 3]) - expected_v));
    }
  }

  if (max_chroma_delta > 1) {
    std::cerr << "YUYV chroma delta exceeded rounding tolerance: "
              << max_chroma_delta << "\n";
    return false;
  }

  return true;
}

bool TestRgb24ToYuyvBackendsMatchScalarReference() {
  using video::internal::Rgb24ToYuyvBackend;

  const std::array<ConvertCase, 11> cases{{
      {1, 3, 5, 8},
      {2, 3, 4, 5},
      {7, 5, 1, 7},
      {8, 5, 8, 3},
      {9, 5, 2, 11},
      {15, 4, 5, 6},
      {16, 4, 3, 9},
      {17, 11, 5, 8},
      {640, 480, 13, 16},
      {1280, 720, 7, 32},
      {1920, 1080, 5, 64},
  }};

  const std::array<BackendCase, 4> backends{{
      {Rgb24ToYuyvBackend::scalar, 0},
      {Rgb24ToYuyvBackend::ssse3, 0},
      {Rgb24ToYuyvBackend::avx2, 0},
      {Rgb24ToYuyvBackend::libyuv, 1},
  }};

  for (const ConvertCase &c : cases) {
    const std::size_t src_stride =
        static_cast<std::size_t>(c.width) * 3u + c.src_padding;
    const std::size_t dst_stride = ActiveYuyvBytes(c.width) + c.dst_padding;
    std::vector<std::uint8_t> src(src_stride *
                                  static_cast<std::size_t>(c.height));
    FillDeterministicRgb(
        &src, c.width, c.height, src_stride,
        static_cast<std::uint32_t>(c.width * 131 + c.height * 17));

    std::vector<std::uint8_t> expected(
        dst_stride * static_cast<std::size_t>(c.height), 0xcd);
    video::internal::Rgb24ToYuyvScalar(src.data(), c.width, c.height,
                                       src_stride, expected.data(), dst_stride);

    for (const BackendCase &backend : backends) {
      if (!video::internal::Rgb24ToYuyvBackendAvailable(backend.backend))
        continue;

      std::vector<std::uint8_t> actual(
          dst_stride * static_cast<std::size_t>(c.height), 0xcd);
      if (!ConvertWithBackend(backend.backend, src.data(), c.width, c.height,
                              src_stride, actual.data(), dst_stride)) {
        std::cerr << "Backend "
                  << video::internal::Rgb24ToYuyvBackendName(backend.backend)
                  << " reported available but conversion failed\n";
        return false;
      }

      const std::string label =
          std::string(
              video::internal::Rgb24ToYuyvBackendName(backend.backend)) +
          " " + std::to_string(c.width) + "x" + std::to_string(c.height);
      if (!CompareYuyvToReference(actual, expected, c.width, c.height,
                                  dst_stride, backend.chroma_tolerance,
                                  label)) {
        return false;
      }
    }
  }

  return true;
}

bool TestRgb24ToYuyvPublicPathMatchesScalarWithScratchVariants() {
  constexpr int width = 31;
  constexpr int height = 9;
  constexpr std::size_t src_stride = width * 3 + 7;
  constexpr std::size_t dst_stride = ActiveYuyvBytes(width) + 13;

  std::vector<std::uint8_t> src(src_stride * height);
  FillDeterministicRgb(&src, width, height, src_stride, 0x5eed1234u);

  std::vector<std::uint8_t> expected(dst_stride * height, 0xcd);
  video::internal::Rgb24ToYuyvScalar(src.data(), width, height, src_stride,
                                     expected.data(), dst_stride);

  {
    std::vector<std::uint8_t> actual(dst_stride * height, 0xcd);
    video::Rgb24ToYuyv(src.data(), width, height, src_stride, actual.data(),
                       dst_stride);
    const int chroma_tolerance =
        video::internal::Rgb24ToYuyvSelectedBackend() ==
                video::internal::Rgb24ToYuyvBackend::libyuv
            ? 1
            : 0;
    if (!CompareYuyvToReference(actual, expected, width, height, dst_stride,
                                chroma_tolerance, "public no-scratch")) {
      return false;
    }
  }

  {
    std::vector<std::uint8_t> actual(dst_stride * height, 0xcd);
    std::array<std::uint8_t, 1> undersized_scratch{0};
    video::Rgb24ToYuyvWithScratch(
        src.data(), width, height, src_stride, actual.data(), dst_stride,
        undersized_scratch.data(), undersized_scratch.size());
    if (!CompareYuyvToReference(actual, expected, width, height, dst_stride, 0,
                                "public undersized scratch")) {
      return false;
    }
  }

  {
    std::vector<std::uint8_t> actual(dst_stride * height, 0xcd);
    const std::size_t scratch_size =
        video::Rgb24ToYuyvScratchBytes(width, height);
    std::vector<std::uint8_t> scratch(scratch_size);
    video::Rgb24ToYuyvWithScratch(
        src.data(), width, height, src_stride, actual.data(), dst_stride,
        scratch.empty() ? nullptr : scratch.data(), scratch.size());
    const int chroma_tolerance =
        video::internal::Rgb24ToYuyvSelectedBackend() ==
                video::internal::Rgb24ToYuyvBackend::libyuv
            ? 1
            : 0;
    if (!CompareYuyvToReference(actual, expected, width, height, dst_stride,
                                chroma_tolerance, "public selected scratch")) {
      return false;
    }
  }

  return true;
}

bool TestRgb24Bgr24BackendsMatchScalarAndPreservePadding() {
  using video::internal::Rgb24Bgr24Backend;

  const std::array<ConvertCase, 10> cases{{
      {1, 3, 5, 8},
      {2, 3, 4, 5},
      {7, 5, 1, 7},
      {15, 4, 5, 6},
      {16, 4, 3, 9},
      {17, 11, 5, 8},
      {31, 9, 7, 13},
      {640, 480, 13, 16},
      {1280, 720, 7, 32},
      {1920, 1080, 5, 64},
  }};

  const std::array<RgbBgrBackendCase, 3> backends{{
      {Rgb24Bgr24Backend::scalar},
      {Rgb24Bgr24Backend::ssse3},
      {Rgb24Bgr24Backend::avx2},
  }};

  for (const ConvertCase &c : cases) {
    const std::size_t src_stride = ActiveRgbBytes(c.width) + c.src_padding;
    const std::size_t dst_stride = ActiveRgbBytes(c.width) + c.dst_padding;
    std::vector<std::uint8_t> src(src_stride *
                                  static_cast<std::size_t>(c.height));
    FillDeterministicRgb(
        &src, c.width, c.height, src_stride,
        static_cast<std::uint32_t>(c.width * 331 + c.height * 29));

    std::vector<std::uint8_t> expected(
        dst_stride * static_cast<std::size_t>(c.height), 0xcd);
    video::internal::Rgb24Bgr24Scalar(src.data(), expected.data(), c.width,
                                      c.height, src_stride, dst_stride);

    for (const RgbBgrBackendCase &backend : backends) {
      if (!video::internal::Rgb24Bgr24BackendAvailable(backend.backend))
        continue;

      std::vector<std::uint8_t> actual(
          dst_stride * static_cast<std::size_t>(c.height), 0xcd);
      if (!ConvertRgbBgrWithBackend(backend.backend, src.data(), c.width,
                                    c.height, src_stride, actual.data(),
                                    dst_stride)) {
        std::cerr << "RGB/BGR backend "
                  << video::internal::Rgb24Bgr24BackendName(backend.backend)
                  << " reported available but conversion failed\n";
        return false;
      }

      for (int y = 0; y < c.height; ++y) {
        const std::uint8_t *a =
            actual.data() + static_cast<std::size_t>(y) * dst_stride;
        const std::uint8_t *e =
            expected.data() + static_cast<std::size_t>(y) * dst_stride;
        for (std::size_t i = 0; i < ActiveRgbBytes(c.width); ++i) {
          if (a[i] != e[i]) {
            std::cerr << "RGB/BGR backend "
                      << video::internal::Rgb24Bgr24BackendName(backend.backend)
                      << " mismatch at row " << y << " byte " << i << " for "
                      << c.width << "x" << c.height << ": got "
                      << static_cast<int>(a[i]) << " expected "
                      << static_cast<int>(e[i]) << "\n";
            return false;
          }
        }
        for (std::size_t i = ActiveRgbBytes(c.width); i < dst_stride; ++i) {
          if (a[i] != 0xcdu) {
            std::cerr << "RGB/BGR backend "
                      << video::internal::Rgb24Bgr24BackendName(backend.backend)
                      << " overwrote destination padding at row " << y
                      << " byte " << i << " for " << c.width << "x" << c.height
                      << "\n";
            return false;
          }
        }
      }
    }
  }

  return true;
}

bool TestRgb24Bgr24PublicPathMatchesScalarInPlace() {
  constexpr int width = 63;
  constexpr int height = 9;
  constexpr std::size_t stride = ActiveRgbBytes(width) + 7;

  std::vector<std::uint8_t> src(stride * height);
  FillDeterministicRgb(&src, width, height, stride, 0x51de00bcu);

  std::vector<std::uint8_t> expected(stride * height, 0xcd);
  video::internal::Rgb24Bgr24Scalar(src.data(), expected.data(), width, height,
                                    stride, stride);

  std::vector<std::uint8_t> out(stride * height, 0xcd);
  video::Rgb24ToBgr24(src.data(), out.data(), width, height, stride, stride);
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (out[i] != expected[i]) {
      std::cerr << "public RGB/BGR conversion mismatch at byte " << i << "\n";
      return false;
    }
  }

  std::vector<std::uint8_t> in_place = src;
  video::Rgb24ToBgr24(in_place.data(), in_place.data(), width, height, stride,
                      stride);
  for (int y = 0; y < height; ++y) {
    const std::uint8_t *a =
        in_place.data() + static_cast<std::size_t>(y) * stride;
    const std::uint8_t *e =
        expected.data() + static_cast<std::size_t>(y) * stride;
    for (std::size_t i = 0; i < ActiveRgbBytes(width); ++i) {
      if (a[i] != e[i]) {
        std::cerr << "in-place RGB/BGR conversion mismatch at row " << y
                  << " byte " << i << "\n";
        return false;
      }
    }
  }

  return true;
}

bool TestResizeRgb24BilinearPreservesActivePixelsAndZerosPadding() {
  constexpr int src_width = 3;
  constexpr int src_height = 2;
  constexpr std::size_t src_stride = ActiveRgbBytes(src_width) + 5u;
  constexpr int dst_width = 5;
  constexpr int dst_height = 4;
  constexpr std::size_t dst_stride = ActiveRgbBytes(dst_width) + 7u;

  std::vector<std::uint8_t> src(src_stride * src_height, 0xa5u);
  FillDeterministicRgb(&src, src_width, src_height, src_stride, 0x12345678u);

  std::vector<std::uint8_t> expected(dst_stride * dst_height, 0xeeu);
  for (int y = 0; y < dst_height; ++y) {
    const float src_y =
        (static_cast<float>(y) + 0.5f) *
            (static_cast<float>(src_height) / static_cast<float>(dst_height)) -
        0.5f;
    const int y0 =
        std::clamp(static_cast<int>(std::floor(src_y)), 0, src_height - 1);
    const int y1 = std::clamp(y0 + 1, 0, src_height - 1);
    const float fy = src_y - static_cast<float>(y0);

    const auto *src_row0 =
        src.data() + static_cast<std::size_t>(y0) * src_stride;
    const auto *src_row1 =
        src.data() + static_cast<std::size_t>(y1) * src_stride;
    auto *dst_row = expected.data() + static_cast<std::size_t>(y) * dst_stride;

    for (int x = 0; x < dst_width; ++x) {
      const float src_x =
          (static_cast<float>(x) + 0.5f) *
              (static_cast<float>(src_width) / static_cast<float>(dst_width)) -
          0.5f;
      const int x0 =
          std::clamp(static_cast<int>(std::floor(src_x)), 0, src_width - 1);
      const int x1 = std::clamp(x0 + 1, 0, src_width - 1);
      const float fx = src_x - static_cast<float>(x0);

      const auto *p00 = src_row0 + static_cast<std::size_t>(x0) * 3u;
      const auto *p10 = src_row0 + static_cast<std::size_t>(x1) * 3u;
      const auto *p01 = src_row1 + static_cast<std::size_t>(x0) * 3u;
      const auto *p11 = src_row1 + static_cast<std::size_t>(x1) * 3u;

      for (int c = 0; c < 3; ++c) {
        const float v0 =
            static_cast<float>(p00[c]) +
            fx * (static_cast<float>(p10[c]) - static_cast<float>(p00[c]));
        const float v1 =
            static_cast<float>(p01[c]) +
            fx * (static_cast<float>(p11[c]) - static_cast<float>(p01[c]));
        const float v = v0 + fy * (v1 - v0);
        const int iv = std::clamp(static_cast<int>(std::lround(v)), 0, 255);
        dst_row[static_cast<std::size_t>(x) * 3u +
                static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(iv);
      }
    }

    std::fill(dst_row + ActiveRgbBytes(dst_width), dst_row + dst_stride,
              std::uint8_t{0});
  }

  std::vector<std::uint8_t> actual(dst_stride * dst_height, 0xeeu);
  std::string err;
  if (!video::ResizeRgb24Bilinear(src.data(), src_width, src_height, src_stride,
                                  dst_width, dst_height, &actual, dst_stride,
                                  &err)) {
    std::cerr << "ResizeRgb24Bilinear failed: " << err << "\n";
    return false;
  }

  if (actual.size() != expected.size()) {
    std::cerr << "ResizeRgb24Bilinear unexpected output size: " << actual.size()
              << " expected " << expected.size() << "\n";
    return false;
  }

  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << "ResizeRgb24Bilinear mismatch at byte " << i << ": got "
                << static_cast<int>(actual[i]) << " expected "
                << static_cast<int>(expected[i]) << "\n";
      return false;
    }
  }

  std::fill(actual.begin(), actual.end(), 0xeeu);
  if (!video::ResizeRgb24Bilinear(src.data(), src_width, src_height, src_stride,
                                  dst_width, dst_height, &actual, dst_stride,
                                  &err)) {
    std::cerr << "ResizeRgb24Bilinear reuse failed: " << err << "\n";
    return false;
  }

  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << "ResizeRgb24Bilinear reuse mismatch at byte " << i << "\n";
      return false;
    }
  }

  video::Rgb24BilinearResizePlan plan;
  if (!plan.Configure(src_width, src_height, dst_width, dst_height, &err)) {
    std::cerr << "Rgb24BilinearResizePlan configure failed: " << err << "\n";
    return false;
  }
  std::fill(actual.begin(), actual.end(), 0xeeu);
  if (!plan.Apply(src.data(), src_stride, &actual, dst_stride, &err)) {
    std::cerr << "Rgb24BilinearResizePlan apply failed: " << err << "\n";
    return false;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << "Rgb24BilinearResizePlan mismatch at byte " << i << "\n";
      return false;
    }
  }

  return true;
}

bool TestResizeRgb24BilinearHandlesDegenerateAxesAndPlanReuse() {
  struct ResizeCase {
    int src_width;
    int src_height;
    int dst_width;
    int dst_height;
  };

  const std::array<ResizeCase, 3> cases{{
      {1, 1, 5, 3},
      {1, 5, 3, 2},
      {7, 1, 2, 4},
  }};

  video::Rgb24BilinearResizePlan plan;
  std::string err;

  for (const ResizeCase &tc : cases) {
    const std::size_t src_stride = ActiveRgbBytes(tc.src_width) + 5u;
    const std::size_t dst_stride = ActiveRgbBytes(tc.dst_width) + 7u;
    std::vector<std::uint8_t> src(src_stride *
                                      static_cast<std::size_t>(tc.src_height),
                                  0xa5u);
    FillDeterministicRgb(&src, tc.src_width, tc.src_height, src_stride,
                         0x2468ace0u + static_cast<std::uint32_t>(
                                            tc.src_width * 31 +
                                            tc.src_height * 17));

    std::vector<std::uint8_t> expected(
        dst_stride * static_cast<std::size_t>(tc.dst_height), 0xeeu);
    for (int y = 0; y < tc.dst_height; ++y) {
      const float src_y =
          (static_cast<float>(y) + 0.5f) *
              (static_cast<float>(tc.src_height) /
               static_cast<float>(tc.dst_height)) -
          0.5f;
      const int y0 =
          std::clamp(static_cast<int>(std::floor(src_y)), 0,
                     tc.src_height - 1);
      const int y1 = std::clamp(y0 + 1, 0, tc.src_height - 1);
      const float fy = src_y - static_cast<float>(y0);

      const auto *src_row0 =
          src.data() + static_cast<std::size_t>(y0) * src_stride;
      const auto *src_row1 =
          src.data() + static_cast<std::size_t>(y1) * src_stride;
      auto *dst_row =
          expected.data() + static_cast<std::size_t>(y) * dst_stride;

      for (int x = 0; x < tc.dst_width; ++x) {
        const float src_x =
            (static_cast<float>(x) + 0.5f) *
                (static_cast<float>(tc.src_width) /
                 static_cast<float>(tc.dst_width)) -
            0.5f;
        const int x0 =
            std::clamp(static_cast<int>(std::floor(src_x)), 0,
                       tc.src_width - 1);
        const int x1 = std::clamp(x0 + 1, 0, tc.src_width - 1);
        const float fx = src_x - static_cast<float>(x0);

        const auto *p00 = src_row0 + static_cast<std::size_t>(x0) * 3u;
        const auto *p10 = src_row0 + static_cast<std::size_t>(x1) * 3u;
        const auto *p01 = src_row1 + static_cast<std::size_t>(x0) * 3u;
        const auto *p11 = src_row1 + static_cast<std::size_t>(x1) * 3u;

        for (int c = 0; c < 3; ++c) {
          const float v0 =
              static_cast<float>(p00[c]) +
              fx * (static_cast<float>(p10[c]) -
                    static_cast<float>(p00[c]));
          const float v1 =
              static_cast<float>(p01[c]) +
              fx * (static_cast<float>(p11[c]) -
                    static_cast<float>(p01[c]));
          const float v = v0 + fy * (v1 - v0);
          const int iv = std::clamp(static_cast<int>(std::lround(v)), 0, 255);
          dst_row[static_cast<std::size_t>(x) * 3u +
                  static_cast<std::size_t>(c)] =
              static_cast<std::uint8_t>(iv);
        }
      }

      std::fill(dst_row + ActiveRgbBytes(tc.dst_width), dst_row + dst_stride,
                std::uint8_t{0});
    }

    std::vector<std::uint8_t> actual(
        dst_stride * static_cast<std::size_t>(tc.dst_height), 0xeeu);
    if (!plan.Configure(tc.src_width, tc.src_height, tc.dst_width,
                        tc.dst_height, &err)) {
      std::cerr << "Rgb24BilinearResizePlan degenerate configure failed: "
                << err << "\n";
      return false;
    }
    if (!plan.Apply(src.data(), src_stride, &actual, dst_stride, &err)) {
      std::cerr << "Rgb24BilinearResizePlan degenerate apply failed: " << err
                << "\n";
      return false;
    }

    if (actual != expected) {
      std::cerr << "Rgb24BilinearResizePlan degenerate mismatch for "
                << tc.src_width << "x" << tc.src_height << " -> "
                << tc.dst_width << "x" << tc.dst_height << "\n";
      return false;
    }
  }

  return true;
}

} // namespace studiocast::tests
