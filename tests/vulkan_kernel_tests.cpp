#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <png.h>

#include "core/video/open_vulkan_auto_frame.h"
#include "core/video/open_vulkan_mirror.h"
#include "core/video/open_vulkan_vignette.h"
#include "core/video/open_vulkan_virtual_background_blur.h"
#include "core/video/open_vulkan_virtual_background_remove.h"
#include "core/video/open_vulkan_virtual_background_replace.h"
#include "core/video/open_vulkan_virtual_key_light.h"
#include "core/vulkan/kernels/utility_kernels.h"
#include "core/vulkan/vulkan_image.h"
#include "core/vulkan/vulkan_tensor.h"

namespace {

bool Require(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

bool RequireVulkanRuntime() {
  const char *value = std::getenv("STUDIOCAST_REQUIRE_VULKAN_RUNTIME");
  return value && std::string(value) == "1";
}

int OptionalSkip(const std::string &reason) {
  std::cout << "[SKIP] Open Vulkan runtime tests: " << reason << "\n";
  return RequireVulkanRuntime() ? 1 : 0;
}

struct ScopedReplaceAssetDirectory {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("studiocast-vulkan-replace-test-" +
       std::to_string(static_cast<unsigned long long>(
           std::chrono::steady_clock::now().time_since_epoch().count())));

  ScopedReplaceAssetDirectory() {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
  }
  ~ScopedReplaceAssetDirectory() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

bool WritePpmFixture(const std::filesystem::path &path, int width, int height,
                     const std::vector<std::uint8_t> &rgb) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out)
    return false;
  out << "P6\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char *>(rgb.data()),
            static_cast<std::streamsize>(rgb.size()));
  return static_cast<bool>(out);
}

bool WritePngFixture(const std::filesystem::path &path, int width, int height,
                     const std::vector<std::uint8_t> &rgb) {
  std::FILE *file = std::fopen(path.c_str(), "wb");
  if (!file)
    return false;
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr,
                                            nullptr, nullptr);
  if (!png) {
    std::fclose(file);
    return false;
  }
  png_infop info = png_create_info_struct(png);
  if (!info) {
    png_destroy_write_struct(&png, nullptr);
    std::fclose(file);
    return false;
  }
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    std::fclose(file);
    return false;
  }
  png_init_io(png, file);
  png_set_IHDR(png, info, static_cast<png_uint_32>(width),
               static_cast<png_uint_32>(height), 8, PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  std::vector<png_bytep> rows(static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    rows[static_cast<std::size_t>(y)] = const_cast<png_bytep>(
        rgb.data() + static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(width) * 3u);
  }
  png_write_image(png, rows.data());
  png_write_end(png, nullptr);
  png_destroy_write_struct(&png, &info);
  std::fclose(file);
  return true;
}

bool WritePngRgbaFixture(const std::filesystem::path &path, int width,
                         int height,
                         const std::vector<std::uint8_t> &rgba) {
  std::FILE *file = std::fopen(path.c_str(), "wb");
  if (!file)
    return false;
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr,
                                            nullptr, nullptr);
  if (!png) {
    std::fclose(file);
    return false;
  }
  png_infop info = png_create_info_struct(png);
  if (!info) {
    png_destroy_write_struct(&png, nullptr);
    std::fclose(file);
    return false;
  }
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    std::fclose(file);
    return false;
  }
  png_init_io(png, file);
  png_set_IHDR(png, info, static_cast<png_uint_32>(width),
               static_cast<png_uint_32>(height), 8, PNG_COLOR_TYPE_RGBA,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  std::vector<png_bytep> rows(static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    rows[static_cast<std::size_t>(y)] = const_cast<png_bytep>(
        rgba.data() + static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(width) * 4u);
  }
  png_write_image(png, rows.data());
  png_write_end(png, nullptr);
  png_destroy_write_struct(&png, &info);
  std::fclose(file);
  return true;
}

studiocast::video::detail::PreparedReplaceBackgroundSource
PreparedReplaceAsset(const std::filesystem::path &path, int generation) {
  studiocast::video::detail::PreparedReplaceBackgroundSource prepared;
  prepared.path = path;
  prepared.mtime = std::filesystem::file_time_type{} +
                   std::chrono::seconds(generation);
  prepared.valid = true;
  return prepared;
}

std::uint8_t RoundClampByte(float v) {
  if (v <= 0.0f)
    return 0;
  if (v >= 255.0f)
    return 255;
  return static_cast<std::uint8_t>(static_cast<int>(v + 0.5f));
}

float ClampAlpha(float v) {
  return studiocast::vulkan::kernels::detail::ClampAlpha01ForComposite(v);
}

std::uint8_t BlendByte(std::uint8_t fg, std::uint8_t bg, float alpha) {
  const float a = ClampAlpha(alpha);
  const float v =
      static_cast<float>(fg) * a + static_cast<float>(bg) * (1.0f - a);
  return RoundClampByte(v);
}

struct AxisSample {
  int i0 = 0;
  int i1 = 0;
  float t = 0.0f;
};

std::size_t PixelIndex(int x, int y, int w) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
         static_cast<std::size_t>(x);
}

std::size_t ChannelIndex(int x, int y, int w, int c) {
  return PixelIndex(x, y, w) * 3u + static_cast<std::size_t>(c);
}

AxisSample U8Sample(int src_len, int dst_len, int dst_i, float crop_pos,
                    float crop_len) {
  const float scale = crop_len / static_cast<float>(dst_len);
  const float src =
      crop_pos + (static_cast<float>(dst_i) + 0.5f) * scale - 0.5f;
  const int i0 = std::clamp(static_cast<int>(src), 0, src_len - 1);
  return {i0, std::clamp(i0 + 1, 0, src_len - 1), src - static_cast<float>(i0)};
}

AxisSample F32ReplicateSample(int src_len, int dst_len, int dst_i) {
  const float scale = static_cast<float>(src_len) / static_cast<float>(dst_len);
  const float src =
      std::clamp((static_cast<float>(dst_i) + 0.5f) * scale - 0.5f, 0.0f,
                 static_cast<float>(src_len - 1));
  const int i0 = static_cast<int>(std::floor(src));
  return {i0, std::clamp(i0 + 1, 0, src_len - 1), src - static_cast<float>(i0)};
}

std::vector<std::uint8_t>
ResizeU8Reference(const std::vector<std::uint8_t> &src, int src_w, int src_h,
                  int dst_w, int dst_h, float crop_x, float crop_y,
                  float crop_w, float crop_h) {
  std::vector<std::uint8_t> dst(static_cast<std::size_t>(dst_w) *
                                static_cast<std::size_t>(dst_h) * 3u);
  for (int y = 0; y < dst_h; ++y) {
    const AxisSample ys = U8Sample(src_h, dst_h, y, crop_y, crop_h);
    for (int x = 0; x < dst_w; ++x) {
      const AxisSample xs = U8Sample(src_w, dst_w, x, crop_x, crop_w);
      for (int c = 0; c < 3; ++c) {
        const auto at = [&](int sx, int sy) {
          return static_cast<float>(src[ChannelIndex(sx, sy, src_w, c)]);
        };
        const float v0 =
            at(xs.i0, ys.i0) + xs.t * (at(xs.i1, ys.i0) - at(xs.i0, ys.i0));
        const float v1 =
            at(xs.i0, ys.i1) + xs.t * (at(xs.i1, ys.i1) - at(xs.i0, ys.i1));
        dst[ChannelIndex(x, y, dst_w, c)] =
            RoundClampByte(v0 + ys.t * (v1 - v0));
      }
    }
  }
  return dst;
}

std::vector<float> ResizeF32Reference(const std::vector<float> &src, int src_w,
                                      int src_h, int dst_w, int dst_h) {
  std::vector<float> dst(static_cast<std::size_t>(dst_w) *
                         static_cast<std::size_t>(dst_h));
  for (int y = 0; y < dst_h; ++y) {
    const AxisSample ys = F32ReplicateSample(src_h, dst_h, y);
    for (int x = 0; x < dst_w; ++x) {
      const AxisSample xs = F32ReplicateSample(src_w, dst_w, x);
      const auto at = [&](int sx, int sy) {
        return src[PixelIndex(sx, sy, src_w)];
      };
      const float v0 =
          at(xs.i0, ys.i0) + xs.t * (at(xs.i1, ys.i0) - at(xs.i0, ys.i0));
      const float v1 =
          at(xs.i0, ys.i1) + xs.t * (at(xs.i1, ys.i1) - at(xs.i0, ys.i1));
      dst[PixelIndex(x, y, dst_w)] = v0 + ys.t * (v1 - v0);
    }
  }
  return dst;
}

std::vector<float> PreprocessReference(
    const std::vector<std::uint8_t> &src, int src_w, int src_h, bool src_bgr,
    const studiocast::vulkan::kernels::ModelPreprocessSpec &spec) {
  std::vector<float> out(3u * static_cast<std::size_t>(spec.dst_w) *
                         static_cast<std::size_t>(spec.dst_h));
  for (int y = 0; y < spec.dst_h; ++y) {
    const AxisSample ys =
        U8Sample(src_h, spec.dst_h, y, 0.0f, static_cast<float>(src_h));
    for (int x = 0; x < spec.dst_w; ++x) {
      const AxisSample xs =
          U8Sample(src_w, spec.dst_w, x, 0.0f, static_cast<float>(src_w));
      float rgb[3] = {};
      for (int semantic = 0; semantic < 3; ++semantic) {
        const int mem_c =
            semantic == 1 ? 1 : (src_bgr ? (2 - semantic) : semantic);
        const auto at = [&](int sx, int sy) {
          return static_cast<float>(src[ChannelIndex(sx, sy, src_w, mem_c)]);
        };
        const float v0 =
            at(xs.i0, ys.i0) + xs.t * (at(xs.i1, ys.i0) - at(xs.i0, ys.i0));
        const float v1 =
            at(xs.i0, ys.i1) + xs.t * (at(xs.i1, ys.i1) - at(xs.i0, ys.i1));
        rgb[semantic] = (v0 + ys.t * (v1 - v0)) / 255.0f;
      }
      const float ordered[3] = {
          spec.dst_order == studiocast::vulkan::kernels::ChannelOrder::bgr
              ? rgb[2]
              : rgb[0],
          rgb[1],
          spec.dst_order == studiocast::vulkan::kernels::ChannelOrder::bgr
              ? rgb[0]
              : rgb[2],
      };
      const std::size_t base = PixelIndex(x, y, spec.dst_w);
      const std::size_t hw = static_cast<std::size_t>(spec.dst_w) *
                             static_cast<std::size_t>(spec.dst_h);
      for (int c = 0; c < 3; ++c)
        out[static_cast<std::size_t>(c) * hw + base] =
            (ordered[c] - spec.mean[c]) / spec.std[c];
    }
  }
  return out;
}

std::vector<std::uint8_t> BlurU8Reference(const std::vector<std::uint8_t> &src,
                                          int w, int h, int radius) {
  radius = std::max(radius, 0);
  const int count = radius * 2 + 1;
  std::vector<std::uint8_t> tmp(src.size());
  std::vector<std::uint8_t> dst(src.size());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < 3; ++c) {
        int sum = 0;
        for (int dx = -radius; dx <= radius; ++dx) {
          const int xx = std::clamp(x + dx, 0, w - 1);
          sum += src[ChannelIndex(xx, y, w, c)];
        }
        tmp[ChannelIndex(x, y, w, c)] =
            static_cast<std::uint8_t>((sum + count / 2) / count);
      }
    }
  }
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < 3; ++c) {
        int sum = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
          const int yy = std::clamp(y + dy, 0, h - 1);
          sum += tmp[ChannelIndex(x, yy, w, c)];
        }
        dst[ChannelIndex(x, y, w, c)] =
            static_cast<std::uint8_t>((sum + count / 2) / count);
      }
    }
  }
  return dst;
}

std::vector<float> BlurF32Reference(const std::vector<float> &src, int w, int h,
                                    int radius) {
  radius = std::max(radius, 0);
  const int count = radius * 2 + 1;
  std::vector<float> tmp(src.size());
  std::vector<float> dst(src.size());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float sum = 0.0f;
      for (int dx = -radius; dx <= radius; ++dx)
        sum += src[PixelIndex(std::clamp(x + dx, 0, w - 1), y, w)];
      tmp[PixelIndex(x, y, w)] = sum / static_cast<float>(count);
    }
  }
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float sum = 0.0f;
      for (int dy = -radius; dy <= radius; ++dy)
        sum += tmp[PixelIndex(x, std::clamp(y + dy, 0, h - 1), w)];
      dst[PixelIndex(x, y, w)] = sum / static_cast<float>(count);
    }
  }
  return dst;
}

std::vector<std::uint8_t>
CompositeReference(const std::vector<std::uint8_t> &fg,
                   const std::vector<std::uint8_t> &bg,
                   const std::vector<float> &alpha) {
  std::vector<std::uint8_t> out(fg.size());
  for (std::size_t i = 0; i < alpha.size(); ++i) {
    for (int c = 0; c < 3; ++c)
      out[i * 3u + static_cast<std::size_t>(c)] =
          BlendByte(fg[i * 3u + static_cast<std::size_t>(c)],
                    bg[i * 3u + static_cast<std::size_t>(c)], alpha[i]);
  }
  return out;
}

std::vector<std::uint8_t>
CompositeSolidReference(const std::vector<std::uint8_t> &fg,
                        const std::vector<float> &alpha,
                        const std::array<std::uint8_t, 3> &bg) {
  std::vector<std::uint8_t> out(fg.size());
  for (std::size_t i = 0; i < alpha.size(); ++i) {
    for (int c = 0; c < 3; ++c)
      out[i * 3u + static_cast<std::size_t>(c)] =
          BlendByte(fg[i * 3u + static_cast<std::size_t>(c)],
                    bg[static_cast<std::size_t>(c)], alpha[i]);
  }
  return out;
}

std::vector<std::uint8_t>
KeyLightReference(const std::vector<std::uint8_t> &src,
                  const std::vector<float> &alpha, int w, int h, float target_r,
                  float target_g, float target_b, float intensity,
                  float direction) {
  std::vector<std::uint8_t> out(src.size());
  target_r = std::clamp(target_r, 0.0f, 255.0f);
  target_g = std::clamp(target_g, 0.0f, 255.0f);
  target_b = std::clamp(target_b, 0.0f, 255.0f);
  intensity = std::clamp(intensity, 0.0f, 1.0f);
  direction = std::clamp(direction, -1.0f, 1.0f);
  const float target[3] = {target_r, target_g, target_b};
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t pix = PixelIndex(x, y, w);
      const float a = ClampAlpha(alpha[pix]);
      if (a <= 0.02f || intensity <= 0.0001f) {
        for (int c = 0; c < 3; ++c)
          out[pix * 3u + static_cast<std::size_t>(c)] =
              src[pix * 3u + static_cast<std::size_t>(c)];
        continue;
      }
      const float cx = static_cast<float>(w) * 0.5f;
      const float inv_cx = cx > 1.0f ? 1.0f / cx : 0.0f;
      const float x_norm = (static_cast<float>(x) - cx) * inv_cx;
      const float field =
          std::clamp(1.0f + direction * x_norm * 0.35f, 0.65f, 1.35f);
      const float t = std::clamp(intensity * a * field, 0.0f, 1.0f);
      for (int c = 0; c < 3; ++c) {
        const auto in =
            static_cast<float>(src[pix * 3u + static_cast<std::size_t>(c)]);
        out[pix * 3u + static_cast<std::size_t>(c)] =
            RoundClampByte(in + (target[c] - in) * t);
      }
    }
  }
  return out;
}

void FillU8(studiocast::vulkan::VulkanImage &image,
            const std::vector<std::uint8_t> &rgb) {
  studiocast::vulkan::PackRgb24ToRgba32(
      rgb.data(), static_cast<std::size_t>(image.width()) * 3u, image.width(),
      image.height(), static_cast<std::uint32_t *>(image.mapped()),
      image.pitch_pixels());
}

std::vector<std::uint8_t> ReadU8(studiocast::vulkan::VulkanImage &image) {
  std::vector<std::uint8_t> out(static_cast<std::size_t>(image.width()) *
                                static_cast<std::size_t>(image.height()) * 3u);
  studiocast::vulkan::UnpackRgba32ToRgb24(
      static_cast<const std::uint32_t *>(image.mapped()), image.pitch_pixels(),
      image.width(), image.height(), out.data(),
      static_cast<std::size_t>(image.width()) * 3u);
  return out;
}

void FillF32(studiocast::vulkan::VulkanImage &image,
             const std::vector<float> &values) {
  std::copy(values.begin(), values.end(), static_cast<float *>(image.mapped()));
}

std::vector<float> ReadF32(studiocast::vulkan::VulkanImage &image) {
  const auto *ptr = static_cast<const float *>(image.mapped());
  return {ptr, ptr + static_cast<std::size_t>(image.width()) *
                         static_cast<std::size_t>(image.height())};
}

std::vector<float> ReadTensor(studiocast::vulkan::VulkanTensor &tensor) {
  const auto *ptr = static_cast<const float *>(tensor.mapped());
  return {ptr, ptr + tensor.ElementCount()};
}

bool AlmostEqual(float a, float b, float tol = 1.0e-4f) {
  return std::abs(a - b) <= tol;
}

bool CompareU8(const std::vector<std::uint8_t> &actual,
               const std::vector<std::uint8_t> &expected,
               const std::string &what) {
  bool ok = true;
  ok &= Require(actual.size() == expected.size(), what + ": size mismatch");
  for (std::size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
    const int diff =
        std::abs(static_cast<int>(actual[i]) - static_cast<int>(expected[i]));
    if (diff > 1) {
      std::cerr << what << ": byte mismatch at " << i << " got "
                << static_cast<int>(actual[i]) << " expected "
                << static_cast<int>(expected[i]) << "\n";
      return false;
    }
  }
  return ok;
}

bool CompareU8Exact(const std::vector<std::uint8_t> &actual,
                    const std::vector<std::uint8_t> &expected,
                    const std::string &what) {
  if (actual == expected)
    return true;
  std::size_t mismatch = 0;
  while (mismatch < actual.size() && mismatch < expected.size() &&
         actual[mismatch] == expected[mismatch]) {
    ++mismatch;
  }
  std::cerr << what << ": exact byte mismatch at " << mismatch << "\n";
  return false;
}

std::vector<std::uint8_t>
MirrorU8Reference(const std::vector<std::uint8_t> &src, int width, int height) {
  std::vector<std::uint8_t> out(src.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        out[ChannelIndex(x, y, width, c)] =
            src[ChannelIndex(width - 1 - x, y, width, c)];
      }
    }
  }
  return out;
}

float CudaVignetteRadialSquaredReference(int x, int y, int width, int height,
                                         float center_x, float center_y) {
  const float inv_half_w = 2.0f / static_cast<float>(width);
  const float inv_half_h = 2.0f / static_cast<float>(height);
  const float fx = (static_cast<float>(x) + 0.5f - center_x) * inv_half_w;
  const float fy = (static_cast<float>(y) + 0.5f - center_y) * inv_half_h;
  float radius = std::sqrt(fx * fx + fy * fy) * 0.70710677f;
  radius = std::clamp(radius, 0.0f, 1.0f);
  return radius * radius;
}

std::vector<std::uint8_t>
VignetteU8ReferencePadded(const std::vector<std::uint8_t> &src, int width,
                          int height, std::size_t stride,
                          int intensity_percent) {
  std::vector<std::uint8_t> out = src;
  const float intensity =
      static_cast<float>(std::clamp(intensity_percent, 0, 100)) / 100.0f;
  const float center_x = static_cast<float>(width) * 0.5f;
  const float center_y = static_cast<float>(height) * 0.5f;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float radial_squared = CudaVignetteRadialSquaredReference(
          x, y, width, height, center_x, center_y);
      const float factor = std::max(0.0f, 1.0f - intensity * radial_squared);
      const std::size_t base = static_cast<std::size_t>(y) * stride +
                               static_cast<std::size_t>(x) * 3u;
      for (int c = 0; c < 3; ++c) {
        out[base + static_cast<std::size_t>(c)] = RoundClampByte(
            static_cast<float>(src[base + static_cast<std::size_t>(c)]) *
            factor);
      }
    }
  }
  return out;
}

bool CompareF32(const std::vector<float> &actual,
                const std::vector<float> &expected, const std::string &what) {
  bool ok = true;
  ok &= Require(actual.size() == expected.size(), what + ": size mismatch");
  for (std::size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
    if (!AlmostEqual(actual[i], expected[i])) {
      std::cerr << what << ": float mismatch at " << i << " got " << actual[i]
                << " expected " << expected[i] << "\n";
      return false;
    }
  }
  return ok;
}

bool AllocateU8(studiocast::vulkan::kernels::UtilityKernels &kernels,
                studiocast::vulkan::VulkanImage *image, int w, int h,
                studiocast::vulkan::VulkanPixelFormat format,
                std::string *error) {
  return image->Allocate(kernels.device(), w, h, format, /*map_memory=*/true,
                         error);
}

bool AllocateF32(studiocast::vulkan::kernels::UtilityKernels &kernels,
                 studiocast::vulkan::VulkanImage *image, int w, int h,
                 std::string *error) {
  return image->Allocate(kernels.device(), w, h,
                         studiocast::vulkan::VulkanPixelFormat::f32_1,
                         /*map_memory=*/true, error);
}

studiocast::open_vulkan::VulkanMattingReadiness
SyntheticProductionMattingReadiness(
    studiocast::vulkan::kernels::UtilityKernels &kernels,
    bool current_frame_inference) {
  studiocast::open_vulkan::VulkanMattingReadiness readiness;
  readiness.production_ready = true;
  readiness.reason_code =
      studiocast::open_vulkan::kOpenVulkanMattingUnavailableReason;
  auto &runtime = readiness.runtime;
  runtime.runtime_name = "synthetic-test-seam";
  runtime.adapter_available = true;
  runtime.production_adapter = true;
  runtime.runtime_created = true;
  runtime.graph_loaded = true;
  runtime.persistent_resources_allocated = true;
  runtime.warmup_complete = true;
  runtime.device_identity_matches = true;
  runtime.input_device_resident = true;
  runtime.alpha_device_resident = true;
  runtime.output_device_resident = true;
  runtime.shared_device_imported = true;
  runtime.queue_ownership_explicit = true;
  runtime.synchronous_completion = true;
  runtime.bounded_reusable_allocations = true;
  runtime.persistent_allocation_count = 2;
  runtime.warmup_inference_count = 1;
  runtime.inference_count = current_frame_inference ? 1 : 0;
  runtime.completion_count = current_frame_inference ? 2 : 1;
  runtime.active_device.ownership_domain = kernels.device();
  runtime.active_device.logical_device =
      reinterpret_cast<std::uintptr_t>(kernels.device()->device());
  runtime.active_device.queue =
      reinterpret_cast<std::uintptr_t>(kernels.device()->queue());
  const auto identity = kernels.device()->context_identity();
  runtime.active_device.context_id = identity.context_id;
  runtime.active_device.context_generation = identity.generation;
  runtime.active_device.non_cpu_device_selected = true;
  runtime.active_device.compute_queue_available = true;
  runtime.active_device.context_healthy = true;
  return readiness;
}

studiocast::vulkan::VulkanDeviceCandidateInfo
DeviceCandidate(std::uint32_t index, std::uint32_t type, int score,
                int compute_queue, const char *name) {
  studiocast::vulkan::VulkanDeviceCandidateInfo candidate;
  candidate.enumeration_index = index;
  candidate.device_type = type;
  candidate.score = score;
  candidate.compute_queue_family_index = compute_queue;
  candidate.device_name = name;
  return candidate;
}

bool TestVulkanDeviceSelectionPolicy() {
  using studiocast::vulkan::VK_PHYSICAL_DEVICE_TYPE_CPU;
  using studiocast::vulkan::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  using studiocast::vulkan::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
  using studiocast::vulkan::VulkanDeviceCandidateInfo;
  using studiocast::vulkan::VulkanDeviceSelection;

  bool ok = true;
  std::string error;
  VulkanDeviceSelection selection;
  ok &= Require(studiocast::vulkan::detail::ParseVulkanDeviceSelection(
                    "", "", &selection, &error),
                "default Vulkan device selection should parse: " + error);
  ok &= Require(
      !selection.requested_index.has_value() && selection.request == "auto" &&
          selection.source == "automatic" && !selection.allow_cpu_in_auto,
      "default Vulkan device selection should be hardware auto");

  ok &= Require(studiocast::vulkan::detail::ParseVulkanDeviceSelection(
                    " 2 ", "true", &selection, &error),
                "explicit Vulkan device selection should parse: " + error);
  ok &= Require(selection.requested_index == 2 &&
                    selection.request == "index:2" &&
                    selection.source == "STUDIOCAST_VULKAN_DEVICE_INDEX",
                "explicit Vulkan device selection should retain its source");
  ok &= Require(!studiocast::vulkan::detail::ParseVulkanDeviceSelection(
                    "-1", "", &selection, &error) &&
                    error.find("non-negative") != std::string::npos,
                "negative Vulkan device indices should fail clearly");
  ok &= Require(!studiocast::vulkan::detail::ParseVulkanDeviceSelection(
                    "", "sometimes", &selection, &error) &&
                    error.find("boolean") != std::string::npos,
                "invalid Vulkan CPU opt-in values should fail clearly");

  std::vector<VulkanDeviceCandidateInfo> candidates = {
      DeviceCandidate(0, VK_PHYSICAL_DEVICE_TYPE_CPU, 100, 0, "llvmpipe"),
      DeviceCandidate(1, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 300, 1,
                      "Intel GPU"),
      DeviceCandidate(2, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 400, 2,
                      "AMD GPU"),
      DeviceCandidate(3, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 400, -1,
                      "No compute GPU"),
  };
  selection = VulkanDeviceSelection{};
  auto selected = studiocast::vulkan::detail::SelectVulkanDeviceCandidate(
      &candidates, selection);
  ok &= Require(selected.ok && selected.candidate_vector_index == 2 &&
                    candidates[2].selected,
                "Vulkan auto selection should prefer a discrete GPU");
  ok &= Require(!candidates[0].eligible &&
                    candidates[0].rejection_reason == "cpu_device_not_enabled",
                "Vulkan auto selection should reject CPU drivers by default");
  ok &= Require(!candidates[3].eligible &&
                    candidates[3].rejection_reason == "no_compute_queue",
                "Vulkan candidates should diagnose missing compute queues");

  std::vector<VulkanDeviceCandidateInfo> no_candidates;
  auto failed = studiocast::vulkan::detail::SelectVulkanDeviceCandidate(
      &no_candidates, VulkanDeviceSelection{});
  ok &= Require(!failed.ok &&
                    failed.failure_reason == "vulkan_no_physical_device",
                "empty Vulkan enumeration must retain its stable reason code");

  auto no_compute_candidates =
      std::vector<VulkanDeviceCandidateInfo>{DeviceCandidate(
          0, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 400, -1, "No compute GPU")};
  failed = studiocast::vulkan::detail::SelectVulkanDeviceCandidate(
      &no_compute_candidates, VulkanDeviceSelection{});
  ok &=
      Require(!failed.ok && failed.failure_reason == "vulkan_no_compute_queue",
              "no-compute Vulkan selection must retain its stable reason "
              "code");
  VulkanDeviceSelection requested_no_compute;
  requested_no_compute.requested_index = 0;
  failed = studiocast::vulkan::detail::SelectVulkanDeviceCandidate(
      &no_compute_candidates, requested_no_compute);
  ok &= Require(
      !failed.ok &&
          failed.failure_reason == "vulkan_requested_device_no_compute_queue",
      "explicit no-compute Vulkan selection must retain its stable reason "
      "code");

  selection.requested_index = 0;
  selection.request = "index:0";
  selection.source = "STUDIOCAST_VULKAN_DEVICE_INDEX";
  selected = studiocast::vulkan::detail::SelectVulkanDeviceCandidate(
      &candidates, selection);
  ok &= Require(selected.ok && selected.candidate_vector_index == 0 &&
                    candidates[0].selected && candidates[0].eligible,
                "an explicitly selected CPU Vulkan device should be allowed");
  ok &= Require(candidates[2].rejection_reason == "not_requested",
                "explicit Vulkan selection should identify other candidates");

  candidates = {
      DeviceCandidate(0, VK_PHYSICAL_DEVICE_TYPE_CPU, 100, 0, "lavapipe")};
  selection = VulkanDeviceSelection{};
  selected = studiocast::vulkan::detail::SelectVulkanDeviceCandidate(
      &candidates, selection);
  ok &= Require(
      !selected.ok &&
          selected.failure_reason == "vulkan_only_cpu_devices_available" &&
          selected.error.find("STUDIOCAST_VULKAN_ALLOW_CPU=1") !=
              std::string::npos,
      "CPU-only Vulkan auto selection should require explicit opt-in");
  selection.allow_cpu_in_auto = true;
  selected = studiocast::vulkan::detail::SelectVulkanDeviceCandidate(
      &candidates, selection);
  ok &= Require(selected.ok && candidates[0].selected,
                "Vulkan CPU auto opt-in should select a software device");

  candidates = {DeviceCandidate(1, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 300,
                                0, "Intel GPU")};
  selection = VulkanDeviceSelection{};
  selection.requested_index = 7;
  selected = studiocast::vulkan::detail::SelectVulkanDeviceCandidate(
      &candidates, selection);
  ok &= Require(!selected.ok &&
                    selected.failure_reason ==
                        "vulkan_requested_device_not_found" &&
                    selected.error.find("index 7") != std::string::npos,
                "missing explicit Vulkan indices should fail clearly");

  const std::string intelStableId =
      studiocast::vulkan::detail::MakeVulkanDeviceStableId(
          0x8086, 0x1234, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
          "Intel Arc Integrated");
  ok &=
      Require(intelStableId == "v1:8086:1234:1:intel-arc-integrated" &&
                  studiocast::vulkan::detail::IsValidVulkanDeviceStableId(
                      intelStableId),
              "stable Vulkan identity should be deterministic and config-safe");
  candidates = {DeviceCandidate(7, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 400, 0,
                                "AMD GPU"),
                DeviceCandidate(2, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 300,
                                0, "Intel Arc Integrated")};
  candidates[1].vendor_id = 0x8086;
  candidates[1].device_id = 0x1234;
  selection = VulkanDeviceSelection{};
  selection.requested_stable_id = intelStableId;
  selection.request = "stable_id:" + intelStableId;
  selection.source = "daemon_config";
  selected = studiocast::vulkan::detail::SelectVulkanDeviceCandidate(
      &candidates, selection);
  ok &= Require(selected.ok && selected.candidate_vector_index == 1 &&
                    candidates[1].selected,
                "stable Vulkan identity should select independently of the "
                "run-local enumeration index");

  selection.requested_stable_id = "v1:1002:ffff:2:missing-amd";
  selected = studiocast::vulkan::detail::SelectVulkanDeviceCandidate(
      &candidates, selection);
  ok &= Require(!selected.ok && selected.failure_reason ==
                                    "vulkan_requested_device_not_found",
                "a missing saved Vulkan identity must fail closed");

  candidates = {DeviceCandidate(0, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 300,
                                0, "Identical GPU"),
                DeviceCandidate(1, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 300,
                                0, "Identical GPU")};
  const std::string ambiguousId =
      studiocast::vulkan::detail::MakeVulkanDeviceStableId(
          0, 0, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, "Identical GPU");
  selection.requested_stable_id = ambiguousId;
  selected = studiocast::vulkan::detail::SelectVulkanDeviceCandidate(
      &candidates, selection);
  ok &= Require(!selected.ok && selected.failure_reason ==
                                    "vulkan_requested_device_ambiguous",
                "indistinguishable saved Vulkan identities must fail closed");

  studiocast::vulkan::OpenVulkanDiagnostics diagnostics;
  diagnostics.device_selection_source = "STUDIOCAST_VULKAN_DEVICE_INDEX";
  diagnostics.device_selection_request = "index:1";
  diagnostics.selected_device_index = 1;
  diagnostics.selected_device_stable_id = intelStableId;
  diagnostics.device_candidates = {DeviceCandidate(
      1, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 300, 0, "Intel GPU")};
  diagnostics.device_candidates[0].stable_id = intelStableId;
  diagnostics.device_candidates[0].selected = true;
  const std::string json = diagnostics.ToJson();
  ok &= Require(
      json.find("\"device_selection_request\":\"index:1\"") !=
              std::string::npos &&
          json.find("\"selected_device_index\":1") != std::string::npos &&
          json.find("\"selected_device_stable_id\":\"" + intelStableId +
                    "\"") != std::string::npos &&
          json.find("\"device_candidates\":[{") != std::string::npos &&
          json.find("\"device_name\":\"Intel GPU\"") != std::string::npos,
      "Vulkan diagnostics JSON should expose selection candidates");
  return ok;
}

} // namespace

int main() {
  using studiocast::vulkan::VulkanImage;
  using studiocast::vulkan::VulkanPixelFormat;
  using studiocast::vulkan::VulkanTensor;
  using studiocast::vulkan::kernels::ChannelOrder;
  using studiocast::vulkan::kernels::ModelPreprocessSpec;
  using studiocast::vulkan::kernels::UtilityKernels;

  bool ok = true;
  std::string error;

  ok &= TestVulkanDeviceSelectionPolicy();
  {
    const auto diagnostics =
        studiocast::vulkan::DiagnoseOpenVulkanDefault();
    const auto blocked = diagnostics.blocked_effects.find("eye_contact");
    const bool falsely_available =
        std::find(diagnostics.available_effects.begin(),
                  diagnostics.available_effects.end(), "eye_contact") !=
        diagnostics.available_effects.end();
    const std::string json = diagnostics.ToJson();
    ok &= Require(
        blocked != diagnostics.blocked_effects.end() &&
            blocked->second == "open_vulkan_eye_contact_unavailable" &&
            !falsely_available &&
            !diagnostics.eye_contact_production_ready &&
            diagnostics.eye_contact_backend_compiled &&
            !diagnostics.eye_contact_live_stage_implemented &&
            !diagnostics.eye_contact_production_adapter_available &&
            !diagnostics.eye_contact_vulkan_inference_provider_available &&
            diagnostics.eye_contact_non_cpu_device_selected ==
                diagnostics.non_cpu_device_selected &&
            diagnostics.eye_contact_compute_queue_available ==
                diagnostics.compute_queue_available &&
            diagnostics.eye_contact_context_healthy ==
                diagnostics.context_healthy &&
            !diagnostics.eye_contact_shared_device_imported &&
            !diagnostics.eye_contact_queue_ownership_explicit &&
            !diagnostics.eye_contact_model_pack_selected &&
            !diagnostics.eye_contact_artifact_contract_validated &&
            !diagnostics.eye_contact_selectable_cpu_fallback &&
            diagnostics.eye_contact_dispatch_count == 0 &&
            diagnostics.eye_contact_cpu_readback_count == 0 &&
            diagnostics.eye_contact_cpu_fallback_count == 0,
        "default diagnostics must keep Vulkan eye contact unavailable with "
        "separate hardware/live/provider/artifact/fallback facts");
    ok &= Require(
        diagnostics.eye_contact_blocker_code ==
                "open_vulkan_eye_contact_runtime_unavailable" &&
            json.find("\"eye_contact_production_ready\":false") !=
                std::string::npos &&
            json.find(
                std::string("\"eye_contact_non_cpu_device_selected\":") +
                (diagnostics.non_cpu_device_selected ? "true" : "false")) !=
                std::string::npos &&
            json.find(
                std::string("\"eye_contact_compute_queue_available\":") +
                (diagnostics.compute_queue_available ? "true" : "false")) !=
                std::string::npos &&
            json.find(std::string("\"eye_contact_context_healthy\":") +
                      (diagnostics.context_healthy ? "true" : "false")) !=
                std::string::npos &&
            json.find("\"eye_contact_dispatch_count\":0") !=
                std::string::npos &&
            json.find("\"eye_contact_cpu_readback_count\":0") !=
                std::string::npos &&
            json.find("\"eye_contact_cpu_fallback_count\":0") !=
                std::string::npos,
        "serialized eye-contact diagnostics must mirror hardware facts and "
        "expose the primary blocker with zero frame work");
  }
  ok &= Require(
      !studiocast::video::OpenVulkanAutoFrameSequenceNeedsReset(0, 7) &&
          !studiocast::video::OpenVulkanAutoFrameSequenceNeedsReset(7, 7) &&
          !studiocast::video::OpenVulkanAutoFrameSequenceNeedsReset(7, 8) &&
          studiocast::video::OpenVulkanAutoFrameSequenceNeedsReset(7, 9) &&
          studiocast::video::OpenVulkanAutoFrameSequenceNeedsReset(9, 8),
      "Auto Frame temporal state must reset deterministically on forward gaps "
      "and sequence restarts, but not first/consecutive/duplicate frames");
  {
    using ReuseKey = studiocast::video::OpenVulkanAutoFrameReuseKey;
    using ResetReason = studiocast::video::OpenVulkanAutoFrameResetReason;
    const auto reset_reason =
        studiocast::video::OpenVulkanAutoFrameReuseKeyResetReason;
    const ReuseKey unobserved{};
    const ReuseKey base{/*observed=*/true,
                        /*enabled=*/true,
                        /*effects_generation=*/41,
                        /*frame_width=*/1280,
                        /*frame_height=*/720,
                        /*model_id=*/"yunet-320x320",
                        /*provider_id=*/"onnxruntime-cpu",
                        /*context_identity=*/{17, 3}};
    ok &= Require(reset_reason(unobserved, base) == ResetReason::none &&
                      reset_reason(base, base) == ResetReason::none,
                  "Auto Frame must initialize or reuse an identical temporal "
                  "configuration without a spurious reset");

    ReuseKey changed = base;
    changed.provider_id = "ncnn-vulkan";
    ok &= Require(reset_reason(base, changed) ==
                      ResetReason::geometry_model_or_provider_change,
                  "Auto Frame provider transitions must reset temporal state");
    changed = base;
    changed.model_id = "yunet-fixed-640x640";
    ok &= Require(reset_reason(base, changed) ==
                      ResetReason::geometry_model_or_provider_change,
                  "Auto Frame model transitions must reset temporal state");
    changed = base;
    changed.frame_width = 1920;
    ok &= Require(reset_reason(base, changed) ==
                      ResetReason::geometry_model_or_provider_change,
                  "Auto Frame geometry transitions must reset temporal state");
    changed = base;
    changed.effects_generation = 42;
    ok &= Require(reset_reason(base, changed) ==
                      ResetReason::enablement_or_effect_generation,
                  "Auto Frame effect generations must reset temporal state");
    changed = base;
    changed.context_identity.generation = 4;
    ok &= Require(reset_reason(base, changed) ==
                      ResetReason::vulkan_context_generation_change,
                  "Auto Frame context generation transitions must reset "
                  "temporal state");
    changed = base;
    changed.enabled = false;
    ok &= Require(reset_reason(base, changed) ==
                          ResetReason::enablement_or_effect_generation &&
                      reset_reason(changed, base) ==
                          ResetReason::enablement_or_effect_generation,
                  "Auto Frame enable and disable transitions must explicitly "
                  "reset temporal state");
  }

  UtilityKernels validation_only;
  VulkanImage invalid;
  ok &= Require(!validation_only.ResizeBilinear(invalid, invalid, &error),
                "Vulkan resize wrapper should reject invalid images before "
                "runtime init");
  ok &= Require(error.find("invalid Vulkan image") != std::string::npos,
                "Vulkan invalid image error should be clear");
  ok &= Require(
      !validation_only.MirrorHorizontalU8x3(invalid, invalid, &error) &&
          error.find("invalid Vulkan image") != std::string::npos,
      "Vulkan mirror wrapper should reject invalid images before runtime init");
  {
    studiocast::video::OpenVulkanMirror invalid_mirror;
    ok &= Require(!invalid_mirror.EnsureInitialized(nullptr, 1, 1, &error) &&
                      error.find("vulkan_effect_initialization_failed") !=
                          std::string::npos,
                  "mirror initialization failures need a stable reason code");
  }
  {
    using studiocast::video::ResolveOpenVulkanVirtualBackgroundBlurParameters;
    const auto below =
        ResolveOpenVulkanVirtualBackgroundBlurParameters(-10);
    const auto one = ResolveOpenVulkanVirtualBackgroundBlurParameters(1);
    const auto default_strength =
        ResolveOpenVulkanVirtualBackgroundBlurParameters(8);
    const auto fifteen =
        ResolveOpenVulkanVirtualBackgroundBlurParameters(15);
    const auto sixteen =
        ResolveOpenVulkanVirtualBackgroundBlurParameters(16);
    const auto sixty_three =
        ResolveOpenVulkanVirtualBackgroundBlurParameters(63);
    const auto sixty_four =
        ResolveOpenVulkanVirtualBackgroundBlurParameters(64);
    const auto above =
        ResolveOpenVulkanVirtualBackgroundBlurParameters(100);
    ok &= Require(
        below.background_radius == 1 && below.alpha_feather_radius == 0 &&
            one.background_radius == 1 && one.alpha_feather_radius == 0 &&
            default_strength.background_radius == 8 &&
            default_strength.alpha_feather_radius == 0 &&
            fifteen.background_radius == 15 &&
            fifteen.alpha_feather_radius == 0 &&
            sixteen.background_radius == 16 &&
            sixteen.alpha_feather_radius == 1 &&
            sixty_three.background_radius == 63 &&
            sixty_three.alpha_feather_radius == 3 &&
            sixty_four.background_radius == 64 &&
            sixty_four.alpha_feather_radius == 4 &&
            above.background_radius == 64 &&
            above.alpha_feather_radius == 4,
        "Vulkan blur strength and feather mapping must match canonical Open "
        "CUDA endpoint/default semantics");

    studiocast::video::OpenVulkanVirtualBackgroundBlur invalid_blur;
    studiocast::open_vulkan::VulkanMattingReadiness unavailable_matting;
    unavailable_matting.reason_code =
        studiocast::open_vulkan::kOpenVulkanMattingUnavailableReason;
    unavailable_matting.blocker_code =
        studiocast::open_vulkan::kOpenVulkanMattingAdapterUnavailableReason;
    unavailable_matting.detail = "reviewed shared-device adapter is absent";
    ok &= Require(
        !invalid_blur.EnsureInitialized(nullptr, 1, 1, unavailable_matting,
                                        &error) &&
            error.find(
                "vulkan_virtual_background_blur_initialization_failed") !=
                std::string::npos,
        "blur initialization failures need an effect-specific stable outer "
        "reason");

    studiocast::vulkan::OpenVulkanDiagnostics fake_hardware;
    fake_hardware.compiled_enabled = true;
    fake_hardware.runtime_library_found = true;
    fake_hardware.physical_device_found = true;
    fake_hardware.non_cpu_device_selected = true;
    fake_hardware.compute_queue_available = true;
    fake_hardware.logical_device_created = true;
    fake_hardware.context_created = true;
    fake_hardware.context_healthy = true;
    fake_hardware.production_hardware_ready = true;
    fake_hardware.shader_pipeline_created = true;
    fake_hardware.ok = true;
    const auto blocked = studiocast::video::
        EvaluateOpenVulkanVirtualBackgroundBlurReadiness(fake_hardware,
                                                         unavailable_matting);
    ok &= Require(!blocked.production_ready &&
                      blocked.reason_code ==
                          studiocast::open_vulkan::
                              kOpenVulkanMattingUnavailableReason &&
                      blocked.detail.find(
                          "open_vulkan_matting_adapter_unavailable") !=
                          std::string::npos,
                  "utility/synthetic-alpha evidence alone must not make blur "
                  "available and must retain the exact matting blocker");
  }
  {
    using studiocast::video::
        ResolveOpenVulkanVirtualBackgroundRemoveParameters;
    const auto below =
        ResolveOpenVulkanVirtualBackgroundRemoveParameters(-10, "#12aBcD");
    const auto fifteen =
        ResolveOpenVulkanVirtualBackgroundRemoveParameters(15, "0X112233");
    const auto sixteen =
        ResolveOpenVulkanVirtualBackgroundRemoveParameters(16, "a1B2c3");
    const auto sixty_four =
        ResolveOpenVulkanVirtualBackgroundRemoveParameters(64, "#ffffff");
    const auto above =
        ResolveOpenVulkanVirtualBackgroundRemoveParameters(100, "0x010203");
    const auto malformed =
        ResolveOpenVulkanVirtualBackgroundRemoveParameters(8, "#xyzxyz");
    ok &= Require(
        below.alpha_feather_radius == 0 && below.background_r == 0x12 &&
            below.background_g == 0xab && below.background_b == 0xcd &&
            fifteen.alpha_feather_radius == 0 &&
            fifteen.background_r == 0x11 && fifteen.background_g == 0x22 &&
            fifteen.background_b == 0x33 &&
            sixteen.alpha_feather_radius == 1 &&
            sixteen.background_r == 0xa1 && sixteen.background_g == 0xb2 &&
            sixteen.background_b == 0xc3 &&
            sixty_four.alpha_feather_radius == 4 &&
            above.alpha_feather_radius == 4 && above.background_r == 1 &&
            above.background_g == 2 && above.background_b == 3 &&
            malformed.background_r == 0 && malformed.background_g == 0 &&
            malformed.background_b == 0,
        "Vulkan remove must preserve strength-as-feather-only endpoints, "
        "compatible hex forms, semantic RGB order, and invalid-color black "
        "fallback");

    studiocast::video::OpenVulkanVirtualBackgroundRemove invalid_remove;
    studiocast::open_vulkan::VulkanMattingReadiness unavailable_matting;
    unavailable_matting.reason_code =
        studiocast::open_vulkan::kOpenVulkanMattingUnavailableReason;
    unavailable_matting.blocker_code =
        studiocast::open_vulkan::kOpenVulkanMattingAdapterUnavailableReason;
    unavailable_matting.detail = "reviewed shared-device adapter is absent";
    ok &= Require(
        !invalid_remove.EnsureInitialized(nullptr, 1, 1, 8, "#000000",
                                          unavailable_matting, &error) &&
            error.find(
                "vulkan_virtual_background_remove_initialization_failed") !=
                std::string::npos,
        "remove initialization failures need an effect-specific stable outer "
        "reason");

    studiocast::vulkan::OpenVulkanDiagnostics fake_hardware;
    fake_hardware.compiled_enabled = true;
    fake_hardware.runtime_library_found = true;
    fake_hardware.physical_device_found = true;
    fake_hardware.non_cpu_device_selected = true;
    fake_hardware.compute_queue_available = true;
    fake_hardware.logical_device_created = true;
    fake_hardware.context_created = true;
    fake_hardware.context_healthy = true;
    fake_hardware.production_hardware_ready = true;
    fake_hardware.shader_pipeline_created = true;
    fake_hardware.ok = true;
    const auto blocked = studiocast::video::
        EvaluateOpenVulkanVirtualBackgroundRemoveReadiness(fake_hardware,
                                                           unavailable_matting);
    ok &= Require(!blocked.production_ready &&
                      blocked.reason_code ==
                          studiocast::open_vulkan::
                              kOpenVulkanMattingUnavailableReason &&
                      blocked.detail.find(
                          "open_vulkan_matting_adapter_unavailable") !=
                          std::string::npos,
                  "utility/synthetic-alpha evidence alone must not make "
                  "remove available and must retain the exact matting "
                  "blocker");
  }
  {
    using studiocast::video::
        ResolveOpenVulkanVirtualBackgroundReplaceParameters;
    const auto below =
        ResolveOpenVulkanVirtualBackgroundReplaceParameters(-10);
    const auto fifteen =
        ResolveOpenVulkanVirtualBackgroundReplaceParameters(15);
    const auto sixteen =
        ResolveOpenVulkanVirtualBackgroundReplaceParameters(16);
    const auto sixty_four =
        ResolveOpenVulkanVirtualBackgroundReplaceParameters(64);
    const auto above =
        ResolveOpenVulkanVirtualBackgroundReplaceParameters(100);
    ok &= Require(below.alpha_feather_radius == 0 &&
                      fifteen.alpha_feather_radius == 0 &&
                      sixteen.alpha_feather_radius == 1 &&
                      sixty_four.alpha_feather_radius == 4 &&
                      above.alpha_feather_radius == 4,
                  "Vulkan replace must preserve Open CUDA strength-as-alpha-"
                  "feather-only endpoint semantics");

    studiocast::video::OpenVulkanVirtualBackgroundReplace invalid_replace;
    studiocast::video::OpenVulkanVirtualBackgroundReplaceCounters counters;
    studiocast::open_vulkan::VulkanMattingReadiness unavailable_matting;
    unavailable_matting.reason_code =
        studiocast::open_vulkan::kOpenVulkanMattingUnavailableReason;
    unavailable_matting.blocker_code =
        studiocast::open_vulkan::kOpenVulkanMattingAdapterUnavailableReason;
    unavailable_matting.detail = "reviewed shared-device adapter is absent";
    studiocast::video::detail::PreparedReplaceBackgroundSource prepared;
    ok &= Require(
        !invalid_replace.EnsureInitialized(nullptr, 1, 1, 8, prepared,
                                           unavailable_matting, &counters,
                                           &error) &&
            error.find(
                "vulkan_virtual_background_replace_initialization_failed") !=
                std::string::npos,
        "replace initialization failures need an effect-specific stable outer "
        "reason");

    studiocast::vulkan::OpenVulkanDiagnostics fake_hardware;
    fake_hardware.compiled_enabled = true;
    fake_hardware.runtime_library_found = true;
    fake_hardware.physical_device_found = true;
    fake_hardware.non_cpu_device_selected = true;
    fake_hardware.compute_queue_available = true;
    fake_hardware.logical_device_created = true;
    fake_hardware.context_created = true;
    fake_hardware.context_healthy = true;
    fake_hardware.production_hardware_ready = true;
    fake_hardware.shader_pipeline_created = true;
    fake_hardware.ok = true;
    const auto blocked = studiocast::video::
        EvaluateOpenVulkanVirtualBackgroundReplaceReadiness(
            fake_hardware, unavailable_matting);
    ok &= Require(!blocked.production_ready &&
                      blocked.reason_code ==
                          studiocast::open_vulkan::
                              kOpenVulkanMattingUnavailableReason &&
                      blocked.detail.find(
                          "open_vulkan_matting_adapter_unavailable") !=
                          std::string::npos,
                  "utility/asset/synthetic-alpha evidence alone must not make "
                  "replace available and must retain the exact matting "
                  "blocker");
  }
  {
    using studiocast::video::ResolveOpenVulkanVirtualKeyLightParameters;
    const auto off = ResolveOpenVulkanVirtualKeyLightParameters(-5, 0, -180);
    const auto neutral = ResolveOpenVulkanVirtualKeyLightParameters(70, 99, 0);
    const auto warm = ResolveOpenVulkanVirtualKeyLightParameters(50, 1, -90);
    const auto cool = ResolveOpenVulkanVirtualKeyLightParameters(150, 2, 180);
    ok &= Require(
        off.passthrough() && off.intensity == 0.0f && off.target_r == 255.0f &&
            off.target_g == 255.0f && off.target_b == 255.0f &&
            off.direction == -1.0f && neutral.intensity == 0.7f &&
            neutral.target_r == 255.0f && neutral.target_g == 255.0f &&
            neutral.target_b == 255.0f && neutral.direction == 0.0f &&
            warm.intensity == 0.5f && warm.target_r == 255.0f &&
            warm.target_g == 242.0f && warm.target_b == 228.0f &&
            warm.direction == -1.0f && cool.intensity == 1.0f &&
            cool.target_r == 228.0f && cool.target_g == 242.0f &&
            cool.target_b == 255.0f && cool.direction == 1.0f,
        "Vulkan key light must preserve Open CUDA intensity, preset, and "
        "pan-saturation endpoint semantics");

    studiocast::video::OpenVulkanVirtualKeyLight invalid_key_light;
    studiocast::open_vulkan::VulkanMattingReadiness unavailable_matting;
    unavailable_matting.reason_code =
        studiocast::open_vulkan::kOpenVulkanMattingUnavailableReason;
    unavailable_matting.blocker_code =
        studiocast::open_vulkan::kOpenVulkanMattingAdapterUnavailableReason;
    unavailable_matting.detail = "reviewed shared-device adapter is absent";
    ok &= Require(
        !invalid_key_light.EnsureInitialized(nullptr, 1, 1, 70, 0, 0,
                                             unavailable_matting, &error) &&
            error.find("vulkan_virtual_key_light_initialization_failed") !=
                std::string::npos,
        "key-light initialization failures need an effect-specific stable "
        "outer reason");

    studiocast::vulkan::OpenVulkanDiagnostics fake_hardware;
    fake_hardware.compiled_enabled = true;
    fake_hardware.runtime_library_found = true;
    fake_hardware.physical_device_found = true;
    fake_hardware.non_cpu_device_selected = true;
    fake_hardware.compute_queue_available = true;
    fake_hardware.logical_device_created = true;
    fake_hardware.context_created = true;
    fake_hardware.context_healthy = true;
    fake_hardware.production_hardware_ready = true;
    fake_hardware.shader_pipeline_created = true;
    fake_hardware.ok = true;
    const auto blocked =
        studiocast::video::EvaluateOpenVulkanVirtualKeyLightReadiness(
            fake_hardware, unavailable_matting);
    ok &= Require(
        !blocked.production_ready &&
            blocked.reason_code ==
                studiocast::open_vulkan::kOpenVulkanMattingUnavailableReason &&
            blocked.detail.find("open_vulkan_matting_adapter_unavailable") !=
                std::string::npos,
        "utility/synthetic-alpha evidence alone must not make key light "
        "available and must retain the exact matting blocker");

    using studiocast::video::detail::
        IsOpenVulkanVirtualKeyLightSameFrameArtifactCompatible;
    studiocast::open_video::FrameMatteArtifactKey key;
    key.provider_id = "open_vulkan";
    key.model_id = "test-matting";
    key.storage = studiocast::open_video::FrameMatteStorage::device_f32_alpha;
    key.frame_width = 1280;
    key.frame_height = 720;
    key.matte_width = 256;
    key.matte_height = 144;
    key.device_context = 0xabc;
    key.stream = 0xdef;
    studiocast::open_video::FrameMatteArtifact artifact;
    artifact.key = key;
    artifact.handle = 0x123;
    artifact.aux_handle = 0x456;
    const auto compatible = [&](std::uint64_t capture_sequence = 41,
                                std::uint64_t cached_matte_sequence = 41,
                                std::uint64_t cached_alpha_sequence = 41,
                                std::string_view active_model =
                                    "test-matting") {
      return IsOpenVulkanVirtualKeyLightSameFrameArtifactCompatible(
          capture_sequence, cached_matte_sequence, cached_alpha_sequence, key,
          &artifact, active_model, 1280, 720, 256, 144, 0xabc, 0xdef, 0x123,
          0x456);
    };
    ok &= Require(compatible(),
                  "exact same-frame key-light matte artifact should be "
                  "accepted");
    artifact.handle = 0x999;
    ok &= Require(!compatible(),
                  "tampered key-light matte object handle must be rejected");
    artifact.handle = 0x123;
    artifact.aux_handle = 0x999;
    ok &= Require(!compatible(),
                  "tampered key-light matte buffer handle must be rejected");
    artifact.aux_handle = 0x456;
    ok &= Require(!compatible(41, 40, 41) && !compatible(41, 41, 40),
                  "stale matte or resized-alpha sequence must be rejected");
    artifact.key.model_id = "tampered-key";
    ok &= Require(!compatible(), "artifact/key mismatch must be rejected");
    artifact.key = key;
    key.device_context = 0xbeef;
    artifact.key = key;
    ok &= Require(!compatible(),
                  "wrong key-light Vulkan device context must be rejected");
    key.device_context = 0xabc;
    key.stream = 0xbeef;
    artifact.key = key;
    ok &=
        Require(!compatible(), "wrong key-light Vulkan queue must be rejected");
    key.stream = 0xdef;
    key.frame_width = 640;
    artifact.key = key;
    ok &= Require(!compatible(),
                  "wrong key-light frame geometry must be rejected");
    key.frame_width = 1280;
    key.matte_height = 256;
    artifact.key = key;
    ok &= Require(!compatible(),
                  "wrong key-light matte geometry must be rejected");
    key.matte_height = 144;
    artifact.key = key;
    ok &= Require(!compatible(41, 41, 41, "other-model"),
                  "wrong active key-light matting model must be rejected");
  }
  {
    studiocast::video::OpenVulkanAutoFrame invalid_auto_frame;
    studiocast::video::OpenVulkanAutoFrameCounters counters;
    ok &= Require(
        !invalid_auto_frame.EnsureInitialized(nullptr, 1, 1, &counters,
                                              &error) &&
            error.find("vulkan_effect_initialization_failed") !=
                std::string::npos &&
            counters.initialization_failures == 1,
        "Auto Frame initialization failures need a stable reason and counter");
    invalid_auto_frame.ResetTemporal(
        studiocast::video::OpenVulkanAutoFrameResetReason::
            enablement_or_effect_generation,
        &counters);
    ok &= Require(counters.temporal_reset_calls == 1 &&
                      counters.cpu_resize_fallback_calls == 0,
                  "Auto Frame reset accounting must not invent a CPU resize "
                  "fallback");
  }
  {
    studiocast::video::OpenVulkanVignette invalid_vignette;
    studiocast::video::OpenVulkanVignetteCounters counters;
    ok &= Require(!invalid_vignette.EnsureInitialized(nullptr, 1, 1, 50,
                                                      &counters, &error) &&
                      error.find("vulkan_effect_initialization_failed") !=
                          std::string::npos,
                  "vignette initialization failures need a stable reason code");

    const struct RadialCase {
      int x;
      int y;
      int width;
      int height;
      float center_x;
      float center_y;
    } radial_cases[] = {
        {0, 0, 1, 1, 0.5f, 0.5f},
        {0, 0, 5, 3, 2.5f, 1.5f},
        {4, 2, 5, 3, 2.5f, 1.5f},
        {5, 3, 6, 4, 3.0f, 2.0f},
        // Off-center coordinates prove the CUDA radius clamp at 1.0 rather
        // than relying only on fixed-center production corners (<1.0).
        {4, 4, 5, 5, 0.0f, 0.0f},
    };
    for (const auto &tc : radial_cases) {
      const float actual =
          studiocast::video::detail::OpenVulkanVignetteRadialSquaredAt(
              tc.x, tc.y, tc.width, tc.height, tc.center_x, tc.center_y);
      const float expected = CudaVignetteRadialSquaredReference(
          tc.x, tc.y, tc.width, tc.height, tc.center_x, tc.center_y);
      ok &= Require(actual == expected,
                    "vignette radial mask must preserve the CUDA coordinate "
                    "normalization and clamp operation order");
    }
    ok &= Require(
        studiocast::video::detail::OpenVulkanVignetteRadialSquaredAt(
            4, 4, 5, 5, 0.0f, 0.0f) == 1.0f,
        "off-center vignette radius beyond one must clamp before squaring");

    const int padded_w = 5;
    const int padded_h = 3;
    const std::size_t padded_stride = 19;
    std::vector<std::uint8_t> padded(padded_stride * padded_h, 0xcd);
    for (int y = 0; y < padded_h; ++y) {
      for (int x = 0; x < padded_w * 3; ++x) {
        padded[static_cast<std::size_t>(y) * padded_stride +
               static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>((y * 53 + x * 17) & 0xff);
      }
    }
    const auto padded_out = VignetteU8ReferencePadded(
        padded, padded_w, padded_h, padded_stride, 100);
    bool padding_preserved = true;
    for (int y = 0; y < padded_h; ++y) {
      for (std::size_t x = static_cast<std::size_t>(padded_w) * 3u;
           x < padded_stride; ++x) {
        padding_preserved &=
            padded_out[static_cast<std::size_t>(y) * padded_stride + x] == 0xcd;
      }
    }
    ok &= Require(padding_preserved,
                  "padded CPU vignette parity reference must leave row "
                  "padding untouched");
  }
  {
    studiocast::vulkan::OpenVulkanDiagnostics diagnostics;
    diagnostics.runtime_library_found = true;
    diagnostics.physical_device_found = true;
    diagnostics.non_cpu_device_selected = true;
    diagnostics.compute_queue_available = true;
    diagnostics.logical_device_created = true;
    diagnostics.context_created = true;
    diagnostics.context_healthy = true;
    diagnostics.production_hardware_ready = true;
    diagnostics.shader_pipeline_created = true;
    diagnostics.ok = true;
    ok &= Require(
        studiocast::video::EvaluateOpenVulkanMirrorReadiness(diagnostics)
            .production_ready,
        "mirror readiness should accept complete production hardware facts");

    diagnostics.cpu_device_selected = true;
    diagnostics.non_cpu_device_selected = false;
    diagnostics.production_hardware_ready = false;
    const auto cpu_readiness =
        studiocast::video::EvaluateOpenVulkanMirrorReadiness(diagnostics);
    ok &= Require(!cpu_readiness.production_ready &&
                      cpu_readiness.shared_reason_code ==
                          "vulkan_only_cpu_devices_available",
                  "CPU/software Vulkan must fail the mirror production "
                  "readiness predicate with a stable shared reason");

    diagnostics.cpu_device_selected = false;
    diagnostics.non_cpu_device_selected = true;
    diagnostics.production_hardware_ready = false;
    const auto unproven_readiness =
        studiocast::video::EvaluateOpenVulkanMirrorReadiness(diagnostics);
    ok &= Require(!unproven_readiness.production_ready &&
                      unproven_readiness.shared_reason_code ==
                          "vulkan_production_hardware_not_ready",
                  "mirror readiness must fail closed when the aggregate "
                  "production hardware fact is not proven");

    diagnostics.production_hardware_ready = true;
    ok &= Require(
        studiocast::video::EvaluateOpenVulkanVignetteReadiness(diagnostics)
            .production_ready,
        "vignette readiness should accept complete production hardware facts");
    diagnostics.cpu_device_selected = true;
    diagnostics.non_cpu_device_selected = false;
    diagnostics.production_hardware_ready = false;
    const auto vignette_cpu_readiness =
        studiocast::video::EvaluateOpenVulkanVignetteReadiness(diagnostics);
    ok &= Require(!vignette_cpu_readiness.production_ready &&
                      vignette_cpu_readiness.shared_reason_code ==
                          "vulkan_only_cpu_devices_available",
                  "CPU/software Vulkan must fail vignette readiness with the "
                  "stable shared reason");
    diagnostics.cpu_device_selected = false;
    diagnostics.non_cpu_device_selected = true;
    diagnostics.production_hardware_ready = true;
    diagnostics.context_healthy = false;
    diagnostics.context_failure_reason = "vulkan_device_lost";
    const auto vignette_unhealthy_readiness =
        studiocast::video::EvaluateOpenVulkanVignetteReadiness(diagnostics);
    ok &= Require(!vignette_unhealthy_readiness.production_ready &&
                      vignette_unhealthy_readiness.shared_reason_code ==
                          "vulkan_device_lost",
                  "an unhealthy reusable context must fail vignette readiness "
                  "with its shared stable reason");
  }
  ok &=
      Require(studiocast::vulkan::kernels::detail::CheckBoxBlurRadiusForKernel(
                  64, "test", &error),
              "Vulkan box blur should accept radius 64");
  ok &=
      Require(!studiocast::vulkan::kernels::detail::CheckBoxBlurRadiusForKernel(
                  65, "test", &error),
              "Vulkan box blur should reject radius above 64");
  ok &= Require(error.find("maximum supported radius 64") != std::string::npos,
                "Vulkan radius error should name the limit");
  studiocast::vulkan::VulkanTensorSize size;
  ok &=
      Require(studiocast::vulkan::CheckedNchwF32Size(1, 3, 2, 2, &size, &error),
              "Vulkan tensor valid shape should pass");
  ok &= Require(size.elements == 12 && size.bytes == 12 * sizeof(float),
                "Vulkan tensor size should be contiguous NCHW f32");
  ok &= Require(
      !studiocast::vulkan::CheckedNchwF32Size(1, 0, 2, 2, &size, &error),
      "Vulkan tensor invalid shape should fail");
  if (!ok)
    return 1;

  UtilityKernels kernels;
  if (!kernels.Initialize(&error))
    return OptionalSkip(error);

  {
    const auto diagnostics = kernels.Diagnostics();
    ok &=
        Require(diagnostics.runtime_library_found &&
                    diagnostics.physical_device_found &&
                    diagnostics.compute_queue_available &&
                    diagnostics.logical_device_created &&
                    diagnostics.context_created && diagnostics.context_healthy,
                "Vulkan capability facts should distinguish a healthy "
                "logical context from loader presence");
    ok &= Require(diagnostics.production_hardware_ready ==
                      !diagnostics.cpu_device_selected,
                  "CPU Vulkan devices must never count as production hardware "
                  "readiness");
    ok &= Require(kernels.device()->context_identity().Valid(),
                  "initialized Vulkan device should expose context/generation "
                  "identity");
  }

  {
    UtilityKernels upload_kernels;
    if (!upload_kernels.Initialize(&error))
      return OptionalSkip("replacement upload Vulkan context unavailable: " +
                          error);
    constexpr int width = 2;
    constexpr int height = 2;
    constexpr std::size_t stride = 8;
    const std::vector<std::uint8_t> padded_rgb = {
        1,  2,  3,  4,  5,  6,  0xee, 0xef,
        11, 12, 13, 14, 15, 16, 0xfe, 0xff,
    };
    const std::vector<std::uint8_t> expected = {
        1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16,
    };
    VulkanImage staging, resident, mapped_readback;
    if (!staging.Allocate(upload_kernels.device(), width, height,
                          VulkanPixelFormat::rgb_u8, true, &error) ||
        !resident.Allocate(upload_kernels.device(), width, height,
                           VulkanPixelFormat::rgb_u8, false, &error) ||
        !mapped_readback.Allocate(upload_kernels.device(), width, height,
                                  VulkanPixelFormat::rgb_u8, true, &error)) {
      return OptionalSkip("replacement upload allocation failed: " + error);
    }
    const auto submissions_before =
        upload_kernels.synchronous_submission_count();
    ok &= Require(upload_kernels.UploadRgb24ToDeviceLocal(
                      padded_rgb.data(), stride, staging, resident, &error),
                  "setup-only RGB24 device upload should dispatch: " + error);
    ok &= Require(upload_kernels.synchronous_submission_count() ==
                      submissions_before + 1 &&
                      staging.mapped() && staging.host_visible() &&
                      !resident.mapped() && resident.device_local(),
                  "replacement upload must use one synchronized transfer from "
                  "mapped staging to non-mapped DEVICE_LOCAL RGB");
    upload_kernels.InvalidateDescriptorBindingCacheForSetup();
    ok &= Require(upload_kernels.ResizeBilinear(resident, mapped_readback,
                                                &error) &&
                      mapped_readback.Invalidate(&error),
                  "uploaded replacement verification copy failed: " + error);
    ok &= CompareU8Exact(ReadU8(mapped_readback), expected,
                         "setup-only padded RGB24 upload");

    const auto rejected_before =
        upload_kernels.synchronous_submission_count();
    ok &= Require(!upload_kernels.UploadRgb24ToDeviceLocal(
                      padded_rgb.data(), 5, staging, resident, &error) &&
                      error.find("tight-or-padded RGB24") !=
                          std::string::npos &&
                      upload_kernels.synchronous_submission_count() ==
                          rejected_before,
                  "replacement upload must reject a short RGB24 stride before "
                  "submission");

    VulkanImage bgr_staging, bgr_resident;
    if (!bgr_staging.Allocate(upload_kernels.device(), width, height,
                              VulkanPixelFormat::bgr_u8, true, &error) ||
        !bgr_resident.Allocate(upload_kernels.device(), width, height,
                               VulkanPixelFormat::bgr_u8, false, &error)) {
      return OptionalSkip("replacement BGR rejection allocation failed: " +
                          error);
    }
    ok &= Require(!upload_kernels.UploadRgb24ToDeviceLocal(
                      expected.data(), width * 3u, bgr_staging, bgr_resident,
                      &error) &&
                      error.find("rgb_u8") != std::string::npos &&
                      upload_kernels.synchronous_submission_count() ==
                          rejected_before,
                  "RGB24 upload must reject BGR staging/destination without "
                  "silently reinterpreting channel order");

    UtilityKernels foreign_upload_kernels;
    if (!foreign_upload_kernels.Initialize(&error))
      return OptionalSkip("foreign replacement upload context unavailable: " +
                          error);
    VulkanImage foreign_resident;
    if (!foreign_resident.Allocate(foreign_upload_kernels.device(), width,
                                   height, VulkanPixelFormat::rgb_u8, false,
                                   &error)) {
      return OptionalSkip("foreign replacement upload allocation failed: " +
                          error);
    }
    ok &= Require(!upload_kernels.UploadRgb24ToDeviceLocal(
                      expected.data(), width * 3u, staging, foreign_resident,
                      &error) &&
                      error.find("foreign") != std::string::npos &&
                      upload_kernels.synchronous_submission_count() ==
                          rejected_before,
                  "replacement upload must reject a foreign destination before "
                  "submission");
  }

  {
    UtilityKernels lost_upload_kernels;
    if (!lost_upload_kernels.Initialize(&error))
      return OptionalSkip("replacement upload loss context unavailable: " +
                          error);
    VulkanImage staging, resident;
    if (!staging.Allocate(lost_upload_kernels.device(), 1, 1,
                          VulkanPixelFormat::rgb_u8, true, &error) ||
        !resident.Allocate(lost_upload_kernels.device(), 1, 1,
                           VulkanPixelFormat::rgb_u8, false, &error)) {
      return OptionalSkip("replacement upload loss allocation failed: " +
                          error);
    }
    const std::array<std::uint8_t, 3> rgb = {7, 8, 9};
    lost_upload_kernels.device()->InjectNextSubmissionResultForTesting(
        studiocast::vulkan::VulkanSubmissionPhase::queue_submit,
        studiocast::vulkan::VK_ERROR_DEVICE_LOST);
    ok &= Require(!lost_upload_kernels.UploadRgb24ToDeviceLocal(
                      rgb.data(), 3, staging, resident, &error) &&
                      error.find("[vulkan_device_lost]") != std::string::npos,
                  "replacement upload device loss must preserve the shared "
                  "stable reason");
    const auto submitted_after_loss =
        lost_upload_kernels.device()->health().submitted_serial;
    ok &= Require(!lost_upload_kernels.UploadRgb24ToDeviceLocal(
                      rgb.data(), 3, staging, resident, &error) &&
                      lost_upload_kernels.device()->health().submitted_serial ==
                          submitted_after_loss,
                  "poisoned replacement upload context must reject without "
                  "another driver submission");
  }

  {
    using studiocast::video::OpenVulkanVirtualKeyLight;
    using studiocast::video::OpenVulkanVirtualKeyLightCounters;
    using studiocast::video::OpenVulkanVirtualKeyLightInput;
    using studiocast::video::OpenVulkanVirtualKeyLightMatteSource;

    UtilityKernels key_light_kernels;
    if (!key_light_kernels.Initialize(&error))
      return OptionalSkip("key-light Vulkan context unavailable: " + error);
    constexpr int w = 4;
    constexpr int h = 2;
    const std::vector<std::uint8_t> foreground = {
        10, 20, 30, 100, 110, 120, 200, 210, 220, 50, 60, 70,
        90, 80, 70, 40,  50,  60,  130, 140, 150, 5,  15, 25,
    };
    const std::vector<float> alpha = {0.0f,  0.5f,  1.0f, 0.01f,
                                      0.25f, 0.75f, 0.9f, 0.1f};
    VulkanImage gpu_foreground, gpu_output, alpha_upload, resident_alpha;
    if (!AllocateU8(key_light_kernels, &gpu_foreground, w, h,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(key_light_kernels, &gpu_output, w, h,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateF32(key_light_kernels, &alpha_upload, w, h, &error) ||
        !resident_alpha.Allocate(key_light_kernels.device(), w, h,
                                 VulkanPixelFormat::f32_1,
                                 /*map_memory=*/false, &error)) {
      return OptionalSkip("production key-light allocation failed: " + error);
    }
    FillU8(gpu_foreground, foreground);
    FillF32(alpha_upload, alpha);
    if (!gpu_foreground.Flush(&error) || !alpha_upload.Flush(&error) ||
        !key_light_kernels.ResizeBilinearF32_1(alpha_upload, resident_alpha,
                                               &error)) {
      return OptionalSkip("production key-light fixture setup failed: " +
                          error);
    }
    ok &= Require(!resident_alpha.mapped() && resident_alpha.device_local() &&
                      gpu_output.mapped(),
                  "production key light must keep alpha non-mapped "
                  "DEVICE_LOCAL and use only the mapped RGB transport");

    auto setup_readiness =
        SyntheticProductionMattingReadiness(key_light_kernels, false);
    OpenVulkanVirtualKeyLight key_light;
    ok &= Require(key_light.EnsureInitialized(&key_light_kernels, w, h, 50, 1,
                                              90, setup_readiness, &error),
                  "synthetic test-seam key-light initialization failed: " +
                      error);

    OpenVulkanVirtualKeyLightInput input;
    input.foreground = &gpu_foreground;
    input.alpha = &resident_alpha;
    input.output = &gpu_output;
    input.capture_sequence = 81;
    input.resident_alpha_sequence = 81;
    input.alpha_resize_completion_count_before = 0;
    input.alpha_resize_completion_count_after = 1;
    input.matting_inference_count_before = 0;
    input.matte_source =
        OpenVulkanVirtualKeyLightMatteSource::independently_inferred;
    input.matting_readiness = &setup_readiness;
    OpenVulkanVirtualKeyLightCounters missing_frame_evidence;
    ok &= Require(!key_light.Apply(input, &missing_frame_evidence, &error) &&
                      error.find("vulkan_virtual_key_light_runtime_failed") !=
                          std::string::npos &&
                      error.find("current-frame resident alpha") !=
                          std::string::npos &&
                      missing_frame_evidence.dispatch_calls == 0,
                  "warmup/utility availability must not replace current-frame "
                  "key-light matting evidence");

    auto current_readiness =
        SyntheticProductionMattingReadiness(key_light_kernels, true);
    input.matting_readiness = &current_readiness;
    OpenVulkanVirtualKeyLightCounters counters;
    const auto allocations_before =
        key_light_kernels.device()->allocation_stats().allocation_count;
    const auto submissions_before =
        key_light_kernels.synchronous_submission_count();
    ok &= Require(key_light.Apply(input, &counters, &error) &&
                      gpu_output.Invalidate(&error),
                  "independent production key-light dispatch failed: " + error);
    ok &= CompareU8Exact(ReadU8(gpu_output),
                         KeyLightReference(foreground, alpha, w, h, 255.0f,
                                           242.0f, 228.0f, 0.5f, 1.0f),
                         "production Vulkan key-light Open CUDA parity");

    input.capture_sequence = 82;
    input.resident_alpha_sequence = 82;
    input.alpha_resize_completion_count_before = 1;
    input.alpha_resize_completion_count_after = 1;
    input.matting_inference_count_before = 1;
    input.matte_source =
        OpenVulkanVirtualKeyLightMatteSource::reused_same_frame;
    ok &= Require(key_light.Apply(input, &counters, &error),
                  "same-frame matte reuse key-light dispatch failed: " + error);
    ok &= Require(
        counters.dispatch_calls == 2 &&
            counters.independent_matte_inference_calls == 1 &&
            counters.shared_matte_reuse_calls == 1 &&
            counters.alpha_readback_calls == 0 &&
            counters.cpu_fallback_calls == 0 &&
            counters.runtime_failure_frames == 0 &&
            key_light_kernels.synchronous_submission_count() ==
                submissions_before + 2 &&
            key_light_kernels.device()->allocation_stats().allocation_count ==
                allocations_before,
        "key-light inference/reuse accounting must be exact, allocation-free, "
        "and prove zero readback/CPU fallback");

    const auto before_bad_reuse =
        key_light_kernels.synchronous_submission_count();
    input.matting_inference_count_before = 0;
    ok &= Require(
        !key_light.Apply(input, &counters, &error) &&
            error.find("reuse unexpectedly performed inference") !=
                std::string::npos &&
            key_light_kernels.synchronous_submission_count() ==
                before_bad_reuse,
        "key light must reject a false same-frame reuse classification before "
        "dispatch");

    ok &= Require(key_light.EnsureInitialized(&key_light_kernels, w, h, 0, 2,
                                              -180, setup_readiness, &error),
                  "zero-intensity key-light setup should retain production "
                  "readiness: " +
                      error);
    input.capture_sequence = 0;
    input.resident_alpha_sequence = 0;
    input.matting_readiness = &setup_readiness;
    const auto before_passthrough =
        key_light_kernels.synchronous_submission_count();
    ok &= Require(key_light.Apply(input, &counters, &error) &&
                      counters.passthrough_frames == 1 &&
                      counters.dispatch_calls == 2 &&
                      key_light_kernels.synchronous_submission_count() ==
                          before_passthrough,
                  "zero-intensity key light must pass through without "
                  "inference evidence or a Vulkan dispatch");

    ok &= Require(key_light.EnsureInitialized(&key_light_kernels, w, h, 50, 1,
                                              90, current_readiness, &error),
                  "key-light reinitialization failed: " + error);
    input.capture_sequence = 83;
    input.resident_alpha_sequence = 83;
    input.alpha_resize_completion_count_before = 1;
    input.alpha_resize_completion_count_after = 1;
    input.matting_inference_count_before = 1;
    input.matte_source =
        OpenVulkanVirtualKeyLightMatteSource::reused_same_frame;
    input.matting_readiness = &current_readiness;
    UtilityKernels foreign_kernels;
    if (!foreign_kernels.Initialize(&error))
      return OptionalSkip("foreign key-light context unavailable: " + error);
    VulkanImage foreign_output;
    if (!AllocateU8(foreign_kernels, &foreign_output, w, h,
                    VulkanPixelFormat::rgb_u8, &error)) {
      return OptionalSkip("foreign key-light output allocation failed: " +
                          error);
    }
    input.output = &foreign_output;
    const auto before_foreign =
        key_light_kernels.synchronous_submission_count();
    ok &= Require(
        !key_light.Apply(input, &counters, &error) &&
            error.find("vulkan_virtual_key_light_runtime_failed") !=
                std::string::npos &&
            error.find("[vulkan_foreign_context]") != std::string::npos &&
            key_light_kernels.synchronous_submission_count() == before_foreign,
        "key light must reject a foreign output before dispatch with nested "
        "stable reasons");
    foreign_kernels.Shutdown();
    ok &= Require(
        !key_light.Apply(input, &counters, &error) &&
            key_light_kernels.synchronous_submission_count() == before_foreign,
        "key light must reject a stale foreign output without a driver "
        "submission");
  }

  {
    using studiocast::video::OpenVulkanVirtualKeyLight;
    using studiocast::video::OpenVulkanVirtualKeyLightCounters;
    using studiocast::video::OpenVulkanVirtualKeyLightInput;
    using studiocast::video::OpenVulkanVirtualKeyLightMatteSource;

    UtilityKernels lost_key_light_kernels;
    if (!lost_key_light_kernels.Initialize(&error))
      return OptionalSkip("key-light device-loss context unavailable: " +
                          error);
    constexpr int w = 2;
    constexpr int h = 2;
    VulkanImage foreground, output, alpha_upload, alpha;
    if (!AllocateU8(lost_key_light_kernels, &foreground, w, h,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(lost_key_light_kernels, &output, w, h,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateF32(lost_key_light_kernels, &alpha_upload, w, h, &error) ||
        !alpha.Allocate(lost_key_light_kernels.device(), w, h,
                        VulkanPixelFormat::f32_1, false, &error)) {
      return OptionalSkip("key-light device-loss allocation failed: " + error);
    }
    FillU8(foreground, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    FillF32(alpha_upload, {0.0f, 0.25f, 0.75f, 1.0f});
    if (!foreground.Flush(&error) || !alpha_upload.Flush(&error) ||
        !lost_key_light_kernels.ResizeBilinearF32_1(alpha_upload, alpha,
                                                    &error)) {
      return OptionalSkip("key-light device-loss fixture setup failed: " +
                          error);
    }
    auto readiness =
        SyntheticProductionMattingReadiness(lost_key_light_kernels, true);
    OpenVulkanVirtualKeyLight key_light;
    ok &= Require(key_light.EnsureInitialized(&lost_key_light_kernels, w, h, 70,
                                              2, -90, readiness, &error),
                  "key-light device-loss wrapper initialization failed: " +
                      error);
    OpenVulkanVirtualKeyLightInput input;
    input.foreground = &foreground;
    input.alpha = &alpha;
    input.output = &output;
    input.capture_sequence = 901;
    input.resident_alpha_sequence = 901;
    input.alpha_resize_completion_count_before = 0;
    input.alpha_resize_completion_count_after = 1;
    input.matting_inference_count_before = 0;
    input.matte_source =
        OpenVulkanVirtualKeyLightMatteSource::independently_inferred;
    input.matting_readiness = &readiness;
    lost_key_light_kernels.device()->InjectNextSubmissionResultForTesting(
        studiocast::vulkan::VulkanSubmissionPhase::queue_submit,
        studiocast::vulkan::VK_ERROR_DEVICE_LOST);
    OpenVulkanVirtualKeyLightCounters counters;
    ok &= Require(
        !key_light.Apply(input, &counters, &error) &&
            error.find("vulkan_virtual_key_light_runtime_failed") !=
                std::string::npos &&
            error.find("[vulkan_device_lost]") != std::string::npos &&
            counters.dispatch_calls == 0 &&
            counters.runtime_failure_frames == 1 &&
            counters.device_loss_frames == 1 &&
            counters.alpha_readback_calls == 0 &&
            counters.cpu_fallback_calls == 0,
        "key-light device loss must retain nested reasons, count no successful "
        "dispatch, and never substitute CPU/readback work");
    const auto submitted_after_loss =
        lost_key_light_kernels.device()->health().submitted_serial;
    ok &= Require(
        !key_light.Apply(input, &counters, &error) &&
            lost_key_light_kernels.device()->health().submitted_serial ==
                submitted_after_loss,
        "poisoned key-light context must reject without another driver "
        "submission");
  }

  {
    using studiocast::video::OpenVulkanVirtualBackgroundBlur;
    using studiocast::video::OpenVulkanVirtualBackgroundBlurCounters;
    using studiocast::video::OpenVulkanVirtualBackgroundBlurInput;
    using studiocast::video::ResolveOpenVulkanVirtualBackgroundBlurParameters;

    UtilityKernels blur_kernels;
    if (!blur_kernels.Initialize(&error))
      return OptionalSkip("blur Vulkan context unavailable: " + error);
    constexpr int w = 5;
    constexpr int h = 3;
    std::vector<std::uint8_t> foreground(
        static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3u);
    for (std::size_t i = 0; i < foreground.size(); ++i)
      foreground[i] = static_cast<std::uint8_t>((i * 47u + 13u) & 0xffu);
    const std::vector<float> alpha = {
        0.0f,  0.1f, 0.25f, 0.5f,  1.0f,  0.75f, 0.6f, 0.4f,
        0.2f,  0.0f, 1.0f,  0.85f, 0.5f, 0.15f, 0.0f,
    };

    VulkanImage gpu_foreground, gpu_output, alpha_upload, resident_alpha;
    VulkanImage alpha_tmp, alpha_feathered, blur_tmp, blurred;
    if (!AllocateU8(blur_kernels, &gpu_foreground, w, h,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(blur_kernels, &gpu_output, w, h,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateF32(blur_kernels, &alpha_upload, w, h, &error) ||
        !resident_alpha.Allocate(blur_kernels.device(), w, h,
                                 VulkanPixelFormat::f32_1,
                                 /*map_memory=*/false, &error) ||
        !alpha_tmp.Allocate(blur_kernels.device(), w, h,
                            VulkanPixelFormat::f32_1,
                            /*map_memory=*/false, &error) ||
        !alpha_feathered.Allocate(blur_kernels.device(), w, h,
                                  VulkanPixelFormat::f32_1,
                                  /*map_memory=*/false, &error) ||
        !blur_tmp.Allocate(blur_kernels.device(), w, h,
                           VulkanPixelFormat::rgb_u8,
                           /*map_memory=*/false, &error) ||
        !blurred.Allocate(blur_kernels.device(), w, h,
                          VulkanPixelFormat::rgb_u8,
                          /*map_memory=*/false, &error)) {
      return OptionalSkip("production blur resource allocation failed: " +
                          error);
    }
    FillU8(gpu_foreground, foreground);
    FillF32(alpha_upload, alpha);
    ok &= Require(gpu_foreground.Flush(&error) && alpha_upload.Flush(&error),
                  "production blur fixture flush failed: " + error);
    blur_kernels.InvalidateDescriptorBindingCacheForSetup();
    ok &= Require(blur_kernels.ResizeBilinearF32_1(alpha_upload,
                                                   resident_alpha, &error),
                  "production blur resident alpha setup failed: " + error);
    ok &= Require(
        !resident_alpha.mapped() && resident_alpha.device_local() &&
            !alpha_tmp.mapped() && alpha_tmp.device_local() &&
            !alpha_feathered.mapped() && alpha_feathered.device_local() &&
            !blur_tmp.mapped() && blur_tmp.device_local() &&
            !blurred.mapped() && blurred.device_local() &&
            gpu_output.mapped(),
        "production blur must keep alpha/RGB scratch non-mapped DEVICE_LOCAL "
        "and use only the mapped final-frame output transport");

    auto setup_readiness =
        SyntheticProductionMattingReadiness(blur_kernels, false);
    OpenVulkanVirtualBackgroundBlur blur;
    ok &= Require(blur.EnsureInitialized(&blur_kernels, w, h, setup_readiness,
                                         &error),
                  "synthetic test-seam blur wrapper initialization failed: " +
                      error);

    OpenVulkanVirtualBackgroundBlurInput input;
    input.foreground = &gpu_foreground;
    input.alpha = &resident_alpha;
    input.alpha_tmp = &alpha_tmp;
    input.alpha_feathered = &alpha_feathered;
    input.blur_tmp = &blur_tmp;
    input.blurred = &blurred;
    input.output = &gpu_output;
    input.capture_sequence = 91;
    input.resident_alpha_sequence = 91;
    input.alpha_resize_completion_count = 1;
    input.matting_readiness = &setup_readiness;

    OpenVulkanVirtualBackgroundBlurCounters missing_frame_evidence;
    input.strength = 8;
    ok &= Require(
        !blur.Apply(input, &missing_frame_evidence, &error) &&
            error.find("vulkan_virtual_background_blur_runtime_failed") !=
                std::string::npos &&
            error.find("current-frame resident alpha") != std::string::npos &&
            missing_frame_evidence.blur_composite_dispatch_calls == 0,
        "warmup/synthetic kernel availability must not replace current-frame "
        "matting inference evidence");

    auto current_readiness =
        SyntheticProductionMattingReadiness(blur_kernels, true);
    input.matting_readiness = &current_readiness;
    OpenVulkanVirtualBackgroundBlurCounters counters;
    const auto allocations_before =
        blur_kernels.device()->allocation_stats().allocation_count;
    const auto submissions_before =
        blur_kernels.synchronous_submission_count();
    const int strengths[] = {1, 8, 16, 64};
    std::uint64_t expected_feather_dispatches = 0;
    for (int strength : strengths) {
      input.strength = strength;
      const auto parameters =
          ResolveOpenVulkanVirtualBackgroundBlurParameters(strength);
      ok &= Require(blur.Apply(input, &counters, &error),
                    "production blur wrapper dispatch failed at strength " +
                        std::to_string(strength) + ": " + error);
      ok &= Require(gpu_output.Invalidate(&error),
                    "production blur final transport invalidate failed: " +
                        error);
      std::vector<float> alpha_reference = alpha;
      if (parameters.alpha_feather_radius > 0) {
        alpha_reference = BlurF32Reference(
            alpha, w, h, parameters.alpha_feather_radius);
        ++expected_feather_dispatches;
      }
      ok &= CompareU8(
          ReadU8(gpu_output),
          CompositeReference(
              foreground,
              BlurU8Reference(foreground, w, h,
                              parameters.background_radius),
              alpha_reference),
          "production Vulkan virtual background blur strength " +
              std::to_string(strength));
    }
    ok &= Require(
        counters.blur_composite_dispatch_calls == std::size(strengths) &&
            counters.alpha_feather_dispatch_calls ==
                expected_feather_dispatches &&
            counters.runtime_failure_frames == 0 &&
            counters.alpha_readback_calls == 0 &&
            counters.cpu_fallback_calls == 0 &&
            blur_kernels.synchronous_submission_count() ==
                submissions_before + std::size(strengths) +
                                         expected_feather_dispatches &&
            blur_kernels.device()->allocation_stats().allocation_count ==
                allocations_before,
        "repeated production blur must reuse bounded resources, account only "
        "successful submissions, and prove zero alpha readback/CPU fallback");

    UtilityKernels foreign_kernels;
    if (!foreign_kernels.Initialize(&error))
      return OptionalSkip("foreign blur Vulkan context unavailable: " + error);
    VulkanImage foreign_output;
    if (!AllocateU8(foreign_kernels, &foreign_output, w, h,
                    VulkanPixelFormat::rgb_u8, &error)) {
      return OptionalSkip("foreign blur output allocation failed: " + error);
    }
    const auto submissions_before_foreign =
        blur_kernels.synchronous_submission_count();
    input.output = &foreign_output;
    input.strength = 8;
    OpenVulkanVirtualBackgroundBlurCounters foreign_counters;
    ok &= Require(
        !blur.Apply(input, &foreign_counters, &error) &&
            error.find("vulkan_virtual_background_blur_runtime_failed") !=
                std::string::npos &&
            error.find("[vulkan_foreign_context]") != std::string::npos &&
            foreign_counters.blur_composite_dispatch_calls == 0 &&
            blur_kernels.synchronous_submission_count() ==
                submissions_before_foreign,
        "blur must reject foreign output before dispatch with nested stable "
        "effect/context reasons");
    foreign_kernels.Shutdown();
    const auto submissions_before_stale =
        blur_kernels.synchronous_submission_count();
    ok &= Require(
        !blur.Apply(input, &foreign_counters, &error) &&
            error.find("vulkan_virtual_background_blur_runtime_failed") !=
                std::string::npos &&
            foreign_counters.blur_composite_dispatch_calls == 0 &&
            blur_kernels.synchronous_submission_count() ==
                submissions_before_stale,
        "blur must reject a stale foreign output without a driver "
        "submission");
  }

  {
    using studiocast::video::OpenVulkanVirtualBackgroundRemove;
    using studiocast::video::OpenVulkanVirtualBackgroundRemoveCounters;
    using studiocast::video::OpenVulkanVirtualBackgroundRemoveInput;
    using studiocast::video::
        ResolveOpenVulkanVirtualBackgroundRemoveParameters;

    UtilityKernels remove_kernels;
    if (!remove_kernels.Initialize(&error))
      return OptionalSkip("remove Vulkan context unavailable: " + error);
    constexpr int w = 5;
    constexpr int h = 3;
    std::vector<std::uint8_t> foreground(
        static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3u);
    for (std::size_t i = 0; i < foreground.size(); ++i)
      foreground[i] = static_cast<std::uint8_t>((i * 29u + 7u) & 0xffu);
    // A binary-exact constant matte keeps the wrapper's byte parity assertion
    // exact across vendors while still exercising both feather dispatches.
    // Spatial f32 blur behavior is covered independently below.
    const std::vector<float> alpha(static_cast<std::size_t>(w) *
                                       static_cast<std::size_t>(h),
                                   0.5f);

    VulkanImage gpu_foreground, gpu_output, alpha_upload, resident_alpha;
    VulkanImage alpha_tmp, alpha_feathered;
    if (!AllocateU8(remove_kernels, &gpu_foreground, w, h,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(remove_kernels, &gpu_output, w, h,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateF32(remove_kernels, &alpha_upload, w, h, &error) ||
        !resident_alpha.Allocate(remove_kernels.device(), w, h,
                                 VulkanPixelFormat::f32_1,
                                 /*map_memory=*/false, &error) ||
        !alpha_tmp.Allocate(remove_kernels.device(), w, h,
                            VulkanPixelFormat::f32_1,
                            /*map_memory=*/false, &error) ||
        !alpha_feathered.Allocate(remove_kernels.device(), w, h,
                                  VulkanPixelFormat::f32_1,
                                  /*map_memory=*/false, &error)) {
      return OptionalSkip("production remove resource allocation failed: " +
                          error);
    }
    FillU8(gpu_foreground, foreground);
    FillF32(alpha_upload, alpha);
    ok &= Require(gpu_foreground.Flush(&error) && alpha_upload.Flush(&error),
                  "production remove fixture flush failed: " + error);
    remove_kernels.InvalidateDescriptorBindingCacheForSetup();
    ok &= Require(remove_kernels.ResizeBilinearF32_1(
                      alpha_upload, resident_alpha, &error),
                  "production remove resident alpha setup failed: " + error);
    ok &= Require(
        !resident_alpha.mapped() && resident_alpha.device_local() &&
            !alpha_tmp.mapped() && alpha_tmp.device_local() &&
            !alpha_feathered.mapped() && alpha_feathered.device_local() &&
            gpu_output.mapped(),
        "production remove must keep alpha scratch non-mapped DEVICE_LOCAL "
        "and use only the mapped final-frame output transport");

    auto setup_readiness =
        SyntheticProductionMattingReadiness(remove_kernels, false);
    OpenVulkanVirtualBackgroundRemove remove;
    ok &= Require(remove.EnsureInitialized(
                      &remove_kernels, w, h, 8, "#123456", setup_readiness,
                      &error),
                  "synthetic test-seam remove wrapper initialization failed: " +
                      error);

    OpenVulkanVirtualBackgroundRemoveInput input;
    input.foreground = &gpu_foreground;
    input.alpha = &resident_alpha;
    input.alpha_tmp = &alpha_tmp;
    input.alpha_feathered = &alpha_feathered;
    input.output = &gpu_output;
    input.capture_sequence = 101;
    input.resident_alpha_sequence = 101;
    input.alpha_resize_completion_count = 1;
    input.matting_readiness = &setup_readiness;

    OpenVulkanVirtualBackgroundRemoveCounters missing_frame_evidence;
    ok &= Require(
        !remove.Apply(input, &missing_frame_evidence, &error) &&
            error.find("vulkan_virtual_background_remove_runtime_failed") !=
                std::string::npos &&
            error.find("current-frame resident alpha") != std::string::npos &&
            missing_frame_evidence.solid_composite_dispatch_calls == 0,
        "warmup/synthetic kernel availability must not replace current-frame "
        "remove matting inference evidence");

    auto current_readiness =
        SyntheticProductionMattingReadiness(remove_kernels, true);
    input.matting_readiness = &current_readiness;
    OpenVulkanVirtualBackgroundRemoveCounters counters;
    const auto allocations_before =
        remove_kernels.device()->allocation_stats().allocation_count;
    const auto submissions_before =
        remove_kernels.synchronous_submission_count();
    const struct RemoveCase {
      int strength;
      const char *remove_color;
      std::array<std::uint8_t, 3> rgb;
    } cases[] = {
        {1, "#123456", {0x12, 0x34, 0x56}},
        {8, "0xABCDEF", {0xab, 0xcd, 0xef}},
        {16, "010203", {1, 2, 3}},
        {64, "invalid", {0, 0, 0}},
    };
    std::uint64_t expected_feather_dispatches = 0;
    for (const auto &tc : cases) {
      ok &= Require(remove.EnsureInitialized(
                        &remove_kernels, w, h, tc.strength, tc.remove_color,
                        current_readiness, &error),
                    "production remove reconfiguration failed: " + error);
      const auto parameters =
          ResolveOpenVulkanVirtualBackgroundRemoveParameters(
              tc.strength, tc.remove_color);
      ok &= Require(remove.Apply(input, &counters, &error),
                    "production remove wrapper dispatch failed at strength " +
                        std::to_string(tc.strength) + ": " + error);
      ok &= Require(gpu_output.Invalidate(&error),
                    "production remove final transport invalidate failed: " +
                        error);
      std::vector<float> alpha_reference = alpha;
      if (parameters.alpha_feather_radius > 0) {
        alpha_reference = BlurF32Reference(
            alpha, w, h, parameters.alpha_feather_radius);
        ++expected_feather_dispatches;
      }
      ok &= CompareU8Exact(
          ReadU8(gpu_output),
          CompositeSolidReference(foreground, alpha_reference, tc.rgb),
          "production Vulkan virtual background remove strength/color parity " +
              std::to_string(tc.strength));
    }
    ok &= Require(
        counters.solid_composite_dispatch_calls == std::size(cases) &&
            counters.alpha_feather_dispatch_calls ==
                expected_feather_dispatches &&
            counters.runtime_failure_frames == 0 &&
            counters.alpha_readback_calls == 0 &&
            counters.cpu_fallback_calls == 0 &&
            remove_kernels.synchronous_submission_count() ==
                submissions_before + std::size(cases) +
                                         expected_feather_dispatches &&
            remove_kernels.device()->allocation_stats().allocation_count ==
                allocations_before,
        "repeated production remove must reuse bounded resources, account "
        "only successful submissions, and prove zero alpha readback/CPU "
        "fallback");

    UtilityKernels foreign_kernels;
    if (!foreign_kernels.Initialize(&error))
      return OptionalSkip("foreign remove Vulkan context unavailable: " +
                          error);
    VulkanImage foreign_output;
    if (!AllocateU8(foreign_kernels, &foreign_output, w, h,
                    VulkanPixelFormat::rgb_u8, &error)) {
      return OptionalSkip("foreign remove output allocation failed: " + error);
    }
    const auto submissions_before_foreign =
        remove_kernels.synchronous_submission_count();
    input.output = &foreign_output;
    OpenVulkanVirtualBackgroundRemoveCounters foreign_counters;
    ok &= Require(
        !remove.Apply(input, &foreign_counters, &error) &&
            error.find("vulkan_virtual_background_remove_runtime_failed") !=
                std::string::npos &&
            error.find("[vulkan_foreign_context]") != std::string::npos &&
            foreign_counters.solid_composite_dispatch_calls == 0 &&
            remove_kernels.synchronous_submission_count() ==
                submissions_before_foreign,
        "remove must reject foreign output before dispatch with nested stable "
        "effect/context reasons");
    foreign_kernels.Shutdown();
    const auto submissions_before_stale =
        remove_kernels.synchronous_submission_count();
    ok &= Require(
        !remove.Apply(input, &foreign_counters, &error) &&
            error.find("vulkan_virtual_background_remove_runtime_failed") !=
                std::string::npos &&
            foreign_counters.solid_composite_dispatch_calls == 0 &&
            remove_kernels.synchronous_submission_count() ==
                submissions_before_stale,
        "remove must reject a stale foreign output without a driver "
        "submission");
  }

  {
    using studiocast::video::OpenVulkanVirtualBackgroundRemove;
    using studiocast::video::OpenVulkanVirtualBackgroundRemoveCounters;
    using studiocast::video::OpenVulkanVirtualBackgroundRemoveInput;

    UtilityKernels lost_remove_kernels;
    if (!lost_remove_kernels.Initialize(&error))
      return OptionalSkip("remove device-loss context unavailable: " + error);
    constexpr int w = 2;
    constexpr int h = 2;
    VulkanImage foreground, output, alpha_upload, alpha, alpha_tmp;
    VulkanImage alpha_feathered;
    if (!AllocateU8(lost_remove_kernels, &foreground, w, h,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(lost_remove_kernels, &output, w, h,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateF32(lost_remove_kernels, &alpha_upload, w, h, &error) ||
        !alpha.Allocate(lost_remove_kernels.device(), w, h,
                        VulkanPixelFormat::f32_1, false, &error) ||
        !alpha_tmp.Allocate(lost_remove_kernels.device(), w, h,
                            VulkanPixelFormat::f32_1, false, &error) ||
        !alpha_feathered.Allocate(lost_remove_kernels.device(), w, h,
                                  VulkanPixelFormat::f32_1, false, &error)) {
      return OptionalSkip("remove device-loss allocation failed: " + error);
    }
    FillU8(foreground, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    FillF32(alpha_upload, {0.0f, 0.25f, 0.75f, 1.0f});
    if (!foreground.Flush(&error) || !alpha_upload.Flush(&error) ||
        !lost_remove_kernels.ResizeBilinearF32_1(alpha_upload, alpha, &error)) {
      return OptionalSkip("remove device-loss fixture setup failed: " + error);
    }
    auto readiness =
        SyntheticProductionMattingReadiness(lost_remove_kernels, true);
    OpenVulkanVirtualBackgroundRemove remove;
    ok &= Require(remove.EnsureInitialized(
                      &lost_remove_kernels, w, h, 1, "#102030", readiness,
                      &error),
                  "remove device-loss wrapper initialization failed: " +
                      error);
    OpenVulkanVirtualBackgroundRemoveInput input;
    input.foreground = &foreground;
    input.alpha = &alpha;
    input.alpha_tmp = &alpha_tmp;
    input.alpha_feathered = &alpha_feathered;
    input.output = &output;
    input.capture_sequence = 17;
    input.resident_alpha_sequence = 17;
    input.alpha_resize_completion_count = 1;
    input.matting_readiness = &readiness;
    lost_remove_kernels.device()->InjectNextSubmissionResultForTesting(
        studiocast::vulkan::VulkanSubmissionPhase::queue_submit,
        studiocast::vulkan::VK_ERROR_DEVICE_LOST);
    OpenVulkanVirtualBackgroundRemoveCounters counters;
    ok &= Require(
        !remove.Apply(input, &counters, &error) &&
            error.find("vulkan_virtual_background_remove_runtime_failed") !=
                std::string::npos &&
            error.find("[vulkan_device_lost]") != std::string::npos &&
            counters.solid_composite_dispatch_calls == 0 &&
            counters.runtime_failure_frames == 1 &&
            counters.device_loss_frames == 1 &&
            counters.alpha_readback_calls == 0 &&
            counters.cpu_fallback_calls == 0,
        "remove device loss must retain nested reasons, count only successful "
        "dispatches, and never substitute CPU/readback work");
    const auto submitted_after_loss =
        lost_remove_kernels.device()->health().submitted_serial;
    ok &= Require(
        !remove.Apply(input, &counters, &error) &&
            lost_remove_kernels.device()->health().submitted_serial ==
                submitted_after_loss,
        "poisoned remove context must reject without another driver "
        "submission");
  }

  {
    using studiocast::video::OpenVulkanVirtualBackgroundReplace;
    using studiocast::video::OpenVulkanVirtualBackgroundReplaceCounters;

    UtilityKernels lost_setup_kernels;
    if (!lost_setup_kernels.Initialize(&error)) {
      return OptionalSkip("replace setup-loss context unavailable: " + error);
    }
    ScopedReplaceAssetDirectory assets;
    const auto ppm_path = assets.path / "setup-loss.ppm";
    if (!WritePpmFixture(ppm_path, 1, 1, {9, 19, 29}))
      return 1;
    const auto readiness =
        SyntheticProductionMattingReadiness(lost_setup_kernels, false);
    lost_setup_kernels.device()->InjectNextSubmissionResultForTesting(
        studiocast::vulkan::VulkanSubmissionPhase::queue_submit,
        studiocast::vulkan::VK_ERROR_DEVICE_LOST);
    OpenVulkanVirtualBackgroundReplace replace;
    OpenVulkanVirtualBackgroundReplaceCounters counters;
    ok &= Require(
        !replace.EnsureInitialized(
            &lost_setup_kernels, 1, 1, 1,
            PreparedReplaceAsset(ppm_path, 1), readiness, &counters, &error) &&
            error.find(
                "vulkan_virtual_background_replace_initialization_failed") !=
                std::string::npos &&
            error.find(
                "vulkan_virtual_background_replace_asset_upload_failed") !=
                std::string::npos &&
            error.find("[vulkan_device_lost]") != std::string::npos &&
            counters.asset_decode_calls == 1 &&
            counters.asset_upload_calls == 0 &&
            counters.asset_resize_dispatch_calls == 0 &&
            counters.alpha_readback_calls == 0 &&
            counters.cpu_fallback_calls == 0 && !replace.asset_valid(),
        "replace setup upload loss must retain initialization/asset/device "
        "reasons, count only completed work, and invalidate the asset");
  }

  {
    using studiocast::video::OpenVulkanVirtualBackgroundReplace;
    using studiocast::video::OpenVulkanVirtualBackgroundReplaceCounters;
    using studiocast::video::OpenVulkanVirtualBackgroundReplaceInput;
    using studiocast::video::
        ResolveOpenVulkanVirtualBackgroundReplaceParameters;

    UtilityKernels replace_kernels;
    if (!replace_kernels.Initialize(&error))
      return OptionalSkip("replace Vulkan context unavailable: " + error);
    constexpr int width = 5;
    constexpr int height = 3;
    ScopedReplaceAssetDirectory assets;
    const auto ppm_path = assets.path / "background.PPM";
    const auto png_path = assets.path / "background.PNG";
    const auto unsupported_path = assets.path / "background.jpg";
    const auto truncated_path = assets.path / "truncated.ppm";
    std::vector<std::uint8_t> background(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
        3u);
    std::vector<std::uint8_t> foreground(background.size());
    for (std::size_t i = 0; i < background.size(); ++i) {
      background[i] = static_cast<std::uint8_t>((i * 17u + 31u) & 0xffu);
      foreground[i] = static_cast<std::uint8_t>((i * 43u + 9u) & 0xffu);
    }
    std::vector<std::uint8_t> png_rgba(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
        4u);
    std::vector<std::uint8_t> png_background(background.size());
    for (std::size_t pixel = 0;
         pixel < static_cast<std::size_t>(width) *
                     static_cast<std::size_t>(height);
         ++pixel) {
      const std::uint8_t asset_alpha =
          static_cast<std::uint8_t>((pixel * 37u + 19u) & 0xffu);
      for (std::size_t channel = 0; channel < 3u; ++channel) {
        const std::uint8_t value = background[pixel * 3u + channel];
        png_rgba[pixel * 4u + channel] = value;
        png_background[pixel * 3u + channel] =
            static_cast<std::uint8_t>(
                (static_cast<std::uint32_t>(value) * asset_alpha + 127u) /
                255u);
      }
      png_rgba[pixel * 4u + 3u] = asset_alpha;
    }
    if (!WritePpmFixture(ppm_path, width, height, background) ||
        !WritePngRgbaFixture(png_path, width, height, png_rgba)) {
      std::cerr << "failed to write hermetic replacement image fixtures\n";
      return 1;
    }
    {
      std::ofstream unsupported(unsupported_path,
                                std::ios::binary | std::ios::trunc);
      unsupported << "not an image";
      std::ofstream truncated(truncated_path,
                              std::ios::binary | std::ios::trunc);
      truncated << "P6\n5 3\n255\n\x01\x02";
    }

    VulkanImage gpu_foreground, gpu_output, alpha_upload, resident_alpha;
    VulkanImage alpha_tmp, alpha_feathered;
    if (!AllocateU8(replace_kernels, &gpu_foreground, width, height,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(replace_kernels, &gpu_output, width, height,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateF32(replace_kernels, &alpha_upload, width, height, &error) ||
        !resident_alpha.Allocate(replace_kernels.device(), width, height,
                                 VulkanPixelFormat::f32_1, false, &error) ||
        !alpha_tmp.Allocate(replace_kernels.device(), width, height,
                            VulkanPixelFormat::f32_1, false, &error) ||
        !alpha_feathered.Allocate(replace_kernels.device(), width, height,
                                  VulkanPixelFormat::f32_1, false, &error)) {
      return OptionalSkip("production replace resource allocation failed: " +
                          error);
    }
    // Spatial variation is required here: a constant matte would make every
    // feather radius produce the same pixels and would only test dispatch
    // accounting, not the strength-dependent alpha semantics.
    const std::vector<float> alpha = {
        0.00f, 0.10f, 0.90f, 0.20f, 1.00f,
        0.75f, 0.30f, 0.50f, 0.95f, 0.05f,
        0.40f, 1.00f, 0.15f, 0.65f, 0.25f,
    };
    FillU8(gpu_foreground, foreground);
    FillF32(alpha_upload, alpha);
    ok &= Require(gpu_foreground.Flush(&error) && alpha_upload.Flush(&error),
                  "production replace fixture flush failed: " + error);
    replace_kernels.InvalidateDescriptorBindingCacheForSetup();
    ok &= Require(replace_kernels.ResizeBilinearF32_1(
                      alpha_upload, resident_alpha, &error),
                  "production replace resident alpha setup failed: " + error);

    auto setup_readiness =
        SyntheticProductionMattingReadiness(replace_kernels, false);
    OpenVulkanVirtualBackgroundReplace replace;
    OpenVulkanVirtualBackgroundReplaceCounters counters;
    auto prepared_ppm = PreparedReplaceAsset(ppm_path, 1);
    const auto setup_submissions_before =
        replace_kernels.synchronous_submission_count();
    ok &= Require(replace.EnsureInitialized(
                      &replace_kernels, width, height, 8, prepared_ppm,
                      setup_readiness, &counters, &error),
                  "synthetic test-seam replace initialization failed: " +
                      error);
    ok &= Require(
        counters.asset_allocation_calls == 3 &&
            counters.asset_decode_calls == 1 &&
            counters.asset_upload_calls == 1 &&
            counters.asset_resize_dispatch_calls == 1 &&
            replace_kernels.synchronous_submission_count() ==
                setup_submissions_before + 2 &&
            replace.upload_staging_image().mapped() &&
            replace.upload_staging_image().host_visible() &&
            !replace.source_image().mapped() &&
            replace.source_image().device_local() &&
            !replace.replacement_image().mapped() &&
            replace.replacement_image().device_local() &&
            !resident_alpha.mapped() && resident_alpha.device_local() &&
            !alpha_tmp.mapped() && alpha_tmp.device_local() &&
            !alpha_feathered.mapped() && alpha_feathered.device_local() &&
            gpu_output.mapped(),
        "replace setup must decode/upload/resize once and retain only explicit "
        "mapped upload/final transport resources around DEVICE_LOCAL source, "
        "replacement, and alpha scratch");

    const auto setup_counters = counters;
    const auto cache_submissions_before =
        replace_kernels.synchronous_submission_count();
    const auto cache_allocations_before =
        replace_kernels.device()->allocation_stats().allocation_count;
    ok &= Require(replace.EnsureInitialized(
                      &replace_kernels, width, height, 15, prepared_ppm,
                      setup_readiness, &counters, &error) &&
                      counters.asset_allocation_calls ==
                          setup_counters.asset_allocation_calls &&
                      counters.asset_decode_calls ==
                          setup_counters.asset_decode_calls &&
                      counters.asset_upload_calls ==
                          setup_counters.asset_upload_calls &&
                      counters.asset_resize_dispatch_calls ==
                          setup_counters.asset_resize_dispatch_calls &&
                      replace_kernels.synchronous_submission_count() ==
                          cache_submissions_before &&
                      replace_kernels.device()
                              ->allocation_stats()
                              .allocation_count == cache_allocations_before,
                  "same path/mtime/geometry/context must reuse the replacement "
                  "asset without allocation or submission while allowing "
                  "strength-only reconfiguration");

    OpenVulkanVirtualBackgroundReplaceInput input;
    input.foreground = &gpu_foreground;
    input.alpha = &resident_alpha;
    input.alpha_tmp = &alpha_tmp;
    input.alpha_feathered = &alpha_feathered;
    input.output = &gpu_output;
    input.capture_sequence = 211;
    input.resident_alpha_sequence = 211;
    input.alpha_resize_completion_count = 1;
    input.matting_readiness = &setup_readiness;
    const auto before_missing_evidence = counters;
    ok &= Require(
        !replace.Apply(input, &counters, &error) &&
            error.find("vulkan_virtual_background_replace_runtime_failed") !=
                std::string::npos &&
            error.find("current-frame resident alpha") != std::string::npos &&
            counters.replacement_composite_dispatch_calls ==
                before_missing_evidence.replacement_composite_dispatch_calls,
        "warmup/asset/kernel readiness must not replace current-frame matting "
        "evidence");

    auto current_readiness =
        SyntheticProductionMattingReadiness(replace_kernels, true);
    input.matting_readiness = &current_readiness;
    const auto allocations_before_frames =
        replace_kernels.device()->allocation_stats().allocation_count;
    const auto submissions_before_frames =
        replace_kernels.synchronous_submission_count();
    const int strengths[] = {1, 8, 16, 64};
    std::uint64_t expected_feathers = 0;
    for (int strength : strengths) {
      ok &= Require(replace.EnsureInitialized(
                        &replace_kernels, width, height, strength, prepared_ppm,
                        current_readiness, &counters, &error),
                    "replace strength reconfiguration failed: " + error);
      ok &= Require(replace.Apply(input, &counters, &error),
                    "production replace dispatch failed at strength " +
                        std::to_string(strength) + ": " + error);
      ok &= Require(gpu_output.Invalidate(&error),
                    "production replace final transport invalidate failed: " +
                        error);
      const auto parameters =
          ResolveOpenVulkanVirtualBackgroundReplaceParameters(strength);
      std::vector<float> alpha_reference = alpha;
      if (parameters.alpha_feather_radius > 0) {
        alpha_reference = BlurF32Reference(
            alpha, width, height, parameters.alpha_feather_radius);
        ++expected_feathers;
      }
      ok &= CompareU8Exact(
          ReadU8(gpu_output),
          CompositeReference(foreground, background, alpha_reference),
          "production Vulkan virtual background replace strength parity " +
              std::to_string(strength));
    }
    ok &= Require(
        counters.replacement_composite_dispatch_calls ==
                std::size(strengths) &&
            counters.alpha_feather_dispatch_calls == expected_feathers &&
            counters.asset_decode_calls == 1 &&
            counters.asset_upload_calls == 1 &&
            counters.asset_resize_dispatch_calls == 1 &&
            counters.alpha_readback_calls == 0 &&
            counters.cpu_fallback_calls == 0 &&
            replace_kernels.synchronous_submission_count() ==
                submissions_before_frames + std::size(strengths) +
                                                 expected_feathers &&
            replace_kernels.device()->allocation_stats().allocation_count ==
                allocations_before_frames,
        "repeated replace frames must reuse bounded setup resources, perform "
        "no per-frame decode/upload/allocation, and prove zero readback/CPU "
        "fallback");

    UtilityKernels foreign_kernels;
    if (!foreign_kernels.Initialize(&error))
      return OptionalSkip("foreign replace context unavailable: " + error);
    VulkanImage foreign_output;
    if (!AllocateU8(foreign_kernels, &foreign_output, width, height,
                    VulkanPixelFormat::rgb_u8, &error)) {
      return OptionalSkip("foreign replace output allocation failed: " +
                          error);
    }
    const auto before_foreign =
        replace_kernels.synchronous_submission_count();
    input.output = &foreign_output;
    ok &= Require(
        !replace.Apply(input, &counters, &error) &&
            error.find("[vulkan_foreign_context]") != std::string::npos &&
            replace_kernels.synchronous_submission_count() == before_foreign,
        "replace must reject a foreign output before dispatch");
    foreign_kernels.Shutdown();
    ok &= Require(
        !replace.Apply(input, &counters, &error) &&
            replace_kernels.synchronous_submission_count() == before_foreign,
        "replace must reject a stale foreign output without driver work");
    input.output = &gpu_output;

    const auto before_geometry = counters;
    const auto geometry_submissions_before =
        replace_kernels.synchronous_submission_count();
    ok &= Require(replace.EnsureInitialized(
                      &replace_kernels, 6, 4, 8, prepared_ppm,
                      current_readiness, &counters, &error) &&
                      counters.asset_decode_calls ==
                          before_geometry.asset_decode_calls &&
                      counters.asset_upload_calls ==
                          before_geometry.asset_upload_calls &&
                      counters.asset_resize_dispatch_calls ==
                          before_geometry.asset_resize_dispatch_calls + 1 &&
                      replace_kernels.synchronous_submission_count() ==
                          geometry_submissions_before + 1,
                  "frame geometry change must reuse the uploaded source and "
                  "refresh only the bounded DEVICE_LOCAL replacement resize");

    const auto prepared_png = PreparedReplaceAsset(png_path, 2);
    ok &= Require(replace.EnsureInitialized(
                      &replace_kernels, width, height, 8, prepared_png,
                      current_readiness, &counters, &error) &&
                      counters.asset_decode_calls ==
                          before_geometry.asset_decode_calls + 1 &&
                      counters.asset_upload_calls ==
                          before_geometry.asset_upload_calls + 1,
                  "case-insensitive PNG replacement must decode/upload once "
                  "on path reconfiguration");
    ok &= Require(replace.Apply(input, &counters, &error) &&
                      gpu_output.Invalidate(&error),
                  "PNG-alpha replace dispatch failed: " + error);
    ok &= CompareU8Exact(
        ReadU8(gpu_output), CompositeReference(foreground, png_background, alpha),
        "production Vulkan PNG replacement alpha-on-black parity");

    std::vector<std::uint8_t> changed_background = background;
    std::reverse(changed_background.begin(), changed_background.end());
    if (!WritePngFixture(png_path, width, height, changed_background)) {
      std::cerr << "failed to rewrite changed replacement fixture\n";
      return 1;
    }
    const auto prepared_changed_png = PreparedReplaceAsset(png_path, 3);
    const auto before_changed = counters;
    ok &= Require(replace.EnsureInitialized(
                      &replace_kernels, width, height, 8,
                      prepared_changed_png, current_readiness, &counters,
                      &error) &&
                      counters.asset_decode_calls ==
                          before_changed.asset_decode_calls + 1 &&
                      counters.asset_upload_calls ==
                          before_changed.asset_upload_calls + 1 &&
                      counters.asset_resize_dispatch_calls ==
                          before_changed.asset_resize_dispatch_calls + 1,
                  "same-path changed prepared mtime must invalidate, decode, "
                  "upload, and resize exactly once");

    const auto before_bad_asset =
        replace_kernels.synchronous_submission_count();
    const auto unsupported = PreparedReplaceAsset(unsupported_path, 4);
    ok &= Require(
        !replace.EnsureInitialized(&replace_kernels, width, height, 8,
                                   unsupported, current_readiness, &counters,
                                   &error) &&
            error.find(
                "vulkan_virtual_background_replace_initialization_failed") !=
                std::string::npos &&
            error.find("vulkan_virtual_background_replace_asset_invalid") !=
                std::string::npos &&
            error.find("supported: .png, .ppm (P6)") != std::string::npos &&
            !replace.asset_valid() &&
            replace_kernels.synchronous_submission_count() == before_bad_asset,
        "unsupported replacement extension must fail setup with nested stable "
        "asset reason and invalidate the old cache without GPU work");
    ok &= Require(!replace.Apply(input, &counters, &error) &&
                      replace_kernels.synchronous_submission_count() ==
                          before_bad_asset,
                  "a failed asset refresh must never reuse the stale image");

    const auto truncated = PreparedReplaceAsset(truncated_path, 5);
    ok &= Require(!replace.EnsureInitialized(
                      &replace_kernels, width, height, 8, truncated,
                      current_readiness, &counters, &error) &&
                      error.find("asset_invalid") != std::string::npos &&
                      error.find("PPM truncated") != std::string::npos,
                  "truncated PPM replacement must fail closed at setup");

    studiocast::video::detail::PreparedReplaceBackgroundSource missing;
    missing.path = assets.path / "missing.png";
    missing.error = "failed to stat replace image: No such file";
    ok &= Require(!replace.EnsureInitialized(
                      &replace_kernels, width, height, 8, missing,
                      current_readiness, &counters, &error) &&
                      error.find("asset_invalid") != std::string::npos &&
                      error.find("failed to stat") != std::string::npos,
                  "missing/unstatable replacement must retain the prepared "
                  "setup failure");
    studiocast::video::detail::PreparedReplaceBackgroundSource empty;
    ok &= Require(!replace.EnsureInitialized(
                      &replace_kernels, width, height, 8, empty,
                      current_readiness, &counters, &error) &&
                      error.find("asset_invalid") != std::string::npos &&
                      error.find("replace_path not set") != std::string::npos,
                  "empty replacement path must retain direct-wrapper stable "
                  "asset failure while the planner keeps its compatibility "
                  "disable rule");
  }

  {
    using studiocast::video::OpenVulkanVirtualBackgroundReplace;
    using studiocast::video::OpenVulkanVirtualBackgroundReplaceCounters;
    using studiocast::video::OpenVulkanVirtualBackgroundReplaceInput;

    UtilityKernels lost_replace_kernels;
    if (!lost_replace_kernels.Initialize(&error))
      return OptionalSkip("replace device-loss context unavailable: " + error);
    constexpr int width = 2;
    constexpr int height = 2;
    ScopedReplaceAssetDirectory assets;
    const auto ppm_path = assets.path / "lost.ppm";
    const std::vector<std::uint8_t> background = {
        20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130,
    };
    if (!WritePpmFixture(ppm_path, width, height, background))
      return 1;
    VulkanImage foreground, output, alpha_upload, alpha, alpha_tmp;
    VulkanImage alpha_feathered;
    if (!AllocateU8(lost_replace_kernels, &foreground, width, height,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(lost_replace_kernels, &output, width, height,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateF32(lost_replace_kernels, &alpha_upload, width, height,
                     &error) ||
        !alpha.Allocate(lost_replace_kernels.device(), width, height,
                        VulkanPixelFormat::f32_1, false, &error) ||
        !alpha_tmp.Allocate(lost_replace_kernels.device(), width, height,
                            VulkanPixelFormat::f32_1, false, &error) ||
        !alpha_feathered.Allocate(lost_replace_kernels.device(), width, height,
                                  VulkanPixelFormat::f32_1, false, &error)) {
      return OptionalSkip("replace device-loss allocation failed: " + error);
    }
    FillU8(foreground, background);
    FillF32(alpha_upload, {0.0f, 0.25f, 0.75f, 1.0f});
    if (!foreground.Flush(&error) || !alpha_upload.Flush(&error) ||
        !lost_replace_kernels.ResizeBilinearF32_1(alpha_upload, alpha,
                                                  &error)) {
      return OptionalSkip("replace device-loss fixture setup failed: " +
                          error);
    }
    auto readiness =
        SyntheticProductionMattingReadiness(lost_replace_kernels, true);
    OpenVulkanVirtualBackgroundReplace replace;
    OpenVulkanVirtualBackgroundReplaceCounters counters;
    ok &= Require(replace.EnsureInitialized(
                      &lost_replace_kernels, width, height, 1,
                      PreparedReplaceAsset(ppm_path, 1), readiness, &counters,
                      &error),
                  "replace device-loss wrapper initialization failed: " +
                      error);
    OpenVulkanVirtualBackgroundReplaceInput input;
    input.foreground = &foreground;
    input.alpha = &alpha;
    input.alpha_tmp = &alpha_tmp;
    input.alpha_feathered = &alpha_feathered;
    input.output = &output;
    input.capture_sequence = 313;
    input.resident_alpha_sequence = 313;
    input.alpha_resize_completion_count = 1;
    input.matting_readiness = &readiness;
    lost_replace_kernels.device()->InjectNextSubmissionResultForTesting(
        studiocast::vulkan::VulkanSubmissionPhase::queue_submit,
        studiocast::vulkan::VK_ERROR_DEVICE_LOST);
    const auto composites_before =
        counters.replacement_composite_dispatch_calls;
    ok &= Require(
        !replace.Apply(input, &counters, &error) &&
            error.find("vulkan_virtual_background_replace_runtime_failed") !=
                std::string::npos &&
            error.find("[vulkan_device_lost]") != std::string::npos &&
            counters.replacement_composite_dispatch_calls ==
                composites_before &&
            counters.runtime_failure_frames == 1 &&
            counters.device_loss_frames == 1 &&
            counters.alpha_readback_calls == 0 &&
            counters.cpu_fallback_calls == 0,
        "replace device loss must retain nested reasons, count only successful "
        "dispatches, and never substitute CPU/readback work");
    const auto submitted_after_loss =
        lost_replace_kernels.device()->health().submitted_serial;
    ok &= Require(!replace.Apply(input, &counters, &error) &&
                      lost_replace_kernels.device()->health().submitted_serial ==
                          submitted_after_loss,
                  "poisoned replace context must reject without another "
                  "driver submission");
  }

  {
    using studiocast::video::OpenVulkanVirtualBackgroundBlur;
    using studiocast::video::OpenVulkanVirtualBackgroundBlurCounters;
    using studiocast::video::OpenVulkanVirtualBackgroundBlurInput;

    UtilityKernels lost_blur_kernels;
    if (!lost_blur_kernels.Initialize(&error))
      return OptionalSkip("blur device-loss context unavailable: " + error);
    constexpr int w = 2;
    constexpr int h = 2;
    VulkanImage foreground, output, alpha_upload, alpha, alpha_tmp;
    VulkanImage alpha_feathered, blur_tmp, blurred;
    if (!AllocateU8(lost_blur_kernels, &foreground, w, h,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(lost_blur_kernels, &output, w, h,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateF32(lost_blur_kernels, &alpha_upload, w, h, &error) ||
        !alpha.Allocate(lost_blur_kernels.device(), w, h,
                        VulkanPixelFormat::f32_1, false, &error) ||
        !alpha_tmp.Allocate(lost_blur_kernels.device(), w, h,
                            VulkanPixelFormat::f32_1, false, &error) ||
        !alpha_feathered.Allocate(lost_blur_kernels.device(), w, h,
                                  VulkanPixelFormat::f32_1, false, &error) ||
        !blur_tmp.Allocate(lost_blur_kernels.device(), w, h,
                           VulkanPixelFormat::rgb_u8, false, &error) ||
        !blurred.Allocate(lost_blur_kernels.device(), w, h,
                          VulkanPixelFormat::rgb_u8, false, &error)) {
      return OptionalSkip("blur device-loss allocation failed: " + error);
    }
    FillU8(foreground, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    FillF32(alpha_upload, {0.0f, 0.25f, 0.75f, 1.0f});
    if (!foreground.Flush(&error) || !alpha_upload.Flush(&error) ||
        !lost_blur_kernels.ResizeBilinearF32_1(alpha_upload, alpha, &error)) {
      return OptionalSkip("blur device-loss fixture setup failed: " + error);
    }
    auto readiness =
        SyntheticProductionMattingReadiness(lost_blur_kernels, true);
    OpenVulkanVirtualBackgroundBlur blur;
    ok &= Require(blur.EnsureInitialized(&lost_blur_kernels, w, h, readiness,
                                         &error),
                  "blur device-loss wrapper initialization failed: " + error);
    OpenVulkanVirtualBackgroundBlurInput input;
    input.foreground = &foreground;
    input.alpha = &alpha;
    input.alpha_tmp = &alpha_tmp;
    input.alpha_feathered = &alpha_feathered;
    input.blur_tmp = &blur_tmp;
    input.blurred = &blurred;
    input.output = &output;
    input.strength = 1;
    input.capture_sequence = 7;
    input.resident_alpha_sequence = 7;
    input.alpha_resize_completion_count = 1;
    input.matting_readiness = &readiness;
    lost_blur_kernels.device()->InjectNextSubmissionResultForTesting(
        studiocast::vulkan::VulkanSubmissionPhase::queue_submit,
        studiocast::vulkan::VK_ERROR_DEVICE_LOST);
    OpenVulkanVirtualBackgroundBlurCounters counters;
    ok &= Require(
        !blur.Apply(input, &counters, &error) &&
            error.find("vulkan_virtual_background_blur_runtime_failed") !=
                std::string::npos &&
            error.find("[vulkan_device_lost]") != std::string::npos &&
            counters.blur_composite_dispatch_calls == 0 &&
            counters.runtime_failure_frames == 1 &&
            counters.device_loss_frames == 1 &&
            counters.alpha_readback_calls == 0 &&
            counters.cpu_fallback_calls == 0,
        "blur device loss must retain nested reasons, count only successful "
        "dispatches, and never substitute CPU/readback work");
    const auto submitted_after_loss =
        lost_blur_kernels.device()->health().submitted_serial;
    ok &= Require(
        !blur.Apply(input, &counters, &error) &&
            lost_blur_kernels.device()->health().submitted_serial ==
                submitted_after_loss,
        "poisoned blur context must reject without another driver submission");
  }

  {
    using studiocast::video::OpenVulkanMirror;
    using studiocast::video::OpenVulkanMirrorCounters;
    using studiocast::video::OpenVulkanMirrorFinalStageInput;

    const struct MirrorCase {
      int width;
      int height;
      VulkanPixelFormat format;
      const char *label;
    } cases[] = {
        {1, 1, VulkanPixelFormat::rgb_u8, "RGB 1x1"},
        {5, 3, VulkanPixelFormat::rgb_u8, "RGB odd"},
        {6, 2, VulkanPixelFormat::rgb_u8, "RGB even"},
        {5, 3, VulkanPixelFormat::bgr_u8, "BGR odd"},
        {6, 2, VulkanPixelFormat::bgr_u8, "BGR even"},
    };

    for (const auto &tc : cases) {
      // Each case destroys its images at the end of the iteration. Prevent a
      // recycled raw VkBuffer handle from making the utility descriptor cache
      // look current when the next fixture allocates replacement images.
      kernels.InvalidateDescriptorBindingCacheForSetup();
      std::vector<std::uint8_t> source(static_cast<std::size_t>(tc.width) *
                                       static_cast<std::size_t>(tc.height) *
                                       3u);
      for (std::size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<std::uint8_t>((i * 73u + 19u) & 0xffu);

      VulkanImage gpu_src, gpu_dst, gpu_round_trip;
      if (!AllocateU8(kernels, &gpu_src, tc.width, tc.height, tc.format,
                      &error) ||
          !AllocateU8(kernels, &gpu_dst, tc.width, tc.height, tc.format,
                      &error) ||
          !AllocateU8(kernels, &gpu_round_trip, tc.width, tc.height, tc.format,
                      &error)) {
        return OptionalSkip(std::string("mirror allocation failed: ") + error);
      }
      FillU8(gpu_src, source);
      ok &= Require(gpu_src.Flush(&error),
                    std::string(tc.label) + " source flush failed: " + error);

      OpenVulkanMirror mirror;
      ok &= Require(
          mirror.EnsureInitialized(&kernels, tc.width, tc.height, &error),
          std::string(tc.label) + " mirror init should pass: " + error);
      OpenVulkanMirrorCounters counters;
      OpenVulkanMirrorFinalStageInput input;
      input.src = &gpu_src;
      input.dst = &gpu_dst;
      ok &= Require(!mirror.ApplyFinal(input, &counters, &error) &&
                        error.find("final-stage ordering boundary") !=
                            std::string::npos &&
                        counters.dispatch_calls == 0 &&
                        counters.runtime_failure_frames == 0,
                    std::string(tc.label) +
                        " mirror must reject execution before final order "
                        "inputs without dispatching");

      input.unmirrored_analysis_complete = true;
      input.output_geometry_ready = true;
      ok &=
          Require(mirror.ApplyFinal(input, &counters, &error),
                  std::string(tc.label) + " mirror dispatch failed: " + error);
      ok &= Require(gpu_dst.Invalidate(&error),
                    std::string(tc.label) +
                        " output invalidate failed: " + error);
      ok &= CompareU8Exact(ReadU8(gpu_dst),
                           MirrorU8Reference(source, tc.width, tc.height),
                           std::string("Vulkan mirror ") + tc.label);

      OpenVulkanMirrorFinalStageInput second;
      second.src = &gpu_dst;
      second.dst = &gpu_round_trip;
      second.unmirrored_analysis_complete = true;
      second.output_geometry_ready = true;
      ok &= Require(mirror.ApplyFinal(second, &counters, &error),
                    std::string(tc.label) +
                        " double mirror dispatch failed: " + error);
      ok &= Require(gpu_round_trip.Invalidate(&error),
                    std::string(tc.label) +
                        " round-trip invalidate failed: " + error);
      ok &= CompareU8Exact(ReadU8(gpu_round_trip), source,
                           std::string("Vulkan double mirror ") + tc.label);
      ok &= Require(counters.dispatch_calls == 2 &&
                        counters.runtime_failure_frames == 0,
                    std::string(tc.label) +
                        " mirror counters should track exactly two resident "
                        "dispatches");
    }

    {
      constexpr int src_w = 3;
      constexpr int src_h = 2;
      constexpr int dst_w = 5;
      constexpr int dst_h = 3;
      const std::vector<std::uint8_t> source = {
          5,  17, 29, 41, 53, 65, 77, 89,  101,
          13, 31, 47, 59, 71, 83, 97, 109, 127,
      };
      VulkanImage gpu_src, gpu_resized, gpu_dst;
      if (!AllocateU8(kernels, &gpu_src, src_w, src_h,
                      VulkanPixelFormat::rgb_u8, &error) ||
          !AllocateU8(kernels, &gpu_resized, dst_w, dst_h,
                      VulkanPixelFormat::rgb_u8, &error) ||
          !AllocateU8(kernels, &gpu_dst, dst_w, dst_h,
                      VulkanPixelFormat::rgb_u8, &error)) {
        return OptionalSkip("mirror resize-batch allocation failed: " + error);
      }
      FillU8(gpu_src, source);
      ok &= Require(gpu_src.Flush(&error),
                    "mirror resize-batch source flush failed: " + error);

      OpenVulkanMirror mirror;
      ok &= Require(mirror.EnsureInitialized(&kernels, dst_w, dst_h, &error),
                    "mirror resize-batch init should pass: " + error);
      OpenVulkanMirrorCounters counters;
      studiocast::video::OpenVulkanMirrorResizeFinalStageInput input;
      input.src = &gpu_src;
      input.resized = &gpu_resized;
      input.dst = &gpu_dst;
      input.unmirrored_analysis_complete = true;
      const std::uint64_t submissions_before =
          kernels.synchronous_submission_count();
      const std::uint64_t completions_before =
          kernels.frame_batch_completion_count();
      ok &= Require(mirror.ApplyResizeFinal(input, &counters, &error),
                    "Vulkan combined resize/mirror should dispatch: " + error);
      ok &= Require(kernels.synchronous_submission_count() ==
                            submissions_before + 1 &&
                        kernels.frame_batch_completion_count() ==
                            completions_before + 1 &&
                        kernels.last_frame_batch_stage_count() == 2,
                    "combined resize/mirror must record two dependent stages "
                    "under exactly one submission/completion");
      ok &= Require(counters.dispatch_calls == 1 &&
                        counters.runtime_failure_frames == 0,
                    "combined resize/mirror should count one final mirror "
                    "operation");
      ok &= Require(gpu_dst.Invalidate(&error),
                    "combined resize/mirror invalidate failed: " + error);
      ok &=
          CompareU8(ReadU8(gpu_dst),
                    MirrorU8Reference(
                        ResizeU8Reference(source, src_w, src_h, dst_w, dst_h,
                                          0.0f, 0.0f, static_cast<float>(src_w),
                                          static_cast<float>(src_h)),
                        dst_w, dst_h),
                    "Vulkan combined resize/mirror");
    }
  }

  {
    using studiocast::video::OpenVulkanVignette;
    using studiocast::video::OpenVulkanVignetteCounters;
    using studiocast::video::OpenVulkanVignetteFinalStageInput;

    const struct VignetteCase {
      int width;
      int height;
      int intensity;
      VulkanPixelFormat format;
      const char *label;
    } cases[] = {
        {1, 1, 100, VulkanPixelFormat::rgb_u8, "RGB 1x1 intensity 100"},
        {5, 3, 0, VulkanPixelFormat::rgb_u8, "RGB odd intensity 0"},
        {6, 4, 100, VulkanPixelFormat::rgb_u8, "RGB even intensity 100"},
        {21, 21, 100, VulkanPixelFormat::rgb_u8,
         "RGB near-center low-radial intensity 100"},
        {5, 3, 35, VulkanPixelFormat::bgr_u8, "BGR odd intensity 35"},
        {6, 4, 73, VulkanPixelFormat::bgr_u8, "BGR even intensity 73"},
    };

    // Reuse one helper across alternating geometries. This exercises mask
    // destruction/reallocation and possible opaque VkBuffer handle reuse.
    OpenVulkanVignette vignette;
    for (const auto &tc : cases) {
      std::vector<std::uint8_t> source(static_cast<std::size_t>(tc.width) *
                                       static_cast<std::size_t>(tc.height) *
                                       3u);
      for (std::size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<std::uint8_t>((i * 61u + 23u) & 0xffu);

      VulkanImage gpu_src, gpu_dst;
      if (!AllocateU8(kernels, &gpu_src, tc.width, tc.height, tc.format,
                      &error) ||
          !AllocateU8(kernels, &gpu_dst, tc.width, tc.height, tc.format,
                      &error)) {
        return OptionalSkip(std::string("vignette allocation failed: ") +
                            error);
      }
      FillU8(gpu_src, source);
      ok &= Require(gpu_src.Flush(&error),
                    std::string(tc.label) + " source flush failed: " + error);

      OpenVulkanVignetteCounters counters;
      ok &= Require(vignette.EnsureInitialized(&kernels, tc.width, tc.height,
                                               tc.intensity, &counters, &error),
                    std::string(tc.label) +
                        " vignette initialization failed: " + error);
      const auto setup_stats = kernels.device()->allocation_stats();
      ok &=
          Require(vignette.EnsureInitialized(&kernels, tc.width, tc.height,
                                             tc.intensity, &counters, &error) &&
                      counters.factor_allocation_calls == 1 &&
                      counters.factor_generation_calls == 1 &&
                      counters.factor_upload_calls == 1,
                  std::string(tc.label) +
                      " repeated initialization must reuse one attenuation "
                      "factor mask");
      ok &= Require(kernels.device()->allocation_stats().current_bytes ==
                            setup_stats.current_bytes &&
                        kernels.device()->allocation_stats().allocation_count ==
                            setup_stats.allocation_count,
                    std::string(tc.label) +
                        " repeated initialization must not allocate again");

      OpenVulkanVignetteFinalStageInput input;
      input.src = &gpu_src;
      input.dst = &gpu_dst;
      input.intensity_percent = tc.intensity;
      ok &= Require(!vignette.ApplyFinal(input, &counters, &error) &&
                        error.find("final-stage ordering boundary") !=
                            std::string::npos &&
                        counters.dispatch_calls == 0 &&
                        counters.runtime_failure_frames == 0,
                    std::string(tc.label) +
                        " must reject execution before the final boundary");
      input.preceding_effects_complete = true;

      const auto allocations_before_frames =
          kernels.device()->allocation_stats();
      const std::uint64_t descriptor_updates_before_frames =
          kernels.descriptor_binding_update_count();
      for (int frame = 0; frame < 2; ++frame) {
        const std::uint64_t submissions_before =
            kernels.synchronous_submission_count();
        ok &= Require(vignette.ApplyFinal(input, &counters, &error),
                      std::string(tc.label) +
                          " Vulkan vignette dispatch failed: " + error);
        ok &= Require(kernels.synchronous_submission_count() ==
                              submissions_before + 1 &&
                          kernels.last_frame_batch_stage_count() == 1,
                      std::string(tc.label) +
                          " must use one stage and one bounded completion");
        ok &= Require(gpu_dst.Invalidate(&error),
                      std::string(tc.label) +
                          " output invalidate failed: " + error);
        const auto padded_reference = VignetteU8ReferencePadded(
            source, tc.width, tc.height,
            static_cast<std::size_t>(tc.width) * 3u, tc.intensity);
        ok &= CompareU8(ReadU8(gpu_dst), padded_reference,
                        std::string("Vulkan vignette ") + tc.label);
      }
      const auto allocations_after_frames =
          kernels.device()->allocation_stats();
      ok &= Require(
          allocations_after_frames.current_bytes ==
                  allocations_before_frames.current_bytes &&
              allocations_after_frames.high_water_bytes ==
                  allocations_before_frames.high_water_bytes &&
              allocations_after_frames.allocation_count ==
                  allocations_before_frames.allocation_count &&
              counters.factor_allocation_calls == 1 &&
              counters.factor_generation_calls == 1 &&
              counters.factor_upload_calls == 1 &&
              counters.dispatch_calls == 2 &&
              kernels.descriptor_binding_update_count() ==
                  descriptor_updates_before_frames + 1,
          std::string(tc.label) +
              " repeated frames must not allocate, regenerate, or upload the "
              "attenuation mask or update stable descriptors");
    }

    // At 21x21 the neighbor of the exact center has radial^2 in (0, .02].
    // The legacy key-light opcode would skip it, while CUDA vignette darkens a
    // full-scale channel by at least one byte.
    const float near_center_radial =
        CudaVignetteRadialSquaredReference(11, 10, 21, 21, 10.5f, 10.5f);
    ok &=
        Require(near_center_radial > 0.0f && near_center_radial <= 0.02f &&
                    RoundClampByte(255.0f * (1.0f - near_center_radial)) < 255,
                "near-center fixture must detect the key-light alpha "
                "early-out mismatch");

    OpenVulkanVignetteCounters intensity_reconfigure_counters;
    const auto intensity_reconfigure_allocations =
        kernels.device()->allocation_stats();
    ok &= Require(
        vignette.EnsureInitialized(&kernels, 6, 4, 73,
                                   &intensity_reconfigure_counters, &error) &&
            intensity_reconfigure_counters.factor_allocation_calls == 0 &&
            intensity_reconfigure_counters.factor_generation_calls == 0 &&
            intensity_reconfigure_counters.factor_upload_calls == 0,
        "unchanged geometry/intensity must not touch setup data");
    ok &= Require(
        vignette.EnsureInitialized(&kernels, 6, 4, 74,
                                   &intensity_reconfigure_counters, &error) &&
            intensity_reconfigure_counters.factor_allocation_calls == 0 &&
            intensity_reconfigure_counters.factor_generation_calls == 1 &&
            intensity_reconfigure_counters.factor_upload_calls == 1,
        "intensity-only reconfigure must regenerate/upload once "
        "without allocating");
    ok &= Require(
        vignette.EnsureInitialized(&kernels, 6, 4, 74,
                                   &intensity_reconfigure_counters, &error) &&
            intensity_reconfigure_counters.factor_generation_calls == 1 &&
            intensity_reconfigure_counters.factor_upload_calls == 1 &&
            kernels.device()->allocation_stats().allocation_count ==
                intensity_reconfigure_allocations.allocation_count,
        "repeated reconfigure must reuse the attenuation buffer");

    {
      constexpr int src_w = 3;
      constexpr int src_h = 2;
      constexpr int dst_w = 5;
      constexpr int dst_h = 3;
      constexpr int intensity = 73;
      const std::vector<std::uint8_t> source = {
          5,  17, 29, 41, 53, 65, 77, 89,  101,
          13, 31, 47, 59, 71, 83, 97, 109, 127,
      };
      VulkanImage gpu_src, gpu_resized, gpu_vignette, gpu_mirror;
      if (!AllocateU8(kernels, &gpu_src, src_w, src_h,
                      VulkanPixelFormat::rgb_u8, &error) ||
          !AllocateU8(kernels, &gpu_resized, dst_w, dst_h,
                      VulkanPixelFormat::rgb_u8, &error) ||
          !AllocateU8(kernels, &gpu_vignette, dst_w, dst_h,
                      VulkanPixelFormat::rgb_u8, &error) ||
          !AllocateU8(kernels, &gpu_mirror, dst_w, dst_h,
                      VulkanPixelFormat::rgb_u8, &error)) {
        return OptionalSkip("combined vignette allocation failed: " + error);
      }
      FillU8(gpu_src, source);
      ok &= Require(gpu_src.Flush(&error),
                    "combined vignette source flush failed: " + error);

      OpenVulkanVignette combined_vignette;
      OpenVulkanVignetteCounters counters;
      ok &= Require(combined_vignette.EnsureInitialized(
                        &kernels, dst_w, dst_h, intensity, &counters, &error),
                    "combined vignette initialization failed: " + error);
      OpenVulkanVignetteFinalStageInput input;
      input.src = &gpu_src;
      input.resize_scratch = &gpu_resized;
      input.dst = &gpu_vignette;
      input.mirrored_dst = &gpu_mirror;
      input.preceding_effects_complete = true;
      input.intensity_percent = intensity;
      const std::uint64_t submissions_before =
          kernels.synchronous_submission_count();
      const std::uint64_t completions_before =
          kernels.frame_batch_completion_count();
      const auto allocations_before = kernels.device()->allocation_stats();
      ok &= Require(combined_vignette.ApplyFinal(input, &counters, &error),
                    "resize/vignette/mirror dispatch failed: " + error);
      ok &= Require(kernels.synchronous_submission_count() ==
                            submissions_before + 1 &&
                        kernels.frame_batch_completion_count() ==
                            completions_before + 1 &&
                        kernels.last_frame_batch_stage_count() == 3,
                    "resize/vignette/mirror must record three dependent "
                    "stages under one completion");
      ok &= Require(gpu_mirror.Invalidate(&error),
                    "combined vignette/mirror invalidate failed: " + error);
      const auto resized = ResizeU8Reference(
          source, src_w, src_h, dst_w, dst_h, 0.0f, 0.0f,
          static_cast<float>(src_w), static_cast<float>(src_h));
      const auto vignetted = VignetteU8ReferencePadded(
          resized, dst_w, dst_h, static_cast<std::size_t>(dst_w) * 3u,
          intensity);
      ok &= CompareU8(ReadU8(gpu_mirror),
                      MirrorU8Reference(vignetted, dst_w, dst_h),
                      "Vulkan resize/vignette/mirror parity");
      ok &= Require(kernels.device()->allocation_stats().current_bytes ==
                            allocations_before.current_bytes &&
                        kernels.device()->allocation_stats().high_water_bytes ==
                            allocations_before.high_water_bytes &&
                        counters.factor_allocation_calls == 1 &&
                        counters.factor_upload_calls == 1,
                    "combined frame must not allocate or re-upload its factor "
                    "mask");
    }
  }

  {
    UtilityKernels foreign_kernels;
    if (!foreign_kernels.Initialize(&error))
      return OptionalSkip("second Vulkan context init failed: " + error);
    VulkanImage same_src, same_dst, foreign_dst;
    if (!AllocateU8(kernels, &same_src, 2, 2, VulkanPixelFormat::rgb_u8,
                    &error) ||
        !AllocateU8(kernels, &same_dst, 2, 2, VulkanPixelFormat::rgb_u8,
                    &error) ||
        !AllocateU8(foreign_kernels, &foreign_dst, 2, 2,
                    VulkanPixelFormat::rgb_u8, &error)) {
      return OptionalSkip("context identity allocation failed: " + error);
    }
    ok &= Require(same_src.BelongsTo(*kernels.device()) &&
                      same_dst.BelongsTo(*kernels.device()),
                  "same-context Vulkan resources should be accepted");
    ok &= Require(!foreign_dst.BelongsTo(*kernels.device()),
                  "foreign Vulkan resource should carry a different context "
                  "generation");
    ok &=
        Require(!kernels.ResizeBilinear(same_src, foreign_dst, &error) &&
                    error.find("[vulkan_foreign_context]") != std::string::npos,
                "utility descriptor binding must reject a foreign context "
                "before dispatch");

    studiocast::video::OpenVulkanMirror mirror;
    ok &= Require(mirror.EnsureInitialized(&kernels, 2, 2, &error),
                  "same-context mirror wrapper should initialize: " + error);
    studiocast::video::OpenVulkanMirrorCounters alias_counters;
    studiocast::video::OpenVulkanMirrorFinalStageInput alias_input;
    alias_input.src = &same_src;
    alias_input.dst = &same_src;
    alias_input.unmirrored_analysis_complete = true;
    alias_input.output_geometry_ready = true;
    ok &= Require(!mirror.ApplyFinal(alias_input, &alias_counters, &error) &&
                      error.find("src/dst must be distinct") !=
                          std::string::npos &&
                      alias_counters.dispatch_calls == 0 &&
                      alias_counters.runtime_failure_frames == 1,
                  "production mirror helper must reject in-place aliasing "
                  "before dispatch");

    studiocast::video::OpenVulkanMirrorCounters mirror_counters;
    studiocast::video::OpenVulkanMirrorFinalStageInput mirror_input;
    mirror_input.src = &same_src;
    mirror_input.dst = &foreign_dst;
    mirror_input.unmirrored_analysis_complete = true;
    mirror_input.output_geometry_ready = true;
    ok &= Require(
        !mirror.ApplyFinal(mirror_input, &mirror_counters, &error) &&
            error.find("[vulkan_effect_runtime_failed]") != std::string::npos &&
            error.find("[vulkan_foreign_context]") != std::string::npos &&
            mirror_counters.dispatch_calls == 0 &&
            mirror_counters.runtime_failure_frames == 1,
        "production mirror helper must reject a foreign context before "
        "dispatch with stable effect/context reasons");

    studiocast::video::OpenVulkanVignette vignette;
    studiocast::video::OpenVulkanVignetteCounters vignette_counters;
    ok &= Require(vignette.EnsureInitialized(&kernels, 2, 2, 50,
                                             &vignette_counters, &error),
                  "same-context vignette wrapper should initialize: " + error);
    studiocast::video::OpenVulkanVignetteFinalStageInput vignette_input;
    vignette_input.src = &same_src;
    vignette_input.dst = &same_src;
    vignette_input.preceding_effects_complete = true;
    vignette_input.intensity_percent = 50;
    ok &= Require(
        !vignette.ApplyFinal(vignette_input, &vignette_counters, &error) &&
            error.find("stage images must be out-of-place") !=
                std::string::npos &&
            vignette_counters.dispatch_calls == 0 &&
            vignette_counters.runtime_failure_frames == 1,
        "production vignette helper must reject in-place aliasing before "
        "dispatch");

    vignette_input.dst = &foreign_dst;
    ok &= Require(
        !vignette.ApplyFinal(vignette_input, &vignette_counters, &error) &&
            error.find("[vulkan_effect_runtime_failed]") != std::string::npos &&
            error.find("[vulkan_foreign_context]") != std::string::npos &&
            vignette_counters.dispatch_calls == 0 &&
            vignette_counters.runtime_failure_frames == 2,
        "production vignette helper must reject a foreign context before "
        "dispatch with stable effect/context reasons");
  }

  {
    studiocast::vulkan::VulkanCommandBatch bounded_batch;
    ok &= Require(bounded_batch.Initialize(kernels.device(), 1, &error) &&
                      bounded_batch.Begin(&error) &&
                      bounded_batch.RecordStage("only_stage", &error),
                  "fixed-capacity Vulkan batch should begin and accept its "
                  "declared slot: " +
                      error);
    ok &= Require(!bounded_batch.RecordStage("overflow", &error) &&
                      error.find("vulkan_batch_capacity_exceeded") !=
                          std::string::npos,
                  "Vulkan batch must enforce bounded stage/slot lifetime");
    bool foreign_thread_accepted = true;
    std::thread foreign_thread([&] {
      std::string thread_error;
      foreign_thread_accepted =
          bounded_batch.RecordStage("foreign_thread", &thread_error);
    });
    foreign_thread.join();
    ok &= Require(!foreign_thread_accepted,
                  "Vulkan batch recording must enforce its serial owner");
    bounded_batch.Abort();
  }

  {
    using studiocast::vulkan::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    using studiocast::vulkan::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    using studiocast::vulkan::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    using studiocast::vulkan::VulkanBuffer;
    using studiocast::vulkan::VulkanDevice;
    using studiocast::vulkan::VulkanDeviceConfig;
    using studiocast::vulkan::VulkanDeviceSelection;

    VulkanDevice budget_device;
    VulkanDeviceConfig config;
    config.allocation_budget_bytes = 4096;
    if (!budget_device.Initialize(VulkanDeviceSelection{}, config, &error))
      return OptionalSkip("budgeted Vulkan context init failed: " + error);
    const auto initial = budget_device.allocation_stats();
    ok &=
        Require(initial.budget_bytes > 0 && initial.budget_bytes <= 4096 &&
                    initial.current_bytes == 0 && initial.allocation_count == 0,
                "Vulkan context should expose its derived/explicit empty "
                "allocation budget");

    VulkanBuffer tracked;
    ok &= Require(tracked.Allocate(&budget_device, 1024,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                   /*map_memory=*/true, &error),
                  "tracked Vulkan allocation should fit the explicit budget: " +
                      error);
    const auto allocated = budget_device.allocation_stats();
    ok &= Require(allocated.current_bytes >= 1024 &&
                      allocated.high_water_bytes == allocated.current_bytes &&
                      allocated.allocation_count == 1,
                  "Vulkan allocation counters should account actual memory "
                  "requirements");
    tracked.Free();
    const auto released = budget_device.allocation_stats();
    ok &= Require(released.current_bytes == 0 &&
                      released.high_water_bytes == allocated.high_water_bytes &&
                      released.allocation_count == 0,
                  "Vulkan allocation release should preserve only high-water "
                  "history");

    VulkanBuffer over_budget;
    ok &=
        Require(!over_budget.Allocate(&budget_device, released.budget_bytes + 1,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      /*map_memory=*/true, &error) &&
                    error.find("vulkan_allocation_budget_exceeded") !=
                        std::string::npos,
                "Vulkan allocation must fail before device memory allocation "
                "when its context budget is exceeded");
    studiocast::vulkan::VulkanTensorSize overflow_size;
    ok &= Require(
        !studiocast::vulkan::CheckedNchwF32Size(
            std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
            &overflow_size, &error) &&
            error.find("overflow") != std::string::npos,
        "Vulkan tensor size arithmetic must fail closed on "
        "overflow");

    VulkanBuffer retained_across_shutdown;
    ok &= Require(retained_across_shutdown.Allocate(
                      &budget_device, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      /*map_memory=*/true, &error),
                  "resource lifetime test allocation should succeed: " + error);
    const auto first_identity = budget_device.context_identity();
    budget_device.Shutdown();
    ok &= Require(!retained_across_shutdown.Valid(),
                  "resource should become unusable when its facade shuts down");
    ok &= Require(
        budget_device.Initialize(VulkanDeviceSelection{}, config, &error),
        "Vulkan device facade should support a new generation: " + error);
    ok &= Require(budget_device.context_identity().context_id ==
                          first_identity.context_id &&
                      budget_device.context_identity().generation >
                          first_identity.generation &&
                      !retained_across_shutdown.BelongsTo(budget_device),
                  "reinitialized facade must reject resources from its prior "
                  "context generation");
    retained_across_shutdown.Free();
  }

  {
    using studiocast::vulkan::VK_ERROR_DEVICE_LOST;
    using studiocast::vulkan::VK_TIMEOUT;
    using studiocast::vulkan::VulkanContextHealth;
    using studiocast::vulkan::VulkanDevice;
    using studiocast::vulkan::VulkanSubmissionPhase;

    VulkanDevice lost_device;
    if (!lost_device.Initialize(&error))
      return OptionalSkip("device-loss test context init failed: " + error);
    const auto fake_command =
        reinterpret_cast<studiocast::vulkan::VkCommandBuffer>(
            std::uintptr_t{1});
    lost_device.InjectNextSubmissionResultForTesting(
        VulkanSubmissionPhase::queue_submit, VK_ERROR_DEVICE_LOST);
    ok &= Require(!lost_device.SubmitAndWait(fake_command, &error) &&
                      error.find("[vulkan_device_lost]") != std::string::npos,
                  "injected Vulkan device loss should produce a stable reason");
    const auto lost = lost_device.health();
    ok &= Require(lost.health == VulkanContextHealth::device_lost &&
                      lost.poisoned && lost.reason_code == "vulkan_device_lost",
                  "Vulkan device loss should poison and latch the context");
    const std::uint64_t submitted_after_loss = lost.submitted_serial;
    ok &= Require(!lost_device.SubmitAndWait(fake_command, &error) &&
                      lost_device.health().submitted_serial ==
                          submitted_after_loss,
                  "poisoned Vulkan context must reject later frames without "
                  "retrying submission");

    VulkanDevice timeout_device;
    if (!timeout_device.Initialize(&error))
      return OptionalSkip("timeout test context init failed: " + error);
    timeout_device.InjectNextSubmissionResultForTesting(
        VulkanSubmissionPhase::wait_for_fence, VK_TIMEOUT);
    ok &= Require(!timeout_device.SubmitAndWait(fake_command, &error) &&
                      timeout_device.health().health ==
                          VulkanContextHealth::unsafe_timeout &&
                      timeout_device.health().reason_code ==
                          "vulkan_submission_timeout",
                  "unsafe Vulkan timeout should poison and latch with a stable "
                  "reason");
  }

  {
    using studiocast::vulkan::VK_ERROR_DEVICE_LOST;
    using studiocast::vulkan::VulkanSubmissionPhase;

    UtilityKernels lost_kernels;
    if (!lost_kernels.Initialize(&error))
      return OptionalSkip("mirror device-loss context init failed: " + error);
    VulkanImage src, dst;
    if (!AllocateU8(lost_kernels, &src, 2, 2, VulkanPixelFormat::rgb_u8,
                    &error) ||
        !AllocateU8(lost_kernels, &dst, 2, 2, VulkanPixelFormat::rgb_u8,
                    &error)) {
      return OptionalSkip("mirror device-loss allocation failed: " + error);
    }
    FillU8(src, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    if (!src.Flush(&error))
      return OptionalSkip("mirror device-loss source flush failed: " + error);

    studiocast::video::OpenVulkanMirror mirror;
    ok &= Require(mirror.EnsureInitialized(&lost_kernels, 2, 2, &error),
                  "mirror device-loss wrapper should initialize: " + error);
    lost_kernels.device()->InjectNextSubmissionResultForTesting(
        VulkanSubmissionPhase::queue_submit, VK_ERROR_DEVICE_LOST);
    studiocast::video::OpenVulkanMirrorCounters counters;
    studiocast::video::OpenVulkanMirrorFinalStageInput input;
    input.src = &src;
    input.dst = &dst;
    input.unmirrored_analysis_complete = true;
    input.output_geometry_ready = true;
    ok &= Require(
        !mirror.ApplyFinal(input, &counters, &error) &&
            error.find("[vulkan_effect_runtime_failed]") != std::string::npos &&
            error.find("[vulkan_device_lost]") != std::string::npos &&
            counters.dispatch_calls == 1 &&
            counters.runtime_failure_frames == 1,
        "mirror device loss must retain shared and effect stable reasons and "
        "count the failed submitted dispatch");
  }

  {
    using studiocast::vulkan::VK_ERROR_DEVICE_LOST;
    using studiocast::vulkan::VulkanSubmissionPhase;

    UtilityKernels lost_auto_frame_kernels;
    if (!lost_auto_frame_kernels.Initialize(&error)) {
      return OptionalSkip("Auto Frame device-loss context init failed: " +
                          error);
    }
    VulkanImage src, dst;
    if (!AllocateU8(lost_auto_frame_kernels, &src, 2, 2,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(lost_auto_frame_kernels, &dst, 2, 2,
                    VulkanPixelFormat::rgb_u8, &error)) {
      return OptionalSkip("Auto Frame device-loss allocation failed: " +
                          error);
    }
    FillU8(src, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    if (!src.Flush(&error)) {
      return OptionalSkip("Auto Frame device-loss source flush failed: " +
                          error);
    }

    studiocast::video::OpenVulkanAutoFrame auto_frame;
    studiocast::video::OpenVulkanAutoFrameCounters counters;
    ok &= Require(auto_frame.EnsureInitialized(&lost_auto_frame_kernels, 2, 2,
                                               &counters, &error),
                  "Auto Frame device-loss wrapper should initialize: " +
                      error);
    lost_auto_frame_kernels.device()->InjectNextSubmissionResultForTesting(
        VulkanSubmissionPhase::queue_submit, VK_ERROR_DEVICE_LOST);
    studiocast::video::OpenVulkanAutoFrameCropInput input;
    input.src = &src;
    input.dst = &dst;
    input.crop_x = 0.0f;
    input.crop_y = 0.0f;
    input.crop_w = 2.0f;
    input.crop_h = 2.0f;
    input.host_analysis_complete = true;
    input.cpu_crop_plan_complete = true;
    ok &= Require(
        !auto_frame.ApplyCrop(input, &counters, &error) &&
            error.find("[vulkan_effect_runtime_failed]") !=
                std::string::npos &&
            error.find("[vulkan_device_lost]") != std::string::npos &&
            counters.crop_resize_dispatch_calls == 1 &&
            counters.runtime_failure_frames == 1 &&
            counters.device_loss_frames == 1 &&
            counters.cpu_resize_fallback_calls == 0,
        "Auto Frame device loss must retain nested reasons, count the failed "
        "dispatch, and never substitute CPU crop/resize");
    const std::uint64_t submissions_after_loss =
        lost_auto_frame_kernels.synchronous_submission_count();
    ok &= Require(
        !auto_frame.ApplyCrop(input, &counters, &error) &&
            lost_auto_frame_kernels.synchronous_submission_count() ==
                submissions_after_loss,
        "poisoned Auto Frame shared context must reject without another "
        "submission");
  }

  {
    using studiocast::vulkan::VK_ERROR_DEVICE_LOST;
    using studiocast::vulkan::VulkanSubmissionPhase;

    UtilityKernels lost_batch_kernels;
    if (!lost_batch_kernels.Initialize(&error)) {
      return OptionalSkip("mirror batch device-loss context init failed: " +
                          error);
    }
    VulkanImage src, resized, dst;
    if (!AllocateU8(lost_batch_kernels, &src, 2, 2, VulkanPixelFormat::rgb_u8,
                    &error) ||
        !AllocateU8(lost_batch_kernels, &resized, 3, 3,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(lost_batch_kernels, &dst, 3, 3, VulkanPixelFormat::rgb_u8,
                    &error)) {
      return OptionalSkip("mirror batch device-loss allocation failed: " +
                          error);
    }
    FillU8(src, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    if (!src.Flush(&error)) {
      return OptionalSkip("mirror batch device-loss source flush failed: " +
                          error);
    }

    studiocast::video::OpenVulkanMirror mirror;
    ok &=
        Require(mirror.EnsureInitialized(&lost_batch_kernels, 3, 3, &error),
                "mirror batch device-loss wrapper should initialize: " + error);
    lost_batch_kernels.device()->InjectNextSubmissionResultForTesting(
        VulkanSubmissionPhase::queue_submit, VK_ERROR_DEVICE_LOST);
    studiocast::video::OpenVulkanMirrorCounters counters;
    studiocast::video::OpenVulkanMirrorResizeFinalStageInput input;
    input.src = &src;
    input.resized = &resized;
    input.dst = &dst;
    input.unmirrored_analysis_complete = true;
    ok &= Require(
        !mirror.ApplyResizeFinal(input, &counters, &error) &&
            error.find("[vulkan_effect_runtime_failed]") != std::string::npos &&
            error.find("[vulkan_device_lost]") != std::string::npos &&
            counters.dispatch_calls == 1 &&
            counters.runtime_failure_frames == 1,
        "combined resize/mirror device loss must retain shared and effect "
        "stable reasons and count the failed submitted batch");
  }

  {
    using studiocast::vulkan::VK_ERROR_DEVICE_LOST;
    using studiocast::vulkan::VulkanSubmissionPhase;

    UtilityKernels lost_vignette_kernels;
    if (!lost_vignette_kernels.Initialize(&error)) {
      return OptionalSkip("vignette device-loss context init failed: " + error);
    }
    VulkanImage src, resized, vignetted, mirrored;
    if (!AllocateU8(lost_vignette_kernels, &src, 2, 2,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(lost_vignette_kernels, &resized, 3, 3,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(lost_vignette_kernels, &vignetted, 3, 3,
                    VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(lost_vignette_kernels, &mirrored, 3, 3,
                    VulkanPixelFormat::rgb_u8, &error)) {
      return OptionalSkip("vignette device-loss allocation failed: " + error);
    }
    FillU8(src, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    if (!src.Flush(&error)) {
      return OptionalSkip("vignette device-loss source flush failed: " + error);
    }

    studiocast::video::OpenVulkanVignette vignette;
    studiocast::video::OpenVulkanVignetteCounters counters;
    ok &= Require(vignette.EnsureInitialized(&lost_vignette_kernels, 3, 3, 50,
                                             &counters, &error),
                  "vignette device-loss wrapper should initialize: " + error);
    lost_vignette_kernels.device()->InjectNextSubmissionResultForTesting(
        VulkanSubmissionPhase::queue_submit, VK_ERROR_DEVICE_LOST);
    studiocast::video::OpenVulkanVignetteFinalStageInput input;
    input.src = &src;
    input.resize_scratch = &resized;
    input.dst = &vignetted;
    input.mirrored_dst = &mirrored;
    input.preceding_effects_complete = true;
    input.intensity_percent = 50;
    ok &= Require(
        !vignette.ApplyFinal(input, &counters, &error) &&
            error.find("[vulkan_effect_runtime_failed]") != std::string::npos &&
            error.find("[vulkan_device_lost]") != std::string::npos &&
            counters.dispatch_calls == 1 &&
            counters.runtime_failure_frames == 1 &&
            lost_vignette_kernels.last_frame_batch_stage_count() == 3,
        "resize/vignette/mirror device loss must retain nested reasons, "
        "record the bounded chain, and count the failed submitted frame");
    const std::uint64_t submitted_after_loss =
        lost_vignette_kernels.synchronous_submission_count();
    const std::uint64_t generation_after_loss =
        counters.factor_generation_calls;
    const std::uint64_t upload_after_loss = counters.factor_upload_calls;
    ok &= Require(
        !vignette.EnsureInitialized(&lost_vignette_kernels, 3, 3, 50, &counters,
                                    &error) &&
            error.find("[vulkan_effect_initialization_failed]") !=
                std::string::npos &&
            error.find("[vulkan_device_lost]") != std::string::npos &&
            counters.factor_generation_calls == generation_after_loss &&
            counters.factor_upload_calls == upload_after_loss,
        "reusable vignette setup must re-evaluate and reject a poisoned "
        "context without regenerating configuration data");
    ok &= Require(!vignette.ApplyFinal(input, &counters, &error) &&
                      lost_vignette_kernels.synchronous_submission_count() ==
                          submitted_after_loss,
                  "poisoned vignette context must reject a later frame without "
                  "another submission");
  }

  auto flush = [&](auto &resource) {
    if (!resource.Flush(&error)) {
      std::cerr << "Flush failed: " << error << "\n";
      ok = false;
    }
  };
  auto invalidate = [&](auto &resource) {
    if (!resource.Invalidate(&error)) {
      std::cerr << "Invalidate failed: " << error << "\n";
      ok = false;
    }
  };

  {
    const int sw = 3, sh = 2, dw = 2, dh = 3;
    const std::vector<std::uint8_t> src = {
        10,  20,  30,  40,  50,  60,  70,  80,  90,
        110, 120, 130, 140, 150, 160, 170, 180, 190,
    };
    VulkanImage gpu_src, gpu_dst;
    if (!AllocateU8(kernels, &gpu_src, sw, sh, VulkanPixelFormat::rgb_u8,
                    &error) ||
        !AllocateU8(kernels, &gpu_dst, dw, dh, VulkanPixelFormat::rgb_u8,
                    &error)) {
      return OptionalSkip("mapped RGB image allocation failed: " + error);
    }
    FillU8(gpu_src, src);
    flush(gpu_src);
    ok &= Require(kernels.CropResizeBilinear(gpu_src, gpu_dst, 0.0f, 0.0f, 3.0f,
                                             2.0f, &error),
                  "Vulkan crop-resize should dispatch: " + error);
    invalidate(gpu_dst);
    ok &= CompareU8(
        ReadU8(gpu_dst),
        ResizeU8Reference(src, sw, sh, dw, dh, 0.0f, 0.0f, 3.0f, 2.0f),
        "Vulkan crop-resize");
  }

  {
    const int w = 3, h = 2;
    const std::vector<std::uint8_t> src = {
        10,  20,  30,  40,  50,  60,  70,  80,  90,
        110, 120, 130, 140, 150, 160, 170, 180, 190,
    };
    VulkanImage gpu_src, gpu_dst;
    if (!AllocateU8(kernels, &gpu_src, w, h, VulkanPixelFormat::rgb_u8,
                    &error) ||
        !AllocateU8(kernels, &gpu_dst, w, h, VulkanPixelFormat::rgb_u8,
                    &error)) {
      return OptionalSkip("Auto Frame crop allocation failed: " + error);
    }
    FillU8(gpu_src, src);
    flush(gpu_src);

    studiocast::video::OpenVulkanAutoFrame auto_frame;
    studiocast::video::OpenVulkanAutoFrameCounters counters;
    ok &= Require(auto_frame.EnsureInitialized(&kernels, w, h, &counters,
                                               &error),
                  "canonical Auto Frame crop should initialize on the shared "
                  "production context: " + error);
    studiocast::video::OpenVulkanAutoFrameCropInput input;
    input.src = &gpu_src;
    input.dst = &gpu_dst;
    input.crop_x = 0.5f;
    input.crop_y = 0.0f;
    input.crop_w = 2.0f;
    input.crop_h = 2.0f;

    studiocast::video::OpenVulkanAutoFrameCounters rejection_counters;
    const std::uint64_t submissions_before_rejections =
        kernels.synchronous_submission_count();
    ok &= Require(
        !auto_frame.ApplyCrop(input, &rejection_counters, &error) &&
            error.find("host-analysis/crop-plan ordering boundary") !=
                std::string::npos &&
            kernels.synchronous_submission_count() ==
                submissions_before_rejections &&
            rejection_counters.crop_resize_dispatch_calls == 0,
        "Auto Frame must reject crop dispatch before host analysis and CPU "
        "crop planning without a Vulkan submission");

    input.host_analysis_complete = true;
    input.cpu_crop_plan_complete = true;
    input.dst = &gpu_src;
    ok &= Require(
        !auto_frame.ApplyCrop(input, &rejection_counters, &error) &&
            error.find("must be out of place") != std::string::npos &&
            kernels.synchronous_submission_count() ==
                submissions_before_rejections &&
            rejection_counters.crop_resize_dispatch_calls == 0,
        "Auto Frame must reject in-place crop/resize without a Vulkan "
        "submission");

    UtilityKernels foreign_kernels;
    if (!foreign_kernels.Initialize(&error)) {
      return OptionalSkip("Auto Frame foreign-context init failed: " + error);
    }
    VulkanImage foreign_src;
    if (!AllocateU8(foreign_kernels, &foreign_src, w, h,
                    VulkanPixelFormat::rgb_u8, &error)) {
      return OptionalSkip("Auto Frame foreign-context allocation failed: " +
                          error);
    }
    const std::uint64_t foreign_submissions_before =
        foreign_kernels.synchronous_submission_count();
    input.src = &foreign_src;
    input.dst = &gpu_dst;
    ok &= Require(
        !auto_frame.ApplyCrop(input, &rejection_counters, &error) &&
            error.find("[vulkan_resource_foreign_context]") !=
                std::string::npos &&
            kernels.synchronous_submission_count() ==
                submissions_before_rejections &&
            foreign_kernels.synchronous_submission_count() ==
                foreign_submissions_before &&
            rejection_counters.crop_resize_dispatch_calls == 0,
        "Auto Frame must reject foreign-context resources with a stable "
        "reason and zero submissions on either context");

    input.src = &gpu_src;
    input.dst = &gpu_dst;
    ok &= Require(auto_frame.ApplyCrop(input, &counters, &error),
                  "canonical Auto Frame crop should dispatch: " + error);
    invalidate(gpu_dst);
    ok &= CompareU8(ReadU8(gpu_dst),
                    ResizeU8Reference(src, w, h, w, h, 0.5f, 0.0f, 2.0f,
                                      2.0f),
                    "canonical Vulkan Auto Frame crop parity");
    ok &= Require(counters.crop_resize_dispatch_calls == 1 &&
                      counters.runtime_failure_frames == 0 &&
                      counters.cpu_resize_fallback_calls == 0,
                  "successful Vulkan Auto Frame must count one real dispatch "
                  "and no CPU resize substitute");
  }

  {
    const int sw = 2, sh = 2;
    const std::vector<std::uint8_t> rgb = {
        10, 20, 30, 110, 120, 130, 210, 220, 230, 30, 40, 50,
    };
    VulkanImage gpu_src;
    VulkanTensor tensor;
    if (!AllocateU8(kernels, &gpu_src, sw, sh, VulkanPixelFormat::rgb_u8,
                    &error) ||
        !tensor.AllocateNchwF32(kernels.device(), 1, 3, 1, 1,
                                /*map_memory=*/true, &error)) {
      return OptionalSkip("mapped preprocess allocation failed: " + error);
    }
    FillU8(gpu_src, rgb);
    flush(gpu_src);
    ModelPreprocessSpec spec;
    spec.dst_w = 1;
    spec.dst_h = 1;
    spec.mean[0] = 0.5f;
    spec.mean[1] = 0.25f;
    spec.mean[2] = 0.0f;
    spec.std[0] = 0.5f;
    spec.std[1] = 0.25f;
    spec.std[2] = 1.0f;
    spec.dst_order = ChannelOrder::rgb;
    ok &= Require(kernels.PreprocessToTensor(gpu_src, tensor, spec, &error),
                  "Vulkan preprocess should dispatch: " + error);
    invalidate(tensor);
    ok &= CompareF32(ReadTensor(tensor),
                     PreprocessReference(rgb, sw, sh, false, spec),
                     "Vulkan preprocess RGB");
  }

  {
    const int sw = 2, sh = 2, dw = 3, dh = 3;
    const std::vector<float> alpha = {0.0f, 1.0f, 0.25f, 0.75f};
    VulkanImage gpu_src, gpu_dst;
    if (!AllocateF32(kernels, &gpu_src, sw, sh, &error) ||
        !AllocateF32(kernels, &gpu_dst, dw, dh, &error)) {
      return OptionalSkip("mapped alpha resize allocation failed: " + error);
    }
    FillF32(gpu_src, alpha);
    flush(gpu_src);
    ok &= Require(kernels.ResizeBilinearF32_1(gpu_src, gpu_dst, &error),
                  "Vulkan f32 resize should dispatch: " + error);
    invalidate(gpu_dst);
    ok &=
        CompareF32(ReadF32(gpu_dst), ResizeF32Reference(alpha, sw, sh, dw, dh),
                   "Vulkan alpha resize");
  }

  {
    const int sw = 2, sh = 2, dw = 3, dh = 3;
    const std::vector<float> alpha = {0.0f, 1.0f, 0.25f, 0.75f};
    VulkanImage upload, resident, readback;
    if (!AllocateF32(kernels, &upload, sw, sh, &error) ||
        !resident.Allocate(kernels.device(), dw, dh, VulkanPixelFormat::f32_1,
                           /*map_memory=*/false, &error) ||
        !readback.Allocate(kernels.device(), dw, dh, VulkanPixelFormat::f32_1,
                           /*map_memory=*/true, &error)) {
      return OptionalSkip("Vulkan alpha readback staging allocation failed: " +
                          error);
    }
    FillF32(upload, alpha);
    flush(upload);
    kernels.InvalidateDescriptorBindingCacheForSetup();
    ok &= Require(resident.device_local() && resident.mapped() == nullptr,
                  "production alpha must be non-mapped device-local memory");
    ok &= Require(kernels.ResizeBilinearF32_1(upload, resident, &error),
                  "Vulkan resident alpha setup should dispatch: " + error);
    const auto expected = ResizeF32Reference(alpha, sw, sh, dw, dh);
    std::vector<float> cpu(expected.size());
    const auto allocations_before =
        kernels.device()->allocation_stats().allocation_count;
    ok &= Require(kernels.ReadbackF32_1(resident, readback, cpu.data(),
                                       cpu.size(), &error),
                  "explicit staged alpha readback should succeed: " + error);
    ok &= CompareF32(cpu, expected, "Vulkan staged alpha readback");
    ok &= Require(kernels.ReadbackF32_1(resident, readback, cpu.data(),
                                       cpu.size(), &error) &&
                      kernels.device()->allocation_stats().allocation_count ==
                          allocations_before,
                  "repeated alpha readback must reuse bounded staging");
    ok &= Require(!kernels.ReadbackF32_1(upload, readback, cpu.data(),
                                        cpu.size(), &error),
                  "host-mapped inference alpha must not pass resident "
                  "readback source validation");

    UtilityKernels foreign_kernels;
    if (!foreign_kernels.Initialize(&error))
      return OptionalSkip("foreign Vulkan readback context unavailable: " +
                          error);
    VulkanImage foreign_staging;
    if (!foreign_staging.Allocate(foreign_kernels.device(), dw, dh,
                                  VulkanPixelFormat::f32_1,
                                  /*map_memory=*/true, &error)) {
      return OptionalSkip("foreign Vulkan staging allocation failed: " +
                          error);
    }
    const auto submissions_before_foreign =
        kernels.synchronous_submission_count();
    ok &= Require(!kernels.ReadbackF32_1(resident, foreign_staging, cpu.data(),
                                        cpu.size(), &error) &&
                      error.find("[vulkan_foreign_context]") !=
                          std::string::npos &&
                      kernels.synchronous_submission_count() ==
                          submissions_before_foreign,
                  "foreign readback staging must fail before submission");
    foreign_kernels.Shutdown();
    ok &= Require(!kernels.ReadbackF32_1(resident, foreign_staging, cpu.data(),
                                        cpu.size(), &error) &&
                      kernels.synchronous_submission_count() ==
                          submissions_before_foreign,
                  "stale readback staging must fail before submission");
  }

  {
    UtilityKernels loss_kernels;
    if (!loss_kernels.Initialize(&error))
      return OptionalSkip("device-loss readback context unavailable: " +
                          error);
    constexpr int w = 2;
    constexpr int h = 2;
    VulkanImage upload, resident, readback;
    if (!AllocateF32(loss_kernels, &upload, w, h, &error) ||
        !resident.Allocate(loss_kernels.device(), w, h,
                           VulkanPixelFormat::f32_1,
                           /*map_memory=*/false, &error) ||
        !readback.Allocate(loss_kernels.device(), w, h,
                           VulkanPixelFormat::f32_1,
                           /*map_memory=*/true, &error)) {
      return OptionalSkip("device-loss readback allocation failed: " + error);
    }
    const std::vector<float> alpha = {0.0f, 1.0f, 0.25f, 0.75f};
    FillF32(upload, alpha);
    flush(upload);
    loss_kernels.InvalidateDescriptorBindingCacheForSetup();
    ok &= Require(loss_kernels.ResizeBilinearF32_1(upload, resident, &error),
                  "device-loss readback setup should dispatch: " + error);
    std::vector<float> cpu(alpha.size());
    loss_kernels.device()->InjectNextSubmissionResultForTesting(
        studiocast::vulkan::VulkanSubmissionPhase::queue_submit,
        studiocast::vulkan::VK_ERROR_DEVICE_LOST);
    ok &= Require(!loss_kernels.ReadbackF32_1(
                      resident, readback, cpu.data(), cpu.size(), &error),
                  "device loss during staged readback must fail");
    const auto poisoned = loss_kernels.device()->health();
    ok &= Require(poisoned.poisoned &&
                      poisoned.health ==
                          studiocast::vulkan::VulkanContextHealth::device_lost,
                  "staged readback device loss must poison the context");
    const auto submitted_after_loss = poisoned.submitted_serial;
    ok &= Require(!loss_kernels.ReadbackF32_1(
                      resident, readback, cpu.data(), cpu.size(), &error) &&
                      loss_kernels.device()->health().submitted_serial ==
                          submitted_after_loss,
                  "poisoned readback context must reject retries before a "
                  "new driver submission");
  }

  {
    const int w = 3, h = 2, radius = 1;
    const std::vector<std::uint8_t> rgb = {
        0,  10,  20,  30,  40,  50,  60,  70,  80,
        90, 100, 110, 120, 130, 140, 150, 160, 170,
    };
    VulkanImage src, tmp, dst;
    if (!AllocateU8(kernels, &src, w, h, VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(kernels, &tmp, w, h, VulkanPixelFormat::rgb_u8, &error) ||
        !AllocateU8(kernels, &dst, w, h, VulkanPixelFormat::rgb_u8, &error)) {
      return OptionalSkip("mapped RGB blur allocation failed: " + error);
    }
    FillU8(src, rgb);
    flush(src);
    ok &= Require(kernels.BoxBlurSeparableU8x3(src, tmp, dst, radius, &error),
                  "Vulkan RGB blur should dispatch: " + error);
    invalidate(dst);
    ok &= CompareU8(ReadU8(dst), BlurU8Reference(rgb, w, h, radius),
                    "Vulkan RGB blur");
  }

  {
    const int w = 3, h = 2, radius = 1;
    const std::vector<std::uint8_t> fg = {
        0,  10,  20,  30,  40,  50,  60,  70,  80,
        90, 100, 110, 120, 130, 140, 150, 160, 170,
    };
    const std::vector<float> alpha = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 0.125f};
    VulkanImage gpu_fg, gpu_tmp, gpu_blurred, gpu_alpha, gpu_out;
    if (!AllocateU8(kernels, &gpu_fg, w, h, VulkanPixelFormat::rgb_u8,
                    &error) ||
        !AllocateU8(kernels, &gpu_tmp, w, h, VulkanPixelFormat::rgb_u8,
                    &error) ||
        !AllocateU8(kernels, &gpu_blurred, w, h, VulkanPixelFormat::rgb_u8,
                    &error) ||
        !AllocateF32(kernels, &gpu_alpha, w, h, &error) ||
        !AllocateU8(kernels, &gpu_out, w, h, VulkanPixelFormat::rgb_u8,
                    &error)) {
      return OptionalSkip("mapped batched blur/composite allocation failed: " +
                          error);
    }
    FillU8(gpu_fg, fg);
    FillF32(gpu_alpha, alpha);
    flush(gpu_fg);
    flush(gpu_alpha);
    const std::uint64_t submissions_before =
        kernels.synchronous_submission_count();
    ok &= Require(kernels.BoxBlurCompositeAlphaU8x3(gpu_fg, gpu_tmp,
                                                    gpu_blurred, gpu_alpha,
                                                    gpu_out, radius, &error),
                  "Vulkan batched blur/composite should dispatch: " + error);
    ok &= Require(kernels.synchronous_submission_count() ==
                      submissions_before + 1,
                  "Vulkan batched blur/composite should use one synchronous "
                  "submission");
    ok &= Require(kernels.last_frame_batch_stage_count() == 3 &&
                      kernels.frame_batch_completion_count() > 0,
                  "real Vulkan frame batch should record three utility stages "
                  "under one completion boundary");
    invalidate(gpu_out);
    ok &= CompareU8(
        ReadU8(gpu_out),
        CompositeReference(fg, BlurU8Reference(fg, w, h, radius), alpha),
        "Vulkan batched blur/composite");

    const std::uint64_t zero_radius_submissions_before =
        kernels.synchronous_submission_count();
    ok &= Require(
        kernels.BoxBlurCompositeAlphaU8x3(gpu_fg, gpu_tmp, gpu_blurred,
                                          gpu_alpha, gpu_out,
                                          /*radius=*/0, &error),
        "Vulkan zero-radius batched blur/composite should dispatch: " + error);
    ok &= Require(kernels.synchronous_submission_count() ==
                      zero_radius_submissions_before + 1,
                  "Vulkan zero-radius batched blur/composite should use one "
                  "synchronous submission");
    ok &= Require(kernels.last_frame_batch_stage_count() == 2,
                  "zero-radius real Vulkan batch should record copy and "
                  "composite stages");
    invalidate(gpu_out);
    ok &= CompareU8(ReadU8(gpu_out), fg,
                    "Vulkan zero-radius batched blur/composite");
  }

  {
    const int w = 3, h = 2, radius = 1;
    const std::vector<float> a = {0.0f, 0.25f, 1.0f, 0.5f, 0.75f, 0.25f};
    VulkanImage src, tmp, dst;
    if (!AllocateF32(kernels, &src, w, h, &error) ||
        !AllocateF32(kernels, &tmp, w, h, &error) ||
        !AllocateF32(kernels, &dst, w, h, &error)) {
      return OptionalSkip("mapped alpha blur allocation failed: " + error);
    }
    FillF32(src, a);
    flush(src);
    ok &= Require(kernels.BoxBlurSeparableF32_1(src, tmp, dst, radius, &error),
                  "Vulkan alpha blur should dispatch: " + error);
    invalidate(dst);
    ok &= CompareF32(ReadF32(dst), BlurF32Reference(a, w, h, radius),
                     "Vulkan alpha blur");
  }

  {
    const int w = 2, h = 2;
    const std::vector<std::uint8_t> fg = {
        10, 20, 30, 100, 110, 120, 200, 210, 220, 250, 5, 15,
    };
    const std::vector<std::uint8_t> bg = {
        1, 2, 3, 50, 60, 70, 80, 90, 100, 200, 210, 220,
    };
    const std::vector<float> alpha = {0.0f, 0.5f, 1.0f,
                                      std::numeric_limits<float>::quiet_NaN()};
    VulkanImage gpu_fg, gpu_bg, gpu_alpha, gpu_out;
    if (!AllocateU8(kernels, &gpu_fg, w, h, VulkanPixelFormat::rgb_u8,
                    &error) ||
        !AllocateU8(kernels, &gpu_bg, w, h, VulkanPixelFormat::rgb_u8,
                    &error) ||
        !AllocateF32(kernels, &gpu_alpha, w, h, &error) ||
        !AllocateU8(kernels, &gpu_out, w, h, VulkanPixelFormat::rgb_u8,
                    &error)) {
      return OptionalSkip("mapped composite allocation failed: " + error);
    }
    FillU8(gpu_fg, fg);
    FillU8(gpu_bg, bg);
    FillF32(gpu_alpha, alpha);
    flush(gpu_fg);
    flush(gpu_bg);
    flush(gpu_alpha);
    ok &= Require(
        kernels.CompositeAlphaU8x3(gpu_fg, gpu_bg, gpu_alpha, gpu_out, &error),
        "Vulkan composite background should dispatch: " + error);
    invalidate(gpu_out);
    ok &= CompareU8(ReadU8(gpu_out), CompositeReference(fg, bg, alpha),
                    "Vulkan composite background");
    ok &= Require(kernels.CompositeAlphaSolidU8x3(gpu_fg, gpu_alpha, 200, 40,
                                                  10, gpu_out, &error),
                  "Vulkan composite solid should dispatch: " + error);
    invalidate(gpu_out);
    ok &= CompareU8(ReadU8(gpu_out),
                    CompositeSolidReference(fg, alpha, {200, 40, 10}),
                    "Vulkan composite solid");
  }

  {
    const int w = 4, h = 1;
    const std::vector<std::uint8_t> src = {
        10, 20, 30, 100, 110, 120, 200, 210, 220, 50, 60, 70,
    };
    const std::vector<float> alpha = {0.0f, 0.5f, 1.0f,
                                      std::numeric_limits<float>::quiet_NaN()};
    VulkanImage gpu_src, gpu_alpha, gpu_out;
    if (!AllocateU8(kernels, &gpu_src, w, h, VulkanPixelFormat::rgb_u8,
                    &error) ||
        !AllocateF32(kernels, &gpu_alpha, w, h, &error) ||
        !AllocateU8(kernels, &gpu_out, w, h, VulkanPixelFormat::rgb_u8,
                    &error)) {
      return OptionalSkip("mapped key-light allocation failed: " + error);
    }
    FillU8(gpu_src, src);
    FillF32(gpu_alpha, alpha);
    flush(gpu_src);
    flush(gpu_alpha);
    ok &=
        Require(kernels.ApplyKeyLightU8x3(gpu_src, gpu_alpha, 255.0f, 242.0f,
                                          228.0f, 0.5f, 1.0f, gpu_out, &error),
                "Vulkan key-light should dispatch: " + error);
    invalidate(gpu_out);
    ok &= CompareU8(
        ReadU8(gpu_out),
        KeyLightReference(src, alpha, w, h, 255.0f, 242.0f, 228.0f, 0.5f, 1.0f),
        "Vulkan key-light");
  }

  return ok ? 0 : 1;
}
