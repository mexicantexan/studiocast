#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

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
  const float v = static_cast<float>(fg) * a +
                  static_cast<float>(bg) * (1.0f - a);
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
  const float src = crop_pos + (static_cast<float>(dst_i) + 0.5f) * scale -
                    0.5f;
  const int i0 = std::clamp(static_cast<int>(src), 0, src_len - 1);
  return {i0, std::clamp(i0 + 1, 0, src_len - 1),
          src - static_cast<float>(i0)};
}

AxisSample F32ReplicateSample(int src_len, int dst_len, int dst_i) {
  const float scale =
      static_cast<float>(src_len) / static_cast<float>(dst_len);
  const float src = std::clamp((static_cast<float>(dst_i) + 0.5f) * scale -
                                   0.5f,
                               0.0f, static_cast<float>(src_len - 1));
  const int i0 = static_cast<int>(std::floor(src));
  return {i0, std::clamp(i0 + 1, 0, src_len - 1),
          src - static_cast<float>(i0)};
}

std::vector<std::uint8_t> ResizeU8Reference(const std::vector<std::uint8_t> &src,
                                            int src_w, int src_h, int dst_w,
                                            int dst_h, float crop_x,
                                            float crop_y, float crop_w,
                                            float crop_h) {
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
        const float v0 = at(xs.i0, ys.i0) + xs.t * (at(xs.i1, ys.i0) -
                                                    at(xs.i0, ys.i0));
        const float v1 = at(xs.i0, ys.i1) + xs.t * (at(xs.i1, ys.i1) -
                                                    at(xs.i0, ys.i1));
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
      const float v0 = at(xs.i0, ys.i0) + xs.t * (at(xs.i1, ys.i0) -
                                                  at(xs.i0, ys.i0));
      const float v1 = at(xs.i0, ys.i1) + xs.t * (at(xs.i1, ys.i1) -
                                                  at(xs.i0, ys.i1));
      dst[PixelIndex(x, y, dst_w)] = v0 + ys.t * (v1 - v0);
    }
  }
  return dst;
}

std::vector<float> PreprocessReference(const std::vector<std::uint8_t> &src,
                                       int src_w, int src_h, bool src_bgr,
                                       const studiocast::vulkan::kernels::
                                           ModelPreprocessSpec &spec) {
  std::vector<float> out(3u * static_cast<std::size_t>(spec.dst_w) *
                         static_cast<std::size_t>(spec.dst_h));
  for (int y = 0; y < spec.dst_h; ++y) {
    const AxisSample ys = U8Sample(src_h, spec.dst_h, y, 0.0f,
                                  static_cast<float>(src_h));
    for (int x = 0; x < spec.dst_w; ++x) {
      const AxisSample xs = U8Sample(src_w, spec.dst_w, x, 0.0f,
                                    static_cast<float>(src_w));
      float rgb[3] = {};
      for (int semantic = 0; semantic < 3; ++semantic) {
        const int mem_c =
            semantic == 1 ? 1 : (src_bgr ? (2 - semantic) : semantic);
        const auto at = [&](int sx, int sy) {
          return static_cast<float>(src[ChannelIndex(sx, sy, src_w, mem_c)]);
        };
        const float v0 = at(xs.i0, ys.i0) + xs.t * (at(xs.i1, ys.i0) -
                                                    at(xs.i0, ys.i0));
        const float v1 = at(xs.i0, ys.i1) + xs.t * (at(xs.i1, ys.i1) -
                                                    at(xs.i0, ys.i1));
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
      const std::size_t hw =
          static_cast<std::size_t>(spec.dst_w) *
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

std::vector<std::uint8_t> CompositeReference(
    const std::vector<std::uint8_t> &fg, const std::vector<std::uint8_t> &bg,
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

std::vector<std::uint8_t> KeyLightReference(
    const std::vector<std::uint8_t> &src, const std::vector<float> &alpha,
    int w, int h, float target_r, float target_g, float target_b,
    float intensity, float direction) {
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
    const int diff = std::abs(static_cast<int>(actual[i]) -
                              static_cast<int>(expected[i]));
    if (diff > 1) {
      std::cerr << what << ": byte mismatch at " << i << " got "
                << static_cast<int>(actual[i]) << " expected "
                << static_cast<int>(expected[i]) << "\n";
      return false;
    }
  }
  return ok;
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
  ok &= Require(
      intelStableId == "v1:8086:1234:1:intel-arc-integrated" &&
          studiocast::vulkan::detail::IsValidVulkanDeviceStableId(
              intelStableId),
      "stable Vulkan identity should be deterministic and config-safe");
  candidates = {
      DeviceCandidate(7, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 400, 0,
                      "AMD GPU"),
      DeviceCandidate(2, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 300, 0,
                      "Intel Arc Integrated")};
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
  ok &= Require(!selected.ok &&
                    selected.failure_reason ==
                        "vulkan_requested_device_not_found",
                "a missing saved Vulkan identity must fail closed");

  candidates = {
      DeviceCandidate(0, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 300, 0,
                      "Identical GPU"),
      DeviceCandidate(1, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 300, 0,
                      "Identical GPU")};
  const std::string ambiguousId =
      studiocast::vulkan::detail::MakeVulkanDeviceStableId(
          0, 0, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, "Identical GPU");
  selection.requested_stable_id = ambiguousId;
  selected = studiocast::vulkan::detail::SelectVulkanDeviceCandidate(
      &candidates, selection);
  ok &= Require(!selected.ok &&
                    selected.failure_reason ==
                        "vulkan_requested_device_ambiguous",
                "indistinguishable saved Vulkan identities must fail closed");

  studiocast::vulkan::OpenVulkanDiagnostics diagnostics;
  diagnostics.device_selection_source = "STUDIOCAST_VULKAN_DEVICE_INDEX";
  diagnostics.device_selection_request = "index:1";
  diagnostics.selected_device_index = 1;
  diagnostics.selected_device_stable_id = intelStableId;
  diagnostics.device_candidates = {
      DeviceCandidate(1, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 300, 0,
                      "Intel GPU")};
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

  UtilityKernels validation_only;
  VulkanImage invalid;
  ok &= Require(!validation_only.ResizeBilinear(invalid, invalid, &error),
                "Vulkan resize wrapper should reject invalid images before "
                "runtime init");
  ok &= Require(error.find("invalid Vulkan image") != std::string::npos,
                "Vulkan invalid image error should be clear");
  ok &= Require(studiocast::vulkan::kernels::detail::
                    CheckBoxBlurRadiusForKernel(64, "test", &error),
                "Vulkan box blur should accept radius 64");
  ok &= Require(!studiocast::vulkan::kernels::detail::
                    CheckBoxBlurRadiusForKernel(65, "test", &error),
                "Vulkan box blur should reject radius above 64");
  ok &= Require(error.find("maximum supported radius 64") != std::string::npos,
                "Vulkan radius error should name the limit");
  studiocast::vulkan::VulkanTensorSize size;
  ok &= Require(studiocast::vulkan::CheckedNchwF32Size(1, 3, 2, 2, &size,
                                                       &error),
                "Vulkan tensor valid shape should pass");
  ok &= Require(size.elements == 12 && size.bytes == 12 * sizeof(float),
                "Vulkan tensor size should be contiguous NCHW f32");
  ok &= Require(!studiocast::vulkan::CheckedNchwF32Size(1, 0, 2, 2, &size,
                                                        &error),
                "Vulkan tensor invalid shape should fail");
  if (!ok)
    return 1;

  UtilityKernels kernels;
  if (!kernels.Initialize(&error))
    return OptionalSkip(error);

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
        10, 20, 30, 40, 50, 60, 70, 80, 90,
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
    ok &= Require(kernels.CropResizeBilinear(gpu_src, gpu_dst, 0.0f, 0.0f,
                                             3.0f, 2.0f, &error),
                  "Vulkan crop-resize should dispatch: " + error);
    invalidate(gpu_dst);
    ok &= CompareU8(ReadU8(gpu_dst),
                    ResizeU8Reference(src, sw, sh, dw, dh, 0.0f, 0.0f, 3.0f,
                                      2.0f),
                    "Vulkan crop-resize");
  }

  {
    const int sw = 2, sh = 2;
    const std::vector<std::uint8_t> rgb = {
        10, 20, 30, 110, 120, 130,
        210, 220, 230, 30, 40, 50,
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
    ok &= CompareF32(ReadF32(gpu_dst),
                     ResizeF32Reference(alpha, sw, sh, dw, dh),
                     "Vulkan alpha resize");
  }

  {
    const int w = 3, h = 2, radius = 1;
    const std::vector<std::uint8_t> rgb = {
        0, 10, 20, 30, 40, 50, 60, 70, 80,
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
    invalidate(gpu_out);
    ok &= CompareU8(
        ReadU8(gpu_out),
        CompositeReference(fg, BlurU8Reference(fg, w, h, radius), alpha),
        "Vulkan batched blur/composite");

    const std::uint64_t zero_radius_submissions_before =
        kernels.synchronous_submission_count();
    ok &= Require(kernels.BoxBlurCompositeAlphaU8x3(
                      gpu_fg, gpu_tmp, gpu_blurred, gpu_alpha, gpu_out,
                      /*radius=*/0, &error),
                  "Vulkan zero-radius batched blur/composite should dispatch: " +
                      error);
    ok &= Require(kernels.synchronous_submission_count() ==
                      zero_radius_submissions_before + 1,
                  "Vulkan zero-radius batched blur/composite should use one "
                  "synchronous submission");
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
        10, 20, 30, 100, 110, 120,
        200, 210, 220, 250, 5, 15,
    };
    const std::vector<std::uint8_t> bg = {
        1, 2, 3, 50, 60, 70,
        80, 90, 100, 200, 210, 220,
    };
    const std::vector<float> alpha = {
        0.0f, 0.5f, 1.0f, std::numeric_limits<float>::quiet_NaN()};
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
    ok &= Require(kernels.CompositeAlphaU8x3(gpu_fg, gpu_bg, gpu_alpha,
                                             gpu_out, &error),
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
        10, 20, 30, 100, 110, 120,
        200, 210, 220, 50, 60, 70,
    };
    const std::vector<float> alpha = {
        0.0f, 0.5f, 1.0f, std::numeric_limits<float>::quiet_NaN()};
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
    ok &= Require(kernels.ApplyKeyLightU8x3(gpu_src, gpu_alpha, 255.0f,
                                            242.0f, 228.0f, 0.5f, 1.0f,
                                            gpu_out, &error),
                  "Vulkan key-light should dispatch: " + error);
    invalidate(gpu_out);
    ok &= CompareU8(ReadU8(gpu_out),
                    KeyLightReference(src, alpha, w, h, 255.0f, 242.0f,
                                      228.0f, 0.5f, 1.0f),
                    "Vulkan key-light");
  }

  return ok ? 0 : 1;
}
