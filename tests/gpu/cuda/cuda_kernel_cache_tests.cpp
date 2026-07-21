#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "core/cuda/kernels/cuda_driver_cache.h"
#include "core/cuda/kernels/open_cuda_vb_kernels.h"
#include "core/cuda/kernels/preprocess_to_nchw.h"
#include "core/cuda/kernels/ptx_pitch_contract.h"
#include "core/cuda/kernels/resize_bilinear.h"

namespace {

bool Require(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

bool Contains(const std::string &s, const std::string &needle) {
  return s.find(needle) != std::string::npos;
}

struct AxisSampleF32 {
  int i0 = 0;
  int i1 = 0;
  float t = 0.0f;
};

AxisSampleF32 ReplicateHalfPixelSampleF32(int src_len, int dst_len,
                                          int dst_i) {
  const float scale =
      static_cast<float>(src_len) / static_cast<float>(dst_len);
  const float src =
      (static_cast<float>(dst_i) + 0.5f) * scale - 0.5f;
  const float clamped =
      std::clamp(src, 0.0f, static_cast<float>(src_len - 1));
  const int i0 = static_cast<int>(std::floor(clamped));
  return AxisSampleF32{i0, std::min(i0 + 1, src_len - 1),
                       clamped - static_cast<float>(i0)};
}

std::vector<float> ResizeF32ReplicateReference(const std::vector<float> &src,
                                               int src_w, int src_h, int dst_w,
                                               int dst_h) {
  std::vector<float> dst(static_cast<std::size_t>(dst_w) *
                         static_cast<std::size_t>(dst_h));
  for (int y = 0; y < dst_h; ++y) {
    const AxisSampleF32 ys =
        ReplicateHalfPixelSampleF32(src_h, dst_h, y);
    for (int x = 0; x < dst_w; ++x) {
      const AxisSampleF32 xs =
          ReplicateHalfPixelSampleF32(src_w, dst_w, x);
      const float v00 = src[static_cast<std::size_t>(ys.i0) *
                                static_cast<std::size_t>(src_w) +
                            static_cast<std::size_t>(xs.i0)];
      const float v10 = src[static_cast<std::size_t>(ys.i0) *
                                static_cast<std::size_t>(src_w) +
                            static_cast<std::size_t>(xs.i1)];
      const float v01 = src[static_cast<std::size_t>(ys.i1) *
                                static_cast<std::size_t>(src_w) +
                            static_cast<std::size_t>(xs.i0)];
      const float v11 = src[static_cast<std::size_t>(ys.i1) *
                                static_cast<std::size_t>(src_w) +
                            static_cast<std::size_t>(xs.i1)];
      const float v0 = v00 + (v10 - v00) * xs.t;
      const float v1 = v01 + (v11 - v01) * xs.t;
      dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_w) +
          static_cast<std::size_t>(x)] = v0 + (v1 - v0) * ys.t;
    }
  }
  return dst;
}

bool NearlyEqual(float a, float b) {
  return std::abs(a - b) <= 1.0e-6f;
}

std::uint8_t BlendByteReference(std::uint8_t fg, std::uint8_t bg,
                                float alpha) {
  const float a =
      studiocast::cuda::kernels::detail::ClampAlpha01ForComposite(alpha);
  const float ia = 1.0f - a;
  float v = static_cast<float>(fg) * a + static_cast<float>(bg) * ia;
  v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
  return static_cast<std::uint8_t>(v + 0.5f);
}

std::array<std::uint8_t, 3>
CompositeSolidReference(studiocast::cuda::PixelFormatGpu format,
                        const std::array<std::uint8_t, 3> &fg, float alpha,
                        std::uint8_t bg_r, std::uint8_t bg_g,
                        std::uint8_t bg_b) {
  const auto bg = studiocast::cuda::kernels::detail::
      SolidBackgroundMemoryChannels(format, bg_r, bg_g, bg_b);
  return {BlendByteReference(fg[0], bg[0], alpha),
          BlendByteReference(fg[1], bg[1], alpha),
          BlendByteReference(fg[2], bg[2], alpha)};
}

std::uint8_t InterpolateAndRoundResizeU8(std::uint8_t p00, std::uint8_t p10,
                                         std::uint8_t p01, std::uint8_t p11,
                                         float tx, float ty) {
  const float v00 = static_cast<float>(p00);
  const float v10 = static_cast<float>(p10);
  const float v01 = static_cast<float>(p01);
  const float v11 = static_cast<float>(p11);
  const float v0 = v00 + (v10 - v00) * tx;
  const float v1 = v01 + (v11 - v01) * tx;
  return studiocast::cuda::kernels::detail::RoundClampResizeBilinearU8(
      v0 + (v1 - v0) * ty);
}

} // namespace

namespace studiocast::tests {

bool TestCudaContextClassifierPrefersCurrentContext() {
  using studiocast::cuda::kernels::detail::ClassifyCurrentContext;
  using studiocast::cuda::kernels::detail::CurrentContextState;

  const auto fake_context = reinterpret_cast<studiocast::maxine::CUcontext>(
      static_cast<std::uintptr_t>(0x1));

  bool ok = true;
  ok &= Require(ClassifyCurrentContext(studiocast::maxine::CUDA_SUCCESS,
                                       fake_context) ==
                    CurrentContextState::current,
                "CUDA context classifier should preserve a current context");
  ok &= Require(ClassifyCurrentContext(studiocast::maxine::CUDA_SUCCESS,
                                       nullptr) == CurrentContextState::none,
                "CUDA context classifier should distinguish no current context");
  ok &= Require(ClassifyCurrentContext(/*result=*/1, fake_context) ==
                    CurrentContextState::error,
                "CUDA context classifier should surface driver errors");
  return ok;
}

bool TestSignedInt32PtxPitchContractIsNoGpuSafe() {
  using studiocast::cuda::CudaImage;
  using studiocast::cuda::CudaTensor;
  using studiocast::cuda::PixelFormatGpu;
  using studiocast::cuda::kernels::ChannelOrder;
  using studiocast::cuda::kernels::ModelPreprocessSpec;
  using studiocast::cuda::kernels::detail::CheckSignedInt32PtxPitch;
  using studiocast::cuda::kernels::detail::kSignedInt32PtxPitchMaxBytes;

  bool ok = true;
  std::string error;

  ok &= Require(CheckSignedInt32PtxPitch(kSignedInt32PtxPitchMaxBytes,
                                         kSignedInt32PtxPitchMaxBytes,
                                         "boundary", &error),
                "signed PTX pitch contract should accept INT32_MAX pitch");
  ok &= Require(error.empty(),
                "successful signed PTX pitch validation should not set errors");

  ok &= Require(!CheckSignedInt32PtxPitch(kSignedInt32PtxPitchMaxBytes + 1u,
                                          1u, "too large", &error),
                "signed PTX pitch contract should reject INT32_MAX+1");
  ok &= Require(Contains(error, "signed 32-bit PTX pitch ABI"),
                "oversize pitch error should document signed PTX ABI");

  ok &= Require(!CheckSignedInt32PtxPitch(11u, 12u, "too small", &error),
                "signed PTX pitch contract should reject pitch < row bytes");
  ok &= Require(Contains(error, "row bytes"),
                "short pitch error should mention row bytes");

  CudaImage rgb_src;
  rgb_src.ptr = 0x1000;
  rgb_src.pitch = kSignedInt32PtxPitchMaxBytes + 1u;
  rgb_src.w = 1;
  rgb_src.h = 1;
  rgb_src.format = PixelFormatGpu::rgb_u8;

  CudaImage rgb_dst;
  rgb_dst.ptr = 0x2000;
  rgb_dst.pitch = 3u;
  rgb_dst.w = 1;
  rgb_dst.h = 1;
  rgb_dst.format = PixelFormatGpu::rgb_u8;

  ok &= Require(!studiocast::cuda::kernels::CropResizeBilinear(
                    rgb_src, rgb_dst, 0.0f, 0.0f, 1.0f, 1.0f,
                    /*stream=*/nullptr, &error),
                "CropResizeBilinear should reject oversized pitch before CUDA");
  ok &= Require(Contains(error, "signed 32-bit PTX pitch ABI"),
                "CropResizeBilinear rejection should cite signed PTX ABI");

  CudaTensor tensor;
  tensor.ptr = 0x3000;
  tensor.pitch = 3u * sizeof(float);
  tensor.bytes = 3u * sizeof(float);
  tensor.n = 1;
  tensor.c = 3;
  tensor.h = 1;
  tensor.w = 1;

  ModelPreprocessSpec spec;
  spec.dst_w = 1;
  spec.dst_h = 1;
  spec.dst_order = ChannelOrder::rgb;
  ok &= Require(!studiocast::cuda::kernels::PreprocessToTensor(
                    rgb_src, tensor, spec, /*stream=*/nullptr, &error),
                "PreprocessToTensor should reject oversized pitch before CUDA");
  ok &= Require(Contains(error, "signed 32-bit PTX pitch ABI"),
                "PreprocessToTensor rejection should cite signed PTX ABI");

  CudaImage f32_src;
  f32_src.ptr = 0x4000;
  f32_src.pitch = kSignedInt32PtxPitchMaxBytes + 1u;
  f32_src.w = 1;
  f32_src.h = 1;
  f32_src.format = PixelFormatGpu::f32_1;

  CudaImage f32_dst;
  f32_dst.ptr = 0x5000;
  f32_dst.pitch = sizeof(float);
  f32_dst.w = 1;
  f32_dst.h = 1;
  f32_dst.format = PixelFormatGpu::f32_1;

  ok &= Require(!studiocast::cuda::kernels::ResizeBilinearF32_1(
                    f32_src, f32_dst, /*stream=*/nullptr, &error),
                "Open CUDA VB resize should reject oversized pitch before CUDA");
  ok &= Require(Contains(error, "signed 32-bit PTX pitch ABI"),
                "Open CUDA VB resize rejection should cite signed PTX ABI");

  return ok;
}

bool TestOpenCudaBoxBlurRadiusBoundsAreNoGpuSafe() {
  using studiocast::cuda::CudaImage;
  using studiocast::cuda::PixelFormatGpu;
  using studiocast::cuda::kernels::detail::CheckBoxBlurRadiusForKernel;
  using studiocast::cuda::kernels::detail::kBoxBlurMaxRadius;
  using studiocast::cuda::kernels::detail::NormalizeBoxBlurRadius;

  bool ok = true;
  std::string error;

  ok &= Require(CheckBoxBlurRadiusForKernel(kBoxBlurMaxRadius, "max", &error),
                "Open CUDA box blur should accept its maximum radius");
  ok &= Require(error.empty(),
                "accepted Open CUDA box blur radius should not set an error");

  ok &= Require(CheckBoxBlurRadiusForKernel(0, "zero", &error),
                "Open CUDA box blur should accept radius zero");
  ok &= Require(NormalizeBoxBlurRadius(-7) == 0,
                "Open CUDA box blur should normalize negative radii to no-op");
  ok &= Require(CheckBoxBlurRadiusForKernel(-7, "negative", &error),
                "Open CUDA box blur should accept negative radii as no-op");

  ok &= Require(!CheckBoxBlurRadiusForKernel(kBoxBlurMaxRadius + 1,
                                             "too large", &error),
                "Open CUDA box blur should reject radius above max");
  ok &= Require(Contains(error, "maximum supported radius 64"),
                "Open CUDA box blur max-radius error should name the limit");

  CudaImage u8_src;
  u8_src.ptr = 0x1000;
  u8_src.pitch = 6u;
  u8_src.w = 2;
  u8_src.h = 2;
  u8_src.format = PixelFormatGpu::rgb_u8;

  CudaImage u8_tmp;
  u8_tmp.ptr = 0x2000;
  u8_tmp.pitch = 6u;
  u8_tmp.w = 2;
  u8_tmp.h = 2;
  u8_tmp.format = PixelFormatGpu::rgb_u8;

  CudaImage u8_dst;
  u8_dst.ptr = 0x3000;
  u8_dst.pitch = 6u;
  u8_dst.w = 2;
  u8_dst.h = 2;
  u8_dst.format = PixelFormatGpu::rgb_u8;

  error.clear();
  ok &= Require(!studiocast::cuda::kernels::BoxBlurSeparableU8x3(
                    u8_src, u8_tmp, u8_dst, kBoxBlurMaxRadius + 1,
                    /*stream=*/nullptr, &error),
                "BoxBlurSeparableU8x3 should reject oversized radius before "
                "CUDA");
  ok &= Require(Contains(error, "BoxBlurSeparableU8x3") &&
                    Contains(error, "maximum supported radius 64"),
                "BoxBlurSeparableU8x3 radius error should be clear");

  CudaImage f32_src;
  f32_src.ptr = 0x4000;
  f32_src.pitch = 2u * sizeof(float);
  f32_src.w = 2;
  f32_src.h = 2;
  f32_src.format = PixelFormatGpu::f32_1;

  CudaImage f32_tmp;
  f32_tmp.ptr = 0x5000;
  f32_tmp.pitch = 2u * sizeof(float);
  f32_tmp.w = 2;
  f32_tmp.h = 2;
  f32_tmp.format = PixelFormatGpu::f32_1;

  CudaImage f32_dst;
  f32_dst.ptr = 0x6000;
  f32_dst.pitch = 2u * sizeof(float);
  f32_dst.w = 2;
  f32_dst.h = 2;
  f32_dst.format = PixelFormatGpu::f32_1;

  error.clear();
  ok &= Require(!studiocast::cuda::kernels::BoxBlurSeparableF32_1(
                    f32_src, f32_tmp, f32_dst, kBoxBlurMaxRadius + 1,
                    /*stream=*/nullptr, &error),
                "BoxBlurSeparableF32_1 should reject oversized radius before "
                "CUDA");
  ok &= Require(Contains(error, "BoxBlurSeparableF32_1") &&
                    Contains(error, "maximum supported radius 64"),
                "BoxBlurSeparableF32_1 radius error should be clear");

  return ok;
}

bool TestOpenCudaF32ResizeBorderContractIsReplicateNoGpuSafe() {
  bool ok = true;

  {
    const std::vector<float> src{
        0.0f, 10.0f,
        100.0f, 110.0f,
    };
    const std::array<float, 16> expected{{
        0.0f, 2.5f, 7.5f, 10.0f,
        25.0f, 27.5f, 32.5f, 35.0f,
        75.0f, 77.5f, 82.5f, 85.0f,
        100.0f, 102.5f, 107.5f, 110.0f,
    }};
    const std::vector<float> actual =
        ResizeF32ReplicateReference(src, 2, 2, 4, 4);

    ok &= Require(actual.size() == expected.size(),
                  "f32 resize border reference produced unexpected size");
    for (std::size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
      if (!NearlyEqual(actual[i], expected[i])) {
        ok &= Require(false, "f32 resize replicate border mismatch");
        break;
      }
    }
  }

  {
    const std::vector<float> src{1.0f, 4.0f, 9.0f};
    const std::array<float, 20> expected{{
        1.0f, 1.0f, 1.0f, 1.0f,
        2.2f, 2.2f, 2.2f, 2.2f,
        4.0f, 4.0f, 4.0f, 4.0f,
        7.0f, 7.0f, 7.0f, 7.0f,
        9.0f, 9.0f, 9.0f, 9.0f,
    }};
    const std::vector<float> actual =
        ResizeF32ReplicateReference(src, 1, 3, 4, 5);

    ok &= Require(actual.size() == expected.size(),
                  "f32 resize degenerate-axis reference produced unexpected "
                  "size");
    for (std::size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
      if (!NearlyEqual(actual[i], expected[i])) {
        ok &= Require(false,
                      "f32 resize degenerate-axis replicate border mismatch");
        break;
      }
    }
  }

  return ok;
}

bool TestCudaU8ResizeRoundingContractIsNoGpuSafe() {
  using studiocast::cuda::kernels::detail::RoundClampResizeBilinearU8;

  struct RoundCase {
    float value;
    std::uint8_t expected;
    const char *name;
  };

  const RoundCase round_cases[] = {
      {-0.25f, 0, "negative clamps before rounding"},
      {0.0f, 0, "zero remains zero"},
      {0.49f, 0, "near tie below half rounds down"},
      {0.5f, 1, "half tie rounds up"},
      {1.49f, 1, "nonzero near tie below half rounds down"},
      {1.5f, 2, "nonzero half tie rounds up"},
      {41.0f, 41, "integer value remains unchanged"},
      {42.25f, 42, "ordinary fractional value rounds down"},
      {42.75f, 43, "ordinary fractional value rounds up"},
      {127.5f, 128, "midrange half tie rounds up"},
      {254.49f, 254, "upper near tie below half rounds down"},
      {254.5f, 255, "upper half tie rounds up"},
      {255.0f, 255, "byte max remains max"},
      {300.0f, 255, "above byte max clamps"},
  };

  bool ok = true;
  for (const RoundCase &tc : round_cases) {
    const std::uint8_t actual = RoundClampResizeBilinearU8(tc.value);
    if (actual != tc.expected) {
      std::cerr << "u8 resize rounding mismatch for " << tc.name << ": got "
                << static_cast<int>(actual) << " expected "
                << static_cast<int>(tc.expected) << "\n";
      ok = false;
    }
  }

  ok &= Require(InterpolateAndRoundResizeU8(10, 11, 10, 11, 0.49f, 0.0f) ==
                    10,
                "u8 resize ordinary bilinear value below half should round "
                "down");
  ok &= Require(InterpolateAndRoundResizeU8(10, 11, 10, 11, 0.5f, 0.0f) == 11,
                "u8 resize bilinear half tie should round up");
  ok &= Require(InterpolateAndRoundResizeU8(10, 11, 10, 11, 0.51f, 0.0f) ==
                    11,
                "u8 resize ordinary bilinear value above half should round up");
  ok &= Require(InterpolateAndRoundResizeU8(20, 24, 28, 32, 0.25f, 0.75f) ==
                    27,
                "u8 resize two-axis ordinary bilinear value should match CPU "
                "rounding");

  return ok;
}

bool TestOpenCudaAlphaClampAndSolidBgrContractNoGpuSafe() {
  using studiocast::cuda::PixelFormatGpu;
  using studiocast::cuda::kernels::detail::ClampAlpha01ForComposite;
  using studiocast::cuda::kernels::detail::SolidBackgroundMemoryChannels;

  bool ok = true;

  ok &= Require(ClampAlpha01ForComposite(-0.25f) == 0.0f,
                "negative alpha should clamp to 0");
  ok &= Require(ClampAlpha01ForComposite(0.25f) == 0.25f,
                "in-range alpha should be preserved");
  ok &= Require(ClampAlpha01ForComposite(1.25f) == 1.0f,
                "alpha above 1 should clamp to 1");
  ok &= Require(ClampAlpha01ForComposite(
                    std::numeric_limits<float>::quiet_NaN()) == 0.0f,
                "NaN alpha should be treated as transparent foreground");
  ok &= Require(ClampAlpha01ForComposite(
                    std::numeric_limits<float>::infinity()) == 0.0f,
                "infinite alpha should be treated as transparent foreground");
  ok &= Require(ClampAlpha01ForComposite(
                    -std::numeric_limits<float>::infinity()) == 0.0f,
                "negative infinite alpha should be treated as transparent "
                "foreground");

  const auto rgb_bg =
      SolidBackgroundMemoryChannels(PixelFormatGpu::rgb_u8, 200, 40, 10);
  ok &= Require(rgb_bg == std::array<std::uint8_t, 3>{200, 40, 10},
                "rgb_u8 solid background should use RGB memory order");

  const auto bgr_bg =
      SolidBackgroundMemoryChannels(PixelFormatGpu::bgr_u8, 200, 40, 10);
  ok &= Require(bgr_bg == std::array<std::uint8_t, 3>{10, 40, 200},
                "bgr_u8 solid background should use BGR memory order");

  const std::array<std::uint8_t, 3> fg{20, 40, 80};
  ok &= Require(CompositeSolidReference(PixelFormatGpu::rgb_u8, fg,
                                        std::numeric_limits<float>::quiet_NaN(),
                                        200, 40, 10) ==
                    std::array<std::uint8_t, 3>{200, 40, 10},
                "rgb_u8 NaN alpha should produce the solid RGB background");
  ok &= Require(CompositeSolidReference(PixelFormatGpu::bgr_u8, fg,
                                        std::numeric_limits<float>::quiet_NaN(),
                                        200, 40, 10) ==
                    std::array<std::uint8_t, 3>{10, 40, 200},
                "bgr_u8 NaN alpha should produce the solid background in BGR "
                "memory order");
  ok &= Require(CompositeSolidReference(PixelFormatGpu::bgr_u8, fg, 0.5f, 200,
                                        40, 10) ==
                    std::array<std::uint8_t, 3>{15, 40, 140},
                "bgr_u8 half alpha should blend against BGR solid channels");

  return ok;
}

bool TestCudaResizeAvailabilityProbeIsThreadSafe() {
  constexpr int kThreads = 8;
  constexpr int kIterations = 3;

  std::atomic<int> successes{0};
  std::atomic<int> failures{0};
  std::atomic<int> empty_failure_errors{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&] {
      for (int j = 0; j < kIterations; ++j) {
        std::string error;
        const bool ok =
            studiocast::cuda::kernels::IsResizeBilinearAvailable(&error);
        if (ok) {
          successes.fetch_add(1, std::memory_order_relaxed);
        } else {
          failures.fetch_add(1, std::memory_order_relaxed);
          if (error.empty())
            empty_failure_errors.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  for (auto &thread : threads)
    thread.join();

  const int total = successes.load(std::memory_order_relaxed) +
                    failures.load(std::memory_order_relaxed);

  bool ok = true;
  ok &= Require(total == kThreads * kIterations,
                "CUDA resize availability concurrency probe lost results");
  ok &= Require(empty_failure_errors.load(std::memory_order_relaxed) == 0,
                "CUDA resize availability failures should report an error");
  return ok;
}

} // namespace studiocast::tests
