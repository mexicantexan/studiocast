#include "core/cuda/kernels/preprocess_to_nchw.h"

#include <cstdint>

#include <cuda_runtime.h>

// Reference CUDA C source for the embedded Driver API PTX module in
// preprocess_to_nchw_ptx.cpp. The CMake freshness test compiles this file to
// PTX and checks the preprocess_to_nchw_f32 entry ABI; StudioCast's normal
// build does not compile this .cu file into the application.
//
// PTX_GENERATE: nvcc -ptx -O3 -arch=compute_52 -I src src/core/cuda/kernels/preprocess_to_nchw.cu -o preprocess_to_nchw.ptx

namespace studiocast::cuda::kernels {
namespace {

__device__ __forceinline__ int ClampInt(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

struct PreprocessParams {
  int dst_w;
  int dst_h;
  int src_is_bgr;
  int dst_is_bgr;
  float mean[3];
  float inv_std[3];
};

__global__ void PreprocessToNchwKernel(const std::uint8_t* src,
                                       std::size_t src_pitch,
                                       int src_w,
                                       int src_h,
                                       float* dst,
                                       PreprocessParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.dst_w || y >= p.dst_h) return;

  const float scale_x = static_cast<float>(src_w) / static_cast<float>(p.dst_w);
  const float scale_y = static_cast<float>(src_h) / static_cast<float>(p.dst_h);

  const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
  const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;

  const int x0 = ClampInt(static_cast<int>(floorf(src_x)), 0, src_w - 1);
  const int y0 = ClampInt(static_cast<int>(floorf(src_y)), 0, src_h - 1);
  const int x1 = ClampInt(x0 + 1, 0, src_w - 1);
  const int y1 = ClampInt(y0 + 1, 0, src_h - 1);

  const float fx = src_x - static_cast<float>(x0);
  const float fy = src_y - static_cast<float>(y0);

  const auto* row0 = src + static_cast<std::size_t>(y0) * src_pitch;
  const auto* row1 = src + static_cast<std::size_t>(y1) * src_pitch;
  const auto* p00 = row0 + static_cast<std::size_t>(x0) * 3u;
  const auto* p10 = row0 + static_cast<std::size_t>(x1) * 3u;
  const auto* p01 = row1 + static_cast<std::size_t>(x0) * 3u;
  const auto* p11 = row1 + static_cast<std::size_t>(x1) * 3u;

  const int r_i = p.src_is_bgr ? 2 : 0;
  const int g_i = 1;
  const int b_i = p.src_is_bgr ? 0 : 2;

  const float3 v00 = make_float3(static_cast<float>(p00[r_i]), static_cast<float>(p00[g_i]), static_cast<float>(p00[b_i]));
  const float3 v10 = make_float3(static_cast<float>(p10[r_i]), static_cast<float>(p10[g_i]), static_cast<float>(p10[b_i]));
  const float3 v01 = make_float3(static_cast<float>(p01[r_i]), static_cast<float>(p01[g_i]), static_cast<float>(p01[b_i]));
  const float3 v11 = make_float3(static_cast<float>(p11[r_i]), static_cast<float>(p11[g_i]), static_cast<float>(p11[b_i]));

  const float3 vx0 = make_float3(v00.x + fx * (v10.x - v00.x), v00.y + fx * (v10.y - v00.y), v00.z + fx * (v10.z - v00.z));
  const float3 vx1 = make_float3(v01.x + fx * (v11.x - v01.x), v01.y + fx * (v11.y - v01.y), v01.z + fx * (v11.z - v01.z));
  const float3 v = make_float3(vx0.x + fy * (vx1.x - vx0.x), vx0.y + fy * (vx1.y - vx0.y), vx0.z + fy * (vx1.z - vx0.z));

  const float inv255 = 1.0f / 255.0f;
  const float r = v.x * inv255;
  const float g = v.y * inv255;
  const float b = v.z * inv255;

  float out[3];
  if (!p.dst_is_bgr) {
    out[0] = r;
    out[1] = g;
    out[2] = b;
  } else {
    out[0] = b;
    out[1] = g;
    out[2] = r;
  }

  const int hw = p.dst_h * p.dst_w;
  const int base = y * p.dst_w + x;
  for (int c = 0; c < 3; ++c) {
    const float norm = (out[c] - p.mean[c]) * p.inv_std[c];
    dst[c * hw + base] = norm;
  }
}

}  // namespace

extern "C" __global__ void preprocess_to_nchw_f32(
    const unsigned char *src, unsigned int src_pitch, unsigned int src_w,
    unsigned int src_h, float *dst, unsigned int dst_w, unsigned int dst_h,
    float mean0, float mean1, float mean2, float inv_std0, float inv_std1,
    float inv_std2, unsigned int dst_is_bgr, unsigned int src_is_bgr,
    unsigned int letterbox_enabled, unsigned int scaled_w,
    unsigned int scaled_h, unsigned int pad_x, unsigned int pad_y,
    unsigned int pad_value) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= static_cast<int>(dst_w) || y >= static_cast<int>(dst_h))
    return;

  const bool padding = letterbox_enabled &&
      (x < static_cast<int>(pad_x) || y < static_cast<int>(pad_y) ||
       x >= static_cast<int>(pad_x + scaled_w) ||
       y >= static_cast<int>(pad_y + scaled_h));
  if (padding) {
    const float value = static_cast<float>(pad_value) / 255.0f;
    const unsigned int hw = dst_w * dst_h;
    const unsigned int base = static_cast<unsigned int>(y) * dst_w +
                              static_cast<unsigned int>(x);
    dst[base] = (value - mean0) * inv_std0;
    dst[hw + base] = (value - mean1) * inv_std1;
    dst[hw + hw + base] = (value - mean2) * inv_std2;
    return;
  }
  const unsigned int sample_w = letterbox_enabled ? scaled_w : dst_w;
  const unsigned int sample_h = letterbox_enabled ? scaled_h : dst_h;
  const int sample_x = letterbox_enabled ? x - static_cast<int>(pad_x) : x;
  const int sample_y = letterbox_enabled ? y - static_cast<int>(pad_y) : y;
  const float scale_x = static_cast<float>(src_w) / static_cast<float>(sample_w);
  const float scale_y = static_cast<float>(src_h) / static_cast<float>(sample_h);
  const float src_x = (static_cast<float>(sample_x) + 0.5f) * scale_x - 0.5f;
  const float src_y = (static_cast<float>(sample_y) + 0.5f) * scale_y - 0.5f;

  const int src_w_max = static_cast<int>(src_w) - 1;
  const int src_h_max = static_cast<int>(src_h) - 1;
  const int x0 = ClampInt(static_cast<int>(src_x), 0, src_w_max);
  const int y0 = ClampInt(static_cast<int>(src_y), 0, src_h_max);
  const int x1 = ClampInt(x0 + 1, 0, src_w_max);
  const int y1 = ClampInt(y0 + 1, 0, src_h_max);

  const float fx = src_x - static_cast<float>(x0);
  const float fy = src_y - static_cast<float>(y0);

  const auto *row0 = src + static_cast<unsigned int>(y0) * src_pitch;
  const auto *row1 = src + static_cast<unsigned int>(y1) * src_pitch;
  const auto *p00 = row0 + static_cast<unsigned int>(x0) * 3u;
  const auto *p10 = row0 + static_cast<unsigned int>(x1) * 3u;
  const auto *p01 = row1 + static_cast<unsigned int>(x0) * 3u;
  const auto *p11 = row1 + static_cast<unsigned int>(x1) * 3u;

  const int r_i = src_is_bgr ? 2 : 0;
  const int g_i = 1;
  const int b_i = src_is_bgr ? 0 : 2;

  const float r00 = static_cast<float>(p00[r_i]);
  const float g00 = static_cast<float>(p00[g_i]);
  const float b00 = static_cast<float>(p00[b_i]);
  const float r10 = static_cast<float>(p10[r_i]);
  const float g10 = static_cast<float>(p10[g_i]);
  const float b10 = static_cast<float>(p10[b_i]);
  const float r01 = static_cast<float>(p01[r_i]);
  const float g01 = static_cast<float>(p01[g_i]);
  const float b01 = static_cast<float>(p01[b_i]);
  const float r11 = static_cast<float>(p11[r_i]);
  const float g11 = static_cast<float>(p11[g_i]);
  const float b11 = static_cast<float>(p11[b_i]);

  const float r0 = r00 + fx * (r10 - r00);
  const float g0 = g00 + fx * (g10 - g00);
  const float b0 = b00 + fx * (b10 - b00);
  const float r1 = r01 + fx * (r11 - r01);
  const float g1 = g01 + fx * (g11 - g01);
  const float b1 = b01 + fx * (b11 - b01);

  const float inv255 = 1.0f / 255.0f;
  const float r = (r0 + fy * (r1 - r0)) * inv255;
  const float g = (g0 + fy * (g1 - g0)) * inv255;
  const float b = (b0 + fy * (b1 - b0)) * inv255;

  const float out0 = dst_is_bgr ? b : r;
  const float out1 = g;
  const float out2 = dst_is_bgr ? r : b;
  const unsigned int hw = dst_w * dst_h;
  const unsigned int base = static_cast<unsigned int>(y) * dst_w +
                            static_cast<unsigned int>(x);

  dst[base] = (out0 - mean0) * inv_std0;
  dst[hw + base] = (out1 - mean1) * inv_std1;
  dst[hw + hw + base] = (out2 - mean2) * inv_std2;
}

bool PreprocessToTensor(const CudaImage& src,
                        const CudaTensor& dst,
                        const ModelPreprocessSpec& spec,
                        studiocast::cuda::CUstream stream,
                        std::string* error_out) {
  if (error_out) error_out->clear();
  if (!src.Valid() || !dst.Valid()) {
    if (error_out) *error_out = "PreprocessToTensor: invalid src/dst.";
    return false;
  }
  if (src.format != PixelFormatGpu::rgb_u8 && src.format != PixelFormatGpu::bgr_u8) {
    if (error_out) *error_out = "PreprocessToTensor: unsupported src format (expected rgb_u8 or bgr_u8).";
    return false;
  }
  if (spec.dst_w <= 0 || spec.dst_h <= 0) {
    if (error_out) *error_out = "PreprocessToTensor: invalid dst size in spec.";
    return false;
  }
  if (dst.n != 1 || dst.c != 3 || dst.h != spec.dst_h || dst.w != spec.dst_w) {
    if (error_out) *error_out = "PreprocessToTensor: dst tensor shape mismatch (expected N=1,C=3,H=spec.dst_h,W=spec.dst_w).";
    return false;
  }

  const std::size_t want_bytes = static_cast<std::size_t>(dst.n) * static_cast<std::size_t>(dst.c) *
                                 static_cast<std::size_t>(dst.h) * static_cast<std::size_t>(dst.w) * sizeof(float);
  if (dst.bytes < want_bytes) {
    if (error_out) *error_out = "PreprocessToTensor: dst tensor buffer too small.";
    return false;
  }
  if (spec.std[0] == 0.0f || spec.std[1] == 0.0f || spec.std[2] == 0.0f) {
    if (error_out) *error_out = "PreprocessToTensor: std contains zero.";
    return false;
  }

  PreprocessParams p{};
  p.dst_w = spec.dst_w;
  p.dst_h = spec.dst_h;
  p.src_is_bgr = (src.format == PixelFormatGpu::bgr_u8) ? 1 : 0;
  p.dst_is_bgr = (spec.dst_order == ChannelOrder::bgr) ? 1 : 0;
  for (int i = 0; i < 3; ++i) {
    p.mean[i] = spec.mean[i];
    p.inv_std[i] = 1.0f / spec.std[i];
  }

  const dim3 block(16, 16, 1);
  const dim3 grid((p.dst_w + block.x - 1) / block.x, (p.dst_h + block.y - 1) / block.y, 1);
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

  const auto* src_ptr = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(src.ptr));
  auto* dst_ptr = reinterpret_cast<float*>(static_cast<std::uintptr_t>(dst.ptr));

  PreprocessToNchwKernel<<<grid, block, 0, s>>>(src_ptr, src.pitch, src.w, src.h, dst_ptr, p);
  const cudaError_t st = cudaGetLastError();
  if (st != cudaSuccess) {
    if (error_out) *error_out = std::string("PreprocessToTensor kernel launch failed: ") + cudaGetErrorString(st);
    return false;
  }
  return true;
}

}  // namespace studiocast::cuda::kernels
