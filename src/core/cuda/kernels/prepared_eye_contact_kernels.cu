#include <cuda_runtime.h>
#include <stdint.h>

namespace {
__device__ int clampi(int v, int lo, int hi) { return max(lo, min(hi, v)); }
__device__ unsigned char round_byte(float v) {
  return static_cast<unsigned char>(clampi(
      static_cast<int>(floorf(fminf(255.f, fmaxf(0.f, v))) + .5f), 0, 255));
}
__device__ void eye_geometry(const float *landmarks, int eye, int fw, int fh,
                             float *g) {
  const int base = eye == 0 ? 42 : 36;
  const float x0 = landmarks[2 * base], y0 = landmarks[2 * base + 1];
  const float x3 = landmarks[2 * (base + 3)],
              y3 = landmarks[2 * (base + 3) + 1];
  const float cx = (x0 + x3) * .5f, cy = (y0 + y3) * .5f;
  const float half_w = fabsf(x3 - x0) * .75f, box_h = 1.5f * half_w;
  const int left = clampi(static_cast<int>(floorf(cx - half_w)), 0, fw - 2);
  const int right = clampi(static_cast<int>(ceilf(cx + half_w)), 1, fw - 1);
  const int top =
      clampi(static_cast<int>(floorf(cy - box_h * (7.f / 12.f))), 0, fh - 2);
  const int bottom =
      clampi(static_cast<int>(ceilf(cy + box_h * (5.f / 12.f))), 1, fh - 1);
  g[0] = static_cast<float>(left);
  g[1] = static_cast<float>(top);
  g[2] = static_cast<float>(max(1, right - left));
  g[3] = static_cast<float>(max(1, bottom - top));
  g[4] = (right - left >= 2 && bottom - top >= 2) ? 1.f : 0.f;
}
__device__ float sample_u8(const unsigned char *src, unsigned pitch, int fw,
                           int fh, int channel, float x, float y) {
  x = fminf(static_cast<float>(fw - 1), fmaxf(0.f, x));
  y = fminf(static_cast<float>(fh - 1), fmaxf(0.f, y));
  const int x0 = static_cast<int>(floorf(x)), y0 = static_cast<int>(floorf(y));
  const int x1 = min(fw - 1, x0 + 1), y1 = min(fh - 1, y0 + 1);
  const float fx = x - x0, fy = y - y0;
  const float a = src[y0 * pitch + 3 * x0 + channel] * (1.f - fx) +
                  src[y0 * pitch + 3 * x1 + channel] * fx;
  const float b = src[y1 * pitch + 3 * x0 + channel] * (1.f - fx) +
                  src[y1 * pitch + 3 * x1 + channel] * fx;
  return a * (1.f - fy) + b * fy;
}
__device__ float sample_nchw(const float *src, int channel, float x, float y) {
  x = fminf(63.f, fmaxf(0.f, x));
  y = fminf(47.f, fmaxf(0.f, y));
  const int x0 = static_cast<int>(floorf(x)), y0 = static_cast<int>(floorf(y));
  const int x1 = min(63, x0 + 1), y1 = min(47, y0 + 1);
  const float fx = x - x0, fy = y - y0;
  const int p = 48 * 64;
  const float a =
      round_byte(fminf(1.f, fmaxf(0.f, src[channel * p + y0 * 64 + x0])) *
                 255.f) *
          (1 - fx) +
      round_byte(fminf(1.f, fmaxf(0.f, src[channel * p + y0 * 64 + x1])) *
                 255.f) *
          fx;
  const float b =
      round_byte(fminf(1.f, fmaxf(0.f, src[channel * p + y1 * 64 + x0])) *
                 255.f) *
          (1 - fx) +
      round_byte(fminf(1.f, fmaxf(0.f, src[channel * p + y1 * 64 + x1])) *
                 255.f) *
          fx;
  return a * (1 - fy) + b * fy;
}
} // namespace

extern "C" __global__ void eye_contact_prepare_inputs(
    const unsigned char *frame, unsigned pitch, unsigned fw, unsigned fh,
    const float *landmarks, float *geometry, float *left_eye, float *right_eye,
    float *left_anchors, float *right_anchors, float *left_angles,
    float *right_angles, float strength) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned threads = gridDim.x * blockDim.x;
  if (fw < 2 || fh < 2)
    return;
  for (int eye = 0; eye < 2; ++eye) {
    float g[5];
    eye_geometry(landmarks, eye, fw, fh, g);
    if (tid < 5)
      geometry[eye * 5 + tid] = g[tid];
    float *rgb = eye == 0 ? left_eye : right_eye;
    float *anchors = eye == 0 ? left_anchors : right_anchors;
    const int base = eye == 0 ? 42 : 36;
    for (unsigned i = tid; i < 3u * 48u * 64u; i += threads) {
      const int c = i / (48 * 64), q = i % (48 * 64), y = q / 64, x = q % 64;
      const float sx = fminf(g[0] + g[2] - 1.f,
                             fmaxf(g[0], g[0] + (x + .5f) * g[2] / 64.f - .5f));
      const float sy = fminf(g[1] + g[3] - 1.f,
                             fmaxf(g[1], g[1] + (y + .5f) * g[3] / 48.f - .5f));
      rgb[i] = static_cast<float>(
                   round_byte(sample_u8(frame, pitch, fw, fh, c, sx, sy))) /
               255.f;
    }
    for (unsigned i = tid; i < 12u * 48u * 64u; i += threads) {
      const int c = i / (48 * 64), q = i % (48 * 64), y = q / 64, x = q % 64;
      const int point_order_left[6] = {3, 2, 1, 0, 5, 4};
      const int point = base + (eye == 0 ? point_order_left[c / 2] : c / 2);
      const int rx = static_cast<int>(
          floorf((landmarks[2 * point] - g[0]) * 64.f / g[2] + .5f));
      const int ry = static_cast<int>(
          floorf((landmarks[2 * point + 1] - g[1]) * 48.f / g[3] + .5f));
      anchors[i] =
          (c & 1) ? static_cast<float>(y - ry) : static_cast<float>(x - rx);
    }
  }
  if (tid == 0) {
    left_angles[0] = right_angles[0] = 0.f;
    left_angles[1] = right_angles[1] = -.10f * strength;
  }
}

extern "C" __global__ void
eye_contact_composite(const unsigned char *src, unsigned src_pitch,
                      unsigned char *dst, unsigned dst_pitch, unsigned width,
                      unsigned height, const float *geometry, const float *left,
                      const float *right, unsigned left_ok, unsigned right_ok,
                      float strength) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  const unsigned so = y * src_pitch + 3 * x, d = y * dst_pitch + 3 * x;
  const float *model = nullptr, *g = nullptr;
  for (int eye = 0; eye < 2; ++eye) {
    const float *candidate = geometry + eye * 5;
    const bool ok = eye == 0 ? left_ok : right_ok;
    if (ok && candidate[4] > .5f && x >= candidate[0] && y >= candidate[1] &&
        x < candidate[0] + candidate[2] && y < candidate[1] + candidate[3]) {
      model = eye == 0 ? left : right;
      g = candidate;
    }
  }
  for (int c = 0; c < 3; ++c) {
    const unsigned char original = src[so + c];
    if (!model || strength <= 0.f) {
      dst[d + c] = original;
      continue;
    }
    const float mx = (static_cast<float>(x) - g[0] + .5f) * 64.f / g[2] - .5f;
    const float my = (static_cast<float>(y) - g[1] + .5f) * 48.f / g[3] - .5f;
    const float corrected = round_byte(sample_nchw(model, c, mx, my));
    dst[d + c] = round_byte(original + (corrected - original) * strength);
  }
}
