#include "core/maxine/cuda_vignette.h"

#include <algorithm>
#include <cstdint>

namespace studiocast::maxine {
namespace {

bool ValidateBgrU8CudaImageForVignette(const NvCVImage &image,
                                       std::string *error_out) {
  const auto status =
      ValidateBgrU8CudaNvCVImage(image, /*allow_zero_dimensions=*/true);
  if (status == NvCVImageValidationStatus::ok)
    return true;

  if (error_out) {
    *error_out = std::string("ApplyInPlace NvCVImage is invalid: ") +
                 NvCVImageValidationStatusToString(status) + ".";
  }
  return false;
}

// In-place vignette (BGR interleaved, U8).
// Parameters:
//   ptr, pitch, w, h,
//   intensity,
//   centerX, centerY,
//   invHalfW, invHalfH
//
// PTX_SOURCE: src/core/maxine/cuda_vignette_kernel.cu
// PTX_GENERATE: nvcc -ptx -O3 -arch=compute_52 -I src src/core/maxine/cuda_vignette_kernel.cu -o cuda_vignette_kernel.ptx
// The embedded body remains hand-authored PTX 6.0 for broad driver JIT
// compatibility; the freshness validator compiles the source and checks the
// vignette_bgr_u8 entry ABI.
static constexpr const char *kVignettePtx = R"ptx(
.version 6.0
.target sm_30
.address_size 64

.visible .entry vignette_bgr_u8(
    .param .u64 ptr,
    .param .u32 pitch,
    .param .u32 w,
    .param .u32 h,
    .param .f32 intensity,
    .param .f32 centerX,
    .param .f32 centerY,
    .param .f32 invHalfW,
    .param .f32 invHalfH
)
{
    .reg .pred  %p<5>;
    .reg .b32   %r<28>;
    .reg .b64   %rd<8>;
    .reg .f32   %f<24>;

    ld.param.u64 %rd1, [ptr];
    ld.param.u32 %r1, [pitch];
    ld.param.u32 %r2, [w];
    ld.param.u32 %r3, [h];
    ld.param.f32 %f1, [intensity];
    ld.param.f32 %f2, [centerX];
    ld.param.f32 %f3, [centerY];
    ld.param.f32 %f4, [invHalfW];
    ld.param.f32 %f5, [invHalfH];

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.s32 %r7, %r4, %r5, %r6; // x

    mov.u32 %r8, %ctaid.y;
    mov.u32 %r9, %ntid.y;
    mov.u32 %r10, %tid.y;
    mad.lo.s32 %r11, %r8, %r9, %r10; // y

    setp.ge.s32 %p1, %r7, %r2;
    setp.ge.s32 %p2, %r11, %r3;
    or.pred %p3, %p1, %p2;
    @%p3 bra DONE;

    // fx = (x + 0.5 - centerX) * invHalfW
    cvt.rn.f32.s32 %f6, %r7;
    add.f32 %f6, %f6, 0f3F000000; // 0.5
    sub.f32 %f6, %f6, %f2;
    mul.f32 %f6, %f6, %f4;

    // fy = (y + 0.5 - centerY) * invHalfH
    cvt.rn.f32.s32 %f7, %r11;
    add.f32 %f7, %f7, 0f3F000000; // 0.5
    sub.f32 %f7, %f7, %f3;
    mul.f32 %f7, %f7, %f5;

    // r = sqrt(fx*fx + fy*fy) * (1/sqrt(2))
    mul.f32 %f8, %f6, %f6;
    mul.f32 %f9, %f7, %f7;
    add.f32 %f10, %f8, %f9;
    sqrt.rn.f32 %f11, %f10;
    mul.f32 %f11, %f11, 0f3F3504F3; // 0.70710677

    // clamp r to [0,1]
    max.f32 %f12, %f11, 0f00000000;
    min.f32 %f12, %f12, 0f3F800000; // 1.0

    // t = r*r
    mul.f32 %f13, %f12, %f12;

    // factor = max(0, 1 - intensity * t)
    mul.f32 %f14, %f1, %f13;
    sub.f32 %f15, 0f3F800000, %f14;
    max.f32 %f15, %f15, 0f00000000;

    // addr = ptr + y*pitch + x*3
    mul.wide.s32 %rd2, %r11, %r1;
    mul.lo.s32 %r12, %r7, 3;
    cvt.u64.u32 %rd3, %r12;
    add.u64 %rd4, %rd1, %rd2;
    add.u64 %rd4, %rd4, %rd3;

    // load BGR
    ld.global.u8 %r13, [%rd4];
    ld.global.u8 %r14, [%rd4+1];
    ld.global.u8 %r15, [%rd4+2];

    // apply factor and clamp
    cvt.rn.f32.u32 %f16, %r13;
    cvt.rn.f32.u32 %f17, %r14;
    cvt.rn.f32.u32 %f18, %r15;

    mul.f32 %f16, %f16, %f15;
    mul.f32 %f17, %f17, %f15;
    mul.f32 %f18, %f18, %f15;

    cvt.rni.u32.f32 %r16, %f16;
    cvt.rni.u32.f32 %r17, %f17;
    cvt.rni.u32.f32 %r18, %f18;

    min.u32 %r16, %r16, 255;
    min.u32 %r17, %r17, 255;
    min.u32 %r18, %r18, 255;

    cvt.u8.u32 %r19, %r16;
    cvt.u8.u32 %r20, %r17;
    cvt.u8.u32 %r21, %r18;

    st.global.u8 [%rd4], %r19;
    st.global.u8 [%rd4+1], %r20;
    st.global.u8 [%rd4+2], %r21;

DONE:
    ret;
}
)ptx";

} // namespace

bool CudaBgrVignette::Initialize(studiocast::cuda::CudaDriverApi *cuda, std::string *error_out) {
  cuda_ = cuda;
  if (!cuda_) {
    if (error_out)
      *error_out = "CudaBgrVignette.Initialize: cuda is null.";
    return false;
  }
  return true;
}

bool CudaBgrVignette::EnsureKernelLoaded(std::string *error_out) {
  if (loaded_)
    return true;
  if (!cuda_ || !cuda_->IsInitialized()) {
    if (error_out)
      *error_out = "CUDA driver API is not initialized.";
    return false;
  }

  const auto &f = cuda_->f();
  studiocast::cuda::CUresult st = f.cuModuleLoadData(&module_, kVignettePtx);
  if (st != studiocast::cuda::CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuModuleLoadData failed: " + cuda_->StatusToString(st);
    return false;
  }

  st = f.cuModuleGetFunction(&fn_, module_, "vignette_bgr_u8");
  if (st != studiocast::cuda::CUDA_SUCCESS || !fn_) {
    if (error_out)
      *error_out = "cuModuleGetFunction(vignette_bgr_u8) failed: " +
                   cuda_->StatusToString(st);
    return false;
  }

  loaded_ = true;
  return true;
}

bool CudaBgrVignette::ApplyInPlace(NvCVImage *bgr_gpu, float intensity,
                                   float center_x_px, float center_y_px,
                                   CUstream stream, std::string *error_out) {
  if (!bgr_gpu) {
    if (error_out)
      *error_out = "ApplyInPlace called with null image.";
    return false;
  }
  if (!ValidateBgrU8CudaImageForVignette(*bgr_gpu, error_out))
    return false;

  const uint32_t w = bgr_gpu->width;
  const uint32_t h = bgr_gpu->height;
  if (w == 0 || h == 0)
    return true;

  intensity = std::max(0.0f, std::min(1.0f, intensity));
  if (intensity <= 0.0f)
    return true;

  if (!EnsureKernelLoaded(error_out))
    return false;

  const uint64_t ptr =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bgr_gpu->pixels));
  const uint32_t pitch = static_cast<uint32_t>(bgr_gpu->pitch);

  const float inv_half_w = 2.0f / static_cast<float>(w);
  const float inv_half_h = 2.0f / static_cast<float>(h);

  void *params[] = {
      (void *)&ptr,         (void *)&pitch,      (void *)&w,
      (void *)&h,           (void *)&intensity,  (void *)&center_x_px,
      (void *)&center_y_px, (void *)&inv_half_w, (void *)&inv_half_h,
  };

  constexpr unsigned int kBlockX = 16;
  constexpr unsigned int kBlockY = 16;
  const unsigned int gridX = (w + kBlockX - 1) / kBlockX;
  const unsigned int gridY = (h + kBlockY - 1) / kBlockY;

  const studiocast::cuda::CUresult st = cuda_->f().cuLaunchKernel(
      fn_, gridX, gridY, 1, kBlockX, kBlockY, 1, 0, stream, params, nullptr);
  if (st != studiocast::cuda::CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuLaunchKernel(vignette_bgr_u8) failed: " +
                   cuda_->StatusToString(st);
    return false;
  }

  return true;
}

} // namespace studiocast::maxine
