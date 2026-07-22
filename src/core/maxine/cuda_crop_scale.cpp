#include "core/maxine/cuda_crop_scale.h"

#include <algorithm>
#include <cstdint>

namespace studiocast::maxine {
namespace {

bool ValidateBgrU8CudaImageForKernel(const NvCVImage &image, const char *label,
                                     std::string *error_out) {
  const auto status = ValidateBgrU8CudaNvCVImage(image);
  if (status == NvCVImageValidationStatus::ok)
    return true;

  if (error_out) {
    *error_out = std::string(label) + " NvCVImage is invalid: " +
                 NvCVImageValidationStatusToString(status) + ".";
  }
  return false;
}

// Nearest-neighbor crop+scale kernel (BGR interleaved, U8).
// Parameters:
//   srcPtr, srcPitch, srcW, srcH,
//   dstPtr, dstPitch, dstW, dstH,
//   cropX, cropY, cropW, cropH
//
// PTX_SOURCE: src/core/maxine/cuda_crop_scale_kernels.cu
// PTX_GENERATE: nvcc -ptx -O3 -arch=compute_52 -I src src/core/maxine/cuda_crop_scale_kernels.cu -o cuda_crop_scale_kernels.ptx
// The embedded body remains hand-authored PTX 6.0 for broad driver JIT
// compatibility; the freshness validator compiles the source and checks the
// crop_scale_bgr_u8 entry ABI.
static constexpr const char *kCropScalePtx = R"ptx(
.version 6.0
.target sm_30
.address_size 64

.visible .entry crop_scale_bgr_u8(
    .param .u64 srcPtr,
    .param .u32 srcPitch,
    .param .u32 srcW,
    .param .u32 srcH,
    .param .u64 dstPtr,
    .param .u32 dstPitch,
    .param .u32 dstW,
    .param .u32 dstH,
    .param .f32 cropX,
    .param .f32 cropY,
    .param .f32 cropW,
    .param .f32 cropH
)
{
    .reg .pred  %p<4>;
    .reg .b32   %r<24>;
    .reg .b64   %rd<12>;
    .reg .f32   %f<16>;

    ld.param.u64 %rd1, [srcPtr];
    ld.param.u32 %r1, [srcPitch];
    ld.param.u32 %r2, [srcW];
    ld.param.u32 %r3, [srcH];
    ld.param.u64 %rd2, [dstPtr];
    ld.param.u32 %r4, [dstPitch];
    ld.param.u32 %r5, [dstW];
    ld.param.u32 %r6, [dstH];
    ld.param.f32 %f1, [cropX];
    ld.param.f32 %f2, [cropY];
    ld.param.f32 %f3, [cropW];
    ld.param.f32 %f4, [cropH];

    mov.u32 %r7, %ctaid.x;
    mov.u32 %r8, %ntid.x;
    mov.u32 %r9, %tid.x;
    mad.lo.s32 %r10, %r7, %r8, %r9; // x

    mov.u32 %r11, %ctaid.y;
    mov.u32 %r12, %ntid.y;
    mov.u32 %r13, %tid.y;
    mad.lo.s32 %r14, %r11, %r12, %r13; // y

    setp.ge.s32 %p1, %r10, %r5;
    setp.ge.s32 %p2, %r14, %r6;
    or.pred %p3, %p1, %p2;
    @%p3 bra DONE;

    // fx = cropX + (x + 0.5) * cropW / dstW
    cvt.rn.f32.s32 %f5, %r10;
    add.f32 %f5, %f5, 0f3F000000; // 0.5
    mul.f32 %f5, %f5, %f3;
    cvt.rn.f32.u32 %f6, %r5;
    div.rn.f32 %f5, %f5, %f6;
    add.f32 %f5, %f5, %f1;

    // fy = cropY + (y + 0.5) * cropH / dstH
    cvt.rn.f32.s32 %f7, %r14;
    add.f32 %f7, %f7, 0f3F000000; // 0.5
    mul.f32 %f7, %f7, %f4;
    cvt.rn.f32.u32 %f8, %r6;
    div.rn.f32 %f7, %f7, %f8;
    add.f32 %f7, %f7, %f2;

    cvt.rzi.s32.f32 %r15, %f5; // ix
    cvt.rzi.s32.f32 %r16, %f7; // iy

    max.s32 %r15, %r15, 0;
    max.s32 %r16, %r16, 0;
    add.s32 %r17, %r2, -1;
    add.s32 %r18, %r3, -1;
    min.s32 %r15, %r15, %r17;
    min.s32 %r16, %r16, %r18;

    // src addr
    mul.wide.s32 %rd3, %r16, %r1;
    mul.lo.s32 %r19, %r15, 3;
    cvt.u64.u32 %rd4, %r19;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd5, %rd5, %rd4;

    // dst addr
    mul.wide.s32 %rd6, %r14, %r4;
    mul.lo.s32 %r20, %r10, 3;
    cvt.u64.u32 %rd7, %r20;
    add.u64 %rd8, %rd2, %rd6;
    add.u64 %rd8, %rd8, %rd7;

    // load/store BGR
    ld.global.u8 %r21, [%rd5];
    ld.global.u8 %r22, [%rd5+1];
    ld.global.u8 %r23, [%rd5+2];
    st.global.u8 [%rd8], %r21;
    st.global.u8 [%rd8+1], %r22;
    st.global.u8 [%rd8+2], %r23;

DONE:
    ret;
}
)ptx";

// Bilinear resize kernel (BGR interleaved, U8).
// Parameters:
//   srcPtr, srcPitch, srcW, srcH,
//   dstPtr, dstPitch, dstW, dstH
//
// PTX_SOURCE: src/core/maxine/cuda_crop_scale_kernels.cu
// PTX_GENERATE: nvcc -ptx -O3 -arch=compute_52 -I src src/core/maxine/cuda_crop_scale_kernels.cu -o cuda_crop_scale_kernels.ptx
// The embedded body remains hand-authored PTX 6.0 for broad driver JIT
// compatibility; the freshness validator compiles the source and checks the
// resize_bilinear_bgr_u8 entry ABI.
static constexpr const char *kResizeBilinearPtx = R"ptx(
.version 6.0
.target sm_30
.address_size 64

.visible .entry resize_bilinear_bgr_u8(
    .param .u64 srcPtr,
    .param .u32 srcPitch,
    .param .u32 srcW,
    .param .u32 srcH,
    .param .u64 dstPtr,
    .param .u32 dstPitch,
    .param .u32 dstW,
    .param .u32 dstH
)
{
    .reg .pred  %p<6>;
    .reg .b32   %r<64>;
    .reg .b64   %rd<24>;
    .reg .f32   %f<64>;

    ld.param.u64 %rd1, [srcPtr];
    ld.param.u32 %r1, [srcPitch];
    ld.param.u32 %r2, [srcW];
    ld.param.u32 %r3, [srcH];
    ld.param.u64 %rd2, [dstPtr];
    ld.param.u32 %r4, [dstPitch];
    ld.param.u32 %r5, [dstW];
    ld.param.u32 %r6, [dstH];

    mov.u32 %r7, %ctaid.x;
    mov.u32 %r8, %ntid.x;
    mov.u32 %r9, %tid.x;
    mad.lo.s32 %r10, %r7, %r8, %r9; // x

    mov.u32 %r11, %ctaid.y;
    mov.u32 %r12, %ntid.y;
    mov.u32 %r13, %tid.y;
    mad.lo.s32 %r14, %r11, %r12, %r13; // y

    setp.ge.s32 %p1, %r10, %r5;
    setp.ge.s32 %p2, %r14, %r6;
    or.pred %p3, %p1, %p2;
    @%p3 bra DONE;

    // sx = srcW / dstW
    cvt.rn.f32.u32 %f1, %r2;
    cvt.rn.f32.u32 %f2, %r5;
    div.rn.f32 %f3, %f1, %f2;

    // sy = srcH / dstH
    cvt.rn.f32.u32 %f4, %r3;
    cvt.rn.f32.u32 %f5, %r6;
    div.rn.f32 %f6, %f4, %f5;

    // srcX = (x + 0.5) * sx - 0.5
    cvt.rn.f32.s32 %f7, %r10;
    add.f32 %f7, %f7, 0f3F000000; // 0.5
    mul.f32 %f7, %f7, %f3;
    add.f32 %f7, %f7, 0fBF000000; // -0.5

    // srcY = (y + 0.5) * sy - 0.5
    cvt.rn.f32.s32 %f8, %r14;
    add.f32 %f8, %f8, 0f3F000000; // 0.5
    mul.f32 %f8, %f8, %f6;
    add.f32 %f8, %f8, 0fBF000000; // -0.5

    // Clamp to [0, srcW-1] / [0, srcH-1]
    add.s32 %r15, %r2, -1;
    add.s32 %r16, %r3, -1;
    cvt.rn.f32.s32 %f9, %r15;
    cvt.rn.f32.s32 %f10, %r16;

    max.f32 %f7, %f7, 0f00000000; // 0
    max.f32 %f8, %f8, 0f00000000; // 0
    min.f32 %f7, %f7, %f9;
    min.f32 %f8, %f8, %f10;

    // x0,y0
    cvt.rzi.s32.f32 %r17, %f7;
    cvt.rzi.s32.f32 %r18, %f8;

    // x1 = min(x0+1, srcW-1), y1 = min(y0+1, srcH-1)
    add.s32 %r19, %r17, 1;
    add.s32 %r20, %r18, 1;
    min.s32 %r19, %r19, %r15;
    min.s32 %r20, %r20, %r16;

    // tx = srcX - float(x0)
    cvt.rn.f32.s32 %f11, %r17;
    sub.f32 %f12, %f7, %f11;
    // ty = srcY - float(y0)
    cvt.rn.f32.s32 %f13, %r18;
    sub.f32 %f14, %f8, %f13;

    // Row base pointers
    mul.wide.s32 %rd3, %r18, %r1;      // y0 * srcPitch
    mul.wide.s32 %rd4, %r20, %r1;      // y1 * srcPitch

    // x offsets in bytes
    mul.lo.s32 %r21, %r17, 3;          // x0*3
    mul.lo.s32 %r22, %r19, 3;          // x1*3
    cvt.u64.u32 %rd5, %r21;
    cvt.u64.u32 %rd6, %r22;

    // src addresses
    add.u64 %rd7, %rd1, %rd3;
    add.u64 %rd8, %rd1, %rd4;

    add.u64 %rd9, %rd7, %rd5;          // p00
    add.u64 %rd10, %rd7, %rd6;         // p10
    add.u64 %rd11, %rd8, %rd5;         // p01
    add.u64 %rd12, %rd8, %rd6;         // p11

    // dst address
    mul.wide.s32 %rd13, %r14, %r4;     // y * dstPitch
    mul.lo.s32 %r23, %r10, 3;          // x*3
    cvt.u64.u32 %rd14, %r23;
    add.u64 %rd15, %rd2, %rd13;
    add.u64 %rd15, %rd15, %rd14;

    // For each channel: v = lerp(lerp(p00,p10,tx), lerp(p01,p11,tx), ty)
    // Channel 0 (B)
    ld.global.u8 %r24, [%rd9];
    ld.global.u8 %r25, [%rd10];
    ld.global.u8 %r26, [%rd11];
    ld.global.u8 %r27, [%rd12];
    cvt.rn.f32.u32 %f15, %r24;
    cvt.rn.f32.u32 %f16, %r25;
    cvt.rn.f32.u32 %f17, %r26;
    cvt.rn.f32.u32 %f18, %r27;
    sub.f32 %f19, %f16, %f15;
    mad.rn.f32 %f20, %f12, %f19, %f15;    // v0
    sub.f32 %f21, %f18, %f17;
    mad.rn.f32 %f22, %f12, %f21, %f17;    // v1
    sub.f32 %f23, %f22, %f20;
    mad.rn.f32 %f24, %f14, %f23, %f20;    // v
    add.f32 %f24, %f24, 0f3F000000;    // +0.5; truncate for half-up byte rounding
    max.f32 %f24, %f24, 0f00000000;
    min.f32 %f24, %f24, 0f437F0000;    // 255
    cvt.rzi.u32.f32 %r28, %f24;
    st.global.u8 [%rd15], %r28;

    // Channel 1 (G)
    ld.global.u8 %r29, [%rd9+1];
    ld.global.u8 %r30, [%rd10+1];
    ld.global.u8 %r31, [%rd11+1];
    ld.global.u8 %r32, [%rd12+1];
    cvt.rn.f32.u32 %f25, %r29;
    cvt.rn.f32.u32 %f26, %r30;
    cvt.rn.f32.u32 %f27, %r31;
    cvt.rn.f32.u32 %f28, %r32;
    sub.f32 %f29, %f26, %f25;
    mad.rn.f32 %f30, %f12, %f29, %f25;
    sub.f32 %f31, %f28, %f27;
    mad.rn.f32 %f32, %f12, %f31, %f27;
    sub.f32 %f33, %f32, %f30;
    mad.rn.f32 %f34, %f14, %f33, %f30;
    add.f32 %f34, %f34, 0f3F000000;
    max.f32 %f34, %f34, 0f00000000;
    min.f32 %f34, %f34, 0f437F0000;
    cvt.rzi.u32.f32 %r33, %f34;
    st.global.u8 [%rd15+1], %r33;

    // Channel 2 (R)
    ld.global.u8 %r34, [%rd9+2];
    ld.global.u8 %r35, [%rd10+2];
    ld.global.u8 %r36, [%rd11+2];
    ld.global.u8 %r37, [%rd12+2];
    cvt.rn.f32.u32 %f35, %r34;
    cvt.rn.f32.u32 %f36, %r35;
    cvt.rn.f32.u32 %f37, %r36;
    cvt.rn.f32.u32 %f38, %r37;
    sub.f32 %f39, %f36, %f35;
    mad.rn.f32 %f40, %f12, %f39, %f35;
    sub.f32 %f41, %f38, %f37;
    mad.rn.f32 %f42, %f12, %f41, %f37;
    sub.f32 %f43, %f42, %f40;
    mad.rn.f32 %f44, %f14, %f43, %f40;
    add.f32 %f44, %f44, 0f3F000000;
    max.f32 %f44, %f44, 0f00000000;
    min.f32 %f44, %f44, 0f437F0000;
    cvt.rzi.u32.f32 %r38, %f44;
    st.global.u8 [%rd15+2], %r38;

DONE:
    ret;
}
)ptx";

} // namespace

bool CudaBgrCropScale::Initialize(studiocast::cuda::CudaDriverApi *cuda, std::string *error_out) {
  cuda_ = cuda;
  if (!cuda_) {
    if (error_out)
      *error_out = "CudaBgrCropScale.Initialize: cuda is null.";
    return false;
  }
  return true;
}

bool CudaBgrCropScale::EnsureKernelLoaded(std::string *error_out) {
  if (loaded_)
    return true;
  if (!cuda_ || !cuda_->IsInitialized()) {
    if (error_out)
      *error_out = "CUDA driver API is not initialized.";
    return false;
  }

  const auto &f = cuda_->f();
  studiocast::cuda::CUresult st = f.cuModuleLoadData(&module_, kCropScalePtx);
  if (st != studiocast::cuda::CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuModuleLoadData failed: " + cuda_->StatusToString(st);
    return false;
  }

  st = f.cuModuleGetFunction(&fn_, module_, "crop_scale_bgr_u8");
  if (st != studiocast::cuda::CUDA_SUCCESS || !fn_) {
    if (error_out)
      *error_out = "cuModuleGetFunction(crop_scale_bgr_u8) failed: " +
                   cuda_->StatusToString(st);
    return false;
  }

  loaded_ = true;
  return true;
}

bool CudaBgrCropScale::CropScale(const NvCVImage &src_bgr_gpu,
                                 NvCVImage *dst_bgr_gpu, float crop_x,
                                 float crop_y, float crop_w, float crop_h,
                                 CUstream stream, std::string *error_out) {
  if (!dst_bgr_gpu) {
    if (error_out)
      *error_out = "CropScale called with null dst.";
    return false;
  }
  if (!ValidateBgrU8CudaImageForKernel(src_bgr_gpu, "CropScale src",
                                       error_out) ||
      !ValidateBgrU8CudaImageForKernel(*dst_bgr_gpu, "CropScale dst",
                                       error_out)) {
    return false;
  }
  if (!EnsureKernelLoaded(error_out))
    return false;

  const uint32_t srcW = src_bgr_gpu.width;
  const uint32_t srcH = src_bgr_gpu.height;
  const uint32_t dstW = dst_bgr_gpu->width;
  const uint32_t dstH = dst_bgr_gpu->height;

  // Clamp crop rect to valid range.
  crop_w = std::max(1.0f, std::min(crop_w, static_cast<float>(srcW)));
  crop_h = std::max(1.0f, std::min(crop_h, static_cast<float>(srcH)));
  crop_x = std::max(0.0f, std::min(crop_x, static_cast<float>(srcW) - crop_w));
  crop_y = std::max(0.0f, std::min(crop_y, static_cast<float>(srcH) - crop_h));

  const uint64_t srcPtr =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src_bgr_gpu.pixels));
  const uint64_t dstPtr =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst_bgr_gpu->pixels));
  const uint32_t srcPitch = static_cast<uint32_t>(src_bgr_gpu.pitch);
  const uint32_t dstPitch = static_cast<uint32_t>(dst_bgr_gpu->pitch);

  void *params[] = {
      (void *)&srcPtr, (void *)&srcPitch, (void *)&srcW,   (void *)&srcH,
      (void *)&dstPtr, (void *)&dstPitch, (void *)&dstW,   (void *)&dstH,
      (void *)&crop_x, (void *)&crop_y,   (void *)&crop_w, (void *)&crop_h,
  };

  constexpr unsigned int kBlockX = 16;
  constexpr unsigned int kBlockY = 16;
  const unsigned int gridX = (dstW + kBlockX - 1) / kBlockX;
  const unsigned int gridY = (dstH + kBlockY - 1) / kBlockY;

  const studiocast::cuda::CUresult st = cuda_->f().cuLaunchKernel(
      fn_, gridX, gridY, 1, kBlockX, kBlockY, 1, 0, stream, params, nullptr);
  if (st != studiocast::cuda::CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuLaunchKernel(crop_scale_bgr_u8) failed: " +
                   cuda_->StatusToString(st);
    return false;
  }

  return true;
}

bool CudaBgrResizeBilinear::Initialize(studiocast::cuda::CudaDriverApi *cuda,
                                       std::string *error_out) {
  cuda_ = cuda;
  if (!cuda_) {
    if (error_out)
      *error_out = "CudaBgrResizeBilinear.Initialize: cuda is null.";
    return false;
  }
  return true;
}

bool CudaBgrResizeBilinear::EnsureKernelLoaded(std::string *error_out) {
  if (loaded_)
    return true;
  if (!cuda_ || !cuda_->IsInitialized()) {
    if (error_out)
      *error_out = "CUDA driver API is not initialized.";
    return false;
  }

  const auto &f = cuda_->f();
  studiocast::cuda::CUresult st = f.cuModuleLoadData(&module_, kResizeBilinearPtx);
  if (st != studiocast::cuda::CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuModuleLoadData failed: " + cuda_->StatusToString(st);
    return false;
  }

  st = f.cuModuleGetFunction(&fn_, module_, "resize_bilinear_bgr_u8");
  if (st != studiocast::cuda::CUDA_SUCCESS || !fn_) {
    if (error_out) {
      *error_out = "cuModuleGetFunction(resize_bilinear_bgr_u8) failed: " +
                   cuda_->StatusToString(st);
    }
    return false;
  }

  loaded_ = true;
  return true;
}

bool CudaBgrResizeBilinear::Resize(const NvCVImage &src_bgr_gpu,
                                   NvCVImage *dst_bgr_gpu, CUstream stream,
                                   std::string *error_out) {
  if (!dst_bgr_gpu) {
    if (error_out)
      *error_out = "Resize called with null dst.";
    return false;
  }
  if (!ValidateBgrU8CudaImageForKernel(src_bgr_gpu, "Resize src",
                                       error_out) ||
      !ValidateBgrU8CudaImageForKernel(*dst_bgr_gpu, "Resize dst",
                                       error_out)) {
    return false;
  }
  if (!EnsureKernelLoaded(error_out))
    return false;

  const uint32_t srcW = src_bgr_gpu.width;
  const uint32_t srcH = src_bgr_gpu.height;
  const uint32_t dstW = dst_bgr_gpu->width;
  const uint32_t dstH = dst_bgr_gpu->height;

  const uint64_t srcPtr =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src_bgr_gpu.pixels));
  const uint64_t dstPtr =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst_bgr_gpu->pixels));
  const uint32_t srcPitch = static_cast<uint32_t>(src_bgr_gpu.pitch);
  const uint32_t dstPitch = static_cast<uint32_t>(dst_bgr_gpu->pitch);

  void *params[] = {
      (void *)&srcPtr, (void *)&srcPitch, (void *)&srcW, (void *)&srcH,
      (void *)&dstPtr, (void *)&dstPitch, (void *)&dstW, (void *)&dstH,
  };

  constexpr unsigned int kBlockX = 16;
  constexpr unsigned int kBlockY = 16;
  const unsigned int gridX = (dstW + kBlockX - 1u) / kBlockX;
  const unsigned int gridY = (dstH + kBlockY - 1u) / kBlockY;

  const auto &f = cuda_->f();
  const studiocast::cuda::CUresult st = f.cuLaunchKernel(fn_, gridX, gridY, 1, kBlockX, kBlockY,
                                       1, 0, stream, params, nullptr);
  if (st != studiocast::cuda::CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuLaunchKernel(resize_bilinear_bgr_u8) failed: " +
                   cuda_->StatusToString(st);
    return false;
  }

  return true;
}

} // namespace studiocast::maxine
