#include "core/cuda/kernels/preprocess_to_nchw.h"

#include "core/cuda/kernels/cuda_driver_cache.h"
#include "core/cuda/kernels/prepared_preprocess_kernel_bundle.h"
#include "core/cuda/kernels/ptx_pitch_contract.h"

#include <cstdint>

namespace studiocast::cuda::kernels {
namespace {

// Preprocess kernel:
// - Bilinear resample interleaved u8x3 pitched image
// - Convert to float in [0,1]
// - Reorder to RGB/BGR as requested
// - Normalize: (v - mean[c]) * invStd[c]
//
// Parameters:
//   srcPtr, srcPitch, srcW, srcH,
//   dstPtr,
//   dstW, dstH,
//   mean0, mean1, mean2,
//   invStd0, invStd1, invStd2,
//   dstIsBgr, srcIsBgr
//
// PTX_SOURCE: src/core/cuda/kernels/preprocess_to_nchw.cu
// PTX_GENERATE: nvcc -ptx -O3 -arch=compute_52 -I src src/core/cuda/kernels/preprocess_to_nchw.cu -o preprocess_to_nchw.ptx
// The embedded body remains hand-authored PTX 6.0 for broad driver JIT
// compatibility; the freshness validator compiles the source and checks the
// preprocess_to_nchw_f32 entry ABI.
static constexpr const char *kPreprocessPtx = R"ptx(
.version 6.0
.target sm_30
.address_size 64

.visible .entry preprocess_to_nchw_f32(
    .param .u64 srcPtr,
    .param .u32 srcPitch,
    .param .u32 srcW,
    .param .u32 srcH,
    .param .u64 dstPtr,
    .param .u32 dstW,
    .param .u32 dstH,
    .param .f32 mean0,
    .param .f32 mean1,
    .param .f32 mean2,
    .param .f32 invStd0,
    .param .f32 invStd1,
    .param .f32 invStd2,
    .param .u32 dstIsBgr,
    .param .u32 srcIsBgr
)
{
    .reg .pred  %p<8>;
    .reg .b32   %r<96>;
    .reg .b64   %rd<32>;
    .reg .f32   %f<96>;

    ld.param.u64 %rd1, [srcPtr];
    ld.param.u32 %r1, [srcPitch];
    ld.param.u32 %r2, [srcW];
    ld.param.u32 %r3, [srcH];
    ld.param.u64 %rd2, [dstPtr];
    ld.param.u32 %r5, [dstW];
    ld.param.u32 %r6, [dstH];
    ld.param.f32 %f60, [mean0];
    ld.param.f32 %f61, [mean1];
    ld.param.f32 %f62, [mean2];
    ld.param.f32 %f63, [invStd0];
    ld.param.f32 %f64, [invStd1];
    ld.param.f32 %f65, [invStd2];
    ld.param.u32 %r70, [dstIsBgr];
    ld.param.u32 %r71, [srcIsBgr];

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

    // x0,y0 = trunc(srcX/srcY), then clamp to [0, srcW-1] / [0, srcH-1]
    add.s32 %r15, %r2, -1;         // srcW-1
    add.s32 %r16, %r3, -1;         // srcH-1
    cvt.rzi.s32.f32 %r17, %f7;
    cvt.rzi.s32.f32 %r18, %f8;
    max.s32 %r17, %r17, 0;
    max.s32 %r18, %r18, 0;
    min.s32 %r17, %r17, %r15;
    min.s32 %r18, %r18, %r16;

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

    // Load interleaved channels for each corner
    // p00 c0,c1,c2
    ld.global.u8 %r24, [%rd9];
    ld.global.u8 %r25, [%rd9+1];
    ld.global.u8 %r26, [%rd9+2];
    // p10
    ld.global.u8 %r27, [%rd10];
    ld.global.u8 %r28, [%rd10+1];
    ld.global.u8 %r29, [%rd10+2];
    // p01
    ld.global.u8 %r30, [%rd11];
    ld.global.u8 %r31, [%rd11+1];
    ld.global.u8 %r32, [%rd11+2];
    // p11
    ld.global.u8 %r33, [%rd12];
    ld.global.u8 %r34, [%rd12+1];
    ld.global.u8 %r35, [%rd12+2];

    // Convert to float
    cvt.rn.f32.u32 %f15, %r24; // p00 c0
    cvt.rn.f32.u32 %f16, %r25; // p00 c1
    cvt.rn.f32.u32 %f17, %r26; // p00 c2
    cvt.rn.f32.u32 %f18, %r27; // p10 c0
    cvt.rn.f32.u32 %f19, %r28; // p10 c1
    cvt.rn.f32.u32 %f20, %r29; // p10 c2
    cvt.rn.f32.u32 %f21, %r30; // p01 c0
    cvt.rn.f32.u32 %f22, %r31; // p01 c1
    cvt.rn.f32.u32 %f23, %r32; // p01 c2
    cvt.rn.f32.u32 %f24, %r33; // p11 c0
    cvt.rn.f32.u32 %f25, %r34; // p11 c1
    cvt.rn.f32.u32 %f26, %r35; // p11 c2

    // Predicates for channel order
    setp.ne.u32 %p4, %r71, 0;   // srcIsBgr
    setp.ne.u32 %p5, %r70, 0;   // dstIsBgr

    // Select canonical R,G,B for each corner based on srcIsBgr
    // R00 = (srcIsBgr ? c2 : c0)
    selp.f32 %f30, %f17, %f15, %p4; // R00
    mov.f32  %f31, %f16;            // G00
    selp.f32 %f32, %f15, %f17, %p4; // B00

    selp.f32 %f33, %f20, %f18, %p4; // R10
    mov.f32  %f34, %f19;            // G10
    selp.f32 %f35, %f18, %f20, %p4; // B10

    selp.f32 %f36, %f23, %f21, %p4; // R01
    mov.f32  %f37, %f22;            // G01
    selp.f32 %f38, %f21, %f23, %p4; // B01

    selp.f32 %f39, %f26, %f24, %p4; // R11
    mov.f32  %f40, %f25;            // G11
    selp.f32 %f41, %f24, %f26, %p4; // B11

    // Interpolate R
    sub.f32 %f42, %f33, %f30;
    mad.rn.f32 %f43, %f12, %f42, %f30;  // v0
    sub.f32 %f44, %f39, %f36;
    mad.rn.f32 %f45, %f12, %f44, %f36;  // v1
    sub.f32 %f46, %f45, %f43;
    mad.rn.f32 %f47, %f14, %f46, %f43;  // R

    // Interpolate G
    sub.f32 %f48, %f34, %f31;
    mad.rn.f32 %f49, %f12, %f48, %f31;
    sub.f32 %f50, %f40, %f37;
    mad.rn.f32 %f51, %f12, %f50, %f37;
    sub.f32 %f52, %f51, %f49;
    mad.rn.f32 %f53, %f14, %f52, %f49;  // G

    // Interpolate B
    sub.f32 %f54, %f35, %f32;
    mad.rn.f32 %f55, %f12, %f54, %f32;
    sub.f32 %f56, %f41, %f38;
    mad.rn.f32 %f57, %f12, %f56, %f38;
    sub.f32 %f58, %f57, %f55;
    mad.rn.f32 %f59, %f14, %f58, %f55;  // B

    // Convert to [0,1]
    mov.f32 %f66, 0f3B808081; // 1/255
    mul.f32 %f67, %f47, %f66; // r
    mul.f32 %f68, %f53, %f66; // g
    mul.f32 %f69, %f59, %f66; // b

    // Output channel order based on dstIsBgr
    // out0 = dstIsBgr ? b : r
    selp.f32 %f70, %f69, %f67, %p5;
    // out2 = dstIsBgr ? r : b
    selp.f32 %f72, %f67, %f69, %p5;
    mov.f32  %f71, %f68; // out1 = g

    // Normalize
    sub.f32 %f73, %f70, %f60;
    mul.f32 %f73, %f73, %f63;
    sub.f32 %f74, %f71, %f61;
    mul.f32 %f74, %f74, %f64;
    sub.f32 %f75, %f72, %f62;
    mul.f32 %f75, %f75, %f65;

    // Compute output addresses (NCHW contiguous)
    mul.lo.u32 %r80, %r5, %r6;        // hw = dstW*dstH
    mad.lo.u32 %r81, %r14, %r5, %r10; // index

    // channel 0
    mul.wide.u32 %rd16, %r81, 4;
    add.u64 %rd20, %rd2, %rd16;
    st.global.f32 [%rd20], %f73;

    // channel 1
    add.u32 %r82, %r81, %r80;
    mul.wide.u32 %rd17, %r82, 4;
    add.u64 %rd21, %rd2, %rd17;
    st.global.f32 [%rd21], %f74;

    // channel 2
    add.u32 %r83, %r80, %r80;
    add.u32 %r84, %r81, %r83;
    mul.wide.u32 %rd18, %r84, 4;
    add.u64 %rd22, %rd2, %rd18;
    st.global.f32 [%rd22], %f75;

DONE:
    ret;
}
)ptx";

bool EnsureCudaReady(studiocast::cuda::CudaDriverApi **out_cuda,
                     std::string *error_out) {
  return detail::EnsureCudaReady(out_cuda, error_out);
}

struct KernelState {
  // CUDA module/function handles are context-bound; keep this cache thread-local
  // with the thread-local driver/context owner in cuda_driver_cache.h.
  bool loaded = false;
  studiocast::cuda::CUmodule module = nullptr;
  studiocast::cuda::CUfunction fn = nullptr;
  studiocast::cuda::CUcontext loaded_ctx = nullptr;
  studiocast::cuda::CudaDriverApi *owner = nullptr;
  studiocast::cuda::CudaContextIdentity context_identity{};
};

KernelState &kernel() {
  thread_local KernelState s;
  return s;
}

bool LoadKernel(studiocast::cuda::CudaDriverApi *cuda,
                bool require_exact_identity, std::string *error_out,
                PreparedPreprocessKernelHandles *handles_out = nullptr,
                PreparedPreprocessKernelStatus *status_out = nullptr) {
  if (error_out)
    error_out->clear();
  if (handles_out)
    *handles_out = {};
  if (status_out)
    *status_out = PreparedPreprocessKernelStatus::invalid;
  studiocast::cuda::CUcontext cur = nullptr;
  if (!detail::GetCurrentContext(cuda, &cur, error_out)) {
    if (status_out)
      *status_out = PreparedPreprocessKernelStatus::unavailable;
    return false;
  }

  KernelState &k = kernel();
  const auto &f = cuda->f();

  const auto *identity = cuda ? cuda->context_identity() : nullptr;
  if (require_exact_identity && (!identity || !identity->Valid())) {
    if (status_out)
      *status_out = PreparedPreprocessKernelStatus::unavailable;
    if (error_out)
      *error_out = "PreparePreprocessKernel: CUDA context identity unavailable.";
    return false;
  }

  if (k.loaded && k.loaded_ctx == cur && k.owner == cuda &&
      (!require_exact_identity ||
       (identity && k.context_identity == *identity))) {
    if (handles_out) {
      handles_out->context = k.loaded_ctx;
      handles_out->module = k.module;
      handles_out->function = k.fn;
    }
    if (status_out)
      *status_out = PreparedPreprocessKernelStatus::ready;
    return true;
  }

  studiocast::cuda::CUresult st = studiocast::cuda::CUDA_SUCCESS;
  std::string jit_log;
  studiocast::cuda::CUmodule module = nullptr;
  studiocast::cuda::CUfunction fn = nullptr;
  if (f.cuModuleLoadDataEx) {
    char info[8192] = {0};
    char err[8192] = {0};

    studiocast::cuda::CUjit_option opts[] = {
        studiocast::cuda::CU_JIT_INFO_LOG_BUFFER,
        studiocast::cuda::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        studiocast::cuda::CU_JIT_ERROR_LOG_BUFFER,
        studiocast::cuda::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        studiocast::cuda::CU_JIT_LOG_VERBOSE,
    };
    void *vals[] = {
        info,
        reinterpret_cast<void *>(static_cast<std::size_t>(sizeof(info))),
        err,
        reinterpret_cast<void *>(static_cast<std::size_t>(sizeof(err))),
        reinterpret_cast<void *>(static_cast<std::size_t>(1)),
    };

    st = f.cuModuleLoadDataEx(&module, kPreprocessPtx,
                              static_cast<unsigned int>(sizeof(opts) /
                                                        sizeof(opts[0])),
                              opts, vals);
    jit_log = std::string("PTX JIT info log:\n") + info +
              "\nPTX JIT error log:\n" + err;
  } else {
    st = f.cuModuleLoadData(&module, kPreprocessPtx);
  }
  if (st != studiocast::cuda::CUDA_SUCCESS) {
    if (status_out)
      *status_out = PreparedPreprocessKernelStatus::module_load_failed;
    if (error_out) {
      *error_out = "cuModuleLoadData(preprocess_to_nchw_f32) failed: " +
                   cuda->StatusToString(st);
      if (!jit_log.empty())
        *error_out += "\n" + jit_log;
    }
    return false;
  }
  st = f.cuModuleGetFunction(&fn, module, "preprocess_to_nchw_f32");
  if (st != studiocast::cuda::CUDA_SUCCESS) {
    if (status_out)
      *status_out =
          PreparedPreprocessKernelStatus::function_resolution_failed;
    if (error_out)
      *error_out = "cuModuleGetFunction(preprocess_to_nchw_f32) failed: " +
                   cuda->StatusToString(st);
    return false;
  }
  k.module = module;
  k.fn = fn;
  k.loaded = true;
  k.loaded_ctx = cur;
  k.owner = cuda;
  k.context_identity = identity ? *identity
                                : studiocast::cuda::CudaContextIdentity{};
  if (handles_out) {
    handles_out->context = k.loaded_ctx;
    handles_out->module = k.module;
    handles_out->function = k.fn;
  }
  if (status_out)
    *status_out = PreparedPreprocessKernelStatus::ready;
  return true;
}

bool ValidatePreparedKernel(
    studiocast::cuda::CudaDriverApi *cuda,
    const studiocast::cuda::CudaContextIdentity *expected,
    studiocast::cuda::CUfunction *out_fn, std::string *error_out) {
  if (out_fn)
    *out_fn = nullptr;
  KernelState &k = kernel();
  const auto *current = cuda ? cuda->context_identity() : nullptr;
  if (!k.loaded || !k.fn || !cuda || k.owner != cuda || !current ||
      !current->Valid() || k.context_identity != *current ||
      (expected && *expected != *current)) {
    if (error_out)
      *error_out = "PreprocessToTensor: setup-prepared CUDA kernel/context "
                   "generation is missing, foreign, or stale.";
    return false;
  }
  if (out_fn)
    *out_fn = k.fn;
  return true;
}

} // namespace

bool PreparePreprocessKernel(studiocast::cuda::CudaDriverApi *cuda,
                             std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!cuda || !cuda->IsInitialized()) {
    if (error_out)
      *error_out = "PreparePreprocessKernel: CUDA driver is not initialized.";
    return false;
  }
  if (!cuda->EnsureContext(error_out))
    return false;
  return LoadKernel(cuda, /*require_exact_identity=*/true, error_out);
}

bool PreparePreprocessKernelHandles(
    CudaDriverApi *driver, const CudaContextIdentity &expected_context,
    PreparedPreprocessKernelHandles *out,
    PreparedPreprocessKernelStatus *status_out, std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (out)
    *out = {};
  if (status_out)
    *status_out = PreparedPreprocessKernelStatus::invalid;
  if (!driver || !driver->IsInitialized() || !expected_context.Valid() ||
      !out) {
    if (status_out)
      *status_out = PreparedPreprocessKernelStatus::invalid_argument;
    if (error_out)
      *error_out = "Prepared preprocess kernel setup binding is invalid.";
    return false;
  }
  if (!driver->EnsureContext(error_out)) {
    if (status_out)
      *status_out = PreparedPreprocessKernelStatus::unavailable;
    return false;
  }

  const auto *current = driver->context_identity();
  if (!current || !current->Valid()) {
    if (status_out)
      *status_out = PreparedPreprocessKernelStatus::unavailable;
    if (error_out)
      *error_out = "Prepared preprocess context identity is unavailable.";
    return false;
  }
  if (current->physical_device != expected_context.physical_device ||
      current->retained_context_id != expected_context.retained_context_id) {
    if (status_out)
      *status_out = PreparedPreprocessKernelStatus::foreign_context;
    if (error_out)
      *error_out = "Prepared preprocess setup context is foreign.";
    return false;
  }
  if (current->logical_generation != expected_context.logical_generation) {
    if (status_out)
      *status_out = PreparedPreprocessKernelStatus::stale_context_generation;
    if (error_out)
      *error_out = "Prepared preprocess setup context generation is stale.";
    return false;
  }
  return LoadKernel(driver, /*require_exact_identity=*/true, error_out, out,
                    status_out);
}

namespace {

bool PreprocessToTensorImpl(
    studiocast::cuda::CudaDriverApi *cuda,
    const studiocast::cuda::CudaContextIdentity *expected_context,
    bool legacy_compatibility,
    const CudaImage &src, const CudaTensor &dst,
    const ModelPreprocessSpec &spec, studiocast::cuda::CUstream stream,
    std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!src.Valid() || !dst.Valid()) {
    if (error_out)
      *error_out = "PreprocessToTensor: invalid src/dst.";
    return false;
  }
  if (src.format != PixelFormatGpu::rgb_u8 &&
      src.format != PixelFormatGpu::bgr_u8) {
    if (error_out)
      *error_out = "PreprocessToTensor: unsupported src format (expected "
                   "rgb_u8 or bgr_u8).";
    return false;
  }
  if (spec.dst_w <= 0 || spec.dst_h <= 0) {
    if (error_out)
      *error_out = "PreprocessToTensor: invalid dst size in spec.";
    return false;
  }
  if (dst.n != 1 || dst.c != 3 || dst.h != spec.dst_h || dst.w != spec.dst_w) {
    if (error_out)
      *error_out = "PreprocessToTensor: dst tensor shape mismatch (expected "
                   "N=1,C=3,H=spec.dst_h,W=spec.dst_w).";
    return false;
  }
  if (!detail::CheckSignedInt32PtxPitch(
          src.pitch, src.RowBytes(), "PreprocessToTensor(src)", error_out))
    return false;
  if (spec.std[0] == 0.0f || spec.std[1] == 0.0f || spec.std[2] == 0.0f) {
    if (error_out)
      *error_out = "PreprocessToTensor: std contains zero.";
    return false;
  }

  const std::size_t want_bytes =
      static_cast<std::size_t>(dst.n) * static_cast<std::size_t>(dst.c) *
      static_cast<std::size_t>(dst.h) * static_cast<std::size_t>(dst.w) *
      sizeof(float);
  if (dst.bytes < want_bytes) {
    if (error_out)
      *error_out = "PreprocessToTensor: dst tensor buffer too small.";
    return false;
  }

  studiocast::cuda::CUfunction fn = nullptr;
  if (legacy_compatibility) {
    if (!EnsureCudaReady(&cuda, error_out) ||
        !LoadKernel(cuda, /*require_exact_identity=*/false, error_out))
      return false;
    fn = kernel().fn;
  } else if (!ValidatePreparedKernel(cuda, expected_context, &fn, error_out)) {
    return false;
  }

  const auto &f = cuda->f();

  const std::uint32_t src_pitch = static_cast<std::uint32_t>(src.pitch);
  const std::uint32_t src_w = static_cast<std::uint32_t>(src.w);
  const std::uint32_t src_h = static_cast<std::uint32_t>(src.h);
  const unsigned long long src_ptr = src.ptr;
  const unsigned long long dst_ptr = dst.ptr;
  const std::uint32_t dst_w = static_cast<std::uint32_t>(spec.dst_w);
  const std::uint32_t dst_h = static_cast<std::uint32_t>(spec.dst_h);
  const float mean0 = spec.mean[0];
  const float mean1 = spec.mean[1];
  const float mean2 = spec.mean[2];
  const float inv0 = 1.0f / spec.std[0];
  const float inv1 = 1.0f / spec.std[1];
  const float inv2 = 1.0f / spec.std[2];
  const std::uint32_t dst_is_bgr =
      (spec.dst_order == ChannelOrder::bgr) ? 1u : 0u;
  const std::uint32_t src_is_bgr =
      (src.format == PixelFormatGpu::bgr_u8) ? 1u : 0u;

  void *args[] = {
      const_cast<unsigned long long *>(&src_ptr),
      const_cast<std::uint32_t *>(&src_pitch),
      const_cast<std::uint32_t *>(&src_w),
      const_cast<std::uint32_t *>(&src_h),
      const_cast<unsigned long long *>(&dst_ptr),
      const_cast<std::uint32_t *>(&dst_w),
      const_cast<std::uint32_t *>(&dst_h),
      const_cast<float *>(&mean0),
      const_cast<float *>(&mean1),
      const_cast<float *>(&mean2),
      const_cast<float *>(&inv0),
      const_cast<float *>(&inv1),
      const_cast<float *>(&inv2),
      const_cast<std::uint32_t *>(&dst_is_bgr),
      const_cast<std::uint32_t *>(&src_is_bgr),
  };

  constexpr unsigned int block_x = 16;
  constexpr unsigned int block_y = 16;
  const unsigned int grid_x =
      (static_cast<unsigned int>(spec.dst_w) + block_x - 1u) / block_x;
  const unsigned int grid_y =
      (static_cast<unsigned int>(spec.dst_h) + block_y - 1u) / block_y;

  const studiocast::cuda::CUresult st = f.cuLaunchKernel(
      fn, grid_x, grid_y, 1, block_x, block_y, 1, 0, stream, args, nullptr);
  if (st != studiocast::cuda::CUDA_SUCCESS) {
    if (error_out)
      *error_out = "cuLaunchKernel(preprocess_to_nchw_f32) failed: " +
                   cuda->StatusToString(st);
    return false;
  }
  return true;
}

} // namespace

bool PreprocessToTensor(const CudaImage &src, const CudaTensor &dst,
                        const ModelPreprocessSpec &spec,
                        studiocast::cuda::CUstream stream,
                        std::string *error_out) {
  return PreprocessToTensorImpl(nullptr, nullptr,
                                /*legacy_compatibility=*/true, src, dst, spec,
                                stream, error_out);
}

bool PreprocessToTensorPrepared(
    studiocast::cuda::CudaDriverApi *cuda,
    const studiocast::cuda::CudaContextIdentity &context_identity,
    const CudaImage &src, const CudaTensor &dst,
    const ModelPreprocessSpec &spec, studiocast::cuda::CUstream stream,
    std::string *error_out) {
  return PreprocessToTensorImpl(cuda, &context_identity,
                                /*legacy_compatibility=*/false, src, dst, spec,
                                stream, error_out);
}

} // namespace studiocast::cuda::kernels
