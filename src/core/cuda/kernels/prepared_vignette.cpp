#include "core/cuda/kernels/prepared_vignette.h"

#include "core/cuda/kernels/ptx_pitch_contract.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace studiocast::cuda::kernels {
namespace {

// Same radial contract as the published CUDA vignette: normalized elliptical
// radius, r^2 falloff, factor=max(0, 1-intensity*r^2), nearest-even byte round.
constexpr const char *kPtx = R"ptx(
.version 6.0
.target sm_30
.address_size 64
.visible .entry vignette_u8x3(
 .param .u64 src,.param .u32 srcPitch,.param .u64 dst,.param .u32 dstPitch,
 .param .u32 w,.param .u32 h,.param .f32 intensity,
 .param .f32 centerX,.param .f32 centerY,.param .f32 invHalfW,.param .f32 invHalfH)
{
 .reg .pred %p<4>; .reg .b32 %r<24>; .reg .b64 %rd<8>; .reg .f32 %f<20>;
 ld.param.u64 %rd1,[src]; ld.param.u32 %r1,[srcPitch];
 ld.param.u64 %rd2,[dst]; ld.param.u32 %r2,[dstPitch];
 ld.param.u32 %r3,[w]; ld.param.u32 %r4,[h];
 ld.param.f32 %f1,[intensity]; ld.param.f32 %f2,[centerX];
 ld.param.f32 %f3,[centerY]; ld.param.f32 %f4,[invHalfW];
 ld.param.f32 %f5,[invHalfH];
 mov.u32 %r5,%ctaid.x; mov.u32 %r6,%ntid.x; mov.u32 %r7,%tid.x;
 mad.lo.s32 %r8,%r5,%r6,%r7;
 mov.u32 %r9,%ctaid.y; mov.u32 %r10,%ntid.y; mov.u32 %r11,%tid.y;
 mad.lo.s32 %r12,%r9,%r10,%r11;
 setp.ge.u32 %p1,%r8,%r3; setp.ge.u32 %p2,%r12,%r4; or.pred %p3,%p1,%p2;
 @%p3 bra DONE;
 cvt.rn.f32.u32 %f6,%r8; add.f32 %f6,%f6,0f3F000000;
 sub.f32 %f6,%f6,%f2; mul.f32 %f6,%f6,%f4;
 cvt.rn.f32.u32 %f7,%r12; add.f32 %f7,%f7,0f3F000000;
 sub.f32 %f7,%f7,%f3; mul.f32 %f7,%f7,%f5;
 mul.f32 %f8,%f6,%f6; mad.rn.f32 %f8,%f7,%f7,%f8;
 sqrt.rn.f32 %f9,%f8; mul.f32 %f9,%f9,0f3F3504F3;
 max.f32 %f9,%f9,0f00000000; min.f32 %f9,%f9,0f3F800000;
 mul.f32 %f9,%f9,%f9; mul.f32 %f14,%f1,%f9;
 sub.f32 %f10,0f3F800000,%f14;
 max.f32 %f10,%f10,0f00000000;
 mul.wide.u32 %rd3,%r12,%r1; mul.wide.u32 %rd4,%r12,%r2;
 mul.lo.u32 %r13,%r8,3; cvt.u64.u32 %rd5,%r13;
 add.u64 %rd6,%rd1,%rd3; add.u64 %rd6,%rd6,%rd5;
 add.u64 %rd7,%rd2,%rd4; add.u64 %rd7,%rd7,%rd5;
 ld.global.u8 %r14,[%rd6]; ld.global.u8 %r15,[%rd6+1]; ld.global.u8 %r16,[%rd6+2];
 cvt.rn.f32.u32 %f11,%r14; cvt.rn.f32.u32 %f12,%r15; cvt.rn.f32.u32 %f13,%r16;
 mul.f32 %f11,%f11,%f10; mul.f32 %f12,%f12,%f10; mul.f32 %f13,%f13,%f10;
 cvt.rni.u32.f32 %r17,%f11; cvt.rni.u32.f32 %r18,%f12; cvt.rni.u32.f32 %r19,%f13;
 min.u32 %r17,%r17,255; min.u32 %r18,%r18,255; min.u32 %r19,%r19,255;
 st.global.u8 [%rd7],%r17; st.global.u8 [%rd7+1],%r18; st.global.u8 [%rd7+2],%r19;
DONE: ret;
})ptx";

bool Fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}
} // namespace

PreparedVignetteKernel::PreparedVignetteKernel(
    PreparedVignetteKernel &&other) noexcept
    : driver(other.driver), owner_context(other.owner_context),
      owner_stream(other.owner_stream), module(other.module),
      function(other.function), live(other.live) {
  other.driver = nullptr;
  other.owner_context = nullptr;
  other.owner_stream = nullptr;
  other.module = nullptr;
  other.function = nullptr;
  other.live = false;
}

bool PrepareVignetteKernel(CudaDriverApi &driver, CUcontext owner_context,
                           CUstream owner_stream, PreparedVignetteKernel *out,
                           std::string *error) {
  if (error)
    error->clear();
  if (!out || out->live || !owner_context || !owner_stream ||
      !driver.f().cuCtxGetCurrent || !driver.f().cuModuleLoadData ||
      !driver.f().cuModuleGetFunction || !driver.f().cuModuleUnload ||
      !driver.f().cuLaunchKernel)
    return Fail(error, "PrepareVignetteKernel: invalid owner or driver API.");
  CUcontext current = nullptr;
  if (driver.f().cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
      current != owner_context)
    return Fail(error, "PrepareVignetteKernel: owner context is not current.");
  CUmodule module = nullptr;
  auto status = driver.f().cuModuleLoadData(&module, kPtx);
  if (status != CUDA_SUCCESS)
    return Fail(error, "PrepareVignetteKernel: PTX module load failed.");
  CUfunction function = nullptr;
  status = driver.f().cuModuleGetFunction(&function, module, "vignette_u8x3");
  if (status != CUDA_SUCCESS || !function) {
    if (driver.f().cuModuleUnload(module) != CUDA_SUCCESS) {
      out->driver = &driver;
      out->owner_context = owner_context;
      out->owner_stream = owner_stream;
      out->module = module;
      out->live = true;
    }
    return Fail(error, "PrepareVignetteKernel: entry resolution failed.");
  }
  out->driver = &driver;
  out->owner_context = owner_context;
  out->owner_stream = owner_stream;
  out->module = module;
  out->function = function;
  out->live = true;
  return true;
}

bool LaunchVignettePrepared(const PreparedVignetteKernel &kernel,
                            const CudaImage &source,
                            const CudaImage &destination, float intensity,
                            float center_x, float center_y,
                            std::string *error) {
  if (error)
    error->clear();
  if (!kernel.live || !kernel.driver || !kernel.function ||
      !kernel.owner_context || !kernel.owner_stream || !source.Valid() ||
      !destination.Valid() || source.w != destination.w ||
      source.h != destination.h || source.format != destination.format ||
      (source.format != PixelFormatGpu::rgb_u8 &&
       source.format != PixelFormatGpu::bgr_u8))
    return Fail(error, "LaunchVignettePrepared: invalid kernel or images.");
  if (!detail::CheckSignedInt32PtxPitch(source.pitch, source.RowBytes(),
                                        "LaunchVignettePrepared(source)", error) ||
      !detail::CheckSignedInt32PtxPitch(destination.pitch,
                                        destination.RowBytes(),
                                        "LaunchVignettePrepared(destination)", error))
    return false;
  if (!std::isfinite(intensity) || intensity < 0.0f || intensity > 1.0f ||
      !std::isfinite(center_x) || !std::isfinite(center_y) || center_x < 0.0f ||
      center_y < 0.0f || center_x > static_cast<float>(source.w) ||
      center_y > static_cast<float>(source.h))
    return Fail(error, "LaunchVignettePrepared: invalid vignette parameters.");
  const auto src = static_cast<unsigned long long>(source.ptr);
  const auto dst = static_cast<unsigned long long>(destination.ptr);
  const auto src_pitch = static_cast<std::uint32_t>(source.pitch);
  const auto dst_pitch = static_cast<std::uint32_t>(destination.pitch);
  const auto width = static_cast<std::uint32_t>(source.w);
  const auto height = static_cast<std::uint32_t>(source.h);
  const float inv_half_width = 2.0f / static_cast<float>(width);
  const float inv_half_height = 2.0f / static_cast<float>(height);
  void *args[] = {const_cast<unsigned long long *>(&src),
                  const_cast<std::uint32_t *>(&src_pitch),
                  const_cast<unsigned long long *>(&dst),
                  const_cast<std::uint32_t *>(&dst_pitch),
                  const_cast<std::uint32_t *>(&width),
                  const_cast<std::uint32_t *>(&height), &intensity, &center_x,
                  &center_y, const_cast<float *>(&inv_half_width),
                  const_cast<float *>(&inv_half_height)};
  constexpr unsigned int block = 16;
  const auto status = kernel.driver->f().cuLaunchKernel(
      kernel.function, (width + block - 1u) / block,
      (height + block - 1u) / block, 1, block, block, 1, 0,
      kernel.owner_stream, args, nullptr);
  return status == CUDA_SUCCESS
             ? true
             : Fail(error, "LaunchVignettePrepared: kernel launch failed.");
}

bool DestroyVignetteKernelPrepared(PreparedVignetteKernel *kernel,
                                   std::string *error) {
  if (error)
    error->clear();
  if (!kernel || !kernel->live)
    return kernel != nullptr;
  CUcontext current = nullptr;
  if (!kernel->driver || !kernel->driver->f().cuCtxGetCurrent ||
      kernel->driver->f().cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
      current != kernel->owner_context)
    return Fail(error,
                "DestroyVignetteKernelPrepared: owner context is not current.");
  if (kernel->driver->f().cuModuleUnload(kernel->module) != CUDA_SUCCESS)
    return Fail(error,
                "DestroyVignetteKernelPrepared: module unload failed; retry required.");
  kernel->driver = nullptr;
  kernel->owner_context = nullptr;
  kernel->owner_stream = nullptr;
  kernel->module = nullptr;
  kernel->function = nullptr;
  kernel->live = false;
  return true;
}

} // namespace studiocast::cuda::kernels
