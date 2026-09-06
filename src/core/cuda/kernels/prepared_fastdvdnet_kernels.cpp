#include "core/cuda/kernels/prepared_fastdvdnet_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace studiocast::cuda::kernels {
namespace {
constexpr const char *kPtx = R"ptx(
.version 6.0
.target sm_30
.address_size 64
.visible .entry fastdvdnet_preprocess(
 .param .u64 src,.param .u32 pitch,.param .u32 sw,.param .u32 sh,
 .param .u64 oldest,.param .u64 middle,.param .u64 current,
 .param .u64 noisy,.param .u64 noise,.param .u32 pw,.param .u32 ph,
 .param .u32 reset,.param .f32 sigma)
{
 .reg .pred %p<4>; .reg .b32 %r<20>; .reg .b64 %rd<20>; .reg .f32 %f<5>;
 mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3;
 mov.u32 %r5,%ctaid.y; mov.u32 %r6,%ntid.y; mov.u32 %r7,%tid.y; mad.lo.s32 %r8,%r5,%r6,%r7;
 ld.param.u32 %r9,[pw]; ld.param.u32 %r10,[ph]; setp.ge.u32 %p1,%r4,%r9; setp.ge.u32 %p2,%r8,%r10; or.pred %p3,%p1,%p2; @%p3 bra DONE;
 ld.param.u32 %r11,[sw]; ld.param.u32 %r12,[sh]; sub.u32 %r13,%r11,1; min.u32 %r13,%r4,%r13; sub.u32 %r14,%r12,1; min.u32 %r14,%r8,%r14;
 ld.param.u32 %r15,[pitch]; mul.wide.u32 %rd1,%r14,%r15; mad.wide.u32 %rd2,%r13,3,%rd1; ld.param.u64 %rd3,[src]; add.u64 %rd3,%rd3,%rd2;
 mul.lo.u32 %r16,%r9,%r10; mad.lo.s32 %r17,%r8,%r9,%r4; cvt.u64.u32 %rd4,%r17; mul.wide.u32 %rd4,%r17,4;
 ld.param.u64 %rd5,[current]; ld.param.u64 %rd6,[noisy]; ld.param.u64 %rd7,[oldest]; ld.param.u64 %rd8,[middle]; ld.param.u32 %r18,[reset];
 mov.u32 %r19,0;
LOOP: setp.ge.u32 %p1,%r19,3; @%p1 bra MAP;
 ld.global.u8 %r1,[%rd3]; cvt.rn.f32.u32 %f1,%r1; mul.f32 %f1,%f1,0f3B808081; st.global.f32 [%rd5+%rd4],%f1;
 setp.ne.u32 %p2,%r18,0; selp.b64 %rd9,%rd5,%rd7,%p2; selp.b64 %rd10,%rd5,%rd8,%p2;
 st.global.f32 [%rd6+%rd4],%f1; mad.wide.u32 %rd11,%r16,4,%rd6; add.u64 %rd11,%rd11,%rd4; ld.global.f32 %f2,[%rd9+%rd4]; st.global.f32 [%rd11],%f2;
 mad.wide.u32 %rd12,%r16,8,%rd6; add.u64 %rd12,%rd12,%rd4; ld.global.f32 %f3,[%rd10+%rd4]; st.global.f32 [%rd12],%f3;
 mad.wide.u32 %rd13,%r16,12,%rd6; add.u64 %rd13,%rd13,%rd4; st.global.f32 [%rd13],%f1; mad.wide.u32 %rd14,%r16,16,%rd6; add.u64 %rd14,%rd14,%rd4; st.global.f32 [%rd14],%f1;
 add.u64 %rd3,%rd3,1; mad.wide.u32 %rd15,%r16,4,%rd5; add.u64 %rd5,%rd15,%rd5; mad.wide.u32 %rd16,%r16,4,%rd7; add.u64 %rd7,%rd16,%rd7; mad.wide.u32 %rd17,%r16,4,%rd8; add.u64 %rd8,%rd17,%rd8; add.u32 %r19,%r19,1; bra LOOP;
MAP: ld.param.u64 %rd18,[noise]; ld.param.f32 %f4,[sigma]; st.global.f32 [%rd18+%rd4],%f4;
DONE: ret;
}
.visible .entry fastdvdnet_postprocess(
 .param .u64 src,.param .u32 pw,.param .u64 dst,.param .u32 pitch,.param .u32 w,.param .u32 h)
{
 .reg .pred %p<3>; .reg .b32 %r<18>; .reg .b64 %rd<15>; .reg .f32 %f<5>;
 mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x; mad.lo.s32 %r4,%r1,%r2,%r3; mov.u32 %r5,%ctaid.y; mov.u32 %r6,%ntid.y; mov.u32 %r7,%tid.y; mad.lo.s32 %r8,%r5,%r6,%r7;
 ld.param.u32 %r9,[w]; ld.param.u32 %r10,[h]; setp.ge.u32 %p1,%r4,%r9; setp.ge.u32 %p2,%r8,%r10; or.pred %p1,%p1,%p2; @%p1 bra OUT;
 ld.param.u32 %r11,[pw]; mad.lo.s32 %r12,%r8,%r11,%r4; mul.wide.u32 %rd1,%r12,4; mul.lo.u32 %r13,%r11,%r10; mul.wide.u32 %rd2,%r13,4; ld.param.u64 %rd3,[src]; ld.global.f32 %f1,[%rd3+%rd1]; add.u64 %rd4,%rd3,%rd2; ld.global.f32 %f2,[%rd4+%rd1]; add.u64 %rd4,%rd4,%rd2; ld.global.f32 %f3,[%rd4+%rd1]; max.f32 %f1,%f1,0f00000000; min.f32 %f1,%f1,0f3F800000; max.f32 %f2,%f2,0f00000000; min.f32 %f2,%f2,0f3F800000; max.f32 %f3,%f3,0f00000000; min.f32 %f3,%f3,0f3F800000; mul.f32 %f1,%f1,0f437F0000; mul.f32 %f2,%f2,0f437F0000; mul.f32 %f3,%f3,0f437F0000; cvt.rni.u32.f32 %r14,%f1; cvt.rni.u32.f32 %r15,%f2; cvt.rni.u32.f32 %r16,%f3;
 ld.param.u32 %r17,[pitch]; mul.wide.u32 %rd5,%r8,%r17; mad.wide.u32 %rd6,%r4,3,%rd5; ld.param.u64 %rd7,[dst]; add.u64 %rd7,%rd7,%rd6; st.global.u8 [%rd7],%r14; st.global.u8 [%rd7+1],%r15; st.global.u8 [%rd7+2],%r16;
OUT: ret;
})ptx";

bool Fail(std::string *error, const char *text) noexcept {
  if (error) *error = text;
  return false;
}
} // namespace

bool PrepareFastDvdnetKernels(CudaDriverApi &driver, CUcontext context,
                              CUstream stream, PreparedFastDvdnetKernels *out,
                              std::string *error) {
  if (error) error->clear();
  if (!out || out->live || !context || !stream || !driver.f().cuCtxGetCurrent ||
      !driver.f().cuModuleLoadData || !driver.f().cuModuleGetFunction ||
      !driver.f().cuModuleUnload || !driver.f().cuLaunchKernel)
    return Fail(error, "FastDVDnet kernel setup contract is invalid");
  CUcontext current = nullptr;
  if (driver.f().cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != context)
    return Fail(error, "FastDVDnet CUDA context is not current");
  CUmodule module = nullptr;
  if (driver.f().cuModuleLoadData(&module, kPtx) != CUDA_SUCCESS)
    return Fail(error, "FastDVDnet PTX module load failed");
  CUfunction pre = nullptr, post = nullptr;
  if (driver.f().cuModuleGetFunction(&pre, module, "fastdvdnet_preprocess") != CUDA_SUCCESS ||
      driver.f().cuModuleGetFunction(&post, module, "fastdvdnet_postprocess") != CUDA_SUCCESS || !pre || !post) {
    driver.f().cuModuleUnload(module);
    return Fail(error, "FastDVDnet kernel entry resolution failed");
  }
  *out = {&driver, context, stream, module, pre, post, true};
  return true;
}

bool LaunchFastDvdnetPreprocess(const PreparedFastDvdnetKernels &k,
                                const CudaImage &src, const CudaTensor &oldest,
                                const CudaTensor &middle, const CudaTensor &current,
                                const CudaTensor &noisy, const CudaTensor &noise,
                                bool reset, float sigma, std::string *error) noexcept {
  if (error) error->clear();
  if (!k.live || !src.Valid() || src.format != PixelFormatGpu::rgb_u8 ||
      !oldest.Valid() || !middle.Valid() || !current.Valid() || !noisy.Valid() ||
      !noise.Valid() || oldest.c != 3 || middle.c != 3 || current.c != 3 ||
      noisy.c != 15 || noise.c != 1 || oldest.w != noisy.w || oldest.h != noisy.h ||
      !std::isfinite(sigma) || sigma < 0.0f || sigma > 1.0f)
    return Fail(error, "FastDVDnet preprocess arguments are invalid");
  auto sp=static_cast<std::uint32_t>(src.pitch), sw=static_cast<std::uint32_t>(src.w), sh=static_cast<std::uint32_t>(src.h), pw=static_cast<std::uint32_t>(noisy.w), ph=static_cast<std::uint32_t>(noisy.h), rst=reset?1u:0u;
  auto s=src.ptr,o=oldest.ptr,m=middle.ptr,c=current.ptr,n=noisy.ptr,nm=noise.ptr;
  void *args[]={&s,&sp,&sw,&sh,&o,&m,&c,&n,&nm,&pw,&ph,&rst,&sigma};
  return k.driver->f().cuLaunchKernel(k.preprocess,(pw+15)/16,(ph+15)/16,1,16,16,1,0,k.stream,args,nullptr)==CUDA_SUCCESS || Fail(error,"FastDVDnet preprocess launch failed");
}

bool LaunchFastDvdnetPostprocess(const PreparedFastDvdnetKernels &k,
                                 const CudaTensor &src, const CudaImage &dst,
                                 std::string *error) noexcept {
  if (error) error->clear();
  if (!k.live || !src.Valid() || src.c != 3 || !dst.Valid() || dst.format != PixelFormatGpu::rgb_u8 || dst.w > src.w || dst.h > src.h)
    return Fail(error,"FastDVDnet postprocess arguments are invalid");
  auto s=src.ptr;
  auto d=dst.ptr;
  auto pw=static_cast<std::uint32_t>(src.w);
  auto pitch=static_cast<std::uint32_t>(dst.pitch);
  auto w=static_cast<std::uint32_t>(dst.w);
  auto h=static_cast<std::uint32_t>(dst.h);
  void *args[]={&s,&pw,&d,&pitch,&w,&h};
  return k.driver->f().cuLaunchKernel(k.postprocess,(w+15)/16,(h+15)/16,1,16,16,1,0,k.stream,args,nullptr)==CUDA_SUCCESS || Fail(error,"FastDVDnet postprocess launch failed");
}

bool DestroyFastDvdnetKernels(PreparedFastDvdnetKernels *k,
                              std::string *error) noexcept {
  if (error) error->clear();
  if (!k || !k->live) return true;
  if (k->driver->f().cuModuleUnload(k->module) != CUDA_SUCCESS)
    return Fail(error,"FastDVDnet module unload failed");
  *k={}; return true;
}
} // namespace studiocast::cuda::kernels
