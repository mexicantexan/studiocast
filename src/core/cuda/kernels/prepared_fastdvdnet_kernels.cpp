#include "core/cuda/kernels/prepared_fastdvdnet_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace studiocast::cuda::kernels {
namespace {
constexpr const char *kPtx =
#include "core/cuda/kernels/prepared_fastdvdnet_kernels_ptx.inc"
    ;

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
  *out = {&driver, context, stream, module, pre, post, CUDA_SUCCESS, true};
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
  k.last_result=k.driver->f().cuLaunchKernel(k.preprocess,(pw+15)/16,(ph+15)/16,1,16,16,1,0,k.stream,args,nullptr);
  return k.last_result==CUDA_SUCCESS || Fail(error,"FastDVDnet preprocess launch failed");
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
  auto ph=static_cast<std::uint32_t>(src.h);
  auto pitch=static_cast<std::uint32_t>(dst.pitch);
  auto w=static_cast<std::uint32_t>(dst.w);
  auto h=static_cast<std::uint32_t>(dst.h);
  void *args[]={&s,&pw,&ph,&d,&pitch,&w,&h};
  k.last_result=k.driver->f().cuLaunchKernel(k.postprocess,(w+15)/16,(h+15)/16,1,16,16,1,0,k.stream,args,nullptr);
  return k.last_result==CUDA_SUCCESS || Fail(error,"FastDVDnet postprocess launch failed");
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
