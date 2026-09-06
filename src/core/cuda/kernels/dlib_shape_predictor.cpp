#include "core/cuda/kernels/dlib_shape_predictor.h"
#include <cstdint>

namespace studiocast::cuda::kernels {
extern const char kDlibShapePredictorPtx[];
namespace {
bool Fail(std::string *e, const char *m) {
  if (e)
    *e = m;
  return false;
}
} // namespace
bool PrepareDlibShapePredictorKernel(CudaDriverApi &d, CUcontext c, CUstream s,
                                     PreparedDlibShapePredictorKernel *o,
                                     std::string *e) {
  if (e)
    e->clear();
  if (!o || o->module || !c || !s || !d.f().cuCtxGetCurrent ||
      !d.f().cuModuleLoadData || !d.f().cuModuleGetFunction ||
      !d.f().cuModuleUnload || !d.f().cuLaunchKernel)
    return Fail(e, "invalid dlib CUDA kernel setup");
  CUcontext current = nullptr;
  if (d.f().cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != c)
    return Fail(e, "dlib CUDA owner context is not current");
  CUmodule m = nullptr;
  if (d.f().cuModuleLoadData(&m, kDlibShapePredictorPtx) != CUDA_SUCCESS)
    return Fail(e, "dlib CUDA PTX load failed");
  CUfunction f = nullptr;
  if (d.f().cuModuleGetFunction(&f, m, "dlib_shape_predictor_68") !=
          CUDA_SUCCESS ||
      !f) {
    if (d.f().cuModuleUnload(m) != CUDA_SUCCESS) {
      o->driver = &d;
      o->context = c;
      o->stream = s;
      o->module = m;
    }
    return Fail(e, "dlib CUDA entry resolution failed");
  }
  o->driver = &d;
  o->context = c;
  o->stream = s;
  o->module = m;
  o->function = f;
  return true;
}
bool LaunchDlibShapePredictor(const PreparedDlibShapePredictorKernel &k,
                              const CudaImage &im, float l, float t, float r,
                              float b, CUdeviceptr initial, CUdeviceptr ct,
                              CUdeviceptr ts, CUdeviceptr tl, CUdeviceptr sp,
                              CUdeviceptr leaves, CUdeviceptr cf,
                              CUdeviceptr anchors, CUdeviceptr deltas,
                              unsigned int cascades, CUdeviceptr shape,
                              CUdeviceptr features, CUdeviceptr output,
                              std::string *e) {
  if (e)
    e->clear();
  if (!k.driver || !k.function || !im.Valid() ||
      im.format != PixelFormatGpu::rgb_u8 || !(r > l && b > t))
    return Fail(e, "invalid dlib CUDA execution");
  auto rgb = (unsigned long long)im.ptr;
  auto pitch = (unsigned int)im.pitch, w = (unsigned int)im.w,
       h = (unsigned int)im.h;
  auto cv = [](CUdeviceptr p) { return (unsigned long long)p; };
  auto i = cv(initial), a = cv(ct), c = cv(ts), d = cv(tl), f = cv(sp),
       g = cv(leaves), j = cv(cf), q = cv(anchors), x = cv(deltas),
       y = cv(shape), z = cv(features), out = cv(output);
  void *args[] = {&rgb, &pitch, &w, &h, &l, &t, &r,        &b, &i, &a,  &c,
                  &d,   &f,     &g, &j, &q, &x, &cascades, &y, &z, &out};
  return k.driver->f().cuLaunchKernel(k.function, 1, 1, 1, 1, 1, 1, 0, k.stream,
                                      args, nullptr) == CUDA_SUCCESS
             ? true
             : Fail(e, "dlib CUDA launch failed");
}
bool DestroyDlibShapePredictorKernel(PreparedDlibShapePredictorKernel *k,
                                     std::string *e) {
  if (e)
    e->clear();
  if (!k)
    return false;
  if (!k->module)
    return true;
  CUcontext c = nullptr;
  if (!k->driver || !k->driver->f().cuCtxGetCurrent ||
      k->driver->f().cuCtxGetCurrent(&c) != CUDA_SUCCESS || c != k->context)
    return Fail(e, "dlib CUDA owner context is not current");
  if (k->driver->f().cuModuleUnload(k->module) != CUDA_SUCCESS)
    return Fail(e, "dlib CUDA module unload failed; retry required");
  k->driver = nullptr;
  k->context = nullptr;
  k->stream = nullptr;
  k->module = nullptr;
  k->function = nullptr;
  return true;
}
} // namespace studiocast::cuda::kernels
