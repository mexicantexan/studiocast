#include "core/cuda/kernels/prepared_eye_contact_kernels.h"

#include "core/cuda/kernels/ptx_pitch_contract.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace studiocast::cuda::kernels {
extern const char kPreparedEyeContactPtx[];
namespace {

bool Fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}

bool Tensor(const CudaTensor *value, int c, int h, int w) {
  return value && value->Valid() && value->n == 1 && value->c == c &&
         value->h == h && value->w == w;
}

bool Span(CUdeviceptr ptr, std::size_t bytes, std::uint64_t *begin,
          std::uint64_t *end) {
  const auto base = static_cast<std::uint64_t>(ptr);
  if (!ptr || bytes == 0 ||
      base > std::numeric_limits<std::uint64_t>::max() - bytes)
    return false;
  *begin = base;
  *end = base + bytes;
  return true;
}

bool Disjoint(CUdeviceptr a, std::size_t as, CUdeviceptr b, std::size_t bs) {
  std::uint64_t ab, ae, bb, be;
  return Span(a, as, &ab, &ae) && Span(b, bs, &bb, &be) &&
         (ae <= bb || be <= ab);
}

std::size_t ImageBytes(const CudaImage &image) {
  if (!image.Valid() || image.h <= 0)
    return 0;
  const auto rows = static_cast<std::size_t>(image.h - 1);
  if (rows && image.pitch > std::numeric_limits<std::size_t>::max() / rows)
    return 0;
  const auto prefix = image.pitch * rows;
  return prefix > std::numeric_limits<std::size_t>::max() - image.RowBytes()
             ? 0
             : prefix + image.RowBytes();
}
} // namespace

PreparedEyeContactKernels::PreparedEyeContactKernels(
    PreparedEyeContactKernels &&other) noexcept
    : driver(other.driver), owner_context(other.owner_context),
      owner_stream(other.owner_stream), module(other.module),
      prepare_inputs(other.prepare_inputs), composite(other.composite),
      live(other.live) {
  other.driver = nullptr;
  other.owner_context = nullptr;
  other.owner_stream = nullptr;
  other.module = nullptr;
  other.prepare_inputs = nullptr;
  other.composite = nullptr;
  other.live = false;
}

bool PrepareEyeContactKernels(CudaDriverApi &driver, CUcontext context,
                              CUstream stream, PreparedEyeContactKernels *out,
                              std::string *error) {
  if (error)
    error->clear();
  if (!out || out->live || !context || !stream || !driver.f().cuCtxGetCurrent ||
      !driver.f().cuModuleLoadData || !driver.f().cuModuleGetFunction ||
      !driver.f().cuModuleUnload || !driver.f().cuLaunchKernel)
    return Fail(error,
                "PrepareEyeContactKernels: invalid owner or driver API.");
  CUcontext current = nullptr;
  if (driver.f().cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
      current != context)
    return Fail(error,
                "PrepareEyeContactKernels: owner context is not current.");
  CUmodule module = nullptr;
  if (driver.f().cuModuleLoadData(&module, kPreparedEyeContactPtx) !=
      CUDA_SUCCESS)
    return Fail(error, "PrepareEyeContactKernels: PTX module load failed.");
  CUfunction prepare = nullptr, composite = nullptr;
  if (driver.f().cuModuleGetFunction(
          &prepare, module, "eye_contact_prepare_inputs") != CUDA_SUCCESS ||
      !prepare ||
      driver.f().cuModuleGetFunction(&composite, module,
                                     "eye_contact_composite") != CUDA_SUCCESS ||
      !composite) {
    if (driver.f().cuModuleUnload(module) != CUDA_SUCCESS) {
      out->driver = &driver;
      out->owner_context = context;
      out->owner_stream = stream;
      out->module = module;
      out->live = true;
    }
    return Fail(error, "PrepareEyeContactKernels: entry resolution failed.");
  }
  out->driver = &driver;
  out->owner_context = context;
  out->owner_stream = stream;
  out->module = module;
  out->prepare_inputs = prepare;
  out->composite = composite;
  out->live = true;
  return true;
}

bool LaunchEyeContactInputsPrepared(const PreparedEyeContactKernels &k,
                                    const CudaImage &image,
                                    const PreparedEyeContactInputs &o,
                                    float strength, std::string *error) {
  if (error)
    error->clear();
  if (!k.live || !k.driver || !k.prepare_inputs || !image.Valid() ||
      image.format != PixelFormatGpu::rgb_u8 ||
      !Tensor(o.landmarks, 1, 1, 136) || !Tensor(o.crop_geometry, 2, 1, 5) ||
      !Tensor(o.left_eye_rgb, 3, 48, 64) ||
      !Tensor(o.right_eye_rgb, 3, 48, 64) ||
      !Tensor(o.left_anchors, 12, 48, 64) ||
      !Tensor(o.right_anchors, 12, 48, 64) || !Tensor(o.left_angles, 2, 1, 1) ||
      !Tensor(o.right_angles, 2, 1, 1) || !std::isfinite(strength) ||
      strength < 0 || strength > 1)
    return Fail(error, "LaunchEyeContactInputsPrepared: invalid contract.");
  if (!detail::CheckSignedInt32PtxPitch(image.pitch, image.RowBytes(),
                                        "LaunchEyeContactInputsPrepared",
                                        error))
    return false;
  const CudaTensor *values[] = {
      o.landmarks,    o.crop_geometry, o.left_eye_rgb, o.right_eye_rgb,
      o.left_anchors, o.right_anchors, o.left_angles,  o.right_angles};
  const auto image_bytes = ImageBytes(image);
  if (!image_bytes)
    return Fail(error, "LaunchEyeContactInputsPrepared: image span overflow.");
  for (const auto *value : values)
    if (!Disjoint(image.ptr, image_bytes, value->ptr, value->bytes))
      return Fail(error,
                  "LaunchEyeContactInputsPrepared: image/tensor overlap.");
  for (std::size_t i = 0; i < 8; ++i)
    for (std::size_t j = i + 1; j < 8; ++j)
      if (!Disjoint(values[i]->ptr, values[i]->bytes, values[j]->ptr,
                    values[j]->bytes))
        return Fail(error, "LaunchEyeContactInputsPrepared: tensor overlap.");
  const auto src = static_cast<unsigned long long>(image.ptr);
  const auto pitch = static_cast<std::uint32_t>(image.pitch);
  const auto width = static_cast<std::uint32_t>(image.w),
             height = static_cast<std::uint32_t>(image.h);
  unsigned long long ptrs[] = {
      static_cast<unsigned long long>(o.landmarks->ptr),
      static_cast<unsigned long long>(o.crop_geometry->ptr),
      static_cast<unsigned long long>(o.left_eye_rgb->ptr),
      static_cast<unsigned long long>(o.right_eye_rgb->ptr),
      static_cast<unsigned long long>(o.left_anchors->ptr),
      static_cast<unsigned long long>(o.right_anchors->ptr),
      static_cast<unsigned long long>(o.left_angles->ptr),
      static_cast<unsigned long long>(o.right_angles->ptr)};
  void *args[] = {const_cast<unsigned long long *>(&src),
                  const_cast<std::uint32_t *>(&pitch),
                  const_cast<std::uint32_t *>(&width),
                  const_cast<std::uint32_t *>(&height),
                  &ptrs[0],
                  &ptrs[1],
                  &ptrs[2],
                  &ptrs[3],
                  &ptrs[4],
                  &ptrs[5],
                  &ptrs[6],
                  &ptrs[7],
                  &strength};
  return k.driver->f().cuLaunchKernel(k.prepare_inputs, 24, 1, 1, 256, 1, 1, 0,
                                      k.owner_stream, args,
                                      nullptr) == CUDA_SUCCESS ||
         Fail(error, "LaunchEyeContactInputsPrepared: launch failed.");
}

bool LaunchEyeContactCompositePrepared(const PreparedEyeContactKernels &k,
                                       const CudaImage &source,
                                       const CudaTensor &geometry,
                                       const CudaTensor &left,
                                       const CudaTensor &right, bool left_ok,
                                       bool right_ok, float strength,
                                       CudaImage *output, std::string *error) {
  if (error)
    error->clear();
  if (!k.live || !k.driver || !k.composite || !output || !source.Valid() ||
      !output->Valid() || source.format != PixelFormatGpu::rgb_u8 ||
      output->format != source.format || output->w != source.w ||
      output->h != source.h || !Tensor(&geometry, 2, 1, 5) ||
      !Tensor(&left, 3, 48, 64) || !Tensor(&right, 3, 48, 64) ||
      !std::isfinite(strength) || strength < 0 || strength > 1)
    return Fail(error, "LaunchEyeContactCompositePrepared: invalid contract.");
  const auto source_bytes = ImageBytes(source),
             output_bytes = ImageBytes(*output);
  if (!source_bytes || !output_bytes ||
      !Disjoint(source.ptr, source_bytes, output->ptr, output_bytes) ||
      !Disjoint(output->ptr, output_bytes, geometry.ptr, geometry.bytes) ||
      !Disjoint(output->ptr, output_bytes, left.ptr, left.bytes) ||
      !Disjoint(output->ptr, output_bytes, right.ptr, right.bytes) ||
      !Disjoint(source.ptr, source_bytes, geometry.ptr, geometry.bytes) ||
      !Disjoint(source.ptr, source_bytes, left.ptr, left.bytes) ||
      !Disjoint(source.ptr, source_bytes, right.ptr, right.bytes) ||
      !Disjoint(geometry.ptr, geometry.bytes, left.ptr, left.bytes) ||
      !Disjoint(geometry.ptr, geometry.bytes, right.ptr, right.bytes) ||
      !Disjoint(left.ptr, left.bytes, right.ptr, right.bytes))
    return Fail(error,
                "LaunchEyeContactCompositePrepared: output must be disjoint.");
  if (!detail::CheckSignedInt32PtxPitch(source.pitch, source.RowBytes(),
                                        "EyeContact source", error) ||
      !detail::CheckSignedInt32PtxPitch(output->pitch, output->RowBytes(),
                                        "EyeContact output", error))
    return false;
  const auto src = static_cast<unsigned long long>(source.ptr),
             dst = static_cast<unsigned long long>(output->ptr);
  const auto geo = static_cast<unsigned long long>(geometry.ptr),
             lp = static_cast<unsigned long long>(left.ptr),
             rp = static_cast<unsigned long long>(right.ptr);
  const auto sp = static_cast<std::uint32_t>(source.pitch),
             dp = static_cast<std::uint32_t>(output->pitch);
  const auto w = static_cast<std::uint32_t>(source.w),
             h = static_cast<std::uint32_t>(source.h);
  const std::uint32_t lok = left_ok, rok = right_ok;
  void *args[] = {const_cast<unsigned long long *>(&src),
                  const_cast<std::uint32_t *>(&sp),
                  const_cast<unsigned long long *>(&dst),
                  const_cast<std::uint32_t *>(&dp),
                  const_cast<std::uint32_t *>(&w),
                  const_cast<std::uint32_t *>(&h),
                  const_cast<unsigned long long *>(&geo),
                  const_cast<unsigned long long *>(&lp),
                  const_cast<unsigned long long *>(&rp),
                  const_cast<std::uint32_t *>(&lok),
                  const_cast<std::uint32_t *>(&rok),
                  &strength};
  constexpr unsigned block = 16;
  return k.driver->f().cuLaunchKernel(
             k.composite, (w + 15) / block, (h + 15) / block, 1, block, block,
             1, 0, k.owner_stream, args, nullptr) == CUDA_SUCCESS ||
         Fail(error, "LaunchEyeContactCompositePrepared: launch failed.");
}

bool DestroyEyeContactKernelsPrepared(PreparedEyeContactKernels *k,
                                      std::string *error) {
  if (error)
    error->clear();
  if (!k || !k->live)
    return k != nullptr;
  CUcontext current = nullptr;
  if (!k->driver || !k->driver->f().cuCtxGetCurrent ||
      k->driver->f().cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
      current != k->owner_context)
    return Fail(
        error,
        "DestroyEyeContactKernelsPrepared: owner context is not current.");
  if (k->driver->f().cuModuleUnload(k->module) != CUDA_SUCCESS)
    return Fail(
        error,
        "DestroyEyeContactKernelsPrepared: unload failed; retry required.");
  k->driver = nullptr;
  k->owner_context = nullptr;
  k->owner_stream = nullptr;
  k->module = nullptr;
  k->prepare_inputs = nullptr;
  k->composite = nullptr;
  k->live = false;
  return true;
}
} // namespace studiocast::cuda::kernels
