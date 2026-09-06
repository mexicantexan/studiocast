#include "core/cuda/kernels/prepared_mirror.h"

#include <cstdint>
#include <limits>

namespace studiocast::cuda::kernels {
namespace {

bool Fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}

bool ImageSpan(const CudaImage &image, std::uint64_t *begin,
               std::uint64_t *end) noexcept {
  const auto base = static_cast<std::uint64_t>(image.ptr);
  const auto rows = static_cast<std::uint64_t>(image.h - 1);
  const auto pitch = static_cast<std::uint64_t>(image.pitch);
  const auto row_bytes = static_cast<std::uint64_t>(image.RowBytes());
  if ((rows != 0 && pitch > std::numeric_limits<std::uint64_t>::max() / rows))
    return false;
  const auto row_offset = rows * pitch;
  if (row_offset > std::numeric_limits<std::uint64_t>::max() - row_bytes ||
      base >
          std::numeric_limits<std::uint64_t>::max() - (row_offset + row_bytes))
    return false;
  *begin = base;
  *end = base + row_offset + row_bytes;
  return true;
}

} // namespace

PreparedMirrorKernel::PreparedMirrorKernel(
    PreparedMirrorKernel &&other) noexcept
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

bool PrepareMirrorKernel(CudaDriverApi &driver, CUcontext owner_context,
                         CUstream owner_stream, PreparedMirrorKernel *out,
                         std::string *error) {
  if (error)
    error->clear();
  if (!out || out->live || !owner_context || !owner_stream ||
      !driver.f().cuCtxGetCurrent || !driver.f().cuModuleLoadData ||
      !driver.f().cuModuleGetFunction || !driver.f().cuModuleUnload ||
      !driver.f().cuLaunchKernel)
    return Fail(error, "PrepareMirrorKernel: invalid owner or driver API.");
  CUcontext current = nullptr;
  if (driver.f().cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
      current != owner_context)
    return Fail(error, "PrepareMirrorKernel: owner context is not current.");

  CUmodule module = nullptr;
  if (driver.f().cuModuleLoadData(&module, kPreparedMirrorPtx) != CUDA_SUCCESS)
    return Fail(error, "PrepareMirrorKernel: PTX module load failed.");
  CUfunction function = nullptr;
  if (driver.f().cuModuleGetFunction(
          &function, module, "mirror_horizontal_u8x3") != CUDA_SUCCESS ||
      !function) {
    if (driver.f().cuModuleUnload(module) != CUDA_SUCCESS) {
      out->driver = &driver;
      out->owner_context = owner_context;
      out->owner_stream = owner_stream;
      out->module = module;
      out->live = true;
    }
    return Fail(error, "PrepareMirrorKernel: entry resolution failed.");
  }

  out->driver = &driver;
  out->owner_context = owner_context;
  out->owner_stream = owner_stream;
  out->module = module;
  out->function = function;
  out->live = true;
  return true;
}

bool LaunchMirrorPrepared(const PreparedMirrorKernel &kernel,
                          const CudaImage &source, const CudaImage &destination,
                          std::string *error) {
  if (error)
    error->clear();
  if (!kernel.live || !kernel.driver || !kernel.function ||
      !kernel.owner_context || !kernel.owner_stream || !source.Valid() ||
      !destination.Valid() || source.w != destination.w ||
      source.h != destination.h || source.format != destination.format ||
      (source.format != PixelFormatGpu::rgb_u8 &&
       source.format != PixelFormatGpu::bgr_u8))
    return Fail(error, "LaunchMirrorPrepared: invalid kernel or images.");

  CUcontext current = nullptr;
  if (!kernel.driver->f().cuCtxGetCurrent ||
      kernel.driver->f().cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
      current != kernel.owner_context)
    return Fail(error, "LaunchMirrorPrepared: owner context is not current.");

  std::uint64_t source_begin = 0, source_end = 0, destination_begin = 0,
                destination_end = 0;
  if (!ImageSpan(source, &source_begin, &source_end) ||
      !ImageSpan(destination, &destination_begin, &destination_end))
    return Fail(error, "LaunchMirrorPrepared: device image span overflow.");
  if (source_begin < destination_end && destination_begin < source_end)
    return Fail(
        error,
        "LaunchMirrorPrepared: source and destination must be disjoint.");

  const auto src = static_cast<unsigned long long>(source.ptr);
  const auto src_pitch = static_cast<unsigned long long>(source.pitch);
  const auto dst = static_cast<unsigned long long>(destination.ptr);
  const auto dst_pitch = static_cast<unsigned long long>(destination.pitch);
  const auto width = static_cast<std::uint32_t>(source.w);
  const auto height = static_cast<std::uint32_t>(source.h);
  void *args[] = {const_cast<unsigned long long *>(&src),
                  const_cast<unsigned long long *>(&src_pitch),
                  const_cast<unsigned long long *>(&dst),
                  const_cast<unsigned long long *>(&dst_pitch),
                  const_cast<std::uint32_t *>(&width),
                  const_cast<std::uint32_t *>(&height)};
  constexpr unsigned int block = 16;
  const auto status = kernel.driver->f().cuLaunchKernel(
      kernel.function, (width + block - 1u) / block,
      (height + block - 1u) / block, 1, block, block, 1, 0, kernel.owner_stream,
      args, nullptr);
  return status == CUDA_SUCCESS
             ? true
             : Fail(error, "LaunchMirrorPrepared: kernel launch failed.");
}

bool DestroyMirrorKernelPrepared(PreparedMirrorKernel *kernel,
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
                "DestroyMirrorKernelPrepared: owner context is not current.");
  if (kernel->driver->f().cuModuleUnload(kernel->module) != CUDA_SUCCESS)
    return Fail(
        error,
        "DestroyMirrorKernelPrepared: module unload failed; retry required.");
  kernel->driver = nullptr;
  kernel->owner_context = nullptr;
  kernel->owner_stream = nullptr;
  kernel->module = nullptr;
  kernel->function = nullptr;
  kernel->live = false;
  return true;
}

} // namespace studiocast::cuda::kernels
