#include "core/cuda/kernels/prepared_matte_bounds.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace studiocast::cuda::kernels {
static_assert(sizeof(DeviceMatteBoundsRecord) == 20);
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
  if ((rows && pitch > std::numeric_limits<std::uint64_t>::max() / rows))
    return false;
  const auto offset = rows * pitch;
  if (offset > std::numeric_limits<std::uint64_t>::max() - row_bytes ||
      base > std::numeric_limits<std::uint64_t>::max() - offset - row_bytes)
    return false;
  *begin = base;
  *end = base + offset + row_bytes;
  return true;
}

void Clear(PreparedMatteBounds *p) noexcept {
  p->driver = nullptr;
  p->owner_context = nullptr;
  p->owner_stream = nullptr;
  p->module = nullptr;
  p->initialize = nullptr;
  p->reduce = nullptr;
  p->device_record = 0;
  p->device_record_bytes = 0;
  p->live = false;
}
} // namespace

PreparedMatteBounds::PreparedMatteBounds(PreparedMatteBounds &&other) noexcept
    : driver(other.driver), owner_context(other.owner_context),
      owner_stream(other.owner_stream), module(other.module),
      initialize(other.initialize), reduce(other.reduce),
      device_record(other.device_record),
      device_record_bytes(other.device_record_bytes), live(other.live) {
  Clear(&other);
}

bool PrepareMatteBounds(CudaDriverApi &driver, CUcontext context,
                        CUstream stream, std::size_t budget,
                        PreparedMatteBounds *out, std::string *error) {
  if (error)
    error->clear();
  if (!out || out->live || !context || !stream ||
      budget < sizeof(DeviceMatteBoundsRecord) || !driver.f().cuCtxGetCurrent ||
      !driver.f().cuModuleLoadData || !driver.f().cuModuleGetFunction ||
      !driver.f().cuModuleUnload || !driver.f().cuLaunchKernel ||
      !driver.f().cuMemAllocPitch || !driver.f().cuMemFree)
    return Fail(error,
                "PrepareMatteBounds: invalid owner, budget, or driver API.");
  CUcontext current = nullptr;
  if (driver.f().cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
      current != context)
    return Fail(error, "PrepareMatteBounds: owner context is not current.");

  CUmodule module = nullptr;
  if (driver.f().cuModuleLoadData(&module, kPreparedMatteBoundsPtx) !=
      CUDA_SUCCESS)
    return Fail(error, "PrepareMatteBounds: PTX module load failed.");
  CUfunction initialize = nullptr, reduce = nullptr;
  if (driver.f().cuModuleGetFunction(
          &initialize, module, "initialize_matte_bounds") != CUDA_SUCCESS ||
      !initialize ||
      driver.f().cuModuleGetFunction(&reduce, module, "reduce_matte_bounds") !=
          CUDA_SUCCESS ||
      !reduce) {
    if (driver.f().cuModuleUnload(module) != CUDA_SUCCESS) {
      out->driver = &driver;
      out->owner_context = context;
      out->owner_stream = stream;
      out->module = module;
      out->live = true;
    }
    return Fail(error, "PrepareMatteBounds: entry resolution failed.");
  }
  CUdeviceptr record = 0;
  std::size_t pitch = 0;
  if (driver.f().cuMemAllocPitch(&record, &pitch,
                                 sizeof(DeviceMatteBoundsRecord), 1,
                                 4) != CUDA_SUCCESS ||
      !record) {
    if (driver.f().cuModuleUnload(module) != CUDA_SUCCESS) {
      out->driver = &driver;
      out->owner_context = context;
      out->owner_stream = stream;
      out->module = module;
      out->initialize = initialize;
      out->reduce = reduce;
      out->live = true;
    }
    return Fail(error, "PrepareMatteBounds: device record allocation failed.");
  }
  if (pitch < sizeof(DeviceMatteBoundsRecord) || pitch > budget) {
    const bool freed = driver.f().cuMemFree(record) == CUDA_SUCCESS;
    const bool unloaded = driver.f().cuModuleUnload(module) == CUDA_SUCCESS;
    if (!freed || !unloaded) {
      out->driver = &driver;
      out->owner_context = context;
      out->owner_stream = stream;
      out->module = unloaded ? nullptr : module;
      out->initialize = unloaded ? nullptr : initialize;
      out->reduce = unloaded ? nullptr : reduce;
      out->device_record = freed ? 0 : record;
      out->device_record_bytes = freed ? 0 : pitch;
      out->live = true;
    }
    return Fail(error, "PrepareMatteBounds: exact device budget exceeded.");
  }
  out->driver = &driver;
  out->owner_context = context;
  out->owner_stream = stream;
  out->module = module;
  out->initialize = initialize;
  out->reduce = reduce;
  out->device_record = record;
  out->device_record_bytes = pitch;
  out->live = true;
  return true;
}

bool LaunchMatteBoundsPrepared(const PreparedMatteBounds &p,
                               const CudaImage &alpha, std::string *error) {
  if (error)
    error->clear();
  if (!p.live || !p.driver || !p.module || !p.initialize || !p.reduce ||
      !p.owner_context || !p.owner_stream || !p.device_record ||
      !alpha.Valid() || alpha.format != PixelFormatGpu::f32_1 || alpha.w <= 0 ||
      alpha.h <= 0)
    return Fail(
        error,
        "LaunchMatteBoundsPrepared: invalid prepared state or alpha image.");
  CUcontext current = nullptr;
  if (!p.driver->f().cuCtxGetCurrent ||
      p.driver->f().cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
      current != p.owner_context)
    return Fail(error,
                "LaunchMatteBoundsPrepared: owner context is not current.");
  const auto pixels =
      static_cast<std::uint64_t>(alpha.w) * static_cast<std::uint64_t>(alpha.h);
  if (pixels > std::numeric_limits<std::uint32_t>::max())
    return Fail(error, "LaunchMatteBoundsPrepared: foreground count overflow.");
  std::uint64_t begin = 0, end = 0;
  if (!ImageSpan(alpha, &begin, &end))
    return Fail(error, "LaunchMatteBoundsPrepared: alpha span overflow.");
  const auto record_begin = static_cast<std::uint64_t>(p.device_record);
  if (record_begin > std::numeric_limits<std::uint64_t>::max() -
                         sizeof(DeviceMatteBoundsRecord))
    return Fail(error, "LaunchMatteBoundsPrepared: output span overflow.");
  const auto record_end = record_begin + sizeof(DeviceMatteBoundsRecord);
  if (begin < record_end && record_begin < end)
    return Fail(error, "LaunchMatteBoundsPrepared: alpha and output overlap.");

  const auto record = static_cast<unsigned long long>(p.device_record);
  const auto width = static_cast<std::uint32_t>(alpha.w);
  const auto height = static_cast<std::uint32_t>(alpha.h);
  void *initialize_args[] = {const_cast<unsigned long long *>(&record),
                             const_cast<std::uint32_t *>(&width),
                             const_cast<std::uint32_t *>(&height)};
  if (p.driver->f().cuLaunchKernel(p.initialize, 1, 1, 1, 1, 1, 1, 0,
                                   p.owner_stream, initialize_args,
                                   nullptr) != CUDA_SUCCESS)
    return Fail(error,
                "LaunchMatteBoundsPrepared: initialization launch failed.");
  const auto source = static_cast<unsigned long long>(alpha.ptr);
  const auto pitch = static_cast<unsigned long long>(alpha.pitch);
  void *reduce_args[] = {const_cast<unsigned long long *>(&source),
                         const_cast<unsigned long long *>(&pitch),
                         const_cast<std::uint32_t *>(&width),
                         const_cast<std::uint32_t *>(&height),
                         const_cast<unsigned long long *>(&record)};
  constexpr unsigned int block = 16;
  if (p.driver->f().cuLaunchKernel(p.reduce, (width + block - 1) / block,
                                   (height + block - 1) / block, 1, block,
                                   block, 1, 0, p.owner_stream, reduce_args,
                                   nullptr) != CUDA_SUCCESS)
    return Fail(error, "LaunchMatteBoundsPrepared: reduction launch failed.");
  return true;
}

CUdeviceptr MatteBoundsDeviceRecord(const PreparedMatteBounds &p) noexcept {
  return p.live ? p.device_record : 0;
}
std::size_t
MatteBoundsDeviceRecordBytes(const PreparedMatteBounds &p) noexcept {
  return p.live && p.device_record ? sizeof(DeviceMatteBoundsRecord) : 0;
}

PreparedMatteBoundsBudget
PreparedMatteBoundsReservedBudget(const PreparedMatteBounds &p) noexcept {
  return p.live ? PreparedMatteBoundsBudget{p.device_record_bytes,
                                            p.device_record ? 1u : 0u}
                : PreparedMatteBoundsBudget{};
}

bool MatteBoundsForegroundFound(const DeviceMatteBoundsRecord &record,
                                int width, int height) noexcept {
  if (width <= 0 || height <= 0 || record.x0 < 0 || record.y0 < 0 ||
      record.x1 < record.x0 || record.y1 < record.y0 || record.x1 >= width ||
      record.y1 >= height)
    return false;
  const auto pixels =
      static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
  const auto minimum = std::max<std::uint64_t>(32, pixels / 500);
  return record.foreground_count >= minimum;
}

bool DestroyMatteBoundsPrepared(PreparedMatteBounds *p, std::string *error) {
  if (error)
    error->clear();
  if (!p || !p->live)
    return p != nullptr;
  CUcontext current = nullptr;
  if (!p->driver || !p->driver->f().cuCtxGetCurrent ||
      p->driver->f().cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
      current != p->owner_context)
    return Fail(error,
                "DestroyMatteBoundsPrepared: owner context is not current.");
  if (p->device_record &&
      p->driver->f().cuMemFree(p->device_record) != CUDA_SUCCESS)
    return Fail(
        error,
        "DestroyMatteBoundsPrepared: device free failed; retry required.");
  p->device_record = 0;
  p->device_record_bytes = 0;
  if (p->module && p->driver->f().cuModuleUnload(p->module) != CUDA_SUCCESS)
    return Fail(
        error,
        "DestroyMatteBoundsPrepared: module unload failed; retry required.");
  Clear(p);
  return true;
}
} // namespace studiocast::cuda::kernels
