#include "maxine_effect_runtime_fake_control.h"

#include "core/maxine/nvcv_types.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace {

using studiocast::maxine::CUstream;
using studiocast::maxine::NvCV_Status;
using studiocast::maxine::NVCV_SUCCESS;
using studiocast::maxine::NvCVImage;
using studiocast::maxine::NvCVImage_ComponentType;
using studiocast::maxine::NvCVImage_PixelFormat;

constexpr std::size_t kCounterCount =
    static_cast<std::size_t>(FakeMaxineCounter::count);
std::array<std::atomic<std::uint64_t>, kCounterCount> g_counters{};
std::atomic<std::uintptr_t> g_next_token{0x1000};
std::atomic<bool> g_fail_vfx_run{false};
std::atomic<bool> g_fail_ar_run{false};
std::atomic<bool> g_fail_ar_set_stream{false};
std::atomic<bool> g_fail_vfx_set_stream{false};
std::atomic<bool> g_fail_nvcv_transfer{false};
std::atomic<bool> g_fail_nvcv_composite{false};
std::array<float, 64> g_boxes{};

void Count(FakeMaxineCounter counter) {
  g_counters[static_cast<std::size_t>(counter)].fetch_add(
      1, std::memory_order_relaxed);
}

void *Token() {
  return reinterpret_cast<void *>(
      g_next_token.fetch_add(0x10, std::memory_order_relaxed));
}

std::uint8_t PixelBytes(NvCVImage_PixelFormat format,
                        NvCVImage_ComponentType type) {
  const std::uint8_t components =
      format == studiocast::maxine::NVCV_A ? 1u : 3u;
  const std::uint8_t component_bytes =
      type == studiocast::maxine::NVCV_F32 ? 4u : 1u;
  return static_cast<std::uint8_t>(components * component_bytes);
}

NvCV_Status AllocateImage(NvCVImage *image, unsigned width, unsigned height,
                          NvCVImage_PixelFormat format,
                          NvCVImage_ComponentType type, unsigned layout,
                          unsigned mem_space) {
  if (!image || width == 0 || height == 0)
    return -1;
  const std::uint8_t pixel_bytes = PixelBytes(format, type);
  const auto pitch = static_cast<std::int32_t>(width * pixel_bytes);
  const auto bytes = static_cast<std::size_t>(pitch) * height;
  void *pixels = std::malloc(bytes);
  if (!pixels)
    return -1;
  *image = {};
  image->width = width;
  image->height = height;
  image->pitch = pitch;
  image->pixelFormat = format;
  image->componentType = type;
  image->pixelBytes = pixel_bytes;
  image->componentBytes = type == studiocast::maxine::NVCV_F32 ? 4u : 1u;
  image->numComponents = format == studiocast::maxine::NVCV_A ? 1u : 3u;
  image->planar = static_cast<std::uint8_t>(layout);
  image->gpuMem = static_cast<std::uint8_t>(mem_space);
  image->pixels = pixels;
  image->bufferBytes = bytes;
  return NVCV_SUCCESS;
}

} // namespace

extern "C" void StudioCastFakeMaxineResetCounters() {
  for (auto &counter : g_counters)
    counter.store(0, std::memory_order_relaxed);
}

extern "C" std::uint64_t
StudioCastFakeMaxineCounter(std::size_t counter_index) {
  if (counter_index >= g_counters.size())
    return 0;
  return g_counters[counter_index].load(std::memory_order_relaxed);
}

extern "C" void StudioCastFakeMaxineFailNextVfxRun() {
  g_fail_vfx_run.store(true, std::memory_order_relaxed);
}

extern "C" void StudioCastFakeMaxineFailNextArRun() {
  g_fail_ar_run.store(true, std::memory_order_relaxed);
}
extern "C" void StudioCastFakeMaxineFailNextArSetStream() {
  g_fail_ar_set_stream.store(true, std::memory_order_relaxed);
}
extern "C" void StudioCastFakeMaxineFailNextVfxSetStream() {
  g_fail_vfx_set_stream.store(true, std::memory_order_relaxed);
}
extern "C" void StudioCastFakeMaxineFailNextNvcvTransfer() {
  g_fail_nvcv_transfer.store(true, std::memory_order_relaxed);
}
extern "C" void StudioCastFakeMaxineFailNextNvcvComposite() {
  g_fail_nvcv_composite.store(true, std::memory_order_relaxed);
}

extern "C" NvCV_Status NvVFX_CreateEffect(const char *, void **handle) {
  Count(FakeMaxineCounter::vfx_create);
  *handle = Token();
  return NVCV_SUCCESS;
}
extern "C" void NvVFX_DestroyEffect(void *) {
  Count(FakeMaxineCounter::vfx_destroy);
}
extern "C" NvCV_Status NvVFX_Load(void *) {
  Count(FakeMaxineCounter::vfx_load);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_Run(void *, int async) {
  Count(FakeMaxineCounter::vfx_run);
  Count(async == 0 ? FakeMaxineCounter::vfx_run_sync
                   : FakeMaxineCounter::vfx_run_async);
  return g_fail_vfx_run.exchange(false, std::memory_order_relaxed)
             ? -9
             : NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_SetImage(void *, const char *, NvCVImage *) {
  Count(FakeMaxineCounter::vfx_set_image);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_SetString(void *, const char *, const char *) {
  Count(FakeMaxineCounter::vfx_set_string);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_GetString(void *, const char *, const char **out) {
  if (out)
    *out = "fake";
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_SetF32(void *, const char *, float) {
  Count(FakeMaxineCounter::vfx_set_f32);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_SetU32(void *, const char *, std::uint32_t) {
  Count(FakeMaxineCounter::vfx_set_u32);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_SetS32(void *, const char *, std::int32_t) {
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_GetU32(void *, const char *, std::uint32_t *out) {
  if (out)
    *out = 64;
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_GetS32(void *, const char *, std::int32_t *out) {
  if (out)
    *out = 0;
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_GetF32(void *, const char *, float *out) {
  if (out)
    *out = 0.0f;
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_SetObject(void *, const char *, void *) {
  Count(FakeMaxineCounter::vfx_set_object);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_GetObject(void *, const char *, void **out,
                                       unsigned long) {
  if (out)
    *out = nullptr;
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_AllocateState(void *, void **state) {
  *state = Token();
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_DeallocateState(void *, void *) {
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_ResetState(void *, void *) { return NVCV_SUCCESS; }
extern "C" NvCV_Status NvVFX_SetStateObjectHandleArray(void *, const char *,
                                                       void **, std::uint32_t) {
  Count(FakeMaxineCounter::vfx_set_state_array);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_CudaStreamCreate(CUstream *stream) {
  Count(FakeMaxineCounter::vfx_stream_create);
  *stream = reinterpret_cast<CUstream>(Token());
  return NVCV_SUCCESS;
}
extern "C" void NvVFX_CudaStreamDestroy(CUstream) {
  Count(FakeMaxineCounter::vfx_stream_destroy);
}
extern "C" NvCV_Status NvVFX_CudaStreamSynchronize(CUstream) {
  Count(FakeMaxineCounter::vfx_stream_synchronize);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_SetCudaStream(void *, const char *, CUstream) {
  Count(FakeMaxineCounter::vfx_set_stream);
  return g_fail_vfx_set_stream.exchange(false, std::memory_order_relaxed)
             ? -7
             : NVCV_SUCCESS;
}
extern "C" NvCV_Status NvVFX_GetCudaStream(void *, const char *,
                                           CUstream *stream) {
  if (stream)
    *stream = nullptr;
  return NVCV_SUCCESS;
}

extern "C" NvCV_Status NvCVImage_Init(NvCVImage *image, unsigned width,
                                      unsigned height, int pitch, void *pixels,
                                      NvCVImage_PixelFormat format,
                                      NvCVImage_ComponentType type,
                                      unsigned layout, unsigned mem_space) {
  if (!image)
    return -1;
  *image = {};
  image->width = width;
  image->height = height;
  image->pitch = pitch;
  image->pixelFormat = format;
  image->componentType = type;
  image->pixelBytes = PixelBytes(format, type);
  image->componentBytes = type == studiocast::maxine::NVCV_F32 ? 4u : 1u;
  image->numComponents = format == studiocast::maxine::NVCV_A ? 1u : 3u;
  image->planar = static_cast<std::uint8_t>(layout);
  image->gpuMem = static_cast<std::uint8_t>(mem_space);
  image->pixels = pixels;
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status
NvCVImage_Alloc(NvCVImage *image, unsigned width, unsigned height,
                NvCVImage_PixelFormat format, NvCVImage_ComponentType type,
                unsigned layout, unsigned mem_space, unsigned) {
  Count(FakeMaxineCounter::nvcv_alloc);
  return AllocateImage(image, width, height, format, type, layout, mem_space);
}
extern "C" NvCV_Status
NvCVImage_Realloc(NvCVImage *image, unsigned width, unsigned height,
                  NvCVImage_PixelFormat format, NvCVImage_ComponentType type,
                  unsigned layout, unsigned mem_space, unsigned alignment) {
  Count(FakeMaxineCounter::nvcv_realloc);
  if (image && image->pixels)
    std::free(image->pixels);
  return NvCVImage_Alloc(image, width, height, format, type, layout, mem_space,
                         alignment);
}
extern "C" NvCV_Status NvCVImage_Dealloc(NvCVImage *image) {
  Count(FakeMaxineCounter::nvcv_dealloc);
  if (image && image->pixels)
    std::free(image->pixels);
  if (image)
    *image = {};
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvCVImage_Transfer(const NvCVImage *, NvCVImage *, float,
                                          CUstream, NvCVImage *) {
  Count(FakeMaxineCounter::nvcv_transfer);
  return g_fail_nvcv_transfer.exchange(false, std::memory_order_relaxed)
             ? -11
             : NVCV_SUCCESS;
}
extern "C" NvCV_Status NvCVImage_Composite(const NvCVImage *, const NvCVImage *,
                                           const NvCVImage *, NvCVImage *,
                                           CUstream) {
  Count(FakeMaxineCounter::nvcv_composite);
  return g_fail_nvcv_composite.exchange(false, std::memory_order_relaxed)
             ? -12
             : NVCV_SUCCESS;
}
extern "C" NvCV_Status NvCVImage_CompositeOverConstant(const NvCVImage *,
                                                       const NvCVImage *,
                                                       const void *,
                                                       NvCVImage *, CUstream) {
  return NVCV_SUCCESS;
}
extern "C" const char *NvCV_GetErrorStringFromCode(NvCV_Status) {
  return "fake error";
}

extern "C" NvCV_Status NvAR_Create(const char *, void **handle) {
  Count(FakeMaxineCounter::ar_create);
  *handle = Token();
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_Load(void *) {
  Count(FakeMaxineCounter::ar_load);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_Run(void *) {
  Count(FakeMaxineCounter::ar_run);
  return g_fail_ar_run.exchange(false, std::memory_order_relaxed)
             ? -8
             : NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_Destroy(void *) {
  Count(FakeMaxineCounter::ar_destroy);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_SetU32(void *, const char *, std::uint32_t) {
  Count(FakeMaxineCounter::ar_set_u32);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_SetS32(void *, const char *, std::int32_t) {
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_SetF32(void *, const char *, float) {
  Count(FakeMaxineCounter::ar_set_f32);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_SetF64(void *, const char *, double) {
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_SetU64(void *, const char *, std::uint64_t) {
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_SetString(void *, const char *, const char *) {
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_SetObject(void *, const char *, void *,
                                      unsigned long) {
  Count(FakeMaxineCounter::ar_set_object);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_SetF32Array(void *, const char *, float *, int) {
  Count(FakeMaxineCounter::ar_set_f32_array);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_GetU32(void *, const char *, std::uint32_t *out) {
  if (out)
    *out = 0;
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_GetS32(void *, const char *, std::int32_t *out) {
  if (out)
    *out = 0;
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_GetF32(void *, const char *, float *out) {
  if (out)
    *out = 0.0f;
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_GetF64(void *, const char *, double *out) {
  if (out)
    *out = 0.0;
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_GetU64(void *, const char *, std::uint64_t *out) {
  if (out)
    *out = 0;
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_GetString(void *, const char *, const char **out) {
  if (out)
    *out = "fake";
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_GetObject(void *, const char *, void **out,
                                      unsigned long) {
  if (out)
    *out = nullptr;
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_GetF32Array(void *, const char *, const float **out,
                                        int *count) {
  if (out)
    *out = g_boxes.data();
  if (count)
    *count = static_cast<int>(g_boxes.size());
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_CudaStreamCreate(CUstream *stream) {
  Count(FakeMaxineCounter::ar_stream_create);
  *stream = reinterpret_cast<CUstream>(Token());
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_CudaStreamDestroy(CUstream) {
  Count(FakeMaxineCounter::ar_stream_destroy);
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_SetCudaStream(void *, const char *, CUstream) {
  Count(FakeMaxineCounter::ar_set_stream);
  return g_fail_ar_set_stream.exchange(false, std::memory_order_relaxed)
             ? -6
             : NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_GetCudaStream(void *, const char *,
                                          CUstream *stream) {
  if (stream)
    *stream = nullptr;
  return NVCV_SUCCESS;
}
extern "C" NvCV_Status NvAR_GetVersion(std::uint32_t *version) {
  if (version)
    *version = 1;
  return NVCV_SUCCESS;
}
