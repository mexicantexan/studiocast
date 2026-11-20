#pragma once

#include <cstddef>
#include <cstdint>

enum class FakeMaxineCounter : std::size_t {
  vfx_create,
  vfx_destroy,
  vfx_load,
  vfx_run,
  vfx_set_image,
  vfx_set_string,
  vfx_set_f32,
  vfx_set_u32,
  vfx_set_object,
  vfx_set_state_array,
  vfx_stream_create,
  vfx_stream_destroy,
  vfx_set_stream,
  vfx_run_sync,
  vfx_run_async,
  vfx_stream_synchronize,
  nvcv_alloc,
  nvcv_realloc,
  nvcv_dealloc,
  nvcv_transfer,
  nvcv_composite,
  ar_create,
  ar_destroy,
  ar_load,
  ar_run,
  ar_set_object,
  ar_set_u32,
  ar_set_f32,
  ar_set_f32_array,
  ar_set_stream,
  ar_stream_create,
  ar_stream_destroy,
  count,
};

extern "C" void StudioCastFakeMaxineResetCounters();
extern "C" std::uint64_t StudioCastFakeMaxineCounter(std::size_t counter_index);
extern "C" void StudioCastFakeMaxineFailNextVfxRun();
extern "C" void StudioCastFakeMaxineFailNextArRun();
extern "C" void StudioCastFakeMaxineFailNextArSetStream();
extern "C" void StudioCastFakeMaxineFailNextVfxSetStream();
extern "C" void StudioCastFakeMaxineFailNextNvcvTransfer();
extern "C" void StudioCastFakeMaxineFailNextNvcvComposite();
