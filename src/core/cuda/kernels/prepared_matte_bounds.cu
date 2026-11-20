#include <stdint.h>

struct DeviceMatteBoundsRecord {
  int x0, y0, x1, y1;
  unsigned foreground_count;
};

extern "C" __global__ void
initialize_matte_bounds(DeviceMatteBoundsRecord *record, unsigned width,
                        unsigned height) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    record->x0 = static_cast<int>(width);
    record->y0 = static_cast<int>(height);
    record->x1 = -1;
    record->y1 = -1;
    record->foreground_count = 0;
  }
}

extern "C" __global__ void
reduce_matte_bounds(const float *alpha, uint64_t pitch, unsigned width,
                    unsigned height, DeviceMatteBoundsRecord *record) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  const float value = *reinterpret_cast<const float *>(
      reinterpret_cast<const unsigned char *>(alpha) +
      static_cast<uint64_t>(y) * pitch + static_cast<uint64_t>(x) * 4u);
  if (value > .35f) {
    atomicMin(&record->x0, static_cast<int>(x));
    atomicMin(&record->y0, static_cast<int>(y));
    atomicMax(&record->x1, static_cast<int>(x));
    atomicMax(&record->y1, static_cast<int>(y));
    atomicAdd(&record->foreground_count, 1u);
  }
}
