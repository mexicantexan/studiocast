#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void
mirror_horizontal_u8x3(const unsigned char *source, uint64_t source_pitch,
                       unsigned char *destination, uint64_t destination_pitch,
                       unsigned width, unsigned height) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  const uint64_t source_offset = static_cast<uint64_t>(y) * source_pitch +
                                 static_cast<uint64_t>(width - 1u - x) * 3u;
  const uint64_t destination_offset =
      static_cast<uint64_t>(y) * destination_pitch +
      static_cast<uint64_t>(x) * 3u;
  destination[destination_offset] = source[source_offset];
  destination[destination_offset + 1u] = source[source_offset + 1u];
  destination[destination_offset + 2u] = source[source_offset + 2u];
}
