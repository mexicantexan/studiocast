extern "C" __global__ void fastdvdnet_preprocess(
    const unsigned char *src, unsigned src_pitch, unsigned src_w,
    unsigned src_h, const float *oldest, const float *middle, float *current,
    float *noisy, float *noise_map, unsigned padded_w, unsigned padded_h,
    unsigned reset, float sigma) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= padded_w || y >= padded_h) return;
  const unsigned pixel = y * padded_w + x;
  const unsigned plane = padded_w * padded_h;
  const unsigned sx = x < src_w ? x : src_w - 1;
  const unsigned sy = y < src_h ? y : src_h - 1;
  const unsigned char *rgb = src + sy * src_pitch + sx * 3;
  for (unsigned channel = 0; channel < 3; ++channel) {
    const unsigned channel_pixel = channel * plane + pixel;
    const float value = static_cast<float>(rgb[channel]) / 255.0f;
    current[channel_pixel] = value;
    noisy[channel_pixel] = reset ? value : oldest[channel_pixel];
    noisy[(3 + channel) * plane + pixel] =
        reset ? value : middle[channel_pixel];
    noisy[(6 + channel) * plane + pixel] = value;
    noisy[(9 + channel) * plane + pixel] = value;
    noisy[(12 + channel) * plane + pixel] = value;
  }
  noise_map[pixel] = sigma;
}

extern "C" __global__ void fastdvdnet_postprocess(
    const float *src, unsigned padded_w, unsigned padded_h,
    unsigned char *dst, unsigned dst_pitch, unsigned width, unsigned height) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const unsigned pixel = y * padded_w + x;
  const unsigned plane = padded_w * padded_h;
  unsigned char *rgb = dst + y * dst_pitch + x * 3;
  for (unsigned channel = 0; channel < 3; ++channel) {
    const float value = fminf(1.0f, fmaxf(0.0f, src[channel * plane + pixel]));
    rgb[channel] = static_cast<unsigned char>(__float2uint_rn(value * 255.0f));
  }
}
