#pragma once

#include <string>

#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_tensor.h"
#include "core/cuda/cuda_driver_api.h"

namespace studiocast::cuda::kernels {

enum class ChannelOrder {
  rgb,
  bgr,
};

struct ModelPreprocessSpec {
  int dst_w = 0;
  int dst_h = 0;

  // Mean/std are per-channel in the destination channel order.
  float mean[3] = {0.0f, 0.0f, 0.0f};
  float std[3] = {1.0f, 1.0f, 1.0f};

  ChannelOrder dst_order = ChannelOrder::rgb;
};

// Convert an interleaved RGB/BGR U8 pitched device image to an NCHW float32
// tensor (N=1, C=3) with normalization:
//   v = (u8 / 255)
//   v = (v - mean[c]) / std[c]
//
// If src dimensions differ from spec.{dst_w,dst_h}, bilinear resampling is
// performed using the same sampling convention as video::ResizeRgb24Bilinear.
//
// The operation is enqueued on the provided stream and does not synchronize.
bool PreprocessToTensor(const CudaImage &src, const CudaTensor &dst,
                        const ModelPreprocessSpec &spec,
                        studiocast::cuda::CUstream stream,
                        std::string *error_out);

} // namespace studiocast::cuda::kernels
