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

struct LetterboxGeometry {
  int scaled_w = 0;
  int scaled_h = 0;
  int pad_x = 0;
  int pad_y = 0;
  std::uint8_t pad_value = 0;

  [[nodiscard]] bool ValidFor(int dst_w, int dst_h) const noexcept {
    return scaled_w > 0 && scaled_h > 0 && pad_x >= 0 && pad_y >= 0 &&
           pad_x + scaled_w <= dst_w && pad_y + scaled_h <= dst_h;
  }
};

// Resolve/JIT the preprocess kernel on the resident execution thread during
// setup and bind the thread-local cached function to this exact driver wrapper
// and retained CUDA context generation.
bool PreparePreprocessKernel(studiocast::cuda::CudaDriverApi *cuda,
                             std::string *error_out);

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

// Strict resident-frame entry point. It performs no CUDA initialization,
// current-context query, module load, or symbol resolution. Missing, foreign,
// or stale setup evidence fails closed before launch.
bool PreprocessToTensorPrepared(
    studiocast::cuda::CudaDriverApi *cuda,
    const studiocast::cuda::CudaContextIdentity &context_identity,
    const CudaImage &src, const CudaTensor &dst,
    const ModelPreprocessSpec &spec, studiocast::cuda::CUstream stream,
    std::string *error_out);

} // namespace studiocast::cuda::kernels
