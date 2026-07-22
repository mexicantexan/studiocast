#pragma once

#include <memory>
#include <string>

#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_tensor.h"
#include "core/cuda/cuda_driver_api.h"
#include "core/open_video/model_pack_registry.h"

namespace studiocast::open_cuda {

// Long-lived matting inference session for the Open CUDA backend.
//
// - Uses ONNX Runtime with the CUDA execution provider.
// - Uses Ort::IoBinding to bind input/output tensors to GPU buffers.
// - Avoids per-frame device allocations (tensors are re-used).
class OpenCudaMattingSession {
public:
  struct Options {
    int device_id = 0;
    bool enable_tensorrt = false;
  };

  OpenCudaMattingSession(studiocast::cuda::CudaDriverApi *cuda,
                         studiocast::open_video::ModelPack pack);
  OpenCudaMattingSession(studiocast::cuda::CudaDriverApi *cuda,
                         studiocast::open_video::ModelPack pack, Options opts);
  ~OpenCudaMattingSession();

  OpenCudaMattingSession(const OpenCudaMattingSession &) = delete;
  OpenCudaMattingSession &operator=(const OpenCudaMattingSession &) = delete;

  // Prepare internal buffers and validate model metadata.
  // Safe to call repeatedly.
  bool EnsureInitialized(int frame_w, int frame_h, std::string *error_out);

  // Run matting inference.
  //
  // - input_rgb_gpu must be a GPU image in rgb_u8 or bgr_u8.
  // - output_alpha_gpu must be a pre-allocated NCHW float32 tensor with shape:
  //     N=1, C=1, H=input.height, W=input.width
  //
  // The operation is enqueued on the given stream. If the ONNX Runtime CUDA EP
  // cannot be configured to use the provided stream, the function may
  // synchronize the stream for correctness.
  bool Run(studiocast::cuda::CUstream stream,
           const studiocast::cuda::CudaImage &input_rgb_gpu,
           studiocast::cuda::CudaTensor *output_alpha_gpu,
           std::string *error_out);

  const studiocast::open_video::ModelPack &pack() const;
  const Options &options() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::open_cuda
