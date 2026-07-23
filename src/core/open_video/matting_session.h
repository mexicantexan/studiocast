#pragma once

#include <memory>
#include <cstdint>
#include <string>

#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_tensor.h"
#include "core/cuda/cuda_driver_api.h"
#include "core/onnx/ort_session.h"
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
    // Compatibility remains the default for current tools/live paths. Future
    // prepared HAL execution must opt in and receives no TensorRT or CPU
    // provider/graph fallback.
    bool strict_cuda_provider = false;
  };

  struct PreparationEvidence {
    int cuda_device_id = 0;
    studiocast::cuda::CudaContextIdentity context;
    std::uintptr_t requested_stream_identity = 0;
    bool ort_user_stream_bound = false;
    bool explicit_stream_synchronization = false;
    bool inference_warmup_complete = false;
    std::uint64_t setup_warmup_inferences = 0;
    std::string active_provider;
    std::string model_id;
    std::string model_artifact_sha256;
    studiocast::onnx::OrtPreparedProviderEvidence provider;
    studiocast::open_video::ValidatedModelArtifactIdentity model_artifact;
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

  // Complete setup for an exact geometry, context, stream, model/provider, and
  // caller-owned output tensor. Session and IoBinding construction happen
  // here, never in Run(). Failed replacement setup retains the old plan.
  bool Prepare(int frame_w, int frame_h, studiocast::cuda::CUstream stream,
               studiocast::cuda::CudaTensor *output_alpha_gpu,
               std::string *error_out);

  // Compatibility entry point for the current live/probe paths. It prepares
  // lazily when needed, then delegates to RunPrepared. HAL execution must call
  // Prepare during setup and use RunPrepared in the frame loop.
  bool Run(studiocast::cuda::CUstream stream,
           const studiocast::cuda::CudaImage &input_rgb_gpu,
           studiocast::cuda::CudaTensor *output_alpha_gpu,
           std::string *error_out);

  // Execute a previously prepared exact contract.
  //
  // - input_rgb_gpu must be a GPU image in rgb_u8 or bgr_u8.
  // - output_alpha_gpu must be a pre-allocated NCHW float32 tensor with shape:
  //     N=1, C=1, H=input.height, W=input.width
  //
  // The operation is enqueued on the given stream. If the ONNX Runtime CUDA EP
  // cannot be configured to use the provided stream, the function may
  // synchronize the stream for correctness. With error_out == nullptr, the
  // successful StudioCast-owned path performs no diagnostic string work and
  // ExecutePreparedCudaIoBinding verifies that no binding/wrapper/container
  // construction occurred. Opaque allocation inside ORT providers is outside
  // this counter and must be qualified on hardware.
  bool RunPrepared(studiocast::cuda::CUstream stream,
                   const studiocast::cuda::CudaImage &input_rgb_gpu,
                   studiocast::cuda::CudaTensor *output_alpha_gpu,
                   std::string *error_out);

  const studiocast::open_video::ModelPack &pack() const;
  const Options &options() const;
  const PreparationEvidence *preparation_evidence() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::open_cuda
