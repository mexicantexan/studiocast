#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_tensor.h"
#include "core/cuda/cuda_driver_api.h"
#include "core/onnx/ort_session.h"
#include "core/open_video/model_pack_registry.h"

namespace studiocast::open_cuda {

enum class MattingPreparedRunStatus : std::uint16_t {
  success = 0,
  not_prepared = 1,
  invalid_input_contract = 2,
  invalid_output_contract = 3,
  foreign_evidence = 4,
  stale_preparation_generation = 5,
  wrong_device = 6,
  stale_context_generation = 7,
  stale_stream = 8,
  stale_artifact = 9,
  stale_ort_session = 10,
  stale_binding = 11,
  provider_unavailable = 12,
  provider_mismatch = 13,
  cpu_fallback_forbidden = 14,
  tensorrt_forbidden = 15,
  resource_exhausted = 16,
  preprocess_failure = 17,
  inference_failure = 18,
  provider_failure_quarantine = 19,
  cancelled_or_shutdown = 20,
  setup_resource_mutation = 21,
};

enum class MattingPreparedRunQuarantine : std::uint8_t {
  none = 0,
  session = 1,
  provider = 2,
};

struct MattingPreparedRunEvidence {
  static constexpr std::uint32_t kSchemaVersion = 2;

  std::uint32_t schema_version = kSchemaVersion;
  std::uint64_t preparation_generation = 0;
  int cuda_device_id = 0;
  bool strict_cuda_only = false;
  studiocast::cuda::CudaContextIdentity context{};
  std::uintptr_t stream_identity = 0;
  studiocast::cuda::CUdeviceptr input_tensor_ptr = 0;
  std::size_t input_tensor_active_bytes = 0;
  std::size_t input_tensor_allocation_pitch_bytes = 0;
  std::size_t input_tensor_allocation_capacity_bytes = 0;
  studiocast::cuda::CUdeviceptr output_tensor_ptr = 0;
  int frame_width = 0;
  int frame_height = 0;
  studiocast::open_video::ValidatedModelArtifactIdentity artifact{};
  studiocast::onnx::OrtPreparedCudaRunEvidence ort{};

  bool Valid() const noexcept;
};

struct MattingPreparedRunResult {
  static constexpr std::uint32_t kSchemaVersion = 1;

  std::uint32_t schema_version = kSchemaVersion;
  MattingPreparedRunStatus status = MattingPreparedRunStatus::not_prepared;
  MattingPreparedRunQuarantine quarantine =
      MattingPreparedRunQuarantine::none;
  std::uint64_t preparation_generation = 0;
  std::uint64_t attempt_sequence = 0;
  std::uint64_t completed_runs = 0;
  studiocast::onnx::OrtPreparedRunResult inference{};

  bool ok() const noexcept {
    return schema_version == kSchemaVersion &&
           status == MattingPreparedRunStatus::success;
  }
};

struct MattingPreparedInputTensorView {
  static constexpr std::uint32_t kSchemaVersion = 2;

  std::uint32_t schema_version = kSchemaVersion;
  studiocast::cuda::CUdeviceptr ptr = 0;
  std::size_t bytes = 0;
  std::size_t allocation_pitch_bytes = 0;
  std::size_t allocation_capacity_bytes = 0;
  int n = 0;
  int c = 0;
  int h = 0;
  int w = 0;
  MattingPreparedRunEvidence evidence{};

  bool Valid() const noexcept {
    studiocast::cuda::CudaTensorSize expected{};
    return studiocast::cuda::CheckedNchwF32Size(n, c, h, w, &expected,
                                                nullptr) &&
           schema_version == kSchemaVersion && ptr != 0 &&
           bytes == expected.bytes &&
           allocation_pitch_bytes >= bytes &&
           allocation_capacity_bytes == allocation_pitch_bytes && n == 1 &&
           c == 3 && h > 0 && w > 0 && evidence.Valid() &&
           ptr == evidence.input_tensor_ptr &&
           bytes == evidence.input_tensor_active_bytes &&
           allocation_pitch_bytes ==
               evidence.input_tensor_allocation_pitch_bytes &&
           allocation_capacity_bytes ==
               evidence.input_tensor_allocation_capacity_bytes;
  }
};

namespace internal {

struct MattingPreparedRunContract {
  bool buffers_ready = false;
  bool execution_prepared = false;
  bool input_valid = false;
  bool output_valid = false;
  bool binding_matches = false;
  bool strict_cuda_only = false;
  bool shutdown_requested = false;
  MattingPreparedRunEvidence active{};
  MattingPreparedRunEvidence requested{};
};

MattingPreparedRunResult ClassifyMattingPreparedRunContract(
    const MattingPreparedRunContract &contract) noexcept;

MattingPreparedRunResult ProjectOrtPreparedRunResult(
    const studiocast::onnx::OrtPreparedRunResult &result) noexcept;

MattingPreparedRunEvidence ResolveMattingReplacementEvidence(
    const MattingPreparedRunEvidence &current,
    const MattingPreparedRunEvidence &candidate,
    bool replacement_complete) noexcept;

} // namespace internal

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
    static constexpr std::uint32_t kSchemaVersion = 2;

    std::uint32_t schema_version = kSchemaVersion;
    int cuda_device_id = 0;
    studiocast::cuda::CudaContextIdentity context;
    std::uintptr_t requested_stream_identity = 0;
    bool ort_user_stream_bound = false;
    bool explicit_stream_synchronization = false;
    bool inference_warmup_complete = false;
    std::uint64_t setup_warmup_inferences = 0;
    std::size_t input_tensor_active_bytes = 0;
    std::size_t input_tensor_allocation_pitch_bytes = 0;
    std::size_t input_tensor_allocation_capacity_bytes = 0;
    std::string active_provider;
    std::string model_id;
    std::string model_artifact_sha256;
    studiocast::onnx::OrtPreparedProviderEvidence provider;
    studiocast::open_video::ValidatedModelArtifactIdentity model_artifact;
  };

  struct PreparedRunStats {
    std::uint64_t preparation_generation = 0;
    std::uint64_t setup_attempts = 0;
    std::uint64_t setup_commits = 0;
    std::uint64_t setup_failures = 0;
    std::uint64_t run_attempts = 0;
    std::uint64_t run_successes = 0;
    std::uint64_t run_failures = 0;
    std::uint64_t preprocess_attempts = 0;
    std::uint64_t inference_attempts = 0;
    std::uint64_t quarantine_events = 0;
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

  MattingPreparedRunResult RunPreparedTyped(
      studiocast::cuda::CUstream stream,
      const studiocast::cuda::CudaImage &input_rgb_gpu,
      studiocast::cuda::CudaTensor *output_alpha_gpu,
      const MattingPreparedRunEvidence &expected,
      bool cancellation_requested = false) noexcept;

  // Inference-only resident seam. The caller must first write the exact
  // prepared input tensor exposed by prepared_input_tensor_view(). This call
  // performs no preprocessing and executes only the frozen ORT IoBinding.
  MattingPreparedRunResult RunPreparedInferenceTyped(
      studiocast::cuda::CUstream stream,
      studiocast::cuda::CudaTensor *output_alpha_gpu,
      const MattingPreparedRunEvidence &expected,
      bool cancellation_requested = false) noexcept;

  const studiocast::open_video::ModelPack &pack() const;
  const Options &options() const;
  const PreparationEvidence *preparation_evidence() const noexcept;
  MattingPreparedRunEvidence prepared_run_evidence() const noexcept;
  MattingPreparedInputTensorView prepared_input_tensor_view() const noexcept;
  PreparedRunStats prepared_run_stats() const noexcept;
  void RequestShutdown() noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  MattingPreparedRunResult RunPreparedTypedImpl(
      studiocast::cuda::CUstream stream,
      const studiocast::cuda::CudaImage *input_rgb_gpu,
      studiocast::cuda::CudaTensor *output_alpha_gpu,
      const MattingPreparedRunEvidence &expected, bool run_preprocess,
      bool cancellation_requested, std::string *compatibility_error) noexcept;
};

} // namespace studiocast::open_cuda
