#pragma once

#include "core/cuda/kernels/prepared_fastdvdnet_kernels.h"
#include "core/open_video/prepared_cuda_fastdvdnet_denoiser.h"

#include <array>
#include <memory>
#include <string>

namespace studiocast::open_video {

class IPreparedFastDvdnetOrtSession {
public:
  virtual ~IPreparedFastDvdnetOrtSession() = default;
  virtual onnx::OrtPreparedCudaRunEvidence evidence() const noexcept = 0;
  virtual bool PrepareBinding(const onnx::OrtSession::CudaBindingInput *inputs,
                              std::size_t input_count,
                              const onnx::OrtSession::CudaBindingOutput *outputs,
                              std::size_t output_count) noexcept = 0;
  virtual onnx::OrtPreparedRunResult Execute(
      const onnx::OrtSession::CudaBindingInput *inputs, std::size_t input_count,
      const onnx::OrtSession::CudaBindingOutput *outputs,
      std::size_t output_count,
      const onnx::OrtPreparedCudaRunEvidence &expected) noexcept = 0;
  virtual void RequestShutdown() noexcept = 0;
};

class IPreparedFastDvdnetOrtFactory {
public:
  virtual ~IPreparedFastDvdnetOrtFactory() = default;
  virtual std::unique_ptr<IPreparedFastDvdnetOrtSession>
  Create(const std::filesystem::path &model,
         const onnx::OrtSessionOptions &options,
         const PreparedCudaFastDvdnetTensorContract &tensors) noexcept = 0;
};

class NativePreparedCudaFastDvdnetRuntime final
    : public IPreparedCudaFastDvdnetRuntime {
public:
  explicit NativePreparedCudaFastDvdnetRuntime(
      IPreparedFastDvdnetOrtFactory *factory = nullptr) noexcept;
  ~NativePreparedCudaFastDvdnetRuntime() override;

  PreparedCudaFastDvdnetRuntimePrepareResult
  Prepare(const PreparedCudaFastDvdnetConfig &config) noexcept override;
  PreparedCudaFastDvdnetRuntimeFrameResult Execute(
      std::uint64_t runtime_handle,
      const PreparedCudaFastDvdnetTemporalPlan &temporal,
      const cuda::CudaImage &input, cuda::CudaImage *output,
      float noise_sigma_over_255) noexcept override;
  PreparedCudaFastDvdnetStatus
  ResetTemporalState(std::uint64_t runtime_handle) noexcept override;
  PreparedCudaFastDvdnetShutdownSnapshot
  Shutdown(std::uint64_t runtime_handle) noexcept override;
  PreparedCudaFastDvdnetRuntimeTelemetry Telemetry() const noexcept override;

private:
  bool CopyHistory(const cuda::CudaTensor &source,
                   const cuda::CudaTensor &destination) noexcept;
  bool QuiescePending() noexcept;
  bool Release() noexcept;
  void Quarantine() noexcept;

  IPreparedFastDvdnetOrtFactory *factory_ = nullptr;
  std::unique_ptr<IPreparedFastDvdnetOrtFactory> owned_factory_;
  std::unique_ptr<IPreparedFastDvdnetOrtSession> session_;
  PreparedCudaFastDvdnetConfig config_{};
  cuda::kernels::PreparedFastDvdnetKernels kernels_{};
  std::array<std::unique_ptr<cuda::CudaTensor>, 3> history_{};
  cuda::CudaTensor noisy_{};
  cuda::CudaTensor noise_map_{};
  cuda::CudaTensor denoised_{};
  std::array<onnx::OrtSession::CudaBindingInput, 2> inputs_{};
  std::array<onnx::OrtSession::CudaBindingOutput, 1> outputs_{};
  PreparedCudaFastDvdnetRuntimeTelemetry telemetry_{};
  onnx::OrtPreparedCudaRunEvidence active_ort_evidence_{};
  cuda::CudaContextIdentity prepared_context_identity_{};
  std::array<std::string, 3> tensor_names_{};
  std::array<std::array<std::int64_t, 4>, 3> tensor_shapes_{};
  std::uint64_t handle_ = 0;
  std::uint64_t shutdown_attempts_ = 0;
  bool prepared_ = false;
  bool shutdown_requested_ = false;
  bool work_pending_ = false;
  bool completion_recorded_ = false;
  bool reset_requested_ = true;
  cuda::CUresult last_copy_result_ = cuda::CUDA_SUCCESS;
  cuda::CUevent completion_event_ = nullptr;
};

} // namespace studiocast::open_video
