#pragma once

#include "core/cuda/kernels/prepared_eye_contact_kernels.h"
#include "core/onnx/ort_session.h"
#include "core/open_video/prepared_cuda_eye_contact.h"

#include <array>
#include <memory>

namespace studiocast::open_video {

bool ValidatePreparedEyeContactSessionMetadata(
    const onnx::OrtSessionInfo &) noexcept;

class IPreparedEyeContactOrtSession {
public:
  virtual ~IPreparedEyeContactOrtSession() = default;
  virtual const onnx::OrtSessionInfo &info() const noexcept = 0;
  virtual onnx::OrtPreparedCudaRunEvidence evidence() const noexcept = 0;
  virtual bool PrepareBinding(const onnx::OrtSession::CudaBindingInput *,
                              std::size_t,
                              const onnx::OrtSession::CudaBindingOutput *,
                              std::size_t) noexcept = 0;
  virtual onnx::OrtPreparedRunResult
  Execute(const onnx::OrtSession::CudaBindingInput *, std::size_t,
          const onnx::OrtSession::CudaBindingOutput *, std::size_t,
          const onnx::OrtPreparedCudaRunEvidence &) noexcept = 0;
  virtual void RequestShutdown() noexcept = 0;
};

class IPreparedEyeContactOrtFactory {
public:
  virtual ~IPreparedEyeContactOrtFactory() = default;
  virtual std::unique_ptr<IPreparedEyeContactOrtSession>
  Create(const std::filesystem::path &,
         const onnx::OrtSessionOptions &) noexcept = 0;
  virtual bool ValidateArtifact(const ModelPack &, const ModelFile &,
                                const ValidatedModelArtifactIdentity &) noexcept = 0;
};

class NativePreparedCudaEyeContactRuntime final
    : public IPreparedCudaEyeContactRuntime {
public:
  explicit NativePreparedCudaEyeContactRuntime(
      IPreparedEyeContactOrtFactory *factory = nullptr) noexcept;
  ~NativePreparedCudaEyeContactRuntime() override;

  PreparedCudaEyeContactStatus
  Prepare(const PreparedCudaEyeContactRuntimeConfig &) noexcept override;
  PreparedCudaEyeContactRuntimeFrameResult
  Execute(const PreparedCudaEyeContactBinding &, const cuda::CudaImage &,
          const cuda::CudaTensor &, float strength) noexcept override;
  PreparedCudaEyeContactStatus Shutdown() noexcept override;

private:
  bool Release() noexcept;
  PreparedCudaEyeContactStatus Quiesce() noexcept;

  IPreparedEyeContactOrtFactory *factory_ = nullptr;
  std::unique_ptr<IPreparedEyeContactOrtFactory> owned_factory_;
  std::array<std::unique_ptr<IPreparedEyeContactOrtSession>, 2> sessions_{};
  PreparedCudaEyeContactRuntimeConfig config_{};
  cuda::kernels::PreparedEyeContactKernels kernels_{};
  cuda::CudaTensor crop_geometry_{};
  std::array<cuda::CudaTensor, 2> eye_rgb_{};
  std::array<cuda::CudaTensor, 2> anchors_{};
  std::array<cuda::CudaTensor, 2> angles_{};
  std::array<cuda::CudaTensor, 2> model_rgb_{};
  cuda::CudaImage output_{};
  std::array<std::array<onnx::OrtSession::CudaBindingInput, 3>, 2> inputs_{};
  std::array<std::array<onnx::OrtSession::CudaBindingOutput, 1>, 2> outputs_{};
  std::array<onnx::OrtPreparedCudaRunEvidence, 2> evidence_{};
  std::array<std::array<std::int64_t, 4>, 4> shapes_{};
  std::array<std::string, 4> names_{};
  cuda::CUevent completion_event_ = nullptr;
  bool prepared_ = false;
  bool ownership_live_ = false;
  bool work_pending_ = false;
  bool shutdown_requested_ = false;
};

} // namespace studiocast::open_video
