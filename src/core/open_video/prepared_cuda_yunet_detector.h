#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_prepared_transfer_completion.h"
#include "core/onnx/ort_session.h"
#include "core/open_video/frame_analysis_cache.h"
#include "core/open_video/model_pack_registry.h"

namespace studiocast::open_video {

class PreparedCudaYunetDetector;

enum class PreparedCudaYunetStatus : std::uint8_t {
  ready,
  success_no_faces,
  success_faces,
  invalid_setup,
  model_invalid,
  provider_mismatch,
  graph_mismatch,
  stale_binding,
  preprocess_failed,
  inference_failed,
  readback_failed,
  completion_failed,
  timeout,
  stop_requested,
  cancelled,
  device_lost,
  shutdown
};

struct PreparedCudaYunetBinding {
  cuda::CudaDriverApi *driver = nullptr;
  cuda::CudaContextIdentity context{};
  cuda::CUstream stream = nullptr;
  std::uint64_t stream_identity = 0;
  std::uint64_t owner_identity = 0;
  std::uint64_t configuration_generation = 0;
};

// Immutable proof that a detector and cache were prepared together. Consumers
// may copy this token, but the detector and cache must outlive them.
struct PreparedCudaYunetPreparedBinding {
  PreparedCudaYunetDetector *producer = nullptr;
  PreparedCudaYunetBinding binding{};
  ValidatedModelArtifactIdentity artifact{};
  std::uint64_t producer_revision = 0;
  FrameAnalysisCache *cache = nullptr;
  int frame_width = 0;
  int frame_height = 0;
};

struct PreparedCudaYunetSetup {
  PreparedCudaYunetBinding binding{};
  std::filesystem::path models_root;
  std::string requested_model_id;
  int cuda_device_id = 0;
  int frame_width = 0;
  int frame_height = 0;
  std::size_t resource_budget_bytes = 0;
  // Exact input tensor declaration from the already-selected model pack.
  // Production Prepare overwrites this from the validated manifest; the test
  // seam supplies it directly with its synthetic artifact identity.
  ModelTensorSpec declared_input{};
};

struct PreparedCudaYunetExecutionResult {
  PreparedCudaYunetStatus status = PreparedCudaYunetStatus::stale_binding;
  std::uint32_t enqueued_readbacks = 0;
  std::uint32_t completed_readbacks = 0;
  std::uint32_t completion_waits = 0;
  std::uint32_t cpu_tail_boundaries = 0;
  bool pending_work = false;
  bool device_lost = false;
};

namespace internal {
class IPreparedCudaYunetSession {
public:
  virtual ~IPreparedCudaYunetSession() = default;
  virtual const onnx::OrtSessionInfo &info() const noexcept = 0;
  virtual bool Prepare(const onnx::OrtSession::CudaBindingInput *, std::size_t,
                       const onnx::OrtSession::CudaBindingOutput *, std::size_t,
                       std::string *) = 0;
  virtual onnx::OrtPreparedCudaRunEvidence evidence() const noexcept = 0;
  virtual onnx::OrtPreparedRunResult
  Execute(const onnx::OrtSession::CudaBindingInput *, std::size_t,
          const onnx::OrtSession::CudaBindingOutput *, std::size_t,
          const onnx::OrtPreparedCudaRunEvidence &) noexcept = 0;
  virtual void Invalidate() noexcept = 0;
};
} // namespace internal

class PreparedCudaYunetDetector {
public:
  static constexpr std::size_t
  RequiredResourceBytes(int width = 640, int height = 640) noexcept {
    return (3u * static_cast<std::size_t>(width) *
                static_cast<std::size_t>(height) +
            32u * (static_cast<std::size_t>(width / 8) *
                       static_cast<std::size_t>(height / 8) +
                   static_cast<std::size_t>(width / 16) *
                       static_cast<std::size_t>(height / 16) +
                   static_cast<std::size_t>(width / 32) *
                       static_cast<std::size_t>(height / 32))) *
           sizeof(float);
  }
  PreparedCudaYunetDetector();
  ~PreparedCudaYunetDetector();
  PreparedCudaYunetDetector(const PreparedCudaYunetDetector &) = delete;
  PreparedCudaYunetDetector &
  operator=(const PreparedCudaYunetDetector &) = delete;
  PreparedCudaYunetDetector(PreparedCudaYunetDetector &&) noexcept;
  PreparedCudaYunetDetector &operator=(PreparedCudaYunetDetector &&) = delete;

  bool Prepare(const PreparedCudaYunetSetup &setup, std::string *error);
  bool PrepareForTesting(
      const PreparedCudaYunetSetup &setup,
      const ValidatedModelArtifactIdentity &validated_artifact,
      std::unique_ptr<internal::IPreparedCudaYunetSession> session,
      std::string *error);
  bool PrepareCache(FrameAnalysisCache *cache, std::string *error);
  bool ExportPreparedBinding(FrameAnalysisCache *cache,
                             PreparedCudaYunetPreparedBinding *out) noexcept;
  bool MatchesPreparedBinding(
      const PreparedCudaYunetPreparedBinding &binding) const noexcept;
  PreparedCudaYunetExecutionResult Execute(
      const PreparedCudaYunetBinding &binding, const cuda::CudaImage &source,
      std::uint64_t capture_sequence, FrameAnalysisCache *cache,
      const cuda::CudaPreparedTransferCompletion::WaitPolicy &wait) noexcept;
  bool Shutdown(std::string *error) noexcept;
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] const ValidatedModelArtifactIdentity &artifact() const noexcept;

private:
  struct Impl;
  bool
  PrepareValidated(const PreparedCudaYunetSetup &setup,
                   const ValidatedModelArtifactIdentity &validated_artifact,
                   std::unique_ptr<internal::IPreparedCudaYunetSession> session,
                   std::string *error);
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::open_video
