#pragma once

#include "core/cuda/cuda_image.h"
#include "core/hal/execution_domain.h"
#include "core/hal/prepared_program.h"
#include "core/onnx/ort_session.h"
#include "core/open_video/model_pack_registry.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace studiocast::open_video {

inline constexpr std::uint32_t kPreparedCudaFastDvdnetApiVersion = 1;

enum class PreparedCudaFastDvdnetStatus : std::uint16_t {
  invalid = 0,
  ready,
  executed,
  optional_failed_open,
  required_failed,
  malformed_contract,
  wrong_provider,
  wrong_execution_domain,
  stale_generation,
  stale_artifact,
  resource_exhausted,
  not_prepared,
  foreign_handle,
  preprocess_failed,
  inference_failed,
  postprocess_failed,
  synchronization_failed,
  completion_pending,
  stop_requested,
  device_lost,
  quarantined,
  shutdown,
};

enum class PreparedCudaFastDvdnetOperation : std::uint8_t {
  none = 0,
  temporal_preprocess,
  inference,
  postprocess,
  completion_wait,
};

struct PreparedCudaFastDvdnetTensorContract {
  const char *noisy_name = nullptr;
  const char *noise_map_name = nullptr;
  const char *denoised_name = nullptr;
  std::array<std::int64_t, 4> noisy_shape{};
  std::array<std::int64_t, 4> noise_map_shape{};
  std::array<std::int64_t, 4> denoised_shape{};
  bool float32 = false;
  bool nchw = false;
};

struct PreparedCudaFastDvdnetConfig {
  cuda::CudaDriverApi *driver = nullptr;
  cuda::CUcontext owner_context = nullptr;
  cuda::CUstream owner_stream = nullptr;
  int cuda_device_id = 0;
  const ModelPack *model_pack = nullptr;
  ValidatedModelArtifactIdentity artifact{};
  hal::ExecutionDomainId execution_domain{};
  PreparedCudaFastDvdnetTensorContract tensors{};
  std::uint64_t configuration_generation = 0;
  std::uint64_t model_generation = 0;
  std::uint32_t source_width = 0;
  std::uint32_t source_height = 0;
  hal::ResourceBudget available_budget{};
};

struct PreparedCudaFastDvdnetHandle {
  std::uint64_t id = 0;
  std::uint64_t configuration_generation = 0;
  std::uint64_t model_generation = 0;
};

struct PreparedCudaFastDvdnetTemporalPlan {
  std::uint64_t capture_sequence = 0;
  bool reset = false;
  // After preprocessing, slots are ordered t-2,t-1,t,t,t. On initial fill or
  // discontinuity all five slots receive the current frame.
  std::array<std::uint8_t, 5> source_slots{0, 1, 2, 2, 2};
};

struct PreparedCudaFastDvdnetRuntimePrepareResult {
  PreparedCudaFastDvdnetStatus status =
      PreparedCudaFastDvdnetStatus::invalid;
  std::uint64_t runtime_handle = 0;
  hal::ResourceBudget reserved_budget{};
  bool reusable_bindings = false;
  bool warmup_complete = false;
  bool strict_cuda_only = false;
  bool user_stream_bound = false;
  bool cpu_fallback_disabled = false;
  onnx::OrtPreparedCudaRunEvidence ort_evidence{};
};

struct PreparedCudaFastDvdnetRuntimeFrameResult {
  PreparedCudaFastDvdnetStatus status =
      PreparedCudaFastDvdnetStatus::invalid;
  PreparedCudaFastDvdnetOperation failed_operation =
      PreparedCudaFastDvdnetOperation::none;
  onnx::OrtPreparedRunResult ort{};
  hal::TransferCounters transfers{};
  std::uint64_t stage_elapsed_ns = 0;
  std::uint64_t frame_elapsed_ns = 0;
  bool output_valid = false;
  // Successful resident output remains stream-ordered until the outer owner's
  // final completion. Failed work may be preserved only when this is false.
  bool pending_work = false;
};

struct PreparedCudaFastDvdnetRuntimeTelemetry {
  std::uint64_t allocation_requests = 0;
  std::uint64_t model_scans = 0;
  std::uint64_t provider_probes = 0;
  std::uint64_t session_constructions = 0;
  std::uint64_t binding_constructions = 0;
  std::uint64_t internal_device_copies = 0;
};

struct PreparedCudaFastDvdnetShutdownSnapshot {
  PreparedCudaFastDvdnetStatus status =
      PreparedCudaFastDvdnetStatus::invalid;
  std::uint64_t attempts = 0;
  bool requested = false;
  bool quiescent = false;
};

class IPreparedCudaFastDvdnetRuntime {
public:
  virtual ~IPreparedCudaFastDvdnetRuntime() = default;
  virtual PreparedCudaFastDvdnetRuntimePrepareResult
  Prepare(const PreparedCudaFastDvdnetConfig &config) noexcept = 0;
  virtual PreparedCudaFastDvdnetRuntimeFrameResult Execute(
      std::uint64_t runtime_handle,
      const PreparedCudaFastDvdnetTemporalPlan &temporal,
      const cuda::CudaImage &input, cuda::CudaImage *output,
      float noise_sigma_over_255) noexcept = 0;
  virtual PreparedCudaFastDvdnetStatus
  ResetTemporalState(std::uint64_t runtime_handle) noexcept = 0;
  virtual PreparedCudaFastDvdnetShutdownSnapshot
  Shutdown(std::uint64_t runtime_handle) noexcept = 0;
  virtual PreparedCudaFastDvdnetRuntimeTelemetry Telemetry() const noexcept = 0;
};

struct PreparedCudaFastDvdnetPrepareResult {
  PreparedCudaFastDvdnetStatus status =
      PreparedCudaFastDvdnetStatus::invalid;
  hal::ReasonId reason = hal::ReasonId::capability_evidence_incomplete;
  PreparedCudaFastDvdnetHandle handle{};
  hal::ResourceBudget reserved_budget{};
};

struct PreparedCudaFastDvdnetFrameResult {
  PreparedCudaFastDvdnetStatus status =
      PreparedCudaFastDvdnetStatus::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  PreparedCudaFastDvdnetOperation failed_operation =
      PreparedCudaFastDvdnetOperation::none;
  hal::TransferCounters transfers{};
  std::uint64_t stage_elapsed_ns = 0;
  std::uint64_t frame_elapsed_ns = 0;
  bool output_valid = false;
  bool pending_work = false;
  bool valid_input_preserved = false;
  bool temporal_reset = false;
};

class PreparedCudaFastDvdnetDenoiser {
public:
  PreparedCudaFastDvdnetDenoiser();
  explicit PreparedCudaFastDvdnetDenoiser(
      IPreparedCudaFastDvdnetRuntime *runtime) noexcept;

  PreparedCudaFastDvdnetPrepareResult
  Prepare(const PreparedCudaFastDvdnetConfig &config) noexcept;
  PreparedCudaFastDvdnetFrameResult Execute(
      const PreparedCudaFastDvdnetHandle &handle,
      const ValidatedModelArtifactIdentity &current_artifact,
      const hal::ExecutionDomainId &current_domain, std::uint64_t capture_sequence,
      const cuda::CudaImage &input, cuda::CudaImage *output, int strength,
      hal::FailureAction failure_action) noexcept;
  PreparedCudaFastDvdnetStatus
  ResetTemporalState(const PreparedCudaFastDvdnetHandle &handle) noexcept;
  PreparedCudaFastDvdnetShutdownSnapshot
  Shutdown(const PreparedCudaFastDvdnetHandle &handle) noexcept;
  PreparedCudaFastDvdnetRuntimeTelemetry Telemetry() const noexcept;

  static std::uint32_t PaddedExtent(std::uint32_t extent) noexcept;
  static float NoiseSigmaOver255(int strength) noexcept;

private:
  bool HandleMatches(const PreparedCudaFastDvdnetHandle &handle) const noexcept;
  void Clear() noexcept;

  std::unique_ptr<IPreparedCudaFastDvdnetRuntime> owned_runtime_;
  IPreparedCudaFastDvdnetRuntime *runtime_ = nullptr;
  PreparedCudaFastDvdnetConfig config_{};
  PreparedCudaFastDvdnetHandle handle_{};
  std::uint64_t runtime_handle_ = 0;
  std::uint64_t last_capture_sequence_ = 0;
  bool have_sequence_ = false;
  bool prepared_ = false;
  bool quarantined_ = false;
  bool shutdown_requested_ = false;
};

} // namespace studiocast::open_video
