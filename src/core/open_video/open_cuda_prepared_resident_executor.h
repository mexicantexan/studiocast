#pragma once

#include "core/cuda/cuda_driver_api.h"
#include "core/cuda/cuda_prepared_transfer_completion.h"
#include "core/hal/prepared_program.h"
#include "core/open_video/model_pack_registry.h"
#include "core/open_video/open_cuda_hal_adapter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace studiocast::open_cuda {

// Matches BuildBroadcastEffectsPlan; all background modes occupy one slot.
inline constexpr unsigned
PreparedOpenCudaCanonicalStageRank(hal::SemanticEffectId semantic) noexcept {
  using S = hal::SemanticEffectId;
  switch (semantic) {
  case S::video_noise_removal:
    return 1;
  case S::eye_contact:
    return 2;
  case S::virtual_background_blur:
  case S::virtual_background_remove:
  case S::virtual_background_replace:
    return 3;
  case S::virtual_key_light:
    return 4;
  case S::auto_frame:
    return 5;
  case S::vignette:
    return 6;
  case S::mirror:
    return 7;
  default:
    return 0;
  }
}

inline constexpr std::uint32_t kPreparedOpenCudaRuntimeApiVersion = 2;

// Stable executor/runtime outcomes. These are deliberately independent of
// CUDA and ONNX Runtime error values so frame-time code never parses strings.
enum class PreparedOpenCudaRuntimeStatus : std::uint16_t {
  invalid = 0,
  ready = 1,
  executed = 2,
  optional_failed_open = 3,
  required_failed = 4,
  malformed_contract = 5,
  unsupported_effect = 6,
  wrong_provider = 7,
  wrong_execution_domain = 8,
  stale_generation = 9,
  stale_artifact = 10,
  resource_exhausted = 11,
  not_prepared = 12,
  foreign_handle = 13,
  host_binding_invalid = 14,
  preprocess_failed = 15,
  inference_failed = 16,
  postprocess_failed = 17,
  transfer_failed = 18,
  synchronization_failed = 19,
  timeout = 20,
  stop_requested = 21,
  cancelled = 22,
  device_lost = 23,
  quarantined = 24,
  shutdown = 25,
  completion_pending = 26,
};

enum class PreparedOpenCudaOperationId : std::uint16_t {
  invalid = 0,
  upload = 1,
  preprocess = 2,
  inference = 3,
  alpha_resize = 4,
  alpha_feather_horizontal = 5,
  alpha_feather_vertical = 6,
  background_blur_horizontal = 7,
  background_blur_vertical = 8,
  composite_background = 9,
  composite_solid = 10,
  final_readback = 11,
  completion_wait = 12,
  composite_replace = 13,
  key_light = 14,
  final_resize = 15,
  vignette = 16,
};

struct PreparedOpenCudaLaunchGeometry {
  PreparedOpenCudaOperationId operation = PreparedOpenCudaOperationId::invalid;
  std::uint32_t grid_x = 0;
  std::uint32_t grid_y = 0;
  std::uint32_t block_x = 0;
  std::uint32_t block_y = 0;
};

struct PreparedOpenCudaVbPlanInput {
  std::uint32_t island_stage_count = 1;
  hal::SemanticEffectId semantic = hal::SemanticEffectId::invalid;
  std::uint64_t frame_width = 0;
  std::uint64_t frame_height = 0;
  std::uint64_t model_width = 0;
  std::uint64_t model_height = 0;
  int requested_strength = 8;
  std::uint64_t input_pitch_bytes = 0;
  std::uint64_t output_pitch_bytes = 0;
  std::uint64_t final_output_width = 0;
  std::uint64_t final_output_height = 0;
  std::uint64_t final_output_pitch_bytes = 0;
  std::uint64_t model_alpha_capacity_bytes = 0;
  std::uint64_t input_tensor_capacity_bytes = 0;
  std::uint64_t resized_alpha_pitch_bytes = 0;
  std::uint64_t alpha_scratch_pitch_bytes = 0;
  std::uint64_t alpha_feather_pitch_bytes = 0;
  std::uint64_t blur_scratch_pitch_bytes = 0;
  std::uint64_t blurred_pitch_bytes = 0;
  std::uint64_t replacement_source_pitch_bytes = 0;
  std::uint64_t replacement_source_height = 0;
  std::uint64_t replacement_frame_pitch_bytes = 0;
  std::uint64_t upload_staging_bytes = 0;
  std::uint64_t readback_staging_bytes = 0;
  hal::ResourceBudget available_budget{};
};

struct PreparedOpenCudaVbPlan {
  static constexpr std::size_t kMaxLaunches = 6;

  std::uint32_t api_version = kPreparedOpenCudaRuntimeApiVersion;
  PreparedOpenCudaRuntimeStatus status = PreparedOpenCudaRuntimeStatus::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::SemanticEffectId semantic = hal::SemanticEffectId::invalid;
  std::uint32_t frame_width = 0;
  std::uint32_t island_stage_count = 1;
  std::uint32_t frame_height = 0;
  std::uint32_t final_output_width = 0;
  std::uint32_t final_output_height = 0;
  std::uint32_t model_width = 0;
  std::uint32_t model_height = 0;
  std::uint32_t strength = 0;
  std::uint32_t alpha_feather_radius = 0;
  hal::ResourceBudget required_budget{};
  std::array<PreparedOpenCudaLaunchGeometry, kMaxLaunches> launches{};
  std::uint32_t launch_count = 0;

  [[nodiscard]] bool ready() const noexcept {
    return status == PreparedOpenCudaRuntimeStatus::ready &&
           reason == hal::ReasonId::none && launch_count > 0 &&
           launch_count <= kMaxLaunches;
  }
};

// Pure setup-time planning seam used by the production runtime and hermetic
// conformance tests. All pitches/capacities are the exact prepared allocation
// values, not logical active-byte estimates.
PreparedOpenCudaVbPlan PlanPreparedOpenCudaVirtualBackground(
    const PreparedOpenCudaVbPlanInput &input) noexcept;

// CudaTensor stores NCHW data in one pitched allocation row. This converts
// that exact allocation metadata into the capacity consumed by setup budgets.
bool PreparedContiguousTensorCapacityBytes(
    std::uint64_t allocation_pitch_bytes, std::uint64_t active_bytes,
    std::uint64_t *capacity_bytes) noexcept;

struct PreparedOpenCudaVignetteBudgetInput {
  std::uint64_t input_pitch_bytes = 0;
  std::uint64_t input_height = 0;
  std::uint64_t output_pitch_bytes = 0;
  std::uint64_t output_height = 0;
  std::uint64_t final_output_pitch_bytes = 0;
  std::uint64_t final_output_height = 0;
  std::uint64_t upload_staging_bytes = 0;
  std::uint64_t readback_staging_bytes = 0;
  hal::ResourceBudget available_budget{};
};

PreparedOpenCudaRuntimeStatus PlanPreparedOpenCudaVignetteBudget(
    const PreparedOpenCudaVignetteBudgetInput &input,
    hal::ResourceBudget *required_budget, hal::ReasonId *reason) noexcept;

struct PreparedOpenCudaProductionConfig;
hal::ReasonId ValidatePreparedOpenCudaProductionProgram(
    const hal::PreparedProgram &program,
    const PreparedOpenCudaProductionConfig &config) noexcept;

struct PreparedOpenCudaRuntimeHandle {
  std::uint64_t handle_id = 0;
  std::uint64_t program_id = 0;
  hal::ProgramInvalidationKey invalidation_key{};
  hal::ExecutionDomainId execution_domain{};
  std::array<std::uint64_t, 2> opaque{};
};

// Setup-selected model dependencies. Each artifact stays bound to its exact
// pack; no frame-time discovery or substitution is permitted.
struct PreparedOpenCudaModelDependencies {
  open_video::ModelPack face{}, landmarks{}, eye{}, denoise{};
  open_video::ValidatedModelArtifactIdentity face_artifact{},
      landmarks_artifact{}, eye_left_artifact{}, eye_right_artifact{},
      denoise_artifact{};
};

struct PreparedOpenCudaProductionConfig {
  cuda::CudaDriverApi *driver = nullptr;
  cuda::CUstream prepared_stream = nullptr;
  const open_video::ModelPack *model_pack = nullptr;
  open_video::ValidatedModelArtifactIdentity model_artifact{};
  int cuda_device_id = 0;
  std::uint64_t adapter_instance_id = 0;
  std::uint64_t stream_identity = 0;
  std::uint64_t configuration_generation = 0;
  std::uint64_t model_preprocess_generation = 0;
  std::uint64_t model_artifact_hash = 0;
  bool executor_owns_stream = false;
  int strength = 8;
  std::array<std::uint8_t, 3> remove_background_rgb{0, 0, 0};
  std::vector<std::uint8_t> replacement_rgb{};
  int replacement_width = 0;
  int replacement_height = 0;
  int key_light_intensity = 50;
  int key_light_temperature_preset = 0;
  int key_light_direction_pan_degrees = 0;
  int vignette_intensity = 35;
  bool vignette_tracked_center = false;
  PreparedOpenCudaModelDependencies dependencies{};
  int denoise_strength = 50;
  int auto_frame_strength = 50;
  int auto_frame_smoothing = 50;
  float auto_frame_headroom = .15F;
  int eye_contact_strength = 50;
  bool eye_contact_look_away = false;
  cuda::CudaPreparedTransferCompletion::WaitPolicy wait_policy{};
};

struct PreparedOpenCudaRuntimeEvidence {
  std::uint32_t api_version = kPreparedOpenCudaRuntimeApiVersion;
  PreparedOpenCudaRuntimeStatus status = PreparedOpenCudaRuntimeStatus::invalid;
  hal::SemanticEffectId semantic = hal::SemanticEffectId::invalid;
  hal::SemanticEffectProviderId provider =
      hal::SemanticEffectProviderId::invalid;
  hal::ExecutionDomainId execution_domain{};
  open_video::ValidatedModelArtifactIdentity model_artifact{};
  hal::ResourceBudget reserved_budget{};
  PreparedOpenCudaVbPlan geometry_plan{};
  bool strict_cuda_only = false;
  bool user_stream_bound = false;
  bool cpu_provider_fallback_disabled = false;
  bool reusable_resources_prepared = false;
  bool setup_warmup_complete = false;
  bool inference_only_frame_path = false;
  bool device_preprocess = false;
  bool device_postprocess = false;
  bool one_upload_one_readback = false;
  bool no_intermediate_crossings = false;
  bool single_in_flight = false;
};

struct PreparedOpenCudaRuntimePrepareResult {
  PreparedOpenCudaRuntimeStatus status = PreparedOpenCudaRuntimeStatus::invalid;
  hal::ReasonId reason = hal::ReasonId::capability_evidence_incomplete;
  PreparedOpenCudaRuntimeHandle handle{};
  PreparedOpenCudaRuntimeEvidence evidence{};
  OpenCudaSetupComponentId failed_component = OpenCudaSetupComponentId::invalid;
};

struct PreparedOpenCudaRuntimeFrame {
  std::uint32_t api_version = kPreparedOpenCudaRuntimeApiVersion;
  std::uint64_t capture_sequence = 0;
  const std::byte *input = nullptr;
  std::byte *output = nullptr;
  std::size_t input_span_bytes = 0;
  std::size_t output_span_bytes = 0;
  std::size_t input_row_stride_bytes = 0;
  std::size_t output_row_stride_bytes = 0;
  bool stage_disabled = false;
  std::uint32_t disabled_stage_mask = 0;
  bool stage_required = false;
  hal::FailureAction failure_action = hal::FailureAction::fail_closed;
};

struct PreparedOpenCudaRuntimeFrameResult {
  std::uint32_t api_version = kPreparedOpenCudaRuntimeApiVersion;
  PreparedOpenCudaRuntimeStatus status = PreparedOpenCudaRuntimeStatus::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  // Policy outcome remains optional_failed_open/required_failed while this
  // fixed status preserves the exact resident operation that failed.
  PreparedOpenCudaRuntimeStatus stage_failure_status =
      PreparedOpenCudaRuntimeStatus::invalid;
  hal::ReasonId stage_failure_reason = hal::ReasonId::none;
  PreparedOpenCudaOperationId failed_operation =
      PreparedOpenCudaOperationId::invalid;
  hal::TransferCounters transfers{};
  std::uint64_t stage_elapsed_ns = 0;
  std::uint64_t frame_elapsed_ns = 0;
  bool output_valid = false;
  bool valid_input_preserved = false;
  bool analysis_ran = false;
  bool inference_only_path = false;
  std::array<hal::StageExecutionResult, hal::kMaxPreparedStages> stages{};
  std::size_t stage_count = 0;
};

struct PreparedOpenCudaRuntimeTelemetry {
  std::uint64_t allocation_requests = 0;
  std::uint64_t provider_probes = 0;
  std::uint64_t model_scans = 0;
  std::uint64_t manifest_parses = 0;
  std::uint64_t session_constructions = 0;
  std::uint64_t json_serializations = 0;
};

struct PreparedOpenCudaShutdownSnapshot {
  std::uint32_t api_version = kPreparedOpenCudaRuntimeApiVersion;
  PreparedOpenCudaRuntimeStatus status = PreparedOpenCudaRuntimeStatus::invalid;
  std::uint64_t shutdown_attempts = 0;
  std::uint64_t stream_destructions = 0;
  std::uint32_t pending_prepared_slots = 0;
  bool shutdown_requested = false;
  bool quiescent = false;
  bool stream_owned = false;
  bool stream_destroyed = false;
  // A concrete owner must retain the executor, CudaDriverApi, retained
  // primary context, and stream while this is false.
  bool driver_context_release_permitted = false;
};

// Fixed, injectable production seam. Prepare/Destroy may allocate and inspect
// already-selected model metadata. ExecuteFrame must be allocation-, probe-,
// session-, model-scan-, manifest-, and JSON-free.
class IPreparedOpenCudaRuntime {
public:
  virtual ~IPreparedOpenCudaRuntime() = default;

  virtual PreparedOpenCudaRuntimePrepareResult
  Prepare(const hal::PreparedProgram &program,
          const PreparedOpenCudaProductionConfig &config) noexcept = 0;
  virtual PreparedOpenCudaRuntimeFrameResult
  ExecuteFrame(const PreparedOpenCudaRuntimeHandle &handle,
               const PreparedOpenCudaRuntimeFrame &frame) noexcept = 0;
  virtual PreparedOpenCudaRuntimeStatus
  ResetTemporalState(const PreparedOpenCudaRuntimeHandle &handle,
                     hal::ReasonId reason) noexcept = 0;
  // A pending return means asynchronous DMA may still reference prepared
  // resources. The runtime retains them and retries on a later Shutdown().
  virtual PreparedOpenCudaRuntimeStatus
  DestroyPrepared(PreparedOpenCudaRuntimeHandle *handle) noexcept = 0;
  virtual PreparedOpenCudaShutdownSnapshot RequestShutdown() noexcept = 0;
  [[nodiscard]] virtual PreparedOpenCudaShutdownSnapshot
  ShutdownSnapshot() const noexcept = 0;
  virtual void Shutdown() noexcept { (void)RequestShutdown(); }
  [[nodiscard]] virtual PreparedOpenCudaRuntimeTelemetry
  Telemetry() const noexcept = 0;
};

class OpenCudaPreparedResidentExecutor final
    : public IPreparedResidentExecutor {
public:
  struct Options {
    cuda::CudaDriverApi *driver = nullptr;
    // The adapter owns this already-created stream for the executor lifetime.
    // stream_identity must be its exact process-local uintptr_t value.
    cuda::CUstream prepared_stream = nullptr;
    open_video::ModelPack model_pack{};
    open_video::ValidatedModelArtifactIdentity model_artifact{};
    int cuda_device_id = 0;
    std::uint64_t adapter_instance_id = 0;
    std::uint64_t stream_identity = 0;
    std::uint64_t configuration_generation = 0;
    std::uint64_t model_preprocess_generation = 0;
    std::uint64_t model_artifact_hash = 0;
    bool executor_owns_stream = false;
    int strength = 8;
    std::array<std::uint8_t, 3> remove_background_rgb{0, 0, 0};
    std::vector<std::uint8_t> replacement_rgb{};
    int replacement_width = 0;
    int replacement_height = 0;
    int key_light_intensity = 50;
    int key_light_temperature_preset = 0;
    int key_light_direction_pan_degrees = 0;
    int vignette_intensity = 35;
    bool vignette_tracked_center = false;
    PreparedOpenCudaModelDependencies dependencies{};
    int denoise_strength = 50;
    int auto_frame_strength = 50;
    int auto_frame_smoothing = 50;
    float auto_frame_headroom = .15F;
    int eye_contact_strength = 50;
    bool eye_contact_look_away = false;
    cuda::CudaPreparedTransferCompletion::WaitPolicy wait_policy{};
    // Null selects the production CUDA/ORT implementation.
    IPreparedOpenCudaRuntime *runtime = nullptr;
  };

  explicit OpenCudaPreparedResidentExecutor(Options options);
  ~OpenCudaPreparedResidentExecutor() override;

  OpenCudaPreparedResidentExecutor(const OpenCudaPreparedResidentExecutor &) =
      delete;
  OpenCudaPreparedResidentExecutor &
  operator=(const OpenCudaPreparedResidentExecutor &) = delete;

  ExecutorPrepareResult
  Prepare(const hal::PreparedProgram &program) noexcept override;
  [[nodiscard]] ExecutorPrepareResult LastPrepareResult() const noexcept;
  hal::ExecuteResult
  Execute(const PreparedExecutorHandle &program,
          const hal::ExecuteRequest &request) noexcept override;
  hal::StatusId BindHostFrame(const PreparedExecutorHandle &program,
                              const hal::HostFrameBinding &binding,
                              hal::HostFrameTokens *tokens,
                              hal::ReasonId *reason) noexcept override;
  hal::StatusId DiscardUnsubmittedHostFrame(
      const PreparedExecutorHandle &program, const hal::HostFrameTokens &tokens,
      std::uint64_t capture_sequence, hal::ReasonId *reason) noexcept override;
  hal::StatusId
  ResolveHostFrameOutput(const PreparedExecutorHandle &program,
                         const hal::CompletionToken &completion,
                         const hal::ResourceHandle &output,
                         hal::HostFrameOutputResolution *resolution,
                         hal::ReasonId *reason) noexcept override;
  hal::StatusId ResetTemporalState(const PreparedExecutorHandle &program,
                                   hal::ReasonId reason) noexcept override;
  void DestroyPrepared(PreparedExecutorHandle *program) noexcept override;
  // The shared HAL keeps its compatible void Shutdown(). Owners of the
  // CUDA driver/context island must use this typed method and retain the
  // complete island while driver_context_release_permitted is false.
  [[nodiscard]] PreparedOpenCudaShutdownSnapshot RequestShutdown() noexcept;
  [[nodiscard]] PreparedOpenCudaShutdownSnapshot
  ShutdownSnapshot() const noexcept;
  void Shutdown() noexcept override;
  [[nodiscard]] ExecutorTelemetrySnapshot Telemetry() const noexcept override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::open_cuda
