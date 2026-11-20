#pragma once

#include "core/hal/static_registry.h"
#include "core/open_video/model_pack_registry.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace studiocast::video {

enum class PreparedOpenCudaEffectsEngineIntent : std::uint8_t {
  automatic = 0,
  maxine,
  open_cuda,
};

enum class PreparedOpenCudaComputeIntent : std::uint8_t {
  automatic = 0,
  cpu,
  cuda,
  vulkan,
};

enum class PreparedOpenCudaHardwareRoute : std::uint8_t {
  legacy_compatibility = 0,
  hal_live,
};

enum class PreparedOpenCudaHardwareEligibility : std::uint8_t {
  eligible = 0,
  automatic_engine_compatibility,
  not_explicit_open_cuda,
  compute_family_disallowed,
  not_strict_cuda,
  provider_mismatch,
  tensorrt_unsupported,
  stage_count_invalid,
  semantic_unsupported,
  mixed_resident_section,
  cross_domain_section,
  unsupported_effect,
  shared_analysis_unsupported,
  temporal_stage_unsupported,
  unsupported_host_layout,
  geometry_mismatch,
  scaling_unsupported,
  output_conversion_unsupported,
  invalid_generation,
  model_artifact_evidence_incomplete,
  model_pack_incompatible,
  resource_budget_invalid,
  wait_policy_invalid,
  writer_tail_invalid,
  runtime_unavailable,
};

using PreparedOpenCudaWaitPredicate = bool (*)(void *context) noexcept;
using PreparedOpenCudaNowNanoseconds =
    std::uint64_t (*)(void *context) noexcept;
using PreparedOpenCudaPollWait =
    void (*)(void *context, std::uint64_t wait_nanoseconds) noexcept;

// Fixed, allocation-free completion policy copied into prepared CUDA
// resources. Production callers bind stop/cancel state through callback_context
// without exposing a CUDA type through the video facade. Every non-null
// callback and callback_context is borrowed and must remain valid until
// Shutdown reports driver_context_release_permitted.
struct PreparedOpenCudaWaitPolicy {
  std::uint64_t timeout_nanoseconds = 0;
  std::uint64_t poll_interval_nanoseconds = 0;
  std::uint32_t maximum_poll_count = 0;
  PreparedOpenCudaWaitPredicate stop_requested = nullptr;
  PreparedOpenCudaWaitPredicate cancel_requested = nullptr;
  PreparedOpenCudaNowNanoseconds now_nanoseconds = nullptr;
  PreparedOpenCudaPollWait poll_wait = nullptr;
  void *callback_context = nullptr;
};

// Setup-only request for the first strict Open Video/ONNX Runtime CUDA live
// route. Model discovery, manifest parsing, artifact validation, provider
// selection, and budget sizing have already completed before this is built.
struct PreparedOpenCudaHardwareRequest {
  PreparedOpenCudaEffectsEngineIntent effects_engine =
      PreparedOpenCudaEffectsEngineIntent::automatic;
  PreparedOpenCudaComputeIntent compute =
      PreparedOpenCudaComputeIntent::automatic;
  hal::BackendId requested_backend = hal::BackendId::invalid;
  bool strict_backend_selection = false;
  hal::SemanticEffectProviderId requested_provider =
      hal::SemanticEffectProviderId::invalid;
  bool strict_provider_selection = false;
  bool tensorrt_enabled = false;

  std::uint32_t stage_id = 0;
  hal::SemanticEffectId semantic = hal::SemanticEffectId::invalid;
  std::size_t semantic_stage_count = 0;
  bool required = false;
  bool single_resident_island = false;
  bool contains_other_execution_domain = false;
  bool contains_other_resident_stage = false;
  bool request_virtual_background_replace = false;
  bool request_vignette = false;
  bool request_video_noise_removal = false;
  bool request_eye_contact = false;
  bool request_virtual_key_light = false;
  bool request_auto_frame = false;
  bool request_mirror = false;
  bool shared_matte_required = false;
  bool temporal_history_required = false;

  std::uint32_t capture_width = 0;
  std::uint32_t capture_height = 0;
  std::uint32_t output_width = 0;
  std::uint32_t output_height = 0;
  bool host_input_is_tight_rgb24 = false;
  bool host_output_is_tight_rgb24 = false;
  bool scaling_required = false;
  bool output_format_conversion_required = false;

  hal::ProgramGenerations generations{};
  hal::ResourceBudget capacity{};
  open_video::ModelPack model_pack{};
  open_video::ValidatedModelArtifactIdentity model_artifact{};
  int cuda_device_id = 0;
  int strength = 8;
  std::array<std::uint8_t, 3> remove_background_rgb{0, 0, 0};
  PreparedOpenCudaWaitPolicy wait{};
  bool declares_writer_copy_cpu_tail = true;
};

struct PreparedOpenCudaProductionLifecycleSnapshot {
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  std::uint64_t shutdown_attempts = 0;
  bool bootstrap_domain_ready = false;
  bool prepared_capability_committed = false;
  bool quarantined = false;
  bool device_lost = false;
  bool shutdown_requested = false;
  bool shutdown_pending = false;
  bool quiescent = false;
  bool driver_context_release_permitted = false;
  bool stopped = false;
};

struct PreparedOpenCudaHardwareSetupResult {
  PreparedOpenCudaHardwareRoute route =
      PreparedOpenCudaHardwareRoute::legacy_compatibility;
  PreparedOpenCudaHardwareEligibility eligibility =
      PreparedOpenCudaHardwareEligibility::runtime_unavailable;
  hal::StatusId status = hal::StatusId::unavailable;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::ExecutionDomainId execution_domain{};
  bool retained_previous_live_program = false;
};

struct PreparedOpenCudaHardwareFrameResult {
  hal::StatusId status = hal::StatusId::failed;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::CompletionToken completion{};
  hal::TransferCounters transfers{};
  hal::StageExecutionResult stage{};
  const std::uint8_t *output = nullptr;
  std::size_t output_size_bytes = 0;
  std::size_t output_row_stride_bytes = 0;
  bool output_ready = false;
  bool completion_pending = false;
};

struct PreparedOpenCudaHardwareExecutionSnapshot {
  PreparedOpenCudaHardwareRoute route =
      PreparedOpenCudaHardwareRoute::legacy_compatibility;
  PreparedOpenCudaHardwareEligibility eligibility =
      PreparedOpenCudaHardwareEligibility::runtime_unavailable;
  hal::PreparedProgram active_program{};
  hal::ReadinessSnapshot readiness{};
  hal::ExecutionTelemetry telemetry{};
  hal::CommittedFrameSnapshot latest_committed_frame{};
  hal::ExecutionDomainId execution_domain{};
  PreparedOpenCudaProductionLifecycleSnapshot production{};
  bool prepared = false;
  bool quarantined = false;
  bool stopped = false;
  bool shutdown_pending = false;
};

[[nodiscard]] PreparedOpenCudaHardwareEligibility
EvaluatePreparedOpenCudaHardwareEligibility(
    const PreparedOpenCudaHardwareRequest &request) noexcept;

// Backend-hidden video facade. The default constructor owns the complete
// production CUDA/ORT resident island behind its PIMPL. The injected
// constructor is a hermetic seam that accepts neutral, already-cached HAL
// evidence only.
class PreparedOpenCudaHardwareExecutionOwner {
public:
  PreparedOpenCudaHardwareExecutionOwner();
  explicit PreparedOpenCudaHardwareExecutionOwner(
      const hal::AdapterRegistration &registration);
  ~PreparedOpenCudaHardwareExecutionOwner();

  PreparedOpenCudaHardwareExecutionOwner(
      const PreparedOpenCudaHardwareExecutionOwner &) = delete;
  PreparedOpenCudaHardwareExecutionOwner &
  operator=(const PreparedOpenCudaHardwareExecutionOwner &) = delete;

  [[nodiscard]] PreparedOpenCudaHardwareSetupResult
  Configure(const PreparedOpenCudaHardwareRequest &request) noexcept;

  [[nodiscard]] PreparedOpenCudaHardwareFrameResult
  ExecuteHostFrame(const std::uint8_t *input, std::size_t input_size_bytes,
                   std::size_t input_row_stride_bytes, std::uint8_t *output,
                   std::size_t output_size_bytes,
                   std::size_t output_row_stride_bytes,
                   std::uint64_t capture_sequence) noexcept;

  [[nodiscard]] hal::StatusId
  CommitOutput(const hal::CompletionToken &completion,
               bool output_write_succeeded, hal::ReasonId *reason) noexcept;

  void Invalidate(hal::ReasonId reason) noexcept;
  void ObserveDeviceLoss() noexcept;
  // Shutdown is retryable. Snapshot().shutdown_pending remains true until the
  // production island proves that driver/context release is permitted.
  void Shutdown() noexcept;

  [[nodiscard]] PreparedOpenCudaHardwareExecutionSnapshot
  Snapshot() const noexcept;
  [[nodiscard]] const PreparedOpenCudaHardwareRequest &
  active_request() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::video
