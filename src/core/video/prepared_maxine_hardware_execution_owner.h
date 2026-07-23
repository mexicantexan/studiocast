#pragma once

#include "core/hal/static_registry.h"
#include "core/video/effects/broadcast_effects.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace studiocast::video {

enum class PreparedMaxineHardwareRoute : std::uint8_t {
  legacy_compatibility = 0,
  hal_live,
};

enum class PreparedMaxineHardwareEligibility : std::uint8_t {
  eligible = 0,
  no_qualified_stage,
  not_exact_maxine,
  multiple_resident_islands,
  cross_domain_chain,
  unsupported_stage,
  model_artifact_evidence_incomplete,
  temporal_reset_unsupported,
  unsupported_host_layout,
  invalid_generation,
  runtime_unavailable,
};

struct PreparedMaxineHardwareStageRequest {
  std::uint32_t stage_id = 0;
  hal::SemanticEffectId semantic = hal::SemanticEffectId::invalid;
  bool required = false;
  // Supplied only by setup-time model validation. Non-model utility stages
  // such as vignette deliberately leave this empty.
  hal::ModelArtifactIdentity model{};
};

// The camera pipeline already owns and prepares these SDK/runtime objects for
// its retained compatibility path. Their concrete types remain private to the
// implementation PIMPL; this view neither discovers nor owns them.
struct PreparedMaxineOpaqueRuntime {
  std::uintptr_t vfx = 0;
  std::uintptr_t nvcv = 0;
  std::uintptr_t ar = 0;
  std::uintptr_t cuda = 0;
  std::uintptr_t device_ops = 0;
};

struct PreparedMaxineReplacementAsset {
  const std::uint8_t *rgb = nullptr;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::size_t stride_bytes = 0;
  std::uint64_t generation = 0;
};

// Setup-only description for one conservative, contiguous Maxine execution
// island. The first production migration intentionally excludes temporal
// stages because the current owner cannot reset SDK history allocation-free.
struct PreparedMaxineHardwareRequest {
  hal::BackendId requested_backend = hal::BackendId::invalid;
  bool strict_backend_selection = false;
  bool single_resident_island = false;
  bool contains_other_execution_domain = false;
  bool temporal_reset_required = false;
  bool host_input_is_tight_rgb24 = false;
  bool host_output_is_tight_bgr24 = false;
  bool declares_output_format_cpu_tail = true;
  std::uint64_t adapter_instance_id = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  float vignette_center_x_px = 0.0F;
  float vignette_center_y_px = 0.0F;
  effects::BroadcastCameraEffects effects{};
  std::array<PreparedMaxineHardwareStageRequest, hal::kMaxPreparedStages>
      stages{};
  std::size_t stage_count = 0;
  hal::ProgramGenerations generations{};
  hal::ResourceBudget capacity{};
  PreparedMaxineOpaqueRuntime runtime{};
  PreparedMaxineReplacementAsset replacement{};
  std::string vfx_model_directory;
};

struct PreparedMaxineHardwareSetupResult {
  PreparedMaxineHardwareRoute route =
      PreparedMaxineHardwareRoute::legacy_compatibility;
  PreparedMaxineHardwareEligibility eligibility =
      PreparedMaxineHardwareEligibility::runtime_unavailable;
  hal::StatusId status = hal::StatusId::unavailable;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::ExecutionDomainId execution_domain{};
  hal::ResourceBudget known_minimum_budget{};
  bool retained_previous_live_program = false;
};

struct PreparedMaxineCompatibilityTelemetry {
  std::uint64_t rgb_to_bgr_calls = 0;
  std::uint64_t upload_attempts = 0;
  std::uint64_t upload_calls = 0;
  std::uint64_t final_download_attempts = 0;
  std::uint64_t final_download_calls = 0;
  std::uint64_t cpu_continuation_download_attempts = 0;
  std::uint64_t cpu_continuation_download_calls = 0;
  std::uint64_t device_bridge_attempts = 0;
  std::uint64_t device_bridge_calls = 0;
  std::uint64_t matte_inference_attempts = 0;
  std::uint64_t matte_inference_calls = 0;
  std::uint64_t shared_matte_reuses = 0;
  std::uint64_t sync_attempts = 0;
  std::uint64_t sync_calls = 0;
  std::uint64_t composite_attempts = 0;
  std::uint64_t composite_calls = 0;
  std::uint64_t synchronous_sdk_run_attempts = 0;
  std::uint64_t synchronous_sdk_run_calls = 0;
  std::uint64_t asynchronous_sdk_run_attempts = 0;
  std::uint64_t asynchronous_sdk_run_calls = 0;
  std::array<std::uint64_t, hal::kMaxPreparedStages> stage_attempts{};
  std::array<std::uint64_t, hal::kMaxPreparedStages> stage_successes{};
};

struct PreparedMaxineHardwareFrameResult {
  hal::StatusId status = hal::StatusId::failed;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::CompletionToken completion{};
  hal::TransferCounters transfers{};
  std::array<hal::StageExecutionResult, hal::kMaxPreparedStages> stages{};
  std::size_t stage_count = 0;
  const std::uint8_t *output = nullptr;
  std::size_t output_size_bytes = 0;
  std::size_t output_row_stride_bytes = 0;
  bool output_ready = false;
  bool completion_pending = false;
  PreparedMaxineCompatibilityTelemetry compatibility{};
};

struct PreparedMaxineHardwareExecutionSnapshot {
  PreparedMaxineHardwareRoute route =
      PreparedMaxineHardwareRoute::legacy_compatibility;
  PreparedMaxineHardwareEligibility eligibility =
      PreparedMaxineHardwareEligibility::runtime_unavailable;
  hal::ReadinessSnapshot readiness{};
  hal::ExecutionTelemetry telemetry{};
  hal::CommittedFrameSnapshot latest_committed_frame{};
  // Exact identity of the program against which readiness, telemetry, and
  // committed attribution must be validated.
  hal::PreparedProgram active_program{};
  hal::ExecutionDomainId execution_domain{};
  bool prepared = false;
  bool quarantined = false;
  bool stopped = false;
};

[[nodiscard]] PreparedMaxineHardwareEligibility
EvaluatePreparedMaxineHardwareEligibility(
    const PreparedMaxineHardwareRequest &request) noexcept;
[[nodiscard]] std::string_view
PreparedMaxineHardwareReasonCode(hal::ReasonId reason) noexcept;

// Backend-hidden owner for the first Maxine live migration. Production
// construction and destruction of MaxineProductionHalOwner, its adapter, its
// executor, and its prepared-program session occur only in the PIMPL.
class PreparedMaxineHardwareExecutionOwner {
public:
  PreparedMaxineHardwareExecutionOwner();
  explicit PreparedMaxineHardwareExecutionOwner(
      const hal::AdapterRegistration &registration);
  ~PreparedMaxineHardwareExecutionOwner();

  PreparedMaxineHardwareExecutionOwner(
      const PreparedMaxineHardwareExecutionOwner &) = delete;
  PreparedMaxineHardwareExecutionOwner &
  operator=(const PreparedMaxineHardwareExecutionOwner &) = delete;

  [[nodiscard]] PreparedMaxineHardwareSetupResult
  Configure(const PreparedMaxineHardwareRequest &request) noexcept;

  [[nodiscard]] PreparedMaxineHardwareFrameResult ExecuteHostFrame(
      const std::uint8_t *input, std::size_t input_size_bytes,
      std::size_t input_row_stride_bytes, std::uint8_t *output_bgr,
      std::size_t output_size_bytes, std::size_t output_row_stride_bytes,
      std::uint64_t capture_sequence) noexcept;

  [[nodiscard]] hal::StatusId
  CommitOutput(const hal::CompletionToken &completion,
               bool output_write_succeeded, hal::ReasonId *reason) noexcept;

  void NotifyDeviceLost() noexcept;
  void Invalidate(hal::ReasonId reason) noexcept;
  void Shutdown() noexcept;

  [[nodiscard]] PreparedMaxineHardwareExecutionSnapshot
  Snapshot() const noexcept;
  [[nodiscard]] const PreparedMaxineHardwareRequest &
  active_request() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::video
