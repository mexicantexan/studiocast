#pragma once

#include "core/hal/static_registry.h"
#include "core/video/hardware_execution_runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace studiocast::video {

enum class PreparedHardwareRoute : std::uint8_t {
  legacy_shadow = 0,
  hal_live,
};

enum class PreparedHardwareEligibility : std::uint8_t {
  eligible = 0,
  not_strict_explicit_vulkan,
  no_qualified_final_stage,
  contains_unqualified_stage,
  tracked_vignette_center,
  unsupported_host_layout,
  invalid_generation,
  unavailable,
};

// Setup-only request for the first bounded production migration. It describes
// canonical intent and host transport; it does not expose a Vulkan object to
// CameraPipeline.
struct PreparedFinalHardwareRequest {
  hal::BackendId requested_backend = hal::BackendId::invalid;
  bool strict_backend_selection = false;
  bool request_fixed_center_vignette = false;
  bool request_mirror = false;
  bool contains_unqualified_stage = false;
  bool tracked_vignette_center = false;
  bool host_input_is_tight_rgb24 = false;
  bool host_output_is_tight_rgb24 = false;
  std::uint32_t input_width = 0;
  std::uint32_t input_height = 0;
  std::uint32_t output_width = 0;
  std::uint32_t output_height = 0;
  int vignette_intensity_percent = 0;
  hal::ProgramGenerations generations{};
};

struct PreparedHardwareSetupResult {
  PreparedHardwareRoute route = PreparedHardwareRoute::legacy_shadow;
  PreparedHardwareEligibility eligibility =
      PreparedHardwareEligibility::unavailable;
  hal::StatusId status = hal::StatusId::unavailable;
  hal::ReasonId reason = hal::ReasonId::none;
  bool retained_previous_live_program = false;
  HardwareExecutionSetupSnapshot setup{};
};

struct PreparedHardwareFrameResult {
  hal::StatusId status = hal::StatusId::failed;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::CompletionToken completion{};
  const std::uint8_t *output = nullptr;
  std::size_t output_size_bytes = 0;
  std::size_t output_row_stride_bytes = 0;
  bool output_ready = false;
  bool completion_pending = false;
};

// Cached setup identity used by control-plane consumers to prove that a
// committed output belongs to the program that is still prepared. It contains
// no backend handle or borrowed resource.
struct PreparedHardwareProgramIdentity {
  bool valid = false;
  std::uint64_t program_id = 0;
  std::uint64_t owner_adapter_instance_id = 0;
  hal::ProgramInvalidationKey invalidation_key{};
  hal::ExecutionDomainId execution_domain{};
  std::size_t stage_count = 0;
};

struct PreparedHardwareExecutionSnapshot {
  PreparedHardwareRoute route = PreparedHardwareRoute::legacy_shadow;
  PreparedHardwareEligibility eligibility =
      PreparedHardwareEligibility::unavailable;
  HardwareExecutionSetupSnapshot setup{};
  PreparedHardwareProgramIdentity active_program{};
  hal::ReadinessSnapshot readiness{};
  hal::ExecutionTelemetry telemetry{};
  hal::CommittedFrameSnapshot latest_committed_frame{};
  bool prepared = false;
};

struct PreparedHardwareCompatibilityTelemetry {
  std::uint64_t active_frames = 0;
  std::uint64_t upload_calls = 0;
  std::uint64_t download_calls = 0;
  std::uint64_t final_download_calls = 0;
  std::uint64_t forced_sync_calls = 0;
  std::uint64_t fallback_frames = 0;
  std::uint64_t runtime_failure_frames = 0;
};

[[nodiscard]] PreparedHardwareEligibility
EvaluatePreparedFinalHardwareEligibility(
    const PreparedFinalHardwareRequest &request) noexcept;
[[nodiscard]] PreparedHardwareCompatibilityTelemetry
ProjectPreparedHardwareCompatibilityTelemetry(
    const PreparedHardwareExecutionSnapshot &snapshot) noexcept;
[[nodiscard]] std::string_view
PreparedHardwareReasonCode(hal::ReasonId reason) noexcept;

// Owns the production adapter composition behind a PIMPL. CameraPipeline sees
// only neutral prepared execution and host-frame contracts. The injected
// constructor is a hermetic test seam and accepts only neutral HAL types.
class PreparedHardwareExecutionOwner {
public:
  PreparedHardwareExecutionOwner();
  explicit PreparedHardwareExecutionOwner(
      const hal::AdapterRegistration &registration);
  ~PreparedHardwareExecutionOwner();

  PreparedHardwareExecutionOwner(const PreparedHardwareExecutionOwner &) =
      delete;
  PreparedHardwareExecutionOwner &
  operator=(const PreparedHardwareExecutionOwner &) = delete;

  [[nodiscard]] PreparedHardwareSetupResult
  ConfigureFinal(const PreparedFinalHardwareRequest &request) noexcept;

  [[nodiscard]] PreparedHardwareFrameResult
  ExecuteHostFrame(const std::uint8_t *input, std::size_t input_size_bytes,
                   std::size_t input_row_stride_bytes, std::uint8_t *output,
                   std::size_t output_size_bytes,
                   std::size_t output_row_stride_bytes,
                   std::uint64_t capture_sequence) noexcept;

  [[nodiscard]] hal::StatusId
  CommitOutput(const hal::CompletionToken &completion,
               bool output_write_succeeded, hal::ReasonId *reason) noexcept;

  void Invalidate(hal::ReasonId reason) noexcept;
  void Shutdown() noexcept;

  [[nodiscard]] PreparedHardwareExecutionSnapshot Snapshot() const noexcept;
  [[nodiscard]] const PreparedFinalHardwareRequest &
  active_request() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::video
