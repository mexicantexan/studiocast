#pragma once

#include "core/hal/domain_planner.h"

#include <cstdint>

namespace studiocast::video {

// v1 deliberately has one live resident island. shadow_only records the new
// resolver/planner decision while the caller keeps its legacy implementation
// live; this facade never falls back to that implementation from the frame
// path.
enum class HardwareExecutionSetupMode : std::uint8_t {
  hal_live = 1,
  shadow_only,
};

struct HardwareExecutionSetupRequest {
  hal::ResolutionRequest resolution{};
  // Supplied by the owning backend factory after model, geometry, provider,
  // device, and resource preparation has completed. It is ignored only for a
  // zero-stage passthrough or shadow-only attempt.
  hal::PreparedProgram prepared_program{};
  HardwareExecutionSetupMode mode = HardwareExecutionSetupMode::hal_live;

  // Transitional comparison input from the existing resolver. It is setup
  // metadata only and cannot be selected as a frame-time fallback.
  bool compare_legacy_resolution = false;
  hal::ResolutionResult legacy_resolution{};
  bool legacy_path_retained_for_setup_rollback = false;
};

struct HardwareExecutionSetupSnapshot {
  HardwareExecutionSetupMode mode = HardwareExecutionSetupMode::hal_live;
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::ResolutionResult resolution{};
  hal::DomainPlan domain_plan{};
  hal::ShadowComparison shadow_comparison{};
  bool shadow_compared = false;
  bool legacy_path_retained_for_setup_rollback = false;
};

struct HardwareFrameResult {
  hal::StatusId status = hal::StatusId::failed;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::CompletionToken completion{};
  hal::HostFrameOutputResolution output{};
  bool output_ready = false;
  bool passthrough = false;
};

class HardwareExecutionRuntime {
public:
  // Registry records and their adapters must outlive this runtime.
  explicit HardwareExecutionRuntime(
      const hal::StaticAdapterRegistry &registry) noexcept;
  ~HardwareExecutionRuntime();

  HardwareExecutionRuntime(const HardwareExecutionRuntime &) = delete;
  HardwareExecutionRuntime &
  operator=(const HardwareExecutionRuntime &) = delete;

  [[nodiscard]] hal::StatusId
  Configure(const HardwareExecutionSetupRequest &request,
            hal::ReasonId *reason) noexcept;

  // Synchronous v1 execution. A successful return owns one pending completion
  // until CommitOutput is called, including when output resolution itself
  // subsequently fails. The live HostFrameBinding is opaque here: its host
  // boundary descriptors are validated by the prepared-session contract and
  // must not be inferred from resident stage input/output descriptors.
  [[nodiscard]] HardwareFrameResult
  ExecuteHostFrame(const hal::HostFrameBinding &binding) noexcept;
  [[nodiscard]] hal::StatusId
  CommitOutput(const hal::CompletionToken &completion,
               bool output_write_succeeded, hal::ReasonId *reason) noexcept;

  // Setup/control-plane operations. Invalidate never chooses another adapter.
  void Invalidate(hal::ReasonId reason) noexcept;
  void Shutdown() noexcept;

  [[nodiscard]] const HardwareExecutionSetupSnapshot &
  latest_setup() const noexcept {
    return latest_setup_;
  }
  [[nodiscard]] const hal::ResolutionResult &
  active_resolution() const noexcept {
    return active_resolution_;
  }
  [[nodiscard]] const hal::DomainPlan &active_domain_plan() const noexcept {
    return active_domain_plan_;
  }
  [[nodiscard]] const hal::ReadinessSnapshot &readiness() const noexcept;
  [[nodiscard]] const hal::ExecutionTelemetry &telemetry() const noexcept;
  [[nodiscard]] const hal::CommittedFrameSnapshot &
  latest_committed_frame() const noexcept;

  [[nodiscard]] bool prepared() const noexcept;
  [[nodiscard]] bool passthrough() const noexcept { return passthrough_; }
  [[nodiscard]] bool frame_pending() const noexcept { return frame_pending_; }
  [[nodiscard]] bool stopped() const noexcept { return stopped_; }
  [[nodiscard]] HardwareExecutionSetupMode active_mode() const noexcept {
    return active_mode_;
  }

private:
  [[nodiscard]] bool
  ValidateExactProgram(const HardwareExecutionSetupRequest &request,
                       const hal::ResolutionResult &resolution,
                       const hal::DomainPlan &domain_plan,
                       hal::ReasonId *reason) const noexcept;
  [[nodiscard]] bool ValidatePassthroughBinding(
      const hal::HostFrameBinding &binding) const noexcept;
  void
  ActivatePassthrough(const HardwareExecutionSetupRequest &request) noexcept;
  void ClearActive() noexcept;

  const hal::StaticAdapterRegistry *registry_ = nullptr;
  hal::PreparedProgramSession session_{};
  HardwareExecutionSetupSnapshot latest_setup_{};
  hal::ResolutionResult active_resolution_{};
  hal::DomainPlan active_domain_plan_{};
  hal::ReadinessSnapshot local_readiness_{};
  hal::ExecutionTelemetry local_telemetry_{};
  hal::CommittedFrameSnapshot local_committed_frame_{};
  hal::CompletionToken pending_completion_{};
  HardwareExecutionSetupMode active_mode_ =
      HardwareExecutionSetupMode::hal_live;
  std::uint64_t passthrough_token_sequence_ = 0;
  bool passthrough_ = false;
  bool frame_pending_ = false;
  bool telemetry_from_session_ = false;
  bool stopped_ = false;
};

} // namespace studiocast::video
