#pragma once

#include "core/hal/prepared_program.h"
#include "core/maxine/hal_resident_adapter.h"
#include "core/maxine/production_resident_frame_executor.h"

#include <cstdint>
#include <memory>
#include <string>

namespace studiocast::maxine {

// Setup-time inputs for one Maxine-owned execution island. Runtime pointers
// and decoded assets must remain valid for the lifetime of the returned owner.
// The factory never discovers SDKs, models, devices, or assets.
struct MaxineProductionBackendRequest {
  std::uint64_t adapter_instance_id = 0;
  ProductionResidentRuntime runtime{};
  ProductionResidentSetup setup{};
  hal::ResourceBudget capacity{};
  IResidentDeviceOps *device_ops = nullptr;
};

struct MaxineProductionPreparationResult {
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::ExecutionDomainId execution_domain{};
  hal::ResourceBudget known_minimum_budget{};
  std::string diagnostic;
};

// Owns the concrete executor, adapter, and prepared-program session in their
// required destruction order. CreateBackend constructs a candidate without
// mutating any prior owner; callers retain the old owner until
// ConfigureProgram succeeds, which provides setup rollback and old-plan
// retention without in-place executor reconfiguration.
class MaxineProductionHalOwner final {
public:
  [[nodiscard]] static std::unique_ptr<MaxineProductionHalOwner>
  CreateBackend(const MaxineProductionBackendRequest &request,
                MaxineProductionPreparationResult *result) noexcept;

  ~MaxineProductionHalOwner();

  MaxineProductionHalOwner(const MaxineProductionHalOwner &) = delete;
  MaxineProductionHalOwner &
  operator=(const MaxineProductionHalOwner &) = delete;

  [[nodiscard]] hal::StatusId
  ConfigureProgram(const hal::PreparedProgram &program,
                   MaxineProductionPreparationResult *result) noexcept;

  [[nodiscard]] const hal::ExecutionDomainId &
  execution_domain() const noexcept {
    return execution_domain_;
  }
  [[nodiscard]] const ResidentFrameKey &resident_key() const noexcept {
    return resident_key_;
  }
  [[nodiscard]] const hal::ResourceBudget &capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] const hal::ResourceBudget &
  known_minimum_budget() const noexcept {
    return known_minimum_budget_;
  }
  [[nodiscard]] bool prepared() const noexcept {
    return session_ && session_->prepared();
  }
  [[nodiscard]] bool stopped() const noexcept { return stopped_; }
  [[nodiscard]] bool device_lost() const noexcept { return device_lost_; }

  [[nodiscard]] hal::PreparedProgramSession *session() noexcept {
    return session_.get();
  }
  [[nodiscard]] const hal::PreparedProgramSession *session() const noexcept {
    return session_.get();
  }
  [[nodiscard]] MaxineHalResidentAdapter *adapter() noexcept {
    return adapter_.get();
  }
  [[nodiscard]] const MaxineHalResidentAdapter *adapter() const noexcept {
    return adapter_.get();
  }
  [[nodiscard]] ProductionResidentFrameExecutor *executor() noexcept {
    return executor_.get();
  }
  [[nodiscard]] const ProductionResidentFrameExecutor *executor() const
      noexcept {
    return executor_.get();
  }
  [[nodiscard]] MaxineOutputFormatConversionTelemetry
  output_format_conversion_telemetry() const noexcept {
    return adapter_ ? adapter_->output_format_conversion_telemetry()
                    : MaxineOutputFormatConversionTelemetry{};
  }

  // Device-loss detection remains outside this owner because the current
  // Maxine wrappers collapse SDK/CUDA failures to a boolean. A cached,
  // authoritative detector may latch loss here; no device probe occurs.
  void NotifyDeviceLost() noexcept;
  void Shutdown() noexcept;

private:
  MaxineProductionHalOwner() = default;

  [[nodiscard]] static hal::ReasonId
  ClassifyFailure(void *owner, ResidentBoundaryResult boundary) noexcept;
  [[nodiscard]] static bool
  QueryHostOutput(void *owner, ResidentReadbackBoundary boundary,
                  std::uint64_t capture_sequence,
                  MaxineFrameOutputBinding *output) noexcept;
  [[nodiscard]] static bool QueryAuthoritativeTelemetry(
      void *owner,
      MaxineAuthoritativeTelemetrySnapshot *snapshot) noexcept;
  static void AdapterShutdown(void *owner) noexcept;

  [[nodiscard]] bool ValidateProgramStageSet(
      const hal::PreparedProgram &program, hal::ReasonId *reason) const noexcept;

  std::unique_ptr<ProductionResidentFrameExecutor> executor_;
  std::unique_ptr<MaxineHalResidentAdapter> adapter_;
  std::unique_ptr<hal::PreparedProgramSession> session_;
  hal::ExecutionDomainId execution_domain_{};
  ResidentFrameKey resident_key_{};
  hal::ResourceBudget capacity_{};
  hal::ResourceBudget known_minimum_budget_{};
  std::uint32_t enabled_stage_mask_ = 0;
  std::uint64_t logical_context_generation_ = 0;
  bool temporal_reset_required_ = false;
  bool device_lost_ = false;
  bool stopped_ = false;
};

// Counts only StudioCast-owned frame, bridge, background, and staging storage.
// Opaque SDK/model allocations remain governed by the caller-supplied capacity
// and are not falsely represented as measurable by this API.
[[nodiscard]] bool KnownMinimumMaxineBudget(
    const ProductionResidentSetup &setup, hal::ResourceBudget *budget) noexcept;

} // namespace studiocast::maxine
