#pragma once

#include "core/video/effects/broadcast_effects.h"
#include "core/video/video_adapter_factory.h"

#include <array>
#include <memory>

namespace studiocast::open_video {

struct PreparedOpenVideoCompatibilityConfig {
  video::VideoExecutionPlanRequest plan{};
  video::effects::BroadcastCameraEffects effects{};
  std::array<hal::SemanticEffectProviderId, hal::kMaxPreparedStages>
      providers{};
  hal::TransferCounters expected_transfers{};
};

struct PreparedOpenVideoCompatibilitySetupResult {
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::ReadinessSnapshot readiness{};
};

// Prepared host-visible Open Video compatibility executor. It intentionally
// covers only the existing CPU-tensor FastDVDnet and gaze-correction routes;
// it never advertises them as a CUDA-resident island. All model/session/scratch
// setup occurs in Create, and Execute uses the prepared-only runtime methods.
class PreparedOpenVideoCompatibilityExecutor final
    : public video::IVideoExecutionExecutor {
public:
  [[nodiscard]] static std::unique_ptr<PreparedOpenVideoCompatibilityExecutor>
  Create(const PreparedOpenVideoCompatibilityConfig &config,
         PreparedOpenVideoCompatibilitySetupResult *result) noexcept;
  ~PreparedOpenVideoCompatibilityExecutor() override;

  PreparedOpenVideoCompatibilityExecutor(
      const PreparedOpenVideoCompatibilityExecutor &) = delete;
  PreparedOpenVideoCompatibilityExecutor &
  operator=(const PreparedOpenVideoCompatibilityExecutor &) = delete;

  [[nodiscard]] video::VideoExecutionFrameResult
  Execute(const hal::HostFrameBinding &binding) noexcept override;
  [[nodiscard]] hal::StatusId
  CommitOutput(const hal::CompletionToken &completion,
               bool output_write_succeeded,
               hal::ReasonId *reason) noexcept override;
  void Invalidate(hal::ReasonId reason) noexcept override;
  [[nodiscard]] hal::StatusId
  ResetTemporal(hal::ReasonId reason) noexcept override;
  void Shutdown() noexcept override;
  [[nodiscard]] video::VideoExecutionAdapterSnapshot
  Snapshot() const noexcept override;

private:
  struct Impl;
  PreparedOpenVideoCompatibilityExecutor();
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::open_video
