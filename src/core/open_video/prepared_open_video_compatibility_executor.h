#pragma once

#include "core/video/effects/broadcast_effects.h"
#include "core/video/video_adapter_factory.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace studiocast::open_video {

struct PreparedOpenVideoCompatibilityConfig {
  using PreparedStageRunner = bool (*)(
      void *context, const video::CanonicalVideoStage &stage,
      std::uint64_t capture_sequence, std::uint8_t *rgb, int width, int height,
      std::size_t stride, std::string *error) noexcept;

  video::VideoExecutionPlanRequest plan{};
  video::effects::BroadcastCameraEffects effects{};
  std::array<hal::SemanticEffectProviderId, hal::kMaxPreparedStages>
      providers{};
  std::array<hal::BackendId, hal::kMaxPreparedStages> backends{};
  hal::BackendId selected_backend = hal::BackendId::invalid;
  hal::SemanticEffectProviderId selected_provider =
      hal::SemanticEffectProviderId::invalid;
  hal::ExecutionDomainId execution_domain{};
  hal::TransferCounters expected_transfers{};
  // Optional already-prepared runner used by backend-private composed stages
  // and hermetic conformance tests. Ownership and preparation remain with the
  // caller; Execute never probes, prepares, or allocates it.
  PreparedStageRunner prepared_stage_runner = nullptr;
  void *prepared_stage_runner_context = nullptr;
  std::shared_ptr<void> prepared_stage_runner_owner;
  hal::ReasonId setup_fallback_reason = hal::ReasonId::none;
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
