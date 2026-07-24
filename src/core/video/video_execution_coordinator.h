#pragma once

#include "core/hal/static_registry.h"
#include "core/video/video_adapter_factory.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace studiocast::video {

inline constexpr std::size_t kMaxVideoAdapterFactories = 16;

struct VideoExecutionPrepareEvidence {
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  VideoExecutionRoute selected_route = VideoExecutionRoute::passthrough;
  hal::ResolutionResult resolution{};
  hal::DomainPlan domain_plan{};
  hal::ReadinessSnapshot readiness{};
  std::uint64_t selected_factory_id = 0;
  bool prepared_factory_attempted = false;
  bool compatibility_factory_attempted = false;
  bool compatibility_consumed_frozen_resolution = false;
  bool candidate_resources_rolled_back = false;
  bool old_plan_retained = false;
};

struct VideoExecutionCoordinatorSnapshot {
  FrozenVideoExecutionPlan plan{};
  VideoExecutionPrepareEvidence prepare{};
  hal::ReadinessSnapshot readiness{};
  hal::ExecutionTelemetry telemetry{};
  hal::CommittedFrameSnapshot latest_committed_frame{};
  bool prepared = false;
  bool frame_pending = false;
  bool quarantined = false;
  bool stopped = false;
};

class VideoExecutionCoordinator {
public:
  // Cached capability registrations and resident adapters must outlive the
  // coordinator. Factories must also outlive it.
  explicit VideoExecutionCoordinator(
      const hal::StaticAdapterRegistry &capabilities) noexcept;
  ~VideoExecutionCoordinator();

  VideoExecutionCoordinator(const VideoExecutionCoordinator &) = delete;
  VideoExecutionCoordinator &
  operator=(const VideoExecutionCoordinator &) = delete;

  [[nodiscard]] hal::StatusId
  RegisterFactory(const VideoAdapterFactoryRegistration &registration,
                  hal::ReasonId *reason) noexcept;

  [[nodiscard]] hal::StatusId Prepare(const VideoExecutionPlanRequest &request,
                                      VideoExecutionPrepareEvidence *evidence,
                                      hal::ReasonId *reason) noexcept;

  [[nodiscard]] VideoExecutionFrameResult
  Execute(const hal::HostFrameBinding &binding) noexcept;
  [[nodiscard]] hal::StatusId
  CommitOutput(const hal::CompletionToken &completion,
               bool output_write_succeeded, hal::ReasonId *reason) noexcept;

  void Invalidate(hal::ReasonId reason) noexcept;
  [[nodiscard]] hal::StatusId ResetTemporal(hal::ReasonId reason) noexcept;
  void Shutdown() noexcept;

  [[nodiscard]] VideoExecutionCoordinatorSnapshot Snapshot() const noexcept;

private:
  [[nodiscard]] const VideoAdapterFactoryRegistration *
  FindFactory(VideoExecutionRoute route,
              const hal::ResolutionResult &resolution,
              const hal::DomainPlan &domain_plan) const noexcept;
  [[nodiscard]] hal::StatusId
  TryPrepareRoute(const VideoExecutionPlanRequest &request,
                  VideoExecutionRoute route,
                  const hal::ResolutionResult &resolution,
                  const hal::DomainPlan &domain_plan,
                  std::unique_ptr<IVideoExecutionExecutor> *candidate,
                  FrozenVideoExecutionPlan *candidate_plan,
                  VideoAdapterFactoryPreflight *preflight,
                  VideoAdapterPrepareEvidence *prepare_evidence,
                  hal::ReasonId *reason) noexcept;
  [[nodiscard]] bool
  MayRetainOld(const VideoExecutionPlanRequest &request) const noexcept;
  void ActivatePassthrough(const VideoExecutionPlanRequest &request,
                           const hal::ResolutionResult &resolution,
                           const hal::DomainPlan &domain_plan) noexcept;
  void ClearPending() noexcept;
  void ClearActive() noexcept;

  const hal::StaticAdapterRegistry *capabilities_ = nullptr;
  std::array<VideoAdapterFactoryRegistration, kMaxVideoAdapterFactories>
      factories_{};
  std::size_t factory_count_ = 0;
  std::unique_ptr<IVideoExecutionExecutor> executor_;
  FrozenVideoExecutionPlan plan_{};
  VideoExecutionPrepareEvidence prepare_{};
  hal::ReadinessSnapshot readiness_{};
  hal::ExecutionTelemetry telemetry_{};
  hal::CommittedFrameSnapshot committed_{};
  hal::CompletionToken pending_completion_{};
  std::array<hal::StageExecutionResult, hal::kMaxPreparedStages>
      pending_stages_{};
  std::size_t pending_stage_count_ = 0;
  std::uint64_t passthrough_token_ = 0;
  bool prepared_ = false;
  bool passthrough_ = false;
  bool frame_pending_ = false;
  bool quarantined_ = false;
  bool stopped_ = false;
};

} // namespace studiocast::video
