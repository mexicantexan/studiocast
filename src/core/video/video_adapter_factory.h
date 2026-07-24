#pragma once

#include "core/video/video_execution_plan.h"

#include <cstdint>
#include <memory>

namespace studiocast::video {

struct VideoExecutionFrameResult {
  hal::StatusId status = hal::StatusId::failed;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::CompletionToken completion{};
  hal::HostFrameOutputResolution output{};
  hal::ExecutionDomainId execution_domain{};
  std::array<hal::StageExecutionResult, hal::kMaxPreparedStages> stages{};
  std::size_t stage_count = 0;
  hal::TransferCounters transfers{};
  std::uint64_t frame_elapsed_ns = 0;
  bool output_ready = false;
  bool passthrough = false;
};

struct VideoExecutionAdapterSnapshot {
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::ReadinessSnapshot readiness{};
  hal::ExecutionTelemetry telemetry{};
  hal::CommittedFrameSnapshot latest_committed_frame{};
  hal::ExecutionDomainId execution_domain{};
  std::uint64_t plan_id = 0;
  bool prepared = false;
  bool stopped = false;
};

class IVideoExecutionExecutor {
public:
  virtual ~IVideoExecutionExecutor() = default;

  // V1 is synchronous but completion-shaped. Implementations must not probe,
  // allocate, rebuild, discover, serialize, or select a provider here.
  [[nodiscard]] virtual VideoExecutionFrameResult
  Execute(const hal::HostFrameBinding &binding) noexcept = 0;
  [[nodiscard]] virtual hal::StatusId
  CommitOutput(const hal::CompletionToken &completion,
               bool output_write_succeeded, hal::ReasonId *reason) noexcept = 0;
  virtual void Invalidate(hal::ReasonId reason) noexcept = 0;
  [[nodiscard]] virtual hal::StatusId
  ResetTemporal(hal::ReasonId reason) noexcept = 0;
  virtual void Shutdown() noexcept = 0;
  [[nodiscard]] virtual VideoExecutionAdapterSnapshot
  Snapshot() const noexcept = 0;
};

struct VideoAdapterFactoryPreflight {
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::BackendId backend = hal::BackendId::invalid;
  hal::SemanticEffectProviderId provider =
      hal::SemanticEffectProviderId::invalid;
  hal::ExecutionDomainId execution_domain{};
  hal::ReadinessSnapshot readiness{};
  hal::PreparedProgram prepared_program{};
  bool cached_evidence_only = false;
  bool accepts_frozen_resolution = false;
  bool performs_selection = false;
};

struct VideoAdapterPrepareEvidence {
  hal::StatusId status = hal::StatusId::invalid;
  hal::ReasonId reason = hal::ReasonId::none;
  hal::ReadinessSnapshot readiness{};
  bool candidate_resources_rolled_back = false;
};

class IVideoAdapterFactory {
public:
  virtual ~IVideoAdapterFactory() = default;

  // Preflight is setup-only and may consume only cached evidence. The
  // resolution and domain plan are frozen by the coordinator and immutable.
  [[nodiscard]] virtual VideoAdapterFactoryPreflight
  Preflight(const VideoExecutionPlanRequest &request,
            const hal::ResolutionResult &frozen_resolution,
            const hal::DomainPlan &frozen_domain_plan,
            VideoExecutionRoute route) noexcept = 0;

  // The factory must consume the frozen plan. There is intentionally no
  // registry or selection argument here, so a compatibility implementation
  // cannot resolve a different backend after prepared preflight fails.
  [[nodiscard]] virtual std::unique_ptr<IVideoExecutionExecutor>
  Prepare(const FrozenVideoExecutionPlan &plan,
          VideoAdapterPrepareEvidence *evidence) noexcept = 0;
};

struct VideoAdapterFactoryRegistration {
  std::uint64_t factory_id = 0;
  VideoExecutionRoute route = VideoExecutionRoute::prepared_resident;
  hal::BackendId backend = hal::BackendId::invalid;
  // Invalid means the factory accepts the already-frozen per-stage providers
  // for a mixed-provider island. It is not permission to select a provider.
  hal::SemanticEffectProviderId provider =
      hal::SemanticEffectProviderId::invalid;
  std::uint64_t adapter_instance_id = 0;
  hal::ExecutionDomainId execution_domain{};
  std::uint32_t selection_priority = 0;
  IVideoAdapterFactory *factory = nullptr;
};

} // namespace studiocast::video
