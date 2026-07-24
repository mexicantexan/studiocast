#pragma once

#include "core/video/video_execution_coordinator.h"

#include <cstdint>
#include <memory>

namespace studiocast::video {

// Runtime-composition boundary for prepared video execution. CameraPipeline
// supplies only a normalized plan and host bindings; backend construction,
// registry population, exact-domain factory registration, and coordinator
// ownership stay behind this PIMPL.
//
// The default instance currently composes only the production-qualified
// Vulkan final island. Other prepared islands remain on their explicit
// compatibility owners until their request-specific native setup evidence can
// be moved behind this same boundary.
class ProductionVideoExecutionComposition {
public:
  ProductionVideoExecutionComposition();

  // Neutral hermetic seam. The pointed-to adapter and factory must outlive the
  // composition. No optional backend type crosses this constructor.
  ProductionVideoExecutionComposition(
      const hal::AdapterRegistration &adapter,
      const VideoAdapterFactoryRegistration &factory);

  ~ProductionVideoExecutionComposition();

  ProductionVideoExecutionComposition(
      const ProductionVideoExecutionComposition &) = delete;
  ProductionVideoExecutionComposition &
  operator=(const ProductionVideoExecutionComposition &) = delete;

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
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::video
