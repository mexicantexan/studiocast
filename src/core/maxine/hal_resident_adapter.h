#pragma once

#include "core/hal/prepared_program.h"
#include "core/maxine/resident_frame_section.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace studiocast::maxine {

inline constexpr std::uint32_t kMaxineHalBindingVersion = 1;
inline constexpr std::size_t kMaxineHalPreparedSlots = 4;

struct ProductionResidentTelemetry;

// This is an adapter-owned snapshot. BindHostFrame copies the view into the
// adapter and returns an opaque neutral handle; callers never place a raw host
// pointer in ResourceHandle::adapter_opaque.
struct MaxineHostFrameInputBinding {
  std::uint32_t version = kMaxineHalBindingVersion;
  std::uint64_t capture_sequence = 0;
  HostRgbFrameView rgb{};
};

enum class MaxineOutputBoundary : std::uint8_t {
  resident = 1,
  final_readback,
  cpu_tail_readback,
};

// ResolveOutput validates an adapter-issued output token before returning this
// adapter-specific view. A host view is BGR U8. A resident view is an NvCV
// image owned by the prepared Maxine island and is never cross-adapter proof.
struct MaxineFrameOutputBinding {
  std::uint32_t version = kMaxineHalBindingVersion;
  std::uint64_t capture_sequence = 0;
  MaxineOutputBoundary boundary = MaxineOutputBoundary::resident;
  const std::uint8_t *host_bgr = nullptr;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::size_t stride_bytes = 0;
  ResidentImage resident{};

  [[nodiscard]] bool ValidHost() const noexcept;
  [[nodiscard]] bool ValidResident(const ResidentFrameKey &key) const noexcept;
};

// Authoritative counters are read directly from the configured production
// executor's ABI-call telemetry. The adapter samples this POD before and after
// Execute; the callback must not probe the device or synchronize it.
struct MaxineAuthoritativeTelemetrySnapshot {
  std::uint32_t version = kMaxineHalBindingVersion;
  std::uint64_t logical_context_generation = 0;
  std::uint64_t stream_identity = 0;
  std::uint64_t host_uploads = 0;
  std::uint64_t final_readbacks = 0;
  std::uint64_t cpu_tail_readbacks = 0;
  std::uint64_t synchronizations = 0;
  std::uint64_t intermediate_uploads = 0;
  std::uint64_t intermediate_readbacks = 0;
  bool device_lost = false;
};

[[nodiscard]] MaxineAuthoritativeTelemetrySnapshot
AuthoritativeSnapshotFromProduction(
    const ProductionResidentTelemetry &telemetry,
    std::uint64_t logical_context_generation,
    std::uint64_t stream_identity) noexcept;

struct MaxineHalAdapterHooks {
  void *owner = nullptr;

  // Called only after the executor reports a failed boundary. It must use
  // cached/latched state and must not probe a device on the frame path.
  hal::ReasonId (*classify_failure)(
      void *owner, ResidentBoundaryResult boundary) noexcept = nullptr;

  // Temporal programs require this setup-provided reset hook. It must be
  // bounded and synchronous in v1.
  bool (*reset_temporal)(void *owner, hal::ReasonId reason) noexcept = nullptr;

  // Required when a plan terminates at a host-visible readback. The callback
  // returns the already-prepared executor's host output without allocating.
  bool (*query_host_output)(void *owner, ResidentReadbackBoundary boundary,
                            std::uint64_t capture_sequence,
                            MaxineFrameOutputBinding *output) noexcept =
      nullptr;

  // Required for every production plan. This is a cached POD snapshot of
  // actual NvCV/CUDA/SDK ABI-call counters, not section orchestration counters.
  bool (*query_authoritative_telemetry)(
      void *owner,
      MaxineAuthoritativeTelemetrySnapshot *snapshot) noexcept = nullptr;

  void (*shutdown)(void *owner) noexcept = nullptr;
};

struct MaxineHalAdapterConfiguration {
  std::uint64_t adapter_instance_id = 0;
  hal::ExecutionDomainId execution_domain{};
  ResidentFrameKey resident_key{};
  hal::ResourceBudget capacity{};
};

// Production-shaped synchronous Maxine resident adapter. The injected executor
// must already be configured: model discovery, SDK/session construction,
// allocation, stream creation/binding, and asset upload belong before this
// adapter is exposed to a PreparedProgramSession.
class MaxineHalResidentAdapter final : public hal::IResidentAdapter {
public:
  MaxineHalResidentAdapter(MaxineHalAdapterConfiguration configuration,
                           IResidentFrameExecutor &executor,
                           MaxineHalAdapterHooks hooks = {}) noexcept;
  ~MaxineHalResidentAdapter() override = default;

  MaxineHalResidentAdapter(const MaxineHalResidentAdapter &) = delete;
  MaxineHalResidentAdapter &
  operator=(const MaxineHalResidentAdapter &) = delete;

  [[nodiscard]] std::uint64_t adapter_instance_id() const noexcept override;
  [[nodiscard]] hal::BackendId backend_id() const noexcept override;

  hal::StatusId Prepare(const hal::PreparedProgram &program,
                        hal::PreparedProgramHandle *candidate,
                        hal::ReadinessSnapshot *readiness,
                        hal::ReasonId *reason) noexcept override;
  hal::ExecuteResult
  Execute(const hal::PreparedProgramHandle &program,
          const hal::ExecuteRequest &request) noexcept override;
  hal::StatusId ResetTemporalState(const hal::PreparedProgramHandle &program,
                                   hal::ReasonId reason) noexcept override;
  void DestroyPrepared(hal::PreparedProgramHandle *program) noexcept override;
  void Shutdown() noexcept override;

  // Bind one borrowed RGB frame to an adapter-issued resource token. This is a
  // bounded POD copy and performs no allocation or provider work.
  [[nodiscard]] hal::StatusId BindHostFrame(const HostRgbFrameView &frame,
                                            std::uint64_t capture_sequence,
                                            std::uint64_t resource_id,
                                            hal::ResourceHandle *resource,
                                            hal::ReasonId *reason) noexcept;

  // Resolve only the most recent output for the prepared slot named by the
  // token. Stale, foreign, or forged tokens fail closed.
  [[nodiscard]] bool
  ResolveOutput(const hal::ResourceHandle &resource,
                MaxineFrameOutputBinding *output) const noexcept;

private:
  struct PreparedSlot {
    bool occupied = false;
    std::uint64_t handle_id = 0;
    std::uint64_t slot_nonce = 0;
    hal::PreparedProgram program{};
    ResidentFramePlan resident_plan{};
    ResidentFrameSection section{};
    MaxineFrameOutputBinding output{};
    hal::ResourceHandle output_resource{};
    std::uint64_t output_resource_id = 0;
    std::uint64_t output_cookie = 0;
  };

  [[nodiscard]] PreparedSlot *
  FindSlot(const hal::PreparedProgramHandle &program) noexcept;
  [[nodiscard]] const PreparedSlot *
  FindSlot(const hal::PreparedProgramHandle &program) const noexcept;
  [[nodiscard]] hal::ReasonId
  ClassifyFailure(ResidentBoundaryResult boundary,
                  hal::ReasonId fallback) const noexcept;
  [[nodiscard]] bool ConfigurationValid() const noexcept;
  [[nodiscard]] bool
  ProgramToResidentPlan(const hal::PreparedProgram &program,
                        ResidentFramePlan *plan,
                        hal::ReasonId *reason) const noexcept;
  [[nodiscard]] bool
  PopulateOutput(PreparedSlot &slot, std::uint64_t capture_sequence,
                 const ResidentFrameExecutionResult &execution,
                 hal::ResourceHandle *resource) noexcept;

  MaxineHalAdapterConfiguration configuration_{};
  IResidentFrameExecutor *executor_ = nullptr;
  MaxineHalAdapterHooks hooks_{};
  std::array<PreparedSlot, kMaxineHalPreparedSlots> slots_{};
  MaxineHostFrameInputBinding input_binding_{};
  hal::ResourceHandle input_resource_{};
  std::uint64_t input_resource_id_ = 0;
  std::uint64_t input_cookie_ = 0;
  std::uint64_t next_nonce_ = 1;
  std::uint64_t next_output_resource_id_ = 1;
  bool stopped_ = false;
};

} // namespace studiocast::maxine
