#pragma once

#include "core/hal/prepared_program.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace studiocast::open_cuda {

// Setup components are failure domains, not execution stages.  The shared
// matting component prepares one model/session used by every VB/KL stage;
// vignette owns only its native kernel resources.
enum class OpenCudaSetupComponentId : std::uint8_t {
  invalid = 0,
  aggregate_island,
  shared_matting,
  native_vignette,
  native_mirror,
  shared_face_analysis,
  auto_frame,
  eye_contact,
  denoise,
};

// Cached setup evidence consumed by the HAL registry. None of these fields may
// be populated by probing from the frame path.
struct ResidentCapabilityFacts {
  bool cuda_driver_ready = false;
  bool stable_device_identity = false;
  bool exact_context_generation = false;
  bool adapter_owned_allocator = false;
  bool adapter_owned_stream = false;
  bool ort_cuda_provider_ready = false;
  bool ort_cuda_iobinding_ready = false;
  bool ort_user_stream_bound = false;
  bool explicit_ort_boundary_synchronization = false;
  bool sessions_prepared = false;
  bool reusable_buffers_prepared = false;
  bool bounded_resources = false;
  bool steady_state_allocation_free = false;
  bool steady_state_probe_free = false;
  bool no_cpu_fallback = false;
  bool shared_frame_analysis = false;
  std::uint32_t completion_synchronizations_per_frame = 0;
  bool device_preprocess = false;
  bool device_postprocess = false;
  bool native_vignette_kernel_prepared = false;
  bool virtual_background_blur = false;
  bool virtual_background_remove = false;
  bool virtual_background_replace = false;
  bool virtual_key_light = false;
  bool vignette = false;
  bool mirror = false;
  bool auto_frame = false;
  bool eye_contact = false;
  bool video_noise_removal = false;
  bool native_mirror_kernel_prepared = false;
  // Bounded metadata only; frame/tensor payloads remain device resident.
  std::uint32_t max_intermediate_readbacks_per_frame = 0;
  std::uint32_t max_cpu_tail_boundaries_per_frame = 0;
};

struct ResidentSemanticCapability {
  hal::SemanticEffectId semantic = hal::SemanticEffectId::invalid;
  hal::SemanticEffectProviderId provider =
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
  hal::StatusId status = hal::StatusId::blocked;
  hal::ReasonId reason = hal::ReasonId::capability_evidence_incomplete;
  bool live_stage = false;
  bool production_eligible = false;
  bool exact_evidence = false;
};

[[nodiscard]] ResidentSemanticCapability
EvaluateResidentCapability(hal::SemanticEffectId semantic,
                           const ResidentCapabilityFacts &facts) noexcept;

struct PreparedStageEvidence {
  std::uint32_t stage_id = 0;
  hal::SemanticEffectId semantic = hal::SemanticEffectId::invalid;
  hal::SemanticEffectProviderId provider =
      hal::SemanticEffectProviderId::invalid;
  hal::StageIoDescriptor input{};
  hal::StageIoDescriptor output{};
  hal::ModelArtifactIdentity model{};
  bool production_qualified = false;
  bool cuda_device_input = false;
  bool cuda_device_output = false;
  bool cuda_iobinding = false;
  bool device_preprocess = false;
  bool device_postprocess = false;
  bool shared_analysis = false;
  bool temporal_history_device_resident = false;
  bool cpu_fallback_disabled = false;
};

struct PreparedExecutorEvidence {
  ResidentCapabilityFacts capability{};
  std::array<PreparedStageEvidence, hal::kMaxPreparedStages> stages{};
  std::size_t stage_count = 0;
};

// Monotonic counters owned by the injected executor. The adapter snapshots
// these around Execute so hidden transfers, setup work, or allocation cannot
// be concealed by a self-reported ExecuteResult.
struct ExecutorTelemetrySnapshot {
  std::uint64_t allocation_requests = 0;
  std::uint64_t provider_probes = 0;
  std::uint64_t model_scans = 0;
  std::uint64_t manifest_parses = 0;
  std::uint64_t session_constructions = 0;
  std::uint64_t json_serializations = 0;
  std::uint64_t uploads = 0;
  std::uint64_t final_readbacks = 0;
  std::uint64_t intermediate_readbacks = 0;
  std::uint64_t intermediate_uploads = 0;
  std::uint64_t synchronizations = 0;
  std::uint64_t cpu_tail_boundaries = 0;
  std::uint64_t shared_analysis_runs = 0;
};

struct PreparedExecutorHandle {
  std::uint64_t handle_id = 0;
  std::uint64_t program_id = 0;
  hal::ExecutionDomainId execution_domain{};
  hal::ResourceBudget reserved_budget{};
  hal::ResourceHandle prepared_output{};
  std::array<std::uint64_t, 4> opaque{};
};

struct ExecutorPrepareResult {
  hal::StatusId status = hal::StatusId::blocked;
  hal::ReasonId reason = hal::ReasonId::capability_evidence_incomplete;
  PreparedExecutorHandle handle{};
  PreparedExecutorEvidence evidence{};
  OpenCudaSetupComponentId failed_component =
      OpenCudaSetupComponentId::invalid;
};

class IPreparedResidentExecutor {
public:
  virtual ~IPreparedResidentExecutor() = default;

  // This is the only setup entrypoint. CUDA contexts, streams, sessions,
  // IoBindings, kernels, buffers, and scratch storage must exist before it
  // returns ready.
  virtual ExecutorPrepareResult
  Prepare(const hal::PreparedProgram &program) noexcept = 0;
  virtual hal::ExecuteResult
  Execute(const PreparedExecutorHandle &program,
          const hal::ExecuteRequest &request) noexcept = 0;
  // These calls bind already-created upload/readback resources to borrowed
  // host storage. Implementations must not allocate, probe, rebuild a session,
  // or inspect model metadata here.
  virtual hal::StatusId BindHostFrame(const PreparedExecutorHandle &program,
                                      const hal::HostFrameBinding &binding,
                                      hal::HostFrameTokens *tokens,
                                      hal::ReasonId *reason) noexcept = 0;
  virtual hal::StatusId
  DiscardUnsubmittedHostFrame(const PreparedExecutorHandle &,
                              const hal::HostFrameTokens &, std::uint64_t,
                              hal::ReasonId *reason) noexcept {
    if (reason)
      *reason = hal::ReasonId::host_frame_binding_unsupported;
    return hal::StatusId::blocked;
  }
  virtual hal::StatusId
  ResolveHostFrameOutput(const PreparedExecutorHandle &program,
                         const hal::CompletionToken &completion,
                         const hal::ResourceHandle &output,
                         hal::HostFrameOutputResolution *resolution,
                         hal::ReasonId *reason) noexcept = 0;
  virtual hal::StatusId
  ResetTemporalState(const PreparedExecutorHandle &program,
                     hal::ReasonId reason) noexcept = 0;
  virtual void DestroyPrepared(PreparedExecutorHandle *program) noexcept = 0;
  virtual void Shutdown() noexcept = 0;
  [[nodiscard]] virtual ExecutorTelemetrySnapshot
  Telemetry() const noexcept = 0;
};

class ResidentAdapter final : public hal::IResidentAdapter {
public:
  static constexpr std::size_t kMaxPreparedPrograms = 4;

  ResidentAdapter(std::uint64_t adapter_instance_id,
                  IPreparedResidentExecutor &executor) noexcept;
  ~ResidentAdapter() override;

  ResidentAdapter(const ResidentAdapter &) = delete;
  ResidentAdapter &operator=(const ResidentAdapter &) = delete;

  [[nodiscard]] std::uint64_t adapter_instance_id() const noexcept override {
    return adapter_instance_id_;
  }
  [[nodiscard]] hal::BackendId backend_id() const noexcept override {
    return hal::BackendId::cuda;
  }

  hal::StatusId Prepare(const hal::PreparedProgram &program,
                        hal::PreparedProgramHandle *candidate,
                        hal::ReadinessSnapshot *readiness,
                        hal::ReasonId *reason) noexcept override;
  hal::ExecuteResult
  Execute(const hal::PreparedProgramHandle &program,
          const hal::ExecuteRequest &request) noexcept override;
  hal::StatusId BindHostFrame(const hal::PreparedProgramHandle &program,
                              const hal::HostFrameBinding &binding,
                              hal::HostFrameTokens *tokens,
                              hal::ReasonId *reason) noexcept override;
  hal::StatusId
  DiscardUnsubmittedHostFrame(const hal::PreparedProgramHandle &program,
                              const hal::HostFrameTokens &tokens,
                              std::uint64_t capture_sequence,
                              hal::ReasonId *reason) noexcept override;
  hal::StatusId
  ResolveHostFrameOutput(const hal::PreparedProgramHandle &program,
                         const hal::CompletionToken &completion,
                         const hal::ResourceHandle &output,
                         hal::HostFrameOutputResolution *resolution,
                         hal::ReasonId *reason) noexcept override;
  hal::StatusId ResetTemporalState(const hal::PreparedProgramHandle &program,
                                   hal::ReasonId reason) noexcept override;
  void DestroyPrepared(hal::PreparedProgramHandle *program) noexcept override;
  void Shutdown() noexcept override;

private:
  struct Slot {
    bool active = false;
    std::uint64_t public_handle_id = 0;
    hal::PreparedProgram program{};
    PreparedExecutorHandle executor_handle{};
    bool requires_explicit_provider_sync = false;
    std::uint32_t completion_synchronizations_per_frame = 0;
    std::uint32_t max_intermediate_readbacks_per_frame = 0;
    std::uint32_t max_cpu_tail_boundaries_per_frame = 0;
    bool quarantined = false;
    hal::HostFrameBinding host_binding{};
    hal::HostFrameTokens host_tokens{};
    hal::CompletionToken host_completion{};
    bool host_binding_active = false;
    bool host_execution_completed = false;
    bool host_execution_started = false;
  };

  [[nodiscard]] Slot *Find(const hal::PreparedProgramHandle &program) noexcept;
  [[nodiscard]] const Slot *
  Find(const hal::PreparedProgramHandle &program) const noexcept;

  std::uint64_t adapter_instance_id_ = 0;
  IPreparedResidentExecutor *executor_ = nullptr;
  std::array<Slot, kMaxPreparedPrograms> slots_{};
  std::uint64_t next_handle_id_ = 1;
  std::uint64_t next_completion_id_ = 1;
  bool stopped_ = false;
};

} // namespace studiocast::open_cuda
