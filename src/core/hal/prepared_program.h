#pragma once

#include "core/hal/descriptors.h"
#include "core/hal/execution_domain.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace studiocast::hal {

inline constexpr std::size_t kMaxPreparedStages = 32;
inline constexpr std::size_t kMaxCpuTails = 8;
inline constexpr std::size_t kMaxLatencyHistogramBuckets = 16;
inline constexpr std::size_t kMaxCapabilityEvidenceRecords = 24;

enum class MemoryDomain : std::uint16_t {
  invalid = 0,
  host,
  upload_staging,
  device_local,
  readback_staging,
  provider_owned,
  scratch,
};

enum class SynchronizationState : std::uint16_t {
  invalid = 0,
  ready,
  adapter_in_use,
  completion_pending,
  device_lost,
};

enum class ResourceLifetime : std::uint16_t {
  invalid = 0,
  frame_borrowed,
  prepared_program,
  adapter_context,
};

struct ResourceBudget {
  std::uint64_t persistent_bytes = 0;
  std::uint64_t scratch_bytes = 0;
  std::uint64_t upload_staging_bytes = 0;
  std::uint64_t readback_staging_bytes = 0;
  std::uint32_t resource_count = 0;
  std::uint32_t in_flight_slots = 1;
};

struct ResourceHandle;
using ResourceDestroyFn = void (*)(void *owner,
                                   ResourceHandle *resource) noexcept;

struct ResourceHandle {
  std::uint64_t resource_id = 0;
  std::uint64_t allocator_owner_id = 0;
  std::uint64_t lifetime_generation = 0;
  MemoryDomain memory_domain = MemoryDomain::invalid;
  ResourceLifetime lifetime = ResourceLifetime::invalid;
  SynchronizationState synchronization_state = SynchronizationState::invalid;
  ExecutionDomainId execution_domain{};
  std::uint64_t size_bytes = 0;
  std::uint32_t alignment_bytes = 0;
  bool mutable_resource = false;
  bool aliasable = false;
  // Adapter-private payload for the underlying host pointer, SDK object,
  // buffer identifier, or registry key. Neutral consumers must never infer
  // interoperability from these values; execution_domain and ownership
  // evidence remain authoritative.
  std::array<std::uint64_t, 4> adapter_opaque{};
  void *destruction_owner = nullptr;
  ResourceDestroyFn destroy = nullptr;
};

enum class StageIoKind : std::uint8_t {
  image = 1,
  tensor,
};

struct StageIoDescriptor {
  StageIoKind kind = StageIoKind::image;
  ImageDescriptor image{};
  TensorDescriptor tensor{};
};

struct ModelArtifactIdentity {
  // Stable hashes supplied by setup-time model validation. The frame path
  // never parses manifests, scans packs, or hashes artifacts.
  std::array<std::uint8_t, 32> model_id_hash{};
  std::array<std::uint8_t, 32> artifact_set_hash{};
  std::uint64_t artifact_generation = 0;
};

enum class FailureAction : std::uint8_t {
  fail_closed = 1,
  fail_open_preserve_input,
  fail_open_preserve_last_valid,
};

enum class CircuitBreakerAction : std::uint8_t {
  disable_stage = 1,
  quarantine_program,
  request_rebuild,
};

struct CircuitBreakerPolicy {
  std::uint32_t consecutive_failure_limit = 1;
  std::uint32_t cooldown_frames = 0;
  CircuitBreakerAction action = CircuitBreakerAction::disable_stage;
};

struct PreparedStage {
  std::uint32_t stage_id = 0;
  SemanticEffectId semantic = SemanticEffectId::invalid;
  SemanticEffectProviderId provider = SemanticEffectProviderId::invalid;
  BackendId backend = BackendId::invalid;
  ExecutionDomainId execution_domain{};
  StageIoDescriptor input{};
  StageIoDescriptor output{};
  ModelArtifactIdentity model{};
  bool required = false;
  FailureAction failure_action = FailureAction::fail_closed;
  CircuitBreakerPolicy circuit_breaker{};
  std::uint64_t scratch_bytes = 0;
  // A stage is either a product semantic or a utility primitive, never both.
  // Appending this field preserves positional aggregate initialization.
  PrimitiveOperationId primitive = PrimitiveOperationId::invalid;
};

enum class FrameDropBehavior : std::uint8_t {
  latest_frame_wins = 1,
  preserve_order,
  reset_temporal_state,
};

enum class TemporalDiscontinuityAction : std::uint8_t {
  reset_and_continue = 1,
  fail_frame,
};

struct TemporalContract {
  bool temporal = false;
  bool requires_contiguous_capture_sequence = false;
  FrameDropBehavior frame_drop_behavior = FrameDropBehavior::latest_frame_wins;
  TemporalDiscontinuityAction discontinuity_action =
      TemporalDiscontinuityAction::reset_and_continue;
  std::uint32_t history_frames = 0;
};

struct ProgramGenerations {
  std::uint64_t configuration = 0;
  std::uint64_t geometry = 0;
  std::uint64_t model = 0;
  std::uint64_t provider = 0;
  std::uint64_t device = 0;
  std::uint64_t context = 0;

  friend bool operator==(const ProgramGenerations &,
                         const ProgramGenerations &) = default;
};

struct ProgramInvalidationKey {
  ProgramGenerations generations{};
  std::uint64_t canonical_plan_hash = 0;
  std::uint64_t model_artifact_hash = 0;
  std::uint64_t execution_domain_hash = 0;

  friend bool operator==(const ProgramInvalidationKey &,
                         const ProgramInvalidationKey &) = default;
};

enum class CpuTailId : std::uint16_t {
  invalid = 0,
  output_format_conversion = 1,
  auto_frame_tracking = 2,
  provider_postprocess = 3,
  writer_copy = 4,
  cpu_primitive_mirror = 5,
  // Setup-request placeholder only. A backend setup source must refine this
  // before the plan may be validated or executed.
  backend_resolved_output = 6,
};
static_assert(static_cast<std::uint16_t>(CpuTailId::writer_copy) == 4);
static_assert(static_cast<std::uint16_t>(CpuTailId::cpu_primitive_mirror) == 5);
static_assert(static_cast<std::uint16_t>(
                  CpuTailId::backend_resolved_output) == 6);

struct DeclaredCpuTail {
  CpuTailId id = CpuTailId::invalid;
  std::uint32_t after_stage_id = 0;
  std::uint64_t scratch_bytes = 0;
  bool required = true;
  enum class Placement : std::uint8_t {
    after_stage = 1,
    before_stage,
  };
  Placement placement = Placement::after_stage;
  std::uint32_t before_stage_id = 0;
  // Zero preserves legacy aggregate initialization. A non-zero value is the
  // exact boundary position in [0, stage_count]: before stage i is i, after
  // stage i is i + 1.
  std::uint32_t boundary_index = 0;
};

enum class SetupFailurePolicy : std::uint8_t {
  retain_old_program = 1,
  stop_pipeline,
};

struct PreparedProgram {
  std::uint32_t contract_version = kContractVersion;
  std::uint64_t program_id = 0;
  std::uint64_t owner_adapter_instance_id = 0;
  ExecutionDomainId execution_domain{};
  // Host transport boundaries are distinct from resident stage I/O. They
  // describe the borrowed CPU-visible frames bound by BindHostFrame, while
  // PreparedStage descriptors continue to describe adapter-resident storage.
  // A format or coherence transition between either boundary and the adjacent
  // resident stage is therefore explicit in the immutable prepared contract.
  ImageDescriptor host_input{};
  ImageDescriptor host_output{};
  ProgramInvalidationKey invalidation_key{};
  TemporalContract temporal{};
  ResourceBudget budget{};
  SetupFailurePolicy setup_failure_policy =
      SetupFailurePolicy::retain_old_program;
  std::array<PreparedStage, kMaxPreparedStages> stages{};
  std::size_t stage_count = 0;
  std::array<DeclaredCpuTail, kMaxCpuTails> cpu_tails{};
  std::size_t cpu_tail_count = 0;
  BackendId requested_backend = BackendId::invalid;
  bool strict_backend_selection = true;
  SemanticEffectProviderId requested_provider =
      SemanticEffectProviderId::invalid;
  bool strict_provider_selection = false;
  ExecutionDomainId requested_execution_domain{};
  bool strict_execution_domain = false;
  bool requires_upload = true;
  bool requires_final_readback = true;
  // Setup-proven bounded metadata only. These allowances never permit host
  // image/model tensors or intermediate frame uploads.
  std::uint32_t maximum_metadata_readbacks_per_frame = 0;
  std::uint32_t maximum_metadata_cpu_tails_per_frame = 0;
};

struct CapabilityEvidenceRecord {
  CapabilityEvidenceCode code = CapabilityEvidenceCode::invalid;
  StatusId status = StatusId::invalid;
  ReasonId reason = ReasonId::none;
  SemanticEffectId semantic = SemanticEffectId::invalid;
  BackendId backend = BackendId::invalid;
  bool satisfied = false;
};

struct ReadinessSnapshot {
  StatusId status = StatusId::invalid;
  ReasonId reason = ReasonId::none;
  BackendId requested_backend = BackendId::invalid;
  BackendId selected_backend = BackendId::invalid;
  SemanticEffectProviderId provider = SemanticEffectProviderId::invalid;
  ExecutionDomainId execution_domain{};
  std::uint32_t ready_stage_mask = 0;
  std::uint32_t blocked_stage_mask = 0;
  std::uint32_t disabled_stage_mask = 0;
  std::array<CapabilityEvidenceRecord, kMaxCapabilityEvidenceRecords>
      evidence{};
  std::size_t evidence_count = 0;
  bool available = false;
  bool degraded = false;
  bool rebuild_requested = false;
};

struct PreparedProgramHandle {
  std::uint64_t handle_id = 0;
  std::uint64_t program_id = 0;
  std::uint64_t adapter_instance_id = 0;
  ProgramInvalidationKey invalidation_key{};
  ExecutionDomainId execution_domain{};
  ResourceBudget reserved_budget{};
  std::array<std::uint64_t, 2> adapter_opaque{};
};

enum class StageExecutionStatus : std::uint8_t {
  not_run = 0,
  succeeded,
  optional_failed_open,
  required_failed,
  disabled_preserved,
};

struct StageExecutionResult {
  std::uint32_t stage_id = 0;
  SemanticEffectId semantic = SemanticEffectId::invalid;
  StageExecutionStatus status = StageExecutionStatus::not_run;
  ReasonId reason = ReasonId::none;
  bool valid_output_preserved = false;
  std::uint64_t elapsed_ns = 0;
  PrimitiveOperationId primitive = PrimitiveOperationId::invalid;
};

struct TransferCounters {
  std::uint64_t uploads = 0;
  std::uint64_t final_readbacks = 0;
  std::uint64_t intermediate_readbacks = 0;
  std::uint64_t intermediate_uploads = 0;
  std::uint64_t synchronizations = 0;
  std::uint64_t cpu_tail_boundaries = 0;
};

struct LatencyHistogram {
  std::array<std::uint64_t, kMaxLatencyHistogramBuckets> bucket_upper_ns{};
  std::array<std::uint64_t, kMaxLatencyHistogramBuckets> bucket_counts{};
  std::uint64_t samples = 0;
  std::uint64_t total_ns = 0;
  std::uint64_t maximum_ns = 0;
  std::uint64_t p95_ns = 0;
  std::uint64_t p99_ns = 0;
};

struct ExecutionTelemetry {
  BackendId requested_backend = BackendId::invalid;
  BackendId selected_backend = BackendId::invalid;
  BackendId active_backend = BackendId::invalid;
  SemanticEffectProviderId active_provider = SemanticEffectProviderId::invalid;
  std::uint64_t execution_attempts = 0;
  std::uint64_t execution_successes = 0;
  std::uint64_t committed_output_frames = 0;
  std::uint64_t active_backend_frames = 0;
  std::uint64_t fallback_frames = 0;
  std::uint64_t optional_fail_open_stages = 0;
  std::uint64_t required_stage_failures = 0;
  std::uint64_t temporal_resets = 0;
  std::uint64_t device_losses = 0;
  std::uint64_t setup_attempts = 0;
  std::uint64_t setup_successes = 0;
  std::uint64_t setup_rollbacks = 0;
  std::uint64_t setup_retention_rejections = 0;
  std::uint64_t circuit_breaker_opens = 0;
  std::uint64_t disabled_stage_frames = 0;
  std::uint64_t rebuild_requests = 0;
  std::uint64_t frame_drop_events = 0;
  std::uint64_t dropped_capture_frames = 0;
  std::uint64_t preserve_order_failures = 0;
  std::uint64_t latest_frame_wins_events = 0;
  std::uint32_t disabled_stage_mask = 0;
  StatusId runtime_status = StatusId::invalid;
  ReasonId runtime_reason = ReasonId::none;
  TransferCounters transfers{};
  LatencyHistogram frame_latency{};
  std::array<LatencyHistogram, kMaxPreparedStages> stage_latency{};
  std::array<std::uint64_t, kMaxPreparedStages> committed_stage_successes{};
};

struct CompletionToken {
  std::uint64_t token_id = 0;
  std::uint64_t program_id = 0;
  std::uint64_t adapter_instance_id = 0;
  std::uint64_t capture_sequence = 0;
  bool complete = false;
};

struct ExecuteRequest {
  ProgramInvalidationKey invalidation_key{};
  ExecutionDomainId execution_domain{};
  std::uint64_t capture_sequence = 0;
  ResourceHandle input{};
  // Filled by PreparedProgramSession from its fixed circuit-breaker state.
  // Callers must pass zero; adapters must return disabled_preserved for every
  // set optional-stage bit without executing that stage.
  std::uint32_t disabled_stage_mask = 0;
};

struct ExecuteResult {
  StatusId status = StatusId::failed;
  ReasonId reason = ReasonId::none;
  CompletionToken completion{};
  ResourceHandle output{};
  bool output_valid = false;
  std::array<StageExecutionResult, kMaxPreparedStages> stages{};
  std::size_t stage_count = 0;
  TransferCounters transfers{};
  std::uint64_t frame_elapsed_ns = 0;
  std::uint64_t dropped_capture_frames = 0;
  bool temporal_state_reset = false;
};

struct OutputCommit {
  CompletionToken completion{};
  bool output_write_succeeded = false;
};

// This snapshot is distinct from cumulative telemetry. It is replaced only
// after the caller confirms that the completed output was published
// successfully, so status consumers cannot infer live activity from setup,
// resolution, or an execution that never reached the writer.
struct CommittedStageActivity {
  std::uint32_t stage_id = 0;
  SemanticEffectId semantic = SemanticEffectId::invalid;
  PrimitiveOperationId primitive = PrimitiveOperationId::invalid;
  BackendId backend = BackendId::invalid;
  SemanticEffectProviderId provider = SemanticEffectProviderId::invalid;
  StageExecutionStatus status = StageExecutionStatus::not_run;
  ReasonId reason = ReasonId::none;
};

struct CommittedFrameSnapshot {
  bool valid = false;
  std::uint64_t program_id = 0;
  ProgramInvalidationKey invalidation_key{};
  ExecutionDomainId execution_domain{};
  std::uint64_t capture_sequence = 0;
  std::uint64_t output_commit_sequence = 0;
  std::array<CommittedStageActivity, kMaxPreparedStages> stages{};
  std::size_t stage_count = 0;
};

struct HostPlaneBinding {
  const std::byte *read_data = nullptr;
  std::byte *write_data = nullptr;
  std::uint64_t span_bytes = 0;
  std::uint64_t row_stride_bytes = 0;
};

struct HostFrameBinding {
  ImageDescriptor input_descriptor{};
  ImageDescriptor output_descriptor{};
  std::array<HostPlaneBinding, kMaxImagePlanes> input_planes{};
  std::array<HostPlaneBinding, kMaxImagePlanes> output_planes{};
  std::size_t input_plane_count = 0;
  std::size_t output_plane_count = 0;
  std::uint64_t capture_sequence = 0;
  OwnershipModel ownership = OwnershipModel::borrowed_until_completion;
  ResourceLifetime lifetime = ResourceLifetime::frame_borrowed;
};

struct HostFrameTokens {
  ResourceHandle input{};
  ResourceHandle output{};
};

struct HostFrameOutputResolution {
  ImageDescriptor descriptor{};
  std::array<HostPlaneBinding, kMaxImagePlanes> planes{};
  std::size_t plane_count = 0;
  std::uint64_t capture_sequence = 0;
  bool output_ready = false;
};

[[nodiscard]] bool
ValidateResourceHandle(const ResourceHandle &resource,
                       const ExecutionDomainId &expected_domain,
                       std::uint64_t expected_lifetime_generation,
                       const ResourceBudget &budget) noexcept;
[[nodiscard]] bool ValidatePreparedProgram(const PreparedProgram &program,
                                           ReasonId *reason) noexcept;
[[nodiscard]] bool ValidateHostFrameBinding(const HostFrameBinding &binding,
                                            const PreparedProgram &program,
                                            ReasonId *reason) noexcept;

class IResidentAdapter {
public:
  virtual ~IResidentAdapter() = default;

  [[nodiscard]] virtual std::uint64_t adapter_instance_id() const noexcept = 0;
  [[nodiscard]] virtual BackendId backend_id() const noexcept = 0;

  // Prepare owns all session/kernel/buffer/scratch construction. On failure it
  // must release its candidate resources and leave any old handle untouched.
  virtual StatusId Prepare(const PreparedProgram &program,
                           PreparedProgramHandle *candidate,
                           ReadinessSnapshot *readiness,
                           ReasonId *reason) noexcept = 0;
  virtual ExecuteResult Execute(const PreparedProgramHandle &program,
                                const ExecuteRequest &request) noexcept = 0;
  virtual StatusId ResetTemporalState(const PreparedProgramHandle &program,
                                      ReasonId reason) noexcept = 0;
  virtual StatusId BindHostFrame(const PreparedProgramHandle &,
                                 const HostFrameBinding &, HostFrameTokens *,
                                 ReasonId *reason) noexcept {
    if (reason)
      *reason = ReasonId::host_frame_binding_unsupported;
    return StatusId::blocked;
  }
  virtual StatusId ResolveHostFrameOutput(const PreparedProgramHandle &,
                                          const CompletionToken &,
                                          const ResourceHandle &,
                                          HostFrameOutputResolution *,
                                          ReasonId *reason) noexcept {
    if (reason)
      *reason = ReasonId::host_frame_binding_unsupported;
    return StatusId::blocked;
  }
  virtual void DestroyPrepared(PreparedProgramHandle *program) noexcept = 0;
  virtual void Shutdown() noexcept = 0;
};

class PreparedProgramSession {
public:
  // The adapter must outlive the session. Construct it first and destroy it
  // after the session so prepared resources can be released through it.
  PreparedProgramSession() = default;
  ~PreparedProgramSession();
  PreparedProgramSession(const PreparedProgramSession &) = delete;
  PreparedProgramSession &operator=(const PreparedProgramSession &) = delete;

  [[nodiscard]] StatusId Configure(const PreparedProgram &program,
                                   IResidentAdapter &adapter,
                                   ReadinessSnapshot *readiness,
                                   ReasonId *reason) noexcept;
  [[nodiscard]] ExecuteResult Execute(const ExecuteRequest &request) noexcept;
  [[nodiscard]] StatusId CommitOutput(const OutputCommit &commit,
                                      ReasonId *reason) noexcept;
  [[nodiscard]] StatusId BindHostFrame(const HostFrameBinding &binding,
                                       HostFrameTokens *tokens,
                                       ReasonId *reason) noexcept;
  [[nodiscard]] StatusId ResolveHostFrameOutput(
      const CompletionToken &completion, const ResourceHandle &output,
      HostFrameOutputResolution *resolution, ReasonId *reason) noexcept;
  [[nodiscard]] StatusId ResetTemporalState(ReasonId reason) noexcept;
  void Invalidate(ReasonId reason) noexcept;
  void Shutdown() noexcept;

  [[nodiscard]] bool prepared() const noexcept { return prepared_; }
  [[nodiscard]] bool quarantined() const noexcept { return quarantined_; }
  [[nodiscard]] bool stopped() const noexcept { return stopped_; }
  [[nodiscard]] bool rebuild_requested() const noexcept {
    return rebuild_requested_;
  }
  [[nodiscard]] std::uint32_t disabled_stage_mask() const noexcept {
    return disabled_stage_mask_;
  }
  [[nodiscard]] const ReadinessSnapshot &readiness() const noexcept {
    return readiness_;
  }
  [[nodiscard]] const PreparedProgram &program() const noexcept {
    return program_;
  }
  [[nodiscard]] const ExecutionTelemetry &telemetry() const noexcept {
    return telemetry_;
  }
  [[nodiscard]] const CommittedFrameSnapshot &
  latest_committed_frame() const noexcept {
    return latest_committed_frame_;
  }

private:
  void DestroyActive() noexcept;
  void AccumulateExecuteTelemetry(const ExecuteResult &result) noexcept;
  [[nodiscard]] bool
  MayRetainOldProgram(const PreparedProgram &candidate) const noexcept;
  void RejectOldProgramRetention(ReasonId reason) noexcept;
  void UpdateCircuitBreakers(const ExecuteResult &result,
                             std::uint64_t capture_sequence) noexcept;

  IResidentAdapter *adapter_ = nullptr;
  PreparedProgram program_{};
  PreparedProgramHandle handle_{};
  ExecutionTelemetry telemetry_{};
  CommittedFrameSnapshot latest_committed_frame_{};
  CompletionToken pending_completion_{};
  ReadinessSnapshot readiness_{};
  ReadinessSnapshot configured_readiness_{};
  HostFrameBinding pending_host_binding_{};
  HostFrameTokens pending_host_tokens_{};
  std::array<StageExecutionResult, kMaxPreparedStages> pending_stages_{};
  std::size_t pending_stage_count_ = 0;
  std::uint64_t last_capture_sequence_ = 0;
  std::array<std::uint32_t, kMaxPreparedStages> consecutive_failures_{};
  std::array<std::uint64_t, kMaxPreparedStages> retry_after_sequence_{};
  std::uint32_t disabled_stage_mask_ = 0;
  std::uint32_t permanent_disabled_stage_mask_ = 0;
  bool have_capture_sequence_ = false;
  bool prepared_ = false;
  bool quarantined_ = false;
  bool stopped_ = false;
  bool completion_pending_ = false;
  bool host_binding_pending_ = false;
  bool rebuild_requested_ = false;
};

} // namespace studiocast::hal
