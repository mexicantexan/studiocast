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

enum class MemoryDomain : std::uint16_t {
  invalid = 0,
  host,
  upload_staging,
  device_local,
  readback_staging,
  provider_owned,
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
  output_format_conversion,
  auto_frame_tracking,
  provider_postprocess,
  writer_copy,
};

struct DeclaredCpuTail {
  CpuTailId id = CpuTailId::invalid;
  std::uint32_t after_stage_id = 0;
  std::uint64_t scratch_bytes = 0;
  bool required = true;
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
  bool requires_upload = true;
  bool requires_final_readback = true;
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
  bool available = false;
  bool degraded = false;
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
};

struct StageExecutionResult {
  std::uint32_t stage_id = 0;
  SemanticEffectId semantic = SemanticEffectId::invalid;
  StageExecutionStatus status = StageExecutionStatus::not_run;
  ReasonId reason = ReasonId::none;
  bool valid_output_preserved = false;
  std::uint64_t elapsed_ns = 0;
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
};

struct OutputCommit {
  CompletionToken completion{};
  bool output_write_succeeded = false;
};

[[nodiscard]] bool
ValidateResourceHandle(const ResourceHandle &resource,
                       const ExecutionDomainId &expected_domain,
                       std::uint64_t expected_lifetime_generation,
                       const ResourceBudget &budget) noexcept;
[[nodiscard]] bool ValidatePreparedProgram(const PreparedProgram &program,
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
  void Invalidate(ReasonId reason) noexcept;
  void Shutdown() noexcept;

  [[nodiscard]] bool prepared() const noexcept { return prepared_; }
  [[nodiscard]] bool quarantined() const noexcept { return quarantined_; }
  [[nodiscard]] bool stopped() const noexcept { return stopped_; }
  [[nodiscard]] const PreparedProgram &program() const noexcept {
    return program_;
  }
  [[nodiscard]] const ExecutionTelemetry &telemetry() const noexcept {
    return telemetry_;
  }

private:
  void DestroyActive() noexcept;
  void AccumulateExecuteTelemetry(const ExecuteResult &result) noexcept;

  IResidentAdapter *adapter_ = nullptr;
  PreparedProgram program_{};
  PreparedProgramHandle handle_{};
  ExecutionTelemetry telemetry_{};
  CompletionToken pending_completion_{};
  std::array<StageExecutionResult, kMaxPreparedStages> pending_stages_{};
  std::size_t pending_stage_count_ = 0;
  std::uint64_t last_capture_sequence_ = 0;
  bool have_capture_sequence_ = false;
  bool prepared_ = false;
  bool quarantined_ = false;
  bool stopped_ = false;
  bool completion_pending_ = false;
};

} // namespace studiocast::hal
