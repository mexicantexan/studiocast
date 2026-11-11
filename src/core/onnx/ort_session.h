#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace studiocast::onnx {

// Best-effort ONNX Runtime runtime information.
struct OrtRuntimeInfo {
  std::string version;

  // Providers advertised by ONNX Runtime. This does not prove that the
  // provider shared library and its CUDA/TensorRT dependencies can be loaded.
  std::vector<std::string> providers;
  bool cuda_provider_present = false;
  bool tensorrt_provider_present = false;
  bool cpu_provider_present = false;
  bool cuda_ep_v2_build = false;
  std::string library_path;
  std::vector<std::string> warnings;
};

// Returns true when the headers used for this build expose the TensorRT EP V2
// provider-options API. Runtime availability still depends on the ORT build and
// installed provider libraries.
bool OrtBuildHasTensorRtEpV2();

// Default TensorRT engine cache path for a CUDA device.
std::filesystem::path DefaultTensorRtCachePath(int cuda_device_id);

enum class OrtProviderPolicy : std::uint8_t {
  // Existing behavior: prefer requested accelerator providers but preserve
  // historical CUDA->CPU and TensorRT->CUDA/CPU compatibility fallbacks.
  compatibility = 0,

  // Prepared Open Video hardware execution: require exactly the CUDA EP and
  // disable ORT CPU graph/provider fallback. TensorRT is a distinct provider
  // and is rejected by this policy.
  strict_cuda_only = 1,
};

enum class OrtExecutionProviderId : std::uint8_t {
  invalid = 0,
  cpu = 1,
  cuda = 2,
  tensorrt = 3,
};

struct OrtPreparedProviderEvidence {
  static constexpr std::uint32_t kSchemaVersion = 1;

  std::uint32_t schema_version = kSchemaVersion;
  OrtExecutionProviderId provider = OrtExecutionProviderId::invalid;
  std::int32_t cuda_device_id = -1;
  std::uint64_t cuda_context_generation = 0;
  std::uintptr_t requested_stream_identity = 0;
  bool user_stream_bound = false;
  bool explicit_stream_synchronization = false;
  bool cpu_ep_fallback_disabled = false;

  bool ValidStrictCuda() const noexcept;
  bool operator==(const OrtPreparedProviderEvidence &other) const noexcept {
    return schema_version == other.schema_version &&
           provider == other.provider &&
           cuda_device_id == other.cuda_device_id &&
           cuda_context_generation == other.cuda_context_generation &&
           requested_stream_identity == other.requested_stream_identity &&
           user_stream_bound == other.user_stream_bound &&
           explicit_stream_synchronization ==
               other.explicit_stream_synchronization &&
           cpu_ep_fallback_disabled == other.cpu_ep_fallback_disabled;
  }
};

bool OrtPreparedProviderEvidenceMatches(
    const OrtPreparedProviderEvidence &prepared, int cuda_device_id,
    std::uint64_t cuda_context_generation,
    std::uintptr_t requested_stream_identity) noexcept;

enum class OrtPreparedRunStatus : std::uint16_t {
  success = 0,
  not_prepared = 1,
  invalid_input_output_contract = 2,
  foreign_provider_evidence = 3,
  stale_session_generation = 4,
  wrong_device = 5,
  stale_context_generation = 6,
  stale_stream = 7,
  stale_binding = 8,
  provider_unavailable = 9,
  provider_mismatch = 10,
  cpu_fallback_forbidden = 11,
  tensorrt_forbidden = 12,
  resource_exhausted = 13,
  inference_failure = 14,
  provider_failure_quarantine = 15,
  cancelled_or_shutdown = 16,
  setup_resource_mutation = 17,
};

enum class OrtPreparedRunQuarantine : std::uint8_t {
  none = 0,
  session = 1,
  provider = 2,
};

// Stable mirror of OrtErrorCode values used at the catch boundary. Keeping the
// values local makes the public header independent of ONNX Runtime headers.
enum class OrtStableErrorCode : std::int32_t {
  none = 0,
  fail = 1,
  invalid_argument = 2,
  no_such_file = 3,
  no_model = 4,
  engine_error = 5,
  runtime_exception = 6,
  invalid_protobuf = 7,
  model_loaded = 8,
  not_implemented = 9,
  invalid_graph = 10,
  execution_provider_failure = 11,
  model_load_cancelled = 12,
  model_requires_compilation = 13,
  not_found = 14,
  allocation_failure = 1000,
  unknown_exception = 1001,
};

struct OrtPreparedCudaRunEvidence {
  static constexpr std::uint32_t kSchemaVersion = 1;

  std::uint32_t schema_version = kSchemaVersion;
  OrtPreparedProviderEvidence provider{};
  std::uint64_t session_generation = 0;

  bool Valid() const noexcept {
    return schema_version == kSchemaVersion && session_generation != 0 &&
           provider.schema_version ==
               OrtPreparedProviderEvidence::kSchemaVersion &&
           provider.provider != OrtExecutionProviderId::invalid;
  }
};

struct OrtPreparedRunResult {
  static constexpr std::uint32_t kSchemaVersion = 1;

  std::uint32_t schema_version = kSchemaVersion;
  OrtPreparedRunStatus status = OrtPreparedRunStatus::not_prepared;
  OrtPreparedRunQuarantine quarantine = OrtPreparedRunQuarantine::none;
  OrtStableErrorCode error_code = OrtStableErrorCode::none;
  std::uint64_t session_generation = 0;
  std::uint64_t attempt_sequence = 0;
  std::uint64_t completed_runs = 0;

  bool ok() const noexcept {
    return schema_version == kSchemaVersion &&
           status == OrtPreparedRunStatus::success;
  }
};

// Options for creating an ONNX Runtime session.
struct OrtSessionOptions {
  struct FreeDimensionOverride {
    std::string name;
    int64_t extent = 0;
  };

  OrtProviderPolicy provider_policy = OrtProviderPolicy::compatibility;

  // Setup-time specialization for symbolic model dimensions. Callers must
  // provide names from a reviewed model contract; ORT ignores unknown names.
  std::vector<FreeDimensionOverride> free_dimension_overrides;

  // If true, attempt to use CUDA EP and fall back to CPU EP if CUDA EP is not
  // available.
  bool prefer_cuda = true;

  // CUDA device id to use when CUDA EP is available.
  int cuda_device_id = 0;

  // Caller-owned CUDA context generation. Strict prepared execution requires
  // a non-zero value and retains it as typed evidence; ORT itself does not own
  // the CUDA primary-context lifecycle.
  std::uint64_t cuda_context_generation = 0;

  // Optional CUDA EP V2 provider options. Zero/empty means "use ORT default".
  std::uint64_t cuda_gpu_mem_limit = 0;
  std::string cuda_arena_extend_strategy;
  std::string cuda_cudnn_conv_algo_search;
  bool cuda_do_copy_in_default_stream = true;

  // Optional compute stream (CUDA stream) to use when supported by the CUDA EP.
  //
  // When non-null and the build exposes CUDA EP V2 provider options
  // (STUDIOCAST_ORT_HAS_CUDA_EP_V2=1), ORT will enqueue its compute on this
  // stream.
  //
  // When null, ORT may use internal streams and the caller may need to
  // synchronize around calls that produce/consume GPU buffers.
  void *user_compute_stream = nullptr;

  // If true, attempt to append TensorRT EP before CUDA EP. This is currently
  // intended for Open CUDA matting only; callers must opt in explicitly.
  bool enable_tensorrt = false;

  // If TensorRT is requested, append CUDA EP after TensorRT so unsupported
  // subgraphs can stay on GPU. CPU remains ORT's final fallback in code paths
  // that already permit CPU sessions.
  bool tensorrt_enable_cuda_fallback = true;

  // TensorRT provider defaults for the Open CUDA matting path.
  std::uint64_t tensorrt_max_workspace_size = 2ull * 1024ull * 1024ull * 1024ull;
  bool tensorrt_fp16_enable = true;
  bool tensorrt_engine_cache_enable = true;
  std::filesystem::path tensorrt_engine_cache_path;
  int tensorrt_builder_optimization_level = 3;
};

// Best-effort model I/O description extracted from the session.
struct OrtSessionInfo {
  bool using_tensorrt = false;
  bool using_cuda = false;

  // Providers advertised by ORT at session-creation time. These are separated
  // from provider append/use results because ORT can advertise CUDA/TensorRT
  // while the provider library or runtime dependencies fail to load.
  std::vector<std::string> advertised_providers;
  bool cuda_provider_advertised = false;
  bool tensorrt_provider_advertised = false;
  bool cpu_provider_advertised = false;

  bool tensorrt_provider_appended = false;
  bool cuda_provider_appended = false;
  bool tensorrt_provider_usable = false;
  bool cuda_provider_usable = false;
  bool cpu_provider_usable = false;

  bool tensorrt_session_create_failed_fell_back_to_cuda = false;
  bool cuda_session_create_failed_fell_back_to_cpu = false;

  // If true, the session is using CUDA EP but is not guaranteed to run on the
  // caller's stream (e.g., user_compute_stream unavailable). Callers
  // integrating with an explicit stream must synchronize for correctness.
  bool cuda_needs_stream_sync = false;
  int cuda_device_id = 0;
  bool cuda_user_compute_stream_requested = false;
  bool cuda_user_compute_stream_bound = false;
  std::uintptr_t cuda_user_compute_stream_identity = 0;
  std::uint64_t cuda_context_generation = 0;
  bool cpu_ep_fallback_disabled = false;
  bool strict_provider_validation_passed = false;
  OrtPreparedProviderEvidence prepared_provider_evidence{};

  // Provider/status diagnostics for tools and engine JSON.
  std::string active_provider;
  std::string appended_provider;
  std::vector<std::string> appended_providers;
  std::string tensorrt_status;
  std::filesystem::path tensorrt_engine_cache_path;

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // Human-friendly strings like: "tensor(float32) shape=[1, 3, 320, 320]".
  std::vector<std::string> input_descriptions;
  std::vector<std::string> output_descriptions;

  // Structured tensor metadata.
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<int> input_elem_types;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int> output_elem_types;

  // Non-fatal warnings collected during session creation.
  std::vector<std::string> warnings;
};

namespace internal {

struct OrtPreparedRunContract {
  bool session_present = false;
  bool binding_present = false;
  bool binding_contract_matches = false;
  bool shutdown_requested = false;
  OrtPreparedCudaRunEvidence active{};
  OrtPreparedCudaRunEvidence requested{};
  bool strict_cuda_only = false;
};

OrtPreparedRunResult
ClassifyOrtPreparedRunContract(const OrtPreparedRunContract &contract) noexcept;

OrtPreparedRunResult
ClassifyOrtPreparedRunFailure(OrtStableErrorCode code,
                              bool exact_cuda_provider) noexcept;

// Internal seam for session-creation-time provider planning. It is intentionally
// limited to provider append/session-create outcomes so tests can cover fallback
// behavior without loading real CUDA/TensorRT ORT providers.
struct OrtProviderAppendResult {
  bool appended = false;
  bool needs_stream_sync = false;
  std::filesystem::path cache_path;
  std::string status;
  std::vector<std::string> warnings;
};

using OrtProviderAppendFn = OrtProviderAppendResult (*)(
    void *context, const OrtSessionOptions &opts);

struct OrtProviderAppendHooks {
  void *context = nullptr;
  OrtProviderAppendFn append_tensorrt = nullptr;
  OrtProviderAppendFn append_cuda = nullptr;
};

enum class OrtSessionCreateAttempt {
  Initial,
  TensorRtDisabled,
  CpuOnly,
};

struct OrtSessionCreateResult {
  bool created = false;
  std::string error;
};

using OrtSessionCreateFn = OrtSessionCreateResult (*)(
    void *context, OrtSessionCreateAttempt attempt,
    const OrtSessionOptions &opts, OrtSessionInfo *info_out);

struct OrtSessionCreateHooks {
  void *context = nullptr;
  OrtSessionCreateFn create_session = nullptr;
};

struct OrtSessionCreatePlanResult {
  bool created = false;
  OrtSessionInfo info;
  std::string error;
};

OrtSessionInfo PlanOrtProviderAttempt(const OrtRuntimeInfo &runtime,
                                      const OrtSessionOptions &opts,
                                      bool tensorrt_ep_v2_build,
                                      const OrtProviderAppendHooks &hooks);

OrtSessionCreatePlanResult
CreateOrtSessionWithProviderFallbacks(const OrtSessionOptions &opts,
                                      const OrtSessionCreateHooks &hooks);

} // namespace internal

// Returns true if the ORT exception text looks like CUDA VRAM exhaustion.
bool OrtErrorLooksLikeVramOom(const std::string &ort_msg);

// Convert raw ORT errors into actionable, user-facing diagnostics.
std::string HumanizeOrtError(const std::string &ort_msg,
                             const std::filesystem::path &model_path);

// Canonical ONNX Runtime session wrapper for StudioCast.
//
// This is shared across "open_video" and "open_cuda" runtimes so the app can:
//   - Prefer Maxine when available
//   - Otherwise run effects on the GPU (one CPU->GPU upload, one GPU->CPU
//   download)
//   - Fall back to CPU if neither Maxine nor CUDA EP is available
class OrtSession {
public:
  static constexpr std::size_t kPreparedBindingSlots = 2;

  struct PreparedRunStats {
    std::uint64_t session_generation = 0;
    std::uint64_t execute_attempts = 0;
    std::uint64_t execute_successes = 0;
    std::uint64_t execute_failures = 0;
    std::uint64_t quarantine_events = 0;
    std::uint64_t runs = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t binding_rebuilds = 0;
    std::uint64_t tensor_wrapper_constructions = 0;
    std::uint64_t io_binding_constructions = 0;
    // Allocation requests made by StudioCast-owned prepared-binding
    // containers. This intentionally excludes opaque ORT allocator activity.
    std::uint64_t application_binding_allocation_requests = 0;
  };
  static OrtRuntimeInfo QueryRuntimeInfo();

  // Create an ORT session for the given model.
  // Returns nullptr on failure and fills `error`.
  static std::unique_ptr<OrtSession>
  Create(const std::filesystem::path &model_path, const OrtSessionOptions &opts,
         OrtSessionInfo *info_out, std::string *error);

  ~OrtSession();

  OrtSession(const OrtSession &) = delete;
  OrtSession &operator=(const OrtSession &) = delete;

  const OrtSessionInfo &info() const;

  // Pre-size reusable run scratch buffers during setup. This is optional, but
  // realtime callers should use it once binding counts are known.
  void ReserveRunScratch(std::size_t input_count, std::size_t output_count);

  struct RunInput {
    const char *name = nullptr;
    const float *data = nullptr;
    std::size_t num_floats = 0;
    const int64_t *shape = nullptr;
    std::size_t shape_rank = 0;
  };

  struct RunOutput {
    const char *name = nullptr;
    float *data = nullptr;
    std::size_t num_floats = 0;
    const int64_t *shape = nullptr;
    std::size_t shape_rank = 0;
  };

  // Run with pre-allocated float32 CPU tensors.
  bool RunCpu(const RunInput *inputs, std::size_t input_count,
              const RunOutput *outputs, std::size_t output_count,
              std::string *error);

  // Run using one of two bounded prepared binding slots. A slot retains the
  // external-buffer tensor wrappers until the session is destroyed or its
  // provider/session/name/pointer/size/shape contract changes. Streaming
  // callers with recurrent state should use one slot for each side of their
  // state ping-pong. Caller-owned names, shapes, and buffers must remain valid
  // until InvalidatePreparedBindings() is called or this session is destroyed.
  bool RunCpuPrepared(std::size_t binding_slot, const RunInput *inputs,
                      std::size_t input_count, const RunOutput *outputs,
                      std::size_t output_count, std::string *error);

  struct CudaBindingInput {
    const char *name = nullptr;
    const float *device_ptr = nullptr;
    std::size_t num_floats = 0;
    const int64_t *shape = nullptr;
    std::size_t shape_rank = 0;
  };

  struct CudaBindingOutput {
    const char *name = nullptr;
    float *device_ptr = nullptr;
    std::size_t num_floats = 0;
    const int64_t *shape = nullptr;
    std::size_t shape_rank = 0;
  };

  // Run with CUDA IoBinding (GPU inputs/outputs).
  //
  // If info().cuda_needs_stream_sync is true, ORT may not use the caller's
  // stream. In that case, callers integrating with an explicit stream must
  // synchronize before/after this call so producer/consumer kernels see
  // consistent data.
  //
  // This convenience API uses prepared slot 0. Caller-owned names, shapes, and
  // buffers must remain valid until InvalidatePreparedBindings() is called or
  // this session is destroyed.
  bool RunCudaIoBinding(const CudaBindingInput *inputs, std::size_t input_count,
                        const CudaBindingOutput *outputs,
                        std::size_t output_count, std::string *error);

  // CUDA equivalent of RunCpuPrepared. The Ort::IoBinding and bound OrtValue
  // objects remain intact across runs while the external device-buffer
  // contract is unchanged.
  bool RunCudaIoBindingPrepared(std::size_t binding_slot,
                                const CudaBindingInput *inputs,
                                std::size_t input_count,
                                const CudaBindingOutput *outputs,
                                std::size_t output_count, std::string *error);

  // Construct and bind all CUDA external-buffer wrappers during setup without
  // issuing inference. This operation is transactional: an existing prepared
  // slot remains usable if replacement construction fails.
  bool PrepareCudaIoBinding(std::size_t binding_slot,
                            const CudaBindingInput *inputs,
                            std::size_t input_count,
                            const CudaBindingOutput *outputs,
                            std::size_t output_count, std::string *error);

  // Execute only an exactly matching setup-prepared CUDA binding. This method
  // never constructs a binding, tensor wrapper, or StudioCast-owned container.
  bool ExecutePreparedCudaIoBinding(std::size_t binding_slot,
                                    const CudaBindingInput *inputs,
                                    std::size_t input_count,
                                    const CudaBindingOutput *outputs,
                                    std::size_t output_count,
                                    std::string *error);

  OrtPreparedRunResult ExecutePreparedCudaIoBindingTyped(
      std::size_t binding_slot, const CudaBindingInput *inputs,
      std::size_t input_count, const CudaBindingOutput *outputs,
      std::size_t output_count,
      const OrtPreparedCudaRunEvidence &expected,
      bool cancellation_requested = false,
      std::string *compatibility_error = nullptr) noexcept;

  OrtPreparedCudaRunEvidence prepared_cuda_run_evidence() const noexcept;
  void RequestShutdown() noexcept;

  // Release all retained CPU/CUDA tensor wrappers and IoBindings before a
  // caller replaces or frees their backing storage. Rebuilding a session for a
  // provider/model change naturally invalidates the old session's bindings.
  void InvalidatePreparedBindings();

  PreparedRunStats prepared_run_stats() const;

  // Returns true if the session has latched a fatal ORT failure (e.g., VRAM
  // OOM).
  bool HasLatchedFailure() const;

  // Returns the latched fatal error message (empty if none).
  const std::string &LatchedError() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit OrtSession(std::unique_ptr<Impl> impl);

  OrtPreparedRunResult ExecutePreparedCudaIoBindingTypedImpl(
      std::size_t binding_slot, const CudaBindingInput *inputs,
      std::size_t input_count, const CudaBindingOutput *outputs,
      std::size_t output_count,
      const OrtPreparedCudaRunEvidence &expected,
      bool cancellation_requested, std::string *compatibility_error) noexcept;
};

} // namespace studiocast::onnx
