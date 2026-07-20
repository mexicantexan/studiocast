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

// Options for creating an ONNX Runtime session.
struct OrtSessionOptions {
  // If true, attempt to use CUDA EP and fall back to CPU EP if CUDA EP is not
  // available.
  bool prefer_cuda = true;

  // CUDA device id to use when CUDA EP is available.
  int cuda_device_id = 0;

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
};

} // namespace studiocast::onnx
