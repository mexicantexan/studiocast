#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace studiocast::open_vulkan {

// Failure classes are intentionally runtime-neutral. A production ncnn
// adapter translates its errors into these values; the lifecycle then latches
// failures which must not be retried from the frame loop.
enum class VulkanMattingRuntimeFailure {
  none,
  unavailable,
  out_of_memory,
  device_lost,
  invalid_graph,
  invalid_shape,
  cpu_layer_rejected,
  device_identity_mismatch,
  context_generation_mismatch,
  residency_check_failed,
  cpu_readback_rejected,
  synchronization_failed,
  allocation_contract_failed,
  execution_failed,
};

const char *
VulkanMattingRuntimeFailureName(VulkanMattingRuntimeFailure failure);

struct VulkanMattingRuntimeResult {
  VulkanMattingRuntimeFailure failure = VulkanMattingRuntimeFailure::none;
  std::string detail;

  bool ok() const { return failure == VulkanMattingRuntimeFailure::none; }

  static VulkanMattingRuntimeResult Success();
  static VulkanMattingRuntimeResult Failure(VulkanMattingRuntimeFailure failure,
                                            std::string detail);
};

// All handles are opaque to the lifecycle. The production adapter is required
// to use this exact ownership domain, physical/logical device, and queue. It
// must not create a second Vulkan device behind StudioCast's back.
struct VulkanMattingDeviceContext {
  const void *ownership_domain = nullptr;
  std::uintptr_t physical_device = 0;
  std::uintptr_t logical_device = 0;
  std::uintptr_t queue = 0;
  std::uint32_t queue_family_index = 0;
  std::uint32_t vendor_id = 0;
  std::uint32_t device_id = 0;
  int selected_device_index = -1;
  std::string stable_device_id;
  std::uint64_t context_id = 0;
  std::uint64_t context_generation = 0;
  bool non_cpu_device_selected = false;
  bool compute_queue_available = false;
  bool context_healthy = false;
};

struct VulkanMattingGraphDescriptor {
  std::string param_path;
  std::string bin_path;
  std::string param_sha256;
  std::string bin_sha256;
  std::string input_blob;
  std::string output_blob;
  std::string converter_name;
  std::string converter_version;
  std::string precision;
  int input_n = 1;
  int input_c = 3;
  int input_h = 0;
  int input_w = 0;
  int output_n = 1;
  int output_c = 1;
  int output_h = 0;
  int output_w = 0;
};

struct VulkanMattingPersistentResources {
  std::size_t input_bytes = 0;
  std::size_t alpha_bytes = 0;
  bool allow_cpu_layers = false;
  bool require_device_residency = true;
};

struct VulkanMattingBufferBinding {
  const void *ownership_domain = nullptr;
  std::uintptr_t logical_device = 0;
  std::uintptr_t buffer = 0;
  std::size_t byte_offset = 0;
  std::size_t byte_size = 0;
  int n = 0;
  int c = 0;
  int h = 0;
  int w = 0;
  bool device_resident = false;
  bool device_local_memory = false;
  bool host_mapped = false;
  std::uint64_t context_id = 0;
  std::uint64_t context_generation = 0;
  // StudioCast and the adapter share one compute queue. These flags are
  // affirmative evidence that the previous producer completed and that the
  // adapter owns the declared read/write access for its synchronous Run().
  bool access_synchronized = false;
  bool runtime_queue_ownership = false;
};

// Evidence is queried after every setup gate and execution. Availability is
// granted only if the adapter proves all of these facts; a successful return
// code alone is insufficient.
struct VulkanMattingRuntimeEvidence {
  std::string runtime_name;
  bool adapter_available = false;
  bool production_adapter = false;
  bool runtime_created = false;
  bool graph_loaded = false;
  bool persistent_resources_allocated = false;
  bool warmup_complete = false;
  bool cpu_layers_used = false;
  bool device_identity_matches = false;
  bool input_device_resident = false;
  bool alpha_device_resident = false;
  bool output_device_resident = false;
  bool shared_device_imported = false;
  bool queue_ownership_explicit = false;
  bool synchronous_completion = false;
  bool bounded_reusable_allocations = false;
  std::uint64_t persistent_allocation_count = 0;
  std::uint64_t dynamic_allocation_count = 0;
  std::uint64_t cpu_readback_count = 0;
  std::uint64_t warmup_inference_count = 0;
  std::uint64_t inference_count = 0;
  std::uint64_t completion_count = 0;
  VulkanMattingDeviceContext active_device;
};

inline constexpr char kOpenVulkanMattingUnavailableReason[] =
    "open_vulkan_matting_unavailable";
inline constexpr char kOpenVulkanMattingBuildDisabledReason[] =
    "open_vulkan_matting_build_disabled";
inline constexpr char kOpenVulkanMattingAdapterUnavailableReason[] =
    "open_vulkan_matting_adapter_unavailable";
inline constexpr char kOpenVulkanMattingModelUnavailableReason[] =
    "open_vulkan_matting_model_unavailable";
inline constexpr char kOpenVulkanMattingModelInvalidReason[] =
    "open_vulkan_matting_model_invalid";
inline constexpr char kOpenVulkanMattingHardwareUnavailableReason[] =
    "open_vulkan_matting_hardware_unavailable";
inline constexpr char kOpenVulkanMattingDeviceMismatchReason[] =
    "open_vulkan_matting_device_mismatch";
inline constexpr char kOpenVulkanMattingCpuLayerReason[] =
    "open_vulkan_matting_cpu_layer_rejected";
inline constexpr char kOpenVulkanMattingCpuReadbackReason[] =
    "open_vulkan_matting_cpu_readback_rejected";
inline constexpr char kOpenVulkanMattingResidencyReason[] =
    "open_vulkan_matting_residency_unproven";
inline constexpr char kOpenVulkanMattingSynchronizationReason[] =
    "open_vulkan_matting_synchronization_unproven";
inline constexpr char kOpenVulkanMattingAllocationReason[] =
    "open_vulkan_matting_allocation_unbounded";
inline constexpr char kOpenVulkanMattingWarmupReason[] =
    "open_vulkan_matting_warmup_failed";
inline constexpr char kOpenVulkanMattingDeviceLostReason[] =
    "open_vulkan_matting_device_lost";
inline constexpr char kOpenVulkanMattingRuntimeFailureReason[] =
    "open_vulkan_matting_runtime_failure";

// One fail-closed readiness verdict is shared by setup, live-session
// activation, and daemon diagnostics. Loader/device/kernel/source-name facts do
// not appear here as shortcuts: every affirmative production fact is required.
struct VulkanMattingReadinessInput {
  bool production_build_enabled = false;
  bool production_adapter_available = false;
  bool model_pack_selected = false;
  bool model_contract_validated = false;
  bool non_cpu_device_selected = false;
  bool compute_queue_available = false;
  bool context_healthy = false;
  bool session_initialized = false;
  bool session_warmed = false;
  bool failure_latched = false;
  VulkanMattingRuntimeFailure failure = VulkanMattingRuntimeFailure::none;
  std::string failure_detail;
  VulkanMattingRuntimeEvidence runtime;
};

struct VulkanMattingReadiness {
  bool production_ready = false;
  // Existing per-effect blocker retained for compatibility.
  std::string reason_code = kOpenVulkanMattingUnavailableReason;
  // Stable, specific prerequisite/failure classification.
  std::string blocker_code = kOpenVulkanMattingBuildDisabledReason;
  std::string detail;
  VulkanMattingRuntimeEvidence runtime;
};

VulkanMattingReadiness
EvaluateVulkanMattingReadiness(const VulkanMattingReadinessInput &input);
std::string
FormatVulkanMattingReadiness(const VulkanMattingReadiness &readiness);

class VulkanMattingRuntime {
public:
  virtual ~VulkanMattingRuntime() = default;

  virtual VulkanMattingRuntimeResult
  Initialize(const VulkanMattingDeviceContext &device) = 0;
  virtual VulkanMattingRuntimeResult
  LoadGraph(const VulkanMattingGraphDescriptor &graph) = 0;
  virtual VulkanMattingRuntimeResult PreparePersistentResources(
      const VulkanMattingPersistentResources &resources) = 0;
  virtual VulkanMattingRuntimeResult Warmup() = 0;
  virtual VulkanMattingRuntimeResult
  Run(const VulkanMattingBufferBinding &input,
      const VulkanMattingBufferBinding &alpha_output) = 0;
  virtual VulkanMattingRuntimeEvidence Evidence() const = 0;
};

// Owns and enforces the production lifecycle. Prepare() performs runtime
// creation, graph load, and persistent allocation at most once. Warmup()
// executes at most once. Run() never performs setup work or allocation.
class VulkanMattingRuntimeLifecycle {
public:
  explicit VulkanMattingRuntimeLifecycle(
      std::unique_ptr<VulkanMattingRuntime> runtime);
  ~VulkanMattingRuntimeLifecycle();

  VulkanMattingRuntimeLifecycle(const VulkanMattingRuntimeLifecycle &) = delete;
  VulkanMattingRuntimeLifecycle &
  operator=(const VulkanMattingRuntimeLifecycle &) = delete;

  bool Prepare(const VulkanMattingDeviceContext &device,
               const VulkanMattingGraphDescriptor &graph,
               const VulkanMattingPersistentResources &resources,
               std::string *error_out);
  bool Warmup(std::string *error_out);
  bool Run(const VulkanMattingBufferBinding &input,
           const VulkanMattingBufferBinding &alpha_output,
           std::string *error_out);
  bool LatchExternalFailure(VulkanMattingRuntimeFailure failure,
                            std::string detail, std::string *error_out);

  bool available() const;
  bool prepared() const;
  bool warmed() const;
  bool failure_latched() const;
  VulkanMattingRuntimeFailure latched_failure() const;
  const std::string &latched_error() const;
  VulkanMattingRuntimeEvidence Evidence() const;

private:
  bool AcceptResult(const VulkanMattingRuntimeResult &result,
                    const char *operation, std::string *error_out);
  bool Fail(VulkanMattingRuntimeFailure failure, std::string detail, bool latch,
            std::string *error_out);
  bool VerifyDeviceIdentity(const VulkanMattingRuntimeEvidence &evidence,
                            std::string *error_out);
  bool VerifySetupEvidence(const VulkanMattingRuntimeEvidence &evidence,
                           std::string *error_out);
  bool VerifyBinding(const VulkanMattingBufferBinding &binding,
                     const char *name, int expected_n, int expected_c,
                     int expected_h, int expected_w, std::string *error_out);
  bool VerifyExecutionEvidence(const VulkanMattingRuntimeEvidence &evidence,
                               bool warmup, std::string *error_out);

  std::unique_ptr<VulkanMattingRuntime> runtime_;
  VulkanMattingDeviceContext device_;
  VulkanMattingGraphDescriptor graph_;
  VulkanMattingPersistentResources resources_;
  bool prepare_attempted_ = false;
  bool prepared_ = false;
  bool warmup_attempted_ = false;
  bool warmed_ = false;
  std::uint64_t persistent_allocation_count_ = 0;
  std::uint64_t completion_count_ = 0;
  VulkanMattingRuntimeFailure latched_failure_ =
      VulkanMattingRuntimeFailure::none;
  std::string latched_error_;
};

// Default fail-closed adapter used when production ncnn Vulkan support is not
// compiled. It never uses ncnn::Mat or any CPU input/output bridge.
std::unique_ptr<VulkanMattingRuntime>
CreateUnavailableVulkanMattingRuntime(std::string reason);

// Build/link availability is necessary but intentionally not sufficient. The
// current ncnn API owns its Vulkan device, so StudioCast keeps this factory
// unavailable until a reviewed adapter can import the already-selected shared
// VkPhysicalDevice/VkDevice/VkQueue and buffers without CPU bridges.
bool ProductionVulkanMattingBuildEnabled();
bool ProductionVulkanMattingAdapterAvailable();
std::unique_ptr<VulkanMattingRuntime> CreateProductionVulkanMattingRuntime();

} // namespace studiocast::open_vulkan
