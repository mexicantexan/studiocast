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
  residency_check_failed,
  execution_failed,
};

const char *VulkanMattingRuntimeFailureName(
    VulkanMattingRuntimeFailure failure);

struct VulkanMattingRuntimeResult {
  VulkanMattingRuntimeFailure failure = VulkanMattingRuntimeFailure::none;
  std::string detail;

  bool ok() const { return failure == VulkanMattingRuntimeFailure::none; }

  static VulkanMattingRuntimeResult Success();
  static VulkanMattingRuntimeResult
  Failure(VulkanMattingRuntimeFailure failure, std::string detail);
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
};

// Evidence is queried after every setup gate and execution. Availability is
// granted only if the adapter proves all of these facts; a successful return
// code alone is insufficient.
struct VulkanMattingRuntimeEvidence {
  std::string runtime_name;
  bool runtime_created = false;
  bool graph_loaded = false;
  bool persistent_resources_allocated = false;
  bool warmup_complete = false;
  bool cpu_layers_used = false;
  bool device_identity_matches = false;
  bool input_device_resident = false;
  bool alpha_device_resident = false;
  bool output_device_resident = false;
  VulkanMattingDeviceContext active_device;
};

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
  bool Fail(VulkanMattingRuntimeFailure failure, std::string detail,
            bool latch, std::string *error_out);
  bool VerifyDeviceIdentity(const VulkanMattingRuntimeEvidence &evidence,
                            std::string *error_out);
  bool VerifySetupEvidence(const VulkanMattingRuntimeEvidence &evidence,
                           std::string *error_out);
  bool VerifyBinding(const VulkanMattingBufferBinding &binding,
                     const char *name, int expected_n, int expected_c,
                     int expected_h, int expected_w,
                     std::string *error_out);

  std::unique_ptr<VulkanMattingRuntime> runtime_;
  VulkanMattingDeviceContext device_;
  VulkanMattingGraphDescriptor graph_;
  VulkanMattingPersistentResources resources_;
  bool prepare_attempted_ = false;
  bool prepared_ = false;
  bool warmup_attempted_ = false;
  bool warmed_ = false;
  VulkanMattingRuntimeFailure latched_failure_ =
      VulkanMattingRuntimeFailure::none;
  std::string latched_error_;
};

// Default fail-closed adapter used when production ncnn Vulkan support is not
// compiled. It never uses ncnn::Mat or any CPU input/output bridge.
std::unique_ptr<VulkanMattingRuntime>
CreateUnavailableVulkanMattingRuntime(std::string reason);

} // namespace studiocast::open_vulkan
