#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "core/vulkan/vulkan_loader.h"

namespace studiocast::vulkan {

class VulkanBuffer;

struct VulkanContextIdentity {
  std::uint64_t context_id = 0;
  std::uint64_t generation = 0;

  bool Valid() const { return context_id != 0 && generation != 0; }
  friend bool operator==(const VulkanContextIdentity &,
                         const VulkanContextIdentity &) = default;
};

enum class VulkanContextHealth {
  uninitialized,
  healthy,
  device_lost,
  unsafe_timeout,
  fatal_submission_error,
  shutdown,
};

const char *VulkanContextHealthName(VulkanContextHealth health);

struct VulkanHealthSnapshot {
  VulkanContextHealth health = VulkanContextHealth::uninitialized;
  bool poisoned = false;
  std::string reason_code = "vulkan_context_uninitialized";
  VkResult last_result = VK_SUCCESS;
  std::string operation;
  std::string message;
  std::uint64_t submitted_serial = 0;
  std::uint64_t completed_serial = 0;
};

struct VulkanAllocationStats {
  VkDeviceSize budget_bytes = 0;
  VkDeviceSize current_bytes = 0;
  VkDeviceSize high_water_bytes = 0;
  std::uint64_t allocation_count = 0;
};

struct VulkanDeviceConfig {
  // Zero derives the budget from the selected physical device's distinct
  // memory heaps. A non-zero value is an explicit context cap and is clamped
  // to the derived physical-device total.
  VkDeviceSize allocation_budget_bytes = 0;
  std::uint64_t fence_timeout_ns = VK_DEFAULT_FENCE_TIMEOUT_NS;
};

enum class VulkanBufferAccess {
  host_write,
  host_or_compute_write,
  transfer_write,
  compute_read,
  compute_write,
  transfer_read,
  host_read,
};

struct VulkanBufferBarrierSpec {
  VkFlags src_access_mask = 0;
  VkFlags dst_access_mask = 0;
  VkFlags src_stage_mask = 0;
  VkFlags dst_stage_mask = 0;
};

VulkanBufferBarrierSpec VulkanBufferBarrier(VulkanBufferAccess source,
                                            VulkanBufferAccess destination);

enum class VulkanSubmissionPhase {
  reset_fence,
  queue_submit,
  wait_for_fence,
};

namespace detail {

class VulkanContextState final {
public:
  VulkanContextState() = default;
  VulkanContextState(const VulkanContextState &) = delete;
  VulkanContextState &operator=(const VulkanContextState &) = delete;
  ~VulkanContextState();

  const VulkanFunctions &f() const { return loader->f(); }
  VulkanFunctions &f() { return loader->f(); }
  bool Active() const;
  bool Healthy() const;
  bool SafeToDestroyChildren() const;
  VulkanHealthSnapshot HealthSnapshot() const;
  VulkanAllocationStats AllocationStats() const;
  bool ReserveAllocation(VkDeviceSize bytes, std::string *error_out);
  void ReleaseAllocation(VkDeviceSize bytes) noexcept;
  bool RecordDriverFailure(VkResult result, std::string_view operation,
                           bool submission_failure, std::string *error_out);
  void MarkShutdown() noexcept;

  std::shared_ptr<VulkanLoader> loader;
  VulkanContextIdentity identity;
  VkInstance instance = nullptr;
  VkPhysicalDevice physical_device = nullptr;
  VkDevice device = nullptr;
  VkQueue queue = nullptr;
  VkCommandPool command_pool = nullptr;
  VkFence fence = nullptr;
  VkPhysicalDeviceMemoryProperties memory_properties{};
  VkDeviceSize allocation_budget_bytes = 0;
  std::uint64_t fence_timeout_ns = VK_DEFAULT_FENCE_TIMEOUT_NS;

  mutable std::mutex health_mutex;
  VulkanHealthSnapshot health;
  mutable std::mutex allocation_mutex;
  VulkanAllocationStats allocation_stats;
  mutable std::mutex submission_mutex;
  mutable std::mutex command_pool_mutex;
  std::optional<std::pair<VulkanSubmissionPhase, VkResult>>
      injected_submission_result;
};

} // namespace detail

struct VulkanDeviceSelection {
  std::optional<std::uint32_t> requested_index;
  std::string requested_stable_id;
  bool allow_cpu_in_auto = false;
  std::string request = "auto";
  std::string source = "automatic";
};

struct VulkanDeviceIdentity {
  std::string stable_id;
  std::uint32_t vendor_id = 0;
  std::uint32_t device_id = 0;
  std::uint32_t device_type = 0;
  std::string device_name;
};

struct VulkanDeviceCandidateInfo {
  std::uint32_t enumeration_index = 0;
  std::uint32_t api_version = 0;
  std::uint32_t driver_version = 0;
  std::uint32_t vendor_id = 0;
  std::uint32_t device_id = 0;
  std::uint32_t device_type = 0;
  std::string stable_id;
  std::string vendor_name;
  std::string device_name;
  int compute_queue_family_index = -1;
  int score = 0;
  bool eligible = false;
  bool selected = false;
  std::string rejection_reason;
};

struct VulkanDeviceSelectionResult {
  bool ok = false;
  std::size_t candidate_vector_index = 0;
  std::string failure_reason;
  std::string error;
};

namespace detail {

bool ParseVulkanDeviceSelection(std::string_view requested_index,
                                std::string_view allow_cpu_in_auto,
                                VulkanDeviceSelection *selection,
                                std::string *error_out);

std::string MakeVulkanDeviceStableId(std::uint32_t vendor_id,
                                     std::uint32_t device_id,
                                     std::uint32_t device_type,
                                     std::string_view device_name);

bool IsValidVulkanDeviceStableId(std::string_view stable_id);

VulkanDeviceSelectionResult
SelectVulkanDeviceCandidate(std::vector<VulkanDeviceCandidateInfo> *candidates,
                            const VulkanDeviceSelection &selection);

} // namespace detail

struct OpenVulkanDiagnostics {
  bool compiled_enabled = true;
  bool ok = false;
  bool runtime_library_found = false;
  std::string runtime_library_path;
  bool instance_created = false;
  bool physical_device_found = false;
  bool non_cpu_device_selected = false;
  bool compute_queue_available = false;
  bool logical_device_created = false;
  bool context_created = false;
  bool context_healthy = false;
  bool production_hardware_ready = false;
  bool shader_pipeline_created = false;
  std::string context_health = "uninitialized";
  std::string context_failure_reason = "vulkan_context_uninitialized";
  std::uint64_t context_id = 0;
  std::uint64_t context_generation = 0;
  std::uint64_t allocation_budget_bytes = 0;

  std::uint32_t api_version = 0;
  std::uint32_t driver_version = 0;
  std::uint32_t vendor_id = 0;
  std::uint32_t device_id = 0;
  std::uint32_t device_type = 0;
  std::string vendor_name;
  std::string device_name;
  std::uint32_t compute_queue_family_index = 0;
  std::string device_selection_source = "automatic";
  std::string device_selection_request = "auto";
  int selected_device_index = -1;
  std::string selected_device_stable_id;
  bool cpu_device_selected = false;
  std::vector<VulkanDeviceCandidateInfo> device_candidates;

  std::string error;
  std::string fallback_reason;
  std::string blocked_reason;
  std::string degraded_reason;

  struct ModelInfo {
    std::string id;
    std::string display_name;
    std::string task;
    int width = 0;
    int height = 0;
  };
  std::vector<std::string> installed_models;
  std::vector<ModelInfo> models;
  std::string default_model_id;
  std::map<std::string, std::string> missing_models;

  std::vector<std::string> available_effects;
  std::map<std::string, std::string> blocked_effects;
  std::vector<std::string> install_hints;

  bool eye_contact_production_ready = false;
  std::string eye_contact_reason_code =
      "open_vulkan_eye_contact_unavailable";
  std::string eye_contact_blocker_code =
      "open_vulkan_eye_contact_runtime_unavailable";
  std::string eye_contact_detail;
  bool eye_contact_backend_compiled = false;
  bool eye_contact_live_stage_implemented = false;
  bool eye_contact_production_adapter_available = false;
  bool eye_contact_vulkan_inference_provider_available = false;
  bool eye_contact_non_cpu_device_selected = false;
  bool eye_contact_compute_queue_available = false;
  bool eye_contact_context_healthy = false;
  bool eye_contact_shared_device_imported = false;
  bool eye_contact_queue_ownership_explicit = false;
  bool eye_contact_model_pack_selected = false;
  bool eye_contact_artifact_contract_validated = false;
  bool eye_contact_device_resident_analysis = false;
  bool eye_contact_device_resident_tensor_io = false;
  bool eye_contact_warmup_complete = false;
  bool eye_contact_bounded_reusable_allocations = false;
  bool eye_contact_synchronization_contract_validated = false;
  bool eye_contact_parity_validated = false;
  bool eye_contact_selectable_cpu_fallback = false;
  std::uint64_t eye_contact_dispatch_count = 0;
  std::uint64_t eye_contact_cpu_readback_count = 0;
  std::uint64_t eye_contact_cpu_fallback_count = 0;

  bool video_noise_removal_production_ready = false;
  std::string video_noise_removal_reason_code =
      "open_vulkan_video_noise_removal_unavailable";
  std::string video_noise_removal_blocker_code =
      "open_vulkan_video_noise_removal_runtime_unavailable";
  std::string video_noise_removal_detail;
  bool video_noise_removal_backend_compiled = false;
  bool video_noise_removal_live_stage_implemented = false;
  bool video_noise_removal_production_adapter_available = false;
  bool video_noise_removal_vulkan_inference_provider_available = false;
  bool video_noise_removal_non_cpu_device_selected = false;
  bool video_noise_removal_compute_queue_available = false;
  bool video_noise_removal_context_healthy = false;
  bool video_noise_removal_shared_device_imported = false;
  bool video_noise_removal_queue_ownership_explicit = false;
  bool video_noise_removal_model_pack_selected = false;
  bool video_noise_removal_artifact_contract_validated = false;
  bool video_noise_removal_fully_device_resident_tensor_io = false;
  bool video_noise_removal_device_resident_preprocess = false;
  bool video_noise_removal_device_resident_postprocess = false;
  bool video_noise_removal_warmup_complete = false;
  bool video_noise_removal_synchronization_contract_validated = false;
  bool video_noise_removal_bounded_reusable_allocations = false;
  bool video_noise_removal_temporal_history_device_resident = false;
  bool video_noise_removal_temporal_history_bounded = false;
  bool video_noise_removal_history_reset_on_disable = false;
  bool video_noise_removal_history_reset_on_reconfigure = false;
  bool video_noise_removal_capture_sequence_discontinuity_reset = false;
  bool video_noise_removal_parity_validated = false;
  bool video_noise_removal_selectable_cpu_fallback = false;
  std::uint64_t video_noise_removal_dispatch_count = 0;
  std::uint64_t video_noise_removal_temporal_history_reset_count = 0;
  std::uint64_t video_noise_removal_cpu_readback_count = 0;
  std::uint64_t video_noise_removal_cpu_fallback_count = 0;

  std::string matting_runtime;
  bool matting_build_enabled = false;
  bool matting_adapter_available = false;
  bool matting_model_pack_selected = false;
  bool matting_model_contract_validated = false;
  bool matting_production_ready = false;
  std::string matting_reason_code = "open_vulkan_matting_unavailable";
  std::string matting_blocker_code;
  std::string matting_detail;
  bool matting_runtime_created = false;
  bool matting_graph_loaded = false;
  bool matting_warmup_complete = false;
  bool matting_cpu_layers_used = false;
  bool matting_shared_device_imported = false;
  bool matting_queue_ownership_explicit = false;
  bool matting_synchronous_completion = false;
  bool matting_bounded_reusable_allocations = false;
  std::uint64_t matting_persistent_allocation_count = 0;
  std::uint64_t matting_dynamic_allocation_count = 0;
  std::uint64_t matting_cpu_readback_count = 0;
  std::uint64_t matting_warmup_inference_count = 0;
  std::uint64_t matting_inference_count = 0;
  std::uint64_t matting_completion_count = 0;
  std::uint64_t matting_context_id = 0;
  std::uint64_t matting_context_generation = 0;
  bool input_device_resident = false;
  bool alpha_device_resident = false;
  bool output_device_resident = false;
  std::string device_residency_mode;
  std::vector<std::string> warnings;

  std::string ToJson() const;
};

class VulkanDevice {
public:
  VulkanDevice();
  VulkanDevice(const VulkanDevice &) = delete;
  VulkanDevice &operator=(const VulkanDevice &) = delete;
  ~VulkanDevice();

  bool Initialize(std::string *error_out);
  bool Initialize(const VulkanDeviceSelection &selection,
                  std::string *error_out);
  bool Initialize(const VulkanDeviceSelection &selection,
                  const VulkanDeviceConfig &config, std::string *error_out);
  void Shutdown() noexcept;

  bool Initialized() const { return initialized_; }
  const VulkanFunctions &f() const { return loader_->f(); }
  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkDevice device() const { return device_; }
  VkQueue queue() const { return queue_; }
  VkCommandPool command_pool() const { return command_pool_; }
  std::uint32_t queue_family_index() const { return queue_family_index_; }
  const VulkanDeviceIdentity &identity() const { return identity_; }
  const VkPhysicalDeviceMemoryProperties &memory_properties() const {
    return memory_properties_;
  }
  const OpenVulkanDiagnostics &diagnostics() const { return diagnostics_; }
  OpenVulkanDiagnostics DiagnosticsSnapshot() const;
  const VulkanContextIdentity &context_identity() const {
    return context_identity_;
  }
  bool OwnsContext(const VulkanContextIdentity &identity) const;
  VulkanHealthSnapshot health() const;
  VulkanAllocationStats allocation_stats() const;
  bool SafeToDestroyResources() const;

  bool FindMemoryType(std::uint32_t type_bits, VkFlags required_flags,
                      VkFlags preferred_flags, std::uint32_t *type_index,
                      std::string *error_out) const;

  bool SubmitAndWait(VkCommandBuffer command_buffer, std::string *error_out);
  bool CheckDriverResult(VkResult result, std::string_view operation,
                         bool poison_on_failure, std::string *error_out) const;
  bool RecordBufferBarrier(VkCommandBuffer command_buffer, VkBuffer buffer,
                           VkDeviceSize size,
                           const VulkanContextIdentity &resource_context,
                           VulkanBufferAccess source,
                           VulkanBufferAccess destination,
                           std::string *error_out) const;
  bool FlushMemory(VkDeviceMemory memory, VkDeviceSize offset,
                   VkDeviceSize size, std::string *error_out) const;
  bool InvalidateMemory(VkDeviceMemory memory, VkDeviceSize offset,
                        VkDeviceSize size, std::string *error_out) const;

  // Deterministic fault-injection seam. When present, SubmitAndWait consumes
  // this result without calling the driver and exercises the same latch path.
  void InjectNextSubmissionResultForTesting(VulkanSubmissionPhase phase,
                                            VkResult result);

private:
  friend class VulkanBuffer;
  friend class VulkanCommandBatch;
  bool LoadInstanceFunctions(std::string *error_out);
  bool LoadDeviceFunctions(std::string *error_out);
  bool PickPhysicalDevice(std::string *error_out);
  bool CreateLogicalDevice(std::string *error_out);
  bool CreateCommandInfrastructure(std::string *error_out);
  void DestroyUnownedHandles() noexcept;
  std::shared_ptr<detail::VulkanContextState> context_handle() const {
    return context_;
  }

  std::shared_ptr<VulkanLoader> loader_;
  std::shared_ptr<detail::VulkanContextState> context_;
  VulkanDeviceSelection selection_;
  VulkanDeviceConfig config_;
  VulkanDeviceIdentity identity_;
  VkInstance instance_ = nullptr;
  VkPhysicalDevice physical_device_ = nullptr;
  VkDevice device_ = nullptr;
  VkQueue queue_ = nullptr;
  VkCommandPool command_pool_ = nullptr;
  VkFence fence_ = nullptr;
  std::uint32_t queue_family_index_ = 0;
  VkPhysicalDeviceMemoryProperties memory_properties_{};
  OpenVulkanDiagnostics diagnostics_{};
  VulkanContextIdentity context_identity_{};
  std::uint64_t context_id_ = 0;
  std::uint64_t next_generation_ = 0;
  bool initialized_ = false;
};

// Fixed-capacity, single-owner command recording scope. It retains the shared
// context generation, enforces same-thread serial use, and has exactly one
// synchronous completion boundary for all recorded stages.
class VulkanCommandBatch {
public:
  VulkanCommandBatch() = default;
  VulkanCommandBatch(const VulkanCommandBatch &) = delete;
  VulkanCommandBatch &operator=(const VulkanCommandBatch &) = delete;
  ~VulkanCommandBatch();

  bool Initialize(VulkanDevice *device, std::size_t stage_capacity,
                  std::string *error_out);
  void Shutdown() noexcept;
  bool Begin(std::string *error_out);
  bool RecordStage(std::string_view label, std::string *error_out);
  bool RecordBufferBarrier(VkBuffer buffer, VkDeviceSize size,
                           const VulkanContextIdentity &resource_context,
                           VulkanBufferAccess source,
                           VulkanBufferAccess destination,
                           std::string *error_out);
  bool Complete(std::string *error_out);
  void Abort() noexcept;

  VkCommandBuffer command_buffer() const;
  bool recording() const;
  std::size_t stage_capacity() const { return stage_capacity_; }
  std::size_t recorded_stage_count() const;
  std::uint64_t completion_count() const;
  const VulkanContextIdentity &context_identity() const {
    return context_identity_;
  }

private:
  bool CheckRecordingOwner(std::string *error_out) const;

  std::shared_ptr<detail::VulkanContextState> context_;
  VulkanContextIdentity context_identity_{};
  VkCommandBuffer command_buffer_ = nullptr;
  std::size_t stage_capacity_ = 0;
  std::size_t recorded_stage_count_ = 0;
  std::uint64_t completion_count_ = 0;
  bool recording_ = false;
  std::thread::id recording_owner_{};
  mutable std::mutex mutex_;
};

OpenVulkanDiagnostics DiagnoseOpenVulkanDefault();

// Installs the daemon-authoritative selection used by subsequently initialized
// VulkanDevice instances in this process. Existing devices are not migrated.
void SetProcessVulkanDeviceSelection(const VulkanDeviceSelection &selection);
void ClearProcessVulkanDeviceSelection();

} // namespace studiocast::vulkan
