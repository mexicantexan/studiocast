#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/vulkan/vulkan_loader.h"

namespace studiocast::vulkan {

struct VulkanDeviceSelection {
  std::optional<std::uint32_t> requested_index;
  bool allow_cpu_in_auto = false;
  std::string request = "auto";
  std::string source = "automatic";
};

struct VulkanDeviceCandidateInfo {
  std::uint32_t enumeration_index = 0;
  std::uint32_t api_version = 0;
  std::uint32_t driver_version = 0;
  std::uint32_t vendor_id = 0;
  std::uint32_t device_id = 0;
  std::uint32_t device_type = 0;
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
  bool compute_queue_available = false;
  bool logical_device_created = false;
  bool shader_pipeline_created = false;

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

  std::string matting_runtime;
  bool matting_runtime_created = false;
  bool matting_graph_loaded = false;
  bool input_device_resident = false;
  bool alpha_device_resident = false;
  bool output_device_resident = false;
  std::string device_residency_mode;
  std::vector<std::string> warnings;

  std::string ToJson() const;
};

class VulkanDevice {
public:
  VulkanDevice() = default;
  VulkanDevice(const VulkanDevice &) = delete;
  VulkanDevice &operator=(const VulkanDevice &) = delete;
  ~VulkanDevice();

  bool Initialize(std::string *error_out);
  void Shutdown() noexcept;

  bool Initialized() const { return initialized_; }
  const VulkanFunctions &f() const { return loader_.f(); }
  VkDevice device() const { return device_; }
  VkQueue queue() const { return queue_; }
  VkCommandPool command_pool() const { return command_pool_; }
  std::uint32_t queue_family_index() const { return queue_family_index_; }
  const VkPhysicalDeviceMemoryProperties &memory_properties() const {
    return memory_properties_;
  }
  const OpenVulkanDiagnostics &diagnostics() const { return diagnostics_; }

  bool FindMemoryType(std::uint32_t type_bits, VkFlags required_flags,
                      VkFlags preferred_flags, std::uint32_t *type_index,
                      std::string *error_out) const;

  bool SubmitAndWait(VkCommandBuffer command_buffer, std::string *error_out);
  bool FlushMemory(VkDeviceMemory memory, VkDeviceSize offset,
                   VkDeviceSize size, std::string *error_out) const;
  bool InvalidateMemory(VkDeviceMemory memory, VkDeviceSize offset,
                        VkDeviceSize size, std::string *error_out) const;

private:
  bool LoadInstanceFunctions(std::string *error_out);
  bool LoadDeviceFunctions(std::string *error_out);
  bool PickPhysicalDevice(std::string *error_out);
  bool CreateLogicalDevice(std::string *error_out);
  bool CreateCommandInfrastructure(std::string *error_out);

  VulkanLoader loader_;
  VulkanDeviceSelection selection_;
  VkInstance instance_ = nullptr;
  VkPhysicalDevice physical_device_ = nullptr;
  VkDevice device_ = nullptr;
  VkQueue queue_ = nullptr;
  VkCommandPool command_pool_ = nullptr;
  VkFence fence_ = nullptr;
  std::uint32_t queue_family_index_ = 0;
  VkPhysicalDeviceMemoryProperties memory_properties_{};
  OpenVulkanDiagnostics diagnostics_{};
  bool initialized_ = false;
};

OpenVulkanDiagnostics DiagnoseOpenVulkanDefault();

} // namespace studiocast::vulkan
