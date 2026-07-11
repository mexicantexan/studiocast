#pragma once

#include <cstdint>
#include <string>

#include "core/vulkan/vulkan_loader.h"

namespace studiocast::vulkan {

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

  std::string error;
  std::string fallback_reason;

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
