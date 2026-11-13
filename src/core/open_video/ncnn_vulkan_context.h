#pragma once
#include "core/vulkan/vulkan_device.h"
#include <cstddef>
#include <string>

namespace studiocast::open_vulkan {
inline constexpr std::size_t kNcnnVulkanMattingAllocationBudgetBytes =
    256u * 1024u * 1024u;
inline constexpr std::size_t kNcnnVulkanMattingAllocationResourceLimit = 512;
// Setup decision shared by native adoption and hermetic low-feature tests.
bool ValidateProductionVulkanCoreFeatures(bool require_shader_float64,
                                          bool shader_float64_supported,
                                          std::string *error_out);
bool InitializeProductionVulkanMattingDevice(
    vulkan::VulkanDevice *device,
    const vulkan::VulkanDeviceSelection &selection,
    const vulkan::VulkanDeviceConfig &config, std::string *error_out,
    bool require_shader_float64 = false);
bool InitializeProductionVulkanMattingDevice(vulkan::VulkanDevice *device,
                                             std::string *error_out);
} // namespace studiocast::open_vulkan
