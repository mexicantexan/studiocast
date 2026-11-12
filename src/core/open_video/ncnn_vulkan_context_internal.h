#pragma once
#if STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
#include <gpu.h>
#include <memory>
namespace studiocast::open_vulkan::detail {
inline constexpr char kNcnnOwnerKind[] = "ncnn-studiocast-replay-v1";
struct NcnnVulkanContext {
  std::shared_ptr<void> instance;
  std::unique_ptr<ncnn::VulkanDevice> device;
  VkQueue queue = VK_NULL_HANDLE;
  ~NcnnVulkanContext() {
    if (device && queue)
      device->reclaim_queue(device->info.compute_queue_family_index(), queue);
  }
};
} // namespace studiocast::open_vulkan::detail
#endif
