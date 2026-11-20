#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

#include "core/vulkan/vulkan_device.h"
#include "core/vulkan/vulkan_image.h"

namespace studiocast::vulkan::kernels {

// Runtime-loaded Vulkan compute RGB24 bilinear resize.
//
// The public boundary is RGB24, while the GPU image storage is pitched RGBA8 in
// storage buffers. Pipeline/descriptors/command buffers are created for stable
// dimensions during setup/reconfiguration; Resize() only updates mapped staging
// memory, submits the pre-recorded command buffer, waits for final readback,
// and unpacks RGB24.
class ResizeBilinear {
public:
  ResizeBilinear() = default;
  ResizeBilinear(const ResizeBilinear &) = delete;
  ResizeBilinear &operator=(const ResizeBilinear &) = delete;
  ~ResizeBilinear();

  bool EnsureInitialized(int src_w, int src_h, int dst_w, int dst_h,
                         std::string *error_out);
  bool Resize(const std::uint8_t *src, std::size_t src_stride_bytes,
              std::uint8_t *dst, std::size_t dst_stride_bytes,
              std::string *error_out);

  void Shutdown() noexcept;

  bool Initialized() const { return initialized_; }
  int src_width() const { return src_w_; }
  int src_height() const { return src_h_; }
  int dst_width() const { return dst_w_; }
  int dst_height() const { return dst_h_; }
  OpenVulkanDiagnostics Diagnostics() const;

private:
  bool CreateBuffers(std::string *error_out);
  bool CreatePipeline(std::string *error_out);
  bool CreateDescriptors(std::string *error_out);
  bool RecordCommandBuffer(std::string *error_out);

  VulkanDevice device_;
  VulkanImage gpu_src_;
  VulkanImage gpu_dst_;
  VulkanBuffer upload_;
  VulkanBuffer readback_;

  VkDescriptorSetLayout descriptor_set_layout_ = nullptr;
  VkDescriptorPool descriptor_pool_ = nullptr;
  VkDescriptorSet descriptor_set_ = nullptr;
  VkShaderModule shader_module_ = nullptr;
  VkPipelineLayout pipeline_layout_ = nullptr;
  VkPipeline pipeline_ = nullptr;
  VkCommandBuffer command_buffer_ = nullptr;

  int src_w_ = 0;
  int src_h_ = 0;
  int dst_w_ = 0;
  int dst_h_ = 0;
  bool initialized_ = false;
  bool pipeline_created_ = false;
  std::string init_error_;
  mutable std::recursive_mutex execution_mutex_;
};

bool IsResizeBilinearAvailable(std::string *error_out);

} // namespace studiocast::vulkan::kernels
