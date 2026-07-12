#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "core/vulkan/vulkan_device.h"

namespace studiocast::vulkan {

enum class VulkanPixelFormat {
  rgba_u8,
  rgb_u8,
  bgr_u8,
  f32_1,
};

std::size_t VulkanStorageBytesPerPixel(VulkanPixelFormat format);

class VulkanBuffer {
public:
  VulkanBuffer() = default;
  VulkanBuffer(const VulkanBuffer &) = delete;
  VulkanBuffer &operator=(const VulkanBuffer &) = delete;
  ~VulkanBuffer();

  bool Allocate(VulkanDevice *device, VkDeviceSize size, VkFlags usage,
                VkFlags required_memory_flags, VkFlags preferred_memory_flags,
                bool map_memory, std::string *error_out);
  void Free() noexcept;

  bool Valid() const { return device_ && buffer_ && memory_ && size_ > 0; }
  VulkanDevice *device() const { return device_; }
  VkBuffer buffer() const { return buffer_; }
  VkDeviceMemory memory() const { return memory_; }
  VkDeviceSize size() const { return size_; }
  void *mapped() const { return mapped_; }

  bool Flush(std::string *error_out) const;
  bool Invalidate(std::string *error_out) const;

private:
  VulkanDevice *device_ = nullptr;
  VkBuffer buffer_ = nullptr;
  VkDeviceMemory memory_ = nullptr;
  VkDeviceSize size_ = 0;
  void *mapped_ = nullptr;
};

class VulkanImage {
public:
  VulkanImage() = default;
  VulkanImage(const VulkanImage &) = delete;
  VulkanImage &operator=(const VulkanImage &) = delete;
  ~VulkanImage();

  bool Allocate(VulkanDevice *device, int width, int height,
                std::string *error_out);
  bool Allocate(VulkanDevice *device, int width, int height,
                VulkanPixelFormat format, bool map_memory,
                std::string *error_out);
  void Free() noexcept;

  bool Valid() const { return storage_.Valid(); }
  VulkanDevice *device() const { return storage_.device(); }
  VkBuffer buffer() const { return storage_.buffer(); }
  VkDeviceMemory memory() const { return storage_.memory(); }
  VkDeviceSize byte_size() const { return storage_.size(); }
  void *mapped() const { return storage_.mapped(); }
  int width() const { return width_; }
  int height() const { return height_; }
  VulkanPixelFormat format() const { return format_; }
  std::size_t pitch_pixels() const { return pitch_pixels_; }
  std::size_t pitch_bytes() const {
    return pitch_pixels_ * VulkanStorageBytesPerPixel(format_);
  }
  std::size_t row_bytes() const {
    return width_ > 0 ? static_cast<std::size_t>(width_) *
                            VulkanStorageBytesPerPixel(format_)
                      : 0u;
  }

  bool Flush(std::string *error_out) const { return storage_.Flush(error_out); }
  bool Invalidate(std::string *error_out) const {
    return storage_.Invalidate(error_out);
  }

private:
  VulkanBuffer storage_;
  int width_ = 0;
  int height_ = 0;
  std::size_t pitch_pixels_ = 0;
  VulkanPixelFormat format_ = VulkanPixelFormat::rgba_u8;
};

void PackRgb24ToRgba32(const std::uint8_t *src, std::size_t src_stride_bytes,
                       int width, int height, std::uint32_t *dst,
                       std::size_t dst_pitch_pixels);

void UnpackRgba32ToRgb24(const std::uint32_t *src,
                         std::size_t src_pitch_pixels, int width, int height,
                         std::uint8_t *dst, std::size_t dst_stride_bytes);

} // namespace studiocast::vulkan
