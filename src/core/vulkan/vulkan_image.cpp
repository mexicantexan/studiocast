#include "core/vulkan/vulkan_image.h"

#include <cstring>
#include <limits>
#include <sstream>

namespace studiocast::vulkan {

std::size_t VulkanStorageBytesPerPixel(VulkanPixelFormat format) {
  switch (format) {
  case VulkanPixelFormat::rgba_u8:
  case VulkanPixelFormat::rgb_u8:
  case VulkanPixelFormat::bgr_u8:
  case VulkanPixelFormat::f32_1:
    return 4u;
  }
  return 0u;
}

VulkanBuffer::~VulkanBuffer() { Free(); }

bool VulkanBuffer::Allocate(VulkanDevice *device, VkDeviceSize size,
                            VkFlags usage, VkFlags required_memory_flags,
                            VkFlags preferred_memory_flags, bool map_memory,
                            std::string *error_out) {
  Free();
  if (error_out)
    error_out->clear();
  if (!device || !device->Initialized()) {
    if (error_out)
      *error_out =
          "VulkanBuffer::Allocate called without an initialized device.";
    return false;
  }
  if (size == 0) {
    if (error_out)
      *error_out = "VulkanBuffer::Allocate called with size=0.";
    return false;
  }

  const auto context = device->context_handle();
  if (!context || !context->Healthy()) {
    if (error_out)
      *error_out = "[vulkan_context_unhealthy] VulkanBuffer::Allocate called "
                   "with an unhealthy context.";
    return false;
  }

  const auto &vf = context->f();
  VkBufferCreateInfo create{};
  create.size = size;
  create.usage = usage;
  VkBuffer buffer = nullptr;
  VkResult result =
      vf.vkCreateBuffer(context->device, &create, nullptr, &buffer);
  if (!context->RecordDriverFailure(result, "vkCreateBuffer",
                                    /*submission_failure=*/false, error_out))
    return false;

  VkMemoryRequirements req{};
  vf.vkGetBufferMemoryRequirements(context->device, buffer, &req);
  if (req.size == 0) {
    if (error_out)
      *error_out = "[vulkan_allocation_size_invalid] Vulkan buffer reported "
                   "zero memory requirements.";
    vf.vkDestroyBuffer(context->device, buffer, nullptr);
    return false;
  }
  std::uint32_t memory_type = 0;
  if (!device->FindMemoryType(req.memoryTypeBits, required_memory_flags,
                              preferred_memory_flags, &memory_type,
                              error_out)) {
    vf.vkDestroyBuffer(context->device, buffer, nullptr);
    return false;
  }
  if (!context->ReserveAllocation(req.size, error_out)) {
    vf.vkDestroyBuffer(context->device, buffer, nullptr);
    return false;
  }

  VkMemoryAllocateInfo alloc{};
  alloc.allocationSize = req.size;
  alloc.memoryTypeIndex = memory_type;
  VkDeviceMemory memory = nullptr;
  result = vf.vkAllocateMemory(context->device, &alloc, nullptr, &memory);
  if (!context->RecordDriverFailure(result, "vkAllocateMemory",
                                    /*submission_failure=*/false, error_out)) {
    context->ReleaseAllocation(req.size);
    vf.vkDestroyBuffer(context->device, buffer, nullptr);
    return false;
  }

  result = vf.vkBindBufferMemory(context->device, buffer, memory, 0);
  if (!context->RecordDriverFailure(result, "vkBindBufferMemory",
                                    /*submission_failure=*/false, error_out)) {
    vf.vkDestroyBuffer(context->device, buffer, nullptr);
    vf.vkFreeMemory(context->device, memory, nullptr);
    context->ReleaseAllocation(req.size);
    return false;
  }

  void *mapped = nullptr;
  if (map_memory) {
    result =
        vf.vkMapMemory(context->device, memory, 0, VK_WHOLE_SIZE, 0, &mapped);
    if (!context->RecordDriverFailure(result, "vkMapMemory",
                                      /*submission_failure=*/false,
                                      error_out)) {
      vf.vkDestroyBuffer(context->device, buffer, nullptr);
      vf.vkFreeMemory(context->device, memory, nullptr);
      context->ReleaseAllocation(req.size);
      return false;
    }
  }

  context_ = context;
  context_identity_ = context->identity;
  device_ = device;
  buffer_ = buffer;
  memory_ = memory;
  size_ = size;
  allocation_size_ = req.size;
  mapped_ = mapped;
  return true;
}

void VulkanBuffer::Free() noexcept {
  if (!context_)
    return;
  const auto context = context_;
  const auto &vf = context->f();
  const bool safe_to_destroy = context->SafeToDestroyChildren();
  if (safe_to_destroy && mapped_ && vf.vkUnmapMemory)
    vf.vkUnmapMemory(context->device, memory_);
  mapped_ = nullptr;
  if (safe_to_destroy && buffer_ && vf.vkDestroyBuffer)
    vf.vkDestroyBuffer(context->device, buffer_, nullptr);
  buffer_ = nullptr;
  if (safe_to_destroy && memory_ && vf.vkFreeMemory)
    vf.vkFreeMemory(context->device, memory_, nullptr);
  memory_ = nullptr;
  context->ReleaseAllocation(allocation_size_);
  size_ = 0;
  allocation_size_ = 0;
  device_ = nullptr;
  context_identity_ = {};
  context_.reset();
}

bool VulkanBuffer::Valid() const {
  return context_ && context_->Active() && context_identity_.Valid() &&
         buffer_ && memory_ && size_ > 0 && allocation_size_ >= size_;
}

bool VulkanBuffer::BelongsTo(const VulkanDevice &device) const {
  return Valid() && device.OwnsContext(context_identity_);
}

bool VulkanBuffer::Flush(std::string *error_out) const {
  if (!Valid()) {
    if (error_out)
      *error_out = "Cannot flush an invalid Vulkan buffer.";
    return false;
  }
  if (!context_->Healthy()) {
    const auto snapshot = context_->HealthSnapshot();
    if (error_out)
      *error_out = "[" + snapshot.reason_code +
                   "] Cannot flush an unhealthy Vulkan buffer context.";
    return false;
  }
  VkMappedMemoryRange range{};
  range.memory = memory_;
  range.size = VK_WHOLE_SIZE;
  const VkResult result =
      context_->f().vkFlushMappedMemoryRanges(context_->device, 1, &range);
  return context_->RecordDriverFailure(result, "vkFlushMappedMemoryRanges",
                                       /*submission_failure=*/false, error_out);
}

bool VulkanBuffer::Invalidate(std::string *error_out) const {
  if (!Valid()) {
    if (error_out)
      *error_out = "Cannot invalidate an invalid Vulkan buffer.";
    return false;
  }
  if (!context_->Healthy()) {
    const auto snapshot = context_->HealthSnapshot();
    if (error_out)
      *error_out = "[" + snapshot.reason_code +
                   "] Cannot invalidate an unhealthy Vulkan buffer context.";
    return false;
  }
  VkMappedMemoryRange range{};
  range.memory = memory_;
  range.size = VK_WHOLE_SIZE;
  const VkResult result =
      context_->f().vkInvalidateMappedMemoryRanges(context_->device, 1, &range);
  return context_->RecordDriverFailure(result, "vkInvalidateMappedMemoryRanges",
                                       /*submission_failure=*/false, error_out);
}

VulkanImage::~VulkanImage() { Free(); }

bool VulkanImage::Allocate(VulkanDevice *device, int width, int height,
                           std::string *error_out) {
  return Allocate(device, width, height, VulkanPixelFormat::rgba_u8,
                  /*map_memory=*/false, error_out);
}

bool VulkanImage::Allocate(VulkanDevice *device, int width, int height,
                           VulkanPixelFormat format, bool map_memory,
                           std::string *error_out) {
  Free();
  if (width <= 0 || height <= 0) {
    if (error_out)
      *error_out = "VulkanImage dimensions must be positive.";
    return false;
  }
  const std::size_t bpp = VulkanStorageBytesPerPixel(format);
  if (bpp == 0) {
    if (error_out)
      *error_out = "VulkanImage unknown pixel format.";
    return false;
  }
  const auto w = static_cast<std::size_t>(width);
  const auto h = static_cast<std::size_t>(height);
  if (w > std::numeric_limits<std::size_t>::max() / h ||
      (w * h) > std::numeric_limits<std::size_t>::max() / bpp) {
    if (error_out)
      *error_out = "VulkanImage allocation size overflow.";
    return false;
  }

  const std::size_t pixels = w * h;
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(pixels * bpp);
  const VkFlags required_memory_flags =
      map_memory ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT : 0u;
  const VkFlags preferred_memory_flags =
      map_memory ? (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                 : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if (!storage_.Allocate(device, bytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         required_memory_flags, preferred_memory_flags,
                         map_memory, error_out)) {
    return false;
  }

  width_ = width;
  height_ = height;
  pitch_pixels_ = w;
  format_ = format;
  return true;
}

void VulkanImage::Free() noexcept {
  storage_.Free();
  width_ = 0;
  height_ = 0;
  pitch_pixels_ = 0;
  format_ = VulkanPixelFormat::rgba_u8;
}

void PackRgb24ToRgba32(const std::uint8_t *src, std::size_t src_stride_bytes,
                       int width, int height, std::uint32_t *dst,
                       std::size_t dst_pitch_pixels) {
  if (!src || !dst || width <= 0 || height <= 0)
    return;
  for (int y = 0; y < height; ++y) {
    const std::uint8_t *row =
        src + static_cast<std::size_t>(y) * src_stride_bytes;
    std::uint32_t *out = dst + static_cast<std::size_t>(y) * dst_pitch_pixels;
    for (int x = 0; x < width; ++x) {
      const std::size_t i = static_cast<std::size_t>(x) * 3u;
      out[x] = std::uint32_t(row[i]) | (std::uint32_t(row[i + 1]) << 8u) |
               (std::uint32_t(row[i + 2]) << 16u) | 0xff000000u;
    }
  }
}

void UnpackRgba32ToRgb24(const std::uint32_t *src, std::size_t src_pitch_pixels,
                         int width, int height, std::uint8_t *dst,
                         std::size_t dst_stride_bytes) {
  if (!src || !dst || width <= 0 || height <= 0)
    return;
  for (int y = 0; y < height; ++y) {
    const std::uint32_t *row =
        src + static_cast<std::size_t>(y) * src_pitch_pixels;
    std::uint8_t *out = dst + static_cast<std::size_t>(y) * dst_stride_bytes;
    for (int x = 0; x < width; ++x) {
      const std::uint32_t px = row[x];
      const std::size_t i = static_cast<std::size_t>(x) * 3u;
      out[i] = static_cast<std::uint8_t>(px & 0xffu);
      out[i + 1] = static_cast<std::uint8_t>((px >> 8u) & 0xffu);
      out[i + 2] = static_cast<std::uint8_t>((px >> 16u) & 0xffu);
    }
  }
}

} // namespace studiocast::vulkan
