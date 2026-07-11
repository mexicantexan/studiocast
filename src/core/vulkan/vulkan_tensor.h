#pragma once

#include <cstddef>
#include <string>

#include "core/vulkan/vulkan_image.h"

namespace studiocast::vulkan {

struct VulkanTensorSize {
  std::size_t elements = 0;
  std::size_t bytes = 0;
};

bool CheckedNchwF32Size(int n, int c, int h, int w,
                        VulkanTensorSize *size_out,
                        std::string *error_out);

class VulkanTensor {
public:
  VulkanTensor() = default;
  VulkanTensor(const VulkanTensor &) = delete;
  VulkanTensor &operator=(const VulkanTensor &) = delete;
  ~VulkanTensor();

  bool AllocateNchwF32(VulkanDevice *device, int n_in, int c_in, int h_in,
                       int w_in, bool map_memory, std::string *error_out);
  void Free() noexcept;

  bool Valid() const;
  std::size_t ElementCount() const;

  VkBuffer buffer() const { return storage_.buffer(); }
  VkDeviceMemory memory() const { return storage_.memory(); }
  VkDeviceSize byte_size() const { return storage_.size(); }
  void *mapped() const { return storage_.mapped(); }

  int n() const { return n_; }
  int c() const { return c_; }
  int h() const { return h_; }
  int w() const { return w_; }
  std::size_t bytes() const { return bytes_; }

  bool Flush(std::string *error_out) const { return storage_.Flush(error_out); }
  bool Invalidate(std::string *error_out) const {
    return storage_.Invalidate(error_out);
  }

private:
  VulkanBuffer storage_;
  std::size_t bytes_ = 0;
  int n_ = 0;
  int c_ = 0;
  int h_ = 0;
  int w_ = 0;
};

} // namespace studiocast::vulkan
