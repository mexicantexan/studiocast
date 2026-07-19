#include "core/vulkan/vulkan_tensor.h"

#include <limits>

namespace studiocast::vulkan {

namespace {

bool CheckedMul(std::size_t a, std::size_t b, std::size_t *out) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    return false;
  *out = a * b;
  return true;
}

void ResetSize(VulkanTensorSize *size_out) {
  if (size_out)
    *size_out = {};
}

} // namespace

bool CheckedNchwF32Size(int n, int c, int h, int w,
                        VulkanTensorSize *size_out,
                        std::string *error_out) {
  if (error_out)
    error_out->clear();
  ResetSize(size_out);

  if (n <= 0 || c <= 0 || h <= 0 || w <= 0) {
    if (error_out)
      *error_out = "VulkanTensor NCHW F32 shape: invalid shape.";
    return false;
  }

  std::size_t elements = 1;
  if (!CheckedMul(elements, static_cast<std::size_t>(n), &elements) ||
      !CheckedMul(elements, static_cast<std::size_t>(c), &elements) ||
      !CheckedMul(elements, static_cast<std::size_t>(h), &elements) ||
      !CheckedMul(elements, static_cast<std::size_t>(w), &elements)) {
    if (error_out)
      *error_out = "VulkanTensor NCHW F32 shape: element count overflow.";
    return false;
  }

  std::size_t bytes = 0;
  if (!CheckedMul(elements, sizeof(float), &bytes)) {
    if (error_out)
      *error_out = "VulkanTensor NCHW F32 shape: byte count overflow.";
    return false;
  }

  if (size_out) {
    size_out->elements = elements;
    size_out->bytes = bytes;
  }
  return true;
}

VulkanTensor::~VulkanTensor() { Free(); }

bool VulkanTensor::AllocateNchwF32(VulkanDevice *device, int n_in, int c_in,
                                   int h_in, int w_in, bool map_memory,
                                   std::string *error_out) {
  Free();
  VulkanTensorSize size;
  if (!CheckedNchwF32Size(n_in, c_in, h_in, w_in, &size, error_out))
    return false;

  const VkFlags required_memory_flags =
      map_memory ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                 : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  const VkFlags preferred_memory_flags =
      map_memory ? (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                 : 0u;
  if (!storage_.Allocate(device, static_cast<VkDeviceSize>(size.bytes),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         required_memory_flags, preferred_memory_flags,
                         map_memory, error_out)) {
    return false;
  }

  bytes_ = size.bytes;
  n_ = n_in;
  c_ = c_in;
  h_ = h_in;
  w_ = w_in;
  return true;
}

void VulkanTensor::Free() noexcept {
  storage_.Free();
  bytes_ = 0;
  n_ = 0;
  c_ = 0;
  h_ = 0;
  w_ = 0;
}

bool VulkanTensor::Valid() const {
  VulkanTensorSize size;
  return storage_.Valid() && CheckedNchwF32Size(n_, c_, h_, w_, &size,
                                                nullptr) &&
         bytes_ == size.bytes && storage_.size() >= bytes_;
}

std::size_t VulkanTensor::ElementCount() const {
  VulkanTensorSize size;
  if (!CheckedNchwF32Size(n_, c_, h_, w_, &size, nullptr))
    return 0;
  return size.elements;
}

} // namespace studiocast::vulkan
