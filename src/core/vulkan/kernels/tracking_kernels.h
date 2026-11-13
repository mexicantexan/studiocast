#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "core/vulkan/vulkan_image.h"

namespace studiocast::vulkan::kernels {

enum class TrackingShader : std::uint8_t {
  tracking,
  dlib_predictor,
  eye_contact
};
struct TrackingBufferBinding {
  VkBuffer buffer = nullptr;
  VkDeviceSize bytes = 0;
  VulkanContextIdentity context{};
};
struct TrackingDispatch {
  TrackingShader shader = TrackingShader::tracking;
  std::array<TrackingBufferBinding, 8> buffers{};
  std::array<std::uint32_t, 32> parameters{};
  std::uint32_t invocations = 1;
};

// Fixed-capacity setup-recorded leaf command storage. Every descriptor tuple is
// immutable once recorded, including on devices without push descriptors.
// Queue submission and completion ownership belong to ResidentCommandBatch.
class TrackingKernels {
public:
  TrackingKernels();
  ~TrackingKernels();
  bool Prepare(VulkanDevice *, bool enable_dlib, std::size_t command_capacity,
               std::size_t descriptor_capacity, std::string *error);
  bool Record(std::size_t command_slot, std::span<const TrackingDispatch>,
              VkCommandBuffer *out, std::string *error);
  bool Shutdown(std::string *error) noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace studiocast::vulkan::kernels
