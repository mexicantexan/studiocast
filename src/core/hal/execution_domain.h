#pragma once

#include "core/hal/identifiers.h"

#include <array>
#include <cstdint>

namespace studiocast::hal {

enum class ApiFamily : std::uint16_t {
  invalid = 0,
  host,
  cuda_driver,
  maxine_nvcv,
  onnxruntime,
  vulkan,
};

enum class AllocatorFamily : std::uint16_t {
  invalid = 0,
  host,
  cuda_driver,
  maxine_nvcv,
  onnxruntime,
  vulkan,
};

enum class MemoryHandleFamily : std::uint16_t {
  invalid = 0,
  host_pointer,
  cuda_device_pointer,
  nvcv_image,
  ort_value,
  vulkan_buffer,
  vulkan_image,
};

enum class OwnershipModel : std::uint16_t {
  invalid = 0,
  adapter_exclusive,
  borrowed_until_completion,
  externally_owned,
};

enum class SynchronizationProtocol : std::uint16_t {
  invalid = 0,
  host_synchronous,
  cuda_stream_ordered,
  maxine_bound_cuda_stream,
  ort_provider_synchronous,
  vulkan_queue_fence,
};

struct PhysicalDeviceIdentity {
  std::uint32_t vendor_id = 0;
  std::uint32_t device_id = 0;
  // Stable UUID/LUID/PCI-derived identity supplied by the adapter. An index or
  // device name is not sufficient.
  std::array<std::uint8_t, 16> stable_id{};

  friend bool operator==(const PhysicalDeviceIdentity &,
                         const PhysicalDeviceIdentity &) = default;
};

struct ExecutionDomainId {
  BackendId backend = BackendId::invalid;
  ApiFamily api_family = ApiFamily::invalid;
  PhysicalDeviceIdentity physical_device{};
  std::uint64_t logical_context_generation = 0;
  AllocatorFamily allocator_family = AllocatorFamily::invalid;
  MemoryHandleFamily memory_handle_family = MemoryHandleFamily::invalid;
  std::uint64_t queue_or_stream_identity = 0;
  OwnershipModel ownership_model = OwnershipModel::invalid;
  SynchronizationProtocol synchronization_protocol =
      SynchronizationProtocol::invalid;
  // One adapter owns each v1 resident island. Equal physical devices or API
  // contexts do not waive this identity check.
  std::uint64_t adapter_instance_id = 0;

  friend bool operator==(const ExecutionDomainId &,
                         const ExecutionDomainId &) = default;
};

[[nodiscard]] bool
ValidateExecutionDomain(const ExecutionDomainId &domain) noexcept;

} // namespace studiocast::hal
