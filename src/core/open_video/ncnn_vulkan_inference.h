#pragma once

#include "core/open_video/model_pack_registry.h"
#include "core/open_video/vulkan_matting_runtime.h"
#include "core/vulkan/vulkan_device.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace studiocast::vulkan {
class ResidentCommandBatch;
}

namespace studiocast::open_vulkan {

// One caller-owned aggregate limit for the complete adopted context, including
// leaf/temporal/staging allocations. Every graph keeps a distinct reuse pool;
// the shared VulkanDevice enforces these limits before native acquisition.
struct VulkanInferenceSharedQuota {
  vulkan::VulkanDevice *device = nullptr;
  vulkan::VulkanContextIdentity context{};
  std::size_t allocation_budget_bytes = 0;
  std::size_t allocation_resource_limit = 0;
};

struct VulkanInferenceTensorBinding {
  VulkanMattingBufferBinding resource{};
  std::uint32_t dims = 0;
  int w = 0;
  int h = 1;
  int c = 1;
  std::size_t row_stride_bytes = 0;
  std::size_t plane_stride_bytes = 0;
};

struct VulkanInferenceBindings {
  std::array<VulkanInferenceTensorBinding,
             open_video::kMaxNcnnVulkanInputs> inputs{};
  std::array<VulkanInferenceTensorBinding,
             open_video::kMaxNcnnVulkanOutputs> outputs{};
  std::size_t input_count = 0;
  std::size_t output_count = 0;
  std::uint64_t owner_generation = 0;
};

// Concrete, record-once GPU-only graph owner. All paths, strings, bindings,
// Extractor calls, packing conversions and allocations are setup-only.
// Enqueue appends three immutable primary command buffers as one participant;
// the exact batch ticket alone can publish completed inference evidence.
class NcnnVulkanInference {
public:
  NcnnVulkanInference();
  ~NcnnVulkanInference();
  NcnnVulkanInference(const NcnnVulkanInference &) = delete;
  NcnnVulkanInference &operator=(const NcnnVulkanInference &) = delete;

  VulkanMattingRuntimeResult Prepare(
      const VulkanMattingDeviceContext &device,
      const open_video::NcnnVulkanNetworkSpec &immutable_graph,
      const VulkanInferenceBindings &bindings,
      std::shared_ptr<VulkanInferenceSharedQuota> shared_quota);
  VulkanMattingRuntimeResult Warmup();
  // Enqueued producers need not be CPU-complete: the exact same-batch context,
  // generation, immutable binding, and recorded prefix barriers prove ordered
  // access. No caller-supplied synchronized bool can override these checks.
  VulkanMattingRuntimeResult Enqueue(vulkan::ResidentCommandBatch &batch);
  VulkanMattingRuntimeEvidence Evidence() const;
  // False retains all resources after unsafe completion; late quiescence may
  // permit retirement but never fabricates successful inference counters.
  bool Shutdown() noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::open_vulkan
