#pragma once

#include "core/hal/prepared_program.h"
#include "core/open_video/ncnn_vulkan_inference.h"
#include "core/vulkan/resident_command_batch.h"
#include "core/vulkan/vulkan_image.h"

#include <cstdint>
#include <memory>
#include <string>

namespace studiocast::open_vulkan {

struct PreparedVulkanFastDvdnetSetup {
  vulkan::VulkanDevice *device = nullptr;
  hal::ExecutionDomainId domain{};
  std::uint64_t owner_generation = 0;
  int width = 0, height = 0, strength = 0;
  open_video::NcnnVulkanNetworkSpec graph;
  std::shared_ptr<VulkanInferenceSharedQuota> quota;
  hal::ResourceBudget reservation{};
};

// Causal CUDA-compatible logical five-frame input [t-2,t-1,t,t,t].
// All temporal/model data remain device-local; only capture/reset parameters
// are CPU-written before one synchronous frame is queued.
class PreparedVulkanFastDvdnet {
public:
  PreparedVulkanFastDvdnet();
  ~PreparedVulkanFastDvdnet();
  PreparedVulkanFastDvdnet(const PreparedVulkanFastDvdnet &) = delete;
  PreparedVulkanFastDvdnet &
  operator=(const PreparedVulkanFastDvdnet &) = delete;

  bool Prepare(const PreparedVulkanFastDvdnetSetup &,
               const vulkan::VulkanImage &fixed_input,
               const vulkan::VulkanImage &fixed_output, std::string *error);
  bool Append(vulkan::ResidentCommandBatch &, std::uint64_t capture_sequence,
              std::string *error);
  bool ResetTemporalState() noexcept;
  bool Shutdown(std::string *error) noexcept;
  bool ready() const noexcept;
  hal::ResourceBudget resources() const noexcept;
  VulkanMattingRuntimeEvidence evidence() const;
  void RefreshExecutionEvidence(VulkanMattingRuntimeEvidence *) const noexcept;
  static std::uint32_t PaddedExtent(std::uint32_t extent) noexcept;
  static float NoiseSigmaOver255(int strength) noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::open_vulkan
