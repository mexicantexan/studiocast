#pragma once

#include "core/video/open_vulkan_final_resident_stage.h"
#include "core/vulkan/hal_resident_adapter.h"

#include <array>
#include <cstdint>

namespace studiocast::vulkan {

// Synchronous production executor for the exact qualified Open Vulkan final
// island: fixed-center vignette followed by final mirror. UtilityKernels and
// its VulkanDevice must outlive this object; no other adapter may submit work
// to this executor's prepared resources.
class QualifiedFinalResidentExecutor final
    : public IQualifiedVulkanResidentExecutor {
public:
  QualifiedFinalResidentExecutor(std::uint64_t adapter_instance_id,
                                 kernels::UtilityKernels &kernels) noexcept;
  ~QualifiedFinalResidentExecutor() override;
  QualifiedFinalResidentExecutor(const QualifiedFinalResidentExecutor &) =
      delete;
  QualifiedFinalResidentExecutor &
  operator=(const QualifiedFinalResidentExecutor &) = delete;

  [[nodiscard]] VulkanResidentCapabilityEvidence
  CapabilityEvidence() const noexcept override;
  hal::StatusId Prepare(const hal::PreparedProgram &program,
                        const VulkanResidentProgramOptions &options,
                        VulkanExecutorPreparedProgram *prepared,
                        hal::ReasonId *reason) noexcept override;
  VulkanExecutorResult
  Execute(const VulkanExecutorPreparedProgram &prepared,
          const hal::PreparedProgram &program,
          const VulkanResidentProgramOptions &options,
          const VulkanHostFrameBinding &binding,
          std::uint64_t capture_sequence) noexcept override;
  void
  DestroyPrepared(VulkanExecutorPreparedProgram *prepared) noexcept override;
  void Shutdown() noexcept override;

private:
  struct Slot {
    bool occupied = false;
    bool quarantined = false;
    std::uint64_t state_id = 0;
    std::uint64_t program_id = 0;
    std::uint64_t canonical_plan_hash = 0;
    hal::ExecutionDomainId execution_domain{};
    VulkanResidentProgramOptions options{};
    std::uint32_t input_width = 0;
    std::uint32_t input_height = 0;
    std::uint32_t output_width = 0;
    std::uint32_t output_height = 0;
    VulkanPixelFormat format = VulkanPixelFormat::rgb_u8;
    VulkanImage upload_staging;
    VulkanImage source;
    VulkanImage resize_scratch;
    VulkanImage vignette_output;
    VulkanImage mirror_output;
    VulkanImage readback_staging;
    video::OpenVulkanVignette vignette;
    video::OpenVulkanVignetteCounters vignette_counters;
    video::OpenVulkanMirror mirror;
    video::OpenVulkanMirrorCounters mirror_counters;
    video::OpenVulkanFinalResidentStageCounters final_counters;
    hal::ResourceBudget reserved{};
  };

  [[nodiscard]] Slot *Find(std::uint64_t state_id) noexcept;
  void Release(Slot *slot) noexcept;
  void RefreshEvidence() noexcept;

  std::uint64_t adapter_instance_id_ = 0;
  kernels::UtilityKernels *kernels_ = nullptr;
  VulkanResidentCapabilityEvidence evidence_{};
  std::array<Slot, kMaxVulkanPreparedPrograms> slots_{};
  std::uint64_t next_state_id_ = 1;
  bool shutdown_ = false;
};

} // namespace studiocast::vulkan
