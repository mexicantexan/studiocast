#pragma once

#include "core/video/open_vulkan_final_resident_stage.h"
#include "core/vulkan/hal_resident_adapter.h"

#include "core/open_video/vulkan_matting_session.h"
#include "core/video/effects/broadcast_effects.h"
#include "core/video/open_vulkan_virtual_background_blur.h"
#include "core/video/open_vulkan_virtual_background_remove.h"
#include "core/video/open_vulkan_virtual_background_replace.h"
#include "core/video/open_vulkan_virtual_key_light.h"
#include <array>
#include <cstdint>

namespace studiocast::vulkan {
struct VulkanVideoExecutionFactoryConfiguration;
struct VulkanProductionModelDiagnostics;

// Synchronous production executor for one exact Open Vulkan island: denoise,
// eye correction, shared matting/background/key light, AutoFrame, final resize,
// fixed or tracked vignette and mirror. UtilityKernels and its adopted or
// owned VulkanDevice must outlive this object; other adapters cannot submit
// work to these prepared resources.
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

  bool InitializeMatting(const open_video::ModelPack &pack,
                         const video::effects::BroadcastCameraEffects &effects,
                         int width, int height, std::string *error);
  bool InitializeModels(const VulkanVideoExecutionFactoryConfiguration &,
                        std::string *error);
  hal::SemanticEffectId InitializationFailedSemantic() const noexcept {
    return initialization_failed_semantic_;
  }
  void CollectModelDiagnostics(VulkanProductionModelDiagnostics *) const;
  // Shutdown-only: retire failed callback tickets after the exact device
  // submission is proven quiescent. Never promotes a failed frame to success.
  bool RetireFailedFrameAfterQuiescence() noexcept;
  const open_vulkan::VulkanMattingReadiness &MattingReadiness() const noexcept {
    return matting_readiness_;
  }

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
  hal::StatusId ResetTemporalState(const VulkanExecutorPreparedProgram &,
                                   hal::ReasonId) noexcept override;
  void
  DestroyPrepared(VulkanExecutorPreparedProgram *prepared) noexcept override;
  void Shutdown() noexcept override;

private:
  struct Slot {
    bool occupied = false;
    bool quarantined = false;
    hal::ReasonId quarantine_reason = hal::ReasonId::none;
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
    VulkanImage alpha_model;
    VulkanImage alpha;
    VulkanImage alpha_tmp;
    VulkanImage alpha_feathered;
    VulkanImage blur_tmp;
    VulkanImage blurred;
    VulkanImage background_output;
    VulkanImage keylight_output;
    video::OpenVulkanVirtualBackgroundBlur background_blur;
    video::OpenVulkanVirtualBackgroundRemove background_remove;
    video::OpenVulkanVirtualBackgroundReplace background_replace;
    video::OpenVulkanVirtualKeyLight keylight;
    video::OpenVulkanVirtualBackgroundBlurCounters blur_counters;
    video::OpenVulkanVirtualBackgroundRemoveCounters remove_counters;
    video::OpenVulkanVirtualBackgroundReplaceCounters replace_counters;
    video::OpenVulkanVirtualKeyLightCounters keylight_counters;
    std::uint64_t alpha_resize_completions = 0;
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
  VulkanExecutorResult ExecuteExpanded(Slot *, const hal::PreparedProgram &,
                                       const VulkanResidentProgramOptions &,
                                       const VulkanHostFrameBinding &,
                                       std::uint64_t capture_sequence) noexcept;
  struct ModelEffects;
  std::unique_ptr<ModelEffects> model_effects_;
  hal::SemanticEffectId initialization_failed_semantic_ =
      hal::SemanticEffectId::invalid;

  std::uint64_t adapter_instance_id_ = 0;
  kernels::UtilityKernels *kernels_ = nullptr;
  VulkanResidentCapabilityEvidence evidence_{};
  std::array<Slot, kMaxVulkanPreparedPrograms> slots_{};
  std::uint64_t next_state_id_ = 1;
  bool shutdown_ = false;
  VulkanImage matting_alpha_;
  std::unique_ptr<open_vulkan::OpenVulkanMattingSession> matting_;
  open_vulkan::VulkanMattingReadiness matting_readiness_;
  video::effects::BroadcastCameraEffects effects_{};
};

} // namespace studiocast::vulkan
