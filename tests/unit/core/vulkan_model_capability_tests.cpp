#include "core/hal/static_registry.h"

#include <array>
#include <initializer_list>
#include <iostream>

namespace {
using namespace studiocast::hal;
using Code = CapabilityEvidenceCode;
void Add(SemanticCapabilityEvidence &capability, Code code) {
  auto &record = capability.evidence[capability.evidence_count++];
  record.code = code;
  record.backend = BackendId::vulkan;
  record.semantic = capability.semantic;
  record.status = StatusId::ready;
  record.reason = ReasonId::none;
  record.satisfied = true;
}
SemanticCapabilityEvidence Make(SemanticEffectId semantic,
                                std::initializer_list<Code> required,
                                bool tracked = false) {
  SemanticCapabilityEvidence capability;
  capability.semantic = semantic;
  capability.supports_tracked_vignette_center = tracked;
  for (std::uint32_t code = 1; code <= 12; ++code)
    Add(capability, static_cast<Code>(code));
  for (auto code : required)
    Add(capability, code);
  return capability;
}
bool Check(SemanticCapabilityEvidence capability) {
  if (!HasVulkanProductionEvidence(capability))
    return false;
  for (std::size_t i = 0; i < capability.evidence_count; ++i) {
    auto changed = capability;
    changed.evidence[i].satisfied = false;
    if (HasVulkanProductionEvidence(changed))
      return false;
    changed = capability;
    changed.evidence[i].backend = BackendId::cuda;
    if (HasVulkanProductionEvidence(changed))
      return false;
    changed = capability;
    changed.evidence[i].status = StatusId::unavailable;
    if (HasVulkanProductionEvidence(changed))
      return false;
  }
  capability.evidence_count = capability.evidence.size() + 1;
  return !HasVulkanProductionEvidence(capability);
}
} // namespace

int main() {
  const std::array cases{
      Make(SemanticEffectId::mirror,
           {Code::open_vulkan_mirror_production_ready}),
      Make(SemanticEffectId::vignette,
           {Code::open_vulkan_vignette_fixed_center_production_ready}),
      Make(SemanticEffectId::virtual_background_blur,
           {Code::vulkan_model_artifacts_valid, Code::vulkan_matting_runtime_ready}),
      Make(SemanticEffectId::virtual_background_remove,
           {Code::vulkan_model_artifacts_valid, Code::vulkan_matting_runtime_ready}),
      Make(SemanticEffectId::virtual_background_replace,
           {Code::vulkan_model_artifacts_valid, Code::vulkan_matting_runtime_ready}),
      Make(SemanticEffectId::virtual_key_light,
           {Code::vulkan_model_artifacts_valid, Code::vulkan_matting_runtime_ready}),
      Make(SemanticEffectId::video_noise_removal,
           {Code::vulkan_model_artifacts_valid, Code::vulkan_denoise_runtime_ready}),
      Make(SemanticEffectId::eye_contact,
           {Code::vulkan_model_artifacts_valid, Code::vulkan_face_runtime_ready,
            Code::vulkan_eye_runtime_ready, Code::vulkan_landmarks_runtime_ready,
            Code::vulkan_shader_float64_enabled}),
      Make(SemanticEffectId::auto_frame,
           {Code::vulkan_model_artifacts_valid, Code::vulkan_face_runtime_ready,
            Code::vulkan_tracking_gpu_ready}),
      Make(SemanticEffectId::vignette,
           {Code::vulkan_model_artifacts_valid, Code::vulkan_face_runtime_ready,
            Code::vulkan_tracking_gpu_ready,
            Code::vulkan_tracked_vignette_center_supported}, true),
  };
  for (const auto &capability : cases) {
    if (!Check(capability)) {
      std::cerr << "Incomplete/foreign evidence admitted or complete effect rejected\n";
      return 1;
    }
  }
  for (const auto semantic : {SemanticEffectId::video_noise_removal,
                             SemanticEffectId::eye_contact,
                             SemanticEffectId::auto_frame}) {
    const auto impostor = Make(semantic,
        {Code::vulkan_model_artifacts_valid, Code::vulkan_matting_runtime_ready});
    if (HasVulkanProductionModelEvidence(impostor)) {
      std::cerr << "Matting-only proof qualified another model effect\n";
      return 1;
    }
  }
  return 0;
}
