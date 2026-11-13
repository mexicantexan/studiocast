#pragma once

// Backend-private setup/diagnostics surface. Neither the neutral manager nor
// production frame execution depends on native diagnostics or model strings.
#include "core/open_video/model_pack_registry.h"
#include "core/open_video/vulkan_matting_runtime.h"
#include "core/vulkan/hal_resident_adapter.h"
#include "core/vulkan/vulkan_device.h"

#include <memory>
#include <string>

namespace studiocast::vulkan {
class VulkanVideoExecutionFactory;
struct VulkanProductionModelDiagnostics {
  OpenVulkanDiagnostics device;
  VulkanAllocationStats allocations;
  VulkanResidentCapabilityEvidence capabilities;
  open_vulkan::VulkanMattingReadiness matting_readiness;
  open_vulkan::VulkanMattingRuntimeEvidence matting, denoise, face, left_eye,
      right_eye;
  std::uint64_t completed_batches = 0;
  std::shared_ptr<const open_video::ModelPack> matting_pack, denoise_pack,
      face_pack, landmarks_pack, eye_pack;
};

// Reads the same actual owner prepared by Initialize. No probing, hashing,
// warmup, GPU submission, synchronization or readback occurs here. Call only
// outside concurrent frame execution; ordinary status consumes its cache.
bool CollectProductionVulkanModelDiagnostics(
    const VulkanVideoExecutionFactory &, VulkanProductionModelDiagnostics *,
    std::string *error);
} // namespace studiocast::vulkan
