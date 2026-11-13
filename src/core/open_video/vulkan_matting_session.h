#pragma once

#include <memory>
#include <string>

#include "core/open_video/model_pack_registry.h"
#include "core/open_video/vulkan_matting_runtime.h"
#include "core/vulkan/kernels/utility_kernels.h"
#include "core/vulkan/vulkan_image.h"

namespace studiocast::open_vulkan {

// Long-lived matting session for the Open Vulkan backend.
//
// Owns persistent Vulkan preprocessing and one recorded GPU-only inference
// graph on the exact StudioCast context. The optional production runtime uses
// fixed resident input/alpha bindings; the CPU-Mat spike never participates.
class OpenVulkanMattingSession {
public:
  struct Options {
    int device_id = 0;
    bool allow_cpu_layers = false;
    bool require_device_residency = true;
    int warmup_runs = 1;
  };

  OpenVulkanMattingSession(studiocast::vulkan::VulkanDevice *device,
                           studiocast::vulkan::kernels::UtilityKernels *kernels,
                           studiocast::open_video::ModelPack pack);
  OpenVulkanMattingSession(studiocast::vulkan::VulkanDevice *device,
                           studiocast::vulkan::kernels::UtilityKernels *kernels,
                           studiocast::open_video::ModelPack pack,
                           Options opts);
  // Runtime injection is a focused no-GPU test seam. Production construction
  // uses the exact-device adapter selected by the build.
  OpenVulkanMattingSession(studiocast::vulkan::VulkanDevice *device,
                           studiocast::vulkan::kernels::UtilityKernels *kernels,
                           studiocast::open_video::ModelPack pack, Options opts,
                           std::unique_ptr<VulkanMattingRuntime> runtime);
  ~OpenVulkanMattingSession();

  OpenVulkanMattingSession(const OpenVulkanMattingSession &) = delete;
  OpenVulkanMattingSession &
  operator=(const OpenVulkanMattingSession &) = delete;

  bool BindAlphaOutputForSetup(const studiocast::vulkan::VulkanImage *alpha,
                               std::string *error_out);
  void UpdateCachedReadiness(VulkanMattingReadiness *readiness) const;

  bool EnsureInitialized(int frame_w, int frame_h, std::string *error_out);
  bool Warmup(std::string *error_out);

  bool Run(const studiocast::vulkan::VulkanImage &input_rgb_gpu,
           studiocast::vulkan::VulkanImage *output_alpha_gpu,
           std::string *error_out);
  bool Enqueue(const studiocast::vulkan::VulkanImage &input_rgb_gpu,
               studiocast::vulkan::VulkanImage *output_alpha_gpu,
               studiocast::vulkan::ResidentCommandBatch &batch,
               std::string *error_out);

  bool DeviceResidentInferenceAvailable() const;
  VulkanMattingReadiness Readiness() const;
  VulkanMattingRuntimeEvidence RuntimeEvidence() const;
  VulkanMattingRuntimeFailure LatchedRuntimeFailure() const;
  const std::string &LatchedRuntimeError() const;
  bool LatchExternalFailure(VulkanMattingRuntimeFailure failure,
                            std::string detail, std::string *error_out);

  const studiocast::open_video::ModelPack &pack() const;
  const Options &options() const;

private:
  bool RunInternal(const studiocast::vulkan::VulkanImage &,
                   studiocast::vulkan::VulkanImage *,
                   studiocast::vulkan::ResidentCommandBatch *, std::string *);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::open_vulkan
