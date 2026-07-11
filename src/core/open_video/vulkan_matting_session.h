#pragma once

#include <memory>
#include <string>

#include "core/open_video/model_pack_registry.h"
#include "core/vulkan/kernels/utility_kernels.h"
#include "core/vulkan/vulkan_image.h"

namespace studiocast::open_vulkan {

// Long-lived matting session for the Open Vulkan backend.
//
// This owns Vulkan preprocess buffers and is the production integration point
// for a device-resident Vulkan inference runtime. The current milestone-4 ncnn
// spike used CPU Mat input/output, so this session refuses activation rather
// than hiding CPU matte transfers behind an Open Vulkan backend.
class OpenVulkanMattingSession {
public:
  struct Options {
    int device_id = 0;
    bool allow_cpu_layers = false;
    bool require_device_residency = true;
    int warmup_runs = 1;
  };

  OpenVulkanMattingSession(
      studiocast::vulkan::VulkanDevice *device,
      studiocast::vulkan::kernels::UtilityKernels *kernels,
      studiocast::open_video::ModelPack pack);
  OpenVulkanMattingSession(
      studiocast::vulkan::VulkanDevice *device,
      studiocast::vulkan::kernels::UtilityKernels *kernels,
      studiocast::open_video::ModelPack pack, Options opts);
  ~OpenVulkanMattingSession();

  OpenVulkanMattingSession(const OpenVulkanMattingSession &) = delete;
  OpenVulkanMattingSession &
  operator=(const OpenVulkanMattingSession &) = delete;

  bool EnsureInitialized(int frame_w, int frame_h, std::string *error_out);
  bool Warmup(std::string *error_out);

  bool Run(const studiocast::vulkan::VulkanImage &input_rgb_gpu,
           studiocast::vulkan::VulkanImage *output_alpha_gpu,
           std::string *error_out);

  bool DeviceResidentInferenceAvailable() const;

  const studiocast::open_video::ModelPack &pack() const;
  const Options &options() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::open_vulkan
