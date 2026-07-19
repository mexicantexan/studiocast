#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

#include "core/vulkan/vulkan_device.h"
#include "core/vulkan/vulkan_image.h"
#include "core/vulkan/vulkan_tensor.h"

namespace studiocast::vulkan::kernels {

enum class ChannelOrder {
  rgb,
  bgr,
};

struct ModelPreprocessSpec {
  int dst_w = 0;
  int dst_h = 0;
  float mean[3] = {0.0f, 0.0f, 0.0f};
  float std[3] = {1.0f, 1.0f, 1.0f};
  ChannelOrder dst_order = ChannelOrder::rgb;
};

namespace detail {

inline constexpr int kBoxBlurMaxRadius = 64;

inline int NormalizeBoxBlurRadius(int radius) {
  return radius < 0 ? 0 : radius;
}

bool CheckBoxBlurRadiusForKernel(int radius, const char *what,
                                 std::string *error_out);

float ClampAlpha01ForComposite(float alpha);

std::array<std::uint8_t, 3>
SolidBackgroundMemoryChannels(VulkanPixelFormat format, std::uint8_t bg_r,
                              std::uint8_t bg_g, std::uint8_t bg_b);

} // namespace detail

class UtilityKernels {
public:
  UtilityKernels() = default;
  UtilityKernels(const UtilityKernels &) = delete;
  UtilityKernels &operator=(const UtilityKernels &) = delete;
  ~UtilityKernels();

  bool Initialize(std::string *error_out);
  void Shutdown() noexcept;

  bool Initialized() const { return initialized_; }
  VulkanDevice *device() { return &device_; }
  const VulkanDevice *device() const { return &device_; }
  OpenVulkanDiagnostics Diagnostics() const;
  std::uint64_t synchronous_submission_count() const {
    return synchronous_submission_count_.load(std::memory_order_relaxed);
  }
  std::size_t last_frame_batch_stage_count() const {
    return frame_batch_.recorded_stage_count();
  }
  std::uint64_t frame_batch_completion_count() const {
    return frame_batch_.completion_count();
  }

  bool CropResizeBilinear(const VulkanImage &src, const VulkanImage &dst,
                          float crop_x, float crop_y, float crop_w,
                          float crop_h, std::string *error_out);
  bool ResizeBilinear(const VulkanImage &src, const VulkanImage &dst,
                      std::string *error_out);

  bool PreprocessToTensor(const VulkanImage &src, const VulkanTensor &dst,
                          const ModelPreprocessSpec &spec,
                          std::string *error_out);

  bool ResizeBilinearF32_1(const VulkanImage &src, const VulkanImage &dst,
                           std::string *error_out);

  bool BoxBlurSeparableU8x3(const VulkanImage &src, const VulkanImage &tmp,
                            const VulkanImage &dst, int radius,
                            std::string *error_out);
  bool BoxBlurSeparableF32_1(const VulkanImage &src, const VulkanImage &tmp,
                             const VulkanImage &dst, int radius,
                             std::string *error_out);

  bool CompositeAlphaU8x3(const VulkanImage &fg, const VulkanImage &bg,
                          const VulkanImage &alpha, const VulkanImage &out,
                          std::string *error_out);
  // Records the two-pass background blur and the dependent alpha composite in
  // one command buffer. This keeps the intermediate image on the GPU and pays
  // one queue submit/fence wait instead of one per public kernel call.
  bool BoxBlurCompositeAlphaU8x3(const VulkanImage &fg,
                                 const VulkanImage &blur_tmp,
                                 const VulkanImage &blurred,
                                 const VulkanImage &alpha,
                                 const VulkanImage &out, int radius,
                                 std::string *error_out);
  bool CompositeAlphaSolidU8x3(const VulkanImage &fg, const VulkanImage &alpha,
                               std::uint8_t bg_r, std::uint8_t bg_g,
                               std::uint8_t bg_b, const VulkanImage &out,
                               std::string *error_out);

  bool ApplyKeyLightU8x3(const VulkanImage &src, const VulkanImage &alpha,
                         float target_r, float target_g, float target_b,
                         float intensity01, float direction,
                         const VulkanImage &out, std::string *error_out);

private:
  enum class Op : std::uint32_t {
    crop_resize_u8x3 = 1,
    preprocess_to_nchw = 2,
    resize_f32_1 = 3,
    blur_h_u8x3 = 4,
    blur_v_u8x3 = 5,
    blur_h_f32_1 = 6,
    blur_v_f32_1 = 7,
    composite_bg = 8,
    composite_solid = 9,
    key_light = 10,
    copy_u8x3 = 11,
    copy_f32_1 = 12,
  };

  struct Params;

  bool EnsurePipeline(std::string *error_out);
  bool EnsureDescriptors(std::string *error_out);
  bool EnsureCommandBuffer(std::string *error_out);
  bool BindBuffersForSet(VkDescriptorSet descriptor_set, VkBuffer params_buffer,
                         std::array<VkBuffer, 7> *bound_buffers,
                         VkBuffer u8_src0, VkBuffer u8_src1, VkBuffer f32_src0,
                         VkBuffer u8_out, VkBuffer f32_out, VkBuffer scratch,
                         std::string *error_out);
  bool BindBuffers(VkBuffer u8_src0, VkBuffer u8_src1, VkBuffer f32_src0,
                   VkBuffer u8_out, VkBuffer f32_out, VkBuffer scratch,
                   std::string *error_out);
  bool Dispatch(const Params &params, Op op, std::uint32_t dispatch_w,
                std::uint32_t dispatch_h, std::string *error_out);
  bool DispatchTwoPass(const Params &params, Op first, Op second,
                       std::uint32_t dispatch_w, std::uint32_t dispatch_h,
                       std::string *error_out);
  bool SubmitRecorded(std::string *error_out);

  VulkanDevice device_;
  VulkanBuffer params_;
  VulkanBuffer batch_params_;
  VulkanBuffer dummy_;

  VkDescriptorSetLayout descriptor_set_layout_ = nullptr;
  VkDescriptorPool descriptor_pool_ = nullptr;
  VkDescriptorSet descriptor_set_ = nullptr;
  VkDescriptorSet batch_descriptor_set_ = nullptr;
  VkShaderModule shader_module_ = nullptr;
  VkPipelineLayout pipeline_layout_ = nullptr;
  VkPipeline pipeline_ = nullptr;
  VkCommandBuffer command_buffer_ = nullptr;
  VulkanCommandBatch frame_batch_;
  std::array<VkBuffer, 7> bound_buffers_{};
  std::array<VkBuffer, 7> batch_bound_buffers_{};

  bool initialized_ = false;
  bool pipeline_created_ = false;
  std::atomic<std::uint64_t> synchronous_submission_count_{0};
  std::string init_error_;
  mutable std::recursive_mutex execution_mutex_;
};

bool IsUtilityKernelsAvailable(std::string *error_out);

} // namespace studiocast::vulkan::kernels
