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
  std::uint64_t descriptor_binding_update_count() const {
    return descriptor_binding_update_count_.load(std::memory_order_relaxed);
  }

  // Setup-time resource destruction followed by allocation may reuse the same
  // opaque VkBuffer handle. Raw-handle descriptor caches cannot distinguish
  // that new object from the destroyed object, so effect-owned resource
  // reconfiguration must invalidate every utility descriptor tuple before its
  // next dispatch. Stable frame resources still avoid descriptor updates on
  // subsequent frames.
  void InvalidateDescriptorBindingCacheForSetup() noexcept;

  bool CropResizeBilinear(const VulkanImage &src, const VulkanImage &dst,
                          float crop_x, float crop_y, float crop_w,
                          float crop_h, std::string *error_out);
  // Exact out-of-place horizontal reversal. This intentionally reuses the
  // existing crop/resize opcode with integer sample coordinates; it does not
  // interpolate, change channel order, or permit src/dst aliasing.
  bool MirrorHorizontalU8x3(const VulkanImage &src, const VulkanImage &dst,
                            std::string *error_out);
  // Resizes src into scratch, then horizontally reverses scratch into dst in
  // one command buffer and one synchronous completion. scratch/dst have final
  // output geometry and all three images remain device resident.
  bool ResizeMirrorHorizontalU8x3(const VulkanImage &src,
                                  const VulkanImage &scratch,
                                  const VulkanImage &dst,
                                  std::string *error_out);
  bool ResizeBilinear(const VulkanImage &src, const VulkanImage &dst,
                      std::string *error_out);

  bool PreprocessToTensor(const VulkanImage &src, const VulkanTensor &dst,
                          const ModelPreprocessSpec &spec,
                          std::string *error_out);

  bool ResizeBilinearF32_1(const VulkanImage &src, const VulkanImage &dst,
                           std::string *error_out);

  // Setup/reconfiguration-only RGB upload seam. Packs tightly packed RGB24
  // into matching rgb_u8 staging/destination images, copies it into an
  // exact-shape non-mapped DEVICE_LOCAL image, and completes the transfer
  // before return.
  // Neither allocation nor filesystem/image decoding belongs here.
  bool UploadRgb24ToDeviceLocal(const std::uint8_t *src,
                                std::size_t src_stride,
                                const VulkanImage &upload_staging,
                                const VulkanImage &device_dst,
                                std::string *error_out);

  // Equivalent explicit-channel-order seam used by production RGB/BGR pixel
  // effects and parity fixtures. The source bytes are already in the matching
  // staging/destination memory channel order; no channel reinterpretation is
  // performed.
  bool UploadU8x3ToDeviceLocal(const std::uint8_t *src,
                               std::size_t src_stride,
                               const VulkanImage &upload_staging,
                               const VulkanImage &device_dst,
                               std::string *error_out);

  // Setup/reconfiguration-only scalar upload seam. Copies exactly one
  // width*height float plane through a caller-owned mapped staging image into
  // a distinct non-mapped DEVICE_LOCAL image, and completes the transfer
  // before return.
  bool UploadF32_1ToDeviceLocal(const float *src, std::size_t src_count,
                                const VulkanImage &upload_staging,
                                const VulkanImage &device_dst,
                                std::string *error_out);

  // Final-output boundary seam. Copies a non-mapped DEVICE_LOCAL RGB/BGR
  // image into one caller-owned mapped staging image, waits for completion,
  // invalidates it, and unpacks the three stored channels into caller-owned
  // CPU memory. This is a final transport operation, not an intermediate
  // effect readback.
  bool ReadbackU8x3(const VulkanImage &src,
                    const VulkanImage &readback_staging, std::uint8_t *dst,
                    std::size_t dst_stride, std::string *error_out);

  // Explicit degraded CPU-tail seam. Copies a device-local, non-mapped alpha
  // image into one caller-owned reusable host-visible staging image, waits for
  // completion, invalidates the staging memory, and copies exactly
  // width*height floats into preallocated CPU storage.
  bool ReadbackF32_1(const VulkanImage &src,
                     const VulkanImage &readback_staging, float *dst,
                     std::size_t dst_count, std::string *error_out);

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

  // Applies the final fixed-center vignette from a device-resident attenuation
  // factor mask. If resize_scratch is supplied, src is first resized to final
  // output geometry. If mirrored_out is supplied, the vignetted result is then
  // horizontally mirrored. Every requested stage is recorded into one command
  // buffer and completed with one bounded queue submission/fence wait.
  bool ApplyFinalVignetteU8x3(const VulkanImage &src,
                              const VulkanImage *resize_scratch,
                              const VulkanImage &attenuation_factor,
                              const VulkanImage &vignette_out,
                              const VulkanImage *mirrored_out,
                              std::string *error_out);

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
  VulkanBuffer final_params_;
  VulkanBuffer dummy_;

  VkDescriptorSetLayout descriptor_set_layout_ = nullptr;
  VkDescriptorPool descriptor_pool_ = nullptr;
  VkDescriptorSet descriptor_set_ = nullptr;
  VkDescriptorSet batch_descriptor_set_ = nullptr;
  VkDescriptorSet final_descriptor_set_ = nullptr;
  VkShaderModule shader_module_ = nullptr;
  VkPipelineLayout pipeline_layout_ = nullptr;
  VkPipeline pipeline_ = nullptr;
  VkCommandBuffer command_buffer_ = nullptr;
  VulkanCommandBatch frame_batch_;
  std::array<VkBuffer, 7> bound_buffers_{};
  std::array<VkBuffer, 7> batch_bound_buffers_{};
  std::array<VkBuffer, 7> final_bound_buffers_{};

  bool initialized_ = false;
  bool pipeline_created_ = false;
  std::atomic<std::uint64_t> synchronous_submission_count_{0};
  std::atomic<std::uint64_t> descriptor_binding_update_count_{0};
  std::string init_error_;
  mutable std::recursive_mutex execution_mutex_;
};

bool IsUtilityKernelsAvailable(std::string *error_out);

} // namespace studiocast::vulkan::kernels
