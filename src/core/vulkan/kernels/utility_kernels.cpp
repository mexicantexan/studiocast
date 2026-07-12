#include "core/vulkan/kernels/utility_kernels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "core/vulkan/kernels/shaders/utility_kernels_spv.h"

namespace studiocast::vulkan::kernels {

namespace {

constexpr std::uint32_t kBlockX = 16;
constexpr std::uint32_t kBlockY = 16;

std::uint32_t CeilDiv(std::uint32_t x, std::uint32_t y) {
  return (x + y - 1u) / y;
}

bool ResultOk(VkResult result, const char *what, std::string *error_out) {
  if (result == VK_SUCCESS)
    return true;
  if (error_out)
    *error_out = std::string(what) + " failed: " + VkResultName(result);
  return false;
}

bool IsRgbOrBgrFormat(VulkanPixelFormat format) {
  return format == VulkanPixelFormat::rgb_u8 ||
         format == VulkanPixelFormat::bgr_u8;
}

bool ValidateRgbOrBgrImage(const VulkanImage &image, const char *what,
                           std::string *error_out) {
  if (!image.Valid()) {
    if (error_out)
      *error_out = std::string(what) + ": invalid Vulkan image.";
    return false;
  }
  if (!IsRgbOrBgrFormat(image.format())) {
    if (error_out)
      *error_out =
          std::string(what) + ": expected rgb_u8 or bgr_u8 image.";
    return false;
  }
  return true;
}

bool ValidateF32Image(const VulkanImage &image, const char *what,
                      std::string *error_out) {
  if (!image.Valid()) {
    if (error_out)
      *error_out = std::string(what) + ": invalid Vulkan image.";
    return false;
  }
  if (image.format() != VulkanPixelFormat::f32_1) {
    if (error_out)
      *error_out = std::string(what) + ": expected f32_1 image.";
    return false;
  }
  return true;
}

bool SameDimensions(const VulkanImage &a, const VulkanImage &b) {
  return a.width() == b.width() && a.height() == b.height();
}

bool SameU8Format(const VulkanImage &a, const VulkanImage &b) {
  return a.format() == b.format();
}

VkBuffer OrDummy(VkBuffer buffer, VkBuffer dummy) {
  return buffer ? buffer : dummy;
}

} // namespace

struct UtilityKernels::Params {
  std::uint32_t src_w = 0;
  std::uint32_t src_h = 0;
  std::uint32_t src_pitch = 0;
  std::uint32_t src1_pitch = 0;
  std::uint32_t alpha_pitch = 0;
  std::uint32_t dst_w = 0;
  std::uint32_t dst_h = 0;
  std::uint32_t dst_pitch = 0;
  std::uint32_t scratch_pitch = 0;
  std::uint32_t src_is_bgr = 0;
  std::uint32_t dst_is_bgr = 0;
  std::int32_t radius = 0;
  float crop_x = 0.0f;
  float crop_y = 0.0f;
  float crop_w = 0.0f;
  float crop_h = 0.0f;
  float mean0 = 0.0f;
  float mean1 = 0.0f;
  float mean2 = 0.0f;
  float inv_std0 = 1.0f;
  float inv_std1 = 1.0f;
  float inv_std2 = 1.0f;
  std::uint32_t bg0 = 0;
  std::uint32_t bg1 = 0;
  std::uint32_t bg2 = 0;
  std::uint32_t reserved0 = 0;
  float target_r = 0.0f;
  float target_g = 0.0f;
  float target_b = 0.0f;
  float intensity = 0.0f;
  float direction = 0.0f;
};

bool detail::CheckBoxBlurRadiusForKernel(int radius, const char *what,
                                         std::string *error_out) {
  if (radius > kBoxBlurMaxRadius) {
    if (error_out) {
      const char *label = what ? what : "Vulkan box blur";
      *error_out = std::string(label) +
                   ": radius exceeds maximum supported radius " +
                   std::to_string(kBoxBlurMaxRadius) + ".";
    }
    return false;
  }
  return true;
}

float detail::ClampAlpha01ForComposite(float alpha) {
  if (!std::isfinite(alpha))
    return 0.0f;
  return alpha < 0.0f ? 0.0f : (alpha > 1.0f ? 1.0f : alpha);
}

std::array<std::uint8_t, 3>
detail::SolidBackgroundMemoryChannels(VulkanPixelFormat format,
                                      std::uint8_t bg_r, std::uint8_t bg_g,
                                      std::uint8_t bg_b) {
  if (format == VulkanPixelFormat::bgr_u8)
    return {bg_b, bg_g, bg_r};
  return {bg_r, bg_g, bg_b};
}

UtilityKernels::~UtilityKernels() { Shutdown(); }

bool UtilityKernels::Initialize(std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (initialized_)
    return true;

  std::string e;
  if (!device_.Initialize(&e) || !EnsurePipeline(&e) ||
      !params_.Allocate(&device_, sizeof(Params),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        /*map_memory=*/true, &e) ||
      !batch_params_.Allocate(&device_, sizeof(Params),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              /*map_memory=*/true, &e) ||
      !dummy_.Allocate(&device_, 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       /*required_memory_flags=*/0,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       /*map_memory=*/false, &e) ||
      !EnsureDescriptors(&e) || !EnsureCommandBuffer(&e)) {
    init_error_ = e;
    if (error_out)
      *error_out = e;
    Shutdown();
    return false;
  }

  initialized_ = true;
  init_error_.clear();
  return true;
}

void UtilityKernels::Shutdown() noexcept {
  const bool have_device = device_.Initialized();
  const auto &vf = device_.f();
  if (have_device && vf.vkDeviceWaitIdle)
    (void)vf.vkDeviceWaitIdle(device_.device());

  if (have_device && command_buffer_ && vf.vkFreeCommandBuffers) {
    vf.vkFreeCommandBuffers(device_.device(), device_.command_pool(), 1,
                            &command_buffer_);
  }
  command_buffer_ = nullptr;
  if (have_device && pipeline_ && vf.vkDestroyPipeline)
    vf.vkDestroyPipeline(device_.device(), pipeline_, nullptr);
  pipeline_ = nullptr;
  if (have_device && pipeline_layout_ && vf.vkDestroyPipelineLayout)
    vf.vkDestroyPipelineLayout(device_.device(), pipeline_layout_, nullptr);
  pipeline_layout_ = nullptr;
  if (have_device && shader_module_ && vf.vkDestroyShaderModule)
    vf.vkDestroyShaderModule(device_.device(), shader_module_, nullptr);
  shader_module_ = nullptr;
  if (have_device && descriptor_pool_ && vf.vkDestroyDescriptorPool)
    vf.vkDestroyDescriptorPool(device_.device(), descriptor_pool_, nullptr);
  descriptor_pool_ = nullptr;
  descriptor_set_ = nullptr;
  batch_descriptor_set_ = nullptr;
  if (have_device && descriptor_set_layout_ && vf.vkDestroyDescriptorSetLayout)
    vf.vkDestroyDescriptorSetLayout(device_.device(), descriptor_set_layout_,
                                    nullptr);
  descriptor_set_layout_ = nullptr;

  dummy_.Free();
  batch_params_.Free();
  params_.Free();
  device_.Shutdown();
  bound_buffers_.fill(nullptr);
  batch_bound_buffers_.fill(nullptr);
  initialized_ = false;
  pipeline_created_ = false;
  synchronous_submission_count_ = 0;
}

OpenVulkanDiagnostics UtilityKernels::Diagnostics() const {
  OpenVulkanDiagnostics d = device_.diagnostics();
  d.shader_pipeline_created = pipeline_created_;
  d.ok = d.ok && pipeline_created_;
  if (!init_error_.empty() && d.error.empty())
    d.error = init_error_;
  return d;
}

bool UtilityKernels::EnsurePipeline(std::string *error_out) {
  if (pipeline_)
    return true;

  const auto &vf = device_.f();
  VkShaderModuleCreateInfo shader{};
  shader.codeSize = shaders::kUtilityKernelsSpirvSizeBytes;
  shader.pCode = shaders::kUtilityKernelsSpirv;
  VkResult result = vf.vkCreateShaderModule(device_.device(), &shader, nullptr,
                                            &shader_module_);
  if (!ResultOk(result, "vkCreateShaderModule", error_out))
    return false;

  std::array<VkDescriptorSetLayoutBinding, 7> bindings{};
  for (std::uint32_t i = 0; i < bindings.size(); ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
  layout_info.pBindings = bindings.data();
  result = vf.vkCreateDescriptorSetLayout(device_.device(), &layout_info,
                                          nullptr, &descriptor_set_layout_);
  if (!ResultOk(result, "vkCreateDescriptorSetLayout", error_out))
    return false;

  VkPushConstantRange push{};
  push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push.offset = 0;
  push.size = sizeof(std::uint32_t);

  VkPipelineLayoutCreateInfo pipeline_layout{};
  pipeline_layout.setLayoutCount = 1;
  pipeline_layout.pSetLayouts = &descriptor_set_layout_;
  pipeline_layout.pushConstantRangeCount = 1;
  pipeline_layout.pPushConstantRanges = &push;
  result = vf.vkCreatePipelineLayout(device_.device(), &pipeline_layout,
                                     nullptr, &pipeline_layout_);
  if (!ResultOk(result, "vkCreatePipelineLayout", error_out))
    return false;

  VkComputePipelineCreateInfo compute{};
  compute.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  compute.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  compute.stage.module = shader_module_;
  compute.stage.pName = "main";
  compute.layout = pipeline_layout_;
  result = vf.vkCreateComputePipelines(device_.device(), nullptr, 1, &compute,
                                       nullptr, &pipeline_);
  if (!ResultOk(result, "vkCreateComputePipelines", error_out))
    return false;

  pipeline_created_ = true;
  return true;
}

bool UtilityKernels::EnsureDescriptors(std::string *error_out) {
  if (descriptor_set_)
    return true;

  const auto &vf = device_.f();
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = 14;
  VkDescriptorPoolCreateInfo pool{};
  pool.maxSets = 2;
  pool.poolSizeCount = 1;
  pool.pPoolSizes = &pool_size;
  VkResult result = vf.vkCreateDescriptorPool(device_.device(), &pool, nullptr,
                                              &descriptor_pool_);
  if (!ResultOk(result, "vkCreateDescriptorPool", error_out))
    return false;

  VkDescriptorSetAllocateInfo alloc{};
  alloc.descriptorPool = descriptor_pool_;
  const std::array<VkDescriptorSetLayout, 2> layouts = {descriptor_set_layout_,
                                                        descriptor_set_layout_};
  std::array<VkDescriptorSet, 2> descriptor_sets{};
  alloc.descriptorSetCount = static_cast<std::uint32_t>(layouts.size());
  alloc.pSetLayouts = layouts.data();
  result = vf.vkAllocateDescriptorSets(device_.device(), &alloc,
                                       descriptor_sets.data());
  if (!ResultOk(result, "vkAllocateDescriptorSets", error_out))
    return false;
  descriptor_set_ = descriptor_sets[0];
  batch_descriptor_set_ = descriptor_sets[1];

  return BindBuffers(dummy_.buffer(), dummy_.buffer(), dummy_.buffer(),
                     dummy_.buffer(), dummy_.buffer(), dummy_.buffer(),
                     error_out) &&
         BindBuffersForSet(batch_descriptor_set_, batch_params_.buffer(),
                           &batch_bound_buffers_, dummy_.buffer(),
                           dummy_.buffer(), dummy_.buffer(), dummy_.buffer(),
                           dummy_.buffer(), dummy_.buffer(), error_out);
}

bool UtilityKernels::EnsureCommandBuffer(std::string *error_out) {
  if (command_buffer_)
    return true;
  VkCommandBufferAllocateInfo alloc{};
  alloc.commandPool = device_.command_pool();
  alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 1;
  const VkResult result = device_.f().vkAllocateCommandBuffers(
      device_.device(), &alloc, &command_buffer_);
  return ResultOk(result, "vkAllocateCommandBuffers", error_out);
}

bool UtilityKernels::BindBuffersForSet(VkDescriptorSet descriptor_set,
                                       VkBuffer params_buffer,
                                       std::array<VkBuffer, 7> *bound_buffers,
                                       VkBuffer u8_src0, VkBuffer u8_src1,
                                       VkBuffer f32_src0, VkBuffer u8_out,
                                       VkBuffer f32_out, VkBuffer scratch,
                                       std::string *error_out) {
  if (!descriptor_set || !bound_buffers) {
    if (error_out)
      *error_out = "Vulkan utility descriptor set is not allocated.";
    return false;
  }
  const std::array<VkBuffer, 7> next = {
      OrDummy(u8_src0, dummy_.buffer()),
      OrDummy(u8_src1, dummy_.buffer()),
      OrDummy(f32_src0, dummy_.buffer()),
      OrDummy(u8_out, dummy_.buffer()),
      OrDummy(f32_out, dummy_.buffer()),
      OrDummy(scratch, dummy_.buffer()),
      params_buffer,
  };
  if (next == *bound_buffers)
    return true;

  std::array<VkDescriptorBufferInfo, 7> infos{};
  std::array<VkWriteDescriptorSet, 7> writes{};
  for (std::uint32_t i = 0; i < next.size(); ++i) {
    infos[i].buffer = next[i];
    infos[i].offset = 0;
    infos[i].range = VK_WHOLE_SIZE;
    writes[i].dstSet = descriptor_set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  device_.f().vkUpdateDescriptorSets(device_.device(),
                                     static_cast<std::uint32_t>(writes.size()),
                                     writes.data(), 0, nullptr);
  *bound_buffers = next;
  return true;
}

bool UtilityKernels::BindBuffers(VkBuffer u8_src0, VkBuffer u8_src1,
                                 VkBuffer f32_src0, VkBuffer u8_out,
                                 VkBuffer f32_out, VkBuffer scratch,
                                 std::string *error_out) {
  return BindBuffersForSet(descriptor_set_, params_.buffer(), &bound_buffers_,
                           u8_src0, u8_src1, f32_src0, u8_out, f32_out, scratch,
                           error_out);
}

bool UtilityKernels::Dispatch(const Params &params, Op op,
                              std::uint32_t dispatch_w,
                              std::uint32_t dispatch_h,
                              std::string *error_out) {
  return DispatchTwoPass(params, op, op, dispatch_w, dispatch_h, error_out);
}

bool UtilityKernels::DispatchTwoPass(const Params &params, Op first, Op second,
                                     std::uint32_t dispatch_w,
                                     std::uint32_t dispatch_h,
                                     std::string *error_out) {
  if (!initialized_) {
    if (error_out)
      *error_out = init_error_.empty()
                       ? "Vulkan utility kernels are not initialized."
                       : init_error_;
    return false;
  }
  if (!params_.mapped()) {
    if (error_out)
      *error_out = "Vulkan utility params buffer is not mapped.";
    return false;
  }
  std::memcpy(params_.mapped(), &params, sizeof(params));
  if (!params_.Flush(error_out))
    return false;

  const auto &vf = device_.f();
  VkResult result = vf.vkResetCommandBuffer(command_buffer_, 0);
  if (!ResultOk(result, "vkResetCommandBuffer", error_out))
    return false;

  VkCommandBufferBeginInfo begin{};
  result = vf.vkBeginCommandBuffer(command_buffer_, &begin);
  if (!ResultOk(result, "vkBeginCommandBuffer", error_out))
    return false;

  std::array<VkBufferMemoryBarrier, 4> read_barriers{};
  const VkBuffer read_buffers[] = {bound_buffers_[0], bound_buffers_[1],
                                   bound_buffers_[2], params_.buffer()};
  for (std::size_t i = 0; i < read_barriers.size(); ++i) {
    read_barriers[i].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    read_barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    read_barriers[i].buffer = read_buffers[i];
    read_barriers[i].size = VK_WHOLE_SIZE;
  }
  vf.vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                          static_cast<std::uint32_t>(read_barriers.size()),
                          read_barriers.data(), 0, nullptr);

  vf.vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                       pipeline_);
  vf.vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline_layout_, 0, 1, &descriptor_set_, 0,
                             nullptr);

  auto dispatch_op = [&](Op op_value) {
    const std::uint32_t op_u32 = static_cast<std::uint32_t>(op_value);
    vf.vkCmdPushConstants(command_buffer_, pipeline_layout_,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(op_u32),
                          &op_u32);
    vf.vkCmdDispatch(command_buffer_, CeilDiv(dispatch_w, kBlockX),
                     CeilDiv(dispatch_h, kBlockY), 1);
  };

  dispatch_op(first);
  if (second != first) {
    VkBufferMemoryBarrier scratch_ready{};
    scratch_ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    scratch_ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    scratch_ready.buffer = bound_buffers_[5];
    scratch_ready.size = VK_WHOLE_SIZE;
    vf.vkCmdPipelineBarrier(command_buffer_,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                            nullptr, 1, &scratch_ready, 0, nullptr);
    dispatch_op(second);
  }

  std::array<VkBufferMemoryBarrier, 3> host_barriers{};
  for (std::size_t i = 0; i < host_barriers.size(); ++i) {
    host_barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    host_barriers[i].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    host_barriers[i].buffer = bound_buffers_[3 + i];
    host_barriers[i].size = VK_WHOLE_SIZE;
  }
  vf.vkCmdPipelineBarrier(command_buffer_,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr,
                          static_cast<std::uint32_t>(host_barriers.size()),
                          host_barriers.data(), 0, nullptr);

  result = vf.vkEndCommandBuffer(command_buffer_);
  if (!ResultOk(result, "vkEndCommandBuffer", error_out))
    return false;

  return SubmitRecorded(error_out);
}

bool UtilityKernels::SubmitRecorded(std::string *error_out) {
  ++synchronous_submission_count_;
  return device_.SubmitAndWait(command_buffer_, error_out);
}

bool UtilityKernels::CropResizeBilinear(const VulkanImage &src,
                                        const VulkanImage &dst, float crop_x,
                                        float crop_y, float crop_w,
                                        float crop_h,
                                        std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!ValidateRgbOrBgrImage(src, "CropResizeBilinear(src)", error_out) ||
      !ValidateRgbOrBgrImage(dst, "CropResizeBilinear(dst)", error_out))
    return false;
  if (src.format() != dst.format()) {
    if (error_out)
      *error_out = "CropResizeBilinear: src/dst formats must match.";
    return false;
  }
  if (!std::isfinite(crop_x) || !std::isfinite(crop_y) ||
      !std::isfinite(crop_w) || !std::isfinite(crop_h)) {
    if (error_out)
      *error_out =
          "CropResizeBilinear: crop rectangle contains non-finite values.";
    return false;
  }
  if (!Initialize(error_out))
    return false;

  crop_w = std::clamp(crop_w, 1.0f, static_cast<float>(src.width()));
  crop_h = std::clamp(crop_h, 1.0f, static_cast<float>(src.height()));
  crop_x = std::clamp(crop_x, 0.0f, static_cast<float>(src.width()) - crop_w);
  crop_y = std::clamp(crop_y, 0.0f, static_cast<float>(src.height()) - crop_h);

  Params p{};
  p.src_w = static_cast<std::uint32_t>(src.width());
  p.src_h = static_cast<std::uint32_t>(src.height());
  p.src_pitch = static_cast<std::uint32_t>(src.pitch_pixels());
  p.dst_w = static_cast<std::uint32_t>(dst.width());
  p.dst_h = static_cast<std::uint32_t>(dst.height());
  p.dst_pitch = static_cast<std::uint32_t>(dst.pitch_pixels());
  p.crop_x = crop_x;
  p.crop_y = crop_y;
  p.crop_w = crop_w;
  p.crop_h = crop_h;

  if (!BindBuffers(src.buffer(), nullptr, nullptr, dst.buffer(), nullptr,
                   nullptr, error_out))
    return false;
  return Dispatch(p, Op::crop_resize_u8x3, p.dst_w, p.dst_h, error_out);
}

bool UtilityKernels::ResizeBilinear(const VulkanImage &src,
                                    const VulkanImage &dst,
                                    std::string *error_out) {
  return CropResizeBilinear(src, dst, 0.0f, 0.0f,
                            static_cast<float>(src.width()),
                            static_cast<float>(src.height()), error_out);
}

bool UtilityKernels::PreprocessToTensor(const VulkanImage &src,
                                        const VulkanTensor &dst,
                                        const ModelPreprocessSpec &spec,
                                        std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!ValidateRgbOrBgrImage(src, "PreprocessToTensor(src)", error_out))
    return false;
  if (!dst.Valid()) {
    if (error_out)
      *error_out = "PreprocessToTensor: invalid dst tensor.";
    return false;
  }
  if (spec.dst_w <= 0 || spec.dst_h <= 0) {
    if (error_out)
      *error_out = "PreprocessToTensor: invalid dst size in spec.";
    return false;
  }
  if (dst.n() != 1 || dst.c() != 3 || dst.h() != spec.dst_h ||
      dst.w() != spec.dst_w) {
    if (error_out)
      *error_out = "PreprocessToTensor: dst tensor shape mismatch (expected "
                   "N=1,C=3,H=spec.dst_h,W=spec.dst_w).";
    return false;
  }
  if (spec.std[0] == 0.0f || spec.std[1] == 0.0f || spec.std[2] == 0.0f) {
    if (error_out)
      *error_out = "PreprocessToTensor: std contains zero.";
    return false;
  }
  if (!Initialize(error_out))
    return false;

  Params p{};
  p.src_w = static_cast<std::uint32_t>(src.width());
  p.src_h = static_cast<std::uint32_t>(src.height());
  p.src_pitch = static_cast<std::uint32_t>(src.pitch_pixels());
  p.dst_w = static_cast<std::uint32_t>(spec.dst_w);
  p.dst_h = static_cast<std::uint32_t>(spec.dst_h);
  p.src_is_bgr = src.format() == VulkanPixelFormat::bgr_u8 ? 1u : 0u;
  p.dst_is_bgr = spec.dst_order == ChannelOrder::bgr ? 1u : 0u;
  p.mean0 = spec.mean[0];
  p.mean1 = spec.mean[1];
  p.mean2 = spec.mean[2];
  p.inv_std0 = 1.0f / spec.std[0];
  p.inv_std1 = 1.0f / spec.std[1];
  p.inv_std2 = 1.0f / spec.std[2];

  if (!BindBuffers(src.buffer(), nullptr, nullptr, nullptr, dst.buffer(),
                   nullptr, error_out))
    return false;
  return Dispatch(p, Op::preprocess_to_nchw, p.dst_w, p.dst_h, error_out);
}

bool UtilityKernels::ResizeBilinearF32_1(const VulkanImage &src,
                                         const VulkanImage &dst,
                                         std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!ValidateF32Image(src, "ResizeBilinearF32_1(src)", error_out) ||
      !ValidateF32Image(dst, "ResizeBilinearF32_1(dst)", error_out))
    return false;
  if (!Initialize(error_out))
    return false;

  Params p{};
  p.src_w = static_cast<std::uint32_t>(src.width());
  p.src_h = static_cast<std::uint32_t>(src.height());
  p.src_pitch = static_cast<std::uint32_t>(src.pitch_pixels());
  p.dst_w = static_cast<std::uint32_t>(dst.width());
  p.dst_h = static_cast<std::uint32_t>(dst.height());
  p.dst_pitch = static_cast<std::uint32_t>(dst.pitch_pixels());

  if (!BindBuffers(nullptr, nullptr, src.buffer(), nullptr, dst.buffer(),
                   nullptr, error_out))
    return false;
  return Dispatch(p, Op::resize_f32_1, p.dst_w, p.dst_h, error_out);
}

bool UtilityKernels::BoxBlurSeparableU8x3(const VulkanImage &src,
                                          const VulkanImage &tmp,
                                          const VulkanImage &dst, int radius,
                                          std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!ValidateRgbOrBgrImage(src, "BoxBlurSeparableU8x3(src)", error_out) ||
      !ValidateRgbOrBgrImage(tmp, "BoxBlurSeparableU8x3(tmp)", error_out) ||
      !ValidateRgbOrBgrImage(dst, "BoxBlurSeparableU8x3(dst)", error_out))
    return false;
  if (!SameDimensions(src, tmp) || !SameDimensions(src, dst)) {
    if (error_out)
      *error_out = "BoxBlurSeparableU8x3: dimension mismatch.";
    return false;
  }
  if (!SameU8Format(src, tmp) || !SameU8Format(src, dst)) {
    if (error_out)
      *error_out = "BoxBlurSeparableU8x3: src/tmp/dst formats must match.";
    return false;
  }
  if (!detail::CheckBoxBlurRadiusForKernel(radius, "BoxBlurSeparableU8x3",
                                           error_out))
    return false;
  radius = detail::NormalizeBoxBlurRadius(radius);
  if (!Initialize(error_out))
    return false;

  Params p{};
  p.src_w = static_cast<std::uint32_t>(src.width());
  p.src_h = static_cast<std::uint32_t>(src.height());
  p.src_pitch = static_cast<std::uint32_t>(src.pitch_pixels());
  p.dst_w = p.src_w;
  p.dst_h = p.src_h;
  p.dst_pitch = static_cast<std::uint32_t>(dst.pitch_pixels());
  p.scratch_pitch = static_cast<std::uint32_t>(tmp.pitch_pixels());
  p.radius = radius;

  if (!BindBuffers(src.buffer(), nullptr, nullptr, dst.buffer(), nullptr,
                   tmp.buffer(), error_out))
    return false;
  if (radius == 0)
    return Dispatch(p, Op::copy_u8x3, p.dst_w, p.dst_h, error_out);
  return DispatchTwoPass(p, Op::blur_h_u8x3, Op::blur_v_u8x3, p.dst_w,
                         p.dst_h, error_out);
}

bool UtilityKernels::BoxBlurSeparableF32_1(const VulkanImage &src,
                                           const VulkanImage &tmp,
                                           const VulkanImage &dst, int radius,
                                           std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!ValidateF32Image(src, "BoxBlurSeparableF32_1(src)", error_out) ||
      !ValidateF32Image(tmp, "BoxBlurSeparableF32_1(tmp)", error_out) ||
      !ValidateF32Image(dst, "BoxBlurSeparableF32_1(dst)", error_out))
    return false;
  if (!SameDimensions(src, tmp) || !SameDimensions(src, dst)) {
    if (error_out)
      *error_out = "BoxBlurSeparableF32_1: dimension mismatch.";
    return false;
  }
  if (!detail::CheckBoxBlurRadiusForKernel(radius, "BoxBlurSeparableF32_1",
                                           error_out))
    return false;
  radius = detail::NormalizeBoxBlurRadius(radius);
  if (!Initialize(error_out))
    return false;

  Params p{};
  p.src_w = static_cast<std::uint32_t>(src.width());
  p.src_h = static_cast<std::uint32_t>(src.height());
  p.src_pitch = static_cast<std::uint32_t>(src.pitch_pixels());
  p.dst_w = p.src_w;
  p.dst_h = p.src_h;
  p.dst_pitch = static_cast<std::uint32_t>(dst.pitch_pixels());
  p.scratch_pitch = static_cast<std::uint32_t>(tmp.pitch_pixels());
  p.radius = radius;

  if (!BindBuffers(nullptr, nullptr, src.buffer(), nullptr, dst.buffer(),
                   tmp.buffer(), error_out))
    return false;
  if (radius == 0)
    return Dispatch(p, Op::copy_f32_1, p.dst_w, p.dst_h, error_out);
  return DispatchTwoPass(p, Op::blur_h_f32_1, Op::blur_v_f32_1, p.dst_w,
                         p.dst_h, error_out);
}

bool UtilityKernels::CompositeAlphaU8x3(const VulkanImage &fg,
                                        const VulkanImage &bg,
                                        const VulkanImage &alpha,
                                        const VulkanImage &out,
                                        std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!ValidateRgbOrBgrImage(fg, "CompositeAlphaU8x3(fg)", error_out) ||
      !ValidateRgbOrBgrImage(bg, "CompositeAlphaU8x3(bg)", error_out) ||
      !ValidateF32Image(alpha, "CompositeAlphaU8x3(alpha)", error_out) ||
      !ValidateRgbOrBgrImage(out, "CompositeAlphaU8x3(out)", error_out))
    return false;
  if (!SameDimensions(fg, bg) || !SameDimensions(fg, alpha) ||
      !SameDimensions(fg, out)) {
    if (error_out)
      *error_out = "CompositeAlphaU8x3: dimension mismatch.";
    return false;
  }
  if (fg.format() != bg.format() || fg.format() != out.format()) {
    if (error_out)
      *error_out = "CompositeAlphaU8x3: fg/bg/out formats must match.";
    return false;
  }
  if (!Initialize(error_out))
    return false;

  Params p{};
  p.src_w = static_cast<std::uint32_t>(fg.width());
  p.src_h = static_cast<std::uint32_t>(fg.height());
  p.src_pitch = static_cast<std::uint32_t>(fg.pitch_pixels());
  p.src1_pitch = static_cast<std::uint32_t>(bg.pitch_pixels());
  p.alpha_pitch = static_cast<std::uint32_t>(alpha.pitch_pixels());
  p.dst_w = p.src_w;
  p.dst_h = p.src_h;
  p.dst_pitch = static_cast<std::uint32_t>(out.pitch_pixels());

  if (!BindBuffers(fg.buffer(), bg.buffer(), alpha.buffer(), out.buffer(),
                   nullptr, nullptr, error_out))
    return false;
  return Dispatch(p, Op::composite_bg, p.dst_w, p.dst_h, error_out);
}

bool UtilityKernels::BoxBlurCompositeAlphaU8x3(
    const VulkanImage &fg, const VulkanImage &blur_tmp,
    const VulkanImage &blurred, const VulkanImage &alpha,
    const VulkanImage &out, int radius, std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!ValidateRgbOrBgrImage(fg, "BoxBlurCompositeAlphaU8x3(fg)", error_out) ||
      !ValidateRgbOrBgrImage(blur_tmp, "BoxBlurCompositeAlphaU8x3(blur_tmp)",
                             error_out) ||
      !ValidateRgbOrBgrImage(blurred, "BoxBlurCompositeAlphaU8x3(blurred)",
                             error_out) ||
      !ValidateF32Image(alpha, "BoxBlurCompositeAlphaU8x3(alpha)", error_out) ||
      !ValidateRgbOrBgrImage(out, "BoxBlurCompositeAlphaU8x3(out)",
                             error_out)) {
    return false;
  }
  if (!SameDimensions(fg, blur_tmp) || !SameDimensions(fg, blurred) ||
      !SameDimensions(fg, alpha) || !SameDimensions(fg, out)) {
    if (error_out)
      *error_out = "BoxBlurCompositeAlphaU8x3: dimension mismatch.";
    return false;
  }
  if (fg.format() != blur_tmp.format() || fg.format() != blurred.format() ||
      fg.format() != out.format()) {
    if (error_out) {
      *error_out =
          "BoxBlurCompositeAlphaU8x3: fg/blur_tmp/blurred/out formats must "
          "match.";
    }
    return false;
  }
  if (!detail::CheckBoxBlurRadiusForKernel(radius, "BoxBlurCompositeAlphaU8x3",
                                           error_out)) {
    return false;
  }
  radius = detail::NormalizeBoxBlurRadius(radius);
  if (!Initialize(error_out))
    return false;

  Params blur_params{};
  blur_params.src_w = static_cast<std::uint32_t>(fg.width());
  blur_params.src_h = static_cast<std::uint32_t>(fg.height());
  blur_params.src_pitch = static_cast<std::uint32_t>(fg.pitch_pixels());
  blur_params.dst_w = blur_params.src_w;
  blur_params.dst_h = blur_params.src_h;
  blur_params.dst_pitch = static_cast<std::uint32_t>(blurred.pitch_pixels());
  blur_params.scratch_pitch =
      static_cast<std::uint32_t>(blur_tmp.pitch_pixels());
  blur_params.radius = radius;

  Params composite_params{};
  composite_params.src_w = blur_params.src_w;
  composite_params.src_h = blur_params.src_h;
  composite_params.src_pitch = static_cast<std::uint32_t>(fg.pitch_pixels());
  composite_params.src1_pitch =
      static_cast<std::uint32_t>(blurred.pitch_pixels());
  composite_params.alpha_pitch =
      static_cast<std::uint32_t>(alpha.pitch_pixels());
  composite_params.dst_w = blur_params.dst_w;
  composite_params.dst_h = blur_params.dst_h;
  composite_params.dst_pitch = static_cast<std::uint32_t>(out.pitch_pixels());

  std::memcpy(params_.mapped(), &blur_params, sizeof(blur_params));
  std::memcpy(batch_params_.mapped(), &composite_params,
              sizeof(composite_params));
  if (!params_.Flush(error_out) || !batch_params_.Flush(error_out))
    return false;
  if (!BindBuffers(fg.buffer(), nullptr, nullptr, blurred.buffer(), nullptr,
                   blur_tmp.buffer(), error_out) ||
      !BindBuffersForSet(batch_descriptor_set_, batch_params_.buffer(),
                         &batch_bound_buffers_, fg.buffer(), blurred.buffer(),
                         alpha.buffer(), out.buffer(), nullptr, nullptr,
                         error_out)) {
    return false;
  }

  const auto &vf = device_.f();
  VkResult result = vf.vkResetCommandBuffer(command_buffer_, 0);
  if (!ResultOk(result, "vkResetCommandBuffer", error_out))
    return false;
  VkCommandBufferBeginInfo begin{};
  result = vf.vkBeginCommandBuffer(command_buffer_, &begin);
  if (!ResultOk(result, "vkBeginCommandBuffer", error_out))
    return false;

  const VkBuffer inputs[] = {fg.buffer(), alpha.buffer(), params_.buffer(),
                             batch_params_.buffer()};
  std::array<VkBufferMemoryBarrier, 4> inputs_ready{};
  for (std::size_t i = 0; i < inputs_ready.size(); ++i) {
    inputs_ready[i].srcAccessMask =
        VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    inputs_ready[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    inputs_ready[i].buffer = inputs[i];
    inputs_ready[i].size = VK_WHOLE_SIZE;
  }
  vf.vkCmdPipelineBarrier(command_buffer_,
                          VK_PIPELINE_STAGE_HOST_BIT |
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                          static_cast<std::uint32_t>(inputs_ready.size()),
                          inputs_ready.data(), 0, nullptr);

  vf.vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                       pipeline_);
  vf.vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline_layout_, 0, 1, &descriptor_set_, 0,
                             nullptr);
  auto dispatch_op = [&](Op op) {
    const std::uint32_t op_u32 = static_cast<std::uint32_t>(op);
    vf.vkCmdPushConstants(command_buffer_, pipeline_layout_,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(op_u32),
                          &op_u32);
    vf.vkCmdDispatch(command_buffer_, CeilDiv(blur_params.dst_w, kBlockX),
                     CeilDiv(blur_params.dst_h, kBlockY), 1);
  };

  if (radius == 0) {
    dispatch_op(Op::copy_u8x3);
  } else {
    dispatch_op(Op::blur_h_u8x3);
    VkBufferMemoryBarrier scratch_ready{};
    scratch_ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    scratch_ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    scratch_ready.buffer = blur_tmp.buffer();
    scratch_ready.size = VK_WHOLE_SIZE;
    vf.vkCmdPipelineBarrier(command_buffer_,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                            1, &scratch_ready, 0, nullptr);
    dispatch_op(Op::blur_v_u8x3);
  }

  VkBufferMemoryBarrier blurred_ready{};
  blurred_ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  blurred_ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  blurred_ready.buffer = blurred.buffer();
  blurred_ready.size = VK_WHOLE_SIZE;
  vf.vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                          1, &blurred_ready, 0, nullptr);

  vf.vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline_layout_, 0, 1, &batch_descriptor_set_, 0,
                             nullptr);
  dispatch_op(Op::composite_bg);

  VkBufferMemoryBarrier out_ready{};
  out_ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  out_ready.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  out_ready.buffer = out.buffer();
  out_ready.size = VK_WHOLE_SIZE;
  vf.vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                          &out_ready, 0, nullptr);

  result = vf.vkEndCommandBuffer(command_buffer_);
  if (!ResultOk(result, "vkEndCommandBuffer", error_out))
    return false;
  return SubmitRecorded(error_out);
}

bool UtilityKernels::CompositeAlphaSolidU8x3(
    const VulkanImage &fg, const VulkanImage &alpha, std::uint8_t bg_r,
    std::uint8_t bg_g, std::uint8_t bg_b, const VulkanImage &out,
    std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!ValidateRgbOrBgrImage(fg, "CompositeAlphaSolidU8x3(fg)", error_out) ||
      !ValidateF32Image(alpha, "CompositeAlphaSolidU8x3(alpha)", error_out) ||
      !ValidateRgbOrBgrImage(out, "CompositeAlphaSolidU8x3(out)", error_out))
    return false;
  if (!SameDimensions(fg, alpha) || !SameDimensions(fg, out)) {
    if (error_out)
      *error_out = "CompositeAlphaSolidU8x3: dimension mismatch.";
    return false;
  }
  if (fg.format() != out.format()) {
    if (error_out)
      *error_out = "CompositeAlphaSolidU8x3: fg/out formats must match.";
    return false;
  }
  if (!Initialize(error_out))
    return false;

  const auto bg =
      detail::SolidBackgroundMemoryChannels(fg.format(), bg_r, bg_g, bg_b);
  Params p{};
  p.src_w = static_cast<std::uint32_t>(fg.width());
  p.src_h = static_cast<std::uint32_t>(fg.height());
  p.src_pitch = static_cast<std::uint32_t>(fg.pitch_pixels());
  p.alpha_pitch = static_cast<std::uint32_t>(alpha.pitch_pixels());
  p.dst_w = p.src_w;
  p.dst_h = p.src_h;
  p.dst_pitch = static_cast<std::uint32_t>(out.pitch_pixels());
  p.bg0 = bg[0];
  p.bg1 = bg[1];
  p.bg2 = bg[2];

  if (!BindBuffers(fg.buffer(), nullptr, alpha.buffer(), out.buffer(), nullptr,
                   nullptr, error_out))
    return false;
  return Dispatch(p, Op::composite_solid, p.dst_w, p.dst_h, error_out);
}

bool UtilityKernels::ApplyKeyLightU8x3(const VulkanImage &src,
                                       const VulkanImage &alpha,
                                       float target_r, float target_g,
                                       float target_b, float intensity01,
                                       float direction,
                                       const VulkanImage &out,
                                       std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!ValidateRgbOrBgrImage(src, "ApplyKeyLightU8x3(src)", error_out) ||
      !ValidateF32Image(alpha, "ApplyKeyLightU8x3(alpha)", error_out) ||
      !ValidateRgbOrBgrImage(out, "ApplyKeyLightU8x3(out)", error_out))
    return false;
  if (src.format() != VulkanPixelFormat::rgb_u8 ||
      out.format() != VulkanPixelFormat::rgb_u8) {
    if (error_out)
      *error_out = "ApplyKeyLightU8x3: src/out must be rgb_u8.";
    return false;
  }
  if (!SameDimensions(src, alpha) || !SameDimensions(src, out)) {
    if (error_out)
      *error_out = "ApplyKeyLightU8x3: dimension mismatch.";
    return false;
  }
  if (!Initialize(error_out))
    return false;

  Params p{};
  p.src_w = static_cast<std::uint32_t>(src.width());
  p.src_h = static_cast<std::uint32_t>(src.height());
  p.src_pitch = static_cast<std::uint32_t>(src.pitch_pixels());
  p.alpha_pitch = static_cast<std::uint32_t>(alpha.pitch_pixels());
  p.dst_w = p.src_w;
  p.dst_h = p.src_h;
  p.dst_pitch = static_cast<std::uint32_t>(out.pitch_pixels());
  p.target_r = std::clamp(target_r, 0.0f, 255.0f);
  p.target_g = std::clamp(target_g, 0.0f, 255.0f);
  p.target_b = std::clamp(target_b, 0.0f, 255.0f);
  p.intensity = std::clamp(intensity01, 0.0f, 1.0f);
  p.direction = std::clamp(direction, -1.0f, 1.0f);

  if (!BindBuffers(src.buffer(), nullptr, alpha.buffer(), out.buffer(),
                   nullptr, nullptr, error_out))
    return false;
  return Dispatch(p, Op::key_light, p.dst_w, p.dst_h, error_out);
}

bool IsUtilityKernelsAvailable(std::string *error_out) {
  UtilityKernels kernels;
  return kernels.Initialize(error_out);
}

} // namespace studiocast::vulkan::kernels
