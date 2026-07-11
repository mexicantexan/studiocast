#include "core/vulkan/kernels/resize_bilinear.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "core/vulkan/kernels/shaders/resize_rgb24_bilinear_spv.h"
#include "core/vulkan/kernels/utility_kernels.h"

namespace studiocast::vulkan::kernels {

namespace {

struct ResizePushConstants {
  std::uint32_t src_w = 0;
  std::uint32_t src_h = 0;
  std::uint32_t src_pitch_pixels = 0;
  std::uint32_t dst_w = 0;
  std::uint32_t dst_h = 0;
  std::uint32_t dst_pitch_pixels = 0;
};

bool ResultOk(VkResult result, const char *what, std::string *error_out) {
  if (result == VK_SUCCESS)
    return true;
  if (error_out)
    *error_out = std::string(what) + " failed: " + VkResultName(result);
  return false;
}

std::uint32_t CeilDiv(std::uint32_t x, std::uint32_t y) {
  return (x + y - 1u) / y;
}

bool ValidateDimensions(int src_w, int src_h, int dst_w, int dst_h,
                        std::string *error_out) {
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
    if (error_out)
      *error_out = "Vulkan resize dimensions must be positive.";
    return false;
  }
  constexpr int kMaxDimension = 32768;
  if (src_w > kMaxDimension || src_h > kMaxDimension || dst_w > kMaxDimension ||
      dst_h > kMaxDimension) {
    if (error_out)
      *error_out = "Vulkan resize dimensions are unreasonably large.";
    return false;
  }
  return true;
}

} // namespace

ResizeBilinear::~ResizeBilinear() { Shutdown(); }

bool ResizeBilinear::EnsureInitialized(int src_w, int src_h, int dst_w,
                                       int dst_h, std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (initialized_ && src_w_ == src_w && src_h_ == src_h && dst_w_ == dst_w &&
      dst_h_ == dst_h) {
    return true;
  }

  Shutdown();
  if (!ValidateDimensions(src_w, src_h, dst_w, dst_h, error_out)) {
    init_error_ = error_out ? *error_out : "Invalid Vulkan resize dimensions.";
    return false;
  }

  src_w_ = src_w;
  src_h_ = src_h;
  dst_w_ = dst_w;
  dst_h_ = dst_h;

  std::string e;
  if (!device_.Initialize(&e) || !CreateBuffers(&e) || !CreatePipeline(&e) ||
      !CreateDescriptors(&e) || !RecordCommandBuffer(&e)) {
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

bool ResizeBilinear::Resize(const std::uint8_t *src,
                            std::size_t src_stride_bytes, std::uint8_t *dst,
                            std::size_t dst_stride_bytes,
                            std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!initialized_) {
    if (error_out)
      *error_out = init_error_.empty()
                       ? "Vulkan resize runtime is not initialized."
                       : init_error_;
    return false;
  }
  if (!src || !dst) {
    if (error_out)
      *error_out = "Vulkan resize received a null CPU buffer.";
    return false;
  }
  if (src_stride_bytes < static_cast<std::size_t>(src_w_) * 3u ||
      dst_stride_bytes < static_cast<std::size_t>(dst_w_) * 3u) {
    if (error_out)
      *error_out = "Vulkan resize CPU stride is smaller than width * 3.";
    return false;
  }
  if (!upload_.mapped() || !readback_.mapped()) {
    if (error_out)
      *error_out = "Vulkan resize staging buffers are not mapped.";
    return false;
  }

  PackRgb24ToRgba32(src, src_stride_bytes, src_w_, src_h_,
                    static_cast<std::uint32_t *>(upload_.mapped()),
                    gpu_src_.pitch_pixels());
  if (!upload_.Flush(error_out))
    return false;

  if (!device_.SubmitAndWait(command_buffer_, error_out))
    return false;

  if (!readback_.Invalidate(error_out))
    return false;

  UnpackRgba32ToRgb24(static_cast<const std::uint32_t *>(readback_.mapped()),
                      gpu_dst_.pitch_pixels(), dst_w_, dst_h_, dst,
                      dst_stride_bytes);
  return true;
}

void ResizeBilinear::Shutdown() noexcept {
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
  if (have_device && descriptor_set_layout_ && vf.vkDestroyDescriptorSetLayout)
    vf.vkDestroyDescriptorSetLayout(device_.device(), descriptor_set_layout_,
                                    nullptr);
  descriptor_set_layout_ = nullptr;

  readback_.Free();
  upload_.Free();
  gpu_dst_.Free();
  gpu_src_.Free();
  device_.Shutdown();

  initialized_ = false;
  pipeline_created_ = false;
  src_w_ = src_h_ = dst_w_ = dst_h_ = 0;
}

OpenVulkanDiagnostics ResizeBilinear::Diagnostics() const {
  OpenVulkanDiagnostics d = device_.diagnostics();
  d.shader_pipeline_created = pipeline_created_;
  d.ok = d.ok && pipeline_created_;
  if (!init_error_.empty() && d.error.empty())
    d.error = init_error_;
  return d;
}

bool ResizeBilinear::CreateBuffers(std::string *error_out) {
  if (!gpu_src_.Allocate(&device_, src_w_, src_h_, error_out))
    return false;
  if (!gpu_dst_.Allocate(&device_, dst_w_, dst_h_, error_out))
    return false;

  if (!upload_.Allocate(&device_, gpu_src_.byte_size(),
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        /*map_memory=*/true, error_out)) {
    return false;
  }
  if (!readback_.Allocate(&device_, gpu_dst_.byte_size(),
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                          VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                          /*map_memory=*/true, error_out)) {
    return false;
  }
  return true;
}

bool ResizeBilinear::CreatePipeline(std::string *error_out) {
  const auto &vf = device_.f();

  VkShaderModuleCreateInfo shader{};
  shader.codeSize = shaders::kResizeRgb24BilinearSpirvSizeBytes;
  shader.pCode = shaders::kResizeRgb24BilinearSpirv;
  VkResult result = vf.vkCreateShaderModule(device_.device(), &shader, nullptr,
                                            &shader_module_);
  if (!ResultOk(result, "vkCreateShaderModule", error_out))
    return false;

  VkDescriptorSetLayoutBinding bindings[2]{};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[1] = bindings[0];
  bindings[1].binding = 1;

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.bindingCount = 2;
  layout_info.pBindings = bindings;
  result = vf.vkCreateDescriptorSetLayout(device_.device(), &layout_info,
                                          nullptr, &descriptor_set_layout_);
  if (!ResultOk(result, "vkCreateDescriptorSetLayout", error_out))
    return false;

  VkPushConstantRange push{};
  push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push.offset = 0;
  push.size = sizeof(ResizePushConstants);

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

bool ResizeBilinear::CreateDescriptors(std::string *error_out) {
  const auto &vf = device_.f();

  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = 2;
  VkDescriptorPoolCreateInfo pool{};
  pool.maxSets = 1;
  pool.poolSizeCount = 1;
  pool.pPoolSizes = &pool_size;
  VkResult result =
      vf.vkCreateDescriptorPool(device_.device(), &pool, nullptr,
                                &descriptor_pool_);
  if (!ResultOk(result, "vkCreateDescriptorPool", error_out))
    return false;

  VkDescriptorSetAllocateInfo alloc{};
  alloc.descriptorPool = descriptor_pool_;
  alloc.descriptorSetCount = 1;
  alloc.pSetLayouts = &descriptor_set_layout_;
  result = vf.vkAllocateDescriptorSets(device_.device(), &alloc,
                                       &descriptor_set_);
  if (!ResultOk(result, "vkAllocateDescriptorSets", error_out))
    return false;

  VkDescriptorBufferInfo infos[2]{};
  infos[0].buffer = gpu_src_.buffer();
  infos[0].offset = 0;
  infos[0].range = gpu_src_.byte_size();
  infos[1].buffer = gpu_dst_.buffer();
  infos[1].offset = 0;
  infos[1].range = gpu_dst_.byte_size();

  VkWriteDescriptorSet writes[2]{};
  writes[0].dstSet = descriptor_set_;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &infos[0];
  writes[1] = writes[0];
  writes[1].dstBinding = 1;
  writes[1].pBufferInfo = &infos[1];

  vf.vkUpdateDescriptorSets(device_.device(), 2, writes, 0, nullptr);
  return true;
}

bool ResizeBilinear::RecordCommandBuffer(std::string *error_out) {
  const auto &vf = device_.f();

  VkCommandBufferAllocateInfo alloc{};
  alloc.commandPool = device_.command_pool();
  alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 1;
  VkResult result =
      vf.vkAllocateCommandBuffers(device_.device(), &alloc, &command_buffer_);
  if (!ResultOk(result, "vkAllocateCommandBuffers", error_out))
    return false;

  VkCommandBufferBeginInfo begin{};
  begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  result = vf.vkBeginCommandBuffer(command_buffer_, &begin);
  if (!ResultOk(result, "vkBeginCommandBuffer", error_out))
    return false;

  VkBufferMemoryBarrier upload_barrier{};
  upload_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  upload_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  upload_barrier.buffer = upload_.buffer();
  upload_barrier.size = upload_.size();
  vf.vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                          &upload_barrier, 0, nullptr);

  VkBufferCopy src_copy{};
  src_copy.size = gpu_src_.byte_size();
  vf.vkCmdCopyBuffer(command_buffer_, upload_.buffer(), gpu_src_.buffer(), 1,
                     &src_copy);

  VkBufferMemoryBarrier src_ready{};
  src_ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  src_ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  src_ready.buffer = gpu_src_.buffer();
  src_ready.size = gpu_src_.byte_size();
  vf.vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                          1, &src_ready, 0, nullptr);

  vf.vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                       pipeline_);
  vf.vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline_layout_, 0, 1, &descriptor_set_, 0,
                             nullptr);

  ResizePushConstants push{};
  push.src_w = static_cast<std::uint32_t>(src_w_);
  push.src_h = static_cast<std::uint32_t>(src_h_);
  push.src_pitch_pixels = static_cast<std::uint32_t>(gpu_src_.pitch_pixels());
  push.dst_w = static_cast<std::uint32_t>(dst_w_);
  push.dst_h = static_cast<std::uint32_t>(dst_h_);
  push.dst_pitch_pixels = static_cast<std::uint32_t>(gpu_dst_.pitch_pixels());
  vf.vkCmdPushConstants(command_buffer_, pipeline_layout_,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  vf.vkCmdDispatch(command_buffer_,
                   CeilDiv(static_cast<std::uint32_t>(dst_w_), 16u),
                   CeilDiv(static_cast<std::uint32_t>(dst_h_), 16u), 1);

  VkBufferMemoryBarrier dst_ready{};
  dst_ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  dst_ready.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  dst_ready.buffer = gpu_dst_.buffer();
  dst_ready.size = gpu_dst_.byte_size();
  vf.vkCmdPipelineBarrier(command_buffer_,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                          &dst_ready, 0, nullptr);

  VkBufferCopy dst_copy{};
  dst_copy.size = gpu_dst_.byte_size();
  vf.vkCmdCopyBuffer(command_buffer_, gpu_dst_.buffer(), readback_.buffer(), 1,
                     &dst_copy);

  VkBufferMemoryBarrier readback_ready{};
  readback_ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  readback_ready.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  readback_ready.buffer = readback_.buffer();
  readback_ready.size = readback_.size();
  vf.vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                          &readback_ready, 0, nullptr);

  result = vf.vkEndCommandBuffer(command_buffer_);
  return ResultOk(result, "vkEndCommandBuffer", error_out);
}

bool IsResizeBilinearAvailable(std::string *error_out) {
  ResizeBilinear resize;
  return resize.EnsureInitialized(4, 4, 4, 4, error_out);
}

} // namespace studiocast::vulkan::kernels

namespace studiocast::vulkan {

OpenVulkanDiagnostics DiagnoseOpenVulkanDefault() {
  kernels::ResizeBilinear resize;
  std::string e;
  if (!resize.EnsureInitialized(4, 4, 4, 4, &e)) {
    OpenVulkanDiagnostics d = resize.Diagnostics();
    if (d.error.empty())
      d.error = e;
    if (d.fallback_reason.empty())
      d.fallback_reason = "open_vulkan_resize_unavailable";
    return d;
  }
  kernels::UtilityKernels utility;
  if (!utility.Initialize(&e)) {
    OpenVulkanDiagnostics d = utility.Diagnostics();
    if (d.error.empty())
      d.error = e;
    if (d.fallback_reason.empty())
      d.fallback_reason = "open_vulkan_utility_kernels_unavailable";
    return d;
  }
  OpenVulkanDiagnostics d = resize.Diagnostics();
  d.ok = true;
  d.shader_pipeline_created = true;
  return d;
}

} // namespace studiocast::vulkan
