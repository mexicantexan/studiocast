#include "core/vulkan/kernels/tracking_kernels.h"

#include "core/util/process_exit.h"

#include "core/vulkan/kernels/shaders/dlib_predictor_spv.h"
#include "core/vulkan/kernels/shaders/eye_contact_spv.h"
#include "core/vulkan/kernels/shaders/tracking_spv.h"

#include <algorithm>
#include <vector>

namespace studiocast::vulkan::kernels {
struct TrackingKernels::Impl {
  VulkanDevice *device = nullptr;
  VulkanContextIdentity context{};
  std::shared_ptr<void> retained;
  VkDescriptorSetLayout descriptor_layout = nullptr;
  VkDescriptorPool pool = nullptr;
  VkPipelineLayout layout = nullptr;
  std::array<VkShaderModule, 3> modules{};
  std::array<VkPipeline, 3> pipelines{};
  std::vector<VkCommandBuffer> commands;
  std::vector<bool> recorded;
  std::vector<VkDescriptorSet> sets;
  std::vector<std::array<TrackingBufferBinding, 8>> bindings;
  std::size_t used_sets = 0;
};

TrackingKernels::TrackingKernels() : impl_(std::make_unique<Impl>()) {}
TrackingKernels::~TrackingKernels() {
  if (!Shutdown(nullptr)) {
    util::MarkUnsafeNativeResourcesRetained();
    (void)impl_.release();
  }
}
bool TrackingKernels::Prepare(VulkanDevice *device, bool enable_dlib,
                              std::size_t command_capacity,
                              std::size_t descriptor_capacity,
                              std::string *error) {
  if (error)
    error->clear();
  if (!device || !device->Initialized() ||
      !device->context_identity().Valid() ||
      !device->health_state().healthy() || !command_capacity ||
      command_capacity > 32 || !descriptor_capacity ||
      descriptor_capacity > 64 || !Shutdown(error)) {
    if (error && error->empty())
      *error = "invalid resident tracking kernel setup";
    return false;
  }
  auto &p = *impl_;
  p.device = device;
  p.context = device->context_identity();
  p.retained = device->RetainContext();
  const auto &f = device->f();
  const auto check = [&](VkResult r, const char *operation) {
    return device->CheckDriverResult(r, operation, false, error);
  };
  std::array<VkDescriptorSetLayoutBinding, 8> bindings{};
  for (std::uint32_t i = 0; i < bindings.size(); ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo descriptor{};
  descriptor.bindingCount = bindings.size();
  descriptor.pBindings = bindings.data();
  if (!check(f.vkCreateDescriptorSetLayout(device->device(), &descriptor,
                                           nullptr, &p.descriptor_layout),
             "tracking descriptor layout"))
    return false;
  VkPushConstantRange push{};
  push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push.size = 128;
  VkPipelineLayoutCreateInfo layout{};
  layout.setLayoutCount = 1;
  layout.pSetLayouts = &p.descriptor_layout;
  layout.pushConstantRangeCount = 1;
  layout.pPushConstantRanges = &push;
  if (!check(f.vkCreatePipelineLayout(device->device(), &layout, nullptr,
                                      &p.layout),
             "tracking pipeline layout"))
    return false;
  const std::array<const std::uint32_t *, 3> code = {
      shaders::kTrackingSpirv, shaders::kDlibPredictorSpirv,
      shaders::kEyeContactSpirv};
  const std::array<std::size_t, 3> bytes = {
      shaders::kTrackingSpirvSizeBytes, shaders::kDlibPredictorSpirvSizeBytes,
      shaders::kEyeContactSpirvSizeBytes};
  for (std::size_t i = 0; i < 3; ++i) {
    if (i == 1 && !enable_dlib)
      continue;
    VkShaderModuleCreateInfo module{};
    module.codeSize = bytes[i];
    module.pCode = code[i];
    if (!check(f.vkCreateShaderModule(device->device(), &module, nullptr,
                                      &p.modules[i]),
               "tracking shader module"))
      return false;
    VkComputePipelineCreateInfo pipeline{};
    pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline.stage.module = p.modules[i];
    pipeline.stage.pName = "main";
    pipeline.layout = p.layout;
    if (!check(f.vkCreateComputePipelines(device->device(), nullptr, 1,
                                          &pipeline, nullptr, &p.pipelines[i]),
               "tracking pipeline"))
      return false;
  }
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = descriptor_capacity * 8;
  VkDescriptorPoolCreateInfo pool{};
  pool.maxSets = descriptor_capacity;
  pool.poolSizeCount = 1;
  pool.pPoolSizes = &pool_size;
  if (!check(
          f.vkCreateDescriptorPool(device->device(), &pool, nullptr, &p.pool),
          "tracking descriptor pool"))
    return false;
  std::vector<VkDescriptorSetLayout> layouts(descriptor_capacity,
                                             p.descriptor_layout);
  p.sets.resize(descriptor_capacity);
  p.bindings.resize(descriptor_capacity);
  VkDescriptorSetAllocateInfo allocate{};
  allocate.descriptorPool = p.pool;
  allocate.descriptorSetCount = descriptor_capacity;
  allocate.pSetLayouts = layouts.data();
  if (!check(f.vkAllocateDescriptorSets(device->device(), &allocate,
                                        p.sets.data()),
             "tracking descriptor allocation"))
    return false;
  p.commands.resize(command_capacity);
  p.recorded.resize(command_capacity);
  VkCommandBufferAllocateInfo command{};
  command.commandPool = device->command_pool();
  command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command.commandBufferCount = command_capacity;
  return check(
      f.vkAllocateCommandBuffers(device->device(), &command, p.commands.data()),
      "tracking command allocation");
}

bool TrackingKernels::Record(std::size_t slot,
                             std::span<const TrackingDispatch> dispatches,
                             VkCommandBuffer *out, std::string *error) {
  if (out)
    *out = nullptr;
  if (!impl_ || !impl_->device || slot >= impl_->commands.size() ||
      impl_->recorded[slot] || dispatches.empty()) {
    if (error)
      *error = "invalid or already recorded tracking command slot";
    return false;
  }
  auto &p = *impl_;
  auto &device = *p.device;
  const auto &f = device.f();
  const auto cmd = p.commands[slot];
  VkCommandBufferBeginInfo begin{};
  if (!device.CheckDriverResult(f.vkBeginCommandBuffer(cmd, &begin),
                                "tracking command begin", false, error))
    return false;
  for (const auto &dispatch : dispatches) {
    const auto pipeline = p.pipelines[static_cast<unsigned>(dispatch.shader)];
    if (!pipeline || !dispatch.invocations) {
      if (error)
        *error = "tracking shader unavailable or dispatch empty";
      return false;
    }
    std::size_t set = p.used_sets;
    for (std::size_t s = 0; s < p.used_sets; ++s) {
      bool same = true;
      for (std::size_t b = 0; b < 8; ++b)
        same &= p.bindings[s][b].buffer == dispatch.buffers[b].buffer &&
                p.bindings[s][b].bytes == dispatch.buffers[b].bytes &&
                p.bindings[s][b].context == dispatch.buffers[b].context;
      if (same) {
        set = s;
        break;
      }
    }
    if (set == p.used_sets) {
      if (set >= p.sets.size()) {
        if (error)
          *error = "tracking descriptor capacity exhausted";
        return false;
      }
      std::array<VkDescriptorBufferInfo, 8> infos{};
      std::array<VkWriteDescriptorSet, 8> writes{};
      for (std::size_t b = 0; b < 8; ++b) {
        const auto &buffer = dispatch.buffers[b];
        if (!buffer.buffer || !buffer.bytes || buffer.context != p.context) {
          if (error)
            *error = "foreign or empty tracking buffer";
          return false;
        }
        infos[b].buffer = buffer.buffer;
        infos[b].range = buffer.bytes;
        writes[b].dstSet = p.sets[set];
        writes[b].dstBinding = b;
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[b].descriptorCount = 1;
        writes[b].pBufferInfo = &infos[b];
      }
      f.vkUpdateDescriptorSets(device.device(), writes.size(), writes.data(), 0,
                               nullptr);
      p.bindings[set] = dispatch.buffers;
      ++p.used_sets;
    }
    for (const auto &buffer : dispatch.buffers)
      if (!device.RecordBufferBarrier(cmd, buffer.buffer, buffer.bytes,
                                      buffer.context,
                                      VulkanBufferAccess::host_or_compute_write,
                                      VulkanBufferAccess::compute_read, error))
        return false;
    f.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    f.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0,
                              1, &p.sets[set], 0, nullptr);
    f.vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 128,
                         dispatch.parameters.data());
    const std::uint32_t groups = (dispatch.invocations + 63) / 64;
    f.vkCmdDispatch(cmd, std::min(groups, 65535u), (groups + 65534) / 65535, 1);
    for (const auto &buffer : dispatch.buffers)
      if (!device.RecordBufferBarrier(cmd, buffer.buffer, buffer.bytes,
                                      buffer.context,
                                      VulkanBufferAccess::compute_write,
                                      VulkanBufferAccess::compute_read, error))
        return false;
  }
  if (!device.CheckDriverResult(f.vkEndCommandBuffer(cmd),
                                "tracking command end", false, error))
    return false;
  p.recorded[slot] = true;
  if (out)
    *out = cmd;
  return true;
}

bool TrackingKernels::Shutdown(std::string *error) noexcept {
  if (!impl_ || !impl_->device)
    return true;
  auto &p = *impl_;
  if (!p.device->SafeToDestroyResources() ||
      p.device->context_identity() != p.context) {
    if (error)
      *error = "tracking kernel release requires exact-context quiescence";
    return false;
  }
  const auto &f = p.device->f();
  const auto device = p.device->device();
  if (!p.commands.empty() && p.commands[0])
    f.vkFreeCommandBuffers(device, p.device->command_pool(), p.commands.size(),
                           p.commands.data());
  if (p.pool)
    f.vkDestroyDescriptorPool(device, p.pool, nullptr);
  for (auto pipeline : p.pipelines)
    if (pipeline)
      f.vkDestroyPipeline(device, pipeline, nullptr);
  for (auto module : p.modules)
    if (module)
      f.vkDestroyShaderModule(device, module, nullptr);
  if (p.layout)
    f.vkDestroyPipelineLayout(device, p.layout, nullptr);
  if (p.descriptor_layout)
    f.vkDestroyDescriptorSetLayout(device, p.descriptor_layout, nullptr);
  p = Impl{};
  return true;
}
} // namespace studiocast::vulkan::kernels
