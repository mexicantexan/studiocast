#include "core/open_video/prepared_vulkan_fastdvdnet.h"

#include "core/util/process_exit.h"

#include "core/vulkan/kernels/shaders/fastdvdnet_spv.h"
#include "core/vulkan/vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace studiocast::open_vulkan {
namespace {
namespace vk = studiocast::vulkan;
bool Fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}
template <class T> std::uintptr_t Handle(T value) {
  return reinterpret_cast<std::uintptr_t>(value);
}
VulkanInferenceTensorBinding TensorBinding(vk::VulkanDevice *device,
                                           const vk::VulkanTensor &tensor) {
  VulkanInferenceTensorBinding out;
  out.resource.ownership_domain = device;
  out.resource.logical_device = Handle(device->device());
  out.resource.buffer = Handle(tensor.buffer());
  out.resource.byte_size = tensor.byte_size();
  out.resource.n = 1;
  out.resource.c = tensor.c();
  out.resource.h = tensor.h();
  out.resource.w = tensor.w();
  out.resource.device_resident = true;
  out.resource.device_local_memory = true;
  out.resource.context_id = device->context_identity().context_id;
  out.resource.context_generation = device->context_identity().generation;
  out.resource.runtime_queue_ownership = true;
  out.dims = 3;
  out.w = tensor.w();
  out.h = tensor.h();
  out.c = tensor.c();
  out.row_stride_bytes = static_cast<std::size_t>(tensor.w()) * sizeof(float);
  out.plane_stride_bytes = out.row_stride_bytes * tensor.h();
  return out;
}
} // namespace

struct PreparedVulkanFastDvdnet::Impl {
  struct Parameters {
    std::uint32_t width, height, padded_width, padded_height;
    std::uint32_t source_pitch, output_pitch, reset;
    float sigma;
  };
  PreparedVulkanFastDvdnetSetup setup;
  NcnnVulkanInference graph;
  std::array<vk::VulkanTensor, 3> history;
  vk::VulkanTensor noisy, noise_map, denoised;
  vk::VulkanBuffer parameters;
  std::array<vk::VkCommandBuffer, 2> commands{};
  std::array<vk::ResidentCommandTicket, 2> tickets{};
  std::array<bool, 2> pending{};
  vk::VkDescriptorSetLayout descriptor_layout = nullptr;
  vk::VkDescriptorPool descriptor_pool = nullptr;
  vk::VkDescriptorSet descriptors = nullptr;
  vk::VkShaderModule shader = nullptr;
  vk::VkPipelineLayout pipeline_layout = nullptr;
  vk::VkPipeline pipeline = nullptr;
  hal::ResourceBudget used{};
  Parameters params{};
  std::uint64_t last_sequence = 0, queued_sequence = 0;
  bool ready = false, have_sequence = false, reset_requested = true;
  bool unsafe = false;

  static void Completion(void *owner, const vk::ResidentCommandTicket &ticket,
                         vk::ResidentCommandCompletion completion) noexcept {
    auto &s = *static_cast<Impl *>(owner);
    for (std::size_t i = 0; i < s.tickets.size(); ++i) {
      if (!s.pending[i] || s.tickets[i] != ticket)
        continue;
      if (completion == vk::ResidentCommandCompletion::unsafe_pending) {
        s.unsafe = true;
        s.reset_requested = true;
        return;
      }
      s.pending[i] = false;
      if (completion == vk::ResidentCommandCompletion::completed && i == 1) {
        s.last_sequence = s.queued_sequence;
        s.have_sequence = true;
        s.reset_requested = false;
      } else if (completion != vk::ResidentCommandCompletion::completed) {
        s.have_sequence = false;
        s.reset_requested = true;
      }
    }
  }

  bool Record(const vk::VulkanImage &input, const vk::VulkanImage &output,
              std::string *error) {
    auto *device = setup.device;
    const auto &f = device->f();
    std::array<vk::VkDescriptorSetLayoutBinding, 9> bindings{};
    for (std::uint32_t i = 0; i < bindings.size(); ++i) {
      bindings[i].binding = i;
      bindings[i].descriptorType = vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = vk::VK_SHADER_STAGE_COMPUTE_BIT;
    }
    vk::VkDescriptorSetLayoutCreateInfo layout{};
    layout.bindingCount = bindings.size();
    layout.pBindings = bindings.data();
    if (!device->CheckDriverResult(
            f.vkCreateDescriptorSetLayout(device->device(), &layout, nullptr,
                                          &descriptor_layout),
            "denoise descriptor layout", false, error))
      return false;
    vk::VkPushConstantRange push{};
    push.stageFlags = vk::VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = sizeof(std::uint32_t);
    vk::VkPipelineLayoutCreateInfo pl{};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &descriptor_layout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &push;
    if (!device->CheckDriverResult(f.vkCreatePipelineLayout(device->device(),
                                                            &pl, nullptr,
                                                            &pipeline_layout),
                                   "denoise pipeline layout", false, error))
      return false;
    vk::VkShaderModuleCreateInfo sm{};
    sm.codeSize = vk::kernels::shaders::kFastDvdnetSpirvSizeBytes;
    sm.pCode = vk::kernels::shaders::kFastDvdnetSpirv;
    if (!device->CheckDriverResult(
            f.vkCreateShaderModule(device->device(), &sm, nullptr, &shader),
            "denoise shader", false, error))
      return false;
    vk::VkComputePipelineCreateInfo cp{};
    cp.layout = pipeline_layout;
    cp.stage.stage = vk::VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = shader;
    cp.stage.pName = "main";
    if (!device->CheckDriverResult(
            f.vkCreateComputePipelines(device->device(), nullptr, 1, &cp,
                                       nullptr, &pipeline),
            "denoise pipeline", false, error))
      return false;
    vk::VkDescriptorPoolSize size{};
    size.type = vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    size.descriptorCount = 9;
    vk::VkDescriptorPoolCreateInfo pool{};
    pool.maxSets = 1;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &size;
    if (!device->CheckDriverResult(f.vkCreateDescriptorPool(device->device(),
                                                            &pool, nullptr,
                                                            &descriptor_pool),
                                   "denoise descriptor pool", false, error))
      return false;
    vk::VkDescriptorSetAllocateInfo ds{};
    ds.descriptorPool = descriptor_pool;
    ds.descriptorSetCount = 1;
    ds.pSetLayouts = &descriptor_layout;
    if (!device->CheckDriverResult(
            f.vkAllocateDescriptorSets(device->device(), &ds, &descriptors),
            "denoise descriptor set", false, error))
      return false;
    const std::array<vk::VkBuffer, 9> buffers{
        parameters.buffer(), input.buffer(),      history[0].buffer(),
        history[1].buffer(), history[2].buffer(), noisy.buffer(),
        noise_map.buffer(),  denoised.buffer(),   output.buffer()};
    std::array<vk::VkDescriptorBufferInfo, 9> info{};
    std::array<vk::VkWriteDescriptorSet, 9> writes{};
    for (std::uint32_t i = 0; i < buffers.size(); ++i) {
      info[i].buffer = buffers[i];
      info[i].range = vk::VK_WHOLE_SIZE;
      writes[i].dstSet = descriptors;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = vk::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &info[i];
    }
    f.vkUpdateDescriptorSets(device->device(), writes.size(), writes.data(), 0,
                             nullptr);
    vk::VkCommandBufferAllocateInfo allocation{};
    allocation.commandPool = device->command_pool();
    allocation.level = vk::VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = commands.size();
    if (!device->CheckDriverResult(f.vkAllocateCommandBuffers(device->device(),
                                                              &allocation,
                                                              commands.data()),
                                   "denoise command buffers", false, error))
      return false;
    for (std::uint32_t op = 0; op < commands.size(); ++op) {
      const auto command = commands[op];
      vk::VkCommandBufferBeginInfo begin{};
      if (!device->CheckDriverResult(f.vkBeginCommandBuffer(command, &begin),
                                     "denoise command begin", false, error))
        return false;
      for (std::size_t i = 0; i < buffers.size(); ++i)
        if (!device->RecordBufferBarrier(
                command, buffers[i], vk::VK_WHOLE_SIZE,
                device->context_identity(),
                i == 0 ? vk::VulkanBufferAccess::host_write
                       : vk::VulkanBufferAccess::host_or_compute_write,
                vk::VulkanBufferAccess::compute_read, error))
          return false;
      f.vkCmdBindPipeline(command, vk::VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline);
      f.vkCmdBindDescriptorSets(command, vk::VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &descriptors, 0,
                                nullptr);
      f.vkCmdPushConstants(command, pipeline_layout,
                           vk::VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(op), &op);
      f.vkCmdDispatch(command, (params.padded_width + 15) / 16,
                      (params.padded_height + 15) / 16, 1);
      for (std::size_t i = 2; i < buffers.size(); ++i)
        if (!device->RecordBufferBarrier(command, buffers[i], vk::VK_WHOLE_SIZE,
                                         device->context_identity(),
                                         vk::VulkanBufferAccess::compute_write,
                                         vk::VulkanBufferAccess::compute_read,
                                         error))
          return false;
      if (!device->CheckDriverResult(f.vkEndCommandBuffer(command),
                                     "denoise command end", false, error))
        return false;
    }
    return true;
  }
};

PreparedVulkanFastDvdnet::PreparedVulkanFastDvdnet()
    : impl_(std::make_unique<Impl>()) {}
PreparedVulkanFastDvdnet::~PreparedVulkanFastDvdnet() {
  if (!Shutdown(nullptr)) {
    util::MarkUnsafeNativeResourcesRetained();
    (void)impl_.release();
  }
}

std::uint32_t
PreparedVulkanFastDvdnet::PaddedExtent(std::uint32_t extent) noexcept {
  return extent > UINT32_MAX - 3u ? 0u : (extent + 3u) & ~3u;
}
float PreparedVulkanFastDvdnet::NoiseSigmaOver255(int strength) noexcept {
  return (55.F * (static_cast<float>(std::clamp(strength, 0, 100)) / 100.F)) /
         255.F;
}

bool PreparedVulkanFastDvdnet::Prepare(
    const PreparedVulkanFastDvdnetSetup &setup, const vk::VulkanImage &input,
    const vk::VulkanImage &output, std::string *error) {
  auto &s = *impl_;
  if (error)
    error->clear();
  const auto pw = PaddedExtent(static_cast<std::uint32_t>(setup.width));
  const auto ph = PaddedExtent(static_cast<std::uint32_t>(setup.height));
  if (s.setup.device || !setup.device || !setup.device->Initialized() ||
      !setup.owner_generation || setup.width <= 0 || setup.height <= 0 ||
      setup.width > 4096 || setup.height > 4096 || !pw || !ph ||
      setup.strength < 0 || setup.strength > 100 || !setup.quota ||
      setup.quota->device != setup.device ||
      setup.quota->context != setup.device->context_identity() ||
      !hal::ValidateExecutionDomain(setup.domain) ||
      setup.domain.backend != hal::BackendId::vulkan ||
      setup.domain.logical_context_generation !=
          setup.device->context_identity().generation ||
      setup.domain.queue_or_stream_identity != Handle(setup.device->queue()) ||
      setup.graph.profile != "fastdvdnet_rgb_v1" ||
      setup.graph.input_count != 2 || setup.graph.output_count != 1 ||
      !input.BelongsTo(*setup.device) || !output.BelongsTo(*setup.device) ||
      input.mapped() || output.mapped() || !input.device_local() ||
      !output.device_local() ||
      input.format() != vk::VulkanPixelFormat::rgb_u8 ||
      output.format() != vk::VulkanPixelFormat::rgb_u8 ||
      input.buffer() == output.buffer() || input.width() != setup.width ||
      input.height() != setup.height || output.width() != setup.width ||
      output.height() != setup.height)
    return Fail(error, "Invalid prepared Vulkan FastDVDnet setup.");
  const std::uint64_t bytes =
      static_cast<std::uint64_t>(pw) * ph * 28u * sizeof(float);
  if (bytes + sizeof(Impl::Parameters) > setup.reservation.persistent_bytes ||
      setup.reservation.resource_count < 7 ||
      setup.reservation.in_flight_slots != 1)
    return Fail(error, "Prepared Vulkan FastDVDnet reservation exhausted.");
  if (!setup.device->ConfigureAllocationLimits(
          setup.quota->allocation_budget_bytes,
          setup.quota->allocation_resource_limit, error))
    return false;
  const auto allocations_before = setup.device->allocation_stats();
  s.setup = setup;
  s.params = {static_cast<std::uint32_t>(setup.width),
              static_cast<std::uint32_t>(setup.height),
              pw,
              ph,
              static_cast<std::uint32_t>(input.pitch_pixels()),
              static_cast<std::uint32_t>(output.pitch_pixels()),
              1,
              NoiseSigmaOver255(setup.strength)};
  for (auto &history : s.history)
    if (!history.AllocateNchwF32(setup.device, 1, 3, ph, pw, false, error))
      return false;
  if (!s.noisy.AllocateNchwF32(setup.device, 1, 15, ph, pw, false, error) ||
      !s.noise_map.AllocateNchwF32(setup.device, 1, 1, ph, pw, false, error) ||
      !s.denoised.AllocateNchwF32(setup.device, 1, 3, ph, pw, false, error) ||
      !s.parameters.Allocate(setup.device, sizeof(s.params),
                             vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                             vk::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true,
                             error) ||
      !s.Record(input, output, error))
    return false;
  VulkanInferenceBindings bindings;
  bindings.owner_generation = setup.owner_generation;
  bindings.input_count = 2;
  bindings.output_count = 1;
  bindings.inputs[0] = TensorBinding(setup.device, s.noisy);
  bindings.inputs[1] = TensorBinding(setup.device, s.noise_map);
  bindings.outputs[0] = TensorBinding(setup.device, s.denoised);
  VulkanMattingDeviceContext context;
  context.ownership_domain = setup.device;
  context.physical_device = Handle(setup.device->physical_device());
  context.logical_device = Handle(setup.device->device());
  context.queue = Handle(setup.device->queue());
  context.queue_family_index = setup.device->queue_family_index();
  context.vendor_id = setup.device->identity().vendor_id;
  context.device_id = setup.device->identity().device_id;
  context.selected_device_index =
      setup.device->diagnostics().selected_device_index;
  context.stable_device_id = setup.device->identity().stable_id;
  context.context_id = setup.device->context_identity().context_id;
  context.context_generation = setup.device->context_identity().generation;
  context.non_cpu_device_selected =
      setup.device->diagnostics().non_cpu_device_selected;
  context.compute_queue_available =
      setup.device->diagnostics().compute_queue_available;
  context.context_healthy = setup.device->health_state().healthy();
  const auto prepared =
      s.graph.Prepare(context, setup.graph, bindings, setup.quota);
  if (!prepared.ok())
    return Fail(error, prepared.detail.c_str());
  const auto warmed = s.graph.Warmup();
  if (!warmed.ok())
    return Fail(error, warmed.detail.c_str());
  const auto allocations_after = setup.device->allocation_stats();
  s.used.persistent_bytes =
      allocations_after.current_bytes - allocations_before.current_bytes;
  const auto resources = allocations_after.allocation_count - allocations_before.allocation_count;
  if (resources > UINT32_MAX)
    return Fail(error, "FastDVDnet resource count exceeds fixed contract.");
  s.used.resource_count = static_cast<std::uint32_t>(resources);
  s.used.in_flight_slots = 1;
  if (s.used.persistent_bytes > setup.reservation.persistent_bytes)
    return Fail(error,
                "Pitched FastDVDnet resources exceed caller reservation.");
  s.ready = true;
  return true;
}

bool PreparedVulkanFastDvdnet::Append(vk::ResidentCommandBatch &batch,
                                      std::uint64_t sequence,
                                      std::string *error) {
  auto &s = *impl_;
  if (!s.ready || s.unsafe || !sequence ||
      sequence != batch.capture_sequence() || s.pending[0] || s.pending[1] ||
      batch.context_identity() != s.setup.device->context_identity())
    return Fail(error,
                "Prepared Vulkan FastDVDnet binding is stale or pending.");
  s.params.reset = s.reset_requested || !s.have_sequence ||
                   s.last_sequence == UINT64_MAX ||
                   sequence != s.last_sequence + 1;
  std::memcpy(s.parameters.mapped(), &s.params, sizeof(s.params));
  if (!s.parameters.Flush(error))
    return false;
  s.queued_sequence = sequence;
  const auto mark = batch.Mark();
  const auto append = [&](std::size_t index) {
    vk::ResidentCommandGroup group{
        s.setup.device->context_identity(), &s.commands[index], 1, &s,
        s.setup.owner_generation,           Impl::Completion};
    const bool ok = batch.Append(group, &s.tickets[index], error);
    s.pending[index] = ok;
    return ok;
  };
  if (!append(0))
    return false;
  const auto enqueued = s.graph.Enqueue(batch);
  if (!enqueued.ok() || !append(1)) {
    if (!enqueued.ok() && error)
      *error = enqueued.detail;
    (void)batch.Rollback(mark, nullptr);
    return false;
  }
  return true;
}

bool PreparedVulkanFastDvdnet::ResetTemporalState() noexcept {
  auto &s = *impl_;
  if (s.unsafe || s.pending[0] || s.pending[1])
    return false;
  s.reset_requested = true;
  s.have_sequence = false;
  s.last_sequence = 0;
  return true;
}

bool PreparedVulkanFastDvdnet::Shutdown(std::string *error) noexcept {
  auto &s = *impl_;
  if (s.pending[0] || s.pending[1] || !s.graph.Shutdown())
    return Fail(error,
                "Prepared Vulkan FastDVDnet completion remains unproven.");
  if (!s.setup.device)
    return true;
  auto *device = s.setup.device;
  if (!device->SafeToDestroyResources())
    return Fail(error, "Prepared Vulkan FastDVDnet resources are quarantined.");
  const auto &f = device->f();
  for (auto &command : s.commands) {
    if (command)
      f.vkFreeCommandBuffers(device->device(), device->command_pool(), 1,
                             &command);
    command = nullptr;
  }
  if (s.pipeline)
    f.vkDestroyPipeline(device->device(), s.pipeline, nullptr);
  if (s.pipeline_layout)
    f.vkDestroyPipelineLayout(device->device(), s.pipeline_layout, nullptr);
  if (s.shader)
    f.vkDestroyShaderModule(device->device(), s.shader, nullptr);
  if (s.descriptor_pool)
    f.vkDestroyDescriptorPool(device->device(), s.descriptor_pool, nullptr);
  if (s.descriptor_layout)
    f.vkDestroyDescriptorSetLayout(device->device(), s.descriptor_layout,
                                   nullptr);
  s.pipeline = nullptr;
  s.pipeline_layout = nullptr;
  s.shader = nullptr;
  s.descriptor_pool = nullptr;
  s.descriptor_layout = nullptr;
  s.descriptors = nullptr;
  for (auto &history : s.history)
    history.Free();
  s.noisy.Free();
  s.noise_map.Free();
  s.denoised.Free();
  s.parameters.Free();
  s.setup.device = nullptr;
  s.ready = false;
  return true;
}
bool PreparedVulkanFastDvdnet::ready() const noexcept { return impl_->ready; }
hal::ResourceBudget PreparedVulkanFastDvdnet::resources() const noexcept {
  return impl_->used;
}
VulkanMattingRuntimeEvidence PreparedVulkanFastDvdnet::evidence() const {
  return impl_->graph.Evidence();
}
void PreparedVulkanFastDvdnet::RefreshExecutionEvidence(
    VulkanMattingRuntimeEvidence *out) const noexcept {
  impl_->graph.RefreshExecutionEvidence(out);
}
} // namespace studiocast::open_vulkan
