#include "core/vulkan/kernels/resize_bilinear.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <utility>

#include "core/open_video/model_pack_registry.h"
#include "core/open_video/vulkan_matting_runtime.h"
#include "core/util/xdg.h"
#include "core/video/effects/broadcast_effect_contract.h"
#include "core/video/open_vulkan_eye_contact.h"
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
  std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
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
  std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
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
  std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
  const bool have_device = device_.Initialized();
  const bool destroy_children = have_device && device_.SafeToDestroyResources();
  const auto &vf = device_.f();

  if (destroy_children && command_buffer_ && vf.vkFreeCommandBuffers) {
    vf.vkFreeCommandBuffers(device_.device(), device_.command_pool(), 1,
                            &command_buffer_);
  }
  command_buffer_ = nullptr;

  if (destroy_children && pipeline_ && vf.vkDestroyPipeline)
    vf.vkDestroyPipeline(device_.device(), pipeline_, nullptr);
  pipeline_ = nullptr;
  if (destroy_children && pipeline_layout_ && vf.vkDestroyPipelineLayout)
    vf.vkDestroyPipelineLayout(device_.device(), pipeline_layout_, nullptr);
  pipeline_layout_ = nullptr;
  if (destroy_children && shader_module_ && vf.vkDestroyShaderModule)
    vf.vkDestroyShaderModule(device_.device(), shader_module_, nullptr);
  shader_module_ = nullptr;
  if (destroy_children && descriptor_pool_ && vf.vkDestroyDescriptorPool)
    vf.vkDestroyDescriptorPool(device_.device(), descriptor_pool_, nullptr);
  descriptor_pool_ = nullptr;
  descriptor_set_ = nullptr;
  if (destroy_children && descriptor_set_layout_ &&
      vf.vkDestroyDescriptorSetLayout)
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
  std::lock_guard<std::recursive_mutex> execution_lock(execution_mutex_);
  OpenVulkanDiagnostics d = device_.DiagnosticsSnapshot();
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
  if (!device_.CheckDriverResult(result, "vkCreateShaderModule", false,
                                 error_out))
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
  if (!device_.CheckDriverResult(result, "vkCreateDescriptorSetLayout", false,
                                 error_out))
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
  if (!device_.CheckDriverResult(result, "vkCreatePipelineLayout", false,
                                 error_out))
    return false;

  VkComputePipelineCreateInfo compute{};
  compute.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  compute.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  compute.stage.module = shader_module_;
  compute.stage.pName = "main";
  compute.layout = pipeline_layout_;
  result = vf.vkCreateComputePipelines(device_.device(), nullptr, 1, &compute,
                                       nullptr, &pipeline_);
  if (!device_.CheckDriverResult(result, "vkCreateComputePipelines", false,
                                 error_out))
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
  VkResult result = vf.vkCreateDescriptorPool(device_.device(), &pool, nullptr,
                                              &descriptor_pool_);
  if (!device_.CheckDriverResult(result, "vkCreateDescriptorPool", false,
                                 error_out))
    return false;

  VkDescriptorSetAllocateInfo alloc{};
  alloc.descriptorPool = descriptor_pool_;
  alloc.descriptorSetCount = 1;
  alloc.pSetLayouts = &descriptor_set_layout_;
  result =
      vf.vkAllocateDescriptorSets(device_.device(), &alloc, &descriptor_set_);
  if (!device_.CheckDriverResult(result, "vkAllocateDescriptorSets", false,
                                 error_out))
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
  if (!device_.CheckDriverResult(result, "vkAllocateCommandBuffers", false,
                                 error_out))
    return false;

  VkCommandBufferBeginInfo begin{};
  begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  result = vf.vkBeginCommandBuffer(command_buffer_, &begin);
  if (!device_.CheckDriverResult(result, "vkBeginCommandBuffer", true,
                                 error_out))
    return false;

  if (!device_.RecordBufferBarrier(
          command_buffer_, upload_.buffer(), upload_.size(),
          upload_.context_identity(), VulkanBufferAccess::host_write,
          VulkanBufferAccess::transfer_read, error_out)) {
    return false;
  }

  VkBufferCopy src_copy{};
  src_copy.size = gpu_src_.byte_size();
  vf.vkCmdCopyBuffer(command_buffer_, upload_.buffer(), gpu_src_.buffer(), 1,
                     &src_copy);

  if (!device_.RecordBufferBarrier(
          command_buffer_, gpu_src_.buffer(), gpu_src_.byte_size(),
          gpu_src_.context_identity(), VulkanBufferAccess::transfer_write,
          VulkanBufferAccess::compute_read, error_out)) {
    return false;
  }

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

  if (!device_.RecordBufferBarrier(
          command_buffer_, gpu_dst_.buffer(), gpu_dst_.byte_size(),
          gpu_dst_.context_identity(), VulkanBufferAccess::compute_write,
          VulkanBufferAccess::transfer_read, error_out)) {
    return false;
  }

  VkBufferCopy dst_copy{};
  dst_copy.size = gpu_dst_.byte_size();
  vf.vkCmdCopyBuffer(command_buffer_, gpu_dst_.buffer(), readback_.buffer(), 1,
                     &dst_copy);

  if (!device_.RecordBufferBarrier(
          command_buffer_, readback_.buffer(), readback_.size(),
          readback_.context_identity(), VulkanBufferAccess::transfer_write,
          VulkanBufferAccess::host_read, error_out)) {
    return false;
  }

  result = vf.vkEndCommandBuffer(command_buffer_);
  return device_.CheckDriverResult(result, "vkEndCommandBuffer", true,
                                   error_out);
}

bool IsResizeBilinearAvailable(std::string *error_out) {
  ResizeBilinear resize;
  return resize.EnsureInitialized(4, 4, 4, 4, error_out);
}

} // namespace studiocast::vulkan::kernels

namespace studiocast::vulkan {

namespace {

struct OpenVulkanMattingModelFacts {
  bool selected = false;
  bool contract_validated = false;
};

OpenVulkanMattingModelFacts
PopulateOpenVulkanModelDiagnostics(OpenVulkanDiagnostics *d) {
  OpenVulkanMattingModelFacts facts;
  if (!d)
    return facts;

  const auto reg = studiocast::open_video::ModelPackRegistry::ScanDefault();
  std::set<std::string> installed_model_ids;
  const auto is_production_candidate = [&](const std::string &id) {
    const auto candidate = reg.Find("matting", id);
    return candidate && candidate->ncnn_vulkan.has_value();
  };

  for (const auto &m : reg.ListModels()) {
    if (m.task != "matting")
      continue;
    installed_model_ids.insert(m.id);
    d->installed_models.push_back(m.id);
    OpenVulkanDiagnostics::ModelInfo mi;
    mi.id = m.id;
    mi.display_name = m.display_name;
    mi.task = m.task;
    if (m.matting) {
      mi.width = m.matting->input.width;
      mi.height = m.matting->input.height;
    }
    d->models.push_back(std::move(mi));
  }

  if (is_production_candidate("modnet-webnn-256-fp32")) {
    d->default_model_id = "modnet-webnn-256-fp32";
  } else if (is_production_candidate("birefnet_lite")) {
    d->default_model_id = "birefnet_lite";
  } else {
    for (const auto &m : reg.ListModels()) {
      if (m.task == "matting" && is_production_candidate(m.id)) {
        d->default_model_id = m.id;
        break;
      }
    }
  }
  facts.selected = !d->default_model_id.empty();
  // Diagnostics polling never hashes artifacts. Only a live session may
  // publish cached validation attestation; this cold diagnostic has none.
  facts.contract_validated = false;

  for (const auto &[id, reason] : reg.Problems()) {
    const std::string lower_reason = [&] {
      std::string out = reason;
      std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      return out;
    }();
    if (installed_model_ids.find(id) != installed_model_ids.end() &&
        lower_reason.find("duplicate model id") != std::string::npos) {
      continue;
    }
    d->missing_models[id] = reason;
  }

  const auto modelsRoot = studiocast::util::StudioCastModelsDir();
  const auto openVideoRoot =
      modelsRoot.empty()
          ? std::string("~/.local/share/studiocast/models/open_video")
          : (modelsRoot / "open_video").string();
  d->install_hints.push_back(std::string("Model packs: ") + openVideoRoot +
                             "/<subject>/<pack_dir>/");
  d->install_hints.push_back("Example: " + openVideoRoot +
                             "/matting/Good Quality/model.json");
  d->install_hints.push_back(
      "Source builds: run ./scripts/install.sh open-video-models to install "
      "curated Open Video packs.");
  d->install_hints.push_back(
      "Device-resident Vulkan virtual background requires a production Vulkan "
      "matting runtime; the ncnn spike CPU Mat path is not used here.");
  return facts;
}

void ApplyOpenVulkanMattingReadiness(
    OpenVulkanDiagnostics *d, const OpenVulkanMattingModelFacts &model_facts) {
  if (!d)
    return;
  studiocast::open_vulkan::VulkanMattingReadinessInput input;
  input.production_build_enabled =
      studiocast::open_vulkan::ProductionVulkanMattingBuildEnabled();
  input.production_adapter_available =
      studiocast::open_vulkan::ProductionVulkanMattingAdapterAvailable();
  input.model_pack_selected = model_facts.selected;
  input.model_contract_validated = model_facts.contract_validated;
  input.non_cpu_device_selected = d->non_cpu_device_selected;
  input.compute_queue_available = d->compute_queue_available;
  input.context_healthy = d->context_healthy;
  const auto readiness =
      studiocast::open_vulkan::EvaluateVulkanMattingReadiness(input);
  const auto &runtime = readiness.runtime;

  d->matting_build_enabled = input.production_build_enabled;
  d->matting_adapter_available = input.production_adapter_available;
  d->matting_model_pack_selected = input.model_pack_selected;
  d->matting_model_contract_validated = input.model_contract_validated;
  d->matting_production_ready = readiness.production_ready;
  d->matting_reason_code = readiness.reason_code;
  d->matting_blocker_code = readiness.blocker_code;
  d->matting_detail = readiness.detail;
  d->matting_runtime =
      runtime.runtime_name.empty() ? "none" : runtime.runtime_name;
  d->matting_runtime_created = runtime.runtime_created;
  d->matting_graph_loaded = runtime.graph_loaded;
  d->matting_warmup_complete = runtime.warmup_complete;
  d->matting_cpu_layers_used = runtime.cpu_layers_used;
  d->matting_shared_device_imported = runtime.shared_device_imported;
  d->matting_queue_ownership_explicit = runtime.queue_ownership_explicit;
  d->matting_synchronous_completion = runtime.synchronous_completion;
  d->matting_bounded_reusable_allocations =
      runtime.bounded_reusable_allocations;
  d->matting_persistent_allocation_count = runtime.persistent_allocation_count;
  d->matting_dynamic_allocation_count = runtime.dynamic_allocation_count;
  d->matting_cpu_readback_count = runtime.cpu_readback_count;
  d->matting_warmup_inference_count = runtime.warmup_inference_count;
  d->matting_inference_count = runtime.inference_count;
  d->matting_completion_count = runtime.completion_count;
  d->matting_context_id = runtime.active_device.context_id;
  d->matting_context_generation = runtime.active_device.context_generation;
  d->input_device_resident = runtime.input_device_resident;
  d->alpha_device_resident = runtime.alpha_device_resident;
  d->output_device_resident = runtime.output_device_resident;
  d->device_residency_mode =
      readiness.production_ready ? "device_resident" : "unavailable";
}

void BlockOpenVulkanVirtualBackground(OpenVulkanDiagnostics *d,
                                      const char *reason_code) {
  if (!d)
    return;
  d->blocked_effects[std::string(
      studiocast::video::effects::contract::kEffectIdVirtualBackgroundBlur)] =
      reason_code;
  d->blocked_effects[std::string(
      studiocast::video::effects::contract::kEffectIdVirtualBackgroundRemove)] =
      reason_code;
  d->blocked_effects[std::string(studiocast::video::effects::contract::
                                     kEffectIdVirtualBackgroundReplace)] =
      reason_code;
}

void ApplyOpenVulkanEyeContactReadiness(OpenVulkanDiagnostics *d) {
  if (!d)
    return;

  auto current_facts = studiocast::video::CurrentOpenVulkanEyeContactFacts();
  current_facts.non_cpu_device_selected = d->non_cpu_device_selected;
  current_facts.compute_queue_available = d->compute_queue_available;
  current_facts.context_healthy = d->context_healthy;
  const auto readiness =
      studiocast::video::EvaluateOpenVulkanEyeContactReadiness(current_facts);
  const auto &facts = readiness.facts;
  d->eye_contact_production_ready = readiness.production_ready;
  d->eye_contact_reason_code = readiness.reason_code;
  d->eye_contact_blocker_code = readiness.blocker_code;
  d->eye_contact_detail = readiness.detail;
  d->eye_contact_backend_compiled = facts.backend_compiled;
  d->eye_contact_live_stage_implemented = facts.live_stage_implemented;
  d->eye_contact_production_adapter_available =
      facts.production_adapter_available;
  d->eye_contact_vulkan_inference_provider_available =
      facts.vulkan_inference_provider_available;
  d->eye_contact_non_cpu_device_selected = facts.non_cpu_device_selected;
  d->eye_contact_compute_queue_available = facts.compute_queue_available;
  d->eye_contact_context_healthy = facts.context_healthy;
  d->eye_contact_shared_device_imported = facts.shared_device_imported;
  d->eye_contact_queue_ownership_explicit = facts.queue_ownership_explicit;
  d->eye_contact_model_pack_selected = facts.model_pack_selected;
  d->eye_contact_artifact_contract_validated =
      facts.artifact_contract_validated;
  d->eye_contact_device_resident_analysis = facts.device_resident_analysis;
  d->eye_contact_device_resident_tensor_io = facts.device_resident_tensor_io;
  d->eye_contact_warmup_complete = facts.warmup_complete;
  d->eye_contact_bounded_reusable_allocations =
      facts.bounded_reusable_allocations;
  d->eye_contact_synchronization_contract_validated =
      facts.synchronization_contract_validated;
  d->eye_contact_parity_validated = facts.parity_validated;
  d->eye_contact_selectable_cpu_fallback = facts.selectable_cpu_fallback;
  d->eye_contact_dispatch_count = facts.dispatch_count;
  d->eye_contact_cpu_readback_count = facts.cpu_readback_count;
  d->eye_contact_cpu_fallback_count = facts.cpu_fallback_count;

  const std::string effect_id(
      studiocast::video::effects::contract::kEffectIdEyeContact);
  d->available_effects.erase(
      std::remove(d->available_effects.begin(), d->available_effects.end(),
                  effect_id),
      d->available_effects.end());
  d->blocked_effects[effect_id] = readiness.reason_code;
}

} // namespace

OpenVulkanDiagnostics DiagnoseOpenVulkanDefault() {
  kernels::ResizeBilinear resize;
  std::string e;
  if (!resize.EnsureInitialized(4, 4, 4, 4, &e)) {
    OpenVulkanDiagnostics d = resize.Diagnostics();
    const auto model_facts = PopulateOpenVulkanModelDiagnostics(&d);
    d.matting_runtime = "none";
    d.device_residency_mode = "unavailable";
    ApplyOpenVulkanMattingReadiness(&d, model_facts);
    if (d.error.empty())
      d.error = e;
    if (d.fallback_reason.empty())
      d.fallback_reason = "open_vulkan_resize_unavailable";
    if (d.blocked_reason.empty())
      d.blocked_reason = d.fallback_reason;
    BlockOpenVulkanVirtualBackground(&d, d.blocked_reason.c_str());
    ApplyOpenVulkanEyeContactReadiness(&d);
    return d;
  }
  kernels::UtilityKernels utility;
  if (!utility.Initialize(&e)) {
    OpenVulkanDiagnostics d = utility.Diagnostics();
    const auto model_facts = PopulateOpenVulkanModelDiagnostics(&d);
    d.matting_runtime = "none";
    d.device_residency_mode = "unavailable";
    ApplyOpenVulkanMattingReadiness(&d, model_facts);
    if (d.error.empty())
      d.error = e;
    if (d.fallback_reason.empty())
      d.fallback_reason = "open_vulkan_utility_kernels_unavailable";
    if (d.blocked_reason.empty())
      d.blocked_reason = d.fallback_reason;
    BlockOpenVulkanVirtualBackground(&d, d.blocked_reason.c_str());
    ApplyOpenVulkanEyeContactReadiness(&d);
    return d;
  }
  OpenVulkanDiagnostics d = resize.Diagnostics();
  d.ok = true;
  d.shader_pipeline_created = true;
  const auto model_facts = PopulateOpenVulkanModelDiagnostics(&d);
  d.matting_runtime = "none";
  d.device_residency_mode = "unavailable";
  d.matting_runtime_created = false;
  d.matting_graph_loaded = false;
  d.input_device_resident = false;
  d.alpha_device_resident = false;
  d.output_device_resident = false;
  ApplyOpenVulkanMattingReadiness(&d, model_facts);
  d.blocked_reason = d.matting_reason_code;
  d.degraded_reason = "[" + d.matting_reason_code + "]";
  if (!d.matting_blocker_code.empty())
    d.degraded_reason += " [" + d.matting_blocker_code + "]";
  if (!d.matting_detail.empty())
    d.degraded_reason += " " + d.matting_detail;
  d.warnings.push_back(
      "The milestone-4 ncnn Vulkan spike used CPU Mat input/output; it is not "
      "used for production Open Vulkan virtual background.");
  BlockOpenVulkanVirtualBackground(&d, d.matting_reason_code.c_str());
  ApplyOpenVulkanEyeContactReadiness(&d);
  return d;
}

} // namespace studiocast::vulkan
