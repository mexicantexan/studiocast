#include "core/vulkan/kernels/resize_bilinear.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "core/open_video/model_pack_registry.h"
#include "core/open_video/ncnn_vulkan_context.h"
#include "core/open_video/vulkan_matting_runtime.h"
#include "core/open_video/vulkan_matting_session.h"
#include "core/util/xdg.h"
#include "core/video/effects/broadcast_effect_contract.h"
#include "core/video/open_vulkan_auto_frame.h"
#include "core/video/open_vulkan_eye_contact.h"
#include "core/video/open_vulkan_mirror.h"
#include "core/video/open_vulkan_video_noise_removal.h"
#include "core/video/open_vulkan_vignette.h"
#include "core/video/open_vulkan_virtual_background_blur.h"
#include "core/video/open_vulkan_virtual_background_remove.h"
#include "core/video/open_vulkan_virtual_background_replace.h"
#include "core/video/open_vulkan_virtual_key_light.h"
#include "core/vulkan/kernels/shaders/resize_rgb24_bilinear_spv.h"
#include "core/vulkan/kernels/utility_kernels.h"
#include "core/vulkan/video_execution_factory.h"
#include "core/vulkan/video_execution_factory_internal.h"

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
  std::optional<open_video::ModelPack> pack;
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
  for (const auto *task :
       {"video_denoise", "face_detection", "face_landmarks", "eye_contact"}) {
    const auto id = reg.DefaultModelIdForTask(task);
    if (!id.empty() && reg.Find(task, id))
      d->default_model_ids[task] = id;
  }
  if (!d->default_model_id.empty())
    d->default_model_ids["matting"] = d->default_model_id;
  facts.selected = !d->default_model_id.empty();
  if (facts.selected)
    facts.pack = reg.Find("matting", d->default_model_id);
  // Only a prepared session may attest validation. Explicit refresh can prepare
  // one below; normal status polling reads the daemon's cached snapshot.
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
    OpenVulkanDiagnostics *d, const OpenVulkanMattingModelFacts &model_facts,
    const open_vulkan::VulkanMattingReadiness *prepared = nullptr) {
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
      prepared ? *prepared
               : studiocast::open_vulkan::EvaluateVulkanMattingReadiness(input);
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
  d->blocked_effects[std::string(
      studiocast::video::effects::contract::kEffectIdVirtualKeyLight)] =
      reason_code;
}

void SetOpenVulkanEffectAvailability(OpenVulkanDiagnostics *d,
                                     std::string_view effect_id,
                                     bool production_ready,
                                     std::string_view blocker_code) {
  if (!d)
    return;
  const std::string id(effect_id);
  d->available_effects.erase(
      std::remove(d->available_effects.begin(), d->available_effects.end(), id),
      d->available_effects.end());
  d->blocked_effects.erase(id);
  if (production_ready) {
    d->available_effects.push_back(id);
  } else {
    d->blocked_effects[id] = std::string(blocker_code);
  }
}

void ApplyOpenVulkanPixelEffectReadiness(OpenVulkanDiagnostics *d) {
  if (!d)
    return;

  const auto mirror = studiocast::video::EvaluateOpenVulkanMirrorReadiness(*d);
  d->mirror_production_ready = mirror.production_ready;
  d->mirror_readiness_code =
      mirror.production_ready
          ? std::string(
                studiocast::video::kOpenVulkanMirrorProductionReadyReason)
          : std::string{};
  d->mirror_blocker_code =
      mirror.production_ready ? std::string{} : mirror.shared_reason_code;
  SetOpenVulkanEffectAvailability(
      d, studiocast::video::effects::contract::kEffectIdMirror,
      mirror.production_ready, d->mirror_blocker_code);

  const auto vignette =
      studiocast::video::EvaluateOpenVulkanVignetteReadiness(*d);
  d->vignette_fixed_center_production_ready = vignette.production_ready;
  d->vignette_readiness_code =
      vignette.production_ready
          ? std::string(studiocast::video::
                            kOpenVulkanVignetteFixedCenterProductionReadyReason)
          : std::string{};
  d->vignette_blocker_code =
      vignette.production_ready ? std::string{} : vignette.shared_reason_code;
  d->vignette_parameter_contract =
      std::string(studiocast::video::kOpenVulkanVignetteParameterContract);
  SetOpenVulkanEffectAvailability(
      d, studiocast::video::effects::contract::kEffectIdVignette,
      vignette.production_ready, d->vignette_blocker_code);
}

void ApplyOpenVulkanAutoFrameReleaseReadiness(OpenVulkanDiagnostics *d) {
  if (!d)
    return;
  d->auto_frame_crop_stage_implemented = true;
  d->auto_frame_production_ready = false;
  d->auto_frame_readiness_code.clear();
  d->auto_frame_blocker_code =
      std::string(studiocast::video::kOpenVulkanAutoFrameFaceProviderReason);
  d->auto_frame_cpu_tail = false;
  d->auto_frame_degraded_reason_code.clear();
  d->auto_frame_selectable_cpu_fallback = false;
  SetOpenVulkanEffectAvailability(
      d, studiocast::video::effects::contract::kEffectIdAutoFrame, false,
      d->auto_frame_blocker_code);
}

void ApplyOpenVulkanEyeContactReadiness(
    OpenVulkanDiagnostics *d,
    const video::OpenVulkanEyeContactCapabilityFacts *prepared = nullptr) {
  if (!d)
    return;

  auto current_facts =
      prepared ? *prepared
               : studiocast::video::CurrentOpenVulkanEyeContactFacts();
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
  d->available_effects.erase(std::remove(d->available_effects.begin(),
                                         d->available_effects.end(), effect_id),
                             d->available_effects.end());
  SetOpenVulkanEffectAvailability(d, effect_id, readiness.production_ready,
                                  readiness.reason_code);
}

void ApplyOpenVulkanVideoNoiseRemovalReadiness(
    OpenVulkanDiagnostics *d,
    const video::OpenVulkanVideoNoiseRemovalCapabilityFacts *prepared =
        nullptr) {
  if (!d)
    return;

  auto current_facts =
      prepared ? *prepared
               : studiocast::video::CurrentOpenVulkanVideoNoiseRemovalFacts();
  current_facts.non_cpu_device_selected = d->non_cpu_device_selected;
  current_facts.compute_queue_available = d->compute_queue_available;
  current_facts.context_healthy = d->context_healthy;
  const auto readiness =
      studiocast::video::EvaluateOpenVulkanVideoNoiseRemovalReadiness(
          current_facts);
  const auto &facts = readiness.facts;
  d->video_noise_removal_production_ready = readiness.production_ready;
  d->video_noise_removal_reason_code = readiness.reason_code;
  d->video_noise_removal_blocker_code = readiness.blocker_code;
  d->video_noise_removal_detail = readiness.detail;
  d->video_noise_removal_backend_compiled = facts.backend_compiled;
  d->video_noise_removal_live_stage_implemented = facts.live_stage_implemented;
  d->video_noise_removal_production_adapter_available =
      facts.production_adapter_available;
  d->video_noise_removal_vulkan_inference_provider_available =
      facts.vulkan_inference_provider_available;
  d->video_noise_removal_non_cpu_device_selected =
      facts.non_cpu_device_selected;
  d->video_noise_removal_compute_queue_available =
      facts.compute_queue_available;
  d->video_noise_removal_context_healthy = facts.context_healthy;
  d->video_noise_removal_shared_device_imported = facts.shared_device_imported;
  d->video_noise_removal_queue_ownership_explicit =
      facts.queue_ownership_explicit;
  d->video_noise_removal_model_pack_selected = facts.model_pack_selected;
  d->video_noise_removal_artifact_contract_validated =
      facts.artifact_contract_validated;
  d->video_noise_removal_fully_device_resident_tensor_io =
      facts.fully_device_resident_tensor_io;
  d->video_noise_removal_device_resident_preprocess =
      facts.device_resident_preprocess;
  d->video_noise_removal_device_resident_postprocess =
      facts.device_resident_postprocess;
  d->video_noise_removal_warmup_complete = facts.warmup_complete;
  d->video_noise_removal_synchronization_contract_validated =
      facts.synchronization_contract_validated;
  d->video_noise_removal_bounded_reusable_allocations =
      facts.bounded_reusable_allocations;
  d->video_noise_removal_temporal_history_device_resident =
      facts.temporal_history_device_resident;
  d->video_noise_removal_temporal_history_bounded =
      facts.temporal_history_bounded;
  d->video_noise_removal_history_reset_on_disable =
      facts.history_reset_on_disable;
  d->video_noise_removal_history_reset_on_reconfigure =
      facts.history_reset_on_reconfigure;
  d->video_noise_removal_capture_sequence_discontinuity_reset =
      facts.capture_sequence_discontinuity_reset;
  d->video_noise_removal_parity_validated = facts.parity_validated;
  d->video_noise_removal_selectable_cpu_fallback =
      facts.selectable_cpu_fallback;
  d->video_noise_removal_dispatch_count = facts.dispatch_count;
  d->video_noise_removal_temporal_history_reset_count =
      facts.temporal_history_reset_count;
  d->video_noise_removal_cpu_readback_count = facts.cpu_readback_count;
  d->video_noise_removal_cpu_fallback_count = facts.cpu_fallback_count;

  const std::string effect_id(
      studiocast::video::effects::contract::kEffectIdVideoNoiseRemoval);
  d->available_effects.erase(std::remove(d->available_effects.begin(),
                                         d->available_effects.end(), effect_id),
                             d->available_effects.end());
  SetOpenVulkanEffectAvailability(d, effect_id, readiness.production_ready,
                                  readiness.reason_code);
}

// This attestation is deliberately limited to the converted artifact/profile
// covered by the numerical and production-manager tests. BYO runtime validity
// alone does not claim that an arbitrary model has passed those comparisons.
bool ReviewedMattingProfile(const open_video::ModelPack &pack) {
  if (pack.id != "modnet-webnn-256-fp32" || !pack.ncnn_vulkan || !pack.matting)
    return false;
  const auto &native = *pack.ncnn_vulkan;
  const auto &model = *pack.matting;
  return native.param_sha256 == "4e205032860a9359fa2555c6ef66dece2a8dcd034a1a61"
                                "466c58d1310af8a1bd" &&
         native.bin_sha256 == "38871660dda4dd324304eb572f43a43ddc5c876c8dff50db"
                              "2cf09d29a17e3fa2" &&
         native.input_blob == "input" && native.output_blob == "output" &&
         model.input.name == "input" && model.output.name == "output" &&
         native.precision == "fp32" && model.input.width == 256 &&
         model.input.height == 256 && model.output.width == 256 &&
         model.output.height == 256 && model.preprocess.color == "rgb" &&
         model.preprocess.range == "0..1" &&
         model.preprocess.mean == std::array<double, 3>{0.5, 0.5, 0.5} &&
         model.preprocess.std == std::array<double, 3>{0.5, 0.5, 0.5};
}

// Called only during explicit startup/diagnostics refresh. GET_STATUS consumes
// the daemon cache and never performs this hashing, preparation or inference.
bool PrepareMattingDiagnostics(OpenVulkanDiagnostics *d,
                               OpenVulkanMattingModelFacts *facts,
                               open_vulkan::VulkanMattingReadiness *readiness,
                               VulkanProductionModelDiagnostics *models) {
  if (!open_vulkan::ProductionVulkanMattingBuildEnabled() ||
      !open_vulkan::ProductionVulkanMattingAdapterAvailable())
    return false;
  try {
    VulkanVideoExecutionFactoryConfiguration configuration;
    configuration.input_width = configuration.output_width = 4;
    configuration.input_height = configuration.output_height = 4;
    // One small startup profile shares one caller-owned aggregate reservation.
    // This is a diagnostic bound, not a camera-resolution memory guarantee.
    configuration.reservation.persistent_bytes = UINT64_C(512) << 20;
    configuration.reservation.scratch_bytes = UINT64_C(256) << 20;
    // The flattened 68-point forest is uploaded once (about 63.7 MiB).
    configuration.reservation.upload_staging_bytes = UINT64_C(64) << 20;
    configuration.reservation.readback_staging_bytes = UINT64_C(1) << 20;
    configuration.reservation.resource_count = 2048;
    configuration.reservation.in_flight_slots = 1;
    auto effects = video::effects::BroadcastCameraEffects{};
    const auto registry = open_video::ModelPackRegistry::ScanDefault();
    const auto select = [&](const char *task, const std::string &id,
                            std::shared_ptr<const open_video::ModelPack> *pack,
                            hal::ModelArtifactIdentity *identity) {
      auto candidate = registry.Find(task, id);
      if (!candidate)
        return false;
      const auto kind = std::string(task) == "face_landmarks"
                            ? "dlib_shape_predictor"
                            : "ncnn_param";
      for (const auto &file : candidate->files) {
        if (file.kind != kind)
          continue;
        const auto validated =
            open_video::ValidateModelArtifactIdentity(*candidate, file);
        if (!validated.ok())
          return false;
        identity->model_id_hash = validated.identity.model_id_sha256;
        identity->artifact_set_hash = validated.identity.artifact_set_sha256;
        identity->artifact_generation = validated.identity.registry_generation;
        *pack = std::make_shared<const open_video::ModelPack>(
            std::move(*candidate));
        return true;
      }
      return false;
    };
    if (facts->pack &&
        select("matting", facts->pack->id, &configuration.matting_pack,
               &configuration.matting_identity)) {
      effects.virtual_background.mode =
          video::effects::VirtualBackgroundMode::blur;
      effects.virtual_key_light.enabled = true;
    }
    effects.video_noise_removal.enabled =
        select("video_denoise", registry.DefaultModelIdForTask("video_denoise"),
               &configuration.denoise_pack, &configuration.denoise_identity);
    effects.auto_frame.enabled = select(
        "face_detection", registry.DefaultModelIdForTask("face_detection"),
        &configuration.face_pack, &configuration.face_identity);
    effects.eye_contact.enabled =
        effects.auto_frame.enabled &&
        select("eye_contact", registry.DefaultModelIdForTask("eye_contact"),
               &configuration.eye_pack, &configuration.eye_identity);
    if (effects.eye_contact.enabled) {
      std::string landmark = registry.DefaultModelIdForTask("face_landmarks");
      for (const auto &dependency : configuration.eye_pack->depends_on)
        if (dependency.rfind("face_landmarks:", 0) == 0)
          landmark = dependency.substr(15);
      effects.eye_contact.enabled =
          select("face_landmarks", landmark, &configuration.landmarks_pack,
                 &configuration.landmarks_identity);
    }
    effects.vignette.enabled = effects.auto_frame.enabled;
    effects.vignette.center_on_tracked_face = true;
    for (std::size_t attempt = 0; attempt < 5; ++attempt) {
      configuration.effects =
          std::make_shared<const video::effects::BroadcastCameraEffects>(
              effects);
      VulkanVideoExecutionFactory factory(configuration);
      hal::ReasonId reason{};
      const auto status = factory.Initialize(&reason);
      if (status == hal::StatusId::ready || status == hal::StatusId::degraded) {
        std::string error;
        if (!CollectProductionVulkanModelDiagnostics(factory, models, &error))
          throw std::runtime_error(error);
        *readiness = models->matting_readiness;
        facts->selected = models->matting_pack != nullptr;
        facts->contract_validated =
            models->capabilities.matting_production_ready;
        facts->pack =
            models->matting_pack
                ? std::optional<open_video::ModelPack>(*models->matting_pack)
                : std::nullopt;
        auto prepared = models->device;
        prepared.ok = true;
        prepared.shader_pipeline_created = true;
        prepared.installed_models = std::move(d->installed_models);
        prepared.models = std::move(d->models);
        prepared.default_model_id = std::move(d->default_model_id);
        prepared.default_model_ids = std::move(d->default_model_ids);
        prepared.missing_models = std::move(d->missing_models);
        prepared.install_hints = std::move(d->install_hints);
        prepared.warnings = std::move(d->warnings);
        prepared.matting_parity_validated =
            facts->pack && ReviewedMattingProfile(*facts->pack) &&
            readiness->production_ready;
        *d = std::move(prepared);
        return true;
      }
      const auto failed = factory.initialization_failed_semantic();
      const auto shutdown = factory.RequestShutdown();
      if (!shutdown.quiescent || !shutdown.driver_context_release_permitted)
        throw std::runtime_error(
            "Diagnostic Vulkan owner could not retire safely");
      // Only a known optional component may be omitted after safe retirement.
      if (failed == hal::SemanticEffectId::video_noise_removal &&
          configuration.denoise_pack) {
        configuration.denoise_pack.reset();
        effects.video_noise_removal.enabled = false;
      } else if (failed == hal::SemanticEffectId::eye_contact &&
                 configuration.eye_pack) {
        configuration.eye_pack.reset();
        configuration.landmarks_pack.reset();
        effects.eye_contact.enabled = false;
      } else if (failed == hal::SemanticEffectId::auto_frame &&
                 configuration.face_pack) {
        configuration.face_pack.reset();
        configuration.eye_pack.reset();
        configuration.landmarks_pack.reset();
        effects.auto_frame.enabled = effects.eye_contact.enabled =
            effects.vignette.enabled = false;
      } else if (failed == hal::SemanticEffectId::virtual_background_blur &&
                 configuration.matting_pack) {
        configuration.matting_pack.reset();
        effects.virtual_background.mode =
            video::effects::VirtualBackgroundMode::none;
        effects.virtual_key_light.enabled = false;
      } else {
        throw std::runtime_error(
            "Production Vulkan diagnostic preparation unavailable");
      }
    }
  } catch (const std::exception &error) {
    readiness->production_ready = false;
    readiness->blocker_code =
        open_vulkan::kOpenVulkanMattingRuntimeFailureReason;
    readiness->detail = error.what();
  }
  return true;
}

bool ReviewedNetwork(const std::shared_ptr<const open_video::ModelPack> &pack,
                     const char *role, const char *profile, const char *param,
                     const char *bin) {
  if (!pack)
    return false;
  for (const auto &network : pack->ncnn_vulkan_networks)
    if (network.role == role && network.profile == profile &&
        network.precision == "fp32" && network.converter_name == "pnnx" &&
        network.converter_version == "20261020" &&
        network.param_sha256 == param && network.bin_sha256 == bin)
      return true;
  return false;
}

void ApplyPreparedModelDiagnostics(OpenVulkanDiagnostics *d,
                                   const VulkanProductionModelDiagnostics &m) {
  const auto &c = m.capabilities;
  const auto record_prepared = [&](const auto &pack, bool prepared) {
    if (pack && prepared)
      d->prepared_model_ids[pack->task] = pack->id;
  };
  record_prepared(m.matting_pack, m.matting.warmup_complete);
  record_prepared(m.denoise_pack, m.denoise.warmup_complete);
  record_prepared(m.face_pack, m.face.warmup_complete);
  record_prepared(m.landmarks_pack, c.landmarks_production_ready);
  record_prepared(m.eye_pack,
                  m.left_eye.warmup_complete && m.right_eye.warmup_complete);
  const bool common =
      c.build_enabled && c.runtime_loaded && c.non_cpu_physical_device &&
      c.compute_queue_available && c.owned_context_healthy &&
      c.utility_kernels_ready && c.exact_device_and_queue_shared &&
      c.persistent_device_local_resources && c.bounded_reusable_allocations &&
      c.no_host_intermediate_storage;
  const bool face_reviewed = ReviewedNetwork(
      m.face_pack, "main", "yunet_640_v1",
      "a79870770c2e87ce218950ba471b4fd82f2ec3c9c5908405b05e4e93fa3e8293",
      "8faa696d61ad6bf5c13ed5d6c8ae35e376c660968db53d723a329020856ca5c9");
  auto eye = video::CurrentOpenVulkanEyeContactFacts();
  eye.non_cpu_device_selected = c.non_cpu_physical_device;
  eye.compute_queue_available = c.compute_queue_available;
  eye.context_healthy = c.owned_context_healthy;
  eye.shared_device_imported = m.face.shared_device_imported &&
                               m.left_eye.shared_device_imported &&
                               m.right_eye.shared_device_imported;
  eye.queue_ownership_explicit = m.face.queue_ownership_explicit &&
                                 m.left_eye.queue_ownership_explicit &&
                                 m.right_eye.queue_ownership_explicit;
  eye.model_pack_selected = m.eye_pack && m.face_pack && m.landmarks_pack;
  eye.artifact_contract_validated = eye.device_resident_analysis =
      eye.device_resident_tensor_io =
          c.eye_production_ready && c.landmarks_production_ready &&
          c.shader_float64_enabled;
  eye.warmup_complete = m.face.warmup_complete && m.left_eye.warmup_complete &&
                        m.right_eye.warmup_complete;
  eye.bounded_reusable_allocations = c.bounded_reusable_allocations;
  eye.synchronization_contract_validated = common && eye.warmup_complete;
  eye.parity_validated =
      eye.model_pack_selected && face_reviewed &&
      ReviewedNetwork(
          m.eye_pack, "left", "flx_prefix_v1",
          "7990a15c75c916c04dcc3cab488f4267bd5b0dd6a1d4095d86e1e0bf69341ee9",
          "7597eb7e151ec1de71fb33ce7f511b856f258d538f939727401598e18d180416") &&
      ReviewedNetwork(
          m.eye_pack, "right", "flx_prefix_v1",
          "7990a15c75c916c04dcc3cab488f4267bd5b0dd6a1d4095d86e1e0bf69341ee9",
          "9333359e7c4924312b4af161ae0678471abf7d301c7893474e534c4f9541802b");
  if (eye.parity_validated) {
    const auto landmark = std::find_if(
        m.landmarks_pack->files.begin(), m.landmarks_pack->files.end(),
        [](const auto &f) { return f.kind == "dlib_shape_predictor"; });
    eye.parity_validated =
        landmark != m.landmarks_pack->files.end() &&
        landmark->sha256 ==
            "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f";
  }
  eye.dispatch_count =
      m.left_eye.warmup_inference_count + m.right_eye.warmup_inference_count;
  ApplyOpenVulkanEyeContactReadiness(d, &eye);
  auto denoise = video::CurrentOpenVulkanVideoNoiseRemovalFacts();
  denoise.non_cpu_device_selected = c.non_cpu_physical_device;
  denoise.compute_queue_available = c.compute_queue_available;
  denoise.context_healthy = c.owned_context_healthy;
  denoise.shared_device_imported = m.denoise.shared_device_imported;
  denoise.queue_ownership_explicit = m.denoise.queue_ownership_explicit;
  denoise.model_pack_selected = m.denoise_pack != nullptr;
  denoise.artifact_contract_validated =
      denoise.fully_device_resident_tensor_io =
          denoise.device_resident_preprocess =
              denoise.device_resident_postprocess = c.denoise_production_ready;
  denoise.warmup_complete = m.denoise.warmup_complete;
  denoise.synchronization_contract_validated =
      common && denoise.warmup_complete;
  denoise.bounded_reusable_allocations = c.bounded_reusable_allocations;
  denoise.temporal_history_device_resident = denoise.temporal_history_bounded =
      denoise.history_reset_on_disable = denoise.history_reset_on_reconfigure =
          denoise.capture_sequence_discontinuity_reset =
              c.denoise_production_ready;
  denoise.parity_validated = ReviewedNetwork(
      m.denoise_pack, "main", "fastdvdnet_rgb_v1",
      "f3bb19a906f5c811f9627e167181afa04a8adc76fa2e77e8f1ab00d27535c102",
      "b31f624553b8072fe53fc2f46a060495b8016ad2f0c6ee9d0f66f88604fdec3f");
  denoise.dispatch_count = m.denoise.warmup_inference_count;
  ApplyOpenVulkanVideoNoiseRemovalReadiness(d, &denoise);
  const bool tracking = common && face_reviewed && m.face.warmup_complete &&
                        c.face_production_ready && c.tracking_gpu_ready;
  d->auto_frame_production_ready = tracking;
  d->auto_frame_gpu_tracking_ready = common && c.tracking_gpu_ready;
  d->auto_frame_model_contract_validated = c.face_production_ready;
  d->auto_frame_warmup_complete = m.face.warmup_complete;
  d->auto_frame_parity_validated = face_reviewed;
  d->auto_frame_shared_device_imported = m.face.shared_device_imported;
  d->auto_frame_queue_ownership_explicit = m.face.queue_ownership_explicit;
  d->auto_frame_bounded_reusable_allocations =
      common && m.face.bounded_reusable_allocations;
  d->auto_frame_synchronization_contract_validated =
      common && m.face.synchronous_completion;
  d->auto_frame_cpu_layers_used = m.face.cpu_layers_used;
  d->auto_frame_dynamic_allocation_count = m.face.dynamic_allocation_count;
  d->auto_frame_cpu_readback_count = m.face.cpu_readback_count;
  d->auto_frame_cpu_fallback_count =
      0; // The prepared tracking island has no CPU fallback.
  d->auto_frame_dispatch_count = m.face.warmup_inference_count;
  d->auto_frame_cpu_tail = false;
  d->auto_frame_degraded_reason_code.clear();
  d->auto_frame_readiness_code =
      tracking ? "open_vulkan_auto_frame_production_ready" : "";
  d->auto_frame_blocker_code =
      tracking ? "" : "vulkan_auto_frame_yunet_unavailable";
  SetOpenVulkanEffectAvailability(d, "auto_frame", tracking,
                                  d->auto_frame_blocker_code);
  d->vignette_tracked_center_production_ready =
      tracking && c.vignette_tracked_center_ready;
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
    ApplyOpenVulkanPixelEffectReadiness(&d);
    ApplyOpenVulkanAutoFrameReleaseReadiness(&d);
    ApplyOpenVulkanEyeContactReadiness(&d);
    ApplyOpenVulkanVideoNoiseRemovalReadiness(&d);
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
    ApplyOpenVulkanPixelEffectReadiness(&d);
    ApplyOpenVulkanAutoFrameReleaseReadiness(&d);
    ApplyOpenVulkanEyeContactReadiness(&d);
    ApplyOpenVulkanVideoNoiseRemovalReadiness(&d);
    return d;
  }
  OpenVulkanDiagnostics d = resize.Diagnostics();
  d.ok = true;
  d.shader_pipeline_created = true;
  auto model_facts = PopulateOpenVulkanModelDiagnostics(&d);
  d.matting_runtime = "none";
  d.device_residency_mode = "unavailable";
  d.matting_runtime_created = false;
  d.matting_graph_loaded = false;
  d.input_device_resident = false;
  d.alpha_device_resident = false;
  d.output_device_resident = false;
  open_vulkan::VulkanMattingReadiness prepared;
  VulkanProductionModelDiagnostics models;
  const bool attempted =
      PrepareMattingDiagnostics(&d, &model_facts, &prepared, &models);
  ApplyOpenVulkanMattingReadiness(&d, model_facts,
                                  attempted ? &prepared : nullptr);
  if (d.matting_production_ready) {
    for (const auto effect :
         {"virtual_background.blur", "virtual_background.remove",
          "virtual_background.replace", "virtual_key_light"})
      SetOpenVulkanEffectAvailability(&d, effect, true, "");
  } else {
    d.blocked_reason = d.matting_reason_code;
    d.degraded_reason = "[" + d.matting_reason_code + "]";
    if (!d.matting_blocker_code.empty())
      d.degraded_reason += " [" + d.matting_blocker_code + "]";
    if (!d.matting_detail.empty())
      d.degraded_reason += " " + d.matting_detail;
    BlockOpenVulkanVirtualBackground(&d, d.matting_reason_code.c_str());
  }
  ApplyOpenVulkanPixelEffectReadiness(&d);
  ApplyOpenVulkanAutoFrameReleaseReadiness(&d);
  ApplyOpenVulkanEyeContactReadiness(&d);
  ApplyOpenVulkanVideoNoiseRemovalReadiness(&d);
  if (models.capabilities.build_enabled)
    ApplyPreparedModelDiagnostics(&d, models);
  return d;
}

} // namespace studiocast::vulkan
