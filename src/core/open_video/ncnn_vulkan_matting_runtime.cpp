#include "core/open_video/vulkan_matting_runtime.h"
#if STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
#include "core/open_video/ncnn_vulkan_context.h"
#include "core/open_video/ncnn_vulkan_context_internal.h"
#include "core/vulkan/vulkan_image.h"
#include "core/vulkan/vulkan_tensor.h"
#include <allocator.h>
#include <array>
#include <command.h>
#include <cstring>
#include <layer.h>
#include <layer_type.h>
#include <limits>
#include <mutex>
#include <net.h>
#include <vector>

namespace studiocast::open_vulkan {
namespace {
namespace vk = studiocast::vulkan;
template <class To, class From> To Handle(From value) {
  return reinterpret_cast<To>(value);
}
using Result = VulkanMattingRuntimeResult;
using Failure = VulkanMattingRuntimeFailure;
struct AllocationBudget {
  std::size_t bytes = 0, resources = 0;
  std::size_t cap = kNcnnVulkanMattingAllocationBudgetBytes;
  std::size_t resource_cap = kNcnnVulkanMattingAllocationResourceLimit;
  bool frozen = false, rejected = false;
};
// Record-once arena: fastFree never releases a buffer referenced by the
// recording.
class ResidentAllocator final : public ncnn::VkAllocator {
public:
  struct Block {
    std::unique_ptr<vk::VulkanBuffer> storage;
    ncnn::VkBufferMemory memory{};
    bool free = false;
  };
  ResidentAllocator(vk::VulkanDevice *device, const ncnn::VulkanDevice *native,
                    AllocationBudget *budget, bool staging)
      : ncnn::VkAllocator(native), device_(device), budget_(budget),
        staging_(staging) {
    mappable = staging;
    coherent = false;
    cached = false;
  }
  ncnn::VkBufferMemory *fastMalloc(std::size_t size) override {
    if (!size || size > std::numeric_limits<std::size_t>::max() - 255)
      return nullptr;
    size = (size + 255) & ~std::size_t(255);
    if (!budget_->frozen) {
      Block *best = nullptr;
      for (const auto &block : blocks)
        if (block->free && block->memory.capacity >= size &&
            (!best || block->memory.capacity < best->memory.capacity))
          best = block.get();
      if (best) {
        best->free = false;
        return &best->memory;
      }
    }
    if (budget_->frozen || size > budget_->cap - budget_->bytes ||
        budget_->resources >= budget_->resource_cap) {
      budget_->rejected = true;
      return nullptr;
    }
    auto block = std::make_unique<Block>();
    block->storage = std::make_unique<vk::VulkanBuffer>();
    std::string error;
    const auto required = staging_ ? vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                   : vk::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (!block->storage->Allocate(device_, size,
                                  vk::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      vk::VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                      vk::VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  required, required, staging_, &error,
                                  budget_->cap - budget_->bytes)) {
      budget_->rejected = true;
      return nullptr;
    }
    auto &memory = block->memory;
    memory.buffer = Handle<VkBuffer>(block->storage->buffer());
    memory.offset = 0;
    memory.capacity = size;
    memory.memory = Handle<VkDeviceMemory>(block->storage->memory());
    memory.mapped_ptr = block->storage->mapped();
    memory.memory_type_index = block->storage->memory_type_index();
    memory.stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    budget_->bytes += block->storage->allocation_size();
    ++budget_->resources;
    auto *out = &memory;
    blocks.push_back(std::move(block));
    return out;
  }
  void fastFree(ncnn::VkBufferMemory *memory) override {
    for (const auto &block : blocks)
      if (&block->memory == memory) {
        block->free = true;
        return;
      }
  }
  void ReleaseFreeBlocks() {
    for (auto it = blocks.begin(); it != blocks.end();) {
      if ((*it)->free) {
        budget_->bytes -= (*it)->storage->allocation_size();
        --budget_->resources;
        it = blocks.erase(it);
      } else
        ++it;
    }
  }
  ncnn::VkImageMemory *fastMalloc(int, int, int, std::size_t, int) override {
    budget_->rejected = true;
    return nullptr;
  }
  void fastFree(ncnn::VkImageMemory *) override {}
  int flush(ncnn::VkBufferMemory *ptr) override {
    for (const auto &block : blocks)
      if (&block->memory == ptr) {
        std::string error;
        return block->storage->Flush(&error) ? 0 : -1;
      }
    return -1;
  }
  int invalidate(ncnn::VkBufferMemory *) override {
    budget_->rejected = true;
    return -1;
  }
  std::vector<std::unique_ptr<Block>> blocks;

private:
  vk::VulkanDevice *device_;
  AllocationBudget *budget_;
  bool staging_;
};
struct Program {
  std::shared_ptr<detail::NcnnVulkanContext> owner;
  std::shared_ptr<vk::detail::VulkanContextState> context;
  AllocationBudget budget;
  std::unique_ptr<ResidentAllocator> resident, staging;
  std::unique_ptr<ncnn::Net> net;
  std::unique_ptr<ncnn::VkTransfer> upload;
  std::unique_ptr<ncnn::VkCompute> inference;
  std::unique_ptr<ncnn::Extractor> extractor;
  ncnn::VkMat input, output;
  ncnn::VkBufferMemory external_input{};
  vk::VkCommandBuffer prefix = nullptr, suffix = nullptr,
                      initialization = nullptr;
  std::array<vk::VkCommandBuffer, 3> commands{};
  ~Program() {
    if (context && context->SafeToDestroyChildren() &&
        context->f().vkFreeCommandBuffers) {
      if (initialization)
        context->f().vkFreeCommandBuffers(
            context->device, context->command_pool, 1, &initialization);
      if (prefix)
        context->f().vkFreeCommandBuffers(context->device,
                                          context->command_pool, 1, &prefix);
      if (suffix)
        context->f().vkFreeCommandBuffers(context->device,
                                          context->command_pool, 1, &suffix);
    }
  }
};
void Quarantine(std::unique_ptr<Program> program) {
  static auto *mutex = new std::mutex;
  static auto *programs = new std::vector<std::unique_ptr<Program>>;
  std::lock_guard lock(*mutex);
  programs->push_back(std::move(program));
}
class NcnnVulkanMattingRuntime final : public VulkanMattingRuntime {
public:
  NcnnVulkanMattingRuntime() {
    evidence_.runtime_name = "ncnn-vk-replay";
    evidence_.adapter_available = evidence_.production_adapter = true;
  }
  ~NcnnVulkanMattingRuntime() override {
    if (program_ && program_->context &&
        !program_->context->SafeToDestroyChildren())
      Quarantine(std::move(program_));
  }
  Result ConfigureResourceBudget(
      const VulkanMattingPersistentResources &resources) override {
    if (!resources.allocation_budget_bytes ||
        !resources.allocation_resource_limit ||
        resources.allocation_budget_bytes >
            kNcnnVulkanMattingAllocationBudgetBytes ||
        resources.allocation_resource_limit >
            kNcnnVulkanMattingAllocationResourceLimit)
      return Fail(Failure::allocation_contract_failed,
                  "Invalid native reservation.");
    resources_ = resources;
    return Result::Success();
  }
  Result Initialize(const VulkanMattingDeviceContext &device) override {
    device_ = static_cast<vk::VulkanDevice *>(
        const_cast<void *>(device.ownership_domain));
    if (!device_ || !device_->Initialized())
      return Fail(Failure::unavailable, "No adopted ncnn context.");
    auto native = device_->NativeOwner(detail::kNcnnOwnerKind);
    if (!native)
      return Fail(Failure::device_identity_mismatch,
                  "Matting requires an adopted ncnn-owned device.");
    program_ = std::make_unique<Program>();
    program_->budget.cap = resources_.allocation_budget_bytes;
    program_->budget.resource_cap = resources_.allocation_resource_limit;
    program_->owner =
        std::static_pointer_cast<detail::NcnnVulkanContext>(native);
    program_->budget.bytes =
        program_->owner->device->studiocast_dummy_allocation_bytes();
    program_->budget.resources =
        program_->owner->device->studiocast_dummy_allocation_count();
    if (program_->budget.bytes > program_->budget.cap ||
        program_->budget.resources > program_->budget.resource_cap)
      return Fail(Failure::allocation_contract_failed,
                  "Native bootstrap exceeds caller reservation.");
    program_->context =
        std::static_pointer_cast<vk::detail::VulkanContextState>(
            device_->RetainContext());
    if (Handle<std::uintptr_t>(program_->owner->device->vkdevice()) !=
            device.logical_device ||
        Handle<std::uintptr_t>(program_->owner->queue) != device.queue)
      return Fail(Failure::device_identity_mismatch,
                  "ncnn device or queue differs from StudioCast.");
    evidence_.active_device = device;
    evidence_.runtime_created = evidence_.device_identity_matches = true;
    evidence_.shared_device_imported = evidence_.queue_ownership_explicit =
        true;
    evidence_.synchronous_completion = true;
    program_->resident = std::make_unique<ResidentAllocator>(
        device_, program_->owner->device.get(), &program_->budget, false);
    program_->staging = std::make_unique<ResidentAllocator>(
        device_, program_->owner->device.get(), &program_->budget, true);
    return Result::Success();
  }
  Result LoadGraph(const VulkanMattingGraphDescriptor &graph) override {
    graph_ = graph;
    if (!program_ || graph.input_n != 1 || graph.input_c != 3 ||
        graph.output_n != 1 || graph.output_c != 1 || graph.input_h <= 0 ||
        graph.input_w <= 0 || graph.output_h <= 0 || graph.output_w <= 0 ||
        (std::size_t(graph.input_h) * std::size_t(graph.input_w)) % 4 != 0)
      return Fail(Failure::invalid_shape, "ncnn replay requires batch one RGB "
                                          "and aligned NCHW float32 planes.");
    program_->net = std::make_unique<ncnn::Net>();
    auto &net = *program_->net;
    net.set_vulkan_device(program_->owner->device.get());
    auto &opt = net.opt;
    opt.use_vulkan_compute = true;
    opt.lightmode = true;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = opt.use_fp16_storage = opt.use_fp16_arithmetic =
        false;
    opt.use_bf16_packed = opt.use_bf16_storage = false;
    opt.use_int8_packed = opt.use_int8_storage = opt.use_int8_arithmetic =
        false;
    opt.use_tensor_storage = false;
    opt.use_weights_in_host_memory = false;
    opt.use_local_pool_allocator = false;
    opt.blob_vkallocator = opt.workspace_vkallocator = program_->resident.get();
    opt.staging_vkallocator = program_->staging.get();
    if (net.load_param(graph.param_path.c_str()) != 0)
      return Fail(Failure::invalid_graph, "ncnn parameter load failed.");
    for (const auto *layer : net.layers()) {
      if (layer && layer->typeindex == ncnn::LayerType::MultiHeadAttention)
        return Fail(Failure::invalid_graph,
                    "MultiHeadAttention requires unsupported internal queue "
                    "submission.");
      if (!layer ||
          (layer->typeindex != ncnn::LayerType::Input &&
           (!layer->support_vulkan || (layer->featmask & (1 << 4))))) {
        evidence_.cpu_layers_used = true;
        return Fail(Failure::cpu_layer_rejected,
                    "Graph contains a layer without Vulkan execution.");
      }
    }
    program_->upload =
        std::make_unique<ncnn::VkTransfer>(program_->owner->device.get(), true);
    if (!program_->upload->studiocast_valid())
      return Fail(Failure::out_of_memory,
                  "ncnn upload command infrastructure failed.");
    opt.studiocast_model_upload = program_->upload.get();
    if (net.load_model(graph.bin_path.c_str()) != 0 ||
        program_->budget.rejected)
      return Fail(program_->budget.rejected
                      ? Failure::allocation_contract_failed
                      : Failure::invalid_graph,
                  "ncnn model load or bounded allocation failed.");
    VkCommandBuffer upload = VK_NULL_HANDLE;
    std::string error;
    if (program_->upload->studiocast_finalize(&upload) != 0 ||
        !device_->SubmitAndWait(Handle<vk::VkCommandBuffer>(upload), &error))
      return DriverFailure(error);
    if (program_->upload->reset() != 0)
      return Fail(Failure::synchronization_failed,
                  "Completed model upload could not reset staging ownership.");
    program_->staging->ReleaseFreeBlocks();
    for (const auto *layer : net.layers()) {
      if (!layer ||
          (layer->typeindex != ncnn::LayerType::Input &&
           (!layer->support_vulkan || (layer->featmask & (1 << 4))))) {
        evidence_.cpu_layers_used = true;
        return Fail(Failure::cpu_layer_rejected,
                    "Graph disabled Vulkan during pipeline construction.");
      }
    }
    evidence_.graph_loaded = true;
    return Result::Success();
  }
  Result PreparePersistentResources(
      const VulkanMattingPersistentResources &resources) override {
    resources_ = resources;
    vk::VulkanTensorSize input_size, output_size;
    std::string shape_error;
    const auto row_bytes = std::size_t(graph_.output_w) * sizeof(float);
    const auto rows_minus_one = std::size_t(graph_.output_h - 1);
    const bool stride_safe =
        resources.alpha_row_stride_bytes >= row_bytes &&
        (rows_minus_one == 0 ||
         resources.alpha_row_stride_bytes <=
             (std::numeric_limits<std::size_t>::max() - row_bytes) /
                 rows_minus_one);
    const auto alpha_extent =
        stride_safe
            ? rows_minus_one * resources.alpha_row_stride_bytes + row_bytes
            : 0;
    if (!program_ ||
        !vk::CheckedNchwF32Size(1, 3, graph_.input_h, graph_.input_w,
                                &input_size, &shape_error) ||
        !vk::CheckedNchwF32Size(1, 1, graph_.output_h, graph_.output_w,
                                &output_size, &shape_error) ||
        resources.input_bytes != input_size.bytes ||
        resources.alpha_bytes != output_size.bytes || !resources.input_buffer ||
        !resources.alpha_buffer ||
        resources.input_buffer == resources.alpha_buffer ||
        resources.input_offset || resources.alpha_offset || !stride_safe ||
        resources.alpha_buffer_capacity_bytes < alpha_extent ||
        resources.allocation_budget_bytes < program_->budget.bytes ||
        resources.allocation_resource_limit < program_->budget.resources)
      return Fail(Failure::allocation_contract_failed,
                  "Invalid fixed GPU binding shape/capacity or insufficient "
                  "native allocation reservation.");
    program_->budget.cap = std::min(resources.allocation_budget_bytes,
                                    kNcnnVulkanMattingAllocationBudgetBytes);
    program_->budget.resource_cap =
        std::min(resources.allocation_resource_limit,
                 kNcnnVulkanMattingAllocationResourceLimit);
    auto &memory = program_->external_input;
    memory.buffer = Handle<VkBuffer>(resources.input_buffer);
    memory.offset = resources.input_offset;
    memory.capacity = resources.input_bytes;
    memory.stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    memory.access_flags = VK_ACCESS_SHADER_WRITE_BIT;
    program_->input =
        ncnn::VkMat(graph_.input_w, graph_.input_h, graph_.input_c, &memory,
                    sizeof(float), program_->resident.get());
    // ncnn light mode inspects input refcounts for in-place safety. This
    // descriptor is borrowed and its allocator never recycles the foreign
    // buffer.
    memory.refcount = 1;
    program_->input.refcount = &memory.refcount;
    program_->inference =
        std::make_unique<ncnn::VkCompute>(program_->owner->device.get(), true);
    if (!program_->inference->studiocast_valid())
      return Fail(Failure::out_of_memory,
                  "ncnn replay command infrastructure failed.");
    program_->extractor =
        std::make_unique<ncnn::Extractor>(program_->net->create_extractor());
    const int input_result =
        program_->extractor->input(graph_.input_blob.c_str(), program_->input);
    const int extract_result = input_result == 0
                                   ? program_->extractor->extract(
                                         graph_.output_blob.c_str(),
                                         program_->output, *program_->inference)
                                   : input_result;
    if (input_result != 0 || extract_result != 0 || program_->budget.rejected)
      return Fail(
          program_->budget.rejected ? Failure::allocation_contract_failed
                                    : Failure::invalid_graph,
          "ncnn fixed graph recording failed: input=" +
              std::to_string(input_result) +
              ", extract=" + std::to_string(extract_result) +
              ", input dims=" + std::to_string(program_->input.dims) +
              ", blobs=" + std::to_string(program_->net->blobs().size()) +
              ", native resources=" +
              std::to_string(program_->budget.resources) +
              ", native bytes=" + std::to_string(program_->budget.bytes));
    const auto &out = program_->output;
    if (!out.data || out.w != graph_.output_w || out.h != graph_.output_h ||
        out.c != 1 || out.elempack != 1 || out.elemsize != sizeof(float) ||
        out.data->mapped_ptr ||
        !program_->owner->device->is_device_local(out.data->memory_type_index))
      return Fail(Failure::residency_check_failed,
                  "ncnn output shape, packing, precision, or residency differs "
                  "from model contract.");
    VkCommandBuffer inference = VK_NULL_HANDLE;
    if (program_->inference->studiocast_finalize(&inference) != 0)
      return Fail(Failure::cpu_readback_rejected,
                  "Recording contains a host bridge or could not finalize.");
    program_->commands[1] = Handle<vk::VkCommandBuffer>(inference);
    std::string error;
    if (!RecordCopies(&error) || !InitializeInput(&error))
      return DriverFailure(error);
    program_->budget.frozen = true;
    evidence_.persistent_allocation_count = program_->budget.resources;
    evidence_.persistent_resources_allocated = true;
    evidence_.input_device_resident = evidence_.alpha_device_resident =
        evidence_.output_device_resident = true;
    evidence_.bounded_reusable_allocations = true;
    return Result::Success();
  }
  Result Warmup() override {
    if (!program_ || !program_->commands[1])
      return Fail(Failure::invalid_graph, "No recorded graph.");
    std::string error;
    // Setup initialized this persistent input on GPU before recording replay.
    if (!device_->SubmitAndWait(program_->commands.data(), 3, &error))
      return DriverFailure(error);
    evidence_.warmup_complete = true;
    ++evidence_.warmup_inference_count;
    ++evidence_.completion_count;
    return Result::Success();
  }
  Result Run(const VulkanMattingBufferBinding &input,
             const VulkanMattingBufferBinding &alpha) override {
    if (input.buffer != resources_.input_buffer ||
        alpha.buffer != resources_.alpha_buffer ||
        input.byte_offset != resources_.input_offset ||
        alpha.byte_offset != resources_.alpha_offset)
      return Fail(Failure::device_identity_mismatch,
                  "Recorded ncnn bindings changed; reprepare the session.");
    std::string error;
    if (!device_->SubmitAndWait(program_->commands.data(), 3, &error))
      return DriverFailure(error);
    ++evidence_.inference_count;
    ++evidence_.completion_count;
    return Result::Success();
  }
  VulkanMattingRuntimeEvidence Evidence() const override { return evidence_; }
  void
  RefreshExecutionEvidence(VulkanMattingRuntimeEvidence *out) const override {
    out->warmup_complete = evidence_.warmup_complete;
    out->warmup_inference_count = evidence_.warmup_inference_count;
    out->inference_count = evidence_.inference_count;
    out->completion_count = evidence_.completion_count;
    out->dynamic_allocation_count = evidence_.dynamic_allocation_count;
    out->cpu_readback_count = evidence_.cpu_readback_count;
    out->cpu_layers_used = evidence_.cpu_layers_used;
    out->active_device.context_healthy =
        program_ && program_->context->HealthState().healthy();
  }

private:
  Result Fail(Failure failure, std::string detail) {
    return Result::Failure(failure, std::move(detail));
  }
  Result DriverFailure(std::string detail) {
    const auto health = device_->health_state();
    return Fail(health.health == vk::VulkanContextHealth::device_lost
                    ? Failure::device_lost
                    : Failure::synchronization_failed,
                detail.empty() ? "ncnn Vulkan command finalization failed."
                               : std::move(detail));
  }
  bool InitializeInput(std::string *error) {
    auto &p = *program_;
    auto *zeros = p.staging->fastMalloc(resources_.input_bytes);
    if (!zeros || !zeros->mapped_ptr)
      return false;
    std::memset(zeros->mapped_ptr, 0, resources_.input_bytes);
    if (p.staging->flush(zeros) != 0)
      return false;
    const auto &f = device_->f();
    vk::VkCommandBufferAllocateInfo alloc;
    alloc.commandPool = device_->command_pool();
    alloc.commandBufferCount = 1;
    if (!device_->CheckDriverResult(
            f.vkAllocateCommandBuffers(device_->device(), &alloc,
                                       &p.initialization),
            "ncnn warmup input allocation", true, error))
      return false;
    vk::VkCommandBufferBeginInfo begin;
    if (!device_->CheckDriverResult(
            f.vkBeginCommandBuffer(p.initialization, &begin),
            "ncnn warmup input begin", true, error))
      return false;
    vk::VkBufferCopy copy;
    copy.size = resources_.input_bytes;
    f.vkCmdCopyBuffer(p.initialization, Handle<vk::VkBuffer>(zeros->buffer),
                      Handle<vk::VkBuffer>(resources_.input_buffer), 1, &copy);
    if (!device_->RecordBufferBarrier(
            p.initialization, Handle<vk::VkBuffer>(resources_.input_buffer),
            resources_.input_bytes, device_->context_identity(),
            vk::VulkanBufferAccess::transfer_write,
            vk::VulkanBufferAccess::compute_read, error) ||
        !device_->CheckDriverResult(f.vkEndCommandBuffer(p.initialization),
                                    "ncnn warmup input end", true, error))
      return false;
    return device_->SubmitAndWait(p.initialization, error);
  }
  bool RecordCopies(std::string *error) {
    auto &p = *program_;
    const auto &f = device_->f();
    vk::VkCommandBufferAllocateInfo alloc;
    alloc.commandPool = device_->command_pool();
    alloc.commandBufferCount = 1;
    if (!device_->CheckDriverResult(
            f.vkAllocateCommandBuffers(device_->device(), &alloc, &p.prefix),
            "ncnn prefix allocation", true, error) ||
        !device_->CheckDriverResult(
            f.vkAllocateCommandBuffers(device_->device(), &alloc, &p.suffix),
            "ncnn suffix allocation", true, error))
      return false;
    vk::VkCommandBufferBeginInfo begin;
    begin.flags = 0;
    if (!device_->CheckDriverResult(f.vkBeginCommandBuffer(p.prefix, &begin),
                                    "ncnn prefix begin", true, error))
      return false;
    // Replay starts with a dependency for every retained activation, including
    // temporary workspaces whose allocator metadata may have been recycled.
    for (const auto &block : p.resident->blocks) {
      vk::VkBufferMemoryBarrier barrier;
      barrier.srcAccessMask =
          vk::VK_ACCESS_SHADER_WRITE_BIT | vk::VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask =
          vk::VK_ACCESS_SHADER_READ_BIT | vk::VK_ACCESS_SHADER_WRITE_BIT |
          vk::VK_ACCESS_TRANSFER_READ_BIT | vk::VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.buffer = block->storage->buffer();
      barrier.size = block->storage->size();
      f.vkCmdPipelineBarrier(p.prefix,
                             vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 vk::VK_PIPELINE_STAGE_TRANSFER_BIT,
                             vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 vk::VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &barrier, 0, nullptr);
    }
    vk::VkBufferMemoryBarrier input_barrier;
    input_barrier.srcAccessMask =
        vk::VK_ACCESS_SHADER_WRITE_BIT | vk::VK_ACCESS_TRANSFER_WRITE_BIT;
    input_barrier.dstAccessMask = vk::VK_ACCESS_SHADER_READ_BIT;
    input_barrier.buffer = Handle<vk::VkBuffer>(resources_.input_buffer);
    input_barrier.size = resources_.input_bytes;
    f.vkCmdPipelineBarrier(p.prefix,
                           vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               vk::VK_PIPELINE_STAGE_TRANSFER_BIT,
                           vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                           nullptr, 1, &input_barrier, 0, nullptr);
    if (!device_->CheckDriverResult(f.vkEndCommandBuffer(p.prefix),
                                    "ncnn prefix end", true, error) ||
        !device_->CheckDriverResult(f.vkBeginCommandBuffer(p.suffix, &begin),
                                    "ncnn suffix begin", true, error))
      return false;
    if (!device_->RecordBufferBarrier(
            p.suffix, Handle<vk::VkBuffer>(p.output.buffer()),
            p.output.data->capacity, device_->context_identity(),
            vk::VulkanBufferAccess::compute_write,
            vk::VulkanBufferAccess::transfer_read, error))
      return false;
    std::vector<vk::VkBufferCopy> rows;
    rows.reserve(static_cast<std::size_t>(graph_.output_h));
    for (int y = 0; y < graph_.output_h; ++y) {
      vk::VkBufferCopy row;
      row.srcOffset = p.output.data->offset +
                      std::size_t(y) *
                          static_cast<std::size_t>(graph_.output_w) *
                          sizeof(float);
      row.dstOffset = resources_.alpha_offset +
                      std::size_t(y) * resources_.alpha_row_stride_bytes;
      row.size = std::size_t(graph_.output_w) * sizeof(float);
      rows.push_back(row);
    }
    f.vkCmdCopyBuffer(p.suffix, Handle<vk::VkBuffer>(p.output.buffer()),
                      Handle<vk::VkBuffer>(resources_.alpha_buffer),
                      static_cast<std::uint32_t>(rows.size()), rows.data());
    if (!device_->RecordBufferBarrier(
            p.suffix, Handle<vk::VkBuffer>(resources_.alpha_buffer),
            resources_.alpha_row_stride_bytes * std::size_t(graph_.output_h),
            device_->context_identity(), vk::VulkanBufferAccess::transfer_write,
            vk::VulkanBufferAccess::compute_read, error) ||
        !device_->CheckDriverResult(f.vkEndCommandBuffer(p.suffix),
                                    "ncnn suffix end", true, error))
      return false;
    p.commands[0] = p.prefix;
    p.commands[2] = p.suffix;
    return true;
  }
  vk::VulkanDevice *device_ = nullptr;
  std::unique_ptr<Program> program_;
  VulkanMattingGraphDescriptor graph_;
  VulkanMattingPersistentResources resources_;
  VulkanMattingRuntimeEvidence evidence_;
};
} // namespace
std::unique_ptr<VulkanMattingRuntime> CreateNcnnVulkanMattingRuntime() {
  return std::make_unique<NcnnVulkanMattingRuntime>();
}
} // namespace studiocast::open_vulkan
#endif
