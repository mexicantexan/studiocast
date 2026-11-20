#include "core/open_video/ncnn_vulkan_inference.h"

#include "core/util/process_exit.h"

#include "core/open_video/ncnn_vulkan_context_internal.h"
#include "core/vulkan/resident_command_batch.h"
#include "core/vulkan/vulkan_image.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>
#if STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
#include <allocator.h>
#include <command.h>
#include <gpu.h>
#include <layer.h>
#include <layer_type.h>
#include <net.h>
#endif
namespace studiocast::open_vulkan {
namespace {
namespace vk = studiocast::vulkan;
using Result = VulkanMattingRuntimeResult;
using Failure = VulkanMattingRuntimeFailure;
Result Fail(Failure failure, std::string detail) {
  return Result::Failure(failure, std::move(detail));
}
#if STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
template <class To, class From> To Handle(From p) {
  return reinterpret_cast<To>(p);
}
struct AllocationBudget {
  std::size_t bytes = 0, resources = 0;
  std::size_t cap = 0;
  std::size_t resource_cap = 0;
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

#endif
} // namespace
struct NcnnVulkanInference::Impl {
  VulkanMattingRuntimeEvidence evidence;
#if STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
  std::shared_ptr<detail::NcnnVulkanContext> owner;
  std::shared_ptr<vk::detail::VulkanContextState> context;
  std::shared_ptr<VulkanInferenceSharedQuota> quota;
  AllocationBudget budget;
  std::unique_ptr<ResidentAllocator> resident, staging;
  std::unique_ptr<ncnn::Net> net;
  std::unique_ptr<ncnn::VkTransfer> upload;
  std::unique_ptr<ncnn::VkCompute> inference;
  std::unique_ptr<ncnn::Extractor> extractor;
  std::array<ncnn::VkBufferMemory, open_video::kMaxNcnnVulkanInputs> external{};
  std::array<ncnn::VkMat, open_video::kMaxNcnnVulkanInputs> input{};
  std::array<ncnn::VkMat, open_video::kMaxNcnnVulkanOutputs> output{},
      unpacked{};
  std::array<vk::VkCommandBuffer, 3> commands{};
  vk::VkCommandBuffer initialization = nullptr;
  std::unique_ptr<vk::VulkanBuffer> zero_staging;
  open_video::NcnnVulkanNetworkSpec graph;
  VulkanInferenceBindings bindings;
  vk::ResidentCommandTicket ticket{};
  bool prepared = false, warmed = false, queued = false, unsafe = false;
  ~Impl() {
    if (context && context->SafeToDestroyChildren()) {
      for (auto index : {0, 2})
        if (commands[index])
          context->f().vkFreeCommandBuffers(
              context->device, context->command_pool, 1, &commands[index]);
      if (initialization)
        context->f().vkFreeCommandBuffers(
            context->device, context->command_pool, 1, &initialization);
    }
  }
  static void Completed(void *opaque, const vk::ResidentCommandTicket &t,
                        vk::ResidentCommandCompletion completion) noexcept {
    auto &p = *static_cast<Impl *>(opaque);
    if (!p.queued || !(p.ticket == t))
      return;
    if (completion == vk::ResidentCommandCompletion::unsafe_pending) {
      p.unsafe = true;
      return;
    }
    p.queued = false;
    if (completion == vk::ResidentCommandCompletion::completed && !p.unsafe) {
      ++p.evidence.inference_count;
      ++p.evidence.completion_count;
    }
    // Retiring failed work never re-enables execution or fabricates successes.
  }
  bool Command(vk::VkCommandBuffer *out, std::string *error) {
    vk::VkCommandBufferAllocateInfo a;
    a.commandPool = context->command_pool;
    a.commandBufferCount = 1;
    return quota->device->CheckDriverResult(
        context->f().vkAllocateCommandBuffers(context->device, &a, out),
        "inference command allocation", true, error);
  }
  bool Begin(vk::VkCommandBuffer command, std::string *error) {
    vk::VkCommandBufferBeginInfo b;
    return quota->device->CheckDriverResult(
        context->f().vkBeginCommandBuffer(command, &b),
        "inference command begin", true, error);
  }
  bool End(vk::VkCommandBuffer command, std::string *error) {
    return quota->device->CheckDriverResult(
        context->f().vkEndCommandBuffer(command), "inference command end", true,
        error);
  }
  void Barrier(vk::VkCommandBuffer cmd, vk::VkBuffer buffer, std::size_t offset,
               std::size_t bytes, vk::VkFlags src, vk::VkFlags dst) {
    vk::VkBufferMemoryBarrier b;
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.buffer = buffer;
    b.offset = offset;
    b.size = bytes;
    context->f().vkCmdPipelineBarrier(cmd,
                                      vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                          vk::VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                          vk::VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      0, 0, nullptr, 1, &b, 0, nullptr);
  }
  bool Record(std::string *error) {
    if (!Command(&commands[0], error) || !Command(&commands[2], error) ||
        !Begin(commands[0], error))
      return false;
    for (const auto &block : resident->blocks)
      Barrier(commands[0], block->storage->buffer(), 0, block->storage->size(),
              vk::VK_ACCESS_SHADER_WRITE_BIT | vk::VK_ACCESS_TRANSFER_WRITE_BIT,
              vk::VK_ACCESS_SHADER_READ_BIT | vk::VK_ACCESS_SHADER_WRITE_BIT |
                  vk::VK_ACCESS_TRANSFER_READ_BIT |
                  vk::VK_ACCESS_TRANSFER_WRITE_BIT);
    for (std::size_t i = 0; i < bindings.input_count; ++i) {
      const auto &b = bindings.inputs[i];
      Barrier(commands[0], Handle<vk::VkBuffer>(b.resource.buffer),
              b.resource.byte_offset, b.resource.byte_size,
              vk::VK_ACCESS_SHADER_WRITE_BIT | vk::VK_ACCESS_TRANSFER_WRITE_BIT,
              vk::VK_ACCESS_SHADER_READ_BIT);
    }
    if (!End(commands[0], error) || !Begin(commands[2], error))
      return false;
    for (std::size_t i = 0; i < bindings.output_count; ++i) {
      const auto &b = bindings.outputs[i];
      const auto &o = unpacked[i];
      Barrier(commands[2], Handle<vk::VkBuffer>(o.buffer()), o.buffer_offset(),
              o.data->capacity,
              vk::VK_ACCESS_SHADER_WRITE_BIT | vk::VK_ACCESS_TRANSFER_WRITE_BIT,
              vk::VK_ACCESS_TRANSFER_READ_BIT);
      Barrier(commands[2], Handle<vk::VkBuffer>(b.resource.buffer),
              b.resource.byte_offset, b.resource.byte_size,
              vk::VK_ACCESS_SHADER_READ_BIT | vk::VK_ACCESS_SHADER_WRITE_BIT |
                  vk::VK_ACCESS_TRANSFER_WRITE_BIT,
              vk::VK_ACCESS_TRANSFER_WRITE_BIT);
      std::vector<vk::VkBufferCopy> copies;
      copies.reserve(std::size_t(b.h) * b.c);
      for (int c = 0; c < b.c; ++c)
        for (int y = 0; y < b.h; ++y) {
          vk::VkBufferCopy r;
          r.srcOffset =
              o.buffer_offset() +
              (std::size_t(c) * o.cstep + std::size_t(y) * b.w) * sizeof(float);
          r.dstOffset = b.resource.byte_offset +
                        std::size_t(c) * b.plane_stride_bytes +
                        std::size_t(y) * b.row_stride_bytes;
          r.size = std::size_t(b.w) * sizeof(float);
          copies.push_back(r);
        }
      context->f().vkCmdCopyBuffer(
          commands[2], Handle<vk::VkBuffer>(o.buffer()),
          Handle<vk::VkBuffer>(b.resource.buffer),
          static_cast<std::uint32_t>(copies.size()), copies.data());
      Barrier(commands[2], Handle<vk::VkBuffer>(b.resource.buffer),
              b.resource.byte_offset, b.resource.byte_size,
              vk::VK_ACCESS_TRANSFER_WRITE_BIT,
              vk::VK_ACCESS_SHADER_READ_BIT | vk::VK_ACCESS_SHADER_WRITE_BIT);
    }
    return End(commands[2], error);
  }
  bool InitializeInputs(std::string *error) {
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < bindings.input_count; ++i)
      bytes += bindings.inputs[i].resource.byte_size;
    zero_staging = std::make_unique<vk::VulkanBuffer>();
    if (!zero_staging->Allocate(
            quota->device, bytes, vk::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, true, error))
      return false;
    std::memset(zero_staging->mapped(), 0, bytes);
    if (!zero_staging->Flush(error) || !Command(&initialization, error) ||
        !Begin(initialization, error))
      return false;
    std::size_t offset = 0;
    for (std::size_t i = 0; i < bindings.input_count; ++i) {
      const auto &b = bindings.inputs[i];
      vk::VkBufferCopy copy;
      copy.srcOffset = offset;
      copy.dstOffset = b.resource.byte_offset;
      copy.size = b.resource.byte_size;
      offset += copy.size;
      context->f().vkCmdCopyBuffer(initialization, zero_staging->buffer(),
                                   Handle<vk::VkBuffer>(b.resource.buffer), 1,
                                   &copy);
      Barrier(initialization, Handle<vk::VkBuffer>(b.resource.buffer),
              b.resource.byte_offset, b.resource.byte_size,
              vk::VK_ACCESS_TRANSFER_WRITE_BIT, vk::VK_ACCESS_SHADER_READ_BIT);
    }
    if (!End(initialization, error) ||
        !quota->device->SubmitAndWait(initialization, error))
      return false;
    zero_staging.reset();
    return true;
  }
#endif
};

NcnnVulkanInference::NcnnVulkanInference() : impl_(std::make_unique<Impl>()) {}
NcnnVulkanInference::~NcnnVulkanInference() {
#if STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
  if (impl_ && !Shutdown()) {
    static auto *mutex = new std::mutex;
    static auto *owners = new std::vector<std::unique_ptr<Impl>>;
    std::lock_guard lock(*mutex);
    owners->push_back(std::move(impl_));
    util::MarkUnsafeNativeResourcesRetained();
  }
#endif
}

VulkanMattingRuntimeResult NcnnVulkanInference::Prepare(
    const VulkanMattingDeviceContext &device,
    const open_video::NcnnVulkanNetworkSpec &graph,
    const VulkanInferenceBindings &bindings,
    std::shared_ptr<VulkanInferenceSharedQuota> quota) {
#if STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
  if (!impl_)
    return Fail(Failure::unavailable, "Inference owner was shut down");
  auto &p = *impl_;
  if (p.context || !quota || !quota->device || !quota->device->Initialized() ||
      quota->context != quota->device->context_identity() ||
      device.ownership_domain != quota->device ||
      device.context_id != quota->context.context_id ||
      device.context_generation != quota->context.generation ||
      device.logical_device !=
          Handle<std::uintptr_t>(quota->device->device()) ||
      device.physical_device !=
          Handle<std::uintptr_t>(quota->device->physical_device()) ||
      device.queue != Handle<std::uintptr_t>(quota->device->queue()) ||
      !bindings.owner_generation || !bindings.input_count ||
      !bindings.output_count || bindings.input_count != graph.input_count ||
      bindings.output_count != graph.output_count ||
      graph.input_count > open_video::kMaxNcnnVulkanInputs ||
      graph.output_count > open_video::kMaxNcnnVulkanOutputs ||
      graph.precision != "fp32")
    return Fail(Failure::device_identity_mismatch,
                "Invalid generic inference identity/binding count");
  std::string error;
  if (!quota->device->ConfigureAllocationLimits(
          quota->allocation_budget_bytes, quota->allocation_resource_limit,
          &error))
    return Fail(Failure::allocation_contract_failed, error);
  const auto native = quota->device->NativeOwner(detail::kNcnnOwnerKind);
  if (!native)
    return Fail(Failure::device_identity_mismatch,
                "An exact ncnn-adopted context is required");
  p.owner = std::static_pointer_cast<detail::NcnnVulkanContext>(native);
  p.quota = std::move(quota);
  p.graph = graph;
  p.bindings = bindings;
  p.context = std::static_pointer_cast<vk::detail::VulkanContextState>(
      p.quota->device->RetainContext());
  auto verify = [&](const VulkanInferenceTensorBinding &b,
                    const open_video::NcnnVulkanTensorSpec &s, bool input) {
    const auto &r = b.resource;
    if (b.dims < 1 || b.dims > 3 || b.dims != s.dims || b.w <= 0 || b.h <= 0 ||
        b.c <= 0 || (b.dims == 1 && (b.h != 1 || b.c != 1)) ||
        (b.dims == 2 && b.c != 1) ||
        (!s.dynamic_hw && (b.w != s.w || b.h != s.h)) || b.c != s.c ||
        (s.dynamic_hw && (b.dims != 3 || b.w % 4 || b.h % 4)) ||
        r.ownership_domain != device.ownership_domain ||
        r.logical_device != device.logical_device ||
        r.context_id != device.context_id ||
        r.context_generation != device.context_generation || !r.buffer ||
        r.byte_size > p.quota->allocation_budget_bytes || !r.device_resident ||
        !r.device_local_memory || r.host_mapped || r.byte_offset % 4 ||
        !r.runtime_queue_ownership)
      return false;
    const std::size_t row = std::size_t(b.w) * sizeof(float);
    if (b.row_stride_bytes < row || b.row_stride_bytes % 4 ||
        std::size_t(b.h) >
            std::numeric_limits<std::size_t>::max() / b.row_stride_bytes)
      return false;
    const std::size_t plane = std::size_t(b.h) * b.row_stride_bytes;
    if (b.plane_stride_bytes < plane || b.plane_stride_bytes % 4 ||
        std::size_t(b.c) >
            std::numeric_limits<std::size_t>::max() / b.plane_stride_bytes)
      return false;
    const std::size_t extent = std::size_t(b.c - 1) * b.plane_stride_bytes +
                               std::size_t(b.h - 1) * b.row_stride_bytes + row;
    if (extent > r.byte_size ||
        r.byte_offset > std::numeric_limits<std::size_t>::max() - r.byte_size)
      return false;
    // Native foreign input planes use ncnn's aligned cstep; these reviewed
    // profiles have plane lengths divisible by four and therefore no padding.
    return !input ||
           (b.row_stride_bytes == row && b.plane_stride_bytes == plane &&
            (b.dims < 3 || (std::size_t(b.w) * b.h) % 4 == 0));
  };
  for (std::size_t i = 0; i < bindings.input_count; ++i)
    if (!verify(bindings.inputs[i], graph.inputs[i], true))
      return Fail(Failure::invalid_shape,
                  "Invalid inference input rank/stride/domain");
  for (std::size_t i = 0; i < bindings.output_count; ++i) {
    if (!verify(bindings.outputs[i], graph.outputs[i], false))
      return Fail(Failure::invalid_shape,
                  "Invalid inference output rank/stride/domain");
    for (std::size_t j = 0; j < bindings.input_count; ++j)
      if (bindings.outputs[i].resource.buffer ==
          bindings.inputs[j].resource.buffer)
        return Fail(Failure::invalid_shape,
                    "Inference output aliases an input");
    for (std::size_t j = 0; j < i; ++j)
      if (bindings.outputs[i].resource.buffer ==
          bindings.outputs[j].resource.buffer)
        return Fail(Failure::invalid_shape, "Inference outputs alias");
  }
  p.budget.cap = p.quota->allocation_budget_bytes;
  p.budget.resource_cap = p.quota->allocation_resource_limit;
  p.resident = std::make_unique<ResidentAllocator>(
      p.quota->device, p.owner->device.get(), &p.budget, false);
  p.staging = std::make_unique<ResidentAllocator>(
      p.quota->device, p.owner->device.get(), &p.budget, true);
  auto param = open_video::OpenVerifiedModelFileSnapshot(
      graph.param_path, graph.param_sha256, 1024u * 1024u, &error);
  auto bin = open_video::OpenVerifiedModelFileSnapshot(
      graph.bin_path, graph.bin_sha256, p.budget.cap, &error);
  if (!param || !bin)
    return Fail(Failure::invalid_graph, error);
  p.net = std::make_unique<ncnn::Net>();
  p.net->set_vulkan_device(p.owner->device.get());
  auto &opt = p.net->opt;
  opt.use_vulkan_compute = true;
  opt.lightmode = true;
  opt.use_packing_layout = true;
  opt.use_fp16_packed = opt.use_fp16_storage = opt.use_fp16_arithmetic = false;
  opt.use_bf16_packed = opt.use_bf16_storage = false;
  opt.use_int8_packed = opt.use_int8_storage = opt.use_int8_arithmetic = false;
  opt.use_tensor_storage = opt.use_weights_in_host_memory =
      opt.use_local_pool_allocator = opt.use_mapped_model_loading = false;
  opt.blob_vkallocator = opt.workspace_vkallocator = p.resident.get();
  opt.staging_vkallocator = p.staging.get();
  auto gpu_only = [&]() {
    for (const auto *layer : p.net->layers())
      if (!layer || layer->typeindex == ncnn::LayerType::MultiHeadAttention ||
          (layer->typeindex != ncnn::LayerType::Input &&
           (!layer->support_vulkan || (layer->featmask & (1 << 4)))))
        return false;
    return true;
  };
  if (p.net->load_param(param.get()))
    return Fail(Failure::invalid_graph, "Generic graph parameter load failed");
  if (!gpu_only())
    return Fail(Failure::cpu_layer_rejected,
                "Generic graph has a non-Vulkan layer");
  p.upload = std::make_unique<ncnn::VkTransfer>(p.owner->device.get(), true);
  if (!p.upload->studiocast_valid())
    return Fail(Failure::out_of_memory, "Model upload command unavailable");
  opt.studiocast_model_upload = p.upload.get();
  if (p.net->load_model(bin.get()) || p.budget.rejected)
    return Fail(
        Failure::allocation_contract_failed,
        "Model load exceeded aggregate reservation or graph is invalid");
  VkCommandBuffer upload = VK_NULL_HANDLE;
  if (p.upload->studiocast_finalize(&upload) ||
      !p.quota->device->SubmitAndWait(Handle<vk::VkCommandBuffer>(upload),
                                      &error))
    return Fail(Failure::synchronization_failed, error);
  if (p.upload->reset())
    return Fail(Failure::synchronization_failed,
                "Completed upload reset failed");
  p.staging->ReleaseFreeBlocks();
  if (!gpu_only())
    return Fail(Failure::cpu_layer_rejected,
                "Graph disabled Vulkan during model load");
  p.inference = std::make_unique<ncnn::VkCompute>(p.owner->device.get(), true);
  if (!p.inference->studiocast_valid())
    return Fail(Failure::out_of_memory, "Graph recording unavailable");
  p.extractor = std::make_unique<ncnn::Extractor>(p.net->create_extractor());
  for (std::size_t i = 0; i < bindings.input_count; ++i) {
    const auto &b = bindings.inputs[i];
    auto &m = p.external[i];
    m.buffer = Handle<VkBuffer>(b.resource.buffer);
    m.offset = b.resource.byte_offset;
    m.capacity = b.resource.byte_size;
    m.stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    m.access_flags = VK_ACCESS_SHADER_WRITE_BIT;
    if (b.dims == 1)
      p.input[i] = ncnn::VkMat(b.w, &m, sizeof(float), p.resident.get());
    else if (b.dims == 2)
      p.input[i] = ncnn::VkMat(b.w, b.h, &m, sizeof(float), p.resident.get());
    else
      p.input[i] =
          ncnn::VkMat(b.w, b.h, b.c, &m, sizeof(float), p.resident.get());
    m.refcount = 1;
    p.input[i].refcount = &m.refcount;
    if (p.extractor->input(graph.inputs[i].blob.c_str(), p.input[i]))
      return Fail(Failure::invalid_graph, "Graph input blob is missing");
  }
  for (std::size_t i = 0; i < bindings.output_count; ++i) {
    if (p.extractor->extract(graph.outputs[i].blob.c_str(), p.output[i],
                             *p.inference))
      return Fail(Failure::invalid_graph, "Graph extraction failed");
    p.owner->device->convert_packing(p.output[i], p.unpacked[i], 1,
                                     *p.inference, opt);
    const auto &o = p.unpacked[i];
    const auto &b = bindings.outputs[i];
    if (p.budget.rejected)
      return Fail(Failure::allocation_contract_failed,
                  "Graph activation reservation exhausted");
    if (!o.data || o.dims != int(b.dims) || o.w != b.w || o.h != b.h ||
        o.c != b.c || o.elempack != 1 || o.elemsize != sizeof(float) ||
        o.data->mapped_ptr ||
        !p.owner->device->is_device_local(o.data->memory_type_index))
      return Fail(Failure::residency_check_failed,
                  "Graph output shape/packing/residency mismatch");
  }
  VkCommandBuffer inference = VK_NULL_HANDLE;
  if (p.inference->studiocast_finalize(&inference))
    return Fail(Failure::cpu_readback_rejected,
                "Graph recording contains a host bridge or failed");
  p.commands[1] = Handle<vk::VkCommandBuffer>(inference);
  if (!p.Record(&error) || !p.InitializeInputs(&error))
    return Fail(Failure::synchronization_failed, error);
  p.budget.frozen = true;
  p.prepared = true;
  auto &e = p.evidence;
  e.runtime_name = "ncnn-vk-multitensor-replay";
  e.adapter_available = e.production_adapter = e.runtime_created =
      e.graph_loaded = true;
  e.persistent_resources_allocated = e.device_identity_matches =
      e.input_device_resident = true;
  e.alpha_device_resident = e.output_device_resident =
      e.shared_device_imported = true;
  e.queue_ownership_explicit = e.synchronous_completion =
      e.bounded_reusable_allocations = true;
  e.persistent_allocation_count = p.budget.resources;
  e.active_device = device;
  return Result::Success();
#else
  (void)device;
  (void)graph;
  (void)bindings;
  (void)quota;
  return Fail(Failure::unavailable,
              "Production ncnn Vulkan inference disabled");
#endif
}
VulkanMattingRuntimeResult NcnnVulkanInference::Warmup() {
#if STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
  if (!impl_)
    return Fail(Failure::unavailable, "Inference owner was shut down");
  auto &p = *impl_;
  if (!p.prepared || p.warmed || p.queued || p.unsafe)
    return Fail(Failure::invalid_graph,
                "Warmup requires one fresh prepared graph");
  std::string error;
  if (!p.quota->device->SubmitAndWait(p.commands.data(), 3, &error)) {
    p.unsafe = true;
    return Fail(Failure::synchronization_failed, error);
  }
  p.warmed = p.evidence.warmup_complete = true;
  ++p.evidence.warmup_inference_count;
  ++p.evidence.completion_count;
  return Result::Success();
#else
  return Fail(Failure::unavailable,
              "Production ncnn Vulkan inference disabled");
#endif
}
VulkanMattingRuntimeResult
NcnnVulkanInference::Enqueue(vk::ResidentCommandBatch &batch) {
#if STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
  if (!impl_)
    return Fail(Failure::unavailable, "Inference owner was shut down");
  auto &p = *impl_;
  if (!p.prepared || !p.warmed || p.queued || p.unsafe ||
      batch.context_identity() != p.quota->context || !p.context->Healthy())
    return Fail(Failure::context_generation_mismatch,
                "Stale/pending inference batch");
  vk::ResidentCommandGroup group;
  group.context = p.quota->context;
  group.commands = p.commands.data();
  group.count = 3;
  group.owner = &p;
  group.owner_generation = p.bindings.owner_generation;
  group.on_completion = Impl::Completed;
  std::string error;
  if (!batch.Append(group, &p.ticket, &error))
    return Fail(Failure::execution_failed, error);
  p.queued = true;
  return Result::Success();
#else
  (void)batch;
  return Fail(Failure::unavailable,
              "Production ncnn Vulkan inference disabled");
#endif
}
VulkanMattingRuntimeEvidence NcnnVulkanInference::Evidence() const {
  if (!impl_)
    return {};
  auto result = impl_->evidence;
#if STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
  if (impl_->context)
    result.active_device.context_healthy = impl_->context->Healthy();
#endif
  return result;
}
void NcnnVulkanInference::RefreshExecutionEvidence(
    VulkanMattingRuntimeEvidence *out) const noexcept {
  if (!out || !impl_)
    return;
  out->inference_count = impl_->evidence.inference_count;
  out->completion_count = impl_->evidence.completion_count;
  out->warmup_inference_count = impl_->evidence.warmup_inference_count;
  out->warmup_complete = impl_->evidence.warmup_complete;
#if STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
  out->active_device.context_healthy =
      impl_->context && impl_->context->Healthy();
#endif
}
bool NcnnVulkanInference::Shutdown() noexcept {
  if (!impl_)
    return true;
#if STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
  if (impl_->queued ||
      (impl_->context && !impl_->context->SafeToDestroyChildren()))
    return false;
  impl_.reset();
  // No reprepare after shutdown: the owner creates a new concrete graph.
#endif
  return true;
}
} // namespace studiocast::open_vulkan
