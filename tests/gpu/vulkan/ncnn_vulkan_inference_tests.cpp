#include "core/open_video/ncnn_vulkan_context.h"
#include "core/open_video/ncnn_vulkan_inference.h"
#include "core/util/json.h"
#include "core/vulkan/resident_command_batch.h"
#include "core/vulkan/vulkan_tensor.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gpu.h>
#include <iostream>
#include <memory>
#include <new>
#include <vector>

namespace {
std::atomic<bool> count_allocations{false};
std::atomic<std::size_t> allocation_count{0};
} // namespace
void *operator new(std::size_t size) {
  if (count_allocations.load(std::memory_order_relaxed))
    ++allocation_count;
  if (auto *p = std::malloc(size ? size : 1))
    return p;
  throw std::bad_alloc();
}
void *operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

namespace {
namespace vk = studiocast::vulkan;
namespace ov = studiocast::open_vulkan;
namespace model = studiocast::open_video;
namespace json = studiocast::util::json;
namespace fs = std::filesystem;
template <class T> std::uintptr_t Opaque(T p) {
  return reinterpret_cast<std::uintptr_t>(p);
}
bool Check(bool yes, const std::string &message) {
  if (!yes)
    std::cerr << message << '\n';
  return yes;
}
std::string Read(const fs::path &p) {
  std::ifstream f(p);
  return {std::istreambuf_iterator<char>(f), {}};
}
json::Value Json(const fs::path &p) {
  json::Value value;
  std::string error;
  if (!json::Parse(Read(p), &value, &error))
    throw std::runtime_error(error);
  return value;
}
const json::Value &At(const json::Value &v, const char *key) {
  return v.AsObject()->at(key);
}
int Number(const json::Value &v) { return static_cast<int>(*v.AsNumber()); }
std::vector<float> Floats(const fs::path &p) {
  const auto text = Read(p);
  if (text.size() % 4)
    throw std::runtime_error("invalid f32 file");
  std::vector<float> v(text.size() / 4);
  std::memcpy(v.data(), text.data(), text.size());
  return v;
}

PFN_vkCreateDevice saved_create = nullptr;
VkPhysicalDevice captured_physical = VK_NULL_HANDLE;
VkDevice captured_logical = VK_NULL_HANDLE;
bool captured_fp64 = false;
VKAPI_ATTR VkResult VKAPI_CALL CaptureCreate(VkPhysicalDevice physical,
                                             const VkDeviceCreateInfo *info,
                                             const VkAllocationCallbacks *alloc,
                                             VkDevice *out) {
  const auto result = saved_create(physical, info, alloc, out);
  if (result == VK_SUCCESS) {
    captured_physical = physical;
    captured_logical = *out;
    captured_fp64 =
        info->pEnabledFeatures && info->pEnabledFeatures->shaderFloat64;
  }
  return result;
}

ov::VulkanMattingDeviceContext Context(vk::VulkanDevice &d) {
  ov::VulkanMattingDeviceContext c;
  const auto &diag = d.diagnostics();
  c.ownership_domain = &d;
  c.physical_device = Opaque(d.physical_device());
  c.logical_device = Opaque(d.device());
  c.queue = Opaque(d.queue());
  c.queue_family_index = d.queue_family_index();
  c.context_id = d.context_identity().context_id;
  c.context_generation = d.context_identity().generation;
  c.non_cpu_device_selected = diag.non_cpu_device_selected;
  c.compute_queue_available = c.context_healthy = true;
  c.stable_device_id = diag.selected_device_stable_id;
  return c;
}

struct Copy {
  vk::VulkanDevice *device = nullptr;
  vk::VkCommandBuffer command = nullptr;
  vk::VulkanBuffer staging;
  std::size_t bytes = 0;
  ~Copy() {
    if (command && device->SafeToDestroyResources())
      device->f().vkFreeCommandBuffers(device->device(), device->command_pool(),
                                       1, &command);
  }
  bool Prepare(vk::VulkanDevice &d, const vk::VulkanTensor &tensor, bool upload,
               std::string *error) {
    device = &d;
    bytes = tensor.bytes();
    if (!staging.Allocate(&d, bytes,
                          vk::VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                              vk::VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                          vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, true, error))
      return false;
    vk::VkCommandBufferAllocateInfo a;
    a.commandPool = d.command_pool();
    a.commandBufferCount = 1;
    vk::VkCommandBufferBeginInfo b;
    if (!d.CheckDriverResult(
            d.f().vkAllocateCommandBuffers(d.device(), &a, &command),
            "test copy allocation", true, error) ||
        !d.CheckDriverResult(d.f().vkBeginCommandBuffer(command, &b),
                             "test copy begin", true, error))
      return false;
    vk::VkBufferMemoryBarrier barrier;
    barrier.buffer = upload ? staging.buffer() : tensor.buffer();
    barrier.size = bytes;
    barrier.srcAccessMask = vk::VK_ACCESS_SHADER_WRITE_BIT |
                            vk::VK_ACCESS_TRANSFER_WRITE_BIT |
                            vk::VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = vk::VK_ACCESS_TRANSFER_READ_BIT;
    d.f().vkCmdPipelineBarrier(command,
                               vk::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT |
                                   vk::VK_PIPELINE_STAGE_HOST_BIT,
                               vk::VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                               nullptr, 1, &barrier, 0, nullptr);
    vk::VkBufferCopy copy;
    copy.size = bytes;
    d.f().vkCmdCopyBuffer(command, upload ? staging.buffer() : tensor.buffer(),
                          upload ? tensor.buffer() : staging.buffer(), 1,
                          &copy);
    barrier.buffer = upload ? tensor.buffer() : staging.buffer();
    barrier.srcAccessMask = vk::VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask =
        upload ? vk::VK_ACCESS_SHADER_READ_BIT : vk::VK_ACCESS_HOST_READ_BIT;
    d.f().vkCmdPipelineBarrier(command, vk::VK_PIPELINE_STAGE_TRANSFER_BIT,
                               vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                   vk::VK_PIPELINE_STAGE_HOST_BIT,
                               0, 0, nullptr, 1, &barrier, 0, nullptr);
    return d.CheckDriverResult(d.f().vkEndCommandBuffer(command),
                               "test copy end", true, error);
  }
  bool Upload(const std::vector<float> &v, std::string *error) {
    if (v.size() * 4 != bytes)
      return false;
    std::memcpy(staging.mapped(), v.data(), bytes);
    return staging.Flush(error) && device->SubmitAndWait(command, error);
  }
  bool Download(std::vector<float> *v, std::string *error) {
    if (!device->SubmitAndWait(command, error) || !staging.Invalidate(error))
      return false;
    v->resize(bytes / 4);
    std::memcpy(v->data(), staging.mapped(), bytes);
    return true;
  }
};

ov::VulkanInferenceTensorBinding
Binding(vk::VulkanDevice &d, const vk::VulkanTensor &t, unsigned dims) {
  ov::VulkanInferenceTensorBinding b;
  b.dims = dims;
  b.w = t.w();
  b.h = t.h();
  b.c = t.c();
  b.row_stride_bytes = std::size_t(b.w) * 4;
  b.plane_stride_bytes = b.row_stride_bytes * b.h;
  auto &r = b.resource;
  r.ownership_domain = &d;
  r.logical_device = Opaque(d.device());
  r.buffer = Opaque(t.buffer());
  r.byte_size = t.bytes();
  r.context_id = d.context_identity().context_id;
  r.context_generation = d.context_identity().generation;
  r.device_resident = r.device_local_memory = r.runtime_queue_ownership = true;
  return b;
}
bool Compare(const std::vector<float> &actual,
             const std::vector<float> &expected, const std::string &name,
             double *maximum, double *mae) {
  if (actual.size() != expected.size() || actual.empty())
    return false;
  double max = 0, sum = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
      return false;
    const double e = std::abs(double(actual[i]) - expected[i]);
    max = std::max(max, e);
    sum += e;
  }
  *maximum = std::max(*maximum, max);
  *mae = std::max(*mae, sum / actual.size());
  return Check(max <= 2e-4 && sum / actual.size() <= 2e-5,
               name + " max=" + std::to_string(max) +
                   " mae=" + std::to_string(sum / actual.size()));
}

bool RunModel(vk::VulkanDevice &device, const fs::path &root,
              const std::string &name, const model::ModelPack &pack,
              const model::NcnnVulkanNetworkSpec &spec) {
  std::string error;
  if (!model::ValidateProductionNcnnVulkanNetworkPack(pack, spec, &error))
    return Check(false, error);
  const auto numerical = root / "work" / name / "numerical";
  const auto fixture = Json(numerical / "fixtures.json");
  std::unique_ptr<ov::NcnnVulkanInference> graph;
  std::array<std::unique_ptr<vk::VulkanTensor>, 3> inputs;
  std::array<std::unique_ptr<vk::VulkanTensor>, 12> outputs;
  std::array<std::unique_ptr<Copy>, 3> uploads;
  std::array<std::unique_ptr<Copy>, 12> downloads;
  auto quota = std::make_shared<ov::VulkanInferenceSharedQuota>();
  quota->device = &device;
  quota->context = device.context_identity();
  quota->allocation_budget_bytes = 1024u * 1024u * 1024u;
  quota->allocation_resource_limit = 4096;
  vk::ResidentCommandBatch batch;
  if (!batch.Initialize(&device, 128, &error))
    return Check(false, error);
  int width = 0, height = 0;
  std::uint64_t sequence = 0;
  double maximum = 0, mae = 0;
  std::size_t max_alloc = 0;
  std::array<float, 64> grid_x{};
  std::array<float, 48> grid_y{};
  if (name == "left" || name == "right") {
    const auto axes = Json(root / "flx-meshgrid-exact-axis-bits.json");
    for (std::size_t i = 0; i < 64; ++i)
      grid_x[i] = std::bit_cast<float>(static_cast<std::uint32_t>(
          *(*At(axes, "x_u32").AsArray())[i].AsNumber()));
    for (std::size_t i = 0; i < 48; ++i)
      grid_y[i] = std::bit_cast<float>(static_cast<std::uint32_t>(
          *(*At(axes, "y_u32").AsArray())[i].AsNumber()));
  }
  for (const auto &row : *At(fixture, "cases").AsArray()) {
    const auto label = *At(row, "case").AsString();
    const auto &shapes = *At(row, "input_shapes").AsArray();
    const auto &first = *shapes[0].AsArray();
    const int w = Number(first.back()),
              h = first.size() == 4 ? Number(first[2]) : 1;
    if (w != width || h != height) {
      if (graph && !graph->Shutdown())
        return false;
      graph.reset();
      for (auto &p : uploads)
        p.reset();
      for (auto &p : downloads)
        p.reset();
      for (auto &p : inputs)
        p.reset();
      for (auto &p : outputs)
        p.reset();
      width = w;
      height = h;
      ov::VulkanInferenceBindings bindings;
      bindings.input_count = spec.input_count;
      bindings.output_count = spec.output_count;
      bindings.owner_generation = ++sequence;
      for (std::size_t i = 0; i < spec.input_count; ++i) {
        const auto &shape = *shapes[i].AsArray();
        const int iw = Number(shape.back()),
                  ih = shape.size() == 4 ? Number(shape[2]) : 1,
                  ic = shape.size() == 4 ? Number(shape[1]) : 1;
        inputs[i] = std::make_unique<vk::VulkanTensor>();
        if (!inputs[i]->AllocateNchwF32(&device, 1, ic, ih, iw, false, &error))
          return Check(false, error);
        bindings.inputs[i] = Binding(device, *inputs[i], spec.inputs[i].dims);
        uploads[i] = std::make_unique<Copy>();
        if (!uploads[i]->Prepare(device, *inputs[i], true, &error))
          return Check(false, error);
      }
      for (std::size_t i = 0; i < spec.output_count; ++i) {
        const auto &s = spec.outputs[i];
        const int ow = s.dynamic_hw ? w : s.w, oh = s.dynamic_hw ? h : s.h;
        outputs[i] = std::make_unique<vk::VulkanTensor>();
        if (!outputs[i]->AllocateNchwF32(&device, 1, s.c, oh, ow, false,
                                         &error))
          return Check(false, error);
        bindings.outputs[i] = Binding(device, *outputs[i], s.dims);
        downloads[i] = std::make_unique<Copy>();
        if (!downloads[i]->Prepare(device, *outputs[i], false, &error))
          return Check(false, error);
      }
      graph = std::make_unique<ov::NcnnVulkanInference>();
      auto prepared = graph->Prepare(Context(device), spec, bindings, quota);
      if (!prepared.ok())
        return Check(false, name + " prepare: " + prepared.detail);
      const auto warmed = graph->Warmup();
      if (!warmed.ok())
        return Check(false, name + " warmup: " + warmed.detail);
      // Abort produces no inference/completion evidence and permits fresh
      // reuse.
      auto before = graph->Evidence();
      if (!batch.BeginFrame(++sequence, &error) || !graph->Enqueue(batch).ok())
        return false;
      batch.AbortUnsubmitted();
      auto after = graph->Evidence();
      if (after.inference_count != before.inference_count ||
          after.completion_count != before.completion_count)
        return false;
    }
    std::array<std::vector<float>, 3> host_inputs;
    for (std::size_t i = 0; i < spec.input_count; ++i) {
      host_inputs[i] =
          Floats(numerical / (label + ".input" + std::to_string(i) + ".f32"));
      if (!uploads[i]->Upload(host_inputs[i], &error))
        return Check(false, error);
    }
    if (!batch.BeginFrame(++sequence, &error))
      return false;
    const auto resources_before = device.allocation_stats();
    allocation_count = 0;
    count_allocations = true;
    const auto queued = graph->Enqueue(batch);
    const bool complete = queued.ok() && batch.Complete(&error);
    count_allocations = false;
    max_alloc = std::max(max_alloc, allocation_count.load());
    if (!complete)
      return Check(false, name + " execution: " + queued.detail + error);
    const auto resources_after = device.allocation_stats();
    if (!Check(allocation_count == 0 &&
                   resources_after.allocation_count ==
                       resources_before.allocation_count &&
                   resources_after.current_bytes ==
                       resources_before.current_bytes,
               "Execute allocated resources"))
      return false;
    std::array<std::vector<float>, 12> host_outputs;
    for (std::size_t i = 0; i < spec.output_count; ++i)
      if (!downloads[i]->Download(&host_outputs[i], &error))
        return Check(false, error);
    if (name == "left" || name == "right") {
      std::vector<float> rgb(3 * 48 * 64);
      for (int y = 0; y < 48; ++y)
        for (int x = 0; x < 64; ++x) {
          const int p = y * 64 + x;
          const float px = (host_outputs[0][p] + grid_x[x] + 1.f) * 32.f,
                      py = (host_outputs[0][3072 + p] + grid_y[y] + 1.f) * 24.f;
          const int fx = int(std::floor(px)), fy = int(std::floor(py));
          const int x0 = std::clamp(fx, 0, 63), x1 = std::clamp(fx + 1, 0, 63),
                    y0 = std::clamp(fy, 0, 47), y1 = std::clamp(fy + 1, 0, 47);
          for (int c = 0; c < 3; ++c) {
            const float *im = host_inputs[0].data() + c * 3072;
            float value = (x1 - px) * (y1 - py) * im[y0 * 64 + x0];
            value += (x1 - px) * (py - y0) * im[y1 * 64 + x0];
            value += (px - x0) * (y1 - py) * im[y0 * 64 + x1];
            value += (px - x0) * (py - y0) * im[y1 * 64 + x1];
            rgb[c * 3072 + p] =
                value * host_outputs[1][p] + host_outputs[1][3072 + p];
          }
        }
      if (!Compare(rgb, Floats(numerical / (label + ".rgb.expected.f32")),
                   name + "/" + label, &maximum, &mae))
        return false;
    } else
      for (std::size_t i = 0; i < spec.output_count; ++i)
        if (!Compare(host_outputs[i],
                     Floats(numerical /
                            (label + ".expected" + std::to_string(i) + ".f32")),
                     name + "/" + label + "/" + spec.outputs[i].blob, &maximum,
                     &mae))
          return false;
  }
  if (graph && !graph->Shutdown())
    return false;
  const auto stats = device.allocation_stats();
  std::cout << name << " max=" << maximum << " mae=" << mae
            << " host_execute_allocs=" << max_alloc
            << " context_peak_bytes=" << stats.high_water_bytes
            << " context_peak_resources=" << stats.high_water_allocation_count
            << '\n';
  return true;
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "SKIP: pass offline candidate root\n";
    return 77;
  }
  if (ncnn::studiocast_acquire_gpu_instance())
    return 1;
  saved_create = ncnn::vkCreateDevice;
  ncnn::vkCreateDevice = CaptureCreate;
  vk::VulkanDevice device;
  std::string error;
  const bool initialized =
      ov::InitializeProductionVulkanMattingDevice(&device, &error);
  ncnn::vkCreateDevice = saved_create;
  if (!initialized) {
    std::cerr << error << '\n';
    ncnn::studiocast_release_gpu_instance();
    return 1;
  }
  if (!Check(Opaque(captured_physical) == Opaque(device.physical_device()) &&
                 Opaque(captured_logical) == Opaque(device.device()) &&
                 captured_fp64 == device.shader_float64_enabled(),
             "FP64 attestation differs from actual VkDeviceCreateInfo"))
    return 1;
  std::cout << "captured_shader_float64_enabled=" << captured_fp64 << '\n';
  const fs::path root = argv[1];
  const auto registry =
      model::ModelPackRegistry::Scan(root / "source/open_video");
  const std::array<std::pair<const char *, const char *>, 4> models{
      {{"fastdvdnet", "fastdvdnet_sigma15"},
       {"yunet", "yunet_opencv_zoo_2023mar_fp32"},
       {"left", "gaze_correction_cam_flx_v0_1_1"},
       {"right", "gaze_correction_cam_flx_v0_1_1"}}};
  bool ok = true;
  for (const auto &[name, id] : models) {
    const auto pack = registry.ResolveModel(id);
    if (!pack) {
      std::cerr << "missing pack " << id << '\n';
      ok = false;
      break;
    }
    const std::string role =
        (std::string(name) == "left" || std::string(name) == "right") ? name
                                                                      : "main";
    const auto it = std::find_if(pack->ncnn_vulkan_networks.begin(),
                                 pack->ncnn_vulkan_networks.end(),
                                 [&](const auto &s) { return s.role == role; });
    if (it == pack->ncnn_vulkan_networks.end() ||
        !RunModel(device, root, name, *pack, *it)) {
      ok = false;
      break;
    }
  }
  device.Shutdown();
  ncnn::studiocast_release_gpu_instance();
  return ok ? 0 : 1;
}
