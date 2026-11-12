#include "core/open_video/ncnn_vulkan_context.h"
#include "core/open_video/vulkan_matting_runtime.h"
#include "core/vulkan/vulkan_image.h"
#include "core/vulkan/vulkan_tensor.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <vector>
namespace {
namespace ov = studiocast::open_vulkan;
namespace vk = studiocast::vulkan;
bool Check(bool value, const std::string &error) {
  if (!value)
    std::cerr << error << '\n';
  return value;
}
template <class T> std::uintptr_t Opaque(T value) {
  return reinterpret_cast<std::uintptr_t>(value);
}
ov::VulkanMattingDeviceContext Context(vk::VulkanDevice &device) {
  ov::VulkanMattingDeviceContext out;
  const auto &diag = device.diagnostics();
  out.ownership_domain = &device;
  out.physical_device = Opaque(device.physical_device());
  out.logical_device = Opaque(device.device());
  out.queue = Opaque(device.queue());
  out.queue_family_index = device.queue_family_index();
  out.vendor_id = diag.vendor_id;
  out.device_id = diag.device_id;
  out.selected_device_index = diag.selected_device_index;
  out.stable_device_id = diag.selected_device_stable_id;
  out.context_id = device.context_identity().context_id;
  out.context_generation = device.context_identity().generation;
  out.non_cpu_device_selected = diag.non_cpu_device_selected;
  out.compute_queue_available = out.context_healthy = true;
  return out;
}
bool Transfer(vk::VulkanDevice &device, vk::VkBuffer source, vk::VkBuffer dest,
              std::size_t bytes, std::string *error) {
  vk::VkCommandBufferAllocateInfo alloc;
  alloc.commandPool = device.command_pool();
  alloc.commandBufferCount = 1;
  vk::VkCommandBuffer command = nullptr;
  if (!device.CheckDriverResult(device.f().vkAllocateCommandBuffers(
                                    device.device(), &alloc, &command),
                                "test allocation", true, error))
    return false;
  vk::VkCommandBufferBeginInfo begin;
  if (!device.CheckDriverResult(
          device.f().vkBeginCommandBuffer(command, &begin), "test begin", true,
          error))
    return false;
  vk::VkBufferMemoryBarrier source_barrier;
  source_barrier.buffer = source;
  source_barrier.size = bytes;
  source_barrier.srcAccessMask = vk::VK_ACCESS_SHADER_WRITE_BIT |
                                 vk::VK_ACCESS_TRANSFER_WRITE_BIT |
                                 vk::VK_ACCESS_HOST_WRITE_BIT;
  source_barrier.dstAccessMask = vk::VK_ACCESS_TRANSFER_READ_BIT;
  device.f().vkCmdPipelineBarrier(command,
                                  vk::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT |
                                      vk::VK_PIPELINE_STAGE_HOST_BIT,
                                  vk::VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                  nullptr, 1, &source_barrier, 0, nullptr);
  vk::VkBufferCopy copy;
  copy.size = bytes;
  device.f().vkCmdCopyBuffer(command, source, dest, 1, &copy);
  vk::VkBufferMemoryBarrier destination_barrier;
  destination_barrier.buffer = dest;
  destination_barrier.size = bytes;
  destination_barrier.srcAccessMask = vk::VK_ACCESS_TRANSFER_WRITE_BIT;
  destination_barrier.dstAccessMask =
      vk::VK_ACCESS_SHADER_READ_BIT | vk::VK_ACCESS_HOST_READ_BIT;
  device.f().vkCmdPipelineBarrier(
      command, vk::VK_PIPELINE_STAGE_TRANSFER_BIT,
      vk::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | vk::VK_PIPELINE_STAGE_HOST_BIT,
      0, 0, nullptr, 1, &destination_barrier, 0, nullptr);
  if (!device.CheckDriverResult(device.f().vkEndCommandBuffer(command),
                                "test end", true, error))
    return false;
  const bool ok = device.SubmitAndWait(command, error);
  if (device.SafeToDestroyResources())
    device.f().vkFreeCommandBuffers(device.device(), device.command_pool(), 1,
                                    &command);
  return ok;
}
bool RealModelParity(vk::VulkanDevice &device, std::string *error) {
  const char *param = std::getenv("STUDIOCAST_NCNN_MATTING_PARAM");
  const char *bin = std::getenv("STUDIOCAST_NCNN_MATTING_BIN");
  const char *golden = std::getenv("STUDIOCAST_NCNN_MATTING_GOLDEN_ROOT");
  if (!param && !bin && !golden)
    return true;
  if (!param || !bin || !golden)
    return Check(
        false,
        "Real-model parity requires param, bin and golden root together.");
  constexpr int width = 256, height = 256;
  constexpr std::size_t elements = width * height;
  vk::VulkanTensor input;
  vk::VulkanImage alpha;
  vk::VulkanBuffer upload, readback;
  if (!input.AllocateNchwF32(&device, 1, 3, height, width, false, error) ||
      !alpha.Allocate(&device, width, height, vk::VulkanPixelFormat::f32_1,
                      false, error) ||
      !upload.Allocate(&device, input.bytes(),
                       vk::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                       vk::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true, error) ||
      !readback.Allocate(&device, alpha.byte_size(),
                         vk::VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                         vk::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true, error))
    return false;
  ov::VulkanMattingGraphDescriptor graph;
  graph.param_path = param;
  graph.bin_path = bin;
  graph.param_sha256 = std::string(64, 'a');
  graph.bin_sha256 = std::string(64, 'b');
  graph.input_blob = "input";
  graph.output_blob = "output";
  graph.converter_name = "onnx2ncnn";
  graph.converter_version = "24ea07e";
  graph.precision = "fp32";
  graph.input_h = graph.output_h = height;
  graph.input_w = graph.output_w = width;
  ov::VulkanMattingPersistentResources resources;
  resources.input_bytes = input.bytes();
  resources.alpha_bytes = elements * sizeof(float);
  resources.input_buffer = Opaque(input.buffer());
  resources.alpha_buffer = Opaque(alpha.buffer());
  resources.alpha_row_stride_bytes = alpha.pitch_bytes();
  resources.alpha_buffer_capacity_bytes = alpha.byte_size();
  ov::VulkanMattingRuntimeLifecycle runtime(
      ov::CreateProductionVulkanMattingRuntime());
  if (!runtime.Prepare(Context(device), graph, resources, error) ||
      !runtime.Warmup(error))
    return false;
  ov::VulkanMattingBufferBinding in, out;
  auto bind = [&](auto &b, std::uintptr_t buffer, std::size_t bytes, int c) {
    b.ownership_domain = &device;
    b.logical_device = Opaque(device.device());
    b.buffer = buffer;
    b.byte_size = bytes;
    b.n = 1;
    b.c = c;
    b.h = height;
    b.w = width;
    b.device_resident = b.device_local_memory = b.access_synchronized =
        b.runtime_queue_ownership = true;
    b.context_id = device.context_identity().context_id;
    b.context_generation = device.context_identity().generation;
  };
  bind(in, resources.input_buffer, resources.input_bytes, 3);
  bind(out, resources.alpha_buffer, resources.alpha_bytes, 1);
  auto read = [](const std::filesystem::path &path, void *bytes,
                 std::size_t count) {
    std::ifstream file(path, std::ios::binary);
    file.read(static_cast<char *>(bytes), static_cast<std::streamsize>(count));
    return bool(file) && file.peek() == std::char_traits<char>::eof();
  };
  bool all_ok = true;
  std::vector<float> expected(elements);
  const auto before = device.allocation_stats();
  for (const char *name :
       {"black", "white", "gray", "gradient", "checker", "seeded_noise"}) {
    const auto root = std::filesystem::path(golden);
    if (!read(root / (std::string(name) + ".input.f32"), upload.mapped(),
              input.bytes()) ||
        !read(root / (std::string(name) + ".expected.f32"), expected.data(),
              expected.size() * sizeof(float)))
      return Check(false, "Missing or wrong-size real-model fixture.");
    if (!upload.Flush(error) ||
        !Transfer(device, upload.buffer(), input.buffer(), input.bytes(),
                  error) ||
        !runtime.Run(in, out, error) ||
        !Transfer(device, alpha.buffer(), readback.buffer(), alpha.byte_size(),
                  error) ||
        !readback.Invalidate(error))
      return false;
    double total = 0, maximum = 0;
    std::uint64_t intersection = 0, union_count = 0;
    bool finite = true;
    for (int y = 0; y < height; ++y) {
      const auto *row = reinterpret_cast<const float *>(
          static_cast<const char *>(readback.mapped()) +
          std::size_t(y) * alpha.pitch_bytes());
      for (int x = 0; x < width; ++x) {
        const float actual = row[x],
                    reference = expected[std::size_t(y) * width + x];
        finite = finite && std::isfinite(actual) && std::isfinite(reference);
        const double difference = std::fabs(double(actual) - reference);
        total += difference;
        maximum = std::max(maximum, difference);
        const bool a = actual >= 0.5f, b = reference >= 0.5f;
        intersection += a && b;
        union_count += a || b;
      }
    }
    const double mae = total / elements,
                 iou = union_count ? double(intersection) / union_count : 1.0;
    const bool ok = finite && maximum <= 2e-4 && mae <= 2e-5 && iou >= 0.999;
    std::cout << "MODNet " << name << " max_abs=" << maximum << " mae=" << mae
              << " iou=" << iou << " finite=" << finite << " pass=" << ok
              << '\n';
    all_ok = all_ok && ok;
  }
  const auto after = device.allocation_stats();
  return Check(before.allocation_count == after.allocation_count,
               "real-model replay allocated new device memory") &&
         all_ok;
}

} // namespace
int main() {
  std::string error;
  vk::VulkanDevice device;
  if (!ov::InitializeProductionVulkanMattingDevice(&device, &error)) {
    if (std::getenv("STUDIOCAST_REQUIRE_VULKAN_RUNTIME"))
      return Check(false, error) ? 0 : 1;
    std::cout << "SKIP: native ncnn Vulkan unavailable: " << error << '\n';
    return 77;
  }
  char pattern[] = "/tmp/studiocast-native-ncnn-test-XXXXXX";
  const char *root = mkdtemp(pattern);
  if (!root)
    return 1;
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code ec;
      std::filesystem::remove_all(path, ec);
    }
  } cleanup{root};
  const auto param = cleanup.path / "model.param",
             bin = cleanup.path / "model.bin";
  {
    std::ofstream file(param);
    file << "7767517\n2 2\nInput input 0 1 rgb\nConvolution matte 1 1 rgb "
            "alpha 0=1 1=1 5=0 6=3\n";
  }
  {
    std::ofstream file(bin, std::ios::binary);
    const std::uint32_t tag = 0;
    const float weights[3] = {1, 0, 0};
    file.write(reinterpret_cast<const char *>(&tag), sizeof(tag));
    file.write(reinterpret_cast<const char *>(weights), sizeof(weights));
  }
  ov::VulkanMattingGraphDescriptor graph;
  graph.param_path = param;
  graph.bin_path = bin;
  graph.param_sha256 = std::string(64, 'a');
  graph.bin_sha256 = std::string(64, 'b');
  graph.input_blob = "rgb";
  graph.output_blob = "alpha";
  graph.converter_name = "test";
  graph.converter_version = "1";
  graph.precision = "fp32";
  graph.input_w = graph.input_h = graph.output_w = graph.output_h = 4;
  vk::VulkanTensor input;
  vk::VulkanImage alpha;
  vk::VulkanBuffer upload, readback;
  if (!Check(input.AllocateNchwF32(&device, 1, 3, 4, 4, false, &error),
             error) ||
      !Check(alpha.Allocate(&device, 4, 4, vk::VulkanPixelFormat::f32_1, false,
                            &error),
             error) ||
      !Check(upload.Allocate(
                 &device, input.bytes(), vk::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                 vk::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true, &error),
             error) ||
      !Check(readback.Allocate(&device, alpha.byte_size(),
                               vk::VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               vk::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                               vk::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true,
                               &error),
             error))
    return 1;
  ov::VulkanMattingPersistentResources resources;
  resources.input_bytes = input.bytes();
  resources.alpha_bytes = 16 * sizeof(float);
  resources.input_buffer = Opaque(input.buffer());
  resources.alpha_buffer = Opaque(alpha.buffer());
  resources.alpha_row_stride_bytes = alpha.pitch_bytes();
  resources.alpha_buffer_capacity_bytes = alpha.byte_size();
  auto fill = [&](float value) {
    auto *pixels = static_cast<float *>(upload.mapped());
    for (int i = 0; i < 48; ++i)
      pixels[i] = i < 16 ? value : 0.f;
    return upload.Flush(&error) &&
           Transfer(device, upload.buffer(), input.buffer(), input.bytes(),
                    &error);
  };
  if (!Check(fill(0.25f), error))
    return 1;
  auto lifecycle = std::make_unique<ov::VulkanMattingRuntimeLifecycle>(
      ov::CreateProductionVulkanMattingRuntime());
  if (!Check(lifecycle->Prepare(Context(device), graph, resources, &error),
             error) ||
      !Check(lifecycle->Warmup(&error), error))
    return 1;
  const auto before = device.allocation_stats();
  ov::VulkanMattingBufferBinding in, out;
  auto bind = [&](auto &binding, std::uintptr_t buffer, std::size_t size,
                  int channels) {
    binding.ownership_domain = &device;
    binding.logical_device = Opaque(device.device());
    binding.buffer = buffer;
    binding.byte_size = size;
    binding.n = 1;
    binding.c = channels;
    binding.w = binding.h = 4;
    binding.device_resident = binding.device_local_memory =
        binding.access_synchronized = binding.runtime_queue_ownership = true;
    binding.context_id = device.context_identity().context_id;
    binding.context_generation = device.context_identity().generation;
  };
  bind(in, resources.input_buffer, input.bytes(), 3);
  bind(out, resources.alpha_buffer, resources.alpha_bytes, 1);
  for (float value : {0.1f, 0.5f, 0.9f}) {
    if (!Check(fill(value), error) ||
        !Check(lifecycle->Run(in, out, &error), error) ||
        !Check(Transfer(device, alpha.buffer(), readback.buffer(),
                        alpha.byte_size(), &error),
               error) ||
        !Check(readback.Invalidate(&error), error))
      return 1;
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x) {
        const auto *row = reinterpret_cast<const float *>(
            static_cast<const char *>(readback.mapped()) +
            y * alpha.pitch_bytes());
        if (!Check(std::fabs(row[x] - value) < 1e-5f,
                   "ncnn resident output differs from known 1x1 convolution"))
          return 1;
      }
  }
  const auto after = device.allocation_stats();
  if (!Check(before.allocation_count == after.allocation_count,
             "native replay allocated new device memory") ||
      !Check(lifecycle->Evidence().inference_count == 3 &&
                 lifecycle->Evidence().cpu_readback_count == 0,
             "runtime accounting differs"))
    return 1;
  {
    const auto allocation_before = device.allocation_stats();
    auto limited = resources;
    limited.allocation_budget_bytes = 1;
    ov::VulkanMattingRuntimeLifecycle rejected(
        ov::CreateProductionVulkanMattingRuntime());
    if (!Check(!rejected.Prepare(Context(device), graph, limited, &error),
               "undersized reservation was accepted") ||
        !Check(rejected.latched_failure() ==
                   ov::VulkanMattingRuntimeFailure::allocation_contract_failed,
               "undersized reservation classification differs") ||
        !Check(allocation_before.allocation_count ==
                   device.allocation_stats().allocation_count,
               "undersized caller reservation allocated native memory"))
      return 1;
  }
  // Unsupported CPU-only graphs must fail before their model is loaded.
  const auto cpu_param = cleanup.path / "cpu.param";
  {
    std::ofstream file(cpu_param);
    file << "7767517\n2 2\nInput input 0 1 rgb\nBNLL cpu 1 1 rgb alpha\n";
  }
  auto cpu = ov::CreateProductionVulkanMattingRuntime();
  if (!Check(cpu->Initialize(Context(device)).ok(),
             "CPU rejection initialization"))
    return 1;
  graph.param_path = cpu_param;
  if (!Check(cpu->LoadGraph(graph).failure ==
                 ov::VulkanMattingRuntimeFailure::cpu_layer_rejected,
             "CPU-only layer was accepted"))
    return 1;
  if (!Check(RealModelParity(device, &error), error))
    return 1;
  // A timed-out submitted replay must poison the domain and retain its program.
  device.InjectNextSubmissionResultForTesting(
      vk::VulkanSubmissionPhase::wait_for_fence, vk::VK_TIMEOUT);
  if (!Check(!lifecycle->Run(in, out, &error), "timeout did not fail closed") ||
      !Check(!device.SafeToDestroyResources(),
             "unknown completion was treated as releasable") ||
      !Check(lifecycle->failure_latched(),
             "timeout did not latch runtime failure"))
    return 1;
  lifecycle.reset();
  std::cout << "ncnn Vulkan native replay, parity, allocation reuse, CPU "
               "rejection and timeout quarantine passed.\n";
  return 0;
}
