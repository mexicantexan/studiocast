#include "core/vulkan/vulkan_device.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <sstream>
#include <vector>

#include "core/util/json.h"

namespace studiocast::vulkan {

namespace {

std::string BoolJson(bool v) { return v ? "true" : "false"; }

std::string JsonEscape(const std::string &s) {
  return studiocast::util::json::EscapeString(s);
}

void AppendJsonStringArray(std::ostringstream *oss,
                           const std::vector<std::string> &a) {
  *oss << "[";
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (i)
      *oss << ",";
    *oss << "\"" << JsonEscape(a[i]) << "\"";
  }
  *oss << "]";
}

void AppendJsonStringMap(std::ostringstream *oss,
                         const std::map<std::string, std::string> &m) {
  *oss << "{";
  bool first = true;
  for (const auto &[k, v] : m) {
    if (!first)
      *oss << ",";
    first = false;
    *oss << "\"" << JsonEscape(k) << "\":";
    *oss << "\"" << JsonEscape(v) << "\"";
  }
  *oss << "}";
}

void AppendJsonModels(
    std::ostringstream *oss,
    const std::vector<OpenVulkanDiagnostics::ModelInfo> &models) {
  *oss << "[";
  for (std::size_t i = 0; i < models.size(); ++i) {
    if (i)
      *oss << ",";
    const auto &m = models[i];
    *oss << "{";
    *oss << "\"id\":\"" << JsonEscape(m.id) << "\",";
    *oss << "\"display_name\":\"" << JsonEscape(m.display_name) << "\",";
    *oss << "\"task\":\"" << JsonEscape(m.task) << "\",";
    *oss << "\"width\":" << m.width << ",";
    *oss << "\"height\":" << m.height;
    *oss << "}";
  }
  *oss << "]";
}

std::string VendorName(std::uint32_t vendor_id) {
  switch (vendor_id) {
  case 0x10de:
    return "NVIDIA";
  case 0x1002:
  case 0x1022:
    return "AMD";
  case 0x8086:
    return "Intel";
  case 0x13b5:
    return "Arm";
  case 0x5143:
    return "Qualcomm";
  default:
    return {};
  }
}

int DeviceScore(const VkPhysicalDeviceProperties &props) {
  switch (props.deviceType) {
  case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
    return 400;
  case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
    return 300;
  case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
    return 200;
  case VK_PHYSICAL_DEVICE_TYPE_CPU:
    return 100;
  default:
    return 0;
  }
}

template <typename T>
bool LoadInstanceProc(const VulkanFunctions &f, VkInstance instance,
                      const char *name, T *out, std::string *error_out) {
  if (out)
    *out = nullptr;
  if (!f.vkGetInstanceProcAddr) {
    if (error_out)
      *error_out = "vkGetInstanceProcAddr is not loaded.";
    return false;
  }
  PFN_vkVoidFunction proc = f.vkGetInstanceProcAddr(instance, name);
  if (!proc) {
    if (error_out)
      *error_out = std::string("Vulkan instance proc not found: ") + name;
    return false;
  }
  if (out)
    *out = reinterpret_cast<T>(proc);
  return true;
}

template <typename T>
bool LoadDeviceProc(const VulkanFunctions &f, VkDevice device, const char *name,
                    T *out, std::string *error_out) {
  if (out)
    *out = nullptr;
  if (!f.vkGetDeviceProcAddr) {
    if (error_out)
      *error_out = "vkGetDeviceProcAddr is not loaded.";
    return false;
  }
  PFN_vkVoidFunction proc = f.vkGetDeviceProcAddr(device, name);
  if (!proc) {
    if (error_out)
      *error_out = std::string("Vulkan device proc not found: ") + name;
    return false;
  }
  if (out)
    *out = reinterpret_cast<T>(proc);
  return true;
}

bool ResultOk(VkResult result, const char *what, std::string *error_out) {
  if (result == VK_SUCCESS)
    return true;
  if (error_out) {
    *error_out = std::string(what) + " failed: " + VkResultName(result);
  }
  return false;
}

} // namespace

std::string OpenVulkanDiagnostics::ToJson() const {
  std::ostringstream oss;
  oss << "{";
  oss << "\"compiled_enabled\":" << BoolJson(compiled_enabled) << ",";
  oss << "\"ok\":" << BoolJson(ok) << ",";
  oss << "\"runtime_library_found\":" << BoolJson(runtime_library_found)
      << ",";
  oss << "\"runtime_library_path\":\"" << JsonEscape(runtime_library_path)
      << "\",";
  oss << "\"instance_created\":" << BoolJson(instance_created) << ",";
  oss << "\"physical_device_found\":" << BoolJson(physical_device_found)
      << ",";
  oss << "\"compute_queue_available\":" << BoolJson(compute_queue_available)
      << ",";
  oss << "\"logical_device_created\":" << BoolJson(logical_device_created)
      << ",";
  oss << "\"shader_pipeline_created\":"
      << BoolJson(shader_pipeline_created) << ",";
  oss << "\"api_version\":" << api_version << ",";
  oss << "\"driver_version\":" << driver_version << ",";
  oss << "\"vendor_id\":" << vendor_id << ",";
  oss << "\"device_id\":" << device_id << ",";
  oss << "\"device_type\":" << device_type << ",";
  oss << "\"vendor_name\":\"" << JsonEscape(vendor_name) << "\",";
  oss << "\"device_name\":\"" << JsonEscape(device_name) << "\",";
  oss << "\"compute_queue_family_index\":" << compute_queue_family_index
      << ",";
  oss << "\"error\":\"" << JsonEscape(error) << "\",";
  oss << "\"fallback_reason\":\"" << JsonEscape(fallback_reason) << "\",";
  oss << "\"blocked_reason\":\"" << JsonEscape(blocked_reason) << "\",";
  oss << "\"degraded_reason\":\"" << JsonEscape(degraded_reason) << "\",";
  oss << "\"installed_models\":";
  AppendJsonStringArray(&oss, installed_models);
  oss << ",";
  oss << "\"default_model_id\":\"" << JsonEscape(default_model_id) << "\",";
  oss << "\"models\":";
  AppendJsonModels(&oss, models);
  oss << ",";
  oss << "\"missing_models\":";
  AppendJsonStringMap(&oss, missing_models);
  oss << ",";
  oss << "\"available_effects\":";
  AppendJsonStringArray(&oss, available_effects);
  oss << ",";
  oss << "\"blocked_effects\":";
  AppendJsonStringMap(&oss, blocked_effects);
  oss << ",";
  oss << "\"matting_runtime\":\"" << JsonEscape(matting_runtime) << "\",";
  oss << "\"matting_runtime_created\":"
      << BoolJson(matting_runtime_created) << ",";
  oss << "\"matting_graph_loaded\":" << BoolJson(matting_graph_loaded)
      << ",";
  oss << "\"input_device_resident\":" << BoolJson(input_device_resident)
      << ",";
  oss << "\"alpha_device_resident\":" << BoolJson(alpha_device_resident)
      << ",";
  oss << "\"output_device_resident\":" << BoolJson(output_device_resident)
      << ",";
  oss << "\"device_residency_mode\":\""
      << JsonEscape(device_residency_mode) << "\",";
  oss << "\"warnings\":";
  AppendJsonStringArray(&oss, warnings);
  oss << ",";
  oss << "\"install_hints\":";
  if (install_hints.empty()) {
    AppendJsonStringArray(
        &oss, {"Install a Vulkan loader/runtime such as libvulkan1.",
               "Rebuild with -DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON to enable "
               "this backend."});
  } else {
    AppendJsonStringArray(&oss, install_hints);
  }
  oss << "}";
  return oss.str();
}

VulkanDevice::~VulkanDevice() { Shutdown(); }

bool VulkanDevice::Initialize(std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (initialized_)
    return true;

  diagnostics_ = OpenVulkanDiagnostics{};
  std::string e;
  if (!loader_.Load(&e)) {
    diagnostics_.error = e;
    diagnostics_.fallback_reason = "vulkan_runtime_not_found";
    if (error_out)
      *error_out = e;
    return false;
  }
  diagnostics_.runtime_library_found = true;
  diagnostics_.runtime_library_path = loader_.LibraryPath().string();

  const VulkanFunctions &lf = loader_.f();
  VkApplicationInfo app{};
  app.pApplicationName = "StudioCast";
  app.applicationVersion = 1;
  app.pEngineName = "StudioCast Open Vulkan";
  app.engineVersion = 1;
  app.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo create{};
  create.pApplicationInfo = &app;
  const VkResult instance_result =
      lf.vkCreateInstance(&create, nullptr, &instance_);
  if (!ResultOk(instance_result, "vkCreateInstance", &e)) {
    diagnostics_.error = e;
    diagnostics_.fallback_reason = "vulkan_instance_create_failed";
    if (error_out)
      *error_out = e;
    return false;
  }
  diagnostics_.instance_created = true;

  if (!LoadInstanceFunctions(&e) || !PickPhysicalDevice(&e) ||
      !CreateLogicalDevice(&e) || !LoadDeviceFunctions(&e) ||
      !CreateCommandInfrastructure(&e)) {
    diagnostics_.error = e;
    if (diagnostics_.fallback_reason.empty())
      diagnostics_.fallback_reason = "vulkan_device_init_failed";
    if (error_out)
      *error_out = e;
    Shutdown();
    return false;
  }

  initialized_ = true;
  diagnostics_.ok = true;
  return true;
}

void VulkanDevice::Shutdown() noexcept {
  const auto &vf = loader_.f();
  if (device_) {
    if (vf.vkDeviceWaitIdle)
      (void)vf.vkDeviceWaitIdle(device_);
    if (fence_ && vf.vkDestroyFence)
      vf.vkDestroyFence(device_, fence_, nullptr);
    fence_ = nullptr;
    if (command_pool_ && vf.vkDestroyCommandPool)
      vf.vkDestroyCommandPool(device_, command_pool_, nullptr);
    command_pool_ = nullptr;
    if (vf.vkDestroyDevice)
      vf.vkDestroyDevice(device_, nullptr);
    device_ = nullptr;
  }
  queue_ = nullptr;
  physical_device_ = nullptr;
  if (instance_ && vf.vkDestroyInstance) {
    vf.vkDestroyInstance(instance_, nullptr);
  }
  instance_ = nullptr;
  initialized_ = false;
}

bool VulkanDevice::LoadInstanceFunctions(std::string *error_out) {
  VulkanFunctions &vf = loader_.f();
  return LoadInstanceProc(vf, instance_, "vkDestroyInstance",
                          &vf.vkDestroyInstance, error_out) &&
         LoadInstanceProc(vf, instance_, "vkEnumeratePhysicalDevices",
                          &vf.vkEnumeratePhysicalDevices, error_out) &&
         LoadInstanceProc(vf, instance_, "vkGetPhysicalDeviceProperties",
                          &vf.vkGetPhysicalDeviceProperties, error_out) &&
         LoadInstanceProc(vf, instance_,
                          "vkGetPhysicalDeviceQueueFamilyProperties",
                          &vf.vkGetPhysicalDeviceQueueFamilyProperties,
                          error_out) &&
         LoadInstanceProc(vf, instance_, "vkGetPhysicalDeviceMemoryProperties",
                          &vf.vkGetPhysicalDeviceMemoryProperties,
                          error_out) &&
         LoadInstanceProc(vf, instance_, "vkCreateDevice",
                          &vf.vkCreateDevice, error_out) &&
         LoadInstanceProc(vf, instance_, "vkGetDeviceProcAddr",
                          &vf.vkGetDeviceProcAddr, error_out);
}

bool VulkanDevice::PickPhysicalDevice(std::string *error_out) {
  const auto &vf = loader_.f();
  std::uint32_t count = 0;
  VkResult result = vf.vkEnumeratePhysicalDevices(instance_, &count, nullptr);
  if (!ResultOk(result, "vkEnumeratePhysicalDevices(count)", error_out))
    return false;
  if (count == 0) {
    diagnostics_.fallback_reason = "vulkan_no_physical_device";
    if (error_out)
      *error_out = "Vulkan runtime has no physical devices.";
    return false;
  }

  std::vector<VkPhysicalDevice> devices(count);
  result = vf.vkEnumeratePhysicalDevices(instance_, &count, devices.data());
  if (!ResultOk(result, "vkEnumeratePhysicalDevices(list)", error_out))
    return false;

  struct Candidate {
    VkPhysicalDevice device = nullptr;
    VkPhysicalDeviceProperties properties{};
    std::uint32_t queue_family_index = 0;
    int score = 0;
  };
  std::vector<Candidate> candidates;
  for (VkPhysicalDevice dev : devices) {
    VkPhysicalDeviceProperties props{};
    vf.vkGetPhysicalDeviceProperties(dev, &props);
    std::uint32_t q_count = 0;
    vf.vkGetPhysicalDeviceQueueFamilyProperties(dev, &q_count, nullptr);
    if (q_count == 0)
      continue;
    std::vector<VkQueueFamilyProperties> queues(q_count);
    vf.vkGetPhysicalDeviceQueueFamilyProperties(dev, &q_count, queues.data());
    for (std::uint32_t i = 0; i < q_count; ++i) {
      if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
          queues[i].queueCount > 0) {
        Candidate c;
        c.device = dev;
        c.properties = props;
        c.queue_family_index = i;
        c.score = DeviceScore(props);
        candidates.push_back(c);
        break;
      }
    }
  }

  if (candidates.empty()) {
    diagnostics_.physical_device_found = true;
    diagnostics_.fallback_reason = "vulkan_no_compute_queue";
    if (error_out)
      *error_out = "No Vulkan physical device exposes a compute-capable queue.";
    return false;
  }

  const auto best =
      std::max_element(candidates.begin(), candidates.end(),
                       [](const Candidate &a, const Candidate &b) {
                         return a.score < b.score;
                       });
  physical_device_ = best->device;
  queue_family_index_ = best->queue_family_index;
  vf.vkGetPhysicalDeviceMemoryProperties(physical_device_,
                                         &memory_properties_);

  diagnostics_.physical_device_found = true;
  diagnostics_.compute_queue_available = true;
  diagnostics_.api_version = best->properties.apiVersion;
  diagnostics_.driver_version = best->properties.driverVersion;
  diagnostics_.vendor_id = best->properties.vendorID;
  diagnostics_.device_id = best->properties.deviceID;
  diagnostics_.device_type =
      static_cast<std::uint32_t>(best->properties.deviceType);
  diagnostics_.vendor_name = VendorName(best->properties.vendorID);
  diagnostics_.device_name = best->properties.deviceName;
  diagnostics_.compute_queue_family_index = queue_family_index_;
  return true;
}

bool VulkanDevice::CreateLogicalDevice(std::string *error_out) {
  const auto &vf = loader_.f();
  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queue{};
  queue.queueFamilyIndex = queue_family_index_;
  queue.queueCount = 1;
  queue.pQueuePriorities = &priority;

  VkDeviceCreateInfo create{};
  create.queueCreateInfoCount = 1;
  create.pQueueCreateInfos = &queue;

  const VkResult result =
      vf.vkCreateDevice(physical_device_, &create, nullptr, &device_);
  if (!ResultOk(result, "vkCreateDevice", error_out)) {
    diagnostics_.fallback_reason = "vulkan_device_create_failed";
    return false;
  }
  diagnostics_.logical_device_created = true;
  return true;
}

bool VulkanDevice::LoadDeviceFunctions(std::string *error_out) {
  VulkanFunctions &vf = loader_.f();
  return LoadDeviceProc(vf, device_, "vkDestroyDevice", &vf.vkDestroyDevice,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkGetDeviceQueue", &vf.vkGetDeviceQueue,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkDeviceWaitIdle",
                        &vf.vkDeviceWaitIdle, error_out) &&
         LoadDeviceProc(vf, device_, "vkCreateCommandPool",
                        &vf.vkCreateCommandPool, error_out) &&
         LoadDeviceProc(vf, device_, "vkDestroyCommandPool",
                        &vf.vkDestroyCommandPool, error_out) &&
         LoadDeviceProc(vf, device_, "vkAllocateCommandBuffers",
                        &vf.vkAllocateCommandBuffers, error_out) &&
         LoadDeviceProc(vf, device_, "vkFreeCommandBuffers",
                        &vf.vkFreeCommandBuffers, error_out) &&
         LoadDeviceProc(vf, device_, "vkCreateFence", &vf.vkCreateFence,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkDestroyFence", &vf.vkDestroyFence,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkResetFences", &vf.vkResetFences,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkWaitForFences", &vf.vkWaitForFences,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkQueueSubmit", &vf.vkQueueSubmit,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkCreateBuffer", &vf.vkCreateBuffer,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkDestroyBuffer", &vf.vkDestroyBuffer,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkGetBufferMemoryRequirements",
                        &vf.vkGetBufferMemoryRequirements, error_out) &&
         LoadDeviceProc(vf, device_, "vkAllocateMemory",
                        &vf.vkAllocateMemory, error_out) &&
         LoadDeviceProc(vf, device_, "vkFreeMemory", &vf.vkFreeMemory,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkBindBufferMemory",
                        &vf.vkBindBufferMemory, error_out) &&
         LoadDeviceProc(vf, device_, "vkMapMemory", &vf.vkMapMemory,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkUnmapMemory", &vf.vkUnmapMemory,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkFlushMappedMemoryRanges",
                        &vf.vkFlushMappedMemoryRanges, error_out) &&
         LoadDeviceProc(vf, device_, "vkInvalidateMappedMemoryRanges",
                        &vf.vkInvalidateMappedMemoryRanges, error_out) &&
         LoadDeviceProc(vf, device_, "vkCreateDescriptorSetLayout",
                        &vf.vkCreateDescriptorSetLayout, error_out) &&
         LoadDeviceProc(vf, device_, "vkDestroyDescriptorSetLayout",
                        &vf.vkDestroyDescriptorSetLayout, error_out) &&
         LoadDeviceProc(vf, device_, "vkCreateDescriptorPool",
                        &vf.vkCreateDescriptorPool, error_out) &&
         LoadDeviceProc(vf, device_, "vkDestroyDescriptorPool",
                        &vf.vkDestroyDescriptorPool, error_out) &&
         LoadDeviceProc(vf, device_, "vkAllocateDescriptorSets",
                        &vf.vkAllocateDescriptorSets, error_out) &&
         LoadDeviceProc(vf, device_, "vkUpdateDescriptorSets",
                        &vf.vkUpdateDescriptorSets, error_out) &&
         LoadDeviceProc(vf, device_, "vkCreatePipelineLayout",
                        &vf.vkCreatePipelineLayout, error_out) &&
         LoadDeviceProc(vf, device_, "vkDestroyPipelineLayout",
                        &vf.vkDestroyPipelineLayout, error_out) &&
         LoadDeviceProc(vf, device_, "vkCreateShaderModule",
                        &vf.vkCreateShaderModule, error_out) &&
         LoadDeviceProc(vf, device_, "vkDestroyShaderModule",
                        &vf.vkDestroyShaderModule, error_out) &&
         LoadDeviceProc(vf, device_, "vkCreateComputePipelines",
                        &vf.vkCreateComputePipelines, error_out) &&
         LoadDeviceProc(vf, device_, "vkDestroyPipeline",
                        &vf.vkDestroyPipeline, error_out) &&
         LoadDeviceProc(vf, device_, "vkBeginCommandBuffer",
                        &vf.vkBeginCommandBuffer, error_out) &&
         LoadDeviceProc(vf, device_, "vkEndCommandBuffer",
                        &vf.vkEndCommandBuffer, error_out) &&
         LoadDeviceProc(vf, device_, "vkResetCommandBuffer",
                        &vf.vkResetCommandBuffer, error_out) &&
         LoadDeviceProc(vf, device_, "vkCmdBindPipeline",
                        &vf.vkCmdBindPipeline, error_out) &&
         LoadDeviceProc(vf, device_, "vkCmdBindDescriptorSets",
                        &vf.vkCmdBindDescriptorSets, error_out) &&
         LoadDeviceProc(vf, device_, "vkCmdPushConstants",
                        &vf.vkCmdPushConstants, error_out) &&
         LoadDeviceProc(vf, device_, "vkCmdDispatch", &vf.vkCmdDispatch,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkCmdCopyBuffer", &vf.vkCmdCopyBuffer,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkCmdPipelineBarrier",
                        &vf.vkCmdPipelineBarrier, error_out);
}

bool VulkanDevice::CreateCommandInfrastructure(std::string *error_out) {
  const auto &vf = loader_.f();
  vf.vkGetDeviceQueue(device_, queue_family_index_, 0, &queue_);
  if (!queue_) {
    if (error_out)
      *error_out = "vkGetDeviceQueue returned a null queue.";
    return false;
  }

  VkCommandPoolCreateInfo pool{};
  pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool.queueFamilyIndex = queue_family_index_;
  VkResult result =
      vf.vkCreateCommandPool(device_, &pool, nullptr, &command_pool_);
  if (!ResultOk(result, "vkCreateCommandPool", error_out))
    return false;

  VkFenceCreateInfo fence{};
  result = vf.vkCreateFence(device_, &fence, nullptr, &fence_);
  return ResultOk(result, "vkCreateFence", error_out);
}

bool VulkanDevice::FindMemoryType(std::uint32_t type_bits,
                                  VkFlags required_flags,
                                  VkFlags preferred_flags,
                                  std::uint32_t *type_index,
                                  std::string *error_out) const {
  if (type_index)
    *type_index = 0;
  auto matches = [&](std::uint32_t i, VkFlags flags) {
    if ((type_bits & (1u << i)) == 0)
      return false;
    const VkFlags props = memory_properties_.memoryTypes[i].propertyFlags;
    return (props & required_flags) == required_flags &&
           (props & flags) == flags;
  };

  for (std::uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
    if (matches(i, preferred_flags)) {
      if (type_index)
        *type_index = i;
      return true;
    }
  }
  for (std::uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
    if (matches(i, 0)) {
      if (type_index)
        *type_index = i;
      return true;
    }
  }

  if (error_out) {
    std::ostringstream oss;
    oss << "No Vulkan memory type matched type_bits=0x" << std::hex
        << type_bits << " required=0x" << required_flags << " preferred=0x"
        << preferred_flags << ".";
    *error_out = oss.str();
  }
  return false;
}

bool VulkanDevice::SubmitAndWait(VkCommandBuffer command_buffer,
                                 std::string *error_out) {
  if (!initialized_) {
    if (error_out)
      *error_out = "Vulkan device is not initialized.";
    return false;
  }
  const auto &vf = loader_.f();
  VkResult result = vf.vkResetFences(device_, 1, &fence_);
  if (!ResultOk(result, "vkResetFences", error_out))
    return false;

  VkSubmitInfo submit{};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command_buffer;
  result = vf.vkQueueSubmit(queue_, 1, &submit, fence_);
  if (!ResultOk(result, "vkQueueSubmit", error_out))
    return false;

  result = vf.vkWaitForFences(device_, 1, &fence_, 1,
                              VK_DEFAULT_FENCE_TIMEOUT_NS);
  if (result == VK_TIMEOUT) {
    if (error_out)
      *error_out = "vkWaitForFences timed out after 10 seconds.";
    return false;
  }
  return ResultOk(result, "vkWaitForFences", error_out);
}

bool VulkanDevice::FlushMemory(VkDeviceMemory memory, VkDeviceSize offset,
                               VkDeviceSize size,
                               std::string *error_out) const {
  VkMappedMemoryRange range{};
  range.memory = memory;
  range.offset = offset;
  range.size = size;
  return ResultOk(loader_.f().vkFlushMappedMemoryRanges(device_, 1, &range),
                  "vkFlushMappedMemoryRanges", error_out);
}

bool VulkanDevice::InvalidateMemory(VkDeviceMemory memory, VkDeviceSize offset,
                                    VkDeviceSize size,
                                    std::string *error_out) const {
  VkMappedMemoryRange range{};
  range.memory = memory;
  range.offset = offset;
  range.size = size;
  return ResultOk(
      loader_.f().vkInvalidateMappedMemoryRanges(device_, 1, &range),
      "vkInvalidateMappedMemoryRanges", error_out);
}

} // namespace studiocast::vulkan
