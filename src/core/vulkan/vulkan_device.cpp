#include "core/vulkan/vulkan_device.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <string_view>
#include <vector>

#include "core/util/json.h"

namespace studiocast::vulkan {

namespace {

std::mutex &ProcessSelectionMutex() {
  static std::mutex mutex;
  return mutex;
}

std::optional<VulkanDeviceSelection> &ProcessSelectionStorage() {
  static std::optional<VulkanDeviceSelection> selection;
  return selection;
}

std::optional<VulkanDeviceSelection> ProcessSelection() {
  std::lock_guard<std::mutex> lock(ProcessSelectionMutex());
  return ProcessSelectionStorage();
}

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

std::string TrimCopy(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())))
    value.remove_prefix(1);
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())))
    value.remove_suffix(1);
  return std::string(value);
}

std::string LowerCopy(std::string_view value) {
  std::string out(value);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

bool ParseBool(std::string_view raw, bool *out) {
  const std::string value = LowerCopy(TrimCopy(raw));
  if (value.empty() || value == "0" || value == "false" || value == "no" ||
      value == "off") {
    *out = false;
    return true;
  }
  if (value == "1" || value == "true" || value == "yes" || value == "on") {
    *out = true;
    return true;
  }
  return false;
}

void AppendJsonDeviceCandidates(
    std::ostringstream *oss,
    const std::vector<VulkanDeviceCandidateInfo> &candidates) {
  *oss << "[";
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (i)
      *oss << ",";
    const auto &c = candidates[i];
    *oss << "{";
    *oss << "\"enumeration_index\":" << c.enumeration_index << ",";
    *oss << "\"api_version\":" << c.api_version << ",";
    *oss << "\"driver_version\":" << c.driver_version << ",";
    *oss << "\"vendor_id\":" << c.vendor_id << ",";
    *oss << "\"device_id\":" << c.device_id << ",";
    *oss << "\"device_type\":" << c.device_type << ",";
    *oss << "\"stable_id\":\"" << JsonEscape(c.stable_id) << "\",";
    *oss << "\"vendor_name\":\"" << JsonEscape(c.vendor_name) << "\",";
    *oss << "\"device_name\":\"" << JsonEscape(c.device_name) << "\",";
    *oss << "\"compute_queue_family_index\":" << c.compute_queue_family_index
         << ",";
    *oss << "\"score\":" << c.score << ",";
    *oss << "\"eligible\":" << BoolJson(c.eligible) << ",";
    *oss << "\"selected\":" << BoolJson(c.selected) << ",";
    *oss << "\"rejection_reason\":\"" << JsonEscape(c.rejection_reason) << "\"";
    *oss << "}";
  }
  *oss << "]";
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

namespace detail {

std::string MakeVulkanDeviceStableId(std::uint32_t vendor_id,
                                     std::uint32_t device_id,
                                     std::uint32_t device_type,
                                     std::string_view device_name) {
  std::string normalized;
  bool separator = false;
  for (const char raw : device_name) {
    const auto c = static_cast<unsigned char>(raw);
    if (std::isalnum(c)) {
      if (separator && !normalized.empty())
        normalized.push_back('-');
      normalized.push_back(static_cast<char>(std::tolower(c)));
      separator = false;
    } else {
      separator = true;
    }
  }
  if (normalized.empty())
    normalized = "unnamed";
  std::ostringstream out;
  out << "v1:" << std::hex << std::nouppercase << std::setfill('0')
      << std::setw(4) << vendor_id << ":" << std::setw(4) << device_id << ":"
      << std::dec << device_type << ":" << normalized;
  return out.str();
}

bool IsValidVulkanDeviceStableId(std::string_view stable_id) {
  if (stable_id.size() < 10 || stable_id.rfind("v1:", 0) != 0)
    return false;
  return std::all_of(stable_id.begin(), stable_id.end(), [](unsigned char c) {
    return std::isalnum(c) || c == ':' || c == '-';
  });
}

bool ParseVulkanDeviceSelection(std::string_view requested_index,
                                std::string_view allow_cpu_in_auto,
                                VulkanDeviceSelection *selection,
                                std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!selection) {
    if (error_out)
      *error_out = "Vulkan device selection output is null.";
    return false;
  }

  *selection = VulkanDeviceSelection{};
  const std::string raw_index = TrimCopy(requested_index);
  if (!raw_index.empty()) {
    std::uint32_t index = 0;
    const char *begin = raw_index.data();
    const char *end = begin + raw_index.size();
    const auto parsed = std::from_chars(begin, end, index);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
      if (error_out) {
        *error_out = "STUDIOCAST_VULKAN_DEVICE_INDEX must be a non-negative "
                     "Vulkan enumeration index; got '" +
                     raw_index + "'.";
      }
      return false;
    }
    selection->requested_index = index;
    selection->request = "index:" + std::to_string(index);
    selection->source = "STUDIOCAST_VULKAN_DEVICE_INDEX";
  }

  bool allow_cpu = false;
  if (!ParseBool(allow_cpu_in_auto, &allow_cpu)) {
    if (error_out) {
      *error_out =
          "STUDIOCAST_VULKAN_ALLOW_CPU must be a boolean value; got '" +
          TrimCopy(allow_cpu_in_auto) + "'.";
    }
    return false;
  }
  selection->allow_cpu_in_auto = allow_cpu;
  if (allow_cpu && !selection->requested_index.has_value()) {
    selection->request = "auto_with_cpu";
    selection->source = "STUDIOCAST_VULKAN_ALLOW_CPU";
  }
  return true;
}

VulkanDeviceSelectionResult
SelectVulkanDeviceCandidate(std::vector<VulkanDeviceCandidateInfo> *candidates,
                            const VulkanDeviceSelection &selection) {
  VulkanDeviceSelectionResult result;
  if (!candidates || candidates->empty()) {
    result.failure_reason = "vulkan_no_physical_device";
    result.error = "Vulkan runtime has no physical devices.";
    return result;
  }

  for (auto &candidate : *candidates) {
    if (candidate.stable_id.empty()) {
      candidate.stable_id = MakeVulkanDeviceStableId(
          candidate.vendor_id, candidate.device_id, candidate.device_type,
          candidate.device_name);
    }
    candidate.eligible = false;
    candidate.selected = false;
    candidate.rejection_reason.clear();
    if (candidate.compute_queue_family_index < 0) {
      candidate.rejection_reason = "no_compute_queue";
      continue;
    }
    if (!selection.requested_index.has_value() &&
        selection.requested_stable_id.empty() &&
        candidate.device_type == VK_PHYSICAL_DEVICE_TYPE_CPU &&
        !selection.allow_cpu_in_auto) {
      candidate.rejection_reason = "cpu_device_not_enabled";
      continue;
    }
    candidate.eligible = true;
  }

  if (!selection.requested_stable_id.empty()) {
    const auto matches = std::count_if(
        candidates->begin(), candidates->end(), [&](const auto &candidate) {
          return candidate.stable_id == selection.requested_stable_id;
        });
    if (matches == 0) {
      result.failure_reason = "vulkan_requested_device_not_found";
      result.error = "Saved Vulkan device '" + selection.requested_stable_id +
                     "' was not enumerated; selection was not changed.";
      return result;
    }
    if (matches > 1) {
      result.failure_reason = "vulkan_requested_device_ambiguous";
      result.error = "Saved Vulkan device identity '" +
                     selection.requested_stable_id +
                     "' matched multiple adapters; select a unique adapter.";
      return result;
    }
    const auto it = std::find_if(
        candidates->begin(), candidates->end(), [&](const auto &candidate) {
          return candidate.stable_id == selection.requested_stable_id;
        });
    if (it->compute_queue_family_index < 0) {
      result.failure_reason = "vulkan_requested_device_no_compute_queue";
      result.error = "Saved Vulkan device '" + selection.requested_stable_id +
                     "' does not expose a compute-capable queue.";
      return result;
    }
    for (auto &candidate : *candidates) {
      if (candidate.stable_id != selection.requested_stable_id &&
          candidate.rejection_reason.empty()) {
        candidate.eligible = false;
        candidate.rejection_reason = "not_requested";
      }
    }
    it->eligible = true;
    it->selected = true;
    it->rejection_reason.clear();
    result.ok = true;
    result.candidate_vector_index =
        static_cast<std::size_t>(std::distance(candidates->begin(), it));
    return result;
  }

  if (selection.requested_index.has_value()) {
    const std::uint32_t requested = *selection.requested_index;
    const auto it = std::find_if(
        candidates->begin(), candidates->end(), [&](const auto &candidate) {
          return candidate.enumeration_index == requested;
        });
    if (it == candidates->end()) {
      result.failure_reason = "vulkan_requested_device_not_found";
      result.error = "Requested Vulkan device index " +
                     std::to_string(requested) + " was not enumerated.";
      return result;
    }
    if (it->compute_queue_family_index < 0) {
      result.failure_reason = "vulkan_requested_device_no_compute_queue";
      result.error = "Requested Vulkan device index " +
                     std::to_string(requested) +
                     " does not expose a compute-capable queue.";
      return result;
    }
    for (auto &candidate : *candidates) {
      if (candidate.enumeration_index != requested &&
          candidate.rejection_reason.empty()) {
        candidate.eligible = false;
        candidate.rejection_reason = "not_requested";
      }
    }
    it->eligible = true;
    it->selected = true;
    it->rejection_reason.clear();
    result.ok = true;
    result.candidate_vector_index =
        static_cast<std::size_t>(std::distance(candidates->begin(), it));
    return result;
  }

  auto best = candidates->end();
  for (auto it = candidates->begin(); it != candidates->end(); ++it) {
    if (!it->eligible)
      continue;
    if (best == candidates->end() || it->score > best->score)
      best = it;
  }
  if (best == candidates->end()) {
    const bool cpu_compute_available =
        std::any_of(candidates->begin(), candidates->end(), [](const auto &c) {
          return c.device_type == VK_PHYSICAL_DEVICE_TYPE_CPU &&
                 c.compute_queue_family_index >= 0;
        });
    if (cpu_compute_available) {
      result.failure_reason = "vulkan_only_cpu_devices_available";
      result.error =
          "Only CPU Vulkan devices were found. Set "
          "STUDIOCAST_VULKAN_ALLOW_CPU=1 to opt into a software Vulkan "
          "device, or install a working GPU driver/ICD.";
    } else {
      result.failure_reason = "vulkan_no_compute_queue";
      result.error =
          "No Vulkan physical device exposes a compute-capable queue.";
    }
    return result;
  }

  best->selected = true;
  result.ok = true;
  result.candidate_vector_index =
      static_cast<std::size_t>(std::distance(candidates->begin(), best));
  return result;
}

} // namespace detail

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
  oss << "\"device_selection_source\":\"" << JsonEscape(device_selection_source)
      << "\",";
  oss << "\"device_selection_request\":\""
      << JsonEscape(device_selection_request) << "\",";
  oss << "\"selected_device_index\":";
  if (selected_device_index < 0)
    oss << "null";
  else
    oss << selected_device_index;
  oss << ",";
  oss << "\"selected_device_stable_id\":\""
      << JsonEscape(selected_device_stable_id) << "\",";
  oss << "\"cpu_device_selected\":" << BoolJson(cpu_device_selected) << ",";
  oss << "\"device_candidates\":";
  AppendJsonDeviceCandidates(&oss, device_candidates);
  oss << ",";
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
  if (const auto process_selection = ProcessSelection())
    return Initialize(*process_selection, error_out);

  VulkanDeviceSelection selection;
  std::string e;
  const char *requested_index = std::getenv("STUDIOCAST_VULKAN_DEVICE_INDEX");
  const char *allow_cpu = std::getenv("STUDIOCAST_VULKAN_ALLOW_CPU");
  if (!detail::ParseVulkanDeviceSelection(
          requested_index ? requested_index : "", allow_cpu ? allow_cpu : "",
          &selection, &e)) {
    diagnostics_ = OpenVulkanDiagnostics{};
    diagnostics_.error = e;
    diagnostics_.fallback_reason = "vulkan_device_selection_invalid";
    if (error_out)
      *error_out = e;
    return false;
  }
  return Initialize(selection, error_out);
}

bool VulkanDevice::Initialize(const VulkanDeviceSelection &selection,
                              std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (initialized_)
    return true;

  diagnostics_ = OpenVulkanDiagnostics{};
  identity_ = VulkanDeviceIdentity{};
  std::string e;
  selection_ = selection;
  diagnostics_.device_selection_source = selection_.source;
  diagnostics_.device_selection_request = selection_.request;

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

  std::vector<VulkanDeviceCandidateInfo> candidate_infos;
  candidate_infos.reserve(devices.size());
  for (std::size_t device_index = 0; device_index < devices.size();
       ++device_index) {
    const VkPhysicalDevice dev = devices[device_index];
    VkPhysicalDeviceProperties props{};
    vf.vkGetPhysicalDeviceProperties(dev, &props);
    VulkanDeviceCandidateInfo candidate;
    candidate.enumeration_index = static_cast<std::uint32_t>(device_index);
    candidate.api_version = props.apiVersion;
    candidate.driver_version = props.driverVersion;
    candidate.vendor_id = props.vendorID;
    candidate.device_id = props.deviceID;
    candidate.device_type = static_cast<std::uint32_t>(props.deviceType);
    candidate.stable_id = detail::MakeVulkanDeviceStableId(
        props.vendorID, props.deviceID,
        static_cast<std::uint32_t>(props.deviceType), props.deviceName);
    candidate.vendor_name = VendorName(props.vendorID);
    candidate.device_name = props.deviceName;
    candidate.score = DeviceScore(props);
    std::uint32_t q_count = 0;
    vf.vkGetPhysicalDeviceQueueFamilyProperties(dev, &q_count, nullptr);
    if (q_count > 0) {
      std::vector<VkQueueFamilyProperties> queues(q_count);
      vf.vkGetPhysicalDeviceQueueFamilyProperties(dev, &q_count, queues.data());
      for (std::uint32_t i = 0; i < q_count; ++i) {
        if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            queues[i].queueCount > 0) {
          candidate.compute_queue_family_index = static_cast<int>(i);
          break;
        }
      }
    }
    candidate_infos.push_back(std::move(candidate));
  }

  const VulkanDeviceSelectionResult selected =
      detail::SelectVulkanDeviceCandidate(&candidate_infos, selection_);
  diagnostics_.physical_device_found = true;
  diagnostics_.device_candidates = candidate_infos;
  if (!selected.ok) {
    diagnostics_.fallback_reason = selected.failure_reason;
    if (error_out)
      *error_out = selected.error;
    return false;
  }

  const auto &best = candidate_infos[selected.candidate_vector_index];
  physical_device_ = devices[best.enumeration_index];
  queue_family_index_ =
      static_cast<std::uint32_t>(best.compute_queue_family_index);
  vf.vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);

  diagnostics_.compute_queue_available = true;
  diagnostics_.api_version = best.api_version;
  diagnostics_.driver_version = best.driver_version;
  diagnostics_.vendor_id = best.vendor_id;
  diagnostics_.device_id = best.device_id;
  diagnostics_.device_type = best.device_type;
  diagnostics_.vendor_name = best.vendor_name;
  diagnostics_.device_name = best.device_name;
  diagnostics_.compute_queue_family_index = queue_family_index_;
  diagnostics_.selected_device_index = static_cast<int>(best.enumeration_index);
  diagnostics_.selected_device_stable_id = best.stable_id;
  identity_.stable_id = best.stable_id;
  identity_.vendor_id = best.vendor_id;
  identity_.device_id = best.device_id;
  identity_.device_type = best.device_type;
  identity_.device_name = best.device_name;
  diagnostics_.cpu_device_selected =
      best.device_type == VK_PHYSICAL_DEVICE_TYPE_CPU;
  if (diagnostics_.cpu_device_selected) {
    diagnostics_.warnings.push_back(
        "A CPU Vulkan device was selected; this is a software fallback, not "
        "hardware GPU acceleration.");
  }
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

void SetProcessVulkanDeviceSelection(const VulkanDeviceSelection &selection) {
  std::lock_guard<std::mutex> lock(ProcessSelectionMutex());
  ProcessSelectionStorage() = selection;
}

void ClearProcessVulkanDeviceSelection() {
  std::lock_guard<std::mutex> lock(ProcessSelectionMutex());
  ProcessSelectionStorage().reset();
}

} // namespace studiocast::vulkan
