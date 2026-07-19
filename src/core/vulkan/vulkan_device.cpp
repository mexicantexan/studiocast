#include "core/vulkan/vulkan_device.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string_view>
#include <vector>

#include "core/util/json.h"

namespace studiocast::vulkan {

namespace {

std::atomic<std::uint64_t> &NextContextId() {
  static std::atomic<std::uint64_t> next{1};
  return next;
}

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

VkDeviceSize DerivedAllocationBudget(
    const VkPhysicalDeviceMemoryProperties &memory_properties) {
  VkDeviceSize total = 0;
  for (std::uint32_t i = 0; i < memory_properties.memoryHeapCount; ++i) {
    const VkDeviceSize heap_size = memory_properties.memoryHeaps[i].size;
    if (heap_size > std::numeric_limits<VkDeviceSize>::max() - total)
      return std::numeric_limits<VkDeviceSize>::max();
    total += heap_size;
  }
  return total;
}

void QuarantineUnsafeTimeoutLoader(std::shared_ptr<VulkanLoader> loader) {
  // A fence timeout means command buffers/resources may still be in use.
  // Calling child destruction or vkDestroyDevice can violate Vulkan lifetime
  // rules or block indefinitely. Retain the loader until process teardown and
  // let the OS reclaim the abandoned poisoned context.
  static auto *mutex = new std::mutex();
  static auto *loaders = new std::vector<std::shared_ptr<VulkanLoader>>();
  std::lock_guard<std::mutex> lock(*mutex);
  loaders->push_back(std::move(loader));
}

} // namespace

const char *VulkanContextHealthName(VulkanContextHealth health) {
  switch (health) {
  case VulkanContextHealth::uninitialized:
    return "uninitialized";
  case VulkanContextHealth::healthy:
    return "healthy";
  case VulkanContextHealth::device_lost:
    return "device_lost";
  case VulkanContextHealth::unsafe_timeout:
    return "unsafe_timeout";
  case VulkanContextHealth::fatal_submission_error:
    return "fatal_submission_error";
  case VulkanContextHealth::shutdown:
    return "shutdown";
  }
  return "unknown";
}

VulkanBufferBarrierSpec VulkanBufferBarrier(VulkanBufferAccess source,
                                            VulkanBufferAccess destination) {
  auto access_and_stage = [](VulkanBufferAccess access) {
    struct AccessStage {
      VkFlags access = 0;
      VkFlags stage = 0;
    };
    switch (access) {
    case VulkanBufferAccess::host_write:
      return AccessStage{VK_ACCESS_HOST_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT};
    case VulkanBufferAccess::host_or_compute_write:
      return AccessStage{VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT |
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    case VulkanBufferAccess::transfer_write:
      return AccessStage{VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT};
    case VulkanBufferAccess::compute_read:
      return AccessStage{VK_ACCESS_SHADER_READ_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    case VulkanBufferAccess::compute_write:
      return AccessStage{VK_ACCESS_SHADER_WRITE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    case VulkanBufferAccess::transfer_read:
      return AccessStage{VK_ACCESS_TRANSFER_READ_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT};
    case VulkanBufferAccess::host_read:
      return AccessStage{VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_HOST_BIT};
    }
    return AccessStage{};
  };
  const auto src = access_and_stage(source);
  const auto dst = access_and_stage(destination);
  return {src.access, dst.access, src.stage, dst.stage};
}

namespace detail {

VulkanContextState::~VulkanContextState() {
  if (!loader)
    return;
  if (!SafeToDestroyChildren()) {
    QuarantineUnsafeTimeoutLoader(std::move(loader));
    return;
  }
  const auto &vf = loader->f();
  if (device) {
    if (fence && vf.vkDestroyFence)
      vf.vkDestroyFence(device, fence, nullptr);
    fence = nullptr;
    if (command_pool && vf.vkDestroyCommandPool)
      vf.vkDestroyCommandPool(device, command_pool, nullptr);
    command_pool = nullptr;
    if (vf.vkDestroyDevice)
      vf.vkDestroyDevice(device, nullptr);
    device = nullptr;
  }
  queue = nullptr;
  physical_device = nullptr;
  if (instance && vf.vkDestroyInstance)
    vf.vkDestroyInstance(instance, nullptr);
  instance = nullptr;
}

bool VulkanContextState::Active() const {
  std::lock_guard<std::mutex> lock(health_mutex);
  return health.health != VulkanContextHealth::uninitialized &&
         health.health != VulkanContextHealth::shutdown;
}

bool VulkanContextState::Healthy() const {
  std::lock_guard<std::mutex> lock(health_mutex);
  return health.health == VulkanContextHealth::healthy && !health.poisoned;
}

bool VulkanContextState::SafeToDestroyChildren() const {
  std::lock_guard<std::mutex> lock(health_mutex);
  return health.health != VulkanContextHealth::unsafe_timeout;
}

VulkanHealthSnapshot VulkanContextState::HealthSnapshot() const {
  std::lock_guard<std::mutex> lock(health_mutex);
  return health;
}

VulkanAllocationStats VulkanContextState::AllocationStats() const {
  std::lock_guard<std::mutex> lock(allocation_mutex);
  return allocation_stats;
}

bool VulkanContextState::ReserveAllocation(VkDeviceSize bytes,
                                           std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!Healthy()) {
    const auto snapshot = HealthSnapshot();
    if (error_out) {
      *error_out = "[" + snapshot.reason_code +
                   "] Vulkan context is not healthy for allocation.";
    }
    return false;
  }
  if (bytes == 0) {
    if (error_out)
      *error_out = "[vulkan_allocation_size_invalid] Vulkan allocation size "
                   "must be non-zero.";
    return false;
  }

  std::lock_guard<std::mutex> lock(allocation_mutex);
  const VkDeviceSize current = allocation_stats.current_bytes;
  if (bytes > std::numeric_limits<VkDeviceSize>::max() - current) {
    if (error_out) {
      *error_out = "[vulkan_allocation_size_overflow] Vulkan allocation "
                   "accounting overflow.";
    }
    return false;
  }
  const VkDeviceSize next = current + bytes;
  if (next > allocation_stats.budget_bytes) {
    if (error_out) {
      *error_out =
          "[vulkan_allocation_budget_exceeded] Vulkan context "
          "allocation budget exceeded (requested=" +
          std::to_string(bytes) + ", current=" + std::to_string(current) +
          ", budget=" + std::to_string(allocation_stats.budget_bytes) + ").";
    }
    return false;
  }
  allocation_stats.current_bytes = next;
  allocation_stats.high_water_bytes =
      std::max(allocation_stats.high_water_bytes, next);
  ++allocation_stats.allocation_count;
  return true;
}

void VulkanContextState::ReleaseAllocation(VkDeviceSize bytes) noexcept {
  if (bytes == 0)
    return;
  std::lock_guard<std::mutex> lock(allocation_mutex);
  if (bytes <= allocation_stats.current_bytes)
    allocation_stats.current_bytes -= bytes;
  else
    allocation_stats.current_bytes = 0;
  if (allocation_stats.allocation_count > 0)
    --allocation_stats.allocation_count;
}

bool VulkanContextState::RecordDriverFailure(VkResult result,
                                             std::string_view operation,
                                             bool submission_failure,
                                             std::string *error_out) {
  if (result == VK_SUCCESS)
    return true;

  std::string reason = "vulkan_driver_error";
  VulkanContextHealth next_health = VulkanContextHealth::healthy;
  bool poison = false;
  if (result == VK_ERROR_DEVICE_LOST) {
    reason = "vulkan_device_lost";
    next_health = VulkanContextHealth::device_lost;
    poison = true;
  } else if (result == VK_TIMEOUT) {
    reason = "vulkan_submission_timeout";
    next_health = VulkanContextHealth::unsafe_timeout;
    poison = true;
  } else if (submission_failure) {
    reason = "vulkan_submission_failed";
    next_health = VulkanContextHealth::fatal_submission_error;
    poison = true;
  } else if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
    reason = "vulkan_out_of_device_memory";
  } else if (result == VK_ERROR_OUT_OF_HOST_MEMORY) {
    reason = "vulkan_out_of_host_memory";
  } else if (result == VK_ERROR_MEMORY_MAP_FAILED) {
    reason = "vulkan_memory_map_failed";
  } else if (result == VK_ERROR_INITIALIZATION_FAILED) {
    reason = "vulkan_initialization_failed";
  }

  const std::string message = "[" + reason + "] " + std::string(operation) +
                              " failed: " + VkResultName(result) + ".";
  if (poison) {
    std::lock_guard<std::mutex> lock(health_mutex);
    if (!health.poisoned) {
      health.health = next_health;
      health.poisoned = true;
      health.reason_code = reason;
      health.last_result = result;
      health.operation = operation;
      health.message = message;
    }
  }
  if (error_out)
    *error_out = message;
  return false;
}

void VulkanContextState::MarkShutdown() noexcept {
  std::lock_guard<std::mutex> lock(health_mutex);
  if (!health.poisoned) {
    health.health = VulkanContextHealth::shutdown;
    health.reason_code = "vulkan_context_shutdown";
    health.operation = "shutdown";
    health.message = "Vulkan context was shut down.";
  }
}

} // namespace detail

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
  oss << "\"runtime_library_found\":" << BoolJson(runtime_library_found) << ",";
  oss << "\"runtime_library_path\":\"" << JsonEscape(runtime_library_path)
      << "\",";
  oss << "\"instance_created\":" << BoolJson(instance_created) << ",";
  oss << "\"physical_device_found\":" << BoolJson(physical_device_found) << ",";
  oss << "\"non_cpu_device_selected\":" << BoolJson(non_cpu_device_selected)
      << ",";
  oss << "\"compute_queue_available\":" << BoolJson(compute_queue_available)
      << ",";
  oss << "\"logical_device_created\":" << BoolJson(logical_device_created)
      << ",";
  oss << "\"context_created\":" << BoolJson(context_created) << ",";
  oss << "\"context_healthy\":" << BoolJson(context_healthy) << ",";
  oss << "\"production_hardware_ready\":" << BoolJson(production_hardware_ready)
      << ",";
  oss << "\"context_health\":\"" << JsonEscape(context_health) << "\",";
  oss << "\"context_failure_reason\":\"" << JsonEscape(context_failure_reason)
      << "\",";
  oss << "\"context_id\":" << context_id << ",";
  oss << "\"context_generation\":" << context_generation << ",";
  oss << "\"allocation_budget_bytes\":" << allocation_budget_bytes << ",";
  oss << "\"shader_pipeline_created\":" << BoolJson(shader_pipeline_created)
      << ",";
  oss << "\"api_version\":" << api_version << ",";
  oss << "\"driver_version\":" << driver_version << ",";
  oss << "\"vendor_id\":" << vendor_id << ",";
  oss << "\"device_id\":" << device_id << ",";
  oss << "\"device_type\":" << device_type << ",";
  oss << "\"vendor_name\":\"" << JsonEscape(vendor_name) << "\",";
  oss << "\"device_name\":\"" << JsonEscape(device_name) << "\",";
  oss << "\"compute_queue_family_index\":" << compute_queue_family_index << ",";
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
  oss << "\"eye_contact_production_ready\":"
      << BoolJson(eye_contact_production_ready) << ",";
  oss << "\"eye_contact_reason_code\":\""
      << JsonEscape(eye_contact_reason_code) << "\",";
  oss << "\"eye_contact_blocker_code\":\""
      << JsonEscape(eye_contact_blocker_code) << "\",";
  oss << "\"eye_contact_detail\":\"" << JsonEscape(eye_contact_detail)
      << "\",";
  oss << "\"eye_contact_backend_compiled\":"
      << BoolJson(eye_contact_backend_compiled) << ",";
  oss << "\"eye_contact_live_stage_implemented\":"
      << BoolJson(eye_contact_live_stage_implemented) << ",";
  oss << "\"eye_contact_production_adapter_available\":"
      << BoolJson(eye_contact_production_adapter_available) << ",";
  oss << "\"eye_contact_vulkan_inference_provider_available\":"
      << BoolJson(eye_contact_vulkan_inference_provider_available) << ",";
  oss << "\"eye_contact_non_cpu_device_selected\":"
      << BoolJson(eye_contact_non_cpu_device_selected) << ",";
  oss << "\"eye_contact_compute_queue_available\":"
      << BoolJson(eye_contact_compute_queue_available) << ",";
  oss << "\"eye_contact_context_healthy\":"
      << BoolJson(eye_contact_context_healthy) << ",";
  oss << "\"eye_contact_shared_device_imported\":"
      << BoolJson(eye_contact_shared_device_imported) << ",";
  oss << "\"eye_contact_queue_ownership_explicit\":"
      << BoolJson(eye_contact_queue_ownership_explicit) << ",";
  oss << "\"eye_contact_model_pack_selected\":"
      << BoolJson(eye_contact_model_pack_selected) << ",";
  oss << "\"eye_contact_artifact_contract_validated\":"
      << BoolJson(eye_contact_artifact_contract_validated) << ",";
  oss << "\"eye_contact_device_resident_analysis\":"
      << BoolJson(eye_contact_device_resident_analysis) << ",";
  oss << "\"eye_contact_device_resident_tensor_io\":"
      << BoolJson(eye_contact_device_resident_tensor_io) << ",";
  oss << "\"eye_contact_warmup_complete\":"
      << BoolJson(eye_contact_warmup_complete) << ",";
  oss << "\"eye_contact_bounded_reusable_allocations\":"
      << BoolJson(eye_contact_bounded_reusable_allocations) << ",";
  oss << "\"eye_contact_synchronization_contract_validated\":"
      << BoolJson(eye_contact_synchronization_contract_validated) << ",";
  oss << "\"eye_contact_parity_validated\":"
      << BoolJson(eye_contact_parity_validated) << ",";
  oss << "\"eye_contact_selectable_cpu_fallback\":"
      << BoolJson(eye_contact_selectable_cpu_fallback) << ",";
  oss << "\"eye_contact_dispatch_count\":" << eye_contact_dispatch_count
      << ",";
  oss << "\"eye_contact_cpu_readback_count\":"
      << eye_contact_cpu_readback_count << ",";
  oss << "\"eye_contact_cpu_fallback_count\":"
      << eye_contact_cpu_fallback_count << ",";
  oss << "\"video_noise_removal_production_ready\":"
      << BoolJson(video_noise_removal_production_ready) << ",";
  oss << "\"video_noise_removal_reason_code\":\""
      << JsonEscape(video_noise_removal_reason_code) << "\",";
  oss << "\"video_noise_removal_blocker_code\":\""
      << JsonEscape(video_noise_removal_blocker_code) << "\",";
  oss << "\"video_noise_removal_detail\":\""
      << JsonEscape(video_noise_removal_detail) << "\",";
  oss << "\"video_noise_removal_backend_compiled\":"
      << BoolJson(video_noise_removal_backend_compiled) << ",";
  oss << "\"video_noise_removal_live_stage_implemented\":"
      << BoolJson(video_noise_removal_live_stage_implemented) << ",";
  oss << "\"video_noise_removal_production_adapter_available\":"
      << BoolJson(video_noise_removal_production_adapter_available) << ",";
  oss << "\"video_noise_removal_vulkan_inference_provider_available\":"
      << BoolJson(video_noise_removal_vulkan_inference_provider_available)
      << ",";
  oss << "\"video_noise_removal_non_cpu_device_selected\":"
      << BoolJson(video_noise_removal_non_cpu_device_selected) << ",";
  oss << "\"video_noise_removal_compute_queue_available\":"
      << BoolJson(video_noise_removal_compute_queue_available) << ",";
  oss << "\"video_noise_removal_context_healthy\":"
      << BoolJson(video_noise_removal_context_healthy) << ",";
  oss << "\"video_noise_removal_shared_device_imported\":"
      << BoolJson(video_noise_removal_shared_device_imported) << ",";
  oss << "\"video_noise_removal_queue_ownership_explicit\":"
      << BoolJson(video_noise_removal_queue_ownership_explicit) << ",";
  oss << "\"video_noise_removal_model_pack_selected\":"
      << BoolJson(video_noise_removal_model_pack_selected) << ",";
  oss << "\"video_noise_removal_artifact_contract_validated\":"
      << BoolJson(video_noise_removal_artifact_contract_validated) << ",";
  oss << "\"video_noise_removal_fully_device_resident_tensor_io\":"
      << BoolJson(video_noise_removal_fully_device_resident_tensor_io) << ",";
  oss << "\"video_noise_removal_device_resident_preprocess\":"
      << BoolJson(video_noise_removal_device_resident_preprocess) << ",";
  oss << "\"video_noise_removal_device_resident_postprocess\":"
      << BoolJson(video_noise_removal_device_resident_postprocess) << ",";
  oss << "\"video_noise_removal_warmup_complete\":"
      << BoolJson(video_noise_removal_warmup_complete) << ",";
  oss << "\"video_noise_removal_synchronization_contract_validated\":"
      << BoolJson(video_noise_removal_synchronization_contract_validated)
      << ",";
  oss << "\"video_noise_removal_bounded_reusable_allocations\":"
      << BoolJson(video_noise_removal_bounded_reusable_allocations) << ",";
  oss << "\"video_noise_removal_temporal_history_device_resident\":"
      << BoolJson(video_noise_removal_temporal_history_device_resident) << ",";
  oss << "\"video_noise_removal_temporal_history_bounded\":"
      << BoolJson(video_noise_removal_temporal_history_bounded) << ",";
  oss << "\"video_noise_removal_history_reset_on_disable\":"
      << BoolJson(video_noise_removal_history_reset_on_disable) << ",";
  oss << "\"video_noise_removal_history_reset_on_reconfigure\":"
      << BoolJson(video_noise_removal_history_reset_on_reconfigure) << ",";
  oss << "\"video_noise_removal_capture_sequence_discontinuity_reset\":"
      << BoolJson(video_noise_removal_capture_sequence_discontinuity_reset)
      << ",";
  oss << "\"video_noise_removal_parity_validated\":"
      << BoolJson(video_noise_removal_parity_validated) << ",";
  oss << "\"video_noise_removal_selectable_cpu_fallback\":"
      << BoolJson(video_noise_removal_selectable_cpu_fallback) << ",";
  oss << "\"video_noise_removal_dispatch_count\":"
      << video_noise_removal_dispatch_count << ",";
  oss << "\"video_noise_removal_temporal_history_reset_count\":"
      << video_noise_removal_temporal_history_reset_count << ",";
  oss << "\"video_noise_removal_cpu_readback_count\":"
      << video_noise_removal_cpu_readback_count << ",";
  oss << "\"video_noise_removal_cpu_fallback_count\":"
      << video_noise_removal_cpu_fallback_count << ",";
  oss << "\"matting_runtime\":\"" << JsonEscape(matting_runtime) << "\",";
  oss << "\"matting_build_enabled\":" << BoolJson(matting_build_enabled) << ",";
  oss << "\"matting_adapter_available\":" << BoolJson(matting_adapter_available)
      << ",";
  oss << "\"matting_model_pack_selected\":"
      << BoolJson(matting_model_pack_selected) << ",";
  oss << "\"matting_model_contract_validated\":"
      << BoolJson(matting_model_contract_validated) << ",";
  oss << "\"matting_production_ready\":" << BoolJson(matting_production_ready)
      << ",";
  oss << "\"matting_reason_code\":\"" << JsonEscape(matting_reason_code)
      << "\",";
  oss << "\"matting_blocker_code\":\"" << JsonEscape(matting_blocker_code)
      << "\",";
  oss << "\"matting_detail\":\"" << JsonEscape(matting_detail) << "\",";
  oss << "\"matting_runtime_created\":" << BoolJson(matting_runtime_created)
      << ",";
  oss << "\"matting_graph_loaded\":" << BoolJson(matting_graph_loaded) << ",";
  oss << "\"matting_warmup_complete\":" << BoolJson(matting_warmup_complete)
      << ",";
  oss << "\"matting_cpu_layers_used\":" << BoolJson(matting_cpu_layers_used)
      << ",";
  oss << "\"matting_shared_device_imported\":"
      << BoolJson(matting_shared_device_imported) << ",";
  oss << "\"matting_queue_ownership_explicit\":"
      << BoolJson(matting_queue_ownership_explicit) << ",";
  oss << "\"matting_synchronous_completion\":"
      << BoolJson(matting_synchronous_completion) << ",";
  oss << "\"matting_bounded_reusable_allocations\":"
      << BoolJson(matting_bounded_reusable_allocations) << ",";
  oss << "\"matting_persistent_allocation_count\":"
      << matting_persistent_allocation_count << ",";
  oss << "\"matting_dynamic_allocation_count\":"
      << matting_dynamic_allocation_count << ",";
  oss << "\"matting_cpu_readback_count\":" << matting_cpu_readback_count << ",";
  oss << "\"matting_warmup_inference_count\":" << matting_warmup_inference_count
      << ",";
  oss << "\"matting_inference_count\":" << matting_inference_count << ",";
  oss << "\"matting_completion_count\":" << matting_completion_count << ",";
  oss << "\"matting_context_id\":" << matting_context_id << ",";
  oss << "\"matting_context_generation\":" << matting_context_generation << ",";
  oss << "\"input_device_resident\":" << BoolJson(input_device_resident) << ",";
  oss << "\"alpha_device_resident\":" << BoolJson(alpha_device_resident) << ",";
  oss << "\"output_device_resident\":" << BoolJson(output_device_resident)
      << ",";
  oss << "\"device_residency_mode\":\"" << JsonEscape(device_residency_mode)
      << "\",";
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

VulkanDevice::VulkanDevice()
    : loader_(std::make_shared<VulkanLoader>()),
      context_id_(NextContextId().fetch_add(1, std::memory_order_relaxed)) {}

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
  return Initialize(selection, VulkanDeviceConfig{}, error_out);
}

bool VulkanDevice::Initialize(const VulkanDeviceSelection &selection,
                              const VulkanDeviceConfig &config,
                              std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (initialized_)
    return true;

  diagnostics_ = OpenVulkanDiagnostics{};
  identity_ = VulkanDeviceIdentity{};
  std::string e;
  selection_ = selection;
  config_ = config;
  diagnostics_.device_selection_source = selection_.source;
  diagnostics_.device_selection_request = selection_.request;

  if (config_.fence_timeout_ns == 0) {
    e = "Vulkan fence timeout must be non-zero.";
    diagnostics_.error = e;
    diagnostics_.fallback_reason = "vulkan_context_config_invalid";
    if (error_out)
      *error_out = e;
    return false;
  }

  if (!loader_->Load(&e)) {
    diagnostics_.error = e;
    diagnostics_.fallback_reason = "vulkan_runtime_not_found";
    if (error_out)
      *error_out = e;
    return false;
  }
  diagnostics_.runtime_library_found = true;
  diagnostics_.runtime_library_path = loader_->LibraryPath().string();

  const VulkanFunctions &lf = loader_->f();
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
    DestroyUnownedHandles();
    return false;
  }

  const VkDeviceSize derived_budget =
      DerivedAllocationBudget(memory_properties_);
  if (derived_budget == 0) {
    e = "Selected Vulkan device reported no allocatable memory heaps.";
    diagnostics_.error = e;
    diagnostics_.fallback_reason = "vulkan_memory_budget_unavailable";
    if (error_out)
      *error_out = e;
    DestroyUnownedHandles();
    return false;
  }

  const VkDeviceSize budget =
      config_.allocation_budget_bytes == 0
          ? derived_budget
          : std::min(config_.allocation_budget_bytes, derived_budget);
  context_identity_ = {context_id_, ++next_generation_};
  context_ = std::make_shared<detail::VulkanContextState>();
  context_->loader = loader_;
  context_->identity = context_identity_;
  context_->instance = instance_;
  context_->physical_device = physical_device_;
  context_->device = device_;
  context_->queue = queue_;
  context_->command_pool = command_pool_;
  context_->fence = fence_;
  context_->memory_properties = memory_properties_;
  context_->allocation_budget_bytes = budget;
  context_->fence_timeout_ns = config_.fence_timeout_ns;
  context_->health.health = VulkanContextHealth::healthy;
  context_->health.reason_code = "vulkan_context_healthy";
  context_->allocation_stats.budget_bytes = budget;

  diagnostics_.context_created = true;
  diagnostics_.context_healthy = true;
  diagnostics_.context_health = "healthy";
  diagnostics_.context_failure_reason.clear();
  diagnostics_.context_id = context_identity_.context_id;
  diagnostics_.context_generation = context_identity_.generation;
  diagnostics_.allocation_budget_bytes = budget;
  diagnostics_.production_hardware_ready =
      diagnostics_.runtime_library_found &&
      diagnostics_.physical_device_found &&
      diagnostics_.non_cpu_device_selected &&
      diagnostics_.compute_queue_available &&
      diagnostics_.logical_device_created && diagnostics_.context_healthy;

  initialized_ = true;
  diagnostics_.ok = true;
  return true;
}

void VulkanDevice::Shutdown() noexcept {
  if (context_) {
    std::lock_guard<std::mutex> submit_lock(context_->submission_mutex);
    context_->MarkShutdown();
    context_.reset();
  } else {
    DestroyUnownedHandles();
  }
  fence_ = nullptr;
  command_pool_ = nullptr;
  device_ = nullptr;
  queue_ = nullptr;
  physical_device_ = nullptr;
  instance_ = nullptr;
  context_identity_ = {};
  initialized_ = false;
}

void VulkanDevice::DestroyUnownedHandles() noexcept {
  const auto &vf = loader_->f();
  if (device_) {
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
  if (instance_ && vf.vkDestroyInstance)
    vf.vkDestroyInstance(instance_, nullptr);
  instance_ = nullptr;
}

bool VulkanDevice::LoadInstanceFunctions(std::string *error_out) {
  VulkanFunctions &vf = loader_->f();
  return LoadInstanceProc(vf, instance_, "vkDestroyInstance",
                          &vf.vkDestroyInstance, error_out) &&
         LoadInstanceProc(vf, instance_, "vkEnumeratePhysicalDevices",
                          &vf.vkEnumeratePhysicalDevices, error_out) &&
         LoadInstanceProc(vf, instance_, "vkGetPhysicalDeviceProperties",
                          &vf.vkGetPhysicalDeviceProperties, error_out) &&
         LoadInstanceProc(
             vf, instance_, "vkGetPhysicalDeviceQueueFamilyProperties",
             &vf.vkGetPhysicalDeviceQueueFamilyProperties, error_out) &&
         LoadInstanceProc(vf, instance_, "vkGetPhysicalDeviceMemoryProperties",
                          &vf.vkGetPhysicalDeviceMemoryProperties, error_out) &&
         LoadInstanceProc(vf, instance_, "vkCreateDevice", &vf.vkCreateDevice,
                          error_out) &&
         LoadInstanceProc(vf, instance_, "vkGetDeviceProcAddr",
                          &vf.vkGetDeviceProcAddr, error_out);
}

bool VulkanDevice::PickPhysicalDevice(std::string *error_out) {
  const auto &vf = loader_->f();
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
  diagnostics_.non_cpu_device_selected = !diagnostics_.cpu_device_selected;
  if (diagnostics_.cpu_device_selected) {
    diagnostics_.warnings.push_back(
        "A CPU Vulkan device was selected; this is a software fallback, not "
        "hardware GPU acceleration.");
  }
  return true;
}

bool VulkanDevice::CreateLogicalDevice(std::string *error_out) {
  const auto &vf = loader_->f();
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
  VulkanFunctions &vf = loader_->f();
  return LoadDeviceProc(vf, device_, "vkDestroyDevice", &vf.vkDestroyDevice,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkGetDeviceQueue", &vf.vkGetDeviceQueue,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkDeviceWaitIdle", &vf.vkDeviceWaitIdle,
                        error_out) &&
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
         LoadDeviceProc(vf, device_, "vkAllocateMemory", &vf.vkAllocateMemory,
                        error_out) &&
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
         LoadDeviceProc(vf, device_, "vkDestroyPipeline", &vf.vkDestroyPipeline,
                        error_out) &&
         LoadDeviceProc(vf, device_, "vkBeginCommandBuffer",
                        &vf.vkBeginCommandBuffer, error_out) &&
         LoadDeviceProc(vf, device_, "vkEndCommandBuffer",
                        &vf.vkEndCommandBuffer, error_out) &&
         LoadDeviceProc(vf, device_, "vkResetCommandBuffer",
                        &vf.vkResetCommandBuffer, error_out) &&
         LoadDeviceProc(vf, device_, "vkCmdBindPipeline", &vf.vkCmdBindPipeline,
                        error_out) &&
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
  const auto &vf = loader_->f();
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
    oss << "No Vulkan memory type matched type_bits=0x" << std::hex << type_bits
        << " required=0x" << required_flags << " preferred=0x"
        << preferred_flags << ".";
    *error_out = oss.str();
  }
  return false;
}

bool VulkanDevice::SubmitAndWait(VkCommandBuffer command_buffer,
                                 std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (!initialized_ || !context_) {
    if (error_out)
      *error_out = "[vulkan_context_uninitialized] Vulkan device is not "
                   "initialized.";
    return false;
  }
  if (!command_buffer) {
    if (error_out)
      *error_out = "[vulkan_command_buffer_invalid] Cannot submit a null "
                   "Vulkan command buffer.";
    return false;
  }

  std::lock_guard<std::mutex> submit_lock(context_->submission_mutex);
  const auto before = context_->HealthSnapshot();
  if (before.poisoned || before.health != VulkanContextHealth::healthy) {
    if (error_out) {
      *error_out = "[" + before.reason_code +
                   "] Vulkan submission rejected because the context is "
                   "latched unhealthy.";
    }
    return false;
  }

  {
    std::lock_guard<std::mutex> health_lock(context_->health_mutex);
    ++context_->health.submitted_serial;
  }

  if (context_->injected_submission_result) {
    const auto injected = *context_->injected_submission_result;
    context_->injected_submission_result.reset();
    const char *operation = "injected Vulkan submission";
    switch (injected.first) {
    case VulkanSubmissionPhase::reset_fence:
      operation = "vkResetFences";
      break;
    case VulkanSubmissionPhase::queue_submit:
      operation = "vkQueueSubmit";
      break;
    case VulkanSubmissionPhase::wait_for_fence:
      operation = "vkWaitForFences";
      break;
    }
    if (!context_->RecordDriverFailure(injected.second, operation,
                                       /*submission_failure=*/true,
                                       error_out)) {
      diagnostics_ = DiagnosticsSnapshot();
      return false;
    }
    std::lock_guard<std::mutex> health_lock(context_->health_mutex);
    context_->health.completed_serial = context_->health.submitted_serial;
    return true;
  }

  const auto &vf = loader_->f();
  VkResult result = vf.vkResetFences(device_, 1, &fence_);
  if (!context_->RecordDriverFailure(result, "vkResetFences",
                                     /*submission_failure=*/true, error_out)) {
    diagnostics_ = DiagnosticsSnapshot();
    return false;
  }

  VkSubmitInfo submit{};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command_buffer;
  result = vf.vkQueueSubmit(queue_, 1, &submit, fence_);
  if (!context_->RecordDriverFailure(result, "vkQueueSubmit",
                                     /*submission_failure=*/true, error_out)) {
    diagnostics_ = DiagnosticsSnapshot();
    return false;
  }

  result =
      vf.vkWaitForFences(device_, 1, &fence_, 1, context_->fence_timeout_ns);
  if (!context_->RecordDriverFailure(result, "vkWaitForFences",
                                     /*submission_failure=*/true, error_out)) {
    diagnostics_ = DiagnosticsSnapshot();
    return false;
  }
  {
    std::lock_guard<std::mutex> health_lock(context_->health_mutex);
    context_->health.completed_serial = context_->health.submitted_serial;
  }
  return true;
}

bool VulkanDevice::CheckDriverResult(VkResult result,
                                     std::string_view operation,
                                     bool poison_on_failure,
                                     std::string *error_out) const {
  if (!context_) {
    if (error_out)
      *error_out = "[vulkan_context_uninitialized] Cannot classify a driver "
                   "result without a Vulkan context.";
    return false;
  }
  return context_->RecordDriverFailure(result, operation, poison_on_failure,
                                       error_out);
}

bool VulkanDevice::RecordBufferBarrier(
    VkCommandBuffer command_buffer, VkBuffer buffer, VkDeviceSize size,
    const VulkanContextIdentity &resource_context, VulkanBufferAccess source,
    VulkanBufferAccess destination, std::string *error_out) const {
  if (error_out)
    error_out->clear();
  if (!initialized_ || !context_ || !command_buffer || !buffer || size == 0) {
    if (error_out)
      *error_out = "[vulkan_barrier_invalid] Vulkan buffer barrier has an "
                   "invalid context, command buffer, buffer, or size.";
    return false;
  }
  if (!OwnsContext(resource_context)) {
    if (error_out)
      *error_out = "[vulkan_foreign_context] Vulkan buffer barrier rejected "
                   "a resource from a foreign context/generation.";
    return false;
  }
  const auto snapshot = context_->HealthSnapshot();
  if (!context_->Healthy()) {
    if (error_out)
      *error_out = "[" + snapshot.reason_code +
                   "] Vulkan buffer barrier rejected an unhealthy context.";
    return false;
  }

  const VulkanBufferBarrierSpec spec = VulkanBufferBarrier(source, destination);
  VkBufferMemoryBarrier barrier{};
  barrier.srcAccessMask = spec.src_access_mask;
  barrier.dstAccessMask = spec.dst_access_mask;
  barrier.buffer = buffer;
  barrier.size = size;
  loader_->f().vkCmdPipelineBarrier(command_buffer, spec.src_stage_mask,
                                    spec.dst_stage_mask, 0, 0, nullptr, 1,
                                    &barrier, 0, nullptr);
  return true;
}

bool VulkanDevice::FlushMemory(VkDeviceMemory memory, VkDeviceSize offset,
                               VkDeviceSize size,
                               std::string *error_out) const {
  if (!context_ || !context_->Healthy()) {
    const auto snapshot = health();
    if (error_out)
      *error_out = "[" + snapshot.reason_code +
                   "] Cannot flush memory for an unhealthy Vulkan context.";
    return false;
  }
  VkMappedMemoryRange range{};
  range.memory = memory;
  range.offset = offset;
  range.size = size;
  const VkResult result =
      loader_->f().vkFlushMappedMemoryRanges(device_, 1, &range);
  return context_->RecordDriverFailure(result, "vkFlushMappedMemoryRanges",
                                       /*submission_failure=*/false, error_out);
}

bool VulkanDevice::InvalidateMemory(VkDeviceMemory memory, VkDeviceSize offset,
                                    VkDeviceSize size,
                                    std::string *error_out) const {
  if (!context_ || !context_->Healthy()) {
    const auto snapshot = health();
    if (error_out)
      *error_out = "[" + snapshot.reason_code +
                   "] Cannot invalidate memory for an unhealthy Vulkan "
                   "context.";
    return false;
  }
  VkMappedMemoryRange range{};
  range.memory = memory;
  range.offset = offset;
  range.size = size;
  const VkResult result =
      loader_->f().vkInvalidateMappedMemoryRanges(device_, 1, &range);
  return context_->RecordDriverFailure(result, "vkInvalidateMappedMemoryRanges",
                                       /*submission_failure=*/false, error_out);
}

bool VulkanDevice::OwnsContext(const VulkanContextIdentity &identity) const {
  return initialized_ && context_ && identity.Valid() &&
         identity == context_identity_;
}

VulkanHealthSnapshot VulkanDevice::health() const {
  if (!context_)
    return {};
  return context_->HealthSnapshot();
}

VulkanAllocationStats VulkanDevice::allocation_stats() const {
  if (!context_)
    return {};
  return context_->AllocationStats();
}

bool VulkanDevice::SafeToDestroyResources() const {
  return context_ && context_->SafeToDestroyChildren();
}

OpenVulkanDiagnostics VulkanDevice::DiagnosticsSnapshot() const {
  OpenVulkanDiagnostics snapshot = diagnostics_;
  if (!context_)
    return snapshot;
  const auto health_snapshot = context_->HealthSnapshot();
  const auto allocations = context_->AllocationStats();
  snapshot.context_created = true;
  snapshot.context_healthy =
      health_snapshot.health == VulkanContextHealth::healthy &&
      !health_snapshot.poisoned;
  snapshot.context_health = VulkanContextHealthName(health_snapshot.health);
  snapshot.context_failure_reason = health_snapshot.reason_code;
  snapshot.context_id = context_->identity.context_id;
  snapshot.context_generation = context_->identity.generation;
  snapshot.allocation_budget_bytes = allocations.budget_bytes;
  snapshot.production_hardware_ready =
      snapshot.runtime_library_found && snapshot.physical_device_found &&
      snapshot.non_cpu_device_selected && snapshot.compute_queue_available &&
      snapshot.logical_device_created && snapshot.context_healthy;
  snapshot.ok = snapshot.ok && snapshot.context_healthy;
  if (health_snapshot.poisoned && snapshot.error.empty())
    snapshot.error = health_snapshot.message;
  return snapshot;
}

void VulkanDevice::InjectNextSubmissionResultForTesting(
    VulkanSubmissionPhase phase, VkResult result) {
  if (!context_)
    return;
  std::lock_guard<std::mutex> submit_lock(context_->submission_mutex);
  context_->injected_submission_result = std::make_pair(phase, result);
}

VulkanCommandBatch::~VulkanCommandBatch() { Shutdown(); }

bool VulkanCommandBatch::Initialize(VulkanDevice *device,
                                    std::size_t stage_capacity,
                                    std::string *error_out) {
  if (error_out)
    error_out->clear();
  std::lock_guard<std::mutex> lock(mutex_);
  if (context_ && command_buffer_ && stage_capacity_ == stage_capacity)
    return true;
  if (context_) {
    if (error_out)
      *error_out = "[vulkan_batch_reconfigure_required] Shut down the Vulkan "
                   "command batch before changing its capacity.";
    return false;
  }
  if (!device || !device->Initialized() || !device->context_ ||
      stage_capacity == 0) {
    if (error_out)
      *error_out = "[vulkan_batch_config_invalid] Vulkan command batch "
                   "requires an initialized device and non-zero capacity.";
    return false;
  }

  const auto context = device->context_;
  if (!context->Healthy()) {
    if (error_out)
      *error_out = "[vulkan_context_unhealthy] Cannot initialize a command "
                   "batch on an unhealthy context.";
    return false;
  }
  VkCommandBufferAllocateInfo allocate{};
  allocate.commandPool = context->command_pool;
  allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate.commandBufferCount = 1;
  VkCommandBuffer command_buffer = nullptr;
  VkResult result = VK_SUCCESS;
  {
    std::lock_guard<std::mutex> pool_lock(context->command_pool_mutex);
    result = context->f().vkAllocateCommandBuffers(context->device, &allocate,
                                                   &command_buffer);
  }
  if (!context->RecordDriverFailure(result, "vkAllocateCommandBuffers",
                                    /*submission_failure=*/true, error_out)) {
    return false;
  }
  context_ = context;
  context_identity_ = context->identity;
  command_buffer_ = command_buffer;
  stage_capacity_ = stage_capacity;
  return true;
}

void VulkanCommandBatch::Shutdown() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (context_ && context_->SafeToDestroyChildren() && command_buffer_ &&
      context_->f().vkFreeCommandBuffers) {
    std::lock_guard<std::mutex> pool_lock(context_->command_pool_mutex);
    context_->f().vkFreeCommandBuffers(context_->device, context_->command_pool,
                                       1, &command_buffer_);
  }
  command_buffer_ = nullptr;
  context_.reset();
  context_identity_ = {};
  stage_capacity_ = 0;
  recorded_stage_count_ = 0;
  recording_ = false;
  recording_owner_ = {};
}

bool VulkanCommandBatch::Begin(std::string *error_out) {
  if (error_out)
    error_out->clear();
  std::lock_guard<std::mutex> lock(mutex_);
  if (!context_ || !command_buffer_ || stage_capacity_ == 0) {
    if (error_out)
      *error_out = "[vulkan_batch_uninitialized] Vulkan command batch is not "
                   "initialized.";
    return false;
  }
  if (recording_) {
    if (error_out)
      *error_out = "[vulkan_batch_already_recording] Vulkan command batch is "
                   "already in use.";
    return false;
  }
  const auto health = context_->HealthSnapshot();
  if (!context_->Healthy()) {
    if (error_out)
      *error_out = "[" + health.reason_code +
                   "] Vulkan command batch rejected an unhealthy context.";
    return false;
  }

  VkResult result = context_->f().vkResetCommandBuffer(command_buffer_, 0);
  if (!context_->RecordDriverFailure(result, "vkResetCommandBuffer",
                                     /*submission_failure=*/true, error_out)) {
    return false;
  }
  VkCommandBufferBeginInfo begin{};
  result = context_->f().vkBeginCommandBuffer(command_buffer_, &begin);
  if (!context_->RecordDriverFailure(result, "vkBeginCommandBuffer",
                                     /*submission_failure=*/true, error_out)) {
    return false;
  }
  recorded_stage_count_ = 0;
  recording_owner_ = std::this_thread::get_id();
  recording_ = true;
  return true;
}

bool VulkanCommandBatch::CheckRecordingOwner(std::string *error_out) const {
  if (!recording_) {
    if (error_out)
      *error_out = "[vulkan_batch_not_recording] Vulkan command batch has no "
                   "active recording scope.";
    return false;
  }
  if (recording_owner_ != std::this_thread::get_id()) {
    if (error_out)
      *error_out = "[vulkan_batch_wrong_thread] Vulkan command batch must be "
                   "recorded and completed by its beginning thread.";
    return false;
  }
  return true;
}

bool VulkanCommandBatch::RecordStage(std::string_view label,
                                     std::string *error_out) {
  if (error_out)
    error_out->clear();
  std::lock_guard<std::mutex> lock(mutex_);
  if (!CheckRecordingOwner(error_out))
    return false;
  if (label.empty()) {
    if (error_out)
      *error_out = "[vulkan_batch_stage_invalid] Vulkan batch stage label is "
                   "empty.";
    return false;
  }
  if (recorded_stage_count_ >= stage_capacity_) {
    if (error_out)
      *error_out = "[vulkan_batch_capacity_exceeded] Vulkan batch stage "
                   "capacity exceeded.";
    return false;
  }
  ++recorded_stage_count_;
  return true;
}

bool VulkanCommandBatch::RecordBufferBarrier(
    VkBuffer buffer, VkDeviceSize size,
    const VulkanContextIdentity &resource_context, VulkanBufferAccess source,
    VulkanBufferAccess destination, std::string *error_out) {
  if (error_out)
    error_out->clear();
  std::lock_guard<std::mutex> lock(mutex_);
  if (!CheckRecordingOwner(error_out))
    return false;
  const auto health = context_->HealthSnapshot();
  if (!context_->Healthy()) {
    if (error_out)
      *error_out = "[" + health.reason_code +
                   "] Vulkan batch barrier rejected an unhealthy context.";
    return false;
  }
  if (!buffer || size == 0 || resource_context != context_identity_) {
    if (error_out)
      *error_out = "[vulkan_foreign_context] Vulkan command batch barrier "
                   "rejected an invalid or foreign-context resource.";
    return false;
  }
  const VulkanBufferBarrierSpec spec = VulkanBufferBarrier(source, destination);
  VkBufferMemoryBarrier barrier{};
  barrier.srcAccessMask = spec.src_access_mask;
  barrier.dstAccessMask = spec.dst_access_mask;
  barrier.buffer = buffer;
  barrier.size = size;
  context_->f().vkCmdPipelineBarrier(command_buffer_, spec.src_stage_mask,
                                     spec.dst_stage_mask, 0, 0, nullptr, 1,
                                     &barrier, 0, nullptr);
  return true;
}

bool VulkanCommandBatch::Complete(std::string *error_out) {
  if (error_out)
    error_out->clear();
  std::lock_guard<std::mutex> lock(mutex_);
  if (!CheckRecordingOwner(error_out))
    return false;
  if (recorded_stage_count_ == 0) {
    if (error_out)
      *error_out = "[vulkan_batch_empty] Vulkan command batch contains no "
                   "recorded stages.";
    return false;
  }

  VkResult result = context_->f().vkEndCommandBuffer(command_buffer_);
  if (!context_->RecordDriverFailure(result, "vkEndCommandBuffer",
                                     /*submission_failure=*/true, error_out)) {
    recording_ = false;
    recording_owner_ = {};
    return false;
  }

  std::lock_guard<std::mutex> submit_lock(context_->submission_mutex);
  const auto before = context_->HealthSnapshot();
  if (!context_->Healthy()) {
    if (error_out)
      *error_out = "[" + before.reason_code +
                   "] Vulkan batch completion rejected an unhealthy context.";
    recording_ = false;
    recording_owner_ = {};
    return false;
  }
  {
    std::lock_guard<std::mutex> health_lock(context_->health_mutex);
    ++context_->health.submitted_serial;
  }

  if (context_->injected_submission_result) {
    const auto injected = *context_->injected_submission_result;
    context_->injected_submission_result.reset();
    const char *operation =
        injected.first == VulkanSubmissionPhase::reset_fence ? "vkResetFences"
        : injected.first == VulkanSubmissionPhase::queue_submit
            ? "vkQueueSubmit"
            : "vkWaitForFences";
    result = injected.second;
    if (!context_->RecordDriverFailure(result, operation,
                                       /*submission_failure=*/true,
                                       error_out)) {
      recording_ = false;
      recording_owner_ = {};
      return false;
    }
  } else {
    result = context_->f().vkResetFences(context_->device, 1, &context_->fence);
    if (!context_->RecordDriverFailure(result, "vkResetFences",
                                       /*submission_failure=*/true,
                                       error_out)) {
      recording_ = false;
      recording_owner_ = {};
      return false;
    }
    VkSubmitInfo submit{};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer_;
    result = context_->f().vkQueueSubmit(context_->queue, 1, &submit,
                                         context_->fence);
    if (!context_->RecordDriverFailure(result, "vkQueueSubmit",
                                       /*submission_failure=*/true,
                                       error_out)) {
      recording_ = false;
      recording_owner_ = {};
      return false;
    }
    result = context_->f().vkWaitForFences(
        context_->device, 1, &context_->fence, 1, context_->fence_timeout_ns);
    if (!context_->RecordDriverFailure(result, "vkWaitForFences",
                                       /*submission_failure=*/true,
                                       error_out)) {
      recording_ = false;
      recording_owner_ = {};
      return false;
    }
  }

  {
    std::lock_guard<std::mutex> health_lock(context_->health_mutex);
    context_->health.completed_serial = context_->health.submitted_serial;
  }
  ++completion_count_;
  recording_ = false;
  recording_owner_ = {};
  return true;
}

void VulkanCommandBatch::Abort() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (recording_ && recording_owner_ == std::this_thread::get_id()) {
    recording_ = false;
    recording_owner_ = {};
    recorded_stage_count_ = 0;
  }
}

VkCommandBuffer VulkanCommandBatch::command_buffer() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!recording_ || recording_owner_ != std::this_thread::get_id())
    return nullptr;
  return command_buffer_;
}

bool VulkanCommandBatch::recording() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return recording_;
}

std::size_t VulkanCommandBatch::recorded_stage_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return recorded_stage_count_;
}

std::uint64_t VulkanCommandBatch::completion_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return completion_count_;
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
