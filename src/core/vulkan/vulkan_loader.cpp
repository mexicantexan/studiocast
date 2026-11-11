#include "core/vulkan/vulkan_loader.h"

#include <cstdlib>
#include <sstream>
#include <string_view>
#include <vector>

namespace studiocast::vulkan {

std::string VkResultName(VkResult result) {
  switch (result) {
  case VK_SUCCESS:
    return "VK_SUCCESS";
  case VK_TIMEOUT:
    return "VK_TIMEOUT";
  case VK_ERROR_OUT_OF_HOST_MEMORY:
    return "VK_ERROR_OUT_OF_HOST_MEMORY";
  case VK_ERROR_OUT_OF_DEVICE_MEMORY:
    return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
  case VK_ERROR_INITIALIZATION_FAILED:
    return "VK_ERROR_INITIALIZATION_FAILED";
  case VK_ERROR_DEVICE_LOST:
    return "VK_ERROR_DEVICE_LOST";
  case VK_ERROR_MEMORY_MAP_FAILED:
    return "VK_ERROR_MEMORY_MAP_FAILED";
  case VK_ERROR_LAYER_NOT_PRESENT:
    return "VK_ERROR_LAYER_NOT_PRESENT";
  case VK_ERROR_EXTENSION_NOT_PRESENT:
    return "VK_ERROR_EXTENSION_NOT_PRESENT";
  case VK_ERROR_FEATURE_NOT_PRESENT:
    return "VK_ERROR_FEATURE_NOT_PRESENT";
  case VK_ERROR_INCOMPATIBLE_DRIVER:
    return "VK_ERROR_INCOMPATIBLE_DRIVER";
  default:
    return "VkResult(" + std::to_string(result) + ")";
  }
}

namespace {

std::vector<std::filesystem::path> VulkanLibraryCandidates() {
  std::vector<std::filesystem::path> out;
  const char *override_path = std::getenv("STUDIOCAST_VULKAN_LIBRARY");
  if (override_path && *override_path)
    out.emplace_back(override_path);
  out.emplace_back("libvulkan.so.1");
  out.emplace_back("libvulkan.so");
  return out;
}

} // namespace

template <typename T>
bool VulkanLoader::LoadGlobalSymbol(const char *name, T *out,
                                    std::string *error_out) {
  return library_.GetSymbol(name, out, error_out);
}

bool VulkanLoader::LoadGlobalProc(const char *name, PFN_vkVoidFunction *out,
                                  std::string *error_out) {
  if (out)
    *out = nullptr;
  if (!f_.vkGetInstanceProcAddr) {
    if (error_out)
      *error_out = "vkGetInstanceProcAddr is not loaded.";
    return false;
  }
  PFN_vkVoidFunction proc = f_.vkGetInstanceProcAddr(nullptr, name);
  if (!proc) {
    if (error_out)
      *error_out = std::string("Vulkan global proc not found: ") + name;
    return false;
  }
  if (out)
    *out = proc;
  return true;
}

bool VulkanLoader::Load(std::string *error_out) {
  if (error_out)
    error_out->clear();
  if (library_.IsOpen())
    return true;

  std::string last_open_error;
  for (const auto &candidate : VulkanLibraryCandidates()) {
    std::string e;
    if (library_.Open(candidate, studiocast::util::DynLib::Scope::Local, &e)) {
      break;
    }
    if (!last_open_error.empty())
      last_open_error += " | ";
    last_open_error += e;
  }

  if (!library_.IsOpen()) {
    if (error_out) {
      *error_out = last_open_error.empty()
                       ? "Vulkan runtime library not found."
                       : last_open_error;
    }
    return false;
  }

  std::string e;
  if (!LoadGlobalSymbol("vkGetInstanceProcAddr", &f_.vkGetInstanceProcAddr,
                        &e)) {
    if (error_out)
      *error_out = e;
    library_.Close();
    f_ = VulkanFunctions{};
    return false;
  }

  PFN_vkVoidFunction proc = nullptr;
  if (!LoadGlobalProc("vkCreateInstance", &proc, &e)) {
    if (error_out)
      *error_out = e;
    library_.Close();
    f_ = VulkanFunctions{};
    return false;
  }
  f_.vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(proc);

  if (LoadGlobalProc("vkEnumerateInstanceVersion", &proc, nullptr))
    f_.vkEnumerateInstanceVersion =
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(proc);

  return true;
}

} // namespace studiocast::vulkan
