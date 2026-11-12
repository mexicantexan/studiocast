#include "core/open_video/ncnn_vulkan_context.h"
#include "core/open_video/ncnn_vulkan_context_internal.h"
#if STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
#include <cstdlib>
#include <mutex>
namespace {
struct InstanceOwner {
  ~InstanceOwner() { ncnn::studiocast_release_gpu_instance(); }
};
std::shared_ptr<void> AcquireInstance() {
  if (ncnn::studiocast_acquire_gpu_instance() != 0)
    return {};
  return std::make_shared<InstanceOwner>();
}
template <class To, class From> To Handle(From value) {
  return reinterpret_cast<To>(value);
}
} // namespace
#endif
namespace studiocast::open_vulkan {
bool InitializeProductionVulkanMattingDevice(
    vulkan::VulkanDevice *device,
    const vulkan::VulkanDeviceSelection &selection,
    const vulkan::VulkanDeviceConfig &config, std::string *error_out) {
#if STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING
  if (!device || device->Initialized()) {
    if (error_out)
      *error_out = "Matting adoption requires a fresh Vulkan device.";
    return false;
  }
  // An explicit StudioCast provider must load successfully and expose the same
  // entry point ncnn actually retained. Never dispatch foreign instance handles
  // through an independently guessed library or silently ignore an override.
  studiocast::util::DynLib requested_library;
  vulkan::PFN_vkGetInstanceProcAddr requested_entry = nullptr;
  if (const char *requested = std::getenv("STUDIOCAST_VULKAN_LIBRARY");
      requested && *requested) {
    if (!requested_library.Open(
            requested, studiocast::util::DynLib::Scope::Local, error_out) ||
        !requested_library.GetSymbol("vkGetInstanceProcAddr", &requested_entry,
                                     error_out))
      return false;
  }
  auto owner = std::make_shared<detail::NcnnVulkanContext>();
  owner->instance = AcquireInstance();
  if (!owner->instance) {
    if (error_out)
      *error_out = "ncnn Vulkan instance unavailable.";
    return false;
  }
  vulkan::VkPhysicalDevice selected = nullptr;
  auto instance = Handle<vulkan::VkInstance>(ncnn::get_gpu_instance());
  auto entry_point = reinterpret_cast<vulkan::PFN_vkGetInstanceProcAddr>(
      ncnn::vkGetInstanceProcAddr);
  if (requested_entry && requested_entry != entry_point) {
    if (error_out)
      *error_out = "[open_vulkan_matting_driver_override_conflict] "
                   "STUDIOCAST_VULKAN_LIBRARY differs from the active ncnn "
                   "Vulkan provider; use the same provider for both runtimes.";
    return false;
  }
  if (!vulkan::VulkanDevice::ChooseAdoptionPhysicalDevice(
          instance, selection, entry_point, owner->instance, &selected,
          error_out))
    return false;
  int index = -1;
  for (int i = 0; i < ncnn::get_gpu_count(); ++i)
    if (Handle<vulkan::VkPhysicalDevice>(
            ncnn::get_gpu_info(i).physical_device()) == selected)
      index = i;
  if (index >= 0 && ncnn::get_gpu_info(index).type() == 3) {
    if (error_out)
      *error_out = "Production ncnn matting requires a non-CPU Vulkan device.";
    return false;
  }
  if (index < 0) {
    if (error_out)
      *error_out = "Selected Vulkan device is unavailable to ncnn.";
    return false;
  }
  owner->device = std::make_unique<ncnn::VulkanDevice>(index, true);
  if (!owner->device->is_valid()) {
    if (error_out)
      *error_out = "ncnn Vulkan device creation failed.";
    return false;
  }
  owner->queue = owner->device->acquire_queue(
      owner->device->info.compute_queue_family_index());
  vulkan::VulkanDeviceAdoption adopted;
  adopted.instance = instance;
  adopted.entry_point = entry_point;
  adopted.physical_device = selected;
  adopted.device = Handle<vulkan::VkDevice>(owner->device->vkdevice());
  adopted.queue = Handle<vulkan::VkQueue>(owner->queue);
  adopted.queue_family_index = owner->device->info.compute_queue_family_index();
  adopted.owner = owner;
  adopted.owner_kind = detail::kNcnnOwnerKind;
  if (!device->InitializeAdopted(selection, config, std::move(adopted),
                                 error_out))
    return false;
  VkCommandBuffer dummy = VK_NULL_HANDLE;
  if (owner->device->studiocast_initialize_dummy(&dummy) != 0 ||
      !device->SubmitAndWait(Handle<vulkan::VkCommandBuffer>(dummy), error_out))
    return false;
  const auto buffer = owner->device->get_dummy_buffer();
  if (!buffer.data || buffer.data->mapped_ptr ||
      !owner->device->is_device_local(buffer.data->memory_type_index)) {
    if (error_out)
      *error_out = "ncnn dummy allocation is not unmapped device-local memory.";
    return false;
  }
  return true;
#else
  (void)device;
  (void)selection;
  (void)config;
  if (error_out)
    *error_out = "Production ncnn Vulkan matting is disabled in this build.";
  return false;
#endif
}
bool InitializeProductionVulkanMattingDevice(vulkan::VulkanDevice *device,
                                             std::string *error_out) {
  vulkan::VulkanDeviceSelection selection;
  return vulkan::VulkanDevice::ResolveSelection(&selection, error_out) &&
         InitializeProductionVulkanMattingDevice(device, selection, {},
                                                 error_out);
}
} // namespace studiocast::open_vulkan
