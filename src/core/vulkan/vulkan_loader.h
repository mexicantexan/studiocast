#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "core/util/dynlib.h"

namespace studiocast::vulkan {

using VkFlags = std::uint32_t;
using VkBool32 = std::uint32_t;
using VkDeviceSize = std::uint64_t;
using VkResult = std::int32_t;

struct VkAllocationCallbacks;
struct VkInstance_T;
struct VkPhysicalDevice_T;
struct VkDevice_T;
struct VkQueue_T;
struct VkCommandPool_T;
struct VkCommandBuffer_T;
struct VkFence_T;
struct VkDeviceMemory_T;
struct VkBuffer_T;
struct VkShaderModule_T;
struct VkDescriptorSetLayout_T;
struct VkDescriptorPool_T;
struct VkDescriptorSet_T;
struct VkPipelineLayout_T;
struct VkPipeline_T;
struct VkPipelineCache_T;
struct VkSemaphore_T;
struct VkSampler_T;
struct VkDescriptorUpdateTemplate_T;
struct VkPipelineLayout_T;

using VkInstance = VkInstance_T *;
using VkPhysicalDevice = VkPhysicalDevice_T *;
using VkDevice = VkDevice_T *;
using VkQueue = VkQueue_T *;
using VkCommandPool = VkCommandPool_T *;
using VkCommandBuffer = VkCommandBuffer_T *;
using VkFence = VkFence_T *;
using VkDeviceMemory = VkDeviceMemory_T *;
using VkBuffer = VkBuffer_T *;
using VkShaderModule = VkShaderModule_T *;
using VkDescriptorSetLayout = VkDescriptorSetLayout_T *;
using VkDescriptorPool = VkDescriptorPool_T *;
using VkDescriptorSet = VkDescriptorSet_T *;
using VkPipelineLayout = VkPipelineLayout_T *;
using VkPipeline = VkPipeline_T *;
using VkPipelineCache = VkPipelineCache_T *;
using VkSemaphore = VkSemaphore_T *;
using VkSampler = VkSampler_T *;
using VkDescriptorUpdateTemplate = VkDescriptorUpdateTemplate_T *;

constexpr VkResult VK_SUCCESS = 0;
constexpr VkResult VK_TIMEOUT = 2;
constexpr VkResult VK_ERROR_OUT_OF_HOST_MEMORY = -1;
constexpr VkResult VK_ERROR_OUT_OF_DEVICE_MEMORY = -2;
constexpr VkResult VK_ERROR_INITIALIZATION_FAILED = -3;
constexpr VkResult VK_ERROR_DEVICE_LOST = -4;
constexpr VkResult VK_ERROR_MEMORY_MAP_FAILED = -5;
constexpr VkResult VK_ERROR_LAYER_NOT_PRESENT = -6;
constexpr VkResult VK_ERROR_EXTENSION_NOT_PRESENT = -7;
constexpr VkResult VK_ERROR_FEATURE_NOT_PRESENT = -8;
constexpr VkResult VK_ERROR_INCOMPATIBLE_DRIVER = -9;

constexpr std::uint32_t VK_API_VERSION_1_0 = (1u << 22);
constexpr std::uint64_t VK_WHOLE_SIZE = ~std::uint64_t{0};
constexpr std::uint32_t VK_QUEUE_FAMILY_IGNORED = ~std::uint32_t{0};
constexpr std::uint64_t VK_DEFAULT_FENCE_TIMEOUT_NS = 10'000'000'000ull;

enum VkStructureType : std::int32_t {
  VK_STRUCTURE_TYPE_APPLICATION_INFO = 0,
  VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
  VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2,
  VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3,
  VK_STRUCTURE_TYPE_SUBMIT_INFO = 4,
  VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5,
  VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE = 6,
  VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8,
  VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12,
  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16,
  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29,
  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30,
  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
  VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33,
  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34,
  VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35,
  VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39,
  VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40,
  VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42,
  VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER = 44,
};

enum VkPhysicalDeviceType : std::int32_t {
  VK_PHYSICAL_DEVICE_TYPE_OTHER = 0,
  VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1,
  VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,
  VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3,
  VK_PHYSICAL_DEVICE_TYPE_CPU = 4,
};

enum VkSharingMode : std::int32_t {
  VK_SHARING_MODE_EXCLUSIVE = 0,
};

enum VkDescriptorType : std::int32_t {
  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,
};

enum VkPipelineBindPoint : std::int32_t {
  VK_PIPELINE_BIND_POINT_COMPUTE = 1,
};

enum VkCommandBufferLevel : std::int32_t {
  VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0,
};

constexpr VkFlags VK_QUEUE_COMPUTE_BIT = 0x00000002u;

constexpr VkFlags VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001u;
constexpr VkFlags VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002u;
constexpr VkFlags VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020u;

constexpr VkFlags VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001u;
constexpr VkFlags VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002u;
constexpr VkFlags VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004u;
constexpr VkFlags VK_MEMORY_PROPERTY_HOST_CACHED_BIT = 0x00000008u;

constexpr VkFlags VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020u;

constexpr VkFlags VK_ACCESS_SHADER_READ_BIT = 0x00000020u;
constexpr VkFlags VK_ACCESS_SHADER_WRITE_BIT = 0x00000040u;
constexpr VkFlags VK_ACCESS_TRANSFER_READ_BIT = 0x00000800u;
constexpr VkFlags VK_ACCESS_TRANSFER_WRITE_BIT = 0x00001000u;
constexpr VkFlags VK_ACCESS_HOST_READ_BIT = 0x00002000u;
constexpr VkFlags VK_ACCESS_HOST_WRITE_BIT = 0x00004000u;

constexpr VkFlags VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800u;
constexpr VkFlags VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000u;
constexpr VkFlags VK_PIPELINE_STAGE_HOST_BIT = 0x00004000u;
constexpr VkFlags VK_PIPELINE_STAGE_ALL_COMMANDS_BIT = 0x00010000u;

constexpr VkFlags VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT =
    0x00000002u;
constexpr VkFlags VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT = 0x00000004u;

struct VkExtent3D {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t depth = 0;
};

struct VkApplicationInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  const void *pNext = nullptr;
  const char *pApplicationName = nullptr;
  std::uint32_t applicationVersion = 0;
  const char *pEngineName = nullptr;
  std::uint32_t engineVersion = 0;
  std::uint32_t apiVersion = VK_API_VERSION_1_0;
};

struct VkInstanceCreateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  const void *pNext = nullptr;
  VkFlags flags = 0;
  const VkApplicationInfo *pApplicationInfo = nullptr;
  std::uint32_t enabledLayerCount = 0;
  const char *const *ppEnabledLayerNames = nullptr;
  std::uint32_t enabledExtensionCount = 0;
  const char *const *ppEnabledExtensionNames = nullptr;
};

struct VkDeviceQueueCreateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  const void *pNext = nullptr;
  VkFlags flags = 0;
  std::uint32_t queueFamilyIndex = 0;
  std::uint32_t queueCount = 0;
  const float *pQueuePriorities = nullptr;
};

struct VkDeviceCreateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  const void *pNext = nullptr;
  VkFlags flags = 0;
  std::uint32_t queueCreateInfoCount = 0;
  const VkDeviceQueueCreateInfo *pQueueCreateInfos = nullptr;
  std::uint32_t enabledLayerCount = 0;
  const char *const *ppEnabledLayerNames = nullptr;
  std::uint32_t enabledExtensionCount = 0;
  const char *const *ppEnabledExtensionNames = nullptr;
  const void *pEnabledFeatures = nullptr;
};

struct VkPhysicalDeviceProperties {
  std::uint32_t apiVersion = 0;
  std::uint32_t driverVersion = 0;
  std::uint32_t vendorID = 0;
  std::uint32_t deviceID = 0;
  VkPhysicalDeviceType deviceType = VK_PHYSICAL_DEVICE_TYPE_OTHER;
  char deviceName[256] = {};
  std::uint8_t pipelineCacheUUID[16] = {};
  std::uint8_t reserved_limits_and_sparse_properties[2048] = {};
};

struct VkQueueFamilyProperties {
  VkFlags queueFlags = 0;
  std::uint32_t queueCount = 0;
  std::uint32_t timestampValidBits = 0;
  VkExtent3D minImageTransferGranularity{};
};

struct VkMemoryType {
  VkFlags propertyFlags = 0;
  std::uint32_t heapIndex = 0;
};

struct VkMemoryHeap {
  VkDeviceSize size = 0;
  VkFlags flags = 0;
};

struct VkPhysicalDeviceMemoryProperties {
  std::uint32_t memoryTypeCount = 0;
  VkMemoryType memoryTypes[32] = {};
  std::uint32_t memoryHeapCount = 0;
  VkMemoryHeap memoryHeaps[16] = {};
};

struct VkMemoryRequirements {
  VkDeviceSize size = 0;
  VkDeviceSize alignment = 0;
  std::uint32_t memoryTypeBits = 0;
};

struct VkBufferCreateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  const void *pNext = nullptr;
  VkFlags flags = 0;
  VkDeviceSize size = 0;
  VkFlags usage = 0;
  VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  std::uint32_t queueFamilyIndexCount = 0;
  const std::uint32_t *pQueueFamilyIndices = nullptr;
};

struct VkMemoryAllocateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  const void *pNext = nullptr;
  VkDeviceSize allocationSize = 0;
  std::uint32_t memoryTypeIndex = 0;
};

struct VkMappedMemoryRange {
  VkStructureType sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  const void *pNext = nullptr;
  VkDeviceMemory memory = nullptr;
  VkDeviceSize offset = 0;
  VkDeviceSize size = VK_WHOLE_SIZE;
};

struct VkDescriptorSetLayoutBinding {
  std::uint32_t binding = 0;
  VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  std::uint32_t descriptorCount = 0;
  VkFlags stageFlags = 0;
  const VkSampler *pImmutableSamplers = nullptr;
};

struct VkDescriptorSetLayoutCreateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  const void *pNext = nullptr;
  VkFlags flags = 0;
  std::uint32_t bindingCount = 0;
  const VkDescriptorSetLayoutBinding *pBindings = nullptr;
};

struct VkDescriptorPoolSize {
  VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  std::uint32_t descriptorCount = 0;
};

struct VkDescriptorPoolCreateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  const void *pNext = nullptr;
  VkFlags flags = 0;
  std::uint32_t maxSets = 0;
  std::uint32_t poolSizeCount = 0;
  const VkDescriptorPoolSize *pPoolSizes = nullptr;
};

struct VkDescriptorSetAllocateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  const void *pNext = nullptr;
  VkDescriptorPool descriptorPool = nullptr;
  std::uint32_t descriptorSetCount = 0;
  const VkDescriptorSetLayout *pSetLayouts = nullptr;
};

struct VkDescriptorBufferInfo {
  VkBuffer buffer = nullptr;
  VkDeviceSize offset = 0;
  VkDeviceSize range = VK_WHOLE_SIZE;
};

struct VkWriteDescriptorSet {
  VkStructureType sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  const void *pNext = nullptr;
  VkDescriptorSet dstSet = nullptr;
  std::uint32_t dstBinding = 0;
  std::uint32_t dstArrayElement = 0;
  std::uint32_t descriptorCount = 0;
  VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  const void *pImageInfo = nullptr;
  const VkDescriptorBufferInfo *pBufferInfo = nullptr;
  const void *pTexelBufferView = nullptr;
};

struct VkPushConstantRange {
  VkFlags stageFlags = 0;
  std::uint32_t offset = 0;
  std::uint32_t size = 0;
};

struct VkPipelineLayoutCreateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  const void *pNext = nullptr;
  VkFlags flags = 0;
  std::uint32_t setLayoutCount = 0;
  const VkDescriptorSetLayout *pSetLayouts = nullptr;
  std::uint32_t pushConstantRangeCount = 0;
  const VkPushConstantRange *pPushConstantRanges = nullptr;
};

struct VkShaderModuleCreateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  const void *pNext = nullptr;
  VkFlags flags = 0;
  std::size_t codeSize = 0;
  const std::uint32_t *pCode = nullptr;
};

struct VkPipelineShaderStageCreateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  const void *pNext = nullptr;
  VkFlags flags = 0;
  VkFlags stage = 0;
  VkShaderModule module = nullptr;
  const char *pName = nullptr;
  const void *pSpecializationInfo = nullptr;
};

struct VkComputePipelineCreateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  const void *pNext = nullptr;
  VkFlags flags = 0;
  VkPipelineShaderStageCreateInfo stage{};
  VkPipelineLayout layout = nullptr;
  VkPipeline basePipelineHandle = nullptr;
  std::int32_t basePipelineIndex = -1;
};

struct VkCommandPoolCreateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  const void *pNext = nullptr;
  VkFlags flags = 0;
  std::uint32_t queueFamilyIndex = 0;
};

struct VkCommandBufferAllocateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  const void *pNext = nullptr;
  VkCommandPool commandPool = nullptr;
  VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  std::uint32_t commandBufferCount = 0;
};

struct VkCommandBufferBeginInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  const void *pNext = nullptr;
  VkFlags flags = 0;
  const void *pInheritanceInfo = nullptr;
};

struct VkFenceCreateInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  const void *pNext = nullptr;
  VkFlags flags = 0;
};

struct VkSubmitInfo {
  VkStructureType sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  const void *pNext = nullptr;
  std::uint32_t waitSemaphoreCount = 0;
  const VkSemaphore *pWaitSemaphores = nullptr;
  const VkFlags *pWaitDstStageMask = nullptr;
  std::uint32_t commandBufferCount = 0;
  const VkCommandBuffer *pCommandBuffers = nullptr;
  std::uint32_t signalSemaphoreCount = 0;
  const VkSemaphore *pSignalSemaphores = nullptr;
};

struct VkBufferCopy {
  VkDeviceSize srcOffset = 0;
  VkDeviceSize dstOffset = 0;
  VkDeviceSize size = 0;
};

struct VkBufferMemoryBarrier {
  VkStructureType sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  const void *pNext = nullptr;
  VkFlags srcAccessMask = 0;
  VkFlags dstAccessMask = 0;
  std::uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  std::uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  VkBuffer buffer = nullptr;
  VkDeviceSize offset = 0;
  VkDeviceSize size = VK_WHOLE_SIZE;
};

using PFN_vkVoidFunction = void (*)();
using PFN_vkGetInstanceProcAddr =
    PFN_vkVoidFunction (*)(VkInstance, const char *);
using PFN_vkGetDeviceProcAddr = PFN_vkVoidFunction (*)(VkDevice, const char *);
using PFN_vkEnumerateInstanceVersion = VkResult (*)(std::uint32_t *);
using PFN_vkCreateInstance = VkResult (*)(
    const VkInstanceCreateInfo *, const VkAllocationCallbacks *, VkInstance *);
using PFN_vkDestroyInstance = void (*)(VkInstance,
                                       const VkAllocationCallbacks *);
using PFN_vkEnumeratePhysicalDevices =
    VkResult (*)(VkInstance, std::uint32_t *, VkPhysicalDevice *);
using PFN_vkGetPhysicalDeviceProperties =
    void (*)(VkPhysicalDevice, VkPhysicalDeviceProperties *);
using PFN_vkGetPhysicalDeviceQueueFamilyProperties =
    void (*)(VkPhysicalDevice, std::uint32_t *, VkQueueFamilyProperties *);
using PFN_vkGetPhysicalDeviceMemoryProperties =
    void (*)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties *);
using PFN_vkCreateDevice = VkResult (*)(
    VkPhysicalDevice, const VkDeviceCreateInfo *,
    const VkAllocationCallbacks *, VkDevice *);

using PFN_vkDestroyDevice = void (*)(VkDevice, const VkAllocationCallbacks *);
using PFN_vkGetDeviceQueue =
    void (*)(VkDevice, std::uint32_t, std::uint32_t, VkQueue *);
using PFN_vkDeviceWaitIdle = VkResult (*)(VkDevice);
using PFN_vkCreateCommandPool =
    VkResult (*)(VkDevice, const VkCommandPoolCreateInfo *,
                 const VkAllocationCallbacks *, VkCommandPool *);
using PFN_vkDestroyCommandPool =
    void (*)(VkDevice, VkCommandPool, const VkAllocationCallbacks *);
using PFN_vkAllocateCommandBuffers =
    VkResult (*)(VkDevice, const VkCommandBufferAllocateInfo *,
                 VkCommandBuffer *);
using PFN_vkFreeCommandBuffers =
    void (*)(VkDevice, VkCommandPool, std::uint32_t, const VkCommandBuffer *);
using PFN_vkCreateFence =
    VkResult (*)(VkDevice, const VkFenceCreateInfo *,
                 const VkAllocationCallbacks *, VkFence *);
using PFN_vkDestroyFence =
    void (*)(VkDevice, VkFence, const VkAllocationCallbacks *);
using PFN_vkResetFences =
    VkResult (*)(VkDevice, std::uint32_t, const VkFence *);
using PFN_vkWaitForFences =
    VkResult (*)(VkDevice, std::uint32_t, const VkFence *, VkBool32,
                 std::uint64_t);
using PFN_vkQueueSubmit =
    VkResult (*)(VkQueue, std::uint32_t, const VkSubmitInfo *, VkFence);

using PFN_vkCreateBuffer =
    VkResult (*)(VkDevice, const VkBufferCreateInfo *,
                 const VkAllocationCallbacks *, VkBuffer *);
using PFN_vkDestroyBuffer =
    void (*)(VkDevice, VkBuffer, const VkAllocationCallbacks *);
using PFN_vkGetBufferMemoryRequirements =
    void (*)(VkDevice, VkBuffer, VkMemoryRequirements *);
using PFN_vkAllocateMemory =
    VkResult (*)(VkDevice, const VkMemoryAllocateInfo *,
                 const VkAllocationCallbacks *, VkDeviceMemory *);
using PFN_vkFreeMemory =
    void (*)(VkDevice, VkDeviceMemory, const VkAllocationCallbacks *);
using PFN_vkBindBufferMemory =
    VkResult (*)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
using PFN_vkMapMemory = VkResult (*)(VkDevice, VkDeviceMemory, VkDeviceSize,
                                     VkDeviceSize, VkFlags, void **);
using PFN_vkUnmapMemory = void (*)(VkDevice, VkDeviceMemory);
using PFN_vkFlushMappedMemoryRanges =
    VkResult (*)(VkDevice, std::uint32_t, const VkMappedMemoryRange *);
using PFN_vkInvalidateMappedMemoryRanges =
    VkResult (*)(VkDevice, std::uint32_t, const VkMappedMemoryRange *);

using PFN_vkCreateDescriptorSetLayout =
    VkResult (*)(VkDevice, const VkDescriptorSetLayoutCreateInfo *,
                 const VkAllocationCallbacks *, VkDescriptorSetLayout *);
using PFN_vkDestroyDescriptorSetLayout =
    void (*)(VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks *);
using PFN_vkCreateDescriptorPool =
    VkResult (*)(VkDevice, const VkDescriptorPoolCreateInfo *,
                 const VkAllocationCallbacks *, VkDescriptorPool *);
using PFN_vkDestroyDescriptorPool =
    void (*)(VkDevice, VkDescriptorPool, const VkAllocationCallbacks *);
using PFN_vkAllocateDescriptorSets =
    VkResult (*)(VkDevice, const VkDescriptorSetAllocateInfo *,
                 VkDescriptorSet *);
using PFN_vkUpdateDescriptorSets =
    void (*)(VkDevice, std::uint32_t, const VkWriteDescriptorSet *,
             std::uint32_t, const void *);
using PFN_vkCreatePipelineLayout =
    VkResult (*)(VkDevice, const VkPipelineLayoutCreateInfo *,
                 const VkAllocationCallbacks *, VkPipelineLayout *);
using PFN_vkDestroyPipelineLayout =
    void (*)(VkDevice, VkPipelineLayout, const VkAllocationCallbacks *);
using PFN_vkCreateShaderModule =
    VkResult (*)(VkDevice, const VkShaderModuleCreateInfo *,
                 const VkAllocationCallbacks *, VkShaderModule *);
using PFN_vkDestroyShaderModule =
    void (*)(VkDevice, VkShaderModule, const VkAllocationCallbacks *);
using PFN_vkCreateComputePipelines =
    VkResult (*)(VkDevice, VkPipelineCache, std::uint32_t,
                 const VkComputePipelineCreateInfo *,
                 const VkAllocationCallbacks *, VkPipeline *);
using PFN_vkDestroyPipeline =
    void (*)(VkDevice, VkPipeline, const VkAllocationCallbacks *);

using PFN_vkBeginCommandBuffer =
    VkResult (*)(VkCommandBuffer, const VkCommandBufferBeginInfo *);
using PFN_vkEndCommandBuffer = VkResult (*)(VkCommandBuffer);
using PFN_vkResetCommandBuffer = VkResult (*)(VkCommandBuffer, VkFlags);
using PFN_vkCmdBindPipeline =
    void (*)(VkCommandBuffer, VkPipelineBindPoint, VkPipeline);
using PFN_vkCmdBindDescriptorSets =
    void (*)(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout,
             std::uint32_t, std::uint32_t, const VkDescriptorSet *,
             std::uint32_t, const std::uint32_t *);
using PFN_vkCmdPushConstants =
    void (*)(VkCommandBuffer, VkPipelineLayout, VkFlags, std::uint32_t,
             std::uint32_t, const void *);
using PFN_vkCmdDispatch =
    void (*)(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t);
using PFN_vkCmdCopyBuffer =
    void (*)(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t,
             const VkBufferCopy *);
using PFN_vkCmdPipelineBarrier =
    void (*)(VkCommandBuffer, VkFlags, VkFlags, VkFlags, std::uint32_t,
             const void *, std::uint32_t, const VkBufferMemoryBarrier *,
             std::uint32_t, const void *);

struct VulkanFunctions {
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
  PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;
  PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion = nullptr;
  PFN_vkCreateInstance vkCreateInstance = nullptr;
  PFN_vkDestroyInstance vkDestroyInstance = nullptr;
  PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
  PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
  PFN_vkGetPhysicalDeviceQueueFamilyProperties
      vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
  PFN_vkGetPhysicalDeviceMemoryProperties
      vkGetPhysicalDeviceMemoryProperties = nullptr;
  PFN_vkCreateDevice vkCreateDevice = nullptr;

  PFN_vkDestroyDevice vkDestroyDevice = nullptr;
  PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
  PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;
  PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
  PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
  PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
  PFN_vkFreeCommandBuffers vkFreeCommandBuffers = nullptr;
  PFN_vkCreateFence vkCreateFence = nullptr;
  PFN_vkDestroyFence vkDestroyFence = nullptr;
  PFN_vkResetFences vkResetFences = nullptr;
  PFN_vkWaitForFences vkWaitForFences = nullptr;
  PFN_vkQueueSubmit vkQueueSubmit = nullptr;

  PFN_vkCreateBuffer vkCreateBuffer = nullptr;
  PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
  PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
  PFN_vkAllocateMemory vkAllocateMemory = nullptr;
  PFN_vkFreeMemory vkFreeMemory = nullptr;
  PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
  PFN_vkMapMemory vkMapMemory = nullptr;
  PFN_vkUnmapMemory vkUnmapMemory = nullptr;
  PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges = nullptr;
  PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges = nullptr;

  PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
  PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
  PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
  PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
  PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = nullptr;
  PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = nullptr;
  PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
  PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;
  PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
  PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;
  PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
  PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;

  PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
  PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
  PFN_vkResetCommandBuffer vkResetCommandBuffer = nullptr;
  PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
  PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
  PFN_vkCmdPushConstants vkCmdPushConstants = nullptr;
  PFN_vkCmdDispatch vkCmdDispatch = nullptr;
  PFN_vkCmdCopyBuffer vkCmdCopyBuffer = nullptr;
  PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
};

std::string VkResultName(VkResult result);

class VulkanLoader {
public:
  VulkanLoader() = default;
  VulkanLoader(const VulkanLoader &) = delete;
  VulkanLoader &operator=(const VulkanLoader &) = delete;
  VulkanLoader(VulkanLoader &&) noexcept = default;
  VulkanLoader &operator=(VulkanLoader &&) noexcept = default;

  bool Load(std::string *error_out);

  bool IsLoaded() const { return library_.IsOpen(); }
  const std::filesystem::path &LibraryPath() const { return library_.path(); }
  const VulkanFunctions &f() const { return f_; }
  VulkanFunctions &f() { return f_; }

private:
  template <typename T>
  bool LoadGlobalSymbol(const char *name, T *out, std::string *error_out);

  bool LoadGlobalProc(const char *name, PFN_vkVoidFunction *out,
                      std::string *error_out);

  studiocast::util::DynLib library_;
  VulkanFunctions f_{};
};

} // namespace studiocast::vulkan
