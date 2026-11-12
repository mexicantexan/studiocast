#pragma once

#include "core/video/production_video_backend_manager.h"

#include <memory>

namespace studiocast::vulkan {

class VulkanVideoExecutionFactory;
[[nodiscard]] std::unique_ptr<video::IProductionVideoSetupLeaseOwner>
CreateProductionVulkanSetupLeaseOwner(
    std::unique_ptr<VulkanVideoExecutionFactory> factory);

// Creates the setup-only source for the production-qualified Vulkan resident
// island. The source exposes only the neutral manager contract; Vulkan
// contexts, queues, kernels, and factory ownership remain backend-private.
[[nodiscard]] std::unique_ptr<video::IProductionVideoBackendSetupSource>
CreateProductionVideoBackendSetupSource() noexcept;

} // namespace studiocast::vulkan
