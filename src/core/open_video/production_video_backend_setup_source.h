#pragma once

#include "core/video/production_video_backend_manager.h"

#include <memory>

namespace studiocast::open_cuda {

// Creates the production setup-only Open CUDA/Open Video source. The returned
// object owns model discovery, artifact validation, provider selection, exact
// resource sizing, and CUDA/ORT setup behind the neutral backend-manager
// contract. AcquireResidentSetup is the only operation that may perform those
// activities; the prepared lease executes from cached resources.
[[nodiscard]] std::unique_ptr<video::IProductionVideoBackendSetupSource>
CreateProductionVideoBackendSetupSource() noexcept;

} // namespace studiocast::open_cuda
