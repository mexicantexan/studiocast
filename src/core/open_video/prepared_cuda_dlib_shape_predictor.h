#pragma once

#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_tensor.h"
#include "core/open_video/model_pack_registry.h"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace studiocast::open_video {
struct PreparedCudaDlibBinding {
  cuda::CudaDriverApi *driver = nullptr;
  cuda::CudaContextIdentity context{};
  cuda::CUcontext owner_context = nullptr;
  cuda::CUstream stream = nullptr;
  std::uint64_t stream_identity = 0, owner_identity = 0,
                configuration_generation = 0;
};
struct PreparedCudaDlibSetup {
  PreparedCudaDlibBinding binding{};
  std::filesystem::path models_root;
  std::string model_id;
  std::size_t resource_budget_bytes = 0;
};
struct PreparedCudaFaceBox {
  float left = 0, top = 0, right = 0, bottom = 0;
};
enum class PreparedCudaDlibStatus : std::uint8_t {
  success,
  unavailable,
  invalid_setup,
  model_invalid,
  resource_exhausted,
  stale_binding,
  launch_failed,
  shutdown_failed,
  shutdown
};
struct PreparedCudaDlibResult {
  PreparedCudaDlibStatus status = PreparedCudaDlibStatus::stale_binding;
  bool pending_work = false;
};

class PreparedCudaDlibShapePredictor {
public:
  PreparedCudaDlibShapePredictor();
  ~PreparedCudaDlibShapePredictor();
  PreparedCudaDlibShapePredictor(const PreparedCudaDlibShapePredictor &) =
      delete;
  PreparedCudaDlibShapePredictor &
  operator=(const PreparedCudaDlibShapePredictor &) = delete;
  bool Prepare(const PreparedCudaDlibSetup &, std::string *,
               PreparedCudaDlibStatus *status_out = nullptr);
  PreparedCudaDlibResult Execute(const PreparedCudaDlibBinding &,
                                 const cuda::CudaImage &,
                                 const PreparedCudaFaceBox &,
                                 std::uint64_t capture_sequence,
                                 std::string *) noexcept;
  bool Shutdown(std::string *) noexcept;
  bool ready() const noexcept;
  const ValidatedModelArtifactIdentity &artifact() const noexcept;
  const cuda::CudaTensor &resident_landmarks() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace studiocast::open_video
