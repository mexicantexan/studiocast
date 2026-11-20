#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "core/cuda/cuda_driver_api.h"

namespace studiocast::cuda::kernels {

inline constexpr std::uint16_t kPreparedOpenCudaVbKernelBundleApiVersion = 1;

// Stable, setup/result evidence for the strict Open CUDA post-processing
// kernel bundle. These values are intentionally independent of CUDA result
// codes so a resident executor can project failures into the neutral HAL
// reason schema without parsing strings.
enum class PreparedOpenCudaVbKernelStatus : std::uint16_t {
  invalid = 0,
  ready = 1,
  launch_enqueued = 2,
  invalid_argument = 3,
  resource_exhausted = 4,
  unavailable = 5,
  module_load_failed = 6,
  function_resolution_failed = 7,
  foreign_context = 8,
  stale_context_generation = 9,
  foreign_stream = 10,
  reconfigured = 11,
  device_lost = 12,
  launch_failed = 13,
  shutdown = 14,
};

enum class OpenCudaVbKernelId : std::uint8_t {
  resize_f32 = 0,
  blur_h_u8x3 = 1,
  blur_v_u8x3 = 2,
  blur_h_f32 = 3,
  blur_v_f32 = 4,
  composite_background = 5,
  composite_solid = 6,
  key_light = 7,
  count = 8,
};

inline constexpr std::size_t kOpenCudaVbKernelCount =
    static_cast<std::size_t>(OpenCudaVbKernelId::count);

struct PreparedOpenCudaVbKernelHandles {
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  std::array<CUfunction, kOpenCudaVbKernelCount> functions{};
};

struct PreparedOpenCudaVbBinding {
  CudaDriverApi *driver = nullptr;
  CudaContextIdentity context_identity{};
  CUstream stream = nullptr;
  // Caller-owned process-local identity for the stream generation. A raw
  // CUstream address alone is not treated as sufficient identity.
  std::uint64_t stream_identity = 0;
  std::uint64_t configuration_generation = 0;
};

struct PreparedOpenCudaVbLaunch {
  OpenCudaVbKernelId kernel = OpenCudaVbKernelId::count;
  unsigned int grid_x = 0;
  unsigned int grid_y = 0;
  unsigned int grid_z = 1;
  unsigned int block_x = 0;
  unsigned int block_y = 0;
  unsigned int block_z = 1;
  unsigned int shared_memory_bytes = 0;
  void **parameters = nullptr;
};

struct PreparedOpenCudaVbKernelEvidence {
  std::uint16_t api_version = kPreparedOpenCudaVbKernelBundleApiVersion;
  PreparedOpenCudaVbKernelStatus status =
      PreparedOpenCudaVbKernelStatus::invalid;
  CudaContextIdentity context_identity{};
  std::uint64_t stream_identity = 0;
  std::uint64_t configuration_generation = 0;
  std::size_t resource_budget_bytes = 0;
  std::size_t required_resource_bytes = 0;
  std::uint64_t successful_launches = 0;
  bool quarantined = false;
};

// Load the embedded module and resolve every function during setup. This is an
// implementation seam used by PreparedOpenCudaVbKernelBundle; frame-time code
// must use the bundle rather than call this function.
bool PrepareOpenCudaVbKernelHandles(CudaDriverApi *driver,
                                    const CudaContextIdentity &expected_context,
                                    PreparedOpenCudaVbKernelHandles *out,
                                    PreparedOpenCudaVbKernelStatus *status_out,
                                    std::string *error_out);

// A fixed-capacity, allocation-free-after-prepare bundle for the Open CUDA
// virtual-background utility kernels.
//
// The owning resident executor must serialize access and keep the driver,
// context, and stream alive through Shutdown(). Kernel launches are enqueued on
// the bound stream; the owning synchronous v1 executor performs its single
// section-completion synchronization after all resident stages.
class PreparedOpenCudaVbKernelBundle {
public:
  static constexpr std::size_t RequiredResourceBytes() noexcept {
    return sizeof(PreparedOpenCudaVbKernelHandles);
  }

  bool Prepare(const PreparedOpenCudaVbBinding &binding,
               std::size_t resource_budget_bytes, std::string *error_out);

  bool Launch(const PreparedOpenCudaVbBinding &binding,
              const PreparedOpenCudaVbLaunch &launch,
              PreparedOpenCudaVbKernelStatus *status_out) noexcept;

  void InvalidateForReconfiguration(
      std::uint64_t new_configuration_generation) noexcept;
  void Shutdown() noexcept;

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] PreparedOpenCudaVbKernelEvidence evidence() const noexcept;

private:
  PreparedOpenCudaVbKernelStatus
  ValidateBinding(const PreparedOpenCudaVbBinding &binding) const noexcept;
  void SetStatus(PreparedOpenCudaVbKernelStatus status,
                 PreparedOpenCudaVbKernelStatus *status_out) noexcept;

  PreparedOpenCudaVbBinding binding_{};
  PreparedOpenCudaVbKernelHandles handles_{};
  PreparedOpenCudaVbKernelEvidence evidence_{};
  bool prepared_ = false;
  bool stopped_ = false;
};

} // namespace studiocast::cuda::kernels
