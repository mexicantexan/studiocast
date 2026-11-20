#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "core/cuda/cuda_driver_api.h"
#include "core/cuda/kernels/preprocess_to_nchw.h"

namespace studiocast::cuda::kernels {

inline constexpr std::uint16_t kPreparedPreprocessKernelApiVersion = 1;

enum class PreparedPreprocessKernelStatus : std::uint16_t {
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
  foreign_preprocess_spec = 11,
  reconfigured = 12,
  device_lost = 13,
  launch_failed = 14,
  shutdown = 15,
};

struct PreparedPreprocessKernelHandles {
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
};

struct PreparedPreprocessBinding {
  CudaDriverApi *driver = nullptr;
  CudaContextIdentity context_identity{};
  CUstream stream = nullptr;
  // Caller-owned process-local generation identity. The CUstream address is
  // also checked, but is not sufficient by itself.
  std::uint64_t stream_identity = 0;
  std::uint64_t configuration_generation = 0;
  std::uint64_t model_preprocess_generation = 0;
};

struct PreparedPreprocessLaunchMetadata {
  unsigned int grid_x = 0;
  unsigned int grid_y = 0;
  unsigned int block_x = 16;
  unsigned int block_y = 16;
  std::uint32_t dst_w = 0;
  std::uint32_t dst_h = 0;
  float mean[3] = {0.0f, 0.0f, 0.0f};
  float inverse_std[3] = {1.0f, 1.0f, 1.0f};
  std::uint32_t dst_is_bgr = 0;
};

struct PreparedPreprocessKernelEvidence {
  std::uint16_t api_version = kPreparedPreprocessKernelApiVersion;
  PreparedPreprocessKernelStatus status =
      PreparedPreprocessKernelStatus::invalid;
  CudaContextIdentity context_identity{};
  std::uint64_t stream_identity = 0;
  std::uint64_t configuration_generation = 0;
  std::uint64_t model_preprocess_generation = 0;
  std::size_t resource_budget_bytes = 0;
  std::size_t required_resource_bytes = 0;
  PreparedPreprocessLaunchMetadata launch{};
  std::uint64_t successful_launches = 0;
  bool quarantined = false;
};

// Setup-only embedded-module seam used by PreparedPreprocessKernelBundle.
bool PreparePreprocessKernelHandles(CudaDriverApi *driver,
                                    const CudaContextIdentity &expected_context,
                                    PreparedPreprocessKernelHandles *out,
                                    PreparedPreprocessKernelStatus *status_out,
                                    std::string *error_out);

// Fixed-capacity preprocess kernel owner. Setup freezes the exact model
// preprocessing semantics and resolves module/function handles. Execute is
// allocation-free and string-free; the caller-owned resident executor is
// responsible for serialized access, making the prepared raw context current
// on its execution thread, and final section synchronization.
class PreparedPreprocessKernelBundle {
public:
  static constexpr std::size_t RequiredResourceBytes() noexcept {
    return sizeof(PreparedPreprocessBinding) + sizeof(ModelPreprocessSpec) +
           sizeof(PreparedPreprocessKernelHandles) +
           sizeof(PreparedPreprocessKernelEvidence);
  }

  bool Prepare(const PreparedPreprocessBinding &binding,
               const ModelPreprocessSpec &spec,
               std::size_t resource_budget_bytes, std::string *error_out);

  bool Execute(const PreparedPreprocessBinding &binding,
               const ModelPreprocessSpec &spec, const CudaImage &source,
               const CudaTensor &destination,
               PreparedPreprocessKernelStatus *status_out) noexcept;

  bool ExecuteLetterboxed(const PreparedPreprocessBinding &binding,
                          const ModelPreprocessSpec &spec,
                          const LetterboxGeometry &letterbox,
                          const CudaImage &source,
                          const CudaTensor &destination,
                          PreparedPreprocessKernelStatus *status_out) noexcept;

  void InvalidateForReconfiguration(
      std::uint64_t configuration_generation,
      std::uint64_t model_preprocess_generation) noexcept;
  void Shutdown() noexcept;

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] PreparedPreprocessKernelEvidence evidence() const noexcept;

private:
  PreparedPreprocessKernelStatus
  ValidateBinding(const PreparedPreprocessBinding &binding,
                  const ModelPreprocessSpec &spec) const noexcept;
  bool ExecuteImpl(const PreparedPreprocessBinding &binding,
                   const ModelPreprocessSpec &spec,
                   const LetterboxGeometry *letterbox,
                   const CudaImage &source, const CudaTensor &destination,
                   PreparedPreprocessKernelStatus *status_out) noexcept;
  void SetStatus(PreparedPreprocessKernelStatus status,
                 PreparedPreprocessKernelStatus *status_out) noexcept;

  PreparedPreprocessBinding binding_{};
  ModelPreprocessSpec spec_{};
  PreparedPreprocessKernelHandles handles_{};
  PreparedPreprocessKernelEvidence evidence_{};
  bool prepared_ = false;
  bool stopped_ = false;
};

} // namespace studiocast::cuda::kernels
