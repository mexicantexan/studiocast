#pragma once

#include "core/maxine/ar_api.h"
#include "core/maxine/cuda_driver_api.h"
#include "core/maxine/nvcv_api.h"
#include "core/maxine/resident_frame_section.h"
#include "core/maxine/vfx_api.h"
#include "core/video/effects/broadcast_effects.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace studiocast::maxine {

enum class ProductionTransferKind : uint8_t {
  host_upload,
  final_download,
  cpu_continuation_download,
  device_format_bridge,
  background_asset_setup_upload,
  count,
};

enum class ProductionCpuStageKind : uint8_t {
  rgb_to_bgr_staging,
  auto_frame_tracking,
  count,
};

enum class ResidentDeviceOp : uint8_t {
  crop_scale,
  resize,
  vignette,
  count,
};

[[nodiscard]] constexpr uint32_t
ResidentDeviceOpBit(ResidentDeviceOp op) noexcept {
  return op < ResidentDeviceOp::count
             ? (uint32_t{1} << static_cast<uint32_t>(op))
             : 0u;
}

struct ProductionCallCounter {
  uint64_t attempts = 0;
  uint64_t successes = 0;
};

struct ProductionResidentTelemetry {
  std::array<ProductionCallCounter,
             static_cast<std::size_t>(ProductionTransferKind::count)>
      transfers{};
  ProductionCallCounter composites{};
  ProductionCallCounter explicit_synchronizations{};
  ProductionCallCounter synchronous_sdk_runs{};
  ProductionCallCounter asynchronous_sdk_runs{};
  ProductionCallCounter matte_inferences{};
  uint64_t shared_matte_reuses = 0;
  std::array<ProductionCallCounter,
             static_cast<std::size_t>(ResidentStageKind::count)>
      stages{};
  std::array<ProductionCallCounter,
             static_cast<std::size_t>(ProductionCpuStageKind::count)>
      cpu_stages{};
  uint64_t prepare_attempts = 0;
  uint64_t prepare_successes = 0;
  uint64_t setup_attempts = 0;
  uint64_t setup_successes = 0;
  uint64_t setup_reuses = 0;
};

struct DecodedBackgroundRgbView {
  const uint8_t *data = nullptr;
  uint32_t width = 0;
  uint32_t height = 0;
  std::size_t stride_bytes = 0;
  uint64_t asset_generation = 0;

  [[nodiscard]] bool Valid() const noexcept;
};

struct ProductionResidentSetup {
  uint64_t configuration_generation = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t output_width = 0;
  uint32_t output_height = 0;
  uintptr_t runtime_identity = 0;
  video::effects::BroadcastCameraEffects effects{};
  // Fixed union of stages assigned to this Maxine island. Effective settings
  // may also include stages resolved to another backend.
  uint32_t enabled_maxine_stage_mask = 0;
  std::filesystem::path vfx_model_directory;
  DecodedBackgroundRgbView replacement_background{};
  float vignette_center_x_px = 0.0f;
  float vignette_center_y_px = 0.0f;
};

[[nodiscard]] constexpr uint32_t
ProductionResidentStageBit(ResidentStageKind kind) noexcept {
  return kind < ResidentStageKind::count
             ? (uint32_t{1} << static_cast<uint32_t>(kind))
             : 0u;
}

struct ProductionResidentRuntime {
  vfx::VfxApi *vfx = nullptr;
  NvcvApi *nvcv = nullptr;
  ar::ArApi *ar = nullptr;
  CudaDriverApi *cuda = nullptr;
};

struct HostBgrOutputView {
  const uint8_t *data = nullptr;
  uint32_t width = 0;
  uint32_t height = 0;
  std::size_t stride_bytes = 0;

  [[nodiscard]] bool Valid() const noexcept {
    return data && width != 0 && height != 0 &&
           stride_bytes >= static_cast<std::size_t>(width) * 3u;
  }
};

// Injectable seam for the CUDA-driver/PTX helpers. Production and fake paths
// share the same executor and all NvCV/NvVFX/NvAR calls; only helper calls that
// cannot run without a real CUDA context are substituted in hermetic tests.
class IResidentDeviceOps {
public:
  virtual ~IResidentDeviceOps() = default;
  virtual bool Initialize(CudaDriverApi *cuda, uintptr_t runtime_identity,
                          uint32_t required_ops_mask, std::string *error) = 0;
  virtual bool CropScale(const NvCVImage &source, NvCVImage *destination,
                         float crop_x, float crop_y, float crop_width,
                         float crop_height, CUstream stream,
                         std::string *error) noexcept = 0;
  virtual bool Resize(const NvCVImage &source, NvCVImage *destination,
                      CUstream stream, std::string *error) noexcept = 0;
  virtual bool Vignette(NvCVImage *image, float intensity, float center_x,
                        float center_y, CUstream stream,
                        std::string *error) noexcept = 0;
  virtual bool Synchronize(CUstream stream, std::string *error) noexcept = 0;
};

class ProductionResidentFrameExecutor final : public IResidentFrameExecutor {
public:
  explicit ProductionResidentFrameExecutor(
      ProductionResidentRuntime runtime,
      IResidentDeviceOps *device_ops = nullptr);
  ~ProductionResidentFrameExecutor() override;

  ProductionResidentFrameExecutor(const ProductionResidentFrameExecutor &) =
      delete;
  ProductionResidentFrameExecutor &
  operator=(const ProductionResidentFrameExecutor &) = delete;

  // Setup is outside the frame loop. It consumes already-resolved assets and
  // performs all allocation, SDK configuration, stream binding and background
  // upload/resize work. It never performs filesystem discovery.
  bool Configure(const ProductionResidentSetup &setup, std::string *error);
  void InvalidateBindings() noexcept;

  [[nodiscard]] ResidentFrameKey key() const noexcept;
  [[nodiscard]] uintptr_t runtime_identity() const noexcept;
  [[nodiscard]] uintptr_t stream_identity() const noexcept;
  [[nodiscard]] HostBgrOutputView host_output() const noexcept;
  [[nodiscard]] std::string_view last_error() const noexcept;
  [[nodiscard]] const ProductionResidentTelemetry &telemetry() const noexcept;
  void ResetTelemetry() noexcept;

  ResidentBoundaryResult Prepare(const ResidentFrameKey &key) noexcept override;
  ResidentBoundaryResult
  StageRgbToBgr(const HostRgbFrameView &host,
                const ResidentFrameKey &key) noexcept override;
  ResidentBoundaryResult
  UploadStagedBgr(const ResidentFrameKey &key,
                  ResidentImage &output) noexcept override;
  ResidentBoundaryResult
  RunSharedMatte(const ResidentImage &current, uint64_t capture_sequence,
                 uint64_t matte_fingerprint,
                 ResidentMatte &output) noexcept override;
  ResidentBoundaryResult
  RunCompatibleStage(ResidentStageKind kind, const ResidentImage &current,
                     const ResidentMatte *matte,
                     ResidentImage &output) noexcept override;
  ResidentBoundaryResult
  DownloadToHost(ResidentReadbackBoundary boundary,
                 const ResidentImage &current) noexcept override;
  ResidentBoundaryResult
  Synchronize(ResidentReadbackBoundary boundary,
              const ResidentFrameKey &key) noexcept override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::string_view
ProductionTransferKindName(ProductionTransferKind kind) noexcept;
[[nodiscard]] std::string_view
ProductionCpuStageKindName(ProductionCpuStageKind kind) noexcept;

} // namespace studiocast::maxine
