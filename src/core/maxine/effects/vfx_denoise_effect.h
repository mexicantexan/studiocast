#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "core/maxine/effects/maxine_effect.h"
#include "core/maxine/nvcv_api.h"
#include "core/maxine/vfx_api.h"

namespace studiocast::maxine::effects {

// Wrapper for Maxine VFX "Denoising" effect.
//
// VFX docs:
//   - Input:  BGRf32 planar normalized
//   - Output: BGRf32 planar normalized
//
// Denoising is temporal and requires a persistent state buffer.
// The state buffer is allocated on the GPU and bound via NVVFX_STATE.
class VfxDenoiseEffect final : public IVfxEffect {
public:
  VfxDenoiseEffect(maxine::vfx::VfxApi *vfx, maxine::NvcvApi *nvcv,
                   std::filesystem::path model_dir);
  ~VfxDenoiseEffect() override;

  VfxDenoiseEffect(const VfxDenoiseEffect &) = delete;
  VfxDenoiseEffect &operator=(const VfxDenoiseEffect &) = delete;

  const char *Id() const override { return "denoise"; }
  const char *DisplayName() const override { return "Denoise"; }

  bool Initialize(std::string *error);

  bool
  Configure(const studiocast::video::effects::BroadcastCameraEffects &settings,
            std::string *error) override;

  NvCV_Status Process(studiocast::video::GpuFrame &frame,
                      std::string *error) override;
  bool SetExternalCudaStream(maxine::CUstream stream,
                             std::string *error) override;
  void InvalidateBindings() noexcept override;

  maxine::NvCVImage *OutputGpu() { return output_ready_ ? &out_gpu_ : nullptr; }
  const maxine::NvCVImage *OutputGpu() const {
    return output_ready_ ? &out_gpu_ : nullptr;
  }

  maxine::CUstream cuda_stream() const { return stream_; }
  const EffectExecutionTelemetry &execution_telemetry() const noexcept {
    return execution_telemetry_;
  }

private:
  bool EnsureEffectCreated(std::string *error);
  bool ApplyConfigLocked(std::string *error);
  bool EnsureStreamBound(std::string *error);
  bool EnsureOutputImage(unsigned width, unsigned height, std::string *error);
  bool EnsureStateBufferLocked(std::string *error);
  bool QueryStateBytesLocked(std::size_t *out_bytes, std::string *error);
  void Destroy();

  static float QuantizeStrength01(float strength01);
  std::string CudaErrorToString(maxine::vfx::VfxApi::cudaError_t err) const;

  maxine::vfx::VfxApi *vfx_ = nullptr;
  maxine::NvcvApi *nvcv_ = nullptr;
  std::filesystem::path model_dir_;

  maxine::vfx::NvVFX_Handle handle_ = nullptr;
  maxine::CUstream stream_ = nullptr;
  maxine::CUstream external_stream_ = nullptr;
  bool external_stream_selected_ = false;
  bool own_stream_ = false;
  bool stream_bound_ = false;
  bool model_bound_ = false;

  bool cfg_dirty_ = true;
  float strength_ = 0.5f; // [0..1]

  maxine::NvCVImage out_gpu_{};
  bool out_allocated_ = false;
  bool output_ready_ = false;
  const maxine::NvCVImage *bound_input_ = nullptr;
  maxine::NvCVImage *bound_output_ = nullptr;
  unsigned bound_width_ = 0;
  unsigned bound_height_ = 0;

  void *state_device_ = nullptr;
  std::size_t state_bytes_ = 0;
  bool state_bound_ = false;
  EffectExecutionTelemetry execution_telemetry_{};
};

} // namespace studiocast::maxine::effects
