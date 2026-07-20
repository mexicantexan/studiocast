#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "core/maxine/effects/maxine_effect.h"
#include "core/maxine/nvcv_api.h"
#include "core/maxine/vfx_api.h"

namespace studiocast::maxine::effects {

// Wrapper for Maxine VFX "Green Screen" effect.
//
// VFX docs:
//   - Input:  BGRu8 chunky
//   - Output: Au8 (matte)
//   - Optional temporal consistency: state handles + NVVFX_TEMPORAL.
class VfxGreenScreenEffect final : public IVfxEffect {
public:
  struct Config {
    // NVVFX_MODE values come from NVIDIA headers. StudioCast does not vendor
    // those headers, so this is treated as a raw numeric.
    std::uint32_t mode = 0;

    // Enable temporal consistency. When enabled, a state handle array is
    // allocated and bound to the effect.
    bool temporal = true;

    // Number of state handles to allocate when temporal is enabled.
    // Default of 1 is adequate for a single-stream pipeline.
    std::uint32_t state_count = 1;
  };

  VfxGreenScreenEffect(maxine::vfx::VfxApi *vfx, maxine::NvcvApi *nvcv,
                       std::filesystem::path model_dir);
  ~VfxGreenScreenEffect() override;

  VfxGreenScreenEffect(const VfxGreenScreenEffect &) = delete;
  VfxGreenScreenEffect &operator=(const VfxGreenScreenEffect &) = delete;

  const char *Id() const override { return "greenscreen"; }
  const char *DisplayName() const override { return "Green Screen"; }

  // Initializes the underlying NvVFX effect (lazy; also called by Process).
  bool Initialize(std::string *error);

  // Configure from canonical settings.
  bool
  Configure(const studiocast::video::effects::BroadcastCameraEffects &settings,
            std::string *error) override;

  // Runs green-screen matte generation on `frame.nvcv_gpu`.
  // Output matte is stored in an internal GPU `NvCVImage`.
  NvCV_Status Process(studiocast::video::GpuFrame &frame,
                      std::string *error) override;
  bool SetExternalCudaStream(maxine::CUstream stream,
                             std::string *error) override;
  void InvalidateBindings() noexcept override;

  // Returns the most recent matte GPU image. Valid after a successful Process.
  const maxine::NvCVImage *MatteGpu() const {
    return matte_ready_ ? &matte_gpu_ : nullptr;
  }

  maxine::CUstream cuda_stream() const { return stream_; }

  const Config &config() const { return cfg_; }
  void SetConfig(const Config &cfg);

private:
  bool EnsureEffectCreated(std::string *error);
  bool ApplyConfigLocked(std::string *error);
  bool EnsureStreamBound(std::string *error);
  bool EnsureTemporalStateLocked(std::string *error);
  bool EnsureMatteImage(unsigned width, unsigned height, std::string *error);

  void Destroy();

  maxine::vfx::VfxApi *vfx_ = nullptr;
  maxine::NvcvApi *nvcv_ = nullptr;
  std::filesystem::path model_dir_;

  maxine::vfx::NvVFX_Handle handle_ = nullptr;
  maxine::CUstream stream_ = nullptr;
  maxine::CUstream external_stream_ = nullptr;
  bool external_stream_selected_ = false;
  bool own_stream_ = false;
  bool stream_bound_ = false;

  // Effect configuration.
  Config cfg_{};
  bool cfg_dirty_ = true;

  // Temporal consistency state.
  std::vector<maxine::vfx::NvVFX_StateObjectHandle> states_;
  bool states_bound_ = false;

  // Output matte (GPU).
  maxine::NvCVImage matte_gpu_{};
  bool matte_ready_ = false;
  bool matte_allocated_ = false;
  const maxine::NvCVImage *bound_input_ = nullptr;
  maxine::NvCVImage *bound_output_ = nullptr;
  unsigned bound_width_ = 0;
  unsigned bound_height_ = 0;
};

} // namespace studiocast::maxine::effects
