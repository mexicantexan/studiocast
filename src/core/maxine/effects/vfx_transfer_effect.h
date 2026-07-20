#pragma once

#include <filesystem>
#include <string>

#include "core/maxine/effects/maxine_effect.h"
#include "core/maxine/nvcv_api.h"
#include "core/maxine/vfx_api.h"

namespace studiocast::maxine::effects {

// Wrapper for Maxine VFX "Transfer" effect.
//
// Used to convert between GPU image formats inside a VFX pipeline.
// In StudioCast we use it to bridge the baseline GPU format (BGRu8 chunky)
// to formats required by specific effects (e.g. Denoising expects BGRf32
// planar).
class VfxTransferEffect final : public IVfxEffect {
public:
  struct OutputFormat {
    maxine::NvCVImage_PixelFormat pixel_format = maxine::NVCV_BGR;
    maxine::NvCVImage_ComponentType component_type = maxine::NVCV_U8;
    unsigned layout = maxine::NVCV_CHUNKY;
    unsigned mem_space = maxine::NVCV_GPU;
  };

  VfxTransferEffect(maxine::vfx::VfxApi *vfx, maxine::NvcvApi *nvcv,
                    std::filesystem::path model_dir, OutputFormat out_fmt);
  ~VfxTransferEffect() override;

  VfxTransferEffect(const VfxTransferEffect &) = delete;
  VfxTransferEffect &operator=(const VfxTransferEffect &) = delete;

  const char *Id() const override { return "transfer"; }
  const char *DisplayName() const override { return "Transfer"; }

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

private:
  bool EnsureEffectCreated(std::string *error);
  bool ApplyConfigLocked(std::string *error);
  bool EnsureStreamBound(std::string *error);
  bool EnsureOutputImage(unsigned width, unsigned height, std::string *error);
  void Destroy();

  maxine::vfx::VfxApi *vfx_ = nullptr;
  maxine::NvcvApi *nvcv_ = nullptr;
  std::filesystem::path model_dir_;

  OutputFormat out_fmt_{};

  maxine::vfx::NvVFX_Handle handle_ = nullptr;
  maxine::CUstream stream_ = nullptr;
  maxine::CUstream external_stream_ = nullptr;
  bool external_stream_selected_ = false;
  bool own_stream_ = false;
  bool stream_bound_ = false;
  bool model_bound_ = false;

  bool cfg_dirty_ = true;

  maxine::NvCVImage out_gpu_{};
  bool out_allocated_ = false;
  bool output_ready_ = false;
  const maxine::NvCVImage *bound_input_ = nullptr;
  maxine::NvCVImage *bound_output_ = nullptr;
  unsigned bound_width_ = 0;
  unsigned bound_height_ = 0;
};

} // namespace studiocast::maxine::effects
