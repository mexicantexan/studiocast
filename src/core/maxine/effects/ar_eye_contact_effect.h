#pragma once

#include <cstdint>
#include <string>

#include "core/maxine/ar_api.h"
#include "core/maxine/effects/maxine_effect.h"

namespace studiocast::maxine::effects {

// NVIDIA Maxine AR Eye Contact (Gaze Redirection) wrapper.
//
// This is intentionally a thin layer over the NvAR ABI, loaded at runtime.
// It does not require Maxine headers at build time.
class ArEyeContactEffect final : public IArFeature {
public:
  explicit ArEyeContactEffect(studiocast::maxine::ar::ArApi *ar);
  ~ArEyeContactEffect() override;

  ArEyeContactEffect(const ArEyeContactEffect &) = delete;
  ArEyeContactEffect &operator=(const ArEyeContactEffect &) = delete;

  const char *Id() const override { return "eye_contact"; }
  const char *DisplayName() const override { return "Eye Contact"; }

  bool
  Configure(const studiocast::video::effects::BroadcastCameraEffects &settings,
            std::string *error) override;
  NvCV_Status Process(studiocast::video::GpuFrame &frame,
                      std::string *error) override;
  bool SetExternalCudaStream(maxine::CUstream stream,
                             std::string *error) override;
  void InvalidateBindings() noexcept override;

  const char *Backend() const override { return "maxine_ar"; }
  const EffectExecutionTelemetry &execution_telemetry() const noexcept {
    return execution_telemetry_;
  }

private:
  bool EnsureCreated(std::string *error);
  bool EnsureLoaded(std::string *error);
  bool EnsureStreamBound(studiocast::video::GpuFrame &frame,
                         std::string *error);
  bool ApplyConfig(std::string *error);

  bool SetU32Required(studiocast::maxine::ar::NvAR_ParameterSelector sel,
                      std::uint32_t val, std::string *error);
  void SetU32Optional(studiocast::maxine::ar::NvAR_ParameterSelector sel,
                      std::uint32_t val);
  void SetF32Optional(studiocast::maxine::ar::NvAR_ParameterSelector sel,
                      float val);

  studiocast::maxine::ar::ArApi *ar_ = nullptr; // non-owning
  studiocast::maxine::ar::NvAR_FeatureHandle handle_ = nullptr;
  bool loaded_ = false;
  bool config_dirty_ = true;

  // Cached configuration.
  bool look_away_enabled_ = true;
  int strength_ = 50; // 0..100 (best-effort mapping)

  // Bound stream used for GPU work. If `frame.cuda_stream` is supplied we use
  // it, otherwise we attempt to create one via NvAR helpers.
  studiocast::maxine::CUstream stream_ = nullptr;
  bool stream_owned_ = false;
  bool external_stream_selected_ = false;
  bool stream_bound_ = false;
  const studiocast::maxine::NvCVImage *bound_input_ = nullptr;
  studiocast::maxine::NvCVImage *bound_output_ = nullptr;
  unsigned bound_width_ = 0;
  unsigned bound_height_ = 0;
  EffectExecutionTelemetry execution_telemetry_{};
};

} // namespace studiocast::maxine::effects
