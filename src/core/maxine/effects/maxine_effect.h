#pragma once

#include <string>

#include "core/maxine/nvcv_types.h"
#include "core/video/effects/broadcast_effects.h"
#include "core/video/gpu_frame.h"

namespace studiocast::maxine::effects {

enum class EffectKind {
  vfx,
  ar,
};

// Maxine-backed GPU effect interface (interfaces only).
//
// Concrete implementations will be responsible for runtime-loading
// Maxine/VFX/AR symbols (dlopen/dlsym) via existing loaders, and for honoring
// Maxine runtime availability/diagnostics.
class IMaxineEffect {
public:
  virtual ~IMaxineEffect() = default;

  // Stable identifier used in config / IPC.
  virtual const char *Id() const = 0;

  // Human-friendly display name (GUI can override / localize later).
  virtual const char *DisplayName() const = 0;

  virtual EffectKind Kind() const = 0;

  // Configure the effect from the canonical effect settings model.
  // Implementations should be safe to call repeatedly and may be invoked while
  // the pipeline is running.
  virtual bool
  Configure(const studiocast::video::effects::BroadcastCameraEffects &settings,
            std::string *error) = 0;

  // Process the frame in place. Implementations may:
  // - run GPU processing via `frame.nvcv_gpu` and/or `frame.device_ptr`
  // - populate `frame.cpu` for v4l2loopback output
  virtual NvCV_Status Process(studiocast::video::GpuFrame &frame,
                              std::string *error) = 0;

  // Select a pipeline-owned CUDA stream. The effect binds but never destroys
  // this stream. Repeating the same selection is idempotent.
  virtual bool SetExternalCudaStream(maxine::CUstream stream,
                                     std::string *error) = 0;

  // Runtime failures invalidate pointer/stream bindings without discarding
  // stable user configuration or ownership telemetry.
  virtual void InvalidateBindings() noexcept = 0;

  virtual const char *Backend() const { return "maxine"; }
};

class IVfxEffect : public IMaxineEffect {
public:
  EffectKind Kind() const final { return EffectKind::vfx; }
};

class IArFeature : public IMaxineEffect {
public:
  EffectKind Kind() const final { return EffectKind::ar; }
};

} // namespace studiocast::maxine::effects
