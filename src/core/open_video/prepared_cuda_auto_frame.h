#pragma once

#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_prepared_transfer_completion.h"
#include "core/hal/prepared_program.h"
#include "core/open_video/frame_analysis_cache.h"
#include "core/open_video/prepared_cuda_yunet_detector.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace studiocast::open_video {

enum class PreparedCudaAutoFrameStatus : std::uint8_t {
  ready,
  executed,
  preserved_input,
  drop_frame,
  invalid_setup,
  stale_binding,
  analysis_failed,
  synchronization_failed,
  device_lost,
  crop_failed,
  shutdown
};
enum class PreparedCudaAutoFrameSource : std::uint8_t { none, face, matte };

struct PreparedCudaAutoFrameRect {
  float x = 0, y = 0, w = 0, h = 0;
};
struct PreparedCudaAutoFrameTrackingBox {
  PreparedCudaAutoFrameSource source = PreparedCudaAutoFrameSource::none;
  PreparedCudaAutoFrameRect box{};
  float confidence = 0;
};
struct PreparedCudaAutoFrameKnobs {
  int strength = 50;
  int smoothing = 50;
  float headroom = .15F;
};
struct PreparedCudaAutoFrameBinding {
  cuda::CudaDriverApi *driver = nullptr;
  cuda::CudaContextIdentity context{};
  cuda::CUcontext owner_context = nullptr;
  cuda::CUstream stream = nullptr;
  std::uint64_t stream_identity = 0, owner_identity = 0;
  std::uint64_t configuration_generation = 0, model_generation = 0;
};
struct PreparedCudaAutoFrameAnalysisResult {
  bool ok = false, found = false, pending_work = false, device_lost = false;
  bool synchronization_failed = false;
  PreparedCudaAutoFrameTrackingBox box{};
  hal::TransferCounters transfers{};
  std::uint32_t completion_waits = 0, cpu_tail_boundaries = 0;
};

class IPreparedCudaAutoFrameAnalysis {
public:
  virtual ~IPreparedCudaAutoFrameAnalysis() = default;
  virtual PreparedCudaAutoFrameAnalysisResult Analyze(
      const PreparedCudaAutoFrameBinding &, const cuda::CudaImage &,
      std::uint64_t, FrameAnalysisCache *,
      const cuda::CudaPreparedTransferCompletion::WaitPolicy &) noexcept = 0;
  virtual bool HasSameFrameArtifact(std::uint64_t,
                                    const FrameAnalysisCache *) const noexcept {
    return false;
  }
};

class PreparedCudaAutoFrameYunetAdapter final
    : public IPreparedCudaAutoFrameAnalysis {
public:
  PreparedCudaAutoFrameYunetAdapter(PreparedCudaYunetDetector *detector,
                                    PreparedCudaYunetBinding binding) noexcept;
  PreparedCudaAutoFrameAnalysisResult
  Analyze(const PreparedCudaAutoFrameBinding &, const cuda::CudaImage &,
          std::uint64_t, FrameAnalysisCache *,
          const cuda::CudaPreparedTransferCompletion::WaitPolicy &) noexcept
      override;

private:
  PreparedCudaYunetDetector *detector_ = nullptr;
  PreparedCudaYunetBinding binding_{};
};

using PreparedCudaAutoFrameMatteProducer = PreparedCudaAutoFrameAnalysisResult (
        *)(void *, const PreparedCudaAutoFrameBinding &,
           const cuda::CudaImage &, std::uint64_t, FrameAnalysisCache *,
           const cuda::CudaPreparedTransferCompletion::WaitPolicy &) noexcept;
class PreparedCudaAutoFrameMatteAdapter final
    : public IPreparedCudaAutoFrameAnalysis {
public:
  PreparedCudaAutoFrameMatteAdapter(void *owner,
                                    PreparedCudaAutoFrameMatteProducer producer,
                                    FrameMatteArtifactKey key, int frame_width,
                                    int frame_height) noexcept;
  PreparedCudaAutoFrameAnalysisResult
  Analyze(const PreparedCudaAutoFrameBinding &, const cuda::CudaImage &,
          std::uint64_t, FrameAnalysisCache *,
          const cuda::CudaPreparedTransferCompletion::WaitPolicy &) noexcept
      override;
  bool HasSameFrameArtifact(std::uint64_t,
                            const FrameAnalysisCache *) const noexcept override;
  static bool ComputeBox(const float *alpha, int alpha_w, int alpha_h,
                         int frame_w, int frame_h,
                         PreparedCudaAutoFrameTrackingBox *out) noexcept;

private:
  void *owner_ = nullptr;
  PreparedCudaAutoFrameMatteProducer producer_ = nullptr;
  FrameMatteArtifactKey key_{};
  int frame_width_ = 0, frame_height_ = 0;
};

class IPreparedCudaAutoFrameCropRuntime {
public:
  virtual ~IPreparedCudaAutoFrameCropRuntime() = default;
  virtual bool Prepare(const PreparedCudaAutoFrameBinding &, int, int,
                       std::size_t, std::string *) noexcept = 0;
  virtual bool Crop(const cuda::CudaImage &, const PreparedCudaAutoFrameRect &,
                    const cuda::CudaImage **, std::string *) noexcept = 0;
  virtual bool Shutdown(std::string *) noexcept = 0;
};

struct PreparedCudaAutoFrameSetup {
  PreparedCudaAutoFrameBinding binding{};
  int frame_width = 0, frame_height = 0;
  PreparedCudaAutoFrameKnobs knobs{};
  IPreparedCudaAutoFrameAnalysis *face = nullptr;
  IPreparedCudaAutoFrameAnalysis *matte = nullptr;
  FrameAnalysisCache *analysis_cache = nullptr;
  std::size_t resource_budget_bytes = 0;
  cuda::CudaPreparedTransferCompletion::WaitPolicy wait{};
};
struct PreparedCudaAutoFrameHandle {
  std::uint64_t id = 0, configuration_generation = 0, model_generation = 0;
};
struct PreparedCudaAutoFrameFrameResult {
  PreparedCudaAutoFrameStatus status =
      PreparedCudaAutoFrameStatus::stale_binding;
  const cuda::CudaImage *output = nullptr;
  PreparedCudaAutoFrameRect source_crop{};
  float tracked_center_x = 0, tracked_center_y = 0;
  PreparedCudaAutoFrameSource source = PreparedCudaAutoFrameSource::none;
  hal::TransferCounters transfers{};
  std::uint32_t completion_waits = 0, cpu_tail_boundaries = 0;
  bool found = false, analysis_ran = false, valid_input_preserved = false;
};

class PreparedCudaAutoFrame {
public:
  PreparedCudaAutoFrame();
  explicit PreparedCudaAutoFrame(IPreparedCudaAutoFrameCropRuntime *) noexcept;
  ~PreparedCudaAutoFrame();
  PreparedCudaAutoFrame(const PreparedCudaAutoFrame &) = delete;
  PreparedCudaAutoFrame &operator=(const PreparedCudaAutoFrame &) = delete;
  bool Prepare(const PreparedCudaAutoFrameSetup &,
               PreparedCudaAutoFrameHandle *, std::string *);
  PreparedCudaAutoFrameFrameResult Execute(const PreparedCudaAutoFrameHandle &,
                                           const PreparedCudaAutoFrameBinding &,
                                           std::uint64_t,
                                           const cuda::CudaImage &) noexcept;
  bool Shutdown(std::string *) noexcept;
  void ResetTracking() noexcept;
  static PreparedCudaAutoFrameRect CenterCrop(int, int, float, float) noexcept;
  static PreparedCudaAutoFrameRect
  TargetCrop(const PreparedCudaAutoFrameTrackingBox &, int, int, float,
             const PreparedCudaAutoFrameKnobs &) noexcept;
  static float SmoothingAlpha(int) noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::open_video
