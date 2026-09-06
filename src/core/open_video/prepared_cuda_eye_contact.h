#pragma once

#include "core/cuda/cuda_image.h"
#include "core/cuda/cuda_prepared_transfer_completion.h"
#include "core/cuda/cuda_tensor.h"
#include "core/open_video/frame_analysis_cache.h"
#include "core/open_video/model_pack_registry.h"
#include "core/open_video/prepared_cuda_dlib_shape_predictor.h"
#include "core/open_video/prepared_cuda_yunet_detector.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace studiocast::open_video {

enum class PreparedCudaEyeContactStatus : std::uint8_t {
  executed,
  preserved_input,
  invalid_setup,
  model_invalid,
  provider_mismatch,
  graph_mismatch,
  resource_exhausted,
  stale_binding,
  analysis_failed,
  landmark_failed,
  kernel_failed,
  inference_failed,
  synchronization_failed,
  timeout,
  stop_requested,
  cancelled,
  device_lost,
  shutdown_failed,
  shutdown,
};

struct PreparedCudaEyeContactBinding {
  cuda::CudaDriverApi *driver = nullptr;
  cuda::CudaContextIdentity context{};
  cuda::CUcontext owner_context = nullptr;
  cuda::CUstream stream = nullptr;
  std::uint64_t stream_identity = 0;
  std::uint64_t owner_identity = 0;
  std::uint64_t configuration_generation = 0;
  std::uint64_t model_generation = 0;
};

struct PreparedCudaEyeModelContract {
  static constexpr const char *kEyeInput = "eye_rgb:0";
  static constexpr const char *kAnchorsInput = "anchors:0";
  static constexpr const char *kAnglesInput = "angles:0";
  static constexpr const char *kOutput = "rgb:0";
};

struct PreparedCudaEyeContactModels {
  const ModelPack *eye_pack = nullptr;
  const ModelFile *left_file = nullptr;
  const ModelFile *right_file = nullptr;
  std::filesystem::path left_path;
  std::filesystem::path right_path;
  ValidatedModelArtifactIdentity yunet;
  ValidatedModelArtifactIdentity landmarks;
  ValidatedModelArtifactIdentity left;
  ValidatedModelArtifactIdentity right;
};

struct PreparedCudaEyeContactSetup {
  PreparedCudaEyeContactBinding binding{};
  PreparedCudaEyeContactModels models{};
  int cuda_device_id = 0;
  int frame_width = 0;
  int frame_height = 0;
  int strength = 0;
  bool look_away_enabled = false;
  std::size_t resource_budget_bytes = 0;
  FrameAnalysisCache *analysis_cache = nullptr;
  PreparedCudaYunetSetup yunet_setup{};
  PreparedCudaDlibSetup dlib_setup{};
  cuda::CudaPreparedTransferCompletion::WaitPolicy wait{};
};

struct PreparedCudaEyeContactHandle {
  std::uint64_t id = 0;
  std::uint64_t configuration_generation = 0;
  std::uint64_t model_generation = 0;
};

struct PreparedCudaEyeContactAnalysisResult {
  PreparedCudaEyeContactStatus status =
      PreparedCudaEyeContactStatus::analysis_failed;
  PreparedCudaFaceBox face{};
  bool found = false;
  bool pending_work = false;
  bool device_lost = false;
  std::uint32_t enqueued_readbacks = 0;
  std::uint32_t completed_readbacks = 0;
  std::uint32_t completion_waits = 0;
  std::uint32_t cpu_tail_boundaries = 0;
};

PreparedCudaEyeContactStatus ValidateResidentDlibLandmarks(
    const PreparedCudaDlibResult &, const cuda::CudaTensor &) noexcept;

class IPreparedCudaEyeContactAnalysis {
public:
  virtual ~IPreparedCudaEyeContactAnalysis() = default;
  virtual PreparedCudaEyeContactStatus
  Prepare(const PreparedCudaEyeContactSetup &) noexcept = 0;
  virtual const ValidatedModelArtifactIdentity &artifact() const noexcept = 0;
  virtual PreparedCudaEyeContactAnalysisResult
  Detect(const PreparedCudaEyeContactBinding &, const cuda::CudaImage &,
         std::uint64_t, FrameAnalysisCache *,
         const cuda::CudaPreparedTransferCompletion::WaitPolicy &) noexcept = 0;
  virtual PreparedCudaEyeContactStatus
  Landmarks(const PreparedCudaEyeContactBinding &, const cuda::CudaImage &,
            const PreparedCudaFaceBox &, std::uint64_t,
            const cuda::CudaTensor **resident_landmarks) noexcept = 0;
  virtual const ValidatedModelArtifactIdentity &
  landmark_artifact() const noexcept = 0;
  virtual PreparedCudaEyeContactStatus Shutdown() noexcept = 0;
};

struct PreparedCudaEyeContactRuntimeConfig {
  PreparedCudaEyeContactBinding binding{};
  PreparedCudaEyeContactModels models{};
  int cuda_device_id = 0;
  int frame_width = 0;
  int frame_height = 0;
  std::size_t resource_budget_bytes = 0;
  cuda::CudaPreparedTransferCompletion::WaitPolicy wait{};
};

struct PreparedCudaEyeContactRuntimeFrameResult {
  PreparedCudaEyeContactStatus status =
      PreparedCudaEyeContactStatus::kernel_failed;
  const cuda::CudaImage *output = nullptr;
  bool pending_work = false;
  bool device_lost = false;
  std::uint32_t inference_attempts = 0;
  std::uint32_t inference_successes = 0;
};

class IPreparedCudaEyeContactRuntime {
public:
  virtual ~IPreparedCudaEyeContactRuntime() = default;
  virtual PreparedCudaEyeContactStatus
  Prepare(const PreparedCudaEyeContactRuntimeConfig &) noexcept = 0;
  virtual PreparedCudaEyeContactRuntimeFrameResult
  Execute(const PreparedCudaEyeContactBinding &, const cuda::CudaImage &,
          const cuda::CudaTensor &, float strength) noexcept = 0;
  virtual PreparedCudaEyeContactStatus Shutdown() noexcept = 0;
};

struct PreparedCudaEyeContactFrameResult {
  PreparedCudaEyeContactStatus status =
      PreparedCudaEyeContactStatus::stale_binding;
  const cuda::CudaImage *output = nullptr;
  bool valid_input_preserved = false;
  bool face_found = false;
  bool pending_work = false;
  bool device_lost = false;
  std::uint32_t enqueued_readbacks = 0;
  std::uint32_t completed_readbacks = 0;
  std::uint32_t completion_waits = 0;
  std::uint32_t cpu_tail_boundaries = 0;
  std::uint32_t inference_attempts = 0;
  std::uint32_t inference_successes = 0;
};

class PreparedCudaEyeContact {
public:
  PreparedCudaEyeContact();
  PreparedCudaEyeContact(IPreparedCudaEyeContactAnalysis *,
                         IPreparedCudaEyeContactRuntime *) noexcept;
  ~PreparedCudaEyeContact();
  PreparedCudaEyeContact(const PreparedCudaEyeContact &) = delete;
  PreparedCudaEyeContact &operator=(const PreparedCudaEyeContact &) = delete;

  bool Prepare(const PreparedCudaEyeContactSetup &,
               PreparedCudaEyeContactHandle *, std::string *);
  PreparedCudaEyeContactFrameResult
  Execute(const PreparedCudaEyeContactHandle &,
          const PreparedCudaEyeContactBinding &, std::uint64_t capture_sequence,
          const cuda::CudaImage &) noexcept;
  bool Shutdown(std::string *) noexcept;
  bool ready() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace studiocast::open_video
