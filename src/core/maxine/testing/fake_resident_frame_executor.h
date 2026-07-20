#pragma once

#include "core/maxine/resident_frame_section.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace studiocast::maxine::testing {

struct FakeResidentFrameCalls {
  uint64_t prepare = 0;
  uint64_t rgb_to_bgr = 0;
  uint64_t upload = 0;
  uint64_t matte = 0;
  uint64_t download = 0;
  uint64_t synchronize = 0;
  std::array<uint64_t, kMaxResidentFrameStages> stage_input_identities{};
  std::array<uint64_t, kMaxResidentFrameStages> stage_output_identities{};
  std::array<ResidentStageKind, kMaxResidentFrameStages> stage_kinds{};
  std::size_t stage_count = 0;
  uint64_t uploaded_image_identity = 0;
  uint64_t matte_source_identity = 0;
  uint64_t last_matte_fingerprint = 0;
  ResidentReadbackBoundary last_download_boundary =
      ResidentReadbackBoundary::final_output;
  ResidentReadbackBoundary last_sync_boundary =
      ResidentReadbackBoundary::final_output;
};

class FakeResidentFrameExecutor final : public IResidentFrameExecutor {
public:
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

  void ResetCalls() noexcept { calls_ = {}; }
  [[nodiscard]] const FakeResidentFrameCalls &calls() const noexcept {
    return calls_;
  }

  void FailPrepare(ResidentBoundaryResult result) noexcept {
    prepare_result_ = result;
  }
  void FailStageRgbToBgr(ResidentBoundaryResult result) noexcept {
    rgb_to_bgr_result_ = result;
  }
  void FailUpload(ResidentBoundaryResult result) noexcept {
    upload_result_ = result;
  }
  void FailMatte(ResidentBoundaryResult result) noexcept {
    matte_result_ = result;
  }
  void FailStage(ResidentStageKind kind, ResidentBoundaryResult result) noexcept {
    failing_stage_ = kind;
    stage_failure_result_ = result;
  }
  void FailDownload(ResidentBoundaryResult result) noexcept {
    download_result_ = result;
  }
  void FailSynchronize(ResidentBoundaryResult result) noexcept {
    synchronize_result_ = result;
  }
  void CorruptNextStageOutput() noexcept { corrupt_next_stage_output_ = true; }
  void CorruptNextMatteOutput() noexcept { corrupt_next_matte_output_ = true; }
  void SetSyntheticWork(uint32_t iterations) noexcept {
    synthetic_work_iterations_ = iterations;
  }
  void ClearFailures() noexcept;

private:
  void ConfigureImages(const ResidentFrameKey &key) noexcept;
  void DoSyntheticWork(uint32_t multiplier = 1) noexcept;
  [[nodiscard]] ResidentImage NextImage() noexcept;

  ResidentFrameKey prepared_key_{};
  std::array<NvCVImage, 3> images_{};
  NvCVImage matte_image_{};
  std::size_t next_image_slot_ = 0;
  uint64_t next_identity_ = 1;
  FakeResidentFrameCalls calls_{};
  ResidentBoundaryResult prepare_result_ = ResidentBoundaryResult::success;
  ResidentBoundaryResult rgb_to_bgr_result_ = ResidentBoundaryResult::success;
  ResidentBoundaryResult upload_result_ = ResidentBoundaryResult::success;
  ResidentBoundaryResult matte_result_ = ResidentBoundaryResult::success;
  ResidentStageKind failing_stage_ = ResidentStageKind::count;
  ResidentBoundaryResult stage_failure_result_ =
      ResidentBoundaryResult::runtime_failure;
  ResidentBoundaryResult download_result_ = ResidentBoundaryResult::success;
  ResidentBoundaryResult synchronize_result_ = ResidentBoundaryResult::success;
  bool corrupt_next_stage_output_ = false;
  bool corrupt_next_matte_output_ = false;
  uint32_t synthetic_work_iterations_ = 0;
  uint64_t work_sink_ = 0;
};

} // namespace studiocast::maxine::testing
