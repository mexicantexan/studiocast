#pragma once

#include "core/maxine/nvcv_types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace studiocast::maxine {

inline constexpr std::size_t kMaxResidentFrameStages = 16;

enum class ResidentStageKind : uint8_t {
  denoise,
  eye_contact,
  background_blur,
  relighting,
  transfer,
  auto_frame,
  count,
};

enum class CpuTailKind : uint8_t {
  key_light,
  auto_frame_tracking,
  output_format_conversion,
  incompatible_stage,
  count,
};

enum class ResidentBoundaryResult : uint8_t {
  success,
  runtime_failure,
  incompatible_output,
};

enum class ResidentReadbackBoundary : uint8_t {
  final_output,
  cpu_continuation,
};

enum class ResidentExecutionStatus : uint8_t {
  no_effects_passthrough,
  resident_output,
  cpu_visible_output,
  cpu_tail_required,
  failed,
};

enum class ResidentInvalidationReason : uint8_t {
  configuration_refresh,
  model_refresh,
  provider_transition,
  runtime_failure,
  count,
};

struct ResidentFrameKey {
  uint64_t configuration_generation = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uintptr_t runtime_identity = 0;
  uintptr_t stream_identity = 0;

  [[nodiscard]] bool Valid() const noexcept;
  friend bool operator==(const ResidentFrameKey &, const ResidentFrameKey &) =
      default;
};

struct HostRgbFrameView {
  const uint8_t *data = nullptr;
  uint32_t width = 0;
  uint32_t height = 0;
  std::size_t stride_bytes = 0;

  [[nodiscard]] bool ValidFor(const ResidentFrameKey &key) const noexcept;
};

struct ResidentImage {
  NvCVImage *image = nullptr;
  uint64_t image_identity = 0;
  ResidentFrameKey key{};

  [[nodiscard]] bool ValidFor(const ResidentFrameKey &expected) const noexcept;
};

struct ResidentMatte {
  const NvCVImage *image = nullptr;
  uint64_t matte_identity = 0;
  uint64_t source_image_identity = 0;
  uint64_t capture_sequence = 0;
  uint64_t fingerprint = 0;
  ResidentFrameKey key{};

  [[nodiscard]] bool ValidFor(const ResidentImage &source,
                              uint64_t expected_capture_sequence,
                              uint64_t expected_fingerprint) const noexcept;
};

struct ResidentStageSpec {
  ResidentStageKind kind = ResidentStageKind::denoise;
  bool optional = true;
  bool requires_shared_matte = false;
  uint64_t matte_fingerprint = 0;
};

struct ResidentFramePlan {
  std::array<ResidentStageSpec, kMaxResidentFrameStages> stages{};
  std::size_t stage_count = 0;
  bool require_cpu_output = true;
  bool has_cpu_tail = false;
  CpuTailKind cpu_tail = CpuTailKind::incompatible_stage;

  [[nodiscard]] bool AddCompatible(ResidentStageKind kind, bool optional,
                                   bool requires_shared_matte = false,
                                   uint64_t matte_fingerprint = 0) noexcept;
  [[nodiscard]] bool SetCpuTail(CpuTailKind kind) noexcept;
  [[nodiscard]] bool Valid() const noexcept;
  [[nodiscard]] bool Empty() const noexcept { return stage_count == 0; }
};

class IResidentFrameExecutor {
public:
  virtual ~IResidentFrameExecutor() = default;

  // Production adapters bind every image operation to key.stream_identity and
  // must keep these hot methods allocation-free after Prepare succeeds.
  virtual ResidentBoundaryResult Prepare(const ResidentFrameKey &key) noexcept = 0;
  virtual ResidentBoundaryResult
  StageRgbToBgr(const HostRgbFrameView &host,
                const ResidentFrameKey &key) noexcept = 0;
  // This is the section's only CPU-to-GPU transfer boundary. A production
  // adapter wraps exactly one NvCV transfer on the prepared stream here.
  virtual ResidentBoundaryResult
  UploadStagedBgr(const ResidentFrameKey &key, ResidentImage &output) noexcept = 0;
  virtual ResidentBoundaryResult
  RunSharedMatte(const ResidentImage &current, uint64_t capture_sequence,
                 uint64_t matte_fingerprint, ResidentMatte &output) noexcept = 0;
  // current is immutable. Adapters run into distinct prepared storage and only
  // publish output on success, which makes optional fail-open preserve current.
  virtual ResidentBoundaryResult
  RunCompatibleStage(ResidentStageKind kind, const ResidentImage &current,
                     const ResidentMatte *matte,
                     ResidentImage &output) noexcept = 0;
  // This is the only GPU-to-CPU transfer boundary. The section invokes it once
  // either for final output or to terminate at an explicitly named CPU tail.
  virtual ResidentBoundaryResult
  DownloadToHost(ResidentReadbackBoundary boundary,
                 const ResidentImage &current) noexcept = 0;
  virtual ResidentBoundaryResult
  Synchronize(ResidentReadbackBoundary boundary,
              const ResidentFrameKey &key) noexcept = 0;
};

struct ResidentFrameCounters {
  uint64_t prepare_attempts = 0;
  uint64_t prepare_successes = 0;
  uint64_t invalidations = 0;
  uint64_t generation_invalidations = 0;
  uint64_t geometry_invalidations = 0;
  uint64_t runtime_invalidations = 0;
  uint64_t stream_invalidations = 0;
  uint64_t explicit_invalidation_requests = 0;
  uint64_t explicit_invalidations = 0;
  std::array<uint64_t,
             static_cast<std::size_t>(ResidentInvalidationReason::count)>
      explicit_invalidations_by_reason{};
  uint64_t active_frames = 0;
  uint64_t rgb_to_bgr_attempts = 0;
  uint64_t rgb_to_bgr_successes = 0;
  uint64_t upload_attempts = 0;
  uint64_t upload_successes = 0;
  uint64_t download_attempts = 0;
  uint64_t download_successes = 0;
  uint64_t final_download_attempts = 0;
  uint64_t final_download_successes = 0;
  uint64_t cpu_continuation_download_attempts = 0;
  uint64_t cpu_continuation_download_successes = 0;
  uint64_t sync_attempts = 0;
  uint64_t sync_successes = 0;
  uint64_t final_sync_attempts = 0;
  uint64_t final_sync_successes = 0;
  uint64_t cpu_continuation_sync_attempts = 0;
  uint64_t cpu_continuation_sync_successes = 0;
  uint64_t matte_inference_attempts = 0;
  uint64_t matte_inference_successes = 0;
  uint64_t shared_matte_reuses = 0;
  uint64_t incompatible_matte_requests = 0;
  uint64_t optional_fail_open = 0;
  uint64_t required_failures = 0;
  uint64_t incompatible_outputs = 0;
  std::array<uint64_t, static_cast<std::size_t>(ResidentStageKind::count)>
      stage_attempts{};
  std::array<uint64_t, static_cast<std::size_t>(ResidentStageKind::count)>
      stage_successes{};
  std::array<uint64_t, static_cast<std::size_t>(CpuTailKind::count)>
      cpu_tail_boundaries{};
};

struct ResidentFrameExecutionResult {
  ResidentExecutionStatus status = ResidentExecutionStatus::failed;
  ResidentBoundaryResult boundary_result =
      ResidentBoundaryResult::runtime_failure;
  ResidentImage current{};
  ResidentMatte matte{};
  // A cpu_tail_required result has consumed exactly this many resident stages;
  // next_stage_index names the pending CPU tail boundary in this section plan.
  // The caller must execute that tail from the downloaded frame and must not
  // silently skip it or re-enter this section for the same frame.
  std::size_t resident_stages_consumed = 0;
  std::size_t next_stage_index = 0;
  CpuTailKind cpu_tail = CpuTailKind::incompatible_stage;
  bool optional_failure_observed = false;
};

class ResidentFrameSection {
public:
  [[nodiscard]] ResidentFrameExecutionResult
  Execute(const ResidentFramePlan &plan, const ResidentFrameKey &key,
          uint64_t capture_sequence, const HostRgbFrameView &host,
          IResidentFrameExecutor &executor) noexcept;

  [[nodiscard]] const ResidentFrameCounters &counters() const noexcept {
    return counters_;
  }
  [[nodiscard]] bool prepared() const noexcept { return prepared_; }
  [[nodiscard]] const ResidentFrameKey &prepared_key() const noexcept {
    return prepared_key_;
  }

  void Invalidate(ResidentInvalidationReason reason) noexcept;
  void ResetSessionStateAndCounters() noexcept;

private:
  [[nodiscard]] bool EnsurePrepared(const ResidentFrameKey &key,
                                    IResidentFrameExecutor &executor,
                                    ResidentFrameExecutionResult &result) noexcept;
  [[nodiscard]] bool ReadbackAndSynchronize(
      ResidentReadbackBoundary boundary, const ResidentImage &current,
      IResidentFrameExecutor &executor,
      ResidentFrameExecutionResult &result) noexcept;

  bool prepared_ = false;
  ResidentFrameKey prepared_key_{};
  ResidentFrameCounters counters_{};
};

[[nodiscard]] std::string_view ResidentStageKindName(ResidentStageKind kind) noexcept;
[[nodiscard]] std::string_view CpuTailKindName(CpuTailKind kind) noexcept;

} // namespace studiocast::maxine
