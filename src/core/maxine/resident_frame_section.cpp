#include "core/maxine/resident_frame_section.h"

#include <limits>

namespace studiocast::maxine {
namespace {

bool ValidMatteImage(const NvCVImage *image,
                     const ResidentFrameKey &key) noexcept {
  if (!image)
    return false;
  NvCVImageValidationSpec spec{};
  spec.pixel_format = NVCV_A;
  spec.pixel_bytes = 1;
  spec.num_components = 1;
  if (ValidateNvCVImage(*image, spec) != NvCVImageValidationStatus::ok)
    return false;
  return image->width == key.width && image->height == key.height;
}

std::size_t StageIndex(ResidentStageKind kind) noexcept {
  return static_cast<std::size_t>(kind);
}

std::size_t CpuTailIndex(CpuTailKind kind) noexcept {
  return static_cast<std::size_t>(kind);
}

void RecordStageFailure(ResidentFrameExecutionResult &result,
                        std::size_t plan_stage_index,
                        const ResidentStageSpec &stage,
                        ResidentBoundaryResult boundary_result,
                        ResidentStageFailurePoint point) noexcept {
  const uint32_t bit = uint32_t{1} << plan_stage_index;
  result.failed_plan_stage_mask |= bit;
  if (stage.optional) {
    result.optional_fail_open_plan_stage_mask |= bit;
    result.optional_failure_observed = true;
  }
  if (boundary_result == ResidentBoundaryResult::incompatible_output)
    result.incompatible_output_plan_stage_mask |= bit;

  if (result.stage_failure_count < result.stage_failures.size()) {
    result.stage_failures[result.stage_failure_count++] = ResidentStageFailure{
        plan_stage_index, stage.kind, boundary_result, point, stage.optional};
  }
}

} // namespace

bool ResidentFrameKey::Valid() const noexcept {
  return configuration_generation != 0 && width != 0 && height != 0 &&
         runtime_identity != 0 && stream_identity != 0;
}

bool HostRgbFrameView::ValidFor(const ResidentFrameKey &key) const noexcept {
  if (!data || width != key.width || height != key.height)
    return false;
  const uint64_t row_bytes = static_cast<uint64_t>(width) * 3u;
  return row_bytes <= std::numeric_limits<std::size_t>::max() &&
         stride_bytes >= static_cast<std::size_t>(row_bytes);
}

bool ResidentImage::ValidFor(const ResidentFrameKey &expected) const noexcept {
  if (!image || image_identity == 0 || key != expected)
    return false;
  if (ValidateBgrU8CudaNvCVImage(*image) != NvCVImageValidationStatus::ok)
    return false;
  return image->width == expected.width && image->height == expected.height;
}

bool ResidentMatte::ValidFor(const ResidentImage &source,
                             uint64_t expected_capture_sequence,
                             uint64_t expected_fingerprint) const noexcept {
  return matte_identity != 0 &&
         source_image_identity == source.image_identity &&
         capture_sequence == expected_capture_sequence &&
         fingerprint == expected_fingerprint && key == source.key &&
         ValidMatteImage(image, source.key);
}

bool ResidentFramePlan::AddCompatible(ResidentStageKind kind, bool optional,
                                      bool requires_shared_matte,
                                      uint64_t matte_fingerprint) noexcept {
  if (has_cpu_tail || kind >= ResidentStageKind::count ||
      stage_count >= stages.size())
    return false;
  stages[stage_count++] = ResidentStageSpec{
      kind, optional, requires_shared_matte, matte_fingerprint};
  return true;
}

bool ResidentFramePlan::SetCpuTail(CpuTailKind kind) noexcept {
  if (has_cpu_tail || stage_count == 0 || kind >= CpuTailKind::count)
    return false;
  has_cpu_tail = true;
  cpu_tail = kind;
  return true;
}

bool ResidentFramePlan::Valid() const noexcept {
  if (stage_count > stages.size() || (stage_count == 0 && has_cpu_tail) ||
      (has_cpu_tail && cpu_tail >= CpuTailKind::count))
    return false;
  for (std::size_t i = 0; i < stage_count; ++i) {
    if (stages[i].kind >= ResidentStageKind::count)
      return false;
  }
  return true;
}

bool ResidentFrameSection::EnsurePrepared(
    const ResidentFrameKey &key, IResidentFrameExecutor &executor,
    ResidentFrameExecutionResult &result) noexcept {
  if (prepared_ && prepared_key_ == key)
    return true;

  ++counters_.prepare_attempts;
  const auto boundary = executor.Prepare(key);
  result.boundary_result = boundary;
  if (boundary != ResidentBoundaryResult::success)
    return false;
  ++counters_.prepare_successes;

  if (prepared_) {
    ++counters_.invalidations;
    if (prepared_key_.configuration_generation != key.configuration_generation)
      ++counters_.generation_invalidations;
    if (prepared_key_.width != key.width || prepared_key_.height != key.height)
      ++counters_.geometry_invalidations;
    if (prepared_key_.runtime_identity != key.runtime_identity)
      ++counters_.runtime_invalidations;
    if (prepared_key_.stream_identity != key.stream_identity)
      ++counters_.stream_invalidations;
  }
  prepared_key_ = key;
  prepared_ = true;
  return true;
}

bool ResidentFrameSection::ReadbackAndSynchronize(
    ResidentReadbackBoundary boundary, const ResidentImage &current,
    IResidentFrameExecutor &executor,
    ResidentFrameExecutionResult &result) noexcept {
  ++counters_.download_attempts;
  if (boundary == ResidentReadbackBoundary::final_output)
    ++counters_.final_download_attempts;
  else
    ++counters_.cpu_continuation_download_attempts;

  auto call_result = executor.DownloadToHost(boundary, current);
  result.boundary_result = call_result;
  if (call_result != ResidentBoundaryResult::success)
    return false;
  ++counters_.download_successes;
  if (boundary == ResidentReadbackBoundary::final_output)
    ++counters_.final_download_successes;
  else
    ++counters_.cpu_continuation_download_successes;

  ++counters_.sync_attempts;
  if (boundary == ResidentReadbackBoundary::final_output)
    ++counters_.final_sync_attempts;
  else
    ++counters_.cpu_continuation_sync_attempts;

  call_result = executor.Synchronize(boundary, current.key);
  result.boundary_result = call_result;
  if (call_result != ResidentBoundaryResult::success)
    return false;
  ++counters_.sync_successes;
  if (boundary == ResidentReadbackBoundary::final_output)
    ++counters_.final_sync_successes;
  else
    ++counters_.cpu_continuation_sync_successes;
  return true;
}

ResidentFrameExecutionResult ResidentFrameSection::Execute(
    const ResidentFramePlan &plan, const ResidentFrameKey &key,
    uint64_t capture_sequence, const HostRgbFrameView &host,
    IResidentFrameExecutor &executor) noexcept {
  ResidentFrameExecutionResult result{};
  if (plan.Empty() && !plan.has_cpu_tail) {
    result.status = ResidentExecutionStatus::no_effects_passthrough;
    result.boundary_result = ResidentBoundaryResult::success;
    return result;
  }
  if (!plan.Valid() || !key.Valid() || !host.ValidFor(key) ||
      !EnsurePrepared(key, executor, result)) {
    ++counters_.required_failures;
    return result;
  }

  ++counters_.rgb_to_bgr_attempts;
  auto boundary = executor.StageRgbToBgr(host, key);
  result.boundary_result = boundary;
  if (boundary != ResidentBoundaryResult::success) {
    ++counters_.required_failures;
    return result;
  }
  ++counters_.rgb_to_bgr_successes;

  ++counters_.upload_attempts;
  ResidentImage uploaded{};
  boundary = executor.UploadStagedBgr(key, uploaded);
  result.boundary_result = boundary;
  if (boundary != ResidentBoundaryResult::success || !uploaded.ValidFor(key)) {
    if (boundary == ResidentBoundaryResult::success) {
      result.boundary_result = ResidentBoundaryResult::incompatible_output;
      ++counters_.incompatible_outputs;
    }
    ++counters_.required_failures;
    return result;
  }
  ++counters_.upload_successes;
  ++counters_.active_frames;
  result.current = uploaded;

  bool matte_attempted = false;
  bool matte_available = false;
  uint64_t matte_fingerprint = 0;
  ResidentImage matte_source{};

  for (std::size_t i = 0; i < plan.stage_count; ++i) {
    const auto &stage = plan.stages[i];
    const ResidentMatte *matte = nullptr;
    if (stage.requires_shared_matte) {
      if (!matte_attempted) {
        matte_attempted = true;
        matte_fingerprint = stage.matte_fingerprint;
        matte_source = result.current;
        ++counters_.matte_inference_attempts;
        ResidentMatte candidate{};
        boundary = executor.RunSharedMatte(matte_source, capture_sequence,
                                           matte_fingerprint, candidate);
        result.boundary_result = boundary;
        if (boundary == ResidentBoundaryResult::success &&
            candidate.ValidFor(matte_source, capture_sequence,
                               matte_fingerprint)) {
          ++counters_.matte_inference_successes;
          result.matte = candidate;
          matte_available = true;
        } else if (boundary == ResidentBoundaryResult::success) {
          result.boundary_result = ResidentBoundaryResult::incompatible_output;
          ++counters_.incompatible_outputs;
        } else if (boundary == ResidentBoundaryResult::incompatible_output) {
          ++counters_.incompatible_outputs;
        }
      } else if (stage.matte_fingerprint == matte_fingerprint &&
                 matte_available &&
                 result.matte.ValidFor(matte_source, capture_sequence,
                                       matte_fingerprint)) {
        ++counters_.shared_matte_reuses;
      } else {
        ++counters_.incompatible_matte_requests;
        matte_available = false;
        result.boundary_result = ResidentBoundaryResult::incompatible_output;
      }

      if (!matte_available) {
        RecordStageFailure(result, i, stage, result.boundary_result,
                           ResidentStageFailurePoint::shared_matte);
        if (stage.optional) {
          ++counters_.optional_fail_open;
          result.resident_stages_consumed = i + 1;
          result.next_stage_index = i + 1;
          continue;
        }
        ++counters_.required_failures;
        result.resident_stages_consumed = i;
        result.next_stage_index = i;
        return result;
      }
      matte = &result.matte;
    }

    const auto stage_index = StageIndex(stage.kind);
    ++counters_.stage_attempts[stage_index];
    ResidentImage candidate{};
    boundary = executor.RunCompatibleStage(stage.kind, result.current, matte,
                                           candidate);
    result.boundary_result = boundary;
    if (boundary == ResidentBoundaryResult::success &&
        candidate.ValidFor(key)) {
      ++counters_.stage_successes[stage_index];
      result.current = candidate;
      result.resident_stages_consumed = i + 1;
      result.next_stage_index = i + 1;
      continue;
    }
    if (boundary == ResidentBoundaryResult::success) {
      result.boundary_result = ResidentBoundaryResult::incompatible_output;
      ++counters_.incompatible_outputs;
    } else if (boundary == ResidentBoundaryResult::incompatible_output) {
      ++counters_.incompatible_outputs;
    }
    RecordStageFailure(result, i, stage, result.boundary_result,
                       ResidentStageFailurePoint::compatible_stage);
    if (stage.optional) {
      ++counters_.optional_fail_open;
      result.resident_stages_consumed = i + 1;
      result.next_stage_index = i + 1;
      continue;
    }
    ++counters_.required_failures;
    result.resident_stages_consumed = i;
    result.next_stage_index = i;
    return result;
  }

  if (plan.has_cpu_tail) {
    if (!ReadbackAndSynchronize(ResidentReadbackBoundary::cpu_continuation,
                                result.current, executor, result)) {
      ++counters_.required_failures;
      return result;
    }
    ++counters_.cpu_tail_boundaries[CpuTailIndex(plan.cpu_tail)];
    result.status = ResidentExecutionStatus::cpu_tail_required;
    result.cpu_tail = plan.cpu_tail;
    result.next_stage_index = plan.stage_count;
    return result;
  }

  if (plan.require_cpu_output) {
    if (!ReadbackAndSynchronize(ResidentReadbackBoundary::final_output,
                                result.current, executor, result)) {
      ++counters_.required_failures;
      return result;
    }
    result.status = ResidentExecutionStatus::cpu_visible_output;
  } else {
    result.status = ResidentExecutionStatus::resident_output;
    result.boundary_result = ResidentBoundaryResult::success;
  }
  return result;
}

void ResidentFrameSection::Invalidate(
    ResidentInvalidationReason reason) noexcept {
  ++counters_.explicit_invalidation_requests;
  if (reason >= ResidentInvalidationReason::count || !prepared_)
    return;
  ++counters_.invalidations;
  ++counters_.explicit_invalidations;
  ++counters_
        .explicit_invalidations_by_reason[static_cast<std::size_t>(reason)];
  prepared_ = false;
  prepared_key_ = {};
}

void ResidentFrameSection::ResetSessionStateAndCounters() noexcept {
  prepared_ = false;
  prepared_key_ = {};
  counters_ = {};
}

std::string_view ResidentStageKindName(ResidentStageKind kind) noexcept {
  switch (kind) {
  case ResidentStageKind::denoise:
    return "denoise";
  case ResidentStageKind::eye_contact:
    return "eye_contact";
  case ResidentStageKind::background_blur:
    return "background_blur";
  case ResidentStageKind::background_remove:
    return "background_remove";
  case ResidentStageKind::background_replace:
    return "background_replace";
  case ResidentStageKind::relighting:
    return "relighting";
  case ResidentStageKind::transfer:
    return "transfer";
  case ResidentStageKind::auto_frame:
    return "auto_frame";
  case ResidentStageKind::vignette:
    return "vignette";
  case ResidentStageKind::count:
    break;
  }
  return "invalid";
}

std::string_view CpuTailKindName(CpuTailKind kind) noexcept {
  switch (kind) {
  case CpuTailKind::key_light:
    return "key_light";
  case CpuTailKind::auto_frame_tracking:
    return "auto_frame_tracking";
  case CpuTailKind::output_format_conversion:
    return "output_format_conversion";
  case CpuTailKind::incompatible_stage:
    return "incompatible_stage";
  case CpuTailKind::count:
    break;
  }
  return "invalid";
}

} // namespace studiocast::maxine
