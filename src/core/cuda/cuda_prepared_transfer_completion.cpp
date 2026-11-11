#include "core/cuda/cuda_prepared_transfer_completion.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <thread>

namespace studiocast::cuda {
namespace {

using Status = CudaPreparedTransferCompletion::Status;
using Result = CudaPreparedTransferCompletion::Result;

Result Failed(
    Status status, CUresult cuda_result = CUDA_SUCCESS,
    std::uint32_t stage = CudaPreparedTransferCompletion::kNoStage) noexcept {
  return Result{status, cuda_result, stage,
                CudaPreparedTransferCompletion::kApiVersion};
}

bool CheckedMultiply(std::size_t left, std::size_t right,
                     std::size_t *out) noexcept {
  if (!out || left == 0 || right == 0 ||
      left > std::numeric_limits<std::size_t>::max() / right) {
    return false;
  }
  *out = left * right;
  return true;
}

std::uint64_t DefaultNowNanoseconds(void *) noexcept {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

void DefaultPollWait(void *, std::uint64_t wait_nanoseconds) noexcept {
  std::this_thread::sleep_for(std::chrono::nanoseconds(wait_nanoseconds));
}

} // namespace

CudaPreparedTransferCompletion::~CudaPreparedTransferCompletion() {
  // If a bounded wait was interrupted while DMA is still outstanding,
  // Shutdown() deliberately retains the staging allocation rather than free
  // memory still owned by CUDA. The owner must retry Shutdown after completion
  // or tear down the CUDA context.
  (void)Shutdown();
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::FromCudaFailure(Status fallback,
                                                CUresult result) noexcept {
  if (IsCudaDeviceLossResult(result)) {
    if (!quarantined_)
      ++counters_.device_losses;
    quarantined_ = true;
    frame_reuse_blocked_ = true;
    return Failed(Status::device_lost, result);
  }
  if (result == CUDA_ERROR_OUT_OF_MEMORY)
    return Failed(Status::resource_exhausted, result);
  return Failed(fallback, result);
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::BindContext() noexcept {
  if (!cuda_ || !binding_.context.Valid())
    return Failed(Status::invalid_context);
  const CUresult result = cuda_->BindPreparedContext(binding_.context);
  if (result == CUDA_SUCCESS)
    return {};
  return FromCudaFailure(Status::stale_context, result);
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::ValidateBinding(
    const Binding &binding) noexcept {
  if (shutdown_)
    return Failed(Status::shutdown);
  if (!prepared_ || !cuda_)
    return Failed(Status::not_prepared);
  if (!binding.context.Valid())
    return Failed(Status::invalid_context);
  if (!binding.stream || binding.stream_identity == 0 ||
      binding.owner_identity == 0 || binding.configuration_generation == 0) {
    return Failed(Status::invalid_argument);
  }
  if (binding.context.physical_device != binding_.context.physical_device ||
      binding.context.retained_context_id !=
          binding_.context.retained_context_id) {
    return Failed(Status::foreign_context);
  }
  if (binding.context.logical_generation !=
          binding_.context.logical_generation ||
      binding.configuration_generation != binding_.configuration_generation) {
    return Failed(Status::stale_context);
  }
  if (binding.owner_identity != binding_.owner_identity)
    return Failed(Status::foreign_owner);
  if (binding.stream != binding_.stream ||
      binding.stream_identity != binding_.stream_identity) {
    return Failed(Status::wrong_stream);
  }

  const CudaContextIdentity *current = cuda_->context_identity();
  if (!current || !current->Valid())
    return Failed(Status::invalid_context);
  if (current->physical_device != binding_.context.physical_device ||
      current->retained_context_id != binding_.context.retained_context_id) {
    return Failed(Status::foreign_context);
  }
  if (current->logical_generation != binding_.context.logical_generation)
    return Failed(Status::stale_context);
  if (quarantined_)
    return Failed(Status::device_lost);
  return {};
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::ValidateFrameBinding(
    const Binding &binding) noexcept {
  Result result = ValidateBinding(binding);
  if (!result.ok())
    return result;
  if (!frame_active_)
    return Failed(Status::invalid_sequence);
  return BindContext();
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::ReleaseCandidate(
    Candidate *candidate) noexcept {
  if (!candidate || !cuda_)
    return Failed(Status::invalid_context);
  Result first_failure{};
  const auto capture_failure = [this, &first_failure](Status fallback,
                                                      CUresult result) {
    if (first_failure.ok() && result != CUDA_SUCCESS)
      first_failure = FromCudaFailure(fallback, result);
  };
  const auto &functions = cuda_->f();
  if (candidate->completion && functions.cuEventDestroy) {
    capture_failure(Status::event_failed,
                    functions.cuEventDestroy(candidate->completion));
    ++counters_.setup_event_destructions;
    candidate->completion = nullptr;
  }
  for (std::size_t index = candidate->stage_count; index > 0; --index) {
    const std::size_t slot = index - 1;
    if (candidate->stage_end[slot] && functions.cuEventDestroy) {
      capture_failure(Status::event_failed,
                      functions.cuEventDestroy(candidate->stage_end[slot]));
      ++counters_.setup_event_destructions;
      candidate->stage_end[slot] = nullptr;
    }
    if (candidate->stage_begin[slot] && functions.cuEventDestroy) {
      capture_failure(Status::event_failed,
                      functions.cuEventDestroy(candidate->stage_begin[slot]));
      ++counters_.setup_event_destructions;
      candidate->stage_begin[slot] = nullptr;
    }
  }
  if (candidate->readback_host && functions.cuMemFreeHost) {
    capture_failure(Status::resource_release_failed,
                    functions.cuMemFreeHost(candidate->readback_host));
    ++counters_.setup_pinned_frees;
    candidate->readback_host = nullptr;
  }
  if (candidate->upload_host && functions.cuMemFreeHost) {
    capture_failure(Status::resource_release_failed,
                    functions.cuMemFreeHost(candidate->upload_host));
    ++counters_.setup_pinned_frees;
    candidate->upload_host = nullptr;
  }
  return first_failure;
}

void CudaPreparedTransferCompletion::ResetFrameState() noexcept {
  stage_elapsed_ns_.fill(0);
  stage_begin_recorded_.fill(false);
  stage_end_recorded_.fill(false);
  next_stage_index_ = 0;
  capture_sequence_ = 0;
  frame_active_ = false;
  frame_reuse_blocked_ = false;
  upload_enqueued_ = false;
  work_submitted_ = false;
  readback_enqueued_ = false;
  next_readback_index_ = 0;
  completion_recorded_ = false;
  failed_frame_completion_ = false;
  wait_attempted_ = false;
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::ReleaseCurrent(bool mark_shutdown) noexcept {
  if (!prepared_) {
    shutdown_ = mark_shutdown;
    return {};
  }

  Result first_failure{};
  if (cuda_ && binding_.context.Valid()) {
    const CUresult bind = cuda_->BindPreparedContext(binding_.context);
    if (bind != CUDA_SUCCESS)
      first_failure = FromCudaFailure(Status::stale_context, bind);
  }

  Candidate current;
  current.upload_host = upload_host_;
  current.readback_host = readback_host_;
  current.stage_begin = stage_begin_events_;
  current.stage_end = stage_end_events_;
  current.completion = completion_event_;
  current.stage_count = timed_stage_count_;
  const Result release = ReleaseCandidate(&current);
  if (first_failure.ok() && !release.ok())
    first_failure = release;

  upload_host_ = nullptr;
  readback_host_ = nullptr;
  stage_begin_events_.fill(nullptr);
  stage_end_events_.fill(nullptr);
  completion_event_ = nullptr;
  upload_row_bytes_ = 0;
  upload_rows_ = 0;
  upload_size_bytes_ = 0;
  readback_row_bytes_ = 0;
  readback_rows_ = 0;
  readback_size_bytes_ = 0;
  readbacks_.fill({});
  readback_count_ = 0;
  resident_work_ = false;
  resource_budget_ = {};
  timed_stage_count_ = 0;
  prepared_ = false;
  shutdown_ = mark_shutdown;
  ResetFrameState();
  binding_ = {};
  cuda_ = nullptr;
  return first_failure;
}

CudaPreparedTransferCompletion::Result CudaPreparedTransferCompletion::Prepare(
    CudaDriverApi *cuda, const PrepareOptions &options) noexcept {
  if (prepared_)
    return Failed(Status::already_prepared);
  if (options.api_version != kApiVersion)
    return Failed(Status::unsupported);
  if (!cuda || !cuda->IsInitialized())
    return Failed(Status::invalid_context);
  if (!cuda->HasPreparedTransferCompletionSupport())
    return Failed(Status::unsupported);
  if (!options.binding.context.Valid())
    return Failed(Status::invalid_context);
  if (!options.binding.stream || options.binding.stream_identity == 0 ||
      options.binding.owner_identity == 0 ||
      options.binding.configuration_generation == 0 ||
      options.timed_stage_count == 0 ||
      options.timed_stage_count > kMaxTimedStages) {
    return Failed(Status::invalid_argument);
  }
  const CudaContextIdentity *current = cuda->context_identity();
  if (!current || !current->Valid())
    return Failed(Status::invalid_context);
  if (current->physical_device != options.binding.context.physical_device ||
      current->retained_context_id !=
          options.binding.context.retained_context_id) {
    return Failed(Status::foreign_context);
  }
  if (current->logical_generation !=
      options.binding.context.logical_generation) {
    return Failed(Status::stale_context);
  }
  std::size_t upload_bytes = 0;
  std::size_t readback_bytes = 0;
  if (!options.resident_work &&
      !CheckedMultiply(options.upload_row_bytes, options.upload_rows,
                       &upload_bytes)) {
    return Failed(Status::invalid_argument);
  }
  auto readbacks = options.readbacks;
  std::uint32_t readback_count = options.readback_count;
  if (readback_count == 0) {
    if (!CheckedMultiply(options.readback_row_bytes, options.readback_rows,
                         &readback_bytes))
      return Failed(Status::invalid_argument);
    readback_count = 1;
    readbacks[0] = {0, 0, 0, options.readback_row_bytes, options.readback_rows};
  } else {
    if (readback_count > kMaxReadbacks)
      return Failed(Status::invalid_argument);
    readback_bytes = 0;
    for (std::uint32_t index = 0; index < readback_count; ++index) {
      const auto &descriptor = readbacks[index];
      std::size_t bytes = 0;
      if (descriptor.source == 0 ||
          descriptor.source_pitch < descriptor.row_bytes ||
          descriptor.row_bytes == 0 || descriptor.rows == 0 ||
          !CheckedMultiply(descriptor.row_bytes, descriptor.rows, &bytes) ||
          descriptor.host_offset_bytes >
              std::numeric_limits<std::size_t>::max() - bytes) {
        return Failed(Status::invalid_argument);
      }
      const auto maximum = std::numeric_limits<CUdeviceptr>::max();
      if (descriptor.source > maximum - descriptor.row_bytes)
        return Failed(Status::invalid_argument);
      const std::size_t last_row = descriptor.rows - 1;
      const auto remaining = maximum - descriptor.source - descriptor.row_bytes;
      if (last_row > 0 && descriptor.source_pitch > remaining / last_row)
        return Failed(Status::invalid_argument);
      readback_bytes =
          std::max(readback_bytes, descriptor.host_offset_bytes + bytes);
      for (std::uint32_t prior = 0; prior < index; ++prior) {
        const auto &other = readbacks[prior];
        const std::size_t other_end =
            other.host_offset_bytes + other.row_bytes * other.rows;
        const std::size_t this_end = descriptor.host_offset_bytes + bytes;
        if (descriptor.host_offset_bytes < other_end &&
            other.host_offset_bytes < this_end)
          return Failed(Status::invalid_argument);
      }
    }
  }
  if (upload_bytes > std::numeric_limits<std::size_t>::max() - readback_bytes)
    return Failed(Status::invalid_argument);
  const std::size_t total_pinned_bytes = upload_bytes + readback_bytes;
  const std::uint32_t required_events = options.timed_stage_count * 2U + 1U;
  if (options.budget.max_pinned_host_bytes < total_pinned_bytes ||
      options.budget.max_events < required_events) {
    return Failed(Status::resource_exhausted);
  }

  const CUresult bind = cuda->BindPreparedContext(options.binding.context);
  if (bind != CUDA_SUCCESS) {
    cuda_ = cuda;
    binding_ = options.binding;
    Result result = FromCudaFailure(Status::stale_context, bind);
    cuda_ = nullptr;
    binding_ = {};
    return result;
  }

  cuda_ = cuda;
  Candidate candidate;
  const auto &functions = cuda->f();
  CUresult result = CUDA_SUCCESS;
  if (!options.resident_work) {
    result = functions.cuMemHostAlloc(&candidate.upload_host, upload_bytes,
                                      CU_MEMHOSTALLOC_DEFAULT);
    if (result != CUDA_SUCCESS || !candidate.upload_host) {
      cuda_ = nullptr;
      return FromCudaFailure(Status::resource_exhausted, result);
    }
    ++counters_.setup_pinned_allocations;
  }

  result = functions.cuMemHostAlloc(&candidate.readback_host, readback_bytes,
                                    CU_MEMHOSTALLOC_DEFAULT);
  if (result != CUDA_SUCCESS || !candidate.readback_host) {
    Result failure = FromCudaFailure(Status::resource_exhausted, result);
    (void)ReleaseCandidate(&candidate);
    cuda_ = nullptr;
    return failure;
  }
  ++counters_.setup_pinned_allocations;

  for (std::uint32_t index = 0; index < options.timed_stage_count; ++index) {
    result = functions.cuEventCreate(&candidate.stage_begin[index],
                                     CU_EVENT_DEFAULT);
    if (result != CUDA_SUCCESS || !candidate.stage_begin[index]) {
      candidate.stage_count = index + 1;
      Result failure = FromCudaFailure(Status::event_failed, result);
      (void)ReleaseCandidate(&candidate);
      cuda_ = nullptr;
      return failure;
    }
    ++counters_.setup_event_creations;
    result =
        functions.cuEventCreate(&candidate.stage_end[index], CU_EVENT_DEFAULT);
    if (result != CUDA_SUCCESS || !candidate.stage_end[index]) {
      candidate.stage_count = index + 1;
      Result failure = FromCudaFailure(Status::event_failed, result);
      (void)ReleaseCandidate(&candidate);
      cuda_ = nullptr;
      return failure;
    }
    ++counters_.setup_event_creations;
    candidate.stage_count = index + 1;
  }
  result = functions.cuEventCreate(&candidate.completion, CU_EVENT_DEFAULT);
  if (result != CUDA_SUCCESS || !candidate.completion) {
    Result failure = FromCudaFailure(Status::event_failed, result);
    (void)ReleaseCandidate(&candidate);
    cuda_ = nullptr;
    return failure;
  }
  ++counters_.setup_event_creations;

  cuda_ = cuda;
  binding_ = options.binding;
  upload_host_ = candidate.upload_host;
  readback_host_ = candidate.readback_host;
  stage_begin_events_ = candidate.stage_begin;
  stage_end_events_ = candidate.stage_end;
  completion_event_ = candidate.completion;
  upload_row_bytes_ = options.upload_row_bytes;
  upload_rows_ = options.upload_rows;
  upload_size_bytes_ = upload_bytes;
  readback_row_bytes_ = options.readback_row_bytes;
  readback_rows_ = options.readback_rows;
  readback_size_bytes_ = readback_bytes;
  readbacks_ = readbacks;
  readback_count_ = readback_count;
  resident_work_ = options.resident_work;
  resource_budget_ = options.budget;
  timed_stage_count_ = options.timed_stage_count;
  candidate = {};
  prepared_ = true;
  shutdown_ = false;
  quarantined_ = false;
  ResetFrameState();
  return {};
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::BeginFrame(
    const Binding &binding, std::uint64_t capture_sequence) noexcept {
  Result validation = ValidateBinding(binding);
  if (!validation.ok())
    return validation;
  if (capture_sequence == 0)
    return Failed(Status::invalid_argument);
  if (frame_active_ || frame_reuse_blocked_)
    return Failed(Status::resource_exhausted);
  validation = BindContext();
  if (!validation.ok())
    return validation;
  ResetFrameState();
  frame_active_ = true;
  capture_sequence_ = capture_sequence;
  return {};
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::EnqueueUpload(
    const Binding &binding, CUdeviceptr destination,
    std::size_t destination_pitch) noexcept {
  Result validation = ValidateFrameBinding(binding);
  if (!validation.ok())
    return validation;
  if (resident_work_ || upload_enqueued_ || destination == 0 ||
      destination_pitch < upload_row_bytes_) {
    return Failed(Status::invalid_sequence);
  }
  CUDA_MEMCPY2D copy{};
  copy.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy.srcHost = upload_host_;
  copy.srcPitch = upload_row_bytes_;
  copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.dstDevice = destination;
  copy.dstPitch = destination_pitch;
  copy.WidthInBytes = upload_row_bytes_;
  copy.Height = upload_rows_;
  const CUresult result = cuda_->f().cuMemcpy2DAsync(&copy, binding_.stream);
  if (result != CUDA_SUCCESS)
    return FromCudaFailure(Status::copy_failed, result);
  upload_enqueued_ = true;
  work_submitted_ = true;
  ++counters_.frame_uploads;
  return {};
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::SubmitResidentWork(
    const Binding &binding) noexcept {
  Result validation = ValidateFrameBinding(binding);
  if (!validation.ok())
    return validation;
  if (!resident_work_ || work_submitted_)
    return Failed(Status::invalid_sequence);
  work_submitted_ = true;
  return {};
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::RecordStageBegin(
    const Binding &binding, std::uint32_t stage_index) noexcept {
  Result validation = ValidateFrameBinding(binding);
  if (!validation.ok())
    return validation;
  if (!work_submitted_ || readback_enqueued_ ||
      stage_index >= timed_stage_count_ || stage_index != next_stage_index_ ||
      stage_begin_recorded_[stage_index]) {
    return Failed(Status::invalid_sequence, CUDA_SUCCESS, stage_index);
  }
  const CUresult result = cuda_->f().cuEventRecord(
      stage_begin_events_[stage_index], binding_.stream);
  if (result != CUDA_SUCCESS) {
    Result failure = FromCudaFailure(Status::event_failed, result);
    failure.stage_index = stage_index;
    return failure;
  }
  stage_begin_recorded_[stage_index] = true;
  ++counters_.event_records;
  return {};
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::RecordStageEnd(
    const Binding &binding, std::uint32_t stage_index) noexcept {
  Result validation = ValidateFrameBinding(binding);
  if (!validation.ok())
    return validation;
  if (stage_index >= timed_stage_count_ || stage_index != next_stage_index_ ||
      !stage_begin_recorded_[stage_index] || stage_end_recorded_[stage_index]) {
    return Failed(Status::invalid_sequence, CUDA_SUCCESS, stage_index);
  }
  const CUresult result =
      cuda_->f().cuEventRecord(stage_end_events_[stage_index], binding_.stream);
  if (result != CUDA_SUCCESS) {
    Result failure = FromCudaFailure(Status::event_failed, result);
    failure.stage_index = stage_index;
    return failure;
  }
  stage_end_recorded_[stage_index] = true;
  ++next_stage_index_;
  ++counters_.event_records;
  return {};
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::EnqueueReadback(
    const Binding &binding, CUdeviceptr source,
    std::size_t source_pitch) noexcept {
  if (readback_count_ != 1)
    return Failed(Status::invalid_sequence);
  const Result result = EnqueueReadbackAt(binding, 0, source, source_pitch);
  if (result.ok())
    ++counters_.frame_final_readbacks;
  return result;
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::EnqueueReadbackAt(
    const Binding &binding, std::uint32_t index, CUdeviceptr source,
    std::size_t source_pitch) noexcept {
  Result validation = ValidateFrameBinding(binding);
  if (!validation.ok())
    return validation;
  if (!work_submitted_ || readback_enqueued_ || index != next_readback_index_ ||
      index >= readback_count_ || next_stage_index_ != timed_stage_count_ ||
      source == 0 || source_pitch < readbacks_[index].row_bytes) {
    return Failed(Status::invalid_sequence);
  }
  CUDA_MEMCPY2D copy{};
  copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.srcDevice = source;
  copy.srcPitch = source_pitch;
  copy.dstMemoryType = CU_MEMORYTYPE_HOST;
  const auto &descriptor = readbacks_[index];
  if (descriptor.source != 0 &&
      (source != descriptor.source || source_pitch != descriptor.source_pitch))
    return Failed(Status::invalid_sequence);
  copy.dstHost =
      static_cast<std::byte *>(readback_host_) + descriptor.host_offset_bytes;
  copy.dstPitch = descriptor.row_bytes;
  copy.WidthInBytes = descriptor.row_bytes;
  copy.Height = descriptor.rows;
  const CUresult result = cuda_->f().cuMemcpy2DAsync(&copy, binding_.stream);
  if (result != CUDA_SUCCESS)
    return FromCudaFailure(Status::copy_failed, result);
  ++next_readback_index_;
  readback_enqueued_ = next_readback_index_ == readback_count_;
  ++counters_.frame_readbacks;
  return {};
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::RecordCompletion(
    const Binding &binding) noexcept {
  Result validation = ValidateFrameBinding(binding);
  if (!validation.ok())
    return validation;
  if (!work_submitted_ || (!resident_work_ && !readback_enqueued_) ||
      failed_frame_completion_ || completion_recorded_)
    return Failed(Status::invalid_sequence);
  const CUresult result =
      cuda_->f().cuEventRecord(completion_event_, binding_.stream);
  if (result != CUDA_SUCCESS)
    return FromCudaFailure(Status::event_failed, result);
  completion_recorded_ = true;
  ++counters_.event_records;
  return {};
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::RecordFailedFrameCompletion(
    const Binding &binding) noexcept {
  const auto validation = ValidateFrameBinding(binding);
  if (!validation.ok())
    return validation;
  if (!work_submitted_)
    return Failed(Status::invalid_sequence);
  failed_frame_completion_ = true;
  frame_reuse_blocked_ = true;
  if (completion_recorded_)
    return {};
  const auto recorded =
      cuda_->f().cuEventRecord(completion_event_, binding_.stream);
  if (recorded != CUDA_SUCCESS)
    return FromCudaFailure(Status::event_failed, recorded);
  completion_recorded_ = true;
  ++counters_.event_records;
  return {};
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::WaitForCompletion(
    const Binding &binding, const WaitPolicy &policy) noexcept {
  Result validation = ValidateFrameBinding(binding);
  if (!validation.ok())
    return validation;
  if (failed_frame_completion_)
    return Failed(Status::invalid_sequence);
  if (!completion_recorded_ || wait_attempted_ ||
      policy.timeout_nanoseconds == 0 ||
      policy.poll_interval_nanoseconds == 0 || policy.maximum_poll_count == 0 ||
      ((policy.now_nanoseconds == nullptr) != (policy.poll_wait == nullptr))) {
    return Failed(Status::invalid_argument);
  }
  wait_attempted_ = true;
  ++counters_.frame_waits;

  const NowNanoseconds now =
      policy.now_nanoseconds ? policy.now_nanoseconds : DefaultNowNanoseconds;
  const PollWait poll_wait =
      policy.poll_wait ? policy.poll_wait : DefaultPollWait;
  const std::uint64_t started = now(policy.callback_context);
  std::uint32_t poll_count = 0;

  for (;;) {
    if (policy.stop_requested &&
        policy.stop_requested(policy.callback_context)) {
      frame_reuse_blocked_ = true;
      return Failed(Status::stop_requested);
    }
    if (policy.cancel_requested &&
        policy.cancel_requested(policy.callback_context)) {
      frame_reuse_blocked_ = true;
      return Failed(Status::cancelled);
    }

    const CUresult query = cuda_->f().cuEventQuery(completion_event_);
    ++counters_.event_queries;
    ++poll_count;
    if (query == CUDA_SUCCESS)
      break;
    if (query != CUDA_ERROR_NOT_READY)
      return FromCudaFailure(Status::event_failed, query);

    const std::uint64_t current = now(policy.callback_context);
    const std::uint64_t elapsed = current >= started ? current - started : 0;
    if (elapsed >= policy.timeout_nanoseconds ||
        poll_count >= policy.maximum_poll_count) {
      frame_reuse_blocked_ = true;
      return Failed(Status::timeout, CUDA_ERROR_NOT_READY);
    }
    const std::uint64_t remaining = policy.timeout_nanoseconds - elapsed;
    poll_wait(policy.callback_context,
              std::min(policy.poll_interval_nanoseconds, remaining));
  }

  for (std::uint32_t index = 0; index < timed_stage_count_; ++index) {
    if (!stage_begin_recorded_[index] || !stage_end_recorded_[index]) {
      frame_active_ = false;
      return Failed(Status::invalid_sequence, CUDA_SUCCESS, index);
    }
    float milliseconds = 0.0F;
    const CUresult elapsed = cuda_->f().cuEventElapsedTime(
        &milliseconds, stage_begin_events_[index], stage_end_events_[index]);
    ++counters_.elapsed_time_queries;
    if (elapsed != CUDA_SUCCESS || !std::isfinite(milliseconds) ||
        milliseconds < 0.0F) {
      frame_active_ = false;
      Result failure = elapsed == CUDA_SUCCESS
                           ? Failed(Status::event_failed, elapsed, index)
                           : FromCudaFailure(Status::event_failed, elapsed);
      failure.stage_index = index;
      return failure;
    }
    const long double nanoseconds =
        static_cast<long double>(milliseconds) * 1000000.0L;
    stage_elapsed_ns_[index] =
        nanoseconds >= static_cast<long double>(
                           std::numeric_limits<std::uint64_t>::max())
            ? std::numeric_limits<std::uint64_t>::max()
            : static_cast<std::uint64_t>(nanoseconds + 0.5L);
  }

  frame_active_ = false;
  frame_reuse_blocked_ = false;
  return {};
}

CudaPreparedTransferCompletion::Result
CudaPreparedTransferCompletion::Shutdown() noexcept {
  if (!prepared_) {
    shutdown_ = true;
    return {};
  }
  if (failed_frame_completion_ && !completion_recorded_)
    return Failed(Status::resource_exhausted);
  if (frame_active_ && work_submitted_ && !quarantined_) {
    if (!completion_recorded_)
      return Failed(Status::resource_exhausted);
    const CUresult query = cuda_->f().cuEventQuery(completion_event_);
    ++counters_.event_queries;
    if (query == CUDA_ERROR_NOT_READY)
      return Failed(Status::resource_exhausted, query);
    if (query != CUDA_SUCCESS && !IsCudaDeviceLossResult(query))
      return Failed(Status::event_failed, query);
    if (IsCudaDeviceLossResult(query))
      (void)FromCudaFailure(Status::event_failed, query);
  }
  return ReleaseCurrent(true);
}

std::uint64_t CudaPreparedTransferCompletion::stage_elapsed_nanoseconds(
    std::uint32_t stage_index) const noexcept {
  return stage_index < timed_stage_count_ ? stage_elapsed_ns_[stage_index] : 0;
}

} // namespace studiocast::cuda
