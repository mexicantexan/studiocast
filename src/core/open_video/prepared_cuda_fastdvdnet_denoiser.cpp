#include "core/open_video/prepared_cuda_fastdvdnet_denoiser.h"
#include "core/open_video/native_prepared_cuda_fastdvdnet_runtime.h"

#include <algorithm>
#include <cstring>

namespace studiocast::open_video {
namespace {

bool NameIs(const char *actual, const char *expected) noexcept {
  return actual && std::strcmp(actual, expected) == 0;
}

bool BudgetFits(const hal::ResourceBudget &need,
                const hal::ResourceBudget &have) noexcept {
  return need.persistent_bytes <= have.persistent_bytes &&
         need.scratch_bytes <= have.scratch_bytes &&
         need.upload_staging_bytes == 0 && need.readback_staging_bytes == 0 &&
         need.resource_count <= have.resource_count &&
         need.in_flight_slots == 1 && have.in_flight_slots >= 1;
}

bool ExactContract(const PreparedCudaFastDvdnetConfig &c) noexcept {
  const auto w = PreparedCudaFastDvdnetDenoiser::PaddedExtent(c.source_width);
  const auto h = PreparedCudaFastDvdnetDenoiser::PaddedExtent(c.source_height);
  return c.model_pack && c.model_pack->task == "video_denoise" &&
         c.artifact.Valid() && c.configuration_generation != 0 &&
         c.model_generation != 0 && c.source_width != 0 && c.source_height != 0 &&
         NameIs(c.tensors.noisy_name, "noisy") &&
         NameIs(c.tensors.noise_map_name, "noise_map") &&
         NameIs(c.tensors.denoised_name, "denoised") && c.tensors.float32 &&
         c.tensors.nchw &&
         c.tensors.noisy_shape == std::array<std::int64_t, 4>{1, 15, h, w} &&
         c.tensors.noise_map_shape == std::array<std::int64_t, 4>{1, 1, h, w} &&
         c.tensors.denoised_shape == std::array<std::int64_t, 4>{1, 3, h, w};
}

hal::ReasonId FailureReason(PreparedCudaFastDvdnetStatus status) noexcept {
  switch (status) {
  case PreparedCudaFastDvdnetStatus::wrong_provider:
    return hal::ReasonId::strict_backend_unavailable;
  case PreparedCudaFastDvdnetStatus::wrong_execution_domain:
    return hal::ReasonId::wrong_execution_domain;
  case PreparedCudaFastDvdnetStatus::stale_generation:
  case PreparedCudaFastDvdnetStatus::stale_artifact:
    return hal::ReasonId::reconfiguration_required;
  case PreparedCudaFastDvdnetStatus::resource_exhausted:
    return hal::ReasonId::resource_exhausted;
  case PreparedCudaFastDvdnetStatus::device_lost:
  case PreparedCudaFastDvdnetStatus::quarantined:
    return hal::ReasonId::device_lost;
  case PreparedCudaFastDvdnetStatus::synchronization_failed:
  case PreparedCudaFastDvdnetStatus::completion_pending:
    return hal::ReasonId::synchronization_failed;
  case PreparedCudaFastDvdnetStatus::shutdown:
  case PreparedCudaFastDvdnetStatus::stop_requested:
    return hal::ReasonId::shutdown_requested;
  default:
    return hal::ReasonId::required_stage_failed;
  }
}

} // namespace

PreparedCudaFastDvdnetDenoiser::PreparedCudaFastDvdnetDenoiser(
    IPreparedCudaFastDvdnetRuntime *runtime) noexcept : runtime_(runtime) {}

PreparedCudaFastDvdnetDenoiser::PreparedCudaFastDvdnetDenoiser()
    : owned_runtime_(std::make_unique<NativePreparedCudaFastDvdnetRuntime>()),
      runtime_(owned_runtime_.get()) {}

std::uint32_t
PreparedCudaFastDvdnetDenoiser::PaddedExtent(std::uint32_t extent) noexcept {
  return extent > UINT32_MAX - 3u ? 0u : (extent + 3u) & ~3u;
}

float PreparedCudaFastDvdnetDenoiser::NoiseSigmaOver255(int strength) noexcept {
  return (55.0f * static_cast<float>(std::clamp(strength, 0, 100)) / 100.0f) /
         255.0f;
}

PreparedCudaFastDvdnetPrepareResult
PreparedCudaFastDvdnetDenoiser::Prepare(
    const PreparedCudaFastDvdnetConfig &config) noexcept {
  PreparedCudaFastDvdnetPrepareResult result;
  if (!runtime_ || prepared_ || !ExactContract(config)) {
    result.status = PreparedCudaFastDvdnetStatus::malformed_contract;
    result.reason = hal::ReasonId::malformed_contract;
    return result;
  }
  if (!hal::ValidateExecutionDomain(config.execution_domain)) {
    result.status = PreparedCudaFastDvdnetStatus::wrong_execution_domain;
    result.reason = hal::ReasonId::wrong_execution_domain;
    return result;
  }
  const auto prepared = runtime_->Prepare(config);
  result.status = prepared.status;
  result.reserved_budget = prepared.reserved_budget;
  if (prepared.status == PreparedCudaFastDvdnetStatus::completion_pending &&
      prepared.runtime_handle != 0) {
    config_ = config;
    runtime_handle_ = prepared.runtime_handle;
    handle_ = {runtime_handle_, config.configuration_generation,
               config.model_generation};
    prepared_ = true;
    shutdown_requested_ = true;
    result.handle = handle_;
    result.reason = hal::ReasonId::synchronization_failed;
    return result;
  }
  const bool evidence_matches =
      prepared.ort_evidence.Valid() &&
      onnx::OrtPreparedProviderEvidenceMatches(
          prepared.ort_evidence.provider, config.cuda_device_id,
          config.execution_domain.logical_context_generation,
          config.execution_domain.queue_or_stream_identity);
  if (prepared.status != PreparedCudaFastDvdnetStatus::ready ||
      prepared.runtime_handle == 0 || !prepared.reusable_bindings ||
      !prepared.warmup_complete || !prepared.strict_cuda_only ||
      !prepared.user_stream_bound || !prepared.cpu_fallback_disabled ||
      !evidence_matches ||
      !BudgetFits(prepared.reserved_budget, config.available_budget)) {
    result.status = prepared.status != PreparedCudaFastDvdnetStatus::ready
                        ? prepared.status
                        : (!evidence_matches
                               ? PreparedCudaFastDvdnetStatus::wrong_provider
                               : PreparedCudaFastDvdnetStatus::malformed_contract);
    const auto rollback = prepared.runtime_handle == 0
                              ? PreparedCudaFastDvdnetShutdownSnapshot{}
                              : runtime_->Shutdown(prepared.runtime_handle);
    if (prepared.runtime_handle != 0 && !rollback.quiescent) {
      config_ = config;
      runtime_handle_ = prepared.runtime_handle;
      handle_ = {runtime_handle_, config.configuration_generation,
                 config.model_generation};
      prepared_ = true;
      shutdown_requested_ = true;
      result.status = PreparedCudaFastDvdnetStatus::completion_pending;
      result.handle = handle_;
    }
    result.reason = FailureReason(result.status);
    return result;
  }

  config_ = config;
  runtime_handle_ = prepared.runtime_handle;
  handle_ = {runtime_handle_, config.configuration_generation,
             config.model_generation};
  prepared_ = true;
  result.handle = handle_;
  result.reason = hal::ReasonId::none;
  return result;
}

bool PreparedCudaFastDvdnetDenoiser::HandleMatches(
    const PreparedCudaFastDvdnetHandle &handle) const noexcept {
  return prepared_ && handle.id == handle_.id &&
         handle.configuration_generation == handle_.configuration_generation &&
         handle.model_generation == handle_.model_generation;
}

PreparedCudaFastDvdnetFrameResult PreparedCudaFastDvdnetDenoiser::Execute(
    const PreparedCudaFastDvdnetHandle &handle,
    const ValidatedModelArtifactIdentity &current_artifact,
    const hal::ExecutionDomainId &current_domain, std::uint64_t capture_sequence,
    const cuda::CudaImage &input, cuda::CudaImage *output, int strength,
    hal::FailureAction failure_action) noexcept {
  PreparedCudaFastDvdnetFrameResult result;
  if (shutdown_requested_) {
    result.status = PreparedCudaFastDvdnetStatus::shutdown;
    result.reason = hal::ReasonId::shutdown_requested;
    return result;
  }
  if (quarantined_) {
    result.status = PreparedCudaFastDvdnetStatus::quarantined;
    result.reason = hal::ReasonId::device_lost;
    return result;
  }
  if (!HandleMatches(handle)) {
    result.status = prepared_ ? PreparedCudaFastDvdnetStatus::foreign_handle
                              : PreparedCudaFastDvdnetStatus::not_prepared;
    result.reason = hal::ReasonId::stale_program;
    return result;
  }
  if (!(current_domain == config_.execution_domain)) {
    result.status = PreparedCudaFastDvdnetStatus::wrong_execution_domain;
    result.reason = hal::ReasonId::wrong_execution_domain;
    return result;
  }
  if (!ModelArtifactIdentityMatches(config_.artifact, current_artifact)) {
    result.status = PreparedCudaFastDvdnetStatus::stale_artifact;
    result.reason = hal::ReasonId::reconfiguration_required;
    return result;
  }
  if (capture_sequence == 0 || !input.Valid() || !output || !output->Valid() ||
      input.format != cuda::PixelFormatGpu::rgb_u8 ||
      output->format != cuda::PixelFormatGpu::rgb_u8 ||
      input.w != static_cast<int>(config_.source_width) ||
      input.h != static_cast<int>(config_.source_height) || input.w != output->w ||
      input.h != output->h || input.ptr == output->ptr) {
    result.status = PreparedCudaFastDvdnetStatus::malformed_contract;
    result.reason = hal::ReasonId::invalid_resource;
    return result;
  }

  PreparedCudaFastDvdnetTemporalPlan temporal;
  temporal.capture_sequence = capture_sequence;
  temporal.reset = !have_sequence_ || capture_sequence != last_capture_sequence_ + 1;
  if (temporal.reset)
    temporal.source_slots = {2, 2, 2, 2, 2};

  const auto frame = runtime_->Execute(runtime_handle_, temporal, input, output,
                                       NoiseSigmaOver255(strength));
  result.status = frame.status;
  result.failed_operation = frame.failed_operation;
  result.transfers = frame.transfers;
  result.stage_elapsed_ns = frame.stage_elapsed_ns;
  result.frame_elapsed_ns = frame.frame_elapsed_ns;
  result.output_valid = frame.output_valid;
  result.pending_work = frame.pending_work;
  result.temporal_reset = temporal.reset;

  if (frame.status == PreparedCudaFastDvdnetStatus::executed &&
      frame.output_valid && frame.transfers.uploads == 0 &&
      frame.transfers.final_readbacks == 0 &&
      frame.transfers.intermediate_readbacks == 0 &&
      frame.transfers.intermediate_uploads == 0) {
    have_sequence_ = true;
    last_capture_sequence_ = capture_sequence;
    result.reason = temporal.reset ? hal::ReasonId::temporal_discontinuity
                                   : hal::ReasonId::none;
    return result;
  }
  if (frame.status == PreparedCudaFastDvdnetStatus::device_lost) {
    quarantined_ = true;
  }
  result.reason = frame.status == PreparedCudaFastDvdnetStatus::executed
                      ? hal::ReasonId::transfer_contract_violated
                      : FailureReason(frame.status);
  result.output_valid = false;
  if (failure_action == hal::FailureAction::fail_open_preserve_input &&
      frame.status != PreparedCudaFastDvdnetStatus::device_lost &&
      frame.status != PreparedCudaFastDvdnetStatus::completion_pending &&
      !frame.pending_work) {
    result.status = PreparedCudaFastDvdnetStatus::optional_failed_open;
    result.reason = hal::ReasonId::optional_stage_failed_open;
    result.valid_input_preserved = true;
  } else {
    result.status = frame.status == PreparedCudaFastDvdnetStatus::completion_pending
                        ? frame.status
                        : PreparedCudaFastDvdnetStatus::required_failed;
  }
  return result;
}

PreparedCudaFastDvdnetStatus PreparedCudaFastDvdnetDenoiser::ResetTemporalState(
    const PreparedCudaFastDvdnetHandle &handle) noexcept {
  if (!HandleMatches(handle))
    return PreparedCudaFastDvdnetStatus::foreign_handle;
  const auto status = runtime_->ResetTemporalState(runtime_handle_);
  if (status == PreparedCudaFastDvdnetStatus::ready) {
    have_sequence_ = false;
    last_capture_sequence_ = 0;
  }
  return status;
}

void PreparedCudaFastDvdnetDenoiser::Clear() noexcept {
  config_ = {};
  handle_ = {};
  runtime_handle_ = 0;
  last_capture_sequence_ = 0;
  have_sequence_ = false;
  prepared_ = false;
  quarantined_ = false;
  shutdown_requested_ = false;
}

PreparedCudaFastDvdnetShutdownSnapshot PreparedCudaFastDvdnetDenoiser::Shutdown(
    const PreparedCudaFastDvdnetHandle &handle) noexcept {
  if (!HandleMatches(handle))
    return {PreparedCudaFastDvdnetStatus::foreign_handle, 0, shutdown_requested_,
            false};
  shutdown_requested_ = true;
  auto snapshot = runtime_->Shutdown(runtime_handle_);
  snapshot.requested = true;
  if (snapshot.quiescent && snapshot.status == PreparedCudaFastDvdnetStatus::shutdown)
    Clear();
  return snapshot;
}

PreparedCudaFastDvdnetRuntimeTelemetry
PreparedCudaFastDvdnetDenoiser::Telemetry() const noexcept {
  return runtime_ ? runtime_->Telemetry()
                  : PreparedCudaFastDvdnetRuntimeTelemetry{};
}

} // namespace studiocast::open_video
