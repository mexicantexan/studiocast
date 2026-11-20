#include "core/open_video/prepared_open_video_compatibility_executor.h"

#include "core/open_video/fastdvdnet_denoiser.h"
#include "core/open_video/frame_analysis_cache.h"
#include "core/open_video/gaze_correction_eye_contact.h"
#include "core/open_video/yunet_face_detector.h"
#include "core/video/cpu_primitive_hal.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

namespace studiocast::open_video {
namespace {

void SetReason(hal::ReasonId *reason, hal::ReasonId value) noexcept {
  if (reason != nullptr)
    *reason = value;
}

bool SameImage(const hal::ImageDescriptor &left,
               const hal::ImageDescriptor &right) noexcept {
  if (left.width != right.width || left.height != right.height ||
      left.plane_count != right.plane_count || left.format != right.format ||
      left.channel_order != right.channel_order ||
      left.color_space != right.color_space ||
      left.color_range != right.color_range ||
      left.alpha_meaning != right.alpha_meaning ||
      left.orientation != right.orientation ||
      left.host_coherence != right.host_coherence ||
      left.mutability != right.mutability || left.aliasing != right.aliasing ||
      left.in_place_permitted != right.in_place_permitted ||
      left.base_alignment_bytes != right.base_alignment_bytes) {
    return false;
  }
  for (std::size_t index = 0; index < left.plane_count; ++index) {
    if (left.planes[index].offset_bytes != right.planes[index].offset_bytes ||
        left.planes[index].row_stride_bytes !=
            right.planes[index].row_stride_bytes ||
        left.planes[index].size_bytes != right.planes[index].size_bytes ||
        left.planes[index].width != right.planes[index].width ||
        left.planes[index].height != right.planes[index].height ||
        left.planes[index].alignment_bytes !=
            right.planes[index].alignment_bytes) {
      return false;
    }
  }
  return true;
}

// The compatibility stages and terminal CPU transport support packed RGB only.
bool SupportedRgbImage(const hal::ImageDescriptor &image,
                       bool output) noexcept {
  if (!hal::ValidateImageDescriptor(image) || image.plane_count != 1 ||
      image.format != hal::PixelFormat::rgb_u8 ||
      image.channel_order != hal::ChannelOrder::rgb ||
      image.color_space != hal::ColorSpace::srgb ||
      image.color_range != hal::ColorRange::full ||
      image.alpha_meaning != hal::AlphaMeaning::none ||
      image.orientation != hal::Orientation::top_left ||
      image.host_coherence != hal::HostCoherence::coherent ||
      image.base_alignment_bytes > alignof(std::max_align_t) ||
      image.width >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max() / 3) ||
      image.height >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      (image.mutability != hal::Mutability::read_write &&
       image.mutability != (output ? hal::Mutability::write_only
                                   : hal::Mutability::read_only))) {
    return false;
  }
  const auto &plane = image.planes[0];
  const auto maximum =
      static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
  const auto row_bytes = std::uint64_t{image.width} * 3;
  return plane.offset_bytes == 0 && plane.width == image.width &&
         plane.height == image.height && plane.size_bytes <= maximum &&
         plane.row_stride_bytes >= row_bytes &&
         plane.row_stride_bytes <= maximum / image.height &&
         image.height - 1 <= (maximum - row_bytes) / plane.row_stride_bytes &&
         plane.size_bytes >=
             plane.row_stride_bytes * (image.height - 1) + row_bytes;
}

bool SameCompletion(const hal::CompletionToken &left,
                    const hal::CompletionToken &right) noexcept {
  return left.token_id == right.token_id &&
         left.program_id == right.program_id &&
         left.adapter_instance_id == right.adapter_instance_id &&
         left.capture_sequence == right.capture_sequence &&
         left.complete == right.complete;
}

bool ProviderMatchesDomain(
    hal::SemanticEffectProviderId provider, hal::BackendId backend,
    const hal::ExecutionDomainId &domain) noexcept {
  if (backend != domain.backend)
    return false;
  const auto residency = hal::ClassifyExecutionResidency(domain);
  if (provider ==
      hal::SemanticEffectProviderId::open_video_onnxruntime_cpu) {
    return backend == hal::BackendId::open_video_onnxruntime &&
           residency == hal::ExecutionResidencyClass::host;
  }
  if (provider ==
      hal::SemanticEffectProviderId::open_video_onnxruntime_cuda) {
    return backend == hal::BackendId::cuda &&
           residency == hal::ExecutionResidencyClass::device;
  }
  return false;
}

std::uint64_t
Elapsed(const std::chrono::steady_clock::time_point &start) noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

} // namespace

struct PreparedOpenVideoCompatibilityExecutor::Impl {
  static constexpr std::size_t kPreparedErrorCapacity = 4096;

  PreparedOpenVideoCompatibilityConfig config{};
  FastDvdnetDenoiser denoiser;
  YunetFaceDetector yunet;
  GazeCorrectionEyeContact gaze;
  FrameAnalysisCache analysis;
  std::vector<std::uint8_t> output;
  std::vector<std::uint8_t> candidate;
  std::vector<std::uint8_t> writer_output;
  std::vector<std::uint8_t> resize_scratch;
  const video::CpuPrimitiveTable *cpu = nullptr;
  video::CpuRgb24BilinearResizePlan resize_plan{};
  std::array<std::string, hal::kMaxPreparedStages> frame_errors{};
  video::VideoExecutionAdapterSnapshot snapshot{};
  hal::CompletionToken pending{};
  std::array<hal::StageExecutionResult, hal::kMaxPreparedStages>
      pending_stages{};
  std::size_t pending_stage_count = 0;
  std::uint64_t next_token = 1;
  std::uint64_t commit_sequence = 0;
  bool have_denoise = false;
  bool have_eye_contact = false;
  bool invalidated = false;

  void ClearCurrentActivity() noexcept {
    snapshot.latest_committed_frame = {};
    snapshot.telemetry.active_backend = hal::BackendId::invalid;
    snapshot.telemetry.active_provider = hal::SemanticEffectProviderId::invalid;
  }

  bool Setup(PreparedOpenVideoCompatibilitySetupResult *result) {
    if (!SupportedRgbImage(config.plan.host_input, false) ||
        !SupportedRgbImage(config.plan.host_output, true) ||
        config.plan.intent.stage_count == 0 ||
        config.plan.intent.stage_count > hal::kMaxPreparedStages) {
      result->status = hal::StatusId::blocked;
      result->reason = hal::ReasonId::malformed_contract;
      return false;
    }
    const int width = static_cast<int>(config.plan.host_input.width);
    const int height = static_cast<int>(config.plan.host_input.height);
    if (config.selected_backend == hal::BackendId::invalid ||
        config.selected_provider == hal::SemanticEffectProviderId::invalid ||
        !hal::ValidateExecutionDomain(config.execution_domain) ||
        !ProviderMatchesDomain(config.selected_provider,
                               config.selected_backend,
                               config.execution_domain)) {
      result->status = hal::StatusId::blocked;
      result->reason = hal::ReasonId::malformed_execution_domain;
      return false;
    }
    const auto &input = config.plan.host_input;
    const auto &writer = config.plan.host_output;
    const bool transport =
        input.width != writer.width || input.height != writer.height ||
        input.planes[0].row_stride_bytes != writer.planes[0].row_stride_bytes ||
        input.planes[0].size_bytes != writer.planes[0].size_bytes;
    // The CPU resizer clears complete destination rows, including final-row
    // padding. Own that physical storage even when a descriptor only exposes
    // its active final row, and never read that extra padding from the caller.
    const auto capture_bytes = static_cast<std::size_t>(
        std::max(input.planes[0].size_bytes,
                 input.planes[0].row_stride_bytes * input.height));
    const auto writer_bytes =
        transport ? static_cast<std::size_t>(std::max(
                        writer.planes[0].size_bytes,
                        writer.planes[0].row_stride_bytes * writer.height))
                  : 0;
    std::size_t scratch_bytes = 0;
    if (transport) {
      cpu = &video::GetCpuPrimitiveTable();
      if (!cpu->resize_rgb24_bilinear_plan_scratch_bytes ||
          !cpu->prepare_resize_rgb24_bilinear ||
          !cpu->apply_resize_rgb24_bilinear) {
        result->status = hal::StatusId::blocked;
        result->reason = hal::ReasonId::capability_evidence_incomplete;
        return false;
      }
      scratch_bytes = cpu->resize_rgb24_bilinear_plan_scratch_bytes(
          width, height, static_cast<int>(writer.width),
          static_cast<int>(writer.height));
    }
    // Reserve capture ping-pong storage and the terminal writer transport
    // without enlarging the caller's prepared CPU compatibility reservation.
    const auto &budget = config.plan.budget;
    if (capture_bytes > budget.persistent_bytes / 2 ||
        writer_bytes > budget.persistent_bytes - 2 * capture_bytes ||
        scratch_bytes > budget.scratch_bytes ||
        (transport && scratch_bytes == 0) ||
        budget.resource_count < (transport ? 4u : 2u) ||
        budget.in_flight_slots != 1) {
      result->status = hal::StatusId::resource_exhausted;
      result->reason = hal::ReasonId::resource_exhausted;
      return false;
    }
    output.resize(capture_bytes);
    candidate.resize(capture_bytes);
    if (transport) {
      writer_output.resize(writer_bytes);
      resize_scratch.resize(scratch_bytes);
      if (cpu->prepare_resize_rgb24_bilinear(
              width, height, static_cast<int>(writer.width),
              static_cast<int>(writer.height), resize_scratch.data(),
              resize_scratch.size(),
              &resize_plan) != video::CpuPrimitiveResult::success) {
        result->status = hal::StatusId::blocked;
        result->reason = hal::ReasonId::malformed_contract;
        return false;
      }
    }
    for (std::size_t index = 0; index < config.plan.intent.stage_count;
         ++index) {
      const auto semantic = config.plan.intent.stages[index].intent.semantic;
      frame_errors[index].reserve(kPreparedErrorCapacity);
      std::string error;
      if (semantic == hal::SemanticEffectId::video_noise_removal) {
        if (config.prepared_stage_runner == nullptr &&
            !denoiser.EnsureInitialized(
                width, height, config.effects.video_noise_removal.model_id,
                &error)) {
          result->status = hal::StatusId::unavailable;
          result->reason = hal::ReasonId::capability_evidence_incomplete;
          return false;
        }
        have_denoise = true;
      } else if (semantic == hal::SemanticEffectId::eye_contact) {
        if (config.prepared_stage_runner == nullptr &&
            (!yunet.EnsureInitialized(config.effects.auto_frame.model_id,
                                      &error) ||
             !gaze.EnsureInitialized(config.effects.eye_contact.model_id,
                                     &error))) {
          result->status = hal::StatusId::unavailable;
          result->reason = hal::ReasonId::capability_evidence_incomplete;
          return false;
        }
        if (config.prepared_stage_runner == nullptr &&
            !gaze.PrepareScratch(width, height, 64, 48, 64, 48, &error)) {
          result->status = hal::StatusId::resource_exhausted;
          result->reason = hal::ReasonId::resource_exhausted;
          return false;
        }
        have_eye_contact = true;
      } else if (config.prepared_stage_runner == nullptr ||
                 (semantic !=
                      hal::SemanticEffectId::virtual_background_replace &&
                  semantic != hal::SemanticEffectId::virtual_key_light &&
                  semantic != hal::SemanticEffectId::auto_frame)) {
        result->status = hal::StatusId::blocked;
        result->reason = hal::ReasonId::semantic_effect_unsupported;
        return false;
      }
      // CPU-tensor Open Video stages remain explicitly host-provider stages
      // even when ORT internally activates CUDA EP.
      if (config.providers[index] != config.selected_provider ||
          config.backends[index] != config.selected_backend ||
          !ProviderMatchesDomain(config.providers[index],
                                 config.backends[index],
                                 config.execution_domain)) {
        result->status = hal::StatusId::blocked;
        result->reason = hal::ReasonId::semantic_provider_unavailable;
        return false;
      }
    }
    const bool host_provider =
        config.selected_provider ==
        hal::SemanticEffectProviderId::open_video_onnxruntime_cpu;
    snapshot.status =
        host_provider ? hal::StatusId::degraded : hal::StatusId::ready;
    snapshot.reason = host_provider
                          ? (config.setup_fallback_reason == hal::ReasonId::none
                                 ? hal::ReasonId::cpu_tail_required
                                 : config.setup_fallback_reason)
                          : hal::ReasonId::none;
    snapshot.plan_id = config.plan.plan_id;
    snapshot.prepared = true;
    snapshot.readiness.status = snapshot.status;
    snapshot.readiness.reason = snapshot.reason;
    snapshot.readiness.requested_backend = config.plan.intent.backend.backend;
    snapshot.readiness.selected_backend = config.selected_backend;
    snapshot.readiness.provider = config.selected_provider;
    snapshot.readiness.execution_domain = config.execution_domain;
    snapshot.readiness.available = true;
    snapshot.readiness.degraded = host_provider;
    snapshot.readiness.ready_stage_mask =
        config.plan.intent.stage_count >= 32
            ? UINT32_MAX
            : ((UINT32_C(1) << config.plan.intent.stage_count) - 1u);
    snapshot.telemetry.requested_backend = config.plan.intent.backend.backend;
    snapshot.execution_domain = config.execution_domain;
    snapshot.telemetry.selected_backend = config.selected_backend;
    snapshot.telemetry.runtime_status = snapshot.status;
    snapshot.telemetry.runtime_reason = snapshot.reason;
    result->status = snapshot.status;
    result->reason = snapshot.reason;
    result->readiness = snapshot.readiness;
    return true;
  }
};

PreparedOpenVideoCompatibilityExecutor::PreparedOpenVideoCompatibilityExecutor()
    : impl_(std::make_unique<Impl>()) {}

PreparedOpenVideoCompatibilityExecutor::
    ~PreparedOpenVideoCompatibilityExecutor() = default;

std::unique_ptr<PreparedOpenVideoCompatibilityExecutor>
PreparedOpenVideoCompatibilityExecutor::Create(
    const PreparedOpenVideoCompatibilityConfig &config,
    PreparedOpenVideoCompatibilitySetupResult *result) noexcept {
  PreparedOpenVideoCompatibilitySetupResult local;
  if (result == nullptr)
    result = &local;
  *result = {};
  try {
    auto executor = std::unique_ptr<PreparedOpenVideoCompatibilityExecutor>(
        new PreparedOpenVideoCompatibilityExecutor());
    executor->impl_->config = config;
    if (!executor->impl_->Setup(result))
      return nullptr;
    return executor;
  } catch (const std::bad_alloc &) {
    result->status = hal::StatusId::resource_exhausted;
    result->reason = hal::ReasonId::resource_exhausted;
  } catch (...) {
    result->status = hal::StatusId::blocked;
    result->reason = hal::ReasonId::capability_evidence_incomplete;
  }
  return nullptr;
}

video::VideoExecutionFrameResult
PreparedOpenVideoCompatibilityExecutor::Execute(
    const hal::HostFrameBinding &binding) noexcept {
  video::VideoExecutionFrameResult result;
  if (!impl_ || impl_->snapshot.stopped) {
    result.status = hal::StatusId::stopped;
    result.reason = hal::ReasonId::shutdown_requested;
    return result;
  }
  if (impl_->invalidated) {
    result.status = hal::StatusId::stale;
    result.reason = hal::ReasonId::reconfiguration_required;
    return result;
  }
  if (impl_->pending.token_id != 0) {
    result.status = hal::StatusId::blocked;
    result.reason = hal::ReasonId::frame_order_violated;
    return result;
  }
  if (binding.input_plane_count != 1 ||
      binding.input_planes[0].read_data == nullptr ||
      binding.capture_sequence == 0 ||
      !SameImage(binding.input_descriptor, impl_->config.plan.host_input) ||
      binding.input_planes[0].span_bytes <
          impl_->config.plan.host_input.planes[0].size_bytes ||
      binding.input_planes[0].row_stride_bytes !=
          impl_->config.plan.host_input.planes[0].row_stride_bytes) {
    result.status = hal::StatusId::blocked;
    result.reason = hal::ReasonId::host_frame_binding_invalid;
    return result;
  }

  const auto frame_start = std::chrono::steady_clock::now();
  std::memcpy(impl_->output.data(), binding.input_planes[0].read_data,
              static_cast<std::size_t>(
                  impl_->config.plan.host_input.planes[0].size_bytes));
  impl_->analysis.BeginFrame(binding.capture_sequence);
  bool degraded = false;
  result.stage_count = impl_->config.plan.intent.stage_count;
  for (std::size_t index = 0; index < result.stage_count; ++index) {
    const auto &stage = impl_->config.plan.intent.stages[index];
    auto &stage_result = result.stages[index];
    stage_result.stage_id = stage.intent.stage_id;
    stage_result.semantic = stage.intent.semantic;
    const auto start = std::chrono::steady_clock::now();
    std::memcpy(impl_->candidate.data(), impl_->output.data(),
                impl_->output.size());
    bool ok = false;
    auto &error = impl_->frame_errors[index];
    error.clear();
    if (impl_->config.prepared_stage_runner != nullptr) {
      ok = impl_->config.prepared_stage_runner(
          impl_->config.prepared_stage_runner_context, stage,
          binding.capture_sequence, impl_->candidate.data(),
          static_cast<int>(impl_->config.plan.host_input.width),
          static_cast<int>(impl_->config.plan.host_input.height),
          static_cast<std::size_t>(
              impl_->config.plan.host_input.planes[0].row_stride_bytes),
          &error);
    } else if (stage.intent.semantic ==
               hal::SemanticEffectId::video_noise_removal) {
      ok = impl_->denoiser.ApplyPreparedRgbInPlace(
          binding.capture_sequence, impl_->candidate.data(),
          static_cast<int>(impl_->config.plan.host_input.width),
          static_cast<int>(impl_->config.plan.host_input.height),
          static_cast<std::size_t>(
              impl_->config.plan.host_input.planes[0].row_stride_bytes),
          impl_->config.effects.video_noise_removal.strength,
          impl_->config.effects.video_noise_removal.model_id, &error);
    } else if (stage.intent.semantic == hal::SemanticEffectId::eye_contact) {
      ok = impl_->gaze.ApplyPreparedRgbInPlace(
          binding.capture_sequence, impl_->candidate.data(),
          static_cast<int>(impl_->config.plan.host_input.width),
          static_cast<int>(impl_->config.plan.host_input.height),
          static_cast<std::size_t>(
              impl_->config.plan.host_input.planes[0].row_stride_bytes),
          impl_->config.effects.eye_contact.strength,
          impl_->config.effects.eye_contact.look_away_enabled,
          impl_->config.effects.auto_frame.model_id,
          impl_->config.effects.eye_contact.model_id, &impl_->yunet,
          &impl_->analysis, &error);
    }
    stage_result.elapsed_ns = Elapsed(start);
    if (ok) {
      impl_->output.swap(impl_->candidate);
      stage_result.status = hal::StageExecutionStatus::succeeded;
      stage_result.valid_output_preserved = true;
    } else if (!stage.intent.required &&
               stage.intent.failure_action ==
                   hal::FailureAction::fail_open_preserve_input) {
      stage_result.status = hal::StageExecutionStatus::optional_failed_open;
      stage_result.reason = hal::ReasonId::optional_stage_failed_open;
      stage_result.valid_output_preserved = true;
      ++impl_->snapshot.telemetry.optional_fail_open_stages;
      degraded = true;
    } else {
      stage_result.status = hal::StageExecutionStatus::required_failed;
      stage_result.reason = hal::ReasonId::required_stage_failed;
      ++impl_->snapshot.telemetry.required_stage_failures;
      result.status = hal::StatusId::failed;
      result.reason = hal::ReasonId::required_stage_failed;
      impl_->ClearCurrentActivity();
      return result;
    }
    impl_->pending_stages[index] = stage_result;
  }

  // Effects remain at capture geometry; this is only the prepared terminal
  // host-output resize/stride tail, not an additional semantic effect provider.
  if (impl_->cpu &&
      impl_->cpu->apply_resize_rgb24_bilinear(
          &impl_->resize_plan, impl_->output.data(),
          static_cast<std::size_t>(
              impl_->config.plan.host_input.planes[0].row_stride_bytes),
          impl_->writer_output.data(),
          static_cast<std::size_t>(
              impl_->config.plan.host_output.planes[0].row_stride_bytes)) !=
          video::CpuPrimitiveResult::success) {
    impl_->ClearCurrentActivity();
    result.status = hal::StatusId::failed;
    result.reason = hal::ReasonId::malformed_contract;
    return result;
  }
  auto &published = impl_->cpu ? impl_->writer_output : impl_->output;
  impl_->pending_stage_count = result.stage_count;
  impl_->pending = {
      impl_->next_token++,
      impl_->config.plan.plan_id,
      impl_->config.execution_domain.adapter_instance_id,
      binding.capture_sequence,
      true,
  };
  result.completion = impl_->pending;
  result.execution_domain = impl_->config.execution_domain;
  result.output.descriptor = impl_->config.plan.host_output;
  result.output.plane_count = 1;
  result.output.planes[0].read_data =
      reinterpret_cast<const std::byte *>(published.data());
  result.output.planes[0].write_data =
      reinterpret_cast<std::byte *>(published.data());
  result.output.planes[0].span_bytes =
      impl_->config.plan.host_output.planes[0].size_bytes;
  result.output.planes[0].row_stride_bytes =
      impl_->config.plan.host_output.planes[0].row_stride_bytes;
  result.output.capture_sequence = binding.capture_sequence;
  result.output.output_ready = true;
  result.output_ready = true;
  result.transfers = impl_->config.expected_transfers;
  result.frame_elapsed_ns = Elapsed(frame_start);
  result.status = degraded ? hal::StatusId::degraded : hal::StatusId::executed;
  result.reason = degraded ? hal::ReasonId::optional_stage_failed_open
                           : hal::ReasonId::none;
  ++impl_->snapshot.telemetry.execution_attempts;
  if (!degraded)
    ++impl_->snapshot.telemetry.execution_successes;
  impl_->snapshot.telemetry.transfers.uploads += result.transfers.uploads;
  impl_->snapshot.telemetry.transfers.final_readbacks +=
      result.transfers.final_readbacks;
  impl_->snapshot.telemetry.transfers.intermediate_readbacks +=
      result.transfers.intermediate_readbacks;
  impl_->snapshot.telemetry.transfers.intermediate_uploads +=
      result.transfers.intermediate_uploads;
  impl_->snapshot.telemetry.transfers.synchronizations +=
      result.transfers.synchronizations;
  return result;
}

hal::StatusId PreparedOpenVideoCompatibilityExecutor::CommitOutput(
    const hal::CompletionToken &completion, bool output_write_succeeded,
    hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  if (!impl_ || impl_->pending.token_id == 0 ||
      !SameCompletion(completion, impl_->pending)) {
    SetReason(reason, hal::ReasonId::output_not_committed);
    return hal::StatusId::blocked;
  }
  // Only an exact pending completion may replace current publication evidence.
  impl_->ClearCurrentActivity();
  auto &committed = impl_->snapshot.latest_committed_frame;
  if (!output_write_succeeded) {
    impl_->pending = {};
    SetReason(reason, hal::ReasonId::output_write_failed);
    return hal::StatusId::failed;
  }
  committed.valid = true;
  committed.program_id = impl_->config.plan.plan_id;
  committed.invalidation_key = impl_->config.plan.invalidation_key;
  committed.execution_domain = impl_->config.execution_domain;
  committed.capture_sequence = completion.capture_sequence;
  committed.output_commit_sequence = ++impl_->commit_sequence;
  committed.stage_count = impl_->pending_stage_count;
  for (std::size_t index = 0; index < committed.stage_count; ++index) {
    committed.stages[index].stage_id = impl_->pending_stages[index].stage_id;
    committed.stages[index].semantic = impl_->pending_stages[index].semantic;
    committed.stages[index].provider = impl_->config.providers[index];
    committed.stages[index].backend = impl_->config.backends[index];
    committed.stages[index].status = impl_->pending_stages[index].status;
    committed.stages[index].reason = impl_->pending_stages[index].reason;
  }
  ++impl_->snapshot.telemetry.committed_output_frames;
  ++impl_->snapshot.telemetry.active_backend_frames;
  impl_->snapshot.telemetry.active_backend = impl_->config.selected_backend;
  impl_->snapshot.telemetry.active_provider = impl_->config.selected_provider;
  impl_->pending = {};
  return hal::StatusId::ready;
}

void PreparedOpenVideoCompatibilityExecutor::Invalidate(
    hal::ReasonId reason) noexcept {
  if (!impl_)
    return;
  impl_->invalidated = true;
  impl_->pending = {};
  impl_->ClearCurrentActivity();
  impl_->snapshot.status = hal::StatusId::stale;
  impl_->snapshot.reason = reason;
  impl_->snapshot.prepared = false;
  impl_->snapshot.readiness.status = impl_->snapshot.status;
  impl_->snapshot.readiness.reason = reason;
  impl_->snapshot.readiness.available = impl_->snapshot.readiness.degraded =
      false;
}

hal::StatusId
PreparedOpenVideoCompatibilityExecutor::ResetTemporal(hal::ReasonId) noexcept {
  if (!impl_ || impl_->snapshot.stopped)
    return hal::StatusId::stopped;
  impl_->denoiser.ResetTemporalState();
  impl_->analysis.BeginFrame(impl_->analysis.capture_sequence + 1);
  ++impl_->snapshot.telemetry.temporal_resets;
  return hal::StatusId::ready;
}

void PreparedOpenVideoCompatibilityExecutor::Shutdown() noexcept {
  if (!impl_ || impl_->snapshot.stopped)
    return;
  impl_->pending = {};
  impl_->ClearCurrentActivity();
  impl_->snapshot.status = hal::StatusId::stopped;
  impl_->snapshot.reason = hal::ReasonId::shutdown_requested;
  impl_->snapshot.prepared = false;
  impl_->snapshot.stopped = true;
  impl_->snapshot.readiness.status = impl_->snapshot.status;
  impl_->snapshot.readiness.reason = impl_->snapshot.reason;
  impl_->snapshot.readiness.available = impl_->snapshot.readiness.degraded =
      false;
}

video::VideoExecutionAdapterSnapshot
PreparedOpenVideoCompatibilityExecutor::Snapshot() const noexcept {
  return impl_ ? impl_->snapshot : video::VideoExecutionAdapterSnapshot{};
}

} // namespace studiocast::open_video
