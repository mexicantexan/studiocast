#include "core/maxine/hal_resident_adapter.h"
#include "core/maxine/production_resident_frame_executor.h"

#include <chrono>
#include <limits>

namespace studiocast::maxine {
namespace {

using hal::BackendId;
using hal::ReasonId;
using hal::SemanticEffectId;
using hal::SemanticEffectProviderId;
using hal::StageExecutionStatus;
using hal::StatusId;

constexpr std::uint64_t kInputCookieSalt = 0x4d4158494e45494eULL;
constexpr std::uint64_t kOutputCookieSalt = 0x4d4158494e454f55ULL;
constexpr std::uint64_t kHostBindingCookieSalt = 0x4d4158494e454842ULL;
constexpr std::uint64_t kHostInputTokenSalt = 0x4d4158494e454849ULL;
constexpr std::uint64_t kHostOutputTokenSalt = 0x4d4158494e45484fULL;

void SetReason(ReasonId *reason, ReasonId value) noexcept {
  if (reason)
    *reason = value;
}

bool BudgetContains(const hal::ResourceBudget &capacity,
                    const hal::ResourceBudget &required) noexcept {
  return capacity.persistent_bytes >= required.persistent_bytes &&
         capacity.scratch_bytes >= required.scratch_bytes &&
         capacity.upload_staging_bytes >= required.upload_staging_bytes &&
         capacity.readback_staging_bytes >= required.readback_staging_bytes &&
         capacity.resource_count >= required.resource_count &&
         capacity.in_flight_slots >= required.in_flight_slots;
}

bool Multiply(std::uint64_t left, std::uint64_t right,
              std::uint64_t *result) noexcept {
  if (!result ||
      (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)) {
    return false;
  }
  *result = left * right;
  return true;
}

bool ValidPointerSpan(const void *pointer, std::uint64_t size_bytes) noexcept {
  if (!pointer || size_bytes == 0 ||
      size_bytes > std::numeric_limits<std::uintptr_t>::max())
    return false;
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  return size_bytes - 1 <= std::numeric_limits<std::uintptr_t>::max() - address;
}

bool SameResource(const hal::ResourceHandle &left,
                  const hal::ResourceHandle &right) noexcept {
  return left.resource_id == right.resource_id &&
         left.allocator_owner_id == right.allocator_owner_id &&
         left.lifetime_generation == right.lifetime_generation &&
         left.memory_domain == right.memory_domain &&
         left.lifetime == right.lifetime &&
         left.synchronization_state == right.synchronization_state &&
         left.execution_domain == right.execution_domain &&
         left.size_bytes == right.size_bytes &&
         left.alignment_bytes == right.alignment_bytes &&
         left.mutable_resource == right.mutable_resource &&
         left.aliasable == right.aliasable &&
         left.adapter_opaque == right.adapter_opaque &&
         left.destruction_owner == right.destruction_owner &&
         left.destroy == right.destroy;
}

bool SameImageStorage(const hal::ImageDescriptor &left,
                      const hal::ImageDescriptor &right) noexcept {
  if (left.width != right.width || left.height != right.height ||
      left.plane_count != right.plane_count || left.format != right.format ||
      left.channel_order != right.channel_order ||
      left.color_space != right.color_space ||
      left.color_range != right.color_range ||
      left.alpha_meaning != right.alpha_meaning ||
      left.orientation != right.orientation ||
      left.host_coherence != right.host_coherence ||
      left.aliasing != right.aliasing ||
      left.base_alignment_bytes != right.base_alignment_bytes ||
      left.in_place_permitted != right.in_place_permitted) {
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

bool SameImageDescriptor(const hal::ImageDescriptor &left,
                         const hal::ImageDescriptor &right) noexcept {
  return SameImageStorage(left, right) && left.mutability == right.mutability;
}

std::uint32_t StageMask(std::size_t stage_count) noexcept {
  return stage_count == 32 ? std::numeric_limits<std::uint32_t>::max()
                           : (std::uint32_t{1} << stage_count) - 1u;
}

bool HasEvidence(const std::array<std::uint8_t, 32> &identity) noexcept {
  for (const auto value : identity) {
    if (value != 0)
      return true;
  }
  return false;
}

bool ModelEvidenceComplete(const hal::PreparedProgram &program,
                           const hal::PreparedStage &stage) noexcept {
  return program.invalidation_key.generations.model != 0 &&
         program.invalidation_key.model_artifact_hash != 0 &&
         stage.model.artifact_generation != 0 &&
         HasEvidence(stage.model.model_id_hash) &&
         HasEvidence(stage.model.artifact_set_hash);
}

bool ValidAuthoritativeSnapshot(
    const MaxineAuthoritativeTelemetrySnapshot &snapshot,
    const MaxineHalAdapterConfiguration &configuration) noexcept {
  return snapshot.version == kMaxineHalBindingVersion &&
         snapshot.logical_context_generation ==
             configuration.execution_domain.logical_context_generation &&
         snapshot.stream_identity ==
             configuration.execution_domain.queue_or_stream_identity;
}

bool CounterDelta(std::uint64_t before, std::uint64_t after,
                  std::uint64_t *delta) noexcept {
  if (!delta || after < before)
    return false;
  *delta = after - before;
  return true;
}

bool TelemetryDelta(const MaxineAuthoritativeTelemetrySnapshot &before,
                    const MaxineAuthoritativeTelemetrySnapshot &after,
                    hal::TransferCounters *delta) noexcept {
  if (!delta)
    return false;
  *delta = {};
  std::uint64_t cpu_tail_readbacks = 0;
  if (!CounterDelta(before.host_uploads, after.host_uploads, &delta->uploads) ||
      !CounterDelta(before.final_readbacks, after.final_readbacks,
                    &delta->final_readbacks) ||
      !CounterDelta(before.cpu_tail_readbacks, after.cpu_tail_readbacks,
                    &cpu_tail_readbacks) ||
      !CounterDelta(before.synchronizations, after.synchronizations,
                    &delta->synchronizations) ||
      !CounterDelta(before.intermediate_uploads, after.intermediate_uploads,
                    &delta->intermediate_uploads) ||
      !CounterDelta(before.intermediate_readbacks, after.intermediate_readbacks,
                    &delta->intermediate_readbacks) ||
      delta->final_readbacks >
          std::numeric_limits<std::uint64_t>::max() - cpu_tail_readbacks) {
    return false;
  }
  delta->final_readbacks += cpu_tail_readbacks;
  return true;
}

bool ValidResidentIo(const hal::StageIoDescriptor &io,
                     const ResidentFrameKey &key,
                     hal::Mutability mutability) noexcept {
  if (io.kind != hal::StageIoKind::image)
    return false;
  const auto &image = io.image;
  std::uint64_t minimum_size = 0;
  const auto &plane = image.planes[0];
  return hal::ValidateImageDescriptor(image) && image.width == key.width &&
         image.height == key.height && image.plane_count == 1 &&
         image.format == hal::PixelFormat::bgr_u8 &&
         image.channel_order == hal::ChannelOrder::bgr &&
         image.orientation == hal::Orientation::top_left &&
         image.host_coherence == hal::HostCoherence::not_host_visible &&
         image.mutability == mutability &&
         image.aliasing == hal::AliasingPermission::forbidden &&
         !image.in_place_permitted && plane.offset_bytes == 0 &&
         plane.width == key.width && plane.height == key.height &&
         plane.row_stride_bytes >= static_cast<std::uint64_t>(key.width) * 3u &&
         Multiply(plane.row_stride_bytes, plane.height, &minimum_size) &&
         plane.size_bytes >= minimum_size;
}

bool ValidHostBoundary(const hal::ImageDescriptor &image,
                       const ResidentFrameKey &key, hal::PixelFormat format,
                       hal::ChannelOrder channel_order,
                       hal::Mutability mutability) noexcept {
  const auto &plane = image.planes[0];
  std::uint64_t minimum_size = 0;
  return hal::ValidateImageDescriptor(image) && image.width == key.width &&
         image.height == key.height && image.plane_count == 1 &&
         image.format == format && image.channel_order == channel_order &&
         image.color_space == hal::ColorSpace::srgb &&
         image.color_range == hal::ColorRange::full &&
         image.alpha_meaning == hal::AlphaMeaning::none &&
         image.orientation == hal::Orientation::top_left &&
         image.host_coherence == hal::HostCoherence::coherent &&
         image.mutability == mutability &&
         image.aliasing == hal::AliasingPermission::forbidden &&
         !image.in_place_permitted && plane.offset_bytes == 0 &&
         plane.width == key.width && plane.height == key.height &&
         plane.row_stride_bytes >= static_cast<std::uint64_t>(key.width) * 3u &&
         Multiply(plane.row_stride_bytes, plane.height, &minimum_size) &&
         plane.size_bytes >= minimum_size;
}

bool MapStage(const hal::PreparedStage &stage, ResidentStageKind *kind,
              bool *requires_matte) noexcept {
  if (!kind || !requires_matte)
    return false;
  *requires_matte = false;
  switch (stage.semantic) {
  case SemanticEffectId::video_noise_removal:
    if (stage.provider != SemanticEffectProviderId::maxine_vfx)
      return false;
    *kind = ResidentStageKind::denoise;
    return true;
  case SemanticEffectId::eye_contact:
    if (stage.provider != SemanticEffectProviderId::maxine_ar)
      return false;
    *kind = ResidentStageKind::eye_contact;
    return true;
  case SemanticEffectId::virtual_background_blur:
    if (stage.provider != SemanticEffectProviderId::maxine_vfx)
      return false;
    *kind = ResidentStageKind::background_blur;
    *requires_matte = true;
    return true;
  case SemanticEffectId::virtual_background_remove:
    if (stage.provider != SemanticEffectProviderId::maxine_vfx)
      return false;
    *kind = ResidentStageKind::background_remove;
    *requires_matte = true;
    return true;
  case SemanticEffectId::virtual_background_replace:
    if (stage.provider != SemanticEffectProviderId::maxine_vfx)
      return false;
    *kind = ResidentStageKind::background_replace;
    *requires_matte = true;
    return true;
  case SemanticEffectId::virtual_key_light:
    if (stage.provider != SemanticEffectProviderId::maxine_vfx)
      return false;
    *kind = ResidentStageKind::relighting;
    *requires_matte = true;
    return true;
  case SemanticEffectId::auto_frame:
    if (stage.provider != SemanticEffectProviderId::maxine_ar)
      return false;
    *kind = ResidentStageKind::auto_frame;
    return true;
  case SemanticEffectId::vignette:
    if (stage.provider != SemanticEffectProviderId::maxine_cuda_utility)
      return false;
    *kind = ResidentStageKind::vignette;
    return true;
  case SemanticEffectId::mirror:
  case SemanticEffectId::invalid:
    return false;
  }
  return false;
}

SemanticEffectProviderId ExpectedProvider(SemanticEffectId semantic) noexcept {
  switch (semantic) {
  case SemanticEffectId::eye_contact:
  case SemanticEffectId::auto_frame:
    return SemanticEffectProviderId::maxine_ar;
  case SemanticEffectId::video_noise_removal:
  case SemanticEffectId::virtual_background_blur:
  case SemanticEffectId::virtual_background_remove:
  case SemanticEffectId::virtual_background_replace:
  case SemanticEffectId::virtual_key_light:
    return SemanticEffectProviderId::maxine_vfx;
  case SemanticEffectId::vignette:
    return SemanticEffectProviderId::maxine_cuda_utility;
  case SemanticEffectId::mirror:
  case SemanticEffectId::invalid:
    return SemanticEffectProviderId::invalid;
  }
  return SemanticEffectProviderId::invalid;
}

std::uint64_t MatteFingerprint(const hal::PreparedProgram &program,
                               const hal::PreparedStage &stage) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto mix = [&hash](std::uint8_t value) {
    hash ^= value;
    hash *= 1099511628211ULL;
  };
  const auto mix_u64 = [&mix](std::uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8)
      mix(static_cast<std::uint8_t>(value >> shift));
  };
  const auto &key = program.invalidation_key;
  mix_u64(key.canonical_plan_hash);
  mix_u64(key.model_artifact_hash);
  mix_u64(key.execution_domain_hash);
  mix_u64(key.generations.configuration);
  mix_u64(key.generations.geometry);
  mix_u64(key.generations.model);
  mix_u64(key.generations.provider);
  mix_u64(key.generations.device);
  mix_u64(key.generations.context);
  mix_u64(program.execution_domain.logical_context_generation);
  mix_u64(program.execution_domain.queue_or_stream_identity);
  mix_u64(program.execution_domain.adapter_instance_id);
  mix_u64(program.execution_domain.physical_device.vendor_id);
  mix_u64(program.execution_domain.physical_device.device_id);
  for (const auto byte : program.execution_domain.physical_device.stable_id)
    mix(byte);
  mix_u64(static_cast<std::uint64_t>(stage.provider));
  mix_u64(static_cast<std::uint64_t>(stage.backend));
  if (stage.input.kind == hal::StageIoKind::image) {
    mix_u64(stage.input.image.width);
    mix_u64(stage.input.image.height);
    mix_u64(static_cast<std::uint64_t>(stage.input.image.format));
    mix_u64(static_cast<std::uint64_t>(stage.input.image.channel_order));
  }
  for (const auto byte : stage.model.model_id_hash)
    mix(byte);
  for (const auto byte : stage.model.artifact_set_hash)
    mix(byte);
  mix_u64(stage.model.artifact_generation);
  return hash == 0 ? 1 : hash;
}

ReasonId CpuTailToResident(hal::CpuTailId id, CpuTailKind *kind) noexcept {
  if (!kind)
    return ReasonId::malformed_contract;
  switch (id) {
  case hal::CpuTailId::output_format_conversion:
  case hal::CpuTailId::writer_copy:
    *kind = CpuTailKind::output_format_conversion;
    return ReasonId::none;
  case hal::CpuTailId::auto_frame_tracking:
    *kind = CpuTailKind::auto_frame_tracking;
    return ReasonId::none;
  case hal::CpuTailId::provider_postprocess:
    *kind = CpuTailKind::incompatible_stage;
    return ReasonId::none;
  case hal::CpuTailId::cpu_primitive_mirror:
  case hal::CpuTailId::backend_resolved_output:
  case hal::CpuTailId::invalid:
    break;
  }
  return ReasonId::malformed_contract;
}

hal::ExecuteResult Failed(StatusId status, ReasonId reason) noexcept {
  hal::ExecuteResult result{};
  result.status = status;
  result.reason = reason;
  return result;
}

bool IsSuccessful(ResidentExecutionStatus status) noexcept {
  return status == ResidentExecutionStatus::resident_output ||
         status == ResidentExecutionStatus::cpu_visible_output ||
         status == ResidentExecutionStatus::cpu_tail_required;
}

class PrevalidatedExecutor final : public IResidentFrameExecutor {
public:
  PrevalidatedExecutor(IResidentFrameExecutor &delegate,
                       ResidentFrameKey prepared_key) noexcept
      : delegate_(delegate), prepared_key_(prepared_key) {}

  ResidentBoundaryResult
  Prepare(const ResidentFrameKey &key) noexcept override {
    return key == prepared_key_ ? ResidentBoundaryResult::success
                                : ResidentBoundaryResult::incompatible_output;
  }
  ResidentBoundaryResult
  StageRgbToBgr(const HostRgbFrameView &host,
                const ResidentFrameKey &key) noexcept override {
    return delegate_.StageRgbToBgr(host, key);
  }
  ResidentBoundaryResult
  UploadStagedBgr(const ResidentFrameKey &key,
                  ResidentImage &output) noexcept override {
    return delegate_.UploadStagedBgr(key, output);
  }
  ResidentBoundaryResult
  RunSharedMatte(const ResidentImage &current, std::uint64_t capture_sequence,
                 std::uint64_t matte_fingerprint,
                 ResidentMatte &output) noexcept override {
    const auto started = std::chrono::steady_clock::now();
    const auto result = delegate_.RunSharedMatte(current, capture_sequence,
                                                 matte_fingerprint, output);
    const auto finished = std::chrono::steady_clock::now();
    matte_elapsed_ns_ = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started)
            .count());
    return result;
  }
  ResidentBoundaryResult
  RunCompatibleStage(ResidentStageKind kind, const ResidentImage &current,
                     const ResidentMatte *matte,
                     ResidentImage &output) noexcept override {
    const auto started = std::chrono::steady_clock::now();
    const auto result =
        delegate_.RunCompatibleStage(kind, current, matte, output);
    const auto finished = std::chrono::steady_clock::now();
    if (stage_elapsed_count_ < stage_elapsed_ns_.size()) {
      stage_elapsed_ns_[stage_elapsed_count_++] = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(finished -
                                                               started)
              .count());
    }
    return result;
  }
  ResidentBoundaryResult
  DownloadToHost(ResidentReadbackBoundary boundary,
                 const ResidentImage &current) noexcept override {
    return delegate_.DownloadToHost(boundary, current);
  }
  ResidentBoundaryResult
  Synchronize(ResidentReadbackBoundary boundary,
              const ResidentFrameKey &key) noexcept override {
    return delegate_.Synchronize(boundary, key);
  }

  [[nodiscard]] std::uint64_t matte_elapsed_ns() const noexcept {
    return matte_elapsed_ns_;
  }
  [[nodiscard]] std::uint64_t
  stage_elapsed_ns(std::size_t index) const noexcept {
    return index < stage_elapsed_count_ ? stage_elapsed_ns_[index] : 0;
  }

private:
  IResidentFrameExecutor &delegate_;
  ResidentFrameKey prepared_key_{};
  std::array<std::uint64_t, kMaxResidentFrameStages> stage_elapsed_ns_{};
  std::size_t stage_elapsed_count_ = 0;
  std::uint64_t matte_elapsed_ns_ = 0;
};

} // namespace

MaxineAuthoritativeTelemetrySnapshot AuthoritativeSnapshotFromProduction(
    const ProductionResidentTelemetry &telemetry,
    std::uint64_t logical_context_generation,
    std::uint64_t stream_identity) noexcept {
  MaxineAuthoritativeTelemetrySnapshot snapshot{};
  snapshot.logical_context_generation = logical_context_generation;
  snapshot.stream_identity = stream_identity;
  snapshot.host_uploads = telemetry
                              .transfers[static_cast<std::size_t>(
                                  ProductionTransferKind::host_upload)]
                              .successes;
  snapshot.final_readbacks = telemetry
                                 .transfers[static_cast<std::size_t>(
                                     ProductionTransferKind::final_download)]
                                 .successes;
  snapshot.cpu_tail_readbacks =
      telemetry
          .transfers[static_cast<std::size_t>(
              ProductionTransferKind::cpu_continuation_download)]
          .successes;
  snapshot.synchronizations = telemetry.explicit_synchronizations.successes;
  return snapshot;
}

bool MaxineFrameOutputBinding::ValidHost() const noexcept {
  if (version != kMaxineHalBindingVersion || !host_bgr || width == 0 ||
      height == 0)
    return false;
  const std::uint64_t row_bytes = static_cast<std::uint64_t>(width) * 3u;
  std::uint64_t size_bytes = 0;
  return row_bytes <= std::numeric_limits<std::size_t>::max() &&
         stride_bytes >= static_cast<std::size_t>(row_bytes) &&
         Multiply(stride_bytes, height, &size_bytes) &&
         ValidPointerSpan(host_bgr, size_bytes);
}

bool MaxineFrameOutputBinding::ValidResident(
    const ResidentFrameKey &key) const noexcept {
  return version == kMaxineHalBindingVersion && resident.ValidFor(key);
}

MaxineHalResidentAdapter::MaxineHalResidentAdapter(
    MaxineHalAdapterConfiguration configuration,
    IResidentFrameExecutor &executor, MaxineHalAdapterHooks hooks) noexcept
    : configuration_(configuration), executor_(&executor), hooks_(hooks) {}

std::uint64_t MaxineHalResidentAdapter::adapter_instance_id() const noexcept {
  return configuration_.adapter_instance_id;
}

BackendId MaxineHalResidentAdapter::backend_id() const noexcept {
  return BackendId::maxine;
}

bool MaxineHalResidentAdapter::ConfigurationValid() const noexcept {
  const auto &domain = configuration_.execution_domain;
  return executor_ && configuration_.adapter_instance_id != 0 &&
         configuration_.adapter_instance_id == domain.adapter_instance_id &&
         hal::ValidateExecutionDomain(domain) &&
         domain.backend == BackendId::maxine &&
         domain.api_family == hal::ApiFamily::maxine_nvcv &&
         domain.allocator_family == hal::AllocatorFamily::maxine_nvcv &&
         domain.memory_handle_family == hal::MemoryHandleFamily::nvcv_image &&
         domain.ownership_model == hal::OwnershipModel::adapter_exclusive &&
         domain.synchronization_protocol ==
             hal::SynchronizationProtocol::maxine_bound_cuda_stream &&
         configuration_.resident_key.Valid() &&
         configuration_.resident_key.stream_identity ==
             domain.queue_or_stream_identity &&
         configuration_.resident_key.configuration_generation != 0 &&
         configuration_.capacity.resource_count != 0 &&
         configuration_.capacity.in_flight_slots == 1 &&
         configuration_.output_format_converter != nullptr;
}

bool MaxineHalResidentAdapter::ProgramToResidentPlan(
    const hal::PreparedProgram &program, ResidentFramePlan *plan,
    ReasonId *reason) const noexcept {
  if (!plan) {
    SetReason(reason, ReasonId::malformed_contract);
    return false;
  }
  *plan = {};
  if (program.stage_count > kMaxResidentFrameStages ||
      program.requires_upload == false ||
      program.invalidation_key.generations.configuration !=
          configuration_.resident_key.configuration_generation) {
    SetReason(reason, ReasonId::malformed_contract);
    return false;
  }
  if (!ValidHostBoundary(program.host_input, configuration_.resident_key,
                         hal::PixelFormat::rgb_u8, hal::ChannelOrder::rgb,
                         hal::Mutability::read_only) ||
      !ValidHostBoundary(program.host_output, configuration_.resident_key,
                         hal::PixelFormat::rgb_u8, hal::ChannelOrder::rgb,
                         hal::Mutability::write_only)) {
    SetReason(reason, ReasonId::malformed_descriptor);
    return false;
  }
  // RGB host input -> BGR resident input is the declared upload-boundary
  // transport conversion. It is covered by the single upload accounting and
  // is not an effect-level CPU fallback or an undeclared CPU tail.

  for (std::size_t index = 0; index < program.stage_count; ++index) {
    const auto &stage = program.stages[index];
    if (stage.backend != BackendId::maxine ||
        stage.execution_domain != configuration_.execution_domain ||
        !ValidResidentIo(stage.input, configuration_.resident_key,
                         hal::Mutability::read_only) ||
        !ValidResidentIo(stage.output, configuration_.resident_key,
                         hal::Mutability::read_write) ||
        !SameImageStorage(stage.input.image, stage.output.image)) {
      SetReason(reason,
                stage.execution_domain != configuration_.execution_domain
                    ? ReasonId::wrong_execution_domain
                    : ReasonId::malformed_descriptor);
      return false;
    }
    ResidentStageKind kind{};
    bool requires_matte = false;
    if (!MapStage(stage, &kind, &requires_matte)) {
      const auto expected = ExpectedProvider(stage.semantic);
      SetReason(reason, expected != SemanticEffectProviderId::invalid &&
                                stage.provider != expected
                            ? ReasonId::semantic_provider_unavailable
                            : ReasonId::semantic_effect_unsupported);
      return false;
    }
    if (kind != ResidentStageKind::vignette &&
        !ModelEvidenceComplete(program, stage)) {
      SetReason(reason, ReasonId::capability_evidence_incomplete);
      return false;
    }
    if (!plan->AddCompatible(kind, !stage.required, requires_matte,
                             requires_matte ? MatteFingerprint(program, stage)
                                            : std::uint64_t{0})) {
      SetReason(reason, ReasonId::malformed_contract);
      return false;
    }
  }

  if (program.cpu_tail_count > 1) {
    SetReason(reason, ReasonId::cpu_tail_required);
    return false;
  }
  if (program.cpu_tail_count == 1) {
    const auto &tail = program.cpu_tails[0];
    if (tail.after_stage_id !=
            program.stages[program.stage_count - 1].stage_id ||
        !program.requires_final_readback) {
      SetReason(reason, ReasonId::malformed_contract);
      return false;
    }
    CpuTailKind tail_kind{};
    const ReasonId tail_reason = CpuTailToResident(tail.id, &tail_kind);
    if (tail_reason != ReasonId::none || !plan->SetCpuTail(tail_kind)) {
      SetReason(reason, tail_reason == ReasonId::none
                            ? ReasonId::malformed_contract
                            : tail_reason);
      return false;
    }
    if (tail.id != hal::CpuTailId::output_format_conversion) {
      SetReason(reason, ReasonId::malformed_contract);
      return false;
    }
  }

  plan->require_cpu_output =
      program.requires_final_readback && !plan->has_cpu_tail;
  return true;
}

StatusId MaxineHalResidentAdapter::Prepare(
    const hal::PreparedProgram &program, hal::PreparedProgramHandle *candidate,
    hal::ReadinessSnapshot *readiness, ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (candidate)
    *candidate = {};
  if (readiness)
    *readiness = {};
  if (stopped_) {
    SetReason(reason, ReasonId::shutdown_requested);
    return StatusId::stopped;
  }
  if (!candidate || !readiness || !ConfigurationValid()) {
    SetReason(reason, ReasonId::malformed_contract);
    return StatusId::blocked;
  }

  ReasonId validation_reason = ReasonId::none;
  if (!hal::ValidatePreparedProgram(program, &validation_reason)) {
    SetReason(reason, validation_reason);
    return validation_reason == ReasonId::resource_exhausted
               ? StatusId::resource_exhausted
               : StatusId::blocked;
  }
  if (program.owner_adapter_instance_id != adapter_instance_id() ||
      program.execution_domain.adapter_instance_id != adapter_instance_id()) {
    SetReason(reason, ReasonId::foreign_adapter);
    return StatusId::blocked;
  }
  if (program.execution_domain != configuration_.execution_domain) {
    SetReason(reason, ReasonId::wrong_execution_domain);
    return StatusId::blocked;
  }
  if (program.execution_domain.backend != BackendId::maxine ||
      (program.strict_backend_selection &&
       program.requested_backend != BackendId::maxine)) {
    SetReason(reason, ReasonId::strict_backend_unavailable);
    return StatusId::blocked;
  }
  const bool backend_fallback = program.requested_backend != BackendId::maxine;
  if (!BudgetContains(configuration_.capacity, program.budget)) {
    SetReason(reason, ReasonId::resource_exhausted);
    return StatusId::resource_exhausted;
  }
  if (program.temporal.temporal && !hooks_.reset_temporal &&
      !configuration_.temporal_reprepare_on_reset) {
    SetReason(reason, ReasonId::capability_evidence_incomplete);
    return StatusId::blocked;
  }
  if ((program.requires_final_readback && !hooks_.query_host_output) ||
      !hooks_.query_authoritative_telemetry) {
    SetReason(reason, ReasonId::capability_evidence_incomplete);
    return StatusId::blocked;
  }
  MaxineAuthoritativeTelemetrySnapshot authoritative{};
  if (!hooks_.query_authoritative_telemetry(hooks_.owner, &authoritative) ||
      !ValidAuthoritativeSnapshot(authoritative, configuration_)) {
    SetReason(reason, ReasonId::capability_evidence_incomplete);
    return StatusId::blocked;
  }
  if (authoritative.device_lost) {
    SetReason(reason, ReasonId::device_lost);
    return StatusId::quarantined;
  }

  ResidentFramePlan resident_plan{};
  ReasonId plan_reason = ReasonId::none;
  if (!ProgramToResidentPlan(program, &resident_plan, &plan_reason)) {
    SetReason(reason, plan_reason);
    return plan_reason == ReasonId::resource_exhausted
               ? StatusId::resource_exhausted
               : StatusId::blocked;
  }

  PreparedSlot *slot = nullptr;
  for (auto &candidate_slot : slots_) {
    if (!candidate_slot.occupied) {
      slot = &candidate_slot;
      break;
    }
  }
  if (!slot) {
    SetReason(reason, ReasonId::resource_exhausted);
    return StatusId::resource_exhausted;
  }

  const ResidentBoundaryResult prepared =
      executor_->Prepare(configuration_.resident_key);
  if (prepared != ResidentBoundaryResult::success) {
    const ReasonId failure =
        ClassifyFailure(prepared, ReasonId::capability_evidence_incomplete);
    SetReason(reason, failure);
    return failure == ReasonId::device_lost ? StatusId::quarantined
                                            : StatusId::blocked;
  }

  std::uint64_t nonce = next_nonce_++;
  if (nonce == 0)
    nonce = next_nonce_++;
  *slot = {};
  slot->occupied = true;
  slot->handle_id = nonce;
  slot->slot_nonce = nonce ^ adapter_instance_id();
  slot->program = program;
  slot->resident_plan = resident_plan;

  candidate->handle_id = slot->handle_id;
  candidate->program_id = program.program_id;
  candidate->adapter_instance_id = adapter_instance_id();
  candidate->invalidation_key = program.invalidation_key;
  candidate->execution_domain = configuration_.execution_domain;
  candidate->reserved_budget = program.budget;
  candidate->adapter_opaque[0] =
      static_cast<std::uint64_t>(slot - slots_.data()) + 1;
  candidate->adapter_opaque[1] = slot->slot_nonce;

  const StatusId preparation_status =
      backend_fallback ? StatusId::degraded : StatusId::ready;
  readiness->status = preparation_status;
  readiness->reason = backend_fallback ? ReasonId::semantic_provider_unavailable
                                       : ReasonId::none;
  readiness->requested_backend = program.requested_backend;
  readiness->selected_backend = BackendId::maxine;
  readiness->provider = program.stage_count == 1
                            ? program.stages[0].provider
                            : SemanticEffectProviderId::invalid;
  readiness->execution_domain = configuration_.execution_domain;
  readiness->ready_stage_mask =
      program.stage_count == 32
          ? std::numeric_limits<std::uint32_t>::max()
          : (std::uint32_t{1} << program.stage_count) - 1u;
  readiness->available = true;
  readiness->degraded = backend_fallback;
  SetReason(reason, readiness->reason);
  return preparation_status;
}

MaxineHalResidentAdapter::PreparedSlot *MaxineHalResidentAdapter::FindSlot(
    const hal::PreparedProgramHandle &program) noexcept {
  if (program.adapter_instance_id != adapter_instance_id() ||
      program.adapter_opaque[0] == 0 ||
      program.adapter_opaque[0] > slots_.size())
    return nullptr;
  auto &slot = slots_[program.adapter_opaque[0] - 1];
  return slot.occupied && slot.handle_id == program.handle_id &&
                 slot.slot_nonce == program.adapter_opaque[1] &&
                 slot.program.program_id == program.program_id &&
                 slot.program.invalidation_key == program.invalidation_key &&
                 program.execution_domain == configuration_.execution_domain
             ? &slot
             : nullptr;
}

const MaxineHalResidentAdapter::PreparedSlot *
MaxineHalResidentAdapter::FindSlot(
    const hal::PreparedProgramHandle &program) const noexcept {
  return const_cast<MaxineHalResidentAdapter *>(this)->FindSlot(program);
}

ReasonId
MaxineHalResidentAdapter::ClassifyFailure(ResidentBoundaryResult boundary,
                                          ReasonId fallback) const noexcept {
  if (hooks_.classify_failure) {
    const ReasonId classified = hooks_.classify_failure(hooks_.owner, boundary);
    if (classified != ReasonId::none)
      return classified;
  }
  if (boundary == ResidentBoundaryResult::incompatible_output)
    return ReasonId::invalid_resource;
  return fallback;
}

StatusId MaxineHalResidentAdapter::BindHostFrame(
    const hal::PreparedProgramHandle &program,
    const hal::HostFrameBinding &binding, hal::HostFrameTokens *tokens,
    ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (tokens)
    *tokens = {};
  if (stopped_) {
    SetReason(reason, ReasonId::shutdown_requested);
    return StatusId::stopped;
  }
  PreparedSlot *slot = FindSlot(program);
  if (!slot) {
    SetReason(reason, ReasonId::stale_program);
    return StatusId::stale;
  }
  ReasonId binding_reason = ReasonId::none;
  if (!tokens || !slot->program.requires_upload ||
      !slot->program.requires_final_readback ||
      slot->program.cpu_tail_count != 1 ||
      slot->program.cpu_tails[0].id !=
          hal::CpuTailId::output_format_conversion ||
      !hal::ValidateHostFrameBinding(binding, slot->program, &binding_reason) ||
      binding.input_plane_count != 1 || binding.output_plane_count != 1 ||
      !SameImageDescriptor(binding.input_descriptor,
                           slot->program.host_input) ||
      !SameImageDescriptor(binding.output_descriptor,
                           slot->program.host_output)) {
    SetReason(reason, binding_reason == ReasonId::none
                          ? ReasonId::host_frame_binding_invalid
                          : binding_reason);
    return StatusId::blocked;
  }

  const auto &input_plane = binding.input_planes[0];
  const auto &output_plane = binding.output_planes[0];
  HostRgbFrameView input_view{
      reinterpret_cast<const std::uint8_t *>(input_plane.read_data),
      binding.input_descriptor.width, binding.input_descriptor.height,
      static_cast<std::size_t>(input_plane.row_stride_bytes)};
  if (!input_view.ValidFor(configuration_.resident_key) ||
      input_plane.span_bytes < binding.input_descriptor.planes[0].size_bytes ||
      output_plane.span_bytes <
          binding.output_descriptor.planes[0].size_bytes ||
      binding.input_descriptor.planes[0].size_bytes >
          slot->program.budget.upload_staging_bytes ||
      binding.output_descriptor.planes[0].size_bytes >
          slot->program.budget.readback_staging_bytes) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return StatusId::blocked;
  }

  std::uint64_t input_id = next_host_resource_id_++;
  if (input_id == 0)
    input_id = next_host_resource_id_++;
  std::uint64_t output_id = next_host_resource_id_++;
  if (output_id == 0)
    output_id = next_host_resource_id_++;
  const std::uint64_t cookie = kHostBindingCookieSalt ^ slot->slot_nonce ^
                               binding.capture_sequence ^ input_id ^ output_id ^
                               adapter_instance_id();

  hal::HostFrameTokens candidate{};
  candidate.input.resource_id = input_id;
  candidate.input.allocator_owner_id = adapter_instance_id();
  candidate.input.lifetime_generation =
      configuration_.execution_domain.logical_context_generation;
  candidate.input.memory_domain = hal::MemoryDomain::upload_staging;
  candidate.input.lifetime = hal::ResourceLifetime::frame_borrowed;
  candidate.input.synchronization_state = hal::SynchronizationState::ready;
  candidate.input.execution_domain = configuration_.execution_domain;
  candidate.input.size_bytes = binding.input_descriptor.planes[0].size_bytes;
  candidate.input.alignment_bytes =
      binding.input_descriptor.planes[0].alignment_bytes;
  candidate.input.mutable_resource = false;
  candidate.input.aliasable = false;
  candidate.input.adapter_opaque[0] =
      static_cast<std::uint64_t>(slot - slots_.data()) + 1;
  candidate.input.adapter_opaque[1] = binding.capture_sequence;
  candidate.input.adapter_opaque[2] = cookie;
  candidate.input.adapter_opaque[3] = kHostInputTokenSalt;

  candidate.output.resource_id = output_id;
  candidate.output.allocator_owner_id = adapter_instance_id();
  candidate.output.lifetime_generation =
      configuration_.execution_domain.logical_context_generation;
  candidate.output.memory_domain = hal::MemoryDomain::readback_staging;
  candidate.output.lifetime = hal::ResourceLifetime::frame_borrowed;
  candidate.output.synchronization_state = hal::SynchronizationState::ready;
  candidate.output.execution_domain = configuration_.execution_domain;
  candidate.output.size_bytes = binding.output_descriptor.planes[0].size_bytes;
  candidate.output.alignment_bytes =
      binding.output_descriptor.planes[0].alignment_bytes;
  candidate.output.mutable_resource = true;
  candidate.output.aliasable = false;
  candidate.output.adapter_opaque[0] =
      static_cast<std::uint64_t>(slot - slots_.data()) + 1;
  candidate.output.adapter_opaque[1] = binding.capture_sequence;
  candidate.output.adapter_opaque[2] = cookie;
  candidate.output.adapter_opaque[3] = kHostOutputTokenSalt;

  slot->host_frame = HostFrameState{true, binding, candidate, cookie};
  slot->output = {};
  slot->output_resource = {};
  slot->output_resource_id = 0;
  slot->output_cookie = 0;
  slot->neutral_output_materialized = false;
  *tokens = candidate;
  return StatusId::ready;
}

StatusId MaxineHalResidentAdapter::BindHostFrame(const HostRgbFrameView &frame,
                                                 std::uint64_t capture_sequence,
                                                 std::uint64_t resource_id,
                                                 hal::ResourceHandle *resource,
                                                 ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (resource)
    *resource = {};
  if (stopped_) {
    SetReason(reason, ReasonId::shutdown_requested);
    return StatusId::stopped;
  }
  if (!resource || resource_id == 0 || capture_sequence == 0 ||
      !ConfigurationValid() || !frame.ValidFor(configuration_.resident_key)) {
    SetReason(reason, ReasonId::malformed_descriptor);
    return StatusId::blocked;
  }
  std::uint64_t size_bytes = 0;
  if (!Multiply(frame.stride_bytes, frame.height, &size_bytes) ||
      size_bytes == 0 ||
      size_bytes > configuration_.capacity.upload_staging_bytes) {
    SetReason(reason, ReasonId::resource_exhausted);
    return StatusId::resource_exhausted;
  }
  if (!ValidPointerSpan(frame.data, size_bytes)) {
    SetReason(reason, ReasonId::malformed_descriptor);
    return StatusId::blocked;
  }

  input_binding_ = MaxineHostFrameInputBinding{kMaxineHalBindingVersion,
                                               capture_sequence, frame};
  input_resource_id_ = resource_id;
  input_cookie_ = kInputCookieSalt ^ resource_id ^ capture_sequence ^
                  adapter_instance_id() ^ next_nonce_;

  resource->resource_id = resource_id;
  resource->allocator_owner_id = adapter_instance_id();
  resource->lifetime_generation =
      configuration_.execution_domain.logical_context_generation;
  resource->memory_domain = hal::MemoryDomain::upload_staging;
  resource->lifetime = hal::ResourceLifetime::frame_borrowed;
  resource->synchronization_state = hal::SynchronizationState::ready;
  resource->execution_domain = configuration_.execution_domain;
  resource->size_bytes = size_bytes;
  resource->alignment_bytes = 1;
  resource->mutable_resource = false;
  resource->aliasable = false;
  resource->adapter_opaque[0] = input_cookie_;
  resource->adapter_opaque[1] = capture_sequence;
  resource->adapter_opaque[2] = kInputCookieSalt;
  input_resource_ = *resource;
  return StatusId::ready;
}

bool MaxineHalResidentAdapter::PopulateOutput(
    PreparedSlot &slot, std::uint64_t capture_sequence,
    const ResidentFrameExecutionResult &execution, bool use_neutral_binding,
    hal::ResourceHandle *resource) noexcept {
  if (!resource)
    return false;
  MaxineFrameOutputBinding output{};
  output.version = kMaxineHalBindingVersion;
  output.capture_sequence = capture_sequence;

  hal::MemoryDomain memory_domain = hal::MemoryDomain::device_local;
  std::uint64_t size_bytes = 0;
  if (execution.status == ResidentExecutionStatus::resident_output) {
    output.boundary = MaxineOutputBoundary::resident;
    output.resident = execution.current;
    if (!output.ValidResident(configuration_.resident_key))
      return false;
    if (!Multiply(configuration_.resident_key.width,
                  configuration_.resident_key.height, &size_bytes) ||
        !Multiply(size_bytes, 3, &size_bytes))
      return false;
  } else {
    const ResidentReadbackBoundary boundary =
        execution.status == ResidentExecutionStatus::cpu_tail_required
            ? ResidentReadbackBoundary::cpu_continuation
            : ResidentReadbackBoundary::final_output;
    if (!hooks_.query_host_output ||
        !hooks_.query_host_output(hooks_.owner, boundary, capture_sequence,
                                  &output) ||
        !output.ValidHost() || output.capture_sequence != capture_sequence) {
      return false;
    }
    output.boundary = boundary == ResidentReadbackBoundary::cpu_continuation
                          ? MaxineOutputBoundary::cpu_tail_readback
                          : MaxineOutputBoundary::final_readback;
    memory_domain = hal::MemoryDomain::readback_staging;
    if (!Multiply(output.stride_bytes, output.height, &size_bytes))
      return false;
  }

  const std::uint64_t available =
      memory_domain == hal::MemoryDomain::readback_staging
          ? slot.program.budget.readback_staging_bytes
          : slot.program.budget.persistent_bytes;
  if (size_bytes == 0 || size_bytes > available)
    return false;

  slot.output = output;
  if (use_neutral_binding) {
    if (!slot.host_frame.bound ||
        slot.host_frame.binding.capture_sequence != capture_sequence ||
        slot.host_frame.tokens.output.memory_domain != memory_domain ||
        slot.host_frame.tokens.output.size_bytes != size_bytes) {
      return false;
    }
    const auto &descriptor = slot.host_frame.binding.output_descriptor;
    const auto &plane = slot.host_frame.binding.output_planes[0];
    if (output.boundary != MaxineOutputBoundary::cpu_tail_readback ||
        descriptor.format != hal::PixelFormat::rgb_u8 ||
        descriptor.channel_order != hal::ChannelOrder::rgb ||
        output.width != descriptor.width || output.height != descriptor.height ||
        plane.row_stride_bytes != descriptor.planes[0].row_stride_bytes ||
        plane.write_data == nullptr) {
      return false;
    }
    const auto conversion_started = std::chrono::steady_clock::now();
    ++output_format_conversion_telemetry_.attempts;
    const auto conversion = configuration_.output_format_converter(
        output.host_bgr,
        reinterpret_cast<std::uint8_t *>(plane.write_data),
        static_cast<int>(descriptor.width), static_cast<int>(descriptor.height),
        output.stride_bytes, static_cast<std::size_t>(plane.row_stride_bytes));
    const auto conversion_finished = std::chrono::steady_clock::now();
    output_format_conversion_telemetry_.elapsed_ns +=
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                conversion_finished - conversion_started)
                .count());
    if (conversion != video::CpuPrimitiveResult::success)
      return false;
    ++output_format_conversion_telemetry_.successes;
    slot.neutral_output_materialized = true;
    slot.output_resource = slot.host_frame.tokens.output;
    slot.output_resource_id = slot.output_resource.resource_id;
    slot.output_cookie = slot.host_frame.token_cookie;
    *resource = slot.output_resource;
    return true;
  }

  std::uint64_t output_id = next_output_resource_id_++;
  if (output_id == 0)
    output_id = next_output_resource_id_++;
  slot.output_resource_id = output_id;
  slot.output_cookie = kOutputCookieSalt ^ output_id ^ capture_sequence ^
                       slot.slot_nonce ^ adapter_instance_id();
  resource->resource_id = output_id;
  resource->allocator_owner_id = adapter_instance_id();
  resource->lifetime_generation =
      configuration_.execution_domain.logical_context_generation;
  resource->memory_domain = memory_domain;
  resource->lifetime = hal::ResourceLifetime::prepared_program;
  resource->synchronization_state = hal::SynchronizationState::ready;
  resource->execution_domain = configuration_.execution_domain;
  resource->size_bytes = size_bytes;
  resource->alignment_bytes = 1;
  resource->mutable_resource = false;
  resource->aliasable = false;
  resource->adapter_opaque[0] =
      static_cast<std::uint64_t>(&slot - slots_.data()) + 1;
  resource->adapter_opaque[1] = capture_sequence;
  resource->adapter_opaque[2] = slot.output_cookie;
  resource->adapter_opaque[3] =
      static_cast<std::uint64_t>(slot.output.boundary);
  slot.output_resource = *resource;
  return true;
}

hal::ExecuteResult
MaxineHalResidentAdapter::Execute(const hal::PreparedProgramHandle &program,
                                  const hal::ExecuteRequest &request) noexcept {
  if (stopped_)
    return Failed(StatusId::stopped, ReasonId::shutdown_requested);
  PreparedSlot *slot = FindSlot(program);
  if (!slot)
    return Failed(StatusId::stale, ReasonId::stale_program);
  if (request.invalidation_key != slot->program.invalidation_key)
    return Failed(StatusId::stale, ReasonId::reconfiguration_required);
  if (request.execution_domain != configuration_.execution_domain)
    return Failed(StatusId::blocked, ReasonId::wrong_execution_domain);
  if (!hal::ValidateResourceHandle(
          request.input, configuration_.execution_domain,
          configuration_.execution_domain.logical_context_generation,
          slot->program.budget)) {
    if (request.input.allocator_owner_id != adapter_instance_id())
      return Failed(StatusId::blocked, ReasonId::foreign_adapter);
    if (request.input.execution_domain != configuration_.execution_domain)
      return Failed(StatusId::blocked, ReasonId::wrong_execution_domain);
    if (request.input.lifetime_generation !=
        configuration_.execution_domain.logical_context_generation) {
      return Failed(StatusId::blocked, ReasonId::wrong_generation);
    }
    return Failed(StatusId::blocked, ReasonId::invalid_resource);
  }
  const bool use_neutral_binding =
      slot->host_frame.bound &&
      slot->host_frame.binding.capture_sequence == request.capture_sequence &&
      SameResource(request.input, slot->host_frame.tokens.input) &&
      request.input.adapter_opaque[0] ==
          static_cast<std::uint64_t>(slot - slots_.data()) + 1 &&
      request.input.adapter_opaque[1] == request.capture_sequence &&
      request.input.adapter_opaque[2] == slot->host_frame.token_cookie &&
      request.input.adapter_opaque[3] == kHostInputTokenSalt;
  const bool use_compatibility_binding =
      request.capture_sequence != 0 &&
      request.input.resource_id == input_resource_id_ &&
      request.input.adapter_opaque[0] == input_cookie_ &&
      request.input.adapter_opaque[1] == request.capture_sequence &&
      request.input.adapter_opaque[2] == kInputCookieSalt &&
      input_binding_.version == kMaxineHalBindingVersion &&
      input_binding_.capture_sequence == request.capture_sequence &&
      input_binding_.rgb.ValidFor(configuration_.resident_key) &&
      SameResource(request.input, input_resource_);
  if (request.capture_sequence == 0 ||
      (!use_neutral_binding && !use_compatibility_binding)) {
    return Failed(StatusId::blocked, ReasonId::invalid_resource);
  }
  const HostRgbFrameView input_view =
      use_neutral_binding
          ? HostRgbFrameView{reinterpret_cast<const std::uint8_t *>(
                                 slot->host_frame.binding.input_planes[0]
                                     .read_data),
                             slot->host_frame.binding.input_descriptor.width,
                             slot->host_frame.binding.input_descriptor.height,
                             static_cast<std::size_t>(
                                 slot->host_frame.binding.input_planes[0]
                                     .row_stride_bytes)}
          : input_binding_.rgb;
  if (!input_view.ValidFor(configuration_.resident_key))
    return Failed(StatusId::blocked, ReasonId::host_frame_binding_invalid);

  const std::uint32_t valid_stage_mask = StageMask(slot->program.stage_count);
  if ((request.disabled_stage_mask & ~valid_stage_mask) != 0)
    return Failed(StatusId::blocked, ReasonId::malformed_contract);
  for (std::size_t index = 0; index < slot->program.stage_count; ++index) {
    const std::uint32_t bit = std::uint32_t{1} << index;
    if ((request.disabled_stage_mask & bit) != 0 &&
        slot->program.stages[index].required) {
      hal::ExecuteResult result =
          Failed(StatusId::failed, ReasonId::circuit_breaker_open);
      result.stage_count = slot->program.stage_count;
      for (std::size_t stage_index = 0; stage_index < slot->program.stage_count;
           ++stage_index) {
        result.stages[stage_index].stage_id =
            slot->program.stages[stage_index].stage_id;
        result.stages[stage_index].semantic =
            slot->program.stages[stage_index].semantic;
        result.stages[stage_index].primitive =
            slot->program.stages[stage_index].primitive;
      }
      result.stages[index].status = StageExecutionStatus::required_failed;
      result.stages[index].reason = ReasonId::circuit_breaker_open;
      return result;
    }
  }

  ResidentFramePlan effective_plan = slot->resident_plan;
  effective_plan.stage_count = 0;
  std::array<std::size_t, kMaxResidentFrameStages> original_to_effective{};
  original_to_effective.fill(kMaxResidentFrameStages);
  for (std::size_t index = 0; index < slot->program.stage_count; ++index) {
    const std::uint32_t bit = std::uint32_t{1} << index;
    if ((request.disabled_stage_mask & bit) != 0)
      continue;
    const std::size_t effective_index = effective_plan.stage_count++;
    effective_plan.stages[effective_index] = slot->resident_plan.stages[index];
    original_to_effective[index] = effective_index;
  }

  MaxineAuthoritativeTelemetrySnapshot authoritative_before{};
  if (!hooks_.query_authoritative_telemetry ||
      !hooks_.query_authoritative_telemetry(hooks_.owner,
                                            &authoritative_before) ||
      !ValidAuthoritativeSnapshot(authoritative_before, configuration_)) {
    return Failed(StatusId::failed, ReasonId::capability_evidence_incomplete);
  }
  if (authoritative_before.device_lost)
    return Failed(StatusId::quarantined, ReasonId::device_lost);

  const auto started = std::chrono::steady_clock::now();
  const auto output_conversion_ns_before =
      output_format_conversion_telemetry_.elapsed_ns;
  const auto counters_before = slot->section.counters();
  PrevalidatedExecutor prevalidated(*executor_, configuration_.resident_key);
  ResidentFrameExecutionResult execution{};
  bool boundary_only_upload_succeeded = false;
  bool boundary_only_readback_succeeded = false;
  bool boundary_only_sync_succeeded = false;
  const bool boundary_only = effective_plan.stage_count == 0;
  if (!boundary_only) {
    execution = slot->section.Execute(
        effective_plan, configuration_.resident_key, request.capture_sequence,
        input_view, prevalidated);
  } else {
    execution.boundary_result =
        prevalidated.StageRgbToBgr(input_view, configuration_.resident_key);
    if (execution.boundary_result == ResidentBoundaryResult::success) {
      ResidentImage uploaded{};
      execution.boundary_result =
          prevalidated.UploadStagedBgr(configuration_.resident_key, uploaded);
      if (execution.boundary_result == ResidentBoundaryResult::success &&
          uploaded.ValidFor(configuration_.resident_key)) {
        boundary_only_upload_succeeded = true;
        execution.current = uploaded;
        if (effective_plan.has_cpu_tail || effective_plan.require_cpu_output) {
          const auto boundary = effective_plan.has_cpu_tail
                                    ? ResidentReadbackBoundary::cpu_continuation
                                    : ResidentReadbackBoundary::final_output;
          execution.boundary_result =
              prevalidated.DownloadToHost(boundary, execution.current);
          if (execution.boundary_result == ResidentBoundaryResult::success) {
            boundary_only_readback_succeeded = true;
            execution.boundary_result =
                prevalidated.Synchronize(boundary, configuration_.resident_key);
            if (execution.boundary_result == ResidentBoundaryResult::success) {
              boundary_only_sync_succeeded = true;
              execution.status =
                  effective_plan.has_cpu_tail
                      ? ResidentExecutionStatus::cpu_tail_required
                      : ResidentExecutionStatus::cpu_visible_output;
              execution.cpu_tail = effective_plan.cpu_tail;
            }
          }
        } else {
          execution.status = ResidentExecutionStatus::resident_output;
        }
      } else if (execution.boundary_result == ResidentBoundaryResult::success) {
        execution.boundary_result = ResidentBoundaryResult::incompatible_output;
      }
    }
  }
  const auto finished = std::chrono::steady_clock::now();
  const auto &counters_after = slot->section.counters();
  MaxineAuthoritativeTelemetrySnapshot authoritative_after{};
  if (!hooks_.query_authoritative_telemetry(hooks_.owner,
                                            &authoritative_after) ||
      !ValidAuthoritativeSnapshot(authoritative_after, configuration_)) {
    return Failed(StatusId::failed, ReasonId::capability_evidence_incomplete);
  }

  hal::ExecuteResult result{};
  result.frame_elapsed_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started)
          .count());
  result.stage_count = slot->program.stage_count;
  std::size_t executor_stage_index = 0;
  bool matte_time_attributed = false;
  for (std::size_t index = 0; index < result.stage_count; ++index) {
    auto &stage = result.stages[index];
    stage.stage_id = slot->program.stages[index].stage_id;
    stage.semantic = slot->program.stages[index].semantic;
    stage.primitive = slot->program.stages[index].primitive;
    const std::uint32_t original_bit = std::uint32_t{1} << index;
    if ((request.disabled_stage_mask & original_bit) != 0) {
      stage.status = StageExecutionStatus::disabled_preserved;
      stage.reason = ReasonId::circuit_breaker_open;
      stage.valid_output_preserved = true;
      continue;
    }
    const std::size_t effective_index = original_to_effective[index];
    if (effective_index >= effective_plan.stage_count) {
      result.status = StatusId::failed;
      result.reason = ReasonId::malformed_contract;
      return result;
    }
    const std::uint32_t effective_bit = std::uint32_t{1} << effective_index;
    bool failed_at_matte = false;
    for (std::size_t failure_index = 0;
         failure_index < execution.stage_failure_count; ++failure_index) {
      const auto &failure = execution.stage_failures[failure_index];
      if (failure.plan_stage_index == effective_index &&
          failure.point == ResidentStageFailurePoint::shared_matte) {
        failed_at_matte = true;
        break;
      }
    }
    if ((execution.optional_fail_open_plan_stage_mask & effective_bit) != 0) {
      stage.status = StageExecutionStatus::optional_failed_open;
      stage.reason = ReasonId::optional_stage_failed_open;
      stage.valid_output_preserved = true;
    } else if ((execution.failed_plan_stage_mask & effective_bit) != 0 ||
               effective_index >= execution.resident_stages_consumed) {
      stage.status = StageExecutionStatus::required_failed;
      stage.reason = ReasonId::required_stage_failed;
    } else {
      stage.status = StageExecutionStatus::succeeded;
      stage.reason = ReasonId::none;
      stage.valid_output_preserved = true;
    }

    if (failed_at_matte) {
      if (!matte_time_attributed) {
        stage.elapsed_ns = prevalidated.matte_elapsed_ns();
        matte_time_attributed = true;
      }
    } else if ((execution.failed_plan_stage_mask & effective_bit) != 0 ||
               effective_index < execution.resident_stages_consumed) {
      stage.elapsed_ns = prevalidated.stage_elapsed_ns(executor_stage_index++);
      if (effective_plan.stages[effective_index].requires_shared_matte &&
          !matte_time_attributed) {
        stage.elapsed_ns += prevalidated.matte_elapsed_ns();
        matte_time_attributed = true;
      }
    }
  }

  if (!TelemetryDelta(authoritative_before, authoritative_after,
                      &result.transfers)) {
    return Failed(StatusId::failed, ReasonId::synchronization_failed);
  }
  if (boundary_only) {
    result.transfers.cpu_tail_boundaries =
        effective_plan.has_cpu_tail && boundary_only_readback_succeeded &&
                boundary_only_sync_succeeded
            ? 1
            : 0;
  } else {
    for (std::size_t index = 0;
         index < static_cast<std::size_t>(CpuTailKind::count); ++index) {
      result.transfers.cpu_tail_boundaries +=
          counters_after.cpu_tail_boundaries[index] -
          counters_before.cpu_tail_boundaries[index];
    }
  }

  const std::uint64_t expected_uploads =
      boundary_only
          ? static_cast<std::uint64_t>(boundary_only_upload_succeeded)
          : counters_after.upload_successes - counters_before.upload_successes;
  const std::uint64_t expected_readbacks =
      boundary_only
          ? static_cast<std::uint64_t>(boundary_only_readback_succeeded)
          : counters_after.download_successes -
                counters_before.download_successes;
  const std::uint64_t expected_synchronizations =
      boundary_only
          ? static_cast<std::uint64_t>(boundary_only_sync_succeeded)
          : counters_after.sync_successes - counters_before.sync_successes;
  if (result.transfers.uploads != expected_uploads ||
      result.transfers.final_readbacks != expected_readbacks ||
      result.transfers.synchronizations != expected_synchronizations ||
      result.transfers.intermediate_uploads != 0 ||
      result.transfers.intermediate_readbacks != 0) {
    result.status = StatusId::failed;
    result.reason = ReasonId::synchronization_failed;
    return result;
  }
  if (authoritative_after.device_lost) {
    result.status = StatusId::quarantined;
    result.reason = ReasonId::device_lost;
    return result;
  }

  if (!IsSuccessful(execution.status)) {
    const ReasonId failure = ClassifyFailure(
        execution.boundary_result,
        execution.resident_stages_consumed < effective_plan.stage_count
            ? ReasonId::required_stage_failed
            : ReasonId::synchronization_failed);
    result.status = failure == ReasonId::device_lost ? StatusId::quarantined
                                                     : StatusId::failed;
    result.reason = failure;
    return result;
  }

  if (!PopulateOutput(*slot, request.capture_sequence, execution,
                      use_neutral_binding, &result.output)) {
    result.status = StatusId::failed;
    result.reason = ReasonId::invalid_resource;
    return result;
  }
  result.frame_elapsed_ns +=
      output_format_conversion_telemetry_.elapsed_ns -
      output_conversion_ns_before;
  result.output_valid = true;
  result.status = execution.optional_failure_observed ? StatusId::degraded
                                                      : StatusId::executed;
  result.reason =
      execution.optional_failure_observed
          ? ReasonId::optional_stage_failed_open
          : (execution.status == ResidentExecutionStatus::cpu_tail_required
                 ? ReasonId::cpu_tail_required
                 : ReasonId::none);
  result.completion.token_id = slot->output_resource_id;
  result.completion.program_id = slot->program.program_id;
  result.completion.adapter_instance_id = adapter_instance_id();
  result.completion.capture_sequence = request.capture_sequence;
  result.completion.complete = true;
  return result;
}

StatusId MaxineHalResidentAdapter::ResetTemporalState(
    const hal::PreparedProgramHandle &program, ReasonId reason) noexcept {
  if (stopped_)
    return StatusId::stopped;
  PreparedSlot *slot = FindSlot(program);
  if (!slot)
    return StatusId::stale;
  if (!slot->program.temporal.temporal)
    return StatusId::ready;
  if (!hooks_.reset_temporal || !hooks_.reset_temporal(hooks_.owner, reason)) {
    return StatusId::failed;
  }
  slot->section.Invalidate(ResidentInvalidationReason::provider_transition);
  return StatusId::ready;
}

StatusId MaxineHalResidentAdapter::ResolveHostFrameOutput(
    const hal::PreparedProgramHandle &program,
    const hal::CompletionToken &completion, const hal::ResourceHandle &resource,
    hal::HostFrameOutputResolution *resolution, ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (resolution)
    *resolution = {};
  if (stopped_) {
    SetReason(reason, ReasonId::shutdown_requested);
    return StatusId::stopped;
  }
  const PreparedSlot *slot = FindSlot(program);
  if (!slot) {
    SetReason(reason, ReasonId::stale_program);
    return StatusId::stale;
  }
  if (!resolution || !slot->host_frame.bound) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return StatusId::blocked;
  }
  if (resource.allocator_owner_id != adapter_instance_id()) {
    SetReason(reason, ReasonId::foreign_adapter);
    return StatusId::blocked;
  }
  if (resource.execution_domain != configuration_.execution_domain) {
    SetReason(reason, ReasonId::wrong_execution_domain);
    return StatusId::blocked;
  }
  if (resource.lifetime_generation !=
      configuration_.execution_domain.logical_context_generation) {
    SetReason(reason, ReasonId::wrong_generation);
    return StatusId::blocked;
  }
  if (!completion.complete || completion.token_id == 0 ||
      completion.token_id != slot->output_resource_id ||
      completion.program_id != slot->program.program_id ||
      completion.adapter_instance_id != adapter_instance_id() ||
      completion.capture_sequence !=
          slot->host_frame.binding.capture_sequence) {
    SetReason(reason, ReasonId::output_not_committed);
    return StatusId::blocked;
  }
  if (!SameResource(resource, slot->host_frame.tokens.output) ||
      !SameResource(resource, slot->output_resource) ||
      resource.lifetime != hal::ResourceLifetime::frame_borrowed ||
      resource.memory_domain != hal::MemoryDomain::readback_staging ||
      resource.adapter_opaque[0] !=
          static_cast<std::uint64_t>(slot - slots_.data()) + 1 ||
      resource.adapter_opaque[1] != completion.capture_sequence ||
      resource.adapter_opaque[2] != slot->host_frame.token_cookie ||
      resource.adapter_opaque[3] != kHostOutputTokenSalt ||
      slot->output.capture_sequence != completion.capture_sequence ||
      (slot->output.boundary != MaxineOutputBoundary::final_readback &&
       slot->output.boundary != MaxineOutputBoundary::cpu_tail_readback) ||
      !slot->output.ValidHost() ||
      !slot->neutral_output_materialized ||
      !SameImageDescriptor(slot->host_frame.binding.output_descriptor,
                           slot->program.host_output)) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return StatusId::blocked;
  }

  resolution->descriptor = slot->host_frame.binding.output_descriptor;
  resolution->planes = slot->host_frame.binding.output_planes;
  resolution->plane_count = slot->host_frame.binding.output_plane_count;
  resolution->capture_sequence = completion.capture_sequence;
  resolution->output_ready = true;
  return StatusId::ready;
}

void MaxineHalResidentAdapter::DestroyPrepared(
    hal::PreparedProgramHandle *program) noexcept {
  if (!program)
    return;
  if (PreparedSlot *slot = FindSlot(*program)) {
    slot->section.Invalidate(ResidentInvalidationReason::configuration_refresh);
    *slot = {};
  }
  *program = {};
}

void MaxineHalResidentAdapter::Shutdown() noexcept {
  if (stopped_)
    return;
  for (auto &slot : slots_) {
    if (slot.occupied)
      slot.section.Invalidate(ResidentInvalidationReason::runtime_failure);
    slot = {};
  }
  input_binding_ = {};
  input_resource_ = {};
  input_resource_id_ = 0;
  input_cookie_ = 0;
  if (hooks_.shutdown)
    hooks_.shutdown(hooks_.owner);
  stopped_ = true;
}

bool MaxineHalResidentAdapter::ResolveOutput(
    const hal::ResourceHandle &resource,
    MaxineFrameOutputBinding *output) const noexcept {
  if (!output || stopped_ ||
      resource.allocator_owner_id != adapter_instance_id() ||
      resource.execution_domain != configuration_.execution_domain ||
      resource.adapter_opaque[0] == 0 ||
      resource.adapter_opaque[0] > slots_.size())
    return false;
  const auto &slot = slots_[resource.adapter_opaque[0] - 1];
  const bool neutral_output =
      slot.host_frame.bound &&
      SameResource(resource, slot.host_frame.tokens.output) &&
      resource.adapter_opaque[1] == slot.output.capture_sequence &&
      resource.adapter_opaque[2] == slot.host_frame.token_cookie &&
      resource.adapter_opaque[3] == kHostOutputTokenSalt;
  const bool compatibility_output =
      resource.adapter_opaque[1] == slot.output.capture_sequence &&
      resource.adapter_opaque[2] == slot.output_cookie &&
      resource.adapter_opaque[3] ==
          static_cast<std::uint64_t>(slot.output.boundary);
  if (!slot.occupied || !SameResource(resource, slot.output_resource) ||
      resource.resource_id != slot.output_resource_id ||
      (!neutral_output && !compatibility_output)) {
    return false;
  }
  *output = slot.output;
  return true;
}

} // namespace studiocast::maxine
