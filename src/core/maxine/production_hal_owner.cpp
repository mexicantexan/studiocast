#include "core/maxine/production_hal_owner.h"

#include "core/cuda/cuda_driver_api.h"
#include "core/video/effects/broadcast_effect_rules.h"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

namespace studiocast::maxine {
namespace {

using hal::ReasonId;
using hal::StatusId;

void ResetResult(MaxineProductionPreparationResult *result) {
  if (result)
    *result = {};
}

void SetResult(MaxineProductionPreparationResult *result, StatusId status,
               ReasonId reason, const char *diagnostic) {
  if (!result)
    return;
  result->status = status;
  result->reason = reason;
  result->diagnostic = diagnostic ? diagnostic : "";
}

bool Add(std::uint64_t value, std::uint64_t *total) noexcept {
  if (!total || value > std::numeric_limits<std::uint64_t>::max() - *total)
    return false;
  *total += value;
  return true;
}

bool Multiply(std::uint64_t left, std::uint64_t right,
              std::uint64_t *value) noexcept {
  if (!value || (left != 0 &&
                 right > std::numeric_limits<std::uint64_t>::max() / left))
    return false;
  *value = left * right;
  return true;
}

bool Contains(const hal::ResourceBudget &capacity,
              const hal::ResourceBudget &required) noexcept {
  return capacity.persistent_bytes >= required.persistent_bytes &&
         capacity.scratch_bytes >= required.scratch_bytes &&
         capacity.upload_staging_bytes >= required.upload_staging_bytes &&
         capacity.readback_staging_bytes >= required.readback_staging_bytes &&
         capacity.resource_count >= required.resource_count &&
         capacity.in_flight_slots >= required.in_flight_slots;
}

bool ValidCapacity(const hal::ResourceBudget &capacity) noexcept {
  return capacity.persistent_bytes != 0 &&
         capacity.upload_staging_bytes != 0 &&
         capacity.readback_staging_bytes != 0 &&
         capacity.resource_count != 0 && capacity.in_flight_slots == 1;
}

bool StageForSemantic(
    hal::SemanticEffectId semantic, ResidentStageKind *kind,
    hal::SemanticEffectProviderId *expected_provider) noexcept {
  if (!kind || !expected_provider)
    return false;
  switch (semantic) {
  case hal::SemanticEffectId::video_noise_removal:
    *kind = ResidentStageKind::denoise;
    *expected_provider = hal::SemanticEffectProviderId::maxine_vfx;
    return true;
  case hal::SemanticEffectId::eye_contact:
    *kind = ResidentStageKind::eye_contact;
    *expected_provider = hal::SemanticEffectProviderId::maxine_ar;
    return true;
  case hal::SemanticEffectId::virtual_background_blur:
    *kind = ResidentStageKind::background_blur;
    *expected_provider = hal::SemanticEffectProviderId::maxine_vfx;
    return true;
  case hal::SemanticEffectId::virtual_background_remove:
    *kind = ResidentStageKind::background_remove;
    *expected_provider = hal::SemanticEffectProviderId::maxine_vfx;
    return true;
  case hal::SemanticEffectId::virtual_background_replace:
    *kind = ResidentStageKind::background_replace;
    *expected_provider = hal::SemanticEffectProviderId::maxine_vfx;
    return true;
  case hal::SemanticEffectId::virtual_key_light:
    *kind = ResidentStageKind::relighting;
    *expected_provider = hal::SemanticEffectProviderId::maxine_vfx;
    return true;
  case hal::SemanticEffectId::auto_frame:
    *kind = ResidentStageKind::auto_frame;
    *expected_provider = hal::SemanticEffectProviderId::maxine_ar;
    return true;
  case hal::SemanticEffectId::vignette:
    *kind = ResidentStageKind::vignette;
    *expected_provider =
        hal::SemanticEffectProviderId::maxine_cuda_utility;
    return true;
  case hal::SemanticEffectId::mirror:
  case hal::SemanticEffectId::invalid:
    return false;
  }
  return false;
}

video::effects::BroadcastEffectStage
CanonicalStage(hal::SemanticEffectId semantic) noexcept {
  using video::effects::BroadcastEffectStage;
  switch (semantic) {
  case hal::SemanticEffectId::video_noise_removal:
    return BroadcastEffectStage::video_noise_removal;
  case hal::SemanticEffectId::eye_contact:
    return BroadcastEffectStage::eye_contact;
  case hal::SemanticEffectId::virtual_background_blur:
    return BroadcastEffectStage::virtual_background_blur;
  case hal::SemanticEffectId::virtual_background_remove:
    return BroadcastEffectStage::virtual_background_remove;
  case hal::SemanticEffectId::virtual_background_replace:
    return BroadcastEffectStage::virtual_background_replace;
  case hal::SemanticEffectId::virtual_key_light:
    return BroadcastEffectStage::virtual_key_light;
  case hal::SemanticEffectId::auto_frame:
    return BroadcastEffectStage::auto_frame;
  case hal::SemanticEffectId::vignette:
    return BroadcastEffectStage::vignette;
  case hal::SemanticEffectId::mirror:
    return BroadcastEffectStage::mirror;
  case hal::SemanticEffectId::invalid:
    return BroadcastEffectStage::none;
  }
  return BroadcastEffectStage::none;
}

bool IsVirtualBackground(hal::SemanticEffectId semantic) noexcept {
  return semantic == hal::SemanticEffectId::virtual_background_blur ||
         semantic == hal::SemanticEffectId::virtual_background_remove ||
         semantic == hal::SemanticEffectId::virtual_background_replace;
}

} // namespace

bool KnownMinimumMaxineBudget(const ProductionResidentSetup &setup,
                              hal::ResourceBudget *budget) noexcept {
  if (!budget)
    return false;
  *budget = {};
  if (setup.width == 0 || setup.height == 0 || setup.output_width == 0 ||
      setup.output_height == 0)
    return false;

  std::uint64_t input_pixels = 0;
  std::uint64_t output_pixels = 0;
  if (!Multiply(setup.width, setup.height, &input_pixels) ||
      !Multiply(setup.output_width, setup.output_height, &output_pixels))
    return false;

  std::uint64_t input_bgr_bytes = 0;
  std::uint64_t output_bgr_bytes = 0;
  if (!Multiply(input_pixels, 3, &input_bgr_bytes) ||
      !Multiply(output_pixels, 3, &output_bgr_bytes))
    return false;

  budget->upload_staging_bytes = input_bgr_bytes;
  budget->readback_staging_bytes = output_bgr_bytes;
  budget->in_flight_slots = 1;

  std::uint64_t persistent = 0;
  std::uint64_t resources = 0;
  // Host input/output plus two canonical resident ping-pong images.
  if (!Add(input_bgr_bytes, &persistent) ||
      !Add(output_bgr_bytes, &persistent) ||
      !Add(input_bgr_bytes, &persistent) ||
      !Add(input_bgr_bytes, &persistent))
    return false;
  resources += 4;

  if (setup.output_width != setup.width ||
      setup.output_height != setup.height) {
    if (!Add(output_bgr_bytes, &persistent))
      return false;
    ++resources;
  }
  if ((setup.enabled_maxine_stage_mask &
       ProductionResidentStageBit(ResidentStageKind::denoise)) != 0) {
    std::uint64_t bridge = 0;
    if (!Multiply(input_pixels, 12, &bridge) || !Add(bridge, &persistent))
      return false;
    ++resources;
  }
  if ((setup.enabled_maxine_stage_mask &
       ProductionResidentStageBit(ResidentStageKind::relighting)) != 0) {
    if (!Add(input_pixels, &persistent))
      return false;
    ++resources;
  }
  if ((setup.enabled_maxine_stage_mask &
       ProductionResidentStageBit(ResidentStageKind::background_replace)) !=
      0) {
    if (!setup.replacement_background.Valid())
      return false;
    std::uint64_t background_pixels = 0;
    std::uint64_t background_bytes = 0;
    if (!Multiply(setup.replacement_background.width,
                  setup.replacement_background.height, &background_pixels) ||
        !Multiply(background_pixels, 3, &background_bytes) ||
        !Add(background_bytes, &persistent) ||
        !Add(background_bytes, &persistent) ||
        !Add(input_bgr_bytes, &persistent))
      return false;
    resources += 3;
  }

  if (resources > std::numeric_limits<std::uint32_t>::max())
    return false;
  budget->persistent_bytes = persistent;
  budget->resource_count = static_cast<std::uint32_t>(resources);
  return persistent != 0 && resources != 0;
}

std::unique_ptr<MaxineProductionHalOwner>
MaxineProductionHalOwner::CreateBackend(
    const MaxineProductionBackendRequest &request,
    MaxineProductionPreparationResult *result) noexcept {
  ResetResult(result);
  try {
    if (request.adapter_instance_id == 0 || !request.runtime.cuda ||
        !request.runtime.cuda->IsInitialized() || !request.runtime.nvcv ||
        !request.runtime.nvcv->IsInitialized() ||
        !ValidCapacity(request.capacity)) {
      SetResult(result, StatusId::blocked,
                ReasonId::capability_evidence_incomplete,
                "Maxine runtime, adapter identity, or resource capacity is "
                "incomplete.");
      return nullptr;
    }

    std::string identity_error;
    if (!request.runtime.cuda->EnsureContext(&identity_error)) {
      SetResult(result, StatusId::blocked,
                ReasonId::capability_evidence_incomplete,
                "CUDA context identity could not be established.");
      if (result && !identity_error.empty())
        result->diagnostic = std::move(identity_error);
      return nullptr;
    }
    const auto *identity = request.runtime.cuda->context_identity();
    if (!identity || !identity->Valid() ||
        identity->retained_context_id >
            std::numeric_limits<std::uintptr_t>::max()) {
      SetResult(result, StatusId::blocked,
                ReasonId::capability_evidence_incomplete,
                "CUDA physical UUID or logical context identity is missing.");
      return nullptr;
    }
    const cuda::CudaContextIdentity identity_before = *identity;

    hal::ResourceBudget minimum{};
    if (!KnownMinimumMaxineBudget(request.setup, &minimum)) {
      SetResult(result, StatusId::blocked, ReasonId::malformed_descriptor,
                "Maxine setup geometry or prepared asset budget is invalid.");
      return nullptr;
    }
    if (!Contains(request.capacity, minimum)) {
      SetResult(result, StatusId::resource_exhausted,
                ReasonId::resource_exhausted,
                "Maxine capacity is smaller than known prepared storage.");
      if (result)
        result->known_minimum_budget = minimum;
      return nullptr;
    }

    auto owner = std::unique_ptr<MaxineProductionHalOwner>(
        new MaxineProductionHalOwner());
    owner->capacity_ = request.capacity;
    owner->known_minimum_budget_ = minimum;
    owner->enabled_stage_mask_ = request.setup.enabled_maxine_stage_mask;
    owner->logical_context_generation_ = identity_before.logical_generation;
    constexpr std::uint32_t matte_stage_mask =
        ProductionResidentStageBit(ResidentStageKind::background_blur) |
        ProductionResidentStageBit(ResidentStageKind::background_remove) |
        ProductionResidentStageBit(ResidentStageKind::background_replace) |
        ProductionResidentStageBit(ResidentStageKind::relighting);
    owner->temporal_reset_required_ =
        (request.setup.enabled_maxine_stage_mask &
         ProductionResidentStageBit(ResidentStageKind::denoise)) != 0 ||
        (request.setup.enabled_maxine_stage_mask &
         ProductionResidentStageBit(ResidentStageKind::eye_contact)) != 0 ||
        (request.setup.enabled_maxine_stage_mask &
         ProductionResidentStageBit(ResidentStageKind::auto_frame)) != 0 ||
        (request.setup.effects.virtual_background.greenscreen_temporal &&
         (request.setup.enabled_maxine_stage_mask & matte_stage_mask) != 0);

    ProductionResidentSetup exact_setup = request.setup;
    const std::uintptr_t exact_runtime_identity =
        static_cast<std::uintptr_t>(identity_before.retained_context_id);
    if (exact_setup.runtime_identity != 0 &&
        exact_setup.runtime_identity != exact_runtime_identity) {
      SetResult(result, StatusId::blocked, ReasonId::wrong_generation,
                "Caller runtime identity does not match the retained CUDA "
                "context identity.");
      return nullptr;
    }
    exact_setup.runtime_identity = exact_runtime_identity;

    owner->executor_ = std::make_unique<ProductionResidentFrameExecutor>(
        request.runtime, request.device_ops,
        ResidentStreamCreationPolicy::cuda_driver_required);
    std::string setup_error;
    if (!owner->executor_->Configure(exact_setup, &setup_error)) {
      SetResult(result, StatusId::blocked,
                ReasonId::capability_evidence_incomplete,
                "Maxine resident executor preparation failed.");
      if (result && !setup_error.empty())
        result->diagnostic = std::move(setup_error);
      return nullptr;
    }

    // Re-establish and compare the authoritative context after Maxine created
    // and bound its stream. This catches context/device transitions during SDK
    // setup without probing on the frame path.
    if (!request.runtime.cuda->EnsureContext(&identity_error)) {
      SetResult(result, StatusId::blocked,
                ReasonId::capability_evidence_incomplete,
                "CUDA context could not be revalidated after Maxine setup.");
      return nullptr;
    }
    const auto *after = request.runtime.cuda->context_identity();
    if (!after || *after != identity_before) {
      SetResult(result, StatusId::blocked, ReasonId::wrong_generation,
                "CUDA context identity changed during Maxine setup.");
      return nullptr;
    }

    owner->resident_key_ = owner->executor_->key();
    if (!owner->resident_key_.Valid() ||
        owner->resident_key_.configuration_generation !=
            exact_setup.configuration_generation ||
        owner->resident_key_.runtime_identity != exact_runtime_identity ||
        owner->resident_key_.stream_identity !=
            owner->executor_->stream_identity()) {
      SetResult(result, StatusId::blocked,
                ReasonId::capability_evidence_incomplete,
                "Prepared Maxine stream identity is incomplete.");
      return nullptr;
    }

    owner->execution_domain_.backend = hal::BackendId::maxine;
    owner->execution_domain_.api_family = hal::ApiFamily::maxine_nvcv;
    owner->execution_domain_.physical_device.vendor_id = 0x10de;
    owner->execution_domain_.physical_device.stable_id =
        after->physical_device.uuid;
    owner->execution_domain_.logical_context_generation =
        after->logical_generation;
    owner->execution_domain_.allocator_family =
        hal::AllocatorFamily::maxine_nvcv;
    owner->execution_domain_.memory_handle_family =
        hal::MemoryHandleFamily::nvcv_image;
    owner->execution_domain_.queue_or_stream_identity =
        owner->resident_key_.stream_identity;
    owner->execution_domain_.ownership_model =
        hal::OwnershipModel::adapter_exclusive;
    owner->execution_domain_.synchronization_protocol =
        hal::SynchronizationProtocol::maxine_bound_cuda_stream;
    owner->execution_domain_.adapter_instance_id =
        request.adapter_instance_id;
    if (!hal::ValidateExecutionDomain(owner->execution_domain_)) {
      SetResult(result, StatusId::blocked,
                ReasonId::malformed_execution_domain,
                "Exact Maxine execution domain validation failed.");
      return nullptr;
    }

    MaxineHalAdapterConfiguration configuration{};
    configuration.adapter_instance_id = request.adapter_instance_id;
    configuration.execution_domain = owner->execution_domain_;
    configuration.resident_key = owner->resident_key_;
    configuration.capacity = request.capacity;
    configuration.temporal_reprepare_on_reset = true;
    MaxineHalAdapterHooks hooks{};
    hooks.owner = owner.get();
    hooks.classify_failure = ClassifyFailure;
    // No reset hook is intentionally supplied. The current production
    // executor cannot reset SDK temporal history without reconstructing
    // resources, so temporal programs fail closed during Prepare.
    hooks.query_host_output = QueryHostOutput;
    hooks.query_authoritative_telemetry = QueryAuthoritativeTelemetry;
    hooks.shutdown = AdapterShutdown;
    owner->adapter_ = std::make_unique<MaxineHalResidentAdapter>(
        configuration, *owner->executor_, hooks);
    owner->session_ = std::make_unique<hal::PreparedProgramSession>();

    if (result) {
      result->status = StatusId::ready;
      result->reason = ReasonId::none;
      result->execution_domain = owner->execution_domain_;
      result->known_minimum_budget = minimum;
      result->diagnostic.clear();
    }
    return owner;
  } catch (const std::bad_alloc &) {
    SetResult(result, StatusId::resource_exhausted,
              ReasonId::resource_exhausted,
              "Maxine production owner allocation failed.");
  } catch (...) {
    SetResult(result, StatusId::blocked,
              ReasonId::capability_evidence_incomplete,
              "Maxine production owner setup failed.");
  }
  return nullptr;
}

StatusId MaxineProductionHalOwner::ConfigureProgram(
    const hal::PreparedProgram &program,
    MaxineProductionPreparationResult *result) noexcept {
  ResetResult(result);
  if (stopped_ || !session_ || !adapter_) {
    SetResult(result, StatusId::stopped, ReasonId::shutdown_requested,
              "Maxine production owner is stopped.");
    return StatusId::stopped;
  }
  ReasonId stage_reason = ReasonId::none;
  if (!ValidateProgramStageSet(program, &stage_reason)) {
    SetResult(result,
              stage_reason == ReasonId::resource_exhausted
                  ? StatusId::resource_exhausted
                  : StatusId::blocked,
              stage_reason,
              "Prepared program does not exactly match the configured Maxine "
              "resident stage set.");
    return result ? result->status : StatusId::blocked;
  }
  hal::ReadinessSnapshot readiness{};
  ReasonId reason = ReasonId::none;
  const StatusId status =
      session_->Configure(program, *adapter_, &readiness, &reason);
  if (result) {
    result->status = status;
    result->reason = reason;
    result->execution_domain = execution_domain_;
    result->known_minimum_budget = known_minimum_budget_;
    result->diagnostic =
        status == StatusId::ready || status == StatusId::degraded
            ? ""
            : "Maxine prepared-program validation failed.";
  }
  return status;
}

bool MaxineProductionHalOwner::ValidateProgramStageSet(
    const hal::PreparedProgram &program, ReasonId *reason) const noexcept {
  if (reason)
    *reason = ReasonId::none;
  if (program.execution_domain != execution_domain_ ||
      program.owner_adapter_instance_id !=
          execution_domain_.adapter_instance_id ||
      program.invalidation_key.generations.configuration !=
          resident_key_.configuration_generation) {
    if (reason)
      *reason = program.execution_domain != execution_domain_
                    ? ReasonId::wrong_execution_domain
                    : ReasonId::wrong_generation;
    return false;
  }
  if (program.stage_count == 0 ||
      program.stage_count > hal::kMaxPreparedStages) {
    if (reason)
      *reason = ReasonId::malformed_contract;
    return false;
  }
  std::uint32_t program_mask = 0;
  auto previous_stage = video::effects::BroadcastEffectStage::none;
  bool has_virtual_background = false;
  for (std::size_t index = 0; index < program.stage_count; ++index) {
    const auto &stage = program.stages[index];
    if (stage.backend != hal::BackendId::maxine) {
      if (reason)
        *reason = ReasonId::strict_backend_unavailable;
      return false;
    }
    ResidentStageKind kind{};
    hal::SemanticEffectProviderId expected_provider =
        hal::SemanticEffectProviderId::invalid;
    if (!StageForSemantic(stage.semantic, &kind, &expected_provider)) {
      if (reason)
        *reason = ReasonId::semantic_effect_unsupported;
      return false;
    }
    if (stage.provider != expected_provider) {
      if (reason)
        *reason = ReasonId::semantic_provider_unavailable;
      return false;
    }
    const auto canonical_stage = CanonicalStage(stage.semantic);
    if (canonical_stage == video::effects::BroadcastEffectStage::none ||
        (index != 0 &&
         static_cast<std::uint8_t>(canonical_stage) <=
             static_cast<std::uint8_t>(previous_stage))) {
      if (reason)
        *reason = ReasonId::frame_order_violated;
      return false;
    }
    if (IsVirtualBackground(stage.semantic)) {
      if (has_virtual_background) {
        if (reason)
          *reason = ReasonId::malformed_contract;
        return false;
      }
      has_virtual_background = true;
    }
    previous_stage = canonical_stage;
    const std::uint32_t bit = ProductionResidentStageBit(kind);
    if ((program_mask & bit) != 0) {
      if (reason)
        *reason = ReasonId::malformed_contract;
      return false;
    }
    program_mask |= bit;
  }
  if (program_mask != enabled_stage_mask_) {
    if (reason)
      *reason = ReasonId::semantic_effect_unsupported;
    return false;
  }
  return true;
}

ReasonId MaxineProductionHalOwner::ClassifyFailure(
    void *owner, ResidentBoundaryResult boundary) noexcept {
  const auto *self = static_cast<const MaxineProductionHalOwner *>(owner);
  if (self && self->device_lost_)
    return ReasonId::device_lost;
  if (boundary == ResidentBoundaryResult::incompatible_output)
    return ReasonId::invalid_resource;
  return ReasonId::none;
}

bool MaxineProductionHalOwner::QueryHostOutput(
    void *owner, ResidentReadbackBoundary boundary,
    std::uint64_t capture_sequence,
    MaxineFrameOutputBinding *output) noexcept {
  auto *self = static_cast<MaxineProductionHalOwner *>(owner);
  if (!self || !output || !self->executor_ || self->stopped_ ||
      self->device_lost_)
    return false;
  const HostBgrOutputView view = self->executor_->host_output();
  if (!view.Valid())
    return false;
  *output = {};
  output->version = kMaxineHalBindingVersion;
  output->capture_sequence = capture_sequence;
  output->boundary =
      boundary == ResidentReadbackBoundary::cpu_continuation
          ? MaxineOutputBoundary::cpu_tail_readback
          : MaxineOutputBoundary::final_readback;
  output->host_bgr = view.data;
  output->width = view.width;
  output->height = view.height;
  output->stride_bytes = view.stride_bytes;
  return true;
}

bool MaxineProductionHalOwner::QueryAuthoritativeTelemetry(
    void *owner, MaxineAuthoritativeTelemetrySnapshot *snapshot) noexcept {
  auto *self = static_cast<MaxineProductionHalOwner *>(owner);
  if (!self || !snapshot || !self->executor_)
    return false;
  *snapshot = AuthoritativeSnapshotFromProduction(
      self->executor_->telemetry(), self->logical_context_generation_,
      self->resident_key_.stream_identity);
  snapshot->device_lost = self->device_lost_;
  return true;
}

void MaxineProductionHalOwner::AdapterShutdown(void *owner) noexcept {
  auto *self = static_cast<MaxineProductionHalOwner *>(owner);
  if (!self)
    return;
  self->stopped_ = true;
  if (self->executor_)
    self->executor_->InvalidateBindings();
}

void MaxineProductionHalOwner::NotifyDeviceLost() noexcept {
  if (stopped_)
    return;
  device_lost_ = true;
  if (session_)
    session_->Invalidate(ReasonId::device_lost);
}

void MaxineProductionHalOwner::Shutdown() noexcept {
  if (session_ && !session_->stopped())
    session_->Shutdown();
  else if (!session_ && adapter_)
    adapter_->Shutdown();
  adapter_.reset();
  executor_.reset();
  stopped_ = true;
}

MaxineProductionHalOwner::~MaxineProductionHalOwner() { Shutdown(); }

} // namespace studiocast::maxine
