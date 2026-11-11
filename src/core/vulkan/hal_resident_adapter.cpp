#include "core/vulkan/hal_resident_adapter.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace studiocast::vulkan {
namespace {

using hal::BackendId;
using hal::FailureAction;
using hal::MemoryDomain;
using hal::ReasonId;
using hal::ResourceBudget;
using hal::ResourceHandle;
using hal::ResourceLifetime;
using hal::SemanticEffectId;
using hal::SemanticEffectProviderId;
using hal::StageExecutionStatus;
using hal::StatusId;
using hal::SynchronizationState;

void SetReason(ReasonId *reason, ReasonId value) noexcept {
  if (reason)
    *reason = value;
}

bool BudgetContains(const ResourceBudget &available,
                    const ResourceBudget &required) noexcept {
  return available.persistent_bytes >= required.persistent_bytes &&
         available.scratch_bytes >= required.scratch_bytes &&
         available.upload_staging_bytes >= required.upload_staging_bytes &&
         available.readback_staging_bytes >= required.readback_staging_bytes &&
         available.resource_count >= required.resource_count &&
         available.in_flight_slots >= required.in_flight_slots;
}

bool CommonEvidenceReady(
    const VulkanResidentCapabilityEvidence &evidence) noexcept {
  return evidence.build_enabled && evidence.runtime_loaded &&
         evidence.non_cpu_physical_device && evidence.compute_queue_available &&
         evidence.owned_context_healthy && evidence.utility_kernels_ready &&
         evidence.exact_device_and_queue_shared &&
         evidence.persistent_device_local_resources &&
         evidence.bounded_reusable_allocations &&
         evidence.no_host_intermediate_storage &&
         evidence.one_upload_transport && evidence.one_final_readback_transport;
}

bool IsZeroModelIdentity(const hal::ModelArtifactIdentity &identity) noexcept {
  return identity.artifact_generation == 0 &&
         std::all_of(identity.model_id_hash.begin(),
                     identity.model_id_hash.end(),
                     [](std::uint8_t value) { return value == 0; }) &&
         std::all_of(identity.artifact_set_hash.begin(),
                     identity.artifact_set_hash.end(),
                     [](std::uint8_t value) { return value == 0; });
}

bool IsExactProductionRgbImage(const hal::ImageDescriptor &descriptor,
                               hal::Mutability mutability) noexcept {
  if (descriptor.width == 0 || descriptor.height == 0 ||
      descriptor.plane_count != 1 ||
      descriptor.format != hal::PixelFormat::rgb_u8 ||
      descriptor.channel_order != hal::ChannelOrder::rgb ||
      descriptor.color_space != hal::ColorSpace::srgb ||
      descriptor.color_range != hal::ColorRange::full ||
      descriptor.alpha_meaning != hal::AlphaMeaning::none ||
      descriptor.orientation != hal::Orientation::top_left ||
      descriptor.host_coherence != hal::HostCoherence::not_host_visible ||
      descriptor.mutability != mutability ||
      descriptor.aliasing != hal::AliasingPermission::forbidden ||
      descriptor.in_place_permitted) {
    return false;
  }
  const auto &plane = descriptor.planes[0];
  if (plane.offset_bytes != 0 || plane.width != descriptor.width ||
      plane.height != descriptor.height ||
      descriptor.width > std::numeric_limits<std::uint64_t>::max() / 3u) {
    return false;
  }
  const std::uint64_t minimum_stride =
      static_cast<std::uint64_t>(descriptor.width) * 3u;
  if (plane.row_stride_bytes < minimum_stride ||
      plane.row_stride_bytes >
          std::numeric_limits<std::uint64_t>::max() / descriptor.height) {
    return false;
  }
  return plane.size_bytes == plane.row_stride_bytes * descriptor.height &&
         plane.alignment_bytes <= descriptor.base_alignment_bytes;
}

bool IsExactHostRgbImage(const hal::ImageDescriptor &descriptor,
                         hal::Mutability mutability) noexcept {
  if (!hal::ValidateImageDescriptor(descriptor) ||
      descriptor.plane_count != 1 ||
      descriptor.format != hal::PixelFormat::rgb_u8 ||
      descriptor.channel_order != hal::ChannelOrder::rgb ||
      descriptor.color_space != hal::ColorSpace::srgb ||
      descriptor.color_range != hal::ColorRange::full ||
      descriptor.alpha_meaning != hal::AlphaMeaning::none ||
      descriptor.orientation != hal::Orientation::top_left ||
      descriptor.host_coherence != hal::HostCoherence::coherent ||
      descriptor.mutability != mutability ||
      descriptor.aliasing != hal::AliasingPermission::forbidden ||
      descriptor.in_place_permitted) {
    return false;
  }
  const auto &plane = descriptor.planes[0];
  return plane.offset_bytes == 0 && plane.width == descriptor.width &&
         plane.height == descriptor.height &&
         plane.row_stride_bytes >=
             static_cast<std::uint64_t>(descriptor.width) * 3u &&
         plane.row_stride_bytes <=
             std::numeric_limits<std::uint64_t>::max() / plane.height &&
         plane.size_bytes ==
             plane.row_stride_bytes * static_cast<std::uint64_t>(plane.height);
}

bool SameImageStorage(const hal::ImageDescriptor &lhs,
                      const hal::ImageDescriptor &rhs) noexcept {
  if (lhs.width != rhs.width || lhs.height != rhs.height ||
      lhs.plane_count != rhs.plane_count || lhs.format != rhs.format ||
      lhs.channel_order != rhs.channel_order ||
      lhs.color_space != rhs.color_space ||
      lhs.color_range != rhs.color_range ||
      lhs.alpha_meaning != rhs.alpha_meaning ||
      lhs.orientation != rhs.orientation ||
      lhs.host_coherence != rhs.host_coherence ||
      lhs.aliasing != rhs.aliasing ||
      lhs.in_place_permitted != rhs.in_place_permitted ||
      lhs.base_alignment_bytes != rhs.base_alignment_bytes) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.plane_count; ++index) {
    const auto &left = lhs.planes[index];
    const auto &right = rhs.planes[index];
    if (left.offset_bytes != right.offset_bytes ||
        left.row_stride_bytes != right.row_stride_bytes ||
        left.size_bytes != right.size_bytes || left.width != right.width ||
        left.height != right.height ||
        left.alignment_bytes != right.alignment_bytes) {
      return false;
    }
  }
  return true;
}

std::uint64_t
RequiredImageSpan(const hal::ImageDescriptor &descriptor) noexcept {
  std::uint64_t required = 0;
  for (std::size_t index = 0; index < descriptor.plane_count; ++index) {
    const auto &plane = descriptor.planes[index];
    if (plane.offset_bytes >
        std::numeric_limits<std::uint64_t>::max() - plane.size_bytes) {
      return 0;
    }
    required = std::max(required, plane.offset_bytes + plane.size_bytes);
  }
  return required;
}

bool PointerSpanValid(const void *pointer, std::uint64_t size) noexcept {
  if (!pointer || size == 0 ||
      size > std::numeric_limits<std::uintptr_t>::max()) {
    return false;
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  return begin <= std::numeric_limits<std::uintptr_t>::max() -
                      static_cast<std::uintptr_t>(size);
}

bool SpansOverlap(const void *lhs, std::uint64_t lhs_size, const void *rhs,
                  std::uint64_t rhs_size) noexcept {
  const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs);
  const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs);
  const auto lhs_end = lhs_begin + static_cast<std::uintptr_t>(lhs_size);
  const auto rhs_end = rhs_begin + static_cast<std::uintptr_t>(rhs_size);
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

bool ResourceMatches(const ResourceHandle &lhs,
                     const ResourceHandle &rhs) noexcept {
  return lhs.resource_id == rhs.resource_id &&
         lhs.allocator_owner_id == rhs.allocator_owner_id &&
         lhs.lifetime_generation == rhs.lifetime_generation &&
         lhs.memory_domain == rhs.memory_domain &&
         lhs.lifetime == rhs.lifetime &&
         lhs.synchronization_state == rhs.synchronization_state &&
         lhs.execution_domain == rhs.execution_domain &&
         lhs.size_bytes == rhs.size_bytes &&
         lhs.alignment_bytes == rhs.alignment_bytes &&
         lhs.mutable_resource == rhs.mutable_resource &&
         lhs.aliasable == rhs.aliasable &&
         lhs.adapter_opaque == rhs.adapter_opaque &&
         lhs.destruction_owner == rhs.destruction_owner &&
         lhs.destroy == rhs.destroy;
}

std::uint64_t NextNonzero(std::uint64_t *counter) noexcept {
  if (!counter)
    return 0;
  if (*counter == 0)
    *counter = 1;
  const std::uint64_t value = *counter;
  ++*counter;
  if (*counter == 0)
    *counter = 1;
  return value;
}

hal::ExecuteResult Failed(StatusId status, ReasonId reason) noexcept {
  hal::ExecuteResult result;
  result.status = status;
  result.reason = reason;
  return result;
}

} // namespace

VulkanHalResidentAdapter::VulkanHalResidentAdapter(
    std::uint64_t adapter_instance_id,
    IQualifiedVulkanResidentExecutor &executor) noexcept
    : adapter_instance_id_(adapter_instance_id), executor_(&executor) {}

VulkanHalResidentAdapter::~VulkanHalResidentAdapter() { Shutdown(); }

bool VulkanHalResidentAdapter::SetProgramOptions(
    const VulkanResidentProgramOptions &options) noexcept {
  if (shutdown_ || options.program_id == 0 ||
      options.canonical_plan_hash == 0 ||
      options.vignette_intensity_percent < 0 ||
      options.vignette_intensity_percent > 100) {
    return false;
  }
  pending_options_ = options;
  pending_options_valid_ = true;
  return true;
}

StatusId VulkanHalResidentAdapter::Prepare(
    const hal::PreparedProgram &program, hal::PreparedProgramHandle *candidate,
    hal::ReadinessSnapshot *readiness, ReasonId *reason) noexcept {
  if (candidate)
    *candidate = {};
  if (readiness) {
    *readiness = {};
    readiness->requested_backend = program.requested_backend;
    readiness->selected_backend = BackendId::vulkan;
    readiness->provider =
        SemanticEffectProviderId::open_vulkan_production_kernels;
    readiness->execution_domain = program.execution_domain;
  }
  SetReason(reason, ReasonId::none);

  const auto fail = [&](StatusId status, ReasonId why) {
    SetReason(reason, why);
    if (readiness) {
      readiness->status = status;
      readiness->reason = why;
      readiness->available = false;
    }
    return status;
  };

  if (shutdown_)
    return fail(StatusId::stopped, ReasonId::shutdown_requested);
  if (!candidate || adapter_instance_id_ == 0 || !executor_)
    return fail(StatusId::blocked, ReasonId::foreign_adapter);

  ReasonId validation_reason = ReasonId::none;
  if (!hal::ValidatePreparedProgram(program, &validation_reason))
    return fail(validation_reason == ReasonId::resource_exhausted
                    ? StatusId::resource_exhausted
                    : StatusId::blocked,
                validation_reason);
  if (program.owner_adapter_instance_id != adapter_instance_id_ ||
      program.execution_domain.adapter_instance_id != adapter_instance_id_ ||
      program.execution_domain.backend != BackendId::vulkan) {
    return fail(StatusId::blocked, ReasonId::foreign_adapter);
  }
  if (!program.requires_upload || !program.requires_final_readback)
    return fail(StatusId::blocked, ReasonId::malformed_contract);
  if (program.temporal.temporal)
    return fail(StatusId::blocked, ReasonId::semantic_effect_unsupported);
  if (program.cpu_tail_count != 0)
    return fail(StatusId::blocked, ReasonId::cpu_tail_required);
  // This route has no setup-proven metadata crossings.
  if (program.maximum_metadata_readbacks_per_frame != 0 ||
      program.maximum_metadata_cpu_tails_per_frame != 0) {
    return fail(StatusId::blocked, ReasonId::transfer_contract_violated);
  }
  if (!IsExactHostRgbImage(program.host_input, hal::Mutability::read_only) ||
      !IsExactHostRgbImage(program.host_output, hal::Mutability::write_only)) {
    return fail(StatusId::blocked, ReasonId::malformed_descriptor);
  }
  if (!pending_options_valid_ ||
      pending_options_.program_id != program.program_id ||
      pending_options_.canonical_plan_hash !=
          program.invalidation_key.canonical_plan_hash) {
    return fail(StatusId::blocked, ReasonId::reconfiguration_required);
  }

  bool has_mirror = false;
  bool has_vignette = false;
  const hal::ImageDescriptor *previous_output = nullptr;
  for (std::size_t index = 0; index < program.stage_count; ++index) {
    const auto &stage = program.stages[index];
    if (stage.backend != BackendId::vulkan ||
        stage.provider !=
            SemanticEffectProviderId::open_vulkan_production_kernels ||
        stage.execution_domain != program.execution_domain) {
      return fail(StatusId::blocked,
                  stage.execution_domain.adapter_instance_id !=
                          adapter_instance_id_
                      ? ReasonId::cross_adapter_residency_forbidden
                      : ReasonId::semantic_provider_unavailable);
    }
    if (stage.input.kind != hal::StageIoKind::image ||
        stage.output.kind != hal::StageIoKind::image ||
        !IsExactProductionRgbImage(stage.input.image,
                                   hal::Mutability::read_only) ||
        !IsExactProductionRgbImage(stage.output.image,
                                   hal::Mutability::read_write) ||
        (previous_output &&
         !SameImageStorage(*previous_output, stage.input.image)) ||
        !IsZeroModelIdentity(stage.model)) {
      return fail(StatusId::blocked, ReasonId::malformed_descriptor);
    }
    if (stage.semantic == SemanticEffectId::mirror) {
      if (has_mirror)
        return fail(StatusId::blocked, ReasonId::malformed_contract);
      has_mirror = true;
    } else if (stage.semantic == SemanticEffectId::vignette) {
      if (has_vignette || has_mirror)
        return fail(StatusId::blocked, ReasonId::malformed_contract);
      has_vignette = true;
    } else {
      if (readiness && index < 32)
        readiness->blocked_stage_mask |= (1u << index);
      return fail(StatusId::blocked, ReasonId::semantic_effect_unsupported);
    }
    if (!stage.required &&
        stage.failure_action != FailureAction::fail_open_preserve_input) {
      return fail(StatusId::blocked, ReasonId::malformed_contract);
    }
    previous_output = &stage.output.image;
  }
  if (has_mirror != pending_options_.request_mirror ||
      has_vignette != pending_options_.request_vignette) {
    return fail(StatusId::blocked, ReasonId::malformed_contract);
  }
  if (has_vignette && pending_options_.vignette_center !=
                          VulkanVignetteCenterMode::fixed_frame_center) {
    return fail(StatusId::blocked, ReasonId::semantic_effect_unsupported);
  }
  const std::uint64_t required_upload = RequiredImageSpan(program.host_input);
  const std::uint64_t required_readback =
      RequiredImageSpan(program.host_output);
  if (required_upload == 0 || required_readback == 0 ||
      required_upload > program.budget.upload_staging_bytes ||
      required_readback > program.budget.readback_staging_bytes) {
    return fail(StatusId::resource_exhausted, ReasonId::resource_exhausted);
  }

  const VulkanResidentCapabilityEvidence evidence =
      executor_->CapabilityEvidence();
  if (evidence.execution_domain != program.execution_domain ||
      !CommonEvidenceReady(evidence) ||
      (has_mirror && !evidence.mirror_production_ready) ||
      (has_vignette && !evidence.vignette_fixed_center_production_ready)) {
    return fail(StatusId::blocked,
                evidence.blocker == ReasonId::none
                    ? ReasonId::capability_evidence_incomplete
                    : evidence.blocker);
  }

  PreparedSlot *slot = nullptr;
  for (auto &entry : slots_) {
    if (!entry.occupied) {
      slot = &entry;
      break;
    }
  }
  if (!slot)
    return fail(StatusId::resource_exhausted, ReasonId::resource_exhausted);

  VulkanExecutorPreparedProgram executor_candidate;
  ReasonId executor_reason = ReasonId::none;
  const StatusId executor_status = executor_->Prepare(
      program, pending_options_, &executor_candidate, &executor_reason);
  if (executor_status != StatusId::ready) {
    if (executor_candidate.state_id != 0)
      executor_->DestroyPrepared(&executor_candidate);
    return fail(executor_status, executor_reason == ReasonId::none
                                     ? ReasonId::capability_evidence_incomplete
                                     : executor_reason);
  }
  if (executor_candidate.state_id == 0 ||
      executor_candidate.reserved_budget.in_flight_slots != 1 ||
      !BudgetContains(program.budget, executor_candidate.reserved_budget)) {
    if (executor_candidate.state_id != 0)
      executor_->DestroyPrepared(&executor_candidate);
    return fail(StatusId::resource_exhausted, ReasonId::resource_exhausted);
  }

  hal::PreparedProgramHandle prepared_handle;
  prepared_handle.handle_id = NextNonzero(&next_handle_id_);
  prepared_handle.program_id = program.program_id;
  prepared_handle.adapter_instance_id = adapter_instance_id_;
  prepared_handle.invalidation_key = program.invalidation_key;
  prepared_handle.execution_domain = program.execution_domain;
  prepared_handle.reserved_budget = executor_candidate.reserved_budget;
  prepared_handle.adapter_opaque = {executor_candidate.state_id,
                                    prepared_handle.handle_id};

  slot->occupied = true;
  slot->program = program;
  slot->handle = prepared_handle;
  slot->options = pending_options_;
  slot->executor_program = executor_candidate;
  pending_options_valid_ = false;

  *candidate = prepared_handle;
  const bool explicit_fallback =
      !program.strict_backend_selection &&
      program.requested_backend != program.execution_domain.backend;
  const bool degraded = explicit_fallback;
  const ReasonId degraded_reason = ReasonId::semantic_provider_unavailable;
  if (readiness) {
    readiness->status = degraded ? StatusId::degraded : StatusId::ready;
    readiness->reason = degraded ? degraded_reason : ReasonId::none;
    readiness->available = true;
    readiness->degraded = degraded;
    readiness->ready_stage_mask =
        program.stage_count == 32 ? std::numeric_limits<std::uint32_t>::max()
                                  : ((1u << program.stage_count) - 1u);
  }
  SetReason(reason, degraded ? degraded_reason : ReasonId::none);
  return degraded ? StatusId::degraded : StatusId::ready;
}

StatusId VulkanHalResidentAdapter::BindFrame(
    std::uint64_t program_id, std::uint64_t capture_sequence,
    const VulkanHostFrameBinding &binding, ResourceHandle *input,
    ReasonId *reason) noexcept {
  if (input)
    *input = {};
  SetReason(reason, ReasonId::none);
  if (shutdown_) {
    SetReason(reason, ReasonId::shutdown_requested);
    return StatusId::stopped;
  }
  PreparedSlot *slot = FindSlot(program_id);
  if (!slot) {
    SetReason(reason, ReasonId::stale_program);
    return StatusId::stale;
  }
  if (!input) {
    SetReason(reason, ReasonId::invalid_resource);
    return StatusId::blocked;
  }

  hal::HostFrameBinding host_binding;
  host_binding.input_descriptor = slot->program.host_input;
  host_binding.output_descriptor = slot->program.host_output;
  host_binding.input_plane_count = 1;
  host_binding.output_plane_count = 1;
  host_binding.capture_sequence = capture_sequence;
  host_binding.ownership = hal::OwnershipModel::borrowed_until_completion;
  host_binding.lifetime = ResourceLifetime::frame_borrowed;
  host_binding.input_planes[0].read_data =
      reinterpret_cast<const std::byte *>(binding.input);
  host_binding.input_planes[0].span_bytes = binding.input_size_bytes;
  host_binding.input_planes[0].row_stride_bytes =
      binding.input_row_stride_bytes;
  host_binding.output_planes[0].write_data =
      reinterpret_cast<std::byte *>(binding.output);
  host_binding.output_planes[0].span_bytes = binding.output_size_bytes;
  host_binding.output_planes[0].row_stride_bytes =
      binding.output_row_stride_bytes;

  hal::HostFrameTokens tokens;
  const StatusId status =
      ArmHostFrame(slot, host_binding, false, &tokens, reason);
  if (status == StatusId::ready)
    *input = tokens.input;
  return status;
}

StatusId VulkanHalResidentAdapter::BindHostFrame(
    const hal::PreparedProgramHandle &program,
    const hal::HostFrameBinding &binding, hal::HostFrameTokens *tokens,
    ReasonId *reason) noexcept {
  if (tokens)
    *tokens = {};
  SetReason(reason, ReasonId::none);
  if (shutdown_) {
    SetReason(reason, ReasonId::shutdown_requested);
    return StatusId::stopped;
  }
  PreparedSlot *slot = FindSlot(program);
  if (!slot) {
    SetReason(reason, ReasonId::stale_program);
    return StatusId::stale;
  }
  return ArmHostFrame(slot, binding, true, tokens, reason);
}

StatusId VulkanHalResidentAdapter::ArmHostFrame(
    PreparedSlot *slot, const hal::HostFrameBinding &binding,
    bool neutral_binding, hal::HostFrameTokens *tokens,
    ReasonId *reason) noexcept {
  if (tokens)
    *tokens = {};
  SetReason(reason, ReasonId::none);
  if (!slot || !tokens) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return StatusId::blocked;
  }
  if (slot->quarantined) {
    SetReason(reason, ReasonId::device_lost);
    return StatusId::quarantined;
  }
  if (slot->executing || slot->binding_armed) {
    SetReason(reason, ReasonId::output_not_committed);
    return StatusId::blocked;
  }

  ReasonId validation_reason = ReasonId::none;
  if (!hal::ValidateHostFrameBinding(binding, slot->program,
                                     &validation_reason)) {
    SetReason(reason, validation_reason);
    return StatusId::blocked;
  }

  const auto &input_descriptor = binding.input_descriptor;
  const auto &output_descriptor = binding.output_descriptor;
  const std::uint64_t required_input = RequiredImageSpan(input_descriptor);
  const std::uint64_t required_output = RequiredImageSpan(output_descriptor);
  const auto *input_pointer = binding.input_planes[0].read_data;
  auto *output_pointer = binding.output_planes[0].write_data;
  if (required_input == 0 || required_output == 0 ||
      required_input > slot->program.budget.upload_staging_bytes ||
      required_output > slot->program.budget.readback_staging_bytes ||
      required_input > std::numeric_limits<std::size_t>::max() ||
      required_output > std::numeric_limits<std::size_t>::max() ||
      binding.input_planes[0].row_stride_bytes >
          std::numeric_limits<std::size_t>::max() ||
      binding.output_planes[0].row_stride_bytes >
          std::numeric_limits<std::size_t>::max() ||
      !PointerSpanValid(input_pointer, required_input) ||
      !PointerSpanValid(output_pointer, required_output) ||
      reinterpret_cast<std::uintptr_t>(input_pointer) %
              input_descriptor.base_alignment_bytes !=
          0 ||
      reinterpret_cast<std::uintptr_t>(output_pointer) %
              output_descriptor.base_alignment_bytes !=
          0) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return StatusId::blocked;
  }

  const bool exact_alias =
      input_pointer == output_pointer && required_input == required_output;
  const bool alias_allowed =
      exact_alias && input_descriptor.in_place_permitted &&
      output_descriptor.in_place_permitted &&
      input_descriptor.aliasing != hal::AliasingPermission::forbidden &&
      output_descriptor.aliasing != hal::AliasingPermission::forbidden;
  if (SpansOverlap(input_pointer, required_input, output_pointer,
                   required_output) &&
      !alias_allowed) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return StatusId::blocked;
  }

  const std::uint64_t binding_id = NextNonzero(&next_binding_id_);
  ResourceHandle input;
  input.resource_id = NextNonzero(&next_resource_id_);
  input.allocator_owner_id = adapter_instance_id_;
  input.lifetime_generation =
      slot->program.execution_domain.logical_context_generation;
  input.memory_domain = MemoryDomain::upload_staging;
  input.lifetime = ResourceLifetime::frame_borrowed;
  input.synchronization_state = SynchronizationState::ready;
  input.execution_domain = slot->program.execution_domain;
  input.size_bytes = required_input;
  input.alignment_bytes = input_descriptor.base_alignment_bytes;
  input.mutable_resource = false;
  input.aliasable = alias_allowed;
  input.adapter_opaque = {slot->handle.handle_id, binding.capture_sequence,
                          binding_id, 1};

  ResourceHandle output;
  output.resource_id = NextNonzero(&next_resource_id_);
  output.allocator_owner_id = adapter_instance_id_;
  output.lifetime_generation =
      slot->program.execution_domain.logical_context_generation;
  output.memory_domain = MemoryDomain::readback_staging;
  output.lifetime = ResourceLifetime::frame_borrowed;
  output.synchronization_state = SynchronizationState::ready;
  output.execution_domain = slot->program.execution_domain;
  output.size_bytes = required_output;
  output.alignment_bytes = output_descriptor.base_alignment_bytes;
  output.mutable_resource = true;
  output.aliasable = alias_allowed;
  output.adapter_opaque = {slot->handle.handle_id, binding.capture_sequence,
                           binding_id, 2};

  if (!hal::ValidateResourceHandle(
          input, slot->program.execution_domain,
          slot->program.execution_domain.logical_context_generation,
          slot->program.budget) ||
      !hal::ValidateResourceHandle(
          output, slot->program.execution_domain,
          slot->program.execution_domain.logical_context_generation,
          slot->program.budget)) {
    SetReason(reason, ReasonId::resource_exhausted);
    return StatusId::resource_exhausted;
  }

  VulkanHostFrameBinding executor_binding;
  executor_binding.input =
      reinterpret_cast<const std::uint8_t *>(input_pointer);
  executor_binding.input_size_bytes = static_cast<std::size_t>(required_input);
  executor_binding.input_row_stride_bytes =
      static_cast<std::size_t>(binding.input_planes[0].row_stride_bytes);
  executor_binding.output = reinterpret_cast<std::uint8_t *>(output_pointer);
  executor_binding.output_size_bytes =
      static_cast<std::size_t>(required_output);
  executor_binding.output_row_stride_bytes =
      static_cast<std::size_t>(binding.output_planes[0].row_stride_bytes);

  slot->binding = executor_binding;
  slot->host_binding = binding;
  slot->binding_resource = input;
  slot->binding_output_resource = output;
  slot->binding_capture_sequence = binding.capture_sequence;
  slot->binding_id = binding_id;
  slot->completed_token = {};
  slot->output_ready = false;
  slot->neutral_binding = neutral_binding;
  slot->binding_armed = true;
  tokens->input = input;
  tokens->output = output;
  return StatusId::ready;
}

hal::ExecuteResult
VulkanHalResidentAdapter::Execute(const hal::PreparedProgramHandle &program,
                                  const hal::ExecuteRequest &request) noexcept {
  if (shutdown_)
    return Failed(StatusId::stopped, ReasonId::shutdown_requested);
  PreparedSlot *slot = FindSlot(program);
  if (!slot)
    return Failed(StatusId::stale, ReasonId::stale_program);
  if (slot->quarantined)
    return Failed(StatusId::quarantined, ReasonId::device_lost);
  if (!slot->binding_armed)
    return Failed(StatusId::blocked, ReasonId::invalid_resource);
  const auto reject_bound_frame = [slot](StatusId status,
                                         ReasonId why) noexcept {
    if (slot->neutral_binding) {
      slot->binding_armed = false;
      slot->output_ready = false;
      slot->completed_token = {};
    }
    return Failed(status, why);
  };
  if (request.input.execution_domain != slot->program.execution_domain)
    return reject_bound_frame(StatusId::blocked,
                              ReasonId::wrong_execution_domain);
  if (request.input.lifetime_generation !=
      slot->program.execution_domain.logical_context_generation) {
    return reject_bound_frame(StatusId::blocked, ReasonId::wrong_generation);
  }
  if (!ResourceMatches(request.input, slot->binding_resource))
    return reject_bound_frame(StatusId::blocked, ReasonId::invalid_resource);
  if (request.capture_sequence == 0 ||
      request.capture_sequence != slot->binding_capture_sequence) {
    return reject_bound_frame(StatusId::stale, ReasonId::stale_program);
  }
  if (request.invalidation_key != slot->program.invalidation_key)
    return reject_bound_frame(StatusId::stale,
                              ReasonId::reconfiguration_required);
  if (request.execution_domain != slot->program.execution_domain)
    return reject_bound_frame(StatusId::blocked,
                              ReasonId::wrong_execution_domain);

  const std::uint32_t valid_stage_mask =
      slot->program.stage_count == 32
          ? std::numeric_limits<std::uint32_t>::max()
          : ((std::uint32_t{1} << slot->program.stage_count) - 1u);
  if ((request.disabled_stage_mask & ~valid_stage_mask) != 0)
    return reject_bound_frame(StatusId::blocked, ReasonId::malformed_contract);
  for (std::size_t index = 0; index < slot->program.stage_count; ++index) {
    const std::uint32_t bit = std::uint32_t{1} << index;
    if ((request.disabled_stage_mask & bit) != 0 &&
        slot->program.stages[index].required) {
      return reject_bound_frame(StatusId::failed,
                                ReasonId::required_stage_failed);
    }
  }

  hal::PreparedProgram execution_program = slot->program;
  VulkanResidentProgramOptions execution_options = slot->options;
  execution_program.stages = {};
  execution_program.stage_count = 0;
  for (std::size_t index = 0; index < slot->program.stage_count; ++index) {
    const std::uint32_t bit = std::uint32_t{1} << index;
    if ((request.disabled_stage_mask & bit) != 0) {
      if (slot->program.stages[index].semantic == SemanticEffectId::mirror)
        execution_options.request_mirror = false;
      if (slot->program.stages[index].semantic == SemanticEffectId::vignette)
        execution_options.request_vignette = false;
      continue;
    }
    execution_program.stages[execution_program.stage_count++] =
        slot->program.stages[index];
  }

  slot->executing = true;
  const VulkanExecutorResult executor_result = executor_->Execute(
      slot->executor_program, execution_program, execution_options,
      slot->binding, request.capture_sequence);
  slot->executing = false;
  slot->binding_armed = false;
  slot->output_ready = false;
  slot->completed_token = {};

  hal::ExecuteResult result;
  result.reason = executor_result.reason;
  result.transfers = executor_result.transfers;
  result.transfers.cpu_tail_boundaries += slot->program.cpu_tail_count;
  result.frame_elapsed_ns = executor_result.frame_elapsed_ns;
  if (executor_result.device_lost ||
      executor_result.reason == ReasonId::device_lost) {
    slot->quarantined = true;
    result.status = StatusId::quarantined;
    result.reason = ReasonId::device_lost;
    return result;
  }
  if (executor_result.stage_count != execution_program.stage_count ||
      executor_result.stage_count > executor_result.stages.size()) {
    result.status = StatusId::failed;
    result.reason = ReasonId::malformed_contract;
    return result;
  }

  result.stage_count = slot->program.stage_count;
  std::size_t executed_index = 0;
  for (std::size_t index = 0; index < result.stage_count; ++index) {
    result.stages[index].stage_id = slot->program.stages[index].stage_id;
    result.stages[index].semantic = slot->program.stages[index].semantic;
    result.stages[index].primitive = slot->program.stages[index].primitive;
    const std::uint32_t bit = std::uint32_t{1} << index;
    if ((request.disabled_stage_mask & bit) != 0) {
      result.stages[index].status = StageExecutionStatus::disabled_preserved;
      result.stages[index].reason = ReasonId::circuit_breaker_open;
      result.stages[index].valid_output_preserved = true;
      result.stages[index].elapsed_ns = 0;
      continue;
    }
    const auto &executor_stage = executor_result.stages[executed_index++];
    result.stages[index].reason = executor_stage.reason;
    result.stages[index].elapsed_ns = executor_stage.elapsed_ns;
    if (executor_stage.succeeded) {
      result.stages[index].status = StageExecutionStatus::succeeded;
      result.stages[index].valid_output_preserved = true;
    } else if (slot->program.stages[index].required) {
      result.stages[index].status = StageExecutionStatus::required_failed;
    } else {
      result.stages[index].status = StageExecutionStatus::optional_failed_open;
      result.stages[index].valid_output_preserved =
          executor_result.output_written;
    }
  }

  if (executor_result.status != StatusId::executed) {
    result.status = executor_result.status;
    result.reason = executor_result.reason;
    return result;
  }

  const bool required_failed = std::any_of(
      result.stages.begin(), result.stages.begin() + result.stage_count,
      [](const hal::StageExecutionResult &stage) {
        return stage.status == StageExecutionStatus::required_failed;
      });
  const bool optional_failed = std::any_of(
      result.stages.begin(), result.stages.begin() + result.stage_count,
      [](const hal::StageExecutionResult &stage) {
        return stage.status == StageExecutionStatus::optional_failed_open;
      });
  if (required_failed) {
    result.status = StatusId::failed;
    result.reason = ReasonId::required_stage_failed;
    return result;
  }
  if (!executor_result.output_written ||
      executor_result.transfers.uploads != 1 ||
      executor_result.transfers.final_readbacks != 1 ||
      executor_result.transfers.intermediate_readbacks != 0 ||
      executor_result.transfers.intermediate_uploads != 0 ||
      executor_result.transfers.cpu_tail_boundaries != 0 ||
      executor_result.transfers.synchronizations == 0) {
    result.status = StatusId::failed;
    result.reason = ReasonId::synchronization_failed;
    return result;
  }

  if (!hal::ValidateResourceHandle(
          slot->binding_output_resource, slot->program.execution_domain,
          slot->program.execution_domain.logical_context_generation,
          slot->program.budget) ||
      slot->binding_output_resource.memory_domain !=
          MemoryDomain::readback_staging ||
      slot->binding_output_resource.lifetime !=
          ResourceLifetime::frame_borrowed ||
      slot->binding_output_resource.adapter_opaque[0] !=
          slot->handle.handle_id ||
      slot->binding_output_resource.adapter_opaque[1] !=
          request.capture_sequence ||
      slot->binding_output_resource.adapter_opaque[2] != slot->binding_id ||
      slot->binding_output_resource.adapter_opaque[3] != 2) {
    result.status = StatusId::failed;
    result.reason = ReasonId::invalid_resource;
    return result;
  }

  result.status = optional_failed ? StatusId::degraded : StatusId::executed;
  result.reason =
      optional_failed ? ReasonId::optional_stage_failed_open : ReasonId::none;
  result.output = slot->binding_output_resource;
  result.output_valid = true;
  result.completion.token_id = NextNonzero(&next_completion_id_);
  result.completion.program_id = slot->program.program_id;
  result.completion.adapter_instance_id = adapter_instance_id_;
  result.completion.capture_sequence = request.capture_sequence;
  result.completion.complete = true;
  slot->completed_token = result.completion;
  slot->output_ready = true;
  return result;
}

StatusId VulkanHalResidentAdapter::ResolveHostFrameOutput(
    const hal::PreparedProgramHandle &program,
    const hal::CompletionToken &completion, const ResourceHandle &output,
    hal::HostFrameOutputResolution *resolution, ReasonId *reason) noexcept {
  if (resolution)
    *resolution = {};
  SetReason(reason, ReasonId::none);
  if (shutdown_) {
    SetReason(reason, ReasonId::shutdown_requested);
    return StatusId::stopped;
  }
  PreparedSlot *slot = FindSlot(program);
  if (!slot) {
    SetReason(reason, ReasonId::stale_program);
    return StatusId::stale;
  }
  if (slot->quarantined) {
    SetReason(reason, ReasonId::device_lost);
    return StatusId::quarantined;
  }
  if (!resolution || !slot->output_ready) {
    SetReason(reason, resolution ? ReasonId::output_not_committed
                                 : ReasonId::host_frame_binding_invalid);
    return StatusId::blocked;
  }
  if (completion.token_id == 0 || !completion.complete ||
      completion.token_id != slot->completed_token.token_id ||
      completion.program_id != slot->completed_token.program_id ||
      completion.adapter_instance_id !=
          slot->completed_token.adapter_instance_id ||
      completion.capture_sequence != slot->completed_token.capture_sequence) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return StatusId::blocked;
  }
  if (output.execution_domain != slot->program.execution_domain) {
    SetReason(reason, ReasonId::wrong_execution_domain);
    return StatusId::blocked;
  }
  if (output.lifetime_generation !=
      slot->program.execution_domain.logical_context_generation) {
    SetReason(reason, ReasonId::wrong_generation);
    return StatusId::blocked;
  }
  if (!ResourceMatches(output, slot->binding_output_resource) ||
      output.memory_domain != MemoryDomain::readback_staging ||
      output.lifetime != ResourceLifetime::frame_borrowed ||
      output.adapter_opaque[0] != slot->handle.handle_id ||
      output.adapter_opaque[1] != completion.capture_sequence ||
      output.adapter_opaque[2] != slot->binding_id ||
      output.adapter_opaque[3] != 2) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return StatusId::blocked;
  }

  ReasonId validation_reason = ReasonId::none;
  if (!hal::ValidateHostFrameBinding(slot->host_binding, slot->program,
                                     &validation_reason) ||
      slot->host_binding.capture_sequence != completion.capture_sequence) {
    SetReason(reason, ReasonId::host_frame_binding_invalid);
    return StatusId::blocked;
  }

  resolution->descriptor = slot->host_binding.output_descriptor;
  resolution->planes = slot->host_binding.output_planes;
  resolution->plane_count = slot->host_binding.output_plane_count;
  resolution->capture_sequence = slot->host_binding.capture_sequence;
  resolution->output_ready = true;
  return StatusId::ready;
}

StatusId VulkanHalResidentAdapter::ResetTemporalState(
    const hal::PreparedProgramHandle &program, ReasonId) noexcept {
  if (shutdown_)
    return StatusId::stopped;
  return FindSlot(program) ? StatusId::ready : StatusId::stale;
}

void VulkanHalResidentAdapter::DestroyPrepared(
    hal::PreparedProgramHandle *program) noexcept {
  if (!program)
    return;
  PreparedSlot *slot = FindSlot(*program);
  if (slot)
    ReleaseSlot(slot);
  *program = {};
}

void VulkanHalResidentAdapter::Shutdown() noexcept {
  if (shutdown_)
    return;
  for (auto &slot : slots_) {
    if (slot.occupied)
      ReleaseSlot(&slot);
  }
  if (executor_)
    executor_->Shutdown();
  pending_options_valid_ = false;
  shutdown_ = true;
}

VulkanHalResidentAdapter::PreparedSlot *VulkanHalResidentAdapter::FindSlot(
    const hal::PreparedProgramHandle &program) noexcept {
  for (auto &slot : slots_) {
    if (slot.occupied && slot.handle.handle_id == program.handle_id &&
        slot.handle.program_id == program.program_id &&
        slot.handle.adapter_instance_id == program.adapter_instance_id &&
        slot.handle.invalidation_key == program.invalidation_key &&
        slot.handle.execution_domain == program.execution_domain &&
        slot.handle.reserved_budget.persistent_bytes ==
            program.reserved_budget.persistent_bytes &&
        slot.handle.reserved_budget.scratch_bytes ==
            program.reserved_budget.scratch_bytes &&
        slot.handle.reserved_budget.upload_staging_bytes ==
            program.reserved_budget.upload_staging_bytes &&
        slot.handle.reserved_budget.readback_staging_bytes ==
            program.reserved_budget.readback_staging_bytes &&
        slot.handle.reserved_budget.resource_count ==
            program.reserved_budget.resource_count &&
        slot.handle.reserved_budget.in_flight_slots ==
            program.reserved_budget.in_flight_slots &&
        slot.handle.adapter_opaque == program.adapter_opaque) {
      return &slot;
    }
  }
  return nullptr;
}

VulkanHalResidentAdapter::PreparedSlot *
VulkanHalResidentAdapter::FindSlot(std::uint64_t program_id) noexcept {
  for (auto &slot : slots_) {
    if (slot.occupied && slot.program.program_id == program_id)
      return &slot;
  }
  return nullptr;
}

void VulkanHalResidentAdapter::ReleaseSlot(PreparedSlot *slot) noexcept {
  if (!slot || !slot->occupied)
    return;
  if (executor_ && slot->executor_program.state_id != 0)
    executor_->DestroyPrepared(&slot->executor_program);
  *slot = {};
}

} // namespace studiocast::vulkan
