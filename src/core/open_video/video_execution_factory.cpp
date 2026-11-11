#include "core/open_video/video_execution_factory.h"
#include "core/open_video/open_cuda_selection_policy.h"
#include "core/video/production_video_backend_manager.h"

#include <limits>
#include <new>
#include <utility>

namespace studiocast::open_cuda {
namespace {

constexpr auto kProvider =
    hal::SemanticEffectProviderId::open_video_onnxruntime_cuda;
constexpr auto kNativeProvider =
    hal::SemanticEffectProviderId::open_cuda_production_kernels;
constexpr std::size_t kMaxCanonicalCudaStageCount = 7;

void SetReason(hal::ReasonId *reason, hal::ReasonId value) noexcept {
  if (reason != nullptr)
    *reason = value;
}

bool IsSetupSuccess(hal::StatusId status) noexcept {
  return status == hal::StatusId::ready || status == hal::StatusId::degraded;
}

bool SameModel(const hal::ModelArtifactIdentity &left,
               const hal::ModelArtifactIdentity &right) noexcept {
  return left.model_id_hash == right.model_id_hash &&
         left.artifact_set_hash == right.artifact_set_hash &&
         left.artifact_generation == right.artifact_generation;
}

bool SameBudget(const hal::ResourceBudget &left,
                const hal::ResourceBudget &right) noexcept;

bool SamePlane(const hal::PlaneDescriptor &left,
               const hal::PlaneDescriptor &right) noexcept {
  return left.offset_bytes == right.offset_bytes &&
         left.row_stride_bytes == right.row_stride_bytes &&
         left.size_bytes == right.size_bytes && left.width == right.width &&
         left.height == right.height &&
         left.alignment_bytes == right.alignment_bytes;
}

bool SameImage(const hal::ImageDescriptor &left,
               const hal::ImageDescriptor &right) noexcept {
  bool exact =
      left.width == right.width && left.height == right.height &&
      left.plane_count == right.plane_count && left.format == right.format &&
      left.channel_order == right.channel_order &&
      left.color_space == right.color_space &&
      left.color_range == right.color_range &&
      left.alpha_meaning == right.alpha_meaning &&
      left.orientation == right.orientation &&
      left.host_coherence == right.host_coherence &&
      left.mutability == right.mutability && left.aliasing == right.aliasing &&
      left.in_place_permitted == right.in_place_permitted &&
      left.base_alignment_bytes == right.base_alignment_bytes;
  for (std::size_t index = 0; exact && index < left.planes.size(); ++index)
    exact = SamePlane(left.planes[index], right.planes[index]);
  return exact;
}

bool SameTensor(const hal::TensorDescriptor &left,
                const hal::TensorDescriptor &right) noexcept {
  return left.element_type == right.element_type &&
         left.layout == right.layout &&
         left.channel_order == right.channel_order &&
         left.color_space == right.color_space &&
         left.color_range == right.color_range &&
         left.alpha_meaning == right.alpha_meaning &&
         left.orientation == right.orientation &&
         left.host_coherence == right.host_coherence &&
         left.mutability == right.mutability &&
         left.aliasing == right.aliasing &&
         left.in_place_permitted == right.in_place_permitted &&
         left.rank == right.rank &&
         left.base_alignment_bytes == right.base_alignment_bytes &&
         left.dimensions == right.dimensions &&
         left.strides_bytes == right.strides_bytes;
}

bool SameStageIo(const hal::StageIoDescriptor &left,
                 const hal::StageIoDescriptor &right) noexcept {
  return left.kind == right.kind && SameImage(left.image, right.image) &&
         SameTensor(left.tensor, right.tensor);
}

bool SameTail(const hal::DeclaredCpuTail &left,
              const hal::DeclaredCpuTail &right) noexcept {
  return left.id == right.id && left.after_stage_id == right.after_stage_id &&
         left.scratch_bytes == right.scratch_bytes &&
         left.required == right.required && left.placement == right.placement &&
         left.before_stage_id == right.before_stage_id &&
         left.boundary_index == right.boundary_index;
}

bool SameTemporal(const hal::TemporalContract &left,
                  const hal::TemporalContract &right) noexcept {
  return left.temporal == right.temporal &&
         left.requires_contiguous_capture_sequence ==
             right.requires_contiguous_capture_sequence &&
         left.frame_drop_behavior == right.frame_drop_behavior &&
         left.discontinuity_action == right.discontinuity_action &&
         left.history_frames == right.history_frames;
}

bool SamePreparedProgram(const hal::PreparedProgram &left,
                         const hal::PreparedProgram &right) noexcept {
  bool exact =
      left.contract_version == right.contract_version &&
      left.program_id == right.program_id &&
      left.owner_adapter_instance_id == right.owner_adapter_instance_id &&
      left.execution_domain == right.execution_domain &&
      SameImage(left.host_input, right.host_input) &&
      SameImage(left.host_output, right.host_output) &&
      left.invalidation_key == right.invalidation_key &&
      SameTemporal(left.temporal, right.temporal) &&
      SameBudget(left.budget, right.budget) &&
      left.setup_failure_policy == right.setup_failure_policy &&
      left.stage_count == right.stage_count &&
      left.maximum_metadata_readbacks_per_frame ==
          right.maximum_metadata_readbacks_per_frame &&
      left.maximum_metadata_cpu_tails_per_frame ==
          right.maximum_metadata_cpu_tails_per_frame &&
      left.cpu_tail_count == right.cpu_tail_count &&
      left.requested_backend == right.requested_backend &&
      left.strict_backend_selection == right.strict_backend_selection &&
      left.requested_provider == right.requested_provider &&
      left.strict_provider_selection == right.strict_provider_selection &&
      left.requested_execution_domain == right.requested_execution_domain &&
      left.strict_execution_domain == right.strict_execution_domain &&
      left.requires_upload == right.requires_upload &&
      left.requires_final_readback == right.requires_final_readback;
  for (std::size_t index = 0; exact && index < left.stage_count; ++index) {
    const auto &a = left.stages[index];
    const auto &b = right.stages[index];
    exact = a.stage_id == b.stage_id && a.semantic == b.semantic &&
            a.provider == b.provider && a.backend == b.backend &&
            a.execution_domain == b.execution_domain &&
            SameStageIo(a.input, b.input) && SameStageIo(a.output, b.output) &&
            SameModel(a.model, b.model) && a.required == b.required &&
            a.failure_action == b.failure_action &&
            a.circuit_breaker.consecutive_failure_limit ==
                b.circuit_breaker.consecutive_failure_limit &&
            a.circuit_breaker.cooldown_frames ==
                b.circuit_breaker.cooldown_frames &&
            a.circuit_breaker.action == b.circuit_breaker.action &&
            a.scratch_bytes == b.scratch_bytes && a.primitive == b.primitive;
  }
  for (std::size_t index = 0; exact && index < left.cpu_tail_count; ++index)
    exact = SameTail(left.cpu_tails[index], right.cpu_tails[index]);
  return exact;
}

bool SameTransfers(const video::PerFrameTransferContract &left,
                   const video::PerFrameTransferContract &right) noexcept {
  return left.contract_version == right.contract_version &&
         left.uploads == right.uploads &&
         left.final_readbacks == right.final_readbacks &&
         left.intermediate_readbacks == right.intermediate_readbacks &&
         left.intermediate_uploads == right.intermediate_uploads &&
         left.cpu_tail_boundaries == right.cpu_tail_boundaries &&
         left.minimum_synchronizations == right.minimum_synchronizations &&
         left.maximum_synchronizations == right.maximum_synchronizations &&
         left.variable_boundaries_declared ==
             right.variable_boundaries_declared &&
         left.minimum_intermediate_readbacks ==
             right.minimum_intermediate_readbacks &&
         left.maximum_intermediate_readbacks ==
             right.maximum_intermediate_readbacks &&
         left.minimum_intermediate_uploads ==
             right.minimum_intermediate_uploads &&
         left.maximum_intermediate_uploads ==
             right.maximum_intermediate_uploads &&
         left.minimum_cpu_tail_boundaries ==
             right.minimum_cpu_tail_boundaries &&
         left.maximum_cpu_tail_boundaries == right.maximum_cpu_tail_boundaries;
}

bool SameIntent(const video::NormalizedVideoExecutionIntent &left,
                const video::NormalizedVideoExecutionIntent &right) noexcept {
  bool exact =
      left.backend.mode == right.backend.mode &&
      left.backend.backend == right.backend.backend &&
      left.backend.strict == right.backend.strict &&
      left.preferences.compute_family == right.preferences.compute_family &&
      left.preferences.effects_engine == right.preferences.effects_engine &&
      left.preferences.strict_effects_engine ==
          right.preferences.strict_effects_engine &&
      left.requested_provider == right.requested_provider &&
      left.strict_provider_selection == right.strict_provider_selection &&
      left.stage_count == right.stage_count;
  for (std::size_t index = 0; exact && index < left.stage_count; ++index) {
    const auto &a = left.stages[index];
    const auto &b = right.stages[index];
    exact = a.intent.stage_id == b.intent.stage_id &&
            a.intent.semantic == b.intent.semantic &&
            a.intent.required == b.intent.required &&
            a.intent.failure_action == b.intent.failure_action &&
            a.intent.requires_tracked_vignette_center ==
                b.intent.requires_tracked_vignette_center &&
            a.parameter_count == b.parameter_count &&
            SameModel(a.model, b.model);
    for (std::size_t parameter = 0; exact && parameter < a.parameter_count;
         ++parameter) {
      const auto &x = a.parameters[parameter];
      const auto &y = b.parameters[parameter];
      exact = x.id == y.id && x.kind == y.kind &&
              x.signed_value == y.signed_value &&
              x.unsigned_value == y.unsigned_value &&
              x.real_value == y.real_value &&
              x.boolean_value == y.boolean_value;
    }
  }
  return exact;
}

bool SameResolution(const hal::ResolutionResult &left,
                    const hal::ResolutionResult &right) noexcept {
  bool exact =
      left.request_mode == right.request_mode &&
      left.strict_backend_selection == right.strict_backend_selection &&
      left.status == right.status && left.reason == right.reason &&
      left.requested_backend == right.requested_backend &&
      left.preferences.compute_family == right.preferences.compute_family &&
      left.preferences.effects_engine == right.preferences.effects_engine &&
      left.preferences.strict_effects_engine ==
          right.preferences.strict_effects_engine &&
      left.selected_backend == right.selected_backend &&
      left.active_backend == right.active_backend &&
      left.selected_provider == right.selected_provider &&
      left.stage_count == right.stage_count &&
      left.available == right.available && left.degraded == right.degraded &&
      left.fallback == right.fallback && left.passthrough == right.passthrough;
  for (std::size_t index = 0; exact && index < left.stage_count; ++index) {
    const auto &a = left.stages[index];
    const auto &b = right.stages[index];
    exact =
        a.intent.stage_id == b.intent.stage_id &&
        a.intent.semantic == b.intent.semantic &&
        a.intent.required == b.intent.required &&
        a.intent.failure_action == b.intent.failure_action &&
        a.intent.requires_tracked_vignette_center ==
            b.intent.requires_tracked_vignette_center &&
        a.status == b.status && a.reason == b.reason &&
        a.selected_backend == b.selected_backend && a.provider == b.provider &&
        a.adapter_instance_id == b.adapter_instance_id &&
        a.execution_domain == b.execution_domain && a.cpu_tail == b.cpu_tail &&
        a.cpu_tail_scratch_bytes == b.cpu_tail_scratch_bytes &&
        a.cpu_tail_placement == b.cpu_tail_placement &&
        a.evidence_count == b.evidence_count &&
        a.skipped_optional == b.skipped_optional &&
        a.fallback_selected == b.fallback_selected;
    for (std::size_t evidence = 0; exact && evidence < a.evidence_count;
         ++evidence) {
      const auto &x = a.evidence[evidence];
      const auto &y = b.evidence[evidence];
      exact = x.code == y.code && x.status == y.status &&
              x.reason == y.reason && x.semantic == y.semantic &&
              x.backend == y.backend && x.satisfied == y.satisfied;
    }
  }
  return exact;
}

bool SameDomainPlan(const hal::DomainPlan &left,
                    const hal::DomainPlan &right) noexcept {
  bool exact =
      left.status == right.status && left.reason == right.reason &&
      left.island_count == right.island_count &&
      left.boundary_count == right.boundary_count &&
      left.expected_transfers.uploads == right.expected_transfers.uploads &&
      left.expected_transfers.final_readbacks ==
          right.expected_transfers.final_readbacks &&
      left.expected_transfers.intermediate_readbacks ==
          right.expected_transfers.intermediate_readbacks &&
      left.expected_transfers.intermediate_uploads ==
          right.expected_transfers.intermediate_uploads &&
      left.expected_transfers.synchronizations ==
          right.expected_transfers.synchronizations &&
      left.expected_transfers.cpu_tail_boundaries ==
          right.expected_transfers.cpu_tail_boundaries &&
      left.passthrough == right.passthrough;
  for (std::size_t index = 0; exact && index < left.island_count; ++index) {
    const auto &a = left.islands[index];
    const auto &b = right.islands[index];
    exact = a.adapter_instance_id == b.adapter_instance_id &&
            a.resident_adapter == b.resident_adapter &&
            a.execution_domain == b.execution_domain &&
            a.stage_count == b.stage_count &&
            a.requires_upload == b.requires_upload &&
            a.requires_final_readback == b.requires_final_readback;
    for (std::size_t stage = 0; exact && stage < a.stage_count; ++stage)
      exact =
          a.resolved_stage_indices[stage] == b.resolved_stage_indices[stage];
  }
  for (std::size_t index = 0; exact && index < left.boundary_count; ++index) {
    const auto &a = left.boundaries[index];
    const auto &b = right.boundaries[index];
    exact = a.kind == b.kind && a.after_stage_id == b.after_stage_id &&
            a.before_stage_id == b.before_stage_id &&
            a.boundary_index == b.boundary_index && a.cpu_tail == b.cpu_tail &&
            a.scratch_bytes == b.scratch_bytes &&
            a.source_domain == b.source_domain &&
            a.destination_domain == b.destination_domain &&
            a.readback_required == b.readback_required &&
            a.following_upload_required == b.following_upload_required;
  }
  return exact;
}

bool SamePrimedPlan(const video::FrozenVideoExecutionPlan &left,
                    const video::FrozenVideoExecutionPlan &right) noexcept {
  bool exact =
      left.contract_version == right.contract_version &&
      left.plan_id == right.plan_id && left.route == right.route &&
      SameIntent(left.intent, right.intent) &&
      SameImage(left.host_input, right.host_input) &&
      SameImage(left.host_output, right.host_output) &&
      left.invalidation_key == right.invalidation_key &&
      SameTemporal(left.temporal, right.temporal) &&
      SameBudget(left.budget, right.budget) &&
      left.cpu_tail_count == right.cpu_tail_count &&
      left.setup_failure_policy == right.setup_failure_policy &&
      left.migration_policy == right.migration_policy &&
      SameResolution(left.resolution, right.resolution) &&
      SameDomainPlan(left.domain_plan, right.domain_plan) &&
      left.factory_id == right.factory_id &&
      SameTransfers(left.per_frame_transfers, right.per_frame_transfers) &&
      SamePreparedProgram(left.prepared_program, right.prepared_program);
  for (std::size_t index = 0; exact && index < left.cpu_tail_count; ++index)
    exact = SameTail(left.cpu_tails[index], right.cpu_tails[index]);
  return exact;
}

bool UsesMatting(hal::SemanticEffectId semantic) noexcept {
  return semantic == hal::SemanticEffectId::virtual_background_blur ||
         semantic == hal::SemanticEffectId::virtual_background_remove ||
         semantic == hal::SemanticEffectId::virtual_background_replace ||
         semantic == hal::SemanticEffectId::virtual_key_light;
}

bool UsesModel(hal::SemanticEffectId semantic) noexcept {
  return UsesMatting(semantic) ||
         semantic == hal::SemanticEffectId::auto_frame ||
         semantic == hal::SemanticEffectId::eye_contact ||
         semantic == hal::SemanticEffectId::video_noise_removal;
}

int StageOrder(hal::SemanticEffectId semantic) noexcept {
  using S = hal::SemanticEffectId;
  switch (semantic) {
  case S::video_noise_removal:
    return 1;
  case S::virtual_background_blur:
  case S::virtual_background_remove:
  case S::virtual_background_replace:
    return 3;
  case S::virtual_key_light:
    return 4;
  case S::auto_frame:
    return 5;
  case S::eye_contact:
    return 2;
  case S::vignette:
    return 6;
  case S::mirror:
    return 7;
  default:
    return 0;
  }
}

hal::SemanticEffectProviderId
ProviderFor(hal::SemanticEffectId semantic) noexcept {
  return UsesModel(semantic) ? kProvider : kNativeProvider;
}

hal::SemanticEffectProviderId AggregateProvider(
    const video::NormalizedVideoExecutionIntent &intent) noexcept {
  if (intent.stage_count == 0)
    return hal::SemanticEffectProviderId::invalid;
  const auto first = ProviderFor(intent.stages[0].intent.semantic);
  for (std::size_t index = 1; index < intent.stage_count; ++index)
    if (ProviderFor(intent.stages[index].intent.semantic) != first)
      return hal::SemanticEffectProviderId::invalid;
  return first;
}

std::size_t
SelectedStageCount(const hal::ResolutionResult &resolution) noexcept {
  std::size_t count = 0;
  for (std::size_t index = 0; index < resolution.stage_count; ++index)
    count += resolution.stages[index].skipped_optional ? 0u : 1u;
  return count;
}

video::NormalizedVideoExecutionIntent
CompactIntent(const video::VideoExecutionPlanRequest &request,
              const hal::ResolutionResult &resolution) noexcept {
  auto compact = request.intent;
  compact.stage_count = 0;
  for (std::size_t index = 0; index < request.intent.stage_count; ++index) {
    if (!resolution.stages[index].skipped_optional)
      compact.stages[compact.stage_count++] = request.intent.stages[index];
  }
  return compact;
}

bool ZeroModel(const hal::ModelArtifactIdentity &model) noexcept {
  return model.model_id_hash == std::array<std::uint8_t, 32>{} &&
         model.artifact_set_hash == std::array<std::uint8_t, 32>{} &&
         model.artifact_generation == 0;
}

bool AddBudgetBytes(std::uint64_t left, std::uint64_t right,
                    std::uint64_t *result) noexcept {
  if (!result || right > std::numeric_limits<std::uint64_t>::max() - left)
    return false;
  *result = left + right;
  return true;
}

bool ValidReservedBudget(const hal::ResourceBudget &budget) noexcept {
  if (budget.resource_count == 0 || budget.in_flight_slots != 1)
    return false;
  std::uint64_t total = 0;
  return AddBudgetBytes(total, budget.persistent_bytes, &total) &&
         AddBudgetBytes(total, budget.scratch_bytes, &total) &&
         AddBudgetBytes(total, budget.upload_staging_bytes, &total) &&
         AddBudgetBytes(total, budget.readback_staging_bytes, &total) &&
         total != 0;
}

bool SameBudget(const hal::ResourceBudget &left,
                const hal::ResourceBudget &right) noexcept {
  return left.persistent_bytes == right.persistent_bytes &&
         left.scratch_bytes == right.scratch_bytes &&
         left.upload_staging_bytes == right.upload_staging_bytes &&
         left.readback_staging_bytes == right.readback_staging_bytes &&
         left.resource_count == right.resource_count &&
         left.in_flight_slots == right.in_flight_slots;
}

bool TightRgb24(const hal::ImageDescriptor &image,
                hal::Mutability mutability) noexcept {
  if (image.width == 0 || image.height == 0 || image.plane_count != 1 ||
      image.format != hal::PixelFormat::rgb_u8 ||
      image.channel_order != hal::ChannelOrder::rgb ||
      image.color_space != hal::ColorSpace::srgb ||
      image.color_range != hal::ColorRange::full ||
      image.alpha_meaning != hal::AlphaMeaning::none ||
      image.orientation != hal::Orientation::top_left ||
      image.host_coherence != hal::HostCoherence::coherent ||
      image.mutability != mutability ||
      image.aliasing != hal::AliasingPermission::forbidden ||
      image.in_place_permitted || image.base_alignment_bytes != 1 ||
      image.planes[0].offset_bytes != 0 ||
      image.planes[0].alignment_bytes != 1 ||
      image.planes[0].width != image.width ||
      image.planes[0].height != image.height) {
    return false;
  }
  const auto stride = static_cast<std::uint64_t>(image.width) * 3u;
  return image.planes[0].row_stride_bytes == stride &&
         image.planes[0].size_bytes ==
             stride * static_cast<std::uint64_t>(image.height);
}

hal::ImageDescriptor ResidentImage(const hal::ImageDescriptor &host,
                                   hal::Mutability mutability) noexcept {
  auto result = host;
  result.host_coherence = hal::HostCoherence::not_host_visible;
  result.mutability = mutability;
  return result;
}

bool SupportedStages(
    const video::NormalizedVideoExecutionIntent &intent) noexcept {
  if (intent.stage_count == 0 || intent.stage_count > hal::kMaxPreparedStages)
    return false;
  int previous = 0;
  for (std::size_t index = 0; index < intent.stage_count; ++index) {
    const auto order = StageOrder(intent.stages[index].intent.semantic);
    if (order <= previous)
      return false;
    previous = order;
  }
  return true;
}

video::PerFrameTransferContract
TransferContract(const OpenCudaVideoExecutionCachedEvidence &cached,
                 const video::NormalizedVideoExecutionIntent &intent,
                 const hal::DomainPlan &domain_plan) noexcept {
  (void)intent;
  if (cached.maximum_metadata_readbacks_per_frame == 0 &&
      cached.maximum_metadata_cpu_tails_per_frame == 0 &&
      cached.maximum_metadata_synchronizations_per_frame == 0)
    return video::BuildPerFrameTransferContract(
        domain_plan, cached.minimum_synchronizations_per_frame,
        cached.maximum_synchronizations_per_frame);
  video::VariablePerFrameTransferBounds bounds;
  bounds.maximum_intermediate_readbacks =
      cached.maximum_metadata_readbacks_per_frame;
  bounds.minimum_cpu_tail_boundaries =
      domain_plan.expected_transfers.cpu_tail_boundaries;
  bounds.maximum_cpu_tail_boundaries =
      bounds.minimum_cpu_tail_boundaries +
      cached.maximum_metadata_cpu_tails_per_frame;
  return video::BuildVariablePerFrameTransferContract(
      domain_plan, cached.minimum_synchronizations_per_frame,
      cached.maximum_synchronizations_per_frame +
          cached.maximum_metadata_synchronizations_per_frame,
      bounds);
}

bool ExactCachedEvidence(const OpenCudaVideoExecutionCachedEvidence &cached,
                         const video::VideoExecutionPlanRequest &request,
                         const hal::ResolutionResult &resolution,
                         const hal::DomainPlan &domain_plan) noexcept {
  const auto compact = CompactIntent(request, resolution);
  const auto transfer_contract = TransferContract(cached, compact, domain_plan);
  hal::ReasonId transfer_reason = hal::ReasonId::none;
  bool has_model_consumer = false;
  for (std::size_t index = 0; index < compact.stage_count; ++index)
    has_model_consumer |= UsesModel(compact.stages[index].intent.semantic);
  if ((has_model_consumer && !cached.model_artifact_validated) ||
      !cached.provider_runtime_validated || !cached.device_context_validated ||
      (has_model_consumer && !cached.cuda_iobinding_validated) ||
      !cached.user_stream_bound || !cached.reusable_resources_supported ||
      (has_model_consumer && !cached.device_preprocess_supported) ||
      !cached.device_postprocess_supported || !cached.one_upload_one_readback ||
      (!cached.no_intermediate_crossings &&
       !transfer_contract.variable_boundaries_declared) ||
      !cached.stop_aware_bounded_wait || !cached.allocation_free_execute ||
      !hal::ValidateExecutionDomain(cached.execution_domain) ||
      domain_plan.island_count != 1 ||
      cached.execution_domain != domain_plan.islands[0].execution_domain ||
      cached.execution_domain.backend != hal::BackendId::cuda ||
      cached.readiness.execution_domain != cached.execution_domain ||
      cached.readiness.requested_backend != hal::BackendId::cuda ||
      cached.readiness.selected_backend != hal::BackendId::cuda ||
      cached.readiness.provider != AggregateProvider(compact) ||
      !cached.readiness.available ||
      (cached.readiness.status != hal::StatusId::ready &&
       cached.readiness.status != hal::StatusId::degraded) ||
      // Readiness masks use prepared-stage slots, not stable stage_id values.
      // The only live Open CUDA route owns slot zero, so its exact mask is 1
      // even when the canonical stage_id is greater than 31.
      cached.stage_count != compact.stage_count ||
      cached.readiness.ready_stage_mask !=
          ((std::uint32_t{1} << compact.stage_count) - 1u) ||
      cached.readiness.blocked_stage_mask != 0 ||
      cached.readiness.disabled_stage_mask != 0 ||
      resolution.selected_backend != hal::BackendId::cuda ||
      resolution.selected_provider != AggregateProvider(compact) ||
      request.invalidation_key.generations != cached.generations ||
      (ValidReservedBudget(cached.reserved_budget) &&
       !SameBudget(request.budget, cached.reserved_budget)) ||
      request.invalidation_key.generations.context !=
          cached.execution_domain.logical_context_generation ||
      !video::ValidatePerFrameTransferContract(transfer_contract, domain_plan,
                                               &transfer_reason)) {
    return false;
  }
  std::size_t prepared_index = 0;
  for (std::size_t index = 0; index < request.intent.stage_count; ++index) {
    if (resolution.stages[index].skipped_optional)
      continue;
    const auto semantic = request.intent.stages[index].intent.semantic;
    if ((UsesModel(semantic) && !SameModel(request.intent.stages[index].model,
                                           cached.models[prepared_index])) ||
        (!UsesModel(semantic) &&
         (!ZeroModel(request.intent.stages[index].model) ||
          !ZeroModel(cached.models[prepared_index]))))
      return false;
    ++prepared_index;
  }
  return true;
}

bool ConservativeRoute(const video::VideoExecutionPlanRequest &request,
                       const hal::ResolutionResult &resolution,
                       const hal::DomainPlan &domain_plan,
                       hal::ReasonId *reason) noexcept {
  SetReason(reason, hal::ReasonId::none);
  const auto compact = CompactIntent(request, resolution);
  if (!AcceptsOpenCudaSelectionIntent(request.intent) ||
      (request.intent.preferences.compute_family !=
           hal::ComputeFamilyAllowance::any &&
       request.intent.preferences.compute_family !=
           hal::ComputeFamilyAllowance::cuda_family) ||
      !SupportedStages(compact) ||
      resolution.stage_count != request.intent.stage_count ||
      domain_plan.island_count != 1 ||
      domain_plan.islands[0].stage_count != compact.stage_count) {
    SetReason(reason, hal::ReasonId::strict_backend_unavailable);
    return false;
  }
  for (std::size_t index = 0; index < request.intent.stage_count; ++index) {
    const auto &intent = request.intent.stages[index].intent;
    if (resolution.stages[index].skipped_optional) {
      if (intent.required) {
        SetReason(reason, hal::ReasonId::required_stage_failed);
        return false;
      }
      continue;
    }
    if (resolution.stages[index].selected_backend != hal::BackendId::cuda ||
        resolution.stages[index].provider != ProviderFor(intent.semantic) ||
        intent.stage_id == 0 ||
        (intent.requires_tracked_vignette_center &&
         intent.semantic != hal::SemanticEffectId::vignette) ||
        (intent.required &&
         intent.failure_action != hal::FailureAction::fail_closed) ||
        (!intent.required &&
         intent.failure_action !=
             hal::FailureAction::fail_open_preserve_input)) {
      SetReason(reason, hal::ReasonId::semantic_effect_unsupported);
      return false;
    }
  }
  bool has_auto_frame = false;
  for (std::size_t index = 0; index < compact.stage_count; ++index) {
    const auto semantic = compact.stages[index].intent.semantic;
    has_auto_frame |= semantic == hal::SemanticEffectId::auto_frame;
  }
  bool optional_auto_frame_skipped = false;
  bool canonical_needs_history = false;
  for (std::size_t index = 0; index < request.intent.stage_count; ++index) {
    const auto &stage = request.intent.stages[index].intent;
    canonical_needs_history |=
        stage.semantic == hal::SemanticEffectId::auto_frame ||
        stage.semantic == hal::SemanticEffectId::eye_contact ||
        stage.semantic == hal::SemanticEffectId::video_noise_removal;
    optional_auto_frame_skipped |=
        stage.semantic == hal::SemanticEffectId::auto_frame &&
        !stage.required && resolution.stages[index].skipped_optional;
  }
  for (std::size_t index = 0; index < compact.stage_count; ++index)
    if (compact.stages[index].intent.requires_tracked_vignette_center &&
        !has_auto_frame && !optional_auto_frame_skipped) {
      SetReason(reason, hal::ReasonId::semantic_effect_unsupported);
      return false;
    }
  if (request.temporal.temporal != canonical_needs_history ||
      (canonical_needs_history &&
       (!request.temporal.temporal ||
        !request.temporal.requires_contiguous_capture_sequence ||
        request.temporal.history_frames == 0))) {
    SetReason(reason, hal::ReasonId::malformed_contract);
    return false;
  }
  const auto &intent = compact.stages[compact.stage_count - 1].intent;
  if (!TightRgb24(request.host_input, hal::Mutability::read_only) ||
      !TightRgb24(request.host_output, hal::Mutability::write_only) ||
      request.cpu_tail_count != 1 ||
      request.cpu_tails[0].id != hal::CpuTailId::writer_copy ||
      request.cpu_tails[0].after_stage_id != intent.stage_id ||
      request.cpu_tails[0].required != true ||
      request.cpu_tails[0].placement !=
          hal::DeclaredCpuTail::Placement::after_stage ||
      request.cpu_tails[0].before_stage_id != 0 ||
      request.cpu_tails[0].boundary_index !=
          domain_plan.islands[0]
                  .resolved_stage_indices[domain_plan.islands[0].stage_count -
                                          1] +
              1 ||
      !domain_plan.islands[0].requires_upload ||
      !domain_plan.islands[0].requires_final_readback ||
      domain_plan.expected_transfers.uploads != 1 ||
      domain_plan.expected_transfers.final_readbacks != 1 ||
      domain_plan.expected_transfers.intermediate_uploads != 0 ||
      domain_plan.expected_transfers.intermediate_readbacks != 0 ||
      domain_plan.expected_transfers.cpu_tail_boundaries != 1) {
    SetReason(reason, hal::ReasonId::malformed_contract);
    return false;
  }
  return true;
}

hal::PreparedProgram
BuildProgram(const video::VideoExecutionPlanRequest &request,
             const hal::ResolutionResult &resolution,
             const hal::DomainPlan &domain_plan,
             const OpenCudaVideoExecutionCachedEvidence &cached) noexcept {
  hal::PreparedProgram program;
  program.maximum_metadata_readbacks_per_frame =
      cached.maximum_metadata_readbacks_per_frame;
  program.maximum_metadata_cpu_tails_per_frame =
      cached.maximum_metadata_cpu_tails_per_frame;
  const auto &domain = domain_plan.islands[0].execution_domain;
  program.program_id = request.plan_id;
  program.owner_adapter_instance_id = domain.adapter_instance_id;
  program.execution_domain = domain;
  program.host_input = request.host_input;
  program.host_output = request.host_output;
  program.invalidation_key = request.invalidation_key;
  program.temporal = request.temporal;
  program.budget = request.budget;
  program.setup_failure_policy = request.setup_failure_policy;
  program.cpu_tails = request.cpu_tails;
  program.cpu_tail_count = request.cpu_tail_count;
  program.requested_backend =
      request.intent.backend.mode == hal::BackendRequestMode::explicit_backend
          ? request.intent.backend.backend
          : resolution.selected_backend;
  program.strict_backend_selection = request.intent.backend.strict;
  program.requested_provider = request.intent.requested_provider !=
                                       hal::SemanticEffectProviderId::invalid
                                   ? request.intent.requested_provider
                                   : resolution.selected_provider;
  program.strict_provider_selection = request.intent.strict_provider_selection;
  program.requested_execution_domain = domain;
  program.strict_execution_domain = true;
  program.requires_upload = true;
  program.requires_final_readback = true;

  const auto resident =
      ResidentImage(request.host_input, hal::Mutability::read_write);
  std::size_t prepared_index = 0;
  for (std::size_t index = 0; index < request.intent.stage_count; ++index) {
    if (resolution.stages[index].skipped_optional)
      continue;
    const auto &intent = request.intent.stages[index].intent;
    auto &stage = program.stages[prepared_index];
    stage.stage_id = intent.stage_id;
    stage.semantic = intent.semantic;
    stage.provider = resolution.stages[index].provider;
    stage.backend = resolution.stages[index].selected_backend;
    stage.execution_domain = domain;
    stage.input.kind = hal::StageIoKind::image;
    stage.input.image = resident;
    stage.input.image.mutability = hal::Mutability::read_only;
    stage.output.kind = hal::StageIoKind::image;
    stage.output.image =
        prepared_index + 1 == SelectedStageCount(resolution)
            ? ResidentImage(request.host_output, hal::Mutability::read_write)
            : resident;
    stage.model = request.intent.stages[index].model;
    stage.required = intent.required;
    stage.failure_action = intent.failure_action;
    stage.circuit_breaker.consecutive_failure_limit = 2;
    stage.circuit_breaker.cooldown_frames = 30;
    stage.circuit_breaker.action =
        intent.required ? hal::CircuitBreakerAction::quarantine_program
                        : hal::CircuitBreakerAction::disable_stage;
    ++prepared_index;
  }
  program.stage_count = prepared_index;
  if (program.cpu_tail_count == 1 && prepared_index != 0) {
    program.cpu_tails[0].after_stage_id =
        program.stages[prepared_index - 1].stage_id;
    program.cpu_tails[0].boundary_index =
        static_cast<std::uint32_t>(prepared_index);
  }
  return program;
}

bool ExactRegistration(const hal::AdapterRegistration &registration,
                       const video::FrozenVideoExecutionPlan &plan) noexcept {
  if (registration.adapter_instance_id !=
          plan.prepared_program.owner_adapter_instance_id ||
      registration.backend != hal::BackendId::cuda ||
      registration.execution_domain != plan.prepared_program.execution_domain ||
      registration.resident_adapter == nullptr ||
      registration.resident_adapter->adapter_instance_id() !=
          registration.adapter_instance_id ||
      registration.resident_adapter->backend_id() != hal::BackendId::cuda) {
    return false;
  }
  for (std::size_t stage = 0; stage < plan.prepared_program.stage_count;
       ++stage) {
    bool found = false;
    for (std::size_t index = 0; index < registration.semantic_capability_count;
         ++index) {
      const auto &capability = registration.semantic_capabilities[index];
      found |=
          capability.semantic == plan.prepared_program.stages[stage].semantic &&
          capability.provider ==
              ProviderFor(plan.prepared_program.stages[stage].semantic) &&
          hal::IsSelectableSemanticCapability(capability);
    }
    if (!found)
      return false;
  }
  return true;
}

bool ExactReadiness(const hal::ReadinessSnapshot &readiness,
                    const video::FrozenVideoExecutionPlan &plan) noexcept {
  return (readiness.status == hal::StatusId::ready ||
          readiness.status == hal::StatusId::degraded) &&
         readiness.available &&
         readiness.requested_backend == hal::BackendId::cuda &&
         readiness.selected_backend == hal::BackendId::cuda &&
         readiness.provider == plan.prepared_program.requested_provider &&
         readiness.execution_domain == plan.prepared_program.execution_domain &&
         readiness.ready_stage_mask ==
             ((std::uint32_t{1} << plan.prepared_program.stage_count) - 1u) &&
         readiness.blocked_stage_mask == 0 &&
         readiness.disabled_stage_mask == 0;
}

class OpenCudaVideoExecutionExecutor final
    : public video::IVideoExecutionExecutor {
public:
  OpenCudaVideoExecutionExecutor(
      OpenCudaVideoExecutionCandidate candidate,
      std::unique_ptr<hal::PreparedProgramSession> session,
      std::uint64_t plan_id) noexcept
      : candidate_(std::move(candidate)), session_(std::move(session)),
        plan_id_(plan_id) {}

  ~OpenCudaVideoExecutionExecutor() override { Shutdown(); }

  video::VideoExecutionFrameResult
  Execute(const hal::HostFrameBinding &binding) noexcept override {
    video::VideoExecutionFrameResult frame;
    if (!session_ || !session_->prepared()) {
      frame.status = session_ && session_->quarantined()
                         ? hal::StatusId::quarantined
                         : hal::StatusId::stale;
      frame.reason = session_ && session_->quarantined()
                         ? hal::ReasonId::device_lost
                         : hal::ReasonId::stale_program;
      return frame;
    }
    hal::HostFrameTokens tokens;
    frame.status = session_->BindHostFrame(binding, &tokens, &frame.reason);
    if (frame.status != hal::StatusId::ready)
      return frame;

    hal::ExecuteRequest request;
    request.invalidation_key = session_->program().invalidation_key;
    request.execution_domain = session_->program().execution_domain;
    request.capture_sequence = binding.capture_sequence;
    request.input = tokens.input;
    const auto execution = session_->Execute(request);
    if (candidate_.owner != nullptr &&
        candidate_.observe_runtime_owner != nullptr) {
      candidate_.observe_runtime_owner(candidate_.owner, execution.status,
                                       execution.reason);
    }
    frame.status = execution.status;
    frame.reason = execution.reason;
    frame.completion = execution.completion;
    frame.execution_domain = session_->program().execution_domain;
    frame.stages = execution.stages;
    frame.stage_count = execution.stage_count;
    frame.transfers = execution.transfers;
    frame.frame_elapsed_ns = execution.frame_elapsed_ns;
    if (!execution.output_valid || !execution.completion.complete)
      return frame;

    hal::ReasonId resolution_reason = hal::ReasonId::none;
    frame.status =
        session_->ResolveHostFrameOutput(execution.completion, execution.output,
                                         &frame.output, &resolution_reason);
    frame.reason = resolution_reason;
    if (frame.status != hal::StatusId::ready || !frame.output.output_ready ||
        frame.output.capture_sequence != binding.capture_sequence) {
      hal::ReasonId ignored = hal::ReasonId::none;
      (void)session_->CommitOutput({execution.completion, false}, &ignored);
      frame.output = {};
      return frame;
    }
    frame.status = execution.status;
    frame.reason = execution.reason;
    frame.output_ready = true;
    return frame;
  }

  hal::StatusId CommitOutput(const hal::CompletionToken &completion,
                             bool output_write_succeeded,
                             hal::ReasonId *reason) noexcept override {
    if (!session_) {
      SetReason(reason, hal::ReasonId::stale_program);
      return hal::StatusId::stale;
    }
    return session_->CommitOutput({completion, output_write_succeeded}, reason);
  }

  void Invalidate(hal::ReasonId reason) noexcept override {
    if (session_)
      session_->Invalidate(reason);
  }

  hal::StatusId ResetTemporal(hal::ReasonId reason) noexcept override {
    if (!session_)
      return hal::StatusId::stale;
    const auto status = session_->ResetTemporalState(reason);
    if (candidate_.owner && candidate_.observe_runtime_owner)
      candidate_.observe_runtime_owner(candidate_.owner, status,
                                       status == hal::StatusId::quarantined
                                           ? hal::ReasonId::device_lost
                                           : reason);
    return status;
  }

  void Shutdown() noexcept override {
    if (session_ && !session_->stopped())
      session_->Shutdown();
    if (candidate_.owner != nullptr && candidate_.shutdown_owner != nullptr)
      candidate_.shutdown_owner(candidate_.owner);
  }

  video::VideoExecutionAdapterSnapshot Snapshot() const noexcept override {
    video::VideoExecutionAdapterSnapshot snapshot;
    snapshot.plan_id = plan_id_;
    if (!session_) {
      snapshot.status = hal::StatusId::stale;
      snapshot.reason = hal::ReasonId::stale_program;
      return snapshot;
    }
    snapshot.readiness = session_->readiness();
    snapshot.telemetry = session_->telemetry();
    snapshot.latest_committed_frame = session_->latest_committed_frame();
    snapshot.prepared = session_->prepared();
    snapshot.stopped = session_->stopped();
    snapshot.execution_domain = snapshot.prepared
                                    ? session_->program().execution_domain
                                    : candidate_.registration.execution_domain;
    snapshot.status =
        snapshot.stopped ? hal::StatusId::stopped
                         : (session_->quarantined()
                                ? hal::StatusId::quarantined
                                : (snapshot.prepared ? snapshot.readiness.status
                                                     : hal::StatusId::stale));
    snapshot.reason = snapshot.stopped ? hal::ReasonId::shutdown_requested
                                       : (session_->quarantined()
                                              ? hal::ReasonId::device_lost
                                              : snapshot.readiness.reason);
    return snapshot;
  }

private:
  // Session must be destroyed before its candidate-owned adapter.
  OpenCudaVideoExecutionCandidate candidate_{};
  std::unique_ptr<hal::PreparedProgramSession> session_;
  std::uint64_t plan_id_ = 0;
};

} // namespace

OpenCudaVideoExecutionCandidate::~OpenCudaVideoExecutionCandidate() { Reset(); }

OpenCudaVideoExecutionCandidate::OpenCudaVideoExecutionCandidate(
    OpenCudaVideoExecutionCandidate &&other) noexcept
    : owner(std::exchange(other.owner, nullptr)),
      destroy_owner(std::exchange(other.destroy_owner, nullptr)),
      commit_prepared_owner(
          std::exchange(other.commit_prepared_owner, nullptr)),
      observe_runtime_owner(
          std::exchange(other.observe_runtime_owner, nullptr)),
      shutdown_owner(std::exchange(other.shutdown_owner, nullptr)),
      registration(other.registration) {
  other.registration = {};
}

OpenCudaVideoExecutionCandidate &OpenCudaVideoExecutionCandidate::operator=(
    OpenCudaVideoExecutionCandidate &&other) noexcept {
  if (this == &other)
    return *this;
  Reset();
  owner = std::exchange(other.owner, nullptr);
  destroy_owner = std::exchange(other.destroy_owner, nullptr);
  commit_prepared_owner = std::exchange(other.commit_prepared_owner, nullptr);
  observe_runtime_owner = std::exchange(other.observe_runtime_owner, nullptr);
  shutdown_owner = std::exchange(other.shutdown_owner, nullptr);
  registration = other.registration;
  other.registration = {};
  return *this;
}

void OpenCudaVideoExecutionCandidate::Reset() noexcept {
  if (owner != nullptr && shutdown_owner != nullptr)
    shutdown_owner(owner);
  if (owner != nullptr && destroy_owner != nullptr)
    destroy_owner(owner);
  owner = nullptr;
  destroy_owner = nullptr;
  commit_prepared_owner = nullptr;
  observe_runtime_owner = nullptr;
  shutdown_owner = nullptr;
  registration = {};
}

struct OpenCudaVideoExecutionFactory::Impl {
  explicit Impl(OpenCudaVideoExecutionFactoryConfig value) noexcept
      : config(std::move(value)) {}

  struct ProductionLease {
    OpenCudaProductionHalOwner *owner = nullptr;
  };

  class BorrowedResidentAdapter final : public hal::IResidentAdapter {
  public:
    explicit BorrowedResidentAdapter(OpenCudaProductionHalOwner *owner) noexcept
        : owner_(owner),
          adapter_instance_id_(owner && owner->adapter()
                                   ? owner->adapter()->adapter_instance_id()
                                   : 0) {}

    std::uint64_t adapter_instance_id() const noexcept override {
      return adapter_instance_id_;
    }
    hal::BackendId backend_id() const noexcept override {
      return adapter_instance_id_ != 0 ? hal::BackendId::cuda
                                       : hal::BackendId::invalid;
    }
    hal::StatusId Prepare(const hal::PreparedProgram &program,
                          hal::PreparedProgramHandle *candidate,
                          hal::ReadinessSnapshot *readiness,
                          hal::ReasonId *reason) noexcept override {
      auto *adapter = NativeAdapter();
      if (!adapter)
        return Stopped(readiness, reason);
      return adapter->Prepare(program, candidate, readiness, reason);
    }
    hal::ExecuteResult
    Execute(const hal::PreparedProgramHandle &program,
            const hal::ExecuteRequest &request) noexcept override {
      auto *adapter = NativeAdapter();
      if (adapter)
        return adapter->Execute(program, request);
      hal::ExecuteResult result;
      result.status = hal::StatusId::stopped;
      result.reason = hal::ReasonId::shutdown_requested;
      return result;
    }
    hal::StatusId BindHostFrame(const hal::PreparedProgramHandle &program,
                                const hal::HostFrameBinding &binding,
                                hal::HostFrameTokens *tokens,
                                hal::ReasonId *reason) noexcept override {
      auto *adapter = NativeAdapter();
      if (!adapter)
        return Stopped(nullptr, reason);
      return adapter->BindHostFrame(program, binding, tokens, reason);
    }
    hal::StatusId DiscardUnsubmittedHostFrame(
        const hal::PreparedProgramHandle &program,
        const hal::HostFrameTokens &tokens, std::uint64_t capture_sequence,
        hal::ReasonId *reason) noexcept override {
      auto *adapter = NativeAdapter();
      if (!adapter)
        return Stopped(nullptr, reason);
      return adapter->DiscardUnsubmittedHostFrame(program, tokens,
                                                   capture_sequence, reason);
    }
    hal::StatusId
    ResolveHostFrameOutput(const hal::PreparedProgramHandle &program,
                           const hal::CompletionToken &completion,
                           const hal::ResourceHandle &output,
                           hal::HostFrameOutputResolution *resolution,
                           hal::ReasonId *reason) noexcept override {
      auto *adapter = NativeAdapter();
      if (!adapter)
        return Stopped(nullptr, reason);
      return adapter->ResolveHostFrameOutput(program, completion, output,
                                             resolution, reason);
    }
    hal::StatusId ResetTemporalState(const hal::PreparedProgramHandle &program,
                                     hal::ReasonId reason) noexcept override {
      auto *adapter = NativeAdapter();
      return adapter ? adapter->ResetTemporalState(program, reason)
                     : hal::StatusId::stopped;
    }
    void
    DestroyPrepared(hal::PreparedProgramHandle *program) noexcept override {
      if (auto *adapter = NativeAdapter())
        adapter->DestroyPrepared(program);
    }
    void Shutdown() noexcept override {
      // The registry owns this view for the factory lifetime. Native adapter
      // shutdown is deferred until factory retirement.
    }

    void Retire() noexcept { retired_ = true; }

  private:
    hal::IResidentAdapter *NativeAdapter() const noexcept {
      return !retired_ && owner_ ? owner_->adapter() : nullptr;
    }

    static hal::StatusId Stopped(hal::ReadinessSnapshot *readiness,
                                 hal::ReasonId *reason) noexcept {
      if (readiness) {
        readiness->status = hal::StatusId::stopped;
        readiness->reason = hal::ReasonId::shutdown_requested;
      }
      SetReason(reason, hal::ReasonId::shutdown_requested);
      return hal::StatusId::stopped;
    }

    OpenCudaProductionHalOwner *owner_ = nullptr;
    std::uint64_t adapter_instance_id_ = 0;
    bool retired_ = false;
  };

  static void DestroyProductionLease(void *opaque) noexcept {
    delete static_cast<ProductionLease *>(opaque);
  }

  static hal::StatusId
  CommitProductionOwner(void *opaque, const hal::PreparedProgram &program,
                        const hal::ReadinessSnapshot &readiness,
                        hal::ReasonId *reason) noexcept {
    auto *lease = static_cast<ProductionLease *>(opaque);
    auto *owner = lease ? lease->owner : nullptr;
    if (!owner) {
      SetReason(reason, hal::ReasonId::stale_program);
      return hal::StatusId::stale;
    }
    const auto committed = owner->CommitPreparedCapability(program, readiness);
    SetReason(reason, committed.reason);
    return committed.prepared_capability_committed
               ? committed.hal_status
               : (committed.hal_status == hal::StatusId::invalid
                      ? hal::StatusId::blocked
                      : committed.hal_status);
  }

  static void ObserveProductionOwner(void *opaque, hal::StatusId status,
                                     hal::ReasonId reason) noexcept {
    auto *lease = static_cast<ProductionLease *>(opaque);
    if (lease && lease->owner)
      lease->owner->ObserveRuntimeStatus(status, reason);
  }

  static void ShutdownProductionLease(void *) noexcept {
    // PreparedProgramSession::Shutdown destroys only this executor's prepared
    // slot. The factory retains the native adapter because the registry also
    // retains its pointer. Factory retirement performs native shutdown after
    // the coordinator has retired all executors.
  }

  static hal::StatusId PrepareProductionCandidate(
      void *opaque, const video::FrozenVideoExecutionPlan &plan,
      OpenCudaVideoExecutionCandidate *candidate,
      video::VideoAdapterPrepareEvidence *evidence) noexcept {
    auto *self = static_cast<Impl *>(opaque);
    if (!self || !candidate || !evidence || !self->production_owner) {
      if (evidence) {
        evidence->status = hal::StatusId::unavailable;
        evidence->reason = hal::ReasonId::stale_program;
        evidence->candidate_resources_rolled_back = true;
      }
      return hal::StatusId::unavailable;
    }
    if (plan.prepared_program.execution_domain !=
        self->production_registration.execution_domain) {
      evidence->status = hal::StatusId::blocked;
      evidence->reason = hal::ReasonId::wrong_execution_domain;
      evidence->candidate_resources_rolled_back = true;
      return evidence->status;
    }
    if (plan.prepared_program.owner_adapter_instance_id !=
        self->production_registration.adapter_instance_id) {
      evidence->status = hal::StatusId::blocked;
      evidence->reason = hal::ReasonId::foreign_adapter;
      evidence->candidate_resources_rolled_back = true;
      return evidence->status;
    }
    if (plan.prepared_program.invalidation_key.generations !=
        self->config.cached.generations) {
      evidence->status = hal::StatusId::stale;
      evidence->reason = hal::ReasonId::wrong_generation;
      evidence->candidate_resources_rolled_back = true;
      return evidence->status;
    }
    if (!SameBudget(plan.prepared_program.budget,
                    self->config.cached.reserved_budget)) {
      evidence->status = hal::StatusId::resource_exhausted;
      evidence->reason = hal::ReasonId::resource_exhausted;
      evidence->candidate_resources_rolled_back = true;
      return evidence->status;
    }
    if (plan.prepared_program.stage_count != self->production_stage_count) {
      evidence->status = hal::StatusId::blocked;
      evidence->reason = hal::ReasonId::semantic_effect_unsupported;
      evidence->candidate_resources_rolled_back = true;
      return evidence->status;
    }
    for (std::size_t index = 0; index < self->production_stage_count; ++index) {
      if (plan.prepared_program.stages[index].semantic !=
              self->production_semantics[index] ||
          !SameModel(plan.prepared_program.stages[index].model,
                     self->config.cached.models[index])) {
        evidence->status = hal::StatusId::blocked;
        evidence->reason = hal::ReasonId::capability_evidence_incomplete;
        evidence->candidate_resources_rolled_back = true;
        return evidence->status;
      }
    }

    auto *lease =
        new (std::nothrow) ProductionLease{self->production_owner.get()};
    if (!lease) {
      evidence->status = hal::StatusId::resource_exhausted;
      evidence->reason = hal::ReasonId::resource_exhausted;
      evidence->candidate_resources_rolled_back = true;
      return evidence->status;
    }
    candidate->owner = lease;
    candidate->destroy_owner = DestroyProductionLease;
    candidate->commit_prepared_owner = CommitProductionOwner;
    candidate->observe_runtime_owner = ObserveProductionOwner;
    candidate->shutdown_owner = ShutdownProductionLease;
    candidate->registration = self->production_registration;
    evidence->status = hal::StatusId::ready;
    evidence->reason = hal::ReasonId::none;
    evidence->candidate_resources_rolled_back = false;
    return evidence->status;
  }

  OpenCudaVideoExecutionFactoryConfig config{};
  std::unique_ptr<OpenCudaProductionHalOwner> production_owner;
  std::unique_ptr<BorrowedResidentAdapter> production_adapter_view;
  hal::AdapterRegistration production_registration{};
  std::array<hal::SemanticEffectId, hal::kMaxPreparedStages>
      production_semantics{};
  std::size_t production_stage_count = 0;
  video::FrozenVideoExecutionPlan preprepared_plan{};
  std::unique_ptr<video::IVideoExecutionExecutor> preprepared_executor;
  bool production_shutdown_requested = false;
  bool production_owner_retired = false;
};

class OpenCudaProductionSetupLeaseOwner final
    : public video::IProductionVideoSetupLeaseOwner {
public:
  explicit OpenCudaProductionSetupLeaseOwner(
      std::unique_ptr<OpenCudaVideoExecutionFactory> factory) noexcept
      : factory_(std::move(factory)) {
    if (factory_) {
      resident_adapter_ = factory_->adapter_registration().resident_adapter;
      video_factory_ = factory_.get();
    }
  }

  hal::IResidentAdapter *resident_adapter() const noexcept override {
    return resident_adapter_;
  }

  video::IVideoAdapterFactory *video_adapter_factory() const noexcept override {
    return video_factory_;
  }

  void Shutdown() noexcept override { (void)RequestShutdown(); }

  video::ProductionVideoSetupLeaseShutdownResult
  RequestShutdown() noexcept override {
    video::ProductionVideoSetupLeaseShutdownResult result;
    result.shutdown_requested = true;
    if (!factory_) {
      result.status = hal::StatusId::stopped;
      result.reason = hal::ReasonId::shutdown_requested;
      result.quiescent = true;
      result.driver_context_release_permitted = true;
      return result;
    }

    const auto native = factory_->RequestProductionOwnerShutdown();
    result.shutdown_attempts = native.shutdown_attempts;
    result.pending = !native.driver_context_release_permitted;
    result.quiescent =
        native.driver_context_release_permitted && native.quiescent;
    result.driver_context_release_permitted =
        native.driver_context_release_permitted;
    if (result.driver_context_release_permitted) {
      result.status = hal::StatusId::stopped;
      result.reason = hal::ReasonId::shutdown_requested;
    } else {
      result.status = native.hal_status;
      result.reason = native.reason;
    }
    return result;
  }

private:
  std::unique_ptr<OpenCudaVideoExecutionFactory> factory_;
  hal::IResidentAdapter *resident_adapter_ = nullptr;
  video::IVideoAdapterFactory *video_factory_ = nullptr;
};

OpenCudaVideoExecutionFactory::OpenCudaVideoExecutionFactory(
    OpenCudaVideoExecutionFactoryConfig config) noexcept
    : impl_(new (std::nothrow) Impl(std::move(config))) {}

OpenCudaVideoExecutionFactory::~OpenCudaVideoExecutionFactory() = default;

std::unique_ptr<OpenCudaVideoExecutionFactory>
OpenCudaVideoExecutionFactory::CreateProduction(
    const OpenCudaProductionVideoExecutionFactoryRequest &request,
    OpenCudaProductionVideoExecutionBootstrapResult *result) noexcept {
  OpenCudaProductionVideoExecutionBootstrapResult local;
  if (!result)
    result = &local;
  *result = {};
  video::NormalizedVideoExecutionIntent supported_intent;
  supported_intent.stage_count = request.stage_count;
  for (std::size_t i = 0;
       i < request.stage_count && i < supported_intent.stages.size(); ++i)
    supported_intent.stages[i].intent.semantic = request.semantics[i];
  const bool supported_semantic = SupportedStages(supported_intent);
  const bool has_canonical_plan = request.plan.intent.stage_count != 0;
  const std::size_t canonical_stage_count =
      has_canonical_plan ? request.plan.intent.stage_count
                         : request.stage_count;
  const auto checked_count =
      std::min(request.stage_count, request.semantics.size());
  const bool has_replace =
      std::find(request.semantics.begin(),
                request.semantics.begin() + checked_count,
                hal::SemanticEffectId::virtual_background_replace) !=
      request.semantics.begin() + checked_count;
  const bool has_key_light =
      std::find(request.semantics.begin(),
                request.semantics.begin() + checked_count,
                hal::SemanticEffectId::virtual_key_light) !=
      request.semantics.begin() + checked_count;
  const bool replacement_contract =
      !has_replace ||
      (request.backend.replacement_width > 0 &&
       request.backend.replacement_height > 0 &&
       static_cast<std::uint64_t>(request.backend.replacement_width) *
               static_cast<std::uint64_t>(request.backend.replacement_height) *
               3u ==
           request.backend.replacement_rgb.size());
  const bool key_light_contract =
      !has_key_light ||
      (request.backend.key_light_intensity >= 0 &&
       request.backend.key_light_intensity <= 100 &&
       request.backend.key_light_temperature_preset >= 0 &&
       request.backend.key_light_temperature_preset <= 2 &&
       request.backend.key_light_direction_pan_degrees >= -90 &&
       request.backend.key_light_direction_pan_degrees <= 90);
  bool has_model_consumer = false;
  bool has_matting = false;
  bool exact_model = request.stage_count <= request.semantics.size();
  for (std::size_t i = 0; i < checked_count; ++i) {
    const auto semantic = request.semantics[i];
    const auto percent = [](int value) { return value >= 0 && value <= 100; };
    if (semantic == hal::SemanticEffectId::video_noise_removal)
      exact_model &= percent(request.backend.denoise_strength);
    if (semantic == hal::SemanticEffectId::auto_frame)
      exact_model &= percent(request.backend.auto_frame_strength) &&
                     percent(request.backend.auto_frame_smoothing) &&
                     request.backend.auto_frame_headroom >= 0.F &&
                     request.backend.auto_frame_headroom <= 1.F;
    if (semantic == hal::SemanticEffectId::eye_contact)
      exact_model &= percent(request.backend.eye_contact_strength);
    if (semantic == hal::SemanticEffectId::vignette)
      exact_model &= percent(request.backend.vignette_intensity);
    has_matting |= UsesMatting(semantic);
    if (!UsesModel(semantic)) {
      exact_model &= ZeroModel(request.models[i]);
      continue;
    }
    has_model_consumer = true;
    const auto &artifact =
        UsesMatting(semantic) ? request.backend.model_artifact
        : semantic == hal::SemanticEffectId::auto_frame
            ? (request.backend.dependencies.face_artifact.Valid()
                   ? request.backend.dependencies.face_artifact
                   : request.backend.model_artifact)
        : semantic == hal::SemanticEffectId::eye_contact
            ? request.backend.dependencies.eye_left_artifact
            : request.backend.dependencies.denoise_artifact;
    exact_model &=
        artifact.Valid() &&
        request.models[i].model_id_hash == artifact.model_id_sha256 &&
        request.models[i].artifact_set_hash == artifact.artifact_set_sha256 &&
        request.models[i].artifact_generation == artifact.registry_generation;
    if (semantic == hal::SemanticEffectId::eye_contact) {
      const auto &deps = request.backend.dependencies;
      exact_model &=
          deps.face_artifact.Valid() && deps.landmarks_artifact.Valid() &&
          deps.eye_right_artifact.Valid() &&
          deps.eye_right_artifact.manifest_sha256 == artifact.manifest_sha256 &&
          deps.eye_right_artifact.model_id_sha256 == artifact.model_id_sha256 &&
          deps.eye_right_artifact.registry_generation ==
              artifact.registry_generation;
    }
  }
  const bool has_auto_frame =
      std::find(request.semantics.begin(),
                request.semantics.begin() + checked_count,
                hal::SemanticEffectId::auto_frame) !=
      request.semantics.begin() + checked_count;
  const bool has_eye = std::find(request.semantics.begin(),
                                 request.semantics.begin() + checked_count,
                                 hal::SemanticEffectId::eye_contact) !=
                       request.semantics.begin() + checked_count;
  const bool face_analysis = (has_auto_frame || has_eye) &&
                             request.backend.dependencies.face_artifact.Valid();
  const bool matte_bounds =
      has_auto_frame && request.backend.model_artifact.Valid();
  exact_model &=
      has_matting ? request.backend.model_artifact.Valid()
                  : (!request.backend.model_artifact.Valid() || has_auto_frame);
  const bool matte_auto_frame_fallback =
      has_auto_frame && !face_analysis && matte_bounds;
  bool optional_auto_frame_unavailable = false;
  for (std::size_t index = 0;
       index < canonical_stage_count &&
       index < request.plan.intent.stages.size() && has_canonical_plan;
       ++index) {
    const auto &stage = request.plan.intent.stages[index].intent;
    optional_auto_frame_unavailable |=
        stage.semantic == hal::SemanticEffectId::auto_frame &&
        !stage.required && !IsSetupSuccess(request.canonical_status[index]);
  }
  if (request.backend.vignette_tracked_center)
    exact_model &= (has_auto_frame && (face_analysis || matte_bounds)) ||
                   optional_auto_frame_unavailable;
  if (request.factory_id == 0 || canonical_stage_count == 0 ||
      canonical_stage_count > hal::kMaxPreparedStages || !supported_semantic ||
      !replacement_contract || !key_light_contract || !exact_model ||
      !ValidReservedBudget(request.reserved_budget) ||
      request.generations.configuration == 0 ||
      request.generations.geometry == 0 ||
      (has_model_consumer && request.generations.model == 0) ||
      request.generations.provider == 0 || request.generations.device == 0 ||
      request.backend.configuration_generation !=
          request.generations.configuration ||
      (has_model_consumer && request.backend.model_preprocess_generation !=
                                 request.generations.model) ||
      (has_model_consumer && !request.model_artifact_validated) ||
      !request.provider_runtime_validated ||
      (has_model_consumer && !request.cuda_iobinding_validated) ||
      !request.user_stream_bound || !request.reusable_resources_supported ||
      (has_model_consumer && !request.device_preprocess_supported) ||
      !request.device_postprocess_supported ||
      !request.stop_aware_bounded_wait || !request.allocation_free_execute) {
    result->status = hal::StatusId::blocked;
    result->reason = hal::ReasonId::capability_evidence_incomplete;
    return nullptr;
  }

  auto owner = OpenCudaProductionHalOwner::CreateBackend(request.backend,
                                                         &result->owner);
  if (!owner || !owner->bootstrap_domain_ready() || !owner->adapter()) {
    result->status = result->owner.hal_status;
    result->reason = result->owner.reason;
    return nullptr;
  }

  try {
    OpenCudaVideoExecutionFactoryConfig config;
    config.factory_id = request.factory_id;
    config.selection_priority = request.selection_priority;
    config.cached.execution_domain = owner->execution_domain();
    config.cached.readiness.status = hal::StatusId::ready;
    config.cached.readiness.requested_backend = hal::BackendId::cuda;
    config.cached.readiness.selected_backend = hal::BackendId::cuda;
    config.cached.readiness.provider = AggregateProvider(supported_intent);
    config.cached.readiness.execution_domain = owner->execution_domain();
    config.cached.readiness.ready_stage_mask =
        (std::uint32_t{1} << request.stage_count) - 1u;
    config.cached.readiness.available = true;
    if (has_canonical_plan && canonical_stage_count != request.stage_count) {
      config.cached.readiness.status = hal::StatusId::degraded;
      config.cached.readiness.reason =
          hal::ReasonId::optional_stage_failed_open;
      config.cached.readiness.degraded = true;
    }
    config.cached.models = request.models;
    config.cached.stage_count = request.stage_count;
    config.cached.generations = request.generations;
    config.cached.reserved_budget = request.reserved_budget;
    config.cached.generations.context =
        owner->execution_domain().logical_context_generation;
    config.cached.model_artifact_validated = request.model_artifact_validated;
    config.cached.provider_runtime_validated = true;
    config.cached.device_context_validated = true;
    config.cached.cuda_iobinding_validated = request.cuda_iobinding_validated;
    config.cached.user_stream_bound = true;
    config.cached.reusable_resources_supported = true;
    config.cached.device_preprocess_supported =
        request.device_preprocess_supported;
    config.cached.device_postprocess_supported = true;
    config.cached.one_upload_one_readback = true;
    config.cached.no_intermediate_crossings = std::none_of(
        request.semantics.begin(), request.semantics.begin() + checked_count,
        [](hal::SemanticEffectId semantic) {
          return semantic == hal::SemanticEffectId::auto_frame ||
                 semantic == hal::SemanticEffectId::eye_contact;
        });
    config.cached.stop_aware_bounded_wait = true;
    config.cached.allocation_free_execute = true;
    config.cached.minimum_synchronizations_per_frame = 1;
    config.cached.maximum_synchronizations_per_frame = 1;
    config.cached.maximum_metadata_readbacks_per_frame =
        (face_analysis ? 12u : 0u) + (matte_bounds ? 1u : 0u);
    config.cached.maximum_metadata_cpu_tails_per_frame =
        (face_analysis ? 1u : 0u) + (has_auto_frame ? 1u : 0u) +
        (matte_bounds ? 1u : 0u);
    config.cached.maximum_metadata_synchronizations_per_frame =
        (face_analysis ? 1u : 0u) + (matte_bounds ? 1u : 0u);
    if (matte_auto_frame_fallback) {
      config.cached.readiness.status = hal::StatusId::degraded;
      config.cached.readiness.reason =
          hal::ReasonId::capability_evidence_incomplete;
      config.cached.readiness.degraded = true;
    }

    auto factory = std::unique_ptr<OpenCudaVideoExecutionFactory>(
        new OpenCudaVideoExecutionFactory(std::move(config)));
    if (!factory->impl_) {
      result->status = hal::StatusId::resource_exhausted;
      result->reason = hal::ReasonId::resource_exhausted;
      return nullptr;
    }
    auto &impl = *factory->impl_;
    impl.production_owner = std::move(owner);
    impl.production_adapter_view =
        std::make_unique<Impl::BorrowedResidentAdapter>(
            impl.production_owner.get());
    impl.production_semantics = request.semantics;
    impl.production_stage_count = request.stage_count;
    impl.production_registration.adapter_instance_id =
        impl.config.cached.execution_domain.adapter_instance_id;
    impl.production_registration.backend = hal::BackendId::cuda;
    impl.production_registration.execution_domain =
        impl.config.cached.execution_domain;
    impl.production_registration.selection_priority =
        request.selection_priority;
    impl.production_registration.resident_adapter =
        impl.production_adapter_view.get();
    std::size_t final_selected_capability = canonical_stage_count;
    for (std::size_t index = canonical_stage_count; index-- > 0;) {
      const auto status = has_canonical_plan ? request.canonical_status[index]
                                             : hal::StatusId::ready;
      if (status == hal::StatusId::ready || status == hal::StatusId::degraded) {
        final_selected_capability = index;
        break;
      }
    }
    for (std::size_t index = 0; index < canonical_stage_count; ++index) {
      const auto semantic =
          has_canonical_plan ? request.plan.intent.stages[index].intent.semantic
                             : request.semantics[index];
      const auto status = has_canonical_plan ? request.canonical_status[index]
                                             : hal::StatusId::ready;
      const auto reason = has_canonical_plan ? request.canonical_reason[index]
                                             : hal::ReasonId::none;
      const bool selectable =
          status == hal::StatusId::ready || status == hal::StatusId::degraded;
      auto &capability =
          impl.production_registration.semantic_capabilities[index];
      capability.semantic = semantic;
      capability.provider = ProviderFor(semantic);
      capability.status = matte_auto_frame_fallback &&
                                  semantic == hal::SemanticEffectId::auto_frame
                              ? hal::StatusId::degraded
                              : status;
      capability.reason = matte_auto_frame_fallback &&
                                  semantic == hal::SemanticEffectId::auto_frame
                              ? hal::ReasonId::capability_evidence_incomplete
                              : reason;
      capability.cpu_tail = index == final_selected_capability
                                ? hal::CpuTailId::writer_copy
                                : hal::CpuTailId::invalid;
      capability.cpu_tail_placement =
          hal::DeclaredCpuTail::Placement::after_stage;
      capability.supports_tracked_vignette_center =
          selectable && semantic == hal::SemanticEffectId::vignette &&
          request.backend.vignette_tracked_center &&
          (has_auto_frame || optional_auto_frame_unavailable);
      capability.live_stage = selectable;
      capability.production_eligible = selectable;
      capability.exact_evidence = selectable;
      capability.evidence[0] = {
          hal::CapabilityEvidenceCode::provider_runtime_available,
          hal::StatusId::ready,
          hal::ReasonId::none,
          semantic,
          hal::BackendId::cuda,
          true};
      capability.evidence[1] = {
          UsesModel(semantic)
              ? hal::CapabilityEvidenceCode::model_artifact_identity_valid
              : hal::CapabilityEvidenceCode::synchronization_evidence_valid,
          hal::StatusId::ready,
          hal::ReasonId::none,
          semantic,
          hal::BackendId::cuda,
          true};
      capability.evidence[2] = {
          hal::CapabilityEvidenceCode::synchronization_evidence_valid,
          hal::StatusId::ready,
          hal::ReasonId::none,
          semantic,
          hal::BackendId::cuda,
          true};
      capability.evidence_count =
          selectable ? (UsesModel(semantic) ? 3 : 2) : 0;
    }
    impl.production_registration.semantic_capability_count =
        canonical_stage_count;
    impl.config.prepare_candidate = Impl::PrepareProductionCandidate;
    impl.config.setup_context = &impl;
    result->status = hal::StatusId::ready;
    result->reason = hal::ReasonId::none;
    result->exact_candidate_ready = !has_canonical_plan;
    auto &snapshot = result->setup_lease;
    snapshot.contract_version =
        video::kProductionVideoSetupLeaseContractVersion;
    snapshot.state = video::ProductionVideoSetupLeaseState::live;
    snapshot.status = impl.config.cached.readiness.status;
    snapshot.reason = impl.config.cached.readiness.reason;
    snapshot.readiness = impl.config.cached.readiness;
    snapshot.backend = hal::BackendId::cuda;
    snapshot.execution_domain = impl.config.cached.execution_domain;
    snapshot.generations = impl.config.cached.generations;
    snapshot.reserved_budget = impl.config.cached.reserved_budget;
    snapshot.providers[0] = ProviderFor(request.semantics[0]);
    snapshot.provider_count = 1;
    for (std::size_t index = 1; index < request.stage_count; ++index) {
      const auto provider = ProviderFor(request.semantics[index]);
      if (provider != snapshot.providers[0] &&
          (snapshot.provider_count < 2 || provider != snapshot.providers[1]))
        snapshot.providers[snapshot.provider_count++] = provider;
    }
    for (std::size_t index = 0; index < request.stage_count; ++index) {
      if (!UsesModel(request.semantics[index]))
        continue;
      auto &leased = snapshot.artifacts[snapshot.artifact_count++];
      leased.semantic = request.semantics[index];
      leased.provider = ProviderFor(request.semantics[index]);
      leased.artifact = impl.config.cached.models[index];
    }
    snapshot.adapter = factory->adapter_registration();
    snapshot.factory = factory->Registration();
    return factory;
  } catch (const std::bad_alloc &) {
    result->status = hal::StatusId::resource_exhausted;
    result->reason = hal::ReasonId::resource_exhausted;
  } catch (...) {
    result->status = hal::StatusId::blocked;
    result->reason = hal::ReasonId::capability_evidence_incomplete;
  }
  return nullptr;
}

hal::StatusId OpenCudaVideoExecutionFactory::AcquireProductionSetupLease(
    const OpenCudaProductionVideoExecutionFactoryRequest &request,
    video::ProductionVideoSetupLease *destination,
    OpenCudaProductionVideoExecutionBootstrapResult *bootstrap,
    video::ProductionVideoSetupLeaseAcquireEvidence *evidence,
    hal::ReasonId *reason) noexcept {
  OpenCudaProductionVideoExecutionBootstrapResult local_bootstrap;
  if (!bootstrap)
    bootstrap = &local_bootstrap;
  if (evidence)
    *evidence = {};
  SetReason(reason, hal::ReasonId::none);

  if (!ValidReservedBudget(request.reserved_budget)) {
    *bootstrap = {};
    bootstrap->status = hal::StatusId::resource_exhausted;
    bootstrap->reason = hal::ReasonId::resource_exhausted;
    SetReason(reason, bootstrap->reason);
    if (evidence) {
      evidence->status = bootstrap->status;
      evidence->reason = bootstrap->reason;
      evidence->readiness = bootstrap->setup_lease.readiness;
      evidence->previous_lease_retained =
          destination != nullptr && destination->live();
    }
    return bootstrap->status;
  }

  auto factory = CreateProduction(request, bootstrap);
  if (!factory) {
    SetReason(reason, bootstrap->reason);
    if (evidence) {
      evidence->status = bootstrap->status;
      evidence->reason = bootstrap->reason;
      evidence->readiness = bootstrap->setup_lease.readiness;
      evidence->previous_lease_retained =
          destination != nullptr && destination->live();
    }
    return bootstrap->status;
  }

  if (request.plan.intent.stage_count != 0) {
    auto exact_request = request.plan;
    exact_request.invalidation_key.generations.context =
        bootstrap->setup_lease.execution_domain.logical_context_generation;
    exact_request.invalidation_key.execution_domain_hash =
        video::ProductionVideoExecutionDomainHash(
            bootstrap->setup_lease.execution_domain);
    exact_request.budget = request.reserved_budget;

    hal::StaticAdapterRegistry registry;
    hal::ReasonId setup_reason = hal::ReasonId::none;
    auto setup_status =
        registry.Register(factory->adapter_registration(), &setup_reason);
    const auto resolution =
        setup_status == hal::StatusId::ready
            ? hal::ResolveSemanticStages(
                  registry,
                  video::BuildVideoResolutionRequest(exact_request.intent))
            : hal::ResolutionResult{};
    const auto domain_plan =
        setup_status == hal::StatusId::ready && resolution.available &&
                !resolution.passthrough
            ? hal::PlanExecutionDomains(registry, resolution,
                                        hal::CrossAdapterPolicy::reject)
            : hal::DomainPlan{};
    if (setup_status == hal::StatusId::ready &&
        (!resolution.available || resolution.passthrough ||
         !IsSetupSuccess(domain_plan.status))) {
      setup_status = resolution.status == hal::StatusId::resource_exhausted
                         ? resolution.status
                         : hal::StatusId::blocked;
      setup_reason = resolution.reason == hal::ReasonId::none
                         ? domain_plan.reason
                         : resolution.reason;
    }

    video::VideoAdapterFactoryPreflight preflight;
    if (setup_status == hal::StatusId::ready) {
      preflight =
          factory->Preflight(exact_request, resolution, domain_plan,
                             video::VideoExecutionRoute::prepared_resident);
      setup_status = preflight.status;
      setup_reason = preflight.reason;
    }

    video::FrozenVideoExecutionPlan frozen;
    if (IsSetupSuccess(setup_status)) {
      frozen.plan_id = exact_request.plan_id;
      frozen.route = video::VideoExecutionRoute::prepared_resident;
      frozen.intent = exact_request.intent;
      frozen.host_input = exact_request.host_input;
      frozen.host_output = exact_request.host_output;
      frozen.invalidation_key = exact_request.invalidation_key;
      frozen.temporal = exact_request.temporal;
      frozen.budget = exact_request.budget;
      frozen.cpu_tails = exact_request.cpu_tails;
      frozen.cpu_tail_count = exact_request.cpu_tail_count;
      frozen.setup_failure_policy = exact_request.setup_failure_policy;
      frozen.migration_policy = exact_request.migration_policy;
      frozen.resolution = resolution;
      frozen.domain_plan = domain_plan;
      frozen.factory_id = request.factory_id;
      frozen.per_frame_transfers = preflight.per_frame_transfers;
      frozen.prepared_program = preflight.prepared_program;
      auto prepared = factory->Prepare(frozen, &bootstrap->prepare);
      setup_status = bootstrap->prepare.status;
      setup_reason = bootstrap->prepare.reason;
      if (prepared && IsSetupSuccess(setup_status)) {
        bootstrap->actual_required_budget =
            factory->impl_->production_owner->LastPrepareResult()
                .handle.reserved_budget;
        factory->impl_->preprepared_plan = frozen;
        factory->impl_->preprepared_executor = std::move(prepared);
        bootstrap->exact_candidate_ready = true;
      } else {
        setup_status = setup_status == hal::StatusId::invalid
                           ? hal::StatusId::blocked
                           : setup_status;
        setup_reason = setup_reason == hal::ReasonId::none
                           ? hal::ReasonId::capability_evidence_incomplete
                           : setup_reason;
      }
    }

    if (!factory->impl_->preprepared_executor) {
      const auto last = factory->impl_->production_owner->LastPrepareResult();
      bootstrap->status = setup_status;
      bootstrap->reason = setup_reason;
      bootstrap->failed_component = last.failed_component;
      const auto shutdown = factory->RequestProductionOwnerShutdown();
      bootstrap->candidate_shutdown.status = shutdown.hal_status;
      bootstrap->candidate_shutdown.reason = shutdown.reason;
      bootstrap->candidate_shutdown.shutdown_attempts =
          shutdown.shutdown_attempts;
      bootstrap->candidate_shutdown.shutdown_requested =
          shutdown.shutdown_requested;
      bootstrap->candidate_shutdown.pending =
          !shutdown.driver_context_release_permitted;
      bootstrap->candidate_shutdown.quiescent = shutdown.quiescent;
      bootstrap->candidate_shutdown.driver_context_release_permitted =
          shutdown.driver_context_release_permitted;
      if (evidence) {
        evidence->status = setup_status;
        evidence->reason = setup_reason;
        evidence->readiness = bootstrap->setup_lease.readiness;
        evidence->candidate_shutdown = bootstrap->candidate_shutdown;
        evidence->candidate_owner_rolled_back =
            bootstrap->candidate_shutdown.driver_context_release_permitted;
        evidence->candidate_cleanup_pending =
            !bootstrap->candidate_shutdown.driver_context_release_permitted;
        evidence->previous_lease_retained =
            destination != nullptr && destination->live();
      }
      SetReason(reason, setup_reason);
      return setup_status;
    }
  }

  std::unique_ptr<video::IProductionVideoSetupLeaseOwner> owner;
  try {
    owner =
        std::make_unique<OpenCudaProductionSetupLeaseOwner>(std::move(factory));
  } catch (const std::bad_alloc &) {
    bootstrap->status = hal::StatusId::resource_exhausted;
    bootstrap->reason = hal::ReasonId::resource_exhausted;
    SetReason(reason, bootstrap->reason);
    if (evidence) {
      evidence->status = bootstrap->status;
      evidence->reason = bootstrap->reason;
      evidence->readiness = bootstrap->setup_lease.readiness;
      evidence->candidate_owner_rolled_back = true;
      evidence->previous_lease_retained =
          destination != nullptr && destination->live();
    }
    return bootstrap->status;
  } catch (...) {
    bootstrap->status = hal::StatusId::blocked;
    bootstrap->reason = hal::ReasonId::capability_evidence_incomplete;
    SetReason(reason, bootstrap->reason);
    if (evidence) {
      evidence->status = bootstrap->status;
      evidence->reason = bootstrap->reason;
      evidence->readiness = bootstrap->setup_lease.readiness;
      evidence->candidate_owner_rolled_back = true;
      evidence->previous_lease_retained =
          destination != nullptr && destination->live();
    }
    return bootstrap->status;
  }

  const auto status = video::ProductionVideoSetupLease::Acquire(
      bootstrap->setup_lease, std::move(owner), destination, evidence, reason);
  if (status != hal::StatusId::ready && status != hal::StatusId::degraded) {
    bootstrap->status = status;
    bootstrap->reason = reason ? *reason : hal::ReasonId::none;
  }
  return status;
}

video::VideoAdapterFactoryRegistration
OpenCudaVideoExecutionFactory::Registration() noexcept {
  video::VideoAdapterFactoryRegistration registration;
  if (!impl_)
    return registration;
  registration.factory_id = impl_->config.factory_id;
  registration.route = video::VideoExecutionRoute::prepared_resident;
  registration.backend = hal::BackendId::cuda;
  registration.provider = impl_->config.cached.readiness.provider;
  registration.adapter_instance_id =
      impl_->config.cached.execution_domain.adapter_instance_id;
  registration.execution_domain = impl_->config.cached.execution_domain;
  registration.selection_priority = impl_->config.selection_priority;
  registration.factory = this;
  return registration;
}

hal::AdapterRegistration
OpenCudaVideoExecutionFactory::adapter_registration() const noexcept {
  return impl_ ? impl_->production_registration : hal::AdapterRegistration{};
}

OpenCudaProductionShutdownResult
OpenCudaVideoExecutionFactory::RequestProductionOwnerShutdown() noexcept {
  OpenCudaProductionShutdownResult result;
  if (!impl_ || !impl_->production_owner) {
    result.status = OpenCudaProductionOwnerStatus::stopped;
    result.hal_status = hal::StatusId::stopped;
    result.reason = hal::ReasonId::shutdown_requested;
    result.shutdown_requested = true;
    result.quiescent = true;
    result.driver_context_release_permitted = true;
    return result;
  }
  if (impl_->production_owner_retired)
    return impl_->production_owner->ShutdownSnapshot();

  impl_->production_shutdown_requested = true;
  // A lease may be retired before the coordinator consumes its setup-prepared
  // executor. Destroy that prepared slot while the adapter facade is still
  // live so the runtime can retain any pending native work and report it from
  // the owner shutdown snapshot.
  impl_->preprepared_executor.reset();
  result = impl_->production_owner->RequestShutdown();
  if (result.driver_context_release_permitted) {
    if (impl_->production_adapter_view)
      impl_->production_adapter_view->Retire();
    impl_->production_owner_retired = true;
  }
  return result;
}

OpenCudaProductionShutdownResult
OpenCudaVideoExecutionFactory::production_shutdown_snapshot() const noexcept {
  if (!impl_ || !impl_->production_owner) {
    OpenCudaProductionShutdownResult result;
    result.status = OpenCudaProductionOwnerStatus::stopped;
    result.hal_status = hal::StatusId::stopped;
    result.reason = hal::ReasonId::shutdown_requested;
    result.shutdown_requested = true;
    result.quiescent = true;
    result.driver_context_release_permitted = true;
    return result;
  }
  return impl_->production_owner->ShutdownSnapshot();
}

bool OpenCudaVideoExecutionFactory::production_owner_retired() const noexcept {
  return impl_ && impl_->production_owner_retired;
}

video::VideoAdapterFactoryPreflight OpenCudaVideoExecutionFactory::Preflight(
    const video::VideoExecutionPlanRequest &request,
    const hal::ResolutionResult &frozen_resolution,
    const hal::DomainPlan &frozen_domain_plan,
    video::VideoExecutionRoute route) noexcept {
  video::VideoAdapterFactoryPreflight result;
  result.backend = hal::BackendId::cuda;
  result.provider = impl_ ? impl_->config.cached.readiness.provider
                          : hal::SemanticEffectProviderId::invalid;
  result.cached_evidence_only = true;
  result.accepts_frozen_resolution = true;
  result.performs_selection = false;
  if (!impl_ || impl_->config.factory_id == 0 ||
      impl_->config.prepare_candidate == nullptr) {
    result.status = hal::StatusId::unavailable;
    result.reason = hal::ReasonId::capability_evidence_incomplete;
    return result;
  }
  result.execution_domain = impl_->config.cached.execution_domain;
  result.readiness = impl_->config.cached.readiness;
  result.per_frame_transfers = TransferContract(
      impl_->config.cached, CompactIntent(request, frozen_resolution),
      frozen_domain_plan);
  if (route != video::VideoExecutionRoute::prepared_resident) {
    result.status = hal::StatusId::blocked;
    result.reason = hal::ReasonId::semantic_effect_unsupported;
    return result;
  }
  hal::ReasonId reason = hal::ReasonId::none;
  if (!ConservativeRoute(request, frozen_resolution, frozen_domain_plan,
                         &reason)) {
    result.status = hal::StatusId::blocked;
    result.reason = reason;
    return result;
  }
  if (!ExactCachedEvidence(impl_->config.cached, request, frozen_resolution,
                           frozen_domain_plan)) {
    result.status = hal::StatusId::blocked;
    result.reason = hal::ReasonId::capability_evidence_incomplete;
    return result;
  }
  result.prepared_program = BuildProgram(
      request, frozen_resolution, frozen_domain_plan, impl_->config.cached);
  if (!hal::ValidatePreparedProgram(result.prepared_program, &reason)) {
    result.status = hal::StatusId::blocked;
    result.reason = reason;
    return result;
  }
  result.status = result.readiness.status;
  result.reason = result.readiness.reason;
  return result;
}

std::unique_ptr<video::IVideoExecutionExecutor>
OpenCudaVideoExecutionFactory::Prepare(
    const video::FrozenVideoExecutionPlan &plan,
    video::VideoAdapterPrepareEvidence *evidence) noexcept {
  video::VideoAdapterPrepareEvidence local;
  if (evidence == nullptr)
    evidence = &local;
  *evidence = {};
  if (!impl_ || impl_->config.prepare_candidate == nullptr) {
    evidence->status = hal::StatusId::unavailable;
    evidence->reason = hal::ReasonId::capability_evidence_incomplete;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }
  // Reject forged counts before setup callbacks and readiness-mask creation.
  // Canonical CUDA programs contain at most one stage in each of seven slots.
  if (plan.prepared_program.stage_count == 0 ||
      plan.prepared_program.stage_count > kMaxCanonicalCudaStageCount ||
      plan.route != video::VideoExecutionRoute::prepared_resident ||
      plan.prepared_program.requested_backend != hal::BackendId::cuda ||
      plan.prepared_program.requested_provider !=
          impl_->config.cached.readiness.provider ||
      !video::ValidatePerFrameTransferContract(plan.per_frame_transfers,
                                               plan.domain_plan, nullptr) ||
      plan.prepared_program.stage_count != impl_->config.cached.stage_count) {
    evidence->status = hal::StatusId::blocked;
    evidence->reason = hal::ReasonId::malformed_contract;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }
  if (plan.prepared_program.execution_domain !=
      impl_->config.cached.execution_domain) {
    evidence->status = hal::StatusId::blocked;
    evidence->reason = hal::ReasonId::wrong_execution_domain;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }
  if (plan.prepared_program.owner_adapter_instance_id !=
      impl_->config.cached.execution_domain.adapter_instance_id) {
    evidence->status = hal::StatusId::blocked;
    evidence->reason = hal::ReasonId::foreign_adapter;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }
  if (plan.prepared_program.invalidation_key.generations !=
      impl_->config.cached.generations) {
    evidence->status = hal::StatusId::stale;
    evidence->reason = hal::ReasonId::wrong_generation;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }
  for (std::size_t index = 0; index < plan.prepared_program.stage_count;
       ++index) {
    if (!SameModel(plan.prepared_program.stages[index].model,
                   impl_->config.cached.models[index])) {
      evidence->status = hal::StatusId::blocked;
      evidence->reason = hal::ReasonId::capability_evidence_incomplete;
      evidence->candidate_resources_rolled_back = true;
      return nullptr;
    }
  }

  if (impl_->preprepared_executor) {
    const auto &primed = impl_->preprepared_plan;
    if (!SamePrimedPlan(primed, plan)) {
      evidence->status = hal::StatusId::stale;
      evidence->reason = hal::ReasonId::stale_program;
      evidence->candidate_resources_rolled_back = false;
      return nullptr;
    }
    const auto snapshot = impl_->preprepared_executor->Snapshot();
    evidence->status = snapshot.status;
    evidence->reason = snapshot.reason;
    evidence->readiness = snapshot.readiness;
    evidence->candidate_resources_rolled_back = false;
    return std::move(impl_->preprepared_executor);
  }

  OpenCudaVideoExecutionCandidate candidate;
  auto status = impl_->config.prepare_candidate(impl_->config.setup_context,
                                                plan, &candidate, evidence);
  if (!IsSetupSuccess(status) || !IsSetupSuccess(evidence->status)) {
    if (evidence->status == hal::StatusId::invalid)
      evidence->status = status;
    if (evidence->reason == hal::ReasonId::none)
      evidence->reason = hal::ReasonId::capability_evidence_incomplete;
    candidate.Reset();
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }
  if (candidate.owner == nullptr || candidate.destroy_owner == nullptr ||
      !ExactRegistration(candidate.registration, plan)) {
    candidate.Reset();
    evidence->status = hal::StatusId::blocked;
    evidence->reason = hal::ReasonId::foreign_adapter;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }

  try {
    auto session = std::make_unique<hal::PreparedProgramSession>();
    hal::ReadinessSnapshot readiness;
    hal::ReasonId reason = hal::ReasonId::none;
    status = session->Configure(plan.prepared_program,
                                *candidate.registration.resident_adapter,
                                &readiness, &reason);
    if (!IsSetupSuccess(status) || !ExactReadiness(readiness, plan)) {
      session->Shutdown();
      candidate.Reset();
      evidence->status =
          IsSetupSuccess(status) ? hal::StatusId::blocked : status;
      evidence->reason = IsSetupSuccess(status)
                             ? hal::ReasonId::capability_evidence_incomplete
                             : reason;
      evidence->readiness = readiness;
      evidence->candidate_resources_rolled_back = true;
      return nullptr;
    }
    if (candidate.commit_prepared_owner != nullptr) {
      status = candidate.commit_prepared_owner(
          candidate.owner, plan.prepared_program, readiness, &reason);
      if (!IsSetupSuccess(status)) {
        session->Shutdown();
        candidate.Reset();
        evidence->status = status;
        evidence->reason = reason == hal::ReasonId::none
                               ? hal::ReasonId::capability_evidence_incomplete
                               : reason;
        evidence->readiness = readiness;
        evidence->candidate_resources_rolled_back = true;
        return nullptr;
      }
    }
    evidence->status = status;
    evidence->reason = reason;
    evidence->readiness = readiness;
    evidence->candidate_resources_rolled_back = false;
    return std::make_unique<OpenCudaVideoExecutionExecutor>(
        std::move(candidate), std::move(session), plan.plan_id);
  } catch (const std::bad_alloc &) {
    candidate.Reset();
    evidence->status = hal::StatusId::resource_exhausted;
    evidence->reason = hal::ReasonId::resource_exhausted;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  } catch (...) {
    candidate.Reset();
    evidence->status = hal::StatusId::blocked;
    evidence->reason = hal::ReasonId::capability_evidence_incomplete;
    evidence->candidate_resources_rolled_back = true;
    return nullptr;
  }
}

} // namespace studiocast::open_cuda
