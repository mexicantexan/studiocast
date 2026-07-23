#include "core/hal/resolver.h"

#include <algorithm>
#include <limits>

namespace studiocast::hal {
namespace {

struct Candidate {
  const AdapterRegistration *registration = nullptr;
  const SemanticCapabilityEvidence *capability = nullptr;
};

struct CandidateSearch {
  Candidate selected{};
  ReasonId unavailable_reason = ReasonId::none;
  ReasonId requested_backend_reason = ReasonId::none;
  bool semantic_evidence_exists = false;
  bool requested_backend_evidence_exists = false;
};

bool IsSelectable(const SemanticCapabilityEvidence &capability) noexcept {
  return capability.live_stage && capability.production_eligible &&
         capability.exact_evidence &&
         (capability.status == StatusId::ready ||
          capability.status == StatusId::degraded);
}

bool BetterCandidate(const Candidate &candidate, const Candidate &current,
                     const BackendRequest &request) noexcept {
  if (current.registration == nullptr)
    return true;
  const bool candidate_requested =
      request.mode == BackendRequestMode::explicit_backend &&
      candidate.registration->backend == request.backend;
  const bool current_requested =
      request.mode == BackendRequestMode::explicit_backend &&
      current.registration->backend == request.backend;
  if (candidate_requested != current_requested)
    return candidate_requested;
  if (candidate.registration->selection_priority !=
      current.registration->selection_priority) {
    return candidate.registration->selection_priority <
           current.registration->selection_priority;
  }
  if (candidate.registration->backend != current.registration->backend) {
    return static_cast<std::uint32_t>(candidate.registration->backend) <
           static_cast<std::uint32_t>(current.registration->backend);
  }
  if (candidate.capability->provider != current.capability->provider) {
    return static_cast<std::uint32_t>(candidate.capability->provider) <
           static_cast<std::uint32_t>(current.capability->provider);
  }
  return candidate.registration->adapter_instance_id <
         current.registration->adapter_instance_id;
}

CandidateSearch FindCandidate(const StaticAdapterRegistry &registry,
                              const CanonicalStageIntent &intent,
                              const BackendRequest &request) noexcept {
  CandidateSearch search;
  std::uint32_t unavailable_priority =
      std::numeric_limits<std::uint32_t>::max();
  std::uint64_t unavailable_adapter = std::numeric_limits<std::uint64_t>::max();
  std::uint32_t requested_unavailable_priority =
      std::numeric_limits<std::uint32_t>::max();
  std::uint64_t requested_unavailable_adapter =
      std::numeric_limits<std::uint64_t>::max();

  for (std::size_t registration_index = 0; registration_index < registry.size();
       ++registration_index) {
    const auto *registration = registry.at(registration_index);
    if (registration == nullptr)
      continue;
    for (std::size_t capability_index = 0;
         capability_index < registration->semantic_capability_count;
         ++capability_index) {
      const auto &capability =
          registration->semantic_capabilities[capability_index];
      if (capability.semantic != intent.semantic)
        continue;
      search.semantic_evidence_exists = true;
      if (request.mode == BackendRequestMode::explicit_backend &&
          registration->backend == request.backend) {
        search.requested_backend_evidence_exists = true;
      }
      const bool configuration_supported =
          !intent.requires_tracked_vignette_center ||
          capability.supports_tracked_vignette_center;
      if (request.mode == BackendRequestMode::explicit_backend &&
          registration->backend == request.backend &&
          (!IsSelectable(capability) || !configuration_supported) &&
          (registration->selection_priority < requested_unavailable_priority ||
           (registration->selection_priority ==
                requested_unavailable_priority &&
            registration->adapter_instance_id <
                requested_unavailable_adapter))) {
        requested_unavailable_priority = registration->selection_priority;
        requested_unavailable_adapter = registration->adapter_instance_id;
        search.requested_backend_reason = capability.reason;
      }
      if (request.strict && registration->backend != request.backend)
        continue;
      if (IsSelectable(capability) && configuration_supported) {
        const Candidate candidate{registration, &capability};
        if (BetterCandidate(candidate, search.selected, request))
          search.selected = candidate;
      } else if (registration->selection_priority < unavailable_priority ||
                 (registration->selection_priority == unavailable_priority &&
                  registration->adapter_instance_id < unavailable_adapter)) {
        unavailable_priority = registration->selection_priority;
        unavailable_adapter = registration->adapter_instance_id;
        search.unavailable_reason = capability.reason;
      }
    }
  }
  return search;
}

ReasonId FallbackReason(const CandidateSearch &search) noexcept {
  return search.requested_backend_reason != ReasonId::none
             ? search.requested_backend_reason
             : ReasonId::semantic_provider_unavailable;
}

ReasonId NoCandidateReason(const CandidateSearch &search,
                           const BackendRequest &request) noexcept {
  if (!search.semantic_evidence_exists)
    return ReasonId::semantic_effect_unsupported;
  if (request.strict && !search.requested_backend_evidence_exists)
    return ReasonId::strict_backend_unavailable;
  if (search.unavailable_reason != ReasonId::none)
    return search.unavailable_reason;
  return ReasonId::semantic_provider_unavailable;
}

bool SameSelectedBackend(const ResolutionResult &result) noexcept {
  BackendId selected = BackendId::invalid;
  for (std::size_t index = 0; index < result.stage_count; ++index) {
    const auto &stage = result.stages[index];
    if (stage.skipped_optional)
      continue;
    if (selected == BackendId::invalid)
      selected = stage.selected_backend;
    else if (stage.selected_backend != selected)
      return false;
  }
  return true;
}

bool SameSelectedProvider(const ResolutionResult &result) noexcept {
  SemanticEffectProviderId selected = SemanticEffectProviderId::invalid;
  for (std::size_t index = 0; index < result.stage_count; ++index) {
    const auto &stage = result.stages[index];
    if (stage.skipped_optional)
      continue;
    if (selected == SemanticEffectProviderId::invalid)
      selected = stage.provider;
    else if (stage.provider != selected)
      return false;
  }
  return true;
}

template <typename T> std::uint64_t Numeric(T value) noexcept {
  return static_cast<std::uint64_t>(value);
}

void AddDiscrepancy(ShadowComparison *comparison, ShadowField field,
                    std::size_t stage_index, std::uint64_t live,
                    std::uint64_t shadow) noexcept {
  if (live == shadow)
    return;
  comparison->exact_match = false;
  if (comparison->discrepancy_count == comparison->discrepancies.size()) {
    comparison->truncated = true;
    return;
  }
  auto &entry = comparison->discrepancies[comparison->discrepancy_count++];
  entry.field = field;
  entry.stage_index = static_cast<std::uint32_t>(stage_index);
  entry.live_value = live;
  entry.shadow_value = shadow;
}

} // namespace

ResolutionResult
ResolveSemanticStages(const StaticAdapterRegistry &registry,
                      const ResolutionRequest &request) noexcept {
  ResolutionResult result;
  result.request_mode = request.backend.mode;
  result.strict_backend_selection = request.backend.strict;
  result.requested_backend = request.backend.backend;
  result.stage_count = request.stage_count;
  if (request.stage_count > request.stages.size() ||
      (request.backend.mode == BackendRequestMode::explicit_backend &&
       request.backend.backend == BackendId::invalid) ||
      (request.backend.mode == BackendRequestMode::automatic &&
       (request.backend.backend != BackendId::invalid ||
        request.backend.strict))) {
    result.status = StatusId::blocked;
    result.reason = ReasonId::malformed_contract;
    return result;
  }
  if (request.stage_count == 0) {
    result.status = StatusId::passthrough;
    result.reason = ReasonId::none;
    result.available = true;
    result.passthrough = true;
    return result;
  }

  std::uint32_t previous_stage_id = 0;
  bool have_selected_stage = false;
  for (std::size_t index = 0; index < request.stage_count; ++index) {
    const auto &intent = request.stages[index];
    auto &resolved = result.stages[index];
    resolved.intent = intent;
    if (intent.stage_id == 0 || intent.stage_id <= previous_stage_id ||
        intent.semantic == SemanticEffectId::invalid ||
        (intent.requires_tracked_vignette_center &&
         intent.semantic != SemanticEffectId::vignette) ||
        (intent.required &&
         intent.failure_action != FailureAction::fail_closed) ||
        (!intent.required &&
         intent.failure_action == FailureAction::fail_closed)) {
      result.status = StatusId::blocked;
      result.reason = ReasonId::malformed_contract;
      return result;
    }
    previous_stage_id = intent.stage_id;

    const CandidateSearch search =
        FindCandidate(registry, intent, request.backend);
    const Candidate selected = search.selected;
    if (selected.registration == nullptr) {
      resolved.status = StatusId::blocked;
      resolved.reason = NoCandidateReason(search, request.backend);
      if (intent.required) {
        result.status = StatusId::blocked;
        result.reason = resolved.reason;
        return result;
      }
      resolved.skipped_optional = true;
      resolved.status = StatusId::degraded;
      result.degraded = true;
      if (result.reason == ReasonId::none)
        result.reason = resolved.reason;
      continue;
    }

    resolved.status = selected.capability->status;
    resolved.reason = selected.capability->reason;
    resolved.selected_backend = selected.registration->backend;
    resolved.provider = selected.capability->provider;
    resolved.adapter_instance_id = selected.registration->adapter_instance_id;
    resolved.execution_domain = selected.registration->execution_domain;
    resolved.cpu_tail = selected.capability->cpu_tail;
    resolved.cpu_tail_scratch_bytes =
        selected.capability->cpu_tail_scratch_bytes;
    have_selected_stage = true;
    const bool fallback_selected =
        request.backend.mode == BackendRequestMode::explicit_backend &&
        !request.backend.strict &&
        selected.registration->backend != request.backend.backend;
    if (fallback_selected) {
      resolved.status = StatusId::degraded;
      resolved.reason = FallbackReason(search);
      resolved.fallback_selected = true;
      result.fallback = true;
      result.degraded = true;
      if (result.reason == ReasonId::none)
        result.reason = resolved.reason;
    } else if (selected.capability->status == StatusId::degraded ||
               selected.capability->cpu_tail != CpuTailId::invalid) {
      result.degraded = true;
      if (result.reason == ReasonId::none) {
        result.reason = selected.capability->reason != ReasonId::none
                            ? selected.capability->reason
                            : ReasonId::cpu_tail_required;
      }
    }
  }

  if (!have_selected_stage) {
    result.status = StatusId::passthrough;
    result.available = true;
    result.passthrough = true;
    result.degraded = true;
    return result;
  }
  result.status = result.degraded ? StatusId::degraded : StatusId::ready;
  result.available = true;
  if (SameSelectedBackend(result)) {
    for (std::size_t index = 0; index < result.stage_count; ++index) {
      if (!result.stages[index].skipped_optional) {
        result.selected_backend = result.stages[index].selected_backend;
        break;
      }
    }
  }
  if (SameSelectedProvider(result)) {
    for (std::size_t index = 0; index < result.stage_count; ++index) {
      if (!result.stages[index].skipped_optional) {
        result.selected_provider = result.stages[index].provider;
        break;
      }
    }
  }
  return result;
}

ShadowComparison
CompareShadowResolution(const ResolutionResult &live,
                        const ResolutionResult &shadow) noexcept {
  ShadowComparison comparison;
  AddDiscrepancy(&comparison, ShadowField::request_mode, 0,
                 Numeric(live.request_mode), Numeric(shadow.request_mode));
  AddDiscrepancy(&comparison, ShadowField::requested_backend, 0,
                 Numeric(live.requested_backend),
                 Numeric(shadow.requested_backend));
  AddDiscrepancy(&comparison, ShadowField::strict_backend_selection, 0,
                 live.strict_backend_selection,
                 shadow.strict_backend_selection);
  AddDiscrepancy(&comparison, ShadowField::selected_backend, 0,
                 Numeric(live.selected_backend),
                 Numeric(shadow.selected_backend));
  AddDiscrepancy(&comparison, ShadowField::selected_provider, 0,
                 Numeric(live.selected_provider),
                 Numeric(shadow.selected_provider));
  AddDiscrepancy(&comparison, ShadowField::active_backend, 0,
                 Numeric(live.active_backend), Numeric(shadow.active_backend));
  AddDiscrepancy(&comparison, ShadowField::overall_status, 0,
                 Numeric(live.status), Numeric(shadow.status));
  AddDiscrepancy(&comparison, ShadowField::overall_reason, 0,
                 Numeric(live.reason), Numeric(shadow.reason));
  AddDiscrepancy(&comparison, ShadowField::available, 0, live.available,
                 shadow.available);
  AddDiscrepancy(&comparison, ShadowField::degraded, 0, live.degraded,
                 shadow.degraded);
  AddDiscrepancy(&comparison, ShadowField::fallback, 0, live.fallback,
                 shadow.fallback);
  AddDiscrepancy(&comparison, ShadowField::passthrough, 0, live.passthrough,
                 shadow.passthrough);
  AddDiscrepancy(&comparison, ShadowField::stage_count, 0, live.stage_count,
                 shadow.stage_count);

  const std::size_t common_count =
      std::min({live.stage_count, shadow.stage_count, live.stages.size(),
                shadow.stages.size()});
  for (std::size_t index = 0; index < common_count; ++index) {
    AddDiscrepancy(&comparison, ShadowField::stage_id, index,
                   live.stages[index].intent.stage_id,
                   shadow.stages[index].intent.stage_id);
    AddDiscrepancy(&comparison, ShadowField::stage_semantic, index,
                   Numeric(live.stages[index].intent.semantic),
                   Numeric(shadow.stages[index].intent.semantic));
    AddDiscrepancy(&comparison, ShadowField::stage_required, index,
                   live.stages[index].intent.required,
                   shadow.stages[index].intent.required);
    AddDiscrepancy(&comparison, ShadowField::stage_backend, index,
                   Numeric(live.stages[index].selected_backend),
                   Numeric(shadow.stages[index].selected_backend));
    AddDiscrepancy(&comparison, ShadowField::stage_provider, index,
                   Numeric(live.stages[index].provider),
                   Numeric(shadow.stages[index].provider));
    AddDiscrepancy(&comparison, ShadowField::stage_status, index,
                   Numeric(live.stages[index].status),
                   Numeric(shadow.stages[index].status));
    AddDiscrepancy(&comparison, ShadowField::stage_reason, index,
                   Numeric(live.stages[index].reason),
                   Numeric(shadow.stages[index].reason));
    AddDiscrepancy(&comparison, ShadowField::stage_skipped_optional, index,
                   live.stages[index].skipped_optional,
                   shadow.stages[index].skipped_optional);
    AddDiscrepancy(&comparison, ShadowField::stage_fallback, index,
                   live.stages[index].fallback_selected,
                   shadow.stages[index].fallback_selected);
  }
  return comparison;
}

} // namespace studiocast::hal
