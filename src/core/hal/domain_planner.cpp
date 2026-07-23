#include "core/hal/domain_planner.h"

namespace studiocast::hal {
namespace {

bool SameIsland(const ResidentIslandPlan &island,
                const ResolvedStage &stage) noexcept {
  return island.adapter_instance_id == stage.adapter_instance_id &&
         island.execution_domain == stage.execution_domain;
}

bool IsDeviceResident(const ExecutionDomainId &domain) noexcept {
  return domain.backend != BackendId::cpu &&
         domain.backend != BackendId::open_video_onnxruntime &&
         domain.api_family != ApiFamily::host;
}

bool AddBoundary(DomainPlan *plan, const ExecutionBoundary &boundary) noexcept {
  if (plan->boundary_count == plan->boundaries.size())
    return false;
  plan->boundaries[plan->boundary_count++] = boundary;
  return true;
}

bool AddIsland(DomainPlan *plan, const AdapterRegistration &registration,
               std::size_t stage_index) noexcept {
  if (plan->island_count == plan->islands.size())
    return false;
  auto &island = plan->islands[plan->island_count++];
  island.adapter_instance_id = registration.adapter_instance_id;
  island.resident_adapter = registration.resident_adapter;
  island.execution_domain = registration.execution_domain;
  island.resolved_stage_indices[island.stage_count++] = stage_index;
  island.requires_upload = IsDeviceResident(registration.execution_domain);
  island.requires_final_readback =
      IsDeviceResident(registration.execution_domain);
  return true;
}

} // namespace

DomainPlan PlanExecutionDomains(const StaticAdapterRegistry &registry,
                                const ResolutionResult &resolution,
                                CrossAdapterPolicy policy) noexcept {
  DomainPlan plan;
  if (policy != CrossAdapterPolicy::reject &&
      policy != CrossAdapterPolicy::split_with_measured_transfer) {
    plan.status = StatusId::blocked;
    plan.reason = ReasonId::malformed_contract;
    return plan;
  }
  if (resolution.status == StatusId::blocked ||
      resolution.status == StatusId::unavailable ||
      resolution.status == StatusId::failed) {
    plan.status = resolution.status;
    plan.reason = resolution.reason;
    return plan;
  }
  if (resolution.passthrough || resolution.stage_count == 0) {
    plan.status = StatusId::passthrough;
    plan.passthrough = true;
    return plan;
  }
  if (resolution.stage_count > resolution.stages.size()) {
    plan.status = StatusId::blocked;
    plan.reason = ReasonId::malformed_contract;
    return plan;
  }

  const ResolvedStage *previous_stage = nullptr;
  bool force_new_island = false;
  for (std::size_t stage_index = 0; stage_index < resolution.stage_count;
       ++stage_index) {
    const auto &stage = resolution.stages[stage_index];
    if (stage.skipped_optional)
      continue;
    const auto *registration = registry.FindAdapter(stage.adapter_instance_id);
    if (registration == nullptr ||
        registration->backend != stage.selected_backend) {
      plan.status = StatusId::blocked;
      plan.reason = ReasonId::foreign_adapter;
      return plan;
    }
    if (registration->execution_domain != stage.execution_domain) {
      plan.status = StatusId::blocked;
      plan.reason = ReasonId::wrong_execution_domain;
      return plan;
    }

    const bool need_new_island =
        plan.island_count == 0 || force_new_island ||
        !SameIsland(plan.islands[plan.island_count - 1], stage);
    if (need_new_island) {
      const bool already_split_by_cpu_tail = force_new_island;
      if (plan.island_count != 0 && previous_stage != nullptr) {
        const auto &prior_island = plan.islands[plan.island_count - 1];
        const bool cross_adapter =
            prior_island.adapter_instance_id != stage.adapter_instance_id ||
            prior_island.execution_domain != stage.execution_domain;
        if (cross_adapter && !already_split_by_cpu_tail &&
            policy == CrossAdapterPolicy::reject) {
          plan.status = StatusId::blocked;
          plan.reason = ReasonId::cross_adapter_residency_forbidden;
          return plan;
        }
        if (cross_adapter && !already_split_by_cpu_tail) {
          ExecutionBoundary boundary;
          boundary.kind = ExecutionBoundaryKind::adapter_transfer;
          boundary.after_stage_id = previous_stage->intent.stage_id;
          boundary.before_stage_id = stage.intent.stage_id;
          boundary.source_domain = prior_island.execution_domain;
          boundary.destination_domain = stage.execution_domain;
          boundary.readback_required = IsDeviceResident(boundary.source_domain);
          boundary.following_upload_required =
              IsDeviceResident(boundary.destination_domain);
          if (!AddBoundary(&plan, boundary)) {
            plan.status = StatusId::resource_exhausted;
            plan.reason = ReasonId::resource_exhausted;
            return plan;
          }
          if (boundary.readback_required)
            ++plan.expected_transfers.intermediate_readbacks;
          if (boundary.following_upload_required)
            ++plan.expected_transfers.intermediate_uploads;
        }
      }
      if (!AddIsland(&plan, *registration, stage_index)) {
        plan.status = StatusId::resource_exhausted;
        plan.reason = ReasonId::resource_exhausted;
        return plan;
      }
    } else {
      auto &island = plan.islands[plan.island_count - 1];
      if (island.stage_count == island.resolved_stage_indices.size()) {
        plan.status = StatusId::resource_exhausted;
        plan.reason = ReasonId::resource_exhausted;
        return plan;
      }
      island.resolved_stage_indices[island.stage_count++] = stage_index;
    }

    force_new_island = false;
    if (stage.cpu_tail != CpuTailId::invalid) {
      ExecutionBoundary boundary;
      boundary.kind = ExecutionBoundaryKind::cpu_tail;
      boundary.after_stage_id = stage.intent.stage_id;
      boundary.cpu_tail = stage.cpu_tail;
      boundary.scratch_bytes = stage.cpu_tail_scratch_bytes;
      boundary.source_domain = stage.execution_domain;
      boundary.readback_required = IsDeviceResident(stage.execution_domain);
      boundary.following_upload_required = false;
      bool has_following_stage = false;
      for (std::size_t following = stage_index + 1;
           following < resolution.stage_count; ++following) {
        if (!resolution.stages[following].skipped_optional) {
          has_following_stage = true;
          boundary.before_stage_id =
              resolution.stages[following].intent.stage_id;
          boundary.destination_domain =
              resolution.stages[following].execution_domain;
          boundary.following_upload_required =
              IsDeviceResident(boundary.destination_domain);
          break;
        }
      }
      if (!AddBoundary(&plan, boundary)) {
        plan.status = StatusId::resource_exhausted;
        plan.reason = ReasonId::resource_exhausted;
        return plan;
      }
      ++plan.expected_transfers.cpu_tail_boundaries;
      if (has_following_stage) {
        if (boundary.readback_required)
          ++plan.expected_transfers.intermediate_readbacks;
        if (boundary.following_upload_required)
          ++plan.expected_transfers.intermediate_uploads;
      }
      force_new_island = has_following_stage;
    }
    previous_stage = &stage;
  }

  if (plan.island_count == 0) {
    plan.status = StatusId::passthrough;
    plan.passthrough = true;
    return plan;
  }
  plan.expected_transfers.uploads = plan.islands[0].requires_upload ? 1 : 0;
  plan.expected_transfers.final_readbacks =
      plan.islands[plan.island_count - 1].requires_final_readback ? 1 : 0;
  plan.expected_transfers.synchronizations = plan.island_count;
  plan.status = resolution.degraded ? StatusId::degraded : StatusId::ready;
  plan.reason = resolution.reason;
  return plan;
}

} // namespace studiocast::hal
