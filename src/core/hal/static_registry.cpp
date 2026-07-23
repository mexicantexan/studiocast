#include "core/hal/static_registry.h"

namespace studiocast::hal {
namespace {

void SetReason(ReasonId *reason, ReasonId value) noexcept {
  if (reason)
    *reason = value;
}

bool IsCapabilityStatus(StatusId status) noexcept {
  return status == StatusId::ready || status == StatusId::degraded ||
         status == StatusId::blocked || status == StatusId::unavailable;
}

bool ProviderMatchesBackend(BackendId backend,
                            SemanticEffectProviderId provider) noexcept {
  switch (backend) {
  case BackendId::cpu:
    return false;
  case BackendId::cuda:
    return provider == SemanticEffectProviderId::open_video_onnxruntime_cuda;
  case BackendId::maxine:
    return provider == SemanticEffectProviderId::maxine_vfx ||
           provider == SemanticEffectProviderId::maxine_ar;
  case BackendId::open_video_onnxruntime:
    return provider == SemanticEffectProviderId::open_video_onnxruntime_cpu;
  case BackendId::vulkan:
    return provider == SemanticEffectProviderId::open_vulkan_production_kernels;
  case BackendId::test_fake:
    return provider == SemanticEffectProviderId::test_fake;
  case BackendId::invalid:
    return false;
  }
  return false;
}

bool ValidateSemanticCapability(
    const AdapterRegistration &registration,
    const SemanticCapabilityEvidence &capability) noexcept {
  if (capability.semantic == SemanticEffectId::invalid ||
      capability.provider == SemanticEffectProviderId::invalid ||
      !ProviderMatchesBackend(registration.backend, capability.provider) ||
      !IsCapabilityStatus(capability.status) ||
      ((capability.status == StatusId::blocked ||
        capability.status == StatusId::unavailable ||
        capability.status == StatusId::degraded) &&
       capability.reason == ReasonId::none) ||
      (capability.status == StatusId::ready &&
       capability.reason != ReasonId::none) ||
      ((capability.status == StatusId::ready ||
        capability.status == StatusId::degraded) &&
       !capability.production_eligible)) {
    return false;
  }
  if (capability.production_eligible &&
      (!capability.live_stage || !capability.exact_evidence ||
       (capability.status != StatusId::ready &&
        capability.status != StatusId::degraded))) {
    return false;
  }
  if (capability.cpu_tail == CpuTailId::invalid &&
      capability.cpu_tail_scratch_bytes != 0) {
    return false;
  }
  if (capability.supports_tracked_vignette_center &&
      capability.semantic != SemanticEffectId::vignette) {
    return false;
  }
  // This is the current reviewed production boundary. Experimental Vulkan
  // stages remain representable as evidence, but cannot register as selectable
  // production providers.
  if (registration.backend == BackendId::vulkan &&
      capability.production_eligible &&
      (!IsVulkanProductionSemantic(capability.semantic) ||
       capability.supports_tracked_vignette_center)) {
    return false;
  }
  return true;
}

} // namespace

StatusId
StaticAdapterRegistry::Register(const AdapterRegistration &registration,
                                ReasonId *reason) noexcept {
  SetReason(reason, ReasonId::none);
  if (registration.adapter_instance_id == 0 ||
      registration.backend == BackendId::invalid ||
      registration.selection_priority == 0 ||
      registration.semantic_capability_count >
          registration.semantic_capabilities.size() ||
      registration.primitive_capability_count >
          registration.primitive_capabilities.size() ||
      !ValidateExecutionDomain(registration.execution_domain) ||
      registration.execution_domain.adapter_instance_id !=
          registration.adapter_instance_id ||
      registration.execution_domain.backend != registration.backend) {
    SetReason(reason, ReasonId::malformed_contract);
    return StatusId::blocked;
  }
  if (registration.resident_adapter != nullptr &&
      (registration.resident_adapter->adapter_instance_id() !=
           registration.adapter_instance_id ||
       registration.resident_adapter->backend_id() != registration.backend)) {
    SetReason(reason, ReasonId::foreign_adapter);
    return StatusId::blocked;
  }
  if (FindAdapter(registration.adapter_instance_id) != nullptr) {
    SetReason(reason, ReasonId::foreign_adapter);
    return StatusId::blocked;
  }

  for (std::size_t index = 0; index < registration.semantic_capability_count;
       ++index) {
    const auto &capability = registration.semantic_capabilities[index];
    if (!ValidateSemanticCapability(registration, capability) ||
        (capability.production_eligible &&
         registration.resident_adapter == nullptr)) {
      SetReason(reason, ReasonId::malformed_contract);
      return StatusId::blocked;
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (registration.semantic_capabilities[previous].semantic ==
              capability.semantic &&
          registration.semantic_capabilities[previous].provider ==
              capability.provider) {
        SetReason(reason, ReasonId::malformed_contract);
        return StatusId::blocked;
      }
    }
  }

  for (std::size_t index = 0; index < registration.primitive_capability_count;
       ++index) {
    const auto &capability = registration.primitive_capabilities[index];
    if (capability.operation == PrimitiveOperationId::invalid ||
        !IsCapabilityStatus(capability.status) ||
        (capability.available && capability.status != StatusId::ready) ||
        (!capability.available && capability.status == StatusId::ready) ||
        (capability.status == StatusId::ready &&
         capability.reason != ReasonId::none) ||
        (capability.status != StatusId::ready &&
         capability.reason == ReasonId::none)) {
      SetReason(reason, ReasonId::malformed_contract);
      return StatusId::blocked;
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (registration.primitive_capabilities[previous].operation ==
          capability.operation) {
        SetReason(reason, ReasonId::malformed_contract);
        return StatusId::blocked;
      }
    }
  }

  if (count_ == registrations_.size()) {
    SetReason(reason, ReasonId::resource_exhausted);
    return StatusId::resource_exhausted;
  }
  registrations_[count_++] = registration;
  return StatusId::ready;
}

const AdapterRegistration *
StaticAdapterRegistry::at(std::size_t index) const noexcept {
  return index < count_ ? &registrations_[index] : nullptr;
}

const AdapterRegistration *StaticAdapterRegistry::FindAdapter(
    std::uint64_t adapter_instance_id) const noexcept {
  for (std::size_t index = 0; index < count_; ++index) {
    if (registrations_[index].adapter_instance_id == adapter_instance_id)
      return &registrations_[index];
  }
  return nullptr;
}

const PrimitiveCapabilityEvidence *StaticAdapterRegistry::FindPrimitive(
    std::uint64_t adapter_instance_id,
    PrimitiveOperationId operation) const noexcept {
  const auto *registration = FindAdapter(adapter_instance_id);
  if (registration == nullptr)
    return nullptr;
  for (std::size_t index = 0; index < registration->primitive_capability_count;
       ++index) {
    if (registration->primitive_capabilities[index].operation == operation)
      return &registration->primitive_capabilities[index];
  }
  return nullptr;
}

} // namespace studiocast::hal
