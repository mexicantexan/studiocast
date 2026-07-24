#include "core/video/production_video_execution_composition.h"

#include "core/vulkan/video_execution_factory.h"

#include <new>
#include <utility>

namespace studiocast::video {
namespace {

void SetReason(hal::ReasonId *reason, hal::ReasonId value) noexcept {
  if (reason)
    *reason = value;
}

bool IsSetupSuccess(hal::StatusId status) noexcept {
  return status == hal::StatusId::ready || status == hal::StatusId::degraded;
}

std::uint64_t HashDomain(const hal::ExecutionDomainId &domain) noexcept {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const auto mix = [&](std::uint64_t value) {
    hash ^= value;
    hash *= UINT64_C(1099511628211);
  };
  mix(static_cast<std::uint64_t>(domain.backend));
  mix(static_cast<std::uint64_t>(domain.api_family));
  mix(domain.physical_device.vendor_id);
  mix(domain.physical_device.device_id);
  for (const auto byte : domain.physical_device.stable_id)
    mix(byte);
  mix(domain.logical_context_generation);
  mix(static_cast<std::uint64_t>(domain.allocator_family));
  mix(static_cast<std::uint64_t>(domain.memory_handle_family));
  mix(domain.queue_or_stream_identity);
  mix(static_cast<std::uint64_t>(domain.ownership_model));
  mix(static_cast<std::uint64_t>(domain.synchronization_protocol));
  mix(domain.adapter_instance_id);
  return hash == 0 ? 1 : hash;
}

} // namespace

struct ProductionVideoExecutionComposition::Impl {
  std::unique_ptr<vulkan::VulkanVideoExecutionFactory> vulkan_factory;
  // Declared after the factory so it is destroyed first. Production Vulkan
  // executors hold a non-owning resident-adapter view into the factory.
  std::unique_ptr<hal::StaticAdapterRegistry> registry;
  std::unique_ptr<VideoExecutionCoordinator> coordinator;
  bool injected = false;
  bool registered = false;

  bool Register(const hal::AdapterRegistration &adapter,
                const VideoAdapterFactoryRegistration &factory,
                hal::ReasonId *reason) noexcept {
    if (registered) {
      SetReason(reason, hal::ReasonId::none);
      return true;
    }
    try {
      auto candidate_registry = std::make_unique<hal::StaticAdapterRegistry>();
      auto status = candidate_registry->Register(adapter, reason);
      if (!IsSetupSuccess(status))
        return false;
      auto candidate_coordinator =
          std::make_unique<VideoExecutionCoordinator>(*candidate_registry);
      status = candidate_coordinator->RegisterFactory(factory, reason);
      if (!IsSetupSuccess(status))
        return false;

      if (coordinator)
        coordinator->Shutdown();
      coordinator = std::move(candidate_coordinator);
      registry = std::move(candidate_registry);
      registered = true;
      return true;
    } catch (const std::bad_alloc &) {
      SetReason(reason, hal::ReasonId::resource_exhausted);
      return false;
    } catch (...) {
      SetReason(reason, hal::ReasonId::malformed_contract);
      return false;
    }
  }

  bool EnsureProductionFactory(const VideoExecutionPlanRequest &request,
                               hal::ReasonId *reason) noexcept {
    if (registered)
      return true;
    if (injected) {
      SetReason(reason, hal::ReasonId::capability_evidence_incomplete);
      return false;
    }
    if (request.intent.backend.mode !=
            hal::BackendRequestMode::explicit_backend ||
        request.intent.backend.backend != hal::BackendId::vulkan ||
        !request.intent.backend.strict) {
      SetReason(reason, hal::ReasonId::strict_backend_unavailable);
      return false;
    }
    try {
      vulkan_factory = std::make_unique<vulkan::VulkanVideoExecutionFactory>();
    } catch (...) {
      SetReason(reason, hal::ReasonId::resource_exhausted);
      return false;
    }
    const auto status = vulkan_factory->Initialize(reason);
    if (!IsSetupSuccess(status)) {
      vulkan_factory.reset();
      return false;
    }
    if (!Register(vulkan_factory->adapter_registration(),
                  vulkan_factory->factory_registration(), reason)) {
      vulkan_factory.reset();
      return false;
    }
    return true;
  }
};

ProductionVideoExecutionComposition::ProductionVideoExecutionComposition()
    : impl_(std::make_unique<Impl>()) {}

ProductionVideoExecutionComposition::ProductionVideoExecutionComposition(
    const hal::AdapterRegistration &adapter,
    const VideoAdapterFactoryRegistration &factory)
    : impl_(std::make_unique<Impl>()) {
  impl_->injected = true;
  hal::ReasonId ignored = hal::ReasonId::none;
  (void)impl_->Register(adapter, factory, &ignored);
}

ProductionVideoExecutionComposition::~ProductionVideoExecutionComposition() {
  Shutdown();
}

hal::StatusId ProductionVideoExecutionComposition::Prepare(
    const VideoExecutionPlanRequest &request,
    VideoExecutionPrepareEvidence *evidence, hal::ReasonId *reason) noexcept {
  if (!impl_) {
    SetReason(reason, hal::ReasonId::resource_exhausted);
    return hal::StatusId::resource_exhausted;
  }
  if (!impl_->EnsureProductionFactory(request, reason)) {
    if (evidence) {
      *evidence = {};
      evidence->status = hal::StatusId::unavailable;
      evidence->reason =
          reason ? *reason : hal::ReasonId::capability_evidence_incomplete;
    }
    return hal::StatusId::unavailable;
  }
  if (!impl_->coordinator) {
    SetReason(reason, hal::ReasonId::capability_evidence_incomplete);
    return hal::StatusId::unavailable;
  }

  VideoExecutionPlanRequest exact = request;
  if (!impl_->injected && impl_->vulkan_factory) {
    const auto adapter = impl_->vulkan_factory->adapter_registration();
    exact.invalidation_key.generations.context =
        adapter.execution_domain.logical_context_generation;
    if (exact.invalidation_key.generations.provider == 0)
      exact.invalidation_key.generations.provider = 1;
    if (exact.invalidation_key.generations.device == 0)
      exact.invalidation_key.generations.device = 1;
    exact.invalidation_key.execution_domain_hash =
        HashDomain(adapter.execution_domain);
  }
  return impl_->coordinator->Prepare(exact, evidence, reason);
}

VideoExecutionFrameResult ProductionVideoExecutionComposition::Execute(
    const hal::HostFrameBinding &binding) noexcept {
  if (!impl_ || !impl_->coordinator) {
    VideoExecutionFrameResult result;
    result.status = hal::StatusId::resource_exhausted;
    result.reason = hal::ReasonId::resource_exhausted;
    return result;
  }
  return impl_->coordinator->Execute(binding);
}

hal::StatusId ProductionVideoExecutionComposition::CommitOutput(
    const hal::CompletionToken &completion, bool output_write_succeeded,
    hal::ReasonId *reason) noexcept {
  if (!impl_ || !impl_->coordinator) {
    SetReason(reason, hal::ReasonId::resource_exhausted);
    return hal::StatusId::resource_exhausted;
  }
  return impl_->coordinator->CommitOutput(completion, output_write_succeeded,
                                          reason);
}

void ProductionVideoExecutionComposition::Invalidate(
    hal::ReasonId reason) noexcept {
  if (impl_ && impl_->coordinator)
    impl_->coordinator->Invalidate(reason);
}

hal::StatusId ProductionVideoExecutionComposition::ResetTemporal(
    hal::ReasonId reason) noexcept {
  if (!impl_ || !impl_->coordinator)
    return hal::StatusId::resource_exhausted;
  return impl_->coordinator->ResetTemporal(reason);
}

void ProductionVideoExecutionComposition::Shutdown() noexcept {
  if (!impl_)
    return;
  if (impl_->coordinator)
    impl_->coordinator->Shutdown();
}

VideoExecutionCoordinatorSnapshot
ProductionVideoExecutionComposition::Snapshot() const noexcept {
  if (!impl_ || !impl_->coordinator)
    return {};
  return impl_->coordinator->Snapshot();
}

} // namespace studiocast::video
