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
  struct RegisteredRuntime {
    // The lease is declared before the registry and coordinator so reverse
    // destruction cannot release the owner while either still retains its
    // published adapter/factory pointers.
    ProductionVideoSetupLease lease;
    std::unique_ptr<hal::StaticAdapterRegistry> registry;
    std::unique_ptr<VideoExecutionCoordinator> coordinator;
    VideoExecutionCoordinatorSnapshot terminal_snapshot{};
    ProductionVideoSetupLeaseShutdownResult shutdown_result{};
    bool shutdown_requested = false;
    bool coordinator_destroyed = false;

    void DestroyCoordinator() noexcept {
      if (coordinator_destroyed)
        return;
      coordinator_destroyed = true;
      if (coordinator)
        coordinator->Shutdown();
      if (coordinator)
        terminal_snapshot = coordinator->Snapshot();
      coordinator.reset();
      registry.reset();
    }

    ProductionVideoSetupLeaseShutdownResult Shutdown() noexcept {
      if (shutdown_requested && !shutdown_result.pending)
        return shutdown_result;
      shutdown_requested = true;
      // Registry/coordinator raw pointers are retired before the first owner
      // attempt and remain retired for every pending retry.
      DestroyCoordinator();
      shutdown_result = lease.Shutdown();
      terminal_snapshot.stopped = !shutdown_result.pending;
      terminal_snapshot.prepared = false;
      terminal_snapshot.frame_pending = false;
      terminal_snapshot.readiness.status = shutdown_result.status;
      terminal_snapshot.readiness.reason = shutdown_result.reason;
      terminal_snapshot.readiness.available = false;
      terminal_snapshot.readiness.degraded = shutdown_result.pending;
      return shutdown_result;
    }

    ~RegisteredRuntime() {
      // The lease destructor performs at most one final bounded attempt.
      DestroyCoordinator();
    }
  };

  std::unique_ptr<vulkan::VulkanVideoExecutionFactory> vulkan_factory;
  // Declared after the default factory so it is destroyed first. Production
  // Vulkan executors hold a non-owning resident-adapter view into the factory.
  std::unique_ptr<RegisteredRuntime> runtime;
  bool injected = false;

  hal::StatusId BuildRuntime(const hal::AdapterRegistration &adapter,
                             const VideoAdapterFactoryRegistration &factory,
                             ProductionVideoSetupLease candidate_lease,
                             std::unique_ptr<RegisteredRuntime> *candidate,
                             hal::ReasonId *reason) noexcept {
    if (candidate == nullptr) {
      SetReason(reason, hal::ReasonId::malformed_contract);
      return hal::StatusId::blocked;
    }
    *candidate = nullptr;
    try {
      auto next = std::make_unique<RegisteredRuntime>();
      next->lease = std::move(candidate_lease);
      next->registry = std::make_unique<hal::StaticAdapterRegistry>();
      auto status = next->registry->Register(adapter, reason);
      if (!IsSetupSuccess(status))
        return status;
      next->coordinator =
          std::make_unique<VideoExecutionCoordinator>(*next->registry);
      status = next->coordinator->RegisterFactory(factory, reason);
      if (!IsSetupSuccess(status))
        return status;
      *candidate = std::move(next);
      return status;
    } catch (const std::bad_alloc &) {
      SetReason(reason, hal::ReasonId::resource_exhausted);
      return hal::StatusId::resource_exhausted;
    } catch (...) {
      SetReason(reason, hal::ReasonId::malformed_contract);
      return hal::StatusId::blocked;
    }
  }

  hal::StatusId
  BuildPassthroughRuntime(const hal::AdapterRegistration &registration,
                          std::unique_ptr<RegisteredRuntime> *candidate,
                          hal::ReasonId *reason) noexcept {
    if (candidate == nullptr) {
      SetReason(reason, hal::ReasonId::malformed_contract);
      return hal::StatusId::blocked;
    }
    *candidate = nullptr;
    try {
      auto next = std::make_unique<RegisteredRuntime>();
      next->registry = std::make_unique<hal::StaticAdapterRegistry>();
      const auto status = next->registry->Register(registration, reason);
      if (!IsSetupSuccess(status))
        return status;
      next->coordinator =
          std::make_unique<VideoExecutionCoordinator>(*next->registry);
      *candidate = std::move(next);
      return status;
    } catch (const std::bad_alloc &) {
      SetReason(reason, hal::ReasonId::resource_exhausted);
      return hal::StatusId::resource_exhausted;
    } catch (...) {
      SetReason(reason, hal::ReasonId::malformed_contract);
      return hal::StatusId::blocked;
    }
  }

  hal::StatusId Register(const hal::AdapterRegistration &adapter,
                         const VideoAdapterFactoryRegistration &factory,
                         hal::ReasonId *reason) noexcept {
    if (runtime) {
      SetReason(reason, hal::ReasonId::none);
      return hal::StatusId::ready;
    }
    std::unique_ptr<RegisteredRuntime> candidate;
    const auto status = BuildRuntime(adapter, factory, {}, &candidate, reason);
    if (!IsSetupSuccess(status))
      return status;
    runtime = std::move(candidate);
    return status;
  }

  bool EnsureProductionFactory(const VideoExecutionPlanRequest &request,
                               hal::ReasonId *reason) noexcept {
    if (runtime)
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
    if (!IsSetupSuccess(Register(vulkan_factory->adapter_registration(),
                                 vulkan_factory->factory_registration(),
                                 reason))) {
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

ProductionVideoExecutionComposition::~ProductionVideoExecutionComposition() =
    default;

hal::StatusId ProductionVideoExecutionComposition::Prepare(
    const VideoExecutionPlanRequest &request,
    VideoExecutionPrepareEvidence *evidence, hal::ReasonId *reason) noexcept {
  if (!impl_) {
    SetReason(reason, hal::ReasonId::resource_exhausted);
    return hal::StatusId::resource_exhausted;
  }
  if (impl_->runtime && impl_->runtime->shutdown_requested) {
    const auto shutdown = impl_->runtime->shutdown_result;
    SetReason(reason, shutdown.pending ? shutdown.reason
                                       : hal::ReasonId::shutdown_requested);
    if (evidence) {
      *evidence = {};
      evidence->status =
          shutdown.pending ? shutdown.status : hal::StatusId::stopped;
      evidence->reason = shutdown.pending ? shutdown.reason
                                          : hal::ReasonId::shutdown_requested;
    }
    return shutdown.pending ? shutdown.status : hal::StatusId::stopped;
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
  if (!impl_->runtime || !impl_->runtime->coordinator) {
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
  return impl_->runtime->coordinator->Prepare(exact, evidence, reason);
}

hal::StatusId ProductionVideoExecutionComposition::PrepareWithSetupLease(
    ProductionVideoSetupLease candidate_lease,
    const VideoExecutionPlanRequest &request,
    VideoExecutionPrepareEvidence *evidence, hal::ReasonId *reason) noexcept {
  if (evidence)
    *evidence = {};
  if (!impl_) {
    SetReason(reason, hal::ReasonId::resource_exhausted);
    return hal::StatusId::resource_exhausted;
  }
  if (impl_->runtime && impl_->runtime->shutdown_requested) {
    const auto shutdown = impl_->runtime->shutdown_result;
    SetReason(reason, shutdown.pending ? shutdown.reason
                                       : hal::ReasonId::shutdown_requested);
    if (evidence) {
      evidence->status =
          shutdown.pending ? shutdown.status : hal::StatusId::stopped;
      evidence->reason = shutdown.pending ? shutdown.reason
                                          : hal::ReasonId::shutdown_requested;
    }
    return shutdown.pending ? shutdown.status : hal::StatusId::stopped;
  }
  if (impl_->runtime && impl_->runtime->coordinator &&
      impl_->runtime->coordinator->Snapshot().frame_pending) {
    SetReason(reason, hal::ReasonId::output_not_committed);
    if (evidence) {
      evidence->status = hal::StatusId::blocked;
      evidence->reason = hal::ReasonId::output_not_committed;
      evidence->old_plan_retained = true;
    }
    return hal::StatusId::blocked;
  }
  if (!candidate_lease.live()) {
    const auto &snapshot = candidate_lease.snapshot();
    const bool pending =
        snapshot.state == ProductionVideoSetupLeaseState::shutdown_pending;
    const auto failure =
        pending ? snapshot.reason
                : (snapshot.state == ProductionVideoSetupLeaseState::stopped
                       ? hal::ReasonId::shutdown_requested
                       : hal::ReasonId::malformed_contract);
    SetReason(reason, failure);
    if (evidence) {
      evidence->status = pending ? snapshot.status
                                 : (failure == hal::ReasonId::shutdown_requested
                                        ? hal::StatusId::stopped
                                        : hal::StatusId::blocked);
      evidence->reason = failure;
      evidence->readiness = snapshot.readiness;
      evidence->old_plan_retained =
          impl_->runtime && impl_->runtime->coordinator &&
          impl_->runtime->coordinator->Snapshot().prepared;
    }
    return pending ? snapshot.status
                   : (failure == hal::ReasonId::shutdown_requested
                          ? hal::StatusId::stopped
                          : hal::StatusId::blocked);
  }

  const auto &snapshot = candidate_lease.snapshot();
  const auto adapter = snapshot.adapter;
  const auto factory = snapshot.factory;
  std::unique_ptr<Impl::RegisteredRuntime> candidate;
  hal::ReasonId registration_reason = hal::ReasonId::none;
  const auto registration_status =
      impl_->BuildRuntime(adapter, factory, std::move(candidate_lease),
                          &candidate, &registration_reason);
  SetReason(reason, registration_reason);
  if (!IsSetupSuccess(registration_status)) {
    if (evidence) {
      evidence->status = registration_status;
      evidence->reason = registration_reason;
      evidence->old_plan_retained =
          impl_->runtime && impl_->runtime->coordinator &&
          impl_->runtime->coordinator->Snapshot().prepared;
    }
    return registration_status;
  }

  const auto prepare_status =
      candidate->coordinator->Prepare(request, evidence, reason);
  if (!IsSetupSuccess(prepare_status)) {
    if (evidence)
      evidence->old_plan_retained =
          impl_->runtime && impl_->runtime->coordinator &&
          impl_->runtime->coordinator->Snapshot().prepared;
    return prepare_status;
  }

  impl_->runtime = std::move(candidate);
  // Any lazily created default factory was retained until its coordinator had
  // been destroyed by the runtime replacement above.
  impl_->vulkan_factory.reset();
  impl_->injected = true;
  return prepare_status;
}

hal::StatusId ProductionVideoExecutionComposition::PreparePassthrough(
    const hal::AdapterRegistration &registration,
    const VideoExecutionPlanRequest &request,
    VideoExecutionPrepareEvidence *evidence, hal::ReasonId *reason) noexcept {
  if (evidence)
    *evidence = {};
  if (!impl_) {
    SetReason(reason, hal::ReasonId::resource_exhausted);
    return hal::StatusId::resource_exhausted;
  }
  if (impl_->runtime && impl_->runtime->shutdown_requested) {
    SetReason(reason, hal::ReasonId::shutdown_requested);
    return hal::StatusId::stopped;
  }
  if (impl_->runtime && impl_->runtime->coordinator &&
      impl_->runtime->coordinator->Snapshot().frame_pending) {
    SetReason(reason, hal::ReasonId::output_not_committed);
    if (evidence) {
      evidence->status = hal::StatusId::blocked;
      evidence->reason = hal::ReasonId::output_not_committed;
      evidence->old_plan_retained = true;
    }
    return hal::StatusId::blocked;
  }

  std::unique_ptr<Impl::RegisteredRuntime> candidate;
  const auto registration_status =
      impl_->BuildPassthroughRuntime(registration, &candidate, reason);
  if (!IsSetupSuccess(registration_status))
    return registration_status;
  const auto status =
      candidate->coordinator->Prepare(request, evidence, reason);
  if (status != hal::StatusId::passthrough) {
    if (evidence)
      evidence->old_plan_retained =
          impl_->runtime && impl_->runtime->coordinator &&
          impl_->runtime->coordinator->Snapshot().prepared;
    return status;
  }
  impl_->runtime = std::move(candidate);
  impl_->vulkan_factory.reset();
  impl_->injected = true;
  return status;
}

VideoExecutionFrameResult ProductionVideoExecutionComposition::Execute(
    const hal::HostFrameBinding &binding) noexcept {
  if (!impl_ || !impl_->runtime || !impl_->runtime->coordinator) {
    VideoExecutionFrameResult result;
    const bool shutdown =
        impl_ && impl_->runtime && impl_->runtime->shutdown_requested;
    const bool pending = shutdown && impl_->runtime->shutdown_result.pending;
    result.status = pending ? impl_->runtime->shutdown_result.status
                            : (shutdown ? hal::StatusId::stopped
                                        : hal::StatusId::resource_exhausted);
    result.reason = pending ? impl_->runtime->shutdown_result.reason
                            : (shutdown ? hal::ReasonId::shutdown_requested
                                        : hal::ReasonId::resource_exhausted);
    return result;
  }
  return impl_->runtime->coordinator->Execute(binding);
}

hal::StatusId ProductionVideoExecutionComposition::CommitOutput(
    const hal::CompletionToken &completion, bool output_write_succeeded,
    hal::ReasonId *reason) noexcept {
  if (!impl_ || !impl_->runtime || !impl_->runtime->coordinator) {
    const bool shutdown =
        impl_ && impl_->runtime && impl_->runtime->shutdown_requested;
    const bool pending = shutdown && impl_->runtime->shutdown_result.pending;
    SetReason(reason, pending ? impl_->runtime->shutdown_result.reason
                              : (shutdown ? hal::ReasonId::shutdown_requested
                                          : hal::ReasonId::resource_exhausted));
    return pending ? impl_->runtime->shutdown_result.status
                   : (shutdown ? hal::StatusId::stopped
                               : hal::StatusId::resource_exhausted);
  }
  return impl_->runtime->coordinator->CommitOutput(
      completion, output_write_succeeded, reason);
}

void ProductionVideoExecutionComposition::Invalidate(
    hal::ReasonId reason) noexcept {
  if (impl_ && impl_->runtime && impl_->runtime->coordinator)
    impl_->runtime->coordinator->Invalidate(reason);
}

hal::StatusId ProductionVideoExecutionComposition::ResetTemporal(
    hal::ReasonId reason) noexcept {
  if (!impl_ || !impl_->runtime || !impl_->runtime->coordinator)
    return impl_ && impl_->runtime && impl_->runtime->shutdown_requested
               ? (impl_->runtime->shutdown_result.pending
                      ? impl_->runtime->shutdown_result.status
                      : hal::StatusId::stopped)
               : hal::StatusId::resource_exhausted;
  return impl_->runtime->coordinator->ResetTemporal(reason);
}

ProductionVideoSetupLeaseShutdownResult
ProductionVideoExecutionComposition::Shutdown() noexcept {
  if (!impl_ || !impl_->runtime)
    return {};
  return impl_->runtime->Shutdown();
}

ProductionVideoSetupLeaseShutdownResult
ProductionVideoExecutionComposition::ShutdownSnapshot() const noexcept {
  if (!impl_ || !impl_->runtime)
    return {};
  return impl_->runtime->shutdown_result;
}

VideoExecutionCoordinatorSnapshot
ProductionVideoExecutionComposition::Snapshot() const noexcept {
  if (!impl_ || !impl_->runtime)
    return {};
  if (impl_->runtime->coordinator)
    return impl_->runtime->coordinator->Snapshot();
  return impl_->runtime->terminal_snapshot;
}

} // namespace studiocast::video
