#include "core/vulkan/production_video_backend_setup_source.h"

#include "core/vulkan/video_execution_factory.h"

#include <memory>
#include <new>
#include <utility>

namespace studiocast::vulkan {
namespace {

constexpr auto kProvider =
    hal::SemanticEffectProviderId::open_vulkan_production_kernels;

bool IsSetupSuccess(hal::StatusId status) noexcept {
  return status == hal::StatusId::ready || status == hal::StatusId::degraded;
}

video::ProductionVideoResidentSetupSourceResult
Failure(hal::StatusId status, hal::ReasonId reason) noexcept {
  video::ProductionVideoResidentSetupSourceResult result;
  result.status = status;
  result.reason = reason;
  result.readiness.status = status;
  result.readiness.reason = reason;
  result.readiness.requested_backend = hal::BackendId::vulkan;
  result.readiness.selected_backend = hal::BackendId::invalid;
  result.readiness.provider = kProvider;
  result.setup_completed_outside_frame_loop = true;
  return result;
}

class VulkanSetupLeaseOwner final
    : public video::IProductionVideoSetupLeaseOwner {
public:
  explicit VulkanSetupLeaseOwner(
      std::unique_ptr<VulkanVideoExecutionFactory> factory) noexcept
      : factory_(std::move(factory)) {}

  hal::IResidentAdapter *resident_adapter() const noexcept override {
    return factory_ ? factory_->adapter_registration().resident_adapter
                    : nullptr;
  }

  video::IVideoAdapterFactory *
  video_adapter_factory() const noexcept override {
    return factory_.get();
  }

  void Shutdown() noexcept override {
    // V1 execution is synchronous. The composition destroys its coordinator
    // and prepared program before asking the lease to shut down, so no work is
    // in flight here. Factory destruction remains ordered after this callback
    // and releases the Vulkan adapter, queue, kernels, and device context.
  }

private:
  std::unique_ptr<VulkanVideoExecutionFactory> factory_;
};

class ProductionSetupSource final
    : public video::IProductionVideoBackendSetupSource {
public:
  video::ProductionVideoResidentSetupSourceResult AcquireResidentSetup(
      const video::ProductionVideoResidentSetup &request) noexcept override {
    const auto &caller = request.plan;
    if (request.contract_version !=
            video::kProductionVideoBackendManagerContractVersion ||
        caller.intent.backend.mode !=
            hal::BackendRequestMode::explicit_backend ||
        caller.intent.backend.backend != hal::BackendId::vulkan ||
        !caller.intent.backend.strict ||
        caller.intent.requested_provider != kProvider ||
        !caller.intent.strict_provider_selection ||
        caller.intent.stage_count == 0) {
      return Failure(hal::StatusId::blocked,
                     hal::ReasonId::strict_backend_unavailable);
    }
    for (std::size_t index = 0; index < caller.intent.stage_count; ++index) {
      const auto semantic = caller.intent.stages[index].intent.semantic;
      if (semantic != hal::SemanticEffectId::mirror &&
          semantic != hal::SemanticEffectId::vignette) {
        return Failure(hal::StatusId::blocked,
                       hal::ReasonId::semantic_effect_unsupported);
      }
    }

    try {
      auto factory = std::make_unique<VulkanVideoExecutionFactory>();
      hal::ReasonId reason = hal::ReasonId::none;
      const auto initialized = factory->Initialize(&reason);
      if (!IsSetupSuccess(initialized))
        return Failure(initialized, reason);

      const auto adapter = factory->adapter_registration();
      const auto factory_registration = factory->factory_registration();
      if (adapter.resident_adapter == nullptr ||
          factory_registration.factory == nullptr) {
        return Failure(hal::StatusId::blocked,
                       hal::ReasonId::capability_evidence_incomplete);
      }

      video::ProductionVideoResidentSetupSourceResult result;
      result.execution_kind = video::ProductionVideoResidentSetupSourceResult::
          ExecutionKind::resident_island;
      result.canonical_effects = request.effects;
      result.setup_completed_outside_frame_loop = true;
      result.plan = caller;
      auto &generations = result.plan.invalidation_key.generations;
      if (generations.model == 0)
        generations.model = 1;
      if (generations.provider == 0)
        generations.provider = 1;
      if (generations.device == 0)
        generations.device = 1;
      generations.context =
          adapter.execution_domain.logical_context_generation;
      result.plan.invalidation_key.execution_domain_hash =
          video::ProductionVideoExecutionDomainHash(
              adapter.execution_domain);

      video::ProductionVideoSetupLeaseSnapshot snapshot;
      snapshot.state = video::ProductionVideoSetupLeaseState::live;
      snapshot.status = initialized;
      snapshot.reason = reason;
      snapshot.backend = hal::BackendId::vulkan;
      snapshot.execution_domain = adapter.execution_domain;
      snapshot.generations = generations;
      snapshot.reserved_budget = result.plan.budget;
      snapshot.providers[snapshot.provider_count++] = kProvider;
      snapshot.adapter = adapter;
      snapshot.factory = factory_registration;
      snapshot.readiness.status = initialized;
      snapshot.readiness.reason = reason;
      snapshot.readiness.requested_backend = hal::BackendId::vulkan;
      snapshot.readiness.selected_backend = hal::BackendId::vulkan;
      snapshot.readiness.provider = kProvider;
      snapshot.readiness.execution_domain = adapter.execution_domain;
      snapshot.readiness.available = true;
      snapshot.readiness.degraded =
          initialized == hal::StatusId::degraded;

      auto owner =
          std::make_unique<VulkanSetupLeaseOwner>(std::move(factory));
      result.status = video::ProductionVideoSetupLease::Acquire(
          snapshot, std::move(owner), &result.lease,
          &result.lease_acquire, &result.reason);
      result.readiness = result.lease_acquire.readiness;
      if (!IsSetupSuccess(result.status) || !result.lease.live())
        return result;
      result.reason = result.lease.snapshot().reason;
      result.readiness = result.lease.snapshot().readiness;
      return result;
    } catch (const std::bad_alloc &) {
      return Failure(hal::StatusId::resource_exhausted,
                     hal::ReasonId::resource_exhausted);
    } catch (...) {
      return Failure(hal::StatusId::blocked,
                     hal::ReasonId::capability_evidence_incomplete);
    }
  }
};

} // namespace

std::unique_ptr<video::IProductionVideoBackendSetupSource>
CreateProductionVideoBackendSetupSource() noexcept {
  try {
    return std::make_unique<ProductionSetupSource>();
  } catch (...) {
    return nullptr;
  }
}

} // namespace studiocast::vulkan
