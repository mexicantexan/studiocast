#include "core/video/production_video_backend_manager.h"

#include "core/maxine/production_video_execution_bootstrap.h"
#include "core/open_video/production_video_backend_setup_source.h"
#include "core/vulkan/production_video_backend_setup_source.h"

#include <array>
#include <memory>
#include <new>
#include <utility>

namespace studiocast::video {
namespace {

enum class StaticSourceId : std::uint8_t {
  invalid = 0,
  maxine,
  open_cuda,
  vulkan,
};

ProductionVideoResidentSetupSourceResult
Failure(const ProductionVideoResidentSetup &request, hal::StatusId status,
        hal::ReasonId reason) noexcept {
  ProductionVideoResidentSetupSourceResult result;
  result.status = status;
  result.reason = reason;
  result.readiness.status = status;
  result.readiness.reason = reason;
  result.readiness.requested_backend =
      request.plan.intent.backend.mode ==
              hal::BackendRequestMode::explicit_backend
          ? request.plan.intent.backend.backend
          : hal::BackendId::invalid;
  result.setup_completed_outside_frame_loop = true;
  return result;
}

StaticSourceId SourceForBackend(hal::BackendId backend) noexcept {
  switch (backend) {
  case hal::BackendId::maxine:
    return StaticSourceId::maxine;
  case hal::BackendId::cuda:
    return StaticSourceId::open_cuda;
  case hal::BackendId::vulkan:
    return StaticSourceId::vulkan;
  case hal::BackendId::cpu:
  case hal::BackendId::open_video_onnxruntime:
  case hal::BackendId::test_fake:
  case hal::BackendId::invalid:
    return StaticSourceId::invalid;
  }
  return StaticSourceId::invalid;
}

StaticSourceId
SourceForProvider(hal::SemanticEffectProviderId provider) noexcept {
  switch (provider) {
  case hal::SemanticEffectProviderId::maxine_vfx:
  case hal::SemanticEffectProviderId::maxine_ar:
  case hal::SemanticEffectProviderId::maxine_cuda_utility:
    return StaticSourceId::maxine;
  case hal::SemanticEffectProviderId::open_video_onnxruntime_cuda:
    return StaticSourceId::open_cuda;
  case hal::SemanticEffectProviderId::open_vulkan_production_kernels:
    return StaticSourceId::vulkan;
  case hal::SemanticEffectProviderId::open_video_onnxruntime_cpu:
  case hal::SemanticEffectProviderId::invalid:
  case hal::SemanticEffectProviderId::test_fake:
    return StaticSourceId::invalid;
  }
  return StaticSourceId::invalid;
}

class StaticProductionVideoBackendSetupSource final
    : public IProductionVideoBackendSetupSource {
public:
  StaticProductionVideoBackendSetupSource() = default;
  explicit StaticProductionVideoBackendSetupSource(
      StaticProductionVideoBackendSetupSourcesForTesting sources)
      : maxine_(std::move(sources.maxine)),
        open_cuda_(std::move(sources.open_cuda)),
        vulkan_(std::move(sources.vulkan)), injected_(true) {}
  ProductionVideoResidentSetupSourceResult AcquireResidentSetup(
      const ProductionVideoResidentSetup &request) noexcept override {
    const auto &intent = request.plan.intent;
    if (request.contract_version !=
            kProductionVideoBackendManagerContractVersion ||
        intent.stage_count == 0) {
      return Failure(request, hal::StatusId::blocked,
                     hal::ReasonId::malformed_contract);
    }
    if (intent.preferences.compute_family ==
        hal::ComputeFamilyAllowance::cpu_only) {
      return Failure(request, hal::StatusId::blocked,
                     hal::ReasonId::semantic_effect_unsupported);
    }

    std::array<StaticSourceId, 3> candidates{};
    std::size_t candidate_count = 0;
    const auto append = [&](StaticSourceId source) {
      if (source == StaticSourceId::invalid)
        return;
      for (std::size_t index = 0; index < candidate_count; ++index) {
        if (candidates[index] == source)
          return;
      }
      candidates[candidate_count++] = source;
    };

    const auto exact_provider =
        SourceForProvider(intent.requested_provider);
    if (intent.strict_provider_selection) {
      append(exact_provider);
    } else if (intent.backend.mode ==
               hal::BackendRequestMode::explicit_backend) {
      append(SourceForBackend(intent.backend.backend));
    } else if (intent.preferences.strict_effects_engine) {
      append(intent.preferences.effects_engine ==
                     hal::EffectsEnginePreferenceConstraint::maxine
                 ? StaticSourceId::maxine
                 : (intent.preferences.effects_engine ==
                            hal::EffectsEnginePreferenceConstraint::open_cuda
                        ? StaticSourceId::open_cuda
                        : StaticSourceId::invalid));
    } else {
      append(exact_provider);
      if (intent.preferences.effects_engine ==
          hal::EffectsEnginePreferenceConstraint::maxine) {
        append(StaticSourceId::maxine);
      } else if (intent.preferences.effects_engine ==
                 hal::EffectsEnginePreferenceConstraint::open_cuda) {
        append(StaticSourceId::open_cuda);
      }
      switch (intent.preferences.compute_family) {
      case hal::ComputeFamilyAllowance::cuda_family:
        append(StaticSourceId::maxine);
        append(StaticSourceId::open_cuda);
        break;
      case hal::ComputeFamilyAllowance::vulkan_only:
        append(StaticSourceId::vulkan);
        break;
      case hal::ComputeFamilyAllowance::any:
        append(StaticSourceId::maxine);
        append(StaticSourceId::open_cuda);
        append(StaticSourceId::vulkan);
        break;
      case hal::ComputeFamilyAllowance::cpu_only:
        break;
      }
    }

    if (candidate_count == 0) {
      return Failure(
          request, hal::StatusId::unavailable,
          intent.backend.strict || intent.strict_provider_selection ||
                  intent.preferences.strict_effects_engine
              ? hal::ReasonId::strict_backend_unavailable
              : hal::ReasonId::semantic_effect_unsupported);
    }

    ProductionVideoResidentSetupSourceResult first_failure;
    bool have_failure = false;
    for (std::size_t index = 0; index < candidate_count; ++index) {
      auto *source = Source(candidates[index]);
      if (source == nullptr) {
        auto failed =
            Failure(request, hal::StatusId::resource_exhausted,
                    hal::ReasonId::resource_exhausted);
        if (!have_failure) {
          first_failure = std::move(failed);
          have_failure = true;
        }
        continue;
      }
      auto result = source->AcquireResidentSetup(request);
      if (result.status == hal::StatusId::ready ||
          result.status == hal::StatusId::degraded) {
        return result;
      }
      if (!have_failure) {
        first_failure = std::move(result);
        have_failure = true;
      }
      const bool exact =
          intent.backend.strict || intent.strict_provider_selection ||
          intent.preferences.strict_effects_engine;
      if (exact)
        break;
    }
    return have_failure
               ? std::move(first_failure)
               : Failure(request, hal::StatusId::unavailable,
                         hal::ReasonId::strict_backend_unavailable);
  }

private:
  IProductionVideoBackendSetupSource *Source(StaticSourceId id) noexcept {
    if (injected_) {
      switch (id) {
      case StaticSourceId::maxine:
        return maxine_.get();
      case StaticSourceId::open_cuda:
        return open_cuda_.get();
      case StaticSourceId::vulkan:
        return vulkan_.get();
      case StaticSourceId::invalid:
        return nullptr;
      }
    }
    auto create_if_missing =
        [](std::unique_ptr<IProductionVideoBackendSetupSource> *source,
           auto &&factory) -> IProductionVideoBackendSetupSource * {
      if (!*source)
        *source = factory();
      return source->get();
    };
    switch (id) {
    case StaticSourceId::maxine:
      return create_if_missing(&maxine_, [] {
        return maxine::CreateMaxineProductionVideoBackendSetupSource();
      });
    case StaticSourceId::open_cuda:
      return create_if_missing(&open_cuda_, [] {
        return open_cuda::CreateProductionVideoBackendSetupSource();
      });
    case StaticSourceId::vulkan:
      return create_if_missing(&vulkan_, [] {
        return vulkan::CreateProductionVideoBackendSetupSource();
      });
    case StaticSourceId::invalid:
      return nullptr;
    }
    return nullptr;
  }

  std::unique_ptr<IProductionVideoBackendSetupSource> maxine_;
  std::unique_ptr<IProductionVideoBackendSetupSource> open_cuda_;
  std::unique_ptr<IProductionVideoBackendSetupSource> vulkan_;
  bool injected_ = false;
};

} // namespace

std::unique_ptr<IProductionVideoBackendSetupSource>
CreateStaticProductionVideoBackendSetupSource() noexcept {
  try {
    return std::make_unique<StaticProductionVideoBackendSetupSource>();
  } catch (const std::bad_alloc &) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

std::unique_ptr<IProductionVideoBackendSetupSource>
CreateStaticProductionVideoBackendSetupSourceForTesting(
    StaticProductionVideoBackendSetupSourcesForTesting sources) noexcept {
  try {
    return std::make_unique<StaticProductionVideoBackendSetupSource>(
        std::move(sources));
  } catch (...) {
    return nullptr;
  }
}

} // namespace studiocast::video
