#include "core/open_video/open_cuda_production_hal_owner.h"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

namespace studiocast::open_cuda {
namespace {

using hal::ReasonId;
using hal::StatusId;
using OwnerStatus = OpenCudaProductionOwnerStatus;

void ResetResult(OpenCudaProductionPreparationResult *result) {
  if (result)
    *result = {};
}

void SetResult(OpenCudaProductionPreparationResult *result, OwnerStatus status,
               StatusId hal_status, ReasonId reason, const char *diagnostic) {
  if (!result)
    return;
  result->status = status;
  result->hal_status = hal_status;
  result->reason = reason;
  result->diagnostic = diagnostic ? diagnostic : "";
}

bool ExactPhysicalIdentity(
    const hal::PhysicalDeviceIdentity &domain,
    const cuda::CudaPhysicalDeviceIdentity &context) noexcept {
  return domain.stable_id == context.uuid;
}

bool ExactBootstrapDomain(const hal::ExecutionDomainId &domain,
                          const cuda::CudaContextIdentity &context,
                          std::uint64_t adapter_instance_id,
                          cuda::CUstream stream) noexcept {
  return stream != nullptr && context.Valid() &&
         hal::ValidateExecutionDomain(domain) &&
         domain.backend == hal::BackendId::cuda &&
         domain.api_family == hal::ApiFamily::cuda_driver &&
         ExactPhysicalIdentity(domain.physical_device,
                               context.physical_device) &&
         domain.logical_context_generation == context.logical_generation &&
         domain.allocator_family == hal::AllocatorFamily::cuda_driver &&
         domain.memory_handle_family ==
             hal::MemoryHandleFamily::cuda_device_pointer &&
         domain.queue_or_stream_identity ==
             reinterpret_cast<std::uintptr_t>(stream) &&
         domain.ownership_model == hal::OwnershipModel::adapter_exclusive &&
         domain.synchronization_protocol ==
             hal::SynchronizationProtocol::cuda_stream_ordered &&
         domain.adapter_instance_id == adapter_instance_id;
}

bool ExactPreparedCapability(const hal::PreparedProgram &program,
                             const hal::ReadinessSnapshot &readiness,
                             const hal::ExecutionDomainId &domain,
                             ReasonId *reason) noexcept {
  if (reason)
    *reason = ReasonId::none;
  if (readiness.status == StatusId::resource_exhausted) {
    if (reason)
      *reason = ReasonId::resource_exhausted;
    return false;
  }
  if (program.invalidation_key.generations.context !=
      domain.logical_context_generation) {
    if (reason)
      *reason = ReasonId::wrong_generation;
    return false;
  }
  if (!hal::ValidateExecutionDomain(program.execution_domain) ||
      !hal::ValidateExecutionDomain(program.requested_execution_domain)) {
    if (reason)
      *reason = ReasonId::malformed_execution_domain;
    return false;
  }
  if (program.owner_adapter_instance_id != domain.adapter_instance_id) {
    if (reason)
      *reason = ReasonId::foreign_adapter;
    return false;
  }
  if (!hal::ValidatePreparedProgram(program, reason))
    return false;
  if (program.execution_domain != domain ||
      program.requested_execution_domain != domain ||
      readiness.execution_domain != domain) {
    if (reason)
      *reason = ReasonId::wrong_execution_domain;
    return false;
  }
  if (program.requested_backend != hal::BackendId::cuda ||
      program.requested_provider !=
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda ||
      !program.strict_execution_domain ||
      readiness.requested_backend != hal::BackendId::cuda ||
      readiness.selected_backend != hal::BackendId::cuda ||
      readiness.provider !=
          hal::SemanticEffectProviderId::open_video_onnxruntime_cuda ||
      !readiness.available ||
      (readiness.status != StatusId::ready &&
       readiness.status != StatusId::degraded)) {
    if (reason)
      *reason = ReasonId::capability_evidence_incomplete;
    return false;
  }
  if (program.stage_count == 0 || program.stage_count > 2 ||
      readiness.ready_stage_mask !=
          ((std::uint32_t{1} << program.stage_count) - 1u) ||
      readiness.blocked_stage_mask != 0 || readiness.disabled_stage_mask != 0) {
    if (reason)
      *reason = ReasonId::capability_evidence_incomplete;
    return false;
  }
  const auto vb = [](hal::SemanticEffectId semantic) {
    return semantic == hal::SemanticEffectId::virtual_background_blur ||
           semantic == hal::SemanticEffectId::virtual_background_remove ||
           semantic == hal::SemanticEffectId::virtual_background_replace;
  };
  if (!((program.stage_count == 1 &&
         (vb(program.stages[0].semantic) ||
          program.stages[0].semantic ==
              hal::SemanticEffectId::virtual_key_light)) ||
        (program.stage_count == 2 && vb(program.stages[0].semantic) &&
         program.stages[1].semantic ==
             hal::SemanticEffectId::virtual_key_light))) {
    if (reason)
      *reason = ReasonId::semantic_effect_unsupported;
    return false;
  }
  for (std::size_t index = 0; index < program.stage_count; ++index) {
    const auto &stage = program.stages[index];
    if (stage.backend != hal::BackendId::cuda ||
        stage.execution_domain != domain ||
        stage.provider !=
            hal::SemanticEffectProviderId::open_video_onnxruntime_cuda) {
      if (reason)
        *reason = ReasonId::semantic_provider_unavailable;
      return false;
    }
  }
  return true;
}

OwnerStatus OwnerStatusForEvidenceReason(ReasonId reason) noexcept {
  switch (reason) {
  case ReasonId::resource_exhausted:
    return OwnerStatus::resource_exhausted;
  case ReasonId::wrong_execution_domain:
    return OwnerStatus::wrong_execution_domain;
  case ReasonId::wrong_generation:
  case ReasonId::reconfiguration_required:
    return OwnerStatus::stale_generation;
  case ReasonId::foreign_adapter:
    return OwnerStatus::foreign_adapter;
  default:
    return OwnerStatus::capability_evidence_incomplete;
  }
}

} // namespace

struct OpenCudaProductionHalOwner::Impl {
  std::unique_ptr<cuda::CudaDriverApi> driver;
  std::unique_ptr<OpenCudaPreparedResidentExecutor> executor;
  std::unique_ptr<ResidentAdapter> adapter;
  cuda::CudaContextIdentity context{};
  hal::ExecutionDomainId domain{};
  cuda::CUstream stream = nullptr;
  std::uint64_t adapter_instance_id = 0;
  OpenCudaPreparedCapabilitySnapshot capability{};
  OpenCudaProductionShutdownResult shutdown{};
  OwnerStatus status = OwnerStatus::invalid;
  bool raw_stream_owned = false;
  bool bootstrap_ready = false;
  bool quarantined = false;
  bool device_lost = false;
  bool stopped = false;

  OpenCudaProductionShutdownResult RequestShutdown() noexcept {
    ++shutdown.shutdown_attempts;
    shutdown.shutdown_requested = true;
    capability = {};
    bootstrap_ready = false;

    if (stopped) {
      shutdown.status = OwnerStatus::stopped;
      shutdown.hal_status = StatusId::stopped;
      shutdown.reason = ReasonId::shutdown_requested;
      shutdown.quiescent = true;
      shutdown.driver_context_release_permitted = true;
      return shutdown;
    }

    if (adapter)
      adapter->Shutdown();

    if (executor) {
      shutdown.executor = executor->RequestShutdown();
      if (!shutdown.executor.driver_context_release_permitted) {
        shutdown.status = OwnerStatus::shutdown_pending;
        shutdown.hal_status = StatusId::degraded;
        shutdown.reason = ReasonId::synchronization_failed;
        shutdown.quiescent = false;
        shutdown.driver_context_release_permitted = false;
        status = shutdown.status;
        return shutdown;
      }
      adapter.reset();
      executor.reset();
      stream = nullptr;
    } else if (raw_stream_owned && stream && driver) {
      ++shutdown.owner_stream_destroy_attempts;
      if (!driver->DestroyStream(stream, nullptr)) {
        shutdown.status = OwnerStatus::stream_destroy_failed;
        shutdown.hal_status = StatusId::degraded;
        shutdown.reason = ReasonId::synchronization_failed;
        shutdown.quiescent = false;
        shutdown.driver_context_release_permitted = false;
        status = shutdown.status;
        return shutdown;
      }
      ++shutdown.owner_stream_destructions;
      raw_stream_owned = false;
      stream = nullptr;
    }

    driver.reset();
    context = {};
    domain = {};
    stopped = true;
    status = OwnerStatus::stopped;
    shutdown.status = OwnerStatus::stopped;
    shutdown.hal_status = StatusId::stopped;
    shutdown.reason = ReasonId::shutdown_requested;
    shutdown.quiescent = true;
    shutdown.driver_context_release_permitted = true;
    return shutdown;
  }
};

OpenCudaProductionHalOwner::OpenCudaProductionHalOwner()
    : impl_(std::make_unique<Impl>()) {}

std::unique_ptr<OpenCudaProductionHalOwner>
OpenCudaProductionHalOwner::CreateBackend(
    const OpenCudaProductionBackendRequest &request,
    OpenCudaProductionPreparationResult *result) noexcept {
  ResetResult(result);
  std::unique_ptr<OpenCudaProductionHalOwner> owner;
  try {
    if (request.api_version != kOpenCudaProductionOwnerApiVersion ||
        request.adapter_instance_id == 0 || request.cuda_device_id < 0 ||
        request.configuration_generation == 0 ||
        ((request.model_artifact.Valid()) !=
         (request.model_preprocess_generation != 0 &&
          request.model_artifact_hash != 0)) ||
        (request.require_expected_context &&
         !request.expected_context.Valid())) {
      SetResult(result, OwnerStatus::malformed_request, StatusId::blocked,
                ReasonId::malformed_contract,
                "Open CUDA production owner request is malformed.");
      return nullptr;
    }

    owner = std::unique_ptr<OpenCudaProductionHalOwner>(
        new OpenCudaProductionHalOwner());
    auto &state = *owner->impl_;
    state.adapter_instance_id = request.adapter_instance_id;
    state.driver = std::make_unique<cuda::CudaDriverApi>();

    std::string error;
    const bool initialized =
        request.initialize_driver
            ? request.initialize_driver(request.initialize_driver_context,
                                        state.driver.get(), &error)
            : state.driver->Initialize(&error);
    if (!initialized) {
      SetResult(result, OwnerStatus::driver_initialize_failed,
                StatusId::unavailable, ReasonId::capability_evidence_incomplete,
                "CUDA driver initialization failed.");
      if (result && !error.empty())
        result->diagnostic = std::move(error);
      return nullptr;
    }
    if (!state.driver->EnsureContextForDevice(request.cuda_device_id, &error)) {
      SetResult(result, OwnerStatus::context_unavailable, StatusId::blocked,
                ReasonId::capability_evidence_incomplete,
                "CUDA context identity could not be established.");
      if (result && !error.empty())
        result->diagnostic = std::move(error);
      return nullptr;
    }
    const auto *identity = state.driver->context_identity();
    if (!identity || !identity->Valid()) {
      SetResult(result, OwnerStatus::context_unavailable, StatusId::blocked,
                ReasonId::capability_evidence_incomplete,
                "CUDA context identity is incomplete.");
      return nullptr;
    }
    state.context = *identity;

    if (!state.driver->CreateStream(&state.stream, &error) ||
        state.stream == nullptr) {
      SetResult(result, OwnerStatus::stream_create_failed, StatusId::blocked,
                ReasonId::capability_evidence_incomplete,
                "Exact non-default CUDA stream creation failed.");
      if (result && !error.empty())
        result->diagnostic = std::move(error);
      return nullptr;
    }
    state.raw_stream_owned = true;

    if (request.require_expected_context &&
        request.expected_context != state.context) {
      SetResult(result, OwnerStatus::context_mismatch, StatusId::blocked,
                ReasonId::wrong_generation,
                "CUDA context does not match setup-time selection evidence.");
      const auto cleanup = state.RequestShutdown();
      if (!cleanup.driver_context_release_permitted) {
        if (result) {
          result->retained_cleanup_island = true;
          result->status = cleanup.status;
          result->hal_status = cleanup.hal_status;
          result->reason = cleanup.reason;
        }
        return owner;
      }
      return nullptr;
    }

    if (state.driver->BindPreparedContext(state.context) !=
            cuda::CUDA_SUCCESS ||
        !state.driver->context_identity() ||
        *state.driver->context_identity() != state.context) {
      SetResult(result, OwnerStatus::context_mismatch, StatusId::blocked,
                ReasonId::wrong_generation,
                "CUDA context changed while preparing the resident island.");
      const auto cleanup = state.RequestShutdown();
      if (!cleanup.driver_context_release_permitted) {
        if (result) {
          result->retained_cleanup_island = true;
          result->status = cleanup.status;
          result->hal_status = cleanup.hal_status;
          result->reason = cleanup.reason;
        }
        return owner;
      }
      return nullptr;
    }

    state.domain.backend = hal::BackendId::cuda;
    state.domain.api_family = hal::ApiFamily::cuda_driver;
    state.domain.physical_device.vendor_id = 0x10de;
    state.domain.physical_device.stable_id = state.context.physical_device.uuid;
    state.domain.logical_context_generation = state.context.logical_generation;
    state.domain.allocator_family = hal::AllocatorFamily::cuda_driver;
    state.domain.memory_handle_family =
        hal::MemoryHandleFamily::cuda_device_pointer;
    state.domain.queue_or_stream_identity =
        reinterpret_cast<std::uintptr_t>(state.stream);
    state.domain.ownership_model = hal::OwnershipModel::adapter_exclusive;
    state.domain.synchronization_protocol =
        hal::SynchronizationProtocol::cuda_stream_ordered;
    state.domain.adapter_instance_id = request.adapter_instance_id;
    if (!ExactBootstrapDomain(state.domain, state.context,
                              request.adapter_instance_id, state.stream)) {
      SetResult(result, OwnerStatus::execution_domain_invalid,
                StatusId::blocked, ReasonId::malformed_execution_domain,
                "Exact Open CUDA execution domain validation failed.");
      const auto cleanup = state.RequestShutdown();
      if (!cleanup.driver_context_release_permitted) {
        if (result) {
          result->retained_cleanup_island = true;
          result->status = cleanup.status;
          result->hal_status = cleanup.hal_status;
          result->reason = cleanup.reason;
        }
        return owner;
      }
      return nullptr;
    }

    OpenCudaPreparedResidentExecutor::Options options;
    options.driver = state.driver.get();
    options.prepared_stream = state.stream;
    options.model_pack = request.model_pack;
    options.model_artifact = request.model_artifact;
    options.cuda_device_id = request.cuda_device_id;
    options.adapter_instance_id = request.adapter_instance_id;
    options.stream_identity = state.domain.queue_or_stream_identity;
    options.configuration_generation = request.configuration_generation;
    options.model_preprocess_generation = request.model_preprocess_generation;
    options.model_artifact_hash = request.model_artifact_hash;
    options.executor_owns_stream = true;
    options.strength = request.strength;
    options.remove_background_rgb = request.remove_background_rgb;
    options.replacement_rgb = request.replacement_rgb;
    options.replacement_width = request.replacement_width;
    options.replacement_height = request.replacement_height;
    options.key_light_intensity = request.key_light_intensity;
    options.key_light_temperature_preset = request.key_light_temperature_preset;
    options.key_light_direction_pan_degrees =
        request.key_light_direction_pan_degrees;
    options.vignette_intensity = request.vignette_intensity;
    options.wait_policy = request.wait_policy;
    options.runtime = request.runtime;
    state.executor =
        std::make_unique<OpenCudaPreparedResidentExecutor>(std::move(options));
    state.raw_stream_owned = false;
    state.adapter = std::make_unique<ResidentAdapter>(
        request.adapter_instance_id, *state.executor);
    state.bootstrap_ready = true;
    state.status = OwnerStatus::domain_ready;

    if (result) {
      result->status = OwnerStatus::domain_ready;
      result->hal_status = StatusId::ready;
      result->reason = ReasonId::none;
      result->context = state.context;
      result->execution_domain = state.domain;
      result->stream_identity = state.domain.queue_or_stream_identity;
      result->bootstrap_domain_ready = true;
      result->prepared_capability_committed = false;
      result->diagnostic.clear();
    }
    return owner;
  } catch (const std::bad_alloc &) {
    SetResult(result, OwnerStatus::allocation_failed,
              StatusId::resource_exhausted, ReasonId::resource_exhausted,
              "Open CUDA production owner allocation failed.");
  } catch (...) {
    SetResult(result, OwnerStatus::capability_evidence_incomplete,
              StatusId::blocked, ReasonId::capability_evidence_incomplete,
              "Open CUDA production owner setup failed.");
  }

  if (owner && owner->impl_) {
    const auto cleanup = owner->impl_->RequestShutdown();
    if (!cleanup.driver_context_release_permitted) {
      if (result) {
        result->retained_cleanup_island = true;
        result->status = cleanup.status;
        result->hal_status = cleanup.hal_status;
        result->reason = cleanup.reason;
      }
      return owner;
    }
  }
  return nullptr;
}

OpenCudaProductionHalOwner::~OpenCudaProductionHalOwner() {
  if (!impl_)
    return;
  const auto shutdown = impl_->RequestShutdown();
  if (!shutdown.driver_context_release_permitted) {
    // Releasing the driver/context or executor while DMA completion is
    // unproven is unsafe. The owner API requires retry before destruction;
    // at process teardown a deliberate retained island is safer than freeing
    // ORT/device/pinned storage under in-flight work.
    (void)impl_.release();
  }
}

const cuda::CudaContextIdentity &
OpenCudaProductionHalOwner::context_identity() const noexcept {
  return impl_->context;
}

const hal::ExecutionDomainId &
OpenCudaProductionHalOwner::execution_domain() const noexcept {
  return impl_->domain;
}

cuda::CUstream OpenCudaProductionHalOwner::prepared_stream() const noexcept {
  return impl_ ? impl_->stream : nullptr;
}

bool OpenCudaProductionHalOwner::bootstrap_domain_ready() const noexcept {
  return impl_ && impl_->bootstrap_ready && !impl_->stopped;
}

bool OpenCudaProductionHalOwner::prepared_capability_committed()
    const noexcept {
  return impl_ && impl_->capability.committed && !impl_->quarantined &&
         !impl_->stopped;
}

cuda::CudaDriverApi *OpenCudaProductionHalOwner::driver() noexcept {
  return impl_ ? impl_->driver.get() : nullptr;
}

const cuda::CudaDriverApi *OpenCudaProductionHalOwner::driver() const noexcept {
  return impl_ ? impl_->driver.get() : nullptr;
}

OpenCudaPreparedResidentExecutor *
OpenCudaProductionHalOwner::executor() noexcept {
  return impl_ ? impl_->executor.get() : nullptr;
}

const OpenCudaPreparedResidentExecutor *
OpenCudaProductionHalOwner::executor() const noexcept {
  return impl_ ? impl_->executor.get() : nullptr;
}

ResidentAdapter *OpenCudaProductionHalOwner::adapter() noexcept {
  return impl_ ? impl_->adapter.get() : nullptr;
}

const ResidentAdapter *OpenCudaProductionHalOwner::adapter() const noexcept {
  return impl_ ? impl_->adapter.get() : nullptr;
}

OpenCudaProductionPreparationResult
OpenCudaProductionHalOwner::CommitPreparedCapability(
    const hal::PreparedProgram &program,
    const hal::ReadinessSnapshot &readiness) noexcept {
  OpenCudaProductionPreparationResult result;
  if (!impl_ || impl_->stopped) {
    SetResult(&result, OwnerStatus::stopped, StatusId::stopped,
              ReasonId::shutdown_requested,
              "Open CUDA production owner is stopped.");
    return result;
  }
  if (!impl_->bootstrap_ready || !impl_->adapter || !impl_->executor ||
      impl_->quarantined) {
    SetResult(&result,
              impl_->quarantined ? OwnerStatus::quarantined
                                 : OwnerStatus::capability_evidence_incomplete,
              impl_->quarantined ? StatusId::quarantined : StatusId::blocked,
              impl_->device_lost ? ReasonId::device_lost
                                 : ReasonId::capability_evidence_incomplete,
              "Open CUDA bootstrap island is not eligible for capability.");
    return result;
  }

  ReasonId reason = ReasonId::none;
  if (!ExactPreparedCapability(program, readiness, impl_->domain, &reason)) {
    SetResult(&result, OwnerStatusForEvidenceReason(reason),
              reason == ReasonId::resource_exhausted
                  ? StatusId::resource_exhausted
                  : StatusId::blocked,
              reason,
              "Strict CUDA/ORT prepared capability evidence is "
              "incomplete or foreign.");
    result.context = impl_->context;
    result.execution_domain = impl_->domain;
    result.stream_identity = impl_->domain.queue_or_stream_identity;
    result.bootstrap_domain_ready = true;
    return result;
  }

  impl_->capability.committed = true;
  impl_->capability.program_id = program.program_id;
  impl_->capability.invalidation_key = program.invalidation_key;
  impl_->capability.execution_domain = program.execution_domain;
  impl_->capability.provider = readiness.provider;
  impl_->capability.ready_stage_mask = readiness.ready_stage_mask;
  impl_->status = OwnerStatus::prepared_capability_committed;

  result.status = OwnerStatus::prepared_capability_committed;
  result.hal_status = readiness.status;
  result.reason = readiness.reason;
  result.context = impl_->context;
  result.execution_domain = impl_->domain;
  result.stream_identity = impl_->domain.queue_or_stream_identity;
  result.bootstrap_domain_ready = true;
  result.prepared_capability_committed = true;
  return result;
}

void OpenCudaProductionHalOwner::ObserveRuntimeStatus(
    StatusId status, ReasonId reason) noexcept {
  if (!impl_ || impl_->stopped)
    return;
  if (status != StatusId::quarantined && reason != ReasonId::device_lost)
    return;
  impl_->quarantined = true;
  impl_->device_lost = reason == ReasonId::device_lost;
  impl_->capability = {};
  impl_->status = OwnerStatus::quarantined;
}

OpenCudaProductionShutdownResult
OpenCudaProductionHalOwner::RequestShutdown() noexcept {
  if (!impl_) {
    OpenCudaProductionShutdownResult result;
    result.status = OwnerStatus::stopped;
    result.hal_status = StatusId::stopped;
    result.reason = ReasonId::shutdown_requested;
    result.shutdown_requested = true;
    result.quiescent = true;
    result.driver_context_release_permitted = true;
    return result;
  }
  return impl_->RequestShutdown();
}

OpenCudaProductionShutdownResult
OpenCudaProductionHalOwner::ShutdownSnapshot() const noexcept {
  if (!impl_) {
    OpenCudaProductionShutdownResult result;
    result.status = OwnerStatus::stopped;
    result.hal_status = StatusId::stopped;
    result.reason = ReasonId::shutdown_requested;
    result.shutdown_requested = true;
    result.quiescent = true;
    result.driver_context_release_permitted = true;
    return result;
  }
  return impl_->shutdown;
}

OpenCudaProductionOwnerSnapshot
OpenCudaProductionHalOwner::Snapshot() const noexcept {
  OpenCudaProductionOwnerSnapshot result;
  if (!impl_) {
    result.status = OwnerStatus::stopped;
    result.stopped = true;
    result.shutdown = ShutdownSnapshot();
    return result;
  }
  result.status = impl_->status;
  result.context = impl_->context;
  result.execution_domain = impl_->domain;
  result.prepared_capability = impl_->capability;
  result.shutdown = impl_->shutdown;
  result.bootstrap_domain_ready = impl_->bootstrap_ready;
  result.quarantined = impl_->quarantined;
  result.device_lost = impl_->device_lost;
  result.stopped = impl_->stopped;
  return result;
}

} // namespace studiocast::open_cuda
