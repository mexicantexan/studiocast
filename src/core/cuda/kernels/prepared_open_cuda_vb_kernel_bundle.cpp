#include "core/cuda/kernels/prepared_open_cuda_vb_kernel_bundle.h"

namespace studiocast::cuda::kernels {

namespace {

constexpr CUresult kCudaErrorDeinitialized = 4;
constexpr CUresult kCudaErrorDeviceUnavailable = 46;
constexpr CUresult kCudaErrorContextIsDestroyed = 709;

bool IsDeviceLoss(CUresult result) noexcept {
  return result == kCudaErrorDeinitialized ||
         result == kCudaErrorDeviceUnavailable ||
         result == kCudaErrorContextIsDestroyed;
}

const char *StatusMessage(PreparedOpenCudaVbKernelStatus status) noexcept {
  switch (status) {
  case PreparedOpenCudaVbKernelStatus::invalid_argument:
    return "prepared Open CUDA VB kernel launch is malformed";
  case PreparedOpenCudaVbKernelStatus::resource_exhausted:
    return "prepared Open CUDA VB kernel resource budget is exhausted";
  case PreparedOpenCudaVbKernelStatus::unavailable:
    return "prepared Open CUDA VB kernel bundle is unavailable";
  case PreparedOpenCudaVbKernelStatus::module_load_failed:
    return "prepared Open CUDA VB kernel module failed to load";
  case PreparedOpenCudaVbKernelStatus::function_resolution_failed:
    return "prepared Open CUDA VB kernel function resolution failed";
  case PreparedOpenCudaVbKernelStatus::foreign_context:
    return "prepared Open CUDA VB kernel context is foreign";
  case PreparedOpenCudaVbKernelStatus::stale_context_generation:
    return "prepared Open CUDA VB kernel context generation is stale";
  case PreparedOpenCudaVbKernelStatus::foreign_stream:
    return "prepared Open CUDA VB kernel stream is foreign";
  case PreparedOpenCudaVbKernelStatus::reconfigured:
    return "prepared Open CUDA VB kernel configuration was invalidated";
  case PreparedOpenCudaVbKernelStatus::device_lost:
    return "prepared Open CUDA VB kernel device is lost";
  case PreparedOpenCudaVbKernelStatus::launch_failed:
    return "prepared Open CUDA VB kernel launch failed";
  case PreparedOpenCudaVbKernelStatus::shutdown:
    return "prepared Open CUDA VB kernel bundle is shut down";
  case PreparedOpenCudaVbKernelStatus::invalid:
  case PreparedOpenCudaVbKernelStatus::ready:
  case PreparedOpenCudaVbKernelStatus::launch_enqueued:
    break;
  }
  return "prepared Open CUDA VB kernel status is invalid";
}

} // namespace

bool PreparedOpenCudaVbKernelBundle::Prepare(
    const PreparedOpenCudaVbBinding &binding, std::size_t resource_budget_bytes,
    std::string *error_out) {
  if (error_out)
    error_out->clear();

  binding_ = {};
  handles_ = {};
  evidence_ = {};
  evidence_.resource_budget_bytes = resource_budget_bytes;
  evidence_.required_resource_bytes = RequiredResourceBytes();
  prepared_ = false;
  stopped_ = false;

  if (resource_budget_bytes < RequiredResourceBytes()) {
    evidence_.status = PreparedOpenCudaVbKernelStatus::resource_exhausted;
    if (error_out)
      *error_out = StatusMessage(evidence_.status);
    return false;
  }
  if (!binding.driver || !binding.driver->IsInitialized() ||
      !binding.context_identity.Valid() || !binding.stream ||
      binding.stream_identity == 0 || binding.configuration_generation == 0) {
    evidence_.status = PreparedOpenCudaVbKernelStatus::invalid_argument;
    if (error_out)
      *error_out = StatusMessage(evidence_.status);
    return false;
  }

  const auto *current = binding.driver->context_identity();
  if (!current || !current->Valid()) {
    evidence_.status = PreparedOpenCudaVbKernelStatus::unavailable;
    if (error_out)
      *error_out = StatusMessage(evidence_.status);
    return false;
  }
  if (current->physical_device != binding.context_identity.physical_device ||
      current->retained_context_id !=
          binding.context_identity.retained_context_id) {
    evidence_.status = PreparedOpenCudaVbKernelStatus::foreign_context;
    if (error_out)
      *error_out = StatusMessage(evidence_.status);
    return false;
  }
  if (current->logical_generation !=
      binding.context_identity.logical_generation) {
    evidence_.status = PreparedOpenCudaVbKernelStatus::stale_context_generation;
    if (error_out)
      *error_out = StatusMessage(evidence_.status);
    return false;
  }

  PreparedOpenCudaVbKernelStatus setup_status =
      PreparedOpenCudaVbKernelStatus::invalid;
  if (!PrepareOpenCudaVbKernelHandles(binding.driver, binding.context_identity,
                                      &handles_, &setup_status, error_out)) {
    evidence_.status = setup_status;
    return false;
  }

  binding_ = binding;
  evidence_.status = PreparedOpenCudaVbKernelStatus::ready;
  evidence_.context_identity = binding.context_identity;
  evidence_.stream_identity = binding.stream_identity;
  evidence_.configuration_generation = binding.configuration_generation;
  prepared_ = true;
  return true;
}

PreparedOpenCudaVbKernelStatus PreparedOpenCudaVbKernelBundle::ValidateBinding(
    const PreparedOpenCudaVbBinding &binding) const noexcept {
  if (stopped_)
    return PreparedOpenCudaVbKernelStatus::shutdown;
  if (evidence_.quarantined)
    return PreparedOpenCudaVbKernelStatus::device_lost;
  if (!prepared_)
    return evidence_.status == PreparedOpenCudaVbKernelStatus::reconfigured
               ? PreparedOpenCudaVbKernelStatus::reconfigured
               : PreparedOpenCudaVbKernelStatus::unavailable;
  if (!binding.driver || binding.driver != binding_.driver)
    return PreparedOpenCudaVbKernelStatus::foreign_context;

  const auto *current = binding.driver->context_identity();
  if (!current || !current->Valid())
    return PreparedOpenCudaVbKernelStatus::stale_context_generation;
  if (current->physical_device != binding_.context_identity.physical_device ||
      current->retained_context_id !=
          binding_.context_identity.retained_context_id ||
      binding.context_identity.physical_device !=
          binding_.context_identity.physical_device ||
      binding.context_identity.retained_context_id !=
          binding_.context_identity.retained_context_id) {
    return PreparedOpenCudaVbKernelStatus::foreign_context;
  }
  if (current->logical_generation !=
          binding_.context_identity.logical_generation ||
      binding.context_identity.logical_generation !=
          binding_.context_identity.logical_generation) {
    return PreparedOpenCudaVbKernelStatus::stale_context_generation;
  }
  if (binding.stream != binding_.stream ||
      binding.stream_identity != binding_.stream_identity) {
    return PreparedOpenCudaVbKernelStatus::foreign_stream;
  }
  if (binding.configuration_generation != binding_.configuration_generation) {
    return PreparedOpenCudaVbKernelStatus::reconfigured;
  }
  return PreparedOpenCudaVbKernelStatus::ready;
}

void PreparedOpenCudaVbKernelBundle::SetStatus(
    PreparedOpenCudaVbKernelStatus status,
    PreparedOpenCudaVbKernelStatus *status_out) noexcept {
  evidence_.status = status;
  if (status_out)
    *status_out = status;
}

bool PreparedOpenCudaVbKernelBundle::Launch(
    const PreparedOpenCudaVbBinding &binding,
    const PreparedOpenCudaVbLaunch &launch,
    PreparedOpenCudaVbKernelStatus *status_out) noexcept {
  if (status_out)
    *status_out = PreparedOpenCudaVbKernelStatus::invalid;

  const auto binding_status = ValidateBinding(binding);
  if (binding_status != PreparedOpenCudaVbKernelStatus::ready) {
    SetStatus(binding_status, status_out);
    return false;
  }

  CUcontext current_context = nullptr;
  const CUresult context_result =
      binding.driver->f().cuCtxGetCurrent(&current_context);
  if (IsDeviceLoss(context_result)) {
    evidence_.quarantined = true;
    prepared_ = false;
    SetStatus(PreparedOpenCudaVbKernelStatus::device_lost, status_out);
    return false;
  }
  if (context_result != CUDA_SUCCESS || !current_context ||
      current_context != handles_.context) {
    SetStatus(PreparedOpenCudaVbKernelStatus::foreign_context, status_out);
    return false;
  }

  const auto kernel_index = static_cast<std::size_t>(launch.kernel);
  if (kernel_index >= handles_.functions.size() ||
      !handles_.functions[kernel_index] || launch.grid_x == 0 ||
      launch.grid_y == 0 || launch.grid_z == 0 || launch.block_x == 0 ||
      launch.block_y == 0 || launch.block_z == 0 || !launch.parameters) {
    SetStatus(PreparedOpenCudaVbKernelStatus::invalid_argument, status_out);
    return false;
  }

  const CUresult result = binding.driver->f().cuLaunchKernel(
      handles_.functions[kernel_index], launch.grid_x, launch.grid_y,
      launch.grid_z, launch.block_x, launch.block_y, launch.block_z,
      launch.shared_memory_bytes, binding.stream, launch.parameters, nullptr);
  if (result != CUDA_SUCCESS) {
    if (IsDeviceLoss(result)) {
      evidence_.quarantined = true;
      prepared_ = false;
      SetStatus(PreparedOpenCudaVbKernelStatus::device_lost, status_out);
    } else {
      SetStatus(PreparedOpenCudaVbKernelStatus::launch_failed, status_out);
    }
    return false;
  }

  ++evidence_.successful_launches;
  SetStatus(PreparedOpenCudaVbKernelStatus::launch_enqueued, status_out);
  return true;
}

void PreparedOpenCudaVbKernelBundle::InvalidateForReconfiguration(
    std::uint64_t new_configuration_generation) noexcept {
  if (stopped_)
    return;
  if (new_configuration_generation == binding_.configuration_generation)
    return;
  prepared_ = false;
  handles_ = {};
  evidence_.status = PreparedOpenCudaVbKernelStatus::reconfigured;
  evidence_.configuration_generation = new_configuration_generation;
}

void PreparedOpenCudaVbKernelBundle::Shutdown() noexcept {
  prepared_ = false;
  stopped_ = true;
  binding_ = {};
  handles_ = {};
  evidence_.status = PreparedOpenCudaVbKernelStatus::shutdown;
}

bool PreparedOpenCudaVbKernelBundle::ready() const noexcept {
  return prepared_ && !stopped_ && !evidence_.quarantined;
}

PreparedOpenCudaVbKernelEvidence
PreparedOpenCudaVbKernelBundle::evidence() const noexcept {
  return evidence_;
}

} // namespace studiocast::cuda::kernels
