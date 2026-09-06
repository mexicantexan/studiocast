#include "core/cuda/kernels/prepared_preprocess_kernel_bundle.h"

#include "core/cuda/kernels/ptx_pitch_contract.h"

#include <cmath>
#include <limits>

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

bool SameSpec(const ModelPreprocessSpec &left,
              const ModelPreprocessSpec &right) noexcept {
  if (left.dst_w != right.dst_w || left.dst_h != right.dst_h ||
      left.dst_order != right.dst_order) {
    return false;
  }
  for (int channel = 0; channel < 3; ++channel) {
    if (left.mean[channel] != right.mean[channel] ||
        left.std[channel] != right.std[channel]) {
      return false;
    }
  }
  return true;
}

bool ValidateSpec(const ModelPreprocessSpec &spec) noexcept {
  if (spec.dst_w <= 0 || spec.dst_h <= 0)
    return false;
  for (int channel = 0; channel < 3; ++channel) {
    if (!std::isfinite(spec.mean[channel]) ||
        !std::isfinite(spec.std[channel]) || spec.std[channel] == 0.0f) {
      return false;
    }
  }
  return true;
}

const char *StatusMessage(PreparedPreprocessKernelStatus status) noexcept {
  switch (status) {
  case PreparedPreprocessKernelStatus::invalid_argument:
    return "prepared CUDA preprocess setup is invalid";
  case PreparedPreprocessKernelStatus::resource_exhausted:
    return "prepared CUDA preprocess resource budget is exhausted";
  case PreparedPreprocessKernelStatus::unavailable:
    return "prepared CUDA preprocess kernel is unavailable";
  case PreparedPreprocessKernelStatus::module_load_failed:
    return "prepared CUDA preprocess module failed to load";
  case PreparedPreprocessKernelStatus::function_resolution_failed:
    return "prepared CUDA preprocess function resolution failed";
  case PreparedPreprocessKernelStatus::foreign_context:
    return "prepared CUDA preprocess context is foreign";
  case PreparedPreprocessKernelStatus::stale_context_generation:
    return "prepared CUDA preprocess context generation is stale";
  case PreparedPreprocessKernelStatus::foreign_stream:
    return "prepared CUDA preprocess stream is foreign";
  case PreparedPreprocessKernelStatus::foreign_preprocess_spec:
    return "prepared CUDA preprocess specification is foreign";
  case PreparedPreprocessKernelStatus::reconfigured:
    return "prepared CUDA preprocess configuration was invalidated";
  case PreparedPreprocessKernelStatus::device_lost:
    return "prepared CUDA preprocess device is lost";
  case PreparedPreprocessKernelStatus::launch_failed:
    return "prepared CUDA preprocess launch failed";
  case PreparedPreprocessKernelStatus::shutdown:
    return "prepared CUDA preprocess bundle is shut down";
  case PreparedPreprocessKernelStatus::invalid:
  case PreparedPreprocessKernelStatus::ready:
  case PreparedPreprocessKernelStatus::launch_enqueued:
    break;
  }
  return "prepared CUDA preprocess status is invalid";
}

} // namespace

bool PreparedPreprocessKernelBundle::Prepare(
    const PreparedPreprocessBinding &binding, const ModelPreprocessSpec &spec,
    std::size_t resource_budget_bytes, std::string *error_out) {
  if (error_out)
    error_out->clear();

  binding_ = {};
  spec_ = {};
  handles_ = {};
  evidence_ = {};
  evidence_.resource_budget_bytes = resource_budget_bytes;
  evidence_.required_resource_bytes = RequiredResourceBytes();
  prepared_ = false;
  stopped_ = false;

  if (resource_budget_bytes < RequiredResourceBytes()) {
    evidence_.status = PreparedPreprocessKernelStatus::resource_exhausted;
    if (error_out)
      *error_out = StatusMessage(evidence_.status);
    return false;
  }
  if (!binding.driver || !binding.driver->IsInitialized() ||
      !binding.context_identity.Valid() || !binding.stream ||
      binding.stream_identity == 0 || binding.configuration_generation == 0 ||
      binding.model_preprocess_generation == 0 || !ValidateSpec(spec)) {
    evidence_.status = PreparedPreprocessKernelStatus::invalid_argument;
    if (error_out)
      *error_out = StatusMessage(evidence_.status);
    return false;
  }

  const auto *current = binding.driver->context_identity();
  if (!current || !current->Valid()) {
    evidence_.status = PreparedPreprocessKernelStatus::unavailable;
    if (error_out)
      *error_out = StatusMessage(evidence_.status);
    return false;
  }
  if (current->physical_device != binding.context_identity.physical_device ||
      current->retained_context_id !=
          binding.context_identity.retained_context_id) {
    evidence_.status = PreparedPreprocessKernelStatus::foreign_context;
    if (error_out)
      *error_out = StatusMessage(evidence_.status);
    return false;
  }
  if (current->logical_generation !=
      binding.context_identity.logical_generation) {
    evidence_.status = PreparedPreprocessKernelStatus::stale_context_generation;
    if (error_out)
      *error_out = StatusMessage(evidence_.status);
    return false;
  }

  PreparedPreprocessKernelStatus setup_status =
      PreparedPreprocessKernelStatus::invalid;
  if (!PreparePreprocessKernelHandles(binding.driver, binding.context_identity,
                                      &handles_, &setup_status, error_out)) {
    evidence_.status = setup_status;
    return false;
  }

  constexpr unsigned int kBlockX = 16;
  constexpr unsigned int kBlockY = 16;
  const auto dst_w = static_cast<unsigned int>(spec.dst_w);
  const auto dst_h = static_cast<unsigned int>(spec.dst_h);

  binding_ = binding;
  spec_ = spec;
  evidence_.status = PreparedPreprocessKernelStatus::ready;
  evidence_.context_identity = binding.context_identity;
  evidence_.stream_identity = binding.stream_identity;
  evidence_.configuration_generation = binding.configuration_generation;
  evidence_.model_preprocess_generation = binding.model_preprocess_generation;
  evidence_.launch.grid_x = (dst_w + kBlockX - 1u) / kBlockX;
  evidence_.launch.grid_y = (dst_h + kBlockY - 1u) / kBlockY;
  evidence_.launch.block_x = kBlockX;
  evidence_.launch.block_y = kBlockY;
  evidence_.launch.dst_w = dst_w;
  evidence_.launch.dst_h = dst_h;
  evidence_.launch.dst_is_bgr = spec.dst_order == ChannelOrder::bgr ? 1u : 0u;
  for (int channel = 0; channel < 3; ++channel) {
    evidence_.launch.mean[channel] = spec.mean[channel];
    evidence_.launch.inverse_std[channel] = 1.0f / spec.std[channel];
  }
  prepared_ = true;
  return true;
}

PreparedPreprocessKernelStatus PreparedPreprocessKernelBundle::ValidateBinding(
    const PreparedPreprocessBinding &binding,
    const ModelPreprocessSpec &spec) const noexcept {
  if (stopped_)
    return PreparedPreprocessKernelStatus::shutdown;
  if (evidence_.quarantined)
    return PreparedPreprocessKernelStatus::device_lost;
  if (!prepared_)
    return evidence_.status == PreparedPreprocessKernelStatus::reconfigured
               ? PreparedPreprocessKernelStatus::reconfigured
               : PreparedPreprocessKernelStatus::unavailable;
  if (!binding.driver || binding.driver != binding_.driver)
    return PreparedPreprocessKernelStatus::foreign_context;

  const auto *current = binding.driver->context_identity();
  if (!current || !current->Valid())
    return PreparedPreprocessKernelStatus::stale_context_generation;
  if (current->physical_device != binding_.context_identity.physical_device ||
      current->retained_context_id !=
          binding_.context_identity.retained_context_id ||
      binding.context_identity.physical_device !=
          binding_.context_identity.physical_device ||
      binding.context_identity.retained_context_id !=
          binding_.context_identity.retained_context_id) {
    return PreparedPreprocessKernelStatus::foreign_context;
  }
  if (current->logical_generation !=
          binding_.context_identity.logical_generation ||
      binding.context_identity.logical_generation !=
          binding_.context_identity.logical_generation) {
    return PreparedPreprocessKernelStatus::stale_context_generation;
  }
  if (binding.stream != binding_.stream ||
      binding.stream_identity != binding_.stream_identity) {
    return PreparedPreprocessKernelStatus::foreign_stream;
  }
  if (binding.configuration_generation != binding_.configuration_generation ||
      binding.model_preprocess_generation !=
          binding_.model_preprocess_generation) {
    return PreparedPreprocessKernelStatus::reconfigured;
  }
  if (!SameSpec(spec, spec_))
    return PreparedPreprocessKernelStatus::foreign_preprocess_spec;
  return PreparedPreprocessKernelStatus::ready;
}

void PreparedPreprocessKernelBundle::SetStatus(
    PreparedPreprocessKernelStatus status,
    PreparedPreprocessKernelStatus *status_out) noexcept {
  evidence_.status = status;
  if (status_out)
    *status_out = status;
}

bool PreparedPreprocessKernelBundle::Execute(
    const PreparedPreprocessBinding &binding, const ModelPreprocessSpec &spec,
    const CudaImage &source, const CudaTensor &destination,
    PreparedPreprocessKernelStatus *status_out) noexcept {
  return ExecuteImpl(binding, spec, nullptr, source, destination, status_out);
}

bool PreparedPreprocessKernelBundle::ExecuteLetterboxed(
    const PreparedPreprocessBinding &binding, const ModelPreprocessSpec &spec,
    const LetterboxGeometry &letterbox, const CudaImage &source,
    const CudaTensor &destination,
    PreparedPreprocessKernelStatus *status_out) noexcept {
  return ExecuteImpl(binding, spec, &letterbox, source, destination, status_out);
}

bool PreparedPreprocessKernelBundle::ExecuteImpl(
    const PreparedPreprocessBinding &binding, const ModelPreprocessSpec &spec,
    const LetterboxGeometry *letterbox, const CudaImage &source,
    const CudaTensor &destination,
    PreparedPreprocessKernelStatus *status_out) noexcept {
  if (status_out)
    *status_out = PreparedPreprocessKernelStatus::invalid;

  const auto binding_status = ValidateBinding(binding, spec);
  if (binding_status != PreparedPreprocessKernelStatus::ready) {
    SetStatus(binding_status, status_out);
    return false;
  }

  CUcontext current_context = nullptr;
  const CUresult context_result =
      binding.driver->f().cuCtxGetCurrent(&current_context);
  if (IsDeviceLoss(context_result)) {
    evidence_.quarantined = true;
    prepared_ = false;
    SetStatus(PreparedPreprocessKernelStatus::device_lost, status_out);
    return false;
  }
  if (context_result != CUDA_SUCCESS || !current_context ||
      current_context != handles_.context) {
    SetStatus(PreparedPreprocessKernelStatus::foreign_context, status_out);
    return false;
  }

  if (!source.Valid() || !destination.Valid() ||
      (source.format != PixelFormatGpu::rgb_u8 &&
       source.format != PixelFormatGpu::bgr_u8) ||
      source.pitch > detail::kSignedInt32PtxPitchMaxBytes ||
      source.pitch < source.RowBytes() || destination.n != 1 ||
      destination.c != 3 ||
      destination.h != static_cast<int>(evidence_.launch.dst_h) ||
      destination.w != static_cast<int>(evidence_.launch.dst_w)) {
    SetStatus(PreparedPreprocessKernelStatus::invalid_argument, status_out);
    return false;
  }
  if (letterbox && !letterbox->ValidFor(destination.w, destination.h)) {
    SetStatus(PreparedPreprocessKernelStatus::invalid_argument, status_out);
    return false;
  }

  const std::size_t element_count = static_cast<std::size_t>(destination.n) *
                                    static_cast<std::size_t>(destination.c) *
                                    static_cast<std::size_t>(destination.h) *
                                    static_cast<std::size_t>(destination.w);
  if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      destination.bytes < element_count * sizeof(float)) {
    SetStatus(PreparedPreprocessKernelStatus::invalid_argument, status_out);
    return false;
  }

  const std::uint32_t source_pitch = static_cast<std::uint32_t>(source.pitch);
  const std::uint32_t source_w = static_cast<std::uint32_t>(source.w);
  const std::uint32_t source_h = static_cast<std::uint32_t>(source.h);
  const unsigned long long source_ptr = source.ptr;
  const unsigned long long destination_ptr = destination.ptr;
  const std::uint32_t destination_w = evidence_.launch.dst_w;
  const std::uint32_t destination_h = evidence_.launch.dst_h;
  const float mean0 = evidence_.launch.mean[0];
  const float mean1 = evidence_.launch.mean[1];
  const float mean2 = evidence_.launch.mean[2];
  const float inverse_std0 = evidence_.launch.inverse_std[0];
  const float inverse_std1 = evidence_.launch.inverse_std[1];
  const float inverse_std2 = evidence_.launch.inverse_std[2];
  const std::uint32_t destination_is_bgr = evidence_.launch.dst_is_bgr;
  const std::uint32_t source_is_bgr =
      source.format == PixelFormatGpu::bgr_u8 ? 1u : 0u;
  const std::uint32_t letterbox_enabled = letterbox ? 1u : 0u;
  const std::uint32_t scaled_w = letterbox
                                     ? static_cast<std::uint32_t>(letterbox->scaled_w)
                                     : destination_w;
  const std::uint32_t scaled_h = letterbox
                                     ? static_cast<std::uint32_t>(letterbox->scaled_h)
                                     : destination_h;
  const std::uint32_t pad_x =
      letterbox ? static_cast<std::uint32_t>(letterbox->pad_x) : 0u;
  const std::uint32_t pad_y =
      letterbox ? static_cast<std::uint32_t>(letterbox->pad_y) : 0u;
  const std::uint32_t pad_value = letterbox ? letterbox->pad_value : 0u;

  void *arguments[] = {
      const_cast<unsigned long long *>(&source_ptr),
      const_cast<std::uint32_t *>(&source_pitch),
      const_cast<std::uint32_t *>(&source_w),
      const_cast<std::uint32_t *>(&source_h),
      const_cast<unsigned long long *>(&destination_ptr),
      const_cast<std::uint32_t *>(&destination_w),
      const_cast<std::uint32_t *>(&destination_h),
      const_cast<float *>(&mean0),
      const_cast<float *>(&mean1),
      const_cast<float *>(&mean2),
      const_cast<float *>(&inverse_std0),
      const_cast<float *>(&inverse_std1),
      const_cast<float *>(&inverse_std2),
      const_cast<std::uint32_t *>(&destination_is_bgr),
      const_cast<std::uint32_t *>(&source_is_bgr),
      const_cast<std::uint32_t *>(&letterbox_enabled),
      const_cast<std::uint32_t *>(&scaled_w),
      const_cast<std::uint32_t *>(&scaled_h),
      const_cast<std::uint32_t *>(&pad_x),
      const_cast<std::uint32_t *>(&pad_y),
      const_cast<std::uint32_t *>(&pad_value),
  };

  const CUresult launch_result = binding.driver->f().cuLaunchKernel(
      handles_.function, evidence_.launch.grid_x, evidence_.launch.grid_y, 1,
      evidence_.launch.block_x, evidence_.launch.block_y, 1, 0, binding.stream,
      arguments, nullptr);
  if (launch_result != CUDA_SUCCESS) {
    if (IsDeviceLoss(launch_result)) {
      evidence_.quarantined = true;
      prepared_ = false;
      SetStatus(PreparedPreprocessKernelStatus::device_lost, status_out);
    } else {
      SetStatus(PreparedPreprocessKernelStatus::launch_failed, status_out);
    }
    return false;
  }

  ++evidence_.successful_launches;
  SetStatus(PreparedPreprocessKernelStatus::launch_enqueued, status_out);
  return true;
}

void PreparedPreprocessKernelBundle::InvalidateForReconfiguration(
    std::uint64_t configuration_generation,
    std::uint64_t model_preprocess_generation) noexcept {
  if (stopped_)
    return;
  if (configuration_generation == binding_.configuration_generation &&
      model_preprocess_generation == binding_.model_preprocess_generation) {
    return;
  }
  prepared_ = false;
  handles_ = {};
  evidence_.status = PreparedPreprocessKernelStatus::reconfigured;
  evidence_.configuration_generation = configuration_generation;
  evidence_.model_preprocess_generation = model_preprocess_generation;
}

void PreparedPreprocessKernelBundle::Shutdown() noexcept {
  prepared_ = false;
  stopped_ = true;
  binding_ = {};
  spec_ = {};
  handles_ = {};
  evidence_.status = PreparedPreprocessKernelStatus::shutdown;
}

bool PreparedPreprocessKernelBundle::ready() const noexcept {
  return prepared_ && !stopped_ && !evidence_.quarantined;
}

PreparedPreprocessKernelEvidence
PreparedPreprocessKernelBundle::evidence() const noexcept {
  return evidence_;
}

} // namespace studiocast::cuda::kernels
