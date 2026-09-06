#include "core/open_video/native_prepared_cuda_fastdvdnet_runtime.h"

#include <algorithm>
#include <atomic>
#include <cstring>

namespace studiocast::open_video {
namespace {
std::atomic<std::uint64_t> g_next_handle{1};

class OrtAdapter final : public IPreparedFastDvdnetOrtSession {
public:
  OrtAdapter(std::unique_ptr<onnx::OrtSession> session,
             onnx::OrtPreparedCudaRunEvidence evidence) noexcept
      : session_(std::move(session)), evidence_(evidence) {}
  onnx::OrtPreparedCudaRunEvidence evidence() const noexcept override {
    return evidence_;
  }
  bool PrepareBinding(const onnx::OrtSession::CudaBindingInput *inputs,
                      std::size_t input_count,
                      const onnx::OrtSession::CudaBindingOutput *outputs,
                      std::size_t output_count) noexcept override {
    return session_->PrepareCudaIoBinding(0, inputs, input_count, outputs,
                                          output_count, nullptr);
  }
  onnx::OrtPreparedRunResult Execute(
      const onnx::OrtSession::CudaBindingInput *inputs, std::size_t input_count,
      const onnx::OrtSession::CudaBindingOutput *outputs,
      std::size_t output_count,
      const onnx::OrtPreparedCudaRunEvidence &expected) noexcept override {
    return session_->ExecutePreparedCudaIoBindingTyped(
        0, inputs, input_count, outputs, output_count, expected);
  }
  void RequestShutdown() noexcept override { session_->RequestShutdown(); }
private:
  std::unique_ptr<onnx::OrtSession> session_;
  onnx::OrtPreparedCudaRunEvidence evidence_{};
};

class OrtFactory final : public IPreparedFastDvdnetOrtFactory {
public:
  std::unique_ptr<IPreparedFastDvdnetOrtSession>
  Create(const std::filesystem::path &model,
         const onnx::OrtSessionOptions &options,
         const PreparedCudaFastDvdnetTensorContract &tensors) noexcept override {
    onnx::OrtSessionInfo info;
    auto session = onnx::OrtSession::Create(model, options, &info, nullptr);
    if (!session || !info.strict_provider_validation_passed ||
        !info.prepared_provider_evidence.ValidStrictCuda() ||
        info.input_names.size() != 2 || info.output_names.size() != 1 ||
        info.input_names[0] != tensors.noisy_name ||
        info.input_names[1] != tensors.noise_map_name ||
        info.output_names[0] != tensors.denoised_name ||
        info.input_shapes.size() != 2 || info.output_shapes.size() != 1 ||
        info.input_shapes[0] != std::vector<std::int64_t>(tensors.noisy_shape.begin(), tensors.noisy_shape.end()) ||
        info.input_shapes[1] != std::vector<std::int64_t>(tensors.noise_map_shape.begin(), tensors.noise_map_shape.end()) ||
        info.output_shapes[0] != std::vector<std::int64_t>(tensors.denoised_shape.begin(), tensors.denoised_shape.end()) ||
        info.input_elem_types.size() != 2 || info.output_elem_types.size() != 1 ||
        info.input_elem_types[0] != 1 || info.input_elem_types[1] != 1 ||
        info.output_elem_types[0] != 1)
      return {};
    auto evidence = session->prepared_cuda_run_evidence();
    if (!evidence.Valid()) return {};
    return std::make_unique<OrtAdapter>(std::move(session), evidence);
  }
};

const ModelFile *MainOnnx(const ModelPack &pack) noexcept {
  const ModelFile *fallback = nullptr;
  for (const auto &file : pack.files) {
    if (file.kind != "onnx") continue;
    if (!fallback) fallback = &file;
    if (file.role == "main") return &file;
  }
  return fallback;
}

bool SameProvider(const onnx::OrtPreparedCudaRunEvidence &a,
                  const onnx::OrtPreparedCudaRunEvidence &b) noexcept {
  return a.Valid() && b.Valid() && a.provider == b.provider;
}
} // namespace

NativePreparedCudaFastDvdnetRuntime::NativePreparedCudaFastDvdnetRuntime(
    IPreparedFastDvdnetOrtFactory *factory) noexcept : factory_(factory) {
  if (!factory_) {
    owned_factory_ = std::make_unique<OrtFactory>();
    factory_ = owned_factory_.get();
  }
}

NativePreparedCudaFastDvdnetRuntime::~NativePreparedCudaFastDvdnetRuntime() {
  if (prepared_) {
    Shutdown(handle_);
  }
}

PreparedCudaFastDvdnetRuntimePrepareResult
NativePreparedCudaFastDvdnetRuntime::Prepare(
    const PreparedCudaFastDvdnetConfig &config) noexcept {
  PreparedCudaFastDvdnetRuntimePrepareResult result;
  ++telemetry_.allocation_requests; // candidate/session owner
  if (prepared_ || !factory_ || !config.driver || !config.owner_context ||
      !config.owner_stream || !config.model_pack || !config.artifact.Valid() ||
      !config.ort_evidence.provider.ValidStrictCuda() ||
      config.ort_evidence.provider.explicit_stream_synchronization ||
      config.ort_evidence.provider.requested_stream_identity !=
          reinterpret_cast<std::uintptr_t>(config.owner_stream)) {
    result.status = PreparedCudaFastDvdnetStatus::malformed_contract;
    return result;
  }
  const auto *identity = config.driver->context_identity();
  cuda::CUcontext current = nullptr;
  if (!identity || identity->logical_generation !=
                       config.execution_domain.logical_context_generation ||
      config.driver->f().cuCtxGetCurrent(&current) != cuda::CUDA_SUCCESS ||
      current != config.owner_context) {
    result.status = PreparedCudaFastDvdnetStatus::wrong_execution_domain;
    return result;
  }
  const ModelFile *model = MainOnnx(*config.model_pack);
  if (!model || model->path.empty()) {
    result.status = PreparedCudaFastDvdnetStatus::stale_artifact;
    return result;
  }
  const int w = static_cast<int>(config.tensors.noisy_shape[3]);
  const int h = static_cast<int>(config.tensors.noisy_shape[2]);
  config_ = config;
  std::string error;
  for (auto &slot : history_) {
    slot = std::make_unique<cuda::CudaTensor>();
    ++telemetry_.allocation_requests;
    if (!slot->AllocateNchwF32(config.driver, 1, 3, h, w, &error)) {
      result.status = PreparedCudaFastDvdnetStatus::resource_exhausted;
      Release(); return result;
    }
  }
  if (!noisy_.AllocateNchwF32(config.driver, 1, 15, h, w, &error) ||
      !noise_map_.AllocateNchwF32(config.driver, 1, 1, h, w, &error) ||
      !denoised_.AllocateNchwF32(config.driver, 1, 3, h, w, &error) ||
      !cuda::kernels::PrepareFastDvdnetKernels(
          *config.driver, config.owner_context, config.owner_stream, &kernels_,
          &error)) {
    result.status = PreparedCudaFastDvdnetStatus::resource_exhausted;
    Release(); return result;
  }
  telemetry_.allocation_requests += 3;
  if (!config.driver->f().cuEventCreate || !config.driver->f().cuEventRecord ||
      !config.driver->f().cuEventQuery || !config.driver->f().cuEventDestroy ||
      config.driver->f().cuEventCreate(&completion_event_, 0) !=
          cuda::CUDA_SUCCESS) {
    result.status = PreparedCudaFastDvdnetStatus::synchronization_failed;
    Release(); return result;
  }

  onnx::OrtSessionOptions options;
  options.provider_policy = onnx::OrtProviderPolicy::strict_cuda_only;
  options.prefer_cuda = true;
  options.enable_tensorrt = false;
  options.cuda_device_id = config.cuda_device_id;
  options.cuda_context_generation =
      config.execution_domain.logical_context_generation;
  options.user_compute_stream = config.owner_stream;
  ++telemetry_.session_constructions;
  session_ = factory_->Create(model->path, options, config.tensors);
  if (!session_ || !SameProvider(session_->evidence(), config.ort_evidence)) {
    result.status = PreparedCudaFastDvdnetStatus::wrong_provider;
    Release(); return result;
  }
  active_ort_evidence_ = session_->evidence();
  inputs_[0] = {config.tensors.noisy_name,
                reinterpret_cast<const float *>(noisy_.ptr),
                noisy_.ElementCount(), config.tensors.noisy_shape.data(), 4};
  inputs_[1] = {config.tensors.noise_map_name,
                reinterpret_cast<const float *>(noise_map_.ptr),
                noise_map_.ElementCount(), config.tensors.noise_map_shape.data(), 4};
  outputs_[0] = {config.tensors.denoised_name,
                 reinterpret_cast<float *>(denoised_.ptr),
                 denoised_.ElementCount(), config.tensors.denoised_shape.data(), 4};
  ++telemetry_.binding_constructions;
  if (!session_->PrepareBinding(inputs_.data(), inputs_.size(), outputs_.data(),
                                outputs_.size())) {
    result.status = PreparedCudaFastDvdnetStatus::inference_failed;
    Release(); return result;
  }
  // Warmup uses zeroed resident tensors; no host tensor or image is involved.
  if (!config.driver->MemsetD8Async(noisy_.ptr, 0, noisy_.bytes,
                                    config.owner_stream, nullptr) ||
      !config.driver->MemsetD8Async(noise_map_.ptr, 0, noise_map_.bytes,
                                    config.owner_stream, nullptr) ||
      !session_->Execute(inputs_.data(), inputs_.size(), outputs_.data(),
                         outputs_.size(), active_ort_evidence_).ok() ||
      !config.driver->StreamSynchronize(config.owner_stream, nullptr)) {
    result.status = PreparedCudaFastDvdnetStatus::inference_failed;
    Release(); return result;
  }
  handle_ = g_next_handle.fetch_add(1, std::memory_order_relaxed);
  if (handle_ == 0) handle_ = g_next_handle.fetch_add(1);
  prepared_ = true;
  reset_requested_ = true;
  const std::uint64_t history_bytes = history_[0]->bytes * 3u;
  result.reserved_budget.persistent_bytes =
      history_bytes + noisy_.bytes + noise_map_.bytes + denoised_.bytes;
  result.reserved_budget.resource_count = 8;
  result.reserved_budget.in_flight_slots = 1;
  result.status = PreparedCudaFastDvdnetStatus::ready;
  result.runtime_handle = handle_;
  result.reusable_bindings = true;
  result.warmup_complete = true;
  result.strict_cuda_only = true;
  result.user_stream_bound = true;
  result.cpu_fallback_disabled = true;
  return result;
}

bool NativePreparedCudaFastDvdnetRuntime::CopyHistory(
    const cuda::CudaTensor &source,
    const cuda::CudaTensor &destination) noexcept {
  cuda::CUDA_MEMCPY2D copy{};
  copy.srcMemoryType = cuda::CU_MEMORYTYPE_DEVICE;
  copy.srcDevice = source.ptr;
  copy.srcPitch = source.bytes;
  copy.dstMemoryType = cuda::CU_MEMORYTYPE_DEVICE;
  copy.dstDevice = destination.ptr;
  copy.dstPitch = destination.bytes;
  copy.WidthInBytes = source.bytes;
  copy.Height = 1;
  const bool ok = config_.driver->f().cuMemcpy2DAsync(&copy, config_.owner_stream) ==
                  cuda::CUDA_SUCCESS;
  if (ok) ++telemetry_.internal_device_copies;
  return ok;
}

PreparedCudaFastDvdnetRuntimeFrameResult
NativePreparedCudaFastDvdnetRuntime::Execute(
    std::uint64_t handle, const PreparedCudaFastDvdnetTemporalPlan &temporal,
    const cuda::CudaImage &input, cuda::CudaImage *output, float sigma) noexcept {
  PreparedCudaFastDvdnetRuntimeFrameResult result;
  if (!prepared_ || handle != handle_) {
    result.status = PreparedCudaFastDvdnetStatus::foreign_handle; return result;
  }
  if (shutdown_requested_) {
    result.status = PreparedCudaFastDvdnetStatus::shutdown; return result;
  }
  if (config_.driver->BindPreparedContext(*config_.driver->context_identity()) !=
      cuda::CUDA_SUCCESS) {
    result.status = PreparedCudaFastDvdnetStatus::device_lost; return result;
  }
  const bool reset = temporal.reset || reset_requested_;
  if (!cuda::kernels::LaunchFastDvdnetPreprocess(
          kernels_, input, *history_[0], *history_[1], *history_[2], noisy_,
          noise_map_, reset, sigma, nullptr)) {
    result.status = cuda::IsCudaDeviceLossResult(kernels_.last_result)
                        ? PreparedCudaFastDvdnetStatus::device_lost
                        : PreparedCudaFastDvdnetStatus::preprocess_failed;
    result.failed_operation = PreparedCudaFastDvdnetOperation::temporal_preprocess;
    return result;
  }
  if (reset && (!CopyHistory(*history_[2], *history_[0]) ||
                !CopyHistory(*history_[2], *history_[1]))) {
    result.status = PreparedCudaFastDvdnetStatus::preprocess_failed;
    result.failed_operation = PreparedCudaFastDvdnetOperation::temporal_preprocess;
    return result;
  }
  result.ort = session_->Execute(inputs_.data(), inputs_.size(), outputs_.data(),
                                 outputs_.size(), active_ort_evidence_);
  if (!result.ort.ok()) {
    result.status = result.ort.status == onnx::OrtPreparedRunStatus::provider_failure_quarantine
                        ? PreparedCudaFastDvdnetStatus::device_lost
                        : PreparedCudaFastDvdnetStatus::inference_failed;
    result.failed_operation = PreparedCudaFastDvdnetOperation::inference;
    return result;
  }
  if (!cuda::kernels::LaunchFastDvdnetPostprocess(kernels_, denoised_, *output,
                                                  nullptr)) {
    result.status = cuda::IsCudaDeviceLossResult(kernels_.last_result)
                        ? PreparedCudaFastDvdnetStatus::device_lost
                        : PreparedCudaFastDvdnetStatus::postprocess_failed;
    result.failed_operation = PreparedCudaFastDvdnetOperation::postprocess;
    return result;
  }
  if (config_.driver->f().cuEventRecord(completion_event_, config_.owner_stream) !=
      cuda::CUDA_SUCCESS) {
    result.status = PreparedCudaFastDvdnetStatus::synchronization_failed;
    result.failed_operation = PreparedCudaFastDvdnetOperation::completion_wait;
    return result;
  }
  work_pending_ = true;
  std::swap(history_[0], history_[1]);
  std::swap(history_[1], history_[2]);
  reset_requested_ = false;
  result.status = PreparedCudaFastDvdnetStatus::executed;
  result.output_valid = true;
  return result;
}

PreparedCudaFastDvdnetStatus NativePreparedCudaFastDvdnetRuntime::ResetTemporalState(
    std::uint64_t handle) noexcept {
  if (!prepared_ || handle != handle_) return PreparedCudaFastDvdnetStatus::foreign_handle;
  reset_requested_ = true;
  return PreparedCudaFastDvdnetStatus::ready;
}

bool NativePreparedCudaFastDvdnetRuntime::Release() noexcept {
  bool ok = true;
  // Drop IoBinding/tensor wrappers before releasing their backing allocations.
  session_.reset();
  if (completion_event_ && config_.driver) {
    ok = config_.driver->f().cuEventDestroy(completion_event_) == cuda::CUDA_SUCCESS;
    if (ok) completion_event_ = nullptr;
  }
  if (kernels_.live) ok = cuda::kernels::DestroyFastDvdnetKernels(&kernels_, nullptr) && ok;
  for (auto &slot : history_) {
    if (slot && slot->owns_memory) ok = slot->Free(config_.driver, nullptr) && ok;
    if (ok) slot.reset();
  }
  if (noisy_.owns_memory) ok = noisy_.Free(config_.driver, nullptr) && ok;
  if (noise_map_.owns_memory) ok = noise_map_.Free(config_.driver, nullptr) && ok;
  if (denoised_.owns_memory) ok = denoised_.Free(config_.driver, nullptr) && ok;
  if (ok) { prepared_ = false; handle_ = 0; config_ = {}; active_ort_evidence_ = {}; }
  return ok;
}

PreparedCudaFastDvdnetShutdownSnapshot
NativePreparedCudaFastDvdnetRuntime::Shutdown(std::uint64_t handle) noexcept {
  ++shutdown_attempts_;
  if (!prepared_ || handle != handle_)
    return {PreparedCudaFastDvdnetStatus::foreign_handle, shutdown_attempts_, true, false};
  shutdown_requested_ = true;
  if (session_) session_->RequestShutdown();
  if (work_pending_) {
    const auto query = config_.driver->f().cuEventQuery(completion_event_);
    if (query == cuda::CUDA_ERROR_NOT_READY)
      return {PreparedCudaFastDvdnetStatus::completion_pending, shutdown_attempts_, true, false};
    if (query != cuda::CUDA_SUCCESS)
      return {PreparedCudaFastDvdnetStatus::device_lost, shutdown_attempts_, true, false};
    work_pending_ = false;
  }
  const bool released = Release();
  return {released ? PreparedCudaFastDvdnetStatus::shutdown
                   : PreparedCudaFastDvdnetStatus::completion_pending,
          shutdown_attempts_, true, released};
}

PreparedCudaFastDvdnetRuntimeTelemetry
NativePreparedCudaFastDvdnetRuntime::Telemetry() const noexcept { return telemetry_; }

} // namespace studiocast::open_video
