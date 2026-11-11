#include "core/open_video/native_prepared_cuda_fastdvdnet_runtime.h"
#include "core/util/fs.h"
#include "core/util/json.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>

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
         const PreparedCudaFastDvdnetTensorContract &tensors,
         const PreparedFastDvdnetModelContract &contract) noexcept override {
    onnx::OrtSessionInfo info;
    auto session = onnx::OrtSession::Create(model, options, &info, nullptr);
    if (!session || !info.strict_provider_validation_passed ||
        !info.prepared_provider_evidence.ValidStrictCuda() ||
        !ValidatePreparedFastDvdnetSessionMetadata(info, contract, tensors))
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

bool ReadManifestTensor(const util::json::Value &value, std::string *name,
                        int *channels, int *width, int *height,
                        bool *dynamic_hw, bool require_layout) {
  const auto *object = value.AsObject();
  if (!object) return false;
  const auto string_value = [&](const char *key) -> const std::string * {
    const auto it = object->find(key);
    return it == object->end() ? nullptr : it->second.AsString();
  };
  const auto *n = string_value("name");
  const auto *layout = string_value("layout");
  const auto *dtype = string_value("dtype");
  const auto channel_it = object->find("channels");
  const auto dynamic_it = object->find("dynamic_hw");
  const auto width_it = object->find("width");
  const auto height_it = object->find("height");
  const auto *number = channel_it == object->end()
                           ? nullptr : channel_it->second.AsNumber();
  const auto *dynamic = dynamic_it == object->end()
                            ? nullptr : dynamic_it->second.AsBool();
  if (!n || (require_layout && (!layout || *layout != "nchw")) ||
      (!require_layout && layout && *layout != "nchw") ||
      !dtype || *dtype != "float32" ||
      !number || *number < 1 || *number > std::numeric_limits<int>::max() ||
      static_cast<double>(static_cast<int>(*number)) != *number || !dynamic)
    return false;
  const auto integer_field = [&](const util::json::Value::Object::const_iterator &it,
                                 int *out) {
    if (it == object->end()) return false;
    const auto *value_number = it->second.AsNumber();
    if (!value_number || *value_number < 1 ||
        *value_number > std::numeric_limits<int>::max() ||
        static_cast<double>(static_cast<int>(*value_number)) != *value_number)
      return false;
    *out = static_cast<int>(*value_number);
    return true;
  };
  if (!*dynamic &&
      (!integer_field(width_it, width) || !integer_field(height_it, height)))
    return false;
  *name = *n;
  *channels = static_cast<int>(*number);
  *dynamic_hw = *dynamic;
  return true;
}

bool ReadFastDvdnetManifest(const ModelPack &pack,
                            PreparedFastDvdnetModelContract *contract) {
  const auto text = util::ReadTextFile(pack.manifest_path.string());
  util::json::Value root;
  std::string error;
  if (!text || !util::json::Parse(*text, &root, &error)) return false;
  const auto *root_object = root.AsObject();
  if (!root_object) return false;
  const auto onnx_it = root_object->find("onnx");
  if (onnx_it == root_object->end()) return false;
  const auto *onnx = onnx_it->second.AsObject();
  if (!onnx) return false;
  const auto inputs_it = onnx->find("inputs");
  const auto outputs_it = onnx->find("outputs");
  const auto *inputs = inputs_it == onnx->end()
                           ? nullptr : inputs_it->second.AsArray();
  const auto *outputs = outputs_it == onnx->end()
                            ? nullptr : outputs_it->second.AsArray();
  return inputs && inputs->size() == 2 && outputs && outputs->size() == 1 &&
         ReadManifestTensor((*inputs)[0], &contract->input_names[0],
                            &contract->input_channels[0],
                            &contract->input_widths[0],
                            &contract->input_heights[0],
                            &contract->input_dynamic_hw[0], true) &&
         ReadManifestTensor((*inputs)[1], &contract->input_names[1],
                            &contract->input_channels[1],
                            &contract->input_widths[1],
                            &contract->input_heights[1],
                            &contract->input_dynamic_hw[1], true) &&
         ReadManifestTensor((*outputs)[0], &contract->output_name,
                            &contract->output_channels,
                            &contract->output_width,
                            &contract->output_height,
                            &contract->output_dynamic_hw, false);
}

} // namespace

bool ValidatePreparedFastDvdnetSessionMetadata(
    const onnx::OrtSessionInfo &info,
    const PreparedFastDvdnetModelContract &model,
    const PreparedCudaFastDvdnetTensorContract &selected) noexcept {
  if (info.input_names.size() != 2 || info.output_names.size() != 1 ||
      info.input_shapes.size() != 2 || info.output_shapes.size() != 1 ||
      info.input_elem_types.size() != 2 || info.output_elem_types.size() != 1 ||
      !selected.noisy_name || !selected.noise_map_name ||
      !selected.denoised_name ||
      model.input_names[0] != selected.noisy_name ||
      model.input_names[1] != selected.noise_map_name ||
      model.output_name != selected.denoised_name ||
      info.input_names[0] != model.input_names[0] ||
      info.input_names[1] != model.input_names[1] ||
      info.output_names[0] != model.output_name ||
      info.input_elem_types[0] != 1 || info.input_elem_types[1] != 1 ||
      info.output_elem_types[0] != 1)
    return false;
  const std::array<const std::vector<std::int64_t> *, 3> actual{
      &info.input_shapes[0], &info.input_shapes[1], &info.output_shapes[0]};
  const std::array<const std::array<std::int64_t, 4> *, 3> selected_shapes{
      &selected.noisy_shape, &selected.noise_map_shape, &selected.denoised_shape};
  const std::array<int, 3> channels{model.input_channels[0],
                                    model.input_channels[1],
                                    model.output_channels};
  const std::array<bool, 3> dynamic{model.input_dynamic_hw[0],
                                    model.input_dynamic_hw[1],
                                    model.output_dynamic_hw};
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (selected_shapes[i]->at(0) != 1 ||
        selected_shapes[i]->at(1) != channels[i] ||
        selected_shapes[i]->at(2) <= 0 || selected_shapes[i]->at(3) <= 0 ||
        actual[i]->size() != 4 || (*actual[i])[0] != 1 ||
        (*actual[i])[1] != channels[i])
      return false;
    if (dynamic[i]) {
      if (model.setup_specialized_dynamic_hw) {
        if (*actual[i] != std::vector<std::int64_t>(selected_shapes[i]->begin(), selected_shapes[i]->end()))
          return false;
      } else if ((*actual[i])[2] >= 0 || (*actual[i])[3] >= 0) return false;
    } else {
      const int manifest_width = i < 2 ? model.input_widths[i]
                                       : model.output_width;
      const int manifest_height = i < 2 ? model.input_heights[i]
                                        : model.output_height;
      if (manifest_width <= 0 || manifest_height <= 0 ||
          selected_shapes[i]->at(2) != manifest_height ||
          selected_shapes[i]->at(3) != manifest_width ||
          *actual[i] != std::vector<std::int64_t>(
                            selected_shapes[i]->begin(),
                            selected_shapes[i]->end()))
        return false;
    }
  }
  return true;
}

NativePreparedCudaFastDvdnetRuntime::NativePreparedCudaFastDvdnetRuntime(
    IPreparedFastDvdnetOrtFactory *factory) noexcept
    : factory_(factory), injected_factory_(factory != nullptr) {
  if (!factory_) {
    owned_factory_ = std::make_unique<OrtFactory>();
    factory_ = owned_factory_.get();
  }
}

NativePreparedCudaFastDvdnetRuntime::~NativePreparedCudaFastDvdnetRuntime() {
  if (prepared_) {
    const auto stopped = Shutdown(handle_);
    if (!stopped.quiescent) Quarantine();
  }
}

PreparedCudaFastDvdnetRuntimePrepareResult
NativePreparedCudaFastDvdnetRuntime::Prepare(
    const PreparedCudaFastDvdnetConfig &config) noexcept {
  PreparedCudaFastDvdnetRuntimePrepareResult result;
  ++telemetry_.allocation_requests; // candidate/session owner
  if (prepared_ || work_pending_ || !factory_ || !config.driver || !config.owner_context ||
      !config.owner_stream || !config.model_pack || !config.artifact.Valid()) {
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
  PreparedFastDvdnetModelContract model_contract;
  if (!injected_factory_) {
    if (!ReadFastDvdnetManifest(*config.model_pack, &model_contract)) {
      result.status = PreparedCudaFastDvdnetStatus::stale_artifact;
      return result;
    }
    const auto validated = ValidateModelArtifactIdentity(*config.model_pack, *model);
    if (!validated.ok() ||
        !ModelArtifactIdentityMatches(validated.identity, config.artifact)) {
      result.status = PreparedCudaFastDvdnetStatus::stale_artifact;
      return result;
    }
  } else {
    model_contract.input_names = {config.tensors.noisy_name,
                                  config.tensors.noise_map_name};
    model_contract.output_name = config.tensors.denoised_name;
    model_contract.input_channels = {15, 1};
    model_contract.output_channels = 3;
  }
  const auto tensor_w = config.tensors.noisy_shape[3];
  const auto tensor_h = config.tensors.noisy_shape[2];
  if (tensor_w <= 0 || tensor_h <= 0 ||
      tensor_w > std::numeric_limits<int>::max() ||
      tensor_h > std::numeric_limits<int>::max()) {
    result.status = PreparedCudaFastDvdnetStatus::malformed_contract;
    return result;
  }
  const int w = static_cast<int>(tensor_w);
  const int h = static_cast<int>(tensor_h);
  constexpr std::uint64_t kFloatBytesPerPixel = 28u * sizeof(float);
  if (w <= 0 || h <= 0 || static_cast<std::uint64_t>(w) >
                              UINT64_MAX / static_cast<std::uint64_t>(h) ||
      static_cast<std::uint64_t>(w) * static_cast<std::uint64_t>(h) >
          UINT64_MAX / kFloatBytesPerPixel) {
    result.status = PreparedCudaFastDvdnetStatus::malformed_contract;
    return result;
  }
  const std::uint64_t required_bytes = static_cast<std::uint64_t>(w) *
                                       static_cast<std::uint64_t>(h) *
                                       kFloatBytesPerPixel;
  if (required_bytes > config.available_budget.persistent_bytes ||
      config.available_budget.resource_count < 8 ||
      config.available_budget.in_flight_slots < 1) {
    result.status = PreparedCudaFastDvdnetStatus::resource_exhausted;
    return result;
  }
  config_ = config;
  prepared_context_identity_ = *identity;
  tensor_names_ = {config.tensors.noisy_name, config.tensors.noise_map_name,
                   config.tensors.denoised_name};
  tensor_shapes_ = {config.tensors.noisy_shape, config.tensors.noise_map_shape,
                    config.tensors.denoised_shape};
  config_.tensors.noisy_name = tensor_names_[0].c_str();
  config_.tensors.noise_map_name = tensor_names_[1].c_str();
  config_.tensors.denoised_name = tensor_names_[2].c_str();
  config_.tensors.noisy_shape = tensor_shapes_[0];
  config_.tensors.noise_map_shape = tensor_shapes_[1];
  config_.tensors.denoised_shape = tensor_shapes_[2];
  std::string error;
  for (auto &slot : history_) {
    slot = std::make_unique<cuda::CudaTensor>();
    ++telemetry_.allocation_requests;
    if (!slot->AllocateNchwF32(config.driver, 1, 3, h, w, &error)) {
      FinishFailedPrepare(PreparedCudaFastDvdnetStatus::resource_exhausted,
                          &result);
      return result;
    }
  }
  if (!noisy_.AllocateNchwF32(config.driver, 1, 15, h, w, &error) ||
      !noise_map_.AllocateNchwF32(config.driver, 1, 1, h, w, &error) ||
      !denoised_.AllocateNchwF32(config.driver, 1, 3, h, w, &error) ||
      !cuda::kernels::PrepareFastDvdnetKernels(
          *config.driver, config.owner_context, config.owner_stream, &kernels_,
          &error)) {
    FinishFailedPrepare(PreparedCudaFastDvdnetStatus::resource_exhausted,
                        &result);
    return result;
  }
  telemetry_.allocation_requests += 3;
  if (!config.driver->f().cuEventCreate || !config.driver->f().cuEventRecord ||
      !config.driver->f().cuEventQuery || !config.driver->f().cuEventDestroy ||
      config.driver->f().cuEventCreate(&completion_event_, 0) !=
          cuda::CUDA_SUCCESS) {
    FinishFailedPrepare(PreparedCudaFastDvdnetStatus::synchronization_failed,
                        &result);
    return result;
  }

  onnx::OrtSessionOptions options;
  options.provider_policy = onnx::OrtProviderPolicy::strict_cuda_only;
  options.prefer_cuda = true;
  options.enable_tensorrt = false;
  options.cuda_device_id = config.cuda_device_id;
  options.cuda_context_generation =
      config.execution_domain.logical_context_generation;
  options.user_compute_stream = config.owner_stream;
  // The corrected curated export declares these reviewed symbolic names.
  // Other artifacts retain strict unsupported-symbol behavior.
  if (config.model_pack->id == "fastdvdnet_sigma15" &&
      model->sha256 == "0895dcad878f44ff4b610e0569bf701c8e23566e675c22a9bb0d59b95dcfc7ac") {
    options.free_dimension_overrides = {{"N", 1}, {"H", h}, {"W", w}};
    model_contract.setup_specialized_dynamic_hw = true;
  }
  ++telemetry_.session_constructions;
  session_ = factory_->Create(model->path, options, config.tensors,
                              model_contract);
  if (!session_ || !session_->evidence().Valid() ||
      !onnx::OrtPreparedProviderEvidenceMatches(
          session_->evidence().provider, config.cuda_device_id,
          config.execution_domain.logical_context_generation,
          reinterpret_cast<std::uintptr_t>(config.owner_stream))) {
    FinishFailedPrepare(PreparedCudaFastDvdnetStatus::wrong_provider, &result);
    return result;
  }
  active_ort_evidence_ = session_->evidence();
  inputs_[0] = {config_.tensors.noisy_name,
                reinterpret_cast<const float *>(noisy_.ptr),
                noisy_.ElementCount(), config_.tensors.noisy_shape.data(), 4};
  inputs_[1] = {config_.tensors.noise_map_name,
                reinterpret_cast<const float *>(noise_map_.ptr),
                noise_map_.ElementCount(), config_.tensors.noise_map_shape.data(), 4};
  outputs_[0] = {config_.tensors.denoised_name,
                 reinterpret_cast<float *>(denoised_.ptr),
                 denoised_.ElementCount(), config_.tensors.denoised_shape.data(), 4};
  ++telemetry_.binding_constructions;
  if (!session_->PrepareBinding(inputs_.data(), inputs_.size(), outputs_.data(),
                                outputs_.size())) {
    FinishFailedPrepare(PreparedCudaFastDvdnetStatus::inference_failed,
                        &result);
    return result;
  }
  // Warmup uses zeroed resident tensors; no host tensor or image is involved.
  work_pending_ = true;
  completion_recorded_ = false;
  const bool warmup_ok =
      config.driver->MemsetD8Async(noisy_.ptr, 0, noisy_.bytes,
                                   config.owner_stream, nullptr) &&
      config.driver->MemsetD8Async(noise_map_.ptr, 0, noise_map_.bytes,
                                   config.owner_stream, nullptr) &&
      session_->Execute(inputs_.data(), inputs_.size(), outputs_.data(),
                        outputs_.size(), active_ort_evidence_).ok();
  const bool warmup_quiescent = QuiescePending();
  if (!warmup_ok || !warmup_quiescent) {
    result.status = PreparedCudaFastDvdnetStatus::inference_failed;
    if (warmup_quiescent) FinishFailedPrepare(result.status, &result);
    else {
      handle_ = g_next_handle.fetch_add(1, std::memory_order_relaxed);
      if (handle_ == 0) handle_ = g_next_handle.fetch_add(1);
      prepared_ = true;
      shutdown_requested_ = true;
      result.status = PreparedCudaFastDvdnetStatus::completion_pending;
      result.runtime_handle = handle_;
    }
    return result;
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
  result.ort_evidence = active_ort_evidence_;
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
  last_copy_result_ =
      config_.driver->f().cuMemcpy2DAsync(&copy, config_.owner_stream);
  const bool ok = last_copy_result_ == cuda::CUDA_SUCCESS;
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
  if (config_.driver->BindPreparedContext(prepared_context_identity_) !=
      cuda::CUDA_SUCCESS) {
    result.status = PreparedCudaFastDvdnetStatus::device_lost; return result;
  }
  work_pending_ = true;
  completion_recorded_ = false;
  const auto finish_failure = [&](PreparedCudaFastDvdnetStatus original,
                                  PreparedCudaFastDvdnetOperation operation) {
    result.status = original;
    result.failed_operation = operation;
    if (original == PreparedCudaFastDvdnetStatus::device_lost) {
      result.pending_work = true;
      return result;
    }
    ++result.transfers.synchronizations;
    const auto quiesced = QuiesceFrame();
    result.pending_work = quiesced != PreparedCudaFastDvdnetStatus::ready;
    if (quiesced != PreparedCudaFastDvdnetStatus::ready)
      result.status = quiesced;
    return result;
  };
  const bool reset = temporal.reset || reset_requested_;
  if (!cuda::kernels::LaunchFastDvdnetPreprocess(
          kernels_, input, *history_[0], *history_[1], *history_[2], noisy_,
          noise_map_, reset, sigma, nullptr)) {
    return finish_failure(cuda::IsCudaDeviceLossResult(kernels_.last_result)
                              ? PreparedCudaFastDvdnetStatus::device_lost
                              : PreparedCudaFastDvdnetStatus::preprocess_failed,
                          PreparedCudaFastDvdnetOperation::temporal_preprocess);
  }
  if (reset && (!CopyHistory(*history_[2], *history_[0]) ||
                !CopyHistory(*history_[2], *history_[1]))) {
    return finish_failure(cuda::IsCudaDeviceLossResult(last_copy_result_)
                              ? PreparedCudaFastDvdnetStatus::device_lost
                              : PreparedCudaFastDvdnetStatus::preprocess_failed,
                          PreparedCudaFastDvdnetOperation::temporal_preprocess);
  }
  result.ort = session_->Execute(inputs_.data(), inputs_.size(), outputs_.data(),
                                 outputs_.size(), active_ort_evidence_);
  if (!result.ort.ok()) {
    return finish_failure(
        result.ort.status == onnx::OrtPreparedRunStatus::provider_failure_quarantine
            ? PreparedCudaFastDvdnetStatus::device_lost
            : PreparedCudaFastDvdnetStatus::inference_failed,
        PreparedCudaFastDvdnetOperation::inference);
  }
  if (!cuda::kernels::LaunchFastDvdnetPostprocess(kernels_, denoised_, *output,
                                                  nullptr)) {
    return finish_failure(cuda::IsCudaDeviceLossResult(kernels_.last_result)
                              ? PreparedCudaFastDvdnetStatus::device_lost
                              : PreparedCudaFastDvdnetStatus::postprocess_failed,
                          PreparedCudaFastDvdnetOperation::postprocess);
  }
  const auto completion_record =
      config_.driver->f().cuEventRecord(completion_event_, config_.owner_stream);
  if (completion_record != cuda::CUDA_SUCCESS) {
    result.status = cuda::IsCudaDeviceLossResult(completion_record)
                        ? PreparedCudaFastDvdnetStatus::device_lost
                        : PreparedCudaFastDvdnetStatus::synchronization_failed;
    result.failed_operation = PreparedCudaFastDvdnetOperation::completion_wait;
    result.pending_work = true;
    ++result.transfers.synchronizations;
    return result;
  }
  completion_recorded_ = true;
  std::swap(history_[0], history_[1]);
  std::swap(history_[1], history_[2]);
  reset_requested_ = false;
  result.status = PreparedCudaFastDvdnetStatus::executed;
  result.output_valid = true;
  result.pending_work = true;
  return result;
}

PreparedCudaFastDvdnetStatus NativePreparedCudaFastDvdnetRuntime::ResetTemporalState(
    std::uint64_t handle) noexcept {
  if (!prepared_ || handle != handle_) return PreparedCudaFastDvdnetStatus::foreign_handle;
  reset_requested_ = true;
  return PreparedCudaFastDvdnetStatus::ready;
}

PreparedCudaFastDvdnetStatus
NativePreparedCudaFastDvdnetRuntime::QuiesceFrame() noexcept {
  if (!work_pending_) return PreparedCudaFastDvdnetStatus::ready;
  if (!completion_recorded_) {
    const auto recorded = config_.driver->f().cuEventRecord(
        completion_event_, config_.owner_stream);
    if (recorded != cuda::CUDA_SUCCESS)
      return cuda::IsCudaDeviceLossResult(recorded)
                 ? PreparedCudaFastDvdnetStatus::device_lost
                 : PreparedCudaFastDvdnetStatus::synchronization_failed;
    completion_recorded_ = true;
  }
  const auto query = config_.driver->f().cuEventQuery(completion_event_);
  if (query == cuda::CUDA_SUCCESS) {
    work_pending_ = false;
    completion_recorded_ = false;
    return PreparedCudaFastDvdnetStatus::ready;
  }
  if (cuda::IsCudaDeviceLossResult(query))
    return PreparedCudaFastDvdnetStatus::device_lost;
  return query == cuda::CUDA_ERROR_NOT_READY
             ? PreparedCudaFastDvdnetStatus::completion_pending
             : PreparedCudaFastDvdnetStatus::synchronization_failed;
}

bool NativePreparedCudaFastDvdnetRuntime::QuiescePending() noexcept {
  if (!work_pending_) return true;
  if (completion_recorded_) {
    const auto query = config_.driver->f().cuEventQuery(completion_event_);
    if (query == cuda::CUDA_SUCCESS) { work_pending_ = false; return true; }
    if (query == cuda::CUDA_ERROR_NOT_READY) return false;
  }
  if (!config_.driver->StreamSynchronize(config_.owner_stream, nullptr))
    return false;
  work_pending_ = false;
  completion_recorded_ = false;
  return true;
}

bool NativePreparedCudaFastDvdnetRuntime::Release() noexcept {
  if (work_pending_) return false;
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
    if (slot && !slot->owns_memory) slot.reset();
  }
  if (noisy_.owns_memory) ok = noisy_.Free(config_.driver, nullptr) && ok;
  if (noise_map_.owns_memory) ok = noise_map_.Free(config_.driver, nullptr) && ok;
  if (denoised_.owns_memory) ok = denoised_.Free(config_.driver, nullptr) && ok;
  if (ok) {
    prepared_ = false; handle_ = 0; config_ = {}; active_ort_evidence_ = {};
    prepared_context_identity_ = {}; tensor_names_ = {}; tensor_shapes_ = {};
    shutdown_requested_ = false; work_pending_ = false;
    completion_recorded_ = false; reset_requested_ = true;
    last_copy_result_ = cuda::CUDA_SUCCESS;
  }
  return ok;
}

void NativePreparedCudaFastDvdnetRuntime::FinishFailedPrepare(
    PreparedCudaFastDvdnetStatus status,
    PreparedCudaFastDvdnetRuntimePrepareResult *result) noexcept {
  result->status = status;
  if (Release()) return;
  handle_ = g_next_handle.fetch_add(1, std::memory_order_relaxed);
  if (handle_ == 0) handle_ = g_next_handle.fetch_add(1);
  prepared_ = true;
  shutdown_requested_ = true;
  result->status = PreparedCudaFastDvdnetStatus::completion_pending;
  result->runtime_handle = handle_;
}

void NativePreparedCudaFastDvdnetRuntime::Quarantine() noexcept {
  // Deliberately retain every owner when CUDA cannot prove quiescence.
  (void)session_.release();
  for (auto &slot : history_) (void)slot.release();
  noisy_.owns_memory = false;
  noise_map_.owns_memory = false;
  denoised_.owns_memory = false;
  completion_event_ = nullptr;
  kernels_ = {};
  prepared_ = false;
}

PreparedCudaFastDvdnetShutdownSnapshot
NativePreparedCudaFastDvdnetRuntime::Shutdown(std::uint64_t handle) noexcept {
  ++shutdown_attempts_;
  if (!prepared_ || handle != handle_)
    return {PreparedCudaFastDvdnetStatus::foreign_handle, shutdown_attempts_, true, false};
  shutdown_requested_ = true;
  if (session_) session_->RequestShutdown();
  if (!QuiescePending())
    return {PreparedCudaFastDvdnetStatus::completion_pending, shutdown_attempts_, true, false};
  const bool released = Release();
  return {released ? PreparedCudaFastDvdnetStatus::shutdown
                   : PreparedCudaFastDvdnetStatus::completion_pending,
          shutdown_attempts_, true, released};
}

PreparedCudaFastDvdnetRuntimeTelemetry
NativePreparedCudaFastDvdnetRuntime::Telemetry() const noexcept { return telemetry_; }

} // namespace studiocast::open_video
