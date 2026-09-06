#include "core/open_video/native_prepared_cuda_eye_contact_runtime.h"

#include <algorithm>
#include <vector>

namespace studiocast::open_video {
namespace {
class OrtAdapter final : public IPreparedEyeContactOrtSession {
public:
  OrtAdapter(std::unique_ptr<onnx::OrtSession> session,
             onnx::OrtSessionInfo info) noexcept
      : session_(std::move(session)), info_(std::move(info)) {}
  const onnx::OrtSessionInfo &info() const noexcept override { return info_; }
  onnx::OrtPreparedCudaRunEvidence evidence() const noexcept override {
    return session_->prepared_cuda_run_evidence();
  }
  bool PrepareBinding(const onnx::OrtSession::CudaBindingInput *in,
                      std::size_t ni,
                      const onnx::OrtSession::CudaBindingOutput *out,
                      std::size_t no) noexcept override {
    return session_->PrepareCudaIoBinding(0, in, ni, out, no, nullptr);
  }
  onnx::OrtPreparedRunResult
  Execute(const onnx::OrtSession::CudaBindingInput *in, std::size_t ni,
          const onnx::OrtSession::CudaBindingOutput *out, std::size_t no,
          const onnx::OrtPreparedCudaRunEvidence &expected) noexcept override {
    return session_->ExecutePreparedCudaIoBindingTyped(0, in, ni, out, no,
                                                       expected);
  }
  void RequestShutdown() noexcept override { session_->RequestShutdown(); }

private:
  std::unique_ptr<onnx::OrtSession> session_;
  onnx::OrtSessionInfo info_{};
};
class OrtFactory final : public IPreparedEyeContactOrtFactory {
public:
  std::unique_ptr<IPreparedEyeContactOrtSession>
  Create(const std::filesystem::path &path,
         const onnx::OrtSessionOptions &options) noexcept override {
    onnx::OrtSessionInfo info;
    auto session = onnx::OrtSession::Create(path, options, &info, nullptr);
    if (!session || !ValidatePreparedEyeContactSessionMetadata(info) ||
        !info.strict_provider_validation_passed ||
        !info.prepared_provider_evidence.ValidStrictCuda())
      return {};
    return std::make_unique<OrtAdapter>(std::move(session), std::move(info));
  }
};
bool SameBinding(const PreparedCudaEyeContactBinding &a,
                 const PreparedCudaEyeContactBinding &b) noexcept {
  return a.driver == b.driver && a.context == b.context &&
         a.owner_context == b.owner_context && a.stream == b.stream &&
         a.stream_identity == b.stream_identity &&
         a.owner_identity == b.owner_identity &&
         a.configuration_generation == b.configuration_generation &&
         a.model_generation == b.model_generation;
}
} // namespace

bool ValidatePreparedEyeContactSessionMetadata(
    const onnx::OrtSessionInfo &info) noexcept {
  const std::vector<std::string> names{
      PreparedCudaEyeModelContract::kEyeInput,
      PreparedCudaEyeModelContract::kAnchorsInput,
      PreparedCudaEyeModelContract::kAnglesInput};
  const std::vector<std::vector<std::int64_t>> shapes{
      {1, 3, 48, 64}, {1, 12, 48, 64}, {1, 2}};
  return info.input_names == names &&
         info.output_names ==
             std::vector<std::string>{PreparedCudaEyeModelContract::kOutput} &&
         info.input_shapes == shapes &&
         info.output_shapes ==
             std::vector<std::vector<std::int64_t>>{{1, 3, 48, 64}} &&
         info.input_elem_types == std::vector<int>{1, 1, 1} &&
         info.output_elem_types == std::vector<int>{1};
}

NativePreparedCudaEyeContactRuntime::NativePreparedCudaEyeContactRuntime(
    IPreparedEyeContactOrtFactory *factory) noexcept
    : factory_(factory) {
  if (!factory_) {
    owned_factory_ = std::make_unique<OrtFactory>();
    factory_ = owned_factory_.get();
  }
}
NativePreparedCudaEyeContactRuntime::~NativePreparedCudaEyeContactRuntime() {
  (void)Shutdown();
}

PreparedCudaEyeContactStatus NativePreparedCudaEyeContactRuntime::Prepare(
    const PreparedCudaEyeContactRuntimeConfig &c) noexcept {
  if (prepared_ && Shutdown() == PreparedCudaEyeContactStatus::shutdown_failed)
    return PreparedCudaEyeContactStatus::shutdown_failed;
  if (!factory_ || !c.binding.driver || !c.binding.context.Valid() ||
      !c.binding.owner_context || !c.binding.stream || c.cuda_device_id < 0 ||
      c.frame_width <= 0 || c.frame_height <= 0 || c.models.left_path.empty() ||
      c.models.right_path.empty() || !c.models.left.Valid() ||
      !c.models.right.Valid())
    return PreparedCudaEyeContactStatus::invalid_setup;
  constexpr std::size_t kEye = 3u * 48u * 64u * sizeof(float);
  constexpr std::size_t kAnchors = 12u * 48u * 64u * sizeof(float);
  constexpr std::size_t kAngles = 2u * sizeof(float);
  const std::size_t frame = static_cast<std::size_t>(c.frame_width) *
                            static_cast<std::size_t>(c.frame_height) * 3u;
  const std::size_t required =
      2u * (2u * kEye + kAnchors + kAngles) + 10u * sizeof(float) + frame;
  if (required > c.resource_budget_bytes)
    return PreparedCudaEyeContactStatus::resource_exhausted;
  if (c.binding.driver->BindPreparedContext(c.binding.context) !=
      cuda::CUDA_SUCCESS)
    return PreparedCudaEyeContactStatus::stale_binding;
  config_ = c;
  std::string error;
  bool ok =
      crop_geometry_.AllocateNchwF32(c.binding.driver, 1, 2, 1, 5, &error) &&
      output_.Allocate(c.binding.driver, c.frame_width, c.frame_height,
                       cuda::PixelFormatGpu::rgb_u8, &error);
  for (std::size_t i = 0; ok && i < 2; ++i)
    ok = eye_rgb_[i].AllocateNchwF32(c.binding.driver, 1, 3, 48, 64, &error) &&
         anchors_[i].AllocateNchwF32(c.binding.driver, 1, 12, 48, 64, &error) &&
         angles_[i].AllocateNchwF32(c.binding.driver, 1, 2, 1, 1, &error) &&
         model_rgb_[i].AllocateNchwF32(c.binding.driver, 1, 3, 48, 64, &error);
  ok = ok && cuda::kernels::PrepareEyeContactKernels(
                 *c.binding.driver, c.binding.owner_context, c.binding.stream,
                 &kernels_, &error);
  if (!ok) {
    (void)Release();
    return PreparedCudaEyeContactStatus::resource_exhausted;
  }
  names_ = {PreparedCudaEyeModelContract::kEyeInput,
            PreparedCudaEyeModelContract::kAnchorsInput,
            PreparedCudaEyeModelContract::kAnglesInput,
            PreparedCudaEyeModelContract::kOutput};
  shapes_ = {{{1, 3, 48, 64}, {1, 12, 48, 64}, {1, 2, 1, 1}, {1, 3, 48, 64}}};
  onnx::OrtSessionOptions options;
  options.provider_policy = onnx::OrtProviderPolicy::strict_cuda_only;
  options.prefer_cuda = true;
  options.enable_tensorrt = false;
  options.cuda_device_id = c.cuda_device_id;
  options.cuda_context_generation = c.binding.context.logical_generation;
  options.user_compute_stream = c.binding.stream;
  const std::array<std::filesystem::path, 2> paths{c.models.left_path,
                                                   c.models.right_path};
  for (std::size_t i = 0; i < 2; ++i) {
    sessions_[i] = factory_->Create(paths[i], options);
    if (!sessions_[i] ||
        !ValidatePreparedEyeContactSessionMetadata(sessions_[i]->info()) ||
        !onnx::OrtPreparedProviderEvidenceMatches(
            sessions_[i]->evidence().provider, c.cuda_device_id,
            c.binding.context.logical_generation,
            reinterpret_cast<std::uintptr_t>(c.binding.stream))) {
      (void)Release();
      return PreparedCudaEyeContactStatus::provider_mismatch;
    }
    evidence_[i] = sessions_[i]->evidence();
    inputs_[i][0] = {names_[0].c_str(),
                     reinterpret_cast<const float *>(eye_rgb_[i].ptr),
                     eye_rgb_[i].ElementCount(), shapes_[0].data(), 4};
    inputs_[i][1] = {names_[1].c_str(),
                     reinterpret_cast<const float *>(anchors_[i].ptr),
                     anchors_[i].ElementCount(), shapes_[1].data(), 4};
    inputs_[i][2] = {names_[2].c_str(),
                     reinterpret_cast<const float *>(angles_[i].ptr),
                     angles_[i].ElementCount(), shapes_[2].data(), 2};
    outputs_[i][0] = {names_[3].c_str(),
                      reinterpret_cast<float *>(model_rgb_[i].ptr),
                      model_rgb_[i].ElementCount(), shapes_[3].data(), 4};
    if (!sessions_[i]->PrepareBinding(inputs_[i].data(), 3, outputs_[i].data(),
                                      1)) {
      (void)Release();
      return PreparedCudaEyeContactStatus::graph_mismatch;
    }
  }
  // Warm both exact bindings while setup owns all failure and allocation work.
  for (std::size_t i = 0; i < 2; ++i) {
    if (!c.binding.driver->MemsetD8Async(eye_rgb_[i].ptr, 0, eye_rgb_[i].bytes,
                                         c.binding.stream, nullptr) ||
        !c.binding.driver->MemsetD8Async(anchors_[i].ptr, 0, anchors_[i].bytes,
                                         c.binding.stream, nullptr) ||
        !c.binding.driver->MemsetD8Async(angles_[i].ptr, 0, angles_[i].bytes,
                                         c.binding.stream, nullptr) ||
        !sessions_[i]
             ->Execute(inputs_[i].data(), 3, outputs_[i].data(), 1,
                       evidence_[i])
             .ok()) {
      work_pending_ = true;
      return PreparedCudaEyeContactStatus::inference_failed;
    }
  }
  work_pending_ = true;
  if (!c.binding.driver->StreamSynchronize(c.binding.stream, nullptr))
    return PreparedCudaEyeContactStatus::synchronization_failed;
  work_pending_ = false;
  prepared_ = true;
  return PreparedCudaEyeContactStatus::executed;
}

PreparedCudaEyeContactRuntimeFrameResult
NativePreparedCudaEyeContactRuntime::Execute(
    const PreparedCudaEyeContactBinding &binding, const cuda::CudaImage &input,
    const cuda::CudaTensor &landmarks, float strength) noexcept {
  PreparedCudaEyeContactRuntimeFrameResult out;
  if (!prepared_ || shutdown_requested_) {
    out.status = PreparedCudaEyeContactStatus::shutdown;
    return out;
  }
  if (!SameBinding(binding, config_.binding) ||
      input.w != config_.frame_width || input.h != config_.frame_height ||
      input.format != cuda::PixelFormatGpu::rgb_u8 || !landmarks.Valid() ||
      landmarks.n != 1 || landmarks.c != 1 || landmarks.h != 1 ||
      landmarks.w != 136) {
    out.status = PreparedCudaEyeContactStatus::stale_binding;
    return out;
  }
  cuda::kernels::PreparedEyeContactInputs prepared{
      &landmarks,   &crop_geometry_, &eye_rgb_[0], &eye_rgb_[1],
      &anchors_[0], &anchors_[1],    &angles_[0],  &angles_[1]};
  if (!cuda::kernels::LaunchEyeContactInputsPrepared(kernels_, input, prepared,
                                                     strength, nullptr)) {
    out.status = PreparedCudaEyeContactStatus::kernel_failed;
    return out;
  }
  std::array<bool, 2> succeeded{};
  work_pending_ = true;
  for (std::size_t i = 0; i < 2; ++i) {
    ++out.inference_attempts;
    const auto run = sessions_[i]->Execute(inputs_[i].data(), 3,
                                           outputs_[i].data(), 1, evidence_[i]);
    succeeded[i] = run.ok();
    if (succeeded[i])
      ++out.inference_successes;
    if (!succeeded[i]) {
      // A failed second session must not expose the successful left-eye write.
      if (!binding.driver->StreamSynchronize(binding.stream, nullptr)) {
        out.status = PreparedCudaEyeContactStatus::synchronization_failed;
        out.pending_work = true;
        return out;
      }
      work_pending_ = false;
      switch (run.status) {
      case onnx::OrtPreparedRunStatus::provider_unavailable:
      case onnx::OrtPreparedRunStatus::provider_mismatch:
      case onnx::OrtPreparedRunStatus::cpu_fallback_forbidden:
      case onnx::OrtPreparedRunStatus::tensorrt_forbidden:
        out.status = PreparedCudaEyeContactStatus::provider_mismatch;
        break;
      case onnx::OrtPreparedRunStatus::wrong_device:
      case onnx::OrtPreparedRunStatus::stale_context_generation:
      case onnx::OrtPreparedRunStatus::provider_failure_quarantine:
        out.status = PreparedCudaEyeContactStatus::device_lost;
        out.device_lost = true;
        out.output = nullptr;
        return out;
      default:
        out.status = PreparedCudaEyeContactStatus::inference_failed;
        break;
      }
      out.output = &input;
      return out;
    }
  }
  if (!cuda::kernels::LaunchEyeContactCompositePrepared(
          kernels_, input, crop_geometry_, model_rgb_[0], model_rgb_[1], true,
          true, strength, &output_, nullptr)) {
    out.status = PreparedCudaEyeContactStatus::kernel_failed;
    out.pending_work = true;
    return out;
  }
  out.status = PreparedCudaEyeContactStatus::executed;
  out.output = &output_;
  out.pending_work = true;
  return out;
}

bool NativePreparedCudaEyeContactRuntime::Release() noexcept {
  if (work_pending_)
    return false;
  bool ok = true;
  sessions_ = {};
  if (kernels_.live)
    ok = cuda::kernels::DestroyEyeContactKernelsPrepared(&kernels_, nullptr) &&
         ok;
  if (output_.owns_memory)
    ok = output_.Free(config_.binding.driver, nullptr) && ok;
  if (crop_geometry_.owns_memory)
    ok = crop_geometry_.Free(config_.binding.driver, nullptr) && ok;
  for (std::size_t i = 0; i < 2; ++i) {
    if (eye_rgb_[i].owns_memory)
      ok = eye_rgb_[i].Free(config_.binding.driver, nullptr) && ok;
    if (anchors_[i].owns_memory)
      ok = anchors_[i].Free(config_.binding.driver, nullptr) && ok;
    if (angles_[i].owns_memory)
      ok = angles_[i].Free(config_.binding.driver, nullptr) && ok;
    if (model_rgb_[i].owns_memory)
      ok = model_rgb_[i].Free(config_.binding.driver, nullptr) && ok;
  }
  if (ok) {
    config_ = {};
    prepared_ = false;
    shutdown_requested_ = false;
    evidence_ = {};
  }
  return ok;
}

PreparedCudaEyeContactStatus
NativePreparedCudaEyeContactRuntime::Shutdown() noexcept {
  if (!prepared_ && !work_pending_)
    return PreparedCudaEyeContactStatus::shutdown;
  shutdown_requested_ = true;
  for (auto &session : sessions_)
    if (session)
      session->RequestShutdown();
  if (work_pending_) {
    if (!config_.binding.driver || !config_.binding.driver->StreamSynchronize(
                                       config_.binding.stream, nullptr))
      return PreparedCudaEyeContactStatus::shutdown_failed;
    work_pending_ = false;
  }
  return Release() ? PreparedCudaEyeContactStatus::shutdown
                   : PreparedCudaEyeContactStatus::shutdown_failed;
}
} // namespace studiocast::open_video
