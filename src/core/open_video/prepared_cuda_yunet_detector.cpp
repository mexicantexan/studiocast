#include "core/open_video/prepared_cuda_yunet_detector.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/cuda/cuda_tensor.h"
#include "core/cuda/kernels/prepared_preprocess_kernel_bundle.h"
#include "core/onnx/ort_session.h"
#include "core/open_video/yunet_face_detector.h"
#include "core/util/fs.h"
#include "core/util/json.h"

namespace studiocast::open_video {
namespace {
constexpr std::size_t kOutputs = 12;
constexpr std::size_t kTopK = 5000;
std::atomic<std::uint64_t> g_cache_revision{1};
const std::array<const char *, kOutputs> kNames = {
    "cls_8",  "cls_16",  "cls_32",  "obj_8", "obj_16", "obj_32",
    "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32"};

bool SameBinding(const PreparedCudaYunetBinding &a,
                 const PreparedCudaYunetBinding &b) noexcept {
  return a.driver == b.driver && a.context == b.context &&
         a.stream == b.stream && a.stream_identity == b.stream_identity &&
         a.owner_identity == b.owner_identity &&
         a.configuration_generation == b.configuration_generation;
}
int Stride(std::string_view name) {
  if (name.ends_with("_8"))
    return 8;
  if (name.ends_with("_16"))
    return 16;
  return 32;
}
int Channels(std::string_view name) {
  if (name.starts_with("bbox_"))
    return 4;
  if (name.starts_with("kps_"))
    return 10;
  return 1;
}

bool ReadDeclaredInput(const std::filesystem::path &manifest,
                       ModelTensorSpec *out, std::string *error) {
  const auto text = util::ReadTextFile(manifest.string());
  util::json::Value root;
  std::string parse_error;
  if (!text || !util::json::Parse(*text, &root, &parse_error)) {
    if (error)
      *error = "YuNet validated manifest cannot be parsed";
    return false;
  }
  const auto *top = root.AsObject();
  if (!top)
    return false;
  const auto onnx_it = top->find("onnx");
  const auto *onnx =
      onnx_it == top->end() ? nullptr : onnx_it->second.AsObject();
  if (!onnx)
    return false;
  const auto inputs_it = onnx->find("inputs");
  const auto *inputs =
      inputs_it == onnx->end() ? nullptr : inputs_it->second.AsArray();
  if (!inputs || inputs->size() != 1)
    return false;
  const auto *input = (*inputs)[0].AsObject();
  if (!input)
    return false;
  const auto string_field = [&](const char *name, std::string *value) {
    const auto it = input->find(name);
    const auto *found = it == input->end() ? nullptr : it->second.AsString();
    if (!found)
      return false;
    *value = *found;
    return true;
  };
  const auto int_field = [&](const char *name, int *value) {
    const auto it = input->find(name);
    const auto *found = it == input->end() ? nullptr : it->second.AsNumber();
    if (!found || !std::isfinite(*found) || std::floor(*found) != *found ||
        *found < 1 || *found > 2048)
      return false;
    *value = static_cast<int>(*found);
    return true;
  };
  if (!string_field("name", &out->name) ||
      !string_field("layout", &out->layout) ||
      !string_field("dtype", &out->dtype) || !int_field("width", &out->width) ||
      !int_field("height", &out->height) ||
      !int_field("channels", &out->channels) || out->layout != "nchw" ||
      out->dtype != "float32" || out->channels != 3) {
    if (error)
      *error = "YuNet manifest requires one fixed NCHW float32 input";
    return false;
  }
  return true;
}

PreparedCudaYunetStatus
CompletionStatus(cuda::CudaPreparedTransferCompletion::Status status) noexcept {
  using Status = cuda::CudaPreparedTransferCompletion::Status;
  switch (status) {
  case Status::timeout:
    return PreparedCudaYunetStatus::timeout;
  case Status::stop_requested:
    return PreparedCudaYunetStatus::stop_requested;
  case Status::cancelled:
    return PreparedCudaYunetStatus::cancelled;
  case Status::device_lost:
    return PreparedCudaYunetStatus::device_lost;
  default:
    return PreparedCudaYunetStatus::completion_failed;
  }
}

bool IsDeviceLost(cuda::CudaPreparedTransferCompletion::Status status) {
  return status == cuda::CudaPreparedTransferCompletion::Status::device_lost;
}

class OrtYunetSession final : public internal::IPreparedCudaYunetSession {
public:
  OrtYunetSession(std::unique_ptr<onnx::OrtSession> session,
                  onnx::OrtSessionInfo info)
      : session_(std::move(session)), info_(std::move(info)) {}
  const onnx::OrtSessionInfo &info() const noexcept override { return info_; }
  bool Prepare(const onnx::OrtSession::CudaBindingInput *inputs, std::size_t ni,
               const onnx::OrtSession::CudaBindingOutput *outputs,
               std::size_t no, std::string *error) override {
    session_->ReserveRunScratch(ni, no);
    return session_->PrepareCudaIoBinding(0, inputs, ni, outputs, no, error);
  }
  onnx::OrtPreparedCudaRunEvidence evidence() const noexcept override {
    return session_->prepared_cuda_run_evidence();
  }
  onnx::OrtPreparedRunResult
  Execute(const onnx::OrtSession::CudaBindingInput *inputs, std::size_t ni,
          const onnx::OrtSession::CudaBindingOutput *outputs, std::size_t no,
          const onnx::OrtPreparedCudaRunEvidence &evidence) noexcept override {
    return session_->ExecutePreparedCudaIoBindingTyped(0, inputs, ni, outputs,
                                                       no, evidence);
  }
  void Invalidate() noexcept override {
    session_->InvalidatePreparedBindings();
  }

private:
  std::unique_ptr<onnx::OrtSession> session_;
  onnx::OrtSessionInfo info_;
};
} // namespace

struct PreparedCudaYunetDetector::Impl {
  struct Output {
    std::string name;
    std::vector<int64_t> shape;
    bool channels_first = false;
    cuda::CudaTensor tensor;
    std::size_t host_offset = 0;
  };
  PreparedCudaYunetSetup setup;
  ValidatedModelArtifactIdentity artifact;
  std::unique_ptr<internal::IPreparedCudaYunetSession> session;
  onnx::OrtPreparedCudaRunEvidence run_evidence;
  cuda::CudaTensor input;
  std::string input_name;
  std::array<int64_t, 4> input_shape{};
  int input_w = 0;
  int input_h = 0;
  std::array<Output, kOutputs> outputs;
  std::array<onnx::OrtSession::CudaBindingInput, 1> ort_input{};
  std::array<onnx::OrtSession::CudaBindingOutput, kOutputs> ort_outputs{};
  cuda::kernels::PreparedPreprocessKernelBundle preprocess;
  cuda::CudaPreparedTransferCompletion completion;
  YunetDetectionScratch scratch;
  bool ready = false;
  bool stopped = false;
  bool warmup_pending = false;
  FrameAnalysisCache *prepared_cache = nullptr;
  std::uint64_t cache_revision = 0;
  ~Impl() {
    // Live prepared generations require explicit after-quiescence Shutdown().
    // Candidate setup rollback is safe here on the setup thread/context.
    if (ready && !stopped)
      return;
    if (session)
      session->Invalidate();
    if (setup.binding.driver) {
      std::string ignored;
      for (auto &output : outputs)
        (void)output.tensor.Free(setup.binding.driver, &ignored);
      (void)input.Free(setup.binding.driver, &ignored);
    }
  }
};

PreparedCudaYunetDetector::PreparedCudaYunetDetector()
    : impl_(std::make_unique<Impl>()) {}
PreparedCudaYunetDetector::~PreparedCudaYunetDetector() {
  if (!impl_)
    return;
  std::string ignored;
  if (!Shutdown(&ignored)) {
    // The owner violated the explicit after-quiescence shutdown contract.  Do
    // not destroy ORT or CUDA storage while submitted work may still refer to
    // it.  Context teardown reclaims this quarantined generation.
    (void)impl_.release();
  }
}
PreparedCudaYunetDetector::PreparedCudaYunetDetector(
    PreparedCudaYunetDetector &&) noexcept = default;

bool PreparedCudaYunetDetector::Prepare(const PreparedCudaYunetSetup &setup,
                                        std::string *error,
                                        PreparedCudaYunetStatus *status_out) {
  if (status_out)
    *status_out = PreparedCudaYunetStatus::invalid_setup;
  if (error)
    error->clear();
  if (impl_ && impl_->warmup_pending) {
    if (error)
      *error = "invalid prepared YuNet setup";
    if (status_out)
      *status_out = PreparedCudaYunetStatus::shutdown_failed;
    return false;
  }
  if (!impl_ || impl_->ready || impl_->warmup_pending ||
      !setup.binding.driver || !setup.binding.context.Valid() ||
      !setup.binding.stream || setup.binding.stream_identity == 0 ||
      setup.binding.owner_identity == 0 ||
      setup.binding.configuration_generation == 0 || setup.frame_width <= 0 ||
      setup.frame_height <= 0) {
    if (error)
      *error = "invalid prepared YuNet setup";
    return false;
  }
  std::optional<ModelPack> discovered;
  const ModelPack *pack = setup.selected_pack;
  if (pack && (pack->task != "face_detection" ||
               (!setup.requested_model_id.empty() && pack->id != setup.requested_model_id))) {
    if (error) *error = "selected face_detection pack does not match requested task and model ID";
    if (status_out) *status_out = PreparedCudaYunetStatus::model_invalid;
    return false;
  }
  if (!pack) {
    auto registry = setup.models_root.empty()
                        ? ModelPackRegistry::ScanDefault()
                        : ModelPackRegistry::Scan(setup.models_root);
    const auto selected = setup.requested_model_id.empty()
        ? registry.DefaultModelIdForTask("face_detection") : setup.requested_model_id;
    discovered = registry.Find("face_detection", selected);
    if (discovered) pack = &*discovered;
  }
  if (!pack) {
    if (error)
      *error = "requested YuNet model is not installed";
    if (status_out)
      *status_out = PreparedCudaYunetStatus::model_invalid;
    return false;
  }
  const ModelFile *main = nullptr;
  for (const auto &file : pack->files)
    if (file.kind == "onnx" && file.role == "main") {
      main = &file;
      break;
    }
  if (!main) {
    if (error)
      *error = "YuNet main ONNX artifact is missing";
    if (status_out)
      *status_out = PreparedCudaYunetStatus::model_invalid;
    return false;
  }
  auto validated_setup = setup;
  if (!ReadDeclaredInput(pack->manifest_path, &validated_setup.declared_input,
                         error)) {
    if (error && error->empty())
      *error = "YuNet validated manifest input is missing";
    if (status_out)
      *status_out = PreparedCudaYunetStatus::model_invalid;
    return false;
  }
  // Validation follows parsing so its manifest checksum proves that the
  // frozen tensor declaration came from this registry generation.
  const auto validation = ValidateModelArtifactIdentity(*pack, *main);
  if (!validation.ok() ||
      (setup.selected_pack && !(validation.identity == setup.selected_artifact))) {
    if (error)
      *error = validation.detail;
    if (status_out)
      *status_out = PreparedCudaYunetStatus::model_invalid;
    return false;
  }

  onnx::OrtSessionOptions options;
  options.provider_policy = onnx::OrtProviderPolicy::strict_cuda_only;
  options.prefer_cuda = true;
  options.enable_tensorrt = false;
  options.cuda_device_id = setup.cuda_device_id;
  options.cuda_context_generation = setup.binding.context.logical_generation;
  options.user_compute_stream = setup.binding.stream;
  onnx::OrtSessionInfo info;
  auto session = onnx::OrtSession::Create(main->path, options, &info, error);
  if (!session || !info.strict_provider_validation_passed ||
      !onnx::OrtPreparedProviderEvidenceMatches(
          info.prepared_provider_evidence, setup.cuda_device_id,
          setup.binding.context.logical_generation,
          reinterpret_cast<std::uintptr_t>(setup.binding.stream)) ||
      info.input_names.size() != 1 || info.input_shapes.size() != 1 ||
      info.output_names.size() != kOutputs) {
    if (error && error->empty())
      *error = "YuNet strict CUDA graph/provider evidence mismatch";
    if (status_out)
      *status_out = !session || !info.strict_provider_validation_passed
                        ? PreparedCudaYunetStatus::provider_mismatch
                        : PreparedCudaYunetStatus::graph_mismatch;
    return false;
  }
  return PrepareValidated(
      validated_setup, validation.identity,
      std::make_unique<OrtYunetSession>(std::move(session), std::move(info)),
      error, status_out);
}

bool PreparedCudaYunetDetector::PrepareForTesting(
    const PreparedCudaYunetSetup &setup,
    const ValidatedModelArtifactIdentity &validated_artifact,
    std::unique_ptr<internal::IPreparedCudaYunetSession> session,
    std::string *error, PreparedCudaYunetStatus *status_out) {
  return PrepareValidated(setup, validated_artifact, std::move(session), error,
                          status_out);
}

bool PreparedCudaYunetDetector::PrepareValidated(
    const PreparedCudaYunetSetup &setup,
    const ValidatedModelArtifactIdentity &validated_artifact,
    std::unique_ptr<internal::IPreparedCudaYunetSession> session,
    std::string *error, PreparedCudaYunetStatus *status_out) {
  if (status_out)
    *status_out = PreparedCudaYunetStatus::invalid_setup;
  if (error)
    error->clear();
  if (!impl_ || impl_->ready || impl_->warmup_pending ||
      !validated_artifact.Valid() || !session) {
    if (error)
      *error = "invalid prepared YuNet validated setup";
    return false;
  }
  const auto &info = session->info();
  const auto &declared = setup.declared_input;
  if (status_out)
    *status_out = PreparedCudaYunetStatus::graph_mismatch;
  if (declared.name.empty() || declared.layout != "nchw" ||
      declared.dtype != "float32" || declared.channels != 3 ||
      declared.width <= 0 || declared.height <= 0 || declared.width > 2048 ||
      declared.height > 2048 || info.input_shapes.size() != 1 ||
      info.input_shapes[0].size() != 4) {
    if (error)
      *error = "YuNet requires fixed bounded NCHW input geometry";
    return false;
  }
  const int input_h = declared.height;
  const int input_w = declared.width;
  if (setup.resource_budget_bytes < RequiredResourceBytes(input_w, input_h)) {
    if (error)
      *error = "YuNet prepared resource budget exceeded";
    if (status_out)
      *status_out = PreparedCudaYunetStatus::resource_exhausted;
    return false;
  }
  if (!setup.binding.driver || !setup.binding.context.Valid() ||
      !setup.binding.stream || setup.binding.stream_identity == 0 ||
      setup.binding.owner_identity == 0 ||
      setup.binding.configuration_generation == 0 || setup.frame_width <= 0 ||
      setup.frame_height <= 0 ||
      !info.strict_provider_validation_passed ||
      !onnx::OrtPreparedProviderEvidenceMatches(
          info.prepared_provider_evidence, setup.cuda_device_id,
          setup.binding.context.logical_generation,
          reinterpret_cast<std::uintptr_t>(setup.binding.stream)) ||
      info.input_names.size() != 1 || info.input_shapes.size() != 1 ||
      info.input_elem_types.size() != 1 || info.input_elem_types[0] != 1 ||
      info.input_names[0] != declared.name ||
      info.output_names.size() != kOutputs) {
    if (error)
      *error = "YuNet strict CUDA graph/provider evidence mismatch";
    if (status_out)
      *status_out = !info.strict_provider_validation_passed
                        ? PreparedCudaYunetStatus::provider_mismatch
                        : PreparedCudaYunetStatus::graph_mismatch;
    return false;
  }
  std::string geometry_error;
  if (!ValidateYunetInputTensorContract(info.input_shapes[0], false, input_w,
                                        input_h, &geometry_error)) {
    if (error)
      *error = geometry_error;
    return false;
  }
  if (status_out)
    *status_out = PreparedCudaYunetStatus::invalid_setup;
  auto candidate = std::make_unique<Impl>();
  candidate->setup = setup;
  candidate->cache_revision =
      g_cache_revision.fetch_add(1, std::memory_order_relaxed);
  candidate->artifact = validated_artifact;
  candidate->input_w = input_w;
  candidate->input_h = input_h;
  candidate->input_shape = {1, 3, input_h, input_w};
  candidate->session = std::move(session);
  if (!candidate->input.AllocateNchwF32(setup.binding.driver, 1, 3, input_h,
                                        input_w, error)) {
    return false;
  }
  std::size_t host_offset = 0;
  for (std::size_t index = 0; index < kOutputs; ++index) {
    const auto found = std::find(info.output_names.begin(),
                                 info.output_names.end(), kNames[index]);
    if (found == info.output_names.end()) {
      if (error)
        *error = "YuNet expected output is missing";
      return false;
    }
    const auto graph_index = static_cast<std::size_t>(
        std::distance(info.output_names.begin(), found));
    const int stride = Stride(kNames[index]);
    const int channels = Channels(kNames[index]);
    const int rows = input_h / stride;
    const int cols = input_w / stride;
    const int anchors = rows * cols;
    if (graph_index >= info.output_shapes.size()) {
      if (error)
        *error = "YuNet output tensor shape mismatch";
      return false;
    }
    auto &out = candidate->outputs[index];
    const auto &declared_output = info.output_shapes[graph_index];
    const auto matches = [](int64_t value, int expected) {
      return value <= 0 || value == expected;
    };
    if (declared_output.size() == 3 && matches(declared_output[0], 1) &&
        matches(declared_output[1], anchors) &&
        matches(declared_output[2], channels)) {
      out.shape = {1, anchors, channels};
    } else if (declared_output.size() == 4 && matches(declared_output[0], 1) &&
               matches(declared_output[1], channels) &&
               matches(declared_output[2], rows) &&
               matches(declared_output[3], cols) && declared_output[1] > 0) {
      out.shape = {1, channels, rows, cols};
      out.channels_first = true;
    } else if (declared_output.size() == 4 && matches(declared_output[0], 1) &&
               matches(declared_output[1], rows) &&
               matches(declared_output[2], cols) &&
               matches(declared_output[3], channels) &&
               declared_output[3] > 0) {
      out.shape = {1, rows, cols, channels};
    } else {
      if (error)
        *error = "YuNet output tensor shape mismatch";
      return false;
    }
    out.name = kNames[index];
    out.host_offset = host_offset;
    if (!out.tensor.AllocateNchwF32(setup.binding.driver, 1, 1, 1,
                                    anchors * channels, error)) {
      return false;
    }
    host_offset += out.tensor.bytes;
    candidate->ort_outputs[index] = {
        out.name.c_str(), reinterpret_cast<float *>(out.tensor.ptr),
        out.tensor.ElementCount(), out.shape.data(), out.shape.size()};
  }
  candidate->input_name = info.input_names[0];
  candidate->ort_input[0] = {
      candidate->input_name.c_str(),
      reinterpret_cast<const float *>(candidate->input.ptr),
      candidate->input.ElementCount(), candidate->input_shape.data(),
      candidate->input_shape.size()};
  // Names/shapes referenced by bindings are retained by the candidate->
  if (!candidate->session->Prepare(candidate->ort_input.data(), 1,
                                   candidate->ort_outputs.data(), kOutputs,
                                   error))
    return false;
  candidate->run_evidence = candidate->session->evidence();
  if (!setup.binding.driver->MemsetD8Async(candidate->input.ptr, 0,
                                           candidate->input.bytes,
                                           setup.binding.stream, error))
    return false;
  const auto warmup = candidate->session->Execute(
      candidate->ort_input.data(), 1, candidate->ort_outputs.data(), kOutputs,
      candidate->run_evidence);
  // Execute may have submitted part of a graph before reporting failure.  The
  // setup candidate cannot release bindings or tensors until the owner stream
  // is known quiescent.
  const bool warmup_quiesced =
      setup.binding.driver->StreamSynchronize(setup.binding.stream, error);
  if (!warmup.ok() || !warmup_quiesced) {
    if (status_out)
      *status_out = !warmup_quiesced
                        ? PreparedCudaYunetStatus::shutdown_failed
                        : PreparedCudaYunetStatus::inference_failed;
    if (error && error->empty())
      *error = "YuNet strict CUDA warmup failed";
    if (!warmup_quiesced) {
      candidate->warmup_pending = true;
      impl_ = std::move(candidate);
    }
    return false;
  }
  cuda::kernels::PreparedPreprocessBinding preprocess_binding{
      setup.binding.driver,
      setup.binding.context,
      setup.binding.stream,
      setup.binding.stream_identity,
      setup.binding.configuration_generation,
      validated_artifact.registry_generation};
  cuda::kernels::ModelPreprocessSpec spec;
  spec.dst_w = input_w;
  spec.dst_h = input_h;
  spec.dst_order = cuda::kernels::ChannelOrder::bgr;
  // OpenCV YuNet consumes bilinearly sampled BGR float values in [0,255].
  // The shared prepared kernel first scales bytes by 1/255, so this reciprocal
  // std preserves the published detector's raw-float input contract.
  std::fill(std::begin(spec.std), std::end(spec.std), 1.0F / 255.0F);
  if (!candidate->preprocess.Prepare(
          preprocess_binding, spec,
          cuda::kernels::PreparedPreprocessKernelBundle::
              RequiredResourceBytes(),
          error))
    return false;
  cuda::CudaPreparedTransferCompletion::PrepareOptions completion_options;
  completion_options.binding = {setup.binding.context, setup.binding.stream,
                                setup.binding.stream_identity,
                                setup.binding.owner_identity,
                                setup.binding.configuration_generation};
  completion_options.timed_stage_count = 1;
  completion_options.resident_work = true;
  completion_options.readback_count = kOutputs;
  completion_options.budget.max_pinned_host_bytes = host_offset;
  completion_options.budget.max_events = 3;
  for (std::size_t i = 0; i < kOutputs; ++i)
    completion_options.readbacks[i] = {candidate->outputs[i].tensor.ptr,
                                       candidate->outputs[i].tensor.bytes,
                                       candidate->outputs[i].host_offset,
                                       candidate->outputs[i].tensor.bytes, 1};
  if (!candidate->completion.Prepare(setup.binding.driver, completion_options)
           .ok()) {
    if (error)
      *error = "YuNet prepared completion setup failed";
    return false;
  }
  if (!candidate->scratch.Configure(kTopK))
    return false;
  candidate->ready = true;
  impl_ = std::move(candidate);
  if (status_out)
    *status_out = PreparedCudaYunetStatus::ready;
  return true;
}

bool PreparedCudaYunetDetector::PrepareCache(FrameAnalysisCache *cache,
                                             std::string *error) {
  if (error)
    error->clear();
  if (!cache || !ready()) {
    if (error)
      *error = "YuNet/cache is not prepared";
    return false;
  }
  (void)cache->PrepareFaceDetections(kTopK);
  impl_->prepared_cache = cache;
  return true;
}

bool PreparedCudaYunetDetector::ExportPreparedBinding(
    FrameAnalysisCache *cache, PreparedCudaYunetPreparedBinding *out) noexcept {
  if (out)
    *out = {};
  if (!out || !cache || !impl_ || !impl_->ready || !impl_->prepared_cache ||
      cache != impl_->prepared_cache)
    return false;
  *out = {this, impl_->setup.binding, impl_->artifact, impl_->cache_revision,
          cache, impl_->setup.frame_width, impl_->setup.frame_height};
  return true;
}

bool PreparedCudaYunetDetector::MatchesPreparedBinding(
    const PreparedCudaYunetPreparedBinding &binding) const noexcept {
  return impl_ && impl_->ready && binding.producer == this &&
         binding.cache == impl_->prepared_cache &&
         SameBinding(binding.binding, impl_->setup.binding) &&
         binding.artifact == impl_->artifact &&
         binding.producer_revision == impl_->cache_revision &&
         binding.frame_width == impl_->setup.frame_width &&
         binding.frame_height == impl_->setup.frame_height;
}

PreparedCudaYunetExecutionResult PreparedCudaYunetDetector::Execute(
    const PreparedCudaYunetBinding &binding, const cuda::CudaImage &source,
    std::uint64_t sequence, FrameAnalysisCache *cache,
    const cuda::CudaPreparedTransferCompletion::WaitPolicy &wait) noexcept {
  PreparedCudaYunetExecutionResult result;
  if (!ready() || !cache || cache != impl_->prepared_cache ||
      !SameBinding(binding, impl_->setup.binding) || !source.Valid() ||
      source.format != cuda::PixelFormatGpu::rgb_u8 ||
      source.w != impl_->setup.frame_width ||
      source.h != impl_->setup.frame_height)
    return result;
  cache->BeginFrame(sequence);
  const auto &provenance = cache->face_detections_provenance;
  if (provenance.complete && provenance.producer == this &&
      provenance.producer_revision == impl_->cache_revision &&
      cache->face_detections) {
    result.status = cache->face_detections->empty()
                        ? PreparedCudaYunetStatus::success_no_faces
                        : PreparedCudaYunetStatus::success_faces;
    return result;
  }
  const float scale = std::min(
      static_cast<float>(impl_->input_w) / static_cast<float>(source.w),
      static_cast<float>(impl_->input_h) / static_cast<float>(source.h));
  const int scaled_w = std::max(
      1, static_cast<int>(std::lround(static_cast<float>(source.w) * scale)));
  const int scaled_h = std::max(
      1, static_cast<int>(std::lround(static_cast<float>(source.h) * scale)));
  const cuda::kernels::LetterboxGeometry geometry{
      scaled_w, scaled_h, (impl_->input_w - scaled_w) / 2,
      (impl_->input_h - scaled_h) / 2, 0};
  const cuda::CudaPreparedTransferCompletion::Binding completion_binding{
      binding.context, binding.stream, binding.stream_identity,
      binding.owner_identity, binding.configuration_generation};
  if (!impl_->completion.BeginFrame(completion_binding, sequence).ok())
    return result;
  const auto submitted =
      impl_->completion.SubmitResidentWork(completion_binding);
  if (!submitted.ok()) {
    result.status = CompletionStatus(submitted.status);
    result.device_lost = IsDeviceLost(submitted.status);
    return result;
  }
  result.pending_work = true;
  const auto stage_begin =
      impl_->completion.RecordStageBegin(completion_binding, 0);
  if (!stage_begin.ok()) {
    result.status = CompletionStatus(stage_begin.status);
    result.device_lost = IsDeviceLost(stage_begin.status);
    const auto recorded =
        impl_->completion.RecordCompletion(completion_binding);
    if (recorded.ok()) {
      ++result.completion_waits;
      const auto completed =
          impl_->completion.WaitForCompletion(completion_binding, wait);
      result.pending_work = !completed.ok();
      result.device_lost = result.device_lost || IsDeviceLost(completed.status);
      if (!completed.ok())
        result.status = CompletionStatus(completed.status);
    }
    return result;
  }
  const auto finish_submitted_work = [&](bool record_stage_end) noexcept {
    if (record_stage_end) {
      const auto stage =
          impl_->completion.RecordStageEnd(completion_binding, 0);
      if (!stage.ok()) {
        result.device_lost = IsDeviceLost(stage.status);
        result.status = CompletionStatus(stage.status);
      }
    }
    const auto recorded =
        impl_->completion.RecordCompletion(completion_binding);
    if (!recorded.ok()) {
      result.device_lost = result.device_lost || IsDeviceLost(recorded.status);
      result.status = CompletionStatus(recorded.status);
      return false;
    }
    ++result.completion_waits;
    const auto completed =
        impl_->completion.WaitForCompletion(completion_binding, wait);
    result.pending_work = !completed.ok();
    if (!completed.ok()) {
      result.device_lost = result.device_lost || IsDeviceLost(completed.status);
      result.status = CompletionStatus(completed.status);
      return false;
    }
    result.completed_readbacks = result.enqueued_readbacks;
    return true;
  };
  cuda::kernels::PreparedPreprocessBinding preprocess_binding{
      binding.driver,
      binding.context,
      binding.stream,
      binding.stream_identity,
      binding.configuration_generation,
      impl_->artifact.registry_generation};
  cuda::kernels::ModelPreprocessSpec spec;
  spec.dst_w = impl_->input_w;
  spec.dst_h = impl_->input_h;
  spec.dst_order = cuda::kernels::ChannelOrder::bgr;
  std::fill(std::begin(spec.std), std::end(spec.std), 1.0F / 255.0F);
  cuda::kernels::PreparedPreprocessKernelStatus preprocess_status{};
  if (!impl_->preprocess.ExecuteLetterboxed(preprocess_binding, spec, geometry,
                                            source, impl_->input,
                                            &preprocess_status)) {
    result.status = PreparedCudaYunetStatus::preprocess_failed;
    result.device_lost =
        preprocess_status ==
        cuda::kernels::PreparedPreprocessKernelStatus::device_lost;
    if (result.device_lost)
      result.status = PreparedCudaYunetStatus::device_lost;
    (void)finish_submitted_work(true);
    return result;
  }
  const auto run = impl_->session->Execute(impl_->ort_input.data(), 1,
                                           impl_->ort_outputs.data(), kOutputs,
                                           impl_->run_evidence);
  if (!run.ok()) {
    result.status = PreparedCudaYunetStatus::inference_failed;
    if (run.quarantine == onnx::OrtPreparedRunQuarantine::provider) {
      result.device_lost = true;
      result.status = PreparedCudaYunetStatus::device_lost;
    }
    (void)finish_submitted_work(true);
    return result;
  }
  const auto stage_end =
      impl_->completion.RecordStageEnd(completion_binding, 0);
  if (!stage_end.ok()) {
    result.status = CompletionStatus(stage_end.status);
    result.device_lost = IsDeviceLost(stage_end.status);
    (void)finish_submitted_work(false);
    return result;
  }
  for (std::uint32_t i = 0; i < kOutputs; ++i) {
    const auto copy = impl_->completion.EnqueueReadbackAt(
        completion_binding, i, impl_->outputs[i].tensor.ptr,
        impl_->outputs[i].tensor.bytes);
    if (!copy.ok()) {
      result.device_lost = IsDeviceLost(copy.status);
      result.status = result.device_lost
                          ? PreparedCudaYunetStatus::device_lost
                          : PreparedCudaYunetStatus::readback_failed;
      break;
    }
    ++result.enqueued_readbacks;
  }
  if (!finish_submitted_work(false))
    return result;
  if (result.completed_readbacks != kOutputs)
    return result;
  impl_->scratch.BeginFrame();
  const auto *host =
      static_cast<const std::byte *>(impl_->completion.readback_data());
  for (int si = 0; si < 3; ++si) {
    const int stride = si == 0 ? 8 : (si == 1 ? 16 : 32),
              cols = impl_->input_w / stride, rows = impl_->input_h / stride;
    const auto *cls = reinterpret_cast<const float *>(
        host + impl_->outputs[static_cast<std::size_t>(si)].host_offset);
    const auto *obj = reinterpret_cast<const float *>(
        host + impl_->outputs[static_cast<std::size_t>(3 + si)].host_offset);
    const auto *box = reinterpret_cast<const float *>(
        host + impl_->outputs[static_cast<std::size_t>(6 + si)].host_offset);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) {
        const int idx = r * cols + c;
        const float score = std::sqrt(std::clamp(cls[idx], 0.f, 1.f) *
                                      std::clamp(obj[idx], 0.f, 1.f));
        if (score < 0.9f)
          continue;
        const auto box_value = [&](int channel) {
          return impl_->outputs[static_cast<std::size_t>(6 + si)].channels_first
                     ? box[channel * rows * cols + idx]
                     : box[idx * 4 + channel];
        };
        const float stride_f = static_cast<float>(stride);
        const float cx = (static_cast<float>(c) + box_value(0)) * stride_f,
                    cy = (static_cast<float>(r) + box_value(1)) * stride_f;
        const float w = std::exp(box_value(2)) * stride_f,
                    h = std::exp(box_value(3)) * stride_f;
        impl_->scratch.Consider({cx - w * .5f, cy - h * .5f, w, h, score});
      }
  }
  const auto &faces = impl_->scratch.Finalize(
      scale, geometry.pad_x, geometry.pad_y, source.w, source.h, .3f);
  auto &cached = cache->PrepareFaceDetections(kTopK);
  cached.assign(faces.begin(), faces.end());
  cache->face_detections_provenance = {this, impl_->cache_revision, true};
  result.cpu_tail_boundaries = 1;
  result.status = faces.empty() ? PreparedCudaYunetStatus::success_no_faces
                                : PreparedCudaYunetStatus::success_faces;
  return result;
}

bool PreparedCudaYunetDetector::Shutdown(std::string *error) noexcept {
  if (error)
    error->clear();
  if (!impl_ || impl_->stopped)
    return true;
  if (impl_->warmup_pending) {
    if (!impl_->setup.binding.driver->StreamSynchronize(
            impl_->setup.binding.stream, error)) {
      if (error && error->empty())
        *error = "YuNet warmup cleanup remains pending";
      return false;
    }
    impl_->warmup_pending = false;
  }
  const auto completion = impl_->completion.Shutdown();
  if (!completion.ok()) {
    if (error)
      *error = "YuNet completion cleanup remains pending";
    return false;
  }
  if (impl_->session)
    impl_->session->Invalidate();
  bool ok = true;
  for (auto &out : impl_->outputs)
    ok = out.tensor.Free(impl_->setup.binding.driver, error) && ok;
  ok = impl_->input.Free(impl_->setup.binding.driver, error) && ok;
  if (ok) {
    impl_->session.reset();
    impl_->ready = false;
    impl_->stopped = true;
  }
  return ok;
}
bool PreparedCudaYunetDetector::ready() const noexcept {
  return impl_ && impl_->ready && !impl_->stopped;
}
const ValidatedModelArtifactIdentity &
PreparedCudaYunetDetector::artifact() const noexcept {
  return impl_->artifact;
}

} // namespace studiocast::open_video
