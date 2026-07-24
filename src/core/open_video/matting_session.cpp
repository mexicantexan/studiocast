#include "core/open_video/matting_session.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "core/cuda/kernels/preprocess_to_nchw.h"
#include "core/onnx/ort_session.h"

namespace studiocast::open_cuda {

namespace {

std::atomic<std::uint64_t> g_matting_preparation_generation{1};

[[maybe_unused]] std::uint64_t NextMattingPreparationGeneration() noexcept {
  std::uint64_t value =
      g_matting_preparation_generation.fetch_add(1, std::memory_order_relaxed);
  if (value != 0)
    return value;
  value =
      g_matting_preparation_generation.fetch_add(1, std::memory_order_relaxed);
  return value == 0 ? 1 : value;
}

void SetCompatibilityError(std::string *error, const char *message) noexcept {
  if (!error)
    return;
  try {
    *error = message ? message : "";
  } catch (...) {
    error->clear();
  }
}

const char *
MattingPreparedRunStatusMessage(MattingPreparedRunStatus status) noexcept {
  switch (status) {
  case MattingPreparedRunStatus::success:
    return "";
  case MattingPreparedRunStatus::not_prepared:
    return "OpenCudaMattingSession: exact execution contract is not prepared.";
  case MattingPreparedRunStatus::invalid_input_contract:
    return "OpenCudaMattingSession: input contract is invalid.";
  case MattingPreparedRunStatus::invalid_output_contract:
    return "OpenCudaMattingSession: output contract is invalid.";
  case MattingPreparedRunStatus::foreign_evidence:
    return "OpenCudaMattingSession: prepared-run evidence is invalid.";
  case MattingPreparedRunStatus::stale_preparation_generation:
    return "OpenCudaMattingSession: preparation generation is stale.";
  case MattingPreparedRunStatus::wrong_device:
    return "OpenCudaMattingSession: CUDA device identity differs from setup.";
  case MattingPreparedRunStatus::stale_context_generation:
    return "OpenCudaMattingSession: CUDA context generation differs from setup.";
  case MattingPreparedRunStatus::stale_stream:
    return "OpenCudaMattingSession: CUDA stream differs from setup.";
  case MattingPreparedRunStatus::stale_artifact:
    return "OpenCudaMattingSession: model artifact evidence is stale.";
  case MattingPreparedRunStatus::stale_ort_session:
    return "OpenCudaMattingSession: ORT session generation is stale.";
  case MattingPreparedRunStatus::stale_binding:
    return "OpenCudaMattingSession: prepared binding contract is stale.";
  case MattingPreparedRunStatus::provider_unavailable:
    return "OpenCudaMattingSession: CUDA provider is unavailable.";
  case MattingPreparedRunStatus::provider_mismatch:
    return "OpenCudaMattingSession: provider evidence does not match.";
  case MattingPreparedRunStatus::cpu_fallback_forbidden:
    return "OpenCudaMattingSession: CPU provider fallback is forbidden.";
  case MattingPreparedRunStatus::tensorrt_forbidden:
    return "OpenCudaMattingSession: TensorRT provider is forbidden.";
  case MattingPreparedRunStatus::resource_exhausted:
    return "OpenCudaMattingSession: prepared execution exhausted resources.";
  case MattingPreparedRunStatus::preprocess_failure:
    return "OpenCudaMattingSession: prepared preprocessing failed.";
  case MattingPreparedRunStatus::inference_failure:
    return "OpenCudaMattingSession: prepared inference failed.";
  case MattingPreparedRunStatus::provider_failure_quarantine:
    return "OpenCudaMattingSession: provider failed and was quarantined.";
  case MattingPreparedRunStatus::cancelled_or_shutdown:
    return "OpenCudaMattingSession: execution was cancelled or is shutting down.";
  case MattingPreparedRunStatus::setup_resource_mutation:
    return "OpenCudaMattingSession: execution mutated setup-only resources.";
  }
  return "OpenCudaMattingSession: prepared execution failed.";
}

bool FileExists(const std::filesystem::path &p) {
  std::error_code ec;
  return std::filesystem::exists(p, ec) && !ec;
}

bool IsFinite3(const std::array<double, 3> &v) {
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

const studiocast::open_video::ModelFile *
FindMainOnnxFile(const studiocast::open_video::ModelPack &p) {
  // Prefer (kind=onnx, role=main|empty), then fall back to the first ONNX.
  for (const auto &f : p.files) {
    if (f.kind == "onnx" && (f.role.empty() || f.role == "main"))
      return &f;
  }
  for (const auto &f : p.files) {
    if (f.kind == "onnx")
      return &f;
  }
  return nullptr;
}

[[maybe_unused]] std::string
DigestToHex(const std::array<std::uint8_t, 32> &digest) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string value;
  value.reserve(digest.size() * 2);
  for (const std::uint8_t byte : digest) {
    value.push_back(kHex[(byte >> 4) & 0x0f]);
    value.push_back(kHex[byte & 0x0f]);
  }
  return value;
}

} // namespace

bool MattingPreparedRunEvidence::Valid() const noexcept {
  return schema_version == kSchemaVersion && preparation_generation != 0 &&
         cuda_device_id >= 0 && context.Valid() &&
         input_tensor_ptr != 0 && input_tensor_active_bytes != 0 &&
         input_tensor_allocation_pitch_bytes >= input_tensor_active_bytes &&
         input_tensor_allocation_capacity_bytes ==
             input_tensor_allocation_pitch_bytes &&
         output_tensor_ptr != 0 && frame_width > 0 && frame_height > 0 &&
         ort.Valid() &&
         (!strict_cuda_only ||
          (stream_identity != 0 && artifact.Valid()));
}

namespace internal {

MattingPreparedRunResult ClassifyMattingPreparedRunContract(
    const MattingPreparedRunContract &contract) noexcept {
  MattingPreparedRunResult result;
  result.preparation_generation = contract.active.preparation_generation;
  if (contract.shutdown_requested) {
    result.status = MattingPreparedRunStatus::cancelled_or_shutdown;
    return result;
  }
  if (!contract.buffers_ready || !contract.execution_prepared) {
    result.status = MattingPreparedRunStatus::not_prepared;
    return result;
  }
  if (!contract.input_valid) {
    result.status = MattingPreparedRunStatus::invalid_input_contract;
    return result;
  }
  if (!contract.output_valid) {
    result.status = MattingPreparedRunStatus::invalid_output_contract;
    return result;
  }
  if (!contract.requested.Valid()) {
    result.status = MattingPreparedRunStatus::foreign_evidence;
    return result;
  }
  if (!contract.active.Valid()) {
    result.status = MattingPreparedRunStatus::provider_unavailable;
    return result;
  }
  if (contract.requested.preparation_generation !=
      contract.active.preparation_generation) {
    result.status = MattingPreparedRunStatus::stale_preparation_generation;
    return result;
  }
  if (contract.requested.cuda_device_id != contract.active.cuda_device_id) {
    result.status = MattingPreparedRunStatus::wrong_device;
    return result;
  }
  if (!(contract.requested.context == contract.active.context)) {
    result.status = MattingPreparedRunStatus::stale_context_generation;
    return result;
  }
  if (contract.requested.stream_identity != contract.active.stream_identity) {
    result.status = MattingPreparedRunStatus::stale_stream;
    return result;
  }
  if (!(contract.requested.artifact == contract.active.artifact)) {
    result.status = MattingPreparedRunStatus::stale_artifact;
    return result;
  }
  if (contract.requested.ort.session_generation !=
      contract.active.ort.session_generation) {
    result.status = MattingPreparedRunStatus::stale_ort_session;
    return result;
  }
  if (!(contract.requested.ort.provider ==
        contract.active.ort.provider)) {
    result.status = MattingPreparedRunStatus::provider_mismatch;
    return result;
  }
  if (contract.strict_cuda_only) {
    if (!contract.requested.strict_cuda_only ||
        !contract.active.strict_cuda_only ||
        !contract.active.artifact.Valid()) {
      result.status = MattingPreparedRunStatus::provider_mismatch;
      return result;
    }
    if (contract.active.ort.provider.provider ==
        studiocast::onnx::OrtExecutionProviderId::cpu) {
      result.status = MattingPreparedRunStatus::cpu_fallback_forbidden;
      return result;
    }
    if (contract.active.ort.provider.provider ==
        studiocast::onnx::OrtExecutionProviderId::tensorrt) {
      result.status = MattingPreparedRunStatus::tensorrt_forbidden;
      return result;
    }
    if (!contract.active.ort.provider.ValidStrictCuda()) {
      result.status = MattingPreparedRunStatus::provider_mismatch;
      return result;
    }
  }
  if (!contract.binding_matches ||
      contract.requested.input_tensor_ptr !=
          contract.active.input_tensor_ptr ||
      contract.requested.input_tensor_active_bytes !=
          contract.active.input_tensor_active_bytes ||
      contract.requested.input_tensor_allocation_pitch_bytes !=
          contract.active.input_tensor_allocation_pitch_bytes ||
      contract.requested.input_tensor_allocation_capacity_bytes !=
          contract.active.input_tensor_allocation_capacity_bytes ||
      contract.requested.output_tensor_ptr !=
          contract.active.output_tensor_ptr ||
      contract.requested.frame_width != contract.active.frame_width ||
      contract.requested.frame_height != contract.active.frame_height) {
    result.status = MattingPreparedRunStatus::stale_binding;
    return result;
  }
  result.status = MattingPreparedRunStatus::success;
  return result;
}

MattingPreparedRunResult ProjectOrtPreparedRunResult(
    const studiocast::onnx::OrtPreparedRunResult &ort) noexcept {
  MattingPreparedRunResult result;
  result.inference = ort;
  if (ort.schema_version !=
      studiocast::onnx::OrtPreparedRunResult::kSchemaVersion) {
    result.status = MattingPreparedRunStatus::foreign_evidence;
    return result;
  }
  switch (ort.status) {
  case studiocast::onnx::OrtPreparedRunStatus::success:
    result.status = MattingPreparedRunStatus::success;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::not_prepared:
    result.status = MattingPreparedRunStatus::not_prepared;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::invalid_input_output_contract:
    result.status = MattingPreparedRunStatus::invalid_output_contract;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::foreign_provider_evidence:
    result.status = MattingPreparedRunStatus::foreign_evidence;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::stale_session_generation:
    result.status = MattingPreparedRunStatus::stale_ort_session;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::wrong_device:
    result.status = MattingPreparedRunStatus::wrong_device;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::stale_context_generation:
    result.status = MattingPreparedRunStatus::stale_context_generation;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::stale_stream:
    result.status = MattingPreparedRunStatus::stale_stream;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::stale_binding:
    result.status = MattingPreparedRunStatus::stale_binding;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::provider_unavailable:
    result.status = MattingPreparedRunStatus::provider_unavailable;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::provider_mismatch:
    result.status = MattingPreparedRunStatus::provider_mismatch;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::cpu_fallback_forbidden:
    result.status = MattingPreparedRunStatus::cpu_fallback_forbidden;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::tensorrt_forbidden:
    result.status = MattingPreparedRunStatus::tensorrt_forbidden;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::resource_exhausted:
    result.status = MattingPreparedRunStatus::resource_exhausted;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::inference_failure:
    result.status = MattingPreparedRunStatus::inference_failure;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::provider_failure_quarantine:
    result.status = MattingPreparedRunStatus::provider_failure_quarantine;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::cancelled_or_shutdown:
    result.status = MattingPreparedRunStatus::cancelled_or_shutdown;
    break;
  case studiocast::onnx::OrtPreparedRunStatus::setup_resource_mutation:
    result.status = MattingPreparedRunStatus::setup_resource_mutation;
    break;
  default:
    result.status = MattingPreparedRunStatus::foreign_evidence;
    break;
  }
  if (ort.quarantine == studiocast::onnx::OrtPreparedRunQuarantine::provider)
    result.quarantine = MattingPreparedRunQuarantine::provider;
  else if (ort.quarantine ==
           studiocast::onnx::OrtPreparedRunQuarantine::session)
    result.quarantine = MattingPreparedRunQuarantine::session;
  return result;
}

MattingPreparedRunEvidence ResolveMattingReplacementEvidence(
    const MattingPreparedRunEvidence &current,
    const MattingPreparedRunEvidence &candidate,
    bool replacement_complete) noexcept {
  return replacement_complete && candidate.Valid() ? candidate : current;
}

} // namespace internal

struct OpenCudaMattingSession::Impl {
  studiocast::cuda::CudaDriverApi *cuda = nullptr;
  studiocast::open_video::ModelPack pack;
  std::filesystem::path onnx_path;
  Options opts;

  bool validated_pack = false;
  bool buffers_ready = false;

  int last_frame_w = 0;
  int last_frame_h = 0;

  studiocast::cuda::kernels::ModelPreprocessSpec preprocess{};
  studiocast::cuda::CudaTensor input_tensor;

  // The CUDA stream the ORT session was created for (when using
  // user_compute_stream).
  studiocast::cuda::CUstream ort_stream = nullptr;

  bool ort_needs_stream_sync = false;

  std::unique_ptr<studiocast::onnx::OrtSession> ort_session;
  studiocast::onnx::OrtSessionInfo ort_info;
  std::string input_name;
  std::string output_name;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  bool execution_prepared = false;
  int prepared_frame_w = 0;
  int prepared_frame_h = 0;
  studiocast::cuda::CUstream prepared_stream = nullptr;
  studiocast::cuda::CUdeviceptr prepared_output_ptr = 0;
  OpenCudaMattingSession::PreparationEvidence preparation_evidence;
  studiocast::open_video::ValidatedModelArtifactIdentity artifact_identity;
  MattingPreparedRunEvidence run_evidence;
  MattingPreparedInputTensorView input_tensor_view;
  OpenCudaMattingSession::PreparedRunStats prepared_run_stats;
  std::atomic<bool> shutdown_requested{false};

  // When the GPU runs out of memory, ORT can spam errors every frame.
  // Latch the first failure so we can return a clear, human-friendly message
  // and avoid hammering the CUDA EP repeatedly.
  bool ort_latched_failure = false;
  std::string ort_latched_error;

  bool CreateOrtSession(studiocast::cuda::CUstream stream,
                        std::string *error_out) {
    if (error_out)
      error_out->clear();

#if !STUDIOCAST_HAVE_ONNXRUNTIME
    (void)stream;
    if (error_out)
      *error_out =
          "OpenCudaMattingSession: built without ONNX Runtime support.";
    return false;
#else
    studiocast::onnx::OrtSessionOptions ort_opts;
    ort_opts.prefer_cuda = true;
    ort_opts.cuda_device_id = opts.device_id;
    ort_opts.enable_tensorrt = opts.enable_tensorrt;
    if (opts.strict_cuda_provider) {
      if (opts.enable_tensorrt) {
        if (error_out)
          *error_out =
              "OpenCudaMattingSession: strict CUDA mode rejects TensorRT.";
        return false;
      }
      if (stream == nullptr) {
        if (error_out)
          *error_out =
              "OpenCudaMattingSession: strict CUDA mode requires a non-null "
              "resident stream.";
        return false;
      }
      const auto *context = cuda ? cuda->context_identity() : nullptr;
      if (!context || !context->Valid()) {
        if (error_out)
          *error_out =
              "OpenCudaMattingSession: strict CUDA context identity is "
              "unavailable.";
        return false;
      }
      ort_opts.provider_policy =
          studiocast::onnx::OrtProviderPolicy::strict_cuda_only;
      ort_opts.cuda_context_generation = context->logical_generation;
    }
    if (stream != nullptr) {
      // Treat CUstream as an opaque handle and pass it as void* to ORT.
      ort_opts.user_compute_stream = reinterpret_cast<void *>(stream);
    }

    std::string ort_err;
    if (onnx_path.empty()) {
      if (error_out)
        *error_out =
            "OpenCudaMattingSession: missing ONNX file path in model pack.";
      return false;
    }

    const auto *spec = pack.matting ? &*pack.matting : nullptr;
    if (!spec) {
      if (error_out)
        *error_out = "OpenCudaMattingSession: matting model pack is missing "
                     "required metadata.";
      return false;
    }

    ort_session = studiocast::onnx::OrtSession::Create(onnx_path, ort_opts,
                                                       &ort_info, &ort_err);
    if (!ort_session) {
      // If this looks like VRAM exhaustion, latch so we don't retry every frame
      // and spam logs.
      if (studiocast::onnx::OrtErrorLooksLikeVramOom(ort_err) ||
          ort_err.find("out of VRAM") != std::string::npos) {
        ort_latched_failure = true;
        ort_latched_error = ort_err;
      }

      if (error_out)
        *error_out = std::move(ort_err);
      return false;
    }

    if (!ort_info.using_cuda && !ort_info.using_tensorrt) {
      // Open CUDA backend requires a CUDA-capable EP.
      std::string msg =
          "OpenCudaMattingSession: ONNX Runtime CUDA-capable EP is "
          "unavailable (CPU-only build?).";
      if (!ort_info.warnings.empty()) {
        msg += " " + ort_info.warnings[0];
      }
      if (error_out)
        *error_out = msg;
      return false;
    }
    if (opts.strict_cuda_provider &&
        (!ort_info.strict_provider_validation_passed ||
         !ort_info.prepared_provider_evidence.ValidStrictCuda() ||
         ort_info.prepared_provider_evidence.provider !=
             studiocast::onnx::OrtExecutionProviderId::cuda ||
         !ort_info.cuda_user_compute_stream_bound ||
         ort_info.cuda_needs_stream_sync)) {
      if (error_out)
        *error_out =
            "OpenCudaMattingSession: exact CUDA-only provider evidence is "
            "unavailable.";
      return false;
    }

    if (ort_info.input_names.empty() || ort_info.output_names.empty()) {
      if (error_out)
        *error_out = "OpenCudaMattingSession: model must expose at least 1 "
                     "input and 1 output.";
      return false;
    }

    input_name = ort_info.input_names[0];
    output_name = ort_info.output_names[0];

    if (!spec->input.name.empty() && input_name != spec->input.name) {
      if (error_out) {
        *error_out = "Model input name mismatch: model='" + input_name +
                     "' manifest='" + spec->input.name + "'";
      }
      return false;
    }
    if (!spec->output.name.empty() && output_name != spec->output.name) {
      if (error_out) {
        *error_out = "Model output name mismatch: model='" + output_name +
                     "' manifest='" + spec->output.name + "'";
      }
      return false;
    }

    ort_needs_stream_sync = ort_info.cuda_needs_stream_sync;
    ort_stream =
        (!ort_needs_stream_sync && stream != nullptr) ? stream : nullptr;

    // Validate model input/output shapes (best-effort; allow dynamic dims).
    // NOTE: OrtSessionInfo stores element types using the ONNX enum numeric
    // values.
    //       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT == 1.
    if (ort_info.input_elem_types.empty() ||
        ort_info.output_elem_types.empty()) {
      if (error_out)
        *error_out =
            "OpenCudaMattingSession: model must expose tensor element types.";
      return false;
    }
    if (ort_info.input_elem_types[0] != 1) {
      if (error_out)
        *error_out = "Model input dtype must be float32.";
      return false;
    }
    if (ort_info.output_elem_types[0] != 1) {
      if (error_out)
        *error_out = "Model output dtype must be float32.";
      return false;
    }

    input_shape = ort_info.input_shapes.empty() ? std::vector<int64_t>{}
                                                : ort_info.input_shapes[0];
    output_shape = ort_info.output_shapes.empty() ? std::vector<int64_t>{}
                                                  : ort_info.output_shapes[0];

    if (input_shape.size() != 4) {
      if (error_out)
        *error_out = "Model input must be rank-4 (NCHW).";
      return false;
    }
    if (output_shape.size() != 4) {
      if (error_out)
        *error_out = "Model output must be rank-4.";
      return false;
    }

    const int64_t want_n = 1;
    const int64_t want_c = 3;
    const int64_t want_h = spec->input.height;
    const int64_t want_w = spec->input.width;

    auto matches_or_dynamic = [](int64_t got, int64_t want) {
      return got == want || got == -1;
    };
    if (!matches_or_dynamic(input_shape[0], want_n) ||
        !matches_or_dynamic(input_shape[1], want_c) ||
        !matches_or_dynamic(input_shape[2], want_h) ||
        !matches_or_dynamic(input_shape[3], want_w)) {
      if (error_out) {
        *error_out =
            "Model input shape does not match manifest (expected 1x3x" +
            std::to_string(want_h) + "x" + std::to_string(want_w) + ")";
      }
      return false;
    }

    // Normalize dynamic dims to concrete shapes for tensor creation.
    input_shape = {1, 3, want_h, want_w};

    // Expected alpha shape: 1x1xH xW.
    const int64_t want_out_n = 1;
    const int64_t want_out_c = 1;
    const int64_t want_out_h = spec->input.height;
    const int64_t want_out_w = spec->input.width;
    if (!matches_or_dynamic(output_shape[0], want_out_n) ||
        !matches_or_dynamic(output_shape[1], want_out_c) ||
        !matches_or_dynamic(output_shape[2], want_out_h) ||
        !matches_or_dynamic(output_shape[3], want_out_w)) {
      if (error_out) {
        *error_out = "Model output shape does not match expected alpha tensor "
                     "(expected 1x1x" +
                     std::to_string(want_out_h) + "x" +
                     std::to_string(want_out_w) + ")";
      }
      return false;
    }
    output_shape = {1, 1, want_out_h, want_out_w};

    return true;
#endif
  }
};

bool ValidateOutputTensor(
    const studiocast::open_video::ModelPack &pack,
    const studiocast::cuda::CudaTensor *output_alpha_gpu,
    std::string *error_out) {
  if (!output_alpha_gpu || !output_alpha_gpu->Valid()) {
    if (error_out)
      *error_out = "OpenCudaMattingSession: output alpha tensor is not "
                   "allocated/valid.";
    return false;
  }
  if (!pack.matting || output_alpha_gpu->n != 1 ||
      output_alpha_gpu->c != 1 ||
      output_alpha_gpu->h != pack.matting->input.height ||
      output_alpha_gpu->w != pack.matting->input.width) {
    if (error_out)
      *error_out =
          "OpenCudaMattingSession: output alpha tensor shape mismatch.";
    return false;
  }
  return true;
}

OpenCudaMattingSession::OpenCudaMattingSession(
    studiocast::cuda::CudaDriverApi *cuda,
    studiocast::open_video::ModelPack pack, Options opts)
    : impl_(std::make_unique<Impl>()) {
  impl_->cuda = cuda;
  impl_->pack = std::move(pack);
  impl_->opts = opts;
}

OpenCudaMattingSession::OpenCudaMattingSession(
    studiocast::cuda::CudaDriverApi *cuda,
    studiocast::open_video::ModelPack pack)
    : OpenCudaMattingSession(cuda, std::move(pack), Options{}) {}

OpenCudaMattingSession::~OpenCudaMattingSession() = default;

const studiocast::open_video::ModelPack &OpenCudaMattingSession::pack() const {
  return impl_->pack;
}

const OpenCudaMattingSession::Options &OpenCudaMattingSession::options() const {
  return impl_->opts;
}

bool OpenCudaMattingSession::EnsureInitialized(int frame_w, int frame_h,
                                               std::string *error_out) {
  if (error_out)
    error_out->clear();

  impl_->last_frame_w = frame_w;
  impl_->last_frame_h = frame_h;

  if (!impl_->cuda || !impl_->cuda->IsInitialized()) {
    if (error_out)
      *error_out = "OpenCudaMattingSession: CUDA driver API not initialized.";
    return false;
  }
  if (frame_w <= 0 || frame_h <= 0) {
    if (error_out)
      *error_out = "OpenCudaMattingSession: invalid frame size.";
    return false;
  }

  std::string err;
  // Setup-time device binding: keep Driver API kernels/allocations and the
  // ORT CUDA provider on the same CUDA device without per-frame probing.
  if (!impl_->cuda->EnsureContextForDevice(impl_->opts.device_id, &err)) {
    if (error_out)
      *error_out =
          std::string(
              "OpenCudaMattingSession: failed to ensure CUDA context: ") +
          err;
    return false;
  }

  if (!impl_->validated_pack) {
    if (impl_->pack.task != "matting") {
      if (error_out)
        *error_out =
            "OpenCudaMattingSession: model pack task must be 'matting'.";
      return false;
    }
    if (!impl_->pack.matting.has_value()) {
      if (error_out)
        *error_out = "OpenCudaMattingSession: matting model pack is missing "
                     "required metadata.";
      return false;
    }

    const auto &spec = *impl_->pack.matting;

    if (spec.input.layout != "nchw") {
      if (error_out)
        *error_out =
            "OpenCudaMattingSession: only NCHW models are supported in v1.";
      return false;
    }
    if (spec.input.dtype != "float32") {
      if (error_out)
        *error_out = "OpenCudaMattingSession: only float32 input models are "
                     "supported in v1.";
      return false;
    }
    if (spec.output.dtype != "float32") {
      if (error_out)
        *error_out = "OpenCudaMattingSession: only float32 output models are "
                     "supported in v1.";
      return false;
    }
    if (spec.input.channels != 3) {
      if (error_out)
        *error_out =
            "OpenCudaMattingSession: model input must have 3 channels.";
      return false;
    }
    if (spec.input.width <= 0 || spec.input.height <= 0) {
      if (error_out)
        *error_out = "OpenCudaMattingSession: invalid model input size.";
      return false;
    }
    if (!IsFinite3(spec.preprocess.mean) || !IsFinite3(spec.preprocess.std)) {
      if (error_out)
        *error_out =
            "OpenCudaMattingSession: preprocess mean/std must be finite.";
      return false;
    }
    if (spec.preprocess.std[0] == 0.0 || spec.preprocess.std[1] == 0.0 ||
        spec.preprocess.std[2] == 0.0) {
      if (error_out)
        *error_out = "OpenCudaMattingSession: preprocess std must be non-zero.";
      return false;
    }
    if (spec.preprocess.color != "rgb" || spec.preprocess.range != "0..1") {
      if (error_out)
        *error_out = "OpenCudaMattingSession: unsupported preprocess spec "
                     "(expected rgb + 0..1).";
      return false;
    }

    const auto *onnx = FindMainOnnxFile(impl_->pack);
    if (!onnx) {
      if (error_out)
        *error_out = "OpenCudaMattingSession: missing ONNX file (kind=onnx).";
      return false;
    }
    impl_->onnx_path = onnx->path;
    if (!FileExists(impl_->onnx_path)) {
      if (error_out)
        *error_out = "OpenCudaMattingSession: missing ONNX file at " +
                     impl_->onnx_path.string();
      return false;
    }

    impl_->preprocess.dst_w = spec.input.width;
    impl_->preprocess.dst_h = spec.input.height;
    for (std::size_t i = 0; i < 3; ++i) {
      impl_->preprocess.mean[i] = static_cast<float>(spec.preprocess.mean[i]);
      impl_->preprocess.std[i] = static_cast<float>(spec.preprocess.std[i]);
    }
    impl_->preprocess.dst_order = studiocast::cuda::kernels::ChannelOrder::rgb;

    impl_->validated_pack = true;
  }

  // Allocate persistent input tensor.
  {
    std::string alloc_err;
    if (!impl_->input_tensor.ReallocIfNeededNchwF32(
            impl_->cuda,
            /*n_in=*/1,
            /*c_in=*/3,
            /*h_in=*/impl_->pack.matting->input.height,
            /*w_in=*/impl_->pack.matting->input.width, &alloc_err)) {
      if (error_out)
        *error_out =
            std::string(
                "OpenCudaMattingSession: failed to allocate input tensor: ") +
            alloc_err;
      return false;
    }
  }

  impl_->buffers_ready = true;
  return true;
}

bool OpenCudaMattingSession::Prepare(
    int frame_w, int frame_h, studiocast::cuda::CUstream stream,
    studiocast::cuda::CudaTensor *output_alpha_gpu, std::string *error_out) {
  struct SetupAttempt {
    PreparedRunStats &stats;
    bool succeeded = false;
    explicit SetupAttempt(PreparedRunStats &value) : stats(value) {
      ++stats.setup_attempts;
    }
    ~SetupAttempt() {
      if (!succeeded)
        ++stats.setup_failures;
    }
  } setup_attempt(impl_->prepared_run_stats);
  if (error_out)
    error_out->clear();
  if (!EnsureInitialized(frame_w, frame_h, error_out) ||
      !ValidateOutputTensor(impl_->pack, output_alpha_gpu, error_out))
    return false;
  if (impl_->opts.strict_cuda_provider && stream == nullptr) {
    if (error_out)
      *error_out =
          "OpenCudaMattingSession: strict CUDA mode requires a non-null "
          "resident stream.";
    return false;
  }

  const auto *context = impl_->cuda->context_identity();
  if (!context || !context->Valid()) {
    if (error_out)
      *error_out =
          "OpenCudaMattingSession: exact CUDA context identity unavailable.";
    return false;
  }
  if (!studiocast::cuda::kernels::PreparePreprocessKernel(impl_->cuda,
                                                           error_out))
    return false;

  if (impl_->execution_prepared && impl_->prepared_frame_w == frame_w &&
      impl_->prepared_frame_h == frame_h &&
      impl_->prepared_stream == stream &&
      impl_->prepared_output_ptr == output_alpha_gpu->ptr &&
      impl_->preparation_evidence.context == *context) {
    setup_attempt.succeeded = true;
    return true;
  }

#if !STUDIOCAST_HAVE_ONNXRUNTIME
  (void)stream;
  if (error_out)
    *error_out = "OpenCudaMattingSession: built without ONNX Runtime support.";
  return false;
#else
  studiocast::open_video::ModelArtifactValidationResult artifact_validation;
  if (impl_->opts.strict_cuda_provider) {
    const auto *model = FindMainOnnxFile(impl_->pack);
    if (!model) {
      if (error_out)
        *error_out =
            "OpenCudaMattingSession: strict model artifact is missing.";
      return false;
    }
    artifact_validation =
        studiocast::open_video::ValidateModelArtifactIdentity(impl_->pack,
                                                               *model);
    if (!artifact_validation.ok()) {
      if (error_out) {
        *error_out =
            "OpenCudaMattingSession: strict model artifact validation failed";
        if (!artifact_validation.detail.empty())
          *error_out += ": " + artifact_validation.detail;
      }
      return false;
    }
  }

  // Build a replacement session and binding in isolation. Nothing below
  // mutates the live preparation until every evidence check and binding step
  // has succeeded.
  Impl candidate;
  candidate.cuda = impl_->cuda;
  candidate.pack = impl_->pack;
  candidate.onnx_path = impl_->onnx_path;
  candidate.opts = impl_->opts;
  candidate.validated_pack = true;
  candidate.preprocess = impl_->preprocess;
  candidate.artifact_identity = artifact_validation.identity;

  std::string setup_error;
  if (!candidate.CreateOrtSession(stream, &setup_error)) {
    if (!impl_->execution_prepared && candidate.ort_latched_failure) {
      impl_->ort_latched_failure = true;
      impl_->ort_latched_error = candidate.ort_latched_error;
    }
    if (error_out)
      *error_out = std::move(setup_error);
    return false;
  }

  const auto &info = candidate.ort_info;
  const std::uintptr_t requested_stream =
      reinterpret_cast<std::uintptr_t>(stream);
  if (info.cuda_device_id != impl_->opts.device_id ||
      info.cuda_user_compute_stream_requested != (stream != nullptr) ||
      info.cuda_user_compute_stream_identity != requested_stream ||
      (stream != nullptr && !info.cuda_needs_stream_sync &&
       !info.cuda_user_compute_stream_bound) ||
      (info.cuda_user_compute_stream_bound &&
       info.cuda_user_compute_stream_identity != requested_stream)) {
    if (error_out)
      *error_out = "OpenCudaMattingSession: ORT provider/device/user-stream "
                   "evidence does not match the requested setup.";
    return false;
  }
  if (impl_->opts.strict_cuda_provider &&
      (!studiocast::onnx::OrtPreparedProviderEvidenceMatches(
           info.prepared_provider_evidence, impl_->opts.device_id,
           context->logical_generation, requested_stream) ||
       !info.prepared_provider_evidence.user_stream_bound ||
       info.prepared_provider_evidence.explicit_stream_synchronization ||
       !candidate.artifact_identity.Valid())) {
    if (error_out)
      *error_out =
          "OpenCudaMattingSession: strict provider/artifact evidence does not "
          "match the requested setup.";
    return false;
  }

  studiocast::onnx::OrtSession::CudaBindingInput in;
  in.name = candidate.input_name.c_str();
  in.device_ptr = reinterpret_cast<float *>(
      static_cast<std::uintptr_t>(impl_->input_tensor.ptr));
  in.num_floats = impl_->input_tensor.ElementCount();
  in.shape = candidate.input_shape.data();
  in.shape_rank = candidate.input_shape.size();

  studiocast::onnx::OrtSession::CudaBindingOutput out;
  out.name = candidate.output_name.c_str();
  out.device_ptr = reinterpret_cast<float *>(
      static_cast<std::uintptr_t>(output_alpha_gpu->ptr));
  out.num_floats = output_alpha_gpu->ElementCount();
  out.shape = candidate.output_shape.data();
  out.shape_rank = candidate.output_shape.size();

  candidate.ort_session->ReserveRunScratch(1, 1);
  if (!candidate.ort_session->PrepareCudaIoBinding(0, &in, 1, &out, 1,
                                                   &setup_error)) {
    if (error_out)
      *error_out = std::move(setup_error);
    return false;
  }

  // Execute one bounded, real inference during setup so provider workspaces,
  // kernels, and graph initialization cannot be deferred to frame 1. Both
  // external buffers are already allocated and bound; zero is a valid finite
  // model input and the caller explicitly lends the output tensor to Prepare.
  if (!impl_->cuda->MemsetD8Async(impl_->input_tensor.ptr, 0,
                                  impl_->input_tensor.bytes, stream,
                                  &setup_error) ||
      !impl_->cuda->MemsetD8Async(output_alpha_gpu->ptr, 0,
                                  output_alpha_gpu->bytes, stream,
                                  &setup_error)) {
    if (error_out)
      *error_out = std::move(setup_error);
    return false;
  }
  if (candidate.ort_needs_stream_sync &&
      !impl_->cuda->StreamSynchronize(stream, &setup_error)) {
    if (error_out)
      *error_out = std::move(setup_error);
    return false;
  }
  const auto stats_before_warmup =
      candidate.ort_session->prepared_run_stats();
  if (!candidate.ort_session->ExecutePreparedCudaIoBinding(
          0, &in, 1, &out, 1, &setup_error) ||
      !impl_->cuda->StreamSynchronize(stream, &setup_error)) {
    if (error_out)
      *error_out = std::move(setup_error);
    return false;
  }
  const auto stats_after_warmup =
      candidate.ort_session->prepared_run_stats();
  if (stats_after_warmup.runs != stats_before_warmup.runs + 1 ||
      stats_after_warmup.binding_rebuilds !=
          stats_before_warmup.binding_rebuilds ||
      stats_after_warmup.tensor_wrapper_constructions !=
          stats_before_warmup.tensor_wrapper_constructions ||
      stats_after_warmup.io_binding_constructions !=
          stats_before_warmup.io_binding_constructions ||
      stats_after_warmup.application_binding_allocation_requests !=
          stats_before_warmup.application_binding_allocation_requests) {
    if (error_out)
      *error_out = "OpenCudaMattingSession: setup warmup mutated prepared "
                   "binding resources.";
    return false;
  }

  MattingPreparedRunEvidence candidate_run_evidence;
  candidate_run_evidence.preparation_generation =
      NextMattingPreparationGeneration();
  candidate_run_evidence.cuda_device_id = impl_->opts.device_id;
  candidate_run_evidence.strict_cuda_only =
      impl_->opts.strict_cuda_provider;
  candidate_run_evidence.context = *context;
  candidate_run_evidence.stream_identity = requested_stream;
  candidate_run_evidence.input_tensor_ptr = impl_->input_tensor.ptr;
  candidate_run_evidence.input_tensor_active_bytes = impl_->input_tensor.bytes;
  candidate_run_evidence.input_tensor_allocation_pitch_bytes =
      impl_->input_tensor.pitch;
  candidate_run_evidence.input_tensor_allocation_capacity_bytes =
      impl_->input_tensor.pitch;
  candidate_run_evidence.output_tensor_ptr = output_alpha_gpu->ptr;
  candidate_run_evidence.frame_width = frame_w;
  candidate_run_evidence.frame_height = frame_h;
  candidate_run_evidence.artifact = candidate.artifact_identity;
  candidate_run_evidence.ort =
      candidate.ort_session->prepared_cuda_run_evidence();
  if (!candidate_run_evidence.Valid()) {
    if (error_out)
      *error_out = "OpenCudaMattingSession: prepared execution evidence is "
                   "incomplete.";
    return false;
  }

  impl_->ort_session = std::move(candidate.ort_session);
  impl_->ort_info = std::move(candidate.ort_info);
  impl_->input_name = std::move(candidate.input_name);
  impl_->output_name = std::move(candidate.output_name);
  impl_->input_shape = std::move(candidate.input_shape);
  impl_->output_shape = std::move(candidate.output_shape);
  impl_->ort_stream = candidate.ort_stream;
  impl_->ort_needs_stream_sync = candidate.ort_needs_stream_sync;
  impl_->ort_latched_failure = false;
  impl_->ort_latched_error.clear();
  impl_->artifact_identity = candidate.artifact_identity;
  impl_->execution_prepared = true;
  impl_->prepared_frame_w = frame_w;
  impl_->prepared_frame_h = frame_h;
  impl_->prepared_stream = stream;
  impl_->prepared_output_ptr = output_alpha_gpu->ptr;
  impl_->preparation_evidence.cuda_device_id = impl_->opts.device_id;
  impl_->preparation_evidence.context = *context;
  impl_->preparation_evidence.requested_stream_identity = requested_stream;
  impl_->preparation_evidence.ort_user_stream_bound =
      impl_->ort_info.cuda_user_compute_stream_bound;
  impl_->preparation_evidence.explicit_stream_synchronization =
      impl_->ort_info.cuda_needs_stream_sync;
  impl_->preparation_evidence.inference_warmup_complete = true;
  impl_->preparation_evidence.setup_warmup_inferences = 1;
  impl_->preparation_evidence.input_tensor_active_bytes =
      impl_->input_tensor.bytes;
  impl_->preparation_evidence.input_tensor_allocation_pitch_bytes =
      impl_->input_tensor.pitch;
  impl_->preparation_evidence.input_tensor_allocation_capacity_bytes =
      impl_->input_tensor.pitch;
  impl_->preparation_evidence.active_provider = impl_->ort_info.active_provider;
  impl_->preparation_evidence.model_id = impl_->pack.id;
  if (const auto *model = FindMainOnnxFile(impl_->pack))
    impl_->preparation_evidence.model_artifact_sha256 = model->sha256;
  impl_->preparation_evidence.provider =
      impl_->ort_info.prepared_provider_evidence;
  impl_->preparation_evidence.model_artifact = impl_->artifact_identity;
  if (impl_->artifact_identity.Valid()) {
    impl_->preparation_evidence.model_artifact_sha256 =
        DigestToHex(impl_->artifact_identity.artifact_sha256);
  }
  impl_->run_evidence = internal::ResolveMattingReplacementEvidence(
      impl_->run_evidence, candidate_run_evidence, true);
  impl_->input_tensor_view = {};
  impl_->input_tensor_view.ptr = impl_->input_tensor.ptr;
  impl_->input_tensor_view.bytes = impl_->input_tensor.bytes;
  impl_->input_tensor_view.allocation_pitch_bytes = impl_->input_tensor.pitch;
  impl_->input_tensor_view.allocation_capacity_bytes =
      impl_->input_tensor.pitch;
  impl_->input_tensor_view.n = impl_->input_tensor.n;
  impl_->input_tensor_view.c = impl_->input_tensor.c;
  impl_->input_tensor_view.h = impl_->input_tensor.h;
  impl_->input_tensor_view.w = impl_->input_tensor.w;
  impl_->input_tensor_view.evidence = impl_->run_evidence;
  impl_->prepared_run_stats.preparation_generation =
      impl_->run_evidence.preparation_generation;
  ++impl_->prepared_run_stats.setup_commits;
  setup_attempt.succeeded = true;
  return true;
#endif
}

bool OpenCudaMattingSession::Run(
    studiocast::cuda::CUstream stream,
    const studiocast::cuda::CudaImage &input_rgb_gpu,
    studiocast::cuda::CudaTensor *output_alpha_gpu, std::string *error_out) {
  if (!Prepare(input_rgb_gpu.w, input_rgb_gpu.h, stream, output_alpha_gpu,
               error_out))
    return false;
  return RunPrepared(stream, input_rgb_gpu, output_alpha_gpu, error_out);
}

bool OpenCudaMattingSession::RunPrepared(
    studiocast::cuda::CUstream stream,
    const studiocast::cuda::CudaImage &input_rgb_gpu,
    studiocast::cuda::CudaTensor *output_alpha_gpu, std::string *error_out) {
  const auto expected = prepared_run_evidence();
  return RunPreparedTypedImpl(stream, &input_rgb_gpu, output_alpha_gpu,
                              expected, true, false, error_out)
      .ok();
}

MattingPreparedRunResult OpenCudaMattingSession::RunPreparedTyped(
    studiocast::cuda::CUstream stream,
    const studiocast::cuda::CudaImage &input_rgb_gpu,
    studiocast::cuda::CudaTensor *output_alpha_gpu,
    const MattingPreparedRunEvidence &expected,
    bool cancellation_requested) noexcept {
  return RunPreparedTypedImpl(stream, &input_rgb_gpu, output_alpha_gpu,
                              expected, true, cancellation_requested, nullptr);
}

MattingPreparedRunResult
OpenCudaMattingSession::RunPreparedInferenceTyped(
    studiocast::cuda::CUstream stream,
    studiocast::cuda::CudaTensor *output_alpha_gpu,
    const MattingPreparedRunEvidence &expected,
    bool cancellation_requested) noexcept {
  return RunPreparedTypedImpl(stream, nullptr, output_alpha_gpu, expected,
                              false, cancellation_requested, nullptr);
}

MattingPreparedRunResult OpenCudaMattingSession::RunPreparedTypedImpl(
    studiocast::cuda::CUstream stream,
    const studiocast::cuda::CudaImage *input_rgb_gpu,
    studiocast::cuda::CudaTensor *output_alpha_gpu,
    const MattingPreparedRunEvidence &expected, bool run_preprocess,
    bool cancellation_requested, std::string *compatibility_error) noexcept {
  if (compatibility_error)
    compatibility_error->clear();
  MattingPreparedRunResult result;
  if (!impl_) {
    SetCompatibilityError(
        compatibility_error,
        MattingPreparedRunStatusMessage(MattingPreparedRunStatus::not_prepared));
    return result;
  }

  auto &stats = impl_->prepared_run_stats;
  result.preparation_generation = impl_->run_evidence.preparation_generation;
  result.attempt_sequence = ++stats.run_attempts;
  result.completed_runs = stats.run_successes;
  const auto fail = [&](MattingPreparedRunResult failed) noexcept {
    failed.preparation_generation =
        impl_->run_evidence.preparation_generation;
    failed.attempt_sequence = result.attempt_sequence;
    failed.completed_runs = stats.run_successes;
    ++stats.run_failures;
    if (failed.quarantine != MattingPreparedRunQuarantine::none)
      ++stats.quarantine_events;
    if (compatibility_error && compatibility_error->empty())
      SetCompatibilityError(
          compatibility_error,
          MattingPreparedRunStatusMessage(failed.status));
    return failed;
  };

  const auto *context =
      impl_->cuda ? impl_->cuda->context_identity() : nullptr;
  const bool output_valid =
      ValidateOutputTensor(impl_->pack, output_alpha_gpu, nullptr);
  const bool input_valid =
      !run_preprocess || (input_rgb_gpu && input_rgb_gpu->Valid());
  const bool binding_matches =
      context && impl_->prepared_frame_w == impl_->run_evidence.frame_width &&
      impl_->prepared_frame_h == impl_->run_evidence.frame_height &&
      impl_->input_tensor.Valid() &&
      impl_->input_tensor.ptr == impl_->run_evidence.input_tensor_ptr &&
      impl_->input_tensor.bytes ==
          impl_->run_evidence.input_tensor_active_bytes &&
      impl_->input_tensor.pitch ==
          impl_->run_evidence.input_tensor_allocation_pitch_bytes &&
      impl_->input_tensor.pitch ==
          impl_->run_evidence.input_tensor_allocation_capacity_bytes &&
      (!run_preprocess ||
       (input_rgb_gpu->w == impl_->prepared_frame_w &&
        input_rgb_gpu->h == impl_->prepared_frame_h)) &&
      impl_->prepared_stream == stream && output_alpha_gpu &&
      impl_->prepared_output_ptr == output_alpha_gpu->ptr &&
      *context == impl_->preparation_evidence.context &&
      impl_->ort_session &&
      (impl_->ort_stream == nullptr || impl_->ort_stream == stream);

  internal::MattingPreparedRunContract contract;
  contract.buffers_ready = impl_->buffers_ready;
  contract.execution_prepared = impl_->execution_prepared;
  contract.input_valid = input_valid;
  contract.output_valid = output_valid;
  contract.binding_matches = binding_matches;
  contract.strict_cuda_only = impl_->opts.strict_cuda_provider;
  contract.shutdown_requested =
      cancellation_requested ||
      impl_->shutdown_requested.load(std::memory_order_acquire);
  contract.active = impl_->run_evidence;
  contract.requested = expected;
  auto classified =
      internal::ClassifyMattingPreparedRunContract(contract);
  if (!classified.ok())
    return fail(classified);

  if (run_preprocess) {
    ++stats.preprocess_attempts;
    if (!studiocast::cuda::kernels::PreprocessToTensorPrepared(
            impl_->cuda, impl_->preparation_evidence.context, *input_rgb_gpu,
            impl_->input_tensor, impl_->preprocess, stream,
            compatibility_error)) {
      if (compatibility_error) {
        try {
          compatibility_error->insert(
              0, "OpenCudaMattingSession::Run: preprocess failed: ");
        } catch (...) {
          SetCompatibilityError(
              compatibility_error,
              "OpenCudaMattingSession::Run: preprocess failed.");
        }
      }
      MattingPreparedRunResult failed;
      failed.status = MattingPreparedRunStatus::preprocess_failure;
      return fail(failed);
    }
  }

  if (impl_->ort_needs_stream_sync &&
      !impl_->cuda->StreamSynchronize(stream, compatibility_error)) {
    if (compatibility_error) {
      try {
        compatibility_error->insert(
            0, "OpenCudaMattingSession::Run: StreamSynchronize failed: ");
      } catch (...) {
        SetCompatibilityError(
            compatibility_error,
            "OpenCudaMattingSession::Run: StreamSynchronize failed.");
      }
    }
    MattingPreparedRunResult failed;
    failed.status = MattingPreparedRunStatus::inference_failure;
    failed.quarantine = MattingPreparedRunQuarantine::session;
    return fail(failed);
  }

  studiocast::onnx::OrtSession::CudaBindingInput in;
  in.name = impl_->input_name.c_str();
  in.device_ptr = reinterpret_cast<float *>(
      static_cast<std::uintptr_t>(impl_->input_tensor.ptr));
  in.num_floats = impl_->input_tensor.ElementCount();
  in.shape = impl_->input_shape.data();
  in.shape_rank = impl_->input_shape.size();

  studiocast::onnx::OrtSession::CudaBindingOutput out;
  out.name = impl_->output_name.c_str();
  out.device_ptr = reinterpret_cast<float *>(
      static_cast<std::uintptr_t>(output_alpha_gpu->ptr));
  out.num_floats = output_alpha_gpu->ElementCount();
  out.shape = impl_->output_shape.data();
  out.shape_rank = impl_->output_shape.size();

  ++stats.inference_attempts;
  const auto ort_result = impl_->ort_session->ExecutePreparedCudaIoBindingTyped(
      0, &in, 1, &out, 1, expected.ort, cancellation_requested,
      compatibility_error);
  auto projected = internal::ProjectOrtPreparedRunResult(ort_result);
  if (!projected.ok())
    return fail(projected);

  ++stats.run_successes;
  projected.preparation_generation =
      impl_->run_evidence.preparation_generation;
  projected.attempt_sequence = result.attempt_sequence;
  projected.completed_runs = stats.run_successes;
  return projected;
}

const OpenCudaMattingSession::PreparationEvidence *
OpenCudaMattingSession::preparation_evidence() const noexcept {
  return impl_ && impl_->execution_prepared ? &impl_->preparation_evidence
                                            : nullptr;
}

MattingPreparedRunEvidence
OpenCudaMattingSession::prepared_run_evidence() const noexcept {
  return impl_ && impl_->execution_prepared ? impl_->run_evidence
                                            : MattingPreparedRunEvidence{};
}

MattingPreparedInputTensorView
OpenCudaMattingSession::prepared_input_tensor_view() const noexcept {
  return impl_ && impl_->execution_prepared &&
                 impl_->input_tensor_view.Valid()
             ? impl_->input_tensor_view
             : MattingPreparedInputTensorView{};
}

OpenCudaMattingSession::PreparedRunStats
OpenCudaMattingSession::prepared_run_stats() const noexcept {
  return impl_ ? impl_->prepared_run_stats : PreparedRunStats{};
}

void OpenCudaMattingSession::RequestShutdown() noexcept {
  if (!impl_)
    return;
  impl_->shutdown_requested.store(true, std::memory_order_release);
  if (impl_->ort_session)
    impl_->ort_session->RequestShutdown();
}

} // namespace studiocast::open_cuda
