#include "core/open_video/matting_session.h"

#include <array>
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

} // namespace

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
  if (error_out)
    error_out->clear();
  if (!EnsureInitialized(frame_w, frame_h, error_out) ||
      !ValidateOutputTensor(impl_->pack, output_alpha_gpu, error_out))
    return false;

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
    return true;
  }

#if !STUDIOCAST_HAVE_ONNXRUNTIME
  (void)stream;
  if (error_out)
    *error_out = "OpenCudaMattingSession: built without ONNX Runtime support.";
  return false;
#else
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
  impl_->preparation_evidence.active_provider = impl_->ort_info.active_provider;
  impl_->preparation_evidence.model_id = impl_->pack.id;
  if (const auto *model = FindMainOnnxFile(impl_->pack))
    impl_->preparation_evidence.model_artifact_sha256 = model->sha256;
  return true;
#endif
}

bool OpenCudaMattingSession::Run(
    studiocast::cuda::CUstream stream,
    const studiocast::cuda::CudaImage &input_rgb_gpu,
    studiocast::cuda::CudaTensor *output_alpha_gpu, std::string *error_out) {
  if (error_out)
    error_out->clear();

  if (!output_alpha_gpu) {
    if (error_out)
      *error_out = "OpenCudaMattingSession::Run: output_alpha_gpu is null.";
    return false;
  }

  if (!impl_->buffers_ready || !impl_->execution_prepared) {
    if (error_out)
      *error_out =
          "OpenCudaMattingSession::Run: exact execution contract was not "
          "prepared during setup.";
    return false;
  }

  if (!input_rgb_gpu.Valid()) {
    if (error_out)
      *error_out = "OpenCudaMattingSession::Run: invalid input image.";
    return false;
  }

  if (!ValidateOutputTensor(impl_->pack, output_alpha_gpu, error_out))
    return false;
  const auto *context = impl_->cuda ? impl_->cuda->context_identity() : nullptr;
  if (impl_->prepared_frame_w != input_rgb_gpu.w ||
      impl_->prepared_frame_h != input_rgb_gpu.h ||
      impl_->prepared_stream != stream ||
      impl_->prepared_output_ptr != output_alpha_gpu->ptr || !context ||
      *context != impl_->preparation_evidence.context) {
    if (error_out)
      *error_out = "OpenCudaMattingSession::Run: geometry, stream, output, or "
                   "CUDA context generation differs from setup.";
    return false;
  }

  // Preprocess RGB/BGR u8 image to NCHW f32 model input.
  {
    if (!studiocast::cuda::kernels::PreprocessToTensorPrepared(
            impl_->cuda, impl_->preparation_evidence.context, input_rgb_gpu,
            impl_->input_tensor, impl_->preprocess, stream, error_out)) {
      if (error_out)
        error_out->insert(0,
                          "OpenCudaMattingSession::Run: preprocess failed: ");
      return false;
    }
  }

#if !STUDIOCAST_HAVE_ONNXRUNTIME
  (void)stream;
  if (error_out)
    *error_out = "OpenCudaMattingSession: built without ONNX Runtime support.";
  return false;
#else
  if (impl_->ort_latched_failure) {
    if (error_out)
      *error_out = impl_->ort_latched_error;
    return false;
  }

  if (!impl_->ort_session ||
      (impl_->ort_stream != nullptr && impl_->ort_stream != stream)) {
    if (error_out) {
      *error_out = "OpenCudaMattingSession::Run: ORT session was created for a "
                   "different CUDA stream. Create one session per stream.";
    }
    return false;
  }

  // If we could not configure ORT to use the caller stream, synchronize before
  // inference so ORT sees fully-written inputs.
  if (impl_->ort_needs_stream_sync) {
    if (!impl_->cuda->StreamSynchronize(stream, error_out)) {
      if (error_out)
        error_out->insert(
            0, "OpenCudaMattingSession::Run: StreamSynchronize failed: ");
      return false;
    }
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

  if (!impl_->ort_session->ExecutePreparedCudaIoBinding(0, &in, 1, &out, 1,
                                                        error_out)) {
    return false;
  }

  return true;
#endif
}

const OpenCudaMattingSession::PreparationEvidence *
OpenCudaMattingSession::preparation_evidence() const noexcept {
  return impl_ && impl_->execution_prepared ? &impl_->preparation_evidence
                                            : nullptr;
}

} // namespace studiocast::open_cuda
