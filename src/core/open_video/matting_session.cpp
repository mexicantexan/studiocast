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

  if (!EnsureInitialized(input_rgb_gpu.w, input_rgb_gpu.h, error_out)) {
    return false;
  }

  if (!impl_->buffers_ready) {
    if (error_out)
      *error_out =
          "OpenCudaMattingSession::Run: internal buffers not initialized.";
    return false;
  }

  if (!input_rgb_gpu.Valid()) {
    if (error_out)
      *error_out = "OpenCudaMattingSession::Run: invalid input image.";
    return false;
  }

  // Validate output tensor shape.
  if (output_alpha_gpu->ptr == 0 || !output_alpha_gpu->Valid()) {
    if (error_out)
      *error_out = "OpenCudaMattingSession::Run: output alpha tensor is not "
                   "allocated/valid.";
    return false;
  }
  if (output_alpha_gpu->n != 1 || output_alpha_gpu->c != 1 ||
      output_alpha_gpu->h != impl_->pack.matting->input.height ||
      output_alpha_gpu->w != impl_->pack.matting->input.width) {
    if (error_out) {
      *error_out = "OpenCudaMattingSession::Run: output alpha tensor shape "
                   "mismatch (expected 1x1x" +
                   std::to_string(impl_->pack.matting->input.height) + "x" +
                   std::to_string(impl_->pack.matting->input.width) + ")";
    }
    return false;
  }

  // Preprocess RGB/BGR u8 image to NCHW f32 model input.
  {
    std::string pp_err;
    if (!studiocast::cuda::kernels::PreprocessToTensor(
            input_rgb_gpu, impl_->input_tensor, impl_->preprocess, stream,
            &pp_err)) {
      if (error_out)
        *error_out =
            std::string("OpenCudaMattingSession::Run: preprocess failed: ") +
            pp_err;
      return false;
    }
  }

#if !STUDIOCAST_HAVE_ONNXRUNTIME
  (void)stream;
  if (error_out)
    *error_out = "OpenCudaMattingSession: built without ONNX Runtime support.";
  return false;
#else
  // Lazily create the ORT session so we can attempt stream interop.
  if (impl_->ort_latched_failure) {
    if (error_out)
      *error_out = impl_->ort_latched_error;
    return false;
  }

  if (!impl_->ort_session) {
    std::string ort_err;
    if (!impl_->CreateOrtSession(stream, &ort_err)) {
      if (error_out)
        *error_out = std::move(ort_err);
      return false;
    }
  }

  if (impl_->ort_stream != nullptr && impl_->ort_stream != stream) {
    if (error_out) {
      *error_out = "OpenCudaMattingSession::Run: ORT session was created for a "
                   "different CUDA stream. Create one session per stream.";
    }
    return false;
  }

  // If we could not configure ORT to use the caller stream, synchronize before
  // inference so ORT sees fully-written inputs.
  if (impl_->ort_needs_stream_sync) {
    std::string sync_err;
    if (!impl_->cuda->StreamSynchronize(stream, &sync_err)) {
      if (error_out)
        *error_out =
            std::string(
                "OpenCudaMattingSession::Run: StreamSynchronize failed: ") +
            sync_err;
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

  std::string ort_err;
  if (!impl_->ort_session->RunCudaIoBinding(&in, 1, &out, 1, &ort_err)) {
    if (error_out)
      *error_out = std::move(ort_err);
    return false;
  }

  return true;
#endif
}

} // namespace studiocast::open_cuda
