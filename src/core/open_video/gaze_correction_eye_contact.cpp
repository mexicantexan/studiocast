#include "core/open_video/gaze_correction_eye_contact.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <sstream>
#include <utility>

#include "core/video/image_ppm.h" // ResizeRgb24Bilinear

namespace studiocast::open_video {

namespace {

static int ClampInt(int v, int lo, int hi) {
  if (v < lo)
    return lo;
  if (v > hi)
    return hi;
  return v;
}

static std::size_t ByteOffset(std::size_t stride, int x, int y) {
  return static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x) * 3;
}

static void BilinearSampleRgbNchw(const float *rgb_nchw, int w, int h, float x,
                                  float y, float *out_r, float *out_g,
                                  float *out_b) {
  // Clamp to valid range.
  if (x < 0.f)
    x = 0.f;
  if (y < 0.f)
    y = 0.f;
  if (x > static_cast<float>(w - 1))
    x = static_cast<float>(w - 1);
  if (y > static_cast<float>(h - 1))
    y = static_cast<float>(h - 1);

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(w - 1, x0 + 1);
  const int y1 = std::min(h - 1, y0 + 1);
  const float tx = x - static_cast<float>(x0);
  const float ty = y - static_cast<float>(y0);

  const int idx00 = y0 * w + x0;
  const int idx10 = y0 * w + x1;
  const int idx01 = y1 * w + x0;
  const int idx11 = y1 * w + x1;

  const int plane = w * h;
  const float r00 = rgb_nchw[idx00];
  const float r10 = rgb_nchw[idx10];
  const float r01 = rgb_nchw[idx01];
  const float r11 = rgb_nchw[idx11];

  const float g00 = rgb_nchw[plane + idx00];
  const float g10 = rgb_nchw[plane + idx10];
  const float g01 = rgb_nchw[plane + idx01];
  const float g11 = rgb_nchw[plane + idx11];

  const float b00 = rgb_nchw[2 * plane + idx00];
  const float b10 = rgb_nchw[2 * plane + idx10];
  const float b01 = rgb_nchw[2 * plane + idx01];
  const float b11 = rgb_nchw[2 * plane + idx11];

  const float r0 = r00 + (r10 - r00) * tx;
  const float r1 = r01 + (r11 - r01) * tx;
  const float g0 = g00 + (g10 - g00) * tx;
  const float g1 = g01 + (g11 - g01) * tx;
  const float b0 = b00 + (b10 - b00) * tx;
  const float b1 = b01 + (b11 - b01) * tx;

  *out_r = r0 + (r1 - r0) * ty;
  *out_g = g0 + (g1 - g0) * ty;
  *out_b = b0 + (b1 - b0) * ty;
}

} // namespace

GazeCorrectionEyeContact::GazeCorrectionEyeContact() = default;
GazeCorrectionEyeContact::~GazeCorrectionEyeContact() = default;

void GazeCorrectionEyeContact::Reset() {
  initialized_ = false;
  disabled_ = false;
  using_cpu_fallback_ = false;
  registry_ = ModelPackRegistry();
  active_model_id_.clear();
  active_requested_model_id_.clear();
  required_landmarks_id_.clear();
  left_model_path_.clear();
  right_model_path_.clear();
  dlib_landmarks_.Reset();
  left_ = EyeRuntime();
  right_ = EyeRuntime();
  left_eye_scratch_.valid = false;
  right_eye_scratch_.valid = false;
  left_corrected_scratch_.clear();
  right_corrected_scratch_.clear();
  upscale_scratch_.clear();
  left_extract_plan_.Clear();
  right_extract_plan_.Clear();
  left_upscale_plan_.Clear();
  right_upscale_plan_.Clear();
  left_extract_geometry_ = {};
  right_extract_geometry_ = {};
  left_upscale_geometry_ = {};
  right_upscale_geometry_ = {};
  scratch_frame_width_ = 0;
  scratch_frame_height_ = 0;
  scratch_left_input_width_ = 0;
  scratch_left_input_height_ = 0;
  scratch_right_input_width_ = 0;
  scratch_right_input_height_ = 0;
  sticky_warning_.clear();
  runtime_failures_ = 0;
}

void GazeCorrectionEyeContact::DisableAfterFailure(const std::string &why) {
  disabled_ = true;
  sticky_warning_ = why;
}

EyeContactRuntimeStatus GazeCorrectionEyeContact::runtime_status() const {
  EyeContactRuntimeStatus s;
  const bool left_active = left_.session_active != nullptr;
  const bool right_active = right_.session_active != nullptr;

  s.left_cuda_ep_active = left_active &&
                          left_.session_active == left_.session_cuda.get() &&
                          left_.session_info.using_cuda;
  s.right_cuda_ep_active = right_active &&
                           right_.session_active == right_.session_cuda.get() &&
                           right_.session_info.using_cuda;
  s.cuda_ep_cpu_tensor_io_active =
      s.left_cuda_ep_active || s.right_cuda_ep_active;
  s.cpu_only_session_active =
      (left_active || right_active) && !s.cuda_ep_cpu_tensor_io_active;

  if (s.cuda_ep_cpu_tensor_io_active) {
    s.summary =
        "Open Video eye contact: CUDA EP active through CPU ORT tensors; "
        "YuNet face detection, dlib landmarks, eye crops, anchor maps, "
        "warp/decode, resize, and composite remain explicit CPU tails. This "
        "is not a device-resident GPU path.";
  } else if (s.cpu_only_session_active) {
    s.summary =
        "Open Video eye contact: CPU ORT session active; YuNet face detection, "
        "dlib landmarks, eye crops, anchor maps, warp/decode, resize, and "
        "composite remain CPU-only. This is not a device-resident GPU path.";
  } else {
    s.summary =
        "Open Video eye contact: not initialized; the gaze-correction path "
        "uses CPU face detection, dlib landmarks, CPU tensor I/O, and CPU "
        "warp/composite when enabled. This is not a device-resident GPU path.";
  }

  return s;
}

EyeContactScratchStatus GazeCorrectionEyeContact::scratch_status() const {
  return EyeContactScratchStatus{
      scratch_geometry_rebuilds_, scratch_resize_plan_rebuilds_,
      scratch_frame_width_, scratch_frame_height_, upscale_scratch_.capacity()};
}

bool GazeCorrectionEyeContact::PrepareScratch(int frame_width, int frame_height,
                                              int left_input_width,
                                              int left_input_height,
                                              int right_input_width,
                                              int right_input_height,
                                              std::string *error) {
  if (error)
    error->clear();
  // CameraPipeline rejects frame geometry above 4096. Eye-contact packs are
  // explicitly bounded more tightly; curated packs currently use 64x48.
  constexpr int kMaxFrameDimension = 4096;
  constexpr int kMaxEyeInputDimension = 512;
  if (frame_width <= 0 || frame_height <= 0 ||
      frame_width > kMaxFrameDimension || frame_height > kMaxFrameDimension ||
      left_input_width <= 0 || left_input_height <= 0 ||
      right_input_width <= 0 || right_input_height <= 0 ||
      left_input_width > kMaxEyeInputDimension ||
      left_input_height > kMaxEyeInputDimension ||
      right_input_width > kMaxEyeInputDimension ||
      right_input_height > kMaxEyeInputDimension) {
    if (error)
      *error = "eye-contact scratch geometry exceeds its declared bounds";
    return false;
  }
  if (scratch_frame_width_ == frame_width &&
      scratch_frame_height_ == frame_height &&
      scratch_left_input_width_ == left_input_width &&
      scratch_left_input_height_ == left_input_height &&
      scratch_right_input_width_ == right_input_width &&
      scratch_right_input_height_ == right_input_height) {
    return true;
  }

  const auto prepare_eye = [](int width, int height, EyeData *eye,
                              std::vector<std::uint8_t> *corrected) {
    const std::size_t plane =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    eye->eye_rgb_u8.reserve(plane * 3u);
    eye->eye_nchw_f32.reserve(plane * 3u);
    eye->anchors_nchw_f32.reserve(plane * 12u);
    corrected->reserve(plane * 3u);
  };
  prepare_eye(left_input_width, left_input_height, &left_eye_scratch_,
              &left_corrected_scratch_);
  prepare_eye(right_input_width, right_input_height, &right_eye_scratch_,
              &right_corrected_scratch_);
  upscale_scratch_.reserve(static_cast<std::size_t>(frame_width) *
                           static_cast<std::size_t>(frame_height) * 3u);

  scratch_frame_width_ = frame_width;
  scratch_frame_height_ = frame_height;
  scratch_left_input_width_ = left_input_width;
  scratch_left_input_height_ = left_input_height;
  scratch_right_input_width_ = right_input_width;
  scratch_right_input_height_ = right_input_height;
  ++scratch_geometry_rebuilds_;
  return true;
}

bool GazeCorrectionEyeContact::EnsureResizePlan(bool left_eye, bool upscale,
                                                int src_w, int src_h, int dst_w,
                                                int dst_h, std::string *error) {
  auto *geometry =
      left_eye
          ? (upscale ? &left_upscale_geometry_ : &left_extract_geometry_)
          : (upscale ? &right_upscale_geometry_ : &right_extract_geometry_);
  auto *plan = left_eye
                   ? (upscale ? &left_upscale_plan_ : &left_extract_plan_)
                   : (upscale ? &right_upscale_plan_ : &right_extract_plan_);
  const std::array<int, 4> desired{src_w, src_h, dst_w, dst_h};
  if (*geometry == desired)
    return true;
  if (!plan->Configure(src_w, src_h, dst_w, dst_h, error))
    return false;
  *geometry = desired;
  ++scratch_resize_plan_rebuilds_;
  return true;
}

float GazeCorrectionEyeContact::Clamp01(float x) {
  if (x < 0.f)
    return 0.f;
  if (x > 1.f)
    return 1.f;
  return x;
}

float GazeCorrectionEyeContact::Lerp(float a, float b, float t) {
  return a + (b - a) * t;
}

std::string
GazeCorrectionEyeContact::ChoosePreferredModelId(const ModelPackRegistry &reg) {
  // Prefer the non-placeholder pack if multiple are installed.
  std::string fallback = reg.DefaultModelIdForTask("eye_contact");
  std::string best;
  for (const auto &p : reg.ListModels()) {
    if (p.task != "eye_contact")
      continue;
    if (best.empty() && p.id == fallback)
      best = p.id;
    if (p.id.find("placeholder") == std::string::npos) {
      // First non-placeholder wins.
      return p.id;
    }
  }
  return best.empty() ? fallback : best;
}

bool GazeCorrectionEyeContact::DetectEyeInputsFromSession(
    const studiocast::onnx::OrtSessionInfo &info, std::string *out_eye_name,
    std::vector<int64_t> *out_eye_shape, std::string *out_anchors_name,
    std::vector<int64_t> *out_anchors_shape, std::string *out_angles_name,
    std::vector<int64_t> *out_angles_shape, std::string *error) {
  if (error)
    error->clear();
  if (!out_eye_name || !out_eye_shape || !out_anchors_name ||
      !out_anchors_shape || !out_angles_name || !out_angles_shape) {
    if (error)
      *error = "DetectEyeInputsFromSession: null out params";
    return false;
  }

  int eye_idx = -1;
  int anchors_idx = -1;
  int angles_idx = -1;

  for (std::size_t i = 0; i < info.input_shapes.size(); ++i) {
    const auto &s = info.input_shapes[i];
    if (s.size() == 4) {
      const bool has_c3 = (s[1] == 3 || s[3] == 3);
      if (has_c3 && eye_idx < 0) {
        eye_idx = static_cast<int>(i);
        continue;
      }
      // Anchors are typically 12 channels (6 points * x/y maps), but allow 1 as
      // a fallback.
      const bool looks_like_anchor = (s[1] == 12 || s[3] == 12 || s[1] == 1 ||
                                      s[3] == 1 || s[1] == -1 || s[3] == -1);
      if (looks_like_anchor && anchors_idx < 0) {
        anchors_idx = static_cast<int>(i);
        continue;
      }
    } else if (s.size() == 2) {
      if ((s[0] == 1 && s[1] == 2) || (s[0] == 2) || (s[1] == 2)) {
        angles_idx = static_cast<int>(i);
      }
    } else if (s.size() == 1) {
      if (s[0] == 2)
        angles_idx = static_cast<int>(i);
    }
  }

  if (eye_idx < 0 || anchors_idx < 0 || angles_idx < 0) {
    if (error) {
      std::ostringstream oss;
      oss << "Unable to infer eye_contact model inputs (need eye_rgb, anchors, "
             "angles).";
      *error = oss.str();
    }
    return false;
  }

  *out_eye_name = info.input_names[static_cast<std::size_t>(eye_idx)];
  *out_eye_shape = info.input_shapes[static_cast<std::size_t>(eye_idx)];
  *out_anchors_name = info.input_names[static_cast<std::size_t>(anchors_idx)];
  *out_anchors_shape = info.input_shapes[static_cast<std::size_t>(anchors_idx)];
  *out_angles_name = info.input_names[static_cast<std::size_t>(angles_idx)];
  *out_angles_shape = info.input_shapes[static_cast<std::size_t>(angles_idx)];
  return true;
}

bool GazeCorrectionEyeContact::DetectOutputFromSession(
    const studiocast::onnx::OrtSessionInfo &info, std::string *out_name,
    std::vector<int64_t> *out_shape, std::string *error) {
  if (error)
    error->clear();
  if (!out_name || !out_shape) {
    if (error)
      *error = "DetectOutputFromSession: null out params";
    return false;
  }

  for (std::size_t i = 0; i < info.output_shapes.size(); ++i) {
    const auto &s = info.output_shapes[i];
    if (s.size() == 4) {
      const int64_t c = (s[1] > 0) ? s[1] : s[3];
      if (c == 2 || c == 3 || c == 1) {
        *out_name = info.output_names[i];
        *out_shape = s;
        return true;
      }
    }
  }

  if (error)
    *error = "Unable to infer eye_contact model output tensor.";
  return false;
}

bool GazeCorrectionEyeContact::LoadModelPack(const std::string &model_id,
                                             std::string *error) {
  if (error)
    error->clear();

  if (model_id.empty()) {
    if (error)
      *error = "No Open Video eye_contact model packs are installed.";
    return false;
  }

  const auto pack_opt = registry_.Find("eye_contact", model_id);
  if (!pack_opt.has_value()) {
    if (error)
      *error =
          "Open Video eye_contact model pack not found: id='" + model_id + "'";
    return false;
  }
  const ModelPack &pack = *pack_opt;

  active_model_id_ = pack.id;

  // Resolve dependency on a face_landmarks pack (best-effort).
  required_landmarks_id_.clear();
  for (const auto &dep : pack.depends_on) {
    const std::string prefix = "face_landmarks:";
    if (dep.rfind(prefix, 0) == 0 && dep.size() > prefix.size()) {
      required_landmarks_id_ = dep.substr(prefix.size());
      break;
    }
  }

  const ModelFile *left = nullptr;
  const ModelFile *right = nullptr;
  for (const auto &f : pack.files) {
    if (f.kind == "onnx" && f.role == "left")
      left = &f;
    if (f.kind == "onnx" && f.role == "right")
      right = &f;
  }
  if (!left || !right) {
    if (error)
      *error =
          "eye_contact pack missing ONNX files with roles 'left' and 'right'.";
    return false;
  }
  left_model_path_ = left->path;
  right_model_path_ = right->path;
  if (left_model_path_.empty() || right_model_path_.empty()) {
    if (error)
      *error = "eye_contact pack has empty model path(s).";
    return false;
  }
  return true;
}

bool GazeCorrectionEyeContact::InitRuntimeForEye(
    const std::filesystem::path &onnx_path, EyeRuntime *rt,
    std::string *error) {
  if (error)
    error->clear();
  if (!rt) {
    if (error)
      *error = "InitRuntimeForEye: rt is null";
    return false;
  }

  // Create both a CUDA-preferred session and a CPU-only session for fallback.
  studiocast::onnx::OrtSessionOptions cuda_opts;
  cuda_opts.prefer_cuda = true;
  studiocast::onnx::OrtSessionInfo cuda_info;
  std::string cuda_err;
  rt->session_cuda = studiocast::onnx::OrtSession::Create(
      onnx_path, cuda_opts, &cuda_info, &cuda_err);

  studiocast::onnx::OrtSessionOptions cpu_opts;
  cpu_opts.prefer_cuda = false;
  studiocast::onnx::OrtSessionInfo cpu_info;
  std::string cpu_err;
  rt->session_cpu = studiocast::onnx::OrtSession::Create(onnx_path, cpu_opts,
                                                         &cpu_info, &cpu_err);

  if (!rt->session_cuda && !rt->session_cpu) {
    if (error) {
      *error = "Failed to create ONNX sessions: cuda='" + cuda_err + "' cpu='" +
               cpu_err + "'";
    }
    return false;
  }

  // Prefer CUDA if it was created; otherwise use CPU.
  if (rt->session_cuda) {
    rt->session_active = rt->session_cuda.get();
    rt->session_info = cuda_info;
  } else {
    rt->session_active = rt->session_cpu.get();
    rt->session_info = cpu_info;
  }

  return ConfigureRuntimeIo(rt, error);
}

bool GazeCorrectionEyeContact::ConfigureRuntimeIo(EyeRuntime *rt,
                                                  std::string *error) {
  if (error)
    error->clear();
  if (!rt || !rt->session_active) {
    if (error)
      *error = "ConfigureRuntimeIo: session is null";
    return false;
  }

  std::string e;
  if (!DetectEyeInputsFromSession(
          rt->session_info, &rt->eye_name, &rt->eye_shape, &rt->anchors_name,
          &rt->anchors_shape, &rt->angles_name, &rt->angles_shape, &e)) {
    if (error)
      *error = e;
    return false;
  }

  if (!DetectOutputFromSession(rt->session_info, &rt->output_name,
                               &rt->output_shape, &e)) {
    if (error)
      *error = e;
    return false;
  }

  // Infer layout and input geometry.
  if (rt->eye_shape.size() == 4) {
    rt->eye_is_nhwc = (rt->eye_shape[3] == 3);
    const int64_t h = rt->eye_is_nhwc ? rt->eye_shape[1] : rt->eye_shape[2];
    const int64_t w = rt->eye_is_nhwc ? rt->eye_shape[2] : rt->eye_shape[3];
    if (h > 0)
      rt->input_h = static_cast<int>(h);
    if (w > 0)
      rt->input_w = static_cast<int>(w);
  }

  if (rt->anchors_shape.size() == 4) {
    const int64_t c = (rt->anchors_shape[1] > 0) ? rt->anchors_shape[1]
                                                 : rt->anchors_shape[3];
    if (c > 0)
      rt->anchor_channels = static_cast<int>(c);
    rt->anchors_is_nhwc = (rt->anchors_shape[3] == c);
  }

  if (rt->output_shape.size() == 4) {
    rt->output_is_nhwc = (rt->output_shape[3] > 0 && rt->output_shape[1] <= 0);
    const int64_t c =
        rt->output_is_nhwc ? rt->output_shape[3] : rt->output_shape[1];
    rt->out_channels = static_cast<int>(c);
  }

  // Allocate buffers based on the inferred shapes, replacing -1 with our
  // expected dims.
  auto fix_shape = [](std::vector<int64_t> *s, int w, int h, int c, bool nhwc) {
    if (s->size() != 4)
      return;
    (*s)[0] = 1;
    if (nhwc) {
      if ((*s)[1] <= 0)
        (*s)[1] = h;
      if ((*s)[2] <= 0)
        (*s)[2] = w;
      if ((*s)[3] <= 0)
        (*s)[3] = c;
    } else {
      if ((*s)[1] <= 0)
        (*s)[1] = c;
      if ((*s)[2] <= 0)
        (*s)[2] = h;
      if ((*s)[3] <= 0)
        (*s)[3] = w;
    }
  };

  fix_shape(&rt->eye_shape, rt->input_w, rt->input_h, 3, rt->eye_is_nhwc);
  fix_shape(&rt->anchors_shape, rt->input_w, rt->input_h, rt->anchor_channels,
            rt->anchors_is_nhwc);
  // Angles: [1,2] best-effort.
  if (rt->angles_shape.size() == 2) {
    rt->angles_shape[0] = 1;
    rt->angles_shape[1] = 2;
  } else if (rt->angles_shape.size() == 1) {
    rt->angles_shape[0] = 2;
  }
  fix_shape(&rt->output_shape, rt->input_w, rt->input_h, rt->out_channels,
            rt->output_is_nhwc);

  const std::size_t input_w_sz = static_cast<std::size_t>(rt->input_w);
  const std::size_t input_h_sz = static_cast<std::size_t>(rt->input_h);
  const std::size_t eye_elems =
      static_cast<std::size_t>(3) * input_w_sz * input_h_sz;
  const std::size_t anchor_elems =
      static_cast<std::size_t>(rt->anchor_channels) * input_w_sz * input_h_sz;
  const std::size_t out_elems =
      static_cast<std::size_t>(rt->out_channels) * input_w_sz * input_h_sz;
  rt->eye_tensor.assign(eye_elems, 0.f);
  rt->anchors_tensor.assign(anchor_elems, 0.f);
  rt->angles_tensor.assign(2, 0.f);
  rt->output_tensor.assign(out_elems, 0.f);

  rt->ort_inputs.clear();
  rt->ort_outputs.clear();
  rt->ort_inputs.push_back(studiocast::onnx::OrtSession::RunInput{
      rt->eye_name.c_str(), rt->eye_tensor.data(), rt->eye_tensor.size(),
      rt->eye_shape.data(), rt->eye_shape.size()});
  rt->ort_inputs.push_back(studiocast::onnx::OrtSession::RunInput{
      rt->anchors_name.c_str(), rt->anchors_tensor.data(),
      rt->anchors_tensor.size(), rt->anchors_shape.data(),
      rt->anchors_shape.size()});

  // Angles: shape rank can be 1 or 2.
  rt->ort_inputs.push_back(studiocast::onnx::OrtSession::RunInput{
      rt->angles_name.c_str(), rt->angles_tensor.data(), 2,
      rt->angles_shape.data(), rt->angles_shape.size()});
  rt->ort_outputs.push_back(studiocast::onnx::OrtSession::RunOutput{
      rt->output_name.c_str(), rt->output_tensor.data(),
      rt->output_tensor.size(), rt->output_shape.data(),
      rt->output_shape.size()});

  return true;
}

bool GazeCorrectionEyeContact::EnsureDlibDependency(std::string *error) {
  if (error)
    error->clear();

  std::string dep_id = required_landmarks_id_;
  if (!dlib_landmarks_.EnsureInitialized(dep_id, error)) {
    return false;
  }
  return true;
}

bool GazeCorrectionEyeContact::EnsureInitialized(
    const std::string &requested_model_id, std::string *error) {
  if (error)
    error->clear();

  if (disabled_) {
    if (error && !sticky_warning_.empty())
      *error = sticky_warning_;
    return false;
  }

  if (initialized_ && left_.session_active && right_.session_active &&
      requested_model_id == active_requested_model_id_) {
    return true;
  }

  // Resolve desired model id.
  ModelPackRegistry reg = ModelPackRegistry::ScanDefault();
  const std::string desired = requested_model_id.empty()
                                  ? ChoosePreferredModelId(reg)
                                  : requested_model_id;
  if (desired.empty()) {
    if (error)
      *error = "No Open Video eye_contact model packs are installed.";
    return false;
  }

  if (initialized_ && desired == active_model_id_) {
    registry_ = std::move(reg);
    active_requested_model_id_ = requested_model_id;
    return true;
  }

  if (initialized_) {
    Reset();
  }

  registry_ = std::move(reg);

#if !STUDIOCAST_HAVE_ONNXRUNTIME
  if (error)
    *error = "ONNX Runtime is not available in this build "
             "(STUDIOCAST_HAVE_ONNXRUNTIME=0).";
  return false;
#else
  std::string e;
  if (!LoadModelPack(desired, &e)) {
    if (error)
      *error = e;
    return false;
  }

  if (!EnsureDlibDependency(&e)) {
    if (error)
      *error = e;
    return false;
  }

  if (!InitRuntimeForEye(left_model_path_, &left_, &e)) {
    if (error)
      *error = "Eye Contact left model init failed: " + e;
    return false;
  }
  if (!InitRuntimeForEye(right_model_path_, &right_, &e)) {
    if (error)
      *error = "Eye Contact right model init failed: " + e;
    return false;
  }

  // Surface any warnings.
  if (!left_.session_info.warnings.empty()) {
    sticky_warning_ = left_.session_info.warnings.front();
  } else if (!right_.session_info.warnings.empty()) {
    sticky_warning_ = right_.session_info.warnings.front();
  }

  initialized_ = true;
  active_requested_model_id_ = requested_model_id;
  return true;
#endif
}

const FaceDetection *GazeCorrectionEyeContact::ChooseBestFace(
    const std::vector<FaceDetection> &faces) {
  if (faces.empty())
    return nullptr;
  const FaceDetection *best = &faces[0];
  float best_score = faces[0].score;
  float best_area = faces[0].w * faces[0].h;
  for (const auto &f : faces) {
    const float area = f.w * f.h;
    const float score = f.score;
    // Prefer higher score; tie-break by area.
    if (score > best_score + 1e-5f ||
        (std::abs(score - best_score) < 1e-5f && area > best_area)) {
      best = &f;
      best_score = score;
      best_area = area;
    }
  }
  return best;
}

bool GazeCorrectionEyeContact::ExtractEyeData(
    const std::uint8_t *rgb, int frame_w, int frame_h, std::size_t frame_stride,
    const FaceLandmarks &lms, bool left_eye, int input_w, int input_h,
    EyeData *out, std::string *error) {
  if (error)
    error->clear();
  if (!out) {
    if (error)
      *error = "ExtractEyeData: out is null";
    return false;
  }
  out->valid = false;
  out->is_left = left_eye;
  out->input_w = input_w;
  out->input_h = input_h;

  if (lms.points.size() < 68) {
    if (error)
      *error = "ExtractEyeData: expected 68 landmarks";
    return false;
  }

  // dlib landmark indices.
  const int idx0 = left_eye ? 42 : 36;
  const int idx1 = idx0 + 1;
  const int idx2 = idx0 + 2;
  const int idx3 = idx0 + 3;
  const int idx4 = idx0 + 4;
  const int idx5 = idx0 + 5;

  const auto p0 = lms.points[static_cast<std::size_t>(idx0)];
  const auto p1 = lms.points[static_cast<std::size_t>(idx1)];
  const auto p2 = lms.points[static_cast<std::size_t>(idx2)];
  const auto p3 = lms.points[static_cast<std::size_t>(idx3)];
  const auto p4 = lms.points[static_cast<std::size_t>(idx4)];
  const auto p5 = lms.points[static_cast<std::size_t>(idx5)];

  // Eye center from corner landmarks.
  const float cx = (p0.first + p3.first) * 0.5f;
  const float cy = (p0.second + p3.second) * 0.5f;

  const float eye_len = std::abs(p3.first - p0.first);
  const float bx_half_w = eye_len * 0.75f;
  const float bx_h = 1.5f * bx_half_w;
  const float sft_up = bx_h * (7.0f / 12.0f);
  const float sft_low = bx_h * (5.0f / 12.0f);

  int top = static_cast<int>(std::floor(cy - sft_up));
  int bottom = static_cast<int>(std::ceil(cy + sft_low));
  int left = static_cast<int>(std::floor(cx - bx_half_w));
  int right = static_cast<int>(std::ceil(cx + bx_half_w));

  top = ClampInt(top, 0, frame_h - 2);
  bottom = ClampInt(bottom, 1, frame_h - 1);
  left = ClampInt(left, 0, frame_w - 2);
  right = ClampInt(right, 1, frame_w - 1);

  const int crop_w = std::max(1, right - left);
  const int crop_h = std::max(1, bottom - top);
  if (crop_w < 2 || crop_h < 2) {
    // Nothing to do; treat as clean bypass.
    return false;
  }

  out->crop_top = top;
  out->crop_left = left;
  out->crop_w = crop_w;
  out->crop_h = crop_h;

  // Resize crop to the model input size.
  const std::uint8_t *crop_ptr = rgb + ByteOffset(frame_stride, left, top);
  std::string rerr;
  if (!EnsureResizePlan(left_eye, false, crop_w, crop_h, input_w, input_h,
                        &rerr) ||
      !(left_eye ? left_extract_plan_ : right_extract_plan_)
           .Apply(crop_ptr, frame_stride, &out->eye_rgb_u8,
                  static_cast<std::size_t>(input_w * 3), &rerr)) {
    if (error)
      *error = "Eye crop resize failed: " + rerr;
    return false;
  }
  // Convert eye to NCHW float32 in 0..1.
  const std::size_t input_w_sz = static_cast<std::size_t>(input_w);
  const std::size_t input_h_sz = static_cast<std::size_t>(input_h);
  const std::size_t plane = input_w_sz * input_h_sz;
  out->eye_nchw_f32.assign(static_cast<std::size_t>(3) * plane, 0.f);
  for (int y = 0; y < input_h; ++y) {
    for (int x = 0; x < input_w; ++x) {
      const std::size_t idx = static_cast<std::size_t>(y) * input_w_sz +
                              static_cast<std::size_t>(x);
      const std::size_t src = idx * 3u;
      out->eye_nchw_f32[idx] = out->eye_rgb_u8[src + 0] / 255.0f;
      out->eye_nchw_f32[plane + idx] = out->eye_rgb_u8[src + 1] / 255.0f;
      out->eye_nchw_f32[2 * plane + idx] = out->eye_rgb_u8[src + 2] / 255.0f;
    }
  }

  // Build anchor map (12 channels = 6 points * {x,y} offset maps).
  // Landmark order matches the upstream implementation:
  //   L: [3,2,1,0,5,4]
  //   R: [0,1,2,3,4,5]
  std::array<std::pair<float, float>, 6> pts = {p0, p1, p2, p3, p4, p5};
  std::array<int, 6> seq = left_eye ? std::array<int, 6>{3, 2, 1, 0, 5, 4}
                                    : std::array<int, 6>{0, 1, 2, 3, 4, 5};

  out->anchors_nchw_f32.assign(static_cast<std::size_t>(12) * plane, 0.f);

  for (int i = 0; i < 6; ++i) {
    const auto pt =
        pts[static_cast<std::size_t>(seq[static_cast<std::size_t>(i)])];
    const int resize_x = static_cast<int>(
        std::round((pt.first - static_cast<float>(left)) *
                   static_cast<float>(input_w) / static_cast<float>(crop_w)));
    const int resize_y = static_cast<int>(
        std::round((pt.second - static_cast<float>(top)) *
                   static_cast<float>(input_h) / static_cast<float>(crop_h)));

    const int c_x = 2 * i;
    const int c_y = 2 * i + 1;
    float *ch_x =
        out->anchors_nchw_f32.data() + static_cast<std::size_t>(c_x) * plane;
    float *ch_y =
        out->anchors_nchw_f32.data() + static_cast<std::size_t>(c_y) * plane;

    for (int y = 0; y < input_h; ++y) {
      const float dy = static_cast<float>(y - resize_y);
      for (int x = 0; x < input_w; ++x) {
        const std::size_t idx = static_cast<std::size_t>(y) * input_w_sz +
                                static_cast<std::size_t>(x);
        ch_x[idx] = static_cast<float>(x - resize_x);
        ch_y[idx] = dy;
      }
    }
  }

  out->valid = true;
  return true;
}

bool GazeCorrectionEyeContact::RunModelForEye(EyeRuntime *rt,
                                              const EyeData &eye, float yaw,
                                              float pitch, std::string *error) {
  if (error)
    error->clear();
  if (!rt || !rt->session_active) {
    if (error)
      *error = "RunModelForEye: rt/session is null";
    return false;
  }
  if (!eye.valid)
    return true;

  const int w = rt->input_w;
  const int h = rt->input_h;

  // Eye tensor.
  if (!rt->eye_is_nhwc) {
    // NCHW
    if (eye.eye_nchw_f32.size() != rt->eye_tensor.size()) {
      if (error)
        *error = "RunModelForEye: eye tensor size mismatch";
      return false;
    }
    rt->eye_tensor = eye.eye_nchw_f32;
  } else {
    // NHWC
    rt->eye_tensor.assign(rt->eye_tensor.size(), 0.f);
    const std::size_t w_sz = static_cast<std::size_t>(w);
    const std::size_t plane = w_sz * static_cast<std::size_t>(h);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const std::size_t idx =
            static_cast<std::size_t>(y) * w_sz + static_cast<std::size_t>(x);
        const std::size_t dst = idx * 3u;
        rt->eye_tensor[dst + 0] = eye.eye_nchw_f32[idx];
        rt->eye_tensor[dst + 1] = eye.eye_nchw_f32[plane + idx];
        rt->eye_tensor[dst + 2] = eye.eye_nchw_f32[2 * plane + idx];
      }
    }
  }

  // Anchors tensor.
  if (!rt->anchors_is_nhwc) {
    if (eye.anchors_nchw_f32.size() != rt->anchors_tensor.size()) {
      // Allow runtime anchor channels to differ; best-effort copy/clamp.
      const std::size_t want = rt->anchors_tensor.size();
      rt->anchors_tensor.assign(want, 0.f);
      const std::size_t have = eye.anchors_nchw_f32.size();
      const std::size_t copy = std::min(want, have);
      std::copy(eye.anchors_nchw_f32.begin(),
                eye.anchors_nchw_f32.begin() + static_cast<long>(copy),
                rt->anchors_tensor.begin());
    } else {
      rt->anchors_tensor = eye.anchors_nchw_f32;
    }
  } else {
    rt->anchors_tensor.assign(rt->anchors_tensor.size(), 0.f);
    const int C = rt->anchor_channels;
    const std::size_t C_sz = static_cast<std::size_t>(C);
    const std::size_t w_sz = static_cast<std::size_t>(w);
    const std::size_t plane = w_sz * static_cast<std::size_t>(h);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const std::size_t idx =
            static_cast<std::size_t>(y) * w_sz + static_cast<std::size_t>(x);
        const std::size_t dst_base = idx * C_sz;
        for (int c = 0; c < C; ++c) {
          const std::size_t src = static_cast<std::size_t>(c) * plane + idx;
          if (src < eye.anchors_nchw_f32.size()) {
            rt->anchors_tensor[dst_base + static_cast<std::size_t>(c)] =
                eye.anchors_nchw_f32[src];
          }
        }
      }
    }
  }

  // Angles.
  rt->angles_tensor.resize(2);
  rt->angles_tensor[0] = yaw;
  rt->angles_tensor[1] = pitch;

  std::string run_err;
  if (rt->session_active->RunCpu(rt->ort_inputs.data(), rt->ort_inputs.size(),
                                 rt->ort_outputs.data(), rt->ort_outputs.size(),
                                 &run_err)) {
    return true;
  }

  // Try CPU fallback if we aren't already using it.
  if (rt->session_active == rt->session_cuda.get() && rt->session_cpu) {
    rt->session_active = rt->session_cpu.get();
    using_cpu_fallback_ = true;
    if (rt->session_active->RunCpu(rt->ort_inputs.data(), rt->ort_inputs.size(),
                                   rt->ort_outputs.data(),
                                   rt->ort_outputs.size(), &run_err)) {
      return true;
    }
  }

  if (error)
    *error = run_err;
  return false;
}

bool GazeCorrectionEyeContact::WarpOrDecodeOutputToRgbU8(
    const EyeRuntime &rt, const EyeData &eye,
    std::vector<std::uint8_t> *out_rgb_u8, std::string *error) const {
  if (error)
    error->clear();
  if (!out_rgb_u8) {
    if (error)
      *error = "WarpOrDecodeOutputToRgbU8: out is null";
    return false;
  }

  const int w = rt.input_w;
  const int h = rt.input_h;
  const std::size_t w_sz = static_cast<std::size_t>(w);
  const std::size_t h_sz = static_cast<std::size_t>(h);
  const std::size_t plane = w_sz * h_sz;

  // If the model outputs RGB directly, just convert.
  if (rt.out_channels == 3) {
    out_rgb_u8->assign(plane * 3u, 0);
    if (!rt.output_is_nhwc) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const std::size_t idx =
              static_cast<std::size_t>(y) * w_sz + static_cast<std::size_t>(x);
          const float r = rt.output_tensor[idx];
          const float g = rt.output_tensor[plane + idx];
          const float b = rt.output_tensor[2 * plane + idx];
          const std::size_t dst = idx * 3u;
          (*out_rgb_u8)[dst + 0] = static_cast<std::uint8_t>(ClampInt(
              static_cast<int>(std::round(Clamp01(r) * 255.f)), 0, 255));
          (*out_rgb_u8)[dst + 1] = static_cast<std::uint8_t>(ClampInt(
              static_cast<int>(std::round(Clamp01(g) * 255.f)), 0, 255));
          (*out_rgb_u8)[dst + 2] = static_cast<std::uint8_t>(ClampInt(
              static_cast<int>(std::round(Clamp01(b) * 255.f)), 0, 255));
        }
      }
    } else {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const std::size_t idx =
              static_cast<std::size_t>(y) * w_sz + static_cast<std::size_t>(x);
          const std::size_t src = idx * 3u;
          const float r = rt.output_tensor[src + 0];
          const float g = rt.output_tensor[src + 1];
          const float b = rt.output_tensor[src + 2];
          const std::size_t dst = idx * 3u;
          (*out_rgb_u8)[dst + 0] = static_cast<std::uint8_t>(ClampInt(
              static_cast<int>(std::round(Clamp01(r) * 255.f)), 0, 255));
          (*out_rgb_u8)[dst + 1] = static_cast<std::uint8_t>(ClampInt(
              static_cast<int>(std::round(Clamp01(g) * 255.f)), 0, 255));
          (*out_rgb_u8)[dst + 2] = static_cast<std::uint8_t>(ClampInt(
              static_cast<int>(std::round(Clamp01(b) * 255.f)), 0, 255));
        }
      }
    }
    return true;
  }

  // Otherwise treat the output as a 2-channel flow field and warp the input
  // eye.
  if (rt.out_channels != 2) {
    if (error)
      *error = "Unexpected eye_contact output channels (expected 2 or 3).";
    return false;
  }

  out_rgb_u8->assign(plane * 3u, 0);

  // Warp using the input eye as the source.
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float flow_x = 0.f;
      float flow_y = 0.f;
      if (!rt.output_is_nhwc) {
        const std::size_t idx =
            static_cast<std::size_t>(y) * w_sz + static_cast<std::size_t>(x);
        flow_x = rt.output_tensor[idx];
        flow_y = rt.output_tensor[plane + idx];
      } else {
        const std::size_t idx =
            static_cast<std::size_t>(y) * w_sz + static_cast<std::size_t>(x);
        const std::size_t src = idx * 2u;
        flow_x = rt.output_tensor[src + 0];
        flow_y = rt.output_tensor[src + 1];
      }

      // Convention: pixel-space flow offsets.
      const float sx = static_cast<float>(x) + flow_x;
      const float sy = static_cast<float>(y) + flow_y;

      float r = 0.f, g = 0.f, b = 0.f;
      BilinearSampleRgbNchw(eye.eye_nchw_f32.data(), w, h, sx, sy, &r, &g, &b);

      const std::size_t idx =
          static_cast<std::size_t>(y) * w_sz + static_cast<std::size_t>(x);
      const std::size_t dst = idx * 3u;
      (*out_rgb_u8)[dst + 0] = static_cast<std::uint8_t>(
          ClampInt(static_cast<int>(std::round(Clamp01(r) * 255.f)), 0, 255));
      (*out_rgb_u8)[dst + 1] = static_cast<std::uint8_t>(
          ClampInt(static_cast<int>(std::round(Clamp01(g) * 255.f)), 0, 255));
      (*out_rgb_u8)[dst + 2] = static_cast<std::uint8_t>(
          ClampInt(static_cast<int>(std::round(Clamp01(b) * 255.f)), 0, 255));
    }
  }
  return true;
}

void GazeCorrectionEyeContact::CompositeEyeIntoFrame(
    std::uint8_t *frame_rgb, int frame_w, int frame_h, std::size_t frame_stride,
    const EyeData &eye, const std::vector<std::uint8_t> &corrected_rgb_u8,
    float strength01, bool left_eye) {
  if (!frame_rgb || !eye.valid)
    return;
  strength01 = Clamp01(strength01);
  if (strength01 <= 0.f)
    return;

  // Upscale corrected eye to the crop size.
  std::string rerr;
  if (!EnsureResizePlan(left_eye, true, eye.input_w, eye.input_h, eye.crop_w,
                        eye.crop_h, &rerr))
    return;
  auto &plan = left_eye ? left_upscale_plan_ : right_upscale_plan_;
  if (!plan.Apply(corrected_rgb_u8.data(),
                  static_cast<std::size_t>(eye.input_w * 3), &upscale_scratch_,
                  static_cast<std::size_t>(eye.crop_w * 3), &rerr))
    return;

  // Blend into the original frame.
  const int x0 = ClampInt(eye.crop_left, 0, frame_w - 1);
  const int y0 = ClampInt(eye.crop_top, 0, frame_h - 1);
  const int x1 = ClampInt(eye.crop_left + eye.crop_w, 0, frame_w);
  const int y1 = ClampInt(eye.crop_top + eye.crop_h, 0, frame_h);

  const int out_w = x1 - x0;
  const int out_h = y1 - y0;
  if (out_w <= 0 || out_h <= 0)
    return;

  for (int y = 0; y < out_h; ++y) {
    std::uint8_t *dst_row = frame_rgb +
                            static_cast<std::size_t>(y0 + y) * frame_stride +
                            static_cast<std::size_t>(x0) * 3;
    const std::uint8_t *src_row =
        upscale_scratch_.data() +
        static_cast<std::size_t>(y) * static_cast<std::size_t>(eye.crop_w) * 3;
    for (int x = 0; x < out_w; ++x) {
      const std::size_t d = static_cast<std::size_t>(x) * 3;
      const std::size_t s = static_cast<std::size_t>(x) * 3;
      for (int c = 0; c < 3; ++c) {
        const float o = dst_row[d + static_cast<std::size_t>(c)];
        const float n = src_row[s + static_cast<std::size_t>(c)];
        const float v = Lerp(o, n, strength01);
        dst_row[d + static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(
            ClampInt(static_cast<int>(std::round(v)), 0, 255));
      }
    }
  }
}

bool GazeCorrectionEyeContact::ApplyRgbInPlace(
    std::uint64_t capture_sequence, std::uint8_t *rgb, int width, int height,
    std::size_t stride, int strength, bool look_away_enabled,
    const std::string &face_detection_model_id,
    const std::string &requested_model_id, YunetFaceDetector *yunet,
    FrameAnalysisCache *cache, std::string *error) {
  if (error)
    error->clear();
  (void)look_away_enabled;

  if (!rgb || width <= 0 || height <= 0 ||
      stride < static_cast<std::size_t>(width * 3)) {
    if (error)
      *error = "ApplyRgbInPlace: invalid RGB frame";
    return false;
  }
  if (!yunet) {
    if (error)
      *error = "ApplyRgbInPlace: yunet is null";
    return false;
  }
  if (!cache) {
    if (error)
      *error = "ApplyRgbInPlace: cache is null";
    return false;
  }

  if (strength <= 0)
    return true;
  const float strength01 = Clamp01(static_cast<float>(strength) / 100.f);

  std::string init_err;
  if (!EnsureInitialized(requested_model_id, &init_err)) {
    // If not initialized (missing models), treat as bypass rather than hard
    // failure.
    if (error)
      *error = init_err;
    return false;
  }

  std::string scratch_error;
  if (!PrepareScratch(width, height, left_.input_w, left_.input_h,
                      right_.input_w, right_.input_h, &scratch_error)) {
    if (error)
      *error = scratch_error;
    return false;
  }

  // Ensure face detections.
  std::string det_err;
  if (!yunet->EnsureDetectionsForFrame(rgb, width, height, stride,
                                       face_detection_model_id,
                                       capture_sequence, cache, &det_err)) {
    if (error)
      *error = det_err;
    return false;
  }
  if (!cache->face_detections.has_value() || cache->face_detections->empty()) {
    return true; // no face => bypass
  }

  const FaceDetection *face = ChooseBestFace(*cache->face_detections);
  if (!face)
    return true;

  // Landmarks (cached).
  std::string lm_err;
  if (!dlib_landmarks_.EnsureLandmarksForFrame(rgb, width, height, stride,
                                               capture_sequence, *face, cache,
                                               &lm_err)) {
    if (error)
      *error = lm_err;
    return false;
  }
  if (!cache->face_landmarks.has_value())
    return true;

  // Extract eye inputs.
  std::string ex_err;
  const auto &lms = *cache->face_landmarks;
  if (!ExtractEyeData(rgb, width, height, stride, lms, true, left_.input_w,
                      left_.input_h, &left_eye_scratch_, &ex_err)) {
    // No usable eye region.
  }
  if (!ExtractEyeData(rgb, width, height, stride, lms, false, right_.input_w,
                      right_.input_h, &right_eye_scratch_, &ex_err)) {
    // No usable eye region.
  }

  if (!left_eye_scratch_.valid && !right_eye_scratch_.valid)
    return true;

  // Placeholder gaze angles: aim slightly upward towards a typical webcam
  // position. Strength scales the effective angle.
  const float yaw = 0.0f * strength01;
  const float pitch = -0.10f * strength01;

  // Run models + composite.
  std::string run_err;

  if (left_eye_scratch_.valid) {
    if (!RunModelForEye(&left_, left_eye_scratch_, yaw, pitch, &run_err)) {
      if (error)
        *error = run_err;
      runtime_failures_++;
      if (runtime_failures_ >= 3) {
        DisableAfterFailure(
            "Open Video eye contact disabled after repeated failures.");
      }
      return false;
    }
    std::string werr;
    if (WarpOrDecodeOutputToRgbU8(left_, left_eye_scratch_,
                                  &left_corrected_scratch_, &werr)) {
      CompositeEyeIntoFrame(rgb, width, height, stride, left_eye_scratch_,
                            left_corrected_scratch_, strength01, true);
    }
  }

  if (right_eye_scratch_.valid) {
    if (!RunModelForEye(&right_, right_eye_scratch_, yaw, pitch, &run_err)) {
      if (error)
        *error = run_err;
      runtime_failures_++;
      if (runtime_failures_ >= 3) {
        DisableAfterFailure(
            "Open Video eye contact disabled after repeated failures.");
      }
      return false;
    }
    std::string werr;
    if (WarpOrDecodeOutputToRgbU8(right_, right_eye_scratch_,
                                  &right_corrected_scratch_, &werr)) {
      CompositeEyeIntoFrame(rgb, width, height, stride, right_eye_scratch_,
                            right_corrected_scratch_, strength01, false);
    }
  }

  return true;
}

} // namespace studiocast::open_video
