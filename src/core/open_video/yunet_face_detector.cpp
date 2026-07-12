#include "core/open_video/yunet_face_detector.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>

#include "core/util/fs.h"
#include "core/util/json.h"
#include "core/util/xdg.h"

namespace studiocast::open_video {

namespace {

// Helper to pick a "best" model id from a list of candidates.
// Our curated packs follow a *_fp32 / *_int8bq / *_int8 naming scheme.
std::string PickPreferredModelId(const std::vector<std::string> &model_ids) {
  if (model_ids.empty())
    return "";

  auto has_id = [&](const std::string &needle) {
    return std::find(model_ids.begin(), model_ids.end(), needle) !=
           model_ids.end();
  };

  // Prefer higher quality variants when available.
  // (These ids are defined in
  // resources/model_packs/open_video/face_detection/*.)
  const std::vector<std::string> prefer = {
      "yunet_opencv_zoo_fp32",
      "yunet_opencv_zoo_int8bq",
      "yunet_opencv_zoo_int8",
  };
  for (const auto &id : prefer) {
    if (has_id(id))
      return id;
  }

  // Otherwise, fall back to the first discovered pack.
  return model_ids.front();
}

// JSON helpers.
std::optional<double> JsonNumber(const util::json::Value::Object &obj,
                                 const std::string &key) {
  auto it = obj.find(key);
  if (it == obj.end())
    return std::nullopt;
  if (const double *n = it->second.AsNumber())
    return *n;
  return std::nullopt;
}

const util::json::Value::Object *
JsonObject(const util::json::Value::Object &obj, const std::string &key) {
  auto it = obj.find(key);
  if (it == obj.end())
    return nullptr;
  return it->second.AsObject();
}

} // namespace

void YunetFaceDetector::Reset() {
  initialized_ = false;
  input_is_nhwc_ = false;
  settings_ = Settings{};
  active_model_id_.clear();
  active_requested_model_id_.clear();
  active_model_path_.clear();
  session_.reset();
  session_info_ = studiocast::onnx::OrtSessionInfo{};
  input_tensor_.clear();
  input_shape_.clear();
  outputs_.clear();
  cls_idx_ = {{-1, -1, -1}};
  obj_idx_ = {{-1, -1, -1}};
  bbox_idx_ = {{-1, -1, -1}};
  kps_idx_ = {{-1, -1, -1}};
}

FaceDetectionRuntimeStatus YunetFaceDetector::runtime_status() const {
  FaceDetectionRuntimeStatus s;
  s.cuda_ep_active = session_ != nullptr && session_info_.using_cuda;
  s.cuda_ep_cpu_tensor_io_active = s.cuda_ep_active;
  s.cpu_only_session_active = session_ != nullptr && !session_info_.using_cuda;

  if (s.cuda_ep_cpu_tensor_io_active) {
    s.summary =
        "Open Video face detection: CUDA EP active through CPU ORT tensors; "
        "letterbox/BGR preprocess, detection decode/NMS, and CPU-visible face "
        "boxes remain explicit CPU tails. This is not a device-resident GPU "
        "path.";
  } else if (s.cpu_only_session_active) {
    s.summary =
        "Open Video face detection: CPU ORT session active; letterbox/BGR "
        "preprocess and detection decode/NMS remain CPU-only. This is not a "
        "device-resident GPU path.";
  } else {
    s.summary =
        "Open Video face detection: not initialized; YuNet uses CPU tensor "
        "preprocess/postprocess when enabled. This is not a "
        "device-resident GPU path.";
  }

  return s;
}

bool YunetFaceDetector::LoadSettingsFromManifest(
    const std::filesystem::path &manifest_path, std::string *error) {
  if (error)
    error->clear();

  const auto text = util::ReadTextFile(manifest_path.string());
  if (!text) {
    if (error) {
      *error = "Failed to read model manifest: " + manifest_path.string();
    }
    return false;
  }

  util::json::Value root;
  std::string parse_err;
  if (!util::json::Parse(*text, &root, &parse_err)) {
    if (error) {
      *error = "Failed to parse model manifest JSON: " + parse_err;
    }
    return false;
  }

  const auto *top = root.AsObject();
  if (!top)
    return true; // Treat unknown schema as "use defaults".

  const auto *onnx = JsonObject(*top, "onnx");
  if (!onnx)
    return true;

  if (const auto *pre = JsonObject(*onnx, "preprocess")) {
    if (auto w = JsonNumber(*pre, "width"))
      settings_.input_w = static_cast<int>(*w);
    if (auto h = JsonNumber(*pre, "height"))
      settings_.input_h = static_cast<int>(*h);
  }

  if (const auto *post = JsonObject(*onnx, "postprocess")) {
    if (auto s = JsonNumber(*post, "score_threshold"))
      settings_.score_threshold = static_cast<float>(*s);
    if (auto n = JsonNumber(*post, "nms_threshold"))
      settings_.nms_threshold = static_cast<float>(*n);
    if (auto k = JsonNumber(*post, "top_k"))
      settings_.top_k = static_cast<int>(*k);
  }

  // Clamp to sane values.
  settings_.input_w = std::clamp(settings_.input_w, 64, 2048);
  settings_.input_h = std::clamp(settings_.input_h, 64, 2048);
  settings_.score_threshold = std::clamp(settings_.score_threshold, 0.0f, 1.0f);
  settings_.nms_threshold = std::clamp(settings_.nms_threshold, 0.0f, 1.0f);
  settings_.top_k = std::clamp(settings_.top_k, 1, 20000);

  return true;
}

bool YunetFaceDetector::BuildBindings(std::string *error) {
  if (error)
    error->clear();
  if (!session_) {
    if (error)
      *error = "ORT session is not initialized";
    return false;
  }

  if (session_info_.input_shapes.empty() || session_info_.input_names.empty()) {
    if (error)
      *error = "ORT session input metadata missing";
    return false;
  }

  // Determine input layout.
  input_is_nhwc_ = false;
  const auto &ishape = session_info_.input_shapes[0];
  if (ishape.size() == 4) {
    if (ishape[3] == 3) {
      input_is_nhwc_ = true;
    } else if (ishape[1] == 3) {
      input_is_nhwc_ = false;
    }
  }

  input_shape_.clear();
  if (input_is_nhwc_) {
    input_shape_ = {1, settings_.input_h, settings_.input_w, 3};
  } else {
    input_shape_ = {1, 3, settings_.input_h, settings_.input_w};
  }

  const std::size_t in_floats = static_cast<std::size_t>(settings_.input_w) *
                                static_cast<std::size_t>(settings_.input_h) *
                                3u;
  input_tensor_.assign(in_floats, 0.0f);

  // Build output bindings.
  outputs_.clear();
  outputs_.reserve(session_info_.output_names.size());

  auto stride_index = [](const std::string &name) -> int {
    // Accept "*_8", "*_16", "*_32".
    if (name.size() >= 2 && name.rfind("_8") == name.size() - 2)
      return 0;
    if (name.size() >= 3 && name.rfind("_16") == name.size() - 3)
      return 1;
    if (name.size() >= 3 && name.rfind("_32") == name.size() - 3)
      return 2;
    return -1;
  };

  auto allocate_output = [&](const std::string &name,
                             const std::vector<int64_t> &declared_shape) {
    OutputBinding ob;
    ob.name = name;

    int stride = 0;
    const int si = stride_index(name);
    switch (si) {
      case 0:
        stride = 8;
        break;
      case 1:
        stride = 16;
        break;
      case 2:
        stride = 32;
        break;
      default:
        stride = 8;
        break;
    }

    const int rows = std::max(1, settings_.input_h / stride);
    const int cols = std::max(1, settings_.input_w / stride);

    int channels = 1;
    if (name.rfind("bbox_", 0) == 0)
      channels = 4;
    else if (name.rfind("kps_", 0) == 0)
      channels = 10;
    else
      channels = 1;

    // Determine layout from declared shape.
    bool nhwc = true;
    if (declared_shape.size() == 4) {
      if (declared_shape[3] == channels)
        nhwc = true;
      else if (declared_shape[1] == channels)
        nhwc = false;
    }

    if (nhwc) {
      ob.shape = {1, rows, cols, channels};
    } else {
      ob.shape = {1, channels, rows, cols};
    }

    std::size_t n = 1;
    for (const auto d : ob.shape) {
      if (d <= 0)
        continue;
      n *= static_cast<std::size_t>(d);
    }
    ob.data.assign(n, 0.0f);
    outputs_.push_back(std::move(ob));
  };

  for (std::size_t i = 0; i < session_info_.output_names.size(); ++i) {
    const auto &name = session_info_.output_names[i];
    const auto &shape = (i < session_info_.output_shapes.size())
                            ? session_info_.output_shapes[i]
                            : std::vector<int64_t>{};
    allocate_output(name, shape);
  }

  auto find_out = [&](const std::string &name) -> int {
    for (std::size_t i = 0; i < outputs_.size(); ++i) {
      if (outputs_[i].name == name)
        return static_cast<int>(i);
    }
    return -1;
  };

  cls_idx_ = {{find_out("cls_8"), find_out("cls_16"), find_out("cls_32")}};
  obj_idx_ = {{find_out("obj_8"), find_out("obj_16"), find_out("obj_32")}};
  bbox_idx_ = {{find_out("bbox_8"), find_out("bbox_16"), find_out("bbox_32")}};
  kps_idx_ = {{find_out("kps_8"), find_out("kps_16"), find_out("kps_32")}};

  for (std::size_t i = 0; i < 3; ++i) {
    if (cls_idx_[i] < 0 || obj_idx_[i] < 0 || bbox_idx_[i] < 0 ||
        kps_idx_[i] < 0) {
      if (error) {
        std::ostringstream oss;
        oss << "YuNet outputs missing expected tensors for stride="
            << (i == 0 ? 8 : (i == 1 ? 16 : 32)) << ". Available outputs:";
        for (const auto &o : outputs_)
          oss << " " << o.name;
        *error = oss.str();
      }
      return false;
    }
  }

  return true;
}

bool YunetFaceDetector::EnsureInitialized(const std::string &requested_model_id,
                                          std::string *error) {
  if (error)
    error->clear();

  if (initialized_ && session_ &&
      requested_model_id == active_requested_model_id_) {
    return true;
  }

  ModelPackRegistry reg = ModelPackRegistry::ScanDefault();
  const auto it = reg.Tasks().find("face_detection");
  if (it == reg.Tasks().end() || it->second.empty()) {
    if (error) {
      *error = "No Open Video face_detection models installed. Install a YuNet "
               "model pack under " +
               (util::StudioCastModelsDir().string() +
                "/open_video/face_detection/.");
    }
    return false;
  }

  const std::string model_id = requested_model_id.empty()
                                   ? PickPreferredModelId(it->second)
                                   : requested_model_id;
  if (initialized_ && model_id == active_model_id_) {
    registry_ = std::move(reg);
    active_requested_model_id_ = requested_model_id;
    return true;
  }

  Reset();
  registry_ = std::move(reg);

  const auto pack = registry_.Find("face_detection", model_id);
  if (!pack) {
    if (error) {
      *error =
          requested_model_id.empty()
              ? ("Failed to resolve face_detection model_id '" + model_id + "'")
              : ("Requested face_detection model_id '" + model_id +
                 "' is not installed");
    }
    return false;
  }

  // Find the main ONNX file.
  std::filesystem::path model_path;
  for (const auto &f : pack->files) {
    if (f.role == "main" && f.kind == "onnx") {
      model_path = f.path;
      break;
    }
  }
  if (model_path.empty()) {
    if (error)
      *error = "YuNet pack is missing a role=main kind=onnx file";
    return false;
  }

  // Load tunables from model.json (best-effort).
  std::string settings_err;
  (void)LoadSettingsFromManifest(pack->manifest_path, &settings_err);

  studiocast::onnx::OrtSessionOptions opts;
  opts.prefer_cuda = true;
  opts.cuda_device_id = 0;
  std::string ort_err;
  auto sess = studiocast::onnx::OrtSession::Create(model_path, opts,
                                                   &session_info_, &ort_err);
  if (!sess) {
    if (error)
      *error = ort_err.empty()
                   ? ("Failed to create ORT session for " + model_path.string())
                   : ort_err;
    return false;
  }

  session_ = std::move(sess);
  active_model_id_ = model_id;
  active_requested_model_id_ = requested_model_id;
  active_model_path_ = model_path;

  std::string bind_err;
  if (!BuildBindings(&bind_err)) {
    if (error)
      *error = bind_err;
    Reset();
    return false;
  }

  initialized_ = true;
  return true;
}

YunetFaceDetector::Letterbox YunetFaceDetector::ComputeLetterbox(int src_w,
                                                                 int src_h,
                                                                 int dst_w,
                                                                 int dst_h) {
  Letterbox lb;
  lb.out_w = std::max(1, dst_w);
  lb.out_h = std::max(1, dst_h);

  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
    lb.scale = 1.0f;
    lb.pad_x = 0;
    lb.pad_y = 0;
    return lb;
  }

  const float sx = static_cast<float>(dst_w) / static_cast<float>(src_w);
  const float sy = static_cast<float>(dst_h) / static_cast<float>(src_h);
  lb.scale = std::min(sx, sy);

  const int resized_w =
      std::max(1, static_cast<int>(std::lround(src_w * lb.scale)));
  const int resized_h =
      std::max(1, static_cast<int>(std::lround(src_h * lb.scale)));
  lb.pad_x = (dst_w - resized_w) / 2;
  lb.pad_y = (dst_h - resized_h) / 2;
  return lb;
}

void YunetFaceDetector::FillInputTensorBgr(const std::uint8_t *rgb, int src_w,
                                           int src_h, std::size_t src_stride,
                                           const Letterbox &lb) {
  const int out_w = settings_.input_w;
  const int out_h = settings_.input_h;

  // Clear to black (letterbox padding).
  std::fill(input_tensor_.begin(), input_tensor_.end(), 0.0f);

  if (!rgb || src_w <= 1 || src_h <= 1)
    return;

  const float inv_scale = (lb.scale > 0.0f) ? (1.0f / lb.scale) : 1.0f;
  const int content_w =
      std::max(1, static_cast<int>(std::lround(src_w * lb.scale)));
  const int content_h =
      std::max(1, static_cast<int>(std::lround(src_h * lb.scale)));
  const int x0 = std::max(0, lb.pad_x);
  const int y0 = std::max(0, lb.pad_y);
  const int x1 = std::min(out_w, lb.pad_x + content_w);
  const int y1 = std::min(out_h, lb.pad_y + content_h);

  const std::size_t plane =
      static_cast<std::size_t>(out_w) * static_cast<std::size_t>(out_h);

  for (int oy = y0; oy < y1; ++oy) {
    // Map to source y.
    const float sy =
        (static_cast<float>(oy - lb.pad_y) + 0.5f) * inv_scale - 0.5f;
    int y_base = static_cast<int>(std::floor(sy));
    float ty = sy - static_cast<float>(y_base);
    if (y_base < 0) {
      y_base = 0;
      ty = 0.0f;
    }
    int y_next = y_base + 1;
    if (y_next >= src_h) {
      y_next = src_h - 1;
      ty = 0.0f;
    }

    const std::uint8_t *row0 =
        rgb + static_cast<std::size_t>(y_base) * src_stride;
    const std::uint8_t *row1 =
        rgb + static_cast<std::size_t>(y_next) * src_stride;

    for (int ox = x0; ox < x1; ++ox) {
      const float sx =
          (static_cast<float>(ox - lb.pad_x) + 0.5f) * inv_scale - 0.5f;
      int x_base = static_cast<int>(std::floor(sx));
      float tx = sx - static_cast<float>(x_base);
      if (x_base < 0) {
        x_base = 0;
        tx = 0.0f;
      }
      int x_next = x_base + 1;
      if (x_next >= src_w) {
        x_next = src_w - 1;
        tx = 0.0f;
      }

      const std::uint8_t *p00 = row0 + static_cast<std::size_t>(x_base) * 3;
      const std::uint8_t *p01 = row0 + static_cast<std::size_t>(x_next) * 3;
      const std::uint8_t *p10 = row1 + static_cast<std::size_t>(x_base) * 3;
      const std::uint8_t *p11 = row1 + static_cast<std::size_t>(x_next) * 3;

      // Input is RGB; model expects BGR.
      const float r00 = static_cast<float>(p00[0]);
      const float g00 = static_cast<float>(p00[1]);
      const float b00 = static_cast<float>(p00[2]);
      const float r01 = static_cast<float>(p01[0]);
      const float g01 = static_cast<float>(p01[1]);
      const float b01 = static_cast<float>(p01[2]);
      const float r10 = static_cast<float>(p10[0]);
      const float g10 = static_cast<float>(p10[1]);
      const float b10 = static_cast<float>(p10[2]);
      const float r11 = static_cast<float>(p11[0]);
      const float g11 = static_cast<float>(p11[1]);
      const float b11 = static_cast<float>(p11[2]);

      const float r0 = r00 + (r01 - r00) * tx;
      const float g0 = g00 + (g01 - g00) * tx;
      const float b0 = b00 + (b01 - b00) * tx;
      const float r1 = r10 + (r11 - r10) * tx;
      const float g1 = g10 + (g11 - g10) * tx;
      const float b1 = b10 + (b11 - b10) * tx;

      const float r = r0 + (r1 - r0) * ty;
      const float g = g0 + (g1 - g0) * ty;
      const float b = b0 + (b1 - b0) * ty;

      const std::size_t idx =
          static_cast<std::size_t>(oy) * static_cast<std::size_t>(out_w) +
          static_cast<std::size_t>(ox);
      if (input_is_nhwc_) {
        const std::size_t base = idx * 3u;
        input_tensor_[base + 0] = b;
        input_tensor_[base + 1] = g;
        input_tensor_[base + 2] = r;
      } else {
        input_tensor_[idx + 0u * plane] = b;
        input_tensor_[idx + 1u * plane] = g;
        input_tensor_[idx + 2u * plane] = r;
      }
    }
  }
}

float YunetFaceDetector::IoU(const FaceDetection &a, const FaceDetection &b) {
  const float ax0 = a.x;
  const float ay0 = a.y;
  const float ax1 = a.x + a.w;
  const float ay1 = a.y + a.h;
  const float bx0 = b.x;
  const float by0 = b.y;
  const float bx1 = b.x + b.w;
  const float by1 = b.y + b.h;

  const float ix0 = std::max(ax0, bx0);
  const float iy0 = std::max(ay0, by0);
  const float ix1 = std::min(ax1, bx1);
  const float iy1 = std::min(ay1, by1);

  const float iw = std::max(0.0f, ix1 - ix0);
  const float ih = std::max(0.0f, iy1 - iy0);
  const float inter = iw * ih;
  const float area_a = std::max(0.0f, ax1 - ax0) * std::max(0.0f, ay1 - ay0);
  const float area_b = std::max(0.0f, bx1 - bx0) * std::max(0.0f, by1 - by0);
  const float uni = area_a + area_b - inter;
  return (uni > 0.0f) ? (inter / uni) : 0.0f;
}

bool YunetFaceDetector::EnsureDetectionsForFrame(
    const std::uint8_t *rgb, int width, int height, std::size_t stride,
    const std::string &requested_model_id, std::uint64_t capture_sequence,
    FrameAnalysisCache *cache, std::string *error) {
  if (error)
    error->clear();
  if (!cache) {
    if (error)
      *error = "FrameAnalysisCache is null";
    return false;
  }

  cache->BeginFrame(capture_sequence);
  if (cache->face_detections.has_value()) {
    return true;
  }

  std::string init_err;
  if (!EnsureInitialized(requested_model_id, &init_err)) {
    if (error)
      *error = init_err;
    return false;
  }

  const Letterbox lb =
      ComputeLetterbox(width, height, settings_.input_w, settings_.input_h);
  FillInputTensorBgr(rgb, width, height, stride, lb);

  // Bind input.
  studiocast::onnx::OrtSession::RunInput in;
  in.name = session_info_.input_names[0].c_str();
  in.data = input_tensor_.data();
  in.num_floats = input_tensor_.size();
  in.shape = input_shape_.data();
  in.shape_rank = input_shape_.size();

  // Bind outputs.
  std::vector<studiocast::onnx::OrtSession::RunOutput> outs;
  outs.reserve(outputs_.size());
  for (auto &ob : outputs_) {
    studiocast::onnx::OrtSession::RunOutput o;
    o.name = ob.name.c_str();
    o.data = ob.data.data();
    o.num_floats = ob.data.size();
    o.shape = ob.shape.data();
    o.shape_rank = ob.shape.size();
    outs.push_back(o);
  }

  std::string run_err;
  if (!session_->RunCpu(&in, 1, outs.data(), outs.size(), &run_err)) {
    if (error)
      *error = run_err;
    return false;
  }

  // Decode YuNet outputs.
  std::vector<FaceDetection> faces;
  faces.reserve(16);

  const int padW = settings_.input_w;
  const int padH = settings_.input_h;
  const int strides[3] = {8, 16, 32};

  for (std::size_t s = 0; s < 3; ++s) {
    const int stride_px = strides[s];
    const int cols = padW / stride_px;
    const int rows = padH / stride_px;
    if (cols <= 0 || rows <= 0)
      continue;

    const auto &cls = outputs_[static_cast<std::size_t>(cls_idx_[s])];
    const auto &obj = outputs_[static_cast<std::size_t>(obj_idx_[s])];
    const auto &bbox = outputs_[static_cast<std::size_t>(bbox_idx_[s])];
    const auto &kps = outputs_[static_cast<std::size_t>(kps_idx_[s])];

    const bool bbox_nhwc = (bbox.shape.size() == 4 && bbox.shape[3] == 4);
    const bool kps_nhwc = (kps.shape.size() == 4 && kps.shape[3] == 10);

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        const int idx = r * cols + c;

        // cls/obj are 1-channel so layout does not matter.
        float cls_score =
            std::clamp(cls.data[static_cast<std::size_t>(idx)], 0.0f, 1.0f);
        float obj_score =
            std::clamp(obj.data[static_cast<std::size_t>(idx)], 0.0f, 1.0f);
        float score = std::sqrt(cls_score * obj_score);
        if (score < settings_.score_threshold)
          continue;

        auto read_bbox = [&](int k) -> float {
          if (bbox_nhwc) {
            return bbox.data[static_cast<std::size_t>(idx) * 4u +
                             static_cast<std::size_t>(k)];
          }
          const std::size_t plane =
              static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
          return bbox.data[static_cast<std::size_t>(k) * plane +
                           static_cast<std::size_t>(idx)];
        };

        // OpenCV FaceDetectorYN decode.
        const float cx = (static_cast<float>(c) + read_bbox(0)) *
                         static_cast<float>(stride_px);
        const float cy = (static_cast<float>(r) + read_bbox(1)) *
                         static_cast<float>(stride_px);
        const float w = std::exp(read_bbox(2)) * static_cast<float>(stride_px);
        const float h = std::exp(read_bbox(3)) * static_cast<float>(stride_px);
        const float x1 = cx - w * 0.5f;
        const float y1 = cy - h * 0.5f;

        (void)kps;
        (void)kps_nhwc;
        // Keypoints are available but are not currently needed for Auto Frame.

        faces.push_back(FaceDetection{
            .x = x1,
            .y = y1,
            .w = w,
            .h = h,
            .score = score,
        });
      }
    }
  }

  // Sort by score descending.
  std::sort(faces.begin(), faces.end(),
            [](const FaceDetection &a, const FaceDetection &b) {
              return a.score > b.score;
            });
  if (static_cast<int>(faces.size()) > settings_.top_k) {
    faces.resize(static_cast<std::size_t>(settings_.top_k));
  }

  // Map to original frame space and run NMS.
  std::vector<FaceDetection> mapped;
  mapped.reserve(faces.size());
  for (const auto &f : faces) {
    FaceDetection out = f;
    // Undo letterbox.
    out.x = (out.x - static_cast<float>(lb.pad_x)) / lb.scale;
    out.y = (out.y - static_cast<float>(lb.pad_y)) / lb.scale;
    out.w = out.w / lb.scale;
    out.h = out.h / lb.scale;

    // Clip to image bounds (best-effort).
    out.x = std::clamp(out.x, 0.0f, static_cast<float>(width));
    out.y = std::clamp(out.y, 0.0f, static_cast<float>(height));
    out.w = std::clamp(out.w, 0.0f, static_cast<float>(width) - out.x);
    out.h = std::clamp(out.h, 0.0f, static_cast<float>(height) - out.y);
    mapped.push_back(out);
  }

  std::vector<FaceDetection> kept;
  kept.reserve(mapped.size());
  for (const auto &cand : mapped) {
    bool suppress = false;
    for (const auto &prev : kept) {
      if (IoU(cand, prev) > settings_.nms_threshold) {
        suppress = true;
        break;
      }
    }
    if (!suppress) {
      kept.push_back(cand);
    }
  }

  cache->face_detections = std::move(kept);
  return true;
}

} // namespace studiocast::open_video
