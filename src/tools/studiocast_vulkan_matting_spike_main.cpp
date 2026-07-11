#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "core/onnx/ort_session.h"
#include "core/open_video/model_pack_registry.h"
#include "core/util/xdg.h"
#include "core/video/image_ppm.h"

#if STUDIOCAST_ENABLE_OPEN_VULKAN
#include "core/vulkan/vulkan_device.h"
#endif

#if STUDIOCAST_HAVE_NCNN && STUDIOCAST_NCNN_HAS_VULKAN
#if __has_include(<net.h>)
#include <net.h>
#else
#include <ncnn/net.h>
#endif
#if __has_include(<gpu.h>)
#include <gpu.h>
#else
#include <ncnn/gpu.h>
#endif
#endif

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  fs::path models_root;
  std::string model_id;
  fs::path param_path;
  fs::path bin_path;
  std::string ncnn_input_blob;
  std::string ncnn_output_blob;
  std::string fixture = "synthetic";
  std::string reference = "auto";
  int width = 1280;
  int height = 720;
  bool width_set = false;
  bool height_set = false;
  int warmup = 20;
  int iterations = 120;
  int ncnn_device = 0;
  double threshold = 0.5;
  bool csv = false;
  bool require_candidate = false;
  bool require_reference = false;
  bool allow_ncnn_cpu_layers = false;
};

struct ResolvedModel {
  studiocast::open_video::ModelPack pack;
  fs::path onnx_path;
  fs::path param_path;
  fs::path bin_path;
  std::string input_name;
  std::string output_name;
  int width = 0;
  int height = 0;
};

struct FrameRgb {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> rgb;
};

struct TimingStats {
  double min_ms = 0.0;
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  double max_ms = 0.0;
};

struct QualityMetrics {
  double mae = std::numeric_limits<double>::quiet_NaN();
  double max_abs = std::numeric_limits<double>::quiet_NaN();
  double mask_iou = std::numeric_limits<double>::quiet_NaN();
  bool available = false;
};

struct RunResult {
  bool ok = false;
  std::string status = "skipped";
  std::string reason;
  double startup_ms = std::numeric_limits<double>::quiet_NaN();
  double first_run_ms = std::numeric_limits<double>::quiet_NaN();
  TimingStats steady;
  std::vector<float> alpha;
  std::size_t implicit_uploads = 0;
  std::size_t alpha_downloads = 0;
  std::size_t forced_syncs = 0;
  bool input_device_resident = false;
  bool output_device_resident = false;
  std::vector<std::string> warnings;
};

void Usage(const char *argv0) {
  std::cout
      << "StudioCast Vulkan Matting Inference Spike\n\n"
      << "Usage:\n"
      << "  " << argv0
      << " [--models-root <open_video_root>] [--model-id <id>]\n"
      << "      [--param <model.ncnn.param>] [--bin <model.ncnn.bin>]\n"
      << "      [--ncnn-input <blob>] [--ncnn-output <blob>]\n"
      << "      [--fixture synthetic|<image.png>|<image.ppm>]\n"
      << "      [--width <W>] [--height <H>]\n"
      << "      [--warmup <N>] [--iterations <N>] [--threshold <0..1>]\n"
      << "      [--reference auto|cpu-ort|none] [--csv]\n"
      << "      [--require-candidate] [--require-reference]\n"
      << "      [--allow-ncnn-cpu-layers]\n\n"
      << "Notes:\n"
      << "  - Conversion is offline. This tool never converts ONNX in the run loop.\n"
      << "  - Default ncnn artifacts are searched beside model.onnx as\n"
      << "    model.ncnn.param/model.ncnn.bin or model.ncnn.opt.param/model.ncnn.opt.bin.\n";
}

bool ParseInt(const std::string &s, int *out) {
  if (!out)
    return false;
  try {
    std::size_t pos = 0;
    const int v = std::stoi(s, &pos);
    if (pos != s.size())
      return false;
    *out = v;
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseDouble(const std::string &s, double *out) {
  if (!out)
    return false;
  try {
    std::size_t pos = 0;
    const double v = std::stod(s, &pos);
    if (pos != s.size())
      return false;
    *out = v;
    return true;
  } catch (...) {
    return false;
  }
}

bool NeedValue(int i, int argc, char **argv, std::string *error) {
  if (i + 1 < argc)
    return true;
  if (error)
    *error = std::string("missing value for ") + argv[i];
  return false;
}

bool ParseArgs(int argc, char **argv, Options *out, std::string *error) {
  if (!out)
    return false;
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i] ? std::string_view(argv[i]) : "";
    auto next = [&]() -> std::string {
      return (i + 1 < argc && argv[i + 1]) ? std::string(argv[++i]) : "";
    };

    if (arg == "--help" || arg == "-h") {
      Usage(argv[0]);
      std::exit(0);
    } else if (arg == "--models-root") {
      if (!NeedValue(i, argc, argv, error))
        return false;
      opts.models_root = next();
    } else if (arg == "--model-id") {
      if (!NeedValue(i, argc, argv, error))
        return false;
      opts.model_id = next();
    } else if (arg == "--param") {
      if (!NeedValue(i, argc, argv, error))
        return false;
      opts.param_path = next();
    } else if (arg == "--bin") {
      if (!NeedValue(i, argc, argv, error))
        return false;
      opts.bin_path = next();
    } else if (arg == "--ncnn-input") {
      if (!NeedValue(i, argc, argv, error))
        return false;
      opts.ncnn_input_blob = next();
    } else if (arg == "--ncnn-output") {
      if (!NeedValue(i, argc, argv, error))
        return false;
      opts.ncnn_output_blob = next();
    } else if (arg == "--fixture") {
      if (!NeedValue(i, argc, argv, error))
        return false;
      opts.fixture = next();
    } else if (arg == "--reference") {
      if (!NeedValue(i, argc, argv, error))
        return false;
      opts.reference = next();
    } else if (arg == "--width") {
      if (!NeedValue(i, argc, argv, error))
        return false;
      if (!ParseInt(next(), &opts.width)) {
        if (error)
          *error = "invalid --width";
        return false;
      }
      opts.width_set = true;
    } else if (arg == "--height") {
      if (!NeedValue(i, argc, argv, error))
        return false;
      if (!ParseInt(next(), &opts.height)) {
        if (error)
          *error = "invalid --height";
        return false;
      }
      opts.height_set = true;
    } else if (arg == "--warmup") {
      if (!NeedValue(i, argc, argv, error))
        return false;
      if (!ParseInt(next(), &opts.warmup)) {
        if (error)
          *error = "invalid --warmup";
        return false;
      }
    } else if (arg == "--iterations") {
      if (!NeedValue(i, argc, argv, error))
        return false;
      if (!ParseInt(next(), &opts.iterations)) {
        if (error)
          *error = "invalid --iterations";
        return false;
      }
    } else if (arg == "--threshold") {
      if (!NeedValue(i, argc, argv, error))
        return false;
      if (!ParseDouble(next(), &opts.threshold)) {
        if (error)
          *error = "invalid --threshold";
        return false;
      }
    } else if (arg == "--ncnn-device") {
      if (!NeedValue(i, argc, argv, error))
        return false;
      if (!ParseInt(next(), &opts.ncnn_device)) {
        if (error)
          *error = "invalid --ncnn-device";
        return false;
      }
    } else if (arg == "--csv") {
      opts.csv = true;
    } else if (arg == "--require-candidate") {
      opts.require_candidate = true;
    } else if (arg == "--require-reference") {
      opts.require_reference = true;
    } else if (arg == "--allow-ncnn-cpu-layers") {
      opts.allow_ncnn_cpu_layers = true;
    } else {
      if (error)
        *error = std::string("unknown argument: ") + std::string(arg);
      return false;
    }
  }

  if (opts.width <= 0 || opts.height <= 0 || opts.warmup < 0 ||
      opts.iterations <= 0 || opts.threshold < 0.0 || opts.threshold > 1.0) {
    if (error)
      *error = "invalid dimensions, iteration count, or threshold";
    return false;
  }
  if (opts.reference != "auto" && opts.reference != "cpu-ort" &&
      opts.reference != "none") {
    if (error)
      *error = "--reference must be auto, cpu-ort, or none";
    return false;
  }
  *out = std::move(opts);
  return true;
}

std::string YesNo(bool v) { return v ? "yes" : "no"; }

double MsSince(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

TimingStats ComputeStats(std::vector<double> values) {
  TimingStats stats;
  if (values.empty())
    return stats;
  std::sort(values.begin(), values.end());
  const auto percentile = [&](double q) {
    const double idx = q * static_cast<double>(values.size() - 1);
    return values[static_cast<std::size_t>(std::round(idx))];
  };
  stats.min_ms = values.front();
  stats.max_ms = values.back();
  stats.p50_ms = percentile(0.50);
  stats.p95_ms = percentile(0.95);
  stats.p99_ms = percentile(0.99);
  stats.mean_ms = std::accumulate(values.begin(), values.end(), 0.0) /
                  static_cast<double>(values.size());
  return stats;
}

std::string FormatDouble(double v, int precision = 3) {
  if (!std::isfinite(v))
    return "n/a";
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << v;
  return oss.str();
}

const studiocast::open_video::ModelFile *
FindMainOnnxFile(const studiocast::open_video::ModelPack &p) {
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

bool ExistsRegular(const fs::path &p) {
  std::error_code ec;
  return fs::is_regular_file(p, ec) && !ec;
}

fs::path DefaultOpenVideoRoot() {
  const auto models_root = studiocast::util::StudioCastModelsDir();
  return models_root.empty() ? fs::path{} : models_root / "open_video";
}

std::optional<std::pair<fs::path, fs::path>>
FindNcnnArtifacts(const fs::path &pack_root, const fs::path &param_override,
                  const fs::path &bin_override) {
  if (!param_override.empty() || !bin_override.empty()) {
    if (!param_override.empty() && !bin_override.empty() &&
        ExistsRegular(param_override) && ExistsRegular(bin_override)) {
      return std::make_pair(param_override, bin_override);
    }
    return std::nullopt;
  }

  const std::vector<std::pair<fs::path, fs::path>> candidates = {
      {pack_root / "model.ncnn.opt.param", pack_root / "model.ncnn.opt.bin"},
      {pack_root / "model.ncnn.param", pack_root / "model.ncnn.bin"},
      {pack_root / "model.opt.param", pack_root / "model.opt.bin"},
      {pack_root / "model.param", pack_root / "model.bin"},
  };
  for (const auto &candidate : candidates) {
    if (ExistsRegular(candidate.first) && ExistsRegular(candidate.second))
      return candidate;
  }
  return std::nullopt;
}

bool ResolveModel(const Options &opts, ResolvedModel *out, std::string *error) {
  if (!out)
    return false;
  const fs::path root =
      opts.models_root.empty() ? DefaultOpenVideoRoot() : opts.models_root;
  if (root.empty()) {
    if (error)
      *error = "could not resolve Open Video models root";
    return false;
  }

  const auto reg = studiocast::open_video::ModelPackRegistry::Scan(root);
  std::string id = opts.model_id;
  if (id.empty())
    id = reg.DefaultModelIdForTask("matting");
  if (id.empty()) {
    if (error)
      *error = "no Open Video matting model packs found under " +
               root.string();
    return false;
  }

  const auto pack = reg.Find("matting", id);
  if (!pack) {
    if (error)
      *error = "matting model id not found: " + id;
    return false;
  }
  if (!pack->matting) {
    if (error)
      *error = "selected model pack is missing matting metadata: " + id;
    return false;
  }

  const auto *onnx = FindMainOnnxFile(*pack);
  if (!onnx) {
    if (error)
      *error = "selected matting model pack has no ONNX file: " + id;
    return false;
  }

  const auto artifacts =
      FindNcnnArtifacts(pack->root_dir, opts.param_path, opts.bin_path);

  ResolvedModel model;
  model.pack = *pack;
  model.onnx_path = onnx->path;
  if (artifacts) {
    model.param_path = artifacts->first;
    model.bin_path = artifacts->second;
  }
  model.input_name = opts.ncnn_input_blob.empty()
                         ? pack->matting->input.name
                         : opts.ncnn_input_blob;
  model.output_name = opts.ncnn_output_blob.empty()
                          ? pack->matting->output.name
                          : opts.ncnn_output_blob;
  model.width = pack->matting->input.width;
  model.height = pack->matting->input.height;
  *out = std::move(model);
  return true;
}

std::vector<std::string> CompatibilityNotes(const ResolvedModel &model) {
  std::vector<std::string> notes;
  if (model.pack.id == "mock_model") {
    notes.push_back("operator_compatibility: mock ReduceMean fixture; plumbing "
                    "only, not real matting quality");
    return notes;
  }
  if (model.pack.id.find("modnet") != std::string::npos) {
    notes.push_back("operator_compatibility: MODNet ONNX is known to contain "
                    "Shape/Gather/Unsqueeze/Resize plus InstanceNormalization; "
                    "conversion and Vulkan layer support must be verified");
    notes.push_back("cpu_fallback_risk: InstanceNormalization nodes are a "
                    "likely ncnn Vulkan fallback risk");
  } else if (model.pack.id.find("birefnet") != std::string::npos) {
    notes.push_back("operator_compatibility: BiRefNet conversion must use the "
                    "post-install patched alpha/Sigmoid output and 1024x1024 "
                    "shape");
  } else {
    notes.push_back("operator_compatibility: unknown until pnnx/onnx2ncnn "
                    "conversion logs and ncnn layer Vulkan support are checked");
  }
  return notes;
}

void PrintConversionPlan(const ResolvedModel &model) {
  std::cout << "\nOffline conversion commands:\n";
  std::cout << "  cd " << model.pack.root_dir.string() << "\n";
  std::cout << "  pnnx " << model.onnx_path.filename().string()
            << " inputshape=[1,3," << model.height << "," << model.width
            << "] ncnnparam=model.ncnn.param ncnnbin=model.ncnn.bin fp16=0 "
               "optlevel=2\n";
  std::cout << "  # legacy fallback:\n";
  std::cout << "  python3 -m onnxsim " << model.onnx_path.filename().string()
            << " model.sim.onnx\n";
  std::cout << "  onnx2ncnn model.sim.onnx model.ncnn.param "
               "model.ncnn.bin\n";
  std::cout << "  ncnnoptimize model.ncnn.param model.ncnn.bin "
               "model.ncnn.opt.param model.ncnn.opt.bin 65536\n";
}

std::vector<std::uint8_t> GenerateSyntheticRgb(int w, int h) {
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(w) *
                                static_cast<std::size_t>(h) * 3u);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t i =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
           static_cast<std::size_t>(x)) *
          3u;
      rgb[i + 0] = static_cast<std::uint8_t>((x * 255) / std::max(1, w - 1));
      rgb[i + 1] = static_cast<std::uint8_t>((y * 255) / std::max(1, h - 1));
      rgb[i + 2] = static_cast<std::uint8_t>(((x ^ y) * 31) & 0xff);
    }
  }
  return rgb;
}

bool LoadFrame(const Options &opts, FrameRgb *out, std::string *error) {
  if (!out)
    return false;
  if (opts.fixture == "synthetic") {
    out->width = opts.width;
    out->height = opts.height;
    out->rgb = GenerateSyntheticRgb(out->width, out->height);
    return true;
  }

  int loaded_w = 0;
  int loaded_h = 0;
  std::vector<std::uint8_t> loaded;
  if (!studiocast::video::LoadImageRgb24(opts.fixture, &loaded_w, &loaded_h,
                                         &loaded, error)) {
    return false;
  }

  int target_w = opts.width_set ? opts.width : loaded_w;
  int target_h = opts.height_set ? opts.height : loaded_h;
  if (target_w == loaded_w && target_h == loaded_h) {
    out->width = loaded_w;
    out->height = loaded_h;
    out->rgb = std::move(loaded);
    return true;
  }

  std::vector<std::uint8_t> resized;
  if (!studiocast::video::ResizeRgb24Bilinear(
          loaded.data(), loaded_w, loaded_h,
          static_cast<std::size_t>(loaded_w) * 3u, target_w, target_h,
          &resized, static_cast<std::size_t>(target_w) * 3u, error)) {
    return false;
  }
  out->width = target_w;
  out->height = target_h;
  out->rgb = std::move(resized);
  return true;
}

struct AxisSample {
  int i0 = 0;
  int i1 = 0;
  float t = 0.0f;
};

AxisSample SampleAxis(int src_len, int dst_len, int dst_i) {
  const float scale = static_cast<float>(src_len) / static_cast<float>(dst_len);
  const float src = (static_cast<float>(dst_i) + 0.5f) * scale - 0.5f;
  const int i0 = std::clamp(static_cast<int>(src), 0, src_len - 1);
  return {i0, std::clamp(i0 + 1, 0, src_len - 1),
          src - static_cast<float>(i0)};
}

std::vector<float> PreprocessCpuNchw(const FrameRgb &frame,
                                     const ResolvedModel &model) {
  const auto &spec = *model.pack.matting;
  const int dst_w = spec.input.width;
  const int dst_h = spec.input.height;
  const std::size_t hw =
      static_cast<std::size_t>(dst_w) * static_cast<std::size_t>(dst_h);
  std::vector<float> out(3u * hw);

  for (int y = 0; y < dst_h; ++y) {
    const AxisSample ys = SampleAxis(frame.height, dst_h, y);
    for (int x = 0; x < dst_w; ++x) {
      const AxisSample xs = SampleAxis(frame.width, dst_w, x);
      for (int c = 0; c < 3; ++c) {
        const auto channel = static_cast<std::size_t>(c);
        const auto at = [&](int sx, int sy) {
          const std::size_t idx =
              (static_cast<std::size_t>(sy) *
                   static_cast<std::size_t>(frame.width) +
               static_cast<std::size_t>(sx)) *
                  3u +
              channel;
          return static_cast<float>(frame.rgb[idx]);
        };
        const float v0 =
            at(xs.i0, ys.i0) + xs.t * (at(xs.i1, ys.i0) - at(xs.i0, ys.i0));
        const float v1 =
            at(xs.i0, ys.i1) + xs.t * (at(xs.i1, ys.i1) - at(xs.i0, ys.i1));
        const float rgb01 = (v0 + ys.t * (v1 - v0)) / 255.0f;
        out[channel * hw +
            static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_w) +
            static_cast<std::size_t>(x)] =
            static_cast<float>((rgb01 - spec.preprocess.mean[channel]) /
                               spec.preprocess.std[channel]);
      }
    }
  }
  return out;
}

RunResult RunCpuOrtReference(const ResolvedModel &model,
                             const std::vector<float> &input,
                             const Options &opts) {
  RunResult result;
  if (opts.reference == "none") {
    result.status = "skipped";
    result.reason = "reference disabled";
    return result;
  }

#if !STUDIOCAST_HAVE_ONNXRUNTIME
  result.status = "skipped";
  result.reason = "built without ONNX Runtime";
  (void)model;
  (void)input;
  return result;
#else
  const auto start = Clock::now();
  studiocast::onnx::OrtSessionOptions ort_opts;
  ort_opts.prefer_cuda = false;
  studiocast::onnx::OrtSessionInfo info;
  std::string err;
  auto session = studiocast::onnx::OrtSession::Create(model.onnx_path, ort_opts,
                                                      &info, &err);
  const auto created = Clock::now();
  result.startup_ms = MsSince(start, created);
  if (!session) {
    result.status = "failed";
    result.reason = err.empty() ? "CPU/ORT session creation failed" : err;
    return result;
  }

  const std::int64_t input_shape[4] = {1, 3, model.height, model.width};
  const std::int64_t output_shape[4] = {1, 1, model.height, model.width};
  const std::size_t output_count =
      static_cast<std::size_t>(model.width) * static_cast<std::size_t>(model.height);
  result.alpha.assign(output_count, 0.0f);

  const auto run_once = [&]() -> bool {
    studiocast::onnx::OrtSession::RunInput in;
    in.name = model.pack.matting->input.name.c_str();
    in.data = input.data();
    in.num_floats = input.size();
    in.shape = input_shape;
    in.shape_rank = 4;

    studiocast::onnx::OrtSession::RunOutput out;
    out.name = model.pack.matting->output.name.c_str();
    out.data = result.alpha.data();
    out.num_floats = result.alpha.size();
    out.shape = output_shape;
    out.shape_rank = 4;
    return session->RunCpu(&in, 1, &out, 1, &err);
  };

  const auto first_start = Clock::now();
  if (!run_once()) {
    result.status = "failed";
    result.reason = err.empty() ? "CPU/ORT run failed" : err;
    return result;
  }
  result.first_run_ms = MsSince(first_start, Clock::now());

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(opts.iterations));
  for (int i = 0; i < opts.warmup; ++i) {
    if (!run_once()) {
      result.status = "failed";
      result.reason = err.empty() ? "CPU/ORT warmup failed" : err;
      return result;
    }
  }
  for (int i = 0; i < opts.iterations; ++i) {
    const auto t0 = Clock::now();
    if (!run_once()) {
      result.status = "failed";
      result.reason = err.empty() ? "CPU/ORT timed run failed" : err;
      return result;
    }
    samples.push_back(MsSince(t0, Clock::now()));
  }
  result.steady = ComputeStats(std::move(samples));
  result.ok = true;
  result.status = "ok";
  result.reason = "cpu-ort";
  return result;
#endif
}

QualityMetrics CompareAlpha(const std::vector<float> &ref,
                            const std::vector<float> &candidate,
                            double threshold) {
  QualityMetrics q;
  if (ref.empty() || ref.size() != candidate.size())
    return q;

  double abs_sum = 0.0;
  double max_abs = 0.0;
  std::size_t intersection = 0;
  std::size_t uni = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double diff =
        std::abs(static_cast<double>(ref[i]) - static_cast<double>(candidate[i]));
    abs_sum += diff;
    max_abs = std::max(max_abs, diff);
    const bool r = static_cast<double>(ref[i]) >= threshold;
    const bool c = static_cast<double>(candidate[i]) >= threshold;
    if (r && c)
      ++intersection;
    if (r || c)
      ++uni;
  }
  q.mae = abs_sum / static_cast<double>(ref.size());
  q.max_abs = max_abs;
  q.mask_iou = uni == 0 ? 1.0 : static_cast<double>(intersection) /
                               static_cast<double>(uni);
  q.available = true;
  return q;
}

#if STUDIOCAST_HAVE_NCNN && STUDIOCAST_NCNN_HAS_VULKAN
class NcnnSpikeSession {
public:
  ~NcnnSpikeSession() { Shutdown(); }

  bool Initialize(const ResolvedModel &model, const Options &opts,
                  std::string *error) {
    model_ = &model;
    opts_ = &opts;
    const int gpu_rc = ncnn::create_gpu_instance();
    gpu_instance_created_ = gpu_rc == 0;
    if (!gpu_instance_created_) {
      if (error)
        *error = "ncnn create_gpu_instance failed";
      return false;
    }
    const int gpu_count = ncnn::get_gpu_count();
    if (gpu_count <= 0) {
      if (error)
        *error = "ncnn Vulkan runtime found no GPU devices";
      return false;
    }
    if (opts.ncnn_device < 0 || opts.ncnn_device >= gpu_count) {
      if (error)
        *error = "ncnn device index out of range";
      return false;
    }

    net_.opt.use_vulkan_compute = true;
    net_.opt.use_fp16_packed = false;
    net_.opt.use_fp16_storage = false;
    net_.opt.use_fp16_arithmetic = false;
    net_.opt.use_int8_storage = false;
    net_.opt.use_int8_arithmetic = false;
    net_.opt.use_packing_layout = false;
    net_.opt.vulkan_device_index = opts.ncnn_device;
    net_.set_vulkan_device(opts.ncnn_device);

    if (net_.load_param(model.param_path.string().c_str()) != 0) {
      if (error)
        *error = "ncnn load_param failed: " + model.param_path.string();
      return false;
    }
    if (net_.load_model(model.bin_path.string().c_str()) != 0) {
      if (error)
        *error = "ncnn load_model failed: " + model.bin_path.string();
      return false;
    }

    for (const ncnn::Layer *layer : net_.layers()) {
      if (!layer)
        continue;
      if (layer->type == "Input")
        continue;
      if (!layer->support_vulkan) {
        non_vulkan_layers_.push_back(layer->type + ":" + layer->name);
      }
    }
    if (!non_vulkan_layers_.empty() && !opts.allow_ncnn_cpu_layers) {
      if (error) {
        *error = "ncnn graph contains layers without Vulkan support; rerun "
                 "with --allow-ncnn-cpu-layers to measure fallback debt";
      }
      return false;
    }
    return true;
  }

  bool Run(const std::vector<float> &input, std::vector<float> *output,
           std::string *error) {
    if (!model_ || !opts_) {
      if (error)
        *error = "ncnn session is not initialized";
      return false;
    }
    ncnn::Mat in(model_->width, model_->height, 3);
    const std::size_t hw = static_cast<std::size_t>(model_->width) *
                           static_cast<std::size_t>(model_->height);
    for (int c = 0; c < 3; ++c) {
      float *dst = in.channel(c);
      std::copy(input.data() + static_cast<std::size_t>(c) * hw,
                input.data() + static_cast<std::size_t>(c + 1) * hw, dst);
    }

    ncnn::Extractor ex = net_.create_extractor();
    ex.set_light_mode(true);
    if (ex.input(model_->input_name.c_str(), in) != 0) {
      if (error)
        *error = "ncnn input failed for blob '" + model_->input_name + "'";
      return false;
    }

    ncnn::Mat out_mat;
    if (ex.extract(model_->output_name.c_str(), out_mat) != 0) {
      if (error)
        *error = "ncnn extract failed for blob '" + model_->output_name + "'";
      return false;
    }
    if (out_mat.elempack != 1) {
      if (error)
        *error = "ncnn output elempack is not 1; spike requested unpacked fp32";
      return false;
    }

    const std::size_t expected = hw;
    output->clear();
    output->reserve(expected);
    if (out_mat.dims == 3) {
      if (out_mat.c != 1 || out_mat.w != model_->width ||
          out_mat.h != model_->height) {
        if (error)
          *error = "ncnn output shape mismatch";
        return false;
      }
      const float *src = out_mat.channel(0);
      output->assign(src, src + expected);
    } else if (out_mat.dims == 2) {
      if (out_mat.w != model_->width || out_mat.h != model_->height) {
        if (error)
          *error = "ncnn output shape mismatch";
        return false;
      }
      const float *src = out_mat;
      output->assign(src, src + expected);
    } else if (out_mat.dims == 1 && out_mat.w == static_cast<int>(expected)) {
      const float *src = out_mat;
      output->assign(src, src + expected);
    } else {
      if (error)
        *error = "unsupported ncnn output dims: " + std::to_string(out_mat.dims);
      return false;
    }
    return true;
  }

  const std::vector<std::string> &non_vulkan_layers() const {
    return non_vulkan_layers_;
  }

private:
  void Shutdown() noexcept {
    net_.clear();
    if (gpu_instance_created_) {
      ncnn::destroy_gpu_instance();
      gpu_instance_created_ = false;
    }
  }

  ncnn::Net net_;
  const ResolvedModel *model_ = nullptr;
  const Options *opts_ = nullptr;
  bool gpu_instance_created_ = false;
  std::vector<std::string> non_vulkan_layers_;
};
#endif

RunResult RunNcnnCandidate(const ResolvedModel &model,
                           const std::vector<float> &input,
                           const Options &opts) {
  RunResult result;
  result.input_device_resident = false;
  result.output_device_resident = false;

  if (model.param_path.empty() || model.bin_path.empty()) {
    result.status = "skipped";
    result.reason =
        "converted ncnn model is missing (model.ncnn.param/model.ncnn.bin)";
    return result;
  }

#if !(STUDIOCAST_HAVE_NCNN && STUDIOCAST_NCNN_HAS_VULKAN)
  result.status = "skipped";
#if !STUDIOCAST_HAVE_NCNN
  result.reason = "built without ncnn";
#else
  result.reason = "ncnn was found but not built with Vulkan support";
#endif
  (void)input;
  (void)opts;
  return result;
#else
  NcnnSpikeSession session;
  std::string err;
  const auto start = Clock::now();
  if (!session.Initialize(model, opts, &err)) {
    result.status = "failed";
    result.reason = err.empty() ? "ncnn session initialization failed" : err;
    for (const auto &layer : session.non_vulkan_layers()) {
      result.warnings.push_back("non_vulkan_layer: " + layer);
    }
    return result;
  }
  const auto created = Clock::now();
  result.startup_ms = MsSince(start, created);
  for (const auto &layer : session.non_vulkan_layers()) {
    result.warnings.push_back("non_vulkan_layer_allowed: " + layer);
  }
  result.warnings.push_back(
      "device_residency: ncnn spike uses CPU Mat input/output; ncnn owns any "
      "internal Vulkan uploads/downloads");

  const auto first_start = Clock::now();
  if (!session.Run(input, &result.alpha, &err)) {
    result.status = "failed";
    result.reason = err.empty() ? "ncnn first run failed" : err;
    return result;
  }
  result.first_run_ms = MsSince(first_start, Clock::now());

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(opts.iterations));
  for (int i = 0; i < opts.warmup; ++i) {
    if (!session.Run(input, &result.alpha, &err)) {
      result.status = "failed";
      result.reason = err.empty() ? "ncnn warmup failed" : err;
      return result;
    }
  }
  for (int i = 0; i < opts.iterations; ++i) {
    const auto t0 = Clock::now();
    if (!session.Run(input, &result.alpha, &err)) {
      result.status = "failed";
      result.reason = err.empty() ? "ncnn timed run failed" : err;
      return result;
    }
    samples.push_back(MsSince(t0, Clock::now()));
  }
  result.steady = ComputeStats(std::move(samples));
  const std::size_t run_count =
      static_cast<std::size_t>(1 + opts.warmup + opts.iterations);
  result.implicit_uploads = run_count;
  result.alpha_downloads = run_count;
  result.forced_syncs = run_count;
  result.ok = true;
  result.status = "ok";
  result.reason = "ncnn-vulkan";
  return result;
#endif
}

void PrintBuildDiagnostics() {
  std::cout << "Build diagnostics:\n";
  std::cout << "  ncnn compiled in: " << YesNo(STUDIOCAST_HAVE_NCNN) << "\n";
  std::cout << "  ncnn Vulkan build: " << YesNo(STUDIOCAST_NCNN_HAS_VULKAN)
            << "\n";
  std::cout << "  StudioCast Open Vulkan compiled: "
            << YesNo(STUDIOCAST_ENABLE_OPEN_VULKAN) << "\n";
#if STUDIOCAST_ENABLE_OPEN_VULKAN
  std::string err;
  const auto diag = studiocast::vulkan::DiagnoseOpenVulkanDefault();
  std::cout << "  StudioCast Vulkan runtime: " << (diag.ok ? "ok" : "missing")
            << "\n";
  std::cout << "  StudioCast Vulkan device: "
            << (diag.device_name.empty() ? "(none)" : diag.device_name)
            << "\n";
  if (!diag.error.empty())
    std::cout << "  StudioCast Vulkan error: " << diag.error << "\n";
  (void)err;
#else
  std::cout << "  StudioCast Vulkan runtime: not checked (backend disabled)\n";
#endif
}

void PrintRun(const char *label, const RunResult &r) {
  std::cout << "\n" << label << ":\n";
  std::cout << "  status: " << r.status << "\n";
  if (!r.reason.empty())
    std::cout << "  reason: " << r.reason << "\n";
  std::cout << "  startup_ms: " << FormatDouble(r.startup_ms) << "\n";
  std::cout << "  first_run_ms: " << FormatDouble(r.first_run_ms) << "\n";
  std::cout << "  steady_mean_ms: " << FormatDouble(r.steady.mean_ms) << "\n";
  std::cout << "  steady_p95_ms: " << FormatDouble(r.steady.p95_ms) << "\n";
  std::cout << "  steady_p99_ms: " << FormatDouble(r.steady.p99_ms) << "\n";
  std::cout << "  input_device_resident: " << YesNo(r.input_device_resident)
            << "\n";
  std::cout << "  output_device_resident: " << YesNo(r.output_device_resident)
            << "\n";
  std::cout << "  implicit_uploads: " << r.implicit_uploads << "\n";
  std::cout << "  alpha_downloads: " << r.alpha_downloads << "\n";
  std::cout << "  forced_syncs: " << r.forced_syncs << "\n";
  if (!r.warnings.empty()) {
    std::cout << "  warnings:\n";
    for (const auto &w : r.warnings)
      std::cout << "    - " << w << "\n";
  }
}

void PrintCsv(const Options &opts, const ResolvedModel &model,
              const RunResult &candidate, const RunResult &reference,
              const QualityMetrics &q) {
  std::cout << "backend,reference,model_id,fixture,width,height,warmup,"
               "iterations,startup_ms,first_run_ms,mean_ms,p50_ms,p95_ms,"
               "p99_ms,max_ms,alpha_mae,alpha_max_abs,mask_iou,"
               "input_device_resident,output_device_resident,implicit_uploads,"
               "alpha_downloads,forced_syncs,status,reference_status,reason\n";
  std::cout << "ncnn-vulkan,cpu-ort," << model.pack.id << "," << opts.fixture
            << "," << opts.width << "," << opts.height << "," << opts.warmup
            << "," << opts.iterations << ","
            << FormatDouble(candidate.startup_ms) << ","
            << FormatDouble(candidate.first_run_ms) << ","
            << FormatDouble(candidate.steady.mean_ms) << ","
            << FormatDouble(candidate.steady.p50_ms) << ","
            << FormatDouble(candidate.steady.p95_ms) << ","
            << FormatDouble(candidate.steady.p99_ms) << ","
            << FormatDouble(candidate.steady.max_ms) << ","
            << FormatDouble(q.mae, 6) << "," << FormatDouble(q.max_abs, 6)
            << "," << FormatDouble(q.mask_iou, 6) << ","
            << YesNo(candidate.input_device_resident) << ","
            << YesNo(candidate.output_device_resident) << ","
            << candidate.implicit_uploads << "," << candidate.alpha_downloads
            << "," << candidate.forced_syncs << "," << candidate.status << ","
            << reference.status << ",\"" << candidate.reason << "\"\n";
}

} // namespace

int main(int argc, char **argv) {
  Options opts;
  std::string error;
  if (!ParseArgs(argc, argv, &opts, &error)) {
    std::cerr << "ERROR: " << error << "\n\n";
    Usage(argv[0]);
    return 2;
  }

  ResolvedModel model;
  if (!ResolveModel(opts, &model, &error)) {
    std::cerr << "ERROR: " << error << "\n";
    return 2;
  }

  FrameRgb frame;
  if (!LoadFrame(opts, &frame, &error)) {
    std::cerr << "ERROR: " << error << "\n";
    return 2;
  }
  opts.width = frame.width;
  opts.height = frame.height;

  const auto input = PreprocessCpuNchw(frame, model);
  const RunResult reference = RunCpuOrtReference(model, input, opts);
  const RunResult candidate = RunNcnnCandidate(model, input, opts);
  const QualityMetrics quality =
      (reference.ok && candidate.ok)
          ? CompareAlpha(reference.alpha, candidate.alpha, opts.threshold)
          : QualityMetrics{};

  if (!opts.csv) {
    std::cout << "StudioCast Vulkan Matting Inference Spike\n\n";
    PrintBuildDiagnostics();
    std::cout << "\nModel:\n";
    std::cout << "  id: " << model.pack.id << "\n";
    std::cout << "  manifest: " << model.pack.manifest_path.string() << "\n";
    std::cout << "  onnx: " << model.onnx_path.string() << "\n";
    std::cout << "  ncnn_param: "
              << (model.param_path.empty() ? "(missing)"
                                           : model.param_path.string())
              << "\n";
    std::cout << "  ncnn_bin: "
              << (model.bin_path.empty() ? "(missing)"
                                         : model.bin_path.string())
              << "\n";
    std::cout << "  model_shape: 1x3x" << model.height << "x" << model.width
              << " -> 1x1x" << model.height << "x" << model.width << "\n";
    std::cout << "  ncnn_input_blob: " << model.input_name << "\n";
    std::cout << "  ncnn_output_blob: " << model.output_name << "\n";
    std::cout << "\nFixture:\n";
    std::cout << "  source: " << opts.fixture << "\n";
    std::cout << "  frame_shape: " << frame.width << "x" << frame.height
              << " rgb24\n";
    std::cout << "\nCompatibility notes:\n";
    for (const auto &note : CompatibilityNotes(model))
      std::cout << "  - " << note << "\n";
    std::cout << "  - device_residency: direct StudioCast VulkanTensor/VulkanImage "
                 "to ncnn VkMat interop is not implemented; current spike path "
                 "uses ncnn-owned Vulkan after CPU Mat input\n";
    PrintConversionPlan(model);
    PrintRun("Reference CPU/ORT", reference);
    PrintRun("Candidate ncnn Vulkan", candidate);
    std::cout << "\nQuality:\n";
    if (quality.available) {
      std::cout << "  alpha_mae: " << FormatDouble(quality.mae, 6) << "\n";
      std::cout << "  alpha_max_abs: " << FormatDouble(quality.max_abs, 6)
                << "\n";
      std::cout << "  mask_iou@" << opts.threshold << ": "
                << FormatDouble(quality.mask_iou, 6) << "\n";
    } else {
      std::cout << "  status: skipped\n";
      std::cout << "  reason: needs both reference and ncnn candidate outputs\n";
    }

    std::cout << "\nRecommendation signal:\n";
    if (candidate.ok && quality.available) {
      std::cout << "  proceed_conditionally: ncnn Vulkan ran; inspect quality "
                   "metrics and resolve device-resident interop before "
                   "production integration\n";
    } else if (candidate.status == "failed") {
      std::cout << "  reject_or_retry: ncnn candidate failed; inspect reason "
                   "and conversion/runtime compatibility\n";
    } else {
      std::cout << "  incomplete: install ncnn with Vulkan and converted "
                   "model.ncnn.param/model.ncnn.bin to measure candidate\n";
    }
  } else {
    PrintCsv(opts, model, candidate, reference, quality);
  }

  if (opts.require_reference && !reference.ok) {
    std::cerr << "ERROR: required reference did not run: " << reference.reason
              << "\n";
    return 2;
  }
  if (opts.require_candidate && !candidate.ok) {
    std::cerr << "ERROR: required ncnn candidate did not run: "
              << candidate.reason << "\n";
    return 2;
  }
  return 0;
}
