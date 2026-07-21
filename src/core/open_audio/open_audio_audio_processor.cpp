#include "core/open_audio/open_audio_audio_processor.h"

#include <algorithm>
#include <cmath>
#include <system_error>

#include "core/open_audio/model_pack_registry.h"
#include "core/util/fs.h"
#include "core/util/json.h"
#include "core/util/xdg.h"

namespace fs = std::filesystem;

namespace studiocast::open_audio {

// Forward-declared in the header; used for 48kHz <-> 16kHz conversion.
// These are intentionally lightweight, dependency-free resamplers tuned for
// 3:1.
struct OpenAudioAudioProcessor::Decimator3 {
  explicit Decimator3(std::vector<float> taps) : h(std::move(taps)) {
    if (h.size() >= 2)
      hist.assign(h.size() - 1, 0.0f);
  }

  void Reset() { std::fill(hist.begin(), hist.end(), 0.0f); }

  bool Process(const float *in, std::size_t in_len, float *out,
               std::size_t out_len) {
    if (!in || !out)
      return false;
    if (h.empty())
      return false;
    if ((in_len % 3) != 0)
      return false;
    const std::size_t expected_out = in_len / 3;
    if (out_len < expected_out)
      return false;

    // Extended input: history + current block.
    tmp.clear();
    tmp.reserve(hist.size() + in_len);
    tmp.insert(tmp.end(), hist.begin(), hist.end());
    tmp.insert(tmp.end(), in, in + in_len);

    // Standard FIR (causal): y[n] = sum_{k=0}^{L-1} h[k] * x[n-k]
    // Then decimate by 3: take y at n = hist_size + i*3.
    const std::size_t hist_size = hist.size();
    const std::size_t L = h.size();
    for (std::size_t i = 0; i < expected_out; ++i) {
      const std::size_t n = hist_size + i * 3;
      double acc = 0.0;
      for (std::size_t k = 0; k < L; ++k) {
        acc += static_cast<double>(h[k]) * static_cast<double>(tmp[n - k]);
      }
      out[i] = static_cast<float>(acc);
    }

    // Update history: last L-1 samples.
    if (hist_size > 0) {
      std::copy(tmp.end() - static_cast<std::ptrdiff_t>(hist_size), tmp.end(),
                hist.begin());
    }
    return true;
  }

  std::vector<float> h;
  std::vector<float> hist;
  std::vector<float> tmp;
};

struct OpenAudioAudioProcessor::Interpolator3 {
  explicit Interpolator3(std::vector<float> taps) {
    // We expect an odd number of taps divisible by 3 for clean polyphase split.
    if (taps.empty())
      return;
    if ((taps.size() % 3) != 0)
      return;

    phase_len = taps.size() / 3;
    for (int p = 0; p < 3; ++p) {
      h_phase[p].reserve(phase_len);
      for (std::size_t k = 0; k < phase_len; ++k) {
        h_phase[p].push_back(taps[k * 3 + static_cast<std::size_t>(p)]);
      }
    }
    if (phase_len >= 2)
      hist.assign(phase_len - 1, 0.0f);
  }

  void Reset() { std::fill(hist.begin(), hist.end(), 0.0f); }

  bool Process(const float *in, std::size_t in_len, float *out,
               std::size_t out_len) {
    if (!in || !out)
      return false;
    if (phase_len == 0)
      return false;
    if (out_len < in_len * 3)
      return false;

    // Polyphase interpolation:
    // y[3n + p] = 3 * sum_{k=0}^{P-1} h_p[k] * x[n-k]
    // where x is the 16k stream, y is 48k, and we scale by 3 to preserve DC
    // gain.
    const std::size_t P = phase_len;
    for (std::size_t n = 0; n < in_len; ++n) {
      for (int p = 0; p < 3; ++p) {
        double acc = 0.0;
        for (std::size_t k = 0; k < P; ++k) {
          const std::ptrdiff_t idx =
              static_cast<std::ptrdiff_t>(n) - static_cast<std::ptrdiff_t>(k);
          float x = 0.0f;
          if (idx >= 0) {
            x = in[static_cast<std::size_t>(idx)];
          } else {
            const std::ptrdiff_t hidx =
                static_cast<std::ptrdiff_t>(hist.size()) + idx;
            if (hidx >= 0 && static_cast<std::size_t>(hidx) < hist.size()) {
              x = hist[static_cast<std::size_t>(hidx)];
            }
          }
          acc += static_cast<double>(h_phase[p][k]) * static_cast<double>(x);
        }
        out[n * 3 + static_cast<std::size_t>(p)] =
            static_cast<float>(3.0 * acc);
      }
    }

    // Update history: last P-1 input samples.
    if (hist.size() > 0) {
      if (in_len >= hist.size()) {
        std::copy(in + (in_len - hist.size()), in + in_len, hist.begin());
      } else {
        const std::size_t keep = hist.size() - in_len;
        std::move(hist.begin() + static_cast<std::ptrdiff_t>(in_len),
                  hist.end(), hist.begin());
        std::copy(in, in + in_len,
                  hist.begin() + static_cast<std::ptrdiff_t>(keep));
      }
    }
    return true;
  }

  std::vector<float> h_phase[3];
  std::vector<float> hist;
  std::size_t phase_len = 0;
};

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

float Clamp01(float x) {
  if (x < 0.0f)
    return 0.0f;
  if (x > 1.0f)
    return 1.0f;
  return x;
}

enum class StrengthMode : int {
  kNoiseRemoval = 0,
  kRoomEchoRemoval = 1,
  kStudioVoice = 2,
  kSpeakerNoiseRemoval = 3,
  kSpeakerRoomEchoRemoval = 4,
};

float StrengthCurve01(StrengthMode mode, float t01) {
  const float t = Clamp01(t01);
  switch (mode) {
  case StrengthMode::kRoomEchoRemoval:
    // More aggressive early ramp: helps make dereverb/echo removal noticeable
    // at mid strengths.
    return std::pow(t, 0.75f);
  case StrengthMode::kStudioVoice:
    // Conservative ramp: avoid over-coloring at low/mid strengths.
    return std::pow(t, 1.6f);
  case StrengthMode::kSpeakerNoiseRemoval:
    // Similar to noise removal.
    return std::pow(t, 1.25f);
  case StrengthMode::kSpeakerRoomEchoRemoval:
    // Match room echo removal ramp for speaker-side dereverb.
    return std::pow(t, 0.75f);
  case StrengthMode::kNoiseRemoval:
  default:
    return std::pow(t, 1.25f);
  }
}

float WetMixFromCurve01(StrengthMode mode, float curve01) {
  // Wet/dry mixing is only used when the model does not expose a native
  // "strength" input. These min values are tuned to keep artifacts low while
  // still allowing the user to dial down.
  const float c = Clamp01(curve01);
  float min_wet = 0.2f;
  switch (mode) {
  case StrengthMode::kStudioVoice:
    min_wet = 0.25f;
    break;
  case StrengthMode::kRoomEchoRemoval:
    min_wet = 0.22f;
    break;
  case StrengthMode::kSpeakerNoiseRemoval:
    min_wet = 0.18f;
    break;
  case StrengthMode::kSpeakerRoomEchoRemoval:
    min_wet = 0.2f;
    break;
  case StrengthMode::kNoiseRemoval:
  default:
    min_wet = 0.2f;
    break;
  }
  return Clamp01(min_wet + (1.0f - min_wet) * c);
}

std::vector<float> MakeLowpassFir(int taps, double cutoff_norm) {
  // cutoff_norm is normalized to sampling rate (0..0.5).
  if (taps <= 0)
    return {};
  if ((taps % 2) == 0)
    taps += 1; // force odd length
  if (cutoff_norm <= 0.0 || cutoff_norm >= 0.5)
    return {};

  const int M = taps - 1;
  const int mid = M / 2;
  std::vector<double> h(static_cast<std::size_t>(taps), 0.0);
  for (int n = 0; n < taps; ++n) {
    const double x = static_cast<double>(n - mid);
    const double sinc =
        (x == 0.0) ? 1.0 : (std::sin(2.0 * kPi * cutoff_norm * x) / (kPi * x));
    const double ideal = 2.0 * cutoff_norm * sinc;

    // Hamming window.
    const double w =
        0.54 - 0.46 * std::cos((2.0 * kPi * static_cast<double>(n)) /
                               static_cast<double>(M));
    h[static_cast<std::size_t>(n)] = ideal * w;
  }

  // Normalize DC gain to 1.
  double sum = 0.0;
  for (double v : h)
    sum += v;
  if (sum == 0.0)
    return {};
  for (double &v : h)
    v /= sum;

  std::vector<float> out;
  out.reserve(h.size());
  for (double v : h)
    out.push_back(static_cast<float>(v));
  return out;
}

fs::path ExpandTilde(fs::path p) {
  const std::string s = p.string();
  if (s == "~") {
    return studiocast::util::HomeDir();
  }
  if (s.rfind("~/", 0) == 0) {
    return studiocast::util::HomeDir() / s.substr(2);
  }
  return p;
}

bool IsSafeRelativePath(const fs::path &p) {
  if (p.empty())
    return false;
  if (p.is_absolute())
    return false;
  for (const auto &part : p) {
    if (part == "." || part == "..")
      return false;
  }
  return true;
}

bool Fail(std::string *error, std::string msg) {
  if (error)
    *error = std::move(msg);
  return false;
}

const studiocast::util::json::Value *
Get(const studiocast::util::json::Value::Object &o, const char *key) {
  auto it = o.find(key);
  if (it == o.end())
    return nullptr;
  return &it->second;
}

bool GetStringRequired(const studiocast::util::json::Value::Object &o,
                       const char *key, std::string *out, std::string *error) {
  const auto *v = Get(o, key);
  if (!v)
    return Fail(error, std::string("model.json: missing required field '") +
                           key + "'");
  const auto *s = v->AsString();
  if (!s)
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be a string");
  if (s->empty())
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be non-empty");
  *out = *s;
  return true;
}

bool GetStringOptional(const studiocast::util::json::Value::Object &o,
                       const char *key, std::string *out, std::string *error) {
  const auto *v = Get(o, key);
  if (!v)
    return true;
  const auto *s = v->AsString();
  if (!s)
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be a string");
  *out = *s;
  return true;
}

bool GetIntOptional(const studiocast::util::json::Value::Object &o,
                    const char *key, int *out, std::string *error) {
  const auto *v = Get(o, key);
  if (!v)
    return true;
  const auto *n = v->AsNumber();
  if (!n)
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be a number");
  const int iv = static_cast<int>(*n);
  if (static_cast<double>(iv) != *n) {
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be an integer");
  }
  *out = iv;
  return true;
}

bool GetStringArrayOptional(const studiocast::util::json::Value::Object &o,
                            const char *key, std::vector<std::string> *out,
                            std::string *error) {
  const auto *v = Get(o, key);
  if (!v)
    return true;
  const auto *a = v->AsArray();
  if (!a)
    return Fail(error, std::string("model.json: field '") + key +
                           "' must be an array");
  out->clear();
  out->reserve(a->size());
  for (const auto &el : *a) {
    const auto *s2 = el.AsString();
    if (!s2)
      return Fail(error, std::string("model.json: field '") + key +
                             "' array must contain strings");
    out->push_back(*s2);
  }
  return true;
}

bool ParseAuxStrengthInput(const studiocast::util::json::Value::Object &ioObj,
                           ResolvedOpenAudioModel::OnnxIo *out,
                           std::string *error) {
  const auto *auxVal = Get(ioObj, "aux_inputs");
  if (!auxVal)
    return true;

  const auto *auxObj = auxVal->AsObject();
  if (!auxObj)
    return Fail(error,
                "model.json: field 'onnx_io.aux_inputs' must be an object");

  auto it = auxObj->find("strength");
  if (it == auxObj->end())
    return true;

  const auto &v = it->second;
  out->has_strength_input = true;

  if (const auto *sname = v.AsString()) {
    if (sname->empty())
      return Fail(error,
                  "model.json: onnx_io.aux_inputs.strength must be non-empty");
    out->strength_input.name = *sname;
  } else {
    const auto *sobj = v.AsObject();
    if (!sobj)
      return Fail(error, "model.json: onnx_io.aux_inputs.strength must be a "
                         "string or an object");

    const auto *nameVal = Get(*sobj, "name");
    if (!nameVal)
      return Fail(error, "model.json: onnx_io.aux_inputs.strength is missing "
                         "required field 'name'");
    const auto *nameStr = nameVal->AsString();
    if (!nameStr || nameStr->empty()) {
      return Fail(
          error,
          "model.json: onnx_io.aux_inputs.strength.name must be non-empty");
    }
    out->strength_input.name = *nameStr;

    const auto *rangeVal = Get(*sobj, "range");
    if (rangeVal) {
      const auto *a = rangeVal->AsArray();
      if (!a || a->size() != 2) {
        return Fail(error, "model.json: onnx_io.aux_inputs.strength.range must "
                           "be an array of 2 numbers");
      }
      const auto *lo = (*a)[0].AsNumber();
      const auto *hi = (*a)[1].AsNumber();
      if (!lo || !hi)
        return Fail(error, "model.json: onnx_io.aux_inputs.strength.range must "
                           "contain only numbers");
      out->strength_input.min_value = static_cast<float>(*lo);
      out->strength_input.max_value = static_cast<float>(*hi);
    }

    const auto *shapeVal = Get(*sobj, "shape");
    if (shapeVal) {
      const auto *a = shapeVal->AsArray();
      if (!a || a->empty()) {
        return Fail(error, "model.json: onnx_io.aux_inputs.strength.shape must "
                           "be a non-empty array of integers");
      }
      out->strength_input.shape.clear();
      out->strength_input.shape.reserve(a->size());
      for (const auto &el : *a) {
        const auto *n = el.AsNumber();
        if (!n)
          return Fail(error, "model.json: onnx_io.aux_inputs.strength.shape "
                             "must contain only integers");
        const int64_t d = static_cast<int64_t>(*n);
        if (static_cast<double>(d) != *n || d <= 0) {
          return Fail(error, "model.json: onnx_io.aux_inputs.strength.shape "
                             "must contain only positive integers");
        }
        out->strength_input.shape.push_back(d);
      }
    }
  }

  if (out->strength_input.name.empty()) {
    return Fail(
        error,
        "model.json: onnx_io.aux_inputs.strength.name must be non-empty");
  }
  if (out->strength_input.shape.empty())
    out->strength_input.shape.push_back(1);
  if (out->strength_input.max_value < out->strength_input.min_value) {
    return Fail(
        error,
        "model.json: onnx_io.aux_inputs.strength.range is invalid (max < min)");
  }

  int64_t prod = 1;
  for (const auto d : out->strength_input.shape) {
    if (d <= 0)
      return Fail(
          error,
          "model.json: onnx_io.aux_inputs.strength.shape must be positive");
    if (prod > 1)
      break;
    prod *= d;
  }
  if (prod != 1) {
    return Fail(error, "model.json: onnx_io.aux_inputs.strength.shape must "
                       "have exactly 1 element (scalar)");
  }

  return true;
}

bool ParseOnnxIo(const studiocast::util::json::Value::Object &obj,
                 ResolvedOpenAudioModel *out, std::string *error) {
  if (!out)
    return true;

  const auto *ioVal = Get(obj, "onnx_io");
  if (!ioVal)
    return true;

  const auto *ioObj = ioVal->AsObject();
  if (!ioObj)
    return Fail(error, "model.json: field 'onnx_io' must be an object");

  out->has_onnx_io = true;
  out->onnx_io = {};

  if (!GetIntOptional(*ioObj, "frame_samples", &out->onnx_io.frame_samples,
                      error))
    return false;
  if (!GetStringOptional(*ioObj, "audio_input", &out->onnx_io.audio_input,
                         error))
    return false;
  if (!GetStringOptional(*ioObj, "audio_output", &out->onnx_io.audio_output,
                         error))
    return false;
  if (!GetStringArrayOptional(*ioObj, "state_inputs",
                              &out->onnx_io.state_inputs, error))
    return false;
  if (!GetStringArrayOptional(*ioObj, "state_outputs",
                              &out->onnx_io.state_outputs, error))
    return false;
  if (!ParseAuxStrengthInput(*ioObj, &out->onnx_io, error))
    return false;

  if (out->onnx_io.frame_samples < 0) {
    return Fail(error, "model.json: onnx_io.frame_samples must be >= 0");
  }
  return true;
}

std::size_t FindNameIndex(const std::vector<std::string> &names,
                          const std::string &name) {
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name)
      return i;
  }
  return static_cast<std::size_t>(-1);
}

bool DeriveAudioShape(const std::vector<int64_t> &ort_shape,
                      std::size_t frame_samples,
                      std::vector<int64_t> *out_shape, std::string *error,
                      const char *what) {
  if (!out_shape)
    return false;
  out_shape->clear();

  if (ort_shape.empty()) {
    // Default to [1, N] if shape is missing.
    out_shape->push_back(1);
    out_shape->push_back(static_cast<int64_t>(frame_samples));
    return true;
  }

  if (ort_shape.size() == 1) {
    const int64_t d0 = ort_shape[0];
    if (d0 > 0 && static_cast<std::size_t>(d0) != frame_samples) {
      return Fail(error, std::string("Open Audio: ") + what +
                             " shape mismatch (expected " +
                             std::to_string(frame_samples) + ", got " +
                             std::to_string(d0) + ")");
    }
    out_shape->push_back(static_cast<int64_t>(frame_samples));
    return true;
  }

  if (ort_shape.size() == 2) {
    int64_t b = ort_shape[0];
    int64_t n = ort_shape[1];
    if (b == -1)
      b = 1;
    if (b != 1) {
      return Fail(error, std::string("Open Audio: ") + what +
                             " batch dimension must be 1 (or dynamic)");
    }
    if (n == -1)
      n = static_cast<int64_t>(frame_samples);
    if (n != static_cast<int64_t>(frame_samples)) {
      return Fail(error, std::string("Open Audio: ") + what +
                             " time dimension mismatch (expected " +
                             std::to_string(frame_samples) + ", got " +
                             std::to_string(n) + ")");
    }
    out_shape->push_back(b);
    out_shape->push_back(n);
    return true;
  }

  return Fail(error, std::string("Open Audio: unsupported ") + what +
                         " rank (expected 1 or 2)");
}

bool DeriveScalarShape(const std::vector<int64_t> &ort_shape,
                       const std::vector<int64_t> &hint_shape,
                       std::vector<int64_t> *out_shape, std::string *error) {
  if (!out_shape)
    return false;
  out_shape->clear();

  if (!hint_shape.empty()) {
    *out_shape = hint_shape;
  } else if (!ort_shape.empty()) {
    *out_shape = ort_shape;
  } else {
    out_shape->push_back(1);
  }

  // Replace dynamic dims with 1 for scalar control inputs.
  for (auto &d : *out_shape) {
    if (d == -1)
      d = 1;
  }

  int64_t prod = 1;
  for (const auto d : *out_shape) {
    if (d <= 0)
      return Fail(error, "Open Audio: strength input shape must be positive");
    if (prod > 1)
      break;
    prod *= d;
  }
  if (prod != 1) {
    return Fail(
        error,
        "Open Audio: strength input must be a scalar (product(shape)==1)");
  }
  return true;
}

bool ResolveFromPackDir(const fs::path &pack_dir, ResolvedOpenAudioModel *out,
                        std::string *error) {
  const fs::path manifestPath = pack_dir / "model.json";
  const auto textOpt = studiocast::util::ReadTextFile(manifestPath.string());
  if (!textOpt) {
    return Fail(error,
                std::string("missing model.json at ") + manifestPath.string());
  }

  studiocast::util::json::Value root;
  std::string parseErr;
  if (!studiocast::util::json::Parse(*textOpt, &root, &parseErr)) {
    return Fail(error, std::string("model.json: ") + parseErr);
  }
  const auto *obj = root.AsObject();
  if (!obj)
    return Fail(error, "model.json: root must be an object");

  std::string id;
  std::string display;
  std::string onnxFilename;
  if (!GetStringRequired(*obj, "id", &id, error))
    return false;
  if (!GetStringRequired(*obj, "display_name", &display, error))
    return false;
  if (!GetStringRequired(*obj, "onnx_filename", &onnxFilename, error))
    return false;

  const fs::path rel(onnxFilename);
  if (!IsSafeRelativePath(rel)) {
    return Fail(
        error,
        "model.json: field 'onnx_filename' must be a safe relative path");
  }
  const fs::path onnxPath = pack_dir / rel;

  std::error_code ec;
  if (!fs::exists(onnxPath, ec) || ec) {
    ec.clear();
    return Fail(error, std::string("missing ONNX file: ") + onnxPath.string());
  }
  if (!fs::is_regular_file(onnxPath, ec) || ec) {
    ec.clear();
    return Fail(error, std::string("ONNX path is not a regular file: ") +
                           onnxPath.string());
  }

  // Optional metadata.
  int sample_rate = 16000;
  int channels = 1;
  std::string meta_err;
  if (!GetIntOptional(*obj, "sample_rate", &sample_rate, &meta_err)) {
    return Fail(error, meta_err);
  }
  if (!GetIntOptional(*obj, "channels", &channels, &meta_err)) {
    return Fail(error, meta_err);
  }
  if (sample_rate < 0)
    return Fail(error, "model.json: sample_rate must be >= 0");
  if (channels <= 0)
    return Fail(error, "model.json: channels must be >= 1");

  if (out) {
    out->model_id = id;
    out->display_name = display;
    out->onnx_path = onnxPath;
    out->sample_rate = sample_rate;
    out->channels = channels;
    out->is_user_path = true;
  }

  // Optional onnx_io section.
  return ParseOnnxIo(*obj, out, error);
}

bool ResolveFromOnnxFile(const fs::path &onnxPath, ResolvedOpenAudioModel *out,
                         std::string *error) {
  std::error_code ec;
  if (!fs::exists(onnxPath, ec) || ec) {
    ec.clear();
    return Fail(error,
                std::string("model_path does not exist: ") + onnxPath.string());
  }
  if (!fs::is_regular_file(onnxPath, ec) || ec) {
    ec.clear();
    return Fail(error,
                std::string("model_path is not a file: ") + onnxPath.string());
  }

  // Extension check is best-effort; allow non-.onnx for advanced users.
  if (out) {
    out->model_id.clear();
    out->display_name = onnxPath.filename().string();
    out->onnx_path = onnxPath;
    out->sample_rate = 0; // unknown; caller may assume pipeline sample rate
    out->channels = 1;
    out->has_onnx_io = false;
    out->onnx_io = {};
    out->is_user_path = true;
  }
  return true;
}

} // namespace

bool ResolveOpenAudioModelForMicrophone(
    const studiocast::audio::effects::BroadcastAudioEffects &fx,
    ResolvedOpenAudioModel *out, std::string *error) {
  if (error)
    error->clear();

  const auto &mic = fx.microphone;

  // 1) Explicit model path wins.
  if (!mic.model_path.empty()) {
    fs::path p = ExpandTilde(fs::path(mic.model_path));
    std::error_code ec;
    if (!fs::exists(p, ec) || ec) {
      ec.clear();
      return Fail(error, std::string("Open Audio model_path does not exist: ") +
                             p.string());
    }
    if (fs::is_directory(p, ec) && !ec) {
      return ResolveFromPackDir(p, out, error);
    }
    ec.clear();
    return ResolveFromOnnxFile(p, out, error);
  }

  // 2/3) Installed pack id or default.
  const auto reg = ModelPackRegistry::ScanDefault();

  std::string id = mic.model_id;
  if (id.empty()) {
    // Choose a default model based on the enabled effect.
    std::string effect;
    if (mic.studio_voice_enabled) {
      effect = "studio_voice";
    } else if (mic.room_echo_removal_enabled) {
      effect = "room_echo_removal";
    } else if (mic.noise_removal_enabled) {
      effect = "noise_removal";
    }

    id = reg.DefaultModelIdForEffect(effect);
    if (id.empty())
      id = reg.DefaultModelId();
  }
  if (id.empty()) {
    return Fail(error,
                "Open Audio: no usable model packs found (install under "
                "~/.local/share/studiocast/models/open_audio/<model_id>/).");
  }

  const auto packOpt = reg.ResolveModel(id);
  if (!packOpt.has_value()) {
    std::string msg =
        std::string("Open Audio: model_id '") + id + "' not found.";
    if (!reg.ListModels().empty()) {
      msg += " Available models: ";
      for (std::size_t i = 0; i < reg.ListModels().size(); ++i) {
        if (i)
          msg += ", ";
        msg += reg.ListModels()[i].id;
      }
      msg += ".";
    }
    return Fail(error, msg);
  }

  const auto &pack = *packOpt;
  std::error_code ec;
  if (!fs::exists(pack.onnx_path, ec) || ec) {
    ec.clear();
    return Fail(error, std::string("Open Audio: missing ONNX file: ") +
                           pack.onnx_path.string());
  }
  if (!fs::is_regular_file(pack.onnx_path, ec) || ec) {
    ec.clear();
    return Fail(error, std::string("Open Audio: ONNX path is not a file: ") +
                           pack.onnx_path.string());
  }

  if (out) {
    out->model_id = pack.id;
    out->display_name = pack.display_name;
    out->onnx_path = pack.onnx_path;
    out->sample_rate = pack.sample_rate;
    out->channels = pack.channels;
    out->has_onnx_io = pack.has_onnx_io;
    if (pack.has_onnx_io) {
      out->onnx_io.frame_samples = pack.onnx_io.frame_samples;
      out->onnx_io.audio_input = pack.onnx_io.audio_input;
      out->onnx_io.audio_output = pack.onnx_io.audio_output;
      out->onnx_io.state_inputs = pack.onnx_io.state_inputs;
      out->onnx_io.state_outputs = pack.onnx_io.state_outputs;
      out->onnx_io.has_strength_input = pack.onnx_io.has_strength_input;
      out->onnx_io.strength_input.name = pack.onnx_io.strength_input.name;
      out->onnx_io.strength_input.min_value =
          pack.onnx_io.strength_input.min_value;
      out->onnx_io.strength_input.max_value =
          pack.onnx_io.strength_input.max_value;
      out->onnx_io.strength_input.shape = pack.onnx_io.strength_input.shape;
    } else {
      out->onnx_io = {};
    }
    out->is_user_path = false;
  }
  return true;
}

bool ResolveOpenAudioModelForSpeaker(
    const studiocast::audio::effects::BroadcastAudioEffects &fx,
    ResolvedOpenAudioModel *out, std::string *error) {
  if (error)
    error->clear();

  const auto &spk = fx.speaker;

  // 1) Explicit model path wins.
  if (!spk.model_path.empty()) {
    fs::path p = ExpandTilde(fs::path(spk.model_path));
    std::error_code ec;
    if (!fs::exists(p, ec) || ec) {
      ec.clear();
      return Fail(
          error, std::string("Open Audio speaker model_path does not exist: ") +
                     p.string());
    }
    if (fs::is_directory(p, ec) && !ec) {
      return ResolveFromPackDir(p, out, error);
    }
    ec.clear();
    return ResolveFromOnnxFile(p, out, error);
  }

  // 2/3) Installed pack id or default.
  const auto reg = ModelPackRegistry::ScanDefault();

  std::string id = spk.model_id;
  if (id.empty()) {
    std::string effect;
    if (spk.room_echo_removal_enabled) {
      effect = "room_echo_removal";
    } else if (spk.noise_removal_enabled) {
      effect = "noise_removal";
    }
    id = reg.DefaultModelIdForEffect(effect);
    if (id.empty())
      id = reg.DefaultModelId();
  }
  if (id.empty()) {
    return Fail(error,
                "Open Audio: no usable model packs found (install under "
                "~/.local/share/studiocast/models/open_audio/<model_id>/).");
  }

  const auto packOpt = reg.ResolveModel(id);
  if (!packOpt.has_value()) {
    std::string msg =
        std::string("Open Audio: model_id '") + id + "' not found.";
    if (!reg.ListModels().empty()) {
      msg += " Available models: ";
      for (std::size_t i = 0; i < reg.ListModels().size(); ++i) {
        if (i)
          msg += ", ";
        msg += reg.ListModels()[i].id;
      }
      msg += ".";
    }
    return Fail(error, msg);
  }

  const auto &pack = *packOpt;
  std::error_code ec;
  if (!fs::exists(pack.onnx_path, ec) || ec) {
    ec.clear();
    return Fail(error, std::string("Open Audio: missing ONNX file: ") +
                           pack.onnx_path.string());
  }
  if (!fs::is_regular_file(pack.onnx_path, ec) || ec) {
    ec.clear();
    return Fail(error, std::string("Open Audio: ONNX path is not a file: ") +
                           pack.onnx_path.string());
  }

  if (out) {
    out->model_id = pack.id;
    out->display_name = pack.display_name;
    out->onnx_path = pack.onnx_path;
    out->sample_rate = pack.sample_rate;
    out->channels = pack.channels;
    out->has_onnx_io = pack.has_onnx_io;
    if (pack.has_onnx_io) {
      out->onnx_io.frame_samples = pack.onnx_io.frame_samples;
      out->onnx_io.audio_input = pack.onnx_io.audio_input;
      out->onnx_io.audio_output = pack.onnx_io.audio_output;
      out->onnx_io.state_inputs = pack.onnx_io.state_inputs;
      out->onnx_io.state_outputs = pack.onnx_io.state_outputs;
      out->onnx_io.has_strength_input = pack.onnx_io.has_strength_input;
      out->onnx_io.strength_input.name = pack.onnx_io.strength_input.name;
      out->onnx_io.strength_input.min_value =
          pack.onnx_io.strength_input.min_value;
      out->onnx_io.strength_input.max_value =
          pack.onnx_io.strength_input.max_value;
      out->onnx_io.strength_input.shape = pack.onnx_io.strength_input.shape;
    } else {
      out->onnx_io = {};
    }
    out->is_user_path = false;
  }
  return true;
}

std::unique_ptr<OpenAudioAudioProcessor>
OpenAudioAudioProcessor::CreateForMicrophone(
    const studiocast::audio::effects::BroadcastAudioEffects &fx,
    ResolvedOpenAudioModel *resolved_out, std::string *error) {
  OrtSessionOptions opts;
  opts.prefer_cuda = true;
  opts.cuda_device_id = 0;
  return CreateForMicrophoneWithOrtOptions(fx, opts, resolved_out, error);
}

std::unique_ptr<OpenAudioAudioProcessor>
OpenAudioAudioProcessor::CreateForSpeaker(
    const studiocast::audio::effects::BroadcastAudioEffects &fx,
    ResolvedOpenAudioModel *resolved_out, std::string *error) {
  OrtSessionOptions opts;
  opts.prefer_cuda = true;
  opts.cuda_device_id = 0;
  return CreateForSpeakerWithOrtOptions(fx, opts, resolved_out, error);
}

std::unique_ptr<OpenAudioAudioProcessor>
OpenAudioAudioProcessor::CreateForSpeakerWithOrtOptions(
    const studiocast::audio::effects::BroadcastAudioEffects &fx,
    const studiocast::open_audio::OrtSessionOptions &ort_opts,
    ResolvedOpenAudioModel *resolved_out, std::string *error) {
#if !STUDIOCAST_ENABLE_OPEN_AUDIO
  (void)fx;
  (void)ort_opts;
  (void)resolved_out;
  if (error)
    *error = "Open Audio backend is disabled in this build "
             "(STUDIOCAST_ENABLE_OPEN_AUDIO=0).";
  return nullptr;
#else
  ResolvedOpenAudioModel resolved;
  std::string err;
  if (!ResolveOpenAudioModelForSpeaker(fx, &resolved, &err)) {
    if (error)
      *error = err;
    return nullptr;
  }
  if (resolved_out)
    *resolved_out = resolved;

#if !STUDIOCAST_HAVE_ONNXRUNTIME
  (void)ort_opts;
  if (error) {
    *error = "Open Audio backend unavailable: ONNX Runtime was not found at "
             "build time (STUDIOCAST_HAVE_ONNXRUNTIME=0).";
  }
  return nullptr;
#else
  OrtSessionInfo si;
  std::string ort_err;
  const auto onnx_path = resolved.onnx_path;
  auto session =
      OpenAudioOrtSession::Create(onnx_path, ort_opts, &si, &ort_err);
  if (!session) {
    if (error) {
      *error =
          ort_err.empty()
              ? "Failed to create ONNX Runtime session for Open Audio model."
              : ort_err;
    }
    return nullptr;
  }

  auto proc = std::make_unique<OpenAudioAudioProcessor>(std::move(resolved));
  proc->UpdateFromSpeakerConfig(fx.speaker);

  if (si.using_cuda) {
    proc->ort_session_cuda_ = std::move(session);
    proc->ort_session_active_ = proc->ort_session_cuda_.get();

    OrtSessionOptions cpu_opts = ort_opts;
    cpu_opts.prefer_cuda = false;
    OrtSessionInfo cpu_si;
    std::string cpu_err;
    auto cpu =
        OpenAudioOrtSession::Create(onnx_path, cpu_opts, &cpu_si, &cpu_err);
    if (cpu) {
      proc->ort_session_cpu_ = std::move(cpu);
    }
  } else {
    proc->ort_session_cpu_ = std::move(session);
    proc->ort_session_active_ = proc->ort_session_cpu_.get();
    proc->using_cpu_fallback_ = true;
  }

  std::string bind_err;
  if (!proc->InitializeBindings(&bind_err)) {
    if (error)
      *error = bind_err;
    return nullptr;
  }

  // Speaker processing runs as stereo in the Pulse pipeline. Configure the post
  // DSP stage for 2 channels (DC blocker + safety limiter).
  {
    std::string derr;
    (void)proc->post_dsp_.Configure(/*sample_rate=*/48000, /*channels=*/2,
                                    &derr);
  }

  return proc;
#endif // STUDIOCAST_HAVE_ONNXRUNTIME
#endif // STUDIOCAST_ENABLE_OPEN_AUDIO
}

std::unique_ptr<OpenAudioAudioProcessor>
OpenAudioAudioProcessor::CreateForMicrophoneWithOrtOptions(
    const studiocast::audio::effects::BroadcastAudioEffects &fx,
    const studiocast::open_audio::OrtSessionOptions &ort_opts,
    ResolvedOpenAudioModel *resolved_out, std::string *error) {
#if !STUDIOCAST_ENABLE_OPEN_AUDIO
  (void)fx;
  (void)ort_opts;
  (void)resolved_out;
  if (error)
    *error = "Open Audio backend is disabled in this build "
             "(STUDIOCAST_ENABLE_OPEN_AUDIO=0).";
  return nullptr;
#else
  ResolvedOpenAudioModel resolved;
  std::string err;
  if (!ResolveOpenAudioModelForMicrophone(fx, &resolved, &err)) {
    if (error)
      *error = err;
    return nullptr;
  }
  if (resolved_out)
    *resolved_out = resolved;

#if !STUDIOCAST_HAVE_ONNXRUNTIME
  (void)ort_opts;
  if (error) {
    *error = "Open Audio backend unavailable: ONNX Runtime was not found at "
             "build time (STUDIOCAST_HAVE_ONNXRUNTIME=0).";
  }
  return nullptr;
#else
  // Create the ORT session up-front so we fail fast (actionable error)
  // and avoid repeatedly attempting to load the model in the realtime thread.
  OrtSessionInfo si;
  std::string ort_err;
  const auto onnx_path = resolved.onnx_path;
  auto session =
      OpenAudioOrtSession::Create(onnx_path, ort_opts, &si, &ort_err);
  if (!session) {
    if (error) {
      *error =
          ort_err.empty()
              ? "Failed to create ONNX Runtime session for Open Audio model."
              : ort_err;
    }
    return nullptr;
  }

  auto proc = std::make_unique<OpenAudioAudioProcessor>(std::move(resolved));
  proc->UpdateFromMicrophoneConfig(fx.microphone);

  if (si.using_cuda) {
    proc->ort_session_cuda_ = std::move(session);
    proc->ort_session_active_ = proc->ort_session_cuda_.get();

    // Best-effort CPU fallback session (used when CUDA session fails at
    // runtime).
    OrtSessionOptions cpu_opts = ort_opts;
    cpu_opts.prefer_cuda = false;
    OrtSessionInfo cpu_si;
    std::string cpu_err;
    auto cpu =
        OpenAudioOrtSession::Create(onnx_path, cpu_opts, &cpu_si, &cpu_err);
    if (cpu) {
      proc->ort_session_cpu_ = std::move(cpu);
    }
  } else {
    proc->ort_session_cpu_ = std::move(session);
    proc->ort_session_active_ = proc->ort_session_cpu_.get();
    proc->using_cpu_fallback_ = true;
  }

  std::string bind_err;
  if (!proc->InitializeBindings(&bind_err)) {
    if (error)
      *error = bind_err;
    return nullptr;
  }

  return proc;
#endif // STUDIOCAST_HAVE_ONNXRUNTIME
#endif // STUDIOCAST_ENABLE_OPEN_AUDIO
}

OpenAudioAudioProcessor::OpenAudioAudioProcessor(ResolvedOpenAudioModel model)
    : model_(std::move(model)) {
  if (model_.sample_rate > 0) {
    model_sample_rate_ = model_.sample_rate;
  }
  if (model_.has_onnx_io && model_.onnx_io.frame_samples > 0) {
    model_frame_samples_ =
        static_cast<std::uint32_t>(model_.onnx_io.frame_samples);
  } else if (model_sample_rate_ > 0) {
    model_frame_samples_ = static_cast<std::uint32_t>(model_sample_rate_ / 100);
  }

  // If model pack didn't specify channels, assume mono.
  if (model_.channels <= 0)
    model_.channels = 1;

  // Configure post-processing DSP stage (48kHz pipeline, mono). This stage is a
  // safety net to prevent clipping and stabilize output levels.
  std::string derr;
  post_dsp_.Configure(/*sample_rate=*/48000, /*channels=*/1, &derr);
  // Presence shelf is enabled dynamically when Studio Voice is active.
  post_dsp_.SetPresenceShelf(
      studiocast::audio::dsp::PostDspChain::PresenceShelfConfig{});
}

OpenAudioAudioProcessor::~OpenAudioAudioProcessor() = default;

std::string OpenAudioAudioProcessor::ActiveProviderForStatus() const {
  if (model_disabled_)
    return "disabled";
  if (!ort_session_active_)
    return {};

  const auto &si = ort_session_active_->info();
  if (!si.active_provider.empty())
    return si.active_provider;
  return using_cpu_fallback_ ? "cpu" : (si.using_cuda ? "cuda" : "cpu");
}

std::string OpenAudioAudioProcessor::LastStartupWarningForStatus() const {
  if (!ort_session_active_)
    return {};
  const auto &warnings = ort_session_active_->info().warnings;
  return warnings.empty() ? std::string{} : warnings.back();
}

OpenAudioOrtSession::PreparedRunStats
OpenAudioAudioProcessor::PreparedRunStatsForTesting() const {
  return ort_session_active_ ? ort_session_active_->prepared_run_stats()
                             : OpenAudioOrtSession::PreparedRunStats{};
}

void OpenAudioAudioProcessor::UpdateFromMicrophoneConfig(
    const studiocast::audio::effects::BroadcastMicrophoneEffects &mic) {
  int s = mic.strength;
  if (s < 0)
    s = 0;
  if (s > 100)
    s = 100;
  strength_.store(s);
  studio_voice_enabled_.store(mic.studio_voice_enabled);

  int mode = static_cast<int>(StrengthMode::kNoiseRemoval);
  if (mic.studio_voice_enabled) {
    mode = static_cast<int>(StrengthMode::kStudioVoice);
  } else if (mic.room_echo_removal_enabled) {
    mode = static_cast<int>(StrengthMode::kRoomEchoRemoval);
  } else {
    mode = static_cast<int>(StrengthMode::kNoiseRemoval);
  }
  strength_mode_.store(mode);
}

void OpenAudioAudioProcessor::UpdateFromSpeakerConfig(
    const studiocast::audio::effects::BroadcastSpeakerEffects &spk) {
  int s = spk.strength;
  if (s < 0)
    s = 0;
  if (s > 100)
    s = 100;
  strength_.store(s);
  studio_voice_enabled_.store(false);
  StrengthMode mode = StrengthMode::kSpeakerNoiseRemoval;
  if (spk.room_echo_removal_enabled) {
    mode = StrengthMode::kSpeakerRoomEchoRemoval;
  }
  strength_mode_.store(static_cast<int>(mode));
}

bool OpenAudioAudioProcessor::InitializeBindings(std::string *error) {
  if (error)
    error->clear();

  if (!ort_session_active_) {
    return Fail(error, "Open Audio: ORT session is not initialized");
  }

  const auto &si = ort_session_active_->info();
  if (si.input_names.empty() || si.output_names.empty()) {
    return Fail(error, "Open Audio: ORT session has no inputs/outputs");
  }

  // Resolve primary waveform I/O names.
  audio_input_name_ =
      (model_.has_onnx_io && !model_.onnx_io.audio_input.empty())
          ? model_.onnx_io.audio_input
          : si.input_names[0];
  audio_output_name_ =
      (model_.has_onnx_io && !model_.onnx_io.audio_output.empty())
          ? model_.onnx_io.audio_output
          : si.output_names[0];

  // If sample rate / frame size metadata is missing (common when using a bare
  // .onnx file), attempt to infer the frame size from the ORT audio input
  // shape.
  const bool need_infer_frame =
      (model_.sample_rate <= 0) &&
      (!model_.has_onnx_io || model_.onnx_io.frame_samples <= 0);
  if (need_infer_frame) {
    const std::size_t idx = FindNameIndex(si.input_names, audio_input_name_);
    if (idx != static_cast<std::size_t>(-1) && idx < si.input_shapes.size()) {
      const auto &shp = si.input_shapes[idx];
      int64_t n = -1;
      if (shp.size() == 1) {
        n = shp[0];
      } else if (shp.size() == 2) {
        n = shp[1];
      }
      if (n > 0) {
        model_frame_samples_ = static_cast<std::uint32_t>(n);
        // Assume 10 ms hop if sample rate is unknown.
        model_sample_rate_ = static_cast<int>(model_frame_samples_) * 100;
      }
    }
  }

  // Strength aux input.
  has_strength_input_ = false;
  strength_input_name_.clear();
  strength_input_shape_.clear();
  strength_input_buf_.clear();
  strength_input_min_ = 0.0f;
  strength_input_max_ = 1.0f;

  if (model_.has_onnx_io && model_.onnx_io.has_strength_input &&
      !model_.onnx_io.strength_input.name.empty()) {
    has_strength_input_ = true;
    strength_input_name_ = model_.onnx_io.strength_input.name;
    strength_input_min_ = model_.onnx_io.strength_input.min_value;
    strength_input_max_ = model_.onnx_io.strength_input.max_value;

    const std::size_t idx = FindNameIndex(si.input_names, strength_input_name_);
    const std::vector<int64_t> ort_shape =
        (idx != static_cast<std::size_t>(-1) && idx < si.input_shapes.size())
            ? si.input_shapes[idx]
            : std::vector<int64_t>{};

    if (!DeriveScalarShape(ort_shape, model_.onnx_io.strength_input.shape,
                           &strength_input_shape_, error))
      return false;
    strength_input_buf_.assign(1, 0.0f);
  }

  // Derive waveform shapes.
  const std::size_t in_idx = FindNameIndex(si.input_names, audio_input_name_);
  const std::vector<int64_t> in_shape =
      (in_idx != static_cast<std::size_t>(-1) &&
       in_idx < si.input_shapes.size())
          ? si.input_shapes[in_idx]
          : std::vector<int64_t>{};
  if (!DeriveAudioShape(in_shape, model_frame_samples_, &audio_input_shape_,
                        error, "audio input"))
    return false;

  const std::size_t out_idx =
      FindNameIndex(si.output_names, audio_output_name_);
  const std::vector<int64_t> out_shape =
      (out_idx != static_cast<std::size_t>(-1) &&
       out_idx < si.output_shapes.size())
          ? si.output_shapes[out_idx]
          : std::vector<int64_t>{};
  if (!DeriveAudioShape(out_shape, model_frame_samples_, &audio_output_shape_,
                        error, "audio output"))
    return false;

  // Determine state I/O names.
  state_input_names_.clear();
  state_output_names_.clear();

  if (model_.has_onnx_io && !model_.onnx_io.state_inputs.empty()) {
    state_input_names_ = model_.onnx_io.state_inputs;
  } else {
    for (const auto &n : si.input_names) {
      if (n == audio_input_name_)
        continue;
      if (has_strength_input_ && n == strength_input_name_)
        continue;
      state_input_names_.push_back(n);
    }
  }

  if (model_.has_onnx_io && !model_.onnx_io.state_outputs.empty()) {
    state_output_names_ = model_.onnx_io.state_outputs;
  } else {
    for (const auto &n : si.output_names) {
      if (n == audio_output_name_)
        continue;
      state_output_names_.push_back(n);
    }
  }

  if (!state_input_names_.empty()) {
    if (state_output_names_.size() != state_input_names_.size()) {
      return Fail(
          error,
          "Open Audio: model exposes state inputs/outputs but they do not "
          "match (pack must declare onnx_io.state_inputs/state_outputs)");
    }
  }

  // Allocate state buffers.
  state_shapes_.clear();
  state_sizes_.clear();
  state_buf_[0].clear();
  state_buf_[1].clear();
  state_toggle_ = 0;

  if (!state_input_names_.empty()) {
    const std::size_t nstates = state_input_names_.size();
    state_shapes_.resize(nstates);
    state_sizes_.resize(nstates);

    state_buf_[0].resize(nstates);
    state_buf_[1].resize(nstates);

    for (std::size_t i = 0; i < nstates; ++i) {
      const std::string &in_name = state_input_names_[i];
      const std::string &out_name = state_output_names_[i];

      const std::size_t idx_in = FindNameIndex(si.input_names, in_name);
      const std::size_t idx_out = FindNameIndex(si.output_names, out_name);
      if (idx_in == static_cast<std::size_t>(-1) ||
          idx_out == static_cast<std::size_t>(-1)) {
        return Fail(
            error,
            "Open Audio: state tensor name not found in ORT session I/O");
      }

      const std::vector<int64_t> shape_in = (idx_in < si.input_shapes.size())
                                                ? si.input_shapes[idx_in]
                                                : std::vector<int64_t>{};
      const std::vector<int64_t> shape_out = (idx_out < si.output_shapes.size())
                                                 ? si.output_shapes[idx_out]
                                                 : std::vector<int64_t>{};

      if (shape_in.empty() || shape_out.empty() ||
          shape_in.size() != shape_out.size()) {
        return Fail(error, "Open Audio: unable to derive state tensor shapes "
                           "(pack may need explicit shapes)");
      }

      std::vector<int64_t> shape = shape_in;
      for (std::size_t d = 0; d < shape.size(); ++d) {
        int64_t a = shape_in[d];
        int64_t b = shape_out[d];
        if (a == -1 && b > 0)
          a = b;
        if (b == -1 && a > 0)
          b = a;
        if (a == -1 && b == -1)
          a = 1, b = 1;
        if (a != b) {
          return Fail(error, "Open Audio: state tensor shape mismatch between "
                             "input and output");
        }
        shape[d] = a;
      }

      int64_t prod = 1;
      for (const auto dim : shape) {
        if (dim <= 0)
          return Fail(error, "Open Audio: state tensor shape must be positive");
        prod *= dim;
      }
      state_shapes_[i] = std::move(shape);
      state_sizes_[i] = static_cast<std::size_t>(prod);

      state_buf_[0][i].assign(state_sizes_[i], 0.0f);
      state_buf_[1][i].assign(state_sizes_[i], 0.0f);
    }
  }

  // Allocate model I/O buffers (mono waveform).
  model_in_.assign(model_frame_samples_, 0.0f);
  model_out_.assign(model_frame_samples_, 0.0f);
  // Allocate pipeline-rate scratch buffers up front. The realtime Process()
  // path should not grow vectors; AudioPipeline enforces 480-frame blocks.
  constexpr std::uint32_t kPipelineFrameSamples = 480;
  mono_in_.assign(kPipelineFrameSamples, 0.0f);
  mono_out_.assign(kPipelineFrameSamples, 0.0f);
  side_.assign(kPipelineFrameSamples, 0.0f);

  // Configure resampler for 48k <-> 16k.
  decim3_.reset();
  interp3_.reset();
  if (model_sample_rate_ == 16000) {
    // 48k -> 16k requires cutoff at 8k: cutoff_norm = 8000/48000 = 1/6.
    auto taps = MakeLowpassFir(/*taps=*/33, /*cutoff_norm=*/1.0 / 6.0);
    if (taps.empty() || (taps.size() % 3) != 0) {
      return Fail(error,
                  "Open Audio: internal resampler initialization failed");
    }
    decim3_ = std::make_unique<Decimator3>(taps);
    interp3_ = std::make_unique<Interpolator3>(taps);
  } else if (model_sample_rate_ != 48000) {
    return Fail(error, "Open Audio: unsupported model sample_rate (only 16000 "
                       "and 48000 are supported)");
  }

  // Build ORT input/output bindings (names + shapes; data pointers are
  // refreshed in Process()).
  ort_inputs_.clear();
  ort_outputs_.clear();
  ort_inputs_.reserve(1 + (has_strength_input_ ? 1 : 0) +
                      state_input_names_.size());
  ort_outputs_.reserve(1 + state_output_names_.size());

  OpenAudioOrtSession::OrtRunInput in0;
  in0.name = audio_input_name_.c_str();
  in0.data = model_in_.data();
  in0.num_floats = model_in_.size();
  in0.shape = audio_input_shape_.data();
  in0.shape_rank = audio_input_shape_.size();
  ort_inputs_.push_back(in0);

  if (has_strength_input_) {
    OpenAudioOrtSession::OrtRunInput si_in;
    si_in.name = strength_input_name_.c_str();
    si_in.data = strength_input_buf_.data();
    si_in.num_floats = strength_input_buf_.size();
    si_in.shape = strength_input_shape_.data();
    si_in.shape_rank = strength_input_shape_.size();
    ort_inputs_.push_back(si_in);
  }

  for (std::size_t i = 0; i < state_input_names_.size(); ++i) {
    OpenAudioOrtSession::OrtRunInput st_in;
    st_in.name = state_input_names_[i].c_str();
    st_in.data = state_buf_[state_toggle_][i].data();
    st_in.num_floats = state_sizes_[i];
    st_in.shape = state_shapes_[i].data();
    st_in.shape_rank = state_shapes_[i].size();
    ort_inputs_.push_back(st_in);
  }

  OpenAudioOrtSession::OrtRunOutput out0;
  out0.name = audio_output_name_.c_str();
  out0.data = model_out_.data();
  out0.num_floats = model_out_.size();
  out0.shape = audio_output_shape_.data();
  out0.shape_rank = audio_output_shape_.size();
  ort_outputs_.push_back(out0);

  for (std::size_t i = 0; i < state_output_names_.size(); ++i) {
    OpenAudioOrtSession::OrtRunOutput st_out;
    st_out.name = state_output_names_[i].c_str();
    st_out.data = state_buf_[1 - state_toggle_][i].data();
    st_out.num_floats = state_sizes_[i];
    st_out.shape = state_shapes_[i].data();
    st_out.shape_rank = state_shapes_[i].size();
    ort_outputs_.push_back(st_out);
  }

  const auto reserve_run_scratch = [&](OpenAudioOrtSession *session) {
    if (session) {
      session->ReserveRunScratch(ort_inputs_.size(), ort_outputs_.size());
    }
  };
  reserve_run_scratch(ort_session_cuda_.get());
  reserve_run_scratch(ort_session_cpu_.get());

  return true;
}

void OpenAudioAudioProcessor::Reset() {
  sticky_warning_.clear();
  model_disabled_ = false;
  using_cpu_fallback_ = false;

  post_dsp_.Reset();

  // Reset ORT session selection.
  if (ort_session_cuda_) {
    ort_session_active_ = ort_session_cuda_.get();
  } else {
    ort_session_active_ = ort_session_cpu_.get();
    using_cpu_fallback_ = true;
  }

  // Reset streaming state buffers.
  state_toggle_ = 0;
  for (auto &bufset : state_buf_) {
    for (auto &buf : bufset) {
      std::fill(buf.begin(), buf.end(), 0.0f);
    }
  }

  // Reset resampler history.
  if (decim3_)
    decim3_->Reset();
  if (interp3_)
    interp3_->Reset();
}

bool OpenAudioAudioProcessor::Process(const float *in, float *out,
                                      std::uint32_t frames,
                                      std::uint32_t channels,
                                      std::string *error) {
  if (!in || !out) {
    if (error)
      *error = "null audio buffer";
    return false;
  }

  const std::uint64_t samples64 =
      static_cast<std::uint64_t>(frames) * static_cast<std::uint64_t>(channels);
  const auto samples = static_cast<std::size_t>(samples64);
  if (samples == 0)
    return true;

  // Sticky warning: surface once via AudioPipeline.
  if (!sticky_warning_.empty()) {
    if (error)
      *error = sticky_warning_;
    sticky_warning_.clear();
  }

  // Fail open.
  if (!ort_session_active_ || model_disabled_ || ort_inputs_.empty() ||
      ort_outputs_.empty()) {
    std::copy_n(in, samples, out);
    return true;
  }

  // Determine the active strength curve mode first so we can decide whether to
  // preserve stereo for speaker processing.
  const StrengthMode mode = static_cast<StrengthMode>(strength_mode_.load());
  const bool use_mid_side_stereo =
      (channels == 2) && (mode == StrengthMode::kSpeakerNoiseRemoval ||
                          mode == StrengthMode::kSpeakerRoomEchoRemoval);
  if (mono_in_.size() < frames || mono_out_.size() < frames ||
      (use_mid_side_stereo && side_.size() < frames)) {
    sticky_warning_ = "Open Audio: unexpected pipeline frame size; bypassing.";
    std::copy_n(in, samples, out);
    return true;
  }

  // Convert interleaved audio to mono.
  //  - Mic path: mono input.
  //  - Speaker path (stereo): process Mid only, preserve Side.
  if (channels == 1) {
    std::copy_n(in, frames, mono_in_.data());
  } else if (use_mid_side_stereo) {
    for (std::uint32_t f = 0; f < frames; ++f) {
      const float l = in[static_cast<std::size_t>(f) * 2 + 0];
      const float r = in[static_cast<std::size_t>(f) * 2 + 1];
      mono_in_[f] = 0.5f * (l + r); // Mid
      side_[f] = 0.5f * (l - r);    // Side
    }
  } else {
    // Generic multichannel fallback: average to mono.
    for (std::uint32_t f = 0; f < frames; ++f) {
      float acc = 0.0f;
      for (std::uint32_t c = 0; c < channels; ++c) {
        const std::size_t idx = static_cast<std::size_t>(f) * channels + c;
        acc += in[idx];
      }
      mono_in_[f] = acc / static_cast<float>(channels);
    }
  }

  // Resample to model sample rate if needed.
  if (model_sample_rate_ == 16000) {
    // Expect 10ms @ 48k = 480 samples, and 10ms @ 16k = 160 samples.
    if (frames != 480 || model_frame_samples_ != 160 || !decim3_ || !interp3_) {
      sticky_warning_ = "Open Audio: incompatible frame sizing for 48k<->16k "
                        "resampling (expected 480->160).";
      std::copy_n(in, samples, out);
      return true;
    }
    if (!decim3_->Process(mono_in_.data(), frames, model_in_.data(),
                          model_in_.size())) {
      sticky_warning_ = "Open Audio: resampler downsample failed; bypassing.";
      std::copy_n(in, samples, out);
      return true;
    }
  } else {
    // 48k model: expect the model frame size to match incoming frame size.
    if (frames != model_frame_samples_) {
      sticky_warning_ = "Open Audio: incompatible frame size (model expects " +
                        std::to_string(model_frame_samples_) +
                        " samples, got " + std::to_string(frames) +
                        "); bypassing.";
      std::copy_n(in, samples, out);
      return true;
    }
    std::copy_n(mono_in_.data(), model_in_.size(), model_in_.data());
  }

  // Strength mapping.
  const int strength = strength_.load();
  const float t = Clamp01(static_cast<float>(strength) / 100.0f);
  const float curve = StrengthCurve01(mode, t);

  float wet = 1.0f;
  float dry = 0.0f;
  if (!has_strength_input_) {
    wet = WetMixFromCurve01(mode, curve);
    dry = 1.0f - wet;
  } else {
    const float s_val = strength_input_min_ +
                        curve * (strength_input_max_ - strength_input_min_);
    if (!strength_input_buf_.empty())
      strength_input_buf_[0] = s_val;
  }

  // Refresh state input/output pointers for ping-pong buffers.
  std::size_t base_in = 1;
  if (has_strength_input_)
    base_in += 1;

  for (std::size_t i = 0; i < state_input_names_.size(); ++i) {
    ort_inputs_[base_in + i].data = state_buf_[state_toggle_][i].data();
  }
  for (std::size_t i = 0; i < state_output_names_.size(); ++i) {
    ort_outputs_[1 + i].data = state_buf_[1 - state_toggle_][i].data();
  }

  // Run inference.
  std::string ort_err;
  const std::size_t binding_slot = state_input_names_.empty()
                                       ? 0u
                                       : static_cast<std::size_t>(state_toggle_);
  if (!ort_session_active_->RunPrepared(
          binding_slot, ort_inputs_.data(), ort_inputs_.size(),
          ort_outputs_.data(), ort_outputs_.size(), &ort_err)) {
    if (error && error->empty())
      *error = std::string("Open Audio ORT run failed: ") + ort_err;

    // If CUDA is active and CPU fallback exists, switch over once.
    if (!using_cpu_fallback_ && ort_session_cuda_ && ort_session_cpu_) {
      ort_session_active_ = ort_session_cpu_.get();
      using_cpu_fallback_ = true;
      sticky_warning_ =
          "Open Audio: switched to CPU fallback after a CUDA runtime failure.";
    } else {
      model_disabled_ = true;
      sticky_warning_ = "Open Audio: disabled after repeated runtime failures.";
    }

    std::copy_n(in, samples, out);
    return true;
  }

  state_toggle_ = 1 - state_toggle_;

  // Resample back to pipeline rate.
  if (model_sample_rate_ == 16000) {
    if (!interp3_->Process(model_out_.data(), model_out_.size(),
                           mono_out_.data(), mono_out_.size())) {
      sticky_warning_ = "Open Audio: resampler upsample failed; bypassing.";
      std::copy_n(in, samples, out);
      return true;
    }
  } else {
    std::copy_n(model_out_.data(), model_out_.size(), mono_out_.data());
  }

  // Mix processed audio back to the pipeline format.
  if (channels == 1) {
    for (std::uint32_t f = 0; f < frames; ++f) {
      out[f] = wet * mono_out_[f] + dry * in[f];
    }
  } else if (use_mid_side_stereo) {
    // Stereo-safe speaker processing: preserve Side.
    for (std::uint32_t f = 0; f < frames; ++f) {
      const float mid = mono_out_[f];
      const float side = (f < side_.size()) ? side_[f] : 0.0f;
      const float proc_l = mid + side;
      const float proc_r = mid - side;

      const std::size_t idx = static_cast<std::size_t>(f) * 2;
      const float dry_l = in[idx + 0];
      const float dry_r = in[idx + 1];
      out[idx + 0] = wet * proc_l + dry * dry_l;
      out[idx + 1] = wet * proc_r + dry * dry_r;
    }
  } else {
    // Generic multichannel fallback: fan out processed mono.
    for (std::uint32_t f = 0; f < frames; ++f) {
      const float wet_sample = mono_out_[f];
      for (std::uint32_t c = 0; c < channels; ++c) {
        const std::size_t idx = static_cast<std::size_t>(f) * channels + c;
        const float dry_sample = in[idx];
        out[idx] = wet * wet_sample + dry * dry_sample;
      }
    }
  }

  // Post-processing polish DSP:
  //  - safety limiter to avoid harsh clips when the model output overshoots
  //  - optional gentle presence shelf for Studio Voice mode
  {
    studiocast::audio::dsp::PostDspChain::PresenceShelfConfig ps;
    if (mode == StrengthMode::kStudioVoice && studio_voice_enabled_.load()) {
      ps.enabled = true;
      ps.freq_hz = 3200.0f;
      ps.slope = 1.0f;
      ps.gain_db = 2.0f * curve;
    }
    post_dsp_.SetPresenceShelf(ps);
    post_dsp_.ProcessInPlace(out, frames, channels);
  }

  return true;
}

} // namespace studiocast::open_audio
