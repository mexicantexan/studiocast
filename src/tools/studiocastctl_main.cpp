#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "core/ipc/daemon_client.h"
#include "core/maxine/reason_codes.h"
#include "core/util/exec.h"
#include "core/util/fs.h"
#include "core/util/json.h"
#include "core/video/effects/broadcast_effect_contract.h"
#include "core/video/effects/broadcast_effects.h"
#include "core/video/v4l2loopback.h"
#include "studiocast/version.h"

namespace {

namespace fs = std::filesystem;

std::string ShellQuote(const std::string &s) {
  // Minimal POSIX shell quoting: wrap in single quotes, escape embedded single
  // quotes. Result can be safely concatenated into a `/bin/sh -c` command
  // string.
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('\'');
  for (const char c : s) {
    if (c == '\'') {
      out.append("'\\''");
    } else {
      out.push_back(c);
    }
  }
  out.push_back('\'');
  return out;
}

std::string BuildShellCommand(const std::vector<std::string> &argv) {
  std::ostringstream oss;
  bool first = true;
  for (const auto &a : argv) {
    if (!first)
      oss << ' ';
    first = false;
    oss << ShellQuote(a);
  }
  return oss.str();
}

struct CommandCaptureResult {
  int exit_code = -1;
  bool timed_out = false;
  std::string output;
  std::string error;
};

CommandCaptureResult RunCommandCapture(const std::vector<std::string> &argv,
                                       int timeout_ms = 5000,
                                       std::size_t max_output_bytes =
                                           1024 * 1024) {
  CommandCaptureResult out;
  if (argv.empty()) {
    out.error = "empty argv";
    return out;
  }

  const std::string cmd = BuildShellCommand(argv) + " 2>&1";
  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = timeout_ms;
  options.max_output_bytes = max_output_bytes;
  const auto result = studiocast::util::ExecCapture(cmd, options);
  out.exit_code = result.exit_code;
  out.timed_out = result.timed_out;
  out.output = result.stdout_str;
  if (result.timed_out)
    out.error = "command timed out after " + std::to_string(timeout_ms) + "ms";
  return out;
}

std::string ResolveSiblingToolPath(const char *argv0, const char *toolName) {
  if (!argv0 || !toolName)
    return toolName ? std::string(toolName) : std::string();

  try {
    const fs::path exe = fs::path(argv0);
    if (exe.has_parent_path()) {
      const fs::path candidate = exe.parent_path() / toolName;
      if (fs::exists(candidate))
        return candidate.string();
    }
  } catch (...) {
    // Fall back to PATH.
  }
  return std::string(toolName);
}

std::string LocalTimestampForFilename() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  if (!::localtime_r(&t, &tm))
    return "unknown-time";
  char buf[64];
  if (std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm) == 0)
    return "unknown-time";
  return std::string(buf);
}

std::string LocalTimestampForHumans() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  if (!::localtime_r(&t, &tm))
    return "unknown";
  char buf[64];
  if (std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S %z", &tm) == 0)
    return "unknown";
  return std::string(buf);
}

using studiocast::util::json::Value;

std::optional<double> ParseDouble(std::string_view s) {
  if (s.empty())
    return std::nullopt;
  std::string tmp(s);
  char *end = nullptr;
  const double v = std::strtod(tmp.c_str(), &end);
  if (!end || *end != '\0')
    return std::nullopt;
  return v;
}

bool ParseBoolArg(std::string_view s, bool *out) {
  if (!out)
    return false;
  if (s == "1" || s == "true" || s == "TRUE" || s == "True") {
    *out = true;
    return true;
  }
  if (s == "0" || s == "false" || s == "FALSE" || s == "False") {
    *out = false;
    return true;
  }
  return false;
}

std::string ToLower(std::string s) {
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string MaxineReasonToEnglish(const std::string &code) {
  return studiocast::maxine::reasons::ToEnglish(code);
}

void PrintMaxinePrettyFromStatusJson(const std::string &statusJson) {
  Value root;
  std::string err;
  if (!studiocast::util::json::Parse(statusJson, &root, &err)) {
    std::cerr << "ERROR: invalid status JSON: " << err << "\n";
    return;
  }

  const auto *o = root.AsObject();
  if (!o) {
    std::cerr << "ERROR: status root is not an object\n";
    return;
  }

  // ----------------
  // Maxine (effects)
  // ----------------
  if (auto itMax = o->find("maxine"); itMax != o->end()) {
    const auto *max = itMax->second.AsObject();
    if (!max) {
      std::cout << "Maxine: (invalid maxine object)\n";
    } else {
      bool supported = false;
      if (auto it = max->find("supported"); it != max->end()) {
        if (const auto *b = it->second.AsBool())
          supported = *b;
      }

      std::string blockedReason;
      if (auto it = max->find("blocked_reason"); it != max->end()) {
        if (const auto *s = it->second.AsString())
          blockedReason = *s;
      }

      std::string summary;
      if (auto it = max->find("summary"); it != max->end()) {
        if (const auto *s = it->second.AsString())
          summary = *s;
      }

      std::cout << "Maxine: "
                << (supported
                        ? std::string("OK")
                        : MaxineReasonToEnglish(
                              blockedReason.empty()
                                  ? std::string(
                                        studiocast::maxine::reasons::kUnknown)
                                  : blockedReason))
                << "\n";
      if (!summary.empty()) {
        std::cout << "  " << summary << "\n";
      }

      if (!supported) {
        if (auto it = max->find("blocked_details"); it != max->end()) {
          if (const auto *arr = it->second.AsArray()) {
            for (const auto &v : *arr) {
              if (const auto *s = v.AsString()) {
                if (!s->empty())
                  std::cout << "  - " << *s << "\n";
              }
            }
          }
        }
      }
    }
  } else {
    std::cout << "Maxine: (not reported by daemon)\n";
  }

  // ----------------
  // Video negotiated formats
  // ----------------
  auto getString = [](const Value::Object *obj,
                      const char *key) -> std::string {
    if (!obj)
      return {};
    if (auto it = obj->find(key); it != obj->end()) {
      if (const auto *s = it->second.AsString())
        return *s;
    }
    return {};
  };
  auto getInt = [](const Value::Object *obj, const char *key) -> int {
    if (!obj)
      return 0;
    if (auto it = obj->find(key); it != obj->end()) {
      if (const auto *n = it->second.AsNumber())
        return static_cast<int>(*n);
    }
    return 0;
  };
  auto getDouble = [](const Value::Object *obj,
                      const char *key) -> std::optional<double> {
    if (!obj)
      return std::nullopt;
    if (auto it = obj->find(key); it != obj->end()) {
      if (const auto *n = it->second.AsNumber())
        return *n;
    }
    return std::nullopt;
  };

  const Value::Object *video = nullptr;
  if (auto it = o->find("video"); it != o->end()) {
    video = it->second.AsObject();
  }
  if (!video) {
    std::cout << "Video: (not reported by daemon)\n";
    return;
  }

  std::cout << "Video:\n";
  {
    std::string inDev = getString(video, "input_device");
    std::string outDev = getString(video, "output_device");
    if (inDev.empty())
      inDev = "(auto)";
    if (outDev.empty())
      outDev = "(auto)";
    std::cout << "  source_device: " << inDev << "\n";
    std::cout << "  loopback_device: " << outDev << "\n";
    const std::string requestedFmt =
        getString(video, "output_format_requested");
    if (!requestedFmt.empty())
      std::cout << "  output_format_requested: " << requestedFmt << "\n";
  }

  auto printFmt = [&](const char *label, const char *key) {
    const Value::Object *fmt = nullptr;
    if (auto it = video->find(key); it != video->end()) {
      fmt = it->second.AsObject();
    }
    if (!fmt) {
      std::cout << "  " << label << ": (not reported)\n";
      return;
    }

    const std::string pixfmt = getString(fmt, "pixfmt");
    const int w = getInt(fmt, "width");
    const int h = getInt(fmt, "height");
    const int fpsNum = getInt(fmt, "fps_num");
    const int fpsDen = getInt(fmt, "fps_den");
    const auto fps = getDouble(fmt, "fps");
    const int bpl = getInt(fmt, "bytesperline");
    const int sz = getInt(fmt, "sizeimage");

    if (pixfmt.empty() && w == 0 && h == 0 && bpl == 0 && sz == 0) {
      std::cout << "  " << label << ": (not negotiated yet)\n";
      return;
    }

    std::ostringstream fpsPretty;
    if (fpsNum > 0 && fpsDen > 0) {
      fpsPretty << fpsNum << "/" << fpsDen;
      if (fps.has_value()) {
        fpsPretty << " (~" << std::fixed << std::setprecision(3) << *fps << ")";
      }
    } else if (fps.has_value()) {
      fpsPretty << std::fixed << std::setprecision(3) << *fps;
    } else {
      fpsPretty << "?";
    }

    std::cout << "  " << label << ": "
              << (pixfmt.empty() ? std::string("(unknown)") : pixfmt) << " "
              << w << "x" << h << " @ " << fpsPretty.str() << " fps"
              << " (bytesperline=" << bpl << ", sizeimage=" << sz << ")\n";
  };

  printFmt("capture_format", "capture_format");
  printFmt("output_format", "output_format");
}

bool IsKnownEffectId(const std::string &id) {
  using namespace studiocast::video::effects::contract;
  return id == kEffectIdMirror || id == kEffectIdVirtualBackgroundBlur ||
         id == kEffectIdVirtualBackgroundRemove ||
         id == kEffectIdVirtualBackgroundReplace || id == kEffectIdAutoFrame ||
         id == kEffectIdEyeContact || id == kEffectIdVideoNoiseRemoval ||
         id == kEffectIdVirtualKeyLight || id == kEffectIdVignette;
}

std::string NormalizeEffectId(std::string id) {
  // Allow a few ergonomic aliases; keep contract IDs canonical.
  if (id == "background_blur" || id == "vb_blur")
    return std::string(
        studiocast::video::effects::contract::kEffectIdVirtualBackgroundBlur);
  if (id == "background_remove" || id == "vb_remove")
    return std::string(
        studiocast::video::effects::contract::kEffectIdVirtualBackgroundRemove);
  if (id == "background_replace" || id == "vb_replace")
    return std::string(studiocast::video::effects::contract::
                           kEffectIdVirtualBackgroundReplace);
  return id;
}

const Value::Object *RootObjForEffectsPatch(const Value &root) {
  const auto *o0 = root.AsObject();
  if (!o0)
    return nullptr;
  if (auto it = o0->find("video_effects"); it != o0->end()) {
    if (const auto *o = it->second.AsObject())
      return o;
  }
  if (auto it = o0->find("broadcast_effects"); it != o0->end()) {
    if (const auto *o = it->second.AsObject())
      return o;
  }
  return o0;
}

bool ValidateNoCpuOptions(const Value &root, std::vector<std::string> *warnings,
                          std::string *error) {
  const Value::Object *obj = RootObjForEffectsPatch(root);
  if (!obj)
    return true;

  auto warn = [&](const std::string &w) {
    if (warnings)
      warnings->push_back(w);
  };

  // Canonical engine selector.
  if (auto it = obj->find("engine"); it != obj->end()) {
    if (const auto *s = it->second.AsString()) {
      const std::string v = ToLower(*s);
      if (v == "cpu") {
        warn(
            "backend 'cpu' is not supported; use engine=auto|maxine|open_cuda");
        if (error)
          *error = "engine must be one of: auto, maxine, open_cuda";
        return false;
      }
      studiocast::video::effects::EffectsEnginePreference ep{};
      if (!studiocast::video::effects::ParseEffectsEnginePreference(v, &ep)) {
        if (error)
          *error = "engine must be one of: auto, maxine, open_cuda";
        return false;
      }
    }
  }

  // Legacy/unknown keys that users might try.
  for (const auto &[k, v] : *obj) {
    const std::string lk = ToLower(k);
    if (lk == "background_backend" || lk == "background-backend" ||
        lk == "backend") {
      if (const auto *s = v.AsString()) {
        const std::string vv = ToLower(*s);
        if (vv == "cpu") {
          warn("backend 'cpu' is not supported; use "
               "engine=auto|maxine|open_cuda");
        } else {
          warn("legacy backend key '" + k +
               "' is ignored; use engine=auto|maxine|open_cuda");
        }
      }
    }
  }

  return true;
}

bool ReadTextFileOrStdin(const std::string &path, std::string *out) {
  if (!out)
    return false;
  out->clear();
  if (path == "-" || path.empty()) {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    *out = ss.str();
    return !out->empty();
  }
  const auto jsonTextOpt = studiocast::util::ReadTextFile(path);
  if (!jsonTextOpt)
    return false;
  *out = *jsonTextOpt;
  return true;
}

std::optional<int> ParseStrengthForEffectId(const std::string &effectId,
                                            std::string_view s,
                                            std::string *error) {
  const auto dOpt = ParseDouble(s);
  if (!dOpt) {
    if (error)
      *error = "invalid number for strength";
    return std::nullopt;
  }
  const double d = *dOpt;

  const bool isVb = effectId.rfind("virtual_background.", 0) == 0;
  if (isVb) {
    // VB strength uses the canonical 1..64 range.
    if (d < 0.0) {
      if (error)
        *error = "strength must be >= 0";
      return std::nullopt;
    }
    int v = 0;
    if (d <= 1.0) {
      v = static_cast<int>(std::lround(
          1.0 +
          d * (studiocast::video::effects::contract::kVbStrengthMax - 1)));
    } else {
      v = static_cast<int>(std::lround(d));
    }
    v = std::max(
        studiocast::video::effects::contract::kVbStrengthMin,
        std::min(studiocast::video::effects::contract::kVbStrengthMax, v));
    return v;
  }

  // Most other effects use 0..100-ish percent.
  if (d < 0.0) {
    if (error)
      *error = "strength must be >= 0";
    return std::nullopt;
  }
  int v = 0;
  if (d <= 1.0)
    v = static_cast<int>(std::lround(d * 100.0));
  else
    v = static_cast<int>(std::lround(d));
  v = std::max(0, std::min(100, v));
  return v;
}

std::optional<int> ParsePercent01Or100(std::string_view s, const char *what,
                                       std::string *error) {
  const auto dOpt = ParseDouble(s);
  if (!dOpt) {
    if (error)
      *error = std::string("invalid number for ") + what;
    return std::nullopt;
  }
  double d = *dOpt;
  if (d < 0.0) {
    if (error)
      *error = std::string(what) + " must be >= 0";
    return std::nullopt;
  }
  int v = 0;
  if (d <= 1.0)
    v = static_cast<int>(std::lround(d * 100.0));
  else
    v = static_cast<int>(std::lround(d));
  v = std::max(0, std::min(100, v));
  return v;
}

std::string BuildEnablePatchJson(
    const std::string &effectId, bool enabled,
    const std::optional<std::string> &engine,
    const std::optional<std::string> &modelId,
    const std::optional<int> &strength, const std::optional<int> &intensity,
    const std::optional<int> &smoothing, const std::optional<double> &headroom,
    const std::optional<bool> &lookAway,
    const std::optional<std::string> &removeColor,
    const std::optional<std::string> &replacePath,
    const std::optional<int> &greenscreenMode,
    const std::optional<bool> &greenscreenTemporal,
    const std::optional<std::string> &temperaturePreset,
    const std::optional<int> &directionPanDegrees,
    const std::optional<std::string> &hdriPath,
    const std::optional<bool> &centerOnTrackedFace) {
  using studiocast::util::json::EscapeString;
  using namespace studiocast::video::effects::contract;
  std::ostringstream oss;
  oss << '{';
  bool first = true;

  if (engine && !engine->empty()) {
    oss << "\"engine\":\"" << EscapeString(*engine) << "\"";
    first = false;
  }

  if (!first)
    oss << ',';
  oss << "\"" << EscapeString(effectId) << "\":{";
  oss << "\"" << param::kEnabled << "\":" << (enabled ? "true" : "false");

  if (modelId && !modelId->empty())
    oss << ",\"" << param::kModelId << "\":\"" << EscapeString(*modelId)
        << "\"";

  if (strength)
    oss << ",\"" << param::kStrength << "\":" << *strength;
  if (intensity)
    oss << ",\"" << param::kIntensity << "\":" << *intensity;
  if (smoothing)
    oss << ",\"" << param::kSmoothing << "\":" << *smoothing;
  if (headroom)
    oss << ",\"" << param::kHeadroom << "\":" << *headroom;
  if (lookAway)
    oss << ",\"" << param::kLookAwayEnabled
        << "\":" << (*lookAway ? "true" : "false");

  if (removeColor)
    oss << ",\"" << param::kVbRemoveColor << "\":\""
        << EscapeString(*removeColor) << "\"";
  if (replacePath)
    oss << ",\"" << param::kVbReplacePath << "\":\""
        << EscapeString(*replacePath) << "\"";
  if (greenscreenMode)
    oss << ",\"" << param::kGreenscreenMode << "\":" << *greenscreenMode;
  if (greenscreenTemporal)
    oss << ",\"" << param::kGreenscreenTemporal
        << "\":" << (*greenscreenTemporal ? "true" : "false");

  if (temperaturePreset)
    oss << ",\"" << param::kTemperaturePreset << "\":\""
        << EscapeString(*temperaturePreset) << "\"";
  if (directionPanDegrees)
    oss << ",\"" << param::kDirectionPanDegrees
        << "\":" << *directionPanDegrees;
  if (hdriPath)
    oss << ",\"" << param::kHdriPath << "\":\"" << EscapeString(*hdriPath)
        << "\"";

  if (centerOnTrackedFace)
    oss << ",\"" << param::kCenterOnTrackedFace
        << "\":" << (*centerOnTrackedFace ? "true" : "false");

  oss << "}}";
  return oss.str();
}

bool ExtractOpenCudaModelsFromStatusJson(
    const std::string &statusJson,
    std::vector<std::pair<std::string, std::string>> *outModels,
    std::string *error) {
  if (!outModels)
    return false;
  outModels->clear();

  Value root;
  std::string jerr;
  if (!studiocast::util::json::Parse(statusJson, &root, &jerr)) {
    if (error)
      *error = "invalid JSON from daemon GET_STATUS: " + jerr;
    return false;
  }
  const auto *rootObj = root.AsObject();
  if (!rootObj) {
    if (error)
      *error = "invalid GET_STATUS payload (expected JSON object)";
    return false;
  }

  const Value::Object *openCuda = nullptr;
  if (auto it = rootObj->find("open_cuda"); it != rootObj->end()) {
    openCuda = it->second.AsObject();
  }
  if (!openCuda) {
    if (auto it = rootObj->find("engines"); it != rootObj->end()) {
      if (const auto *engines = it->second.AsObject()) {
        if (auto it2 = engines->find("open_cuda"); it2 != engines->end()) {
          openCuda = it2->second.AsObject();
        }
      }
    }
  }
  if (!openCuda) {
    if (error)
      *error = "GET_STATUS does not include open_cuda engine information";
    return false;
  }

  // Preferred: full model objects.
  if (auto it = openCuda->find("models"); it != openCuda->end()) {
    if (const auto *a = it->second.AsArray()) {
      for (const auto &v : *a) {
        const auto *o = v.AsObject();
        if (!o)
          continue;
        std::string id;
        std::string display;
        if (auto itId = o->find("id"); itId != o->end()) {
          if (const auto *s = itId->second.AsString())
            id = *s;
        }
        if (id.empty())
          continue;
        if (auto itDn = o->find("display_name"); itDn != o->end()) {
          if (const auto *s = itDn->second.AsString())
            display = *s;
        }
        if (display.empty())
          display = id;
        outModels->push_back({id, display});
      }
    }
  }

  // Back-compat: older daemons may only provide installed model IDs.
  if (outModels->empty()) {
    if (auto it = openCuda->find("installed_models"); it != openCuda->end()) {
      if (const auto *a = it->second.AsArray()) {
        for (const auto &v : *a) {
          if (const auto *s = v.AsString()) {
            if (!s->empty())
              outModels->push_back({*s, *s});
          }
        }
      }
    }
  }

  return true;
}

bool PrintOpenCudaModelsFromDaemon() {
  studiocast::ipc::DaemonCallResult res;
  std::string err;
  if (!studiocast::ipc::DaemonCall("GET_STATUS", &res, &err)) {
    std::cerr << "ERROR: " << err << "\n";
    return false;
  }
  if (!res.ok) {
    std::cerr << (res.error_json.empty()
                      ? std::string("{\"error\":\"daemon_error\"}")
                      : res.error_json)
              << "\n";
    return false;
  }

  std::vector<std::pair<std::string, std::string>> models;
  std::string perr;
  if (!ExtractOpenCudaModelsFromStatusJson(res.json, &models, &perr)) {
    std::cerr << "ERROR: " << perr << "\n";
    return false;
  }
  if (models.empty()) {
    std::cerr << "ERROR: no Open CUDA models reported by daemon "
                 "(open_cuda.models/installed_models is empty)\n";
    return false;
  }

  using studiocast::util::json::EscapeString;
  std::ostringstream oss;
  oss << '[';
  bool first = true;
  for (const auto &[id, display] : models) {
    if (id.empty())
      continue;
    if (!first)
      oss << ',';
    first = false;
    oss << "{\"id\":\"" << EscapeString(id) << "\"";
    if (!display.empty() && display != id) {
      oss << ",\"display_name\":\"" << EscapeString(display) << "\"";
    }
    oss << '}';
  }
  oss << ']';
  std::cout << oss.str() << "\n";
  return true;
}

void Usage(const char *argv0) {
  std::cout
      << "studiocastctl - control StudioCast daemon (studiocastd)\n\n"
      << "Usage:\n"
      << "  " << argv0 << " status [--pretty]\n"
      << "  " << argv0 << " debug-report [--out <path>]\n"
      << "  " << argv0 << " config\n"
      << "  " << argv0 << " effects get\n"
      << "  " << argv0 << " effects set --file <effects.json|->\n"
      << "  " << argv0 << " audio get\n"
      << "  " << argv0 << " audio set --file <audio.json|->\n"
      << "  " << argv0 << " audio start\n"
      << "  " << argv0 << " audio stop\n"
      << "  " << argv0
      << " effects enable <effect_id> [--engine auto|maxine|open_cuda] "
         "[--model <id>] [--strength N|0..1] [--intensity N|0..1] ...\n"
      << "  " << argv0 << " effects disable <effect_id>\n"
      << "  " << argv0 << " enable <0|1>\n"
      << "  " << argv0
      << " video set [input=/dev/videoX|auto] [output=/dev/videoY|auto] "
         "[width=N] [height=N] [fps=N] [always_on=0|1] "
         "[allow_cpu_resize=0|1] [compute_backend=auto|cpu|cuda|vulkan] "
         "[vulkan_device=auto|v1:...] [vulkan_allow_cpu=0|1] "
         "[output_format=rgb24|yuyv]\n"
      << "  " << argv0
      << " video vb --model <id> [--mode blur|remove|replace] [--engine "
         "auto|maxine|open_cuda]\n"
      << "  " << argv0 << " video vb --list-models\n"
      << "  " << argv0
      << " video effects [mirror=0|1] "
         "[background=none|blur|remove|replace|auto_frame] "
      << "[background_backend=auto|maxine|open_cuda] [background_strength=N] "
         "[background_remove_color=#RRGGBB] [background_replace_image=/path]\n"
      << "  " << argv0 << " video effects --from <effects.json>\n\n"
      << "Examples:\n"
      << "  " << argv0 << " status\n"
      << "  " << argv0 << " status --pretty\n"
      << "  " << argv0 << " debug-report --out studiocast-debug-report.txt\n"
      << "  " << argv0 << " enable 1\n"
      << "  " << argv0
      << " video set input=/dev/video0 output=/dev/video10 width=1280 "
         "height=720 fps=30 allow_cpu_resize=1 output_format=rgb24\n"
      << "  " << argv0 << " video vb --list-models\n"
      << "  " << argv0
      << " video vb --model modnet-webnn-256-fp32 --mode remove\n"
      << "  " << argv0
      << " video effects mirror=1 background=blur background_strength=10\n"
      << "  " << argv0 << " video effects --from effects.json\n\n"
      << "Notes:\n"
      << "- 'effects get' prints the canonical Broadcast effects JSON "
         "(GET_CONFIG).\n"
      << "- 'config' is an alias for 'effects get'.\n"
      << "- 'effects set' expects a JSON patch (file-based, avoids shell "
         "quoting issues).\n"
      << "- 'video effects' key=value flags are deprecated and mapped "
         "server-side; expect a warning.\n\n"
      << "The canonical effects JSON schema uses effect IDs as keys, e.g.:\n"
      << "  {\n"
      << "    \"mirror\":{\"enabled\":true},\n"
      << "    \"virtual_background.blur\":{\"enabled\":true,\"strength\":8},\n"
      << "    \"auto_frame\":{\"enabled\":false},\n"
      << "    \"eye_contact\":{\"enabled\":false}\n"
      << "  }\n";
}

bool CallOrDie(const std::string &req) {
  studiocast::ipc::DaemonCallResult res;
  std::string err;
  if (!studiocast::ipc::DaemonCall(req, &res, &err)) {
    std::cerr << "ERROR: " << err << "\n";
    return false;
  }

  if (res.ok) {
    if (!res.json.empty()) {
      // Print warnings cleanly to stderr, while keeping stdout
      // machine-readable.
      Value v;
      std::string jerr;
      if (studiocast::util::json::Parse(res.json, &v, &jerr)) {
        if (const auto *o = v.AsObject()) {
          if (auto it = o->find("warnings"); it != o->end()) {
            if (const auto *a = it->second.AsArray()) {
              for (const auto &wv : *a) {
                if (const auto *ws = wv.AsString()) {
                  std::cerr << "WARNING: " << *ws << "\n";
                }
              }
            }
          }
        }
      }
      std::cout << res.json << "\n";
    }
    return true;
  }

  std::cerr << (res.error_json.empty() ? std::string("{\"error\":\"unknown\"}")
                                       : res.error_json)
            << "\n";
  return false;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    Usage(argv[0]);
    return 2;
  }

  const std::string cmd = argv[1];
  if (cmd == "--version" || cmd == "-v") {
    std::printf("studiocastctl %s (%s)\n", STUDIOCAST_VERSION,
                STUDIOCAST_GIT_SHA);
    return 0;
  }
  if (cmd == "--help" || cmd == "-h" || cmd == "help") {
    Usage(argv[0]);
    return 0;
  }

  if (cmd == "status") {
    bool pretty = false;
    for (int i = 2; i < argc; ++i) {
      const std::string_view a =
          argv[i] ? std::string_view(argv[i]) : std::string_view();
      if (a == "--pretty")
        pretty = true;
    }
    if (!pretty) {
      return CallOrDie("GET_STATUS") ? 0 : 1;
    }

    studiocast::ipc::DaemonCallResult res;
    std::string err;
    if (!studiocast::ipc::DaemonCall("GET_STATUS", &res, &err)) {
      std::cerr << "ERROR: " << err << "\n";
      return 1;
    }
    if (!res.ok) {
      std::cerr << (res.error_json.empty()
                        ? std::string("{\"error\":\"daemon_error\"}")
                        : res.error_json)
                << "\n";
      return 1;
    }

    PrintMaxinePrettyFromStatusJson(res.json);
    return 0;
  }

  if (cmd == "debug-report") {
    std::string outPath;
    for (int i = 2; i < argc; ++i) {
      const std::string_view a =
          argv[i] ? std::string_view(argv[i]) : std::string_view();
      if (a == "--out" || a == "-o") {
        if (i + 1 >= argc || !argv[i + 1]) {
          std::cerr << "ERROR: --out requires a path\n";
          return 2;
        }
        outPath = argv[i + 1] ? std::string(argv[i + 1]) : std::string();
        ++i;
        continue;
      }
      if (a == "--help" || a == "-h") {
        Usage(argv[0]);
        return 0;
      }
    }
    if (outPath.empty()) {
      outPath = std::string("studiocast-debug-report-") +
                LocalTimestampForFilename() + ".txt";
    }

    std::ofstream out(outPath, std::ios::out | std::ios::trunc);
    if (!out) {
      std::cerr << "ERROR: failed to open output file: " << outPath << "\n";
      return 1;
    }

    auto section = [&](const std::string &name) {
      out << "\n";
      out << "================================================================="
             "===============\n";
      out << name << "\n";
      out << "================================================================="
             "===============\n";
    };

    out << "StudioCast debug report\n";
    out << "Generated: " << LocalTimestampForHumans() << "\n";
    out << "studiocastctl: " << STUDIOCAST_VERSION << " (" << STUDIOCAST_GIT_SHA
        << ")\n";

    bool okAll = true;

    section("IPC: GET_STATUS");
    {
      studiocast::ipc::DaemonCallResult res;
      std::string err;
      if (!studiocast::ipc::DaemonCall("GET_STATUS", &res, &err)) {
        okAll = false;
        out << "DaemonCall failed: " << err << "\n";
      } else if (!res.ok) {
        okAll = false;
        out << "Daemon returned ok=false\n";
        out << (res.error_json.empty()
                    ? std::string("{\"error\":\"daemon_error\"}")
                    : res.error_json)
            << "\n";
      } else {
        out << res.json << "\n";
      }
    }

    section("Video: v4l2loopback diagnostics");
    {
      const auto rep = studiocast::video::ProbeLoopbackDiagnostics();
      const auto text = rep.ToText();
      out << text;
      if (!text.empty() && text.back() != '\n')
        out << "\n";
    }

    section("IPC: GET_AUDIO_CONFIG");
    {
      studiocast::ipc::DaemonCallResult res;
      std::string err;
      if (!studiocast::ipc::DaemonCall("GET_AUDIO_CONFIG", &res, &err)) {
        okAll = false;
        out << "DaemonCall failed: " << err << "\n";
      } else if (!res.ok) {
        okAll = false;
        out << "Daemon returned ok=false\n";
        out << (res.error_json.empty()
                    ? std::string("{\"error\":\"daemon_error\"}")
                    : res.error_json)
            << "\n";
      } else {
        out << res.json << "\n";
      }
    }

    auto pulseSnapshot = [&](const std::vector<std::string> &cmdArgv) {
      section(std::string("Exec: ") + BuildShellCommand(cmdArgv));
      const auto r = RunCommandCapture(cmdArgv, /*timeout_ms=*/1500,
                                       /*max_output_bytes=*/64 * 1024);
      out << "exit_code: " << r.exit_code << "\n";
      if (r.timed_out)
        out << "timed_out: true\n";
      if (!r.error.empty())
        out << "note: " << r.error << "\n";
      out << r.output;
      if (!r.output.empty() && r.output.back() != '\n')
        out << "\n";
    };

    pulseSnapshot({"pactl", "info"});
    pulseSnapshot({"pactl", "get-default-source"});
    pulseSnapshot({"pactl", "get-default-sink"});
    pulseSnapshot({"pactl", "list", "short", "sources"});
    pulseSnapshot({"pactl", "list", "short", "sinks"});
    pulseSnapshot({"pactl", "list", "short", "modules"});

    const std::string probePath =
        ResolveSiblingToolPath(argv[0], "studiocast-probe");
    const std::string maxinePath =
        ResolveSiblingToolPath(argv[0], "studiocast-maxine");
    const std::string openPath =
        ResolveSiblingToolPath(argv[0], "studiocast-open");

    section(std::string("Exec: ") + probePath + " --json");
    {
      const auto r = RunCommandCapture({probePath, "--json"});
      if (!r.error.empty()) {
        okAll = false;
        out << "spawn error: " << r.error << "\n";
      }
      out << "exit_code: " << r.exit_code << "\n";
      if (r.exit_code != 0)
        okAll = false;
      out << r.output;
      if (!r.output.empty() && r.output.back() != '\n')
        out << "\n";
    }

    section(std::string("Exec: ") + maxinePath + " paths");
    {
      const auto r = RunCommandCapture({maxinePath, "paths"});
      if (!r.error.empty()) {
        okAll = false;
        out << "spawn error: " << r.error << "\n";
      }
      out << "exit_code: " << r.exit_code << "\n";
      if (r.exit_code != 0)
        okAll = false;
      out << r.output;
      if (!r.output.empty() && r.output.back() != '\n')
        out << "\n";
    }

    section(std::string("Exec: ") + maxinePath + " install-hints");
    {
      const auto r = RunCommandCapture({maxinePath, "install-hints"});
      if (!r.error.empty()) {
        okAll = false;
        out << "spawn error: " << r.error << "\n";
      }
      out << "exit_code: " << r.exit_code << "\n";
      if (r.exit_code != 0)
        okAll = false;
      out << r.output;
      if (!r.output.empty() && r.output.back() != '\n')
        out << "\n";
    }

    section(std::string("Exec: ") + openPath + " audio-list-models");
    {
      const auto r = RunCommandCapture({openPath, "audio-list-models"});
      if (!r.error.empty()) {
        okAll = false;
        out << "spawn error: " << r.error << "\n";
      }
      out << "exit_code: " << r.exit_code << "\n";
      if (r.exit_code != 0)
        okAll = false;
      out << r.output;
      if (!r.output.empty() && r.output.back() != '\n')
        out << "\n";
    }

    section(std::string("Exec: ") + openPath + " audio-install-hints");
    {
      const auto r = RunCommandCapture({openPath, "audio-install-hints"});
      if (!r.error.empty()) {
        okAll = false;
        out << "spawn error: " << r.error << "\n";
      }
      out << "exit_code: " << r.exit_code << "\n";
      if (r.exit_code != 0)
        okAll = false;
      out << r.output;
      if (!r.output.empty() && r.output.back() != '\n')
        out << "\n";
    }

    out.flush();

    std::cout << outPath << "\n";
    return okAll ? 0 : 1;
  }

  if (cmd == "config") {
    // Back-compat alias.
    return CallOrDie("GET_CONFIG") ? 0 : 1;
  }
  if (cmd == "effects") {
    if (argc < 3) {
      Usage(argv[0]);
      return 2;
    }
    const std::string sub = argv[2] ? std::string(argv[2]) : std::string();
    if (sub == "get") {
      return CallOrDie("GET_CONFIG") ? 0 : 1;
    }
    if (sub == "set") {
      std::string file;
      for (int i = 3; i < argc; ++i) {
        const std::string_view a =
            argv[i] ? std::string_view(argv[i]) : std::string_view();
        if (a == "--file" || a == "--from") {
          if (i + 1 >= argc) {
            std::cerr << "ERROR: --file requires a path (or '-' for stdin)\n";
            return 2;
          }
          file = argv[i + 1] ? std::string(argv[i + 1]) : std::string();
          break;
        }
      }
      if (file.empty()) {
        std::cerr << "ERROR: effects set requires --file <effects.json|->\n";
        return 2;
      }

      std::string jsonText;
      if (!ReadTextFileOrStdin(file, &jsonText)) {
        std::cerr << "ERROR: failed to read effects JSON from: " << file
                  << "\n";
        return 2;
      }

      Value root;
      std::string jerr;
      if (!studiocast::util::json::Parse(jsonText, &root, &jerr)) {
        std::cerr << "ERROR: invalid JSON: " << jerr << "\n";
        return 2;
      }

      std::vector<std::string> warnings;
      std::string verr;
      if (!ValidateNoCpuOptions(root, &warnings, &verr)) {
        for (const auto &w : warnings)
          std::cerr << "WARNING: " << w << "\n";
        std::cerr << "ERROR: " << verr << "\n";
        return 2;
      }
      for (const auto &w : warnings)
        std::cerr << "WARNING: " << w << "\n";

      const std::string minified = studiocast::util::json::Minify(jsonText);
      const std::string req = std::string("SET_VIDEO_EFFECTS_JSON ") + minified;
      return CallOrDie(req) ? 0 : 1;
    }

    if (sub == "enable" || sub == "disable") {
      if (argc < 4) {
        std::cerr << "ERROR: effects " << sub << " requires an effect_id\n";
        return 2;
      }
      const bool enabled = (sub == "enable");
      std::string effectId =
          NormalizeEffectId(argv[3] ? std::string(argv[3]) : std::string());
      if (!IsKnownEffectId(effectId)) {
        std::cerr << "ERROR: unknown effect_id: " << effectId << "\n";
        return 2;
      }

      std::optional<std::string> engine;
      std::optional<std::string> modelId;
      std::optional<int> strength;
      std::optional<int> intensity;
      std::optional<int> smoothing;
      std::optional<double> headroom;
      std::optional<bool> lookAway;
      std::optional<std::string> removeColor;
      std::optional<std::string> replacePath;
      std::optional<int> greenscreenMode;
      std::optional<bool> greenscreenTemporal;
      std::optional<std::string> temperaturePreset;
      std::optional<int> directionPanDegrees;
      std::optional<std::string> hdriPath;
      std::optional<bool> centerOnTrackedFace;

      for (int i = 4; i < argc; ++i) {
        const std::string_view a =
            argv[i] ? std::string_view(argv[i]) : std::string_view();
        auto needValue =
            [&](const char *flag) -> std::optional<std::string_view> {
          if (a != flag)
            return std::nullopt;
          if (i + 1 >= argc || !argv[i + 1]) {
            std::cerr << "ERROR: " << flag << " requires a value\n";
            return std::nullopt;
          }
          ++i;
          return std::string_view(argv[i]);
        };

        if (auto v = needValue("--engine")) {
          const std::string vv = ToLower(std::string(*v));
          if (vv == "cpu") {
            std::cerr << "WARNING: backend 'cpu' is not supported; use "
                         "engine=auto|maxine|open_cuda\n";
            std::cerr << "ERROR: engine must be auto|maxine|open_cuda\n";
            return 2;
          }
          studiocast::video::effects::EffectsEnginePreference ep{};
          if (!studiocast::video::effects::ParseEffectsEnginePreference(vv,
                                                                        &ep)) {
            std::cerr << "ERROR: engine must be auto|maxine|open_cuda\n";
            return 2;
          }
          engine = studiocast::video::effects::ToString(ep);
          continue;
        }
        if (auto v = needValue("--model")) {
          modelId = std::string(*v);
          continue;
        }
        if (auto v = needValue("--strength")) {
          std::string perr;
          auto s2 = ParseStrengthForEffectId(effectId, *v, &perr);
          if (!s2) {
            std::cerr << "ERROR: " << perr << "\n";
            return 2;
          }
          strength = *s2;
          continue;
        }
        if (auto v = needValue("--intensity")) {
          std::string perr;
          auto p = ParsePercent01Or100(*v, "intensity", &perr);
          if (!p) {
            std::cerr << "ERROR: " << perr << "\n";
            return 2;
          }
          intensity = *p;
          continue;
        }
        if (auto v = needValue("--smoothing")) {
          auto d = ParseDouble(*v);
          if (!d) {
            std::cerr << "ERROR: invalid number for smoothing\n";
            return 2;
          }
          smoothing =
              std::max(0, std::min(100, static_cast<int>(std::lround(*d))));
          continue;
        }
        if (auto v = needValue("--headroom")) {
          auto d = ParseDouble(*v);
          if (!d) {
            std::cerr << "ERROR: invalid number for headroom\n";
            return 2;
          }
          headroom = std::max(0.0, std::min(1.0, *d));
          continue;
        }
        if (auto v = needValue("--look-away-enabled")) {
          bool b = false;
          if (!ParseBoolArg(*v, &b)) {
            std::cerr << "ERROR: --look-away-enabled must be 0|1|true|false\n";
            return 2;
          }
          lookAway = b;
          continue;
        }
        if (auto v = needValue("--remove-color")) {
          removeColor = std::string(*v);
          continue;
        }
        if (auto v = needValue("--replace-path")) {
          replacePath = std::string(*v);
          continue;
        }
        if (auto v = needValue("--greenscreen-mode")) {
          auto d = ParseDouble(*v);
          if (!d) {
            std::cerr << "ERROR: invalid number for greenscreen-mode\n";
            return 2;
          }
          greenscreenMode = std::max(0, static_cast<int>(std::lround(*d)));
          continue;
        }
        if (auto v = needValue("--greenscreen-temporal")) {
          bool b = false;
          if (!ParseBoolArg(*v, &b)) {
            std::cerr
                << "ERROR: --greenscreen-temporal must be 0|1|true|false\n";
            return 2;
          }
          greenscreenTemporal = b;
          continue;
        }
        if (auto v = needValue("--temperature-preset")) {
          temperaturePreset = ToLower(std::string(*v));
          continue;
        }
        if (auto v = needValue("--direction-pan-degrees")) {
          auto d = ParseDouble(*v);
          if (!d) {
            std::cerr << "ERROR: invalid number for direction-pan-degrees\n";
            return 2;
          }
          directionPanDegrees = static_cast<int>(std::lround(*d));
          continue;
        }
        if (auto v = needValue("--hdri-path")) {
          hdriPath = std::string(*v);
          continue;
        }
        if (auto v = needValue("--center-on-tracked-face")) {
          bool b = false;
          if (!ParseBoolArg(*v, &b)) {
            std::cerr
                << "ERROR: --center-on-tracked-face must be 0|1|true|false\n";
            return 2;
          }
          centerOnTrackedFace = b;
          continue;
        }
      }

      if (enabled &&
          effectId == std::string(studiocast::video::effects::contract::
                                      kEffectIdVirtualBackgroundReplace) &&
          !replacePath) {
        std::cerr << "WARNING: virtual_background.replace requires "
                     "--replace-path; daemon will reject if empty\n";
      }

      const std::string patch = BuildEnablePatchJson(
          effectId, enabled, engine, modelId, strength, intensity, smoothing,
          headroom, lookAway, removeColor, replacePath, greenscreenMode,
          greenscreenTemporal, temperaturePreset, directionPanDegrees, hdriPath,
          centerOnTrackedFace);
      const std::string req = std::string("SET_VIDEO_EFFECTS_JSON ") + patch;
      return CallOrDie(req) ? 0 : 1;
    }

    std::cerr << "Unknown effects subcommand: " << sub << "\n";
    return 2;
  }
  if (cmd == "enable") {
    if (argc < 3) {
      std::cerr << "enable requires 0|1\n";
      return 2;
    }
    return CallOrDie(std::string("SET_ENABLED ") + argv[2]) ? 0 : 1;
  }

  if (cmd == "audio") {
    if (argc < 3) {
      Usage(argv[0]);
      return 2;
    }

    const std::string sub = argv[2];
    if (sub == "get" || sub == "config") {
      return CallOrDie("GET_AUDIO_CONFIG") ? 0 : 1;
    }
    if (sub == "start") {
      return CallOrDie("AUDIO_START") ? 0 : 1;
    }
    if (sub == "stop") {
      return CallOrDie("AUDIO_STOP") ? 0 : 1;
    }
    if (sub == "set") {
      std::string path;
      for (int i = 3; i < argc; ++i) {
        const std::string_view a =
            argv[i] ? std::string_view(argv[i]) : std::string_view();
        if (a == "--file") {
          if (i + 1 >= argc) {
            std::cerr << "--file requires a file path (or - for stdin)\n";
            return 2;
          }
          path = argv[i + 1] ? std::string(argv[i + 1]) : std::string();
          break;
        }
      }
      if (path.empty()) {
        std::cerr << "audio set requires --file <audio.json|->\n";
        return 2;
      }

      std::string text;
      if (!ReadTextFileOrStdin(path, &text)) {
        std::cerr << "ERROR: failed to read input: " << path << "\n";
        return 2;
      }
      const std::string minified = studiocast::util::json::Minify(text);
      const std::string req = std::string("SET_AUDIO_CONFIG ") + minified;
      return CallOrDie(req) ? 0 : 1;
    }

    std::cerr << "Unknown audio subcommand: " << sub << "\n";
    return 2;
  }

  if (cmd == "video") {
    if (argc < 3) {
      Usage(argv[0]);
      return 2;
    }
    const std::string sub = argv[2];
    if (sub == "set") {
      std::string req = "SET_VIDEO_CONFIG";
      for (int i = 3; i < argc; ++i) {
        req.push_back(' ');
        req.append(argv[i]);
      }
      return CallOrDie(req) ? 0 : 1;
    }

    if (sub == "vb") {
      bool listModels = false;
      std::optional<std::string> modelId;
      std::optional<std::string> engine;
      std::string effectId = std::string(studiocast::video::effects::contract::
                                             kEffectIdVirtualBackgroundRemove);

      for (int i = 3; i < argc; ++i) {
        const std::string_view a =
            argv[i] ? std::string_view(argv[i]) : std::string_view();
        auto needValue =
            [&](const char *flag) -> std::optional<std::string_view> {
          if (a != flag)
            return std::nullopt;
          if (i + 1 >= argc || !argv[i + 1]) {
            std::cerr << "ERROR: " << flag << " requires a value\n";
            return std::nullopt;
          }
          ++i;
          return std::string_view(argv[i]);
        };

        if (a == "--list-models" || a == "--models") {
          listModels = true;
          continue;
        }
        if (auto v = needValue("--model")) {
          modelId = std::string(*v);
          continue;
        }
        if (auto v = needValue("--mode")) {
          const std::string m = ToLower(std::string(*v));
          if (m == "blur")
            effectId = std::string(studiocast::video::effects::contract::
                                       kEffectIdVirtualBackgroundBlur);
          else if (m == "remove")
            effectId = std::string(studiocast::video::effects::contract::
                                       kEffectIdVirtualBackgroundRemove);
          else if (m == "replace")
            effectId = std::string(studiocast::video::effects::contract::
                                       kEffectIdVirtualBackgroundReplace);
          else {
            std::cerr << "ERROR: --mode must be blur|remove|replace\n";
            return 2;
          }
          continue;
        }
        if (auto v = needValue("--engine")) {
          const std::string vv = ToLower(std::string(*v));
          if (vv == "cpu") {
            std::cerr << "WARNING: backend 'cpu' is not supported; use "
                         "engine=auto|maxine|open_cuda\n";
            std::cerr << "ERROR: engine must be auto|maxine|open_cuda\n";
            return 2;
          }
          studiocast::video::effects::EffectsEnginePreference ep{};
          if (!studiocast::video::effects::ParseEffectsEnginePreference(vv,
                                                                        &ep)) {
            std::cerr << "ERROR: engine must be auto|maxine|open_cuda\n";
            return 2;
          }
          engine = studiocast::video::effects::ToString(ep);
          continue;
        }
        if (a == "--help" || a == "-h") {
          Usage(argv[0]);
          return 0;
        }

        std::cerr << "ERROR: unknown flag for 'video vb': " << a << "\n";
        return 2;
      }

      if (listModels) {
        return PrintOpenCudaModelsFromDaemon() ? 0 : 1;
      }

      if (!modelId || modelId->empty()) {
        std::cerr
            << "ERROR: video vb requires --model <id> (or --list-models)\n";
        return 2;
      }

      const std::string patch =
          BuildEnablePatchJson(effectId,
                               /*enabled=*/true, engine, modelId,
                               /*strength=*/std::nullopt,
                               /*intensity=*/std::nullopt,
                               /*smoothing=*/std::nullopt,
                               /*headroom=*/std::nullopt,
                               /*lookAway=*/std::nullopt,
                               /*removeColor=*/std::nullopt,
                               /*replacePath=*/std::nullopt,
                               /*greenscreenMode=*/std::nullopt,
                               /*greenscreenTemporal=*/std::nullopt,
                               /*temperaturePreset=*/std::nullopt,
                               /*directionPanDegrees=*/std::nullopt,
                               /*hdriPath=*/std::nullopt,
                               /*centerOnTrackedFace=*/std::nullopt);
      const std::string req = std::string("SET_VIDEO_EFFECTS_JSON ") + patch;
      return CallOrDie(req) ? 0 : 1;
    }
    if (sub == "effects") {
      std::string fromPath;
      for (int i = 3; i < argc; ++i) {
        const std::string_view a =
            argv[i] ? std::string_view(argv[i]) : std::string_view();
        if (a == "--from") {
          if (i + 1 >= argc) {
            std::cerr << "--from requires a file path\n";
            return 2;
          }
          fromPath = argv[i + 1] ? std::string(argv[i + 1]) : std::string();
          break;
        }
      }

      if (!fromPath.empty()) {
        const auto jsonTextOpt = studiocast::util::ReadTextFile(fromPath);
        if (!jsonTextOpt) {
          std::cerr << "ERROR: failed to read file: " << fromPath << "\n";
          return 2;
        }
        const std::string minified =
            studiocast::util::json::Minify(*jsonTextOpt);
        const std::string req =
            std::string("SET_VIDEO_EFFECTS_JSON ") + minified;
        return CallOrDie(req) ? 0 : 1;
      }

      // Legacy key=value flags (server maps into canonical effects and returns
      // a warning).
      std::string req = "SET_VIDEO_EFFECTS";
      for (int i = 3; i < argc; ++i) {
        if (!argv[i])
          continue;
        req.push_back(' ');
        req.append(argv[i]);
      }
      return CallOrDie(req) ? 0 : 1;
    }

    std::cerr << "Unknown video subcommand: " << sub << "\n";
    return 2;
  }

  std::cerr << "Unknown command: " << cmd << "\n";
  Usage(argv[0]);
  return 2;
}
