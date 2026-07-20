#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "../core/open_video/diagnose.h"
#include "core/audio/audio_device_safety.h"
#include "core/audio/effects/broadcast_audio_effects_json.h"
#include "core/audio/virtual_audio_service.h"
#include "core/config/daemon_config.h"
#include "core/ipc/daemon_server.h"
#include "core/ipc/daemon_socket.h"
#include "core/maxine/maxine_manager.h"
#include "core/open_audio/open_audio_diagnose.h"
#include "core/util/json.h"
#include "core/util/xdg.h"
#include "core/video/broadcast_camera_effects_json.h"
#include "core/video/camera_effects_json.h"
#include "core/video/effects/broadcast_effect_contract.h"
#include "core/video/effects/broadcast_effect_rules.h"
#include "core/video/v4l2loopback.h"
#include "core/video/virtual_camera_service.h"
#include "studiocast/version.h"

#if STUDIOCAST_ENABLE_OPEN_VULKAN
#include "core/vulkan/vulkan_device.h"
#endif

namespace {

std::atomic_bool g_running{true};

void HandleSignal(int) { g_running.store(false); }

bool HasArg(int argc, char **argv, const std::string &flag) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] && flag == argv[i])
      return true;
  }
  return false;
}

std::string GetArgValue(int argc, char **argv, const std::string &key) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] && key == argv[i]) {
      return argv[i + 1] ? std::string(argv[i + 1]) : std::string();
    }
  }
  return {};
}

int GetArgInt(int argc, char **argv, const std::string &key, int fallback) {
  const auto v = GetArgValue(argc, argv, key);
  if (v.empty())
    return fallback;
  return std::atoi(v.c_str());
}

void Usage(const char *argv0) {
  std::cout
      << "StudioCast background service (studiocastd)\n\n"
      << "Usage:\n"
      << "  " << argv0 << " [options]\n\n"
      << "Options:\n"
      << "  --input /dev/videoX      Input camera (default: auto)\n"
      << "  --output /dev/videoY     Output v4l2loopback (default: auto)\n"
      << "  --capture-mode M         Capture mode: requested|auto (default: "
         "requested)\n"
      << "  --width N                Requested width (default: 1280)\n"
      << "  --height N               Requested height (default: 720)\n"
      << "  --fps N                  Requested fps (default: 30)\n"
      << "  --output-format F        Virtual camera output: rgb24|yuyv "
         "(default: rgb24)\n"
      << "  --mirror                 Enable mirror (horizontal flip)\n"
      << "  --background MODE         Background effect: "
         "none|blur|remove|replace|auto_frame (default: none)\n"
      << "  --background-backend B    Effects engine preference: "
         "auto|maxine|open_cuda (default: auto)\n"
      << "  --background-strength N   Intensity knob (default: 8)\n"
      << "  --background-remove-color #RRGGBB  Remove-mode background color "
         "(default: #000000)\n"
      << "  --background-replace-image PATH    Replace-mode background image "
         "path\n"
      << "  --poll-ms N              Consumer poll interval (default: 250)\n"
      << "  --stop-grace-ms N        Stop after N ms without consumers "
         "(default: 1000)\n"
      << "  --always-on              Run pipeline even with no consumers\n"
      << "  --version                Print version and exit\n"
      << "  -h, --help               Show this help\n\n"
      << "Notes:\n"
      << "  - This daemon does NOT run modprobe for you.\n"
      << "  - Consumer-driven start/stop is based on scanning /proc/*/fd for "
         "open handles\n"
      << "    to the v4l2loopback device (best-effort; typically works when "
         "OBS/Zoom run\n"
      << "    under the same user).\n";
}

std::string ChooseWritableLoopbackDevice() {
  const auto rep = studiocast::video::ProbeLoopback();
  for (const auto &d : rep.devices) {
    if (d.is_loopback && d.can_write)
      return d.dev_node;
  }
  return {};
}

// -----------------------------
// IPC helpers
// -----------------------------

std::string JsonEscape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 16);
  for (char c : s) {
    switch (c) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        // Control chars -> \u00XX
        const char *hex = "0123456789abcdef";
        out += "\\u00";
        out.push_back(hex[(c >> 4) & 0xF]);
        out.push_back(hex[c & 0xF]);
      } else {
        out.push_back(c);
      }
    }
  }
  return out;
}

std::string BoolJson(bool v) { return v ? "true" : "false"; }

std::string OpenAudioRuntimeStatusToJson(
    const studiocast::audio::OpenAudioRuntimeStatus &st) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"active\":" << BoolJson(st.active) << ",";
  oss << "\"active_provider\":\"" << JsonEscape(st.active_provider) << "\",";
  oss << "\"using_cpu_fallback\":" << BoolJson(st.using_cpu_fallback) << ",";
  oss << "\"disabled\":" << BoolJson(st.disabled) << ",";
  oss << "\"selected_model_id\":\"" << JsonEscape(st.selected_model_id)
      << "\",";
  oss << "\"selected_model_path\":\"" << JsonEscape(st.selected_model_path)
      << "\",";
  oss << "\"last_runtime_warning\":\"" << JsonEscape(st.last_runtime_warning)
      << "\"";
  oss << "}";
  return oss.str();
}

struct ParsedCommand {
  std::string cmd;
  std::map<std::string, std::string> kv;
  std::vector<std::string> args;
};

ParsedCommand ParseLine(const std::string &line) {
  std::istringstream iss(line);
  ParsedCommand pc;
  iss >> pc.cmd;
  std::string tok;
  while (iss >> tok) {
    const auto eq = tok.find('=');
    if (eq != std::string::npos) {
      pc.kv[tok.substr(0, eq)] = tok.substr(eq + 1);
    } else {
      pc.args.push_back(tok);
    }
  }
  return pc;
}

bool ParseBoolArg(const std::string &raw, bool *out) {
  if (!out)
    return false;
  std::string s = raw;
  for (char &c : s) {
    if (c >= 'A' && c <= 'Z')
      c = static_cast<char>(c - 'A' + 'a');
  }
  if (s == "1" || s == "true" || s == "yes" || s == "on" || s == "enable" ||
      s == "enabled") {
    *out = true;
    return true;
  }
  if (s == "0" || s == "false" || s == "no" || s == "off" || s == "disable" ||
      s == "disabled") {
    *out = false;
    return true;
  }
  return false;
}

bool ParseRgbHex(const std::string &raw, std::uint32_t *out) {
  if (!out)
    return false;
  std::string s = raw;
  // trim (minimal; enough for our tokenized IPC)
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' ||
                        s.back() == '\n' || s.back() == '\r'))
    s.pop_back();
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t' ||
                        s.front() == '\n' || s.front() == '\r'))
    s.erase(s.begin());

  if (s.empty())
    return false;
  if (!s.empty() && s[0] == '#')
    s.erase(0, 1);
  if (s.size() != 6)
    return false;

  std::uint32_t v = 0;
  for (const char c : s) {
    v <<= 4u;
    if (c >= '0' && c <= '9') {
      v |= static_cast<std::uint32_t>(c - '0');
    } else if (c >= 'a' && c <= 'f') {
      v |= static_cast<std::uint32_t>(c - 'a' + 10);
    } else if (c >= 'A' && c <= 'F') {
      v |= static_cast<std::uint32_t>(c - 'A' + 10);
    } else {
      return false;
    }
  }
  *out = v;
  return true;
}

std::string FormatRgbHex(std::uint32_t rgb) {
  std::ostringstream oss;
  oss << "#" << std::hex << std::nouppercase << std::setfill('0')
      << std::setw(6) << (rgb & 0xFFFFFFu);
  return oss.str();
}

std::string ToLowerAscii(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

std::string TrimAscii(std::string s) {
  auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!s.empty() && is_space(static_cast<unsigned char>(s.back())))
    s.pop_back();
  while (!s.empty() && is_space(static_cast<unsigned char>(s.front())))
    s.erase(s.begin());
  return s;
}

bool IsValidVulkanStableSelector(std::string_view value) {
  if (value.size() < 10 || value.rfind("v1:", 0) != 0)
    return false;
  return std::all_of(value.begin(), value.end(), [](unsigned char c) {
    return std::isalnum(c) || c == ':' || c == '-';
  });
}

struct DiagnosticsJsonSnapshot {
  std::string maxine;
  std::string open_cuda;
  std::string open_vulkan;
  std::string open_audio;
  std::string loopback;
};

std::mutex &DiagnosticsJsonCacheMutex() {
  static std::mutex mu;
  return mu;
}

DiagnosticsJsonSnapshot &DiagnosticsJsonCacheStorage() {
  static DiagnosticsJsonSnapshot snapshot;
  return snapshot;
}

#if STUDIOCAST_ENABLE_OPEN_VULKAN
bool ConfigureProcessVulkanSelection(
    const studiocast::config::DaemonConfig &config, std::string *error) {
  studiocast::vulkan::VulkanDeviceSelection selection;
  if (config.video_vulkan_device != "auto" &&
      !config.video_vulkan_device.empty()) {
    selection.requested_stable_id = config.video_vulkan_device;
    selection.allow_cpu_in_auto = config.video_vulkan_allow_cpu;
    selection.request = "stable_id:" + config.video_vulkan_device;
    selection.source = "daemon_config";
  } else {
    const char *index = std::getenv("STUDIOCAST_VULKAN_DEVICE_INDEX");
    const char *allowCpu = std::getenv("STUDIOCAST_VULKAN_ALLOW_CPU");
    if (!studiocast::vulkan::detail::ParseVulkanDeviceSelection(
            index ? index : "", allowCpu ? allowCpu : "", &selection,
            error)) {
      studiocast::vulkan::ClearProcessVulkanDeviceSelection();
      return false;
    }
    if (config.video_vulkan_allow_cpu) {
      selection.allow_cpu_in_auto = true;
      if (!selection.requested_index.has_value()) {
        selection.request = "auto_with_cpu";
        selection.source = "daemon_config";
      }
    }
  }
  studiocast::vulkan::SetProcessVulkanDeviceSelection(selection);
  return true;
}
#endif

DiagnosticsJsonSnapshot GetDiagnosticsJsonCacheSnapshot() {
  std::lock_guard<std::mutex> lock(DiagnosticsJsonCacheMutex());
  return DiagnosticsJsonCacheStorage();
}

DiagnosticsJsonSnapshot ComputeDiagnosticsJsonSnapshot() {
  DiagnosticsJsonSnapshot snapshot;

  studiocast::maxine::MaxineManager mm;
  snapshot.maxine = mm.Diagnose(/*verbose_probe=*/false).ToJson();
  snapshot.open_cuda =
      studiocast::open_cuda::DiagnoseOpenCudaDefault().ToJson();
#if STUDIOCAST_ENABLE_OPEN_VULKAN
  snapshot.open_vulkan =
      studiocast::vulkan::DiagnoseOpenVulkanDefault().ToJson();
#else
  snapshot.open_vulkan =
      "{\"compiled_enabled\":false,\"ok\":false,"
      "\"runtime_library_found\":false,\"runtime_library_path\":\"\","
      "\"instance_created\":false,\"physical_device_found\":false,"
      "\"non_cpu_device_selected\":false,"
      "\"compute_queue_available\":false,\"logical_device_created\":false,"
      "\"context_created\":false,\"context_healthy\":false,"
      "\"production_hardware_ready\":false,"
      "\"context_health\":\"uninitialized\","
      "\"context_failure_reason\":\"vulkan_backend_disabled_in_build\","
      "\"shader_pipeline_created\":false,\"api_version\":0,"
      "\"driver_version\":0,\"vendor_id\":0,\"device_id\":0,"
      "\"device_type\":0,\"vendor_name\":\"\",\"device_name\":\"\","
      "\"compute_queue_family_index\":0,\"error\":\"Open Vulkan backend is "
      "disabled in this build.\",\"fallback_reason\":\"disabled_in_build\","
      "\"blocked_reason\":\"disabled_in_build\","
      "\"degraded_reason\":\"Open Vulkan backend is disabled in this build.\","
      "\"installed_models\":[],\"default_model_id\":\"\",\"models\":[],"
      "\"missing_models\":{},\"available_effects\":[],"
      "\"blocked_effects\":{\"mirror\":\"vulkan_backend_disabled_in_build\","
      "\"virtual_background.blur\":\"vulkan_backend_disabled_in_build\","
      "\"virtual_background.remove\":\"vulkan_backend_disabled_in_build\","
      "\"virtual_background.replace\":\"vulkan_backend_disabled_in_build\","
      "\"auto_frame\":\"vulkan_backend_disabled_in_build\","
      "\"eye_contact\":\"open_vulkan_eye_contact_unavailable\","
      "\"video_noise_removal\":"
      "\"open_vulkan_video_noise_removal_unavailable\","
      "\"virtual_key_light\":\"vulkan_backend_disabled_in_build\","
      "\"vignette\":\"vulkan_backend_disabled_in_build\"},"
      "\"mirror_production_ready\":false,"
      "\"mirror_readiness_code\":\"\","
      "\"mirror_blocker_code\":\"vulkan_backend_disabled_in_build\","
      "\"vignette_fixed_center_production_ready\":false,"
      "\"vignette_readiness_code\":\"\","
      "\"vignette_blocker_code\":\"vulkan_backend_disabled_in_build\","
      "\"vignette_parameter_contract\":\"fixed_center\","
      "\"auto_frame_crop_stage_implemented\":false,"
      "\"auto_frame_production_ready\":false,"
      "\"auto_frame_readiness_code\":\"\","
      "\"auto_frame_blocker_code\":\"vulkan_backend_disabled_in_build\","
      "\"auto_frame_cpu_tail\":false,"
      "\"auto_frame_degraded_reason_code\":\"vulkan_effect_cpu_tail\","
      "\"auto_frame_selectable_cpu_fallback\":false,"
      "\"eye_contact_production_ready\":false,"
      "\"eye_contact_reason_code\":\"open_vulkan_eye_contact_unavailable\","
      "\"eye_contact_blocker_code\":"
      "\"open_vulkan_eye_contact_runtime_unavailable\","
      "\"eye_contact_detail\":\"no production eye-contact runtime can "
      "import StudioCast's exact Vulkan device, compute queue, and resident "
      "buffers; the current ONNX/dlib path uses CPU analysis, CPU tensors, "
      "and CPU postprocess, and its manifests do not declare a complete "
      "Vulkan artifact, gaze, or look-away contract\","
      "\"eye_contact_backend_compiled\":false,"
      "\"eye_contact_live_stage_implemented\":false,"
      "\"eye_contact_production_adapter_available\":false,"
      "\"eye_contact_vulkan_inference_provider_available\":false,"
      "\"eye_contact_non_cpu_device_selected\":false,"
      "\"eye_contact_compute_queue_available\":false,"
      "\"eye_contact_context_healthy\":false,"
      "\"eye_contact_shared_device_imported\":false,"
      "\"eye_contact_queue_ownership_explicit\":false,"
      "\"eye_contact_model_pack_selected\":false,"
      "\"eye_contact_artifact_contract_validated\":false,"
      "\"eye_contact_device_resident_analysis\":false,"
      "\"eye_contact_device_resident_tensor_io\":false,"
      "\"eye_contact_warmup_complete\":false,"
      "\"eye_contact_bounded_reusable_allocations\":false,"
      "\"eye_contact_synchronization_contract_validated\":false,"
      "\"eye_contact_parity_validated\":false,"
      "\"eye_contact_selectable_cpu_fallback\":false,"
      "\"eye_contact_dispatch_count\":0,"
      "\"eye_contact_cpu_readback_count\":0,"
      "\"eye_contact_cpu_fallback_count\":0,"
      "\"video_noise_removal_production_ready\":false,"
      "\"video_noise_removal_reason_code\":"
      "\"open_vulkan_video_noise_removal_unavailable\","
      "\"video_noise_removal_blocker_code\":"
      "\"open_vulkan_video_noise_removal_runtime_unavailable\","
      "\"video_noise_removal_detail\":\"no production video-denoise runtime "
      "can import StudioCast's exact Vulkan device, compute queue, and "
      "resident "
      "buffers; the current FastDVDnet ONNX path supports only CUDA/CPU "
      "providers with host temporal history and CPU "
      "preprocessing/postprocessing, "
      "while the available model packs declare ONNX-only artifacts\","
      "\"video_noise_removal_backend_compiled\":false,"
      "\"video_noise_removal_live_stage_implemented\":false,"
      "\"video_noise_removal_production_adapter_available\":false,"
      "\"video_noise_removal_vulkan_inference_provider_available\":false,"
      "\"video_noise_removal_non_cpu_device_selected\":false,"
      "\"video_noise_removal_compute_queue_available\":false,"
      "\"video_noise_removal_context_healthy\":false,"
      "\"video_noise_removal_shared_device_imported\":false,"
      "\"video_noise_removal_queue_ownership_explicit\":false,"
      "\"video_noise_removal_model_pack_selected\":false,"
      "\"video_noise_removal_artifact_contract_validated\":false,"
      "\"video_noise_removal_fully_device_resident_tensor_io\":false,"
      "\"video_noise_removal_device_resident_preprocess\":false,"
      "\"video_noise_removal_device_resident_postprocess\":false,"
      "\"video_noise_removal_warmup_complete\":false,"
      "\"video_noise_removal_synchronization_contract_validated\":false,"
      "\"video_noise_removal_bounded_reusable_allocations\":false,"
      "\"video_noise_removal_temporal_history_device_resident\":false,"
      "\"video_noise_removal_temporal_history_bounded\":false,"
      "\"video_noise_removal_history_reset_on_disable\":false,"
      "\"video_noise_removal_history_reset_on_reconfigure\":false,"
      "\"video_noise_removal_capture_sequence_discontinuity_reset\":false,"
      "\"video_noise_removal_parity_validated\":false,"
      "\"video_noise_removal_selectable_cpu_fallback\":false,"
      "\"video_noise_removal_dispatch_count\":0,"
      "\"video_noise_removal_temporal_history_reset_count\":0,"
      "\"video_noise_removal_cpu_readback_count\":0,"
      "\"video_noise_removal_cpu_fallback_count\":0,"
      "\"matting_runtime\":\"none\",\"matting_runtime_created\":false,"
      "\"matting_graph_loaded\":false,\"input_device_resident\":false,"
      "\"alpha_device_resident\":false,\"output_device_resident\":false,"
      "\"device_residency_mode\":\"disabled\",\"warnings\":[],"
      "\"install_hints\":[\"Rebuild with -DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON "
      "to enable this backend.\"]}";
#endif
  snapshot.open_audio =
      studiocast::open_audio::DiagnoseOpenAudioDefault().ToJson();
  snapshot.loopback = studiocast::video::ProbeLoopbackDiagnostics().ToJson();

  return snapshot;
}

DiagnosticsJsonSnapshot RefreshDiagnosticsJsonCache() {
  DiagnosticsJsonSnapshot snapshot = ComputeDiagnosticsJsonSnapshot();
  {
    std::lock_guard<std::mutex> lock(DiagnosticsJsonCacheMutex());
    DiagnosticsJsonCacheStorage() = snapshot;
  }
  return snapshot;
}

std::string
DiagnosticsJsonSnapshotToJson(const DiagnosticsJsonSnapshot &snapshot) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"engines\":{";
  oss << "\"maxine\":"
      << (snapshot.maxine.empty() ? std::string("{}") : snapshot.maxine) << ",";
  oss << "\"open_cuda\":"
      << (snapshot.open_cuda.empty() ? std::string("{}") : snapshot.open_cuda)
      << ",";
  oss << "\"open_vulkan\":"
      << (snapshot.open_vulkan.empty() ? std::string("{}")
                                       : snapshot.open_vulkan)
      << ",";
  oss << "\"open_audio\":"
      << (snapshot.open_audio.empty() ? std::string("{}")
                                      : snapshot.open_audio);
  oss << "},";
  oss << "\"maxine\":"
      << (snapshot.maxine.empty() ? std::string("{}") : snapshot.maxine) << ",";
  oss << "\"open_cuda\":"
      << (snapshot.open_cuda.empty() ? std::string("{}") : snapshot.open_cuda)
      << ",";
  oss << "\"open_vulkan\":"
      << (snapshot.open_vulkan.empty() ? std::string("{}")
                                       : snapshot.open_vulkan)
      << ",";
  oss << "\"open_audio\":"
      << (snapshot.open_audio.empty() ? std::string("{}") : snapshot.open_audio)
      << ",";
  oss << "\"virtual_device_diagnostics\":"
      << (snapshot.loopback.empty() ? std::string("{}") : snapshot.loopback);
  oss << "}";
  return oss.str();
}

using JsonObject = studiocast::util::json::Value::Object;
using JsonArray = studiocast::util::json::Value::Array;

const JsonObject *JsonObjectField(const JsonObject &obj,
                                  const std::string &key) {
  const auto it = obj.find(key);
  if (it == obj.end())
    return nullptr;
  return it->second.AsObject();
}

const std::string *JsonStringField(const JsonObject &obj,
                                   const std::string &key) {
  const auto it = obj.find(key);
  if (it == obj.end())
    return nullptr;
  return it->second.AsString();
}

bool JsonBoolField(const JsonObject &obj, const std::string &key,
                   bool fallback = false) {
  const auto it = obj.find(key);
  if (it == obj.end())
    return fallback;
  const bool *value = it->second.AsBool();
  return value ? *value : fallback;
}

double JsonNumberField(const JsonObject &obj, const std::string &key,
                       double fallback = 0.0) {
  const auto it = obj.find(key);
  if (it == obj.end())
    return fallback;
  const double *value = it->second.AsNumber();
  return value ? *value : fallback;
}

std::set<std::string> JsonStringArraySet(const JsonObject &obj,
                                         const std::string &key) {
  std::set<std::string> out;
  const auto it = obj.find(key);
  if (it == obj.end())
    return out;
  const JsonArray *array = it->second.AsArray();
  if (!array)
    return out;
  for (const auto &value : *array) {
    const std::string *s = value.AsString();
    if (s && !s->empty())
      out.insert(*s);
  }
  return out;
}

std::vector<std::string> JsonStringArrayVector(const JsonObject &obj,
                                               const std::string &key) {
  std::vector<std::string> out;
  const auto it = obj.find(key);
  if (it == obj.end())
    return out;
  const JsonArray *array = it->second.AsArray();
  if (!array)
    return out;
  for (const auto &value : *array) {
    const std::string *s = value.AsString();
    if (s && !s->empty())
      out.push_back(*s);
  }
  return out;
}

std::string JsonStringArrayJoined(const studiocast::util::json::Value &value) {
  if (const std::string *s = value.AsString())
    return *s;
  const JsonArray *array = value.AsArray();
  if (!array)
    return {};
  std::string out;
  for (const auto &entry : *array) {
    const std::string *s = entry.AsString();
    if (!s || s->empty())
      continue;
    if (!out.empty())
      out += "; ";
    out += *s;
  }
  return out;
}

std::map<std::string, std::string> JsonStringishMap(const JsonObject &obj,
                                                    const std::string &key) {
  std::map<std::string, std::string> out;
  const JsonObject *map = JsonObjectField(obj, key);
  if (!map)
    return out;
  for (const auto &kv : *map) {
    std::string value = JsonStringArrayJoined(kv.second);
    if (!value.empty())
      out[kv.first] = std::move(value);
  }
  return out;
}

struct EngineDiagnosticsSummary {
  bool present = false;
  bool compiled_enabled = false;
  bool ok = false;
  bool supported = false;
  std::string summary;
  std::string error;
  std::string blocked_reason;
  std::string fallback_reason;
  std::string degraded_reason;
  std::string default_model_id;
  std::vector<std::string> onnxruntime_providers;
  bool onnxruntime_cuda_provider_present = false;
  bool onnxruntime_tensorrt_provider_present = false;
  bool tensorrt_requested = false;
  bool tensorrt_available = false;
  std::string tensorrt_status;
  bool cuda_context_available = false;
  int cuda_device_count = -1;
  bool runtime_library_found = false;
  bool instance_created = false;
  bool physical_device_found = false;
  bool non_cpu_device_selected = false;
  bool cpu_device_selected = false;
  bool compute_queue_available = false;
  bool logical_device_created = false;
  bool context_created = false;
  bool context_healthy = false;
  bool production_hardware_ready = false;
  bool shader_pipeline_created = false;
  std::string context_failure_reason;
  std::uint32_t vendor_id = 0;
  std::uint32_t device_id = 0;
  std::string vendor_name;
  std::string device_name;
  std::string matting_runtime;
  std::string device_residency_mode;
  bool mirror_production_ready = false;
  std::string mirror_readiness_code;
  std::string mirror_blocker_code;
  bool vignette_fixed_center_production_ready = false;
  std::string vignette_readiness_code;
  std::string vignette_blocker_code;
  std::string vignette_parameter_contract;
  std::string eye_contact_blocker_code;
  std::string eye_contact_detail;
  std::string video_noise_removal_blocker_code;
  std::string video_noise_removal_detail;
  bool input_device_resident = false;
  bool alpha_device_resident = false;
  bool output_device_resident = false;
  std::set<std::string> installed_models;
  std::set<std::string> available_effects;
  std::map<std::string, std::string> blocked_effects;
  std::map<std::string, std::string> missing_models;
  std::map<std::string, std::string> missing_effects;
};

EngineDiagnosticsSummary
ParseEngineDiagnosticsSummary(const std::string &json) {
  EngineDiagnosticsSummary out;
  if (json.empty())
    return out;

  studiocast::util::json::Value root;
  std::string parseError;
  if (!studiocast::util::json::Parse(json, &root, &parseError))
    return out;
  const JsonObject *obj = root.AsObject();
  if (!obj)
    return out;

  out.present = true;
  out.compiled_enabled = JsonBoolField(*obj, "compiled_enabled", false);
  out.ok = JsonBoolField(*obj, "ok", false);
  out.supported = JsonBoolField(*obj, "supported", out.ok);
  if (const std::string *s = JsonStringField(*obj, "summary"))
    out.summary = *s;
  if (const std::string *s = JsonStringField(*obj, "error"))
    out.error = *s;
  if (const std::string *s = JsonStringField(*obj, "blocked_reason"))
    out.blocked_reason = *s;
  if (const std::string *s = JsonStringField(*obj, "fallback_reason"))
    out.fallback_reason = *s;
  if (const std::string *s = JsonStringField(*obj, "degraded_reason"))
    out.degraded_reason = *s;
  if (out.blocked_reason.empty())
    out.blocked_reason = !out.fallback_reason.empty() ? out.fallback_reason
                                                      : out.degraded_reason;
  if (const std::string *s = JsonStringField(*obj, "default_model_id"))
    out.default_model_id = *s;
  out.onnxruntime_providers = JsonStringArrayVector(*obj, "onnxruntime_providers");
  out.onnxruntime_cuda_provider_present =
      JsonBoolField(*obj, "onnxruntime_cuda_provider_present", false);
  out.onnxruntime_tensorrt_provider_present =
      JsonBoolField(*obj, "onnxruntime_tensorrt_provider_present", false);
  out.tensorrt_requested = JsonBoolField(*obj, "tensorrt_requested", false);
  out.tensorrt_available = JsonBoolField(*obj, "tensorrt_available", false);
  if (const std::string *s = JsonStringField(*obj, "tensorrt_status"))
    out.tensorrt_status = *s;
  out.cuda_context_available =
      JsonBoolField(*obj, "cuda_context_available", false);
  out.cuda_device_count =
      static_cast<int>(JsonNumberField(*obj, "cuda_device_count", -1.0));
  out.runtime_library_found =
      JsonBoolField(*obj, "runtime_library_found", false);
  out.instance_created = JsonBoolField(*obj, "instance_created", false);
  out.physical_device_found =
      JsonBoolField(*obj, "physical_device_found", false);
  out.non_cpu_device_selected =
      JsonBoolField(*obj, "non_cpu_device_selected", false);
  out.cpu_device_selected = JsonBoolField(*obj, "cpu_device_selected", false);
  out.compute_queue_available =
      JsonBoolField(*obj, "compute_queue_available", false);
  out.logical_device_created =
      JsonBoolField(*obj, "logical_device_created", false);
  out.context_created = JsonBoolField(*obj, "context_created", false);
  out.context_healthy = JsonBoolField(*obj, "context_healthy", false);
  out.production_hardware_ready =
      JsonBoolField(*obj, "production_hardware_ready", false);
  out.shader_pipeline_created =
      JsonBoolField(*obj, "shader_pipeline_created", false);
  if (const std::string *s = JsonStringField(*obj, "context_failure_reason"))
    out.context_failure_reason = *s;
  out.vendor_id =
      static_cast<std::uint32_t>(JsonNumberField(*obj, "vendor_id", 0.0));
  out.device_id =
      static_cast<std::uint32_t>(JsonNumberField(*obj, "device_id", 0.0));
  if (const std::string *s = JsonStringField(*obj, "vendor_name"))
    out.vendor_name = *s;
  if (const std::string *s = JsonStringField(*obj, "device_name"))
    out.device_name = *s;
  if (const std::string *s = JsonStringField(*obj, "matting_runtime"))
    out.matting_runtime = *s;
  if (const std::string *s = JsonStringField(*obj, "device_residency_mode"))
    out.device_residency_mode = *s;
  out.mirror_production_ready =
      JsonBoolField(*obj, "mirror_production_ready", false);
  if (const std::string *s = JsonStringField(*obj, "mirror_readiness_code"))
    out.mirror_readiness_code = *s;
  if (const std::string *s = JsonStringField(*obj, "mirror_blocker_code"))
    out.mirror_blocker_code = *s;
  out.vignette_fixed_center_production_ready =
      JsonBoolField(*obj, "vignette_fixed_center_production_ready", false);
  if (const std::string *s = JsonStringField(*obj, "vignette_readiness_code"))
    out.vignette_readiness_code = *s;
  if (const std::string *s = JsonStringField(*obj, "vignette_blocker_code"))
    out.vignette_blocker_code = *s;
  if (const std::string *s =
          JsonStringField(*obj, "vignette_parameter_contract"))
    out.vignette_parameter_contract = *s;
  if (const std::string *s =
          JsonStringField(*obj, "eye_contact_blocker_code"))
    out.eye_contact_blocker_code = *s;
  if (const std::string *s = JsonStringField(*obj, "eye_contact_detail"))
    out.eye_contact_detail = *s;
  if (const std::string *s =
          JsonStringField(*obj, "video_noise_removal_blocker_code"))
    out.video_noise_removal_blocker_code = *s;
  if (const std::string *s =
          JsonStringField(*obj, "video_noise_removal_detail"))
    out.video_noise_removal_detail = *s;
  out.input_device_resident =
      JsonBoolField(*obj, "input_device_resident", false);
  out.alpha_device_resident =
      JsonBoolField(*obj, "alpha_device_resident", false);
  out.output_device_resident =
      JsonBoolField(*obj, "output_device_resident", false);
  out.installed_models = JsonStringArraySet(*obj, "installed_models");
  out.available_effects = JsonStringArraySet(*obj, "available_effects");
  out.blocked_effects = JsonStringishMap(*obj, "blocked_effects");
  out.missing_models = JsonStringishMap(*obj, "missing_models");
  out.missing_effects = JsonStringishMap(*obj, "missing_effects");
  return out;
}

std::map<std::string, std::string>
ParseEffectBackendMap(const std::string &raw) {
  std::map<std::string, std::string> out;
  std::size_t pos = 0;
  while (pos < raw.size()) {
    const std::size_t comma = raw.find(',', pos);
    const std::string token = raw.substr(
        pos, comma == std::string::npos ? std::string::npos : comma - pos);
    const std::size_t colon = token.find(':');
    if (colon != std::string::npos) {
      const std::string id = TrimAscii(token.substr(0, colon));
      const std::string backend = TrimAscii(token.substr(colon + 1));
      if (!id.empty() && !backend.empty())
        out[id] = backend;
    }
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return out;
}

bool HasAuthoritativeRunningVideoFrame(
    const studiocast::video::VirtualCameraServiceStatus &status) {
  return status.pipeline.running && !status.pipeline.starting &&
         status.pipeline_state == "running";
}

const std::string &AuthoritativeLiveEffectBackends(
    const studiocast::video::VirtualCameraServiceStatus &status) {
  static const std::string empty;
  return HasAuthoritativeRunningVideoFrame(status)
             ? status.pipeline.effects_backends
             : empty;
}

std::string VideoEffectLabel(const std::string &id) {
  using namespace studiocast::video::effects::contract;
  if (id == kEffectIdMirror)
    return "Mirror";
  if (id == kEffectIdVirtualBackgroundBlur)
    return "Virtual background blur";
  if (id == kEffectIdVirtualBackgroundRemove)
    return "Virtual background removal";
  if (id == kEffectIdVirtualBackgroundReplace)
    return "Virtual background replacement";
  if (id == kEffectIdAutoFrame)
    return "Auto frame";
  if (id == kEffectIdEyeContact)
    return "Eye contact";
  if (id == kEffectIdVideoNoiseRemoval)
    return "Video noise removal";
  if (id == kEffectIdVirtualKeyLight)
    return "Virtual key light";
  if (id == kEffectIdVignette)
    return "Vignette";
  return id;
}

bool IsOpenCudaModelReadinessEffect(const std::string &id) {
  using namespace studiocast::video::effects::contract;
  return id == kEffectIdVirtualBackgroundBlur ||
         id == kEffectIdVirtualBackgroundRemove ||
         id == kEffectIdVirtualBackgroundReplace || id == kEffectIdAutoFrame ||
         id == kEffectIdVirtualKeyLight || id == kEffectIdEyeContact ||
         id == kEffectIdVideoNoiseRemoval;
}

std::string RequestedVideoModelId(
    const studiocast::video::effects::BroadcastCameraEffects &fx,
    const std::string &id) {
  using namespace studiocast::video::effects::contract;
  if (id == kEffectIdVirtualBackgroundBlur ||
      id == kEffectIdVirtualBackgroundRemove ||
      id == kEffectIdVirtualBackgroundReplace) {
    return fx.virtual_background.model_id;
  }
  if (id == kEffectIdAutoFrame)
    return fx.auto_frame.model_id;
  if (id == kEffectIdEyeContact)
    return fx.eye_contact.model_id;
  if (id == kEffectIdVideoNoiseRemoval)
    return fx.video_noise_removal.model_id;
  return {};
}

bool IsOpenVulkanBackend(const std::string &backend) {
  return backend == "open_vulkan" || backend == "vulkan";
}

bool IsActiveCudaOrMaxineVignetteBackend(const std::string &backend) {
  return backend == "cuda" || backend == "open_cuda" || backend == "maxine" ||
         backend == "maxine_ar" || backend == "maxine_ar_cuda";
}

std::string ExactOpenVulkanEffectBlocker(const EngineDiagnosticsSummary &diag,
                                         const std::string &id) {
  using namespace studiocast::video::effects::contract;
  if (id == kEffectIdMirror && !diag.mirror_blocker_code.empty())
    return diag.mirror_blocker_code;
  if (id == kEffectIdVignette && !diag.vignette_blocker_code.empty())
    return diag.vignette_blocker_code;
  const auto blocked = diag.blocked_effects.find(id);
  return blocked == diag.blocked_effects.end() ? std::string{}
                                               : blocked->second;
}

std::string
OpenVulkanCommonProductionBlocker(const EngineDiagnosticsSummary &diag,
                                  const std::string &id) {
  if (!diag.present)
    return "open_vulkan_runtime_diagnostics_unavailable";

  const std::string exactBlocker = ExactOpenVulkanEffectBlocker(diag, id);
  if (!diag.compiled_enabled)
    return exactBlocker.empty() ? "open_vulkan_runtime_evidence_inconsistent"
                                : exactBlocker;
  if (!diag.runtime_library_found)
    return exactBlocker.empty() ? "vulkan_runtime_not_found" : exactBlocker;
  if (!diag.instance_created)
    return "vulkan_instance_create_failed";
  if (!diag.physical_device_found)
    return exactBlocker.empty() ? "vulkan_no_physical_device" : exactBlocker;
  if (diag.cpu_device_selected)
    return "vulkan_only_cpu_devices_available";
  if (!diag.non_cpu_device_selected)
    return exactBlocker.empty() ? "vulkan_production_hardware_not_ready"
                                : exactBlocker;
  if (!diag.compute_queue_available)
    return exactBlocker.empty() ? "vulkan_no_compute_queue" : exactBlocker;
  if (!diag.logical_device_created)
    return exactBlocker.empty() ? "vulkan_device_create_failed" : exactBlocker;
  if (!diag.context_created)
    return !diag.context_failure_reason.empty()
               ? diag.context_failure_reason
               : (exactBlocker.empty() ? "vulkan_context_uninitialized"
                                       : exactBlocker);
  if (!diag.context_healthy)
    return !diag.context_failure_reason.empty()
               ? diag.context_failure_reason
               : (exactBlocker.empty() ? "vulkan_context_uninitialized"
                                       : exactBlocker);
  if (!diag.production_hardware_ready)
    return exactBlocker.empty() ? "vulkan_production_hardware_not_ready"
                                : exactBlocker;
  if (!diag.shader_pipeline_created || !diag.ok)
    return "open_vulkan_utility_kernels_unavailable";
  return {};
}

struct OpenVulkanEffectEvidence {
  bool ready = false;
  std::string blocker;
};

OpenVulkanEffectEvidence EvaluateOpenVulkanPixelEffectEvidence(
    const std::string &id,
    const studiocast::video::effects::BroadcastCameraEffects &fx,
    const EngineDiagnosticsSummary &diag) {
  using namespace studiocast::video::effects::contract;

  OpenVulkanEffectEvidence out;
  out.blocker = OpenVulkanCommonProductionBlocker(diag, id);
  if (!out.blocker.empty())
    return out;

  bool productionReady = false;
  std::string readinessCode;
  std::string expectedCode;
  std::string inconsistentCode;
  if (id == kEffectIdMirror) {
    productionReady = diag.mirror_production_ready;
    readinessCode = diag.mirror_readiness_code;
    expectedCode = "open_vulkan_mirror_production_ready";
    inconsistentCode = "open_vulkan_mirror_production_evidence_inconsistent";
  } else if (id == kEffectIdVignette) {
    if (fx.vignette.center_on_tracked_face && fx.auto_frame.enabled &&
        diag.available_effects.count(std::string(kEffectIdAutoFrame)) != 0) {
      out.blocker = "vulkan_vignette_tracked_center_not_supported";
      return out;
    }
    if (diag.vignette_parameter_contract != "fixed_center") {
      out.blocker = "open_vulkan_vignette_production_evidence_inconsistent";
      return out;
    }
    productionReady = diag.vignette_fixed_center_production_ready;
    readinessCode = diag.vignette_readiness_code;
    expectedCode = "open_vulkan_vignette_fixed_center_production_ready";
    inconsistentCode = "open_vulkan_vignette_production_evidence_inconsistent";
  } else {
    out.blocker = "open_vulkan_effect_evidence_not_defined";
    return out;
  }

  if (!productionReady) {
    out.blocker = ExactOpenVulkanEffectBlocker(diag, id);
    if (out.blocker.empty())
      out.blocker = inconsistentCode;
    return out;
  }
  if (readinessCode != expectedCode || diag.available_effects.count(id) == 0) {
    out.blocker = inconsistentCode;
    return out;
  }

  out.ready = true;
  return out;
}

std::string ConfiguredVirtualBackgroundEffectId(
    const studiocast::video::effects::BroadcastCameraEffects &fx) {
  using namespace studiocast::video::effects;
  using namespace studiocast::video::effects::contract;
  switch (fx.virtual_background.mode) {
  case VirtualBackgroundMode::blur:
    return std::string(kEffectIdVirtualBackgroundBlur);
  case VirtualBackgroundMode::remove:
    return std::string(kEffectIdVirtualBackgroundRemove);
  case VirtualBackgroundMode::replace:
    return std::string(kEffectIdVirtualBackgroundReplace);
  case VirtualBackgroundMode::none:
    break;
  }
  return {};
}

void AppendReadinessJson(std::ostringstream &oss, const std::string &state,
                         const std::string &summary,
                         const std::string &detail) {
  oss << "{";
  oss << "\"state\":\"" << JsonEscape(state) << "\",";
  oss << "\"summary\":\"" << JsonEscape(summary) << "\",";
  oss << "\"detail\":\"" << JsonEscape(detail) << "\"";
  oss << "}";
}

struct EndpointReadiness {
  std::string action;
  std::string state;
  std::string summary;
  std::string detail;
};

bool LooksLikeMissingModelError(const std::string &error) {
  if (error.empty())
    return false;
  const std::string lower = ToLowerAscii(error);
  return lower.find("model") != std::string::npos &&
         (lower.find("missing") != std::string::npos ||
          lower.find("not found") != std::string::npos ||
          lower.find("no open audio") != std::string::npos ||
          lower.find("no usable") != std::string::npos);
}

EndpointReadiness BuildMicrophoneEndpointReadiness(
    const studiocast::audio::VirtualAudioServiceStatus &ast,
    const studiocast::audio::VirtualAudioServiceConfig &acfg,
    const std::string &sourceError) {
  EndpointReadiness out;
  if (!sourceError.empty()) {
    out.action = "choose_microphone";
    out.state = "missing_physical_device";
    out.summary = "Choose a physical microphone input.";
    out.detail = sourceError;
    return out;
  }

  if (!ast.mic_present && acfg.create_virtual_mic) {
    out.action = "create_virtual_microphone";
    out.state = "missing_virtual_device";
    out.summary = "StudioCast Microphone needs setup.";
    out.detail = "The virtual microphone is not present.";
    return out;
  }

  if (!ast.last_error.empty()) {
    out.action = LooksLikeMissingModelError(ast.last_error)
                     ? "choose_open_audio_model"
                     : "retry_microphone";
    out.state = LooksLikeMissingModelError(ast.last_error)
                    ? "missing_model"
                    : "recoverable_error";
    out.summary = LooksLikeMissingModelError(ast.last_error)
                      ? "Microphone model is missing."
                      : "Microphone processing reported an error.";
    out.detail = ast.last_error;
    return out;
  }

  const std::string pipelineState = ToLowerAscii(ast.pipeline_state);
  if (ast.pipeline_running || ast.pipeline_starting ||
      pipelineState == "running" || pipelineState == "starting") {
    out.action = ast.pipeline_starting || pipelineState == "starting"
                     ? "wait"
                     : "stop_processing";
    out.state = "processing";
    out.summary = ast.pipeline_starting || pipelineState == "starting"
                      ? "Microphone processing is starting."
                      : "Microphone processing is active.";
    return out;
  }

  if (acfg.enabled && pipelineState == "idle_no_consumer") {
    out.action = "wait_for_app";
    out.state = "idle_no_consumer";
    out.summary = "Ready. Waiting for an app to use StudioCast Microphone.";
    out.detail = ast.pipeline_idle_reason;
    return out;
  }

  out.action = acfg.enabled ? "wait" : "enable_processing";
  out.state = "ready";
  out.summary = "StudioCast Microphone is present; processing is off.";
  return out;
}

EndpointReadiness BuildSpeakersEndpointReadiness(
    const studiocast::audio::VirtualAudioServiceStatus &ast,
    const studiocast::audio::VirtualAudioServiceConfig &acfg,
    const std::string &targetError) {
  EndpointReadiness out;
  if (!targetError.empty()) {
    out.action = "choose_speaker_output";
    out.state = "missing_physical_device";
    out.summary = "Choose a physical speaker output.";
    out.detail = targetError;
    return out;
  }

  const bool wantSpeakersDevice =
      acfg.create_virtual_speakers || acfg.speakers_enabled;
  if (!ast.speakers_present && wantSpeakersDevice) {
    out.action = "create_virtual_speakers";
    out.state = "missing_virtual_device";
    out.summary = "StudioCast Speakers need setup.";
    out.detail = "The virtual speakers device is not present.";
    return out;
  }

  const std::string lastError = !ast.speakers_last_error.empty()
                                    ? ast.speakers_last_error
                                    : ast.speakers_pipeline_last_error;
  if (!lastError.empty()) {
    out.action = LooksLikeMissingModelError(lastError)
                     ? "choose_open_audio_model"
                     : "retry_routing";
    out.state = LooksLikeMissingModelError(lastError) ? "missing_model"
                                                      : "recoverable_error";
    out.summary = LooksLikeMissingModelError(lastError)
                      ? "Speaker model is missing."
                      : "Speaker routing reported an error.";
    out.detail = lastError;
    return out;
  }

  const std::string routeMode = ToLowerAscii(ast.speakers_route_mode);
  const std::string pipelineState = ToLowerAscii(ast.speakers_pipeline_state);
  if (ast.speakers_routing_active || ast.speakers_pipeline_running ||
      pipelineState == "running") {
    out.action = "stop_routing";
    out.state = "processing";
    out.summary = routeMode == "pipeline" || ast.speakers_pipeline_running
                      ? "Processed speaker routing is active."
                      : "Speaker pass-through routing is active.";
    return out;
  }

  if (ast.speakers_pipeline_starting || pipelineState == "starting") {
    out.action = "wait";
    out.state = "processing";
    out.summary = "Speaker routing is starting.";
    return out;
  }

  if (acfg.speakers_enabled && routeMode == "pipeline" &&
      pipelineState == "idle_no_consumer") {
    out.action = "wait_for_app";
    out.state = "idle_no_consumer";
    out.summary = "Ready. Waiting for an app to use StudioCast Speakers.";
    out.detail = ast.speakers_pipeline_idle_reason;
    return out;
  }

  out.action = acfg.speakers_enabled ? "wait" : "start_routing";
  out.state = "ready";
  out.summary = "StudioCast Speakers are present; routing is off.";
  return out;
}

void AppendEndpointObject(std::ostringstream &oss,
                          const EndpointReadiness &readiness,
                          const std::string &activeDevice,
                          const std::string &activeBackend) {
  oss << "{";
  oss << "\"action\":\"" << JsonEscape(readiness.action) << "\",";
  oss << "\"readiness\":";
  AppendReadinessJson(oss, readiness.state, readiness.summary,
                      readiness.detail);
  oss << ",\"active_device\":\"" << JsonEscape(activeDevice) << "\",";
  oss << "\"active_backend\":\"" << JsonEscape(activeBackend) << "\"";
  oss << "}";
}

struct VideoEffectReadinessEntry {
  std::string id;
  std::string state;
  std::string summary;
  std::string detail;
  std::string backend;
  std::string requested_model_id;
  std::string resolved_model_id;
  std::string reason;
};

bool ReasonMeansMissingModel(const std::string &reason) {
  const std::string lower = ToLowerAscii(reason);
  return lower.find("missing_model") != std::string::npos ||
         lower.find("model_packs") != std::string::npos ||
         (lower.find("model") != std::string::npos &&
          (lower.find("missing") != std::string::npos ||
           lower.find("not found") != std::string::npos));
}

std::string ChooseVideoEffectBackend(
    const std::string &id,
    const studiocast::video::effects::BroadcastCameraEffects &fx,
    const std::map<std::string, std::string> &activeBackends,
    const EngineDiagnosticsSummary &maxineDiag,
    const EngineDiagnosticsSummary &openCudaDiag,
    const EngineDiagnosticsSummary &openVulkanDiag,
    studiocast::video::ComputeBackendPreference computePreference) {
  const auto activeIt = activeBackends.find(id);
  if (activeIt != activeBackends.end())
    return activeIt->second;

  if (computePreference == studiocast::video::ComputeBackendPreference::vulkan)
    return "open_vulkan";

  using namespace studiocast::video::effects::contract;
  if (id == kEffectIdMirror || id == kEffectIdVignette) {
    if (computePreference == studiocast::video::ComputeBackendPreference::cpu) {
      return "cpu";
    }
    if (computePreference ==
        studiocast::video::ComputeBackendPreference::cuda) {
      return "cuda";
    }
    return "auto";
  }

  using studiocast::video::effects::EffectsEnginePreference;
  if (fx.engine == EffectsEnginePreference::maxine)
    return "maxine";
  if (fx.engine == EffectsEnginePreference::open_cuda)
    return "open_cuda";

  if (maxineDiag.present && maxineDiag.available_effects.find(id) !=
                                maxineDiag.available_effects.end()) {
    return "maxine";
  }
  if (openCudaDiag.present && openCudaDiag.available_effects.find(id) !=
                                  openCudaDiag.available_effects.end()) {
    return "open_cuda";
  }
  if (openVulkanDiag.present && openVulkanDiag.available_effects.find(id) !=
                                    openVulkanDiag.available_effects.end()) {
    return "open_vulkan";
  }
  if (maxineDiag.present && (maxineDiag.ok || maxineDiag.supported))
    return "maxine";
  if (openCudaDiag.present && openCudaDiag.ok)
    return "open_cuda";
  return "auto";
}

VideoEffectReadinessEntry BuildVideoEffectReadinessEntry(
    const std::string &id,
    const studiocast::video::effects::BroadcastCameraEffects &fx,
    const studiocast::video::CameraPipelineStatus::DegradedEffect &degraded,
    const std::map<std::string, std::string> &activeBackends,
    const std::map<std::string, std::string> &ruleDisabled,
    const EngineDiagnosticsSummary &maxineDiag,
    const EngineDiagnosticsSummary &openCudaDiag,
    const EngineDiagnosticsSummary &openVulkanDiag,
    studiocast::video::ComputeBackendPreference computePreference,
    bool pipelineActive) {
  VideoEffectReadinessEntry out;
  out.id = id;
  out.backend =
      ChooseVideoEffectBackend(id, fx, activeBackends, maxineDiag,
                               openCudaDiag, openVulkanDiag, computePreference);
  out.requested_model_id = RequestedVideoModelId(fx, id);
  if (out.backend == "open_cuda" || out.backend == "open_vulkan")
    out.resolved_model_id = out.requested_model_id.empty()
                                ? (out.backend == "open_vulkan"
                                       ? openVulkanDiag.default_model_id
                                       : openCudaDiag.default_model_id)
                                : out.requested_model_id;

  const std::string label = VideoEffectLabel(id);

  const auto ruleIt = ruleDisabled.find(id);
  if (ruleIt != ruleDisabled.end()) {
    out.state = "blocked";
    out.summary = label + " is blocked by effect rules.";
    out.detail = ruleIt->second;
    out.reason = "effect_rule_blocked";
    return out;
  }

  if (degraded.active && degraded.effect_id == id) {
    out.backend = degraded.backend.empty() ? out.backend : degraded.backend;
    out.state = "recoverable_error";
    out.summary = label + " is temporarily degraded.";
    out.detail = degraded.reason;
    out.reason = degraded.state.empty() ? "degraded" : degraded.state;
    return out;
  }

  using namespace studiocast::video::effects::contract;
  const bool hasActiveBackend = activeBackends.find(id) != activeBackends.end();
  if (id == kEffectIdMirror && !IsOpenVulkanBackend(out.backend)) {
    out.state = "backend_unavailable";
    out.summary = label + " is unavailable.";
    out.detail = "Mirror has no selectable live path on backend " +
                 out.backend + ". Select the Open Vulkan compute backend.";
    out.reason = out.backend == "cpu" ? "mirror_no_selectable_cpu_path"
                                      : "mirror_no_live_backend_path";
    return out;
  }

  if (id == kEffectIdVignette && !IsOpenVulkanBackend(out.backend)) {
    if (hasActiveBackend && IsActiveCudaOrMaxineVignetteBackend(out.backend)) {
      out.state = "ready";
      out.summary = label + " is ready.";
      return out;
    }
    out.state = "backend_unavailable";
    out.summary = label + " is unavailable.";
    out.detail = out.backend == "cpu"
                     ? "Vignette has no selectable CPU live path."
                     : "No active CUDA or Maxine vignette path is proven by "
                       "the live pipeline.";
    out.reason = out.backend == "cpu"
                     ? "vignette_no_selectable_cpu_path"
                     : "vignette_active_backend_evidence_missing";
    return out;
  }

  const EngineDiagnosticsSummary *diag = nullptr;
  if (out.backend == "maxine")
    diag = &maxineDiag;
  else if (out.backend == "open_cuda" || out.backend == "open_video")
    diag = &openCudaDiag;
  else if (IsOpenVulkanBackend(out.backend))
    diag = &openVulkanDiag;

  if (!diag || !diag->present) {
    out.state = "unknown";
    out.summary = label + " readiness is unknown.";
    out.detail = out.backend == "auto"
                     ? "No effect backend diagnostics are available yet."
                     : out.backend + " diagnostics are not available yet.";
    out.reason = "diagnostics_unavailable";
    return out;
  }

  if (IsOpenVulkanBackend(out.backend) &&
      (id == kEffectIdMirror || id == kEffectIdVignette)) {
    if (pipelineActive && !hasActiveBackend) {
      out.state = "backend_unavailable";
      out.summary = label + " is not active in the live pipeline.";
      out.detail = "Exact Open Vulkan readiness exists, but the live pipeline "
                   "did not publish this effect in effects_backends.";
      out.reason = id == kEffectIdMirror
                       ? "open_vulkan_mirror_live_stage_not_active"
                       : "open_vulkan_vignette_live_stage_not_active";
      return out;
    }
    const OpenVulkanEffectEvidence evidence =
        EvaluateOpenVulkanPixelEffectEvidence(id, fx, *diag);
    if (evidence.ready) {
      out.state = "ready";
      out.summary = label + " is ready.";
      return out;
    }
    out.state = "backend_unavailable";
    out.summary = label + " is blocked by the selected backend.";
    out.detail = evidence.blocker;
    out.reason = evidence.blocker;
    return out;
  }

  const auto missingEffectIt = diag->missing_effects.find(id);
  if (missingEffectIt != diag->missing_effects.end()) {
    out.state = "backend_unavailable";
    out.summary = label + " is unavailable.";
    out.detail = missingEffectIt->second;
    out.reason = "missing_effect";
    return out;
  }

  const auto blockedIt = diag->blocked_effects.find(id);
  if (blockedIt != diag->blocked_effects.end()) {
    out.state = ReasonMeansMissingModel(blockedIt->second)
                    ? "missing_model"
                    : "backend_unavailable";
    out.summary = ReasonMeansMissingModel(blockedIt->second)
                      ? label + " model is missing."
                      : label + " is blocked by the selected backend.";
    out.detail = blockedIt->second;
    if (id == studiocast::video::effects::contract::kEffectIdEyeContact &&
        out.backend == "open_vulkan" &&
        !diag->eye_contact_blocker_code.empty()) {
      out.detail = "[" + blockedIt->second + "] [" +
                   diag->eye_contact_blocker_code + "]";
      if (!diag->eye_contact_detail.empty())
        out.detail += " " + diag->eye_contact_detail;
    }
    if (id ==
            studiocast::video::effects::contract::kEffectIdVideoNoiseRemoval &&
        out.backend == "open_vulkan" &&
        !diag->video_noise_removal_blocker_code.empty()) {
      out.detail = "[" + blockedIt->second + "] [" +
                   diag->video_noise_removal_blocker_code + "]";
      if (!diag->video_noise_removal_detail.empty())
        out.detail += " " + diag->video_noise_removal_detail;
    }
    out.reason = blockedIt->second;
    return out;
  }

  if ((out.backend == "open_cuda" || out.backend == "open_vulkan") &&
      IsOpenCudaModelReadinessEffect(id)) {
    if (!out.requested_model_id.empty()) {
      const auto missingModelIt =
          diag->missing_models.find(out.requested_model_id);
      if (missingModelIt != diag->missing_models.end()) {
        out.state = "missing_model";
        out.summary = label + " model is missing.";
        out.detail = missingModelIt->second;
        out.reason = "missing_model";
        return out;
      }
      if (!diag->installed_models.empty() &&
          diag->installed_models.find(out.requested_model_id) ==
              diag->installed_models.end()) {
        out.state = "missing_model";
        out.summary = label + " model is missing.";
        out.detail =
            "Configured model " + out.requested_model_id + " is not installed.";
        out.reason = "missing_model";
        return out;
      }
    } else if (diag->default_model_id.empty()) {
      out.state = "missing_model";
      out.summary = label + " model is missing.";
      out.detail = "No default Open Video model is installed.";
      out.reason = "missing_model";
      return out;
    }
  }

  if (out.backend == "open_vulkan" &&
      diag->available_effects.find(id) == diag->available_effects.end()) {
    out.state = "backend_unavailable";
    out.summary = label + " is blocked by the selected backend.";
    out.detail = !diag->blocked_reason.empty()
                     ? diag->blocked_reason
                     : "Open Vulkan does not provide this effect.";
    out.reason = !diag->blocked_reason.empty() ? diag->blocked_reason
                                               : "missing_effect";
    return out;
  }

  if ((diag->ok || diag->supported) ||
      diag->available_effects.find(id) != diag->available_effects.end()) {
    out.state = "ready";
    out.summary = label + " is ready.";
    return out;
  }

  out.state = "backend_unavailable";
  out.summary = label + " backend is unavailable.";
  out.detail = !diag->summary.empty() ? diag->summary : diag->blocked_reason;
  out.reason = !diag->blocked_reason.empty() ? diag->blocked_reason
                                             : "backend_unavailable";
  return out;
}

std::string VideoEffectReadinessToJson(
    const studiocast::video::VirtualCameraServiceStatus &st,
    const studiocast::video::VirtualCameraServiceConfig &cfg,
    const std::string &maxineJson, const std::string &openCudaJson,
    const std::string &openVulkanJson) {
  const auto &fx = cfg.pipeline.effects;
  const auto plan = studiocast::video::effects::BuildBroadcastEffectsPlan(fx);

  std::set<std::string> ids;
  for (const auto &id : plan.ordered_effect_ids)
    ids.insert(id);

  std::map<std::string, std::string> ruleDisabled;
  for (const auto &disabled : plan.disabled) {
    if (disabled.id.empty())
      continue;
    ids.insert(disabled.id);
    ruleDisabled[disabled.id] = disabled.reason;
  }

  if (fx.mirror)
    ids.insert(
        std::string(studiocast::video::effects::contract::kEffectIdMirror));
  if (const std::string vb = ConfiguredVirtualBackgroundEffectId(fx);
      !vb.empty())
    ids.insert(vb);
  if (fx.auto_frame.enabled)
    ids.insert(
        std::string(studiocast::video::effects::contract::kEffectIdAutoFrame));
  if (fx.eye_contact.enabled)
    ids.insert(
        std::string(studiocast::video::effects::contract::kEffectIdEyeContact));
  if (fx.video_noise_removal.enabled)
    ids.insert(std::string(
        studiocast::video::effects::contract::kEffectIdVideoNoiseRemoval));
  if (fx.virtual_key_light.enabled)
    ids.insert(std::string(
        studiocast::video::effects::contract::kEffectIdVirtualKeyLight));
  if (fx.vignette.enabled)
    ids.insert(
        std::string(studiocast::video::effects::contract::kEffectIdVignette));
  if (st.pipeline.degraded_effect.active &&
      !st.pipeline.degraded_effect.effect_id.empty())
    ids.insert(st.pipeline.degraded_effect.effect_id);

  const bool pipelineActive = HasAuthoritativeRunningVideoFrame(st);
  auto activeBackends =
      ParseEffectBackendMap(AuthoritativeLiveEffectBackends(st));
  const std::string vignetteId(
      studiocast::video::effects::contract::kEffectIdVignette);
  if (fx.vignette.enabled && activeBackends.count(vignetteId) == 0 &&
      !plan.vignette_attach_to_effect_id.empty()) {
    const auto attached =
        activeBackends.find(plan.vignette_attach_to_effect_id);
    if (attached != activeBackends.end() &&
        (attached->second == "maxine" || attached->second == "maxine_ar" ||
         attached->second == "maxine_ar_cuda")) {
      // The live Maxine path applies its attached vignette in the same GPU
      // stage. Inherit attribution only from that authoritative active map;
      // an availability probe or configured preference is not enough.
      activeBackends[vignetteId] = attached->second;
    }
  }
  const EngineDiagnosticsSummary maxineDiag =
      ParseEngineDiagnosticsSummary(maxineJson);
  const EngineDiagnosticsSummary openCudaDiag =
      ParseEngineDiagnosticsSummary(openCudaJson);
  const EngineDiagnosticsSummary openVulkanDiag =
      ParseEngineDiagnosticsSummary(openVulkanJson);

  std::ostringstream oss;
  oss << "{";
  bool first = true;
  for (const auto &id : ids) {
    const VideoEffectReadinessEntry entry = BuildVideoEffectReadinessEntry(
        id, fx, st.pipeline.degraded_effect, activeBackends, ruleDisabled,
        maxineDiag, openCudaDiag, openVulkanDiag, cfg.pipeline.compute_backend,
        pipelineActive);
    if (!first)
      oss << ",";
    first = false;
    oss << "\"" << JsonEscape(entry.id) << "\":{";
    oss << "\"state\":\"" << JsonEscape(entry.state) << "\",";
    oss << "\"summary\":\"" << JsonEscape(entry.summary) << "\",";
    oss << "\"detail\":\"" << JsonEscape(entry.detail) << "\",";
    oss << "\"backend\":\"" << JsonEscape(entry.backend) << "\",";
    oss << "\"requested_model_id\":\"" << JsonEscape(entry.requested_model_id)
        << "\",";
    oss << "\"resolved_model_id\":\"" << JsonEscape(entry.resolved_model_id)
        << "\",";
    oss << "\"reason\":\"" << JsonEscape(entry.reason) << "\"";
    oss << "}";
  }
  oss << "}";
  return oss.str();
}

bool VideoConfigRequestsCompute(
    const studiocast::video::effects::BroadcastCameraEffects &fx) {
  const auto plan = studiocast::video::effects::BuildBroadcastEffectsPlan(fx);
  return studiocast::video::effects::BroadcastEffectsPlanRequestsCompute(plan);
}

int ParseKeyLightTemperaturePreset(const std::string &raw, int fallback) {
  const auto v = ToLowerAscii(raw);
  if (v == "0" || v == "neutral")
    return 0;
  if (v == "1" || v == "warm")
    return 1;
  if (v == "2" || v == "cool")
    return 2;
  return fallback;
}

std::string FormatKeyLightTemperaturePreset(int preset) {
  switch (preset) {
  case 1:
    return "warm";
  case 2:
    return "cool";
  default:
    return "neutral";
  }
}

static double FpsToDouble(int fps, int fps_num, int fps_den) {
  // V4L2 streamparm uses time-per-frame (numerator/denominator).
  // We want frames-per-second for GUI/status consumers.
  if (fps > 0)
    return static_cast<double>(fps);
  if (fps_den > 0 && fps_num > 0) {
    // V4L2 uses time-per-frame as a rational: numerator/denominator seconds.
    // Convert to frames-per-second.
    return static_cast<double>(fps_den) / static_cast<double>(fps_num);
  }
  return 0.0;
}

static std::string
CaptureFormatToJson(const studiocast::video::CaptureFormat &f) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"pixfmt\":\"" << JsonEscape(f.pixfmt) << "\",";
  oss << "\"width\":" << f.width << ",";
  oss << "\"height\":" << f.height << ",";
  oss << "\"fps\":" << std::setprecision(6)
      << FpsToDouble(f.fps, f.fps_num, f.fps_den) << ",";
  oss << "\"fps_num\":" << f.fps_num << ",";
  oss << "\"fps_den\":" << f.fps_den << ",";
  oss << "\"bytesperline\":" << f.bytes_per_line << ",";
  oss << "\"sizeimage\":" << f.size_image;
  oss << "}";
  return oss.str();
}

static std::string
ActualFormatToJson(const studiocast::video::ActualFormat &f) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"pixfmt\":\"" << JsonEscape(f.pixfmt) << "\",";
  oss << "\"width\":" << f.width << ",";
  oss << "\"height\":" << f.height << ",";
  oss << "\"fps\":" << std::setprecision(6)
      << FpsToDouble(f.fps, f.fps_num, f.fps_den) << ",";
  oss << "\"fps_num\":" << f.fps_num << ",";
  oss << "\"fps_den\":" << f.fps_den << ",";
  oss << "\"bytesperline\":" << f.bytes_per_line << ",";
  oss << "\"sizeimage\":" << f.size_image;
  oss << "}";
  return oss.str();
}

void AppendJsonStringVector(std::ostringstream &oss,
                            const std::vector<std::string> &values) {
  oss << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i)
      oss << ",";
    oss << "\"" << JsonEscape(values[i]) << "\"";
  }
  oss << "]";
}

std::string DiagnosticUnavailableReason(const EngineDiagnosticsSummary &diag) {
  if (!diag.present || diag.ok || diag.supported)
    return {};
  if (!diag.blocked_reason.empty())
    return diag.blocked_reason;
  if (!diag.fallback_reason.empty())
    return diag.fallback_reason;
  if (!diag.degraded_reason.empty())
    return diag.degraded_reason;
  if (!diag.error.empty())
    return diag.error;
  if (!diag.blocked_effects.empty())
    return diag.blocked_effects.begin()->second;
  if (!diag.summary.empty())
    return diag.summary;
  return "diagnostics_unavailable";
}

std::string
OpenVulkanRequestedEffectBlockerDetail(const std::string &id,
                                       const std::string &blocker,
                                       const EngineDiagnosticsSummary &diag) {
  using namespace studiocast::video::effects::contract;
  if (id == kEffectIdEyeContact && !diag.eye_contact_blocker_code.empty()) {
    std::string detail =
        "[" + blocker + "] [" + diag.eye_contact_blocker_code + "]";
    if (!diag.eye_contact_detail.empty())
      detail += " " + diag.eye_contact_detail;
    return detail;
  }
  if (id == kEffectIdVideoNoiseRemoval &&
      !diag.video_noise_removal_blocker_code.empty()) {
    std::string detail =
        "[" + blocker + "] [" + diag.video_noise_removal_blocker_code + "]";
    if (!diag.video_noise_removal_detail.empty())
      detail += " " + diag.video_noise_removal_detail;
    return detail;
  }
  if (blocker == diag.blocked_reason && !diag.degraded_reason.empty())
    return "[" + blocker + "] " + diag.degraded_reason;
  return blocker;
}

struct RequestedOpenVulkanEvidence {
  bool any_ready = false;
  std::string first_ready_effect;
  std::string first_blocker;
};

RequestedOpenVulkanEvidence EvaluateRequestedOpenVulkanEvidence(
    const studiocast::video::effects::BroadcastCameraEffects &fx,
    const EngineDiagnosticsSummary &diag) {
  using namespace studiocast::video::effects::contract;
  const auto plan = studiocast::video::effects::BuildBroadcastEffectsPlan(fx);
  RequestedOpenVulkanEvidence out;
  for (const std::string &id : plan.ordered_effect_ids) {
    std::string blocker;
    if (id == kEffectIdMirror || id == kEffectIdVignette) {
      const OpenVulkanEffectEvidence evidence =
          EvaluateOpenVulkanPixelEffectEvidence(id, fx, diag);
      if (evidence.ready) {
        out.any_ready = true;
        if (out.first_ready_effect.empty())
          out.first_ready_effect = id;
        continue;
      }
      blocker = evidence.blocker;
    } else {
      const auto blocked = diag.blocked_effects.find(id);
      if (blocked != diag.blocked_effects.end()) {
        blocker = blocked->second;
      } else if (diag.available_effects.count(id) != 0) {
        out.any_ready = true;
        if (out.first_ready_effect.empty())
          out.first_ready_effect = id;
        continue;
      } else {
        blocker = OpenVulkanCommonProductionBlocker(diag, id);
        if (blocker.empty())
          blocker = "open_vulkan_requested_effect_unavailable:" + id;
      }
    }
    if (out.first_blocker.empty()) {
      out.first_blocker =
          OpenVulkanRequestedEffectBlockerDetail(id, blocker, diag);
    }
  }
  return out;
}

std::string OpenVulkanLiveStageNotActiveReason(const std::string &effectId) {
  using namespace studiocast::video::effects::contract;
  if (effectId == kEffectIdMirror)
    return "open_vulkan_mirror_live_stage_not_active";
  if (effectId == kEffectIdVignette)
    return "open_vulkan_vignette_live_stage_not_active";
  return "open_vulkan_requested_effect_live_stage_not_active:" + effectId;
}

std::string NormalizeComputeEngineName(const std::string &backend) {
  if (backend == "cuda" || backend == "open_cuda" ||
      backend == "open_video")
    return "open_cuda";
  if (backend == "vulkan" || backend == "open_vulkan")
    return "open_vulkan";
  if (backend == "maxine" || backend.rfind("maxine_", 0) == 0)
    return "maxine";
  if (backend == "cpu")
    return "cpu";
  return {};
}

std::string
ActiveComputeBackendFromEffectMap(const std::string &liveEffectBackends) {
  const auto activeBackends = ParseEffectBackendMap(liveEffectBackends);
  std::string candidate;
  for (const auto &[_, backend] : activeBackends) {
    const std::string normalized = NormalizeComputeEngineName(backend);
    if (normalized == "open_vulkan")
      return "vulkan";
    if (normalized == "maxine")
      candidate = "maxine";
    else if (normalized == "open_cuda" && candidate.empty())
      candidate = "cuda";
  }
  return candidate;
}

std::vector<std::string>
ActiveComputeEngines(const std::string &liveEffectBackends,
                     const std::string &computeActive) {
  std::set<std::string> engines;
  const auto activeBackends = ParseEffectBackendMap(liveEffectBackends);
  for (const auto &[_, backend] : activeBackends) {
    const std::string normalized = NormalizeComputeEngineName(backend);
    if (!normalized.empty())
      engines.insert(normalized);
  }
  if (engines.empty() && !computeActive.empty()) {
    const std::string normalized = NormalizeComputeEngineName(computeActive);
    if (!normalized.empty())
      engines.insert(normalized);
  }
  return std::vector<std::string>(engines.begin(), engines.end());
}

std::string ComputeFallbackCode(const std::string &preference,
                                const std::string &resolved,
                                const std::string &active) {
  if (preference == "vulkan" && active != "vulkan")
    return "vulkan_unavailable";
  if (preference == "cuda" && active != "cuda")
    return "cuda_unavailable";
  if (active == "cpu" && preference != "cpu")
    return "gpu_unavailable";
  if (active == resolved)
    return {};
  return "backend_changed";
}

struct ComputeProviderStatus {
  std::string mode;
  std::string active_provider;
  std::string device;
  std::string tensor_io_mode;
};

ComputeProviderStatus BuildComputeProviderStatus(
    const std::string &active,
    const std::vector<std::string> &activeEngines,
    const EngineDiagnosticsSummary &maxineDiag,
    const EngineDiagnosticsSummary &openCudaDiag,
    const EngineDiagnosticsSummary &openVulkanDiag) {
  const auto hasEngine = [&](const char *name) {
    return std::find(activeEngines.begin(), activeEngines.end(), name) !=
           activeEngines.end();
  };

  ComputeProviderStatus out;
  if (active == "vulkan" || hasEngine("open_vulkan")) {
    out.mode = "open_vulkan";
    out.active_provider = "Vulkan";
    out.device = openVulkanDiag.device_name.empty()
                     ? std::string("vulkan")
                     : openVulkanDiag.device_name;
    out.tensor_io_mode = openVulkanDiag.device_residency_mode.empty()
                             ? std::string("vulkan_device_kernels")
                             : openVulkanDiag.device_residency_mode;
    return out;
  }

  if (active == "maxine" || hasEngine("maxine")) {
    out.mode = "maxine";
    out.active_provider = "NVIDIA Maxine";
    out.device = maxineDiag.device_name.empty() ? std::string("cuda")
                                                : maxineDiag.device_name;
    out.tensor_io_mode = "cuda_device";
    return out;
  }

  if (active == "cuda" || hasEngine("open_cuda")) {
    out.mode = "open_cuda";
    if (openCudaDiag.tensorrt_requested && openCudaDiag.tensorrt_available) {
      out.active_provider = "TensorrtExecutionProvider";
      out.tensor_io_mode = "cuda_device_iobinding";
    } else if (openCudaDiag.onnxruntime_cuda_provider_present) {
      out.active_provider = "CUDAExecutionProvider";
      out.tensor_io_mode = "cuda_device_iobinding";
    } else {
      out.active_provider = "CUDA driver kernels";
      out.tensor_io_mode = "cuda_device_kernels";
    }
    out.device =
        openCudaDiag.cuda_device_count > 0 || openCudaDiag.cuda_context_available
            ? std::string("cuda:0")
            : std::string("cuda");
    return out;
  }

  if (active == "cpu" || hasEngine("cpu")) {
    out.mode = "cpu";
    out.active_provider = "CPU";
    out.device = "host";
    out.tensor_io_mode = "host";
  }

  return out;
}

void AppendCpuTailStages(std::ostringstream &oss,
                         const studiocast::video::CameraPipelineStatus &p) {
  std::vector<std::string> stages;
  const auto addIf = [&](const char *name, std::uint64_t count) {
    if (count > 0)
      stages.push_back(name);
  };
  addIf("key_light", p.open_cuda_transfers.cpu_tail_key_light_calls +
                         p.open_vulkan_transfers.cpu_tail_key_light_calls);
  addIf("auto_frame", p.open_cuda_transfers.cpu_tail_auto_frame_calls +
                          p.open_vulkan_transfers.cpu_tail_auto_frame_calls);
  addIf("auto_frame_face_tracking",
        p.open_cuda_transfers.cpu_tail_auto_frame_face_tracking_calls +
            p.open_vulkan_transfers.cpu_tail_auto_frame_face_tracking_calls);
  addIf("auto_frame_matte_tracking",
        p.open_cuda_transfers.cpu_tail_auto_frame_matte_tracking_calls +
            p.open_vulkan_transfers.cpu_tail_auto_frame_matte_tracking_calls);
  addIf("auto_frame_cpu_crop",
        p.open_cuda_transfers.cpu_tail_auto_frame_cpu_crop_calls +
            p.open_vulkan_transfers.cpu_tail_auto_frame_cpu_crop_calls);
  addIf("denoise", p.open_cuda_transfers.cpu_tail_denoise_calls);
  addIf("maxine_incompatible_stage",
        p.maxine_transfers.cpu_tail_stage_calls);
  AppendJsonStringVector(oss, stages);
}

void AppendVideoComputeStatusJson(
    std::ostringstream &oss,
    const studiocast::video::CameraPipelineStatus &pipeline,
    const std::string &liveEffectBackends, const std::string &preference,
    const std::string &resolved, const std::string &active,
    const std::string &fallbackReason, const std::string &degradedReason,
    const EngineDiagnosticsSummary &maxineDiag,
    const EngineDiagnosticsSummary &openCudaDiag,
    const EngineDiagnosticsSummary &openVulkanDiag) {
  const std::vector<std::string> activeEngines =
      ActiveComputeEngines(liveEffectBackends, active);
  std::string cudaUnavailable = DiagnosticUnavailableReason(openCudaDiag);
  std::string vulkanUnavailable = DiagnosticUnavailableReason(openVulkanDiag);
  std::string maxineUnavailable = DiagnosticUnavailableReason(maxineDiag);
  if (vulkanUnavailable.empty() && preference == "vulkan" &&
      active != "vulkan")
    vulkanUnavailable =
        !fallbackReason.empty() ? fallbackReason : degradedReason;
  if (cudaUnavailable.empty() && preference == "cuda" && active != "cuda")
    cudaUnavailable =
        !fallbackReason.empty() ? fallbackReason : degradedReason;

  // A resolved backend is a setup/preflight result.  It is not a fallback
  // merely because no pipeline is currently consuming it.
  const bool fallbackActive =
      !active.empty() &&
      (!fallbackReason.empty() ||
       (!resolved.empty() && resolved != active));
  const std::string fallbackCode =
      fallbackActive ? ComputeFallbackCode(preference, resolved, active)
                     : std::string();
  const ComputeProviderStatus provider = BuildComputeProviderStatus(
      active, activeEngines, maxineDiag, openCudaDiag, openVulkanDiag);

  const auto &cu = pipeline.open_cuda_transfers;
  const auto &vk = pipeline.open_vulkan_transfers;
  const auto &mx = pipeline.maxine_transfers;
  std::uint64_t maxineStageAttempts = 0;
  std::uint64_t maxineStageSuccesses = 0;
  for (std::size_t i = 0; i < mx.stage_attempts.size(); ++i) {
    maxineStageAttempts += mx.stage_attempts[i];
    maxineStageSuccesses += mx.stage_successes[i];
  }
  const std::uint64_t cpuTailStages =
      cu.cpu_tail_stage_calls + vk.cpu_tail_stage_calls +
      mx.cpu_tail_stage_calls;

  oss << "{";
  oss << "\"preference\":\"" << JsonEscape(preference) << "\",";
  oss << "\"resolved_backend\":\"" << JsonEscape(resolved) << "\",";
  oss << "\"active_backend\":\"" << JsonEscape(active) << "\",";
  oss << "\"fallback_reason\":\"" << JsonEscape(fallbackReason) << "\",";
  oss << "\"degraded_reason\":\"" << JsonEscape(degradedReason) << "\",";
  oss << "\"active_engines\":";
  AppendJsonStringVector(oss, activeEngines);
  oss << ",\"unavailable_reasons\":{";
  oss << "\"cuda\":\"" << JsonEscape(cudaUnavailable) << "\",";
  oss << "\"vulkan\":\"" << JsonEscape(vulkanUnavailable) << "\",";
  oss << "\"maxine\":\"" << JsonEscape(maxineUnavailable) << "\"";
  oss << "},";
  oss << "\"fallback\":{";
  oss << "\"active\":" << BoolJson(fallbackActive) << ",";
  oss << "\"from\":\"" << JsonEscape(preference) << "\",";
  oss << "\"to\":\""
      << JsonEscape(active.empty() ? resolved : active) << "\",";
  oss << "\"code\":\"" << JsonEscape(fallbackCode) << "\",";
  oss << "\"detail\":\""
      << JsonEscape(!fallbackReason.empty() ? fallbackReason : degradedReason)
      << "\"";
  oss << "},";
  oss << "\"provider\":{";
  oss << "\"mode\":\"" << JsonEscape(provider.mode) << "\",";
  oss << "\"active_provider\":\"" << JsonEscape(provider.active_provider)
      << "\",";
  oss << "\"device\":\"" << JsonEscape(provider.device) << "\",";
  oss << "\"tensor_io_mode\":\"" << JsonEscape(provider.tensor_io_mode)
      << "\"";
  oss << "},";
  oss << "\"cpu_tails\":{";
  oss << "\"active\":" << BoolJson(cpuTailStages > 0) << ",";
  oss << "\"stage_calls\":" << cpuTailStages << ",";
  oss << "\"stages\":";
  AppendCpuTailStages(oss, pipeline);
  oss << ",\"cuda_tensor_tail\":"
      << BoolJson(cu.cpu_tail_denoise_calls > 0) << ",";
  oss << "\"counts\":{";
  oss << "\"open_cuda\":" << cu.cpu_tail_stage_calls << ",";
  oss << "\"open_vulkan\":" << vk.cpu_tail_stage_calls << ",";
  oss << "\"maxine\":" << mx.cpu_tail_stage_calls << ",";
  oss << "\"denoise\":" << cu.cpu_tail_denoise_calls;
  oss << "}";
  oss << "},";
  oss << "\"transfers\":{";
  oss << "\"open_cuda\":{";
  oss << "\"active_frames\":" << cu.active_frames << ",";
  oss << "\"uploads\":" << cu.upload_calls << ",";
  oss << "\"downloads\":" << cu.download_calls << ",";
  oss << "\"final_downloads\":" << cu.final_download_calls << ",";
  oss << "\"cpu_continuation_downloads\":" << cu.cpu_continuation_download_calls
      << ",";
  oss << "\"alpha_downloads\":" << cu.alpha_download_calls << ",";
  oss << "\"matte_frame_uploads\":" << cu.matte_frame_upload_calls << ",";
  oss << "\"standalone_scaler_uploads\":" << cu.standalone_scaler_upload_calls
      << ",";
  oss << "\"standalone_scaler_downloads\":"
      << cu.standalone_scaler_download_calls << ",";
  oss << "\"denoise_tensor_uploads\":" << cu.denoise_tensor_upload_calls << ",";
  oss << "\"denoise_tensor_downloads\":" << cu.denoise_tensor_download_calls
      << ",";
  oss << "\"forced_syncs\":" << cu.forced_sync_calls;
  oss << "},";
  oss << "\"open_vulkan\":{";
  oss << "\"active_frames\":" << vk.active_frames << ",";
  oss << "\"uploads\":" << vk.upload_calls << ",";
  oss << "\"downloads\":" << vk.download_calls << ",";
  oss << "\"final_downloads\":" << vk.final_download_calls << ",";
  oss << "\"cpu_continuation_downloads\":" << vk.cpu_continuation_download_calls
      << ",";
  oss << "\"alpha_downloads\":" << vk.alpha_download_calls << ",";
  oss << "\"matte_frame_uploads\":" << vk.matte_frame_upload_calls << ",";
  oss << "\"background_uploads\":" << vk.background_upload_calls << ",";
  oss << "\"standalone_scaler_uploads\":" << vk.standalone_scaler_upload_calls
      << ",";
  oss << "\"standalone_scaler_downloads\":"
      << vk.standalone_scaler_download_calls << ",";
  oss << "\"dispatches\":"
      << (vk.preprocess_dispatch_calls + vk.alpha_resize_dispatch_calls +
          vk.blur_dispatch_calls + vk.composite_dispatch_calls +
          vk.key_light_dispatch_calls + vk.crop_resize_dispatch_calls)
      << ",";
  oss << "\"forced_syncs\":" << vk.forced_sync_calls << ",";
  oss << "\"fallback_frames\":" << vk.fallback_frames << ",";
  oss << "\"runtime_failure_frames\":" << vk.runtime_failure_frames;
  oss << "},";
  oss << "\"maxine\":{";
  oss << "\"active_frames\":" << mx.active_frames << ",";
  oss << "\"upload_attempts\":" << mx.upload_attempts << ",";
  oss << "\"uploads\":" << mx.upload_calls << ",";
  oss << "\"downloads\":" << mx.download_calls << ",";
  oss << "\"final_download_attempts\":" << mx.final_download_attempts << ",";
  oss << "\"final_downloads\":" << mx.final_download_calls << ",";
  oss << "\"cpu_continuation_download_attempts\":"
      << mx.cpu_continuation_download_attempts << ",";
  oss << "\"cpu_continuation_downloads\":" << mx.cpu_continuation_download_calls
      << ",";
  oss << "\"device_bridge_attempts\":" << mx.device_bridge_attempts << ",";
  oss << "\"device_bridges\":" << mx.device_bridge_calls << ",";
  oss << "\"background_setup_upload_attempts\":"
      << mx.background_setup_upload_attempts << ",";
  oss << "\"background_setup_uploads\":"
      << mx.background_setup_upload_calls << ",";
  oss << "\"standalone_scaler_uploads\":" << mx.standalone_scaler_upload_calls
      << ",";
  oss << "\"standalone_scaler_downloads\":"
      << mx.standalone_scaler_download_calls << ",";
  oss << "\"matte_inference_attempts\":" << mx.matte_inference_attempts
      << ",";
  oss << "\"matte_inferences\":" << mx.green_screen_calls << ",";
  oss << "\"stage_attempts\":" << maxineStageAttempts << ",";
  oss << "\"stage_successes\":" << maxineStageSuccesses << ",";
  oss << "\"forced_sync_attempts\":" << mx.forced_sync_attempts << ",";
  oss << "\"forced_syncs\":" << mx.forced_sync_calls << ",";
  oss << "\"composite_attempts\":" << mx.composite_attempts << ",";
  oss << "\"composites\":" << mx.composite_calls << ",";
  oss << "\"synchronous_sdk_run_attempts\":"
      << mx.synchronous_sdk_run_attempts << ",";
  oss << "\"synchronous_sdk_runs\":" << mx.synchronous_sdk_run_calls
      << ",";
  oss << "\"asynchronous_sdk_run_attempts\":"
      << mx.asynchronous_sdk_run_attempts << ",";
  oss << "\"asynchronous_sdk_runs\":" << mx.asynchronous_sdk_run_calls
      << ",";
  oss << "\"setup_attempts\":" << mx.setup_attempts << ",";
  oss << "\"setup_successes\":" << mx.setup_successes << ",";
  oss << "\"cpu_tail_stage_calls\":" << mx.cpu_tail_stage_calls << ",";
  oss << "\"runtime_failure_frames\":" << mx.runtime_failure_frames << ",";
  oss << "\"deferred_readbacks\":" << mx.deferred_readbacks;
  oss << "}";
  oss << "}";
  oss << "}";
}

std::string
StatusToJson(const studiocast::video::VirtualCameraServiceStatus &st,
             const studiocast::video::VirtualCameraServiceConfig &cfg,
             const studiocast::audio::VirtualAudioServiceStatus &ast,
             const studiocast::audio::VirtualAudioServiceConfig &acfg,
             const std::filesystem::path &socketPath,
             const std::string &maxineJson, const std::string &openCudaJson,
             const std::string &openVulkanJson,
             const std::string &openAudioJson,
             const std::string &loopbackJson,
             const studiocast::config::DaemonConfig *daemonConfig = nullptr) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"version\":\"" << JsonEscape(STUDIOCAST_VERSION) << "\",";
  oss << "\"git_sha\":\"" << JsonEscape(STUDIOCAST_GIT_SHA) << "\",";
  oss << "\"socket\":\"" << JsonEscape(socketPath.string()) << "\",";
  oss << "\"service_running\":" << BoolJson(st.service_running) << ",";

  // Global Maxine diagnostics payload (used by GUI/CLI to disable unsupported
  // effects).
  if (!maxineJson.empty()) {
    oss << "\"maxine\":" << maxineJson << ",";
  }

  // Engine diagnostics (preferred scalable shape). Keep top-level `maxine` for
  // compatibility.
  oss << "\"engines\":{";
  if (!maxineJson.empty()) {
    oss << "\"maxine\":" << maxineJson << ",";
  }
  oss << "\"open_cuda\":"
      << (openCudaJson.empty() ? std::string("{}") : openCudaJson);
  oss << ",\"open_vulkan\":"
      << (openVulkanJson.empty() ? std::string("{}") : openVulkanJson);
  oss << ",\"open_audio\":"
      << (openAudioJson.empty() ? std::string("{}") : openAudioJson);
  oss << "},";

  // Convenience top-level alias.
  if (!openCudaJson.empty()) {
    oss << "\"open_cuda\":" << openCudaJson << ",";
  }

  if (!openVulkanJson.empty()) {
    oss << "\"open_vulkan\":" << openVulkanJson << ",";
  }

  if (!openAudioJson.empty()) {
    oss << "\"open_audio\":" << openAudioJson << ",";
  }

  const std::string effectiveInputDevice =
      st.pipeline.input_device.empty() ? cfg.pipeline.input_device
                                       : st.pipeline.input_device;
  const std::string effectiveOutputDevice =
      st.pipeline.output_device.empty() ? cfg.pipeline.output_device
                                        : st.pipeline.output_device;
  const bool computeRequested =
      VideoConfigRequestsCompute(cfg.pipeline.effects);
  const std::string computePreference =
      studiocast::video::ComputeBackendPreferenceToString(
          cfg.pipeline.compute_backend);
  const EngineDiagnosticsSummary computeMaxineDiag =
      ParseEngineDiagnosticsSummary(maxineJson);
  const EngineDiagnosticsSummary computeOpenCudaDiag =
      ParseEngineDiagnosticsSummary(openCudaJson);
  const EngineDiagnosticsSummary computeOpenVulkanDiag =
      ParseEngineDiagnosticsSummary(openVulkanJson);
  std::string computeResolved = st.pipeline.compute_backend_resolved.empty()
                                    ? std::string("cpu")
                                    : st.pipeline.compute_backend_resolved;
  const bool pipelineHasLiveFrame = HasAuthoritativeRunningVideoFrame(st);
  const std::string &liveEffectBackends = AuthoritativeLiveEffectBackends(st);
  // Active means executing in a successfully written running frame. Aggregate
  // setup fields and maps from any other lifecycle state are not live evidence.
  std::string computeActive = pipelineHasLiveFrame
                                  ? (st.pipeline.compute_backend_active.empty()
                                         ? std::string("cpu")
                                         : st.pipeline.compute_backend_active)
                                  : std::string{};
  std::string computeFallback = st.pipeline.compute_backend_fallback_reason;
  std::string computeDegraded = st.pipeline.compute_backend_degraded_reason;
  const std::string mappedActive =
      ActiveComputeBackendFromEffectMap(liveEffectBackends);
  if (!mappedActive.empty()) {
    // Per-effect runtime attribution is the most specific live evidence. It
    // also repairs stale/default aggregate fields without guessing from a
    // configured preference.
    computeActive = mappedActive;
    computeResolved = mappedActive == "maxine" ? "cuda" : mappedActive;
    if ((cfg.pipeline.compute_backend ==
             studiocast::video::ComputeBackendPreference::vulkan &&
         mappedActive == "vulkan") ||
        (cfg.pipeline.compute_backend ==
             studiocast::video::ComputeBackendPreference::cuda &&
         (mappedActive == "cuda" || mappedActive == "maxine"))) {
      computeFallback.clear();
      computeDegraded.clear();
    }
  } else if (computeRequested && pipelineHasLiveFrame) {
    // A running pipeline without per-effect attribution has not proven that
    // its configured aggregate GPU backend executed the requested effect.
    // Report the actual pass-through/CPU execution until effects_backends
    // supplies the authoritative live mapping.
    computeActive = "cpu";
  } else if (computeRequested && !pipelineHasLiveFrame &&
             cfg.pipeline.compute_backend ==
                 studiocast::video::ComputeBackendPreference::vulkan) {
    // Idle, waiting, starting, and failed-start pipelines have no active
    // backend. Preflight only the exact requested effects: one retained ready
    // effect is enough to resolve Vulkan, while a blocked sibling remains
    // blocked in effect_readiness.
    const RequestedOpenVulkanEvidence evidence =
        EvaluateRequestedOpenVulkanEvidence(cfg.pipeline.effects,
                                            computeOpenVulkanDiag);
    if (evidence.any_ready) {
      computeResolved = "vulkan";
      const bool retainFailedStartReason =
          st.pipeline_state == "backing_off" ||
          st.last_transition == "start_failed" ||
          (st.pipeline_start_failures > 0 &&
           !st.pipeline.last_error.empty());
      if (!retainFailedStartReason) {
        computeFallback.clear();
        computeDegraded.clear();
      }
    } else if (!evidence.first_blocker.empty()) {
      computeResolved = "cpu";
      if (computeFallback.empty())
        computeFallback = evidence.first_blocker;
      if (computeDegraded.empty())
        computeDegraded = computeFallback;
    }
  }
  if (computeRequested && computeActive == "cpu") {
    if (cfg.pipeline.compute_backend ==
        studiocast::video::ComputeBackendPreference::cpu) {
      computeResolved = "cpu";
      computeActive = "cpu";
      if (computeDegraded.empty()) {
        computeDegraded = "CPU compute backend selected; GPU-backed video "
                          "effects are disabled.";
      }
    } else if (cfg.pipeline.compute_backend ==
               studiocast::video::ComputeBackendPreference::vulkan) {
      computeResolved = "cpu";
      computeActive = "cpu";
      if (computeFallback.empty()) {
        const RequestedOpenVulkanEvidence evidence =
            EvaluateRequestedOpenVulkanEvidence(cfg.pipeline.effects,
                                                computeOpenVulkanDiag);
        if (evidence.any_ready && !evidence.first_ready_effect.empty()) {
          computeFallback =
              OpenVulkanLiveStageNotActiveReason(evidence.first_ready_effect);
        } else {
          computeFallback = evidence.first_blocker;
        }
        if (computeFallback.empty()) {
          computeFallback =
              "Vulkan compute backend requested, but Vulkan compute is not "
              "available in this build.";
        }
      }
      if (computeDegraded.empty())
        computeDegraded = computeFallback;
    } else if (cfg.pipeline.compute_backend ==
                   studiocast::video::ComputeBackendPreference::auto_select &&
               (st.pipeline.running || st.pipeline.starting ||
                st.pipeline_active_needed)) {
      if (computeFallback.empty()) {
        computeFallback =
            "No GPU compute backend is active; using CPU/pass-through "
            "fallback.";
      }
      if (computeDegraded.empty())
        computeDegraded = computeFallback;
    }
  }

  oss << "\"video\":{";
  oss << "\"enabled\":" << BoolJson(cfg.enabled) << ",";
  oss << "\"always_on\":" << BoolJson(cfg.always_on) << ",";
  oss << "\"allow_cpu_resize\":" << BoolJson(cfg.pipeline.allow_cpu_resize)
      << ",";
  oss << "\"vulkan_adapter\":{";
  oss << "\"configured_device\":\""
      << JsonEscape(daemonConfig ? daemonConfig->video_vulkan_device : "auto")
      << "\",";
  oss << "\"allow_cpu\":"
      << BoolJson(daemonConfig && daemonConfig->video_vulkan_allow_cpu) << ",";
  oss << "\"apply_policy\":\"next_vulkan_device_initialization\"";
  oss << "},";
  oss << "\"compute\":";
  AppendVideoComputeStatusJson(
      oss, st.pipeline, liveEffectBackends, computePreference, computeResolved,
      computeActive, computeFallback, computeDegraded, computeMaxineDiag,
      computeOpenCudaDiag, computeOpenVulkanDiag);
  oss << ",";
  oss << "\"output_format_requested\":\""
      << JsonEscape(
             studiocast::video::PixelFormatName(cfg.pipeline.output_format))
      << "\",";
  oss << "\"virtual_device_present\":" << BoolJson(st.virtual_device_present)
      << ",";
  oss << "\"virtual_device_available\":"
      << BoolJson(st.virtual_device_available) << ",";
  oss << "\"virtual_device_error\":\"" << JsonEscape(st.virtual_device_error)
      << "\",";
  oss << "\"consumer_present\":" << BoolJson(st.consumer_present) << ",";
  oss << "\"consumer_count\":" << st.consumer_count << ",";
  oss << "\"consumer_error\":\"" << JsonEscape(st.consumer_error) << "\",";

  // Supervisor diagnostics (Phase 2): helps diagnose consumer-driven camera
  // probing/flapping behavior in apps like Discord.
  oss << "\"supervisor\":{";
  oss << "\"consumer_poll_ms\":" << cfg.consumer_poll_ms << ",";
  oss << "\"start_grace_ms\":" << cfg.start_grace_ms << ",";
  oss << "\"stop_grace_ms\":" << cfg.stop_grace_ms << ",";
  oss << "\"min_run_ms\":" << cfg.min_run_ms << ",";
  oss << "\"pipeline_start_attempts\":" << st.pipeline_start_attempts << ",";
  oss << "\"pipeline_starts\":" << st.pipeline_starts << ",";
  oss << "\"pipeline_start_failures\":" << st.pipeline_start_failures << ",";
  oss << "\"pipeline_stops\":" << st.pipeline_stops << ",";
  oss << "\"pipeline_config_restarts\":" << st.pipeline_config_restarts << ",";
  oss << "\"stabilizing\":" << BoolJson(st.stabilizing) << ",";
  oss << "\"thrash_events_10s\":" << st.thrash_events_10s << ",";
  oss << "\"last_transition\":\"" << JsonEscape(st.last_transition) << "\",";
  oss << "\"last_transition_ms_ago\":" << st.last_transition_ms_ago << ",";
  oss << "\"next_start_retry_ms\":" << st.next_start_retry_ms;
  oss << "},";

  oss << "\"input_device\":\"" << JsonEscape(effectiveInputDevice) << "\",";
  oss << "\"output_device\":\"" << JsonEscape(effectiveOutputDevice)
      << "\",";
  if (!loopbackJson.empty()) {
    oss << "\"virtual_device_diagnostics\":" << loopbackJson << ",";
  }

  // Negotiated formats (what the driver actually gave us / accepted).
  oss << "\"capture_format\":" << CaptureFormatToJson(st.pipeline.capture)
      << ",";
  oss << "\"capture_fallback\":{";
  oss << "\"state\":\""
      << JsonEscape(st.pipeline.capture_fallback_state.empty()
                        ? std::string("none")
                        : st.pipeline.capture_fallback_state)
      << "\",";
  oss << "\"reason\":\""
      << JsonEscape(st.pipeline.capture_fallback_reason) << "\"";
  oss << "},";
  oss << "\"output_format\":" << ActualFormatToJson(st.pipeline.output) << ",";

  // Scaling status (active backend + from/to formats).
  oss << "\"scaling\":{";
  oss << "\"backend_active\":\""
      << JsonEscape(st.pipeline.scaling_backend_active) << "\",";
  oss << "\"from\":" << CaptureFormatToJson(st.pipeline.scaling_from) << ",";
  oss << "\"to\":" << ActualFormatToJson(st.pipeline.scaling_to);
  oss << "},";

  const char *capture_mode_label =
      (cfg.pipeline.capture_mode == studiocast::video::CaptureMode::auto_best)
          ? "auto_best"
          : "requested";
  oss << "\"capture_mode\":\"" << capture_mode_label << "\",";

  oss << "\"width\":" << cfg.pipeline.width << ",";
  oss << "\"height\":" << cfg.pipeline.height << ",";
  oss << "\"fps\":" << cfg.pipeline.fps << ",";
  oss << "\"mirror\":" << BoolJson(cfg.pipeline.effects.mirror) << ",";
  // Legacy flat fields (kept for compatibility): derived from the canonical
  // Broadcast schema.
  oss << "\"background\":\""
      << JsonEscape(studiocast::video::effects::ToString(
             cfg.pipeline.effects.virtual_background.mode))
      << "\",";
  oss << "\"background_backend\":\""
      << JsonEscape(
             studiocast::video::effects::ToString(cfg.pipeline.effects.engine))
      << "\",";
  oss << "\"background_strength\":"
      << cfg.pipeline.effects.virtual_background.strength << ",";
  oss << "\"background_remove_color\":\""
      << JsonEscape(cfg.pipeline.effects.virtual_background.remove_color)
      << "\",";
  oss << "\"background_replace_image\":\""
      << JsonEscape(cfg.pipeline.effects.virtual_background.replace_path)
      << "\",";

  // Canonical effect model (Broadcast schema) for GUI/CLI.
  oss << "\"video_effects\":"
      << studiocast::video::BroadcastCameraEffectsContractToJson(
             cfg.pipeline.effects)
      << ",";
  oss << "\"effect_readiness\":"
      << VideoEffectReadinessToJson(st, cfg, maxineJson, openCudaJson,
                                    openVulkanJson)
      << ",";

  const int vkl_intensity = std::max(
      0, std::min(100, cfg.pipeline.effects.virtual_key_light.intensity));
  oss << "\"virtual_key_light\":"
      << BoolJson(cfg.pipeline.effects.virtual_key_light.enabled) << ",";
  oss << "\"virtual_key_light_intensity\":" << vkl_intensity << ",";
  oss << "\"virtual_key_light_temperature\":\""
      << JsonEscape(FormatKeyLightTemperaturePreset(
             cfg.pipeline.effects.virtual_key_light.temperature_preset))
      << "\",";
  oss << "\"virtual_key_light_pan\":"
      << cfg.pipeline.effects.virtual_key_light.direction_pan_degrees << ",";
  oss << "\"virtual_key_light_hdri\":\""
      << JsonEscape(cfg.pipeline.effects.virtual_key_light.hdri_path) << "\",";

  oss << "\"pipeline\":{";
  oss << "\"running\":" << BoolJson(st.pipeline.running) << ",";
  oss << "\"starting\":" << BoolJson(st.pipeline.starting) << ",";
  oss << "\"active_needed\":" << BoolJson(st.pipeline_active_needed) << ",";
  oss << "\"state\":\""
      << JsonEscape(st.pipeline_state.empty() ? std::string("disabled")
                                              : st.pipeline_state)
      << "\",";
  oss << "\"idle_reason\":\"" << JsonEscape(st.pipeline_idle_reason) << "\",";
  oss << "\"frame_index\":" << st.pipeline.frame_index << ",";
  oss << "\"effects_backends\":\"" << JsonEscape(liveEffectBackends) << "\",";
  oss << "\"effects_note\":\"" << JsonEscape(st.pipeline.effects_note) << "\",";
  if (st.pipeline.degraded_effect.active) {
    const auto &fx = st.pipeline.degraded_effect;
    oss << "\"degraded_effect\":{";
    oss << "\"id\":\"" << JsonEscape(fx.effect_id) << "\",";
    oss << "\"backend\":\"" << JsonEscape(fx.backend) << "\",";
    oss << "\"reason\":\"" << JsonEscape(fx.reason) << "\",";
    oss << "\"state\":\"" << JsonEscape(fx.state) << "\",";
    oss << "\"failure_count\":" << fx.failure_count << ",";
    oss << "\"cooldown_frames\":" << fx.cooldown_frames;
    oss << "},";
  }

  // Lightweight rolling perf counters for quick CPU vs GPU scaling comparisons.
  oss << "\"fps_actual\":" << std::setprecision(6) << st.pipeline.fps_actual
      << ",";
  oss << "\"ms_per_frame\":{";
  oss << "\"capture\":" << std::setprecision(6)
      << st.pipeline.ms_per_frame.capture << ",";
  oss << "\"scale\":" << std::setprecision(6) << st.pipeline.ms_per_frame.scale
      << ",";
  oss << "\"effects\":" << std::setprecision(6)
      << st.pipeline.ms_per_frame.effects << ",";
  oss << "\"write\":" << std::setprecision(6) << st.pipeline.ms_per_frame.write;
  oss << "},";
  oss << "\"perf_sample_frames\":" << st.pipeline.perf_sample_frames;

  const bool debug_open_cuda_transfers =
      (std::getenv("STUDIOCAST_DEBUG_OPEN_CUDA_TRANSFERS") != nullptr) ||
      (std::getenv("STUDIOCAST_DEBUG_CUDA_UPLOADS") != nullptr);
  const bool debug_open_vulkan_transfers =
      std::getenv("STUDIOCAST_DEBUG_OPEN_VULKAN_TRANSFERS") != nullptr;

  // Open CUDA transfer counters are a focused debug surface that can be enabled
  // without turning on the broader STUDIOCAST_DEBUG_VIDEO_STATS payload.
  if (debug_open_cuda_transfers) {
    oss << ",\"open_cuda_transfers\":{";
    oss << "\"active_frames\":" << st.pipeline.open_cuda_transfers.active_frames
        << ",";
    oss << "\"upload_calls\":" << st.pipeline.open_cuda_transfers.upload_calls
        << ",";
    oss << "\"download_calls\":"
        << st.pipeline.open_cuda_transfers.download_calls << ",";
    oss << "\"final_download_calls\":"
        << st.pipeline.open_cuda_transfers.final_download_calls << ",";
    oss << "\"cpu_continuation_download_calls\":"
        << st.pipeline.open_cuda_transfers.cpu_continuation_download_calls
        << ",";
    oss << "\"alpha_download_calls\":"
        << st.pipeline.open_cuda_transfers.alpha_download_calls << ",";
    oss << "\"matte_frame_upload_calls\":"
        << st.pipeline.open_cuda_transfers.matte_frame_upload_calls << ",";
    oss << "\"matting_inference_calls\":"
        << st.pipeline.open_cuda_transfers.matting_inference_calls << ",";
    oss << "\"standalone_scaler_upload_calls\":"
        << st.pipeline.open_cuda_transfers.standalone_scaler_upload_calls
        << ",";
    oss << "\"standalone_scaler_download_calls\":"
        << st.pipeline.open_cuda_transfers.standalone_scaler_download_calls
        << ",";
    oss << "\"denoise_tensor_upload_calls\":"
        << st.pipeline.open_cuda_transfers.denoise_tensor_upload_calls << ",";
    oss << "\"denoise_tensor_download_calls\":"
        << st.pipeline.open_cuda_transfers.denoise_tensor_download_calls << ",";
    oss << "\"forced_sync_calls\":"
        << st.pipeline.open_cuda_transfers.forced_sync_calls << ",";
    oss << "\"cpu_tail_stage_calls\":"
        << st.pipeline.open_cuda_transfers.cpu_tail_stage_calls << ",";
    oss << "\"cpu_tail_key_light_calls\":"
        << st.pipeline.open_cuda_transfers.cpu_tail_key_light_calls << ",";
    oss << "\"cpu_tail_auto_frame_calls\":"
        << st.pipeline.open_cuda_transfers.cpu_tail_auto_frame_calls << ",";
    oss << "\"cpu_tail_auto_frame_face_tracking_calls\":"
        << st.pipeline.open_cuda_transfers
               .cpu_tail_auto_frame_face_tracking_calls
        << ",";
    oss << "\"cpu_tail_auto_frame_matte_tracking_calls\":"
        << st.pipeline.open_cuda_transfers
               .cpu_tail_auto_frame_matte_tracking_calls
        << ",";
    oss << "\"cpu_tail_auto_frame_cpu_crop_calls\":"
        << st.pipeline.open_cuda_transfers.cpu_tail_auto_frame_cpu_crop_calls
        << ",";
    oss << "\"cpu_tail_denoise_calls\":"
        << st.pipeline.open_cuda_transfers.cpu_tail_denoise_calls;
    oss << "}";
  }

  if (debug_open_vulkan_transfers) {
    oss << ",\"open_vulkan_transfers\":{";
    oss << "\"active_frames\":"
        << st.pipeline.open_vulkan_transfers.active_frames << ",";
    oss << "\"upload_calls\":"
        << st.pipeline.open_vulkan_transfers.upload_calls << ",";
    oss << "\"download_calls\":"
        << st.pipeline.open_vulkan_transfers.download_calls << ",";
    oss << "\"final_download_calls\":"
        << st.pipeline.open_vulkan_transfers.final_download_calls << ",";
    oss << "\"cpu_continuation_download_calls\":"
        << st.pipeline.open_vulkan_transfers.cpu_continuation_download_calls
        << ",";
    oss << "\"alpha_download_calls\":"
        << st.pipeline.open_vulkan_transfers.alpha_download_calls << ",";
    oss << "\"matte_frame_upload_calls\":"
        << st.pipeline.open_vulkan_transfers.matte_frame_upload_calls << ",";
    oss << "\"preprocess_dispatch_calls\":"
        << st.pipeline.open_vulkan_transfers.preprocess_dispatch_calls << ",";
    oss << "\"matting_inference_calls\":"
        << st.pipeline.open_vulkan_transfers.matting_inference_calls << ",";
    oss << "\"alpha_resize_dispatch_calls\":"
        << st.pipeline.open_vulkan_transfers.alpha_resize_dispatch_calls
        << ",";
    oss << "\"blur_dispatch_calls\":"
        << st.pipeline.open_vulkan_transfers.blur_dispatch_calls << ",";
    oss << "\"virtual_background_blur_dispatch_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_blur_dispatch_calls
        << ",";
    oss << "\"virtual_background_blur_alpha_readback_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_blur_alpha_readback_calls
        << ",";
    oss << "\"virtual_background_blur_cpu_fallback_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_blur_cpu_fallback_calls
        << ",";
    oss << "\"virtual_background_blur_runtime_failure_frames\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_blur_runtime_failure_frames
        << ",";
    oss << "\"virtual_background_blur_device_loss_frames\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_blur_device_loss_frames
        << ",";
    oss << "\"virtual_background_remove_dispatch_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_remove_dispatch_calls
        << ",";
    oss << "\"virtual_background_remove_alpha_readback_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_remove_alpha_readback_calls
        << ",";
    oss << "\"virtual_background_remove_cpu_fallback_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_remove_cpu_fallback_calls
        << ",";
    oss << "\"virtual_background_remove_runtime_failure_frames\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_remove_runtime_failure_frames
        << ",";
    oss << "\"virtual_background_remove_device_loss_frames\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_remove_device_loss_frames
        << ",";
    oss << "\"virtual_background_replace_asset_allocation_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_replace_asset_allocation_calls
        << ",";
    oss << "\"virtual_background_replace_asset_decode_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_replace_asset_decode_calls
        << ",";
    oss << "\"virtual_background_replace_asset_upload_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_replace_asset_upload_calls
        << ",";
    oss << "\"virtual_background_replace_asset_resize_dispatch_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_replace_asset_resize_dispatch_calls
        << ",";
    oss << "\"virtual_background_replace_dispatch_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_replace_dispatch_calls
        << ",";
    oss << "\"virtual_background_replace_alpha_readback_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_replace_alpha_readback_calls
        << ",";
    oss << "\"virtual_background_replace_cpu_fallback_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_replace_cpu_fallback_calls
        << ",";
    oss << "\"virtual_background_replace_runtime_failure_frames\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_replace_runtime_failure_frames
        << ",";
    oss << "\"virtual_background_replace_device_loss_frames\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_background_replace_device_loss_frames
        << ",";
    oss << "\"background_upload_calls\":"
        << st.pipeline.open_vulkan_transfers.background_upload_calls << ",";
    oss << "\"composite_dispatch_calls\":"
        << st.pipeline.open_vulkan_transfers.composite_dispatch_calls << ",";
    oss << "\"key_light_dispatch_calls\":"
        << st.pipeline.open_vulkan_transfers.key_light_dispatch_calls << ",";
    oss << "\"virtual_key_light_shared_matte_reuse_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_key_light_shared_matte_reuse_calls
        << ",";
    oss << "\"virtual_key_light_independent_matte_inference_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_key_light_independent_matte_inference_calls
        << ",";
    oss << "\"virtual_key_light_passthrough_frames\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_key_light_passthrough_frames
        << ",";
    oss << "\"virtual_key_light_alpha_readback_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_key_light_alpha_readback_calls
        << ",";
    oss << "\"virtual_key_light_cpu_fallback_calls\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_key_light_cpu_fallback_calls
        << ",";
    oss << "\"virtual_key_light_runtime_failure_frames\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_key_light_runtime_failure_frames
        << ",";
    oss << "\"virtual_key_light_device_loss_frames\":"
        << st.pipeline.open_vulkan_transfers
               .virtual_key_light_device_loss_frames
        << ",";
    oss << "\"crop_resize_dispatch_calls\":"
        << st.pipeline.open_vulkan_transfers.crop_resize_dispatch_calls << ",";
    oss << "\"standalone_scaler_upload_calls\":"
        << st.pipeline.open_vulkan_transfers.standalone_scaler_upload_calls
        << ",";
    oss << "\"standalone_scaler_download_calls\":"
        << st.pipeline.open_vulkan_transfers.standalone_scaler_download_calls
        << ",";
    oss << "\"forced_sync_calls\":"
        << st.pipeline.open_vulkan_transfers.forced_sync_calls << ",";
    oss << "\"cpu_tail_stage_calls\":"
        << st.pipeline.open_vulkan_transfers.cpu_tail_stage_calls << ",";
    oss << "\"cpu_tail_key_light_calls\":"
        << st.pipeline.open_vulkan_transfers.cpu_tail_key_light_calls << ",";
    oss << "\"cpu_tail_auto_frame_calls\":"
        << st.pipeline.open_vulkan_transfers.cpu_tail_auto_frame_calls << ",";
    oss << "\"cpu_tail_auto_frame_face_tracking_calls\":"
        << st.pipeline.open_vulkan_transfers
               .cpu_tail_auto_frame_face_tracking_calls
        << ",";
    oss << "\"cpu_tail_auto_frame_matte_tracking_calls\":"
        << st.pipeline.open_vulkan_transfers
               .cpu_tail_auto_frame_matte_tracking_calls
        << ",";
    oss << "\"cpu_tail_auto_frame_cpu_crop_calls\":"
        << st.pipeline.open_vulkan_transfers
               .cpu_tail_auto_frame_cpu_crop_calls
        << ",";
    oss << "\"fallback_frames\":"
        << st.pipeline.open_vulkan_transfers.fallback_frames << ",";
    oss << "\"runtime_failure_frames\":"
        << st.pipeline.open_vulkan_transfers.runtime_failure_frames;
    oss << "}";
  }

  if (std::getenv("STUDIOCAST_DEBUG_MAXINE_TRANSFERS")) {
    oss << ",\"maxine_transfers\":{";
    oss << "\"active_frames\":" << st.pipeline.maxine_transfers.active_frames
        << ",";
    oss << "\"rgb_to_bgr_calls\":"
        << st.pipeline.maxine_transfers.rgb_to_bgr_calls << ",";
    oss << "\"upload_attempts\":"
        << st.pipeline.maxine_transfers.upload_attempts << ",";
    oss << "\"upload_calls\":" << st.pipeline.maxine_transfers.upload_calls
        << ",";
    oss << "\"matte_inference_attempts\":"
        << st.pipeline.maxine_transfers.matte_inference_attempts << ",";
    oss << "\"green_screen_calls\":"
        << st.pipeline.maxine_transfers.green_screen_calls << ",";
    oss << "\"duplicate_green_screen_calls\":"
        << st.pipeline.maxine_transfers.duplicate_green_screen_calls << ",";
    oss << "\"shared_green_screen_matte_reuse_calls\":"
        << st.pipeline.maxine_transfers.shared_green_screen_matte_reuse_calls
        << ",";
    oss << "\"shared_green_screen_matte_incompatible_calls\":"
        << st.pipeline.maxine_transfers
               .shared_green_screen_matte_incompatible_calls
        << ",";
    oss << "\"shared_green_screen_input_incompatible_calls\":"
        << st.pipeline.maxine_transfers
               .shared_green_screen_input_incompatible_calls
        << ",";
    oss << "\"download_calls\":" << st.pipeline.maxine_transfers.download_calls
        << ",";
    oss << "\"final_download_attempts\":"
        << st.pipeline.maxine_transfers.final_download_attempts << ",";
    oss << "\"final_download_calls\":"
        << st.pipeline.maxine_transfers.final_download_calls << ",";
    oss << "\"cpu_continuation_download_attempts\":"
        << st.pipeline.maxine_transfers.cpu_continuation_download_attempts
        << ",";
    oss << "\"cpu_continuation_download_calls\":"
        << st.pipeline.maxine_transfers.cpu_continuation_download_calls << ",";
    oss << "\"device_bridge_attempts\":"
        << st.pipeline.maxine_transfers.device_bridge_attempts << ",";
    oss << "\"device_bridge_calls\":"
        << st.pipeline.maxine_transfers.device_bridge_calls << ",";
    oss << "\"background_setup_upload_attempts\":"
        << st.pipeline.maxine_transfers.background_setup_upload_attempts
        << ",";
    oss << "\"background_setup_upload_calls\":"
        << st.pipeline.maxine_transfers.background_setup_upload_calls << ",";
    oss << "\"bgr_to_rgb_calls\":"
        << st.pipeline.maxine_transfers.bgr_to_rgb_calls << ",";
    oss << "\"deferred_readbacks\":"
        << st.pipeline.maxine_transfers.deferred_readbacks << ",";
    oss << "\"forced_sync_attempts\":"
        << st.pipeline.maxine_transfers.forced_sync_attempts << ",";
    oss << "\"forced_sync_calls\":"
        << st.pipeline.maxine_transfers.forced_sync_calls << ",";
    oss << "\"composite_attempts\":"
        << st.pipeline.maxine_transfers.composite_attempts << ",";
    oss << "\"composite_calls\":"
        << st.pipeline.maxine_transfers.composite_calls << ",";
    oss << "\"synchronous_sdk_run_attempts\":"
        << st.pipeline.maxine_transfers.synchronous_sdk_run_attempts << ",";
    oss << "\"synchronous_sdk_run_calls\":"
        << st.pipeline.maxine_transfers.synchronous_sdk_run_calls << ",";
    oss << "\"asynchronous_sdk_run_attempts\":"
        << st.pipeline.maxine_transfers.asynchronous_sdk_run_attempts << ",";
    oss << "\"asynchronous_sdk_run_calls\":"
        << st.pipeline.maxine_transfers.asynchronous_sdk_run_calls << ",";
    oss << "\"setup_attempts\":"
        << st.pipeline.maxine_transfers.setup_attempts << ",";
    oss << "\"setup_successes\":"
        << st.pipeline.maxine_transfers.setup_successes << ",";
    oss << "\"cpu_tail_stage_calls\":"
        << st.pipeline.maxine_transfers.cpu_tail_stage_calls << ",";
    oss << "\"runtime_failure_frames\":"
        << st.pipeline.maxine_transfers.runtime_failure_frames << ",";
    oss << "\"stages\":{";
    for (std::size_t i = 0;
         i < st.pipeline.maxine_transfers.stage_attempts.size(); ++i) {
      if (i != 0)
        oss << ",";
      const auto kind =
          static_cast<studiocast::maxine::ResidentStageKind>(i);
      oss << "\"" << studiocast::maxine::ResidentStageKindName(kind)
          << "\":{";
      oss << "\"attempts\":"
          << st.pipeline.maxine_transfers.stage_attempts[i] << ",";
      oss << "\"successes\":"
          << st.pipeline.maxine_transfers.stage_successes[i];
      oss << "}";
    }
    oss << "},";
    oss << "\"standalone_scaler_upload_calls\":"
        << st.pipeline.maxine_transfers.standalone_scaler_upload_calls << ",";
    oss << "\"standalone_scaler_download_calls\":"
        << st.pipeline.maxine_transfers.standalone_scaler_download_calls;
    oss << "}";
  }

  // Optional debug stats. Kept out of the default schema unless explicitly
  // enabled.
  if (std::getenv("STUDIOCAST_DEBUG_VIDEO_STATS")) {
    oss << ",\"debug\":{";
    oss << "\"latency_ms\":" << std::setprecision(6)
        << st.pipeline.debug.latency_ms << ",";
    oss << "\"capture_sequence\":" << st.pipeline.debug.capture_sequence << ",";
    oss << "\"dropped_capture_frames\":"
        << st.pipeline.debug.dropped_capture_frames << ",";
    oss << "\"output_format_changes\":"
        << st.pipeline.debug.output_format_changes << ",";
    oss << "\"output_refresh_failures\":"
        << st.pipeline.debug.output_refresh_failures << ",";
    oss << "\"output_write_recoveries\":"
        << st.pipeline.debug.output_write_recoveries;
    oss << ",\"yuyv_capture_to_rgb_calls\":"
        << st.pipeline.debug.yuyv_capture_to_rgb_calls;
    oss << ",\"yuyv_output_from_rgb_calls\":"
        << st.pipeline.debug.yuyv_output_from_rgb_calls;
    oss << ",\"raw_yuyv_passthrough_frames\":"
        << st.pipeline.debug.raw_yuyv_passthrough_frames;
    oss << ",\"raw_yuyv_passthrough_bytes\":"
        << st.pipeline.debug.raw_yuyv_passthrough_bytes;

    oss << ",\"pace_sleep_ms\":" << std::setprecision(6)
        << st.pipeline.debug.pace_sleep_ms;
    oss << ",\"pace_late_ms\":" << std::setprecision(6)
        << st.pipeline.debug.pace_late_ms;
    oss << ",\"pace_sleeps\":" << st.pipeline.debug.pace_sleeps;
    oss << ",\"pace_late_frames\":" << st.pipeline.debug.pace_late_frames;
    oss << ",\"pace_resyncs\":" << st.pipeline.debug.pace_resyncs;

    oss << "}";
  }
  oss << ",";

  // Deterministic effect ordering + rule-based disable reasons.
  const auto plan = studiocast::video::effects::BuildBroadcastEffectsPlan(
      cfg.pipeline.effects);
  oss << "\"effects_plan\":{";
  oss << "\"ordered\":[";
  for (std::size_t i = 0; i < plan.ordered_effect_ids.size(); ++i) {
    if (i)
      oss << ",";
    oss << "\"" << JsonEscape(plan.ordered_effect_ids[i]) << "\"";
  }
  oss << "],";

  oss << "\"vignette_attach_to\":\""
      << JsonEscape(plan.vignette_attach_to_effect_id) << "\",";

  oss << "\"disabled\":[";
  for (std::size_t i = 0; i < plan.disabled.size(); ++i) {
    if (i)
      oss << ",";
    oss << "{";
    oss << "\"id\":\"" << JsonEscape(plan.disabled[i].id) << "\",";
    oss << "\"reason\":\"" << JsonEscape(plan.disabled[i].reason) << "\"";
    oss << "}";
  }
  oss << "]";
  oss << "}";
  oss << "},";

  oss << "\"last_error\":\"" << JsonEscape(st.last_error) << "\"";
  oss << "}"; // video

  std::string audioSourceResolved =
      ast.selected_source.empty() ? acfg.source_name : ast.selected_source;
  std::string audioSourceError = ast.source_error;
  std::vector<std::string> audioSourceWarnings = ast.source_warnings;
  {
    std::string reason;
    if (audioSourceError.empty() &&
        studiocast::audio::IsUnsafeInputSourceName(acfg.source_name, &reason)) {
      audioSourceError = reason;
    }
  }

  std::string speakerTargetResolved = ast.speaker_target_sink_active.empty()
                                          ? acfg.speaker_target_sink
                                          : ast.speaker_target_sink_active;
  std::string speakerTargetError;
  {
    std::string reason;
    if (studiocast::audio::IsUnsafeSpeakerTargetSinkName(
            acfg.speaker_target_sink, &reason)) {
      speakerTargetError = reason;
    }
  }

  // Audio status.
  oss << ",\"audio\":{";
  oss << "\"enabled\":" << BoolJson(acfg.enabled) << ",";
  oss << "\"create_virtual_mic\":" << BoolJson(acfg.create_virtual_mic) << ",";
  oss << "\"create_virtual_speakers\":"
      << BoolJson(acfg.create_virtual_speakers) << ",";
  oss << "\"source\":\""
      << JsonEscape(acfg.source_name.empty() ? std::string("auto")
                                             : acfg.source_name)
      << "\",";
  oss << "\"source_resolved\":\""
      << JsonEscape(audioSourceResolved.empty() ? std::string("auto")
                                                : audioSourceResolved)
      << "\",";
  oss << "\"source_availability\":\""
      << JsonEscape(ast.source_availability.empty() ? std::string("unknown")
                                                    : ast.source_availability)
      << "\",";
  oss << "\"source_error\":\"" << JsonEscape(audioSourceError) << "\",";
  oss << "\"source_warnings\":[";
  for (std::size_t i = 0; i < audioSourceWarnings.size(); ++i) {
    if (i)
      oss << ",";
    oss << "\"" << JsonEscape(audioSourceWarnings[i]) << "\"";
  }
  oss << "],";
  oss << "\"mic_present\":" << BoolJson(ast.mic_present) << ",";
  oss << "\"mic_consumer_present\":" << BoolJson(ast.mic_consumer_present)
      << ",";
  oss << "\"mic_consumer_count\":" << ast.mic_consumer_count << ",";
  oss << "\"mic_consumer_error\":\"" << JsonEscape(ast.mic_consumer_error)
      << "\",";

  const EndpointReadiness microphoneReadiness =
      BuildMicrophoneEndpointReadiness(ast, acfg, audioSourceError);
  oss << "\"microphone\":";
  AppendEndpointObject(oss, microphoneReadiness,
                       audioSourceResolved.empty() ? std::string("auto")
                                                   : audioSourceResolved,
                       ast.effects_backend_active);
  oss << ",";

  const double spk_proc_avg_ms =
      ast.speakers_pipeline_frames_processed
          ? (static_cast<double>(ast.speakers_pipeline_process_time_us_sum) /
             static_cast<double>(ast.speakers_pipeline_frames_processed) /
             1000.0)
          : 0.0;
  const double mic_proc_avg_ms =
      ast.pipeline_frames_processed
          ? (static_cast<double>(ast.pipeline_process_time_us_sum) /
             static_cast<double>(ast.pipeline_frames_processed) / 1000.0)
          : 0.0;
  const EndpointReadiness speakersReadiness =
      BuildSpeakersEndpointReadiness(ast, acfg, speakerTargetError);

  // Speakers routing status. `route_mode` distinguishes pass-through
  // module-loopback from the consumer-gated processed speaker pipeline.
  oss << "\"speakers\":{";
  oss << "\"action\":\"" << JsonEscape(speakersReadiness.action) << "\",";
  oss << "\"readiness\":";
  AppendReadinessJson(oss, speakersReadiness.state, speakersReadiness.summary,
                      speakersReadiness.detail);
  oss << ",";
  oss << "\"enabled\":" << BoolJson(acfg.speakers_enabled) << ",";
  oss << "\"target_sink\":\""
      << JsonEscape(acfg.speaker_target_sink.empty() ? std::string("auto")
                                                     : acfg.speaker_target_sink)
      << "\",";
  oss << "\"target_sink_resolved\":\""
      << JsonEscape(speakerTargetResolved.empty() ? std::string("auto")
                                                  : speakerTargetResolved)
      << "\",";
  oss << "\"target_sink_error\":\"" << JsonEscape(speakerTargetError) << "\",";
  oss << "\"latency_ms\":" << acfg.speaker_latency_ms << ",";
  oss << "\"present\":" << BoolJson(ast.speakers_present) << ",";
  oss << "\"consumer_present\":" << BoolJson(ast.speakers_consumer_present)
      << ",";
  oss << "\"consumer_count\":" << ast.speakers_consumer_count << ",";
  oss << "\"consumer_error\":\"" << JsonEscape(ast.speakers_consumer_error)
      << "\",";
  oss << "\"routing_active\":" << BoolJson(ast.speakers_routing_active) << ",";
  oss << "\"route_mode\":\""
      << JsonEscape(ast.speakers_route_mode.empty() ? std::string("off")
                                                    : ast.speakers_route_mode)
      << "\",";
  oss << "\"pipeline_running\":" << BoolJson(ast.speakers_pipeline_running)
      << ",";
  oss << "\"pipeline_starting\":" << BoolJson(ast.speakers_pipeline_starting)
      << ",";
  oss << "\"pipeline_active_needed\":"
      << BoolJson(ast.speakers_pipeline_active_needed) << ",";
  oss << "\"pipeline_state\":\""
      << JsonEscape(ast.speakers_pipeline_state.empty()
                        ? std::string("disabled")
                        : ast.speakers_pipeline_state)
      << "\",";
  oss << "\"pipeline_idle_reason\":\""
      << JsonEscape(ast.speakers_pipeline_idle_reason) << "\",";
  oss << "\"backend_active\":\"" << JsonEscape(ast.speakers_backend_active)
      << "\",";
  oss << "\"effects_note\":\"" << JsonEscape(ast.speakers_effects_note)
      << "\",";
  oss << "\"intensity\":" << ast.speakers_intensity << ",";
  oss << "\"target_sink_active\":\""
      << JsonEscape(ast.speaker_target_sink_active) << "\",";
  oss << "\"last_error\":\"" << JsonEscape(ast.speakers_last_error) << "\",";
  oss << "\"pipeline_last_error\":\""
      << JsonEscape(ast.speakers_pipeline_last_error) << "\",";
  oss << "\"open_audio_runtime\":"
      << OpenAudioRuntimeStatusToJson(ast.speakers_open_audio_runtime) << ",";
  oss << "\"pipeline_perf\":{";
  oss << "\"frames_processed\":" << ast.speakers_pipeline_frames_processed
      << ",";
  oss << "\"process_ms_avg\":" << spk_proc_avg_ms << ",";
  oss << "\"process_us_last\":" << ast.speakers_pipeline_process_time_us_last
      << ",";
  oss << "\"process_us_max\":" << ast.speakers_pipeline_process_time_us_max
      << ",";
  oss << "\"process_overruns\":" << ast.speakers_pipeline_process_overruns;

  // Optional debug stats. Kept out of the default schema unless explicitly
  // enabled.
  if (std::getenv("STUDIOCAST_DEBUG_AUDIO_STATS")) {
    oss << ",\"debug\":{";
    oss << "\"pulse_capture_latency_ms\":"
        << (ast.speakers_pipeline_pulse_capture_latency_us_last / 1000u) << ",";
    oss << "\"pulse_playback_latency_ms\":"
        << (ast.speakers_pipeline_pulse_playback_latency_us_last / 1000u)
        << ",";
    oss << "\"pulse_latency_ms_max\":"
        << (ast.speakers_pipeline_pulse_latency_us_max / 1000u) << ",";
    oss << "\"resync_events\":" << ast.speakers_pipeline_resync_events;
    oss << "}";
  }
  oss << "}";
  oss << "},";

  // Canonical effect model for GUI/CLI.
  oss << "\"audio_effects\":"
      << studiocast::audio::effects::BroadcastAudioEffectsToJson(acfg.effects)
      << ",";

  // What effect is currently active (stable-ish summary string).
  std::string mic_mode = "none";
  if (acfg.effects.microphone.studio_voice_enabled) {
    mic_mode = "studio_voice";
  } else if (acfg.effects.microphone.noise_removal_enabled &&
             acfg.effects.microphone.room_echo_removal_enabled) {
    mic_mode = "noise_echo_removal";
  } else if (acfg.effects.microphone.noise_removal_enabled) {
    mic_mode = "noise_removal";
  } else if (acfg.effects.microphone.room_echo_removal_enabled) {
    mic_mode = "room_echo_removal";
  }
  oss << "\"mic_mode\":\"" << JsonEscape(mic_mode) << "\",";

  oss << "\"pipeline\":{";
  oss << "\"running\":" << BoolJson(ast.pipeline_running) << ",";
  oss << "\"starting\":" << BoolJson(ast.pipeline_starting) << ",";
  oss << "\"active_needed\":" << BoolJson(ast.pipeline_active_needed) << ",";
  oss << "\"state\":\""
      << JsonEscape(ast.pipeline_state.empty() ? std::string("disabled")
                                               : ast.pipeline_state)
      << "\",";
  oss << "\"idle_reason\":\"" << JsonEscape(ast.pipeline_idle_reason) << "\",";
  oss << "\"sink\":\"" << JsonEscape(ast.pipeline_sink) << "\",";
  oss << "\"backend_active\":\"" << JsonEscape(ast.effects_backend_active)
      << "\",";
  oss << "\"open_audio_runtime\":"
      << OpenAudioRuntimeStatusToJson(ast.open_audio_runtime) << ",";
  oss << "\"effects_note\":\"" << JsonEscape(ast.effects_note) << "\",";
  oss << "\"effect_selector\":\"" << JsonEscape(ast.effect_selector) << "\",";
  oss << "\"feature_id\":\"" << JsonEscape(ast.feature_id) << "\",";
  oss << "\"intensity\":" << ast.intensity << ",";
  oss << "\"frames_processed\":" << ast.pipeline_frames_processed << ",";
  oss << "\"process_ms_avg\":" << mic_proc_avg_ms << ",";
  oss << "\"process_us_last\":" << ast.pipeline_process_time_us_last << ",";
  oss << "\"process_us_max\":" << ast.pipeline_process_time_us_max << ",";
  oss << "\"process_overruns\":" << ast.pipeline_process_overruns << ",";

  // Optional debug stats. Kept out of the default schema unless explicitly
  // enabled.
  if (std::getenv("STUDIOCAST_DEBUG_AUDIO_STATS")) {
    oss << "\"debug\":{";
    oss << "\"pulse_capture_latency_ms\":"
        << (ast.pipeline_pulse_capture_latency_us_last / 1000u) << ",";
    oss << "\"pulse_playback_latency_ms\":"
        << (ast.pipeline_pulse_playback_latency_us_last / 1000u) << ",";
    oss << "\"pulse_latency_ms_max\":"
        << (ast.pipeline_pulse_latency_us_max / 1000u) << ",";
    oss << "\"resync_events\":" << ast.pipeline_resync_events;
    oss << "},";
  }

  oss << "\"gpu\":{";
  oss << "\"index\":" << ast.gpu_index << ",";
  oss << "\"name\":\"" << JsonEscape(ast.gpu_name) << "\",";
  oss << "\"compute_cap\":\"" << JsonEscape(ast.gpu_compute_cap) << "\"";
  oss << "},";
  oss << "\"last_error\":\"" << JsonEscape(ast.last_error) << "\"";
  oss << "}"; // pipeline

  oss << "}"; // audio

  oss << "}";
  return oss.str();
}

std::string
ConfigToJson(const studiocast::video::VirtualCameraServiceConfig &cfg,
             const studiocast::config::DaemonConfig *daemonConfig = nullptr) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"enabled\":" << BoolJson(cfg.enabled) << ",";
  oss << "\"always_on\":" << BoolJson(cfg.always_on) << ",";
  oss << "\"consumer_poll_ms\":" << cfg.consumer_poll_ms << ",";
  oss << "\"start_grace_ms\":" << cfg.start_grace_ms << ",";
  oss << "\"stop_grace_ms\":" << cfg.stop_grace_ms << ",";
  oss << "\"min_run_ms\":" << cfg.min_run_ms << ",";
  oss << "\"input_device\":\"" << JsonEscape(cfg.pipeline.input_device)
      << "\",";
  oss << "\"output_device\":\"" << JsonEscape(cfg.pipeline.output_device)
      << "\",";
  oss << "\"width\":" << cfg.pipeline.width << ",";
  oss << "\"height\":" << cfg.pipeline.height << ",";
  oss << "\"fps\":" << cfg.pipeline.fps << ",";
  oss << "\"output_format_requested\":\""
      << JsonEscape(
             studiocast::video::PixelFormatName(cfg.pipeline.output_format))
      << "\",";
  oss << "\"compute_backend\":\""
      << JsonEscape(studiocast::video::ComputeBackendPreferenceToString(
             cfg.pipeline.compute_backend))
      << "\",";
  oss << "\"allow_cpu_resize\":" << BoolJson(cfg.pipeline.allow_cpu_resize)
      << ",";
  oss << "\"vulkan_device\":\""
      << JsonEscape(daemonConfig ? daemonConfig->video_vulkan_device : "auto")
      << "\",";
  oss << "\"vulkan_allow_cpu\":"
      << BoolJson(daemonConfig && daemonConfig->video_vulkan_allow_cpu) << ",";
  oss << "\"vulkan_selection_apply_policy\":"
         "\"next_vulkan_device_initialization\",";
  oss << "\"mirror\":" << BoolJson(cfg.pipeline.effects.mirror) << ",";
  // Legacy flat fields (kept for compatibility): derived from the canonical
  // Broadcast schema.
  oss << "\"background\":\""
      << JsonEscape(studiocast::video::effects::ToString(
             cfg.pipeline.effects.virtual_background.mode))
      << "\",";
  oss << "\"background_backend\":\""
      << JsonEscape(
             studiocast::video::effects::ToString(cfg.pipeline.effects.engine))
      << "\",";
  oss << "\"background_strength\":"
      << cfg.pipeline.effects.virtual_background.strength << ",";
  oss << "\"background_remove_color\":\""
      << JsonEscape(cfg.pipeline.effects.virtual_background.remove_color)
      << "\",";
  oss << "\"background_replace_image\":\""
      << JsonEscape(cfg.pipeline.effects.virtual_background.replace_path)
      << "\",";

  const int vkl_intensity = std::max(
      0, std::min(100, cfg.pipeline.effects.virtual_key_light.intensity));
  oss << "\"virtual_key_light\":"
      << BoolJson(cfg.pipeline.effects.virtual_key_light.enabled) << ",";
  oss << "\"virtual_key_light_intensity\":" << vkl_intensity << ",";
  oss << "\"virtual_key_light_temperature\":\""
      << JsonEscape(FormatKeyLightTemperaturePreset(
             cfg.pipeline.effects.virtual_key_light.temperature_preset))
      << "\",";
  oss << "\"virtual_key_light_pan\":"
      << cfg.pipeline.effects.virtual_key_light.direction_pan_degrees << ",";
  oss << "\"virtual_key_light_hdri\":\""
      << JsonEscape(cfg.pipeline.effects.virtual_key_light.hdri_path) << "\",";

  const int vignette_intensity =
      std::max(0, std::min(100, cfg.pipeline.effects.vignette.intensity));
  oss << "\"vignette\":" << BoolJson(cfg.pipeline.effects.vignette.enabled)
      << ",";
  oss << "\"vignette_intensity\":" << vignette_intensity << ",";
  oss << "\"vignette_center_on_face\":"
      << BoolJson(cfg.pipeline.effects.vignette.center_on_tracked_face) << ",";

  // Canonical, nested effects model (safe for file paths with spaces).
  oss << "\"video_effects\":"
      << studiocast::video::BroadcastCameraEffectsContractToJson(
             cfg.pipeline.effects);
  oss << "}";
  return oss.str();
}

std::string
AudioConfigToJson(const studiocast::audio::VirtualAudioServiceConfig &cfg) {
  std::ostringstream oss;
  oss << "{";
  oss << "\"enabled\":" << BoolJson(cfg.enabled) << ",";
  oss << "\"create_virtual_mic\":" << BoolJson(cfg.create_virtual_mic) << ",";
  oss << "\"create_virtual_speakers\":" << BoolJson(cfg.create_virtual_speakers)
      << ",";
  oss << "\"speakers_enabled\":" << BoolJson(cfg.speakers_enabled) << ",";
  oss << "\"speaker_target_sink\":\""
      << JsonEscape(cfg.speaker_target_sink.empty() ? std::string("auto")
                                                    : cfg.speaker_target_sink)
      << "\",";
  oss << "\"speaker_latency_ms\":" << cfg.speaker_latency_ms << ",";
  oss << "\"source\":\""
      << JsonEscape(cfg.source_name.empty() ? std::string("auto")
                                            : cfg.source_name)
      << "\",";
  oss << "\"audio_effects\":"
      << studiocast::audio::effects::BroadcastAudioEffectsToJson(cfg.effects);
  oss << "}";
  return oss.str();
}

bool ApplyAudioConfigPatchJsonText(
    const std::string &jsonText,
    studiocast::audio::VirtualAudioServiceConfig *cfg,
    std::vector<std::string> *warnings, std::string *error) {
  if (!cfg) {
    if (error)
      *error = "config pointer is null";
    return false;
  }

  studiocast::util::json::Value root;
  if (!studiocast::util::json::Parse(jsonText, &root, error))
    return false;
  const auto *obj = root.AsObject();
  if (!obj) {
    if (error)
      *error = "audio config must be a JSON object";
    return false;
  }

  // enabled
  if (auto it = obj->find("enabled"); it != obj->end()) {
    const bool *b = it->second.AsBool();
    if (!b) {
      if (error)
        *error = "enabled must be a boolean";
      return false;
    }
    cfg->enabled = *b;
  }

  // create_virtual_mic
  if (auto it = obj->find("create_virtual_mic"); it != obj->end()) {
    const bool *b = it->second.AsBool();
    if (!b) {
      if (error)
        *error = "create_virtual_mic must be a boolean";
      return false;
    }
    cfg->create_virtual_mic = *b;
  }

  // create_virtual_speakers
  if (auto it = obj->find("create_virtual_speakers"); it != obj->end()) {
    const bool *b = it->second.AsBool();
    if (!b) {
      if (error)
        *error = "create_virtual_speakers must be a boolean";
      return false;
    }
    cfg->create_virtual_speakers = *b;
  }

  // speakers_enabled
  if (auto it = obj->find("speakers_enabled"); it != obj->end()) {
    const bool *b = it->second.AsBool();
    if (!b) {
      if (error)
        *error = "speakers_enabled must be a boolean";
      return false;
    }
    cfg->speakers_enabled = *b;
  }

  // speaker_target_sink
  if (auto it = obj->find("speaker_target_sink"); it != obj->end()) {
    const std::string *s = it->second.AsString();
    if (!s) {
      if (error)
        *error = "speaker_target_sink must be a string";
      return false;
    }
    cfg->speaker_target_sink = (*s == "auto") ? std::string() : *s;
  }

  // speaker_latency_ms
  if (auto it = obj->find("speaker_latency_ms"); it != obj->end()) {
    const double *n = it->second.AsNumber();
    if (!n) {
      if (error)
        *error = "speaker_latency_ms must be a number";
      return false;
    }
    const int v = static_cast<int>(std::lround(*n));
    if (std::fabs(*n - static_cast<double>(v)) > 1e-6) {
      if (error)
        *error = "speaker_latency_ms must be an integer";
      return false;
    }
    if (v < 1 || v > 5000) {
      if (error)
        *error = "speaker_latency_ms out of range (expected 1..5000)";
      return false;
    }
    cfg->speaker_latency_ms = v;
  }

  // Routing implies the speakers device must exist.
  if (cfg->speakers_enabled) {
    cfg->create_virtual_speakers = true;
  }

  // source
  if (auto it = obj->find("source"); it != obj->end()) {
    const std::string *s = it->second.AsString();
    if (!s) {
      if (error)
        *error = "source must be a string";
      return false;
    }
    cfg->source_name = (*s == "auto") ? std::string() : *s;
  }

  // effects blob
  const studiocast::util::json::Value *fxVal = nullptr;
  if (auto it = obj->find("audio_effects"); it != obj->end()) {
    fxVal = &it->second;
  } else if (auto it2 = obj->find("effects"); it2 != obj->end()) {
    // Accept alias for convenience.
    fxVal = &it2->second;
    if (warnings)
      warnings->push_back("effects: alias accepted; please use audio_effects");
  }

  if (fxVal) {
    studiocast::audio::effects::BroadcastAudioEffectsJsonParseOptions options;
    options.allow_unknown_keys = true;
    std::vector<std::string> parseWarnings;
    std::string parseError;
    if (!studiocast::audio::effects::ApplyBroadcastAudioEffectsPatchJson(
            *fxVal, &cfg->effects, options, &parseWarnings, &parseError)) {
      if (error)
        *error = parseError;
      return false;
    }
    if (warnings)
      warnings->insert(warnings->end(), parseWarnings.begin(),
                       parseWarnings.end());
  }

  return true;
}

bool ValidateAudioConfigSafetyForDaemon(
    const studiocast::audio::VirtualAudioServiceConfig &cfg,
    std::vector<std::string> *warnings, std::string *error) {
  std::string reason;
  if (studiocast::audio::IsUnsafeInputSourceName(cfg.source_name, &reason)) {
    if (error) {
      *error = "Unsafe audio source: " + reason +
               " Select a physical microphone/input source or use "
               "source=\"auto\".";
    }
    return false;
  }

  if (studiocast::audio::IsUnsafeSpeakerTargetSinkName(cfg.speaker_target_sink,
                                                       &reason)) {
    if (error) {
      *error = "Unsafe speaker target sink: " + reason +
               " Select a physical output sink or use "
               "speaker_target_sink=\"auto\".";
    }
    return false;
  }

  if (cfg.enabled) {
    const auto source =
        studiocast::audio::ResolveSafeInputSourceName(cfg.source_name);
    if (!source.ok) {
      if (error)
        *error = "Unsafe audio source configuration: " + source.error;
      return false;
    }
    if (warnings) {
      warnings->insert(warnings->end(), source.warnings.begin(),
                       source.warnings.end());
    }
  }

  if (cfg.speakers_enabled && cfg.speaker_target_sink.empty()) {
    std::string sinkErr;
    if (!studiocast::audio::ChooseSafeSpeakerTargetSinkName(
            cfg.speaker_target_sink, &sinkErr)) {
      if (error) {
        *error = "Unsafe speaker routing configuration: " +
                 (sinkErr.empty() ? "no safe physical output sink was found"
                                  : sinkErr);
      }
      return false;
    }
  }

  return true;
}

std::string ExtractRawTailAfterCmd(const std::string &line,
                                   const std::string &cmd) {
  std::size_t pos = 0;
  // The cmd is the first token.
  if (line.rfind(cmd, 0) == 0) {
    pos = cmd.size();
  } else {
    // Fallback: find the first occurrence.
    pos = line.find(cmd);
    if (pos == std::string::npos)
      return {};
    pos += cmd.size();
  }
  while (pos < line.size() &&
         std::isspace(static_cast<unsigned char>(line[pos])))
    ++pos;
  return (pos < line.size()) ? line.substr(pos) : std::string();
}

std::string ErrorJson(const std::string &msg) {
  return std::string("{\"error\":\"") + JsonEscape(msg) + "\"}";
}

std::string
AppendWarningsToObjectJson(const std::string &objJson,
                           const std::vector<std::string> &warnings) {
  if (warnings.empty())
    return objJson;

  // Find last '}' (skip trailing whitespace).
  std::size_t end = objJson.size();
  while (end > 0 && std::isspace(static_cast<unsigned char>(objJson[end - 1])))
    --end;
  if (end == 0 || objJson[end - 1] != '}')
    return objJson;

  const std::size_t close = end - 1;
  const std::size_t open = objJson.find('{');
  if (open == std::string::npos || open >= close)
    return objJson;

  bool isEmpty = true;
  for (std::size_t i = open + 1; i < close; ++i) {
    if (!std::isspace(static_cast<unsigned char>(objJson[i]))) {
      isEmpty = false;
      break;
    }
  }

  std::ostringstream w;
  w << "\"warnings\":[";
  for (std::size_t i = 0; i < warnings.size(); ++i) {
    if (i)
      w << ',';
    w << "\"" << JsonEscape(warnings[i]) << "\"";
  }
  w << ']';

  std::string out;
  out.reserve(objJson.size() + warnings.size() * 32);
  out.append(objJson.substr(0, close));
  out.append(isEmpty ? "" : ",");
  out.append(w.str());
  out.push_back('}');
  out.append(objJson.substr(end));
  return out;
}

} // namespace

int main(int argc, char **argv) {
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  if (HasArg(argc, argv, "--version") || HasArg(argc, argv, "-v")) {
    std::printf("studiocastd %s (%s)\n", STUDIOCAST_VERSION,
                STUDIOCAST_GIT_SHA);
    return 0;
  }

  if (HasArg(argc, argv, "--help") || HasArg(argc, argv, "-h")) {
    Usage(argv[0]);
    return 0;
  }

  // Load persisted config and then apply CLI overrides for this run.
  auto daemonCfg = studiocast::config::LoadDaemonConfig();
#if STUDIOCAST_ENABLE_OPEN_VULKAN
  std::string vulkanSelectionError;
  if (!ConfigureProcessVulkanSelection(daemonCfg, &vulkanSelectionError)) {
    std::cerr << "WARN: invalid Vulkan adapter selection: "
              << vulkanSelectionError << "\n";
  }
#endif
  studiocast::video::VirtualCameraServiceConfig cfg =
      studiocast::config::ToVideoServiceConfig(daemonCfg);
  studiocast::audio::VirtualAudioServiceConfig acfg =
      studiocast::config::ToAudioServiceConfig(daemonCfg);

  if (const auto v = GetArgValue(argc, argv, "--input"); !v.empty())
    cfg.pipeline.input_device = v;
  if (const auto v = GetArgValue(argc, argv, "--output"); !v.empty())
    cfg.pipeline.output_device = v;

  bool capture_mode_explicit = false;
  if (const auto v = GetArgValue(argc, argv, "--capture-mode"); !v.empty()) {
    const auto t = ToLowerAscii(v);
    if (t == "auto" || t == "auto_best" || t == "autobest") {
      cfg.pipeline.capture_mode = studiocast::video::CaptureMode::auto_best;
      capture_mode_explicit = true;
    } else if (t == "requested") {
      cfg.pipeline.capture_mode = studiocast::video::CaptureMode::requested;
      capture_mode_explicit = true;
    } else {
      std::cerr << "WARN: unknown --capture-mode value: " << v
                << " (expected requested|auto)\n";
    }
  }

  cfg.pipeline.width = GetArgInt(argc, argv, "--width", cfg.pipeline.width);
  cfg.pipeline.height = GetArgInt(argc, argv, "--height", cfg.pipeline.height);
  cfg.pipeline.fps = GetArgInt(argc, argv, "--fps", cfg.pipeline.fps);
  if (const auto v = GetArgValue(argc, argv, "--output-format"); !v.empty()) {
    if (const auto parsed = studiocast::video::ParsePixelFormat(v)) {
      cfg.pipeline.output_format = *parsed;
    } else {
      std::cerr << "WARN: unknown --output-format value: " << v
                << " (expected rgb24|yuyv)\n";
    }
  }

  // Convenience: if the user sets a sentinel width/height and didn't explicitly
  // set a capture mode, treat it as capture auto.
  if (!capture_mode_explicit &&
      (cfg.pipeline.width <= 0 || cfg.pipeline.height <= 0)) {
    cfg.pipeline.capture_mode = studiocast::video::CaptureMode::auto_best;
  }

  if (HasArg(argc, argv, "--mirror"))
    cfg.pipeline.effects.mirror = true;

  // Legacy CLI flags: map to canonical Broadcast schema.
  if (const auto v = GetArgValue(argc, argv, "--background"); !v.empty()) {
    studiocast::video::effects::VirtualBackgroundMode mode{};
    if (studiocast::video::effects::ParseVirtualBackgroundMode(v, &mode)) {
      cfg.pipeline.effects.virtual_background.mode = mode;
      if (mode != studiocast::video::effects::VirtualBackgroundMode::none) {
        cfg.pipeline.effects.auto_frame.enabled = false;
      }
    } else if (v == "auto_frame" || v == "autoframe") {
      cfg.pipeline.effects.auto_frame.enabled = true;
      cfg.pipeline.effects.virtual_background.mode =
          studiocast::video::effects::VirtualBackgroundMode::none;
    } else {
      std::cerr << "WARN: unknown --background value: " << v << "\n";
    }
  }
  if (const auto v = GetArgValue(argc, argv, "--background-backend");
      !v.empty()) {
    studiocast::video::effects::EffectsEnginePreference eng{};
    if (studiocast::video::effects::ParseEffectsEnginePreference(v, &eng)) {
      cfg.pipeline.effects.engine = eng;
    } else {
      std::cerr << "WARN: unknown --background-backend value: " << v
                << " (expected auto|maxine|open_cuda)\n";
    }
  }
  if (const int v = GetArgInt(argc, argv, "--background-strength", -1); v > 0) {
    cfg.pipeline.effects.virtual_background.strength =
        std::max(1, std::min(64, v));
  }

  if (const auto v = GetArgValue(argc, argv, "--background-remove-color");
      !v.empty()) {
    // Canonical form is "#RRGGBB". Accept legacy formats too.
    std::uint32_t rgb = 0;
    if (ParseRgbHex(v, &rgb)) {
      cfg.pipeline.effects.virtual_background.remove_color = FormatRgbHex(rgb);
    } else {
      std::cerr
          << "WARN: invalid --background-remove-color (expected #RRGGBB): " << v
          << "\n";
    }
  }
  if (const auto v = GetArgValue(argc, argv, "--background-replace-image");
      !v.empty()) {
    cfg.pipeline.effects.virtual_background.replace_path = v;
  }

  if (const int v = GetArgInt(argc, argv, "--poll-ms", -1); v > 0)
    cfg.consumer_poll_ms = v;
  if (const int v = GetArgInt(argc, argv, "--start-grace-ms", -1); v >= 0)
    cfg.start_grace_ms = v;
  if (const int v = GetArgInt(argc, argv, "--stop-grace-ms", -1); v > 0)
    cfg.stop_grace_ms = v;
  if (const int v = GetArgInt(argc, argv, "--min-run-ms", -1); v >= 0)
    cfg.min_run_ms = v;
  if (HasArg(argc, argv, "--always-on"))
    cfg.always_on = true;

  if (cfg.pipeline.output_device.empty()) {
    // If possible, pre-fill output for a nicer startup experience.
    cfg.pipeline.output_device = ChooseWritableLoopbackDevice();
  }

  const auto rep = studiocast::video::ProbeLoopback();
  if (!rep.ReadyForVirtualCamera()) {
    std::cout << rep.ToText() << "\n\n";
  }

  std::cout << "studiocastd " << STUDIOCAST_VERSION << " ("
            << STUDIOCAST_GIT_SHA << ")\n";
  std::cout << "Virtual camera supervisor started. Press Ctrl+C to stop.\n\n";

  studiocast::video::VirtualCameraService svc;
  std::string err;
  if (!svc.Start(cfg, &err)) {
    std::cerr << "ERROR: " << err << "\n";
    return 1;
  }

  studiocast::audio::VirtualAudioService audioSvc;
  std::string aerr;
  if (!audioSvc.Start(acfg, &aerr)) {
    std::cerr << "ERROR: " << aerr << "\n";
    svc.Stop();
    return 1;
  }

  // Start IPC server for GUI / studiocastctl.
  studiocast::ipc::DaemonServer server;
  std::string sockErr;
  const auto socketPath = studiocast::ipc::DaemonSocketPath(&sockErr);
  if (socketPath.empty()) {
    std::cerr << "ERROR: failed to compute socket path: " << sockErr << "\n";
    svc.Stop();
    audioSvc.Stop();
    return 2;
  }

  std::mutex controlMu;

  std::string serverErr;
  if (!server.Start(
          socketPath,
          [&](const std::string &line) -> std::string {
            const auto pc = ParseLine(line);
            const auto persistVideo =
                [&](const studiocast::video::VirtualCameraServiceConfig &newCfg,
                    std::string *error) -> bool {
              auto nextDaemonCfg = daemonCfg;
              studiocast::config::ApplyVideoServiceConfigToDaemonConfig(
                  newCfg, &nextDaemonCfg);
              if (!studiocast::config::SaveDaemonConfig(nextDaemonCfg, error))
                return false;
              daemonCfg = nextDaemonCfg;
              return true;
            };
            const auto persistAudio =
                [&](const studiocast::audio::VirtualAudioServiceConfig &newCfg,
                    std::string *error) -> bool {
              auto nextDaemonCfg = daemonCfg;
              studiocast::config::ApplyAudioServiceConfigToDaemonConfig(
                  newCfg, &nextDaemonCfg);
              if (!studiocast::config::SaveDaemonConfig(nextDaemonCfg, error))
                return false;
              daemonCfg = nextDaemonCfg;
              return true;
            };

            if (pc.cmd == "PING") {
              return std::string("OK {\"pong\":true}");
            }

            if (pc.cmd == "GET_STATUS") {
              const auto st = svc.Status();
              const auto current = svc.Config();

              const auto ast = audioSvc.Status();
              const auto acurrent = audioSvc.Config();
              const DiagnosticsJsonSnapshot diagnostics =
                  GetDiagnosticsJsonCacheSnapshot();

              return std::string("OK ") +
                     StatusToJson(st, current, ast, acurrent, socketPath,
                                  diagnostics.maxine, diagnostics.open_cuda,
                                  diagnostics.open_vulkan,
                                  diagnostics.open_audio, diagnostics.loopback,
                                  &daemonCfg);
            }

            if (pc.cmd == "GET_DIAGNOSTICS" ||
                pc.cmd == "REFRESH_DIAGNOSTICS") {
              if (pc.cmd == "REFRESH_DIAGNOSTICS")
                audioSvc.RefreshPreparation();
              const DiagnosticsJsonSnapshot diagnostics =
                  RefreshDiagnosticsJsonCache();
              return std::string("OK ") +
                     DiagnosticsJsonSnapshotToJson(diagnostics);
            }

            if (pc.cmd == "GET_CONFIG") {
              const auto current = svc.Config();
              return std::string("OK ") +
                     studiocast::video::BroadcastCameraEffectsContractToJson(
                         current.pipeline.effects);
            }

            if (pc.cmd == "GET_AUDIO_CONFIG") {
              const auto current = audioSvc.Config();
              return std::string("OK ") + AudioConfigToJson(current);
            }

            if (pc.cmd == "SET_ENABLED") {
              bool enabled = true;
              bool ok = false;
              if (auto it = pc.kv.find("enabled"); it != pc.kv.end()) {
                ok = ParseBoolArg(it->second, &enabled);
              } else if (!pc.args.empty()) {
                ok = ParseBoolArg(pc.args[0], &enabled);
              }

              if (!ok) {
                return std::string("ERR ") +
                       ErrorJson("SET_ENABLED requires enabled=0|1");
              }

              std::lock_guard<std::mutex> lock(controlMu);
              auto newCfg = svc.Config();
              newCfg.enabled = enabled;

              std::string perr;
              if (!persistVideo(newCfg, &perr)) {
                return std::string("ERR ") +
                       ErrorJson("failed to save config: " + perr);
              }

              svc.UpdateConfig(newCfg);

              return std::string("OK {\"enabled\":") + BoolJson(enabled) + "}";
            }

            if (pc.cmd == "AUDIO_START") {
              std::lock_guard<std::mutex> lock(controlMu);
              auto newCfg = audioSvc.Config();
              newCfg.enabled = true;

              std::vector<std::string> warnings;
              std::string verr;
              if (!ValidateAudioConfigSafetyForDaemon(newCfg, &warnings,
                                                      &verr)) {
                return std::string("ERR ") +
                       ErrorJson(verr.empty() ? "invalid audio config" : verr);
              }

              std::string perr;
              if (!persistAudio(newCfg, &perr)) {
                return std::string("ERR ") +
                       ErrorJson("failed to save config: " + perr);
              }

              audioSvc.UpdateConfig(newCfg);

              return std::string("OK ") +
                     AppendWarningsToObjectJson("{\"enabled\":true}", warnings);
            }

            if (pc.cmd == "AUDIO_STOP") {
              std::lock_guard<std::mutex> lock(controlMu);
              auto newCfg = audioSvc.Config();
              newCfg.enabled = false;

              std::string perr;
              if (!persistAudio(newCfg, &perr)) {
                return std::string("ERR ") +
                       ErrorJson("failed to save config: " + perr);
              }

              audioSvc.UpdateConfig(newCfg);

              return std::string("OK {\"enabled\":false}");
            }

            if (pc.cmd == "SET_AUDIO_CONFIG") {
              const std::string jsonText = ExtractRawTailAfterCmd(line, pc.cmd);
              if (jsonText.empty()) {
                return std::string("ERR ") +
                       ErrorJson(
                           "SET_AUDIO_CONFIG requires a JSON object argument");
              }

              std::lock_guard<std::mutex> lock(controlMu);
              auto newCfg = audioSvc.Config();

              std::vector<std::string> warnings;
              std::string jerr;
              if (!ApplyAudioConfigPatchJsonText(jsonText, &newCfg, &warnings,
                                                 &jerr)) {
                return std::string("ERR ") +
                       ErrorJson(jerr.empty() ? "invalid audio config JSON"
                                              : jerr);
              }
              if (!ValidateAudioConfigSafetyForDaemon(newCfg, &warnings,
                                                      &jerr)) {
                return std::string("ERR ") +
                       ErrorJson(jerr.empty() ? "invalid audio config" : jerr);
              }

              std::string perr;
              if (!persistAudio(newCfg, &perr)) {
                return std::string("ERR ") +
                       ErrorJson("failed to save config: " + perr);
              }

              audioSvc.UpdateConfig(newCfg);

              return std::string("OK ") +
                     AppendWarningsToObjectJson(AudioConfigToJson(newCfg),
                                                warnings);
            }

            if (pc.cmd == "SET_VIDEO_CONFIG") {
              std::lock_guard<std::mutex> lock(controlMu);
              auto newCfg = svc.Config();
              std::string nextVulkanDevice = daemonCfg.video_vulkan_device;
              bool nextVulkanAllowCpu = daemonCfg.video_vulkan_allow_cpu;
              bool vulkanSelectionChanged = false;

              if (auto it = pc.kv.find("input"); it != pc.kv.end()) {
                newCfg.pipeline.input_device =
                    (it->second == "auto") ? std::string() : it->second;
              }
              if (auto it = pc.kv.find("output"); it != pc.kv.end()) {
                newCfg.pipeline.output_device =
                    (it->second == "auto") ? std::string() : it->second;
              }
              if (auto it = pc.kv.find("width"); it != pc.kv.end()) {
                newCfg.pipeline.width = std::atoi(it->second.c_str());
              }
              if (auto it = pc.kv.find("height"); it != pc.kv.end()) {
                newCfg.pipeline.height = std::atoi(it->second.c_str());
              }
              if (auto it = pc.kv.find("fps"); it != pc.kv.end()) {
                newCfg.pipeline.fps = std::atoi(it->second.c_str());
              }
              if (auto it = pc.kv.find("output_format"); it != pc.kv.end()) {
                const auto parsed =
                    studiocast::video::ParsePixelFormat(it->second);
                if (!parsed) {
                  return std::string("ERR ") +
                         ErrorJson("output_format must be rgb24|yuyv");
                }
                newCfg.pipeline.output_format = *parsed;
              }
              if (auto it = pc.kv.find("compute_backend");
                  it != pc.kv.end()) {
                studiocast::video::ComputeBackendPreference pref;
                if (!studiocast::video::ParseComputeBackendPreference(
                        it->second, &pref)) {
                  return std::string("ERR ") +
                         ErrorJson("compute_backend must be "
                                   "auto|cpu|cuda|vulkan");
                }
                newCfg.pipeline.compute_backend = pref;
              }
              if (auto it = pc.kv.find("vulkan_device"); it != pc.kv.end()) {
                if (it->second != "auto" &&
                    !IsValidVulkanStableSelector(it->second)) {
                  return std::string("ERR ") +
                         ErrorJson("vulkan_device must be auto or a stable "
                                   "v1:... identity from daemon status");
                }
                nextVulkanDevice = it->second;
                vulkanSelectionChanged =
                    nextVulkanDevice != daemonCfg.video_vulkan_device;
              }
              if (auto it = pc.kv.find("vulkan_allow_cpu");
                  it != pc.kv.end()) {
                bool v = false;
                if (!ParseBoolArg(it->second, &v)) {
                  return std::string("ERR ") +
                         ErrorJson("vulkan_allow_cpu expects "
                                   "0|1|true|false");
                }
                vulkanSelectionChanged =
                    vulkanSelectionChanged ||
                    v != daemonCfg.video_vulkan_allow_cpu;
                nextVulkanAllowCpu = v;
              }
              if (auto it = pc.kv.find("always_on"); it != pc.kv.end()) {
                bool v = false;
                if (!ParseBoolArg(it->second, &v)) {
                  return std::string("ERR ") +
                         ErrorJson("always_on expects 0|1|true|false");
                }
                newCfg.always_on = v;
              }
              if (auto it = pc.kv.find("allow_cpu_resize"); it != pc.kv.end()) {
                bool v = false;
                if (!ParseBoolArg(it->second, &v)) {
                  return std::string("ERR ") +
                         ErrorJson("allow_cpu_resize expects 0|1|true|false");
                }
                newCfg.pipeline.allow_cpu_resize = v;
              }

              auto nextDaemonCfg = daemonCfg;
              studiocast::config::ApplyVideoServiceConfigToDaemonConfig(
                  newCfg, &nextDaemonCfg);
              nextDaemonCfg.video_vulkan_device = nextVulkanDevice;
              nextDaemonCfg.video_vulkan_allow_cpu = nextVulkanAllowCpu;
              std::string perr;
              if (!studiocast::config::SaveDaemonConfig(nextDaemonCfg,
                                                        &perr)) {
                return std::string("ERR ") +
                       ErrorJson("failed to save config: " + perr);
              }
              daemonCfg = nextDaemonCfg;

              svc.UpdateConfig(newCfg);

#if STUDIOCAST_ENABLE_OPEN_VULKAN
              if (vulkanSelectionChanged) {
                std::string selectionError;
                if (!ConfigureProcessVulkanSelection(daemonCfg,
                                                     &selectionError)) {
                  std::cerr << "WARN: invalid Vulkan compatibility selector: "
                            << selectionError << "\n";
                }
                (void)RefreshDiagnosticsJsonCache();
              }
#else
              (void)vulkanSelectionChanged;
#endif

              return std::string("OK ") + ConfigToJson(newCfg, &daemonCfg);
            }

            if (pc.cmd == "SET_VIDEO_EFFECTS_JSON") {
              const std::string jsonText = ExtractRawTailAfterCmd(line, pc.cmd);
              if (jsonText.empty()) {
                return std::string("ERR ") +
                       ErrorJson("SET_VIDEO_EFFECTS_JSON requires a JSON "
                                 "object argument");
              }

              std::lock_guard<std::mutex> lock(controlMu);
              auto newCfg = svc.Config();

              std::vector<std::string> warnings;

              std::string jerr;
              auto bfx = newCfg.pipeline.effects;

              if (!studiocast::video::ApplyBroadcastCameraEffectsPatchJsonText(
                      jsonText, &bfx, &jerr)) {
                // Legacy compatibility: accept old `CameraEffects` JSON
                // patches, but warn. Note: explicit requests for backend `cpu`
                // are rejected by the legacy parser.
                std::string lerr;
                auto legacy = studiocast::video::ToLegacyCameraEffects(bfx);
                if (!studiocast::video::ApplyCameraEffectsPatchJsonText(
                        jsonText, &legacy, &lerr)) {
                  std::string msg =
                      jerr.empty() ? "invalid effects JSON" : jerr;
                  if (!lerr.empty()) {
                    msg += "; legacy parse: " + lerr;
                  }
                  return std::string("ERR ") + ErrorJson(msg);
                }
                warnings.emplace_back(
                    "Legacy effects JSON accepted; please migrate to the "
                    "Broadcast effects schema (video_effects).");
                bfx = studiocast::video::ToBroadcastCameraEffects(legacy);
              }
              newCfg.pipeline.effects = bfx;

              std::string perr;
              if (!persistVideo(newCfg, &perr)) {
                return std::string("ERR ") +
                       ErrorJson("failed to save config: " + perr);
              }

              svc.UpdateConfig(newCfg);

              return std::string("OK ") +
                     AppendWarningsToObjectJson(
                         ConfigToJson(newCfg, &daemonCfg), warnings);
            }

            if (pc.cmd == "SET_VIDEO_EFFECTS") {
              std::lock_guard<std::mutex> lock(controlMu);
              auto newCfg = svc.Config();

              std::vector<std::string> warnings;
              warnings.emplace_back("SET_VIDEO_EFFECTS is deprecated; use "
                                    "SET_VIDEO_EFFECTS_JSON");

              auto bfx = newCfg.pipeline.effects;

              if (auto it = pc.kv.find("mirror"); it != pc.kv.end()) {
                bool mirror = false;
                if (!ParseBoolArg(it->second, &mirror)) {
                  return std::string("ERR ") + ErrorJson("mirror must be 0|1");
                }
                bfx.mirror = mirror;
              }

              if (auto it = pc.kv.find("background"); it != pc.kv.end()) {
                const auto v = it->second;
                studiocast::video::effects::VirtualBackgroundMode mode{};
                if (studiocast::video::effects::ParseVirtualBackgroundMode(
                        v, &mode)) {
                  bfx.virtual_background.mode = mode;
                  if (mode !=
                      studiocast::video::effects::VirtualBackgroundMode::none) {
                    bfx.auto_frame.enabled = false;
                  }
                } else if (v == "auto_frame" || v == "autoframe") {
                  bfx.auto_frame.enabled = true;
                  bfx.virtual_background.mode =
                      studiocast::video::effects::VirtualBackgroundMode::none;
                } else {
                  return std::string("ERR ") +
                         ErrorJson("background must be "
                                   "none|blur|remove|replace|auto_frame");
                }
              }

              if (auto it = pc.kv.find("background_backend");
                  it != pc.kv.end()) {
                // Deprecated flat field: map to canonical engine preference.
                warnings.emplace_back(
                    "background_backend is deprecated; use engine");

                {
                  std::string v = it->second;
                  std::transform(v.begin(), v.end(), v.begin(),
                                 [](unsigned char c) {
                                   return static_cast<char>(std::tolower(c));
                                 });
                  if (v == "cpu") {
                    return std::string("ERR ") +
                           ErrorJson("backend 'cpu' is not supported");
                  }
                }

                studiocast::video::effects::EffectsEnginePreference eng{};
                if (!studiocast::video::effects::ParseEffectsEnginePreference(
                        it->second, &eng)) {
                  return std::string("ERR ") +
                         ErrorJson("background_backend must be "
                                   "auto|maxine|open_cuda");
                }
                bfx.engine = eng;
              }

              if (auto it = pc.kv.find("background_strength");
                  it != pc.kv.end()) {
                const int v = std::atoi(it->second.c_str());
                if (v <= 0) {
                  return std::string("ERR ") +
                         ErrorJson(
                             "background_strength must be a positive integer");
                }
                bfx.virtual_background.strength = std::max(1, std::min(64, v));
              }

              if (auto it = pc.kv.find("background_remove_color");
                  it != pc.kv.end()) {
                std::uint32_t rgb = 0;
                if (!ParseRgbHex(it->second, &rgb)) {
                  return std::string("ERR ") +
                         ErrorJson("background_remove_color must be #RRGGBB");
                }
                bfx.virtual_background.remove_color = FormatRgbHex(rgb);
              }

              if (auto it = pc.kv.find("background_replace_image");
                  it != pc.kv.end()) {
                bfx.virtual_background.replace_path = it->second;
              }

              if (auto it = pc.kv.find("virtual_key_light");
                  it != pc.kv.end()) {
                bool en = false;
                if (!ParseBoolArg(it->second, &en)) {
                  return std::string("ERR ") +
                         ErrorJson("virtual_key_light must be 0|1");
                }
                bfx.virtual_key_light.enabled = en;
              }

              if (auto it = pc.kv.find("virtual_key_light_intensity");
                  it != pc.kv.end()) {
                const int v = std::atoi(it->second.c_str());
                if (v < 0 || v > 100) {
                  return std::string("ERR ") +
                         ErrorJson(
                             "virtual_key_light_intensity must be 0..100");
                }
                bfx.virtual_key_light.intensity = v;
              }

              if (auto it = pc.kv.find("virtual_key_light_temperature");
                  it != pc.kv.end()) {
                const int preset =
                    ParseKeyLightTemperaturePreset(it->second, -1);
                if (preset < 0) {
                  return std::string("ERR ") +
                         ErrorJson("virtual_key_light_temperature must be "
                                   "neutral|warm|cool");
                }
                bfx.virtual_key_light.temperature_preset = preset;
                // Match KelvinFromPreset() in broadcast_effects_json.cpp.
                switch (preset) {
                case 1:
                  bfx.virtual_key_light.temperature = 3200;
                  break;
                case 2:
                  bfx.virtual_key_light.temperature = 6500;
                  break;
                default:
                  bfx.virtual_key_light.temperature = 4500;
                  break;
                }
              }

              if (auto it = pc.kv.find("virtual_key_light_pan");
                  it != pc.kv.end()) {
                const int v = std::atoi(it->second.c_str());
                bfx.virtual_key_light.direction_pan_degrees =
                    std::max(-180, std::min(180, v));
              }

              if (auto it = pc.kv.find("virtual_key_light_hdri");
                  it != pc.kv.end()) {
                bfx.virtual_key_light.hdri_path = it->second;
              }

              if (auto it = pc.kv.find("vignette"); it != pc.kv.end()) {
                bool en = false;
                if (!ParseBoolArg(it->second, &en)) {
                  return std::string("ERR ") +
                         ErrorJson("vignette must be 0|1");
                }
                bfx.vignette.enabled = en;
              }

              if (auto it = pc.kv.find("vignette_intensity");
                  it != pc.kv.end()) {
                const int v = std::atoi(it->second.c_str());
                if (v < 0 || v > 100) {
                  return std::string("ERR ") +
                         ErrorJson("vignette_intensity must be 0..100");
                }
                bfx.vignette.intensity = v;
              }

              if (auto it = pc.kv.find("vignette_center_on_face");
                  it != pc.kv.end()) {
                bool en = false;
                if (!ParseBoolArg(it->second, &en)) {
                  return std::string("ERR ") +
                         ErrorJson("vignette_center_on_face must be 0|1");
                }
                bfx.vignette.center_on_tracked_face = en;
              }

              // Persist via canonical schema.
              newCfg.pipeline.effects = bfx;

              std::string perr;
              if (!persistVideo(newCfg, &perr)) {
                return std::string("ERR ") +
                       ErrorJson("failed to save config: " + perr);
              }

              svc.UpdateConfig(newCfg);

              return std::string("OK ") +
                     AppendWarningsToObjectJson(
                         ConfigToJson(newCfg, &daemonCfg), warnings);
            }

            return std::string("ERR ") + ErrorJson("unknown_command");
          },
          &serverErr)) {
    std::cerr << "ERROR: failed to start IPC server: " << serverErr << "\n";
    svc.Stop();
    audioSvc.Stop();
    return 3;
  }

  studiocast::video::VirtualCameraServiceStatus prev;

  // Print initial status once so users immediately see which devices were
  // selected (especially when running in auto mode).
  {
    const auto st = svc.Status();
    std::cout << "[status] consumers=" << st.consumer_count << " running="
              << (st.pipeline.running
                      ? "yes"
                      : (st.pipeline.starting ? "starting" : "no"))
              << " in="
              << (st.pipeline.input_device.empty() ? "(auto)"
                                                   : st.pipeline.input_device)
              << " out="
              << (st.pipeline.output_device.empty() ? "(auto)"
                                                    : st.pipeline.output_device)
              << "\n";
    if (!st.last_error.empty()) {
      std::cout << "[last_error] " << st.last_error << "\n";
    }
    std::cout.flush();
    prev = st;
  }

  while (g_running.load()) {
    const auto st = svc.Status();

    // Print state transitions.
    if (st.consumer_present != prev.consumer_present ||
        st.pipeline.running != prev.pipeline.running ||
        st.pipeline.starting != prev.pipeline.starting ||
        st.pipeline.input_device != prev.pipeline.input_device ||
        st.pipeline.output_device != prev.pipeline.output_device ||
        st.last_error != prev.last_error) {

      std::cout << "[status] consumers=" << st.consumer_count << " running="
                << (st.pipeline.running
                        ? "yes"
                        : (st.pipeline.starting ? "starting" : "no"))
                << " in="
                << (st.pipeline.input_device.empty() ? "(auto)"
                                                     : st.pipeline.input_device)
                << " out="
                << (st.pipeline.output_device.empty()
                        ? "(auto)"
                        : st.pipeline.output_device)
                << "\n";

      if (!st.last_error.empty()) {
        std::cout << "[last_error] " << st.last_error << "\n";
      }

      std::cout.flush();
      prev = st;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }

  std::cout << "\nStopping...\n";
  server.Stop();
  svc.Stop();
  audioSvc.Stop();
  return 0;
}
