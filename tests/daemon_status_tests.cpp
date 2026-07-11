#include <filesystem>
#include <iostream>
#include <string>

#define main studiocastd_main_disabled_for_tests
#include "daemon/studiocastd_main.cpp"
#undef main

namespace {

bool Expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

const JsonObject *ObjectAt(const JsonObject &obj, const std::string &key,
                           const char *message) {
  const JsonObject *child = JsonObjectField(obj, key);
  if (!child)
    std::cerr << message << "\n";
  return child;
}

const std::string *StringAt(const JsonObject &obj, const std::string &key,
                            const char *message) {
  const std::string *value = JsonStringField(obj, key);
  if (!value)
    std::cerr << message << "\n";
  return value;
}

const JsonArray *ArrayAt(const JsonObject &obj, const std::string &key,
                         const char *message) {
  const auto it = obj.find(key);
  if (it == obj.end()) {
    std::cerr << message << "\n";
    return nullptr;
  }
  const JsonArray *value = it->second.AsArray();
  if (!value)
    std::cerr << message << "\n";
  return value;
}

struct ReadinessFields {
  std::string state;
  std::string reason;
  std::string backend;
  bool present = false;
};

ReadinessFields ReadinessEntryFor(const std::string &statusJson,
                                  const std::string &effectId) {
  ReadinessFields out;
  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(statusJson, &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return out;
  }

  const JsonObject *root = rootValue.AsObject();
  if (!root) {
    std::cerr << "status root should be an object\n";
    return out;
  }

  const JsonObject *video = ObjectAt(*root, "video", "video should exist");
  if (!video)
    return out;
  const JsonObject *readiness =
      ObjectAt(*video, "effect_readiness", "effect_readiness should exist");
  if (!readiness)
    return out;
  const JsonObject *entry = ObjectAt(*readiness, effectId.c_str(),
                                     "effect readiness entry should exist");
  if (!entry)
    return out;

  const std::string *state =
      StringAt(*entry, "state", "readiness state should exist");
  const std::string *reason =
      StringAt(*entry, "reason", "readiness reason should exist");
  const std::string *backend =
      StringAt(*entry, "backend", "readiness backend should exist");
  if (!state || !reason || !backend)
    return out;

  out.state = *state;
  out.reason = *reason;
  out.backend = *backend;
  out.present = true;
  return out;
}

std::string StatusForEffects(
    const studiocast::video::effects::BroadcastCameraEffects &effects) {
  studiocast::video::VirtualCameraServiceStatus videoStatus;
  videoStatus.service_running = true;
  videoStatus.virtual_device_present = true;
  videoStatus.virtual_device_available = true;

  studiocast::video::VirtualCameraServiceConfig videoConfig;
  videoConfig.enabled = true;
  videoConfig.pipeline.effects = effects;

  studiocast::audio::VirtualAudioServiceStatus audioStatus;
  audioStatus.service_running = true;
  audioStatus.mic_present = true;
  audioStatus.speakers_present = true;

  studiocast::audio::VirtualAudioServiceConfig audioConfig;

  return StatusToJson(videoStatus, videoConfig, audioStatus, audioConfig,
                      std::filesystem::path("/tmp/studiocastd-test.sock"),
                      /*maxineJson=*/"", /*openCudaJson=*/"",
                      /*openVulkanJson=*/"",
                      /*openAudioJson=*/"", /*loopbackJson=*/"");
}

std::string StatusForVideoConfig(
    const studiocast::video::VirtualCameraServiceConfig &videoConfig) {
  studiocast::video::VirtualCameraServiceStatus videoStatus;
  videoStatus.service_running = true;
  videoStatus.virtual_device_present = true;
  videoStatus.virtual_device_available = true;

  studiocast::audio::VirtualAudioServiceStatus audioStatus;
  audioStatus.service_running = true;
  audioStatus.mic_present = true;
  audioStatus.speakers_present = true;

  studiocast::audio::VirtualAudioServiceConfig audioConfig;

  return StatusToJson(videoStatus, videoConfig, audioStatus, audioConfig,
                      std::filesystem::path("/tmp/studiocastd-test.sock"),
                      /*maxineJson=*/"", /*openCudaJson=*/"",
                      /*openVulkanJson=*/"",
                      /*openAudioJson=*/"", /*loopbackJson=*/"");
}

std::string StatusForVideoConfigWithDiagnostics(
    const studiocast::video::VirtualCameraServiceConfig &videoConfig,
    const std::string &maxineJson, const std::string &openCudaJson,
    const std::string &openVulkanJson) {
  studiocast::video::VirtualCameraServiceStatus videoStatus;
  videoStatus.service_running = true;
  videoStatus.virtual_device_present = true;
  videoStatus.virtual_device_available = true;

  studiocast::audio::VirtualAudioServiceStatus audioStatus;
  audioStatus.service_running = true;
  audioStatus.mic_present = true;
  audioStatus.speakers_present = true;

  studiocast::audio::VirtualAudioServiceConfig audioConfig;

  return StatusToJson(videoStatus, videoConfig, audioStatus, audioConfig,
                      std::filesystem::path("/tmp/studiocastd-test.sock"),
                      maxineJson, openCudaJson, openVulkanJson,
                      /*openAudioJson=*/"", /*loopbackJson=*/"");
}

bool TestVideoStatusReportsAllowCpuResize() {
  studiocast::video::VirtualCameraServiceConfig videoConfig;
  videoConfig.pipeline.allow_cpu_resize = false;

  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(StatusForVideoConfig(videoConfig),
                                     &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return false;
  }

  const JsonObject *root = rootValue.AsObject();
  if (!root) {
    std::cerr << "status root should be an object\n";
    return false;
  }

  const JsonObject *video = ObjectAt(*root, "video", "video should exist");
  if (!video)
    return false;

  return Expect(!JsonBoolField(*video, "allow_cpu_resize", true),
                "video status should report allow_cpu_resize=false");
}

bool TestVideoStatusReportsRequestedOutputFormat() {
  studiocast::video::VirtualCameraServiceConfig videoConfig;
  videoConfig.pipeline.output_format = studiocast::video::PixelFormat::yuyv;

  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(StatusForVideoConfig(videoConfig),
                                     &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return false;
  }

  const JsonObject *root = rootValue.AsObject();
  if (!root) {
    std::cerr << "status root should be an object\n";
    return false;
  }

  const JsonObject *video = ObjectAt(*root, "video", "video should exist");
  if (!video)
    return false;

  const std::string *requested =
      StringAt(*video, "output_format_requested",
               "video status should include requested output format");
  if (!requested)
    return false;

  const JsonObject *actual =
      ObjectAt(*video, "output_format", "actual output_format should exist");
  if (!actual)
    return false;

  studiocast::util::json::Value configValue;
  if (!studiocast::util::json::Parse(ConfigToJson(videoConfig), &configValue,
                                     &error)) {
    std::cerr << "config JSON should parse: " << error << "\n";
    return false;
  }
  const JsonObject *config = configValue.AsObject();
  if (!config) {
    std::cerr << "config root should be an object\n";
    return false;
  }
  const std::string *configRequested =
      StringAt(*config, "output_format_requested",
               "config should include requested output format");
  if (!configRequested)
    return false;

  return Expect(*requested == "yuyv",
                "video status should report yuyv requested output format") &&
         Expect(*configRequested == "yuyv",
                "video config should report yuyv requested output format") &&
         Expect(actual->find("width") != actual->end(),
                "actual output_format should remain negotiated object");
}

bool TestVideoStatusReportsComputeBackend() {
  studiocast::video::VirtualCameraServiceConfig videoConfig;
  videoConfig.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::vulkan;
  videoConfig.pipeline.effects.virtual_background.mode =
      studiocast::video::effects::VirtualBackgroundMode::blur;

  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(StatusForVideoConfig(videoConfig),
                                     &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return false;
  }

  const JsonObject *root = rootValue.AsObject();
  if (!root) {
    std::cerr << "status root should be an object\n";
    return false;
  }

  const JsonObject *video = ObjectAt(*root, "video", "video should exist");
  if (!video)
    return false;
  const JsonObject *compute =
      ObjectAt(*video, "compute", "video.compute should exist");
  if (!compute)
    return false;

  const std::string *preference =
      StringAt(*compute, "preference", "compute preference should exist");
  const std::string *resolved = StringAt(
      *compute, "resolved_backend", "compute resolved backend should exist");
  const std::string *active =
      StringAt(*compute, "active_backend", "compute active backend should exist");
  const std::string *fallback =
      StringAt(*compute, "fallback_reason", "compute fallback should exist");
  const std::string *degraded =
      StringAt(*compute, "degraded_reason", "compute degraded should exist");
  if (!preference || !resolved || !active || !fallback || !degraded)
    return false;

  studiocast::util::json::Value configValue;
  if (!studiocast::util::json::Parse(ConfigToJson(videoConfig), &configValue,
                                     &error)) {
    std::cerr << "config JSON should parse: " << error << "\n";
    return false;
  }
  const JsonObject *config = configValue.AsObject();
  if (!config) {
    std::cerr << "config root should be an object\n";
    return false;
  }
  const std::string *configBackend =
      StringAt(*config, "compute_backend",
               "config should include compute_backend");
  if (!configBackend)
    return false;

  return Expect(*preference == "vulkan",
                "compute preference should report vulkan") &&
         Expect(*resolved == "cpu",
                "vulkan unavailable status should resolve to cpu") &&
         Expect(*active == "cpu",
                "vulkan unavailable status should be active cpu") &&
         Expect(fallback->find("Vulkan compute backend requested") !=
                    std::string::npos,
                "vulkan unavailable status should include fallback reason") &&
         Expect(*degraded == *fallback,
                "vulkan unavailable degraded reason should match fallback") &&
         Expect(*configBackend == "vulkan",
                "config JSON should report vulkan compute_backend");
}

bool TestVideoConfigMapsComputeBackendPreference() {
  studiocast::config::DaemonConfig daemonConfig;
  daemonConfig.video_compute_backend = "cuda";
  auto runtime = studiocast::config::ToVideoServiceConfig(daemonConfig);
  if (runtime.pipeline.compute_backend !=
      studiocast::video::ComputeBackendPreference::cuda) {
    std::cerr << "daemon config should map video.compute.backend=cuda into "
                 "runtime config\n";
    return false;
  }

  runtime.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::vulkan;
  studiocast::config::ApplyVideoServiceConfigToDaemonConfig(runtime,
                                                            &daemonConfig);
  return Expect(daemonConfig.video_compute_backend == "vulkan",
                "runtime config should persist compute backend preference");
}

bool TestVideoStatusReportsCaptureFallbackState() {
  studiocast::video::VirtualCameraServiceStatus videoStatus;
  videoStatus.service_running = true;
  videoStatus.virtual_device_present = true;
  videoStatus.virtual_device_available = true;
  videoStatus.pipeline.running = true;
  videoStatus.pipeline.capture_fallback_state =
      "raw_after_mjpeg_decode_failure";
  videoStatus.pipeline.capture_fallback_reason =
      "MJPEG decode failed: synthetic bad frame.";

  studiocast::video::VirtualCameraServiceConfig videoConfig;
  videoConfig.enabled = true;

  studiocast::audio::VirtualAudioServiceStatus audioStatus;
  audioStatus.service_running = true;
  audioStatus.mic_present = true;

  studiocast::audio::VirtualAudioServiceConfig audioConfig;

  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(
          StatusToJson(videoStatus, videoConfig, audioStatus, audioConfig,
                       std::filesystem::path("/tmp/studiocastd-test.sock"),
                       /*maxineJson=*/"", /*openCudaJson=*/"",
                       /*openVulkanJson=*/"",
                       /*openAudioJson=*/"", /*loopbackJson=*/""),
          &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return false;
  }

  const JsonObject *root = rootValue.AsObject();
  if (!root) {
    std::cerr << "status root should be an object\n";
    return false;
  }
  const JsonObject *video = ObjectAt(*root, "video", "video should exist");
  if (!video)
    return false;
  const JsonObject *fallback =
      ObjectAt(*video, "capture_fallback", "capture_fallback should exist");
  if (!fallback)
    return false;

  const std::string *state =
      StringAt(*fallback, "state", "fallback state should exist");
  const std::string *reason =
      StringAt(*fallback, "reason", "fallback reason should exist");
  if (!state || !reason)
    return false;

  return Expect(*state == "raw_after_mjpeg_decode_failure",
                "status should report capture fallback state") &&
         Expect(reason->find("MJPEG decode failed") != std::string::npos,
                "status should report capture fallback reason");
}

bool TestVideoStatusReportsConfiguredDevicesWhenPipelineIdle() {
  studiocast::video::VirtualCameraServiceStatus videoStatus;
  videoStatus.service_running = true;
  videoStatus.virtual_device_present = true;
  videoStatus.virtual_device_available = true;

  studiocast::video::VirtualCameraServiceConfig videoConfig;
  videoConfig.enabled = true;
  videoConfig.pipeline.input_device = "/dev/video0";
  videoConfig.pipeline.output_device = "/dev/video10";

  studiocast::audio::VirtualAudioServiceStatus audioStatus;
  audioStatus.service_running = true;
  audioStatus.mic_present = true;

  studiocast::audio::VirtualAudioServiceConfig audioConfig;

  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(
          StatusToJson(videoStatus, videoConfig, audioStatus, audioConfig,
                       std::filesystem::path("/tmp/studiocastd-test.sock"),
                       /*maxineJson=*/"", /*openCudaJson=*/"",
                       /*openVulkanJson=*/"",
                       /*openAudioJson=*/"", /*loopbackJson=*/""),
          &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return false;
  }

  const JsonObject *root = rootValue.AsObject();
  if (!root) {
    std::cerr << "status root should be an object\n";
    return false;
  }
  const JsonObject *video = ObjectAt(*root, "video", "video should exist");
  if (!video)
    return false;

  const std::string *input =
      StringAt(*video, "input_device", "input_device should exist");
  const std::string *output =
      StringAt(*video, "output_device", "output_device should exist");
  if (!input || !output)
    return false;

  return Expect(*input == "/dev/video0",
                "idle video status should keep configured input device") &&
         Expect(*output == "/dev/video10",
                "idle video status should keep resolved output device");
}

bool TestExplicitOpenCudaEffectUnknownWithoutDiagnostics() {
  studiocast::video::effects::BroadcastCameraEffects effects;
  effects.engine =
      studiocast::video::effects::EffectsEnginePreference::open_cuda;
  effects.virtual_background.mode =
      studiocast::video::effects::VirtualBackgroundMode::blur;

  const ReadinessFields entry =
      ReadinessEntryFor(StatusForEffects(effects), "virtual_background.blur");
  if (!entry.present)
    return false;

  return Expect(entry.backend == "open_cuda",
                "explicit Open CUDA effect should keep backend attribution") &&
         Expect(entry.state == "unknown",
                "explicit Open CUDA readiness should be unknown when "
                "diagnostics are absent") &&
         Expect(entry.reason == "diagnostics_unavailable",
                "unknown readiness should explain missing diagnostics");
}

bool TestExplicitVulkanVirtualBackgroundReportsOpenVulkanBlocked() {
  studiocast::video::VirtualCameraServiceConfig videoConfig;
  videoConfig.enabled = true;
  videoConfig.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::vulkan;
  videoConfig.pipeline.effects.virtual_background.mode =
      studiocast::video::effects::VirtualBackgroundMode::blur;

  const std::string openVulkanJson =
      "{\"compiled_enabled\":true,\"ok\":true,"
      "\"available_effects\":[],"
      "\"blocked_effects\":{\"virtual_background.blur\":"
      "\"open_vulkan_matting_unavailable\"},"
      "\"installed_models\":[\"modnet-webnn-256-fp32\"],"
      "\"default_model_id\":\"modnet-webnn-256-fp32\","
      "\"blocked_reason\":\"open_vulkan_matting_unavailable\"}";

  const ReadinessFields entry = ReadinessEntryFor(
      StatusForVideoConfigWithDiagnostics(videoConfig, /*maxineJson=*/"",
                                          /*openCudaJson=*/"",
                                          openVulkanJson),
      "virtual_background.blur");
  if (!entry.present)
    return false;

  return Expect(entry.backend == "open_vulkan",
                "explicit Vulkan VB should keep open_vulkan attribution") &&
         Expect(entry.state == "backend_unavailable",
                "blocked Vulkan VB should report backend_unavailable") &&
         Expect(entry.reason == "open_vulkan_matting_unavailable",
                "blocked Vulkan VB should report Vulkan matting reason");
}

bool TestBuiltinEffectReadyWithoutDiagnostics() {
  studiocast::video::effects::BroadcastCameraEffects effects;
  effects.engine =
      studiocast::video::effects::EffectsEnginePreference::open_cuda;
  effects.vignette.enabled = true;

  const ReadinessFields entry =
      ReadinessEntryFor(StatusForEffects(effects), "vignette");
  if (!entry.present)
    return false;

  return Expect(entry.backend == "builtin",
                "vignette should report the builtin backend") &&
         Expect(entry.state == "ready",
                "builtin effects should remain ready without diagnostics");
}

bool TestAudioStatusReportsResolvedSourceAndWarnings() {
  studiocast::video::VirtualCameraServiceStatus videoStatus;
  studiocast::video::VirtualCameraServiceConfig videoConfig;

  studiocast::audio::VirtualAudioServiceStatus audioStatus;
  audioStatus.service_running = true;
  audioStatus.mic_present = true;
  audioStatus.selected_source = "alsa_input.usb_status_mic";
  audioStatus.source_warnings.push_back(
      "Using safe source 'alsa_input.usb_status_mic' instead of unsafe Pulse "
      "default source 'studiocast_speakers.monitor'.");

  studiocast::audio::VirtualAudioServiceConfig audioConfig;
  audioConfig.source_name.clear(); // auto

  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(
          StatusToJson(videoStatus, videoConfig, audioStatus, audioConfig,
                       std::filesystem::path("/tmp/studiocastd-test.sock"),
                       /*maxineJson=*/"", /*openCudaJson=*/"",
                       /*openVulkanJson=*/"",
                       /*openAudioJson=*/"", /*loopbackJson=*/""),
          &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return false;
  }

  const JsonObject *root = rootValue.AsObject();
  if (!root) {
    std::cerr << "status root should be an object\n";
    return false;
  }
  const JsonObject *audio = ObjectAt(*root, "audio", "audio should exist");
  if (!audio)
    return false;

  const std::string *source =
      StringAt(*audio, "source", "audio source should exist");
  const std::string *resolved =
      StringAt(*audio, "source_resolved", "resolved source should exist");
  const JsonArray *warnings =
      ArrayAt(*audio, "source_warnings", "source_warnings should exist");
  if (!source || !resolved || !warnings)
    return false;

  return Expect(*source == "auto",
                "configured source should remain auto in status") &&
         Expect(*resolved == "alsa_input.usb_status_mic",
                "status should report resolved physical source") &&
         Expect(!warnings->empty(),
                "status should propagate source resolution warnings");
}

bool TestAudioStatusPropagatesSourceErrorFromService() {
  studiocast::video::VirtualCameraServiceStatus videoStatus;
  studiocast::video::VirtualCameraServiceConfig videoConfig;

  studiocast::audio::VirtualAudioServiceStatus audioStatus;
  audioStatus.service_running = true;
  audioStatus.mic_present = true;
  audioStatus.selected_source = "alsa_input.disconnected_mic";
  audioStatus.source_error =
      "Configured Pulse source 'alsa_input.disconnected_mic' is not currently "
      "available.";

  studiocast::audio::VirtualAudioServiceConfig audioConfig;
  audioConfig.source_name = "alsa_input.disconnected_mic";

  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(
          StatusToJson(videoStatus, videoConfig, audioStatus, audioConfig,
                       std::filesystem::path("/tmp/studiocastd-test.sock"),
                       /*maxineJson=*/"", /*openCudaJson=*/"",
                       /*openVulkanJson=*/"",
                       /*openAudioJson=*/"", /*loopbackJson=*/""),
          &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return false;
  }

  const JsonObject *root = rootValue.AsObject();
  if (!root) {
    std::cerr << "status root should be an object\n";
    return false;
  }
  const JsonObject *audio = ObjectAt(*root, "audio", "audio should exist");
  if (!audio)
    return false;

  const std::string *source =
      StringAt(*audio, "source", "audio source should exist");
  const std::string *resolved =
      StringAt(*audio, "source_resolved", "resolved source should exist");
  const std::string *sourceError =
      StringAt(*audio, "source_error", "source_error should exist");
  if (!source || !resolved || !sourceError)
    return false;

  return Expect(*source == "alsa_input.disconnected_mic",
                "configured disconnected source should be preserved") &&
         Expect(*resolved == "alsa_input.disconnected_mic",
                "resolved source should still identify selected source") &&
         Expect(sourceError->find("not currently available") !=
                    std::string::npos,
                "source_error should propagate service availability error");
}

} // namespace

int main() {
  bool ok = true;
  ok = TestVideoStatusReportsAllowCpuResize() && ok;
  ok = TestVideoStatusReportsRequestedOutputFormat() && ok;
  ok = TestVideoStatusReportsComputeBackend() && ok;
  ok = TestVideoConfigMapsComputeBackendPreference() && ok;
  ok = TestVideoStatusReportsCaptureFallbackState() && ok;
  ok = TestVideoStatusReportsConfiguredDevicesWhenPipelineIdle() && ok;
  ok = TestExplicitOpenCudaEffectUnknownWithoutDiagnostics() && ok;
  ok = TestExplicitVulkanVirtualBackgroundReportsOpenVulkanBlocked() && ok;
  ok = TestBuiltinEffectReadyWithoutDiagnostics() && ok;
  ok = TestAudioStatusReportsResolvedSourceAndWarnings() && ok;
  ok = TestAudioStatusPropagatesSourceErrorFromService() && ok;
  return ok ? 0 : 1;
}
