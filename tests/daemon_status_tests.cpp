#include <cstdlib>
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

bool ArrayContainsString(const JsonArray *array, const std::string &needle) {
  if (!array)
    return false;
  for (const auto &value : *array) {
    const std::string *s = value.AsString();
    if (s && *s == needle)
      return true;
  }
  return false;
}

class ScopedEnvironmentVariable {
public:
  ScopedEnvironmentVariable(const char *name, const char *value) : name_(name) {
    if (const char *previous = std::getenv(name)) {
      had_previous_ = true;
      previous_ = previous;
    }
    setenv(name, value, 1);
  }

  ~ScopedEnvironmentVariable() {
    if (had_previous_)
      setenv(name_.c_str(), previous_.c_str(), 1);
    else
      unsetenv(name_.c_str());
  }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable &) = delete;
  ScopedEnvironmentVariable &
  operator=(const ScopedEnvironmentVariable &) = delete;

private:
  std::string name_;
  std::string previous_;
  bool had_previous_ = false;
};

struct ReadinessFields {
  std::string state;
  std::string reason;
  std::string backend;
  std::string detail;
  bool present = false;
};

struct ComputeFields {
  std::string resolved;
  std::string active;
  std::string fallback;
  std::string degraded;
  std::string provider_mode;
  bool fallback_active = false;
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
  const std::string *detail =
      StringAt(*entry, "detail", "readiness detail should exist");
  if (!state || !reason || !backend || !detail)
    return out;

  out.state = *state;
  out.reason = *reason;
  out.backend = *backend;
  out.detail = *detail;
  out.present = true;
  return out;
}

ComputeFields ComputeFieldsFor(const std::string &statusJson) {
  ComputeFields out;
  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(statusJson, &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return out;
  }
  const JsonObject *root = rootValue.AsObject();
  const JsonObject *video = root ? JsonObjectField(*root, "video") : nullptr;
  const JsonObject *compute =
      video ? JsonObjectField(*video, "compute") : nullptr;
  const JsonObject *fallback =
      compute ? JsonObjectField(*compute, "fallback") : nullptr;
  const JsonObject *provider =
      compute ? JsonObjectField(*compute, "provider") : nullptr;
  if (!compute || !fallback || !provider)
    return out;
  const std::string *resolved = JsonStringField(*compute, "resolved_backend");
  const std::string *active = JsonStringField(*compute, "active_backend");
  const std::string *fallbackReason =
      JsonStringField(*compute, "fallback_reason");
  const std::string *degraded = JsonStringField(*compute, "degraded_reason");
  const std::string *providerMode = JsonStringField(*provider, "mode");
  if (!resolved || !active || !fallbackReason || !degraded || !providerMode)
    return out;
  out.resolved = *resolved;
  out.active = *active;
  out.fallback = *fallbackReason;
  out.degraded = *degraded;
  out.provider_mode = *providerMode;
  out.fallback_active = JsonBoolField(*fallback, "active", false);
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

std::string StatusForVideoStateWithDiagnostics(
    const studiocast::video::VirtualCameraServiceStatus &videoStatus,
    const studiocast::video::VirtualCameraServiceConfig &videoConfig,
    const std::string &maxineJson, const std::string &openCudaJson,
    const std::string &openVulkanJson) {
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

std::string ReadyOpenVulkanPixelDiagnostics(
    const std::string &availableEffects, const std::string &pixelEvidence,
    const std::string &extra = std::string{},
    const std::string &blockedEffects = std::string("{}")) {
  return "{\"compiled_enabled\":true,\"ok\":true,"
         "\"runtime_library_found\":true,\"instance_created\":true,"
         "\"physical_device_found\":true,"
         "\"non_cpu_device_selected\":true,\"cpu_device_selected\":false,"
         "\"compute_queue_available\":true,"
         "\"logical_device_created\":true,\"context_created\":true,"
         "\"context_healthy\":true,\"production_hardware_ready\":true,"
         "\"shader_pipeline_created\":true,"
         "\"context_failure_reason\":\"\","
         "\"available_effects\":" +
         availableEffects + ",\"blocked_effects\":" + blockedEffects +
         ",\"blocked_reason\":"
         "\"open_vulkan_matting_unavailable\","
         "\"degraded_reason\":\"unrelated matting runtime blocker\"," +
         pixelEvidence + extra + "}";
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
  const JsonArray *activeEngines =
      ArrayAt(*compute, "active_engines",
              "compute active_engines should exist");
  const JsonObject *unavailable = ObjectAt(
      *compute, "unavailable_reasons",
      "compute unavailable_reasons should exist");
  const JsonObject *fallbackObject =
      ObjectAt(*compute, "fallback", "compute fallback object should exist");
  const JsonObject *provider =
      ObjectAt(*compute, "provider", "compute provider should exist");
  const JsonObject *cpuTails =
      ObjectAt(*compute, "cpu_tails", "compute cpu_tails should exist");
  const JsonObject *transfers =
      ObjectAt(*compute, "transfers", "compute transfers should exist");
  if (!activeEngines || !unavailable || !fallbackObject || !provider ||
      !cpuTails || !transfers)
    return false;
  const std::string *vulkanUnavailable = StringAt(
      *unavailable, "vulkan", "compute vulkan unavailable reason should exist");
  const std::string *fallbackCode =
      StringAt(*fallbackObject, "code", "compute fallback code should exist");
  const std::string *providerMode =
      StringAt(*provider, "mode", "compute provider mode should exist");
  const JsonObject *cudaTransfers =
      ObjectAt(*transfers, "open_cuda", "open_cuda transfers should exist");
  const JsonObject *vulkanTransfers =
      ObjectAt(*transfers, "open_vulkan",
               "open_vulkan transfers should exist");
  const JsonObject *maxineTransfers =
      ObjectAt(*transfers, "maxine", "maxine transfers should exist");
  if (!vulkanUnavailable || !fallbackCode || !providerMode ||
      !cudaTransfers || !vulkanTransfers || !maxineTransfers)
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
         Expect(*fallback == "open_vulkan_runtime_diagnostics_unavailable",
                "missing Vulkan diagnostics should use a stable exact "
                "fallback reason") &&
         Expect(*degraded == *fallback,
                "vulkan unavailable degraded reason should match fallback") &&
         Expect(ArrayContainsString(activeEngines, "cpu"),
                "vulkan unavailable status should report active cpu engine") &&
         Expect(*vulkanUnavailable ==
                    "open_vulkan_runtime_diagnostics_unavailable",
                "Vulkan unavailable reason should preserve the exact "
                "runtime evidence blocker") &&
         Expect(JsonBoolField(*fallbackObject, "active", false),
                "fallback object should mark fallback active") &&
         Expect(*fallbackCode == "vulkan_unavailable",
                "fallback object should report vulkan_unavailable code") &&
         Expect(*providerMode == "cpu",
                "provider should report CPU mode for CPU fallback") &&
         Expect(!JsonBoolField(*cpuTails, "active", true),
                "cpu tail summary should report inactive with zero counters") &&
         Expect(*configBackend == "vulkan",
                "config JSON should report vulkan compute_backend");
}

bool TestVideoComputeStatusReportsCachedCountersAndProvider() {
  studiocast::video::VirtualCameraServiceStatus videoStatus;
  videoStatus.service_running = true;
  videoStatus.virtual_device_present = true;
  videoStatus.virtual_device_available = true;
  videoStatus.pipeline.running = true;
  videoStatus.pipeline.compute_backend_resolved = "cuda";
  videoStatus.pipeline.compute_backend_active = "cuda";
  videoStatus.pipeline.effects_backends =
      "video_noise_removal:open_cuda,virtual_key_light:open_cuda";
  videoStatus.pipeline.open_cuda_transfers.active_frames = 12;
  videoStatus.pipeline.open_cuda_transfers.upload_calls = 3;
  videoStatus.pipeline.open_cuda_transfers.download_calls = 4;
  videoStatus.pipeline.open_cuda_transfers.final_download_calls = 5;
  videoStatus.pipeline.open_cuda_transfers.forced_sync_calls = 2;
  videoStatus.pipeline.open_cuda_transfers.cpu_tail_stage_calls = 2;
  videoStatus.pipeline.open_cuda_transfers.cpu_tail_key_light_calls = 1;
  videoStatus.pipeline.open_cuda_transfers.cpu_tail_denoise_calls = 1;

  studiocast::video::VirtualCameraServiceConfig videoConfig;
  videoConfig.enabled = true;
  videoConfig.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::cuda;
  videoConfig.pipeline.effects.video_noise_removal.enabled = true;
  videoConfig.pipeline.effects.virtual_key_light.enabled = true;

  studiocast::audio::VirtualAudioServiceStatus audioStatus;
  audioStatus.service_running = true;
  audioStatus.mic_present = true;
  audioStatus.speakers_present = true;

  studiocast::audio::VirtualAudioServiceConfig audioConfig;

  const std::string openCudaJson =
      "{\"ok\":true,\"onnxruntime_cuda_provider_present\":true,"
      "\"onnxruntime_tensorrt_provider_present\":true,"
      "\"tensorrt_requested\":true,\"tensorrt_available\":true,"
      "\"cuda_context_available\":true,\"cuda_device_count\":1,"
      "\"available_effects\":[\"video_noise_removal\","
      "\"virtual_key_light\"]}";

  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(
          StatusToJson(videoStatus, videoConfig, audioStatus, audioConfig,
                       std::filesystem::path("/tmp/studiocastd-test.sock"),
                       /*maxineJson=*/"", openCudaJson,
                       /*openVulkanJson=*/"", /*openAudioJson=*/"",
                       /*loopbackJson=*/""),
          &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return false;
  }

  const JsonObject *root = rootValue.AsObject();
  if (!root)
    return false;
  const JsonObject *video = ObjectAt(*root, "video", "video should exist");
  if (!video)
    return false;
  const JsonObject *compute =
      ObjectAt(*video, "compute", "video.compute should exist");
  if (!compute)
    return false;

  const JsonArray *activeEngines =
      ArrayAt(*compute, "active_engines",
              "compute active_engines should exist");
  const JsonObject *provider =
      ObjectAt(*compute, "provider", "compute provider should exist");
  const JsonObject *cpuTails =
      ObjectAt(*compute, "cpu_tails", "compute cpu_tails should exist");
  const JsonObject *transfers =
      ObjectAt(*compute, "transfers", "compute transfers should exist");
  if (!activeEngines || !provider || !cpuTails || !transfers)
    return false;

  const JsonArray *tailStages =
      ArrayAt(*cpuTails, "stages", "cpu tail stages should exist");
  const JsonObject *openCudaTransfers =
      ObjectAt(*transfers, "open_cuda", "open_cuda transfers should exist");
  if (!tailStages || !openCudaTransfers)
    return false;
  const std::string *providerMode =
      StringAt(*provider, "mode", "provider mode should exist");
  const std::string *activeProvider = StringAt(
      *provider, "active_provider", "active provider should exist");
  const std::string *tensorIo =
      StringAt(*provider, "tensor_io_mode", "tensor I/O mode should exist");
  if (!providerMode || !activeProvider || !tensorIo)
    return false;

  return Expect(ArrayContainsString(activeEngines, "open_cuda"),
                "active engines should include open_cuda") &&
         Expect(*providerMode == "open_cuda",
                "provider mode should report open_cuda") &&
         Expect(*activeProvider == "TensorrtExecutionProvider",
                "provider should prefer requested available TensorRT") &&
         Expect(*tensorIo == "cuda_device_iobinding",
                "provider should report CUDA device tensor I/O") &&
         Expect(JsonBoolField(*cpuTails, "active", false),
                "cpu tails should report active when counters are cached") &&
         Expect(ArrayContainsString(tailStages, "key_light"),
                "cpu tail stages should include key_light") &&
         Expect(ArrayContainsString(tailStages, "denoise"),
                "cpu tail stages should include denoise") &&
         Expect(JsonNumberField(*openCudaTransfers, "forced_syncs", 0.0) ==
                    2.0,
                "transfer summary should include forced sync count");
}

bool TestVideoComputeTransferTotalsDoNotDoubleCountSubcounters() {
  studiocast::video::VirtualCameraServiceStatus videoStatus;
  videoStatus.service_running = true;
  videoStatus.virtual_device_present = true;
  videoStatus.virtual_device_available = true;
  videoStatus.pipeline.running = true;

  auto &cu = videoStatus.pipeline.open_cuda_transfers;
  cu.upload_calls = 17;
  cu.download_calls = 19;
  cu.matte_frame_upload_calls = 5;
  cu.standalone_scaler_upload_calls = 7;
  cu.denoise_tensor_upload_calls = 11;
  cu.final_download_calls = 3;
  cu.cpu_continuation_download_calls = 4;
  cu.alpha_download_calls = 6;
  cu.standalone_scaler_download_calls = 8;
  cu.denoise_tensor_download_calls = 10;

  auto &vk = videoStatus.pipeline.open_vulkan_transfers;
  vk.upload_calls = 23;
  vk.download_calls = 29;
  vk.background_upload_calls = 13;
  vk.standalone_scaler_upload_calls = 17;
  vk.final_download_calls = 19;
  vk.cpu_continuation_download_calls = 2;
  vk.alpha_download_calls = 3;
  vk.standalone_scaler_download_calls = 5;

  auto &mx = videoStatus.pipeline.maxine_transfers;
  mx.upload_calls = 31;
  mx.download_calls = 37;
  mx.final_download_calls = 7;
  mx.cpu_continuation_download_calls = 11;
  mx.standalone_scaler_upload_calls = 13;
  mx.standalone_scaler_download_calls = 17;

  studiocast::video::VirtualCameraServiceConfig videoConfig;
  videoConfig.enabled = true;
  studiocast::audio::VirtualAudioServiceStatus audioStatus;
  studiocast::audio::VirtualAudioServiceConfig audioConfig;

  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(
          StatusToJson(videoStatus, videoConfig, audioStatus, audioConfig,
                       std::filesystem::path("/tmp/studiocastd-test.sock"),
                       /*maxineJson=*/"", /*openCudaJson=*/"",
                       /*openVulkanJson=*/"", /*openAudioJson=*/"",
                       /*loopbackJson=*/""),
          &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return false;
  }

  const JsonObject *root = rootValue.AsObject();
  if (!root)
    return false;
  const JsonObject *video = ObjectAt(*root, "video", "video should exist");
  if (!video)
    return false;
  const JsonObject *compute =
      ObjectAt(*video, "compute", "video.compute should exist");
  if (!compute)
    return false;
  const JsonObject *transfers =
      ObjectAt(*compute, "transfers", "compute transfers should exist");
  if (!transfers)
    return false;

  const JsonObject *openCuda =
      ObjectAt(*transfers, "open_cuda", "open_cuda transfers should exist");
  const JsonObject *openVulkan =
      ObjectAt(*transfers, "open_vulkan", "open_vulkan transfers should exist");
  const JsonObject *maxine =
      ObjectAt(*transfers, "maxine", "maxine transfers should exist");
  if (!openCuda || !openVulkan || !maxine)
    return false;

  return Expect(JsonNumberField(*openCuda, "uploads", -1.0) == 17.0,
                "Open CUDA uploads should be the aggregate total") &&
         Expect(JsonNumberField(*openCuda, "downloads", -1.0) == 19.0,
                "Open CUDA downloads should be the aggregate total") &&
         Expect(JsonNumberField(*openCuda, "matte_frame_uploads", -1.0) == 5.0,
                "Open CUDA matte upload subcounter should remain available") &&
         Expect(
             JsonNumberField(*openCuda, "denoise_tensor_downloads", -1.0) ==
                 10.0,
             "Open CUDA denoise download subcounter should remain available") &&
         Expect(JsonNumberField(*openVulkan, "uploads", -1.0) == 23.0,
                "Open Vulkan uploads should be the aggregate total") &&
         Expect(JsonNumberField(*openVulkan, "downloads", -1.0) == 29.0,
                "Open Vulkan downloads should be the aggregate total") &&
         Expect(JsonNumberField(*openVulkan, "background_uploads", -1.0) ==
                    13.0,
                "Open Vulkan background upload subcounter should remain "
                "available") &&
         Expect(JsonNumberField(*maxine, "uploads", -1.0) == 31.0,
                "Maxine uploads should be the aggregate total") &&
         Expect(JsonNumberField(*maxine, "downloads", -1.0) == 37.0,
                "Maxine downloads should be the aggregate total") &&
         Expect(JsonNumberField(*maxine, "standalone_scaler_downloads", -1.0) ==
                    17.0,
                "Maxine scaler download subcounter should remain available");
}

bool TestVulkanVirtualBackgroundBlurDebugCounters() {
  ScopedEnvironmentVariable debug("STUDIOCAST_DEBUG_OPEN_VULKAN_TRANSFERS",
                                  "1");
  studiocast::video::VirtualCameraServiceStatus videoStatus;
  auto &vk = videoStatus.pipeline.open_vulkan_transfers;
  vk.virtual_background_blur_dispatch_calls = 11;
  vk.virtual_background_blur_alpha_readback_calls = 13;
  vk.virtual_background_blur_cpu_fallback_calls = 17;
  vk.virtual_background_blur_runtime_failure_frames = 19;
  vk.virtual_background_blur_device_loss_frames = 23;

  studiocast::video::VirtualCameraServiceConfig videoConfig;
  studiocast::audio::VirtualAudioServiceStatus audioStatus;
  studiocast::audio::VirtualAudioServiceConfig audioConfig;

  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(
          StatusToJson(videoStatus, videoConfig, audioStatus, audioConfig,
                       std::filesystem::path("/tmp/studiocastd-test.sock"),
                       /*maxineJson=*/"", /*openCudaJson=*/"",
                       /*openVulkanJson=*/"", /*openAudioJson=*/"",
                       /*loopbackJson=*/""),
          &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return false;
  }

  const JsonObject *root = rootValue.AsObject();
  if (!root)
    return false;
  const JsonObject *video = ObjectAt(*root, "video", "video should exist");
  if (!video)
    return false;
  const JsonObject *pipeline =
      ObjectAt(*video, "pipeline", "video.pipeline should exist");
  if (!pipeline)
    return false;
  const JsonObject *debugCounters =
      ObjectAt(*pipeline, "open_vulkan_transfers",
               "Open Vulkan debug transfer counters should exist");
  if (!debugCounters)
    return false;

  return Expect(JsonNumberField(*debugCounters,
                                "virtual_background_blur_dispatch_calls",
                                -1.0) == 11.0,
                "blur dispatch counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_blur_alpha_readback_calls", -1.0) ==
                    13.0,
                "blur alpha readback counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_blur_cpu_fallback_calls", -1.0) ==
                    17.0,
                "blur CPU fallback counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_blur_runtime_failure_frames", -1.0) ==
                    19.0,
                "blur runtime failure counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_blur_device_loss_frames", -1.0) ==
                    23.0,
                "blur device loss counter should be published");
}

bool TestVulkanVirtualBackgroundRemoveDebugCounters() {
  ScopedEnvironmentVariable debug("STUDIOCAST_DEBUG_OPEN_VULKAN_TRANSFERS",
                                  "1");
  studiocast::video::VirtualCameraServiceStatus videoStatus;
  auto &vk = videoStatus.pipeline.open_vulkan_transfers;
  vk.virtual_background_remove_dispatch_calls = 29;
  vk.virtual_background_remove_alpha_readback_calls = 31;
  vk.virtual_background_remove_cpu_fallback_calls = 37;
  vk.virtual_background_remove_runtime_failure_frames = 41;
  vk.virtual_background_remove_device_loss_frames = 43;

  studiocast::video::VirtualCameraServiceConfig videoConfig;
  studiocast::audio::VirtualAudioServiceStatus audioStatus;
  studiocast::audio::VirtualAudioServiceConfig audioConfig;

  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(
          StatusToJson(videoStatus, videoConfig, audioStatus, audioConfig,
                       std::filesystem::path("/tmp/studiocastd-test.sock"),
                       /*maxineJson=*/"", /*openCudaJson=*/"",
                       /*openVulkanJson=*/"", /*openAudioJson=*/"",
                       /*loopbackJson=*/""),
          &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return false;
  }

  const JsonObject *root = rootValue.AsObject();
  if (!root)
    return false;
  const JsonObject *video = ObjectAt(*root, "video", "video should exist");
  if (!video)
    return false;
  const JsonObject *pipeline =
      ObjectAt(*video, "pipeline", "video.pipeline should exist");
  if (!pipeline)
    return false;
  const JsonObject *debugCounters =
      ObjectAt(*pipeline, "open_vulkan_transfers",
               "Open Vulkan debug transfer counters should exist");
  if (!debugCounters)
    return false;

  return Expect(JsonNumberField(*debugCounters,
                                "virtual_background_remove_dispatch_calls",
                                -1.0) == 29.0,
                "remove dispatch counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_remove_alpha_readback_calls", -1.0) ==
                    31.0,
                "remove alpha readback counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_remove_cpu_fallback_calls", -1.0) ==
                    37.0,
                "remove CPU fallback counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_remove_runtime_failure_frames", -1.0) ==
                    41.0,
                "remove runtime failure counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_remove_device_loss_frames", -1.0) ==
                    43.0,
                "remove device loss counter should be published");
}

bool TestVulkanVirtualBackgroundReplaceDebugCounters() {
  ScopedEnvironmentVariable debug("STUDIOCAST_DEBUG_OPEN_VULKAN_TRANSFERS",
                                  "1");
  studiocast::video::VirtualCameraServiceStatus videoStatus;
  auto &vk = videoStatus.pipeline.open_vulkan_transfers;
  vk.virtual_background_replace_asset_allocation_calls = 47;
  vk.virtual_background_replace_asset_decode_calls = 53;
  vk.virtual_background_replace_asset_upload_calls = 59;
  vk.virtual_background_replace_asset_resize_dispatch_calls = 61;
  vk.virtual_background_replace_dispatch_calls = 67;
  vk.virtual_background_replace_alpha_readback_calls = 71;
  vk.virtual_background_replace_cpu_fallback_calls = 73;
  vk.virtual_background_replace_runtime_failure_frames = 79;
  vk.virtual_background_replace_device_loss_frames = 83;

  studiocast::video::VirtualCameraServiceConfig videoConfig;
  studiocast::audio::VirtualAudioServiceStatus audioStatus;
  studiocast::audio::VirtualAudioServiceConfig audioConfig;

  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(
          StatusToJson(videoStatus, videoConfig, audioStatus, audioConfig,
                       std::filesystem::path("/tmp/studiocastd-test.sock"),
                       /*maxineJson=*/"", /*openCudaJson=*/"",
                       /*openVulkanJson=*/"", /*openAudioJson=*/"",
                       /*loopbackJson=*/""),
          &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return false;
  }

  const JsonObject *root = rootValue.AsObject();
  if (!root)
    return false;
  const JsonObject *video = ObjectAt(*root, "video", "video should exist");
  if (!video)
    return false;
  const JsonObject *pipeline =
      ObjectAt(*video, "pipeline", "video.pipeline should exist");
  if (!pipeline)
    return false;
  const JsonObject *debugCounters =
      ObjectAt(*pipeline, "open_vulkan_transfers",
               "Open Vulkan debug transfer counters should exist");
  if (!debugCounters)
    return false;

  return Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_replace_asset_allocation_calls",
                    -1.0) == 47.0,
                "replace asset allocation counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_replace_asset_decode_calls", -1.0) ==
                    53.0,
                "replace asset decode counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_replace_asset_upload_calls", -1.0) ==
                    59.0,
                "replace asset upload counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_replace_asset_resize_dispatch_calls",
                    -1.0) == 61.0,
                "replace setup resize counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_replace_dispatch_calls", -1.0) ==
                    67.0,
                "replace dispatch counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_replace_alpha_readback_calls", -1.0) ==
                    71.0,
                "replace alpha readback counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_replace_cpu_fallback_calls", -1.0) ==
                    73.0,
                "replace CPU fallback counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_replace_runtime_failure_frames",
                    -1.0) == 79.0,
                "replace runtime failure counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_background_replace_device_loss_frames", -1.0) ==
                    83.0,
                "replace device loss counter should be published");
}

bool TestVulkanVirtualKeyLightDebugCounters() {
  ScopedEnvironmentVariable debug("STUDIOCAST_DEBUG_OPEN_VULKAN_TRANSFERS",
                                  "1");
  studiocast::video::VirtualCameraServiceStatus videoStatus;
  auto &vk = videoStatus.pipeline.open_vulkan_transfers;
  vk.key_light_dispatch_calls = 89;
  vk.virtual_key_light_shared_matte_reuse_calls = 97;
  vk.virtual_key_light_independent_matte_inference_calls = 101;
  vk.virtual_key_light_passthrough_frames = 103;
  vk.virtual_key_light_alpha_readback_calls = 107;
  vk.virtual_key_light_cpu_fallback_calls = 109;
  vk.virtual_key_light_runtime_failure_frames = 113;
  vk.virtual_key_light_device_loss_frames = 127;

  studiocast::video::VirtualCameraServiceConfig videoConfig;
  studiocast::audio::VirtualAudioServiceStatus audioStatus;
  studiocast::audio::VirtualAudioServiceConfig audioConfig;

  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(
          StatusToJson(videoStatus, videoConfig, audioStatus, audioConfig,
                       std::filesystem::path("/tmp/studiocastd-test.sock"),
                       /*maxineJson=*/"", /*openCudaJson=*/"",
                       /*openVulkanJson=*/"", /*openAudioJson=*/"",
                       /*loopbackJson=*/""),
          &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return false;
  }

  const JsonObject *root = rootValue.AsObject();
  if (!root)
    return false;
  const JsonObject *video = ObjectAt(*root, "video", "video should exist");
  if (!video)
    return false;
  const JsonObject *pipeline =
      ObjectAt(*video, "pipeline", "video.pipeline should exist");
  if (!pipeline)
    return false;
  const JsonObject *debugCounters =
      ObjectAt(*pipeline, "open_vulkan_transfers",
               "Open Vulkan debug transfer counters should exist");
  if (!debugCounters)
    return false;

  return Expect(JsonNumberField(*debugCounters, "key_light_dispatch_calls",
                                -1.0) == 89.0,
                "key-light dispatch counter should be published") &&
         Expect(JsonNumberField(*debugCounters,
                                "virtual_key_light_shared_matte_reuse_calls",
                                -1.0) == 97.0,
                "key-light reuse counter should be published") &&
         Expect(JsonNumberField(
                    *debugCounters,
                    "virtual_key_light_independent_matte_inference_calls",
                    -1.0) == 101.0,
                "key-light independent inference counter should be "
                "published") &&
         Expect(JsonNumberField(*debugCounters,
                                "virtual_key_light_passthrough_frames",
                                -1.0) == 103.0,
                "key-light pass-through counter should be published") &&
         Expect(JsonNumberField(*debugCounters,
                                "virtual_key_light_alpha_readback_calls",
                                -1.0) == 107.0,
                "key-light alpha readback counter should be published") &&
         Expect(JsonNumberField(*debugCounters,
                                "virtual_key_light_cpu_fallback_calls",
                                -1.0) == 109.0,
                "key-light CPU fallback counter should be published") &&
         Expect(JsonNumberField(*debugCounters,
                                "virtual_key_light_runtime_failure_frames",
                                -1.0) == 113.0,
                "key-light runtime failure counter should be published") &&
         Expect(JsonNumberField(*debugCounters,
                                "virtual_key_light_device_loss_frames",
                                -1.0) == 127.0,
                "key-light device loss counter should be published");
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

bool TestPersistentVulkanAdapterConfigAndStatus() {
  const std::string stableId = "v1:8086:1234:1:intel-arc-integrated";
  studiocast::config::DaemonConfig daemonConfig;
  daemonConfig.video_vulkan_device = stableId;
  daemonConfig.video_vulkan_allow_cpu = true;

  auto runtime = studiocast::config::ToVideoServiceConfig(daemonConfig);
  runtime.pipeline.width = 1920;
  studiocast::config::ApplyVideoServiceConfigToDaemonConfig(runtime,
                                                            &daemonConfig);
  if (!Expect(daemonConfig.video_vulkan_device == stableId &&
                  daemonConfig.video_vulkan_allow_cpu,
              "video config patches must preserve Vulkan adapter fields")) {
    return false;
  }

  const char *oldXdg = std::getenv("XDG_CONFIG_HOME");
  const bool hadOldXdg = oldXdg != nullptr;
  const std::string oldXdgValue = oldXdg ? oldXdg : "";
  const auto temp = std::filesystem::temp_directory_path() /
                    ("studiocast-vulkan-config-test-" +
                     std::to_string(std::chrono::steady_clock::now()
                                        .time_since_epoch()
                                        .count()));
  (void)setenv("XDG_CONFIG_HOME", temp.c_str(), 1);
  std::string saveError;
  const bool saved =
      studiocast::config::SaveDaemonConfig(daemonConfig, &saveError);
  const auto loaded = studiocast::config::LoadDaemonConfig();
  std::error_code removeError;
  std::filesystem::remove_all(temp, removeError);
  if (hadOldXdg)
    (void)setenv("XDG_CONFIG_HOME", oldXdgValue.c_str(), 1);
  else
    (void)unsetenv("XDG_CONFIG_HOME");
  if (!Expect(saved, "Vulkan adapter config should save")) {
    std::cerr << saveError << "\n";
    return false;
  }
  if (!Expect(loaded.video_vulkan_device == stableId &&
                  loaded.video_vulkan_allow_cpu,
              "Vulkan adapter config should survive daemon reload")) {
    return false;
  }

  studiocast::video::VirtualCameraServiceStatus videoStatus;
  videoStatus.service_running = true;
  studiocast::audio::VirtualAudioServiceStatus audioStatus;
  studiocast::audio::VirtualAudioServiceConfig audioConfig;
  const std::string json = StatusToJson(
      videoStatus, runtime, audioStatus, audioConfig,
      std::filesystem::path("/tmp/studiocastd-test.sock"), "", "", "", "",
      "", &daemonConfig);
  return Expect(json.find("\"configured_device\":\"" + stableId +
                          "\"") != std::string::npos &&
                    json.find("\"allow_cpu\":true") != std::string::npos,
                "daemon status should expose canonical Vulkan adapter config");
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
      "\"blocked_reason\":\"open_vulkan_matting_unavailable\","
      "\"degraded_reason\":\"Open Vulkan runtime is available, but no "
      "production device-resident matting inference runtime is available.\"}";

  const std::string statusJson = StatusForVideoConfigWithDiagnostics(
      videoConfig, /*maxineJson=*/"", /*openCudaJson=*/"", openVulkanJson);

  const ReadinessFields entry =
      ReadinessEntryFor(statusJson, "virtual_background.blur");
  if (!entry.present)
    return false;

  studiocast::util::json::Value rootValue;
  std::string error;
  if (!studiocast::util::json::Parse(statusJson, &rootValue, &error)) {
    std::cerr << "status JSON should parse: " << error << "\n";
    return false;
  }
  const JsonObject *root = rootValue.AsObject();
  if (!root)
    return false;
  const JsonObject *video = ObjectAt(*root, "video", "video should exist");
  if (!video)
    return false;
  const JsonObject *compute =
      ObjectAt(*video, "compute", "video.compute should exist");
  if (!compute)
    return false;
  const std::string *resolved = StringAt(
      *compute, "resolved_backend", "compute resolved backend should exist");
  const std::string *active =
      StringAt(*compute, "active_backend", "compute active backend should exist");
  const std::string *fallback =
      StringAt(*compute, "fallback_reason", "compute fallback should exist");
  const std::string *degraded =
      StringAt(*compute, "degraded_reason", "compute degraded should exist");
  const JsonObject *fallbackObject =
      ObjectAt(*compute, "fallback", "compute fallback object should exist");
  const JsonObject *provider =
      ObjectAt(*compute, "provider", "compute provider should exist");
  const JsonObject *unavailable = ObjectAt(
      *compute, "unavailable_reasons",
      "compute unavailable_reasons should exist");
  if (!resolved || !active || !fallback || !degraded || !fallbackObject ||
      !provider || !unavailable)
    return false;
  const std::string *fallbackCode =
      StringAt(*fallbackObject, "code", "compute fallback code should exist");
  const std::string *providerMode =
      StringAt(*provider, "mode", "compute provider mode should exist");
  const std::string *vulkanUnavailable = StringAt(
      *unavailable, "vulkan", "compute vulkan unavailable reason should exist");
  if (!fallbackCode || !providerMode || !vulkanUnavailable)
    return false;

  return Expect(entry.backend == "open_vulkan",
                "explicit Vulkan VB should keep open_vulkan attribution") &&
         Expect(entry.state == "backend_unavailable",
                "blocked Vulkan VB should report backend_unavailable") &&
         Expect(entry.reason == "open_vulkan_matting_unavailable",
                "blocked Vulkan VB should report Vulkan matting reason") &&
         Expect(*resolved == "cpu",
                "blocked explicit Vulkan should resolve compute to cpu") &&
         Expect(*active == "cpu",
                "blocked explicit Vulkan should report active cpu") &&
         Expect(fallback->find("production device-resident matting") !=
                    std::string::npos,
                "blocked explicit Vulkan fallback should report matting "
                "readiness") &&
         Expect(*degraded == *fallback,
                "blocked explicit Vulkan degraded reason should match "
                "fallback") &&
         Expect(vulkanUnavailable->find("production device-resident matting") !=
                    std::string::npos,
                "blocked explicit Vulkan unavailable reason should report "
                "matting readiness") &&
         Expect(JsonBoolField(*fallbackObject, "active", false),
                "blocked explicit Vulkan should mark fallback active") &&
         Expect(*fallbackCode == "vulkan_unavailable",
                "blocked explicit Vulkan should use vulkan_unavailable code") &&
         Expect(*providerMode == "cpu",
                "blocked explicit Vulkan must not report a CUDA provider");
}

bool TestExplicitVulkanEyeContactReportsExactFailClosedFacts() {
  studiocast::video::VirtualCameraServiceConfig videoConfig;
  videoConfig.enabled = true;
  videoConfig.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::vulkan;
  videoConfig.pipeline.effects.eye_contact.enabled = true;

  const std::string openVulkanJson =
      "{\"compiled_enabled\":true,\"ok\":true,"
      "\"available_effects\":[\"mirror\"],"
      "\"blocked_effects\":{\"eye_contact\":"
      "\"open_vulkan_eye_contact_unavailable\"},"
      "\"eye_contact_production_ready\":false,"
      "\"eye_contact_reason_code\":\"open_vulkan_eye_contact_unavailable\","
      "\"eye_contact_blocker_code\":"
      "\"open_vulkan_eye_contact_runtime_unavailable\","
      "\"eye_contact_detail\":\"current ONNX/dlib path uses CPU tensors\","
      "\"eye_contact_backend_compiled\":true,"
      "\"eye_contact_live_stage_implemented\":false,"
      "\"eye_contact_production_adapter_available\":false,"
      "\"eye_contact_vulkan_inference_provider_available\":false,"
      "\"eye_contact_shared_device_imported\":false,"
      "\"eye_contact_queue_ownership_explicit\":false,"
      "\"eye_contact_model_pack_selected\":false,"
      "\"eye_contact_artifact_contract_validated\":false,"
      "\"eye_contact_selectable_cpu_fallback\":false,"
      "\"eye_contact_dispatch_count\":0,"
      "\"eye_contact_cpu_readback_count\":0,"
      "\"eye_contact_cpu_fallback_count\":0}";

  const std::string statusJson = StatusForVideoConfigWithDiagnostics(
      videoConfig, /*maxineJson=*/"", /*openCudaJson=*/"", openVulkanJson);
  const ReadinessFields entry = ReadinessEntryFor(statusJson, "eye_contact");
  if (!entry.present)
    return false;

  studiocast::util::json::Value diagnosticsValue;
  std::string error;
  if (!studiocast::util::json::Parse(openVulkanJson, &diagnosticsValue,
                                     &error)) {
    std::cerr << "Open Vulkan fixture should parse: " << error << "\n";
    return false;
  }
  const JsonObject *diagnostics = diagnosticsValue.AsObject();
  if (!diagnostics)
    return false;
  const std::string *blocker = StringAt(
      *diagnostics, "eye_contact_blocker_code",
      "eye-contact runtime blocker should exist");
  if (!blocker)
    return false;

  return Expect(entry.backend == "open_vulkan",
                "explicit Vulkan eye contact must keep backend attribution") &&
         Expect(entry.state == "backend_unavailable",
                "explicit Vulkan eye contact must fail closed") &&
         Expect(entry.reason == "open_vulkan_eye_contact_unavailable",
                "eye contact readiness must expose the stable outer reason") &&
         Expect(entry.detail.find(
                    "[open_vulkan_eye_contact_unavailable] "
                    "[open_vulkan_eye_contact_runtime_unavailable]") !=
                    std::string::npos &&
                    entry.detail.find("CPU tensors") != std::string::npos,
                "eye-contact readiness detail must expose the nested runtime "
                "and CPU boundary") &&
         Expect(*blocker == "open_vulkan_eye_contact_runtime_unavailable",
                "eye contact diagnostics must expose the primary blocker") &&
         Expect(JsonBoolField(*diagnostics, "eye_contact_backend_compiled",
                              false),
                "compiled backend must remain a separate true fact") &&
         Expect(!JsonBoolField(*diagnostics,
                               "eye_contact_live_stage_implemented", true),
                "eye contact must not claim a live implementation") &&
         Expect(!JsonBoolField(*diagnostics,
                               "eye_contact_production_adapter_available",
                               true),
                "eye contact must not claim a production adapter") &&
         Expect(!JsonBoolField(
                    *diagnostics,
                    "eye_contact_vulkan_inference_provider_available", true),
                "eye contact must not claim a Vulkan inference provider") &&
         Expect(!JsonBoolField(*diagnostics,
                               "eye_contact_shared_device_imported", true) &&
                    !JsonBoolField(
                        *diagnostics,
                        "eye_contact_queue_ownership_explicit", true),
                "eye contact must not claim shared device/queue ownership") &&
         Expect(!JsonBoolField(*diagnostics,
                               "eye_contact_model_pack_selected", true),
                "eye contact must not select an ONNX-only pack as Vulkan") &&
         Expect(!JsonBoolField(*diagnostics,
                               "eye_contact_artifact_contract_validated",
                               true),
                "eye contact must not claim a Vulkan artifact contract") &&
         Expect(!JsonBoolField(*diagnostics,
                               "eye_contact_selectable_cpu_fallback", true),
                "eye contact must not expose an internal CPU tail as fallback") &&
         Expect(JsonNumberField(*diagnostics, "eye_contact_dispatch_count",
                                1.0) == 0.0 &&
                    JsonNumberField(*diagnostics,
                                    "eye_contact_cpu_readback_count",
                                    1.0) == 0.0 &&
                    JsonNumberField(*diagnostics,
                                    "eye_contact_cpu_fallback_count",
                                    1.0) == 0.0,
                "blocked eye contact must prove zero frame work");
}

bool TestExplicitVulkanVideoNoiseRemovalReportsExactFailClosedFacts() {
  studiocast::video::VirtualCameraServiceConfig videoConfig;
  videoConfig.enabled = true;
  videoConfig.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::vulkan;
  videoConfig.pipeline.effects.video_noise_removal.enabled = true;

  const std::string openVulkanJson =
      "{\"compiled_enabled\":true,\"ok\":true,"
      "\"available_effects\":[\"mirror\"],"
      "\"blocked_effects\":{\"video_noise_removal\":"
      "\"open_vulkan_video_noise_removal_unavailable\"},"
      "\"video_noise_removal_production_ready\":false,"
      "\"video_noise_removal_reason_code\":"
      "\"open_vulkan_video_noise_removal_unavailable\","
      "\"video_noise_removal_blocker_code\":"
      "\"open_vulkan_video_noise_removal_runtime_unavailable\","
      "\"video_noise_removal_detail\":\"FastDVDnet has host temporal "
      "history, CPU preprocessing/postprocessing, and ONNX-only artifacts\","
      "\"video_noise_removal_backend_compiled\":true,"
      "\"video_noise_removal_live_stage_implemented\":false,"
      "\"video_noise_removal_production_adapter_available\":false,"
      "\"video_noise_removal_vulkan_inference_provider_available\":false,"
      "\"video_noise_removal_non_cpu_device_selected\":true,"
      "\"video_noise_removal_compute_queue_available\":true,"
      "\"video_noise_removal_context_healthy\":true,"
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
      "\"video_noise_removal_cpu_fallback_count\":0}";

  const std::string statusJson = StatusForVideoConfigWithDiagnostics(
      videoConfig, /*maxineJson=*/"", /*openCudaJson=*/"", openVulkanJson);
  const ReadinessFields entry =
      ReadinessEntryFor(statusJson, "video_noise_removal");
  if (!entry.present)
    return false;

  studiocast::util::json::Value diagnosticsValue;
  std::string error;
  if (!studiocast::util::json::Parse(openVulkanJson, &diagnosticsValue,
                                     &error)) {
    std::cerr << "Open Vulkan fixture should parse: " << error << "\n";
    return false;
  }
  const JsonObject *diagnostics = diagnosticsValue.AsObject();
  if (!diagnostics)
    return false;
  const std::string *blocker =
      StringAt(*diagnostics, "video_noise_removal_blocker_code",
               "video-denoise runtime blocker should exist");
  if (!blocker)
    return false;

  return Expect(
             entry.backend == "open_vulkan",
             "explicit Vulkan video denoise must keep backend attribution") &&
         Expect(entry.state == "backend_unavailable",
                "explicit Vulkan video denoise must fail closed") &&
         Expect(entry.reason == "open_vulkan_video_noise_removal_unavailable",
                "video denoise must expose the stable outer reason") &&
         Expect(entry.detail.find(
                    "[open_vulkan_video_noise_removal_unavailable] "
                    "[open_vulkan_video_noise_removal_runtime_unavailable]") !=
                        std::string::npos &&
                    entry.detail.find("host temporal history") !=
                        std::string::npos &&
                    entry.detail.find("ONNX-only") != std::string::npos,
                "video-denoise detail must expose nested runtime, temporal, "
                "and artifact boundaries") &&
         Expect(*blocker ==
                    "open_vulkan_video_noise_removal_runtime_unavailable",
                "video denoise diagnostics must expose the primary blocker") &&
         Expect(JsonBoolField(*diagnostics,
                              "video_noise_removal_backend_compiled", false),
                "compiled backend must remain a separate true fact") &&
         Expect(
             !JsonBoolField(*diagnostics,
                            "video_noise_removal_live_stage_implemented",
                            true) &&
                 !JsonBoolField(
                     *diagnostics,
                     "video_noise_removal_production_adapter_available",
                     true) &&
                 !JsonBoolField(
                     *diagnostics,
                     "video_noise_removal_vulkan_inference_provider_available",
                     true),
             "video denoise must not claim a live adapter/provider") &&
         Expect(
             JsonBoolField(*diagnostics,
                           "video_noise_removal_non_cpu_device_selected",
                           false) &&
                 JsonBoolField(*diagnostics,
                               "video_noise_removal_compute_queue_available",
                               false) &&
                 JsonBoolField(*diagnostics,
                               "video_noise_removal_context_healthy", false),
             "hardware facts must remain independent from effect readiness") &&
         Expect(!JsonBoolField(*diagnostics,
                               "video_noise_removal_shared_device_imported",
                               true) &&
                    !JsonBoolField(
                        *diagnostics,
                        "video_noise_removal_queue_ownership_explicit", true),
                "video denoise must not claim shared device/queue ownership") &&
         Expect(
             !JsonBoolField(*diagnostics,
                            "video_noise_removal_model_pack_selected", true) &&
                 !JsonBoolField(
                     *diagnostics,
                     "video_noise_removal_artifact_contract_validated", true),
             "ONNX-only packs must not claim a Vulkan artifact contract") &&
         Expect(
             !JsonBoolField(
                 *diagnostics,
                 "video_noise_removal_fully_device_resident_tensor_io", true) &&
                 !JsonBoolField(
                     *diagnostics,
                     "video_noise_removal_device_resident_preprocess", true) &&
                 !JsonBoolField(
                     *diagnostics,
                     "video_noise_removal_device_resident_postprocess", true),
             "host pre/post and tensor I/O must remain explicit") &&
         Expect(
             !JsonBoolField(
                 *diagnostics,
                 "video_noise_removal_temporal_history_device_resident",
                 true) &&
                 !JsonBoolField(*diagnostics,
                                "video_noise_removal_temporal_history_bounded",
                                true) &&
                 !JsonBoolField(*diagnostics,
                                "video_noise_removal_history_reset_on_disable",
                                true) &&
                 !JsonBoolField(
                     *diagnostics,
                     "video_noise_removal_history_reset_on_reconfigure",
                     true) &&
                 !JsonBoolField(
                     *diagnostics,
                     "video_noise_removal_capture_sequence_discontinuity_reset",
                     true),
             "Vulkan temporal residency/reset facts must fail closed") &&
         Expect(!JsonBoolField(*diagnostics,
                               "video_noise_removal_selectable_cpu_fallback",
                               true),
                "internal CPU behavior is not a selectable fallback") &&
         Expect(JsonNumberField(*diagnostics,
                                "video_noise_removal_dispatch_count",
                                1.0) == 0.0 &&
                    JsonNumberField(
                        *diagnostics,
                        "video_noise_removal_temporal_history_reset_count",
                        1.0) == 0.0 &&
                    JsonNumberField(*diagnostics,
                                    "video_noise_removal_cpu_readback_count",
                                    1.0) == 0.0 &&
                    JsonNumberField(*diagnostics,
                                    "video_noise_removal_cpu_fallback_count",
                                    1.0) == 0.0,
                "blocked video denoise must prove zero frame/history work");
}

bool TestOpenVulkanDisabledBuildKeepsVideoNoiseRemovalSchema() {
#if STUDIOCAST_ENABLE_OPEN_VULKAN
  return true;
#else
  const DiagnosticsJsonSnapshot snapshot = ComputeDiagnosticsJsonSnapshot();
  studiocast::util::json::Value value;
  std::string error;
  if (!studiocast::util::json::Parse(snapshot.open_vulkan, &value, &error)) {
    std::cerr << "Vulkan-off diagnostics should parse: " << error << "\n";
    return false;
  }
  const JsonObject *diagnostics = value.AsObject();
  if (!diagnostics)
    return false;
  const JsonObject *blocked =
      ObjectAt(*diagnostics, "blocked_effects",
               "Vulkan-off blocked_effects should be present");
  if (!blocked)
    return false;
  const std::string *blockedReason =
      StringAt(*blocked, "video_noise_removal",
               "Vulkan-off video-denoise blocker should be present");
  const std::string *primaryBlocker =
      StringAt(*diagnostics, "video_noise_removal_blocker_code",
               "Vulkan-off video-denoise primary blocker should be present");
  const std::string *mirrorBlocker =
      StringAt(*blocked, "mirror",
               "Vulkan-off mirror blocker should be present");
  const std::string *vignetteBlocker =
      StringAt(*blocked, "vignette",
               "Vulkan-off vignette blocker should be present");
  const std::string *autoFrameBlocker =
      StringAt(*blocked, "auto_frame",
               "Vulkan-off Auto Frame blocker should be present");
  const std::string *vignetteContract = StringAt(
      *diagnostics, "vignette_parameter_contract",
      "Vulkan-off vignette parameter contract should be present");
  if (!blockedReason || !primaryBlocker || !mirrorBlocker ||
      !vignetteBlocker || !autoFrameBlocker || !vignetteContract)
    return false;

  return Expect(*mirrorBlocker == "vulkan_backend_disabled_in_build" &&
                    *vignetteBlocker == "vulkan_backend_disabled_in_build" &&
                    *autoFrameBlocker == "vulkan_backend_disabled_in_build",
                "Vulkan-off pixel/Auto Frame effects must fail closed with "
                "the exact build blocker") &&
         Expect(!JsonBoolField(*diagnostics, "mirror_production_ready", true) &&
                    !JsonBoolField(
                        *diagnostics,
                        "vignette_fixed_center_production_ready", true) &&
                    !JsonBoolField(*diagnostics,
                                   "auto_frame_crop_stage_implemented", true) &&
                    !JsonBoolField(*diagnostics,
                                   "auto_frame_production_ready", true) &&
                    !JsonBoolField(*diagnostics,
                                   "auto_frame_selectable_cpu_fallback", true) &&
                    *vignetteContract == "fixed_center",
                "Vulkan-off additive readiness facts must remain false while "
                "preserving the fixed-center contract") &&
         Expect(*blockedReason == "open_vulkan_video_noise_removal_unavailable",
                "Vulkan-off schema must preserve the stable outer reason") &&
         Expect(*primaryBlocker ==
                    "open_vulkan_video_noise_removal_runtime_unavailable",
                "Vulkan-off schema must preserve the stable primary blocker") &&
         Expect(!JsonBoolField(*diagnostics,
                               "video_noise_removal_backend_compiled", true) &&
                    !JsonBoolField(
                        *diagnostics,
                        "video_noise_removal_non_cpu_device_selected", true) &&
                    !JsonBoolField(
                        *diagnostics,
                        "video_noise_removal_compute_queue_available", true) &&
                    !JsonBoolField(*diagnostics,
                                   "video_noise_removal_context_healthy", true),
                "Vulkan-off hardware/compiled facts must stay false") &&
         Expect(
             !JsonBoolField(*diagnostics,
                            "video_noise_removal_temporal_history_bounded",
                            true) &&
                 !JsonBoolField(*diagnostics,
                                "video_noise_removal_history_reset_on_disable",
                                true) &&
                 !JsonBoolField(
                     *diagnostics,
                     "video_noise_removal_history_reset_on_reconfigure",
                     true) &&
                 !JsonBoolField(
                     *diagnostics,
                     "video_noise_removal_capture_sequence_discontinuity_reset",
                     true),
             "Vulkan-off temporal contract facts must stay false") &&
         Expect(JsonNumberField(*diagnostics,
                                "video_noise_removal_dispatch_count",
                                1.0) == 0.0 &&
                    JsonNumberField(
                        *diagnostics,
                        "video_noise_removal_temporal_history_reset_count",
                        1.0) == 0.0 &&
                    JsonNumberField(*diagnostics,
                                    "video_noise_removal_cpu_readback_count",
                                    1.0) == 0.0 &&
                    JsonNumberField(*diagnostics,
                                    "video_noise_removal_cpu_fallback_count",
                                    1.0) == 0.0,
                "Vulkan-off video denoise must report zero work counters");
#endif
}

bool TestExactOpenVulkanPixelEffectsReadyAndIgnoreMattingBlocker() {
  studiocast::video::VirtualCameraServiceConfig mirrorConfig;
  mirrorConfig.enabled = true;
  mirrorConfig.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::vulkan;
  mirrorConfig.pipeline.effects.mirror = true;
  const std::string mirrorDiagnostics = ReadyOpenVulkanPixelDiagnostics(
      "[\"mirror\"]", "\"mirror_production_ready\":true,"
                      "\"mirror_readiness_code\":"
                      "\"open_vulkan_mirror_production_ready\","
                      "\"mirror_blocker_code\":\"\"");
  const std::string mirrorStatus = StatusForVideoConfigWithDiagnostics(
      mirrorConfig, "", "", mirrorDiagnostics);
  const ReadinessFields mirror = ReadinessEntryFor(mirrorStatus, "mirror");
  const ComputeFields mirrorCompute = ComputeFieldsFor(mirrorStatus);

  studiocast::video::VirtualCameraServiceConfig vignetteConfig;
  vignetteConfig.enabled = true;
  vignetteConfig.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::vulkan;
  vignetteConfig.pipeline.effects.vignette.enabled = true;
  const std::string vignetteDiagnostics = ReadyOpenVulkanPixelDiagnostics(
      "[\"vignette\"]",
      "\"vignette_fixed_center_production_ready\":true,"
      "\"vignette_readiness_code\":"
      "\"open_vulkan_vignette_fixed_center_production_ready\","
      "\"vignette_blocker_code\":\"\","
      "\"vignette_parameter_contract\":\"fixed_center\"");
  const std::string vignetteStatus = StatusForVideoConfigWithDiagnostics(
      vignetteConfig, "", "", vignetteDiagnostics);
  const ReadinessFields vignette =
      ReadinessEntryFor(vignetteStatus, "vignette");
  const ComputeFields vignetteCompute = ComputeFieldsFor(vignetteStatus);

  const auto mirrorPlan = studiocast::video::effects::BuildBroadcastEffectsPlan(
      mirrorConfig.pipeline.effects);
  return Expect(studiocast::video::effects::BroadcastEffectsPlanRequestsCompute(
                    mirrorPlan) &&
                    VideoConfigRequestsCompute(mirrorConfig.pipeline.effects),
                "daemon compute determination must use the canonical "
                "mirror-aware helper") &&
         Expect(mirror.present && mirror.state == "ready" &&
                    mirror.backend == "open_vulkan" && mirror.reason.empty(),
                "exact mirror production evidence must report Vulkan ready") &&
         Expect(mirrorCompute.present && mirrorCompute.resolved == "vulkan" &&
                    mirrorCompute.active == "vulkan" &&
                    mirrorCompute.provider_mode == "open_vulkan" &&
                    !mirrorCompute.fallback_active &&
                    mirrorCompute.fallback.empty(),
                "stopped mirror-only explicit Vulkan must resolve Vulkan") &&
         Expect(vignette.present && vignette.state == "ready" &&
                    vignette.backend == "open_vulkan" &&
                    vignette.reason.empty(),
                "exact fixed-center vignette evidence must report Vulkan "
                "ready") &&
         Expect(vignetteCompute.present &&
                    vignetteCompute.resolved == "vulkan" &&
                    vignetteCompute.active == "vulkan" &&
                    !vignetteCompute.fallback_active &&
                    vignetteCompute.fallback.empty(),
                "stopped vignette-only explicit Vulkan must resolve Vulkan");
}

bool TestLiveOpenVulkanPixelReadinessRequiresActiveMap() {
  studiocast::video::VirtualCameraServiceConfig config;
  config.enabled = true;
  config.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::vulkan;
  config.pipeline.effects.mirror = true;
  const std::string diagnostics = ReadyOpenVulkanPixelDiagnostics(
      "[\"mirror\"]", "\"mirror_production_ready\":true,"
                      "\"mirror_readiness_code\":"
                      "\"open_vulkan_mirror_production_ready\","
                      "\"mirror_blocker_code\":\"\"");

  studiocast::video::VirtualCameraServiceStatus inactive;
  inactive.service_running = true;
  inactive.pipeline.running = true;
  const std::string inactiveStatus =
      StatusForVideoStateWithDiagnostics(inactive, config, "", "", diagnostics);
  const ReadinessFields missingMap =
      ReadinessEntryFor(inactiveStatus, "mirror");
  const ComputeFields missingMapCompute = ComputeFieldsFor(inactiveStatus);

  auto active = inactive;
  active.pipeline.effects_backends = "mirror:open_vulkan";
  const std::string activeStatus =
      StatusForVideoStateWithDiagnostics(active, config, "", "", diagnostics);
  const ReadinessFields activeMirror =
      ReadinessEntryFor(activeStatus, "mirror");
  const ComputeFields activeCompute = ComputeFieldsFor(activeStatus);

  return Expect(missingMap.state == "backend_unavailable" &&
                    missingMap.reason ==
                        "open_vulkan_mirror_live_stage_not_active",
                "a running pipeline must not claim mirror without active-map "
                "evidence") &&
         Expect(missingMapCompute.active == "cpu" &&
                    missingMapCompute.fallback ==
                        "open_vulkan_mirror_live_stage_not_active",
                "running mirror fallback must be effect-specific") &&
         Expect(activeMirror.state == "ready" &&
                    activeMirror.backend == "open_vulkan",
                "authoritative live Vulkan mirror attribution must be ready") &&
         Expect(activeCompute.resolved == "vulkan" &&
                    activeCompute.active == "vulkan" &&
                    !activeCompute.fallback_active,
                "active map must repair aggregate compute attribution");
}

bool TestOpenVulkanPixelEvidenceFailsClosedWhenInconsistent() {
  studiocast::video::VirtualCameraServiceConfig config;
  config.enabled = true;
  config.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::vulkan;
  config.pipeline.effects.mirror = true;

  const std::vector<std::pair<std::string, std::string>> cases = {
      {"missing production fact",
       ReadyOpenVulkanPixelDiagnostics("[\"mirror\"]",
                                       "\"mirror_blocker_code\":\"\"")},
      {"false production fact",
       ReadyOpenVulkanPixelDiagnostics("[\"mirror\"]",
                                       "\"mirror_production_ready\":false,"
                                       "\"mirror_readiness_code\":\"\","
                                       "\"mirror_blocker_code\":\"\"")},
      {"wrong success code",
       ReadyOpenVulkanPixelDiagnostics(
           "[\"mirror\"]", "\"mirror_production_ready\":true,"
                           "\"mirror_readiness_code\":\"wrong_success_code\","
                           "\"mirror_blocker_code\":\"\"")},
      {"missing available membership",
       ReadyOpenVulkanPixelDiagnostics(
           "[]", "\"mirror_production_ready\":true,"
                 "\"mirror_readiness_code\":"
                 "\"open_vulkan_mirror_production_ready\","
                 "\"mirror_blocker_code\":\"\"")},
  };

  bool ok = true;
  for (const auto &[name, diagnostics] : cases) {
    const ReadinessFields entry = ReadinessEntryFor(
        StatusForVideoConfigWithDiagnostics(config, "", "", diagnostics),
        "mirror");
    ok = Expect(entry.present && entry.backend == "open_vulkan" &&
                    entry.state == "backend_unavailable" &&
                    entry.reason ==
                        "open_vulkan_mirror_production_evidence_inconsistent",
                ("mirror must fail closed for " + name).c_str()) &&
         ok;
  }
  return ok;
}

bool TestOpenVulkanPixelCommonHardwareFactsFailClosed() {
  studiocast::video::VirtualCameraServiceConfig config;
  config.enabled = true;
  config.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::vulkan;
  config.pipeline.effects.mirror = true;

  const auto diagnostics =
      [](bool compiled, bool instance, bool nonCpu, bool cpu, bool queue,
         bool contextHealthy, bool diagnosticsOk,
         const std::string &contextReason, const std::string &effectBlocker) {
        return std::string("{\"compiled_enabled\":") +
               (compiled ? "true" : "false") +
               ",\"ok\":" + (diagnosticsOk ? "true" : "false") +
               ",\"runtime_library_found\":true,\"instance_created\":" +
               (instance ? "true" : "false") +
               ","
               "\"physical_device_found\":true,\"non_cpu_device_selected\":" +
               (nonCpu ? "true" : "false") +
               ",\"cpu_device_selected\":" + (cpu ? "true" : "false") +
               ",\"compute_queue_available\":" + (queue ? "true" : "false") +
               ",\"logical_device_created\":true,\"context_created\":true,"
               "\"context_healthy\":" +
               (contextHealthy ? "true" : "false") +
               ",\"production_hardware_ready\":" +
               (nonCpu && queue && contextHealthy ? "true" : "false") +
               ",\"shader_pipeline_created\":true,"
               "\"context_failure_reason\":\"" +
               contextReason +
               "\",\"available_effects\":[\"mirror\"],"
               "\"blocked_effects\":{},\"mirror_production_ready\":true,"
               "\"mirror_readiness_code\":"
               "\"open_vulkan_mirror_production_ready\","
               "\"mirror_blocker_code\":\"" +
               effectBlocker + "\"}";
      };
  const std::vector<std::pair<std::string, std::string>> cases = {
      {diagnostics(false, false, false, false, false, false, false, "",
                   "vulkan_backend_disabled_in_build"),
       "vulkan_backend_disabled_in_build"},
      {diagnostics(true, false, true, false, true, true, true, "", ""),
       "vulkan_instance_create_failed"},
      {diagnostics(true, true, true, false, true, true, false, "", ""),
       "open_vulkan_utility_kernels_unavailable"},
      {diagnostics(true, true, false, true, true, true, true, "", ""),
       "vulkan_only_cpu_devices_available"},
      {diagnostics(true, true, true, false, false, true, true, "", ""),
       "vulkan_no_compute_queue"},
      {diagnostics(true, true, true, false, true, false, true,
                   "vulkan_device_lost", ""),
       "vulkan_device_lost"},
  };

  bool ok = true;
  for (const auto &[fixture, reason] : cases) {
    const ReadinessFields entry = ReadinessEntryFor(
        StatusForVideoConfigWithDiagnostics(config, "", "", fixture), "mirror");
    ok = Expect(entry.present && entry.state == "backend_unavailable" &&
                    entry.reason == reason,
                ("mirror hardware failure must report " + reason).c_str()) &&
         ok;
  }
  return ok;
}

bool TestPixelEffectsRequireRealBackendPaths() {
  bool ok = true;
  for (const auto pref :
       {studiocast::video::ComputeBackendPreference::cpu,
        studiocast::video::ComputeBackendPreference::cuda,
        studiocast::video::ComputeBackendPreference::auto_select}) {
    studiocast::video::VirtualCameraServiceConfig config;
    config.enabled = true;
    config.pipeline.compute_backend = pref;
    config.pipeline.effects.mirror = true;
    const ReadinessFields entry =
        ReadinessEntryFor(StatusForVideoConfig(config), "mirror");
    ok = Expect(entry.present && entry.state == "backend_unavailable" &&
                    entry.backend != "builtin",
                "CPU/CUDA/auto mirror must not claim a live path") &&
         ok;
  }

  studiocast::video::VirtualCameraServiceConfig mirrorConfig;
  mirrorConfig.enabled = true;
  mirrorConfig.pipeline.effects.mirror = true;
  studiocast::video::VirtualCameraServiceStatus mirrorStatus;
  mirrorStatus.service_running = true;
  mirrorStatus.pipeline.effects_backends = "mirror:maxine";
  const ReadinessFields maxineMirror =
      ReadinessEntryFor(StatusForVideoStateWithDiagnostics(
                            mirrorStatus, mirrorConfig, "", "", ""),
                        "mirror");
  ok = Expect(maxineMirror.state == "backend_unavailable" &&
                  maxineMirror.reason == "mirror_no_live_backend_path",
              "Maxine mirror attribution must not invent an implementation") &&
       ok;

  studiocast::video::VirtualCameraServiceConfig cpuVignetteConfig;
  cpuVignetteConfig.enabled = true;
  cpuVignetteConfig.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::cpu;
  cpuVignetteConfig.pipeline.effects.vignette.enabled = true;
  const ReadinessFields cpuVignette =
      ReadinessEntryFor(StatusForVideoConfig(cpuVignetteConfig), "vignette");
  ok = Expect(cpuVignette.state == "backend_unavailable" &&
                  cpuVignette.reason == "vignette_no_selectable_cpu_path",
              "vignette must not expose a CPU path") &&
       ok;

  studiocast::video::VirtualCameraServiceConfig cudaVignetteConfig;
  cudaVignetteConfig.enabled = true;
  cudaVignetteConfig.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::cuda;
  cudaVignetteConfig.pipeline.effects.vignette.enabled = true;
  studiocast::video::VirtualCameraServiceStatus cudaVignetteStatus;
  cudaVignetteStatus.service_running = true;
  cudaVignetteStatus.pipeline.running = true;
  cudaVignetteStatus.pipeline.effects_backends = "vignette:cuda";
  const ReadinessFields cudaVignette =
      ReadinessEntryFor(StatusForVideoStateWithDiagnostics(
                            cudaVignetteStatus, cudaVignetteConfig, "", "", ""),
                        "vignette");
  ok = Expect(cudaVignette.state == "ready" && cudaVignette.backend == "cuda",
              "an authoritative active CUDA vignette may report ready") &&
       ok;

  studiocast::video::VirtualCameraServiceConfig maxineVignetteConfig;
  maxineVignetteConfig.enabled = true;
  maxineVignetteConfig.pipeline.effects.auto_frame.enabled = true;
  maxineVignetteConfig.pipeline.effects.vignette.enabled = true;
  studiocast::video::VirtualCameraServiceStatus maxineVignetteStatus;
  maxineVignetteStatus.service_running = true;
  maxineVignetteStatus.pipeline.running = true;
  maxineVignetteStatus.pipeline.effects_backends = "auto_frame:maxine_ar_cuda";
  const ReadinessFields maxineVignette = ReadinessEntryFor(
      StatusForVideoStateWithDiagnostics(maxineVignetteStatus,
                                         maxineVignetteConfig, "", "", ""),
      "vignette");
  return Expect(maxineVignette.state == "ready" &&
                    maxineVignette.backend == "maxine_ar_cuda",
                "an attached vignette may inherit authoritative active "
                "Maxine attribution") &&
         ok;
}

bool TestTrackedCenterVignetteFailsClosedWhenAutoFrameIsRetained() {
  studiocast::video::VirtualCameraServiceConfig config;
  config.enabled = true;
  config.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::vulkan;
  config.pipeline.effects.auto_frame.enabled = true;
  config.pipeline.effects.vignette.enabled = true;
  config.pipeline.effects.vignette.center_on_tracked_face = true;
  const std::string diagnostics = ReadyOpenVulkanPixelDiagnostics(
      "[\"auto_frame\",\"vignette\"]",
      "\"vignette_fixed_center_production_ready\":true,"
      "\"vignette_readiness_code\":"
      "\"open_vulkan_vignette_fixed_center_production_ready\","
      "\"vignette_blocker_code\":\"\","
      "\"vignette_parameter_contract\":\"fixed_center\"");
  const std::string status =
      StatusForVideoConfigWithDiagnostics(config, "", "", diagnostics);
  const ReadinessFields vignette = ReadinessEntryFor(status, "vignette");
  const ComputeFields compute = ComputeFieldsFor(status);
  return Expect(vignette.state == "backend_unavailable" &&
                    vignette.reason ==
                        "vulkan_vignette_tracked_center_not_supported",
                "retained tracked-center Vulkan vignette must fail closed") &&
         Expect(compute.resolved == "vulkan" && compute.active == "vulkan",
                "retained Auto Frame evidence may still resolve Vulkan");
}

bool TestMixedVulkanRequestKeepsReadyPixelEffectAndExactBlocker() {
  studiocast::video::VirtualCameraServiceConfig config;
  config.enabled = true;
  config.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::vulkan;
  config.pipeline.effects.mirror = true;
  config.pipeline.effects.virtual_background.mode =
      studiocast::video::effects::VirtualBackgroundMode::blur;
  const std::string diagnostics =
      ReadyOpenVulkanPixelDiagnostics("[\"mirror\"]",
                                      "\"mirror_production_ready\":true,"
                                      "\"mirror_readiness_code\":"
                                      "\"open_vulkan_mirror_production_ready\","
                                      "\"mirror_blocker_code\":\"\"",
                                      "",
                                      "{\"virtual_background.blur\":"
                                      "\"open_vulkan_matting_unavailable\"}");
  const std::string status =
      StatusForVideoConfigWithDiagnostics(config, "", "", diagnostics);
  const ReadinessFields mirror = ReadinessEntryFor(status, "mirror");
  const ReadinessFields blur =
      ReadinessEntryFor(status, "virtual_background.blur");
  const ComputeFields compute = ComputeFieldsFor(status);
  return Expect(mirror.state == "ready" && mirror.backend == "open_vulkan",
                "mixed request must retain ready mirror") &&
         Expect(blur.state == "backend_unavailable" &&
                    blur.reason == "open_vulkan_matting_unavailable",
                "mixed request must preserve the blocked sibling reason") &&
         Expect(compute.resolved == "vulkan" && compute.active == "vulkan" &&
                    !compute.fallback_active && compute.fallback.empty(),
                "a retained ready Vulkan effect must keep compute on Vulkan");
}

bool TestUnavailableVulkanRequestUsesRequestedEffectBlocker() {
  studiocast::video::VirtualCameraServiceConfig config;
  config.enabled = true;
  config.pipeline.compute_backend =
      studiocast::video::ComputeBackendPreference::vulkan;
  config.pipeline.effects.eye_contact.enabled = true;
  const std::string diagnostics =
      "{\"compiled_enabled\":true,\"ok\":true,"
      "\"runtime_library_found\":true,\"physical_device_found\":true,"
      "\"non_cpu_device_selected\":true,"
      "\"compute_queue_available\":true,"
      "\"logical_device_created\":true,\"context_created\":true,"
      "\"context_healthy\":true,\"production_hardware_ready\":true,"
      "\"shader_pipeline_created\":true,\"available_effects\":[],"
      "\"blocked_effects\":{\"eye_contact\":"
      "\"open_vulkan_eye_contact_unavailable\"},"
      "\"blocked_reason\":\"open_vulkan_matting_unavailable\","
      "\"eye_contact_blocker_code\":"
      "\"open_vulkan_eye_contact_runtime_unavailable\","
      "\"eye_contact_detail\":\"exact eye contact runtime missing\"}";
  const ComputeFields compute = ComputeFieldsFor(
      StatusForVideoConfigWithDiagnostics(config, "", "", diagnostics));
  return Expect(compute.present && compute.resolved == "cpu" &&
                    compute.active == "cpu" && compute.fallback_active,
                "an unavailable sole Vulkan effect must fail over") &&
         Expect(compute.fallback.find("open_vulkan_eye_contact_unavailable") !=
                        std::string::npos &&
                    compute.fallback.find(
                        "open_vulkan_eye_contact_runtime_unavailable") !=
                        std::string::npos &&
                    compute.fallback.find("open_vulkan_matting_unavailable") ==
                        std::string::npos,
                "fallback must use the requested effect blocker only");
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
  ok = TestVideoComputeStatusReportsCachedCountersAndProvider() && ok;
  ok = TestVideoComputeTransferTotalsDoNotDoubleCountSubcounters() && ok;
  ok = TestVulkanVirtualBackgroundBlurDebugCounters() && ok;
  ok = TestVulkanVirtualBackgroundRemoveDebugCounters() && ok;
  ok = TestVulkanVirtualBackgroundReplaceDebugCounters() && ok;
  ok = TestVulkanVirtualKeyLightDebugCounters() && ok;
  ok = TestVideoConfigMapsComputeBackendPreference() && ok;
  ok = TestPersistentVulkanAdapterConfigAndStatus() && ok;
  ok = TestVideoStatusReportsCaptureFallbackState() && ok;
  ok = TestVideoStatusReportsConfiguredDevicesWhenPipelineIdle() && ok;
  ok = TestExplicitOpenCudaEffectUnknownWithoutDiagnostics() && ok;
  ok = TestExplicitVulkanVirtualBackgroundReportsOpenVulkanBlocked() && ok;
  ok = TestExplicitVulkanEyeContactReportsExactFailClosedFacts() && ok;
  ok = TestExplicitVulkanVideoNoiseRemovalReportsExactFailClosedFacts() && ok;
  ok = TestOpenVulkanDisabledBuildKeepsVideoNoiseRemovalSchema() && ok;
  ok = TestExactOpenVulkanPixelEffectsReadyAndIgnoreMattingBlocker() && ok;
  ok = TestLiveOpenVulkanPixelReadinessRequiresActiveMap() && ok;
  ok = TestOpenVulkanPixelEvidenceFailsClosedWhenInconsistent() && ok;
  ok = TestOpenVulkanPixelCommonHardwareFactsFailClosed() && ok;
  ok = TestPixelEffectsRequireRealBackendPaths() && ok;
  ok = TestTrackedCenterVignetteFailsClosedWhenAutoFrameIsRetained() && ok;
  ok = TestMixedVulkanRequestKeepsReadyPixelEffectAndExactBlocker() && ok;
  ok = TestUnavailableVulkanRequestUsesRequestedEffectBlocker() && ok;
  ok = TestAudioStatusReportsResolvedSourceAndWarnings() && ok;
  ok = TestAudioStatusPropagatesSourceErrorFromService() && ok;
  return ok ? 0 : 1;
}
