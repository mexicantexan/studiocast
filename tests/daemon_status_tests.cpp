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
         Expect(fallback->find("Vulkan compute backend requested") !=
                    std::string::npos,
                "vulkan unavailable status should include fallback reason") &&
         Expect(*degraded == *fallback,
                "vulkan unavailable degraded reason should match fallback") &&
         Expect(ArrayContainsString(activeEngines, "cpu"),
                "vulkan unavailable status should report active cpu engine") &&
         Expect(vulkanUnavailable->find("Vulkan compute backend requested") !=
                    std::string::npos,
                "vulkan unavailable reason should be normalized") &&
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
  ok = TestBuiltinEffectReadyWithoutDiagnostics() && ok;
  ok = TestAudioStatusReportsResolvedSourceAndWarnings() && ok;
  ok = TestAudioStatusPropagatesSourceErrorFromService() && ok;
  return ok ? 0 : 1;
}
