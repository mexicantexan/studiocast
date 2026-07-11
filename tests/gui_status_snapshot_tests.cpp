#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include <QApplication>
#include <QAbstractItemModel>
#include <QComboBox>
#include <QEventLoop>
#include <QFrame>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QTimer>
#include <unistd.h>

#include "core/ipc/daemon_server.h"
#include "core/ipc/daemon_socket.h"
#include "gui/pages/engines_models_page.h"
#include "gui/pages/video_page.h"
#include "gui/status/daemon_status_snapshot.h"
#include "gui/status/pending_daemon_write_guard.h"
#include "gui/status/status_poller.h"

namespace {

namespace fs = std::filesystem;
using namespace std::chrono_literals;

bool Expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

class ScopedRuntimeDir {
public:
  explicit ScopedRuntimeDir(const std::string &name) {
    hadOld_ = std::getenv("XDG_RUNTIME_DIR") != nullptr;
    if (hadOld_)
      old_ = std::getenv("XDG_RUNTIME_DIR");

    std::error_code ec;
    const auto tmp = fs::temp_directory_path(ec);
    if (ec) {
      error_ = "failed to resolve temp directory: " + ec.message();
      return;
    }

    dir_ =
        tmp / (name + "-" + std::to_string(static_cast<long long>(::getpid())));
    fs::remove_all(dir_, ec);
    fs::create_directories(dir_, ec);
    if (ec) {
      error_ = "failed to create temp runtime dir: " + ec.message();
      return;
    }

    if (::setenv("XDG_RUNTIME_DIR", dir_.string().c_str(), 1) != 0)
      error_ = std::string("setenv failed: ") + std::strerror(errno);
  }

  ~ScopedRuntimeDir() {
    if (hadOld_) {
      (void)::setenv("XDG_RUNTIME_DIR", old_.c_str(), 1);
    } else {
      (void)::unsetenv("XDG_RUNTIME_DIR");
    }

    std::error_code ec;
    fs::remove_all(dir_, ec);
  }

  bool ok() const { return error_.empty(); }
  const std::string &error() const { return error_; }

private:
  fs::path dir_;
  bool hadOld_ = false;
  std::string old_;
  std::string error_;
};

bool TestUnreachableStatus() {
  const auto s = studiocast::gui::DaemonStatusSnapshot::Unreachable(
      QStringLiteral("connect failed"));
  return Expect(!s.reachable, "unreachable snapshot should not be reachable") &&
         Expect(s.camera.state ==
                    studiocast::gui::ReadinessState::DaemonUnavailable,
                "camera should report daemon unavailable") &&
         Expect(s.ServiceSummary() == QStringLiteral("Service unavailable"),
                "service summary should report unavailable");
}

bool TestStatusJsonCompatibilityShapes() {
  const QString json = QStringLiteral(
      R"({
        "version":"0.2.0",
        "git_sha":"abc123",
        "socket":"/run/user/1000/studiocast/studiocastd.sock",
        "service_running":true,
        "maxine":{
          "supported":false,
          "ok":false,
          "summary":"Maxine unavailable.",
          "blocked_reason":"driver_missing",
          "blocked_details":["Install a supported NVIDIA driver."]
        },
        "engines":{
          "open_cuda":{
            "ok":true,
            "installed_models":["matting"],
            "models":[{"id":"matting","display_name":"Matting"}],
            "missing_models":{"denoise":"Missing denoise model."},
            "install_hints":["studiocast-open install-hints"]
          },
          "open_vulkan":{
            "compiled_enabled":false,
            "ok":false,
            "runtime_library_found":false,
            "compute_queue_available":false,
            "shader_pipeline_created":false,
            "fallback_reason":"disabled_in_build",
            "install_hints":["rebuild with vulkan"]
          }
        },
        "open_audio":{
          "ok":true,
          "installed_models":["rnnoise"],
          "models":[{"id":"rnnoise","display_name":"RNNoise"}],
          "missing_models":{}
        },
        "video":{
          "enabled":false,
          "compute":{
            "preference":"vulkan",
            "resolved_backend":"cpu",
            "active_backend":"cpu",
            "fallback_reason":"Vulkan unavailable.",
            "degraded_reason":"Vulkan unavailable.",
            "active_engines":["cpu"],
            "fallback":{
              "active":true,
              "from":"vulkan",
              "to":"cpu",
              "code":"vulkan_unavailable",
              "detail":"Vulkan unavailable."
            },
            "provider":{
              "mode":"cpu",
              "active_provider":"CPU",
              "device":"host",
              "tensor_io_mode":"host"
            },
            "cpu_tails":{
              "active":true,
              "stages":["key_light","denoise"]
            },
            "transfers":{
              "open_cuda":{"active_frames":0,"uploads":0,"downloads":0,"forced_syncs":0},
              "open_vulkan":{"active_frames":0,"uploads":0,"downloads":0,"dispatches":0,"forced_syncs":0},
              "maxine":{"active_frames":0,"uploads":0,"downloads":0,"forced_syncs":0}
            }
          },
          "virtual_device_present":true,
          "virtual_device_available":true,
          "consumer_count":0,
          "video_effects":{"engine":"open_cuda"},
          "effects_plan":{
            "disabled":[
              {"id":"mirror","reason":"Disabled: mirror is not supported."}
            ]
          },
          "pipeline":{"running":false,"starting":false}
        },
        "audio":{
          "mic_present":true,
          "source_error":"",
          "audio_effects":{"engine":"open_source"},
          "pipeline":{"running":false,"starting":false,"last_error":""},
          "speakers":{
            "present":true,
            "target_sink_error":"",
            "routing_active":false,
            "route_mode":"off",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  const auto s = studiocast::gui::DaemonStatusSnapshot::FromJson(json);
  return Expect(s.reachable, "snapshot should be reachable") &&
         Expect(s.parsed, "snapshot should parse") &&
         Expect(s.rawJson == json, "raw json should be preserved") &&
         Expect(s.version == QStringLiteral("0.2.0"), "version should parse") &&
         Expect(s.serviceRunning, "service_running should parse") &&
         Expect(s.camera.state == studiocast::gui::ReadinessState::Ready,
                "camera should be ready") &&
         Expect(s.camera.disabledReasons.size() == 1,
                "camera disabled effect reasons should parse") &&
         Expect(s.videoEffectPlan.disabledEffects.size() == 1,
                "legacy video.effects_plan should parse into typed plan") &&
         Expect(s.videoEffectPlan.disabledEffects.front().effectId ==
                    QStringLiteral("mirror"),
                "legacy disabled effect id should parse") &&
         Expect(s.microphone.state == studiocast::gui::ReadinessState::Ready,
                "microphone should be ready") &&
         Expect(s.speakers.state == studiocast::gui::ReadinessState::Ready,
                "speakers should be ready") &&
         Expect(s.videoEffectsEnginePreference == QStringLiteral("open_cuda"),
                "video engine preference should parse") &&
         Expect(s.videoComputePreference == QStringLiteral("vulkan"),
                "video compute preference should parse") &&
         Expect(s.videoComputeResolvedBackend == QStringLiteral("cpu"),
                "video compute resolved backend should parse") &&
         Expect(s.videoComputeActiveBackend == QStringLiteral("cpu"),
                "video compute active backend should parse") &&
         Expect(s.videoComputeFallbackReason ==
                    QStringLiteral("Vulkan unavailable."),
                "video compute fallback reason should parse") &&
         Expect(s.videoComputeDegradedReason ==
                    QStringLiteral("Vulkan unavailable."),
                "video compute degraded reason should parse") &&
         Expect(s.videoComputeActiveEngines.contains(QStringLiteral("cpu")),
                "video compute active engines should parse") &&
         Expect(s.videoComputeFallbackActive,
                "video compute fallback active should parse") &&
         Expect(s.videoComputeFallbackCode ==
                    QStringLiteral("vulkan_unavailable"),
                "video compute fallback code should parse") &&
         Expect(s.videoComputeProviderMode == QStringLiteral("cpu"),
                "video compute provider mode should parse") &&
         Expect(s.videoComputeActiveProvider == QStringLiteral("CPU"),
                "video compute active provider should parse") &&
         Expect(s.videoComputeTensorIoMode == QStringLiteral("host"),
                "video compute tensor I/O mode should parse") &&
         Expect(s.videoComputeCpuTailsActive,
                "video compute CPU tail state should parse") &&
         Expect(s.videoComputeCpuTailStages.contains(
                    QStringLiteral("key_light")),
                "video compute CPU tail stages should parse") &&
         Expect(s.audioEffectsEnginePreference == QStringLiteral("open_source"),
                "audio engine preference should parse") &&
         Expect(s.maxine.present && !s.maxine.supported,
                "top-level maxine diagnostics should parse") &&
         Expect(s.openCuda.present && s.openCuda.ok &&
                    s.openCuda.missingModelCount == 1,
                "nested open_cuda diagnostics should parse") &&
         Expect(s.openCuda.missingModels.contains(
                    QStringLiteral("denoise: Missing denoise model.")),
                "open_cuda missing model details should parse") &&
         Expect(s.openCuda.installHints.contains(
                    QStringLiteral("studiocast-open install-hints")),
                "open_cuda install hints should parse") &&
         Expect(s.openVulkan.present && !s.openVulkan.ok,
                "nested open_vulkan diagnostics should parse") &&
         Expect(s.openVulkan.installHints.contains(
                    QStringLiteral("rebuild with vulkan")),
                "open_vulkan install hints should parse") &&
         Expect(s.openAudio.present && s.openAudio.ok &&
                    s.openAudio.installedModelCount == 1,
                "top-level open_audio diagnostics should parse");
}

bool TestNestedPipelineEffectsPlanAndRawEffectsPreservation() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "video":{
          "enabled":false,
          "virtual_device_present":true,
          "virtual_device_available":true,
          "consumer_count":0,
          "video_effects":{
            "engine":"open_cuda",
            "virtual_background":{
              "mode":"blur",
              "model_id":"matting-good",
              "hidden_future_field":{"kept":true}
            }
          },
          "pipeline":{
            "running":false,
            "starting":false,
            "effects_plan":{
              "ordered":["mirror","virtual_background.blur"],
              "vignette_attach_to":"virtual_background.blur",
              "disabled":[
                {"id":"virtual_key_light","reason":"disabled by rule"}
              ]
            }
          }
        },
        "audio":{
          "enabled":false,
          "mic_present":true,
          "source_error":"",
          "audio_effects":{
            "engine":"open_source",
            "microphone":{
              "noise_removal_enabled":true,
              "model_id":"rnnoise",
              "preserve_this":{"nested":42}
            },
            "speaker":{"room_echo_removal_enabled":true}
          },
          "pipeline":{"running":false,"starting":false,"last_error":""},
          "speakers":{
            "present":true,
            "target_sink_error":"",
            "routing_active":false,
            "route_mode":"off",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  const auto s = studiocast::gui::DaemonStatusSnapshot::FromJson(json);
  return Expect(s.parsed, "nested effects plan payload should parse") &&
         Expect(s.camera.disabledReasons.size() == 1,
                "nested pipeline effects_plan should feed disabled reasons") &&
         Expect(s.camera.disabledReasons.front().contains(
                    QStringLiteral("virtual_key_light")),
                "nested disabled reason should include effect id") &&
         Expect(s.videoEffectPlan.orderedEffectIds.size() == 2,
                "nested plan ordered ids should parse") &&
         Expect(s.videoEffectPlan.orderedEffectIds.back() ==
                    QStringLiteral("virtual_background.blur"),
                "nested plan ordered contract id should parse") &&
         Expect(s.videoEffectPlan.vignetteAttachToEffectId ==
                    QStringLiteral("virtual_background.blur"),
                "nested plan vignette attachment should parse") &&
         Expect(s.videoEffectPlan.disabledEffects.size() == 1,
                "nested plan disabled entries should parse") &&
         Expect(s.videoEffectPlan.disabledEffects.front().effectId ==
                    QStringLiteral("virtual_key_light"),
                "nested plan disabled effect id should parse") &&
         Expect(s.videoEffectPlan.disabledEffects.front().reason ==
                    QStringLiteral("disabled by rule"),
                "nested plan disabled reason should parse") &&
         Expect(s.rawVideoEffectsJson.contains(
                    QStringLiteral("hidden_future_field")),
                "raw video_effects json should preserve hidden fields") &&
         Expect(s.rawAudioEffectsJson.contains(QStringLiteral("preserve_this")),
                "raw audio_effects json should preserve hidden fields");
}

bool TestVideoEffectReadinessMissingModelWhileIdle() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "video":{
          "enabled":true,
          "virtual_device_present":true,
          "virtual_device_available":true,
          "consumer_present":false,
          "consumer_count":0,
          "video_effects":{
            "engine":"open_cuda",
            "virtual_background":{"mode":"blur","model_id":"configured-missing"}
          },
          "effect_readiness":{
            "virtual_background.blur":{
              "state":"missing_model",
              "summary":"Virtual background model is missing.",
              "detail":"Configured model configured-missing is not installed.",
              "backend":"open_cuda",
              "requested_model_id":"configured-missing",
              "resolved_model_id":"",
              "reason":"missing_model"
            },
            "mirror":{
              "state":"ready",
              "summary":"Mirror is ready.",
              "backend":"builtin"
            }
          },
          "pipeline":{
            "running":false,
            "starting":false,
            "state":"idle_no_consumer",
            "idle_reason":"No active virtual camera consumer."
          }
        },
        "audio":{
          "mic_present":true,
          "source_error":"",
          "pipeline":{"running":false,"starting":false,"last_error":""},
          "speakers":{
            "present":true,
            "target_sink_error":"",
            "routing_active":false,
            "route_mode":"off",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  const auto s = studiocast::gui::DaemonStatusSnapshot::FromJson(json);
  const auto vb =
      s.videoEffectReadiness.value(QStringLiteral("virtual_background.blur"));
  const auto mirror = s.videoEffectReadiness.value(QStringLiteral("mirror"));
  return Expect(s.parsed, "effect readiness payload should parse") &&
         Expect(s.camera.state == studiocast::gui::ReadinessState::MissingModel,
                "missing effect model should drive camera missing-model state "
                "while idle") &&
         Expect(s.videoEffectReadiness.contains(
                    QStringLiteral("virtual_background.blur")),
                "effect readiness should be keyed by contract effect id") &&
         Expect(vb.effectId == QStringLiteral("virtual_background.blur"),
                "readiness effect id should parse from object key") &&
         Expect(vb.state == studiocast::gui::ReadinessState::MissingModel,
                "effect readiness state should parse missing_model") &&
         Expect(vb.backend == QStringLiteral("open_cuda"),
                "effect readiness backend should parse") &&
         Expect(vb.requestedModelId == QStringLiteral("configured-missing"),
                "effect readiness requested model should parse") &&
         Expect(vb.reason == QStringLiteral("missing_model"),
                "effect readiness reason should parse") &&
         Expect(mirror.state == studiocast::gui::ReadinessState::Ready,
                "other effect readiness entries should parse");
}

bool TestAudioEndpointActionReadinessFields() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "audio":{
          "enabled":true,
          "create_virtual_mic":true,
          "create_virtual_speakers":true,
          "source":"studio_usb_mic",
          "source_resolved":"alsa_input.usb_mic",
          "source_error":"",
          "mic_present":true,
          "mic_consumer_present":false,
          "mic_consumer_count":0,
          "microphone":{
            "action":"choose_open_audio_model",
            "readiness":{
              "state":"missing_model",
              "summary":"Microphone model is missing.",
              "detail":"Configured Open Audio model mic-missing is not installed."
            }
          },
          "audio_effects":{
            "engine":"open_source",
            "microphone":{"model_id":"mic-missing"},
            "speaker":{"model_id":"speaker-good"}
          },
          "pipeline":{
            "running":false,
            "starting":false,
            "state":"idle_no_consumer",
            "backend_active":"open_audio",
            "last_error":""
          },
          "speakers":{
            "action":"stop_routing",
            "readiness":{
              "state":"processing",
              "summary":"Processed speaker routing is active.",
              "detail":"Speaker cleanup is processing audio."
            },
            "enabled":true,
            "present":true,
            "target_sink":"alsa_output.pci",
            "target_sink_resolved":"alsa_output.pci",
            "target_sink_active":"alsa_output.pci",
            "target_sink_error":"",
            "consumer_present":true,
            "consumer_count":2,
            "routing_active":true,
            "route_mode":"pipeline",
            "pipeline_running":true,
            "pipeline_starting":false,
            "pipeline_state":"running",
            "backend_active":"open_audio",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  const auto s = studiocast::gui::DaemonStatusSnapshot::FromJson(json);
  return Expect(s.parsed, "audio endpoint status payload should parse") &&
         Expect(
             s.microphone.state ==
                 studiocast::gui::ReadinessState::MissingModel,
             "explicit microphone readiness should drive microphone state") &&
         Expect(s.microphoneEndpoint.action ==
                    QStringLiteral("choose_open_audio_model"),
                "microphone action should parse from daemon status") &&
         Expect(s.microphoneEndpoint.readiness.state ==
                    studiocast::gui::ReadinessState::MissingModel,
                "microphone endpoint readiness state should parse") &&
         Expect(s.microphoneEndpoint.configuredDevice ==
                    QStringLiteral("studio_usb_mic"),
                "microphone configured source should parse") &&
         Expect(s.microphoneEndpoint.resolvedDevice ==
                    QStringLiteral("alsa_input.usb_mic"),
                "microphone resolved source should parse") &&
         Expect(s.microphoneEndpoint.activeBackend ==
                    QStringLiteral("open_audio"),
                "microphone active backend should parse") &&
         Expect(s.speakersEndpoint.action == QStringLiteral("stop_routing"),
                "speaker action should parse from daemon status") &&
         Expect(s.speakersEndpoint.readiness.state ==
                    studiocast::gui::ReadinessState::Processing,
                "speaker endpoint readiness state should parse") &&
         Expect(s.speakersEndpoint.configuredDevice ==
                    QStringLiteral("alsa_output.pci"),
                "speaker configured target should parse") &&
         Expect(s.speakersEndpoint.resolvedDevice ==
                    QStringLiteral("alsa_output.pci"),
                "speaker resolved target should parse") &&
         Expect(s.speakersEndpoint.activeDevice ==
                    QStringLiteral("alsa_output.pci"),
                "speaker active target should parse") &&
         Expect(s.speakersEndpoint.routeMode == QStringLiteral("pipeline"),
                "speaker route mode should parse") &&
         Expect(s.speakersEndpoint.consumerCount == 2,
                "speaker consumer count should parse");
}

bool TestEngineModelDetailsAndConfiguredSelections() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "engines":{
          "maxine":{
            "supported":false,
            "ok":false,
            "summary":"Maxine VideoFX SDK not found.",
            "blocked_reason":"sdk_missing",
            "blocked_details":["Install Maxine VideoFX and AudioFX."],
            "hints":["Run studiocast-maxine install-hints"],
            "components":{
              "vfx":{
                "feature_status":{
                  "BackgroundBlur":{"installed":false,"details":"Missing feature file."}
                }
              }
            }
          },
          "open_cuda":{
            "ok":true,
            "onnxruntime_version":"1.20.0",
            "onnxruntime_providers":["CUDAExecutionProvider","CPUExecutionProvider"],
            "onnxruntime_cuda_provider_present":true,
            "onnxruntime_tensorrt_provider_present":false,
            "onnxruntime_cpu_provider_present":true,
            "onnxruntime_cuda_ep_v2_build":true,
            "onnxruntime_library_path":"/opt/ort/lib/libonnxruntime.so",
            "cuda_driver_api_available":true,
            "cuda_context_available":true,
            "cuda_device_count":1,
            "cuda_driver_version":12040,
            "installed_models":["matting-good"],
            "default_model_id":"matting-good",
            "models":[
              {"id":"matting-good","display_name":"Good Matting","task":"matting","width":256,"height":256}
            ],
            "missing_models":{"configured-missing":"model.json is missing"},
            "available_effects":["video_noise_removal"],
            "blocked_effects":{"auto_frame":"missing_model_packs"},
            "install_hints":["Open Video hint"]
          },
          "open_audio":{
            "ok":true,
            "onnxruntime_version":"1.20.0",
            "onnxruntime_providers":["CPUExecutionProvider"],
            "onnxruntime_cuda_provider_present":false,
            "onnxruntime_tensorrt_provider_present":false,
            "onnxruntime_cpu_provider_present":true,
            "onnxruntime_cuda_ep_v2_build":false,
            "onnxruntime_library_path":"/opt/ort/lib/libonnxruntime.so",
            "acceleration_likely":"cpu_fallback",
            "installed_models":["fast-enhancer"],
            "models":[
              {"id":"fast-enhancer","display_name":"Fast Enhancer","effects":["noise_removal"],"sample_rate":48000,"channels":1}
            ],
            "missing_models":{"gone":"No such model pack"},
            "install_hints":["Open Audio hint"]
          }
        },
        "video":{
          "enabled":true,
          "virtual_device_present":true,
          "virtual_device_available":true,
          "consumer_count":1,
          "video_effects":{
            "engine":"open_cuda",
            "virtual_background":{"model_id":"matting-good"},
            "auto_frame":{"model_id":"configured-missing"},
            "eye_contact":{"model_id":"not-reported"}
          },
          "pipeline":{
            "running":true,
            "starting":false,
            "effects_backends":"virtual_background.blur:open_cuda,mirror:passthrough"
          }
        },
        "audio":{
          "mic_present":true,
          "source_error":"",
          "audio_effects":{
            "engine":"open_source",
            "microphone":{"model_id":"fast-enhancer","model_path":""},
            "speaker":{"model_id":"gone","model_path":"/tmp/explicit.onnx"}
          },
          "pipeline":{"running":true,"starting":false,"last_error":"","backend_active":"open_audio"},
          "speakers":{
            "present":true,
            "target_sink_error":"",
            "routing_active":true,
            "route_mode":"pipeline",
            "backend_active":"open_audio",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  const auto s = studiocast::gui::DaemonStatusSnapshot::FromJson(json);
  return Expect(s.parsed, "engine model details payload should parse") &&
         Expect(s.camera.state == studiocast::gui::ReadinessState::Processing,
                "running camera should report processing") &&
         Expect(s.microphone.state ==
                    studiocast::gui::ReadinessState::Processing,
                "running microphone should report processing") &&
         Expect(s.speakers.state == studiocast::gui::ReadinessState::Processing,
                "active speaker routing should report processing") &&
         Expect(
             s.videoEffectsActiveBackends.contains(QStringLiteral("open_cuda")),
             "video active backends should parse") &&
         Expect(s.microphoneActiveBackend == QStringLiteral("open_audio"),
                "microphone active backend should parse") &&
         Expect(s.speakersActiveBackend == QStringLiteral("open_audio"),
                "speaker active backend should parse") &&
         Expect(s.maxine.installHints.contains(
                    QStringLiteral("Run studiocast-maxine install-hints")),
                "maxine hints should be preserved") &&
         Expect(s.maxine.missingModelCount == 1,
                "maxine missing feature state should parse") &&
         Expect(!s.openCuda.rawJson.isEmpty(),
                "raw open_cuda diagnostics should be preserved") &&
         Expect(s.openCuda.rawJson.contains(
                    QStringLiteral("onnxruntime_cuda_provider_present")),
                "raw open_cuda diagnostics should preserve runtime fields") &&
         Expect(
             s.openCuda.rawJson.contains(QStringLiteral("cuda_driver_version")),
             "raw open_cuda diagnostics should preserve cuda fields") &&
         Expect(s.openCuda.installedModels.size() == 1 &&
                    s.openCuda.installedModels.front().displayName ==
                        QStringLiteral("Good Matting"),
                "open video model display names should parse") &&
         Expect(s.openCuda.configuredModels.size() == 3,
                "configured open video model IDs should parse") &&
         Expect(s.openCuda.configuredMissingModelCount == 2,
                "configured missing open video IDs should be counted") &&
         Expect(s.openAudio.installedModels.size() == 1 &&
                    s.openAudio.installedModels.front().displayName ==
                        QStringLiteral("Fast Enhancer"),
                "open audio model display names should parse") &&
         Expect(s.openAudio.rawJson.contains(
                    QStringLiteral("acceleration_likely")),
                "raw open_audio diagnostics should preserve runtime fields") &&
         Expect(s.openAudio.rawJson.contains(
                    QStringLiteral("onnxruntime_cpu_provider_present")),
                "raw open_audio diagnostics should preserve provider fields") &&
         Expect(s.openAudio.configuredModels.size() == 2,
                "configured open audio model IDs should parse") &&
         Expect(s.openAudio.configuredMissingModelCount == 1,
                "configured missing open audio IDs should be counted") &&
         Expect(s.openAudio.configuredModels.back().modelPath ==
                    QStringLiteral("/tmp/explicit.onnx"),
                "explicit model paths should be preserved in details");
}

bool TestEnginesModelsPageShowsOpenCudaSetupFix() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "engines":{
          "open_cuda":{
            "ok":false,
            "installed_models":["modnet-webnn-256-fp32"],
            "models":[
              {"id":"modnet-webnn-256-fp32","display_name":"MODNet","task":"matting"}
            ],
            "missing_models":{},
            "blocked_effects":{
              "virtual_background.blur":"disabled_in_build",
              "auto_frame":"disabled_in_build"
            },
            "install_hints":[
              "Open CUDA backend is disabled in this build.",
              "Rebuild with -DSTUDIOCAST_ENABLE_OPEN_CUDA=ON."
            ]
          }
        },
        "video":{
          "enabled":false,
          "virtual_device_present":true,
          "virtual_device_available":true,
          "consumer_count":0,
          "video_effects":{"engine":"open_cuda"},
          "pipeline":{"running":false,"starting":false}
        },
        "audio":{
          "mic_present":true,
          "source_error":"",
          "pipeline":{"running":false,"starting":false,"last_error":""},
          "speakers":{
            "present":true,
            "target_sink_error":"",
            "routing_active":false,
            "route_mode":"off",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  studiocast::gui::EnginesModelsPage page;
  page.UpdateStatus(studiocast::gui::DaemonStatusSnapshot::FromJson(json));

  auto *state =
      page.findChild<QLabel *>(QStringLiteral("open_cuda_engine_state"));
  auto *summary =
      page.findChild<QLabel *>(QStringLiteral("open_cuda_engine_summary"));
  auto *disclaimer =
      page.findChild<QLabel *>(QStringLiteral("open_cuda_setup_disclaimer"));
  auto *disclaimerBanner = page.findChild<QFrame *>(
      QStringLiteral("open_cuda_setup_disclaimer_banner"));
  auto *downloadStatus =
      page.findChild<QLabel *>(QStringLiteral("open_cuda_download_status"));
  auto *downloadButton = page.findChild<QPushButton *>(
      QStringLiteral("open_cuda_download_button"));
  auto *repairButton = page.findChild<QPushButton *>(
      QStringLiteral("open_cuda_repair_setup_button"));
  auto *details = page.findChild<QPlainTextEdit *>(
      QStringLiteral("open_cuda_engine_details"));

  return Expect(state != nullptr, "open cuda state label should be findable") &&
         Expect(summary != nullptr,
                "open cuda summary label should be findable") &&
         Expect(disclaimer != nullptr,
                "open cuda setup disclaimer should be findable") &&
         Expect(disclaimerBanner != nullptr,
                "open cuda setup disclaimer banner should be findable") &&
         Expect(downloadStatus != nullptr,
                "open cuda download status should be findable") &&
         Expect(downloadButton != nullptr,
                "open cuda download button should be findable") &&
         Expect(repairButton != nullptr,
                "open cuda repair setup button should be findable") &&
         Expect(details != nullptr, "open cuda details should be findable") &&
         Expect(state->text() == QStringLiteral("Selected setup required"),
                "disabled Open CUDA build should show setup-required state") &&
         Expect(summary->text() ==
                    QStringLiteral("This backend is currently selected."),
                "selected-backend text should stay separate from setup "
                "disclaimer") &&
         Expect(!disclaimer->isHidden() && !disclaimer->text().isEmpty(),
                "setup disclaimer should be shown for setup blockers") &&
         Expect(!disclaimerBanner->isHidden(),
                "setup disclaimer banner should be shown") &&
         Expect(disclaimerBanner->property("scBanner").toString() ==
                    QStringLiteral("warning"),
                "setup disclaimer should use the warning banner style") &&
         Expect(!repairButton->isHidden(),
                "setup repair button should be shown for setup blockers") &&
         Expect(repairButton->isEnabled(),
                "setup repair button should be enabled when no repair is "
                "running") &&
         Expect(
             repairButton->text() == QStringLiteral("Repair Open Video setup"),
             "Open CUDA setup repair button should use the Open Video label") &&
         Expect(repairButton != downloadButton,
                "setup repair action should be separate from model download") &&
         Expect(disclaimer->text().contains(
                    QStringLiteral("disabled in the running StudioCast build")),
                "disclaimer should explain disabled build") &&
         Expect(disclaimer->text().contains(
                    QStringLiteral("-DSTUDIOCAST_ENABLE_OPEN_CUDA=ON")),
                "disclaimer should include rebuild flag") &&
         Expect(!disclaimer->text().contains(
                    QStringLiteral("This backend is currently selected")),
                "setup disclaimer should not include selected-backend text") &&
         Expect(downloadStatus->text().contains(
                    QStringLiteral("setup issue above")),
                "model install status should reference the separate setup "
                "disclaimer") &&
         Expect(
             !downloadStatus->text().contains(
                 QStringLiteral("Ready to install")),
             "disabled build should not claim models are ready to install") &&
         Expect(
             !downloadButton->isEnabled(),
             "model download button should be disabled for setup blockers") &&
         Expect(details->toPlainText().contains(
                    QStringLiteral("cmake -S . -B build")),
                "details should include source-build fix command");
}

bool TestEnginesModelsPageShowsOpenAudioSetupFix() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "engines":{
          "open_audio":{
            "ok":false,
            "installed_models":["fastenhancer_s_vd_v1"],
            "models":[
              {"id":"fastenhancer_s_vd_v1","display_name":"FastEnhancer-S","effects":["noise_removal"],"sample_rate":16000,"channels":1}
            ],
            "missing_models":{},
            "blocked_effects":{
              "noise_removal":"disabled_in_build",
              "room_echo_removal":"disabled_in_build"
            },
            "install_hints":[
              "Open Audio backend is disabled in this build.",
              "Rebuild with -DSTUDIOCAST_ENABLE_OPEN_AUDIO=ON."
            ]
          }
        },
        "video":{
          "enabled":false,
          "virtual_device_present":true,
          "virtual_device_available":true,
          "consumer_count":0,
          "video_effects":{"engine":"auto"},
          "pipeline":{"running":false,"starting":false}
        },
        "audio":{
          "mic_present":true,
          "source_error":"",
          "audio_effects":{"engine":"open_source"},
          "pipeline":{"running":false,"starting":false,"last_error":""},
          "speakers":{
            "present":true,
            "target_sink_error":"",
            "routing_active":false,
            "route_mode":"off",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  studiocast::gui::EnginesModelsPage page;
  page.UpdateStatus(studiocast::gui::DaemonStatusSnapshot::FromJson(json));

  auto *state =
      page.findChild<QLabel *>(QStringLiteral("open_audio_engine_state"));
  auto *summary =
      page.findChild<QLabel *>(QStringLiteral("open_audio_engine_summary"));
  auto *disclaimer =
      page.findChild<QLabel *>(QStringLiteral("open_audio_setup_disclaimer"));
  auto *disclaimerBanner = page.findChild<QFrame *>(
      QStringLiteral("open_audio_setup_disclaimer_banner"));
  auto *downloadStatus =
      page.findChild<QLabel *>(QStringLiteral("open_audio_download_status"));
  auto *downloadButton = page.findChild<QPushButton *>(
      QStringLiteral("open_audio_download_button"));
  auto *repairButton = page.findChild<QPushButton *>(
      QStringLiteral("open_audio_repair_setup_button"));

  return Expect(state != nullptr,
                "open audio state label should be findable") &&
         Expect(summary != nullptr,
                "open audio summary label should be findable") &&
         Expect(disclaimer != nullptr,
                "open audio setup disclaimer should be findable") &&
         Expect(disclaimerBanner != nullptr,
                "open audio setup disclaimer banner should be findable") &&
         Expect(downloadStatus != nullptr,
                "open audio download status should be findable") &&
         Expect(downloadButton != nullptr,
                "open audio download button should be findable") &&
         Expect(repairButton != nullptr,
                "open audio repair setup button should be findable") &&
         Expect(state->text() == QStringLiteral("Selected setup required"),
                "disabled Open Audio build should show setup-required state") &&
         Expect(summary->text() ==
                    QStringLiteral("This backend is currently selected."),
                "selected Open Audio text should stay separate from setup "
                "disclaimer") &&
         Expect(!disclaimer->isHidden() && !disclaimer->text().isEmpty(),
                "Open Audio setup disclaimer should be shown") &&
         Expect(!disclaimerBanner->isHidden(),
                "Open Audio setup disclaimer banner should be shown") &&
         Expect(
             disclaimerBanner->property("scBanner").toString() ==
                 QStringLiteral("warning"),
             "Open Audio setup disclaimer should use warning banner style") &&
         Expect(!repairButton->isHidden(),
                "Open Audio setup repair button should be shown for setup "
                "blockers") &&
         Expect(repairButton->isEnabled(),
                "Open Audio setup repair button should be enabled when no "
                "repair is running") &&
         Expect(repairButton->text() ==
                    QStringLiteral("Repair Open Audio setup"),
                "Open Audio setup repair button should use the Open Audio "
                "label") &&
         Expect(repairButton != downloadButton,
                "Open Audio setup repair action should be separate from model "
                "download") &&
         Expect(disclaimer->text().contains(
                    QStringLiteral("Open Audio is disabled in the running "
                                   "StudioCast build")),
                "Open Audio disclaimer should explain disabled build") &&
         Expect(disclaimer->text().contains(
                    QStringLiteral("-DSTUDIOCAST_ENABLE_OPEN_AUDIO=ON")),
                "Open Audio disclaimer should include rebuild flag") &&
         Expect(!disclaimer->text().contains(
                    QStringLiteral("This backend is currently selected")),
                "Open Audio disclaimer should not include selected-backend "
                "text") &&
         Expect(downloadStatus->text().contains(
                    QStringLiteral("Open Audio setup issue above")),
                "Open Audio model status should reference setup disclaimer") &&
         Expect(!downloadStatus->text().contains(
                    QStringLiteral("Ready to install")),
                "disabled Open Audio build should not claim models are ready "
                "to install") &&
         Expect(!downloadButton->isEnabled(),
                "Open Audio model download button should be disabled for setup "
                "blockers");
}

bool TestEnginesModelsPageKeepsModelInstallsSeparateFromSetupRepair() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "engines":{
          "open_cuda":{
            "ok":false,
            "installed_models":[],
            "models":[],
            "missing_models":{"modnet-webnn-256-fp32":"model.onnx is missing"},
            "blocked_effects":{
              "virtual_background.blur":"missing_model_packs"
            },
            "install_hints":["No usable Open Video matting model packs were found."]
          },
          "open_audio":{
            "ok":false,
            "installed_models":[],
            "models":[],
            "missing_models":{"fastenhancer_s_vd_v1":"model.onnx is missing"},
            "blocked_effects":{
              "noise_removal":"missing_model_packs"
            },
            "install_hints":["No usable Open Audio model packs were found."]
          }
        },
        "video":{
          "enabled":false,
          "virtual_device_present":true,
          "virtual_device_available":true,
          "consumer_count":0,
          "video_effects":{"engine":"open_cuda"},
          "pipeline":{"running":false,"starting":false}
        },
        "audio":{
          "mic_present":true,
          "source_error":"",
          "audio_effects":{"engine":"open_source"},
          "pipeline":{"running":false,"starting":false,"last_error":""},
          "speakers":{
            "present":true,
            "target_sink_error":"",
            "routing_active":false,
            "route_mode":"off",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  studiocast::gui::EnginesModelsPage page;
  page.UpdateStatus(studiocast::gui::DaemonStatusSnapshot::FromJson(json));

  auto *openVideoRepair = page.findChild<QPushButton *>(
      QStringLiteral("open_cuda_repair_setup_button"));
  auto *openAudioRepair = page.findChild<QPushButton *>(
      QStringLiteral("open_audio_repair_setup_button"));
  auto *openVideoDisclaimer = page.findChild<QFrame *>(
      QStringLiteral("open_cuda_setup_disclaimer_banner"));
  auto *openAudioDisclaimer = page.findChild<QFrame *>(
      QStringLiteral("open_audio_setup_disclaimer_banner"));
  auto *openVideoDownload = page.findChild<QPushButton *>(
      QStringLiteral("open_cuda_download_button"));
  auto *openAudioDownload = page.findChild<QPushButton *>(
      QStringLiteral("open_audio_download_button"));
  auto *openVideoDownloadStatus =
      page.findChild<QLabel *>(QStringLiteral("open_cuda_download_status"));
  auto *openAudioDownloadStatus =
      page.findChild<QLabel *>(QStringLiteral("open_audio_download_status"));

  return Expect(openVideoRepair != nullptr,
                "open video repair setup button should be findable") &&
         Expect(openAudioRepair != nullptr,
                "open audio repair setup button should be findable") &&
         Expect(openVideoDisclaimer != nullptr,
                "open video setup disclaimer banner should be findable") &&
         Expect(openAudioDisclaimer != nullptr,
                "open audio setup disclaimer banner should be findable") &&
         Expect(openVideoDownload != nullptr,
                "open video model download button should be findable") &&
         Expect(openAudioDownload != nullptr,
                "open audio model download button should be findable") &&
         Expect(openVideoDownloadStatus != nullptr,
                "open video model download status should be findable") &&
         Expect(openAudioDownloadStatus != nullptr,
                "open audio model download status should be findable") &&
         Expect(
             openVideoRepair->isHidden(),
             "missing Open Video models alone should not show setup repair") &&
         Expect(
             openAudioRepair->isHidden(),
             "missing Open Audio models alone should not show setup repair") &&
         Expect(openVideoDisclaimer->isHidden(),
                "missing Open Video models alone should not show setup "
                "disclaimer") &&
         Expect(openAudioDisclaimer->isHidden(),
                "missing Open Audio models alone should not show setup "
                "disclaimer") &&
         Expect(
             openVideoDownload->isEnabled(),
             "missing Open Video models should keep model download enabled") &&
         Expect(
             openAudioDownload->isEnabled(),
             "missing Open Audio models should keep model download enabled") &&
         Expect(openVideoDownloadStatus->text().contains(
                    QStringLiteral("Ready to install")),
                "missing Open Video models should use model install path") &&
         Expect(openAudioDownloadStatus->text().contains(
                    QStringLiteral("Ready to install")),
                "missing Open Audio models should use model install path");
}

bool TestCameraPageShowsOpenCudaSetupFix() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "engines":{
          "open_cuda":{
            "ok":false,
            "installed_models":["modnet-webnn-256-fp32"],
            "models":[
              {"id":"modnet-webnn-256-fp32","display_name":"MODNet","task":"matting"}
            ],
            "missing_models":{},
            "blocked_effects":{
              "virtual_background.blur":"disabled_in_build",
              "auto_frame":"disabled_in_build"
            },
            "install_hints":[
              "Open CUDA backend is disabled in this build.",
              "Rebuild with -DSTUDIOCAST_ENABLE_OPEN_CUDA=ON."
            ]
          }
        },
        "video":{
          "enabled":false,
          "virtual_device_present":true,
          "virtual_device_available":true,
          "consumer_count":0,
          "video_effects":{
            "engine":"open_cuda",
            "virtual_background":{"mode":"blur","model_id":"modnet-webnn-256-fp32"}
          },
          "pipeline":{"running":false,"starting":false}
        },
        "audio":{
          "mic_present":true,
          "source_error":"",
          "pipeline":{"running":false,"starting":false,"last_error":""},
          "speakers":{
            "present":true,
            "target_sink_error":"",
            "routing_active":false,
            "route_mode":"off",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  studiocast::gui::VideoPage page;
  page.UpdateStatus(studiocast::gui::DaemonStatusSnapshot::FromJson(json));

  auto *banner =
      page.findChild<QLabel *>(QStringLiteral("cameraEngineWarningBanner"));
  return Expect(banner != nullptr,
                "camera engine warning banner should be findable") &&
         Expect(!banner->text().contains(
                    QStringLiteral("No usable model packs were found")),
                "camera page should not blame installed models for setup "
                "blockers") &&
         Expect(banner->text().contains(
                    QStringLiteral("disabled in the running StudioCast build")),
                "camera page should explain disabled Open CUDA build") &&
         Expect(banner->text().contains(
                    QStringLiteral("-DSTUDIOCAST_ENABLE_OPEN_CUDA=ON")),
                "camera page should include rebuild flag") &&
         Expect(
             banner->text().contains(QStringLiteral("Model packs were found")),
             "camera page should say models are present but unusable until "
             "setup is fixed");
}

bool TestOpenAudioRuntimeStatusFields() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "audio":{
          "enabled":true,
          "mic_present":true,
          "source_error":"",
          "audio_effects":{"engine":"open_source"},
          "pipeline":{
            "running":true,
            "starting":false,
            "state":"running",
            "backend_active":"open_source",
            "last_error":"",
            "open_audio_runtime":{
              "active":true,
              "active_provider":"cpu",
              "using_cpu_fallback":true,
              "disabled":false,
              "selected_model_id":"fast-enhancer",
              "selected_model_path":"/models/open_audio/fast/model.onnx",
              "last_runtime_warning":"Open Audio: switched to CPU fallback after a CUDA runtime failure."
            }
          },
          "speakers":{
            "enabled":true,
            "present":true,
            "target_sink_error":"",
            "routing_active":false,
            "route_mode":"pipeline",
            "backend_active":"open_source",
            "last_error":"",
            "pipeline_last_error":"",
            "open_audio_runtime":{
              "active":false,
              "active_provider":"disabled",
              "using_cpu_fallback":true,
              "disabled":true,
              "selected_model_id":"speaker-enhancer",
              "selected_model_path":"/models/open_audio/speaker/model.onnx",
              "last_runtime_warning":"Open Audio: disabled after repeated runtime failures."
            }
          }
        }
      })");

  const auto s = studiocast::gui::DaemonStatusSnapshot::FromJson(json);
  return Expect(s.parsed, "open audio runtime payload should parse") &&
         Expect(s.microphoneOpenAudioRuntime.present,
                "microphone runtime status should be present") &&
         Expect(s.microphoneOpenAudioRuntime.active,
                "microphone runtime active flag should parse") &&
         Expect(s.microphoneOpenAudioRuntime.activeProvider ==
                    QStringLiteral("cpu"),
                "microphone runtime provider should parse") &&
         Expect(s.microphoneOpenAudioRuntime.usingCpuFallback,
                "microphone CPU fallback should parse") &&
         Expect(!s.microphoneOpenAudioRuntime.disabled,
                "microphone disabled flag should parse") &&
         Expect(s.microphoneOpenAudioRuntime.selectedModelId ==
                    QStringLiteral("fast-enhancer"),
                "microphone runtime model id should parse") &&
         Expect(s.microphoneOpenAudioRuntime.selectedModelPath ==
                    QStringLiteral("/models/open_audio/fast/model.onnx"),
                "microphone runtime model path should parse") &&
         Expect(s.microphoneOpenAudioRuntime.lastRuntimeWarning.contains(
                    QStringLiteral("CPU fallback")),
                "microphone runtime warning should parse") &&
         Expect(s.speakersOpenAudioRuntime.present,
                "speaker runtime status should be present") &&
         Expect(!s.speakersOpenAudioRuntime.active,
                "speaker runtime active flag should parse") &&
         Expect(s.speakersOpenAudioRuntime.disabled,
                "speaker runtime disabled flag should parse") &&
         Expect(s.speakersOpenAudioRuntime.activeProvider ==
                    QStringLiteral("disabled"),
                "speaker runtime provider should parse") &&
         Expect(s.speakersOpenAudioRuntime.usingCpuFallback,
                "speaker CPU fallback should parse") &&
         Expect(s.speakersOpenAudioRuntime.selectedModelId ==
                    QStringLiteral("speaker-enhancer"),
                "speaker runtime model id should parse") &&
         Expect(s.speakersOpenAudioRuntime.selectedModelPath ==
                    QStringLiteral("/models/open_audio/speaker/model.onnx"),
                "speaker runtime model path should parse") &&
         Expect(s.speakersOpenAudioRuntime.lastRuntimeWarning.contains(
                    QStringLiteral("repeated runtime failures")),
                "speaker runtime warning should parse");
}

bool TestMissingVirtualDevices() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "video":{
          "enabled":true,
          "virtual_device_present":false,
          "virtual_device_available":false,
          "virtual_device_error":"v4l2loopback device is missing.",
          "consumer_count":0,
          "pipeline":{"running":false,"starting":false,"state":"device_unavailable"}
        },
        "audio":{
          "enabled":true,
          "mic_present":false,
          "source_error":"",
          "pipeline":{"running":false,"starting":false,"state":"disabled","last_error":""},
          "speakers":{
            "enabled":true,
            "present":false,
            "target_sink_error":"",
            "routing_active":false,
            "route_mode":"off",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  const auto s = studiocast::gui::DaemonStatusSnapshot::FromJson(json);
  return Expect(s.parsed, "missing virtual device payload should parse") &&
         Expect(s.camera.state ==
                    studiocast::gui::ReadinessState::MissingVirtualDevice,
                "camera should report missing virtual camera") &&
         Expect(s.microphone.state ==
                    studiocast::gui::ReadinessState::MissingVirtualDevice,
                "microphone should report missing virtual microphone") &&
         Expect(s.speakers.state ==
                    studiocast::gui::ReadinessState::MissingVirtualDevice,
                "speakers should report missing virtual speakers");
}

bool TestMissingPhysicalDevices() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "video":{
          "enabled":true,
          "virtual_device_present":true,
          "virtual_device_available":true,
          "consumer_count":1,
          "last_error":"Pipeline start failed: No readable camera device found (no non-loopback /dev/video* with read access).",
          "pipeline":{"running":false,"starting":false,"state":"backing_off"}
        },
        "audio":{
          "enabled":true,
          "mic_present":true,
          "source_error":"Select a physical microphone/input source before enabling audio.",
          "pipeline":{"running":false,"starting":false,"state":"idle_no_consumer","last_error":""},
          "speakers":{
            "enabled":true,
            "present":true,
            "target_sink_error":"Failed to choose a physical speaker target sink. Choose a physical output sink.",
            "routing_active":false,
            "route_mode":"pipeline",
            "pipeline_running":false,
            "pipeline_starting":false,
            "pipeline_state":"idle_no_consumer",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  const auto s = studiocast::gui::DaemonStatusSnapshot::FromJson(json);
  return Expect(s.parsed, "missing physical device payload should parse") &&
         Expect(s.camera.state ==
                    studiocast::gui::ReadinessState::NoPhysicalDevice,
                "camera should report physical input problem when supported") &&
         Expect(s.microphone.state ==
                    studiocast::gui::ReadinessState::NoPhysicalDevice,
                "microphone should report physical source problem") &&
         Expect(s.speakers.state ==
                    studiocast::gui::ReadinessState::NoPhysicalDevice,
                "speakers should report physical target problem");
}

bool TestIdleNoConsumerStates() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "video":{
          "enabled":true,
          "virtual_device_present":true,
          "virtual_device_available":true,
          "consumer_present":false,
          "consumer_count":0,
          "pipeline":{
            "running":false,
            "starting":false,
            "state":"idle_no_consumer",
            "idle_reason":"no_consumer"
          }
        },
        "audio":{
          "enabled":true,
          "mic_present":true,
          "source_error":"",
          "mic_consumer_present":false,
          "mic_consumer_count":0,
          "mic_consumer_error":"",
          "pipeline":{
            "running":false,
            "starting":false,
            "state":"idle_no_consumer",
            "idle_reason":"No active virtual microphone consumer.",
            "last_error":""
          },
          "speakers":{
            "enabled":true,
            "present":true,
            "target_sink_error":"",
            "consumer_present":false,
            "consumer_count":0,
            "consumer_error":"",
            "routing_active":false,
            "route_mode":"pipeline",
            "pipeline_running":false,
            "pipeline_starting":false,
            "pipeline_active_needed":false,
            "pipeline_state":"idle_no_consumer",
            "pipeline_idle_reason":"No active virtual speakers consumer.",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  const auto s = studiocast::gui::DaemonStatusSnapshot::FromJson(json);
  return Expect(s.parsed, "idle no-consumer payload should parse") &&
         Expect(s.camera.state == studiocast::gui::ReadinessState::Idle,
                "camera should report idle no-consumer") &&
         Expect(s.microphone.state == studiocast::gui::ReadinessState::Idle,
                "microphone should report idle no-consumer") &&
         Expect(s.speakers.state == studiocast::gui::ReadinessState::Idle,
                "speakers should report idle no-consumer");
}

bool TestConsumerDetectionErrorsAreNotIdle() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "video":{
          "enabled":true,
          "virtual_device_present":true,
          "virtual_device_available":true,
          "consumer_error":"Failed to scan consumers.",
          "consumer_count":0,
          "pipeline":{"running":false,"starting":false,"state":"consumer_detection_error"}
        },
        "audio":{
          "enabled":true,
          "mic_present":true,
          "source_error":"",
          "mic_consumer_error":"PulseAudio consumer detection unavailable.",
          "pipeline":{"running":false,"starting":false,"state":"idle_no_consumer","last_error":""},
          "speakers":{
            "enabled":true,
            "present":true,
            "target_sink_error":"",
            "consumer_error":"PulseAudio consumer detection unavailable.",
            "routing_active":false,
            "route_mode":"pipeline",
            "pipeline_running":false,
            "pipeline_starting":false,
            "pipeline_state":"idle_no_consumer",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  const auto s = studiocast::gui::DaemonStatusSnapshot::FromJson(json);
  return Expect(s.parsed, "consumer detection error payload should parse") &&
         Expect(s.camera.state ==
                    studiocast::gui::ReadinessState::RecoverableError,
                "camera consumer detection error should not report idle") &&
         Expect(s.microphone.state ==
                    studiocast::gui::ReadinessState::RecoverableError,
                "microphone consumer detection error should not report idle") &&
         Expect(s.speakers.state ==
                    studiocast::gui::ReadinessState::RecoverableError,
                "speakers consumer detection error should not report idle");
}

bool TestInvalidJsonPreservesRawPayload() {
  const QString json = QStringLiteral("{not-json");
  const auto s = studiocast::gui::DaemonStatusSnapshot::FromJson(json);
  return Expect(s.reachable,
                "parse errors still came from a reachable daemon") &&
         Expect(!s.parsed, "invalid json should not parse") &&
         Expect(s.rawJson == json, "invalid raw json should be preserved") &&
         Expect(!s.parseError.isEmpty(), "parse error should be reported") &&
         Expect(s.RawDiagnosticsText() == json,
                "raw diagnostics should preserve invalid raw payloads");
}

bool TestRawDiagnosticsFallbacks() {
  const auto unreachable = studiocast::gui::DaemonStatusSnapshot::Unreachable(
      QStringLiteral("connect failed"));
  studiocast::gui::DaemonStatusSnapshot empty;
  return Expect(unreachable.RawDiagnosticsText() ==
                    QStringLiteral("Daemon unavailable: connect failed"),
                "unreachable raw diagnostics should include transport error") &&
         Expect(empty.RawDiagnosticsText() ==
                    QStringLiteral("Daemon status has not been read."),
                "empty raw diagnostics should explain status is unread");
}

bool TestStatusPollerRefreshesDiagnosticsOutOfBand() {
  ScopedRuntimeDir runtime("studiocast-gui-status-poller");
  if (!runtime.ok()) {
    std::cerr << runtime.error() << "\n";
    return false;
  }

  std::string err;
  const auto socketPath = studiocast::ipc::DaemonSocketPath(&err);
  if (socketPath.empty()) {
    std::cerr << "DaemonSocketPath failed: " << err << "\n";
    return false;
  }

  std::atomic<int> statusCalls{0};
  std::atomic<int> diagnosticsCalls{0};
  studiocast::ipc::DaemonServer server;
  if (!server.Start(
          socketPath,
          [&statusCalls, &diagnosticsCalls](const std::string &line) {
            if (line == "GET_STATUS") {
              ++statusCalls;
              return std::string(
                  "OK {\"service_running\":true,"
                  "\"engines\":{\"open_cuda\":{\"ok\":true,"
                  "\"summary\":\"cached diagnostics\"}},"
                  "\"video\":{\"enabled\":false,"
                  "\"virtual_device_present\":true,"
                  "\"virtual_device_available\":true,"
                  "\"consumer_count\":0,"
                  "\"pipeline\":{\"running\":false,\"starting\":false}},"
                  "\"audio\":{\"enabled\":false,\"mic_present\":true,"
                  "\"source_error\":\"\","
                  "\"pipeline\":{\"running\":false,\"starting\":false},"
                  "\"speakers\":{\"present\":true,"
                  "\"target_sink_error\":\"\","
                  "\"routing_active\":false,\"route_mode\":\"off\"}}}");
            }
            if (line == "REFRESH_DIAGNOSTICS") {
              ++diagnosticsCalls;
              std::this_thread::sleep_for(50ms);
              return std::string(
                  "OK {\"engines\":{\"open_cuda\":{\"ok\":true}}}");
            }
            return std::string("ERR {\"error\":\"unexpected\"}");
          },
          &err)) {
    std::cerr << "server.Start failed: " << err << "\n";
    return false;
  }

  studiocast::gui::StatusPoller poller;
  int changedCount = 0;
  bool refreshFinished = false;
  bool refreshOk = false;

  QEventLoop loop;
  QObject::connect(
      &poller, &studiocast::gui::StatusPoller::StatusChanged, &loop,
      [&](const studiocast::gui::DaemonStatusSnapshot &) { ++changedCount; });
  QObject::connect(&poller,
                   &studiocast::gui::StatusPoller::DiagnosticsRefreshFinished,
                   &loop, [&](bool ok, const QString &) {
                     refreshFinished = true;
                     refreshOk = ok;
                     loop.quit();
                   });

  poller.RefreshDiagnosticsNow();
  poller.RefreshDiagnosticsNow();

  QTimer::singleShot(3000, &loop, &QEventLoop::quit);
  loop.exec();
  server.Stop();

  return Expect(refreshFinished,
                "diagnostics refresh should finish asynchronously") &&
         Expect(refreshOk, "diagnostics refresh should report success") &&
         Expect(diagnosticsCalls.load() == 1,
                "duplicate diagnostics refreshes should coalesce") &&
         Expect(
             statusCalls.load() == 1,
             "diagnostics refresh should trigger one follow-up status poll") &&
         Expect(changedCount == 1,
                "follow-up status poll should emit one snapshot");
}

bool TestPendingDaemonWriteGuardSkipsRoutineStatusUntilWriteSettles() {
  studiocast::gui::PendingDaemonWriteGuard guard;
  if (!Expect(guard.ShouldApplyRoutineStatus(),
              "fresh guard should allow routine status"))
    return false;

  guard.MarkPending();
  if (!Expect(!guard.ShouldApplyRoutineStatus(),
              "pending local write should suppress routine status resync"))
    return false;

  guard.MarkWriteAccepted();
  if (!Expect(guard.ShouldApplyRoutineStatus(),
              "accepted write should re-enable routine status resync"))
    return false;

  guard.MarkPending();
  if (!Expect(!guard.ShouldApplyRoutineStatus(),
              "second pending write should suppress routine status resync"))
    return false;

  guard.MarkWriteRejected();
  return Expect(guard.ShouldApplyRoutineStatus(),
                "rejected write should allow forced daemon resync");
}

bool TestVideoPageKeepsUserSelectedInputWhenRoutineStatusStillAuto() {
  studiocast::gui::VideoPage page;
  auto *inputCombo =
      page.findChild<QComboBox *>(QStringLiteral("videoInputCombo"));
  if (!Expect(inputCombo != nullptr, "video input combo should be findable"))
    return false;

  const QString selectedDevice = QStringLiteral("/dev/video-test-selection");
  if (inputCombo->findData(selectedDevice) < 0)
    inputCombo->addItem(QStringLiteral("Test Camera"), selectedDevice);

  const int selectedIndex = inputCombo->findData(selectedDevice);
  inputCombo->setCurrentIndex(selectedIndex);
  if (!Expect(inputCombo->currentData().toString() == selectedDevice,
              "test setup should select explicit input camera"))
    return false;

  const QString daemonStillAuto = QStringLiteral(
      R"({
        "service_running":true,
        "video":{
          "enabled":false,
          "virtual_device_present":true,
          "virtual_device_available":true,
          "consumer_count":0,
          "input_device":"",
          "output_device":"",
          "width":1280,
          "height":720,
          "fps":30,
          "pipeline":{"running":false,"starting":false}
        },
        "audio":{
          "enabled":false,
          "mic_present":true,
          "source_error":"",
          "pipeline":{"running":false,"starting":false,"last_error":""},
          "speakers":{
            "present":true,
            "target_sink_error":"",
            "routing_active":false,
            "route_mode":"off",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  page.UpdateStatus(
      studiocast::gui::DaemonStatusSnapshot::FromJson(daemonStillAuto));
  return Expect(inputCombo->currentData().toString() == selectedDevice,
                "routine status must not reset a user-selected input to auto");
}

bool TestVideoPageEnablesAvailableVirtualBackgroundRemoveMode() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "maxine":{
          "supported":false,
          "ok":false,
          "summary":"Maxine unavailable.",
          "blocked_reason":"gpu_unsupported"
        },
        "open_cuda":{
          "ok":true,
          "installed_models":["modnet-webnn-256-fp32"],
          "models":[
            {
              "id":"modnet-webnn-256-fp32",
              "display_name":"MODNet",
              "task":"matting"
            }
          ],
          "missing_models":{},
          "available_effects":[
            "virtual_background.blur",
            "virtual_background.remove",
            "virtual_background.replace"
          ],
          "blocked_effects":{},
          "install_hints":[]
        },
        "video":{
          "enabled":false,
          "virtual_device_present":true,
          "virtual_device_available":true,
          "consumer_count":0,
          "video_effects":{
            "engine":"open_cuda",
            "virtual_background":{"mode":"none"}
          },
          "pipeline":{"running":false,"starting":false}
        },
        "audio":{
          "enabled":false,
          "mic_present":true,
          "source_error":"",
          "pipeline":{"running":false,"starting":false,"last_error":""},
          "speakers":{
            "present":true,
            "target_sink_error":"",
            "routing_active":false,
            "route_mode":"off",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  studiocast::gui::VideoPage page;
  page.UpdateStatus(studiocast::gui::DaemonStatusSnapshot::FromJson(json));

  auto *backgroundCombo =
      page.findChild<QComboBox *>(QStringLiteral("videoBackgroundModeCombo"));
  if (!Expect(backgroundCombo != nullptr,
              "video background mode combo should be findable"))
    return false;

  const int removeIndex =
      backgroundCombo->findData(QStringLiteral("remove"));
  if (!Expect(removeIndex >= 0,
              "video background remove mode should exist in the combo"))
    return false;

  auto *model = backgroundCombo->model();
  if (!Expect(model != nullptr,
              "video background mode combo should have an item model"))
    return false;

  const QModelIndex removeModelIndex = model->index(removeIndex, 0);
  return Expect(removeModelIndex.isValid(),
                "video background remove model index should be valid") &&
         Expect(model->flags(removeModelIndex).testFlag(Qt::ItemIsEnabled),
                "available virtual_background.remove should be enabled in "
                "the Camera page mode selector");
}

bool TestVideoPageKeepsReplaceModeSelectedWhileImagePathIsMissing() {
  const QString json = QStringLiteral(
      R"({
        "service_running":true,
        "maxine":{
          "supported":false,
          "ok":false,
          "summary":"Maxine unavailable.",
          "blocked_reason":"gpu_unsupported"
        },
        "open_cuda":{
          "ok":true,
          "installed_models":["modnet-webnn-256-fp32"],
          "models":[
            {
              "id":"modnet-webnn-256-fp32",
              "display_name":"MODNet",
              "task":"matting"
            }
          ],
          "missing_models":{},
          "available_effects":[
            "virtual_background.blur",
            "virtual_background.remove",
            "virtual_background.replace"
          ],
          "blocked_effects":{},
          "install_hints":[]
        },
        "video":{
          "enabled":false,
          "virtual_device_present":true,
          "virtual_device_available":true,
          "consumer_count":0,
          "video_effects":{
            "engine":"open_cuda",
            "virtual_background.blur":{"enabled":false,"model_id":"","strength":8},
            "virtual_background.remove":{
              "enabled":false,
              "strength":8,
              "model_id":"",
              "remove_color":"#000000",
              "greenscreen_mode":0,
              "greenscreen_temporal":true
            },
            "virtual_background.replace":{
              "enabled":true,
              "strength":8,
              "model_id":"",
              "remove_color":"#000000",
              "replace_path":"",
              "greenscreen_mode":0,
              "greenscreen_temporal":true
            }
          },
          "pipeline":{
            "running":false,
            "starting":false,
            "effects_plan":{
              "ordered":[],
              "disabled":[
                {
                  "id":"virtual_background.replace",
                  "reason":"Disabled: virtual background replace requires `replace_path`."
                }
              ]
            }
          }
        },
        "audio":{
          "enabled":false,
          "mic_present":true,
          "source_error":"",
          "pipeline":{"running":false,"starting":false,"last_error":""},
          "speakers":{
            "present":true,
            "target_sink_error":"",
            "routing_active":false,
            "route_mode":"off",
            "last_error":"",
            "pipeline_last_error":""
          }
        }
      })");

  studiocast::gui::VideoPage page;
  page.UpdateStatus(studiocast::gui::DaemonStatusSnapshot::FromJson(json));

  auto *backgroundCombo =
      page.findChild<QComboBox *>(QStringLiteral("videoBackgroundModeCombo"));
  auto *replaceEdit = page.findChild<QLineEdit *>(
      QStringLiteral("videoBackgroundReplaceImageEdit"));
  auto *browseButton = page.findChild<QPushButton *>(
      QStringLiteral("videoBackgroundBrowseReplaceImageButton"));

  return Expect(backgroundCombo != nullptr,
                "video background mode combo should be findable") &&
         Expect(replaceEdit != nullptr,
                "replace image path edit should be findable") &&
         Expect(browseButton != nullptr,
                "replace image browse button should be findable") &&
         Expect(backgroundCombo->currentData().toString() ==
                    QStringLiteral("replace"),
                "Camera page should keep Replace selected while image path is "
                "missing") &&
         Expect(!replaceEdit->isHidden(),
                "replace image path edit should be visible when Replace is "
                "selected") &&
         Expect(replaceEdit->isEnabled(),
                "replace image path edit should stay enabled so the user can "
                "choose an image") &&
         Expect(browseButton->isEnabled(),
                "replace image browse button should stay enabled so the user "
                "can choose an image");
}

} // namespace

int main(int argc, char **argv) {
  (void)::setenv("QT_QPA_PLATFORM", "offscreen", 0);
  QApplication app(argc, argv);

  bool ok = true;
  ok = TestUnreachableStatus() && ok;
  ok = TestStatusJsonCompatibilityShapes() && ok;
  ok = TestNestedPipelineEffectsPlanAndRawEffectsPreservation() && ok;
  ok = TestVideoEffectReadinessMissingModelWhileIdle() && ok;
  ok = TestAudioEndpointActionReadinessFields() && ok;
  ok = TestEngineModelDetailsAndConfiguredSelections() && ok;
  ok = TestEnginesModelsPageShowsOpenCudaSetupFix() && ok;
  ok = TestEnginesModelsPageShowsOpenAudioSetupFix() && ok;
  ok = TestEnginesModelsPageKeepsModelInstallsSeparateFromSetupRepair() && ok;
  ok = TestCameraPageShowsOpenCudaSetupFix() && ok;
  ok = TestOpenAudioRuntimeStatusFields() && ok;
  ok = TestMissingVirtualDevices() && ok;
  ok = TestMissingPhysicalDevices() && ok;
  ok = TestIdleNoConsumerStates() && ok;
  ok = TestConsumerDetectionErrorsAreNotIdle() && ok;
  ok = TestInvalidJsonPreservesRawPayload() && ok;
  ok = TestRawDiagnosticsFallbacks() && ok;
  ok = TestStatusPollerRefreshesDiagnosticsOutOfBand() && ok;
  ok = TestPendingDaemonWriteGuardSkipsRoutineStatusUntilWriteSettles() && ok;
  ok = TestVideoPageKeepsUserSelectedInputWhenRoutineStatusStillAuto() && ok;
  ok = TestVideoPageEnablesAvailableVirtualBackgroundRemoveMode() && ok;
  ok = TestVideoPageKeepsReplaceModeSelectedWhileImagePathIsMissing() && ok;
  return ok ? 0 : 1;
}
