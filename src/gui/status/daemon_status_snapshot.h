#pragma once

#include <QMap>
#include <QString>
#include <QStringList>

#include <vector>

namespace studiocast::gui {

enum class ReadinessState {
  Unknown,
  Ready,
  NeedsSetup,
  DaemonUnavailable,
  MissingVirtualDevice,
  NoPhysicalDevice,
  Idle,
  Processing,
  RecoverableError,
  FatalError,
  MissingModel,
};

struct DeviceReadiness {
  ReadinessState state = ReadinessState::Unknown;
  QString summary;
  QString detail;
  QStringList notes;
  QStringList disabledReasons;
};

struct EffectPlanDisabledReason {
  QString effectId;
  QString reason;
};

struct VideoEffectPlan {
  QStringList orderedEffectIds;
  QString vignetteAttachToEffectId;
  std::vector<EffectPlanDisabledReason> disabledEffects;
};

struct EffectReadiness {
  QString effectId;
  ReadinessState state = ReadinessState::Unknown;
  QString summary;
  QString detail;
  QString backend;
  QString requestedModelId;
  QString resolvedModelId;
  QString reason;
};

struct AudioEndpointStatus {
  bool present = false;
  bool enabled = false;
  bool virtualDevicePresent = false;
  bool consumerPresent = false;
  int consumerCount = 0;
  QString action;
  DeviceReadiness readiness;
  QString configuredDevice;
  QString resolvedDevice;
  QString activeDevice;
  QString routeMode;
  QString activeBackend;
  QString rawJson;
};

struct EngineModelEntry {
  QString id;
  QString displayName;
  QString category;
  QString details;
  bool installed = true;
};

struct ConfiguredModelEntry {
  QString owner;
  QString modelId;
  QString modelPath;
  bool modelIdReported = false;
  bool modelIdExplicitlyMissing = false;
};

struct OpenAudioRuntimeSnapshot {
  bool present = false;
  bool active = false;
  bool usingCpuFallback = false;
  bool disabled = false;
  QString activeProvider;
  QString selectedModelId;
  QString selectedModelPath;
  QString lastRuntimeWarning;
};

struct EngineStatus {
  QString id;
  QString label;
  bool present = false;
  bool ok = false;
  bool supported = false;
  QString rawJson;
  QString summary;
  QString blockedReason;
  QStringList blockedDetails;
  QStringList missingModels;
  QStringList blockedEffects;
  QStringList installHints;
  QStringList availableEffects;
  QString defaultModelId;
  std::vector<EngineModelEntry> installedModels;
  std::vector<EngineModelEntry> missingModelEntries;
  std::vector<ConfiguredModelEntry> configuredModels;
  int installedModelCount = 0;
  int knownModelCount = 0;
  int missingModelCount = 0;
  int configuredMissingModelCount = 0;
};

struct DaemonStatusSnapshot {
  bool reachable = false;
  bool parsed = false;
  QString transportError;
  QString parseError;

  QString rawJson;
  QString version;
  QString gitSha;
  QString socketPath;
  bool serviceRunning = false;

  DeviceReadiness camera;
  DeviceReadiness microphone;
  DeviceReadiness speakers;
  QString videoComputePreference;
  QString videoComputeResolvedBackend;
  QString videoComputeActiveBackend;
  QString videoComputeFallbackReason;
  QString videoComputeDegradedReason;
  QString videoEffectsEnginePreference;
  QString audioEffectsEnginePreference;
  QString rawVideoEffectsJson;
  QString rawAudioEffectsJson;
  VideoEffectPlan videoEffectPlan;
  QMap<QString, EffectReadiness> videoEffectReadiness;
  QString videoEffectsActiveBackends;
  QString microphoneActiveBackend;
  QString speakersActiveBackend;
  QString speakersRouteMode;
  AudioEndpointStatus microphoneEndpoint;
  AudioEndpointStatus speakersEndpoint;
  OpenAudioRuntimeSnapshot microphoneOpenAudioRuntime;
  OpenAudioRuntimeSnapshot speakersOpenAudioRuntime;

  EngineStatus maxine;
  EngineStatus openCuda;
  EngineStatus openVulkan;
  EngineStatus openAudio;

  static DaemonStatusSnapshot Unreachable(const QString &error);
  static DaemonStatusSnapshot FromJson(const QString &json);

  QString ServiceSummary() const;
  QString ServiceDetail() const;
  QString UserServiceSummary() const;
  QString UserServiceDetail() const;
  QString RawDiagnosticsText() const;
};

QString ReadinessLabel(ReadinessState state);

} // namespace studiocast::gui
