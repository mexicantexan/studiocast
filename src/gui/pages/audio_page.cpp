#include "audio_page.h"

#include <QClipboard>
#include <QComboBox>
#include <QFileDialog>
#include <QFrame>
#include <QGroupBox>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QThread>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/audio/audio_device_safety.h"
#include "core/audio/effects/broadcast_audio_effects.h"
#include "core/audio/pulse/pactl.h"
#include "core/audio/virtual_mic.h"
#include "core/audio/virtual_speaker.h"
#include "core/ipc/daemon_client.h"
#include "core/maxine/reason_codes.h"
#include "gui/status/daemon_status_snapshot.h"
#include "gui/text_edit_utils.h"

namespace studiocast::gui {
namespace {
constexpr const char *kAutoPulseSource = "auto";
constexpr const char *kAutoPulseSink = "auto";
constexpr const char *kAudioControlsUnavailable = "Audio controls unavailable";

bool IsBadLoopbackSourceCandidate(const std::string &name) {
  std::string reason;
  return studiocast::audio::IsUnsafeInputSourceName(name, &reason);
}

bool ParseJsonObject(const std::string &json, QJsonObject *outRoot,
                     QString *error) {
  QJsonParseError perr;
  const auto doc =
      QJsonDocument::fromJson(QByteArray::fromStdString(json), &perr);
  if (perr.error != QJsonParseError::NoError || !doc.isObject()) {
    if (error)
      *error = "JSON parse error: " + perr.errorString();
    return false;
  }
  if (outRoot)
    *outRoot = doc.object();
  return true;
}

QString FormatMaxineReasonCode(const QString &code) {
  if (code.isEmpty())
    return {};
  const std::string s = code.toStdString();
  return QString::fromStdString(studiocast::maxine::reasons::ToEnglish(s));
}

bool DaemonRequest(const std::string &request, std::string *outJson,
                   QString *outErr) {
  studiocast::ipc::DaemonCallResult res;
  studiocast::ipc::DaemonCallOptions options;
  options.connect_timeout_ms = 500;
  options.io_timeout_ms = 2500;
  std::string err;
  if (!studiocast::ipc::DaemonCall(request, &res, &err, options)) {
    if (outErr)
      *outErr = QString::fromStdString(err);
    return false;
  }
  if (!res.ok) {
    if (outErr)
      *outErr = QString::fromStdString(res.error_json.empty() ? "daemon_error"
                                                              : res.error_json);
    return false;
  }
  if (outJson)
    *outJson = res.json;
  return true;
}

bool ConfirmDestructiveAction(QWidget *parent, const QString &title,
                              const QString &text, const QString &detail) {
  QMessageBox box(parent);
  box.setIcon(QMessageBox::Warning);
  box.setWindowTitle(title);
  box.setText(text);
  box.setInformativeText(detail);
  auto *destroyButton =
      box.addButton(QStringLiteral("Destroy"), QMessageBox::AcceptRole);
  box.addButton(QMessageBox::Cancel);
  box.setDefaultButton(QMessageBox::Cancel);
  box.exec();
  return box.clickedButton() == destroyButton;
}
QString FirstLine(const QString &s) {
  const QString t = s.trimmed();
  const qsizetype nl = t.indexOf('\n');
  if (nl < 0)
    return t;
  return t.left(nl).trimmed();
}

QString FriendlyBackendLabel(const QString &id) {
  const QString v = id.trimmed().toLower();
  if (v.isEmpty())
    return QStringLiteral("—");
  if (v == QStringLiteral("maxine"))
    return QStringLiteral("Maxine");
  if (v == QStringLiteral("open_source") || v == QStringLiteral("open_audio"))
    return QStringLiteral("Open Audio");
  if (v == QStringLiteral("passthrough"))
    return QStringLiteral("Pass-through");
  if (v == QStringLiteral("loopback"))
    return QStringLiteral("Loopback");
  if (v == QStringLiteral("pipeline"))
    return QStringLiteral("Pipeline");
  if (v == QStringLiteral("off"))
    return QStringLiteral("Off");
  return id;
}

QString FriendlySpeakerRouteMode(const QString &routeMode) {
  const QString mode = routeMode.trimmed().toLower();
  if (mode.isEmpty() || mode == QStringLiteral("off"))
    return QStringLiteral("Off");
  if (mode == QStringLiteral("loopback"))
    return QStringLiteral("Loopback / pass-through");
  if (mode == QStringLiteral("pipeline"))
    return QStringLiteral("Processed pipeline");
  return routeMode;
}

QString FriendlySpeakerTarget(const QString &configured,
                              const QString &resolved,
                              const QString &active) {
  const QString activeTrimmed = active.trimmed();
  if (!activeTrimmed.isEmpty())
    return activeTrimmed;

  const QString resolvedTrimmed = resolved.trimmed();
  if (!resolvedTrimmed.isEmpty())
    return resolvedTrimmed;

  const QString configuredTrimmed = configured.trimmed();
  if (configuredTrimmed.isEmpty() ||
      configuredTrimmed == QString::fromLatin1(kAutoPulseSink)) {
    return QStringLiteral("Auto (Pulse default)");
  }
  return configuredTrimmed;
}

QLabel *MutedLabel(const QString &text, QWidget *parent) {
  auto *label = new QLabel(text, parent);
  label->setProperty("scRole", "muted");
  label->setWordWrap(true);
  return label;
}

QLabel *ValueLabel(const QString &text, QWidget *parent) {
  auto *label = new QLabel(text, parent);
  label->setProperty("scRole", "value");
  label->setWordWrap(true);
  return label;
}

QLabel *SectionLabel(const QString &text, QWidget *parent) {
  auto *label = new QLabel(text, parent);
  label->setProperty("scRole", "sectionTitle");
  return label;
}

void SetDynamicProperty(QWidget *widget, const char *name,
                        const QString &value) {
  if (!widget)
    return;
  widget->setProperty(name, value);
  widget->style()->unpolish(widget);
  widget->style()->polish(widget);
}

QFrame *MicrophonePanel(const QString &title, QWidget *parent,
                        QVBoxLayout **layoutOut) {
  auto *panel = new QFrame(parent);
  panel->setProperty("scRole", "microphonePanel");
  auto *layout = new QVBoxLayout(panel);
  layout->setContentsMargins(12, 10, 12, 12);
  layout->setSpacing(10);
  layout->addWidget(SectionLabel(title, panel));
  if (layoutOut)
    *layoutOut = layout;
  return panel;
}

} // namespace

AudioPage::AudioPage(AudioPageMode mode, QWidget *parent)
    : QWidget(parent), mode_(mode) {
  auto *root = new QVBoxLayout(this);
  root->setSpacing(12);
  root->setContentsMargins(16, 16, 16, 16);

  // -----------------------
  // Title row
  // -----------------------
  {
    auto *titleRow = new QHBoxLayout();
    titleRow->setContentsMargins(0, 0, 0, 0);

    titleLabel_ = new QLabel(
        mode_ == AudioPageMode::Microphone ? "Microphone" : "Speakers", this);
    titleLabel_->setProperty("scRole", "title");
    titleRow->addWidget(titleLabel_);

    titleRow->addStretch(1);

    advancedToggle_ = new QToolButton(this);
    advancedToggle_->setText("Show Diagnostics");
    advancedToggle_->setToolButtonStyle(Qt::ToolButtonTextOnly);
    advancedToggle_->setCheckable(true);
    advancedToggle_->setChecked(false);
    titleRow->addWidget(advancedToggle_);

    root->addLayout(titleRow);
  }

  // -----------------------
  // Backend / availability
  // -----------------------
  backendBox_ = new QGroupBox(
      mode_ == AudioPageMode::Microphone ? "Status" : "Effects Engine", this);
  {
    auto *backendLayout = new QVBoxLayout(backendBox_);
    backendLayout->setSpacing(10);

    if (mode_ == AudioPageMode::Microphone) {
      micStateLabel_ = new QLabel("Checking service", backendBox_);
      micStateLabel_->setProperty("scRole", "statusPill");
      micStateLabel_->setAlignment(Qt::AlignCenter);
      backendLayout->addWidget(micStateLabel_, 0, Qt::AlignLeft);

      micDetailLabel_ =
          MutedLabel(QStringLiteral("Waiting for microphone status."),
                     backendBox_);
      backendLayout->addWidget(micDetailLabel_);

      micSourceStatusLabel_ = MutedLabel(QString(), backendBox_);
      micSourceStatusLabel_->setVisible(false);
      backendLayout->addWidget(micSourceStatusLabel_);
    }

    auto *engineRow = new QHBoxLayout();
    engineRow->addWidget(new QLabel(mode_ == AudioPageMode::Microphone
                                        ? "Preference:"
                                        : "Preference:",
                                    backendBox_));
    engineCombo_ = new QComboBox(backendBox_);
    engineCombo_->addItem("Auto", "auto");
    engineCombo_->addItem("Maxine", "maxine");
    engineCombo_->addItem("Open Audio", "open_source");
    engineCombo_->addItem("Off", "off");
    engineRow->addWidget(engineCombo_);

    engineRow->addSpacing(12);
    engineRow->addWidget(new QLabel("Active:", backendBox_));
    engineActiveValue_ = ValueLabel("—", backendBox_);
    engineRow->addWidget(engineActiveValue_);
    engineRow->addStretch(1);
    backendLayout->addLayout(engineRow);

    aiInfoBanner_ = new QLabel(backendBox_);
    aiInfoBanner_->setWordWrap(true);
    aiInfoBanner_->setProperty("scBanner", "info");
    aiInfoBanner_->setVisible(false);
    backendLayout->addWidget(aiInfoBanner_);

    aiBanner_ = new QLabel(backendBox_);
    aiBanner_->setWordWrap(true);
    // Used for "daemon unavailable" and similar top-level issues.
    aiBanner_->setProperty("scBanner", "warning");
    aiBanner_->setVisible(false);
    backendLayout->addWidget(aiBanner_);

  }
  // -----------------------
  // Mode-specific content
  // -----------------------
  if (mode_ == AudioPageMode::Microphone) {
    // -----------------------
    // Microphone source
    // -----------------------
    micSourceBox_ = new QGroupBox("Source", this);
    {
      auto *sourceLayout = new QVBoxLayout(micSourceBox_);
      sourceLayout->setSpacing(10);

      auto *devRow = new QHBoxLayout();
      devRow->addWidget(new QLabel("Input:", micSourceBox_));
      sourceCombo_ = new QComboBox(micSourceBox_);
      sourceCombo_->setSizeAdjustPolicy(QComboBox::AdjustToContents);
      devRow->addWidget(sourceCombo_, 1);
      refreshSourcesBtn_ = new QPushButton("Refresh", micSourceBox_);
      devRow->addWidget(refreshSourcesBtn_);
      sourceLayout->addLayout(devRow);

      sourceLayout->addWidget(MutedLabel(
          "Virtual and loopback sources are hidden from this list. A configured "
          "source that is not currently available stays visible as missing.",
          micSourceBox_));
    }
    root->addWidget(micSourceBox_);

    // -----------------------
    // Microphone cleanup effects
    // -----------------------
    micEffectsBox_ = new QGroupBox("Cleanup Effects", this);
    {
      auto *aiLayout = new QVBoxLayout(micEffectsBox_);
      aiLayout->setSpacing(10);

      // Effect selection (Broadcast-style single selector).
      auto *effectRow = new QHBoxLayout();
      effectRow->addWidget(new QLabel("Effect:", micEffectsBox_));
      micEffectCombo_ = new QComboBox(micEffectsBox_);
      micEffectCombo_->setSizeAdjustPolicy(QComboBox::AdjustToContents);
      micEffectCombo_->addItem("Off", "off");
      micEffectCombo_->addItem("Noise removal", "noise");
      micEffectCombo_->addItem("Room echo removal", "echo");
      micEffectCombo_->addItem("Noise + echo (combined)", "noise_echo");
      micEffectCombo_->addItem("Studio voice", "studio_voice");
      effectRow->addWidget(micEffectCombo_, 1);
      aiLayout->addLayout(effectRow);

      // Strength
      auto *strengthRow = new QHBoxLayout();
      strengthRow->addWidget(new QLabel("Strength:", micEffectsBox_));
      strengthSlider_ = new QSlider(Qt::Horizontal, micEffectsBox_);
      strengthSlider_->setRange(0, 100);
      strengthSlider_->setValue(50);
      strengthRow->addWidget(strengthSlider_, 1);
      strengthValueLabel_ = new QLabel("50", micEffectsBox_);
      strengthValueLabel_->setMinimumWidth(32);
      strengthValueLabel_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
      strengthRow->addWidget(strengthValueLabel_);
      aiLayout->addLayout(strengthRow);

      aiLayout->addWidget(MutedLabel(
          "Set Effect to Off for pass-through. Other apps should use "
          "StudioCast Microphone.",
          micEffectsBox_));
    }
    root->addWidget(micEffectsBox_);

    root->addWidget(backendBox_);

    // -----------------------
    // Diagnostics
    // -----------------------
    micDetailsBox_ = new QGroupBox("Diagnostics", this);
    micDetailsBox_->setVisible(false);
    {
      auto *detailsLayout = new QVBoxLayout(micDetailsBox_);
      detailsLayout->setSpacing(10);

      micDetailsContent_ = new QWidget(micDetailsBox_);
      auto *detailsContentLayout = new QVBoxLayout(micDetailsContent_);
      detailsContentLayout->setContentsMargins(0, 0, 0, 0);
      detailsContentLayout->setSpacing(10);

      detailsContentLayout->addWidget(MutedLabel(
          "Technical microphone details, Open Audio model selection, and "
          "legacy audio controls stay here.",
          micDetailsContent_));

      QVBoxLayout *modelPanelLayout = nullptr;
      auto *modelPanel =
          MicrophonePanel("Open Audio Model", micDetailsContent_,
                          &modelPanelLayout);

      auto *modelRow = new QHBoxLayout();
      openAudioModelLabel_ = new QLabel("Model:", modelPanel);
      openAudioModelCombo_ = new QComboBox(modelPanel);
      openAudioModelCombo_->setSizeAdjustPolicy(QComboBox::AdjustToContents);
      openAudioModelCombo_->addItem("<auto>", "");
      modelRow->addWidget(openAudioModelLabel_);
      modelRow->addWidget(openAudioModelCombo_, 1);
      modelPanelLayout->addLayout(modelRow);

      auto *modelPathRow = new QHBoxLayout();
      openAudioModelPathLabel_ = new QLabel("Model path:", modelPanel);
      openAudioModelPathEdit_ = new QLineEdit(modelPanel);
      openAudioModelPathEdit_->setPlaceholderText(
          "(optional) /path/to/model.onnx");
      browseOpenAudioModelBtn_ = new QPushButton("Browse…", modelPanel);
      modelPathRow->addWidget(openAudioModelPathLabel_);
      modelPathRow->addWidget(openAudioModelPathEdit_, 1);
      modelPathRow->addWidget(browseOpenAudioModelBtn_);
      modelPanelLayout->addLayout(modelPathRow);

      openAudioInstallHintsBtn_ =
          new QPushButton("Open Audio install hints", modelPanel);
      openAudioInstallHintsBtn_->setEnabled(false);
      modelPanelLayout->addWidget(openAudioInstallHintsBtn_, 0, Qt::AlignLeft);
      detailsContentLayout->addWidget(modelPanel);

      // Virtual mic controls (advanced/system).
      vmicBox_ = new QGroupBox("Virtual Microphone Lifecycle",
                               micDetailsContent_);
      {
        auto *vmicLayout = new QVBoxLayout(vmicBox_);
        vmicLayout->setSpacing(10);

        auto *buttonsRow = new QHBoxLayout();
        createBtn_ = new QPushButton("Create virtual mic", vmicBox_);
        destroyBtn_ = new QPushButton("Destroy virtual mic", vmicBox_);
        destroyBtn_->setProperty("scVariant", "danger");

        buttonsRow->addWidget(createBtn_);
        buttonsRow->addWidget(destroyBtn_);
        buttonsRow->addStretch(1);
        vmicLayout->addLayout(buttonsRow);

        vmicLayout->addWidget(MutedLabel(
            "Other apps should select StudioCast Microphone. The processed "
            "feed comes from the daemon audio pipeline.",
            vmicBox_));
      }
      detailsContentLayout->addWidget(vmicBox_);

      // Legacy loopback controls (advanced/debug).
      legacyInputBox_ =
          new QGroupBox("Legacy Loopback", micDetailsContent_);
      {
        auto *inputLayout = new QVBoxLayout(legacyInputBox_);
        inputLayout->setSpacing(10);

        auto *portRow = new QHBoxLayout();
        portRow->addWidget(new QLabel("Input port:", legacyInputBox_));
        portCombo_ = new QComboBox(legacyInputBox_);
        portRow->addWidget(portCombo_, 1);
        inputLayout->addLayout(portRow);

        auto *latencyRow = new QHBoxLayout();
        latencyRow->addWidget(new QLabel("Latency (ms):", legacyInputBox_));
        latencySpin_ = new QSpinBox(legacyInputBox_);
        latencySpin_->setRange(1, 200);
        latencySpin_->setValue(10);
        latencyRow->addWidget(latencySpin_);
        latencyRow->addStretch(1);
        inputLayout->addLayout(latencyRow);

        auto *loopbackButtons = new QHBoxLayout();
        startBtn_ = new QPushButton("Start loopback", legacyInputBox_);
        stopBtn_ = new QPushButton("Stop loopback", legacyInputBox_);
        loopbackButtons->addWidget(startBtn_);
        loopbackButtons->addWidget(stopBtn_);
        loopbackButtons->addStretch(1);
        inputLayout->addLayout(loopbackButtons);

        inputLayout->addWidget(MutedLabel(
            "Legacy module-loopback path for debug/dev use. The daemon "
            "pipeline is the preferred path.",
            legacyInputBox_));
      }
      detailsContentLayout->addWidget(legacyInputBox_);

      // Raw status.
      statusBox_ = new QGroupBox("Raw Microphone Status", micDetailsContent_);
      {
        auto *statusLayout = new QVBoxLayout(statusBox_);
        statusLayout->setSpacing(10);

        statusText_ = new QPlainTextEdit(statusBox_);
        statusText_->setReadOnly(true);
        statusText_->setMinimumHeight(220);
        statusLayout->addWidget(statusText_, 1);

        auto *buttonsRow = new QHBoxLayout();
        refreshStatusBtn_ = new QPushButton("Refresh status", statusBox_);
        buttonsRow->addWidget(refreshStatusBtn_);
        auto *copyStatusBtn =
            new QPushButton("Copy raw microphone status", statusBox_);
        buttonsRow->addWidget(copyStatusBtn);
        buttonsRow->addStretch(1);
        statusLayout->addLayout(buttonsRow);

        connect(copyStatusBtn, &QPushButton::clicked, this, [this] {
          if (auto *cb = QGuiApplication::clipboard())
            cb->setText(statusText_ ? statusText_->toPlainText() : QString());
        });
      }
      detailsContentLayout->addWidget(statusBox_);

      micDetailsContent_->setVisible(false);
      detailsLayout->addWidget(micDetailsContent_);
    }
    root->addWidget(micDetailsBox_);

  } else {
    // -----------------------
    // Speaker routing / output target
    // -----------------------
    speakersBox_ = new QGroupBox("Routing & Output Target", this);
    {
      auto *spkLayout = new QVBoxLayout(speakersBox_);
      spkLayout->setSpacing(10);

      auto *targetRow = new QHBoxLayout();
      targetRow->addWidget(new QLabel("Physical output:", speakersBox_));
      speakerTargetCombo_ = new QComboBox(speakersBox_);
      speakerTargetCombo_->setSizeAdjustPolicy(QComboBox::AdjustToContents);
      targetRow->addWidget(speakerTargetCombo_, 1);
      refreshSpeakerTargetsBtn_ = new QPushButton("Refresh", speakersBox_);
      targetRow->addWidget(refreshSpeakerTargetsBtn_);
      spkLayout->addLayout(targetRow);

      speakerTargetStatusLabel_ = MutedLabel(QString(), speakersBox_);
      speakerTargetStatusLabel_->setVisible(false);
      spkLayout->addWidget(speakerTargetStatusLabel_);

      auto *buttonsRow = new QHBoxLayout();
      enableSpeakersBtn_ =
          new QPushButton("Enable speakers device", speakersBox_);
      enableSpeakersBtn_->setProperty("scVariant", "primary");
      stopSpeakersBtn_ = new QPushButton("Stop routing", speakersBox_);
      buttonsRow->addWidget(enableSpeakersBtn_);
      buttonsRow->addWidget(stopSpeakersBtn_);
      buttonsRow->addStretch(1);
      spkLayout->addLayout(buttonsRow);

      spkLayout->addWidget(MutedLabel(
          "In other apps, select “StudioCast Speakers” as the output device. "
          "StudioCast routes that audio to the physical output selected here.",
          speakersBox_));
    }
    root->addWidget(speakersBox_);

    // -----------------------
    // Speaker route state
    // -----------------------
    speakerRouteStateBox_ = new QGroupBox("Route State", this);
    {
      auto *routeLayout = new QVBoxLayout(speakerRouteStateBox_);
      routeLayout->setSpacing(10);

      speakerRouteStateLabel_ = new QLabel("Checking service",
                                           speakerRouteStateBox_);
      speakerRouteStateLabel_->setProperty("scRole", "statusPill");
      speakerRouteStateLabel_->setAlignment(Qt::AlignCenter);
      routeLayout->addWidget(speakerRouteStateLabel_, 0, Qt::AlignLeft);

      speakerRouteDetailLabel_ = MutedLabel(
          QStringLiteral("Waiting for speaker routing status."),
          speakerRouteStateBox_);
      routeLayout->addWidget(speakerRouteDetailLabel_);

      auto *modeRow = new QHBoxLayout();
      modeRow->addWidget(new QLabel("Mode:", speakerRouteStateBox_));
      speakerRouteModeValue_ = ValueLabel("—", speakerRouteStateBox_);
      modeRow->addWidget(speakerRouteModeValue_, 1);
      modeRow->addSpacing(12);
      modeRow->addWidget(new QLabel("Target:", speakerRouteStateBox_));
      speakerRouteTargetValue_ = ValueLabel("—", speakerRouteStateBox_);
      modeRow->addWidget(speakerRouteTargetValue_, 2);
      routeLayout->addLayout(modeRow);
    }
    root->addWidget(speakerRouteStateBox_);

    // -----------------------
    // Speaker cleanup effects
    // -----------------------
    speakerEffectsBox_ = new QGroupBox("Cleanup Effects", this);
    {
      auto *spkLayout = new QVBoxLayout(speakerEffectsBox_);
      spkLayout->setSpacing(10);

      auto *effectRow = new QHBoxLayout();
      effectRow->addWidget(new QLabel("Effect:", speakerEffectsBox_));
      speakerEffectCombo_ = new QComboBox(speakerEffectsBox_);
      speakerEffectCombo_->setSizeAdjustPolicy(QComboBox::AdjustToContents);
      speakerEffectCombo_->addItem("Off", "off");
      speakerEffectCombo_->addItem("Noise removal", "noise");
      speakerEffectCombo_->addItem("Room echo removal", "echo");
      speakerEffectCombo_->addItem("Noise + echo (combined)", "noise_echo");
      effectRow->addWidget(speakerEffectCombo_, 1);
      spkLayout->addLayout(effectRow);

      auto *strengthRow = new QHBoxLayout();
      strengthRow->addWidget(new QLabel("Strength:", speakerEffectsBox_));
      speakerStrengthSlider_ = new QSlider(Qt::Horizontal, speakerEffectsBox_);
      speakerStrengthSlider_->setRange(0, 100);
      speakerStrengthSlider_->setValue(50);
      strengthRow->addWidget(speakerStrengthSlider_, 1);
      speakerStrengthValueLabel_ = new QLabel("50", speakerEffectsBox_);
      speakerStrengthValueLabel_->setMinimumWidth(32);
      speakerStrengthValueLabel_->setAlignment(Qt::AlignRight |
                                               Qt::AlignVCenter);
      strengthRow->addWidget(speakerStrengthValueLabel_);
      spkLayout->addLayout(strengthRow);

      spkLayout->addWidget(MutedLabel(
          "Set Effect to Off for pass-through. Cleanup applies when the "
          "StudioCast Speakers use the processed pipeline.",
          speakerEffectsBox_));
    }
    root->addWidget(speakerEffectsBox_);

    root->addWidget(backendBox_);

    // -----------------------
    // Diagnostics
    // -----------------------
    speakerDetailsBox_ = new QGroupBox("Diagnostics", this);
    speakerDetailsBox_->setVisible(false);
    {
      auto *detailsLayout = new QVBoxLayout(speakerDetailsBox_);
      detailsLayout->setSpacing(10);

      speakerDetailsContent_ = new QWidget(speakerDetailsBox_);
      auto *detailsContentLayout = new QVBoxLayout(speakerDetailsContent_);
      detailsContentLayout->setContentsMargins(0, 0, 0, 0);
      detailsContentLayout->setSpacing(10);

      detailsContentLayout->addWidget(MutedLabel(
          "Technical speaker details, Open Audio model selection, and "
          "destructive lifecycle controls stay here.",
          speakerDetailsContent_));

      QVBoxLayout *modelPanelLayout = nullptr;
      auto *modelPanel =
          MicrophonePanel("Open Audio Model", speakerDetailsContent_,
                          &modelPanelLayout);

      auto *modelRow = new QHBoxLayout();
      speakerOpenAudioModelLabel_ = new QLabel("Model:", modelPanel);
      speakerOpenAudioModelCombo_ = new QComboBox(modelPanel);
      speakerOpenAudioModelCombo_->setSizeAdjustPolicy(
          QComboBox::AdjustToContents);
      speakerOpenAudioModelCombo_->addItem("<auto>", "");
      modelRow->addWidget(speakerOpenAudioModelLabel_);
      modelRow->addWidget(speakerOpenAudioModelCombo_, 1);
      modelPanelLayout->addLayout(modelRow);

      auto *modelPathRow = new QHBoxLayout();
      speakerOpenAudioModelPathLabel_ = new QLabel("Model path:", modelPanel);
      speakerOpenAudioModelPathEdit_ = new QLineEdit(modelPanel);
      speakerOpenAudioModelPathEdit_->setPlaceholderText(
          "(optional) /path/to/model.onnx");
      speakerBrowseOpenAudioModelBtn_ =
          new QPushButton("Browse…", modelPanel);
      modelPathRow->addWidget(speakerOpenAudioModelPathLabel_);
      modelPathRow->addWidget(speakerOpenAudioModelPathEdit_, 1);
      modelPathRow->addWidget(speakerBrowseOpenAudioModelBtn_);
      modelPanelLayout->addLayout(modelPathRow);

      openAudioInstallHintsBtn_ =
          new QPushButton("Open Audio install hints", modelPanel);
      openAudioInstallHintsBtn_->setEnabled(false);
      modelPanelLayout->addWidget(openAudioInstallHintsBtn_, 0, Qt::AlignLeft);
      detailsContentLayout->addWidget(modelPanel);

      speakerLifecycleBox_ =
          new QGroupBox("Advanced Speaker Lifecycle", speakerDetailsContent_);
      {
        auto *lifecycleLayout = new QVBoxLayout(speakerLifecycleBox_);
        lifecycleLayout->setSpacing(10);

        auto *buttonsRow = new QHBoxLayout();
        destroySpeakersBtn_ =
            new QPushButton("Destroy speakers device", speakerLifecycleBox_);
        destroySpeakersBtn_->setProperty("scVariant", "danger");
        buttonsRow->addWidget(destroySpeakersBtn_);
        buttonsRow->addStretch(1);
        lifecycleLayout->addLayout(buttonsRow);

        lifecycleLayout->addWidget(MutedLabel(
            "Destroy removes the virtual speakers device. Use Enable speakers "
            "device above to recreate it through the daemon.",
            speakerLifecycleBox_));
      }
      detailsContentLayout->addWidget(speakerLifecycleBox_);

      statusBox_ = new QGroupBox("Raw Speaker Status", speakerDetailsContent_);
      {
        auto *statusLayout = new QVBoxLayout(statusBox_);
        statusLayout->setSpacing(10);

        statusText_ = new QPlainTextEdit(statusBox_);
        statusText_->setReadOnly(true);
        statusText_->setMinimumHeight(220);
        statusLayout->addWidget(statusText_, 1);

        auto *buttonsRow = new QHBoxLayout();
        refreshStatusBtn_ = new QPushButton("Refresh status", statusBox_);
        buttonsRow->addWidget(refreshStatusBtn_);
        auto *copyStatusBtn =
            new QPushButton("Copy raw speaker status", statusBox_);
        buttonsRow->addWidget(copyStatusBtn);
        buttonsRow->addStretch(1);
        statusLayout->addLayout(buttonsRow);

        connect(copyStatusBtn, &QPushButton::clicked, this, [this] {
          if (auto *cb = QGuiApplication::clipboard())
            cb->setText(statusText_ ? statusText_->toPlainText() : QString());
        });
      }
      detailsContentLayout->addWidget(statusBox_);

      speakerDetailsContent_->setVisible(false);
      detailsLayout->addWidget(speakerDetailsContent_);
    }
    root->addWidget(speakerDetailsBox_);
  }
  root->addStretch(1);

  // -----------------------
  // Wiring
  // -----------------------
  connect(advancedToggle_, &QToolButton::toggled, this,
          &AudioPage::OnToggleAdvanced);

  if (refreshSourcesBtn_)
    connect(refreshSourcesBtn_, &QPushButton::clicked, this,
            &AudioPage::RefreshSources);
  if (refreshSpeakerTargetsBtn_)
    connect(refreshSpeakerTargetsBtn_, &QPushButton::clicked, this,
            &AudioPage::RefreshSpeakerTargets);
  if (refreshStatusBtn_)
    connect(refreshStatusBtn_, &QPushButton::clicked, this,
            &AudioPage::RefreshStatus);

  if (engineCombo_) {
    connect(engineCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &AudioPage::OnAiEngineChanged);
  }
  if (openAudioModelCombo_) {
    connect(openAudioModelCombo_,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &AudioPage::OnAiOpenAudioModelChanged);
  }
  if (openAudioModelPathEdit_) {
    connect(openAudioModelPathEdit_, &QLineEdit::editingFinished, this,
            &AudioPage::OnAiOpenAudioModelPathEdited);
  }
  if (browseOpenAudioModelBtn_) {
    connect(browseOpenAudioModelBtn_, &QPushButton::clicked, this,
            &AudioPage::OnAiBrowseOpenAudioModel);
  }
  if (openAudioInstallHintsBtn_) {
    connect(openAudioInstallHintsBtn_, &QPushButton::clicked, this,
            &AudioPage::OnOpenAudioInstallHints);
  }

  if (micEffectCombo_) {
    connect(micEffectCombo_,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &AudioPage::OnMicEffectChanged);
  }
  if (strengthSlider_)
    connect(strengthSlider_, &QSlider::valueChanged, this,
            &AudioPage::OnAiStrengthChanged);

  if (speakerEffectCombo_) {
    connect(speakerEffectCombo_,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &AudioPage::OnSpeakerEffectChanged);
  }
  if (speakerStrengthSlider_)
    connect(speakerStrengthSlider_, &QSlider::valueChanged, this,
            &AudioPage::OnAiSpeakerStrengthChanged);
  if (speakerOpenAudioModelCombo_) {
    connect(speakerOpenAudioModelCombo_,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &AudioPage::OnAiSpeakerOpenAudioModelChanged);
  }
  if (speakerOpenAudioModelPathEdit_) {
    connect(speakerOpenAudioModelPathEdit_, &QLineEdit::editingFinished, this,
            &AudioPage::OnAiSpeakerOpenAudioModelPathEdited);
  }
  if (speakerBrowseOpenAudioModelBtn_) {
    connect(speakerBrowseOpenAudioModelBtn_, &QPushButton::clicked, this,
            &AudioPage::OnAiSpeakerBrowseOpenAudioModel);
  }

  if (createBtn_)
    connect(createBtn_, &QPushButton::clicked, this,
            &AudioPage::OnCreateVirtualMic);
  if (destroyBtn_)
    connect(destroyBtn_, &QPushButton::clicked, this,
            &AudioPage::OnDestroyVirtualMic);
  if (startBtn_)
    connect(startBtn_, &QPushButton::clicked, this,
            &AudioPage::OnStartLoopback);
  if (stopBtn_)
    connect(stopBtn_, &QPushButton::clicked, this, &AudioPage::OnStopLoopback);

  if (enableSpeakersBtn_)
    connect(enableSpeakersBtn_, &QPushButton::clicked, this,
            &AudioPage::OnEnableVirtualSpeakers);
  if (stopSpeakersBtn_)
    connect(stopSpeakersBtn_, &QPushButton::clicked, this,
            &AudioPage::OnStopSpeakersRouting);
  if (destroySpeakersBtn_)
    connect(destroySpeakersBtn_, &QPushButton::clicked, this,
            &AudioPage::OnDestroyVirtualSpeakers);
  if (speakerTargetCombo_) {
    connect(speakerTargetCombo_,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &AudioPage::OnSpeakerTargetChanged);
  }

  if (sourceCombo_) {
    connect(sourceCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &AudioPage::OnSourceChanged);
  }

  audioWriteDebounceTimer_ = new QTimer(this);
  audioWriteDebounceTimer_->setSingleShot(true);
  audioWriteDebounceTimer_->setInterval(180);
  connect(audioWriteDebounceTimer_, &QTimer::timeout, this,
          [this] { PushDaemonAudioConfig(); });

  SetAdvancedVisible(false);

  // Initial state.
  if (mode_ == AudioPageMode::Microphone) {
    RefreshSources();
  } else {
    RefreshSpeakerTargets();
  }
  RefreshStatus();

#ifdef NDEBUG
  // In release/production builds we do not support pass-through routing via
  // module-loopback. The processed feed is expected to come from the audio
  // pipeline.
  if (legacyInputBox_) {
    legacyInputBox_->setTitle(
        "Input (legacy loopback - disabled in release builds)");
    legacyInputBox_->setEnabled(false);
  }
  if (startBtn_)
    startBtn_->setVisible(false);
  if (stopBtn_)
    stopBtn_->setVisible(false);
#endif
}

void AudioPage::ShowError(const QString &title, const QString &details) {
  QMessageBox::critical(this, title, details);
}

void AudioPage::UpdateStatus(const DaemonStatusSnapshot &snapshot) {
  daemonLastStatusJson_ = snapshot.rawJson;
  daemonStatusDetail_ = snapshot.UserServiceDetail();

  daemonMicrophoneAction_ = snapshot.microphoneEndpoint.action;
  daemonMicVirtualDevicePresent_ =
      snapshot.microphoneEndpoint.virtualDevicePresent;
  daemonSpeakersAction_ = snapshot.speakersEndpoint.action;
  daemonSpeakersVirtualDevicePresent_ =
      snapshot.speakersEndpoint.virtualDevicePresent;
  daemonSpeakersRoutingActive_ =
      snapshot.speakersEndpoint.readiness.state == ReadinessState::Processing ||
      snapshot.speakersEndpoint.action == QStringLiteral("stop_routing");
  daemonSpeakersRouteMode_ = snapshot.speakersEndpoint.routeMode;
  daemonSpeakerTarget_ = snapshot.speakersEndpoint.configuredDevice;

  if (!snapshot.reachable || !snapshot.parsed) {
    SetAiControlsEnabled(false, snapshot.UserServiceDetail());
    daemonStatusText_ = snapshot.RawDiagnosticsText();
    UpdateReleaseControlsFromCachedStatus();
    if (statusText_)
      SetPlainTextPreservingScroll(statusText_, daemonStatusText_);
    return;
  }

  RefreshStatus();
}

void AudioPage::OnToggleAdvanced(bool checked) { SetAdvancedVisible(checked); }

void AudioPage::SetAdvancedVisible(bool visible) {
  if (advancedToggle_) {
    advancedToggle_->setChecked(visible);
    advancedToggle_->setText(visible ? QStringLiteral("Hide Diagnostics")
                                     : QStringLiteral("Show Diagnostics"));
  }

  if (micDetailsBox_) {
    micDetailsBox_->setVisible(visible);
    if (micDetailsContent_)
      micDetailsContent_->setVisible(visible);
    return;
  }

  if (speakerDetailsBox_) {
    speakerDetailsBox_->setVisible(visible);
    if (speakerDetailsContent_)
      speakerDetailsContent_->setVisible(visible);
    return;
  }

  // Advanced items are intentionally per-mode.
  if (legacyInputBox_)
    legacyInputBox_->setVisible(visible);
  if (vmicBox_)
    vmicBox_->setVisible(visible);
  if (statusBox_)
    statusBox_->setVisible(visible);
}

void AudioPage::SetMicStatusSummary(const QString &state,
                                    const QString &detail,
                                    const QString &status) {
  if (micStateLabel_) {
    micStateLabel_->setText(state);
    SetDynamicProperty(micStateLabel_, "scStatus", status);
  }
  if (micDetailLabel_) {
    micDetailLabel_->setText(detail);
    micDetailLabel_->setVisible(!detail.trimmed().isEmpty());
  }
}

void AudioPage::SetSpeakerRouteSummary(const QString &state,
                                       const QString &detail,
                                       const QString &status,
                                       const QString &routeMode,
                                       const QString &target) {
  if (speakerRouteStateLabel_) {
    speakerRouteStateLabel_->setText(state);
    SetDynamicProperty(speakerRouteStateLabel_, "scStatus", status);
  }
  if (speakerRouteDetailLabel_) {
    speakerRouteDetailLabel_->setText(detail);
    speakerRouteDetailLabel_->setVisible(!detail.trimmed().isEmpty());
  }
  if (speakerRouteModeValue_) {
    speakerRouteModeValue_->setText(routeMode.trimmed().isEmpty()
                                        ? QStringLiteral("Unknown")
                                        : routeMode.trimmed());
  }
  if (speakerRouteTargetValue_) {
    speakerRouteTargetValue_->setText(target.trimmed().isEmpty()
                                          ? QStringLiteral("Unknown")
                                          : target.trimmed());
  }
}

void AudioPage::RefreshSources() {
  if (!sourceCombo_)
    return;
  if (sourceRefreshThread_)
    return;

  updatingSourceUi_ = true;
  sourceCombo_->blockSignals(true);
  sourceCombo_->clear();
  sourceCombo_->addItem(QStringLiteral("Loading microphone inputs..."));
  sourceCombo_->setEnabled(false);
  if (portCombo_) {
    portCombo_->clear();
    portCombo_->setEnabled(false);
  }
  cachedSources_.clear();
  sourceCombo_->blockSignals(false);
  updatingSourceUi_ = false;
  if (refreshSourcesBtn_)
    refreshSourcesBtn_->setEnabled(false);

  auto result = std::make_shared<SourceRefreshResult>();
  auto *thread = QThread::create([result] {
    result->pactlOk =
        studiocast::audio::pulse::PactlAvailable(&result->pactlDetails);
    if (!result->pactlOk)
      return;
    result->sources =
        studiocast::audio::pulse::ListSourcesDetailed(&result->listError);
    std::string defaultError;
    result->defaultSource =
        studiocast::audio::pulse::GetDefaultSourceName(&defaultError);
  });
  sourceRefreshThread_ = thread;
  connect(thread, &QThread::finished, thread, &QObject::deleteLater);
  connect(thread, &QThread::finished, this, [this, thread, result] {
    if (sourceRefreshThread_ == thread)
      sourceRefreshThread_ = nullptr;
    ApplySourceRefreshResult(*result);
    if (refreshSourcesBtn_)
      refreshSourcesBtn_->setEnabled(true);
  });
  thread->start();
}

void AudioPage::ApplySourceRefreshResult(const SourceRefreshResult &result) {
  if (!sourceCombo_)
    return;

  updatingSourceUi_ = true;
  sourceCombo_->blockSignals(true);
  sourceCombo_->clear();
  if (portCombo_) {
    portCombo_->clear();
    portCombo_->setEnabled(false);
  }
  cachedSources_.clear();

  if (!result.pactlOk) {
    sourceCombo_->addItem(QString::fromLatin1(kAudioControlsUnavailable));
    sourceCombo_->setEnabled(false);
    sourceCombo_->blockSignals(false);
    updatingSourceUi_ = false;
    if (statusText_) {
      SetPlainTextPreservingScroll(
          statusText_,
          QStringLiteral("Audio input control error:\n%1")
              .arg(QString::fromStdString(result.pactlDetails)));
    }
    ShowError("Audio",
              QStringLiteral("Microphone inputs cannot be listed.\n\nOpen "
                             "Support for technical details."));
    return;
  }

  sourceCombo_->setEnabled(true);
  sourceCombo_->addItem("Auto (Pulse default)",
                        QVariant(QString::fromLatin1(kAutoPulseSource)));

  cachedSources_ = result.sources;
  if (!result.listError.empty() && statusText_) {
    // Non-fatal warning, but useful
    SetPlainTextPreservingScroll(statusText_,
                                 QString::fromStdString("Warning: " +
                                                        result.listError));
  }

  const QString daemonSource = daemonSource_.trimmed();
  int defaultIndex = 0;
  int daemonIndex = -1;
  int added = 1;

  for (const auto &s : cachedSources_) {
    if (s.name.empty())
      continue;
    if (IsBadLoopbackSourceCandidate(s.name))
      continue;

    const std::string label = s.description.empty() ? s.name : s.description;

    // Display label, but store raw source name in item data.
    sourceCombo_->addItem(QString::fromStdString(label),
                          QVariant(QString::fromStdString(s.name)));

    if (result.defaultSource && s.name == *result.defaultSource) {
      defaultIndex = added;
    }
    if (!daemonSource.isEmpty() &&
        daemonSource != QString::fromLatin1(kAutoPulseSource) &&
        s.name == daemonSource.toStdString()) {
      daemonIndex = added;
    }
    ++added;
  }

  if (added == 1) {
    sourceCombo_->addItem("<no suitable sources found>");
  }

  int targetIndex = 0;
  if (daemonSource.isEmpty() ||
      daemonSource == QString::fromLatin1(kAutoPulseSource)) {
    targetIndex = 0;
  } else if (daemonIndex >= 0) {
    targetIndex = daemonIndex;
  } else {
    const int insertAt = std::min(1, sourceCombo_->count());
    sourceCombo_->insertItem(insertAt,
                             QStringLiteral("<missing: %1>").arg(daemonSource),
                             QVariant(daemonSource));
    targetIndex = insertAt;
  }

  if (daemonSource.isEmpty() && defaultIndex >= 0) {
    // Before the first daemon status arrives, show the current Pulse default
    // without writing it back to the daemon.
    targetIndex = defaultIndex;
  }

  sourceCombo_->setCurrentIndex(targetIndex);
  sourceCombo_->blockSignals(false);
  updatingSourceUi_ = false;
  UpdatePortControlsForSelectedSource(false);
}

void AudioPage::RefreshSpeakerTargets() {
  if (!speakerTargetCombo_)
    return;
  if (speakerTargetRefreshThread_)
    return;

  updatingSpeakerTargetUi_ = true;
  speakerTargetCombo_->blockSignals(true);
  speakerTargetCombo_->clear();
  speakerTargetCombo_->addItem(QStringLiteral("Loading speaker outputs..."));
  speakerTargetCombo_->setEnabled(false);
  speakerTargetCombo_->blockSignals(false);
  updatingSpeakerTargetUi_ = false;
  if (refreshSpeakerTargetsBtn_)
    refreshSpeakerTargetsBtn_->setEnabled(false);

  auto result = std::make_shared<SpeakerTargetRefreshResult>();
  auto *thread = QThread::create([result] {
    result->pactlOk =
        studiocast::audio::pulse::PactlAvailable(&result->pactlDetails);
    if (!result->pactlOk)
      return;
    result->sinks = studiocast::audio::pulse::ListSinks(&result->listError);
    std::string defaultError;
    result->defaultSink =
        studiocast::audio::pulse::GetDefaultSinkName(&defaultError);
  });
  speakerTargetRefreshThread_ = thread;
  connect(thread, &QThread::finished, thread, &QObject::deleteLater);
  connect(thread, &QThread::finished, this, [this, thread, result] {
    if (speakerTargetRefreshThread_ == thread)
      speakerTargetRefreshThread_ = nullptr;
    ApplySpeakerTargetRefreshResult(*result);
    if (refreshSpeakerTargetsBtn_)
      refreshSpeakerTargetsBtn_->setEnabled(true);
  });
  thread->start();
}

void AudioPage::ApplySpeakerTargetRefreshResult(
    const SpeakerTargetRefreshResult &result) {
  if (!speakerTargetCombo_)
    return;

  updatingSpeakerTargetUi_ = true;
  speakerTargetCombo_->blockSignals(true);
  speakerTargetCombo_->clear();

  if (!result.pactlOk) {
    speakerTargetCombo_->addItem(QString::fromLatin1(kAudioControlsUnavailable));
    speakerTargetCombo_->setEnabled(false);
    if (speakerTargetStatusLabel_) {
      speakerTargetStatusLabel_->setText(QStringLiteral(
          "Speaker outputs cannot be listed. Open Support for technical "
          "details."));
      speakerTargetStatusLabel_->setVisible(true);
    }
    speakerTargetCombo_->blockSignals(false);
    updatingSpeakerTargetUi_ = false;
    if (statusText_) {
      SetPlainTextPreservingScroll(
          statusText_,
          QStringLiteral("Audio output control error:\n%1")
              .arg(QString::fromStdString(result.pactlDetails)));
    }
    ShowError("Audio",
              QStringLiteral("Speaker outputs cannot be listed.\n\nOpen "
                             "Support for technical details."));
    return;
  }

  speakerTargetCombo_->setEnabled(true);
  speakerTargetCombo_->addItem("Auto (Pulse default)",
                               QVariant(QString::fromLatin1(kAutoPulseSink)));

  if (!result.listError.empty() && statusText_) {
    SetPlainTextPreservingScroll(statusText_,
                                 QString::fromStdString("Warning: " +
                                                        result.listError));
  }

  const QString daemonTarget = daemonSpeakerTarget_.trimmed();
  int defaultIndex = 0;
  int daemonIndex = -1;
  int added = 1;
  QStringList targetNotes;

  for (const auto &sink : result.sinks) {
    if (sink.name.empty())
      continue;

    std::string reason;
    if (studiocast::audio::IsUnsafeSpeakerTargetSinkName(sink.name, &reason))
      continue;

    std::string label = sink.name;
    if (result.defaultSink && sink.name == *result.defaultSink)
      label += " (default)";

    speakerTargetCombo_->addItem(QString::fromStdString(label),
                                 QVariant(QString::fromStdString(sink.name)));

    if (result.defaultSink && sink.name == *result.defaultSink)
      defaultIndex = added;
    if (!daemonTarget.isEmpty() &&
        daemonTarget != QString::fromLatin1(kAutoPulseSink) &&
        sink.name == daemonTarget.toStdString()) {
      daemonIndex = added;
    }
    ++added;
  }

  if (added == 1) {
    speakerTargetCombo_->addItem("<no physical output sinks found>");
    targetNotes << QStringLiteral(
        "No safe physical output sinks were reported by PulseAudio.");
  }

  int targetIndex = 0;
  if (daemonTarget.isEmpty() ||
      daemonTarget == QString::fromLatin1(kAutoPulseSink)) {
    targetIndex = 0;
  } else if (daemonIndex >= 0) {
    targetIndex = daemonIndex;
  } else {
    const int insertAt = std::min(1, speakerTargetCombo_->count());
    speakerTargetCombo_->insertItem(
        insertAt, QStringLiteral("<disconnected: %1>").arg(daemonTarget),
        QVariant(daemonTarget));
    targetIndex = insertAt;
    targetNotes << QStringLiteral(
        "The configured speaker target is disconnected and is preserved until "
        "you choose another output.");
  }

  if (daemonTarget.isEmpty() && defaultIndex >= 0)
    targetIndex = defaultIndex;

  speakerTargetCombo_->setCurrentIndex(targetIndex);
  if (speakerTargetStatusLabel_) {
    speakerTargetStatusLabel_->setText(targetNotes.join(QStringLiteral("\n")));
    speakerTargetStatusLabel_->setVisible(!targetNotes.isEmpty());
  }
  speakerTargetCombo_->blockSignals(false);
  updatingSpeakerTargetUi_ = false;
}

void AudioPage::OnSourceChanged(int /*index*/) {
  if (updatingSourceUi_)
    return;
  UpdatePortControlsForSelectedSource(true);
}

void AudioPage::OnSpeakerTargetChanged(int /*index*/) {
  if (updatingSpeakerTargetUi_)
    return;
  if (!daemonAiSupported_ || !speakerTargetCombo_ ||
      !speakerTargetCombo_->isEnabled()) {
    return;
  }

  const std::string target =
      speakerTargetCombo_->currentData().toString().toStdString();
  if (target.empty())
    return;

  QJsonObject patch;
  patch.insert("speaker_target_sink", QString::fromStdString(target));
  const auto json = QJsonDocument(patch).toJson(QJsonDocument::Compact);

  std::string out;
  QString err;
  if (!DaemonRequest(std::string("SET_AUDIO_CONFIG ") + json.toStdString(),
                     &out, &err)) {
    if (statusText_) {
      SetPlainTextPreservingScroll(
          statusText_,
          QStringLiteral("Speaker output save failed:\n%1").arg(err));
    }
    ShowError("Audio",
              QStringLiteral("Speaker output was not changed.\n\nOpen Support "
                             "for technical details."));
    SyncSpeakerTargetSelectionFromDaemon(daemonSpeakerTarget_);
    return;
  }

  emit StatusRefreshRequested();
}

void AudioPage::UpdatePortControlsForSelectedSource(bool pushDaemon) {
  if (!sourceCombo_)
    return;
  if (portCombo_) {
    portCombo_->clear();
    portCombo_->setEnabled(false);
  }

  if (!sourceCombo_->isEnabled())
    return;

  const std::string srcName =
      sourceCombo_->currentData().toString().toStdString();
  if (srcName.empty())
    return;
  if (srcName == kAutoPulseSource) {
    if (pushDaemon)
      PushDaemonSourceSelection();
    return;
  }

  const studiocast::audio::pulse::PactlSourceInfo *info = nullptr;
  for (const auto &s : cachedSources_) {
    if (s.name == srcName) {
      info = &s;
      break;
    }
  }
  if (!info)
    return;

  if (!portCombo_) {
    // Port selection is an advanced/legacy control.
    if (pushDaemon)
      PushDaemonSourceSelection();
    return;
  }

  if (info->ports.empty()) {
    // Many sources will have no explicit ports; that's fine.
    portCombo_->setEnabled(false);
    if (pushDaemon)
      PushDaemonSourceSelection();
    return;
  }

  portCombo_->setEnabled(true);

  int activeIdx = -1;
  int firstAvailable = -1;

  for (std::size_t i = 0; i < info->ports.size(); ++i) {
    const int idx = static_cast<int>(i);
    const auto &p = info->ports[i];

    std::string label = p.description.empty() ? p.name : p.description;
    if (!p.available)
      label += " (unavailable)";

    portCombo_->addItem(QString::fromStdString(label),
                        QVariant(QString::fromStdString(p.name)));

    if (!info->active_port.empty() && p.name == info->active_port)
      activeIdx = idx;
    if (firstAvailable < 0 && p.available)
      firstAvailable = idx;
  }

  if (activeIdx >= 0)
    portCombo_->setCurrentIndex(activeIdx);
  else if (firstAvailable >= 0)
    portCombo_->setCurrentIndex(firstAvailable);
  else
    portCombo_->setCurrentIndex(0);

  if (pushDaemon)
    PushDaemonSourceSelection();
}

void AudioPage::SyncSourceSelectionFromDaemon(const QString &source) {
  QString wanted = source.trimmed();
  if (wanted.isEmpty())
    wanted = QString::fromLatin1(kAutoPulseSource);
  daemonSource_ = wanted;

  if (!sourceCombo_ || !sourceCombo_->isEnabled())
    return;

  updatingSourceUi_ = true;
  sourceCombo_->blockSignals(true);
  int idx = sourceCombo_->findData(wanted);
  if (idx < 0 && wanted != QString::fromLatin1(kAutoPulseSource)) {
    for (int i = sourceCombo_->count() - 1; i >= 0; --i) {
      if (sourceCombo_->itemText(i).startsWith(QStringLiteral("<missing:"))) {
        sourceCombo_->removeItem(i);
      }
    }
    idx = sourceCombo_->findData(wanted);
    if (idx < 0) {
      const int insertAt = std::min(1, sourceCombo_->count());
      sourceCombo_->insertItem(insertAt,
                               QStringLiteral("<missing: %1>").arg(wanted),
                               QVariant(wanted));
      idx = insertAt;
    }
  }

  if (idx < 0)
    idx = 0;

  sourceCombo_->setCurrentIndex(idx);
  sourceCombo_->blockSignals(false);
  updatingSourceUi_ = false;
  UpdatePortControlsForSelectedSource(false);
}

void AudioPage::SyncSpeakerTargetSelectionFromDaemon(const QString &target) {
  QString wanted = target.trimmed();
  if (wanted.isEmpty())
    wanted = QString::fromLatin1(kAutoPulseSink);
  daemonSpeakerTarget_ = wanted;

  if (!speakerTargetCombo_ || !speakerTargetCombo_->isEnabled())
    return;

  updatingSpeakerTargetUi_ = true;
  speakerTargetCombo_->blockSignals(true);
  int idx = speakerTargetCombo_->findData(wanted);
  if (idx < 0 && wanted != QString::fromLatin1(kAutoPulseSink)) {
    for (int i = speakerTargetCombo_->count() - 1; i >= 0; --i) {
      if (speakerTargetCombo_->itemText(i).startsWith(
              QStringLiteral("<disconnected:"))) {
        speakerTargetCombo_->removeItem(i);
      }
    }
    idx = speakerTargetCombo_->findData(wanted);
    if (idx < 0) {
      const int insertAt = std::min(1, speakerTargetCombo_->count());
      speakerTargetCombo_->insertItem(
          insertAt, QStringLiteral("<disconnected: %1>").arg(wanted),
          QVariant(wanted));
      idx = insertAt;
    }
  }

  if (idx < 0)
    idx = 0;

  speakerTargetCombo_->setCurrentIndex(idx);
  speakerTargetCombo_->blockSignals(false);
  updatingSpeakerTargetUi_ = false;
}

void AudioPage::RefreshStatus() {
  RefreshStatusFromCachedDaemon(/*forceControlResync=*/false);
}

void AudioPage::RefreshStatusFromCachedDaemon(bool forceControlResync) {
  ApplyCachedDaemonAudioStatus(forceControlResync);

  if (statusText_) {
    QString text;
    if (!daemonStatusText_.trimmed().isEmpty()) {
      text += "Daemon audio status:\n";
      text += daemonStatusText_.trimmed();
    } else if (!daemonStatusDetail_.trimmed().isEmpty()) {
      text += daemonStatusDetail_.trimmed();
    } else {
      text += "Daemon audio status has not been read.";
    }
    if (!daemonLastStatusJson_.trimmed().isEmpty()) {
      text += "\n\n---\nRaw daemon status:\n";
      text += daemonLastStatusJson_;
    }
    SetPlainTextPreservingScroll(statusText_, text);
  }

  UpdateReleaseControlsFromCachedStatus();
}

void AudioPage::UpdateReleaseControlsFromCachedStatus() {
  const bool daemonOk = daemonAiSupported_;
  constexpr bool debugPactlOk = false;

  const QString speakerAction = daemonSpeakersAction_.trimmed();

  if (createBtn_) {
    createBtn_->setEnabled(
        daemonOk || (!daemonOk && debugPactlOk));
    createBtn_->setToolTip(
        daemonOk ? QString()
                 : QStringLiteral(
                       "Debug fallback requires local PulseAudio access."));
  }
  if (destroyBtn_) {
    destroyBtn_->setEnabled(
        (daemonOk && daemonMicVirtualDevicePresent_) ||
        (!daemonOk && debugPactlOk));
    destroyBtn_->setToolTip(
        daemonOk ? QString()
                 : QStringLiteral(
                       "Debug fallback requires local PulseAudio access."));
  }

  if (startBtn_) {
    startBtn_->setEnabled(debugPactlOk);
    startBtn_->setToolTip(
        debugPactlOk ? QString()
                     : QStringLiteral(
                           "Legacy loopback is a debug-only PulseAudio path."));
  }
  if (stopBtn_) {
    stopBtn_->setEnabled(debugPactlOk);
    stopBtn_->setToolTip(
        debugPactlOk ? QString()
                     : QStringLiteral(
                           "Legacy loopback is a debug-only PulseAudio path."));
  }

  const bool speakerCanStart =
      daemonOk &&
      (speakerAction.isEmpty() ||
       speakerAction == QStringLiteral("start_routing") ||
       speakerAction == QStringLiteral("create_virtual_speakers") ||
       speakerAction == QStringLiteral("retry_routing") ||
       speakerAction == QStringLiteral("wait_for_app") ||
       speakerAction == QStringLiteral("choose_speaker_output"));
  const bool speakerCanStop =
      daemonOk &&
      (speakerAction == QStringLiteral("stop_routing") ||
       daemonSpeakersRoutingActive_ ||
       (!daemonSpeakersRouteMode_.trimmed().isEmpty() &&
        daemonSpeakersRouteMode_.trimmed().toLower() !=
            QStringLiteral("off")));

  if (enableSpeakersBtn_)
    enableSpeakersBtn_->setEnabled(speakerCanStart);
  if (stopSpeakersBtn_)
    stopSpeakersBtn_->setEnabled(speakerCanStop);
  if (destroySpeakersBtn_)
    destroySpeakersBtn_->setEnabled(daemonOk &&
                                    daemonSpeakersVirtualDevicePresent_);
}

void AudioPage::ScheduleDaemonAudioConfigWrite() {
  if (!audioWriteDebounceTimer_)
    return;
  audioWriteGuard_.MarkPending();
  audioWriteDebounceTimer_->start();
}

void AudioPage::SetAiControlsEnabled(bool enabled, const QString &reason) {
  daemonAiSupported_ = enabled;
  daemonAiDisableReason_ = reason;

  if (engineCombo_)
    engineCombo_->setEnabled(enabled);

  // Microphone open-source controls
  if (openAudioModelCombo_)
    openAudioModelCombo_->setEnabled(enabled);
  if (openAudioModelPathEdit_)
    openAudioModelPathEdit_->setEnabled(enabled);
  if (browseOpenAudioModelBtn_)
    browseOpenAudioModelBtn_->setEnabled(enabled);

  // Speaker open-source controls
  if (speakerOpenAudioModelCombo_)
    speakerOpenAudioModelCombo_->setEnabled(enabled);
  if (speakerOpenAudioModelPathEdit_)
    speakerOpenAudioModelPathEdit_->setEnabled(enabled);
  if (speakerBrowseOpenAudioModelBtn_)
    speakerBrowseOpenAudioModelBtn_->setEnabled(enabled);

  // Microphone controls
  if (sourceCombo_) {
    const bool pactlUnavailable =
        sourceCombo_->count() > 0 &&
        sourceCombo_->itemText(0) ==
            QString::fromLatin1(kAudioControlsUnavailable);
    sourceCombo_->setEnabled(enabled && !pactlUnavailable);
  }
  if (micEffectCombo_)
    micEffectCombo_->setEnabled(enabled);
  if (strengthSlider_)
    strengthSlider_->setEnabled(enabled);
  if (strengthValueLabel_)
    strengthValueLabel_->setEnabled(enabled);

  // Speaker controls
  if (speakerTargetCombo_) {
    const bool pactlUnavailable = speakerTargetCombo_->count() > 0 &&
                                  speakerTargetCombo_->itemText(0) ==
                                      QString::fromLatin1(
                                          kAudioControlsUnavailable);
    speakerTargetCombo_->setEnabled(enabled && !pactlUnavailable);
  }
  if (speakerEffectCombo_)
    speakerEffectCombo_->setEnabled(enabled);
  if (speakerStrengthSlider_)
    speakerStrengthSlider_->setEnabled(enabled);
  if (speakerStrengthValueLabel_)
    speakerStrengthValueLabel_->setEnabled(enabled);

  if (aiBanner_) {
    aiBanner_->setVisible(!enabled && !reason.isEmpty());
    aiBanner_->setText(reason);
  }

  if (!enabled && micStateLabel_) {
    SetMicStatusSummary(QStringLiteral("Service unavailable"), reason,
                        QStringLiteral("error"));
    if (micSourceStatusLabel_)
      micSourceStatusLabel_->setVisible(false);
    if (openAudioInstallHintsBtn_)
      openAudioInstallHintsBtn_->setEnabled(false);
  }

  if (!enabled && speakerRouteStateLabel_) {
    SetSpeakerRouteSummary(QStringLiteral("Service unavailable"), reason,
                           QStringLiteral("error"),
                           QStringLiteral("Unknown"),
                           QStringLiteral("Unknown"));
    if (speakerTargetStatusLabel_) {
      speakerTargetStatusLabel_->setText(reason);
      speakerTargetStatusLabel_->setVisible(!reason.trimmed().isEmpty());
    }
    if (openAudioInstallHintsBtn_)
      openAudioInstallHintsBtn_->setEnabled(false);
  }

  // Avoid stacked banners: when we show a warning, hide the info note.
  if (aiInfoBanner_ && (!enabled || (aiBanner_ && aiBanner_->isVisible()))) {
    aiInfoBanner_->setVisible(false);
    aiInfoBanner_->setToolTip(QString());
  }

  // Disable strength sliders when effect is Off (Broadcast-like).
  if (enabled) {
    if (micEffectCombo_ && strengthSlider_) {
      const bool active = micEffectCombo_->currentData().toString() != "off";
      strengthSlider_->setEnabled(active);
      if (strengthValueLabel_)
        strengthValueLabel_->setEnabled(active);
    }
    if (speakerEffectCombo_ && speakerStrengthSlider_) {
      const bool active =
          speakerEffectCombo_->currentData().toString() != "off";
      speakerStrengthSlider_->setEnabled(active);
      if (speakerStrengthValueLabel_)
        speakerStrengthValueLabel_->setEnabled(active);
    }
  }

  UpdateEngineUiVisibility();
}

void AudioPage::UpdateEngineUiVisibility() {
  const QString eng = engineCombo_ ? engineCombo_->currentData().toString()
                                   : QStringLiteral("auto");

  // AUTO behaves like Maxine when Maxine is actually active; only surface
  // Open Audio controls when Open Audio is explicitly selected, or when AUTO
  // has fallen back to Open Audio.
  bool activeIsOpen = false;
  if (engineActiveValue_) {
    QString active = engineActiveValue_->toolTip();
    if (active.isEmpty())
      active = engineActiveValue_->text();
    active = active.trimmed().toLower();
    activeIsOpen =
        (active == "open_source") || (active == "open_audio") ||
        (active == "open_cuda") || (active == "open_video") ||
        (active == "open source") || active.contains("open_source") ||
        active.contains("open_audio") || active.contains("open_cuda") ||
        active.contains("open_video") || active.contains("open source");
  }

  const bool showOpen =
      (eng == "open_source") || (eng == "auto" && activeIsOpen);
  const bool showMicOpenDetails =
      mode_ == AudioPageMode::Microphone ? true : showOpen;
  const bool speakerHasOpenConfig =
      (speakerOpenAudioModelCombo_ &&
       !speakerOpenAudioModelCombo_->currentData().toString().trimmed().isEmpty()) ||
      (speakerOpenAudioModelPathEdit_ &&
       !speakerOpenAudioModelPathEdit_->text().trimmed().isEmpty());
  const bool showSpeakerOpenDetails = showOpen || speakerHasOpenConfig;

  // Microphone model controls
  if (openAudioModelLabel_)
    openAudioModelLabel_->setVisible(showMicOpenDetails);
  if (openAudioModelCombo_)
    openAudioModelCombo_->setVisible(showMicOpenDetails);
  if (openAudioModelPathLabel_)
    openAudioModelPathLabel_->setVisible(showMicOpenDetails);
  if (openAudioModelPathEdit_)
    openAudioModelPathEdit_->setVisible(showMicOpenDetails);
  if (browseOpenAudioModelBtn_)
    browseOpenAudioModelBtn_->setVisible(showMicOpenDetails);

  // Speaker model controls
  if (speakerOpenAudioModelLabel_)
    speakerOpenAudioModelLabel_->setVisible(showSpeakerOpenDetails);
  if (speakerOpenAudioModelCombo_)
    speakerOpenAudioModelCombo_->setVisible(showSpeakerOpenDetails);
  if (speakerOpenAudioModelPathLabel_)
    speakerOpenAudioModelPathLabel_->setVisible(showSpeakerOpenDetails);
  if (speakerOpenAudioModelPathEdit_)
    speakerOpenAudioModelPathEdit_->setVisible(showSpeakerOpenDetails);
  if (speakerBrowseOpenAudioModelBtn_)
    speakerBrowseOpenAudioModelBtn_->setVisible(showSpeakerOpenDetails);

  // Microphone keeps install hints in Details. Speakers show them when Open
  // Audio is relevant, or when an explicit speaker model/path is configured.
  if (openAudioInstallHintsBtn_)
    openAudioInstallHintsBtn_->setVisible(mode_ == AudioPageMode::Microphone
                                              ? true
                                              : showSpeakerOpenDetails);
}

void AudioPage::ApplyCachedDaemonAudioStatus(bool forceControlResync) {
  daemonStatusText_.clear();
  daemonSpeakersRoutingActive_ = false;
  daemonSpeakersRouteMode_.clear();

  if (daemonLastStatusJson_.trimmed().isEmpty()) {
    SetAiControlsEnabled(
        false,
        daemonStatusDetail_.trimmed().isEmpty()
            ? QStringLiteral("StudioCast background service is unavailable. "
                             "Open Support for technical details.")
            : daemonStatusDetail_);
    daemonStatusText_ = "daemon_unavailable: " + daemonStatusDetail_;
    return;
  }

  QJsonObject root;
  QString jerr;
  if (!ParseJsonObject(daemonLastStatusJson_.toStdString(), &root, &jerr)) {
    SetAiControlsEnabled(
        false,
        QStringLiteral("StudioCast received an unreadable status update. Open "
                       "Support for technical details."));
    daemonStatusText_ = "invalid_json";
    if (!jerr.trimmed().isEmpty())
      daemonStatusText_ += ": " + jerr;
    return;
  }

  // Keep Maxine diagnostics for user guidance, but do not hard-disable the UI.
  // The runtime resolver handles Maxine/OpenAudio selection and fallback.
  QString maxineDiag;
  {
    const auto maxine = root.value("maxine").toObject();
    const bool supported = maxine.value("supported").toBool(false);
    const QString summary = maxine.value("summary").toString();
    const QString blockedReason = maxine.value("blocked_reason").toString();
    const auto blockedDetails = maxine.value("blocked_details").toArray();

    bool gpuOk = true;
    if (maxine.contains("gpu")) {
      gpuOk = maxine.value("gpu").toObject().value("ok").toBool(true);
    }

    bool afxOk = true;
    if (maxine.contains("afx")) {
      afxOk = maxine.value("afx").toObject().value("ok").toBool(true);
    } else if (maxine.contains("components")) {
      afxOk = maxine.value("components")
                  .toObject()
                  .value("afx")
                  .toObject()
                  .value("found")
                  .toBool(true);
    }

    if (!supported || !gpuOk || !afxOk) {
      maxineDiag = "maxine_unavailable: true\n";
      if (!summary.isEmpty())
        maxineDiag += "maxine_summary: " + summary + "\n";
      const auto english = FormatMaxineReasonCode(blockedReason);
      if (!english.isEmpty())
        maxineDiag += "maxine_reason: " + english + "\n";
      if (!blockedDetails.isEmpty()) {
        maxineDiag += "maxine_details:\n";
        for (const auto &v : blockedDetails) {
          maxineDiag += "- " + v.toString() + "\n";
        }
      }
    }
  }

  SetAiControlsEnabled(true, "");

  // Transitional compatibility: routine delivery is centralized through
  // DaemonStatusSnapshot, but this page still renders several legacy fields
  // that are not all promoted to typed members. Parse only the cached shared
  // snapshot payload; do not issue page-local status IPC here.
  const auto audio = root.value("audio").toObject();
  const bool audioEnabled = audio.value("enabled").toBool(false);
  const auto microphoneEndpoint = audio.value("microphone").toObject();
  daemonMicrophoneAction_ =
      FirstLine(microphoneEndpoint.value("action").toString());
  SyncSourceSelectionFromDaemon(audio.value("source").toString());
  const QString sourceResolved = audio.value("source_resolved").toString();
  const QString sourceErr = audio.value("source_error").toString();
  const auto sourceWarnings = audio.value("source_warnings").toArray();
  const QString micMode = audio.value("mic_mode").toString();
  const auto pipeline = audio.value("pipeline").toObject();
  const bool running = pipeline.value("running").toBool(false);
  const bool starting = pipeline.value("starting").toBool(false);
  const bool activeNeeded = pipeline.value("active_needed").toBool(false);
  const QString pipelineState =
      pipeline.value("state").toString().trimmed().toLower();
  const QString pipelineIdleReason =
      pipeline.value("idle_reason").toString().trimmed();
  const QString lastErr = pipeline.value("last_error").toString();
  const QString backendActive = pipeline.value("backend_active").toString();
  const QString effectsNote = pipeline.value("effects_note").toString();
  const auto micOpenAudioRuntime =
      pipeline.value("open_audio_runtime").toObject();
  bool openAudioCpuFallback =
      micOpenAudioRuntime.value("using_cpu_fallback").toBool(false);
  bool openAudioDisabled =
      micOpenAudioRuntime.value("disabled").toBool(false);
  QString openAudioRuntimeWarning =
      micOpenAudioRuntime.value("last_runtime_warning").toString().trimmed();
  const bool micConsumerPresent =
      audio.value("mic_consumer_present").toBool(false);
  const int micConsumerCount = audio.value("mic_consumer_count").toInt(0);
  const QString micConsumerError =
      audio.value("mic_consumer_error").toString().trimmed();

  if (mode_ == AudioPageMode::Microphone) {
    const bool micPresent = audio.value("mic_present").toBool(
        audio.value("create_virtual_mic").toBool(false));
    daemonMicVirtualDevicePresent_ = micPresent;

    QString state = QStringLiteral("Available");
    QString detail =
        QStringLiteral("StudioCast Microphone is available to other apps.");
    QString status = QStringLiteral("good");

    if (!sourceErr.trimmed().isEmpty()) {
      state = QStringLiteral("No microphone input is selected");
      detail = QStringLiteral(
          "Choose a microphone input, or open Support for technical details.");
      status = QStringLiteral("warning");
    } else if (!micPresent) {
      state = QStringLiteral("StudioCast Microphone is missing");
      detail = QStringLiteral(
          "The StudioCast Microphone virtual device is not present.");
      status = QStringLiteral("warning");
    } else if (!lastErr.trimmed().isEmpty()) {
      state = QStringLiteral("Needs attention");
      detail = QStringLiteral("Microphone processing stopped. Open Support "
                              "for technical details.");
      status = QStringLiteral("error");
    } else if (running || pipelineState == QStringLiteral("running")) {
      state = QStringLiteral("Processing active");
      detail = backendActive.trimmed().isEmpty()
                   ? QStringLiteral("Microphone cleanup is running.")
                   : QStringLiteral("Microphone cleanup is running with %1.")
                         .arg(FriendlyBackendLabel(backendActive));
    } else if (starting || pipelineState == QStringLiteral("starting")) {
      state = QStringLiteral("Starting");
      detail = QStringLiteral("Microphone cleanup is starting.");
      status = QStringLiteral("warning");
    } else if (!micConsumerError.isEmpty() && audioEnabled) {
      state = QStringLiteral("Consumer detection error");
      detail = QStringLiteral(
          "StudioCast cannot tell when other apps are using the microphone. "
          "Open Support for technical details.");
      status = QStringLiteral("error");
    } else if (audioEnabled &&
               pipelineState == QStringLiteral("idle_no_consumer")) {
      state = QStringLiteral("Waiting for app");
      detail = pipelineIdleReason.isEmpty()
                   ? QStringLiteral("Ready. Processing starts when an app uses "
                                    "StudioCast Microphone.")
                   : pipelineIdleReason;
    } else if (!audioEnabled || pipelineState == QStringLiteral("disabled")) {
      state = QStringLiteral("Off");
      detail = QStringLiteral(
          "Microphone processing is disabled by the current backend setting.");
      status = QStringLiteral("warning");
    } else {
      // Older daemon payloads did not expose enough state to prove
      // idle/no-consumer, so keep this generic.
      state = QStringLiteral("Not processing");
      detail = QStringLiteral("Microphone status is reported, but processing "
                              "is not active.");
    }
    SetMicStatusSummary(state, detail, status);

    if (micSourceStatusLabel_) {
      QStringList sourceLines;
      if (!sourceResolved.trimmed().isEmpty()) {
        sourceLines << QStringLiteral("Using input: %1")
                           .arg(sourceResolved.trimmed());
      }
      if (micConsumerCount > 0 || micConsumerPresent) {
        sourceLines << QStringLiteral("Apps using microphone: %1")
                           .arg(micConsumerCount);
      }
      bool hasWarning = false;
      for (const auto &v : sourceWarnings) {
        const QString warning = v.toString().trimmed();
        if (!warning.isEmpty()) {
          hasWarning = true;
          break;
        }
      }
      if (hasWarning)
        sourceLines << QStringLiteral(
            "The selected microphone input needs attention. Open Support for "
            "technical details.");
      micSourceStatusLabel_->setText(sourceLines.join(QStringLiteral("\n")));
      micSourceStatusLabel_->setVisible(!sourceLines.isEmpty());
    }
  }

  // Tab-specific summary: in Speakers mode, prefer the speakers pipeline
  // backend and note rather than the microphone pipeline.
  QString backendForUi = backendActive;
  QString noteForUi = effectsNote;
  if (mode_ == AudioPageMode::Speakers) {
    backendForUi.clear();
    noteForUi.clear();

    if (audio.contains("speakers")) {
      const auto spk = audio.value("speakers").toObject();
      const bool spkRouting = spk.value("routing_active").toBool(false);
      const QString spkRouteMode = spk.value("route_mode").toString();
      const QString spkBackend = spk.value("backend_active").toString();
      const QString spkNote = spk.value("effects_note").toString();
      const auto spkOpenAudioRuntime =
          spk.value("open_audio_runtime").toObject();
      if (!spkOpenAudioRuntime.isEmpty()) {
        openAudioCpuFallback =
            spkOpenAudioRuntime.value("using_cpu_fallback").toBool(false);
        openAudioDisabled =
            spkOpenAudioRuntime.value("disabled").toBool(false);
        openAudioRuntimeWarning =
            spkOpenAudioRuntime.value("last_runtime_warning")
                .toString()
                .trimmed();
      }

      QString spkPipeBackend;
      QString spkPipeNote;
      if (spk.contains("pipeline")) {
        const auto spkPipe = spk.value("pipeline").toObject();
        spkPipeBackend = spkPipe.value("backend_active").toString();
        spkPipeNote = spkPipe.value("effects_note").toString();
        const auto spkPipeOpenAudioRuntime =
            spkPipe.value("open_audio_runtime").toObject();
        if (!spkPipeOpenAudioRuntime.isEmpty()) {
          openAudioCpuFallback =
              spkPipeOpenAudioRuntime.value("using_cpu_fallback").toBool(false);
          openAudioDisabled =
              spkPipeOpenAudioRuntime.value("disabled").toBool(false);
          openAudioRuntimeWarning =
              spkPipeOpenAudioRuntime.value("last_runtime_warning")
                  .toString()
                  .trimmed();
        }
      }

      backendForUi = spkBackend.isEmpty() ? spkPipeBackend : spkBackend;
      noteForUi = spkNote.trimmed().isEmpty() ? spkPipeNote : spkNote;

      const QString rm = spkRouteMode.trimmed().toLower();
      if (rm == QStringLiteral("loopback")) {
        backendForUi = QStringLiteral("loopback");
        if (noteForUi.trimmed().isEmpty() && spkRouting) {
          noteForUi =
              QStringLiteral("Speakers routed via loopback (pass-through).");
        }
      } else if (rm == QStringLiteral("off")) {
        backendForUi = QStringLiteral("off");
        noteForUi.clear();
      }
    }
  }

  if (engineActiveValue_) {
    const QString backendLower = backendForUi.trimmed().toLower();
    QString activeLabel = FriendlyBackendLabel(backendForUi);
    if (backendLower == QStringLiteral("open_source") ||
        backendLower == QStringLiteral("open_audio")) {
      if (openAudioDisabled) {
        activeLabel = QStringLiteral("Open Audio disabled");
      } else if (openAudioCpuFallback) {
        activeLabel = QStringLiteral("Open Audio (CPU fallback)");
      }
    }
    engineActiveValue_->setText(activeLabel);
    engineActiveValue_->setToolTip(backendForUi);
  }

  if (aiInfoBanner_) {
    QStringList notes;
    if (!openAudioRuntimeWarning.isEmpty())
      notes << openAudioRuntimeWarning;
    if (!noteForUi.trimmed().isEmpty())
      notes << noteForUi.trimmed();
    const QString full = notes.join(QStringLiteral("\n")).trimmed();
    const QString first = FirstLine(full);
    aiInfoBanner_->setVisible(!first.isEmpty());
    aiInfoBanner_->setText(first);
    aiInfoBanner_->setToolTip(full);
  }

  const QString pipelineLabel =
      pipelineState.isEmpty()
          ? QString(running ? "running" : (starting ? "starting" : "stopped"))
          : pipelineState;
  daemonStatusText_ =
      QString("enabled=%1\nmic_mode=%2\npipeline=%3 active_needed=%4\n")
          .arg(audioEnabled ? "true" : "false")
          .arg(micMode.isEmpty() ? "(none)" : micMode)
          .arg(pipelineLabel)
          .arg(activeNeeded ? "true" : "false");

  if (pipeline.contains("gpu")) {
    const auto gpu = pipeline.value("gpu").toObject();
    const int idx = gpu.value("index").toInt(-1);
    const QString name = gpu.value("name").toString();
    const QString cc = gpu.value("compute_cap").toString();
    if (idx >= 0 || !name.isEmpty()) {
      daemonStatusText_ += QString("gpu: #%1 %2%3\n")
                               .arg(idx)
                               .arg(name.isEmpty() ? "(unknown)" : name)
                               .arg(cc.isEmpty() ? "" : (" (cc " + cc + ")"));
    }
  }
  if (!lastErr.isEmpty())
    daemonStatusText_ += "last_error: " + lastErr + "\n";
  if (!sourceResolved.isEmpty())
    daemonStatusText_ += "source_resolved: " + sourceResolved + "\n";
  if (!sourceErr.isEmpty())
    daemonStatusText_ += "source_error: " + sourceErr + "\n";
  daemonStatusText_ += QString("mic_consumer_present=%1 mic_consumer_count=%2\n")
                           .arg(micConsumerPresent ? "true" : "false")
                           .arg(micConsumerCount);
  if (!micConsumerError.isEmpty())
    daemonStatusText_ += "mic_consumer_error: " + micConsumerError + "\n";
  for (const auto &v : sourceWarnings) {
    const QString warning = v.toString();
    if (!warning.isEmpty())
      daemonStatusText_ += "source_warning: " + warning + "\n";
  }

  // Speakers status (if present).
  if (audio.contains("speakers")) {
    const auto spk = audio.value("speakers").toObject();

    daemonSpeakersRoutingActive_ = spk.value("routing_active").toBool(false);
    daemonSpeakersAction_ = FirstLine(spk.value("action").toString());
    daemonSpeakersRouteMode_ = spk.value("route_mode").toString();

    const bool spkPresent = spk.value("present").toBool(
        audio.value("create_virtual_speakers").toBool(false));
    daemonSpeakersVirtualDevicePresent_ = spkPresent;
    const bool spkRouting = spk.value("routing_active").toBool(false);
    const QString spkRouteMode = spk.value("route_mode").toString();
    const QString spkConfiguredTarget = spk.value("target_sink").toString();
    const QString spkResolvedTarget =
        spk.value("target_sink_resolved").toString();
    const QString spkTargetErr = spk.value("target_sink_error").toString();
    const QString spkTarget = spk.value("target_sink_active").toString();
    const QString spkErr = spk.value("last_error").toString();

    const bool spkPipeRunning = spk.value("pipeline_running").toBool(false);
    const bool spkPipeStarting = spk.value("pipeline_starting").toBool(false);
    const bool spkPipeActiveNeeded =
        spk.value("pipeline_active_needed").toBool(false);
    const QString spkPipeState =
        spk.value("pipeline_state").toString().trimmed().toLower();
    const QString spkPipeIdleReason =
        spk.value("pipeline_idle_reason").toString().trimmed();
    const bool spkConsumerPresent =
        spk.value("consumer_present").toBool(false);
    const int spkConsumerCount = spk.value("consumer_count").toInt(0);
    const QString spkConsumerError =
        spk.value("consumer_error").toString().trimmed();
    const QString spkBackend = spk.value("backend_active").toString();
    const QString spkNote = spk.value("effects_note").toString();
    const QString spkPipeErr = spk.value("pipeline_last_error").toString();

    if (mode_ == AudioPageMode::Speakers) {
      const QString rm = spkRouteMode.trimmed().toLower();
      const QString routeModeLabel = FriendlySpeakerRouteMode(spkRouteMode);
      const QString targetLabel =
          FriendlySpeakerTarget(spkConfiguredTarget, spkResolvedTarget,
                                spkTarget);
      const QString routeError =
          spkErr.trimmed().isEmpty() ? spkPipeErr.trimmed()
                                     : spkErr.trimmed();
      const QString backendLabel = FriendlyBackendLabel(
          spkBackend.trimmed().isEmpty() ? backendForUi : spkBackend);

      QString state = QStringLiteral("Off");
      QString detail = QStringLiteral("Speaker routing is stopped.");
      QString status = QStringLiteral("warning");

      if (!routeError.isEmpty()) {
        state = QStringLiteral("Needs attention");
        detail = QStringLiteral("Speaker routing stopped. Open Support for "
                                "technical details.");
        status = QStringLiteral("error");
      } else if (!spkTargetErr.trimmed().isEmpty()) {
        state = QStringLiteral("No speaker output is selected");
        detail = QStringLiteral(
            "Choose a speaker output, or open Support for technical details.");
        status = QStringLiteral("warning");
      } else if (!spkPresent) {
        state = QStringLiteral("StudioCast Speakers are missing");
        detail = QStringLiteral(
            "The StudioCast Speakers virtual device is not present.");
        status = QStringLiteral("warning");
      } else if (rm == QStringLiteral("loopback")) {
        state = spkRouting ? QStringLiteral("Pass-through active")
                           : QStringLiteral("Pass-through stopped");
        detail = spkRouting
                     ? QStringLiteral("Loopback/pass-through routing is active.")
                     : QStringLiteral("Loopback/pass-through routing is "
                                      "configured but stopped.");
        status = spkRouting ? QStringLiteral("good")
                            : QStringLiteral("warning");
      } else if (rm == QStringLiteral("pipeline")) {
        if (spkPipeRunning || spkRouting ||
            spkPipeState == QStringLiteral("running")) {
          state = QStringLiteral("Processed pipeline active");
          detail = backendLabel == QStringLiteral("—")
                       ? QStringLiteral("Speaker cleanup is processing audio.")
                       : QStringLiteral("Speaker cleanup is processing audio "
                                        "with %1.")
                             .arg(backendLabel);
          status = QStringLiteral("good");
        } else if (spkPipeStarting ||
                   spkPipeState == QStringLiteral("starting")) {
          state = QStringLiteral("Processed pipeline starting");
          detail = QStringLiteral("Speaker cleanup is starting.");
          status = QStringLiteral("warning");
        } else if (!spkConsumerError.isEmpty()) {
          state = QStringLiteral("Consumer detection error");
          detail = QStringLiteral(
              "StudioCast cannot tell when other apps are using the speakers. "
              "Open Support for technical details.");
          status = QStringLiteral("error");
        } else if (spkPipeState == QStringLiteral("idle_no_consumer")) {
          state = QStringLiteral("Waiting for app");
          detail = spkPipeIdleReason.isEmpty()
                       ? QStringLiteral("Ready. Processing starts when an app "
                                        "uses StudioCast Speakers.")
                       : spkPipeIdleReason;
          status = QStringLiteral("good");
        } else if (spkPipeState.isEmpty()) {
          // Older daemon payloads did not expose enough state to prove
          // idle/no-consumer, so keep this generic.
          state = QStringLiteral("Processed pipeline stopped");
          detail = QStringLiteral("Processed speaker routing is not active.");
          status = QStringLiteral("warning");
        } else {
          state = QStringLiteral("Processed pipeline stopped");
          detail = QStringLiteral("Processed speaker routing is stopped.");
          status = QStringLiteral("warning");
        }
      } else if (spkRouting) {
        state = QStringLiteral("Routing active");
        detail = QStringLiteral("Speaker routing is active.");
        status = QStringLiteral("good");
      }

      SetSpeakerRouteSummary(state, detail, status, routeModeLabel,
                             targetLabel);

      if (speakerTargetStatusLabel_) {
        QStringList targetLines;
        if (!spkConfiguredTarget.trimmed().isEmpty() &&
            spkConfiguredTarget.trimmed() !=
                QString::fromLatin1(kAutoPulseSink)) {
          targetLines << QStringLiteral("Configured output: %1")
                             .arg(spkConfiguredTarget.trimmed());
        }
        if (!spkResolvedTarget.trimmed().isEmpty()) {
          targetLines << QStringLiteral("Using output: %1")
                             .arg(spkResolvedTarget.trimmed());
        }
        if (!spkTargetErr.trimmed().isEmpty())
          targetLines << QStringLiteral(
              "The selected speaker output needs attention. Open Support for "
              "technical details.");
        if (spkConsumerCount > 0 || spkConsumerPresent) {
          targetLines << QStringLiteral("Apps using speakers: %1")
                             .arg(spkConsumerCount);
        }
        if (!targetLines.isEmpty()) {
          speakerTargetStatusLabel_->setText(
              targetLines.join(QStringLiteral("\n")));
          speakerTargetStatusLabel_->setVisible(true);
        } else {
          speakerTargetStatusLabel_->clear();
          speakerTargetStatusLabel_->setVisible(false);
        }
      }
    }

    daemonStatusText_ +=
        QString("speakers_present=%1 speakers_route=%2 route_mode=%3 "
                "pipeline_state=%4 active_needed=%5\n")
            .arg(spkPresent ? "true" : "false")
            .arg(spkRouting ? "true" : "false")
            .arg(spkRouteMode.isEmpty() ? "(none)" : spkRouteMode)
            .arg(spkPipeState.isEmpty() ? "(none)" : spkPipeState)
            .arg(spkPipeActiveNeeded ? "true" : "false");
    daemonStatusText_ +=
        QString("speakers_consumer_present=%1 speakers_consumer_count=%2\n")
            .arg(spkConsumerPresent ? "true" : "false")
            .arg(spkConsumerCount);
    if (!spkConsumerError.isEmpty())
      daemonStatusText_ +=
          "speakers_consumer_error: " + spkConsumerError + "\n";
    if (!spkConfiguredTarget.isEmpty())
      daemonStatusText_ +=
          "speakers_target_sink_configured: " + spkConfiguredTarget + "\n";
    if (!spkResolvedTarget.isEmpty())
      daemonStatusText_ +=
          "speakers_target_sink_resolved: " + spkResolvedTarget + "\n";
    if (!spkTargetErr.isEmpty())
      daemonStatusText_ += "speakers_target_sink_error: " + spkTargetErr + "\n";

    if (spk.contains("pipeline")) {
      const auto spkPipe = spk.value("pipeline").toObject();
      const QString spkPipeBackend = spkPipe.value("backend_active").toString();
      const QString spkPipeNote = spkPipe.value("effects_note").toString();
      if (!spkPipeBackend.isEmpty())
        daemonStatusText_ +=
            "speakers_pipeline_backend_active: " + spkPipeBackend + "\n";
      if (!spkPipeNote.trimmed().isEmpty())
        daemonStatusText_ +=
            "speakers_pipeline_note: " + spkPipeNote.trimmed() + "\n";
    }

    if (!spkBackend.isEmpty())
      daemonStatusText_ += "speakers_backend_active: " + spkBackend + "\n";
    if (!spkNote.trimmed().isEmpty())
      daemonStatusText_ += "speakers_note: " + spkNote.trimmed() + "\n";
    if (!spkTarget.isEmpty())
      daemonStatusText_ += "speakers_target_sink_active: " + spkTarget + "\n";
    if (!spkErr.isEmpty())
      daemonStatusText_ += "speakers_last_error: " + spkErr + "\n";
    if (!spkPipeErr.isEmpty())
      daemonStatusText_ += "speakers_pipeline_last_error: " + spkPipeErr + "\n";

    if (spk.contains("pipeline_perf")) {
      const auto perf = spk.value("pipeline_perf").toObject();
      const double avgMs = perf.value("process_ms_avg").toDouble(0.0);
      const double lastMs =
          perf.value("process_us_last").toDouble(0.0) / 1000.0;
      const double maxMs = perf.value("process_us_max").toDouble(0.0) / 1000.0;
      const int overruns = perf.value("process_overruns").toInt(0);
      const qint64 frames =
          static_cast<qint64>(perf.value("frames_processed").toDouble(0.0));
      daemonStatusText_ +=
          QString("speakers_proc_ms_avg=%1 speakers_proc_ms_last=%2 "
                  "speakers_proc_ms_max=%3 "
                  "speakers_overruns=%4 speakers_frames=%5\n")
              .arg(avgMs, 0, 'f', 3)
              .arg(lastMs, 0, 'f', 3)
              .arg(maxMs, 0, 'f', 3)
              .arg(overruns)
              .arg(frames);
    }

	    // Mirror the most user-relevant pipeline state.
	    daemonStatusText_ +=
	        QString("speakers_pipeline=%1\n")
	            .arg(spkPipeState.isEmpty()
	                     ? QString(spkPipeRunning
	                                   ? "running"
	                                   : (spkPipeStarting ? "starting" : "stopped"))
	                     : spkPipeState);

    SyncSpeakerTargetSelectionFromDaemon(spkConfiguredTarget);
  } else if (mode_ == AudioPageMode::Speakers) {
    SetSpeakerRouteSummary(
        QStringLiteral("Unknown"),
        QStringLiteral(
            "Speaker routing status is unavailable. Open Support for "
            "technical details."),
        QStringLiteral("warning"), QStringLiteral("Unknown"),
        QStringLiteral("Unknown"));
  }

  if (!backendActive.isEmpty())
    daemonStatusText_ += "backend_active: " + backendActive + "\n";
  if (!maxineDiag.isEmpty())
    daemonStatusText_ += maxineDiag;

  if (!daemonAiSupported_) {
    // Keep widget states but prevent edits.
    return;
  }

  if (!forceControlResync && !audioWriteGuard_.ShouldApplyRoutineStatus())
    return;

  // Sync UI from daemon config.
  const auto fx = audio.value("audio_effects").toObject();
  lastAudioEffectsObj_ = fx;

  // Engine preference (schema v4).
  const QString engine = fx.value("engine").toString("auto");
  if (engineCombo_) {
    engineCombo_->blockSignals(true);
    const int idx = engineCombo_->findData(engine);
    engineCombo_->setCurrentIndex(idx >= 0 ? idx : 0);
    engineCombo_->blockSignals(false);
  }

  // Open Audio diagnostics.
  openAudioStatusPresent_ = false;
  openAudioOk_ = false;
  openAudioInstallHints_.clear();

  QJsonObject openAudio;
  if (root.contains("engines")) {
    const auto engines = root.value("engines").toObject();
    openAudio = engines.value("open_audio").toObject();
  }
  if (openAudio.isEmpty()) {
    openAudio = root.value("open_audio").toObject();
  }
  if (!openAudio.isEmpty()) {
    openAudioStatusPresent_ = true;
    openAudioOk_ = openAudio.value("ok").toBool(false);
    const auto hints = openAudio.value("install_hints").toArray();
    for (const auto &v : hints) {
      const auto s = v.toString();
      if (!s.isEmpty())
        openAudioInstallHints_.push_back(s);
    }
  }

  if (openAudioStatusPresent_) {
    daemonStatusText_ += QStringLiteral("open_audio:\n");
    daemonStatusText_ +=
        QString::fromUtf8(QJsonDocument(openAudio)
                              .toJson(QJsonDocument::Indented))
            .trimmed();
    daemonStatusText_ += QStringLiteral("\n");
  }

  if (openAudioInstallHintsBtn_) {
    openAudioInstallHintsBtn_->setEnabled(openAudioStatusPresent_ &&
                                          !openAudioInstallHints_.isEmpty());
  }

  // Populate model list from Open Audio diagnostics.
  auto populateModels = [&](QComboBox *combo) {
    if (!combo)
      return;
    const QString prior = combo->currentData().toString();
    combo->blockSignals(true);
    combo->clear();
    combo->addItem("<auto>", "");
    if (!openAudioStatusPresent_) {
      combo->addItem("<Open Audio status not reported>", "");
    } else if (!openAudioOk_) {
      combo->addItem("<Open Audio unavailable>", "");
    } else {
      const auto models = openAudio.value("models").toArray();
      if (models.isEmpty()) {
        combo->addItem("<no models installed>", "");
      }
      for (const auto &mv : models) {
        const auto m = mv.toObject();
        const QString id = m.value("id").toString();
        const QString name = m.value("display_name").toString();
        if (id.isEmpty())
          continue;
        const QString label = name.isEmpty() ? id : (name + " (" + id + ")");
        combo->addItem(label, id);
      }
    }

    int restore = combo->findData(prior);
    if (restore < 0)
      restore = 0;
    combo->setCurrentIndex(restore);
    combo->blockSignals(false);
  };

  populateModels(openAudioModelCombo_);
  populateModels(speakerOpenAudioModelCombo_);

  const auto mic = fx.value("microphone").toObject();
  const bool noise = mic.value("noise_removal_enabled").toBool(false);
  const bool echo = mic.value("room_echo_removal_enabled").toBool(false);
  const bool studio = mic.value("studio_voice_enabled").toBool(false);
  const int strength = mic.value("strength").toInt(50);

  const QString micModelId = mic.value("model_id").toString();
  const QString micModelPath = mic.value("model_path").toString();

  const auto spkFx = fx.value("speaker").toObject();
  const bool spkNoise = spkFx.value("noise_removal_enabled").toBool(false);
  const bool spkEcho = spkFx.value("room_echo_removal_enabled").toBool(false);
  const int spkStrength = spkFx.value("strength").toInt(50);
  const QString spkModelId = spkFx.value("model_id").toString();
  const QString spkModelPath = spkFx.value("model_path").toString();

  updatingAiUi_ = true;

  auto setComboById = [](QComboBox *combo, const QString &id) {
    if (!combo)
      return;
    const int idx = combo->findData(id);
    combo->setCurrentIndex(idx >= 0 ? idx : 0);
  };

  auto setModelComboWithMissing = [](QComboBox *combo, const QString &id) {
    if (!combo)
      return;
    const QString want = id.trimmed();
    combo->blockSignals(true);
    for (int i = combo->count() - 1; i >= 0; --i) {
      if (combo->itemText(i).startsWith(QStringLiteral("<missing:"))) {
        combo->removeItem(i);
      }
    }
    if (want.isEmpty()) {
      combo->setCurrentIndex(0);
      combo->blockSignals(false);
      return;
    }
    int idx = combo->findData(want);
    if (idx >= 0) {
      combo->setCurrentIndex(idx);
      combo->blockSignals(false);
      return;
    }
    const int insertAt = std::min(1, combo->count());
    combo->insertItem(insertAt, QStringLiteral("<missing: %1>").arg(want),
                      want);
    combo->setCurrentIndex(insertAt);
    combo->blockSignals(false);
  };

  if (micEffectCombo_) {
    QString id = "off";
    if (studio) {
      id = "studio_voice";
    } else if (noise && echo) {
      id = "noise_echo";
    } else if (noise) {
      id = "noise";
    } else if (echo) {
      id = "echo";
    }
    setComboById(micEffectCombo_, id);
  }

  if (strengthSlider_) {
    strengthSlider_->setValue(std::max(0, std::min(100, strength)));
  }
  if (strengthValueLabel_ && strengthSlider_) {
    strengthValueLabel_->setText(QString::number(strengthSlider_->value()));
  }
  setModelComboWithMissing(openAudioModelCombo_, micModelId);
  if (openAudioModelPathEdit_) {
    openAudioModelPathEdit_->setText(micModelPath);
  }

  if (speakerEffectCombo_) {
    QString id = "off";
    if (spkNoise && spkEcho) {
      id = "noise_echo";
    } else if (spkNoise) {
      id = "noise";
    } else if (spkEcho) {
      id = "echo";
    }
    setComboById(speakerEffectCombo_, id);
  }

  if (speakerStrengthSlider_) {
    speakerStrengthSlider_->setValue(std::max(0, std::min(100, spkStrength)));
  }
  if (speakerStrengthValueLabel_ && speakerStrengthSlider_) {
    speakerStrengthValueLabel_->setText(
        QString::number(speakerStrengthSlider_->value()));
  }
  setModelComboWithMissing(speakerOpenAudioModelCombo_, spkModelId);
  if (speakerOpenAudioModelPathEdit_) {
    speakerOpenAudioModelPathEdit_->setText(spkModelPath);
  }

  updatingAiUi_ = false;

  // Disable strength sliders when effect is Off (Broadcast-like).
  if (micEffectCombo_ && strengthSlider_) {
    const bool active = daemonAiSupported_ &&
                        (micEffectCombo_->currentData().toString() != "off");
    strengthSlider_->setEnabled(active);
    if (strengthValueLabel_)
      strengthValueLabel_->setEnabled(active);
  }
  if (speakerEffectCombo_ && speakerStrengthSlider_) {
    const bool active =
        daemonAiSupported_ &&
        (speakerEffectCombo_->currentData().toString() != "off");
    speakerStrengthSlider_->setEnabled(active);
    if (speakerStrengthValueLabel_)
      speakerStrengthValueLabel_->setEnabled(active);
  }

  UpdateEngineUiVisibility();
}

void AudioPage::PushDaemonSourceSelection() {
  if (!daemonAiSupported_)
    return;
  if (!sourceCombo_)
    return;
  if (!sourceCombo_->isEnabled())
    return;

  const std::string srcName =
      sourceCombo_->currentData().toString().toStdString();
  if (srcName.empty())
    return;

  QJsonObject patch;
  patch.insert("source", QString::fromStdString(srcName));
  const auto json = QJsonDocument(patch).toJson(QJsonDocument::Compact);

  std::string out;
  QString err;
  if (!DaemonRequest(std::string("SET_AUDIO_CONFIG ") + json.toStdString(),
                     &out, &err)) {
    daemonStatusText_ += "\nfailed_to_set_source: " + err;
    if (statusText_) {
      SetPlainTextPreservingScroll(
          statusText_,
          QStringLiteral("Microphone source save failed:\n%1").arg(err));
    }
    ShowError("Audio",
              QStringLiteral("Microphone input was not changed.\n\nOpen "
                             "Support for technical details."));
    SyncSourceSelectionFromDaemon(daemonSource_);
    return;
  }

  emit StatusRefreshRequested();
}

void AudioPage::PushDaemonAudioConfig() {
  if (!daemonAiSupported_) {
    audioWriteGuard_.MarkWriteRejected();
    return;
  }
  if (audioWriteDebounceTimer_ && audioWriteDebounceTimer_->isActive())
    audioWriteDebounceTimer_->stop();

  const QString engine = engineCombo_ ? engineCombo_->currentData().toString()
                                      : QStringLiteral("auto");

  QJsonObject effects = lastAudioEffectsObj_;
  effects.insert(
      "schema_version",
      studiocast::audio::effects::kBroadcastAudioEffectsSchemaVersion);
  effects.insert("engine", engine);

  // Microphone (only if this page has mic controls).
  if (micEffectCombo_ && strengthSlider_) {
    const QString sel = micEffectCombo_->currentData().toString();
    const bool studio = (sel == "studio_voice");
    const bool noise = (sel == "noise") || (sel == "noise_echo");
    const bool echo = (sel == "echo") || (sel == "noise_echo");
    const int strength = std::max(0, std::min(100, strengthSlider_->value()));

    QJsonObject mic = effects.value("microphone").toObject();
    mic.insert("studio_voice_enabled", studio);
    mic.insert("noise_removal_enabled", noise);
    mic.insert("room_echo_removal_enabled", echo);
    mic.insert("strength", strength);
    if (openAudioModelCombo_)
      mic.insert("model_id", openAudioModelCombo_->currentData().toString());
    if (openAudioModelPathEdit_)
      mic.insert("model_path", openAudioModelPathEdit_->text().trimmed());
    effects.insert("microphone", mic);
  }

  // Speaker (only if this page has speaker controls).
  if (speakerEffectCombo_ && speakerStrengthSlider_) {
    const QString sel = speakerEffectCombo_->currentData().toString();
    const bool spkNoise = (sel == "noise") || (sel == "noise_echo");
    const bool spkEcho = (sel == "echo") || (sel == "noise_echo");
    const int spkStrength =
        std::max(0, std::min(100, speakerStrengthSlider_->value()));

    QJsonObject spk = effects.value("speaker").toObject();
    spk.insert("noise_removal_enabled", spkNoise);
    spk.insert("room_echo_removal_enabled", spkEcho);
    spk.insert("strength", spkStrength);
    if (speakerOpenAudioModelCombo_)
      spk.insert("model_id",
                 speakerOpenAudioModelCombo_->currentData().toString());
    if (speakerOpenAudioModelPathEdit_)
      spk.insert("model_path",
                 speakerOpenAudioModelPathEdit_->text().trimmed());
    effects.insert("speaker", spk);
  }

  QJsonObject patch;
  patch.insert("audio_effects", effects);

  if (speakerEffectCombo_ && speakerTargetCombo_ &&
      speakerTargetCombo_->isEnabled()) {
    const auto target = speakerTargetCombo_->currentData().toString();
    if (!target.isEmpty())
      patch.insert("speaker_target_sink", target);
  }

  // In microphone mode we also toggle the service enabled flag.
  //
  // Note: Effect "Off" is still meaningful (it requests pass-through), so we
  // keep the pipeline running unless the backend is explicitly Off.
  if (micEffectCombo_) {
    const bool enabled = (engine != "off");
    patch.insert("enabled", enabled);
    patch.insert("create_virtual_mic", true);
  }

  const auto json = QJsonDocument(patch).toJson(QJsonDocument::Compact);

  std::string out;
  QString err;
  if (!DaemonRequest(std::string("SET_AUDIO_CONFIG ") + json.toStdString(),
                     &out, &err)) {
    daemonStatusText_ += "\nfailed_to_set_audio_config: " + err;
    if (statusText_) {
      SetPlainTextPreservingScroll(
          statusText_,
          QStringLiteral("Audio settings save failed:\n%1").arg(err));
    }
    ShowError("Audio",
              QStringLiteral("Audio settings were not saved.\n\nOpen Support "
                             "for technical details."));
    audioWriteGuard_.MarkWriteRejected();
    RefreshStatusFromCachedDaemon(/*forceControlResync=*/true);
    emit StatusRefreshRequested();
    return;
  }
  audioWriteGuard_.MarkWriteAccepted();
  emit StatusRefreshRequested();
}

void AudioPage::OnAiEngineChanged(int /*index*/) {
  if (updatingAiUi_)
    return;
  UpdateEngineUiVisibility();
  PushDaemonAudioConfig();
}

void AudioPage::OnAiOpenAudioModelChanged(int /*index*/) {
  if (updatingAiUi_)
    return;
  ScheduleDaemonAudioConfigWrite();
}

void AudioPage::OnAiOpenAudioModelPathEdited() {
  if (updatingAiUi_)
    return;
  ScheduleDaemonAudioConfigWrite();
}

void AudioPage::OnAiBrowseOpenAudioModel() {
  if (!openAudioModelPathEdit_)
    return;
  const QString start = openAudioModelPathEdit_->text().trimmed();
  const QString path = QFileDialog::getOpenFileName(
      this, "Select ONNX model", start, "ONNX model (*.onnx);;All files (*)");
  if (path.isEmpty())
    return;
  openAudioModelPathEdit_->setText(path);
  if (updatingAiUi_)
    return;
  PushDaemonAudioConfig();
}

void AudioPage::OnOpenAudioInstallHints() {
  if (openAudioInstallHints_.isEmpty()) {
    ShowError("Open Audio", "No install hints are available.");
    return;
  }
  QString msg;
  for (const auto &s : openAudioInstallHints_) {
    msg += "- " + s + "\n";
  }
  QMessageBox::information(this, "Open Audio install hints", msg);
}

void AudioPage::OnMicEffectChanged(int /*index*/) {
  if (updatingAiUi_)
    return;

  if (micEffectCombo_ && strengthSlider_) {
    const bool active = daemonAiSupported_ &&
                        (micEffectCombo_->currentData().toString() != "off");
    strengthSlider_->setEnabled(active);
    if (strengthValueLabel_)
      strengthValueLabel_->setEnabled(active);
  }

  PushDaemonAudioConfig();
}

void AudioPage::OnSpeakerEffectChanged(int /*index*/) {
  if (updatingAiUi_)
    return;

  if (speakerEffectCombo_ && speakerStrengthSlider_) {
    const bool active =
        daemonAiSupported_ &&
        (speakerEffectCombo_->currentData().toString() != "off");
    speakerStrengthSlider_->setEnabled(active);
    if (speakerStrengthValueLabel_)
      speakerStrengthValueLabel_->setEnabled(active);
  }

  PushDaemonAudioConfig();
}

void AudioPage::OnAiSpeakerStrengthChanged(int v) {
  if (speakerStrengthValueLabel_) {
    speakerStrengthValueLabel_->setText(
        QString::number(std::max(0, std::min(100, v))));
  }
  if (updatingAiUi_)
    return;
  ScheduleDaemonAudioConfigWrite();
}

void AudioPage::OnAiSpeakerOpenAudioModelChanged(int /*index*/) {
  if (updatingAiUi_)
    return;
  PushDaemonAudioConfig();
}

void AudioPage::OnAiSpeakerOpenAudioModelPathEdited() {
  if (updatingAiUi_)
    return;
  PushDaemonAudioConfig();
}

void AudioPage::OnAiSpeakerBrowseOpenAudioModel() {
  if (!speakerOpenAudioModelPathEdit_)
    return;
  const QString start = speakerOpenAudioModelPathEdit_->text().trimmed();
  const QString path = QFileDialog::getOpenFileName(
      this, "Select ONNX model", start, "ONNX model (*.onnx);;All files (*)");
  if (path.isEmpty())
    return;
  speakerOpenAudioModelPathEdit_->setText(path);
  if (updatingAiUi_)
    return;
  PushDaemonAudioConfig();
}

void AudioPage::OnAiStrengthChanged(int v) {
  if (strengthValueLabel_) {
    strengthValueLabel_->setText(
        QString::number(std::max(0, std::min(100, v))));
  }
  if (updatingAiUi_)
    return;
  ScheduleDaemonAudioConfigWrite();
}

void AudioPage::OnCreateVirtualMic() {
  if (daemonAiSupported_) {
    QJsonObject patch;
    patch.insert("create_virtual_mic", true);
    const auto json = QJsonDocument(patch).toJson(QJsonDocument::Compact);
    std::string out;
    QString err;
    if (!DaemonRequest(std::string("SET_AUDIO_CONFIG ") + json.toStdString(),
                       &out, &err)) {
      if (statusText_) {
        SetPlainTextPreservingScroll(
            statusText_,
            QStringLiteral("Virtual microphone create failed:\n%1").arg(err));
      }
      ShowError("Create virtual mic failed",
                QStringLiteral("StudioCast Microphone was not created.\n\nOpen "
                               "Support for technical details."));
      return;
    }
    emit StatusRefreshRequested();
    RefreshSources();
    return;
  }

#ifdef NDEBUG
  ShowError("Create virtual mic failed",
            "StudioCast background service is unavailable. Open Support for "
            "technical details.");
  return;
#else
  std::string err;
  if (!studiocast::audio::CreateVirtualMic(&err)) {
    ShowError("Create virtual mic failed", QString::fromStdString(err));
    return;
  }
  RefreshStatus();
  RefreshSources();
#endif
}

void AudioPage::OnDestroyVirtualMic() {
  if (!ConfirmDestructiveAction(
          this, QStringLiteral("Destroy Virtual Microphone"),
          QStringLiteral("Destroy StudioCast Microphone?"),
          QStringLiteral("This stops microphone processing and removes the "
                         "StudioCast virtual microphone device."))) {
    return;
  }

  if (daemonAiSupported_) {
    QJsonObject patch;
    patch.insert("enabled", false);
    patch.insert("create_virtual_mic", false);
    const auto json = QJsonDocument(patch).toJson(QJsonDocument::Compact);
    std::string out;
    QString err;
    if (!DaemonRequest(std::string("SET_AUDIO_CONFIG ") + json.toStdString(),
                       &out, &err)) {
      if (statusText_) {
        SetPlainTextPreservingScroll(
            statusText_,
            QStringLiteral("Virtual microphone remove failed:\n%1").arg(err));
      }
      ShowError("Destroy virtual mic failed",
                QStringLiteral("StudioCast Microphone was not removed.\n\nOpen "
                               "Support for technical details."));
      return;
    }
    emit StatusRefreshRequested();
    RefreshSources();
    return;
  }

#ifdef NDEBUG
  ShowError("Destroy virtual mic failed",
            "StudioCast background service is unavailable. Open Support for "
            "technical details.");
  return;
#else
  std::string err;
  if (!studiocast::audio::DestroyVirtualMic(&err)) {
    ShowError("Destroy virtual mic failed", QString::fromStdString(err));
    return;
  }
  RefreshStatus();
  RefreshSources();
#endif
}

void AudioPage::OnStartLoopback() {
  if (!sourceCombo_ || !sourceCombo_->isEnabled()) {
    ShowError("Start loopback failed", "No valid input source selected.");
    return;
  }

  const auto selected = sourceCombo_->currentData().toString();
  const std::string source = selected.toStdString();
  const int latency = latencySpin_ ? latencySpin_->value() : 10;

  // Optional: set port before loopback (helps laptop internal mic/headset mic
  // routing).
  if (portCombo_ && portCombo_->isEnabled()) {
    const std::string port = portCombo_->currentData().toString().toStdString();
    if (!port.empty()) {
      std::string perr;
      if (!studiocast::audio::pulse::SetSourcePort(source, port, &perr)) {
        ShowError("Set input port failed", QString::fromStdString(perr));
        return;
      }
    }
  }

  std::string err;
  if (!studiocast::audio::StartLoopback(source, latency, &err)) {
    ShowError("Start loopback failed", QString::fromStdString(err));
    return;
  }
  RefreshStatus();
}

void AudioPage::OnStopLoopback() {
  std::string err;
  if (!studiocast::audio::StopLoopback(&err)) {
    ShowError("Stop loopback failed", QString::fromStdString(err));
    return;
  }
  RefreshStatus();
}

void AudioPage::OnEnableVirtualSpeakers() {
  // Preferred path: let the daemon manage the virtual speakers device +
  // routing.
  if (daemonAiSupported_) {
    QJsonObject patch;
    patch.insert("create_virtual_speakers", true);
    patch.insert("speakers_enabled", true);
    patch.insert("speaker_latency_ms", 10);
    if (speakerTargetCombo_ && speakerTargetCombo_->isEnabled()) {
      const auto target = speakerTargetCombo_->currentData().toString();
      if (!target.isEmpty())
        patch.insert("speaker_target_sink", target);
    }

    const auto json = QJsonDocument(patch).toJson(QJsonDocument::Compact);
    std::string out;
    QString err;
    if (!DaemonRequest(std::string("SET_AUDIO_CONFIG ") + json.toStdString(),
                       &out, &err)) {
      if (statusText_) {
        SetPlainTextPreservingScroll(
            statusText_,
            QStringLiteral("Speakers enable failed:\n%1").arg(err));
      }
      ShowError("Audio",
                QStringLiteral("StudioCast Speakers were not enabled.\n\nOpen "
                               "Support for technical details."));
      return;
    }
    emit StatusRefreshRequested();
    return;
  }

#ifdef NDEBUG
  ShowError("Audio",
            "StudioCast background service is unavailable. Open Support for "
            "technical details.");
  return;
#else
  // Fallback: direct pactl manipulation (debug/dev only).
  std::string err;
  if (!studiocast::audio::CreateVirtualSpeaker(&err)) {
    ShowError("Enable speakers failed", QString::fromStdString(err));
    return;
  }

  err.clear();
  if (!studiocast::audio::StartSpeakerLoopback("", 10, &err)) {
    ShowError("Start speakers routing failed", QString::fromStdString(err));
    return;
  }

  RefreshStatus();
#endif
}

void AudioPage::OnStopSpeakersRouting() {
  if (daemonAiSupported_) {
    QJsonObject patch;
    patch.insert("speakers_enabled", false);
    const auto json = QJsonDocument(patch).toJson(QJsonDocument::Compact);
    std::string out;
    QString err;
    if (!DaemonRequest(std::string("SET_AUDIO_CONFIG ") + json.toStdString(),
                       &out, &err)) {
      if (statusText_) {
        SetPlainTextPreservingScroll(
            statusText_,
            QStringLiteral("Speakers stop failed:\n%1").arg(err));
      }
      ShowError("Audio",
                QStringLiteral("Speaker routing was not stopped.\n\nOpen "
                               "Support for technical details."));
      return;
    }
    emit StatusRefreshRequested();
    return;
  }

#ifdef NDEBUG
  ShowError("Audio",
            "StudioCast background service is unavailable. Open Support for "
            "technical details.");
  return;
#else
  std::string err;
  if (!studiocast::audio::StopSpeakerLoopback(&err)) {
    ShowError("Stop speakers routing failed", QString::fromStdString(err));
    return;
  }
  RefreshStatus();
#endif
}

void AudioPage::OnDestroyVirtualSpeakers() {
  if (!ConfirmDestructiveAction(
          this, QStringLiteral("Destroy StudioCast Speakers"),
          QStringLiteral("Destroy StudioCast Speakers?"),
          QStringLiteral("This stops speaker routing and removes the "
                         "StudioCast Speakers virtual device."))) {
    return;
  }

  if (daemonAiSupported_) {
    QJsonObject patch;
    patch.insert("speakers_enabled", false);
    patch.insert("create_virtual_speakers", false);
    const auto json = QJsonDocument(patch).toJson(QJsonDocument::Compact);
    std::string out;
    QString err;
    if (!DaemonRequest(std::string("SET_AUDIO_CONFIG ") + json.toStdString(),
                       &out, &err)) {
      if (statusText_) {
        SetPlainTextPreservingScroll(
            statusText_,
            QStringLiteral("Speakers remove failed:\n%1").arg(err));
      }
      ShowError("Audio",
                QStringLiteral("StudioCast Speakers were not removed.\n\nOpen "
                               "Support for technical details."));
      return;
    }
    emit StatusRefreshRequested();
    return;
  }

#ifdef NDEBUG
  ShowError("Audio",
            "StudioCast background service is unavailable. Open Support for "
            "technical details.");
  return;
#else
  std::string err;
  if (!studiocast::audio::DestroyVirtualSpeaker(&err)) {
    ShowError("Destroy speakers failed", QString::fromStdString(err));
    return;
  }
  RefreshStatus();
#endif
}

} // namespace studiocast::gui
