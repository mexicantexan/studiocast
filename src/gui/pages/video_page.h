#pragma once

#include <QWidget>
#include <chrono>
#include <string>
#include <vector>

#include "core/ipc/daemon_client.h"
#include "core/video/effects/broadcast_effects.h"
#include "core/video/v4l2_capture.h"
#include "gui/status/pending_daemon_write_guard.h"

class QCheckBox;
class QComboBox;
class QGroupBox;
class QSlider;
class QPlainTextEdit;
class QPushButton;
class QSpinBox;
class QLabel;
class QTimer;
class QLineEdit;
class QToolButton;

namespace studiocast::gui {
struct DaemonStatusSnapshot;
class VideoPreviewWidget;
} // namespace studiocast::gui

namespace studiocast::gui {

class VideoPage final : public QWidget {
  Q_OBJECT
public:
  explicit VideoPage(QWidget *parent = nullptr);
  ~VideoPage() override;
  void UpdateStatus(const DaemonStatusSnapshot &snapshot);

signals:
  void StatusRefreshRequested();

private slots:
  void Refresh();
  void CopySuggestedCommand();
  void OnStart();
  void OnStop();
  void OnEnginePreferenceChanged(int index);
  void OnComputeBackendChanged(int index);
  void OnMirrorToggled(bool checked);
  void OnBackgroundChanged(int index);
  void OnVbModelChanged(int index);
  void OnBackgroundStrengthChanged(int value);
  void OnBackgroundRemoveColorChanged();
  void OnBackgroundReplaceImageChanged();
  void OnBrowseReplaceImage();

  void OnAutoFrameToggled(bool checked);
  void OnAutoFrameZoomChanged(int value);
  void OnAutoFrameModelChanged(int index);

  void OnEyeContactToggled(bool checked);
  void OnEyeContactStrengthChanged(int value);
  void OnEyeContactLookAwayToggled(bool checked);
  void OnEyeContactModelChanged(int index);

  void OnDenoiseToggled(bool checked);
  void OnDenoiseStrengthChanged(int value);
  void OnDenoiseModelChanged(int index);

  void OnOpenInstallHints();

  void OnVirtualKeyLightToggled(bool checked);
  void OnVirtualKeyLightIntensityChanged(int value);
  void OnVirtualKeyLightTemperatureChanged(int index);
  void OnVirtualKeyLightPanChanged(int value);
  void OnVirtualKeyLightHdriChanged();
  void OnBrowseVirtualKeyLightHdri();

  void OnVignetteToggled(bool checked);
  void OnVignetteIntensityChanged(int value);
  void OnVignetteCenterOnFaceToggled(bool checked);

  void OnPreviewToggled(bool checked);

private:
  void ShowError(const QString &title, const QString &details);
  void UpdateUiEnabled();
  void UpdateBackgroundModeOptionVisibility();
  void UpdateStatusText();
  bool SyncFromCachedDaemonStatus();
  void ScheduleDaemonVideoEffectsWrite();
  void ResyncControlsFromCachedStatus(bool force = false);
  void MarkSetupControlsEdited();
  studiocast::video::effects::BroadcastCameraEffects
  BuildCandidateEffectsFromUi() const;

  bool SendDaemonVideoConfig();
  bool SendDaemonComputeBackendPreference();
  bool SendDaemonVideoEffects();
  bool SendDaemonEnabled(bool enabled);

  void StartPreview();
  void StopPreview();
  void OnPreviewTick();
  void SetDiagnosticsVisible(bool visible);

  QComboBox *inputCombo_ = nullptr;
  QComboBox *outputCombo_ = nullptr;
  QComboBox *outputFormatCombo_ = nullptr;

  QSpinBox *widthSpin_ = nullptr;
  QSpinBox *heightSpin_ = nullptr;
  QSpinBox *fpsSpin_ = nullptr;

  QComboBox *engineCombo_ = nullptr;
  QComboBox *computeBackendCombo_ = nullptr;
  QCheckBox *mirrorCheck_ = nullptr;

  QLabel *cameraStateLabel_ = nullptr;
  QLabel *cameraDetailLabel_ = nullptr;
  QLabel *setupLockLabel_ = nullptr;
  QLineEdit *suggestedCmdEdit_ = nullptr;

  QComboBox *backgroundCombo_ = nullptr;
  QLabel *backgroundStrengthLabel_ = nullptr;
  QSpinBox *backgroundStrengthSpin_ = nullptr;

  QLabel *vbModelLabel_ = nullptr;
  QComboBox *vbModelCombo_ = nullptr;

  QLabel *backgroundRemoveColorLabel_ = nullptr;
  QLineEdit *backgroundRemoveColorEdit_ = nullptr;  // #RRGGBB
  QLabel *backgroundReplaceImageLabel_ = nullptr;
  QLineEdit *backgroundReplaceImageEdit_ = nullptr; // path (PPM/P6 for now)
  QPushButton *browseReplaceImageBtn_ = nullptr;

  // Auto Frame (Maxine AR)
  QCheckBox *autoFrameCheck_ = nullptr;
  QSlider *autoFrameZoomSlider_ = nullptr;
  QLabel *autoFrameZoomValue_ = nullptr;
  QLabel *autoFrameModelLabel_ = nullptr;
  QComboBox *autoFrameModelCombo_ = nullptr;

  // Eye Contact (Maxine AR)
  QCheckBox *eyeContactCheck_ = nullptr;
  QSlider *eyeContactStrengthSlider_ = nullptr;
  QLabel *eyeContactStrengthValue_ = nullptr;
  QCheckBox *eyeContactLookAwayCheck_ = nullptr;
  QLabel *eyeContactModelLabel_ = nullptr;
  QComboBox *eyeContactModelCombo_ = nullptr;

  // Video Noise Removal (Maxine VFX)
  QCheckBox *denoiseCheck_ = nullptr;
  QSlider *denoiseStrengthSlider_ = nullptr;
  QLabel *denoiseStrengthValue_ = nullptr;
  QLabel *denoiseModelLabel_ = nullptr;
  QComboBox *denoiseModelCombo_ = nullptr;

  // Virtual Key Light (Maxine relighting)
  QCheckBox *virtualKeyLightCheck_ = nullptr;
  QSpinBox *virtualKeyLightIntensitySpin_ = nullptr; // 0..100
  QComboBox *virtualKeyLightTempCombo_ = nullptr;    // neutral|warm|cool
  QSpinBox *virtualKeyLightPanSpin_ = nullptr;       // -180..180
  QLineEdit *virtualKeyLightHdriEdit_ = nullptr;     // path override
  QPushButton *browseVirtualKeyLightHdriBtn_ = nullptr;

  // Vignette (GPU post-process)
  QCheckBox *vignetteCheck_ = nullptr;
  QSlider *vignetteIntensitySlider_ = nullptr; // 0..100
  QLabel *vignetteIntensityValue_ = nullptr;
  QCheckBox *vignetteCenterOnFaceCheck_ = nullptr;

  QLabel *effectEngineValue_ = nullptr;
  QLabel *computeBackendValue_ = nullptr;
  QLabel *engineInfoBanner_ = nullptr;
  QLabel *maxineBanner_ = nullptr;

  QPushButton *openInstallHintsBtn_ = nullptr;
  QToolButton *diagnosticsToggle_ = nullptr;
  QGroupBox *diagnosticsBox_ = nullptr;
  QWidget *diagnosticsContent_ = nullptr;
  QPlainTextEdit *diagnosticsText_ = nullptr;

  QPushButton *refreshBtn_ = nullptr;
  QPushButton *copyCmdBtn_ = nullptr;
  QPushButton *startBtn_ = nullptr;
  QPushButton *stopBtn_ = nullptr;

  VideoPreviewWidget *preview_ = nullptr;
  QCheckBox *previewCheck_ = nullptr;
  QTimer *previewTimer_ = nullptr;
  QTimer *effectsWriteDebounceTimer_ = nullptr;
  PendingDaemonWriteGuard effectsWriteGuard_;

  QPlainTextEdit *statusText_ = nullptr;

  QString suggestedCmd_;
  std::string baseStatusText_;

  // Cached signature to avoid repopulating the VB model combo on every poll.
  QString vbModelItemsSig_;

  // Cached signatures to avoid rescanning Open Video model packs on every poll.
  QString autoFrameModelItemsSig_;
  QString eyeContactModelItemsSig_;
  QString denoiseModelItemsSig_;

  bool daemonReachable_ = false;
  std::string daemonLastStatusJson_;
  QString daemonStatusDetail_;
  bool setupControlsDirty_ = false;

  // Canonical local effects model (Broadcast schema). This is the single
  // source of truth for what the GUI intends to apply.
  studiocast::video::effects::BroadcastCameraEffects effects_{};

  // Preview is implemented by opening the virtual camera (output device)
  // as a consumer and rendering frames inside the GUI.
  studiocast::video::V4l2Capture previewCapture_;
  std::vector<uint8_t> previewRgb_;
  int previewW_ = 0;
  int previewH_ = 0;
  int previewBpl_ = 0;
  // Prevent tight auto-retry loops when preview open fails (e.g. when
  // v4l2loopback is temporarily OUTPUT-only in exclusive_caps mode).
  std::chrono::steady_clock::time_point previewAutoRetryAfter_{};
  int previewAutoRetryFailures_ = 0;
};

} // namespace studiocast::gui
