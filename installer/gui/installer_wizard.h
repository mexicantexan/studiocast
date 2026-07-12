#pragma once

#include <QJsonObject>
#include <QList>
#include <QProcess>
#include <QStringList>
#include <QWizard>
#include <QWizardPage>

class QButtonGroup;
class QCheckBox;
class QComboBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;
class QRadioButton;
class QSpinBox;

namespace studiocast::installer {

enum WizardPageId {
  PageIntro = 0,
  PageCompatibility,
  PageDependencyPlan,
  PageBuildOptions,
  PageInstallLocation,
  PageServiceOptions,
  PageReview,
  PageProgress,
  PageFinish,
  PageUninstall,
};

class InstallerWizard : public QWizard {
public:
  explicit InstallerWizard(QWidget *parent = nullptr);
  ~InstallerWizard() override;

  QString backendPath() const;
  QString workflow() const;
  QString sourceDir() const;
  QString releaseArchive() const;
  QString buildDir() const;
  QString buildType() const;

  bool useReleaseArchive() const;
  bool installDeps() const;
  bool configureV4l2Loopback() const;
  bool loadLoopback() const;
  bool persistLoopback() const;
  bool installService() const;
  bool openBackendsSetup() const;
  bool openCuda() const;
  bool openAudio() const;
  bool openVulkan() const;
  bool installVulkanRuntime() const;
  bool installMesaVulkan() const;
  bool installShaderTools() const;
  bool installModels() const;
  QStringList modelPackIds() const;
  QString modelDestination() const;
  int v4lDeviceNumber() const;
  QString v4lLabel() const;
  bool v4lExclusiveCaps() const;
  bool freshBuild() const;
  bool allowUnsupported() const;
  bool removeUserData() const;

  QJsonObject statusObject() const { return statusObject_; }
  QJsonObject osObject() const { return osObject_; }
  QJsonObject factsObject() const { return factsObject_; }
  QJsonObject planObject() const { return planObject_; }
  QJsonObject releaseStatusObject() const { return releaseStatusObject_; }
  QJsonObject executionResult() const { return executionResult_; }
  bool analysisAvailable() const { return analysisAvailable_; }
  QString analysisError() const { return analysisError_; }
  QString detectedRoute() const;
  QString primaryAction() const;
  bool customRoute() const { return customRoute_; }
  bool planReady() const { return planReady_; }
  QString planError() const { return planError_; }
  QString reviewedPlanPath() const { return reviewedPlanPath_; }
  QString verifiedReleaseReceiptPath() const {
    return verifiedReleaseReceiptPath_;
  }

  void refreshStatus();
  bool refreshPlan(QString *error = nullptr);
  bool launchVerifiedSelfUpdate(QString *error = nullptr);
  QStringList executionCommandArguments() const;
  QStringList backendOptions(bool forPlan) const;
  QStringList workflowCommandArguments(bool dryRun = false) const;
  int nextId() const override;
  void reject() override;

  void setWorkflow(const QString &workflow);
  void setCustomRoute(bool enabled);
  void setSourceDir(const QString &path);
  void setReleaseArchive(const QString &path);
  void setUseReleaseArchive(bool enabled);
  void setBuildDir(const QString &path);
  void setBuildType(const QString &type);
  void setInstallDeps(bool enabled);
  void setConfigureV4l2Loopback(bool enabled);
  void setLoadLoopback(bool enabled);
  void setPersistLoopback(bool enabled);
  void setInstallService(bool enabled);
  void setOpenBackendsSetup(bool enabled);
  void setOpenCuda(bool enabled);
  void setOpenAudio(bool enabled);
  void setOpenVulkan(bool enabled);
  void setInstallVulkanRuntime(bool enabled);
  void setInstallMesaVulkan(bool enabled);
  void setInstallShaderTools(bool enabled);
  void setInstallModels(bool enabled);
  void setModelPackIds(const QStringList &ids);
  void setModelDestination(const QString &path);
  void setV4lDeviceNumber(int number);
  void setV4lLabel(const QString &label);
  void setV4lExclusiveCaps(bool enabled);
  void setFreshBuild(bool enabled);
  void setAllowUnsupported(bool enabled);
  void setRemoveUserData(bool enabled);
  void setExecutionResult(const QJsonObject &result);

private:
  bool runBackendJson(const QStringList &args, QJsonObject *object,
                      QString *error) const;
  bool refreshReleaseStatus(QString *error);
  QStringList releaseChannelOptions(bool offline) const;
  void updatePreferenceProgressBars();
  void invalidatePlan();
  void seedFromPriorDesiredConfiguration();
  bool validateAndPersistPlan(const QJsonObject &plan, QString *error);
  QString backendPath_;
  QString workflow_ = QStringLiteral("install");
  QString sourceDir_;
  QString releaseArchive_;
  QString buildDir_;
  QString buildType_ = QStringLiteral("Release");

  bool useReleaseArchive_ = false;
  bool installDeps_ = true;
  bool configureV4l2Loopback_ = true;
  bool loadLoopback_ = true;
  bool persistLoopback_ = true;
  bool installService_ = true;
  bool openCuda_ = true;
  bool openAudio_ = true;
  bool openVulkan_ = false;
  bool installVulkanRuntime_ = false;
  bool installMesaVulkan_ = false;
  bool installShaderTools_ = false;
  bool installModels_ = false;
  QStringList modelPackIds_;
  QString modelDestination_;
  int v4lDeviceNumber_ = 10;
  QString v4lLabel_ = QStringLiteral("StudioCast Camera");
  bool v4lExclusiveCaps_ = true;
  bool freshBuild_ = false;
  bool allowUnsupported_ = false;
  bool removeUserData_ = false;
  bool customRoute_ = false;
  bool priorConfigurationSeeded_ = false;
  bool analysisAvailable_ = false;
  bool planReady_ = false;

  QString analysisError_;
  QString planError_;
  QString reviewedPlanPath_;
  QString verifiedReleaseReceiptPath_;
  QJsonObject factsObject_;
  QJsonObject statusObject_;
  QJsonObject osObject_;
  QJsonObject planObject_;
  QJsonObject releaseStatusObject_;
  QJsonObject executionResult_;
};

class IntroPage : public QWizardPage {
public:
  explicit IntroPage(QWidget *parent = nullptr);
  void initializePage() override;
  bool validatePage() override;
  bool isComplete() const override;

private:
  QLabel *osValue_ = nullptr;
  QLabel *versionValue_ = nullptr;
  QLabel *installValue_ = nullptr;
  QButtonGroup *workflowGroup_ = nullptr;
  QRadioButton *primaryAction_ = nullptr;
  QRadioButton *customizeAction_ = nullptr;
  QRadioButton *repairAction_ = nullptr;
  QRadioButton *reinstallAction_ = nullptr;
  QRadioButton *uninstallAction_ = nullptr;
  QGroupBox *selfUpdateBox_ = nullptr;
  QLabel *selfUpdateLabel_ = nullptr;
  QPushButton *restartInstallerButton_ = nullptr;
  QPushButton *manualDownloadButton_ = nullptr;
};

class CompatibilityPage : public QWizardPage {
public:
  explicit CompatibilityPage(QWidget *parent = nullptr);
  void initializePage() override;
  bool isComplete() const override;

private:
  QLabel *statusLabel_ = nullptr;
  QPlainTextEdit *details_ = nullptr;
  QCheckBox *allowUnsupported_ = nullptr;
};

class DependencyPlanPage : public QWizardPage {
public:
  explicit DependencyPlanPage(QWidget *parent = nullptr);
  void initializePage() override;

private:
  QPlainTextEdit *planText_ = nullptr;
};

class UninstallPage : public QWizardPage {
public:
  explicit UninstallPage(QWidget *parent = nullptr);
  void initializePage() override;
  bool validatePage() override;
  bool isComplete() const override;

private:
  void refreshPlanText();
  QLabel *statusLabel_ = nullptr;
  QLabel *dataWarning_ = nullptr;
  QCheckBox *removeUserData_ = nullptr;
  QPlainTextEdit *planText_ = nullptr;
};

class BuildOptionsPage : public QWizardPage {
public:
  explicit BuildOptionsPage(QWidget *parent = nullptr);
  void initializePage() override;
  bool validatePage() override;

private:
  QRadioButton *sourceDirRadio_ = nullptr;
  QRadioButton *archiveRadio_ = nullptr;
  QLineEdit *sourceDirEdit_ = nullptr;
  QLineEdit *archiveEdit_ = nullptr;
  QLineEdit *buildDirEdit_ = nullptr;
  QComboBox *buildTypeCombo_ = nullptr;
  QCheckBox *installDeps_ = nullptr;
  QCheckBox *freshBuild_ = nullptr;
};

class InstallLocationPage : public QWizardPage {
public:
  explicit InstallLocationPage(QWidget *parent = nullptr);
  void initializePage() override;

private:
  QLabel *binaryLocation_ = nullptr;
  QLabel *serviceLocation_ = nullptr;
  QLabel *manifestLocation_ = nullptr;
  QLabel *buildLocation_ = nullptr;
};

class ServiceOptionsPage : public QWizardPage {
public:
  explicit ServiceOptionsPage(QWidget *parent = nullptr);
  void initializePage() override;
  bool validatePage() override;

private:
  QCheckBox *installService_ = nullptr;
  QCheckBox *configureV4l2_ = nullptr;
  QCheckBox *loadLoopback_ = nullptr;
  QCheckBox *persistLoopback_ = nullptr;
  QSpinBox *v4lDeviceNumber_ = nullptr;
  QLineEdit *v4lLabel_ = nullptr;
  QCheckBox *v4lExclusiveCaps_ = nullptr;
  QCheckBox *cameraLimitationAck_ = nullptr;
  QCheckBox *openCuda_ = nullptr;
  QCheckBox *openAudio_ = nullptr;
  QCheckBox *openVulkan_ = nullptr;
  QCheckBox *installVulkanRuntime_ = nullptr;
  QCheckBox *installMesaVulkan_ = nullptr;
  QCheckBox *installShaderTools_ = nullptr;
  QCheckBox *installModels_ = nullptr;
  QList<QCheckBox *> modelPackChecks_;
  QLineEdit *modelDestination_ = nullptr;
  QCheckBox *removeUserData_ = nullptr;
  QLabel *optionalComponentsNotice_ = nullptr;
};

class ReviewPage : public QWizardPage {
public:
  explicit ReviewPage(QWidget *parent = nullptr);
  void initializePage() override;
  bool isComplete() const override;

private:
  QPlainTextEdit *reviewText_ = nullptr;
};

class ProgressPage : public QWizardPage {
public:
  explicit ProgressPage(QWidget *parent = nullptr);
  ~ProgressPage() override;
  void initializePage() override;
  bool isComplete() const override;
  bool validatePage() override;
  void requestCancellation();
  bool isRunning() const;

private:
  void appendOutput(const QByteArray &bytes);
  void consumeStructuredProgress(const QByteArray &bytes);
  void signalProcessGroup(int signal);
  QPlainTextEdit *logText_ = nullptr;
  QLabel *stateLabel_ = nullptr;
  QProcess *process_ = nullptr;
  bool started_ = false;
  bool complete_ = false;
  int exitCode_ = -1;
  QByteArray stdoutBuffer_;
  QByteArray stderrBuffer_;
  QByteArray progressLineBuffer_;
  bool cancellationRequested_ = false;
  bool isolatedProcessGroup_ = false;
};

class FinishPage : public QWizardPage {
public:
  explicit FinishPage(QWidget *parent = nullptr);
  void initializePage() override;

private:
  QLabel *summaryLabel_ = nullptr;
  QPlainTextEdit *details_ = nullptr;
  QPushButton *retryButton_ = nullptr;
  QPushButton *continueButton_ = nullptr;
};

} // namespace studiocast::installer
