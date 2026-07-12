#pragma once

#include <QJsonObject>
#include <QProcess>
#include <QWizard>
#include <QWizardPage>

class QButtonGroup;
class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QRadioButton;

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
};

class InstallerWizard : public QWizard {
public:
  explicit InstallerWizard(QWidget *parent = nullptr);

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
  bool openVulkan() const;
  bool installVulkanRuntime() const;
  bool installMesaVulkan() const;
  bool installShaderTools() const;
  bool installModels() const;
  bool freshBuild() const;
  bool allowUnsupported() const;
  bool removeUserData() const;

  QJsonObject statusObject() const { return statusObject_; }
  QJsonObject osObject() const { return osObject_; }
  QJsonObject planObject() const { return planObject_; }

  void refreshStatus();
  bool refreshPlan(QString *error = nullptr);
  QStringList backendOptions(bool forPlan) const;
  QStringList workflowCommandArguments(bool dryRun = false) const;

  void setWorkflow(const QString &workflow);
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
  void setOpenVulkan(bool enabled);
  void setInstallVulkanRuntime(bool enabled);
  void setInstallMesaVulkan(bool enabled);
  void setInstallShaderTools(bool enabled);
  void setInstallModels(bool enabled);
  void setFreshBuild(bool enabled);
  void setAllowUnsupported(bool enabled);
  void setRemoveUserData(bool enabled);

private:
  bool runBackendJson(const QStringList &args, QJsonObject *object,
                      QString *error) const;
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
  bool openBackendsSetup_ = true;
  bool openVulkan_ = false;
  bool installVulkanRuntime_ = false;
  bool installMesaVulkan_ = false;
  bool installShaderTools_ = false;
  bool installModels_ = false;
  bool freshBuild_ = false;
  bool allowUnsupported_ = false;
  bool removeUserData_ = false;

  QJsonObject statusObject_;
  QJsonObject osObject_;
  QJsonObject planObject_;
};

class IntroPage : public QWizardPage {
public:
  explicit IntroPage(QWidget *parent = nullptr);
  void initializePage() override;
  bool validatePage() override;

private:
  QLabel *osValue_ = nullptr;
  QLabel *versionValue_ = nullptr;
  QLabel *installValue_ = nullptr;
  QButtonGroup *workflowGroup_ = nullptr;
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
  QCheckBox *openBackendsSetup_ = nullptr;
  QCheckBox *openVulkan_ = nullptr;
  QCheckBox *installVulkanRuntime_ = nullptr;
  QCheckBox *installMesaVulkan_ = nullptr;
  QCheckBox *installShaderTools_ = nullptr;
  QCheckBox *installModels_ = nullptr;
  QCheckBox *removeUserData_ = nullptr;
  QLabel *optionalComponentsNotice_ = nullptr;
};

class ReviewPage : public QWizardPage {
public:
  explicit ReviewPage(QWidget *parent = nullptr);
  void initializePage() override;

private:
  QPlainTextEdit *reviewText_ = nullptr;
};

class ProgressPage : public QWizardPage {
public:
  explicit ProgressPage(QWidget *parent = nullptr);
  void initializePage() override;
  bool isComplete() const override;
  bool validatePage() override;

private:
  void appendOutput(const QByteArray &bytes);
  QPlainTextEdit *logText_ = nullptr;
  QLabel *stateLabel_ = nullptr;
  QProcess *process_ = nullptr;
  bool started_ = false;
  bool complete_ = false;
  int exitCode_ = -1;
};

class FinishPage : public QWizardPage {
public:
  explicit FinishPage(QWidget *parent = nullptr);
  void initializePage() override;

private:
  QLabel *summaryLabel_ = nullptr;
  QPlainTextEdit *details_ = nullptr;
};

} // namespace studiocast::installer
