#include "installer_wizard.h"
#include "installer_wizard_internal.h"

#include <QApplication>
#include <QBoxLayout>
#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QMap>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSpinBox>
#include <QStandardPaths>
#include <QStyle>
#include <QTemporaryFile>
#include <QTextCursor>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <csignal>
#ifdef Q_OS_UNIX
#include <unistd.h>
#endif

#include "studiocast/version.h"

#ifndef STUDIOCAST_INSTALLER_BACKEND_SOURCE_PATH
#define STUDIOCAST_INSTALLER_BACKEND_SOURCE_PATH ""
#endif

#ifndef STUDIOCAST_SOURCE_DIR
#define STUDIOCAST_SOURCE_DIR ""
#endif

namespace studiocast::installer {
using namespace detail;

InstallerWizard::InstallerWizard(QWidget *parent) : QWizard(parent) {
  backendPath_ = findBackendPath();
  sourceDir_ = QString::fromUtf8(STUDIOCAST_SOURCE_DIR);
  if (sourceDir_.isEmpty() || !QFileInfo(sourceDir_).isDir()) {
    sourceDir_ = QDir::currentPath();
  }
  const QString bundledSourceArchive = findBundledSourceArchive();
  if (!bundledSourceArchive.isEmpty()) {
    releaseArchive_ = bundledSourceArchive;
    useReleaseArchive_ = true;
  }
  buildDir_ = defaultCacheBuildDir();
  modelPackIds_ = defaultModelPackIds();
  modelDestination_ = defaultModelDestination();

  setWindowTitle(QStringLiteral("StudioCast Installer"));
  setWizardStyle(QWizard::ModernStyle);
  setOption(QWizard::NoBackButtonOnStartPage);
  setOption(QWizard::NoCancelButtonOnLastPage);
  setButtonText(QWizard::CommitButton, QStringLiteral("Install"));
  setPixmap(QWizard::LogoPixmap,
            style()->standardIcon(QStyle::SP_ComputerIcon).pixmap(48, 48));

  setPage(PageIntro, new IntroPage);
  setPage(PageCompatibility, new CompatibilityPage);
  setPage(PageDependencyPlan, new DependencyPlanPage);
  setPage(PageBuildOptions, new BuildOptionsPage);
  setPage(PageInstallLocation, new InstallLocationPage);
  setPage(PageServiceOptions, new ServiceOptionsPage);
  setPage(PageReview, new ReviewPage);
  setPage(PageProgress, new ProgressPage);
  setPage(PageFinish, new FinishPage);
  setPage(PageUninstall, new UninstallPage);
  addPreferenceProgressBars(this);
  updatePreferenceProgressBars();
  setStartId(PageIntro);

  resize(920, 680);
  refreshStatus();
}

InstallerWizard::~InstallerWizard() {
  invalidatePlan();
  if (!verifiedReleaseReceiptPath_.isEmpty()) {
    QFile::remove(verifiedReleaseReceiptPath_);
  }
}

QString InstallerWizard::backendPath() const { return backendPath_; }
QString InstallerWizard::workflow() const { return workflow_; }
QString InstallerWizard::sourceDir() const { return sourceDir_; }
QString InstallerWizard::releaseArchive() const { return releaseArchive_; }
QString InstallerWizard::buildDir() const { return buildDir_; }
QString InstallerWizard::buildType() const { return buildType_; }

bool InstallerWizard::useReleaseArchive() const { return useReleaseArchive_; }
bool InstallerWizard::installDeps() const { return installDeps_; }
bool InstallerWizard::configureV4l2Loopback() const {
  return configureV4l2Loopback_;
}
bool InstallerWizard::loadLoopback() const { return loadLoopback_; }
bool InstallerWizard::persistLoopback() const { return persistLoopback_; }
bool InstallerWizard::installService() const { return installService_; }
bool InstallerWizard::openBackendsSetup() const {
  return openCuda_ || openAudio_;
}
bool InstallerWizard::openCuda() const { return openCuda_; }
bool InstallerWizard::openAudio() const { return openAudio_; }
bool InstallerWizard::openVulkan() const { return openVulkan_; }
bool InstallerWizard::installVulkanRuntime() const {
  return installVulkanRuntime_;
}
bool InstallerWizard::installMesaVulkan() const { return installMesaVulkan_; }
bool InstallerWizard::installShaderTools() const { return installShaderTools_; }
bool InstallerWizard::installModels() const { return installModels_; }
QStringList InstallerWizard::modelPackIds() const { return modelPackIds_; }
QString InstallerWizard::modelDestination() const { return modelDestination_; }
int InstallerWizard::v4lDeviceNumber() const { return v4lDeviceNumber_; }
QString InstallerWizard::v4lLabel() const { return v4lLabel_; }
bool InstallerWizard::v4lExclusiveCaps() const { return v4lExclusiveCaps_; }
bool InstallerWizard::freshBuild() const { return freshBuild_; }
bool InstallerWizard::allowUnsupported() const { return allowUnsupported_; }
bool InstallerWizard::removeUserData() const { return removeUserData_; }

QString InstallerWizard::detectedRoute() const {
  return jsonString(statusObject_, QStringLiteral("route"));
}

QString InstallerWizard::primaryAction() const {
  return jsonString(statusObject_, QStringLiteral("primary_action"));
}

void InstallerWizard::invalidatePlan() {
  if (!reviewedPlanPath_.isEmpty()) {
    QFile::remove(reviewedPlanPath_);
  }
  reviewedPlanPath_.clear();
  planObject_ = {};
  planReady_ = false;
  planError_.clear();
}

void InstallerWizard::setWorkflow(const QString &workflow) {
  if (workflow_ != workflow) {
    invalidatePlan();
  }
  workflow_ = workflow;
  updatePreferenceProgressBars();
}
void InstallerWizard::setCustomRoute(bool enabled) {
  if (customRoute_ != enabled) {
    invalidatePlan();
  }
  customRoute_ = enabled;
}
void InstallerWizard::setAdvancedSourceSelected(bool enabled) {
  if (advancedSourceSelected_ != enabled) {
    invalidatePlan();
  }
  advancedSourceSelected_ = enabled;
}
void InstallerWizard::setSourceDir(const QString &path) {
  if (sourceDir_ != path)
    invalidatePlan();
  sourceDir_ = path;
}
void InstallerWizard::setReleaseArchive(const QString &path) {
  if (releaseArchive_ != path)
    invalidatePlan();
  releaseArchive_ = path;
}
void InstallerWizard::setUseReleaseArchive(bool enabled) {
  if (useReleaseArchive_ != enabled)
    invalidatePlan();
  useReleaseArchive_ = enabled;
}
void InstallerWizard::setBuildDir(const QString &path) {
  if (buildDir_ != path)
    invalidatePlan();
  buildDir_ = path;
}
void InstallerWizard::setBuildType(const QString &type) {
  if (buildType_ != type)
    invalidatePlan();
  buildType_ = type;
}
void InstallerWizard::setInstallDeps(bool enabled) {
  if (installDeps_ != enabled)
    invalidatePlan();
  installDeps_ = enabled;
}
void InstallerWizard::setConfigureV4l2Loopback(bool enabled) {
  if (configureV4l2Loopback_ != enabled || !v4lChoiceResolved_)
    invalidatePlan();
  configureV4l2Loopback_ = enabled;
  v4lChoiceResolved_ = true;
}
void InstallerWizard::setLoadLoopback(bool enabled) {
  if (loadLoopback_ != enabled || !loadLoopbackChoiceResolved_)
    invalidatePlan();
  loadLoopback_ = enabled;
  loadLoopbackChoiceResolved_ = true;
}
void InstallerWizard::setPersistLoopback(bool enabled) {
  if (persistLoopback_ != enabled || !persistLoopbackChoiceResolved_)
    invalidatePlan();
  persistLoopback_ = enabled;
  persistLoopbackChoiceResolved_ = true;
}
void InstallerWizard::setInstallService(bool enabled) {
  if (installService_ != enabled || !serviceChoiceResolved_)
    invalidatePlan();
  installService_ = enabled;
  serviceChoiceResolved_ = true;
}
void InstallerWizard::setOpenBackendsSetup(bool enabled) {
  if (openCuda_ != enabled || openAudio_ != enabled)
    invalidatePlan();
  openCuda_ = enabled;
  openAudio_ = enabled;
}
void InstallerWizard::setOpenCuda(bool enabled) {
  if (openCuda_ != enabled)
    invalidatePlan();
  openCuda_ = enabled;
}
void InstallerWizard::setOpenAudio(bool enabled) {
  if (openAudio_ != enabled)
    invalidatePlan();
  openAudio_ = enabled;
}
void InstallerWizard::setOpenVulkan(bool enabled) {
  if (openVulkan_ != enabled)
    invalidatePlan();
  openVulkan_ = enabled;
}
void InstallerWizard::setInstallVulkanRuntime(bool enabled) {
  if (installVulkanRuntime_ != enabled)
    invalidatePlan();
  installVulkanRuntime_ = enabled;
  if (!enabled) {
    installMesaVulkan_ = false;
  }
}
void InstallerWizard::setInstallMesaVulkan(bool enabled) {
  if (installMesaVulkan_ != enabled)
    invalidatePlan();
  installMesaVulkan_ = enabled;
  if (enabled) {
    installVulkanRuntime_ = true;
  }
}
void InstallerWizard::setInstallShaderTools(bool enabled) {
  if (installShaderTools_ != enabled)
    invalidatePlan();
  installShaderTools_ = enabled;
}
void InstallerWizard::setInstallModels(bool enabled) {
  if (installModels_ != enabled)
    invalidatePlan();
  installModels_ = enabled;
}
void InstallerWizard::setModelPackIds(const QStringList &ids) {
  QStringList normalized;
  const QStringList defaults = defaultModelPackIds();
  for (const QString &id : defaults) {
    if (ids.contains(id))
      normalized.append(id);
  }
  if (modelPackIds_ != normalized)
    invalidatePlan();
  modelPackIds_ = normalized;
}
void InstallerWizard::setModelDestination(const QString &path) {
  const QString normalized = QDir::cleanPath(path.trimmed());
  if (modelDestination_ != normalized)
    invalidatePlan();
  modelDestination_ = normalized;
}
void InstallerWizard::setV4lDeviceNumber(int number) {
  if (v4lDeviceNumber_ != number)
    invalidatePlan();
  v4lDeviceNumber_ = number;
}
void InstallerWizard::setV4lLabel(const QString &label) {
  const QString normalized = label.trimmed();
  if (v4lLabel_ != normalized)
    invalidatePlan();
  v4lLabel_ = normalized;
}
void InstallerWizard::setV4lExclusiveCaps(bool enabled) {
  if (v4lExclusiveCaps_ != enabled)
    invalidatePlan();
  v4lExclusiveCaps_ = enabled;
}
void InstallerWizard::setFreshBuild(bool enabled) {
  if (freshBuild_ != enabled)
    invalidatePlan();
  freshBuild_ = enabled;
}
void InstallerWizard::setAllowUnsupported(bool enabled) {
  allowUnsupported_ = enabled;
}
void InstallerWizard::setRemoveUserData(bool enabled) {
  if (removeUserData_ != enabled)
    invalidatePlan();
  removeUserData_ = enabled;
}

void InstallerWizard::setExecutionResult(const QJsonObject &result) {
  executionResult_ = result;
}

bool InstallerWizard::runBackendJson(const QStringList &args,
                                     QJsonObject *object,
                                     QString *error) const {
  if (backendPath_.isEmpty() || !QFileInfo(backendPath_).isExecutable()) {
    if (error) {
      *error = QStringLiteral("Installer backend is not executable: %1")
                   .arg(backendPath_);
    }
    return false;
  }

  QProcess process;
  process.start(backendPath_, args);
  if (!process.waitForFinished(15000)) {
    process.kill();
    if (error) {
      *error = QStringLiteral("Timed out running backend: %1 %2")
                   .arg(backendPath_, args.join(QLatin1Char(' ')));
    }
    return false;
  }

  const QByteArray stdoutData = process.readAllStandardOutput();
  const QByteArray stderrData = process.readAllStandardError();
  if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
    if (error) {
      *error = QString::fromLocal8Bit(stderrData);
      if (error->isEmpty()) {
        *error = QStringLiteral("Backend exited with code %1")
                     .arg(process.exitCode());
      }
    }
    return false;
  }

  QJsonParseError parseError;
  const QJsonDocument document =
      QJsonDocument::fromJson(stdoutData, &parseError);
  if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
    if (error) {
      *error = QStringLiteral("Backend returned invalid JSON: %1")
                   .arg(parseError.errorString());
    }
    return false;
  }

  if (object) {
    *object = document.object();
  }
  return true;
}

QStringList InstallerWizard::releaseChannelOptions(bool offline) const {
  const QProcessEnvironment environment =
      QProcessEnvironment::systemEnvironment();
  QStringList options;
  const QString stableUrl =
      environment.value(QStringLiteral("STUDIOCAST_STABLE_CHANNEL_URL"));
  const QString stableSignatureUrl = environment.value(
      QStringLiteral("STUDIOCAST_STABLE_CHANNEL_SIGNATURE_URL"));
  const QString releaseCache =
      environment.value(QStringLiteral("STUDIOCAST_RELEASE_CACHE_DIR"));
  if (!stableUrl.isEmpty())
    options << QStringLiteral("--stable-url") << stableUrl;
  if (!stableSignatureUrl.isEmpty())
    options << QStringLiteral("--stable-signature-url") << stableSignatureUrl;
  if (!releaseCache.isEmpty())
    options << QStringLiteral("--release-cache-dir") << releaseCache;
  options << QStringLiteral("--prepare-self-update");
  const QString appImage = environment.value(QStringLiteral("APPIMAGE"));
  const QFileInfo appImageInfo(appImage);
  if (appImageInfo.isAbsolute() && appImageInfo.isFile() &&
      !appImageInfo.isSymLink()) {
    options << QStringLiteral("--appimage-path")
            << appImageInfo.absoluteFilePath();
  }
  if (offline)
    options << QStringLiteral("--offline");
  return options;
}

bool InstallerWizard::refreshReleaseStatus(QString *error) {
  releaseStatusObject_ = {};
  if (!verifiedReleaseReceiptPath_.isEmpty()) {
    QFile::remove(verifiedReleaseReceiptPath_);
    verifiedReleaseReceiptPath_.clear();
  }

  const QString tempRoot =
      QStandardPaths::writableLocation(QStandardPaths::TempLocation);
  QTemporaryFile receipt(QDir(tempRoot).filePath(
      QStringLiteral("studiocast-verified-release-XXXXXX.json")));
  receipt.setAutoRemove(false);
  if (!receipt.open()) {
    if (error)
      *error = QStringLiteral("Could not allocate a private release receipt.");
    return false;
  }
  const QString receiptPath = receipt.fileName();
  receipt.close();
  QFile::remove(receiptPath);

  const bool explicitlyOffline =
      QProcessEnvironment::systemEnvironment().value(QStringLiteral(
          "STUDIOCAST_INSTALLER_OFFLINE")) == QStringLiteral("1");
  QString onlineError;
  QJsonObject releaseStatus;
  QStringList args{QStringLiteral("release-status"), QStringLiteral("--json"),
                   QStringLiteral("--release-receipt-out"), receiptPath};
  args += releaseChannelOptions(explicitlyOffline);
  bool verified = runBackendJson(args, &releaseStatus,
                                 explicitlyOffline ? nullptr : &onlineError);
  QString offlineError;
  if (!verified && !explicitlyOffline) {
    args = {QStringLiteral("release-status"), QStringLiteral("--json"),
            QStringLiteral("--release-receipt-out"), receiptPath};
    args += releaseChannelOptions(true);
    verified = runBackendJson(args, &releaseStatus, &offlineError);
  } else if (!verified) {
    offlineError = QStringLiteral(
        "No verified signed release was available in the offline cache.");
  }

  if (!verified ||
      releaseStatus.value(QStringLiteral("schema_version")).toInt() != 1 ||
      releaseStatus.value(QStringLiteral("release_status_version"))
              .toString() != QStringLiteral("installer-release-status/v1") ||
      releaseStatus.value(QStringLiteral("available_version"))
          .toString()
          .isEmpty() ||
      (explicitlyOffline &&
       !jsonBool(releaseStatus, QStringLiteral("offline"))) ||
      !QFileInfo(receiptPath).isFile()) {
    QFile::remove(receiptPath);
    if (error) {
      if (explicitlyOffline) {
        *error = offlineError.trimmed().isEmpty()
                     ? QStringLiteral("Verified signed offline release status "
                                      "was incomplete.")
                     : offlineError.trimmed();
      } else if (!offlineError.trimmed().isEmpty()) {
        *error =
            QStringLiteral("Stable release verification failed online (%1) "
                           "and no verified offline cache was usable (%2).")
                .arg(onlineError.trimmed(), offlineError.trimmed());
      } else if (!onlineError.trimmed().isEmpty()) {
        *error = onlineError.trimmed();
      } else {
        *error = QStringLiteral("Stable release status or signed receipt "
                                "evidence was incomplete.");
      }
    }
    return false;
  }

  QFile(receiptPath)
      .setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner);
  verifiedReleaseReceiptPath_ = receiptPath;
  releaseStatusObject_ = releaseStatus;
  return true;
}

bool InstallerWizard::launchVerifiedSelfUpdate(QString *error) {
  const QJsonObject selfUpdate =
      releaseStatusObject_.value(QStringLiteral("self_update")).toObject();
  const QJsonArray commandValue =
      selfUpdate.value(QStringLiteral("relaunch_command")).toArray();
  QStringList command;
  for (const QJsonValue &value : commandValue) {
    if (!value.isString() || value.toString().isEmpty()) {
      if (error)
        *error = QStringLiteral("The verified restart command is malformed.");
      return false;
    }
    command << value.toString();
  }
  const QString verifiedPath =
      jsonString(selfUpdate, QStringLiteral("verified_appimage"));
  const QFileInfo verifiedInfo(verifiedPath);
  const QProcessEnvironment environment =
      QProcessEnvironment::systemEnvironment();
  const QString currentPath =
      jsonString(selfUpdate, QStringLiteral("current_appimage"));
  const QFileInfo currentInfo(currentPath);
  const QFileInfo detectedInfo(environment.value(QStringLiteral("APPIMAGE")));
  if (jsonString(selfUpdate, QStringLiteral("state")) !=
          QStringLiteral("offer_restart") ||
      !jsonBool(selfUpdate, QStringLiteral("requires_confirmation")) ||
      jsonBool(selfUpdate, QStringLiteral("running_artifact_replaced"), true) ||
      command.size() != 1 || command.first() != verifiedPath ||
      !verifiedInfo.isAbsolute() || !verifiedInfo.isFile() ||
      verifiedInfo.isSymLink() || !verifiedInfo.isExecutable() ||
      verifiedInfo.canonicalFilePath() != verifiedInfo.absoluteFilePath() ||
      !currentInfo.isFile() || currentInfo.isSymLink() ||
      currentInfo.canonicalFilePath() != detectedInfo.canonicalFilePath() ||
      currentInfo.canonicalFilePath() == verifiedInfo.canonicalFilePath()) {
    if (error)
      *error = QStringLiteral(
          "The installer update is not a valid verified restart offer.");
    return false;
  }

  const bool testConfirmed =
      environment.value(
          QStringLiteral("STUDIOCAST_INSTALLER_TEST_CONFIRM_SELF_UPDATE")) ==
      QStringLiteral("1");
  if (!testConfirmed &&
      QMessageBox::question(
          this, QStringLiteral("Restart with verified installer update?"),
          QStringLiteral("StudioCast verified a newer installer at:\n%1\n\n"
                         "Restart now? The running AppImage will not be "
                         "overwritten.")
              .arg(verifiedPath),
          QMessageBox::Yes | QMessageBox::No,
          QMessageBox::No) != QMessageBox::Yes) {
    if (error)
      error->clear();
    return false;
  }

  const QString launchLog = environment.value(
      QStringLiteral("STUDIOCAST_INSTALLER_TEST_SELF_UPDATE_LAUNCH_LOG"));
  if (!launchLog.isEmpty()) {
    QFile file(launchLog);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
      if (error)
        *error = QStringLiteral("Could not record the test restart command.");
      return false;
    }
    file.write(QJsonDocument(QJsonArray::fromStringList(command))
                   .toJson(QJsonDocument::Compact));
    return true;
  }

  const QString program = command.takeFirst();
  if (!QProcess::startDetached(program, command)) {
    if (error)
      *error =
          QStringLiteral("Could not launch the verified installer update.");
    return false;
  }
  QCoreApplication::quit();
  return true;
}

void InstallerWizard::refreshStatus() {
  analysisAvailable_ = false;
  analysisError_.clear();
  QString releaseError;
  const bool releaseVerified = refreshReleaseStatus(&releaseError);
  QString error;
  QJsonObject facts;
  QStringList analyzeArgs{QStringLiteral("analyze"), QStringLiteral("--json")};
  if (releaseVerified) {
    analyzeArgs << QStringLiteral("--target-version")
                << releaseStatusObject_
                       .value(QStringLiteral("available_version"))
                       .toString();
  }
  if (!runBackendJson(analyzeArgs, &facts, &error)) {
    factsObject_ = {};
    statusObject_ = {{QStringLiteral("backend_error"), error},
                     {QStringLiteral("route"), QStringLiteral("analysis_error")},
                     {QStringLiteral("primary_action"), QStringLiteral("retry")}};
    osObject_ = {};
    analysisError_ = error;
    return;
  }
  QJsonObject status;
  QStringList statusArgs{QStringLiteral("status"), QStringLiteral("--json")};
  if (releaseVerified) {
    statusArgs << QStringLiteral("--release-receipt")
               << verifiedReleaseReceiptPath_;
  }
  if (runBackendJson(statusArgs, &status, &error)) {
    if (facts.value(QStringLiteral("facts_version")).toString() !=
            QStringLiteral("installer-facts/v1") ||
        status.value(QStringLiteral("schema_version")).toInt() != 2) {
      analysisError_ = QStringLiteral("Unsupported installer analysis protocol.");
      statusObject_ = {{QStringLiteral("backend_error"), analysisError_},
                       {QStringLiteral("route"), QStringLiteral("analysis_error")},
                       {QStringLiteral("primary_action"), QStringLiteral("retry")}};
      factsObject_ = {};
      osObject_ = {};
      return;
    }
    factsObject_ = facts;
    statusObject_ = status;
    if (!releaseVerified &&
        jsonString(statusObject_, QStringLiteral("route")) !=
            QStringLiteral("unsupported")) {
      statusObject_[QStringLiteral("route")] = QStringLiteral("offline");
      statusObject_[QStringLiteral("primary_action")] = QStringLiteral("stop");
      statusObject_[QStringLiteral("release_error")] = releaseError;
    }
    osObject_ = facts.value(QStringLiteral("os")).toObject();
    analysisAvailable_ = true;
    seedFromPriorDesiredConfiguration();
  } else {
    factsObject_ = {};
    statusObject_ = {{QStringLiteral("backend_error"), error},
                     {QStringLiteral("route"), QStringLiteral("analysis_error")},
                     {QStringLiteral("primary_action"), QStringLiteral("retry")}};
    osObject_ = {};
    analysisError_ = error;
  }
}

void InstallerWizard::seedFromPriorDesiredConfiguration() {
  if (priorConfigurationSeeded_) {
    return;
  }
  priorConfigurationSeeded_ = true;
  const QJsonObject installation =
      factsObject_.value(QStringLiteral("installation")).toObject();
  const QJsonObject prior =
      installation.value(QStringLiteral("desired_configuration")).toObject();
  if (!prior.isEmpty()) {
    buildType_ = jsonString(prior, QStringLiteral("build_type"), buildType_);
    const QJsonObject features =
        prior.value(QStringLiteral("features")).toObject();
    openCuda_ = jsonBool(features, QStringLiteral("open_cuda"), openCuda_);
    openAudio_ = jsonBool(features, QStringLiteral("open_audio"), openAudio_);
    openVulkan_ = jsonBool(features, QStringLiteral("open_vulkan"), false);
    const QJsonObject service = prior.value(QStringLiteral("service")).toObject();
    const QString serviceDesired =
        jsonString(service, QStringLiteral("desired"));
    serviceChoiceResolved_ =
        serviceDesired == QStringLiteral("enabled_started") ||
        serviceDesired == QStringLiteral("disabled");
    installService_ = serviceDesired == QStringLiteral("enabled_started");
    const QJsonObject camera = prior.value(QStringLiteral("v4l")).toObject();
    v4lChoiceResolved_ = camera.value(QStringLiteral("desired")).isBool();
    configureV4l2Loopback_ =
        v4lChoiceResolved_ ? camera.value(QStringLiteral("desired")).toBool()
                           : false;
    loadLoopbackChoiceResolved_ = v4lChoiceResolved_;
    loadLoopback_ = configureV4l2Loopback_;
    persistLoopbackChoiceResolved_ =
        camera.value(QStringLiteral("persist")).isBool();
    persistLoopback_ = persistLoopbackChoiceResolved_
                           ? camera.value(QStringLiteral("persist")).toBool()
                           : false;
    v4lDeviceNumber_ =
        camera.value(QStringLiteral("device_number")).toInt(v4lDeviceNumber_);
    v4lLabel_ = jsonString(camera, QStringLiteral("label"), v4lLabel_);
    v4lExclusiveCaps_ =
        jsonBool(camera, QStringLiteral("exclusive_caps"), v4lExclusiveCaps_);
    const QJsonArray priorModels =
        prior.value(QStringLiteral("model_pack_ids")).toArray();
    QStringList selectedModels;
    for (const QJsonValue &value : priorModels) {
      if (value.isString())
        selectedModels.append(value.toString());
    }
    setModelPackIds(selectedModels);
    installModels_ = !modelPackIds_.isEmpty();
    modelDestination_ = jsonString(prior, QStringLiteral("model_destination"),
                                   modelDestination_);
    removeUserData_ =
        !jsonBool(prior, QStringLiteral("preserve_settings"), true);
  } else {
    installModels_ = true;
    modelPackIds_ = defaultModelPackIds();
    openVulkan_ = false;
  }
}

bool InstallerWizard::refreshPlan(QString *error) {
  invalidatePlan();
  if (!analysisAvailable_) {
    planError_ = analysisError_.isEmpty()
                     ? QStringLiteral("Analysis is unavailable.")
                     : analysisError_;
    if (error)
      *error = planError_;
    return false;
  }
  if (detectedRoute() == QStringLiteral("unsupported") ||
      detectedRoute() == QStringLiteral("offline") ||
      detectedRoute() == QStringLiteral("analysis_error")) {
    planError_ = QStringLiteral(
        "Automatic changes are blocked for this analyzed system state.");
    if (error)
      *error = planError_;
    return false;
  }
  if (workflow_ != QStringLiteral("uninstall") &&
      (!serviceChoiceResolved_ || !v4lChoiceResolved_ ||
       (configureV4l2Loopback_ &&
        (!loadLoopbackChoiceResolved_ || !persistLoopbackChoiceResolved_)))) {
    planError_ = QStringLiteral(
        "Migrated service or virtual-camera choices are unknown. Choose "
        "Customize and explicitly review each unknown choice before Apply.");
    if (error)
      *error = planError_;
    return false;
  }
  if (workflow_ != QStringLiteral("repair") &&
      workflow_ != QStringLiteral("uninstall") && !advancedSourceSelected_ &&
      (verifiedReleaseReceiptPath_.isEmpty() ||
       !QFileInfo(verifiedReleaseReceiptPath_).isFile())) {
    planError_ = QStringLiteral(
        "Official Recommended or Custom installation requires a current "
        "signed-release receipt.");
    if (error)
      *error = planError_;
    return false;
  }
  QJsonObject plan;
  QStringList args{QStringLiteral("plan"), workflow_, QStringLiteral("--json")};
  args += backendOptions(true);
  if (!runBackendJson(args, &plan, error)) {
    planError_ = error ? *error : QStringLiteral("Plan generation failed.");
    return false;
  }
  return validateAndPersistPlan(plan, error);
}

bool InstallerWizard::validateAndPersistPlan(const QJsonObject &plan,
                                              QString *error) {
  const QString digest = jsonString(plan, QStringLiteral("plan_digest"));
  const QString token = jsonString(plan, QStringLiteral("approval_token"));
  const QJsonArray operations = plan.value(QStringLiteral("operations")).toArray();
  QString validationError;
  if (plan.value(QStringLiteral("schema_version")).toInt() != 1 ||
      jsonString(plan, QStringLiteral("plan_version")) !=
          QStringLiteral("installer-plan/v1")) {
    validationError = QStringLiteral("Unsupported reviewed plan schema.");
  } else if (!digest.startsWith(QStringLiteral("sha256:")) || token.isEmpty()) {
    validationError = QStringLiteral("Reviewed plan is missing its digest or approval token.");
  } else if (operations.isEmpty()) {
    validationError = QStringLiteral("Reviewed plan contains no operations.");
  } else if (!plan.value(QStringLiteral("blockers")).toArray().isEmpty()) {
    validationError = QStringLiteral("Reviewed plan contains blocking conditions.");
  } else {
    const QDateTime expiry = QDateTime::fromString(
        jsonString(plan, QStringLiteral("expires_at")), Qt::ISODate);
    if (!expiry.isValid() || expiry <= QDateTime::currentDateTimeUtc()) {
      validationError = QStringLiteral("Reviewed plan is expired or has no valid expiry.");
    }
  }
  if (!validationError.isEmpty()) {
    planError_ = validationError;
    if (error)
      *error = validationError;
    return false;
  }

  const QString tempRoot =
      QStandardPaths::writableLocation(QStandardPaths::TempLocation);
  QTemporaryFile file(
      QDir(tempRoot).filePath(QStringLiteral("studiocast-reviewed-plan-XXXXXX.json")));
  file.setAutoRemove(false);
  if (!file.open() ||
      file.write(QJsonDocument(plan).toJson(QJsonDocument::Indented)) < 0 ||
      !file.flush()) {
    planError_ = QStringLiteral("Could not persist the exact reviewed plan privately.");
    if (error)
      *error = planError_;
    return false;
  }
  file.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner);
  reviewedPlanPath_ = file.fileName();
  file.close();
  planObject_ = plan;
  planReady_ = true;
  planError_.clear();
  return true;
}

QStringList InstallerWizard::executionCommandArguments() const {
  if (!planReady_ || reviewedPlanPath_.isEmpty()) {
    return {};
  }
  return {QStringLiteral("execute-plan"), QStringLiteral("--plan"),
          reviewedPlanPath_, QStringLiteral("--digest"),
          jsonString(planObject_, QStringLiteral("plan_digest")),
          QStringLiteral("--token"),
          jsonString(planObject_, QStringLiteral("approval_token"))};
}

QStringList InstallerWizard::backendOptions(bool forPlan) const {
  QStringList args;

  if (workflow_ == QStringLiteral("uninstall")) {
    args << (removeUserData_ ? QStringLiteral("--remove-user-data")
                             : QStringLiteral("--preserve-user-data"));
    if (!forPlan) {
      args << QStringLiteral("--yes");
    }
    return args;
  }

  const bool needsSource = workflow_ != QStringLiteral("uninstall") &&
                           workflow_ != QStringLiteral("advanced");
  const bool recommendedSelection =
      !customRoute_ && detectedRoute() == QStringLiteral("recommended") &&
      workflow_ == QStringLiteral("install");
  args << QStringLiteral("--route")
       << (advancedSourceSelected_ ? QStringLiteral("advanced")
           : customRoute_          ? QStringLiteral("custom")
                                   : QStringLiteral("recommended"));
  if (jsonBool(releaseStatusObject_, QStringLiteral("offline"))) {
    args << QStringLiteral("--offline");
  }
  if (needsSource) {
    if (!advancedSourceSelected_ && !verifiedReleaseReceiptPath_.isEmpty()) {
      args << QStringLiteral("--release-receipt")
           << verifiedReleaseReceiptPath_;
    } else if (useReleaseArchive_ && !releaseArchive_.isEmpty()) {
      args << QStringLiteral("--release-archive") << releaseArchive_;
    } else if (!sourceDir_.isEmpty()) {
      args << QStringLiteral("--source-dir") << sourceDir_;
    }
    if (!buildDir_.isEmpty()) {
      args << QStringLiteral("--build-dir") << buildDir_;
    }
    const QString effectiveBuildType =
        recommendedSelection ? QStringLiteral("Release") : buildType_;
    if (!effectiveBuildType.isEmpty()) {
      args << QStringLiteral("--build-type") << effectiveBuildType;
    }
    if (workflow_ == QStringLiteral("repair") ||
        workflow_ == QStringLiteral("reinstall") ||
        workflow_ == QStringLiteral("clean-install") ||
        workflow_ == QStringLiteral("modify")) {
      args << QStringLiteral("--allow-same-version");
    }
  }

  if (installDeps_) {
    args << QStringLiteral("--with-deps");
  } else {
    args << QStringLiteral("--skip-deps");
  }

  const bool effectiveV4l = recommendedSelection || configureV4l2Loopback_;
  if (!v4lChoiceResolved_ ||
      (configureV4l2Loopback_ &&
       (!loadLoopbackChoiceResolved_ || !persistLoopbackChoiceResolved_))) {
    if (!forPlan)
      args << QStringLiteral("--yes");
    return args;
  }
  if (effectiveV4l) {
    args << QStringLiteral("--v4l2loopback");
  } else {
    args << QStringLiteral("--no-v4l2loopback");
  }
  args << ((effectiveV4l && (recommendedSelection || loadLoopback_))
               ? QStringLiteral("--load-loopback")
               : QStringLiteral("--no-load-loopback"));
  args << ((effectiveV4l && (recommendedSelection || persistLoopback_))
               ? QStringLiteral("--persist-loopback")
               : QStringLiteral("--no-persist-loopback"));
  args << QStringLiteral("--v4l-device-number")
       << QString::number(recommendedSelection ? 10 : v4lDeviceNumber_);
  args << QStringLiteral("--v4l-label")
       << (recommendedSelection ? QStringLiteral("StudioCast Camera")
                                : v4lLabel_);
  args << ((recommendedSelection || v4lExclusiveCaps_)
               ? QStringLiteral("--v4l-exclusive-caps")
               : QStringLiteral("--no-v4l-exclusive-caps"));
  if (serviceChoiceResolved_) {
    args << (installService_ ? QStringLiteral("--service")
                             : QStringLiteral("--no-service"));
  }
  args << ((recommendedSelection || openCuda_)
               ? QStringLiteral("--open-cuda")
               : QStringLiteral("--no-open-cuda"));
  args << ((recommendedSelection || openAudio_)
               ? QStringLiteral("--open-audio")
               : QStringLiteral("--no-open-audio"));
  args << (openVulkan_ ? QStringLiteral("--open-vulkan")
                       : QStringLiteral("--no-open-vulkan"));
  args << (installVulkanRuntime_ ? QStringLiteral("--vulkan-runtime")
                                 : QStringLiteral("--no-vulkan-runtime"));
  if (!recommendedSelection && installMesaVulkan_) {
    args << QStringLiteral("--mesa-vulkan");
  }
  if (!recommendedSelection && installShaderTools_) {
    args << QStringLiteral("--shader-tools");
  }
  const bool effectiveModels = recommendedSelection || installModels_;
  args << (effectiveModels ? QStringLiteral("--models")
                           : QStringLiteral("--no-models"));
  const QStringList effectiveModelIds =
      recommendedSelection ? defaultModelPackIds() : modelPackIds_;
  if (effectiveModels) {
    for (const QString &id : effectiveModelIds)
      args << QStringLiteral("--model-id") << id;
  }
  args << QStringLiteral("--model-destination")
       << (recommendedSelection ? defaultModelDestination()
                                : modelDestination_);
  args << (freshBuild_ ? QStringLiteral("--fresh-build")
                       : QStringLiteral("--no-fresh-build"));
  args << (removeUserData_ ? QStringLiteral("--remove-user-data")
                           : QStringLiteral("--preserve-user-data"));
  if (!forPlan) {
    args << QStringLiteral("--yes");
  }
  return args;
}

QStringList InstallerWizard::workflowCommandArguments(bool dryRun) const {
  QStringList args = executionCommandArguments();
  if (dryRun) {
    args << QStringLiteral("--dry-run");
  }
  return args;
}

int InstallerWizard::nextId() const {
  switch (currentId()) {
  case PageIntro:
    if (workflow_ == QStringLiteral("uninstall"))
      return PageUninstall;
    if (workflow_ == QStringLiteral("reinstall") ||
        workflow_ == QStringLiteral("clean-install"))
      return PageServiceOptions;
    return customRoute_ ? PageDependencyPlan : PageReview;
  case PageCompatibility:
    return PageDependencyPlan;
  case PageDependencyPlan:
    return PageBuildOptions;
  case PageBuildOptions:
    return PageInstallLocation;
  case PageInstallLocation:
    return PageServiceOptions;
  case PageServiceOptions:
    return PageReview;
  case PageReview:
  case PageUninstall:
    return PageProgress;
  case PageProgress:
    return PageFinish;
  case PageFinish:
  default:
    return -1;
  }
}

void InstallerWizard::reject() {
  auto *progress = dynamic_cast<ProgressPage *>(page(PageProgress));
  if (currentId() == PageProgress && progress && progress->isRunning()) {
    progress->requestCancellation();
    return;
  }
  QWizard::reject();
}

void InstallerWizard::updatePreferenceProgressBars() {
  const bool uninstall = workflow_ == QStringLiteral("uninstall");
  for (int pageId = PageIntro; pageId <= PageReview; ++pageId) {
    auto *page = this->page(pageId);
    auto *bar = page ? page->findChild<QProgressBar *>(
                           QStringLiteral("scInstallerPreferenceProgress"))
                     : nullptr;
    if (!bar) {
      continue;
    }
    if (uninstall && pageId == PageIntro) {
      bar->setRange(0, 2);
      bar->setValue(1);
      bar->setAccessibleDescription(QStringLiteral("Step 1 of 2"));
    } else {
      bar->setRange(0, preferenceProgressMaximum());
      bar->setValue(preferenceProgressValue(pageId));
      bar->setAccessibleDescription(QStringLiteral("Step %1 of %2")
                                        .arg(preferenceProgressValue(pageId))
                                        .arg(preferenceProgressMaximum()));
    }
  }
}

} // namespace studiocast::installer
