#include "installer_wizard.h"

#include <QApplication>
#include <QBoxLayout>
#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QDateTime>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMap>
#include <QPlainTextEdit>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QSignalBlocker>
#include <QSizePolicy>
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
namespace {

QString defaultCacheBuildDir() {
  const QString cache =
      QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
  if (!cache.isEmpty()) {
    return QDir(cache).filePath(QStringLiteral("build"));
  }
  return QDir::home().filePath(QStringLiteral(".cache/studiocast/build"));
}

QString findBackendPath() {
  const QString envPath =
      QString::fromLocal8Bit(qgetenv("STUDIOCAST_INSTALLER_BACKEND"));
  if (!envPath.isEmpty() && QFileInfo(envPath).isExecutable()) {
    return envPath;
  }

  const QDir appDir(QCoreApplication::applicationDirPath());
  const QString installed =
      appDir.filePath(QStringLiteral("../share/studiocast/installer/"
                                     "studiocast-installer-backend"));
  if (QFileInfo(installed).isExecutable()) {
    return QFileInfo(installed).absoluteFilePath();
  }

  const QString sourcePath =
      QString::fromUtf8(STUDIOCAST_INSTALLER_BACKEND_SOURCE_PATH);
  if (!sourcePath.isEmpty() && QFileInfo(sourcePath).isExecutable()) {
    return sourcePath;
  }

  const QString relative =
      QDir(QString::fromUtf8(STUDIOCAST_SOURCE_DIR))
          .filePath(QStringLiteral("installer/backend/"
                                   "studiocast-installer-backend"));
  if (QFileInfo(relative).isExecutable()) {
    return relative;
  }

  return sourcePath;
}

QString findBundledSourceArchive() {
  const QDir appDir(QCoreApplication::applicationDirPath());
  const QString archiveName = QStringLiteral("StudioCast-%1-source.tar.gz")
                                  .arg(QStringLiteral(STUDIOCAST_VERSION));
  const QString installed = appDir.filePath(
      QStringLiteral("../share/studiocast/source/%1").arg(archiveName));
  if (QFileInfo(installed).isFile()) {
    return QFileInfo(installed).absoluteFilePath();
  }
  return QString();
}

QString jsonString(const QJsonObject &object, const QString &key,
                   const QString &fallback = QString()) {
  const QJsonValue value = object.value(key);
  return value.isString() ? value.toString() : fallback;
}

bool jsonBool(const QJsonObject &object, const QString &key,
              bool fallback = false) {
  const QJsonValue value = object.value(key);
  return value.isBool() ? value.toBool() : fallback;
}

QStringList jsonStringList(const QJsonObject &object, const QString &key) {
  QStringList out;
  const QJsonValue value = object.value(key);
  if (!value.isArray()) {
    return out;
  }
  const QJsonArray array = value.toArray();
  for (const QJsonValue &entry : array) {
    if (entry.isString()) {
      out.push_back(entry.toString());
    }
  }
  return out;
}

QString docLink(const QString &sourceDir, const QString &relativePath,
                const QString &label) {
  const QString path = QDir(sourceDir).filePath(relativePath);
  if (QFileInfo(path).isFile()) {
    return QStringLiteral("<a href=\"%1\">%2</a>")
        .arg(QUrl::fromLocalFile(path).toString().toHtmlEscaped(),
             label.toHtmlEscaped());
  }
  return QStringLiteral("%1 (%2)").arg(label.toHtmlEscaped(),
                                       relativePath.toHtmlEscaped());
}

QString optionalComponentsNoticeText(const QJsonObject &status,
                                     const QString &sourceDir) {
  const QJsonObject optional =
      status.value(QStringLiteral("optional_components")).toObject();
  if (optional.isEmpty()) {
    return QString();
  }

  QStringList lines;
  const QJsonObject cuda = optional.value(QStringLiteral("cuda")).toObject();
  const QJsonObject onnx =
      optional.value(QStringLiteral("onnxruntime_cuda")).toObject();
  const QJsonObject maxine =
      optional.value(QStringLiteral("maxine_sdk")).toObject();
  const QJsonObject vulkan =
      optional.value(QStringLiteral("vulkan")).toObject();

  const QString openDocs = docLink(
      sourceDir, QStringLiteral("docs/open_source_video_models_install.md"),
      QStringLiteral("Open Source backend setup instructions"));
  const QString maxineDocs =
      docLink(sourceDir, QStringLiteral("docs/maxine_install.md"),
              QStringLiteral("Maxine SDK install instructions"));
  const QString vulkanDocs =
      docLink(sourceDir, QStringLiteral("docs/SETUP.md"),
              QStringLiteral("Open Vulkan setup instructions"));

  if (!jsonBool(cuda, QStringLiteral("available"))) {
    lines << QStringLiteral(
                 "<b>NVIDIA CUDA driver/GPU not detected.</b> Open CUDA and "
                 "Maxine GPU effects need a working NVIDIA driver. See %1.")
                 .arg(openDocs);
  }

  if (!jsonBool(onnx, QStringLiteral("cuda_provider_present"))) {
    const bool ortPresent = jsonBool(onnx, QStringLiteral("present"));
    lines << (ortPresent
                  ? QStringLiteral(
                        "<b>ONNX Runtime CUDA provider not detected.</b> Open "
                        "CUDA needs an ONNX Runtime GPU build. See %1.")
                        .arg(openDocs)
                  : QStringLiteral(
                        "<b>ONNX Runtime not detected.</b> Open Video/Open "
                        "CUDA and Open Audio setup can install it through the "
                        "dependency step. See %1.")
                        .arg(openDocs));
  }

  if (!jsonBool(maxine, QStringLiteral("complete"))) {
    lines << QStringLiteral(
                 "<b>NVIDIA Maxine SDK assets not detected.</b> This is "
                 "optional and not redistributed by StudioCast. See %1.")
                 .arg(maxineDocs);
  }

  if (!jsonBool(vulkan, QStringLiteral("loader_present"))) {
    lines << QStringLiteral(
                 "<b>Vulkan loader not detected.</b> Open Vulkan is optional "
                 "and runtime-loaded. Select the Vulkan runtime packages "
                 "below and ensure a working GPU driver/ICD is installed. "
                 "See %1.")
                 .arg(vulkanDocs);
  } else if (!jsonBool(vulkan, QStringLiteral("vulkaninfo_present"))) {
    lines << QStringLiteral(
                 "<b>Vulkan diagnostics not detected.</b> The loader is "
                 "present, but the optional Vulkan runtime package selection "
                 "can add vulkaninfo. A working GPU driver/ICD is still "
                 "required. See %1.")
                 .arg(vulkanDocs);
  }

  if (lines.isEmpty()) {
    return QStringLiteral("CUDA, ONNX Runtime CUDA provider, Maxine SDK "
                          "assets, and Vulkan loader diagnostics were "
                          "detected. Vulkan device/ICD usability is checked "
                          "at runtime.");
  }

  return lines.join(QStringLiteral("<br>"));
}

QString sectionText(const QString &title, const QStringList &items) {
  if (items.isEmpty()) {
    return QString();
  }
  QString text = title + QStringLiteral("\n");
  for (const QString &item : items) {
    text += QStringLiteral("  - ") + item + QStringLiteral("\n");
  }
  text += QStringLiteral("\n");
  return text;
}

QString planTextFromObject(const QJsonObject &plan) {
  QString text;
  text += QStringLiteral("Action: %1\nRoute: %2\nPlan: %3\nDigest: %4\n\n")
              .arg(jsonString(plan, QStringLiteral("intent"),
                              jsonString(plan, QStringLiteral("workflow"),
                                         QStringLiteral("unknown"))),
                   jsonString(plan, QStringLiteral("route"),
                              QStringLiteral("unknown")),
                   jsonString(plan, QStringLiteral("plan_id"),
                              QStringLiteral("legacy")),
                   jsonString(plan, QStringLiteral("plan_digest"),
                              QStringLiteral("missing")));

  const QJsonObject desired =
      plan.value(QStringLiteral("desired_state")).toObject();
  if (!desired.isEmpty()) {
    text += QStringLiteral("Application files:\n  - User-local scope\n");
    text += QStringLiteral("  - Build type: %1\n")
                .arg(jsonString(desired, QStringLiteral("build_type"),
                                QStringLiteral("unknown")));
    const QJsonObject camera =
        desired.value(QStringLiteral("virtual_camera")).toObject();
    text += QStringLiteral("\nVirtual camera/system changes:\n  - %1\n")
                .arg(jsonBool(camera, QStringLiteral("desired"))
                         ? QStringLiteral("Required/selected StudioCast virtual camera")
                         : QStringLiteral("Virtual camera disabled; installation will be degraded"));
    const QJsonArray packs =
        desired.value(QStringLiteral("model_pack_ids")).toArray();
    text += QStringLiteral("\nModels/downloads (%1 pack IDs):\n").arg(packs.size());
    int artifactCount = 0;
    for (const QJsonValue &pack : packs) {
      text += QStringLiteral("  - %1\n").arg(pack.toString());
      artifactCount += pack.toString() ==
                               QStringLiteral("gaze_correction_cam_flx_v0_1_1")
                           ? 2
                           : 1;
    }
    text += QStringLiteral("  %1 verified artifact files expected\n")
                .arg(artifactCount);
  }

  QMap<QString, QStringList> operationsByCategory;
  const QJsonArray operations = plan.value(QStringLiteral("operations")).toArray();
  for (const QJsonValue &value : operations) {
    const QJsonObject operation = value.toObject();
    const QString category =
        jsonString(operation.value(QStringLiteral("review")).toObject(),
                   QStringLiteral("category"), QStringLiteral("other"));
    QString description = jsonString(operation, QStringLiteral("id"));
    if (jsonString(operation, QStringLiteral("privilege")) ==
        QStringLiteral("trusted_helper")) {
      description += QStringLiteral(" [privileged trusted helper]");
    }
    operationsByCategory[category].push_back(description);
  }
  for (auto it = operationsByCategory.cbegin(); it != operationsByCategory.cend();
       ++it) {
    text += QStringLiteral("\n%1:\n").arg(it.key());
    for (const QString &operation : it.value()) {
      text += QStringLiteral("  - %1\n").arg(operation);
    }
  }

  const auto codedItems = [&plan](const QString &key) {
    QStringList items;
    for (const QJsonValue &value : plan.value(key).toArray()) {
      const QJsonObject item = value.toObject();
      items.push_back(jsonString(item, QStringLiteral("code"),
                                 QStringLiteral("unknown")));
    }
    return items;
  };
  text += sectionText(QStringLiteral("BLOCKERS:"),
                      codedItems(QStringLiteral("blockers")));
  text += sectionText(QStringLiteral("Warnings:"),
                      codedItems(QStringLiteral("warnings")));
  text += sectionText(QStringLiteral("Summary:"),
                      jsonStringList(plan, QStringLiteral("summary")));
  text += sectionText(QStringLiteral("Changes:"),
                      jsonStringList(plan, QStringLiteral("changes")));
  text += sectionText(
      QStringLiteral("Privileged operations:"),
      jsonStringList(plan, QStringLiteral("privileged_operations")));
  text += sectionText(QStringLiteral("Preserved:"),
                      jsonStringList(plan, QStringLiteral("preserved")));
  text += sectionText(QStringLiteral("Removed:"),
                      jsonStringList(plan, QStringLiteral("removed")));
  text += sectionText(QStringLiteral("Backend commands:"),
                      jsonStringList(plan, QStringLiteral("commands")));
  return text.trimmed();
}

QJsonObject lastJsonObject(const QByteArray &bytes) {
  for (qsizetype offset = bytes.size() - 1; offset >= 0; --offset) {
    if (bytes.at(offset) != '{') {
      continue;
    }
    QJsonParseError error;
    const QJsonDocument document = QJsonDocument::fromJson(bytes.mid(offset), &error);
    if (error.error == QJsonParseError::NoError && document.isObject()) {
      return document.object();
    }
  }
  return {};
}

QString actionLabel(const QString &intent) {
  if (intent == QStringLiteral("install"))
    return QStringLiteral("Install");
  if (intent == QStringLiteral("repair"))
    return QStringLiteral("Repair");
  if (intent == QStringLiteral("reinstall") ||
      intent == QStringLiteral("clean-install"))
    return QStringLiteral("Reinstall");
  if (intent == QStringLiteral("uninstall"))
    return QStringLiteral("Uninstall");
  return QStringLiteral("Update");
}

QLabel *mutedLabel(const QString &text, QWidget *parent = nullptr) {
  auto *label = new QLabel(text, parent);
  label->setWordWrap(true);
  label->setProperty("scRole", "muted");
  return label;
}

QFrame *line(QWidget *parent = nullptr) {
  auto *frame = new QFrame(parent);
  frame->setProperty("scRole", "separator");
  frame->setFrameShape(QFrame::HLine);
  frame->setFrameShadow(QFrame::Sunken);
  return frame;
}

InstallerWizard *installerWizard(QWizardPage *page) {
  return dynamic_cast<InstallerWizard *>(page->window());
}

void setReadOnlyLogStyle(QPlainTextEdit *edit) {
  edit->setReadOnly(true);
  edit->setLineWrapMode(QPlainTextEdit::NoWrap);
  QFont font = edit->font();
  font.setFamily(QStringLiteral("monospace"));
  edit->setFont(font);
}

int preferenceProgressMaximum() { return PageReview - PageIntro + 1; }

int preferenceProgressValue(int pageId) {
  return std::clamp(pageId - PageIntro + 1, 1, preferenceProgressMaximum());
}

void addPreferenceProgressBar(QWizardPage *page, int progressValue,
                              int progressMaximum) {
  auto *layout = qobject_cast<QBoxLayout *>(page->layout());
  if (!layout) {
    return;
  }

  auto *bar = new QProgressBar(page);
  bar->setObjectName(QStringLiteral("scInstallerPreferenceProgress"));
  bar->setProperty("scRole", "installerPreferenceProgress");
  bar->setRange(0, progressMaximum);
  bar->setValue(progressValue);
  bar->setTextVisible(false);
  bar->setFocusPolicy(Qt::NoFocus);
  bar->setFixedHeight(2);
  bar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  bar->setAccessibleName(QStringLiteral("Installer preference progress"));
  bar->setAccessibleDescription(
      QStringLiteral("Step %1 of %2").arg(progressValue).arg(progressMaximum));
  layout->insertWidget(0, bar);
}

void addPreferenceProgressBars(QWizard *wizard) {
  for (int pageId = PageIntro; pageId <= PageReview; ++pageId) {
    if (auto *page = wizard->page(pageId)) {
      addPreferenceProgressBar(page, preferenceProgressValue(pageId),
                               preferenceProgressMaximum());
    }
  }
  if (auto *page = wizard->page(PageUninstall)) {
    addPreferenceProgressBar(page, 2, 2);
  }
}

} // namespace

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
bool InstallerWizard::openBackendsSetup() const { return openBackendsSetup_; }
bool InstallerWizard::openVulkan() const { return openVulkan_; }
bool InstallerWizard::installVulkanRuntime() const {
  return installVulkanRuntime_;
}
bool InstallerWizard::installMesaVulkan() const { return installMesaVulkan_; }
bool InstallerWizard::installShaderTools() const { return installShaderTools_; }
bool InstallerWizard::installModels() const { return installModels_; }
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
  if (configureV4l2Loopback_ != enabled)
    invalidatePlan();
  configureV4l2Loopback_ = enabled;
}
void InstallerWizard::setLoadLoopback(bool enabled) {
  if (loadLoopback_ != enabled)
    invalidatePlan();
  loadLoopback_ = enabled;
}
void InstallerWizard::setPersistLoopback(bool enabled) {
  if (persistLoopback_ != enabled)
    invalidatePlan();
  persistLoopback_ = enabled;
}
void InstallerWizard::setInstallService(bool enabled) {
  if (installService_ != enabled)
    invalidatePlan();
  installService_ = enabled;
}
void InstallerWizard::setOpenBackendsSetup(bool enabled) {
  if (openBackendsSetup_ != enabled)
    invalidatePlan();
  openBackendsSetup_ = enabled;
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

  QString onlineError;
  QJsonObject releaseStatus;
  QStringList args{QStringLiteral("release-status"), QStringLiteral("--json"),
                   QStringLiteral("--release-receipt-out"), receiptPath};
  args += releaseChannelOptions(false);
  bool verified = runBackendJson(args, &releaseStatus, &onlineError);
  QString offlineError;
  if (!verified) {
    args = {QStringLiteral("release-status"), QStringLiteral("--json"),
            QStringLiteral("--release-receipt-out"), receiptPath};
    args += releaseChannelOptions(true);
    verified = runBackendJson(args, &releaseStatus, &offlineError);
  }

  if (!verified ||
      releaseStatus.value(QStringLiteral("schema_version")).toInt() != 1 ||
      releaseStatus.value(QStringLiteral("release_status_version"))
              .toString() != QStringLiteral("installer-release-status/v1") ||
      releaseStatus.value(QStringLiteral("available_version"))
          .toString()
          .isEmpty() ||
      !QFileInfo(receiptPath).isFile() ||
      !QFileInfo(releaseStatus.value(QStringLiteral("verified_source_archive"))
                     .toString())
           .isFile()) {
    QFile::remove(receiptPath);
    if (error) {
      if (!offlineError.trimmed().isEmpty()) {
        *error =
            QStringLiteral("Stable release verification failed online (%1) "
                           "and no verified offline cache was usable (%2).")
                .arg(onlineError.trimmed(), offlineError.trimmed());
      } else if (!onlineError.trimmed().isEmpty()) {
        *error = onlineError.trimmed();
      } else {
        *error = QStringLiteral("Stable release status or verified archive "
                                "evidence was incomplete.");
      }
    }
    return false;
  }

  QFile(receiptPath)
      .setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner);
  verifiedReleaseReceiptPath_ = receiptPath;
  releaseStatusObject_ = releaseStatus;
  releaseArchive_ =
      releaseStatus.value(QStringLiteral("verified_source_archive")).toString();
  useReleaseArchive_ = true;
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
    openBackendsSetup_ =
        jsonBool(features, QStringLiteral("open_cuda"), openBackendsSetup_) ||
        jsonBool(features, QStringLiteral("open_audio"), openBackendsSetup_);
    openVulkan_ = jsonBool(features, QStringLiteral("open_vulkan"), false);
    const QJsonObject service = prior.value(QStringLiteral("service")).toObject();
    installService_ =
        jsonString(service, QStringLiteral("desired")) ==
        QStringLiteral("enabled_started");
    const QJsonObject camera = prior.value(QStringLiteral("v4l")).toObject();
    configureV4l2Loopback_ =
        jsonBool(camera, QStringLiteral("desired"), configureV4l2Loopback_);
    loadLoopback_ = configureV4l2Loopback_;
    persistLoopback_ = jsonBool(camera, QStringLiteral("persist"),
                                persistLoopback_);
    installModels_ = !prior.value(QStringLiteral("model_pack_ids"))
                          .toArray()
                          .isEmpty();
    removeUserData_ =
        !jsonBool(prior, QStringLiteral("preserve_settings"), true);
  } else {
    installModels_ = true;
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
  if (!customRoute_ && workflow_ != QStringLiteral("repair") &&
      workflow_ != QStringLiteral("uninstall") &&
      (verifiedReleaseReceiptPath_.isEmpty() ||
       !QFileInfo(verifiedReleaseReceiptPath_).isFile() ||
       !useReleaseArchive_ || releaseArchive_.isEmpty() ||
       !QFileInfo(releaseArchive_).isFile())) {
    planError_ = QStringLiteral(
        "Recommended installation requires a current signed-release receipt "
        "and its verified official source archive.");
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
  if (needsSource) {
    if (!customRoute_ && !verifiedReleaseReceiptPath_.isEmpty()) {
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
    if (!buildType_.isEmpty()) {
      args << QStringLiteral("--build-type") << buildType_;
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

  if (configureV4l2Loopback_) {
    args << QStringLiteral("--v4l2loopback");
  } else {
    args << QStringLiteral("--no-v4l2loopback");
  }
  args << (loadLoopback_ ? QStringLiteral("--load-loopback")
                         : QStringLiteral("--no-load-loopback"));
  args << (persistLoopback_ ? QStringLiteral("--persist-loopback")
                            : QStringLiteral("--no-persist-loopback"));
  args << (installService_ ? QStringLiteral("--service")
                           : QStringLiteral("--no-service"));
  args << (openBackendsSetup_ ? QStringLiteral("--open-backends")
                              : QStringLiteral("--no-open-backends"));
  args << (openVulkan_ ? QStringLiteral("--open-vulkan")
                       : QStringLiteral("--no-open-vulkan"));
  args << (installVulkanRuntime_ ? QStringLiteral("--vulkan-runtime")
                                 : QStringLiteral("--no-vulkan-runtime"));
  if (installMesaVulkan_) {
    args << QStringLiteral("--mesa-vulkan");
  }
  if (installShaderTools_) {
    args << QStringLiteral("--shader-tools");
  }
  args << (installModels_ ? QStringLiteral("--models")
                          : QStringLiteral("--no-models"));
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

IntroPage::IntroPage(QWidget *parent) : QWizardPage(parent) {
  setTitle(QStringLiteral("StudioCast Installer"));
  setSubTitle(QStringLiteral("Choose an action based on the analyzed installation state."));

  auto *layout = new QVBoxLayout(this);
  auto *statusBox = new QGroupBox(QStringLiteral("Detected system"), this);
  auto *form = new QFormLayout(statusBox);
  osValue_ = new QLabel(statusBox);
  osValue_->setWordWrap(true);
  versionValue_ = new QLabel(statusBox);
  installValue_ = new QLabel(statusBox);
  installValue_->setWordWrap(true);
  form->addRow(QStringLiteral("OS"), osValue_);
  form->addRow(QStringLiteral("StudioCast version"), versionValue_);
  form->addRow(QStringLiteral("Install status"), installValue_);
  layout->addWidget(statusBox);

  auto *workflowBox = new QGroupBox(QStringLiteral("Available actions"), this);
  auto *workflowLayout = new QVBoxLayout();
  workflowGroup_ = new QButtonGroup(workflowBox);
  const auto addAction = [this, workflowBox,
                          workflowLayout](const QString &name) {
    auto *radio = new QRadioButton(workflowBox);
    radio->setObjectName(name);
    workflowGroup_->addButton(radio);
    workflowLayout->addWidget(radio);
    connect(radio, &QRadioButton::toggled, this, [this, radio](bool checked) {
      if (checked) {
        if (auto *w = installerWizard(this)) {
          w->setWorkflow(radio->property("workflow").toString());
          w->setCustomRoute(radio->property("customRoute").toBool());
          if (radio->property("recommended").toBool()) {
            w->setBuildType(QStringLiteral("Release"));
            w->setConfigureV4l2Loopback(true);
            w->setLoadLoopback(true);
            w->setPersistLoopback(true);
            w->setInstallModels(true);
            w->setOpenVulkan(false);
            w->setInstallVulkanRuntime(false);
            w->setInstallMesaVulkan(false);
            w->setInstallShaderTools(false);
            w->setInstallService(jsonBool(
                w->factsObject()
                    .value(QStringLiteral("systemd_user"))
                    .toObject(),
                QStringLiteral("manager_usable"), true));
          }
        }
        emit completeChanged();
      }
    });
    return radio;
  };
  primaryAction_ = addAction(QStringLiteral("scInstallerPrimaryAction"));
  customizeAction_ = addAction(QStringLiteral("scInstallerCustomizeAction"));
  repairAction_ = addAction(QStringLiteral("scInstallerRepairAction"));
  reinstallAction_ = addAction(QStringLiteral("scInstallerReinstallAction"));
  uninstallAction_ = addAction(QStringLiteral("scInstallerUninstallAction"));
  workflowBox->setLayout(workflowLayout);
  layout->addWidget(workflowBox);

  layout->addWidget(mutedLabel(
      QStringLiteral("The GUI is not run as root. Privileged operations are "
                     "delegated only through the trusted, typed StudioCast "
                     "system helper when the exact reviewed plan requires it. "
                     "Application files always remain user-local."),
      this));
  layout->addStretch(1);
}

void IntroPage::initializePage() {
  auto *w = installerWizard(this);
  w->refreshStatus();
  const QJsonObject status = w->statusObject();
  const QJsonObject os = w->osObject();

  if (!w->analysisAvailable()) {
    osValue_->setText(QStringLiteral("Analysis unavailable"));
    versionValue_->setText(QStringLiteral("Unknown"));
    installValue_->setText(w->analysisError());
    for (QAbstractButton *button : workflowGroup_->buttons()) {
      button->setVisible(false);
    }
    emit completeChanged();
    return;
  }

  osValue_->setText(QStringLiteral("%1 (base %2 %3)")
                        .arg(jsonString(os, QStringLiteral("pretty_name"),
                                        QStringLiteral("unknown")),
                             jsonString(os, QStringLiteral("base_id"),
                                        QStringLiteral("unknown")),
                             jsonString(os, QStringLiteral("base_version_id"),
                                        QStringLiteral("unknown"))));

  const QString installedVersion =
      jsonString(status, QStringLiteral("installed_version"));
  versionValue_->setText(
      QStringLiteral("Available %1, installed %2")
          .arg(jsonString(status, QStringLiteral("target_version"),
                          QStringLiteral(STUDIOCAST_VERSION)),
               installedVersion.isEmpty() ? QStringLiteral("not installed")
                                          : installedVersion));

  QString installText = QStringLiteral("%1 (%2)").arg(
      jsonString(status, QStringLiteral("classification"),
                 QStringLiteral("unknown")),
      jsonString(status, QStringLiteral("version_relation"),
                 QStringLiteral("unknown")));
  const QString releaseError =
      jsonString(status, QStringLiteral("release_error"));
  if (!releaseError.isEmpty()) {
    installText +=
        QStringLiteral("\nStable release unavailable: %1").arg(releaseError);
  }
  installValue_->setText(installText);

  const QString route = w->detectedRoute();
  const QString action = w->primaryAction();
  QString primaryWorkflow;
  QString primaryText;
  bool primaryCustom = false;
  if (action == QStringLiteral("install")) {
    primaryWorkflow = QStringLiteral("install");
    primaryText = QStringLiteral("Install recommended");
  } else if (action == QStringLiteral("update")) {
    primaryWorkflow = QStringLiteral("update");
    primaryText = QStringLiteral("Update recommended");
  } else if (action == QStringLiteral("modify")) {
    primaryWorkflow = QStringLiteral("modify");
    primaryText = QStringLiteral("Modify installation");
    primaryCustom = true;
  } else if (action == QStringLiteral("repair") ||
             action == QStringLiteral("reconstruct")) {
    primaryWorkflow = QStringLiteral("repair");
    primaryText = action == QStringLiteral("reconstruct")
                      ? QStringLiteral("Reconstruct/repair")
                      : QStringLiteral("Repair recommended");
  } else if (action == QStringLiteral("keep")) {
    primaryWorkflow = QStringLiteral("keep");
    primaryText = QStringLiteral("Keep current (no changes)");
  } else {
    primaryWorkflow = QStringLiteral("diagnostics");
    primaryText = route == QStringLiteral("unsupported")
                      ? QStringLiteral("View diagnostics and manual instructions")
                      : route == QStringLiteral("offline")
                            ? QStringLiteral("Required verified cache unavailable")
                            : QStringLiteral("Retry analysis");
  }
  primaryAction_->setText(primaryText);
  primaryAction_->setProperty("workflow", primaryWorkflow);
  primaryAction_->setProperty("customRoute", primaryCustom);
  primaryAction_->setProperty("recommended", !primaryCustom &&
                                                 primaryWorkflow !=
                                                     QStringLiteral("keep") &&
                                                 primaryWorkflow !=
                                                     QStringLiteral("diagnostics"));
  primaryAction_->setVisible(true);

  const QString classification =
      jsonString(status, QStringLiteral("classification"));
  const bool automaticAllowed = route != QStringLiteral("unsupported") &&
                                route != QStringLiteral("offline") &&
                                route != QStringLiteral("analysis_error");
  const bool installed = jsonBool(status, QStringLiteral("installed"));
  const QString customizableWorkflow =
      primaryWorkflow == QStringLiteral("keep") ? QStringLiteral("modify")
                                                 : primaryWorkflow;
  customizeAction_->setText(QStringLiteral("Customize / Advanced"));
  customizeAction_->setProperty("workflow", customizableWorkflow);
  customizeAction_->setProperty("customRoute", true);
  customizeAction_->setVisible(automaticAllowed &&
                               customizableWorkflow != QStringLiteral("diagnostics"));
  repairAction_->setText(QStringLiteral("Repair"));
  repairAction_->setProperty("workflow", QStringLiteral("repair"));
  repairAction_->setProperty("customRoute", false);
  repairAction_->setVisible(automaticAllowed && installed &&
                            classification != QStringLiteral("partial"));
  reinstallAction_->setText(QStringLiteral("Reinstall / reset"));
  reinstallAction_->setProperty("workflow", QStringLiteral("reinstall"));
  reinstallAction_->setProperty("customRoute", false);
  reinstallAction_->setVisible(automaticAllowed && installed);
  uninstallAction_->setText(QStringLiteral("Uninstall"));
  uninstallAction_->setProperty("workflow", QStringLiteral("uninstall"));
  uninstallAction_->setProperty("customRoute", false);
  uninstallAction_->setVisible(automaticAllowed && installed);

  primaryAction_->setChecked(true);
  emit completeChanged();
}

bool IntroPage::validatePage() {
  auto *w = installerWizard(this);
  const QAbstractButton *button = workflowGroup_->checkedButton();
  if (button) {
    w->setWorkflow(button->property("workflow").toString());
  }
  return isComplete();
}

bool IntroPage::isComplete() const {
  const auto *w = dynamic_cast<const InstallerWizard *>(window());
  const QAbstractButton *button = workflowGroup_->checkedButton();
  if (!w || !w->analysisAvailable() || !button) {
    return false;
  }
  const QString selected = button->property("workflow").toString();
  return selected != QStringLiteral("diagnostics") &&
         selected != QStringLiteral("keep");
}

CompatibilityPage::CompatibilityPage(QWidget *parent) : QWizardPage(parent) {
  setTitle(QStringLiteral("OS Compatibility"));
  setSubTitle(QStringLiteral("The installer supports Ubuntu 22.04, Ubuntu "
                             "24.04, and practical Linux Mint mappings."));

  auto *layout = new QVBoxLayout(this);
  statusLabel_ = new QLabel(this);
  statusLabel_->setWordWrap(true);
  layout->addWidget(statusLabel_);

  details_ = new QPlainTextEdit(this);
  setReadOnlyLogStyle(details_);
  details_->setMinimumHeight(220);
  layout->addWidget(details_, 1);

  allowUnsupported_ = new QCheckBox(
      QStringLiteral("Unsupported overrides are CLI/developer-only"), this);
  allowUnsupported_->setEnabled(false);
  connect(allowUnsupported_, &QCheckBox::toggled, this, [this](bool checked) {
    installerWizard(this)->setAllowUnsupported(checked);
    emit completeChanged();
  });
  layout->addWidget(allowUnsupported_);
}

void CompatibilityPage::initializePage() {
  auto *w = installerWizard(this);
  const QJsonObject os = w->osObject();
  const bool supported = jsonBool(os, QStringLiteral("supported"));
  allowUnsupported_->setChecked(w->allowUnsupported());
  allowUnsupported_->setVisible(!supported);
  statusLabel_->setText(
      supported ? QStringLiteral("This OS is supported by the installer.")
                : QStringLiteral("This OS is not supported by the installer. "
                                 "Manual source-build instructions remain "
                                 "available in docs/SETUP.md."));

  QString text;
  text +=
      QStringLiteral("Detected OS: ") +
      jsonString(os, QStringLiteral("pretty_name"), QStringLiteral("unknown")) +
      QStringLiteral("\n");
  text += QStringLiteral("ID: ") + jsonString(os, QStringLiteral("id")) +
          QStringLiteral("\n");
  text += QStringLiteral("VERSION_ID: ") +
          jsonString(os, QStringLiteral("version_id")) + QStringLiteral("\n");
  text += QStringLiteral("VERSION_CODENAME: ") +
          jsonString(os, QStringLiteral("version_codename")) +
          QStringLiteral("\n");
  text += QStringLiteral("UBUNTU_CODENAME: ") +
          jsonString(os, QStringLiteral("ubuntu_codename")) +
          QStringLiteral("\n");
  text += QStringLiteral("ID_LIKE: ") +
          jsonString(os, QStringLiteral("id_like")) + QStringLiteral("\n");
  text += QStringLiteral("Mapped Ubuntu base: ") +
          jsonString(os, QStringLiteral("base_version_id"),
                     QStringLiteral("unknown")) +
          QStringLiteral(" / ") +
          jsonString(os, QStringLiteral("base_codename"),
                     QStringLiteral("unknown")) +
          QStringLiteral("\n");
  text += QStringLiteral("Mapping: ") +
          jsonString(os, QStringLiteral("mapping_note")) + QStringLiteral("\n");
  text += QStringLiteral("Support: ") +
          jsonString(os, QStringLiteral("support_reason"));
  details_->setPlainText(text);
  emit completeChanged();
}

bool CompatibilityPage::isComplete() const {
  const auto *w = dynamic_cast<const InstallerWizard *>(window());
  if (!w) {
    return false;
  }
  return w->analysisAvailable() &&
         jsonBool(w->osObject(), QStringLiteral("supported"));
}

DependencyPlanPage::DependencyPlanPage(QWidget *parent) : QWizardPage(parent) {
  setTitle(QStringLiteral("Dependency Plan"));
  setSubTitle(
      QStringLiteral("Review package/module work before anything runs."));
  auto *layout = new QVBoxLayout(this);
  planText_ = new QPlainTextEdit(this);
  setReadOnlyLogStyle(planText_);
  layout->addWidget(planText_, 1);
}

void DependencyPlanPage::initializePage() {
  auto *w = installerWizard(this);
  QString error;
  if (w->refreshPlan(&error)) {
    planText_->setPlainText(planTextFromObject(w->planObject()));
  } else {
    planText_->setPlainText(error);
  }
}

UninstallPage::UninstallPage(QWidget *parent) : QWizardPage(parent) {
  setTitle(QStringLiteral("Uninstall StudioCast"));
  setSubTitle(QStringLiteral(
      "Review what will be removed before uninstalling StudioCast."));
  setCommitPage(true);

  auto *layout = new QVBoxLayout(this);
  statusLabel_ = new QLabel(this);
  statusLabel_->setObjectName(QStringLiteral("scInstallerUninstallStatus"));
  statusLabel_->setWordWrap(true);
  layout->addWidget(statusLabel_);

  removeUserData_ = new QCheckBox(
      QStringLiteral("Also remove user config, downloaded models, logs, and "
                     "cache"),
      this);
  removeUserData_->setObjectName(
      QStringLiteral("scInstallerUninstallRemoveUserData"));
  layout->addWidget(removeUserData_);

  dataWarning_ = mutedLabel(
      QStringLiteral("The additional user-data removal is permanent. Leave "
                     "this unchecked to preserve your settings and downloads."),
      this);
  dataWarning_->setObjectName(
      QStringLiteral("scInstallerUninstallDataWarning"));
  layout->addWidget(dataWarning_);

  auto *planBox = new QGroupBox(QStringLiteral("Uninstall summary"), this);
  auto *planLayout = new QVBoxLayout(planBox);
  planText_ = new QPlainTextEdit(planBox);
  planText_->setObjectName(QStringLiteral("scInstallerUninstallPlan"));
  setReadOnlyLogStyle(planText_);
  planLayout->addWidget(planText_);
  layout->addWidget(planBox, 1);

  connect(removeUserData_, &QCheckBox::toggled, this, [this](bool checked) {
    auto *w = installerWizard(this);
    w->setRemoveUserData(checked);
    dataWarning_->setVisible(checked);
    refreshPlanText();
  });
}

void UninstallPage::initializePage() {
  auto *w = installerWizard(this);
  w->setButtonText(QWizard::CommitButton, QStringLiteral("Uninstall"));

  const QJsonObject status = w->statusObject();
  const QString installedVersion =
      jsonString(status, QStringLiteral("installed_version"));
  if (jsonBool(status, QStringLiteral("installed"))) {
    statusLabel_->setText(
        QStringLiteral("StudioCast %1 is installed and ready to be removed.")
            .arg(installedVersion.isEmpty()
                     ? QStringLiteral("(version unknown)")
                     : installedVersion));
  } else {
    statusLabel_->setText(QStringLiteral(
        "No installer-managed StudioCast installation was detected. "
        "Continuing will clean up any remaining app links, user service, "
        "desktop entry, and runtime artifacts."));
  }

  {
    const QSignalBlocker blocker(removeUserData_);
    removeUserData_->setChecked(w->removeUserData());
  }
  dataWarning_->setVisible(removeUserData_->isChecked());
  refreshPlanText();
}

bool UninstallPage::validatePage() {
  installerWizard(this)->setRemoveUserData(removeUserData_->isChecked());
  return installerWizard(this)->planReady();
}

bool UninstallPage::isComplete() const {
  const auto *w = dynamic_cast<const InstallerWizard *>(window());
  return w && w->planReady();
}

void UninstallPage::refreshPlanText() {
  auto *w = installerWizard(this);
  QString error;
  if (w->refreshPlan(&error)) {
    planText_->setPlainText(planTextFromObject(w->planObject()));
  } else {
    planText_->setPlainText(error);
  }
  emit completeChanged();
}

BuildOptionsPage::BuildOptionsPage(QWidget *parent) : QWizardPage(parent) {
  setTitle(QStringLiteral("Build Options"));
  setSubTitle(
      QStringLiteral("Choose the source, build type, and build cache."));

  auto *layout = new QVBoxLayout(this);
  auto *sourceBox = new QGroupBox(QStringLiteral("Source"), this);
  auto *sourceLayout = new QVBoxLayout(sourceBox);

  sourceDirRadio_ = new QRadioButton(
      QStringLiteral("Build from source directory"), sourceBox);
  archiveRadio_ = new QRadioButton(
      QStringLiteral("Build from selected release archive"), sourceBox);
  sourceDirRadio_->setChecked(true);
  sourceLayout->addWidget(sourceDirRadio_);

  auto *sourceRow = new QHBoxLayout;
  sourceDirEdit_ = new QLineEdit(sourceBox);
  auto *browseSource = new QPushButton(
      style()->standardIcon(QStyle::SP_DirOpenIcon), QString(), sourceBox);
  browseSource->setToolTip(QStringLiteral("Choose source directory"));
  sourceRow->addWidget(sourceDirEdit_, 1);
  sourceRow->addWidget(browseSource);
  sourceLayout->addLayout(sourceRow);

  sourceLayout->addWidget(archiveRadio_);
  auto *archiveRow = new QHBoxLayout;
  archiveEdit_ = new QLineEdit(sourceBox);
  auto *browseArchive = new QPushButton(
      style()->standardIcon(QStyle::SP_FileIcon), QString(), sourceBox);
  browseArchive->setToolTip(QStringLiteral("Choose release archive"));
  archiveRow->addWidget(archiveEdit_, 1);
  archiveRow->addWidget(browseArchive);
  sourceLayout->addLayout(archiveRow);
  layout->addWidget(sourceBox);

  connect(browseSource, &QPushButton::clicked, this, [this] {
    const QString path = QFileDialog::getExistingDirectory(
        this, QStringLiteral("Choose StudioCast source directory"),
        sourceDirEdit_->text());
    if (!path.isEmpty()) {
      sourceDirEdit_->setText(path);
    }
  });
  connect(browseArchive, &QPushButton::clicked, this, [this] {
    const QString path = QFileDialog::getOpenFileName(
        this, QStringLiteral("Choose StudioCast release archive"),
        QDir::homePath(),
        QStringLiteral("Archives (*.tar *.tar.gz *.tgz *.tar.xz *.txz *.zip);;"
                       "All files (*)"));
    if (!path.isEmpty()) {
      archiveEdit_->setText(path);
    }
  });

  auto *buildBox = new QGroupBox(QStringLiteral("Build"), this);
  auto *form = new QFormLayout(buildBox);
  buildDirEdit_ = new QLineEdit(buildBox);
  auto *buildDirRow = new QWidget(buildBox);
  auto *buildDirLayout = new QHBoxLayout(buildDirRow);
  buildDirLayout->setContentsMargins(0, 0, 0, 0);
  auto *browseBuild = new QPushButton(
      style()->standardIcon(QStyle::SP_DirOpenIcon), QString(), buildDirRow);
  browseBuild->setToolTip(QStringLiteral("Choose build directory"));
  buildDirLayout->addWidget(buildDirEdit_, 1);
  buildDirLayout->addWidget(browseBuild);
  form->addRow(QStringLiteral("Build directory"), buildDirRow);

  buildTypeCombo_ = new QComboBox(buildBox);
  buildTypeCombo_->addItems({QStringLiteral("Release"),
                             QStringLiteral("RelWithDebInfo"),
                             QStringLiteral("Debug")});
  form->addRow(QStringLiteral("Build type"), buildTypeCombo_);

  installDeps_ = new QCheckBox(
      QStringLiteral("Install or refresh build/runtime dependencies"),
      buildBox);
  freshBuild_ = new QCheckBox(
      QStringLiteral("Remove the selected build directory first"), buildBox);
  form->addRow(QString(), installDeps_);
  form->addRow(QString(), freshBuild_);
  layout->addWidget(buildBox);
  layout->addStretch(1);

  connect(browseBuild, &QPushButton::clicked, this, [this] {
    const QString path = QFileDialog::getExistingDirectory(
        this, QStringLiteral("Choose build directory"), buildDirEdit_->text());
    if (!path.isEmpty()) {
      buildDirEdit_->setText(path);
    }
  });
}

void BuildOptionsPage::initializePage() {
  auto *w = installerWizard(this);
  const bool sourceRelevant = w->workflow() != QStringLiteral("uninstall") &&
                              w->workflow() != QStringLiteral("advanced");
  sourceDirRadio_->setEnabled(sourceRelevant);
  archiveRadio_->setEnabled(sourceRelevant);
  sourceDirEdit_->setEnabled(sourceRelevant);
  archiveEdit_->setEnabled(sourceRelevant);
  buildDirEdit_->setEnabled(sourceRelevant);
  buildTypeCombo_->setEnabled(sourceRelevant);
  installDeps_->setEnabled(sourceRelevant);
  freshBuild_->setEnabled(sourceRelevant);

  sourceDirRadio_->setChecked(!w->useReleaseArchive());
  archiveRadio_->setChecked(w->useReleaseArchive());
  sourceDirEdit_->setText(w->sourceDir());
  archiveEdit_->setText(w->releaseArchive());
  buildDirEdit_->setText(w->buildDir());
  buildTypeCombo_->setCurrentText(w->buildType());
  installDeps_->setChecked(w->installDeps());
  freshBuild_->setChecked(w->freshBuild());
}

bool BuildOptionsPage::validatePage() {
  auto *w = installerWizard(this);
  const bool sourceRelevant = w->workflow() != QStringLiteral("uninstall") &&
                              w->workflow() != QStringLiteral("advanced");
  if (sourceRelevant) {
    if (archiveRadio_->isChecked()) {
      if (archiveEdit_->text().trimmed().isEmpty() ||
          !QFileInfo(archiveEdit_->text().trimmed()).isFile()) {
        QMessageBox::warning(this, QStringLiteral("StudioCast Installer"),
                             QStringLiteral("Choose a valid release archive."));
        return false;
      }
    } else if (sourceDirEdit_->text().trimmed().isEmpty() ||
               !QFileInfo(sourceDirEdit_->text().trimmed()).isDir()) {
      QMessageBox::warning(this, QStringLiteral("StudioCast Installer"),
                           QStringLiteral("Choose a valid source directory."));
      return false;
    }
    if (buildDirEdit_->text().trimmed().isEmpty()) {
      QMessageBox::warning(this, QStringLiteral("StudioCast Installer"),
                           QStringLiteral("Choose a build directory."));
      return false;
    }
  }
  w->setUseReleaseArchive(archiveRadio_->isChecked());
  w->setSourceDir(sourceDirEdit_->text().trimmed());
  w->setReleaseArchive(archiveEdit_->text().trimmed());
  w->setBuildDir(buildDirEdit_->text().trimmed());
  w->setBuildType(buildTypeCombo_->currentText());
  w->setInstallDeps(installDeps_->isChecked());
  w->setFreshBuild(freshBuild_->isChecked());
  return true;
}

InstallLocationPage::InstallLocationPage(QWidget *parent)
    : QWizardPage(parent) {
  setTitle(QStringLiteral("Install Location"));
  setSubTitle(
      QStringLiteral("StudioCast installs as user-level app files and a "
                     "systemd user service."));

  auto *layout = new QVBoxLayout(this);
  auto *box = new QGroupBox(QStringLiteral("Paths"), this);
  auto *form = new QFormLayout(box);
  binaryLocation_ = new QLabel(box);
  binaryLocation_->setWordWrap(true);
  serviceLocation_ = new QLabel(box);
  serviceLocation_->setWordWrap(true);
  manifestLocation_ = new QLabel(box);
  manifestLocation_->setWordWrap(true);
  buildLocation_ = new QLabel(box);
  buildLocation_->setWordWrap(true);
  form->addRow(QStringLiteral("Binaries"), binaryLocation_);
  form->addRow(QStringLiteral("User service"), serviceLocation_);
  form->addRow(QStringLiteral("Manifest"), manifestLocation_);
  form->addRow(QStringLiteral("Build cache"), buildLocation_);
  layout->addWidget(box);
  layout->addWidget(mutedLabel(
      QStringLiteral("Uninstall removes app symlinks, the StudioCast user "
                     "service, and runtime artifacts. User config, downloaded "
                     "model packs, logs, and cache are preserved unless the "
                     "review page explicitly says they will be removed."),
      this));
  layout->addStretch(1);
}

void InstallLocationPage::initializePage() {
  const QJsonObject status = installerWizard(this)->statusObject();
  binaryLocation_->setText(
      jsonString(status, QStringLiteral("local_bin_dir"),
                 QDir::home().filePath(QStringLiteral(".local/bin"))));
  serviceLocation_->setText(
      jsonString(status.value(QStringLiteral("service")).toObject(),
                 QStringLiteral("path"),
                 QDir::home().filePath(QStringLiteral(
                     ".config/systemd/user/studiocastd.service"))));
  manifestLocation_->setText(
      jsonString(status, QStringLiteral("manifest_path"),
                 QDir::home().filePath(QStringLiteral(
                     ".local/share/studiocast/install-manifest.json"))));
  buildLocation_->setText(installerWizard(this)->buildDir());
}

ServiceOptionsPage::ServiceOptionsPage(QWidget *parent) : QWizardPage(parent) {
  setTitle(QStringLiteral("Service and Optional Components"));
  setSubTitle(
      QStringLiteral("Choose runtime integration and optional downloads."));
  auto *layout = new QVBoxLayout(this);

  installService_ = new QCheckBox(
      QStringLiteral("Install and start the systemd user service"), this);
  configureV4l2_ = new QCheckBox(
      QStringLiteral("Install/configure v4l2loopback virtual camera support"),
      this);
  loadLoopback_ = new QCheckBox(
      QStringLiteral("Load the StudioCast virtual camera now"), this);
  persistLoopback_ = new QCheckBox(
      QStringLiteral("Persist the virtual camera across reboot"), this);
  openBackendsSetup_ =
      new QCheckBox(QStringLiteral("Enable Open Source Backend Setup"), this);
  openBackendsSetup_->setObjectName(QStringLiteral("scInstallerOpenBackends"));
  openBackendsSetup_->setToolTip(QStringLiteral(
      "Configures and rebuilds StudioCast with Open Video/Open CUDA and Open "
      "Audio enabled. Dependency setup can install ONNX Runtime; model "
      "downloads stay controlled by the model-pack checkbox."));
  installModels_ = new QCheckBox(
      QStringLiteral("Download default Open Audio/Open Video model packs"),
      this);
  installModels_->setToolTip(
      QStringLiteral("Downloads exactly seven default packs containing eight "
                     "verified model artifact files, including open-source "
                     "fallbacks when Maxine is present."));
  removeUserData_ = new QCheckBox(
      QStringLiteral("Preserve settings"), this);
  removeUserData_->setObjectName(QStringLiteral("scInstallerPreserveSettings"));

  layout->addWidget(installService_);
  layout->addWidget(line(this));
  layout->addWidget(configureV4l2_);
  layout->addWidget(loadLoopback_);
  layout->addWidget(persistLoopback_);
  auto *cameraLimitation = mutedLabel(
      QStringLiteral("Without the virtual camera, StudioCast cannot be selected "
                     "as a camera. Custom installation will complete in a "
                     "degraded state."),
      this);
  cameraLimitation->setObjectName(QStringLiteral("scInstallerCameraLimitation"));
  layout->addWidget(cameraLimitation);
  layout->addWidget(line(this));
  layout->addWidget(openBackendsSetup_);
  layout->addWidget(installModels_);

  auto *vulkanBox =
      new QGroupBox(QStringLiteral("Open Vulkan (optional)"), this);
  auto *vulkanLayout = new QVBoxLayout(vulkanBox);
  openVulkan_ = new QCheckBox(
      QStringLiteral("Build StudioCast with Open Vulkan support"), vulkanBox);
  openVulkan_->setObjectName(QStringLiteral("scInstallerOpenVulkan"));
  openVulkan_->setToolTip(QStringLiteral(
      "Passes --open-vulkan so CMake enables the runtime-loaded Open Vulkan "
      "backend. This does not prove that a Vulkan device or production "
      "Vulkan matting runtime is available."));
  installVulkanRuntime_ = new QCheckBox(
      QStringLiteral("Install Vulkan loader and diagnostic packages"),
      vulkanBox);
  installVulkanRuntime_->setObjectName(
      QStringLiteral("scInstallerVulkanRuntime"));
  installVulkanRuntime_->setToolTip(QStringLiteral(
      "Installs libvulkan1 and vulkan-tools. These packages do not include or "
      "guarantee a working GPU driver/ICD."));
  installMesaVulkan_ = new QCheckBox(
      QStringLiteral("Install Mesa Intel/AMD Vulkan ICDs"), vulkanBox);
  installMesaVulkan_->setObjectName(QStringLiteral("scInstallerMesaVulkan"));
  installMesaVulkan_->setToolTip(QStringLiteral(
      "Installs mesa-vulkan-drivers for supported Intel/AMD GPUs and also "
      "selects the Vulkan loader/runtime packages."));
  installShaderTools_ = new QCheckBox(
      QStringLiteral("Install shader developer tools (optional)"), vulkanBox);
  installShaderTools_->setObjectName(QStringLiteral("scInstallerShaderTools"));
  installShaderTools_->setToolTip(QStringLiteral(
      "Installs glslang-tools for developers who validate or regenerate "
      "embedded SPIR-V. Normal StudioCast builds use committed shaders."));
  vulkanLayout->addWidget(openVulkan_);
  vulkanLayout->addWidget(installVulkanRuntime_);
  vulkanLayout->addWidget(installMesaVulkan_);
  vulkanLayout->addWidget(installShaderTools_);
  vulkanLayout->addWidget(mutedLabel(
      QStringLiteral(
          "Open Vulkan is runtime-loaded and needs a working GPU driver/ICD. "
          "Installing packages does not guarantee device support or full "
          "CUDA/NVIDIA effect parity; production Vulkan virtual-background "
          "matting is still unavailable."),
      vulkanBox));
  layout->addWidget(vulkanBox);
  layout->addWidget(removeUserData_);
  optionalComponentsNotice_ = new QLabel(this);
  optionalComponentsNotice_->setWordWrap(true);
  optionalComponentsNotice_->setTextFormat(Qt::RichText);
  optionalComponentsNotice_->setOpenExternalLinks(true);
  layout->addWidget(optionalComponentsNotice_);
  layout->addWidget(mutedLabel(
      QStringLiteral(
          "Default model packs are optional network downloads. "
          "NVIDIA Maxine SDK files are optional and are not shipped "
          "by StudioCast. Missing SDKs do not block the open-source "
          "build; the installed app reports unavailable engines with "
          "diagnostic hints."),
      this));
  layout->addStretch(1);

  connect(configureV4l2_, &QCheckBox::toggled, this, [this](bool checked) {
    loadLoopback_->setEnabled(checked);
    persistLoopback_->setEnabled(checked);
    if (auto *limitation = findChild<QLabel *>(
            QStringLiteral("scInstallerCameraLimitation"))) {
      limitation->setVisible(!checked);
    }
  });
  connect(installVulkanRuntime_, &QCheckBox::toggled, this,
          [this](bool checked) {
            installMesaVulkan_->setEnabled(checked && openVulkan_->isEnabled());
            if (!checked) {
              installMesaVulkan_->setChecked(false);
            }
          });
  connect(installMesaVulkan_, &QCheckBox::toggled, this, [this](bool checked) {
    if (checked) {
      installVulkanRuntime_->setChecked(true);
    }
  });
}

void ServiceOptionsPage::initializePage() {
  auto *w = installerWizard(this);
  const QString workflow = w->workflow();
  const bool installLike = workflow != QStringLiteral("uninstall") &&
                           workflow != QStringLiteral("advanced");
  installService_->setEnabled(installLike);
  configureV4l2_->setEnabled(installLike);
  loadLoopback_->setEnabled(installLike && w->configureV4l2Loopback());
  persistLoopback_->setEnabled(installLike && w->configureV4l2Loopback());
  openBackendsSetup_->setEnabled(installLike);
  openVulkan_->setEnabled(installLike);
  installVulkanRuntime_->setEnabled(installLike);
  installShaderTools_->setEnabled(installLike);
  installModels_->setEnabled(installLike);
  const bool reinstall = workflow == QStringLiteral("reinstall") ||
                         workflow == QStringLiteral("clean-install");
  removeUserData_->setEnabled(reinstall);
  removeUserData_->setVisible(reinstall);

  installService_->setChecked(w->installService());
  configureV4l2_->setChecked(w->configureV4l2Loopback());
  loadLoopback_->setChecked(w->loadLoopback());
  persistLoopback_->setChecked(w->persistLoopback());
  openBackendsSetup_->setChecked(w->openBackendsSetup());
  openVulkan_->setChecked(w->openVulkan());
  installVulkanRuntime_->setChecked(w->installVulkanRuntime());
  installMesaVulkan_->setChecked(w->installMesaVulkan());
  installMesaVulkan_->setEnabled(installLike &&
                                 installVulkanRuntime_->isChecked());
  installShaderTools_->setChecked(w->installShaderTools());
  installModels_->setChecked(w->installModels());
  removeUserData_->setChecked(!w->removeUserData());
  if (auto *limitation = findChild<QLabel *>(
          QStringLiteral("scInstallerCameraLimitation"))) {
    limitation->setVisible(!configureV4l2_->isChecked());
  }

  const QString optionalNotice =
      optionalComponentsNoticeText(w->statusObject(), w->sourceDir());
  optionalComponentsNotice_->setText(optionalNotice);
  optionalComponentsNotice_->setVisible(!optionalNotice.isEmpty());
}

bool ServiceOptionsPage::validatePage() {
  auto *w = installerWizard(this);
  w->setInstallService(installService_->isChecked());
  w->setConfigureV4l2Loopback(configureV4l2_->isChecked());
  w->setLoadLoopback(loadLoopback_->isChecked());
  w->setPersistLoopback(persistLoopback_->isChecked());
  w->setOpenBackendsSetup(openBackendsSetup_->isChecked());
  w->setOpenVulkan(openVulkan_->isChecked());
  w->setInstallVulkanRuntime(installVulkanRuntime_->isChecked());
  w->setInstallMesaVulkan(installMesaVulkan_->isChecked());
  w->setInstallShaderTools(installShaderTools_->isChecked());
  w->setInstallModels(installModels_->isChecked());
  if (w->workflow() == QStringLiteral("reinstall") ||
      w->workflow() == QStringLiteral("clean-install")) {
    w->setRemoveUserData(!removeUserData_->isChecked());
  }
  return true;
}

ReviewPage::ReviewPage(QWidget *parent) : QWizardPage(parent) {
  setTitle(QStringLiteral("Review Changes"));
  setSubTitle(
      QStringLiteral("Nothing runs until you continue past this page."));
  setCommitPage(true);
  auto *layout = new QVBoxLayout(this);
  reviewText_ = new QPlainTextEdit(this);
  setReadOnlyLogStyle(reviewText_);
  layout->addWidget(reviewText_, 1);
}

void ReviewPage::initializePage() {
  auto *w = installerWizard(this);
  w->setButtonText(QWizard::CommitButton, actionLabel(w->workflow()));
  QString error;
  if (w->refreshPlan(&error)) {
    reviewText_->setPlainText(planTextFromObject(w->planObject()));
  } else {
    reviewText_->setPlainText(error);
  }
  emit completeChanged();
}

bool ReviewPage::isComplete() const {
  const auto *w = dynamic_cast<const InstallerWizard *>(window());
  return w && w->planReady();
}

ProgressPage::ProgressPage(QWidget *parent) : QWizardPage(parent) {
  setTitle(QStringLiteral("Progress"));
  setSubTitle(QStringLiteral("Backend output is streamed here."));

  auto *layout = new QVBoxLayout(this);
  stateLabel_ = new QLabel(QStringLiteral("Waiting to start"), this);
  stateLabel_->setObjectName(QStringLiteral("scInstallerProgressPhase"));
  stateLabel_->setWordWrap(true);
  layout->addWidget(stateLabel_);
  logText_ = new QPlainTextEdit(this);
  setReadOnlyLogStyle(logText_);
  layout->addWidget(logText_, 1);
}

ProgressPage::~ProgressPage() {
  if (isRunning()) {
#ifdef Q_OS_UNIX
    signalProcessGroup(SIGKILL);
#endif
    process_->kill();
    process_->waitForFinished(1000);
  }
}

void ProgressPage::initializePage() {
  if (started_) {
    return;
  }
  started_ = true;
  complete_ = false;
  exitCode_ = -1;
  stdoutBuffer_.clear();
  stderrBuffer_.clear();
  progressLineBuffer_.clear();
  cancellationRequested_ = false;
  isolatedProcessGroup_ = false;
  emit completeChanged();

  auto *w = installerWizard(this);
  if (w->workflow() == QStringLiteral("uninstall")) {
    setTitle(QStringLiteral("Uninstalling StudioCast"));
    setSubTitle(QStringLiteral("StudioCast app files are being removed."));
  } else {
    setTitle(QStringLiteral("Progress"));
    setSubTitle(QStringLiteral("Backend output is streamed here."));
  }
  process_ = new QProcess(this);
#if defined(Q_OS_UNIX) && QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
  process_->setChildProcessModifier([] { ::setpgid(0, 0); });
  isolatedProcessGroup_ = true;
#endif
  process_->setProcessChannelMode(QProcess::SeparateChannels);
  connect(process_, &QProcess::readyReadStandardOutput, this, [this] {
    const QByteArray bytes = process_->readAllStandardOutput();
    stdoutBuffer_ += bytes;
    consumeStructuredProgress(bytes);
    appendOutput(bytes);
  });
  connect(process_, &QProcess::readyReadStandardError, this, [this] {
    const QByteArray bytes = process_->readAllStandardError();
    stderrBuffer_ += bytes;
    appendOutput(bytes);
  });
  connect(
      process_,
      static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(
          &QProcess::finished),
      this, [this](int code, QProcess::ExitStatus status) {
        stdoutBuffer_ += process_->readAllStandardOutput();
        stderrBuffer_ += process_->readAllStandardError();
        exitCode_ = code;
        auto *pageWizard = installerWizard(this);
        QJsonObject result = lastJsonObject(stdoutBuffer_);
        if (result.isEmpty()) {
          result = lastJsonObject(stderrBuffer_);
        }
        QString protocolError;
        const QString state = jsonString(result, QStringLiteral("state"));
        if (jsonString(result, QStringLiteral("result_version")) !=
            QStringLiteral("installer-result/v1")) {
          protocolError = QStringLiteral("missing installer-result/v1 terminal result");
        } else if (!pageWizard ||
                   jsonString(result, QStringLiteral("plan_id")) !=
                       jsonString(pageWizard->planObject(), QStringLiteral("plan_id")) ||
                   jsonString(result, QStringLiteral("plan_digest")) !=
                       jsonString(pageWizard->planObject(), QStringLiteral("plan_digest"))) {
          protocolError = QStringLiteral("terminal result does not match the reviewed plan");
        } else if (!QStringList{QStringLiteral("committed"),
                                QStringLiteral("degraded"),
                                QStringLiteral("failed"),
                                QStringLiteral("rolled_back"),
                                QStringLiteral("cancelled")}
                        .contains(state)) {
          protocolError = QStringLiteral("terminal result has an unknown state");
        } else {
          QStringList reviewed;
          for (const QJsonValue &value :
               result.value(QStringLiteral("reviewed_operation_ids")).toArray())
            reviewed << value.toString();
          QStringList planned;
          for (const QJsonValue &value :
               pageWizard->planObject().value(QStringLiteral("operations")).toArray())
            planned << jsonString(value.toObject(), QStringLiteral("id"));
          if (reviewed != planned)
            protocolError = QStringLiteral("executed result names a different reviewed operation set");
        }
        const bool exitMatches =
            status == QProcess::NormalExit &&
            ((state == QStringLiteral("committed") && code == 0) ||
             (state == QStringLiteral("degraded") && code == 3) ||
             ((state == QStringLiteral("failed") ||
               state == QStringLiteral("rolled_back")) &&
              code != 0) ||
             (state == QStringLiteral("cancelled") && code == 130));
        if (protocolError.isEmpty() && !exitMatches) {
          protocolError = QStringLiteral("process exit does not corroborate the terminal result");
        }
        if (!protocolError.isEmpty()) {
          result = {{QStringLiteral("result_version"),
                     QStringLiteral("installer-result/v1")},
                    {QStringLiteral("plan_id"),
                     pageWizard ? jsonString(pageWizard->planObject(),
                                             QStringLiteral("plan_id"))
                                : QString()},
                    {QStringLiteral("plan_digest"),
                     pageWizard ? jsonString(pageWizard->planObject(),
                                             QStringLiteral("plan_digest"))
                                : QString()},
                    {QStringLiteral("state"), QStringLiteral("failed")},
                    {QStringLiteral("core_committed"), false},
                    {QStringLiteral("error"),
                     QJsonObject{{QStringLiteral("code"),
                                  cancellationRequested_
                                      ? QStringLiteral("execution.interrupted_uncertain")
                                      : QStringLiteral("execution.protocol_failure")},
                                 {QStringLiteral("message"), protocolError}}}};
        }
        if (pageWizard)
          pageWizard->setExecutionResult(result);
        complete_ = true;
        stateLabel_->setText(QStringLiteral("Transaction finished: %1")
                                 .arg(jsonString(result, QStringLiteral("state"))));
        emit completeChanged();
      });

  const QStringList args = w->executionCommandArguments();
  if (args.isEmpty()) {
    w->setExecutionResult(
        {{QStringLiteral("result_version"), QStringLiteral("installer-result/v1")},
         {QStringLiteral("state"), QStringLiteral("failed")},
         {QStringLiteral("core_committed"), false},
         {QStringLiteral("error"),
          QJsonObject{{QStringLiteral("code"), QStringLiteral("plan.unavailable")},
                      {QStringLiteral("message"),
                       QStringLiteral("No valid reviewed plan is available.")}}}});
    stateLabel_->setText(QStringLiteral("Execution blocked: reviewed plan unavailable."));
    complete_ = true;
    emit completeChanged();
    return;
  }
  stateLabel_->setText(QStringLiteral("Executing %1 reviewed operations")
                           .arg(w->planObject()
                                    .value(QStringLiteral("operations"))
                                    .toArray()
                                    .size()));
  logText_->appendPlainText(QStringLiteral("$ %1 %2").arg(
      w->backendPath(), args.join(QLatin1Char(' '))));
  process_->start(w->backendPath(), args);
  if (!process_->waitForStarted(5000)) {
    appendOutput(QStringLiteral("Failed to start installer backend: %1\n")
                     .arg(process_->errorString())
                     .toLocal8Bit());
    w->setExecutionResult(
        {{QStringLiteral("result_version"), QStringLiteral("installer-result/v1")},
         {QStringLiteral("plan_id"),
          jsonString(w->planObject(), QStringLiteral("plan_id"))},
         {QStringLiteral("plan_digest"),
          jsonString(w->planObject(), QStringLiteral("plan_digest"))},
         {QStringLiteral("state"), QStringLiteral("failed")},
         {QStringLiteral("core_committed"), false},
         {QStringLiteral("error"),
          QJsonObject{{QStringLiteral("code"),
                       QStringLiteral("execution.start_failed")},
                      {QStringLiteral("message"), process_->errorString()}}}});
    complete_ = true;
    emit completeChanged();
  }
}

bool ProgressPage::isComplete() const { return complete_; }

bool ProgressPage::validatePage() {
  return complete_ && !installerWizard(this)->executionResult().isEmpty();
}

bool ProgressPage::isRunning() const {
  return process_ && process_->state() != QProcess::NotRunning;
}

void ProgressPage::requestCancellation() {
  if (!isRunning() || cancellationRequested_)
    return;
  cancellationRequested_ = true;
  stateLabel_->setText(QStringLiteral("Cancelling; waiting for backend rollback result…"));
#ifdef Q_OS_UNIX
  signalProcessGroup(SIGINT);
#else
  process_->terminate();
#endif
  QTimer::singleShot(5000, this, [this] {
    if (isRunning()) {
      stateLabel_->setText(QStringLiteral(
          "Cancellation timed out; transaction state is uncertain."));
#ifdef Q_OS_UNIX
      signalProcessGroup(SIGKILL);
#endif
      process_->kill();
    }
  });
}

void ProgressPage::signalProcessGroup(int signal) {
#ifdef Q_OS_UNIX
  if (!process_ || process_->processId() <= 0)
    return;
  const pid_t pid = static_cast<pid_t>(process_->processId());
  ::kill(isolatedProcessGroup_ ? -pid : pid, signal);
#else
  Q_UNUSED(signal);
#endif
}

void ProgressPage::consumeStructuredProgress(const QByteArray &bytes) {
  progressLineBuffer_ += bytes;
  qsizetype newline = -1;
  while ((newline = progressLineBuffer_.indexOf('\n')) >= 0) {
    const QByteArray line = progressLineBuffer_.left(newline).trimmed();
    progressLineBuffer_.remove(0, newline + 1);
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(line, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject())
      continue;
    const QJsonObject event = document.object();
    if (jsonString(event, QStringLiteral("event_version")) !=
        QStringLiteral("installer-progress/v1"))
      continue;
    const auto *w = dynamic_cast<const InstallerWizard *>(window());
    if (!w ||
        jsonString(event, QStringLiteral("plan_id")) !=
            jsonString(w->planObject(), QStringLiteral("plan_id")) ||
        jsonString(event, QStringLiteral("plan_digest")) !=
            jsonString(w->planObject(), QStringLiteral("plan_digest")))
      continue;
    const QString phase = jsonString(event, QStringLiteral("phase"));
    if (phase.isEmpty())
      continue;
    const QString operation = jsonString(event, QStringLiteral("operation_id"));
    const QString state = jsonString(event, QStringLiteral("state"));
    stateLabel_->setText(
        operation.isEmpty() ? QStringLiteral("Phase: %1 (%2)").arg(phase, state)
                            : QStringLiteral("Phase: %1 — %2 (%3)")
                                  .arg(phase, operation, state));
  }
}

void ProgressPage::appendOutput(const QByteArray &bytes) {
  logText_->moveCursor(QTextCursor::End);
  logText_->insertPlainText(QString::fromLocal8Bit(bytes));
  logText_->moveCursor(QTextCursor::End);
}

FinishPage::FinishPage(QWidget *parent) : QWizardPage(parent) {
  setTitle(QStringLiteral("Finish"));
  setSubTitle(QStringLiteral("Installer workflow summary."));
  auto *layout = new QVBoxLayout(this);
  summaryLabel_ = new QLabel(this);
  summaryLabel_->setObjectName(QStringLiteral("scInstallerCompletionSummary"));
  summaryLabel_->setWordWrap(true);
  layout->addWidget(summaryLabel_);
  auto *actions = new QHBoxLayout;
  retryButton_ =
      new QPushButton(QStringLiteral("Retry optional operations"), this);
  retryButton_->setObjectName(QStringLiteral("scInstallerDegradedRetry"));
  continueButton_ =
      new QPushButton(QStringLiteral("Continue with limitations"), this);
  continueButton_->setObjectName(QStringLiteral("scInstallerDegradedContinue"));
  retryButton_->setVisible(false);
  continueButton_->setVisible(false);
  actions->addWidget(retryButton_);
  actions->addWidget(continueButton_);
  actions->addStretch(1);
  layout->addLayout(actions);
  details_ = new QPlainTextEdit(this);
  setReadOnlyLogStyle(details_);
  layout->addWidget(details_, 1);

  connect(retryButton_, &QPushButton::clicked, this, [this] {
    auto *w = installerWizard(this);
    w->setWorkflow(QStringLiteral("repair"));
    w->setCustomRoute(false);
    w->setStartId(PageReview);
    w->restart();
  });
  connect(continueButton_, &QPushButton::clicked, this,
          [this] { installerWizard(this)->accept(); });
}

void FinishPage::initializePage() {
  auto *w = installerWizard(this);
  const QJsonObject result = w->executionResult();
  const QString state = jsonString(result, QStringLiteral("state"),
                                   QStringLiteral("failed"));
  const bool degraded = state == QStringLiteral("degraded");
  retryButton_->setVisible(degraded);
  continueButton_->setVisible(degraded);
  retryButton_->setEnabled(
      !result.value(QStringLiteral("retryable_operation_ids"))
           .toArray()
           .isEmpty());
  retryButton_->setToolTip(
      retryButton_->isEnabled()
          ? QStringLiteral(
                "Re-analyze and review a repair plan before retrying.")
          : QStringLiteral(
                "The backend did not report a retryable operation."));
  if (state == QStringLiteral("committed")) {
    setTitle(actionLabel(w->workflow()) + QStringLiteral(" complete"));
    summaryLabel_->setText(
        w->workflow() == QStringLiteral("uninstall")
            ? QStringLiteral("StudioCast uninstall completed.")
            : QStringLiteral("The reviewed transaction committed successfully."));
  } else if (state == QStringLiteral("degraded")) {
    setTitle(QStringLiteral("Complete with limitations"));
    summaryLabel_->setText(QStringLiteral(
        "The core installation is active, but optional operations failed. "
        "You may retry those operations or continue with the listed limitations."));
  } else if (state == QStringLiteral("rolled_back")) {
    setTitle(QStringLiteral("Action failed; previous version restored"));
    summaryLabel_->setText(QStringLiteral(
        "The update did not commit. The prior active payload was preserved."));
  } else if (state == QStringLiteral("cancelled")) {
    setTitle(QStringLiteral("Action cancelled"));
    summaryLabel_->setText(QStringLiteral(
        "The backend reported cancellation and completed its transaction handling."));
  } else {
    setTitle(QStringLiteral("Action failed"));
    const QJsonObject error = result.value(QStringLiteral("error")).toObject();
    summaryLabel_->setText(
        QStringLiteral("No successful outcome is being claimed. %1: %2")
            .arg(jsonString(error, QStringLiteral("code"),
                            QStringLiteral("execution.failed")),
                 jsonString(error, QStringLiteral("message"),
                            QStringLiteral("See details."))));
  }
  setSubTitle(QStringLiteral("Structured transaction result and reconciliation."));

  const QJsonObject resultSnapshot = result;
  w->refreshStatus();
  const QJsonObject status = w->statusObject();

  QString text;
  text += QStringLiteral("Terminal result (authoritative):\n");
  text += QString::fromUtf8(
      QJsonDocument(resultSnapshot).toJson(QJsonDocument::Indented));
  text += QStringLiteral("\nRe-analysis (secondary evidence):\n");
  text += QString::fromUtf8(QJsonDocument(status).toJson(QJsonDocument::Indented));
  text += QStringLiteral("\nReviewed plan:\n");
  text += planTextFromObject(w->planObject());
  details_->setPlainText(text.trimmed());
}

} // namespace studiocast::installer
