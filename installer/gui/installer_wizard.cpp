#include "installer_wizard.h"

#include <QApplication>
#include <QBoxLayout>
#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
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
#include <QPlainTextEdit>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QSizePolicy>
#include <QStandardPaths>
#include <QStyle>
#include <QTextCursor>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>

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

  const QString openDocs = docLink(
      sourceDir, QStringLiteral("docs/open_source_video_models_install.md"),
      QStringLiteral("Open Source backend setup instructions"));
  const QString maxineDocs =
      docLink(sourceDir, QStringLiteral("docs/maxine_install.md"),
              QStringLiteral("Maxine SDK install instructions"));

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

  if (lines.isEmpty()) {
    return QStringLiteral("CUDA, ONNX Runtime CUDA provider, and Maxine SDK "
                          "assets were detected.");
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
  text +=
      QStringLiteral("Workflow: ") +
      jsonString(plan, QStringLiteral("workflow"), QStringLiteral("unknown")) +
      QStringLiteral("\n");
  text += QStringLiteral("Supported OS: ") +
          QString(jsonBool(plan, QStringLiteral("supported"))
                      ? QStringLiteral("yes")
                      : QStringLiteral("no")) +
          QStringLiteral("\n");
  text += QStringLiteral("Support: ") +
          jsonString(plan, QStringLiteral("support_reason")) +
          QStringLiteral("\n\n");
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

void addPreferenceProgressBar(QWizardPage *page, int pageId) {
  auto *layout = qobject_cast<QBoxLayout *>(page->layout());
  if (!layout) {
    return;
  }

  const int progressValue = preferenceProgressValue(pageId);
  const int progressMaximum = preferenceProgressMaximum();
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
      addPreferenceProgressBar(page, pageId);
    }
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
  setButtonText(QWizard::CommitButton, QStringLiteral("Finish"));
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
  addPreferenceProgressBars(this);
  setStartId(PageIntro);

  resize(920, 680);
  refreshStatus();
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
bool InstallerWizard::installModels() const { return installModels_; }
bool InstallerWizard::freshBuild() const { return freshBuild_; }
bool InstallerWizard::allowUnsupported() const { return allowUnsupported_; }
bool InstallerWizard::removeUserData() const { return removeUserData_; }

void InstallerWizard::setWorkflow(const QString &workflow) {
  workflow_ = workflow;
}
void InstallerWizard::setSourceDir(const QString &path) { sourceDir_ = path; }
void InstallerWizard::setReleaseArchive(const QString &path) {
  releaseArchive_ = path;
}
void InstallerWizard::setUseReleaseArchive(bool enabled) {
  useReleaseArchive_ = enabled;
}
void InstallerWizard::setBuildDir(const QString &path) { buildDir_ = path; }
void InstallerWizard::setBuildType(const QString &type) { buildType_ = type; }
void InstallerWizard::setInstallDeps(bool enabled) { installDeps_ = enabled; }
void InstallerWizard::setConfigureV4l2Loopback(bool enabled) {
  configureV4l2Loopback_ = enabled;
}
void InstallerWizard::setLoadLoopback(bool enabled) { loadLoopback_ = enabled; }
void InstallerWizard::setPersistLoopback(bool enabled) {
  persistLoopback_ = enabled;
}
void InstallerWizard::setInstallService(bool enabled) {
  installService_ = enabled;
}
void InstallerWizard::setOpenBackendsSetup(bool enabled) {
  openBackendsSetup_ = enabled;
}
void InstallerWizard::setInstallModels(bool enabled) {
  installModels_ = enabled;
}
void InstallerWizard::setFreshBuild(bool enabled) { freshBuild_ = enabled; }
void InstallerWizard::setAllowUnsupported(bool enabled) {
  allowUnsupported_ = enabled;
}
void InstallerWizard::setRemoveUserData(bool enabled) {
  removeUserData_ = enabled;
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

void InstallerWizard::refreshStatus() {
  QString error;
  QJsonObject status;
  if (runBackendJson(
          QStringList{QStringLiteral("status"), QStringLiteral("--json")},
          &status, &error)) {
    statusObject_ = status;
    osObject_ = status.value(QStringLiteral("os")).toObject();
  } else {
    statusObject_ = QJsonObject{
        {QStringLiteral("project_version"), QStringLiteral(STUDIOCAST_VERSION)},
        {QStringLiteral("installed"), false},
        {QStringLiteral("installed_version"), QString()},
        {QStringLiteral("backend_error"), error},
    };
    osObject_ = QJsonObject{
        {QStringLiteral("supported"), false},
        {QStringLiteral("support_reason"), error},
    };
  }
}

bool InstallerWizard::refreshPlan(QString *error) {
  QJsonObject plan;
  QStringList args{QStringLiteral("plan"), workflow_, QStringLiteral("--json")};
  args += backendOptions(true);
  if (!runBackendJson(args, &plan, error)) {
    planObject_ = QJsonObject();
    return false;
  }
  planObject_ = plan;
  return true;
}

QStringList InstallerWizard::backendOptions(bool forPlan) const {
  QStringList args;

  const bool needsSource = workflow_ != QStringLiteral("uninstall") &&
                           workflow_ != QStringLiteral("advanced");
  if (needsSource) {
    if (useReleaseArchive_ && !releaseArchive_.isEmpty()) {
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
  args << (installModels_ ? QStringLiteral("--models")
                          : QStringLiteral("--no-models"));
  args << (freshBuild_ ? QStringLiteral("--fresh-build")
                       : QStringLiteral("--no-fresh-build"));
  args << (removeUserData_ ? QStringLiteral("--remove-user-data")
                           : QStringLiteral("--preserve-user-data"));
  if (allowUnsupported_) {
    args << QStringLiteral("--allow-unsupported");
  }
  if (!forPlan) {
    args << QStringLiteral("--yes");
  }
  return args;
}

QStringList InstallerWizard::workflowCommandArguments(bool dryRun) const {
  if (workflow_ == QStringLiteral("advanced")) {
    return QStringList{QStringLiteral("advanced")};
  }
  QStringList args{workflow_};
  args += backendOptions(false);
  if (dryRun) {
    args << QStringLiteral("--dry-run");
  }
  return args;
}

IntroPage::IntroPage(QWidget *parent) : QWizardPage(parent) {
  setTitle(QStringLiteral("StudioCast Installer"));
  setSubTitle(QStringLiteral("Select the workflow to run on this system."));

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

  auto *workflowBox = new QGroupBox(QStringLiteral("Workflow"), this);
  auto *workflowLayout = new QVBoxLayout();
  workflowGroup_ = new QButtonGroup(workflowBox);
  const QList<QPair<QString, QString>> workflows = {
      {QStringLiteral("install"), QStringLiteral("Install StudioCast")},
      {QStringLiteral("update"), QStringLiteral("Update StudioCast")},
      {QStringLiteral("repair"), QStringLiteral("Repair installation")},
      {QStringLiteral("uninstall"), QStringLiteral("Uninstall")},
      {QStringLiteral("clean-install"), QStringLiteral("Clean install")},
      {QStringLiteral("advanced"), QStringLiteral("Advanced/manual options")},
  };
  int id = 0;
  for (const auto &workflow : workflows) {
    auto *radio = new QRadioButton(workflow.second, workflowBox);
    radio->setProperty("workflow", workflow.first);
    workflowGroup_->addButton(radio, id++);
    workflowLayout->addWidget(radio);
    if (workflow.first == QStringLiteral("install")) {
      radio->setChecked(true);
    }
  }
  workflowBox->setLayout(workflowLayout);
  layout->addWidget(workflowBox);

  layout->addWidget(mutedLabel(
      QStringLiteral("The GUI is not run as root. Privileged operations are "
                     "delegated only to specific backend/setup commands when "
                     "the selected plan requires them."),
      this));
  layout->addStretch(1);
}

void IntroPage::initializePage() {
  auto *w = installerWizard(this);
  w->refreshStatus();
  const QJsonObject status = w->statusObject();
  const QJsonObject os = w->osObject();

  osValue_->setText(QStringLiteral("%1 (base %2 %3) - %4")
                        .arg(jsonString(os, QStringLiteral("pretty_name"),
                                        QStringLiteral("unknown")),
                             jsonString(os, QStringLiteral("base_version_id"),
                                        QStringLiteral("unknown")),
                             jsonString(os, QStringLiteral("base_codename"),
                                        QStringLiteral("unknown")),
                             jsonString(os, QStringLiteral("support_reason"))));

  const QString projectVersion =
      jsonString(status, QStringLiteral("project_version"),
                 QStringLiteral(STUDIOCAST_VERSION));
  const QString installedVersion =
      jsonString(status, QStringLiteral("installed_version"));
  versionValue_->setText(
      QStringLiteral("Project %1, installed %2")
          .arg(projectVersion, installedVersion.isEmpty()
                                   ? QStringLiteral("not installed")
                                   : installedVersion));

  installValue_->setText(
      jsonBool(status, QStringLiteral("installed"))
          ? QStringLiteral("Installed; service %1, manifest %2")
                .arg(jsonString(
                         status.value(QStringLiteral("service")).toObject(),
                         QStringLiteral("active"), QStringLiteral("unknown")),
                     jsonString(status, QStringLiteral("manifest_path")))
          : QStringLiteral("Not installed through the StudioCast installer"));
}

bool IntroPage::validatePage() {
  auto *w = installerWizard(this);
  const QAbstractButton *button = workflowGroup_->checkedButton();
  if (button) {
    w->setWorkflow(button->property("workflow").toString());
  }
  return true;
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
      QStringLiteral("Allow this workflow on an unsupported distro"), this);
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
  return jsonBool(w->osObject(), QStringLiteral("supported")) ||
         allowUnsupported_->isChecked();
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
  const QString workflow = w->workflow();
  w->setInstallDeps(workflow == QStringLiteral("install") ||
                    workflow == QStringLiteral("clean-install"));
  w->setConfigureV4l2Loopback(workflow == QStringLiteral("install") ||
                              workflow == QStringLiteral("clean-install"));
  w->setLoadLoopback(w->configureV4l2Loopback());
  w->setPersistLoopback(w->configureV4l2Loopback());
  w->setInstallService(workflow != QStringLiteral("uninstall") &&
                       workflow != QStringLiteral("advanced"));
  w->setFreshBuild(workflow == QStringLiteral("clean-install"));

  QString error;
  if (w->refreshPlan(&error)) {
    planText_->setPlainText(planTextFromObject(w->planObject()));
  } else {
    planText_->setPlainText(error);
  }
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
  openBackendsSetup_->setToolTip(QStringLiteral(
      "Configures and rebuilds StudioCast with Open Video/Open CUDA and Open "
      "Audio enabled. Dependency setup can install ONNX Runtime; model "
      "downloads stay controlled by the model-pack checkbox."));
  installModels_ = new QCheckBox(
      QStringLiteral("Download default Open Audio/Open Video model packs"),
      this);
  installModels_->setToolTip(
      QStringLiteral("Downloads the default model packs used by automatic "
                     "Open Audio and Open Video model selection."));
  removeUserData_ = new QCheckBox(
      QStringLiteral("Remove user config, downloaded models, logs, and cache"),
      this);

  layout->addWidget(installService_);
  layout->addWidget(line(this));
  layout->addWidget(configureV4l2_);
  layout->addWidget(loadLoopback_);
  layout->addWidget(persistLoopback_);
  layout->addWidget(line(this));
  layout->addWidget(openBackendsSetup_);
  layout->addWidget(installModels_);
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
  installModels_->setEnabled(installLike);
  removeUserData_->setEnabled(workflow == QStringLiteral("uninstall") ||
                              workflow == QStringLiteral("clean-install"));

  installService_->setChecked(w->installService());
  configureV4l2_->setChecked(w->configureV4l2Loopback());
  loadLoopback_->setChecked(w->loadLoopback());
  persistLoopback_->setChecked(w->persistLoopback());
  openBackendsSetup_->setChecked(w->openBackendsSetup());
  installModels_->setChecked(w->installModels());
  removeUserData_->setChecked(w->removeUserData());

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
  w->setInstallModels(installModels_->isChecked());
  w->setRemoveUserData(removeUserData_->isChecked());
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
  QString error;
  if (w->refreshPlan(&error)) {
    reviewText_->setPlainText(planTextFromObject(w->planObject()));
  } else {
    reviewText_->setPlainText(error);
  }
}

ProgressPage::ProgressPage(QWidget *parent) : QWizardPage(parent) {
  setTitle(QStringLiteral("Progress"));
  setSubTitle(QStringLiteral("Backend output is streamed here."));

  auto *layout = new QVBoxLayout(this);
  stateLabel_ = new QLabel(QStringLiteral("Waiting to start"), this);
  stateLabel_->setWordWrap(true);
  layout->addWidget(stateLabel_);
  logText_ = new QPlainTextEdit(this);
  setReadOnlyLogStyle(logText_);
  layout->addWidget(logText_, 1);
}

void ProgressPage::initializePage() {
  if (started_) {
    return;
  }
  started_ = true;
  complete_ = false;
  exitCode_ = -1;
  emit completeChanged();

  auto *w = installerWizard(this);
  if (w->workflow() == QStringLiteral("advanced")) {
    logText_->setPlainText(planTextFromObject(w->planObject()));
    stateLabel_->setText(QStringLiteral("Advanced plan shown. No changes were "
                                        "made."));
    complete_ = true;
    emit completeChanged();
    return;
  }

  process_ = new QProcess(this);
  process_->setProcessChannelMode(QProcess::SeparateChannels);
  connect(process_, &QProcess::readyReadStandardOutput, this,
          [this] { appendOutput(process_->readAllStandardOutput()); });
  connect(process_, &QProcess::readyReadStandardError, this,
          [this] { appendOutput(process_->readAllStandardError()); });
  connect(
      process_,
      static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(
          &QProcess::finished),
      this, [this](int code, QProcess::ExitStatus status) {
        exitCode_ = code;
        complete_ = true;
        if (status == QProcess::NormalExit && code == 0) {
          stateLabel_->setText(QStringLiteral("Workflow completed."));
        } else {
          stateLabel_->setText(
              QStringLiteral("Workflow failed with exit code %1.").arg(code));
        }
        emit completeChanged();
      });

  const QStringList args = w->workflowCommandArguments(false);
  stateLabel_->setText(QStringLiteral("Running backend workflow: %1")
                           .arg(args.join(QLatin1Char(' '))));
  logText_->appendPlainText(QStringLiteral("$ %1 %2").arg(
      w->backendPath(), args.join(QLatin1Char(' '))));
  process_->start(w->backendPath(), args);
  if (!process_->waitForStarted(5000)) {
    appendOutput(QStringLiteral("Failed to start installer backend: %1\n")
                     .arg(process_->errorString())
                     .toLocal8Bit());
    complete_ = true;
    emit completeChanged();
  }
}

bool ProgressPage::isComplete() const { return complete_; }

bool ProgressPage::validatePage() {
  installerWizard(this)->refreshStatus();
  return true;
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
  summaryLabel_->setWordWrap(true);
  layout->addWidget(summaryLabel_);
  details_ = new QPlainTextEdit(this);
  setReadOnlyLogStyle(details_);
  layout->addWidget(details_, 1);
}

void FinishPage::initializePage() {
  auto *w = installerWizard(this);
  w->refreshStatus();
  const QJsonObject status = w->statusObject();
  const QString installedVersion =
      jsonString(status, QStringLiteral("installed_version"));
  const bool installed = jsonBool(status, QStringLiteral("installed"));

  summaryLabel_->setText(
      installed
          ? QStringLiteral("StudioCast is installed. Version: %1")
                .arg(installedVersion.isEmpty() ? QStringLiteral("unknown")
                                                : installedVersion)
          : QStringLiteral("StudioCast is not currently installed through the "
                           "installer manifest."));

  QString text;
  text += QStringLiteral("Workflow: ") + w->workflow() + QStringLiteral("\n");
  text += QStringLiteral("Manifest: ") +
          jsonString(status, QStringLiteral("manifest_path")) +
          QStringLiteral("\n");
  text += QStringLiteral("Service: ") +
          jsonString(status.value(QStringLiteral("service")).toObject(),
                     QStringLiteral("active"), QStringLiteral("unknown")) +
          QStringLiteral("\n");
  text += QStringLiteral("Local bin: ") +
          jsonString(status, QStringLiteral("local_bin_dir")) +
          QStringLiteral("\n\n");
  text += planTextFromObject(w->planObject());
  details_->setPlainText(text.trimmed());
}

} // namespace studiocast::installer
