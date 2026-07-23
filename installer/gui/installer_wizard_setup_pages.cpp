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

BuildOptionsPage::BuildOptionsPage(QWidget *parent) : QWizardPage(parent) {
  setTitle(QStringLiteral("Build Options"));
  setSubTitle(
      QStringLiteral("Choose the source, build type, and build cache."));

  auto *layout = new QVBoxLayout(this);
  auto *sourceBox = new QGroupBox(QStringLiteral("Source"), this);
  auto *sourceLayout = new QVBoxLayout(sourceBox);

  sourceDirRadio_ = new QRadioButton(
      QStringLiteral("Advanced: build from a selected source directory"),
      sourceBox);
  sourceDirRadio_->setObjectName(
      QStringLiteral("scInstallerAdvancedSourceDirectory"));
  archiveRadio_ = new QRadioButton(
      QStringLiteral("Advanced: build from a selected source archive"),
      sourceBox);
  archiveRadio_->setObjectName(
      QStringLiteral("scInstallerAdvancedSourceArchive"));
  officialReleaseRadio_ = new QRadioButton(
      QStringLiteral("Use the verified official stable release"), sourceBox);
  officialReleaseRadio_->setObjectName(
      QStringLiteral("scInstallerVerifiedOfficialSource"));
  officialReleaseRadio_->setChecked(true);
  sourceLayout->addWidget(officialReleaseRadio_);
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
  sourceLayout->addWidget(mutedLabel(
      QStringLiteral("The official option keeps the signed release identity. "
                     "Selecting either Advanced option changes the plan to "
                     "an arbitrary developer source."),
      sourceBox));
  const auto updateSourceInputs = [this, browseSource, browseArchive] {
    const bool sourceAllowed = sourceDirRadio_->isEnabled();
    sourceDirEdit_->setEnabled(sourceAllowed && sourceDirRadio_->isChecked());
    browseSource->setEnabled(sourceAllowed && sourceDirRadio_->isChecked());
    archiveEdit_->setEnabled(sourceAllowed && archiveRadio_->isChecked());
    browseArchive->setEnabled(sourceAllowed && archiveRadio_->isChecked());
  };
  connect(officialReleaseRadio_, &QRadioButton::toggled, this,
          updateSourceInputs);
  connect(sourceDirRadio_, &QRadioButton::toggled, this, updateSourceInputs);
  connect(archiveRadio_, &QRadioButton::toggled, this, updateSourceInputs);
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
      QStringLiteral("Install exact missing build/runtime packages"), buildBox);
  installDeps_->setToolTip(QStringLiteral(
      "This controls normal build/runtime package reconciliation only. "
      "Virtual-camera package/module work is reviewed separately."));
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
  officialReleaseRadio_->setEnabled(sourceRelevant);
  sourceDirRadio_->setEnabled(sourceRelevant);
  archiveRadio_->setEnabled(sourceRelevant);
  buildDirEdit_->setEnabled(sourceRelevant);
  buildTypeCombo_->setEnabled(sourceRelevant);
  installDeps_->setEnabled(sourceRelevant);
  freshBuild_->setEnabled(sourceRelevant);

  officialReleaseRadio_->setChecked(!w->advancedSourceSelected());
  sourceDirRadio_->setChecked(w->advancedSourceSelected() &&
                              !w->useReleaseArchive());
  archiveRadio_->setChecked(w->advancedSourceSelected() &&
                            w->useReleaseArchive());
  sourceDirEdit_->setEnabled(sourceRelevant && sourceDirRadio_->isChecked());
  archiveEdit_->setEnabled(sourceRelevant && archiveRadio_->isChecked());
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
    if (officialReleaseRadio_->isChecked()) {
      if (w->verifiedReleaseReceiptPath().isEmpty() ||
          !QFileInfo(w->verifiedReleaseReceiptPath()).isFile()) {
        QMessageBox::warning(
            this, QStringLiteral("StudioCast Installer"),
            QStringLiteral("The verified official release receipt is no "
                           "longer available. Return to Analyze and retry."));
        return false;
      }
    } else if (archiveRadio_->isChecked()) {
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
  w->setAdvancedSourceSelected(!officialReleaseRadio_->isChecked());
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
  auto *pageLayout = new QVBoxLayout(this);
  auto *scroll = new QScrollArea(this);
  scroll->setWidgetResizable(true);
  auto *content = new QWidget(scroll);
  auto *layout = new QVBoxLayout(content);
  scroll->setWidget(content);
  pageLayout->addWidget(scroll);

  installService_ = new QCheckBox(
      QStringLiteral("Install and start the systemd user service"), this);
  installService_->setObjectName(QStringLiteral("scInstallerService"));
  configureV4l2_ = new QCheckBox(
      QStringLiteral(
          "Reconcile required v4l2loopback package and module support"),
      this);
  configureV4l2_->setObjectName(QStringLiteral("scInstallerConfigureV4l"));
  loadLoopback_ = new QCheckBox(
      QStringLiteral("Load the StudioCast virtual-camera module/device now"),
      this);
  loadLoopback_->setObjectName(QStringLiteral("scInstallerLoadV4l"));
  persistLoopback_ = new QCheckBox(
      QStringLiteral(
          "Write namespaced automatic-load configuration for reboot"),
      this);
  persistLoopback_->setObjectName(QStringLiteral("scInstallerPersistV4l"));
  v4lDeviceNumber_ = new QSpinBox(this);
  v4lDeviceNumber_->setRange(0, 63);
  v4lDeviceNumber_->setObjectName(
      QStringLiteral("scInstallerV4lDeviceNumber"));
  v4lLabel_ = new QLineEdit(this);
  v4lLabel_->setMaxLength(64);
  v4lLabel_->setObjectName(QStringLiteral("scInstallerV4lLabel"));
  v4lExclusiveCaps_ =
      new QCheckBox(QStringLiteral("Exclusive capture mode"), this);
  v4lExclusiveCaps_->setObjectName(
      QStringLiteral("scInstallerV4lExclusiveCaps"));
  cameraLimitationAck_ = new QCheckBox(
      QStringLiteral("I understand this installation will not provide a "
                     "StudioCast camera and will be marked degraded"),
      this);
  cameraLimitationAck_->setObjectName(
      QStringLiteral("scInstallerCameraLimitationAcknowledged"));
  openCuda_ = new QCheckBox(
      QStringLiteral("Enable Open Video / Open CUDA backend"), this);
  openCuda_->setObjectName(QStringLiteral("scInstallerOpenCuda"));
  openAudio_ =
      new QCheckBox(QStringLiteral("Enable Open Audio backend"), this);
  openAudio_->setObjectName(QStringLiteral("scInstallerOpenAudio"));
  installModels_ = new QCheckBox(
      QStringLiteral("Download selected curated model packs"), this);
  installModels_->setObjectName(QStringLiteral("scInstallerInstallModels"));
  installModels_->setToolTip(
      QStringLiteral("Recommended downloads exactly seven default packs with "
                     "eight verified artifacts. Custom may select individual "
                     "curated packs; Maxine does not disable these fallbacks."));
  removeUserData_ = new QCheckBox(
      QStringLiteral("Preserve settings"), this);
  removeUserData_->setObjectName(QStringLiteral("scInstallerPreserveSettings"));

  layout->addWidget(installService_);
  auto *unknownChoices = mutedLabel(
      QStringLiteral("A migrated installer manifest did not record one or "
                     "more service or virtual-camera choices. Partially "
                     "checked controls are unresolved and must be explicitly "
                     "selected or cleared before review."),
      this);
  unknownChoices->setObjectName(
      QStringLiteral("scInstallerUnknownMigratedChoices"));
  layout->addWidget(unknownChoices);
  layout->addWidget(line(this));
  auto *cameraBox = new QGroupBox(QStringLiteral("Virtual camera"), this);
  auto *cameraLayout = new QFormLayout(cameraBox);
  cameraLayout->addRow(configureV4l2_);
  cameraLayout->addRow(loadLoopback_);
  cameraLayout->addRow(persistLoopback_);
  cameraLayout->addRow(QStringLiteral("Device number"), v4lDeviceNumber_);
  cameraLayout->addRow(QStringLiteral("Camera label"), v4lLabel_);
  cameraLayout->addRow(v4lExclusiveCaps_);
  auto *cameraLimitation = mutedLabel(
      QStringLiteral("LIMITATION: without the virtual camera, StudioCast "
                     "cannot be selected as a camera. Custom installation "
                     "will complete in a degraded state, not full success."),
      this);
  cameraLimitation->setObjectName(QStringLiteral("scInstallerCameraLimitation"));
  cameraLayout->addRow(cameraLimitation);
  cameraLayout->addRow(cameraLimitationAck_);
  layout->addWidget(cameraBox);
  layout->addWidget(line(this));
  auto *openBox = new QGroupBox(QStringLiteral("Open-source backends"), this);
  auto *openLayout = new QVBoxLayout(openBox);
  openLayout->addWidget(openCuda_);
  openLayout->addWidget(openAudio_);
  layout->addWidget(openBox);
  auto *modelsBox = new QGroupBox(QStringLiteral("Curated model packs"), this);
  auto *modelsLayout = new QGridLayout(modelsBox);
  modelsLayout->addWidget(installModels_, 0, 0, 1, 2);
  const QStringList packs = defaultModelPackIds();
  for (int index = 0; index < packs.size(); ++index) {
    auto *pack = new QCheckBox(packs.at(index), modelsBox);
    pack->setObjectName(QStringLiteral("scInstallerModelPack_%1")
                            .arg(packs.at(index)));
    pack->setProperty("modelPackId", packs.at(index));
    modelPackChecks_.append(pack);
    modelsLayout->addWidget(pack, 1 + index / 2, index % 2);
  }
  modelDestination_ = new QLineEdit(modelsBox);
  modelDestination_->setObjectName(
      QStringLiteral("scInstallerModelDestination"));
  modelsLayout->addWidget(new QLabel(QStringLiteral("User-local destination"),
                                    modelsBox),
                          5, 0);
  modelsLayout->addWidget(modelDestination_, 5, 1);
  layout->addWidget(modelsBox);

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

  connect(configureV4l2_, &QCheckBox::stateChanged, this, [this](int state) {
    const bool checked = state == Qt::Checked;
    const bool unresolved = state == Qt::PartiallyChecked;
    loadLoopback_->setEnabled(checked);
    persistLoopback_->setEnabled(checked);
    v4lDeviceNumber_->setEnabled(checked);
    v4lLabel_->setEnabled(checked);
    v4lExclusiveCaps_->setEnabled(checked);
    if (!checked && !unresolved) {
      loadLoopback_->setChecked(false);
      persistLoopback_->setChecked(false);
    }
    if (auto *limitation = findChild<QLabel *>(
            QStringLiteral("scInstallerCameraLimitation"))) {
      limitation->setVisible(!checked && !unresolved);
    }
    cameraLimitationAck_->setVisible(!checked && !unresolved);
    if (checked)
      cameraLimitationAck_->setChecked(false);
  });
  connect(installModels_, &QCheckBox::toggled, this, [this](bool checked) {
    for (QCheckBox *pack : modelPackChecks_)
      pack->setEnabled(checked);
    modelDestination_->setEnabled(checked);
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
  v4lDeviceNumber_->setEnabled(installLike && w->configureV4l2Loopback());
  v4lLabel_->setEnabled(installLike && w->configureV4l2Loopback());
  v4lExclusiveCaps_->setEnabled(installLike && w->configureV4l2Loopback());
  openCuda_->setEnabled(installLike);
  openAudio_->setEnabled(installLike);
  openVulkan_->setEnabled(installLike);
  installVulkanRuntime_->setEnabled(installLike);
  installShaderTools_->setEnabled(installLike);
  installModels_->setEnabled(installLike);
  const bool reinstall = workflow == QStringLiteral("reinstall") ||
                         workflow == QStringLiteral("clean-install");
  removeUserData_->setEnabled(reinstall);
  removeUserData_->setVisible(reinstall);

  {
    const QSignalBlocker serviceBlocker(installService_);
    installService_->setTristate(!w->serviceChoiceResolved());
    installService_->setCheckState(
        w->serviceChoiceResolved()
            ? (w->installService() ? Qt::Checked : Qt::Unchecked)
            : Qt::PartiallyChecked);
  }
  {
    const QSignalBlocker cameraBlocker(configureV4l2_);
    const QSignalBlocker loadBlocker(loadLoopback_);
    const QSignalBlocker persistBlocker(persistLoopback_);
    configureV4l2_->setTristate(!w->v4lChoiceResolved());
    configureV4l2_->setCheckState(
        w->v4lChoiceResolved()
            ? (w->configureV4l2Loopback() ? Qt::Checked : Qt::Unchecked)
            : Qt::PartiallyChecked);
    loadLoopback_->setTristate(!w->loadLoopbackChoiceResolved());
    loadLoopback_->setCheckState(
        w->loadLoopbackChoiceResolved()
            ? (w->configureV4l2Loopback() && w->loadLoopback() ? Qt::Checked
                                                               : Qt::Unchecked)
            : Qt::PartiallyChecked);
    persistLoopback_->setTristate(!w->persistLoopbackChoiceResolved());
    persistLoopback_->setCheckState(
        w->persistLoopbackChoiceResolved()
            ? (w->configureV4l2Loopback() && w->persistLoopback()
                   ? Qt::Checked
                   : Qt::Unchecked)
            : Qt::PartiallyChecked);
  }
  const bool cameraExplicitlyEnabled =
      configureV4l2_->checkState() == Qt::Checked;
  loadLoopback_->setEnabled(installLike && cameraExplicitlyEnabled);
  persistLoopback_->setEnabled(installLike && cameraExplicitlyEnabled);
  v4lDeviceNumber_->setValue(w->v4lDeviceNumber());
  v4lLabel_->setText(w->v4lLabel());
  v4lExclusiveCaps_->setChecked(w->v4lExclusiveCaps());
  cameraLimitationAck_->setChecked(false);
  cameraLimitationAck_->setVisible(configureV4l2_->checkState() ==
                                   Qt::Unchecked);
  openCuda_->setChecked(w->openCuda());
  openAudio_->setChecked(w->openAudio());
  openVulkan_->setChecked(w->openVulkan());
  installVulkanRuntime_->setChecked(w->installVulkanRuntime());
  installMesaVulkan_->setChecked(w->installMesaVulkan());
  installMesaVulkan_->setEnabled(installLike &&
                                 installVulkanRuntime_->isChecked());
  installShaderTools_->setChecked(w->installShaderTools());
  installModels_->setChecked(w->installModels());
  for (QCheckBox *pack : modelPackChecks_) {
    pack->setChecked(
        w->modelPackIds().contains(pack->property("modelPackId").toString()));
    pack->setEnabled(installLike && installModels_->isChecked());
  }
  modelDestination_->setText(w->modelDestination());
  modelDestination_->setEnabled(installLike && installModels_->isChecked());
  removeUserData_->setChecked(!w->removeUserData());
  if (auto *limitation = findChild<QLabel *>(
          QStringLiteral("scInstallerCameraLimitation"))) {
    limitation->setVisible(configureV4l2_->checkState() == Qt::Unchecked);
  }
  if (auto *unknown = findChild<QLabel *>(
          QStringLiteral("scInstallerUnknownMigratedChoices"))) {
    unknown->setVisible(!w->serviceChoiceResolved() ||
                        !w->v4lChoiceResolved() ||
                        !w->loadLoopbackChoiceResolved() ||
                        !w->persistLoopbackChoiceResolved());
  }

  const QString optionalNotice =
      optionalComponentsNoticeText(w->statusObject(), w->sourceDir());
  optionalComponentsNotice_->setText(optionalNotice);
  optionalComponentsNotice_->setVisible(!optionalNotice.isEmpty());
}

bool ServiceOptionsPage::validatePage() {
  auto *w = installerWizard(this);
  QStringList selectedModels;
  for (QCheckBox *pack : modelPackChecks_) {
    if (pack->isChecked())
      selectedModels.append(pack->property("modelPackId").toString());
  }
  if (installService_->checkState() == Qt::PartiallyChecked ||
      configureV4l2_->checkState() == Qt::PartiallyChecked ||
      (configureV4l2_->checkState() == Qt::Checked &&
       (loadLoopback_->checkState() == Qt::PartiallyChecked ||
        persistLoopback_->checkState() == Qt::PartiallyChecked))) {
    QMessageBox::warning(
        this, QStringLiteral("Review migrated choices"),
        QStringLiteral("Explicitly select or clear every partially checked "
                       "service and virtual-camera option before review."));
    return false;
  }
  if (!configureV4l2_->isChecked() &&
      !cameraLimitationAck_->isChecked()) {
    QMessageBox::warning(
        this, QStringLiteral("Virtual camera limitation"),
        QStringLiteral("Confirm that this Custom installation may complete "
                       "degraded without a StudioCast virtual camera."));
    return false;
  }
  static const QRegularExpression labelPattern(
      QStringLiteral("^[ A-Za-z0-9_.()+-]{1,64}$"));
  if (configureV4l2_->isChecked() &&
      !labelPattern.match(v4lLabel_->text().trimmed()).hasMatch()) {
    QMessageBox::warning(
        this, QStringLiteral("Invalid camera label"),
        QStringLiteral("Use 1-64 letters, numbers, spaces, or . _ ( ) + -."));
    return false;
  }
  if (installModels_->isChecked() && selectedModels.isEmpty()) {
    QMessageBox::warning(this, QStringLiteral("No model packs selected"),
                         QStringLiteral("Select at least one curated model "
                                        "pack or disable model downloads."));
    return false;
  }
  if (installModels_->isChecked() &&
      !isUserLocalModelDestination(modelDestination_->text())) {
    QMessageBox::warning(
        this, QStringLiteral("Invalid model destination"),
        QStringLiteral("Choose an absolute destination inside %1.")
            .arg(defaultModelDestination()));
    return false;
  }
  w->setInstallService(installService_->isChecked());
  w->setConfigureV4l2Loopback(configureV4l2_->isChecked());
  w->setLoadLoopback(loadLoopback_->isChecked());
  w->setPersistLoopback(persistLoopback_->isChecked());
  w->setV4lDeviceNumber(v4lDeviceNumber_->value());
  w->setV4lLabel(v4lLabel_->text());
  w->setV4lExclusiveCaps(v4lExclusiveCaps_->isChecked());
  w->setOpenCuda(openCuda_->isChecked());
  w->setOpenAudio(openAudio_->isChecked());
  w->setOpenVulkan(openVulkan_->isChecked());
  w->setInstallVulkanRuntime(installVulkanRuntime_->isChecked());
  w->setInstallMesaVulkan(installMesaVulkan_->isChecked());
  w->setInstallShaderTools(installShaderTools_->isChecked());
  w->setInstallModels(installModels_->isChecked());
  w->setModelPackIds(selectedModels);
  w->setModelDestination(modelDestination_->text());
  if (w->workflow() == QStringLiteral("reinstall") ||
      w->workflow() == QStringLiteral("clean-install")) {
    w->setRemoveUserData(!removeUserData_->isChecked());
  }
  return true;
}


} // namespace studiocast::installer
