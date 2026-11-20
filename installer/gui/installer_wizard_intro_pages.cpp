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
          if (!radio->property("customRoute").toBool()) {
            w->setAdvancedSourceSelected(false);
          }
          if (radio->property("recommended").toBool() &&
              radio->property("workflow").toString() ==
                  QStringLiteral("install")) {
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

  selfUpdateBox_ = new QGroupBox(QStringLiteral("Installer update"), this);
  auto *selfUpdateLayout = new QVBoxLayout(selfUpdateBox_);
  selfUpdateLabel_ = new QLabel(selfUpdateBox_);
  selfUpdateLabel_->setWordWrap(true);
  selfUpdateLayout->addWidget(selfUpdateLabel_);
  auto *selfUpdateActions = new QHBoxLayout;
  restartInstallerButton_ = new QPushButton(
      QStringLiteral("Restart with verified update"), selfUpdateBox_);
  restartInstallerButton_->setObjectName(
      QStringLiteral("scInstallerSelfUpdateRestart"));
  manualDownloadButton_ =
      new QPushButton(QStringLiteral("Open manual download"), selfUpdateBox_);
  manualDownloadButton_->setObjectName(
      QStringLiteral("scInstallerSelfUpdateManual"));
  selfUpdateActions->addWidget(restartInstallerButton_);
  selfUpdateActions->addWidget(manualDownloadButton_);
  selfUpdateActions->addStretch(1);
  selfUpdateLayout->addLayout(selfUpdateActions);
  selfUpdateBox_->setVisible(false);
  layout->addWidget(selfUpdateBox_);

  connect(restartInstallerButton_, &QPushButton::clicked, this, [this] {
    QString error;
    if (!installerWizard(this)->launchVerifiedSelfUpdate(&error) &&
        !error.isEmpty()) {
      QMessageBox::warning(
          this, QStringLiteral("Installer restart unavailable"), error);
    }
  });
  connect(manualDownloadButton_, &QPushButton::clicked, this, [this] {
    const QUrl url(manualDownloadButton_->property("downloadUrl").toString());
    if (url.scheme() == QStringLiteral("https"))
      QDesktopServices::openUrl(url);
  });

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
  const QJsonObject selfUpdate =
      w->releaseStatusObject().value(QStringLiteral("self_update")).toObject();
  const QString selfUpdateState =
      jsonString(selfUpdate, QStringLiteral("state"));
  const QString manualUrl =
      jsonString(selfUpdate, QStringLiteral("manual_download_url"));
  const QUrl parsedManualUrl(manualUrl);
  const bool manualAvailable =
      parsedManualUrl.isValid() &&
      parsedManualUrl.scheme() == QStringLiteral("https");
  const bool restartAvailable =
      selfUpdateState == QStringLiteral("offer_restart") &&
      jsonBool(selfUpdate, QStringLiteral("requires_confirmation"));
  selfUpdateBox_->setVisible(
      restartAvailable || selfUpdateState == QStringLiteral("manual_download"));
  restartInstallerButton_->setVisible(restartAvailable);
  manualDownloadButton_->setVisible(manualAvailable);
  manualDownloadButton_->setProperty("downloadUrl", manualUrl);
  manualDownloadButton_->setToolTip(manualUrl);
  if (restartAvailable) {
    selfUpdateLabel_->setText(QStringLiteral(
        "A newer installer AppImage was downloaded and verified. Restart is "
        "offered explicitly; the running AppImage will not be replaced."));
  } else if (selfUpdateState == QStringLiteral("manual_download")) {
    selfUpdateLabel_->setText(
        QStringLiteral("Automatic installer restart is unavailable (%1). Use "
                       "the verified release's manual download link.")
            .arg(jsonString(selfUpdate, QStringLiteral("reason_code"),
                            QStringLiteral("self_update.unavailable"))));
  }

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
      QStringLiteral("Also remove settings, downloaded models, other user "
                     "data, logs, and caches"),
      this);
  removeUserData_->setObjectName(
      QStringLiteral("scInstallerUninstallRemoveUserData"));
  layout->addWidget(removeUserData_);

  dataWarning_ = mutedLabel(
      QStringLiteral("The additional user-data removal is permanent. Leave "
                     "this unchecked to preserve settings, downloaded "
                     "models, other user data, logs, and caches."),
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
    const QString dataPolicy =
        removeUserData_->isChecked()
            ? QStringLiteral(
                  "User data policy: REMOVE settings, downloaded models, "
                  "other user data, logs, and caches.\n\n")
            : QStringLiteral(
                  "User data policy: PRESERVE settings, downloaded models, "
                  "other user data, logs, and caches.\n\n");
    planText_->setPlainText(dataPolicy + planTextFromObject(w->planObject()));
  } else {
    planText_->setPlainText(error);
  }
  emit completeChanged();
}


} // namespace studiocast::installer
