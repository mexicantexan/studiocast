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

namespace studiocast::installer::detail {

QString defaultCacheBuildDir() {
  const QString cache =
      QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
  if (!cache.isEmpty()) {
    return QDir(cache).filePath(QStringLiteral("build"));
  }
  return QDir::home().filePath(QStringLiteral(".cache/studiocast/build"));
}

QStringList defaultModelPackIds() {
  return {QStringLiteral("fastenhancer_s_vd_v1"),
          QStringLiteral("fastenhancer_m_vd_v1"),
          QStringLiteral("modnet-webnn-256-fp32"),
          QStringLiteral("yunet_opencv_zoo_2023mar_fp32"),
          QStringLiteral("dlib_68_ibug_300w"),
          QStringLiteral("gaze_correction_cam_flx_v0_1_1"),
          QStringLiteral("fastdvdnet_sigma15")};
}

QString defaultModelDestination() {
  QString data =
      QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation);
  if (data.isEmpty())
    data = QDir::home().filePath(QStringLiteral(".local/share"));
  return QDir(data).filePath(QStringLiteral("studiocast/models"));
}

bool isUserLocalModelDestination(const QString &path) {
  const QString candidate = QDir::cleanPath(path.trimmed());
  const QString root = QDir::cleanPath(defaultModelDestination());
  return QFileInfo(candidate).isAbsolute() &&
         (candidate == root ||
          candidate.startsWith(root + QDir::separator()));
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
                   const QString &fallback) {
  const QJsonValue value = object.value(key);
  return value.isString() ? value.toString() : fallback;
}

bool jsonBool(const QJsonObject &object, const QString &key,
              bool fallback) {
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
    for (const QJsonValue &pack : packs) {
      text += QStringLiteral("  - %1\n").arg(pack.toString());
    }
    int artifactCount = 0;
    qint64 modelBytes = 0;
    qint64 allDownloadBytes = 0;
    for (const QJsonValue &value : plan.value(QStringLiteral("downloads")).toArray()) {
      const QJsonObject download = value.toObject();
      const qint64 bytes = static_cast<qint64>(
          download.value(QStringLiteral("size")).toDouble(-1));
      if (bytes > 0)
        allDownloadBytes += bytes;
      if (!download.contains(QStringLiteral("pack_id")))
        continue;
      ++artifactCount;
      if (bytes > 0)
        modelBytes += bytes;
      text += QStringLiteral("  - %1: %2 bytes\n")
                  .arg(jsonString(download, QStringLiteral("artifact_id"),
                                  QStringLiteral("unknown")))
                  .arg(bytes);
    }
    text += QStringLiteral("  %1 verified artifact files, %2 model bytes\n")
                .arg(artifactCount)
                .arg(modelBytes);
    const QJsonObject disk = plan.value(QStringLiteral("disk")).toObject();
    text += QStringLiteral("  Reviewed downloads: %1 bytes; disk required/free: %2/%3 bytes\n")
                .arg(allDownloadBytes)
                .arg(static_cast<qint64>(disk.value(QStringLiteral("required_bytes")).toDouble()))
                .arg(static_cast<qint64>(disk.value(QStringLiteral("free_bytes")).toDouble()));
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

QLabel *mutedLabel(const QString &text, QWidget *parent) {
  auto *label = new QLabel(text, parent);
  label->setWordWrap(true);
  label->setProperty("scRole", "muted");
  return label;
}

QFrame *line(QWidget *parent) {
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

} // namespace studiocast::installer::detail
