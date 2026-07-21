#include "engines_models_page.h"

#include <QCoreApplication>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QStandardPaths>
#include <QStyle>
#include <QVBoxLayout>

#include <algorithm>

#include "gui/status/daemon_status_snapshot.h"
#include "gui/text_edit_utils.h"

#ifndef STUDIOCAST_SOURCE_DIR
#define STUDIOCAST_SOURCE_DIR ""
#endif

namespace studiocast::gui {
namespace {

constexpr const char *kOpenAudioDefaultModelIds[] = {
    "fastenhancer_s_vd_v1",
    "fastenhancer_m_vd_v1",
};

constexpr const char *kOpenVideoDefaultModelIds[] = {
    "modnet-webnn-256-fp32", "yunet_opencv_zoo_2023mar_fp32",
    "dlib_68_ibug_300w",     "gaze_correction_cam_flx_v0_1_1",
    "fastdvdnet_sigma15",
};

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

QPlainTextEdit *DetailsBox(QWidget *parent, int minHeight = 94) {
  auto *text = new QPlainTextEdit(parent);
  text->setReadOnly(true);
  text->setLineWrapMode(QPlainTextEdit::WidgetWidth);
  text->setMinimumHeight(minHeight);
  return text;
}

void SetDynamicProperty(QWidget *widget, const char *name,
                        const QString &value) {
  if (!widget)
    return;
  widget->setProperty(name, value);
  widget->style()->unpolish(widget);
  widget->style()->polish(widget);
}

QString FriendlyBackendLabel(const QString &id) {
  const QString v = id.trimmed().toLower();
  if (v.isEmpty())
    return QStringLiteral("Unknown");
  if (v == QStringLiteral("auto"))
    return QStringLiteral("Auto");
  if (v == QStringLiteral("maxine"))
    return QStringLiteral("Maxine");
  if (v == QStringLiteral("open_cuda") || v == QStringLiteral("open_video"))
    return QStringLiteral("Open Video");
  if (v == QStringLiteral("open_vulkan") || v == QStringLiteral("vulkan"))
    return QStringLiteral("Open Vulkan");
  if (v == QStringLiteral("open_source") || v == QStringLiteral("open_audio"))
    return QStringLiteral("Open Audio");
  if (v == QStringLiteral("off"))
    return QStringLiteral("Off");
  if (v == QStringLiteral("passthrough"))
    return QStringLiteral("Pass-through");
  if (v == QStringLiteral("loopback"))
    return QStringLiteral("Loopback / pass-through");
  if (v == QStringLiteral("pipeline"))
    return QStringLiteral("Processed pipeline");
  return id;
}

QString ActiveBackendSummary(
    const QString &raw,
    const QString &emptyLabel = QStringLiteral("Pass-through / idle")) {
  const QString trimmed = raw.trimmed();
  if (trimmed.isEmpty())
    return emptyLabel;

  QStringList labels;
  const QStringList parts = trimmed.split(',', Qt::SkipEmptyParts);
  for (const QString &part : parts) {
    const QString p = part.trimmed();
    const qsizetype colon = p.indexOf(QChar(':'));
    const QString backend = colon >= 0 ? p.mid(colon + 1).trimmed() : p;
    const QString label = FriendlyBackendLabel(backend);
    if (!label.isEmpty() && !labels.contains(label))
      labels.push_back(label);
  }

  if (labels.isEmpty())
    return FriendlyBackendLabel(trimmed);
  if (labels.size() == 1)
    return labels.first();
  return QStringLiteral("Mixed: %1").arg(labels.join(QStringLiteral(", ")));
}

QString EngineProperty(const EngineStatus &engine) {
  if (!engine.present)
    return QStringLiteral("warning");
  if (!(engine.ok || engine.supported))
    return QStringLiteral("error");
  if (engine.missingModelCount > 0 || engine.configuredMissingModelCount > 0 ||
      !engine.blockedEffects.isEmpty()) {
    return QStringLiteral("warning");
  }
  return QStringLiteral("good");
}

QString ReasonCodeFromBlockedLine(const QString &line) {
  const QString trimmed = line.trimmed();
  const qsizetype colon = trimmed.lastIndexOf(QStringLiteral(": "));
  if (colon >= 0)
    return trimmed.mid(colon + 2).trimmed();
  return trimmed;
}

QStringList EngineBlockerCodes(const EngineStatus &engine) {
  QStringList codes;
  const auto add = [&](const QString &raw) {
    const QString code = ReasonCodeFromBlockedLine(raw);
    if (!code.isEmpty() && !codes.contains(code))
      codes.push_back(code);
  };
  add(engine.blockedReason);
  for (const QString &effect : engine.blockedEffects)
    add(effect);
  return codes;
}

bool EngineHasBlockerCode(const EngineStatus &engine, const QString &code) {
  return EngineBlockerCodes(engine).contains(code);
}

bool EngineHasSetupBlocker(const EngineStatus &engine) {
  if (engine.id != QStringLiteral("open_cuda") &&
      engine.id != QStringLiteral("open_audio")) {
    return false;
  }

  const QStringList codes = EngineBlockerCodes(engine);
  for (const QString &code : codes) {
    if (code == QStringLiteral("disabled_in_build") ||
        code == QStringLiteral("onnxruntime_not_found")) {
      return true;
    }
    if (engine.id == QStringLiteral("open_cuda") &&
        (code == QStringLiteral("onnxruntime_cuda_provider_unavailable") ||
         code == QStringLiteral("cuda_unavailable"))) {
      return true;
    }
  }
  return false;
}

QString EngineSetupReasonText(const EngineStatus &engine, const QString &code) {
  if (code == QStringLiteral("disabled_in_build")) {
    return QStringLiteral("%1 is disabled in the running StudioCast build.")
        .arg(engine.label);
  }
  if (code == QStringLiteral("onnxruntime_not_found")) {
    return QStringLiteral(
        "This StudioCast build was compiled without ONNX Runtime.");
  }
  if (engine.id == QStringLiteral("open_cuda") &&
      code == QStringLiteral("onnxruntime_cuda_provider_unavailable")) {
    return QStringLiteral(
        "ONNX Runtime is available, but CUDAExecutionProvider is not.");
  }
  if (engine.id == QStringLiteral("open_cuda") &&
      code == QStringLiteral("cuda_unavailable")) {
    return QStringLiteral("The daemon could not initialize CUDA.");
  }
  return {};
}

QString EngineSetupFixText(const EngineStatus &engine, const QString &code) {
  if (code == QStringLiteral("disabled_in_build")) {
    if (engine.id == QStringLiteral("open_audio")) {
      return QStringLiteral(
          "Fix: rebuild StudioCast with -DSTUDIOCAST_ENABLE_OPEN_AUDIO=ON, "
          "then restart the GUI and daemon.");
    }
    return QStringLiteral(
        "Fix: rebuild StudioCast with -DSTUDIOCAST_ENABLE_OPEN_CUDA=ON, then "
        "restart the GUI and daemon.");
  }
  if (code == QStringLiteral("onnxruntime_not_found")) {
    return QStringLiteral(
        "Fix: install or point CMake at ONNX Runtime, rebuild StudioCast, then "
        "restart the GUI and daemon.");
  }
  if (engine.id == QStringLiteral("open_cuda") &&
      code == QStringLiteral("onnxruntime_cuda_provider_unavailable")) {
    return QStringLiteral(
        "Fix: install or build ONNX Runtime with CUDAExecutionProvider "
        "support, "
        "rebuild StudioCast, then restart the GUI and daemon.");
  }
  if (engine.id == QStringLiteral("open_cuda") &&
      code == QStringLiteral("cuda_unavailable")) {
    return QStringLiteral("Fix: check the NVIDIA driver/CUDA runtime and "
                          "restart StudioCast after "
                          "CUDA initializes successfully.");
  }
  return {};
}

QStringList EngineSetupGuidanceLines(const EngineStatus &engine) {
  QStringList lines;
  if (!EngineHasSetupBlocker(engine))
    return lines;

  for (const QString &code : EngineBlockerCodes(engine)) {
    const QString reason = EngineSetupReasonText(engine, code);
    const QString fix = EngineSetupFixText(engine, code);
    if (!reason.isEmpty() && !lines.contains(reason))
      lines.push_back(reason);
    if (!fix.isEmpty() && !lines.contains(fix))
      lines.push_back(fix);
  }

  if (EngineHasBlockerCode(engine, QStringLiteral("disabled_in_build"))) {
    const QString flag = engine.id == QStringLiteral("open_audio")
                             ? QStringLiteral("STUDIOCAST_ENABLE_OPEN_AUDIO")
                             : QStringLiteral("STUDIOCAST_ENABLE_OPEN_CUDA");
    const QString command =
        QStringLiteral("Source build command: cmake -S . -B build -D%1=ON && "
                       "cmake --build build --target studiocast studiocastd")
            .arg(flag);
    if (!lines.contains(command))
      lines.push_back(command);
  }

  return lines;
}

QString EngineSetupSummary(const EngineStatus &engine) {
  const QStringList lines = EngineSetupGuidanceLines(engine);
  if (lines.isEmpty())
    return {};

  QString summary = lines.first();
  if (lines.size() > 1)
    summary += QChar(' ') + lines.at(1);
  return summary;
}

QString EngineSetupDisclaimerText(const EngineStatus &engine) {
  const QStringList lines = EngineSetupGuidanceLines(engine);
  if (lines.isEmpty())
    return {};
  return lines.join(QStringLiteral("\n"));
}

QString EngineStateLabel(const EngineStatus &engine,
                         bool selectedByPreference) {
  if (!engine.present)
    return QStringLiteral("Unknown");
  if (!(engine.ok || engine.supported)) {
    if (EngineHasSetupBlocker(engine)) {
      return selectedByPreference ? QStringLiteral("Selected setup required")
                                  : QStringLiteral("Setup required");
    }
    return selectedByPreference ? QStringLiteral("Selected unavailable")
                                : QStringLiteral("Unavailable");
  }
  if (engine.missingModelCount > 0) {
    return QStringLiteral("Missing models");
  }
  if (engine.configuredMissingModelCount > 0)
    return QStringLiteral("Model selection review");
  if (EngineHasSetupBlocker(engine))
    return QStringLiteral("Partial setup required");
  if (!engine.blockedEffects.isEmpty())
    return QStringLiteral("Partially blocked");
  return QStringLiteral("Available");
}

QString EngineSummary(const EngineStatus &engine, bool selectedByPreference) {
  if (!engine.present)
    return QStringLiteral("The daemon did not report %1 diagnostics.")
        .arg(engine.label);

  if (!(engine.ok || engine.supported)) {
    if (EngineHasSetupBlocker(engine))
      return selectedByPreference
                 ? QStringLiteral("This backend is currently selected.")
                 : QStringLiteral("%1 requires setup before it can run.")
                       .arg(engine.label);

    QString summary = engine.summary.trimmed();
    if (summary.isEmpty())
      summary = QStringLiteral("%1 is unavailable.").arg(engine.label);
    if (selectedByPreference)
      summary += QStringLiteral(" This backend is currently selected.");
    return summary;
  }

  const QString setup = EngineSetupSummary(engine);
  if (!setup.isEmpty()) {
    return QStringLiteral("%1 is available, but some effects require setup.")
        .arg(engine.label);
  }

  if (engine.missingModelCount > 0) {
    if (engine.installedModelCount > 0) {
      return QStringLiteral("%1 has usable model packs, but %2 model pack%3 "
                            "are missing or invalid.")
          .arg(engine.label)
          .arg(engine.missingModelCount)
          .arg(engine.missingModelCount == 1 ? QString() : QStringLiteral("s"));
    }
    return QStringLiteral("%1 model packs are missing or invalid.")
        .arg(engine.label);
  }

  if (engine.configuredMissingModelCount > 0) {
    return QStringLiteral("Some configured %1 model selections are not "
                          "reported installed by daemon diagnostics.")
        .arg(engine.label);
  }

  if (engine.id == QStringLiteral("open_cuda") &&
      engine.installedModelCount == 0) {
    return QStringLiteral("Open Video runtime is available. Model-backed "
                          "camera effects still need model packs.");
  }

  if (engine.id == QStringLiteral("open_audio") &&
      engine.installedModelCount == 0) {
    return QStringLiteral("Open Audio runtime is available, but no usable "
                          "audio model packs are installed.");
  }

  if (!engine.blockedEffects.isEmpty()) {
    return QStringLiteral("%1 is available, but some effects are blocked.")
        .arg(engine.label);
  }

  return engine.summary.trimmed().isEmpty()
             ? QStringLiteral("%1 is available.").arg(engine.label)
             : engine.summary.trimmed();
}

QString ModelSummary(const EngineStatus &engine) {
  if (!engine.present)
    return QStringLiteral("No model diagnostics reported.");

  QStringList parts;
  if (engine.id == QStringLiteral("maxine")) {
    if (engine.installedModelCount == 0 && engine.missingModelCount == 0) {
      parts << QStringLiteral("No Maxine feature status reported.");
    } else {
      parts << QStringLiteral("%1 feature file%2 installed")
                   .arg(engine.installedModelCount)
                   .arg(engine.installedModelCount == 1 ? QString()
                                                        : QStringLiteral("s"));
    }
  } else if (engine.installedModelCount == 0) {
    parts << QStringLiteral("No installed model packs reported");
  } else {
    parts << QStringLiteral("%1 installed model pack%2")
                 .arg(engine.installedModelCount)
                 .arg(engine.installedModelCount == 1 ? QString()
                                                      : QStringLiteral("s"));
  }

  if (engine.missingModelCount > 0) {
    parts << QStringLiteral("%1 missing or invalid")
                 .arg(engine.missingModelCount);
  }
  if (engine.configuredMissingModelCount > 0) {
    parts << QStringLiteral("%1 configured selection%2 not reported installed")
                 .arg(engine.configuredMissingModelCount)
                 .arg(engine.configuredMissingModelCount == 1
                          ? QString()
                          : QStringLiteral("s"));
  }
  return parts.join(QStringLiteral("; ")) + QChar('.');
}

QString ModelEntryLine(const EngineModelEntry &entry) {
  QString line = entry.displayName.trimmed().isEmpty()
                     ? entry.id.trimmed()
                     : entry.displayName.trimmed();
  QStringList suffix;
  if (!entry.category.trimmed().isEmpty())
    suffix << entry.category.trimmed();
  if (!entry.id.trimmed().isEmpty())
    suffix << QStringLiteral("id: %1").arg(entry.id.trimmed());
  if (!entry.details.trimmed().isEmpty())
    suffix << entry.details.trimmed();
  if (!suffix.isEmpty())
    line += QStringLiteral(" (%1)").arg(suffix.join(QStringLiteral("; ")));
  return line;
}

QString ConfiguredModelLine(const EngineStatus &engine,
                            const ConfiguredModelEntry &entry) {
  QStringList parts;
  if (!entry.modelId.isEmpty()) {
    QString state;
    if (entry.modelIdReported) {
      state = QStringLiteral("reported installed");
    } else if (entry.modelIdExplicitlyMissing) {
      state = QStringLiteral("reported missing or invalid");
    } else if (engine.present) {
      state = QStringLiteral("not reported by daemon diagnostics");
    } else {
      state = QStringLiteral("engine diagnostics unavailable");
    }
    parts << QStringLiteral("model_id=%1 (%2)").arg(entry.modelId, state);
  }
  if (!entry.modelPath.isEmpty())
    parts << QStringLiteral("model_path=%1").arg(entry.modelPath);
  return QStringLiteral("%1: %2").arg(entry.owner,
                                      parts.join(QStringLiteral("; ")));
}

QString EngineDetailsText(const EngineStatus &engine) {
  QStringList lines;
  lines << QStringLiteral("Summary: %1")
               .arg(engine.summary.trimmed().isEmpty()
                        ? QStringLiteral("No summary reported.")
                        : engine.summary.trimmed());
  if (!engine.blockedReason.trimmed().isEmpty())
    lines << QStringLiteral("Blocked reason: %1").arg(engine.blockedReason);
  if (!engine.blockedDetails.isEmpty()) {
    lines << QStringLiteral("");
    lines << QStringLiteral("Blocked details:");
    for (const QString &detail : engine.blockedDetails)
      lines << QStringLiteral("- %1").arg(detail);
  }
  const QStringList setupGuidance = EngineSetupGuidanceLines(engine);
  if (!setupGuidance.isEmpty()) {
    lines << QStringLiteral("");
    lines << QStringLiteral("Setup guidance:");
    for (const QString &line : setupGuidance)
      lines << QStringLiteral("- %1").arg(line);
  }
  if (!engine.availableEffects.isEmpty()) {
    lines << QStringLiteral("");
    lines << QStringLiteral("Available effects:");
    for (const QString &effect : engine.availableEffects)
      lines << QStringLiteral("- %1").arg(effect);
  }
  if (!engine.installedModels.empty()) {
    lines << QStringLiteral("");
    lines << QStringLiteral("Installed model details:");
    for (const EngineModelEntry &entry : engine.installedModels)
      lines << QStringLiteral("- %1").arg(ModelEntryLine(entry));
  }
  if (!engine.missingModelEntries.empty()) {
    lines << QStringLiteral("");
    lines << QStringLiteral("Missing or invalid model details:");
    for (const EngineModelEntry &entry : engine.missingModelEntries)
      lines << QStringLiteral("- %1").arg(ModelEntryLine(entry));
  }
  if (!engine.configuredModels.empty()) {
    lines << QStringLiteral("");
    lines << QStringLiteral("Configured model selections:");
    for (const ConfiguredModelEntry &entry : engine.configuredModels)
      lines << QStringLiteral("- %1").arg(ConfiguredModelLine(engine, entry));
  }
  if (!engine.blockedEffects.isEmpty()) {
    lines << QStringLiteral("");
    lines << QStringLiteral("Blocked effects:");
    for (const QString &effect : engine.blockedEffects)
      lines << QStringLiteral("- %1").arg(effect);
  }
  if (!engine.defaultModelId.trimmed().isEmpty()) {
    lines << QStringLiteral("");
    lines << QStringLiteral("Default model ID: %1").arg(engine.defaultModelId);
  }
  return lines.join(QStringLiteral("\n")).trimmed();
}

QStringList DefaultModelIdsForEngine(const QString &engineId) {
  QStringList ids;
  if (engineId == QStringLiteral("open_audio")) {
    for (const char *id : kOpenAudioDefaultModelIds)
      ids.push_back(QString::fromLatin1(id));
  } else if (engineId == QStringLiteral("open_cuda")) {
    for (const char *id : kOpenVideoDefaultModelIds)
      ids.push_back(QString::fromLatin1(id));
  }
  return ids;
}

QStringList MissingModelIdsForInstall(const EngineStatus &engine) {
  QStringList ids;
  for (const EngineModelEntry &entry : engine.missingModelEntries) {
    const QString id = entry.id.trimmed();
    if (!id.isEmpty() && !ids.contains(id))
      ids.push_back(id);
  }
  return ids;
}

QStringList ModelInstallArgsForEngine(const EngineStatus &engine) {
  QStringList args;
  if (engine.id == QStringLiteral("open_audio")) {
    args << QStringLiteral("open-audio-models");
  } else if (engine.id == QStringLiteral("open_cuda")) {
    args << QStringLiteral("open-video-models");
  } else {
    return args;
  }

  QStringList modelIds = MissingModelIdsForInstall(engine);
  if (modelIds.isEmpty())
    modelIds = DefaultModelIdsForEngine(engine.id);

  if (engine.id == QStringLiteral("open_cuda")) {
    const bool installsEyeContact =
        std::any_of(modelIds.cbegin(), modelIds.cend(), [](const QString &id) {
          return id.startsWith(QStringLiteral("gaze_correction_cam"));
        });
    if (installsEyeContact &&
        !modelIds.contains(QStringLiteral("dlib_68_ibug_300w"))) {
      modelIds.push_back(QStringLiteral("dlib_68_ibug_300w"));
    }
  }

  for (const QString &id : modelIds)
    args << QStringLiteral("--model") << id;
  return args;
}

bool HasMissingModelBlock(const EngineStatus &engine) {
  for (const QString &effect : engine.blockedEffects) {
    if (effect.contains(QStringLiteral("missing_model_packs")))
      return true;
  }
  return false;
}

bool ModelInstallRecommended(const EngineStatus &engine) {
  if (engine.id != QStringLiteral("open_audio") &&
      engine.id != QStringLiteral("open_cuda")) {
    return false;
  }
  if (!engine.present)
    return false;
  if (EngineHasSetupBlocker(engine))
    return false;
  if (engine.missingModelCount > 0 || engine.configuredMissingModelCount > 0 ||
      HasMissingModelBlock(engine)) {
    return true;
  }
  return (engine.ok || engine.supported) && engine.installedModelCount == 0;
}

QString ModelInstallButtonText(const EngineStatus &engine) {
  const QString label = engine.id == QStringLiteral("open_audio")
                            ? QStringLiteral("Open Audio")
                            : QStringLiteral("Open Video");
  if (engine.missingModelCount > 0)
    return QStringLiteral("Download missing %1 models").arg(label);
  return QStringLiteral("Download default %1 models").arg(label);
}

QString ModelInstallStatusText(const EngineStatus &engine,
                               bool installRecommended) {
  if (engine.id != QStringLiteral("open_audio") &&
      engine.id != QStringLiteral("open_cuda")) {
    return {};
  }
  if (!engine.present)
    return QStringLiteral("Model diagnostics unavailable.");
  if (EngineHasSetupBlocker(engine)) {
    return QStringLiteral("Resolve the %1 setup issue above before "
                          "downloading model packs.")
        .arg(engine.label);
  }
  if (!installRecommended) {
    if (!(engine.ok || engine.supported) && engine.installedModelCount == 0) {
      return QStringLiteral(
          "Resolve runtime/build issues before downloading model packs.");
    }
    return QStringLiteral("Default model packs appear installed.");
  }

  const QStringList args = ModelInstallArgsForEngine(engine);
  QStringList modelIds;
  for (int i = 0; i + 1 < args.size(); ++i) {
    if (args.at(i) == QStringLiteral("--model"))
      modelIds.push_back(args.at(i + 1));
  }
  if (!modelIds.isEmpty()) {
    return QStringLiteral("Ready to install: %1.")
        .arg(modelIds.join(QStringLiteral(", ")));
  }
  return QStringLiteral("Ready to install default model packs.");
}

QString ManifestSourceDir() {
  const QString dataHome =
      QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation);
  if (dataHome.isEmpty())
    return {};

  QFile file(QDir(dataHome).filePath(
      QStringLiteral("studiocast/install-manifest.json")));
  if (!file.open(QIODevice::ReadOnly))
    return {};

  const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
  if (!doc.isObject())
    return {};
  return doc.object().value(QStringLiteral("source_path")).toString().trimmed();
}

void AddInstallScriptCandidate(QStringList *candidates,
                               const QString &sourceDir) {
  if (!candidates || sourceDir.trimmed().isEmpty())
    return;
  const QString script =
      QDir(sourceDir.trimmed()).filePath(QStringLiteral("scripts/install.sh"));
  if (!candidates->contains(script))
    candidates->push_back(script);
}

void AddInstallerBackendCandidate(QStringList *candidates,
                                  const QString &sourceDir) {
  if (!candidates || sourceDir.trimmed().isEmpty())
    return;
  const QString backend =
      QDir(sourceDir.trimmed())
          .filePath(QStringLiteral("installer/backend/"
                                   "studiocast-installer-backend"));
  if (!candidates->contains(backend))
    candidates->push_back(backend);
}

QString InstallHintsText(const EngineStatus &engine) {
  if (engine.installHints.isEmpty())
    return QStringLiteral("No install hints reported by daemon diagnostics.");
  QStringList lines;
  for (const QString &hint : engine.installHints)
    lines << hint;
  return lines.join(QStringLiteral("\n"));
}

bool PreferenceSelectsMaxine(const DaemonStatusSnapshot &snapshot) {
  const QString video =
      snapshot.videoEffectsEnginePreference.trimmed().toLower();
  const QString audio =
      snapshot.audioEffectsEnginePreference.trimmed().toLower();
  return video == QStringLiteral("maxine") || audio == QStringLiteral("maxine");
}

bool PreferenceSelectsOpenVideo(const DaemonStatusSnapshot &snapshot) {
  return snapshot.videoEffectsEnginePreference.trimmed().toLower() ==
         QStringLiteral("open_cuda");
}

bool PreferenceSelectsOpenAudio(const DaemonStatusSnapshot &snapshot) {
  const QString audio =
      snapshot.audioEffectsEnginePreference.trimmed().toLower();
  return audio == QStringLiteral("open_source") ||
         audio == QStringLiteral("open_audio");
}

QString RepairSetupButtonText(const QString &engineId) {
  if (engineId == QStringLiteral("open_audio"))
    return QStringLiteral("Repair Open Audio setup");
  return QStringLiteral("Repair Open Video setup");
}

QStringList SetupRepairArgsForEngine(const EngineStatus &engine) {
  if (!EngineHasSetupBlocker(engine))
    return {};

  if (engine.id != QStringLiteral("open_cuda") &&
      engine.id != QStringLiteral("open_audio")) {
    return {};
  }

  return {
      QStringLiteral("repair"),      QStringLiteral("--open-backends"),
      QStringLiteral("--with-deps"), QStringLiteral("--no-models"),
      QStringLiteral("--service"),   QStringLiteral("--yes"),
  };
}

QString TailForDialog(const QString &text, qsizetype maxChars = 5000) {
  QString trimmed = text.trimmed();
  if (trimmed.size() > maxChars)
    trimmed = QStringLiteral("...\n") + trimmed.right(maxChars);
  return trimmed;
}

QLabel *DialogTitleLabel(const QString &text, QWidget *parent) {
  auto *label = new QLabel(text, parent);
  label->setProperty("scRole", "homeCardTitle");
  label->setWordWrap(true);
  return label;
}

QLabel *DialogBanner(const QString &text, const QString &status,
                     QWidget *parent) {
  auto *label = new QLabel(text, parent);
  label->setProperty("scBanner", status);
  label->setWordWrap(true);
  return label;
}

void SetPrimaryDialogButton(QPushButton *button) {
  if (!button)
    return;
  button->setProperty("scVariant", "primary");
}

bool ConfirmSetupRepairDialog(QWidget *parent, const QString &planText) {
  QDialog dialog(parent);
  dialog.setObjectName(QStringLiteral("setupRepairConfirmDialog"));
  dialog.setWindowTitle(QStringLiteral("StudioCast Setup Repair"));
  dialog.setModal(true);
  dialog.resize(720, 560);

  auto *root = new QVBoxLayout(&dialog);
  root->setContentsMargins(18, 18, 18, 18);
  root->setSpacing(12);

  root->addWidget(
      DialogTitleLabel(QStringLiteral("Run setup repair now?"), &dialog));
  root->addWidget(MutedLabel(
      QStringLiteral("StudioCast will refresh prerequisites, configure and "
                     "rebuild Open Video/Open CUDA and Open Audio support, "
                     "reinstall the user service, and skip model downloads."),
      &dialog));
  root->addWidget(DialogBanner(
      QStringLiteral("Some repair steps may need sudo. If your password is "
                     "required, StudioCast will pause the repair and show a "
                     "password prompt in this GUI."),
      QStringLiteral("warning"), &dialog));

  auto *details = DetailsBox(&dialog, 280);
  details->setObjectName(QStringLiteral("setupRepairPlanDetails"));
  details->setPlainText(TailForDialog(planText, 12000));
  root->addWidget(details, 1);

  auto *buttons = new QDialogButtonBox(QDialogButtonBox::Cancel, &dialog);
  auto *runButton = buttons->addButton(QStringLiteral("Run Repair"),
                                       QDialogButtonBox::AcceptRole);
  SetPrimaryDialogButton(runButton);
  runButton->setDefault(true);
  QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog,
                   &QDialog::accept);
  QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog,
                   &QDialog::reject);
  root->addWidget(buttons);

  return dialog.exec() == QDialog::Accepted;
}

void ShowSetupRepairDialog(QWidget *parent, const QString &heading,
                           const QString &message, const QString &detailsText,
                           const QString &status) {
  QDialog dialog(parent);
  dialog.setObjectName(QStringLiteral("setupRepairResultDialog"));
  dialog.setWindowTitle(QStringLiteral("StudioCast Setup Repair"));
  dialog.setModal(true);
  dialog.resize(680, detailsText.trimmed().isEmpty() ? 260 : 520);

  auto *root = new QVBoxLayout(&dialog);
  root->setContentsMargins(18, 18, 18, 18);
  root->setSpacing(12);
  root->addWidget(DialogTitleLabel(heading, &dialog));
  root->addWidget(DialogBanner(message, status, &dialog));

  if (!detailsText.trimmed().isEmpty()) {
    auto *details = DetailsBox(&dialog, 260);
    details->setObjectName(QStringLiteral("setupRepairResultDetails"));
    details->setPlainText(detailsText);
    root->addWidget(details, 1);
  }

  auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close, &dialog);
  auto *closeButton = buttons->button(QDialogButtonBox::Close);
  SetPrimaryDialogButton(closeButton);
  QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog,
                   &QDialog::reject);
  root->addWidget(buttons);
  dialog.exec();
}

bool LooksLikeSudoPasswordPrompt(const QString &text) {
  const QString lower = text.toLower();
  return lower.contains(QStringLiteral("[sudo] password")) ||
         lower.contains(QStringLiteral("sudo password")) ||
         (lower.contains(QStringLiteral("password for ")) &&
          lower.contains(QChar(':')));
}

} // namespace

EnginesModelsPage::EnginesModelsPage(QWidget *parent) : QWidget(parent) {
  auto *root = new QVBoxLayout(this);
  root->setContentsMargins(16, 16, 16, 16);
  root->setSpacing(12);

  auto *backendBox = new QGroupBox(QStringLiteral("Backend Selection"), this);
  auto *backendGrid = new QGridLayout(backendBox);
  backendGrid->setColumnStretch(0, 0);
  backendGrid->setColumnStretch(1, 1);
  backendGrid->setColumnStretch(2, 1);
  backendGrid->setHorizontalSpacing(18);
  backendGrid->setVerticalSpacing(8);

  backendGrid->addWidget(
      MutedLabel(QStringLiteral("Camera effects"), backendBox), 0, 0);
  backendGrid->addWidget(MutedLabel(QStringLiteral("Preference"), backendBox),
                         0, 1);
  backendGrid->addWidget(
      MutedLabel(QStringLiteral("Active backend"), backendBox), 0, 2);
  videoPreferenceLabel_ = ValueLabel(QStringLiteral("Unknown"), backendBox);
  videoActiveLabel_ = ValueLabel(QStringLiteral("Unknown"), backendBox);
  videoActiveLabel_->setObjectName(
      QStringLiteral("enginesVideoActiveBackendValue"));
  backendGrid->addWidget(videoPreferenceLabel_, 1, 1);
  backendGrid->addWidget(videoActiveLabel_, 1, 2);

  backendGrid->addWidget(
      MutedLabel(QStringLiteral("Audio cleanup"), backendBox), 2, 0);
  backendGrid->addWidget(MutedLabel(QStringLiteral("Preference"), backendBox),
                         2, 1);
  backendGrid->addWidget(
      MutedLabel(QStringLiteral("Active backends"), backendBox), 2, 2);
  audioPreferenceLabel_ = ValueLabel(QStringLiteral("Unknown"), backendBox);
  auto *activeAudio = new QWidget(backendBox);
  auto *activeAudioLayout = new QVBoxLayout(activeAudio);
  activeAudioLayout->setContentsMargins(0, 0, 0, 0);
  activeAudioLayout->setSpacing(2);
  microphoneActiveLabel_ =
      ValueLabel(QStringLiteral("Microphone: Unknown"), activeAudio);
  speakersActiveLabel_ =
      ValueLabel(QStringLiteral("Speakers: Unknown"), activeAudio);
  activeAudioLayout->addWidget(microphoneActiveLabel_);
  activeAudioLayout->addWidget(speakersActiveLabel_);
  backendGrid->addWidget(audioPreferenceLabel_, 3, 1);
  backendGrid->addWidget(activeAudio, 3, 2);
  root->addWidget(backendBox);

  maxineCard_ = CreateEngineCard(QStringLiteral("Maxine"),
                                 QStringLiteral("maxine"), this);
  openVideoCard_ = CreateEngineCard(QStringLiteral("Open Video / Open CUDA"),
                                    QStringLiteral("open_cuda"), this);
  openAudioCard_ = CreateEngineCard(QStringLiteral("Open Audio"),
                                    QStringLiteral("open_audio"), this);
  connect(openVideoCard_.downloadButton, &QPushButton::clicked, this,
          [this] { StartModelInstall(&openVideoCard_); });
  connect(openAudioCard_.downloadButton, &QPushButton::clicked, this,
          [this] { StartModelInstall(&openAudioCard_); });
  connect(openVideoCard_.repairSetupButton, &QPushButton::clicked, this,
          [this] { StartSetupRepair(&openVideoCard_); });
  connect(openAudioCard_.repairSetupButton, &QPushButton::clicked, this,
          [this] { StartSetupRepair(&openAudioCard_); });
  root->addWidget(maxineCard_.frame);
  root->addWidget(openVideoCard_.frame);
  root->addWidget(openAudioCard_.frame);
  root->addStretch(1);
}

EnginesModelsPage::EngineCard
EnginesModelsPage::CreateEngineCard(const QString &title,
                                    const QString &engineId, QWidget *parent) {
  EngineCard card;
  card.engineId = engineId;
  card.frame = new QFrame(parent);
  card.frame->setObjectName(engineId + QStringLiteral("_engine_card"));
  card.frame->setProperty("scRole", "engineCard");
  card.frame->setProperty("scStatus", "warning");
  auto *layout = new QVBoxLayout(card.frame);
  layout->setContentsMargins(14, 14, 14, 14);
  layout->setSpacing(10);

  auto *header = new QHBoxLayout();
  header->setContentsMargins(0, 0, 0, 0);
  header->setSpacing(10);
  card.title = ValueLabel(title, card.frame);
  card.title->setObjectName(engineId + QStringLiteral("_engine_title"));
  card.title->setProperty("scRole", "homeCardTitle");
  card.state = new QLabel(QStringLiteral("Unknown"), card.frame);
  card.state->setObjectName(engineId + QStringLiteral("_engine_state"));
  card.state->setProperty("scRole", "statusPill");
  card.state->setProperty("scStatus", "warning");
  card.state->setAlignment(Qt::AlignCenter);
  header->addWidget(card.title, 1);
  header->addWidget(card.state, 0);
  layout->addLayout(header);

  card.summary =
      MutedLabel(QStringLiteral("Diagnostics have not been read."), card.frame);
  card.summary->setObjectName(engineId + QStringLiteral("_engine_summary"));
  card.setupDisclaimerBanner = new QFrame(card.frame);
  card.setupDisclaimerBanner->setObjectName(
      engineId + QStringLiteral("_setup_disclaimer_banner"));
  card.setupDisclaimerBanner->setProperty("scBanner", "warning");
  card.setupDisclaimerBanner->setVisible(false);
  auto *setupLayout = new QVBoxLayout(card.setupDisclaimerBanner);
  setupLayout->setContentsMargins(12, 10, 12, 10);
  setupLayout->setSpacing(8);
  card.setupDisclaimer = new QLabel(card.setupDisclaimerBanner);
  card.setupDisclaimer->setObjectName(engineId +
                                      QStringLiteral("_setup_disclaimer"));
  card.setupDisclaimer->setWordWrap(true);
  card.setupDisclaimer->setVisible(false);
  card.repairSetupButton = new QPushButton(RepairSetupButtonText(engineId),
                                           card.setupDisclaimerBanner);
  card.repairSetupButton->setObjectName(engineId +
                                        QStringLiteral("_repair_setup_button"));
  card.repairSetupButton->setVisible(false);
  card.repairSetupStatus = MutedLabel(QString(), card.setupDisclaimerBanner);
  card.repairSetupStatus->setObjectName(engineId +
                                        QStringLiteral("_repair_setup_status"));
  card.repairSetupStatus->setVisible(false);
  auto *repairRow = new QHBoxLayout();
  repairRow->setContentsMargins(0, 0, 0, 0);
  repairRow->setSpacing(10);
  repairRow->addWidget(card.repairSetupButton, 0);
  repairRow->addWidget(card.repairSetupStatus, 1);
  setupLayout->addWidget(card.setupDisclaimer);
  setupLayout->addLayout(repairRow);
  card.models =
      MutedLabel(QStringLiteral("No model diagnostics reported."), card.frame);
  card.models->setObjectName(engineId + QStringLiteral("_engine_models"));
  layout->addWidget(card.summary);
  layout->addWidget(card.setupDisclaimerBanner);
  layout->addWidget(card.models);

  auto *actions = new QHBoxLayout();
  actions->setContentsMargins(0, 0, 0, 0);
  actions->setSpacing(10);
  card.downloadButton =
      new QPushButton(QStringLiteral("Download default models"), card.frame);
  card.downloadButton->setObjectName(engineId +
                                     QStringLiteral("_download_button"));
  card.downloadButton->setProperty("scVariant", "primary");
  card.downloadButton->setVisible(engineId == QStringLiteral("open_cuda") ||
                                  engineId == QStringLiteral("open_audio"));
  card.downloadButton->setEnabled(false);
  card.downloadStatus = MutedLabel(QString(), card.frame);
  card.downloadStatus->setObjectName(engineId +
                                     QStringLiteral("_download_status"));
  card.downloadStatus->setVisible(card.downloadButton->isVisible());
  actions->addWidget(card.downloadButton, 0);
  actions->addWidget(card.downloadStatus, 1);
  layout->addLayout(actions);

  auto *detailsGrid = new QGridLayout();
  detailsGrid->setContentsMargins(0, 0, 0, 0);
  detailsGrid->setHorizontalSpacing(10);
  detailsGrid->setVerticalSpacing(8);
  detailsGrid->addWidget(MutedLabel(QStringLiteral("Details"), card.frame), 0,
                         0);
  detailsGrid->addWidget(
      MutedLabel(QStringLiteral("Install Hints"), card.frame), 0, 1);
  card.details = DetailsBox(card.frame, 118);
  card.details->setObjectName(engineId + QStringLiteral("_engine_details"));
  card.installHints = DetailsBox(card.frame, 118);
  card.installHints->setObjectName(engineId + QStringLiteral("_install_hints"));
  detailsGrid->addWidget(card.details, 1, 0);
  detailsGrid->addWidget(card.installHints, 1, 1);
  layout->addLayout(detailsGrid);

  layout->addWidget(
      MutedLabel(QStringLiteral("Raw Engine Diagnostics"), card.frame));
  card.rawDetails = DetailsBox(card.frame, 130);
  card.rawDetails->setObjectName(engineId + QStringLiteral("_raw_details"));
  layout->addWidget(card.rawDetails);

  return card;
}

void EnginesModelsPage::UpdateEngineCard(EngineCard *card,
                                         const EngineStatus &engine,
                                         bool selectedByPreference) {
  if (!card)
    return;

  const QString prop = EngineProperty(engine);
  card->state->setText(EngineStateLabel(engine, selectedByPreference));
  SetDynamicProperty(card->state, "scStatus", prop);
  SetDynamicProperty(card->frame, "scStatus", prop);

  card->summary->setText(EngineSummary(engine, selectedByPreference));
  if (card->setupDisclaimerBanner && card->setupDisclaimer) {
    const QString disclaimer = EngineSetupDisclaimerText(engine);
    card->setupDisclaimer->setText(disclaimer);
    const bool showDisclaimer = !disclaimer.isEmpty();
    card->setupDisclaimer->setVisible(showDisclaimer);
    card->setupDisclaimerBanner->setVisible(showDisclaimer);
    if (!showDisclaimer && card->repairSetupStatus) {
      card->repairSetupStatus->clear();
      card->repairSetupStatus->setVisible(false);
    }
  }
  if (card->repairSetupButton && card->repairSetupStatus) {
    const bool repairRelevant = EngineHasSetupBlocker(engine);
    card->repairArgs = SetupRepairArgsForEngine(engine);
    card->repairRecommended = repairRelevant && !card->repairArgs.isEmpty();
    card->repairSetupButton->setText(RepairSetupButtonText(engine.id));
    card->repairSetupButton->setToolTip(
        card->repairArgs.isEmpty()
            ? QStringLiteral("No setup repair command is available.")
            : QStringLiteral("studiocast-installer-backend %1")
                  .arg(card->repairArgs.join(QStringLiteral(" "))));
    card->repairSetupButton->setVisible(repairRelevant);
    card->repairSetupStatus->setVisible(
        repairRelevant && !card->repairSetupStatus->text().isEmpty());
    if (!repairRelevant) {
      card->repairArgs.clear();
      card->repairRecommended = false;
      card->repairSetupStatus->clear();
    }
  }
  card->models->setText(ModelSummary(engine));
  if (card->downloadButton && card->downloadStatus) {
    const bool relevant = engine.id == QStringLiteral("open_cuda") ||
                          engine.id == QStringLiteral("open_audio");
    card->downloadButton->setVisible(relevant);
    card->downloadStatus->setVisible(relevant);
    if (relevant) {
      card->installArgs = ModelInstallArgsForEngine(engine);
      card->installRecommended = ModelInstallRecommended(engine);
      card->downloadButton->setText(ModelInstallButtonText(engine));
      card->downloadButton->setToolTip(
          card->installArgs.isEmpty()
              ? QStringLiteral("No model installer command is available.")
              : QStringLiteral("./scripts/install.sh %1")
                    .arg(card->installArgs.join(QStringLiteral(" "))));
      card->downloadStatus->setText(
          ModelInstallStatusText(engine, card->installRecommended));
    } else {
      card->installArgs.clear();
      card->installRecommended = false;
      card->downloadStatus->clear();
    }
  }
  SetPlainTextPreservingScroll(card->details, EngineDetailsText(engine));
  SetPlainTextPreservingScroll(card->installHints, InstallHintsText(engine));
  SetPlainTextPreservingScroll(
      card->rawDetails,
      engine.rawJson.trimmed().isEmpty()
          ? QStringLiteral("No raw engine diagnostics reported.")
          : engine.rawJson);
  RefreshDownloadButtons();
}

void EnginesModelsPage::RefreshDownloadButtons() {
  const bool modelInstallRunning = modelInstallProcess_ != nullptr;
  const bool setupRepairRunning = setupRepairProcess_ != nullptr;
  const bool anyInstallerRunning = modelInstallRunning || setupRepairRunning;
  auto update = [&](EngineCard *card) {
    if (!card || !card->downloadButton || card->downloadButton->isHidden())
      return;

    const bool enabled = !anyInstallerRunning && card->installRecommended &&
                         !card->installArgs.isEmpty();
    card->downloadButton->setEnabled(enabled);
    if (modelInstallRunning && card == activeInstallCard_ &&
        card->downloadStatus) {
      card->downloadStatus->setText(
          QStringLiteral("Downloading model packs..."));
    } else if (anyInstallerRunning && card->downloadStatus) {
      card->downloadStatus->setText(
          setupRepairRunning
              ? QStringLiteral("Setup repair is running.")
              : QStringLiteral("Another model download is running."));
    }

    if (card->repairSetupButton) {
      const bool repairEnabled = !anyInstallerRunning &&
                                 card->repairRecommended &&
                                 !card->repairArgs.isEmpty();
      card->repairSetupButton->setEnabled(repairEnabled);
    }
    if (setupRepairRunning && card == activeRepairCard_ &&
        card->repairSetupStatus) {
      card->repairSetupStatus->setVisible(true);
    }
  };

  update(&openVideoCard_);
  update(&openAudioCard_);
}

QString EnginesModelsPage::ResolveInstallScript(QString *error) const {
  QStringList candidates;
  const QString envPath =
      QString::fromLocal8Bit(qgetenv("STUDIOCAST_INSTALL_SCRIPT")).trimmed();
  if (!envPath.isEmpty())
    candidates.push_back(envPath);

  AddInstallScriptCandidate(&candidates,
                            QString::fromUtf8(STUDIOCAST_SOURCE_DIR));
  AddInstallScriptCandidate(&candidates, ManifestSourceDir());

  const QDir appDir(QCoreApplication::applicationDirPath());
  AddInstallScriptCandidate(&candidates, appDir.filePath(QStringLiteral("..")));
  AddInstallScriptCandidate(&candidates, QDir::currentPath());

  QStringList checked;
  for (const QString &candidate : candidates) {
    const QFileInfo info(candidate);
    checked.push_back(info.absoluteFilePath());
    if (info.isFile())
      return info.absoluteFilePath();
  }

  if (error) {
    *error = QStringLiteral("Could not find scripts/install.sh. Checked:\n%1")
                 .arg(checked.join(QStringLiteral("\n")));
  }
  return {};
}

QString EnginesModelsPage::ResolveInstallerBackend(QString *error) const {
  QStringList candidates;
  const QString envPath =
      QString::fromLocal8Bit(qgetenv("STUDIOCAST_INSTALLER_BACKEND")).trimmed();
  if (!envPath.isEmpty())
    candidates.push_back(envPath);

  AddInstallerBackendCandidate(&candidates,
                               QString::fromUtf8(STUDIOCAST_SOURCE_DIR));

  const QDir appDir(QCoreApplication::applicationDirPath());
  const QString installed =
      appDir.filePath(QStringLiteral("../share/studiocast/installer/"
                                     "studiocast-installer-backend"));
  if (!candidates.contains(installed))
    candidates.push_back(installed);

  AddInstallerBackendCandidate(&candidates, ManifestSourceDir());
  AddInstallerBackendCandidate(&candidates, QDir::currentPath());

  QStringList checked;
  for (const QString &candidate : candidates) {
    const QFileInfo info(candidate);
    checked.push_back(info.absoluteFilePath());
    if (info.isFile() && info.isExecutable())
      return info.absoluteFilePath();
  }

  if (error) {
    *error = QStringLiteral(
                 "Could not find studiocast-installer-backend. Checked:\n%1")
                 .arg(checked.join(QStringLiteral("\n")));
  }
  return {};
}

void EnginesModelsPage::StartModelInstall(EngineCard *card) {
  if (!card || modelInstallProcess_ || setupRepairProcess_)
    return;

  QString error;
  const QString script = ResolveInstallScript(&error);
  if (script.isEmpty()) {
    QMessageBox::warning(this, QStringLiteral("StudioCast Models"), error);
    return;
  }

  const QString bash = QStandardPaths::findExecutable(QStringLiteral("bash"));
  if (bash.isEmpty()) {
    QMessageBox::warning(
        this, QStringLiteral("StudioCast Models"),
        QStringLiteral("Could not find bash to run %1.").arg(script));
    return;
  }

  if (card->installArgs.isEmpty()) {
    QMessageBox::information(
        this, QStringLiteral("StudioCast Models"),
        QStringLiteral("No model installer command is available for this "
                       "engine."));
    return;
  }

  modelInstallOutput_.clear();
  activeInstallCard_ = card;
  modelInstallProcess_ = new QProcess(this);
  modelInstallProcess_->setProcessChannelMode(QProcess::SeparateChannels);
  modelInstallProcess_->setWorkingDirectory(QFileInfo(script).absolutePath());

  connect(modelInstallProcess_, &QProcess::readyReadStandardOutput, this,
          [this] {
            if (modelInstallProcess_) {
              modelInstallOutput_ += QString::fromLocal8Bit(
                  modelInstallProcess_->readAllStandardOutput());
            }
          });
  connect(modelInstallProcess_, &QProcess::readyReadStandardError, this,
          [this] {
            if (modelInstallProcess_) {
              modelInstallOutput_ += QString::fromLocal8Bit(
                  modelInstallProcess_->readAllStandardError());
            }
          });
  connect(
      modelInstallProcess_, &QProcess::errorOccurred, this,
      [this](QProcess::ProcessError processError) {
        if (processError != QProcess::FailedToStart || !modelInstallProcess_) {
          return;
        }

        const QString message = modelInstallProcess_->errorString();
        if (activeInstallCard_ && activeInstallCard_->downloadStatus) {
          activeInstallCard_->downloadStatus->setText(
              QStringLiteral("Model installer failed to start."));
        }
        QProcess *process = modelInstallProcess_;
        modelInstallProcess_ = nullptr;
        activeInstallCard_ = nullptr;
        if (process)
          process->deleteLater();
        RefreshDownloadButtons();
        QMessageBox::warning(
            this, QStringLiteral("StudioCast Models"),
            QStringLiteral("Failed to start model installer: %1").arg(message));
      });
  connect(modelInstallProcess_,
          QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
          [this](int exitCode, QProcess::ExitStatus exitStatus) {
            QProcess *process = modelInstallProcess_;
            if (!process)
              return;
            if (process) {
              modelInstallOutput_ +=
                  QString::fromLocal8Bit(process->readAllStandardOutput());
              modelInstallOutput_ +=
                  QString::fromLocal8Bit(process->readAllStandardError());
            }

            const bool ok = exitStatus == QProcess::NormalExit && exitCode == 0;
            if (activeInstallCard_ && activeInstallCard_->downloadStatus) {
              activeInstallCard_->downloadStatus->setText(
                  ok ? QStringLiteral("Model download completed. Status will "
                                      "refresh shortly.")
                     : QStringLiteral("Model download failed. See details."));
            }

            const QString title = QStringLiteral("StudioCast Models");
            if (ok) {
              QMessageBox::information(
                  this, title,
                  QStringLiteral("Model download completed. StudioCast will "
                                 "refresh engine diagnostics shortly."));
            } else {
              QString details = modelInstallOutput_.trimmed();
              if (details.size() > 4000)
                details = details.right(4000);
              QMessageBox::warning(
                  this, title,
                  QStringLiteral("Model download failed with exit code %1.%2")
                      .arg(exitCode)
                      .arg(details.isEmpty()
                               ? QString()
                               : QStringLiteral("\n\n%1").arg(details)));
            }

            if (process)
              process->deleteLater();
            modelInstallProcess_ = nullptr;
            activeInstallCard_ = nullptr;
            emit ModelsInstallFinished();
            RefreshDownloadButtons();
          });

  QStringList processArgs;
  processArgs << script;
  processArgs << card->installArgs;
  card->downloadStatus->setText(QStringLiteral("Starting model download..."));
  RefreshDownloadButtons();
  modelInstallProcess_->start(bash, processArgs);
}

void EnginesModelsPage::StartSetupRepair(EngineCard *card) {
  if (!card || setupRepairProcess_ || modelInstallProcess_)
    return;

  if (card->repairArgs.isEmpty()) {
    QMessageBox::information(
        this, QStringLiteral("StudioCast Setup Repair"),
        QStringLiteral(
            "No setup repair command is available for this engine."));
    return;
  }

  QString error;
  setupRepairBackend_ = ResolveInstallerBackend(&error);
  if (setupRepairBackend_.isEmpty()) {
    QMessageBox::warning(this, QStringLiteral("StudioCast Setup Repair"),
                         error);
    return;
  }

  activeRepairCard_ = card;
  setupRepairPlanText_.clear();
  setupRepairPromptBuffer_.clear();
  setupRepairPasswordDialogOpen_ = false;
  setupRepairPasswordCancelled_ = false;
  QStringList planArgs;
  planArgs << QStringLiteral("plan") << card->repairArgs;
  StartSetupRepairProcess(SetupRepairPhase::Plan, planArgs,
                          QStringLiteral("Planning setup repair..."));
}

void EnginesModelsPage::StartSetupRepairProcess(SetupRepairPhase phase,
                                                const QStringList &arguments,
                                                const QString &statusText) {
  if (setupRepairProcess_ || setupRepairBackend_.isEmpty())
    return;

  setupRepairOutput_.clear();
  setupRepairPromptBuffer_.clear();
  setupRepairProcess_ = new QProcess(this);
  setupRepairProcess_->setProcessChannelMode(QProcess::SeparateChannels);
  setupRepairProcess_->setWorkingDirectory(
      QFileInfo(setupRepairBackend_).absolutePath());
  QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
  environment.insert(QStringLiteral("STUDIOCAST_GUI_SUDO_STDIN"),
                     QStringLiteral("1"));
  environment.insert(QStringLiteral("STUDIOCAST_GUI_SUDO_PROMPT"),
                     QStringLiteral("[sudo] password for %u: "));
  setupRepairProcess_->setProcessEnvironment(environment);

  if (activeRepairCard_ && activeRepairCard_->repairSetupStatus) {
    activeRepairCard_->repairSetupStatus->setText(statusText);
    activeRepairCard_->repairSetupStatus->setVisible(true);
  }

  connect(
      setupRepairProcess_, &QProcess::readyReadStandardOutput, this, [this] {
        if (setupRepairProcess_) {
          AppendSetupRepairOutput(setupRepairProcess_->readAllStandardOutput());
        }
      });
  connect(setupRepairProcess_, &QProcess::readyReadStandardError, this, [this] {
    if (setupRepairProcess_) {
      AppendSetupRepairErrorOutput(setupRepairProcess_->readAllStandardError());
    }
  });
  connect(
      setupRepairProcess_, &QProcess::errorOccurred, this,
      [this](QProcess::ProcessError processError) {
        if (processError != QProcess::FailedToStart || !setupRepairProcess_) {
          return;
        }

        const QString message = setupRepairProcess_->errorString();
        if (activeRepairCard_ && activeRepairCard_->repairSetupStatus) {
          activeRepairCard_->repairSetupStatus->setText(
              QStringLiteral("Setup repair failed to start."));
          activeRepairCard_->repairSetupStatus->setVisible(true);
        }
        QProcess *process = setupRepairProcess_;
        setupRepairProcess_ = nullptr;
        activeRepairCard_ = nullptr;
        if (process)
          process->deleteLater();
        RefreshDownloadButtons();
        QMessageBox::warning(
            this, QStringLiteral("StudioCast Setup Repair"),
            QStringLiteral("Failed to start setup repair: %1").arg(message));
      });
  connect(setupRepairProcess_,
          QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
          [this, phase](int exitCode, QProcess::ExitStatus exitStatus) {
            if (phase == SetupRepairPhase::Plan) {
              FinishSetupRepairPlan(exitCode, exitStatus);
            } else {
              FinishSetupRepairExecution(exitCode, exitStatus);
            }
          });

  RefreshDownloadButtons();
  setupRepairProcess_->start(setupRepairBackend_, arguments);
}

void EnginesModelsPage::AppendSetupRepairOutput(const QByteArray &bytes) {
  setupRepairOutput_ += QString::fromLocal8Bit(bytes);
}

void EnginesModelsPage::AppendSetupRepairErrorOutput(const QByteArray &bytes) {
  const QString text = QString::fromLocal8Bit(bytes);
  setupRepairOutput_ += text;
  setupRepairPromptBuffer_ += text;
  if (setupRepairPromptBuffer_.size() > 1000)
    setupRepairPromptBuffer_ = setupRepairPromptBuffer_.right(1000);

  if (!setupRepairPasswordDialogOpen_ &&
      LooksLikeSudoPasswordPrompt(setupRepairPromptBuffer_)) {
    setupRepairPromptBuffer_.clear();
    PromptForSetupRepairPassword();
  }
}

void EnginesModelsPage::PromptForSetupRepairPassword() {
  if (!setupRepairProcess_ || setupRepairPasswordDialogOpen_)
    return;

  setupRepairPasswordDialogOpen_ = true;
  if (activeRepairCard_ && activeRepairCard_->repairSetupStatus) {
    activeRepairCard_->repairSetupStatus->setText(
        QStringLiteral("Waiting for sudo password..."));
    activeRepairCard_->repairSetupStatus->setVisible(true);
  }

  QDialog dialog(this);
  dialog.setObjectName(QStringLiteral("setupRepairPasswordDialog"));
  dialog.setWindowTitle(QStringLiteral("StudioCast Setup Repair"));
  dialog.setModal(true);
  dialog.resize(460, 220);

  auto *root = new QVBoxLayout(&dialog);
  root->setContentsMargins(18, 18, 18, 18);
  root->setSpacing(12);
  root->addWidget(
      DialogTitleLabel(QStringLiteral("Sudo password required"), &dialog));
  root->addWidget(MutedLabel(
      QStringLiteral("StudioCast needs your account password to continue the "
                     "repair steps that install packages or update system "
                     "configuration."),
      &dialog));

  auto *passwordEdit = new QLineEdit(&dialog);
  passwordEdit->setObjectName(QStringLiteral("setupRepairPasswordEdit"));
  passwordEdit->setEchoMode(QLineEdit::Password);
  passwordEdit->setPlaceholderText(QStringLiteral("Password"));
  root->addWidget(passwordEdit);

  auto *buttons = new QDialogButtonBox(QDialogButtonBox::Cancel, &dialog);
  auto *continueButton = buttons->addButton(QStringLiteral("Continue"),
                                            QDialogButtonBox::AcceptRole);
  SetPrimaryDialogButton(continueButton);
  continueButton->setDefault(true);
  QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog,
                   &QDialog::accept);
  QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog,
                   &QDialog::reject);
  root->addWidget(buttons);

  passwordEdit->setFocus(Qt::OtherFocusReason);
  const int result = dialog.exec();
  setupRepairPasswordDialogOpen_ = false;

  if (!setupRepairProcess_)
    return;

  if (result == QDialog::Accepted) {
    setupRepairProcess_->write(passwordEdit->text().toLocal8Bit());
    setupRepairProcess_->write("\n");
    if (activeRepairCard_ && activeRepairCard_->repairSetupStatus) {
      activeRepairCard_->repairSetupStatus->setText(
          QStringLiteral("Repairing setup..."));
      activeRepairCard_->repairSetupStatus->setVisible(true);
    }
    return;
  }

  setupRepairPasswordCancelled_ = true;
  setupRepairOutput_ += QStringLiteral(
      "\n[StudioCast] Setup repair cancelled while waiting for sudo "
      "password.\n");
  setupRepairProcess_->terminate();
}

void EnginesModelsPage::FinishSetupRepairPlan(int exitCode,
                                              QProcess::ExitStatus exitStatus) {
  QProcess *process = setupRepairProcess_;
  if (!process)
    return;

  AppendSetupRepairOutput(process->readAllStandardOutput());
  setupRepairOutput_ += QString::fromLocal8Bit(process->readAllStandardError());
  const bool ok = exitStatus == QProcess::NormalExit && exitCode == 0;
  process->deleteLater();
  setupRepairProcess_ = nullptr;

  if (!ok) {
    if (activeRepairCard_ && activeRepairCard_->repairSetupStatus) {
      activeRepairCard_->repairSetupStatus->setText(
          QStringLiteral("Setup repair plan failed. See details."));
      activeRepairCard_->repairSetupStatus->setVisible(true);
    }
    QString details = TailForDialog(setupRepairOutput_);
    if (details.isEmpty())
      details = QStringLiteral("No output was captured.");
    ShowSetupRepairDialog(
        this, QStringLiteral("Setup repair plan failed"),
        QStringLiteral("The installer backend exited with code %1.")
            .arg(exitCode),
        details, QStringLiteral("error"));
    activeRepairCard_ = nullptr;
    RefreshDownloadButtons();
    return;
  }

  setupRepairPlanText_ = setupRepairOutput_.trimmed();
  if (activeRepairCard_ && activeRepairCard_->repairSetupStatus) {
    activeRepairCard_->repairSetupStatus->setText(
        QStringLiteral("Setup repair plan is ready."));
    activeRepairCard_->repairSetupStatus->setVisible(true);
  }
  RefreshDownloadButtons();

  if (!ConfirmSetupRepairDialog(this, setupRepairPlanText_)) {
    if (activeRepairCard_ && activeRepairCard_->repairSetupStatus) {
      activeRepairCard_->repairSetupStatus->setText(
          QStringLiteral("Setup repair cancelled."));
      activeRepairCard_->repairSetupStatus->setVisible(true);
    }
    activeRepairCard_ = nullptr;
    RefreshDownloadButtons();
    return;
  }

  if (!activeRepairCard_) {
    RefreshDownloadButtons();
    return;
  }
  StartSetupRepairProcess(SetupRepairPhase::Execute,
                          activeRepairCard_->repairArgs,
                          QStringLiteral("Repairing setup..."));
}

void EnginesModelsPage::FinishSetupRepairExecution(
    int exitCode, QProcess::ExitStatus exitStatus) {
  QProcess *process = setupRepairProcess_;
  if (!process)
    return;

  AppendSetupRepairOutput(process->readAllStandardOutput());
  setupRepairOutput_ += QString::fromLocal8Bit(process->readAllStandardError());
  const bool ok = exitStatus == QProcess::NormalExit && exitCode == 0;
  const bool cancelled = setupRepairPasswordCancelled_;

  if (activeRepairCard_ && activeRepairCard_->repairSetupStatus) {
    activeRepairCard_->repairSetupStatus->setText(
        cancelled ? QStringLiteral("Setup repair cancelled.")
        : ok      ? QStringLiteral(
                        "Setup repair completed. Status will refresh shortly.")
             : QStringLiteral("Setup repair failed. See details."));
    activeRepairCard_->repairSetupStatus->setVisible(true);
  }

  if (cancelled) {
    ShowSetupRepairDialog(
        this, QStringLiteral("Setup repair cancelled"),
        QStringLiteral("The repair was cancelled before sudo authentication "
                       "completed."),
        QString(), QStringLiteral("warning"));
  } else if (ok) {
    ShowSetupRepairDialog(
        this, QStringLiteral("Setup repair completed"),
        QStringLiteral("Setup repair completed. StudioCast will refresh engine "
                       "diagnostics shortly."),
        QString(), QStringLiteral("info"));
  } else {
    QString details = TailForDialog(setupRepairOutput_);
    if (details.isEmpty())
      details = QStringLiteral("No output was captured.");
    ShowSetupRepairDialog(
        this, QStringLiteral("Setup repair failed"),
        QStringLiteral("The installer backend exited with code %1.")
            .arg(exitCode),
        details, QStringLiteral("error"));
  }

  process->deleteLater();
  setupRepairProcess_ = nullptr;
  activeRepairCard_ = nullptr;
  setupRepairPromptBuffer_.clear();
  setupRepairPasswordDialogOpen_ = false;
  setupRepairPasswordCancelled_ = false;
  emit SetupRepairFinished();
  RefreshDownloadButtons();
}

void EnginesModelsPage::UpdateStatus(const DaemonStatusSnapshot &snapshot) {
  if (videoPreferenceLabel_)
    videoPreferenceLabel_->setText(
        FriendlyBackendLabel(snapshot.videoEffectsEnginePreference));
  if (videoActiveLabel_) {
    QString active;
    if (!snapshot.reachable || !snapshot.parsed) {
      active = QStringLiteral("Unknown");
    } else if (snapshot.videoPipelineStarting) {
      active = QStringLiteral("Starting");
    } else if (!snapshot.videoPipelineRunning ||
               snapshot.videoPipelineState != QStringLiteral("running")) {
      active = QStringLiteral("Inactive");
    } else {
      active = ActiveBackendSummary(snapshot.videoEffectsActiveBackends,
                                    QStringLiteral("Pass-through"));
    }
    videoActiveLabel_->setText(active);
    videoActiveLabel_->setToolTip(
        snapshot.videoPipelineRunning && !snapshot.videoPipelineStarting &&
                snapshot.videoPipelineState == QStringLiteral("running")
            ? snapshot.videoEffectsActiveBackends
            : QString());
  }

  if (audioPreferenceLabel_)
    audioPreferenceLabel_->setText(
        FriendlyBackendLabel(snapshot.audioEffectsEnginePreference));
  if (microphoneActiveLabel_) {
    const QString microphoneActive =
        !snapshot.reachable || !snapshot.parsed
            ? QStringLiteral("Unknown")
            : FriendlyBackendLabel(snapshot.microphoneActiveBackend);
    microphoneActiveLabel_->setText(
        QStringLiteral("Microphone: %1").arg(microphoneActive));
    microphoneActiveLabel_->setToolTip(snapshot.microphoneActiveBackend);
  }
  if (speakersActiveLabel_) {
    QString speakerActive;
    const QString routeMode = snapshot.speakersRouteMode.trimmed().toLower();
    if (!snapshot.reachable || !snapshot.parsed) {
      speakerActive = QStringLiteral("Unknown");
    } else if (routeMode == QStringLiteral("loopback")) {
      speakerActive = QStringLiteral("Loopback / pass-through");
    } else if (routeMode == QStringLiteral("off") &&
               snapshot.speakersActiveBackend.trimmed().isEmpty()) {
      speakerActive = QStringLiteral("Off");
    } else {
      speakerActive = FriendlyBackendLabel(snapshot.speakersActiveBackend);
    }
    speakersActiveLabel_->setText(
        QStringLiteral("Speakers: %1").arg(speakerActive));
    speakersActiveLabel_->setToolTip(snapshot.speakersActiveBackend);
  }

  UpdateEngineCard(&maxineCard_, snapshot.maxine,
                   PreferenceSelectsMaxine(snapshot));
  UpdateEngineCard(&openVideoCard_, snapshot.openCuda,
                   PreferenceSelectsOpenVideo(snapshot));
  UpdateEngineCard(&openAudioCard_, snapshot.openAudio,
                   PreferenceSelectsOpenAudio(snapshot));
}

} // namespace studiocast::gui
