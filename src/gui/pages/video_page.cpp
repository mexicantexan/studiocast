#include "video_page.h"

#include <QCheckBox>
#include <QClipboard>
#include <QComboBox>
#include <QCoreApplication>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPaintEvent>
#include <QPainter>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QSizePolicy>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QStyle>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QVector>

#include <sstream>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "core/maxine/reason_codes.h"
#include "core/video/broadcast_camera_effects_json.h"
#include "core/video/convert.h"
#include "core/video/effects/broadcast_effect_contract.h"
#include "core/video/effects/effect_descriptors.h"
#include "core/video/v4l2loopback.h"
#include "gui/status/daemon_status_snapshot.h"
#include "gui/text_edit_utils.h"

namespace studiocast::gui {

namespace {

// Debug helper for diagnosing preview start/stop behavior.
// Enable with: STUDIOCAST_DEBUG_GUI_PREVIEW=1
bool DebugGuiPreview() {
  static const bool enabled =
      (std::getenv("STUDIOCAST_DEBUG_GUI_PREVIEW") != nullptr);
  return enabled;
}

void GuiPreviewDbg(const std::string &msg) {
  if (!DebugGuiPreview())
    return;
  std::cerr << "[gui_preview_dbg] " << msg << "\n";
}

bool LooksLikeCameraInputError(const QString &error) {
  const QString trimmed = error.trimmed();
  return trimmed.contains(QStringLiteral("Failed to open capture device")) ||
         trimmed.contains(QStringLiteral("No readable camera device found")) ||
         trimmed.contains(QStringLiteral("Failed to auto-select a usable camera"));
}

} // namespace

class VideoPreviewWidget final : public QWidget {
public:
  explicit VideoPreviewWidget(QWidget *parent = nullptr) : QWidget(parent) {
    setMinimumSize(320, 240);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  }

  void SetStatusText(QString text) {
    frame_ = QImage{};
    statusText_ = std::move(text);
    update();
  }

  void SetFrame(const QImage &frame) {
    // Detach from the caller's backing buffer.
    frame_ = frame.copy();
    statusText_.clear();
    update();
  }

protected:
  void paintEvent(QPaintEvent * /*event*/) override {
    QPainter p(this);
    // Paint using palette colors to match the active global theme.
    p.fillRect(rect(), palette().color(QPalette::Base));

    p.setPen(palette().color(QPalette::Mid));
    p.drawRect(rect().adjusted(0, 0, -1, -1));

    const QRect content = rect().adjusted(8, 8, -8, -8);

    if (!frame_.isNull()) {
      QSize scaled = frame_.size();
      scaled.scale(content.size(), Qt::KeepAspectRatio);
      QRect target(QPoint(0, 0), scaled);
      target.moveCenter(content.center());
      p.drawImage(target, frame_);
      return;
    }

    p.setPen(palette().color(QPalette::Text));
    const QString text =
        statusText_.isEmpty() ? QStringLiteral("Preview") : statusText_;
    p.drawText(content, Qt::AlignCenter | Qt::TextWordWrap, text);
  }

  QSize sizeHint() const override { return {640, 360}; }

private:
  QImage frame_;
  QString statusText_;
};

namespace {

QString DeviceLabel(const studiocast::video::VideoDevice &d) {
  QString label = QString::fromStdString(d.dev_node);
  if (!d.name.empty())
    label += " — " + QString::fromStdString(d.name);
  if (!d.driver.empty())
    label += " (" + QString::fromStdString(d.driver) + ")";
  if (d.is_loopback)
    label += " [loopback]";
  return label;
}

enum class EffectAvailabilityKind {
  always,
  gpu_utility,
  maxine_listed,
};

const std::unordered_map<std::string,
                         studiocast::video::effects::VideoEffectDescriptor> &
EffectDescriptorById() {
  static const auto *map = []() {
    auto *m = new std::unordered_map<
        std::string, studiocast::video::effects::VideoEffectDescriptor>();
    for (const auto &d : studiocast::video::effects::VideoEffectDescriptors()) {
      (*m)[d.id] = d;
    }
    return m;
  }();
  return *map;
}

EffectAvailabilityKind AvailabilityKindForEffectId(const QString &id) {
  const auto &m = EffectDescriptorById();
  const auto it = m.find(id.toStdString());
  if (it == m.end()) {
    // Conservative default for unknown IDs.
    return EffectAvailabilityKind::maxine_listed;
  }
  bool needsMaxine = false;
  bool needsGpuUtility = false;
  for (const auto c : it->second.required_components) {
    needsMaxine =
        needsMaxine ||
        (c == studiocast::video::effects::RequiredComponent::maxine_vfx ||
         c == studiocast::video::effects::RequiredComponent::maxine_ar);
    needsGpuUtility =
        needsGpuUtility ||
        (c == studiocast::video::effects::RequiredComponent::gpu_utility);
  }
  if (needsMaxine)
    return EffectAvailabilityKind::maxine_listed;
  if (needsGpuUtility)
    return EffectAvailabilityKind::gpu_utility;
  return EffectAvailabilityKind::always;
}

struct DaemonVideoStatus {
  struct NegotiatedFormat {
    bool present = false;
    QString pixfmt;
    int width = 0;
    int height = 0;
    double fps = 0.0;
    int fps_num = 0;
    int fps_den = 0;
    int bytes_per_line = 0;
    int size_image = 0;
  };

  bool reachable = false;

  bool enabled = false;
  bool always_on = false;

  bool virtual_device_present = false;
  bool virtual_device_available = false;
  QString virtual_device_error;

  bool consumer_present = false;
  int consumer_count = 0;
  QString consumer_error;

  bool pipeline_running = false;
  bool pipeline_starting = false;
  bool pipeline_active_needed = false;
  QString pipeline_state;
  QString pipeline_idle_reason;
  long long frame_index = 0;

  int consumer_poll_ms = 0;
  int start_grace_ms = 0;
  int stop_grace_ms = 0;
  int min_run_ms = 0;
  long long pipeline_start_attempts = 0;
  long long pipeline_starts = 0;
  long long pipeline_start_failures = 0;
  long long pipeline_stops = 0;
  long long pipeline_config_restarts = 0;
  bool stabilizing = false;
  int thrash_events_10s = 0;
  QString last_transition;
  long long last_transition_ms_ago = -1;
  long long next_start_retry_ms = -1;

  int width = 0;
  int height = 0;
  int fps = 0;
  QString output_format_requested = QStringLiteral("rgb24");

  NegotiatedFormat capture_format;
  NegotiatedFormat output_format;
  QString capture_fallback_state;
  QString capture_fallback_reason;

  // Output scaling info (from daemon status).
  QString scaling_backend_active;
  NegotiatedFormat scaling_from;
  NegotiatedFormat scaling_to;

  // Video compute backend info (from daemon status).
  QString compute_preference = QStringLiteral("auto");
  QString compute_resolved_backend = QStringLiteral("cpu");
  QString compute_active_backend = QStringLiteral("cpu");
  QString compute_fallback_reason;
  QString compute_degraded_reason;

  studiocast::video::effects::BroadcastCameraEffects effects{};
  bool effects_valid = false;

  // Rule-based disable reasons and deterministic ordering (from daemon status).
  QStringList effects_plan_ordered;
  QString effects_plan_vignette_attach_to;
  QMap<QString, QString> effects_plan_disabled;

  // Maxine runtime diagnostics (from daemon status)
  bool maxine_ok = false;
  bool maxine_supported = false;
  QString maxine_summary;
  QString maxine_blocked_reason;
  QStringList maxine_blocked_details;
  QStringList maxine_available_effects;
  QMap<QString, QStringList> maxine_missing_effects;
  bool virtual_key_light_available = false;

  // Open CUDA runtime diagnostics (from daemon status)
  bool open_cuda_present = false;
  bool open_cuda_ok = false;
  QString open_cuda_summary;
  QString open_cuda_blocked_reason;
  QStringList open_cuda_blocked_details;
  QString open_cuda_default_model_id;

  struct OpenCudaModelInfo {
    QString id;
    QString display_name;
    QString task;
    int width = 0;
    int height = 0;
  };
  std::vector<OpenCudaModelInfo> open_cuda_models;

  QStringList open_cuda_installed_models;
  QMap<QString, QString> open_cuda_missing_models;
  QStringList open_cuda_available_effects;
  QMap<QString, QString> open_cuda_blocked_effects;
  QStringList open_cuda_install_hints;

  QString effects_backends;
  QString effects_note;

  QString input_device;
  QString output_device;
  QString last_error;
};

bool ParseDaemonStatusJson(const std::string &json, DaemonVideoStatus *out,
                           QString *error) {
  if (!out)
    return false;

  QJsonParseError perr;
  const auto doc =
      QJsonDocument::fromJson(QByteArray::fromStdString(json), &perr);
  if (perr.error != QJsonParseError::NoError || !doc.isObject()) {
    if (error)
      *error = "JSON parse error: " + perr.errorString();
    return false;
  }

  const QJsonObject root = doc.object();
  const QJsonObject video = root.value("video").toObject();
  if (video.isEmpty()) {
    if (error)
      *error = "Missing 'video' object";
    return false;
  }

  out->reachable = true;
  out->enabled = video.value("enabled").toBool(false);
  out->always_on = video.value("always_on").toBool(false);
  out->virtual_device_present =
      video.value("virtual_device_present").toBool(false);
  out->virtual_device_available =
      video.value("virtual_device_available").toBool(false);
  out->virtual_device_error =
      video.value("virtual_device_error").toString();
  out->consumer_present = video.value("consumer_present").toBool(false);
  out->consumer_count = video.value("consumer_count").toInt(0);
  out->consumer_error = video.value("consumer_error").toString();

  out->width = video.value("width").toInt(0);
  out->height = video.value("height").toInt(0);
  out->fps = video.value("fps").toInt(0);
  out->output_format_requested =
      video.value("output_format_requested").toString(QStringLiteral("rgb24"));

  const auto parseFormat =
      [](const QJsonObject &fmt) -> DaemonVideoStatus::NegotiatedFormat {
    DaemonVideoStatus::NegotiatedFormat f;
    if (fmt.isEmpty())
      return f;

    f.present = true;
    f.pixfmt = fmt.value("pixfmt").toString();
    f.width = fmt.value("width").toInt(0);
    f.height = fmt.value("height").toInt(0);
    f.fps = fmt.value("fps").toDouble(0);
    f.fps_num = fmt.value("fps_num").toInt(0);
    f.fps_den = fmt.value("fps_den").toInt(0);
    f.bytes_per_line = fmt.value("bytesperline").toInt(0);
    f.size_image = fmt.value("sizeimage").toInt(0);
    return f;
  };

  out->capture_format = parseFormat(video.value("capture_format").toObject());
  out->output_format = parseFormat(video.value("output_format").toObject());
  const QJsonObject captureFallback =
      video.value("capture_fallback").toObject();
  out->capture_fallback_state =
      captureFallback.value("state").toString(QStringLiteral("none"));
  out->capture_fallback_reason = captureFallback.value("reason").toString();

  const QJsonObject scaling = video.value("scaling").toObject();
  if (!scaling.isEmpty()) {
    out->scaling_backend_active = scaling.value("backend_active").toString();
    out->scaling_from = parseFormat(scaling.value("from").toObject());
    out->scaling_to = parseFormat(scaling.value("to").toObject());
  } else {
    out->scaling_backend_active.clear();
    out->scaling_from = {};
    out->scaling_to = {};
  }

  const QJsonObject compute = video.value("compute").toObject();
  if (!compute.isEmpty()) {
    out->compute_preference =
        compute.value("preference").toString(QStringLiteral("auto"));
    out->compute_resolved_backend =
        compute.value("resolved_backend").toString(QStringLiteral("cpu"));
    out->compute_active_backend =
        compute.value("active_backend").toString(QStringLiteral("cpu"));
    out->compute_fallback_reason =
        compute.value("fallback_reason").toString();
    out->compute_degraded_reason =
        compute.value("degraded_reason").toString();
  } else {
    out->compute_preference = QStringLiteral("auto");
    out->compute_resolved_backend = QStringLiteral("cpu");
    out->compute_active_backend = QStringLiteral("cpu");
    out->compute_fallback_reason.clear();
    out->compute_degraded_reason.clear();
  }

  out->input_device = video.value("input_device").toString();
  out->output_device = video.value("output_device").toString();

  const QJsonObject pipe = video.value("pipeline").toObject();
  out->pipeline_running = pipe.value("running").toBool(false);
  out->pipeline_starting = pipe.value("starting").toBool(false);
  out->pipeline_active_needed = pipe.value("active_needed").toBool(false);
  out->pipeline_state = pipe.value("state").toString();
  out->pipeline_idle_reason = pipe.value("idle_reason").toString();
  out->frame_index =
      static_cast<long long>(pipe.value("frame_index").toDouble(0));

  out->effects_backends = pipe.value("effects_backends").toString();
  out->effects_note = pipe.value("effects_note").toString();

  const QJsonObject sup = video.value("supervisor").toObject();
  if (!sup.isEmpty()) {
    out->consumer_poll_ms = sup.value("consumer_poll_ms").toInt(0);
    out->start_grace_ms = sup.value("start_grace_ms").toInt(0);
    out->stop_grace_ms = sup.value("stop_grace_ms").toInt(0);
    out->min_run_ms = sup.value("min_run_ms").toInt(0);
    out->pipeline_start_attempts =
        static_cast<long long>(sup.value("pipeline_start_attempts").toDouble(0));
    out->pipeline_starts =
        static_cast<long long>(sup.value("pipeline_starts").toDouble(0));
    out->pipeline_start_failures = static_cast<long long>(
        sup.value("pipeline_start_failures").toDouble(0));
    out->pipeline_stops =
        static_cast<long long>(sup.value("pipeline_stops").toDouble(0));
    out->pipeline_config_restarts = static_cast<long long>(
        sup.value("pipeline_config_restarts").toDouble(0));
    out->stabilizing = sup.value("stabilizing").toBool(false);
    out->thrash_events_10s = sup.value("thrash_events_10s").toInt(0);
    out->last_transition = sup.value("last_transition").toString();
    out->last_transition_ms_ago =
        static_cast<long long>(sup.value("last_transition_ms_ago").toDouble(-1));
    out->next_start_retry_ms =
        static_cast<long long>(sup.value("next_start_retry_ms").toDouble(-1));
  }

  // Canonical effects model (Broadcast schema).
  out->effects = {};
  out->effects_valid = false;
  const QJsonObject fx = video.value("video_effects").toObject();
  if (!fx.isEmpty()) {
    const QByteArray txt = QJsonDocument(fx).toJson(QJsonDocument::Compact);
    std::string jerr;
    if (studiocast::video::ApplyBroadcastCameraEffectsPatchJsonText(
            txt.toStdString(), &out->effects, &jerr)) {
      out->effects_valid = true;
    }
  }

  // Effect plan (ordering + disable reasons).
  out->effects_plan_ordered.clear();
  out->effects_plan_vignette_attach_to.clear();
  out->effects_plan_disabled.clear();
  const QJsonObject plan = pipe.value("effects_plan").toObject();
  if (!plan.isEmpty()) {
    const auto ordered = plan.value("ordered").toArray();
    for (const auto &v : ordered) {
      const QString s = v.toString();
      if (!s.isEmpty())
        out->effects_plan_ordered.push_back(s);
    }

    out->effects_plan_vignette_attach_to =
        plan.value("vignette_attach_to").toString();

    const auto disabled = plan.value("disabled").toArray();
    for (const auto &v : disabled) {
      const QJsonObject o = v.toObject();
      const QString id = o.value("id").toString();
      const QString reason = o.value("reason").toString();
      if (!id.isEmpty() && !reason.isEmpty()) {
        out->effects_plan_disabled.insert(id, reason);
      }
    }
  }

  const QJsonObject maxine = root.value("maxine").toObject();
  if (!maxine.isEmpty()) {
    out->maxine_ok = maxine.value("ok").toBool(false);
    out->maxine_supported = maxine.value("supported").toBool(out->maxine_ok);
    out->maxine_summary = maxine.value("summary").toString();
    out->maxine_blocked_reason = maxine.value("blocked_reason").toString();

    out->maxine_blocked_details.clear();
    const auto blocked = maxine.value("blocked_details").toArray();
    for (const auto &v : blocked) {
      const QString s = v.toString();
      if (!s.isEmpty())
        out->maxine_blocked_details.push_back(s);
    }

    out->maxine_available_effects.clear();
    out->maxine_missing_effects.clear();
    out->virtual_key_light_available = false;
    const auto arr = maxine.value("available_effects").toArray();
    for (const auto &v : arr) {
      const QString id = v.toString();
      if (!id.isEmpty())
        out->maxine_available_effects.push_back(id);
      if (id == "virtual_key_light") {
        out->virtual_key_light_available = true;
      }
    }

    const QJsonObject missing = maxine.value("missing_effects").toObject();
    for (auto it = missing.begin(); it != missing.end(); ++it) {
      const QString id = it.key();
      if (id.isEmpty())
        continue;
      QStringList reasons;
      const auto reasonArr = it.value().toArray();
      for (const auto &rv : reasonArr) {
        const QString s = rv.toString();
        if (!s.isEmpty())
          reasons.push_back(s);
      }
      out->maxine_missing_effects.insert(id, reasons);
    }
  }

  // Open CUDA diagnostics payload.
  out->open_cuda_present = false;
  out->open_cuda_ok = false;
  out->open_cuda_summary.clear();
  out->open_cuda_blocked_reason.clear();
  out->open_cuda_blocked_details.clear();
  out->open_cuda_default_model_id.clear();
  out->open_cuda_models.clear();
  out->open_cuda_installed_models.clear();
  out->open_cuda_missing_models.clear();
  out->open_cuda_available_effects.clear();
  out->open_cuda_blocked_effects.clear();
  out->open_cuda_install_hints.clear();

  QJsonObject openCuda = root.value("open_cuda").toObject();
  if (openCuda.isEmpty()) {
    const QJsonObject engines = root.value("engines").toObject();
    openCuda = engines.value("open_cuda").toObject();
  }
  if (!openCuda.isEmpty()) {
    out->open_cuda_present = true;
    out->open_cuda_ok = openCuda.value("ok").toBool(false);
    out->open_cuda_summary = openCuda.value("summary").toString();
    out->open_cuda_blocked_reason =
        openCuda.value("blocked_reason").toString();
    const auto blockedDetails =
        openCuda.value("blocked_details").toArray();
    for (const auto &v : blockedDetails) {
      const QString s = v.toString();
      if (!s.isEmpty())
        out->open_cuda_blocked_details.push_back(s);
    }

    out->open_cuda_default_model_id =
        openCuda.value("default_model_id").toString();

    const auto models = openCuda.value("models").toArray();
    for (const auto &v : models) {
      const QJsonObject o = v.toObject();
      const QString id = o.value("id").toString();
      if (id.isEmpty())
        continue;
      DaemonVideoStatus::OpenCudaModelInfo mi;
      mi.id = id;
      mi.display_name = o.value("display_name").toString(id);
      mi.task = o.value("task").toString();
      mi.width = o.value("width").toInt(0);
      mi.height = o.value("height").toInt(0);
      out->open_cuda_models.push_back(mi);
    }

    const auto installed = openCuda.value("installed_models").toArray();
    for (const auto &v : installed) {
      const QString s = v.toString();
      if (!s.isEmpty())
        out->open_cuda_installed_models.push_back(s);
    }

    if (out->open_cuda_installed_models.isEmpty() &&
        !out->open_cuda_models.empty()) {
      for (const auto &m : out->open_cuda_models) {
        if (!m.id.isEmpty())
          out->open_cuda_installed_models.push_back(m.id);
      }
    }

    // Backward compatibility: older daemons only provide installed model IDs.
    if (out->open_cuda_models.empty() &&
        !out->open_cuda_installed_models.isEmpty()) {
      out->open_cuda_models.reserve(
          static_cast<std::size_t>(out->open_cuda_installed_models.size()));
      for (const auto &id : out->open_cuda_installed_models) {
        DaemonVideoStatus::OpenCudaModelInfo mi;
        mi.id = id;
        mi.display_name = id;
        out->open_cuda_models.push_back(mi);
      }
    }

    const QJsonObject missing = openCuda.value("missing_models").toObject();
    for (auto it = missing.begin(); it != missing.end(); ++it) {
      const QString id = it.key();
      const QString reason = it.value().toString();
      if (!id.isEmpty() && !reason.isEmpty()) {
        out->open_cuda_missing_models.insert(id, reason);
      }
    }

    const auto available = openCuda.value("available_effects").toArray();
    for (const auto &v : available) {
      const QString id = v.toString();
      if (!id.isEmpty())
        out->open_cuda_available_effects.push_back(id);
    }

    const QJsonObject blocked = openCuda.value("blocked_effects").toObject();
    for (auto it = blocked.begin(); it != blocked.end(); ++it) {
      const QString id = it.key();
      const QString reason = it.value().toString();
      if (!id.isEmpty() && !reason.isEmpty()) {
        out->open_cuda_blocked_effects.insert(id, reason);
      }
    }

    const auto hints = openCuda.value("install_hints").toArray();
    for (const auto &v : hints) {
      const QString s = v.toString();
      if (!s.isEmpty())
        out->open_cuda_install_hints.push_back(s);
    }
  }

  out->last_error = video.value("last_error").toString();
  return true;
}

bool ParseJsonObject(const std::string &json, QJsonObject *outRoot,
                     QString *error) {
  QJsonParseError perr;
  const auto doc =
      QJsonDocument::fromJson(QByteArray::fromStdString(json), &perr);
  if (perr.error != QJsonParseError::NoError || !doc.isObject()) {
    if (error)
      *error = "JSON parse error: " + perr.errorString();
    return false;
  }

  if (outRoot)
    *outRoot = doc.object();
  return true;
}

QString FormatMaxineReasonCode(const QString &code) {
  if (code.isEmpty())
    return {};
  const std::string s = code.toStdString();
  return QString::fromStdString(studiocast::maxine::reasons::ToEnglish(s));
}

QString ReasonCodeFromBlockedLine(const QString &line) {
  const QString trimmed = line.trimmed();
  const qsizetype colon = trimmed.lastIndexOf(QStringLiteral(": "));
  if (colon >= 0)
    return trimmed.mid(colon + 2).trimmed();
  return trimmed;
}

QStringList OpenCudaBlockerCodes(const DaemonVideoStatus &st) {
  QStringList codes;
  const auto add = [&](const QString &raw) {
    const QString code = ReasonCodeFromBlockedLine(raw);
    if (!code.isEmpty() && !codes.contains(code))
      codes.push_back(code);
  };

  add(st.open_cuda_blocked_reason);
  for (auto it = st.open_cuda_blocked_effects.constBegin();
       it != st.open_cuda_blocked_effects.constEnd(); ++it) {
    add(it.value());
  }
  return codes;
}

bool OpenCudaHasBlockerCode(const DaemonVideoStatus &st, const QString &code) {
  return OpenCudaBlockerCodes(st).contains(code);
}

bool OpenCudaHasSetupBlocker(const DaemonVideoStatus &st) {
  const QStringList codes = OpenCudaBlockerCodes(st);
  for (const QString &code : codes) {
    if (code == QStringLiteral("disabled_in_build") ||
        code == QStringLiteral("onnxruntime_not_found") ||
        code == QStringLiteral("onnxruntime_cuda_provider_unavailable") ||
        code == QStringLiteral("cuda_unavailable")) {
      return true;
    }
  }
  return false;
}

QString OpenCudaSetupReasonText(const QString &code) {
  if (code == QStringLiteral("disabled_in_build")) {
    return QStringLiteral(
        "Open Video / Open CUDA is disabled in the running StudioCast build.");
  }
  if (code == QStringLiteral("onnxruntime_not_found")) {
    return QStringLiteral(
        "This StudioCast build was compiled without ONNX Runtime.");
  }
  if (code == QStringLiteral("onnxruntime_cuda_provider_unavailable")) {
    return QStringLiteral(
        "ONNX Runtime is available, but CUDAExecutionProvider is not.");
  }
  if (code == QStringLiteral("cuda_unavailable")) {
    return QStringLiteral("The daemon could not initialize CUDA.");
  }
  if (code == QStringLiteral("missing_model_packs"))
    return QStringLiteral("Required Open Video model packs are missing.");
  return {};
}

QString OpenCudaSetupFixText(const QString &code) {
  if (code == QStringLiteral("disabled_in_build")) {
    return QStringLiteral(
        "Fix: rebuild StudioCast with -DSTUDIOCAST_ENABLE_OPEN_CUDA=ON, then "
        "restart the GUI and daemon.");
  }
  if (code == QStringLiteral("onnxruntime_not_found")) {
    return QStringLiteral(
        "Fix: install or point CMake at ONNX Runtime, rebuild StudioCast, then "
        "restart the GUI and daemon.");
  }
  if (code == QStringLiteral("onnxruntime_cuda_provider_unavailable")) {
    return QStringLiteral(
        "Fix: install or build ONNX Runtime with CUDAExecutionProvider support, "
        "rebuild StudioCast, then restart the GUI and daemon.");
  }
  if (code == QStringLiteral("cuda_unavailable")) {
    return QStringLiteral(
        "Fix: check the NVIDIA driver/CUDA runtime and restart StudioCast after "
        "CUDA initializes successfully.");
  }
  return {};
}

QString OpenCudaSetupText(const DaemonVideoStatus &st,
                          bool includeInstallHints) {
  QStringList lines;
  lines << QStringLiteral("Open Video / Open CUDA unavailable.");

  if (!st.open_cuda_summary.trimmed().isEmpty())
    lines << st.open_cuda_summary.trimmed();

  for (const QString &code : OpenCudaBlockerCodes(st)) {
    const QString reason = OpenCudaSetupReasonText(code);
    const QString fix = OpenCudaSetupFixText(code);
    if (!reason.isEmpty() && !lines.contains(reason))
      lines << reason;
    if (!fix.isEmpty() && !lines.contains(fix))
      lines << fix;
  }

  if (OpenCudaHasBlockerCode(st, QStringLiteral("disabled_in_build"))) {
    lines << QStringLiteral(
        "Source build command: cmake -S . -B build "
        "-DSTUDIOCAST_ENABLE_OPEN_CUDA=ON && cmake --build build --target "
        "studiocast studiocastd");
  }

  if (OpenCudaHasSetupBlocker(st) && !st.open_cuda_installed_models.isEmpty()) {
    lines << QStringLiteral(
        "Model packs were found, but the backend cannot use them until this "
        "setup issue is fixed.");
  }

  if (!st.open_cuda_blocked_details.isEmpty()) {
    lines << QStringLiteral("");
    lines << st.open_cuda_blocked_details;
  }

  if (!st.open_cuda_missing_models.isEmpty()) {
    lines << QStringLiteral("");
    lines << QStringLiteral("Missing/invalid model packs:");
    for (auto it = st.open_cuda_missing_models.begin();
         it != st.open_cuda_missing_models.end(); ++it) {
      lines << QStringLiteral("- %1: %2").arg(it.key(), it.value());
    }
  }

  if (includeInstallHints && !st.open_cuda_install_hints.isEmpty()) {
    lines << QStringLiteral("");
    lines << st.open_cuda_install_hints;
  }

  return lines.join(QStringLiteral("\n")).trimmed();
}

QString FirstLine(const QString &s) {
  const QString t = s.trimmed();
  const qsizetype nl = t.indexOf('\n');
  if (nl < 0)
    return t;
  return t.left(nl).trimmed();
}

QString FriendlyBackendLabel(const QString &id) {
  const QString v = id.trimmed().toLower();
  if (v.isEmpty())
    return QStringLiteral("—");

  // Collapse internal engine IDs into user-facing labels.
  if (v == QStringLiteral("maxine") || v.startsWith(QStringLiteral("maxine")))
    return QStringLiteral("Maxine");
  if (v == QStringLiteral("open_cuda") || v == QStringLiteral("open_source") ||
      v == QStringLiteral("open_video") || v == QStringLiteral("open"))
    return QStringLiteral("Open Source");

  if (v == QStringLiteral("passthrough"))
    return QStringLiteral("Pass-through");

  // Common compute descriptors that can appear in daemon mapping.
  if (v == QStringLiteral("cuda") || v.startsWith(QStringLiteral("cuda_")) ||
      v.contains(QStringLiteral("cuda_ep")))
    return QStringLiteral("GPU");
  if (v == QStringLiteral("gpu"))
    return QStringLiteral("GPU");
  if (v == QStringLiteral("cpu"))
    return QStringLiteral("CPU");

  if (v == QStringLiteral("builtin"))
    return QStringLiteral("Built-in");

  return id;
}

QString SanitizeBackendNote(QString note) {
  note.replace(QStringLiteral("Open CUDA"), QStringLiteral("Open Source"),
               Qt::CaseInsensitive);
  note.replace(QStringLiteral("open_cuda"), QStringLiteral("open_source"),
               Qt::CaseInsensitive);
  return note;
}

QString SummarizeEffectsBackends(const QString &raw) {
  const QString t = raw.trimmed();
  if (t.isEmpty())
    return {};

  QStringList keys;
  const auto parts = t.split(',', Qt::SkipEmptyParts);
  for (const auto &part : parts) {
    const QString p = part.trimmed();
    if (p.isEmpty())
      continue;

    QString backend;
    const qsizetype colon = p.indexOf(QChar(':'));
    if (colon >= 0) {
      backend = p.mid(colon + 1).trimmed();
    } else {
      backend = p;
    }

    if (backend.isEmpty())
      continue;
    const QString key = backend.trimmed().toLower();
    if (!keys.contains(key))
      keys.push_back(key);
  }

  if (keys.isEmpty())
    return {};

  QStringList labels;
  for (const auto &k : keys) {
    labels.push_back(FriendlyBackendLabel(k));
  }

  if (labels.size() == 1)
    return labels[0];
  return QStringLiteral("Mixed (%1)").arg(labels.join(QStringLiteral(" + ")));
}

bool DaemonRequest(const std::string &request, std::string *outJson,
                   QString *outErr) {
  studiocast::ipc::DaemonCallResult res;
  studiocast::ipc::DaemonCallOptions options;
  options.connect_timeout_ms = 500;
  options.io_timeout_ms = 2500;
  std::string err;
  if (!studiocast::ipc::DaemonCall(request, &res, &err, options)) {
    if (outErr)
      *outErr = QString::fromStdString(err);
    return false;
  }
  if (!res.ok) {
    if (outErr)
      *outErr = QString::fromStdString(res.error_json.empty() ? "daemon_error"
                                                              : res.error_json);
    return false;
  }
  if (outJson)
    *outJson = res.json;
  return true;
}

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

QLabel *SectionLabel(const QString &text, QWidget *parent) {
  auto *label = new QLabel(text, parent);
  label->setProperty("scRole", "sectionTitle");
  return label;
}

void SetDynamicProperty(QWidget *widget, const char *name,
                        const QString &value) {
  if (!widget)
    return;
  widget->setProperty(name, value);
  widget->style()->unpolish(widget);
  widget->style()->polish(widget);
}

QFrame *EffectPanel(const QString &title, QWidget *parent,
                    QVBoxLayout **layoutOut) {
  auto *panel = new QFrame(parent);
  panel->setProperty("scRole", "cameraEffect");
  auto *layout = new QVBoxLayout(panel);
  layout->setContentsMargins(12, 10, 12, 12);
  layout->setSpacing(10);
  layout->addWidget(SectionLabel(title, panel));
  if (layoutOut)
    *layoutOut = layout;
  return panel;
}

void ConfigureModelCombo(QComboBox *combo) {
  if (!combo)
    return;
  combo->setMinimumContentsLength(28);
  combo->setSizeAdjustPolicy(
      QComboBox::AdjustToMinimumContentsLengthWithIcon);
  combo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

} // namespace

VideoPage::VideoPage(QWidget *parent) : QWidget(parent) {
  auto *root = new QVBoxLayout(this);
  root->setContentsMargins(16, 16, 16, 16);
  root->setSpacing(12);

  auto *titleRow = new QHBoxLayout();
  titleRow->setContentsMargins(0, 0, 0, 0);

  auto *title = new QLabel("Camera", this);
  title->setProperty("scRole", "title");
  titleRow->addWidget(title);

  titleRow->addStretch(1);

  diagnosticsToggle_ = new QToolButton(this);
  diagnosticsToggle_->setText("Show Diagnostics");
  diagnosticsToggle_->setToolButtonStyle(Qt::ToolButtonTextOnly);
  diagnosticsToggle_->setCheckable(true);
  diagnosticsToggle_->setChecked(false);
  titleRow->addWidget(diagnosticsToggle_);

  root->addLayout(titleRow);

  auto *workspace = new QWidget(this);
  auto *workspaceLayout = new QHBoxLayout(workspace);
  workspaceLayout->setContentsMargins(0, 0, 0, 0);
  workspaceLayout->setSpacing(12);

  auto *previewBox = new QGroupBox("Preview", workspace);
  previewBox->setProperty("scRole", "cameraPrimary");
  auto *previewLayout = new QVBoxLayout(previewBox);
  previewLayout->setSpacing(10);
  preview_ = new VideoPreviewWidget(previewBox);
  preview_->setMinimumHeight(360);
  preview_->SetStatusText("Preview off");
  previewLayout->addWidget(preview_, 1);

  auto *previewRow = new QHBoxLayout();
  previewCheck_ = new QCheckBox("Preview", previewBox);
  previewCheck_->setToolTip(
      "Opens the virtual camera in this GUI. This counts as a consumer.");
  previewRow->addWidget(previewCheck_);
  previewRow->addStretch(1);
  previewLayout->addLayout(previewRow);

  previewLayout->addWidget(MutedLabel(
      "Preview opens StudioCast Camera in this GUI and counts as a camera "
      "consumer.",
      previewBox));

  auto *runBox = new QGroupBox("Run Camera", workspace);
  runBox->setMaximumWidth(360);
  auto *runLayout = new QVBoxLayout(runBox);
  runLayout->setSpacing(10);

  cameraStateLabel_ = new QLabel("Checking service", runBox);
  cameraStateLabel_->setProperty("scRole", "statusPill");
  cameraStateLabel_->setAlignment(Qt::AlignCenter);
  runLayout->addWidget(cameraStateLabel_);

  cameraDetailLabel_ = MutedLabel(QString(), runBox);
  runLayout->addWidget(cameraDetailLabel_);

  auto *ctlRow = new QHBoxLayout();
  startBtn_ = new QPushButton("Start Camera", runBox);
  startBtn_->setProperty("scVariant", "primary");
  startBtn_->setMinimumHeight(40);
  stopBtn_ = new QPushButton("Stop", runBox);
  stopBtn_->setProperty("scVariant", "danger");
  stopBtn_->setMinimumHeight(40);
  ctlRow->addWidget(startBtn_, 1);
  ctlRow->addWidget(stopBtn_, 0);
  runLayout->addLayout(ctlRow);

  auto *engineRow = new QHBoxLayout();
  engineRow->addWidget(new QLabel("Backend:", runBox));
  engineCombo_ = new QComboBox(runBox);
  engineCombo_->addItem("Auto", "auto");
  engineCombo_->addItem("Maxine", "maxine");
  engineCombo_->addItem("Open Source", "open_cuda");
  engineRow->addWidget(engineCombo_, 1);
  runLayout->addLayout(engineRow);

  auto *computeRow = new QHBoxLayout();
  computeRow->addWidget(new QLabel("Compute:", runBox));
  computeBackendCombo_ = new QComboBox(runBox);
  computeBackendCombo_->setObjectName(
      QStringLiteral("videoComputeBackendCombo"));
  computeBackendCombo_->addItem("Auto", "auto");
  computeBackendCombo_->addItem("CPU", "cpu");
  computeBackendCombo_->addItem("CUDA", "cuda");
  computeBackendCombo_->addItem("Vulkan", "vulkan");
  computeRow->addWidget(computeBackendCombo_, 1);
  runLayout->addLayout(computeRow);

  auto *activeRow = new QHBoxLayout();
  activeRow->addWidget(new QLabel("Active:", runBox));
  effectEngineValue_ = ValueLabel("—", runBox);
  activeRow->addWidget(effectEngineValue_, 1);
  runLayout->addLayout(activeRow);

  auto *computeActiveRow = new QHBoxLayout();
  computeActiveRow->addWidget(new QLabel("Active compute:", runBox));
  computeBackendValue_ = ValueLabel("—", runBox);
  computeBackendValue_->setObjectName(
      QStringLiteral("videoComputeBackendActiveValue"));
  computeActiveRow->addWidget(computeBackendValue_, 1);
  runLayout->addLayout(computeActiveRow);

  engineInfoBanner_ = new QLabel(runBox);
  engineInfoBanner_->setWordWrap(true);
  engineInfoBanner_->setProperty("scBanner", "info");
  engineInfoBanner_->setVisible(false);
  runLayout->addWidget(engineInfoBanner_);

  maxineBanner_ = new QLabel(runBox);
  maxineBanner_->setObjectName(QStringLiteral("cameraEngineWarningBanner"));
  maxineBanner_->setWordWrap(true);
  maxineBanner_->setProperty("scBanner", "warning");
  maxineBanner_->setVisible(false);
  runLayout->addWidget(maxineBanner_);
  runLayout->addStretch(1);

  workspaceLayout->addWidget(previewBox, 3);
  workspaceLayout->addWidget(runBox, 1);
  root->addWidget(workspace);

  auto *setupBox = new QGroupBox("Setup", this);
  auto *setupLayout = new QVBoxLayout(setupBox);
  setupLayout->setSpacing(10);

  setupLockLabel_ = MutedLabel(
      "Setup is locked while camera processing is running.", setupBox);
  setupLockLabel_->setVisible(false);
  setupLayout->addWidget(setupLockLabel_);

  auto *setupGrid = new QGridLayout();
  setupGrid->setColumnStretch(1, 1);
  setupGrid->setHorizontalSpacing(10);
  setupGrid->setVerticalSpacing(10);

  setupGrid->addWidget(new QLabel("Input camera:", setupBox), 0, 0);
  inputCombo_ = new QComboBox(setupBox);
  inputCombo_->setObjectName(QStringLiteral("videoInputCombo"));
  setupGrid->addWidget(inputCombo_, 0, 1);
  refreshBtn_ = new QPushButton("Refresh", setupBox);
  setupGrid->addWidget(refreshBtn_, 0, 2);

  setupGrid->addWidget(new QLabel("Virtual camera:", setupBox), 1, 0);
  outputCombo_ = new QComboBox(setupBox);
  outputCombo_->setObjectName(QStringLiteral("videoOutputCombo"));
  setupGrid->addWidget(outputCombo_, 1, 1, 1, 2);

  setupGrid->addWidget(new QLabel("Output format:", setupBox), 2, 0);
  outputFormatCombo_ = new QComboBox(setupBox);
  outputFormatCombo_->setObjectName(QStringLiteral("videoOutputFormatCombo"));
  outputFormatCombo_->addItem("RGB24 / RGB3", "rgb24");
  outputFormatCombo_->addItem("YUYV 4:2:2", "yuyv");
  setupGrid->addWidget(outputFormatCombo_, 2, 1, 1, 2);

  auto *formatRow = new QHBoxLayout();
  formatRow->addWidget(new QLabel("Width:", setupBox));
  widthSpin_ = new QSpinBox(setupBox);
  widthSpin_->setRange(160, 3840);
  widthSpin_->setValue(1280);
  formatRow->addWidget(widthSpin_);

  formatRow->addWidget(new QLabel("Height:", setupBox));
  heightSpin_ = new QSpinBox(setupBox);
  heightSpin_->setRange(120, 2160);
  heightSpin_->setValue(720);
  formatRow->addWidget(heightSpin_);

  formatRow->addWidget(new QLabel("FPS:", setupBox));
  fpsSpin_ = new QSpinBox(setupBox);
  fpsSpin_->setRange(1, 120);
  fpsSpin_->setValue(30);
  formatRow->addWidget(fpsSpin_);
  formatRow->addStretch(1);
  auto *formatWidget = new QWidget(setupBox);
  formatWidget->setLayout(formatRow);
  setupGrid->addWidget(formatWidget, 3, 1, 1, 2);
  setupLayout->addLayout(setupGrid);
  root->addWidget(setupBox);

  auto *effectsBox = new QGroupBox("Effects", this);
  auto *effectsLayout = new QVBoxLayout(effectsBox);
  effectsLayout->setSpacing(10);

  auto *quickFxRow = new QHBoxLayout();
  mirrorCheck_ = new QCheckBox("Mirror", effectsBox);
  mirrorCheck_->setToolTip("Saves the camera mirror setting.");
  quickFxRow->addWidget(mirrorCheck_);
  quickFxRow->addStretch(1);
  effectsLayout->addLayout(quickFxRow);

  auto *effectsGrid = new QGridLayout();
  effectsGrid->setHorizontalSpacing(10);
  effectsGrid->setVerticalSpacing(10);
  effectsGrid->setColumnStretch(0, 1);
  effectsGrid->setColumnStretch(1, 1);

  // Virtual Background
  QVBoxLayout *vbLayout = nullptr;
  auto *vbBox = EffectPanel("Virtual Background", effectsBox, &vbLayout);

  auto *vbRow = new QHBoxLayout();
  vbRow->addWidget(new QLabel("Mode:", vbBox));
  backgroundCombo_ = new QComboBox(vbBox);
  backgroundCombo_->setObjectName(QStringLiteral("videoBackgroundModeCombo"));
  backgroundCombo_->addItem("None", "none");
  backgroundCombo_->addItem("Blur", "blur");
  backgroundCombo_->addItem("Remove", "remove");
  backgroundCombo_->addItem("Replace", "replace");
  vbRow->addWidget(backgroundCombo_, 1);
  vbRow->addSpacing(12);
  backgroundStrengthLabel_ = new QLabel("Blur strength:", vbBox);
  vbRow->addWidget(backgroundStrengthLabel_);
  backgroundStrengthSpin_ = new QSpinBox(vbBox);
  backgroundStrengthSpin_->setRange(0, 100);
  backgroundStrengthSpin_->setValue(50);
  backgroundStrengthSpin_->setSuffix("%");
  backgroundStrengthSpin_->setMaximumWidth(90);
  vbRow->addWidget(backgroundStrengthSpin_);
  vbLayout->addLayout(vbRow);

  auto *vbModelRow = new QHBoxLayout();
  vbModelLabel_ = new QLabel("Model:", vbBox);
  vbModelCombo_ = new QComboBox(vbBox);
  ConfigureModelCombo(vbModelCombo_);
  vbModelRow->addWidget(vbModelLabel_);
  vbModelRow->addWidget(vbModelCombo_, 1);
  vbLayout->addLayout(vbModelRow);

  vbModelLabel_->setVisible(false);
  vbModelCombo_->setVisible(false);

  auto *vbParamRow = new QHBoxLayout();
  backgroundRemoveColorLabel_ = new QLabel("Remove color (#RRGGBB):", vbBox);
  vbParamRow->addWidget(backgroundRemoveColorLabel_);
  backgroundRemoveColorEdit_ = new QLineEdit(vbBox);
  backgroundRemoveColorEdit_->setPlaceholderText("#000000");
  backgroundRemoveColorEdit_->setMaximumWidth(110);
  vbParamRow->addWidget(backgroundRemoveColorEdit_);

  vbParamRow->addSpacing(12);
  backgroundReplaceImageLabel_ = new QLabel("Replace image:", vbBox);
  vbParamRow->addWidget(backgroundReplaceImageLabel_);
  backgroundReplaceImageEdit_ = new QLineEdit(vbBox);
  backgroundReplaceImageEdit_->setObjectName(
      QStringLiteral("videoBackgroundReplaceImageEdit"));
  vbParamRow->addWidget(backgroundReplaceImageEdit_, 1);
  browseReplaceImageBtn_ = new QPushButton("Browse…", vbBox);
  browseReplaceImageBtn_->setObjectName(
      QStringLiteral("videoBackgroundBrowseReplaceImageButton"));
  vbParamRow->addWidget(browseReplaceImageBtn_);
  vbLayout->addLayout(vbParamRow);

  // Auto Frame
  QVBoxLayout *afLayout = nullptr;
  auto *afBox = EffectPanel("Auto Frame", effectsBox, &afLayout);

  auto *afRow = new QHBoxLayout();
  autoFrameCheck_ = new QCheckBox("Enable", afBox);
  afRow->addWidget(autoFrameCheck_);
  afRow->addSpacing(12);
  afRow->addWidget(new QLabel("Zoom:", afBox));
  autoFrameZoomSlider_ = new QSlider(Qt::Horizontal, afBox);
  autoFrameZoomSlider_->setRange(0, 100);
  autoFrameZoomSlider_->setValue(50);
  afRow->addWidget(autoFrameZoomSlider_, 1);
  autoFrameZoomValue_ = new QLabel("50%", afBox);
  autoFrameZoomValue_->setMinimumWidth(44);
  autoFrameZoomValue_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  afRow->addWidget(autoFrameZoomValue_);
  afLayout->addLayout(afRow);

  auto *afModelRow = new QHBoxLayout();
  autoFrameModelLabel_ = new QLabel("Model:", afBox);
  autoFrameModelCombo_ = new QComboBox(afBox);
  ConfigureModelCombo(autoFrameModelCombo_);
  afModelRow->addWidget(autoFrameModelLabel_);
  afModelRow->addWidget(autoFrameModelCombo_, 1);
  afLayout->addLayout(afModelRow);
  autoFrameModelLabel_->setVisible(false);
  autoFrameModelCombo_->setVisible(false);

  // Eye Contact
  QVBoxLayout *ecLayout = nullptr;
  auto *ecBox = EffectPanel("Eye Contact", effectsBox, &ecLayout);

  auto *ecRow = new QHBoxLayout();
  eyeContactCheck_ = new QCheckBox("Enable", ecBox);
  ecRow->addWidget(eyeContactCheck_);
  ecRow->addSpacing(12);
  ecRow->addWidget(new QLabel("Strength:", ecBox));
  eyeContactStrengthSlider_ = new QSlider(Qt::Horizontal, ecBox);
  eyeContactStrengthSlider_->setRange(0, 100);
  eyeContactStrengthSlider_->setValue(50);
  ecRow->addWidget(eyeContactStrengthSlider_, 1);
  eyeContactStrengthValue_ = new QLabel("50%", ecBox);
  eyeContactStrengthValue_->setMinimumWidth(44);
  eyeContactStrengthValue_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  ecRow->addWidget(eyeContactStrengthValue_);
  ecRow->addSpacing(12);
  eyeContactLookAwayCheck_ = new QCheckBox("Allow look-away", ecBox);
  eyeContactLookAwayCheck_->setChecked(true);
  ecRow->addWidget(eyeContactLookAwayCheck_);
  ecLayout->addLayout(ecRow);

  auto *ecModelRow = new QHBoxLayout();
  eyeContactModelLabel_ = new QLabel("Model:", ecBox);
  eyeContactModelCombo_ = new QComboBox(ecBox);
  ConfigureModelCombo(eyeContactModelCombo_);
  ecModelRow->addWidget(eyeContactModelLabel_);
  ecModelRow->addWidget(eyeContactModelCombo_, 1);
  ecLayout->addLayout(ecModelRow);
  eyeContactModelLabel_->setVisible(false);
  eyeContactModelCombo_->setVisible(false);

  // Video Noise Removal
  QVBoxLayout *dnLayout = nullptr;
  auto *dnBox = EffectPanel("Video Noise Removal", effectsBox, &dnLayout);

  auto *dnRow = new QHBoxLayout();
  denoiseCheck_ = new QCheckBox("Enable", dnBox);
  dnRow->addWidget(denoiseCheck_);
  dnRow->addSpacing(12);
  dnRow->addWidget(new QLabel("Strength:", dnBox));
  denoiseStrengthSlider_ = new QSlider(Qt::Horizontal, dnBox);
  denoiseStrengthSlider_->setRange(0, 100);
  denoiseStrengthSlider_->setValue(50);
  dnRow->addWidget(denoiseStrengthSlider_, 1);
  denoiseStrengthValue_ = new QLabel("50%", dnBox);
  denoiseStrengthValue_->setMinimumWidth(44);
  denoiseStrengthValue_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  dnRow->addWidget(denoiseStrengthValue_);
  dnLayout->addLayout(dnRow);

  auto *dnModelRow = new QHBoxLayout();
  denoiseModelLabel_ = new QLabel("Model:", dnBox);
  denoiseModelCombo_ = new QComboBox(dnBox);
  ConfigureModelCombo(denoiseModelCombo_);
  dnModelRow->addWidget(denoiseModelLabel_);
  dnModelRow->addWidget(denoiseModelCombo_, 1);
  dnLayout->addLayout(dnModelRow);
  denoiseModelLabel_->setVisible(false);
  denoiseModelCombo_->setVisible(false);

  // Virtual Key Light
  QVBoxLayout *vklLayout = nullptr;
  auto *vklBox = EffectPanel("Virtual Key Light", effectsBox, &vklLayout);

  auto *vklRow = new QHBoxLayout();
  virtualKeyLightCheck_ = new QCheckBox("Enable", vklBox);
  vklRow->addWidget(virtualKeyLightCheck_);
  vklRow->addSpacing(12);

  vklRow->addWidget(new QLabel("Intensity:", vklBox));
  virtualKeyLightIntensitySpin_ = new QSpinBox(vklBox);
  virtualKeyLightIntensitySpin_->setRange(0, 100);
  virtualKeyLightIntensitySpin_->setValue(70);
  virtualKeyLightIntensitySpin_->setSuffix("%");
  virtualKeyLightIntensitySpin_->setMaximumWidth(90);
  vklRow->addWidget(virtualKeyLightIntensitySpin_);

  vklRow->addSpacing(12);
  vklRow->addWidget(new QLabel("Temp:", vklBox));
  virtualKeyLightTempCombo_ = new QComboBox(vklBox);
  virtualKeyLightTempCombo_->addItem("Neutral", "neutral");
  virtualKeyLightTempCombo_->addItem("Warm", "warm");
  virtualKeyLightTempCombo_->addItem("Cool", "cool");
  vklRow->addWidget(virtualKeyLightTempCombo_);

  vklRow->addSpacing(12);
  vklRow->addWidget(new QLabel("Pan:", vklBox));
  virtualKeyLightPanSpin_ = new QSpinBox(vklBox);
  virtualKeyLightPanSpin_->setRange(-180, 180);
  virtualKeyLightPanSpin_->setValue(0);
  virtualKeyLightPanSpin_->setSuffix("°");
  virtualKeyLightPanSpin_->setMaximumWidth(90);
  vklRow->addWidget(virtualKeyLightPanSpin_);

  vklRow->addStretch(1);
  vklLayout->addLayout(vklRow);

  auto *vklRow2 = new QHBoxLayout();
  vklRow2->addWidget(new QLabel("HDRI (optional):", vklBox));
  virtualKeyLightHdriEdit_ = new QLineEdit(vklBox);
  vklRow2->addWidget(virtualKeyLightHdriEdit_, 1);
  browseVirtualKeyLightHdriBtn_ = new QPushButton("Browse…", vklBox);
  vklRow2->addWidget(browseVirtualKeyLightHdriBtn_);
  vklLayout->addLayout(vklRow2);

  // Vignette
  QVBoxLayout *vigOuterLayout = nullptr;
  auto *vigBox = EffectPanel("Vignette", effectsBox, &vigOuterLayout);
  auto *vigLayout = new QHBoxLayout();
  vignetteCheck_ = new QCheckBox("Enable", vigBox);
  vigLayout->addWidget(vignetteCheck_);
  vigLayout->addSpacing(12);
  vigLayout->addWidget(new QLabel("Intensity:", vigBox));
  vignetteIntensitySlider_ = new QSlider(Qt::Horizontal, vigBox);
  vignetteIntensitySlider_->setRange(0, 100);
  vignetteIntensitySlider_->setValue(35);
  vigLayout->addWidget(vignetteIntensitySlider_, 1);
  vignetteIntensityValue_ = new QLabel("35%", vigBox);
  vignetteIntensityValue_->setMinimumWidth(44);
  vignetteIntensityValue_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  vigLayout->addWidget(vignetteIntensityValue_);
  vigLayout->addSpacing(12);
  vignetteCenterOnFaceCheck_ =
      new QCheckBox("Center on Auto Frame subject", vigBox);
  vignetteCenterOnFaceCheck_->setChecked(true);
  vigLayout->addWidget(vignetteCenterOnFaceCheck_);
  vigOuterLayout->addLayout(vigLayout);

  effectsGrid->addWidget(vbBox, 0, 0, 1, 2);
  effectsGrid->addWidget(afBox, 1, 0);
  effectsGrid->addWidget(ecBox, 1, 1);
  effectsGrid->addWidget(dnBox, 2, 0);
  effectsGrid->addWidget(vklBox, 2, 1);
  effectsGrid->addWidget(vigBox, 3, 0, 1, 2);
  effectsLayout->addLayout(effectsGrid);
  root->addWidget(effectsBox);

  diagnosticsBox_ = new QGroupBox("Diagnostics", this);
  diagnosticsBox_->setVisible(false);
  auto *detailsLayout = new QVBoxLayout(diagnosticsBox_);
  detailsLayout->setSpacing(10);

  diagnosticsContent_ = new QWidget(diagnosticsBox_);
  auto *detailsContentLayout = new QVBoxLayout(diagnosticsContent_);
  detailsContentLayout->setContentsMargins(0, 0, 0, 0);
  detailsContentLayout->setSpacing(10);

  detailsContentLayout->addWidget(MutedLabel(
      "Technical camera details stay here for setup and support.",
      diagnosticsContent_));

  auto *cmdRow = new QHBoxLayout();
  cmdRow->addWidget(new QLabel("v4l2loopback command:", diagnosticsContent_));
  suggestedCmdEdit_ = new QLineEdit(diagnosticsContent_);
  suggestedCmdEdit_->setReadOnly(true);
  suggestedCmdEdit_->setProperty("scRole", "copyValue");
  suggestedCmdEdit_->setPlaceholderText("No suggested modprobe command.");
  cmdRow->addWidget(suggestedCmdEdit_, 1);
  copyCmdBtn_ = new QPushButton("Copy command", diagnosticsContent_);
  cmdRow->addWidget(copyCmdBtn_);
  detailsContentLayout->addLayout(cmdRow);

  auto *detailsActionsRow = new QHBoxLayout();
  openInstallHintsBtn_ =
      new QPushButton("Open install hints", diagnosticsContent_);
  detailsActionsRow->addWidget(openInstallHintsBtn_, 0, Qt::AlignLeft);
  auto *copyStatusBtn =
      new QPushButton("Copy raw camera details", diagnosticsContent_);
  detailsActionsRow->addWidget(copyStatusBtn, 0, Qt::AlignLeft);
  detailsActionsRow->addStretch(1);
  detailsContentLayout->addLayout(detailsActionsRow);

  diagnosticsText_ = new QPlainTextEdit(diagnosticsContent_);
  diagnosticsText_->setReadOnly(true);
  diagnosticsText_->setMinimumHeight(140);
  detailsContentLayout->addWidget(
      new QLabel("Engine diagnostics:", diagnosticsContent_));
  detailsContentLayout->addWidget(diagnosticsText_);

  statusText_ = new QPlainTextEdit(diagnosticsContent_);
  statusText_->setReadOnly(true);
  statusText_->setMinimumHeight(260);
  detailsContentLayout->addWidget(
      new QLabel("Raw camera status:", diagnosticsContent_));
  detailsContentLayout->addWidget(statusText_, 1);

  diagnosticsContent_->setVisible(false);
  detailsLayout->addWidget(diagnosticsContent_);
  connect(copyStatusBtn, &QPushButton::clicked, this, [this] {
    if (auto *cb = QGuiApplication::clipboard())
      cb->setText(statusText_ ? statusText_->toPlainText() : QString());
  });

  root->addWidget(diagnosticsBox_);
  root->addStretch(1);

  connect(refreshBtn_, &QPushButton::clicked, this, &VideoPage::Refresh);
  connect(copyCmdBtn_, &QPushButton::clicked, this,
          &VideoPage::CopySuggestedCommand);
  connect(inputCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, [this](int) { MarkSetupControlsEdited(); });
  connect(outputCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, [this](int) { MarkSetupControlsEdited(); });
  connect(outputFormatCombo_,
          QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          [this](int) { MarkSetupControlsEdited(); });
  connect(widthSpin_, QOverload<int>::of(&QSpinBox::valueChanged), this,
          [this](int) { MarkSetupControlsEdited(); });
  connect(heightSpin_, QOverload<int>::of(&QSpinBox::valueChanged), this,
          [this](int) { MarkSetupControlsEdited(); });
  connect(fpsSpin_, QOverload<int>::of(&QSpinBox::valueChanged), this,
          [this](int) { MarkSetupControlsEdited(); });
  connect(startBtn_, &QPushButton::clicked, this, &VideoPage::OnStart);
  connect(stopBtn_, &QPushButton::clicked, this, &VideoPage::OnStop);
  connect(previewCheck_, &QCheckBox::toggled, this,
          &VideoPage::OnPreviewToggled);
  connect(diagnosticsToggle_, &QToolButton::toggled, this,
          &VideoPage::SetDiagnosticsVisible);

  connect(engineCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &VideoPage::OnEnginePreferenceChanged);
  connect(computeBackendCombo_,
          QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          &VideoPage::OnComputeBackendChanged);
  connect(mirrorCheck_, &QCheckBox::toggled, this,
          &VideoPage::OnMirrorToggled);

  connect(backgroundCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &VideoPage::OnBackgroundChanged);
  connect(vbModelCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &VideoPage::OnVbModelChanged);
  connect(backgroundStrengthSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
          this, &VideoPage::OnBackgroundStrengthChanged);

  connect(backgroundRemoveColorEdit_, &QLineEdit::editingFinished, this,
          &VideoPage::OnBackgroundRemoveColorChanged);
  connect(backgroundReplaceImageEdit_, &QLineEdit::editingFinished, this,
          &VideoPage::OnBackgroundReplaceImageChanged);
  connect(browseReplaceImageBtn_, &QPushButton::clicked, this,
          &VideoPage::OnBrowseReplaceImage);

  connect(autoFrameCheck_, &QCheckBox::toggled, this,
          &VideoPage::OnAutoFrameToggled);
  connect(autoFrameZoomSlider_, &QSlider::valueChanged, this,
          &VideoPage::OnAutoFrameZoomChanged);
  connect(autoFrameModelCombo_,
          QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          &VideoPage::OnAutoFrameModelChanged);

  connect(eyeContactCheck_, &QCheckBox::toggled, this,
          &VideoPage::OnEyeContactToggled);
  connect(eyeContactStrengthSlider_, &QSlider::valueChanged, this,
          &VideoPage::OnEyeContactStrengthChanged);
  connect(eyeContactLookAwayCheck_, &QCheckBox::toggled, this,
          &VideoPage::OnEyeContactLookAwayToggled);
  connect(eyeContactModelCombo_,
          QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          &VideoPage::OnEyeContactModelChanged);

  connect(denoiseCheck_, &QCheckBox::toggled, this,
          &VideoPage::OnDenoiseToggled);
  connect(denoiseStrengthSlider_, &QSlider::valueChanged, this,
          &VideoPage::OnDenoiseStrengthChanged);
  connect(denoiseModelCombo_,
          QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          &VideoPage::OnDenoiseModelChanged);

  connect(openInstallHintsBtn_, &QPushButton::clicked, this,
          &VideoPage::OnOpenInstallHints);

  connect(virtualKeyLightCheck_, &QCheckBox::toggled, this,
          &VideoPage::OnVirtualKeyLightToggled);
  connect(virtualKeyLightIntensitySpin_,
          QOverload<int>::of(&QSpinBox::valueChanged), this,
          &VideoPage::OnVirtualKeyLightIntensityChanged);
  connect(virtualKeyLightTempCombo_,
          QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          &VideoPage::OnVirtualKeyLightTemperatureChanged);
  connect(virtualKeyLightPanSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
          this, &VideoPage::OnVirtualKeyLightPanChanged);
  connect(virtualKeyLightHdriEdit_, &QLineEdit::editingFinished, this,
          &VideoPage::OnVirtualKeyLightHdriChanged);
  connect(browseVirtualKeyLightHdriBtn_, &QPushButton::clicked, this,
          &VideoPage::OnBrowseVirtualKeyLightHdri);

  connect(vignetteCheck_, &QCheckBox::toggled, this,
          &VideoPage::OnVignetteToggled);
  connect(vignetteIntensitySlider_, &QSlider::valueChanged, this,
          &VideoPage::OnVignetteIntensityChanged);
  connect(vignetteCenterOnFaceCheck_, &QCheckBox::toggled, this,
          &VideoPage::OnVignetteCenterOnFaceToggled);

  previewTimer_ = new QTimer(this);
  previewTimer_->setInterval(33);
  connect(previewTimer_, &QTimer::timeout, this, &VideoPage::OnPreviewTick);

  effectsWriteDebounceTimer_ = new QTimer(this);
  effectsWriteDebounceTimer_->setSingleShot(true);
  effectsWriteDebounceTimer_->setInterval(180);
  connect(effectsWriteDebounceTimer_, &QTimer::timeout, this,
          [this] { (void)SendDaemonVideoEffects(); });

  Refresh();
  UpdateStatusText();
  UpdateUiEnabled();
}

VideoPage::~VideoPage() { StopPreview(); }

void VideoPage::ShowError(const QString &title, const QString &details) {
  QMessageBox::critical(this, title, details);
}

void VideoPage::SetDiagnosticsVisible(bool visible) {
  if (diagnosticsToggle_) {
    diagnosticsToggle_->setChecked(visible);
    diagnosticsToggle_->setText(visible ? QStringLiteral("Hide Diagnostics")
                                        : QStringLiteral("Show Diagnostics"));
  }
  if (diagnosticsBox_)
    diagnosticsBox_->setVisible(visible);
  if (diagnosticsContent_)
    diagnosticsContent_->setVisible(visible);
}

void VideoPage::UpdateBackgroundModeOptionVisibility() {
  const QString bgMode =
      backgroundCombo_ ? backgroundCombo_->currentData().toString() : QString();
  const bool showBlur = (bgMode == QStringLiteral("blur"));
  const bool showRemove = (bgMode == QStringLiteral("remove"));
  const bool showReplace = (bgMode == QStringLiteral("replace"));

  if (backgroundStrengthLabel_)
    backgroundStrengthLabel_->setVisible(showBlur);
  if (backgroundStrengthSpin_)
    backgroundStrengthSpin_->setVisible(showBlur);

  if (backgroundRemoveColorLabel_)
    backgroundRemoveColorLabel_->setVisible(showRemove);
  if (backgroundRemoveColorEdit_)
    backgroundRemoveColorEdit_->setVisible(showRemove);

  if (backgroundReplaceImageLabel_)
    backgroundReplaceImageLabel_->setVisible(showReplace);
  if (backgroundReplaceImageEdit_)
    backgroundReplaceImageEdit_->setVisible(showReplace);
  if (browseReplaceImageBtn_)
    browseReplaceImageBtn_->setVisible(showReplace);
}

void VideoPage::Refresh() {
  const auto rep = studiocast::video::ProbeLoopback();
  baseStatusText_ = rep.ToText();

  const QString prevIn = inputCombo_->currentData().toString();
  const QString prevOut = outputCombo_->currentData().toString();

  const QSignalBlocker blockInput(inputCombo_);
  const QSignalBlocker blockOutput(outputCombo_);

  inputCombo_->clear();
  outputCombo_->clear();

  // Always provide an explicit "auto" option so the daemon can choose.
  inputCombo_->addItem("<auto>", "auto");

  int inSet = (prevIn == "auto") ? 0 : -1;
  int outSet = (prevOut == "auto") ? 0 : -1;

  int inAdded = 0;
  int outAdded = 0;

  for (const auto &d : rep.devices) {
    const QString deviceNode = QString::fromStdString(d.dev_node);
    const QString label = DeviceLabel(d);

    if (d.can_read) {
      inputCombo_->addItem(label, deviceNode);
      ++inAdded;
      if (!prevIn.isEmpty() && prevIn == deviceNode)
        inSet = inputCombo_->count() - 1;
    }

    if (d.is_loopback && d.can_write) {
      // Populate output combo lazily; insert <auto> only if we have at least
      // one loopback.
      if (outputCombo_->count() == 0) {
        outputCombo_->addItem("<auto>", "auto");
      }
      outputCombo_->addItem(label, deviceNode);
      ++outAdded;
      if (!prevOut.isEmpty() && prevOut == deviceNode)
        outSet = outputCombo_->count() - 1;
    }
  }

  if (inAdded == 0) {
    inputCombo_->setEnabled(true); // still allow <auto>
    inputCombo_->setCurrentIndex(0);
  } else {
    inputCombo_->setEnabled(true);
    inputCombo_->setCurrentIndex(inSet >= 0 ? inSet : 0);
  }

  if (outAdded == 0) {
    outputCombo_->addItem("<virtual camera missing>", "");
    outputCombo_->setEnabled(false);
  } else {
    outputCombo_->setEnabled(true);
    outputCombo_->setCurrentIndex(outSet >= 0 ? outSet : 0);
  }

  suggestedCmd_ = QString::fromStdString(rep.suggested_modprobe_cmd);
  copyCmdBtn_->setEnabled(!suggestedCmd_.isEmpty());
  if (suggestedCmdEdit_)
    suggestedCmdEdit_->setText(suggestedCmd_);

  ResyncControlsFromCachedStatus();
  UpdateStatusText();
  UpdateUiEnabled();
}

void VideoPage::CopySuggestedCommand() {
  if (suggestedCmd_.isEmpty())
    return;
  if (auto *cb = QGuiApplication::clipboard())
    cb->setText(suggestedCmd_);
}

void VideoPage::UpdateStatus(const DaemonStatusSnapshot &snapshot) {
  daemonReachable_ = snapshot.reachable && snapshot.parsed;
  daemonStatusDetail_ = snapshot.UserServiceDetail();
  daemonLastStatusJson_ = snapshot.rawJson.toStdString();

  if (daemonReachable_) {
    ResyncControlsFromCachedStatus();
  } else {
    StopPreview();
  }

  UpdateStatusText();
  UpdateUiEnabled();
}

bool VideoPage::SyncFromCachedDaemonStatus() {
  if (!daemonReachable_ || daemonLastStatusJson_.empty())
    return false;

  // Transitional compatibility: the shared snapshot owns routine status
  // delivery, but a few legacy camera controls still need fields that are not
  // typed in DaemonStatusSnapshot yet. Parse the cached snapshot payload only;
  // do not issue page-local status IPC here.
  DaemonVideoStatus st;
  QString parseErr;
  if (!ParseDaemonStatusJson(daemonLastStatusJson_, &st, &parseErr)) {
    daemonReachable_ = false;
    daemonStatusDetail_ = parseErr;
    return false;
  }

  const QString input = st.input_device;
  const QString output = st.output_device;
  const int w = st.width;
  const int h = st.height;
  const int fps = st.fps;
  const QString outputFormat = st.output_format_requested.isEmpty()
                                   ? QStringLiteral("rgb24")
                                   : st.output_format_requested;
  const bool applySetupControls = st.enabled || !setupControlsDirty_;
  effects_ = st.effects_valid
                 ? st.effects
                 : studiocast::video::effects::BroadcastCameraEffects{};

  if (engineCombo_) {
    engineCombo_->blockSignals(true);
    const QString v = QString::fromStdString(
        studiocast::video::effects::ToString(effects_.engine));
    const int idx = engineCombo_->findData(v);
    engineCombo_->setCurrentIndex(idx >= 0 ? idx : 0);
    engineCombo_->blockSignals(false);
  }

  if (computeBackendCombo_) {
    computeBackendCombo_->blockSignals(true);
    const QString v =
        st.compute_preference.isEmpty() ? QStringLiteral("auto")
                                        : st.compute_preference;
    const int idx = computeBackendCombo_->findData(v);
    computeBackendCombo_->setCurrentIndex(idx >= 0 ? idx : 0);
    computeBackendCombo_->blockSignals(false);
  }

  if (mirrorCheck_) {
    mirrorCheck_->blockSignals(true);
    mirrorCheck_->setChecked(effects_.mirror);
    mirrorCheck_->blockSignals(false);
  }

  const bool autoFrame = effects_.auto_frame.enabled;
  const int autoFrameStrength = effects_.auto_frame.strength;
  const QString vbMode = QString::fromStdString(
      studiocast::video::effects::ToString(effects_.virtual_background.mode));
  const int vbStrength = effects_.virtual_background.strength;
  const QString vbRemoveColor =
      QString::fromStdString(effects_.virtual_background.remove_color);
  const QString vbReplacePath =
      QString::fromStdString(effects_.virtual_background.replace_path);

  const bool eyeContact = effects_.eye_contact.enabled;
  const int eyeContactStrength = effects_.eye_contact.strength;
  const bool eyeContactLookAway = effects_.eye_contact.look_away_enabled;

  const bool denoise = effects_.video_noise_removal.enabled;
  const int denoiseStrength = effects_.video_noise_removal.strength;

  const bool virtualKeyLight = effects_.virtual_key_light.enabled;
  const int virtualKeyLightIntensity = effects_.virtual_key_light.intensity;
  const auto vklPresetToStr = [](int p) -> const char * {
    switch (p) {
    case 1:
      return "warm";
    case 2:
      return "cool";
    default:
      return "neutral";
    }
  };
  const QString virtualKeyLightTemp = QString::fromUtf8(
      vklPresetToStr(effects_.virtual_key_light.temperature_preset));
  const int virtualKeyLightPan =
      effects_.virtual_key_light.direction_pan_degrees;
  const QString virtualKeyLightHdri =
      QString::fromStdString(effects_.virtual_key_light.hdri_path);

  const bool vignette = effects_.vignette.enabled;
  const int vignetteIntensity = effects_.vignette.intensity;
  const bool vignetteCenterOnFace = effects_.vignette.center_on_tracked_face;

  // Apply setup controls only while they have no unsaved local edits. Effects
  // still resync routinely because effect edits are written immediately.
  if (applySetupControls) {
    const QString inKey = input.isEmpty() ? "auto" : input;
    const int inIdx = inputCombo_->findData(inKey);
    if (inIdx >= 0) {
      const QSignalBlocker blockInput(inputCombo_);
      inputCombo_->setCurrentIndex(inIdx);
    }

    const QString outKey = output.isEmpty() ? "auto" : output;
    const int outIdx = outputCombo_->findData(outKey);
    if (outIdx >= 0) {
      const QSignalBlocker blockOutput(outputCombo_);
      outputCombo_->setCurrentIndex(outIdx);
    }

    if (w > 0) {
      const QSignalBlocker blockWidth(widthSpin_);
      widthSpin_->setValue(w);
    }
    if (h > 0) {
      const QSignalBlocker blockHeight(heightSpin_);
      heightSpin_->setValue(h);
    }
    if (fps > 0) {
      const QSignalBlocker blockFps(fpsSpin_);
      fpsSpin_->setValue(fps);
    }
    if (outputFormatCombo_) {
      const int fmtIdx = outputFormatCombo_->findData(outputFormat);
      if (fmtIdx >= 0) {
        const QSignalBlocker blockFmt(outputFormatCombo_);
        outputFormatCombo_->setCurrentIndex(fmtIdx);
      }
    }

    if (st.enabled)
      setupControlsDirty_ = false;
  }

  if (backgroundCombo_) {
    backgroundCombo_->blockSignals(true);
    const QString key = vbMode.isEmpty() ? "none" : vbMode;
    const int idx = backgroundCombo_->findData(key);
    if (idx >= 0)
      backgroundCombo_->setCurrentIndex(idx);
    backgroundCombo_->blockSignals(false);
  }

  if (backgroundStrengthSpin_) {
    backgroundStrengthSpin_->blockSignals(true);
    backgroundStrengthSpin_->setValue(std::max(0, std::min(100, vbStrength)));
    backgroundStrengthSpin_->blockSignals(false);
  }

  if (backgroundRemoveColorEdit_) {
    backgroundRemoveColorEdit_->blockSignals(true);
    backgroundRemoveColorEdit_->setText(vbRemoveColor);
    backgroundRemoveColorEdit_->blockSignals(false);
  }
  if (backgroundReplaceImageEdit_) {
    backgroundReplaceImageEdit_->blockSignals(true);
    backgroundReplaceImageEdit_->setText(vbReplacePath);
    backgroundReplaceImageEdit_->blockSignals(false);
  }

  if (autoFrameCheck_) {
    autoFrameCheck_->blockSignals(true);
    autoFrameCheck_->setChecked(autoFrame);
    autoFrameCheck_->blockSignals(false);
  }
  if (autoFrameZoomSlider_) {
    autoFrameZoomSlider_->blockSignals(true);
    autoFrameZoomSlider_->setValue(
        std::max(0, std::min(100, autoFrameStrength)));
    autoFrameZoomSlider_->blockSignals(false);
  }
  if (autoFrameZoomValue_ && autoFrameZoomSlider_) {
    autoFrameZoomValue_->setText(
        QString::number(autoFrameZoomSlider_->value()) + "%");
  }

  if (eyeContactCheck_) {
    eyeContactCheck_->blockSignals(true);
    eyeContactCheck_->setChecked(eyeContact);
    eyeContactCheck_->blockSignals(false);
  }
  if (eyeContactStrengthSlider_) {
    eyeContactStrengthSlider_->blockSignals(true);
    eyeContactStrengthSlider_->setValue(
        std::max(0, std::min(100, eyeContactStrength)));
    eyeContactStrengthSlider_->blockSignals(false);
  }
  if (eyeContactStrengthValue_ && eyeContactStrengthSlider_) {
    eyeContactStrengthValue_->setText(
        QString::number(eyeContactStrengthSlider_->value()) + "%");
  }
  if (eyeContactLookAwayCheck_) {
    eyeContactLookAwayCheck_->blockSignals(true);
    eyeContactLookAwayCheck_->setChecked(eyeContactLookAway);
    eyeContactLookAwayCheck_->blockSignals(false);
  }

  if (denoiseCheck_) {
    denoiseCheck_->blockSignals(true);
    denoiseCheck_->setChecked(denoise);
    denoiseCheck_->blockSignals(false);
  }
  if (denoiseStrengthSlider_) {
    denoiseStrengthSlider_->blockSignals(true);
    denoiseStrengthSlider_->setValue(
        std::max(0, std::min(100, denoiseStrength)));
    denoiseStrengthSlider_->blockSignals(false);
  }
  if (denoiseStrengthValue_ && denoiseStrengthSlider_) {
    denoiseStrengthValue_->setText(
        QString::number(denoiseStrengthSlider_->value()) + "%");
  }

  if (virtualKeyLightCheck_) {
    virtualKeyLightCheck_->blockSignals(true);
    virtualKeyLightCheck_->setChecked(virtualKeyLight);
    virtualKeyLightCheck_->blockSignals(false);
  }
  if (virtualKeyLightIntensitySpin_) {
    virtualKeyLightIntensitySpin_->blockSignals(true);
    virtualKeyLightIntensitySpin_->setValue(
        std::max(0, std::min(100, virtualKeyLightIntensity)));
    virtualKeyLightIntensitySpin_->blockSignals(false);
  }
  if (virtualKeyLightTempCombo_) {
    virtualKeyLightTempCombo_->blockSignals(true);
    const QString key =
        virtualKeyLightTemp.isEmpty() ? "neutral" : virtualKeyLightTemp;
    const int idx = virtualKeyLightTempCombo_->findData(key);
    if (idx >= 0)
      virtualKeyLightTempCombo_->setCurrentIndex(idx);
    virtualKeyLightTempCombo_->blockSignals(false);
  }
  if (virtualKeyLightPanSpin_) {
    virtualKeyLightPanSpin_->blockSignals(true);
    virtualKeyLightPanSpin_->setValue(
        std::max(-180, std::min(180, virtualKeyLightPan)));
    virtualKeyLightPanSpin_->blockSignals(false);
  }
  if (virtualKeyLightHdriEdit_) {
    virtualKeyLightHdriEdit_->blockSignals(true);
    virtualKeyLightHdriEdit_->setText(virtualKeyLightHdri);
    virtualKeyLightHdriEdit_->blockSignals(false);
  }

  if (vignetteCheck_) {
    vignetteCheck_->blockSignals(true);
    vignetteCheck_->setChecked(vignette);
    vignetteCheck_->blockSignals(false);
  }
  if (vignetteIntensitySlider_) {
    vignetteIntensitySlider_->blockSignals(true);
    vignetteIntensitySlider_->setValue(
        std::max(0, std::min(100, vignetteIntensity)));
    vignetteIntensitySlider_->blockSignals(false);
  }
  if (vignetteIntensityValue_ && vignetteIntensitySlider_) {
    vignetteIntensityValue_->setText(
        QString::number(vignetteIntensitySlider_->value()) + "%");
  }
  if (vignetteCenterOnFaceCheck_) {
    vignetteCenterOnFaceCheck_->blockSignals(true);
    vignetteCenterOnFaceCheck_->setChecked(vignetteCenterOnFace);
    vignetteCenterOnFaceCheck_->blockSignals(false);
  }

  // Enable per-effect parameter controls.
  const QString bgMode =
      backgroundCombo_ ? backgroundCombo_->currentData().toString() : QString();
  UpdateBackgroundModeOptionVisibility();
  if (backgroundStrengthSpin_)
    backgroundStrengthSpin_->setEnabled(bgMode == QStringLiteral("blur"));
  if (backgroundRemoveColorEdit_)
    backgroundRemoveColorEdit_->setEnabled(bgMode == QStringLiteral("remove"));
  if (backgroundReplaceImageEdit_)
    backgroundReplaceImageEdit_->setEnabled(bgMode == QStringLiteral("replace"));
  if (browseReplaceImageBtn_)
    browseReplaceImageBtn_->setEnabled(bgMode == QStringLiteral("replace"));

  if (autoFrameZoomSlider_ && autoFrameCheck_)
    autoFrameZoomSlider_->setEnabled(autoFrameCheck_->isChecked());
  if (eyeContactStrengthSlider_ && eyeContactCheck_)
    eyeContactStrengthSlider_->setEnabled(eyeContactCheck_->isChecked());
  if (eyeContactLookAwayCheck_ && eyeContactCheck_)
    eyeContactLookAwayCheck_->setEnabled(eyeContactCheck_->isChecked());
  if (denoiseStrengthSlider_ && denoiseCheck_)
    denoiseStrengthSlider_->setEnabled(denoiseCheck_->isChecked());

  return true;
}

void VideoPage::ResyncControlsFromCachedStatus(bool force) {
  if (!force && !effectsWriteGuard_.ShouldApplyRoutineStatus())
    return;
  (void)SyncFromCachedDaemonStatus();
}

void VideoPage::ScheduleDaemonVideoEffectsWrite() {
  if (!effectsWriteDebounceTimer_)
    return;
  effectsWriteGuard_.MarkPending();
  effectsWriteDebounceTimer_->start();
}

void VideoPage::MarkSetupControlsEdited() { setupControlsDirty_ = true; }

studiocast::video::effects::BroadcastCameraEffects
VideoPage::BuildCandidateEffectsFromUi() const {
  auto candidate = effects_;

  candidate.engine =
      studiocast::video::effects::EffectsEnginePreference::auto_select;
  if (engineCombo_) {
    const QString s = engineCombo_->currentData().toString();
    studiocast::video::effects::EffectsEnginePreference ep = candidate.engine;
    if (studiocast::video::effects::ParseEffectsEnginePreference(
            s.toStdString(), &ep)) {
      candidate.engine = ep;
    }
  }
  if (mirrorCheck_)
    candidate.mirror = mirrorCheck_->isChecked();

  const QString bg =
      backgroundCombo_ ? backgroundCombo_->currentData().toString() : QString();
  const bool bgIsAutoFrame = (bg == "auto_frame");

  candidate.auto_frame.enabled =
      (autoFrameCheck_ && autoFrameCheck_->isChecked()) || bgIsAutoFrame;
  if (autoFrameZoomSlider_)
    candidate.auto_frame.strength = autoFrameZoomSlider_->value();
  if (autoFrameModelCombo_ && autoFrameModelCombo_->count() > 0) {
    candidate.auto_frame.model_id =
        autoFrameModelCombo_->currentData().toString().toStdString();
  }

  if (bgIsAutoFrame) {
    candidate.virtual_background.mode =
        studiocast::video::effects::VirtualBackgroundMode::none;
  } else {
    studiocast::video::effects::VirtualBackgroundMode m =
        studiocast::video::effects::VirtualBackgroundMode::none;
    (void)studiocast::video::effects::ParseVirtualBackgroundMode(
        bg.toStdString(), &m);
    candidate.virtual_background.mode = m;
  }
  if (vbModelCombo_ && vbModelCombo_->count() > 0) {
    candidate.virtual_background.model_id =
        vbModelCombo_->currentData().toString().toStdString();
  }
  if (backgroundStrengthSpin_)
    candidate.virtual_background.strength = backgroundStrengthSpin_->value();
  if (backgroundRemoveColorEdit_) {
    candidate.virtual_background.remove_color =
        backgroundRemoveColorEdit_->text().trimmed().toStdString();
  }
  if (backgroundReplaceImageEdit_) {
    candidate.virtual_background.replace_path =
        backgroundReplaceImageEdit_->text().trimmed().toStdString();
  }

  if (eyeContactCheck_)
    candidate.eye_contact.enabled = eyeContactCheck_->isChecked();
  if (eyeContactStrengthSlider_)
    candidate.eye_contact.strength = eyeContactStrengthSlider_->value();
  if (eyeContactLookAwayCheck_) {
    candidate.eye_contact.look_away_enabled =
        eyeContactLookAwayCheck_->isChecked();
  }
  if (eyeContactModelCombo_ && eyeContactModelCombo_->count() > 0) {
    candidate.eye_contact.model_id =
        eyeContactModelCombo_->currentData().toString().toStdString();
  }

  if (denoiseCheck_)
    candidate.video_noise_removal.enabled = denoiseCheck_->isChecked();
  if (denoiseStrengthSlider_)
    candidate.video_noise_removal.strength = denoiseStrengthSlider_->value();
  if (denoiseModelCombo_ && denoiseModelCombo_->count() > 0) {
    candidate.video_noise_removal.model_id =
        denoiseModelCombo_->currentData().toString().toStdString();
  }

  if (virtualKeyLightCheck_)
    candidate.virtual_key_light.enabled = virtualKeyLightCheck_->isChecked();
  if (virtualKeyLightIntensitySpin_) {
    candidate.virtual_key_light.intensity =
        virtualKeyLightIntensitySpin_->value();
  }
  if (virtualKeyLightTempCombo_) {
    const QString t = virtualKeyLightTempCombo_->currentData().toString();
    if (t == "warm")
      candidate.virtual_key_light.temperature_preset = 1;
    else if (t == "cool")
      candidate.virtual_key_light.temperature_preset = 2;
    else
      candidate.virtual_key_light.temperature_preset = 0;
  }
  if (virtualKeyLightPanSpin_) {
    candidate.virtual_key_light.direction_pan_degrees =
        virtualKeyLightPanSpin_->value();
  }
  if (virtualKeyLightHdriEdit_) {
    candidate.virtual_key_light.hdri_path =
        virtualKeyLightHdriEdit_->text().trimmed().toStdString();
  }

  if (vignetteCheck_)
    candidate.vignette.enabled = vignetteCheck_->isChecked();
  if (vignetteIntensitySlider_)
    candidate.vignette.intensity = vignetteIntensitySlider_->value();
  if (vignetteCenterOnFaceCheck_) {
    candidate.vignette.center_on_tracked_face =
        vignetteCenterOnFaceCheck_->isChecked();
  }

  return candidate;
}

bool VideoPage::SendDaemonVideoConfig() {
  const QString inDev = inputCombo_->currentData().toString();
  const QString selectedOutDev = outputCombo_->currentData().toString();
  QString outDev = selectedOutDev;

  if (outDev.isEmpty()) {
    ShowError("Start failed",
              "Virtual camera is missing.\n\nOpen Diagnostics for the setup "
              "command, or open Support for technical details.");
    return false;
  }

  const bool explicitInput = !inDev.isEmpty() && inDev != "auto";

  if (explicitInput && outDev == "auto" &&
      outputCombo_->findData(inDev) >= 0) {
    bool resolvedDifferentOutput = false;
    for (int i = 0; i < outputCombo_->count(); ++i) {
      const QString candidate = outputCombo_->itemData(i).toString();
      if (!candidate.isEmpty() && candidate != "auto" && candidate != inDev) {
        outDev = candidate;
        resolvedDifferentOutput = true;
        break;
      }
    }
    if (!resolvedDifferentOutput) {
      ShowError("Start failed",
                "The selected input is also the only virtual camera output.\n\n"
                "Choose a different input camera or create a separate virtual "
                "camera.");
      return false;
    }
  }

  if (explicitInput && !outDev.isEmpty() && outDev != "auto" &&
      inDev == outDev) {
    ShowError("Start failed",
              "The input camera and virtual camera cannot be the same device.\n\n"
              "Choose a physical/readable input device and a different "
              "virtual camera.");
    return false;
  }

  std::ostringstream req;
  req << "SET_VIDEO_CONFIG";
  req << " input=" << (inDev.isEmpty() ? "auto" : inDev.toStdString());
  req << " output=" << (outDev.isEmpty() ? "auto" : outDev.toStdString());
  req << " width=" << widthSpin_->value();
  req << " height=" << heightSpin_->value();
  req << " fps=" << fpsSpin_->value();
  req << " output_format="
      << (outputFormatCombo_
              ? outputFormatCombo_->currentData().toString().toStdString()
              : std::string("rgb24"));

  QString err;
  if (!DaemonRequest(req.str(), nullptr, &err)) {
    if (statusText_) {
      SetPlainTextPreservingScroll(
          statusText_,
          QStringLiteral("Camera settings save failed:\n%1").arg(err));
    }
    ShowError("Start failed",
              "Camera settings were not saved.\n\nStudioCast background "
              "service is unavailable. Open Support for technical details.");
    return false;
  }

  setupControlsDirty_ = false;
  emit StatusRefreshRequested();
  return true;
}

bool VideoPage::SendDaemonComputeBackendPreference() {
  if (!computeBackendCombo_)
    return false;

  const QString backend = computeBackendCombo_->currentData().toString();
  std::ostringstream req;
  req << "SET_VIDEO_CONFIG compute_backend=" << backend.toStdString();

  QString err;
  if (!DaemonRequest(req.str(), nullptr, &err)) {
    if (statusText_) {
      SetPlainTextPreservingScroll(
          statusText_,
          QStringLiteral("Compute backend save failed:\n%1").arg(err));
    }
    ShowError("Compute backend update failed",
              QStringLiteral("Camera compute preference was not saved.\n\nOpen "
                             "Support for technical details."));
    emit StatusRefreshRequested();
    return false;
  }

  emit StatusRefreshRequested();
  return true;
}

bool VideoPage::SendDaemonVideoEffects() {
  if (effectsWriteDebounceTimer_ && effectsWriteDebounceTimer_->isActive())
    effectsWriteDebounceTimer_->stop();

  const auto candidate = BuildCandidateEffectsFromUi();
  const std::string json =
      studiocast::video::BroadcastCameraEffectsContractToJson(candidate);
  const std::string req = std::string("SET_VIDEO_EFFECTS_JSON ") + json;

  QString err;
  if (!DaemonRequest(req, nullptr, &err)) {
    effectsWriteGuard_.MarkWriteRejected();
    if (statusText_) {
      SetPlainTextPreservingScroll(
          statusText_,
          QStringLiteral("Camera effects save failed:\n%1").arg(err));
    }
    ShowError("Effects update failed",
              QStringLiteral("Camera effects were not saved.\n\nOpen Support "
                             "for technical details."));
    ResyncControlsFromCachedStatus(/*force=*/true);
    emit StatusRefreshRequested();
    return false;
  }

  effects_ = candidate;
  effectsWriteGuard_.MarkWriteAccepted();
  emit StatusRefreshRequested();
  return true;
}

bool VideoPage::SendDaemonEnabled(bool enabled) {
  std::string req = std::string("SET_ENABLED enabled=") + (enabled ? "1" : "0");
  QString err;
  if (!DaemonRequest(req, nullptr, &err)) {
    if (statusText_) {
      SetPlainTextPreservingScroll(
          statusText_,
          QStringLiteral("Camera start/stop failed:\n%1").arg(err));
    }
    ShowError("Start/Stop failed",
              QStringLiteral("Camera state was not changed.\n\nOpen Support "
                             "for technical details."));
    return false;
  }
  emit StatusRefreshRequested();
  return true;
}

void VideoPage::OnEnginePreferenceChanged(int /*index*/) {
  UpdateUiEnabled();
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnComputeBackendChanged(int /*index*/) {
  UpdateUiEnabled();
  (void)SendDaemonComputeBackendPreference();
}

void VideoPage::OnMirrorToggled(bool checked) {
  (void)checked;
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnBackgroundChanged(int /*index*/) {
  // Enable per-effect parameter controls.
  const QString bg =
      backgroundCombo_ ? backgroundCombo_->currentData().toString() : QString();
  UpdateBackgroundModeOptionVisibility();
  if (backgroundStrengthSpin_)
    backgroundStrengthSpin_->setEnabled(bg == QStringLiteral("blur"));
  if (backgroundRemoveColorEdit_)
    backgroundRemoveColorEdit_->setEnabled(bg == QStringLiteral("remove"));
  if (backgroundReplaceImageEdit_)
    backgroundReplaceImageEdit_->setEnabled(bg == QStringLiteral("replace"));
  if (browseReplaceImageBtn_)
    browseReplaceImageBtn_->setEnabled(bg == QStringLiteral("replace"));
  UpdateUiEnabled();
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnVbModelChanged(int /*index*/) {
  if (!vbModelCombo_ || vbModelCombo_->count() <= 0)
    return;
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnAutoFrameModelChanged(int /*index*/) {
  if (!autoFrameModelCombo_ || autoFrameModelCombo_->count() <= 0)
    return;
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnEyeContactModelChanged(int /*index*/) {
  if (!eyeContactModelCombo_ || eyeContactModelCombo_->count() <= 0)
    return;
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnDenoiseModelChanged(int /*index*/) {
  if (!denoiseModelCombo_ || denoiseModelCombo_->count() <= 0)
    return;
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnBackgroundStrengthChanged(int /*value*/) {
  ScheduleDaemonVideoEffectsWrite();
}

void VideoPage::OnBackgroundRemoveColorChanged() {
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnBackgroundReplaceImageChanged() {
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnBrowseReplaceImage() {
  const QString file = QFileDialog::getOpenFileName(
      this, "Select background image", QString(),
      "PPM (P6) Images (*.ppm *.PPM);;All files (*)");
  if (file.isEmpty())
    return;
  if (backgroundReplaceImageEdit_)
    backgroundReplaceImageEdit_->setText(file);
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnAutoFrameToggled(bool /*checked*/) {
  UpdateUiEnabled();
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnAutoFrameZoomChanged(int value) {
  if (autoFrameZoomValue_)
    autoFrameZoomValue_->setText(QString::number(value) + "%");
  ScheduleDaemonVideoEffectsWrite();
}

void VideoPage::OnEyeContactToggled(bool /*checked*/) {
  UpdateUiEnabled();
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnEyeContactStrengthChanged(int value) {
  if (eyeContactStrengthValue_)
    eyeContactStrengthValue_->setText(QString::number(value) + "%");
  ScheduleDaemonVideoEffectsWrite();
}

void VideoPage::OnEyeContactLookAwayToggled(bool /*checked*/) {
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnDenoiseToggled(bool /*checked*/) {
  UpdateUiEnabled();
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnDenoiseStrengthChanged(int value) {
  if (denoiseStrengthValue_)
    denoiseStrengthValue_->setText(QString::number(value) + "%");
  ScheduleDaemonVideoEffectsWrite();
}

void VideoPage::OnOpenInstallHints() {
  studiocast::video::effects::EffectsEnginePreference pref =
      studiocast::video::effects::EffectsEnginePreference::auto_select;
  if (engineCombo_) {
    const QString s = engineCombo_->currentData().toString();
    (void)studiocast::video::effects::ParseEffectsEnginePreference(
        s.toStdString(), &pref);
  }

  const auto resolveProgram = [&](const char *exeName) -> QString {
    QString program = QCoreApplication::applicationDirPath() + "/" + exeName;
    if (QFileInfo::exists(program))
      return program;
    return QString::fromUtf8(exeName);
  };

  QString title;
  QVector<QPair<QString, QString>> commands;
  if (pref == studiocast::video::effects::EffectsEnginePreference::open_cuda) {
    title = "Open Source install hints";
    commands.push_back(
        {resolveProgram("studiocast-open"), QStringLiteral("studiocast-open")});
  } else if (pref ==
             studiocast::video::effects::EffectsEnginePreference::maxine) {
    title = "Maxine install hints";
    commands.push_back({resolveProgram("studiocast-maxine"),
                        QStringLiteral("studiocast-maxine")});
  } else {
    title = "Engine install hints";
    commands.push_back({resolveProgram("studiocast-maxine"),
                        QStringLiteral("studiocast-maxine")});
    commands.push_back(
        {resolveProgram("studiocast-open"), QStringLiteral("studiocast-open")});
  }

  struct HintsState {
    QString title;
    QStringList sections;
    int remaining = 0;
  };

  auto state = std::make_shared<HintsState>();
  state->title = title;
  state->remaining = static_cast<int>(commands.size());

  if (openInstallHintsBtn_)
    openInstallHintsBtn_->setEnabled(false);

  const auto showIfDone = [this, state] {
    if (state->remaining > 0)
      return;
    if (openInstallHintsBtn_)
      openInstallHintsBtn_->setEnabled(true);
    QMessageBox mb(this);
    mb.setWindowTitle(state->title);
    mb.setText("See details.");
    mb.setDetailedText(state->sections.join(QStringLiteral("\n\n")).trimmed());
    mb.exec();
  };

  for (const auto &command : commands) {
    auto *process = new QProcess(this);
    process->setProgram(command.first);
    process->setArguments({QStringLiteral("install-hints")});

    auto *timeout = new QTimer(process);
    timeout->setSingleShot(true);
    timeout->setInterval(15000);

    const QString label = command.second;
    const auto finishProcess = [state, showIfDone, process, timeout,
                                label](const QString &extraError) {
      if (process->property("studiocastDone").toBool())
        return;
      process->setProperty("studiocastDone", true);
      timeout->stop();

      QString out = QString::fromUtf8(process->readAllStandardOutput());
      QString err = QString::fromUtf8(process->readAllStandardError());
      if (!extraError.trimmed().isEmpty()) {
        if (!err.isEmpty())
          err += QChar('\n');
        err += extraError.trimmed();
      }
      const QString text =
          (out + (err.isEmpty() ? QString() : (QStringLiteral("\n") + err)))
              .trimmed();
      state->sections.push_back(label + QStringLiteral(":\n") +
                                (text.isEmpty() ? QStringLiteral("(no output)")
                                                : text));
      --state->remaining;
      process->deleteLater();
      showIfDone();
    };

    connect(timeout, &QTimer::timeout, this, [process] {
      process->setProperty("studiocastTimedOut", true);
      process->kill();
    });
    connect(process,
            QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
            [finishProcess, process](int /*exitCode*/,
                                     QProcess::ExitStatus /*status*/) {
              const QString extra =
                  process->property("studiocastTimedOut").toBool()
                      ? QStringLiteral("Timed out after 15 seconds.")
                      : QString();
              finishProcess(extra);
            });
    connect(process, &QProcess::errorOccurred, this,
            [finishProcess, process](QProcess::ProcessError error) {
              if (error == QProcess::FailedToStart)
                finishProcess(process->errorString());
            });

    process->start();
    timeout->start();
  }
}

void VideoPage::OnVirtualKeyLightToggled(bool /*checked*/) {
  UpdateUiEnabled();
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnVirtualKeyLightIntensityChanged(int /*value*/) {
  ScheduleDaemonVideoEffectsWrite();
}

void VideoPage::OnVirtualKeyLightTemperatureChanged(int /*index*/) {
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnVirtualKeyLightPanChanged(int /*value*/) {
  ScheduleDaemonVideoEffectsWrite();
}

void VideoPage::OnVirtualKeyLightHdriChanged() {
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnBrowseVirtualKeyLightHdri() {
  const QString file =
      QFileDialog::getOpenFileName(this, "Select HDRI (.hdr/.exr)", QString(),
                                   "HDRI (*.hdr *.exr);;All files (*)");
  if (file.isEmpty())
    return;
  if (virtualKeyLightHdriEdit_)
    virtualKeyLightHdriEdit_->setText(file);
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnVignetteToggled(bool checked) {
  (void)checked;
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnVignetteIntensityChanged(int value) {
  if (vignetteIntensityValue_)
    vignetteIntensityValue_->setText(QString::number(value) + "%");
  ScheduleDaemonVideoEffectsWrite();
}

void VideoPage::OnVignetteCenterOnFaceToggled(bool checked) {
  (void)checked;
  (void)SendDaemonVideoEffects();
}

void VideoPage::OnStart() {
  if (DebugGuiPreview()) {
    GuiPreviewDbg("OnStart clicked: sending config/effects and enabling video");
  }

  if (!SendDaemonVideoConfig()) {
    if (DebugGuiPreview()) {
      GuiPreviewDbg("OnStart: SendDaemonVideoConfig failed");
    }
    UpdateStatusText();
    UpdateUiEnabled();
    return;
  }

  if (!SendDaemonVideoEffects()) {
    if (DebugGuiPreview()) {
      GuiPreviewDbg("OnStart: SendDaemonVideoEffects failed");
    }
    UpdateStatusText();
    UpdateUiEnabled();
    return;
  }

  if (!SendDaemonEnabled(true)) {
    if (DebugGuiPreview()) {
      GuiPreviewDbg("OnStart: SendDaemonEnabled(true) failed");
    }
    UpdateStatusText();
    UpdateUiEnabled();
    return;
  }

  if (DebugGuiPreview()) {
    GuiPreviewDbg("OnStart: daemon enabled=true");
  }

  UpdateStatusText();
  UpdateUiEnabled();
}

void VideoPage::OnStop() {
  if (DebugGuiPreview()) {
    GuiPreviewDbg("OnStop clicked: stopping preview and disabling video");
  }
  StopPreview();
  (void)SendDaemonEnabled(false);
  UpdateStatusText();
  UpdateUiEnabled();
}

void VideoPage::OnPreviewToggled(bool checked) {
  if (!checked) {
    StopPreview();
    return;
  }

  DaemonVideoStatus st;
  if (daemonReachable_ && !daemonLastStatusJson_.empty()) {
    QString perr;
    (void)ParseDaemonStatusJson(daemonLastStatusJson_, &st, &perr);
  }

  if (!daemonReachable_ || !st.enabled) {
    if (preview_)
      preview_->SetStatusText("Preview available when camera is started");
    return;
  }

  StartPreview();
}

void VideoPage::StartPreview() {
  if (DebugGuiPreview()) {
    GuiPreviewDbg("StartPreview: begin");
  }
  StopPreview();

  // Determine which device to open for preview.
  QString outDev = outputCombo_->currentData().toString();
  int wantW = widthSpin_->value();
  int wantH = heightSpin_->value();
  int wantFps = fpsSpin_->value();
  std::optional<studiocast::video::CapturePixelFormat> wantFmt;

  // Ask daemon for the resolved output device and negotiated output format.
  //
  // Using the daemon's negotiated format avoids capture-side renegotiation on
  // v4l2loopback devices (which can destabilize the producer and cause
  // start/stop thrashing).
  {
    std::string json;
    QString err;
    if (DaemonRequest("GET_STATUS", &json, &err)) {
      DaemonVideoStatus st;
      QString perr;
      if (ParseDaemonStatusJson(json, &st, &perr)) {
        if (DebugGuiPreview()) {
          std::ostringstream oss;
          oss << "StartPreview: GET_STATUS enabled=" << (st.enabled ? 1 : 0)
              << " consumer_count=" << st.consumer_count << " output_device='"
              << st.output_device.toStdString() << "'"
              << " output_fmt=" << st.output_format.pixfmt.toStdString() << " "
              << st.output_format.width << "x" << st.output_format.height
              << " fps=" << st.output_format.fps;
          GuiPreviewDbg(oss.str());
        }
        if ((outDev.isEmpty() || outDev == "auto") &&
            !st.output_device.isEmpty()) {
          outDev = st.output_device;
        }

        const bool outMatches =
            (!st.output_device.isEmpty() && outDev == st.output_device);
        if (outMatches && !st.output_format.pixfmt.isEmpty() &&
            st.output_format.width > 0 && st.output_format.height > 0) {
          wantW = st.output_format.width;
          wantH = st.output_format.height;
          double fpsFromStatus = st.output_format.fps;
          if (st.output_format.fps_num > 0 && st.output_format.fps_den > 0) {
            // V4L2 reports time-per-frame (numerator/denominator). Convert to
            // FPS.
            fpsFromStatus = static_cast<double>(st.output_format.fps_den) /
                            static_cast<double>(st.output_format.fps_num);
          }
          if (fpsFromStatus > 0.0) {
            wantFps = std::clamp(
                static_cast<int>(std::floor(fpsFromStatus + 0.5)), 1, 240);
          }

          if (st.output_format.pixfmt == "RGB3") {
            wantFmt = studiocast::video::CapturePixelFormat::rgb24;
          } else if (st.output_format.pixfmt == "YUYV") {
            wantFmt = studiocast::video::CapturePixelFormat::yuyv;
          }
        }
      }
    }
  }

  if (outDev.isEmpty() || outDev == "auto") {
    if (DebugGuiPreview()) {
      GuiPreviewDbg(
          "StartPreview: no output device selected (outDev empty/auto)");
    }
    preview_->SetStatusText("Preview unavailable (no output device selected)");
    return;
  }

  const auto openFmt = [&](studiocast::video::CapturePixelFormat fmt,
                           std::string *outErr) -> bool {
    return previewCapture_.Open(outDev.toStdString(), wantW, wantH, wantFps,
                                fmt, false, outErr);
  };

  // Prefer the daemon's negotiated output format when available.
  const auto firstFmt =
      wantFmt.value_or(studiocast::video::CapturePixelFormat::rgb24);
  const auto secondFmt =
      (firstFmt == studiocast::video::CapturePixelFormat::rgb24)
          ? studiocast::video::CapturePixelFormat::yuyv
          : studiocast::video::CapturePixelFormat::rgb24;

  std::string err;
  if (!openFmt(firstFmt, &err)) {
    std::string err2;
    if (!openFmt(secondFmt, &err2)) {
      if (DebugGuiPreview()) {
        GuiPreviewDbg(
            std::string("StartPreview: previewCapture_.Open failed: ") + err2);
      }

      // Prevent UpdateUiEnabled() from immediately retrying in a tight loop.
      previewAutoRetryFailures_ = std::min(previewAutoRetryFailures_ + 1, 20);
      const auto now = std::chrono::steady_clock::now();
      const int backoffMs = std::min(2000, 250 * previewAutoRetryFailures_);
      previewAutoRetryAfter_ = now + std::chrono::milliseconds(backoffMs);

      preview_->SetStatusText("Preview open failed:\n" +
                              QString::fromStdString(err2));
      return;
    }
  }
  // Preview open succeeded; clear any prior auto-retry backoff.
  previewAutoRetryFailures_ = 0;
  previewAutoRetryAfter_ = std::chrono::steady_clock::time_point{};

  const auto fmt = previewCapture_.Actual();
  previewW_ = fmt.width;
  previewH_ = fmt.height;
  previewBpl_ = previewW_ * 3;
  previewRgb_.assign(static_cast<std::size_t>(previewBpl_ * previewH_), 0);

  previewTimer_->start();
  preview_->SetStatusText("Preview starting...");

  if (DebugGuiPreview()) {
    const auto a = previewCapture_.Actual();
    std::ostringstream oss;
    oss << "StartPreview: Open OK dev='" << outDev.toStdString() << "'"
        << " fmt=" << a.pixfmt << " " << a.width << "x" << a.height
        << " fps=" << a.fps << " bpl=" << a.bytes_per_line
        << " size=" << a.size_image;
    GuiPreviewDbg(oss.str());
  }
}

void VideoPage::StopPreview() {
  if (DebugGuiPreview()) {
    GuiPreviewDbg(
        std::string("StopPreview: timer=") +
        ((previewTimer_ && previewTimer_->isActive()) ? "active" : "stopped") +
        " capture_open=" +
        (previewCapture_.IsOpen() ? std::string("yes") : std::string("no")));
  }
  if (previewTimer_)
    previewTimer_->stop();
  if (previewCapture_.IsOpen())
    previewCapture_.Close();
  previewRgb_.clear();
  previewW_ = previewH_ = previewBpl_ = 0;

  if (preview_)
    preview_->SetStatusText("Preview stopped");
}

void VideoPage::OnPreviewTick() {
  if (!previewCapture_.IsOpen() || previewRgb_.empty())
    return;

  studiocast::video::CapturedFrameView f;
  std::string err;
  if (!previewCapture_.AcquireFrame(&f, 0, &err)) {
    // Normal: no frame ready yet.
    return;
  }

  const auto fmt = previewCapture_.Actual();

  if (fmt.format == studiocast::video::CapturePixelFormat::rgb24) {
    const std::size_t srcStride = fmt.bytes_per_line;
    const std::size_t dstStride = static_cast<std::size_t>(previewBpl_);
    const auto *src = f.data;
    auto *dst = previewRgb_.data();
    for (int y = 0; y < previewH_; ++y) {
      std::memcpy(dst + static_cast<std::size_t>(y) * dstStride,
                  src + static_cast<std::size_t>(y) * srcStride,
                  std::min(dstStride, srcStride));
    }
  } else {
    studiocast::video::YuyvToRgb24(f.data, previewW_, previewH_,
                                   fmt.bytes_per_line, previewRgb_.data(),
                                   static_cast<std::size_t>(previewBpl_));
  }

  std::string rerr;
  (void)previewCapture_.ReleaseFrame(f, &rerr);

  QImage img(previewRgb_.data(), previewW_, previewH_, previewBpl_,
             QImage::Format_RGB888);
  preview_->SetFrame(img);
}

void VideoPage::UpdateUiEnabled() {
  // Query daemon status (best-effort) to determine whether controls should be
  // editable.
  DaemonVideoStatus st;
  if (daemonReachable_ && !daemonLastStatusJson_.empty()) {
    QString perr;
    if (!ParseDaemonStatusJson(daemonLastStatusJson_, &st, &perr)) {
      daemonReachable_ = false;
      daemonLastStatusJson_.clear();
    }
  }

  const auto currentEnginePref =
      [&]() -> studiocast::video::effects::EffectsEnginePreference {
    studiocast::video::effects::EffectsEnginePreference ep =
        studiocast::video::effects::EffectsEnginePreference::auto_select;
    if (!engineCombo_)
      return ep;
    const QString s = engineCombo_->currentData().toString();
    (void)studiocast::video::effects::ParseEffectsEnginePreference(
        s.toStdString(), &ep);
    return ep;
  };

  const bool enabled = daemonReachable_ ? st.enabled : false;
  const bool maxineSupported = daemonReachable_ && st.maxine_supported;
  const bool openCudaSupported =
      daemonReachable_ && st.open_cuda_present && st.open_cuda_ok;
  const auto enginePref = currentEnginePref();

  refreshBtn_->setEnabled(!enabled);
  copyCmdBtn_->setEnabled(!suggestedCmd_.isEmpty());

  const bool outSelectable = outputCombo_->count() > 0 &&
                             !outputCombo_->itemData(0).toString().isEmpty();

  inputCombo_->setEnabled(!enabled);
  outputCombo_->setEnabled(!enabled && outSelectable);

  if (outputFormatCombo_)
    outputFormatCombo_->setEnabled(!enabled);
  widthSpin_->setEnabled(!enabled);
  heightSpin_->setEnabled(!enabled);
  fpsSpin_->setEnabled(!enabled);
  if (setupLockLabel_)
    setupLockLabel_->setVisible(enabled);

  if (engineCombo_) {
    engineCombo_->setEnabled(daemonReachable_);
  }
  if (computeBackendCombo_) {
    computeBackendCombo_->setEnabled(daemonReachable_);
  }
  if (effectEngineValue_) {
    if (!daemonReachable_) {
      effectEngineValue_->setText("—");
      effectEngineValue_->setToolTip(QString());
    } else if (!enabled) {
      effectEngineValue_->setText(QStringLiteral("Off"));
      effectEngineValue_->setToolTip(QString());
    } else if (!st.pipeline_running) {
      effectEngineValue_->setText(QStringLiteral("Starting…"));
      effectEngineValue_->setToolTip(QString());
    } else if (st.effects_backends.trimmed().isEmpty()) {
      effectEngineValue_->setText(QStringLiteral("Pass-through"));
      effectEngineValue_->setToolTip(QString());
    } else {
      const QString raw = st.effects_backends.trimmed();
      const QString summary = SummarizeEffectsBackends(raw);
      effectEngineValue_->setText(summary.isEmpty() ? raw : summary);
      effectEngineValue_->setToolTip(raw);
    }
  }
  if (computeBackendValue_) {
    if (!daemonReachable_) {
      computeBackendValue_->setText("—");
      computeBackendValue_->setToolTip(QString());
    } else if (!enabled) {
      computeBackendValue_->setText(QStringLiteral("Off"));
      computeBackendValue_->setToolTip(QString());
    } else if (!st.pipeline_running && st.pipeline_starting) {
      computeBackendValue_->setText(QStringLiteral("Starting…"));
      computeBackendValue_->setToolTip(QString());
    } else {
      const QString active =
          st.compute_active_backend.isEmpty() ? QStringLiteral("cpu")
                                              : st.compute_active_backend;
      const QString resolved =
          st.compute_resolved_backend.isEmpty() ? active
                                                : st.compute_resolved_backend;
      QString text = active;
      if (active != resolved)
        text += QStringLiteral(" (%1)").arg(resolved);
      computeBackendValue_->setText(text);

      QStringList detail;
      detail << QStringLiteral("Preference: %1")
                    .arg(st.compute_preference.isEmpty()
                             ? QStringLiteral("auto")
                             : st.compute_preference);
      detail << QStringLiteral("Resolved: %1").arg(resolved);
      detail << QStringLiteral("Active: %1").arg(active);
      if (!st.compute_fallback_reason.isEmpty())
        detail << st.compute_fallback_reason;
      if (!st.compute_degraded_reason.isEmpty() &&
          st.compute_degraded_reason != st.compute_fallback_reason)
        detail << st.compute_degraded_reason;
      computeBackendValue_->setToolTip(detail.join(QStringLiteral("\n")));
    }
  }

  if (engineInfoBanner_) {
    if (!daemonReachable_) {
      engineInfoBanner_->setVisible(false);
      engineInfoBanner_->setToolTip(QString());
    } else {
      const QString full = SanitizeBackendNote(st.effects_note).trimmed();
      const QString first = FirstLine(full);
      engineInfoBanner_->setVisible(!first.isEmpty());
      engineInfoBanner_->setText(first);
      engineInfoBanner_->setToolTip(full);
    }
  }

  // Engine blocking banner + diagnostics (best-effort)
  if (maxineBanner_) {
    if (!daemonReachable_) {
      maxineBanner_->setVisible(false);
    } else {
      QString msg;
      bool show = false;

      const auto fmtMaxineBlocked = [&]() -> QString {
        QString s = "Maxine unavailable.";
        if (!st.maxine_blocked_reason.isEmpty()) {
          const QString reason =
              FormatMaxineReasonCode(st.maxine_blocked_reason);
          if (!reason.isEmpty())
            s += "\n" + reason;
        }
        return s.trimmed();
      };

      const auto fmtOpenCudaBlocked = [&]() -> QString {
        if (st.open_cuda_present && OpenCudaHasSetupBlocker(st))
          return OpenCudaSetupText(st, /*includeInstallHints=*/true);

        QString s = "Open Video / Open CUDA unavailable.";
        if (!st.open_cuda_present) {
          s += "\nStatus is not available.";
        } else if (!st.open_cuda_ok) {
          if (st.open_cuda_installed_models.isEmpty()) {
            s += "\nNo usable model packs were found.";
          }
        }

        if (!st.open_cuda_missing_models.isEmpty()) {
          s += QStringLiteral("\nMissing/invalid model pack(s): %1")
                   .arg(st.open_cuda_missing_models.size());
        }

        if (!st.open_cuda_install_hints.isEmpty()) {
          s += "\n\n";
          s += st.open_cuda_install_hints.join(QStringLiteral("\n"));
        }
        s += "\nRun: ./build/studiocast-open video-install-hints";
        return s.trimmed();
      };

      if (enginePref ==
          studiocast::video::effects::EffectsEnginePreference::maxine) {
        if (!st.maxine_supported) {
          msg = fmtMaxineBlocked();
          msg += "\n\nEffects disabled. Open Diagnostics for details.";
          show = true;
        }
      } else if (enginePref == studiocast::video::effects::
                                   EffectsEnginePreference::open_cuda) {
        if (!st.open_cuda_present || !st.open_cuda_ok) {
          msg = fmtOpenCudaBlocked();
          msg += "\n\nEffects disabled. Open Diagnostics for details.";
          show = true;
        }
      } else {
        // auto_select
        if (!st.maxine_supported &&
            !(st.open_cuda_present && st.open_cuda_ok)) {
          msg = "No backend is available.";
          msg += "\n\n" + fmtMaxineBlocked();
          msg += "\n\n" + fmtOpenCudaBlocked();
          show = true;
        }
      }

      maxineBanner_->setText(msg);
      maxineBanner_->setVisible(show);
      if (engineInfoBanner_ && show)
        engineInfoBanner_->setVisible(false);
    }
  }

  if (openInstallHintsBtn_) {
    if (enginePref ==
        studiocast::video::effects::EffectsEnginePreference::open_cuda) {
      openInstallHintsBtn_->setText("Open Source install hints");
    } else if (enginePref ==
               studiocast::video::effects::EffectsEnginePreference::maxine) {
      openInstallHintsBtn_->setText("Maxine install hints");
    } else {
      openInstallHintsBtn_->setText("Engine install hints");
    }
  }

  if (diagnosticsText_ && daemonReachable_ && !daemonLastStatusJson_.empty()) {
    QJsonObject root;
    QString perr;
    if (ParseJsonObject(daemonLastStatusJson_, &root, &perr)) {
      QJsonObject diag;
      if (enginePref ==
          studiocast::video::effects::EffectsEnginePreference::open_cuda) {
        diag = root.value("open_cuda").toObject();
        if (diag.isEmpty())
          diag = root.value("engines").toObject().value("open_cuda").toObject();
      } else if (enginePref ==
                 studiocast::video::effects::EffectsEnginePreference::maxine) {
        diag = root.value("maxine").toObject();
        if (diag.isEmpty())
          diag = root.value("engines").toObject().value("maxine").toObject();
      } else {
        diag = root.value("engines").toObject();
      }

      QString note;
      if (enginePref ==
              studiocast::video::effects::EffectsEnginePreference::open_cuda &&
          (!st.open_cuda_missing_models.isEmpty() ||
           OpenCudaHasSetupBlocker(st))) {
        if (OpenCudaHasSetupBlocker(st)) {
          note = QStringLiteral("NOTE: %1\n")
                     .arg(OpenCudaSetupText(st, /*includeInstallHints=*/false));
        } else {
          note = QStringLiteral(
              "NOTE: Some Open Video model packs are missing/invalid.\n");
          for (auto it = st.open_cuda_missing_models.begin();
               it != st.open_cuda_missing_models.end(); ++it) {
            note += QStringLiteral("- ") + it.key() + QStringLiteral(": ") +
                    it.value() + QChar('\n');
          }
        }
        note = note.trimmed();
        note += QStringLiteral("\n\n");
      }

      SetPlainTextPreservingScroll(
          diagnosticsText_,
          note + QString::fromUtf8(
                     QJsonDocument(diag).toJson(QJsonDocument::Indented)));
    } else {
      SetPlainTextPreservingScroll(
          diagnosticsText_, "(failed to parse status JSON)\n" + perr);
    }
  }

  // Per-effect availability comes from daemon status:
  //  - `maxine.available_effects` / `maxine.missing_effects` (MaxineManager)
  //  - `pipeline.effects_plan.disabled` (rule-based gating; single source of
  //  truth)
  // Plus static effect descriptors that tell us which effects depend on Maxine.
  auto effectAvailable = [&](const QString &id) -> bool {
    if (!daemonReachable_)
      return false;
    if (st.effects_plan_disabled.contains(id))
      return false;

    const auto kind = AvailabilityKindForEffectId(id);
    if (kind == EffectAvailabilityKind::always)
      return true;
    if (kind == EffectAvailabilityKind::gpu_utility) {
      // GPU utility effects require a CUDA-capable engine (Maxine or Open
      // CUDA).
      return maxineSupported || openCudaSupported;
    }

    // Engine-specific availability for effects that are not always-on.
    if (enginePref ==
        studiocast::video::effects::EffectsEnginePreference::maxine) {
      if (!maxineSupported)
        return false;
      if (st.maxine_available_effects.isEmpty() &&
          st.maxine_missing_effects.isEmpty()) {
        // Disable-by-default when daemon did not report availability.
        return false;
      }
      return st.maxine_available_effects.contains(id);
    }

    if (enginePref ==
        studiocast::video::effects::EffectsEnginePreference::open_cuda) {
      if (!(st.open_cuda_present && st.open_cuda_ok))
        return false;
      return st.open_cuda_available_effects.contains(id);
    }

    // auto_select: prefer Maxine if available, else Open Source.
    if (maxineSupported && st.maxine_available_effects.contains(id))
      return true;
    if (st.open_cuda_present && st.open_cuda_ok &&
        st.open_cuda_available_effects.contains(id))
      return true;
    return false;
  };

  auto effectUnavailableTooltip = [&](const QString &id) -> QString {
    if (!daemonReachable_)
      return "Background service unavailable.";

    if (st.effects_plan_disabled.contains(id)) {
      return st.effects_plan_disabled.value(id);
    }

    const auto kind = AvailabilityKindForEffectId(id);
    if (kind == EffectAvailabilityKind::always) {
      return "Effect is unavailable.";
    }

    if (kind == EffectAvailabilityKind::gpu_utility) {
      if (!(maxineSupported || openCudaSupported))
        return "GPU processing unavailable (no CUDA engine available).";
      return "GPU processing unavailable.";
    }

    if (enginePref ==
        studiocast::video::effects::EffectsEnginePreference::open_cuda) {
      if (!st.open_cuda_present) {
        return "Open Video / Open CUDA status is not available.";
      }
      if (!st.open_cuda_ok) {
        if (OpenCudaHasSetupBlocker(st))
          return OpenCudaSetupText(st, /*includeInstallHints=*/true);

        QStringList lines;
        lines << "Open Video / Open CUDA unavailable.";
        if (st.open_cuda_installed_models.isEmpty()) {
          lines << "No usable Open Video model packs were found.";
        }
        if (!st.open_cuda_missing_models.isEmpty()) {
          lines << "";
          lines << "Missing/invalid model packs:";
          for (auto it = st.open_cuda_missing_models.begin();
               it != st.open_cuda_missing_models.end(); ++it) {
            lines << QStringLiteral("- %1: %2").arg(it.key(), it.value());
          }
        }
        if (!st.open_cuda_install_hints.isEmpty()) {
          lines << "";
          lines << st.open_cuda_install_hints;
        }
        lines << "";
        lines << "Run: ./build/studiocast-open video-install-hints";
        return lines.join("\n");
      }
      if (st.open_cuda_blocked_effects.contains(id)) {
        const QString reason = st.open_cuda_blocked_effects.value(id);
        const QString code = ReasonCodeFromBlockedLine(reason);
        if (!OpenCudaSetupReasonText(code).isEmpty())
          return OpenCudaSetupText(st, /*includeInstallHints=*/true);
        return reason;
      }
      return "Effect is unavailable.";
    }

    // Maxine (or auto_select with Maxine-specific reason).
    if (!maxineSupported) {
      if (!st.maxine_blocked_details.isEmpty())
        return st.maxine_blocked_details.join("\n");
      if (!st.maxine_summary.isEmpty())
        return st.maxine_summary;
      if (!st.maxine_blocked_reason.isEmpty())
        return FormatMaxineReasonCode(st.maxine_blocked_reason);
      return "Maxine unavailable.";
    }

    if (st.maxine_missing_effects.contains(id)) {
      const auto reasons = st.maxine_missing_effects.value(id);
      if (!reasons.isEmpty()) {
        QStringList out;
        out.reserve(reasons.size());
        for (const auto &r : reasons)
          out.push_back(FormatMaxineReasonCode(r));
        return out.join("\n");
      }
      return "Effect is unavailable.";
    }

    if (st.maxine_available_effects.isEmpty() &&
        st.maxine_missing_effects.isEmpty()) {
      return "Effect availability is not available.";
    }
    return "Effect is unavailable.";
  };

  auto setAvail = [&](QWidget *w, bool avail, const QString &tooltip) {
    if (!w)
      return;
    w->setEnabled(avail);
    w->setToolTip(avail ? QString() : tooltip);
  };

  if (mirrorCheck_) {
    const bool mirrorAvailable = effectAvailable(QStringLiteral("mirror"));
    const bool mirrorOn = mirrorCheck_->isChecked();
    mirrorCheck_->setEnabled(daemonReachable_ && (mirrorAvailable || mirrorOn));
    if (!daemonReachable_) {
      mirrorCheck_->setToolTip(
          QStringLiteral("Background service unavailable."));
    } else if (st.effects_plan_disabled.contains(QStringLiteral("mirror"))) {
      mirrorCheck_->setToolTip(
          st.effects_plan_disabled.value(QStringLiteral("mirror")));
    } else {
      mirrorCheck_->setToolTip(QStringLiteral(
          "Saves the camera mirror setting."));
    }
  }

  // Virtual Background
  const QString vbMode =
      backgroundCombo_ ? backgroundCombo_->currentData().toString() : QString();
  UpdateBackgroundModeOptionVisibility();
  const bool vbBlurAvail =
      effectAvailable(QStringLiteral("virtual_background.blur"));
  const bool vbRemoveAvail =
      effectAvailable(QStringLiteral("virtual_background.remove"));
  const bool vbReplaceAvail =
      effectAvailable(QStringLiteral("virtual_background.replace"));
  const bool afAvail = effectAvailable(QStringLiteral("auto_frame"));
  const bool vbAnyAvail = vbBlurAvail || vbRemoveAvail || vbReplaceAvail;
  const bool vbOn =
      (vbMode == "blur" || vbMode == "remove" || vbMode == "replace");
  const bool afOn = (vbMode == "auto_frame");
  if (backgroundCombo_) {
    // Allow switching back to "off" even when Maxine is unavailable / effect is
    // missing.
    const bool allow = (vbAnyAvail || afAvail) || vbOn || afOn;
    const QString tipId =
        vbOn ? (QStringLiteral("virtual_background.") + vbMode)
             : (afOn ? QStringLiteral("auto_frame")
                     : QStringLiteral("virtual_background.blur"));
    setAvail(backgroundCombo_, allow, effectUnavailableTooltip(tipId));

    // Disable unavailable modes, but keep "off" selectable.
    auto *m = qobject_cast<QStandardItemModel *>(backgroundCombo_->model());
    if (m) {
      for (int i = 0; i < backgroundCombo_->count(); ++i) {
        const QString mode = backgroundCombo_->itemData(i).toString();
        bool itemEnabled = true;
        if (mode == "blur")
          itemEnabled = vbBlurAvail;
        else if (mode == "remove")
          itemEnabled = vbRemoveAvail;
        else if (mode == "replace")
          itemEnabled = vbReplaceAvail;
        else if (mode == "auto_frame")
          itemEnabled = afAvail;
        // "off" stays enabled.
        if (auto *item = m->item(i))
          item->setEnabled(itemEnabled);
      }
    }
  }
  if (backgroundStrengthSpin_ && backgroundCombo_) {
    const bool on = (vbMode == QStringLiteral("blur"));
    const QString tip =
        on && !vbBlurAvail
            ? effectUnavailableTooltip(QStringLiteral("virtual_background.blur"))
            : QString();
    backgroundStrengthSpin_->setEnabled(on);
    backgroundStrengthSpin_->setToolTip(tip);
    if (backgroundStrengthLabel_)
      backgroundStrengthLabel_->setToolTip(tip);
  }
  if (backgroundRemoveColorEdit_ && backgroundCombo_) {
    const bool on = (vbMode == QStringLiteral("remove"));
    const QString tip =
        on && !vbRemoveAvail
            ? effectUnavailableTooltip(QStringLiteral("virtual_background.remove"))
            : QString();
    backgroundRemoveColorEdit_->setEnabled(on);
    backgroundRemoveColorEdit_->setToolTip(tip);
    if (backgroundRemoveColorLabel_)
      backgroundRemoveColorLabel_->setToolTip(tip);
  }
  if (backgroundReplaceImageEdit_ && backgroundCombo_) {
    const bool on = (vbMode == QStringLiteral("replace"));
    const QString tip =
        on && !vbReplaceAvail
            ? effectUnavailableTooltip(QStringLiteral("virtual_background.replace"))
            : QString();
    backgroundReplaceImageEdit_->setEnabled(on);
    backgroundReplaceImageEdit_->setToolTip(tip);
    if (backgroundReplaceImageLabel_)
      backgroundReplaceImageLabel_->setToolTip(tip);
    if (browseReplaceImageBtn_) {
      browseReplaceImageBtn_->setEnabled(on);
      browseReplaceImageBtn_->setToolTip(tip);
    }
  }

  // Virtual Background model selection (Open Source-only).
  if (vbModelLabel_ && vbModelCombo_) {
    // Populate model list from daemon status (best-effort).
    if (daemonReachable_ && st.open_cuda_present) {
      bool hasTaskMeta = false;
      for (const auto &m : st.open_cuda_models) {
        if (!m.task.isEmpty()) {
          hasTaskMeta = true;
          break;
        }
      }

      QString sig;
      sig += QStringLiteral("matting|");
      sig += st.open_cuda_default_model_id;
      sig += QChar('|');

      for (const auto &m : st.open_cuda_models) {
        if (hasTaskMeta && m.task != QStringLiteral("matting"))
          continue;
        sig += m.id + QChar('\n');
        sig += m.display_name + QChar('\n');
        sig += m.task + QChar('\n');
      }

      if (sig != vbModelItemsSig_) {
        vbModelCombo_->blockSignals(true);
        vbModelCombo_->clear();
        if (!st.open_cuda_default_model_id.isEmpty()) {
          vbModelCombo_->addItem(
              QStringLiteral("<auto: %1>").arg(st.open_cuda_default_model_id),
              QString());
        } else {
          vbModelCombo_->addItem(QStringLiteral("<auto>"), QString());
        }

        for (const auto &m : st.open_cuda_models) {
          if (m.id.isEmpty())
            continue;
          if (hasTaskMeta && m.task != QStringLiteral("matting"))
            continue;
          const QString label = m.display_name.isEmpty()
                                    ? m.id
                                    : (m.display_name + QStringLiteral("  [") +
                                       m.id + QChar(']'));
          vbModelCombo_->addItem(label, m.id);
        }

        vbModelItemsSig_ = sig;
        vbModelCombo_->blockSignals(false);
      }

      // Keep selection in sync with the canonical local model.
      const QString selectedId =
          QString::fromStdString(effects_.virtual_background.model_id);
      vbModelCombo_->blockSignals(true);

      // If config references a model that isn't installed, show it explicitly
      // instead of silently falling back to <auto>.
      for (int i = vbModelCombo_->count() - 1; i >= 0; --i) {
        if (vbModelCombo_->itemText(i).startsWith(
                QStringLiteral("<missing:"))) {
          vbModelCombo_->removeItem(i);
        }
      }

      int idx = vbModelCombo_->findData(selectedId);
      if (!selectedId.isEmpty() && idx < 0) {
        const int insertAt = std::min(1, vbModelCombo_->count());
        vbModelCombo_->insertItem(
            insertAt, QStringLiteral("<missing: %1>").arg(selectedId),
            selectedId);
        idx = insertAt;
      }

      vbModelCombo_->setCurrentIndex(idx >= 0 ? idx : 0);
      vbModelCombo_->blockSignals(false);
    }

    // Show only when VB is active and Open Source is the selected (or
    // effective) backend.
    QMap<QString, QString> backendById;
    if (!st.effects_backends.isEmpty()) {
      const auto parts = st.effects_backends.split(',', Qt::SkipEmptyParts);
      for (const auto &raw : parts) {
        const QString p = raw.trimmed();
        const qsizetype colon = p.indexOf(QChar(':'));
        if (colon <= 0)
          continue;
        const QString id = p.left(colon).trimmed();
        const QString backend = p.mid(colon + 1).trimmed();
        if (!id.isEmpty() && !backend.isEmpty())
          backendById.insert(id, backend);
      }
    }

    QString vbEffectId;
    if (vbMode == QStringLiteral("blur"))
      vbEffectId = QStringLiteral("virtual_background.blur");
    else if (vbMode == QStringLiteral("remove"))
      vbEffectId = QStringLiteral("virtual_background.remove");
    else if (vbMode == QStringLiteral("replace"))
      vbEffectId = QStringLiteral("virtual_background.replace");

    const QString vbBackend = backendById.value(vbEffectId);

    auto isOpenBackend = [](const QString &backend) -> bool {
      const QString b = backend.trimmed().toLower();
      return b == QStringLiteral("open_cuda") ||
             b == QStringLiteral("open_video") ||
             b == QStringLiteral("open_source") || b == QStringLiteral("open");
    };

    bool showModelRow = false;
    if (daemonReachable_ && st.open_cuda_present && vbOn) {
      if (enginePref ==
          studiocast::video::effects::EffectsEnginePreference::open_cuda) {
        showModelRow = true;
      } else if (enginePref == studiocast::video::effects::
                                   EffectsEnginePreference::auto_select) {
        const bool maxAvail =
            maxineSupported && st.maxine_available_effects.contains(vbEffectId);
        const bool openAvail =
            st.open_cuda_present && st.open_cuda_ok &&
            st.open_cuda_available_effects.contains(vbEffectId);

        if (isOpenBackend(vbBackend)) {
          showModelRow = true;
        } else if (!maxineSupported && openAvail) {
          showModelRow = true;
        } else if (openAvail && !maxAvail) {
          showModelRow = true;
        }
      }
    }

    vbModelLabel_->setVisible(showModelRow);
    vbModelCombo_->setVisible(showModelRow);
    if (showModelRow) {
      const bool backendReady = st.open_cuda_present && st.open_cuda_ok;
      const QString tip =
          backendReady
              ? QString()
              : (st.open_cuda_present
                     ? OpenCudaSetupText(st, /*includeInstallHints=*/true)
                     : QStringLiteral(
                           "Open Video / Open CUDA status is not available."));
      vbModelCombo_->setEnabled(daemonReachable_ && backendReady);
      vbModelLabel_->setEnabled(daemonReachable_ && backendReady);
      vbModelCombo_->setToolTip(tip);
      vbModelLabel_->setToolTip(tip);
    } else {
      vbModelCombo_->setToolTip(QString());
      vbModelLabel_->setToolTip(QString());
    }
  }

  // Open Video model selection for other effects. Model availability is
  // daemon-reported; this routine must not scan local model directories.
  {
    // Determine when Open Source is the active engine for each effect. In AUTO,
    // some effects may run on Maxine while others fall back to Open Source.
    QMap<QString, QString> backendById;
    if (!st.effects_backends.isEmpty()) {
      const auto parts = st.effects_backends.split(',', Qt::SkipEmptyParts);
      for (const auto &raw : parts) {
        const QString p = raw.trimmed();
        const qsizetype colon = p.indexOf(QChar(':'));
        if (colon <= 0)
          continue;
        const QString id = p.left(colon).trimmed();
        const QString backend = p.mid(colon + 1).trimmed();
        if (!id.isEmpty() && !backend.isEmpty())
          backendById.insert(id, backend);
      }
    }

    auto isOpenBackend = [](const QString &backend) -> bool {
      const QString b = backend.trimmed().toLower();
      return b == QStringLiteral("open_cuda") ||
             b == QStringLiteral("open_video") ||
             b == QStringLiteral("open_source") || b == QStringLiteral("open");
    };

    auto showOpenModelRow = [&](const QString &effectId,
                                bool effectEnabled) -> bool {
      if (!effectEnabled)
        return false;

      // Explicit Open Source mode: always show the per-effect model selector.
      if (enginePref ==
          studiocast::video::effects::EffectsEnginePreference::open_cuda)
        return true;
      if (enginePref ==
          studiocast::video::effects::EffectsEnginePreference::maxine)
        return false;

      // AUTO: show only when Open Source is the likely or actual backend.
      if (isOpenBackend(backendById.value(effectId)))
        return true;

      const bool maxAvail =
          maxineSupported && st.maxine_available_effects.contains(effectId);
      const bool openAvail = st.open_cuda_present && st.open_cuda_ok &&
                             st.open_cuda_available_effects.contains(effectId);

      // Global fallback: Maxine is unavailable, but Open Source can run.
      if (!maxineSupported && openAvail)
        return true;

      // Effect-level fallback: effect exists in Open Source but not Maxine.
      if (openAvail && !maxAvail)
        return true;

      return false;
    };

    auto update_open_video_model_combo = [&](const char *task, QLabel *label,
                                             QComboBox *combo,
                                             QString *items_sig,
                                             const std::string
                                                 &selected_model_id,
                                             bool show_row) {
      if (!label || !combo || !items_sig)
        return;

      label->setVisible(show_row);
      combo->setVisible(show_row);

      std::vector<DaemonVideoStatus::OpenCudaModelInfo> packs;
      packs.reserve(st.open_cuda_models.size());
      const QString taskKey = QString::fromUtf8(task);
      bool hasTaskMeta = false;
      for (const auto &m : st.open_cuda_models) {
        if (!m.task.isEmpty()) {
          hasTaskMeta = true;
          break;
        }
      }
      for (const auto &m : st.open_cuda_models) {
        if (m.id.isEmpty())
          continue;
        if (hasTaskMeta && m.task != taskKey)
          continue;
        packs.push_back(m);
      }
      std::sort(packs.begin(), packs.end(), [](const auto &a, const auto &b) {
        const QString an = a.display_name.isEmpty() ? a.id : a.display_name;
        const QString bn = b.display_name.isEmpty() ? b.id : b.display_name;
        return an < bn;
      });

      QString defaultId;
      if (!st.open_cuda_default_model_id.isEmpty()) {
        const auto it = std::find_if(
            packs.begin(), packs.end(), [&](const auto &m) {
              return m.id == st.open_cuda_default_model_id;
            });
        if (it != packs.end())
          defaultId = st.open_cuda_default_model_id;
      }
      if (defaultId.isEmpty() && !packs.empty())
        defaultId = packs.front().id;

      QString sig = taskKey + QChar('|') + defaultId;
      for (const auto &p : packs) {
        sig += QChar('|') + p.id + QChar(':') + p.display_name;
      }

      if (sig != *items_sig) {
        QSignalBlocker b(combo);
        combo->clear();

        if (packs.empty()) {
          combo->addItem("<no daemon-reported models>", QString());
        } else {
          if (!defaultId.isEmpty()) {
            combo->addItem(QString("<auto: %1>").arg(defaultId), QString());
          } else {
            combo->addItem("<auto>", QString());
          }

          for (const auto &p : packs) {
            QString text = p.id;
            if (!p.display_name.isEmpty() && p.display_name != p.id) {
              text = QString("%1 (%2)").arg(p.display_name, p.id);
            } else {
              text = p.id;
            }
            combo->addItem(text, p.id);
          }
        }

        *items_sig = sig;
      }

      {
        QSignalBlocker b(combo);

        // If config references a model that isn't installed, show it explicitly
        // instead of silently falling back to <auto>.
        for (int i = combo->count() - 1; i >= 0; --i) {
          if (combo->itemText(i).startsWith(QStringLiteral("<missing:")))
            combo->removeItem(i);
        }

        const QString want = QString::fromStdString(selected_model_id);
        if (want.isEmpty()) {
          combo->setCurrentIndex(0);
        } else {
          int idx = -1;
          for (int i = 0; i < combo->count(); ++i) {
            if (combo->itemData(i).toString() == want) {
              idx = i;
              break;
            }
          }

          if (idx >= 0) {
            combo->setCurrentIndex(idx);
          } else {
            combo->insertItem(1, QString("<missing: %1>").arg(want), want);
            combo->setCurrentIndex(1);
          }
        }
      }

      const bool has_models = !packs.empty();
      const bool backend_ready = st.open_cuda_present && st.open_cuda_ok;
      combo->setEnabled(daemonReachable_ && has_models && backend_ready);
      label->setEnabled(daemonReachable_ && backend_ready);
      if (!backend_ready) {
        const QString tip =
            st.open_cuda_present
                ? OpenCudaSetupText(st, /*includeInstallHints=*/true)
                : QStringLiteral(
                      "Open Video / Open CUDA status is not available.");
        combo->setToolTip(tip);
        label->setToolTip(tip);
      } else if (!has_models) {
        const QString tip =
            QString("No daemon-reported models for task '%1'.\nOpen Engines & "
                    "Models for status and install hints.")
                .arg(task);
        combo->setToolTip(tip);
        label->setToolTip(tip);
      } else {
        combo->setToolTip(QString());
        label->setToolTip(QString());
      }
    };

    update_open_video_model_combo(
        "face_detection", autoFrameModelLabel_, autoFrameModelCombo_,
        &autoFrameModelItemsSig_, effects_.auto_frame.model_id,
        showOpenModelRow(QStringLiteral("auto_frame"),
                         effects_.auto_frame.enabled));
    update_open_video_model_combo(
        "eye_contact", eyeContactModelLabel_, eyeContactModelCombo_,
        &eyeContactModelItemsSig_, effects_.eye_contact.model_id,
        showOpenModelRow(QStringLiteral("eye_contact"),
                         effects_.eye_contact.enabled));
    update_open_video_model_combo(
        "video_denoise", denoiseModelLabel_, denoiseModelCombo_,
        &denoiseModelItemsSig_, effects_.video_noise_removal.model_id,
        showOpenModelRow(QStringLiteral("video_noise_removal"),
                         effects_.video_noise_removal.enabled));
  }

  // Auto Frame
  const bool afAvailable = effectAvailable(QStringLiteral("auto_frame"));
  const bool afCheckOn = autoFrameCheck_ ? autoFrameCheck_->isChecked() : false;
  if (autoFrameCheck_) {
    const bool allow = afAvailable || afCheckOn;
    setAvail(autoFrameCheck_, allow,
             effectUnavailableTooltip(QStringLiteral("auto_frame")));
  }
  if (autoFrameZoomSlider_)
    setAvail(autoFrameZoomSlider_, afAvailable && afCheckOn,
             effectUnavailableTooltip(QStringLiteral("auto_frame")));

  // Eye Contact
  const bool ecAvailable = effectAvailable("eye_contact");
  const bool ecOn = eyeContactCheck_ ? eyeContactCheck_->isChecked() : false;
  if (eyeContactCheck_) {
    const bool allow = ecAvailable || ecOn;
    setAvail(eyeContactCheck_, allow, effectUnavailableTooltip("eye_contact"));
  }
  if (eyeContactStrengthSlider_)
    setAvail(eyeContactStrengthSlider_, ecAvailable && ecOn,
             effectUnavailableTooltip("eye_contact"));
  if (eyeContactLookAwayCheck_)
    setAvail(eyeContactLookAwayCheck_, ecAvailable && ecOn,
             effectUnavailableTooltip("eye_contact"));

  // Video Noise Removal
  const bool dnAvailable = effectAvailable("video_noise_removal");
  const bool dnOn = denoiseCheck_ ? denoiseCheck_->isChecked() : false;
  if (denoiseCheck_) {
    const bool allow = dnAvailable || dnOn;
    setAvail(denoiseCheck_, allow,
             effectUnavailableTooltip("video_noise_removal"));
  }
  if (denoiseStrengthSlider_)
    setAvail(denoiseStrengthSlider_, dnAvailable && dnOn,
             effectUnavailableTooltip("video_noise_removal"));

  // Virtual Key Light gating based on Maxine diagnostics.
  const bool vklAvailable = effectAvailable("virtual_key_light");
  const bool vklOn =
      virtualKeyLightCheck_ ? virtualKeyLightCheck_->isChecked() : false;

  if (virtualKeyLightCheck_) {
    const bool allow = vklAvailable || vklOn;
    setAvail(virtualKeyLightCheck_, allow,
             effectUnavailableTooltip("virtual_key_light"));
  }
  if (virtualKeyLightIntensitySpin_)
    setAvail(virtualKeyLightIntensitySpin_, vklAvailable && vklOn,
             effectUnavailableTooltip("virtual_key_light"));
  if (virtualKeyLightTempCombo_)
    setAvail(virtualKeyLightTempCombo_, vklAvailable && vklOn,
             effectUnavailableTooltip("virtual_key_light"));
  if (virtualKeyLightPanSpin_)
    setAvail(virtualKeyLightPanSpin_, vklAvailable && vklOn,
             effectUnavailableTooltip("virtual_key_light"));
  if (virtualKeyLightHdriEdit_)
    setAvail(virtualKeyLightHdriEdit_, vklAvailable && vklOn,
             effectUnavailableTooltip("virtual_key_light"));
  if (browseVirtualKeyLightHdriBtn_)
    setAvail(browseVirtualKeyLightHdriBtn_, vklAvailable && vklOn,
             effectUnavailableTooltip("virtual_key_light"));

  // Vignette (GPU utility).
  const bool vigAvailable = effectAvailable(QStringLiteral("vignette"));
  const bool vigOn = vignetteCheck_ ? vignetteCheck_->isChecked() : false;
  if (vignetteCheck_) {
    const bool allow = vigAvailable || vigOn;
    setAvail(vignetteCheck_, allow,
             effectUnavailableTooltip(QStringLiteral("vignette")));
  }
  if (vignetteIntensitySlider_)
    setAvail(vignetteIntensitySlider_, vigAvailable && vigOn,
             effectUnavailableTooltip(QStringLiteral("vignette")));
  if (vignetteCenterOnFaceCheck_)
    setAvail(vignetteCenterOnFaceCheck_, vigAvailable && vigOn,
             effectUnavailableTooltip(QStringLiteral("vignette")));

  startBtn_->setEnabled(daemonReachable_ && !enabled && outSelectable &&
                        !outputCombo_->currentData().toString().isEmpty());
  stopBtn_->setEnabled(daemonReachable_ && enabled);

  const bool previewRequested =
      previewCheck_ ? previewCheck_->isChecked() : false;

  if (previewCheck_) {
    previewCheck_->setEnabled(daemonReachable_);
    if (!daemonReachable_) {
      previewCheck_->setToolTip("Background service unavailable.");
    } else if (!enabled) {
      previewCheck_->setToolTip("Start the camera before opening preview.");
    } else {
      previewCheck_->setToolTip(
          "Opens the virtual camera in this GUI. This counts as a consumer.");
    }
  }

  // Keep preview in sync with explicit user request.
  if (enabled && daemonReachable_ && previewRequested &&
      !previewCapture_.IsOpen()) {
    const auto now = std::chrono::steady_clock::now();
    if (previewAutoRetryFailures_ > 0 && now < previewAutoRetryAfter_) {
      if (DebugGuiPreview()) {
        GuiPreviewDbg("UpdateUiEnabled: enabled=1 but preview not open; "
                      "auto-retry backoff active");
      }
    } else {
      if (DebugGuiPreview()) {
        GuiPreviewDbg("UpdateUiEnabled: enabled=1 but preview not open -> "
                      "StartPreview()");
      }
      StartPreview();
    }
  }
  if ((!enabled || !daemonReachable_ || !previewRequested) &&
      previewCapture_.IsOpen()) {
    if (DebugGuiPreview()) {
      GuiPreviewDbg("UpdateUiEnabled: preview not requested/available but "
                    "open -> StopPreview()");
    }
    StopPreview();
  }
}

void VideoPage::UpdateStatusText() {
  DaemonVideoStatus st;
  QString parseErr;
  bool parsedStatus = false;
  if (daemonReachable_ && !daemonLastStatusJson_.empty()) {
    parsedStatus = ParseDaemonStatusJson(daemonLastStatusJson_, &st, &parseErr);
  }

  std::ostringstream oss;
  oss << baseStatusText_ << "\n\n---\nDaemon (studiocastd)\n";

  if (!daemonReachable_ || !parsedStatus) {
    if (cameraStateLabel_) {
      cameraStateLabel_->setText(QStringLiteral("Service unavailable"));
      SetDynamicProperty(cameraStateLabel_, "scStatus", QStringLiteral("error"));
    }
    if (cameraDetailLabel_) {
      QStringList lines;
      lines << QStringLiteral(
          "StudioCast background service is unavailable.");
      lines << QStringLiteral("Open Support for technical details.");
      cameraDetailLabel_->setText(lines.join(QStringLiteral("\n")));
    }

    oss << "  status: not running / not reachable";
    if (!parseErr.isEmpty())
      oss << " (invalid status JSON)";
    oss << "\n";
    if (!daemonStatusDetail_.isEmpty())
      oss << "  detail: " << daemonStatusDetail_.toStdString() << "\n";
    if (!parseErr.isEmpty())
      oss << "  detail: " << parseErr.toStdString() << "\n";
    oss << "\nTips\n"
        << "  - Start the daemon in a terminal:\n"
        << "      ./build/studiocastd\n"
        << "  - Or enable the systemd user service (packaging step).\n";

    SetPlainTextPreservingScroll(statusText_,
                                 QString::fromStdString(oss.str()));
    return;
  }

  const QString pipelineState =
      st.pipeline_state.isEmpty()
          ? QString::fromUtf8(st.pipeline_running
                                  ? "running"
                                  : (st.pipeline_starting ? "starting"
                                                          : "stopped"))
          : st.pipeline_state;

  if (cameraStateLabel_ || cameraDetailLabel_) {
    QString state = QStringLiteral("Ready");
    QString statusProperty = QStringLiteral("good");
    QStringList details;
    const QString pipelineStateLower = pipelineState.trimmed().toLower();
    const bool inputError = LooksLikeCameraInputError(st.last_error);

    if (!st.virtual_device_present || !st.virtual_device_available) {
      state = QStringLiteral("Virtual camera missing");
      statusProperty = QStringLiteral("warning");
      details << QStringLiteral("Virtual camera is missing.");
      details << QStringLiteral("Open Diagnostics for setup guidance.");
    } else if (!st.consumer_error.trimmed().isEmpty()) {
      state = QStringLiteral("Needs attention");
      statusProperty = QStringLiteral("error");
      details << QStringLiteral(
          "StudioCast cannot tell when other apps are using the camera. Open "
          "Support for technical details.");
    } else if (inputError) {
      state = QStringLiteral("No camera input is selected");
      statusProperty = QStringLiteral("warning");
      details << QStringLiteral("Choose a readable physical camera input.");
    } else if (st.enabled && (st.pipeline_running ||
                              pipelineStateLower == QStringLiteral("running"))) {
      state = QStringLiteral("Running");
      details << QStringLiteral("Processing is active.");
    } else if (st.enabled && (st.pipeline_starting ||
                              pipelineStateLower == QStringLiteral("starting"))) {
      state = QStringLiteral("Starting");
      statusProperty = QStringLiteral("warning");
      details << QStringLiteral("Camera processing is starting.");
    } else if (st.enabled &&
               pipelineStateLower == QStringLiteral("backing_off")) {
      state = QStringLiteral("Retry pending");
      statusProperty = QStringLiteral("warning");
      details << QStringLiteral("Camera processing will retry shortly.");
    } else if (st.enabled &&
               pipelineStateLower ==
                   QStringLiteral("waiting_for_stable_consumer")) {
      state = QStringLiteral("Waiting for consumer");
      details << QStringLiteral(
          "An app opened StudioCast Camera; processing starts after the "
          "consumer is stable.");
    } else if (st.enabled) {
      state = QStringLiteral("Waiting for consumer");
      details << QStringLiteral(
          "Ready. Processing starts when an app opens StudioCast Camera.");
    } else {
      details << QStringLiteral("Ready to start StudioCast Camera.");
    }

    details << QStringLiteral("Apps using camera: %1").arg(st.consumer_count);
    if (!st.last_error.isEmpty()) {
      if (!inputError)
        statusProperty = QStringLiteral("error");
      if (!inputError) {
        details << QStringLiteral(
            "Camera processing reported a problem. Open Support for technical "
            "details.");
      }
    }

    if (cameraStateLabel_) {
      cameraStateLabel_->setText(state);
      SetDynamicProperty(cameraStateLabel_, "scStatus", statusProperty);
    }
    if (cameraDetailLabel_)
      cameraDetailLabel_->setText(details.join(QStringLiteral("\n")));
  }

  oss << "  enabled:    " << (st.enabled ? "yes" : "no") << "\n";
  oss << "  virtual:    "
      << (st.virtual_device_present ? "present" : "missing") << " / "
      << (st.virtual_device_available ? "available" : "unavailable") << "\n";
  if (!st.virtual_device_error.isEmpty()) {
    oss << "  virtual err: " << st.virtual_device_error.toStdString() << "\n";
  }
  oss << "  consumers:  " << st.consumer_count
      << (st.consumer_present ? " (present)" : "") << "\n";
  if (!st.consumer_error.isEmpty()) {
    oss << "  consumer err: " << st.consumer_error.toStdString() << "\n";
  }
  oss << "  pipeline:   " << pipelineState.toStdString();
  if (!st.pipeline_idle_reason.isEmpty())
    oss << " (" << st.pipeline_idle_reason.toStdString() << ")";
  if (st.pipeline_active_needed)
    oss << " active_needed";
  oss << "\n";
  if (st.next_start_retry_ms >= 0) {
    oss << "  retry in:   " << st.next_start_retry_ms << " ms\n";
  }
  if (!st.last_transition.isEmpty()) {
    oss << "  transition: " << st.last_transition.toStdString();
    if (st.last_transition_ms_ago >= 0)
      oss << " (" << st.last_transition_ms_ago << " ms ago)";
    oss << "\n";
  }
  if (st.stabilizing || st.thrash_events_10s > 0) {
    oss << "  stabilize:  " << (st.stabilizing ? "yes" : "no")
        << " thrash_events_10s=" << st.thrash_events_10s << "\n";
  }
  if (st.consumer_poll_ms > 0) {
    oss << "  grace:      start=" << st.start_grace_ms
        << " stop=" << st.stop_grace_ms << " min_run=" << st.min_run_ms
        << " poll=" << st.consumer_poll_ms << " ms\n";
  }
  oss << "  starts:     attempts=" << st.pipeline_start_attempts
      << " ok=" << st.pipeline_starts
      << " failed=" << st.pipeline_start_failures
      << " stops=" << st.pipeline_stops
      << " restarts=" << st.pipeline_config_restarts << "\n";

  const auto fmtPixfmt = [](const QString &pixfmt) -> QString {
    if (pixfmt == QStringLiteral("RGB3"))
      return QStringLiteral("RGB24");
    if (pixfmt == QStringLiteral("BGR3"))
      return QStringLiteral("BGR24");
    return pixfmt;
  };

  const auto fmtFps = [](double fps) -> std::string {
    if (fps <= 0.0)
      return {};
    const double r = std::round(fps);
    if (std::fabs(fps - r) < 0.01) {
      return std::to_string(static_cast<int>(r));
    }
    QString s = QString::number(fps, 'f', 2);
    while (s.contains('.') && (s.endsWith('0') || s.endsWith('.')))
      s.chop(1);
    return s.toStdString();
  };

  const auto fmtDims = [](int w, int h) -> std::string {
    return (QString::number(w) + QChar(0x00D7) + QString::number(h))
        .toStdString();
  };

  const auto fmtLine = [&](const DaemonVideoStatus::NegotiatedFormat &f,
                           bool withPixfmtFirst) -> std::string {
    if (!f.present || f.width <= 0 || f.height <= 0)
      return "—";

    std::ostringstream line;
    const QString pix = fmtPixfmt(f.pixfmt);
    const std::string fpsStr = fmtFps(f.fps);

    if (withPixfmtFirst && !pix.isEmpty()) {
      line << pix.toStdString() << " ";
    }
    line << fmtDims(f.width, f.height);
    if (!fpsStr.empty()) {
      line << " @ " << fpsStr;
    }
    if (!withPixfmtFirst && !pix.isEmpty()) {
      line << " (" << pix.toStdString() << ")";
    }
    if (f.bytes_per_line > 0) {
      line << " (stride " << f.bytes_per_line << ")";
    }
    return line.str();
  };

  if (st.pipeline_running) {
    oss << "  Capture:    "
        << fmtLine(st.capture_format, /*withPixfmtFirst=*/true) << "\n";
    if (!st.capture_fallback_state.isEmpty() &&
        st.capture_fallback_state != QStringLiteral("none")) {
      oss << "  Fallback:   " << st.capture_fallback_state.toStdString();
      if (!st.capture_fallback_reason.isEmpty()) {
        oss << " (" << st.capture_fallback_reason.toStdString() << ")";
      }
      oss << "\n";
    }
    oss << "  Output:     "
        << fmtLine(st.output_format, /*withPixfmtFirst=*/false) << "\n";

    {
      std::ostringstream line;
      if (!st.scaling_backend_active.isEmpty()) {
        line << st.scaling_backend_active.toStdString();
      } else {
        line << "—";
      }

      const auto &from =
          st.scaling_from.present ? st.scaling_from : st.capture_format;
      const auto &to = st.scaling_to.present ? st.scaling_to : st.output_format;

      if (from.present && to.present && from.width > 0 && from.height > 0 &&
          to.width > 0 && to.height > 0) {
        line << " (" << fmtDims(from.width, from.height) << " → "
             << fmtDims(to.width, to.height) << ")";
      }

      oss << "  Scaling:    " << line.str() << "\n";
    }
  } else {
    oss << "  Capture:    —\n";
    oss << "  Output:     —\n";
    oss << "  Scaling:    —\n";
  }

  oss << "  input:      " << st.input_device.toStdString() << "\n";
  oss << "  output:     " << st.output_device.toStdString() << "\n";
  oss << "  requested:  " << st.width << "x" << st.height << " @ " << st.fps
      << " fps, output_format="
      << (st.output_format_requested.isEmpty()
              ? std::string("rgb24")
              : st.output_format_requested.toStdString())
      << "\n";
  oss << "  Compute:    preference="
      << (st.compute_preference.isEmpty() ? std::string("auto")
                                          : st.compute_preference.toStdString())
      << ", resolved="
      << (st.compute_resolved_backend.isEmpty()
              ? std::string("cpu")
              : st.compute_resolved_backend.toStdString())
      << ", active="
      << (st.compute_active_backend.isEmpty()
              ? std::string("cpu")
              : st.compute_active_backend.toStdString())
      << "\n";
  if (!st.compute_fallback_reason.isEmpty()) {
    oss << "  fallback:   " << st.compute_fallback_reason.toStdString()
        << "\n";
  }
  if (!st.compute_degraded_reason.isEmpty() &&
      st.compute_degraded_reason != st.compute_fallback_reason) {
    oss << "  degraded:   " << st.compute_degraded_reason.toStdString()
        << "\n";
  }
  if (st.effects_valid) {
    oss << "  mirror:     " << (st.effects.mirror ? "on" : "off") << "\n";

    const bool autoFrame = st.effects.auto_frame.enabled;
    const std::string vbMode = autoFrame
                                   ? "auto_frame"
                                   : studiocast::video::effects::ToString(
                                         st.effects.virtual_background.mode);

    oss << "  background: " << vbMode;
    if (vbMode == "blur" || vbMode == "remove" || vbMode == "replace") {
      oss << " strength=" << st.effects.virtual_background.strength;
    }
    if ((vbMode == "remove" || vbMode == "replace") &&
        !st.effects.virtual_background.remove_color.empty()) {
      oss << " color=" << st.effects.virtual_background.remove_color;
    }
    if (vbMode == "replace" &&
        !st.effects.virtual_background.replace_path.empty()) {
      oss << " image=" << st.effects.virtual_background.replace_path;
    }
    oss << "\n";
  } else {
    oss << "  effects:    (failed to parse video_effects)\n";
  }

  if (!st.maxine_summary.isEmpty()) {
    oss << "  maxine:     " << st.maxine_summary.toStdString() << "\n";
    oss << "  vkl avail:  " << (st.virtual_key_light_available ? "yes" : "no")
        << "\n";
  }

  if (st.effects_valid) {
    const auto presetToStr = [](int p) -> const char * {
      switch (p) {
      case 1:
        return "warm";
      case 2:
        return "cool";
      default:
        return "neutral";
      }
    };
    oss << "  key light:  "
        << (st.effects.virtual_key_light.enabled ? "on" : "off")
        << " intensity=" << st.effects.virtual_key_light.intensity << "%"
        << " temp="
        << presetToStr(st.effects.virtual_key_light.temperature_preset)
        << " pan=" << st.effects.virtual_key_light.direction_pan_degrees;
    if (!st.effects.virtual_key_light.hdri_path.empty()) {
      oss << " hdri=" << st.effects.virtual_key_light.hdri_path;
    }
    oss << "\n";
  }

  if (!st.effects_backends.isEmpty()) {
    oss << "  effects:    " << st.effects_backends.toStdString() << "\n";
  }
  if (!st.effects_note.isEmpty()) {
    oss << "  fx note:    " << st.effects_note.toStdString() << "\n";
  }
  oss << "  frames:     " << st.frame_index << "\n";

  if (!st.last_error.isEmpty()) {
    oss << "  last error: " << st.last_error.toStdString() << "\n";
  }

  oss << "\nNotes\n"
      << "  - The daemon only runs heavy processing when consumers are present "
         "(OBS/Zoom/GUI preview).\n"
      << "  - Preview counts as a consumer: it opens the v4l2loopback device "
         "for capture.\n";

  SetPlainTextPreservingScroll(statusText_,
                               QString::fromStdString(oss.str()));
}

} // namespace studiocast::gui
