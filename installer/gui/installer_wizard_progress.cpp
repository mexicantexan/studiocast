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


} // namespace studiocast::installer
