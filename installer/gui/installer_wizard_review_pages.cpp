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
