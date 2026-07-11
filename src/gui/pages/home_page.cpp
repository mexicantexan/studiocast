#include "home_page.h"

#include <QApplication>
#include <QClipboard>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QStyle>
#include <QVBoxLayout>

#include <vector>

#include "gui/status/daemon_status_snapshot.h"

namespace studiocast::gui {
namespace {

constexpr const char *kCameraExternalName = "StudioCast Camera";
constexpr const char *kMicrophoneExternalName = "StudioCast Microphone";
constexpr const char *kSpeakersExternalName = "StudioCast Speakers";

struct RepairIssue {
  QString title;
  QString detail;
  QString copyText;
  HomePage::Destination destination = HomePage::Destination::Support;
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

void SetDynamicProperty(QWidget *widget, const char *name,
                        const QString &value) {
  widget->setProperty(name, value);
  widget->style()->unpolish(widget);
  widget->style()->polish(widget);
}

QString ReadinessProperty(ReadinessState state) {
  switch (state) {
  case ReadinessState::Ready:
  case ReadinessState::Idle:
  case ReadinessState::Processing:
    return QStringLiteral("good");
  case ReadinessState::NeedsSetup:
  case ReadinessState::MissingVirtualDevice:
  case ReadinessState::NoPhysicalDevice:
  case ReadinessState::MissingModel:
  case ReadinessState::Unknown:
    return QStringLiteral("warning");
  case ReadinessState::DaemonUnavailable:
  case ReadinessState::RecoverableError:
  case ReadinessState::FatalError:
    return QStringLiteral("error");
  }
  return QStringLiteral("warning");
}

bool IsBlockingState(ReadinessState state) {
  switch (state) {
  case ReadinessState::Ready:
  case ReadinessState::Idle:
  case ReadinessState::Processing:
    return false;
  case ReadinessState::Unknown:
  case ReadinessState::NeedsSetup:
  case ReadinessState::DaemonUnavailable:
  case ReadinessState::MissingVirtualDevice:
  case ReadinessState::NoPhysicalDevice:
  case ReadinessState::RecoverableError:
  case ReadinessState::FatalError:
  case ReadinessState::MissingModel:
    return true;
  }
  return true;
}

QString JoinNonEmpty(const QStringList &lines) {
  QStringList out;
  for (const QString &line : lines) {
    const QString trimmed = line.trimmed();
    if (!trimmed.isEmpty())
      out.push_back(trimmed);
  }
  return out.join(QStringLiteral("\n"));
}

QString PrimaryDeviceSummary(const QString &deviceName,
                             const DeviceReadiness &readiness) {
  switch (readiness.state) {
  case ReadinessState::DaemonUnavailable:
    return QStringLiteral(
        "StudioCast background service is unavailable.");
  case ReadinessState::MissingVirtualDevice:
    if (deviceName == QStringLiteral("Camera"))
      return QStringLiteral("Virtual camera is missing.");
    if (deviceName == QStringLiteral("Microphone"))
      return QStringLiteral("StudioCast Microphone is missing.");
    return QStringLiteral("StudioCast Speakers are missing.");
  case ReadinessState::NoPhysicalDevice:
    if (deviceName == QStringLiteral("Speakers"))
      return QStringLiteral("No speaker output is selected.");
    if (deviceName == QStringLiteral("Camera"))
      return QStringLiteral("No camera input is selected.");
    return QStringLiteral("No microphone input is selected.");
  case ReadinessState::RecoverableError:
  case ReadinessState::FatalError:
    return QStringLiteral("%1 needs attention.").arg(deviceName);
  case ReadinessState::Unknown:
    return QStringLiteral("%1 status is unavailable.").arg(deviceName);
  case ReadinessState::NeedsSetup:
    return QStringLiteral("%1 needs setup.").arg(deviceName);
  case ReadinessState::MissingModel:
    return QStringLiteral("%1 needs a model pack.").arg(deviceName);
  case ReadinessState::Ready:
  case ReadinessState::Idle:
  case ReadinessState::Processing:
    return readiness.summary.isEmpty()
               ? QStringLiteral("%1 is ready.").arg(deviceName)
               : readiness.summary;
  }
  return readiness.summary.isEmpty()
             ? QStringLiteral("%1 status is unavailable.").arg(deviceName)
             : readiness.summary;
}

QString PrimaryDeviceDetail(const QString &deviceName,
                            const DeviceReadiness &readiness) {
  switch (readiness.state) {
  case ReadinessState::DaemonUnavailable:
    return QStringLiteral("Open Support for technical details.");
  case ReadinessState::MissingVirtualDevice:
    if (deviceName == QStringLiteral("Camera"))
      return QStringLiteral("Open Camera for virtual camera setup.");
    return QStringLiteral("Open %1 to recreate the StudioCast device.")
        .arg(deviceName);
  case ReadinessState::NoPhysicalDevice:
    if (deviceName == QStringLiteral("Speakers"))
      return QStringLiteral("Open Speakers to choose a physical output.");
    if (deviceName == QStringLiteral("Camera"))
      return QStringLiteral("Open Camera to choose a physical input.");
    return QStringLiteral("Open Microphone to choose a physical input.");
  case ReadinessState::RecoverableError:
  case ReadinessState::FatalError:
    return QStringLiteral(
        "Open %1 for next steps, or open Support for technical details.")
        .arg(deviceName);
  case ReadinessState::Unknown:
    return QStringLiteral("Open Support for technical details.");
  case ReadinessState::NeedsSetup:
    return QStringLiteral("Open %1 for setup.").arg(deviceName);
  case ReadinessState::MissingModel:
    return QStringLiteral("Open Engines & Models for install hints.");
  case ReadinessState::Ready:
  case ReadinessState::Idle:
  case ReadinessState::Processing:
    return {};
  }
  return {};
}

QString DeviceGuidance(const QString &deviceName, ReadinessState state) {
  if (state == ReadinessState::MissingVirtualDevice) {
    if (deviceName == QStringLiteral("Camera")) {
      return QStringLiteral("Open Camera for virtual camera setup.");
    }
    if (deviceName == QStringLiteral("Microphone")) {
      return QStringLiteral(
          "Open Microphone to inspect the StudioCast virtual microphone.");
    }
    return QStringLiteral(
        "Open Speakers to inspect the StudioCast Speakers device.");
  }

  if (state == ReadinessState::NoPhysicalDevice) {
    if (deviceName == QStringLiteral("Camera"))
      return QStringLiteral("Open Camera to choose a physical input.");
    if (deviceName == QStringLiteral("Microphone"))
      return QStringLiteral("Open Microphone to choose a physical input.");
    return QStringLiteral("Open Speakers to choose a physical output.");
  }

  return QStringLiteral("Open the device page for details.");
}

QString DestinationAction(HomePage::Destination destination) {
  switch (destination) {
  case HomePage::Destination::Camera:
    return QStringLiteral("Open Camera");
  case HomePage::Destination::Microphone:
    return QStringLiteral("Open Microphone");
  case HomePage::Destination::Speakers:
    return QStringLiteral("Open Speakers");
  case HomePage::Destination::Engines:
    return QStringLiteral("Open Engines && Models");
  case HomePage::Destination::Support:
    return QStringLiteral("Open Support");
  }
  return QStringLiteral("Open");
}

void ClearLayout(QLayout *layout) {
  while (QLayoutItem *item = layout->takeAt(0)) {
    if (QWidget *widget = item->widget())
      widget->deleteLater();
    delete item;
  }
}

void CopyText(const QString &text) {
  if (QClipboard *clipboard = QApplication::clipboard())
    clipboard->setText(text);
}

void AddIssue(std::vector<RepairIssue> *issues, const QString &title,
              const QString &detail, const QString &guidance,
              HomePage::Destination destination,
              const QString &extraCopy = {}) {
  QStringList copyLines;
  copyLines << title << detail << guidance << extraCopy;
  issues->push_back(RepairIssue{title, JoinNonEmpty({detail, guidance}),
                                JoinNonEmpty(copyLines), destination});
}

void AddDeviceIssues(std::vector<RepairIssue> *issues, const QString &name,
                     const DeviceReadiness &readiness,
                     HomePage::Destination destination) {
  if (IsBlockingState(readiness.state)) {
    AddIssue(issues, PrimaryDeviceSummary(name, readiness),
             PrimaryDeviceDetail(name, readiness),
             DeviceGuidance(name, readiness.state), destination);
  }

  // Effect disable notes can be informational or intentional. Backend/model
  // blockers are added separately by AddEngineIssue, so keep this repair queue
  // limited to actual readiness blockers.
}

QString EngineIssueDetail(const EngineStatus &engine) {
  QStringList lines;
  lines << engine.summary;
  if (!engine.blockedReason.trimmed().isEmpty())
    lines << engine.blockedReason;
  lines << engine.blockedDetails;
  if (!engine.missingModels.isEmpty()) {
    lines << QStringLiteral("Missing or invalid models:");
    lines << engine.missingModels;
  }
  if (!engine.blockedEffects.isEmpty()) {
    lines << QStringLiteral("Blocked effects:");
    lines << engine.blockedEffects;
  }
  return JoinNonEmpty(lines);
}

void AddEngineIssue(std::vector<RepairIssue> *issues,
                    const EngineStatus &engine, bool selected) {
  const bool hasModelBlocker = engine.missingModelCount > 0;
  const bool hasEffectBlocker = !engine.blockedEffects.isEmpty();
  const bool selectedUnavailable =
      selected && engine.present && !(engine.ok || engine.supported);
  if (!hasModelBlocker && !hasEffectBlocker && !selectedUnavailable)
    return;

  const QString title =
      QStringLiteral("%1: Models or backend need attention").arg(engine.label);
  const QString detail = EngineIssueDetail(engine);
  const QString guidance =
      QStringLiteral("Open Engines & Models for install hints and backend "
                     "diagnostics.");
  AddIssue(issues, title, detail, guidance, HomePage::Destination::Engines,
           JoinNonEmpty(engine.installHints));
}

std::vector<RepairIssue> BuildRepairIssues(
    const DaemonStatusSnapshot &snapshot) {
  std::vector<RepairIssue> issues;

  if (!snapshot.reachable) {
    AddIssue(&issues,
             QStringLiteral("StudioCast background service is unavailable"),
             snapshot.UserServiceDetail(),
             QStringLiteral("Open Support for technical details."),
             HomePage::Destination::Support);
    return issues;
  }

  if (!snapshot.parsed) {
    AddIssue(&issues, QStringLiteral("Status needs attention"),
             snapshot.UserServiceDetail(),
             QStringLiteral("Open Support for technical details."),
             HomePage::Destination::Support);
    return issues;
  }

  if (!snapshot.serviceRunning) {
    AddIssue(&issues, QStringLiteral("Background service is not ready"),
             snapshot.UserServiceDetail(),
             QStringLiteral("Open Support for technical details."),
             HomePage::Destination::Support);
    return issues;
  }

  AddDeviceIssues(&issues, QStringLiteral("Camera"), snapshot.camera,
                  HomePage::Destination::Camera);
  AddDeviceIssues(&issues, QStringLiteral("Microphone"), snapshot.microphone,
                  HomePage::Destination::Microphone);
  AddDeviceIssues(&issues, QStringLiteral("Speakers"), snapshot.speakers,
                  HomePage::Destination::Speakers);

  const QString videoEngine =
      snapshot.videoEffectsEnginePreference.trimmed().toLower();
  const QString audioEngine =
      snapshot.audioEffectsEnginePreference.trimmed().toLower();
  const bool maxineSelected =
      videoEngine == QStringLiteral("maxine") ||
      audioEngine == QStringLiteral("maxine");
  const bool openVideoSelected = videoEngine == QStringLiteral("open_cuda");
  const bool openAudioSelected =
      audioEngine == QStringLiteral("open_source") ||
      audioEngine == QStringLiteral("open_audio");

  AddEngineIssue(&issues, snapshot.maxine, maxineSelected);
  AddEngineIssue(&issues, snapshot.openCuda, openVideoSelected);
  AddEngineIssue(&issues, snapshot.openAudio, openAudioSelected);

  return issues;
}

} // namespace

HomePage::HomePage(QWidget *parent) : QWidget(parent) {
  auto *root = new QVBoxLayout(this);
  root->setContentsMargins(16, 16, 16, 16);
  root->setSpacing(12);

  auto *summaryFrame = new QFrame(this);
  summaryFrame->setProperty("scRole", "homeSummary");
  auto *summaryLayout = new QVBoxLayout(summaryFrame);
  summaryLayout->setContentsMargins(14, 14, 14, 14);
  summaryLayout->setSpacing(4);
  overallLabel_ = ValueLabel(QStringLiteral("Checking StudioCast"), summaryFrame);
  overallLabel_->setProperty("scRole", "homeHeadline");
  overallDetailLabel_ =
      MutedLabel(QStringLiteral("Waiting for service status."), summaryFrame);
  summaryLayout->addWidget(overallLabel_);
  summaryLayout->addWidget(overallDetailLabel_);
  root->addWidget(summaryFrame);

  auto *cardsFrame = new QFrame(this);
  cardsFrame->setProperty("scRole", "homeCards");
  auto *cardsLayout = new QVBoxLayout(cardsFrame);
  cardsLayout->setContentsMargins(0, 0, 0, 0);
  cardsLayout->setSpacing(10);

  cards_[0] = CreateCard(QStringLiteral("Camera"),
                         QString::fromUtf8(kCameraExternalName),
                         Destination::Camera, cardsFrame);
  cards_[1] = CreateCard(QStringLiteral("Microphone"),
                         QString::fromUtf8(kMicrophoneExternalName),
                         Destination::Microphone, cardsFrame);
  cards_[2] = CreateCard(QStringLiteral("Speakers"),
                         QString::fromUtf8(kSpeakersExternalName),
                         Destination::Speakers, cardsFrame);

  for (const ReadinessCard &card : cards_)
    cardsLayout->addWidget(card.frame);
  root->addWidget(cardsFrame);

  repairGroup_ = new QGroupBox(QStringLiteral("Repair / Setup Queue"), this);
  repairGroup_->setVisible(false);
  repairListLayout_ = new QVBoxLayout(repairGroup_);
  repairListLayout_->setSpacing(10);
  root->addWidget(repairGroup_);
  root->addStretch(1);
}

HomePage::ReadinessCard
HomePage::CreateCard(const QString &title, const QString &deviceName,
                     Destination destination, QWidget *parent) {
  ReadinessCard card;
  card.frame = new QFrame(parent);
  card.frame->setProperty("scRole", "homeCard");
  card.frame->setProperty("scStatus", "warning");
  auto *layout = new QVBoxLayout(card.frame);
  layout->setContentsMargins(14, 14, 14, 14);
  layout->setSpacing(10);

  auto *header = new QHBoxLayout();
  header->setContentsMargins(0, 0, 0, 0);
  header->setSpacing(10);
  card.title = ValueLabel(title, card.frame);
  card.title->setProperty("scRole", "homeCardTitle");
  card.state = new QLabel(QStringLiteral("Unknown"), card.frame);
  card.state->setProperty("scRole", "statusPill");
  card.state->setProperty("scStatus", "warning");
  card.state->setAlignment(Qt::AlignCenter);
  header->addWidget(card.title, 1);
  header->addWidget(card.state, 0);
  layout->addLayout(header);

  card.summary = MutedLabel(QStringLiteral("Status has not been read."),
                            card.frame);
  layout->addWidget(card.summary);

  card.detail = MutedLabel(QString(), card.frame);
  card.detail->setVisible(false);
  layout->addWidget(card.detail);

  auto *deviceLabel =
      MutedLabel(QStringLiteral("External app device name"), card.frame);
  layout->addWidget(deviceLabel);

  auto *deviceRow = new QHBoxLayout();
  deviceRow->setContentsMargins(0, 0, 0, 0);
  deviceRow->setSpacing(8);
  card.deviceName = new QLineEdit(deviceName, card.frame);
  card.deviceName->setReadOnly(true);
  card.deviceName->setProperty("scRole", "copyValue");
  card.copyButton = new QPushButton(QStringLiteral("Copy"), card.frame);
  connect(card.copyButton, &QPushButton::clicked, card.frame,
          [lineEdit = card.deviceName] { CopyText(lineEdit->text()); });
  deviceRow->addWidget(card.deviceName, 1);
  deviceRow->addWidget(card.copyButton, 0);
  layout->addLayout(deviceRow);

  auto *actions = new QHBoxLayout();
  actions->setContentsMargins(0, 0, 0, 0);
  actions->addStretch(1);
  card.openButton = new QPushButton(DestinationAction(destination), card.frame);
  connect(card.openButton, &QPushButton::clicked, this,
          [this, destination] { OpenDestination(destination); });
  actions->addWidget(card.openButton);
  layout->addLayout(actions);

  return card;
}

void HomePage::UpdateCard(ReadinessCard *card,
                          const DeviceReadiness &readiness) {
  const QString status = ReadinessProperty(readiness.state);
  card->state->setText(ReadinessLabel(readiness.state));
  SetDynamicProperty(card->state, "scStatus", status);
  SetDynamicProperty(card->frame, "scStatus", status);

  card->summary->setText(PrimaryDeviceSummary(card->title->text(), readiness));

  QString detail = PrimaryDeviceDetail(card->title->text(), readiness).trimmed();
  if (detail.isEmpty() && !readiness.notes.isEmpty())
    detail = readiness.notes.front().trimmed();
  card->detail->setVisible(!detail.isEmpty());
  card->detail->setText(detail);
}

void HomePage::UpdateStatus(const DaemonStatusSnapshot &snapshot) {
  UpdateCard(&cards_[0], snapshot.camera);
  UpdateCard(&cards_[1], snapshot.microphone);
  UpdateCard(&cards_[2], snapshot.speakers);

  const std::vector<RepairIssue> issues = BuildRepairIssues(snapshot);
  if (!snapshot.reachable || !snapshot.parsed || !snapshot.serviceRunning) {
    overallLabel_->setText(snapshot.UserServiceSummary());
    overallDetailLabel_->setText(snapshot.UserServiceDetail());
    SetDynamicProperty(overallLabel_, "scStatus", "error");
  } else if (issues.empty()) {
    overallLabel_->setText(QStringLiteral("No blockers reported"));
    overallDetailLabel_->setText(
        QStringLiteral("Use the device names below in other apps; each card "
                       "shows the current device state."));
    SetDynamicProperty(overallLabel_, "scStatus", "good");
  } else {
    overallLabel_->setText(
        QStringLiteral("%1 item%2 need attention")
            .arg(issues.size())
            .arg(issues.size() == 1 ? QString() : QStringLiteral("s")));
    overallDetailLabel_->setText(
        QStringLiteral("Open the linked page for next steps."));
    SetDynamicProperty(overallLabel_, "scStatus", "warning");
  }

  ClearLayout(repairListLayout_);
  repairGroup_->setVisible(!issues.empty());

  for (const RepairIssue &issue : issues) {
    auto *row = new QFrame(repairGroup_);
    row->setProperty("scRole", "homeRepairItem");
    auto *rowLayout = new QVBoxLayout(row);
    rowLayout->setContentsMargins(12, 12, 12, 12);
    rowLayout->setSpacing(8);

    rowLayout->addWidget(ValueLabel(issue.title, row));
    if (!issue.detail.trimmed().isEmpty())
      rowLayout->addWidget(MutedLabel(issue.detail, row));

    auto *buttons = new QHBoxLayout();
    buttons->setContentsMargins(0, 0, 0, 0);
    buttons->setSpacing(8);
    buttons->addStretch(1);

    auto *copyButton = new QPushButton(QStringLiteral("Copy Details"), row);
    connect(copyButton, &QPushButton::clicked, row,
            [text = issue.copyText] { CopyText(text); });
    buttons->addWidget(copyButton);

    auto *openButton =
        new QPushButton(DestinationAction(issue.destination), row);
    connect(openButton, &QPushButton::clicked, this,
            [this, destination = issue.destination] {
              OpenDestination(destination);
            });
    buttons->addWidget(openButton);

    rowLayout->addLayout(buttons);
    repairListLayout_->addWidget(row);
  }
}

void HomePage::OpenDestination(Destination destination) {
  switch (destination) {
  case Destination::Camera:
    emit CameraRequested();
    break;
  case Destination::Microphone:
    emit MicrophoneRequested();
    break;
  case Destination::Speakers:
    emit SpeakersRequested();
    break;
  case Destination::Engines:
    emit EnginesRequested();
    break;
  case Destination::Support:
    emit SupportRequested();
    break;
  }
}

} // namespace studiocast::gui
