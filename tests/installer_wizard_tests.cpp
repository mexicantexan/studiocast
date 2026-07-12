#include <cstdlib>
#include <iostream>

#include <QApplication>
#include <QCheckBox>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QPlainTextEdit>
#include <QRadioButton>
#include <QTemporaryDir>
#include <QThread>

#include "installer_wizard.h"

namespace {

bool Expect(bool condition, const char *message) {
  if (!condition)
    std::cerr << message << "\n";
  return condition;
}

QJsonObject BaseFacts(const QString &classification = QStringLiteral("absent"),
                      bool supported = true) {
  return {{QStringLiteral("schema_version"), 1},
          {QStringLiteral("facts_version"), QStringLiteral("installer-facts/v1")},
          {QStringLiteral("host_fingerprint"), QStringLiteral("sha256:facts")},
          {QStringLiteral("os"),
           QJsonObject{{QStringLiteral("supported"), supported},
                       {QStringLiteral("pretty_name"), QStringLiteral("Fixture Linux")},
                       {QStringLiteral("base_id"), QStringLiteral("ubuntu")},
                       {QStringLiteral("base_version_id"), QStringLiteral("24.04")}}},
          {QStringLiteral("installation"),
           QJsonObject{{QStringLiteral("classification"), classification},
                       {QStringLiteral("active_version"),
                        classification == QStringLiteral("absent")
                            ? QJsonValue()
                            : QJsonValue(QStringLiteral("0.2.8"))},
                       {QStringLiteral("target_version"), QStringLiteral("0.2.9")},
                       {QStringLiteral("version_relation"),
                        classification == QStringLiteral("absent")
                            ? QStringLiteral("not_installed")
                            : QStringLiteral("upgrade")},
                       {QStringLiteral("desired_configuration"), QJsonObject{}}}}};
}

QJsonObject Status(const QString &classification, const QString &route,
                   const QString &action, bool installed = false) {
  return {{QStringLiteral("schema_version"), 2},
          {QStringLiteral("classification"), classification},
          {QStringLiteral("installed"), installed},
          {QStringLiteral("installed_version"),
           installed ? QStringLiteral("0.2.8") : QString()},
          {QStringLiteral("target_version"), QStringLiteral("0.2.9")},
          {QStringLiteral("version_relation"),
           installed ? QStringLiteral("upgrade")
                     : QStringLiteral("not_installed")},
          {QStringLiteral("route"), route},
          {QStringLiteral("primary_action"), action}};
}

QStringList DefaultPacks() {
  return {QStringLiteral("fastenhancer_s_vd_v1"),
          QStringLiteral("fastenhancer_m_vd_v1"),
          QStringLiteral("modnet-webnn-256-fp32"),
          QStringLiteral("yunet_opencv_zoo_2023mar_fp32"),
          QStringLiteral("dlib_68_ibug_300w"),
          QStringLiteral("gaze_correction_cam_flx_v0_1_1"),
          QStringLiteral("fastdvdnet_sigma15")};
}

QJsonObject Plan(const QString &intent = QStringLiteral("install"),
                 bool blocked = false) {
  QJsonArray packs;
  for (const QString &pack : DefaultPacks())
    packs.append(pack);
  return {{QStringLiteral("schema_version"), 1},
          {QStringLiteral("plan_version"), QStringLiteral("installer-plan/v1")},
          {QStringLiteral("policy_version"),
           QStringLiteral("studiocast-installer-policy/1")},
          {QStringLiteral("plan_id"), QStringLiteral("plan-1")},
          {QStringLiteral("created_at"), QStringLiteral("2098-01-01T00:00:00Z")},
          {QStringLiteral("expires_at"), QStringLiteral("2099-01-01T00:00:00Z")},
          {QStringLiteral("intent"), intent},
          {QStringLiteral("route"), QStringLiteral("recommended")},
          {QStringLiteral("desired_state"),
           QJsonObject{{QStringLiteral("build_type"), QStringLiteral("Release")},
                       {QStringLiteral("virtual_camera"),
                        QJsonObject{{QStringLiteral("desired"), true},
                                    {QStringLiteral("required_for_success"), true}}},
                       {QStringLiteral("model_pack_ids"), packs}}},
          {QStringLiteral("operations"),
           QJsonArray{QJsonObject{{QStringLiteral("id"),
                                  QStringLiteral("preflight.validate")},
                                 {QStringLiteral("privilege"),
                                  QStringLiteral("none")},
                                 {QStringLiteral("review"),
                                  QJsonObject{{QStringLiteral("category"),
                                               QStringLiteral("validation")}}}}}},
          {QStringLiteral("blockers"),
           blocked ? QJsonArray{QJsonObject{{QStringLiteral("code"),
                                            QStringLiteral("fixture.blocker")}}}
                   : QJsonArray{}},
          {QStringLiteral("warnings"), QJsonArray{}},
          {QStringLiteral("plan_digest"), QStringLiteral("sha256:reviewed")},
          {QStringLiteral("approval_token"), QStringLiteral("secret-token")}};
}

QJsonObject Result(const QString &state, bool coreCommitted) {
  return {{QStringLiteral("result_version"),
           QStringLiteral("installer-result/v1")},
          {QStringLiteral("plan_id"), QStringLiteral("plan-1")},
          {QStringLiteral("plan_digest"), QStringLiteral("sha256:reviewed")},
          {QStringLiteral("transaction_id"), QStringLiteral("tx-1")},
          {QStringLiteral("state"), state},
          {QStringLiteral("core_committed"), coreCommitted},
          {QStringLiteral("active_version"),
           coreCommitted ? QStringLiteral("0.2.9") : QStringLiteral("0.2.8")},
          {QStringLiteral("reviewed_operation_ids"),
           QJsonArray{QStringLiteral("preflight.validate")}},
          {QStringLiteral("executed_operation_ids"),
           QJsonArray{QStringLiteral("preflight.validate")}},
          {QStringLiteral("retryable_operation_ids"),
           state == QStringLiteral("degraded")
               ? QJsonArray{QStringLiteral("model.pack.fixture.obtain")}
               : QJsonArray{}},
          {QStringLiteral("error"),
           state == QStringLiteral("failed")
               ? QJsonValue(QJsonObject{{QStringLiteral("code"),
                                         QStringLiteral("fixture.failure")},
                                        {QStringLiteral("message"),
                                         QStringLiteral("fixture failed")}})
               : QJsonValue()}};
}

class FakeBackend {
public:
  FakeBackend() {
    facts_ = dir_.filePath(QStringLiteral("facts.json"));
    status_ = dir_.filePath(QStringLiteral("status.json"));
    plan_ = dir_.filePath(QStringLiteral("plan.json"));
    result_ = dir_.filePath(QStringLiteral("result.json"));
    args_ = dir_.filePath(QStringLiteral("args.txt"));
    archive_ = dir_.filePath(QStringLiteral("official-source.tar.gz"));
    script_ = dir_.filePath(QStringLiteral("backend"));
    QFile script(script_);
    script.open(QIODevice::WriteOnly | QIODevice::Truncate);
    script.write("#!/bin/sh\n"
                 "case \"$1\" in\n"
                 " analyze) cat \"$SC_FAKE_FACTS\";;\n"
                 " status) cat \"$SC_FAKE_STATUS\";;\n"
                 " plan) if [ \"${SC_FAKE_PLAN_FAIL:-0}\" = 1 ]; then echo plan-failed >&2; exit 2; fi; cat \"$SC_FAKE_PLAN\";;\n"
                 " execute-plan) printf '%s\\n' \"$@\" >\"$SC_FAKE_ARGS\"; cat \"$SC_FAKE_RESULT\"; exit \"${SC_FAKE_EXIT:-0}\";;\n"
                 " *) exit 2;;\n"
                 "esac\n");
    script.close();
    script.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner |
                          QFileDevice::ExeOwner);
    QFile archive(archive_);
    archive.open(QIODevice::WriteOnly);
    archive.write("verified fixture archive");
    qputenv("STUDIOCAST_INSTALLER_BACKEND", script_.toLocal8Bit());
    qputenv("SC_FAKE_FACTS", facts_.toLocal8Bit());
    qputenv("SC_FAKE_STATUS", status_.toLocal8Bit());
    qputenv("SC_FAKE_PLAN", plan_.toLocal8Bit());
    qputenv("SC_FAKE_RESULT", result_.toLocal8Bit());
    qputenv("SC_FAKE_ARGS", args_.toLocal8Bit());
    setScenario(BaseFacts(), Status(QStringLiteral("absent"),
                                    QStringLiteral("recommended"),
                                    QStringLiteral("install")),
                Plan(), Result(QStringLiteral("committed"), true), 0);
  }

  ~FakeBackend() {
    qunsetenv("STUDIOCAST_INSTALLER_BACKEND");
    qunsetenv("SC_FAKE_PLAN_FAIL");
    qunsetenv("SC_FAKE_EXIT");
  }

  void setScenario(const QJsonObject &facts, const QJsonObject &status,
                   const QJsonObject &plan, const QJsonObject &result,
                   int exitCode) {
    write(facts_, facts);
    write(status_, status);
    write(plan_, plan);
    write(result_, result);
    qputenv("SC_FAKE_EXIT", QByteArray::number(exitCode));
    qunsetenv("SC_FAKE_PLAN_FAIL");
  }

  void failPlan() { qputenv("SC_FAKE_PLAN_FAIL", "1"); }
  QString arguments() const {
    QFile file(args_);
    file.open(QIODevice::ReadOnly);
    return QString::fromUtf8(file.readAll());
  }
  QString archive() const { return archive_; }

private:
  static void write(const QString &path, const QJsonObject &object) {
    QFile file(path);
    file.open(QIODevice::WriteOnly | QIODevice::Truncate);
    file.write(QJsonDocument(object).toJson(QJsonDocument::Compact));
  }

  QTemporaryDir dir_;
  QString facts_, status_, plan_, result_, args_, script_, archive_;
};

bool WaitForProgress(studiocast::installer::ProgressPage *page) {
  QElapsedTimer timer;
  timer.start();
  while (!page->isComplete() && timer.elapsed() < 3000) {
    QApplication::processEvents();
    QThread::msleep(5);
  }
  return page->isComplete();
}

bool TestStateDrivenRoutes(FakeBackend &backend) {
  bool ok = true;
  struct Scenario {
    QString classification, route, action, label, workflow;
    bool installed;
  };
  const QList<Scenario> scenarios = {
      {QStringLiteral("absent"), QStringLiteral("recommended"),
       QStringLiteral("install"), QStringLiteral("Install recommended"),
       QStringLiteral("install"), false},
      {QStringLiteral("healthy"), QStringLiteral("recommended"),
       QStringLiteral("update"), QStringLiteral("Update recommended"),
       QStringLiteral("update"), true},
      {QStringLiteral("healthy"), QStringLiteral("modify"),
       QStringLiteral("modify"), QStringLiteral("Modify installation"),
       QStringLiteral("modify"), true},
      {QStringLiteral("partial"), QStringLiteral("repair"),
       QStringLiteral("repair"), QStringLiteral("Repair recommended"),
       QStringLiteral("repair"), true},
      {QStringLiteral("tombstone"), QStringLiteral("repair"),
       QStringLiteral("reconstruct"), QStringLiteral("Reconstruct/repair"),
       QStringLiteral("repair"), false},
  };
  for (const Scenario &scenario : scenarios) {
    backend.setScenario(BaseFacts(scenario.classification),
                        Status(scenario.classification, scenario.route,
                               scenario.action, scenario.installed),
                        Plan(scenario.workflow), Result(QStringLiteral("committed"), true), 0);
    studiocast::installer::InstallerWizard wizard;
    auto *intro = wizard.page(studiocast::installer::PageIntro);
    intro->initializePage();
    auto *primary = intro->findChild<QRadioButton *>(
        QStringLiteral("scInstallerPrimaryAction"));
    ok = Expect(primary && primary->text() == scenario.label,
                "primary action should be derived from analyzed state") && ok;
    ok = Expect(wizard.workflow() == scenario.workflow,
                "primary action should select the matching intent") && ok;
  }
  return ok;
}

bool TestRecommendedCustomAndContextualRoutes(FakeBackend &backend) {
  backend.setScenario(BaseFacts(QStringLiteral("healthy")),
                      Status(QStringLiteral("healthy"), QStringLiteral("recommended"),
                             QStringLiteral("update"), true),
                      Plan(QStringLiteral("update")), Result(QStringLiteral("committed"), true), 0);
  studiocast::installer::InstallerWizard wizard;
  wizard.setReleaseArchive(backend.archive());
  wizard.setUseReleaseArchive(true);
  wizard.setStartId(studiocast::installer::PageIntro);
  wizard.restart();
  wizard.show();
  QApplication::processEvents();
  auto *intro = wizard.page(studiocast::installer::PageIntro);
  intro->initializePage();
  auto *customize = intro->findChild<QRadioButton *>(
      QStringLiteral("scInstallerCustomizeAction"));
  auto *repair = intro->findChild<QRadioButton *>(
      QStringLiteral("scInstallerRepairAction"));
  auto *reinstall = intro->findChild<QRadioButton *>(
      QStringLiteral("scInstallerReinstallAction"));
  auto *uninstall = intro->findChild<QRadioButton *>(
      QStringLiteral("scInstallerUninstallAction"));
  bool ok = Expect(wizard.nextId() == studiocast::installer::PageReview,
                   "recommended route should go directly to review") &&
            Expect(customize && !customize->isHidden(),
                   "healthy state should expose Customize") &&
            Expect(repair && !repair->isHidden(),
                   "healthy state should expose Repair") &&
            Expect(reinstall && !reinstall->isHidden(),
                   "healthy state should expose Reinstall") &&
            Expect(uninstall && !uninstall->isHidden(),
                   "healthy state should expose Uninstall");
  customize->setChecked(true);
  ok = Expect(wizard.customRoute() &&
                  wizard.nextId() == studiocast::installer::PageDependencyPlan,
              "Customize should enter the reused Custom/Advanced pages") && ok;
  reinstall->setChecked(true);
  ok = Expect(wizard.nextId() == studiocast::installer::PageServiceOptions,
              "Reinstall should expose preservation before review") && ok;
  auto *service = wizard.page(studiocast::installer::PageServiceOptions);
  service->initializePage();
  auto *preserve = service->findChild<QCheckBox *>(
      QStringLiteral("scInstallerPreserveSettings"));
  ok = Expect(preserve && preserve->isChecked(),
              "Reinstall should default Preserve settings on") && ok;
  const auto allChecks = wizard.findChildren<QCheckBox *>();
  for (const QCheckBox *check : allChecks)
    ok = Expect(!check->text().contains(QStringLiteral("system-wide"),
                                        Qt::CaseInsensitive),
                "GUI must not expose system-wide installation") && ok;
  return ok;
}

bool TestUnsupportedOfflineAndPriorReuse(FakeBackend &backend) {
  QJsonObject unsupportedFacts = BaseFacts(QStringLiteral("absent"), false);
  backend.setScenario(unsupportedFacts,
                      Status(QStringLiteral("absent"), QStringLiteral("unsupported"),
                             QStringLiteral("manual_instructions")),
                      Plan(), Result(QStringLiteral("failed"), false), 2);
  studiocast::installer::InstallerWizard unsupported;
  auto *intro = unsupported.page(studiocast::installer::PageIntro);
  intro->initializePage();
  bool ok = Expect(!intro->isComplete(),
                   "unsupported OS should fail closed without GUI override");

  backend.setScenario(BaseFacts(),
                      Status(QStringLiteral("absent"), QStringLiteral("offline"),
                             QStringLiteral("stop")),
                      Plan(), Result(QStringLiteral("failed"), false), 2);
  studiocast::installer::InstallerWizard offline;
  offline.page(studiocast::installer::PageIntro)->initializePage();
  ok = Expect(!offline.page(studiocast::installer::PageIntro)->isComplete(),
              "offline cache miss should block mutation") && ok;

  QJsonObject facts = BaseFacts(QStringLiteral("healthy"));
  QJsonObject installation = facts.value(QStringLiteral("installation")).toObject();
  installation[QStringLiteral("desired_configuration")] =
      QJsonObject{{QStringLiteral("build_type"), QStringLiteral("Debug")},
                  {QStringLiteral("features"),
                   QJsonObject{{QStringLiteral("open_cuda"), false},
                               {QStringLiteral("open_audio"), true},
                               {QStringLiteral("open_vulkan"), false}}},
                  {QStringLiteral("service"),
                   QJsonObject{{QStringLiteral("desired"), QStringLiteral("disabled")}}},
                  {QStringLiteral("v4l"),
                   QJsonObject{{QStringLiteral("desired"), false},
                               {QStringLiteral("persist"), false}}},
                  {QStringLiteral("model_pack_ids"), QJsonArray{}},
                  {QStringLiteral("preserve_settings"), true}};
  facts[QStringLiteral("installation")] = installation;
  backend.setScenario(facts,
                      Status(QStringLiteral("healthy"), QStringLiteral("recommended"),
                             QStringLiteral("update"), true),
                      Plan(QStringLiteral("update")), Result(QStringLiteral("committed"), true), 0);
  studiocast::installer::InstallerWizard prior;
  ok = Expect(prior.buildType() == QStringLiteral("Debug") &&
                  !prior.installService() && !prior.configureV4l2Loopback() &&
                  !prior.installModels() && !prior.openVulkan(),
              "update should seed controls from prior desired configuration") && ok;
  return ok;
}

bool TestPlanValidationExactExecutionAndLabels(FakeBackend &backend) {
  backend.setScenario(BaseFacts(),
                      Status(QStringLiteral("absent"), QStringLiteral("recommended"),
                             QStringLiteral("install")),
                      Plan(), Result(QStringLiteral("committed"), true), 0);
  studiocast::installer::InstallerWizard wizard;
  wizard.setReleaseArchive(backend.archive());
  wizard.setUseReleaseArchive(true);
  auto *review = wizard.page(studiocast::installer::PageReview);
  review->initializePage();
  QStringList args = wizard.executionCommandArguments();
  bool ok = Expect(review->isComplete() && wizard.planReady(),
                   "valid blocker-free plan should enable Apply") &&
            Expect(args.size() == 7 && args.at(0) == QStringLiteral("execute-plan") &&
                       args.at(1) == QStringLiteral("--plan") &&
                       args.at(3) == QStringLiteral("--digest") &&
                       args.at(4) == QStringLiteral("sha256:reviewed") &&
                       args.at(5) == QStringLiteral("--token") &&
                       args.at(6) == QStringLiteral("secret-token"),
                   "execution must consume only exact reviewed plan/digest/token") &&
            Expect(wizard.buttonText(QWizard::CommitButton) ==
                       QStringLiteral("Install"),
                   "Install review action must not be labeled Finish");
  QFile planFile(wizard.reviewedPlanPath());
  ok = Expect(planFile.permissions().testFlag(QFileDevice::ReadOwner) &&
                  !planFile.permissions().testFlag(QFileDevice::ReadGroup) &&
                  !planFile.permissions().testFlag(QFileDevice::ReadOther),
              "reviewed plan should be persisted privately") && ok;
  const auto *reviewText = review->findChild<QPlainTextEdit *>();
  ok = Expect(reviewText && reviewText->toPlainText().contains(
                                  QStringLiteral("Models/downloads (7 pack IDs)")) &&
                  reviewText->toPlainText().contains(
                      QStringLiteral("8 verified artifact files")),
              "review should list exact seven packs and eight artifacts") && ok;

  auto *progress = dynamic_cast<studiocast::installer::ProgressPage *>(
      wizard.page(studiocast::installer::PageProgress));
  progress->initializePage();
  ok = Expect(WaitForProgress(progress), "fake exact-plan execution should finish") && ok;
  const QString executed = backend.arguments();
  ok = Expect(executed.startsWith(QStringLiteral("execute-plan\n--plan\n")) &&
                  !executed.contains(QStringLiteral("--source-dir")) &&
                  !executed.contains(QStringLiteral("--v4l2loopback")),
              "executor args must not reconstruct reviewed selection flags") && ok;

  backend.setScenario(BaseFacts(),
                      Status(QStringLiteral("absent"), QStringLiteral("recommended"),
                             QStringLiteral("install")),
                      Plan(QStringLiteral("install"), true),
                      Result(QStringLiteral("failed"), false), 2);
  studiocast::installer::InstallerWizard blocked;
  blocked.setReleaseArchive(backend.archive());
  blocked.setUseReleaseArchive(true);
  auto *blockedReview = blocked.page(studiocast::installer::PageReview);
  blockedReview->initializePage();
  ok = Expect(!blockedReview->isComplete() && !blocked.planReady(),
              "plan blocker must disable Apply") && ok;
  backend.failPlan();
  studiocast::installer::InstallerWizard failedPlan;
  failedPlan.setReleaseArchive(backend.archive());
  failedPlan.setUseReleaseArchive(true);
  auto *failedReview = failedPlan.page(studiocast::installer::PageReview);
  failedReview->initializePage();
  ok = Expect(!failedReview->isComplete(),
              "plan generation failure must disable Apply") && ok;
  qunsetenv("SC_FAKE_PLAN_FAIL");
  const QList<QPair<QString, QString>> labels = {
      {QStringLiteral("update"), QStringLiteral("Update")},
      {QStringLiteral("repair"), QStringLiteral("Repair")},
      {QStringLiteral("reinstall"), QStringLiteral("Reinstall")},
      {QStringLiteral("uninstall"), QStringLiteral("Uninstall")}};
  for (const auto &entry : labels) {
    backend.setScenario(BaseFacts(QStringLiteral("healthy")),
                        Status(QStringLiteral("healthy"),
                               QStringLiteral("recommended"), entry.first,
                               true),
                        Plan(entry.first), Result(QStringLiteral("committed"), true), 0);
    studiocast::installer::InstallerWizard labeled;
    labeled.setReleaseArchive(backend.archive());
    labeled.setUseReleaseArchive(true);
    labeled.setWorkflow(entry.first);
    QWizardPage *page =
        entry.first == QStringLiteral("uninstall")
            ? labeled.page(studiocast::installer::PageUninstall)
            : labeled.page(studiocast::installer::PageReview);
    page->initializePage();
    ok = Expect(labeled.buttonText(QWizard::CommitButton) == entry.second,
                "commit label should name the planned action") && ok;
  }
  return ok;
}

bool TestFailureAndDegradedCompletionTruth(FakeBackend &backend) {
  bool ok = true;
  for (const auto &scenario :
       QList<QPair<QString, int>>{{QStringLiteral("failed"), 2},
                                  {QStringLiteral("degraded"), 3}}) {
    backend.setScenario(BaseFacts(QStringLiteral("healthy")),
                        Status(QStringLiteral("healthy"), QStringLiteral("recommended"),
                               QStringLiteral("update"), true),
                        Plan(QStringLiteral("update")),
                        Result(scenario.first,
                               scenario.first == QStringLiteral("degraded")),
                        scenario.second);
    studiocast::installer::InstallerWizard wizard;
    wizard.setReleaseArchive(backend.archive());
    wizard.setUseReleaseArchive(true);
    wizard.setWorkflow(QStringLiteral("update"));
    wizard.page(studiocast::installer::PageReview)->initializePage();
    auto *progress = dynamic_cast<studiocast::installer::ProgressPage *>(
        wizard.page(studiocast::installer::PageProgress));
    progress->initializePage();
    ok = Expect(WaitForProgress(progress),
                "structured terminal failure/degraded result should finish") && ok;
    ok = Expect(wizard.executionResult().value(QStringLiteral("state")).toString() ==
                    scenario.first,
                "structured terminal state must be preserved") && ok;
    auto *finish = wizard.page(studiocast::installer::PageFinish);
    finish->initializePage();
    auto *summary = finish->findChild<QLabel *>(
        QStringLiteral("scInstallerCompletionSummary"));
    ok = Expect(summary != nullptr, "completion should expose result summary") && ok;
    if (scenario.first == QStringLiteral("failed")) {
      ok = Expect(finish->title() == QStringLiteral("Action failed") &&
                      !summary->text().contains(QStringLiteral("is installed")),
                  "old installed status must not mask update failure") && ok;
    } else {
      ok = Expect(finish->title() == QStringLiteral("Complete with limitations") &&
                      summary->text().contains(QStringLiteral("retry"),
                                               Qt::CaseInsensitive),
                  "degraded completion should offer accurate Retry/Continue guidance") && ok;
    }
  }
  return ok;
}

} // namespace

int main(int argc, char **argv) {
  (void)::setenv("QT_QPA_PLATFORM", "offscreen", 1);
  QApplication app(argc, argv);
  FakeBackend backend;
  bool ok = true;
  ok = TestStateDrivenRoutes(backend) && ok;
  ok = TestRecommendedCustomAndContextualRoutes(backend) && ok;
  ok = TestUnsupportedOfflineAndPriorReuse(backend) && ok;
  ok = TestPlanValidationExactExecutionAndLabels(backend) && ok;
  ok = TestFailureAndDegradedCompletionTruth(backend) && ok;
  return ok ? 0 : 1;
}
