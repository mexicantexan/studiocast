#include <cerrno>
#include <csignal>
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
#include <QPushButton>
#include <QRadioButton>
#include <QTemporaryDir>
#include <QThread>

#ifdef Q_OS_UNIX
#include <unistd.h>
#endif

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
  return {
      {QStringLiteral("schema_version"), 2},
      {QStringLiteral("classification"), classification},
      {QStringLiteral("installed"), installed},
      {QStringLiteral("installed_version"),
       installed ? QStringLiteral("0.2.8") : QString()},
      {QStringLiteral("target_version"), QStringLiteral("0.2.9")},
      {QStringLiteral("version_relation"),
       installed ? QStringLiteral("upgrade") : QStringLiteral("not_installed")},
      {QStringLiteral("route"), route},
      {QStringLiteral("primary_action"), action},
      {QStringLiteral("stable_release"),
       QJsonObject{{QStringLiteral("verified"), true},
                   {QStringLiteral("channel"), QStringLiteral("stable")},
                   {QStringLiteral("version"), QStringLiteral("0.2.9")}}}};
}

QJsonObject ReleaseStatus(const QString &archive, bool offline = false) {
  return {
      {QStringLiteral("schema_version"), 1},
      {QStringLiteral("release_status_version"),
       QStringLiteral("installer-release-status/v1")},
      {QStringLiteral("channel"), QStringLiteral("stable")},
      {QStringLiteral("available_version"), QStringLiteral("0.2.9")},
      {QStringLiteral("installed_version"), QJsonValue()},
      {QStringLiteral("action"), QStringLiteral("install")},
      {QStringLiteral("relation"), QStringLiteral("not_installed")},
      {QStringLiteral("reason_code"), QStringLiteral("release.install")},
      {QStringLiteral("minimum_installer_version"), QStringLiteral("0.2.9")},
      {QStringLiteral("installer_meets_minimum"), true},
      {QStringLiteral("offline"), offline},
      {QStringLiteral("verified_source_archive"), archive}};
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
    release_ = dir_.filePath(QStringLiteral("release.json"));
    args_ = dir_.filePath(QStringLiteral("args.txt"));
    planArgs_ = dir_.filePath(QStringLiteral("plan-args.txt"));
    releaseCalls_ = dir_.filePath(QStringLiteral("release-calls.txt"));
    childPid_ = dir_.filePath(QStringLiteral("child.pid"));
    archive_ = dir_.filePath(QStringLiteral("official-source.tar.gz"));
    script_ = dir_.filePath(QStringLiteral("backend"));
    QFile script(script_);
    script.open(QIODevice::WriteOnly | QIODevice::Truncate);
    script.write(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        " analyze) cat \"$SC_FAKE_FACTS\";;\n"
        " status) cat \"$SC_FAKE_STATUS\";;\n"
        " release-status) printf '%s\\n' \"$@\" >>\"$SC_FAKE_RELEASE_CALLS\"; "
        "offline=0; receipt=; previous=; for argument in \"$@\"; do if [ "
        "\"$previous\" = receipt ]; then receipt=$argument; previous=; elif [ "
        "\"$argument\" = --release-receipt-out ]; then previous=receipt; elif "
        "[ \"$argument\" = --offline ]; then offline=1; fi; done; if [ "
        "\"$offline\" = 0 ] && [ \"${SC_FAKE_RELEASE_FAIL_ONLINE:-0}\" = 1 ]; "
        "then echo online-release-failed >&2; exit 2; fi; if [ \"$offline\" = "
        "1 ] && [ \"${SC_FAKE_RELEASE_FAIL_OFFLINE:-0}\" = 1 ]; then echo "
        "offline-cache-miss >&2; exit 2; fi; printf "
        "'{\"schema_version\":1,\"receipt_version\":\"studiocast-verified-"
        "release/v1\"}\\n' >\"$receipt\"; cat \"$SC_FAKE_RELEASE\";;\n"
        " plan) printf '%s\\n' \"$@\" >\"$SC_FAKE_PLAN_ARGS\"; if [ "
        "\"${SC_FAKE_PLAN_FAIL:-0}\" = 1 ]; then echo plan-failed >&2; exit 2; "
        "fi; cat \"$SC_FAKE_PLAN\";;\n"
        " execute-plan) printf '%s\\n' \"$@\" >\"$SC_FAKE_ARGS\"; if [ "
        "\"${SC_FAKE_CANCEL_CHILD:-0}\" = 1 ]; then trap 'kill \"$child\" "
        "2>/dev/null || true; wait \"$child\" 2>/dev/null || true; cat "
        "\"$SC_FAKE_RESULT\"; exit 130' INT TERM; sleep 30 & child=$!; printf "
        "'%s\\n' \"$child\" >\"$SC_FAKE_CHILD_PID\"; wait \"$child\"; fi; if [ "
        "\"${SC_FAKE_PROGRESS:-0}\" = 1 ]; then printf "
        "'{\"event_version\":\"installer-progress/"
        "v1\",\"plan_id\":\"plan-1\",\"plan_digest\":\"sha256:reviewed\","
        "\"phase\":\"payload\",\"operation_id\":\"preflight.validate\","
        "\"state\":\"running\"}\\n'; sleep \"${SC_FAKE_PROGRESS_HOLD:-0}\"; "
        "fi; cat \"$SC_FAKE_RESULT\"; exit \"${SC_FAKE_EXIT:-0}\";;\n"
        " *) exit 2;;\n"
        "esac\n");
    script.close();
    script.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner |
                          QFileDevice::ExeOwner);
    QFile archive(archive_);
    archive.open(QIODevice::WriteOnly);
    archive.write("verified fixture archive");
    archive.close();
    write(release_, ReleaseStatus(archive_));
    qputenv("STUDIOCAST_INSTALLER_BACKEND", script_.toLocal8Bit());
    qputenv("SC_FAKE_FACTS", facts_.toLocal8Bit());
    qputenv("SC_FAKE_STATUS", status_.toLocal8Bit());
    qputenv("SC_FAKE_PLAN", plan_.toLocal8Bit());
    qputenv("SC_FAKE_RESULT", result_.toLocal8Bit());
    qputenv("SC_FAKE_RELEASE", release_.toLocal8Bit());
    qputenv("SC_FAKE_ARGS", args_.toLocal8Bit());
    qputenv("SC_FAKE_PLAN_ARGS", planArgs_.toLocal8Bit());
    qputenv("SC_FAKE_RELEASE_CALLS", releaseCalls_.toLocal8Bit());
    qputenv("SC_FAKE_CHILD_PID", childPid_.toLocal8Bit());
    qputenv("STUDIOCAST_STABLE_CHANNEL_URL",
            "https://fixture.invalid/manifest.json");
    qputenv("STUDIOCAST_STABLE_CHANNEL_SIGNATURE_URL",
            "https://fixture.invalid/manifest.json.sig");
    qputenv("STUDIOCAST_RELEASE_CACHE_DIR",
            dir_.filePath(QStringLiteral("release-cache")).toLocal8Bit());
    setScenario(BaseFacts(), Status(QStringLiteral("absent"),
                                    QStringLiteral("recommended"),
                                    QStringLiteral("install")),
                Plan(), Result(QStringLiteral("committed"), true), 0);
  }

  ~FakeBackend() {
    qunsetenv("STUDIOCAST_INSTALLER_BACKEND");
    qunsetenv("SC_FAKE_PLAN_FAIL");
    qunsetenv("SC_FAKE_EXIT");
    qunsetenv("SC_FAKE_RELEASE_FAIL_ONLINE");
    qunsetenv("SC_FAKE_RELEASE_FAIL_OFFLINE");
    qunsetenv("SC_FAKE_CANCEL_CHILD");
    qunsetenv("SC_FAKE_PROGRESS");
    qunsetenv("SC_FAKE_PROGRESS_HOLD");
    qunsetenv("STUDIOCAST_STABLE_CHANNEL_URL");
    qunsetenv("STUDIOCAST_STABLE_CHANNEL_SIGNATURE_URL");
    qunsetenv("STUDIOCAST_RELEASE_CACHE_DIR");
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
  void setReleaseFailures(bool online, bool offline) {
    qputenv("SC_FAKE_RELEASE_FAIL_ONLINE", online ? "1" : "0");
    qputenv("SC_FAKE_RELEASE_FAIL_OFFLINE", offline ? "1" : "0");
  }
  void enableCancellationChild(bool enabled) {
    qputenv("SC_FAKE_CANCEL_CHILD", enabled ? "1" : "0");
  }
  void enableProgressEvent(bool enabled, const QByteArray &hold = "0") {
    qputenv("SC_FAKE_PROGRESS", enabled ? "1" : "0");
    qputenv("SC_FAKE_PROGRESS_HOLD", hold);
  }
  QString arguments() const {
    QFile file(args_);
    file.open(QIODevice::ReadOnly);
    return QString::fromUtf8(file.readAll());
  }
  QString archive() const { return archive_; }
  QString planArguments() const { return read(planArgs_); }
  QString releaseCalls() const { return read(releaseCalls_); }
  qint64 childPid() const { return read(childPid_).trimmed().toLongLong(); }

private:
  static void write(const QString &path, const QJsonObject &object) {
    QFile file(path);
    file.open(QIODevice::WriteOnly | QIODevice::Truncate);
    file.write(QJsonDocument(object).toJson(QJsonDocument::Compact));
  }

  static QString read(const QString &path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
      return {};
    return QString::fromUtf8(file.readAll());
  }

  QTemporaryDir dir_;
  QString facts_, status_, plan_, result_, release_, args_, planArgs_,
      releaseCalls_, childPid_, script_, archive_;
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

bool TestSignedReleaseAndOfflineReceipt(FakeBackend &backend) {
  backend.setScenario(BaseFacts(),
                      Status(QStringLiteral("absent"),
                             QStringLiteral("recommended"),
                             QStringLiteral("install")),
                      Plan(), Result(QStringLiteral("committed"), true), 0);
  backend.setReleaseFailures(false, false);
  studiocast::installer::InstallerWizard online;
  online.page(studiocast::installer::PageReview)->initializePage();
  const QString onlinePlanArgs = backend.planArguments();
  bool ok =
      Expect(online.planReady() &&
                 !online.verifiedReleaseReceiptPath().isEmpty(),
             "Recommended should require verified signed-release evidence") &&
      Expect(
          onlinePlanArgs.contains(QStringLiteral("--release-receipt")) &&
              !onlinePlanArgs.contains(QStringLiteral("--official-source")) &&
              !onlinePlanArgs.contains(QStringLiteral("--release-archive")),
          "Recommended planning must use a reverified receipt, not an "
          "official-source claim") &&
      Expect(backend.releaseCalls().contains(
                 QStringLiteral("https://fixture.invalid/manifest.json")) &&
                 backend.releaseCalls().contains(QStringLiteral(
                     "https://fixture.invalid/manifest.json.sig")),
             "GUI should pass the configurable stable manifest endpoints");

  backend.setReleaseFailures(true, false);
  studiocast::installer::InstallerWizard cached;
  cached.page(studiocast::installer::PageReview)->initializePage();
  ok =
      Expect(
          cached.planReady() &&
              backend.releaseCalls().contains(QStringLiteral("--offline")) &&
              backend.releaseCalls().contains(QStringLiteral("release-cache")),
          "Verified offline cache should remain usable after online failure") &&
      ok;

  backend.setReleaseFailures(true, true);
  studiocast::installer::InstallerWizard unavailable;
  auto *intro = unavailable.page(studiocast::installer::PageIntro);
  intro->initializePage();
  ok = Expect(!intro->isComplete() &&
                  unavailable.detectedRoute() == QStringLiteral("offline"),
              "Missing online and offline signed evidence must fail closed") &&
       ok;
  unavailable.page(studiocast::installer::PageReview)->initializePage();
  ok = Expect(!unavailable.planReady(),
              "Signed-release failure must block Apply before mutation") &&
       ok;
  backend.setReleaseFailures(false, false);
  return ok;
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
  ok = Expect(backend.planArguments().contains(
                  QStringLiteral("--release-receipt")) &&
                  !backend.planArguments().contains(
                      QStringLiteral("--official-source")),
              "reviewed plan must be built from signed receipt evidence") &&
       ok;

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
      auto *retry = finish->findChild<QPushButton *>(
          QStringLiteral("scInstallerDegradedRetry"));
      auto *continueButton = finish->findChild<QPushButton *>(
          QStringLiteral("scInstallerDegradedContinue"));
      ok = Expect(retry && !retry->isHidden() && retry->isEnabled() &&
                      continueButton && !continueButton->isHidden(),
                  "degraded completion must expose dedicated Retry and "
                  "Continue controls") &&
           ok;
      continueButton->click();
      ok = Expect(wizard.result() == QDialog::Accepted,
                  "Continue should explicitly accept the degraded outcome") &&
           ok;
    }
  }
  return ok;
}

bool TestStructuredProgressAndDegradedRetry(FakeBackend &backend) {
  backend.setScenario(BaseFacts(QStringLiteral("healthy")),
                      Status(QStringLiteral("healthy"),
                             QStringLiteral("recommended"),
                             QStringLiteral("update"), true),
                      Plan(QStringLiteral("update")),
                      Result(QStringLiteral("degraded"), true), 3);
  backend.enableProgressEvent(true, "0.25");
  studiocast::installer::InstallerWizard wizard;
  wizard.setWorkflow(QStringLiteral("update"));
  wizard.page(studiocast::installer::PageReview)->initializePage();
  auto *progress = dynamic_cast<studiocast::installer::ProgressPage *>(
      wizard.page(studiocast::installer::PageProgress));
  progress->initializePage();
  auto *phase =
      progress->findChild<QLabel *>(QStringLiteral("scInstallerProgressPhase"));
  QElapsedTimer timer;
  timer.start();
  while (phase && !phase->text().contains(QStringLiteral("Phase:")) &&
         timer.elapsed() < 1000) {
    QApplication::processEvents();
    QThread::msleep(5);
  }
  bool ok =
      Expect(phase && phase->text().contains(QStringLiteral("payload")) &&
                 phase->text().contains(QStringLiteral("preflight.validate")),
             "bound backend events should drive structured progress phases") &&
      Expect(WaitForProgress(progress),
             "structured progress fixture should finish");
  auto *finish = wizard.page(studiocast::installer::PageFinish);
  finish->initializePage();
  auto *retry = finish->findChild<QPushButton *>(
      QStringLiteral("scInstallerDegradedRetry"));
  retry->click();
  QApplication::processEvents();
  ok =
      Expect(
          wizard.workflow() == QStringLiteral("repair") &&
              wizard.currentId() == studiocast::installer::PageReview &&
              wizard.planReady(),
          "Retry should re-analyze and present a new repair plan for review") &&
      ok;
  backend.enableProgressEvent(false);
  return ok;
}

bool TestCancellationCleansChild(FakeBackend &backend) {
#ifdef Q_OS_UNIX
  backend.setScenario(BaseFacts(),
                      Status(QStringLiteral("absent"),
                             QStringLiteral("recommended"),
                             QStringLiteral("install")),
                      Plan(), Result(QStringLiteral("cancelled"), false), 130);
  backend.enableCancellationChild(true);
  studiocast::installer::InstallerWizard wizard;
  wizard.page(studiocast::installer::PageReview)->initializePage();
  auto *progress = dynamic_cast<studiocast::installer::ProgressPage *>(
      wizard.page(studiocast::installer::PageProgress));
  progress->initializePage();
  QElapsedTimer timer;
  timer.start();
  while (backend.childPid() <= 0 && timer.elapsed() < 1000) {
    QApplication::processEvents();
    QThread::msleep(5);
  }
  const qint64 child = backend.childPid();
  bool ok =
      Expect(child > 0, "cancellation fixture should start a child process");
  progress->requestCancellation();
  ok = Expect(WaitForProgress(progress),
              "cancelled backend should reach a structured terminal result") &&
       ok;
  timer.restart();
  while (::kill(static_cast<pid_t>(child), 0) == 0 && timer.elapsed() < 1000) {
    QThread::msleep(5);
  }
  errno = 0;
  const int probe = ::kill(static_cast<pid_t>(child), 0);
  ok = Expect(probe == -1 && errno == ESRCH,
              "cancellation must not leave the backend child running") &&
       ok;
  ok = Expect(
           wizard.executionResult().value(QStringLiteral("state")).toString() ==
               QStringLiteral("cancelled"),
           "cancellation should preserve the backend's structured result") &&
       ok;
  backend.enableCancellationChild(false);
  return ok;
#else
  Q_UNUSED(backend);
  return true;
#endif
}

} // namespace

int main(int argc, char **argv) {
  (void)::setenv("QT_QPA_PLATFORM", "offscreen", 1);
  QApplication app(argc, argv);
  FakeBackend backend;
  bool ok = true;
  ok = TestSignedReleaseAndOfflineReceipt(backend) && ok;
  ok = TestStateDrivenRoutes(backend) && ok;
  ok = TestRecommendedCustomAndContextualRoutes(backend) && ok;
  ok = TestUnsupportedOfflineAndPriorReuse(backend) && ok;
  ok = TestPlanValidationExactExecutionAndLabels(backend) && ok;
  ok = TestFailureAndDegradedCompletionTruth(backend) && ok;
  ok = TestStructuredProgressAndDegradedRetry(backend) && ok;
  ok = TestCancellationCleansChild(backend) && ok;
  return ok ? 0 : 1;
}
