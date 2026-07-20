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
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QRegularExpression>
#include <QSpinBox>
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

QJsonObject ReleaseStatus(const QString &archive = {}, bool offline = false) {
  QJsonObject status = {
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
      {QStringLiteral("source_archive_state"),
       offline ? QStringLiteral("candidate_verified")
               : QStringLiteral("acquisition_pending")}};
  if (!archive.isEmpty())
    status[QStringLiteral("verified_source_archive")] = archive;
  return status;
}

QJsonObject
SelfUpdateStatus(const QString &currentAppImage,
                 const QString &verifiedAppImage, const QString &manualUrl,
                 const QString &state = QStringLiteral("offer_restart")) {
  QJsonObject status = ReleaseStatus();
  status[QStringLiteral("self_update")] = QJsonObject{
      {QStringLiteral("state"), state},
      {QStringLiteral("reason_code"),
       state == QStringLiteral("offer_restart")
           ? QStringLiteral("self_update.verified_offer")
           : QStringLiteral("self_update.appimage_not_detected")},
      {QStringLiteral("current_appimage"), currentAppImage},
      {QStringLiteral("verified_appimage"), verifiedAppImage},
      {QStringLiteral("relaunch_command"),
       verifiedAppImage.isEmpty() ? QJsonValue(QJsonArray{})
                                  : QJsonValue(QJsonArray{verifiedAppImage})},
      {QStringLiteral("manual_download_url"), manualUrl},
      {QStringLiteral("requires_confirmation"),
       state == QStringLiteral("offer_restart")},
      {QStringLiteral("running_artifact_replaced"), false}};
  return status;
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
  const QList<qint64> sizes = {832775, 2033628, 25888640, 232589,
                               99693937, 1062936, 1062994, 416788};
  const QStringList artifacts = {
      QStringLiteral("fastenhancer_s_vd_v1:model.onnx"),
      QStringLiteral("fastenhancer_m_vd_v1:model.onnx"),
      QStringLiteral("modnet-webnn-256-fp32:model.onnx"),
      QStringLiteral("yunet_opencv_zoo_2023mar_fp32:model.onnx"),
      QStringLiteral("dlib_68_ibug_300w:shape_predictor_68_face_landmarks.dat"),
      QStringLiteral("gaze_correction_cam_flx_v0_1_1:gaze_flx_left.onnx"),
      QStringLiteral("gaze_correction_cam_flx_v0_1_1:gaze_flx_right.onnx"),
      QStringLiteral("fastdvdnet_sigma15:model.onnx")};
  QJsonArray downloads;
  for (qsizetype index = 0; index < artifacts.size(); ++index) {
    const QString artifactId = artifacts.at(index);
    downloads.append(QJsonObject{
        {QStringLiteral("artifact_id"), artifactId},
        {QStringLiteral("pack_id"), artifactId.section(QLatin1Char(':'), 0, 0)},
        {QStringLiteral("size"), sizes.at(index)}});
  }
  const qint64 modelBytes = 131224287;
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
          {QStringLiteral("downloads"), downloads},
          {QStringLiteral("disk"),
           QJsonObject{{QStringLiteral("free_bytes"), 20000000000LL},
                       {QStringLiteral("base_required_bytes"), 3},
                       {QStringLiteral("download_bytes"), modelBytes},
                       {QStringLiteral("required_bytes"), modelBytes + 3}}},
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
    write(release_, ReleaseStatus());
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
    qunsetenv("STUDIOCAST_INSTALLER_OFFLINE");
    qunsetenv("APPIMAGE");
    qunsetenv("STUDIOCAST_INSTALLER_TEST_CONFIRM_SELF_UPDATE");
    qunsetenv("STUDIOCAST_INSTALLER_TEST_SELF_UPDATE_LAUNCH_LOG");
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
  void setReleaseStatus(const QJsonObject &status) { write(release_, status); }
  QString arguments() const {
    QFile file(args_);
    file.open(QIODevice::ReadOnly);
    return QString::fromUtf8(file.readAll());
  }
  QString archive() const { return archive_; }
  QString planArguments() const { return read(planArgs_); }
  QString releaseCalls() const { return read(releaseCalls_); }
  void clearReleaseCalls() { QFile::remove(releaseCalls_); }
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

bool TestExplicitOfflineUsesOnlyVerifiedCache(FakeBackend &backend) {
  QJsonObject facts = BaseFacts();
  facts[QStringLiteral("connectivity")] = QJsonObject{
      {QStringLiteral("release_source"),
       QJsonObject{{QStringLiteral("state"), QStringLiteral("offline")}}}};
  facts[QStringLiteral("cache")] = QJsonObject{
      {QStringLiteral("release_artifacts"),
       QJsonObject{{QStringLiteral("0.2.9"),
                    QJsonObject{{QStringLiteral("verified"), true}}}}}};
  backend.setScenario(facts,
                      Status(QStringLiteral("absent"),
                             QStringLiteral("recommended"),
                             QStringLiteral("install")),
                      Plan(), Result(QStringLiteral("committed"), true), 0);
  backend.setReleaseStatus(ReleaseStatus(backend.archive(), true));
  backend.setReleaseFailures(false, false);
  backend.clearReleaseCalls();
  qputenv("STUDIOCAST_INSTALLER_OFFLINE", "1");

  studiocast::installer::InstallerWizard offline;
  offline.page(studiocast::installer::PageReview)->initializePage();
  const QString releaseCalls = backend.releaseCalls();
  const QString planArgs = backend.planArguments();
  bool ok =
      Expect(releaseCalls.count(QStringLiteral("release-status\n")) == 1 &&
                 releaseCalls.contains(QStringLiteral("--offline")),
             "explicit offline mode must issue one offline release-status "
             "request and never probe online") &&
      Expect(offline.planReady() &&
                 planArgs.contains(QStringLiteral("--offline")) &&
                 planArgs.contains(QStringLiteral("--release-receipt")),
             "verified offline status should permit planning and bind offline "
             "plus the signed receipt into the reviewed arguments");

  backend.setScenario(facts,
                      Status(QStringLiteral("absent"),
                             QStringLiteral("offline"), QStringLiteral("stop")),
                      Plan(), Result(QStringLiteral("failed"), false), 2);
  backend.setReleaseFailures(false, true);
  backend.clearReleaseCalls();
  studiocast::installer::InstallerWizard missingCache;
  missingCache.page(studiocast::installer::PageReview)->initializePage();
  const QString missingCalls = backend.releaseCalls();
  ok = Expect(missingCalls.count(QStringLiteral("release-status\n")) == 1 &&
                  missingCalls.contains(QStringLiteral("--offline")) &&
                  !missingCache.planReady(),
              "an explicit offline cache miss must stop before planning "
              "without an online fallback") &&
       ok;

  qunsetenv("STUDIOCAST_INSTALLER_OFFLINE");
  backend.setReleaseFailures(false, false);
  backend.setReleaseStatus(ReleaseStatus());
  return ok;
}

bool TestCustomRetainsVerifiedOfficialSource(FakeBackend &backend) {
  backend.setScenario(BaseFacts(),
                      Status(QStringLiteral("absent"),
                             QStringLiteral("recommended"),
                             QStringLiteral("install")),
                      Plan(), Result(QStringLiteral("committed"), true), 0);
  studiocast::installer::InstallerWizard custom;
  custom.setCustomRoute(true);
  custom.setReleaseArchive(backend.archive());
  auto *build = custom.page(studiocast::installer::PageBuildOptions);
  build->initializePage();
  auto *official = build->findChild<QRadioButton *>(
      QStringLiteral("scInstallerVerifiedOfficialSource"));
  auto *advancedArchive = build->findChild<QRadioButton *>(
      QStringLiteral("scInstallerAdvancedSourceArchive"));
  bool ok =
      Expect(official && official->isChecked() && build->validatePage(),
             "Customize should begin with the verified official source") &&
      Expect(custom.backendOptions(true).contains(
                 QStringLiteral("--release-receipt")) &&
                 custom.backendOptions(true).value(
                     custom.backendOptions(true).indexOf(
                         QStringLiteral("--route")) +
                     1) == QStringLiteral("custom") &&
                 !custom.backendOptions(true).contains(
                     QStringLiteral("--release-archive")),
             "Custom planning must retain signed identity and use the custom "
             "route");

  advancedArchive->setChecked(true);
  ok = Expect(build->validatePage() && custom.advancedSourceSelected(),
              "an explicit Advanced archive selection should validate") &&
       ok;
  const QStringList advancedArgs = custom.backendOptions(true);
  const qsizetype routeIndex = advancedArgs.indexOf(QStringLiteral("--route"));
  ok = Expect(routeIndex >= 0 &&
                  advancedArgs.value(routeIndex + 1) ==
                      QStringLiteral("advanced") &&
                  advancedArgs.contains(QStringLiteral("--release-archive")) &&
                  !advancedArgs.contains(QStringLiteral("--release-receipt")),
              "only an explicitly selected arbitrary source should switch "
              "Custom to Advanced trust semantics") &&
       ok;
  return ok;
}

bool TestExplicitSelfUpdateOffer(FakeBackend &backend) {
  QTemporaryDir directory;
  const QString runningPath =
      directory.filePath(QStringLiteral("running.AppImage"));
  const QString verifiedPath =
      directory.filePath(QStringLiteral("verified.AppImage"));
  const QString launchLog = directory.filePath(QStringLiteral("launch.json"));
  QFile running(runningPath);
  running.open(QIODevice::WriteOnly);
  running.write("running image remains unchanged");
  running.close();
  QFile verified(verifiedPath);
  verified.open(QIODevice::WriteOnly);
  verified.write("cryptographically verified fixture image");
  verified.close();
  verified.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner |
                          QFileDevice::ExeOwner);

  backend.setReleaseStatus(SelfUpdateStatus(
      runningPath, verifiedPath,
      QStringLiteral("https://fixture.invalid/installer.AppImage")));
  qputenv("APPIMAGE", runningPath.toLocal8Bit());
  qputenv("STUDIOCAST_INSTALLER_TEST_CONFIRM_SELF_UPDATE", "1");
  qputenv("STUDIOCAST_INSTALLER_TEST_SELF_UPDATE_LAUNCH_LOG",
          launchLog.toLocal8Bit());
  studiocast::installer::InstallerWizard offered;
  auto *intro = offered.page(studiocast::installer::PageIntro);
  intro->initializePage();
  auto *restart = intro->findChild<QPushButton *>(
      QStringLiteral("scInstallerSelfUpdateRestart"));
  auto *manual = intro->findChild<QPushButton *>(
      QStringLiteral("scInstallerSelfUpdateManual"));
  bool ok =
      Expect(
          restart && !restart->isHidden() && manual && !manual->isHidden(),
          "verified self-update should expose restart and manual fallback") &&
      Expect(!QFileInfo(launchLog).exists(),
             "verified self-update must not relaunch before explicit consent") &&
      Expect(backend.releaseCalls().contains(
                 QStringLiteral("--prepare-self-update")) &&
                 backend.releaseCalls().contains(runningPath),
             "AppImage detection should request a verified self-update offer");
  restart->click();
  QFile launch(launchLog);
  launch.open(QIODevice::ReadOnly);
  const QJsonArray command = QJsonDocument::fromJson(launch.readAll()).array();
  running.open(QIODevice::ReadOnly);
  ok =
      Expect(
          command.size() == 1 && command.first().toString() == verifiedPath,
          "restart must launch only the verified AppImage without a shell") &&
      Expect(running.readAll() == QByteArray("running image remains unchanged"),
             "self-update must never overwrite the running AppImage") &&
      ok;

  backend.setReleaseStatus(SelfUpdateStatus(
      runningPath, {},
      QStringLiteral("https://fixture.invalid/manual.AppImage"),
      QStringLiteral("manual_download")));
  studiocast::installer::InstallerWizard fallback;
  auto *fallbackIntro = fallback.page(studiocast::installer::PageIntro);
  fallbackIntro->initializePage();
  auto *fallbackRestart = fallbackIntro->findChild<QPushButton *>(
      QStringLiteral("scInstallerSelfUpdateRestart"));
  auto *fallbackManual = fallbackIntro->findChild<QPushButton *>(
      QStringLiteral("scInstallerSelfUpdateManual"));
  ok = Expect(fallbackRestart && fallbackRestart->isHidden() &&
                  fallbackManual && !fallbackManual->isHidden() &&
                  fallbackManual->toolTip() ==
                      QStringLiteral("https://fixture.invalid/manual.AppImage"),
              "unavailable relaunch should retain an HTTPS manual-download "
              "fallback") &&
       ok;

  backend.setReleaseStatus(ReleaseStatus());
  qunsetenv("APPIMAGE");
  qunsetenv("STUDIOCAST_INSTALLER_TEST_CONFIRM_SELF_UPDATE");
  qunsetenv("STUDIOCAST_INSTALLER_TEST_SELF_UPDATE_LAUNCH_LOG");
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
                      QStringLiteral("8 verified artifact files, 131224287 model bytes")) &&
                  reviewText->toPlainText().contains(
                      QStringLiteral("dlib_68_ibug_300w:shape_predictor_68_face_landmarks.dat: 99693937 bytes")) &&
                  reviewText->toPlainText().contains(
                      QStringLiteral("disk required/free: 131224290/20000000000 bytes")),
              "review should derive exact pack, artifact, size, and disk totals from the plan") && ok;

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

bool TestCustomAdvancedSelectionsAndRecommendedDefaults(FakeBackend &backend) {
  backend.setScenario(BaseFacts(),
                      Status(QStringLiteral("absent"), QStringLiteral("recommended"),
                             QStringLiteral("install")),
                      Plan(), Result(QStringLiteral("committed"), true), 0);
  studiocast::installer::InstallerWizard custom;
  custom.setCustomRoute(true);
  auto *page = custom.page(studiocast::installer::PageServiceOptions);
  page->initializePage();
  auto *openCuda = page->findChild<QCheckBox *>(
      QStringLiteral("scInstallerOpenCuda"));
  auto *openAudio = page->findChild<QCheckBox *>(
      QStringLiteral("scInstallerOpenAudio"));
  auto *device = page->findChild<QSpinBox *>(
      QStringLiteral("scInstallerV4lDeviceNumber"));
  auto *label =
      page->findChild<QLineEdit *>(QStringLiteral("scInstallerV4lLabel"));
  auto *exclusive = page->findChild<QCheckBox *>(
      QStringLiteral("scInstallerV4lExclusiveCaps"));
  auto *destination = page->findChild<QLineEdit *>(
      QStringLiteral("scInstallerModelDestination"));
  auto *installModels = page->findChild<QCheckBox *>(
      QStringLiteral("scInstallerInstallModels"));
  bool ok = Expect(openCuda && openAudio && device && label && exclusive &&
                       destination && installModels,
                   "Custom route should expose typed backend, v4l, and model controls");
  openCuda->setChecked(false);
  openAudio->setChecked(true);
  device->setValue(42);
  label->setText(QStringLiteral("StudioCast Custom Camera"));
  exclusive->setChecked(false);
  installModels->setChecked(true);
  const auto packChecks = page->findChildren<QCheckBox *>(
      QRegularExpression(QStringLiteral("^scInstallerModelPack_")));
  for (QCheckBox *pack : packChecks)
    pack->setChecked(false);
  page->findChild<QCheckBox *>(
          QStringLiteral("scInstallerModelPack_fastenhancer_s_vd_v1"))
      ->setChecked(true);
  page->findChild<QCheckBox *>(QStringLiteral(
          "scInstallerModelPack_gaze_correction_cam_flx_v0_1_1"))
      ->setChecked(true);
  ok = Expect(page->validatePage(),
              "valid Custom Advanced selections should pass validation") && ok;
  const QStringList customArgs = custom.backendOptions(true);
  ok = Expect(customArgs.contains(QStringLiteral("--no-open-cuda")) &&
                  customArgs.contains(QStringLiteral("--open-audio")) &&
                  !customArgs.contains(QStringLiteral("--open-backends")) &&
                  customArgs.value(customArgs.indexOf(
                                       QStringLiteral("--v4l-device-number")) + 1) ==
                      QStringLiteral("42") &&
                  customArgs.value(customArgs.indexOf(
                                       QStringLiteral("--v4l-label")) + 1) ==
                      QStringLiteral("StudioCast Custom Camera") &&
                  customArgs.contains(QStringLiteral("--no-v4l-exclusive-caps")) &&
                  customArgs.count(QStringLiteral("--model-id")) == 2 &&
                  customArgs.contains(
                      QStringLiteral("gaze_correction_cam_flx_v0_1_1")) &&
                  customArgs.contains(QStringLiteral("--model-destination")),
              "Custom plan arguments should preserve each reviewed typed selection") && ok;

  auto *configure = page->findChild<QCheckBox *>(
      QStringLiteral("scInstallerConfigureV4l"));
  configure->setChecked(false);
  auto *limitation = page->findChild<QLabel *>(
      QStringLiteral("scInstallerCameraLimitation"));
  auto *ack = page->findChild<QCheckBox *>(
      QStringLiteral("scInstallerCameraLimitationAcknowledged"));
  ok = Expect(limitation && !limitation->isHidden() && ack && !ack->isHidden(),
              "Custom camera opt-out should show a prominent degraded-state "
              "limitation and acknowledgement") && ok;
  ack->setChecked(true);
  ok = Expect(page->validatePage() &&
                  custom.backendOptions(true).contains(
                      QStringLiteral("--no-v4l2loopback")),
              "acknowledged Custom camera opt-out should remain available") && ok;

  studiocast::installer::InstallerWizard recommended;
  recommended.setBuildType(QStringLiteral("Debug"));
  recommended.setConfigureV4l2Loopback(false);
  recommended.setLoadLoopback(false);
  recommended.setPersistLoopback(false);
  recommended.setOpenCuda(false);
  recommended.setOpenAudio(false);
  recommended.setInstallModels(false);
  recommended.setModelPackIds({});
  recommended.setModelDestination(QStringLiteral("/tmp/not-user-models"));
  recommended.setInstallMesaVulkan(true);
  recommended.setInstallShaderTools(true);
  const QStringList recommendedArgs = recommended.backendOptions(true);
  const qsizetype buildTypeIndex =
      recommendedArgs.indexOf(QStringLiteral("--build-type"));
  const qsizetype destinationIndex =
      recommendedArgs.indexOf(QStringLiteral("--model-destination"));
  ok = Expect(buildTypeIndex >= 0 &&
                  recommendedArgs.value(buildTypeIndex + 1) ==
                      QStringLiteral("Release") &&
                  recommendedArgs.contains(QStringLiteral("--v4l2loopback")) &&
                  recommendedArgs.contains(QStringLiteral("--load-loopback")) &&
                  recommendedArgs.contains(QStringLiteral("--persist-loopback")) &&
                  recommendedArgs.contains(QStringLiteral("--open-cuda")) &&
                  recommendedArgs.contains(QStringLiteral("--open-audio")) &&
                  recommendedArgs.count(QStringLiteral("--model-id")) == 7 &&
                  destinationIndex >= 0 &&
                  recommendedArgs.value(destinationIndex + 1)
                      .endsWith(QStringLiteral("/studiocast/models")) &&
                  !recommendedArgs.contains(QStringLiteral("--mesa-vulkan")) &&
                  !recommendedArgs.contains(QStringLiteral("--shader-tools")),
              "Recommended planning should enforce Release, required camera, "
              "open fallbacks, seven packs, and a user-local destination") && ok;
  return ok;
}

bool TestMigratedUnknownChoicesRequireExplicitReview(FakeBackend &backend) {
  QJsonObject facts = BaseFacts(QStringLiteral("healthy"));
  QJsonObject installation =
      facts.value(QStringLiteral("installation")).toObject();
  installation[QStringLiteral("desired_configuration")] = QJsonObject{
      {QStringLiteral("features"), QJsonObject{}},
      {QStringLiteral("service"),
       QJsonObject{{QStringLiteral("desired"), QStringLiteral("unknown")}}},
      {QStringLiteral("v4l"),
       QJsonObject{{QStringLiteral("desired"), QStringLiteral("unknown")}}}};
  facts[QStringLiteral("installation")] = installation;
  backend.setScenario(facts,
                      Status(QStringLiteral("healthy"),
                             QStringLiteral("recommended"),
                             QStringLiteral("update"), true),
                      Plan(QStringLiteral("update")),
                      Result(QStringLiteral("committed"), true), 0);

  studiocast::installer::InstallerWizard migrated;
  migrated.setWorkflow(QStringLiteral("update"));
  QString error;
  bool ok = Expect(
      !migrated.refreshPlan(&error) &&
          error.contains(QStringLiteral("unknown"), Qt::CaseInsensitive) &&
          !migrated.backendOptions(true).contains(
              QStringLiteral("--v4l2loopback")) &&
          !migrated.backendOptions(true).contains(
              QStringLiteral("--no-v4l2loopback")),
      "unknown migrated choices must block review without becoming "
      "automatic boolean work");

  migrated.setCustomRoute(true);
  auto *page = migrated.page(studiocast::installer::PageServiceOptions);
  page->initializePage();
  auto *service =
      page->findChild<QCheckBox *>(QStringLiteral("scInstallerService"));
  auto *camera =
      page->findChild<QCheckBox *>(QStringLiteral("scInstallerConfigureV4l"));
  auto *load =
      page->findChild<QCheckBox *>(QStringLiteral("scInstallerLoadV4l"));
  auto *persist =
      page->findChild<QCheckBox *>(QStringLiteral("scInstallerPersistV4l"));
  auto *notice = page->findChild<QLabel *>(
      QStringLiteral("scInstallerUnknownMigratedChoices"));
  ok = Expect(service && camera && load && persist && notice &&
                  service->checkState() == Qt::PartiallyChecked &&
                  camera->checkState() == Qt::PartiallyChecked &&
                  load->checkState() == Qt::PartiallyChecked &&
                  persist->checkState() == Qt::PartiallyChecked &&
                  !notice->isHidden(),
              "unknown migrated choices should be visibly tri-state") &&
       ok;

  service->setCheckState(Qt::Checked);
  camera->setCheckState(Qt::Checked);
  load->setCheckState(Qt::Checked);
  persist->setCheckState(Qt::Unchecked);
  ok = Expect(page->validatePage() && migrated.refreshPlan(&error),
              "explicitly reviewed service/load/persistence choices should "
              "unblock a fresh exact plan") &&
       ok;
  const QString planArgs = backend.planArguments();
  ok =
      Expect(planArgs.contains(QStringLiteral("--service")) &&
                 planArgs.contains(QStringLiteral("--v4l2loopback")) &&
                 planArgs.contains(QStringLiteral("--load-loopback")) &&
                 planArgs.contains(QStringLiteral("--no-persist-loopback")) &&
                 planArgs.contains(QStringLiteral("--route\ncustom")),
             "reviewed controls should map to exact typed backend semantics") &&
      ok;
  ok = Expect(camera->text().contains(QStringLiteral("package")) &&
                  load->text().contains(QStringLiteral("module/device")) &&
                  persist->text().contains(QStringLiteral("namespaced")),
              "dependency, load, and persistence labels should describe the "
              "operations they control") &&
       ok;
  return ok;
}

bool TestUninstallPreservationReviewMatchesFlags(FakeBackend &backend) {
  backend.setScenario(BaseFacts(QStringLiteral("healthy")),
                      Status(QStringLiteral("healthy"),
                             QStringLiteral("modify"), QStringLiteral("modify"),
                             true),
                      Plan(QStringLiteral("uninstall")),
                      Result(QStringLiteral("committed"), false), 0);
  studiocast::installer::InstallerWizard uninstall;
  uninstall.setWorkflow(QStringLiteral("uninstall"));
  auto *page = uninstall.page(studiocast::installer::PageUninstall);
  page->initializePage();
  auto *remove = page->findChild<QCheckBox *>(
      QStringLiteral("scInstallerUninstallRemoveUserData"));
  auto *review = page->findChild<QPlainTextEdit *>(
      QStringLiteral("scInstallerUninstallPlan"));
  bool ok =
      Expect(remove && !remove->isChecked() && review &&
                 review->toPlainText().contains(QStringLiteral("PRESERVE")) &&
                 review->toPlainText().contains(QStringLiteral("logs")) &&
                 review->toPlainText().contains(QStringLiteral("caches")) &&
                 backend.planArguments().contains(
                     QStringLiteral("--preserve-user-data")),
             "ordinary uninstall should review and pass preservation for all "
             "user-owned data categories");
  remove->setChecked(true);
  QApplication::processEvents();
  ok = Expect(review->toPlainText().contains(QStringLiteral("REMOVE")) &&
                  review->toPlainText().contains(
                      QStringLiteral("other user data")) &&
                  backend.planArguments().contains(
                      QStringLiteral("--remove-user-data")),
              "explicit full user-data removal should update both review "
              "copy and exact plan flag") &&
       ok;
  return ok;
}

} // namespace

int main(int argc, char **argv) {
  (void)::setenv("QT_QPA_PLATFORM", "offscreen", 1);
  QApplication app(argc, argv);
  FakeBackend backend;
  bool ok = true;
  ok = TestSignedReleaseAndOfflineReceipt(backend) && ok;
  ok = TestExplicitOfflineUsesOnlyVerifiedCache(backend) && ok;
  ok = TestCustomRetainsVerifiedOfficialSource(backend) && ok;
  ok = TestExplicitSelfUpdateOffer(backend) && ok;
  ok = TestStateDrivenRoutes(backend) && ok;
  ok = TestRecommendedCustomAndContextualRoutes(backend) && ok;
  ok = TestUnsupportedOfflineAndPriorReuse(backend) && ok;
  ok = TestPlanValidationExactExecutionAndLabels(backend) && ok;
  ok = TestFailureAndDegradedCompletionTruth(backend) && ok;
  ok = TestStructuredProgressAndDegradedRetry(backend) && ok;
  ok = TestCancellationCleansChild(backend) && ok;
  ok = TestCustomAdvancedSelectionsAndRecommendedDefaults(backend) && ok;
  ok = TestMigratedUnknownChoicesRequireExplicitReview(backend) && ok;
  ok = TestUninstallPreservationReviewMatchesFlags(backend) && ok;
  return ok ? 0 : 1;
}
