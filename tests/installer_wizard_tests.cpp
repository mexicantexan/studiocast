#include <cstdlib>
#include <iostream>

#include <QApplication>
#include <QCheckBox>
#include <QProgressBar>
#include <QStringList>

#include "installer_wizard.h"

namespace {

bool Expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

bool TestOpenBackendSetupArguments() {
  studiocast::installer::InstallerWizard wizard;
  wizard.setWorkflow(QStringLiteral("repair"));

  wizard.setOpenBackendsSetup(true);
  QStringList args = wizard.backendOptions(true);
  if (!Expect(
          args.contains(QStringLiteral("--open-backends")),
          "enabled Open Source backend setup should pass --open-backends") ||
      !Expect(!args.contains(QStringLiteral("--no-open-backends")),
              "enabled Open Source backend setup should not pass "
              "--no-open-backends")) {
    return false;
  }

  wizard.setOpenBackendsSetup(false);
  args = wizard.backendOptions(true);
  return Expect(args.contains(QStringLiteral("--no-open-backends")),
                "disabled Open Source backend setup should pass "
                "--no-open-backends") &&
         Expect(!args.contains(QStringLiteral("--open-backends")),
                "disabled Open Source backend setup should not pass "
                "--open-backends");
}

bool TestVulkanInstallerArguments() {
  studiocast::installer::InstallerWizard wizard;
  wizard.setWorkflow(QStringLiteral("repair"));

  QStringList args = wizard.backendOptions(true);
  if (!Expect(args.contains(QStringLiteral("--no-open-vulkan")),
              "Open Vulkan should default off") ||
      !Expect(args.contains(QStringLiteral("--no-vulkan-runtime")),
              "Vulkan runtime packages should default off") ||
      !Expect(!args.contains(QStringLiteral("--mesa-vulkan")),
              "Mesa Vulkan ICDs should default off") ||
      !Expect(!args.contains(QStringLiteral("--shader-tools")),
              "shader tools should default off")) {
    return false;
  }

  wizard.setOpenVulkan(true);
  wizard.setInstallVulkanRuntime(true);
  wizard.setInstallMesaVulkan(true);
  wizard.setInstallShaderTools(true);
  args = wizard.backendOptions(true);

  return Expect(args.contains(QStringLiteral("--open-vulkan")),
                "selected Open Vulkan should pass --open-vulkan") &&
         Expect(!args.contains(QStringLiteral("--no-open-vulkan")),
                "selected Open Vulkan should omit --no-open-vulkan") &&
         Expect(args.contains(QStringLiteral("--vulkan-runtime")),
                "selected Vulkan packages should pass --vulkan-runtime") &&
         Expect(!args.contains(QStringLiteral("--no-vulkan-runtime")),
                "selected Vulkan packages should omit --no-vulkan-runtime") &&
         Expect(args.contains(QStringLiteral("--mesa-vulkan")),
                "selected Mesa ICDs should pass --mesa-vulkan") &&
         Expect(args.contains(QStringLiteral("--shader-tools")),
                "selected shader tools should pass --shader-tools");
}

bool TestVulkanControlsAreVisibleForRepair() {
  studiocast::installer::InstallerWizard wizard;
  wizard.setWorkflow(QStringLiteral("repair"));
  QWizardPage *page = wizard.page(studiocast::installer::PageServiceOptions);
  if (!Expect(page != nullptr, "service options page should exist")) {
    return false;
  }
  page->initializePage();

  const auto control = [page](const char *name) {
    return page->findChild<QCheckBox *>(QString::fromLatin1(name));
  };
  QCheckBox *openVulkan = control("scInstallerOpenVulkan");
  QCheckBox *runtime = control("scInstallerVulkanRuntime");
  QCheckBox *mesa = control("scInstallerMesaVulkan");
  QCheckBox *shaderTools = control("scInstallerShaderTools");

  return Expect(openVulkan != nullptr && openVulkan->isEnabled(),
                "repair should expose the Open Vulkan build option") &&
         Expect(runtime != nullptr && runtime->isEnabled(),
                "repair should expose Vulkan runtime packages") &&
         Expect(mesa != nullptr,
                "repair should expose Mesa Intel/AMD Vulkan ICDs") &&
         Expect(shaderTools != nullptr && shaderTools->isEnabled(),
                "repair should expose optional Vulkan shader tools") &&
         Expect(
             openVulkan->toolTip().contains(QStringLiteral("runtime-loaded")),
             "Open Vulkan control should explain runtime loading") &&
         Expect(runtime->toolTip().contains(QStringLiteral("driver/ICD")),
                "Vulkan runtime control should explain the driver/ICD need");
}

bool TestReviewPageUsesFinishCommitButton() {
  studiocast::installer::InstallerWizard wizard;
  const QWizardPage *reviewPage =
      wizard.page(studiocast::installer::PageReview);
  const QWizardPage *progressPage =
      wizard.page(studiocast::installer::PageProgress);

  return Expect(reviewPage != nullptr, "review page should exist") &&
         Expect(reviewPage->isCommitPage(),
                "review page should use the commit button") &&
         Expect(progressPage != nullptr, "progress page should exist") &&
         Expect(!progressPage->isCommitPage(),
                "progress page should not use the commit button") &&
         Expect(wizard.buttonText(QWizard::CommitButton) ==
                    QStringLiteral("Finish"),
                "review page commit button should read Finish");
}

bool TestUninstallUsesDedicatedShortRoute() {
  studiocast::installer::InstallerWizard wizard;
  wizard.setWorkflow(QStringLiteral("uninstall"));
  wizard.setStartId(studiocast::installer::PageIntro);
  wizard.restart();

  if (!Expect(wizard.currentId() == studiocast::installer::PageIntro,
              "wizard should restart on the workflow page") ||
      !Expect(wizard.nextId() == studiocast::installer::PageUninstall,
              "uninstall should bypass compatibility and install pages")) {
    return false;
  }

  wizard.setStartId(studiocast::installer::PageUninstall);
  wizard.restart();
  return Expect(wizard.currentId() == studiocast::installer::PageUninstall,
                "uninstall confirmation page should be reachable") &&
         Expect(wizard.nextId() == studiocast::installer::PageProgress,
                "uninstall confirmation should proceed directly to progress");
}

bool TestUninstallConfirmationAndArguments() {
  studiocast::installer::InstallerWizard wizard;
  wizard.setWorkflow(QStringLiteral("uninstall"));
  QWizardPage *page = wizard.page(studiocast::installer::PageUninstall);
  if (!Expect(page != nullptr, "uninstall confirmation page should exist")) {
    return false;
  }
  page->initializePage();

  auto *removeUserData = page->findChild<QCheckBox *>(
      QStringLiteral("scInstallerUninstallRemoveUserData"));
  if (!Expect(page->isCommitPage(),
              "uninstall confirmation should be a commit page") ||
      !Expect(wizard.buttonText(QWizard::CommitButton) ==
                  QStringLiteral("Uninstall"),
              "uninstall commit button should describe the action") ||
      !Expect(removeUserData != nullptr,
              "uninstall should expose its user-data choice") ||
      !Expect(!removeUserData->isChecked(),
              "uninstall should preserve user data by default")) {
    return false;
  }

  QStringList args = wizard.workflowCommandArguments(false);
  if (!Expect(
          args == QStringList{QStringLiteral("uninstall"),
                              QStringLiteral("--preserve-user-data"),
                              QStringLiteral("--yes")},
          "default uninstall command should contain only removal options")) {
    return false;
  }

  removeUserData->setChecked(true);
  args = wizard.workflowCommandArguments(false);
  return Expect(args == QStringList{QStringLiteral("uninstall"),
                                    QStringLiteral("--remove-user-data"),
                                    QStringLiteral("--yes")},
                "user-data removal should be explicit in uninstall arguments");
}

bool TestPreferenceProgressBars() {
  studiocast::installer::InstallerWizard wizard;
  constexpr int kFirstPage = studiocast::installer::PageIntro;
  constexpr int kLastPreferencePage = studiocast::installer::PageReview;
  constexpr int kMaximum = kLastPreferencePage - kFirstPage + 1;

  bool ok = true;
  for (int pageId = kFirstPage; pageId <= kLastPreferencePage; ++pageId) {
    const QWizardPage *page = wizard.page(pageId);
    ok = Expect(page != nullptr, "preference page should exist") && ok;
    if (!page) {
      continue;
    }

    const auto *bar = page->findChild<QProgressBar *>(
        QStringLiteral("scInstallerPreferenceProgress"));
    ok =
        Expect(bar != nullptr, "preference page should include progress bar") &&
        ok;
    if (!bar) {
      continue;
    }

    ok = Expect(bar->minimum() == 0, "progress bar should start at zero") && ok;
    ok = Expect(bar->maximum() == kMaximum,
                "progress bar maximum should match preference page count") &&
         ok;
    ok = Expect(bar->value() == pageId - kFirstPage + 1,
                "progress bar value should match page position") &&
         ok;
    ok = Expect(bar->property("scRole").toString() ==
                    QStringLiteral("installerPreferenceProgress"),
                "progress bar should use installer style role") &&
         ok;
    ok = Expect(!bar->isTextVisible(), "progress bar text should be hidden") &&
         ok;
  }

  QWizardPage *uninstallPage =
      wizard.page(studiocast::installer::PageUninstall);
  auto *uninstallBar =
      uninstallPage ? uninstallPage->findChild<QProgressBar *>(
                          QStringLiteral("scInstallerPreferenceProgress"))
                    : nullptr;
  ok = Expect(uninstallBar != nullptr,
              "uninstall confirmation should include a progress bar") &&
       ok;
  if (uninstallBar) {
    ok = Expect(uninstallBar->maximum() == 2 && uninstallBar->value() == 2,
                "uninstall confirmation should be step two of two") &&
         ok;
  }

  wizard.setWorkflow(QStringLiteral("uninstall"));
  auto *introBar = wizard.page(studiocast::installer::PageIntro)
                       ->findChild<QProgressBar *>(
                           QStringLiteral("scInstallerPreferenceProgress"));
  ok = Expect(introBar != nullptr && introBar->maximum() == 2 &&
                  introBar->value() == 1,
              "uninstall selection should show a two-step preference flow") &&
       ok;

  return ok;
}

} // namespace

int main(int argc, char **argv) {
  (void)::setenv("QT_QPA_PLATFORM", "offscreen", 0);
  QApplication app(argc, argv);

  bool ok = true;
  ok = TestOpenBackendSetupArguments() && ok;
  ok = TestVulkanInstallerArguments() && ok;
  ok = TestVulkanControlsAreVisibleForRepair() && ok;
  ok = TestReviewPageUsesFinishCommitButton() && ok;
  ok = TestUninstallUsesDedicatedShortRoute() && ok;
  ok = TestUninstallConfirmationAndArguments() && ok;
  ok = TestPreferenceProgressBars() && ok;
  return ok ? 0 : 1;
}
