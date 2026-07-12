#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <unistd.h>

#include "core/util/exec.h"

#ifndef STUDIOCAST_SOURCE_DIR
#define STUDIOCAST_SOURCE_DIR ""
#endif

namespace {

namespace fs = std::filesystem;

bool Expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

bool ExpectContains(const std::string &name, const std::string &haystack,
                    const std::string &needle) {
  if (haystack.find(needle) == std::string::npos) {
    std::cerr << name << " missing expected text: " << needle << "\n";
    std::cerr << "Output:\n" << haystack << "\n";
    return false;
  }
  return true;
}

std::string ShellQuote(const std::string &value) {
  std::string out = "'";
  for (char ch : value) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out += ch;
    }
  }
  out += "'";
  return out;
}

bool WriteExecutable(const fs::path &path, const std::string &text,
                     std::string *error = nullptr) {
  std::error_code ec;
  fs::create_directories(path.parent_path(), ec);
  if (ec) {
    if (error)
      *error = "create_directories failed: " + ec.message();
    return false;
  }

  std::ofstream file(path);
  if (!file) {
    if (error)
      *error = "open failed: " + path.string();
    return false;
  }
  file << text;
  file.close();

  fs::permissions(path,
                  fs::perms::owner_read | fs::perms::owner_write |
                      fs::perms::owner_exec,
                  fs::perm_options::replace, ec);
  if (ec) {
    if (error)
      *error = "permissions failed: " + ec.message();
    return false;
  }
  return true;
}

class ScopedTempDir {
public:
  explicit ScopedTempDir(const std::string &prefix) {
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    if (ec) {
      error_ = "temp_directory_path failed: " + ec.message();
      return;
    }
    path_ = base /
            (prefix + "-" + std::to_string(static_cast<long long>(::getpid())));
    fs::remove_all(path_, ec);
    fs::create_directories(path_, ec);
    if (ec)
      error_ = "create_directories failed: " + ec.message();
  }

  ~ScopedTempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }

  bool ok() const { return error_.empty(); }
  const std::string &error() const { return error_; }
  const fs::path &path() const { return path_; }

private:
  fs::path path_;
  std::string error_;
};

std::string BackendCommand(const ScopedTempDir &root,
                           const std::string &subcommand,
                           const std::string &extra_options = "") {
  const fs::path repo = fs::path(STUDIOCAST_SOURCE_DIR);
  const fs::path backend =
      repo / "installer" / "backend" / "studiocast-installer-backend";
  const fs::path buildDir = root.path() / "build";
  std::string command =
      "HOME=" + ShellQuote((root.path() / "home").string()) + " " +
      ShellQuote(backend.string()) + " " + subcommand + " --source-dir " +
      ShellQuote(repo.string()) + " --build-dir " +
      ShellQuote(buildDir.string()) +
      " --skip-deps --no-v4l2loopback --no-service --no-models "
      "--allow-unsupported";
  if (!extra_options.empty()) {
    command += " " + extra_options;
  }
  return command;
}

bool TestRepairPlanIncludesDefaultOpenBackendConfigureFlags() {
  ScopedTempDir temp("studiocast-installer-backend-plan");
  if (!Expect(temp.ok(), temp.error().c_str()))
    return false;

  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = 10000;
  const auto result = studiocast::util::ExecCapture(
      BackendCommand(temp, "plan repair --json"), options);

  return Expect(result.exit_code == 0,
                "installer backend plan repair should exit successfully") &&
         ExpectContains("installer backend plan", result.stdout_str,
                        "-DSTUDIOCAST_ENABLE_OPEN_CUDA=ON") &&
         ExpectContains("installer backend plan", result.stdout_str,
                        "-DSTUDIOCAST_ENABLE_OPEN_AUDIO=ON") &&
         Expect(result.stdout_str.find("-DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON") ==
                    std::string::npos,
                "default installer plan should not enable Open Vulkan") &&
         Expect(result.stdout_str.find("--vulkan-runtime") == std::string::npos,
                "default installer plan should not install Vulkan runtime "
                "packages") &&
         ExpectContains("installer backend plan", result.stdout_str,
                        "Enable CPU resize fallback in the daemon config by "
                        "default") &&
         ExpectContains("installer backend plan", result.stdout_str,
                        "Force Linux CMake configuration to keep Open Video/"
                        "Open CUDA and Open Audio enabled");
}

bool TestRepairPlanCanOptIntoVulkanRuntimeAndBackend() {
  ScopedTempDir temp("studiocast-installer-backend-vulkan-plan");
  if (!Expect(temp.ok(), temp.error().c_str()))
    return false;

  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = 10000;
  const auto result = studiocast::util::ExecCapture(
      BackendCommand(temp, "plan repair --json",
                     "--with-deps --vulkan-runtime --mesa-vulkan "
                     "--shader-tools --open-vulkan"),
      options);

  return Expect(result.exit_code == 0,
                "installer backend Vulkan plan should exit successfully") &&
         ExpectContains("installer backend Vulkan plan", result.stdout_str,
                        "--vulkan-runtime") &&
         ExpectContains("installer backend Vulkan plan", result.stdout_str,
                        "--mesa-vulkan") &&
         ExpectContains("installer backend Vulkan plan", result.stdout_str,
                        "--shader-tools") &&
         ExpectContains("installer backend Vulkan plan", result.stdout_str,
                        "-DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON") &&
         ExpectContains("installer backend Vulkan plan", result.stdout_str,
                        "optional Vulkan loader/diagnostic packages") &&
         ExpectContains("installer backend Vulkan plan", result.stdout_str,
                        "Mesa Intel/AMD Vulkan ICD") &&
         ExpectContains("installer backend Vulkan plan", result.stdout_str,
                        "developer shader tools") &&
         ExpectContains("installer backend Vulkan plan", result.stdout_str,
                        "runtime-loaded Open Vulkan") &&
         ExpectContains("installer backend Vulkan plan", result.stdout_str,
                        "working GPU driver/ICD") &&
         ExpectContains("installer backend Vulkan plan", result.stdout_str,
                        "virtual-background matting remains blocked");
}

bool TestRepairPlanInstallsOnlySelectedVulkanPackages() {
  ScopedTempDir temp("studiocast-installer-backend-vulkan-repair-plan");
  if (!Expect(temp.ok(), temp.error().c_str()))
    return false;

  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = 10000;
  const auto result = studiocast::util::ExecCapture(
      BackendCommand(temp, "plan repair --json",
                     "--vulkan-runtime --mesa-vulkan --shader-tools "
                     "--open-vulkan"),
      options);

  return Expect(result.exit_code == 0,
                "installer backend Vulkan-only repair plan should exit "
                "successfully") &&
         ExpectContains("installer backend Vulkan-only repair plan",
                        result.stdout_str,
                        "scripts/setup.sh --vulkan-runtime --mesa-vulkan "
                        "--shader-tools") &&
         Expect(result.stdout_str.find("scripts/setup.sh --deps") ==
                    std::string::npos,
                "Vulkan-only repair should not force the full dependency "
                "bundle");
}

bool TestRepairPlanRuntimeOnlyExplainsVulkanBoundary() {
  ScopedTempDir temp("studiocast-installer-backend-vulkan-runtime-plan");
  if (!Expect(temp.ok(), temp.error().c_str()))
    return false;

  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = 10000;
  const auto result = studiocast::util::ExecCapture(
      BackendCommand(temp, "plan repair --json", "--vulkan-runtime"), options);

  return Expect(result.exit_code == 0,
                "installer backend Vulkan runtime-only plan should exit "
                "successfully") &&
         ExpectContains("installer backend Vulkan runtime-only plan",
                        result.stdout_str,
                        "Open Vulkan is optional and runtime-loaded") &&
         ExpectContains("installer backend Vulkan runtime-only plan",
                        result.stdout_str, "working GPU driver/ICD") &&
         ExpectContains("installer backend Vulkan runtime-only plan",
                        result.stdout_str,
                        "virtual-background matting remains blocked") &&
         Expect(result.stdout_str.find("-DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON") ==
                    std::string::npos,
                "runtime-only package selection should not implicitly enable "
                "the Open Vulkan build");
}

bool TestRepairPlanCanDisableOpenBackendConfigureFlags() {
  ScopedTempDir temp("studiocast-installer-backend-no-open");
  if (!Expect(temp.ok(), temp.error().c_str()))
    return false;

  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = 10000;
  const auto result = studiocast::util::ExecCapture(
      BackendCommand(temp, "plan repair --json", "--no-open-backends"),
      options);

  return Expect(result.exit_code == 0,
                "installer backend plan repair without open backends should "
                "exit successfully") &&
         Expect(result.stdout_str.find("-DSTUDIOCAST_ENABLE_OPEN_CUDA=ON") ==
                    std::string::npos,
                "disabled Open Source backend setup should omit Open CUDA "
                "configure flag") &&
         Expect(result.stdout_str.find("-DSTUDIOCAST_ENABLE_OPEN_AUDIO=ON") ==
                    std::string::npos,
                "disabled Open Source backend setup should omit Open Audio "
                "configure flag");
}

bool TestRepairDryRunIncludesOpenBackendConfigureFlags() {
  ScopedTempDir temp("studiocast-installer-backend-dry-run");
  if (!Expect(temp.ok(), temp.error().c_str()))
    return false;

  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = 10000;
  const auto result = studiocast::util::ExecCapture(
      BackendCommand(temp, "repair --dry-run"), options);

  return Expect(result.exit_code == 0,
                "installer backend repair dry-run should exit successfully") &&
         ExpectContains("installer backend repair dry-run", result.stdout_str,
                        "-DSTUDIOCAST_ENABLE_OPEN_CUDA=ON") &&
         ExpectContains("installer backend repair dry-run", result.stdout_str,
                        "-DSTUDIOCAST_ENABLE_OPEN_AUDIO=ON") &&
         ExpectContains("installer backend repair dry-run", result.stdout_str,
                        "video.scaling.allow_cpu_resize");
}

bool TestStatusReportsOptionalComponents() {
  ScopedTempDir temp("studiocast-installer-backend-status");
  if (!Expect(temp.ok(), temp.error().c_str()))
    return false;

  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = 10000;
  const auto result = studiocast::util::ExecCapture(
      BackendCommand(temp, "status --json"), options);

  return Expect(result.exit_code == 0,
                "installer backend status should exit successfully") &&
         ExpectContains("installer backend status", result.stdout_str,
                        "\"optional_components\"") &&
         ExpectContains("installer backend status", result.stdout_str,
                        "\"onnxruntime_cuda\"") &&
         ExpectContains("installer backend status", result.stdout_str,
                        "\"vulkan\"") &&
         ExpectContains("installer backend status", result.stdout_str,
                        "\"libvulkan1\"") &&
         ExpectContains("installer backend status", result.stdout_str,
                        "\"vulkan-tools\"") &&
         ExpectContains("installer backend status", result.stdout_str,
                        "\"mesa-vulkan-drivers\"") &&
         ExpectContains("installer backend status", result.stdout_str,
                        "docs/open_source_video_models_install.md") &&
         ExpectContains("installer backend status", result.stdout_str,
                        "docs/maxine_install.md");
}

bool IsSkippedPackageSafetyDir(const fs::path &p) {
  const std::string name = p.filename().string();
  return name == ".git" || name == ".idea" || name == "build" ||
         name.rfind("cmake-build", 0) == 0;
}

bool IsForbiddenBundledMaxineArtifact(const fs::path &p) {
  const std::string name = p.filename().string();
  const char *forbidden_prefixes[] = {
      "libVideoFX.so",        "libnvvfx.so",          "libNvVFX.so",
      "libnvVideoEffects.so", "libNVVideoEffects.so", "libnvARPose.so",
      "libnvar.so",           "libNvAR.so",           "libnv_audiofx.so",
      "NVIDIA_VFX_SDK_linux", "NVIDIA_AR_SDK_linux",  "Audio_Effects_SDK.tar",
  };
  for (const char *prefix : forbidden_prefixes) {
    if (name.rfind(prefix, 0) == 0) {
      return true;
    }
  }
  return false;
}

bool CmakeInstallBlocksContainForbiddenMaxineArtifact(const std::string &cmake,
                                                      std::string *matched) {
  const std::string forbidden[] = {
      "VideoFX",           "ARSDK",
      "Audio_Effects_SDK", "libVideoFX.so",
      "libnv_audiofx.so",  "libnvARPose.so",
  };

  std::size_t pos = 0;
  while ((pos = cmake.find("install(", pos)) != std::string::npos) {
    std::size_t end = cmake.find("\n)", pos);
    if (end == std::string::npos) {
      end = cmake.find(')', pos);
    }
    if (end == std::string::npos) {
      end = cmake.size();
    }
    const std::string block = cmake.substr(pos, end - pos + 1);
    for (const std::string &needle : forbidden) {
      if (block.find(needle) != std::string::npos) {
        if (matched)
          *matched = needle;
        return true;
      }
    }
    pos = end + 1;
  }
  return false;
}

bool TestPackageSafetyDoesNotBundleOrInstallMaxineArtifacts() {
  const fs::path repo = fs::path(STUDIOCAST_SOURCE_DIR);
  if (!Expect(!repo.empty() && fs::exists(repo),
              "source directory should exist for package safety test")) {
    return false;
  }

  std::error_code ec;
  for (fs::recursive_directory_iterator it(repo, ec), end; it != end;
       it.increment(ec)) {
    if (ec) {
      std::cerr << "repository walk failed: " << ec.message() << "\n";
      return false;
    }
    if (it->is_directory(ec) && IsSkippedPackageSafetyDir(it->path())) {
      it.disable_recursion_pending();
      continue;
    }
    if (it->is_regular_file(ec) &&
        IsForbiddenBundledMaxineArtifact(it->path())) {
      std::cerr << "forbidden bundled Maxine SDK artifact found: " << it->path()
                << "\n";
      return false;
    }
  }

  const std::string cmake = [&] {
    std::ifstream in(repo / "CMakeLists.txt");
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
  }();
  std::string matched;
  if (CmakeInstallBlocksContainForbiddenMaxineArtifact(cmake, &matched)) {
    std::cerr << "CMake install/package surface mentions forbidden Maxine "
                 "artifact pattern: "
              << matched << "\n";
    return false;
  }

  ScopedTempDir temp("studiocast-installer-backend-maxine-safety");
  if (!Expect(temp.ok(), temp.error().c_str()))
    return false;

  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = 10000;
  const auto result = studiocast::util::ExecCapture(
      BackendCommand(temp, "plan repair --json"), options);

  return Expect(result.exit_code == 0,
                "installer backend plan repair should exit successfully") &&
         Expect(result.stdout_str.find("scripts/setup/maxine.sh") ==
                    std::string::npos,
                "default installer plan should not run Maxine setup") &&
         Expect(result.stdout_str.find("install_feature.sh") ==
                    std::string::npos,
                "default installer plan should not run Maxine feature "
                "installers") &&
         Expect(result.stdout_str.find("download_features.sh") ==
                    std::string::npos,
                "default installer plan should not run Maxine feature "
                "downloaders") &&
         Expect(result.stdout_str.find("NGC_API_KEY") == std::string::npos &&
                    result.stdout_str.find("NGC_CLI_API_KEY") ==
                        std::string::npos,
                "default installer plan should not mention NGC secrets");
}

bool TestUserServiceDryRunRestartsServiceAfterInstall() {
  ScopedTempDir temp("studiocast-user-service-dry-run");
  if (!Expect(temp.ok(), temp.error().c_str()))
    return false;

  std::string error;
  const fs::path fakeSystemctl = temp.path() / "bin" / "systemctl";
  const fs::path buildDir = temp.path() / "build";
  const fs::path daemon = buildDir / "studiocastd";
  if (!Expect(WriteExecutable(fakeSystemctl, "#!/usr/bin/env bash\nexit 0\n",
                              &error),
              error.c_str()) ||
      !Expect(WriteExecutable(daemon, "#!/usr/bin/env bash\nexit 0\n", &error),
              error.c_str())) {
    return false;
  }

  const fs::path repo = fs::path(STUDIOCAST_SOURCE_DIR);
  const fs::path script = repo / "scripts" / "install" / "user_service.sh";
  const char *pathEnv = std::getenv("PATH");
  const std::string command =
      "HOME=" + ShellQuote((temp.path() / "home").string()) + " " +
      "XDG_CONFIG_HOME=" + ShellQuote((temp.path() / "config").string()) +
      " PATH=" +
      ShellQuote((temp.path() / "bin").string() + ":" +
                 (pathEnv ? pathEnv : "")) +
      " " + ShellQuote(script.string()) + " --dry-run --yes --build-dir " +
      ShellQuote(buildDir.string());

  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = 10000;
  const auto result = studiocast::util::ExecCapture(command, options);

  return Expect(result.exit_code == 0,
                "user service dry-run should exit successfully") &&
         ExpectContains("user service dry-run", result.stdout_str,
                        "systemctl --user enable studiocastd.service") &&
         ExpectContains("user service dry-run", result.stdout_str,
                        "systemctl --user restart studiocastd.service") &&
         Expect(result.stdout_str.find("enable --now") == std::string::npos,
                "user service install should not rely on enable --now for "
                "already-running daemons");
}

} // namespace

int main() {
  bool ok = true;
  ok = TestRepairPlanIncludesDefaultOpenBackendConfigureFlags() && ok;
  ok = TestRepairPlanCanOptIntoVulkanRuntimeAndBackend() && ok;
  ok = TestRepairPlanInstallsOnlySelectedVulkanPackages() && ok;
  ok = TestRepairPlanRuntimeOnlyExplainsVulkanBoundary() && ok;
  ok = TestRepairPlanCanDisableOpenBackendConfigureFlags() && ok;
  ok = TestRepairDryRunIncludesOpenBackendConfigureFlags() && ok;
  ok = TestStatusReportsOptionalComponents() && ok;
  ok = TestPackageSafetyDoesNotBundleOrInstallMaxineArtifacts() && ok;
  ok = TestUserServiceDryRunRestartsServiceAfterInstall() && ok;
  return ok ? 0 : 1;
}
