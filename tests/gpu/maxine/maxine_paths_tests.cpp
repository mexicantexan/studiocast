#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#include "core/maxine/afx_api.h"
#include "core/maxine/paths.h"

#ifndef STUDIOCAST_CXX_COMPILER
#define STUDIOCAST_CXX_COMPILER "c++"
#endif

namespace {
namespace fs = std::filesystem;

class EnvGuard {
public:
  EnvGuard(const char *name, const std::string &value) : name_(name) {
    if (const char *old = std::getenv(name)) {
      old_value_ = std::string(old);
    }
    ::setenv(name, value.c_str(), 1);
  }

  ~EnvGuard() {
    if (old_value_) {
      ::setenv(name_, old_value_->c_str(), 1);
    } else {
      ::unsetenv(name_);
    }
  }

  EnvGuard(const EnvGuard &) = delete;
  EnvGuard &operator=(const EnvGuard &) = delete;

private:
  const char *name_;
  std::optional<std::string> old_value_;
};

bool Touch(const fs::path &path) {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  return out.good();
}

bool Require(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << "\n";
    return false;
  }
  return true;
}

std::string ShellQuote(const std::string &value) {
  std::string out = "'";
  for (const char c : value) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

bool CommandExitedOk(int rc) {
  if (rc == 0)
    return true;
#ifdef WIFEXITED
  return WIFEXITED(rc) && WEXITSTATUS(rc) == 0;
#else
  return false;
#endif
}

bool WriteFakeAfxLibrarySource(const fs::path &path) {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out)
    return false;

  out << R"cpp(
extern "C" int NvAFX_CreateEffect(const char *, void **) { return 0; }
extern "C" int NvAFX_DestroyEffect(void *) { return 0; }
extern "C" int NvAFX_SetU32(void *, const char *, unsigned int) { return 0; }
extern "C" int NvAFX_SetFloat(void *, const char *, float) { return 0; }
extern "C" int NvAFX_SetString(void *, const char *, const char *) { return 0; }
extern "C" int NvAFX_GetU32(void *, const char *, unsigned int *) { return 0; }
extern "C" int NvAFX_Load(void *) { return 0; }
extern "C" int NvAFX_Run(void *, const float *, float *, unsigned int) { return 0; }
)cpp";
  return out.good();
}

bool CompileFakeAfxLibrary(const fs::path &library_path) {
  std::error_code ec;
  fs::create_directories(library_path.parent_path(), ec);
  if (ec) {
    std::cerr << "failed to create fake AFX lib directory: " << ec.message()
              << "\n";
    return false;
  }

  const fs::path source_path = library_path.parent_path() / "fake_afx.cpp";
  if (!WriteFakeAfxLibrarySource(source_path)) {
    std::cerr << "failed to write fake AFX source: " << source_path << "\n";
    return false;
  }

  std::ostringstream cmd;
  cmd << ShellQuote(STUDIOCAST_CXX_COMPILER) << " -shared -fPIC "
      << ShellQuote(source_path.string()) << " -o "
      << ShellQuote(library_path.string());
  const int rc = std::system(cmd.str().c_str());
  if (!CommandExitedOk(rc)) {
    std::cerr << "failed to compile fake AFX library with command:\n"
              << cmd.str() << "\n";
    return false;
  }
  return true;
}

bool RunAfxLoaderPriorityChild() {
  const char *sdk_root_env = std::getenv("STUDIOCAST_AFX_TEST_SDK_ROOT");
  const char *expected_env = std::getenv("STUDIOCAST_AFX_TEST_EXPECTED_LIB");
  if (!sdk_root_env || !expected_env) {
    std::cerr << "AFX loader priority child missing environment\n";
    return false;
  }

  const fs::path sdk_root = sdk_root_env;
  const fs::path expected = expected_env;

  studiocast::maxine::afx::AfxApi api;
  std::string err;
  if (!api.Initialize({sdk_root}, &err)) {
    std::cerr << "fake AFX Initialize failed: " << err << "\n";
    return false;
  }

  return Require(api.library_path() == expected,
                 "expected AFX loader to prefer SDK root library " +
                     expected.string() + ", got " +
                     api.library_path().string());
}

bool TestAfxLoaderPrefersExplicitSdkRootBeforeBareLoaderPath(
    const char *argv0) {
  const fs::path root =
      fs::temp_directory_path() /
      ("studiocast-afx-loader-priority-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(root, ec);

  const fs::path sdk_root = root / "Audio_Effects_SDK";
  const fs::path sdk_lib = sdk_root / "nvafx" / "lib" / "libnv_audiofx.so";
  const fs::path bare_dir = root / "bare";
  const fs::path bare_lib = bare_dir / "libnv_audiofx.so";

  bool ok = CompileFakeAfxLibrary(sdk_lib) && CompileFakeAfxLibrary(bare_lib);
  if (!ok) {
    fs::remove_all(root, ec);
    return false;
  }

  const std::string old_ld = std::getenv("LD_LIBRARY_PATH")
                                 ? std::getenv("LD_LIBRARY_PATH")
                                 : "";
  const std::string child_ld =
      old_ld.empty() ? bare_dir.string() : (bare_dir.string() + ":" + old_ld);

  ::setenv("STUDIOCAST_AFX_LOADER_PRIORITY_CHILD", "1", 1);
  ::setenv("STUDIOCAST_AFX_TEST_SDK_ROOT", sdk_root.string().c_str(), 1);
  ::setenv("STUDIOCAST_AFX_TEST_EXPECTED_LIB", sdk_lib.string().c_str(), 1);
  ::setenv("LD_LIBRARY_PATH", child_ld.c_str(), 1);

  const fs::path self = fs::absolute(argv0, ec);
  const std::string self_path = ec ? std::string(argv0) : self.string();
  const int rc = std::system(ShellQuote(self_path).c_str());

  ::unsetenv("STUDIOCAST_AFX_LOADER_PRIORITY_CHILD");
  ::unsetenv("STUDIOCAST_AFX_TEST_SDK_ROOT");
  ::unsetenv("STUDIOCAST_AFX_TEST_EXPECTED_LIB");
  if (old_ld.empty()) {
    ::unsetenv("LD_LIBRARY_PATH");
  } else {
    ::setenv("LD_LIBRARY_PATH", old_ld.c_str(), 1);
  }

  fs::remove_all(root, ec);

  return Require(CommandExitedOk(rc),
                 "AFX loader priority child failed; bare loader path may have "
                 "won over explicit SDK root");
}

bool TestCurrentLinuxMaxineLibraryNamesResolve() {
  const fs::path root =
      fs::temp_directory_path() /
      ("studiocast-maxine-paths-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(root, ec);

  const fs::path vfx = root / "VideoFX";
  const fs::path ar = root / "ARSDK";
  const fs::path afx = root / "Audio_Effects_SDK";

  fs::create_directories(vfx / "lib", ec);
  fs::create_directories(vfx / "models", ec);
  fs::create_directories(vfx / "features", ec);
  fs::create_directories(ar / "lib", ec);
  fs::create_directories(ar / "models", ec);
  fs::create_directories(ar / "features", ec);
  fs::create_directories(afx / "nvafx" / "lib", ec);
  fs::create_directories(afx / "features", ec);
  if (ec) {
    std::cerr << "failed to create test SDK layout: " << ec.message() << "\n";
    return false;
  }

  if (!Touch(vfx / "lib" / "libVideoFX.so") ||
      !Touch(ar / "lib" / "libnvARPose.so") ||
      !Touch(afx / "nvafx" / "lib" / "libnv_audiofx.so")) {
    std::cerr << "failed to create fake Maxine libraries\n";
    fs::remove_all(root, ec);
    return false;
  }

  EnvGuard vfx_env("STUDIOCAST_VFX_SDK_ROOT", vfx.string());
  EnvGuard ar_env("STUDIOCAST_AR_SDK_ROOT", ar.string());
  EnvGuard afx_env("STUDIOCAST_AFX_SDK_ROOT", afx.string());

  const auto rep = studiocast::maxine::ResolveMaxinePaths();

  bool ok = true;
  ok &= Require(rep.vfx.ok, "expected VFX component to resolve");
  ok &= Require(rep.vfx.library == vfx / "lib" / "libVideoFX.so",
                "expected VFX to resolve libVideoFX.so, got " +
                    rep.vfx.library.string());
  ok &= Require(rep.ar.ok, "expected AR component to resolve");
  ok &= Require(rep.ar.library == ar / "lib" / "libnvARPose.so",
                "expected AR to resolve libnvARPose.so, got " +
                    rep.ar.library.string());
  ok &= Require(rep.afx.ok, "expected AFX component to resolve");
  ok &= Require(rep.afx.library == afx / "nvafx" / "lib" / "libnv_audiofx.so",
                "expected AFX to resolve libnv_audiofx.so, got " +
                    rep.afx.library.string());

  fs::remove_all(root, ec);
  return ok;
}

} // namespace

int main(int argc, char **argv) {
  if (std::getenv("STUDIOCAST_AFX_LOADER_PRIORITY_CHILD")) {
    return RunAfxLoaderPriorityChild() ? 0 : 1;
  }

  if (!TestCurrentLinuxMaxineLibraryNamesResolve()) {
    std::cout << "[FAIL] current Linux Maxine library names resolve\n";
    return 1;
  }

  std::cout << "[PASS] current Linux Maxine library names resolve\n";
  if (argc <= 0 || !argv || !argv[0] ||
      !TestAfxLoaderPrefersExplicitSdkRootBeforeBareLoaderPath(argv[0])) {
    std::cout << "[FAIL] AFX loader prefers explicit SDK root before bare "
                 "loader path\n";
    return 1;
  }

  std::cout << "[PASS] AFX loader prefers explicit SDK root before bare "
               "loader path\n";
  return 0;
}
