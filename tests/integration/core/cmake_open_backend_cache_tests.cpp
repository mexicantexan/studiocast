#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <unistd.h>

#include "core/util/exec.h"

#ifndef STUDIOCAST_SOURCE_DIR
#define STUDIOCAST_SOURCE_DIR ""
#endif

#ifndef STUDIOCAST_CMAKE_COMMAND
#define STUDIOCAST_CMAKE_COMMAND "cmake"
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

std::string ReadFile(const fs::path &path) {
  std::ifstream in(path);
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

bool TestMissingOnnxRuntimeDoesNotForceOpenBackendsOff() {
  ScopedTempDir temp("studiocast-cmake-open-backend-cache");
  if (!Expect(temp.ok(), temp.error().c_str()))
    return false;

  const fs::path repo = fs::path(STUDIOCAST_SOURCE_DIR);
  const fs::path buildDir = temp.path() / "build";
  const fs::path noPkgConfig = temp.path() / "empty-pkgconfig";
  std::error_code ec;
  fs::create_directories(noPkgConfig, ec);
  if (!Expect(!ec, "failed to create empty pkg-config directory"))
    return false;

  std::string command = "env -u ONNXRUNTIME_ROOT PKG_CONFIG_LIBDIR=" +
                        ShellQuote(noPkgConfig.string()) + " " +
                        ShellQuote(STUDIOCAST_CMAKE_COMMAND) + " -S " +
                        ShellQuote(repo.string()) + " -B " +
                        ShellQuote(buildDir.string()) +
                        " -DBUILD_TESTING=OFF"
                        " -DSTUDIOCAST_ENABLE_DLIB=OFF"
                        " -DSTUDIOCAST_ENABLE_OPEN_CUDA=ON"
                        " -DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON"
                        " -DSTUDIOCAST_ENABLE_NCNN_SPIKE=ON"
                        " -DSTUDIOCAST_ENABLE_OPEN_AUDIO=ON"
                        " -DCMAKE_DISABLE_FIND_PACKAGE_onnxruntime=ON"
                        " -DCMAKE_DISABLE_FIND_PACKAGE_ncnn=ON"
                        " -DCMAKE_DISABLE_FIND_PACKAGE_Python3=ON"
                        " 2>&1";

  studiocast::util::ExecCaptureOptions options;
  options.timeout_ms = 60000;
  options.max_output_bytes = 2 * 1024 * 1024;
  const auto result = studiocast::util::ExecCapture(command, options);
  if (!Expect(result.exit_code == 0,
              "nested CMake configure without ONNX Runtime should succeed")) {
    std::cerr << result.stdout_str << "\n";
    return false;
  }

  const std::string cache = ReadFile(buildDir / "CMakeCache.txt");
  bool ok = ExpectContains("nested CMake cache", cache,
                           "STUDIOCAST_ENABLE_OPEN_CUDA:BOOL=ON") &&
            ExpectContains("nested CMake cache", cache,
                           "STUDIOCAST_ENABLE_OPEN_VULKAN:BOOL=ON") &&
            ExpectContains("nested CMake cache", cache,
                           "STUDIOCAST_ENABLE_NCNN_SPIKE:BOOL=ON") &&
            ExpectContains("nested CMake cache", cache,
                           "STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING:BOOL=OFF") &&
            ExpectContains("nested CMake cache", cache,
                           "STUDIOCAST_ENABLE_OPEN_AUDIO:BOOL=ON") &&
            Expect(cache.find("STUDIOCAST_ENABLE_OPEN_CUDA:BOOL=OFF") ==
                       std::string::npos,
                   "Open CUDA must not be force-cached OFF when ONNX Runtime is "
                   "missing");
  if (!ok)
    return false;

  const fs::path defaultBuildDir = temp.path() / "build-defaults";
  command = "env -u ONNXRUNTIME_ROOT PKG_CONFIG_LIBDIR=" +
            ShellQuote(noPkgConfig.string()) + " " +
            ShellQuote(STUDIOCAST_CMAKE_COMMAND) + " -S " +
            ShellQuote(repo.string()) + " -B " +
            ShellQuote(defaultBuildDir.string()) +
            " -DBUILD_TESTING=OFF"
            " -DSTUDIOCAST_ENABLE_DLIB=OFF"
            " -DSTUDIOCAST_ENABLE_OPEN_CUDA=OFF"
            " -DSTUDIOCAST_ENABLE_OPEN_AUDIO=OFF"
            " -DCMAKE_DISABLE_FIND_PACKAGE_onnxruntime=ON"
            " -DCMAKE_DISABLE_FIND_PACKAGE_Python3=ON"
            " 2>&1";
  const auto defaultResult =
      studiocast::util::ExecCapture(command, options);
  if (!Expect(defaultResult.exit_code == 0,
              "nested CMake configure with Open Vulkan default should "
              "succeed")) {
    std::cerr << defaultResult.stdout_str << "\n";
    return false;
  }

  const std::string defaultCache =
      ReadFile(defaultBuildDir / "CMakeCache.txt");
  if (!ExpectContains("nested default CMake cache", defaultCache,
                      "STUDIOCAST_ENABLE_OPEN_VULKAN:BOOL=OFF") ||
      !ExpectContains("nested default CMake cache", defaultCache,
                      "STUDIOCAST_ENABLE_NCNN_VULKAN_MATTING:BOOL=OFF")) {
    return false;
  }

  const fs::path missingNcnnBuild = temp.path() / "build-production-no-ncnn";
  command = "env PKG_CONFIG_LIBDIR=" + ShellQuote(noPkgConfig.string()) + " " +
            ShellQuote(STUDIOCAST_CMAKE_COMMAND) + " -S " +
            ShellQuote(repo.string()) + " -B " +
            ShellQuote(missingNcnnBuild.string()) +
            " -DBUILD_TESTING=OFF"
            " -DSTUDIOCAST_ENABLE_DLIB=OFF"
            " -DSTUDIOCAST_ENABLE_OPEN_CUDA=OFF"
            " -DSTUDIOCAST_ENABLE_OPEN_AUDIO=OFF"
            " -DSTUDIOCAST_ENABLE_OPEN_VULKAN=ON"
            " -DSTUDIOCAST_ENABLE_NCNN_VULKAN_MATTING=ON"
            " -DCMAKE_DISABLE_FIND_PACKAGE_ncnn=ON"
            " 2>&1";
  const auto missingNcnnResult =
      studiocast::util::ExecCapture(command, options);
  if (!Expect(missingNcnnResult.exit_code != 0,
              "production ncnn Vulkan configure must reject a missing dependency") ||
      !ExpectContains("missing production ncnn configure",
                      missingNcnnResult.stdout_str,
                      "requires ncnn built with Vulkan")) {
    std::cerr << missingNcnnResult.stdout_str << "\n";
    return false;
  }

  const fs::path noVulkanBuild = temp.path() / "build-production-no-vulkan";
  command = ShellQuote(STUDIOCAST_CMAKE_COMMAND) + " -S " +
            ShellQuote(repo.string()) + " -B " +
            ShellQuote(noVulkanBuild.string()) +
            " -DBUILD_TESTING=OFF"
            " -DSTUDIOCAST_ENABLE_DLIB=OFF"
            " -DSTUDIOCAST_ENABLE_OPEN_CUDA=OFF"
            " -DSTUDIOCAST_ENABLE_OPEN_AUDIO=OFF"
            " -DSTUDIOCAST_ENABLE_OPEN_VULKAN=OFF"
            " -DSTUDIOCAST_ENABLE_NCNN_VULKAN_MATTING=ON"
            " 2>&1";
  const auto noVulkanResult = studiocast::util::ExecCapture(command, options);
  if (!Expect(noVulkanResult.exit_code != 0,
              "production ncnn Vulkan configure must require Open Vulkan") ||
      !ExpectContains("missing Open Vulkan configure",
                      noVulkanResult.stdout_str,
                      "STUDIOCAST_ENABLE_OPEN_VULKAN=ON.")) {
    std::cerr << noVulkanResult.stdout_str << "\n";
    return false;
  }
  return true;
}

} // namespace

int main() {
  return TestMissingOnnxRuntimeDoesNotForceOpenBackendsOff() ? 0 : 1;
}
