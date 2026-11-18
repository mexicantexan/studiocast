#pragma once

#include "core/util/exec.h"
#include "core/util/process_exit.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace studiocast::tests {
inline constexpr std::string_view kNativeRetentionChildArgument =
    "--native-retention-child";

inline bool ExpectedNativeRetentionResult(const util::ExecResult &result,
                                          std::string_view marker) {
  return result.exit_code == 1 && !result.timed_out && !result.cancelled &&
         result.stdout_str.find(std::string(marker) + '\n') !=
             std::string::npos &&
         result.stdout_str.find("unsafe native resources retained;") !=
             std::string::npos;
}

inline bool CheckNativeRetentionChild(std::string_view marker) {
  std::error_code error;
  const auto executable =
      std::filesystem::read_symlink("/proc/self/exe", error);
  if (error) {
    std::cerr << "Cannot resolve native retention test executable: "
              << error.message() << '\n';
    return false;
  }
  std::string quoted = "'";
  for (const char value : executable.string()) {
    if (value == '\'')
      quoted += "'\\''";
    else
      quoted += value;
  }
  quoted += '\'';
  util::ExecCaptureOptions options;
  options.timeout_ms = 60000;
  // exec keeps a native signal distinguishable from a normal child exit.
  const auto result = util::ExecCapture(
      "exec " + quoted + " " + std::string(kNativeRetentionChildArgument) +
          " 2>&1",
      options);
  std::cout << result.stdout_str;
  if (!ExpectedNativeRetentionResult(result, marker)) {
    std::cerr << "Native retention child did not prove the expected terminal "
                 "failure: exit="
              << result.exit_code << " timed_out=" << result.timed_out
              << " cancelled=" << result.cancelled << '\n';
    return false;
  }
  return true;
}

inline int FailNativeSetup(std::string_view message) {
  std::cerr << "Native retention child setup failed: " << message << '\n';
  return 1;
}

inline int FinishNativeRetentionChild(int status, std::string_view marker) {
  if (status != 0 || !util::UnsafeNativeResourcesRetained()) {
    std::cerr << "Native retention child did not complete its assertions and "
                 "retain the unsafe native owner.\n";
    return util::FinalizeProcessExit(status == 0 ? 1 : status);
  }
  // RunMain has returned: every ordinary local owner has already been
  // destroyed.
  std::cout << marker << '\n';
  std::cout.flush();
  return util::FinalizeProcessExit(0);
}
} // namespace studiocast::tests
