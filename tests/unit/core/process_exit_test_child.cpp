#include "core/util/process_exit.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string_view>

namespace {
void AtExit() { std::puts("atexit"); }
int RunMain(bool retain, int status) {
  struct Owner {};
  struct Local {
    bool retain;
    std::shared_ptr<Owner> owner = std::make_shared<Owner>();
    std::weak_ptr<Owner> witness = owner;
    ~Local() {
      if (retain) {
        // Model actual permanent quarantine without retaining through witness.
        (void)new std::shared_ptr<Owner>(std::move(owner));
        studiocast::util::MarkUnsafeNativeResourcesRetained();
        std::puts(witness.expired() ? "owner_lost" : "owner_retained");
      }
      std::puts("local_cleanup");
    }
  } local{retain};
  return status;
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 3 || std::atexit(AtExit) != 0)
    return 2;
  const bool retain = std::string_view(argv[1]) == "retain";
  const std::string_view requested_status(argv[2]);
  const int status = requested_status == "zero"   ? 0
                     : requested_status == "skip" ? 77
                                                  : 7;
  return studiocast::util::FinalizeProcessExit(RunMain(retain, status));
}
