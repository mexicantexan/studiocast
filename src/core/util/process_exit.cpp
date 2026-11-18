#include "core/util/process_exit.h"

#include <atomic>

#include <cstdio>
#include <cstdlib>

namespace studiocast::util {
namespace {
constinit std::atomic<bool> retained{false};
}
void MarkUnsafeNativeResourcesRetained() noexcept {
  retained.store(true, std::memory_order_relaxed);
}
bool UnsafeNativeResourcesRetained() noexcept {
  return retained.load(std::memory_order_relaxed);
}
int FinalizeProcessExit(int status) noexcept {
  if (!UnsafeNativeResourcesRetained())
    return status;
  std::fputs("ERROR: unsafe native resources retained; application cleanup "
             "finished, skipping driver exit finalizers.\n",
             stderr);
  std::fflush(stdout);
  std::fflush(stderr);
  // A caller's skip code (for example CTest 77) must never hide quarantine.
  std::_Exit(EXIT_FAILURE);
}
} // namespace studiocast::util
