#pragma once

namespace studiocast::util {
// Permanent native quarantine is irreversible for this process. Temporary
// timeout/pending shutdown is not sufficient to set this latch.
void MarkUnsafeNativeResourcesRetained() noexcept;
[[nodiscard]] bool UnsafeNativeResourcesRetained() noexcept;

// Call only after the application body and all its local owners have returned.
// Unsafe retained drivers must not run userspace exit finalizers. This is a
// terminal failure, not quiescence proof or a clean sanitizer conclusion.
[[nodiscard]] int FinalizeProcessExit(int status) noexcept;
} // namespace studiocast::util
