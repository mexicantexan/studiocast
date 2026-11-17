#pragma once

// Include in exactly one translation unit per executable. The smoke and its
// GPU-free regression test must exercise these same global replacements.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <new>

namespace {
std::atomic<bool> count_allocations{false};
std::atomic<std::uint64_t> allocations{0};
void CountAllocation() noexcept {
  if (count_allocations.load(std::memory_order_relaxed))
    allocations.fetch_add(1, std::memory_order_relaxed);
}
} // namespace
void *operator new(std::size_t size) {
  CountAllocation();
  if (auto *memory = std::malloc(size ? size : 1))
    return memory;
  throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept {
  std::free(memory);
}
void *operator new(std::size_t size, std::align_val_t alignment) {
  CountAllocation();
  void *memory = nullptr;
  if (posix_memalign(&memory, static_cast<std::size_t>(alignment),
                     size ? size : 1) != 0)
    throw std::bad_alloc();
  return memory;
}
void *operator new[](std::size_t size, std::align_val_t alignment) {
  return ::operator new(size, alignment);
}
void operator delete(void *memory, std::align_val_t) noexcept {
  std::free(memory);
}
void operator delete[](void *memory, std::align_val_t) noexcept {
  std::free(memory);
}
void operator delete(void *memory, std::size_t, std::align_val_t) noexcept {
  std::free(memory);
}
void operator delete[](void *memory, std::size_t, std::align_val_t) noexcept {
  std::free(memory);
}

// Replace the nothrow family too: sanitizer/runtime defaults may otherwise
// allocate with their operator new and reach our free-based ordinary delete.
void *operator new(std::size_t size, const std::nothrow_t &) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}
void *operator new[](std::size_t size, const std::nothrow_t &) noexcept {
  return ::operator new(size, std::nothrow);
}
void *operator new(std::size_t size, std::align_val_t alignment,
                   const std::nothrow_t &) noexcept {
  try {
    return ::operator new(size, alignment);
  } catch (...) {
    return nullptr;
  }
}
void *operator new[](std::size_t size, std::align_val_t alignment,
                     const std::nothrow_t &) noexcept {
  return ::operator new(size, alignment, std::nothrow);
}
void operator delete(void *memory, const std::nothrow_t &) noexcept {
  std::free(memory);
}
void operator delete[](void *memory, const std::nothrow_t &) noexcept {
  std::free(memory);
}
void operator delete(void *memory, std::align_val_t,
                     const std::nothrow_t &) noexcept {
  std::free(memory);
}
void operator delete[](void *memory, std::align_val_t,
                       const std::nothrow_t &) noexcept {
  std::free(memory);
}
