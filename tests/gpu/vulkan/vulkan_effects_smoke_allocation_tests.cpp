#include "vulkan_effects_smoke_allocations.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace {
constexpr auto kAlignment = std::align_val_t{64};

void *Allocate(std::size_t size, bool array, bool aligned, bool nothrow) {
  if (aligned) {
    if (nothrow)
      return array ? ::operator new[](size, kAlignment, std::nothrow)
                   : ::operator new(size, kAlignment, std::nothrow);
    return array ? ::operator new[](size, kAlignment)
                 : ::operator new(size, kAlignment);
  }
  if (nothrow)
    return array ? ::operator new[](size, std::nothrow)
                 : ::operator new(size, std::nothrow);
  return array ? ::operator new[](size) : ::operator new(size);
}

void Deallocate(void *memory, std::size_t size, bool array, bool aligned,
                unsigned mode) {
  // Nothrow new normally reaches ordinary delete; its matching placement
  // delete is also required when a constructor throws.
  if (aligned) {
    if (mode == 2) {
      if (array)
        ::operator delete[](memory, kAlignment, std::nothrow);
      else
        ::operator delete(memory, kAlignment, std::nothrow);
    } else if (mode == 1) {
      if (array)
        ::operator delete[](memory, size, kAlignment);
      else
        ::operator delete(memory, size, kAlignment);
    } else if (array) {
      ::operator delete[](memory, kAlignment);
    } else {
      ::operator delete(memory, kAlignment);
    }
  } else if (mode == 2) {
    if (array)
      ::operator delete[](memory, std::nothrow);
    else
      ::operator delete(memory, std::nothrow);
  } else if (mode == 1) {
    if (array)
      ::operator delete[](memory, size);
    else
      ::operator delete(memory, size);
  } else if (array) {
    ::operator delete[](memory);
  } else {
    ::operator delete(memory);
  }
}

bool CheckPairs(bool counted) {
  const auto initial = allocations.load(std::memory_order_relaxed);
  std::uint64_t expected = initial;
  // Explicit operator calls prevent new/delete-expression allocation elision.
  for (std::size_t size : {std::size_t{0}, std::size_t{37}}) {
    for (unsigned flags = 0; flags != 8; ++flags) {
      const bool array = (flags & 1) != 0;
      const bool aligned = (flags & 2) != 0;
      const bool nothrow = (flags & 4) != 0;
      for (unsigned mode = 0; mode != (nothrow ? 3U : 2U); ++mode) {
        void *memory = Allocate(size, array, aligned, nothrow);
        if (!memory || (aligned && reinterpret_cast<std::uintptr_t>(memory) %
                                       static_cast<std::size_t>(kAlignment))) {
          std::fprintf(stderr, "FAIL allocation/alignment flags=%u mode=%u\n",
                       flags, mode);
          return false;
        }
        std::memset(memory, 0xa5, size ? size : 1);
        expected += counted ? 1U : 0U;
        const auto after_new = allocations.load(std::memory_order_relaxed);
        Deallocate(memory, size, array, aligned, mode);
        if (after_new != expected ||
            allocations.load(std::memory_order_relaxed) != expected) {
          std::fprintf(stderr, "FAIL allocation count flags=%u mode=%u\n",
                       flags, mode);
          return false;
        }
      }
    }
  }
  return expected == initial + (counted ? 40U : 0U);
}
} // namespace

int main() {
  if (!CheckPairs(false))
    return 1;
  count_allocations.store(true, std::memory_order_relaxed);
  const bool counted = CheckPairs(true);
  count_allocations.store(false, std::memory_order_relaxed);
  if (!counted || !CheckPairs(false) ||
      allocations.load(std::memory_order_relaxed) != 40)
    return 1;
  std::puts("PASS 120 allocation/delete pairs; exactly 40 gated allocations, "
            "including scalar/array/aligned/nothrow and zero-size forms");
  return 0;
}
