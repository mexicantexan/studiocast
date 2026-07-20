#pragma once

#include <atomic>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/config/settings.h"

namespace studiocast::maxine {

struct SelectedGpu {
  int index = -1; // CUDA / nvidia-smi GPU index
  std::string uuid;
  std::string name;
  std::optional<std::pair<int, int>> compute_capability; // {major, minor}

  bool IsComputeCapKnown() const { return compute_capability.has_value(); }
  std::string ComputeCapString() const;
};

struct GpuSelectionResult {
  std::vector<SelectedGpu> all_gpus;
  std::optional<SelectedGpu> selected;
  std::string error;

  bool ok() const { return selected.has_value() && error.empty(); }
};

// Maxine VFX is effectively RTX-class/Turing+ in practice (Tensor cores).
bool IsComputeCapabilitySupported(int major, int minor);

GpuSelectionResult SelectGpu(const config::GpuSelection &policy,
                             const std::atomic_bool *stop_requested = nullptr);

} // namespace studiocast::maxine
