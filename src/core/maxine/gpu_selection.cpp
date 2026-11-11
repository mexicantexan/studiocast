#include "core/maxine/gpu_selection.h"

#include "core/util/exec.h"
#include "core/util/strings.h"

#include <algorithm>
#include <sstream>

namespace studiocast::maxine {

namespace {

std::vector<SelectedGpu>
ListGpusViaNvidiaSmi(std::string *error_out,
                     const std::atomic_bool *stop_requested) {
  util::ExecCaptureOptions options;
  options.stop_requested = stop_requested;
  const auto res =
      util::ExecCapture("nvidia-smi --query-gpu=index,uuid,name,compute_cap "
                        "--format=csv,noheader,nounits",
                        options);
  if (res.cancelled) {
    if (error_out)
      *error_out = "nvidia-smi cancelled during service shutdown.";
    return {};
  }
  if (res.exit_code != 0) {
    if (error_out) {
      *error_out =
          "nvidia-smi failed (exit code " + std::to_string(res.exit_code) + ")";
    }
    return {};
  }

  std::vector<SelectedGpu> gpus;
  for (auto line : util::SplitLines(res.stdout_str)) {
    line = util::TrimCopy(line);
    if (line.empty()) {
      continue;
    }

    // Expected format:
    //   0, GPU-xxxx..., NVIDIA GeForce RTX..., 8.9
    // This is CSV without quoted fields.
    const auto fields = util::Split(line, ',');
    if (fields.size() < 4) {
      continue;
    }

    SelectedGpu gpu;
    gpu.index = std::stoi(util::TrimCopy(fields[0]));
    gpu.uuid = util::TrimCopy(fields[1]);

    // Name can theoretically contain commas; join middle fields.
    {
      std::ostringstream name;
      for (size_t i = 2; i + 1 < fields.size(); ++i) {
        if (i != 2) {
          name << ",";
        }
        name << util::TrimCopy(fields[i]);
      }
      gpu.name = name.str();
    }

    const std::string cc_str = util::TrimCopy(fields.back());
    const auto dot = cc_str.find('.');
    if (dot != std::string::npos) {
      const std::string maj = cc_str.substr(0, dot);
      const std::string min = cc_str.substr(dot + 1);
      try {
        gpu.compute_capability = {std::stoi(maj), std::stoi(min)};
      } catch (...) {
        // leave unknown
      }
    }

    gpus.push_back(std::move(gpu));
  }

  if (gpus.empty() && error_out) {
    *error_out = "No GPUs returned by nvidia-smi.";
  }
  return gpus;
}

} // namespace

std::string SelectedGpu::ComputeCapString() const {
  if (!compute_capability.has_value()) {
    return "unknown";
  }
  return std::to_string(compute_capability->first) + "." +
         std::to_string(compute_capability->second);
}

bool IsComputeCapabilitySupported(int major, int minor) {
  // Maxine VFX requires Tensor cores; in practice this means Turing (7.5) or
  // newer.
  if (major > 7) {
    return true;
  }
  if (major == 7 && minor >= 5) {
    return true;
  }
  return false;
}

GpuSelectionResult SelectGpu(const config::GpuSelection &policy,
                             const std::atomic_bool *stop_requested) {
  GpuSelectionResult result;

  std::string list_error;
  result.all_gpus = ListGpusViaNvidiaSmi(&list_error, stop_requested);
  if (result.all_gpus.empty()) {
    result.error = list_error.empty() ? "No NVIDIA GPUs found." : list_error;
    return result;
  }

  auto pick_best_supported = [&]() -> std::optional<SelectedGpu> {
    std::optional<SelectedGpu> best;
    for (const auto &gpu : result.all_gpus) {
      if (!gpu.compute_capability.has_value()) {
        continue;
      }
      const auto [maj, min] = *gpu.compute_capability;
      if (!IsComputeCapabilitySupported(maj, min)) {
        continue;
      }
      if (!best.has_value()) {
        best = gpu;
        continue;
      }
      const auto [bmaj, bmin] = *best->compute_capability;
      if (maj > bmaj || (maj == bmaj && min > bmin)) {
        best = gpu;
      }
    }
    return best;
  };

  if (policy.mode == config::GpuSelectMode::Uuid) {
    if (policy.uuid.empty()) {
      result.error = "GPU selection mode is uuid, but uuid is empty.";
      return result;
    }
    for (const auto &gpu : result.all_gpus) {
      if (gpu.uuid == policy.uuid) {
        if (!gpu.compute_capability.has_value()) {
          result.error = "Selected GPU has unknown compute capability.";
          return result;
        }
        const auto [maj, min] = *gpu.compute_capability;
        if (!IsComputeCapabilitySupported(maj, min)) {
          result.error = "Selected GPU compute capability " +
                         gpu.ComputeCapString() +
                         " is not supported by Maxine.";
          return result;
        }
        result.selected = gpu;
        return result;
      }
    }
    result.error = "Requested GPU UUID not found: " + policy.uuid;
    return result;
  }

  if (policy.mode == config::GpuSelectMode::Index) {
    if (!policy.index.has_value()) {
      result.error = "GPU selection mode is index, but index is not set.";
      return result;
    }
    for (const auto &gpu : result.all_gpus) {
      if (gpu.index == *policy.index) {
        if (!gpu.compute_capability.has_value()) {
          result.error = "Selected GPU has unknown compute capability.";
          return result;
        }
        const auto [maj, min] = *gpu.compute_capability;
        if (!IsComputeCapabilitySupported(maj, min)) {
          result.error = "Selected GPU compute capability " +
                         gpu.ComputeCapString() +
                         " is not supported by Maxine.";
          return result;
        }
        result.selected = gpu;
        return result;
      }
    }
    result.error =
        "Requested GPU index not found: " + std::to_string(*policy.index);
    return result;
  }

  // Auto
  result.selected = pick_best_supported();
  if (!result.selected.has_value()) {
    result.error =
        "No supported NVIDIA GPU found (requires Turing/RTX class or newer).";
  }
  return result;
}

} // namespace studiocast::maxine
