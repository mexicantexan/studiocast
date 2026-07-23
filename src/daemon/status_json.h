#pragma once

#include "core/video/prepared_hardware_execution_owner.h"
#include "core/video/video_runtime_status.h"

#include <string>

namespace studiocast::daemon {

struct ProductCapabilityJsonResult {
  bool ok = false;
  video::CapabilityValidationResult validation{};
  std::string json;
};

struct VideoRuntimeStatusJsonResult {
  bool ok = false;
  video::RuntimeStatusValidationResult validation{};
  std::string json;
};

struct PreparedHardwareExecutionJsonResult {
  bool ok = false;
  bool snapshot_valid = false;
  bool active_attribution_valid = false;
  std::string validation_code;
  std::string json;
};

// Pure control-plane serializers. They consume an already-cached snapshot and
// perform no probing, refresh, model discovery, session work, synchronization,
// or pipeline mutation. The returned JSON allocation belongs on GET_STATUS or
// another bounded diagnostics path, never in the frame loop.
[[nodiscard]] ProductCapabilityJsonResult SerializeProductCapabilityStatusJson(
    const video::ProductCapabilitySnapshot &snapshot);

// Produces the additive typed schema and compatibility projections used by
// existing GET_STATUS consumers:
//   engines.{maxine,open_cuda,open_vulkan}
//   video.compute
//   video.effect_readiness
//   video.pipeline.effects_backends
// Invalid snapshots return ok=false and no JSON, preventing partial or
// success-looking legacy facts from escaping.
[[nodiscard]] VideoRuntimeStatusJsonResult SerializeVideoRuntimeStatusJson(
    const video::VideoRuntimeStatusSnapshot &snapshot);

// Serializes the bounded prepared-hardware migration snapshot directly from
// typed structures. supervisor_pipeline_running is an authority gate: cached
// setup, telemetry, or an old commit never imply live activity after stop or
// while a start is still in progress. Malformed snapshots still produce a
// bounded fail-closed diagnostics object, but never active attribution.
[[nodiscard]] PreparedHardwareExecutionJsonResult
SerializePreparedHardwareExecutionJson(
    const video::PreparedHardwareExecutionSnapshot &snapshot,
    bool supervisor_pipeline_running,
    bool supervisor_pipeline_starting = false);

} // namespace studiocast::daemon
