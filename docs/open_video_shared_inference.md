# Open Video shared inference contract

- Status: Shared-analysis design and prepared-runtime checkpoint
- Applies To: Open Video shared analysis and latest-frame scheduling primitives
- Audience: Video runtime and model-backend maintainers
- Last Verified: 2026-11-10

StudioCast’s open-source video effects can require **multiple ML inferences per frame**:

- segmentation / matting (for virtual background, auto-frame fallback)
- face detection (for auto frame, key light weighting, eye contact)
- face landmarks (for eye contact)
- video denoise (for video noise removal)

Running the same model twice in a single frame is wasted compute and increases latency. We don't want that.

## Design goal

**One inference per task per frame**, shared across effects in the pipeline.

StudioCast achieves this by keeping a small per-frame cache keyed by a monotonically increasing
`capture_sequence` id.

## Current implementation

Phase M6 introduces a lightweight contract:

- `src/core/open_video/frame_analysis_cache.h`

It provides:

- `FrameAnalysisCache::BeginFrame(seq)` to clear stale data
- cached slots for:
  - `face_detections`
  - `face_landmarks`

The prepared CUDA owner also composes same-frame matting, face detection,
landmarks, and dependent effects in one owned execution domain. That bounded
resident composition has its own prepared resources and temporal history; the
host `FrameAnalysisCache` example alone does not describe its GPU ownership or
prove production routing.

## Expected usage pattern

At the start of a camera frame:

```cpp
cache.BeginFrame(capture_sequence);
```

Then each effect can do:

```cpp
if (!cache.face_detections) {
  cache.face_detections = RunFaceDetector(...);
}
// Use cache.face_detections
```

## Notes

- Caching is **best-effort**: if a model is missing, too slow, or fails, the cache entry can remain empty.
- Effects must always implement a safe fallback path.

“Safe fallback” does not mean changing a strict GPU/provider request. Optional
effects may preserve a valid input frame and publish exact degradation
evidence, but an explicit Open CUDA request must not silently run ONNX Runtime
CPU, Maxine, or Vulkan.

## Prepared Open CUDA route

The prepared route has expanded beyond the original single explicit
blur/remove, same-geometry slice. Background modes, key light, final geometry,
and fixed-center vignette have production routing; prepared native mirror,
YuNet, landmarks, Auto Frame, Eye Contact, and temporal FastDVDnet components
are composed by the runtime candidate under verification. Strict and automatic
selection, exact artifact/provider identity, optional failures, temporal reset,
and supported combinations still require the family gate in
[HAL_PARITY_COMPLETION.md](HAL_PARITY_COMPLETION.md). Component implementation
must not be reported as completed production parity.

The camera pipeline and production smoke executable use the shared
`BuildProductionVideoBackendSetupRequest` builder in
`src/core/video/production_video_setup_request.*`. It freezes canonical stage
order, parameters, selection, geometry, and temporal intent during setup.
Selected Maxine matting alone adds greenscreen temporal history; the Maxine
knob does not make CUDA matting temporal. Denoise keeps five-frame history;
Auto Frame and Eye Contact retain their own history/discontinuity contracts.

An eligible CUDA island owns one exact CUDA domain, stream, allocator, and
context generation. Setup validates model packs/artifact identity, prepares
preprocessing and postprocessing kernels, IoBinding sessions, replacement
images, buffers, scratch, and bounded completion waiting. Model discovery,
manifest parsing, hashing, provider selection, and resource sizing stay outside
the frame loop. Native kernels use their exact CUDA-kernel provider; inference
uses ONNX Runtime CUDA. No strict CUDA stage silently uses a CPU provider.

Frame pixels use one upload and one final readback, followed by the declared
writer/output tail. Analysis consumers additionally declare bounded metadata
readbacks and CPU detection/tracking/crop-plan tails: face analysis allows at
most twelve metadata readbacks per frame, and matte bounds at most one. These
are measured crossings, not an unqualified zero-intermediate-readback claim.
Exact CUDA and Vulkan mirror can stay resident; automatic selection and Maxine
preserve the established measured host mirror finalizer.

Active attribution requires a completed output and successful writer commit.
A failed explicit or automatic temporal reset must prevent further execution
and publication until valid preparation, and clear stale committed attribution.
The automatic-discontinuity failure guard is integrated at `af2d1db`; final
integrated verification remains pending at this checkpoint. A frame failure
after prepared execution starts must not retry through a different backend or
CPU provider in that frame. Retained
compatibility is a bounded preservation route, not family completion evidence.

## Low-latency scheduling primitive

Workstream 5 adds `src/core/open_video/latest_frame_wins_worker.h` as a small
standalone primitive for future heavy video inference integration.

The worker keeps one persistent thread and at most one pending frame. When a
processor is busy, a newer submission replaces the pending frame instead of
growing a queue. Each task carries the current generation; model, configuration,
or frame-size changes can advance the generation so late results from older
work are rejected instead of being published.

This worker is not wired into the synchronous prepared CUDA execution island.
The current matting and virtual-background behavior remains synchronous and
deterministic. The HAL completion token is evolution-shaped but already
complete in v1; genuine asynchronous scheduling must first define bounded
in-flight ownership, reuse protection, cancellation, stale-generation handling,
device-loss quarantine, stop-aware waits, shutdown, and temporal dependencies.
