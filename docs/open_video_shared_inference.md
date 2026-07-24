# Open Video shared inference contract

- Status: Design note with an implemented cache/worker contract
- Applies To: Open Video shared analysis and latest-frame scheduling primitives
- Audience: Video runtime and model-backend maintainers
- Last Verified: 2026-07-23

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

Future phases will extend this contract with additional cached artifacts (e.g. eye-crop tensors
or denoise frame history) and wire it into the video pipeline.

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

The first production Open CUDA HAL route is intentionally smaller than the
shared-inference design above. It owns one strict ONNX Runtime CUDA
virtual-background blur or remove stage with:

- one setup-validated Open Video matting pack and artifact identity;
- one owned CUDA execution domain, stream, allocator, and context generation;
- prepared preprocessing, CUDA IoBinding inference, postprocessing, buffers,
  scratch, and bounded completion waiting;
- one upload, no intermediate host crossing, one final readback, one
  synchronization, and one declared writer-copy CPU tail; and
- active attribution only after the completed output is written successfully.

Eligibility currently requires an explicit Open CUDA effect engine, CUDA (or
automatic compute allowance), tight same-geometry RGB24 host transport, one
blur/remove semantic, TensorRT disabled, and no replacement, mixed resident
stage, shared-analysis consumer, temporal history, scaling, or output
conversion. Model discovery, manifest parsing, hashing, provider selection,
session construction, and resource sizing happen during setup, not per frame.

If prepared setup is ineligible or unavailable, the same explicit Open CUDA
compatibility route may remain in use. Once prepared execution begins, a frame
failure does not retry through the compatibility implementation in that frame
and never selects a different GPU backend or CPU provider.

## Low-latency scheduling primitive

Workstream 5 adds `src/core/open_video/latest_frame_wins_worker.h` as a small
standalone primitive for future heavy video inference integration.

The worker keeps one persistent thread and at most one pending frame. When a
processor is busy, a newer submission replaces the pending frame instead of
growing a queue. Each task carries the current generation; model, configuration,
or frame-size changes can advance the generation so late results from older
work are rejected instead of being published.

This worker is not wired into the prepared Open CUDA virtual-background route.
The current matting and virtual-background behavior remains synchronous and
deterministic. The HAL completion token is evolution-shaped but already
complete in v1; genuine asynchronous scheduling must first define bounded
in-flight ownership, reuse protection, cancellation, stale-generation handling,
device-loss quarantine, stop-aware waits, shutdown, and temporal dependencies.
