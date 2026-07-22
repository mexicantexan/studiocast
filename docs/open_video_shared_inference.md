# Open Video shared inference contract

- Status: Design note with an implemented cache/worker contract
- Applies To: Open Video shared analysis and latest-frame scheduling primitives
- Audience: Video runtime and model-backend maintainers
- Last Verified: 2026-07-22

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

## Low-latency scheduling primitive

Workstream 5 adds `src/core/open_video/latest_frame_wins_worker.h` as a small
standalone primitive for future heavy video inference integration.

The worker keeps one persistent thread and at most one pending frame. When a
processor is busy, a newer submission replaces the pending frame instead of
growing a queue. Each task carries the current generation; model, configuration,
or frame-size changes can advance the generation so late results from older
work are rejected instead of being published.

This is not wired into the Open CUDA virtual-background path yet. The current
matting and VB behavior remains synchronous and deterministic by default.
