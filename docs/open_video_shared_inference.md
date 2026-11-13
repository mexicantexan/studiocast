# Open Video shared inference contract

- Status: Shared-analysis design and prepared-runtime checkpoint
- Applies To: Open Video shared analysis and latest-frame scheduling primitives
- Audience: Video runtime and model-backend maintainers
- Last Verified: 2026-11-12

StudioCast’s open-source video effects can require **multiple ML inferences per frame**:

- segmentation / matting (for virtual background, auto-frame fallback)
- face detection (for auto frame, key light weighting, eye contact)
- face landmarks (for eye contact)
- video denoise (for video noise removal)

Running the same model twice in a single frame is wasted compute and increases latency. We don't want that.

## Design goal

**One inference per compatible analysis producer per capture sequence**, shared
across effects in the pipeline. Different selected artifacts, configuration, or
execution domains do not become compatible merely because their tasks match.

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
prove production routing. The resident Vulkan owner described below keeps its
analysis and decisions on GPU rather than populating these host cache slots.

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
evidence, but explicit Open CUDA or Vulkan requests must preserve their selected
backend/provider. CPU analysis, another GPU backend, or Maxine is not an
implicit replacement.

## Prepared Open CUDA route

The prepared route has expanded beyond the original single explicit
blur/remove, same-geometry slice. Local production composition is integrated at
`c43717e`: denoise, Eye Contact with YuNet/landmark dependencies, background
modes, key light, Auto Frame, tracked or fixed vignette, resident mirror, and
final geometry share one owned CUDA island in canonical order. Strict and
automatic selection, exact artifact/provider identity, optional failures,
temporal reset, and supported combinations remain subject to the family gate
in [HAL_PARITY_COMPLETION.md](HAL_PARITY_COMPLETION.md). The
[bounded evidence table](HARDWARE_EXECUTION_LAYER.md#evidence-and-remaining-gates)
separates native generated-input execution from final regression/review,
real-camera quality/latency, and release qualification.

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
Explicit and automatic-discontinuity failure guards, exact unsubmitted-binding
retirement, and disabled final-stage timing are integrated at final source
`b204b0a`; see the evidence table for bounded final verification and review. A frame failure
after prepared execution starts must not retry through a different backend or
CPU provider in that frame. Native-to-host denoise/Eye Contact fallback is
selected only during setup under unconstrained automatic intent, following a
permitted availability failure and proven safe cleanup. Strict CUDA or
incompatible provider requests, mixed CUDA semantics, pending cleanup, and
quarantine never take it. Explicit typed CPU compatibility remains separate.
Retained compatibility is a bounded preservation route, not family completion evidence.

## Resident Vulkan shared analysis

The additive non-matting graph contract is documented in
[how_to_add_your_own_model.md](how_to_add_your_own_model.md#vulkan-non-matting-schema-v2).
It preserves original model IDs/ONNX files and MODNet's existing singular
contract. The implementation uses one exact adopted ncnn Vulkan device,
queue, context generation, provider, and aggregate allocation budget. Setup
validates and seals artifacts, binds fixed-capacity tensors, prepares graph and
leaf commands, and warms selected models. Discovery, parsing, hashing, graph
construction, and allocation stay outside the live frame path. A selected
zero-strength effect still validates/prepares its models; its live work is
omitted.

Canonical order is denoise → Eye Contact → lazy shared current-frame
matte/background → key light → Auto Frame → resize → vignette → mirror.
Five-frame denoise history feeds the current resident image used by eye
landmark evaluation. Substituting the original CPU capture for that image
would change the effect. YuNet's twelve heads, decode/stable top-K 5000/NMS,
complete face list, dlib landmarks, eye crop/anchor inputs, Auto Frame crop
planning/smoothing, matte bounds, and tracked vignette center stay resident.
Eye Contact selects highest confidence (area breaks confidence ties); Auto
Frame selects largest face area from the shared list. They must not share
only a preselected face.

Reuse is scoped to the current `capture_sequence` and compatible prepared
producer identity: selected artifacts/dependencies, configuration, source
analysis revision, and domain/context generation. The canonical first eligible
analysis supplies compatible downstream consumers, matching existing shared
analysis semantics; subsequent eye/background pixel changes do not by
themselves request a second compatible detector. A distinct producer or model
requires separate prepared resources and analysis. No stale result from a
previous capture may satisfy a current-frame request. Matting requested by
another compatible consumer can also serve Auto Frame; otherwise the bounded
fallback cadence preserves its existing capture-sequence semantics. An
unavailable current matte is explicitly excluded rather than reusing stale
alpha as current evidence.

FLX learned-prefix inference and the exact sampler/lighting/composite tail are
GPU operations, as are all tracking decisions. Unlike the declared prepared
CUDA metadata tails above, this Vulkan contract permits no intermediate image,
tensor, detection, landmark, or crop-state readback and no CPU tracking tail.
It uses the island's input upload and final output readback/writer boundary.
Offline CPU oracles validate behavior; they are not live fallback paths. Fixed
recorded graphs may execute even when GPU no-face or cadence flags mask their
effect on output. Evidence counts actual graph executions rather than claiming
an inference was skipped because its result was unused.

Nonconsecutive/dropped capture sequences, disable/reconfiguration, model,
provider, context or geometry changes, explicit reset, and execution failure
invalidate the relevant temporal history and shared-analysis validity. A reset
must clear stale crop/center state as well as denoise/eye state. Completion is
bound to exact batch/epoch/sequence/owner-generation tickets. Aborted commands
cannot promote counters; unsafe pending work retains resources until proven
quiescent, and later retirement does not become successful inference. Optional
failure may preserve a valid resident input, while pending completion or failed
reset prevents reuse/publication. Active attribution still requires successful
output completion and writer commit.

These contracts and native oracle coverage do not by themselves establish
production-manager parity, camera quality/latency, recommendation, or vendor
deployment coverage. See [VULKAN_EFFECTS_COMPLETION.md](VULKAN_EFFECTS_COMPLETION.md)
and [HAL_PARITY_COMPLETION.md](HAL_PARITY_COMPLETION.md) for remaining family
gates. Offline candidate publication remains a separate approved transaction;
installed user packs are unchanged by local conversion or native validation.

## Low-latency scheduling primitive

Workstream 5 adds `src/core/open_video/latest_frame_wins_worker.h` as a small
standalone primitive for future heavy video inference integration.

The worker keeps one persistent thread and at most one pending frame. When a
processor is busy, a newer submission replaces the pending frame instead of
growing a queue. Each task carries the current generation; model, configuration,
or frame-size changes can advance the generation so late results from older
work are rejected instead of being published.

This worker is not wired into the synchronous prepared CUDA or Vulkan islands.
The current matting and virtual-background behavior remains synchronous and
deterministic. The HAL completion token is evolution-shaped but already
complete in v1; genuine asynchronous scheduling must first define bounded
in-flight ownership, reuse protection, cancellation, stale-generation handling,
device-loss quarantine, stop-aware waits, shutdown, and temporal dependencies.
