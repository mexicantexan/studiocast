# Complete production Vulkan effects

- Status: Implementation in progress; discovery and shared-contract freeze complete
- Baseline: `15b8862398bbae774f8156ed70c2378245c9bef9`, branch `v0.2.9`
- User decision, 2026-11-12: finish the remaining Vulkan model effects in this
  pass rather than keep them outside the earlier matting phase. The prior
  permission to record unavailable hardware coverage for deployment feedback
  remains; integrity, ownership, residency, truthful readiness, and bounded
  execution are not waived.

## Completion scope

Implement FastDVDnet denoise, YuNet face inference and Auto Frame, GPU dlib
landmarks and FLX Eye Contact, and tracked-center vignette through the actual
production manager, alongside the existing background modes, key light, resize,
fixed vignette, and mirror. Preserve canonical parameters, explicit/default
model selection, dependency identity, optional failure isolation, committed
attribution, and temporal resets. Include converted model contracts, offline
reproduction/numerical tests, catalog/installer integration, normal deployment
runtime provisioning, current changelog/documentation, and fresh independent
critique after integration. Other Vulkan model effects are no longer excluded.

Model publication still requires a concrete verified artifact proposal and
explicit external-publication authority under AGENTS.md. The prior four-file
MODNet approval does not silently authorize new bytes. Installed user packs,
host packages/services/drivers, application release, and Git push remain outside
local validation. Prepare all authorized work before requesting any final
external approval.

## Frozen implementation contract

- One exact ncnn-adopted physical/logical device, queue, provider, and ownership
  domain. Preserve replay ABI 2 safeguards and sealed checksum-verified artifact
  snapshots. Query/retain FP64 supported/enabled proof on that same context for
  the exact ordered dlib predictor. Missing features block only affected stages.
- Generic persistent FP32 inference: at most three inputs/twelve outputs,
  explicit native rank, shape, offset/capacity and row/plane strides. Bind and
  extract once, retain every VkMat and borrowed buffer, unpack/copy outputs on
  GPU, and replay cached commands. Existing published MODNet metadata remains
  unchanged. Separate graph pools share aggregate charged quota/lifetime; no
  cross-graph workspace reuse while recorded commands reference it.
- FastDVDnet uses a five-frame GPU history, causal center output/noise-map and
  canonical strength. Pad odd extents by edge replication to multiples of four,
  then crop output. Reset on discontinuity/drop/disable/config/model/context
  changes and failure; no frame-time allocation or graph rebuild.
- YuNet keeps all twelve heads resident. Decode/top-K 5000/NMS, distinct eye
  confidence and Auto Frame area selection, dlib prediction, crop smoothing,
  matte bounds, and tracked-center transformation all execute on GPU. No raw
  host-frame landmark substitution, intermediate/model/metadata readbacks, or
  CPU tracking tails. Existing strict Vulkan recommendation predicates remain
  meaningful rather than being weakened to hide degraded processing.
- FLX remains the curated FP32 corrected-RGB contract and model ID, with three
  inputs per eye. Its reviewed ncnn learned prefix produces flow/lcm; an exact
  native sampler/lighting tail restores full ONNX RGB semantics, including
  clipped-neighbor border weights. Preserve left-before-right compositing,
  strength/rounding, no-face preservation, and the existing no-op look-away knob.
- Canonical order: denoise → eye → lazy shared current-frame matte/background
  → key light → Auto Frame → resize → vignette → mirror. Cache identity includes
  producer revision, selected artifacts, configuration, capture sequence, and
  context/domain generations. Eye/Auto Frame share only compatible analysis.
- Fixed-capacity frame collector: 128 command buffers/participants, exact tickets
  with batch/epoch/sequence/owner generations, typed completed/unsubmitted-abort/
  unsafe-pending callbacks. One finite multi-buffer submission where possible;
  preserve 16 synchronizations and 100 ms fence policy. Per-dispatch parameter/
  descriptor slots remain distinct, including non-push descriptors. Pending or
  failed proof retains/quarantines the complete island; no successful counters
  or writer publication before completion.
- Original 256 MiB matting-only program limit cannot be silently multiplied or
  claimed as a multi-model arena. Caller-owned aggregate bounds must account
  actual graph pools, temporal/leaf buffers, staging, descriptors and quarantined
  owners. Measure complete preparations at supported geometries and validate
  all six caller axes; document any reviewed new bound explicitly.
- Direct recorded graphs may run while GPU no-face/cadence guards preserve
  output. Strength-zero is setup-known and omits work. Record actual inference
  counters and measure enabled no-face overhead; do not claim CUDA's skipped
  inference count unless commands truly skip. GPU conditional execution is an
  optimization only if measurements justify its added runtime contract.

## Ownership and waves

All agents use isolated worktrees and focused commits; root alone integrates.

| Owner | Files / acceptance |
| --- | --- |
| Runtime/model | New generic ncnn runtime/API; context/device/FP64/shared quota; model registry/schema and tests; offline conversion/locks/oracles. Freeze public headers first; retain old matting adapter compatibility and existing ONNX identities. |
| Tracking/leaf | CUDA-free bounded forest-data helper; prepared GPU tracking/eye/crop/vignette owners and kernels/shaders/SPIR-V; exact CPU/dlib numerical oracles. No common device, registry, or production monolith edits. |
| Production integration | New resident command collector; deferred utility slots; Vulkan factory/setup/adapter/executor and denoise leaf; reset/discard/optional failure and production tests. No shared device edits. |
| Root | Contract review, CMake/source ownership, neutral evidence, daemon/installer/catalog/docs coordination, native scheduling, integration and independent review. Installer provisioning may be delegated once an implementation slot is free. |

Wave 1 commits shared headers and bounded low-level implementations. Wave 2
integrates leaves and generic inference. Wave 3 integrates the actual manager,
installer/provisioning/model surface and diagnostics. Then focused and broad
OFF/ON builds, available native validation, independent critique/remediation,
and a final complete handoff. No accepted implementation is left unmerged.

## Discovery evidence

Read-only audits: `vulkan_all_models_inference_audit`,
`vulkan_all_models_tracking_audit`, `vulkan_all_models_pipeline_audit`, at the
baseline above. Existing CUDA provides the ordered GPU dlib predictor; using
original host capture would change denoise→eye semantics. The pinned ncnn already
enables its queried core physical features, including advertised shaderFloat64;
no SDK patch/ABI bump is needed solely to enable FP64.

Offline feasibility in
`/tmp/studiocast-vulkan-inference-feasibility-20261112/` established:
FastDVDnet dynamic 8×12/48×64/72×88 max error `5.96e-7`/MAE `8.04e-8`;
all twelve YuNet heads max `3.81e-6`; complete FLX prefix/native-tail CPU oracle
matches original ONNX exactly. Converted-prefix final RGB samples reached left
max `1.555085e-4`/MAE `2.295879e-6`, right max `1.920015e-5`/MAE `1.379545e-6`.
These are feasibility evidence, not final model or GPU qualification. Direct
full FLX conversion exits successfully while retaining unsupported operators;
it must never be promoted as a valid GPU graph.

The actual catalog uses the Better Quality FP32 FLX pack (left `1a512997…`,
right `736b7c6…`); Good Quality is a different FP16 ID/contract. Existing native
no-face matrices do not demonstrate gaze correction. A licensed face-positive
fixture, dlib zero-coordinate oracle, original-ONNX numerical matrices, and full
single/composed camera cases are required before claiming completion.

## Verification and remaining work

Run existing tests before edits, focused tests after each accepted merge, then
production ncnn ON, Vulkan/runtime OFF, and all-optional/OFF broad suites under
temporary HOME/XDG. Native scheduling is exclusive and coordinated by root.
Exercise each effect, full combinations/all backgrounds, real face/no face,
odd geometry, zero/nonzero strengths, sharing/provenance, reset/discontinuity,
writer and optional failures, low aggregate budgets, missing/mutated artifacts,
FP64 absence, CPU-layer rejection, unsafe submission/quarantine, and descriptor
reuse. Assert one image upload/final readback, zero intermediate/model/metadata
host transfers or CPU tails, no Execute allocations, and bounded resources and
submissions. Preserve the existing P1000 evidence without generalizing it to
unavailable AMD/Intel, real-camera/V4L2, quality, or optimized latency coverage.

Implementation, model validation/publication proposal, installer provisioning,
full native matrix, and fresh independent critique remain in progress. Exact
commits, merged branches, tests/results, reviewed budget, and remaining external
prerequisites will be appended as work is integrated.
