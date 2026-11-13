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
- FastDVDnet keeps three physical GPU histories and the causal logical window
  `[t-2,t-1,t,t,t]`, with center output/noise-map and
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
  unsafe-pending/late-retirement callbacks. One finite multi-buffer submission where possible;
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
  output. Strength-zero is setup-known and omits live inference; setup still validates
  and warms its selected graph. Record actual inference
  counters and measure enabled no-face overhead; do not claim CUDA's skipped
  inference count unless commands truly skip. GPU conditional execution is an
  optimization only if measurements justify its added runtime contract.
- Model and tracking graphs record once. Utilities reuse bounded prepared slots;
  a slot may be reset and recorded after its prior completion, with no frame
  allocations or descriptor mutation while queued.

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

## Integrated evidence ledger (in progress)

The root integrated the shared contracts and deferred command collector first,
then the runtime branch at `e2e8c1a`, installer provisioning at `353ae41`, and
tracking at `0842bae`. Runtime tip `05df764` includes fixed native tensor ranks,
shared allocation quotas and a monotonic native-allocation counter. Tracking tip
`2a6ca5b` includes BYO checksum compatibility: an undeclared checksum still seals
against the exact digest computed during selection. No installed pack was changed.

The isolated tracking native run used the P1000 with the unchanged 100 ms fence:
20 dlib cases matched every coordinate; 6,400 detector candidates matched the
stable top-K/NMS reference (3,311 kept); crop, smoothing, discontinuity, 64-bit
sequence carry, tracked vignette, matte cadence and seven composite cases matched
exact output bytes. Thirty native FLX sampler/tail cases matched original full
ONNX RGB with maximum error zero. Maximum standalone execution was 35.002 ms.
These are leaf results, not combined production-manager or face-positive proof.
The root's new shader-source hash checks and byte-for-byte recompilation of all
three tracking shaders passed using isolated glslang 16.6.0. Older utility/resize
bytecode uses its original compiler; no unrelated bytecode was regenerated.

The generic runtime's earlier native oracle covered 45 FastDVDnet cases, 60
YuNet head comparisons, and 15 cases per eye with unchanged maximum 2e-4 / mean
2e-5 thresholds. Its post-counter integrated rerun remains pending. The measured
214,668,160-byte / 336-resource peak was sequential inference qualification,
not the combined pipeline or an FHD budget.

Eight converter checks and eighteen hermetic SDK tests passed after integration.
Actual pinned source archives passed bounded extraction/patch preparation; a full
isolated SDK build is running without host installation. Root production-on
compilation, full manager matrix, aggregate budget review, model-publication
approval, cached readiness integration and fresh critique remain open.

The exact bounded capability-evidence capacity grows from 24 to 25 records:
the complete shared Vulkan proof union now includes explicit successful model
warmup (17) alongside the six new model/FP64 codes. Per-semantic and aggregate
proofs retain the warmup record; neither silently drops it. The installer parser
uses the same capacity and still rejects overflows and inconsistent typed pairs.

The actual archive-mode SDK build completed successfully in an isolated temporary
prefix: 62 verified files / 36,913,614 bytes, exact upstream/glslang/replay/patch
receipt, all required notices retained, no host installation. Integrated native
generic inference and tracking reruns passed; tracking maximum standalone fence
execution was 34.9349 ms. Ten integrated focused hermetic/loader-denied tests and
five disabled-build focused tests passed with no failures.

The production manager matrix exposed a High Eye Contact setup blocker: physical
angle tensor storage held two scalar channels while its native rank-one graph
view required width two/channel one. The same issue was independently confirmed
by early security audit at `30f17c5`. Remediation `cf14d37` preserves the same two
float bytes and changes only the rank-aware binding; native Eye64 preparation
passed. The commit also fixes Auto Frame's distinct tracking proof (54 instead
of duplicate face proof 51) and emits explicit completed warmup proof 17. The
early audit found no other verified Critical/High in the inspected runtime,
quota, command-lifetime, dlib parser or SDK trust paths. It does not replace the
required final fresh critique after combined validation.

Native prepare-only measured denoise64 at 117,346,480 live bytes/262 allocations,
214,630,832 setup high-water bytes/407 allocations; Eye64 at 147,369,952 live
bytes/367 allocations, 147,488,744 high-water bytes/390 allocations. All used the
unchanged 100 ms native fence and 1,024 caller resource cap. Debug setup wall time
includes selection, hashing, shader/graph preparation and predictor parsing; it
is not an inference latency claim. Combined/FHD profiles, production execution,
face-positive proof, Release diagnostic-refresh timing and final budgets remain
under investigation before qualification.
