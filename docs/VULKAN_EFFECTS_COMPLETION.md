# Complete production Vulkan effects

- Status: Local implementation and validation complete; publication approval and hardware qualification remain
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

All accepted local implementation is integrated. Publication/catalog activation
awaits explicit authority, and the available-hardware failures below remain
unqualified. The following ledger records chronological checkpoints; the final
local gate at the end supersedes earlier in-progress observations.

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

## Integrated branches and follow-up verification

| Topic | Accepted source / integrated commit |
| --- | --- |
| Generic inference/conversion | `codex/vulkan-all-runtime-20261112`, tip `05df764`, merged `e2e8c1a` |
| Resident tracking/eye kernels | `codex/vulkan-all-tracking-20261112`, tip `2a6ca5b`, merged `0842bae` |
| Readiness wrappers | `codex/vulkan-model-readiness-20261112`, `686a587` → `f0ee829` |
| Model/shared-inference docs | `codex/vulkan-model-docs-20261112`, `df66ac3` → `f7a2835` |
| Production integration/isolation | `codex/vulkan-all-pipeline-20261112`, `f06d6ff` → `eebff48`; rank-one/warmup changes `edc8c76` → `cf14d37` |
| SDK installer | `codex/vulkan-installer-runtime-20261112`, `54decff` → `353ae41`; smoke `c0af1de` → `efdd822` |
| Legacy status/recommendation | `codex/vulkan-legacy-projection-20261112`, `9263541` → `b769b55` |
| Opt-in full-model CI | `codex/vulkan-models-hardware-ci-20261112`, `9e31f3e` → `e869a81` |

The initial Auto Frame proof correction misread the base runtime code: it already
emitted tracking 54, so the additive record must remain face 51. `cf8ec26` corrects
the introduced duplicate 54 and restores exact 51 + 54. The complete seven-stage
factory test is integrated at `885cf97`. `1a0890a` then validates every raw factory
capability with the shared full-proof predicate and static registry before
publishing readiness; missing 51/54/53/55 and duplicate-proof negatives pass. This
prevents invalid raw readiness from reaching a later lease rejection.

Cold diagnostics initially under-reserved the forest setup upload at 1 MiB.
`892158d` reserves 64 MiB for the exact roughly 63.7 MiB flattened forest; the whole
small startup profile is explicitly 833 MiB with 2,048 resources, not a per-model
multiplier. Actual optimized cold diagnostics then passed all nine effects on the
same prepared owner in 23.51 s, within the existing 35 s installed refresh timeout.
The native log is
`/tmp/studiocast-vulkan-all-pipeline-release-cold-diagnostics-20261112.log`.

The real full seven-stage 64×64 preparation passed at 375,072,288 retained bytes /
802 native allocations, with 815 setup high-water allocations and 8.59 s Release
setup. All six caller axes remain unchanged; the general manager's existing
maximum caller reservation is 8 GiB persistent + 4 GiB scratch + 1 GiB upload + 1 GiB readback,
1,024 resources and one in-flight frame. The actual context clamps the aggregate
to the selected physical memory limit and charges every pool/buffer. This is an
explicit aggregate upper limit, not a claim that each effect consumes its own
256 MiB or that FHD preparation/execution is qualified.

The full batch first hit the 100 ms fence and exited 139 during unsafe teardown.
A debugger rerun completed four batches in approximately 90.9/88.9/48.4/48.0 ms
with one upload/final readback/synchronization, no CPU tails and zero C++ Execute
allocations. The timeout/crash is still under diagnosis; these four observations
do not establish latency percentiles or waive the 100 ms limit. Independent
lifetime inspection found complete island retention but a genuine missing
production bridge to recheck a poisoned context's exact pending fence during
shutdown. A bounded late-retirement API and scoped crash reproduction are in
progress; successful late retirement must never promote a failed frame.

The native matrix also exposed a valid neutral-runtime scope extension:
a successful publication followed by a failed writer retained stale active
attribution in PreparedProgramSession and VideoExecutionCoordinator. A separate
isolated remediation owns only those two implementations and their existing
focused tests. It must preserve selected/readiness state and historical counters,
clear current activity only after a validated failed publication, reject forged
completions and recover on the next successful writer. The native assertion is
retained unchanged.

Before these latest remediation changes, broad loader-denied suites passed:
production ncnn 100 tests (88 executed / 12 native opt-ins skipped), all-optional 94
(88 / 6 skipped), both 49.47 s. Disabled 85 tests had only the disk-dependent fake-build
backend test fail; its two fixtures were corrected to use declared fixture free
space and the affected backend test passed afterward. No assertion or disk gate
was weakened. Fresh final-head suites remain required after remediation.

Independent model/publication audit found no Critical/High asset/license blocker:
all 14 file hashes, four complete finite FP32 graphs, original ONNX and prefix
checks,45 denoise / 60 face-head / 15 per-eye numerical cases and 30 exact-tail cases,
full upstream license receipts, local/offline transactions and unmarked-user-pack
preservation passed. Promotion restores all ten original catalog artifact objects
(including public ONNX source candidates) and adds only the eight new artifacts. The
exact fourteen-file publication approval remains pending; no new artifact was published
or active catalog promoted.

## Integrated failure remediation (2026-11-13)

The earlier timeout/crash investigation found a reproducible driver-exit fault:
the NVIDIA worker crashed in `libnvidia-glcore.so.535.309.01` during process exit
while the failed submission was still pending. Complete owner retention prevented
premature resource release but did not supply a shutdown retirement proof.
The integrated bounded late-fence API (`f1447c8`, `3da0121`, `a709548`,
`82230ef`) now permits one exact pending-fence check per shutdown request. A
successful check retires failed callback tickets before destruction, keeps the
context poisoned and never converts the failed frame into successful output.
Lost devices and unproven completion remain retained and fail closed. Neither
the 100 ms fence policy nor the normal 16-synchronization limit changed.

Injected native denoise and full-program timeout tests now exit successfully
after proven retirement, reject further execution and publish no failed output.
The minimal shutdown proof took 52.70 ms. The full-program shutdown observation
was 154.56 ms including destruction; that wall time is not a longer fence wait.
One earlier full-program probe remained pending at its 100 ms limit and is
retained in the evidence rather than counted as success. Logs:
`/tmp/studiocast-vulkan-all-pipeline-release-unsafe-timeout-minimal-retired-20261112.log`
and `/tmp/studiocast-vulkan-all-pipeline-release-unsafe-timeout-all-retired-20261112.log`.

Vulkan timeout/fatal quarantine is now distinct from actual device loss
(`2f82443`, `5e13901`). The neutral session, facade and coordinator repair
(`adf2d5f`, source `b6cd07c` on
`codex/quarantine-reason-retention-20261112`) retains the first actual cause
through retries, binding/resolution, reset, discard, invalidation and rejected
reconfiguration. Status-only quarantine defaults to synchronization failure;
the reset trigger is not device-loss evidence. Actual device-loss counters
advance only for an actual device-loss cause. Failed validated execution or
publication clears current activity while preserving selection and historical
counters. Forged completion tokens, including a changed complete bit, cannot
clear or publish a real pending frame. Regression-first checks failed before
the fixes; all nine focused neutral suites passed in the isolated disabled
build with temporary HOME/XDG roots. Integrated Vulkan adapter/factory/executor
checks passed three of three after the Vulkan cause fixes.

The full 64×64 blur composition passed after writer-attribution repair:
sequences 1/2/7/9 (failed writer)/10 (recovery), five completed GPU batches and
four committed outputs, one upload/final readback/synchronization per batch,
zero execution-time C++ allocations, and reset/gap assertions. Frame wall times
were 90.16/88.38/50.54/47.99/47.92 ms, not latency percentiles. Log:
`/tmp/studiocast-vulkan-all-pipeline-release-execute-all64-writer-fixed-20261112.log`.
Other backgrounds, face-positive composition and final-head broad suites
remain in progress.

The larger-geometry investigation has isolated a 640×480 denoise-only warmup
timeout on this P1000. Partial actual allocation at failure was 864,332,464
bytes retained / 883,993,264 peak and 258 / 407 native allocations, below the
unchanged 1,024-resource caller cap. These are partial denoise observations,
not a successful whole-program budget or FHD qualification. The original full
640×480 preparation also failed closed. No packing/precision setting or wait
limit has been changed to hide this hardware limit. The exact minimized
capture is `/tmp/studiocast-vulkan-all-pipeline-profile-denoise640-gdb-20261112.log`.

Cached diagnostics now enumerate every installed Open Video task and expose
separate per-task daemon-default and actually prepared model IDs (`40a4564`).
New-task cold probes use registry defaults; ordinary status remains cache-only.
Existing matting native-compatible default behavior is preserved. Production
new-task default selection and the independent custom-pack native regression
are being integrated separately. The scope documentation now covers all effects
together (`efb399c`), while exact per-effect qualification remains mandatory.

## Independent critique and final integration follow-up

The packaging/model/installer/CI critic reviewed `82230ef` in
`/tmp/studiocast-vulkan-packaging-critique-20261113-82230ef`, found no new
Blocker/Critical/High, and independently passed 35 checks: SDK security 18,
typed evidence 2, workflow propagation 7 and converter contracts 8. The known
default-selection repair was excluded from that bounded review. That repair
is now integrated at `a2fba7d` (source `6dbd021`): the actual native BYO-only
default regression passes empty/default selection, explicit missing selection
and an incompatible daemon default without silent replacement.

The runtime critic at `9b75d99`, refreshed for `a2fba7d`, found one valid Medium:
YuNet preprocessing extrapolated negative border coordinates and used rounded
per-axis letterbox ratios rather than the canonical uniform inverse scale.
`b7d4772` integrates source `1d92a07` from
`codex/vulkan-yunet-preprocess-parity-20261113`. The new direct tensor oracle
failed the old SPIR-V with maximum error 101.79, then passed eight geometries
with maximum error 1.52588e-5 against the independent canonical CPU oracle.
Generic Auto Frame crop sampling remains unchanged because its established
contract intentionally differs. Only tracking SPIR-V was regenerated, and
pinned-compiler byte-for-byte freshness passed.

A new test fixture initially released preprocessing buffers while its recorded
commands retained borrowed bindings. One-variable controls isolated the fixture
lifetime: original test order with the new shader passed; releasing the new
buffers before subsequent tests failed; retaining them through kernel shutdown
passed. Production tracking already shuts kernels down before freeing its
buffers and required no runtime change. The corrected full native oracle passed
all eight preprocess cases, 20 exact dlib cases, detector/NMS, crop/matte and
compositing cases, and 30 exact FLX tails; maximum standalone execution was
17.5097 ms. Evidence and preserved negative controls are in
`/tmp/studiocast-yunet-preprocess-evidence-20261113`.

The wider manager matrix then exposed key-light-only execution still using the
older per-kernel submission path: its unchanged one-synchronization assertion
failed at seven synchronizations. `5e6b699` integrates source `26ed2be` and
prepares the same collector for matting-only and adequately budgeted utility
camera programs. The original key-light case now passes five batches/four
commits at 22.75–22.83 ms, one inference per frame, one upload/final readback/
synchronization and zero execution-time allocations. Existing tiny utility
reservations retain their bounded legacy route. No fence or upper synchronization
limit changed. Log:
`/tmp/studiocast-vulkan-all-pipeline-keylight-batch-regression-20261113.log`.

The reset review also found a bounded neutral scope extension: non-ready reset
could leave a generic coordinator executor eligible, while the outer manager
invented device loss from status-only quarantine and could retry an invalidated
compatibility pointer. `6315d2c` and `31a86af` integrate focused sources
`46b07e7` and `4d2b66d` from `codex/coordinator-reset-recovery-20261113`.
Regression-first cases now require successful preparation before execution,
publication or reset retries; actual current snapshot causes remain distinct
from reset triggers. All nine neutral suites passed after root integration
(0 skips/failures, temporary HOME/XDG, ICD discovery denied). Evidence:
`/tmp/studiocast-neutral-reset-integrated-final-tests-20261113.log`.

A fresh neutral critic at `31a86af` confirmed those paths but found two further
Medium wrapper gaps using real wrappers with fake runners/adapters: Open CUDA
snapshot/reset observers converted cause 21 to loss 13, and direct compatibility
failed publication retained prior current activity. These are assigned as a
bounded follow-up to `codex/open-video-projection-repair-20261113`, including
the same cause projection in the prepared CUDA owner and their existing tests.
No CUDA algorithms, GPU execution or public contract changes are authorized by
this extension. A separate SDK-off optional-model isolation check and final
combined/native/cold diagnostics matrix are still in progress. Seven integrated
Vulkan hermetic suites passed at `5e6b699`; fresh final runtime critique and
final-head broad suites remain required after the pending fixes.

## Compatibility transport scope extension

Later fresh review of the shared publication boundary found two concrete High
issues in the retained host compatibility executor. Its successful output exposed
only a read pointer while the actual camera consumer required a writable output
view, so processed effects were discarded. It also allocated/copied using writer
geometry while validating only capture-sized input and preparing models for
capture geometry. A guarded 24-byte input with a 48-byte requested output
reproduced an overread in a hermetic child process. These are valid existing-route
preservation and memory-safety requirements, not hardware qualification waivers.

The bounded repair is assigned to the existing compatibility owner. Keep effects
at validated capture geometry, then reuse the existing prepared CPU bilinear
resize for the final writer geometry/stride. Return stable owned read/write
aliases until exact publication; retain all prior lifecycle, pending-token and
historical-counter guarantees. Model/runtime algorithms, camera processing and
Vulkan residency are unchanged by this seam. Frame buffers and resize lookup
scratch must fit explicit caller reservations and be prepared outside execution.
Required verification includes actual camera acceptance, guarded up/down scaling,
odd and padded geometry, malformed descriptors, allocation-denied execution and
the existing publication/reset suites. This High remediation is integrated at
`63953fe`; its final tests and fresh independent critique are recorded below.

The source baseline for the completed available Vulkan matrix was `7c20594`.
It passed 26 case assertions and 106 successful frames, all with one upload,
one final readback and one synchronization, zero intermediate transfers/CPU tails
and zero execution-time C++ allocations. Two original cases failed and remain
unqualified: combined odd replacement exceeded its first-frame fence at 104.93 ms;
full-face 320×240 setup omitted denoise after its warmup timed out. A single scoped
debugger control identified `video_noise_removal` and the exact `VK_TIMEOUT` without
replacing that failed qualification. Two other full-face modes were not run and
remain unqualified. Normal cold diagnostics passed in 10.71 s, and missing-landmarks
diagnostics passed in 4.36 s with Eye Contact isolated and the other eight effects
ready. The SDK-off native survivor test and explicit wrong-build negative passed.

A subsequent Release rebuild had exact tracked-tree equality with root `e10e5b3`
(`0bc7e4d38a11befbd567fb86d5c4c265743111b7`). Its requested final-head Eye Contact
320×240 plus zero-strength control passed ten frames and changed 648 final RGB
bytes. The combined blur control instead hit its first-frame fence at 103.06 ms
and exited 1 normally, without a retry. This confirms that combined cold-start
deadline qualification is incomplete on this P1000 even at small geometry; passing
component/combination cases do not erase the failed observations. The existing
100 ms fence and 16-synchronization cap are unchanged. Exact commands, executable
SHA-256, metrics and all pass/fail/skip logs are recorded in
`/tmp/studiocast-vulkan-all-pipeline-final-handoff-20261113.txt`.

## Final local gate (2026-11-13)

The complete prepared Vulkan program now includes denoise, Eye Contact and its
resident face/landmark dependencies, all background modes/key light, Auto Frame,
resize, tracked/fixed vignette and mirror. It retains one exact owner/device/
queue, prepared graph bindings, aggregate reservation, GPU intermediates and
bounded completion. Optional prerequisites remain isolated; required failures
remain fail-closed. Normal installer SDK provisioning and strict opt-in native
CI are integrated. No additional requested Vulkan effect is deferred.

The compatibility High repair (`005f6c9` → `63953fe`) changes only the existing
executor and tests. Effects use capture geometry; the existing prepared CPU
primitive handles final writer geometry/stride. Physical backing covers complete
rows while the caller's logical span remains exact. All owned frame/lookup bytes
fit explicit reservations before allocation/model setup. The output has stable
readable/writable aliases, and frame/completion/committed domains identify the
actual owner. Guarded tight/minimal padded captures, up/down/odd/padded resize,
optional partial failure, allocation-denied replay, malformed descriptors,
exact/short reservations, foreign owners/generations and publication recovery pass.

Fresh critic `compatibility_geometry_final_critique`, detached at `63953fe`,
reviewed the actual camera consumer and independently ran the integrated
compatibility binary with temporary HOME/XDG and Vulkan ICD discovery denied.
It found no remaining Blocker/Critical/High or additional unmet requirement in
that boundary. The earlier two High findings are resolved. Separate final Vulkan
critique through `adb69a4` found no additional source Blocker/High and explicitly
retained the failed hardware profiles. No accepted source change remains on an
unmerged topic branch.

All three complete Debug builds and broad hermetic suites pass at source head
`63953fecc610b99137005c319a747d0cb8978083`:

| Configuration | CTest total | Executed passes | Native opt-in skips | Failures | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vulkan + production ncnn + dlib | 100 | 88 | 12 | 0 | 64.69 s |
| CUDA/Vulkan/audio/dlib, ncnn off | 94 | 88 | 6 | 0 | 64.41 s |
| Optional backends off | 85 | 80 | 5 | 0 | 64.34 s |

Each suite used fresh HOME/XDG roots, a real CMake executable on PATH and
nonexistent `VK_DRIVER_FILES`/`VK_ICD_FILENAMES`. No native result is inferred
from those loader-denied suites. The exact command was
`ctest --test-dir <build> --output-on-failure -j2`; logs begin with the tested
source SHA and are
`/tmp/studiocast-vulkan-safe-transport-final-{production,alloptional,disabled}-suite-20261113.log`.
The integrated 13-suite publication/reset/owner/compatibility subset also passed
with zero skips/failures in 0.34 s; log
`/tmp/studiocast-safe-transport-integrated-final-tests-20261113.log`.
Documentation validation passes all 43 Markdown files; source/bytecode provenance
and the isolated pinned-compiler shader reproduction also pass.

### Follow-up topic integration ledger

These rows supplement the initial topic table above. All listed accepted source
commits are on `v0.2.9`; read-only critics made no source commits.

| Topic branch | Focused source → integrated commit |
| --- | --- |
| `codex/vulkan-late-retirement-20261112` | `9fad0fd` → `f1447c8`; `617655b` → `3da0121` |
| `codex/quarantine-reason-retention-20261112` | `b6cd07c` → `adf2d5f` |
| `codex/coordinator-reset-recovery-20261113` | `46b07e7` → `6315d2c`; `4d2b66d` → `31a86af` |
| `codex/vulkan-yunet-preprocess-parity-20261113` | `1d92a07` → `b7d4772` |
| `codex/vulkan-all-pipeline-20261112` | `8c0bc14` → `2f82443`; `4dfc5e7` → `5e13901`; `6dbd021` → `a2fba7d`; `26ed2be` → `5e6b699`; `bb61e5b` → `3388a9c`; `7c20594` → `adb69a4` |
| `codex/open-video-projection-repair-20261113` | `bbec0a8` → `ef3ff23`; `883cf90` → `1f7f641`; test-only `2c685a8` → `a80e7f6` |
| `codex/compatibility-current-activity-20261113` | `423ae3b` → `dc7cf2e`; `5d56762` → `e10e5b3` |
| `codex/compatibility-output-transport-20261113` | `005f6c9` → `63953fe` |

Fresh review iterations identified and closed the YuNet preprocessing Medium,
neutral/CUDA cause projection and compatibility attribution/pending Mediums,
and the two compatibility transport Highs. Each implementation handoff records
its RED/GREEN evidence, exact tests, file ownership and clean isolated worktree.
The final compatibility review is in
`/tmp/studiocast-compat-final-critique-20261113-63953fe`; the read-only contract
audit at `20cf31f` was guidance only and was not used as final acceptance.

### Remaining authority and qualification

The only prepared, unmerged proposal is the explicitly unapproved catalog branch
`codex/vulkan-catalog-unapproved-proposal-20261113`, commit `01eaaa4`. It contains
metadata/count/packaging changes only, no model binaries. The corrected proposal
maps all eight new artifacts to the exact immutable publication destinations;
the historical `62da60bd` derivative is not activation-ready. Its final catalog
SHA is `097cb1f2551263c52c52357839bd4350c090c93c5619193688432135a9c42d9e`.
It passes 21 model, 56 backend, 13 packaging and 14 offscreen GUI scenarios. The
active catalog remains unchanged at seven packs/ten artifacts until approval,
public immutable retrieval, factual receipt/version/source pin and fresh catalog
review. See [the publication proposal](VULKAN_MODEL_PUBLICATION.md).

Hardware qualification is incomplete: combined cold batches can exceed 100 ms
on this P1000; full-face 320×240 denoise warmup and 640×480 denoise preparation
failed their limits. AMD/Intel, full-resolution combined throughput, actual
camera/V4L2 transport, portrait quality, forced descriptor fallback, full native
sanitizers and validation-layer cleanliness are not qualified. Existing look-away
behavior remains its documented no-op contract. These limits are explicit
deployment-feedback scope under the user's hardware decision, not passing tests
or waived fences. No application release, Git push, installed-pack mutation or
new external publication occurred during this expansion.
