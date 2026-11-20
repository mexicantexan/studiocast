# Complete production Vulkan effects

- Status: Local implementation, approved model publication and catalog activation complete; hardware qualification limits remain
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

All accepted local implementation is integrated. The user subsequently approved
model publication/catalog activation, now integrated as recorded at the end.
Available-hardware failures remain unqualified. The following ledger records
chronological checkpoints; the final gates supersede earlier in-progress observations.

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

The earlier unapproved proposal `01eaaa4` was held separately until the user's
explicit approval. It is now integrated, with factual activation/version/source
pinning, as recorded below. The historical `62da60bd` derivative is not
activation-ready and remains unchanged. See
[the publication record](VULKAN_MODEL_PUBLICATION.md).

Hardware qualification is incomplete: combined cold batches can exceed 100 ms
on this P1000; full-face 320×240 denoise warmup and 640×480 denoise preparation
failed their limits. AMD/Intel, full-resolution combined throughput, actual
camera/V4L2 transport, portrait quality, forced descriptor fallback, full native
sanitizers and validation-layer cleanliness are not qualified. Existing look-away
behavior remains its documented no-op contract. These limits are explicit
deployment-feedback scope under the user's hardware decision, not passing tests
or waived fences. No application release, Git push or installed-pack mutation
occurred. The model publication below was separately approved and verified.

## Approved publication and catalog activation (2026-11-14 UTC)

The user approved the fourteen-file publication and local catalog activation
after the local implementation handoff. Root reverified the unchanged inventory
SHA `8411bb3f45ae31111f515724859128c13d7b59abf4b3058cef139f01fe8a0406`
and all source bytes, inspected the actual remote tree and refused overwrites.
One additive Hugging Face commit
[`4713473184bf7ece0be6c097ff75f6c6c528c8ec`](https://huggingface.co/10dallasj/studiocast/commit/4713473184bf7ece0be6c097ff75f6c6c528c8ec)
with parent `f6be359c49ba07163287fe3732dc4a3024b91f60` added exactly fourteen
files / 22,066,504 bytes and preserved all 62 existing paths, blobs and sizes.
Every new file and all ten original artifacts were retrieved without
authentication at that immutable resulting commit and passed exact size/SHA-256
verification. The factual receipt is
`/tmp/studiocast-vulkan-model-publication-20261113/publication-receipt.json`,
SHA `89b171deb9a5218f0b3da2947f09ed972b4c14fd0bdec532150aa2b646fd3e95`.
Publication time was `2026-11-14T01:41:23.325423+00:00`, November 13 locally.

Activation topic `codex/vulkan-catalog-activation-20261113` contains proposal
cherry-pick `5af3452` → root `15a873b`, then activation `9b6dec2` → root
`ffc50d6`. No accepted metadata work remains stranded on the proposal branch.
Catalog version `2026-11-14`, SHA
`2e5a02a067765fbc54aa1d798d44b25f9d940629e83509b153274ef7d7cdca1b`,
pins its source to that verified immutable commit. Seven packs now contain
eighteen artifacts / 188,659,386 bytes. Ten original artifact objects, IDs,
defaults, source candidates and four unaffected pack objects are preserved.
Six packaged metadata/license files match the published bytes exactly. The
three affected packs retain upstream provenance and append actual publication
identity. Artifact count/bytes/hash fixtures, GUI review and package verification
remain strict and are updated together.

The activation agent's focused checks passed: 21 model transactions, 56 backend,
13 packaging and 14 offscreen GUI scenarios, plus receipt/preservation/hash/docs
assertions and clean diff. Root independently rechecked all new source bindings,
old artifact/default/pack preservation and the integrated catalog hash. Local
handoff: `/tmp/studiocast-vulkan-catalog-activation-handoff-20261113.md`.
The first integrated broad checks found one genuine exporter regression: the
ONNX-only FastDVDnet exporter inherited the newly curated ncnn artifacts and
network metadata without generating those files. Its real local-source install
failed on the missing ncnn parameter file. The bounded repair creates an exact
ONNX-only candidate manifest and artifact list, retaining pinned upstream/tool
verification, graph validation, CPU equivalence, I/O, provenance and license
composition. The active catalog and published bytes are unchanged. Topic
`codex/fastdvdnet-export-catalog-isolation-20261113`, source `185232b` → integrated
`e3c325b`, changes only the exporter and its existing regression test. RED/GREEN
evidence and focused checks (7 export, 21 model, 7 MODNet conversion and 8 video
conversion tests) are recorded in
`/tmp/studiocast-fastdvdnet-export-catalog-isolation-handoff-20261113.md`.
Root independently passed all seven export checks, including three small real
ONNX graph cases using the existing ONNX installation; no dependency download
or new full-weight export was required.

All three activation builds passed. After integrating the exporter repair,
the final broad checks passed at source HEAD
`e3c325b5aec6576af7a4871abba90e44f905fc82`:

| Configuration | CTest total | Executed passes | Native opt-in skips | Failures | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vulkan + production ncnn + dlib | 100 | 88 | 12 | 0 | 103.02 s |
| CUDA/Vulkan/audio/dlib, ncnn off | 94 | 88 | 6 | 0 | 102.79 s |
| Optional backends off | 85 | 80 | 5 | 0 | 102.23 s |

Each suite used fresh HOME/XDG roots, a real CMake executable on PATH and
nonexistent `VK_DRIVER_FILES`/`VK_ICD_FILENAMES`, with
`ctest --test-dir <build> --output-on-failure -j2`. Logs begin with the exact
source SHA and are
`/tmp/studiocast-catalog-export-final-{production,alloptional,disabled}-suite-20261113.log`.
These loader-denied suites do not imply additional native hardware coverage.

Fresh independent publication critique at `351da86` verified the full inventory,
receipt, remote preservation, packaged metadata, immutable source bindings and
catalog/installer contracts; its own 21 model and 13 packaging checks passed.
It correctly withheld overall completion until the exporter regression was
repaired. The separate fresh-context post-repair critic at `e3c325b` found no
blocker/high finding or valid unmet requirement. It independently passed all 7
export, 21 model, 7 MODNet conversion and 8 video conversion checks under
temporary HOME/XDG roots with the pre-existing ONNX installation, and confirmed
the active catalog and packaged resources are unchanged by the repair. Neither
read-only critic made a source commit. Documentation validation and integrated
diff checks pass. All accepted work is integrated locally on `v0.2.9`.

Model publication does not change any hardware qualification claim or authorize
an application release, Git push or installed-pack mutation. Those actions and
the hardware limits above are the remaining delivery/qualification boundaries,
not unfinished local implementation or unpublished approved model payloads.

## Hosted PR delivery follow-up (2026-11-17)

The user's continuation resumed the earlier explicit authorization to push
`origin/v0.2.9`. Root fetched the remote, confirmed the clean local branch and
preserved fast-forward ancestry, then pushed the completed work at `bb00bc3`.
The existing PR is [#22](https://github.com/mexicantexan/studiocast/pull/22).
Its description now reflects the actual integrated production effects and
approved model publication rather than the superseded missing-adapter blocker.

Hosted [run 44105178922](https://github.com/mexicantexan/studiocast/actions/runs/44105178922)
passed package smoke but exposed one compiler portability defect in all four
build jobs: the Vulkan factory test used `std::find_if` without directly
including `<algorithm>`. GCC 12/13 rejected the translation unit. This is a
bounded delivery-blocking test repair; no runtime behavior or model contract
changes are included. Topic `codex/portable-vulkan-test-include-20261117`, source
`8ed406a` → integrated `5a674a4`, owns only
`tests/gpu/vulkan/video_execution_factory_tests.cpp` and adds that one include.

The isolated GCC 12 CI-style Release compilation reproduced the exact error
before the fix and passed afterward. Its affected target build and CTest passed
(1/1, 0.03 s). Evidence is
`/tmp/studiocast-portable-vulkan-test-evidence-20261117/{red,green,configure,build,ctest}.log`.
Root rebuilt the affected target and reran its CTest in the existing production
ncnn/dlib, enabled-without-ncnn and disabled configurations: all three passed,
with zero skips or failures (0.15/0.12/0.04 s). These tests used isolated HOME/XDG
roots and nonexistent Vulkan ICD paths. Logs begin with the tested source SHA:
`/tmp/studiocast-pr22-include-{production,enabled,disabled}-integrated-20261117.log`.

Fresh independent critic `portable_include_final_critique` reviewed integrated
`5a674a4` and this documentation follow-up, finding no blocker/high or valid
unmet requirement. Its own GCC 12.3 Release syntax checks reproduced the
baseline failure and passed with Vulkan both OFF and ON after the fix. The
existing GCC 12 factory test binary also passed independently under fresh
HOME/XDG roots with a nonexistent ICD. The critic confirmed that every other
source file is unchanged and made no source commit.

The runtime, active catalog, packaged model metadata, published artifact bytes
and hardware qualification limits are unchanged. Fresh hosted CI for the
repaired pushed head remains the external build/test gate before merge. The
existing master-push workflow owns the normal patch-version bump; this delivery
does not merge the PR, release an application or deploy to a user system.

### Additional CUDA test portability checks

The next hosted [run 44107603362](https://github.com/mexicantexan/studiocast/actions/runs/44107603362)
at `12730ea` passed package smoke and compiled past the Vulkan test. All four
build jobs then exposed the same missing direct `<algorithm>` dependency in
`tests/gpu/cuda/prepared_vignette_tests.cpp` and
`tests/gpu/cuda/prepared_preprocess_kernel_bundle_tests.cpp`: GCC 12/13 rejected
`std::clamp`; subsequent preprocess diagnostics were cascades from that error.
The bounded routine repair is owned by topic
`codex/portable-cuda-test-includes-20261117`, based on `12730ea`, and remains
limited to direct standard-library includes in tests. Both reported translation
units reproduced the exact failure and passed the same GCC 12 commands after
adding `<algorithm>`.

The complete GCC 12 build then found the same missing include for `std::none_of`
in `tests/gpu/open_video/open_cuda_production_hal_owner_tests.cpp` and
`std::clamp` in `tests/gpu/open_video/open_cuda_production_smoke_tests.cpp`.
Continuing with Ninja's keep-going option checked every remaining default
target before another push. The topic's final source scope is four test files,
each adding the direct `<algorithm>` include; no assertions or executable logic
are changed.

Topic source `098cc92` is integrated as `a196dbf`. The complete GCC 12 Release
build passed with Open CUDA ON and Vulkan, Open Audio, dlib, CUDA kernels and
benchmarks OFF. The local build detected the existing ONNX Runtime; this is
recorded rather than represented as an absent-SDK build. Broad hermetic CTest
passed: 85 total, 80 executed, five explicit native opt-in skips, zero failures,
57.26 s. Evidence is under
`/tmp/studiocast-portable-cuda-test-evidence-20261117`: per-file RED/GREEN logs,
the retained failed keep-going compilation logs, successful
`full-build-final.log`, and `ctest.log`.

After integration, root rebuilt all four affected targets in the existing
production ncnn/dlib, enabled-without-ncnn and disabled configurations. Each
CTest selection passed three tests and explicitly skipped the native smoke
(four total, zero failures; 0.16/0.18/0.06 s). Fresh HOME/XDG roots, nonexistent
ICD paths and no native opt-ins kept validation hermetic. Logs begin with the
tested source SHA:
`/tmp/studiocast-pr22-cuda-headers-{production,enabled,disabled}-integrated-20261117.log`.
Fresh independent critic `cuda_header_batch_final_critique` reviewed integrated
`a196dbf` and this documentation follow-up, finding no blocker/high or valid
unmet requirement. It independently reproduced all four baseline failures and
passed all eight fixed syntax checks (four translation units, each with
`STUDIOCAST_HAVE_ONNXRUNTIME=0` and `1`). Its isolated runs passed the three
focused binaries and explicitly skipped the native production smoke. Its scan
of all C++ test sources/headers found twenty files using the checked algorithm
functions, all with direct `<algorithm>` includes. Evidence is under
`/tmp/studiocast-cuda-header-final-critic-evidence-20261117`. The critic made no
source commit. Documentation validation passes all 43 Markdown files, and
integrated diff checks pass.

All accepted local repairs are integrated. No assertion, runtime, model, catalog
or hardware qualification boundary is changed by this repair. Fresh hosted CI
on the pushed final head remains the external gate before merge; the PR records
its live run and outcome without treating skipped hardware jobs as qualification.

### Hosted numerical-test dependency provisioning

Hosted [run 44112036790](https://github.com/mexicantexan/studiocast/actions/runs/44112036790)
at `e4579c4` successfully compiled all four build profiles and passed package
smoke. Each full CTest suite had one failure: the FLX sampler/numerical regression
could not import NumPy. All other tests passed or explicitly skipped native
opt-ins (85 tests per default profile, 94 with enabled features). This is a
missing test-environment prerequisite, not a failed numerical comparison.

NumPy is an existing conversion-tool dependency declared in
`scripts/models/video-ncnn.lock.json`; its small regression cases must remain
active. The bounded repair provisions the distro `python3-numpy` test package in
the existing no-GPU CI build jobs and binds CTest to `/usr/bin/python3`, matching
the interpreter for which the package is installed. This does not change the
pinned full-export/conversion tooling, install packages on the development host,
add a runtime dependency or modify numerical assertions. GPU opt-in and release
jobs remain outside this repair.

Topic `codex/ci-distro-numpy-validation-20261117`, source `d514a44` → integrated
`977f7fa`, changes only `.github/workflows/ci.yml` and
`docs/DEVELOPER_GUIDE.md`. It reproduced the exact missing-NumPy error with
Python site packages disabled. Using the existing distro NumPy 1.21.5, all eight
video conversion, seven MODNet conversion and seven hardware-workflow checks
passed; the numerical cases executed. YAML parsing, job-scope comparison and
shell syntax checks passed. The agent ran no host package installation.

Root independently configured CMake from a fresh HOME/XDG root with explicit
`Python3_EXECUTABLE=/usr/bin/python3`, the no-GPU Release profile and a
nonexistent ICD. All three selected CTest suites passed in 4.01 s, including
both numerical converter suites. The interpreter imported the existing NumPy
1.21.5 from `/usr/lib/python3/dist-packages`. Log
`/tmp/studiocast-pr22-numpy-integrated-20261117.log` begins with tested HEAD
`977f7fa`; agent evidence is under `/tmp/studiocast-ci-numpy-evidence-20261117`.
Fresh independent critic `ci_numpy_final_critique` reviewed integrated
`977f7fa` and this documentation follow-up, finding no blocker/high or valid
unmet requirement. It reproduced the missing-NumPy failure and independently
passed all eight video, seven MODNet and seven hardware-workflow checks without
skips. Its scratch CMake configuration passed four selected CTests, including
documentation validation, in 3.94 s; CTest's JSON confirmed `/usr/bin/python3`
for every selected suite. Its YAML comparison verified exactly three dependency
additions and three interpreter flags, unchanged remaining jobs, correct
install/configure/test order and valid shell syntax. Evidence is under
`/tmp/studiocast-ci-numpy-critic-evidence-20261117`. The critic made no source
commit or host changes. All accepted repairs are integrated; actual hosted
package installation and the complete build/test/self-test matrix remain the
external gate recorded in the PR.

## Hardware qualification checkpoint (2026-11-17)

At this checkpoint, merge was on hold for a native timeout/teardown SIGSEGV and
hardware qualification was incomplete. The terminal-cleanup follow-up below
records the subsequent repair and its remaining hardware limits.
The frozen qualification source is
`73afcecb1915c2b6f456664ffac5471c07c203d4`, with the activated `2026-11-14`
catalog and its unchanged seven packs/eighteen artifacts. Native tests use
new isolated fixture metadata and hash-verified existing artifact bytes; they
do not repair or replace installed user packs. Release builds and evidence
remain isolated from the user's configuration.

Available hardware is an NVIDIA Quadro P1000, driver 535.309.01, 4 GiB VRAM,
with approximately 1.2 GiB occupied by existing desktop work at discovery.
Repository runner inventory reports zero self-hosted runners, so AMD/Intel
execution cannot be obtained here. Only Mesa overlay and Intel null-hardware
layer manifests are installed; Khronos validation and `vulkaninfo` are absent.
The source and pinned ncnn runtime expose no supported forced non-push-descriptor
override. These missing prerequisites remain unqualified.

The HD camera `/dev/video0` supports the bounded 320×240 test; `/dev/video1`
and `/dev/video3` are metadata nodes and `/dev/video2` is an IR-only GREY camera.
The existing `/dev/video10` loopback is output-only with exclusive caps and an
uninitialized 0×0 format. Query opens succeed and no holder is observed, but
partial `/proc` visibility prevents an absolute free-device guarantee. The
installed user daemon is inactive. The planned private daemon/consumer would
stop on busy devices, disable all audio creation, discard frames and preserve
user settings; it has not run.

The qualification phases are serialized native Vulkan/CUDA matrices and cold
diagnostics, followed by a bounded real-camera/loopback run. A separate
fake-context ASan/UBSan/LeakSanitizer target does not initialize the GPU and
cannot substitute for full native sanitizer qualification. Earlier failed
deadline cases remain retained; fresh outcomes do not erase them or change the
100 ms fence and synchronization limits.

The first native pass stopped immediately when `all_remove_odd`, 65×49 →
32×24, timed out at 104.312 ms with zero active/output success and then exited
by SIGSEGV. Other matrices, cold diagnostics, CUDA and real camera tests were
paused at that safety boundary. Original evidence is retained under
`/tmp/studiocast-hw-qualification-20261117`, including `qualification.json`,
`REPORT.txt` and the exact case log. This outcome is a merge blocker; successful
component or software CI results do not exonerate it.

Auto Frame also failed the complete-artifact matrix's expected matte-inference
check after a face-only setup. A read-only contract audit confirms that matting
is optional for ordinary face-based Auto Frame. The original qualification
failure remains retained until its absent matte is explained; runtime behavior
and the healthy-fixture assertion are not being weakened to relabel it a pass.

Seven scoped diagnostic invocations retain the original binary, models,
geometry and deadline. The seventh reproduced the natural timeout followed by
SIGSEGV with a fresh private cache. Its faulting worker stack is inside
`libnvidia-glcore.so.535.309.01`; that location does not establish a driver-only
cause. Debugger observations in separate invocations proved a successful late
fence wait and child shutdown, so those observations do not cover a failed late
proof or clear the intermittent crash. Evidence is under
`/tmp/studiocast-hw-crash-diagnosis-20261117`; the original qualification logs
and their hashes remain unchanged.

A standard GNU ASan/UBSan production-smoke build from the same frozen source
completed, reusing the existing external ncnn and system libraries. One native
`all_remove_odd` invocation exited with an allocation/deallocation mismatch:
the smoke harness overrides deleting operators with `free`, but omits the
nothrow allocation overload used to create the production executor. The stack
reaches explicit terminal shutdown after the frame/lifecycle checks; buffered
stdout did not survive the sanitizer abort, so exact counters are unavailable.
This is a separate harness defect, not evidence that the original crash is
fixed. Leak checks were enabled but did not reach a clean conclusion. Evidence
is under `/tmp/studiocast-asan-all-remove-odd-20261117-y1p2azmm`.

The bounded harness repair is topic
`fix/vulkan-smoke-nothrow-allocation-family`, source `a648594` → integrated
`125fa55`. The production smoke and a GPU-free regression now share the exact
replacement operators, including scalar/array/aligned nothrow allocation and
matching deletion. No production code, model, deadline or native assertion is
changed. The regression reproduced the original ASan mismatch before the fix;
afterward it passes 120 allocation/deletion pairs and exactly 40 gated counts
with mismatch, address, undefined-behavior and leak checks enabled. Root's
independent integrated-source compile/run also passes with warnings treated as
errors; evidence is under
`/run/user/1000/studiocast-allocator-integrated-y4lbqmvp`.

Fresh-context critic `smoke_allocator_final_critique` found no repair-specific
blocker, High finding or valid unmet requirement. Its frozen-source GCC 11.4
`-O3 -flto -Werror` sanitizer run and fresh CMake Release sanitizer CTest both
pass, with mismatch and leak checks enabled. Optimized plain Clang 14 and direct
Clang sanitizer launches also pass. Two Clang sanitizer launches through Python
`subprocess.PIPE` spun and were bounded/terminated; shell-pipe and PTY launches
pass, and disassembly shows no recursive allocation forwarding. This
launch-dependent caveat remains unclassified, without a universal Clang
sanitizer claim. Allocation failure was reviewed statically, not fault-injected.
Critic evidence is under
`/run/user/1000/studiocast-allocator-critique-t3w76z1u`; it made no source commit.

The reviewed harness was rebuilt into the existing GNU production sanitizer
profile. Its GPU-free allocation CTest passes with stack-use-after-return,
mismatch and leak checks enabled. Exactly one new native `all_remove_odd`
invocation used fresh private state, the same models/geometry/deadlines and
ordinary PTY stdout capture. It reached the 180-second outer diagnostic limit
during setup and was terminated, with no frame or summary rows. No sanitizer
error was reported before termination, but there is no clean sanitizer or leak
conclusion. Source is `125fa55`, binary SHA-256
`ebe04071f3d03342fa197461bf61c3110368e67ed62867ccd3c7676f34bf7011`;
evidence is under
`/tmp/studiocast-asan-all-remove-odd-125fa55-20261117-22kxo3zu`.

Automatic safety review rejected continuation of the diagnostic agent for
"possible cybersecurity risk." The already-started invocation's completion
was collected, its child and wrapper exited, and no restart or alternate native
execution followed. The rejected post-fault attachment proposal and the later
standard-sanitizer workflow are recorded as tooling limitations, not as a
cybersecurity finding in StudioCast. The original crash and the outstanding
hardware gates remain unresolved; the harness repair does not clear the merge
hold.

Before the safety stop, the six MODNet golden cases, four generic model golden
comparisons (FastDVDnet, YuNet and both FLX eyes), native GPU tracking/dlib/FLX
oracles and final-stage integration passed. MODNet's worst maximum absolute
error was 0.000112742 with mask IoU 1; generic-model maximum errors ranged from
0.00000220537 to 0.0000290871. The generic tests observed zero host Execute
allocations and actual FP64 device creation. These component results do not
substitute for the production matrix.

The first production pass reached sixteen cases: nine passed (`mirror`,
`vignette`, `denoise`, `denoise_zero`, `eye_no_face`, `eye_zero`,
`tracked_vignette`, `keylight`, `all_blur`) and seven failed:

| Case | Retained observation |
| --- | --- |
| `auto_frame` | Face-only setup; expected healthy-fixture matte inference absent. |
| `blur` | Degraded setup, status 3/reason 28, zero stages; cause unproven. |
| `all_blur_odd` | First frame timed out at 105.736 ms. |
| `all_blur_face` | Degraded setup, six of seven stages, status 3/reason 28; cause unproven. |
| `remove` | Degraded setup, status 3/reason 28, zero stages; cause unproven. |
| `all_remove` | First frame timed out at 105.251 ms. |
| `all_remove_odd` | First frame timed out at 104.312 ms, then SIGSEGV. |

Successful production records observed one upload, one final readback and one
synchronization, with zero intermediate readbacks, CPU tails or C++ Execute
allocations. There were 37 writer commits; the small sample does not establish
global p95 latency qualification. Remaining compositions, isolation/budget and
default checks, cold diagnostics, native CUDA and real-camera/loopback execution
remain unrun after the safety stop. No installed daemon, camera settings,
loopback configuration or user model pack was changed.

The pre-repair native C++ allocation counts have the original replacement
family's coverage limitation. The repaired complete-family harness has not
completed a native frame run, so those historical zero counts are not complete
allocation qualification. Numerical comparisons and independently recorded GPU
resource/transfer counters remain separate observations.

### Strict local compiler verification

Proportional software verification also exercised the existing disabled-backend
Release configuration with `STUDIOCAST_ENABLE_WERROR=ON`. Its failures are
retained; warning flags, native opt-ins and behavioral assertions are not
relaxed. Routine delivery repair extended to the following bounded changes:

| Topic | Source → integrated commit | Change and focused evidence |
| --- | --- | --- |
| `fix/ncnn-network-parser-shadow` | `7ab3a4f` → `6aa941a` | Rename one loop-local JSON entry; identical strict compile fails before and passes after. |
| `fix/vulkan-factory-feature-guard` | `50800b3` → `7824224` | Guard the Vulkan-only semantic-bit helper with its callers' feature condition; strict disabled/enabled-without-ncnn compiles pass, enabled preprocessing is byte-identical. |
| `repair/cuda-auto-frame-werror` | `7b295fa` → `65a74f2` | Make existing numeric conversions explicit; strict compile passes and Release `.text` bytes match baseline. Existing fake-driver Auto Frame CTest passes 1/1. |
| `fix/camera-pipeline-unused-backend-argument` | `62c29e2` → `0be1183` | Remove an unused private lambda argument and its pure field-read caller argument; identical strict compile passes. |
| `fix/vulkan-quarantine-test-coownership` | `266eec4` → `aa451ff` | Use one shared aggregate and an effects alias to retain the shutdown-proof flag, avoiding a GCC allocation warning; strict compile and GPU-denied full factory test pass. All retention and positive-proof cleanup assertions are unchanged. |
| `repair/cuda-fixture-werror` | `b1c50b5` → `a784f02` | Explicit conversions in four CUDA kernel fixtures; all strict compiles pass and all four complete object files match baseline byte-for-byte. Three default fake tests pass; native-only FastDVDnet correctly skips with exit 77. |
| `fix/maxine-path-copy-string-construction` | `e78ca46` → `e49f5fa` | Explicit string append avoids a GCC12 `restrict` diagnostic; strict GCC11/12 compiles and 30 public path-formatting cases per compiler pass, preserving existing HOME-prefix behavior. |
| `fix/gcc12-status-copy-bound` | `b3cc22e` → `91198f4` | Make the effects destination capacity explicit; strict GCC12 compile and focused status CTest pass, preserving malformed-count rejection and validation precedence. |
| `repair/string-construction-werror` | `b767576` → `a3e73bd` | Append command spaces before the unchanged quoting helper, and explicitly construct an audit token. Strict GCC11/12 compiles, four existing exact-command/failure checks and the capability audit pass. |
| `fix/optimized-test-allocation-hooks` | `b0c42c5` → `89e4a1e` | Keep real frees visible to ASan behind a test-only noinline boundary, and complete YuNet's independently failing local allocation family. Strict GCC11/12/Clang14 builds and GPU-denied YuNet fake tests pass. |
| `repair/auto-frame-fixture-werror` | `99a8232` → `ae05dd8` | Two explicit fixture conversions; GCC11/12 strict compiles and default fake tests pass, with complete objects byte-identical to baseline. |

These repairs preserve runtime algorithms, routing precedence, model integrity
checks and lifetime requirements. The source repairs introduce no dependency,
catalog or installed-pack change. The CUDA Auto Frame focused test build uses
the ordinary non-Werror profile because unrelated baseline errors exist there;
its separate exact-flags object compile retains `-Werror`. Its first `ctest`
Python wrapper could not import CMake under a temporary HOME; the existing
system `/usr/bin/ctest` runs the same test successfully without a host change.

Focused evidence is under `/run/user/1000/studiocast-parser-shadow-check`,
`studiocast-vulkan-factory-guard-check`, `studiocast-camera-unused-argument-checks`
and `studiocast-vulkan-quarantine-test-check`, plus the
`/run/user/1000/studiocast-cuda-auto-frame-werror-*.log` files. Quarantine-test
linking uses a hash-verified private snapshot of eight existing disabled-backend
archives, not mutable build outputs. Broad integrated verification remains the
gate for the combined repairs.

The four-fixture batch preserves numerical formulas, input bytes, tolerances,
assertions and native opt-ins. Its 5/1/10/50 original diagnostics and successful
strict compiles, object comparisons and private-archive links are recorded
under `/run/user/1000/studiocast-cuda-fixture-werror-checks`. The FastDVDnet skip
is not represented as hardware qualification. None of these software repairs
changes the unresolved native timeout/SIGSEGV outcome.

The fresh GCC12 discovery pass retained seven failing translation units and
all dependency-blocked targets before batching repairs. The status-copy
diagnostic refers to offset 336776, exactly one past its nine-element effects
array; both source and destination capacities are nine, and the original guard
permits only indices 0–8. It is a compiler bounds-analysis false positive, not
evidence of a runtime overrun. The additional bound leaves the original count,
zero defaults and validation sequence intact. Tests cover counts 0/1/8/9/10/
SIZE_MAX, final-element copying, partial-prefix diagnostics, cleared outputs and
generation/aggregate-overflow precedence. Evidence is under
`/run/user/1000/studiocast-gcc12-status-check`.

The allocation-hook follow-up distinguishes two findings. The complete Vulkan
family already passes runtime sanitizer checks before the noinline boundary;
its optimized compiler warning is a false positive. YuNet's original local
hooks independently reproduce the real nothrow allocation/free mismatch, so
its family is completed too. GNU11/12 optimized ASan/UBSan checks pass the
unchanged Vulkan 120-pair/40-gated-count regression and 40 YuNet pairs/counts,
with mismatch and leak checks enabled. A negative control omitting nothrow
replacements still fails ASan across the new boundary. No sanitizer check,
warning flag, native opt-in or behavioral assertion is disabled. Evidence is
under `/run/user/1000/studiocast-optimized-allocation-hooks-check`.

Additional focused evidence is under
`/run/user/1000/studiocast-maxine-path-copy-check`,
`/run/user/1000/studiocast-string-construction-checks` and
`/run/user/1000/studiocast-auto-frame-fixture-checks`. Audio checks use existing
fake command hooks, empty PATH and an aborting fallback executor; no real audio
command is executed.

Final integrated code head `ae05dd810974b1c5e67c83dbc1253f1dae622a31`
passes both strict Release builds and full CTest suites:

| Compiler/profile | Full build | Full CTest |
| --- | --- | --- |
| GNU11, optional backends/kernels/dlib disabled, benchmarks built | Exit 0, 57.955 s incremental final build | 86 registered: 81 passed, five explicit native skips, zero failed; 111.51 s. |
| GNU12, optional backends/kernels/dlib and benchmarks disabled | Exit 0, 209.968 s incremental final build | 86 registered: 81 passed, five explicit native skips, zero failed; 52.588 s. |

Both profiles retain `STUDIOCAST_ENABLE_WERROR=ON`, private HOME/XDG roots,
nonexistent Vulkan ICD paths and absent native opt-ins. The skips are Open CUDA
production smoke and native matte bounds, mirror, Eye Contact and FastDVDnet
kernels. Building benchmark tools does not execute them. No local self-test or
custom driver interposition is run during this follow-up; hosted CI's portable
self-test remains a separate final-head gate.

GNU11 evidence is under
`/run/user/1000/studiocast-hardware-handoff-checks-8excdglf`, including
`gcc11-final-{build,ctest}.log` and `gcc11-final-results.json`. Initial failures,
the keep-going discovery logs and the recorded resource/integration changes
remain retained. Two unchanged GNU11 test translation units eventually compile
successfully after unusually long optimization; no optimization flag is
reduced. The final build and suite above run after all repairs are integrated.

GNU12 evidence is under
`/run/user/1000/studiocast-final-gcc12-xwgYeSAr`. Its first full CTest run fails
four IPC/GUI/installer tests because the long private TMPDIR makes UNIX socket
paths exceed the OS limit. Only the fixture environment is corrected: a short
private `/tmp/s12-0hj0j6tt` root and mode 0700 for its own XDG runtime directory.
Conservative socket lengths including NUL change from 124/119/140/114 to
91/86/107/81 bytes. The full rerun uses identical code, compiler, flags and
native opt-ins and passes all four previously failing tests. The failed
`final-ctest.log` is preserved alongside `final-ctest-short-temp.log`, fixture
length/permission evidence and exact environments. Successful final build and
CTest log SHA-256 values are respectively
`9537661561fd1884ca554390d2ec6038900715dd8157e3ea993b9c01d1463c20` and
`1eee9c6001ef5ae64c3c2f94fc1418bf4d0a1b1aa2cbc4147e87e10130429032`.

Fresh-context critic `hardware_delivery_final_critique` reviewed every changed
source file since `73afcec`, both documentation diffs and the applicable
policies/contracts. It found no new repair-specific blocker, High finding or
unmet requirement. Independent frozen-source GNU11/12 `-O3 -flto -Werror`
ASan/UBSan runs pass the 120-pair/40-gated-count allocator regression with
mismatch/leak checks enabled in a GPU-denied sandbox. It verifies both complete
CTest results, the GNU12 hashes and retained fixture-failure evidence, and
passes documentation validation for all 43 Markdown files. Evidence is under
`/run/user/1000/studiocast-final-critique-b43cs9im`. This approves the repairs'
scope; it does not approve merge or qualify the hardware. The original native
SIGSEGV, remaining matrices/camera transport and clean full-native sanitizer
gate remain unresolved. The critic makes no source commit.

The task's agent ownership and handoffs are recorded here; source/topic and
integrated commit mappings are given above:

| Agent | Ownership/outcome |
| --- | --- |
| `hardware_native_qualification` | Frozen native qualification and seven diagnostic probes; read-only teardown evidence and CPU-only debugger feasibility check; no source commit. |
| `camera_qualification_audit` | Camera plan, isolated sanitizer builds and the two native sanitizer attempts; camera remains unrun; no source commit. |
| `auto_frame_hardware_failure_audit` | Read-only optional-matte and native-owner destruction-order audits; no source commit. |
| `smoke_allocator_repair` | Six focused topics: initial smoke family, parser rename, Vulkan feature guard, quarantine fixture, Maxine path display and optimized allocation hooks; all accepted commits integrated. |
| `cuda_auto_frame_warning_repair` | Four focused topics: Auto Frame conversions, four kernel fixtures, string construction and Auto Frame fixture conversions; all accepted commits integrated. |
| `camera_unused_argument_repair` | Private status-selector argument cleanup; accepted commit integrated. |
| `snapshot_copy_bound_repair` | Fixed-capacity copy proof and boundary tests; accepted commit integrated. |
| `final_gcc12_software_validation` | Independent build/test operation on frozen code and causal temporary-root correction; no source commit. |
| `smoke_allocator_final_critique` | Independent initial allocator and catalog/documentation critique; no source commit. |
| `hardware_delivery_final_critique` | Fresh final integrated critique described above; no source commit. |

Root reviews and integrates all twelve accepted implementation commits. At the
pre-push audit, `origin/master` remains
`adfb648efe6b7666f5bdcc1455e5a977971eda7b` and is an ancestor of code head
`ae05dd8`; `origin/v0.2.9` remains `73afcec`. PR #22 is open with no auto-merge
request. The conditionally authorized merge is withheld because the native
safety failure remains unresolved. No release, tag, deployment or installed-pack
mutation is performed.

### Hosted same-version CPU SDK acquisition repair

The first post-qualification push, documentation head `bda542c`, starts hosted
[run 44214521214](https://github.com/mexicantexan/studiocast/actions/runs/44214521214).
Ubuntu 22.04, Ubuntu 24.04, Linux Mint and package smoke pass; the enabled-features
job fails before compilation while downloading its existing CPU ONNX Runtime
1.17.3 tarball. Direct URL, release API and repository API checks also return
HTTP 404. This establishes current endpoint unavailability, not deletion of a
particular release. No blind CI retry or SDK upgrade is used.

Microsoft's [official C++ documentation](https://onnxruntime.ai/docs/get-started/with-cpp.html)
identifies `Microsoft.ML.OnnxRuntime` as its CPU release distribution. The
official same-version NuGet package is available and independently audited:

| Identity | Verified value |
| --- | --- |
| Package/version | `Microsoft.ML.OnnxRuntime` / `1.17.3`, Microsoft authors/owners |
| Archive | 83,662,121 bytes; SHA256 `e7fecf6fe6ae54d5e43664c4e2354138b5b86d4f0ccf946fbab1ef48650524e3` |
| Published integrity | Computed SHA512 matches official NuGet response metadata. |
| Native library | 15,469,240 bytes; SHA256 `8bdcd79ab25e38d1d7646948e07a7d87ed95c2164a04e43dd685147fd5c86b4c` |
| Native identity | Linux x64 CPU, SONAME `libonnxruntime.so.1.17.3`, API 17; only standard glibc/libstdc++ dependencies. |
| Preserved contents | Nine C/C++ headers, `LICENSE` and `ThirdPartyNotices.txt`; no managed or GPU companion is needed for this native SDK. |

The bounded repair changes acquisition of the already-declared Microsoft SDK,
preserving its vendor, CPU flavor, x64 architecture and exact version. It does
not claim tarball byte equivalence. Topic
`fix/ci-onnxruntime-cpu-acquisition`, source `6de1812` → integrated `821fa75`,
changes only the existing CPU acquisition step's run body, a fixed-identity
stdlib staging helper, one hermetic test script and its CTest registration.
Existing install prefix, pkg-config, ldconfig, feature flags, assertions and
opt-in jobs remain unchanged. Normal user/GPU SDK setup is not modified or
requalified by this repair; the original native safety failure remains a gate.

The helper checks size and pinned SHA256 before ZIP parsing and stages only
the fixed allowlist into a new destination. It rejects unsafe/duplicate,
encrypted, symlink/special or colliding members, wrong identity/API and existing
or symlinked destinations. Library bytes are staged under their versioned
SONAME with a relative `libonnxruntime.so` link. Both notices are preserved.
No archive metadata is executed, and the helper exposes no checksum bypass.

Agent `ort_cpu_acquisition_audit` audits and implements the focused topic in an
isolated worktree. Four hermetic test methods, seven existing hardware-workflow
checks, eight video-converter and seven MODNet checks pass. YAML parsing, all
57 workflow shell blocks and parsed-workflow comparison pass; only the CPU
acquisition run body differs. Actual audited archive staging and a C++17 header
syntax check pass without linking, loading, executing or installing the SDK.
Root independently repeats the four preparation and seven workflow tests,
stages the audited bytes, verifies the library hash and passes the header-only
check. Evidence is under `/run/user/1000/studiocast-ort-audit-52ciighw` and
`/run/user/1000/studiocast-ort-root-review-lk4h8m21`.

Fresh-context critic `ci_sdk_acquisition_final_critique` finds no new blocker,
High finding or unmet requirement in `bda542c..821fa75`. It independently
passes the four helper and seven workflow tests, real-archive staging, exact
library/license checks and static ELF inspection; it confirms the preserved
SDK and workflow contracts. It makes no source commit or SDK/runtime changes.
Evidence is under `/run/user/1000/studiocast-ort-critic-odir6cir`.

At integrated code head `821fa750d60a38f78f997cf46572b486b1747a12`, both
strict Release profiles pass full builds and CTest: 87 registered, 82 passed,
five explicit native skips and zero failures each. GNU11 build/CTest take
167.472/104.98 s; GNU12 take 92.266/44.320 s. The extra registered case is the
hermetic SDK preparation test. Evidence is under
`/run/user/1000/studiocast-ort-integrated-r8a1lhsp` and the existing GNU12 root's
`ort821-{build,ctest}.log`/`ort821-results.json`. Prior failed logs remain intact.
All checks use private roots and absent native opt-ins; no local self-test or
downloaded SDK execution, linking, loading, host installation or service change
occurs. A fresh hosted final-head run remains the end-to-end acquisition and
self-test gate; this repair does not clear PR #22's native SIGSEGV hold.

## Vulkan terminal cleanup follow-up (2026-11-18 through 2026-11-20)

Status: the exit crash is repaired and its native regression is validated;
hardware qualification is partial. PR #22 records final-head hosted CI and merge
state. The user renewed implementation, GitHub Actions and conditional merge
authority, explicitly accepting best-effort hardware qualification and deployment
feedback. No release, driver/package installation, service mutation or installed
model change is included.

Hosted Actions run `44220246291` completed successfully at exact source
`7d529938c48f79759ac29ce2a7de37007317b199`. Ubuntu 22, Ubuntu 24 and Mint each
passed 82 tests with five native skips and `SELFTEST OK`; enabled features passed
90 tests with six native skips and `SELFTEST OK`. Package smoke passed AppDir,
source archive and bundle verification. The official ONNX Runtime 1.17.3 package
acquisition succeeded. All six hardware jobs skipped; the runner inventory is
still zero. The enabled build is not warning-clean. Evidence is under
`/run/user/1000/studiocast-ci-confirm-pw27_q73`.

Read-only ownership audits found a real release-proof gap: the factory could
permit release after a void executor shutdown refused callback or child
retirement. The concrete checked shutdown now gates normal shutdown, failed
setup rollback and destruction. A fresh critic then found that public
`Initialize()` could replace a retained failed candidate before any explicit
shutdown attempt. The retained-candidate guard closes that path; safe discarded
failures remain retryable.

| Topic | Source commit → integrated commit | Verification |
| --- | --- | --- |
| `fix/vulkan-checked-executor-shutdown` | `71b19e6` → `8ca35a0` | Factory regression red before repair, green afterward; real fake-context executor refuses shutdown despite global fence proof until callbacks retire. |
| `fix/vulkan-initialize-retained-candidate` | `7f1e27c` → `8729c4b` | Public initialization/rollback regression red before guard, green afterward; strict disabled and enabled compilation passes. |
| `test/vulkan-retained-timeout-proof` | `28725dd` → `ae4fd30` | Exact private dispatch timeout stub, scope/stickiness negative control and separate opt-in retained-owner case; no production frame-loop change. |

At `8ca35a0`, strict GNU11 and GNU12 Release builds each pass all 87 registered
tests: 82 passed, five native skips, zero failures. At integrated `ae4fd30`, GNU11
again passes the full build and the same 87-test suite (35.04 s); the native ncnn
build's three GPU-denied factory/batch/lifetime tests also pass. Fresh critic
`retention_guards_fresh_critique` finds no new repair-specific Blocker/High or
unmet requirement at `ae4fd30`. Earlier critic's initialization-retention High
finding is remediated, rather than waived. Evidence includes
`/run/user/1000/studiocast-retention-integrated-ks9apnc6` and
`/run/user/1000/studiocast-retention-critique-h874g87m`.

These repairs alone do not fix the native crash: the original uninstrumented
`all_remove_odd` reproduces a natural 101.838 ms timeout followed by SIGSEGV at
`8ca35a0`. Standard launch-under-GDB also reproduces it. All-stop GDB captures
NVIDIA's `[vkrt] Analysis` worker faulting at glcore ELF offset `0x19309f0`, while
the main thread is inside NVIDIA/GLX cleanup through the dynamic loader. Signal
location alone is not a driver-only attribution. Evidence is under
`/run/user/1000/studiocast-native-checked-f1m9yv3h` and
`/run/user/1000/studiocast-current-allstop-f85hresv`.

The isolated `unsafe_timeout_retained` case submits actual work, injects an
initial timeout and denies every subsequent fence proof through its private
dispatch table. It verifies no output/activity, rejected retries, unchanged
allocations, pending shutdown, denied release, and weak configuration/context
retention after manager destruction. Normal process return then reproduces
SIGSEGV (20.61 s). Reducing this test to denoise alone also reproduces SIGSEGV
(7.39 s). A diagnostic twin changes only terminal return after those assertions:
flushing output and calling `std::_Exit(0)` avoids the fault (15.50 s). That
diagnostic zero status is not production success, quiescence proof or sanitizer
qualification. Original, twin and minimized binaries and patches are preserved
under `studiocast-native-retained-proof-jbatj3jy`,
`studiocast-retained-exit-twin-21x8xt0u` and
`studiocast-retained-minimal-xwelgtd9` in `/run/user/1000`. Temporary source edits
are restored.

The implemented repair records actual permanent native retention, allows normal
operation/fallback to continue, and makes the already-requested terminal process
exit fail visibly after ordinary application cleanup. It preserves pending and
denied-release evidence, avoids unsafe userspace finalizers, and adds no GPU
wait or frame-loop work. This does not prove bounded kernel/driver reclamation
or clean native sanitizer reporting.

| Topic | Source commit → integrated commit | Result |
| --- | --- | --- |
| `fix/vulkan-retained-process-exit` | `232da9f` → `a5a1e25`; `8eb7661` → `6034ffb` | Neutral utility latch and terminal guard, permanent-retention producers and native entry points; pure fake tables do not select native exit handling. |
| `fix/native-vulkan-retention-test-isolation` | `38ba80f` → `52ffe40` | Required native mains use the same terminal guard; intentional permanent-quarantine assertions run in checked exec children. |
| `fix/qualified-vulkan-pre-submit-loss-test` | `75b4df0` → `78243ca` | Restore the original pre-submit device-loss assertions in the normal test path; this fixture has no pending native submission and does not require permanent retention. |

`core/util/process_exit` has one monotonic latch. A temporary timeout or pending
shutdown alone does not set it. Actual irreversible retention does, including the
setup lease's whole-factory retention path where the factory destructor never
runs. Each native application body returns before `FinalizeProcessExit` checks
the latch, so ordinary local service, camera, audio and manager cleanup finishes
first. Normal exit statuses remain unchanged. Retention emits an error, flushes
stdout/stderr and terminates with status 1 without userspace finalizers. In
particular, an existing status 77 cannot turn unsafe retention into a CTest skip.

The process-exit regression covers six real GPU-free subprocess cases, ordinary
statuses 0/7/77 and retained statuses 0/7/77. It proves local destruction, weak
owner retention and skipped `atexit` handlers. Native negative children require
exact status 1, their completed-assertions marker and the guard diagnostic;
signals, timeouts, missing evidence and skips fail. No generic `WILL_FAIL` is
used. The kernel move preserves all 293 assertion/comparison calls. A native run
exposed the incorrect assumption that a synthetic queue-submit loss necessarily
retained an owner; the original assertions were restored verbatim rather than
changing production cleanup or weakening them.

### Final local evidence and limitations

At runtime source `52ffe40`, GNU11 and GNU12 strict disabled-backend Release
profiles each pass 88 tests: 83 passed, five explicit native skips, zero failures.
GNU12 build/CTest take 33.334/34.516 s. At final code `78243ca`, GNU11 again passes
the full build and 88-test suite (63.135 s). Evidence is under
`/run/user/1000/studiocast-exit-final-xvu5lzom`,
`/run/user/1000/studiocast-final-gcc12-xwgYeSAr/exit52-*` and
`/run/user/1000/studiocast-delivery-final-u5cei7rd`. Local CMake/CTest invocations
use their actual executables because the user-local Python shims cannot import
their package under an isolated HOME. Failed launcher logs remain preserved.

The native Vulkan Release build retains the prior profile and verified replay
SDK. The formerly crashing retained-owner case now exits exactly 1 with both
retention witnesses and the guard diagnostic, without a signal. Four additional
original `all_remove_odd` processes produce three passes and one deadline
failure, preserved as normal exit 1 without a signal. Existing `unsafe_timeout_all`
passes its positive late-proof control. Evidence is under
`/run/user/1000/studiocast-vulkan-exit-qualified-40qan67z`.

All six native component selections pass: MODNet numerical goldens, generic
FastDVDnet/YuNet/FLX goldens, GPU tracking/dlib oracles, utility kernels with the
proper empty-model control, final-stage integration and the corrected qualified
executor. The actual manager matrix passes 25 of 29 selections, including
Auto Frame's matte-inference assertion, real-face eye correction, model
isolation, six budget axes, installed defaults and both timeout controls.
Successful cases retain the one-upload/one-final-readback/one-synchronization,
zero-intermediate-transfer, zero-CPU-tail, complete-family zero-Execute-allocation
assertions. Both normal and missing-landmarks cold diagnostics pass. Native
evidence is under `/run/user/1000/studiocast-native-final-xtj7i06x`; the corrected
executor rerun is under
`/run/user/1000/studiocast-qualified-native-fixed-d0gzvmmk`.

The four manager failures remain failures, not skips or qualification passes:
`all_blur_face`, `all_remove_face` and `all_replace_face` at 320×240 degrade to
six stages during setup; a scoped diagnostic captures their model-preparation
`vkWaitForFences` `VK_TIMEOUT`. `all_replace_odd` times out on its first frame at
102.530 ms. The 100 ms policy is unchanged. No signal occurs in these tests.
The temporary setup diagnostic is preserved under
`/run/user/1000/studiocast-setup-failure-trace-_ggnw83v` and removed from source.
These observations do not qualify sustained latency, FHD or full face-sized
composition on this P1000. Existing desktop work and a concurrent two-job
CPU-only CUDA build also exclude controlled performance claims.

The separate current-source CUDA Release build matches the frozen prior CUDA
configuration with zero profile differences. Required CUDA runtime self-test
and all sixteen strict/automatic effect compositions pass. The pre-existing
SDK and frozen binaries remain unchanged; no dependency was installed or
downloaded. Build and execution evidence are under
`/run/user/1000/studiocast-cuda-current-iwn8f5na` and
`/run/user/1000/studiocast-cuda-final-_22ejumd`. This uses generated RGB and an
in-memory writer; it does not qualify real-face CUDA correction or camera
transport. The final test-only correction does not change CUDA runtime sources.

Camera qualification is partial. The private daemon captured from `/dev/video0`
and recorded 280 real Vulkan mirror/vignette writer commits at approximately
15 FPS, with 280 uploads/readbacks/synchronizations and zero intermediate
transfers, CPU tails or fallbacks. The raw consumer did not receive frames, so
this is not an end-to-end camera pass. The physical camera's original
1280×720 MJPG / 30 FPS settings were restored and the private daemon exited 0.
Subsequently `/dev/video10` reports neither capture nor output capability and
rejects format negotiation with `EINVAL`, preventing the second phase. No
kernel/module reset, installed service change or privilege change was attempted.
The loopback is not restored to its original uninitialized format; its current
unavailable state needs host-side recovery/qualification. No captured image
buffer was saved to a file. Evidence is under the `studiocast-camera-final-*` folders
in `/run/user/1000`, particularly `yavvmawl` and `5ep5wk_2`.

Automatic approval review rejected the subsequent read-only camera/loopback
audit for "possible cybersecurity risk." This is a tooling limitation, not an
application security finding. That audit was not retried through another
mechanism. Camera consumer reads, model-backed camera transport, AMD/Intel,
Khronos validation layers, forced non-push-descriptor execution, sustained
latency and clean full-native sanitizer reporting remain unqualified. The user
explicitly authorized best-effort hardware work and deployment feedback; these
limits do not become successful hardware claims.

Fresh critics approved the integrated retention guards (`ae4fd30`), process exit
implementation (`52ffe40`) and qualified-loss correction (`78243ca`) without an
unresolved Blocker/High or unmet source requirement. They independently ran
GPU-denied tests and compared restored assertions; they did not run the GPU.
Their evidence includes `studiocast-retention-critique-h874g87m` and
`studiocast-exit-critic-o_fetg9k` under `/run/user/1000`; the final scoped critic
`qualified_loss_final_critique` independently verifies the restored assertions
and recorded native rerun. The final-head hosted workflow is still required before
the conditionally authorized normal PR merge. A merge is not a signed release
or deployment.
