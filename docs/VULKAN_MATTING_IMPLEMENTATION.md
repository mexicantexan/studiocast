# Production Vulkan matting implementation

- Status: Implementation in progress
- Baseline: `51b96122ab2ccca92af9bed35c86c7fe8e5060a0`, branch `v0.2.9`
- User decision, 2026-11-11: implement production Vulkan matting and its model
  and validation requirements; validate available hardware and record unavailable
  hardware coverage for deployment feedback.

## Scope and contract

Implement MODNet-backed background blur, remove, replace, and key light through
the existing production camera manager and prepared Vulkan island. Preserve
strict device/provider selection, model identity, optional-stage behavior,
writer-commit attribution, one upload, zero intermediate host readbacks, and one
final readback. Mirror, fixed-center vignette, and final resize remain resident
and composable. Other Vulkan model effects are outside this phase.

The runtime will adopt an ncnn-owned Vulkan context before allocating StudioCast
resources, matching the configured physical-device identity. One retained owner
holds the exact logical device and queue through every child resource. A pinned,
reviewed ncnn build patch permits recording inference once and replaying it on
that queue through StudioCast's bounded submission and health protocol. Setup
performs graph validation, GPU-only layer checks, allocation, recording, and
warmup; frames reuse the recorded graph and device buffers. Unsafe completion
must retain resources rather than release pending GPU work.

Convert the existing verified curated MODNet ONNX model offline with pinned
tooling. Preserve its ONNX artifact, identity, license, preprocessing, and
defaults; add exact ncnn artifacts, blob names, geometry, checksums, and converter
provenance. Candidate installation must work from its own local source and
verified offline cache. Active catalog URLs must name published verified bytes.
Any new external model publication needs a concrete, separately authorized
publication plan; no installed user packs are modified by implementation tests.

The user's hardware decision permits implementing and delivering this phase
without unavailable AMD/Intel hardware evidence. It does not waive integrity,
residency, queue ownership, bounded waits, or truthful runtime diagnostics.
Unvalidated device families and real-camera/visual-quality/latency coverage will
remain explicit deployment validation items, not completed claims.

## Ownership and integration

| Agent / topic | Ownership | Acceptance |
| --- | --- | --- |
| `vulkan_runtime_design` | Adopted context lifetime, ncnn bridge/runtime, pinned build patch/helper, native runtime tests | Exact context/queue, bounded submission, persistent graph, no host bridge, fail-closed unavailable build |
| `vulkan_model_design` | Offline model conversion/validation, candidate metadata, model tests | Reproducible artifacts, numerical comparison, license/hash provenance, isolated install/cache |
| `vulkan_pipeline_design` | Prepared factory/setup/executor, matting session, camera/status integration, focused tests | Actual manager executes all four consumers and supported combinations with correct attribution/transfers |
| Root | Shared CMake registration, catalog/publication integration, installer facts, docs, integrated validation | Review and merge focused commits, safe broad tests, available native evidence, fresh independent critique |

Each implementation uses an isolated worktree from the baseline. Only root
integrates into `v0.2.9`. Shared-file changes are serialized. Exact commit and
validation evidence will be recorded as each component is accepted.

## Validation and remaining work

Accepted model tooling: `ac93711` on `codex/modnet-ncnn-conversion-20261111`,
integrated as `46f743f`. Seven converter and 19 transaction tests passed; root
independently reran the seven converter tests and verified all five candidate
payload hashes/sizes. Two real exports reproduced identical payloads and six
numerical results; candidate local-source and offline-cache transactions passed.
The clean pinned AVX2/FMA reference reached maximum alpha error
`2.6345252990722656e-5`, maximum MAE `9.3061352e-8`, and zero threshold-mask
disagreements. The SSE-only reference failed the unchanged `2e-4` maximum-error
gate on black (`0.0002557933`); diagnostics isolated FP32 InstanceNorm reduction
accumulation error. This is not architecture-independent CPU qualification.
Candidate evidence and the four-file additive publication plan are under
`/tmp/studiocast-modnet-conversion-design-20261111/`; publication remains pending.

Runtime implementation `396b360` and follow-up `e825f17` on
`codex/vulkan-ncnn-runtime-20261111` are integrated as `5b55fa4`. This source
checkpoint has syntax checks and exact tracked dependency-tree reproduction;
native execution remains pending. Root registered both optional source files,
the opt-in native runtime test, and the hermetic converter suite. Focused
optional-backend-off factory/adapter/converter CTest passed three of three.

Pipeline integration: `222b025` and `7250f3f` are integrated as `5951f0b`;
quiescence/capacity repair `9b98cb5` as `9ce89d3`; allocation-counting smoke
test `5b71597` as `fa0311d`. The neutral registry now requires model/runtime
evidence for exactly the four matting consumers, and final semantic descriptors
carry writer geometry. Pending/unsafe Vulkan shutdown retains the complete
factory and denies release until native quiescence is proven. Root reran the six
optional-backend-off factory, adapter, migration, status, resolver, and contract
tests: six passed, zero failed. The integrated production-on Debug build passed.

Root added real session preparation to explicit startup/diagnostics refresh;
normal status polling still reads the daemon cache. Installer parity attestation
is limited to the exact converted MODNet artifacts and preprocessing profile,
plus successful local preparation/warmup and canonical consumer predicates.
Compatible BYO runtime validity alone does not attest numerical comparison.
The production smoke checks this refresh as well as actual manager execution.
Native model-backed acceptance is still pending: the known 1x1 convolution
already passes changing-output, allocation-reuse, CPU-rejection, low-budget, and
timeout-quarantine checks, but the full MODNet graph currently fails recording
at its first InnerProduct layer. This initial failure was subsequently traced
to resource retention and fixed as recorded below.

Runtime repair `4e7fe14`, integrated as `366103d`, preserves the 256 MiB / 512
resource caps while reusing released setup workspace and retiring completed
upload staging. It also enforces replay ABI 2, rejects internal submission/reset,
propagates command-begin failures, and adopts the exact native provider entry
point. Required P1000 native tests passed, including six real MODNet golden
cases (worst maximum error `0.000112742`, worst MAE `7.06813e-7`, all IoU 1),
changing toy outputs, allocation reuse, CPU-layer and low-budget rejection,
provider and initial-recording negatives, and timeout quarantine. Retained native
allocations: 232; complete context peak: 111,266,560 bytes. Evidence:
`/tmp/studiocast-ncnn-native-v2-final-test-20261111.log`.

Independent interim audit `vulkan_matting_safety_audit` confirmed hidden ncnn
submission and ignored initial command-begin errors; ABI 2 repairs both with
native negative tests. It also reproduced an artifact replacement race using
actual validators and the ncnn CPU loader: different weights loaded under the
old typed identity. Immutable verified snapshot loading is assigned to the
runtime agent and remains pending. Model hashes and context checks alone are
not accepted as closing that race.

The first actual camera-manager run passed real cold diagnostics (all four
consumers, exact reviewed profile, resident warmup, and no CPU fallback), then
failed blur setup with `malformed_contract` after successful native preparation.
The pipeline agent is diagnosing that integration failure; it is not an
accepted hardware limitation. The branch is not ready for deployment yet.

Root repaired the installer shadow parser's remaining two-effect Vulkan
allowlist. Exact four-consumer snapshots now require common evidence 1–12 plus
runtime/artifact evidence 15/16; incomplete evidence remains rejected and
`recommendation_eligible` remains false. The regression reproduced four rejected
complete packets before the fix, then passed. All 54 installer backend tests
passed (21.145 seconds).

Scope extension: the preceding pushed baseline's CI run `43031290469` exposed
three bounded delivery failures. Root owns the repair: explicitly include
`<algorithm>` in the Open CUDA factory; use ONNX Runtime's stable C API for
free-dimension overrides so the pinned 1.17.3 headers compile; and stage the
installer's imported `typed_video_status.py` module. Packaging fixtures now use
the CMake-selected module list so missing install entries affect real imports.
The original logs are retained at
`/tmp/studiocast-vulkan-task-pr-ci-failures-20261111.log`.
The repaired translation unit compiled against the actual upstream ORT 1.17.3
headers (syntax check exit 0), the optional-backend-off Open Video target built,
all nine packaging tests passed, shell syntax passed, and the 41-file docs
validator passed. Broad integrated verification remains pending.

Root added a separate opt-in `vulkan-matting-production` CI job using an
explicit preprovisioned ncnn prefix, model root/identity, replacement image, and
runner labels. It builds the production configuration and invokes the actual
manager smoke executable in temporary HOME/XDG roots. Missing inputs, a missing
binary, native failure, and native skip 77 fail the job; logs retain exact source
and configuration. Five hermetic workflow tests pass, including literal handling
of shell metacharacters in input paths. An installer regression verifies all four
matting consumers require complete runtime evidence and that missing matting
evidence does not disable mirror. This change does not execute remote GPU CI.

Pending: runtime, model, and pipeline implementation; focused tests; integrated
optional-backend-off/on builds and CTest; available P1000 native model-backed
tests; model-source publication decision; documentation reconciliation; and fresh
independent critique with remediation of valid findings.

Host inventory: NVIDIA Quadro P1000, driver 535.309.01, 4 GiB, Vulkan loader and
headers present. No AMD/Intel GPU is available. Dependencies are built only in
isolated task directories; no host package, service, driver, or kernel changes
are included. This task does not authorize deployment, application release, or
new publication of model artifacts. The preceding completed changes were pushed
to `origin/v0.2.9` at the baseline; further delivery is recorded separately.
