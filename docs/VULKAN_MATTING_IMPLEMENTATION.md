# Production Vulkan matting implementation

- Status: Local implementation and native matrix complete; broad verification complete;
  independent final critique in progress; model publication/catalog pending approval
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

## Integrated architecture and commit ledger

All accepted implementation is integrated into `v0.2.9`; no accepted topic is
stranded. The source checkpoint for the final native camera matrix is
`22c8724d82532e483c269e15c44952926c1e6964`. Documentation integration `0219ee6`
does not change that runtime. Root owns this ledger and common build/test/docs
files; implementation agents used isolated worktrees.

| Agent / topic branch | Focused commits | Integration commits on `v0.2.9` |
| --- | --- | --- |
| `vulkan_runtime_design` / `codex/vulkan-ncnn-runtime-20261111` | `396b360`, `e825f17`, `f74bcd6`, `4e7fe14`, `aaa0113` | `5b55fa4`, `7b4a766`, `366103d`, `5b6fb4c` |
| `vulkan_pipeline_design` / `codex/vulkan-matting-pipeline-20261111` | `222b025`, `7250f3f`, `9b98cb5`, `5b71597`, `6209a26`, `58612f0`, `8eb16cf`, `a90a2d9` | `5951f0b`, `9ce89d3`, `fa0311d`, `db6937a`, `342136d`, `ab5d09b`, `22c8724` |
| `vulkan_model_design` / `codex/modnet-ncnn-conversion-20261111` | `ac93711` | `46f743f` |
| `vulkan_matting_docs` / `codex/vulkan-matting-docs-20261111` | `0dc32c4` | `0219ee6` |
| Root / integration branch | `c77f93f`, `5ee1586`, `8860c73`, `27d5623`, `2e1d6f8`, `b5bc89d`, `13846de`, `cd3c743` | Direct focused commits; no parallel writes to shared files |
| `vulkan_matting_safety_audit` | Read-only interim audit; no commit | Confirmed findings repaired as described below |

The four consumers now execute through the real neutral production manager,
setup leases, and prepared Vulkan island. The exact selected physical GPU and
ncnn-owned logical device/queue are adopted before StudioCast allocations.
The owner retains its provider entry point through every child resource. A
pinned ncnn patch exposes replay ABI 2; plain upstream/older ABI configurations
are rejected. Setup validates GPU-only layers, records once, warms up, and
retires completed upload staging. Frames reuse buffers and recorded commands.
Internal ncnn submission/reset is forbidden. Camera execution retains its
100 ms fence deadline and synchronization limit of 16. The native caps remain
256 MiB and 512 resources. Pending/unsafe completion retains the complete
program/factory and denies lease release until quiescence is proven.

Artifacts are loaded only from bounded sealed memfd snapshots, with SHA-256
verified after sealing; ncnn receives FILE readers over these immutable bytes.
The artifact identity is also bound into the immutable setup lease. Cold explicit
diagnostics perform actual preparation/warmup; ordinary status polling consumes
the daemon cache. Curated numerical attestation additionally requires the exact
converted hashes, blob names, 256 geometry, RGB range, and normalization profile.
Compatible BYO packs do not inherit that attestation. Installer shadow facts
admit all four consumers only with complete common/runtime/artifact evidence;
`recommendation_eligible` remains false.

## Model conversion and publication boundary

Seven converter and 19 transaction tests passed. Root independently reran the
seven converter tests and verified all five candidate payload hashes/sizes.
Two real exports reproduced identical payloads and six numerical results;
candidate local-source and verified offline-cache transactions passed. The clean
pinned AVX2/FMA reference reached maximum alpha error `2.6345252990722656e-5`,
maximum MAE `9.3061352e-8`, and zero threshold-mask disagreements. The SSE-only
reference failed the unchanged `2e-4` maximum-error gate on black
(`0.0002557933`); diagnostics isolated FP32 InstanceNorm accumulation error.
No architecture-independent CPU qualification or relaxed threshold is claimed.

Candidate evidence, numerical fixtures, and publication plan are retained under
`/tmp/studiocast-modnet-conversion-design-20261111/`. The existing ONNX artifact,
model ID `modnet-webnn-256-fp32`, defaults, and other packs are preserved.
The additive publication candidate has exactly four files (25,869,914 bytes):

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `model.ncnn.param` | 12,113 | `4e205032860a9359fa2555c6ef66dece2a8dcd034a1a61466c58d1310af8a1bd` |
| `model.ncnn.bin` | 25,844,000 | `38871660dda4dd324304eb572f43a43ddc5c876c8dff50db2cf09d29a17e3fa2` |
| `model.json` | 1,825 | `3b94bc7bac4ae116659bf7b6bac93fe3a4d90448553661de536eb6c428f13bcc` |
| `LICENSE.txt` | 11,976 | `77b9ad63b2659c31905d1a820403fea0ba0729c7d0530b6e6c6f8b717fd40f81` |

Publication to the existing `10dallasj/studiocast` source requires the pending
explicit approval requested under AGENTS.md. The prepared helper verifies exact
file bindings, hashes, sizes, parent commit, additive delta, and unauthenticated
remote bytes. It has only been dry-run. Active catalog/templates remain unchanged
(seven packs/eight artifacts; ONNX-only MODNet). Only verified published bytes may
be promoted into catalog URLs. Candidate installation is distinct from signed
application release and never modifies installed user packs.

## Native and integrated validation

Pinned ncnn is `24ea07e87f544a3fbc59980b57885a92c75f2d24`, glslang is
`5017667d5a5300dff6aab57db1931a495ff9a10c`, replay patch SHA-256 is
`2028ae171020ac18fe11f0e99fdc228de5887311a3be8390dcf89f2144cc63f0`.
The network-free build helper reproduces the exact tracked SDK trees and retains
licenses/receipt in the isolated prefix
`/tmp/studiocast-ncnn-native-prefix-20261111`.

Available hardware is NVIDIA Quadro P1000, driver 535.309.01, 4 GiB. Required
native ABI-2 tests passed, including six real MODNet golden cases, changing toy
outputs across three frames, allocation reuse after warmup, CPU-layer rejection,
low-budget rejection before allocation, provider/initial-recording negatives,
forbidden internal-submit/reset negatives, and timeout quarantine. Worst maximum
error was `0.000112742`, worst MAE `7.06813e-7`, all IoU 1; unchanged gates are
max `2e-4`, MAE `2e-5`, IoU at least `.999`. Retained native allocations: 232;
complete context peak: 111,266,560 bytes. Root independently reran the final
integrated unit after immutable snapshot loading and verified source-replacement
survival. Logs:

- `/tmp/studiocast-ncnn-native-v2-final-test-20261111.log`
- `/tmp/studiocast-vulkan-matting-final-integrated-native-20261111.log`

The real production camera-manager matrix passed eight cases with three frames
each: blur/remove/replace alone and each with key light, fixed vignette, mirror,
and resize; key light alone and with those final stages. At source `22c8724`,
24 writer commits and 57 successful stages produced exactly 24 uploads and
24 final readbacks, zero intermediate transfers, zero CPU tails, zero fallback,
and zero measured C++ heap allocations during Execute. There were 195
synchronizations, 7–9 per frame (under 16). Cold diagnostics independently
prepared/warmed the exact profile and exposed all four consumers. Debug synthetic
in-memory frame times (24 samples) were mean 25.456 ms, median 24.361 ms,
p95 nearest rank 34.315 ms, max/p99 35.019 ms. Evidence:
`/tmp/studiocast-vulkan-camera-smoke-20261111-seity1rh/{run.json,summary.json,matrix.log}`.
The smoke binary SHA-256 is
`25844225c385675ba660175d5b4a7ab7f55b59c23d5c6bee766b42871b65c87d`.

This is bounded P1000 evidence with 64x48 generated input and 32x24 output. No
AMD/Intel GPU, Khronos validation layers, forced no-push execution, real camera,
V4L2 writer, portrait visual quality, optimized-build latency, or full vendor
family qualification is available/claimed. These remain deployment validation
items under the user's explicit hardware-coverage decision.

The production-on Debug build passed all 138 build steps. Six focused disabled
backend contracts passed; installer backend tests passed 54/54 (21.145 s),
converter 7/7, packaging 9/9, and optional hardware-workflow tests 5/5. A separate
opt-in CI job requires a preprovisioned patched prefix, exact model/root IDs,
replacement image, runner labels, and actual manager success; skip 77 is a
failure in that required job. Remote GPU CI has not been run. Production-on CTest passed 88 total / 82 executed / six expected skips / zero
failures (31.69 s); optional-backend-off build passed 159 steps and CTest passed
81 total / 76 executed / five expected skips / zero failures (22.88 s). Both
suites used temporary HOME/XDG roots and direct CMake package binaries (the
Python entry-point wrapper cannot import its user-site package under temporary
HOME). The required real-model tests and camera matrix above passed separately;
opt-in skips are not hardware qualification. Logs are
`/tmp/studiocast-vulkan-matting-final-{production,disabled}-ctest-20261111.log`.
The documentation validator passed all 41 files, capability audit passed, and
`git diff --check` passed.

## Critique and bounded repairs

The read-only interim audit confirmed hidden ncnn automatic submissions,
ignored initial command-begin errors, and an artifact replacement race that
loaded different weights under an old typed identity. Replay ABI 2 fixes the
first two with negative native tests; commit `aaa0113` fixes the last with sealed
checksum-verified snapshots and a source-replacement regression. Initial graph
recording exhaustion was repaired with lifetime-aware pool reuse without raising
caps. The first camera run's malformed contract, opaque SDK greenscreen mode,
and over-budget synchronization/deadline mismatches were implementation failures,
subsequently repaired and covered by the passing final eight-case matrix.
No such failure is waived as unavailable hardware.

Scope extension: baseline CI run `43031290469` exposed three bounded failures.
Root added `<algorithm>` to the Open CUDA factory, used ORT's stable C API for
free-dimension overrides (actual pinned ORT 1.17.3 syntax check passed), and staged
`typed_video_status.py` in packaging. Packaging fixtures now import the actual
CMake-selected module list. Original evidence:
`/tmp/studiocast-vulkan-task-pr-ci-failures-20261111.log`.
A fresh final independent critique follows integrated broad tests; any valid
blocker/high finding requires focused remediation, revalidation, and fresh review.

## Remaining delivery boundaries

- Complete fresh final critique and record its exact result here.
- Await explicit model publication decision; then verify remote bytes before
  catalog/template promotion and rerun affected model/installer/packaging checks.
- Collect unavailable hardware, real-camera/visual-quality, and optimized latency
  feedback during deployment; keep fail-closed runtime checks intact.
- Git push/application release/deployment are separate actions. This task has
  made no new remote Git push, application release, installed-pack change, host
  package/service/driver/kernel change, or model publication. The preceding
  completed scope was pushed to `origin/v0.2.9` at baseline `51b9612`.
