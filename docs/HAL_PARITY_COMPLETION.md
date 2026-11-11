# HAL parity completion and qualification

- Status: Active migration and release contract
- Applies To: Prepared HAL ownership, compatibility-route removal, and backend
  capability claims
- Audience: Video runtime, backend, daemon, test, and release maintainers
- Last Verified: 2026-11-11

The Hardware Execution Layer is StudioCast's intended video execution
architecture. Neutral contracts, planning, and ownership are necessary
milestones, but none of them alone proves parity with the existing production
pipeline. Parity means that the prepared route preserves the established
selection, effect, failure, output, status, and rollback behavior for a bounded
family on supported hardware.

## Migration order

Complete existing-functionality Open CUDA parity first. This includes strict
CUDA requests, `auto` selection and fallback, each already supported effect,
and supported combinations and dependencies such as Auto Frame. Integration
must be verified against every applicable case below. Uncovered behavior must
remain on a bounded compatibility route or remain explicitly
blocked; neutral ownership must not be used to claim that it is fixed. Any
regression of behavior that worked before the migration must be restored for
the CUDA parity milestone. A compatibility route is temporary preservation,
not an acceptable permanent parity result.

After CUDA parity, expand production Vulkan in bounded phases. Each phase must
name its effects, adapter and execution-domain contract, models and artifacts,
hardware matrix, transfer budget, failure behavior, and rollback boundary.
Passing one phase does not promote other Vulkan effects.

Vulkan matting remains blocked until all of the following exist and are
reviewed together:

- shared-device GPU-only inference using StudioCast's selected physical and
  logical device, queue, buffers, ownership, and synchronization protocol;
- converted schema-v2 ncnn Vulkan parameter/bin artifacts with pinned,
  verified checksums and the required tensor/converter metadata;
- no CPU layers, host inference input/output, intermediate readback, or hidden
  CPU tail; and
- real-hardware execution evidence for every device family claimed by the
  release.

## Family parity gate

Before removing a working route or describing a family as prepared-HAL
complete, record evidence that:

- canonical engine/compute preferences resolve identically, including strict
  requests, `auto`, fallback, unavailable, and degraded outcomes;
- every existing single effect and supported combination has equivalent stage
  order, shared dependencies, parameter behavior, and output semantics;
- model selection, artifact identity, missing/invalid-model behavior, and
  provider requirements remain deterministic and daemon-owned;
- optional failure, circuit breaking, device loss, reconfiguration, frame
  drop, stop, and retry behavior preserve valid output and bounded latency;
- writer-commit attribution, per-effect status, blocker reasons, fallback
  reasons, transfers, CPU tails, and active backend facts remain truthful;
- focused regression and integration tests cover the family, and applicable
  real-hardware tests pass on the supported matrix; and
- a reviewed rollback keeps or restores the previous working route without
  changing persisted canonical configuration.

Compatibility may remain only where it preserves existing behavior while a
known parity gap is open. Its effect coverage must be explicit in typed status,
it must obey strict-selection and GPU-only rules, and it must not be counted as
prepared-HAL capability or completion evidence.

## Route-removal gate

A compatibility route or legacy direct dependency may be eliminated only when
the family parity evidence above is reviewed and the replacement is enabled in
the same configurations and supported hardware contexts as the working route.
The change must include a tested rollback boundary. Source ownership or CMake
dependency cleanup is insufficient evidence by itself.

## Qualification and release gate

A production capability or release note may claim only the exact effects and
hardware covered by reviewed evidence. Synthetic adapters, loader/device
discovery, compilation, unit conformance, and a neutral manager cutover do not
substitute for model-backed real-hardware execution. Release qualification
must retain the existing policies in [PIPELINE_POLICY.md](PIPELINE_POLICY.md),
[LATENCY_POLICY.md](LATENCY_POLICY.md), [MODEL_POLICY.md](MODEL_POLICY.md), and
[MANUAL_TESTING.md](MANUAL_TESTING.md), plus backend-specific installer
evidence such as
[INSTALLER_VULKAN_CAPABILITIES.md](INSTALLER_VULKAN_CAPABILITIES.md).

Local family parity completion does not itself qualify a production capability
or release. Until every applicable release gate is satisfied, documentation
and PR text must name the remaining compatibility or blocked routes and avoid
claiming production readiness or release completion. Until the family parity
gate is satisfied, they must also avoid claiming end-to-end parity.

## Local remediation record (2026-11-11)

Branch: `v0.2.9`. Starting baseline:
`99a7fbcb28f0f00e049c660765b192ec1f774234`. Integrated source checkpoint:
`b204b0aff69f7872a6687ab3470b4e513b48fb14`.

The local change addresses canonical CUDA setup and temporal defaults,
production composition, tracked vignette, reproducible FastDVDnet repair,
required hardware CI, and documentation reconciliation. It retains the neutral
camera manager: setup selects exact models/providers and prepares one resident
island; execution reuses it, counts metadata/CPU boundaries, and publishes active
attribution only after the writer commits.

### Accepted implementation and ownership

All accepted changes below are integrated locally. Topic branches are retained
for traceability; no Git push, remote PR, release, or user-installed-model
transaction was performed during the runtime implementation. The resumed
publication and catalog task is recorded separately below.

| Agent / topic branch | Accepted commits | Integration |
| --- | --- | --- |
| `qualification_ci` / `codex/qualification-ci-20261110` | `2da87f0` required production hardware CI | `cd86a34`; pipeline failure propagation `9fbb8f3` |
| `model_disposition` / `codex/fastdvdnet-repair-20261110` | `9384cd1` pinned offline exporter, graph/equivalence validation, repair decision | `dcb0b5c` |
| `cuda_contract` / `codex/camera-contract-20261110` | `9770ad4` shared camera/smoke setup, temporal and tracked-center semantics | `8386905` |
| `cuda_factory_owner` / `codex/cuda-factory-owner-20261110` | `cbb7569`, `81de5f7`, `65829c0`, `d5d0ff3`, `3f242ad` dependency, metadata, and temporal contracts; `9cc5d3d`, `10c4112` canonical owner/factory; `91d8d2c`, `af2d1db`, `d24abb7` safety fixes | `dec3a30`, `56d89ef`, `6f0d543`, `4991f25`, `28c124e` |
| `cuda_production_integration` / `codex/cuda-production-parity-20261110` | `36341ee`, `05d607a` exact selected analysis; `d2ddf75`, `6ccf5a4`, `8f49902` composition, isolation, setup fallback/budgets; `f248151`, `3cc5f7d` artifact contracts; `faf0a1d`, `b5e03c9`, `feb8426` failure fences; `bd56e19`, `7e35516`, `0c93225` camera-equivalent smoke | `7b2da70`, `c43717e` |
| `cuda_contract` / `codex/eye-optional-tests-20261110` | `d98b642` optional-dlib fixture boundaries | Cherry-picked as `956fe34`; test-only linkage `98fa7c9` |
| `qualification_ci` / `codex/noncuda-metadata-20261110` | `e645712` fail-closed Maxine/Vulkan metadata allowances | `8682c73` |
| `qualification_docs` / `codex/parity-docs-20261110` | `03e6a43`, `c235945` bounded evidence and route documentation | `343e821`, `cc5b9f8` |
| `cuda_ci_composition` / `codex/cuda-ci-composition-20261110` | `cb1808a` sixteen required CUDA cases, exact preinstalled prerequisites | `d994679`; inherited-environment fixture correction `d850c0b` |
| `probe_selftest_repair` / `codex/probe-selftest-repair-20261110` | `a22c1c0` diagnostic target's private backend feature contracts | `56fdb20` |
| `cuda_factory_owner` / `codex/unsubmitted-frame-retirement-20261110` | `e73a8c6` exact unsubmitted discard through every CUDA layer; `da61119` explicit matte-coordinate conversions | `7810477` |
| `cuda_production_integration` / `codex/disabled-stage-timing-20261111` | `9e41f5c` canonical final timing slots, zero disabled/unrun duration, native-body regression fixture | `2e70c1b` |
| `probe_selftest_repair` / `codex/probe-native-cleanup-20261111`, completed by root | `f64a00d` Vulkan shutdown retains context until guard destruction; friend-only unique/shared-owner regression fixture | `2623b88` |
| `model_disposition` / `codex/fastdvdnet-promotion-20261111` | `337f3d8` independent idempotent full-license exporter repair only | `b204b0a`; old proposal was not accepted; resumed promotion is recorded below |
| `model_disposition` / `codex/fastdvdnet-promotion-final-20261111` | `cf68618` verified published sigma15 catalog/template/fixture promotion | `d7f7f5d`; resumed publication and validation below |
| Root integration | `8d4393c` Python test registration; `1060a6f` integration assertions follow shared setup and preserve valid bindings after forged requests; `211da52` changed-line formatting | Direct local commits |

### Critique and remediation

- Fresh `first_wave_critique` found automatic reset failure bypassing session
  invalidation. `af2d1db` clears stale attribution and requires safe recovery.
- Fresh `reset_remediation_critique` found legitimate frame-order rejections
  retaining bindings. `d24abb7` fixes neutral retirement. Root's real-layer
  reproducer then exposed retained CUDA adapter/executor bindings; `e73a8c6`
  adds exact, never-submitted discard and conservative reconfiguration when an
  adapter cannot prove release.
- Fresh `integrated_final_critique` found disabled mirror timing preceding a
  deferred vignette, plus nonzero disabled duration. `9e41f5c` preserves helper
  ordering and tests actual production execution with fake CUDA functions,
  including required failure and optional failure/cooldown/retry.
- Fresh `timing_remediation_final_critique` found no further confirmed source
  blocker/Critical/High at `2e70c1b`, but withheld approval after the required
  native probe aborted. Isolation and an eight-line native ASan
  initialize/shutdown reproducer proved a Vulkan shutdown use-after-free:
  resetting the last context owner destroyed its mutex before the guard
  unlocked. `f64a00d` moves that owner into condition-local storage so the
  inner guard dies first. The two-case custom-deleter test exposes the old
  unique-owner failure and preserves retained-owner behavior; both cases pass
  against the repaired library. This is a source lifetime fix, with no host
  driver/provider change.
- Fresh `native_cleanup_final_critique` approved the bounded local implementation:
  runtime source `b204b0a` and eight-doc commit `81a8989`, integrated as
  `e619d93`, have no unresolved Blocker/Critical/High finding. The critic
  independently passed six exporter tests, four workflow tests, and the lifetime
  binary, and reviewed the final broad/native evidence and documentation limits.
  This approval does not establish external publication or release completion.

### Final source verification and bounded diagnostics

Final source is `b204b0aff69f7872a6687ab3470b4e513b48fb14`; later docs-only
commits must be recorded separately from tested runtime identity. Both full
Debug builds passed with CUDA/Vulkan/audio/dlib enabled and with all four
disabled. Enabled CTest passed 87 total (81 executed, six explicit opt-in
skips, zero failures; 108.12 seconds). Disabled CTest passed 80 total (75
executed, five skips, zero failures; 95.90 seconds). Disabled probe exited 0
with `SELFTEST OK`. The final required native matrix also passed as below.
The previous `2e70c1b` enabled/disabled CTest results (86/80 total, six/five
explicit skips) remain historical, not final-checkpoint evidence.

At the corrected topic checkpoint, native Vulkan kernel/final/lifetime tests
passed three of three and the required CUDA probe passed five of five.
Integrated hermetic Vulkan lifetime/adapter/factory selection passed three of
three. The full ASan hermetic lifetime fixture passed with leak checks enabled.
The actual native ASan helper no longer reports the use-after-free, but exits 1
with LeakSanitizer reporting 12,080 bytes in 81 allocations in unloaded/unknown
modules, plus a child-symbolizer `LD_PRELOAD` problem. This is an explicit
remaining diagnostic limitation: no clean full-native-sanitizer claim and no
attribution of those allocations to source, SDK, or driver without proof.

The prior native probe abort at `2e70c1b` is superseded by the causal lifetime
repair; disabling Vulkan ICD discovery was diagnostic isolation only and is
not part of accepted verification. The final required Vulkan resident
integration and combined required probe both exited 0 at `b204b0a`. All
sixteen required CUDA cases passed with command exit 0 and three commits per
case: 48 committed frames, 48 uploads, 48 final RGB readbacks, 188 metadata
readbacks, 71 synchronizations, 83 CPU-tail boundaries, and zero fallback.
The strict seven-stage and automatic six-stage resident cases each committed
three frames with three uploads/final RGB readbacks, 39 metadata readbacks,
nine synchronizations, and twelve tails.

Evidence is recorded under
`/tmp/studiocast-final-hardware-20261111/`, including
`vulkan-required-b204b0a.log`, `probe-required-b204b0a.log`,
`cuda-matrix-b204b0a.log`, `qualification-summary.json`, and `qualification/`.
`qualification/source.txt` records exact `b204b0a`; model identities are in
`model-evidence.json`. Earlier sixteen-case CUDA success at `211da52` and the
`0c93225` 64x48 → 96x64 combined run remain historical bounded evidence.

Commands used for the final checkpoint (with temporary HOME/XDG roots and the
preinstalled read-only model root) are:

```bash
cmake --build /tmp/studiocast-parity-enabled-20261110
cmake --build /tmp/studiocast-parity-disabled-debug-20261111
ctest --test-dir /tmp/studiocast-parity-enabled-20261110 --output-on-failure
ctest --test-dir /tmp/studiocast-parity-disabled-debug-20261111 --output-on-failure
STUDIOCAST_REQUIRE_VULKAN_RUNTIME=1 \
  /tmp/studiocast-parity-enabled-20261110/studiocast-vulkan-final-stage-integration-tests
/tmp/studiocast-parity-enabled-20261110/studiocast-probe --self-test --require-cuda-runtime
/tmp/studiocast-parity-disabled-debug-20261111/studiocast-probe --self-test
bash /tmp/studiocast-final-hardware-20261111/cuda-workflow-local.sh
```

The serialized local workflow script executes the sixteen cases from required
hardware CI with exact model/image inputs, three frames at 64x48 → 32x24, and
the camera's 100 ms / 1 ms / 100-poll completion policy. Automatic selection
covers six resident stages without mirror; automatic host mirror still needs
camera-level evidence. All native runs use P1000 / driver 535.309.01,
generated no-face input, an isolated model root, and an in-memory writer.
They do not qualify real camera/V4L2 transport, real-face correction, visual
quality, optimized latency, AMD/Intel, remote CI, or signed release.

### Resumed publication and local catalog promotion

The resumed task starts from clean local baseline
`cef868c3cb9e288e9ccfd97b7e71af8050790294`. Runtime source remains the tested
`b204b0a`; this task changes catalog metadata, templates, fixture pins, and
documentation. The initial HTTP 401/login blocker was resolved when the user
configured normal secure Hugging Face authentication.

Root published exactly the three approved corrected sigma15 files and
verified their remote bytes at
[Hugging Face commit 8ffe968884e31ded74f857a2d0dca19f571ba044](https://huggingface.co/10dallasj/studiocast/commit/8ffe968884e31ded74f857a2d0dca19f571ba044),
whose parent is `944316407869cb72fc7e05ae86898c8738d21c1c`. Existing source
paths were preserved. The publication helper exited 0 and receipt status is
`published_remote_bytes_verified` in
`/tmp/studiocast-fastdvdnet-assets-20261110/review-candidate/publication-receipt.json`.

| Approved file | Verified bytes | Verified SHA-256 |
| --- | ---: | --- |
| `model.onnx` | 9,948,304 | `0895dcad878f44ff4b610e0569bf701c8e23566e675c22a9bb0d59b95dcfc7ac` |
| `model.json` | 1,881 | `9daf66251194aeaab5c67f18331e18a40596b951a8049b4d359d36fe1e97285e` |
| `LICENSE.txt` | 1,671 | `e419e198d08f7d62fbc6bbb74409836289246db14ed28a134f79e1a2ea5afedc` |

The fresh local promotion is topic
`cf686180650468830deb4be5809fe06c213982d4`, integrated as
`d7f7f5db1323ecdbea4430ebd71b49f9d82656b6`. Active catalog `2026-11-11`
contains 140,755,803 total artifact bytes and has SHA-256
`ac7c199bfac83859dd192174bb43e6a5c251647ab8936cebc9639408d4bc3f7e`.
It preserves `fastdvdnet_sigma15`, its default selection, and the seven
packs/eight artifacts. The metadata and full license are byte-identical to the
verified published files; the artifact source selects its corrected versioned
path without an invalid-artifact fallback. The old unaccepted proposal at
`/tmp/studiocast-fastdvdnet-promotion-20261111` is superseded by this fresh
implementation, not accepted separately.

The model agent passed 19 model, 52 backend, nine packaging, and one installer
wizard test. Root post-integration model 19 and packaging nine passed; backend
and wizard final results and the fresh independent critique are pending at this
documentation checkpoint. Root additionally verified all three immutable HTTPS
files without authentication; evidence is
`/tmp/studiocast-fastdvdnet-assets-20261110/review-candidate/publication-public-https-verification.json`.

The integrated installer completed an online HTTPS transaction from immutable
`8ffe9688`, then an offline verified-cache transaction, in temporary destinations
under `/tmp/studiocast-published-fastdvdnet-transaction-2tIjbx/`. Both succeeded
with the exact 9,948,304-byte artifact verified above; JSON evidence is
`/tmp/studiocast-fastdvdnet-published-transaction-online-20261111.json` and
`/tmp/studiocast-fastdvdnet-published-transaction-offline-20261111.json`.
These temporary test activations did not modify user-installed packs.

The artifact hash is unchanged from the corrected isolated model used in the
recorded native matrix, and no runtime code changed; this catalog/docs task does
not add new GPU qualification claims. No user-installed pack was changed.
Old unmarked sigma15 metadata remains preserved with
`curated_ownership_unverified`; an install/repair must retain ownership checks.
Sigma25/sigma50 remain untouched noncatalog BYO templates. No Git push or
signed application release was performed or authorized by model publication.
The source archive/AppImage signing chain and the recorded camera, real-face
quality, optimized latency, other vendor, and native-sanitizer limitations remain
separate gates. See [INSTALLER_MODEL_CATALOG.md](INSTALLER_MODEL_CATALOG.md).
