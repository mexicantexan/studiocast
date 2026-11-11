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
`2e70c1b888b52fde6358b18b505554d70de092d1`.

The local change addresses canonical CUDA setup and temporal defaults,
production composition, tracked vignette, reproducible FastDVDnet repair,
required hardware CI, and documentation reconciliation. It retains the neutral
camera manager: setup selects exact models/providers and prepares one resident
island; execution reuses it, counts metadata/CPU boundaries, and publishes active
attribution only after the writer commits.

### Accepted implementation and ownership

All accepted changes below are integrated locally. Topic branches are retained
for traceability; no push, remote PR, release, or installed-model transaction was
performed.

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
  blocker/Critical/High at `2e70c1b`, but withheld approval after the native probe
  abort below. A new final verification/review gate remains open.

### Verification checkpoint and active blocker

At `2e70c1b`, full Debug builds succeeded with CUDA/Vulkan/audio/dlib enabled
and with all four disabled. Isolated HOME/XDG runs passed enabled CTest
86 total (80 executed, six explicit opt-in skips; 65.49 seconds) and disabled
CTest 80 total (75 executed, five skips; 59.97 seconds), with zero failures.
The enabled six-suite lifecycle selection and disabled native-body executor
test also passed. No skipped test is counted as native qualification.

The sixteen-case P1000 CUDA matrix passed at the earlier `211da52` checkpoint,
using generated input, isolated packs, and an in-memory writer. At `2e70c1b`,
required Vulkan final resident integration passed, but the subsequent required
probe printed `SELFTEST OK` and aborted with `corrupted double-linked list`
(exit 134). The final CUDA repeat correctly stopped before execution.
GDB reproduced the abort in NVIDIA PTX compiler cleanup. Disabling only Vulkan
ICD discovery lets the same required CUDA probe exit successfully; this is
diagnostic isolation, not a gate waiver or an accepted fix.

Bounded incidental scope extension: `probe_selftest_repair` owns diagnosis in
`codex/probe-native-cleanup-20261111`, from `2e70c1b`, of the combined native
probe/loader/resize cleanup. Causal evidence and file ownership must be reviewed
before a source fix; no host driver/provider, installation, service, or system
change is authorized. Evidence is under
`/tmp/studiocast-final-hardware-20261111/`, notably
`probe-required-2e70c1b.log`, `probe-crash-gdb.log`, and
`probe-no-vulkan-isolation.log`. Root must integrate any accepted fix, rerun
affected and broad checks, finish the required native matrix, and obtain fresh
critique before closing this local delivery gate.

FastDVDnet publication/catalog promotion remains separately unapproved. The
corrected artifact and matching metadata/license are reviewable in
`/tmp/studiocast-fastdvdnet-assets-20261110/review-candidate/`; the shipped
catalog and installed packs remain unchanged. Real camera/V4L2 transport,
real-face correction, visual quality, optimized latency, other claimed vendor
families, remote CI execution, and signed release delivery remain distinct
qualification gates. See [HARDWARE_EXECUTION_LAYER.md](HARDWARE_EXECUTION_LAYER.md)
and [INSTALLER_MODEL_CATALOG.md](INSTALLER_MODEL_CATALOG.md).
