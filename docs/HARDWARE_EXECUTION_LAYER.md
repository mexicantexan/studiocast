# StudioCast Hardware Execution Layer

- Status: Active contract with a staged production migration
- Applies To: StudioCast v0.2.9 and the current source tree
- Audience: Video-pipeline, backend, daemon, installer, and test maintainers
- Last Verified: 2026-11-11

The StudioCast Hardware Execution Layer (HAL) separates canonical effect
planning from backend-specific resource ownership and execution. It is an
in-tree static extension point for the existing CPU, Maxine, Open CUDA/Open
Video, and Open Vulkan implementations. It is not OpenCV's macro HAL and does
not imply that CUDA, Maxine, ONNX Runtime, Vulkan, or another API can share
resources merely because they use the same physical GPU.

The production camera path now has one neutral plan owner. This is an
architecture milestone, not a parity-completion claim. A qualified resident
route owns its live execution island; retained compatibility execution is a
typed setup-source result with exact provider and execution-domain evidence.
Compatibility is bounded to preserving existing behavior while a family is
migrated and does not relax strict backend selection, production capability
evidence, transfer accounting, or the removal gates in
[HAL_PARITY_COMPLETION.md](HAL_PARITY_COMPLETION.md).

## Two separate layers

### CPU primitive-kernel HAL

`src/core/video/cpu_primitive_hal.h` defines a synchronous host-memory function
table. `GetCpuPrimitiveTable()` constructs one immutable process-lifetime table,
selecting existing scalar, SSSE3, AVX2, and libyuv conversion implementations
once. The current table exposes:

- YUYV-to-RGB24, RGB24-to-YUYV, and RGB24/BGR24 conversion;
- in-place RGB24 mirror; and
- prepared scalar bilinear RGB24 resize, including setup-time scratch sizing.

The function table reports primitive-operation capability only. Its semantic
effect provider is always invalid. A mirror, resize, blur, or composite
primitive therefore cannot make a product effect available, and the CPU lane
for all nine canonical product effects remains fail-closed. New CPU libraries
may extend this table only with parity tests and setup-time scratch planning;
they must not create an implicit effect-level CPU fallback.

### Resident accelerator HAL

`src/core/hal/` defines the neutral prepared-program contract.
`studiocast_hal_contracts` owns the versioned identifiers, descriptors,
resources, execution-domain identity, lifecycle, completion, failure, and
telemetry types. `studiocast_hal_runtime` owns the deterministic static
registry, semantic resolver, shadow comparison, and domain planner.
`studiocast_video_execution_runtime` adds the fixed-capacity video facade and
typed status contracts.

These targets contain no optional SDK headers, vendor libraries, backend
feature definitions, provider discovery, or model discovery. Backend targets
translate their native resources into the neutral contract.

## Contract identity and resource rules

Stable numeric IDs distinguish:

- backend families;
- primitive operations;
- canonical semantic effects and their providers;
- status and blocker reasons; and
- additive typed capability evidence.

Image and tensor descriptors carry dimensions, planes, format and layout,
channel order, color space/range, alpha meaning, strides, alignment,
orientation, host coherence, mutability, and aliasing/in-place permissions.
Resource handles also identify allocator ownership, resource and context
generation, memory domain, lifetime, synchronization state, size, alignment,
destruction callback, and reserved budget.

An `ExecutionDomainId` is exact. It binds:

- backend and API family;
- stable physical-device identity;
- logical-context generation;
- allocator and memory-handle family;
- queue or stream identity;
- ownership model;
- synchronization protocol; and
- owning adapter instance.

Equal device names, indices, PCI devices, UUIDs, CUDA contexts, or Vulkan
devices are not by themselves proof of one execution domain. In v1, one adapter
owns each resident island. A stage from another adapter either creates an
explicit measured boundary in a planning-only split or makes the single-island
live facade reject the plan. Cross-adapter residency remains forbidden until
both adapters supply tested import/export, ownership, lifetime, and
synchronization evidence.

## Prepared-program lifecycle

A prepared program fixes the canonical stage order and exact per-stage
backend/provider,
input/output descriptors, model-artifact identity, execution domain,
configuration/geometry/model/provider/device/context generations, temporal
rules, required/optional behavior, circuit-breaker policy, declared CPU tails,
resource budget, and setup rollback policy.

Preparation owns session, kernel, buffer, binding, and scratch construction.
It happens outside the frame loop. Candidate setup must roll back its resources
on failure; retention of an old program is allowed only when the configured
policy and identity remain compatible. Changes to the canonical plan,
configuration, geometry, artifact/provider, device, context, or execution
domain invalidate the program. Setup checks persistent, scratch, upload-staging,
readback-staging, resource-count, and in-flight-slot limits against the caller's
reservation before native acquisition; it does not enlarge that reservation.

V1 execution is synchronous but completion-shaped. `Execute` returns a
completion token that may already be complete. The caller resolves the output
and then reports writer success through `CommitOutput`. The runtime does not
claim an active backend or committed stage until the effect succeeded and that
output was successfully written. Failed writer publication leaves selected and
readiness evidence intact but cannot create active attribution.

Optional stages may fail open only according to their declared policy while
preserving valid input or the last valid resident frame. Required failures,
foreign/stale resources, generation or domain mismatches, exhausted budgets,
device loss, and synchronization failures remain typed and fail closed.
Circuit breakers, temporal resets, frame-drop rules, shutdown, and quarantine
are program state, not frame-time capability probes. Failed submitted CUDA frames
receive a binding-validated retirement fence and freeze further normal enqueue
and timing operations. A failed or device-lost fence does not prove quiescence;
the owner retains the island until shutdown has explicit release proof.

The current interface intentionally does not schedule true asynchronous work.
Before asynchronous execution is added, the contract needs bounded in-flight
slots, resource-reuse protection, cancellation, stale-generation handling,
device-loss quarantine, stop-aware waits, shutdown behavior, and explicit
temporal dependencies.

## Registry, resolver, and planner

The static registry consumes cached setup-time capability evidence. Registration
and resolution perform no device/provider probing, model scanning, manifest
parsing, session construction, or filesystem work. Selection is deterministic:
priority ties are resolved with stable backend, provider, and adapter
identities, never insertion order.

Primitive and semantic capability records are distinct. A semantic capability
is selectable only when it has a live stage, production eligibility, exact
evidence, and a ready or explicitly degraded status. Strict requests never
substitute another backend or provider.

The shadow resolver compares its result with the retained live resolver without
changing the live decision. Discrepancies are fixed-capacity typed records
available to tests and setup diagnostics. A shadow result is not a frame-time
fallback.

The contract, resolver, runtime, and owner tests use fake resident adapters.
Those fakes exercise malformed descriptors, foreign adapter/resource handles,
stale programs, wrong generations and domains, resource exhaustion, device
loss, reconfiguration, temporal discontinuity, shutdown, transfer counts,
optional-stage preservation, and committed-output attribution without probing
real hardware.

## Current production adapters and migration gates

Prepared HAL completion proceeds by existing-functionality family. Open CUDA
parity remains the first completion target. Local production composition is
integrated at `c43717e`: denoise, Eye Contact and its YuNet/landmark dependencies,
background modes, key light, Auto Frame, tracked or fixed vignette, mirror, and
final geometry can execute in one exact CUDA island in canonical order. Model
and provider identities, optional-stage isolation, temporal history, and
metadata crossings stay explicit. This integration and the bounded smoke below
do not by themselves satisfy the family gate in
[HAL_PARITY_COMPLETION.md](HAL_PARITY_COMPLETION.md).

### Evidence and remaining gates

| Evidence layer | Recorded result | Qualification boundary / remaining gate |
| --- | --- | --- |
| Local source checkpoint | `b204b0aff69f7872a6687ab3470b4e513b48fb14`: canonical CUDA composition, exact dependency binding, strict selection, reset invalidation, six-axis reservation, failed-frame fences, and exact unsubmitted-binding retirement | This is the tested source identity; subsequent documentation commits do not change runtime source. Family and release gates remain separate |
| CUDA timing and lifecycle repair | `9e41f5c` / `2e70c1b` preserve final-stage timing order and zero disabled/unrun durations; actual executor fixture covers all four final-stage masks, next frame, required failure, and optional cooldown/retry | Hermetic native-body evidence is distinct from the required hardware matrix |
| Vulkan shutdown repair | `f64a00d` / `2623b88` retain the context until its submission guard unlocks; two-case lifecycle fixture exposes the old unique-owner failure, preserves retained-owner behavior, and passes against the fix | Native initialization/shutdown ASan reproducer proved the prior use-after-free; no driver/provider change was made |
| Native CUDA matrix | All sixteen required cases passed at clean `b204b0a` (command exit 0), with three commits each; Debug camera policy 100 ms / 1 ms / 100 polls, 64x48 → 32x24, fifteen strict cases plus automatic six-stage resident composition without mirror | Generated input has no detected face; in-memory writer only. No real camera/V4L2, real-face correction, quality, optimized latency, AMD, or Intel qualification |
| Final matrix accounting | 48 committed frames, 48 uploads, 48 final RGB readbacks, 188 metadata readbacks, 71 synchronizations, 83 CPU-tail boundaries, zero fallback. Strict seven-stage and automatic six-stage cases each report three uploads/final RGB/commits, 39 metadata readbacks, nine synchronizations, twelve tails | Exact source, inputs, model identities, and per-case logs are retained with the ledger; metadata and CPU tracking/planning remain measured boundaries |
| Model identity | Corrected FastDVDnet, current 640x640 YuNet metadata, dlib, and real FP32 left/right eye models in an isolated root; exact identities recorded with final hardware evidence | Eye models and dependencies are prepared, but no-face input does not demonstrate gaze correction; installed packs remain untouched |
| Native Vulkan and probe | Final `b204b0a` required Vulkan final integration and combined probe exited 0; the corrected topic kernel/final/lifetime selection passed three of three and required probe five of five | Only mirror/fixed-center vignette retain Vulkan production eligibility; no other effect/vendor promotion |
| Integrated regression and critique | Both final Debug builds passed; enabled CTest 87 total / 81 executed / six skips / zero failures (108.12 s); disabled 80 / 75 / five / zero (95.90 s). Disabled probe also exited 0. Fresh source critique found no concrete Blocker/Critical/High and independently passed six exporter tests, four workflow tests, and the lifetime binary; final documentation consistency verdict remains pending | Source critique is complete; the final docs diff still needs consistency review. Explicit opt-in skips are not native qualification |
| Sanitizer diagnostic | Full ASan hermetic lifecycle fixture passed with leak checks. Native helper no longer reports the use-after-free, but exits 1 with LeakSanitizer reporting 12,080 bytes in 81 allocations in unloaded/unknown modules and a child-symbolizer `LD_PRELOAD` issue | No clean full-native-sanitizer claim and no source/driver attribution for these remaining diagnostics without causal evidence |
| Required hardware CI | `d994679` enforces sixteen CUDA topologies with dlib, exact preinstalled model IDs, and a replacement image, plus required per-vendor Vulkan final execution | Local command execution is distinct from remote CI execution; automatic host mirror still needs camera-level evidence |
| Model publication and release | Three-file FastDVDnet publication and local catalog update are authorized, but publication is authentication-blocked (HTTP 401 / no active login); no remote files, active catalog/templates, or installed packs changed | The staged promotion proposal is unaccepted; catalog promotion, family/claimed vendor qualification, and signed clean-HEAD release remain gates |

The vignette stage and resident alpha-capacity correction were integrated at
`0adc8ac`. Strict CUDA MODNet uses setup-time specialization of free dimensions.
The corrected capacity check respects a one-row pitched tensor allocation
rather than multiplying its complete allocation pitch by model height again.

The installed YuNet metadata was stale (320x320 for the pinned 640x640 graph);
the repository template is already 640x640. Tests use current metadata in an
isolated root; validation does not repair an installed pack implicitly.

The approved FastDVDnet decision is repair/re-export with the existing model
ID and defaults. A self-contained corrected candidate and reproducibility
results exist, while the shipped catalog retains the old published identity
and its graph-validation failure. Authorized publication remains blocked on
authentication. Catalog promotion,
required qualification, and the signed clean-HEAD release chain remain
separate gates; installed packs are untouched. See
[INSTALLER_MODEL_CATALOG.md](INSTALLER_MODEL_CATALOG.md).

The camera pipeline uses one `ProductionVideoBackendManager` backed by typed
setup leases. Backend factories own native sessions, contexts, streams/queues,
buffers, and adapters; the pipeline sees a canonical setup request, host-frame
bindings, completion tokens, transfer counters, and snapshots rather than
backend SDK objects. Candidate compositions replace a live composition only
after registration and preparation succeed, so failed reconfiguration retains
the prior valid plan.

| Route | Current prepared live eligibility | Exact limitation |
| --- | --- | --- |
| CPU primitives | Conversion, mirror, and prepared bilinear resize only | No semantic product-effect capability or silent CPU effect fallback is inferred |
| Maxine resident | A qualified contiguous Maxine island with tight RGB24 host input/output and one exact RGB/BGR output-conversion tail | Runtime and model evidence remain required; unsupported or unavailable Maxine work fails closed instead of changing GPU backend |
| Open Vulkan final | Explicit strict Vulkan; only production-qualified fixed-center vignette and/or mirror; tight RGB24 transport; zero CPU tails | Matting, tracked-center vignette, Auto Frame, eye contact, denoise, key light, and background modes remain blocked or non-production exactly as documented |
| Open CUDA resident | Strict CUDA and automatic CUDA selection use exact model/provider evidence; canonical effect composition and geometry are integrated | Bounded generated-input/in-memory-writer evidence is recorded above; final integrated evidence and critique are recorded above; family/release gates remain required |
| Open Video compatibility | Native-to-host denoise/Eye Contact fallback is setup-only under unconstrained automatic selection, after permitted availability failure and proven safe cleanup | Strict CUDA or incompatible provider intent, mixed CUDA semantics, pending cleanup, and quarantine never take this fallback; explicit typed CPU compatibility remains separate and is not CUDA attribution |

The setup request declares a setup-only terminal
`backend_resolved_output` marker. It is never executable. The selected setup
source must bind it exactly to zero tails for Vulkan and host compatibility, an
output-format conversion for Maxine, or a writer-copy/provider tail for Open
CUDA. The manager accepts only that same terminal-boundary refinement; normal
plan and prepared-program validation reject an unresolved marker.

The Maxine owner generalizes the existing resident-section executor and
preserves its shared upload, matte reuse, bounded resources, final readback,
failure handling, and compatibility counters. Its canonical host output is
RGB24; SDK-local BGR remains internal and the startup-selected CPU conversion
is measured after the single final readback. The Open CUDA owner contains CUDA
driver/stream ownership, prepared preprocessing and postprocessing kernels,
ONNX Runtime CUDA IoBinding, validated artifact identity, reusable device
resources, bounded completion waiting, and retryable shutdown. Replacement
images are decoded, uploaded, and resized during setup. Model discovery,
manifest parsing, artifact validation, provider selection, session creation,
and resource sizing happen at setup.

An Open CUDA island may contain exact per-stage providers under one unchanged
execution domain. ONNX Runtime model stages use
`open_video_onnxruntime_cuda`; model-free native kernels use
`open_cuda_production_kernels`. A mixed island leaves its aggregate provider
invalid and retains exact provider identity on every stage. This does not relax
strict provider requests: an explicit provider constraint may select only
stages from that provider. The native provider ID is a contract prerequisite,
not evidence that a native stage is already integrated or production-ready.

Setup-lease shutdown is typed and retryable. The composition retires its
coordinator and registry before requesting native shutdown, and it marks the
lease stopped only after the owner proves quiescence and permission to release
its driver context. Open CUDA performs one bounded native shutdown attempt per
explicit lease retry and retains the complete execution island while release is
pending; immediate owners such as the current Maxine bootstrap return an
already-complete result.

The Vulkan adapter may advertise product capability only for mirror and
fixed-center vignette, with the exact evidence described in
[INSTALLER_VULKAN_CAPABILITIES.md](INSTALLER_VULKAN_CAPABILITIES.md). Vulkan
matting, tracked-center vignette, Auto Frame, eye contact, video denoise,
virtual key light, and virtual-background modes retain their existing
production, degraded, experimental, diagnostics-only, or blocked states.
Creating a generic Vulkan adapter does not promote them.

`production_video_setup_request.*` owns the shared
`BuildProductionVideoBackendSetupRequest` builder used by the camera pipeline
and hardware smoke executable. It fixes canonical order, parameters, geometry,
selection intent, and temporal requirements during setup. Greenscreen temporal
history is a Maxine-only refinement after backend selection; its default must
not make ordinary CUDA matting temporal. Denoise, Auto Frame, and Eye Contact
retain their own declared history and discontinuity requirements. The camera
pipeline consumes the manager's typed setup result. The static
backend-composition target owns the optional Maxine, Open CUDA/Open Video, and
Vulkan dependencies privately;
`studiocast_video_runtime` exposes neutral contracts, execution runtime, video
I/O, and support dependencies and receives no optional backend feature
definitions. Any retained compatibility execution is still part of the live
behavioral surface and must remain visible in status and parity accounting.
Moving selection, construction, probing, or allocation behind the manager does
not by itself prove that every former route is prepared-HAL owned.

## Transfers, CPU tails, and telemetry

One eligible full-domain GPU island reports:

```text
host input -> one upload -> contiguous resident stages
           -> one final readback -> exact declared writer/format CPU tail
```

This full-frame shape does not imply zero metadata crossings for analysis
consumers. The CUDA analysis contract separately bounds face metadata
readbacks (up to twelve per frame) and matte-bounds readback (up to one), plus
their synchronization and CPU detection/tracking/crop-plan tails. Image and
inference tensors remain resident; metadata work must be counted and reported
rather than described as a transfer-free path.

Intermediate uploads/readbacks, synchronization points, and CPU-tail
boundaries are explicit counters. An adapter cannot hide a host round trip
inside provider work or infer interoperation across domains. CPU-only
passthrough remains CPU-only with zero GPU transfers.

Exact CUDA and Vulkan selection can include mirror in their resident island.
Automatic selection and Maxine retain the established host mirror finalizer;
this is an explicit limitation, even when automatic selection resolves to
CUDA. The host primitive mirror is a typed CPU-tail activity record. It is
published only after the same frame is committed by the writer. A dropped,
unwritten, disabled, or fail-open frame cannot leave stale active mirror
attribution, and this primitive record never becomes semantic-provider
capability evidence.

Typed telemetry includes requested, selected, and active attribution; setup and
execution attempts/successes; committed output frames; fallback and optional
fail-open counts; device loss and temporal resets; transfer/synchronization
counters; per-stage timing; and frame p95/p99 latency state. Status serialization
uses cached snapshots and never adds GPU synchronization, provider probing, or
JSON work to the frame loop.

`studiocast-prepared-open-cuda-facade-bench` measures the facade and contract
path with a fake host-copy adapter. Its output is explicitly
`synthetic_fake` and reports average, p50, p95, p99, transfer shape, and
committed attribution. It is useful regression evidence for facade overhead
and contract accounting; it is not CUDA, ONNX Runtime, model, driver, or
hardware-qualification evidence.

## Daemon status and compatibility

The daemon remains authoritative for requested, selected, available, active,
degraded, blocked, and fallback state. `GET_STATUS` consumes the coherent typed
manager selection and writer-commit snapshot as its live authority. Legacy raw
backend diagnostics remain compatibility and installer evidence; they cannot
override a current typed result. Failed, pending, stopped, or quarantined
runtime state invalidates stale committed attribution. The pipeline also caches
a fixed-capacity multi-island snapshot with:

- all nine stable product effects and all capability lanes;
- every current prepared island and its exact execution domain;
- readiness, generations, stage resolution, failure policy, and telemetry;
- one monotonic writer-commit identity; and
- aggregate transfers summed once across current islands.

`GET_VIDEO_HAL_STATUS_V1` returns this validated standalone JSON object only
when its fixed-capacity authority coverage is complete. The snapshot carries a
prepared-route effect mask and a compatibility-uncovered effect mask. A
requested effect without a selected current prepared island is uncovered:
the retained compatibility pipeline may still be executing it, so the command
returns an error instead of serializing a prepared-only blocked, unavailable,
or active claim as daemon-wide truth. A malformed, stale, mixed-generation,
foreign-domain, strict-substitution, mixed-writer-commit, or incomplete
authority snapshot likewise returns an error rather than partial success-looking
facts. A stopped or no-GPU pipeline with no requested effects publishes a valid
fail-closed idle/disabled or passthrough snapshot.

The existing `GET_STATUS` bytes and canonical effect/parameter IDs are
unchanged by this versioned command. The GUI remains a controller and continues
to use daemon-owned state. Consumers must not combine setup selection or
cumulative counters into active attribution; only a current successful writer
commit is active evidence.

The installer contains a strict parser for a
`typed_multi_island_video_status` diagnostics value, but it is shadow-only and
sets `recommendation_eligible=false`; it also rejects absent, malformed, or
incomplete authority coverage. Existing exact per-effect installer facts remain
authoritative. Connecting the new daemon command to installer recommendation is
a separate parity-reviewed migration.

## Extension and security boundary

New in-tree adapters must:

- keep optional SDK types and feature definitions out of neutral targets;
- publish exact semantic evidence separately from primitive availability;
- prepare all sessions, models, kernels, buffers, and scratch outside the frame
  loop;
- use one exact execution domain per v1 island;
- reject foreign/stale handles, generations, contexts, streams/queues, and
  adapters;
- declare CPU tails and all transfer boundaries;
- preserve strict selection and optional-stage failure semantics; and
- pass the common contract/conformance tests with optional backends disabled as
  well as their backend-specific tests.

The neutral contract and runtime tests must remain buildable when Open CUDA and
Open Vulkan are independently or jointly disabled. Enabling a backend adds its
conformance and hardware tests but must not change neutral headers or compile
definitions. Hardware-required tests and benchmarks must fail or skip
explicitly according to their test contract; fake or synthetic results never
substitute for NVIDIA, AMD, Intel, multi-GPU, or Arm qualification.

ROCm/HIP, KleidiCV, FastCV, Qualcomm, and other vendor SDK integrations are not
implemented. The static contracts are their future extension point; they do
not constitute capability claims.

There is no dynamic plugin ABI or loader. Arbitrary path-based native loading
is intentionally out of scope. A future loader must first define a versioned
C-compatible POD ABI, struct-size/version negotiation, opaque ownership and
destruction, alignment, exception and threading rules, unload safety, trusted
discovery roots, mode/owner/symlink validation, TOCTOU-resistant loading,
package or signature provenance, dependencies, and crash policy. Until those
requirements and adversarial tests exist, dynamic plugins remain an unresolved
security boundary and must not be enabled through a weaker loader.
