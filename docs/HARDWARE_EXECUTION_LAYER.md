# StudioCast Hardware Execution Layer

- Status: Active contract with a staged production migration
- Applies To: StudioCast v0.2.9 and the current source tree
- Audience: Video-pipeline, backend, daemon, installer, and test maintainers
- Last Verified: 2026-07-23

The StudioCast Hardware Execution Layer (HAL) separates canonical effect
planning from backend-specific resource ownership and execution. It is an
in-tree static extension point for the existing CPU, Maxine, Open CUDA/Open
Video, and Open Vulkan implementations. It is not OpenCV's macro HAL and does
not imply that CUDA, Maxine, ONNX Runtime, Vulkan, or another API can share
resources merely because they use the same physical GPU.

The migration is deliberately incremental. A qualified prepared route owns its
live execution island; configurations outside that route continue through the
existing compatibility implementation. Compatibility does not relax strict
backend selection, production capability evidence, or transfer accounting.

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

A prepared program fixes the canonical stage order and exact backend/provider,
input/output descriptors, model-artifact identity, execution domain,
configuration/geometry/model/provider/device/context generations, temporal
rules, required/optional behavior, circuit-breaker policy, declared CPU tails,
resource budget, and setup rollback policy.

Preparation owns session, kernel, buffer, binding, and scratch construction.
It happens outside the frame loop. Candidate setup must roll back its resources
on failure; retention of an old program is allowed only when the configured
policy and identity remain compatible. Changes to the canonical plan,
configuration, geometry, artifact/provider, device, context, or execution
domain invalidate the program.

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
are program state, not frame-time capability probes.

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

The camera pipeline uses backend-hidden owner/facade classes. It sees neutral
prepared requests, host-frame bindings, completion tokens, transfer counters,
and snapshots rather than backend SDK objects. The current live migration is
intentionally narrower than the full compatibility pipeline:

| Route | Current prepared live eligibility | Compatibility behavior |
| --- | --- | --- |
| CPU primitives | Conversion, mirror, and prepared bilinear resize only | Existing callers use the process-lifetime function table; no product-effect capability is inferred |
| Maxine resident | One qualified contiguous non-temporal Maxine island with tight RGB24 input, tight BGR24 output, and an explicit output-format CPU tail | Other Maxine configurations use the retained resident executor; unavailable resolved Maxine work is disabled rather than sent through the old per-effect upload/download path |
| Open Vulkan final | Explicit strict Vulkan; only production-qualified fixed-center vignette and/or mirror; tight RGB24 transport; no other stage in the plan | Ineligible configurations retain the existing exact Vulkan/legacy path and its capability gates |
| Open CUDA/Open Video | Explicit Open CUDA effect engine; strict CUDA and ONNX Runtime CUDA provider; one blur or remove virtual-background stage; same capture/output geometry; tight RGB24; no mixed, replacement, temporal, shared-analysis, scaling, or conversion stage | Setup may retain the same explicit Open CUDA compatibility route; after a prepared frame starts, failure never retries the legacy route in that frame |

The Maxine owner generalizes the existing resident-section executor and
preserves its shared upload, matte reuse, bounded resources, final readback,
failure handling, and compatibility counters. The Open CUDA owner contains CUDA
driver/stream ownership, prepared preprocessing and postprocessing kernels,
ONNX Runtime CUDA IoBinding, validated artifact identity, reusable device
resources, bounded completion waiting, and retryable shutdown. Model discovery,
manifest parsing, artifact validation, provider selection, session creation,
and resource sizing happen at setup.

The Vulkan adapter may advertise product capability only for mirror and
fixed-center vignette, with the exact evidence described in
[INSTALLER_VULKAN_CAPABILITIES.md](INSTALLER_VULKAN_CAPABILITIES.md). Vulkan
matting, tracked-center vignette, Auto Frame, eye contact, video denoise,
virtual key light, and virtual-background modes retain their existing
production, degraded, experimental, diagnostics-only, or blocked states.
Creating a generic Vulkan adapter does not promote them.

`camera_pipeline.cpp` still contains compatibility orchestration while effect
families move through parity, failure, transfer, and attribution gates.
Consequently, `studiocast_video_runtime` still links all optional backend
targets. This is migration debt, not the intended final dependency topology.
Remove a compatibility planner or dependency only after every affected live
route has passed its rollback gate; do not leave two indefinite planners.

## Transfers, CPU tails, and telemetry

One eligible full-domain GPU island reports:

```text
host input -> one upload -> contiguous resident stages
           -> one final readback -> declared writer/format CPU tail
```

Intermediate uploads/readbacks, synchronization points, and CPU-tail
boundaries are explicit counters. An adapter cannot hide a host round trip
inside provider work or infer interoperation across domains. CPU-only
passthrough remains CPU-only with zero GPU transfers.

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
degraded, blocked, and fallback state. The pipeline caches a fixed-capacity
multi-island snapshot with:

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
