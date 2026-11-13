# Installer Vulkan Capability Audit

- Status: Active cross-subsystem capability contract
- Applies To: StudioCast v0.2.9 installer recommendation and runtime evidence
- Audience: Installer, runtime, packaging, and test maintainers
- Last Verified: 2026-11-12

This document is the per-effect installer policy input for StudioCast `v0.2.9`.
It describes the canonical live pipeline after the Vulkan production program,
not what a loader, source filename, utility kernel, benchmark, synthetic test
seam, or unused helper could eventually support.

The status vocabulary is:

- **production usable**: implemented in the canonical live pipeline, covered as
  a product path, and eligible for automatic recommendation when exact runtime
  evidence is true on the analyzed machine;
- **usable with degraded behavior**: connected to the live pipeline, but with a
  documented CPU tail or other material limitation;
- **experimental**: a guarded live implementation exists, but distributed
  runtime/model evidence does not make it production-ready;
- **diagnostics-only**: facts and failure diagnostics exist, but no callable
  canonical Vulkan effect stage exists;
- **stub/unavailable**: no selectable Vulkan live stage is present.

## Per-effect matrix

| Canonical effect ID | Vulkan implementation status | Live evidence and limitation | Selectable production CPU fallback | Installer readiness/blocker codes |
| --- | --- | --- | --- | --- |
| `mirror` | production usable | The canonical final visual transform uses the shared production Vulkan context. One bounded mapped upload staging buffer feeds a non-mapped `DEVICE_LOCAL` source; resize scratch and mirror output are non-mapped `DEVICE_LOCAL`; one distinct mapped staging buffer is used only for the final writer-boundary readback. Real-device parity/live-pipeline tests assert the allocation properties, one upload, zero intermediate readbacks/CPU fallback, one final readback, synchronization, ordering, and reuse. Recommendation still requires exact per-effect readiness plus non-CPU device, compute queue, healthy context, and utility-kernel evidence. | none | ready: `open_vulkan_mirror_production_ready`; otherwise the exact common Vulkan blocker |
| `virtual_background.blur` | production usable (opt-in matting) | The prepared camera manager runs resident MODNet inference, alpha resize/feather, and background blur/composite. Requires the production matting build, converted reviewed profile, and actual exact-device warmup/readiness. | none | ready: `open_vulkan_matting_production_ready`; otherwise `open_vulkan_matting_unavailable` with the exact nested blocker |
| `virtual_background.remove` | production usable (opt-in matting) | The prepared resident solid composite uses the same current-frame matte and exact runtime contract as blur. | none | ready: `open_vulkan_matting_production_ready`; otherwise `open_vulkan_matting_unavailable` with the exact nested blocker |
| `virtual_background.replace` | production usable (opt-in matting) | The prepared replacement composite decodes/uploads/resizes its image only at setup, validates that asset separately, and consumes the same resident matte. | none | ready: `open_vulkan_matting_production_ready`; otherwise `open_vulkan_matting_unavailable` with the exact nested blocker |
| `auto_frame` | resident implementation; exact qualification required | YuNet inference, decode/top-K/NMS, crop planning/smoothing, matte bounds/cadence and crop/resize execute on the prepared Vulkan context. Exact reviewed artifact, warmup, GPU tracking, ownership, bounded allocation and zero CPU-transfer proofs are mandatory. | none | ready: `open_vulkan_auto_frame_production_ready`; unavailable: `vulkan_auto_frame_yunet_unavailable`; retained legacy CPU-tail evidence stays degraded |
| `eye_contact` | resident implementation; exact qualification required | GPU YuNet and ordered dlib landmarks feed two FP32 FLX learned prefixes and exact resident RGB sampler/lighting/composite tails. Requires compiled dlib, physical FP64 support and logical-device enablement, all selected artifacts, warmup, reviewed numerical attestation and no CPU analysis/tensor tail. Look-away remains the existing no-op contract. | none | ready: `open_vulkan_eye_contact_production_ready`; otherwise `open_vulkan_eye_contact_unavailable`, nested `open_vulkan_eye_contact_runtime_unavailable` |
| `video_noise_removal` | resident implementation; exact qualification required | FP32 FastDVDnet uses three physical resident histories and causal logical `[t-2,t-1,t,t,t]`, resident packing/padding/noise/postprocess and reset-on-discontinuity. Zero strength omits live inference; setup still validates/warms the selected graph. | none | ready: `open_vulkan_video_noise_removal_production_ready`; otherwise `open_vulkan_video_noise_removal_unavailable`, nested `open_vulkan_video_noise_removal_runtime_unavailable` |
| `virtual_key_light` | production usable (opt-in matting) | The prepared relight stage reuses the same current-frame resident matte when combined with a background mode. Independent key light runs one resident inference. | none | ready: `open_vulkan_matting_production_ready`; otherwise `open_vulkan_matting_unavailable` with the exact nested blocker |
| `vignette` | fixed and tracked resident implementations | Fixed-center keeps its exact parameter/readiness contract. When Auto Frame and tracked center are enabled, GPU tracking transforms the selected center through crop/resize before the resident vignette and mirror. Missing tracking blocks only affected semantics. | none | ready: `open_vulkan_vignette_fixed_center_production_ready` or `open_vulkan_vignette_tracked_center_production_ready`; otherwise the exact per-effect/common blocker |

Mirror and fixed-center vignette require their exact daemon readiness facts.
The four matting consumers additionally require the opt-in production matting
build and reviewed converted MODNet profile, successfully prepared and warmed on
the analyzed machine. Every recommendation still requires exact per-effect
`production_ready` and common runtime facts. Missing prerequisites remain
blocked. Model effects are promoted only after their independent exact proofs,
not from matting or common utility readiness. The full combined qualification
ledger and pending multi-model publication are in
[VULKAN_EFFECTS_COMPLETION.md](VULKAN_EFFECTS_COMPLETION.md).
The active curated catalog includes the approved, remotely verified ncnn MODNet
artifacts alongside the preserved ONNX file. Installing them does not enable a
missing production runtime or establish exact-device readiness. See
[INSTALLER_MODEL_CATALOG.md](INSTALLER_MODEL_CATALOG.md) for publication evidence.

## Evidence layers and fail-closed facts

The analyzer and recommendation deliberately separate these layers:

1. **Build**: Open Vulkan is compiled in.
2. **Loader/instance**: the runtime library loads and an instance is created.
3. **Physical device**: a device exists and is not a CPU/software Vulkan
   implementation.
4. **Queue/device/context**: a compute queue, logical device, and healthy owned
   context exist. Device loss or fatal submission state invalidates readiness.
5. **Kernels**: the utility shader pipeline is created and healthy.
6. **Runtime/model**: model-backed effects prove their exact runtime, artifact,
   warm-up, and model contract. Loader or kernel success is not model evidence.
7. **Residency/ownership**: the exact StudioCast device and queue are shared;
   claimed persistent and intermediate buffers are non-mapped `DEVICE_LOCAL`;
   mapped memory is restricted to bounded upload/final-readback transport;
   CPU layers/intermediate readbacks are absent; and allocations are bounded.
8. **Per-effect live stage**: the exact canonical effect has an implemented
   live stage and an effect-specific production-ready attestation.

`available_effects` is useful live-stage evidence but is never sufficient by
itself. For mirror and vignette, the installer also requires the additive
`mirror_production_ready` or `vignette_fixed_center_production_ready` fact, its
matching success code, all common hardware/context/kernel facts, and the exact
effect in `available_effects`. Missing, legacy, or internally inconsistent
diagnostics fail closed. Matting's global `blocked_reason` must not disable
unrelated mirror or vignette capability.

Success/readiness codes and blocker codes are separate facts. The stable common
blockers include:

- `vulkan_backend_disabled_in_build`;
- `vulkan_runtime_not_found` and installer probe code
  `vulkan_loader_unavailable`;
- `vulkan_no_physical_device`;
- `vulkan_no_compute_queue`;
- `vulkan_only_cpu_devices_available`;
- `vulkan_requested_device_not_found`;
- `vulkan_requested_device_no_compute_queue`;
- `vulkan_requested_device_ambiguous`;
- `vulkan_device_create_failed`;
- `vulkan_context_uninitialized`, `vulkan_device_lost`, and other latched
  context-health reasons;
- `vulkan_production_hardware_not_ready`;
- `open_vulkan_utility_kernels_unavailable`;
- `open_vulkan_runtime_diagnostics_unavailable` when installed daemon evidence
  cannot be obtained.

Recommendation additionally emits
`effect.<id>.<engine>.capability_evidence_missing` when a canonical per-effect
or per-engine capability fact is absent. A production-looking Vulkan fact set
that fails its exact evidence or success-attestation contract and supplies no
more specific upstream Vulkan blocker emits
`effect.<id>.vulkan.production_evidence_inconsistent`. Existing exact upstream
blockers take precedence and are preserved verbatim.

A loader-only host, CPU Vulkan device, missing compute queue, unhealthy or lost
device, mismatched/absent per-effect evidence, missing model/artifact/runtime,
failed warm-up, mismatched device identity, CPU layers/readbacks/tails, or
missing live stage all prevent recommendation for the affected effect. They do
not disable an unrelated effect whose exact evidence remains valid.

Installer analysis does not treat a previously cached or cold daemon snapshot
as current evidence. Before consuming installed status it invokes the bounded
installed CLI diagnostics-refresh operation and then reads status. Missing CLI
support, a failed refresh, timeout, malformed response, or absent refreshed
per-effect facts remains fail-closed as
`open_vulkan_runtime_diagnostics_unavailable` or the more exact reported
blocker; it never falls back to loader/device guesses.

## Matting and model boundary

The production matting lifecycle in
`src/core/open_video/vulkan_matting_session.*` and
`vulkan_matting_runtime.*` requires an opt-in ncnn Vulkan build, reviewed
schema-v2 `ncnn_vulkan` parameter/bin artifacts with checksums, the exact
StudioCast physical/logical device and compute queue, no CPU layers, warmed
runtime/graph, device-resident input/alpha/output, synchronous ownership, and
bounded reusable allocations.

The production owner adopts an ncnn-owned logical device and compute queue
before allocating StudioCast resources. It matches configured physical-device
intent and retains the same provider entry point, ownership domain, and native
lifetime through every child. Pinned replay ABI 2 records inference once,
forbids internal unbounded submission/reset, and uses StudioCast's finite fence
waits. Caller deadlines and the existing 16-synchronization facade cap remain
enforced. The original matting-only arena uses 256 MiB / 512 resources.
Expanded model pools share a single caller-owned aggregate allocation quota;
per-model arenas do not multiply that bound. Actual prepared/high-water bytes
and resource counts and the reviewed full profile are recorded in the expanded
completion ledger. Model loading
consumes bounded, sealed, checksum-verified snapshots, so path replacement or
in-place mutation cannot change the bytes attached to the prepared identity.

Explicit diagnostics refresh prepares and warms a real session; ordinary status
polling reads the daemon cache. `matting_parity_validated` is limited to the
reviewed exact MODNet parameter/bin hashes, blob names, geometry, and RGB
normalization profile, plus consumer readiness. A compatible BYO graph does not
inherit that numerical attestation. The historical CPU `ncnn::Mat` spike remains
diagnostics-only and is not the production adapter.

P1000/535.309.01 evidence covers six numerical model cases and eight actual
manager cases with three frames each: one upload/one final readback, no
intermediate transfers, CPU tails, execution-time host allocations, or fallback.
It does not qualify AMD/Intel, forced non-push-descriptor execution, Khronos
validation cleanliness, real-camera/V4L2 transport, portrait quality, optimized
latency, or clean full-native sanitizers. See
[VULKAN_MATTING_IMPLEMENTATION.md](VULKAN_MATTING_IMPLEMENTATION.md) for exact
source/model/dependency evidence and the separate publication decision.

## Recommendation and CPU fallback

Recommendation is a pure per-effect choice with precedence:

```text
Maxine -> CUDA -> Vulkan -> CPU
```

Vulkan is selected only for an exact `production_usable` matrix value backed by
that effect's `vulkan_evidence.production_ready=true`. A legacy status string,
global Vulkan boolean, loader, device, utility kernel, or `available_effects`
entry cannot promote it. Auto Frame's CPU-tail/degraded facts are never treated
as production Vulkan evidence.

The CPU step is considered only if that exact canonical effect has a selectable
production CPU engine. None of the nine does today. Host mirror helpers,
placeholder background classes, CPU tracking inside Auto Frame, CPU provider
fallback inside another backend, and unused classes are not selectable CPU
engine evidence. Installer facts report
`effect.<id>.cpu.no_selectable_production_path` independently for every effect.

Recommended selection sets the Open Vulkan build feature to true exactly when
at least one effect actually selects Vulkan, and explicitly sets it false when
none does. Advanced/custom selections and an existing installation's Modify
route retain their compatibility behavior. Maxine or CUDA still wins
independently for an effect with higher-precedence production evidence.

## Evidence map

- `src/core/video/open_vulkan_mirror.*` and
  `open_vulkan_vignette.*`: strict production predicates and canonical effect
  wrappers.
- `src/core/video/open_vulkan_final_resident_stage.*` and
  `tests/gpu/vulkan/vulkan_final_resident_stage_integration_tests.cpp`: the exact
  CameraPipeline final resident orchestration and its real-device executable
  coverage for device-local source/scratch/effect/mask allocation, bounded
  upload/readback staging, resize, fixed-center vignette, mirror, combined
  ordering, isolation, allocation reuse, synchronization counters, and the
  final-only readback boundary.
- `src/core/video/camera_pipeline.cpp`: canonical live calls, final-stage
  ordering and the neutral production manager. The strict Vulkan factory
  implements resident model effects; retained legacy CPU tracking remains
  explicitly degraded and cannot qualify production Vulkan.
- `src/core/vulkan/kernels/resize_bilinear.cpp` and
  `src/core/vulkan/vulkan_device.*`: daemon per-effect readiness publication,
  common device/context/kernel evidence, model/runtime evidence, and JSON.
- `src/core/open_video/vulkan_matting_runtime.*` and
  `vulkan_matting_session.*`: schema-v2 artifact and resident-runtime contract.
- `src/core/video/open_vulkan_eye_contact.*` and
  `open_vulkan_video_noise_removal.*`: exact compile/runtime/model/parity
  readiness predicates; neither compiled support nor utility readiness promotes
  an effect without the prepared model proofs.
- `installer/backend/studiocast-installer-backend`: per-effect facts and pure
  recommendation precedence, including the bounded installed-CLI cold
  diagnostics refresh before status consumption.
- `tests/installer/backend/installer_backend_core_tests.py`: hermetic NVIDIA, AMD, Intel, hybrid,
  no-GPU, loader-only, CPU-device-only, no-compute, device-loss, precedence,
  isolation, fixed-center, and no-CPU-continuation fixtures.
- `tests/gpu/vulkan/vulkan_capability_audit_tests.cpp`,
  `tests/gpu/vulkan/vulkan_kernel_tests.cpp`, and
  `tests/integration/core/daemon_status_tests.cpp`: documentation/source consistency,
  strict readiness, daemon schema, and ON/OFF behavior.

## Expanded model and installer contracts

The additive schema-v2 `ncnn_vulkan_networks[]` contract retains original ONNX
files and supports only the reviewed native FastDVDnet/YuNet/FLX profiles.
Existing singular MODNet `ncnn_vulkan` metadata remains unchanged. Generic graph
validity does not grant a curated numerical attestation: cold readiness checks
exact reviewed parameter/bin hashes/profile and the prepared owner. BYO compatible
models retain their stable IDs and explicit artifact contracts; a missing dlib
manifest checksum can be sealed against the exact selection digest.

Typed status validates independent model effect proofs: common Vulkan evidence
1–12 plus artifact16; denoise50; eye51/52/53/55; Auto Frame51/54; tracked
vignette19/51/54. A matting-only proof cannot qualify those effects. The legacy
projection cannot invent reviewed parity or execution counters absent from its
typed snapshot. The installer requires the complete cold prepared evidence before
recommendation; typed multi-island status remains shadow-only for that decision.

Normal installer Open Vulkan builds provision the exact pinned ncnn/glslang replay
SDK in an owned cache. Two SDK source downloads and the build reserve are separate
from model totals. Archive hashes, extraction bounds, tooling/patch identities,
plan bindings, prefix inventory and receipts are verified before atomic activation.
Offline operation accepts only verified cached products/archives. Dlib discovery
must succeed before installer activation; developer-only builds may omit it and
keep unrelated Vulkan effects. No SDK/model download occurs while streaming or
polling status. Installed model packs retain their ownership-safe transaction gate.
