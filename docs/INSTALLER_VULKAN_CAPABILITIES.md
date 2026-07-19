# Installer Vulkan Capability Audit

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
| `mirror` | production usable | The canonical final visual transform uses the shared production Vulkan context, stays resident through the final effect boundary, and is covered by deterministic parity/live-pipeline tests. Recommendation still requires exact per-effect readiness plus non-CPU device, compute queue, healthy context, and utility-kernel evidence. | none | ready: `open_vulkan_mirror_production_ready`; otherwise the exact common Vulkan blocker |
| `virtual_background.blur` | experimental | A canonical resident blur/composite stage exists. Production remains fail-closed because the distributed packs are ONNX-only and no reviewed exact-device ncnn Vulkan adapter/runtime is available. Synthetic matting seams are test evidence only. | none; the CPU class is an unwired center-focus placeholder, not semantic matting | `open_vulkan_matting_unavailable`, with the exact nested adapter/artifact/runtime blocker |
| `virtual_background.remove` | experimental | A canonical resident solid composite exists, behind the same production matting contract as blur. | none; the CPU class is an unwired center-focus placeholder | `open_vulkan_matting_unavailable`, with the exact nested blocker |
| `virtual_background.replace` | experimental | A canonical resident replacement composite and bounded setup-time image upload lifecycle exist. It remains behind production matting readiness and validates the replacement image separately. | none | `open_vulkan_matting_unavailable`, with the exact nested blocker |
| `auto_frame` | usable with degraded behavior; unavailable on the analyzed machine | The crop/resize stage is Vulkan, but tracking and crop-plan smoothing are explicit CPU tails. The shipped YuNet tracker/artifact is not production-ready on the analyzed machine, so cold daemon evidence is fail-closed even though the crop stage exists. A future valid CPU-only tracker would remain degraded, not Vulkan-native. | none; its CPU tracking is an internal tail, not a selectable CPU engine | unavailable: `vulkan_auto_frame_yunet_unavailable`; a valid degraded path reports `vulkan_effect_cpu_tail` |
| `eye_contact` | diagnostics-only / stub-unavailable | Exact per-effect diagnostics exist, but there is no callable Vulkan live stage, exact-device inference provider/adapter, validated Vulkan artifact contract, resident analysis/tensor path, or production gaze/look-away contract. | none | `open_vulkan_eye_contact_unavailable`, nested `open_vulkan_eye_contact_runtime_unavailable` |
| `video_noise_removal` | diagnostics-only / stub-unavailable | Exact per-effect diagnostics exist, but FastDVDnet currently uses CUDA/CPU providers, host temporal history, CPU preprocessing/postprocessing, and ONNX-only artifacts. There is no callable Vulkan stage or bounded resident temporal-history contract. | none | `open_vulkan_video_noise_removal_unavailable`, nested `open_vulkan_video_noise_removal_runtime_unavailable` |
| `virtual_key_light` | experimental | The canonical relight stage is resident, but production readiness requires the same exact same-frame, exact-device production matte evidence as the virtual-background modes. | none; CPU work in other backends is not a selectable CPU engine | `open_vulkan_matting_unavailable`, with the exact nested blocker |
| `vignette` | production usable (fixed center only) | The canonical final Vulkan stage matches the CUDA-reference fixed-center parameter contract and reuses a device-resident factor mask. `center_on_tracked_face` is not supported by this Vulkan implementation; when that semantic becomes observable with retained Auto Frame, only vignette is removed. | none | ready: `open_vulkan_vignette_fixed_center_production_ready`; tracked-center: `vulkan_vignette_tracked_center_not_supported`; otherwise the exact common Vulkan blocker |

Mirror and fixed-center vignette are the only canonical effects currently
eligible for a Vulkan recommendation, and only when their exact daemon
`production_ready` facts and all common runtime facts are true. The analyzed
machine's cold diagnostics do not make Auto Frame, matting consumers, eye
contact, or video noise removal production-ready.

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
7. **Residency/ownership**: the exact StudioCast device and queue are shared,
   buffers stay resident, CPU layers/readbacks are absent, and allocations are
   bounded.
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

## Matting and model boundary

The production matting lifecycle in
`src/core/open_video/vulkan_matting_session.*` and
`vulkan_matting_runtime.*` requires an opt-in ncnn Vulkan build, reviewed
schema-v2 `ncnn_vulkan` parameter/bin artifacts with checksums, the exact
StudioCast physical/logical device and compute queue, no CPU layers, warmed
runtime/graph, device-resident input/alpha/output, synchronous ownership, and
bounded reusable allocations.

The curated matting packs remain ONNX-only. Upstream ncnn does not expose the
reviewed external-device import needed by StudioCast's selected device, queue,
and buffers. The milestone spike uses CPU `ncnn::Mat` input/output and is
diagnostics-only; it cannot satisfy production residency. Consequently the
three virtual-background modes and virtual key light remain experimental and
fail closed.

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
  `tests/vulkan_final_resident_stage_integration_tests.cpp`: the exact
  CameraPipeline final resident orchestration and its real-device executable
  coverage for resize, fixed-center vignette, mirror, combined ordering,
  isolation, allocation reuse, and the final-only readback boundary.
- `src/core/video/camera_pipeline.cpp`: canonical live calls, final-stage
  ordering, fixed-center compatibility, Auto Frame CPU tails, matting consumers,
  and fail-closed ML stage filtering.
- `src/core/vulkan/kernels/resize_bilinear.cpp` and
  `src/core/vulkan/vulkan_device.*`: daemon per-effect readiness publication,
  common device/context/kernel evidence, model/runtime evidence, and JSON.
- `src/core/open_video/vulkan_matting_runtime.*` and
  `vulkan_matting_session.*`: schema-v2 artifact and resident-runtime contract.
- `src/core/video/open_vulkan_eye_contact.*` and
  `open_vulkan_video_noise_removal.*`: diagnostics-only exact blockers.
- `installer/backend/studiocast-installer-backend`: per-effect facts and pure
  recommendation precedence.
- `tests/installer_backend_core_tests.py`: hermetic NVIDIA, AMD, Intel, hybrid,
  no-GPU, loader-only, CPU-device-only, no-compute, device-loss, precedence,
  isolation, fixed-center, and no-CPU-continuation fixtures.
- `tests/vulkan_capability_audit_tests.cpp`, `tests/vulkan_kernel_tests.cpp`,
  and `tests/daemon_status_tests.cpp`: documentation/source consistency,
  strict readiness, daemon schema, and ON/OFF behavior.
