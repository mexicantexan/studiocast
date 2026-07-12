# Installer Vulkan Capability Audit

This audit is the installer policy input for StudioCast `v0.2.9`. It describes
the current product path, not what the Vulkan utility library could eventually
support. The installer must evaluate capability per effect and must not treat a
loadable Vulkan library, a physical device, or successful utility-kernel setup
as proof that every Open Video effect is usable.

The status vocabulary is:

- **production usable**: implemented in the live pipeline, covered as a product
  path, and suitable for automatic recommendation;
- **usable with degraded behavior**: functionally connected to the live
  pipeline, but contains a documented CPU tail or other material limitation;
- **experimental**: a guarded implementation exists, but the default product
  diagnostics or distributed artifacts do not make it available;
- **diagnostics-only**: useful for probes or development measurements, but not
  selectable as a live effect backend;
- **stub/unavailable**: absent from the Vulkan live path or deliberately removed
  when Vulkan is explicitly selected.

## Per-effect matrix

| Canonical effect ID | Current Vulkan status | Evidence and limitation | Selectable CPU fallback | Installer capability/reason code |
| --- | --- | --- | --- | --- |
| `mirror` | stub/unavailable | The canonical planner explicitly disables mirror. A host-side `MirrorEffect` class exists, but the live planner never schedules it. | none | `vulkan_effect_not_implemented` |
| `virtual_background.blur` | experimental | Vulkan blur/composite kernels and a fail-closed production matting-session contract exist. Default daemon diagnostics still block this effect because no production device-resident matting runtime is exposed. | none; the CPU class is an unwired center-focus placeholder, not semantic matting | `open_vulkan_matting_unavailable` |
| `virtual_background.remove` | experimental | Same device-resident matting gate as blur. | none; the CPU class is an unwired center-focus placeholder | `open_vulkan_matting_unavailable` |
| `virtual_background.replace` | experimental | Same device-resident matting gate as blur, plus the normal replacement-image validation. | none | `open_vulkan_matting_unavailable` |
| `auto_frame` | usable with degraded behavior | The live Vulkan path performs crop/resize on Vulkan. Subject tracking is a CPU tail: YuNet face detection is preferred; otherwise a Vulkan matte is read back periodically for CPU box extraction. | none as a selectable CPU engine; the tracking tail is internal to this Vulkan path | `vulkan_effect_cpu_tail` |
| `eye_contact` | stub/unavailable | Explicit Vulkan selection removes this stage. The Open Video implementation is not a Vulkan implementation. | none through the explicit CPU compute backend | `vulkan_effect_not_implemented` |
| `video_noise_removal` | stub/unavailable | Explicit Vulkan selection removes this stage. FastDVDnet/Open Video and the lightweight temporal path are not Vulkan effect paths. | none through the explicit CPU compute backend | `vulkan_effect_not_implemented` |
| `virtual_key_light` | experimental | The live Vulkan relight kernel is connected, but it requires the same foreground matte that default diagnostics mark unavailable. | none; CPU work seen in other backends is a tail, not a selectable CPU engine | `open_vulkan_matting_unavailable` |
| `vignette` | stub/unavailable | Explicit Vulkan selection removes vignette; the standalone implementation is CUDA-only. | none | `vulkan_effect_not_implemented` |

No canonical effect is currently in the **production usable** Vulkan category.
The ncnn Vulkan spike is **diagnostics-only** and is not a canonical effect or a
production fallback.

## Virtual-background conclusion

The documented production virtual-background limitation remains true for the
default product surface.

`DiagnoseOpenVulkanDefault()` reports utility-kernel/device readiness, but sets
the matting runtime to `none`, device-residency mode to `unavailable`, and
blocks all three virtual-background modes. The production session and runtime
lifecycle in `src/core/open_video/vulkan_matting_session.*` and
`vulkan_matting_runtime.*` are deliberately fail closed: they require an
opt-in ncnn Vulkan build, reviewed schema-v2 `ncnn_vulkan` artifacts with
checksums, matching Vulkan device ownership, no CPU layers, and device-resident
input and alpha output. The curated default matting packs remain ONNX-only, so
the presence of `modnet-webnn-256-fp32` does not satisfy this contract.

The milestone spike does not change that conclusion. It uses CPU `ncnn::Mat`
input/output and explicitly does not prove zero-copy interoperation with
StudioCast's Vulkan allocations.

## CPU fallback conclusion

The installer must not infer a CPU fallback from source-file names or from an
internal CPU tail. With `video.compute.backend=cpu`, the service suppresses the
current compute effects. The CPU background blur/remove classes are standalone
placeholder/benchmark implementations and are not scheduled by the canonical
live planner. Mirror also has a host implementation but is explicitly disabled
by that planner. Therefore none of the nine canonical effects currently has a
selectable production CPU engine fallback for recommendation purposes.

Some Open Video implementations can create CPU ONNX Runtime sessions after a
provider failure, and Vulkan Auto Frame deliberately uses CPU tracking. These
are degraded behavior inside another selected backend, not evidence for a
general CPU effect engine.

## Recommendation gates

Recommended installation must not select Vulkan for an effect in this release.
Custom/Advanced may expose Open Vulkan for diagnostics or explicit developer
use, with the matrix limitation attached to each selected effect. Revisit this
gate per effect when a row becomes **production usable**; do not promote the
backend globally.

At minimum, analyzer facts must distinguish:

- `vulkan_backend_disabled_in_build`;
- `vulkan_loader_unavailable`;
- `vulkan_no_physical_device`;
- `vulkan_no_compute_queue`;
- `vulkan_only_cpu_devices_available`;
- `vulkan_requested_device_not_found`;
- `vulkan_requested_device_no_compute_queue`;
- `vulkan_requested_device_ambiguous`;
- `open_vulkan_utility_kernels_unavailable`;
- `open_vulkan_matting_unavailable`;
- `vulkan_effect_cpu_tail`;
- `vulkan_effect_not_implemented`.

The existing device-selection reason codes should be preserved verbatim where
they already exist. A CPU Vulkan device such as lavapipe is not hardware compute
capability unless the developer-only software-device opt-in is explicit.

For a future virtual-background or key-light promotion, the facts must require
all of the following rather than loader presence alone:

1. Open Vulkan compiled in, loader and instance usable.
2. A selected non-CPU physical device with a compute queue and logical device.
3. Utility shader pipelines created successfully.
4. The exact effect appears in daemon `available_effects` and not in
   `blocked_effects`.
5. A production matting runtime was created, its graph loaded and warmed, and
   device identity matched StudioCast's selected Vulkan device.
6. Input, alpha, and output residency evidence is true, with no CPU layers.
7. The selected model pack contains verified, compatible `ncnn_vulkan`
   artifacts. An ONNX-only matting pack is insufficient.

For Auto Frame, analyzer output must additionally expose the CPU tracking tail
and its provider. Its present **usable with degraded behavior** status is not an
automatic recommendation.

## Evidence map

- `src/core/video/camera_pipeline.cpp`: explicit-Vulkan stage filtering,
  per-effect initialization, Vulkan Auto Frame CPU tails, and live stage calls.
- `src/core/vulkan/kernels/resize_bilinear.cpp`:
  `DiagnoseOpenVulkanDefault()` and the three virtual-background blockers.
- `src/core/open_video/vulkan_matting_runtime.*` and
  `vulkan_matting_session.*`: fail-closed device-residency lifecycle.
- `src/core/vulkan/vulkan_device.*`: loader, physical-device, compute-queue,
  hardware-first selection, and stable device reason codes.
- `src/core/video/effects/broadcast_effect_rules.cpp`: mirror suppression and
  canonical effect ordering.
- `tests/vulkan_kernel_tests.cpp`, `tests/vulkan_matting_runtime_tests.cpp`,
  `tests/vulkan_frame_execution_plan_tests.cpp`,
  `tests/open_vulkan_matting_setup_policy_tests.cpp`, and
  `tests/daemon_status_tests.cpp`: device selection, runtime evidence,
  frame-plan barriers, setup caching, and blocked status characterization.
