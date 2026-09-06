# StudioCast Architecture

- Status: Active architecture contract
- Applies To: StudioCast v0.2.9 and the current source tree
- Audience: Contributors and maintainers
- Last Verified: 2026-07-23

StudioCast is a daemon-owned Linux audio/video processing application. The Qt
GUI and command-line clients control `studiocastd`; they do not own production
pipelines. The daemon coordinates V4L2 camera capture and v4l2loopback output,
Pulse-based virtual audio services, persisted configuration, optional effect
backends, model discovery, and status reporting.

Implemented components include:

- A Qt controller GUI and standalone Qt installer wizard.
- `studiocastd`, which owns the production video and audio services.
- Neutral contracts, configuration, IPC, utilities, and runtime-loaded CUDA
  support beneath the runtime composition layer.
- Optional Maxine, Open CUDA/Open Video, Open Audio, and Open Vulkan backends.
- Scriptable installer, trusted system-helper protocol, model transactions, and
  signed release-channel support.

## CMake dependency layers

Production code is split at neutral, reusable boundaries before backend and
runtime composition layers:

```text
neutral/support:
  contracts ------> util    hal_contracts
  hal_runtime ----> hal_contracts
  config ---------> contracts, util
  ipc ------------> util    cuda_support ---> util
  probe_support --> config, util
  model_support --> util    audio_support

platform:
  video_support ---> contracts, hal_contracts, util
  video_io ------> contracts, video_support, util
  onnx_support --> ONNX Runtime when enabled

backends:
  maxine_backend -----> audio_support, config, contracts, cuda_support,
                        probe_support, video_support, util
  open_audio_backend -> audio_support, contracts, model_support,
                        onnx_support, util
  open_video_backend -> contracts, cuda_support, model_support,
                        onnx_support, video_support, util
  vulkan_backend -----> contracts, model_support, video_support, util

runtime:
  audio_runtime ---> audio_support, config, contracts, Maxine, Open Audio
  video_execution_runtime ---> hal_runtime
  video_backend_composition ---> video_execution_runtime; privately owns
                                  Maxine, Open Video, Vulkan
  video_runtime ---> contracts, video_backend_composition,
                     video_execution_runtime, video I/O and support
  runtime_config --> audio_runtime, config, video_runtime
  studiocast_core --> runtime_config, ipc, probe_support
```

Each arrow points from a dependent to its dependency.

- `studiocast_util` owns generic filesystem, process, JSON, string, dynamic
  library, OS-release, and XDG helpers.
- `studiocast_contracts` owns stable audio/video value types, effect schemas,
  reason codes, serialization, and backend-independent execution plans.
- `studiocast_hal_contracts` owns the versioned primitive, descriptor,
  resource, execution-domain, prepared-program, completion, telemetry, and
  committed-frame contracts. `studiocast_hal_runtime` owns the deterministic
  static registry, resolver, shadow comparison, and execution-domain planner.
  Both targets are neutral: they contain no optional SDK headers, feature
  definitions, provider discovery, or backend libraries.
- `studiocast_config` owns persisted/runtime configuration parsing and
  validation. Runtime service adapters remain in the high-level core.
- `studiocast_ipc` owns Unix-socket protocol mechanics and has std-only public
  headers.
- `studiocast_audio_support` owns dependency-free DSP shared by audio
  backends, while `studiocast_audio_runtime` owns Pulse/device/service
  composition.
- `studiocast_video_support` owns conversion and CPU effect primitives;
  `studiocast_video_io` owns capture, decoding, v4l2loopback, and writer
  support; `studiocast_video_execution_runtime` owns the fixed-capacity
  backend-neutral prepared execution facade and typed capability/live-status
  snapshots; and `studiocast_video_runtime` owns the existing camera pipeline
  and virtual-camera composition. `studiocast_video_backend_composition`
  privately owns the optional Maxine, Open Video, and Vulkan dependencies and
  exposes the neutral execution surface consumed by the video runtime.
  Reaching that ownership boundary is not evidence that existing behavior has
  reached parity. Bounded compatibility routes must preserve working behavior
  until each live family passes the parity and rollback gates in
  [HAL_PARITY_COMPLETION.md](HAL_PARITY_COMPLETION.md). Release qualification
  remains a separate gate after local parity completion.
- `studiocast_cuda_support` owns the runtime-loaded CUDA driver wrapper,
  buffers, and embedded kernels without a CUDA toolkit dependency. Its
  canonical API is `core/cuda/cuda_driver_api.h` in namespace
  `studiocast::cuda`.
- `studiocast_model_support` owns the Open Video model catalog and the shared
  model-integrity implementation consumed by both open backends.
- `studiocast_onnx_support`, `studiocast_maxine_backend`,
  `studiocast_open_audio_backend`, `studiocast_open_video_backend`, and
  `studiocast_vulkan_backend` isolate optional runtime ownership.
- `studiocast_runtime_config` adapts persisted configuration to the audio and
  video services.
- `studiocast_core` remains an interface compatibility aggregate. New
  executables and tests should link the narrowest owning targets instead.

The historical `core/maxine/cuda_driver_api.h` path remains a forwarding
alias surface only because the protected camera-pipeline source still includes
that path. All other consumers use the neutral CUDA header and namespace; new
code must not add dependencies on the forwarding path.

CMake validates every allowed direct StudioCast target edge, rejects optional
SDK or GUI dependencies on neutral targets, and verifies unique ownership of
every configured production translation unit. Because every production edge
is asserted exactly, an unreviewed dependency or cycle fails during configure.
New backends should link down to the smallest neutral layer that owns their
shared concepts; sibling backends must not link to one another for shared types
or utilities.

The complete HAL contract, exact execution-domain identity, current prepared
adapter gates, status migration, and deferred plugin security boundary are
documented in
[HARDWARE_EXECUTION_LAYER.md](HARDWARE_EXECUTION_LAYER.md). The important
separation is that CPU primitive availability never establishes selectable
product-effect capability, while each v1 resident island is owned by exactly
one adapter. Completion and release claims must also satisfy
[HAL_PARITY_COMPLETION.md](HAL_PARITY_COMPLETION.md); a neutral ownership
cutover alone does not satisfy those gates.

## Canonical effect model: `BroadcastCameraEffects`

The single canonical effect schema across **config persistence**, **IPC**, **daemon pipeline**, **GUI rendering**, and
**CLI JSON** is `BroadcastCameraEffects`.

Authoritative contract (stable effect IDs + parameter IDs):

- `src/core/video/effects/broadcast_effect_contract.h`

Hard rule: do not rename existing effect IDs or parameter IDs. If an effect must change semantics, add a new ID.

## GUI/daemon control plane (JSON)

The daemon exposes a small IPC protocol over a Unix socket.

- `GET_CONFIG` returns the canonical Broadcast effects JSON (effect IDs as keys).
  - Implementation: `studiocastd` uses
    `studiocast::video::BroadcastCameraEffectsContractToJson(...)` in
    `src/daemon/studiocastd_main.cpp`.

- `SET_VIDEO_EFFECTS_JSON <json>` applies a JSON patch against `BroadcastCameraEffects`.
  - Implementation: `studiocastd` uses
    `ApplyBroadcastCameraEffectsPatchJsonText(...)` in `src/daemon/studiocastd_main.cpp`.
  - Legacy compatibility: if the Broadcast patch parser fails, the daemon will try the legacy `CameraEffects` patch
    parser and return a warning if it had to do so.

CLI helpers:

- `studiocastctl effects get` → `GET_CONFIG`
- `studiocastctl effects set --file <effects.json|->` → `SET_VIDEO_EFFECTS_JSON` (file-based to avoid shell quoting)

## Effects engines: `auto_select`, `maxine`, `open_cuda`

The canonical effect schema includes an engine selector:

- `BroadcastCameraEffects.engine`

Accepted JSON values:

- `"auto"` / `"auto_select"`: prefer Maxine when available; otherwise fall back to Open CUDA for supported effects
- `"maxine"`: force Maxine (unsupported effects remain blocked/unavailable)
- `"open_cuda"`: force Open CUDA (GPU-only; requires ONNX Runtime + CUDA EP + model packs)

Installation docs:

- Maxine: `docs/maxine_install.md`
- Open CUDA: `docs/open_source_video_models_install.md`

## Availability: daemon diagnostics are authoritative

Effect availability must be computed **only** by the daemon. Legacy consumers
receive it through `GET_STATUS`; the versioned HAL command below is an additive
typed status surface.

- The daemon includes per-engine diagnostics in `GET_STATUS` under:
  - `engines.maxine` (and a legacy top-level `maxine` alias)
  - `engines.open_cuda` (and a convenience top-level `open_cuda` alias)
- The GUI should treat daemon status as the single source of truth for:
  - whether each engine is usable on this machine
  - which effects are available vs blocked (with reason codes)
  - what remediation/install hints to show

The additive `GET_VIDEO_HAL_STATUS_V1` command exposes a validated typed
multi-island snapshot only when every requested effect is covered by a current
prepared HAL island. Its fixed-capacity authority masks identify the
prepared-route scope. While staged migration retains a compatibility route for
any requested effect, the command fails closed with `ERR` instead of presenting
prepared-only blocked, unavailable, or active facts as authoritative for the
whole live pipeline. Complete prepared-route snapshots distinguish requested,
selected, available, active, degraded, blocked, and fallback state and
attribute activity only after a successful output write. Legacy `GET_STATUS`
bytes remain unchanged and authoritative for compatibility-route state. The
installer parser remains shadow-only, rejects incomplete authority, and is
ineligible to affect recommendations.

## Open CUDA model packs (XDG)

The `open_cuda` backend (ONNX Runtime CUDA EP) discovers user-supplied model packs at runtime.

Default root:

- `~/.local/share/studiocast/models/open_video/<subject>/<pack_dir>/`

Where:

- `<subject>` is a functional category like `matting` (legacy name: `segmentation`).
- `<pack_dir>` is any directory name (it does not need to match `model.json:id`).

Each pack contains:

- `model.json` (matting packs include v1-style fields; multi-task packs use schema v2 with `files[]`)
- model artifact files (e.g. `model.onnx`, `shape_predictor_68_face_landmarks.dat`)
- `LICENSE.txt`

Helper tool:

- `studiocast-open paths` / `studiocast-open install-hints` / `studiocast-open list-models`

### `model.json` schema v1 (draft)

Required top-level fields:

- `id`, `display_name`
- `task`: currently only `"matting"`
- `onnx_filename`
- `input`: `{ name, layout, dtype, width, height, channels }`
- `output`: `{ name, kind, dtype }` where `kind` is `"alpha"`
- `preprocess`: `{ mean[3], std[3], color, range }` where `color` is `"rgb"` and `range` is `"0..1"`

### Production Vulkan matting schema-v2 extension

A Vulkan matting candidate must use schema v2, declare distinct
`ncnn_param`/`ncnn_bin` files with `role: "vulkan_matting"` and installed
SHA-256 digests, and provide an `ncnn_vulkan` object. Its top-level tensor
contract is exact rather than inferred from the graph:

```json
{
  "schema_version": 2,
  "task": "matting",
  "input": {
    "name": "input", "layout": "nchw", "dtype": "float32",
    "width": 256, "height": 256, "channels": 3
  },
  "output": {
    "name": "alpha", "kind": "alpha", "layout": "nchw",
    "dtype": "float32", "width": 160, "height": 144, "channels": 1
  },
  "ncnn_vulkan": {
    "param_file": "model.ncnn.param",
    "bin_file": "model.ncnn.bin",
    "input_blob": "input",
    "output_blob": "alpha",
    "converter": { "name": "pnnx", "version": "20250503" },
    "precision": "fp16"
  }
}
```

`precision` describes the offline ncnn artifact/weights (`fp32` or `fp16`);
StudioCast's external resident input and alpha bindings remain float32. The
output dimensions may differ from the input dimensions and govern alpha
allocation, artifact identity, resize, and element counts.

## No CPU fallback (product rule)

Effects are GPU-only (Maxine + small CUDA post-process where needed). If Maxine, drivers, supported GPU, or feature
models are missing, the pipeline must not run effects and the UI must communicate the “unavailable” state.


## Open Audio (microphone ML) model packs

The Open Audio backend (`open_audio`) uses ONNX Runtime and user-supplied model packs to run streaming
microphone enhancement (noise removal + “studio voice”). Model packs live under:

```
${XDG_DATA_HOME:-~/.local/share}/studiocast/models/open_audio/<model_id>/
```

See `docs/open_source_audio_models_install.md` for the installer script, pack
schema, and validation tooling.

## Audio device safety

Production GUI audio changes go through daemon IPC (`SET_AUDIO_CONFIG`). The daemon and audio pipeline reject
StudioCast virtual microphone sources, Pulse monitor sources, and StudioCast virtual speaker target sinks to avoid
feedback loops. `source: "auto"` is resolved by the daemon/audio pipeline to a safe physical microphone source when the
Pulse default source is unsafe, or fails with an actionable status/config error if no safe source exists.
