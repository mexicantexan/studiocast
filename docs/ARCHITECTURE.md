# Architecture (draft)

Phase 0 is mostly scaffolding, but there are already some hard “single source of truth” decisions that new
contributors should follow.

Planned components:
- GUI (Qt)
- Audio Service (PipeWire graph node)
- Video Service (V4L2 input + v4l2loopback output)
- Effects Engine abstraction layer (multiple GPU engines: Maxine + Open CUDA)
- SDK Manager (downloads/installs user-obtained Maxine assets)

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
- Open CUDA: `docs/open_cuda_install.md`

## Availability: daemon diagnostics are authoritative

Effect availability must be computed **only** by the daemon and exposed via `GET_STATUS`.

- The daemon includes per-engine diagnostics in `GET_STATUS` under:
  - `engines.maxine` (and a legacy top-level `maxine` alias)
  - `engines.open_cuda` (and a convenience top-level `open_cuda` alias)
- The GUI should treat daemon status as the single source of truth for:
  - whether each engine is usable on this machine
  - which effects are available vs blocked (with reason codes)
  - what remediation/install hints to show

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
