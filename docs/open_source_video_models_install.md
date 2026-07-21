# Open CUDA backend (ONNX Runtime CUDA) — install & model packs

The **Open CUDA** backend (`open_cuda`) is the non-Maxine path for GPU-only video effects.

- It does **not** depend on NVIDIA Maxine SDK assets.
- It runs inference via **ONNX Runtime** with the **CUDA Execution Provider** (CUDA EP).
- There is **no CPU fallback** in this backend: if CUDA / ONNX Runtime / model packs are missing, effects are marked unavailable.

This document covers prerequisites, model pack layout, and common troubleshooting.

## 1) Prerequisites

### NVIDIA driver / CUDA

You need a working NVIDIA driver stack such that the CUDA driver API is available.

Quick sanity check:

```bash
nvidia-smi
```

If `nvidia-smi` fails, fix the driver install first. The Open CUDA backend will be blocked with `cuda_unavailable` in daemon status.

### ONNX Runtime with CUDA EP

StudioCast discovers ONNX Runtime via CMake. For Open CUDA you need an ONNX Runtime build that includes the **CUDA execution provider**.

Repo-provided helper (Ubuntu 22.04+):

```bash
./scripts/setup.sh --deps --onnxruntime-flavor gpu
```

This script (via `scripts/setup/ubuntu.sh`):

- installs build prerequisites (Qt/CMake/Ninja/etc.),
- downloads the official ONNX Runtime tarball,
- installs it under `/opt/studiocast/onnxruntime/<version>/...`,
- and provides a `pkg-config` entry (`onnxruntime.pc`) so CMake can find it.

Without `--onnxruntime-flavor gpu`, the helper chooses `gpu` only when `nvidia-smi` works; otherwise it installs the CPU flavor. Open CUDA requires the GPU flavor with `CUDAExecutionProvider`.

If you install ONNX Runtime another way, make sure this works:

```bash
pkg-config --modversion onnxruntime
```

## 2) Build (ensure Open CUDA is enabled)

On Linux, `STUDIOCAST_ENABLE_OPEN_CUDA` defaults to **ON**.
CUDA remains the preferred default compute backend for usable NVIDIA systems.

If you want to force it:

```bash
cmake -S . -B <build-dir> -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSTUDIOCAST_ENABLE_OPEN_CUDA=ON
cmake --build <build-dir> --target studiocastd studiocastctl studiocast-open
```

If the daemon reports `onnxruntime_not_found`, it usually means ONNX Runtime wasn’t found at build time.

## 3) Install a model pack

Open CUDA discovers **user-supplied model packs** at runtime.

The recommended (human-friendly) model root is:

```
~/.local/share/studiocast/models/open_video/
```

To print the authoritative path on your machine:

```bash
./<build-dir>/studiocast-open paths
```

For a general overview of the open-source video model pack taxonomy (face detection, eye contact, etc.), see:

- `docs/open_video_model_packs.md`

For sourcing/conversion/hosting of the actual model binaries (ONNX / dlib files), see:

- `docs/open_source_video_model_conversion_to_onnx.md`

### Model pack layout

Open CUDA currently consumes only the **matting** packs (`task = "matting"`).

Each Open CUDA matting pack is a directory containing `model.json`, `model.onnx`, and `LICENSE.txt`.
The directory name does **not** need to match `model.json:id` (human-friendly names are OK).

Recommended layout (mirrors `resources/model_packs/open_video`):

```text
~/.local/share/studiocast/models/open_video/matting/<pack_dir>/
  model.onnx
  model.json
  LICENSE.txt
  README.md        (optional)
```

You can validate discovery with:

```bash
./<build-dir>/studiocast-open list-models
```

### `model.json` (metadata)

`model.json` describes how StudioCast should feed the model (layout, tensor names, normalization, etc.).

Minimal example (you must adjust the tensor names and shape to match your ONNX model):

```json
{
  "id": "modnet",
  "display_name": "MODNet",
  "task": "matting",
  "onnx_filename": "model.onnx",
  "input": {
    "name": "input",
    "layout": "nchw",
    "dtype": "float32",
    "width": 512,
    "height": 512,
    "channels": 3
  },
  "output": {
    "name": "alpha",
    "kind": "alpha",
    "dtype": "float32"
  },
  "preprocess": {
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "color": "rgb",
    "range": "0..1"
  }
}
```

Notes:

- `onnx_filename` must be a **safe relative path** within the pack directory.
- `input.name` and `output.name` must match the **actual tensor names** in your ONNX graph.
- `width`/`height` must match what the model expects.

## 4) Verify the backend is runnable

### Verify model pack discovery

```bash
./<build-dir>/studiocast-open list-models
```

You should see your `<model_id>` under “Valid model packs”.

### Verify daemon status (`GET_STATUS`)

Start the daemon:

```bash
./<build-dir>/studiocastd
```

In another terminal, check status JSON and look for `open_cuda.ok`:

```bash
./<build-dir>/studiocastctl status
```

The status payload includes:

- `engines.open_cuda` (preferred)
- a convenience top-level `open_cuda` alias

When runnable, you should see:

- `open_cuda.ok: true`
- `open_cuda.available_effects` containing the virtual background effect IDs

### Force Open CUDA + enable blur (no Maxine required)

This uses the canonical effect IDs (see `src/core/video/effects/broadcast_effect_contract.h`).

```bash
./<build-dir>/studiocastctl effects enable virtual_background.blur \
  --engine open_cuda \
  --strength 0.25
```

If everything is installed correctly, the pipeline will select the Open CUDA engine for virtual background blur.

## Backend selection and fallback

StudioCast reports backend selection in `studiocastctl status` under `video.compute`:

- `preference`: user/config preference (`auto`, `cpu`, `cuda`, or `vulkan`)
- `resolved_backend`: setup-time backend selected by the daemon
- `active_backend`: backend currently active in the pipeline
- `fallback_reason` / `degraded_reason`: why the daemon is not using the requested GPU path

`auto` keeps CUDA preferred on NVIDIA systems when CUDA and the ONNX Runtime CUDA provider are usable. Explicit `vulkan` does not silently run CUDA; if Vulkan is unavailable or lacks the requested effect, status reports Vulkan as unavailable/degraded and the active backend falls back to CPU/pass-through where applicable.

## 5) Selecting an Open CUDA model for Virtual Background

Model packs are an **Open CUDA-only** concern.

- If Virtual Background is **actually running on Open CUDA**, the GUI shows a **Model** dropdown.
- If Virtual Background is **actually running on Maxine**, the Model dropdown is **hidden**.

### How to show the Model dropdown

1. Ensure model packs are installed and discoverable:

   ```bash
   ./<build-dir>/studiocast-open list-models
   ```

2. Start the daemon:

   ```bash
   ./<build-dir>/studiocastd
   ```

3. In the GUI (Video page):

    - Set **Effect engine** to **Open CUDA** (or leave **Auto** if Maxine is not available).
    - Enable Virtual Background (Blur / Remove / Replace).

When the daemon reports the active backend for the selected Virtual Background mode as `open_cuda`, the **Model** row
appears.

### Default vs explicit model selection

- **Default (auto)** corresponds to an empty `model_id` in config. The daemon then uses a deterministic default model
  (`open_cuda.default_model_id` in `studiocastctl status`).
- Selecting a specific entry (e.g. `birefnet_lite`, `birefnet_portrait`, `modnet-webnn-256-fp32`) sets
  `virtual_background.model_id` to that pack ID.

### Where the selection is persisted

The selection is stored in the canonical video effects config, persisted by the daemon at:

- `~/.config/studiocast/daemon.conf` (respecting `XDG_CONFIG_HOME`)

The key is:

- `video.effects.json` (a single-line JSON blob)

Inside that JSON, the field is:

- `virtual_background.model_id`

Example (formatted for readability; the on-disk value is on one line):

```json
{
  "schema_version": 1,
  "engine": "open_cuda",
  "virtual_background": {
    "mode": "blur",
    "model_id": "birefnet_lite"
  }
}
```

## Troubleshooting

### `open_cuda.ok` is false and `blocked_effects` contains `onnxruntime_not_found`

- Your build was compiled without ONNX Runtime.
- Install ONNX Runtime with CUDA provider support and rebuild.

On Ubuntu 22.04+, the intended path is:

```bash
./scripts/setup.sh --deps --onnxruntime-flavor gpu
```

### `open_cuda.ok` is false and `blocked_effects` contains `cuda_unavailable`

- NVIDIA driver / CUDA runtime is not available to the process.
- Verify `nvidia-smi` works.

### `open_cuda.ok` is false and `blocked_effects` contains `missing_model_packs`

- No usable model packs were discovered.
- Run `studiocast-open list-models` to see “Problems” (e.g., missing files or invalid JSON).

### Model pack is listed under “Problems”

Common causes:

- `model.json` is invalid JSON
- `model.onnx` is missing
- `onnx_filename` points outside the pack directory (rejected for safety)

### The Model dropdown is hidden in the GUI

The Model dropdown is shown only when **Virtual Background is enabled** and the daemon reports that the active backend
for the selected Virtual Background mode is **`open_cuda`**.

Common reasons it’s hidden:

- Virtual Background is set to **Off**.
- The pipeline is not running (daemon not started, or video disabled).
- The engine preference is **Auto** and Maxine is available, so the active backend is **Maxine**.
    - To force Open CUDA: set **Effect engine → Open CUDA** in the GUI, or use the CLI example above with
      `--engine open_cuda`.

You can confirm the active backend with:

```bash
./<build-dir>/studiocastctl status
```

Look for `effects_backends` (per-effect backend mapping) and `engines.open_cuda.ok`.

### Selected `model_id` is missing / invalid

If `virtual_background.model_id` references a model pack that is not currently installed:

- The GUI may show a “Missing: <model_id>” entry to make the mismatch visible.
- The daemon will fail to resolve the selected model pack and Virtual Background will not run on Open CUDA until a
  valid pack is selected.

Fix:

- Install the missing pack under `~/.local/share/studiocast/models/open_video/<subject>/<pack_dir>/`, or
- Switch the Model dropdown back to **Default (auto)** (clears `model_id`), or pick another installed model.
