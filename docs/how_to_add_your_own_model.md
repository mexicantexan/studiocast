# Open-source video model packs (open_video)

StudioCast does **not** ship model binaries in git.

Instead, model packs are defined as **metadata + licensing** in `./resources/model_packs/open_video/` and are installed into:

- `${XDG_DATA_HOME:-$HOME/.local/share}/studiocast/models/open_video/`

At runtime, StudioCast discovers installed packs by scanning that directory.

For sourcing, conversion, and hosting of the actual model binaries (ONNX / dlib data files), see:

- `docs/open_source_video_model_conversion_to_onnx.md`

## Directory taxonomy

Packs are organized by **subject area**, then by **model pack**:

```
open_video/
  matting/
    <pack_dir>/
      model.json
      LICENSE.txt
      model.onnx
  face_detection/
    <pack_dir>/
      model.json
      LICENSE.txt
      ...
  face_landmarks/
    <pack_dir>/
      model.json
      LICENSE.txt
      ...
  video_denoise/
    <pack_dir>/
      model.json
      LICENSE.txt
      ...
  eye_contact/
    <pack_dir>/
      model.json
      LICENSE.txt
      ...
```

Notes:

- `./resources/model_packs/open_video/**` contains **templates only** (metadata + license notes).
- Installed packs live under `~/.local/share/studiocast/models/open_video/**`.
- Multiple quality/latency variants are represented as multiple packs within the same subject.

### Curated quality tiers

StudioCast templates typically provide **Good / Better / Best** tiers within a subject.

For example, `face_detection/` includes YuNet in three tiers:

- **Good Quality (YuNet INT8)** — fastest; slightly reduced accuracy / compatibility depends on int8 kernel support
- **Better Quality (YuNet INT8 BQ)** — blockwise-quantized int8 variant (good accuracy, still fast)
- **Best Quality (YuNet FP32)** — reference fp32 model (highest compatibility)

Similarly, `video_denoise/` includes FastDVDnet sigma preset packs:

- **Good Quality (FastDVDnet sigma15)** — light denoise / detail-preserving (best for mild noise)
- **Better Quality (FastDVDnet sigma25)** — bring-your-own preset/example for moderate noise
- **Best Quality (FastDVDnet sigma50)** — bring-your-own stronger-noise preset/example; may soften fine detail

The curated installer default is exactly `fastdvdnet_sigma15`; sigma25/sigma50
are noncatalog BYO templates, not default or published curated artifacts. Their
existing metadata/artifact hashes remain untouched by the sigma15 repair.
The active sigma15 catalog still points to the known invalid published graph;
the corrected candidate's authorized publication is authentication-blocked. See
[INSTALLER_MODEL_CATALOG.md](INSTALLER_MODEL_CATALOG.md).

And `eye_contact/` includes gaze-correction-cam packs in Good/Better/Best variants:

- **Good Quality (Gaze Correction Cam v0.1.1 FP16)** — fastest on GPU; intended for CUDA/TensorRT (CPU EP may not support FP16).
- **Better Quality (Gaze Correction Cam v0.1.1 FP32)** — reference fp32 export; best compatibility (CPU EP and GPU EP).
- **Best Quality (Gaze Correction Cam v3 Placeholder)** — placeholder for upstream “v3” model updates; installer drops in the new ONNX files when available.

Eye contact packs typically depend on `face_landmarks/` (e.g. `dlib_68_ibug_300w`) to locate and crop eye regions.

## Tasks

`model.json` uses a `task` field to declare what the model is used for.

Current/planned tasks:

- `matting` (a.k.a. segmentation matte) — Virtual Background, Auto Frame (matte-derived ROI), Virtual Key Light.
- `face_detection` — Auto Frame (face ROI), Eye Contact (face ROI).
- `face_landmarks` — Eye Contact (eye region extraction / anchors).
- `video_denoise` — Video Noise Removal.
- `eye_contact` — Eye Contact (warp/flow model).

## `model.json` schema

StudioCast supports **v1-style matting fields** (input/output/preprocess) required by the Open CUDA backend.

Most non-matting tasks use a more flexible **v2 schema** (`schema_version = 2`) with an explicit `files[]` list.

### Matting fields (Open CUDA)

Open CUDA matting packs require the v1-style `input` / `output` / `preprocess` fields. Packs may be schema v1 or schema v2, but must include these fields.

Required fields:

- `id` (string, stable identifier)
- `display_name` (string)
- `task` = `"matting"`
- `onnx_filename` (string)
- `input` (object): `layout`, `dtype`, `width`, `height`, `channels`
- `output` (object): `name`, `kind` (currently `"alpha"`), `dtype`
- `preprocess` (object): `kind` (`"linear"`), `scale`, `mean[]`, `std[]`

Example (see `./resources/model_packs/open_video/matting/**/model.json`).

### Schema v2 (multi-task)

V2 is designed for tasks that:

- have **multiple inputs/outputs**,
- require **non-ONNX artifacts** (e.g. dlib landmark predictor),
- ship as **multiple model files** (e.g. L/R eye models),
- require task-specific post-processing.

V2 fields are:

- `schema_version` = `2`
- `id`, `display_name`, `task`
- `files[]` (optional but recommended):
  - `name` (filename inside the pack directory)
  - `kind` (`"onnx"`, `"dlib_shape_predictor"`, `"mediapipe_task"`, `"data"`, ...)
  - `sha256` (optional; installer may verify)
  - `role` (optional; e.g. `"left"`, `"right"`, `"main"`)
- `onnx` (optional):
  - `main` (filename from `files[]`)
  - `inputs[]` / `outputs[]` (name/layout/dtype/shape)
  - `preprocess` / `postprocess` (task-specific)
- `origin` (optional): where the model comes from (repo, paper, hosted asset URL, etc.)

The v2 schema is intentionally permissive; the runtime validates only what the corresponding effect requires.

## Licensing

Each pack directory **must** include `LICENSE.txt` summarizing:

- upstream repository / model source
- license type
- any notable restrictions (e.g. non-commercial, attribution)

StudioCast is non-commercial, but **you are still responsible** for complying with third‑party license terms.

## Discovery and debugging

Model packs are discovered from the user-local models root:

```bash
./<build-dir>/studiocast-open video-paths
./<build-dir>/studiocast-open video-list-models
./<build-dir>/studiocast-open video-list-models --task face_detection
```

## Shared inference

Many effects share the same underlying analysis (face detections, landmarks, matte). StudioCast keeps a per-frame cache to avoid running the same ML model multiple times in one frame.

See:

- `docs/open_video_shared_inference.md`
