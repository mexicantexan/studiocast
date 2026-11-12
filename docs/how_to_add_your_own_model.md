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
Catalog `2026-11-11` promotes the corrected, reproducible sigma15 artifact
published at immutable Hugging Face commit `8ffe9688`. This metadata update
does not replace installed packs or complete a signed application release. See
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

## Vulkan matting schema v2

The opt-in resident Vulkan runtime consumes schema-v2 matting packs with the
existing ONNX artifact and `input` / `output` / `preprocess` metadata preserved.
It adds two checksum-pinned `files[]` entries and an `ncnn_vulkan` contract:

```json
{
  "ncnn_vulkan": {
    "param_file": "model.ncnn.param",
    "bin_file": "model.ncnn.bin",
    "input_blob": "input",
    "output_blob": "output",
    "converter": {"name": "pnnx", "version": "20261020"},
    "precision": "fp32"
  }
}
```

This is an additional manifest field, not a complete standalone manifest.
`param_file` and `bin_file` must name unique `files[]` entries of kind
`ncnn_param` and `ncnn_bin`, respectively, with role `vulkan_matting` and valid
SHA-256 values. All artifact paths must stay inside the pack. Converter identity,
blob names, precision, and exact NCHW float32 input/output metadata are mandatory.
The runtime verifies immutable snapshots before loading; missing files,
path escapes, checksum mismatches, incompatible tensors, CPU layers, provider
mismatch, failed warmup, or unsafe completion fail closed. Validation and
hashing happen during setup, never in the frame loop.

The reviewed MODNet profile is FP32 RGB NCHW `1x3x256x256`, linear normalized
to `[-1,1]`, with blob `input`; output is float32 alpha NCHW `1x1x256x256`,
blob `output`. No retraining or ONNX replacement occurs. Its candidate artifacts
are:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `model.onnx` | 25,888,640 | `07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9` |
| `model.ncnn.param` | 12,113 | `4e205032860a9359fa2555c6ef66dece2a8dcd034a1a61466c58d1310af8a1bd` |
| `model.ncnn.bin` | 25,844,000 | `38871660dda4dd324304eb572f43a43ddc5c876c8dff50db2cf09d29a17e3fa2` |

A compatible BYO pack can satisfy runtime validation without inheriting the
curated profile's numerical-parity attestation or installer recommendation.
Vulkan support here covers background blur/remove/replace and virtual key
light. It does not enable Vulkan Auto Frame tracking, Eye Contact, or denoise.
Model-free mirror, fixed-center vignette, and resize can share the resident
island. See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#building-resident-vulkan-matting)
for the required patched ncnn dependency and build switch.

### Reproduce the MODNet conversion offline

`scripts/models/convert_modnet_ncnn.py` takes only pre-existing local inputs.
`scripts/models/modnet-ncnn.lock.json` pins the source ONNX, pnnx executable
and wheel, full upstream license, Python versions, ncnn revision, and CPU
reference options. Prepare those inputs separately; conversion performs no
model download, install, publication, or active catalog update.

Build the native CPU reference against the exact ncnn commit from the lock
(`24ea07e87f544a3fbc59980b57885a92c75f2d24`). This is an AVX2/FMA reference
for an appropriate x86_64 CPU, separate from the production Vulkan dependency:

```bash
cmake -S /path/to/exact/ncnn-checkout -B /tmp/modnet-ncnn-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_RUNTIME_CPU=OFF \
  -DNCNN_AVX=ON -DNCNN_AVX2=ON -DNCNN_FMA=ON -DNCNN_AVXVNNI=ON \
  -DNCNN_AVX512=OFF -DNCNN_INT8=OFF -DNCNN_BF16=OFF -DNCNN_OPENMP=OFF \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF
cmake --build /tmp/modnet-ncnn-cpu
c++ -O2 -std=c++17 -I/path/to/exact/ncnn-checkout/src \
  -I/tmp/modnet-ncnn-cpu/src scripts/models/modnet_ncnn_reference.cpp \
  /tmp/modnet-ncnn-cpu/src/libncnn.a -pthread -o /tmp/modnet-ncnn-reference
python3 -m venv /tmp/modnet-conversion-env
/tmp/modnet-conversion-env/bin/python -m pip install \
  numpy==2.1.3 onnx==1.16.0 onnxruntime==1.23.2
/tmp/modnet-conversion-env/bin/python scripts/models/convert_modnet_ncnn.py \
  --onnx /path/to/verified/model.onnx --pnnx /path/to/pinned/pnnx \
  --license /path/to/full/upstream/LICENSE \
  --reference /tmp/modnet-ncnn-reference --output /tmp/modnet-candidate
python3 tests/installer/models/modnet_ncnn_conversion_tests.py -v
```

Use fresh build/output paths. The converter output must not exist. It verifies
input hashes before executing tooling, fixes the conversion to 256x256 FP32,
normalizes external blob names, rejects unreviewed/quantized/dynamic-weight
layers, checks ONNX completeness, preserves the original ONNX, and compares
ncnn CPU alpha against ONNX Runtime CPU across black, white, gray, gradient,
checker, and seeded-noise cases. `validation.json` records tool identities,
artifact hashes/sizes, reference build features, and numerical results;
`numerical/` retains inputs and ONNX golden tensors for native Vulkan tests.
The generated candidate catalog points at its own `source/` tree and includes
hash-pinned metadata/full license under `metadata/` for isolated transaction
validation.

Acceptance gates remain maximum error `2e-4`, MAE `2e-5`, and mask IoU at 0.5
of at least `0.999`. The pinned AVX2/FMA reference passed all six cases with
maximum error `2.6345252990722656e-5`, maximum MAE `9.3061352e-8`, and zero
mask disagreements. An SSE-only build failed the unchanged maximum-error gate
on black (`0.0002557933`), traced to FP32 InstanceNorm reduction accumulation.
These results do not claim architecture-independent CPU equivalence or camera
quality. CPU reference buffers are offline artifact evidence and never become
a production Vulkan fallback.

The approved publication preserves `modnet-webnn-256-fp32` and existing defaults.
Its two ncnn artifacts, updated metadata, and full license were independently
verified at immutable Hugging Face commit `f6be359c49ba07163287fe3732dc4a3024b91f60`.
The active `2026-11-12` catalog has seven packs and ten artifacts. Future exports
remain candidates until separately reviewed and published; do not substitute
unpublished URLs or alter installed user packs for validation. See
[INSTALLER_MODEL_CATALOG.md](INSTALLER_MODEL_CATALOG.md) for current publication
and ownership-safe transaction state, and
[MANUAL_TESTING.md](MANUAL_TESTING.md#resident-vulkan-matting-validation) for
available native evidence and remaining deployment checks.

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
