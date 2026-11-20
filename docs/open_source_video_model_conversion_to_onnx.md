# Open Video model sourcing, conversion, and hosting

StudioCast’s open-source video effects use **model packs** installed under:

```text
${XDG_DATA_HOME:-$HOME/.local/share}/studiocast/models/open_video/
```

The repository contains **metadata-only templates** (no binaries) in:

```text
./resources/model_packs/open_video/
```

This doc explains how to:

- source upstream model weights,
- convert them to **C++-friendly artifacts** (usually ONNX),
- host the converted artifacts (Hugging Face or Google Drive),
- and populate your local model-pack install tree.

If you only want to understand the directory layout and schema, start with:

- `docs/how_to_add_your_own_model.md`

---

## 1) Populate a model pack from a template

Each template directory already contains:

- `model.json`
- `LICENSE.txt`

For a new BYO pack directory (example: YuNet FP32), copy the template without
overwriting an existing installation. Use a separate temporary XDG data root
for conversion/qualification; installed user packs are not test fixtures.

```bash
# Model root
ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/studiocast/models/open_video"

# Pick a template (metadata-only)
TPL="./resources/model_packs/open_video/face_detection/Best Quality (YuNet FP32)"

# Install directory (human-friendly; does NOT need to match model.json:id)
DEST="$ROOT/face_detection/Best Quality (YuNet FP32)"

mkdir -p "$DEST"
cp -n "$TPL/model.json" "$DEST/model.json"
cp -n "$TPL/LICENSE.txt" "$DEST/LICENSE.txt"

# Then add the required artifact(s) with the exact filenames referenced by model.json.
# e.g. $DEST/face_detection_yunet_2023mar.onnx
```

Validate discovery:

```bash
# Show discovered Open Video packs.
./build/studiocast-open video-list-models

# Validate that ONNX Runtime can load the model (requires ORT-enabled build).
./build/studiocast-open video-self-test --task face_detection
```

---

## 2) Hosting converted artifacts

StudioCast does not ship model binaries in git. You can host them anywhere you control.

### Option A: Hugging Face (recommended)

Pros:

- stable direct-download URLs,
- supports large files via Git LFS,
- easy versioning.

Typical direct download URL pattern:

```text
https://huggingface.co/<user>/<repo>/resolve/main/<path>/<file>
```

For a curated catalog proposal, replace `main` with an immutable reviewed
revision and verify exact sizes, SHA-256 values, metadata, and full licenses.
Generating a local candidate does not authorize publication or an active
catalog update. Follow [INSTALLER_MODEL_CATALOG.md](INSTALLER_MODEL_CATALOG.md)
and preserve installed-pack ownership.

### Option B: Google Drive

Pros:

- quick and easy.

Cons:

- URLs and permissions are easier to break.

If you use Drive, make the file publicly readable and use a stable direct-download URL.

---

## 3) Model notes and conversions

### 3.1 YuNet face detection (OpenCV Zoo) — ONNX source

The YuNet templates expect one of these ONNX files:

- `face_detection_yunet_2023mar.onnx` (FP32)
- `face_detection_yunet_2023mar_int8.onnx` (INT8)
- `face_detection_yunet_2023mar_int8bq.onnx` (INT8 + BQ)

Upstream:

- https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

Simply download the desired `.onnx` and copy it into the installed pack directory.
The resident Vulkan profile described below additionally needs converted FP32
ncnn artifacts; an ONNX-only or INT8 pack does not satisfy that profile.

---

### 3.2 dlib 68-point face landmarks (iBUG 300-W) — decompress only

The dlib landmark template expects:

- `shape_predictor_68_face_landmarks.dat`

Upstream download (compressed):

- http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Install steps:

```bash
cd "$DEST"  # pack directory
curl -fL -o shape_predictor_68_face_landmarks.dat.bz2 \
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 -f shape_predictor_68_face_landmarks.dat.bz2
```

Notes:

- This predictor is commonly distributed with non-commercial restrictions. See the template’s `LICENSE.txt`.
- Vulkan reuses the same serialized predictor with a bounded loader and GPU
  evaluation, not a new model ID. BYO files may omit a declared checksum;
  preparation then computes a digest and seals that exact snapshot without
  implying verified origin or curated parity.

---

### 3.3 FastDVDnet video denoise — export PyTorch → ONNX

FastDVDnet is upstream PyTorch code + a pretrained `model.pth` weight file.

Upstream repo:

- https://github.com/m-tassano/fastdvdnet

The StudioCast templates assume an ONNX export with:

- input `noisy`: **float32** `[N, 15, H, W]` in **RGB [0..1]**
  - 5 contiguous frames concatenated on channels: `[t-2|t-1|t|t+1|t+2]`
- input `noise_map`: **float32** `[N, 1, H, W]` in **[0..1]**
  - for a sigma preset, use a constant map: `sigma/255`
- output `denoised`: **float32** `[N, 3, H, W]` in **RGB [0..1]**

Important:

- The network downsamples twice, so **H and W should be multiples of 4**.
  - Many webcam modes already satisfy this (e.g., 1280×720, 1920×1080).
  - If you export without padding logic, your runtime must pad/crop similarly to upstream `temp_denoise()`.

A minimal conversion script outline (run inside a cloned `fastdvdnet` repo):

```python
import torch
from models import FastDVDnet

def strip_dp(sd):
    # handle DataParallel checkpoints (keys prefixed with "module.")
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

weights = "./model.pth"  # download from upstream / your own checkpoint
sd = strip_dp(torch.load(weights, map_location="cpu"))

m = FastDVDnet(num_input_frames=5)
m.load_state_dict(sd)
m.eval()

# Dummy inputs (pick any H/W divisible by 4)
noisy = torch.randn(1, 15, 128, 128)
noise_map = torch.full((1, 1, 128, 128), 25.0 / 255.0)

torch.onnx.export(
    m,
    (noisy, noise_map),
    "fastdvdnet.onnx",
    input_names=["noisy", "noise_map"],
    output_names=["denoised"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={
        "noisy": {0: "N", 2: "H", 3: "W"},
        "noise_map": {0: "N", 2: "H", 3: "W"},
        "denoised": {0: "N", 2: "H", 3: "W"},
    },
)
```

After export, place the ONNX file into one or more StudioCast FastDVDnet pack directories.
The live causal runtime supplies its established five-frame history with
current-frame repetition where future frames are unavailable; the exported
tensor layout does not introduce future-frame buffering. Vulkan edge-pads odd
extents to multiples of four, then crops back to the requested geometry.

---

### 3.4 Eye contact (gaze-correction-cam) — convert TF checkpoints → ONNX

StudioCast’s Eye Contact templates track WangWilly’s `gaze-correction-cam` project.

Upstream repo:

- https://github.com/WangWilly/gaze-correction-cam

Upstream distributes TensorFlow checkpoints for the FLX warping models (left/right eye) plus a face-landmark backend.

The StudioCast templates expect two ONNX models:

- `gaze_flx_left.onnx`
- `gaze_flx_right.onnx`

The curated FP32 artifacts have this exact signature:

- `eye_rgb:0`: float32 `[1, 3, 48, 64]` (RGB [0..1])
- `anchors:0`: float32 `[1, 12, 48, 64]` (six landmark-guidance
  points encoded as paired x/y offset maps)
- `angles:0`: float32 `[1, 2]` (yaw/pitch delta)
- output `rgb:0`: float32 `[1, 3, 48, 64]` (corrected RGB eye crop in
  NCHW layout and the 0..1 range)

A typical conversion pipeline is:

1. Clone the upstream repo and download the published checkpoint assets (see the upstream README / release notes).
2. Load the checkpoint into a TF graph / SavedModel.
3. Convert SavedModel → ONNX using `tf2onnx`.

The exact export script depends on the upstream model code (and is likely to change when the repo lands its v3 update).

Recommendation:

- Treat `gaze_flx_left.onnx` / `gaze_flx_right.onnx` as **your own hosted artifacts**, versioned by upstream tag.
- When upstream updates to v3, keep the same StudioCast pack layout but bump `model.json:id` and filenames.

## 4) Offline FP32 ncnn candidates for Vulkan

`scripts/models/convert_video_ncnn.py` adds graphs for the existing curated
FastDVDnet sigma15, YuNet FP32, and Better Quality FLX FP32 packs. It retains
their original ONNX bytes, IDs, defaults, and dependencies. It does not convert
the Good Quality FP16 FLX pack or sigma25/sigma50 BYO presets. MODNet continues
to use its separate converter and singular `ncnn_vulkan` contract; see
[the schema guide](how_to_add_your_own_model.md#vulkan-non-matting-schema-v2).

Prepare existing local inputs matching `scripts/models/video-ncnn.lock.json`:
the four source ONNX files, pnnx executable/version/hash, full YuNet and FLX
licenses, and Python environment (`numpy==2.1.3`, `onnx==1.16.0`,
`onnxruntime==1.23.2`). No fetching, package installation, model installation,
publication, or active catalog update occurs during conversion. The output
directory must not exist.

Build the offline CPU reference from the lock's clean ncnn commit
`24ea07e87f544a3fbc59980b57885a92c75f2d24` on an AVX2/FMA-capable x86_64 CPU.
This is separate from the patched production Vulkan build:

```bash
cmake -S /path/to/exact/ncnn-checkout -B /tmp/video-ncnn-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_RUNTIME_CPU=OFF \
  -DNCNN_AVX=ON -DNCNN_AVX2=ON -DNCNN_FMA=ON -DNCNN_AVXVNNI=ON \
  -DNCNN_AVX512=OFF -DNCNN_INT8=OFF -DNCNN_BF16=OFF -DNCNN_OPENMP=OFF \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF
cmake --build /tmp/video-ncnn-cpu --parallel 2
c++ -O2 -std=c++17 -I/path/to/exact/ncnn-checkout/src \
  -I/tmp/video-ncnn-cpu/src scripts/models/video_ncnn_reference.cpp \
  /tmp/video-ncnn-cpu/src/libncnn.a -pthread -o /tmp/video-ncnn-reference
/path/to/locked-python scripts/models/convert_video_ncnn.py \
  --fastdvdnet /path/to/verified/fastdvdnet.onnx \
  --yunet /path/to/verified/face_detection_yunet_2023mar.onnx \
  --left /path/to/verified/gaze_flx_left.onnx \
  --right /path/to/verified/gaze_flx_right.onnx \
  --pnnx /path/to/pinned/pnnx --reference /tmp/video-ncnn-reference \
  --yunet-license /path/to/full/yunet/LICENSE \
  --flx-license /path/to/full/flx/LICENSE --output /tmp/video-ncnn-candidate
python3 tests/installer/models/video_ncnn_conversion_tests.py -v
```

The converter invokes pinned pnnx with `fp16=0`, normalizes external blob names,
rejects unreviewed operators/quantized or dynamic weights, and checks exact
source hashes and tensor signatures. Its only FLX operator rewrites are the
reviewed Tile and average-pool forms. A successful pnnx exit alone is not proof
of a supported graph: direct conversion of the full FLX RGB graph retains
unsupported tail operators.

Instead, each FLX graph exports the learned `flow` and `lcm` tensors, each
FP32 `[1,2,48,64]`. The resident Vulkan tail restores the original corrected-RGB
contract. It adds flow to the original meshgrid using exact embedded axis
bit patterns; recomputing a linspace changes rounding. Coordinates use
`px=(x+1)*32`, `py=(y+1)*24`. Both floor and upper neighbor indices are clipped
**before** interpolation weights are formed, preserving the original four-term
addition order. This differs at borders from ordinary clamped bilinear
sampling. The final RGB is `warped*lcm[:,0:1]+lcm[:,1:2]`, broadcast over RGB.
Canonical eye orientation/anchors, byte clamp/rounding, strength blend, and
left-before-right compositing remain part of the effect, not new model IDs.
The existing look-away parameter remains a no-op; this conversion adds no
blink or gaze-estimation semantics.

Offline validation compares black, white, gradient, checker, and seeded-noise
inputs; FastDVDnet adds three geometries and three sigma values, and FLX adds
three angles per eye. The source-prefix plus exact tail must equal original
full-ONNX RGB with zero error before converted RGB is accepted. Converted
outputs must satisfy maximum absolute error `2e-4` and mean absolute error
`2e-5`. CPU ONNX/ncnn execution here is an offline oracle, never a production
Vulkan fallback or CPU tracking tail.

The new output contains `source/open_video/`, hash-pinned metadata/full
licenses under `metadata/`, eight added parameter/binary artifacts, numerical
fixtures under `work/`, `validation.json`, exact FLX axis bits, and
`curated-model-catalog-v1.candidate.json`. Its status is `candidate_unpublished`
and GPU qualification starts as `not_run`. Preserve this isolated tree for
review. No published URL, curated recommendation, device coverage, camera
quality, latency, or manager-readiness claim follows from conversion success.

### Opt-in native qualification

Use a build with the required patched ncnn Vulkan dependency and dlib enabled;
see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#building-resident-vulkan-matting).
These commands execute on a real GPU, consume the local candidate fixtures,
and perform no download or installation. Run them explicitly on an otherwise
idle test device; they are not the default hermetic documentation checks:

```bash
./<build-dir>/studiocast-ncnn-vulkan-inference-tests /tmp/video-ncnn-candidate
STUDIOCAST_REQUIRE_VULKAN_TRACKING=1 \
STUDIOCAST_DLIB_ORACLE_MODELS_ROOT=/path/to/isolated/open_video \
STUDIOCAST_VULKAN_MODEL_CANDIDATE_ROOT=/tmp/video-ncnn-candidate \
  ./<build-dir>/studiocast-vulkan-tracking-kernel-tests
```

The isolated model root must contain the selected dlib dependency. Without
the positional candidate root or tracking opt-in, the corresponding test exits
77 (skip). Kernel/model tests do not replace actual production-manager tests,
face-positive Eye Contact comparisons, canonical-order combinations, reset and
optional-failure checks, or camera/vendor/deployment qualification. A local face
fixture must have recorded identity/provenance and suitable permission; do not
commit a photo merely because it is installed with a system package. Track
remaining gates in [VULKAN_EFFECTS_COMPLETION.md](VULKAN_EFFECTS_COMPLETION.md).
