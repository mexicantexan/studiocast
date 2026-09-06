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

To install a pack locally (example: YuNet FP32):

```bash
# Model root
ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/studiocast/models/open_video"

# Pick a template (metadata-only)
TPL="./resources/model_packs/open_video/face_detection/Best Quality (YuNet FP32)"

# Install directory (human-friendly; does NOT need to match model.json:id)
DEST="$ROOT/face_detection/Best Quality (YuNet FP32)"

mkdir -p "$DEST"
cp -f "$TPL/model.json" "$DEST/model.json"
cp -f "$TPL/LICENSE.txt" "$DEST/LICENSE.txt"

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

### Option B: Google Drive

Pros:

- quick and easy.

Cons:

- URLs and permissions are easier to break.

If you use Drive, make the file publicly readable and use a stable direct-download URL.

---

## 3) Model notes and conversions

### 3.1 YuNet face detection (OpenCV Zoo) — no conversion needed

The YuNet templates expect one of these ONNX files:

- `face_detection_yunet_2023mar.onnx` (FP32)
- `face_detection_yunet_2023mar_int8.onnx` (INT8)
- `face_detection_yunet_2023mar_int8bq.onnx` (INT8 + BQ)

Upstream:

- https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

Simply download the desired `.onnx` and copy it into the installed pack directory.

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
