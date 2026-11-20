# StudioCast open_video model pack templates

This directory contains **metadata-only templates** for open-source video model packs.

StudioCast does **not** commit model binaries into git.

Installed model packs live at:

- `${XDG_DATA_HOME:-$HOME/.local/share}/studiocast/models/open_video/`

For sourcing, conversion, and hosting of the actual model binaries (ONNX / dlib data files), see:

- `docs/open_source_video_model_conversion_to_onnx.md`

## Subject areas

- `matting/` — matte/alpha models (`task = "matting"`)
- `face_detection/` — face detection models
  - Templates include YuNet in Good/Better/Best variants (int8 / int8bq / fp32).
- `face_landmarks/` — facial landmark detectors
- `video_denoise/` — video denoising models
  - Templates include FastDVDnet sigma presets in Good/Better/Best variants (sigma15 / sigma25 / sigma50).
- `eye_contact/` — gaze/eye contact models
  - Templates include gaze-correction-cam in Good/Better/Best variants (FP16 / FP32 / v3 placeholder).

## What’s in a pack?

A pack is a directory containing:

- `model.json` — pack metadata (see `docs/how_to_add_your_own_model.md`)
- `LICENSE.txt` — upstream licensing summary
- model artifacts (downloaded at install time), e.g.
  - `*.onnx`
  - `shape_predictor_68_face_landmarks.dat`
