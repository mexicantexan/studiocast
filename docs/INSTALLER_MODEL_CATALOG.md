# Installer curated-model catalog and transactions

## Default catalog

`packaging/models/curated-model-catalog-v1.json` is the versioned,
machine-readable model contract used by the installer. Its defaults are exactly:

Open Audio:

- `fastenhancer_s_vd_v1`
- `fastenhancer_m_vd_v1`

Open Video:

- `modnet-webnn-256-fp32`
- `yunet_opencv_zoo_2023mar_fp32`
- `dlib_68_ibug_300w`
- `gaze_correction_cam_flx_v0_1_1`
- `fastdvdnet_sigma15`

These are seven packs and eight artifacts because the gaze-correction pack has
separate left- and right-eye ONNX files. Maxine is not redistributed and its
presence never removes these open-source fallback defaults.

Each artifact records its destination name, pinned SHA-256, source candidates,
provenance, and license metadata. Runtime `model.json` and `LICENSE.txt` files
come only from hash-pinned files in the verified StudioCast source/package
surface; they are not accepted from a model download source.

Catalog version `2026-07-19` records the independently verified local byte
counts paired with the pre-existing pinned hashes:

| Pack/artifact | Bytes |
| --- | ---: |
| `fastenhancer_s_vd_v1/model.onnx` | 832,775 |
| `fastenhancer_m_vd_v1/model.onnx` | 2,033,628 |
| `modnet-webnn-256-fp32/model.onnx` | 25,888,640 |
| `yunet_opencv_zoo_2023mar_fp32/model.onnx` | 232,589 |
| `dlib_68_ibug_300w/shape_predictor_68_face_landmarks.dat` | 99,693,937 |
| `gaze_correction_cam_flx_v0_1_1/gaze_flx_left.onnx` | 1,062,936 |
| `gaze_correction_cam_flx_v0_1_1/gaze_flx_right.onnx` | 1,062,994 |
| `fastdvdnet_sigma15/model.onnx` | 416,788 |

The total is 131,224,287 bytes. Every entry is `size_status: known`, and the
catalog-level evidence reason is
`artifact_sizes_independently_verified_against_pinned_sha256`. A trusted
catalog requires every artifact to have a strictly positive integer size; a
missing, null, zero, negative, boolean, floating-point, string, inconsistent,
partial, or unknown size contract is rejected before filesystem mutation.
`trusted: false` continues to produce `artifact_sizes_untrusted` in Recommended
mode even if numbers are present.

The release manifest v1 does not list model artifacts directly. Instead, its
signed, hash-pinned source archive and Installer AppImage authenticate the
catalog transitively: official archives are produced from a clean `git archive
HEAD`, the signed manifest authenticates the exact archive/AppImage bytes, and
the packaged catalog is part of those authenticated bytes. The production
public trust root and signing workflow described in `RELEASE_CHANNEL.md` remain
a release prerequisite; catalog trust metadata alone cannot substitute for
that signed distribution chain. Sizes must never be estimated or invented.

The dlib iBUG 300-W landmark predictor has non-commercial training-data
restrictions. Its catalog license is `LicenseRef-iBUG-300-W-NonCommercial`, and
the installed `LICENSE.txt` calls out that restriction. Review text must expose
this before download.

## Transaction behavior

`installer/models/model_transactions.py` and the
`studiocast-model-transaction` launcher implement unprivileged user-local model
transactions. The shell entrypoints remain:

```bash
scripts/install/open_audio_models.sh --list
scripts/install/open_video_models.sh --list
scripts/install/open_audio_models.sh --source /verified/local/source
scripts/install/open_video_models.sh --offline
```

Sources may be HTTP(S), `file://`, or local paths. Tests and offline workflows
use local sources; the daemon and live pipeline never download models. The old
`STUDIOCAST_HF_REPO_ID` and `STUDIOCAST_HF_REV` variables remain compatible,
while `STUDIOCAST_MODEL_SOURCE` selects a complete alternate source base.

For every artifact, the transaction checks a good installed file, then the
content-addressed verified cache. Online transfers use a `.part` file and
resume at its current byte offset when the source supports it. A completed file
must match its pinned hash and, when authoritative metadata exists, byte size
before it atomically replaces the cache entry. A mismatch is never activated.
An offline transaction succeeds only from verified installed/cache content.

All artifacts for one pack are acquired before mutation. The transaction then
creates a same-filesystem staging directory, copies hash-pinned metadata,
license, and verified artifacts, re-verifies the stage, writes a curated
ownership marker, and atomically activates the directory. If acquisition or
staging fails, an existing pack remains untouched. If activation fails after
moving an old pack aside, it is restored.

Pack failures are structured and isolated. When some selected optional packs
succeed and another fails, the overall state is `degraded` with reason
`optional_model_failure`; callers can offer Retry or Continue without hiding
the committed packs.

Recommended review plans carry all eight exact sizes and bind the selected
catalog version/hash, canonical pack objects, download rows, and disk totals.
Apply reloads the packaged catalog and revalidates those bindings before any
operation can mutate the filesystem. Custom and Advanced routes retain their
optional degraded/retry behavior, but never bypass size or SHA-256 checks.

## Preservation and removal

Catalog paths are safe relative paths beneath the explicitly selected family
root. Traversal, absolute paths, unsafe symlinks, duplicate IDs/paths/files, and
unknown catalog fields are rejected.

The transaction never enumerates arbitrary directories for installation or
removal. Custom packs at paths outside the catalog are untouched during
install, update, repair, and curated cleanup. If a catalog destination already
exists without a StudioCast ownership marker, it is migrated only when its
hash-pinned `model.json` and `LICENSE.txt` exactly match the legacy curated
template; otherwise the operation fails with `curated_ownership_unverified` and
preserves it.

`remove-curated` removes only exact catalog destinations bearing the matching
StudioCast marker. Missing packs are idempotent success. A custom, symlinked,
unmarked, or mismatched destination is preserved and reported as a typed
failure. This contract is intended for clean-install model reset; broad removal
of the whole model root is forbidden.

## Release and test requirements

- Release tooling may update hashes, trusted sizes, provenance, and licenses
  only from reviewed signed metadata; it must not add a private signing key.
- Network-free tests use a temporary local transport, HOME/XDG roots, and tiny
  fixtures. They must never download real model binaries.
- Existing verified cache entries survive source failures.
- Checksum failures must preserve a good active pack.
- Update/repair must not delete custom packs.
- Clean install removes only marked curated packs, then reacquires the seven
  defaults from a verified source/cache.
