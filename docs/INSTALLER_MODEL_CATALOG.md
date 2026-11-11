# Installer curated-model catalog and transactions

- Status: Active installer/model transaction contract
- Applies To: StudioCast v0.2.9 curated catalog and installer packaging
- Audience: Installer, model, packaging, and release maintainers
- Last Verified: 2026-11-11

## Default catalog

`installer/models/curated-model-catalog-v1.json` is the versioned,
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

### FastDVDnet correction decision and promotion gate (2026-11-10)

The approved product decision is to repair/re-export `fastdvdnet_sigma15`,
preserving its ID, default selection, and the seven-pack/eight-artifact contract.
The currently published 416,788-byte artifact matches its pinned hash but is
invalid: 26 convolution-weight references have no producer/initializer, and 68
initializer tensors reference an absent `fastdvdnet.onnx.data`. Hash verification
proves identity, not graph validity. Keep this failure visible; never substitute
random weights, alias missing tensors, silently switch models, or overwrite an
installed pack during validation.

`scripts/models/export_fastdvdnet.py` provides an offline, reproducible correction
using the upstream network and pretrained `model.pth` at commit
`c8fdf6182a0340e89dd18f5df25b47337cbede6f`. Its embedded hashes verify the Python
source, checkpoint, and full license before import/deserialization. Source:
[upstream immutable tree](https://github.com/m-tassano/fastdvdnet/tree/c8fdf6182a0340e89dd18f5df25b47337cbede6f).
It exports the established FP32 RGB NCHW `noisy`/`noise_map`/`denoised` contract
as one opset-17 ONNX file with dynamic N/H/W, validates a complete graph without
external data, and compares ONNX Runtime CPU output with the strictly loaded
upstream network across nine geometry/noise cases. H/W must be multiples of four.
CPU equivalence is artifact evidence, not CUDA or full-camera qualification.

Reproduction uses Python 3.10 on Linux x86_64 and an isolated environment:

```bash
python3 -m venv /tmp/fastdvdnet-export-env
/tmp/fastdvdnet-export-env/bin/python -m pip install \
  -r scripts/models/fastdvdnet-export.requirements.txt
# Manually fetch models.py, model.pth, LICENSE from the immutable raw GitHub URL
# printed as UPSTREAM_BASE in the script, into /tmp/fastdvdnet-upstream.
/tmp/fastdvdnet-export-env/bin/python scripts/models/export_fastdvdnet.py \
  --upstream /tmp/fastdvdnet-upstream --output /tmp/fastdvdnet-candidate
python3 tests/installer/models/fastdvdnet_export_tests.py -v
```

The corrected candidate is 9,948,304 bytes with SHA-256
`0895dcad878f44ff4b610e0569bf701c8e23566e675c22a9bb0d59b95dcfc7ac`.
Repeated independent invocations produced identical artifact bytes. The
`337f3d8` / `b204b0a` exporter fix composes the full upstream license
idempotently; a fresh full-graph check and nine-case CPU export reproduced all
three candidate files byte-for-byte. Six exporter regressions passed. Nine CPU
cases passed with maximum absolute output error `4.76837158203125e-7`.
The output contains an isolated source tree, candidate `model.json` and full
license, a complete candidate catalog, and `validation.json` provenance/results.
It never changes the shipped catalog, installed models, or external hosting.

The shipped catalog below deliberately retains the existing published identity
until publication and reviewed promotion are complete. Three-file Hugging Face
publication and local catalog update are authorized. The actual preupload
request returned HTTP 401, and the browser was logged out. No working
authentication is available; the user reported no credential available. This is an authentication blocker, not missing approval. No remote
file was published, and no active catalog/template or installed pack changed.
Promotion requires verified publication of the exact corrected artifact to a
usable source, matching candidate metadata/license hashes, a new catalog version and measured total,
updated pinned-catalog tests, required CUDA qualification, and the normal signed
clean-HEAD source/AppImage release chain. The candidate catalog is review
material, not signed release metadata and not a recommended-install trust grant.
Without publication, simply changing the default artifact hash would make the
download fail integrity verification. Local `--source` transport may validate a
candidate in temporary HOME/XDG roots; it must not activate it in the user's
installed tree without a separately authorized install.

The unaccepted eight-file promotion proposal is staged separately at
`/tmp/studiocast-fastdvdnet-promotion-20261111`. It proposes catalog version
`2026-11-11`, 140,755,803 total bytes, and the corrected sigma15 artifact; these
are **proposed**, not active defaults. Proposal-only checks passed: 19 model
transactions, 52 backend, nine packaging, and one GUI test, plus corrected real
artifact local-source and offline-cache transactions in temporary roots. These
checks do not establish remote availability or accept that proposal into the
main tree. Sigma25/sigma50 remain untouched noncatalog BYO templates with their
previous artifact hashes, not published defaults.

Resume material is in
`/tmp/studiocast-fastdvdnet-assets-20261110/review-candidate/`:
`publication-plan.json`, `promotion-proposal.patch`,
`promotion-handoff-20261111.json`, and `publish_verified_fastdvdnet.py`.
Establish normal secure Hugging Face authentication with write access to
`10dallasj/studiocast`; keep credentials out of chat, source, and logs. The
publication helper must return a verified receipt for the exact three approved
files at an immutable commit before the orchestrator accepts catalog provenance
and reruns its pinned checks. Publication does not install a model or bypass the
signed release chain.

An old unmarked installed sigma15 pack will fail the proposed metadata
ownership check with `curated_ownership_unverified` and remain preserved. Do
not weaken ownership verification or promise an automatic overwrite; any
installed-pack repair needs a separately authorized ownership-safe transaction.

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
