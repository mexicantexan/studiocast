# StudioCast stable release channel

## Outcome and trust model

The initial stable channel uses GitHub Releases as transport, but it does not
trust GitHub transport alone. The installer fetches the exact bytes of a
versioned JSON manifest plus its detached Ed25519 signature, verifies the
signature against a public key pinned by the installer package, then validates
the JSON with a strict, closed schema. Core source and installer artifacts each
have an exact byte size, SHA-256, and their own detached Ed25519 signature in
the signed manifest.

The default manifest endpoints are:

```text
https://github.com/mexicantexan/studiocast/releases/latest/download/studiocast-release-manifest.json
https://github.com/mexicantexan/studiocast/releases/latest/download/studiocast-release-manifest.json.sig
```

Consumers may configure another channel URL for a mirror or hermetic test, but
changing transport does not change the pinned trust root. Production accepts
HTTPS only. Tests can explicitly allow `file:` metadata and inject a transport;
they never contact GitHub.

There is deliberately no production signing secret in this repository. The
stable trust root is the Ed25519 public key committed as
`installer/release/keys/studiocast-release-2026.pem` and packaged under the
durable key ID `studiocast-release-2026`. CI must receive the corresponding
signing capability only through the protected `RELEASE_SIGNING_KEY_B64` secret;
`RELEASE_SIGNING_KEY_ID` must equal the packaged key ID. The key in
`tests/data/installer_release` is visibly test-only and must never be trusted by
a shipped installer.

GitHub stores `RELEASE_SIGNING_KEY_B64` only in the protected
`release-signing` environment. That environment names `mexicantexan` as its
required reviewer and permits deployments only from branch `master` or tags
matching `v*`. Its current settings allow the workflow initiator to approve
their own deployment (`prevent_self_review=false`) and allow administrators to
bypass environment protection (`can_admins_bypass=true`), so review is not the
only trust boundary. The non-secret `RELEASE_SIGNING_KEY_ID` remains a
repository Actions variable.

Release-grade AppImage packaging also requires each production public key
explicitly:

```bash
packaging/appimage/build_appimage.sh --appimage-required \
  --appimage-runtime /path/to/sha-pinned/runtime-x86_64 \
  --trusted-release-key \
  studiocast-release-2026=installer/release/keys/studiocast-release-2026.pem
```

The packaging script validates that each input is a non-symlink Ed25519 public
key, rejects fixture/test key IDs and paths, and stages it as
`usr/share/studiocast/installer/trust/keys/<key-id>.pem`. It refuses
`--appimage-required` without at least one key, so a release bundle cannot be
produced that is cryptographically incapable of consuming its stable channel.
The published-release workflow must be wired to the matching public key before
the first release; a signing secret alone is insufficient.

## Manifest v1

The normative machine-readable shape is
[`installer/release/release-manifest-v1.schema.json`](../installer/release/release-manifest-v1.schema.json).
Runtime validation is intentionally stricter than permissive JSON decoders:

- UTF-8 only; duplicate object keys and non-finite numbers are rejected.
- Unknown and missing fields are rejected at every security-relevant level.
- `schema_version` is `1`, `manifest_version` is
  `studiocast-release-manifest/v1`, and `channel` is `stable`.
- Release versions use strict SemVer and `tag` must equal `v<version>`.
- Artifact URLs are HTTPS in production; filenames cannot contain directories.
- Source archive and Installer AppImage metadata contain exact byte size,
  lowercase SHA-256, Ed25519 key ID/signature, repository/commit/workflow
  provenance, and license identifier/name/URL.
- Activation policy is side-by-side with an atomic current pointer, retains the
  previous payload, and declares rollback support.

The manifest detached-signature envelope is canonical JSON:

```json
{
  "algorithm": "ed25519",
  "key_id": "studiocast-release-2026",
  "schema_version": 1,
  "signature": "<base64 signature over exact manifest bytes>"
}
```

Signatures are verified before manifest parsing. Artifact hashes and signatures
are verified before an artifact can be atomically moved into its final cache
name. A mutable or unhashed core source is never accepted.

## Version and update policy

Version precedence follows SemVer 2.0. Build metadata does not affect
precedence; prereleases sort below the corresponding stable version. The
default decisions are:

| Installed compared with available | Default action |
| --- | --- |
| absent | install |
| older | update |
| same | keep current |
| newer | keep current |

Same-version reinstall and downgrade require separate explicit policy flags;
neither is inferred from the channel. An installer older than
`minimum_installer_version` must offer its verified installer self-update or a
manual download before it plans an incompatible application update.

Application updates reuse the manifest-v2 desired configuration. They build
the verified source in a version-specific cache, stage and validate a durable
versioned payload, atomically switch `current`, and retain the previous payload.
If build or core validation fails, the current pointer is never changed. The
release contract's `activation` object and `rollback_metadata()` provide the
transport/backend handoff; the backend owns the actual activation journal.

## Cache, resume, and offline rules

`acquire_artifact()` uses `<filename>` only after full verification and keeps an
incomplete `.<filename>.part` for retry. HTTPS resume requests use `Range`; a
server that ignores a nonzero range fails rather than concatenating a full
response. Local fixture transport seeks to the same byte offset.

An existing verified cache file is returned without transport. Offline mode
never opens a URL and distinguishes cache miss from corrupt cache. A failed
refresh cannot replace a previously verified final file. Temporary data is
moved with `os.replace` only after size, SHA-256, and Ed25519 checks succeed.

`fetch_release_manifest()` stores each verified manifest/signature pair in a
generation directory named by the manifest SHA-256 and atomically replaces a
small `current` pointer. Offline analysis reads only that selected generation,
so a crash cannot expose a new manifest with an old detached signature.
Unsafe cache pointers, symlinked generation content, and hard-linked partial
artifacts are rejected. A verified online refresh repairs a corrupt generation
without selecting the corrupt bytes. HTTPS transport rejects redirects to a
weaker scheme and validates the start offset of resumed responses.

## Packaged source and model surface

Official source archives are always generated with `git archive HEAD`. There
is no working-tree tar fallback: it could capture untracked files or produce
bytes that do not correspond to the signed commit. Packaging also rejects a
working `VERSION` that differs from the value committed at `HEAD`, and refuses
dirty worktrees so the built installer and archived source share one commit
identity.

The Installer component contains the curated model transaction launcher,
module, catalog, and referenced `model.json`/`LICENSE.txt` resources. These
files form a trusted metadata and transaction surface independent of a
user-selected source tree. No ONNX, dlib, TensorRT, or other model binary is
redistributed in the AppDir. The verifier asserts the exact seven default packs
represent eight artifacts, requires their catalog sizes to be trusted, known,
and strictly positive, and rejects staged model binaries.

Manifest v1 does not enumerate model artifacts. The model sizes and pinned
hashes are authenticated transitively because the catalog is inside the
Installer AppImage and clean-HEAD source archive whose exact bytes are
hash-pinned and signed by the manifest. Recommended planning therefore trusts
the complete packaged catalog only after strict validation and binds its hash,
version, selected packs, exact download sizes, and disk totals into the reviewed
plan. Apply revalidates that packaged surface before mutation. This does not
remove the production prerequisite for the offline/HSM-backed signing key,
matching packaged public key, and signed release workflow described above.

## Installer self-update

Self-update uses the AppImage artifact from the same signed manifest. The
consumer detects an AppImage from an explicit absolute path or the `APPIMAGE`
environment input, downloads the newer image into the user cache, verifies it,
and returns an `offer_restart` object with the verified path and proposed argv.
It never overwrites the running AppImage and never launches it as a side effect.

Only a separate, explicit user confirmation may call `relaunch_self_update()`.
If the running AppImage cannot be identified, the cache/download fails, or a
desktop relaunch is unavailable, the result is `manual_download` with the
signed manifest's HTTPS URL. Same-version installers return `not_needed`.

## Release generation and signing

`packaging/release/release_tool.py` computes artifact sizes and SHA-256 values,
signs both artifacts, writes canonical manifest JSON, then signs the exact
manifest bytes. It invokes OpenSSL's Ed25519 implementation without a shell.
The private key is an explicit input path and is never generated, copied, or
retained by the tool.

Example (with the key supplied from an external secret store):

```bash
python3 packaging/release/release_tool.py \
  --version 0.3.0 \
  --minimum-installer-version 0.2.9 \
  --source-archive dist/appimage/StudioCast-0.3.0-source.tar.gz \
  --installer-appimage dist/appimage/StudioCast-Installer-0.3.0-x86_64.AppImage \
  --base-url https://github.com/mexicantexan/studiocast/releases/download/v0.3.0 \
  --release-page-url https://github.com/mexicantexan/studiocast/releases/tag/v0.3.0 \
  --commit 0123456789abcdef0123456789abcdef01234567 \
  --workflow-run-url https://github.com/mexicantexan/studiocast/actions/runs/123 \
  --private-key /run/secrets/studiocast-release-ed25519.pem \
  --key-id studiocast-release-2026 \
  --output dist/appimage/studiocast-release-manifest.json \
  --signature-output dist/appimage/studiocast-release-manifest.json.sig
```

Release CI separates building from signing. The build job never references a
signing variable or secret. It verifies and uploads an exact unsigned artifact
set: AppImage, AppDir archive, source archive, and checksum file. A fresh signing
job downloads those bytes, checks their exact names and checksums, binds the
source archive to the event commit by regenerating the exact canonical
`git archive --format=tar.gz` bytes and comparing them, verifies the event
commit is an ancestor of the fetched remote default branch, then checks the
version/tag identity and compares the AppDir's public key and staged source to
the committed inputs before the protected environment secret is injected into
the signing step. The archive's forgeable pax commit header is not treated as
identity evidence. A missing remote default-branch ref, failed ancestry check,
or byte-different source archive fails closed for both published releases and
signed dry runs. This workflow check still runs when self-review or
administrator bypass permits the protected job to start.

Only the signing job references `RELEASE_SIGNING_KEY_B64` and the repository
variable `RELEASE_SIGNING_KEY_ID=studiocast-release-2026`. It runs in the
protected `release-signing` environment and checks out signing tooling at the
immutable workflow commit rather than from the artifact-producing ref. The
ephemeral private-key file is mode `0600`; the base64 environment value is
unset immediately after decoding, and the file is removed before artifact code
is executed for AppImage inspection.

Published Release events may enter the signing job and must still use a release
tag exactly equal to `v<VERSION>`. Workflow dispatch builds unsigned packaging
artifacts by default. Its explicit `signing_dry_run` boolean may enter the
signing environment only when the workflow itself is dispatched from the
repository default branch; a signed dispatch from another ref fails. The signed
artifact upload includes the manifest and signature with the four verified
packaging files, and a release maintainer must attach the release files to the
matching GitHub Release.

After the private-key file has been removed, CI verifies the detached manifest
signature and both artifacts' signed size, SHA-256, filename, and Ed25519
identity against the committed public key. AppImage inspection never executes
the unsigned target. Before secret injection, CI downloads a type-2 runtime by
immutable GitHub asset ID, verifies its pinned SHA-256, and uses the fixed
`/usr/bin/unsquashfs` package tool to locate and extract exactly one SquashFS
payload. The AppImage must have the expected ELF/type-2 header and a byte-exact
runtime prefix; only the runtime's non-executable 16-byte AppImage digest field
may differ. The SquashFS must begin immediately after that runtime, end at the
canonical padded file boundary with no appended payload, and reproduce the
verified AppDir archive's complete file types, paths, modes, regular-file
sizes/hashes, and symlink targets. Unsafe, dangling, cyclic, or escaping links
fail closed. CI then byte-compares the embedded source archive to the verified
standalone source, compares
`usr/share/studiocast/installer/trust/keys/studiocast-release-2026.pem` to the
committed key, and rejects forbidden secret PEM markers in staged and uploaded
release artifacts.

Key rotation must first ship the replacement public key in an installer signed
by `studiocast-release-2026`. Release manifests may use the replacement key ID
only after that trusted installer is available.

## Verification

Hermetic coverage is in
`tests/installer/release/release_channel_tests.py`:

```bash
python3 -m py_compile installer/release/release_channel.py \
  packaging/release/release_tool.py \
  packaging/release/verify_signed_release.py \
  tests/installer/release/release_channel_tests.py
python3 tests/installer/release/release_channel_tests.py
```

The suite covers valid and invalid manifest signatures, strict JSON/schema,
both artifact hash/signature failures, SemVer and update policy, minimum
installer compatibility, offline cache states, interrupted resume, atomic
replacement, rollback metadata, and AppImage offer/manual/mock-relaunch paths.
