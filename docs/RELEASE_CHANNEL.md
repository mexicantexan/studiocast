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

There is deliberately no production private key in this repository. Before the
first signed release, release engineering must provision an offline/HSM-backed
Ed25519 private key as a protected CI secret and commit/package its public key
under a stable key ID. Until that prerequisite is met, production consumers
must report `release.signature.unknown_key` and stop. The key in
`tests/data/installer_release` is visibly test-only and must never be trusted by
a shipped installer.

## Manifest v1

The normative machine-readable shape is
[`packaging/release/release-manifest-v1.schema.json`](../packaging/release/release-manifest-v1.schema.json).
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

For a published-release workflow, CI requires `RELEASE_SIGNING_KEY_B64` and the
repository variable `RELEASE_SIGNING_KEY_ID`. The temporary private-key file is
mode `0600` and removed by a shell trap. CI uploads the manifest and signature
with the AppImage/source artifacts; a release maintainer must attach all of
them to the matching GitHub Release. Workflow dispatch continues to build
unsigned packaging artifacts unless signing is explicitly available.

## Verification

Hermetic coverage is in `tests/release_channel_tests.py`:

```bash
python3 -m py_compile packaging/release/release_channel.py \
  packaging/release/release_tool.py tests/release_channel_tests.py
python3 tests/release_channel_tests.py
```

The suite covers valid and invalid manifest signatures, strict JSON/schema,
both artifact hash/signature failures, SemVer and update policy, minimum
installer compatibility, offline cache states, interrupted resume, atomic
replacement, rollback metadata, and AppImage offer/manual/mock-relaunch paths.
