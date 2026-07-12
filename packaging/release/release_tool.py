#!/usr/bin/env python3
"""Generate and sign StudioCast release-channel metadata.

Private keys are accepted only as explicit command-line paths.  This repository
contains no private key and the tool never generates or persists one.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path

from release_channel import MANIFEST_VERSION, canonical_json, sha256_file, validate_manifest


def sign_file(path: Path, private_key: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="studiocast-release-sign-") as temp:
        signature = Path(temp) / "signature"
        result = subprocess.run(
            ["openssl", "pkeyutl", "-sign", "-inkey", str(private_key), "-rawin", "-in", str(path), "-out", str(signature)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )
        if result.returncode != 0:
            raise SystemExit(f"OpenSSL signing failed: {result.stderr.decode('utf-8', 'replace').strip()}")
        return base64.b64encode(signature.read_bytes()).decode("ascii")


def artifact(path: Path, artifact_id: str, url: str, args: argparse.Namespace) -> dict:
    return {
        "artifact_id": artifact_id,
        "filename": path.name,
        "url": url,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "signature": {"algorithm": "ed25519", "key_id": args.key_id, "value": sign_file(path, args.private_key)},
        "provenance": {
            "repository": args.repository,
            "commit": args.commit,
            "workflow_run_url": args.workflow_run_url,
        },
        "license": {
            "spdx_id": args.license_spdx,
            "name": args.license_name,
            "url": args.license_url,
        },
    }


def generate(args: argparse.Namespace) -> None:
    now = args.generated_at or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    base = args.base_url.rstrip("/")
    manifest = {
        "schema_version": 1,
        "manifest_version": MANIFEST_VERSION,
        "channel": "stable",
        "generated_at": now,
        "release": {
            "version": args.version,
            "tag": f"v{args.version}",
            "published_at": args.published_at or now,
            "page_url": args.release_page_url,
        },
        "minimum_installer_version": args.minimum_installer_version,
        "artifacts": {
            "source_archive": artifact(args.source_archive, "studiocast-source", f"{base}/{args.source_archive.name}", args),
            "installer_appimage": artifact(args.installer_appimage, "studiocast-installer-appimage", f"{base}/{args.installer_appimage.name}", args),
        },
        "activation": {"mode": "side_by_side_atomic_pointer", "retain_previous": True, "rollback_supported": True},
    }
    validate_manifest(manifest)
    args.output.write_bytes(canonical_json(manifest))
    signature = {
        "schema_version": 1,
        "algorithm": "ed25519",
        "key_id": args.key_id,
        "signature": sign_file(args.output, args.private_key),
    }
    args.signature_output.write_bytes(canonical_json(signature))
    os.chmod(args.output, 0o644)
    os.chmod(args.signature_output, 0o644)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--version", required=True)
    result.add_argument("--minimum-installer-version", required=True)
    result.add_argument("--source-archive", type=Path, required=True)
    result.add_argument("--installer-appimage", type=Path, required=True)
    result.add_argument("--base-url", required=True)
    result.add_argument("--release-page-url", required=True)
    result.add_argument("--repository", default="https://github.com/mexicantexan/studiocast")
    result.add_argument("--commit", required=True)
    result.add_argument("--workflow-run-url", required=True)
    result.add_argument("--private-key", type=Path, required=True)
    result.add_argument("--key-id", required=True)
    result.add_argument("--license-spdx", default="MPL-2.0")
    result.add_argument("--license-name", default="Mozilla Public License 2.0")
    result.add_argument("--license-url", default="https://www.mozilla.org/MPL/2.0/")
    result.add_argument("--generated-at")
    result.add_argument("--published-at")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--signature-output", type=Path, required=True)
    return result


if __name__ == "__main__":
    generate(parser().parse_args())
