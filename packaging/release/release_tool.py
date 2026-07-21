#!/usr/bin/env python3
"""Generate and sign StudioCast release-channel metadata.

Private keys are accepted only as explicit command-line paths.  This repository
contains no private key and the tool never generates or persists one.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _load_release_channel():
    repository_root = Path(__file__).resolve().parents[2]
    expected_directory = (repository_root / "installer/release").resolve(strict=True)
    path = repository_root / "installer/release/release_channel.py"
    resolved = path.resolve(strict=True)
    if path.is_symlink() or resolved.parent != expected_directory or not resolved.is_file():
        raise RuntimeError("canonical installer release contract is missing or unsafe")
    spec = importlib.util.spec_from_file_location("studiocast_installer_release_channel", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load canonical installer release contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_release_channel = _load_release_channel()
MANIFEST_VERSION = _release_channel.MANIFEST_VERSION
canonical_json = _release_channel.canonical_json
sha256_file = _release_channel.sha256_file
validate_manifest = _release_channel.validate_manifest


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


def atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate(args: argparse.Namespace) -> None:
    destinations = {args.output.resolve(), args.signature_output.resolve()}
    protected_inputs = {
        args.source_archive.resolve(),
        args.installer_appimage.resolve(),
        args.private_key.resolve(),
    }
    if len(destinations) != 2:
        raise SystemExit("manifest and signature outputs must be different files")
    if destinations & protected_inputs:
        raise SystemExit("release outputs must not overwrite an artifact or private key")
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
    raw_manifest = canonical_json(manifest)
    with tempfile.TemporaryDirectory(prefix="studiocast-release-manifest-") as temporary:
        staged_manifest = Path(temporary) / "manifest.json"
        staged_manifest.write_bytes(raw_manifest)
        manifest_signature = sign_file(staged_manifest, args.private_key)
    signature = {
        "schema_version": 1,
        "algorithm": "ed25519",
        "key_id": args.key_id,
        "signature": manifest_signature,
    }
    # Both signatures are complete before either previous output is replaced.
    # Write the detached signature first so the manifest remains the commit point.
    atomic_write(args.signature_output, canonical_json(signature))
    atomic_write(args.output, raw_manifest)


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
