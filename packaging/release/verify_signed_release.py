#!/usr/bin/env python3
"""Hermetically verify a generated StudioCast release and its signing identity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from release_channel import (
    ReleaseChannelError,
    TrustedKeyStore,
    strict_json_loads,
    verify_artifact,
    verify_manifest,
)


def _require_key_id(value: object, expected: str, where: str) -> None:
    if value != expected:
        raise ReleaseChannelError(
            "release.signature.key_id_mismatch",
            f"{where} key ID does not match packaged trust root",
        )


def verify_release(args: argparse.Namespace) -> dict[str, object]:
    if args.public_key.is_symlink() or not args.public_key.is_file():
        raise ReleaseChannelError(
            "release.signature.production_key_missing",
            "committed release public key is missing or unsafe",
        )

    raw_manifest = args.manifest.read_bytes()
    raw_signature = args.signature.read_bytes()
    signature_envelope = strict_json_loads(raw_signature)
    if not isinstance(signature_envelope, dict):
        raise ReleaseChannelError(
            "release.signature.envelope", "detached manifest signature must be an object"
        )
    _require_key_id(signature_envelope.get("key_id"), args.key_id, "manifest")

    keys = TrustedKeyStore({args.key_id: args.public_key})
    manifest = verify_manifest(raw_manifest, raw_signature, keys)

    verified: dict[str, dict[str, object]] = {}
    for artifact_name, artifact_path in (
        ("source_archive", args.source_archive),
        ("installer_appimage", args.installer_appimage),
    ):
        metadata = manifest["artifacts"][artifact_name]
        _require_key_id(metadata["signature"]["key_id"], args.key_id, artifact_name)
        if metadata["filename"] != artifact_path.name:
            raise ReleaseChannelError(
                "release.artifact.filename_mismatch",
                f"{artifact_name} filename does not match the signed manifest",
            )
        verify_artifact(artifact_path, metadata, keys)
        verified[artifact_name] = {
            "filename": artifact_path.name,
            "sha256": metadata["sha256"],
            "size_bytes": metadata["size_bytes"],
            "key_id": metadata["signature"]["key_id"],
        }

    return {
        "schema_version": 1,
        "verification": "studiocast-signed-release/v1",
        "key_id": args.key_id,
        "version": manifest["release"]["version"],
        "manifest_key_id": signature_envelope["key_id"],
        "artifacts": verified,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a signed release manifest and both release artifacts."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--installer-appimage", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--key-id", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        result = verify_release(parse_args())
    except (OSError, ReleaseChannelError, ValueError) as exc:
        code = exc.code if isinstance(exc, ReleaseChannelError) else "release.verification.failed"
        print(json.dumps({"error": {"code": code, "message": str(exc)}}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
