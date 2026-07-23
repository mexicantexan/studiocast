"""Packaged release, model, and trusted-helper adapters."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import uuid
import zipfile
from pathlib import Path
from typing import Any

from .contracts import (
    BackendError,
    DEFAULT_MODEL_ARTIFACT_COUNT,
    DEFAULT_MODEL_PACKS,
    atomic_write,
    file_sha256,
    load_json,
    roots,
)

def release_module() -> Any:
    candidates = [
        Path(__file__).resolve().parent.parent / "release/release_channel.py",
        Path(__file__).resolve().parent.parent.parent / "release/release_channel.py",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise BackendError("release.verifier_unavailable", "Packaged release verifier is unavailable")
    spec = importlib.util.spec_from_file_location("studiocast_packaged_release_channel", path)
    if spec is None or spec.loader is None:
        raise BackendError("release.verifier_unavailable", "Could not load packaged release verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def trust_root() -> Path:
    installed = Path(__file__).resolve().parent.parent / "trust/keys"
    return installed if installed.is_dir() else Path(__file__).resolve().parent.parent.parent / "release/keys"


def release_keys(args: argparse.Namespace, rc: Any) -> Any:
    keys = {path.stem: path for path in trust_root().glob("*.pem") if path.is_file() and not path.is_symlink()}
    explicit = getattr(args, "trusted_key", []) or []
    if explicit:
        raise BackendError("release.signature.caller_key_forbidden",
                           "Recommended installation accepts only packaged production trust roots")
    if not keys:
        raise BackendError("release.signature.production_key_missing",
                           f"No production release public key is installed in {trust_root()}")
    return rc.TrustedKeyStore(keys)


def release_cache_dir(args: argparse.Namespace) -> Path:
    value = getattr(args, "release_cache_dir", None)
    return Path(value).absolute() if value else roots()["cache"] / "release"


def model_module() -> Any:
    candidates = [
        Path(__file__).resolve().parent.parent / "models/model_transactions.py",
        Path(__file__).resolve().parent.parent.parent / "models/model_transactions.py",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise BackendError("model.transaction_unavailable", "Packaged model transaction module is unavailable")
    spec = importlib.util.spec_from_file_location("studiocast_packaged_model_transactions", path)
    if spec is None or spec.loader is None:
        raise BackendError("model.transaction_unavailable", "Could not load packaged model transaction module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_component_root() -> Path:
    candidates = [Path(__file__).resolve().parent.parent.parent, Path(__file__).resolve().parents[3]]
    return next((candidate for candidate in candidates
                 if (candidate / "installer/models/curated-model-catalog-v1.json").is_file()), candidates[-1])


def privilege_client_module() -> Any | None:
    candidates = [
        Path(__file__).resolve().parent.parent / "privileged/client_contract.py",
        Path(__file__).resolve().parent.parent.parent / "privileged/client_contract.py",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("studiocast_packaged_privilege_client", path)
    if spec is None or spec.loader is None:
        return None
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if not callable(getattr(module, "preflight_trusted_helper", None)):
            return None
        return module
    except Exception:
        sys.modules.pop(spec.name, None)
        return None


def load_model_catalog() -> tuple[Any, dict[str, Any], Path, str]:
    module = model_module()
    root = model_component_root()
    path = root / "installer/models/curated-model-catalog-v1.json"
    try:
        catalog = module.load_catalog(path)
    except (OSError, module.CatalogError) as exc:
        code = getattr(exc, "reason_code", "model.catalog_unavailable")
        raise BackendError(code, str(exc)) from exc
    return module, catalog, root, file_sha256(path)


def catalog_size_summary(catalog: dict[str, Any], catalog_sha256: str) -> dict[str, Any]:
    """Return the bounded catalog evidence carried by analysis facts."""
    catalog_packs = {pack["id"]: pack for pack in catalog["packs"]}
    if any(pack_id not in catalog_packs for pack_id in DEFAULT_MODEL_PACKS):
        raise BackendError("model.catalog_missing_default",
                           "The packaged catalog omits a default model pack")
    packs: list[dict[str, Any]] = []
    for pack_id in DEFAULT_MODEL_PACKS:
        artifacts = catalog_packs[pack_id]["artifacts"]
        known = all(type(item.get("size_bytes")) is int and item["size_bytes"] > 0 and
                    item.get("size_status") == "known" for item in artifacts)
        packs.append({
            "id": pack_id,
            "artifact_count": len(artifacts),
            "download_bytes": (sum(item["size_bytes"] for item in artifacts) if known else None),
            "size_status": "known" if known else "unknown",
        })
    all_known = all(pack["size_status"] == "known" for pack in packs)
    return {
        "schema_version": 1,
        "catalog_version": catalog["catalog_version"],
        "catalog_sha256": catalog_sha256,
        "trusted": catalog["size_metadata"]["trusted"],
        "reason_code": catalog["size_metadata"]["reason_code"],
        "pack_count": len(packs),
        "artifact_count": sum(pack["artifact_count"] for pack in packs),
        "download_bytes": (sum(pack["download_bytes"] for pack in packs) if all_known else None),
        "packs": packs,
    }


def validated_catalog_size_summary(facts: dict[str, Any]) -> dict[str, Any] | None:
    """Validate only declared facts; never consult the packaged catalog."""
    models = facts.get("models")
    if not isinstance(models, dict) or models.get("default_pack_ids") != list(DEFAULT_MODEL_PACKS):
        return None
    summary = models.get("catalog_size_summary")
    required = {"schema_version", "catalog_version", "catalog_sha256", "trusted",
                "reason_code", "pack_count", "artifact_count", "download_bytes", "packs"}
    if not isinstance(summary, dict) or set(summary) != required:
        return None
    if (type(summary["schema_version"]) is not int or summary["schema_version"] != 1 or
            not isinstance(summary["catalog_version"], str) or not summary["catalog_version"] or
            not isinstance(summary["catalog_sha256"], str) or
            re.fullmatch(r"sha256:[0-9a-f]{64}", summary["catalog_sha256"]) is None or
            summary["trusted"] is not True or
            not isinstance(summary["reason_code"], str) or not summary["reason_code"].strip() or
            type(summary["pack_count"]) is not int or
            summary["pack_count"] != len(DEFAULT_MODEL_PACKS) or
            type(summary["artifact_count"]) is not int or
            summary["artifact_count"] != DEFAULT_MODEL_ARTIFACT_COUNT or
            type(summary["download_bytes"]) is not int or summary["download_bytes"] <= 0):
        return None
    packs = summary["packs"]
    if not isinstance(packs, list) or len(packs) != summary["pack_count"]:
        return None
    expected_pack_fields = {"id", "artifact_count", "download_bytes", "size_status"}
    for expected_id, pack in zip(DEFAULT_MODEL_PACKS, packs):
        if (not isinstance(pack, dict) or set(pack) != expected_pack_fields or
                pack.get("id") != expected_id or
                type(pack.get("artifact_count")) is not int or pack["artifact_count"] <= 0 or
                type(pack.get("download_bytes")) is not int or pack["download_bytes"] <= 0 or
                pack.get("size_status") != "known"):
            return None
    if (sum(pack["artifact_count"] for pack in packs) != summary["artifact_count"] or
            sum(pack["download_bytes"] for pack in packs) != summary["download_bytes"]):
        return None
    return summary


def model_download_bytes_from_facts(facts: dict[str, Any], selected_pack_ids: Any) -> int | None:
    summary = validated_catalog_size_summary(facts)
    if summary is None or not isinstance(selected_pack_ids, list):
        return None
    if (any(not isinstance(pack_id, str) for pack_id in selected_pack_ids) or
            len(selected_pack_ids) != len(set(selected_pack_ids))):
        return None
    pack_sizes = {pack["id"]: pack["download_bytes"] for pack in summary["packs"]}
    if any(pack_id not in pack_sizes for pack_id in selected_pack_ids):
        return None
    return sum(pack_sizes[pack_id] for pack_id in selected_pack_ids)


def verify_release_evidence(args: argparse.Namespace) -> dict[str, Any] | None:
    receipt_path = getattr(args, "release_receipt", None)
    manifest_path = getattr(args, "release_manifest", None)
    signature_path = getattr(args, "release_signature", None)
    stable_url = getattr(args, "stable_url", None)
    offline = bool(getattr(args, "offline", False))
    if (offline and (getattr(args, "source_dir", None) or
                     (getattr(args, "release_archive", None) and not
                      any((receipt_path, manifest_path, stable_url))))):
        return None
    if not any((receipt_path, manifest_path, stable_url, offline)):
        return None
    rc = release_module()
    keys = release_keys(args, rc)
    allow_file = os.environ.get("STUDIOCAST_INSTALLER_TEST_MODE") == "1"
    receipt_input: dict[str, Any] | None = None
    if receipt_path:
        receipt_input = load_json(Path(receipt_path))
        if (receipt_input.get("schema_version") != 1 or
                receipt_input.get("receipt_version") != "studiocast-verified-release/v1"):
            raise BackendError("release.receipt.schema", "Unsupported verified-release receipt")
        manifest_path = receipt_input.get("manifest_path")
        signature_path = receipt_input.get("signature_path")
        if not getattr(args, "release_archive", None):
            args.release_archive = (receipt_input.get("archive_candidate_path")
                if "archive_candidate_path" in receipt_input else receipt_input.get("archive_path"))
        offline = offline or receipt_input.get("offline") is True
        args.offline = offline
    if manifest_path or signature_path:
        if not manifest_path or not signature_path:
            raise BackendError("release.manifest.arguments", "Manifest and signature paths must be supplied together")
        manifest_file, signature_file = Path(manifest_path).absolute(), Path(signature_path).absolute()
        try:
            manifest = rc.verify_manifest(manifest_file.read_bytes(), signature_file.read_bytes(), keys,
                                          allow_file_urls=allow_file)
        except (OSError, rc.ReleaseChannelError) as exc:
            code = getattr(exc, "code", "release.manifest.unavailable")
            raise BackendError(code, str(exc)) from exc
    else:
        cache = release_cache_dir(args) / "manifests"
        manifest_url = stable_url or rc.DEFAULT_STABLE_MANIFEST_URL
        signature_url = getattr(args, "stable_signature_url", None) or manifest_url + ".sig"
        try:
            manifest = rc.fetch_release_manifest(cache, keys, manifest_url=manifest_url,
                                                 signature_url=signature_url, offline=offline,
                                                 allow_file_urls=allow_file)
        except rc.ReleaseChannelError as exc:
            raise BackendError(exc.code, str(exc)) from exc
        generation = (cache / "current").read_text(encoding="ascii").strip()
        manifest_file = cache / generation / "manifest.json"
        signature_file = cache / generation / "manifest.json.sig"
    source = manifest["artifacts"]["source_archive"]
    archive_arg = getattr(args, "release_archive", None)
    archive: Path | None = None
    if archive_arg:
        archive = Path(archive_arg).absolute()
        try:
            rc.verify_artifact(archive, source, keys)
        except rc.ReleaseChannelError as exc:
            raise BackendError(exc.code, str(exc)) from exc
    signature_envelope = rc.strict_json_loads(signature_file.read_bytes())
    cache_path = release_cache_dir(args) / "artifacts" / source["filename"]
    cached_verified = False
    if archive is None and cache_path.is_file() and not cache_path.is_symlink():
        try:
            rc.verify_artifact(cache_path, source, keys)
            cached_verified = True
        except rc.ReleaseChannelError:
            cached_verified = False
    receipt = {
        "schema_version": 1, "receipt_version": "studiocast-verified-release/v1",
        "channel": manifest["channel"], "version": manifest["release"]["version"],
        "minimum_installer_version": manifest["minimum_installer_version"],
        "manifest_path": str(manifest_file), "signature_path": str(signature_file),
        "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "manifest_key_id": signature_envelope["key_id"],
        "archive_path": str(cache_path),
        "archive_candidate_path": str(archive) if archive is not None else None,
        "archive_state": "candidate_verified" if archive is not None else "acquisition_pending",
        "archive_cache_state": "cached_verified" if cached_verified else "absent_or_unverified",
        "archive_sha256": source["sha256"], "archive_size": source["size_bytes"],
        "archive_filename": source["filename"],
        "archive_signature": copy.deepcopy(source["signature"]),
        "archive_url": source["url"], "provenance": copy.deepcopy(source["provenance"]),
        "license": copy.deepcopy(source["license"]),
        "installer_appimage": copy.deepcopy(manifest["artifacts"]["installer_appimage"]),
        "release_page_url": manifest["release"]["page_url"],
        "offline": offline,
    }
    # A receipt never substitutes for verification: receipt input reaches this
    # point only after manifest, signature, archive size/hash/signature are all
    # revalidated against the current trusted key store.
    out = getattr(args, "release_receipt_out", None)
    if out:
        atomic_write(Path(out).absolute(), receipt)
    args.release_archive = str(archive) if archive is not None else None
    args.target_version = receipt["version"]
    args.verified_release = receipt
    return receipt
