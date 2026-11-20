"""Installed manifest and owned host-configuration contracts."""

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
    POLICY_VERSION,
    V4L_OWNED_PATHS,
    file_sha256,
    load_json,
    utc_now,
)

def manifest_state(path: Path) -> tuple[str, dict[str, Any] | None, list[str]]:
    if not path.exists():
        return "absent", None, ["install.manifest.absent"]
    try:
        value = load_json(path)
    except BackendError as exc:
        return "unhealthy", None, ["install.manifest.corrupt", exc.code]
    schema = value.get("schema_version")
    if schema not in {1, 2}:
        return "unknown_schema", value, ["install.manifest.unknown_schema"]
    if schema == 1:
        if value.get("state") == "uninstalled":
            return "tombstone", value, ["install.manifest.healthy", "install.state.tombstone"]
        return "partial", value, ["install.manifest.legacy_v1"]
    tx_state = value.get("transaction", {}).get("state")
    if tx_state == "uninstalled":
        return "tombstone", value, ["install.manifest.healthy", "install.state.tombstone"]
    active = value.get("payloads", {}).get("active")
    if tx_state == "degraded" and active:
        return "partial", value, ["install.manifest.healthy", "install.state.degraded"]
    if not active:
        return "partial", value, ["install.payload.missing"]
    active_path = Path(active.get("path", ""))
    if not active_path.is_dir():
        return "stale", value, ["install.payload.missing"]
    return "healthy", value, ["install.manifest.healthy", "install.payload.healthy"]

def validated_v4l_ownership(manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not manifest or manifest.get("schema_version") != 2:
        return []
    recorded = manifest.get("v4l", {}).get("owned_configuration", [])
    if not isinstance(recorded, list):
        return []
    validated: list[dict[str, Any]] = []
    for item in recorded:
        if not isinstance(item, dict) or item.get("id") not in V4L_OWNED_PATHS:
            continue
        expected_path = V4L_OWNED_PATHS[item["id"]]
        expected_hash = item.get("sha256")
        try:
            info = expected_path.lstat()
            actual_hash = file_sha256(expected_path) if stat.S_ISREG(info.st_mode) else None
        except OSError:
            continue
        if str(expected_path) == item.get("path") and expected_hash == actual_hash:
            validated.append({"id": item["id"], "path": str(expected_path), "sha256": actual_hash})
    return validated

def migrate_v1(value: dict[str, Any], r: dict[str, Path]) -> dict[str, Any]:
    if value.get("schema_version") != 1:
        raise BackendError("manifest.not_v1", "Manifest is not schema v1")
    state = "uninstalled" if value.get("state") == "uninstalled" else "prepared"
    version = value.get("installed_version")
    return {"schema_version": 2, "policy_version": POLICY_VERSION,
        "transaction": {"id": str(uuid.uuid4()), "state": state, "intent": "migration", "started_at": utc_now(),
            "updated_at": utc_now(), "completed_at": None, "plan_id": None, "plan_digest": None},
        "install_scope": "user", "versions": {"active": version if state != "uninstalled" else None, "previous": None, "target": version, "relation_at_start": "unknown"},
        "payloads": {"root": str(r["data"] / "payloads"), "current_link": str(r["data"] / "current"), "active": None, "previous": None, "target": None},
        "source": {"channel": "legacy", "official": False, "legacy_hints": {"source_path": value.get("source_path"), "build_path": value.get("build_path")}},
        "desired_configuration": {"build_type": "unknown", "features": {}, "service": {"desired": "unknown"},
            "v4l": {"desired": "unknown"}, "model_pack_ids": [], "preserve_settings": True},
        "effective_build_configuration": {"cmake": {}}, "service": {"desired": "unknown", "actual": "unknown"},
        "v4l": {"desired": "unknown", "actual": {}, "owned_configuration": []}, "models": {"packs": {}},
        "ownership": {"application_files": [], "links": [], "system_configuration": [], "shared_dependencies_observed": []},
        "preservation": {"preserve_settings": True, "restored": False},
        "health": {"core": "unknown", "overall": "migration_required", "validation": [{"code": "manifest.v1.paths_untrusted"}]}, "journal": []}
