"""StudioCast user-local installer transaction backend.

The installed copy is deliberately self-contained.  Host mutations are never
performed by this program: they are represented as typed requests for the
separately packaged, root-owned StudioCast system helper.
"""

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


POLICY_VERSION = "studiocast-installer-policy/1"
FACTS_VERSION = "installer-facts/v1"
RECOMMENDATION_VERSION = "installer-recommendation/v1"
PLAN_VERSION = "installer-plan/v1"
RESULT_VERSION = "installer-result/v1"
PROGRESS_VERSION = "installer-progress/v1"
HELPER_PATH = Path("/usr/libexec/studiocast/studiocast-system-helper")
PKEXEC_PATH = Path("/usr/bin/pkexec")
POLKIT_POLICY_PATH = Path("/usr/share/polkit-1/actions/com.studiocast.Installer1.policy")
SYSTEM_BUS_PATH = Path("/run/dbus/system_bus_socket")
POLKIT_SERVICE_PATH = Path("/usr/share/dbus-1/system-services/org.freedesktop.PolicyKit1.service")
SERVICE_NAME = "studiocastd.service"
BINARIES = (
    "studiocast", "studiocastd", "studiocastctl", "studiocast-probe",
    "studiocast-open", "studiocast-maxine", "studiocast-video", "studiocast-audio",
)
DEFAULT_MODEL_PACKS = (
    "fastenhancer_s_vd_v1", "fastenhancer_m_vd_v1",
    "modnet-webnn-256-fp32", "yunet_opencv_zoo_2023mar_fp32",
    "dlib_68_ibug_300w", "gaze_correction_cam_flx_v0_1_1",
    "fastdvdnet_sigma15",
)
DEFAULT_MODEL_ARTIFACT_COUNT = 8
ALLOWED_OPERATION_KINDS = {
    "preflight", "settings_snapshot", "release_acquire", "source_extract", "build_cache_prepare",
    "build_configure", "build_compile", "payload_stage", "payload_validate",
    "trusted_helper", "payload_activate", "links_reconcile", "service_unit_reconcile",
    "service_state_reconcile", "model_transaction",
    "settings_restore", "cleanup", "manifest_commit", "uninstall_tombstone",
}
PRIVILEGED_TYPES = {
    "packages.ensure.v1", "v4l.module.load.v1", "v4l.persistence.write.v1",
    "v4l.persistence.remove.v1",
}
FORBIDDEN_HELPER_FIELDS = {
    "command", "executable", "argv", "shell", "script", "script_path",
    "source_path", "archive_path", "build_path", "destination_path", "cwd",
    "environment", "url", "download", "file_content", "module_name",
    "package_manager_flags",
}
PRIVILEGED_INPUT_FIELDS = {
    "packages.ensure.v1": frozenset(("operation_type", "packages")),
    "v4l.module.load.v1": frozenset(("operation_type", "device_number", "label", "exclusive_caps")),
    "v4l.persistence.write.v1": frozenset(("operation_type", "device_number", "label", "exclusive_caps")),
    "v4l.persistence.remove.v1": frozenset(("operation_type", "expected_hashes")),
}
V4L_OWNED_PATHS = {
    "v4l_modules_load": Path("/etc/modules-load.d/studiocast-v4l2loopback.conf"),
    "v4l_modprobe": Path("/etc/modprobe.d/studiocast-v4l2loopback.conf"),
}
CANONICAL_EFFECTS = (
    "mirror", "virtual_background.blur", "virtual_background.remove",
    "virtual_background.replace", "auto_frame", "eye_contact",
    "video_noise_removal", "virtual_key_light", "vignette",
)

class BackendError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise BackendError("json.duplicate_key", f"Duplicate JSON key: {key}")
        out[key] = value
    return out


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=reject_duplicates)
    except BackendError:
        raise
    except json.JSONDecodeError as exc:
        raise BackendError("json.corrupt", f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise BackendError("file.read_failed", f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BackendError("json.type", f"Expected a JSON object in {path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def atomic_write(path: Path, value: dict[str, Any], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        dirfd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def roots() -> dict[str, Path]:
    home_text = os.environ.get("HOME")
    if not home_text:
        raise BackendError("path.home.unset", "HOME is not set")
    home = Path(home_text).absolute()
    data_home = Path(os.environ.get("XDG_DATA_HOME", home / ".local/share")).absolute()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")).absolute()
    state_home = Path(os.environ.get("XDG_STATE_HOME", home / ".local/state")).absolute()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache")).absolute()
    return {
        "home": home, "data": data_home / "studiocast", "config": config_home / "studiocast",
        "state": state_home / "studiocast", "cache": cache_home / "studiocast",
        "bin": home / ".local/bin", "systemd": config_home / "systemd/user",
    }


def service_unit_content(data_root: Path) -> str:
    executable = str(data_root / "current/bin/studiocastd")
    if any(character in executable for character in ("\x00", "\n", "\r")):
        raise BackendError("service.path_invalid", "Service executable path contains control characters")
    escaped = executable.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("%", "%%").replace("$", "$$")
    return ("[Unit]\nDescription=StudioCast daemon\n[Service]\n"
            f'ExecStart="{escaped}"\nRestart=on-failure\n'
            "[Install]\nWantedBy=default.target\n")


def is_beneath(path: Path, root: Path, allow_root: bool = False) -> bool:
    try:
        rel = path.absolute().relative_to(root.absolute())
    except ValueError:
        return False
    return allow_root or bool(rel.parts)


def validate_mutation_path(path: Path, allowed_roots: list[Path], allow_root: bool = False) -> None:
    if not path.is_absolute() or ".." in path.parts or str(path) == "/":
        raise BackendError("path.unsafe", f"Unsafe mutation path: {path}")
    if not any(is_beneath(path, root, allow_root) for root in allowed_roots):
        raise BackendError("path.outside_allowed_root", f"Path is outside StudioCast roots: {path}")
    current = Path(path.anchor)
    for component in path.parts[1:-1]:
        current /= component
        if current.is_symlink():
            raise BackendError("path.symlink_parent", f"Symlink parent rejected: {current}")


def safe_remove(path: Path, allowed_roots: list[Path], *, allow_root: bool = False) -> None:
    validate_mutation_path(path, allowed_roots, allow_root)
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def read_version(source: Path | None) -> str:
    if source:
        try:
            value = (source / "VERSION").read_text(encoding="utf-8").strip()
            if value:
                return value
        except OSError:
            pass
    return "unknown"


def installer_version() -> str:
    candidates = [Path(__file__).resolve().parents[3] / "VERSION"]
    for candidate in candidates:
        try:
            value = candidate.read_text(encoding="utf-8").strip()
            if value:
                return value
        except OSError:
            pass
    return "unknown"


def semver_tuple(value: str | None) -> tuple[int, int, int, tuple[Any, ...]] | None:
    if not value:
        return None
    match = re.fullmatch(r"v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?", value)
    if not match:
        return None
    pre: tuple[Any, ...] = ()
    if match.group(4):
        pre = tuple(int(x) if x.isdigit() else x for x in match.group(4).split("."))
    else:
        pre = (float("inf"),)
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), pre


def version_relation(active: str | None, target: str | None) -> str:
    if not active:
        return "not_installed"
    a, b = semver_tuple(active), semver_tuple(target)
    if not a or not b:
        return "unknown"
    if a == b:
        return "same"
    return "upgrade" if a < b else "installed_newer"
