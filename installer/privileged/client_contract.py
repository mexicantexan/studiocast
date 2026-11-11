"""Unprivileged client-side trust and authorization result contract."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


FIXED_HELPER_PATH = Path("/usr/libexec/studiocast/studiocast-system-helper")
FIXED_PKEXEC_PATH = Path("/usr/bin/pkexec")
FIXED_POLICY_PATH = Path(
    "/usr/share/polkit-1/actions/com.studiocast.Installer1.policy"
)
FIXED_SYSTEM_BUS_PATH = Path("/run/dbus/system_bus_socket")
FIXED_POLKIT_SERVICE_PATH = Path(
    "/usr/share/dbus-1/system-services/org.freedesktop.PolicyKit1.service"
)
POLKIT_ACTION_ID = "com.studiocast.Installer1.manage-host-integration"
POLKIT_EXEC_ANNOTATION = "org.freedesktop.policykit.exec.path"
EXPECTED_HELPER_VERSION = "1.0.0"
SUPPORTED_PROTOCOL = 1
VERSION_PROBE_TIMEOUT_SECONDS = 2

AUTHORIZATION_REASONS = {
    "granted": "authorization_granted",
    "unavailable": "authorization_unavailable",
    "denied": "authorization_denied",
    "cancelled": "authorization_cancelled",
    "timeout": "authorization_timeout",
}


def authorization_record(status: str) -> dict[str, str]:
    if status not in AUTHORIZATION_REASONS:
        raise ValueError("unknown authorization outcome")
    return {"status": status, "reason_code": AUTHORIZATION_REASONS[status]}


@dataclass(frozen=True)
class HelperTrustResult:
    trusted: bool
    reason_code: str
    message: str


@dataclass(frozen=True)
class HelperPreflightResult:
    path: str
    present: bool
    trusted: bool
    compatible: bool
    authorization_transport_available: bool
    helper_version: str | None
    protocol_versions: tuple[int, ...]
    pkexec_path: str
    policy_path: str
    reason_code: str

    def as_dict(self) -> dict[str, object]:
        """Return the stable installer-facts representation."""
        return {
            "path": self.path,
            "present": self.present,
            "trusted": self.trusted,
            "compatible": self.compatible,
            "authorization_transport_available": self.authorization_transport_available,
            "helper_version": self.helper_version,
            "protocol_versions": list(self.protocol_versions),
            "pkexec_path": self.pkexec_path,
            "policy_path": self.policy_path,
            "reason_codes": [self.reason_code],
        }


def _trusted_file(path: Path, required_uid: int, *, executable: bool) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        return False
    if info.st_uid != required_uid or info.st_mode & 0o022:
        return False
    return not executable or bool(info.st_mode & 0o111)


def _default_version_probe(path: Path) -> str:
    completed = subprocess.run(
        [str(path), "--version", "--json"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=VERSION_PROBE_TIMEOUT_SECONDS,
        check=False,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
    )
    if completed.returncode != 0 or len(completed.stdout.encode("utf-8")) > 4096:
        raise OSError("helper version probe failed")
    return completed.stdout


def _parse_version(raw: str) -> tuple[str, tuple[int, ...]] | None:
    try:
        version = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(version, dict)
        or set(version) != {"helper_version", "protocol_versions"}
        or version.get("helper_version") != EXPECTED_HELPER_VERSION
        or version.get("protocol_versions") != [SUPPORTED_PROTOCOL]
    ):
        return None
    return version["helper_version"], tuple(version["protocol_versions"])


def _policy_binds_fixed_helper(path: Path, helper_path: Path) -> bool:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return False
    for action in root.findall("action"):
        if action.get("id") != POLKIT_ACTION_ID:
            continue
        annotations = {
            item.get("key"): (item.text or "").strip()
            for item in action.findall("annotate")
        }
        return (
            annotations.get(POLKIT_EXEC_ANNOTATION) == str(helper_path)
            and annotations.get("org.freedesktop.policykit.exec.allow_gui") == "true"
        )
    return False


def _system_bus_available(path: Path) -> bool:
    try:
        return stat.S_ISSOCK(os.lstat(path).st_mode)
    except OSError:
        return False


def preflight_trusted_helper(
    *,
    helper_path: Path = FIXED_HELPER_PATH,
    pkexec_path: Path = FIXED_PKEXEC_PATH,
    policy_path: Path = FIXED_POLICY_PATH,
    system_bus_path: Path = FIXED_SYSTEM_BUS_PATH,
    polkit_service_path: Path = FIXED_POLKIT_SERVICE_PATH,
    version_probe: Callable[[Path], str] | None = None,
    required_uid: int = 0,
    test_mode: bool = False,
) -> HelperPreflightResult:
    """Read-only helper/polkit readiness check that never requests authorization."""
    helper_path = Path(helper_path)
    pkexec_path = Path(pkexec_path)
    policy_path = Path(policy_path)
    system_bus_path = Path(system_bus_path)
    polkit_service_path = Path(polkit_service_path)

    fixed_paths = (
        helper_path == FIXED_HELPER_PATH,
        pkexec_path == FIXED_PKEXEC_PATH,
        policy_path == FIXED_POLICY_PATH,
        system_bus_path == FIXED_SYSTEM_BUS_PATH,
        polkit_service_path == FIXED_POLKIT_SERVICE_PATH,
    )
    present = helper_path.exists() and not helper_path.is_symlink()

    def result(
        reason_code: str,
        *,
        trusted: bool = False,
        compatible: bool = False,
        transport: bool = False,
        helper_version: str | None = None,
        protocols: tuple[int, ...] = (),
    ) -> HelperPreflightResult:
        return HelperPreflightResult(
            path=str(helper_path),
            present=present,
            trusted=trusted,
            compatible=compatible,
            authorization_transport_available=transport,
            helper_version=helper_version,
            protocol_versions=protocols,
            pkexec_path=str(pkexec_path),
            policy_path=str(policy_path),
            reason_code=reason_code,
        )

    if not test_mode and not all(fixed_paths):
        return result("privileged_helper_untrusted")
    try:
        os.lstat(helper_path)
    except FileNotFoundError:
        return result("privileged_helper_missing")
    except OSError:
        return result("privileged_helper_untrusted")
    if not _trusted_file(helper_path, required_uid, executable=True):
        return result("privileged_helper_untrusted")

    probe = version_probe or _default_version_probe
    try:
        parsed_version = _parse_version(probe(helper_path))
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        parsed_version = None
    if parsed_version is None:
        return result("privileged_helper_incompatible")
    helper_version, protocols = parsed_version

    transport_available = (
        _trusted_file(pkexec_path, required_uid, executable=True)
        and _trusted_file(policy_path, required_uid, executable=False)
        and _policy_binds_fixed_helper(policy_path, helper_path)
        and _system_bus_available(system_bus_path)
        and _trusted_file(polkit_service_path, required_uid, executable=False)
    )
    if not transport_available:
        return result(
            "authorization_transport_unavailable",
            compatible=True,
            helper_version=helper_version,
            protocols=protocols,
        )
    return result(
        "privileged_helper_usable",
        trusted=True,
        compatible=True,
        transport=True,
        helper_version=helper_version,
        protocols=protocols,
    )


def probe_helper(
    path: Path = FIXED_HELPER_PATH,
    *,
    version_probe: Callable[[Path], str],
    required_uid: int = 0,
    test_mode: bool = False,
) -> HelperTrustResult:
    if not test_mode and path != FIXED_HELPER_PATH:
        return HelperTrustResult(False, "privileged_helper_untrusted", "Helper path is not fixed.")
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return HelperTrustResult(False, "privileged_helper_missing", "System helper package is not installed.")
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        return HelperTrustResult(False, "privileged_helper_untrusted", "Helper is not a regular file.")
    if info.st_uid != required_uid or info.st_mode & 0o022:
        return HelperTrustResult(False, "privileged_helper_untrusted", "Helper ownership or permissions are unsafe.")
    try:
        raw = version_probe(path)
        version = json.loads(raw)
    except (OSError, ValueError, json.JSONDecodeError):
        return HelperTrustResult(False, "privileged_helper_incompatible", "Helper version probe failed.")
    if (
        not isinstance(version, dict)
        or set(version) != {"helper_version", "protocol_versions"}
        or version.get("helper_version") != EXPECTED_HELPER_VERSION
        or version.get("protocol_versions") != [SUPPORTED_PROTOCOL]
    ):
        return HelperTrustResult(False, "privileged_helper_incompatible", "Helper protocol is incompatible.")
    return HelperTrustResult(True, "helper_trusted", "Trusted system helper is available.")
