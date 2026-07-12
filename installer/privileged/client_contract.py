"""Unprivileged client-side trust and authorization result contract."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


FIXED_HELPER_PATH = Path("/usr/libexec/studiocast/studiocast-system-helper")
EXPECTED_HELPER_VERSION = "1.0.0"
SUPPORTED_PROTOCOL = 1

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
