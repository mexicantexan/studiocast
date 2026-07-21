#!/usr/bin/env python3
"""Verified StudioCast stable release and installer self-update primitives.

This module intentionally performs no host mutation beyond a caller-selected cache
directory.  Network transport is injected; the default downloader supports HTTPS
and local ``file:`` fixtures, while offline mode never opens a URL.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple


DEFAULT_STABLE_MANIFEST_URL = (
    "https://github.com/mexicantexan/studiocast/releases/latest/download/"
    "studiocast-release-manifest.json"
)
DEFAULT_STABLE_SIGNATURE_URL = DEFAULT_STABLE_MANIFEST_URL + ".sig"
MANIFEST_VERSION = "studiocast-release-manifest/v1"
SUPPORTED_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KEY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SEMVER_RE = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


class ReleaseChannelError(RuntimeError):
    """Structured release-channel failure with a stable reason code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _reject_duplicate_pairs(pairs: Iterable[Tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseChannelError("release.json.duplicate_key", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(raw: bytes | str) -> Any:
    """Parse UTF-8 JSON while rejecting duplicate keys and non-finite numbers."""

    try:
        text = raw.decode("utf-8", "strict") if isinstance(raw, bytes) else raw
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReleaseChannelError("release.json.non_finite", f"invalid JSON number: {value}")
            ),
        )
    except ReleaseChannelError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseChannelError("release.json.invalid", f"invalid release JSON: {exc}") from exc


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _expect_object(
    value: Any,
    where: str,
    required: set[str],
    optional: Optional[set[str]] = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseChannelError("release.manifest.schema", f"{where} must be an object")
    keys = set(value)
    missing = required - keys
    unknown = keys - required - (optional or set())
    if missing or unknown:
        raise ReleaseChannelError(
            "release.manifest.schema",
            f"{where} fields invalid (missing={sorted(missing)}, unknown={sorted(unknown)})",
        )
    return value


def _expect_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseChannelError("release.manifest.schema", f"{where} must be a non-empty string")
    return value


def _expect_timestamp(value: Any, where: str) -> str:
    timestamp = _expect_string(value, where)
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", timestamp):
        raise ReleaseChannelError("release.manifest.schema", f"{where} must be a UTC RFC 3339 timestamp")
    try:
        dt.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ReleaseChannelError("release.manifest.schema", f"{where} is not a valid UTC timestamp") from exc
    return timestamp


def _validate_https_url(value: Any, where: str, *, allow_file: bool) -> str:
    url = _expect_string(value, where)
    parsed = urllib.parse.urlparse(url)
    allowed = {"https"} | ({"file"} if allow_file else set())
    if parsed.scheme not in allowed or (parsed.scheme == "https" and not parsed.netloc):
        raise ReleaseChannelError("release.manifest.url", f"{where} must use {sorted(allowed)}")
    if parsed.username or parsed.password or parsed.fragment:
        raise ReleaseChannelError("release.manifest.url", f"{where} contains forbidden URL components")
    return url


@dataclasses.dataclass(frozen=True)
class SemVer:
    major: int
    minor: int
    patch: int
    prerelease: tuple[str, ...] = ()
    build: tuple[str, ...] = ()

    @classmethod
    def parse(cls, text: str) -> "SemVer":
        match = _SEMVER_RE.fullmatch(text)
        if not match:
            raise ReleaseChannelError("release.version.invalid", f"invalid semantic version: {text}")
        prerelease = tuple(match.group(4).split(".")) if match.group(4) else ()
        for item in prerelease:
            if item.isdigit() and len(item) > 1 and item.startswith("0"):
                raise ReleaseChannelError("release.version.invalid", f"invalid numeric prerelease: {text}")
        build = tuple(match.group(5).split(".")) if match.group(5) else ()
        return cls(int(match.group(1)), int(match.group(2)), int(match.group(3)), prerelease, build)

    def _precedence(self) -> tuple[Any, ...]:
        parts: list[tuple[int, Any]] = []
        for identifier in self.prerelease:
            parts.append((0, int(identifier)) if identifier.isdigit() else (1, identifier))
        return self.major, self.minor, self.patch, not self.prerelease, tuple(parts)

    def compare(self, other: "SemVer") -> int:
        left, right = self._precedence(), other._precedence()
        return (left > right) - (left < right)


@dataclasses.dataclass(frozen=True)
class UpdateDecision:
    action: str
    relation: str
    reason_code: str


def decide_update(
    installed_version: Optional[str],
    available_version: str,
    *,
    allow_same_version: bool = False,
    allow_downgrade: bool = False,
) -> UpdateDecision:
    available = SemVer.parse(available_version)
    if installed_version is None:
        return UpdateDecision("install", "not_installed", "release.install.available")
    relation = SemVer.parse(installed_version).compare(available)
    if relation < 0:
        return UpdateDecision("update", "upgrade", "release.update.available")
    if relation == 0:
        if allow_same_version:
            return UpdateDecision("reinstall", "same", "release.same.explicit_reinstall")
        return UpdateDecision("keep_current", "same", "release.same.no_update")
    if allow_downgrade:
        return UpdateDecision("downgrade", "downgrade", "release.downgrade.explicit")
    return UpdateDecision("keep_current", "installed_newer", "release.downgrade.blocked")


def installer_meets_minimum(installer_version: str, minimum_version: str) -> bool:
    return SemVer.parse(installer_version).compare(SemVer.parse(minimum_version)) >= 0


@dataclasses.dataclass(frozen=True)
class TrustedKeyStore:
    keys: Mapping[str, Path]

    def resolve(self, key_id: str) -> Path:
        if not _KEY_ID_RE.fullmatch(key_id) or key_id not in self.keys:
            raise ReleaseChannelError("release.signature.unknown_key", f"untrusted signing key: {key_id}")
        path = Path(self.keys[key_id])
        if not path.is_file():
            raise ReleaseChannelError("release.signature.key_unavailable", f"public key unavailable: {key_id}")
        return path


def verify_ed25519(data: bytes, signature_b64: str, key_id: str, keys: TrustedKeyStore) -> None:
    with tempfile.TemporaryDirectory(prefix="studiocast-signature-data-") as temp:
        data_path = Path(temp) / "content"
        data_path.write_bytes(data)
        _verify_ed25519_path(data_path, signature_b64, key_id, keys)


def _verify_ed25519_path(data_path: Path, signature_b64: str, key_id: str, keys: TrustedKeyStore) -> None:
    try:
        signature = base64.b64decode(signature_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ReleaseChannelError("release.signature.invalid_encoding", "signature is not valid base64") from exc
    if len(signature) != 64:
        raise ReleaseChannelError("release.signature.invalid", "Ed25519 signature must be 64 bytes")
    key = keys.resolve(key_id)
    with tempfile.TemporaryDirectory(prefix="studiocast-signature-") as temp:
        sig_path = Path(temp) / "signature"
        sig_path.write_bytes(signature)
        try:
            result = subprocess.run(
                ["openssl", "pkeyutl", "-verify", "-pubin", "-inkey", str(key), "-sigfile", str(sig_path), "-rawin", "-in", str(data_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ReleaseChannelError("release.signature.verifier_unavailable", f"Ed25519 verifier unavailable: {exc}") from exc
    if result.returncode != 0:
        raise ReleaseChannelError("release.signature.mismatch", "Ed25519 signature verification failed")


def parse_signature_envelope(raw: bytes | str) -> tuple[str, str]:
    envelope = _expect_object(strict_json_loads(raw), "signature", {"schema_version", "algorithm", "key_id", "signature"})
    if envelope["schema_version"] != 1 or envelope["algorithm"] != "ed25519":
        raise ReleaseChannelError("release.signature.schema", "unsupported signature envelope")
    return _expect_string(envelope["key_id"], "signature.key_id"), _expect_string(envelope["signature"], "signature.signature")


def _validate_artifact(value: Any, name: str, artifact_id: str, *, allow_file_urls: bool) -> dict[str, Any]:
    artifact = _expect_object(
        value,
        f"artifacts.{name}",
        {"artifact_id", "filename", "url", "size_bytes", "sha256", "signature", "provenance", "license"},
    )
    if _expect_string(artifact["artifact_id"], f"artifacts.{name}.artifact_id") != artifact_id:
        raise ReleaseChannelError(
            "release.manifest.schema",
            f"artifacts.{name}.artifact_id must be {artifact_id}",
        )
    filename = _expect_string(artifact["filename"], f"artifacts.{name}.filename")
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ReleaseChannelError("release.manifest.path", f"unsafe artifact filename: {filename}")
    _validate_https_url(artifact["url"], f"artifacts.{name}.url", allow_file=allow_file_urls)
    if type(artifact["size_bytes"]) is not int or artifact["size_bytes"] < 1:
        raise ReleaseChannelError("release.manifest.schema", f"artifacts.{name}.size_bytes must be positive")
    if not isinstance(artifact["sha256"], str) or not _SHA256_RE.fullmatch(artifact["sha256"]):
        raise ReleaseChannelError("release.manifest.schema", f"artifacts.{name}.sha256 is invalid")
    signature = _expect_object(artifact["signature"], f"artifacts.{name}.signature", {"algorithm", "key_id", "value"})
    if signature["algorithm"] != "ed25519":
        raise ReleaseChannelError("release.manifest.schema", "only Ed25519 artifact signatures are supported")
    key_id = _expect_string(signature["key_id"], f"artifacts.{name}.signature.key_id")
    if not _KEY_ID_RE.fullmatch(key_id):
        raise ReleaseChannelError("release.manifest.schema", f"artifacts.{name}.signature.key_id is invalid")
    signature_value = _expect_string(signature["value"], f"artifacts.{name}.signature.value")
    try:
        if len(base64.b64decode(signature_value, validate=True)) != 64:
            raise ValueError("wrong length")
    except (ValueError, TypeError) as exc:
        raise ReleaseChannelError("release.manifest.schema", f"artifacts.{name}.signature.value is invalid") from exc
    provenance = _expect_object(artifact["provenance"], f"artifacts.{name}.provenance", {"repository", "commit", "workflow_run_url"})
    _validate_https_url(provenance["repository"], f"artifacts.{name}.provenance.repository", allow_file=False)
    commit = _expect_string(provenance["commit"], f"artifacts.{name}.provenance.commit")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ReleaseChannelError("release.manifest.schema", f"artifacts.{name}.provenance.commit is invalid")
    _validate_https_url(provenance["workflow_run_url"], f"artifacts.{name}.provenance.workflow_run_url", allow_file=False)
    license_info = _expect_object(artifact["license"], f"artifacts.{name}.license", {"spdx_id", "name", "url"})
    _expect_string(license_info["spdx_id"], f"artifacts.{name}.license.spdx_id")
    _expect_string(license_info["name"], f"artifacts.{name}.license.name")
    _validate_https_url(license_info["url"], f"artifacts.{name}.license.url", allow_file=False)
    return artifact


def validate_manifest(value: Any, *, allow_file_urls: bool = False) -> dict[str, Any]:
    manifest = _expect_object(
        value,
        "manifest",
        {"schema_version", "manifest_version", "channel", "generated_at", "release", "minimum_installer_version", "artifacts", "activation"},
    )
    if manifest["schema_version"] != SUPPORTED_SCHEMA_VERSION or manifest["manifest_version"] != MANIFEST_VERSION:
        raise ReleaseChannelError("release.manifest.unsupported_schema", "unsupported release manifest schema")
    if manifest["channel"] != "stable":
        raise ReleaseChannelError("release.manifest.channel", "release manifest is not the stable channel")
    _expect_timestamp(manifest["generated_at"], "generated_at")
    release = _expect_object(manifest["release"], "release", {"version", "tag", "published_at", "page_url"})
    version = _expect_string(release["version"], "release.version")
    SemVer.parse(version)
    if release["tag"] != f"v{version}":
        raise ReleaseChannelError("release.manifest.version_tag", "release tag does not match version")
    _expect_timestamp(release["published_at"], "release.published_at")
    _validate_https_url(release["page_url"], "release.page_url", allow_file=False)
    SemVer.parse(_expect_string(manifest["minimum_installer_version"], "minimum_installer_version"))
    artifacts = _expect_object(manifest["artifacts"], "artifacts", {"source_archive", "installer_appimage"})
    _validate_artifact(
        artifacts["source_archive"],
        "source_archive",
        "studiocast-source",
        allow_file_urls=allow_file_urls,
    )
    _validate_artifact(
        artifacts["installer_appimage"],
        "installer_appimage",
        "studiocast-installer-appimage",
        allow_file_urls=allow_file_urls,
    )
    activation = _expect_object(manifest["activation"], "activation", {"mode", "retain_previous", "rollback_supported"})
    if activation != {"mode": "side_by_side_atomic_pointer", "retain_previous": True, "rollback_supported": True}:
        raise ReleaseChannelError("release.manifest.activation", "stable releases require side-by-side atomic activation and rollback")
    return manifest


def verify_manifest(raw_manifest: bytes, raw_signature: bytes, keys: TrustedKeyStore, *, allow_file_urls: bool = False) -> dict[str, Any]:
    key_id, signature = parse_signature_envelope(raw_signature)
    verify_ed25519(raw_manifest, signature, key_id, keys)
    return validate_manifest(strict_json_loads(raw_manifest), allow_file_urls=allow_file_urls)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(path: Path, metadata: Mapping[str, Any], keys: TrustedKeyStore) -> None:
    if not path.is_file() or path.is_symlink():
        raise ReleaseChannelError("release.artifact.unavailable", f"artifact is not a regular file: {path}")
    if path.stat().st_size != metadata["size_bytes"]:
        raise ReleaseChannelError("release.artifact.size_mismatch", f"artifact size mismatch: {path.name}")
    if sha256_file(path) != metadata["sha256"]:
        raise ReleaseChannelError("release.artifact.hash_mismatch", f"artifact hash mismatch: {path.name}")
    _verify_ed25519_path(path, metadata["signature"]["value"], metadata["signature"]["key_id"], keys)


class Transport:
    """Transport interface used by acquire_artifact; implementations must append."""

    def append(self, url: str, destination: Path, offset: int) -> None:
        raise NotImplementedError


class UrlTransport(Transport):
    def __init__(self, opener: Optional[Callable[..., Any]] = None):
        self._opener = opener or urllib.request.urlopen

    def append(self, url: str, destination: Path, offset: int) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "file":
            source = Path(urllib.request.url2pathname(parsed.path))
            if not source.is_file() or source.is_symlink():
                raise ReleaseChannelError("release.download.source_unavailable", f"local source unavailable: {source}")
            with source.open("rb") as incoming, destination.open("ab") as outgoing:
                incoming.seek(offset)
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            return
        if parsed.scheme != "https":
            raise ReleaseChannelError("release.download.scheme", "only HTTPS and local file transports are supported")
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with self._opener(request, timeout=30) as incoming:
                final_url = getattr(incoming, "geturl", lambda: url)()
                if urllib.parse.urlparse(final_url).scheme != "https":
                    raise ReleaseChannelError(
                        "release.download.redirect_scheme",
                        "HTTPS download redirected to a non-HTTPS URL",
                    )
                status = getattr(incoming, "status", 200)
                if offset and status != 206:
                    raise ReleaseChannelError("release.download.resume_unsupported", "server did not honor range request")
                content_range = incoming.headers.get("Content-Range") if offset and hasattr(incoming, "headers") else None
                if offset and (not content_range or not content_range.startswith(f"bytes {offset}-")):
                    raise ReleaseChannelError(
                        "release.download.resume_mismatch",
                        "server returned an invalid range for the partial artifact",
                    )
                with destination.open("ab") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
        except ReleaseChannelError:
            raise
        except (OSError, urllib.error.URLError) as exc:
            raise ReleaseChannelError("release.download.failed", f"download failed: {exc}") from exc


def fetch_release_manifest(
    cache_dir: Path,
    keys: TrustedKeyStore,
    *,
    manifest_url: str = DEFAULT_STABLE_MANIFEST_URL,
    signature_url: str = DEFAULT_STABLE_SIGNATURE_URL,
    transport: Optional[Transport] = None,
    offline: bool = False,
    allow_file_urls: bool = False,
) -> dict[str, Any]:
    """Fetch and atomically select a verified manifest/signature generation."""

    root = Path(cache_dir)
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise ReleaseChannelError("release.cache.unsafe_path", f"manifest cache is unavailable: {root}") from exc
    if root.is_symlink():
        raise ReleaseChannelError("release.cache.unsafe_path", "manifest cache may not be a symlink")
    pointer = root / "current"

    def load_current() -> dict[str, Any]:
        if pointer.is_symlink() or (pointer.exists() and not pointer.is_file()):
            raise ReleaseChannelError("release.manifest_cache.corrupt", "manifest cache pointer is unsafe")
        try:
            generation = pointer.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as exc:
            raise ReleaseChannelError("release.manifest_cache.miss", "no cached signed release manifest") from exc
        if not _SHA256_RE.fullmatch(generation):
            raise ReleaseChannelError("release.manifest_cache.corrupt", "invalid manifest cache pointer")
        directory = root / generation
        if directory.is_symlink() or not directory.is_dir():
            raise ReleaseChannelError("release.manifest_cache.corrupt", "manifest generation may not be a symlink")
        try:
            for filename in ("manifest.json", "manifest.json.sig"):
                candidate = directory / filename
                if candidate.is_symlink() or not candidate.is_file():
                    raise ReleaseChannelError(
                        "release.manifest_cache.corrupt",
                        f"cached {filename} is not a regular file",
                    )
            return verify_manifest(
                (directory / "manifest.json").read_bytes(),
                (directory / "manifest.json.sig").read_bytes(),
                keys,
                allow_file_urls=allow_file_urls,
            )
        except OSError as exc:
            raise ReleaseChannelError("release.manifest_cache.corrupt", "cached manifest generation is incomplete") from exc

    if offline:
        try:
            return load_current()
        except ReleaseChannelError as exc:
            if exc.code == "release.manifest_cache.miss":
                raise
            raise ReleaseChannelError("release.manifest_cache.corrupt", str(exc)) from exc

    active_transport = transport or UrlTransport()
    with tempfile.TemporaryDirectory(prefix=".manifest-download-", dir=root) as temporary:
        staging = Path(temporary)
        manifest_path = staging / "manifest.json"
        signature_path = staging / "manifest.json.sig"
        active_transport.append(manifest_url, manifest_path, 0)
        active_transport.append(signature_url, signature_path, 0)
        raw_manifest = manifest_path.read_bytes()
        raw_signature = signature_path.read_bytes()
        manifest = verify_manifest(raw_manifest, raw_signature, keys, allow_file_urls=allow_file_urls)
        generation = hashlib.sha256(raw_manifest).hexdigest()
        final_generation = root / generation
        generation_is_current = False
        if final_generation.is_dir() and not final_generation.is_symlink():
            manifest_file = final_generation / "manifest.json"
            signature_file = final_generation / "manifest.json.sig"
            generation_is_current = (
                manifest_file.is_file()
                and not manifest_file.is_symlink()
                and signature_file.is_file()
                and not signature_file.is_symlink()
                and manifest_file.read_bytes() == raw_manifest
                and signature_file.read_bytes() == raw_signature
            )
        if not generation_is_current:
            generation_staging = Path(tempfile.mkdtemp(prefix=f".{generation}.staging-", dir=root))
            try:
                (generation_staging / "manifest.json").write_bytes(raw_manifest)
                (generation_staging / "manifest.json.sig").write_bytes(raw_signature)
                if final_generation.exists() or final_generation.is_symlink():
                    if final_generation.is_dir() and not final_generation.is_symlink():
                        shutil.rmtree(final_generation)
                    else:
                        final_generation.unlink()
                os.replace(generation_staging, final_generation)
            finally:
                if generation_staging.exists():
                    shutil.rmtree(generation_staging)
        pointer_temp = root / f".current-{os.getpid()}"
        pointer_temp.write_text(generation + "\n", encoding="ascii")
        os.replace(pointer_temp, pointer)
        return manifest


def _safe_cache_target(cache_dir: Path, filename: str) -> Path:
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ReleaseChannelError("release.cache.unsafe_path", f"unsafe cache filename: {filename}")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise ReleaseChannelError("release.cache.unsafe_path", f"cache directory is unavailable: {cache_dir}") from exc
    if cache_dir.is_symlink():
        raise ReleaseChannelError("release.cache.unsafe_path", "cache directory may not be a symlink")
    return cache_dir / filename


def acquire_artifact(
    metadata: Mapping[str, Any],
    cache_dir: Path,
    keys: TrustedKeyStore,
    *,
    transport: Optional[Transport] = None,
    offline: bool = False,
    force_refresh: bool = False,
) -> Path:
    """Return a verified cache artifact, downloading atomically when permitted."""

    target = _safe_cache_target(Path(cache_dir), metadata["filename"])
    existing_good = False
    if target.exists():
        try:
            verify_artifact(target, metadata, keys)
            existing_good = True
        except ReleaseChannelError:
            if offline:
                raise ReleaseChannelError("release.cache.corrupt", f"offline cache artifact is corrupt: {target.name}")
    if existing_good and (offline or not force_refresh):
        return target
    if offline:
        raise ReleaseChannelError("release.cache.miss", f"verified offline artifact is unavailable: {target.name}")
    partial = target.with_name(f".{target.name}.part")
    if (
        partial.is_symlink()
        or (partial.exists() and not partial.is_file())
        or (partial.exists() and partial.stat().st_nlink != 1)
    ):
        raise ReleaseChannelError("release.cache.unsafe_path", "partial cache path is unsafe")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > metadata["size_bytes"]:
        partial.unlink()
        offset = 0
    (transport or UrlTransport()).append(metadata["url"], partial, offset)
    try:
        verify_artifact(partial, metadata, keys)
        os.chmod(partial, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(partial, target)
    except ReleaseChannelError:
        # A previously verified target is deliberately left untouched.
        raise
    return target


@dataclasses.dataclass(frozen=True)
class SelfUpdateOffer:
    state: str
    reason_code: str
    current_appimage: Optional[str]
    verified_appimage: Optional[str]
    relaunch_command: tuple[str, ...]
    manual_download_url: str


def prepare_self_update(
    manifest: Mapping[str, Any],
    current_installer_version: str,
    cache_dir: Path,
    keys: TrustedKeyStore,
    *,
    appimage_path: Optional[str] = None,
    argv: Sequence[str] = (),
    transport: Optional[Transport] = None,
    offline: bool = False,
) -> SelfUpdateOffer:
    metadata = manifest["artifacts"]["installer_appimage"]
    available = manifest["release"]["version"]
    decision = decide_update(current_installer_version, available)
    current = appimage_path or os.environ.get("APPIMAGE")
    manual_url = metadata["url"]
    if decision.action != "update":
        return SelfUpdateOffer("not_needed", decision.reason_code, current, None, (), manual_url)
    if not current or not Path(current).is_absolute() or not Path(current).is_file():
        return SelfUpdateOffer("manual_download", "self_update.appimage_not_detected", current, None, (), manual_url)
    try:
        cached = acquire_artifact(metadata, cache_dir, keys, transport=transport, offline=offline)
        os.chmod(cached, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    except ReleaseChannelError as exc:
        return SelfUpdateOffer("manual_download", exc.code, current, None, (), manual_url)
    return SelfUpdateOffer("offer_restart", "self_update.verified_offer", current, str(cached), (str(cached), *argv), manual_url)


def relaunch_self_update(offer: SelfUpdateOffer, *, user_confirmed: bool, launcher: Callable[[Sequence[str]], Any]) -> bool:
    """Launch only after explicit caller confirmation; never replaces current AppImage."""

    if not user_confirmed or offer.state != "offer_restart" or not offer.relaunch_command:
        return False
    launcher(offer.relaunch_command)
    return True


def rollback_metadata(active_version: str, previous_version: Optional[str], payload_root: Path) -> dict[str, Any]:
    root = Path(payload_root)
    return {
        "activation_mode": "side_by_side_atomic_pointer",
        "active_version": active_version,
        "active_payload": str(root / active_version),
        "previous_version": previous_version,
        "previous_payload": str(root / previous_version) if previous_version else None,
        "rollback_available": previous_version is not None,
    }
