#!/usr/bin/python3 -I
"""Verified, resumable user-local transactions for curated model packs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Protocol


CATALOG_SCHEMA = 1
RESULT_SCHEMA = 1
MARKER_NAME = ".studiocast-curated-pack.json"
MAX_CATALOG_BYTES = 1024 * 1024
MAX_PACKS = 64
MAX_ARTIFACTS_PER_PACK = 16
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
FAMILIES = frozenset(("open_audio", "open_video"))


class CatalogError(ValueError):
    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


class TransactionError(RuntimeError):
    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


class DuplicateKeyError(ValueError):
    pass


def _no_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path, maximum: int = MAX_CATALOG_BYTES) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CatalogError("catalog_unavailable", f"Cannot read catalog: {exc}") from exc
    if not raw or len(raw) > maximum:
        raise CatalogError("invalid_catalog", "Catalog is empty or exceeds the size limit.")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=lambda value: _invalid_number(value),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, DuplicateKeyError, ValueError) as exc:
        raise CatalogError("invalid_catalog", f"Catalog JSON is invalid: {exc}") from exc


def _invalid_number(value: str) -> None:
    raise ValueError(f"invalid JSON number: {value}")


def _exact_fields(value: Mapping[str, Any], required: set[str], optional: set[str], context: str) -> None:
    unknown = set(value) - required - optional
    missing = required - set(value)
    if unknown:
        raise CatalogError("invalid_catalog", f"Unknown {context} field: {sorted(unknown)[0]}")
    if missing:
        raise CatalogError("invalid_catalog", f"Missing {context} field: {sorted(missing)[0]}")


def _safe_relative(value: Any, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise CatalogError("unsafe_catalog_path", f"Invalid {context} path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise CatalogError("unsafe_catalog_path", f"Unsafe {context} path: {value!r}")
    return path


def _sha(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise CatalogError("invalid_catalog", f"Invalid SHA-256 for {context}.")
    return value


def _validate_catalog(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError("invalid_catalog", "Catalog root must be an object.")
    _exact_fields(
        value,
        {"schema_version", "catalog_version", "policy_version", "default_source", "size_metadata", "packs"},
        set(),
        "catalog",
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != CATALOG_SCHEMA:
        raise CatalogError("unknown_catalog_schema", "Unsupported model catalog schema.")
    if type(value["policy_version"]) is not int or value["policy_version"] < 1:
        raise CatalogError("invalid_catalog", "Invalid model policy version.")
    if not isinstance(value["catalog_version"], str) or not value["catalog_version"]:
        raise CatalogError("invalid_catalog", "catalog_version must be a string.")
    if not isinstance(value["default_source"], str) or not value["default_source"]:
        raise CatalogError("invalid_catalog", "default_source must be a string.")
    size_metadata = value["size_metadata"]
    if not isinstance(size_metadata, dict):
        raise CatalogError("invalid_catalog", "size_metadata must be an object.")
    _exact_fields(size_metadata, {"trusted", "reason_code"}, set(), "size_metadata")
    if not isinstance(size_metadata["trusted"], bool) or not isinstance(size_metadata["reason_code"], str):
        raise CatalogError("invalid_catalog", "Invalid size metadata contract.")
    packs = value["packs"]
    if not isinstance(packs, list) or not packs or len(packs) > MAX_PACKS:
        raise CatalogError("invalid_catalog", "packs must be a bounded non-empty array.")
    seen_ids: set[str] = set()
    seen_paths: set[tuple[str, str]] = set()
    for pack in packs:
        if not isinstance(pack, dict):
            raise CatalogError("invalid_catalog", "Each pack must be an object.")
        _exact_fields(
            pack,
            {"id", "family", "task", "default", "pack_path", "metadata", "license", "provenance", "artifacts"},
            set(),
            "pack",
        )
        pack_id = pack["id"]
        if not isinstance(pack_id, str) or ID_RE.fullmatch(pack_id) is None or pack_id in seen_ids:
            raise CatalogError("invalid_catalog", "Pack IDs must be unique safe identifiers.")
        seen_ids.add(pack_id)
        if pack["family"] not in FAMILIES or not isinstance(pack["task"], str):
            raise CatalogError("invalid_catalog", f"Invalid family/task for {pack_id}.")
        if not isinstance(pack["default"], bool):
            raise CatalogError("invalid_catalog", f"default must be boolean for {pack_id}.")
        pack_path = str(_safe_relative(pack["pack_path"], f"{pack_id}.pack_path"))
        path_key = (pack["family"], pack_path)
        if path_key in seen_paths:
            raise CatalogError("invalid_catalog", "Curated pack paths must be unique.")
        seen_paths.add(path_key)
        for object_name in ("license", "provenance"):
            if not isinstance(pack[object_name], dict):
                raise CatalogError("invalid_catalog", f"{object_name} must be an object for {pack_id}.")
        metadata = pack["metadata"]
        if not isinstance(metadata, list) or len(metadata) < 2 or len(metadata) > 8:
            raise CatalogError("invalid_catalog", f"Invalid metadata list for {pack_id}.")
        metadata_destinations: set[str] = set()
        for item in metadata:
            if not isinstance(item, dict):
                raise CatalogError("invalid_catalog", "Metadata entries must be objects.")
            _exact_fields(item, {"source", "destination", "sha256"}, set(), "metadata")
            _safe_relative(item["source"], "metadata source")
            destination = str(_safe_relative(item["destination"], "metadata destination"))
            if destination in metadata_destinations:
                raise CatalogError("invalid_catalog", "Duplicate metadata destination.")
            metadata_destinations.add(destination)
            _sha(item["sha256"], "metadata")
        if "model.json" not in metadata_destinations or "LICENSE.txt" not in metadata_destinations:
            raise CatalogError("invalid_catalog", f"{pack_id} must install model.json and LICENSE.txt.")
        artifacts = pack["artifacts"]
        if not isinstance(artifacts, list) or not artifacts or len(artifacts) > MAX_ARTIFACTS_PER_PACK:
            raise CatalogError("invalid_catalog", f"Invalid artifacts for {pack_id}.")
        artifact_names: set[str] = set()
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise CatalogError("invalid_catalog", "Artifact entries must be objects.")
            _exact_fields(
                artifact,
                {"name", "sha256", "size_bytes", "size_status", "source_paths"},
                set(),
                "artifact",
            )
            name = str(_safe_relative(artifact["name"], "artifact name"))
            if name in artifact_names:
                raise CatalogError("invalid_catalog", "Duplicate artifact destination.")
            artifact_names.add(name)
            _sha(artifact["sha256"], "artifact")
            size = artifact["size_bytes"]
            if size is not None and (type(size) is not int or size < 0):
                raise CatalogError("invalid_catalog", "Artifact size must be a non-negative integer or null.")
            expected_status = "known" if size is not None else "unknown"
            if artifact["size_status"] != expected_status:
                raise CatalogError("invalid_catalog", "Artifact size status does not match size_bytes.")
            sources = artifact["source_paths"]
            if not isinstance(sources, list) or not sources or len(sources) > 8:
                raise CatalogError("invalid_catalog", "Artifact source_paths must be bounded.")
            for source in sources:
                _safe_relative(source, "artifact source")
            if size_metadata["trusted"] and size is None:
                raise CatalogError("invalid_catalog", "Trusted size metadata cannot omit artifact sizes.")
    return value


def load_catalog(path: Path) -> dict[str, Any]:
    return _validate_catalog(_load_json(path))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_regular_nosymlink(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(info.st_mode) and not path.is_symlink()


def _verified_file(path: Path, expected_sha: str, expected_size: int | None = None) -> bool:
    if not _is_regular_nosymlink(path):
        return False
    try:
        if expected_size is not None and path.stat().st_size != expected_size:
            return False
        return sha256_path(path) == expected_sha
    except OSError:
        return False


def _ensure_root(root: Path) -> Path:
    if root.is_symlink():
        raise TransactionError("unsafe_destination", "Destination root must not be a symlink.")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TransactionError("unsafe_destination", f"Cannot create destination root: {exc}") from exc
    if not root.is_dir():
        raise TransactionError("unsafe_destination", "Destination root is not a directory.")
    return root.resolve()


def _ensure_relative_dirs(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise TransactionError("unsafe_destination", f"Unsafe destination component: {part}")
        if current.exists():
            if not current.is_dir():
                raise TransactionError("unsafe_destination", f"Unsafe destination component: {part}")
        else:
            try:
                current.mkdir()
            except OSError as exc:
                raise TransactionError("unsafe_destination", f"Cannot create destination component: {part}") from exc
    return current


def _existing_destination(root: Path, relative: PurePosixPath) -> tuple[Path, bool]:
    """Return a catalog destination and whether every parent is safe/present."""
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise TransactionError("unsafe_destination", f"Unsafe destination component: {part}")
        if not current.exists():
            return root / Path(*relative.parts), False
    return current / relative.parts[-1], True


def _copy_regular(source: Path, destination: Path) -> None:
    if not _is_regular_nosymlink(source):
        raise TransactionError("unsafe_source", f"Source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, flags)
    try:
        destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(source_fd, "rb", closefd=False) as src, os.fdopen(
                destination_fd, "wb", closefd=False
            ) as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            os.fchmod(destination_fd, 0o644)
        finally:
            os.close(destination_fd)
    finally:
        os.close(source_fd)


class Transport(Protocol):
    def fetch(self, relative_path: str, part_path: Path, offset: int) -> int:
        """Append bytes starting at offset and return bytes written."""


class FileTransport:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def fetch(self, relative_path: str, part_path: Path, offset: int) -> int:
        relative = _safe_relative(relative_path, "transport source")
        candidate = self.root / Path(*relative.parts)
        if candidate.is_symlink():
            raise TransactionError("unsafe_source", "Artifact source must not be a symlink.")
        source = candidate.resolve()
        try:
            source.relative_to(self.root)
        except ValueError as exc:
            raise TransactionError("unsafe_source", "Artifact source escapes configured root.") from exc
        if not _is_regular_nosymlink(source):
            raise TransactionError("source_unavailable", f"Local artifact is unavailable: {relative_path}")
        size = source.stat().st_size
        if offset > size:
            raise TransactionError("resume_range_invalid", "Partial file exceeds source size.")
        mode = "ab" if offset else "wb"
        written = 0
        with source.open("rb") as src, part_path.open(mode) as dst:
            src.seek(offset)
            while True:
                block = src.read(1024 * 1024)
                if not block:
                    break
                dst.write(block)
                written += len(block)
            dst.flush()
            os.fsync(dst.fileno())
        return written


class HttpTransport:
    def __init__(self, base_url: str, timeout_seconds: int = 120):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise TransactionError("invalid_source", "HTTP source must use http or https.")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def fetch(self, relative_path: str, part_path: Path, offset: int) -> int:
        relative = _safe_relative(relative_path, "transport source")
        encoded = "/".join(urllib.parse.quote(part, safe="") for part in relative.parts)
        request = urllib.request.Request(f"{self.base_url}/{encoded}")
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout_seconds)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransactionError("source_unavailable", f"Artifact request failed: {exc}") from exc
        with response:
            status = getattr(response, "status", 200)
            append = offset > 0 and status == 206
            mode = "ab" if append else "wb"
            written = 0
            with part_path.open(mode) as dst:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    dst.write(block)
                    written += len(block)
                dst.flush()
                os.fsync(dst.fileno())
        return written


def transport_for(source: str) -> Transport:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in ("http", "https"):
        return HttpTransport(source)
    if parsed.scheme == "file":
        return FileTransport(Path(urllib.request.url2pathname(parsed.path)))
    if parsed.scheme:
        raise TransactionError("invalid_source", "Only local, file, http, and https sources are supported.")
    return FileTransport(Path(source))


@dataclass
class ArtifactAcquisition:
    path: Path
    status: str
    attempts: int
    resumed: bool


class ArtifactProvider:
    def __init__(self, cache_root: Path, transport: Transport | None, *, offline: bool, retries: int):
        self.cache_root = _ensure_root(cache_root)
        self.transport = transport
        self.offline = offline
        self.retries = max(0, min(retries, 10))
        self.artifact_cache = _ensure_relative_dirs(self.cache_root, PurePosixPath("sha256"))

    def acquire(
        self,
        artifact: Mapping[str, Any],
        existing: Path | None,
        *,
        force: bool,
    ) -> ArtifactAcquisition:
        expected = artifact["sha256"]
        size = artifact["size_bytes"]
        if not force and existing is not None and _verified_file(existing, expected, size):
            return ArtifactAcquisition(existing, "existing_verified", 0, False)
        final_cache = self.artifact_cache / expected
        cache_verified = _verified_file(final_cache, expected, size)
        if cache_verified and (not force or self.offline):
            return ArtifactAcquisition(final_cache, "cache_verified", 0, False)
        if not cache_verified and (final_cache.is_symlink() or final_cache.exists()):
            if final_cache.is_symlink() or not final_cache.is_file():
                raise TransactionError("unsafe_cache", "Cache artifact path is unsafe.")
            final_cache.unlink()
        part_suffix = ".refresh.part" if cache_verified else ".part"
        part = self.artifact_cache / f"{expected}{part_suffix}"
        if part.is_symlink() or (part.exists() and not part.is_file()):
            raise TransactionError("unsafe_cache", "Partial artifact path is unsafe.")
        if _verified_file(part, expected, size):
            os.replace(part, final_cache)
            return ArtifactAcquisition(final_cache, "cache_resumed_verified", 0, True)
        if self.offline:
            raise TransactionError("offline_cache_missing", "No verified cached artifact is available offline.")
        if self.transport is None:
            raise TransactionError("source_unavailable", "No artifact transport is configured.")

        attempts = 0
        resumed = part.exists() and part.stat().st_size > 0
        last_error = TransactionError("source_unavailable", "No artifact source succeeded.")
        for source_path in artifact["source_paths"]:
            for _ in range(self.retries + 1):
                attempts += 1
                offset = part.stat().st_size if part.exists() else 0
                resumed = resumed or offset > 0
                try:
                    self.transport.fetch(source_path, part, offset)
                except TransactionError as exc:
                    last_error = exc
                    if exc.reason_code == "resume_range_invalid" and part.exists():
                        part.unlink()
                    continue
                except OSError as exc:
                    last_error = TransactionError("source_unavailable", f"Artifact transfer failed: {exc}")
                    continue
                if _verified_file(part, expected, size):
                    os.replace(part, final_cache)
                    return ArtifactAcquisition(final_cache, "downloaded_verified", attempts, resumed)
                last_error = TransactionError("checksum_mismatch", "Downloaded artifact checksum or size is invalid.")
                if part.exists():
                    part.unlink()
        raise last_error


def _repo_file(repo_root: Path, item: Mapping[str, Any]) -> Path:
    relative = _safe_relative(item["source"], "metadata source")
    candidate = repo_root / Path(*relative.parts)
    if candidate.is_symlink():
        raise TransactionError("unsafe_metadata", "Trusted metadata source must not be a symlink.")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise TransactionError("unsafe_metadata", "Metadata source escapes the trusted root.") from exc
    if not _verified_file(resolved, item["sha256"]):
        raise TransactionError("metadata_mismatch", f"Trusted metadata is missing or changed: {relative}")
    return resolved


def _pack_current(pack: Mapping[str, Any], destination: Path, repo_root: Path) -> bool:
    if not destination.is_dir() or destination.is_symlink():
        return False
    for item in pack["metadata"]:
        target = destination / Path(*_safe_relative(item["destination"], "metadata destination").parts)
        if not _verified_file(target, item["sha256"]):
            return False
    for artifact in pack["artifacts"]:
        target = destination / Path(*_safe_relative(artifact["name"], "artifact name").parts)
        if not _verified_file(target, artifact["sha256"], artifact["size_bytes"]):
            return False
    marker = destination / MARKER_NAME
    if not _is_regular_nosymlink(marker):
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and value.get("pack_id") == pack["id"]
        and value.get("pack_path") == pack["pack_path"]
    )


def _curated_destination_owned(pack: Mapping[str, Any], destination: Path) -> bool:
    """Accept a helper marker or an exact legacy StudioCast metadata template."""
    if destination.is_symlink():
        return False
    if not destination.exists():
        return True
    if not destination.is_dir():
        return False
    marker = destination / MARKER_NAME
    if _is_regular_nosymlink(marker):
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return (
            isinstance(value, dict)
            and value.get("schema_version") == 1
            and value.get("pack_id") == pack["id"]
            and value.get("pack_path") == pack["pack_path"]
        )
    # Migrate only an old installer-created pack whose trusted model and license
    # metadata both still match exactly. An arbitrary/custom pack is preserved.
    return all(
        _verified_file(
            destination / Path(*_safe_relative(item["destination"], "metadata destination").parts),
            item["sha256"],
        )
        for item in pack["metadata"]
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        os.fchmod(fd, 0o644)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def _activate_pack(stage: Path, destination: Path) -> None:
    if destination.exists() and (destination.is_symlink() or not destination.is_dir()):
        raise TransactionError("unsafe_destination", "Curated pack destination is not a regular directory.")
    backup = destination.with_name(f".{destination.name}.backup.{uuid.uuid4().hex}")
    moved_old = False
    try:
        if destination.exists():
            os.rename(destination, backup)
            moved_old = True
        os.rename(stage, destination)
    except OSError as exc:
        if moved_old and not destination.exists() and backup.exists():
            os.rename(backup, destination)
        raise TransactionError("activation_failed", f"Could not atomically activate model pack: {exc}") from exc
    if moved_old:
        try:
            shutil.rmtree(backup)
        except OSError:
            # Activation is already committed. A later repair can reconcile the
            # same-directory backup without misreporting the active pack.
            pass


class ModelTransactionInstaller:
    def __init__(
        self,
        catalog: Mapping[str, Any],
        repo_root: Path,
        destination_root: Path,
        cache_root: Path,
        transport: Transport | None,
        *,
        offline: bool = False,
        retries: int = 2,
    ):
        self.catalog = catalog
        self.repo_root = repo_root.resolve()
        self.destination_root = destination_root
        self.cache_root = cache_root
        self.transport = transport
        self.offline = offline
        self.retries = retries
        self.provider: ArtifactProvider | None = None

    def _prepare_install(self) -> None:
        self.destination_root = _ensure_root(self.destination_root)
        self.provider = ArtifactProvider(
            self.cache_root,
            self.transport,
            offline=self.offline,
            retries=self.retries,
        )

    def _selected(self, family: str, model_ids: list[str] | None) -> list[Mapping[str, Any]]:
        packs = [pack for pack in self.catalog["packs"] if pack["family"] == family]
        if model_ids:
            requested = set(model_ids)
            selected = [pack for pack in packs if pack["id"] in requested]
            missing = requested - {pack["id"] for pack in selected}
            if missing:
                raise TransactionError("unknown_model", f"Unknown curated model ID: {sorted(missing)[0]}")
            return selected
        return [pack for pack in packs if pack["default"]]

    def install(
        self,
        family: str,
        model_ids: list[str] | None = None,
        *,
        force: bool = False,
        recommended: bool = False,
    ) -> dict[str, Any]:
        if family not in FAMILIES:
            raise TransactionError("invalid_family", "Unknown model family.")
        selected = self._selected(family, model_ids)
        if recommended:
            unknown = [
                f"{pack['id']}/{artifact['name']}"
                for pack in selected
                for artifact in pack["artifacts"]
                if artifact["size_bytes"] is None
            ]
            if unknown or not self.catalog["size_metadata"]["trusted"]:
                return {
                    "schema_version": RESULT_SCHEMA,
                    "state": "blocked",
                    "reason_code": "artifact_sizes_untrusted",
                    "message": "Recommended installation requires trusted signed artifact sizes.",
                    "blockers": unknown,
                    "packs": [],
                }
        self._prepare_install()
        results = [self._install_pack(pack, force=force) for pack in selected]
        succeeded = sum(result["status"] in ("installed", "unchanged") for result in results)
        failed = len(results) - succeeded
        if failed == 0:
            state, reason = "succeeded", "models_installed"
        elif succeeded:
            state, reason = "degraded", "optional_model_failure"
        else:
            state, reason = "failed", "model_install_failed"
        return {
            "schema_version": RESULT_SCHEMA,
            "state": state,
            "reason_code": reason,
            "packs": results,
        }

    def _install_pack(self, pack: Mapping[str, Any], *, force: bool) -> dict[str, Any]:
        if self.provider is None:
            raise TransactionError("transaction_not_prepared", "Install transaction is not prepared.")
        relative = _safe_relative(pack["pack_path"], "pack path")
        parent = _ensure_relative_dirs(self.destination_root, PurePosixPath(*relative.parts[:-1]))
        destination = parent / relative.parts[-1]
        if not _curated_destination_owned(pack, destination):
            return {
                "id": pack["id"],
                "status": "failed",
                "reason_code": "curated_ownership_unverified",
                "message": "Existing destination is not a verified StudioCast-curated pack.",
                "changed": False,
                "artifacts": [],
            }
        if not force and _pack_current(pack, destination, self.repo_root):
            return {
                "id": pack["id"],
                "status": "unchanged",
                "reason_code": "already_verified",
                "changed": False,
                "artifacts": [],
            }

        acquisitions: list[tuple[Mapping[str, Any], ArtifactAcquisition]] = []
        artifact_results: list[dict[str, Any]] = []
        try:
            for item in pack["metadata"]:
                _repo_file(self.repo_root, item)
            for artifact in pack["artifacts"]:
                artifact_relative = _safe_relative(artifact["name"], "artifact name")
                existing = destination / Path(*artifact_relative.parts) if destination.is_dir() else None
                acquired = self.provider.acquire(artifact, existing, force=force)
                acquisitions.append((artifact, acquired))
                artifact_results.append(
                    {
                        "name": artifact["name"],
                        "status": acquired.status,
                        "reason_code": "artifact_verified",
                        "sha256": artifact["sha256"],
                        "size_bytes": artifact["size_bytes"],
                        "attempts": acquired.attempts,
                        "resumed": acquired.resumed,
                    }
                )
        except TransactionError as exc:
            artifact_results.append(
                {
                    "name": artifact["name"] if "artifact" in locals() else None,
                    "status": "failed",
                    "reason_code": exc.reason_code,
                    "message": exc.message,
                }
            )
            return {
                "id": pack["id"],
                "status": "failed",
                "reason_code": exc.reason_code,
                "changed": False,
                "artifacts": artifact_results,
            }

        stage = Path(tempfile.mkdtemp(prefix=".studiocast-stage-", dir=parent))
        try:
            for item in pack["metadata"]:
                source = _repo_file(self.repo_root, item)
                relative_destination = _safe_relative(item["destination"], "metadata destination")
                target = stage / Path(*relative_destination.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                _copy_regular(source, target)
            for artifact, acquired in acquisitions:
                relative_artifact = _safe_relative(artifact["name"], "artifact name")
                target = stage / Path(*relative_artifact.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                _copy_regular(acquired.path, target)
                if not _verified_file(target, artifact["sha256"], artifact["size_bytes"]):
                    raise TransactionError("staging_verification_failed", "Staged artifact failed verification.")
            marker = {
                "schema_version": 1,
                "catalog_version": self.catalog["catalog_version"],
                "pack_id": pack["id"],
                "family": pack["family"],
                "pack_path": pack["pack_path"],
                "artifacts": [
                    {"name": artifact["name"], "sha256": artifact["sha256"]}
                    for artifact in pack["artifacts"]
                ],
            }
            _write_json_atomic(stage / MARKER_NAME, marker)
            _activate_pack(stage, destination)
            return {
                "id": pack["id"],
                "status": "installed",
                "reason_code": "pack_activated",
                "changed": True,
                "artifacts": artifact_results,
                "destination": str(destination),
            }
        except TransactionError as exc:
            return {
                "id": pack["id"],
                "status": "failed",
                "reason_code": exc.reason_code,
                "message": exc.message,
                "changed": False,
                "artifacts": artifact_results,
            }
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def remove_curated(self, family: str, model_ids: list[str] | None = None) -> dict[str, Any]:
        self.destination_root = _ensure_root(self.destination_root)
        selected = self._selected(family, model_ids)
        results = []
        for pack in selected:
            relative = _safe_relative(pack["pack_path"], "pack path")
            try:
                destination, parents_present = _existing_destination(self.destination_root, relative)
            except TransactionError as exc:
                results.append({"id": pack["id"], "status": "failed", "changed": False, "reason_code": exc.reason_code})
                continue
            if not parents_present or (not destination.exists() and not destination.is_symlink()):
                results.append({"id": pack["id"], "status": "absent", "changed": False, "reason_code": "already_absent"})
                continue
            if destination.is_symlink() or not destination.is_dir():
                results.append({"id": pack["id"], "status": "failed", "changed": False, "reason_code": "unsafe_destination"})
                continue
            marker = destination / MARKER_NAME
            try:
                marker_value = json.loads(marker.read_text(encoding="utf-8")) if _is_regular_nosymlink(marker) else None
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                marker_value = None
            if not isinstance(marker_value, dict) or marker_value.get("schema_version") != 1 or marker_value.get("pack_id") != pack["id"] or marker_value.get("pack_path") != pack["pack_path"]:
                results.append({"id": pack["id"], "status": "failed", "changed": False, "reason_code": "curated_ownership_unverified"})
                continue
            tomb = destination.with_name(f".{destination.name}.remove.{uuid.uuid4().hex}")
            try:
                os.rename(destination, tomb)
                shutil.rmtree(tomb)
                results.append({"id": pack["id"], "status": "removed", "changed": True, "reason_code": "curated_pack_removed"})
            except OSError:
                if tomb.exists() and not destination.exists():
                    try:
                        os.rename(tomb, destination)
                    except OSError:
                        results.append({"id": pack["id"], "status": "failed", "changed": True, "reason_code": "curated_removal_partial"})
                        continue
                results.append({"id": pack["id"], "status": "failed", "changed": False, "reason_code": "curated_removal_failed"})
        failures = [result for result in results if result["status"] == "failed"]
        state = "succeeded" if not failures else ("degraded" if len(failures) < len(results) else "failed")
        return {
            "schema_version": RESULT_SCHEMA,
            "state": state,
            "reason_code": "curated_removal_complete" if not failures else "curated_removal_incomplete",
            "packs": results,
        }


def catalog_summary(catalog: Mapping[str, Any], family: str | None = None) -> dict[str, Any]:
    packs = [pack for pack in catalog["packs"] if family is None or pack["family"] == family]
    return {
        "schema_version": 1,
        "catalog_version": catalog["catalog_version"],
        "policy_version": catalog["policy_version"],
        "size_metadata": catalog["size_metadata"],
        "pack_count": len(packs),
        "artifact_count": sum(len(pack["artifacts"]) for pack in packs),
        "packs": [
            {
                "id": pack["id"],
                "family": pack["family"],
                "task": pack["task"],
                "default": pack["default"],
                "pack_path": pack["pack_path"],
                "license": pack["license"],
                "provenance": pack["provenance"],
                "artifacts": [
                    {
                        "name": artifact["name"],
                        "sha256": artifact["sha256"],
                        "size_bytes": artifact["size_bytes"],
                        "size_status": artifact["size_status"],
                    }
                    for artifact in pack["artifacts"]
                ],
            }
            for pack in packs
        ],
    }


def _print_result(result: Mapping[str, Any], json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return
    if "packs" in result:
        for pack in result["packs"]:
            identifier = pack.get("id", "unknown")
            print(f"{identifier}: {pack.get('status', 'available')} ({pack.get('reason_code', '')})")
    print(f"state: {result.get('state', 'catalog')} ({result.get('reason_code', '')})")


def _default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    catalog = repo_root / "packaging/models/curated-model-catalog-v1.json"
    return repo_root, catalog


def build_parser() -> argparse.ArgumentParser:
    repo_root, catalog = _default_paths()
    parser = argparse.ArgumentParser(description="StudioCast verified curated-model transactions")
    parser.add_argument("--catalog", type=Path, default=catalog)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    subparsers = parser.add_subparsers(dest="command", required=True)
    listing = subparsers.add_parser("list")
    listing.add_argument("--family", choices=sorted(FAMILIES))
    listing.add_argument("--json", action="store_true")
    for name in ("install", "remove-curated"):
        command = subparsers.add_parser(name)
        command.add_argument("--family", choices=sorted(FAMILIES), required=True)
        command.add_argument("--dest", type=Path, required=True)
        command.add_argument("--model", action="append", default=[])
        command.add_argument("--json", action="store_true")
        if name == "install":
            command.add_argument("--cache", type=Path, required=True)
            command.add_argument("--source")
            command.add_argument("--offline", action="store_true")
            command.add_argument("--force", action="store_true")
            command.add_argument("--recommended", action="store_true")
            command.add_argument("--retries", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
        if args.command == "list":
            result = catalog_summary(catalog, args.family)
            if args.json:
                print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            else:
                for pack in result["packs"]:
                    sizes = ", ".join(
                        f"{artifact['name']}={artifact['size_bytes'] if artifact['size_bytes'] is not None else 'unknown size'}"
                        for artifact in pack["artifacts"]
                    )
                    print(f"{pack['id']} ({pack['task']}): {sizes}; license={pack['license']['spdx']}")
            return 0
        source = getattr(args, "source", None) or catalog["default_source"]
        transport = None if getattr(args, "offline", False) else transport_for(source)
        cache = getattr(args, "cache", None)
        if cache is None:
            cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            cache = cache_home / "studiocast/models"
        installer = ModelTransactionInstaller(
            catalog,
            args.repo_root,
            args.dest,
            cache,
            transport,
            offline=getattr(args, "offline", False),
            retries=getattr(args, "retries", 2),
        )
        if args.command == "install":
            result = installer.install(
                args.family,
                args.model or None,
                force=args.force,
                recommended=args.recommended,
            )
        else:
            result = installer.remove_curated(args.family, args.model or None)
        _print_result(result, args.json)
        return {"succeeded": 0, "blocked": 2, "degraded": 3}.get(result["state"], 1)
    except (CatalogError, TransactionError) as exc:
        result = {
            "schema_version": RESULT_SCHEMA,
            "state": "failed",
            "reason_code": exc.reason_code,
            "message": exc.message,
            "packs": [],
        }
        json_output = bool(getattr(args, "json", False))
        _print_result(result, json_output)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
