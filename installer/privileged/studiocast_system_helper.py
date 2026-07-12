#!/usr/bin/python3
"""Root-side StudioCast host integration helper.

This file is installed by the separately signed studiocast-system-helper OS
package.  It intentionally has no general command, path, or content operation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn


HELPER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
POLICY_VERSION = 1
MAX_REQUEST_BYTES = 64 * 1024
MAX_OPERATIONS = 32
COMMAND_TIMEOUT_SECONDS = 15 * 60
SUPPORTED_BASE_RELEASES = frozenset(("22.04", "24.04"))
FIXED_ENV = {
    "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}

APT_GET = "/usr/bin/apt-get"
MODPROBE = "/usr/sbin/modprobe"

PACKAGE_ALLOWLIST = frozenset(
    (
        "build-essential",
        "ca-certificates",
        "clang",
        "clang-format",
        "clang-tidy",
        "cmake",
        "curl",
        "dkms",
        "git",
        "glslang-tools",
        "libblas-dev",
        "libdlib-dev",
        "libjpeg-turbo8",
        "libjpeg-turbo8-dev",
        "liblapack-dev",
        "libpng-dev",
        "libpulse-dev",
        "libpulse0",
        "libsqlite3-dev",
        "libvulkan1",
        "libxkbcommon-dev",
        "mesa-vulkan-drivers",
        "ninja-build",
        "pkg-config",
        "pulseaudio-utils",
        "qt6-base-dev",
        "qt6-base-dev-tools",
        "qt6-tools-dev",
        "qt6-tools-dev-tools",
        "qtbase5-dev",
        "tar",
        "v4l-utils",
        "v4l2loopback-dkms",
        "v4l2loopback-utils",
        "vulkan-tools",
    )
)
KERNEL_PACKAGE_PREFIXES = (
    "linux-headers-",
    "linux-modules-extra-",
    "linux-modules-",
)

OPERATION_TYPES = frozenset(
    (
        "packages.ensure.v1",
        "v4l.module.load.v1",
        "v4l.module.reload.v1",
        "v4l.persistence.write.v1",
        "v4l.persistence.remove.v1",
    )
)

ENVELOPE_FIELDS = frozenset(
    (
        "schema_version",
        "request_id",
        "transaction_id",
        "plan_digest",
        "policy_version",
        "preconditions",
        "operations",
    )
)
PRECONDITION_FIELDS = frozenset(("base_os", "base_release", "kernel_release"))
MODULE_FIELDS = frozenset(
    ("id", "type", "device_number", "label", "exclusive_caps")
)
RELOAD_FIELDS = MODULE_FIELDS | frozenset(("module_unload_safe", "device_not_busy"))
REMOVE_HASH_KEYS = frozenset(("v4l_modules_load", "v4l_modprobe"))
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
KERNEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+:~_-]{0,127}$")
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9 ._+()-]{1,64}$")
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

FILE_SPECS = {
    "v4l_modules_load": (
        "modules-load.d",
        "studiocast-v4l2loopback.conf",
        "# StudioCast system helper schema=1 file=v4l_modules_load\n",
    ),
    "v4l_modprobe": (
        "modprobe.d",
        "studiocast-v4l2loopback.conf",
        "# StudioCast system helper schema=1 file=v4l_modprobe\n",
    ),
}


class DuplicateKeyError(ValueError):
    pass


class ProtocolError(Exception):
    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


class OperationCancelled(Exception):
    pass


class PersistenceMutationError(ProtocolError):
    def __init__(self, reason_code: str, message: str, files: list[dict[str, Any]]):
        super().__init__(reason_code, message)
        self.files = files


def _pairs_no_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate field: {key}")
        result[key] = value
    return result


def parse_request(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > MAX_REQUEST_BYTES:
        raise ProtocolError("malformed_request", "Request is empty or exceeds the size limit.")
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda value: (_raise_value_error(f"invalid number: {value}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, DuplicateKeyError, ValueError) as exc:
        raise ProtocolError("malformed_request", f"Invalid JSON request: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("malformed_request", "Request root must be an object.")
    return value


def _raise_value_error(message: str) -> NoReturn:
    raise ValueError(message)


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], context: str) -> None:
    unknown = set(value) - expected
    if unknown:
        raise ProtocolError("unknown_field", f"Unknown {context} field: {sorted(unknown)[0]}")
    missing = expected - set(value)
    if missing:
        raise ProtocolError("invalid_argument", f"Missing {context} field: {sorted(missing)[0]}")


def _string(value: Any, name: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str):
        raise ProtocolError("invalid_argument", f"{name} must be a string.")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ProtocolError("invalid_argument", f"{name} has an invalid value.")
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ProtocolError("invalid_argument", f"{name} must be an integer from {minimum} to {maximum}.")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError("invalid_argument", f"{name} must be a boolean.")
    return value


def _uuid_string(value: Any, name: str) -> str:
    text = _string(value, name)
    try:
        parsed = uuid.UUID(text)
    except ValueError as exc:
        raise ProtocolError("invalid_argument", f"{name} must be a UUID.") from exc
    if str(parsed) != text:
        raise ProtocolError("invalid_argument", f"{name} must use canonical UUID syntax.")
    return text


def _validate_module_fields(operation: Mapping[str, Any], expected: frozenset[str]) -> None:
    _exact_fields(operation, expected, "operation")
    _string(operation["id"], "operation.id", ID_PATTERN)
    _integer(operation["device_number"], "device_number", 0, 63)
    _string(operation["label"], "label", LABEL_PATTERN)
    _boolean(operation["exclusive_caps"], "exclusive_caps")


def validate_operation(operation: Any, kernel_release: str) -> dict[str, Any]:
    if not isinstance(operation, dict):
        raise ProtocolError("invalid_argument", "Each operation must be an object.")
    operation_type = operation.get("type")
    if not isinstance(operation_type, str) or operation_type not in OPERATION_TYPES:
        raise ProtocolError("unknown_operation", "Unknown privileged operation type.")

    if operation_type == "packages.ensure.v1":
        _exact_fields(operation, frozenset(("id", "type", "packages")), "operation")
        _string(operation["id"], "operation.id", ID_PATTERN)
        packages = operation["packages"]
        if not isinstance(packages, list) or not packages or len(packages) > 64:
            raise ProtocolError("invalid_package", "packages must be a non-empty bounded array.")
        seen: set[str] = set()
        for package in packages:
            if not isinstance(package, str) or package in seen:
                raise ProtocolError("invalid_package", "Package names must be unique strings.")
            seen.add(package)
            if package in PACKAGE_ALLOWLIST:
                continue
            if any(package == prefix + kernel_release for prefix in KERNEL_PACKAGE_PREFIXES):
                continue
            raise ProtocolError("invalid_package", f"Package is not allowlisted: {package!r}")
    elif operation_type in ("v4l.module.load.v1", "v4l.persistence.write.v1"):
        _validate_module_fields(operation, MODULE_FIELDS)
    elif operation_type == "v4l.module.reload.v1":
        _validate_module_fields(operation, RELOAD_FIELDS)
        if not _boolean(operation["module_unload_safe"], "module_unload_safe"):
            raise ProtocolError("precondition_changed", "Module unload was not reviewed as safe.")
        if not _boolean(operation["device_not_busy"], "device_not_busy"):
            raise ProtocolError("precondition_changed", "Device was not reviewed as idle.")
    else:
        _exact_fields(operation, frozenset(("id", "type", "expected_hashes")), "operation")
        _string(operation["id"], "operation.id", ID_PATTERN)
        hashes = operation["expected_hashes"]
        if not isinstance(hashes, dict):
            raise ProtocolError("invalid_argument", "expected_hashes must be an object.")
        _exact_fields(hashes, REMOVE_HASH_KEYS, "expected_hashes")
        for key in REMOVE_HASH_KEYS:
            _string(hashes[key], f"expected_hashes.{key}", DIGEST_PATTERN)
    return dict(operation)


def validate_request(value: dict[str, Any], system: "SystemState") -> dict[str, Any]:
    _exact_fields(value, ENVELOPE_FIELDS, "request")
    if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise ProtocolError("unknown_schema", "Unsupported privileged request schema.")
    if type(value["policy_version"]) is not int or value["policy_version"] != POLICY_VERSION:
        raise ProtocolError("unknown_schema", "Unsupported privilege policy version.")
    _uuid_string(value["request_id"], "request_id")
    _uuid_string(value["transaction_id"], "transaction_id")
    _string(value["plan_digest"], "plan_digest", DIGEST_PATTERN)
    preconditions = value["preconditions"]
    if not isinstance(preconditions, dict):
        raise ProtocolError("invalid_argument", "preconditions must be an object.")
    _exact_fields(preconditions, PRECONDITION_FIELDS, "precondition")
    base_os = _string(preconditions["base_os"], "base_os")
    base_release = _string(preconditions["base_release"], "base_release")
    kernel_release = _string(preconditions["kernel_release"], "kernel_release", KERNEL_PATTERN)
    actual_os, actual_release = system.base_os_release()
    if (
        base_os != "ubuntu"
        or base_release not in SUPPORTED_BASE_RELEASES
        or (actual_os, actual_release) != (base_os, base_release)
    ):
        raise ProtocolError("unsupported_os", "Host OS precondition is unsupported or changed.")
    if kernel_release != system.kernel_release():
        raise ProtocolError("precondition_changed", "Running kernel changed after review.")
    operations = value["operations"]
    if not isinstance(operations, list) or not operations or len(operations) > MAX_OPERATIONS:
        raise ProtocolError("invalid_argument", "operations must be a non-empty bounded array.")
    validated = [validate_operation(operation, kernel_release) for operation in operations]
    ids = [operation["id"] for operation in validated]
    if len(ids) != len(set(ids)):
        raise ProtocolError("invalid_argument", "Operation IDs must be unique.")
    result = dict(value)
    result["operations"] = validated
    return result


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    timed_out: bool = False
    cancelled: bool = False


class FixedCommandExecutor:
    is_test_adapter = False

    def run(self, argv: list[str], timeout_seconds: int = COMMAND_TIMEOUT_SECONDS) -> CommandResult:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=FIXED_ENV,
            cwd="/",
            shell=False,
            start_new_session=True,
        )
        try:
            process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            self._terminate_group(process)
            return CommandResult(124, timed_out=True)
        except OperationCancelled:
            self._terminate_group(process)
            return CommandResult(130, cancelled=True)
        except BaseException:
            self._terminate_group(process)
            raise
        return CommandResult(process.returncode)

    @staticmethod
    def _terminate_group(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=2)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()


class SystemState:
    is_test_adapter = False

    def kernel_release(self) -> str:
        return os.uname().release

    def base_os_release(self) -> tuple[str, str]:
        fields: dict[str, str] = {}
        try:
            for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                fields[key] = value.strip().strip('"')
        except OSError:
            return "unknown", "unknown"
        distro = fields.get("ID", "")
        ubuntu_codename = fields.get("UBUNTU_CODENAME", "")
        if distro == "ubuntu":
            release = fields.get("VERSION_ID", "")
        elif distro == "linuxmint" or "ubuntu" in fields.get("ID_LIKE", "").split():
            release = {"jammy": "22.04", "noble": "24.04"}.get(ubuntu_codename, "")
        else:
            return "unknown", "unknown"
        return "ubuntu", release

    @staticmethod
    def _driver_for_device(device_number: int) -> str:
        path = Path(f"/sys/class/video4linux/video{device_number}/device/driver")
        try:
            return path.resolve(strict=True).name
        except OSError:
            return ""

    def device_conflicts(self, device_number: int, allow_loopback: bool) -> bool:
        device = Path(f"/sys/class/video4linux/video{device_number}")
        if not device.exists():
            return False
        return not (allow_loopback and self._driver_for_device(device_number) == "v4l2loopback")

    def loopback_busy(self) -> bool:
        loopback_nodes = set()
        for path in Path("/sys/class/video4linux").glob("video*"):
            suffix = path.name.removeprefix("video")
            if suffix.isdigit() and self._driver_for_device(int(suffix)) == "v4l2loopback":
                loopback_nodes.add(f"/dev/{path.name}")
        if not loopback_nodes:
            return False
        for fd_path in Path("/proc").glob("[0-9]*/fd/*"):
            try:
                if os.readlink(fd_path) in loopback_nodes:
                    return True
            except OSError:
                continue
        return False


class PersistenceStore:
    def __init__(
        self,
        etc_root: Path = Path("/etc"),
        *,
        test_mode: bool = False,
        required_uid: int = 0,
        required_gid: int = 0,
    ):
        resolved = etc_root.resolve()
        if test_mode and resolved == Path("/etc"):
            raise ValueError("test mode refuses the host /etc root")
        if not test_mode and resolved != Path("/etc"):
            raise ValueError("production mode requires the fixed /etc root")
        self.etc_root = resolved
        self.test_mode = test_mode
        self.required_uid = required_uid
        self.required_gid = required_gid

    def _open_dir(self, relative_dir: str) -> int:
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            return os.open(self.etc_root / relative_dir, flags)
        except OSError as exc:
            raise ProtocolError("unsafe_path_state", "Persistence directory is unavailable or unsafe.") from exc

    @staticmethod
    def _target_stat(directory_fd: int, name: str) -> os.stat_result | None:
        try:
            return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ProtocolError("unsafe_path_state", "Could not inspect persistence target.") from exc

    def _read_existing(self, directory_fd: int, name: str, marker: str) -> bytes | None:
        info = self._target_stat(directory_fd, name)
        if info is None:
            return None
        if not stat.S_ISREG(info.st_mode):
            raise ProtocolError("unsafe_path_state", "Persistence target is not a regular file.")
        if info.st_uid != self.required_uid:
            raise ProtocolError("ownership_mismatch", "Persistence target owner does not match.")
        if stat.S_IMODE(info.st_mode) != 0o644:
            raise ProtocolError("ownership_mismatch", "Persistence target mode does not match.")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(name, flags, dir_fd=directory_fd)
            with os.fdopen(fd, "rb") as handle:
                data = handle.read(MAX_REQUEST_BYTES + 1)
        except OSError as exc:
            raise ProtocolError("unsafe_path_state", "Could not safely read persistence target.") from exc
        if len(data) > MAX_REQUEST_BYTES or not data.startswith(marker.encode("utf-8")):
            raise ProtocolError("unsafe_path_state", "Existing persistence target is not StudioCast-owned.")
        return data

    def _atomic_write(self, logical_id: str, content: bytes) -> dict[str, Any]:
        relative_dir, name, marker = FILE_SPECS[logical_id]
        directory_fd = self._open_dir(relative_dir)
        temp_name = ""
        try:
            existing = self._read_existing(directory_fd, name, marker)
            if existing is not None and not self._is_generated_content(logical_id, existing):
                raise ProtocolError("unsafe_path_state", "Existing persistence content is not canonical.")
            temp_name = f".{name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(temp_name, flags, 0o600, dir_fd=directory_fd)
            try:
                offset = 0
                while offset < len(content):
                    offset += os.write(fd, content[offset:])
                os.fchmod(fd, 0o644)
                os.fchown(fd, self.required_uid, self.required_gid)
                os.fsync(fd)
            finally:
                os.close(fd)
            current = self._target_stat(directory_fd, name)
            if current is not None and (not stat.S_ISREG(current.st_mode) or current.st_uid != self.required_uid):
                raise ProtocolError("unsafe_path_state", "Persistence target changed during replacement.")
            os.replace(temp_name, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            temp_name = ""
            os.fsync(directory_fd)
        except ProtocolError:
            raise
        except OSError as exc:
            raise ProtocolError("operation_failed", "Atomic persistence write failed.") from exc
        finally:
            if temp_name:
                try:
                    os.unlink(temp_name, dir_fd=directory_fd)
                except OSError:
                    pass
            os.close(directory_fd)
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        return {
            "id": logical_id,
            "path": f"/etc/{relative_dir}/{name}",
            "sha256": digest,
            "mode": "0644",
            "uid": self.required_uid,
        }

    def write_v4l(self, device_number: int, label: str, exclusive_caps: bool) -> list[dict[str, Any]]:
        escaped_label = label.replace(" ", "\\ ")
        modules = (FILE_SPECS["v4l_modules_load"][2] + "v4l2loopback\n").encode("utf-8")
        modprobe = (
            FILE_SPECS["v4l_modprobe"][2]
            + "options v4l2loopback devices=1 "
            + f"video_nr={device_number} card_label={escaped_label} "
            + f"exclusive_caps={1 if exclusive_caps else 0}\n"
        ).encode("utf-8")
        # Validate both current targets before replacing either one.
        for logical_id in ("v4l_modules_load", "v4l_modprobe"):
            relative_dir, name, marker = FILE_SPECS[logical_id]
            directory_fd = self._open_dir(relative_dir)
            try:
                existing = self._read_existing(directory_fd, name, marker)
                if existing is not None and not self._is_generated_content(logical_id, existing):
                    raise ProtocolError("unsafe_path_state", "Existing persistence content is not canonical.")
            finally:
                os.close(directory_fd)
        files: list[dict[str, Any]] = []
        try:
            files.append(self._atomic_write("v4l_modules_load", modules))
            files.append(self._atomic_write("v4l_modprobe", modprobe))
        except ProtocolError as exc:
            if files:
                raise PersistenceMutationError(exc.reason_code, exc.message, files) from exc
            raise
        return files

    @staticmethod
    def _is_generated_content(logical_id: str, content: bytes) -> bool:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return False
        marker = FILE_SPECS[logical_id][2]
        if logical_id == "v4l_modules_load":
            return text == marker + "v4l2loopback\n"
        pattern = re.compile(
            re.escape(marker)
            + r"options v4l2loopback devices=1 video_nr=(?:[0-9]|[1-5][0-9]|6[0-3]) "
            + r"card_label=(?:[A-Za-z0-9._+()-]|\\ )+ exclusive_caps=[01]\n"
        )
        return pattern.fullmatch(text) is not None

    def remove_v4l(self, expected_hashes: Mapping[str, str]) -> list[dict[str, Any]]:
        pending: list[tuple[str, str, str, bytes | None, str | None]] = []
        # Validate every target before removing either one. This makes an owner,
        # marker, type, or hash mismatch fail without a partial deletion.
        for logical_id in ("v4l_modules_load", "v4l_modprobe"):
            relative_dir, name, marker = FILE_SPECS[logical_id]
            directory_fd = self._open_dir(relative_dir)
            try:
                data = self._read_existing(directory_fd, name, marker)
                actual = None if data is None else "sha256:" + hashlib.sha256(data).hexdigest()
                if actual is not None and actual != expected_hashes[logical_id]:
                    raise ProtocolError("hash_mismatch", "Persistence target hash does not match manifest.")
                pending.append((logical_id, relative_dir, name, data, actual))
            finally:
                os.close(directory_fd)
        files: list[dict[str, Any]] = []
        for logical_id, relative_dir, name, data, actual in pending:
            if data is None:
                files.append({"id": logical_id, "path": f"/etc/{relative_dir}/{name}", "removed": False})
                continue
            directory_fd = self._open_dir(relative_dir)
            try:
                # Revalidate after the preflight to close replacement races.
                current = self._read_existing(directory_fd, name, FILE_SPECS[logical_id][2])
                current_hash = None if current is None else "sha256:" + hashlib.sha256(current).hexdigest()
                if current_hash != actual:
                    raise ProtocolError("precondition_changed", "Persistence target changed before removal.")
                os.unlink(name, dir_fd=directory_fd)
                os.fsync(directory_fd)
                files.append({"id": logical_id, "path": f"/etc/{relative_dir}/{name}", "removed": True, "sha256": actual})
            except ProtocolError as exc:
                if any(item.get("removed") for item in files):
                    raise PersistenceMutationError(exc.reason_code, exc.message, files) from exc
                raise
            except OSError as exc:
                if any(item.get("removed") for item in files):
                    raise PersistenceMutationError("operation_failed", "Persistence removal failed.", files) from exc
                raise ProtocolError("operation_failed", "Persistence removal failed.") from exc
            finally:
                os.close(directory_fd)
        return files


def _operation_result(operation: Mapping[str, Any], status: str, changed: bool, reason: str, message: str, *, exit_code: int = 0, files: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": operation["id"],
        "type": operation["type"],
        "status": status,
        "changed": changed,
        "reason_code": reason,
        "exit_code": exit_code,
        "message": message,
        "files": files or [],
    }


class Helper:
    def __init__(self, executor: FixedCommandExecutor, system: SystemState, store: PersistenceStore, *, test_mode: bool = False):
        if test_mode:
            if not getattr(executor, "is_test_adapter", False) or not getattr(system, "is_test_adapter", False) or not store.test_mode:
                raise ValueError("test mode requires explicit fake adapters and redirected /etc")
        elif getattr(executor, "is_test_adapter", False) or getattr(system, "is_test_adapter", False) or store.test_mode:
            raise ValueError("production mode refuses test adapters")
        self.executor = executor
        self.system = system
        self.store = store

    def _run_command(self, operation: Mapping[str, Any], argv: list[str]) -> dict[str, Any]:
        result = self.executor.run(argv)
        if result.cancelled:
            return _operation_result(operation, "failed", False, "operation_cancelled", "Operation was cancelled.", exit_code=130)
        if result.timed_out:
            return _operation_result(operation, "failed", False, "operation_timeout", "Operation timed out.", exit_code=124)
        if result.returncode != 0:
            return _operation_result(operation, "failed", False, "operation_failed", "Fixed privileged command failed.", exit_code=result.returncode)
        return _operation_result(operation, "succeeded", True, "operation_succeeded", "Operation completed.")

    def execute_operation(self, operation: Mapping[str, Any]) -> dict[str, Any]:
        operation_type = operation["type"]
        if operation_type == "packages.ensure.v1":
            updated = self._run_command(operation, [APT_GET, "update"])
            if updated["status"] != "succeeded":
                updated["changed"] = False
                return updated
            return self._run_command(operation, [APT_GET, "install", "-y", "--", *operation["packages"]])
        if operation_type in ("v4l.module.load.v1", "v4l.module.reload.v1"):
            reload_module = operation_type == "v4l.module.reload.v1"
            if self.system.device_conflicts(operation["device_number"], allow_loopback=reload_module):
                return _operation_result(operation, "failed", False, "precondition_changed", "Virtual-camera device number is now occupied.", exit_code=3)
            if reload_module:
                if self.system.loopback_busy():
                    return _operation_result(operation, "failed", False, "precondition_changed", "v4l2loopback is busy and cannot be reloaded.", exit_code=3)
                unloaded = self._run_command(operation, [MODPROBE, "-r", "v4l2loopback"])
                if unloaded["status"] != "succeeded":
                    return unloaded
            argv = [
                MODPROBE,
                "v4l2loopback",
                "devices=1",
                f"video_nr={operation['device_number']}",
                f"card_label={operation['label']}",
                f"exclusive_caps={1 if operation['exclusive_caps'] else 0}",
            ]
            return self._run_command(operation, argv)
        try:
            if operation_type == "v4l.persistence.write.v1":
                files = self.store.write_v4l(operation["device_number"], operation["label"], operation["exclusive_caps"])
                return _operation_result(operation, "succeeded", True, "operation_succeeded", "StudioCast persistence files were written.", files=files)
            files = self.store.remove_v4l(operation["expected_hashes"])
            return _operation_result(operation, "succeeded", any(item.get("removed") for item in files), "operation_succeeded", "StudioCast persistence files were reconciled.", files=files)
        except PersistenceMutationError as exc:
            return _operation_result(operation, "failed", True, "partial_failure", exc.message, exit_code=4, files=exc.files)
        except ProtocolError as exc:
            return _operation_result(operation, "failed", False, exc.reason_code, exc.message, exit_code=3)

    def handle(self, raw: bytes) -> tuple[dict[str, Any], int]:
        request_hint: dict[str, Any] = {}
        try:
            request_hint = parse_request(raw)
            request = validate_request(request_hint, self.system)
        except ProtocolError as exc:
            return _result_envelope(request_hint, "failed", [], exc.reason_code, exc.message), 2
        results = []
        changed = False
        for operation in request["operations"]:
            try:
                result = self.execute_operation(operation)
            except OperationCancelled:
                result = _operation_result(
                    operation,
                    "failed",
                    False,
                    "operation_cancelled",
                    "Operation was cancelled.",
                    exit_code=130,
                )
            results.append(result)
            changed = changed or result["changed"]
            if result["status"] != "succeeded":
                state = "partial_failure" if changed else "failed"
                reason = "partial_failure" if changed else result["reason_code"]
                return _result_envelope(request, state, results, reason, result["message"]), 4 if changed else 3
        return _result_envelope(request, "succeeded", results, "operation_succeeded", "All privileged operations completed."), 0


def _safe_hint(value: Mapping[str, Any], name: str) -> str | None:
    candidate = value.get(name)
    return candidate if isinstance(candidate, str) and len(candidate) <= 128 else None


def _result_envelope(request: Mapping[str, Any], state: str, results: list[dict[str, Any]], reason_code: str, message: str, *, authorization_status: str = "granted", authorization_reason: str = "authorization_granted") -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": _safe_hint(request, "request_id"),
        "transaction_id": _safe_hint(request, "transaction_id"),
        "plan_digest": _safe_hint(request, "plan_digest"),
        "helper_version": HELPER_VERSION,
        "authorization": {"status": authorization_status, "reason_code": authorization_reason},
        "state": state,
        "reason_code": reason_code,
        "message": message,
        "results": results,
    }


def _print_json(value: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def main(argv: list[str]) -> int:
    if argv == ["--version", "--json"]:
        _print_json({"helper_version": HELPER_VERSION, "protocol_versions": [SCHEMA_VERSION]})
        return 0
    if argv != ["--json-stdin"]:
        _print_json(_result_envelope({}, "failed", [], "invalid_argument", "Only --json-stdin is supported."))
        return 2
    if os.geteuid() != 0:
        _print_json(
            _result_envelope(
                {},
                "failed",
                [],
                "authorization_unavailable",
                "The trusted helper must be launched through polkit.",
                authorization_status="unavailable",
                authorization_reason="authorization_unavailable",
            )
        )
        return 126
    installed_path = Path("/usr/libexec/studiocast/studiocast-system-helper")
    try:
        own_info = os.lstat(__file__)
        trusted_self = (
            Path(__file__).resolve() == installed_path
            and stat.S_ISREG(own_info.st_mode)
            and own_info.st_uid == 0
            and not own_info.st_mode & 0o022
        )
    except OSError:
        trusted_self = False
    if not trusted_self:
        _print_json(
            _result_envelope(
                {},
                "failed",
                [],
                "privileged_helper_untrusted",
                "Refusing to mutate the host from an untrusted helper location.",
                authorization_status="unavailable",
                authorization_reason="privileged_helper_untrusted",
            )
        )
        return 126
    def cancel_operation(_signum: int, _frame: Any) -> NoReturn:
        raise OperationCancelled()

    signal.signal(signal.SIGTERM, cancel_operation)
    signal.signal(signal.SIGINT, cancel_operation)
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    helper = Helper(FixedCommandExecutor(), SystemState(), PersistenceStore())
    result, exit_code = helper.handle(raw)
    _print_json(result)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
