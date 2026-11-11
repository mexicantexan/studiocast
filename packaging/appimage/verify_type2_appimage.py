#!/usr/bin/env python3
"""Verify and extract a type-2 AppImage without executing its runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import stat
import struct
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path


UNSQUASHFS = Path("/usr/bin/unsquashfs")
MAX_APPIMAGE_BYTES = 4 * 1024 * 1024 * 1024
MAX_RUNTIME_BYTES = 16 * 1024 * 1024
MAX_SQUASHFS_CANDIDATES = 64
MAX_SYMLINK_EXPANSIONS = 40
SQUASHFS_MAGIC = b"hsqs"
AI_TYPE2_MAGIC = b"AI\x02"


class VerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Entry:
    kind: str
    mode: int
    size: int = 0
    sha256: str = ""
    target: str = ""


def fail(message: str) -> None:
    raise VerificationError(message)


def regular_file(path: Path, label: str, maximum: int | None = None) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        fail(f"{label} is unavailable: {path}: {exc}")
    if not stat.S_ISREG(value.st_mode):
        fail(f"{label} must be a regular non-symlink file: {path}")
    if maximum is not None and value.st_size > maximum:
        fail(f"{label} exceeds the {maximum}-byte verification limit: {path}")
    return value


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_runtime(path: Path, label: str, limit: int) -> tuple[int, int]:
    size = regular_file(path, label, limit).st_size
    if size < 64:
        fail(f"{label} is too small to be an ELF64 AppImage runtime")
    with path.open("rb") as stream:
        header = stream.read(64)
        if header[:4] != b"\x7fELF":
            fail(f"{label} is not ELF")
        if header[4:7] != b"\x02\x01\x01":
            fail(f"{label} must be little-endian ELF64")
        if header[8:11] != AI_TYPE2_MAGIC:
            fail(f"{label} does not contain AppImage type-2 magic")
        fields = struct.unpack("<16sHHIQQQIHHHHHH", header)
        _, elf_type, machine, version, _, _, section_offset, _, header_size, _, _, section_size, section_count, strings_index = fields
        if elf_type != 3 or machine != 62 or version != 1 or header_size != 64:
            fail(f"{label} is not the expected x86-64 PIE runtime")
        if section_size != 64 or not (2 <= section_count <= 256):
            fail(f"{label} has an invalid ELF section table")
        table_end = section_offset + section_size * section_count
        if section_offset < 64 or table_end > size or strings_index >= section_count:
            fail(f"{label} ELF section table is out of bounds")
        stream.seek(section_offset)
        sections = [stream.read(section_size) for _ in range(section_count)]
        if any(len(section) != section_size for section in sections):
            fail(f"{label} ELF section table is truncated")

        def section_fields(index: int) -> tuple[int, int, int, int, int]:
            name, kind, flags, _, offset, length = struct.unpack(
                "<IIQQQQ", sections[index][:40]
            )
            if kind != 8 and offset + length > size:
                fail(f"{label} ELF section {index} is out of bounds")
            return name, kind, flags, offset, length

        _, string_kind, _, string_offset, string_length = section_fields(strings_index)
        if string_kind != 3 or string_length > 1024 * 1024:
            fail(f"{label} ELF section-name table is invalid")
        stream.seek(string_offset)
        names = stream.read(string_length)

        digest_sections: list[tuple[int, int]] = []
        for index in range(section_count):
            name_offset, kind, flags, offset, length = section_fields(index)
            if name_offset >= len(names):
                fail(f"{label} ELF section name is out of bounds")
            end = names.find(b"\0", name_offset)
            if end < 0:
                fail(f"{label} ELF section name is unterminated")
            name = names[name_offset:end]
            if name == b".digest_md5":
                if kind != 1 or flags & 0x4 or length != 16:
                    fail(f"{label} .digest_md5 section is unsafe")
                digest_sections.append((offset, length))
        if len(digest_sections) != 1:
            fail(f"{label} must contain exactly one .digest_md5 section")
        return digest_sections[0]


def compare_runtime_prefix(appimage: Path, runtime: Path, expected_sha256: str) -> int:
    runtime_size = regular_file(runtime, "pinned runtime", MAX_RUNTIME_BYTES).st_size
    if sha256_path(runtime) != expected_sha256:
        fail("pinned runtime SHA-256 does not match the release lock")
    runtime_digest = parse_runtime(runtime, "pinned runtime", MAX_RUNTIME_BYTES)
    appimage_size = regular_file(appimage, "AppImage", MAX_APPIMAGE_BYTES).st_size
    if appimage_size <= runtime_size:
        fail("AppImage has no payload after its runtime prefix")
    appimage_digest = parse_runtime(appimage, "AppImage runtime", MAX_APPIMAGE_BYTES)
    if appimage_digest != runtime_digest:
        fail("AppImage runtime ELF metadata differs from the pinned runtime")

    digest_start, digest_size = runtime_digest
    with runtime.open("rb") as expected, appimage.open("rb") as actual:
        position = 0
        while position < runtime_size:
            length = min(1024 * 1024, runtime_size - position)
            expected_bytes = bytearray(expected.read(length))
            actual_bytes = actual.read(length)
            if len(expected_bytes) != length or len(actual_bytes) != length:
                fail("AppImage runtime prefix is truncated")
            overlap_start = max(position, digest_start)
            overlap_end = min(position + length, digest_start + digest_size)
            if overlap_start < overlap_end:
                begin = overlap_start - position
                end = overlap_end - position
                expected_bytes[begin:end] = actual_bytes[begin:end]
            if bytes(expected_bytes) != actual_bytes:
                fail("AppImage executable runtime differs from the pinned runtime")
            position += length
    return runtime_size


def squashfs_magic_offsets(path: Path) -> list[int]:
    offsets: list[int] = []
    overlap = b""
    position = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            data = overlap + chunk
            base = position - len(overlap)
            start = 0
            while True:
                found = data.find(SQUASHFS_MAGIC, start)
                if found < 0:
                    break
                offsets.append(base + found)
                if len(offsets) > MAX_SQUASHFS_CANDIDATES:
                    fail("AppImage contains too many SquashFS magic candidates")
                start = found + 1
            overlap = data[-(len(SQUASHFS_MAGIC) - 1) :]
            position += len(chunk)
    return sorted(set(offsets))


def valid_squashfs_offsets(path: Path) -> list[int]:
    if not UNSQUASHFS.is_file() or not os.access(UNSQUASHFS, os.X_OK):
        fail(f"trusted unsquashfs is unavailable: {UNSQUASHFS}")
    valid: list[int] = []
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    for offset in squashfs_magic_offsets(path):
        try:
            result = subprocess.run(
                [str(UNSQUASHFS), "-s", "-o", str(offset), str(path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            fail(f"SquashFS validation timed out at offset {offset}")
        if result.returncode == 0:
            valid.append(offset)
    return valid


def validate_squashfs_extent(appimage: Path, offset: int) -> None:
    with appimage.open("rb") as stream:
        stream.seek(offset)
        superblock = stream.read(96)
        if len(superblock) != 96 or superblock[:4] != SQUASHFS_MAGIC:
            fail("SquashFS superblock is truncated")
        major, minor = struct.unpack_from("<HH", superblock, 28)
        bytes_used = struct.unpack_from("<Q", superblock, 40)[0]
        if (major, minor) != (4, 0) or bytes_used < 96:
            fail("SquashFS superblock version or size is invalid")
        padded_size = (bytes_used + 4095) & ~4095
        expected_size = offset + padded_size
        actual_size = appimage.stat().st_size
        if expected_size != actual_size:
            fail("AppImage has ambiguous or trailing data after its SquashFS payload")
        stream.seek(offset + bytes_used)
        remaining = padded_size - bytes_used
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk or any(chunk):
                fail("AppImage SquashFS padding is not canonical zero padding")
            remaining -= len(chunk)


def safe_relative(value: str, label: str, allow_root: bool = False) -> str:
    if not value or value.startswith("/") or "\\" in value or "\0" in value:
        fail(f"unsafe {label}: {value!r}")
    normalized = posixpath.normpath(value)
    if normalized == "." and allow_root:
        return ""
    if normalized != value or normalized == ".." or normalized.startswith("../"):
        fail(f"unsafe {label}: {value!r}")
    if any(part in ("", ".", "..") for part in value.split("/")):
        fail(f"unsafe {label}: {value!r}")
    return value


def hash_stream(stream) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def tar_manifest(path: Path) -> dict[str, Entry]:
    regular_file(path, "AppDir archive", MAX_APPIMAGE_BYTES)
    entries: dict[str, Entry] = {}
    root: str | None = None
    try:
        archive = tarfile.open(path, "r:gz")
    except (OSError, tarfile.TarError) as exc:
        fail(f"AppDir archive is unreadable: {exc}")
    with archive:
        for member in archive.getmembers():
            raw_name = member.name[:-1] if member.name.endswith("/") else member.name
            safe_relative(raw_name, "AppDir archive path")
            first = raw_name.split("/", 1)[0]
            if root is None:
                root = first
            if first != root:
                fail("AppDir archive contains more than one top-level root")
            relative = raw_name[len(root) :].removeprefix("/")
            if relative in entries:
                fail(f"duplicate AppDir archive path: {relative or root}")
            mode = member.mode & 0o7777
            if member.isdir():
                entry = Entry("dir", mode)
            elif member.isreg():
                stream = archive.extractfile(member)
                if stream is None:
                    fail(f"could not read AppDir archive file: {raw_name}")
                entry = Entry("file", mode, member.size, hash_stream(stream))
            elif member.issym():
                target = safe_relative(member.linkname, "AppDir symlink target") if not member.linkname.startswith("../") else member.linkname
                entry = Entry("symlink", mode, target=target)
            else:
                fail(f"unsupported AppDir archive entry type: {raw_name}")
            entries[relative] = entry
    if root is None or "" not in entries or entries[""].kind != "dir":
        fail("AppDir archive has no explicit top-level directory")
    validate_symlinks(entries, "AppDir archive")
    return entries


def filesystem_manifest(root: Path) -> dict[str, Entry]:
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode) or root.is_symlink():
        fail("extracted SquashFS root is not a real directory")
    entries = {"": Entry("dir", stat.S_IMODE(root_stat.st_mode))}

    def visit(directory: Path, relative_parent: str) -> None:
        for item in sorted(os.scandir(directory), key=lambda entry: entry.name):
            safe_relative(item.name, "SquashFS path component")
            relative = f"{relative_parent}/{item.name}" if relative_parent else item.name
            value = item.stat(follow_symlinks=False)
            mode = stat.S_IMODE(value.st_mode)
            if stat.S_ISDIR(value.st_mode):
                entries[relative] = Entry("dir", mode)
                visit(Path(item.path), relative)
            elif stat.S_ISREG(value.st_mode):
                file_path = Path(item.path)
                entries[relative] = Entry(
                    "file", mode, value.st_size, sha256_path(file_path)
                )
            elif stat.S_ISLNK(value.st_mode):
                entries[relative] = Entry(
                    "symlink", mode, target=os.readlink(item.path)
                )
            else:
                fail(f"unsupported SquashFS entry type: {relative}")

    visit(root, "")
    validate_symlinks(entries, "SquashFS payload")
    return entries


def symlink_target_components(link: str, target: str, label: str) -> list[str]:
    if not target or target.startswith("/") or "\\" in target or "\0" in target:
        fail(f"unsafe {label} symlink target at {link}: {target!r}")
    return target.split("/")


def validate_symlinks(entries: dict[str, Entry], label: str) -> None:
    for link, entry in entries.items():
        if entry.kind != "symlink":
            continue
        parent = posixpath.dirname(link)
        resolved = parent.split("/") if parent else []
        pending = symlink_target_components(link, entry.target, label)
        expansions = 0
        while pending:
            component = pending.pop(0)
            if component in ("", "."):
                continue
            if component == "..":
                if not resolved:
                    fail(f"escaping {label} symlink target at {link}: {entry.target!r}")
                resolved.pop()
                continue

            current = "/".join([*resolved, component])
            target_entry = entries.get(current)
            if target_entry is None:
                fail(f"dangling {label} symlink at {link}: {entry.target!r}")
            if target_entry.kind == "symlink":
                expansions += 1
                if expansions > MAX_SYMLINK_EXPANSIONS:
                    fail(f"cyclic or excessive {label} symlink chain at {link}")
                pending = symlink_target_components(
                    current, target_entry.target, label
                ) + pending
            elif pending and target_entry.kind != "dir":
                fail(f"{label} symlink traverses a non-directory at {current}")
            else:
                resolved.append(component)


def compare_manifests(expected: dict[str, Entry], actual: dict[str, Entry]) -> None:
    if expected.keys() != actual.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        fail(f"AppImage payload topology mismatch; missing={missing[:5]}, extra={extra[:5]}")
    for name, wanted in expected.items():
        found = actual[name]
        if wanted.kind != found.kind:
            fail(f"AppImage payload type mismatch: {name or '.'}")
        if wanted.kind in ("file", "dir") and wanted.mode != found.mode:
            fail(f"AppImage payload mode mismatch: {name or '.'}")
        if wanted.kind == "file" and (
            wanted.size != found.size or wanted.sha256 != found.sha256
        ):
            fail(f"AppImage payload content mismatch: {name}")
        if wanted.kind == "symlink" and wanted.target != found.target:
            fail(f"AppImage payload symlink mismatch: {name}")


def extract_and_verify(args: argparse.Namespace) -> dict[str, object]:
    expected_sha256 = args.runtime_sha256.lower()
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        fail("runtime SHA-256 must be 64 lowercase hexadecimal characters")
    runtime_size = compare_runtime_prefix(args.appimage, args.runtime_file, expected_sha256)
    offsets = valid_squashfs_offsets(args.appimage)
    if offsets != [runtime_size]:
        fail(f"AppImage must contain exactly one valid SquashFS at offset {runtime_size}; found {offsets}")
    validate_squashfs_extent(args.appimage, runtime_size)

    output = args.extract_dir
    if output.exists() or output.is_symlink():
        fail(f"extraction destination must not exist: {output}")
    parent = output.parent
    parent_stat = parent.lstat()
    if not stat.S_ISDIR(parent_stat.st_mode) or parent.is_symlink():
        fail(f"extraction parent must be a real directory: {parent}")
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    try:
        result = subprocess.run(
            [
                str(UNSQUASHFS), "-no-xattrs", "-no-progress", "-dest", str(output),
                "-o", str(runtime_size), str(args.appimage),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        fail("SquashFS extraction timed out")
    if result.returncode != 0:
        fail(f"trusted unsquashfs extraction failed: {result.stderr.decode('utf-8', 'replace').strip()}")

    expected = tar_manifest(args.appdir_tar)
    actual = filesystem_manifest(output)
    compare_manifests(expected, actual)
    return {
        "schema_version": 1,
        "verification": "studiocast-type2-appimage/v1",
        "runtime_sha256": expected_sha256,
        "runtime_size": runtime_size,
        "payload_entries": len(actual),
        "extract_dir": str(output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appimage", type=Path, required=True)
    parser.add_argument("--runtime-file", type=Path, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--appdir-tar", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        result = extract_and_verify(parse_args())
    except (OSError, VerificationError, ValueError, struct.error) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
