#!/usr/bin/env python3
"""Hermetic security tests for release archive and AppImage verification."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import struct
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
APPIMAGE_VERIFIER = ROOT / "packaging/appimage/verify_type2_appimage.py"
SOURCE_VERIFIER = ROOT / "packaging/release/verify_canonical_source_archive.sh"


def synthetic_runtime() -> bytes:
    result = bytearray(4096)
    names = b"\0.shstrtab\0.digest_md5\0"
    section_offset = 256
    strings_offset = 512
    digest_offset = 1024
    identity = bytearray(16)
    identity[:7] = b"\x7fELF\x02\x01\x01"
    identity[8:11] = b"AI\x02"
    result[:64] = struct.pack(
        "<16sHHIQQQIHHHHHH",
        bytes(identity), 3, 62, 1, 0, 0, section_offset, 0, 64, 0, 0, 64, 3, 1,
    )
    sections = [
        bytes(64),
        struct.pack(
            "<IIQQQQIIQQ", names.index(b".shstrtab"), 3, 0, 0,
            strings_offset, len(names), 0, 0, 1, 0,
        ),
        struct.pack(
            "<IIQQQQIIQQ", names.index(b".digest_md5"), 1, 3, 0,
            digest_offset, 16, 0, 0, 16, 0,
        ),
    ]
    for index, section in enumerate(sections):
        begin = section_offset + index * 64
        result[begin : begin + 64] = section
    result[strings_offset : strings_offset + len(names)] = names
    return bytes(result)


@unittest.skipUnless(shutil.which("mksquashfs") and Path("/usr/bin/unsquashfs").is_file(),
                     "squashfs-tools are required")
class Type2AppImageVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="studiocast-appimage-security-")
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime-x86_64"
        self.runtime.write_bytes(synthetic_runtime())
        self.runtime_sha256 = hashlib.sha256(self.runtime.read_bytes()).hexdigest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def appdir(self, name: str = "Fixture.AppDir") -> Path:
        root = self.root / name
        (root / "usr/bin").mkdir(parents=True)
        executable = root / "usr/bin/studiocast-installer"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        (root / "usr/share/studiocast/source").mkdir(parents=True)
        (root / "usr/share/studiocast/source/source.tar.gz").write_bytes(b"source")
        (root / "AppRun").symlink_to("usr/bin/studiocast-installer")
        return root

    def archive(self, appdir: Path, name: str = "appdir.tar.gz") -> Path:
        output = self.root / name
        with tarfile.open(output, "w:gz", format=tarfile.PAX_FORMAT) as archive:
            archive.add(appdir, arcname=appdir.name, recursive=True)
        return output

    def payload(self, appdir: Path, name: str = "payload.squashfs") -> Path:
        output = self.root / name
        result = subprocess.run(
            [
                "mksquashfs", str(appdir), str(output), "-noappend", "-no-xattrs",
                "-all-root", "-mkfs-time", "0", "-all-time", "0", "-quiet",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        return output

    def image(self, payload: Path, name: str = "fixture.AppImage") -> Path:
        output = self.root / name
        runtime = bytearray(self.runtime.read_bytes())
        runtime[1024:1040] = bytes(range(16))
        output.write_bytes(runtime + payload.read_bytes())
        output.chmod(0o755)
        return output

    def verify(self, image: Path, archive: Path, name: str = "extracted") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python3", str(APPIMAGE_VERIFIER),
                "--appimage", str(image),
                "--runtime-file", str(self.runtime),
                "--runtime-sha256", self.runtime_sha256,
                "--appdir-tar", str(archive),
                "--extract-dir", str(self.root / name),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_valid_runtime_and_payload_are_verified_without_execution(self) -> None:
        appdir = self.appdir()
        result = self.verify(self.image(self.payload(appdir)), self.archive(appdir))
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('"verification": "studiocast-type2-appimage/v1"', result.stdout)

    def test_malicious_runtime_and_ambiguous_squashfs_are_rejected(self) -> None:
        appdir = self.appdir()
        archive = self.archive(appdir)
        payload = self.payload(appdir)
        malicious = bytearray(self.image(payload, "malicious.AppImage").read_bytes())
        malicious[128] ^= 0x40
        malicious_path = self.root / "modified-runtime.AppImage"
        malicious_path.write_bytes(malicious)
        result = self.verify(malicious_path, archive, "malicious-output")
        self.assertEqual(2, result.returncode)
        self.assertIn("runtime differs from the pinned runtime", result.stderr)

        ambiguous = self.image(payload, "ambiguous.AppImage")
        with ambiguous.open("ab") as stream:
            stream.write(payload.read_bytes())
        result = self.verify(ambiguous, archive, "ambiguous-output")
        self.assertEqual(2, result.returncode)
        self.assertIn("exactly one valid SquashFS", result.stderr)

    def test_payload_mismatch_and_escaping_symlink_are_rejected(self) -> None:
        appdir = self.appdir()
        archive = self.archive(appdir)
        (appdir / "usr/share/studiocast/source/source.tar.gz").write_bytes(b"changed")
        result = self.verify(self.image(self.payload(appdir)), archive)
        self.assertEqual(2, result.returncode)
        self.assertIn("payload content mismatch", result.stderr)

        unsafe = self.appdir("Unsafe.AppDir")
        (unsafe / "escape").symlink_to("../../outside")
        result = self.verify(
            self.image(self.payload(unsafe, "unsafe.squashfs"), "unsafe.AppImage"),
            self.archive(unsafe, "unsafe.tar.gz"),
            "unsafe-output",
        )
        self.assertEqual(2, result.returncode)
        self.assertIn("escaping", result.stderr)

    def test_intermediate_symlink_escape_chain_is_rejected(self) -> None:
        valid = self.appdir("ValidLinks.AppDir")
        (valid / "links/real").mkdir(parents=True)
        (valid / "links/real/value").write_text("safe\n", encoding="utf-8")
        (valid / "links/alias").symlink_to("real")
        (valid / "links/final").symlink_to("alias/value")
        result = self.verify(
            self.image(self.payload(valid, "valid-links.squashfs"), "valid-links.AppImage"),
            self.archive(valid, "valid-links.tar.gz"),
            "valid-links-output",
        )
        self.assertEqual(0, result.returncode, result.stderr)

        unsafe = self.appdir("EscapeChain.AppDir")
        (unsafe / "a").mkdir()
        (unsafe / "a/x").symlink_to("../a/..")
        (unsafe / "a/l").symlink_to("../a/x/..")
        result = self.verify(
            self.image(
                self.payload(unsafe, "escape-chain.squashfs"),
                "escape-chain.AppImage",
            ),
            self.archive(unsafe, "escape-chain.tar.gz"),
            "escape-chain-output",
        )
        self.assertEqual(2, result.returncode)
        self.assertIn("escaping", result.stderr)

    def test_dangling_cyclic_and_non_directory_symlinks_are_rejected(self) -> None:
        cases = [
            ("Dangling", {"bad": "missing"}, "dangling"),
            ("Cyclic", {"one": "two", "two": "one"}, "cyclic"),
            (
                "NonDirectory",
                {"bad": "usr/bin/studiocast-installer/child"},
                "non-directory",
            ),
        ]
        for name, links, expected_error in cases:
            with self.subTest(name=name):
                appdir = self.appdir(f"{name}.AppDir")
                for link, target in links.items():
                    (appdir / link).symlink_to(target)
                result = self.verify(
                    self.image(
                        self.payload(appdir, f"{name}.squashfs"),
                        f"{name}.AppImage",
                    ),
                    self.archive(appdir, f"{name}.tar.gz"),
                    f"{name}-output",
                )
                self.assertEqual(2, result.returncode)
                self.assertIn(expected_error, result.stderr)


class PrivateKeyMarkerScanTests(unittest.TestCase):
    def test_binary_private_key_markers_are_not_skipped(self) -> None:
        verifier = (ROOT / "packaging/appimage/verify_bundle.sh").read_text(
            encoding="utf-8"
        )
        workflow = (ROOT / ".github/workflows/release-packaging.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("grep -IlE", verifier)
        self.assertNotIn("grep -IlE", workflow)
        self.assertIn("grep -alE", verifier)
        self.assertIn("grep -alE", workflow)

        with tempfile.TemporaryDirectory(prefix="studiocast-pem-scan-") as root:
            artifact = Path(root) / "binary-artifact"
            artifact.write_bytes(b"binary\0-----BEGIN PRIVATE KEY-----\0payload")
            result = subprocess.run(
                [
                    "grep", "-alE", "--",
                    r"-----BEGIN ([A-Z0-9]+ )*PRIVATE KEY-----", str(artifact),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(str(artifact), result.stdout.strip())


class CanonicalSourceArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="studiocast-source-security-")
        self.root = Path(self.temporary.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "config", "user.name", "Fixture"], check=True)
        subprocess.run(["git", "-C", str(self.root), "config", "user.email", "fixture@example.invalid"], check=True)
        (self.root / "VERSION").write_text("1.2.3\n", encoding="utf-8")
        (self.root / "payload.txt").write_text("trusted\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.root), "add", "VERSION", "payload.txt"], check=True)
        subprocess.run(["git", "-C", str(self.root), "commit", "-q", "-m", "fixture"], check=True)
        self.commit = subprocess.check_output(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True
        ).strip()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def verifier(self, archive: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(SOURCE_VERIFIER), str(archive), self.commit, "1.2.3", str(self.root)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_canonical_archive_passes_and_forged_pax_commit_archive_fails(self) -> None:
        canonical = self.root / "canonical.tar.gz"
        subprocess.run(
            [
                "git", "-C", str(self.root), "archive", "--format=tar.gz",
                "--prefix=StudioCast-1.2.3/", f"--output={canonical}", self.commit,
            ],
            check=True,
        )
        self.assertEqual(0, self.verifier(canonical).returncode)

        forged = self.root / "forged.tar.gz"
        with tarfile.open(
            forged, "w:gz", format=tarfile.PAX_FORMAT,
            pax_headers={"comment": self.commit},
        ) as archive:
            version = tarfile.TarInfo("StudioCast-1.2.3/VERSION")
            version.mode = 0o644
            version.size = len(b"1.2.3\n")
            archive.addfile(version, io.BytesIO(b"1.2.3\n"))
            payload = tarfile.TarInfo("StudioCast-1.2.3/payload.txt")
            payload.mode = 0o644
            payload.size = len(b"forged\n")
            archive.addfile(payload, io.BytesIO(b"forged\n"))

        claimed = subprocess.run(
            ["git", "get-tar-commit-id"],
            input=subprocess.check_output(["gzip", "-cd", str(forged)]),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(0, claimed.returncode, claimed.stderr.decode())
        self.assertEqual(self.commit, claimed.stdout.decode().strip())
        rejected = self.verifier(forged)
        self.assertEqual(2, rejected.returncode)
        self.assertIn("not the canonical git archive", rejected.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
