#!/usr/bin/env python3
"""Hermetic checks of the fixed CI-only ONNX Runtime archive boundary."""
import hashlib
import importlib.util
import io
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "prepare_ci_onnxruntime", ROOT / "scripts/dev/prepare_ci_onnxruntime.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class PreparationTests(unittest.TestCase):
    def fixture(self, changes=None, extra=()):
        members = {name: b"fixture" for name in helper.MEMBERS}
        members["build/native/include/onnxruntime_c_api.h"] = b"#define ORT_API_VERSION 17\n"
        members[helper.NUSPEC] = (
            b'<package xmlns="http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd">'
            b'<metadata><id>Microsoft.ML.OnnxRuntime</id><version>1.17.3</version>'
            b'<authors>Microsoft</authors><owners>Microsoft</owners></metadata></package>')
        members.update(changes or {})
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as tree, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for name, content in list(members.items()) + list(extra):
                if content is not None:
                    tree.writestr(name, content)
        return buffer.getvalue()

    def prepare_fixture(self, raw, archive, output):
        archive.write_bytes(raw)
        with patch.object(helper, "ARCHIVE_SIZE", len(raw)), patch.object(
                helper, "ARCHIVE_SHA256", hashlib.sha256(raw).hexdigest()):
            helper.prepare(archive, output)

    def test_verified_allowlist_and_relative_soname_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare_fixture(self.fixture(extra=(("unused/platform.bin", b"unused"),)),
                                 root / "sdk.nupkg", root / "output")
            files = {str(p.relative_to(root / "output")) for p in (root / "output").rglob("*")
                     if p.is_file() or p.is_symlink()}
            self.assertEqual(files, set(helper.MEMBERS.values()) | {"lib/libonnxruntime.so"})
            self.assertEqual((root / "output/lib/libonnxruntime.so").readlink(),
                             Path("libonnxruntime.so.1.17.3"))
            self.assertEqual((root / "output/LICENSE").read_bytes(), b"fixture")

    def test_untrusted_digest_rejected_before_zip_parse_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "sdk.nupkg"
            archive.write_bytes(b"not a zip")
            with patch.object(helper, "ARCHIVE_SIZE", archive.stat().st_size), patch.object(
                    helper.zipfile, "ZipFile", side_effect=AssertionError("parsed untrusted ZIP")):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    helper.prepare(archive, root / "output")
            with self.assertRaisesRegex(ValueError, "size"):
                helper.prepare(archive, root / "output")
            self.assertFalse((root / "output").exists())

    def test_malformed_missing_and_unsafe_members_fail_before_output(self):
        special = zipfile.ZipInfo("special")
        special.create_system = 3
        special.external_attr = (stat.S_IFIFO | 0o600) << 16
        link = zipfile.ZipInfo("link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        cases = [(self.fixture(changes={name: None}), "missing " + name)
                 for name in tuple(helper.MEMBERS) + (helper.NUSPEC,)]
        cases += [(self.fixture(extra=((name, b"bad"),)), str(name)) for name in
                  ("../escape", "/absolute", "dir/../escape", "dir\\escape", "a//b",
                   "C:/drive", "LICENSE", link, special, "LICENSE/child")]
        cases += [(self.fixture(changes={helper.NUSPEC: b'<package><metadata><id>other</id>'
                                        b'</metadata></package>'}), "identity"),
                  (self.fixture(changes={"build/native/include/onnxruntime_c_api.h":
                                         b"#define ORT_API_VERSION 18\n"}), "API")]
        # Set encryption flags in both local and central headers of a valid ZIP.
        encrypted = bytearray(self.fixture())
        offset = encrypted.index(b"PK\x01\x02")
        encrypted[6] |= 1
        encrypted[offset + 8] |= 1
        cases.append((bytes(encrypted), "encrypted"))
        nul_name = self.fixture(extra=(("bad_name", b"bad"),)).replace(b"bad_name", b"bad\x00name")
        cases.append((nul_name, "NUL name"))
        for raw, label in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.assertRaises(ValueError):
                    self.prepare_fixture(raw, root / "sdk.nupkg", root / "output")
                self.assertFalse((root / "output").exists())

    def test_existing_destination_and_symlink_boundaries_preserve_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "existing"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_bytes(b"preserve")
            parent = root / "symlink-parent"
            parent.symlink_to(output, target_is_directory=True)
            dest_link = root / "dangling"
            dest_link.symlink_to(root / "absent")
            for target in (output, sentinel, parent / "new", dest_link, root / "missing/child",
                           root / "existing/../new"):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    self.prepare_fixture(self.fixture(), root / "sdk.nupkg", target)
            archive_link = root / "linked.nupkg"
            archive_link.symlink_to(root / "sdk.nupkg")
            with self.assertRaises(ValueError):
                helper.prepare(archive_link, root / "new")
            archive_hardlink = root / "hardlinked.nupkg"
            archive_hardlink.hardlink_to(root / "sdk.nupkg")
            with self.assertRaises(ValueError):
                helper.prepare(archive_hardlink, root / "new")
            self.assertEqual(sentinel.read_bytes(), b"preserve")
            self.assertFalse((output / "new").exists())


if __name__ == "__main__":
    unittest.main()
