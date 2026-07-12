#!/usr/bin/env python3
"""Hermetic tests for the StudioCast signed stable release channel."""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/data/installer_release"
sys.path.insert(0, str(ROOT / "packaging/release"))

import release_channel as rc  # noqa: E402


class RecordingTransport(rc.Transport):
    def __init__(self, source: pathlib.Path):
        self.source = source
        self.offsets: list[int] = []

    def append(self, url: str, destination: pathlib.Path, offset: int) -> None:
        self.offsets.append(offset)
        with self.source.open("rb") as incoming, destination.open("ab") as outgoing:
            incoming.seek(offset)
            outgoing.write(incoming.read())


class InterruptingTransport(RecordingTransport):
    def __init__(self, source: pathlib.Path):
        super().__init__(source)
        self.interrupted = False

    def append(self, url: str, destination: pathlib.Path, offset: int) -> None:
        self.offsets.append(offset)
        with self.source.open("rb") as incoming, destination.open("ab") as outgoing:
            incoming.seek(offset)
            if not self.interrupted:
                self.interrupted = True
                outgoing.write(incoming.read(7))
                raise rc.ReleaseChannelError("release.download.failed", "simulated interruption")
            outgoing.write(incoming.read())


class MappingTransport(rc.Transport):
    def __init__(self, sources: dict[str, pathlib.Path]):
        self.sources = sources
        self.urls: list[str] = []

    def append(self, url: str, destination: pathlib.Path, offset: int) -> None:
        self.urls.append(url)
        with self.sources[url].open("rb") as incoming, destination.open("ab") as outgoing:
            incoming.seek(offset)
            outgoing.write(incoming.read())


class ReleaseChannelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.keys = rc.TrustedKeyStore({"fixture-ed25519-2026": FIXTURES / "fixture-ed25519-public.pem"})
        self.raw_manifest = (FIXTURES / "manifest.json").read_bytes()
        self.raw_signature = (FIXTURES / "manifest.json.sig").read_bytes()
        self.manifest = rc.verify_manifest(self.raw_manifest, self.raw_signature, self.keys)

    def assertCode(self, code: str, callback) -> None:
        with self.assertRaises(rc.ReleaseChannelError) as caught:
            callback()
        self.assertEqual(code, caught.exception.code)

    def test_valid_signed_manifest_and_strict_schema(self) -> None:
        self.assertEqual("0.3.0", self.manifest["release"]["version"])
        self.assertEqual("studiocast-release-manifest/v1", self.manifest["manifest_version"])
        duplicated = b'{"schema_version":1,"schema_version":1}'
        self.assertCode("release.json.duplicate_key", lambda: rc.strict_json_loads(duplicated))
        unknown = copy.deepcopy(self.manifest)
        unknown["surprise"] = True
        self.assertCode("release.manifest.schema", lambda: rc.validate_manifest(unknown))

    def test_bad_manifest_signature_and_unknown_key_fail_closed(self) -> None:
        changed = self.raw_manifest.replace(b'"0.3.0"', b'"0.3.1"', 1)
        self.assertCode("release.signature.mismatch", lambda: rc.verify_manifest(changed, self.raw_signature, self.keys))
        envelope = json.loads(self.raw_signature)
        envelope["key_id"] = "not-trusted"
        self.assertCode(
            "release.signature.unknown_key",
            lambda: rc.verify_manifest(self.raw_manifest, rc.canonical_json(envelope), self.keys),
        )

    def test_source_and_installer_signatures_and_hashes(self) -> None:
        for key, fixture in (("source_archive", "source.tar.gz"), ("installer_appimage", "installer.AppImage")):
            metadata = self.manifest["artifacts"][key]
            path = FIXTURES / fixture
            rc.verify_artifact(path, metadata, self.keys)
            with tempfile.TemporaryDirectory() as temporary:
                bad = pathlib.Path(temporary) / fixture
                bad.write_bytes(path.read_bytes() + b"corrupt")
                self.assertCode("release.artifact.size_mismatch", lambda: rc.verify_artifact(bad, metadata, self.keys))
                same_size = bytearray(path.read_bytes())
                same_size[0] ^= 1
                bad.write_bytes(same_size)
                self.assertCode("release.artifact.hash_mismatch", lambda: rc.verify_artifact(bad, metadata, self.keys))
                wrong_signature = copy.deepcopy(metadata)
                wrong_signature["signature"]["value"] = self.manifest["artifacts"][
                    "installer_appimage" if key == "source_archive" else "source_archive"
                ]["signature"]["value"]
                self.assertCode("release.signature.mismatch", lambda: rc.verify_artifact(path, wrong_signature, self.keys))

    def test_semver_and_update_policy(self) -> None:
        cases = [
            ("1.2.3-alpha", "1.2.3-alpha.1", -1),
            ("1.2.3-alpha.1", "1.2.3-beta", -1),
            ("1.2.3-rc.1", "1.2.3", -1),
            ("1.2.3+one", "1.2.3+two", 0),
            ("2.0.0", "1.99.99", 1),
        ]
        for left, right, result in cases:
            self.assertEqual(result, rc.SemVer.parse(left).compare(rc.SemVer.parse(right)))
        self.assertEqual("update", rc.decide_update("0.2.9", "0.3.0").action)
        self.assertEqual("keep_current", rc.decide_update("0.3.0", "0.3.0").action)
        self.assertEqual("reinstall", rc.decide_update("0.3.0", "0.3.0", allow_same_version=True).action)
        self.assertEqual("keep_current", rc.decide_update("0.4.0", "0.3.0").action)
        self.assertEqual("downgrade", rc.decide_update("0.4.0", "0.3.0", allow_downgrade=True).action)
        self.assertTrue(rc.installer_meets_minimum("0.2.9", self.manifest["minimum_installer_version"]))
        self.assertFalse(rc.installer_meets_minimum("0.2.9-rc.1", self.manifest["minimum_installer_version"]))

    def test_offline_cache_hit_miss_and_corrupt(self) -> None:
        metadata = self.manifest["artifacts"]["source_archive"]
        with tempfile.TemporaryDirectory() as temporary:
            cache = pathlib.Path(temporary)
            target = cache / metadata["filename"]
            target.write_bytes((FIXTURES / "source.tar.gz").read_bytes())
            with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
                self.assertEqual(target, rc.acquire_artifact(metadata, cache, self.keys, offline=True))
            target.unlink()
            self.assertCode("release.cache.miss", lambda: rc.acquire_artifact(metadata, cache, self.keys, offline=True))
            target.write_bytes(b"bad")
            self.assertCode("release.cache.corrupt", lambda: rc.acquire_artifact(metadata, cache, self.keys, offline=True))

    def test_signed_manifest_transport_and_offline_generation_cache(self) -> None:
        manifest_url = "mock:manifest"
        signature_url = "mock:signature"
        transport = MappingTransport(
            {manifest_url: FIXTURES / "manifest.json", signature_url: FIXTURES / "manifest.json.sig"}
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache = pathlib.Path(temporary)
            loaded = rc.fetch_release_manifest(
                cache,
                self.keys,
                manifest_url=manifest_url,
                signature_url=signature_url,
                transport=transport,
            )
            self.assertEqual("0.3.0", loaded["release"]["version"])
            self.assertEqual([manifest_url, signature_url], transport.urls)
            with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
                offline = rc.fetch_release_manifest(cache, self.keys, offline=True)
            self.assertEqual(loaded, offline)
            pointer_before = (cache / "current").read_text()
            bad_signature = cache / "bad-signature"
            bad_signature.write_bytes(b"bad")
            failed_transport = MappingTransport(
                {manifest_url: FIXTURES / "manifest.json", signature_url: bad_signature}
            )
            self.assertCode(
                "release.json.invalid",
                lambda: rc.fetch_release_manifest(
                    cache,
                    self.keys,
                    manifest_url=manifest_url,
                    signature_url=signature_url,
                    transport=failed_transport,
                ),
            )
            self.assertEqual(pointer_before, (cache / "current").read_text())
            self.assertEqual(loaded, rc.fetch_release_manifest(cache, self.keys, offline=True))
            generation = (cache / "current").read_text().strip()
            (cache / generation / "manifest.json.sig").write_bytes(b"bad")
            self.assertCode(
                "release.manifest_cache.corrupt",
                lambda: rc.fetch_release_manifest(cache, self.keys, offline=True),
            )
        with tempfile.TemporaryDirectory() as empty:
            self.assertCode(
                "release.manifest_cache.miss",
                lambda: rc.fetch_release_manifest(pathlib.Path(empty), self.keys, offline=True),
            )

    def test_local_transport_resume_and_atomic_move(self) -> None:
        metadata = copy.deepcopy(self.manifest["artifacts"]["source_archive"])
        source = FIXTURES / "source.tar.gz"
        metadata["url"] = source.resolve().as_uri()
        with tempfile.TemporaryDirectory() as temporary:
            cache = pathlib.Path(temporary)
            transport = InterruptingTransport(source)
            self.assertCode(
                "release.download.failed",
                lambda: rc.acquire_artifact(metadata, cache, self.keys, transport=transport),
            )
            self.assertTrue((cache / ".source.tar.gz.part").exists())
            result = rc.acquire_artifact(metadata, cache, self.keys, transport=transport)
            self.assertEqual([0, 7], transport.offsets)
            rc.verify_artifact(result, metadata, self.keys)
            self.assertFalse((cache / ".source.tar.gz.part").exists())

    def test_failed_refresh_preserves_verified_existing_file(self) -> None:
        metadata = copy.deepcopy(self.manifest["artifacts"]["source_archive"])
        source = FIXTURES / "source.tar.gz"
        with tempfile.TemporaryDirectory() as temporary:
            cache = pathlib.Path(temporary)
            target = cache / metadata["filename"]
            target.write_bytes(source.read_bytes())
            bad_source = cache / "bad-source"
            bad_source.write_bytes(b"x" * metadata["size_bytes"])
            transport = RecordingTransport(bad_source)
            self.assertCode(
                "release.artifact.hash_mismatch",
                lambda: rc.acquire_artifact(metadata, cache, self.keys, transport=transport, force_refresh=True),
            )
            self.assertEqual(source.read_bytes(), target.read_bytes())
            rc.verify_artifact(target, metadata, self.keys)

    def test_rollback_metadata_contract(self) -> None:
        metadata = rc.rollback_metadata("0.3.0", "0.2.9", pathlib.Path("/data/payloads"))
        self.assertEqual("side_by_side_atomic_pointer", metadata["activation_mode"])
        self.assertEqual("/data/payloads/0.2.9", metadata["previous_payload"])
        self.assertTrue(metadata["rollback_available"])

    def test_self_update_offer_manual_fallback_and_mock_relaunch(self) -> None:
        app_metadata = copy.deepcopy(self.manifest["artifacts"]["installer_appimage"])
        app_metadata["url"] = (FIXTURES / "installer.AppImage").resolve().as_uri()
        manifest = copy.deepcopy(self.manifest)
        manifest["artifacts"]["installer_appimage"] = app_metadata
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            running = root / "running.AppImage"
            running.write_bytes(b"old running image")
            offer = rc.prepare_self_update(
                manifest,
                "0.2.9",
                root / "cache",
                self.keys,
                appimage_path=str(running),
                argv=("--resume-update",),
                transport=RecordingTransport(FIXTURES / "installer.AppImage"),
            )
            self.assertEqual("offer_restart", offer.state)
            self.assertEqual(b"old running image", running.read_bytes())
            calls: list[tuple[str, ...]] = []
            launcher = lambda command: calls.append(tuple(command))
            self.assertFalse(rc.relaunch_self_update(offer, user_confirmed=False, launcher=launcher))
            self.assertEqual([], calls)
            self.assertTrue(rc.relaunch_self_update(offer, user_confirmed=True, launcher=launcher))
            self.assertEqual([offer.relaunch_command], calls)
            manual = rc.prepare_self_update(manifest, "0.2.9", root / "other", self.keys, appimage_path=None)
            self.assertEqual("manual_download", manual.state)
            self.assertEqual(app_metadata["url"], manual.manual_download_url)
            bad_manifest = copy.deepcopy(manifest)
            bad_manifest["artifacts"]["installer_appimage"]["sha256"] = "0" * 64
            bad = rc.prepare_self_update(
                bad_manifest,
                "0.2.9",
                root / "bad-cache",
                self.keys,
                appimage_path=str(running),
                transport=RecordingTransport(FIXTURES / "installer.AppImage"),
            )
            self.assertEqual("manual_download", bad.state)
            self.assertEqual("release.artifact.hash_mismatch", bad.reason_code)
            same = rc.prepare_self_update(manifest, "0.3.0", root / "same", self.keys, appimage_path=str(running))
            self.assertEqual("not_needed", same.state)

    def test_release_tool_generates_verifiable_manifest_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            private = root / "private.pem"
            public = root / "public.pem"
            subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)], check=True)
            subprocess.run(["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)], check=True)
            source = root / "StudioCast-1.0.0-source.tar.gz"
            appimage = root / "StudioCast-Installer-1.0.0-x86_64.AppImage"
            source.write_bytes(b"source")
            appimage.write_bytes(b"appimage")
            output, signature = root / "manifest.json", root / "manifest.json.sig"
            subprocess.run(
                [
                    sys.executable, str(ROOT / "packaging/release/release_tool.py"),
                    "--version", "1.0.0", "--minimum-installer-version", "0.2.9",
                    "--source-archive", str(source), "--installer-appimage", str(appimage),
                    "--base-url", "https://github.com/mexicantexan/studiocast/releases/download/v1.0.0",
                    "--release-page-url", "https://github.com/mexicantexan/studiocast/releases/tag/v1.0.0",
                    "--commit", "a" * 40, "--workflow-run-url", "https://github.com/mexicantexan/studiocast/actions/runs/1",
                    "--private-key", str(private), "--key-id", "ephemeral-test", "--generated-at", "2026-07-12T00:00:00Z",
                    "--output", str(output), "--signature-output", str(signature),
                ],
                check=True,
            )
            generated = rc.verify_manifest(output.read_bytes(), signature.read_bytes(), rc.TrustedKeyStore({"ephemeral-test": public}))
            rc.verify_artifact(source, generated["artifacts"]["source_archive"], rc.TrustedKeyStore({"ephemeral-test": public}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
