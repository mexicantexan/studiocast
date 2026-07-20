#!/usr/bin/python3
"""Hermetic tests for curated model download/install transactions."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "installer/models/model_transactions.py"
REAL_CATALOG_PATH = REPO_ROOT / "packaging/models/curated-model-catalog-v1.json"


def load_module():
    spec = importlib.util.spec_from_file_location("studiocast_model_transactions", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load model transaction module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


model_tx = load_module()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Fixture:
    def __init__(self, root: Path, pack_count: int = 1, source_pack_count: int | None = None):
        self.root = root
        self.repo = root / "repo"
        self.source = root / "source"
        self.destination = root / "models"
        self.cache = root / "cache"
        self.catalog_path = self.repo / "catalog.json"
        self.repo.mkdir(parents=True)
        self.source.mkdir()
        if source_pack_count is None:
            source_pack_count = pack_count
        packs = []
        self.artifacts = {}
        for index in range(pack_count):
            pack_id = f"fixture_pack_{index}"
            artifact = (f"verified fixture artifact {index}\n" * 4).encode()
            model = json.dumps(
                {
                    "id": pack_id,
                    "display_name": f"Fixture {index}",
                    "onnx_filename": "model.onnx",
                },
                sort_keys=True,
            ).encode() + b"\n"
            license_text = f"Fixture license {index}\n".encode()
            metadata_dir = self.repo / "metadata" / pack_id
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "model.json").write_bytes(model)
            (metadata_dir / "LICENSE.txt").write_bytes(license_text)
            source_relative = f"objects/{pack_id}/model.onnx"
            if index < source_pack_count:
                source_file = self.source / source_relative
                source_file.parent.mkdir(parents=True)
                source_file.write_bytes(artifact)
            packs.append(
                {
                    "id": pack_id,
                    "family": "open_audio",
                    "task": "audio_enhancement",
                    "default": True,
                    "pack_path": pack_id,
                    "metadata": [
                        {
                            "source": f"metadata/{pack_id}/model.json",
                            "destination": "model.json",
                            "sha256": digest(model),
                        },
                        {
                            "source": f"metadata/{pack_id}/LICENSE.txt",
                            "destination": "LICENSE.txt",
                            "sha256": digest(license_text),
                        },
                    ],
                    "license": {
                        "spdx": "MIT",
                        "redistribution": "permitted",
                        "notes": "fixture",
                    },
                    "provenance": {
                        "project": "fixture",
                        "url": "https://example.invalid/fixture",
                        "release": "test",
                    },
                    "artifacts": [
                        {
                            "name": "model.onnx",
                            "sha256": digest(artifact),
                            "size_bytes": len(artifact),
                            "size_status": "known",
                            "source_paths": [source_relative],
                        }
                    ],
                }
            )
            self.artifacts[pack_id] = artifact
        catalog = {
            "schema_version": 1,
            "catalog_version": "fixture-v1",
            "policy_version": 1,
            "default_source": str(self.source),
            "size_metadata": {"trusted": True, "reason_code": "signed_fixture"},
            "packs": packs,
        }
        self.catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
        self.catalog = model_tx.load_catalog(self.catalog_path)

    def installer(
        self,
        *,
        transport=None,
        offline: bool = False,
        retries: int = 0,
        cache: Path | None = None,
    ):
        if transport is None and not offline:
            transport = model_tx.FileTransport(self.source)
        return model_tx.ModelTransactionInstaller(
            self.catalog,
            self.repo,
            self.destination,
            cache or self.cache,
            transport,
            offline=offline,
            retries=retries,
        )


class TempTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="studiocast-model-transaction-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()


class CatalogContractTests(TempTest):
    def test_real_catalog_has_exact_seven_defaults_and_eight_artifacts(self):
        catalog = model_tx.load_catalog(REAL_CATALOG_PATH)
        ids = [pack["id"] for pack in catalog["packs"] if pack["default"]]
        self.assertEqual(
            set(ids),
            {
                "fastenhancer_s_vd_v1",
                "fastenhancer_m_vd_v1",
                "modnet-webnn-256-fp32",
                "yunet_opencv_zoo_2023mar_fp32",
                "dlib_68_ibug_300w",
                "gaze_correction_cam_flx_v0_1_1",
                "fastdvdnet_sigma15",
            },
        )
        self.assertEqual(len(ids), 7)
        self.assertEqual(sum(len(pack["artifacts"]) for pack in catalog["packs"]), 8)
        gaze = next(pack for pack in catalog["packs"] if pack["id"] == "gaze_correction_cam_flx_v0_1_1")
        self.assertEqual({item["name"] for item in gaze["artifacts"]}, {"gaze_flx_left.onnx", "gaze_flx_right.onnx"})
        expected = {
            ("fastenhancer_s_vd_v1", "model.onnx"): (832775, "e2d0e91bbfab4af1316bb2c41126a38c8b3cd015b93bb630d651af8fdbf7f2e8"),
            ("fastenhancer_m_vd_v1", "model.onnx"): (2033628, "367059e724dd367c056dc906e9698ec5864c80c9a88e0597f7a2b0f81c506aaa"),
            ("modnet-webnn-256-fp32", "model.onnx"): (25888640, "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9"),
            ("yunet_opencv_zoo_2023mar_fp32", "model.onnx"): (232589, "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
            ("dlib_68_ibug_300w", "shape_predictor_68_face_landmarks.dat"): (99693937, "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f"),
            ("gaze_correction_cam_flx_v0_1_1", "gaze_flx_left.onnx"): (1062936, "1a5129972dbc2ace9170d79ad4986e8c5130a79ac2180c01c826ef242d8e3d00"),
            ("gaze_correction_cam_flx_v0_1_1", "gaze_flx_right.onnx"): (1062994, "736b7c6b91d1475d45083b00325d54d380ae1c51767e1f7ef69a3d14083cfcf1"),
            ("fastdvdnet_sigma15", "model.onnx"): (416788, "500517e2706130ab84d140add26353ded2afb1317b6ab747c6417fa7134963dc"),
        }
        actual = {(pack["id"], item["name"]): (item["size_bytes"], item["sha256"])
                  for pack in catalog["packs"] for item in pack["artifacts"]}
        self.assertEqual(actual, expected)
        self.assertEqual(sum(size for size, _ in actual.values()), 131224287)
        self.assertEqual(catalog["catalog_version"], "2026-07-19")
        self.assertEqual(catalog["size_metadata"], {
            "trusted": True,
            "reason_code": "artifact_sizes_independently_verified_against_pinned_sha256",
        })
        self.assertTrue(all(item["size_status"] == "known" for pack in catalog["packs"]
                            for item in pack["artifacts"]))
        self.assertNotIn("maxine", json.dumps(catalog).lower())

    def test_catalog_metadata_and_license_hashes_match_trusted_sources(self):
        catalog = model_tx.load_catalog(REAL_CATALOG_PATH)
        for pack in catalog["packs"]:
            destinations = set()
            for item in pack["metadata"]:
                source = REPO_ROOT / item["source"]
                self.assertEqual(model_tx.sha256_path(source), item["sha256"])
                destinations.add(item["destination"])
            self.assertEqual(destinations, {"model.json", "LICENSE.txt"})
            self.assertTrue(pack["license"]["spdx"])
            self.assertTrue(pack["provenance"]["url"])

    def test_recommended_untrusted_known_sizes_block_before_filesystem_mutation(self):
        fixture = Fixture(self.root)
        raw = json.loads(fixture.catalog_path.read_text())
        raw["size_metadata"] = {"trusted": False, "reason_code": "fixture_untrusted"}
        fixture.catalog_path.write_text(json.dumps(raw))
        catalog = model_tx.load_catalog(fixture.catalog_path)
        destination = self.root / "never-created-destination"
        cache = self.root / "never-created-cache"
        installer = model_tx.ModelTransactionInstaller(
            catalog, fixture.repo, destination, cache, None, offline=False
        )
        result = installer.install("open_audio", recommended=True)
        self.assertEqual((result["state"], result["reason_code"]), ("blocked", "artifact_sizes_untrusted"))
        self.assertEqual(result["blockers"], [])
        self.assertFalse(destination.exists())
        self.assertFalse(cache.exists())

    def test_size_validation_matrix_and_closed_schema_fail_closed(self):
        fixture = Fixture(self.root, pack_count=2)
        base = json.loads(fixture.catalog_path.read_text())
        cases = {}
        missing = json.loads(json.dumps(base))
        del missing["packs"][0]["artifacts"][0]["size_bytes"]
        cases["missing"] = missing
        for label, invalid in (("null", None), ("zero", 0), ("negative", -1),
                               ("bool", True), ("float", 1.5), ("string", "123")):
            value = json.loads(json.dumps(base))
            value["packs"][0]["artifacts"][0]["size_bytes"] = invalid
            if invalid is None:
                value["packs"][0]["artifacts"][0]["size_status"] = "unknown"
            cases[label] = value
        status = json.loads(json.dumps(base))
        status["packs"][0]["artifacts"][0]["size_status"] = "unknown"
        cases["status_mismatch"] = status
        partial = json.loads(json.dumps(base))
        partial["packs"][0]["artifacts"][0].update(size_bytes=None, size_status="unknown")
        cases["partial_trusted"] = partial
        empty_reason = json.loads(json.dumps(base))
        empty_reason["size_metadata"]["reason_code"] = ""
        cases["empty_reason"] = empty_reason
        traversal = json.loads(json.dumps(base))
        traversal["packs"][0]["pack_path"] = "../outside"
        cases["traversal"] = traversal
        unknown = json.loads(json.dumps(base))
        unknown["packs"][0]["command"] = "touch sentinel"
        cases["unknown_field"] = unknown
        duplicate = json.loads(json.dumps(base))
        duplicate["packs"].append(json.loads(json.dumps(duplicate["packs"][0])))
        cases["duplicate"] = duplicate
        for label, value in cases.items():
            path = self.root / f"bad-{label}.json"
            path.write_text(json.dumps(value))
            with self.subTest(label=label), self.assertRaises(model_tx.CatalogError):
                model_tx.load_catalog(path)
        truncated = self.root / "bad-truncated.json"
        truncated.write_text('{"schema_version":1')
        with self.assertRaises(model_tx.CatalogError):
            model_tx.load_catalog(truncated)


class InstallTransactionTests(TempTest):
    def test_local_install_stages_metadata_license_and_artifact_atomically(self):
        fixture = Fixture(self.root)
        custom = fixture.destination / "my_custom_pack"
        custom.mkdir(parents=True)
        (custom / "keep.txt").write_text("keep")
        result = fixture.installer().install("open_audio", recommended=True)
        self.assertEqual(result["state"], "succeeded")
        pack = fixture.destination / "fixture_pack_0"
        self.assertEqual((pack / "model.onnx").read_bytes(), fixture.artifacts["fixture_pack_0"])
        self.assertTrue((pack / "model.json").is_file())
        self.assertTrue((pack / "LICENSE.txt").is_file())
        marker = json.loads((pack / model_tx.MARKER_NAME).read_text())
        self.assertEqual(marker["pack_id"], "fixture_pack_0")
        self.assertEqual((custom / "keep.txt").read_text(), "keep")
        self.assertFalse(any(path.name.startswith(".studiocast-stage-") for path in fixture.destination.iterdir()))
        second = fixture.installer().install("open_audio")
        self.assertEqual(second["packs"][0]["status"], "unchanged")

    def test_checksum_mismatch_preserves_good_existing_pack(self):
        fixture = Fixture(self.root)
        first = fixture.installer().install("open_audio")
        self.assertEqual(first["state"], "succeeded")
        pack = fixture.destination / "fixture_pack_0"
        sentinel = pack / "preserved.txt"
        sentinel.write_text("still here")
        before = (pack / "model.onnx").read_bytes()
        source_file = fixture.source / "objects/fixture_pack_0/model.onnx"
        source_file.write_bytes(b"bad replacement")
        fresh_cache = self.root / "fresh-cache"
        result = fixture.installer(cache=fresh_cache).install("open_audio", force=True)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["packs"][0]["reason_code"], "checksum_mismatch")
        self.assertEqual((pack / "model.onnx").read_bytes(), before)
        self.assertEqual(sentinel.read_text(), "still here")

    def test_optional_pack_failure_is_structured_degraded(self):
        fixture = Fixture(self.root, pack_count=2, source_pack_count=1)
        result = fixture.installer().install("open_audio")
        self.assertEqual((result["state"], result["reason_code"]), ("degraded", "optional_model_failure"))
        self.assertEqual([pack["status"] for pack in result["packs"]], ["installed", "failed"])
        self.assertTrue((fixture.destination / "fixture_pack_0/model.onnx").exists())
        self.assertFalse((fixture.destination / "fixture_pack_1").exists())

    def test_custom_or_unverified_catalog_destination_is_never_replaced(self):
        fixture = Fixture(self.root)
        custom_destination = fixture.destination / "fixture_pack_0"
        custom_destination.mkdir(parents=True)
        (custom_destination / "model.json").write_text('{"id":"custom"}\n')
        (custom_destination / "LICENSE.txt").write_text("custom license\n")
        (custom_destination / "model.onnx").write_text("custom model\n")
        result = fixture.installer().install("open_audio")
        self.assertEqual(result["packs"][0]["reason_code"], "curated_ownership_unverified")
        self.assertEqual((custom_destination / "model.onnx").read_text(), "custom model\n")

    def test_exact_legacy_curated_metadata_can_migrate_without_touching_custom_sibling(self):
        fixture = Fixture(self.root)
        pack_definition = fixture.catalog["packs"][0]
        legacy = fixture.destination / "fixture_pack_0"
        legacy.mkdir(parents=True)
        for item in pack_definition["metadata"]:
            shutil_source = fixture.repo / item["source"]
            (legacy / item["destination"]).write_bytes(shutil_source.read_bytes())
        (legacy / "model.onnx").write_bytes(b"corrupt old artifact")
        custom = fixture.destination / "custom"
        custom.mkdir()
        (custom / "keep").write_text("yes")
        result = fixture.installer().install("open_audio")
        self.assertEqual(result["state"], "succeeded")
        self.assertTrue((legacy / model_tx.MARKER_NAME).exists())
        self.assertEqual((custom / "keep").read_text(), "yes")


class ResumeCacheRetryTests(TempTest):
    def test_interrupted_partial_resumes_and_retries(self):
        fixture = Fixture(self.root)
        artifact = fixture.artifacts["fixture_pack_0"]

        class InterruptOnce:
            def __init__(self):
                self.offsets = []

            def fetch(self, relative_path, part_path, offset):
                self.offsets.append(offset)
                if len(self.offsets) == 1:
                    half = len(artifact) // 2
                    part_path.write_bytes(artifact[:half])
                    raise model_tx.TransactionError("source_unavailable", "simulated interruption")
                with part_path.open("ab") as handle:
                    handle.write(artifact[offset:])
                return len(artifact) - offset

        transport = InterruptOnce()
        result = fixture.installer(transport=transport, retries=1).install("open_audio")
        self.assertEqual(result["state"], "succeeded")
        artifact_result = result["packs"][0]["artifacts"][0]
        self.assertEqual(transport.offsets, [0, len(artifact) // 2])
        self.assertTrue(artifact_result["resumed"])
        self.assertEqual(artifact_result["attempts"], 2)
        self.assertFalse(any(path.suffix == ".part" for path in (fixture.cache / "sha256").iterdir()))

    def test_offline_verified_cache_installs_and_missing_cache_fails(self):
        fixture = Fixture(self.root)
        pack = fixture.catalog["packs"][0]
        artifact = pack["artifacts"][0]
        cache_file = fixture.cache / "sha256" / artifact["sha256"]
        cache_file.parent.mkdir(parents=True)
        cache_file.write_bytes(fixture.artifacts[pack["id"]])
        result = fixture.installer(offline=True).install("open_audio")
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(result["packs"][0]["artifacts"][0]["status"], "cache_verified")

        missing_fixture = Fixture(self.root / "missing")
        missing = missing_fixture.installer(offline=True).install("open_audio")
        self.assertEqual(missing["state"], "failed")
        self.assertEqual(missing["packs"][0]["reason_code"], "offline_cache_missing")

    def test_good_cache_is_preserved_when_source_is_broken(self):
        fixture = Fixture(self.root)
        pack = fixture.catalog["packs"][0]
        artifact = pack["artifacts"][0]
        cache_file = fixture.cache / "sha256" / artifact["sha256"]
        cache_file.parent.mkdir(parents=True)
        good = fixture.artifacts[pack["id"]]
        cache_file.write_bytes(good)
        (fixture.source / artifact["source_paths"][0]).write_bytes(b"bad source")
        result = fixture.installer().install("open_audio", force=True)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(cache_file.read_bytes(), good)
        self.assertEqual(result["packs"][0]["reason_code"], "checksum_mismatch")


class RemovalAndCliTests(TempTest):
    def test_remove_curated_removes_only_marked_catalog_pack(self):
        fixture = Fixture(self.root)
        fixture.installer().install("open_audio")
        custom = fixture.destination / "custom_pack"
        custom.mkdir()
        (custom / "model.onnx").write_text("custom")
        result = fixture.installer(offline=True).remove_curated("open_audio")
        self.assertEqual(result["state"], "succeeded")
        self.assertFalse((fixture.destination / "fixture_pack_0").exists())
        self.assertEqual((custom / "model.onnx").read_text(), "custom")
        shutil_cache = fixture.cache
        if shutil_cache.exists():
            shutil.rmtree(shutil_cache)
        reinstalled = fixture.installer().install("open_audio")
        self.assertEqual(reinstalled["state"], "succeeded")
        self.assertTrue((fixture.destination / "fixture_pack_0/model.onnx").exists())
        self.assertEqual((custom / "model.onnx").read_text(), "custom")

    def test_remove_preserves_unmarked_or_symlinked_catalog_destination(self):
        fixture = Fixture(self.root)
        destination = fixture.destination / "fixture_pack_0"
        destination.mkdir(parents=True)
        (destination / "custom").write_text("keep")
        result = fixture.installer(offline=True).remove_curated("open_audio")
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["packs"][0]["reason_code"], "curated_ownership_unverified")
        self.assertEqual((destination / "custom").read_text(), "keep")

    def test_cli_local_transport_and_shell_lists_are_hermetic(self):
        fixture = Fixture(self.root)
        command = [
            sys.executable,
            "-I",
            str(MODULE_PATH),
            "--catalog",
            str(fixture.catalog_path),
            "--repo-root",
            str(fixture.repo),
            "install",
            "--family",
            "open_audio",
            "--dest",
            str(fixture.destination),
            "--cache",
            str(fixture.cache),
            "--source",
            str(fixture.source),
            "--json",
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["state"], "succeeded")

        audio = subprocess.run(
            [str(REPO_ROOT / "scripts/install/open_audio_models.sh"), "--list"],
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
        video = subprocess.run(
            [str(REPO_ROOT / "scripts/install/open_video_models.sh"), "--list"],
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual((audio.returncode, video.returncode), (0, 0))
        self.assertIn("fastenhancer_s_vd_v1", audio.stdout)
        self.assertIn("fastenhancer_m_vd_v1", audio.stdout)
        self.assertNotIn("fastenhancer_l_vd_v1", audio.stdout)
        for model_id in (
            "modnet-webnn-256-fp32",
            "yunet_opencv_zoo_2023mar_fp32",
            "dlib_68_ibug_300w",
            "gaze_correction_cam_flx_v0_1_1",
            "fastdvdnet_sigma15",
        ):
            self.assertIn(model_id, video.stdout)
        self.assertNotIn("placeholder", video.stdout.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
