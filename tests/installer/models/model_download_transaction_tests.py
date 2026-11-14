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


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "installer/models/model_transactions.py"
REAL_CATALOG_PATH = REPO_ROOT / "installer/models/curated-model-catalog-v1.json"


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
    def test_real_catalog_has_exact_seven_defaults_and_eighteen_artifacts(self):
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
        self.assertEqual(sum(len(pack["artifacts"]) for pack in catalog["packs"]), 18)
        gaze = next(pack for pack in catalog["packs"] if pack["id"] == "gaze_correction_cam_flx_v0_1_1")
        self.assertEqual({item["name"] for item in gaze["artifacts"]}, {"gaze_flx_left.onnx", "gaze_flx_right.onnx", "gaze_flx_left.ncnn.param", "gaze_flx_left.ncnn.bin", "gaze_flx_right.ncnn.param", "gaze_flx_right.ncnn.bin"})
        expected = {
            ("fastenhancer_s_vd_v1", "model.onnx"): (832775, "e2d0e91bbfab4af1316bb2c41126a38c8b3cd015b93bb630d651af8fdbf7f2e8"),
            ("fastenhancer_m_vd_v1", "model.onnx"): (2033628, "367059e724dd367c056dc906e9698ec5864c80c9a88e0597f7a2b0f81c506aaa"),
            ("modnet-webnn-256-fp32", "model.onnx"): (25888640, "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9"),
            ("modnet-webnn-256-fp32", "model.ncnn.param"): (12113, "4e205032860a9359fa2555c6ef66dece2a8dcd034a1a61466c58d1310af8a1bd"),
            ("modnet-webnn-256-fp32", "model.ncnn.bin"): (25844000, "38871660dda4dd324304eb572f43a43ddc5c876c8dff50db2cf09d29a17e3fa2"),
            ("yunet_opencv_zoo_2023mar_fp32", "model.onnx"): (232589, "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
            ("dlib_68_ibug_300w", "shape_predictor_68_face_landmarks.dat"): (99693937, "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f"),
            ("gaze_correction_cam_flx_v0_1_1", "gaze_flx_left.onnx"): (1062936, "1a5129972dbc2ace9170d79ad4986e8c5130a79ac2180c01c826ef242d8e3d00"),
            ("gaze_correction_cam_flx_v0_1_1", "gaze_flx_right.onnx"): (1062994, "736b7c6b91d1475d45083b00325d54d380ae1c51767e1f7ef69a3d14083cfcf1"),
            ("fastdvdnet_sigma15", "model.onnx"): (9948304, "0895dcad878f44ff4b610e0569bf701c8e23566e675c22a9bb0d59b95dcfc7ac"),
            ('yunet_opencv_zoo_2023mar_fp32', 'model.ncnn.param'): (6423, 'a79870770c2e87ce218950ba471b4fd82f2ec3c9c5908405b05e4e93fa3e8293'),
            ('yunet_opencv_zoo_2023mar_fp32', 'model.ncnn.bin'): (212628, '8faa696d61ad6bf5c13ed5d6c8ae35e376c660968db53d723a329020856ca5c9'),
            ('gaze_correction_cam_flx_v0_1_1', 'gaze_flx_left.ncnn.param'): (2726, '7990a15c75c916c04dcc3cab488f4267bd5b0dd6a1d4095d86e1e0bf69341ee9'),
            ('gaze_correction_cam_flx_v0_1_1', 'gaze_flx_left.ncnn.bin'): (1000192, '7597eb7e151ec1de71fb33ce7f511b856f258d538f939727401598e18d180416'),
            ('gaze_correction_cam_flx_v0_1_1', 'gaze_flx_right.ncnn.param'): (2726, '7990a15c75c916c04dcc3cab488f4267bd5b0dd6a1d4095d86e1e0bf69341ee9'),
            ('gaze_correction_cam_flx_v0_1_1', 'gaze_flx_right.ncnn.bin'): (1000192, '9333359e7c4924312b4af161ae0678471abf7d301c7893474e534c4f9541802b'),
            ('fastdvdnet_sigma15', 'model.ncnn.param'): (7383, 'f3bb19a906f5c811f9627e167181afa04a8adc76fa2e77e8f1ab00d27535c102'),
            ('fastdvdnet_sigma15', 'model.ncnn.bin'): (19815200, 'b31f624553b8072fe53fc2f46a060495b8016ad2f0c6ee9d0f66f88604fdec3f'),
        }
        actual = {(pack["id"], item["name"]): (item["size_bytes"], item["sha256"])
                  for pack in catalog["packs"] for item in pack["artifacts"]}
        self.assertEqual(actual, expected)
        self.assertEqual(sum(size for size, _ in actual.values()), 188659386)
        self.assertEqual(catalog["catalog_version"], "2026-11-14")
        self.assertEqual(catalog["size_metadata"], {
            "trusted": True,
            "reason_code": "artifact_sizes_independently_verified_against_pinned_sha256",
        })
        self.assertTrue(all(item["size_status"] == "known" for pack in catalog["packs"]
                            for item in pack["artifacts"]))
        self.assertNotIn("maxine", json.dumps(catalog).lower())

    def test_added_vulkan_graphs_bind_original_onnx_and_content_named_sources(self):
        catalog = model_tx.load_catalog(REAL_CATALOG_PATH)
        publication = "4713473184bf7ece0be6c097ff75f6c6c528c8ec"
        self.assertEqual(catalog["default_source"],
                         "https://huggingface.co/10dallasj/studiocast/resolve/" + publication)
        expected = {
            "fastdvdnet_sigma15": ("fastdvdnet_rgb_v1", {"main"},
                "31dee74f844200c8ca8f22a6416e02a682b3f04058b4ede17dc29c40b47bbff0"),
            "yunet_opencv_zoo_2023mar_fp32": ("yunet_640_v1", {"main"},
                "c078387a8a5f3cc83543e7141e945ec03f05fd54dc14406eb8c8c5ce5514e321"),
            "gaze_correction_cam_flx_v0_1_1": ("flx_prefix_v1", {"left", "right"},
                "918075a9a3502a38c6df94f27f1ec4a765c09bd226de1229e8e5bd57d853bf20"),
        }
        for pack in catalog["packs"]:
            if pack["id"] not in expected:
                continue
            profile, roles, revision = expected[pack["id"]]
            self.assertIn("Hugging Face publication " + publication,
                          pack["provenance"]["release"])
            manifest = json.loads((REPO_ROOT / pack["metadata"][0]["source"]).read_text())
            files = {item["name"]: item for item in manifest["files"]}
            artifacts = {item["name"]: item for item in pack["artifacts"]}
            networks = manifest["ncnn_vulkan_networks"]
            self.assertEqual({network["role"] for network in networks}, roles)
            self.assertEqual(len(networks), len(roles))
            for network in networks:
                self.assertEqual(network["profile"], profile)
                self.assertEqual(network["precision"], "fp32")
                self.assertEqual(network["converter"], {"name": "pnnx", "version": "20261020"})
                source, = [item for item in files.values()
                           if item["kind"] == "onnx" and item["role"] == network["role"]]
                self.assertEqual(network["source_onnx_sha256"], source["sha256"])
                for field, kind in (("param_file", "ncnn_param"), ("bin_file", "ncnn_bin")):
                    name = network[field]
                    self.assertEqual(files[name]["kind"], kind)
                    self.assertEqual(files[name]["role"], "vulkan_" + network["role"])
                    self.assertEqual(files[name]["sha256"], artifacts[name]["sha256"])
                    self.assertEqual(artifacts[name]["source_paths"], [
                        "open_video/" + pack["pack_path"] + "/revisions/ncnn-video-fp32-20261020-"
                        + revision + "/" + name])

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

    def test_curated_modnet_adds_only_published_ncnn_artifacts_and_exact_metadata(self):
        catalog = model_tx.load_catalog(REAL_CATALOG_PATH)
        pack = next(item for item in catalog["packs"] if item["id"] == "modnet-webnn-256-fp32")
        self.assertEqual(pack["pack_path"], "matting/Good Quality")
        self.assertTrue(pack["default"])
        onnx, param, weights = pack["artifacts"]
        self.assertEqual(onnx, {
            "name": "model.onnx",
            "sha256": "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9",
            "size_bytes": 25888640, "size_status": "known",
            "source_paths": ["open_video/matting/Good Quality/model.onnx",
                             "matting/Good Quality/model.onnx",
                             "open_cuda/matting/Good Quality/model.onnx"],
        })
        prefix = ("open_video/matting/Good Quality/revisions/ncnn-fp32-20261020-"
                  "38871660dda4dd324304eb572f43a43ddc5c876c8dff50db2cf09d29a17e3fa2/")
        for artifact in (param, weights):
            self.assertEqual(artifact["source_paths"], [prefix + artifact["name"]])
        self.assertIn("Hugging Face publication f6be359c49ba07163287fe3732dc4a3024b91f60",
                      pack["provenance"]["release"])
        expected_metadata = {
            "model.json": "3b94bc7bac4ae116659bf7b6bac93fe3a4d90448553661de536eb6c428f13bcc",
            "LICENSE.txt": "77b9ad63b2659c31905d1a820403fea0ba0729c7d0530b6e6c6f8b717fd40f81",
        }
        self.assertEqual({item["destination"]: item["sha256"] for item in pack["metadata"]},
                         expected_metadata)
        manifest = json.loads((REPO_ROOT / pack["metadata"][0]["source"]).read_text())
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["id"], pack["id"])
        self.assertEqual(manifest["onnx_filename"], onnx["name"])
        self.assertEqual(manifest["files"], [
            {"name": artifact["name"], "kind": kind, "role": role, "sha256": artifact["sha256"]}
            for artifact, kind, role in ((onnx, "onnx", "main"),
                                         (param, "ncnn_param", "vulkan_matting"),
                                         (weights, "ncnn_bin", "vulkan_matting"))
        ])
        self.assertEqual(manifest["ncnn_vulkan"], {
            "param_file": param["name"], "bin_file": weights["name"],
            "input_blob": "input", "output_blob": "output", "precision": "fp32",
            "converter": {"name": "pnnx", "version": "20261020"},
        })

    def test_curated_fastdvdnet_repair_uses_only_versioned_verified_artifact(self):
        catalog = model_tx.load_catalog(REAL_CATALOG_PATH)
        pack = next(item for item in catalog["packs"] if item["id"] == "fastdvdnet_sigma15")
        artifact = pack["artifacts"][0]
        expected_hash = "0895dcad878f44ff4b610e0569bf701c8e23566e675c22a9bb0d59b95dcfc7ac"
        self.assertEqual(artifact["source_paths"], [
            "open_video/video_denoise/Good Quality (FastDVDnet sigma15)/revisions/"
            + expected_hash + "/model.onnx",
        ])
        self.assertIn("Hugging Face publication 8ffe968884e31ded74f857a2d0dca19f571ba044",
                      pack["provenance"]["release"])
        metadata = json.loads((REPO_ROOT / pack["metadata"][0]["source"]).read_text())
        self.assertEqual(metadata["id"], "fastdvdnet_sigma15")
        self.assertEqual([item for item in metadata["files"] if item["kind"] == "onnx"], [{"name": "model.onnx", "kind": "onnx",
                                             "role": "main", "sha256": expected_hash}])
        self.assertEqual(metadata["origin"]["upstream_commit"],
                         "c8fdf6182a0340e89dd18f5df25b47337cbede6f")
        self.assertEqual(metadata["origin"]["weights_sha256"],
                         "9d9d8413c33e3d9d961d07c530237befa1197610b9d60602ff42fd77975d2a17")
        self.assertEqual([(item["name"], item["channels"], item["dtype"], item["dynamic_hw"])
                          for item in metadata["onnx"]["inputs"]],
                         [("noisy", 15, "float32", True), ("noise_map", 1, "float32", True)])
        license_text = (REPO_ROOT / pack["metadata"][1]["source"]).read_text()
        self.assertIn("Copyright 2024 Matias Tassano", license_text)
        self.assertIn("THE SOFTWARE IS PROVIDED", license_text)

    def test_curated_yunet_fp32_artifact_has_verified_tensor_contract(self):
        catalog = model_tx.load_catalog(REAL_CATALOG_PATH)
        pack = next(
            item for item in catalog["packs"]
            if item["id"] == "yunet_opencv_zoo_2023mar_fp32"
        )
        self.assertEqual(pack["provenance"]["project"], "OpenCV Zoo YuNet")
        self.assertEqual(pack["provenance"]["release"],
                         "2023mar FP32; FP32 ncnn conversion via pnnx 20261020; "
                         "Hugging Face publication 4713473184bf7ece0be6c097ff75f6c6c528c8ec")
        self.assertEqual(
            pack["artifacts"][:1],
            [{
                "name": "model.onnx",
                "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
                "size_bytes": 232589,
                "size_status": "known",
                "source_paths": [
                    "open_video/face_detection/Best Quality (YuNet FP32)/model.onnx",
                    "face_detection/Best Quality (YuNet FP32)/model.onnx",
                    "open_cuda/face_detection/Best Quality (YuNet FP32)/model.onnx",
                ],
            }],
        )

        model_metadata = next(
            item for item in pack["metadata"] if item["destination"] == "model.json"
        )
        metadata_path = REPO_ROOT / model_metadata["source"]
        self.assertEqual(model_tx.sha256_path(metadata_path), model_metadata["sha256"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["id"], pack["id"])
        self.assertEqual(metadata["files"][0]["sha256"], pack["artifacts"][0]["sha256"])
        self.assertEqual(
            metadata["onnx"]["inputs"],
            [{
                "name": "input", "layout": "nchw", "dtype": "float32",
                "width": 640, "height": 640, "channels": 3,
            }],
        )
        expected_heads = {
            "cls_8": "scores", "cls_16": "scores", "cls_32": "scores",
            "obj_8": "objectness", "obj_16": "objectness", "obj_32": "objectness",
            "bbox_8": "boxes", "bbox_16": "boxes", "bbox_32": "boxes",
            "kps_8": "keypoints", "kps_16": "keypoints", "kps_32": "keypoints",
        }
        self.assertEqual(
            {head["name"]: head["kind"] for head in metadata["onnx"]["outputs"]},
            expected_heads,
        )
        self.assertTrue(all(head["dtype"] == "float32" for head in metadata["onnx"]["outputs"]))
        self.assertEqual({stride: (640 // stride) ** 2 for stride in (8, 16, 32)},
                         {8: 6400, 16: 1600, 32: 400})

    def test_curated_gaze_fp32_artifacts_have_verified_tensor_contract(self):
        catalog = model_tx.load_catalog(REAL_CATALOG_PATH)
        pack = next(
            item for item in catalog["packs"]
            if item["id"] == "gaze_correction_cam_flx_v0_1_1"
        )
        expected_artifacts = {
            "gaze_flx_left.onnx":
                "1a5129972dbc2ace9170d79ad4986e8c5130a79ac2180c01c826ef242d8e3d00",
            "gaze_flx_right.onnx":
                "736b7c6b91d1475d45083b00325d54d380ae1c51767e1f7ef69a3d14083cfcf1",
        }
        self.assertEqual(
            {item["name"]: item["sha256"] for item in pack["artifacts"] if item["name"].endswith(".onnx")},
            expected_artifacts,
        )

        model_metadata = next(
            item for item in pack["metadata"] if item["destination"] == "model.json"
        )
        metadata_path = REPO_ROOT / model_metadata["source"]
        self.assertEqual(model_tx.sha256_path(metadata_path), model_metadata["sha256"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["id"], pack["id"])
        self.assertEqual(
            {item["name"]: item["sha256"] for item in metadata["files"] if item["kind"] == "onnx"},
            expected_artifacts,
        )
        self.assertEqual(metadata["onnx"]["inputs"], [
            {
                "name": "eye_rgb:0", "layout": "nchw", "dtype": "float32",
                "width": 64, "height": 48, "channels": 3,
                "note": "Normalized eye crop (48x64).",
            },
            {
                "name": "anchors:0", "layout": "nchw", "dtype": "float32",
                "width": 64, "height": 48, "channels": 12,
                "note": "Six landmark-guidance points encoded as paired x/y offset maps.",
            },
            {
                "name": "angles:0", "layout": "nc", "dtype": "float32",
                "dim": 2,
                "note": "Yaw/pitch target delta. Exact convention matches the upstream FLX model conversion.",
            },
        ])
        self.assertEqual(metadata["onnx"]["outputs"], [{
            "name": "rgb:0", "kind": "rgb", "dtype": "float32",
            "width": 64, "height": 48, "channels": 3,
            "note": "Corrected RGB eye crop in NCHW layout and the 0..1 range.",
        }])

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
    def test_versioned_artifact_source_and_offline_cache_install_without_old_source(self):
        fixture = Fixture(self.root)
        pack = fixture.catalog["packs"][0]
        artifact = pack["artifacts"][0]
        original = fixture.source / artifact["source_paths"][0]
        relative = f"objects/{pack['id']}/revisions/{artifact['sha256']}/model.onnx"
        versioned = fixture.source / relative
        versioned.parent.mkdir(parents=True)
        shutil.copyfile(original, versioned)
        original.write_bytes(b"old revision must not be selected or changed")
        artifact["source_paths"] = [relative]
        first = fixture.installer().install("open_audio")
        self.assertEqual(first["state"], "succeeded")
        self.assertEqual((fixture.destination / pack["pack_path"] / "model.onnx").read_bytes(),
                         fixture.artifacts[pack["id"]])
        versioned.unlink()
        offline = model_tx.ModelTransactionInstaller(
            fixture.catalog, fixture.repo, self.root / "offline-models", fixture.cache,
            None, offline=True, retries=0,
        ).install("open_audio")
        self.assertEqual(offline["state"], "succeeded")
        self.assertEqual(original.read_bytes(), b"old revision must not be selected or changed")

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
